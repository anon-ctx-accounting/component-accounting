"""Version-gated Codex JSONL fallback ingestion.

The stable ``codex exec --json`` event family is accepted as a fallback when
hook coverage is missing.  Transcript text, prompts, commands, and tool
responses are reduced to lengths/hashes before they reach SQLite.  Unknown
event or item schemas remain observation-only and lower reported coverage;
they are never guessed into canonical events.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from karc.hooks import canonical
from karc.ingest import pipeline
from karc.util import sha256_hex, utc_now_iso

RUNTIME = "codex"
ADAPTER_VERSION = "karc-codex-jsonl-0.1.0"
SCHEMA_FAMILY = "codex-exec-jsonl-0.144.x-v1"
KNOWN_EVENT_TYPES = {
    "thread.started", "turn.started", "item.started", "item.updated",
    "item.completed", "turn.completed", "turn.failed", "error",
}
KNOWN_ITEM_TYPES = {
    "agent_message", "reasoning", "command_execution", "file_change",
    "mcp_tool_call", "web_search", "plan_update", "todo_list", "error",
}


@dataclass
class ParseStats:
    files: int = 0
    lines: int = 0
    malformed: int = 0
    unknown_events: int = 0
    unknown_items: int = 0
    observations: int = 0
    canonical_events: int = 0
    errors: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def _hash(value) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return sha256_hex(raw.encode("utf-8"))


def _command(item: dict) -> str:
    value = item.get("command")
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(str(v) for v in value)
    return ""


def _matches(conn: sqlite3.Connection, value: str) -> list[tuple[str, str]]:
    if not value:
        return []
    squashed = re.sub(r"[\"'`\\]", "", value)
    rows = conn.execute(
        "SELECT scope_id, artifact_id, canonical_path, logical_name FROM artifacts "
        "WHERE canonical_path IS NOT NULL"
    ).fetchall()
    return [(scope, aid) for scope, aid, absolute, logical in rows
            if absolute in squashed or (logical and logical.replace(os.sep, "/") in squashed)]


def _reduced_item(item: dict) -> dict:
    kind = item.get("type")
    reduced = {
        "event_type": "item.completed", "item_id": item.get("id"),
        "item_type": kind, "status": item.get("status"),
    }
    if kind == "agent_message":
        text = item.get("text") if isinstance(item.get("text"), str) else ""
        reduced.update(text_len=len(text), text_hash=sha256_hex(text.encode()))
    elif kind == "command_execution":
        command = _command(item)
        reduced.update(command_len=len(command), command_hash=sha256_hex(command.encode()))
    elif kind == "file_change":
        reduced["changes_hash"] = _hash(item.get("changes") or item.get("files"))
    elif kind == "mcp_tool_call":
        reduced.update(
            server=item.get("server") or item.get("server_name"),
            tool=item.get("tool") or item.get("tool_name"),
            arguments_hash=_hash(item.get("arguments") or item.get("input")),
            result_hash=_hash(item.get("result") or item.get("output")),
            karc_call_id=canonical.extract_karc_call_id(
                item.get("result") or item.get("output")
            ),
        )
    elif kind == "error":
        message = item.get("message") if isinstance(item.get("message"), str) else ""
        reduced.update(message_len=len(message), message_hash=sha256_hex(message.encode()))
    return reduced


def ingest_file(conn: sqlite3.Connection | None, path: str | Path, *,
                runtime_version: str, schema_family: str = SCHEMA_FAMILY,
                dry_run: bool = False, stats: ParseStats | None = None) -> ParseStats:
    stats = stats or ParseStats()
    stats.files += 1
    if schema_family != SCHEMA_FAMILY or not runtime_version.startswith("codex-cli 0.144"):
        stats.errors["unsupported_schema_family"] = (
            stats.errors.get("unsupported_schema_family", 0) + 1
        )
        return stats
    thread_id = None
    item_state: dict[str, dict] = {}
    with Path(path).open(encoding="utf-8", errors="replace") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            stats.lines += 1
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                stats.malformed += 1
                continue
            if not isinstance(event, dict):
                stats.malformed += 1
                continue
            etype = event.get("type")
            if etype not in KNOWN_EVENT_TYPES:
                stats.unknown_events += 1
                continue
            if etype == "thread.started":
                thread_id = event.get("thread_id") or thread_id
                continue
            if etype not in ("item.started", "item.updated", "item.completed"):
                continue
            item = event.get("item")
            if not isinstance(item, dict):
                stats.malformed += 1
                continue
            iid = str(item.get("id") or f"line-{line_no}")
            merged = {**item_state.get(iid, {}), **item}
            item_state[iid] = merged
            if etype != "item.completed":
                continue
            kind = merged.get("type")
            known = kind in KNOWN_ITEM_TYPES
            if not known:
                stats.unknown_items += 1
            reduced = _reduced_item(merged)
            reduced.update(known_schema=known, schema_family=schema_family)
            stats.observations += 1
            if dry_run or conn is None:
                continue
            obs_id = canonical.observe(
                conn, source_channel="transcript", runtime=RUNTIME,
                runtime_version=runtime_version, adapter_version=ADAPTER_VERSION,
                source_native_id=iid, payload=reduced,
                observation_seed=f"codex-jsonl|{thread_id}|{iid}|{path}",
            )
            if not known or not thread_id:
                continue
            if kind == "mcp_tool_call" and reduced.get("karc_call_id"):
                canonical.link_karc_call(conn, obs_id, reduced["karc_call_id"])
                continue
            event_type = "read" if kind == "command_execution" else (
                "applied" if kind == "file_change" else None
            )
            if event_type is None:
                continue
            match_text = (_command(merged) if kind == "command_execution" else
                          json.dumps(merged.get("changes") or merged.get("files") or [],
                                     ensure_ascii=False, sort_keys=True))
            for scope_id, artifact_id in _matches(conn, match_text):
                sid = canonical.ensure_session(
                    conn, runtime=RUNTIME, native_session_id=thread_id,
                    scope_id=scope_id,
                )
                eid = canonical.emit_event(
                    conn,
                    dedup_key=f"codex:{thread_id}:tu:{iid}:{artifact_id}",
                    event_type=event_type, occurred_at=utc_now_iso(),
                    scope_id=scope_id, runtime=RUNTIME, artifact_id=artifact_id,
                    source_channel="transcript", confidence="approx", session_id=sid,
                )
                canonical.link_observation(conn, obs_id, eid)
                stats.canonical_events += 1
    return stats


def _since_timestamp(value: str | None) -> float | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def ingest_sessions_dir(conn: sqlite3.Connection | None, sessions_dir: str | Path, *,
                        runtime_version: str, schema_family: str = SCHEMA_FAMILY,
                        since: str | None = None, dry_run: bool = False) -> ParseStats:
    stats = ParseStats()
    cutoff = _since_timestamp(since)
    for path in sorted(Path(sessions_dir).rglob("*.jsonl")):
        if cutoff is not None and path.stat().st_mtime < cutoff:
            continue
        ingest_file(conn, path, runtime_version=runtime_version,
                    schema_family=schema_family, dry_run=dry_run, stats=stats)
    return stats
