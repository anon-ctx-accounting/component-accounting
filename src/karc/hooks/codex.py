"""Codex lifecycle-hook adapter with reduced-payload privacy.

Known schemas are canonicalized conservatively.  Unknown events/tools remain
observation-only so schema drift cannot silently create false K-ARC evidence.
Every CLI invocation exits zero, including malformed input and database locks.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

from karc.hooks import canonical
from karc.ingest import pipeline
from karc.util import sha256_hex, utc_now_iso

RUNTIME = "codex"
ADAPTER_VERSION = "karc-codex-hook-0.1.0"
MAX_STDIN_BYTES = 1024 * 1024
_KNOWN_EVENTS = {"SessionStart", "PreToolUse", "PermissionRequest", "PostToolUse", "Stop"}


def _hash_json(value) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return sha256_hex(raw.encode("utf-8"))


def _git_root(cwd: str) -> Path:
    cur = Path(cwd).resolve()
    for candidate in (cur, *cur.parents):
        if (candidate / ".git").exists():
            return candidate
    return cur


def discover_agents(cwd: str | None) -> list[Path]:
    if not cwd:
        return []
    cur = Path(cwd).resolve()
    root = _git_root(str(cur))
    chain = []
    probe = cur
    while True:
        chain.append(probe)
        if probe == root or probe.parent == probe:
            break
        probe = probe.parent
    out = []
    for directory in reversed(chain):
        path = directory / "AGENTS.md"
        if path.is_file():
            out.append(path)
    return out


def _session(conn, payload: dict, scope_id: str | None = None) -> str | None:
    native = payload.get("session_id")
    if not isinstance(native, str) or not native:
        return None
    return canonical.ensure_session(
        conn, runtime=RUNTIME, native_session_id=native, scope_id=scope_id,
        model=payload.get("model") if isinstance(payload.get("model"), str) else None,
        transcript_path=(payload.get("transcript_path")
                         if isinstance(payload.get("transcript_path"), str) else None),
    )


def _obs_seed(payload: dict) -> str:
    return "|".join(str(x or "") for x in (
        "codex-hook", payload.get("session_id"), payload.get("turn_id"),
        payload.get("hook_event_name"), payload.get("tool_use_id"),
        payload.get("tool_name"), payload.get("source"),
    ))


def _observe(conn, payload: dict, reduced: dict) -> str:
    return canonical.observe(
        conn, source_channel="hook", runtime=RUNTIME,
        adapter_version=ADAPTER_VERSION,
        source_native_id=(payload.get("tool_use_id") or payload.get("turn_id")
                          or payload.get("session_id")),
        payload=reduced, observation_seed=_obs_seed(payload),
    )


def _artifact_matches(conn: sqlite3.Connection, value, cwd: str | None) -> list[tuple]:
    """Match registered artifact paths in a command/patch without storing it."""
    if not isinstance(value, str) or not value:
        return []
    squashed = re.sub(r"[\"'`\\]", "", value)
    rows = conn.execute(
        "SELECT a.scope_id, a.artifact_id, a.canonical_path, a.logical_name "
        "FROM artifacts a WHERE a.canonical_path IS NOT NULL"
    ).fetchall()
    matches = []
    for scope_id, artifact_id, absolute, logical in rows:
        relative = logical.replace(os.sep, "/") if isinstance(logical, str) else ""
        if absolute in squashed or (relative and relative in squashed):
            matches.append((scope_id, artifact_id, absolute))
    return matches


def _handle_session_start(conn, payload: dict, result: dict) -> None:
    paths = discover_agents(payload.get("cwd"))
    obs_id = _observe(conn, payload, {
        "hook_event": "SessionStart", "source": payload.get("source"),
        "model": payload.get("model"),
        "agents_path_hashes": [sha256_hex(str(p).encode()) for p in paths],
    })
    result["observation_id"] = obs_id
    for path in paths:
        resolved = canonical.resolve_scope_artifact(
            conn, str(path), payload.get("cwd"), occurred_at=utc_now_iso()
        )
        if resolved is None:
            continue
        sid = _session(conn, payload, resolved.scope_id)
        bucket = pipeline._time_bucket(utc_now_iso())
        eid = canonical.emit_event(
            conn,
            dedup_key=(f"codex:{payload.get('session_id')}:loaded:"
                       f"{resolved.artifact_id}:{bucket}"),
            event_type="loaded", occurred_at=utc_now_iso(),
            scope_id=resolved.scope_id, runtime=RUNTIME,
            artifact_id=resolved.artifact_id, source_channel="hook",
            confidence="inferred", session_id=sid,
            model=payload.get("model"), load_reason="codex_project_doc",
            load_class="preload",
        )
        canonical.link_observation(conn, obs_id, eid)
        result.setdefault("events", []).append(eid)


def _handle_post_tool(conn, payload: dict, result: dict) -> None:
    tool = str(payload.get("tool_name") or "")
    tool_id = payload.get("tool_use_id")
    input_value = payload.get("tool_input")
    response = payload.get("tool_response")
    call_id = canonical.extract_karc_call_id(response) if tool.startswith("mcp__karc__") else None
    matches = _artifact_matches(conn, input_value.get("command")
                                if isinstance(input_value, dict) else None,
                                payload.get("cwd"))
    reduced = {
        "hook_event": "PostToolUse", "tool": tool,
        "tool_input_hash": _hash_json(input_value),
        "tool_response_hash": _hash_json(response),
        "karc_call_id": call_id,
        "matched_artifact_ids": sorted({m[1] for m in matches}),
    }
    obs_id = _observe(conn, payload, reduced)
    result["observation_id"] = obs_id
    if call_id:
        result["linked_events"] = canonical.link_karc_call(conn, obs_id, call_id)
        return
    event_type = "read" if tool == "Bash" else "applied" if tool == "apply_patch" else None
    if event_type is None:
        return
    for scope_id, artifact_id, _path in matches:
        sid = _session(conn, payload, scope_id)
        turn = payload.get("turn_id")
        dedup = (f"codex:{payload.get('session_id')}:tu:{tool_id}:{artifact_id}"
                 if tool_id else
                 f"codex:{payload.get('session_id')}:{event_type}:{artifact_id}:"
                 f"{pipeline._time_bucket(utc_now_iso())}")
        eid = canonical.emit_event(
            conn, dedup_key=dedup, event_type=event_type,
            occurred_at=utc_now_iso(), scope_id=scope_id, runtime=RUNTIME,
            artifact_id=artifact_id, source_channel="hook", confidence="approx",
            session_id=sid, prompt_id=f"codex:{turn}" if turn else None,
            model=payload.get("model"),
        )
        canonical.link_observation(conn, obs_id, eid)
        result.setdefault("events", []).append(eid)


def record_hook_event(conn: sqlite3.Connection, db_path: str,
                      payload: dict) -> dict:
    event = payload.get("hook_event_name")
    result = {"recorded": False, "hook_event_name": event}
    if event == "SessionStart":
        _handle_session_start(conn, payload, result)
    elif event == "PostToolUse":
        _handle_post_tool(conn, payload, result)
    else:
        reduced = {
            "hook_event": event, "tool": payload.get("tool_name"),
            "tool_input_hash": _hash_json(payload.get("tool_input")),
            "known_schema": event in _KNOWN_EVENTS,
        }
        result["observation_id"] = _observe(conn, payload, reduced)
        _session(conn, payload)
    result["recorded"] = True
    return result


def main(argv: list[str] | None = None) -> int:
    import argparse
    from karc.db import connection

    parser = argparse.ArgumentParser(prog="karc hook --runtime codex")
    parser.add_argument("--db", default=str(connection.DEFAULT_DB_PATH))
    args = parser.parse_args(argv)
    payload = None
    try:
        raw = sys.stdin.buffer.read(MAX_STDIN_BYTES + 1)
        if len(raw) > MAX_STDIN_BYTES:
            return 0
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            return 0
        for attempt, wait_s in enumerate((0.0, 0.025, 0.075)):
            if wait_s:
                time.sleep(wait_s)
            conn = None
            try:
                conn = connection.connect(args.db)
                connection.migrate(conn, db_path=args.db)
                conn.execute("BEGIN IMMEDIATE")
                record_hook_event(conn, args.db, payload)
                conn.execute("COMMIT")
                conn.close()
                break
            except sqlite3.OperationalError as exc:
                if conn is not None:
                    try:
                        conn.execute("ROLLBACK")
                    except sqlite3.Error:
                        pass
                    conn.close()
                if "locked" not in str(exc).lower() or attempt == 2:
                    break
            except Exception:
                if conn is not None:
                    try:
                        conn.execute("ROLLBACK")
                    except sqlite3.Error:
                        pass
                    conn.close()
                break
    except Exception:
        return 0
    finally:
        if isinstance(payload, dict) and payload.get("hook_event_name") == "Stop":
            # Codex Stop hooks require JSON stdout when exiting zero.
            print("{}")
    return 0
