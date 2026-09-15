"""Parse a Claude Code hook payload → observations + canonical events.

Handled events (telemetry §1.1 measured payloads):

- ``PostToolUse`` with ``tool_name == 'Read'`` → ``read`` event, dedup tier-2
  ``claude-code:<session>:tu:<tool_use_id>`` (converges with transcript batch
  ingestion, which uses the same key).
- ``PostToolUse`` with ``tool_name`` matching ``mcp__karc__.*`` → the MCP call
  result carries a ``karc_call_id``; the adapter records a hook observation
  keyed on that id and links it to the canonical event the MCP server already
  wrote (tier-1 convergence, §4.2). For ``mcp__karc__search`` it also links to
  the per-candidate discovered events.
- ``InstructionsLoaded`` → ``loaded`` event. ``load_reason`` is the runtime raw
  value preserved verbatim; ``load_class`` is the D2 normalization (all
  instruction loads are ``preload`` — funnel-only, never a hit, §2.2). R-15
  requires both columns on ``loaded``.

All privacy rules (R-9) hold: reduced payloads only — ids/paths/metadata, never
content or prompts.
"""

from __future__ import annotations

import json
import os
import sqlite3
import unicodedata

from karc.ingest import pipeline
from karc.util import new_ulid, sha256_hex, utc_now_iso

RUNTIME = "claude-code"
ADAPTER_VERSION = "karc-hook-0.1.0"
EVENT_SCHEMA_VERSION = 1

# D2 normalization for InstructionsLoaded raw load_reason values. Instruction
# loads are runtime-driven context injection → preload (funnel-only, §2.2).
_LOAD_CLASS = {
    "session_start": "preload",
    "nested_traversal": "preload",
    "path_glob_match": "preload",
    "include": "preload",
    "compact": "preload",
}


class HookResult(dict):
    """What the adapter did (for tests/diagnostics). Never surfaced to agent."""


def _ensure_session(
    conn: sqlite3.Connection, native_session_id: str, scope_id: str | None
) -> str:
    sid = f"cc:{native_session_id}"
    conn.execute(
        "INSERT OR IGNORE INTO sessions (session_id, runtime, native_session_id, scope_id) "
        "VALUES (?, ?, ?, ?)",
        (sid, RUNTIME, native_session_id, scope_id),
    )
    return sid


def _observe(
    conn: sqlite3.Connection,
    obs_id: str,
    source_native_id: str | None,
    payload: dict,
) -> str:
    conn.execute(
        """INSERT OR IGNORE INTO ingest_observations
             (observation_id, source_channel, runtime, adapter_version,
              schema_version, source_native_id, observed_payload)
           VALUES (?, 'hook', ?, ?, ?, ?, ?)""",
        (
            obs_id,
            RUNTIME,
            ADAPTER_VERSION,
            EVENT_SCHEMA_VERSION,
            source_native_id,
            json.dumps({**payload, "reduced": True}, ensure_ascii=False, sort_keys=True),
        ),
    )
    return obs_id


def _link_obs(conn: sqlite3.Connection, obs_id: str, event_id: str) -> None:
    conn.execute(
        "UPDATE ingest_observations SET event_id = ? WHERE observation_id = ? "
        "AND event_id IS NULL",
        (event_id, obs_id),
    )


def _resolve_scope_artifact(
    conn: sqlite3.Connection, raw_path: str | None, cwd: str | None
):
    if not raw_path:
        return None, None
    resolved, existed, inode = pipeline.normalize_path(raw_path, cwd)
    scope_rows = pipeline._scope_table(conn)
    scope_id = pipeline._assign_scope(scope_rows, resolved)
    if scope_id is None:
        return None, None
    stats = pipeline.IngestStats()
    artifact_id = pipeline._resolve_artifact(
        conn, resolved, raw_path, scope_id, utc_now_iso(), existed, inode, stats
    )
    return scope_id, artifact_id


def _emit_event(
    conn: sqlite3.Connection,
    dedup_key: str,
    event_type: str,
    occurred_at: str,
    scope_id: str,
    artifact_id: str,
    session_id: str | None,
    *,
    load_reason: str | None = None,
    load_class: str | None = None,
    token_cost: int | None = None,
) -> str:
    version_row = conn.execute(
        "SELECT version_id FROM versions WHERE artifact_id = ? AND invalidated_at IS NULL",
        (artifact_id,),
    ).fetchone()
    version_id = version_row[0] if version_row else None
    event_id = new_ulid()
    cur = conn.execute(
        """INSERT OR IGNORE INTO events
             (event_id, dedup_key, event_type, occurred_at, scope_id, runtime,
              session_id, artifact_id, version_id, load_reason, load_class,
              token_cost, confidence, source_channel, schema_version)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'direct', 'hook', ?)""",
        (
            event_id,
            dedup_key,
            event_type,
            occurred_at,
            scope_id,
            RUNTIME,
            session_id,
            artifact_id,
            version_id,
            load_reason,
            load_class,
            token_cost,
            EVENT_SCHEMA_VERSION,
        ),
    )
    if cur.rowcount:
        return event_id
    return conn.execute(
        "SELECT event_id FROM events WHERE dedup_key = ?", (dedup_key,)
    ).fetchone()[0]


def _handle_mcp_karc(
    conn: sqlite3.Connection, payload: dict, session_id: str, result: HookResult
) -> None:
    """PostToolUse for an mcp__karc__* tool. The MCP server already wrote the
    canonical event(s); we record a hook observation keyed on karc_call_id and
    link it to those events (tier-1 convergence)."""
    tool_name = payload.get("tool_name", "")
    resp = payload.get("tool_response")
    call_id = _extract_call_id(resp)
    obs_id = sha256_hex(f"hook|{session_id}|mcpkarc|{call_id or new_ulid()}".encode())
    _observe(conn, obs_id, call_id, {"tool": tool_name, "karc_call_id": call_id,
                                     "hook_event": "PostToolUse"})
    result["karc_call_id"] = call_id
    if not call_id:
        return
    # link to whatever canonical events exist for this call (get: karc:<id>;
    # search: karc:<id>:d:*). INSERT is the MCP server's job; we just converge.
    rows = conn.execute(
        "SELECT event_id FROM events WHERE dedup_key = ? OR dedup_key LIKE ?",
        (f"karc:{call_id}", f"karc:{call_id}:d:%"),
    ).fetchall()
    for (eid,) in rows:
        _link_obs(conn, obs_id, eid)
        result.setdefault("linked_events", []).append(eid)


def _extract_call_id(tool_response) -> str | None:
    """Pull karc_call_id out of a PostToolUse tool_response (the MCP result)."""
    if isinstance(tool_response, dict):
        sc = tool_response.get("structuredContent")
        if isinstance(sc, dict) and sc.get("karc_call_id"):
            return sc["karc_call_id"]
        if tool_response.get("karc_call_id"):
            return tool_response["karc_call_id"]
        content = tool_response.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    try:
                        obj = json.loads(block.get("text", ""))
                    except (json.JSONDecodeError, TypeError):
                        continue
                    if isinstance(obj, dict) and obj.get("karc_call_id"):
                        return obj["karc_call_id"]
    return None


def record_hook_event(
    conn: sqlite3.Connection, db_path: str, payload: dict
) -> HookResult:
    """Record one hook payload. Best-effort: any failure is caught by the
    caller (``main``); this function may raise and the CLI still exits 0."""
    result = HookResult(recorded=False)
    event_name = payload.get("hook_event_name")
    session = payload.get("session_id")
    cwd = payload.get("cwd")
    occurred_at = payload.get("timestamp") or utc_now_iso()
    result["hook_event_name"] = event_name

    if event_name == "PostToolUse":
        tool_name = payload.get("tool_name", "")
        result["tool_name"] = tool_name
        if tool_name.startswith("mcp__karc__"):
            sid = _ensure_session(conn, session, None) if session else None
            _handle_mcp_karc(conn, payload, sid or session or "", result)
            result["recorded"] = True
            return result
        if tool_name != "Read":
            # metadata-only observation for other tools
            tuid = payload.get("tool_use_id")
            obs_id = sha256_hex(f"hook|{session}|{tuid or new_ulid()}".encode())
            _observe(conn, obs_id, tuid, {"tool": tool_name, "hook_event": "PostToolUse"})
            result["recorded"] = True
            return result
        # Read → reference event
        file_path = (payload.get("tool_input") or {}).get("file_path")
        tuid = payload.get("tool_use_id")
        obs_id = sha256_hex(f"hook|{session}|{tuid or file_path}".encode())
        _observe(conn, obs_id, tuid, {"tool": "Read", "file_path": file_path,
                                      "hook_event": "PostToolUse"})
        scope_id, artifact_id = _resolve_scope_artifact(conn, file_path, cwd)
        if scope_id is None:
            result["recorded"] = True
            result["out_of_scope"] = True
            return result
        sid = _ensure_session(conn, session, scope_id) if session else None
        dedup = (
            f"{RUNTIME}:{session}:tu:{tuid}"
            if tuid
            else f"{RUNTIME}:{session}:read:{artifact_id}:{pipeline._time_bucket(occurred_at)}"
        )
        eid = _emit_event(conn, dedup, "read", occurred_at, scope_id, artifact_id, sid)
        _link_obs(conn, obs_id, eid)
        result.update(recorded=True, event_id=eid, artifact_id=artifact_id)
        return result

    if event_name == "InstructionsLoaded":
        file_path = payload.get("file_path")
        load_reason = payload.get("load_reason") or "session_start"
        load_class = _LOAD_CLASS.get(load_reason, "preload")
        obs_id = sha256_hex(
            f"hook|{session}|IL|{file_path}|{load_reason}".encode()
        )
        _observe(conn, obs_id, None,
                 {"file_path": file_path, "load_reason": load_reason,
                  "memory_type": payload.get("memory_type"),
                  "hook_event": "InstructionsLoaded"})
        scope_id, artifact_id = _resolve_scope_artifact(conn, file_path, cwd)
        if scope_id is None:
            result.update(recorded=True, out_of_scope=True)
            return result
        sid = _ensure_session(conn, session, scope_id) if session else None
        dedup = (
            f"{RUNTIME}:{session}:loaded:{artifact_id}:"
            f"{pipeline._time_bucket(occurred_at)}"
        )
        eid = _emit_event(
            conn, dedup, "loaded", occurred_at, scope_id, artifact_id, sid,
            load_reason=load_reason, load_class=load_class,
        )
        _link_obs(conn, obs_id, eid)
        result.update(recorded=True, event_id=eid, artifact_id=artifact_id,
                      load_class=load_class)
        return result

    # unhandled hook event: record a bare observation for auditability
    obs_id = sha256_hex(f"hook|{session}|{event_name}|{new_ulid()}".encode())
    _observe(conn, obs_id, None, {"hook_event": event_name})
    result["recorded"] = True
    return result


def main(argv: list[str] | None = None) -> int:
    """CLI entry (`karc hook`). Reads one JSON payload on stdin. ALWAYS exits 0
    — telemetry must never block or influence the agent."""
    import argparse
    import sys

    from karc.db import connection

    parser = argparse.ArgumentParser(prog="karc hook")
    parser.add_argument("--db", default=str(connection.DEFAULT_DB_PATH))
    args = parser.parse_args(argv)
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw)
        conn = connection.connect(args.db)
        connection.migrate(conn, db_path=args.db)
        conn.execute("BEGIN IMMEDIATE")
        try:
            record_hook_event(conn, args.db, payload)
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        conn.close()
    except Exception:
        # Swallow everything: best-effort telemetry never breaks the agent.
        return 0
    return 0
