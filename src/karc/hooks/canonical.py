"""Runtime-neutral helpers for hook/transcript canonicalization.

Adapters own source-schema interpretation and privacy reduction.  This module
owns the common database operations so multiple observation channels can
converge on the same immutable canonical event.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

from karc.ingest import pipeline
from karc.util import new_ulid, sha256_hex, utc_now_iso

EVENT_SCHEMA_VERSION = 1


@dataclass
class ResolvedArtifact:
    scope_id: str
    artifact_id: str
    resolved_path: str


def ensure_session(conn: sqlite3.Connection, *, runtime: str,
                   native_session_id: str, scope_id: str | None,
                   model: str | None = None,
                   transcript_path: str | None = None) -> str:
    sid = f"{runtime}:{native_session_id}"
    conn.execute(
        """INSERT OR IGNORE INTO sessions
             (session_id, runtime, native_session_id, scope_id, model, transcript_path)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (sid, runtime, native_session_id, scope_id, model, transcript_path),
    )
    conn.execute(
        """UPDATE sessions SET
             scope_id=COALESCE(scope_id, ?), model=COALESCE(model, ?),
             transcript_path=COALESCE(transcript_path, ?)
           WHERE session_id=?""",
        (scope_id, model, transcript_path, sid),
    )
    return sid


def observe(conn: sqlite3.Connection, *, source_channel: str, runtime: str,
            adapter_version: str, source_native_id: str | None,
            payload: dict, runtime_version: str | None = None,
            observation_seed: str | None = None) -> str:
    seed = observation_seed or (
        f"{source_channel}|{runtime}|{source_native_id or new_ulid()}|"
        f"{payload.get('hook_event') or payload.get('event_type') or ''}"
    )
    obs_id = sha256_hex(seed.encode("utf-8"))
    reduced = {**payload, "reduced": True}
    conn.execute(
        """INSERT OR IGNORE INTO ingest_observations
             (observation_id, source_channel, runtime, runtime_version,
              adapter_version, schema_version, source_native_id, observed_payload)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (obs_id, source_channel, runtime, runtime_version, adapter_version,
         EVENT_SCHEMA_VERSION, source_native_id,
         json.dumps(reduced, ensure_ascii=False, sort_keys=True)),
    )
    return obs_id


def link_observation(conn: sqlite3.Connection, obs_id: str, event_id: str) -> None:
    conn.execute(
        "UPDATE ingest_observations SET event_id=? WHERE observation_id=? "
        "AND event_id IS NULL", (event_id, obs_id)
    )


def resolve_scope_artifact(conn: sqlite3.Connection, raw_path: str | None,
                           cwd: str | None, *, occurred_at: str | None = None
                           ) -> ResolvedArtifact | None:
    if not raw_path:
        return None
    resolved, existed, inode = pipeline.normalize_path(raw_path, cwd)
    scope_id = pipeline._assign_scope(pipeline._scope_table(conn), resolved)
    if scope_id is None:
        return None
    stats = pipeline.IngestStats()
    artifact_id = pipeline._resolve_artifact(
        conn, resolved, raw_path, scope_id, occurred_at or utc_now_iso(),
        existed, inode, stats,
    )
    return ResolvedArtifact(scope_id, artifact_id, resolved)


def emit_event(conn: sqlite3.Connection, *, dedup_key: str, event_type: str,
               occurred_at: str, scope_id: str, runtime: str,
               artifact_id: str, source_channel: str,
               confidence: str = "direct", session_id: str | None = None,
               prompt_id: str | None = None, model: str | None = None,
               agent: str | None = None, load_reason: str | None = None,
               load_class: str | None = None,
               token_cost: int | None = None) -> str:
    row = conn.execute(
        "SELECT version_id FROM versions WHERE artifact_id=? AND invalidated_at IS NULL",
        (artifact_id,),
    ).fetchone()
    version_id = row[0] if row else None
    event_id = new_ulid()
    cur = conn.execute(
        """INSERT OR IGNORE INTO events
             (event_id, dedup_key, event_type, occurred_at, scope_id, runtime,
              agent, model, session_id, prompt_id, artifact_id, version_id,
              load_reason, load_class, token_cost, confidence, source_channel,
              schema_version)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (event_id, dedup_key, event_type, occurred_at, scope_id, runtime,
         agent, model, session_id, prompt_id, artifact_id, version_id,
         load_reason, load_class, token_cost, confidence, source_channel,
         EVENT_SCHEMA_VERSION),
    )
    if cur.rowcount:
        return event_id
    return conn.execute(
        "SELECT event_id FROM events WHERE dedup_key=?", (dedup_key,)
    ).fetchone()[0]


def extract_karc_call_id(value) -> str | None:
    """Find a K-ARC call id without returning or retaining response bodies."""
    if isinstance(value, dict):
        direct = value.get("karc_call_id")
        if isinstance(direct, str):
            return direct
        for nested in value.values():
            found = extract_karc_call_id(nested)
            if found:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = extract_karc_call_id(nested)
            if found:
                return found
    elif isinstance(value, str) and "karc_call_id" in value:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            # MCP get includes a human-readable header with karc_call_id.
            import re
            match = re.search(r"\bkarc_call_id=([0-9A-Za-z_-]+)", value)
            return match.group(1) if match else None
        return extract_karc_call_id(parsed)
    return None


def link_karc_call(conn: sqlite3.Connection, obs_id: str,
                   call_id: str) -> list[str]:
    rows = conn.execute(
        "SELECT event_id FROM events WHERE dedup_key=? OR dedup_key LIKE ?",
        (f"karc:{call_id}", f"karc:{call_id}:d:%"),
    ).fetchall()
    event_ids = [row[0] for row in rows]
    # One observation has one FK. Link to the get event or the first discovery;
    # all remaining convergence is still inspectable by the shared call id.
    if event_ids:
        link_observation(conn, obs_id, event_ids[0])
    return event_ids
