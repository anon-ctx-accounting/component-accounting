"""Ingestion pipeline: parsed observations → canonical rows (idempotent).

Implements the data-model doc's rules for M0:

- §4 ingestion: ``ingest_observations`` (append-only, deterministic
  observation_id) → ``events`` (canonical, ``INSERT OR IGNORE`` on
  ``dedup_key``). Re-ingesting the same data inserts 0 new rows.
- §4.2 dedup tiers used here: tier 2 ``claude-code:<session>:tu:<tool_use_id>``
  when the tool_use_id exists, else tier 3 time-bucket key (5s bucket).
- §5/§6 identity: logical identity keyed by the normalized path (N1–N7; for
  files that no longer exist we fall back to lexical normalization, N3).
  If the file currently exists a version row records the sha256 of the
  CURRENT on-disk content — this is a present-time hash taken at ingest
  time, NOT the historical content the event actually read (there is no way
  to recover past content from transcripts). ``versions.source_channel`` is
  set to ``'ingest_stat'`` to make that provenance explicit.
- N5 scope assignment: longest-prefix match against ``scopes.root_path``.
  Paths outside every scope are recorded as observations but NOT promoted to
  events (R-9 default: 미수집), and counted in stats.
- Privacy (R-9): observations store reduced payloads only — tool name, ids,
  paths, sizes, timestamps. Never prompt/response text or file contents.

Only ``Read`` tool calls become canonical events in M0 (``event_type='read'``,
E0-1 reference track). Glob/Grep/Edit/Write/... are stored as metadata-only
observations (``event_id`` stays NULL).
"""

from __future__ import annotations

import json
import os
import sqlite3
import unicodedata
from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import Path

from karc.ingest import claude_code as cc
from karc.util import new_ulid, sha256_file, sha256_hex, utc_now_iso

RUNTIME = "claude-code"
ADAPTER_VERSION = "karc-m0-0.1.0"
EVENT_SCHEMA_VERSION = 1
TIER3_BUCKET_SECONDS = 5

_DOC_EXT = {".md", ".markdown", ".rst", ".txt", ".adoc"}
_INSTRUCTION_NAMES = {"claude.md", "agents.md", "memory.md"}


@dataclass
class IngestStats:
    projects: int = 0
    files: int = 0
    lines: int = 0
    parse_errors: int = 0
    error_files: dict = field(default_factory=dict)
    scopes_created: int = 0
    sessions_upserted: int = 0
    tasks_inserted: int = 0
    user_turns: int = 0
    tool_calls_by_name: dict = field(default_factory=dict)
    read_calls: int = 0
    read_calls_no_path: int = 0
    out_of_scope_reads: int = 0
    events_inserted: int = 0
    events_deduped: int = 0
    observations_inserted: int = 0
    observations_existing: int = 0
    artifacts_created: int = 0
    versions_created: int = 0
    missing_files: int = 0  # referenced paths that no longer exist on disk

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


# ---------------------------------------------------------------------------
# Path normalization (data-model §6, N1–N7 subset used in M0)
# ---------------------------------------------------------------------------

def normalize_path(raw_path: str, cwd: str | None) -> tuple[str, bool, tuple | None]:
    """Apply N1–N4/N6: returns (resolved_path, existed, (st_dev, st_ino)|None).

    N2: ~/env expansion, absolutize against the observing line's cwd,
        lexical ``..``/``.`` removal.
    N3: realpath when the file exists NOW (batch parsing caveat: existence is
        checked at ingest time, not event time — doc §6 한계 명시); lexical
        fallback otherwise.
    N4: NFC unicode normalization.
    N6: st_dev/st_ino recorded as rename hints when stat() succeeds.
    """
    p = os.path.expandvars(os.path.expanduser(raw_path))
    if not os.path.isabs(p):
        base = cwd if cwd else os.getcwd()
        p = os.path.join(base, p)
    p = os.path.normpath(p)
    existed = os.path.isfile(p)
    inode: tuple | None = None
    if existed:
        p = os.path.realpath(p)
        try:
            st = os.stat(p)
            inode = (st.st_dev, st.st_ino)
        except OSError:
            inode = None
    return unicodedata.normalize("NFC", p), existed, inode


def classify_artifact_type(resolved_path: str) -> str:
    name = os.path.basename(resolved_path).lower()
    if name in _INSTRUCTION_NAMES:
        return "instruction"
    ext = os.path.splitext(name)[1]
    if ext in _DOC_EXT:
        return "document"
    return "other"


# ---------------------------------------------------------------------------
# Scope / session / task helpers
# ---------------------------------------------------------------------------

def _ensure_scope(conn: sqlite3.Connection, root_path: str, stats: IngestStats) -> str:
    root_path = os.path.normpath(root_path)
    if os.path.isdir(root_path):
        root_path = os.path.realpath(root_path)  # N3 parity with artifact paths
    root_path = unicodedata.normalize("NFC", root_path)
    row = conn.execute(
        "SELECT scope_id FROM scopes WHERE root_path = ?", (root_path,)
    ).fetchone()
    if row:
        return row[0]
    scope_id = new_ulid()
    conn.execute(
        "INSERT INTO scopes (scope_id, root_path, display_name) VALUES (?, ?, ?)",
        (scope_id, root_path, os.path.basename(root_path) or root_path),
    )
    stats.scopes_created += 1
    return scope_id


def _internal_session_id(native_session_id: str) -> str:
    return f"cc:{native_session_id}"


def _ensure_session(
    conn: sqlite3.Connection,
    info: cc.SessionInfo,
    scope_id: str | None,
    stats: IngestStats,
) -> str:
    sid = _internal_session_id(info.native_session_id)
    cur = conn.execute(
        """
        INSERT OR IGNORE INTO sessions
          (session_id, runtime, native_session_id, scope_id, transcript_path,
           started_at, ended_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            sid,
            RUNTIME,
            info.native_session_id,
            scope_id,
            info.transcript_path,
            info.started_at,
            info.ended_at,
        ),
    )
    if cur.rowcount:
        stats.sessions_upserted += 1
    return sid


def _task_id(session_id: str, ordinal: int) -> str:
    return f"{session_id}:t{ordinal:04d}"


def _ensure_tasks(
    conn: sqlite3.Connection,
    session_id: str,
    turns: list[cc.UserTurn],
    stats: IngestStats,
) -> list[tuple[str, str]]:
    """Insert tasks for a session's real user turns (deterministic ids).

    Returns [(started_at, task_id), ...] sorted by started_at, for timestamp
    attribution of reference events (boundary_method='user_turn', R-4).
    """
    turns_sorted = sorted(turns, key=lambda t: t.timestamp)
    out: list[tuple[str, str]] = []
    for ordinal, turn in enumerate(turns_sorted, start=1):
        tid = _task_id(session_id, ordinal)
        ended_at = (
            turns_sorted[ordinal].timestamp if ordinal < len(turns_sorted) else None
        )
        cur = conn.execute(
            """
            INSERT OR IGNORE INTO tasks
              (task_id, session_id, ordinal, boundary_method, native_task_ref,
               started_at, ended_at)
            VALUES (?, ?, ?, 'user_turn', ?, ?, ?)
            """,
            (tid, session_id, ordinal, turn.prompt_id, turn.timestamp, ended_at),
        )
        if cur.rowcount:
            stats.tasks_inserted += 1
        out.append((turn.timestamp, tid))
    return out


# ---------------------------------------------------------------------------
# Artifact registry (identity case 1 / case 7 of §5.1 + version tracking)
# ---------------------------------------------------------------------------

def _resolve_artifact(
    conn: sqlite3.Connection,
    resolved_path: str,
    raw_path: str,
    scope_id: str,
    occurred_at: str,
    existed: bool,
    inode: tuple | None,
    stats: IngestStats,
) -> str:
    row = conn.execute(
        """
        SELECT artifact_id FROM artifact_paths
        WHERE resolved_path = ? AND status IN ('active_canonical','active_alias')
        """,
        (resolved_path,),
    ).fetchone()
    if row:
        conn.execute(
            "UPDATE artifact_paths SET last_seen_at = ? WHERE resolved_path = ? "
            "AND status IN ('active_canonical','active_alias') AND last_seen_at < ?",
            (occurred_at, resolved_path, occurred_at),
        )
        artifact_id = row[0]
    else:
        artifact_id = new_ulid()
        scope_root = conn.execute(
            "SELECT root_path FROM scopes WHERE scope_id = ?", (scope_id,)
        ).fetchone()[0]
        logical_name = os.path.relpath(resolved_path, scope_root)
        conn.execute(
            """
            INSERT INTO artifacts
              (artifact_id, scope_id, artifact_type, logical_name, canonical_path)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                artifact_id,
                scope_id,
                classify_artifact_type(resolved_path),
                logical_name,
                resolved_path,
            ),
        )
        conn.execute(
            """
            INSERT INTO artifact_paths
              (artifact_id, resolved_path, raw_paths, status, st_dev, st_ino,
               first_seen_at, last_seen_at)
            VALUES (?, ?, ?, 'active_canonical', ?, ?, ?, ?)
            """,
            (
                artifact_id,
                resolved_path,
                json.dumps([raw_path], ensure_ascii=False),
                inode[0] if inode else None,
                inode[1] if inode else None,
                occurred_at,
                occurred_at,
            ),
        )
        stats.artifacts_created += 1

    if existed:
        _ensure_current_version(conn, artifact_id, resolved_path, stats)
    else:
        stats.missing_files += 1
    return artifact_id


def _ensure_current_version(
    conn: sqlite3.Connection, artifact_id: str, resolved_path: str, stats: IngestStats
) -> None:
    """Record a version for the CURRENT on-disk content (present-time hash).

    NOTE: this hash describes the file as it exists at ingest time — it is
    NOT the content the historical event actually read. source_channel
    'ingest_stat' marks this provenance (doc §5.1 case 1 + M0 제약).
    """
    try:
        content_hash = sha256_file(resolved_path)
        size_bytes = os.path.getsize(resolved_path)
    except OSError:
        return
    live = conn.execute(
        "SELECT version_id, content_hash FROM versions "
        "WHERE artifact_id = ? AND invalidated_at IS NULL",
        (artifact_id,),
    ).fetchone()
    if live and live[1] == content_hash:
        return
    now = utc_now_iso()
    new_version_id = new_ulid()
    if live:
        # uq_versions_live: invalidate the old live version BEFORE inserting
        # the new one, then link the supersede chain (reason='new_version').
        conn.execute(
            "UPDATE versions SET invalidated_at = ? WHERE version_id = ?",
            (now, live[0]),
        )
    conn.execute(
        """
        INSERT INTO versions
          (version_id, artifact_id, content_hash, size_bytes, size_tokens,
           token_estimator, observed_at, source_channel)
        VALUES (?, ?, ?, ?, ?, 'bytes_div4', ?, 'ingest_stat')
        """,
        (
            new_version_id,
            artifact_id,
            content_hash,
            size_bytes,
            size_bytes // 4,
            now,
        ),
    )
    if live:
        conn.execute(
            "UPDATE versions SET superseded_by_version_id = ?, "
            "supersede_reason = 'new_version' WHERE version_id = ?",
            (new_version_id, live[0]),
        )
    conn.execute(
        "UPDATE artifacts SET current_version_id = ?, updated_at = ? "
        "WHERE artifact_id = ?",
        (new_version_id, now, artifact_id),
    )
    stats.versions_created += 1


# ---------------------------------------------------------------------------
# Dedup keys (§4.2) and observation ids
# ---------------------------------------------------------------------------

def _dedup_key(
    native_session_id: str,
    tool_use_id: str | None,
    event_type: str,
    artifact_id: str,
    occurred_at: str,
) -> str:
    if tool_use_id:
        return f"{RUNTIME}:{native_session_id}:tu:{tool_use_id}"  # tier 2
    bucket = _time_bucket(occurred_at)
    return f"{RUNTIME}:{native_session_id}:{event_type}:{artifact_id}:{bucket}"  # tier 3


def _time_bucket(iso_ts: str) -> str:
    """5-second bucket of an ISO8601 'YYYY-MM-DDTHH:MM:SS.mmmZ' timestamp."""
    date_part, time_part = iso_ts.split("T", 1)
    hh, mm, rest = time_part.split(":", 2)
    seconds = int(float(rest.rstrip("Z")))
    bucket = seconds - (seconds % TIER3_BUCKET_SECONDS)
    return f"{date_part}T{hh}:{mm}:{bucket:02d}"


def _observation_id(call: cc.ToolCall) -> str:
    if call.tool_use_id:
        aux = call.tool_use_id
    else:
        aux = f"{os.path.basename(call.source_file)}:{call.source_line}"
    return sha256_hex(
        f"transcript|{call.native_session_id}|{aux}".encode("utf-8")
    )


def _observation_payload(call: cc.ToolCall) -> str:
    """Reduced payload (R-9): ids/paths/metadata only, no content, no prompts."""
    return json.dumps(
        {
            "tool": call.tool_name,
            "tool_use_id": call.tool_use_id,
            "file_path": call.file_path,
            "input_meta": call.input_meta,
            "timestamp": call.timestamp,
            "cwd": call.cwd,
            "agent_id": call.agent_id,
            "is_sidechain": call.is_sidechain,
            "source_line": call.source_line,
            "reduced": True,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def ingest_projects_root(
    conn: sqlite3.Connection, projects_root: str | Path
) -> IngestStats:
    """Ingest every encoded project dir under a Claude Code projects root."""
    projects_root = Path(projects_root)
    stats = IngestStats()
    project_dirs = sorted(p for p in projects_root.iterdir() if p.is_dir())
    # Pass 1: parse everything and create all scopes first so that N5
    # longest-prefix assignment sees the full scope set.
    parsed: list[tuple[Path, cc.ParseResult]] = []
    for pdir in project_dirs:
        result = cc.parse_project_dir(pdir)
        if not result.sessions and not result.tool_calls:
            continue
        parsed.append((pdir, result))
    conn.execute("BEGIN IMMEDIATE")
    try:
        for pdir, result in parsed:
            stats.projects += 1
            for info in result.sessions:
                if info.scope_cwd:
                    _ensure_scope(conn, info.scope_cwd, stats)
        for pdir, result in parsed:
            _ingest_parse_result(conn, result, stats)
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    return stats


def _scope_table(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    """[(root_path, scope_id)] sorted by root_path length desc (N5)."""
    rows = conn.execute("SELECT root_path, scope_id FROM scopes").fetchall()
    return sorted(rows, key=lambda r: len(r[0]), reverse=True)


def _assign_scope(
    scope_rows: list[tuple[str, str]], resolved_path: str
) -> str | None:
    for root, scope_id in scope_rows:
        if resolved_path == root or resolved_path.startswith(root.rstrip("/") + "/"):
            return scope_id
    return None


def _ingest_parse_result(
    conn: sqlite3.Connection, result: cc.ParseResult, stats: IngestStats
) -> None:
    stats.files += result.stats.files
    stats.lines += result.stats.lines
    stats.parse_errors += result.stats.parse_errors
    stats.error_files.update(result.stats.error_files)
    stats.user_turns += result.stats.real_user_turns
    for name, n in result.stats.tool_calls_by_name.items():
        stats.tool_calls_by_name[name] = stats.tool_calls_by_name.get(name, 0) + n

    scope_rows = _scope_table(conn)

    # Sessions + tasks (task boundaries from main-file real user turns, R-4)
    session_scope: dict[str, str | None] = {}
    task_index: dict[str, tuple[list[str], list[str]]] = {}
    turns_by_session: dict[str, list[cc.UserTurn]] = {}
    for turn in result.user_turns:
        turns_by_session.setdefault(turn.native_session_id, []).append(turn)
    for info in result.sessions:
        scope_id = None
        if info.scope_cwd:
            norm_root = unicodedata.normalize("NFC", os.path.normpath(info.scope_cwd))
            scope_id = _assign_scope(scope_rows, norm_root)
        sid = _ensure_session(conn, info, scope_id, stats)
        session_scope[info.native_session_id] = scope_id
        tasks = _ensure_tasks(
            conn, sid, turns_by_session.get(info.native_session_id, []), stats
        )
        task_index[info.native_session_id] = (
            [t[0] for t in tasks],
            [t[1] for t in tasks],
        )

    # Tool calls → observations (+ canonical read events)
    for call in result.tool_calls:
        if call.native_session_id not in session_scope:
            # Orphan subagent with no session info at all: synthesize session.
            info = cc.SessionInfo(
                native_session_id=call.native_session_id,
                scope_cwd=call.cwd,
                transcript_path=call.source_file,
            )
            scope_id = None
            if call.cwd:
                scope_id = _assign_scope(
                    scope_rows,
                    unicodedata.normalize("NFC", os.path.normpath(call.cwd)),
                )
            _ensure_session(conn, info, scope_id, stats)
            session_scope[call.native_session_id] = scope_id
            task_index[call.native_session_id] = ([], [])
        _ingest_tool_call(conn, call, scope_rows, task_index, stats)


def _insert_observation(
    conn: sqlite3.Connection, call: cc.ToolCall, stats: IngestStats
) -> tuple[str, bool]:
    obs_id = _observation_id(call)
    cur = conn.execute(
        """
        INSERT OR IGNORE INTO ingest_observations
          (observation_id, source_channel, runtime, adapter_version,
           schema_version, source_native_id, observed_payload)
        VALUES (?, 'transcript', ?, ?, ?, ?, ?)
        """,
        (
            obs_id,
            RUNTIME,
            ADAPTER_VERSION,
            EVENT_SCHEMA_VERSION,
            call.tool_use_id,
            _observation_payload(call),
        ),
    )
    if cur.rowcount:
        stats.observations_inserted += 1
        return obs_id, True
    stats.observations_existing += 1
    return obs_id, False


def _ingest_tool_call(
    conn: sqlite3.Connection,
    call: cc.ToolCall,
    scope_rows: list[tuple[str, str]],
    task_index: dict[str, tuple[list[str], list[str]]],
    stats: IngestStats,
) -> None:
    obs_id, _ = _insert_observation(conn, call, stats)

    if call.tool_name not in cc.REFERENCE_TOOLS:
        return  # metadata-only observation (Glob/Grep/Edit/Write/...)

    stats.read_calls += 1
    if not call.file_path:
        stats.read_calls_no_path += 1
        return

    resolved, existed, inode = normalize_path(call.file_path, call.cwd)
    scope_id = _assign_scope(scope_rows, resolved)
    if scope_id is None:
        stats.out_of_scope_reads += 1
        return  # N5: 기본 미수집 — observation kept, no event

    artifact_id = _resolve_artifact(
        conn, resolved, call.file_path, scope_id, call.timestamp, existed, inode, stats
    )
    version_id = None
    row = conn.execute(
        "SELECT version_id FROM versions WHERE artifact_id = ? AND invalidated_at IS NULL",
        (artifact_id,),
    ).fetchone()
    if row:
        version_id = row[0]

    session_id = _internal_session_id(call.native_session_id)
    starts, tids = task_index.get(call.native_session_id, ([], []))
    task_id = None
    if starts:
        i = bisect_right(starts, call.timestamp)
        if i > 0:
            task_id = tids[i - 1]

    dedup = _dedup_key(
        call.native_session_id, call.tool_use_id, "read", artifact_id, call.timestamp
    )
    event_id = new_ulid()
    cur = conn.execute(
        """
        INSERT OR IGNORE INTO events
          (event_id, dedup_key, event_type, occurred_at, scope_id, runtime,
           agent, session_id, task_id, artifact_id, version_id,
           confidence, source_channel, schema_version)
        VALUES (?, ?, 'read', ?, ?, ?, ?, ?, ?, ?, ?, 'direct', 'transcript', ?)
        """,
        (
            event_id,
            dedup,
            call.timestamp,
            scope_id,
            RUNTIME,
            call.agent_id,
            session_id,
            task_id,
            artifact_id,
            version_id,
            EVENT_SCHEMA_VERSION,
        ),
    )
    if cur.rowcount:
        stats.events_inserted += 1
        final_event_id = event_id
    else:
        stats.events_deduped += 1
        final_event_id = conn.execute(
            "SELECT event_id FROM events WHERE dedup_key = ?", (dedup,)
        ).fetchone()[0]
    # Link the observation to its canonical event (the only UPDATE the
    # observations guard trigger permits).
    conn.execute(
        "UPDATE ingest_observations SET event_id = ? WHERE observation_id = ? "
        "AND event_id IS NULL",
        (final_event_id, obs_id),
    )
