"""E2-2 per-run MCP index (A-9): populate the DB the karc server queries.

The ``karc`` MCP server's ``search``/``get`` tools read the SQLite
``artifacts``/``versions`` tables (data-model §4), not the filesystem. Nothing
in the fixture materialization populated that DB, so a fresh MCP arm run had an
empty index and both tools returned nothing. This module builds a **per-run**
index scoped to the run's workdir, containing only the managed artifacts that
were relocated under ``.karc/managed/`` for that run (1–3 docs/task).

It reuses ``mcp.server.ensure_scope`` to compute the scope row so the scope_id
matches exactly what the server will resolve on startup (``root_path`` is
UNIQUE, so the server finds this pre-inserted row rather than creating a second
one — otherwise ``search`` would query an empty scope). The DB lives inside the
workdir (``.karc/index.db``, a sibling of — not inside — ``.karc/managed/`` so
the C2 native-read deny does not touch it) and is discarded with the workdir
after the run (R-9: ids/paths/hashes/token counts only; content lives on disk
and is read on demand, never copied into the DB).
"""

from __future__ import annotations

import hashlib
import unicodedata
from pathlib import Path

from karc.db import connection
from karc.mcp.server import ensure_scope
from karc.util import new_ulid, utc_now_iso


def build_index(
    workdir: Path,
    manifest: dict,
    managed_ids: list[str],
    *,
    managed_root: str,
    db_path: Path | str,
) -> str:
    """Index ``managed_ids`` (already relocated under ``workdir/managed_root``)
    into a fresh DB at ``db_path``, scoped to ``workdir``. Returns the scope_id.

    ``size_tokens`` comes from the frozen manifest (identical to Stage-1 token
    accounting); ``content_hash``/``size_bytes`` from the relocated file on
    disk. ``lifecycle_state`` stays 'active' so ``search`` surfaces them."""
    workdir = Path(workdir)
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    arts = manifest["artifacts"]

    conn = connection.connect(str(db_path))
    connection.migrate(conn, db_path=str(db_path))
    conn.execute("BEGIN IMMEDIATE")
    try:
        scope_id = ensure_scope(conn, str(workdir))
        now = utc_now_iso()
        for aid in managed_ids:
            entry = arts.get(aid)
            if entry is None:
                continue
            rel = entry["path"]
            fpath = (workdir / managed_root / rel)
            if not fpath.is_file():
                continue
            raw = fpath.read_bytes()
            content_hash = hashlib.sha256(raw).hexdigest()
            # NFC + realpath — must equal ingest.pipeline.normalize_path output
            # so a path-based `get` resolves by canonical-path equality (P2).
            canonical = unicodedata.normalize("NFC", str(fpath.resolve()))
            criticality = "critical" if entry.get("critical") else "normal"
            version_id = new_ulid()
            # artifact_id = the fixture id (traceable; TEXT PK). If a prior run
            # in a shared DB already inserted it, INSERT OR IGNORE keeps it.
            conn.execute(
                "INSERT OR IGNORE INTO artifacts (artifact_id, scope_id, "
                "artifact_type, logical_name, canonical_path, criticality, "
                "lifecycle_state) VALUES (?, ?, 'document', ?, ?, ?, 'active')",
                (aid, scope_id, Path(rel).name, canonical, criticality),
            )
            conn.execute(
                "INSERT INTO versions (version_id, artifact_id, content_hash, "
                "size_bytes, size_tokens, token_estimator, observed_at, "
                "source_channel) VALUES (?, ?, ?, ?, ?, 'measured', ?, 'fixture')",
                (version_id, aid, content_hash, len(raw),
                 int(entry["size_tok"]), now),
            )
            conn.execute(
                "UPDATE artifacts SET current_version_id = ?, canonical_path = ? "
                "WHERE artifact_id = ?",
                (version_id, canonical, aid),
            )
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    return scope_id
