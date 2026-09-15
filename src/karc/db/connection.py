"""SQLite connection management + migration runner.

Requirements implemented here (data-model doc §2, §13.1, §16):

- PRAGMA foreign_keys=ON on every connection (SQLite is per-connection).
- WAL journal mode + busy_timeout (doc §11 write-contention guidance).
- Minimum SQLite version check: schema v1 uses a recursive CTE inside a
  trigger body (``trg_versions_no_cycle``). WITH clauses in trigger bodies
  are supported since SQLite 3.31.0 (2020-01-22); the data-model doc's own
  verification ran on 3.51.0 (doc §16). We enforce a floor of 3.31.0 AND run
  a functional probe (authoritative) that actually executes a recursive CTE
  inside a temp trigger, so an incapable build fails loudly at connect time
  instead of corrupting migration state.
- Migration runner: 순번제 forward-only. Before applying migrations to an
  existing DB file, a file backup is created (doc §13.1). ``schema_migrations``
  + ``PRAGMA user_version`` are kept in sync.
"""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

from karc.util import utc_now_iso

DEFAULT_DB_PATH = Path(".karc") / "karc.db"
MIGRATIONS_DIR = Path(__file__).parent / "migrations"

# Recursive CTE inside trigger bodies: available since SQLite 3.31.0.
# The functional probe below is the authoritative check; this floor exists
# to produce a clear error message on very old builds.
MIN_SQLITE_VERSION = (3, 31, 0)


class SQLiteVersionError(RuntimeError):
    """The linked SQLite library cannot run schema v1."""


def _check_version() -> None:
    if sqlite3.sqlite_version_info < MIN_SQLITE_VERSION:
        raise SQLiteVersionError(
            f"SQLite >= {'.'.join(map(str, MIN_SQLITE_VERSION))} required "
            f"(recursive CTE in trigger body, data-model §16); "
            f"found {sqlite3.sqlite_version}"
        )


def _probe_recursive_cte_trigger(conn: sqlite3.Connection) -> None:
    """Functional probe: execute a recursive CTE inside a temp trigger.

    Mirrors the construct used by ``trg_versions_no_cycle`` so unsupported
    SQLite builds fail here with a clear error, not mid-migration.
    """
    try:
        conn.executescript(
            """
            CREATE TEMP TABLE _karc_probe(x INTEGER);
            CREATE TEMP TRIGGER _karc_probe_trg BEFORE INSERT ON _karc_probe
            BEGIN
              SELECT RAISE(ABORT, 'probe')
              WHERE EXISTS (
                WITH RECURSIVE c(n) AS (SELECT 1 UNION SELECT n + 1 FROM c WHERE n < 3)
                SELECT 1 FROM c WHERE n = 999
              );
            END;
            INSERT INTO _karc_probe VALUES (1);
            DROP TRIGGER _karc_probe_trg;
            DROP TABLE _karc_probe;
            """
        )
    except sqlite3.Error as exc:  # pragma: no cover - only on old SQLite
        raise SQLiteVersionError(
            f"SQLite {sqlite3.sqlite_version} cannot run a recursive CTE "
            f"inside a trigger body (required by trg_versions_no_cycle): {exc}"
        ) from exc


def connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Open a connection with the required PRAGMAs and version checks."""
    _check_version()
    path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    # isolation_level=None: autocommit; transactions are managed explicitly
    # (BEGIN IMMEDIATE) by writers, per doc §11.
    conn = sqlite3.connect(str(path), isolation_level=None)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 1000")
    _probe_recursive_cte_trigger(conn)
    return conn


def _migration_files() -> list[tuple[int, str, Path]]:
    files = []
    for p in sorted(MIGRATIONS_DIR.glob("[0-9][0-9][0-9]_*.sql")):
        version = int(p.stem.split("_", 1)[0])
        files.append((version, p.stem, p))
    return files


def applied_versions(conn: sqlite3.Connection) -> set[int]:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_migrations'"
    ).fetchone()
    if row is None:
        return set()
    return {r[0] for r in conn.execute("SELECT version FROM schema_migrations")}


def migrate(
    conn: sqlite3.Connection, db_path: str | Path | None = None
) -> list[tuple[int, str]]:
    """Apply pending migrations. Returns [(version, name), ...] applied.

    Forward-only. If the DB file already exists and has content, a file
    backup ``<name>.pre-migrate-<ts>.bak`` is created first (doc §13.1 —
    downgrades are unsupported, the backup is the rollback path).
    """
    done = applied_versions(conn)
    pending = [(v, name, p) for v, name, p in _migration_files() if v not in done]
    if not pending:
        return []

    if db_path is not None:
        path = Path(db_path)
        if path.exists() and path.stat().st_size > 0 and done:
            ts = utc_now_iso().replace(":", "").replace(".", "")
            backup = path.with_suffix(f".pre-migrate-{ts}.bak")
            shutil.copy2(path, backup)

    applied: list[tuple[int, str]] = []
    for version, name, p in pending:
        sql = p.read_text(encoding="utf-8")
        conn.executescript(sql)
        conn.execute(
            "INSERT INTO schema_migrations (version, name) VALUES (?, ?)",
            (version, name),
        )
        conn.execute(f"PRAGMA user_version = {version}")
        applied.append((version, name))
    return applied
