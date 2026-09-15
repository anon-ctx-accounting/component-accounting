"""Content-addressed snapshot store (data-model §9).

Layout (§9.1), rooted at the DB's own directory so MCP server / hook adapter /
CLI all share one store::

    <db_dir>/
      karc.db
      objects/
        sha256/
          ab/abcdef01…ef.zz      # compressed blobs too large for a DB BLOB

Design points implemented here:

- Identity is always ``sha256(원문, 비압축)`` (§9.1) — compression-independent,
  directly comparable to ``versions.content_hash`` / ``fs_snapshot_files``.
- Compression is **zlib**, a documented deviation from the schema's zstd
  (zstd is absent from the Python 3.12 stdlib; stdlib-only R-3 takes
  precedence). If zlib does not shrink the payload it is stored raw with
  ``compression='none'`` so tiny/incompressible blobs pay no overhead.
- Blob placement (§9.1): stored size ≤ 512 KiB → SQLite BLOB (``storage='db'``);
  larger → external file under ``objects/``.
- Dedup is structural via the ``content_hash`` PK + ``INSERT OR IGNORE``.
- No hard delete (§10): ``trg_no_del_cas`` guards the row; external files are
  never removed here (fsck reports, never deletes — §9.4).
"""

from __future__ import annotations

import sqlite3
import zlib
from pathlib import Path

from karc.util import sha256_hex, utc_now_iso

# Placement threshold: compressed/stored size ≤ this → DB BLOB, else external.
DB_BLOB_MAX_BYTES = 512 * 1024
# zlib level 6 (default): good ratio/speed balance; deterministic output.
ZLIB_LEVEL = 6


class CASCorruption(RuntimeError):
    """Stored object failed sha256 re-verification (§9.3 step 2, §9.4)."""


def objects_root(db_path: str | Path) -> Path:
    return Path(db_path).parent / "objects"


def _external_rel(content_hash: str) -> Path:
    return Path("sha256") / content_hash[:2] / f"{content_hash[2:]}.zz"


def _compress(raw: bytes) -> tuple[bytes, str]:
    """Return (stored_bytes, compression). Falls back to raw when zlib does
    not shrink the payload (incompressible / tiny)."""
    comp = zlib.compress(raw, ZLIB_LEVEL)
    if len(comp) < len(raw):
        return comp, "zlib"
    return raw, "none"


def _decompress(stored: bytes, compression: str) -> bytes:
    if compression == "zlib":
        return zlib.decompress(stored)
    if compression == "none":
        return stored
    # 'zstd' is declared in the enum for forward-compat but never written by
    # this stdlib-only build.
    raise CASCorruption(f"unsupported compression {compression!r}")


def store_bytes(conn: sqlite3.Connection, db_path: str | Path, raw: bytes) -> str:
    """Store ``raw`` and return its content_hash. Idempotent (dedup by hash)."""
    content_hash = sha256_hex(raw)
    row = conn.execute(
        "SELECT 1 FROM cas_objects WHERE content_hash = ?", (content_hash,)
    ).fetchone()
    if row is not None:
        return content_hash  # already stored — structural dedup

    stored, compression = _compress(raw)
    if len(stored) <= DB_BLOB_MAX_BYTES:
        conn.execute(
            """INSERT OR IGNORE INTO cas_objects
                 (content_hash, algo, size_bytes, stored_size, compression,
                  storage, data, last_verified_at)
               VALUES (?, 'sha256', ?, ?, ?, 'db', ?, ?)""",
            (content_hash, len(raw), len(stored), compression, stored, utc_now_iso()),
        )
    else:
        rel = _external_rel(content_hash)
        abs_path = objects_root(db_path) / rel
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = abs_path.with_suffix(abs_path.suffix + ".tmp")
        with open(tmp, "wb") as f:
            f.write(stored)
            f.flush()
            import os

            os.fsync(f.fileno())
        tmp.replace(abs_path)  # atomic rename
        conn.execute(
            """INSERT OR IGNORE INTO cas_objects
                 (content_hash, algo, size_bytes, stored_size, compression,
                  storage, external_path, last_verified_at)
               VALUES (?, 'sha256', ?, ?, ?, 'external', ?, ?)""",
            (content_hash, len(raw), len(stored), compression, str(rel), utc_now_iso()),
        )
    return content_hash


def store_file(conn: sqlite3.Connection, db_path: str | Path, path: str | Path) -> str:
    """Store the current content of ``path`` (read fully; snapshots are small
    knowledge artifacts, not arbitrary large binaries)."""
    with open(path, "rb") as f:
        raw = f.read()
    return store_bytes(conn, db_path, raw)


def load(conn: sqlite3.Connection, db_path: str | Path, content_hash: str) -> bytes:
    """Return the original bytes for ``content_hash``, re-verifying sha256
    (§9.3 step 2). Raises CASCorruption on mismatch or missing object."""
    row = conn.execute(
        "SELECT compression, storage, data, external_path FROM cas_objects "
        "WHERE content_hash = ?",
        (content_hash,),
    ).fetchone()
    if row is None:
        raise CASCorruption(f"CAS object {content_hash} not found")
    compression, storage, data, external_path = row
    if storage == "db":
        stored = bytes(data)
    else:
        abs_path = objects_root(db_path) / external_path
        with open(abs_path, "rb") as f:
            stored = f.read()
    raw = _decompress(stored, compression)
    actual = sha256_hex(raw)
    if actual != content_hash:
        raise CASCorruption(
            f"CAS object {content_hash} failed verification (got {actual})"
        )
    return raw


def verify(conn: sqlite3.Connection, db_path: str | Path, content_hash: str) -> bool:
    """Re-hash a stored object; update last_verified_at on success."""
    try:
        load(conn, db_path, content_hash)
    except (CASCorruption, OSError):
        return False
    conn.execute(
        "UPDATE cas_objects SET last_verified_at = ? WHERE content_hash = ?",
        (utc_now_iso(), content_hash),
    )
    return True


def exists(conn: sqlite3.Connection, content_hash: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM cas_objects WHERE content_hash = ?", (content_hash,)
        ).fetchone()
        is not None
    )
