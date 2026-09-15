"""Integrity verification (``karc fsck``) — data-model §9.4 / §7.3-5.

Reports only; never deletes (hard-delete absence, §10). Checks:

1. CAS re-hash — decompress + sha256, oldest ``last_verified_at`` first when
   incremental. Corrupt objects are listed.
2. §9.2 invariant — every ``lifecycle_state IN ('cold','archived')`` artifact's
   current version content_hash must exist in ``cas_objects`` (the restore
   guarantee).
3. Supersede chain — recursive-CTE scan for cycles and runaway depth (>64).
4. Structural — >1 live version per artifact, >1 active row per resolved_path.
5. Orphan external blobs — files under ``objects/`` with no ``cas_objects`` row
   (reported, not removed).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from karc.snapshot import cas


@dataclass
class FsckReport:
    cas_checked: int = 0
    cas_corrupt: list[str] = field(default_factory=list)
    missing_archive_content: list[str] = field(default_factory=list)
    supersede_cycles: list[str] = field(default_factory=list)
    supersede_overdepth: list[str] = field(default_factory=list)
    multi_live_versions: list[str] = field(default_factory=list)
    dup_active_paths: list[str] = field(default_factory=list)
    orphan_blobs: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (
            self.cas_corrupt
            or self.missing_archive_content
            or self.supersede_cycles
            or self.supersede_overdepth
            or self.multi_live_versions
            or self.dup_active_paths
        )

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "cas_checked": self.cas_checked,
            "cas_corrupt": self.cas_corrupt,
            "missing_archive_content": self.missing_archive_content,
            "supersede_cycles": self.supersede_cycles,
            "supersede_overdepth": self.supersede_overdepth,
            "multi_live_versions": self.multi_live_versions,
            "dup_active_paths": self.dup_active_paths,
            "orphan_blobs": self.orphan_blobs,
        }


def run(
    conn: sqlite3.Connection, db_path: str | Path, limit: int | None = None
) -> FsckReport:
    rep = FsckReport()

    # 1. CAS re-hash (oldest verified first — incremental friendly)
    q = "SELECT content_hash FROM cas_objects ORDER BY last_verified_at IS NULL DESC, last_verified_at"
    if limit:
        q += f" LIMIT {int(limit)}"
    for (content_hash,) in conn.execute(q).fetchall():
        rep.cas_checked += 1
        if not cas.verify(conn, db_path, content_hash):
            rep.cas_corrupt.append(content_hash)

    # 2. §9.2 invariant
    for (artifact_id, content_hash) in conn.execute(
        """SELECT a.artifact_id, v.content_hash
             FROM artifacts a LEFT JOIN versions v ON a.current_version_id = v.version_id
            WHERE a.lifecycle_state IN ('cold','archived')"""
    ).fetchall():
        if content_hash is None or not cas.exists(conn, content_hash):
            rep.missing_archive_content.append(artifact_id)

    # 3. supersede chain integrity (§7.3-5)
    for (vid,) in conn.execute("SELECT version_id FROM versions").fetchall():
        seen = set()
        cur = vid
        depth = 0
        while True:
            nxt = conn.execute(
                "SELECT superseded_by_version_id FROM versions WHERE version_id = ?",
                (cur,),
            ).fetchone()
            nxt = nxt[0] if nxt else None
            if nxt is None:
                break
            if nxt in seen or nxt == vid:
                rep.supersede_cycles.append(vid)
                break
            seen.add(nxt)
            depth += 1
            if depth > 64:
                rep.supersede_overdepth.append(vid)
                break
            cur = nxt

    # 4. structural
    for (artifact_id, n) in conn.execute(
        "SELECT artifact_id, COUNT(*) FROM versions WHERE invalidated_at IS NULL "
        "GROUP BY artifact_id HAVING COUNT(*) > 1"
    ).fetchall():
        rep.multi_live_versions.append(artifact_id)
    for (path, n) in conn.execute(
        "SELECT resolved_path, COUNT(*) FROM artifact_paths "
        "WHERE status IN ('active_canonical','active_alias') "
        "GROUP BY resolved_path HAVING COUNT(*) > 1"
    ).fetchall():
        rep.dup_active_paths.append(path)

    # 5. orphan external blobs (report only)
    root = cas.objects_root(db_path)
    if root.is_dir():
        known = set()
        for (ep,) in conn.execute(
            "SELECT external_path FROM cas_objects WHERE storage = 'external'"
        ).fetchall():
            if ep:
                known.add(str(root / ep))
        for p in root.rglob("*"):
            if p.is_file() and not p.name.endswith(".tmp") and str(p) not in known:
                rep.orphan_blobs.append(str(p.relative_to(root)))

    return rep
