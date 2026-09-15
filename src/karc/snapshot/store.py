"""Mutation core — the only path that changes artifact lifecycle state.

Every mutation the product surface performs (archive/restore/pin/unpin, and
review acceptances) routes through here so that the safety principles (brief
§10, data-model §10) hold uniformly:

- **No hard delete** — archive is a ``lifecycle_state`` transition, never a
  file/row delete (10-1/10-2). The on-disk file is left in place; archive only
  removes the artifact from discovery/preload and preserves its content in the
  CAS so ``restore`` is guaranteed (§9.2 invariant).
- **Snapshot before mutation** — archive stores the current content; restore
  takes a ``pre_restore`` snapshot of whatever is on disk first (10-3, §9.3).
- **Atomic apply** — restore writes to a temp file, fsync, atomic rename
  (§9.3 step 4). Multi-file restores stage all temps before renaming.
- **Audit everything** — one append-only ``audit_log`` row per mutation with
  before/after hashes and ``rollback_of`` linkage (10-10).
- **Explainable friction** — unpinning a ``critical`` artifact requires a
  typed filename confirmation, enforced by the caller via ``confirm_token``.
"""

from __future__ import annotations

import fnmatch
import json
import os
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from karc.snapshot import cas
from karc.util import sha256_file, utc_now_iso


class MutationError(RuntimeError):
    """A mutation was refused (unknown artifact, missing confirm token, …)."""


class UnpinConfirmationRequired(MutationError):
    """Unpinning a critical artifact needs a typed filename confirmation."""


@dataclass
class MutationResult:
    action: str
    artifact_id: str
    audit_id: int | None = None
    fs_snapshot_id: int | None = None
    detail: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# audit / snapshot primitives
# ---------------------------------------------------------------------------

def audit(
    conn: sqlite3.Connection,
    actor_type: str,
    actor: str | None,
    action: str,
    object_type: str,
    object_id: str,
    details: dict | None = None,
    rollback_of: int | None = None,
) -> int:
    cur = conn.execute(
        """INSERT INTO audit_log
             (actor_type, actor, action, object_type, object_id, details, rollback_of)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            actor_type,
            actor,
            action,
            object_type,
            object_id,
            json.dumps(details, ensure_ascii=False, sort_keys=True) if details else None,
            rollback_of,
        ),
    )
    return int(cur.lastrowid)


def create_fs_snapshot(
    conn: sqlite3.Connection,
    db_path: str | Path,
    scope_id: str,
    reason: str,
    files: list[tuple[str | None, str]],
    rec_id: str | None = None,
) -> int | None:
    """Snapshot the current on-disk content of ``files`` (list of
    ``(artifact_id, path)``). Files that do not exist on disk are skipped.
    Returns the snapshot_id, or None when no file could be captured."""
    captured: list[tuple[str | None, str, str, int | None, str | None]] = []
    for artifact_id, path in files:
        if not path or not os.path.isfile(path):
            continue
        content_hash = cas.store_file(conn, db_path, path)
        try:
            st = os.stat(path)
            mode, mtime = st.st_mode, str(st.st_mtime)
        except OSError:
            mode, mtime = None, None
        captured.append((artifact_id, path, content_hash, mode, mtime))
    if not captured:
        return None
    cur = conn.execute(
        "INSERT INTO fs_snapshots (scope_id, reason, rec_id) VALUES (?, ?, ?)",
        (scope_id, reason, rec_id),
    )
    snapshot_id = int(cur.lastrowid)
    for artifact_id, path, content_hash, mode, mtime in captured:
        conn.execute(
            """INSERT INTO fs_snapshot_files
                 (snapshot_id, artifact_id, path, content_hash, file_mode, mtime)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (snapshot_id, artifact_id, path, content_hash, mode, mtime),
        )
    return snapshot_id


# ---------------------------------------------------------------------------
# artifact resolution
# ---------------------------------------------------------------------------

@dataclass
class ArtifactRow:
    artifact_id: str
    scope_id: str
    logical_name: str
    canonical_path: str | None
    criticality: str
    pinned: int
    lifecycle_state: str
    current_version_id: str | None


def get_artifact(conn: sqlite3.Connection, artifact_id: str) -> ArtifactRow | None:
    row = conn.execute(
        "SELECT artifact_id, scope_id, logical_name, canonical_path, criticality, "
        "pinned, lifecycle_state, current_version_id FROM artifacts WHERE artifact_id = ?",
        (artifact_id,),
    ).fetchone()
    return ArtifactRow(*row) if row else None


def resolve_artifact(
    conn: sqlite3.Connection, ref: str, scope_id: str | None = None
) -> ArtifactRow | None:
    """Resolve a user-supplied reference to a single artifact.

    Accepts an artifact_id, a logical name, or a path (relative or absolute).
    Ambiguous names raise MutationError with the candidates.
    """
    direct = get_artifact(conn, ref)
    if direct is not None:
        return direct
    clauses = ["logical_name = ?", "canonical_path = ?"]
    params: list = [ref, ref]
    # also match by basename of logical_name
    q = (
        "SELECT artifact_id, scope_id, logical_name, canonical_path, criticality, "
        "pinned, lifecycle_state, current_version_id FROM artifacts "
        f"WHERE ({' OR '.join(clauses)})"
    )
    if scope_id:
        q += " AND scope_id = ?"
        params.append(scope_id)
    rows = conn.execute(q, params).fetchall()
    if not rows:
        # basename fallback
        q2 = (
            "SELECT artifact_id, scope_id, logical_name, canonical_path, criticality, "
            "pinned, lifecycle_state, current_version_id FROM artifacts "
            "WHERE logical_name LIKE ?"
        )
        p2: list = [f"%{ref}"]
        if scope_id:
            q2 += " AND scope_id = ?"
            p2.append(scope_id)
        rows = [
            r
            for r in conn.execute(q2, p2).fetchall()
            if os.path.basename(r[2]) == ref
        ]
    if not rows:
        return None
    if len(rows) > 1:
        raise MutationError(
            f"{ref!r} is ambiguous ({len(rows)} artifacts): "
            + ", ".join(r[0] for r in rows[:5])
        )
    return ArtifactRow(*rows[0])


def _glob_to_regex(pattern: str):
    import re

    # fnmatch's '*' already spans '/', so 'runbooks/**' and 'runbooks/*' both
    # match any descendant path — sufficient for path-glob pins.
    return re.compile(fnmatch.translate(pattern))


def match_artifacts(
    conn: sqlite3.Connection, pattern: str, scope_id: str | None = None
) -> list[ArtifactRow]:
    """Return artifacts whose logical_name / canonical_path / basename match a
    glob ``pattern`` (or a single exact reference)."""
    single = None
    try:
        single = resolve_artifact(conn, pattern, scope_id)
    except MutationError:
        single = None
    if single is not None and not any(c in pattern for c in "*?["):
        return [single]
    rx = _glob_to_regex(pattern)
    q = (
        "SELECT artifact_id, scope_id, logical_name, canonical_path, criticality, "
        "pinned, lifecycle_state, current_version_id FROM artifacts"
    )
    params: list = []
    if scope_id:
        q += " WHERE scope_id = ?"
        params.append(scope_id)
    out: list[ArtifactRow] = []
    for r in conn.execute(q, params).fetchall():
        art = ArtifactRow(*r)
        cands = [art.logical_name, os.path.basename(art.logical_name)]
        if art.canonical_path:
            cands.append(art.canonical_path)
        if any(rx.match(c) for c in cands):
            out.append(art)
    return out


# ---------------------------------------------------------------------------
# mutations
# ---------------------------------------------------------------------------

def pin(
    conn: sqlite3.Connection,
    artifact_id: str,
    actor: str = "user",
    critical: bool = False,
) -> MutationResult:
    art = get_artifact(conn, artifact_id)
    if art is None:
        raise MutationError(f"unknown artifact {artifact_id}")
    new_crit = "critical" if critical else art.criticality
    conn.execute(
        "UPDATE artifacts SET pinned = 1, criticality = ?, updated_at = ? "
        "WHERE artifact_id = ?",
        (new_crit, utc_now_iso(), artifact_id),
    )
    audit_id = audit(
        conn, "user", actor, "pin", "artifact", artifact_id,
        {"pinned": {"before": art.pinned, "after": 1},
         "criticality": {"before": art.criticality, "after": new_crit}},
    )
    return MutationResult("pin", artifact_id, audit_id=audit_id,
                          detail={"criticality": new_crit})


def unpin(
    conn: sqlite3.Connection,
    artifact_id: str,
    actor: str = "user",
    confirm_token: str | None = None,
) -> MutationResult:
    art = get_artifact(conn, artifact_id)
    if art is None:
        raise MutationError(f"unknown artifact {artifact_id}")
    # friction-proportional-to-risk (UX §5-6): critical artifacts require a
    # typed filename confirmation to unpin.
    if art.criticality == "critical":
        expected = os.path.basename(art.logical_name)
        if confirm_token != expected:
            raise UnpinConfirmationRequired(
                f"unpinning critical artifact requires typing its filename "
                f"({expected!r})"
            )
    conn.execute(
        "UPDATE artifacts SET pinned = 0, updated_at = ? WHERE artifact_id = ?",
        (utc_now_iso(), artifact_id),
    )
    audit_id = audit(
        conn, "user", actor, "unpin", "artifact", artifact_id,
        {"pinned": {"before": art.pinned, "after": 0}},
    )
    return MutationResult("unpin", artifact_id, audit_id=audit_id)


def _current_version_hash(conn: sqlite3.Connection, artifact_id: str) -> str | None:
    row = conn.execute(
        "SELECT v.content_hash FROM versions v JOIN artifacts a "
        "ON a.current_version_id = v.version_id WHERE a.artifact_id = ?",
        (artifact_id,),
    ).fetchone()
    return row[0] if row else None


def archive(
    conn: sqlite3.Connection,
    db_path: str | Path,
    artifact_id: str,
    actor: str = "user",
    rec_id: str | None = None,
    target_state: str = "archived",
) -> MutationResult:
    """Archive/cold = lifecycle transition + content preservation (never delete).

    The file stays on disk; only discovery/preload exclude it (cold keeps the
    artifact discoverable). We snapshot the current content into the CAS so the
    §9.2 invariant holds (``lifecycle_state IN ('cold','archived')`` ⇒
    content_hash ∈ cas_objects). ``target_state`` is 'archived' (default) or
    'cold' (for an approved critical cold_transition)."""
    if target_state not in ("archived", "cold"):
        raise MutationError(f"invalid target_state {target_state!r}")
    art = get_artifact(conn, artifact_id)
    if art is None:
        raise MutationError(f"unknown artifact {artifact_id}")
    if art.lifecycle_state == target_state:
        raise MutationError(f"{artifact_id} is already {target_state}")

    snapshot_id = None
    stored_hash = None
    if art.canonical_path and os.path.isfile(art.canonical_path):
        snapshot_id = create_fs_snapshot(
            conn, db_path, art.scope_id, "archive",
            [(artifact_id, art.canonical_path)], rec_id=rec_id,
        )
        stored_hash = sha256_file(art.canonical_path)
    else:
        # File already gone: preserve whatever the current version points to,
        # if its content is already in the CAS (best-effort).
        cur_hash = _current_version_hash(conn, artifact_id)
        if cur_hash and cas.exists(conn, cur_hash):
            stored_hash = cur_hash

    conn.execute(
        "UPDATE artifacts SET lifecycle_state = ?, updated_at = ? WHERE artifact_id = ?",
        (target_state, utc_now_iso(), artifact_id),
    )
    audit_id = audit(
        conn, "user", actor, target_state, "artifact", artifact_id,
        {"lifecycle_state": {"before": art.lifecycle_state, "after": target_state},
         "content_hash": stored_hash, "fs_snapshot_id": snapshot_id},
    )
    if rec_id is not None:
        conn.execute(
            "UPDATE recommendations SET fs_snapshot_id = COALESCE(fs_snapshot_id, ?) "
            "WHERE rec_id = ?",
            (snapshot_id, rec_id),
        )
    return MutationResult(
        target_state, artifact_id, audit_id=audit_id, fs_snapshot_id=snapshot_id,
        detail={"content_hash": stored_hash, "restorable": stored_hash is not None},
    )


def dependents(conn: sqlite3.Connection, artifact_id: str) -> list[dict]:
    """Live edges pointing AT this artifact (§9.3 'dependency 재연결 안내')."""
    rows = conn.execute(
        """SELECT e.edge_type, e.src_artifact_id, a.logical_name
             FROM artifact_edges e JOIN artifacts a ON a.artifact_id = e.src_artifact_id
            WHERE e.dst_artifact_id = ? AND e.invalidated_at IS NULL""",
        (artifact_id,),
    ).fetchall()
    return [{"edge_type": r[0], "src_artifact_id": r[1], "src_name": r[2]} for r in rows]


def restore(
    conn: sqlite3.Connection,
    db_path: str | Path,
    artifact_id: str,
    actor: str = "user",
    archive_audit_id: int | None = None,
) -> MutationResult:
    """Restore an archived artifact: re-enable discovery and, if the on-disk
    file is missing or diverged, rewrite it from the CAS (verified, atomic)."""
    art = get_artifact(conn, artifact_id)
    if art is None:
        raise MutationError(f"unknown artifact {artifact_id}")

    target_hash = _current_version_hash(conn, artifact_id)
    reconnected = dependents(conn, artifact_id)
    wrote_file = False
    pre_restore_snapshot = None

    if target_hash and cas.exists(conn, target_hash) and art.canonical_path:
        path = art.canonical_path
        on_disk = (
            sha256_file(path) if os.path.isfile(path) else None
        )
        if on_disk != target_hash:
            # pre_restore snapshot of whatever is currently on disk (§9.3-3)
            if os.path.isfile(path):
                pre_restore_snapshot = create_fs_snapshot(
                    conn, db_path, art.scope_id, "pre_restore", [(artifact_id, path)]
                )
            raw = cas.load(conn, db_path, target_hash)  # verifies sha256
            tmp = Path(str(path) + ".karc-restore.tmp")
            tmp.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "wb") as f:
                f.write(raw)
                f.flush()
                os.fsync(f.fileno())
            tmp.replace(path)  # atomic rename (§9.3-4)
            wrote_file = True

    conn.execute(
        "UPDATE artifacts SET lifecycle_state = 'active', updated_at = ? "
        "WHERE artifact_id = ?",
        (utc_now_iso(), artifact_id),
    )
    audit_id = audit(
        conn, "user", actor, "restore", "artifact", artifact_id,
        {"lifecycle_state": {"before": art.lifecycle_state, "after": "active"},
         "content_hash": target_hash, "wrote_file": wrote_file,
         "pre_restore_snapshot_id": pre_restore_snapshot,
         "reconnected": reconnected},
        rollback_of=archive_audit_id,
    )
    return MutationResult(
        "restore", artifact_id, audit_id=audit_id, fs_snapshot_id=pre_restore_snapshot,
        detail={"wrote_file": wrote_file, "reconnected": reconnected,
                "content_hash": target_hash},
    )
