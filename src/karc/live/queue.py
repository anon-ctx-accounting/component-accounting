"""Persist the live recommendation queue and apply human decisions.

The fold (``state.compute``) is the source of truth for *what* should be
recommended; this module reconciles that into the append-only
``recommendations`` table so ``review`` has stable rec_ids and a decision
history (data-model §11 Q4), and routes every acceptance through the mutation
core (checkpoint a) — the only path that changes artifact state.

Reconciliation rules (no hard delete — trg_no_del_recs):

- A candidate whose (scope, action, artifact_id) already has a *decided* row
  (accepted/rejected/rolled_back) is NOT re-surfaced.
- A candidate with no row gets a fresh ``pending`` row.
- A ``pending`` row whose candidate disappeared is marked ``superseded``.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from karc.live import state as live_state
from karc.snapshot import store
from karc.util import new_ulid, utc_now_iso

_DECIDED = ("accepted", "rejected", "rolled_back")


def refresh(
    conn: sqlite3.Connection,
    db_path: str | Path,
    scope_id: str,
    budget=live_state.DEFAULT_BUDGET,
    config_overrides: dict | None = None,
) -> live_state.LiveState:
    """Fold + reconcile pending recommendations. Returns the LiveState with
    ``rec_id`` populated on each active item."""
    ls = live_state.compute(conn, db_path, scope_id, budget, config_overrides)
    active_keys = {(r.action, r.artifact_id) for r in ls.active}

    # supersede pending rows whose candidate is gone
    for rec_id, action, artifact_id in conn.execute(
        "SELECT rec_id, action, artifact_id FROM recommendations "
        "WHERE scope_id = ? AND status = 'pending'",
        (scope_id,),
    ).fetchall():
        if (action, artifact_id) not in active_keys:
            conn.execute(
                "UPDATE recommendations SET status = 'superseded' WHERE rec_id = ?",
                (rec_id,),
            )

    # ensure each active candidate has a live (pending) row, unless already decided
    for item in ls.active:
        existing = conn.execute(
            "SELECT rec_id, status FROM recommendations "
            "WHERE scope_id = ? AND action = ? AND COALESCE(artifact_id,'') = ? "
            "ORDER BY created_at DESC LIMIT 1",
            (scope_id, item.action, item.artifact_id),
        ).fetchone()
        if existing is not None and existing[1] in _DECIDED:
            item.rec_id = None  # decided already; not offered again
            continue
        if existing is not None and existing[1] == "pending":
            item.rec_id = existing[0]
            continue
        rec_id = new_ulid()
        conn.execute(
            """INSERT INTO recommendations
                 (rec_id, scope_id, artifact_id, action, rule_id, rationale, risk,
                  requires_approval)
               VALUES (?, ?, ?, ?, ?, ?, ?, 1)""",
            (
                rec_id,
                scope_id,
                item.artifact_id,
                item.action,
                item.rule_id,
                json.dumps({"summary": item.rationale}, ensure_ascii=False),
                item.risk,
            ),
        )
        item.rec_id = rec_id
    # drop items that were already decided (no rec_id) from the active queue
    ls.active = [i for i in ls.active if i.rec_id is not None]
    return ls


def pending_rows(conn: sqlite3.Connection, scope_id: str) -> list[dict]:
    rows = conn.execute(
        """SELECT r.rec_id, r.action, r.artifact_id, r.rule_id, r.rationale, r.risk,
                  a.logical_name
             FROM recommendations r
             LEFT JOIN artifacts a ON a.artifact_id = r.artifact_id
            WHERE r.scope_id = ? AND r.status = 'pending'
            ORDER BY r.action, r.created_at""",
        (scope_id,),
    ).fetchall()
    out = []
    for rec_id, action, artifact_id, rule_id, rationale, risk, name in rows:
        try:
            rat = json.loads(rationale).get("summary", rationale)
        except (json.JSONDecodeError, TypeError):
            rat = rationale
        out.append({
            "rec_id": rec_id, "action": action, "artifact_id": artifact_id,
            "rule_id": rule_id, "rationale": rat, "risk": risk, "name": name,
        })
    return out


# action → how an acceptance is applied through the mutation core
def _apply_accept(conn, db_path, rec_id, action, artifact_id, actor, confirm_token):
    if action == "pin":
        # pin-suggestions come from critical path patterns → protect as critical
        return store.pin(conn, artifact_id, actor=actor, critical=True)
    if action == "archive":
        return store.archive(conn, db_path, artifact_id, actor=actor, rec_id=rec_id)
    if action == "cold_transition":
        return store.archive(
            conn, db_path, artifact_id, actor=actor, rec_id=rec_id, target_state="cold"
        )
    if action == "invalidate_or_correct":
        # Approved invalidate: remove from the working set, content preserved &
        # restorable (MVP mapping — documented; R-7 keeps it reversible).
        return store.archive(conn, db_path, artifact_id, actor=actor, rec_id=rec_id)
    if action == "split":
        # R-7: split is recommendation-only — accepting acknowledges it, no
        # file mutation. Recorded in the audit log for traceability.
        audit_id = store.audit(
            conn, "user", actor, "split_acknowledged", "artifact", artifact_id or "",
            {"rec_id": rec_id},
        )
        return store.MutationResult("split", artifact_id or "", audit_id=audit_id)
    raise store.MutationError(f"cannot apply action {action!r}")


def decide(
    conn: sqlite3.Connection,
    db_path: str | Path,
    rec_id: str,
    decision: str,
    actor: str = "user",
    confirm_token: str | None = None,
) -> dict:
    """Apply accept/reject to one pending recommendation. Accept routes through
    the mutation core; reject just records the decision."""
    row = conn.execute(
        "SELECT action, artifact_id, status FROM recommendations WHERE rec_id = ?",
        (rec_id,),
    ).fetchone()
    if row is None:
        raise store.MutationError(f"unknown recommendation {rec_id}")
    action, artifact_id, status = row
    if status != "pending":
        raise store.MutationError(f"recommendation {rec_id} is {status}, not pending")

    if decision == "reject":
        conn.execute(
            "UPDATE recommendations SET status='rejected', decided_at=?, decided_by=? "
            "WHERE rec_id=?",
            (utc_now_iso(), actor, rec_id),
        )
        return {"rec_id": rec_id, "action": action, "decision": "rejected"}

    if decision == "accept":
        res = _apply_accept(conn, db_path, rec_id, action, artifact_id, actor, confirm_token)
        conn.execute(
            "UPDATE recommendations SET status='accepted', decided_at=?, decided_by=?, "
            "applied_audit_id=?, fs_snapshot_id=COALESCE(fs_snapshot_id, ?) WHERE rec_id=?",
            (utc_now_iso(), actor, res.audit_id, res.fs_snapshot_id, rec_id),
        )
        return {"rec_id": rec_id, "action": action, "decision": "accepted",
                "result": res.detail, "audit_id": res.audit_id}

    raise store.MutationError(f"unknown decision {decision!r}")
