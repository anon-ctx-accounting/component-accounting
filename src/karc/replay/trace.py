"""Real-trace replay — FR-R8: fold canonical events from the M0 database.

Implements the data-model §8 identity: ``state = fold(config, sort(events))``
with sort order ``ORDER BY occurred_at, event_id`` — the same ``on_event``
code path as live/batch ingestion and synthetic replay.

Fixture ground truth does not exist for real traces, so metrics are
online-hit based (experiment-design E1-3): for each task, the share of its
referenced artifacts that were already in the working set when the task
started (task-weighted), plus token-weighted variants.

Size resolution (FR-R5 fallback chain, Amendment A-3(d) backfill): per-
artifact max observed ``events.token_cost`` → file-size estimate from a stat
of the artifact's current ``canonical_path`` (bytes/4, §8.1 heuristic; the
DB is never mutated — backfill happens at registry-load time) → corpus-
median default 800. Per-source counts are reported in the run summary
(``size_backfill``) so Q9-style sensitivity can be assessed later.
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass

from karc.db import connection
from karc.policy import InvariantViolation, PolicyConfig, make_policy
from karc.policy.model import ArtifactMeta, Event
from karc.replay.runner import HARNESS_VERSION, git_hash, resolve_budget

DEFAULT_SIZE_TOK = 800


@dataclass
class TraceRunResult:
    summary: dict
    task_rows: list[dict]
    violation: dict | None = None

    @property
    def ok(self) -> bool:
        return self.violation is None


def list_scopes(db_path: str) -> list[dict]:
    conn = connection.connect(db_path)
    rows = conn.execute(
        """SELECT s.scope_id, s.root_path,
                  (SELECT COUNT(*) FROM events e WHERE e.scope_id = s.scope_id) AS n_events,
                  (SELECT COUNT(*) FROM artifacts a WHERE a.scope_id = s.scope_id) AS n_artifacts
           FROM scopes s ORDER BY n_events DESC"""
    ).fetchall()
    return [
        {"scope_id": r[0], "root_path": r[1], "events": r[2], "artifacts": r[3]}
        for r in rows
    ]


def stat_token_estimate(path: str | None) -> int | None:
    """A-3(d): file-size based token estimate for a currently existing file.

    bytes/4 heuristic (§8.1 default; a stat cannot see content, so no
    language-aware divisor — reported as a limitation). None when the path
    is missing, not a regular file, or empty.
    """
    if not path:
        return None
    try:
        st = os.stat(path)
    except OSError:
        return None
    if not os.path.isfile(path) or st.st_size <= 0:
        return None
    return max(1, int(st.st_size) // 4)


def _load_registry(
    conn: sqlite3.Connection, scope_id: str
) -> tuple[dict[str, ArtifactMeta], dict[str, int]]:
    """Registry + size-provenance counts (A-3(d) backfill, DB untouched)."""
    sizes: dict[str, int] = {}
    for a, tc in conn.execute(
        "SELECT artifact_id, MAX(token_cost) FROM events "
        "WHERE scope_id = ? AND token_cost IS NOT NULL GROUP BY artifact_id",
        (scope_id,),
    ):
        sizes[a] = int(tc)
    registry: dict[str, ArtifactMeta] = {}
    provenance = {"from_token_cost": 0, "from_file_stat": 0, "default": 0}
    for a, name, path, crit, pinned in conn.execute(
        "SELECT artifact_id, logical_name, canonical_path, criticality, pinned "
        "FROM artifacts WHERE scope_id = ?",
        (scope_id,),
    ):
        if a in sizes:
            size = sizes[a]
            provenance["from_token_cost"] += 1
        else:
            est = stat_token_estimate(path)
            if est is not None:
                size = est
                provenance["from_file_stat"] += 1
            else:
                size = DEFAULT_SIZE_TOK
                provenance["default"] += 1
        registry[a] = ArtifactMeta(
            artifact_id=a,
            size_tok=size,
            path=path or name or a,
            critical=(crit == "critical"),
            pinned=bool(pinned),
        )
    return registry, provenance


def _load_events(conn: sqlite3.Connection, scope_id: str) -> list[Event]:
    rows = conn.execute(
        """SELECT event_id, event_type, occurred_at, artifact_id, task_id,
                  session_id, load_class, token_cost
           FROM events WHERE scope_id = ?
           ORDER BY occurred_at, event_id""",
        (scope_id,),
    ).fetchall()
    return [
        Event(
            event_id=r[0],
            event_type=r[1],
            occurred_at=r[2],
            artifact_id=r[3],
            task_id=r[4],
            session_id=r[5],
            scope_id=scope_id,
            load_class=r[6],
            token_cost=r[7],
            origin="trace",
        )
        for r in rows
    ]


def replay_trace(
    db_path: str,
    scope_id: str,
    policy_name: str = "karc",
    budget: str | int = "25pct",
    config_overrides: dict | None = None,
    debug_invariants: bool = True,
) -> TraceRunResult:
    conn = connection.connect(db_path)
    registry, size_backfill = _load_registry(conn, scope_id)
    events = _load_events(conn, scope_id)
    corpus_tokens = sum(m.size_tok for m in registry.values())
    budget_tokens = resolve_budget(budget, max(1, corpus_tokens))
    config = PolicyConfig.from_overrides(
        PolicyConfig(c=budget_tokens, debug_invariants=debug_invariants),
        **(config_overrides or {}),
    )
    policy = make_policy(policy_name, config, registry)

    # online-hit metrics per task (ground truth 없음 — E1-3 대체 지표)
    task_rows: list[dict] = []
    violation: dict | None = None
    current_task: str | None = None
    row: dict | None = None
    seen_in_task: set[str] = set()

    def close_row():
        if row is not None:
            task_rows.append(dict(row))

    run_id = f"trace.{policy_name}.{scope_id[:12]}.c{budget_tokens}"
    try:
        for e in events:
            tid = e.task_id or "(no-task)"
            if tid != current_task:
                close_row()
                current_task = tid
                seen_in_task = set()
                row = {
                    "task_id": tid,
                    "refs_distinct": 0,
                    "online_hits": 0,
                    "token_refs": 0,
                    "token_hits": 0,
                    "events": 0,
                }
            row["events"] += 1
            if e.is_reference() and e.artifact_id not in seen_in_task:
                seen_in_task.add(e.artifact_id)
                size = registry.get(
                    e.artifact_id, ArtifactMeta(e.artifact_id, DEFAULT_SIZE_TOK)
                ).size_tok
                row["refs_distinct"] += 1
                row["token_refs"] += size
                if policy.is_resident(e.artifact_id):
                    row["online_hits"] += 1
                    row["token_hits"] += size
            policy.on_event(e)
    except InvariantViolation as exc:
        violation = {"run_id": run_id, "config": config.as_dict(), **exc.dump()}
    close_row()

    refs = sum(r["refs_distinct"] for r in task_rows)
    hits = sum(r["online_hits"] for r in task_rows)
    tok_refs = sum(r["token_refs"] for r in task_rows)
    tok_hits = sum(r["token_hits"] for r in task_rows)
    summary = {
        "run_id": run_id,
        "mode": "real-trace",
        "policy": policy_name,
        "scope_id": scope_id,
        "budget": str(budget),
        "budget_tokens": budget_tokens,
        "n_events": len(events),
        "n_tasks": len(task_rows),
        "n_artifacts": len(registry),
        "corpus_tokens_estimate": corpus_tokens,
        "size_default_used": DEFAULT_SIZE_TOK,
        "size_backfill": size_backfill,  # A-3(d) provenance counts
        "invariant_violations": 0 if violation is None else 1,
        "metrics": {
            "online_hit_rate": (hits / refs) if refs else None,
            "token_weighted_online_hit_rate": (tok_hits / tok_refs) if tok_refs else None,
            "ghost_hits": policy.summary().get("ghost_hits"),
            "critical_auto_cold": policy.summary().get("cold_auto_critical"),
            "recommendations": policy.summary().get("recommendations", {}),
        },
        "state_hash": policy.state_hash(),
        "stamp": {
            "harness_version": HARNESS_VERSION,
            "git_hash": git_hash(),
            "db_path": str(db_path),
            "config": config.as_dict(),
        },
    }
    return TraceRunResult(summary=summary, task_rows=task_rows, violation=violation)
