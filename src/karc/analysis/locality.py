"""E0-1 locality metrics (experiment plan §4, cache-policy Q10).

Computed per scope, for two breakdowns (all files / ``.md`` files only):

- task-distinct re-reference rate: fraction of referenced artifacts that were
  referenced in >= 2 distinct tasks (reference stream is task-coalesced,
  cache-policy §2.3).
- reuse distance distribution in task units: for each artifact, gaps between
  successive distinct task indices that referenced it (task order = task
  ``started_at`` within the scope); percentiles reported.
- frequency skew: log-log Zipf slope (least squares on rank vs task-distinct
  reference count) + share of references held by the top 10% artifacts.
- virtual ARC replay ghost hits: unit-size plain ARC (Megiddo & Modha 2003)
  replayed over the task-coalesced reference stream at capacity = 10% and
  25% of distinct artifacts; B1/B2 ghost hits counted.

Judgment thresholds (experiment plan E0-1, pre-registered — do not modify):
  re-reference rate >= 20% AND ghost hits observed -> ARC retained;
  < 5% -> low-locality pivot (FIFO-equivalent);
  5–20% -> intermediate (follow-up V12 + FIFO replay).

Data adequacy gate (experiment plan E0-1): >= 3 scopes with data AND >= 500
total tasks. The gate is evaluated by ``data_adequacy``; callers must not run
the judgment when the gate fails.
"""

from __future__ import annotations

import math
import sqlite3
from collections import OrderedDict
from dataclasses import dataclass, field

MD_EXTENSIONS = (".md", ".markdown")

# Pre-registered thresholds (experiment plan E0-1) — DO NOT CHANGE.
RE_REF_RETAIN = 0.20
RE_REF_PIVOT = 0.05
MIN_SCOPES = 3
MIN_TASKS = 500
CAPACITY_FRACTIONS = (0.10, 0.25)


# ---------------------------------------------------------------------------
# Plain ARC (unit size) for ghost-hit replay
# ---------------------------------------------------------------------------

@dataclass
class ArcCounters:
    capacity: int = 0
    accesses: int = 0
    hits: int = 0
    misses: int = 0
    b1_ghost_hits: int = 0
    b2_ghost_hits: int = 0

    @property
    def ghost_hits(self) -> int:
        return self.b1_ghost_hits + self.b2_ghost_hits


class PlainARC:
    """Textbook ARC with unit-size pages (Megiddo & Modha, FAST '03)."""

    def __init__(self, capacity: int):
        assert capacity >= 1
        self.c = capacity
        self.p = 0.0
        # OrderedDicts: first item = LRU, last item = MRU
        self.t1: OrderedDict = OrderedDict()
        self.t2: OrderedDict = OrderedDict()
        self.b1: OrderedDict = OrderedDict()
        self.b2: OrderedDict = OrderedDict()
        self.counters = ArcCounters(capacity=capacity)

    def _replace(self, x, in_b2: bool) -> None:
        if self.t1 and (len(self.t1) > self.p or (in_b2 and len(self.t1) == self.p)):
            lru, _ = self.t1.popitem(last=False)
            self.b1[lru] = None
        else:
            lru, _ = self.t2.popitem(last=False)
            self.b2[lru] = None

    def access(self, x) -> str:
        """Process one reference. Returns 'hit' | 'b1' | 'b2' | 'miss'."""
        ctr = self.counters
        ctr.accesses += 1
        if x in self.t1 or x in self.t2:
            self.t1.pop(x, None)
            self.t2.pop(x, None)
            self.t2[x] = None
            ctr.hits += 1
            return "hit"
        if x in self.b1:
            self.p = min(float(self.c), self.p + max(1.0, len(self.b2) / len(self.b1)))
            self._replace(x, in_b2=False)
            del self.b1[x]
            self.t2[x] = None
            ctr.b1_ghost_hits += 1
            return "b1"
        if x in self.b2:
            self.p = max(0.0, self.p - max(1.0, len(self.b1) / len(self.b2)))
            self._replace(x, in_b2=True)
            del self.b2[x]
            self.t2[x] = None
            ctr.b2_ghost_hits += 1
            return "b2"
        # miss
        ctr.misses += 1
        l1 = len(self.t1) + len(self.b1)
        if l1 == self.c:
            if len(self.t1) < self.c:
                self.b1.popitem(last=False)
                self._replace(x, in_b2=False)
            else:
                self.t1.popitem(last=False)
        else:
            total = l1 + len(self.t2) + len(self.b2)
            if total >= self.c:
                if total == 2 * self.c:
                    self.b2.popitem(last=False)
                self._replace(x, in_b2=False)
        self.t1[x] = None
        return "miss"


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def _percentiles(sorted_values: list, points=(50, 75, 90, 95)) -> dict:
    if not sorted_values:
        return {f"p{q}": None for q in points} | {"max": None, "n": 0}
    out = {}
    n = len(sorted_values)
    for q in points:
        # nearest-rank percentile
        rank = max(1, math.ceil(q / 100.0 * n))
        out[f"p{q}"] = sorted_values[rank - 1]
    out["max"] = sorted_values[-1]
    out["n"] = n
    return out


def _zipf_slope(counts_desc: list[int]) -> float | None:
    """Least-squares slope of log10(count) vs log10(rank). None if < 3 points."""
    pts = [(math.log10(r), math.log10(c)) for r, c in enumerate(counts_desc, 1) if c > 0]
    if len(pts) < 3:
        return None
    n = len(pts)
    sx = sum(x for x, _ in pts)
    sy = sum(y for _, y in pts)
    sxx = sum(x * x for x, _ in pts)
    sxy = sum(x * y for x, y in pts)
    denom = n * sxx - sx * sx
    if denom == 0:
        return None
    return (n * sxy - sx * sy) / denom


@dataclass
class BreakdownMetrics:
    label: str
    n_read_events: int = 0
    n_events_with_task: int = 0
    n_events_without_task: int = 0
    n_artifacts: int = 0
    n_tasks_referencing: int = 0
    n_task_distinct_refs: int = 0
    n_rereferenced_artifacts: int = 0
    re_reference_rate: float | None = None
    reuse_distance: dict = field(default_factory=dict)
    zipf_slope: float | None = None
    top10pct_share: float | None = None
    arc_replays: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def _compute_breakdown(
    label: str,
    events: list[tuple[str, str, str | None, str, str]],
    task_order: dict[str, int],
) -> BreakdownMetrics:
    """events: [(occurred_at, event_id, task_id, artifact_id, path)] sorted."""
    m = BreakdownMetrics(label=label)
    m.n_read_events = len(events)

    # Task-coalesced reference stream, in chronological order of first
    # reference within each (task, artifact) pair.
    seen_pairs: set[tuple[str, str]] = set()
    stream: list[str] = []  # artifact ids
    artifact_tasks: dict[str, list[int]] = {}
    for occurred_at, event_id, task_id, artifact_id, _path in events:
        if task_id is None or task_id not in task_order:
            m.n_events_without_task += 1
            continue
        m.n_events_with_task += 1
        pair = (task_id, artifact_id)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        stream.append(artifact_id)
        artifact_tasks.setdefault(artifact_id, []).append(task_order[task_id])

    m.n_artifacts = len(artifact_tasks)
    m.n_task_distinct_refs = len(stream)
    m.n_tasks_referencing = len({t for t, _ in seen_pairs})

    if m.n_artifacts == 0:
        return m

    # Re-reference rate
    re_referenced = [a for a, ts in artifact_tasks.items() if len(set(ts)) >= 2]
    m.n_rereferenced_artifacts = len(re_referenced)
    m.re_reference_rate = len(re_referenced) / m.n_artifacts

    # Reuse distance (task units)
    gaps: list[int] = []
    for ts in artifact_tasks.values():
        uniq = sorted(set(ts))
        gaps.extend(b - a for a, b in zip(uniq, uniq[1:]))
    m.reuse_distance = _percentiles(sorted(gaps))

    # Frequency skew
    counts = sorted((len(set(ts)) for ts in artifact_tasks.values()), reverse=True)
    m.zipf_slope = _zipf_slope(counts)
    total_refs = sum(counts)
    top_n = max(1, math.ceil(0.10 * len(counts)))
    m.top10pct_share = sum(counts[:top_n]) / total_refs if total_refs else None

    # Virtual ARC replay (unit size), capacities = fractions of distinct artifacts
    for frac in CAPACITY_FRACTIONS:
        cap = max(1, round(frac * m.n_artifacts))
        arc = PlainARC(cap)
        for a in stream:
            arc.access(a)
        c = arc.counters
        m.arc_replays.append(
            {
                "capacity_fraction": frac,
                "capacity": cap,
                "accesses": c.accesses,
                "hits": c.hits,
                "misses": c.misses,
                "b1_ghost_hits": c.b1_ghost_hits,
                "b2_ghost_hits": c.b2_ghost_hits,
                "ghost_hits": c.ghost_hits,
            }
        )
    return m


def _is_md(path: str) -> bool:
    return path.lower().endswith(MD_EXTENSIONS)


def analyze_scope(conn: sqlite3.Connection, scope_id: str) -> dict:
    scope_row = conn.execute(
        "SELECT root_path, display_name FROM scopes WHERE scope_id = ?", (scope_id,)
    ).fetchone()
    # Scope task order: all tasks of sessions assigned to this scope.
    task_rows = conn.execute(
        """
        SELECT t.task_id FROM tasks t JOIN sessions s ON s.session_id = t.session_id
        WHERE s.scope_id = ? ORDER BY t.started_at, t.task_id
        """,
        (scope_id,),
    ).fetchall()
    task_order = {r[0]: i for i, r in enumerate(task_rows)}

    events = conn.execute(
        """
        SELECT e.occurred_at, e.event_id, e.task_id, e.artifact_id,
               COALESCE(a.canonical_path, a.logical_name)
        FROM events e JOIN artifacts a ON a.artifact_id = e.artifact_id
        WHERE e.event_type = 'read' AND e.scope_id = ?
        ORDER BY e.occurred_at, e.event_id
        """,
        (scope_id,),
    ).fetchall()

    all_files = _compute_breakdown("all_files", events, task_order)
    md_only = _compute_breakdown(
        "md_only", [e for e in events if _is_md(e[4])], task_order
    )
    return {
        "scope_id": scope_id,
        "root_path": scope_row[0],
        "display_name": scope_row[1],
        "n_tasks": len(task_order),
        "breakdowns": {"all_files": all_files.as_dict(), "md_only": md_only.as_dict()},
        "judgment": {
            "all_files": judge(all_files),
            "md_only": judge(md_only),
        },
    }


def judge(m: BreakdownMetrics) -> dict:
    """Pre-registered E0-1 judgment (experiment plan §4 — verbatim rules)."""
    if m.re_reference_rate is None:
        return {"verdict": "no-data", "re_reference_rate": None, "ghost_hits": 0}
    ghost = sum(r["ghost_hits"] for r in m.arc_replays)
    if m.re_reference_rate >= RE_REF_RETAIN and ghost > 0:
        verdict = "arc-retain"
    elif m.re_reference_rate < RE_REF_PIVOT:
        verdict = "low-locality-pivot"
    else:
        verdict = "intermediate-v12-fifo-replay"
    return {
        "verdict": verdict,
        "re_reference_rate": m.re_reference_rate,
        "ghost_hits": ghost,
    }


def data_adequacy(conn: sqlite3.Connection) -> dict:
    """E0-1 data adequacy gate: >= 3 scopes with data, >= 500 tasks total."""
    scope_rows = conn.execute(
        """
        SELECT s.scope_id, sc.root_path, COUNT(DISTINCT t.task_id)
        FROM scopes sc
        LEFT JOIN sessions s ON s.scope_id = sc.scope_id
        LEFT JOIN tasks t ON t.session_id = s.session_id
        GROUP BY sc.scope_id
        """
    ).fetchall()
    n_tasks = sum(r[2] for r in scope_rows)
    scopes_with_tasks = sum(1 for r in scope_rows if r[2] > 0)
    return {
        "n_scopes": len({r[1] for r in scope_rows}),
        "n_scopes_with_tasks": scopes_with_tasks,
        "n_tasks_total": n_tasks,
        "min_scopes_required": MIN_SCOPES,
        "min_tasks_required": MIN_TASKS,
        "adequate": scopes_with_tasks >= MIN_SCOPES and n_tasks >= MIN_TASKS,
    }


def analyze_all(conn: sqlite3.Connection) -> dict:
    """Full E0-1 analysis: per-scope metrics + task-weighted aggregate."""
    scope_ids = [
        r[0] for r in conn.execute("SELECT scope_id FROM scopes ORDER BY root_path")
    ]
    per_scope = [analyze_scope(conn, sid) for sid in scope_ids]

    aggregate = {}
    for label in ("all_files", "md_only"):
        entries = [
            (s["breakdowns"][label], s["n_tasks"])
            for s in per_scope
            if s["breakdowns"][label]["re_reference_rate"] is not None
        ]
        if not entries:
            aggregate[label] = None
            continue
        w_total = sum(w for _, w in entries) or 1
        weighted_rate = sum(b["re_reference_rate"] * w for b, w in entries) / w_total
        pooled_artifacts = sum(b["n_artifacts"] for b, _ in entries)
        pooled_rerefs = sum(b["n_rereferenced_artifacts"] for b, _ in entries)
        ghost = sum(sum(r["ghost_hits"] for r in b["arc_replays"]) for b, _ in entries)
        aggregate[label] = {
            "task_weighted_re_reference_rate": weighted_rate,
            "pooled_re_reference_rate": (
                pooled_rerefs / pooled_artifacts if pooled_artifacts else None
            ),
            "total_ghost_hits": ghost,
            "n_scopes_with_data": len(entries),
        }
    return {
        "adequacy": data_adequacy(conn),
        "per_scope": per_scope,
        "aggregate": aggregate,
        "parameters": {
            "capacity_fractions": list(CAPACITY_FRACTIONS),
            "thresholds": {
                "retain_re_reference_rate": RE_REF_RETAIN,
                "pivot_re_reference_rate": RE_REF_PIVOT,
            },
            "coalescing": "task-distinct (artifact_id, task_id)",
            "task_boundary": "user_turn (R-4 heuristic)",
        },
    }


def artifact_reference_counts(conn: sqlite3.Connection, scope_id: str) -> list[dict]:
    """Per-artifact reference counts (path + counts only — R-9)."""
    rows = conn.execute(
        """
        SELECT COALESCE(a.canonical_path, a.logical_name) AS path,
               COUNT(*) AS read_events,
               COUNT(DISTINCT e.task_id) AS distinct_tasks
        FROM events e JOIN artifacts a ON a.artifact_id = e.artifact_id
        WHERE e.event_type = 'read' AND e.scope_id = ?
        GROUP BY e.artifact_id
        ORDER BY read_events DESC
        """,
        (scope_id,),
    ).fetchall()
    return [
        {"path": r[0], "read_events": r[1], "distinct_tasks": r[2]} for r in rows
    ]
