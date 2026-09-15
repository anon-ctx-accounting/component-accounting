"""Metric calculators — FR-R7 (전부 secondary/탐색적 지표 패널; primary 판정
규칙 자체는 pre-registration §2.1이 정의하고 통계 스크립트가 수행한다).

Precise operationalizations (INT-13 — names/roles follow the pre-registered
docs; where the doc leaves the fine definition open, it is fixed HERE, in
code, before any official run):

- coverage(task)        = |required ∩ WS_start| / |required| (None if the
                          task has no required artifacts; run mean is over
                          defined tasks — §2.1 P3 "task 가중 평균").
- stale_exposure(task)  = tok(WS_start ∩ invalid_at(task)) / tok(WS_start)
                          where invalid_at comes from the workload's ground
                          truth timeline (not the policy's belief).
- token-weighted hit rate = Σ size_tok(hit refs) / Σ size_tok(refs), counted
                          once per (task, artifact) first observed reference
                          (task-distinct; retry loops must not inflate it),
                          residency tested at access time. The raw per-event
                          rate is reported alongside as *_event.
- adaptation lag        = number of tasks after the shift until per-task
                          coverage first reaches ≥ 95% of the mean coverage
                          of the last 10 pre-shift tasks (None if never).
- archive regret rate   = |cold-transitioned artifacts later referenced
                          (ground truth)| / |cold-transitioned artifacts|.
- churn                 = evictions per task (resident-set removals).
- promotion precision   = fraction of T1→T2 promotions whose artifact is
                          referenced in ≥ 1 later task (ground truth).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from karc.policy.base import Policy
from karc.policy.model import Event
from karc.replay.workload import TaskSpec, Workload


@dataclass
class TaskRow:
    task_idx: int
    task_id: str
    segment: str
    coverage: float | None
    ws_tokens: int
    ws_size: int
    stale_exposure: float
    stale_tokens: int
    refs_distinct: int
    hits_distinct: int
    token_refs: int
    token_hits: int
    events: int
    success: bool | None = None
    supplied_forbidden: int = 0
    validated_emitted: int = 0
    corrected_emitted: int = 0
    evictions_delta: int = 0
    ghost_hits_delta: int = 0
    recommendations_delta: int = 0

    def as_dict(self) -> dict:
        return dict(self.__dict__)


class MetricsAccumulator:
    def __init__(self, workload: Workload, policy: Policy):
        self.workload = workload
        self.policy = policy
        self.rows: list[TaskRow] = []
        self.invalid_from: dict[str, int] = dict(workload.meta.get("invalid_from", {}))
        self._task_seen: set[tuple[str, str]] = set()
        self._event_task: dict[str, int] = {}
        self._cold_seen: set[str] = set()
        self._cold_at: dict[str, int] = {}  # artifact -> task idx of cold transition
        self._raw_refs = 0
        self._raw_hits = 0
        self._raw_token_refs = 0
        self._raw_token_hits = 0
        self._prev = {"evictions": 0, "ghost_hits": 0, "recs": 0}

    # ------------------------------------------------------------------
    def task_start(self, task: TaskSpec, ws: dict[str, int]) -> TaskRow:
        invalid = {a for a, t0 in self.invalid_from.items() if t0 <= task.idx}
        required = set(task.required)
        coverage = None
        if required:
            coverage = len(required & set(ws)) / len(required)
        ws_tokens = sum(ws.values())
        stale_tokens = sum(tok for a, tok in ws.items() if a in invalid)
        row = TaskRow(
            task_idx=task.idx,
            task_id=task.task_id,
            segment=task.segment,
            coverage=coverage,
            ws_tokens=ws_tokens,
            ws_size=len(ws),
            stale_exposure=(stale_tokens / ws_tokens) if ws_tokens else 0.0,
            stale_tokens=stale_tokens,
            refs_distinct=0,
            hits_distinct=0,
            token_refs=0,
            token_hits=0,
            events=len(task.events),
        )
        self.rows.append(row)
        return row

    def observe_event(self, task: TaskSpec, e: Event, resident_before: bool, size: int) -> None:
        self._event_task[e.event_id] = task.idx
        if not e.is_reference():  # metric-layer reference track (D2, fixed)
            return
        self._raw_refs += 1
        self._raw_token_refs += size
        if resident_before:
            self._raw_hits += 1
            self._raw_token_hits += size
        key = (task.task_id, e.artifact_id)
        if key in self._task_seen:
            return
        self._task_seen.add(key)
        row = self.rows[-1]
        row.refs_distinct += 1
        row.token_refs += size
        if resident_before:
            row.hits_distinct += 1
            row.token_hits += size

    def task_end(self, task: TaskSpec, outcome) -> None:
        row = self.rows[-1]
        if outcome is not None:
            row.success = outcome.success
            row.supplied_forbidden = len(outcome.supplied_forbidden)
            row.validated_emitted = len(outcome.validated_emitted)
            row.corrected_emitted = len(outcome.corrected_emitted)
        summ = self.policy.summary()
        ev = int(summ.get("evictions", 0))
        gh = int(summ.get("ghost_hits", 0))
        recs = len(self.policy.recommendations)
        row.evictions_delta = ev - self._prev["evictions"]
        row.ghost_hits_delta = gh - self._prev["ghost_hits"]
        row.recommendations_delta = recs - self._prev["recs"]
        self._prev = {"evictions": ev, "ghost_hits": gh, "recs": recs}
        # cold transitions observed at task granularity (archive regret)
        cold_ever = getattr(self.policy, "cold_ever", None)
        if cold_ever is not None:
            for a in cold_ever - self._cold_seen:
                self._cold_at[a] = task.idx
            self._cold_seen = set(cold_ever)

    # ------------------------------------------------------------------
    def _adaptation_lag(self) -> float | None:
        shift = self.workload.meta.get("shift_task_idx")
        if shift is None:
            return None
        pre = [r.coverage for r in self.rows[max(0, shift - 10) : shift] if r.coverage is not None]
        if not pre:
            return None
        target = 0.95 * (sum(pre) / len(pre))
        for r in self.rows[shift:]:
            if r.coverage is not None and r.coverage >= target:
                return float(r.task_idx - shift)
        return None

    def _archive_regret(self) -> tuple[float | None, int, int]:
        if not self._cold_at:
            return None, 0, 0
        regret = 0
        for a, t0 in self._cold_at.items():
            for task in self.workload.tasks[t0 + 1 :]:
                if any(x == a for _, x in task.refs):
                    regret += 1
                    break
        return regret / len(self._cold_at), regret, len(self._cold_at)

    def _promotion_precision(self) -> tuple[float | None, int]:
        transitions = getattr(self.policy, "transitions", [])
        promos = [
            t
            for t in transitions
            if t.rule_id.endswith("two-distinct-task-refs")
        ]
        if not promos:
            return None, 0
        # ground-truth reference timeline per artifact
        refs_at: dict[str, list[int]] = {}
        for task in self.workload.tasks:
            for _, a in task.refs:
                refs_at.setdefault(a, []).append(task.idx)
        good = 0
        for t in promos:
            promo_task = self._event_task.get(t.evidence_event_id)
            if promo_task is None:
                continue
            later = [i for i in refs_at.get(t.artifact_id, []) if i > promo_task]
            if later:
                good += 1
        return good / len(promos), len(promos)

    def finalize(self) -> dict:
        cov = [r.coverage for r in self.rows if r.coverage is not None]
        stale = [r.stale_exposure for r in self.rows]
        tok_refs = sum(r.token_refs for r in self.rows)
        tok_hits = sum(r.token_hits for r in self.rows)
        succ = [r.success for r in self.rows if r.success is not None]
        regret_rate, regret_n, cold_n = self._archive_regret()
        promo_precision, promo_n = self._promotion_precision()
        summ = self.policy.summary()
        return {
            "n_tasks": len(self.rows),
            "coverage_mean": (sum(cov) / len(cov)) if cov else None,
            "full_coverage_rate": (
                sum(1 for c in cov if c >= 1.0) / len(cov) if cov else None
            ),
            "stale_exposure_mean": sum(stale) / len(stale) if stale else 0.0,
            "token_weighted_hit_rate": (tok_hits / tok_refs) if tok_refs else None,
            "hit_rate_distinct": (
                sum(r.hits_distinct for r in self.rows)
                / max(1, sum(r.refs_distinct for r in self.rows))
            ),
            "token_weighted_hit_rate_event": (
                self._raw_token_hits / self._raw_token_refs if self._raw_token_refs else None
            ),
            "hit_rate_event": (self._raw_hits / self._raw_refs) if self._raw_refs else None,
            "adaptation_lag": self._adaptation_lag(),
            "archive_regret_rate": regret_rate,
            "archive_regret_n": regret_n,
            "cold_transitions": cold_n,
            "churn_per_task": (
                sum(r.evictions_delta for r in self.rows) / max(1, len(self.rows))
            ),
            "promotion_precision": promo_precision,
            "promotions": promo_n,
            "ghost_hits": summ.get("ghost_hits"),
            "critical_auto_cold": summ.get("cold_auto_critical"),
            "critical_resident_evictions": summ.get("critical_resident_evictions"),
            "success_rate": (sum(1 for s in succ if s) / len(succ)) if succ else None,
            "validated_emitted": sum(r.validated_emitted for r in self.rows),
            "corrected_emitted": sum(r.corrected_emitted for r in self.rows),
            "recommendations": summ.get("recommendations", {}),
        }
