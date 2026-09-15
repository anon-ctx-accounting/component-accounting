"""Deterministic outcome injector — experiment-design §6.5 / FR-R4.

Rule (verbatim from the pre-registration, fixed in simulator code — no human
or LLM judgement):

    success := (공급 집합 ⊇ required) ∧ (공급 집합 ∩ forbidden = ∅)
    success        → ``validated`` on required ∩ 공급 집합
    forbidden 공급 → ``corrected`` on 공급 집합 ∩ forbidden

Emission is subsampled at observation rate α_obs (E1-5) with a per-
(seed, task, artifact, kind) hash — NOT a sequential RNG — so the subsample
is identical across policies and pairing is preserved (§8.4-2).

Interpretation decisions:
INT-15 공급 집합(task) := working set at task START ∪ ground-truth references
       during the task. K-ARC is advisory (reads are never blocked), so
       everything the agent read was supplied on demand; telemetry noise
       (drop) must not change the world, so the ground-truth reference plan
       — not the noisy observed stream — feeds the rule.
INT-12 The loop is closed per policy: outcomes derive from the evaluated
       policy's own supplied set (§6.5's replay procedure; live analog is
       grader feedback 회송). §8.4-2's "동일 공급" holds for the exogenous
       stream; classic policies receiving their own outcome events ignore
       them by design.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from karc.policy.model import Event
from karc.replay.corpus import _unit
from karc.replay.workload import TASK_GAP_MIN, TaskSpec, _ts


@dataclass
class OutcomeRecord:
    task_id: str
    success: bool
    supplied_forbidden: list[str] = field(default_factory=list)
    missing_required: list[str] = field(default_factory=list)
    validated_emitted: list[str] = field(default_factory=list)
    corrected_emitted: list[str] = field(default_factory=list)


def _keep(seed: int, task_id: str, artifact: str, kind: str, alpha_obs: float) -> bool:
    if alpha_obs >= 1.0:
        return True
    if alpha_obs <= 0.0:
        return False
    return _unit(seed, "alpha_obs", task_id, artifact, kind) < alpha_obs


def outcome_events(
    task: TaskSpec,
    ws_start: dict[str, int],
    seed: int,
    alpha_obs: float,
) -> tuple[OutcomeRecord, list[Event]]:
    """Compute the §6.5 rule for one finished task and build the events."""
    supplied = set(ws_start) | {a for _, a in task.refs}  # INT-15
    required = set(task.required)
    forbidden = set(task.forbidden)
    supplied_forbidden = sorted(supplied & forbidden)
    missing = sorted(required - supplied)
    success = not missing and not supplied_forbidden

    rec = OutcomeRecord(
        task_id=task.task_id,
        success=success,
        supplied_forbidden=supplied_forbidden,
        missing_required=missing,
    )
    events: list[Event] = []
    base_min = task.idx * TASK_GAP_MIN + 40.0  # after task events, before next task
    if success:
        targets = [(a, "validated") for a in sorted(required & supplied)]
    elif supplied_forbidden:
        targets = [(a, "corrected") for a in supplied_forbidden]
    else:
        targets = []  # missing-required failure emits nothing (§6.5)
    for j, (a, kind) in enumerate(targets):
        if not _keep(seed, task.task_id, a, kind, alpha_obs):
            continue
        events.append(
            Event(
                event_id=f"out.{task.task_id}.{a}.{kind[0]}",
                event_type=kind,
                occurred_at=_ts(base_min + j * (5.0 / 60.0)),
                artifact_id=a,
                task_id=task.task_id,
                session_id=task.session_id,
                scope_id="fixture",
                origin="outcome-injector",
            )
        )
        (rec.validated_emitted if kind == "validated" else rec.corrected_emitted).append(a)
    return rec, events
