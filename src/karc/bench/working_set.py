"""Stage 2 c* working-set derivation (E2-1 arms (b) plain ARC / (c) K-ARC).

E2-1 runs the frozen fixture task sequence (``tasks.json``, ``seq`` order) so
that a policy's state *accumulates across tasks* (§7 E2-1). Before running the
LLM on task *i*, the classic/K-ARC arms must materialize the working set the
policy would supply given everything it has learned from tasks ``1..i-1``.
This module produces that supply plan — deterministically, with **no LLM
call** — so it can be fixed before any run (pre-registration friendly) and fed
identically to both policies (§8.4-2 fairness).

Two pieces:

1. ``stage2_task_specs`` — the task-sequence → event-stream converter. Each
   frozen task expands into a scripted reference stream plus session-boundary
   preload loads, reusing the Stage-1 ``TaskSpec``/``Event`` value types so the
   exact same replay fold (``policy.on_event`` + §6.5 ``outcome_events``)
   drives the state. The converter is the single source of the *exogenous*
   stream both policies observe.

2. ``derive_working_sets`` / ``build_plans`` — the working-set deriver. Folds
   the scripted stream through a configured policy and snapshots
   ``policy.working_set()`` **before** each task's events (i.e. the state after
   tasks ``1..i-1``), which is exactly the set the harness materializes for
   task *i*.

Operationalization choices (fixed here, before any run — mirroring the E1
driver convention of pinning choices in the docstring):

- **Sequence index / sessions.** ``idx = task["seq"]`` (1-based). Sessions
  group five tasks (``SESSION_TASKS`` = Stage-1 ``workload.SESSION_TASKS``);
  ``session_id = "S{(seq-1)//5:03d}"``. Session starts (``(seq-1) % 5 == 0``)
  carry a preload block, matching the Stage-1 ``preload_block`` cadence.
- **Reference events.** Each ``required_artifacts`` entry (chains listed in
  order) emits exactly one ``read`` — the minimal reference that makes the
  §6.5 supplied set a superset of ``required``. No probabilistic cited/applied
  (Stage-1 adds those for realism; here determinism and a clean supply plan
  win). These reads are the ground-truth ``refs`` that feed the outcome rule.
- **Preload events.** Session starts emit ``loaded``/``preload`` for every
  ``preload_set`` artifact (sorted). Under the confirmed config
  (``preload_as_hit=False``) and plain ARC these are reference-inert — they
  never enter the reference track and never gain residency — so the ARC/K-ARC
  working set is built from the reference stream alone, NOT the preload dump.
  They are still emitted so both policies see the identical stream (§8.4-2)
  and so the funnel counters (``preload_count``) match a real session.
- **forbidden_stale.** Recorded on the ``TaskSpec`` but never *read*: an agent
  should not read a superseded doc. The only way a forbidden doc reaches the
  supplied set is by being resident in ``ws_start``; for ARC/K-ARC the stale
  v1 docs live only in the (reference-inert) preload stream, so they never
  become resident and the §6.5 correction path stays dormant for these arms.
  (The static-full arm, by contrast, materializes the whole preload set incl.
  the stale v1 docs — that asymmetry is the point of the counterfactual, not a
  fairness break.) No ``validity_changed`` events are scripted: with the v1
  docs never resident there is nothing to invalidate in the resident set.
- **Outcome.** The §6.5 deterministic rule at ``alpha_obs = 1.0`` via
  ``replay.outcome.outcome_events`` — the *scripted* outcome, NOT an LLM
  grader result. The supply decision is made before the run, so it cannot
  depend on the run; and a per-arm grader loop would diverge the streams and
  break §8.4-2. Here the outcome events coincide across policies anyway (every
  task succeeds, so ``validated`` fires on ``required``; ``corrected`` never
  fires), so the "same event stream" property holds exactly.

Determinism (I9): the fold is a pure function of (corpus seed, frozen tasks,
budget, config); reruns are byte-identical. The plan carries the corpus
content hash so a caller can bind it to the frozen fixture.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from karc.policy import InvariantViolation, PolicyConfig, make_policy
from karc.policy.model import Event
from karc.replay.corpus import Corpus
from karc.replay.outcome import outcome_events
from karc.replay.workload import TASK_GAP_MIN, TaskSpec, _ts

SESSION_TASKS = 5  # == workload.SESSION_TASKS (preload cadence)


def stage2_task_specs(corpus: Corpus, tasks: list[dict]) -> list[TaskSpec]:
    """Expand frozen fixture tasks (seq order) into scripted ``TaskSpec``s.

    Returns one ``TaskSpec`` per task with its observed ``events`` (preload +
    read) and ground-truth ``refs``/``required``/``forbidden``. The stream is
    identical for every policy (the exogenous track of §8.4-2)."""
    ordered = sorted(tasks, key=lambda t: t["seq"])
    preload_ids = sorted(corpus.groups["preload_set"])
    specs: list[TaskSpec] = []
    eid = 0
    for t in ordered:
        seq = int(t["seq"])
        idx = seq
        session_id = f"S{(seq - 1) // SESSION_TASKS:03d}"
        spec = TaskSpec(
            idx=idx,
            task_id=t["task_id"],
            session_id=session_id,
            segment=str(t.get("phase", "")),
            required=list(t.get("required_artifacts", [])),
            forbidden=list(t.get("forbidden_artifacts", t.get("forbidden_stale", []))),
        )
        # (etype, artifact, load_class) — preload block first, then reads.
        pending: list[tuple[str, str, str | None]] = []
        if (seq - 1) % SESSION_TASKS == 0:
            pending += [("loaded", a, "preload") for a in preload_ids]
        for a in t.get("required_artifacts", []):
            pending.append(("read", a, None))
            spec.refs.append(("read", a))  # ground truth (INT-15 supplied set)
        base_min = idx * TASK_GAP_MIN
        for j, (etype, artifact, load_class) in enumerate(pending):
            eid += 1
            spec.events.append(
                Event(
                    event_id=f"s2.e{eid:06d}",
                    event_type=etype,
                    occurred_at=_ts(base_min + j * (5.0 / 60.0)),
                    artifact_id=artifact,
                    task_id=t["task_id"],
                    session_id=session_id,
                    scope_id="fixture",
                    load_class=load_class,
                )
            )
        specs.append(spec)
    return specs


@dataclass
class WorkingSetPlan:
    """Per-task supply plan for one policy at budget c*."""

    policy: str
    budget_tokens: int
    corpus_content_hash: str
    by_task: dict[str, list[str]] = field(default_factory=dict)
    tokens_by_task: dict[str, int] = field(default_factory=dict)
    final_state_hash: str = ""
    violation: dict | None = None

    @property
    def ok(self) -> bool:
        return self.violation is None

    def as_dict(self) -> dict:
        return {
            "policy": self.policy,
            "budget_tokens": self.budget_tokens,
            "corpus_content_hash": self.corpus_content_hash,
            "by_task": {k: list(v) for k, v in self.by_task.items()},
            "tokens_by_task": dict(self.tokens_by_task),
            "final_state_hash": self.final_state_hash,
            "violation": self.violation,
        }


def derive_working_sets(
    policy_name: str,
    corpus: Corpus,
    tasks: list[dict],
    *,
    budget_tokens: int,
    config_overrides: dict | None = None,
    alpha_obs: float = 1.0,
    inject_outcomes: bool = True,
    specs: list[TaskSpec] | None = None,
) -> WorkingSetPlan:
    """Fold the scripted stream through ``policy_name`` at budget ``c*``.

    Snapshots ``policy.working_set()`` *before* each task's events — the state
    after tasks ``1..i-1`` — as the supply set for task *i*. Same fold as
    ``replay.runner.run_replay`` (INT-10: plain ARC applies its factory preset
    on top of ``config_overrides``, so token metering is identical)."""
    specs = specs if specs is not None else stage2_task_specs(corpus, tasks)
    config = PolicyConfig.from_overrides(
        PolicyConfig(c=budget_tokens, debug_invariants=True), **(config_overrides or {})
    )
    registry = corpus.registry()
    policy = make_policy(policy_name, config, registry)
    plan = WorkingSetPlan(
        policy=policy.name,
        budget_tokens=budget_tokens,
        corpus_content_hash=corpus.content_hash(),
    )
    for task in specs:
        ws_start = policy.working_set()  # state after tasks 1..i-1 = supply for i
        ws_ids = sorted(ws_start)
        plan.by_task[task.task_id] = ws_ids
        plan.tokens_by_task[task.task_id] = sum(
            registry[a].size_tok for a in ws_ids if a in registry
        )
        try:
            for e in task.events:
                policy.on_event(e)
            if inject_outcomes:
                _rec, out_events = outcome_events(task, ws_start, corpus.seed, alpha_obs)
                for e in out_events:
                    policy.on_event(e)
        except InvariantViolation as exc:
            plan.violation = {"task_id": task.task_id, "task_idx": task.idx, **exc.dump()}
            break
    plan.final_state_hash = policy.state_hash()
    return plan


def build_plans(
    corpus: Corpus,
    tasks: list[dict],
    *,
    budget_tokens: int,
    confirmed_config: dict,
) -> dict[str, WorkingSetPlan]:
    """Both E2-1 counterfactual arms from one shared stream.

    ``classic`` → plain ARC (factory preset over the confirmed config, INT-10);
    ``karc`` → K-ARC at the confirmed config. Identical ``specs`` guarantee the
    §8.4-2 same-event-stream property."""
    specs = stage2_task_specs(corpus, tasks)
    return {
        "classic": derive_working_sets(
            "arc", corpus, tasks, budget_tokens=budget_tokens,
            config_overrides=confirmed_config, specs=specs,
        ),
        "karc": derive_working_sets(
            "karc", corpus, tasks, budget_tokens=budget_tokens,
            config_overrides=confirmed_config, specs=specs,
        ),
    }
