"""Replay runner — FR-R1/R2/R4/R5/R6/R7/R9 orchestration.

One run = (policy, family, seed, budget, noise, α_obs, config) → per-task
JSONL rows + run summary with reproduction stamp (corpus content hash,
harness git hash, full config) and final state hash.

Determinism (I9/FR-R2): randomness exists only inside the seeded workload
generator; the run itself is a pure fold. ``run_replay`` executed twice with
the same spec yields byte-identical summaries (asserted by tests).

Invariant violations (FR-R6): the engine raises ``InvariantViolation``; the
runner marks the run failed and attaches the reproduction dump (config,
seed, offending event, state payload) instead of crashing the batch.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from karc.policy import InvariantViolation, PolicyConfig, make_policy
from karc.replay.corpus import Corpus, generate_corpus
from karc.replay.metrics import MetricsAccumulator
from karc.replay.outcome import outcome_events
from karc.replay.workload import Workload, generate_workload

HARNESS_VERSION = "karc-replay-0.1.0"


def resolve_budget(budget: str | int | float, corpus_tokens: int) -> int:
    """FR-R5: budget as absolute tokens ('8000') or corpus share ('10pct')."""
    if isinstance(budget, (int, float)):
        return int(budget)
    b = str(budget).strip().lower()
    if b.endswith("pct"):
        return max(1, int(corpus_tokens * float(b[:-3]) / 100.0))
    if b.endswith("%"):
        return max(1, int(corpus_tokens * float(b[:-1]) / 100.0))
    return int(b)


def git_hash(repo_dir: str | Path | None = None) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_dir) if repo_dir else None,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


@dataclass
class RunSpec:
    policy: str
    family: str
    seed: int
    budget: str | int = "10pct"
    alpha_obs: float = 1.0
    drop_rate: float = 0.0
    shuffle: bool = False
    task_id_missing_rate: float = 0.0
    inject_outcomes: bool = True
    debug_invariants: bool = True
    config_overrides: dict = field(default_factory=dict)

    def run_id(self, budget_tokens: int) -> str:
        noise = ""
        if self.drop_rate or self.shuffle or self.task_id_missing_rate:
            noise = f".d{self.drop_rate}.sh{int(self.shuffle)}.tm{self.task_id_missing_rate}"
        return (
            f"{self.policy}.{self.family}.s{self.seed}.c{budget_tokens}"
            f".a{self.alpha_obs}{noise}"
        )


@dataclass
class RunResult:
    summary: dict
    task_rows: list[dict]
    violation: dict | None = None

    @property
    def ok(self) -> bool:
        return self.violation is None


def run_replay(
    spec: RunSpec,
    corpus: Corpus | None = None,
    workload: Workload | None = None,
    _git_hash: str | None = None,
) -> RunResult:
    corpus = corpus or generate_corpus(spec.seed)
    budget_tokens = resolve_budget(spec.budget, corpus.total_tokens)
    if workload is None:
        workload = generate_workload(
            spec.family,
            spec.seed,
            corpus=corpus,
            budget_tokens=budget_tokens,
            drop_rate=spec.drop_rate,
            shuffle=spec.shuffle,
            task_id_missing_rate=spec.task_id_missing_rate,
        )
    config = PolicyConfig.from_overrides(
        PolicyConfig(c=budget_tokens, debug_invariants=spec.debug_invariants),
        **spec.config_overrides,
    )
    registry = corpus.registry()
    policy = make_policy(spec.policy, config, registry)
    metrics = MetricsAccumulator(workload, policy)
    violation: dict | None = None

    for task in workload.tasks:
        ws_start = policy.working_set()
        metrics.task_start(task, ws_start)
        try:
            for e in task.events:
                resident = policy.is_resident(e.artifact_id)
                size = registry[e.artifact_id].resolved_size(config.size_mode) \
                    if e.artifact_id in registry else 800
                metrics.observe_event(task, e, resident, size)
                policy.on_event(e)
            outcome = None
            if spec.inject_outcomes:
                outcome, out_events = outcome_events(
                    task, ws_start, spec.seed, spec.alpha_obs
                )
                for e in out_events:
                    metrics._event_task[e.event_id] = task.idx
                    policy.on_event(e)
            metrics.task_end(task, outcome)
        except InvariantViolation as exc:  # FR-R6: fail run, keep repro dump
            violation = {
                "run_id": spec.run_id(budget_tokens),
                "task_idx": task.idx,
                "config": config.as_dict(),
                "seed": spec.seed,
                **exc.dump(),
            }
            break

    summary = {
        "run_id": spec.run_id(budget_tokens),
        "policy": spec.policy,
        "family": spec.family,
        "seed": spec.seed,
        "budget": str(spec.budget),
        "budget_tokens": budget_tokens,
        "alpha_obs": spec.alpha_obs,
        "noise": {
            "drop_rate": spec.drop_rate,
            "shuffle": spec.shuffle,
            "task_id_missing_rate": spec.task_id_missing_rate,
        },
        "n_events": workload.n_events,
        "invariant_violations": 0 if violation is None else 1,
        "metrics": metrics.finalize(),
        "state_hash": policy.state_hash(),
        "policy_summary": policy.summary(),
        # ---- reproduction stamp (FR-R9) ----
        "stamp": {
            "harness_version": HARNESS_VERSION,
            "git_hash": _git_hash if _git_hash is not None else git_hash(),
            "corpus_hash": corpus.content_hash(),
            "corpus_tokens": corpus.total_tokens,
            "preload_tokens": corpus.preload_tokens,
            "config": config.as_dict(),
        },
    }
    return RunResult(
        summary=summary,
        task_rows=[r.as_dict() for r in metrics.rows],
        violation=violation,
    )


def run_batch(specs: list[RunSpec], out_dir: str | Path | None = None):
    """Multi-run batch (CLI). Shares corpus/workload per (family, seed, noise,
    budget) so paired comparisons reuse identical streams (§8.4-2)."""
    gh = git_hash()
    corpus_cache: dict[int, Corpus] = {}
    cache: dict[tuple, tuple[Corpus, Workload]] = {}
    results: list[RunResult] = []
    out_path = Path(out_dir) if out_dir else None
    if out_path:
        out_path.mkdir(parents=True, exist_ok=True)
    runs_fh = open(out_path / "runs.jsonl", "a", encoding="utf-8") if out_path else None
    try:
        for spec in specs:
            if spec.seed not in corpus_cache:
                corpus_cache[spec.seed] = generate_corpus(spec.seed)
            corpus = corpus_cache[spec.seed]
            budget_tokens = resolve_budget(spec.budget, corpus.total_tokens)
            key = (
                spec.family,
                spec.seed,
                budget_tokens,
                spec.drop_rate,
                spec.shuffle,
                spec.task_id_missing_rate,
            )
            if key not in cache:
                cache[key] = (
                    corpus,
                    generate_workload(
                        spec.family,
                        spec.seed,
                        corpus=corpus,
                        budget_tokens=budget_tokens,
                        drop_rate=spec.drop_rate,
                        shuffle=spec.shuffle,
                        task_id_missing_rate=spec.task_id_missing_rate,
                    ),
                )
            corpus, workload = cache[key]
            res = run_replay(spec, corpus=corpus, workload=workload, _git_hash=gh)
            results.append(res)
            if out_path:
                runs_fh.write(json.dumps(res.summary, ensure_ascii=False) + "\n")
                tasks_file = out_path / f"tasks.{res.summary['run_id']}.jsonl"
                with open(tasks_file, "w", encoding="utf-8") as f:
                    for row in res.task_rows:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                if res.violation is not None:
                    dump = out_path / f"violation.{res.summary['run_id']}.json"
                    dump.write_text(
                        json.dumps(res.violation, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
    finally:
        if runs_fh:
            runs_fh.close()
    return results
