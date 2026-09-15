"""LLM benchmark harness orchestration (FR-H4/H5/H6/H9).

Ties together the driver (FR-H2), materializer (FR-H1), and grader/leakage/
adoption (FR-H3/H7/H8) into a scheduled, fault-classified, reproducibly
packaged run.

- FR-H5 scheduler: the full ``arm × task × rep`` grid is interleaved and
  deterministically shuffled by ``seed`` (never one arm run to completion),
  optionally executed with ``concurrency`` worker threads (the work is
  subprocess-bound), with exponential rate-limit backoff on API errors.
- FR-H6 failure classification: ``task`` (agent ran, grading failed — never
  retried), ``api_error`` and ``harness_error`` (auto-retried up to 2 times
  with full attempt history preserved).
- FR-H2 model pin: the first observed ``reported_model`` that differs from the
  pinned model raises ``ModelMismatch`` and aborts the whole benchmark.
- FR-H4 token accounting: injected-knowledge tokens (from the materializer)
  and API tokens (cache-reflected and cache-excluded) recorded per run.
- FR-H9 packaging: run-unit JSONL + ``config.yaml`` (model_id, code_git_hash,
  seed, corpus hash) per the R-16 convention. Report generation stays with the
  experiment-execution step.
"""

from __future__ import annotations

import json
import random
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from karc.bench import grade as grade_mod
from karc.bench import materialize as mzt
from karc.bench.driver import Driver, DriverRequest, ModelMismatch
from karc.replay.runner import git_hash

RETRYABLE = ("api_error", "harness_error")


@dataclass
class Arm:
    """One knowledge-supply condition. ``mode`` is the materializer arm
    ('closed-book' | 'static-full' | 'classic' | 'karc' | 'mcp').

    ``working_set`` is a single artifact-id list used for every task.
    ``working_set_by_task`` (E2-1 c* arms) supplies a *per-task* list keyed by
    ``task_id``: the classic/K-ARC arms accumulate policy state across the task
    sequence, so task *i* gets the working set derived from tasks ``1..i-1``
    (see ``bench.working_set``). When set, it takes precedence over
    ``working_set`` for tasks present in the map (a missing task → empty set).
    """
    name: str
    mode: str
    working_set: list[str] | None = None
    working_set_by_task: dict[str, list[str]] | None = None
    managed: list[str] | None = None
    managed_by_task: dict[str, list[str]] | None = None
    mcp_condition: str | None = None
    max_turns: int = 30
    allowed_tools: list[str] | None = None
    # E3 A-E3-1: Codex has no public --max-turns flag.  Opt in to the vetted
    # per-run PreToolUse cap while preserving every earlier experiment.
    enforce_tool_cap: bool = False

    def resolve_working_set(self, task: dict) -> list[str] | None:
        """Per-task working set (E2-1) if a map is present, else the fixed one."""
        if self.working_set_by_task is not None:
            return self.working_set_by_task.get(task["task_id"], [])
        return self.working_set

    def resolve_managed(self, task: dict) -> list[str] | None:
        """Per-task managed set (E2-2: each task's required docs are what karc
        manages) if a map is present, else the fixed ``managed`` list."""
        if self.managed_by_task is not None:
            return self.managed_by_task.get(task["task_id"], [])
        return self.managed


# Closed-book (E0-3): no tools, single shot — "도구 없는 단발 호출" (§4).
CLOSED_BOOK_ARM = Arm(name="closed-book", mode="closed-book", max_turns=1,
                      allowed_tools=[])


@dataclass
class BenchSpec:
    experiment_id: str
    arms: list[Arm]
    tasks: list[dict]
    reps: int = 1
    seed: int = 0
    model: str = "claude-sonnet-4-5"
    fixture_root: Path = Path("fixture/stage2")
    concurrency: int = 1
    max_retries: int = 2
    backoff_base_s: float = 0.0  # 0 in tests; set >0 for real rate-limit backoff
    timeout_s: int = 300
    db_path: str | None = None
    runtime: str = "claude-code"
    reasoning_effort: str = "high"
    # E4 §4.3: unmatched document-content attempts may be denied before
    # exposure, while ordinary experiments preserve the historical observer.
    unresolved_document_policy: str = "observe"
    reproduction: dict = field(default_factory=dict)


@dataclass
class Attempt:
    n: int
    error_class: str | None
    error_detail: str | None
    reported_model: str | None


@dataclass
class RunOutcome:
    run_id: str
    experiment_id: str
    arm: str
    mode: str
    task_id: str
    rep: int
    passed: bool
    # "ok" | "task" | "api_error" | "harness_error" | "usage_limit"
    # usage_limit (P3): subscription-limit refusal — not a task result, not
    # retried (attempts stop at 1), and a signal to pause/stop scheduling.
    failure_class: str
    leakage: bool
    canaries_found: list[str]
    injected_knowledge_tokens: int
    api_input_tokens_no_cache: int
    api_input_tokens_with_cache: int
    api_output_tokens: int
    api_reasoning_tokens: int
    api_usage_raw: dict
    grader_detail: str
    answer_class: str
    format_ok: bool | None
    abstained: bool | None
    reported_model: str | None
    num_turns: int | None
    attempts: list[dict]
    adoption: dict | None = None
    model_verification: str | None = None
    native_session_id: str | None = None
    driver_metadata: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        return d


# --------------------------------------------------------------------------
# scheduler (FR-H5)
# --------------------------------------------------------------------------
def schedule(spec: BenchSpec) -> list[tuple[Arm, dict, int]]:
    """Build the interleaved, deterministically shuffled arm×task×rep grid."""
    units: list[tuple[Arm, dict, int]] = []
    for arm in spec.arms:
        for task in spec.tasks:
            for rep in range(spec.reps):
                units.append((arm, task, rep))
    random.Random(spec.seed).shuffle(units)
    return units


def _run_id(spec: BenchSpec, arm: Arm, task: dict, rep: int) -> str:
    return f"{spec.experiment_id}.{arm.name}.{task['task_id']}.r{rep}"


# --------------------------------------------------------------------------
# one unit (FR-H1 → H2 → H3/H4/H6/H7/H8)
# --------------------------------------------------------------------------
class Harness:
    def __init__(self, spec: BenchSpec, driver: Driver, manifest: dict,
                 temp_root: str | Path | None = None):
        self.spec = spec
        self.driver = driver
        self.manifest = manifest
        self.canaries = grade_mod.all_canaries(manifest)
        self._temp_root = Path(temp_root) if temp_root else None
        self._model_lock = threading.Lock()
        self._verified_model: str | None = None

    def _verify_model(self, reported: str | None) -> None:
        """FR-H2: abort the benchmark if a reported model differs from the pin.
        ``None`` reported (e.g. mock without a model) is tolerated."""
        if reported is None:
            return
        pin = self.spec.model
        if reported == pin or reported.startswith(pin) or pin.startswith(reported):
            with self._model_lock:
                self._verified_model = reported
            return
        raise ModelMismatch(f"reported model {reported!r} != pinned {pin!r}")

    def run_unit(self, arm: Arm, task: dict, rep: int) -> RunOutcome:
        started = time.perf_counter()
        run_id = _run_id(self.spec, arm, task, rep)
        attempts: list[Attempt] = []
        last = None
        for n in range(self.spec.max_retries + 1):
            _err, result, materialized = self._attempt(arm, task)
            reported = result.reported_model if result else None
            attempts.append(Attempt(
                n=n,
                error_class=(result.error_class if result else "harness_error"),
                error_detail=(result.error_detail if result else _err),
                reported_model=reported,
            ))
            self._verify_model(reported)
            retryable = result is None or result.error_class in RETRYABLE
            if not retryable:
                last = (result, materialized)
                break
            # discard this attempt's temp tree before retrying/giving up
            if last is not None and last[1] is not None:
                self._cleanup(getattr(last[1], "_tmp", None))
            last = (result, materialized)
            if n < self.spec.max_retries and self.spec.backoff_base_s > 0:
                time.sleep(self.spec.backoff_base_s * (2 ** n))

        result, materialized = last
        outcome = self._finalize(
            run_id, arm, task, rep, result, materialized, attempts,
        )
        outcome.driver_metadata = dict(outcome.driver_metadata or {})
        outcome.driver_metadata["run_wall_time_s"] = time.perf_counter() - started
        return outcome

    def _attempt(self, arm: Arm, task: dict):
        """Materialize (FR-H1) → drive (FR-H2). Returns (err_detail|None,
        DriverResult|None, Materialized|None). Materialization/IO faults are
        classified harness_error."""
        tmp = tempfile.mkdtemp(prefix="karc-bench-", dir=self._temp_root)
        workdir = Path(tmp) / "run"
        try:
            m = mzt.materialize(
                arm.mode, workdir, fixture_root=Path(self.spec.fixture_root),
                manifest=self.manifest, working_set=arm.resolve_working_set(task),
                managed=arm.resolve_managed(task), mcp_condition=arm.mcp_condition,
                db_path=self.spec.db_path, runtime=self.spec.runtime,
            )
        except Exception as exc:  # materialization is our code → harness_error
            self._cleanup(tmp)
            from karc.bench.driver import DriverResult
            return None, DriverResult(ok=False, error_class="harness_error",
                                      error_detail=f"materialize: {exc}"), None
        environment = ({"KARC_CODEX_MAX_TOOL_CALLS": str(arm.max_turns)}
                       if arm.enforce_tool_cap else {})
        fixture_manifest = Path(self.spec.fixture_root).resolve() / "manifest.json"
        if fixture_manifest.is_file():
            environment["KARC_CODEX_FIXTURE_MANIFEST"] = str(fixture_manifest)
            environment["KARC_CODEX_INJECTED_ARTIFACTS"] = json.dumps(
                m.injected_artifacts, separators=(",", ":")
            )
            environment["KARC_CODEX_UNRESOLVED_POLICY"] = (
                self.spec.unresolved_document_policy
            )
        req = DriverRequest(
            prompt=task["prompt"], cwd=str(m.workdir), model=self.spec.model,
            max_turns=arm.max_turns, allowed_tools=arm.allowed_tools,
            mcp_config=m.mcp_config, settings=m.settings, timeout_s=self.spec.timeout_s,
            runtime=self.spec.runtime,
            config_overrides=tuple(getattr(m, "codex_config_overrides", ())),
            hook_mode=getattr(m, "hook_mode", "observe"),
            reasoning_effort=self.spec.reasoning_effort,
            environment=environment,
        )
        try:
            result = self.driver.run(req)
        except Exception as exc:  # driver-layer fault outside its own handling
            from karc.bench.driver import DriverResult
            result = DriverResult(ok=False, error_class="harness_error",
                                  error_detail=f"driver: {exc}")
        finally:
            # grading needs the output text only; temp tree can go now unless a
            # test-command grader needs cwd. Keep it until finalize for safety.
            pass
        m._tmp = tmp  # stash for cleanup after finalize
        return None, result, m

    def _finalize(self, run_id, arm, task, rep, result, materialized, attempts):
        tmp = getattr(materialized, "_tmp", None) if materialized else None
        try:
            injected = materialized.injected_knowledge_tokens if materialized else 0
            if result is None:
                return self._error_outcome(run_id, arm, task, rep,
                                           "harness_error", injected, attempts)
            if result.error_class:  # retryables exhausted OR usage_limit (P3)
                return self._error_outcome(run_id, arm, task, rep,
                                           result.error_class, injected, attempts,
                                           result=result)
            # clean completion → grade + leakage (+ adoption for mcp)
            cwd = str(materialized.workdir) if materialized else None
            if task.get("answer_fact") and task.get("format_regex"):
                # E4 primary success is value correctness; exact-format
                # compliance is an independent secondary field (§3.5).
                from karc.bench.e4_fixture import grade_answer

                e4_grade = grade_answer(task, result.output_text)
                g = grade_mod.GradeResult(e4_grade.success, "e4-value")
                answer_class = (
                    "correct" if e4_grade.value_match == "expected" else
                    "stale" if e4_grade.value_match == "stale" else
                    "abstain" if e4_grade.abstained else "other-wrong"
                )
                format_ok = e4_grade.format_ok
                abstained = e4_grade.abstained
            else:
                g = grade_mod.grade(task["expected_answer"], result.output_text, cwd=cwd)
                answer_class = grade_mod.classify_answer(task, result.output_text)
                format_ok = None
                abstained = None
            found = grade_mod.detect_leakage(result.output_text, result.transcript,
                                             self.canaries)
            adoption = None
            if arm.mode == "mcp":
                adoption = grade_mod.mcp_adoption(
                    result.transcript, materialized.managed_prefixes).as_dict()
            return RunOutcome(
                run_id=run_id, experiment_id=self.spec.experiment_id,
                arm=arm.name, mode=arm.mode, task_id=task["task_id"], rep=rep,
                passed=g.passed, failure_class=("task" if not g.passed else "ok"),
                leakage=bool(found), canaries_found=found,
                injected_knowledge_tokens=injected,
                api_input_tokens_no_cache=result.total_input_tokens(with_cache=False),
                api_input_tokens_with_cache=result.total_input_tokens(with_cache=True),
                api_output_tokens=result.output_tokens,
                api_reasoning_tokens=result.reasoning_tokens,
                api_usage_raw=result.usage_raw or result.usage,
                grader_detail=g.detail, reported_model=result.reported_model,
                answer_class=answer_class, format_ok=format_ok,
                abstained=abstained,
                num_turns=result.num_turns,
                attempts=[a.__dict__ for a in attempts], adoption=adoption,
                model_verification=result.model_verification,
                native_session_id=result.native_session_id,
                driver_metadata=result.driver_metadata,
            )
        finally:
            if tmp:
                self._cleanup(tmp)

    def _error_outcome(self, run_id, arm, task, rep, cls, injected, attempts,
                       result=None):
        return RunOutcome(
            run_id=run_id, experiment_id=self.spec.experiment_id, arm=arm.name,
            mode=arm.mode, task_id=task["task_id"], rep=rep, passed=False,
            failure_class=cls, leakage=False, canaries_found=[],
            injected_knowledge_tokens=injected,
            api_input_tokens_no_cache=(result.total_input_tokens(with_cache=False)
                                       if result else 0),
            api_input_tokens_with_cache=(result.total_input_tokens(with_cache=True)
                                         if result else 0),
            api_output_tokens=(result.output_tokens if result else 0),
            api_reasoning_tokens=(result.reasoning_tokens if result else 0),
            api_usage_raw=((result.usage_raw or result.usage) if result else {}),
            grader_detail="", answer_class="other-wrong", format_ok=None,
            abstained=None,
            reported_model=(result.reported_model if result else None),
            num_turns=None, attempts=[a.__dict__ for a in attempts], adoption=None,
            model_verification=(result.model_verification if result else None),
            native_session_id=(result.native_session_id if result else None),
            driver_metadata=(result.driver_metadata if result else {}),
        )

    @staticmethod
    def _cleanup(tmp: str | None) -> None:
        if not tmp:
            return
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)

    # -- top-level run ------------------------------------------------------
    def run(self, out_dir: str | Path | None = None,
            preflight: dict | None = None,
            stop_on_usage_limit: bool = True) -> list[RunOutcome]:
        """Run the scheduled grid. P3: a ``usage_limit`` outcome stops further
        scheduling immediately (in-flight units still finish) — retrying into a
        closed limit window only destroys sample (E2-2 §4.2). Un-run units stay
        pending for a checkpoint resume (``bench.checkpoint``)."""
        units = schedule(self.spec)
        outcomes: list[RunOutcome] = []
        stop = False
        if self.spec.concurrency <= 1:
            for arm, task, rep in units:
                if stop:
                    break
                o = self.run_unit(arm, task, rep)
                outcomes.append(o)
                if stop_on_usage_limit and o.failure_class == "usage_limit":
                    stop = True
        else:
            from concurrent.futures import FIRST_COMPLETED, wait
            it = iter(units)
            with ThreadPoolExecutor(max_workers=self.spec.concurrency) as ex:
                futs = set()

                def submit_next() -> bool:
                    try:
                        arm, task, rep = next(it)
                    except StopIteration:
                        return False
                    futs.add(ex.submit(self.run_unit, arm, task, rep))
                    return True

                for _ in range(min(self.spec.concurrency, len(units))):
                    submit_next()
                while futs:
                    done, futs = wait(futs, return_when=FIRST_COMPLETED)
                    for f in done:
                        o = f.result()
                        outcomes.append(o)
                        if stop_on_usage_limit and o.failure_class == "usage_limit":
                            stop = True
                    if not stop:
                        while len(futs) < self.spec.concurrency and submit_next():
                            pass
        # stable order for reproducible packaging
        outcomes.sort(key=lambda o: o.run_id)
        if out_dir:
            self.package(outcomes, out_dir, preflight=preflight)
        return outcomes

    # -- FR-H9 packaging ----------------------------------------------------
    def package(self, outcomes: list[RunOutcome], out_dir: str | Path,
                preflight: dict | None = None) -> None:
        out = Path(out_dir)
        (out / "raw").mkdir(parents=True, exist_ok=True)
        with open(out / "raw" / "runs.jsonl", "w", encoding="utf-8") as f:
            for o in outcomes:
                f.write(json.dumps(o.as_dict(), ensure_ascii=False, sort_keys=True) + "\n")
        write_config_yaml(out / "config.yaml", self.spec, self.manifest,
                          verified_model=self._verified_model, preflight=preflight)


# --------------------------------------------------------------------------
# config.yaml (R-16, stdlib-only emitter)
# --------------------------------------------------------------------------
def _yaml(obj, indent: int = 0) -> list[str]:
    pad = "  " * indent
    lines: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)) and v:
                lines.append(f"{pad}{k}:")
                lines += _yaml(v, indent + 1)
            else:
                lines.append(f"{pad}{k}: {_scalar(v)}")
    elif isinstance(obj, list):
        for v in obj:
            if isinstance(v, (dict, list)) and v:
                lines.append(f"{pad}-")
                lines += _yaml(v, indent + 1)
            else:
                lines.append(f"{pad}- {_scalar(v)}")
    return lines


def _scalar(v) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return json.dumps(v)
    return json.dumps(str(v), ensure_ascii=False)


def write_config_yaml(path: Path, spec: BenchSpec, manifest: dict,
                      verified_model: str | None = None,
                      preflight: dict | None = None) -> None:
    cfg = {
        "experiment_id": spec.experiment_id,
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "code_git_hash": git_hash(Path(__file__).resolve().parents[3]),
        "corpus_content_hash": manifest.get("corpus_content_hash"),
        "manifest_sha256": manifest.get("manifest_sha256"),
        "model_id": spec.model,
        "model_id_verified": verified_model,
        "runtime": spec.runtime,
        "reasoning_effort": spec.reasoning_effort,
        "seeds": {"schedule_seed": spec.seed},
        "parameters": {
            "reps": spec.reps,
            "concurrency": spec.concurrency,
            "max_retries": spec.max_retries,
            "fixture_root": str(spec.fixture_root),
            "arms": [{"name": a.name, "mode": a.mode,
                      "mcp_condition": a.mcp_condition,
                      "max_turns": a.max_turns,
                      "enforce_tool_cap": a.enforce_tool_cap} for a in spec.arms],
            "n_tasks": len(spec.tasks),
        },
    }
    if preflight:
        cfg["cli_preflight"] = preflight
    if spec.reproduction:
        cfg["reproduction"] = spec.reproduction
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(_yaml(cfg)) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# E0-3 verdict logic (FR-H8) — prepared, NOT executed here
# --------------------------------------------------------------------------
def e0_3_verdict(outcomes: list[RunOutcome], reps: int) -> dict:
    """Apply the pre-registered E0-3 judgment (§4 E0-3, unchanged):

      - per task: closed-book success rate ≤ 5% over its reps; with 3 reps a
        single success (1/3 ≈ 33%) trips it, so a task passes the gate only if
        it succeeds in ZERO reps. Any leakage (canary in output) also fails it.
      - overall: mean closed-book success rate ≤ 2%.

    Returns the verdict object; the caller writes report.md. This runs on
    already-collected outcomes and performs no LLM call."""
    by_task: dict[str, list[RunOutcome]] = {}
    for o in outcomes:
        by_task.setdefault(o.task_id, []).append(o)
    per_task = {}
    n_success_runs = 0
    n_runs = 0
    flagged = []
    leaked = []
    for tid, runs in by_task.items():
        succ = sum(1 for r in runs if r.passed)
        leak = any(r.leakage for r in runs)
        n_success_runs += succ
        n_runs += len(runs)
        rate = succ / len(runs) if runs else 0.0
        per_task[tid] = {"reps": len(runs), "successes": succ,
                         "success_rate": rate, "leakage": leak}
        if succ > 0 or leak:  # ≤5% with ≤3 reps ⇒ must be 0 successes
            flagged.append(tid)
        if leak:
            leaked.append(tid)
    overall = n_success_runs / n_runs if n_runs else 0.0
    return {
        "overall_success_rate": overall,
        "overall_pass": overall <= 0.02 and not flagged,
        "threshold_overall": 0.02,
        "threshold_per_task": 0.05,
        "flagged_tasks": sorted(flagged),
        "leaked_tasks": sorted(leaked),
        "per_task": per_task,
        "n_runs": n_runs,
    }
