"""E5-G1 Codex-runtime cache-asymmetry canary.

The build path is deterministic and model-free.  Model execution is isolated
in :mod:`scripts.run_e5_g1` and uses the persistent-session primitive from the
provider-neutral driver.  This module intentionally persists only normalized
usage, identifiers, grades, and hashes; prompts, outputs, transcripts, tool
responses, commands, and credentials never enter experiment raw data.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

from karc.bench import materialize as mzt
from karc.bench import mcp_index
from karc.bench.e3_retrieval import bm25_plans, chunk_fixture
from karc.bench.e4_fixture import grade_answer
from karc.bench.e4_replay import load_confirmed_config, replay_cell
from karc.bench.e5_killgate import (
    ENGINE_RHO_LABEL,
    ENGINE_SIGMA_LABEL,
    _manifest_at_budget,
    build_reuse_schedule,
)


SCHEMA = "e5-g1-cache-asymmetry-v1"
CORPUS_SEED = 4000
SESSION_LENGTH = 8
REUSE_FACTOR = 0.5
BUDGET_PCT = 5
SMOKE_SCHEDULE_SEED = 4100
CANARY_SCHEDULE_SEED = 4200
SMOKE_SESSIONS = 4
CANARY_SESSION_CHOICES = (8, 12)
ARMS = ("karc-full", "rag-bm25")
MODEL = "gpt-5.6-luna"
REASONING_EFFORT = "high"
BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 5101
ONE_SIDED_ALPHA = 0.05
POWER = 0.80
Z_ALPHA = 1.6448536269514722
Z_POWER = 0.8416212335729143
MARGIN_CANDIDATES = (0.05, 0.10)
MAX_TURN_RETRIES = 2
SMOKE_ATTEMPT_CAP = 80


KARC_SESSION_INSTRUCTION = (
    "For each benchmark task, first use the injected resident knowledge and "
    "knowledge already present in this conversation. If the required fact is "
    "not available, use only the `karc` MCP server's `search` and `get` tools "
    "to retrieve K-ARC-managed knowledge. Never use shell, file-read, grep, or "
    "other generic tools to access `.karc/managed/`. Return only the exact "
    "assignment requested by the task."
)
RAG_SESSION_INSTRUCTION = (
    "For each benchmark task, answer from the BM25 evidence supplied in that "
    "task message. The evidence is a deterministic budget-parity retrieval "
    "plan in BM25 rank order. Do not use local tools. Return only the exact "
    "assignment requested by the task."
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode()


def hash_json(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_bundle(
    repo_root: str | Path, fixture_root: str | Path, *,
    schedule_seed: int, session_count: int,
    session_length: int = SESSION_LENGTH,
    allow_unregistered_session_length: bool = False,
) -> dict:
    """Create the r=.5/c=5% session schedule, replay, and BM25 plans."""
    repo_root = Path(repo_root)
    fixture_root = Path(fixture_root)
    schedule, schedule_manifest, tasks = build_reuse_schedule(
        repo_root, schedule_seed=schedule_seed,
        session_length=session_length, reuse_factor=REUSE_FACTOR,
        session_count=session_count, allow_unregistered_seed=True,
        allow_unregistered_session_length=allow_unregistered_session_length,
    )
    manifest = _manifest_at_budget(schedule_manifest, BUDGET_PCT)
    replay_summary, replay_rows = replay_cell(
        rho=ENGINE_RHO_LABEL, sigma=ENGINE_SIGMA_LABEL,
        budget_pct=BUDGET_PCT,
        confirmed_config=load_confirmed_config(repo_root),
        fixture_manifest=manifest, fixture_tasks=tasks,
    )
    chunks = chunk_fixture(fixture_root, manifest)
    plans = bm25_plans(
        chunks, tasks, manifest, int(manifest["cell"]["budget_tokens"]),
    )
    replay_by_task = {row["task_id"]: row for row in replay_rows}
    sessions: list[dict] = []
    by_session: dict[str, list[dict]] = defaultdict(list)
    for task in tasks:
        by_session[task["session_id"]].append(task)
    for session_id, session_tasks in by_session.items():
        session_tasks.sort(key=lambda row: int(row["session_task"]))
        first = replay_by_task[session_tasks[0]["task_id"]]
        initial = list(first["arms"]["karc"]["working_set_versions"])
        rows = []
        for task in session_tasks:
            replay = replay_by_task[task["task_id"]]
            resident = list(replay["arms"]["karc"]["working_set_versions"])
            required = set(task["required_versions"])
            plan = plans[task["task_id"]]
            if not required <= set(plan["artifact_ids"]):
                raise AssertionError("BM25 budget-parity plan omitted gold evidence")
            rows.append({
                "task": task,
                "policy_resident_versions": resident,
                "policy_resident_hit": required <= set(resident),
                "initial_prefix_hit": required <= set(initial),
                "rag_plan": plan,
            })
        sessions.append({
            "schedule_seed": schedule_seed,
            "session_id": session_id,
            "initial_resident_versions": initial,
            "tasks": rows,
        })
    if len(sessions) != session_count:
        raise AssertionError("session grouping changed cardinality")
    return {
        "schema": SCHEMA,
        "schedule": schedule,
        "manifest": manifest,
        "tasks": tasks,
        "sessions": sessions,
        "replay_summary": replay_summary,
        "plans": plans,
        "retrieval_audit": {
            "gold_containment_rate": sum(
                set(task["required_versions"]) <= set(plans[task["task_id"]]["artifact_ids"])
                for task in tasks
            ) / len(tasks),
            "mean_budget_utilization": statistics.mean(
                float(plans[task["task_id"]]["budget_utilization"])
                for task in tasks
            ),
            "rank_order_preserved": True,
            "plan_tokens_mean": statistics.mean(
                int(plans[task["task_id"]]["tokens"]) for task in tasks
            ),
        },
    }


def bundle_snapshot(bundle: dict) -> dict:
    """Content-free schedule/supply snapshot suitable for committed raw."""
    manifest = bundle["manifest"]
    sessions = []
    for session in bundle["sessions"]:
        sessions.append({
            "schedule_seed": session["schedule_seed"],
            "session_id": session["session_id"],
            "initial_resident_versions": session["initial_resident_versions"],
            "initial_resident_tokens": sum(
                int(manifest["artifacts"][v]["size_tok"])
                for v in session["initial_resident_versions"]
            ),
            "tasks": [{
                "task_id": item["task"]["task_id"],
                "position": item["task"]["session_task"],
                "required_versions": item["task"]["required_versions"],
                "reuse": bool(item["task"]["e5_resident_reuse"]),
                "policy_resident_hit": item["policy_resident_hit"],
                "initial_prefix_hit": item["initial_prefix_hit"],
                "rag_artifact_ids": item["rag_plan"]["artifact_ids"],
                "rag_tokens": item["rag_plan"]["tokens"],
            } for item in session["tasks"]],
        })
    value = {
        "schema": SCHEMA,
        "schedule": {
            key: bundle["schedule"][key] for key in (
                "schedule_seed", "sessions", "session_length",
                "reuse_factor_requested", "reuse_factor_measured",
                "reuse_count", "reuse_denominator_eligible_followups",
                "corpus_content_hash", "source_e4_manifest_sha256",
                "manifest_sha256", "tasks_sha256", "sha256",
            )
        },
        "cell": manifest["cell"],
        "schedule_manifest_sha256": manifest["manifest_sha256"],
        "retrieval_audit": bundle["retrieval_audit"],
        "sessions": sessions,
    }
    value["sha256"] = hash_json(value)
    return value


def _init_git(workdir: Path) -> None:
    proc = subprocess.run(
        ["git", "init", "--quiet"], cwd=workdir,
        capture_output=True, text=True, timeout=30,
    )
    if proc.returncode:
        raise RuntimeError("could not initialize isolated session repository")


def materialize_session(
    arm: str, workdir: str | Path, *, fixture_root: str | Path,
    manifest: dict, initial_resident_versions: Sequence[str],
) -> dict:
    """Materialize one persistent conversation without invoking a model."""
    workdir = Path(workdir)
    fixture_root = Path(fixture_root)
    workdir.mkdir(parents=True, exist_ok=True)
    _init_git(workdir)
    if arm == "rag-bm25":
        sha, nbytes = mzt._codex_agents(
            workdir, manifest, [], instruction=RAG_SESSION_INSTRUCTION,
        )
        return {
            "workdir": workdir, "hook_mode": "deny-all",
            "config_overrides": (f"project_doc_max_bytes={nbytes + 4096}",),
            "agents_sha256": sha, "agents_bytes": nbytes,
            "managed_versions": [], "initial_resident_versions": [],
        }
    if arm != "karc-full":
        raise ValueError(f"unknown E5-G1 arm {arm!r}")

    shutil.copytree(fixture_root / "repo", workdir, dirs_exist_ok=True)
    resident = [v for v in initial_resident_versions if v in manifest["artifacts"]]
    sha, nbytes = mzt._codex_agents(
        workdir, manifest, resident, instruction=KARC_SESSION_INSTRUCTION,
    )
    managed = sorted(
        manifest["artifacts"], key=lambda v: manifest["artifacts"][v]["path"],
    )
    managed_dir = workdir / mzt.MANAGED_ROOT
    for version_id in managed:
        entry = manifest["artifacts"][version_id]
        source = workdir / entry["path"]
        target = managed_dir / entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.exists():
            shutil.move(str(source), str(target))
    index_db = workdir / ".karc" / "index.db"
    mcp_index.build_index(
        workdir, manifest, managed,
        managed_root=mzt.MANAGED_ROOT, db_path=index_db,
    )
    overrides = [*mzt._codex_mcp_overrides(workdir, index_db),
                 f"project_doc_max_bytes={nbytes + 4096}"]
    return {
        "workdir": workdir, "hook_mode": "deny-managed",
        "config_overrides": tuple(overrides),
        "agents_sha256": sha, "agents_bytes": nbytes,
        "managed_versions": managed,
        "initial_resident_versions": resident,
    }


def task_prompt(arm: str, item: dict, fixture_root: str | Path,
                manifest: dict) -> str:
    task = item["task"]
    header = (
        f"Benchmark task {task['session_task']}/{SESSION_LENGTH}. "
        f"{task['prompt']} Return only the requested assignment."
    )
    if arm == "karc-full":
        return (
            header + " Use resident or prior-session-turn knowledge when it "
            "contains the fact; otherwise use only K-ARC MCP search/get."
        )
    if arm != "rag-bm25":
        raise ValueError(arm)
    fixture_root = Path(fixture_root)
    blocks = [header, "", "## BM25 evidence (rank order)", ""]
    for rank, version_id in enumerate(item["rag_plan"]["artifact_ids"], 1):
        entry = manifest["artifacts"][version_id]
        body = (fixture_root / "repo" / entry["path"]).read_text(encoding="utf-8")
        blocks += [
            f"### rank={rank} id={version_id} path={entry['path']}",
            body.rstrip("\n"), "",
        ]
    return "\n".join(blocks).rstrip() + "\n"


def grade_turn(task: dict, output_text: str) -> dict:
    grade = grade_answer(task, output_text)
    return {
        "passed": grade.success,
        "value_match": grade.value_match,
        "format_ok": grade.format_ok,
        "abstained": grade.abstained,
    }


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("quantile needs observations")
    ordered = sorted(float(value) for value in values)
    position = q * (len(ordered) - 1)
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    if lo == hi:
        return ordered[lo]
    weight = position - lo
    return ordered[lo] * (1 - weight) + ordered[hi] * weight


def bootstrap_lower(
    values: Sequence[float], *, reps: int = BOOTSTRAP_REPS,
    seed: int = BOOTSTRAP_SEED, alpha: float = ONE_SIDED_ALPHA,
) -> float:
    if not values:
        raise ValueError("bootstrap needs observations")
    values = [float(value) for value in values]
    rng = random.Random(seed)
    means = []
    for _ in range(reps):
        means.append(statistics.mean(
            values[rng.randrange(len(values))] for _ in values
        ))
    return _quantile(means, alpha)


def _latest_complete(rows: Sequence[dict]) -> list[dict]:
    grouped: dict[tuple[str, int, str, int], list[dict]] = defaultdict(list)
    for row in rows:
        key = (
            row["arm"], int(row["schedule_seed"]), row["session_id"],
            int(row.get("session_attempt", 0)),
        )
        grouped[key].append(row)
    complete: dict[tuple[str, int, str], tuple[int, list[dict]]] = {}
    for (arm, seed, sid, attempt), unit in grouped.items():
        successful = [row for row in unit if row["failure_class"] in {"ok", "task"}]
        positions = {int(row["position"]) for row in successful}
        if len(successful) != SESSION_LENGTH or positions != set(range(1, 9)):
            continue
        key = (arm, seed, sid)
        if key not in complete or attempt > complete[key][0]:
            complete[key] = (attempt, successful)
    out = [row for _attempt, unit in complete.values() for row in unit]
    return sorted(out, key=lambda row: (
        row["schedule_seed"], row["session_id"], row["arm"], row["position"],
    ))


def summarize(rows: Sequence[dict], expected_sessions: int) -> dict:
    valid = _latest_complete(rows)
    by_session_arm: dict[tuple[int, str, str], list[dict]] = defaultdict(list)
    for row in valid:
        by_session_arm[(int(row["schedule_seed"]), row["session_id"], row["arm"])].append(row)
    session_rows = []
    for (seed, sid, arm), unit in sorted(by_session_arm.items()):
        reuse = [row for row in unit if row["reuse"]]
        session_rows.append({
            "schedule_seed": seed, "session_id": sid, "arm": arm,
            "turns": len(unit),
            "fresh_per_task": statistics.mean(row["api_input_tokens_no_cache"] for row in unit),
            "gross_per_task": statistics.mean(row["api_input_tokens_with_cache"] for row in unit),
            "cache_read_per_task": statistics.mean(row["api_cache_read_tokens"] for row in unit),
            "reuse_fresh_per_task": statistics.mean(
                row["api_input_tokens_no_cache"] for row in reuse
            ),
            "reuse_turns": len(reuse),
        })
    index = {(row["schedule_seed"], row["session_id"], row["arm"]): row
             for row in session_rows}
    pairs = []
    session_keys = sorted({(row["schedule_seed"], row["session_id"])
                           for row in session_rows})
    for seed, sid in session_keys:
        karc = index.get((seed, sid, "karc-full"))
        rag = index.get((seed, sid, "rag-bm25"))
        if not karc or not rag:
            continue
        pairs.append({
            "schedule_seed": seed, "session_id": sid,
            "reuse_fresh_saving": 1 - karc["reuse_fresh_per_task"] / rag["reuse_fresh_per_task"],
            "amortized_fresh_saving": 1 - karc["fresh_per_task"] / rag["fresh_per_task"],
            "karc_reuse_fresh": karc["reuse_fresh_per_task"],
            "rag_reuse_fresh": rag["reuse_fresh_per_task"],
            "karc_amortized_fresh": karc["fresh_per_task"],
            "rag_amortized_fresh": rag["fresh_per_task"],
        })
    reuse_values = [row["reuse_fresh_saving"] for row in pairs]
    amortized_values = [row["amortized_fresh_saving"] for row in pairs]
    position: dict[str, dict[str, dict]] = {}
    for arm in ARMS:
        position[arm] = {}
        for pos in range(1, SESSION_LENGTH + 1):
            unit = [row for row in valid if row["arm"] == arm and row["position"] == pos]
            position[arm][str(pos)] = {
                "turns": len(unit),
                "fresh_mean": statistics.mean(row["api_input_tokens_no_cache"] for row in unit) if unit else None,
                "gross_mean": statistics.mean(row["api_input_tokens_with_cache"] for row in unit) if unit else None,
                "cache_read_mean": statistics.mean(row["api_cache_read_tokens"] for row in unit) if unit else None,
                "cache_hit_rate": (
                    sum(row["api_cache_read_tokens"] for row in unit)
                    / sum(row["api_input_tokens_with_cache"] for row in unit)
                    if unit and sum(row["api_input_tokens_with_cache"] for row in unit) else 0
                ),
            }
    arm_summary = {}
    for arm in ARMS:
        unit = [row for row in valid if row["arm"] == arm]
        gross = sum(row["api_input_tokens_with_cache"] for row in unit)
        cached = sum(row["api_cache_read_tokens"] for row in unit)
        arm_summary[arm] = {
            "turns": len(unit),
            "fresh_per_task": statistics.mean(row["api_input_tokens_no_cache"] for row in unit) if unit else None,
            "gross_per_task": statistics.mean(row["api_input_tokens_with_cache"] for row in unit) if unit else None,
            "cache_read_per_task": statistics.mean(row["api_cache_read_tokens"] for row in unit) if unit else None,
            "cache_hit_rate": cached / gross if gross else 0,
            "correct_rate": sum(row["passed"] for row in unit) / len(unit) if unit else None,
        }
    karc_rows = [row for row in valid if row["arm"] == "karc-full"]
    policy_misses = [row for row in karc_rows if not row["policy_resident_hit"]]
    native_attempts = sum(row["native_managed_attempts"] for row in karc_rows)
    mcp_calls = sum(row["mcp_search_calls"] + row["mcp_get_calls"] for row in karc_rows)
    operational = {
        "policy_resident_hit_rate": (
            sum(row["policy_resident_hit"] for row in karc_rows) / len(karc_rows)
            if karc_rows else None
        ),
        "initial_prefix_hit_rate": (
            sum(row["initial_prefix_hit"] for row in karc_rows) / len(karc_rows)
            if karc_rows else None
        ),
        "mcp_get_on_policy_miss_rate": (
            sum(row["mcp_get_calls"] > 0 for row in policy_misses) / len(policy_misses)
            if policy_misses else None
        ),
        "mcp_task_adoption_rate": (
            sum((row["mcp_search_calls"] + row["mcp_get_calls"]) > 0 for row in karc_rows)
            / len(karc_rows) if karc_rows else None
        ),
        "mcp_channel_adoption_rate": (
            mcp_calls / (mcp_calls + native_attempts)
            if mcp_calls + native_attempts else None
        ),
        "mcp_search_calls": sum(row["mcp_search_calls"] for row in karc_rows),
        "mcp_get_calls": sum(row["mcp_get_calls"] for row in karc_rows),
        "native_managed_attempts": native_attempts,
        "native_managed_denied": sum(row["native_managed_denied"] for row in karc_rows),
    }
    return {
        "schema": SCHEMA,
        "expected_sessions": expected_sessions,
        "complete_paired_sessions": len(pairs),
        "complete": len(pairs) == expected_sessions,
        "valid_turns": len(valid),
        "expected_turns": expected_sessions * len(ARMS) * SESSION_LENGTH,
        "session_arm": session_rows,
        "paired_session_metrics": pairs,
        "reuse_fresh_saving": {
            "point": statistics.mean(reuse_values) if reuse_values else None,
            "ci_lower_one_sided_95": bootstrap_lower(reuse_values) if reuse_values else None,
            "sample_sd": statistics.stdev(reuse_values) if len(reuse_values) > 1 else 0,
        },
        "amortized_fresh_saving": {
            "point": statistics.mean(amortized_values) if amortized_values else None,
            "ci_lower_one_sided_95": bootstrap_lower(
                amortized_values, seed=BOOTSTRAP_SEED + 1,
            ) if amortized_values else None,
            "sample_sd": statistics.stdev(amortized_values) if len(amortized_values) > 1 else 0,
        },
        "by_arm": arm_summary,
        "by_position": position,
        "karc_operations": operational,
        "rag_cache_credit_observed": bool(
            arm_summary.get("rag-bm25", {}).get("cache_read_per_task")
        ),
    }


def choose_canary_freeze(smoke: dict, *, code_git_hash: str,
                         source_sha256: dict[str, str]) -> dict:
    if not smoke.get("complete"):
        raise ValueError("cannot freeze canary from incomplete smoke")
    reuse_point = float(smoke["reuse_fresh_saving"]["point"])
    margin = MARGIN_CANDIDATES[0]
    for candidate in MARGIN_CANDIDATES:
        if reuse_point >= 2 * candidate:
            margin = candidate
    amortized_point = float(smoke["amortized_fresh_saving"]["point"])

    def required(sd: float, distance: float) -> int:
        distance = max(abs(distance), 0.025)
        return math.ceil(((Z_ALPHA + Z_POWER) * sd / distance) ** 2)

    n_reuse = required(
        float(smoke["reuse_fresh_saving"]["sample_sd"]),
        reuse_point - margin,
    )
    n_amortized = required(
        float(smoke["amortized_fresh_saving"]["sample_sd"]),
        amortized_point,
    )
    uncapped = max(n_reuse, n_amortized, min(CANARY_SESSION_CHOICES))
    choices = [value for value in CANARY_SESSION_CHOICES if value >= uncapped]
    sessions = choices[0] if choices else max(CANARY_SESSION_CHOICES)
    planned = sessions * len(ARMS) * SESSION_LENGTH
    attempt_cap = planned + max(16, planned // 4)
    value = {
        "schema": "e5-g1-canary-freeze-v1",
        "authority": "docs/analysis/experiment-design--e5-fair-comparison.md §3b",
        "created_after_smoke_before_canary": True,
        "cell": {"reuse_factor": REUSE_FACTOR, "session_length": SESSION_LENGTH,
                 "budget_pct": BUDGET_PCT},
        "model": MODEL, "reasoning_effort": REASONING_EFFORT,
        "runtime": "Codex CLI",
        "schedule_seed": CANARY_SCHEDULE_SEED,
        "session_count": sessions,
        "session_count_choices_pre_smoke": list(CANARY_SESSION_CHOICES),
        "right_size": {
            "method": "paired-session normal approximation, one-sided alpha=.05, power=.80",
            "n_reuse": n_reuse, "n_amortized": n_amortized,
            "uncapped_max": uncapped, "selected": sessions,
            "cap": max(CANARY_SESSION_CHOICES),
            "cap_binding": uncapped > max(CANARY_SESSION_CHOICES),
        },
        "reuse_margin": margin,
        "margin_selection_rule": (
            "largest of {5%,10%} no greater than half the smoke point saving; "
            "fallback 5%"
        ),
        "decision_rule": {
            "ci": "paired-session bootstrap 10,000, deterministic one-sided 95% lower",
            "CONFIRM": (
                "reuse fresh-saving CI lower >= margin AND amortized fresh-saving "
                "CI lower >= 0"
            ),
            "REFUTE": "CONFIRM conjunction not met",
        },
        "planned_turns": planned,
        "attempt_cap": attempt_cap,
        "max_turn_retries": MAX_TURN_RETRIES,
        "bootstrap_reps": BOOTSTRAP_REPS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "smoke_summary_sha256": hash_json(smoke),
        "execution_code_git_hash": code_git_hash,
        "source_sha256": source_sha256,
    }
    value["sha256"] = hash_json(value)
    return value


def canary_verdict(summary: dict, freeze: dict, retrieval_audit: dict) -> dict:
    complete = bool(summary.get("complete"))
    margin = float(freeze["reuse_margin"])
    reuse_lower = summary["reuse_fresh_saving"]["ci_lower_one_sided_95"]
    amortized_lower = summary["amortized_fresh_saving"]["ci_lower_one_sided_95"]
    reuse_pass = complete and reuse_lower is not None and reuse_lower >= margin
    amortized_pass = complete and amortized_lower is not None and amortized_lower >= 0
    if not complete:
        status = "INCOMPLETE"
    else:
        status = "CONFIRM" if reuse_pass and amortized_pass else "REFUTE"
    karc_ops = summary["karc_operations"]
    rag = summary["by_arm"].get("rag-bm25", {})
    strong_baseline = (
        retrieval_audit["gold_containment_rate"] == 1
        and retrieval_audit["mean_budget_utilization"] >= 0.8
        and retrieval_audit["rank_order_preserved"]
    )
    value = {
        "schema": "e5-g1-verdict-v1",
        "computed_status": status,
        "decision_owner": "main session",
        "runtime_scope": "Codex CLI 0.144.x / gpt-5.6-luna subscription runtime",
        "complete": complete,
        "conjuncts": {
            "reuse_margin": {"margin": margin, "ci_lower": reuse_lower,
                             "pass": reuse_pass},
            "amortized_real_cache": {"threshold": 0,
                                     "ci_lower": amortized_lower,
                                     "pass": amortized_pass},
        },
        "stage2_disposition": (
            "full GO candidate; main confirmation required" if status == "CONFIRM"
            else "quality-only positioning recommended" if status == "REFUTE"
            else "no decision"
        ),
        "retrieval_crippling_self_audit": {
            "pass": strong_baseline,
            "gold_containment_rate": retrieval_audit["gold_containment_rate"],
            "mean_budget_utilization": retrieval_audit["mean_budget_utilization"],
            "bm25_rank_order_preserved": retrieval_audit["rank_order_preserved"],
            "rag_cache_credit_observed": summary["rag_cache_credit_observed"],
            "rag_cache_read_per_task": rag.get("cache_read_per_task"),
        },
        "mcp_routing": {
            "channel_adoption_rate": karc_ops["mcp_channel_adoption_rate"],
            "native_managed_attempts": karc_ops["native_managed_attempts"],
            "native_managed_denied": karc_ops["native_managed_denied"],
        },
        "freeze_sha256": freeze["sha256"],
    }
    value["sha256"] = hash_json(value)
    return value


def error_digest(detail: str | None) -> dict:
    raw = (detail or "").encode("utf-8", errors="replace")
    return {"length": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def tool_counts(transcript: Iterable[dict], guard_events: Iterable[dict]) -> dict:
    search = 0
    get = 0
    for event in transcript:
        if event.get("kind") != "tool_use" or event.get("logical_server") != "karc":
            continue
        search += int(event.get("logical_tool") == "search")
        get += int(event.get("logical_tool") == "get")
    native = 0
    denied = 0
    for event in guard_events:
        if event.get("hook_event_name") != "PreToolUse":
            continue
        if event.get("mcp_server") == "karc":
            continue
        if event.get("mode") == "deny-managed" and event.get("denied"):
            native += 1
            denied += 1
    return {
        "mcp_search_calls": search, "mcp_get_calls": get,
        "native_managed_attempts": native, "native_managed_denied": denied,
    }
