"""E5-G0 rev2: deterministic, model-free cache/retrieval kill-gate.

The module reuses the E4 seed-4000 corpus, policy replay, version metadata,
and hot-supersession events.  It never imports or invokes model/embedding
code.  All decision rules are the preregistered rev2 rules.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import re
from pathlib import Path
from typing import Iterable, Sequence

from karc.bench.e3_retrieval import bm25_plans, chunk_fixture, tokenize
from karc.bench.e4_replay import load_confirmed_config, replay_cell
from karc.replay.runner import git_hash


CORPUS_SEED = 4000
SOURCE_FIXTURE_REL = Path("fixture/e4-v2")
SCHEDULE_SEEDS = (4000, 4001, 4002)
R_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
BUDGET_GRID = (5, 10)
SESSION_LENGTHS = (8, 24)
PRIMARY_SESSION_LENGTH = 8
PRIMARY_BUDGET = 5
N_SESSIONS = 4  # makes (n-1)*4 divisible by four for exact r quarters
ENGINE_RHO_LABEL = 0.5
ENGINE_SIGMA_LABEL = 0.3
CACHE_ARMS = ("karc", "karc-no-outcome", "classic")
RETRIEVAL_ARMS = ("rag-bm25", "search-only")
HIGH_REUSE_MIN = 0.75
F_K3_RATIO = 0.5
PLAUSIBLE_R_MAX = 0.5


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode()


def hash_json(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _active(entry: dict, seq: int) -> bool:
    valid_from = entry.get("valid_from_seq")
    superseded_at = entry.get("superseded_at_seq")
    return (
        valid_from is not None
        and int(valid_from) <= seq
        and (superseded_at is None or int(superseded_at) > seq)
    )


def _source_manifest(repo_root: Path) -> tuple[dict, dict]:
    """Return the frozen E4-v2 corpus; its original schedule is discarded."""
    fixture = repo_root / SOURCE_FIXTURE_REL
    manifest = json.loads((fixture / "manifest.json").read_text(encoding="utf-8"))
    build = json.loads((fixture / "BUILD.json").read_text(encoding="utf-8"))
    freeze = json.loads((fixture / "freeze.json").read_text(encoding="utf-8"))
    if manifest.get("seed") != CORPUS_SEED or not freeze.get("pass"):
        raise ValueError("E5 requires the frozen seed-4000 E4-v2 corpus")
    if manifest.get("manifest_sha256") != freeze.get("manifest_sha256"):
        raise ValueError("E4-v2 freeze/manifest hash mismatch")
    manifest = copy.deepcopy(manifest)
    for v1 in manifest["groups"]["versioned_v1"]:
        v2 = manifest["artifacts"][v1]["superseded_by_version_id"]
        manifest["artifacts"][v1].update({
            "valid_from_seq": 1, "superseded_at_seq": None,
            "is_current_at_end": True,
        })
        manifest["artifacts"][v2].update({
            "valid_from_seq": None, "superseded_at_seq": None,
            "is_current_at_end": False,
        })
    manifest["experiment"] = "E5-G0"
    manifest["engine_source_labels_non_control"] = {
        "rho": ENGINE_RHO_LABEL, "sigma": ENGINE_SIGMA_LABEL,
        "reason": "replay_cell compatibility only; E5 controls r, not rho/sigma",
    }
    manifest.pop("manifest_sha256", None)
    return manifest, build


def _task_row(
    *, seq: int, session_id: str, session_task: int, required: str,
    manifest: dict, pre_events: Iterable[dict] = (),
    forbidden: Iterable[str] = (), outcome: Iterable[str] = (),
    reuse: bool,
) -> dict:
    entry = manifest["artifacts"][required]
    stale_values = [manifest["artifacts"][v]["value"] for v in forbidden]
    return {
        "task_id": f"E5-{session_id}-T{session_task:02d}",
        "seq": seq,
        "session_id": session_id,
        "session_task": session_task,
        "language": "en",
        "difficulty_tier": "controlled",
        "structure": "reuse-gold" if reuse else "new-gold",
        "task_type": "e5-reuse",
        "required_artifacts": [entry["artifact_id"]],
        "required_versions": [required],
        "forbidden_versions": list(forbidden),
        "pre_events": list(pre_events),
        "outcome_inject": list(outcome),
        "outcome_targets": [],
        "max_turns": 1,
        "prompt": (
            f"Return the current value for `{entry['fact_key']}` as exactly "
            f"`{entry['fact_key']} = <value>`."
        ),
        "answer_fact": {
            "artifact_id": entry["artifact_id"],
            "version_id": required,
            "key": entry["fact_key"],
            "value": entry["value"],
        },
        "stale_values": stale_values,
        "format_regex": (
            rf"\A\s*{re.escape(entry['fact_key'])}\s*=\s*"
            r"E4-[0-9A-F]{4}-[0-9A-F]{4}\s*\Z"
        ),
        "e5_resident_reuse": reuse,
    }


def _repeat_allocation(
    reuse_factor: float, session_length: int, session_count: int = N_SESSIONS,
) -> list[int]:
    opportunities = session_count * (session_length - 1)
    total = int(round(reuse_factor * opportunities))
    if total != reuse_factor * opportunities:
        raise AssertionError("session batch cannot represent requested r exactly")
    base, extra = divmod(total, session_count)
    return [base + int(index < extra) for index in range(session_count)]


def build_reuse_schedule(
    repo_root: str | Path, *, schedule_seed: int, session_length: int,
    reuse_factor: float, session_count: int = N_SESSIONS,
    allow_unregistered_seed: bool = False,
    allow_unregistered_session_length: bool = False,
) -> tuple[dict, dict, list[dict]]:
    """Build persistent-policy sessions with an exact arm-neutral r.

    E5-G0 callers retain the registered four-session/three-seed defaults.
    E5-G1 may request another pre-frozen schedule seed and a multiple-of-four
    session batch without changing the decision-bearing G0 grid.
    """
    repo_root = Path(repo_root)
    if schedule_seed not in SCHEDULE_SEEDS and not allow_unregistered_seed:
        raise ValueError(f"schedule_seed must be one of {SCHEDULE_SEEDS}")
    if session_length not in SESSION_LENGTHS and not allow_unregistered_session_length:
        raise ValueError(f"session_length must be one of {SESSION_LENGTHS}")
    if reuse_factor not in R_GRID:
        raise ValueError(f"reuse_factor must be one of {R_GRID}")
    if session_count <= 0 or session_count % 4:
        raise ValueError("session_count must be a positive multiple of four")
    manifest, source_build = _source_manifest(repo_root)
    rng = random.Random(schedule_seed * 1009 + session_length * 37)
    stable = list(manifest["groups"]["stable"])
    versioned = list(zip(
        manifest["groups"]["versioned_v1"],
        manifest["groups"]["versioned_v2"],
    ))
    rng.shuffle(stable)
    rng.shuffle(versioned)
    stable_cursor = 0
    if session_count > len(versioned):
        raise ValueError("session_count exceeds the versioned anchor pool")
    repeat_counts = _repeat_allocation(
        reuse_factor, session_length, session_count,
    )
    tasks: list[dict] = []
    assignments: list[dict] = []
    reused_artifacts: set[str] = set()
    superseded_artifacts: set[str] = set()

    for session_index, repeat_count in enumerate(repeat_counts):
        session_id = f"S{session_index + 1:02d}"
        flags = [True] * repeat_count + [False] * (
            session_length - 1 - repeat_count
        )
        random.Random(
            schedule_seed * 7919 + session_length * 101
            + int(reuse_factor * 100) * 53 + session_index
        ).shuffle(flags)
        anchor_v1, anchor_v2 = versioned[session_index]
        current_by_artifact: dict[str, str] = {}
        history: list[str] = []
        anchor_superseded = False

        for session_task in range(1, session_length + 1):
            seq = len(tasks) + 1
            pre_events: list[dict] = []
            forbidden: list[str] = []
            is_reuse = False
            if session_task == 1:
                required = anchor_v1
            else:
                is_reuse = flags[session_task - 2]
                if is_reuse and not anchor_superseded:
                    required = anchor_v2
                    pre_events.append({
                        "event_type": "validity_changed",
                        "version_id": anchor_v1,
                        "origin": "script",
                        "new_validity": "STALE",
                    })
                    forbidden.append(anchor_v1)
                    manifest["artifacts"][anchor_v1].update({
                        "superseded_at_seq": seq,
                        "is_current_at_end": False,
                    })
                    manifest["artifacts"][anchor_v2].update({
                        "valid_from_seq": seq,
                        "is_current_at_end": True,
                    })
                    current_by_artifact[
                        manifest["artifacts"][anchor_v1]["artifact_id"]
                    ] = anchor_v2
                    anchor_superseded = True
                    superseded_artifacts.add(
                        manifest["artifacts"][anchor_v1]["artifact_id"]
                    )
                elif is_reuse:
                    candidates = sorted(current_by_artifact.values())
                    required = candidates[
                        rng.randrange(len(candidates))
                    ]
                else:
                    required = stable[stable_cursor]
                    stable_cursor += 1

            # A common, exogenous correction creates a K-ARC/classic branch
            # even at r=0.  It never changes the current task's gold.
            if session_task == 4 and history:
                target = history[1] if len(history) > 1 else history[0]
                target_artifact = manifest["artifacts"][target]["artifact_id"]
                target = current_by_artifact.get(target_artifact, target)
                pre_events.append({
                    "event_type": "corrected", "version_id": target,
                    "origin": "script",
                })
            outcome = ("validated",) if session_task == 2 else ()
            task = _task_row(
                seq=seq, session_id=session_id, session_task=session_task,
                required=required, manifest=manifest, pre_events=pre_events,
                forbidden=forbidden, outcome=outcome, reuse=is_reuse,
            )
            artifact = task["required_artifacts"][0]
            if is_reuse:
                if artifact not in current_by_artifact:
                    raise AssertionError("reuse task did not select prior-session gold")
                reused_artifacts.add(artifact)
            elif artifact in current_by_artifact:
                raise AssertionError("new task reused same-session gold")
            current_by_artifact[artifact] = required
            history.append(required)
            tasks.append(task)
            assignments.append({
                "task_id": task["task_id"], "seq": seq,
                "session_id": session_id, "session_task": session_task,
                "required_version": required, "required_artifact": artifact,
                "resident_reuse": is_reuse,
                "hot_supersession": bool(forbidden),
            })

    opportunities = session_count * (session_length - 1)
    reuse_count = sum(row["resident_reuse"] for row in assignments)
    measured_r = reuse_count / opportunities
    if measured_r != reuse_factor:
        raise AssertionError((reuse_factor, measured_r))
    measured_sigma = (
        len(superseded_artifacts) / len(reused_artifacts)
        if reused_artifacts else None
    )
    manifest["structure"] = {
        "tasks": len(tasks), "sessions": session_count,
        "session_length": session_length,
        "reuse_opportunities": opportunities,
        "reused_occurrences": reuse_count,
        "reused_doc_ids": len(reused_artifacts),
        "r_requested": reuse_factor, "r_measured": measured_r,
        "rho_measured_all_tasks": reuse_count / len(tasks),
        "supersession_events": sum(
            event["event_type"] == "validity_changed"
            for task in tasks for event in task["pre_events"]
        ),
        "sigma_denominator_reused_doc_ids": len(reused_artifacts),
        "sigma_measured": measured_sigma,
        "sigma_defined": measured_sigma is not None,
        "corrected_events": sum(
            event["event_type"] == "corrected"
            for task in tasks for event in task["pre_events"]
        ),
        "validated_outcome_events": sum(
            event == "validated" for task in tasks
            for event in task["outcome_inject"]
        ),
    }
    manifest["fixture_contract"] = {
        **manifest["fixture_contract"],
        "status": "E5-G0 deterministic model-free schedule",
        "controlled_axis": "same-session prior-gold reuse factor r",
        "accounting_reset": "each session",
        "policy_state_reset": "never within schedule batch",
    }
    manifest.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = hash_json(manifest)
    schedule = {
        "schema": "e5-g0-reuse-schedule-v2",
        "corpus_seed": CORPUS_SEED,
        "schedule_seed": schedule_seed,
        "session_length": session_length,
        "sessions": session_count,
        "reuse_factor_requested": reuse_factor,
        "reuse_factor_measured": measured_r,
        "reuse_count": reuse_count,
        "reuse_denominator_eligible_followups": opportunities,
        "definition": (
            "fraction of non-initial session tasks whose gold artifact was "
            "gold for an earlier task in the same session"
        ),
        "rho_measured_all_tasks": manifest["structure"]["rho_measured_all_tasks"],
        "sigma_measured": measured_sigma,
        "sigma_defined": measured_sigma is not None,
        "assignments": assignments,
        "corpus_content_hash": manifest["corpus_content_hash"],
        "source_e4_manifest_sha256": source_build["manifest_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "tasks_sha256": hash_json(tasks),
    }
    schedule["sha256"] = hash_json(schedule)
    return schedule, manifest, tasks


def _manifest_at_budget(manifest: dict, budget_pct: int) -> dict:
    value = copy.deepcopy(manifest)
    total = sum(int(entry["size_tok"]) for entry in value["artifacts"].values())
    value["cell"] = {
        "rho": ENGINE_RHO_LABEL, "sigma": ENGINE_SIGMA_LABEL,
        "budget_pct": budget_pct,
        "budget_tokens": max(1, int(total * budget_pct / 100.0)),
    }
    value.pop("manifest_sha256", None)
    value["manifest_sha256"] = hash_json(value)
    return value


def _search_only_plans(
    tasks: Sequence[dict], manifest: dict, budget_tokens: int,
) -> dict[str, dict]:
    """Deterministic oracle-targeted, budget-parity secondary search plans."""
    plans: dict[str, dict] = {}
    for task in tasks:
        required = list(task["required_versions"])
        active = [
            version for version, entry in manifest["artifacts"].items()
            if _active(entry, int(task["seq"])) and version not in required
        ]
        active.sort(key=lambda version: hashlib.sha256(
            f"{CORPUS_SEED}|{task['task_id']}|{version}".encode()
        ).hexdigest())
        selected: list[str] = []
        used = 0
        for version in [*required, *active]:
            cost = int(manifest["artifacts"][version]["size_tok"])
            if used + cost <= budget_tokens:
                selected.append(version)
                used += cost
        plans[task["task_id"]] = {
            "artifact_ids": selected, "tokens": used,
            "budget_tokens": budget_tokens,
            "budget_utilization": used / budget_tokens,
            "oracle_target_first": True,
        }
    return plans


def _tokens(manifest: dict, versions: Iterable[str]) -> int:
    return sum(int(manifest["artifacts"][v]["size_tok"]) for v in versions)


def _stale_versions(manifest: dict, seq: int) -> set[str]:
    return {
        version for version, entry in manifest["artifacts"].items()
        if entry.get("superseded_at_seq") is not None
        and int(entry["superseded_at_seq"]) <= seq
    }


def _by_session(tasks: Sequence[dict], rows: Sequence[dict]):
    task_sessions: dict[str, list[dict]] = {}
    row_sessions: dict[str, list[dict]] = {}
    for task, row in zip(tasks, rows):
        task_sessions.setdefault(task["session_id"], []).append(task)
        row_sessions.setdefault(task["session_id"], []).append(row)
    return [(sid, task_sessions[sid], row_sessions[sid]) for sid in task_sessions]


def _cache_account(
    *, arm: str, tasks: Sequence[dict], replay_rows: Sequence[dict],
    plans: dict[str, dict], manifest: dict,
) -> tuple[dict, list[dict]]:
    resident_fresh = 0
    miss_retrieval_tokens = 0
    gold_only_fresh = 0
    resident_gross = 0
    c_init = 0
    misses = 0
    containment = 0
    stale_hits = 0
    ledger: list[dict] = []
    for session_id, session_tasks, session_rows in _by_session(tasks, replay_rows):
        initial = set(session_rows[0]["arms"][arm]["working_set_versions"])
        seen_resident = set(initial)
        seen_gold = set(initial)
        init_tokens = _tokens(manifest, initial)
        resident_fresh += init_tokens
        gold_only_fresh += init_tokens
        resident_gross += init_tokens
        c_init += init_tokens
        for task, row in zip(session_tasks, session_rows):
            resident = set(row["arms"][arm]["working_set_versions"])
            required = set(task["required_versions"])
            sufficient = required <= resident
            miss = not sufficient
            misses += int(miss)
            containment += int(sufficient)
            retrieval = set(plans[task["task_id"]]["artifact_ids"])
            gold_context = resident | (required if miss else set())
            resident_new = resident - seen_resident
            gold_new = gold_context - seen_gold
            resident_new_tokens = _tokens(manifest, resident_new)
            plan_tokens = int(plans[task["task_id"]]["tokens"]) if miss else 0
            resident_tokens = _tokens(manifest, resident)
            # rev2 requires a retrieval-equivalent miss: the complete plan is
            # charged per miss with no prefix-cache credit, exactly as it is
            # for the primary retrieval family.  Only resident-set documents
            # receive cache-family doc-identity credit.
            resident_fresh += resident_new_tokens
            miss_retrieval_tokens += plan_tokens
            gold_only_fresh += _tokens(manifest, gold_new)
            resident_gross += resident_tokens
            seen_resident |= resident
            seen_gold |= gold_context
            stale = resident & _stale_versions(manifest, int(task["seq"]))
            stale_hits += int(bool(stale))
            ledger.append({
                "arm": arm, "session_id": session_id,
                "task_id": task["task_id"], "seq": task["seq"],
                "resident_versions": sorted(resident),
                "required_versions": sorted(required),
                "resident_sufficient": sufficient,
                "miss": miss,
                "miss_retrieval_versions": sorted(retrieval) if miss else [],
                "resident_fresh_versions": sorted(resident_new),
                "resident_fresh_tokens": resident_new_tokens,
                "miss_retrieval_tokens_no_credit": plan_tokens,
                "primary_fresh_tokens": resident_new_tokens + plan_tokens,
                "resident_gross_tokens_w1": resident_tokens,
                "gross_tokens_w1": resident_tokens + plan_tokens,
                "gold_only_context_versions": sorted(gold_context),
                "stale_resident_versions": sorted(stale),
            })
    n = len(tasks)
    primary_fresh = resident_fresh + miss_retrieval_tokens
    gross = resident_gross + miss_retrieval_tokens
    return {
        "tasks": n,
        "sessions": N_SESSIONS,
        "c_init_tokens": c_init,
        "resident_fresh_tokens": resident_fresh,
        "miss_retrieval_tokens_no_credit": miss_retrieval_tokens,
        "miss_tasks": misses,
        "miss_rate": misses / n,
        "primary_total_fresh_tokens": primary_fresh,
        "primary_amortized_fresh_tokens": primary_fresh / n,
        "oracle_gold_only_total_fresh_tokens": gold_only_fresh,
        "oracle_gold_only_amortized_fresh_tokens": gold_only_fresh / n,
        "total_gross_tokens_w1": gross,
        "amortized_w0": primary_fresh / n,
        "amortized_w0_1": (
            miss_retrieval_tokens + resident_fresh
            + .1 * (resident_gross - resident_fresh)
        ) / n,
        "amortized_w1": gross / n,
        "oracle_sufficient_tasks": containment,
        "oracle_sufficiency_rate": containment / n,
        "stale_exposure_tasks": stale_hits,
        "stale_exposure_rate": stale_hits / n,
        "miss_accounting": (
            "BM25 budget-parity whole plan charged per miss without cache "
            "credit; doc-identity credit applies only to resident context"
        ),
    }, ledger


def _retrieval_account(
    *, arm: str, tasks: Sequence[dict], plans: dict[str, dict], manifest: dict,
) -> tuple[dict, list[dict]]:
    primary = 0
    symmetric_fresh = 0
    gross = 0
    containment = 0
    stale_hits = 0
    ledger: list[dict] = []
    for session_id in dict.fromkeys(task["session_id"] for task in tasks):
        seen: set[str] = set()
        for task in (value for value in tasks if value["session_id"] == session_id):
            context = set(plans[task["task_id"]]["artifact_ids"])
            fresh = context - seen
            context_tokens = _tokens(manifest, context)
            fresh_tokens = _tokens(manifest, fresh)
            primary += context_tokens  # rev2: no retrieval session-cache credit
            symmetric_fresh += fresh_tokens
            gross += context_tokens
            seen |= context
            required = set(task["required_versions"])
            sufficient = required <= context
            containment += int(sufficient)
            stale = context & _stale_versions(manifest, int(task["seq"]))
            stale_hits += int(bool(stale))
            ledger.append({
                "arm": arm, "session_id": session_id,
                "task_id": task["task_id"], "seq": task["seq"],
                "required_versions": sorted(required),
                "context_versions": sorted(context),
                "primary_fresh_versions": sorted(context),
                "primary_fresh_tokens": context_tokens,
                "symmetric_fresh_versions": sorted(fresh),
                "symmetric_fresh_tokens": fresh_tokens,
                "gross_tokens_w1": context_tokens,
                "retrieval_contains_gold": sufficient,
                "stale_versions_exposed": sorted(stale),
            })
    n = len(tasks)
    source_tokens = sum(
        int(entry["size_tok"]) for entry in manifest["artifacts"].values()
    )
    return {
        "tasks": n,
        "sessions": N_SESSIONS,
        "c_init_index_source_tokens_separate": source_tokens,
        "primary_total_fresh_tokens": primary,
        "primary_amortized_fresh_tokens": primary / n,
        "symmetric_total_fresh_tokens": symmetric_fresh,
        "symmetric_amortized_fresh_tokens": symmetric_fresh / n,
        "total_gross_tokens_w1": gross,
        "amortized_w0": symmetric_fresh / n,
        "amortized_w0_1": (
            symmetric_fresh + .1 * (gross - symmetric_fresh)
        ) / n,
        "amortized_w1": gross / n,
        "retrieval_contains_gold_tasks": containment,
        "retrieval_contains_gold_rate": containment / n,
        "stale_exposure_tasks": stale_hits,
        "stale_exposure_rate": stale_hits / n,
        "primary_cache_credit": "none",
    }, ledger


def belady_containment(tasks: Sequence[dict], manifest: dict, budget: int) -> dict:
    """Farthest-next-use gold cache, persistent across session boundaries."""
    required = [task["required_versions"][0] for task in tasks]
    cache: set[str] = set()
    hits = 0
    rows: list[dict] = []
    for index, task in enumerate(tasks):
        for event in task["pre_events"]:
            if event["event_type"] == "validity_changed":
                cache.discard(event["version_id"])
        version = required[index]
        hit = version in cache
        hits += int(hit)
        before = sorted(cache)
        cache.add(version)
        while _tokens(manifest, cache) > budget:
            def next_use(candidate: str):
                try:
                    return required.index(candidate, index + 1)
                except ValueError:
                    return math.inf
            victim = max(cache, key=lambda value: (next_use(value), value))
            cache.remove(victim)
        rows.append({
            "task_id": task["task_id"], "seq": task["seq"],
            "required_version": version, "cache_before": before,
            "contained": hit, "cache_after": sorted(cache),
        })
    return {
        "algorithm": "size-aware capacity with Belady farthest-next-use eviction",
        "tasks": len(tasks), "contained_tasks": hits,
        "containment_rate": hits / len(tasks), "rows": rows,
    }


def _stale_frontier(rows: Sequence[dict], manifest: dict) -> dict:
    result: dict[str, dict] = {}
    for arm in CACHE_ARMS:
        hits = 0
        for row in rows:
            resident = set(row["arms"][arm]["working_set_versions"])
            hits += int(bool(resident & _stale_versions(manifest, int(row["seq"]))))
        result[arm] = {
            "tasks": len(rows), "stale_exposure_tasks": hits,
            "stale_exposure_rate": hits / len(rows),
        }
    result["decomposition"] = {
        "classic_minus_karc_tasks": (
            result["classic"]["stale_exposure_tasks"]
            - result["karc"]["stale_exposure_tasks"]
        ),
        "no_outcome_minus_karc_tasks": (
            result["karc-no-outcome"]["stale_exposure_tasks"]
            - result["karc"]["stale_exposure_tasks"]
        ),
    }
    return result


def evaluate_cell(
    repo_root: str | Path, fixture_root: str | Path, *, schedule_seed: int,
    session_length: int, budget_pct: int, reuse_factor: float,
) -> tuple[dict, list[dict], dict]:
    repo_root = Path(repo_root)
    fixture_root = Path(fixture_root)
    schedule, schedule_manifest, tasks = build_reuse_schedule(
        repo_root, schedule_seed=schedule_seed,
        session_length=session_length, reuse_factor=reuse_factor,
    )
    manifest = _manifest_at_budget(schedule_manifest, budget_pct)
    budget = int(manifest["cell"]["budget_tokens"])
    replay_summary, replay_rows = replay_cell(
        rho=ENGINE_RHO_LABEL, sigma=ENGINE_SIGMA_LABEL,
        budget_pct=budget_pct,
        confirmed_config=load_confirmed_config(repo_root),
        fixture_manifest=manifest, fixture_tasks=tasks,
    )
    chunks = chunk_fixture(fixture_root, manifest)
    plans = {
        "rag-bm25": bm25_plans(chunks, tasks, manifest, budget),
        "search-only": _search_only_plans(tasks, manifest, budget),
    }
    costs: dict[str, dict] = {}
    ledger: list[dict] = []
    for arm in CACHE_ARMS:
        summary, rows = _cache_account(
            arm=arm, tasks=tasks, replay_rows=replay_rows,
            plans=plans["rag-bm25"], manifest=manifest,
        )
        costs[arm] = summary
        ledger.extend(rows)
    for arm in RETRIEVAL_ARMS:
        summary, rows = _retrieval_account(
            arm=arm, tasks=tasks, plans=plans[arm], manifest=manifest,
        )
        costs[arm] = summary
        ledger.extend(rows)
    belady = belady_containment(tasks, manifest, budget)
    jaccard = [row["jaccard_karc_classic"] for row in replay_rows]
    cell = {
        "schema": "e5-g0-cell-v2",
        "cell": {
            "schedule_seed": schedule_seed,
            "session_length": session_length,
            "budget_pct": budget_pct, "budget_tokens": budget,
            "reuse_factor": reuse_factor,
        },
        "measured": {
            "r": schedule["reuse_factor_measured"],
            "rho_all_tasks": schedule["rho_measured_all_tasks"],
            "sigma": schedule["sigma_measured"],
            "sigma_defined": schedule["sigma_defined"],
        },
        "schedule_sha256": schedule["sha256"],
        "fixture": {
            "corpus_seed": CORPUS_SEED,
            "corpus_content_hash": manifest["corpus_content_hash"],
            "manifest_sha256": manifest["manifest_sha256"],
            "tasks_sha256": schedule["tasks_sha256"],
        },
        "costs": costs,
        "clairvoyant_belady": {
            key: value for key, value in belady.items() if key != "rows"
        },
        "stale_frontier": _stale_frontier(replay_rows, manifest),
        "g_e4_act_recheck": {
            "criterion": "at least one Jaccard(karc,classic)<1 task",
            "divergent_tasks": sum(value < 1.0 for value in jaccard),
            "minimum_jaccard": min(jaccard),
            "pass": any(value < 1.0 for value in jaccard),
        },
        "retrieval_build_compute": {
            "rag-bm25": {
                "algorithm": "stdlib BM25Index", "chunks": len(chunks),
                "term_occurrences": sum(len(tokenize(chunk.text)) for chunk in chunks),
                "source_tokens": sum(
                    int(entry["size_tok"]) for entry in manifest["artifacts"].values()
                ),
            },
            "search-only": {
                "algorithm": "deterministic active-manifest scan, oracle target first",
                "artifacts_scanned_per_task": len(manifest["artifacts"]),
            },
        },
        "replay": {"tasks": len(replay_rows),
                   "e4_replay_sha256": replay_summary["sha256"]},
        "model_calls": 0, "embedding_calls": 0,
    }
    cell["sha256"] = hash_json(cell)
    cell_key = dict(cell["cell"])
    for row in ledger:
        row["cell"] = cell_key
    for row in belady["rows"]:
        row["arm"] = "clairvoyant-belady"
        row["cell"] = cell_key
        ledger.append(row)
    return cell, ledger, schedule


def _select_cells(
    cells: Sequence[dict], *, session_length: int, budget_pct: int,
    reuse_factor: float,
) -> list[dict]:
    return [
        cell for cell in cells
        if cell["cell"]["session_length"] == session_length
        and cell["cell"]["budget_pct"] == budget_pct
        and cell["cell"]["reuse_factor"] == reuse_factor
    ]


def _cost(cell: dict, arm: str, accounting: str) -> float:
    row = cell["costs"][arm]
    if accounting == "primary":
        return float(row["primary_amortized_fresh_tokens"])
    if accounting == "symmetric":
        return float(row.get(
            "symmetric_amortized_fresh_tokens",
            row["primary_amortized_fresh_tokens"],
        ))
    return float(row[{"w0": "amortized_w0", "w0.1": "amortized_w0_1",
                      "w1": "amortized_w1"}[accounting]])


def _category(r_star: float | None) -> str:
    if r_star is None:
        return "STOP"
    if r_star <= PLAUSIBLE_R_MAX:
        return "PASS"
    return "PASS-NARROW"


def crossover(
    cells: Sequence[dict], *, session_length: int, budget_pct: int,
    cache_arm: str = "karc", retrieval_arm: str = "rag-bm25",
    accounting: str = "primary", schedule_seed: int | None = None,
) -> dict:
    curve = []
    r_star = None
    for reuse in R_GRID:
        unit = _select_cells(
            cells, session_length=session_length, budget_pct=budget_pct,
            reuse_factor=reuse,
        )
        if schedule_seed is not None:
            unit = [row for row in unit
                    if row["cell"]["schedule_seed"] == schedule_seed]
        if not unit:
            raise ValueError("crossover cell selection is empty")
        cache_values = [_cost(row, cache_arm, accounting) for row in unit]
        retrieval_values = [_cost(row, retrieval_arm, accounting) for row in unit]
        cache = sum(cache_values) / len(cache_values)
        retrieval = sum(retrieval_values) / len(retrieval_values)
        crossed = cache <= retrieval
        if crossed and r_star is None:
            r_star = reuse
        curve.append({
            "reuse_factor": reuse,
            "cache": cache, "retrieval": retrieval,
            "cache_seed_min": min(cache_values), "cache_seed_max": max(cache_values),
            "retrieval_seed_min": min(retrieval_values),
            "retrieval_seed_max": max(retrieval_values),
            "cache_le_retrieval": crossed,
        })
    return {
        "session_length": session_length, "budget_pct": budget_pct,
        "cache_arm": cache_arm, "retrieval_arm": retrieval_arm,
        "accounting": accounting, "schedule_seed": schedule_seed,
        "curve": curve, "r_star": r_star, "category": _category(r_star),
    }


def evaluate_verdict(cells: Sequence[dict]) -> dict:
    primary = crossover(
        cells, session_length=PRIMARY_SESSION_LENGTH,
        budget_pct=PRIMARY_BUDGET,
    )
    classic = crossover(
        cells, session_length=PRIMARY_SESSION_LENGTH,
        budget_pct=PRIMARY_BUDGET, cache_arm="classic",
    )
    budget_robustness = crossover(
        cells, session_length=PRIMARY_SESSION_LENGTH, budget_pct=10,
    )
    length_robustness = crossover(
        cells, session_length=24, budget_pct=PRIMARY_BUDGET,
    )
    length_budget_robustness = crossover(
        cells, session_length=24, budget_pct=10,
    )
    symmetric = crossover(
        cells, session_length=PRIMARY_SESSION_LENGTH,
        budget_pct=PRIMARY_BUDGET, accounting="symmetric",
    )
    weight_curves = {
        weight: crossover(
            cells, session_length=PRIMARY_SESSION_LENGTH,
            budget_pct=PRIMARY_BUDGET, accounting=weight,
        ) for weight in ("w0", "w0.1", "w1")
    }
    per_seed = {
        str(seed): crossover(
            cells, session_length=PRIMARY_SESSION_LENGTH,
            budget_pct=PRIMARY_BUDGET, schedule_seed=seed,
        ) for seed in SCHEDULE_SEEDS
    }
    seed_categories = {row["category"] for row in per_seed.values()}
    seed_agreement = len(seed_categories) == 1

    f_k0_rows = []
    for seed in SCHEDULE_SEEDS:
        cell = _select_cells(
            cells, session_length=PRIMARY_SESSION_LENGTH,
            budget_pct=PRIMARY_BUDGET, reuse_factor=0.0,
        )
        cell = next(row for row in cell if row["cell"]["schedule_seed"] == seed)
        cache = _cost(cell, "karc", "primary")
        retrieval = _cost(cell, "rag-bm25", "primary")
        f_k0_rows.append({
            "schedule_seed": seed, "cache": cache, "retrieval": retrieval,
            "retrieval_lt_cache": retrieval < cache,
        })
    f_k0_pass = all(row["retrieval_lt_cache"] for row in f_k0_rows)

    sufficiency = []
    f_k3 = False
    clairvoyant_capacity_low = False
    for reuse in R_GRID:
        unit = _select_cells(
            cells, session_length=PRIMARY_SESSION_LENGTH,
            budget_pct=PRIMARY_BUDGET, reuse_factor=reuse,
        )
        karc = sum(row["costs"]["karc"]["oracle_sufficiency_rate"]
                   for row in unit) / len(unit)
        clairvoyant = sum(row["clairvoyant_belady"]["containment_rate"]
                          for row in unit) / len(unit)
        ratio = karc / clairvoyant if clairvoyant else None
        triggered = (
            reuse >= HIGH_REUSE_MIN and clairvoyant > 0
            and karc < F_K3_RATIO * clairvoyant
        )
        f_k3 |= triggered
        clairvoyant_capacity_low |= reuse >= HIGH_REUSE_MIN and clairvoyant < .5
        sufficiency.append({
            "reuse_factor": reuse, "karc": karc,
            "clairvoyant": clairvoyant, "karc_over_clairvoyant": ratio,
            "f_k3_triggered": triggered,
        })

    primary_cells = [
        cell for cell in cells
        if cell["cell"]["session_length"] == PRIMARY_SESSION_LENGTH
        and cell["cell"]["budget_pct"] == PRIMARY_BUDGET
    ]
    reuse_act_cells = [
        row for row in primary_cells if row["cell"]["reuse_factor"] > 0
    ]
    act_pass = all(row["g_e4_act_recheck"]["pass"] for row in reuse_act_cells)
    f_k5 = not act_pass
    positive_stale = [row for row in primary_cells
                      if row["measured"]["sigma_defined"]]
    stale_totals = {
        arm: sum(row["stale_frontier"][arm]["stale_exposure_tasks"]
                 for row in positive_stale)
        for arm in CACHE_ARMS
    }
    f_k4 = stale_totals["classic"] - stale_totals["karc"] <= 0
    f_k6 = (
        primary["r_star"] is not None and classic["r_star"] is not None
        and primary["r_star"] > classic["r_star"]
    )
    accounting_sensitive = symmetric["category"] != primary["category"]
    w01_sensitive = weight_curves["w0.1"]["category"] != primary["category"]

    if not f_k0_pass:
        status = "VOID"
        void_reason = "F-K0 failed: r=0 did not reproduce retrieval<cache"
    elif not seed_agreement:
        status = "VOID"
        void_reason = "schedule-seed verdict categories disagree"
    else:
        status = primary["category"]
        void_reason = None
    automatic_stage2 = (
        status == "PASS" and not f_k3 and not f_k5
        and not accounting_sensitive
    )
    verdict = {
        "schema": "e5-g0-verdict-v2",
        "gate": "E5-G0",
        "binding_computed_status": status,
        "decision_owner": "main session",
        "void_reason": void_reason,
        "automatic_stage2_recommendation": automatic_stage2,
        "primary_crossover": primary,
        "classic_crossover": classic,
        "robustness": {
            "c10_n8": budget_robustness,
            "c5_n24": length_robustness,
            "c10_n24": length_budget_robustness,
        },
        "accounting_sensitivity": {
            "symmetric_retrieval_credit": symmetric,
            "weight_curves": weight_curves,
            "accounting_sensitive": accounting_sensitive,
            "w0_1_category_flip": w01_sensitive,
        },
        "seed_stability": {
            "required_seed_count": 3,
            "per_seed": per_seed,
            "categories": sorted(seed_categories),
            "category_agreement": seed_agreement,
        },
        "f_k0_control": {"pass": f_k0_pass, "per_seed": f_k0_rows},
        "sufficiency": {
            "cells": sufficiency, "f_k3_triggered": f_k3,
            "clairvoyant_capacity_low": clairvoyant_capacity_low,
        },
        "stale_decomposition": {
            "sigma_defined_cells": len(positive_stale),
            "exposure_task_totals": stale_totals,
            "classic_minus_karc_tasks": stale_totals["classic"] - stale_totals["karc"],
            "no_outcome_minus_karc_tasks": (
                stale_totals["karc-no-outcome"] - stale_totals["karc"]
            ),
        },
        "g_e4_act_recheck": {
            "scope": "all primary r>0 schedule-seed cells",
            "pass": act_pass, "f_k5_triggered": f_k5,
            "passing_cells": sum(
                row["g_e4_act_recheck"]["pass"] for row in reuse_act_cells
            ),
            "cells": len(reuse_act_cells),
            "divergent_tasks_min": min(
                row["g_e4_act_recheck"]["divergent_tasks"]
                for row in reuse_act_cells
            ),
            "divergent_tasks_max": max(
                row["g_e4_act_recheck"]["divergent_tasks"]
                for row in reuse_act_cells
            ),
        },
        "karc_increment": {
            "karc_r_star": primary["r_star"],
            "classic_r_star": classic["r_star"],
            "f_k6_triggered": f_k6,
        },
        "falsification": {
            "F-K0": {"triggered": not f_k0_pass,
                     "disposition": "VOID; redesign accounting and rerun" if not f_k0_pass else "not triggered"},
            "F-K1": {"triggered": status == "STOP",
                     "disposition": "binding STOP; algorithm re-search" if status == "STOP" else "not triggered"},
            "F-K2": {"triggered": status == "PASS-NARROW",
                     "disposition": "user decision on high-reuse-only Stage 2" if status == "PASS-NARROW" else "not triggered"},
            "F-K3": {"triggered": f_k3,
                     "disposition": "redesign curation then rerun; not a round kill" if f_k3 else "not triggered"},
            "F-K4": {"triggered": f_k4,
                     "disposition": "diagnose curation stale value" if f_k4 else "not triggered"},
            "F-K5": {"triggered": f_k5,
                     "disposition": "generic cache-family only; block Stage 2a" if f_k5 else "not triggered"},
            "F-K6": {"triggered": f_k6,
                     "disposition": "K-ARC increment not established" if f_k6 else "not triggered"},
        },
        "component_biases": {
            "cache_favorable": [
                "doc-identity cache-read ignores churn within each session",
                "oracle gold-only miss is shown only as a non-decision bracket",
            ],
            "cache_unfavorable": [
                "cache C_init is repaid every session",
                "retrieval index-build source tokens are separated from primary context tokens",
            ],
            "primary_miss_rule": "BM25 budget-parity plan, including distractors",
            "primary_retrieval_cache_credit": "none",
        },
        "interpretation": (
            "PASS is a necessary condition under the registered mixed-bias "
            "accounting, not a blanket upper bound or sufficient model result. "
            "STOP after F-K0 is a strong negative signal. Belady and gold-only "
            "figures are explicit idealized brackets only."
        ),
        "model_calls": 0, "embedding_calls": 0,
    }
    verdict["sha256"] = hash_json(verdict)
    return verdict


def run_grid(
    repo_root: str | Path, fixture_root: str | Path,
) -> tuple[list[dict], list[dict], list[dict], dict, dict]:
    repo_root = Path(repo_root)
    fixture_root = Path(fixture_root)
    cells: list[dict] = []
    ledger: list[dict] = []
    schedules: dict[tuple[int, int, float], dict] = {}
    for schedule_seed in SCHEDULE_SEEDS:
        for session_length in SESSION_LENGTHS:
            for reuse in R_GRID:
                for budget_pct in BUDGET_GRID:
                    cell, rows, schedule = evaluate_cell(
                        repo_root, fixture_root, schedule_seed=schedule_seed,
                        session_length=session_length,
                        budget_pct=budget_pct, reuse_factor=reuse,
                    )
                    cells.append(cell)
                    ledger.extend(rows)
                    schedules[(schedule_seed, session_length, reuse)] = schedule
    verdict = evaluate_verdict(cells)
    source_paths = (
        "src/karc/bench/e5_killgate.py",
        "src/karc/bench/e4_replay.py",
        "src/karc/bench/e4_fixture.py",
        "src/karc/bench/e3_retrieval.py",
        "scripts/run_e5_g0.py",
    )
    config = {
        "schema": "e5-g0-config-v2",
        "experiment": "E5-G0",
        "authority": "docs/analysis/experiment-design--e5-fair-comparison.md §2 rev2",
        "corpus_seed": CORPUS_SEED,
        "schedule_seeds": list(SCHEDULE_SEEDS),
        "grid": {"reuse_factor": list(R_GRID),
                 "budget_pct": list(BUDGET_GRID),
                 "session_length": list(SESSION_LENGTHS)},
        "sessions_per_cell": N_SESSIONS,
        "primary_cell": {"session_length": PRIMARY_SESSION_LENGTH,
                         "budget_pct": PRIMARY_BUDGET},
        "controlled_axis": "same-session prior-gold reuse r",
        "rho_sigma": "measured only; sigma undefined at r=0",
        "session_semantics": {
            "accounting": "fresh/cache-read and C_init reset each session",
            "policy_state": "persists across all four sessions in a cell",
        },
        "primary_accounting": {
            "cache_miss": "same task's full budget-parity BM25 plan",
            "retrieval_cache_credit": "none",
            "index_build": "source tokens and compute proxy reported separately",
        },
        "secondary_accounting": {
            "symmetric_retrieval_doc_identity_credit": True,
            "cache_read_weights": [0, .1, 1],
            "gold_only_miss": "non-decision idealized bracket",
        },
        "binding": {
            "F-K0": "r=0 retrieval < cache for every schedule seed",
            "PASS": "r* <= 0.5", "PASS-NARROW": "0.5 < r* <= 1",
            "STOP": "r* absent", "primary_only": "c=5%, n=8",
        },
        "model_calls": 0, "embedding_calls": 0, "stdlib_only": True,
        "fixture_root": str(fixture_root.relative_to(repo_root)),
        "confirmed_config_source": "docs/experiments/E1-2-parameter-suite/confirmed-config.json",
        "source_sha256": {path: sha256_file(repo_root / path) for path in source_paths},
        "code_git_hash": git_hash(repo_root),
    }
    config["sha256"] = hash_json(config)
    return cells, ledger, list(schedules.values()), verdict, config


def _fmt(value: float) -> str:
    return f"{value:,.2f}"


def _curve_table(lines: list[str], title: str, curve: dict, cells: Sequence[dict]) -> None:
    lines += [f"### {title}", "",
              "| r | K-ARC primary fresh/task | rag-bm25 primary fresh/task | classic primary fresh/task | search-only primary fresh/task | K-ARC gold-only miss bracket |",
              "|---:|---:|---:|---:|---:|---:|"]
    for point in curve["curve"]:
        reuse = point["reuse_factor"]
        unit = _select_cells(
            cells, session_length=curve["session_length"],
            budget_pct=curve["budget_pct"], reuse_factor=reuse,
        )
        classic = sum(_cost(row, "classic", "primary") for row in unit) / len(unit)
        search = sum(_cost(row, "search-only", "primary") for row in unit) / len(unit)
        gold = sum(row["costs"]["karc"]["oracle_gold_only_amortized_fresh_tokens"]
                   for row in unit) / len(unit)
        lines.append(
            f"| {reuse:.2f} | {_fmt(point['cache'])} | {_fmt(point['retrieval'])} "
            f"| {_fmt(classic)} | {_fmt(search)} | {_fmt(gold)} |"
        )
    lines += ["", f"- r*: **{curve['r_star']}**; category: **{curve['category']}**", ""]


def render_report(
    cells: Sequence[dict], verdict: dict, config: dict,
    raw_hashes: dict[str, str],
) -> str:
    status = verdict["binding_computed_status"]
    lines = [
        "# E5-G0 무모델 교차 kill-gate (rev2)", "",
        f"## 계산 판정: **{status}**", "",
        "사전등록 rev2의 binding 계산 결과다. main 병합·kill/PASS 확정과 Stage 2 결정은 메인 세션 소유다.", "",
        f"- primary `(n=8, c=5%)` r*: **{verdict['primary_crossover']['r_star']}**",
        f"- F-K0: **{'PASS' if verdict['f_k0_control']['pass'] else 'FAIL'}**",
        f"- schedule seed 판정 구간 일치: **{verdict['seed_stability']['category_agreement']}**",
        f"- 자동 Stage 2 권고 가능: **{verdict['automatic_stage2_recommendation']}**",
        "- 모델/LLM/embedding 호출: **0 / 0**", "",
        "## 1. rev2 실행 계약", "",
        ("유일 통제축 `r`은 각 세션의 첫 task를 제외한 eligible task 중 같은 세션의 "
         "앞선 gold artifact가 다시 gold가 된 비율이다. n=8에서 정확한 quarter를 만들기 "
         "위해 4개 세션(28 eligible task)을 한 cell로 묶었다. 회계와 `C_init`은 세션마다 "
         "reset하지만 policy state는 4개 세션 내내 지속한다. schedule seed는 4000/4001/4002다."), "",
        ("cache miss는 같은 task의 **전체 BM25 budget-parity plan**(distractor 포함)을 지불한다. "
         "gold-only miss는 비판정 bracket이다. primary retrieval은 매 task plan token을 fresh로 "
         "청구하며 session-cache credit이 없다. index source token/build proxy는 별도 축이다."), "",
        "## 2. primary·budget robustness 비용 곡선", "",
    ]
    _curve_table(lines, "primary n=8, c=5%", verdict["primary_crossover"], cells)
    _curve_table(lines, "budget robustness n=8, c=10%", verdict["robustness"]["c10_n8"], cells)
    lines += ["## 3. session-length·회계 민감도", "",
              "| variant | r* | category | primary 대비 구간 변경 |",
              "|---|---:|---|:---:|"]
    variants = [
        ("n=24,c=5%", verdict["robustness"]["c5_n24"]),
        ("n=24,c=10%", verdict["robustness"]["c10_n24"]),
        ("retrieval symmetric doc-cache", verdict["accounting_sensitivity"]["symmetric_retrieval_credit"]),
        ("w=0", verdict["accounting_sensitivity"]["weight_curves"]["w0"]),
        ("w=0.1", verdict["accounting_sensitivity"]["weight_curves"]["w0.1"]),
        ("w=1", verdict["accounting_sensitivity"]["weight_curves"]["w1"]),
    ]
    for label, row in variants:
        changed = row["category"] != verdict["primary_crossover"]["category"]
        lines.append(f"| {label} | {row['r_star']} | {row['category']} | {changed} |")
    lines += ["", f"- accounting-sensitive(auto Stage 2 금지): **{verdict['accounting_sensitivity']['accounting_sensitive']}**",
              f"- w=0.1 category flip: **{verdict['accounting_sensitivity']['w0_1_category_flip']}**", "",
              "## 4. F-K0와 schedule-seed 안정성", "",
              "| seed | r=0 cache | r=0 retrieval | F5 방향 | r* | category |",
              "|---:|---:|---:|:---:|---:|---|"]
    controls = {row["schedule_seed"]: row for row in verdict["f_k0_control"]["per_seed"]}
    for seed in SCHEDULE_SEEDS:
        control = controls[seed]
        curve = verdict["seed_stability"]["per_seed"][str(seed)]
        lines.append(
            f"| {seed} | {_fmt(control['cache'])} | {_fmt(control['retrieval'])} "
            f"| {'PASS' if control['retrieval_lt_cache'] else 'FAIL'} "
            f"| {curve['r_star']} | {curve['category']} |"
        )
    lines += ["", "## 5. resident sufficiency / clairvoyant(Belady) 분모", "",
              "| r | K-ARC containment | Belady containment | K-ARC/Belady | F-K3 |",
              "|---:|---:|---:|---:|:---:|"]
    for row in verdict["sufficiency"]["cells"]:
        ratio = "N/A" if row["karc_over_clairvoyant"] is None else f"{row['karc_over_clairvoyant']:.2%}"
        lines.append(
            f"| {row['reuse_factor']:.2f} | {row['karc']:.2%} "
            f"| {row['clairvoyant']:.2%} | {ratio} "
            f"| {'TRIGGER' if row['f_k3_triggered'] else 'no'} |"
        )
    lines += ["", "## 6. measured ρ·σ와 stale frontier", "",
              "`ρ`와 `σ`는 통제축이 아니며 cell별 측정치다. r=0은 reused doc이 없어 σ가 정의되지 않는다.", "",
              "| r | measured ρ(all tasks) | measured σ | K-ARC stale | no-outcome stale | classic stale |",
              "|---:|---:|---:|---:|---:|---:|"]
    for reuse in R_GRID:
        unit = _select_cells(cells, session_length=8, budget_pct=5, reuse_factor=reuse)
        rho = sum(row["measured"]["rho_all_tasks"] for row in unit) / len(unit)
        sigma_values = [row["measured"]["sigma"] for row in unit if row["measured"]["sigma"] is not None]
        sigma = "undefined" if not sigma_values else f"{sum(sigma_values)/len(sigma_values):.4f}"
        rates = {arm: sum(row["stale_frontier"][arm]["stale_exposure_rate"]
                          for row in unit) / len(unit) for arm in CACHE_ARMS}
        lines.append(
            f"| {reuse:.2f} | {rho:.4f} | {sigma} | {rates['karc']:.2%} "
            f"| {rates['karc-no-outcome']:.2%} | {rates['classic']:.2%} |"
        )
    stale = verdict["stale_decomposition"]
    lines += ["", (f"σ-defined primary cells aggregate: classic−K-ARC **{stale['classic_minus_karc_tasks']} task**, "
                    f"no-outcome−K-ARC **{stale['no_outcome_minus_karc_tasks']} task**. "
                    "후자의 0은 이 stale-version 지표의 감소가 validity 처리에서 왔음을 "
                    "뜻하며 outcome 층 전체의 무가치 증거로 일반화하지 않는다."), "",
              "## 7. G-E4-ACT·K-ARC 증분·falsification", "",
              (f"- G-E4-ACT Jaccard 분기: **{'PASS' if verdict['g_e4_act_recheck']['pass'] else 'FAIL'}** "
               f"— r>0 primary seed-cell {verdict['g_e4_act_recheck']['passing_cells']}/"
               f"{verdict['g_e4_act_recheck']['cells']} PASS, cell당 divergent task "
               f"{verdict['g_e4_act_recheck']['divergent_tasks_min']}–"
               f"{verdict['g_e4_act_recheck']['divergent_tasks_max']} / 32"),
              f"- r*(K-ARC) / r*(classic): **{verdict['karc_increment']['karc_r_star']} / {verdict['karc_increment']['classic_r_star']}**", ""]
    for key in ("F-K0", "F-K1", "F-K2", "F-K3", "F-K4", "F-K5", "F-K6"):
        row = verdict["falsification"][key]
        lines.append(f"- **{key}**: {'TRIGGERED' if row['triggered'] else 'not triggered'} — {row['disposition']}")
    lines += ["", "## 8. 이상화 bracket과 성분별 편향", "",
              ("rev2 primary는 단일 방향의 ‘이상화 상한’이 아니다. cache-우호 성분은 "
               "doc-identity cache-read가 churn을 무시한다는 점이고, cache-불리 성분은 "
               "`C_init`을 매 세션 재지불하면서 retrieval index-build token을 별도 축으로 둔 점이다. "
               "gold-only miss와 Belady는 명시적 이상화 bracket일 뿐 판정에 쓰지 않았다. "
               "따라서 PASS는 모델 Stage 2의 필요조건이지 충분조건이 아니며, F-K0 통과 후 STOP은 강한 음성 신호다."), "",
              "## 9. 재현 스탬프와 raw", "",
              f"- code_git_hash: `{config['code_git_hash']}`",
              f"- corpus seed/hash: `{config['corpus_seed']}` / `{cells[0]['fixture']['corpus_content_hash']}`",
              f"- schedule seeds: `{config['schedule_seeds']}`",
              f"- config sha256: `{config['sha256']}`",
              f"- cells: `{len(cells)}`; model/embedding calls: `0/0`",
              "- 실행: `PYTHONPATH=src python3 scripts/run_e5_g0.py`", ""]
    for name, digest in sorted(raw_hashes.items()):
        lines.append(f"- `{name}`: `{digest}`")
    lines += ["", ("`history/rev1-invalid-concurrent-prereg/`는 실행 도중 main에 rev2가 "
                    "도착하기 전에 생성된 무효 rev1 산출을 raw 무삭제 원칙으로 보존한 것이다. "
                    "현재 판정에는 사용하지 않았다."), ""]
    return "\n".join(lines)
