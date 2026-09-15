"""E4 model-stage gates, fixture freeze, supplies, and smoke statistics.

All persisted model observations are normalized IDs, counts, usage, hashes,
and deterministic grades.  Prompts live only in the versioned fixture; model
output and transcripts are never written (R-9).
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import shutil
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

from karc.bench import checkpoint as cp
from karc.bench.e3_retrieval import (
    BM25_B,
    BM25_K1,
    bm25_plans,
    chunks_sha256,
)
from karc.bench.e4_fixture import EMBED_MODEL_ID, EMBED_REVISION, MAX_TURNS
from karc.bench.e4_instrument import (
    FetchAccess,
    classify_lifecycle,
    exposure_set,
    refs_for_versions,
)
from karc.bench.e4_replay import load_confirmed_config, replay_cell
from karc.bench.e4_retrieval import RAG_VA_RULES, in_memory_chunks, rag_va_plans
from karc.bench.harness import Arm
from karc.replay.runner import git_hash


SEED = 4000
REPS = 3
CANARY_SCHEDULE_SEED = 4041
TRAP_SCHEDULE_SEED = 4042
SMOKE_SELECTION_SEED = 4040
SMOKE_SCHEDULE_SEED = 4043
SMOKE_TASKS = 15
TRAP_PROBES = 20
ARMS = (
    "karc", "karc-no-outcome", "classic", "rag-bm25", "rag-va",
    "solar-adm", "static-full", "lru", "karc+va",
)
VAR_COMPARATORS = ("classic", "rag-bm25")
DEFAULT_SUCCESS_MARGIN = 0.05
N_TASK_FLOOR = 124
N_TASK_CAP = 640


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode()


def hash_json(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def right_size_from_variance(max_variance: float, margin: float) -> tuple[int, int]:
    """Return (uncapped, registered floor/cap result) for an E4b margin."""
    if not 0 < margin < 1:
        raise ValueError("success margin must be between 0 and 1")
    uncapped = math.ceil(6.185 * max_variance / (margin ** 2))
    return uncapped, min(N_TASK_CAP, max(N_TASK_FLOOR, uncapped))


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                               indent=2) + "\n", encoding="utf-8")


def load_fixture(path: str | Path, *, require_frozen: bool = False):
    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    tasks = json.loads((root / "tasks.json").read_text(encoding="utf-8"))
    build = json.loads((root / "BUILD.json").read_text(encoding="utf-8"))
    if manifest.get("seed") != SEED or build.get("seed") != SEED:
        raise SystemExit("fixture seed differs from E4 contract")
    unsigned = dict(manifest)
    recorded = unsigned.pop("manifest_sha256")
    if hash_json(unsigned) != recorded or build.get("manifest_sha256") != recorded:
        raise SystemExit("E4 manifest hash verification failed")
    if hash_json(tasks) != build.get("tasks_sha256"):
        raise SystemExit("E4 task hash verification failed")
    corpus_payload: list[str] = []
    for version_id, entry in sorted(manifest["artifacts"].items(),
                                    key=lambda item: item[1]["path"]):
        doc = root / "repo" / entry["path"]
        if not doc.is_file() or sha256_file(doc) != entry["sha256"]:
            raise SystemExit(f"E4 artifact hash mismatch: {version_id}")
        corpus_payload.append(f"{entry['path']}\0{entry['sha256']}")
    if hashlib.sha256("\n".join(corpus_payload).encode()).hexdigest() != manifest["corpus_content_hash"]:
        raise SystemExit("E4 corpus content hash verification failed")
    if require_frozen:
        freeze = json.loads((root / "freeze.json").read_text(encoding="utf-8"))
        if not freeze.get("pass") or freeze["manifest_sha256"] != recorded:
            raise SystemExit("E4 fixture is not frozen or freeze hash differs")
    return root, manifest, tasks, build


def reproduction_stamp(repo_root: Path, fixture_root: Path, manifest: dict,
                       build: dict, execution: dict) -> dict:
    stamp = {
        "code_git_hash": git_hash(repo_root),
        "fixture_root": str(fixture_root),
        "fixture_seed": manifest["seed"],
        "corpus_content_hash": manifest["corpus_content_hash"],
        "manifest_sha256": manifest["manifest_sha256"],
        "tasks_sha256": build["tasks_sha256"],
        "execution_config_sha256": hash_json(execution),
    }
    freeze_path = fixture_root / "freeze.json"
    if freeze_path.is_file():
        stamp["freeze_sha256"] = sha256_file(freeze_path)
        stamp["retrieval_config_sha256"] = sha256_file(
            fixture_root / "retrieval" / "config.json"
        )
        stamp["retrieval_plans_sha256"] = sha256_file(
            fixture_root / "retrieval" / "plans.json"
        )
    return stamp


def _latest_graded(rows: list[dict]) -> list[dict]:
    return [row for row in cp.latest_by_run_id(rows).values()
            if row.get("failure_class") in {"ok", "task"}]


def canary_verdict(rows: list[dict], tasks: list[dict], reps: int = REPS) -> dict:
    latest = list(cp.latest_by_run_id(rows).values())
    graded = [row for row in latest if row.get("failure_class") in {"ok", "task"}]
    by_task: dict[str, list[int]] = defaultdict(list)
    for row in graded:
        by_task[row["task_id"]].append(int(bool(row.get("passed"))))
    task_rates = {
        task["task_id"]: (sum(by_task[task["task_id"]]) / reps
                          if len(by_task[task["task_id"]]) == reps else None)
        for task in tasks
    }
    flagged = sorted(tid for tid, rate in task_rates.items()
                     if rate is not None and rate > 0.05)
    uuid_runs = sorted(row["run_id"] for row in graded
                       if row.get("leakage") or row.get("canaries_found"))
    successes = sum(bool(row.get("passed")) for row in graded)
    expected = len(tasks) * reps
    overall = successes / expected if expected else 0.0
    complete = len(graded) == expected and all(rate is not None for rate in task_rates.values())
    passed = complete and not flagged and overall <= 0.02 and not uuid_runs
    return {
        "gate": "G-E4-CANARY", "pass": passed, "complete": complete,
        "graded_runs": len(graded), "expected_runs": expected,
        "successes": successes, "overall_success_rate": overall,
        "thresholds": {"per_task_max": 0.05, "overall_max": 0.02,
                       "uuid_output_max": 0},
        "task_success_rate": task_rates, "flagged_tasks": flagged,
        "uuid_leakage_runs": uuid_runs,
        "fixture_action": "none" if passed else "revise-or-discard flagged tasks then re-canary",
    }


def trap_probe_tasks(manifest: dict, count: int = TRAP_PROBES) -> tuple[list[dict], dict[str, list[str]], dict]:
    explicit_unique_pool = "trap_candidates" in manifest["groups"]
    versions = list(manifest["groups"].get(
        "trap_candidates", manifest["groups"]["versioned_v1"],
    ))
    if explicit_unique_pool and len(versions) < count:
        raise ValueError(f"fixture has only {len(versions)} unique trap candidates")
    versions = sorted(versions)
    selected = (random.Random(TRAP_SCHEDULE_SEED).sample(versions, count)
                if explicit_unique_pool else
                [versions[index % len(versions)] for index in range(count)])
    if not selected:
        raise ValueError("fixture has no v1 trap candidates")
    tasks: list[dict] = []
    supplies: dict[str, list[str]] = {}
    sources: list[dict] = []
    for index, version_id in enumerate(selected):
        entry = manifest["artifacts"][version_id]
        task_id = f"E4-TP-{index + 1:03d}"
        task = {
            "task_id": task_id, "seq": index + 1, "session_id": "E4-TRAP",
            "language": "en", "difficulty_tier": "controlled-trap",
            "structure": "v1-only-potency-probe", "max_turns": 1,
            "required_artifacts": [entry["artifact_id"]],
            "required_versions": [version_id], "forbidden_versions": [],
            "stale_values": [], "pre_events": [], "outcome_inject": [],
            "prompt": (
                f"Using only the supplied document, return its operational value for "
                f"`{entry['fact_key']}` as exactly `{entry['fact_key']} = <value>`."
            ),
            "answer_fact": {"artifact_id": entry["artifact_id"],
                            "version_id": version_id, "key": entry["fact_key"],
                            "value": entry["value"]},
            "format_regex": (
                rf"\A\s*{re.escape(entry['fact_key'])}\s*=\s*"
                r"E4-[0-9A-F]{4}-[0-9A-F]{4}\s*\Z"
            ),
        }
        tasks.append(task)
        supplies[task_id] = [version_id]
        sources.append({"probe_id": task_id, "source_version_id": version_id,
                        "source_task_sha256": hash_json(task)})
    snapshot = {
        "schema": "e4-trap-probe-selection-v2", "n_probe_items": count,
        "n_unique_source_versions": len(set(v[0] for v in supplies.values())),
        "selection_seed": TRAP_SCHEDULE_SEED,
        "selection": ("seeded sample without replacement from frozen trap_candidates"
                      if explicit_unique_pool else
                      "legacy sorted versioned_v1 round-robin"),
        "sources": sources,
    }
    return tasks, supplies, snapshot


def trap_verdict(rows: list[dict], tasks: list[dict], reps: int = REPS) -> dict:
    graded = _latest_graded(rows)
    expected = len(tasks) * reps
    successes = sum(bool(row.get("passed")) for row in graded)
    potency = successes / expected if expected else 0.0
    by_probe: dict[str, list[int]] = defaultdict(list)
    for row in graded:
        by_probe[row["task_id"]].append(int(bool(row.get("passed"))))
    probe_rates = {task["task_id"]: sum(by_probe[task["task_id"]]) / reps
                   if len(by_probe[task["task_id"]]) == reps else None
                   for task in tasks}
    low = sorted(tid for tid, rate in probe_rates.items()
                 if rate is not None and rate < 0.70)
    complete = len(graded) == expected and all(rate is not None for rate in probe_rates.values())
    return {
        "gate": "G-ACT-5(ii)-TRAP-POTENCY", "pass": complete and potency >= 0.70,
        "complete": complete, "graded_runs": len(graded), "expected_runs": expected,
        "v1_answer_successes": successes, "potency": potency, "threshold": 0.70,
        "probe_success_rate": probe_rates, "below_threshold_probe_aliases": low,
        "fixture_action": "none" if complete and potency >= 0.70
                          else "arm-neutral trap revision then re-probe and affected-task re-canary",
    }


def select_smoke_tasks(tasks: list[dict]) -> list[dict]:
    if any(task.get("task_type") in {
            "corrected", "harmful", "rehabilitation", "plain-stale",
    } for task in tasks) and len(tasks) >= 640:
        strata = (
            ("harmful-current-forbidden", 4),
            ("rehab-post-validated", 4),
            ("hot-supersession-trap", 3),
            ("corrected-demote", 2),
            ("rehab-validated-by-injector", 2),
        )
        rng = random.Random(SMOKE_SELECTION_SEED)
        selected: list[dict] = []
        for structure, count in strata:
            candidates = sorted(
                (task for task in tasks if task.get("structure") == structure),
                key=lambda task: task["task_id"],
            )
            if len(candidates) < count:
                raise ValueError(f"smoke stratum {structure} has {len(candidates)} < {count}")
            selected.extend(rng.sample(candidates, count))
        return sorted(selected, key=lambda task: int(task["seq"]))
    traps = [task for task in tasks if task.get("structure") == "hot-supersession-trap"]
    if len(traps) > SMOKE_TASKS:
        raise ValueError("more trap tasks than smoke slots")
    remaining = [task for task in tasks if task not in traps]
    chosen = random.Random(SMOKE_SELECTION_SEED).sample(
        remaining, SMOKE_TASKS - len(traps),
    )
    return sorted([*traps, *chosen], key=lambda task: int(task["seq"]))


def smoke_selection_snapshot(tasks: list[dict]) -> dict:
    structures = Counter(task["structure"] for task in tasks)
    return {
        "schema": "e4-smoke-selection-v2", "selection_seed": SMOKE_SELECTION_SEED,
        "rule": ("A-B3 stratified seeded selection" if any(
                 task.get("task_type") in {"corrected", "harmful", "rehabilitation", "plain-stale"}
                 for task in tasks)
                 else "include every primary-cell hot-supersession trap, then seeded sample without replacement"),
        "structure_counts": dict(sorted(structures.items())),
        "tasks": [{"task_id": task["task_id"], "seq": task["seq"],
                   "tier": task["difficulty_tier"], "structure": task["structure"],
                   "task_sha256": hash_json(task)} for task in tasks],
    }


def freeze_fixture(draft_root: str | Path, frozen_root: str | Path, *,
                   canary_verdict_path: str | Path,
                   trap_verdict_path: str | Path) -> dict:
    draft, manifest, tasks, build = load_fixture(draft_root)
    canary = json.loads(Path(canary_verdict_path).read_text(encoding="utf-8"))
    trap = json.loads(Path(trap_verdict_path).read_text(encoding="utf-8"))
    if not canary.get("pass") or not trap.get("pass"):
        raise SystemExit("canary and trap potency must pass before fixture freeze")
    out = Path(frozen_root)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite frozen fixture path: {out}")
    shutil.copytree(draft, out)

    frozen_manifest = dict(manifest)
    contract = dict(frozen_manifest["fixture_contract"])
    fixture_version = out.name
    contract.update({"status": "frozen-after-canary-and-trap-potency",
                     "fixture_version": fixture_version})
    frozen_manifest["fixture_contract"] = contract
    frozen_manifest["fixture_version"] = fixture_version
    frozen_manifest.pop("manifest_sha256", None)
    frozen_manifest["manifest_sha256"] = hash_json(frozen_manifest)
    _write_json(out / "manifest.json", frozen_manifest)

    entries_with_text = {}
    for version_id, entry in frozen_manifest["artifacts"].items():
        row = dict(entry)
        row["text"] = (out / "repo" / entry["path"]).read_text(encoding="utf-8")
        entries_with_text[version_id] = row
    chunks = in_memory_chunks(entries_with_text)
    budget = int(frozen_manifest["cell"]["budget_tokens"])
    plans = {
        "rag-bm25": bm25_plans(chunks, tasks, frozen_manifest, budget),
        "rag-va": rag_va_plans(chunks, tasks, frozen_manifest, budget),
    }
    retrieval_dir = out / "retrieval"
    retrieval_dir.mkdir(parents=True)
    _write_json(retrieval_dir / "chunks.json", [asdict(chunk) for chunk in chunks])
    _write_json(retrieval_dir / "plans.json", plans)
    retrieval_config = {
        "schema": "e4-retrieval-freeze-v1",
        "chunking": {"strategy": "one-document-one-chunk", "count": len(chunks),
                     "sha256": chunks_sha256(chunks)},
        "budget": {"tokens": budget, "selector": "whole-artifact-greedy"},
        "rag-bm25": {"k1": BM25_K1, "b": BM25_B, "tuning": "none"},
        "rag-va": dict(RAG_VA_RULES),
        "embedder_pin": {"model_id": EMBED_MODEL_ID, "revision": EMBED_REVISION,
                         "used_by_e4b_smoke": False},
        "plans_sha256": sha256_file(retrieval_dir / "plans.json"),
    }
    _write_json(retrieval_dir / "config.json", retrieval_config)

    frozen_build = dict(build)
    frozen_build.update({
        "status": "frozen — canary and trap potency passed",
        "fixture_version": fixture_version,
        "manifest_sha256": frozen_manifest["manifest_sha256"],
        "retrieval_plans_sha256": sha256_file(retrieval_dir / "plans.json"),
        "retrieval_config_sha256": sha256_file(retrieval_dir / "config.json"),
    })
    freeze = {
        "schema": "e4-fixture-freeze-v2", "pass": True,
        "fixture_version": fixture_version,
        "seed": SEED, "path": str(out), "n_tasks": len(tasks),
        "corpus_content_hash": frozen_manifest["corpus_content_hash"],
        "manifest_sha256": frozen_manifest["manifest_sha256"],
        "tasks_sha256": frozen_build["tasks_sha256"],
        "retrieval_plans_sha256": frozen_build["retrieval_plans_sha256"],
        "retrieval_config_sha256": frozen_build["retrieval_config_sha256"],
        "embedder": {"model_id": EMBED_MODEL_ID, "revision": EMBED_REVISION},
        "rag_va_rules": dict(RAG_VA_RULES),
        "canary_verdict_sha256": sha256_file(canary_verdict_path),
        "trap_verdict_sha256": sha256_file(trap_verdict_path),
        "smoke_selection": smoke_selection_snapshot(select_smoke_tasks(tasks)),
        "post_freeze_mutation": "prohibited",
    }
    _write_json(out / "freeze.json", freeze)
    frozen_build["freeze_sha256"] = sha256_file(out / "freeze.json")
    _write_json(out / "BUILD.json", frozen_build)
    load_fixture(out, require_frozen=True)
    return freeze


def _solar_admission(tasks: list[dict], manifest: dict, budget: int) -> dict[str, list[str]]:
    """Frozen fidelity-limited SOLAR-like Bayesian utility admission.

    Beta(1,1) posterior reuse probability divided by document size is ranked
    deterministically.  It observes only the common past read stream and does
    not consume validity or outcome labels.
    """
    hits: Counter[str] = Counter()
    first_seen: dict[str, int] = {}
    plans: dict[str, list[str]] = {}
    for task in tasks:
        seq = int(task["seq"])
        scored = []
        for version_id, count in hits.items():
            opportunities = max(1, seq - first_seen[version_id])
            posterior = (count + 1.0) / (opportunities + 2.0)
            size = int(manifest["artifacts"][version_id]["size_tok"])
            scored.append((-(posterior / size), version_id, size))
        used = 0
        chosen: list[str] = []
        for _neg_utility, version_id, size in sorted(scored):
            if used + size <= budget:
                chosen.append(version_id)
                used += size
        plans[task["task_id"]] = chosen
        for version_id in task["required_versions"]:
            first_seen.setdefault(version_id, seq)
            hits[version_id] += 1
    return plans


def build_supplies(repo_root: Path, fixture_root: Path, manifest: dict,
                   tasks: list[dict]) -> tuple[dict[str, dict[str, list[str]]], dict]:
    confirmed = load_confirmed_config(repo_root)
    _cell, replay_rows = replay_cell(
        rho=float(manifest["cell"]["rho"]), sigma=float(manifest["cell"]["sigma"]),
        budget_pct=int(manifest["cell"]["budget_pct"]), confirmed_config=confirmed,
        fixture_manifest=manifest, fixture_tasks=tasks,
    )
    replay_by_task = {row["task_id"]: row for row in replay_rows}
    retrieval = json.loads((fixture_root / "retrieval" / "plans.json").read_text(encoding="utf-8"))
    all_versions = sorted(manifest["artifacts"], key=lambda version_id: manifest["artifacts"][version_id]["path"])
    supplies: dict[str, dict[str, list[str]]] = {arm: {} for arm in ARMS}
    solar = _solar_admission(tasks, manifest, int(manifest["cell"]["budget_tokens"]))
    for task in tasks:
        tid = task["task_id"]
        row = replay_by_task[tid]["arms"]
        supplies["karc"][tid] = row["karc"]["working_set_versions"]
        supplies["karc-no-outcome"][tid] = row["karc-no-outcome"]["working_set_versions"]
        supplies["classic"][tid] = row["classic"]["working_set_versions"]
        supplies["lru"][tid] = row["lru"]["working_set_versions"]
        supplies["rag-bm25"][tid] = retrieval["rag-bm25"][tid]["artifact_ids"]
        supplies["rag-va"][tid] = retrieval["rag-va"][tid]["artifact_ids"]
        supplies["solar-adm"][tid] = solar[tid]
        supplies["static-full"][tid] = all_versions
        karc = supplies["karc"][tid]
        if set(task["required_versions"]) <= set(karc):
            supplies["karc+va"][tid] = list(karc)
        else:
            supplies["karc+va"][tid] = list(dict.fromkeys(
                [*karc, *supplies["rag-va"][tid]]
            ))
    config = {
        "working_set_source": "deterministic full-stream E4 replay before smoke selection",
        "solar_adm": {"family": "fidelity-limited SOLAR-like reconstruction",
                      "prior": "Beta(1,1)", "score": "posterior_reuse_probability/size_tok",
                      "observes": "past common read stream only", "tuning": "none"},
        "karc_plus_va": "K-ARC residents; on required-version miss append frozen rag-va plan",
        "supply_sha256": {arm: hash_json(plans) for arm, plans in supplies.items()},
    }
    return supplies, config


def build_sensitivity_supplies(
        repo_root: Path, fixture_root: Path, manifest: dict,
        tasks: list[dict], *, budget_pct: int,
) -> tuple[dict[str, dict[str, list[str]]], dict]:
    """Replay the frozen E4-v2 corpus at a preregistered sensitivity budget.

    The fixture, prompts, chunks, retrieval rules, rho, and sigma are retained.
    Only ``c`` and its mechanically derived token count change.  This helper
    writes nothing to the fixture and intentionally exposes only the four
    sensitivity arms registered by E4b.
    """
    if budget_pct not in {10, 20}:
        raise ValueError("E4b grid work-order permits sensitivity c=10 or c=20")
    total_tokens = sum(int(entry["size_tok"])
                       for entry in manifest["artifacts"].values())
    budget = max(1, int(total_tokens * budget_pct / 100.0))
    sensitivity_manifest = json.loads(json.dumps(manifest))
    sensitivity_manifest["cell"] = {
        **manifest["cell"], "budget_pct": budget_pct,
        "budget_tokens": budget,
    }
    confirmed = load_confirmed_config(repo_root)
    _cell, replay_rows = replay_cell(
        rho=float(manifest["cell"]["rho"]),
        sigma=float(manifest["cell"]["sigma"]),
        budget_pct=budget_pct, confirmed_config=confirmed,
        fixture_manifest=sensitivity_manifest, fixture_tasks=tasks,
    )
    replay_by_task = {row["task_id"]: row for row in replay_rows}
    entries_with_text = {}
    for version_id, entry in manifest["artifacts"].items():
        row = dict(entry)
        row["text"] = (fixture_root / "repo" / entry["path"]).read_text(
            encoding="utf-8"
        )
        entries_with_text[version_id] = row
    chunks = in_memory_chunks(entries_with_text)
    retrieval = {
        "rag-bm25": bm25_plans(chunks, tasks, sensitivity_manifest, budget),
        "rag-va": rag_va_plans(chunks, tasks, sensitivity_manifest, budget),
    }
    supplies = {arm: {} for arm in (
        "karc", "classic", "rag-bm25", "rag-va",
    )}
    for task in tasks:
        task_id = task["task_id"]
        row = replay_by_task[task_id]["arms"]
        supplies["karc"][task_id] = row["karc"]["working_set_versions"]
        supplies["classic"][task_id] = row["classic"]["working_set_versions"]
        supplies["rag-bm25"][task_id] = retrieval["rag-bm25"][task_id]["artifact_ids"]
        supplies["rag-va"][task_id] = retrieval["rag-va"][task_id]["artifact_ids"]
    config = {
        "schema": "e4b-budget-sensitivity-supply-v1",
        "fixture_manifest_sha256": manifest["manifest_sha256"],
        "rho": manifest["cell"]["rho"], "sigma": manifest["cell"]["sigma"],
        "budget_pct": budget_pct, "budget_tokens": budget,
        "budget_derivation": "floor(sum(frozen artifact size_tok) * c / 100)",
        "chunking": "frozen one-document-one-chunk rule",
        "retrieval": {"rag-bm25": {"k1": BM25_K1, "b": BM25_B},
                      "rag-va": dict(RAG_VA_RULES)},
        "tuning": "none",
        "supply_sha256": {arm: hash_json(plan)
                          for arm, plan in supplies.items()},
    }
    return supplies, config


def build_arms(supplies: dict[str, dict[str, list[str]]]) -> list[Arm]:
    modes = {
        "karc": "karc", "karc-no-outcome": "karc", "classic": "classic",
        "rag-bm25": "rag-bm25", "rag-va": "rag-bm25", "solar-adm": "karc",
        "static-full": "static-full", "lru": "recency", "karc+va": "karc",
    }
    return [Arm(name, modes[name], working_set_by_task=supplies[name],
                max_turns=MAX_TURNS, enforce_tool_cap=True) for name in ARMS]


def _guard_index(path: Path) -> dict[str, dict]:
    index: dict[str, dict] = defaultdict(lambda: {
        "documents": {}, "unresolved": 0, "blocked_unresolved": 0,
        "tool_calls": 0, "session_start": False,
    })
    if not path.is_file():
        return index
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rid = row.get("run_id")
            if not rid:
                continue
            item = index[rid]
            if row.get("hook_event_name") == "SessionStart":
                item["session_start"] = True
            if row.get("hook_event_name") != "PreToolUse":
                continue
            item["tool_calls"] += 1
            item["unresolved"] += int(row.get("unresolved_document_accesses", 0))
            item["blocked_unresolved"] += int(
                row.get("blocked_unresolved_document_accesses", 0)
            )
            for doc in row.get("resolved_documents", []):
                item["documents"][doc["version_id"]] = doc
    return index


def lifecycle_rows(rows: list[dict], tasks: list[dict], manifest: dict,
                   supplies: dict[str, dict[str, list[str]]],
                   guard_path: Path) -> list[dict]:
    task_map = {task["task_id"]: task for task in tasks}
    guards = _guard_index(guard_path)
    values: list[dict] = []
    for row in _latest_graded(rows):
        if row["arm"] not in supplies or row["task_id"] not in task_map:
            continue
        task = task_map[row["task_id"]]
        guard = guards[row["run_id"]]
        fetches = [FetchAccess(doc["artifact_id"], doc["version_id"],
                                doc["access_type"], int(doc.get("seq") or 0))
                   for doc in guard["documents"].values()]
        injected = supplies[row["arm"]][row["task_id"]]
        exposure = exposure_set(manifest, working_set=injected, fetches=fetches,
                                require_resolved=False)
        required = refs_for_versions(manifest, task["required_versions"])
        stale = refs_for_versions(manifest, task["forbidden_versions"])
        value_match = ("expected" if row["answer_class"] == "correct" else
                       "stale" if row["answer_class"] == "stale" else "other")
        classification = classify_lifecycle(
            required=required, stale=stale, exposure=exposure,
            value_match=value_match, format_ok=bool(row.get("format_ok")),
            abstained=bool(row.get("abstained")), format_is_failure=False,
        )
        values.append({
            "run_id": row["run_id"], "arm": row["arm"],
            "task_id": row["task_id"], "rep": row["rep"],
            "injected_version_ids": sorted(injected),
            "fetched_version_ids": sorted(guard["documents"]),
            "exposed_version_ids": sorted(ref.version_id for ref in exposure),
            "tool_calls": guard["tool_calls"], "unresolved": guard["unresolved"],
            "blocked_unresolved": guard["blocked_unresolved"],
            "actual_fetch_observable": guard["session_start"],
            "format_ok": row.get("format_ok"), **asdict(classification),
        })
    return values


def write_lifecycle(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in sorted(values, key=lambda value: value["run_id"]):
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def smoke_verdict(rows: list[dict], tasks: list[dict], all_tasks: list[dict],
                  manifest: dict, supplies: dict[str, dict[str, list[str]]],
                  guard_path: Path, reps: int = REPS,
                  margin: float = DEFAULT_SUCCESS_MARGIN) -> tuple[dict, list[dict]]:
    graded = _latest_graded(rows)
    wanted = {task["task_id"] for task in tasks}
    graded = [row for row in graded if row["arm"] in ARMS and row["task_id"] in wanted]
    expected_per_arm = len(tasks) * reps
    by_arm_task: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in graded:
        by_arm_task[(row["arm"], row["task_id"])].append(row)
    counts = {arm: sum(len(by_arm_task[(arm, tid)]) for tid in wanted) for arm in ARMS}
    complete = all(counts[arm] == expected_per_arm for arm in ARMS) and all(
        len(by_arm_task[(arm, tid)]) == reps for arm in ARMS for tid in wanted
    )
    arm_success = {
        arm: {"successes": sum(bool(row["passed"]) for tid in wanted
                               for row in by_arm_task[(arm, tid)]),
              "runs": counts[arm]}
        for arm in ARMS
    }
    for value in arm_success.values():
        value["rate"] = value["successes"] / value["runs"] if value["runs"] else None
    rates = [value["rate"] for value in arm_success.values() if value["rate"] is not None]
    spread = max(rates) - min(rates) if rates else 0.0

    tiers = sorted({task["difficulty_tier"] for task in tasks})
    oracle = {}
    for tier in tiers:
        tier_ids = {task["task_id"] for task in tasks if task["difficulty_tier"] == tier}
        oracle_rows = [row for tid in tier_ids for row in by_arm_task[("static-full", tid)]]
        rate = sum(bool(row["passed"]) for row in oracle_rows) / len(oracle_rows) if oracle_rows else 0.0
        oracle[tier] = {"successes": sum(bool(row["passed"]) for row in oracle_rows),
                        "runs": len(oracle_rows), "rate": rate, "pass": rate >= 0.70}

    differences: dict[str, list[float]] = {}
    variances: dict[str, float] = {}
    for comparator in VAR_COMPARATORS:
        ds = []
        for task in tasks:
            tid = task["task_id"]
            left = [int(bool(row["passed"])) for row in by_arm_task[("karc", tid)]]
            right = [int(bool(row["passed"])) for row in by_arm_task[(comparator, tid)]]
            if len(left) == reps and len(right) == reps:
                ds.append(statistics.mean(left) - statistics.mean(right))
        differences[comparator] = ds
        variances[comparator] = statistics.variance(ds) if len(ds) > 1 else 0.0
    max_variance = max(variances.values(), default=0.0)
    uncapped, n_task = right_size_from_variance(max_variance, margin)

    life = lifecycle_rows(graded, tasks, manifest, supplies, guard_path)
    lifecycle_by_arm = {}
    for arm in ARMS:
        unit = [row for row in life if row["arm"] == arm]
        counts_l = Counter(row["lifecycle"] or "success" for row in unit)
        lifecycle_by_arm[arm] = {
            "distribution": {key: counts_l.get(key, 0)
                             for key in ("success", "L1", "L2", "L3", "L5", "L6")},
            "L4_stale_exposure": sum(row["stale_exposure"] for row in unit),
            "format_failures": sum(row.get("format_ok") is False for row in unit),
            "tool_calls": sum(row["tool_calls"] for row in unit),
            "unresolved": sum(row["unresolved"] for row in unit),
            "blocked_unresolved": sum(row["blocked_unresolved"] for row in unit),
        }
    l4_total = sum(row["stale_exposure"] for row in life)
    l5_total = sum(row["lifecycle"] == "L5" for row in life)
    unresolved_total = sum(row["unresolved"] for row in life)
    blocked_unresolved_total = sum(row["blocked_unresolved"] for row in life)
    observable = len(life) == len(graded) and all(row["actual_fetch_observable"] for row in life)

    task_map = {task["task_id"]: task for task in tasks}
    outcome_task_ids = {
        task["task_id"] for task in tasks
        if task.get("task_type") in {"harmful", "rehabilitation"}
    }
    supply_divergent = sorted(
        tid for tid in outcome_task_ids
        if set(supplies["karc"][tid]) != set(supplies["karc-no-outcome"][tid])
    )
    divergent_by_type = Counter(task_map[tid]["task_type"] for tid in supply_divergent)
    harmful_tasks = [task for task in tasks
                     if task["structure"] == "harmful-current-forbidden"]
    harmful_blocked = []
    for task in harmful_tasks:
        forbidden = set(task["forbidden_versions"])
        if (not (forbidden & set(supplies["karc"][task["task_id"]]))
                and forbidden & set(supplies["karc-no-outcome"][task["task_id"]])):
            harmful_blocked.append(task["task_id"])
    rehab_tasks = [task for task in tasks
                   if task["structure"] in {
                       "rehab-validated-by-injector", "rehab-post-validated",
                   }]
    rehab_coverage = {arm: sum(
        set(task["required_versions"]) <= set(supplies[arm][task["task_id"]])
        for task in rehab_tasks
    ) for arm in ("karc", "karc-no-outcome")}
    target_lifecycle = {}
    for arm in ("karc", "karc-no-outcome"):
        unit = [row for row in life
                if row["arm"] == arm and row["task_id"] in outcome_task_ids]
        dist = Counter(row["lifecycle"] or "success" for row in unit)
        target_lifecycle[arm] = {
            "runs": len(unit),
            "success": dist.get("success", 0),
            "L2": dist.get("L2", 0),
            "L5": dist.get("L5", 0),
            "L4_stale_exposure": sum(row["stale_exposure"] for row in unit),
        }
    outcome_contract_applies = manifest.get("generator_profile") == "balanced-pool-v2"
    outcome_activation_pass = (bool(supply_divergent and harmful_blocked)
                               if outcome_contract_applies else True)
    outcome_layer = {
        "contract_applies": outcome_contract_applies,
        "target_task_count": len(outcome_task_ids),
        "supply_divergent_tasks": len(supply_divergent),
        "supply_divergent_task_ids": supply_divergent,
        "supply_divergent_by_type": dict(sorted(divergent_by_type.items())),
        "harmful_probe_tasks": len(harmful_tasks),
        "harmful_blocked_by_karc_retained_by_no_outcome": len(harmful_blocked),
        "harmful_blocked_task_ids": harmful_blocked,
        "rehab_probe_tasks": len(rehab_tasks),
        "rehab_required_coverage": rehab_coverage,
        "target_lifecycle": target_lifecycle,
        "activation_pass": outcome_activation_pass,
    }
    oracle_pass = all(value["pass"] for value in oracle.values())
    discrimination_pass = spread >= 0.05
    signal_pass = l4_total > 0 and l5_total > 0
    instrument_pass = observable and unresolved_total == 0
    variance_pass = uncapped <= N_TASK_CAP
    smoke_pass = (complete and oracle_pass and discrimination_pass and signal_pass
                  and instrument_pass and variance_pass and outcome_activation_pass)
    capacity = len(all_tasks) >= n_task
    looks = {str(t): {"tasks": math.ceil(n_task * t),
                      "runs": 9 * math.ceil(n_task * t) * reps}
             for t in (0.25, 0.5, 0.75, 1.0)}
    verdict = {
        "gate": "G-E4b-SMOKE", "pass": smoke_pass, "grid_ready": smoke_pass and capacity,
        "complete": complete, "graded_by_arm": counts, "arm_success": arm_success,
        "oracle_per_tier": oracle, "oracle_pass": oracle_pass,
        "discrimination": {"max_minus_min": spread, "threshold": 0.05,
                           "pass": discrimination_pass},
        "lifecycle": {"by_arm": lifecycle_by_arm, "L4_total": l4_total,
                      "L5_total": l5_total, "nonempty_pass": signal_pass},
        "instrument": {"actual_fetch_observable_all": observable,
                       "unresolved_total": unresolved_total,
                       "blocked_unresolved_total": blocked_unresolved_total,
                       "pass": instrument_pass},
        "outcome_layer": outcome_layer,
        "paired_task_differences": differences, "variance_d": variances,
        "max_variance_d": max_variance, "n_task_uncapped": uncapped,
        "n_task": n_task,
        "success_margin": margin,
        "n_task_formula": (
            f"min({N_TASK_CAP}, max({N_TASK_FLOOR}, "
            f"ceil(6.185 * max Var(d) / {margin ** 2:g})))"
        ),
        "variance_within_cap": variance_pass,
        "frozen_task_pool": len(all_tasks), "fixture_capacity_pass": capacity,
        "primary_grid": {"arms": 9, "reps": reps, "full_runs": 9 * n_task * reps,
                         "sequential_looks": looks,
                         "expected_if_E3_scale_effect": [looks["0.25"]["runs"],
                                                         looks["0.5"]["runs"]]},
        "grid_opinion": (
            "eligible for separate go-ahead" if smoke_pass and capacity else
            "not eligible: frozen task pool is smaller than right-sized n_task"
            if smoke_pass and not capacity else
            "not eligible: one or more smoke gates failed"
        ),
    }
    return verdict, life
