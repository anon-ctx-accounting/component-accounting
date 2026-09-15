"""Frozen selection and task-cluster statistics for the E4b grid.

This module is deliberately model-runtime agnostic.  It consumes normalized
R-9-safe rows and exposes only the preregistered sequential panels at a look.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from karc.bench import checkpoint as cp
from karc.bench.e4_model import ARMS, hash_json, select_smoke_tasks


N_TASK = 165
LOOK_TASKS = (42, 83, 124, 165)
REPS = 3
MARGIN = 0.10
N_BOOT = 10_000
SELECTION_SEED = 4050
ORDER_SEED = 4051
SCHEDULE_SEED = 4052
BOOTSTRAP_SEED = 4053
PRIMARY_COMPARATORS = ("classic", "rag-bm25")
RETAINED_AFTER_EARLY_PASS = (
    "karc", "karc-no-outcome", "classic", "rag-va",
)
DROPPED_AFTER_EARLY_PASS = tuple(
    arm for arm in ARMS if arm not in RETAINED_AFTER_EARLY_PASS
)
SENSITIVITY_BUDGETS = (10, 20)
SENSITIVITY_ARMS = ("karc", "classic", "rag-bm25", "rag-va")


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode()


def select_primary_tasks(tasks: list[dict]) -> tuple[list[dict], dict]:
    """Freeze a smoke-disjoint single-tier selection and execution order."""
    smoke_ids = {task["task_id"] for task in select_smoke_tasks(tasks)}
    eligible = sorted(
        (task for task in tasks if task["task_id"] not in smoke_ids),
        key=lambda task: task["task_id"],
    )
    if len(eligible) < N_TASK:
        raise ValueError(f"only {len(eligible)} smoke-disjoint tasks for n={N_TASK}")
    selected = random.Random(SELECTION_SEED).sample(eligible, N_TASK)

    # Every E4-v2 task has the preregistered `controlled` tier.  Shuffling the
    # sole tier is therefore exactly the specified tier-stratified ordering.
    tiers = {str(task["difficulty_tier"]) for task in selected}
    if tiers != {"controlled"}:
        raise ValueError(f"unexpected E4b difficulty tiers: {sorted(tiers)}")
    random.Random(ORDER_SEED).shuffle(selected)
    snapshot = {
        "schema": "e4b-primary-selection-v1",
        "n_task": N_TASK,
        "look_tasks": list(LOOK_TASKS),
        "selection_seed": SELECTION_SEED,
        "order_seed": ORDER_SEED,
        "model_schedule_seed": SCHEDULE_SEED,
        "selection_rule": "seeded sample without replacement after smoke exclusion",
        "order_rule": "tier-stratified fixed random order; E4-v2 has one controlled tier",
        "smoke_excluded_task_ids": sorted(smoke_ids),
        "eligible_count": len(eligible),
        "task_type_counts": dict(sorted(Counter(
            str(task.get("task_type")) for task in selected
        ).items())),
        "tasks": [
            {
                "order": index,
                "task_id": task["task_id"],
                "seq": task["seq"],
                "difficulty_tier": task["difficulty_tier"],
                "task_type": task.get("task_type"),
                "structure": task["structure"],
                "task_sha256": hash_json(task),
            }
            for index, task in enumerate(selected, start=1)
        ],
    }
    snapshot["sha256"] = hashlib.sha256(_canonical(snapshot)).hexdigest()
    return selected, snapshot


def verify_selection(tasks: list[dict], snapshot: dict) -> list[dict]:
    """Resolve a frozen snapshot and reject fixture or seed drift."""
    unsigned = dict(snapshot)
    recorded = unsigned.pop("sha256")
    if hashlib.sha256(_canonical(unsigned)).hexdigest() != recorded:
        raise ValueError("selection snapshot hash mismatch")
    regenerated, expected = select_primary_tasks(tasks)
    if expected != snapshot:
        raise ValueError("selection snapshot differs from deterministic E4b plan")
    return regenerated


def _latest_graded(rows: list[dict]) -> list[dict]:
    return [row for row in cp.latest_by_run_id(rows).values()
            if row.get("failure_class") in {"ok", "task"}]


def _by_arm_task(rows: list[dict], task_ids: set[str]) -> dict:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in _latest_graded(rows):
        if row.get("task_id") in task_ids:
            grouped[(row["arm"], row["task_id"])].append(row)
    return grouped


def complete_task_ids(rows: list[dict], ordered_task_ids: list[str],
                      arms: tuple[str, ...] = ARMS) -> list[str]:
    """Return the longest task-complete prefix for the specified arms."""
    grouped = _by_arm_task(rows, set(ordered_task_ids))
    complete: list[str] = []
    for task_id in ordered_task_ids:
        if all(len(grouped[(arm, task_id)]) == REPS for arm in arms):
            complete.append(task_id)
        else:
            break
    return complete


def _task_metric(grouped: dict, arm: str, task_ids: list[str],
                 metric: str) -> dict[str, float]:
    result = {}
    for task_id in task_ids:
        rows = grouped.get((arm, task_id), [])
        if len(rows) != REPS:
            continue
        if metric == "success":
            values = [float(bool(row.get("passed"))) for row in rows]
        elif metric == "stale":
            values = [float(row.get("answer_class") == "stale") for row in rows]
        else:
            values = [float(row.get(metric, 0) or 0) for row in rows]
        result[task_id] = statistics.mean(values)
    return result


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("quantile requires values")
    position = probability * (len(ordered) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _metric_seed(label: str) -> int:
    digest = hashlib.sha256(label.encode()).hexdigest()
    return BOOTSTRAP_SEED + int(digest[:8], 16)


def paired_bootstrap_panel(*, task_ids: list[str], left: dict[str, float],
                           right: dict[str, float], orientation: str,
                           nominal_alpha: float, null: float,
                           label: str, n_boot: int = N_BOOT) -> dict:
    """Bootstrap an oriented paired task-cluster mean difference.

    ``left-minus-right`` is used for success. ``right-minus-left`` is used for
    cost and stale rates so positive always means K-ARC is better.
    """
    paired = [(task_id, left[task_id], right[task_id]) for task_id in task_ids
              if task_id in left and task_id in right]
    if orientation == "left-minus-right":
        diffs = [left_value - right_value
                 for _tid, left_value, right_value in paired]
    elif orientation == "right-minus-left":
        diffs = [right_value - left_value
                 for _tid, left_value, right_value in paired]
    else:
        raise ValueError(f"unknown orientation: {orientation}")
    if not diffs:
        return {
            "n_task_clusters": 0, "point": None, "lower": None, "upper": None,
            "nominal_one_sided_alpha": nominal_alpha, "null": null,
            "efficacy": False, "reversal": False,
        }
    rng = random.Random(_metric_seed(label))
    n = len(diffs)
    boot = [statistics.mean(diffs[rng.randrange(n)] for _ in range(n))
            for _ in range(n_boot)]
    lower = _quantile(boot, nominal_alpha)
    upper = _quantile(boot, 1.0 - nominal_alpha)
    return {
        "n_task_clusters": n,
        "point": statistics.mean(diffs),
        "lower": lower,
        "upper": upper,
        "nominal_one_sided_alpha": nominal_alpha,
        "confidence_level_one_sided": 1.0 - nominal_alpha,
        "null": null,
        "orientation": orientation,
        "n_boot": n_boot,
        "bootstrap_seed_base": BOOTSTRAP_SEED,
        "bootstrap_label": label,
        "efficacy": lower > null,
        "reversal": upper < null,
    }


def sequential_look(*, rows: list[dict], tasks: list[dict], look_index: int,
                    boundary_config: dict) -> dict:
    """Produce the only inspectable output at an E4b sequential look."""
    if look_index < 1 or look_index > len(LOOK_TASKS):
        raise ValueError("look index outside frozen schedule")
    look_n = LOOK_TASKS[look_index - 1]
    cumulative_tasks = tasks[:look_n]
    task_ids = [task["task_id"] for task in cumulative_tasks]
    grouped = _by_arm_task(rows, set(task_ids))
    missing = [
        {"arm": arm, "task_id": task_id, "graded": len(grouped[(arm, task_id)])}
        for task_id in task_ids for arm in ARMS
        if len(grouped[(arm, task_id)]) != REPS
    ]
    if missing:
        raise ValueError(f"look {look_index} is not task-complete: {missing[:3]}")
    boundary = boundary_config["efficacy"]["looks"][look_index - 1]
    alpha = float(boundary["nominal_one_sided_alpha"])
    stale_ids = [task["task_id"] for task in cumulative_tasks
                 if task.get("forbidden_versions")]
    metrics = {
        arm: {
            "success": _task_metric(grouped, arm, task_ids, "success"),
            "cost": _task_metric(grouped, arm, task_ids,
                                 "api_input_tokens_no_cache"),
            "stale": _task_metric(grouped, arm, stale_ids, "stale"),
        }
        for arm in ("karc", *PRIMARY_COMPARATORS)
    }
    comparisons = {}
    components = []
    for comparator in PRIMARY_COMPARATORS:
        panels = {
            "success_noninferiority": paired_bootstrap_panel(
                task_ids=task_ids, left=metrics["karc"]["success"],
                right=metrics[comparator]["success"],
                orientation="left-minus-right", nominal_alpha=alpha,
                null=-MARGIN, label=f"look{look_index}:{comparator}:success",
            ),
            "c2_total_input_superiority": paired_bootstrap_panel(
                task_ids=task_ids, left=metrics["karc"]["cost"],
                right=metrics[comparator]["cost"],
                orientation="right-minus-left", nominal_alpha=alpha,
                null=0.0, label=f"look{look_index}:{comparator}:cost",
            ),
            "l5_stale_superiority": paired_bootstrap_panel(
                task_ids=stale_ids, left=metrics["karc"]["stale"],
                right=metrics[comparator]["stale"],
                orientation="right-minus-left", nominal_alpha=alpha,
                null=0.0, label=f"look{look_index}:{comparator}:stale",
            ),
        }
        components.extend(panels.values())
        comparisons[comparator] = {
            "components": panels,
            "registered_comparator_pass": (
                panels["success_noninferiority"]["efficacy"]
                and (panels["c2_total_input_superiority"]["efficacy"]
                     or panels["l5_stale_superiority"]["efficacy"])
            ),
            "passing_advantage_axes": [
                axis for axis in ("c2_total_input_superiority",
                                  "l5_stale_superiority")
                if panels[axis]["efficacy"]
            ],
        }
    any_reversal = any(panel["reversal"] for panel in components)
    all_efficacy = all(panel["efficacy"] for panel in components)
    primary_conjunction = all(
        comparisons[comparator]["registered_comparator_pass"]
        for comparator in PRIMARY_COMPARATORS
    )
    if any_reversal:
        action = "STOP_FAIL"
    elif all_efficacy:
        action = "STOP_PASS_DROP_ARMS"
    elif look_index == len(LOOK_TASKS):
        action = "STOP_PASS" if primary_conjunction else "STOP_FAIL"
    else:
        action = "CONTINUE"
    result = {
        "schema": "e4b-sequential-look-v1",
        "look_index": look_index,
        "task_count": look_n,
        "information_fraction": look_n / N_TASK,
        "z_boundary": boundary["z_boundary"],
        "nominal_one_sided_alpha": alpha,
        "comparisons": comparisons,
        "six_component_all_efficacy": all_efficacy,
        "six_component_any_reversal": any_reversal,
        "registered_primary_conjunction_at_boundary": primary_conjunction,
        "action": action,
        "binding": True,
        "point_estimate_bias_warning": action.startswith("STOP"),
        "row_authority": "normalized R-9-safe rows; no model content persisted",
    }
    result["sha256"] = hashlib.sha256(_canonical(result)).hexdigest()
    return result


def fixed_paired_panel(*, task_ids: list[str], left: dict[str, float],
                       right: dict[str, float], orientation: str,
                       label: str) -> dict:
    return paired_bootstrap_panel(
        task_ids=task_ids, left=left, right=right, orientation=orientation,
        nominal_alpha=0.05, null=0.0, label=f"fixed:{label}",
    )


def descriptive_analysis(*, rows: list[dict], tasks: list[dict],
                         lifecycle: list[dict], supplies: dict) -> dict:
    """H2/H3 and full arm summaries at the binding stop sample."""
    task_ids = [task["task_id"] for task in tasks]
    grouped = _by_arm_task(rows, set(task_ids))
    life_by_arm_task: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in lifecycle:
        if row["task_id"] in set(task_ids):
            life_by_arm_task[(row["arm"], row["task_id"])].append(row)

    def metric(arm: str, name: str) -> dict[str, float]:
        if name in {"success", "api_input_tokens_no_cache"}:
            return _task_metric(grouped, arm, task_ids, name)
        if name == "l5":
            return {
                task_id: statistics.mean(
                    float(row.get("lifecycle") == "L5")
                    for row in life_by_arm_task[(arm, task_id)]
                )
                for task_id in task_ids
                if len(life_by_arm_task[(arm, task_id)]) == REPS
            }
        raise ValueError(name)

    task_map = {task["task_id"]: task for task in tasks}

    def supply_metric(arm: str, kind: str) -> dict[str, float]:
        result = {}
        for task_id in task_ids:
            task = task_map[task_id]
            injected = set(supplies[arm][task_id])
            if kind == "harmful_block" and task.get("task_type") == "harmful":
                result[task_id] = float(not (set(task["forbidden_versions"]) & injected))
            elif kind == "rehab_coverage" and task.get("task_type") == "rehabilitation":
                result[task_id] = float(set(task["required_versions"]) <= injected)
        return result

    all_metrics = {
        arm: {name: metric(arm, name) for name in (
            "success", "api_input_tokens_no_cache", "l5",
        )}
        for arm in ARMS
    }

    def decomposition(left_arm: str, right_arm: str, label: str) -> dict:
        return {
            "success": fixed_paired_panel(
                task_ids=task_ids, left=all_metrics[left_arm]["success"],
                right=all_metrics[right_arm]["success"],
                orientation="left-minus-right", label=f"{label}:success",
            ),
            "c2_total_input_advantage": fixed_paired_panel(
                task_ids=task_ids,
                left=all_metrics[left_arm]["api_input_tokens_no_cache"],
                right=all_metrics[right_arm]["api_input_tokens_no_cache"],
                orientation="right-minus-left", label=f"{label}:c2",
            ),
            "l5_stale_advantage": fixed_paired_panel(
                task_ids=task_ids, left=all_metrics[left_arm]["l5"],
                right=all_metrics[right_arm]["l5"],
                orientation="right-minus-left", label=f"{label}:l5",
            ),
            "harmful_block_rate": fixed_paired_panel(
                task_ids=task_ids,
                left=supply_metric(left_arm, "harmful_block"),
                right=supply_metric(right_arm, "harmful_block"),
                orientation="left-minus-right", label=f"{label}:harmful-block",
            ),
            "rehab_required_coverage": fixed_paired_panel(
                task_ids=task_ids,
                left=supply_metric(left_arm, "rehab_coverage"),
                right=supply_metric(right_arm, "rehab_coverage"),
                orientation="left-minus-right", label=f"{label}:rehab-coverage",
            ),
        }

    by_arm = {}
    for arm in ARMS:
        arm_rows = [row for row in _latest_graded(rows)
                    if row.get("arm") == arm and row.get("task_id") in set(task_ids)]
        life_rows = [row for row in lifecycle if row["arm"] == arm]
        dist = Counter((row.get("lifecycle") or "success") for row in life_rows)
        by_arm[arm] = {
            "graded_runs": len(arm_rows),
            "task_clusters": len(all_metrics[arm]["success"]),
            "success_rate": (sum(bool(row.get("passed")) for row in arm_rows)
                             / len(arm_rows) if arm_rows else None),
            "answer_class": dict(sorted(Counter(
                str(row.get("answer_class")) for row in arm_rows
            ).items())),
            "c1_injected_tokens_mean": (statistics.mean(
                float(row.get("injected_knowledge_tokens", 0)) for row in arm_rows
            ) if arm_rows else None),
            "c2_input_no_cache_mean": (statistics.mean(
                float(row.get("api_input_tokens_no_cache", 0)) for row in arm_rows
            ) if arm_rows else None),
            "c2_input_with_cache_mean": (statistics.mean(
                float(row.get("api_input_tokens_with_cache", 0)) for row in arm_rows
            ) if arm_rows else None),
            "c3_tool_calls_mean": (statistics.mean(
                float(row.get("tool_calls", 0)) for row in life_rows
            ) if life_rows else None),
            "c4_run_wall_time_s_mean": (statistics.mean(
                float((row.get("driver_metadata") or {}).get("run_wall_time_s", 0))
                for row in arm_rows
            ) if arm_rows else None),
            "turns_mean": (statistics.mean(
                float(row.get("num_turns", 0) or 0) for row in arm_rows
            ) if arm_rows else None),
            "lifecycle": {key: dist.get(key, 0)
                          for key in ("success", "L1", "L2", "L3", "L5", "L6")},
            "l4_stale_exposure": sum(bool(row.get("stale_exposure"))
                                     for row in life_rows),
            "unresolved": sum(int(row.get("unresolved", 0)) for row in life_rows),
        }

    jaccards = []
    for task_id in task_ids:
        left = set(supplies["karc"][task_id])
        right = set(supplies["classic"][task_id])
        union = left | right
        jaccards.append(len(left & right) / len(union) if union else 1.0)
    return {
        "h2": {
            "outcome_contribution_karc_minus_no_outcome": decomposition(
                "karc", "karc-no-outcome", "outcome"
            ),
            "validity_contribution_no_outcome_minus_classic": decomposition(
                "karc-no-outcome", "classic", "validity"
            ),
        },
        "h3_karc_vs_rag_va": decomposition("karc", "rag-va", "h3"),
        "solar_adm_vs_karc_input": decomposition(
            "karc", "solar-adm", "solar-adm"
        ),
        "by_arm": by_arm,
        "lifecycle_rows": len(lifecycle),
        "jaccard_karc_classic": {
            "n": len(jaccards),
            "divergent_tasks": sum(value < 1.0 for value in jaccards),
            "minimum": min(jaccards) if jaccards else None,
            "mean": statistics.mean(jaccards) if jaccards else None,
        },
    }


def sensitivity_analysis(*, rows: list[dict], tasks: list[dict],
                         budget_pct: int) -> dict:
    """Fixed-n secondary P1/P2 direction panel; never promotes primary."""
    task_ids = [task["task_id"] for task in tasks]
    grouped = _by_arm_task(rows, set(task_ids))
    expected = len(task_ids) * REPS
    counts = {
        arm: sum(len(grouped[(arm, task_id)]) for task_id in task_ids)
        for arm in SENSITIVITY_ARMS
    }
    if any(counts[arm] != expected for arm in SENSITIVITY_ARMS):
        raise ValueError(f"incomplete c={budget_pct} sensitivity: {counts}")
    stale_ids = [task["task_id"] for task in tasks
                 if task.get("forbidden_versions")]
    metrics = {
        arm: {
            "success": _task_metric(grouped, arm, task_ids, "success"),
            "cost": _task_metric(grouped, arm, task_ids,
                                 "api_input_tokens_no_cache"),
            "stale": _task_metric(grouped, arm, stale_ids, "stale"),
        }
        for arm in SENSITIVITY_ARMS
    }
    comparisons = {}
    for comparator in PRIMARY_COMPARATORS:
        success = paired_bootstrap_panel(
            task_ids=task_ids, left=metrics["karc"]["success"],
            right=metrics[comparator]["success"],
            orientation="left-minus-right", nominal_alpha=.05,
            null=-MARGIN, label=f"sensitivity-c{budget_pct}:{comparator}:success",
        )
        cost = paired_bootstrap_panel(
            task_ids=task_ids, left=metrics["karc"]["cost"],
            right=metrics[comparator]["cost"],
            orientation="right-minus-left", nominal_alpha=.05,
            null=0.0, label=f"sensitivity-c{budget_pct}:{comparator}:cost",
        )
        stale = paired_bootstrap_panel(
            task_ids=stale_ids, left=metrics["karc"]["stale"],
            right=metrics[comparator]["stale"],
            orientation="right-minus-left", nominal_alpha=.05,
            null=0.0, label=f"sensitivity-c{budget_pct}:{comparator}:stale",
        )
        comparisons[comparator] = {
            "success_noninferiority": success,
            "c2_total_input_superiority": cost,
            "l5_stale_superiority": stale,
            "directional_conjunction": (
                success["efficacy"] and (cost["efficacy"] or stale["efficacy"])
            ),
        }
    return {
        "schema": "e4b-budget-sensitivity-fixed-n-v1",
        "budget_pct": budget_pct, "n_task": len(tasks), "reps": REPS,
        "secondary_only_no_primary_promotion": True,
        "graded_by_arm": counts,
        "comparisons": comparisons,
    }


def load_jsonl(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows
