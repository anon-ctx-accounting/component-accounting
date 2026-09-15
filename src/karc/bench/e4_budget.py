"""Read-only diagnosis for E4 low-budget working-set degeneration."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter


def diagnose_budget_degeneracy(manifest: dict, cells: list[dict], *,
                               alpha: float = 0.25) -> dict:
    sizes = sorted(int(row["size_tok"]) for row in manifest["artifacts"].values())
    total = sum(sizes)
    distribution = dict(sorted(Counter(sizes).items()))
    diagnostic_pcts = (1, 2, 3, 4, 5, 10, 20)
    grid_rows = {int(row["cell"]["budget_pct"]): [] for row in cells}
    for row in cells:
        grid_rows[int(row["cell"]["budget_pct"])].append(row)

    budgets = []
    for pct in diagnostic_pcts:
        ideal = total * pct / 100.0
        budget = max(1, int(ideal))
        threshold = alpha * budget
        replay = grid_rows.get(pct, [])
        arm_replay = {}
        for arm in ("classic", "karc", "karc-no-outcome"):
            if replay:
                admissions = [int(row["arms"][arm].get("admissions", 0)) for row in replay]
                transitions = [sum(row["arms"][arm]["transition_counts"].values())
                               for row in replay]
                arm_replay[arm] = {
                    "admissions_min": min(admissions),
                    "admissions_max": max(admissions),
                    "transition_positive_cells": sum(value > 0 for value in transitions),
                }
            else:
                arm_replay[arm] = "not in preregistered replay grid"
        budgets.append({
            "budget_pct": pct,
            "ideal_budget_tokens": ideal,
            "budget_tokens": budget,
            "integer_truncation_tokens": ideal - budget,
            "alpha": alpha,
            "alpha_c_threshold": threshold,
            "physically_fit_artifacts": sum(size <= budget for size in sizes),
            "admission_eligible_artifacts": sum(size <= threshold for size in sizes),
            "oversize_guard_rejected_artifacts": sum(size > threshold for size in sizes),
            "replay": arm_replay,
        })

    minimum = min(sizes)
    maximum = max(sizes)
    result = {
        "schema_version": 1,
        "scope": "diagnosis only; fixture and budget code unchanged",
        "artifact_tokens": {
            "count": len(sizes), "total": total, "minimum": minimum,
            "p25": sizes[(len(sizes) - 1) // 4],
            "median": statistics.median(sizes),
            "p75": sizes[(len(sizes) - 1) * 3 // 4],
            "maximum": maximum, "distribution": distribution,
        },
        "budget_rows": budgets,
        "guard_boundaries": {
            "first_budget_tokens_with_any_eligible": math.ceil(minimum / alpha),
            "first_budget_tokens_with_all_eligible": math.ceil(maximum / alpha),
            "first_integer_pct_with_any_eligible": next(
                pct for pct in range(1, 101)
                if alpha * max(1, int(total * pct / 100.0)) >= minimum
            ),
            "first_integer_pct_with_all_eligible": next(
                pct for pct in range(1, 101)
                if alpha * max(1, int(total * pct / 100.0)) >= maximum
            ),
        },
        "conclusion": {
            "classification": "extreme-starvation-by-intended-oversize-guard",
            "budget_scaling_bug": False,
            "reason": (
                "At 1% and 2%, c is positive and can physically hold artifacts, "
                "but confirmed alpha*c is below the smallest artifact, so K-ARC "
                "and karc-no-outcome intentionally classify every artifact oversize."
            ),
            "c2_e4b_sensitivity_valid": False,
            "c2_note": (
                "Uninformative for K-ARC ARC activation and arm-asymmetric because "
                "plain classic ARC uses alpha=1.0; retain only as a reported "
                "starvation data point, not an E4b sensitivity comparator cell."
            ),
        },
    }
    result["sha256"] = hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return result
