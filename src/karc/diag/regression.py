"""Frozen G-DIAG-1 regression checks."""

from __future__ import annotations

from typing import Any, Mapping


E4_V2_EXPECTED_DIVERGENCE = {
    (4000, 0.25): 26,
    (4000, 0.5): 31,
    (4000, 0.75): 31,
    (4000, 1.0): 31,
    (4001, 0.25): 30,
    (4001, 0.5): 31,
    (4001, 0.75): 30,
    (4001, 1.0): 31,
    (4002, 0.25): 31,
    (4002, 0.5): 31,
    (4002, 0.75): 31,
    (4002, 1.0): 31,
}
E4_V2_EXPECTED_QUERIES_PER_CELL = 32
E7_EXPECTED_PRIMARY_K = 5
E7_EXPECTED_QUERIES = 90
E7_EXPECTED_DIVERGENCE = 82


def _close(left: float, right: float, tolerance: float = 5e-12) -> bool:
    return abs(left - right) <= tolerance


def _evaluate_positive_controls(
    positive_controls: Mapping[str, Any],
) -> dict[str, Any]:
    e4 = positive_controls["e4_v2"]
    observed_e4 = {
        (int(cell["schedule_seed"]), float(cell["reuse_factor"])): cell
        for cell in e4["cells"]
    }
    e4_exact_cells = (
        set(observed_e4) == set(E4_V2_EXPECTED_DIVERGENCE)
        and all(
            observed_e4[key]["queries"]
            == E4_V2_EXPECTED_QUERIES_PER_CELL
            and observed_e4[key]["jaccard_lt_1_queries"] == expected
            for key, expected in E4_V2_EXPECTED_DIVERGENCE.items()
        )
    )
    e4_checks = {
        "source_pins": all(e4["source_pins"].values()),
        "twelve_exact_primary_reuse_cells": e4_exact_cells,
        "all_cells_detect_divergence": (
            len(observed_e4) == 12
            and all(
                cell["jaccard_lt_1_queries"] > 0
                for cell in observed_e4.values()
            )
        ),
        "observed_range_26_to_31_of_32": (
            len(observed_e4) == 12
            and min(
                cell["jaccard_lt_1_queries"]
                for cell in observed_e4.values()
            )
            == 26
            and max(
                cell["jaccard_lt_1_queries"]
                for cell in observed_e4.values()
            )
            == 31
        ),
    }

    e7 = positive_controls["e7_lazy_kg"]
    e7_checks = {
        "source_pins": all(e7["source_pins"].values()),
        "regenerated_byte_identical": e7["regenerated_byte_identical"] is True,
        "primary_k_5": e7["primary_k"] == E7_EXPECTED_PRIMARY_K,
        "divergence_82_of_90": (
            e7["queries"] == E7_EXPECTED_QUERIES
            and e7["jaccard_lt_1_queries"] == E7_EXPECTED_DIVERGENCE
        ),
        "provenance_label": (
            e7["provenance_label"] == "reconstructed-not-reproduced"
        ),
    }
    rows = {
        "fixture/e4-v2": {
            "polarity": "positive",
            "expected": {
                "cells": 12,
                "queries_per_cell": E4_V2_EXPECTED_QUERIES_PER_CELL,
                "jaccard_lt_1_by_cell": {
                    f"seed={seed},r={reuse:g}": count
                    for (seed, reuse), count in sorted(
                        E4_V2_EXPECTED_DIVERGENCE.items()
                    )
                },
                "range": [26, 31],
            },
            "observed": {
                "cells": len(observed_e4),
                "jaccard_lt_1_by_cell": {
                    f"seed={seed},r={reuse:g}": (
                        observed_e4[(seed, reuse)][
                            "jaccard_lt_1_queries"
                        ]
                    )
                    for seed, reuse in sorted(observed_e4)
                },
            },
            "checks": e4_checks,
            "pass": all(e4_checks.values()),
        },
        "E7-LAZY-KG": {
            "polarity": "positive",
            "provenance_label": "reconstructed-not-reproduced",
            "expected": {
                "primary_k": E7_EXPECTED_PRIMARY_K,
                "queries": E7_EXPECTED_QUERIES,
                "jaccard_lt_1_queries": E7_EXPECTED_DIVERGENCE,
            },
            "observed": {
                "primary_k": e7["primary_k"],
                "queries": e7["queries"],
                "jaccard_lt_1_queries": e7[
                    "jaccard_lt_1_queries"
                ],
                "mean_jaccard": e7["mean_jaccard"],
            },
            "checks": e7_checks,
            "pass": all(e7_checks.values()),
        },
    }
    return {
        "detector": positive_controls["detector"],
        "matched_rows": sum(row["pass"] for row in rows.values()),
        "required_rows": 2,
        "pass": all(row["pass"] for row in rows.values()),
        "rows": rows,
    }


def evaluate_regressions(
    benchmarks: Mapping[str, Mapping[str, Any]],
    positive_controls: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    locomo = benchmarks["LoCoMo"]
    mab = benchmarks["MemoryAgentBench"]
    memops = benchmarks["MemOps"]
    lme = benchmarks["LongMemEval-V2"]

    locomo_checks = {
        "upper_bound_96_of_1986": (
            locomo["prevalence"]["oracle_derived"]["count"] == 96
            and locomo["prevalence"]["denominator_questions"] == 1986
            and _close(
                locomo["prevalence"]["oracle_derived"]["fraction"],
                96 / 1986,
            )
        ),
        "actionable_50_of_1986": (
            locomo["prevalence"]["actionable_chain"]["count"] == 50
        ),
        "stale_112_of_5882": (
            locomo["prevalence"]["stale_candidate_artifacts"]["count"] == 112
            and locomo["prevalence"]["stale_candidate_artifacts"][
                "denominator_conversation_turns"
            ]
            == 5882
        ),
    }
    mab_validity = mab["family_audit"]["validity-only"]["cells"]
    mab_checks = {
        "native_800_of_3671": (
            mab["prevalence"]["native"]["count"] == 800
            and mab["prevalence"]["denominator_questions"] == 3671
            and _close(mab["prevalence"]["native"]["fraction"], 800 / 3671)
        ),
        "legacy_jaccard_0_of_800_mean_1": all(
            mab_validity[label]["jaccard_lt_1_queries"] == 0
            and mab_validity[label]["queries"] == 800
            and mab_validity[label]["mean_jaccard"] == 1.0
            for label in ("budget-k10", "budget-k20", "base-k50")
        ),
    }
    memops_validity = memops["family_audit"]["validity-only"]["cells"]
    memops_checks = {
        "native_1440_of_4012": (
            memops["prevalence"]["native"]["count"] == 1440
            and memops["prevalence"]["denominator_questions"] == 4012
            and _close(
                memops["prevalence"]["native"]["fraction"], 1440 / 4012
            )
        ),
        "legacy_jaccard_0_of_4012_mean_1": all(
            memops_validity[label]["jaccard_lt_1_queries"] == 0
            and memops_validity[label]["queries"] == 4012
            and memops_validity[label]["mean_jaccard"] == 1.0
            for label in ("budget-k10", "budget-k20", "base-k50")
        ),
    }
    lme_checks = {
        "strict_0_of_451": (
            lme["prevalence"]["strict_structured_ingest"]["count"] == 0
            and lme["prevalence"]["strict_structured_ingest"]["denominator"]
            == 451
        )
    }
    per_benchmark = {
        "LoCoMo": {
            "checks": locomo_checks,
            "pass": all(locomo_checks.values()),
        },
        "MemoryAgentBench": {
            "checks": mab_checks,
            "pass": all(mab_checks.values()),
        },
        "MemOps": {
            "checks": memops_checks,
            "pass": all(memops_checks.values()),
        },
        "LongMemEval-V2": {
            "checks": lme_checks,
            "pass": all(lme_checks.values()),
        },
    }
    result = {
        "gate": "G-DIAG-1",
        "oracle_version": (
            "v2-negative-plus-positive"
            if positive_controls is not None
            else "v1-negative-only-legacy"
        ),
        "matched_benchmarks": sum(
            result["pass"] for result in per_benchmark.values()
        ),
        "required_benchmarks": 4,
        "pass": all(result["pass"] for result in per_benchmark.values()),
        "benchmarks": per_benchmark,
    }
    if positive_controls is not None:
        positive = _evaluate_positive_controls(positive_controls)
        result["positive_controls"] = positive
        result["negative_regression_pass"] = result["pass"]
        result["assay_sensitivity_pass"] = positive["pass"]
        result["pass"] = result["pass"] and positive["pass"]
    return result
