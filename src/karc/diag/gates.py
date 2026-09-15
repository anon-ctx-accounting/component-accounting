"""Binding E6-DIAGKIT thresholds.

These values are intentionally module constants.  The public gate functions
accept results only; neither adapters nor the CLI can inject replacements.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping

from karc.diag.model import MechanismFamily, ensure_complete_family_set


NATIVE_PREVALENCE_MIN = 0.15
QUERY_DIVERGENCE_FRACTION_MIN = 0.50
MIN_DIVERGENT_CELLS_PER_FAMILY = 2
MIN_DIVERGENT_FAMILIES = 3
MAX_DATA_REVISIONS = 2

THRESHOLDS: Mapping[str, float | int] = MappingProxyType(
    {
        "native_prevalence_min": NATIVE_PREVALENCE_MIN,
        "query_divergence_fraction_min": QUERY_DIVERGENCE_FRACTION_MIN,
        "min_divergent_cells_per_family": MIN_DIVERGENT_CELLS_PER_FAMILY,
        "min_divergent_families": MIN_DIVERGENT_FAMILIES,
        "max_data_revisions": MAX_DATA_REVISIONS,
    }
)


def family_gate(family_results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    families = [MechanismFamily(name) for name in family_results]
    ensure_complete_family_set(families)
    reported: dict[str, Any] = {}
    divergent_family_count = 0
    for family in MechanismFamily:
        cells = family_results[family.value]["cells"]
        divergent_cells = [
            label
            for label, result in cells.items()
            if result["jaccard_lt_1_fraction"]
            >= QUERY_DIVERGENCE_FRACTION_MIN
        ]
        passed = len(divergent_cells) >= MIN_DIVERGENT_CELLS_PER_FAMILY
        divergent_family_count += int(passed)
        reported[family.value] = {
            "divergent_cells": divergent_cells,
            "divergent_cell_count": len(divergent_cells),
            "pass": passed,
        }
    return {
        "families": reported,
        "divergent_family_count": divergent_family_count,
        "required_divergent_families": MIN_DIVERGENT_FAMILIES,
        "pass": divergent_family_count >= MIN_DIVERGENT_FAMILIES,
    }


def benchmark_gate(
    *,
    native_count: int,
    denominator: int,
    native_outcome_loop: bool,
    s2_gate: Mapping[str, Any],
) -> str:
    prevalence = 0.0 if denominator == 0 else native_count / denominator
    if prevalence < NATIVE_PREVALENCE_MIN or not s2_gate["pass"]:
        return "NO-GO"
    if native_outcome_loop:
        return "GO"
    return "CONDITIONAL(validity-only)"


def validate_revision_round(revision_round: int) -> None:
    if not 0 <= revision_round <= MAX_DATA_REVISIONS:
        raise ValueError(
            f"revision_round must be 0..{MAX_DATA_REVISIONS}; "
            "the preregistered data-revision cap cannot be exceeded"
        )
