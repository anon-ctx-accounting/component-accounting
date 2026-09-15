"""Deterministic signal-density substrate for E8 Lane D2.

The controlled estimand is the fraction of 100 incoming update artifacts
whose predecessor link is resolvable at ingestion time.  Each update lives in
an independent query scope with a fixed, uncounted resident-state scaffold.
This permits the registered endpoints p=0 and p=1 without cyclic lineage.

The four mechanism treatments are the E6-DIAGKIT families and the capacity
grid is the pre-existing E6 grid.  Only the supersession-link assignment
changes across p; access and retrieval signals remain fixed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from karc.diag.adapters.base import standard_capacity_grid
from karc.diag.model import (
    IngestionRecord,
    MechanismFamily,
    NativeSignal,
    PolicyQueryView,
    QueryPoint,
    SignalKind,
    SignalSource,
)
from karc.diag.replay import audit_families


SEED = 20260729
QUERY_SCOPES = 100
SCAFFOLD_ARTIFACTS_PER_SCOPE = 219
UPDATE_ARTIFACTS_PER_SCOPE = 1
P_GRID: tuple[tuple[str, int], ...] = (
    ("0.00", 0),
    ("0.10", 10),
    ("0.25", 25),
    ("0.50", 50),
    ("0.75", 75),
    ("0.90", 90),
    ("1.00", 100),
)

_REFERENCE_IDS = tuple(
    f"item-{index:03d}"
    for index in range(SCAFFOLD_ARTIFACTS_PER_SCOPE - 1)
)
_REFERENCE_SIGNAL = NativeSignal(
    SignalKind.REFERENCE,
    _REFERENCE_IDS,
    SignalSource.ACCESS_NATIVE,
)
_SUPERSEDES_SIGNAL = NativeSignal(
    SignalKind.SUPERSEDES,
    (f"item-{SCAFFOLD_ARTIFACTS_PER_SCOPE - 1:03d}",),
    SignalSource.INGEST_NATIVE,
)
_RETRIEVAL_SIGNAL = NativeSignal(
    SignalKind.RETRIEVAL,
    ("item-000",),
    SignalSource.RETRIEVAL_NATIVE,
)


def _selection_digest(scope_index: int) -> str:
    return hashlib.sha256(
        f"e8-d2:{SEED}:{scope_index}".encode("ascii")
    ).hexdigest()


def scope_permutation() -> tuple[int, ...]:
    """Return the fixed, implementation-independent scope permutation."""

    return tuple(
        sorted(range(QUERY_SCOPES), key=lambda index: (_selection_digest(index), index))
    )


def linked_scope_indices(linked_n: int) -> frozenset[int]:
    if not 0 <= linked_n <= QUERY_SCOPES:
        raise ValueError(f"linked_n must be in 0..{QUERY_SCOPES}")
    return frozenset(scope_permutation()[:linked_n])


@dataclass(frozen=True)
class ControlledDensityAdapter:
    """One p point of the fixed E8-D2 controlled substrate."""

    linked_n: int

    name = "e8-d2-controlled"
    revision_round = 0

    def __post_init__(self) -> None:
        linked_scope_indices(self.linked_n)

    @property
    def linked_scopes(self) -> frozenset[int]:
        return linked_scope_indices(self.linked_n)

    def ingestion_stream(self) -> tuple[IngestionRecord, ...]:
        records: list[IngestionRecord] = []
        for scope_index in range(QUERY_SCOPES):
            scope_id = f"scope-{scope_index:03d}"
            for artifact_index in range(SCAFFOLD_ARTIFACTS_PER_SCOPE):
                records.append(
                    IngestionRecord(
                        scope_id=scope_id,
                        artifact_id=f"item-{artifact_index:03d}",
                        version="1",
                        size_tok=1,
                    )
                )
            signals = [_REFERENCE_SIGNAL]
            if scope_index in self.linked_scopes:
                signals.append(_SUPERSEDES_SIGNAL)
            records.append(
                IngestionRecord(
                    scope_id=scope_id,
                    artifact_id=f"item-{SCAFFOLD_ARTIFACTS_PER_SCOPE:03d}",
                    version="2",
                    size_tok=1,
                    native_signals=tuple(signals),
                )
            )
        return tuple(records)

    def query_points(self) -> tuple[QueryPoint, ...]:
        return tuple(
            QueryPoint(
                PolicyQueryView(
                    scope_id=f"scope-{scope_index:03d}",
                    query_id=f"query-{scope_index:03d}",
                    after_ingest=(
                        SCAFFOLD_ARTIFACTS_PER_SCOPE
                        + UPDATE_ARTIFACTS_PER_SCOPE
                    ),
                    native_signals=(_RETRIEVAL_SIGNAL,),
                )
            )
            for scope_index in range(QUERY_SCOPES)
        )

    def capacity_grid(self):
        return standard_capacity_grid()


def audit_density_point(linked_n: int) -> dict[str, Any]:
    """Run all four preregistered families for one frozen p point."""

    return audit_families(ControlledDensityAdapter(linked_n))


def expected_divergent_jaccard(capacity: int) -> float:
    """Jaccard for one scope when a treatment changes its resident set."""

    if capacity <= 1:
        raise ValueError("controlled capacity must be greater than one")
    return (capacity - 1) / (capacity + 1)


def flatten_sweep(
    point_results: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return exact cell rows and family-level pooled curve rows."""

    cells: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    linked_by_label = dict(P_GRID)
    for p_label, _linked_n in P_GRID:
        family_results = point_results[p_label]
        for family in MechanismFamily:
            family_cells = family_results[family.value]["cells"]
            pooled_queries = 0
            pooled_divergent = 0
            pooled_jaccard_total = 0.0
            for cell_label, metrics in family_cells.items():
                row = {
                    "p": float(p_label),
                    "p_label": p_label,
                    "linked_n": linked_by_label[p_label],
                    "family": family.value,
                    "cell": cell_label,
                    **metrics,
                }
                cells.append(row)
                pooled_queries += metrics["queries"]
                pooled_divergent += metrics["jaccard_lt_1_queries"]
                pooled_jaccard_total += (
                    metrics["mean_jaccard"] * metrics["queries"]
                )
            curves.append(
                {
                    "p": float(p_label),
                    "p_label": p_label,
                    "linked_n": linked_by_label[p_label],
                    "family": family.value,
                    "cell": "pooled",
                    "capacity": None,
                    "queries": pooled_queries,
                    "jaccard_lt_1_queries": pooled_divergent,
                    "jaccard_lt_1_fraction": (
                        pooled_divergent / pooled_queries
                    ),
                    "mean_jaccard": pooled_jaccard_total / pooled_queries,
                }
            )
    return cells, curves


def find_p_star(
    cells: list[dict[str, Any]],
    curves: list[dict[str, Any]],
) -> dict[str, Any]:
    """Find the first observed p meeting the registered 50% definition."""

    threshold = 0.50
    validity = MechanismFamily.VALIDITY.value
    pooled = [
        row
        for row in curves
        if row["family"] == validity
        and row["jaccard_lt_1_fraction"] >= threshold
    ]
    per_cell: dict[str, float | None] = {}
    cell_labels = [cell.label for cell in standard_capacity_grid()]
    for cell_label in cell_labels:
        candidates = [
            row
            for row in cells
            if row["family"] == validity
            and row["cell"] == cell_label
            and row["jaccard_lt_1_fraction"] >= threshold
        ]
        per_cell[cell_label] = (
            min(row["p"] for row in candidates) if candidates else None
        )
    return {
        "definition": (
            "minimum frozen grid p where validity-only differs from the "
            "policy-blind control at >=50% of queries"
        ),
        "threshold": threshold,
        "interpolation": "none",
        "pooled": min(row["p"] for row in pooled) if pooled else None,
        "by_cell": per_cell,
    }


def monotonicity_audit(
    cells: list[dict[str, Any]],
    curves: list[dict[str, Any]],
) -> dict[str, Any]:
    """Report, rather than repair, every adjacent monotonicity violation."""

    rows = cells + curves
    violations: list[dict[str, Any]] = []
    tolerance = 1e-12
    for family in MechanismFamily:
        labels = {
            row["cell"] for row in rows if row["family"] == family.value
        }
        for cell_label in sorted(labels):
            series = sorted(
                (
                    row
                    for row in rows
                    if row["family"] == family.value
                    and row["cell"] == cell_label
                ),
                key=lambda row: row["p"],
            )
            for previous, current in zip(series, series[1:]):
                if (
                    current["jaccard_lt_1_fraction"] + tolerance
                    < previous["jaccard_lt_1_fraction"]
                ):
                    violations.append(
                        {
                            "family": family.value,
                            "cell": cell_label,
                            "metric": "jaccard_lt_1_fraction",
                            "from_p": previous["p"],
                            "to_p": current["p"],
                            "from_value": previous[
                                "jaccard_lt_1_fraction"
                            ],
                            "to_value": current[
                                "jaccard_lt_1_fraction"
                            ],
                        }
                    )
                if (
                    current["mean_jaccard"]
                    > previous["mean_jaccard"] + tolerance
                ):
                    violations.append(
                        {
                            "family": family.value,
                            "cell": cell_label,
                            "metric": "mean_jaccard",
                            "from_p": previous["p"],
                            "to_p": current["p"],
                            "from_value": previous["mean_jaccard"],
                            "to_value": current["mean_jaccard"],
                        }
                    )
    return {
        "expected_direction": {
            "jaccard_lt_1_fraction": "nondecreasing",
            "mean_jaccard": "nonincreasing",
        },
        "adjacent_comparisons": (
            len(tuple(MechanismFamily))
            * (len(standard_capacity_grid()) + 1)
            * (len(P_GRID) - 1)
            * 2
        ),
        "violation_count": len(violations),
        "violations": violations,
        "pass": not violations,
    }

