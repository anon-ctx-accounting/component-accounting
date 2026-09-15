"""Thin, model-free resident-set manipulation checks.

Each mechanism family is a one-signal treatment against the same
policy-blind bounded resident set.  The replay measures treatment effects,
not answer quality: divergence means the signal changed system state.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from karc.diag.adapters.base import BenchmarkAdapter
from karc.diag.model import (
    MechanismFamily,
    NativeSignal,
    PolicyQueryView,
    SignalKind,
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return 1.0 if not union else len(left & right) / len(union)


@dataclass
class _ResidentStore:
    capacity: int
    family: MechanismFamily | None
    registry: Mapping[str, int]

    def __post_init__(self) -> None:
        self.resident: OrderedDict[str, int] = OrderedDict()
        self.frequency: dict[str, int] = {}
        self.sequence: dict[str, int] = {}
        self.capacity_eviction_events = 0
        self.capacity_evicted_items: set[str] = set()
        self._clock = 0

    def _used(self) -> int:
        return sum(self.resident.values())

    def _victim(self) -> str:
        if self.family is MechanismFamily.FREQUENCY:
            return min(
                self.resident,
                key=lambda item: (
                    self.frequency.get(item, 0),
                    self.sequence[item],
                    item,
                ),
            )
        return next(iter(self.resident))

    def _admit(self, artifact_id: str) -> None:
        if artifact_id not in self.registry:
            raise ValueError(f"signal names unknown artifact: {artifact_id}")
        size = self.registry[artifact_id]
        if size > self.capacity:
            return
        if artifact_id in self.resident:
            return
        while self.resident and self._used() + size > self.capacity:
            victim = self._victim()
            self.resident.pop(victim)
            self.capacity_eviction_events += 1
            self.capacity_evicted_items.add(victim)
        self._clock += 1
        self.resident[artifact_id] = size
        self.sequence[artifact_id] = self._clock
        self.frequency.setdefault(artifact_id, 1)

    def ingest(self, artifact_id: str, signals: Iterable[NativeSignal]) -> None:
        signals = tuple(signals)
        if self.family is MechanismFamily.VALIDITY:
            for signal in signals:
                if signal.kind is SignalKind.SUPERSEDES:
                    for old_id in signal.artifact_ids:
                        self.resident.pop(old_id, None)
        # Native reference/retrieval signals attached to an ingestion event are
        # observed before admission, so they can affect the pending eviction.
        self.apply_signals(signals)
        self._admit(artifact_id)

    def apply_signals(self, signals: Iterable[NativeSignal]) -> None:
        if self.family is None:
            return
        for signal in signals:
            if (
                self.family is MechanismFamily.RECENCY
                and signal.kind is SignalKind.REFERENCE
            ):
                for artifact_id in signal.artifact_ids:
                    if artifact_id in self.resident:
                        self.resident.move_to_end(artifact_id)
            elif (
                self.family is MechanismFamily.FREQUENCY
                and signal.kind is SignalKind.REFERENCE
            ):
                for artifact_id in signal.artifact_ids:
                    if artifact_id in self.registry:
                        self.frequency[artifact_id] = (
                            self.frequency.get(artifact_id, 0) + 1
                        )
            elif (
                self.family is MechanismFamily.VALIDITY
                and signal.kind is SignalKind.SUPERSEDES
            ):
                for old_id in signal.artifact_ids:
                    self.resident.pop(old_id, None)
            elif (
                self.family is MechanismFamily.RETRIEVAL
                and signal.kind is SignalKind.RETRIEVAL
            ):
                for artifact_id in signal.artifact_ids:
                    if artifact_id in self.resident:
                        self.resident.move_to_end(artifact_id)
                    else:
                        self._admit(artifact_id)

    def snapshot(self) -> set[str]:
        return set(self.resident)

    def used(self) -> int:
        return self._used()

    def saturated(self) -> bool:
        return self._used() >= self.capacity


def _scope_inputs(adapter: BenchmarkAdapter):
    records: dict[str, list[Any]] = defaultdict(list)
    queries: dict[str, list[Any]] = defaultdict(list)
    for record in adapter.ingestion_stream():
        records[record.scope_id].append(record)
    for point in adapter.query_points():
        queries[point.policy.scope_id].append(point)
    if set(records) != set(queries):
        raise ValueError("every replay scope must have records and queries")
    for scope in queries:
        queries[scope].sort(
            key=lambda point: (point.policy.after_ingest, point.policy.query_id)
        )
        if queries[scope][-1].policy.after_ingest > len(records[scope]):
            raise ValueError(f"query point exceeds stream in {scope}")
    return records, queries


def audit_state_pairs(
    pairs: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Measure state-set divergence with the same detector used by E6 cells.

    Each row must contain ``query_id``, ``control`` and ``treatment``.  The
    latter two values are interpreted as artifact-ID sets.  This small public
    boundary lets positive controls exercise the exact Jaccard detector and
    observation hash used by the benchmark-family replay.
    """

    observations: list[dict[str, Any]] = []
    for pair in pairs:
        left = set(pair["control"])
        right = set(pair["treatment"])
        observations.append(
            {
                "query_id": str(pair["query_id"]),
                "jaccard": _jaccard(left, right),
                "symmetric_difference_n": len(left ^ right),
            }
        )
    if not observations:
        raise ValueError("state-pair audit requires at least one observation")
    divergent = sum(row["jaccard"] < 1.0 for row in observations)
    payload_hash = hashlib.sha256(
        _canonical_json(observations).encode("utf-8")
    ).hexdigest()
    return {
        "queries": len(observations),
        "jaccard_lt_1_queries": divergent,
        "jaccard_lt_1_fraction": divergent / len(observations),
        "mean_jaccard": statistics.fmean(
            row["jaccard"] for row in observations
        ),
        "min_jaccard": min(row["jaccard"] for row in observations),
        "max_symmetric_difference_n": max(
            row["symmetric_difference_n"] for row in observations
        ),
        "observation_sha256": payload_hash,
    }


def _run_cell(
    *,
    records_by_scope: Mapping[str, Sequence[Any]],
    queries_by_scope: Mapping[str, Sequence[Any]],
    capacity: int,
    family: MechanismFamily,
) -> dict[str, Any]:
    pairs: list[dict[str, Any]] = []
    control_eviction_events = 0
    treatment_eviction_events = 0
    control_evicted_items: set[tuple[str, str]] = set()
    treatment_evicted_items: set[tuple[str, str]] = set()
    control_saturated_queries = 0
    treatment_saturated_queries = 0
    unique_stream_items = 0
    unique_artifact_footprint = 0
    scope_footprints: list[int] = []
    for scope in sorted(records_by_scope):
        records = records_by_scope[scope]
        registry = {record.artifact_id: record.size_tok for record in records}
        if len(registry) != len(records):
            raise ValueError(f"duplicate ingestion artifact id in {scope}")
        scope_footprint = sum(registry.values())
        scope_footprints.append(scope_footprint)
        unique_stream_items += len(registry)
        unique_artifact_footprint += scope_footprint
        control = _ResidentStore(capacity, None, registry)
        treatment = _ResidentStore(capacity, family, registry)
        cursor = 0
        for point in queries_by_scope[scope]:
            policy: PolicyQueryView = point.policy
            while cursor < policy.after_ingest:
                record = records[cursor]
                control.ingest(record.artifact_id, ())
                treatment.ingest(record.artifact_id, record.native_signals)
                cursor += 1
            # Only the typed policy view crosses this boundary.
            control.apply_signals(())
            treatment.apply_signals(policy.native_signals)
            left = control.snapshot()
            right = treatment.snapshot()
            control_saturated_queries += int(control.saturated())
            treatment_saturated_queries += int(treatment.saturated())
            pairs.append(
                {
                    "query_id": policy.query_id,
                    "control": left,
                    "treatment": right,
                }
            )
        control_eviction_events += control.capacity_eviction_events
        treatment_eviction_events += treatment.capacity_eviction_events
        control_evicted_items.update(
            (scope, artifact_id)
            for artifact_id in control.capacity_evicted_items
        )
        treatment_evicted_items.update(
            (scope, artifact_id)
            for artifact_id in treatment.capacity_evicted_items
        )
    divergence = audit_state_pairs(pairs)
    query_count = divergence["queries"]
    scope_count = len(scope_footprints)
    classification = (
        "BINDING" if treatment_eviction_events > 0 else "NON-BINDING"
    )
    return {
        **divergence,
        "capacity_binding": {
            "classification": classification,
            "accounting_unit": "IngestionRecord.size_tok",
            "capacity_per_scope": capacity,
            "scope_count": scope_count,
            "unique_stream_items": unique_stream_items,
            "unique_artifact_footprint_w": unique_artifact_footprint,
            "k_over_w": capacity / unique_artifact_footprint,
            "aggregate_scope_capacity": capacity * scope_count,
            "aggregate_scope_capacity_over_w": (
                capacity * scope_count / unique_artifact_footprint
            ),
            "scope_footprint_min": min(scope_footprints),
            "scope_footprint_max": max(scope_footprints),
            "control": {
                "capacity_eviction_events": control_eviction_events,
                "capacity_saturated_queries": control_saturated_queries,
                "capacity_saturated_query_fraction": (
                    control_saturated_queries / query_count
                ),
                "ever_evicted_items": len(control_evicted_items),
                "ever_evicted_item_fraction": (
                    len(control_evicted_items) / unique_stream_items
                ),
            },
            "treatment": {
                "capacity_eviction_events": treatment_eviction_events,
                "capacity_saturated_queries": treatment_saturated_queries,
                "capacity_saturated_query_fraction": (
                    treatment_saturated_queries / query_count
                ),
                "ever_evicted_items": len(treatment_evicted_items),
                "ever_evicted_item_fraction": (
                    len(treatment_evicted_items) / unique_stream_items
                ),
            },
            "control_treatment_eviction_parity": (
                control_eviction_events == treatment_eviction_events
                and control_evicted_items == treatment_evicted_items
            ),
        },
    }


def audit_families(adapter: BenchmarkAdapter) -> dict[str, Any]:
    """Audit all four preregistered families with no caller-selected subset."""

    records, queries = _scope_inputs(adapter)
    families: dict[str, Any] = {}
    for family in MechanismFamily:
        cells: dict[str, Any] = {}
        for cell in adapter.capacity_grid():
            cells[cell.label] = {
                "capacity": cell.capacity,
                "is_base": cell.is_base,
                "legacy_regression": cell.legacy_regression,
                **_run_cell(
                    records_by_scope=records,
                    queries_by_scope=queries,
                    capacity=cell.capacity,
                    family=family,
                ),
            }
        families[family.value] = {"cells": cells}
    return families
