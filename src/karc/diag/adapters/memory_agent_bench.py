"""MemoryAgentBench adapter using the frozen official chunk manifest."""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import Any, Sequence

from karc.diag.adapters.base import (
    checked_file,
    read_json,
    read_jsonl,
    standard_capacity_grid,
)
from karc.diag.gates import validate_revision_round
from karc.diag.model import (
    CapacityCell,
    IngestionRecord,
    MeasurementOnly,
    PolicyQueryView,
    QueryPoint,
)


CONFLICT_PARQUET_SHA256 = (
    "24d5c3f09ce0ce15625cb9f8a98f44f0d864ca6c94d7b4ad04eb697ca3a5ff45"
)


class MemoryAgentBenchAdapter:
    name = "MemoryAgentBench"

    def __init__(
        self,
        conflict_parquet: Path,
        context_metrics: Path,
        bench_act_metrics: Path,
        *,
        revision_round: int = 0,
    ):
        validate_revision_round(revision_round)
        self.conflict_parquet = conflict_parquet
        self.context_metrics = context_metrics
        self.bench_act_metrics = bench_act_metrics
        self.revision_round = revision_round
        self._source_file = checked_file(
            conflict_parquet, CONFLICT_PARQUET_SHA256
        )

    @cached_property
    def _frozen(self) -> dict[str, Any]:
        return read_json(self.bench_act_metrics)["MemoryAgentBench"]

    @cached_property
    def _contexts(self) -> tuple[dict[str, Any], ...]:
        rows = [
            row for row in read_jsonl(self.context_metrics) if row["k"] == 10
        ]
        if len(rows) != 8 or len({row["context_id"] for row in rows}) != 8:
            raise ValueError("expected eight frozen conflict contexts")
        return tuple(sorted(rows, key=lambda row: row["context_id"]))

    @cached_property
    def _stream(self) -> tuple[IngestionRecord, ...]:
        records = []
        for context in self._contexts:
            scope = f"mab:{context['context_id']}"
            for index in range(context["ingestion_units"]):
                records.append(
                    IngestionRecord(
                        scope_id=scope,
                        artifact_id=f"{scope}:chunk:{index + 1}",
                        version=str(index + 1),
                        size_tok=1,
                    )
                )
        return tuple(records)

    @cached_property
    def _queries(self) -> tuple[QueryPoint, ...]:
        points = []
        for context in self._contexts:
            scope = f"mab:{context['context_id']}"
            for index in range(context["queries"]):
                points.append(
                    QueryPoint(
                        policy=PolicyQueryView(
                            scope_id=scope,
                            query_id=f"{scope}:q{index + 1}",
                            after_ingest=context["ingestion_units"],
                        ),
                        measurement=MeasurementOnly(
                            category="Conflict Resolution"
                        ),
                    )
                )
        return tuple(points)

    def ingestion_stream(self) -> Sequence[IngestionRecord]:
        return self._stream

    def query_points(self) -> Sequence[QueryPoint]:
        return self._queries

    def capacity_grid(self) -> Sequence[CapacityCell]:
        return standard_capacity_grid()

    def category_census(self) -> dict[str, Any]:
        census = self._frozen["A_category_composition"]
        return {
            "denominator_questions": census["denominator_questions"],
            "categories": census["categories"],
            "conflict_contexts": len(self._contexts),
            "conflict_ingestion_units": len(self._stream),
        }

    def prevalence(self) -> dict[str, Any]:
        frozen = self._frozen["B_native_validity"]
        denominator = frozen["denominator_questions"]
        native = frozen["primary_count"]
        oracle_extra = frozen["oracle_upper_bound"]["extra_count"]
        return {
            "denominator_questions": denominator,
            "native": {"count": native, "fraction": native / denominator},
            "oracle_derived": {
                "count": oracle_extra,
                "fraction": oracle_extra / denominator,
                "label": "additional knowledge-update query metadata",
            },
            "llm_required": {"count": 0, "fraction": 0.0},
            "unclassified": denominator - native - oracle_extra,
            "actionable_target_binding": {
                "count": 0,
                "reason": "serial order has no same-slot old-artifact pointer",
            },
        }

    def outcome_loop(self) -> dict[str, Any]:
        return dict(self._frozen["D_native_outcome_loop"])

    def mapping_feasibility(self) -> dict[str, Any]:
        return dict(self._frozen["E_mapping"])

    def source_manifest(self) -> dict[str, Any]:
        return {
            "source_revision": "455306dcabc3842526eb83cd4e225e5d486c5c5d",
            "dataset_revision": "7ea066982b140a19337e17e60d45d4076e042faf",
            "files": [self._source_file],
            "adapter_mode": (
                "official parquet hash plus E5 official-chunker unit manifest"
            ),
            "official_source_modified": False,
        }
