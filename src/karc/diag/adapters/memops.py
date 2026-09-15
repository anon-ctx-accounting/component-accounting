"""MemOps adapter using the pinned session-unit manifest."""

from __future__ import annotations

import subprocess
from functools import cached_property
from pathlib import Path
from typing import Any, Sequence

from karc.diag.adapters.base import (
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


MEMOPS_REVISION = "312af65e2c7b6d1b70f062ffa8b4cde32aaf6f35"


class MemOpsAdapter:
    name = "MemOps"

    def __init__(
        self,
        source_root: Path,
        context_metrics: Path,
        bench_act_metrics: Path,
        *,
        revision_round: int = 0,
    ):
        validate_revision_round(revision_round)
        self.source_root = source_root
        self.context_metrics = context_metrics
        self.bench_act_metrics = bench_act_metrics
        self.revision_round = revision_round
        observed = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if observed != MEMOPS_REVISION:
            raise ValueError(f"MemOps revision mismatch: {observed}")
        self._revision = observed

    @cached_property
    def _frozen(self) -> dict[str, Any]:
        return read_json(self.bench_act_metrics)["MemOps"]

    @cached_property
    def _contexts(self) -> tuple[dict[str, Any], ...]:
        rows = [
            row for row in read_jsonl(self.context_metrics) if row["k"] == 10
        ]
        keys = {(row["setting"], row["context_id"]) for row in rows}
        if len(rows) != 806 or len(keys) != 806:
            raise ValueError("expected 806 frozen MemOps contexts")
        return tuple(
            sorted(rows, key=lambda row: (row["setting"], row["context_id"]))
        )

    @cached_property
    def _stream(self) -> tuple[IngestionRecord, ...]:
        records = []
        for context in self._contexts:
            scope = f"memops:{context['setting']}:{context['context_id']}"
            for index in range(context["ingestion_units"]):
                records.append(
                    IngestionRecord(
                        scope_id=scope,
                        artifact_id=f"{scope}:session:{index + 1}",
                        version=str(index + 1),
                        size_tok=1,
                    )
                )
        return tuple(records)

    @cached_property
    def _queries(self) -> tuple[QueryPoint, ...]:
        points = []
        for context in self._contexts:
            scope = f"memops:{context['setting']}:{context['context_id']}"
            for index in range(context["queries"]):
                points.append(
                    QueryPoint(
                        policy=PolicyQueryView(
                            scope_id=scope,
                            query_id=f"{scope}:q{index + 1}",
                            after_ingest=context["ingestion_units"],
                        ),
                        measurement=MeasurementOnly(category=context["setting"]),
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
            "operation_family_files": census["operation_family_files"],
            "questions_by_operation_family": census[
                "questions_by_operation_family"
            ],
            "questions_by_probe": census["questions_by_probe"],
            "settings": census["settings"],
            "contexts": len(self._contexts),
            "ingestion_units": len(self._stream),
        }

    def prevalence(self) -> dict[str, Any]:
        frozen = self._frozen["B_native_validity"]
        denominator = frozen["denominator_questions"]
        native = frozen["primary_count"]
        oracle_total = frozen["oracle_upper_bound"]["count"]
        return {
            "denominator_questions": denominator,
            "native": {"count": native, "fraction": native / denominator},
            "oracle_derived": {
                "count": oracle_total - native,
                "fraction": (oracle_total - native) / denominator,
                "label": "additional gold-operation upper-bound cases",
            },
            "llm_required": {"count": 0, "fraction": 0.0},
            "unclassified": denominator - oracle_total,
            "actionable_target_binding": {
                "count": 0,
                "reason": "native cues do not bind a resident old/new target",
            },
        }

    def outcome_loop(self) -> dict[str, Any]:
        return dict(self._frozen["D_native_outcome_loop"])

    def mapping_feasibility(self) -> dict[str, Any]:
        return dict(self._frozen["E_mapping"])

    def source_manifest(self) -> dict[str, Any]:
        adjacent = (
            self.source_root / "generated_result" / "2-evidence_conversation"
        )
        longitudinal = (
            self.source_root
            / "generated_result"
            / "4-inject_evidence_with_distractors"
        )
        adjacent_count = len(list(adjacent.glob("*.json")))
        longitudinal_count = len(list(longitudinal.glob("*.json")))
        if (adjacent_count, longitudinal_count) != (403, 403):
            raise ValueError("MemOps official file inventory changed")
        return {
            "source_revision": self._revision,
            "files": [
                {
                    "name": "generated_result/2-evidence_conversation",
                    "json_files": adjacent_count,
                    "read_only": True,
                },
                {
                    "name": "generated_result/4-inject_evidence_with_distractors",
                    "json_files": longitudinal_count,
                    "read_only": True,
                },
            ],
            "adapter_mode": "official git pin plus E5 session-unit manifest",
            "official_source_modified": False,
        }
