"""LongMemEval-V2 adapter over pinned questions and small haystacks."""

from __future__ import annotations

import json
from collections import Counter
from functools import cached_property
from pathlib import Path
from typing import Any, Sequence

from karc.diag.adapters.base import checked_file, standard_capacity_grid
from karc.diag.gates import validate_revision_round
from karc.diag.model import (
    CapacityCell,
    IngestionRecord,
    MeasurementOnly,
    PolicyQueryView,
    QueryPoint,
)


QUESTIONS_SHA256 = "0a3ae5ebea938c24d7800e1e0b0828e08ae1646f939a53853b2b8cdc08e292b7"
SMALL_HAYSTACK_SHA256 = (
    "9b5301defb23a088a5f06e45ff8d5f35e569d78305a66d492046a9fff9b46593"
)


class LongMemEvalV2Adapter:
    name = "LongMemEval-V2"

    def __init__(self, data_root: Path, *, revision_round: int = 0):
        validate_revision_round(revision_round)
        self.data_root = data_root
        self.questions_path = data_root / "questions.jsonl"
        self.haystack_path = data_root / "haystacks" / "lme_v2_small.json"
        self.revision_round = revision_round
        self._source_files = [
            checked_file(self.questions_path, QUESTIONS_SHA256),
            checked_file(self.haystack_path, SMALL_HAYSTACK_SHA256),
        ]

    @cached_property
    def _questions(self) -> tuple[dict[str, Any], ...]:
        rows = []
        with self.questions_path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                # Explicit allow-list: question text and gold answer are dropped.
                rows.append(
                    {
                        "id": row["id"],
                        "question_type": row["question_type"],
                        "domain": row["domain"],
                    }
                )
        if len(rows) != 451:
            raise ValueError("expected 451 LongMemEval-V2 questions")
        return tuple(rows)

    @cached_property
    def _haystacks(self) -> dict[str, list[str]]:
        payload = json.loads(self.haystack_path.read_text(encoding="utf-8"))
        return {str(key): list(value) for key, value in payload.items()}

    @cached_property
    def _stream(self) -> tuple[IngestionRecord, ...]:
        records = []
        for question in self._questions:
            scope = f"lme:{question['id']}"
            trajectory_ids = self._haystacks[question["id"]]
            for index, trajectory_id in enumerate(trajectory_ids, start=1):
                records.append(
                    IngestionRecord(
                        scope_id=scope,
                        artifact_id=f"lme:trajectory:{trajectory_id}",
                        version=str(index),
                        size_tok=1,
                    )
                )
        return tuple(records)

    @cached_property
    def _queries(self) -> tuple[QueryPoint, ...]:
        return tuple(
            QueryPoint(
                policy=PolicyQueryView(
                    scope_id=f"lme:{question['id']}",
                    query_id=f"lme:{question['id']}:q",
                    after_ingest=len(self._haystacks[question["id"]]),
                ),
                measurement=MeasurementOnly(
                    category=question["question_type"],
                    heldout_metadata={"domain": question["domain"]},
                ),
            )
            for question in self._questions
        )

    def ingestion_stream(self) -> Sequence[IngestionRecord]:
        return self._stream

    def query_points(self) -> Sequence[QueryPoint]:
        return self._queries

    def capacity_grid(self) -> Sequence[CapacityCell]:
        return standard_capacity_grid()

    def category_census(self) -> dict[str, Any]:
        categories = Counter(row["question_type"] for row in self._questions)
        domains = Counter(row["domain"] for row in self._questions)
        return {
            "denominator_questions": len(self._questions),
            "categories": dict(sorted(categories.items())),
            "domains": dict(sorted(domains.items())),
            "small_haystack_ingestion_units": len(self._stream),
        }

    def prevalence(self) -> dict[str, Any]:
        denominator = len(self._questions)
        oracle = sum(
            row["question_type"]
            in {"dynamic-environment", "dynamic-environment-abs"}
            for row in self._questions
        )
        return {
            "denominator_questions": denominator,
            "native": {"count": 0, "fraction": 0.0},
            "oracle_derived": {
                "count": oracle,
                "fraction": oracle / denominator,
                "label": "dynamic query metadata, unavailable at ingest",
            },
            "llm_required": {"count": 0, "fraction": 0.0},
            "unclassified": denominator - oracle,
            "strict_structured_ingest": {"count": 0, "denominator": denominator},
        }

    def outcome_loop(self) -> dict[str, Any]:
        return {
            "native": False,
            "status": "N/A",
            "native_trajectory_outcome_field": True,
            "reason": "Past outcomes are input attributes, not current-QA feedback.",
        }

    def mapping_feasibility(self) -> dict[str, Any]:
        return {
            "llm_free": False,
            "identity_and_order": True,
            "native_same_slot_target": False,
            "reason": "No ingest field binds old and new values for one target.",
        }

    def source_manifest(self) -> dict[str, Any]:
        return {
            "source_revision": "6f020ac2fc3275e46c706d3406e02c3ed79b7be2",
            "dataset_revision": "f152293e235517d504809563c833d7190b8c713b",
            "files": self._source_files,
            "adapter_mode": "official-json-direct, allow-listed non-text fields",
            "official_source_modified": False,
        }
