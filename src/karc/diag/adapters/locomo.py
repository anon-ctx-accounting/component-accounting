"""LoCoMo adapter over the pinned official ``locomo10.json``."""

from __future__ import annotations

import json
import re
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


LOCOMO_SHA256 = "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"


def _numbered_session_items(conversation: dict[str, Any]):
    pattern = re.compile(r"^session_(\d+)$")
    found = []
    for key, value in conversation.items():
        match = pattern.match(key)
        if match and isinstance(value, list):
            found.append((int(match.group(1)), value))
    return sorted(found)


def _evidence_ids(value: Any) -> tuple[str, ...]:
    entries = value if isinstance(value, list) else [value]
    flattened: list[str] = []
    for entry in entries:
        if isinstance(entry, str):
            flattened.extend(part for part in entry.split() if part)
    return tuple(dict.fromkeys(flattened))


def _turn_order(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"D(\d+):(\d+)", value)
    if not match:
        raise ValueError(f"non-canonical LoCoMo evidence id: {value}")
    return int(match.group(1)), int(match.group(2))


class LoCoMoAdapter:
    name = "LoCoMo"

    def __init__(self, source: Path, *, revision_round: int = 0):
        validate_revision_round(revision_round)
        self.source = source
        self.revision_round = revision_round
        self._source_file = checked_file(source, LOCOMO_SHA256)

    @cached_property
    def _rows(self) -> list[dict[str, Any]]:
        return json.loads(self.source.read_text(encoding="utf-8"))

    @cached_property
    def _stream(self) -> tuple[IngestionRecord, ...]:
        records: list[IngestionRecord] = []
        for index, row in enumerate(self._rows):
            scope = f"locomo:{index}"
            for _, turns in _numbered_session_items(row["conversation"]):
                for turn in turns:
                    artifact_id = f"{scope}:{turn['dia_id']}"
                    records.append(
                        IngestionRecord(
                            scope_id=scope,
                            artifact_id=artifact_id,
                            version=turn["dia_id"],
                            size_tok=1,
                        )
                    )
        return tuple(records)

    @cached_property
    def _queries(self) -> tuple[QueryPoint, ...]:
        points: list[QueryPoint] = []
        for index, row in enumerate(self._rows):
            scope = f"locomo:{index}"
            ingest_count = sum(
                len(turns)
                for _, turns in _numbered_session_items(row["conversation"])
            )
            for query_index, query in enumerate(row["qa"], start=1):
                points.append(
                    QueryPoint(
                        policy=PolicyQueryView(
                            scope_id=scope,
                            query_id=f"{scope}:q{query_index}",
                            after_ingest=ingest_count,
                        ),
                        measurement=MeasurementOnly(
                            category=str(query.get("category"))
                        ),
                    )
                )
        return tuple(points)

    @cached_property
    def _validity(self) -> dict[str, Any]:
        denominator = 0
        oracle = 0
        actionable = 0
        stale: set[tuple[int, str]] = set()
        for index, row in enumerate(self._rows):
            denominator += len(row["qa"])
            for query in row["qa"]:
                if query.get("category") != 3:
                    continue
                oracle += 1
                evidence = _evidence_ids(query.get("evidence") or [])
                if len(evidence) < 2:
                    continue
                actionable += 1
                ordered = sorted(evidence, key=_turn_order)
                stale.update((index, artifact_id) for artifact_id in ordered[:-1])
        return {
            "denominator": denominator,
            "oracle": oracle,
            "actionable": actionable,
            "stale": len(stale),
        }

    def ingestion_stream(self) -> Sequence[IngestionRecord]:
        return self._stream

    def query_points(self) -> Sequence[QueryPoint]:
        return self._queries

    def capacity_grid(self) -> Sequence[CapacityCell]:
        return standard_capacity_grid()

    def category_census(self) -> dict[str, Any]:
        counts = Counter(
            str(query.get("category"))
            for row in self._rows
            for query in row["qa"]
        )
        return {
            "denominator_questions": sum(counts.values()),
            "categories": dict(sorted(counts.items())),
            "sessions": len(self._rows),
            "ingestion_units": len(self._stream),
        }

    def prevalence(self) -> dict[str, Any]:
        denominator = self._validity["denominator"]
        oracle = self._validity["oracle"]
        return {
            "denominator_questions": denominator,
            "native": {"count": 0, "fraction": 0.0},
            "oracle_derived": {
                "count": oracle,
                "fraction": oracle / denominator,
                "label": "official category==3 upper bound; held out from policy",
            },
            "llm_required": {"count": 0, "fraction": 0.0},
            "unclassified": denominator - oracle,
            "actionable_chain": {
                "count": self._validity["actionable"],
                "fraction": self._validity["actionable"] / denominator,
            },
            "stale_candidate_artifacts": {
                "count": self._validity["stale"],
                "denominator_conversation_turns": len(self._stream),
                "fraction": self._validity["stale"] / len(self._stream),
            },
        }

    def outcome_loop(self) -> dict[str, Any]:
        return {
            "native": False,
            "status": "N/A",
            "reason": "QA correctness is not returned to the official memory state.",
        }

    def mapping_feasibility(self) -> dict[str, Any]:
        return {
            "llm_free": "partial",
            "identity_and_order": True,
            "native_same_slot_target": False,
            "reason": "Temporal QA evidence is held-out oracle annotation.",
        }

    def source_manifest(self) -> dict[str, Any]:
        return {
            "source_revision": "locomo10.json@79fa87e9",
            "files": [self._source_file],
            "adapter_mode": "official-json-direct",
        }
