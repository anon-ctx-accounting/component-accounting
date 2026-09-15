"""Adapter protocol and source-freeze helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Protocol, Sequence

from karc.diag.model import CapacityCell, IngestionRecord, QueryPoint


class BenchmarkAdapter(Protocol):
    """The preregistered read-only benchmark boundary."""

    name: str
    revision_round: int

    def ingestion_stream(self) -> Sequence[IngestionRecord]: ...

    def query_points(self) -> Sequence[QueryPoint]: ...

    def capacity_grid(self) -> Sequence[CapacityCell]: ...

    def category_census(self) -> dict[str, Any]: ...

    def prevalence(self) -> dict[str, Any]: ...

    def outcome_loop(self) -> dict[str, Any]: ...

    def mapping_feasibility(self) -> dict[str, Any]: ...

    def source_manifest(self) -> dict[str, Any]: ...


def standard_capacity_grid() -> tuple[CapacityCell, ...]:
    """Base plus four fixed budget cells; legacy K cells stay identifiable."""

    return (
        CapacityCell("budget-k10", 10, legacy_regression=True),
        CapacityCell("budget-k20", 20, legacy_regression=True),
        CapacityCell("base-k50", 50, is_base=True, legacy_regression=True),
        CapacityCell("budget-k100", 100),
        CapacityCell("budget-k200", 200),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_file(path: Path, expected_sha256: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise ValueError(
            f"source freeze mismatch for {path.name}: "
            f"{observed} != {expected_sha256}"
        )
    return {
        "name": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": observed,
        "read_only": True,
    }


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
