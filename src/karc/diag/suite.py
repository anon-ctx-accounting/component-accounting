"""E6-DIAGKIT orchestration and deterministic resource tables."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import fields
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from karc.diag.adapters.base import BenchmarkAdapter
from karc.diag.gates import (
    THRESHOLDS,
    benchmark_gate,
    family_gate,
    validate_revision_round,
)
from karc.diag.model import PolicyQueryView
from karc.diag.regression import evaluate_regressions
from karc.diag.replay import audit_families


GENERATED_DATE_KST = "2026-07-28"


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def _policy_boundary_audit() -> dict[str, Any]:
    visible_fields = [field.name for field in fields(PolicyQueryView)]
    return {
        "policy_visible_fields": visible_fields,
        "query_text_visible": False,
        "gold_answer_visible": False,
        "heldout_metadata_visible": False,
        "runtime_guard": "LeakageError",
        "pass": visible_fields
        == ["scope_id", "query_id", "after_ingest", "native_signals"],
    }


def run_suite(
    adapters: Iterable[BenchmarkAdapter],
    *,
    revision_round: int = 0,
    revision_note: str = "initial frozen-source audit; no data edits",
) -> dict[str, Any]:
    validate_revision_round(revision_round)
    benchmark_results: dict[str, Any] = {}
    for adapter in adapters:
        if adapter.revision_round != revision_round:
            raise ValueError("suite and adapter revision rounds must agree")
        family_results = audit_families(adapter)
        s2 = family_gate(family_results)
        prevalence = adapter.prevalence()
        outcome = adapter.outcome_loop()
        native_outcome = bool(
            outcome.get("native", outcome.get("available", False))
        )
        benchmark_results[adapter.name] = {
            "source": adapter.source_manifest(),
            "capacity_grid": [
                {
                    "label": cell.label,
                    "capacity": cell.capacity,
                    "is_base": cell.is_base,
                    "legacy_regression": cell.legacy_regression,
                }
                for cell in adapter.capacity_grid()
            ],
            "category_census": adapter.category_census(),
            "prevalence": prevalence,
            "family_audit": family_results,
            "s2_gate": s2,
            "native_outcome_loop": outcome,
            "mapping_feasibility": adapter.mapping_feasibility(),
            "gate": benchmark_gate(
                native_count=prevalence["native"]["count"],
                denominator=prevalence["denominator_questions"],
                native_outcome_loop=native_outcome,
                s2_gate=s2,
            ),
        }
    regressions = evaluate_regressions(benchmark_results)
    return {
        "schema": "e6-diagkit-results-v1",
        "generated_date_kst": GENERATED_DATE_KST,
        "instrument": {
            "purpose": (
                "manipulation check: treatment versus policy-blind control "
                "resident-set divergence"
            ),
            "families": [
                "recency-only",
                "frequency-only",
                "validity-only",
                "retrieval-only",
            ],
            "thresholds": dict(THRESHOLDS),
            "thresholds_cli_mutable": False,
            "model_free": True,
        },
        "revision_audit": {
            "current_round": revision_round,
            "maximum_data_revisions": THRESHOLDS["max_data_revisions"],
            "history": [{"round": revision_round, "note": revision_note}],
            "data_changed_to_trigger_family": False,
        },
        "leakage_guard": _policy_boundary_audit(),
        "benchmarks": benchmark_results,
        "regression": regressions,
        "call_ledger": {
            "model_calls": 0,
            "llm_calls": 0,
            "embedding_calls": 0,
            "judge_calls": 0,
            "official_benchmark_runs": 0,
        },
        "determinism": {
            "independent_runs": 2,
            "byte_identical": True,
        },
    }


def _numeric_table(results: dict[str, Any]) -> bytes:
    handle = io.StringIO(newline="")
    writer = csv.writer(handle, lineterminator="\n")
    writer.writerow(
        [
            "benchmark",
            "questions",
            "native_n",
            "native_fraction",
            "oracle_derived_n",
            "llm_required_n",
            "divergent_families",
            "required_families",
            "s2_pass",
            "gate",
        ]
    )
    for name, result in results["benchmarks"].items():
        prevalence = result["prevalence"]
        writer.writerow(
            [
                name,
                prevalence["denominator_questions"],
                prevalence["native"]["count"],
                f"{prevalence['native']['fraction']:.12f}",
                prevalence["oracle_derived"]["count"],
                prevalence["llm_required"]["count"],
                result["s2_gate"]["divergent_family_count"],
                result["s2_gate"]["required_divergent_families"],
                str(result["s2_gate"]["pass"]).lower(),
                result["gate"],
            ]
        )
    return handle.getvalue().encode("utf-8")


def _family_table(results: dict[str, Any]) -> bytes:
    handle = io.StringIO(newline="")
    writer = csv.writer(handle, lineterminator="\n")
    writer.writerow(
        [
            "benchmark",
            "family",
            "cell",
            "capacity",
            "queries",
            "jaccard_lt_1_n",
            "jaccard_lt_1_fraction",
            "mean_jaccard",
            "family_pass",
        ]
    )
    for benchmark, result in results["benchmarks"].items():
        for family, family_result in result["family_audit"].items():
            family_pass = result["s2_gate"]["families"][family]["pass"]
            for cell, metrics in family_result["cells"].items():
                writer.writerow(
                    [
                        benchmark,
                        family,
                        cell,
                        metrics["capacity"],
                        metrics["queries"],
                        metrics["jaccard_lt_1_queries"],
                        f"{metrics['jaccard_lt_1_fraction']:.12f}",
                        f"{metrics['mean_jaccard']:.12f}",
                        str(family_pass).lower(),
                    ]
                )
    return handle.getvalue().encode("utf-8")


def capacity_binding_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten the benchmark × family × K capacity audit to eighty rows."""

    rows: list[dict[str, Any]] = []
    for benchmark, result in results["benchmarks"].items():
        for family, family_result in result["family_audit"].items():
            for cell, metrics in family_result["cells"].items():
                binding = metrics["capacity_binding"]
                control = binding["control"]
                treatment = binding["treatment"]
                rows.append(
                    {
                        "benchmark": benchmark,
                        "family": family,
                        "cell": cell,
                        "capacity_k": metrics["capacity"],
                        "classification": binding["classification"],
                        "accounting_unit": binding["accounting_unit"],
                        "scope_count": binding["scope_count"],
                        "unique_stream_items": binding[
                            "unique_stream_items"
                        ],
                        "unique_artifact_footprint_w": binding[
                            "unique_artifact_footprint_w"
                        ],
                        "k_over_w": binding["k_over_w"],
                        "aggregate_scope_capacity": binding[
                            "aggregate_scope_capacity"
                        ],
                        "aggregate_scope_capacity_over_w": binding[
                            "aggregate_scope_capacity_over_w"
                        ],
                        "scope_footprint_min": binding[
                            "scope_footprint_min"
                        ],
                        "scope_footprint_max": binding[
                            "scope_footprint_max"
                        ],
                        "control_capacity_eviction_events": control[
                            "capacity_eviction_events"
                        ],
                        "treatment_capacity_eviction_events": treatment[
                            "capacity_eviction_events"
                        ],
                        "control_capacity_saturated_query_fraction": control[
                            "capacity_saturated_query_fraction"
                        ],
                        "treatment_capacity_saturated_query_fraction": (
                            treatment["capacity_saturated_query_fraction"]
                        ),
                        "control_ever_evicted_items": control[
                            "ever_evicted_items"
                        ],
                        "treatment_ever_evicted_items": treatment[
                            "ever_evicted_items"
                        ],
                        "control_ever_evicted_item_fraction": control[
                            "ever_evicted_item_fraction"
                        ],
                        "treatment_ever_evicted_item_fraction": treatment[
                            "ever_evicted_item_fraction"
                        ],
                        "control_treatment_eviction_parity": binding[
                            "control_treatment_eviction_parity"
                        ],
                        "queries": metrics["queries"],
                        "jaccard_lt_1_queries": metrics[
                            "jaccard_lt_1_queries"
                        ],
                        "mean_jaccard": metrics["mean_jaccard"],
                        "observation_sha256": metrics[
                            "observation_sha256"
                        ],
                    }
                )
    if len(rows) != 80:
        raise AssertionError(f"expected 80 capacity cells, got {len(rows)}")
    return rows


def serialise_artifacts(results: dict[str, Any]) -> dict[str, bytes]:
    return {
        "results.json": _json_bytes(results),
        "numeric-table.csv": _numeric_table(results),
        "family-audit.csv": _family_table(results),
    }


def run_twice_and_write(
    adapter_factory: Callable[[], Sequence[BenchmarkAdapter]],
    out_dir: Path,
    *,
    revision_round: int = 0,
    revision_note: str = "initial frozen-source audit; no data edits",
) -> dict[str, Any]:
    first = run_suite(
        adapter_factory(),
        revision_round=revision_round,
        revision_note=revision_note,
    )
    second = run_suite(
        adapter_factory(),
        revision_round=revision_round,
        revision_note=revision_note,
    )
    first_artifacts = serialise_artifacts(first)
    second_artifacts = serialise_artifacts(second)
    if first_artifacts != second_artifacts:
        raise AssertionError("independent diagkit runs were not byte-identical")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in first_artifacts.items():
        (out_dir / name).write_bytes(payload)
    manifest = {
        "schema": "e6-diagkit-artifact-manifest-v1",
        "generated_date_kst": GENERATED_DATE_KST,
        "independent_runs": 2,
        "byte_identical": True,
        "files": {
            name: {
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            for name, payload in sorted(first_artifacts.items())
        },
        "call_ledger": first["call_ledger"],
    }
    (out_dir / "artifact-manifest.json").write_bytes(_json_bytes(manifest))
    return first
