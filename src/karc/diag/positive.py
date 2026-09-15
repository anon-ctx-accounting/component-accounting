"""Deterministic positive controls for the E6 activation diagnostic.

The two substrates are existing, frozen model-free streams:

* E4-v2's E5-G0 primary reuse cells (classic ARC versus K-ARC).
* E7-LAZY-KG's reconstructed primary-k stream (FBR versus RTF).

Both are reduced to state-set pairs and passed through
``karc.diag.replay.audit_state_pairs``.  No query text, gold answer, model,
embedder, judge, or network service enters the diagnostic detector.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

from karc.bench.e4_replay import load_confirmed_config, replay_cell
from karc.bench.e5_killgate import (
    ENGINE_RHO_LABEL,
    ENGINE_SIGMA_LABEL,
    PRIMARY_BUDGET,
    PRIMARY_SESSION_LENGTH,
    SCHEDULE_SEEDS,
    _manifest_at_budget,
    build_reuse_schedule,
)
from karc.diag.replay import audit_state_pairs


E4_V2_MANIFEST_SHA256 = (
    "d8ec0bcf8354a5dc1046c7787b9ce4550086148ecfb4e42c1791bbc4b81ccc43"
)
E4_V2_TASKS_SHA256 = (
    "d9175adf0998b4aaaf8ba1c287e6cef898d854bebe931bb05772a17399f4dfdd"
)
E5_G0_VERDICT_FILE_SHA256 = (
    "932c3f71ca79f4c61066c78d94b05a94783909abd3c951e60d42a56e9fc8d314"
)
E7_RUN_FILE_SHA256 = (
    "b24f52b46f75efda251735d2a811c7e47830d51a67b81e6bd06f791c566e8a40"
)
E7_QUERY_RESULTS_FILE_SHA256 = (
    "c01e702b5d5b71a0e447bbec8bb2a16714e664c919b3dd4eed67069d34902f17"
)
E4_REUSE_GRID = (0.25, 0.5, 0.75, 1.0)
E7_PRIMARY_K = 5


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def audit_e4_v2(repo_root: str | Path) -> dict[str, Any]:
    """Rerun the twelve frozen E5-G0 primary reuse cells on E4-v2."""

    repo = Path(repo_root)
    freeze = json.loads(
        (repo / "fixture/e4-v2/freeze.json").read_text(encoding="utf-8")
    )
    verdict_path = repo / "docs/experiments/E5-G0/raw/verdict.json"
    source_pins = {
        "freeze_pass": freeze.get("pass") is True,
        "manifest_sha256_match": (
            freeze.get("manifest_sha256") == E4_V2_MANIFEST_SHA256
        ),
        "tasks_sha256_match": (
            freeze.get("tasks_sha256") == E4_V2_TASKS_SHA256
        ),
        "e5_g0_verdict_file_sha256_match": (
            _sha256_file(verdict_path) == E5_G0_VERDICT_FILE_SHA256
        ),
    }
    if not all(source_pins.values()):
        raise ValueError(f"E4-v2 positive-control source pin failed: {source_pins}")

    confirmed = load_confirmed_config(repo)
    cells: list[dict[str, Any]] = []
    for schedule_seed in SCHEDULE_SEEDS:
        for reuse_factor in E4_REUSE_GRID:
            _, manifest, tasks = build_reuse_schedule(
                repo,
                schedule_seed=schedule_seed,
                session_length=PRIMARY_SESSION_LENGTH,
                reuse_factor=reuse_factor,
            )
            manifest = _manifest_at_budget(manifest, PRIMARY_BUDGET)
            replay_summary, replay_rows = replay_cell(
                rho=ENGINE_RHO_LABEL,
                sigma=ENGINE_SIGMA_LABEL,
                budget_pct=PRIMARY_BUDGET,
                confirmed_config=confirmed,
                fixture_manifest=manifest,
                fixture_tasks=tasks,
            )
            state_pairs = (
                {
                    "query_id": row["task_id"],
                    "control": row["arms"]["classic"][
                        "working_set_versions"
                    ],
                    "treatment": row["arms"]["karc"][
                        "working_set_versions"
                    ],
                }
                for row in replay_rows
            )
            cells.append(
                {
                    "schedule_seed": schedule_seed,
                    "reuse_factor": reuse_factor,
                    "session_length": PRIMARY_SESSION_LENGTH,
                    "budget_pct": PRIMARY_BUDGET,
                    "replay_sha256": replay_summary["sha256"],
                    **audit_state_pairs(state_pairs),
                }
            )
    return {
        "substrate": "fixture/e4-v2 via E5-G0 primary reuse cells",
        "comparison": "classic control versus K-ARC treatment",
        "source_pins": source_pins,
        "cells": cells,
        "model_calls": 0,
        "embedding_calls": 0,
        "judge_calls": 0,
        "network_calls": 0,
    }


def audit_e7_lazy_kg(repo_root: str | Path) -> dict[str, Any]:
    """Regenerate E7 and audit primary-k LAZY-FBR versus LAZY-RTF."""

    repo = Path(repo_root)
    run_path = repo / "docs/experiments/E7-LAZY-KG/raw/run.py"
    committed_query_results = (
        repo / "docs/experiments/E7-LAZY-KG/raw/query-results.jsonl"
    )
    source_pins = {
        "run_file_sha256_match": (
            _sha256_file(run_path) == E7_RUN_FILE_SHA256
        ),
        "query_results_file_sha256_match": (
            _sha256_file(committed_query_results)
            == E7_QUERY_RESULTS_FILE_SHA256
        ),
    }
    if not all(source_pins.values()):
        raise ValueError(
            f"E7 positive-control source pin failed: {source_pins}"
        )

    with tempfile.TemporaryDirectory(prefix="e8-d1-e7-") as temp_dir:
        output = Path(temp_dir)
        environment = dict(os.environ)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        subprocess.run(
            [
                sys.executable,
                str(run_path),
                "--output-dir",
                str(output),
            ],
            cwd=repo,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        regenerated = output / "query-results.jsonl"
        regenerated_sha256 = _sha256_file(regenerated)
        byte_identical = (
            regenerated.read_bytes() == committed_query_results.read_bytes()
        )
        rows = [
            row
            for row in _read_jsonl(regenerated)
            if row["k"] == E7_PRIMARY_K
        ]
        state_pairs = (
            {
                "query_id": row["query_id"],
                "control": row["contexts"]["LAZY-FBR"],
                "treatment": row["contexts"]["LAZY-RTF"],
            }
            for row in rows
        )
        measured = audit_state_pairs(state_pairs)

    return {
        "substrate": "E7-LAZY-KG reconstructed stream",
        "provenance_label": "reconstructed-not-reproduced",
        "comparison": "LAZY-FBR control versus LAZY-RTF treatment",
        "primary_k": E7_PRIMARY_K,
        "source_pins": source_pins,
        "regenerated_query_results_sha256": regenerated_sha256,
        "regenerated_byte_identical": byte_identical,
        **measured,
        "model_calls": 0,
        "embedding_calls": 0,
        "judge_calls": 0,
        "network_calls": 0,
    }


def audit_positive_controls(repo_root: str | Path) -> dict[str, Any]:
    return {
        "schema": "g-diag-1-positive-controls-v1",
        "detector": "karc.diag.replay.audit_state_pairs",
        "e4_v2": audit_e4_v2(repo_root),
        "e7_lazy_kg": audit_e7_lazy_kg(repo_root),
        "call_ledger": {
            "model_calls": 0,
            "llm_calls": 0,
            "embedding_calls": 0,
            "judge_calls": 0,
            "network_calls": 0,
        },
    }
