"""Mechanical G-E4-ACT evaluation over a frozen replay result."""

from __future__ import annotations

import hashlib
import json

from karc.bench.e4_fixture import PRIMARY_CELL


ARC_ARMS = ("classic", "karc", "karc-no-outcome")
ARC_TRANSITIONS = (
    "t1_to_t2", "resident_to_b", "b1_revival", "b2_revival",
    "corrected_t2_to_t1",
)


def primary_cell(cells: list[dict]) -> dict:
    rho, sigma, budget = PRIMARY_CELL
    matches = [row for row in cells if row["cell"]["rho"] == rho
               and row["cell"]["sigma"] == sigma
               and row["cell"]["budget_pct"] == budget]
    if len(matches) != 1:
        raise ValueError(f"expected one primary cell, found {len(matches)}")
    return matches[0]


def evaluate_activation_gate(cells: list[dict]) -> dict:
    cell = primary_cell(cells)
    structure = cell["fixture"]["structure"]
    events = cell["events"]
    transitions = cell["arms"]["karc"]["transition_counts"]
    # A-C2: state-machine exercise is a sweep-level ARC-family requirement;
    # curation activation (G-ACT-3) remains primary-cell-only.
    sweep_transitions: dict[str, dict] = {}
    for transition_name in ARC_TRANSITIONS:
        positive_cells = []
        candidates = []
        for candidate_cell in cells:
            arm_counts = {
                arm: candidate_cell["arms"][arm]["transition_counts"][transition_name]
                for arm in ARC_ARMS
            }
            maximum = max(arm_counts.values())
            if maximum <= 0:
                continue
            positive_cells.append(candidate_cell["cell"])
            for arm, count in arm_counts.items():
                if count > 0:
                    candidates.append({
                        "arm": arm, "cell": candidate_cell["cell"], "count": count,
                    })
        candidates.sort(key=lambda row: (
            -row["count"], row["cell"]["rho"], row["cell"]["sigma"],
            row["cell"]["budget_pct"], row["arm"],
        ))
        sweep_transitions[transition_name] = {
            "positive_cells": len(positive_cells),
            "witness": candidates[0] if candidates else None,
        }
    criteria = {
        "G-ACT-1": {
            "status": "PASS" if structure["reused_occurrences"] > 0
            and transitions["t1_to_t2"] > 0 else "FAIL",
            "measured": {"distinct_task_reuse": structure["reused_occurrences"],
                         "rho": structure["rho_measured"],
                         "t1_to_t2": transitions["t1_to_t2"]},
        },
        "G-ACT-2": {
            "status": "PASS" if events["corrected"] > 0
            and events["supersession"] > 0
            and events["corrected_t2_resident"] > 0
            and events["supersession_t2_resident"] > 0 else "FAIL",
            "measured": dict(events),
        },
        "G-ACT-3": {
            "status": "PASS" if cell["jaccard"]["divergent_tasks"] >= 1 else "FAIL",
            "measured": dict(cell["jaccard"]),
            "note": "10% divergence is reference-only, not a hard threshold",
        },
        "G-ACT-4": {
            "status": "PASS" if all(
                sweep_transitions[key]["positive_cells"] > 0
                for key in ARC_TRANSITIONS
            ) else "FAIL",
            "measured": {
                "scope": "60-cell ARC-family sweep",
                "arc_arms": list(ARC_ARMS),
                "by_transition": sweep_transitions,
                "primary_karc_reference": dict(transitions),
            },
        },
        "G-ACT-5": {
            "status": "PASS-MODEL-SUBGATE-NOT-EVALUATED"
            if cell["arms"]["classic"]["stale_exposure_tasks"] > 0
            and cell["classic_stale_karc_clean_tasks"] > 0 else "FAIL",
            "measured": {
                "any_arm_stale_exposure_tasks": max(
                    row["stale_exposure_tasks"] for row in cell["arms"].values()
                ),
                "classic_stale_exposure_tasks": cell["arms"]["classic"]["stale_exposure_tasks"],
                "classic_exposed_karc_not_tasks": cell["classic_stale_karc_clean_tasks"],
                "trap_potency_G_ACT_5ii": "NOT EVALUATED (model required; prohibited milestone)",
            },
        },
        "G-ACT-6": {
            "status": "N/A",
            "measured": {"fixture": "E4 controlled", "external_adapter": False},
        },
    }
    hard = ["G-ACT-1", "G-ACT-2", "G-ACT-3", "G-ACT-4", "G-ACT-5"]
    hard_pass = all(criteria[key]["status"].startswith("PASS") for key in hard)
    verdict = {
        "schema_version": 2,
        "gate": "G-E4-ACT",
        "milestone_scope": "model-free; G-ACT-5(ii) excluded",
        "primary_cell": cell["cell"],
        "criterion_scopes": {
            "G-ACT-1": "primary", "G-ACT-2": "primary",
            "G-ACT-3": "primary", "G-ACT-4": "60-cell sweep",
            "G-ACT-5": "primary", "G-ACT-6": "controlled N/A",
        },
        "criteria": criteria,
        "overall": "PASS" if hard_pass else "FAIL",
        "failed_hard_criteria": [key for key in hard
                                 if not criteria[key]["status"].startswith("PASS")],
        "model_stage_authorized": False,
    }
    if not hard_pass:
        transition_audit = cell.get("karc_transition_audit", [])
        b1_entries = [row for row in transition_audit if row.get("to") == "B1"]
        b1_paths = []
        for entry in b1_entries:
            later = [row for row in transition_audit
                     if row["seq"] > entry["seq"]
                     and row["version_id"] == entry["version_id"]]
            b1_paths.append({
                "entry_seq": entry["seq"], "version_id": entry["version_id"],
                "entry_rule": entry["rule_id"],
                "next_transition": later[0] if later else None,
            })
        verdict["mechanical_diagnosis"] = {
            "b1_entry_count": len(b1_entries), "b1_paths": b1_paths,
        }
    verdict["sha256"] = hashlib.sha256(
        json.dumps(verdict, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return verdict
