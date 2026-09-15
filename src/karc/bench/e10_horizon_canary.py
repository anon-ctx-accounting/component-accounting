"""Model-free pre-execution contract for the Paper 2 16-turn sensitivity.

This module extends the frozen E5-G1 whole-document design only along the
session horizon.  It builds and audits the schedule and approval contract; it
does not contain a provider call path.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from karc.bench import e5_cache_canary as g1
from karc.bench.e5_runtime_accounting import ledger_from_rows


EXPERIMENT = "E10-P2-H16"
SCHEMA = "e10-p2-h16-pre-execution-v1"
TURN_SCHEMA = "e10-p2-h16-turn-v1"
ARMS = g1.ARMS
SESSION_LENGTH = 16
SESSION_COUNT = 12
REUSE_FACTOR = 0.5
BUDGET_PCT = 5
SCHEDULE_SEED = 10160
PLANNED_VALID_TURNS = SESSION_COUNT * len(ARMS) * SESSION_LENGTH
MODEL_SPEND_ATTEMPT_CAP = 480
INFRA_FAILURE_ATTEMPT_CAP = 96
TOTAL_API_CALL_CAP = MODEL_SPEND_ATTEMPT_CAP + INFRA_FAILURE_ATTEMPT_CAP
MODEL = "gpt-5.6-luna"
REASONING_EFFORT = "high"
CODEX_CLI_VERSION = "codex-cli 0.144.5"
HORIZONS = (4, 8, 12, 16)
CACHE_READ_WEIGHTS = (0, 0.1, 1)
FORBIDDEN_RAW_KEYS = frozenset({
    "prompt",
    "output",
    "output_text",
    "transcript",
    "command",
    "argv",
    "tool_response",
    "credential",
    "environment",
    "cwd",
})


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode()


def hash_json(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_bundle(repo_root: str | Path, fixture_root: str | Path) -> dict:
    """Build the exact model-free 12-session, 16-turn E5-G1 extension."""
    return g1.prepare_bundle(
        repo_root,
        fixture_root,
        schedule_seed=SCHEDULE_SEED,
        session_count=SESSION_COUNT,
        session_length=SESSION_LENGTH,
        allow_unregistered_session_length=True,
    )


def validate_bundle(bundle: dict) -> dict:
    """Fail closed if the horizon schedule changes its registered meaning."""
    schedule = bundle["schedule"]
    assignments = schedule["assignments"]
    expected_opportunities = SESSION_COUNT * (SESSION_LENGTH - 1)
    expected_reuse = int(REUSE_FACTOR * expected_opportunities)
    failures: list[str] = []

    checks = {
        "session_count": schedule["sessions"] == SESSION_COUNT,
        "session_length": schedule["session_length"] == SESSION_LENGTH,
        "reuse_requested": schedule["reuse_factor_requested"] == REUSE_FACTOR,
        "reuse_measured": schedule["reuse_factor_measured"] == REUSE_FACTOR,
        "reuse_denominator": (
            schedule["reuse_denominator_eligible_followups"]
            == expected_opportunities
        ),
        "reuse_count": schedule["reuse_count"] == expected_reuse,
        "task_count": len(assignments) == SESSION_COUNT * SESSION_LENGTH,
        "bundle_sessions": len(bundle["sessions"]) == SESSION_COUNT,
        "retrieval_gold_containment": (
            bundle["retrieval_audit"]["gold_containment_rate"] == 1
        ),
        "retrieval_budget_parity": (
            bundle["retrieval_audit"]["mean_budget_utilization"] >= 0.8
        ),
    }

    by_session: dict[str, list[dict]] = defaultdict(list)
    for row in assignments:
        by_session[row["session_id"]].append(row)
    reuse_counts = []
    lineage_ok = True
    position_ok = True
    for rows in by_session.values():
        rows.sort(key=lambda row: int(row["session_task"]))
        position_ok &= [int(row["session_task"]) for row in rows] == list(
            range(1, SESSION_LENGTH + 1)
        )
        seen: set[str] = set()
        for row in rows:
            is_prior_gold = row["required_artifact"] in seen
            lineage_ok &= is_prior_gold is bool(row["resident_reuse"])
            seen.add(row["required_artifact"])
        reuse_counts.append(sum(bool(row["resident_reuse"]) for row in rows))
    checks["position_coverage"] = position_ok
    checks["reuse_means_prior_session_gold"] = lineage_ok
    checks["balanced_reuse_allocation"] = sorted(reuse_counts) == [7] * 6 + [8] * 6
    checks["planned_valid_turns"] = PLANNED_VALID_TURNS == 384
    checks["model_spend_attempt_cap"] = MODEL_SPEND_ATTEMPT_CAP == 480

    failures.extend(name for name, passed in checks.items() if not passed)
    return {
        "pass": not failures,
        "checks": checks,
        "failures": failures,
        "reuse_denominator": expected_opportunities,
        "reuse_count": expected_reuse,
        "reuse_per_session": reuse_counts,
        "planned_valid_turns": PLANNED_VALID_TURNS,
        "model_spend_attempt_cap": MODEL_SPEND_ATTEMPT_CAP,
    }


def config_json_schema() -> dict:
    """Return the immutable scientific-config JSON Schema."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "urn:k-arc:e10-p2-h16:pre-execution:v1",
        "title": "E10-P2-H16 pre-execution scientific contract",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema", "experiment", "status", "authority", "design",
            "runtime_pin", "attempt_contract", "analysis_contract",
            "stopping_contract", "privacy_contract", "fixture",
            "schedule_sha256", "preparation_source_sha256", "cost_plan",
            "approval_gate", "model_calls_at_freeze", "embedding_calls_at_freeze",
            "sha256",
        ],
        "properties": {
            "schema": {"const": SCHEMA},
            "experiment": {"const": EXPERIMENT},
            "status": {"const": "AWAITING_USER_APPROVAL"},
            "authority": {"type": "array", "minItems": 1, "items": {"type": "string"}},
            "design": {
                "type": "object",
                "additionalProperties": True,
                "required": [
                    "arms", "session_length", "paired_sessions", "reuse_factor",
                    "budget_pct", "granularity", "planned_valid_turns",
                ],
                "properties": {
                    "arms": {"const": list(ARMS)},
                    "session_length": {"const": SESSION_LENGTH},
                    "paired_sessions": {"const": SESSION_COUNT},
                    "reuse_factor": {"const": REUSE_FACTOR},
                    "budget_pct": {"const": BUDGET_PCT},
                    "granularity": {"const": "whole-document E5-G1"},
                    "planned_valid_turns": {"const": PLANNED_VALID_TURNS},
                },
            },
            "runtime_pin": {"type": "object"},
            "attempt_contract": {"type": "object"},
            "analysis_contract": {"type": "object"},
            "stopping_contract": {"type": "object"},
            "privacy_contract": {"type": "object"},
            "fixture": {"type": "object"},
            "schedule_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "preparation_source_sha256": {"type": "object"},
            "preparation_base_git_hash": {"type": "string", "pattern": "^[0-9a-f]{40}$"},
            "cost_plan": {"type": "object"},
            "approval_gate": {"type": "object"},
            "model_calls_at_freeze": {"const": 0},
            "embedding_calls_at_freeze": {"const": 0},
            "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        },
    }


def turn_json_schema() -> dict:
    """Return the content-free normalized turn-row contract."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "urn:k-arc:e10-p2-h16:turn:v1",
        "title": "E10-P2-H16 normalized turn row",
        "type": "object",
        "additionalProperties": True,
        "required": [
            "schema", "arm", "session_id", "position", "task_id",
            "failure_class", "model_spend_attempts", "infra_failure_attempts",
            "api_input_tokens_no_cache", "api_input_tokens_with_cache",
            "api_cache_read_tokens", "api_output_tokens", "api_reasoning_tokens",
            "mcp_search_calls", "mcp_get_calls", "prompt_sha256", "privacy",
        ],
        "properties": {
            "schema": {"const": TURN_SCHEMA},
            "arm": {"enum": list(ARMS)},
            "session_id": {"type": "string", "pattern": "^S[0-9]{2}$"},
            "position": {"type": "integer", "minimum": 1, "maximum": SESSION_LENGTH},
            "task_id": {"type": "string"},
            "failure_class": {"type": "string"},
            "model_spend_attempts": {"type": "integer", "minimum": 0},
            "infra_failure_attempts": {"type": "integer", "minimum": 0},
            "api_input_tokens_no_cache": {"type": "integer", "minimum": 0},
            "api_input_tokens_with_cache": {"type": "integer", "minimum": 0},
            "api_cache_read_tokens": {"type": "integer", "minimum": 0},
            "api_output_tokens": {"type": "integer", "minimum": 0},
            "api_reasoning_tokens": {"type": "integer", "minimum": 0},
            "mcp_search_calls": {"type": "integer", "minimum": 0},
            "mcp_get_calls": {"type": "integer", "minimum": 0},
            "prompt_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "privacy": {"const": "normalized-no-content"},
        },
        "not": {"anyOf": [{"required": [key]} for key in sorted(FORBIDDEN_RAW_KEYS)]},
    }


def build_config(
    bundle: dict,
    *,
    source_sha256: dict[str, str],
    base_git_hash: str,
) -> dict:
    """Build the self-hashed pre-execution scientific contract."""
    audit = validate_bundle(bundle)
    if not audit["pass"]:
        raise ValueError(f"invalid H16 bundle: {audit['failures']}")
    snapshot = g1.bundle_snapshot(bundle)
    value = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "status": "AWAITING_USER_APPROVAL",
        "authority": [
            "user-directed E10 planning discussion (approval still pending for execution)",
            "docs/analysis/e5-g1-cache-asymmetry-canary--codex-workorder.md",
            "docs/analysis/e9-writeup-and-release--codex-workorder.md §0/§1/Lane P2",
        ],
        "design": {
            "objective": (
                "measure whether the E5-G1 turn-position level-versus-growth "
                "pattern persists through position 16"
            ),
            "arms": list(ARMS),
            "session_length": SESSION_LENGTH,
            "paired_sessions": SESSION_COUNT,
            "reuse_factor": REUSE_FACTOR,
            "reuse_definition": bundle["schedule"]["definition"],
            "reuse_denominator": SESSION_COUNT * (SESSION_LENGTH - 1),
            "reuse_count": int(REUSE_FACTOR * SESSION_COUNT * (SESSION_LENGTH - 1)),
            "budget_pct": BUDGET_PCT,
            "granularity": "whole-document E5-G1",
            "planned_valid_turns": PLANNED_VALID_TURNS,
            "schedule_seed": SCHEDULE_SEED,
            "same_task_schedule_for_both_arms": True,
            "new_experiment_not_a_reinterpretation_of_E5_G1": True,
        },
        "runtime_pin": {
            "codex_cli": CODEX_CLI_VERSION,
            "model": MODEL,
            "reasoning_effort": REASONING_EFFORT,
            "auth_mode": "ChatGPT subscription",
            "provider_model_availability": "UNKNOWN_WITHOUT_APPROVED_PROBE",
            "substitution": "FORBIDDEN",
            "mismatch_disposition": "STOP_WITHOUT_SUBSTITUTION",
        },
        "attempt_contract": {
            "planned_valid_turns": PLANNED_VALID_TURNS,
            "model_spend_attempt_cap": MODEL_SPEND_ATTEMPT_CAP,
            "model_spend_definition": (
                "successful turn or any failed call with non-zero normalized model usage"
            ),
            "max_model_spend_attempts_per_turn": 3,
            "infra_failure_attempt_cap": INFRA_FAILURE_ATTEMPT_CAP,
            "infra_failure_definition": "failed call with zero normalized model usage",
            "total_api_call_cap": TOTAL_API_CALL_CAP,
            "infra_failures_are_append_only_and_do_not_refund_total_api_call_cap": True,
            "provider_probe": (
                "not part of decision-bearing raw; requires separate explicit approval and cap"
            ),
        },
        "analysis_contract": {
            "accounting": "F=fresh input; R=gross-fresh cache-read; C_w=F+wR",
            "cache_read_weights": list(CACHE_READ_WEIGHTS),
            "horizons": list(HORIZONS),
            "primary_outputs": [
                "arm-by-position mean F, R, and gross for positions 1..16",
                "paired-session cumulative C_w difference and ratio at H=4,8,12,16",
                "first observed per-turn and cumulative gross crossover, if any",
            ],
            "uncertainty": (
                "paired-session percentile bootstrap, 10000 resamples, two-sided 95% CI"
            ),
            "curve_fits": (
                "linear and quadratic descriptive fits over observed positions 1..16 only"
            ),
            "quality": (
                "report exact-answer correctness by arm and horizon; withhold a "
                "cost-optimal recommendation if quality differs materially"
            ),
            "claim_scope": (
                "this 16-turn horizon, frozen schedule, accounting, model, and runtime only"
            ),
            "forbidden_interpretations": [
                "no structural or asymptotic complexity claim",
                "no extrapolation beyond position 16",
                "no K-ARC performance-superiority claim",
            ],
        },
        "stopping_contract": {
            "before_first_turn": [
                "stop if Codex CLI is not exactly 0.144.5",
                "stop if gpt-5.6-luna/high cannot be selected; do not substitute",
                "stop if fixture, schedule, schema, or execution-source hash mismatches",
                "stop until the user explicitly approves model calls and experiment-only K-ARC MCP",
            ],
            "during_run": [
                "stop globally on reported-model mismatch",
                "stop globally on missing or internally inconsistent token telemetry",
                "stop globally on context overflow or runtime truncation; do not shorten prompts",
                "stop globally if rag-bm25 records any K-ARC MCP search/get call",
                "stop when any registered attempt cap is reached",
            ],
            "task_errors": (
                "grade and retain normalized metadata; task-answer error alone is not a stop"
            ),
        },
        "privacy_contract": {
            "persist": [
                "normalized token usage", "grade", "IDs", "counts", "hashes",
                "failure category and hashed error digest",
            ],
            "forbid": sorted(FORBIDDEN_RAW_KEYS),
            "raw_is_append_only": True,
            "original_fixture_content_is_not_copied_to_experiment_raw": True,
        },
        "fixture": {
            "root": "fixture/e4-v2",
            "corpus_seed": g1.CORPUS_SEED,
            "corpus_content_hash": bundle["manifest"]["corpus_content_hash"],
            "source_manifest_sha256": snapshot["schedule"]["source_e4_manifest_sha256"],
            "budget_tokens": bundle["manifest"]["cell"]["budget_tokens"],
        },
        "schedule_sha256": snapshot["sha256"],
        "preparation_source_sha256": source_sha256,
        "preparation_base_git_hash": base_git_hash,
        "cost_plan": {
            "source": "E5-G1 observed positions 1..8; planning scenarios only",
            "observed_E5_G1": {
                "valid_turns": 192,
                "fresh_input_tokens": 4_047_596,
                "cache_read_tokens": 18_917_632,
                "gross_input_tokens": 22_965_228,
                "output_tokens": 58_828,
                "dollar_cost": None,
            },
            "H16_planning_gross_tokens": {
                "flat_at_position_8_scenario": 64_598_324,
                "descriptive_fit_scenario": 94_663_136,
                "not_a_confidence_interval": True,
            },
            "H16_descriptive_fit_fresh_tokens": 13_302_163,
            "H16_output_tokens_linear_turn_count_scenario": 117_656,
            "dollar_formula_per_million_tokens": (
                "13.302163*p_fresh + 81.360973*p_cache_read + "
                "0.117656*p_output; subscription/runtime may not expose dollar cost"
            ),
        },
        "approval_gate": {
            "scientific_contract_edits_after_approval": "FORBIDDEN; amendment required",
            "execution_source_freeze": (
                "must be created after approval and before the first probe/model turn; "
                "it may implement but may not alter this scientific contract"
            ),
            "required_user_approvals": [
                "384 planned valid model turns and 480 model-spend attempt cap",
                "experiment-only K-ARC MCP search/get in karc-full; zero in rag-bm25",
                "separately capped runtime/model availability probe",
                "the planning token range and unavailable exact dollar amount",
            ],
        },
        "model_calls_at_freeze": 0,
        "embedding_calls_at_freeze": 0,
    }
    value["sha256"] = hash_json(value)
    return value


def validate_config(config: dict) -> dict:
    unsigned = dict(config)
    recorded = unsigned.pop("sha256", None)
    checks = {
        "self_hash": recorded == hash_json(unsigned),
        "schema": config.get("schema") == SCHEMA,
        "experiment": config.get("experiment") == EXPERIMENT,
        "status": config.get("status") == "AWAITING_USER_APPROVAL",
        "planned_valid_turns": (
            config.get("design", {}).get("planned_valid_turns")
            == PLANNED_VALID_TURNS
        ),
        "model_spend_attempt_cap": (
            config.get("attempt_contract", {}).get("model_spend_attempt_cap")
            == MODEL_SPEND_ATTEMPT_CAP
        ),
        "no_substitution": (
            config.get("runtime_pin", {}).get("substitution") == "FORBIDDEN"
        ),
        "zero_model_calls": config.get("model_calls_at_freeze") == 0,
        "zero_embedding_calls": config.get("embedding_calls_at_freeze") == 0,
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "failures": [name for name, passed in checks.items() if not passed],
    }


def audit_turn_rows(rows: Iterable[dict], *, require_complete: bool) -> dict:
    """Audit normalized/mock rows without retaining provider content."""
    rows = list(rows)
    failures: list[str] = []
    forbidden = sorted({key for row in rows for key in row if key in FORBIDDEN_RAW_KEYS})
    if forbidden:
        failures.append("privacy_forbidden_keys")
    rag_mcp = sum(
        int(row.get("mcp_search_calls", 0)) + int(row.get("mcp_get_calls", 0))
        for row in rows if row.get("arm") == "rag-bm25"
    )
    if rag_mcp:
        failures.append("rag_karc_mcp_nonzero")
    keys = [
        (row.get("arm"), row.get("session_id"), row.get("position"))
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        failures.append("duplicate_turn_key")
    if any(row.get("schema") != TURN_SCHEMA for row in rows):
        failures.append("turn_schema_mismatch")
    if any(
        int(row.get("api_input_tokens_no_cache", 0))
        + int(row.get("api_cache_read_tokens", 0))
        != int(row.get("api_input_tokens_with_cache", 0))
        for row in rows
    ):
        failures.append("token_accounting_mismatch")
    ledger = ledger_from_rows(rows)
    if ledger["model_spend_attempts"] > MODEL_SPEND_ATTEMPT_CAP:
        failures.append("model_spend_cap_exceeded")
    if ledger["infra_failure_attempts"] > INFRA_FAILURE_ATTEMPT_CAP:
        failures.append("infra_failure_cap_exceeded")
    if ledger["api_calls_total"] > TOTAL_API_CALL_CAP:
        failures.append("total_api_call_cap_exceeded")
    if require_complete:
        if len(rows) != PLANNED_VALID_TURNS:
            failures.append("valid_turn_count")
        expected = {
            (arm, f"S{session:02d}", position)
            for arm in ARMS
            for session in range(1, SESSION_COUNT + 1)
            for position in range(1, SESSION_LENGTH + 1)
        }
        if set(keys) != expected:
            failures.append("turn_key_coverage")
    return {
        "pass": not failures,
        "failures": failures,
        "rows": len(rows),
        "rag_karc_mcp_calls": rag_mcp,
        **ledger,
    }
