"""Model-free pre-execution contract for E11-BASE3.

The module reuses the frozen E10-P2-H16 fixture, task schedule, and reader pin.
It defines three additional baseline arms but deliberately contains no model
or provider invocation path.  Execution remains blocked until the user and
main approve a separately frozen runner.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Iterable

from karc.bench import e10_horizon_canary as h16
from karc.bench import e5_cache_canary as g1
from karc.bench import materialize as mzt


EXPERIMENT = "E11-BASE3"
SCHEMA = "e11-base3-pre-execution-v1"
TURN_SCHEMA = "e11-base3-turn-v1"
ARMS = ("full-history", "sliding-window-compaction", "stateless-rag")
REFERENCE_ARMS = ("karc-full", "rag-bm25")
SESSION_COUNT = h16.SESSION_COUNT
SESSION_LENGTH = h16.SESSION_LENGTH
REUSE_FACTOR = h16.REUSE_FACTOR
BUDGET_PCT = h16.BUDGET_PCT
SCHEDULE_SEED = h16.SCHEDULE_SEED
WINDOW_COMPLETED_TURNS = 4
PLANNED_VALID_TURNS = SESSION_COUNT * SESSION_LENGTH * len(ARMS)
MODEL_SPEND_ATTEMPT_CAP = 720
INFRA_FAILURE_ATTEMPT_CAP = 144
TOTAL_API_CALL_CAP = 864
PROVIDER_PROBE_ATTEMPT_CAP = 2
MAX_MODEL_SPEND_ATTEMPTS_PER_TURN = 3
MODEL = h16.MODEL
REASONING_EFFORT = h16.REASONING_EFFORT
CODEX_CLI_VERSION = h16.CODEX_CLI_VERSION
HORIZONS = h16.HORIZONS
CACHE_READ_WEIGHTS = h16.CACHE_READ_WEIGHTS
BOOTSTRAP_REPS = 10_000
QUALITY_NONINFERIORITY_MARGIN = 0.05
H16_SCHEDULE_SHA256 = "e9ee352fdaa8b8f3942cf77398b3c6b453952535a89e022d8b53aa08405f7006"
H16_CONFIG_SELF_SHA256 = "75dd169f2d3eb05f7a40717ebd1199e10c3a8b22f918aaf7d53a280f966d44a2"
FIXTURE_CONTENT_SHA256 = "4378159b89fe3128e031e4398c96495cd661ff2c91cfc6e457827b250fb9228a"
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
    "history_text",
    "compaction_text",
})

FULL_HISTORY_INSTRUCTION = (
    "Answer each benchmark task only from the complete injected fixture below. "
    "The entire fixture is retained for all sixteen turns without eviction or "
    "compaction. Do not use local tools, MCP, or external retrieval. Return only "
    "the exact assignment requested by the task."
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode()


def hash_json(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_bundle(repo_root: str | Path, fixture_root: str | Path) -> dict:
    """Regenerate the exact H16 bundle; E11 is not allowed a new schedule."""
    return h16.prepare_bundle(repo_root, fixture_root)


def validate_bundle(bundle: dict) -> dict:
    """Prove that fixture, schedule, and cell axes are identical to H16."""
    parent = h16.validate_bundle(bundle)
    snapshot = g1.bundle_snapshot(bundle)
    checks = {
        "h16_bundle_audit": parent["pass"],
        "schedule_sha256_exact": snapshot["sha256"] == H16_SCHEDULE_SHA256,
        "fixture_content_sha256_exact": (
            bundle["manifest"]["corpus_content_hash"] == FIXTURE_CONTENT_SHA256
        ),
        "session_count_exact": bundle["schedule"]["sessions"] == SESSION_COUNT,
        "session_length_exact": (
            bundle["schedule"]["session_length"] == SESSION_LENGTH
        ),
        "reuse_exact": bundle["schedule"]["reuse_factor_measured"] == REUSE_FACTOR,
        "budget_pct_exact": bundle["manifest"]["cell"]["budget_pct"] == BUDGET_PCT,
        "planned_valid_turns": PLANNED_VALID_TURNS == 576,
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "failures": [name for name, passed in checks.items() if not passed],
        "h16_audit": parent,
        "schedule_sha256": snapshot["sha256"],
        "planned_valid_turns": PLANNED_VALID_TURNS,
    }


def expected_state(arm: str, position: int) -> dict:
    """Return the frozen cross-turn state supplied at one model turn."""
    if arm not in ARMS:
        raise ValueError(f"unknown E11-BASE3 arm {arm!r}")
    if not 1 <= position <= SESSION_LENGTH:
        raise ValueError(f"position outside 1..{SESSION_LENGTH}: {position}")
    prior = position - 1
    if arm == "full-history":
        return {
            "session_mode": "persistent-unbounded",
            "history_turns_supplied": prior,
            "discarded_prior_turns": 0,
            "compaction_summary_calls": 0,
            "retrieval_mode": "none-full-corpus-injected",
            "provider_session_reused": position > 1,
        }
    if arm == "sliding-window-compaction":
        kept = min(prior, WINDOW_COMPLETED_TURNS)
        return {
            "session_mode": "rebuilt-bounded-window",
            "history_turns_supplied": kept,
            "discarded_prior_turns": prior - kept,
            "compaction_summary_calls": 0,
            "retrieval_mode": "frozen-bm25-current-turn",
            "provider_session_reused": False,
        }
    return {
        "session_mode": "isolated-per-turn",
        "history_turns_supplied": 0,
        "discarded_prior_turns": prior,
        "compaction_summary_calls": 0,
        "retrieval_mode": "frozen-bm25-current-turn",
        "provider_session_reused": False,
    }


def materialize_arm(
    arm: str,
    workdir: str | Path,
    *,
    fixture_root: str | Path,
    manifest: dict,
) -> dict:
    """Materialize an arm without invoking a model, embedding, or MCP tool."""
    workdir = Path(workdir)
    fixture_root = Path(fixture_root)
    if arm == "full-history":
        shutil.copytree(fixture_root / "repo", workdir, dirs_exist_ok=True)
        mzt._init_git(workdir)
        artifact_ids = sorted(
            manifest["artifacts"], key=lambda key: manifest["artifacts"][key]["path"],
        )
        agents_sha256, agents_bytes = mzt._codex_agents(
            workdir,
            manifest,
            artifact_ids,
            instruction=FULL_HISTORY_INSTRUCTION,
        )
        return {
            "arm": arm,
            "workdir": workdir,
            "hook_mode": "deny-all",
            "config_overrides": (f"project_doc_max_bytes={agents_bytes + 4096}",),
            "agents_sha256": agents_sha256,
            "agents_bytes": agents_bytes,
            "injected_artifact_count": len(artifact_ids),
            "injected_knowledge_tokens": sum(
                int(manifest["artifacts"][key]["size_tok"])
                for key in artifact_ids
            ),
            "mcp_configured": False,
        }
    if arm not in ("sliding-window-compaction", "stateless-rag"):
        raise ValueError(f"unknown E11-BASE3 arm {arm!r}")
    materialized = g1.materialize_session(
        "rag-bm25",
        workdir,
        fixture_root=fixture_root,
        manifest=manifest,
        initial_resident_versions=[],
    )
    has_mcp = any(
        value.startswith("mcp_servers.karc.")
        for value in materialized["config_overrides"]
    )
    return {
        "arm": arm,
        "workdir": materialized["workdir"],
        "hook_mode": materialized["hook_mode"],
        "config_overrides": materialized["config_overrides"],
        "agents_sha256": materialized["agents_sha256"],
        "agents_bytes": materialized["agents_bytes"],
        "injected_artifact_count": 0,
        "injected_knowledge_tokens": 0,
        "mcp_configured": has_mcp,
    }


def config_json_schema() -> dict:
    """Return the immutable scientific-config JSON Schema."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "urn:k-arc:e11-base3:pre-execution:v1",
        "title": "E11-BASE3 pre-execution scientific contract",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema", "experiment", "status", "authority", "design",
            "reference_cell", "runtime_pin", "attempt_contract",
            "analysis_contract", "decision_contract", "stopping_contract",
            "privacy_contract", "fixture", "schedule_sha256",
            "preparation_source_sha256", "preparation_base_git_hash",
            "cost_plan", "approval_gate", "model_calls_at_freeze",
            "embedding_calls_at_freeze", "mcp_tool_calls_at_freeze", "sha256",
        ],
        "properties": {
            "schema": {"const": SCHEMA},
            "experiment": {"const": EXPERIMENT},
            "status": {"const": "AWAITING_USER_APPROVAL"},
            "authority": {"type": "array", "minItems": 1, "items": {"type": "string"}},
            "design": {
                "type": "object",
                "required": [
                    "arms", "session_length", "paired_sessions", "reuse_factor",
                    "budget_pct", "granularity", "planned_valid_turns",
                    "window_completed_turns",
                ],
                "properties": {
                    "arms": {"const": list(ARMS)},
                    "session_length": {"const": SESSION_LENGTH},
                    "paired_sessions": {"const": SESSION_COUNT},
                    "reuse_factor": {"const": REUSE_FACTOR},
                    "budget_pct": {"const": BUDGET_PCT},
                    "granularity": {"const": "whole-document E5-G1/H16"},
                    "planned_valid_turns": {"const": PLANNED_VALID_TURNS},
                    "window_completed_turns": {"const": WINDOW_COMPLETED_TURNS},
                },
            },
            "reference_cell": {"type": "object"},
            "runtime_pin": {"type": "object"},
            "attempt_contract": {"type": "object"},
            "analysis_contract": {"type": "object"},
            "decision_contract": {"type": "object"},
            "stopping_contract": {"type": "object"},
            "privacy_contract": {"type": "object"},
            "fixture": {"type": "object"},
            "schedule_sha256": {"const": H16_SCHEDULE_SHA256},
            "preparation_source_sha256": {"type": "object"},
            "preparation_base_git_hash": {"type": "string", "pattern": "^[0-9a-f]{40}$"},
            "cost_plan": {"type": "object"},
            "approval_gate": {"type": "object"},
            "model_calls_at_freeze": {"const": 0},
            "embedding_calls_at_freeze": {"const": 0},
            "mcp_tool_calls_at_freeze": {"const": 0},
            "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        },
    }


def turn_json_schema() -> dict:
    """Return the content-free normalized decision-bearing row schema."""
    required = [
        "schema", "arm", "session_id", "position", "task_id",
        "failure_class", "passed", "session_mode", "history_turns_supplied",
        "discarded_prior_turns", "compaction_summary_calls", "retrieval_mode",
        "provider_session_reused", "model_spend_attempts",
        "infra_failure_attempts", "api_input_tokens_no_cache",
        "api_input_tokens_with_cache", "api_cache_read_tokens",
        "api_output_tokens", "api_reasoning_tokens", "mcp_search_calls",
        "mcp_get_calls", "prompt_sha256", "privacy",
    ]
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "urn:k-arc:e11-base3:turn:v1",
        "title": "E11-BASE3 normalized turn row",
        "type": "object",
        "additionalProperties": True,
        "required": required,
        "properties": {
            "schema": {"const": TURN_SCHEMA},
            "arm": {"enum": list(ARMS)},
            "session_id": {"type": "string", "pattern": "^S[0-9]{2}$"},
            "position": {"type": "integer", "minimum": 1, "maximum": SESSION_LENGTH},
            "task_id": {"type": "string"},
            "failure_class": {"type": "string"},
            "passed": {"type": "boolean"},
            "session_mode": {
                "enum": [
                    "persistent-unbounded", "rebuilt-bounded-window",
                    "isolated-per-turn",
                ],
            },
            "history_turns_supplied": {"type": "integer", "minimum": 0, "maximum": 15},
            "discarded_prior_turns": {"type": "integer", "minimum": 0, "maximum": 15},
            "compaction_summary_calls": {"const": 0},
            "retrieval_mode": {
                "enum": ["none-full-corpus-injected", "frozen-bm25-current-turn"],
            },
            "provider_session_reused": {"type": "boolean"},
            "model_spend_attempts": {"type": "integer", "minimum": 0, "maximum": 3},
            "infra_failure_attempts": {"type": "integer", "minimum": 0},
            "api_input_tokens_no_cache": {"type": "integer", "minimum": 0},
            "api_input_tokens_with_cache": {"type": "integer", "minimum": 0},
            "api_cache_read_tokens": {"type": "integer", "minimum": 0},
            "api_output_tokens": {"type": "integer", "minimum": 0},
            "api_reasoning_tokens": {"type": "integer", "minimum": 0},
            "mcp_search_calls": {"const": 0},
            "mcp_get_calls": {"const": 0},
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
    reference_hashes: dict[str, str],
) -> dict:
    """Build the self-hashed, observation-independent scientific contract."""
    audit = validate_bundle(bundle)
    if not audit["pass"]:
        raise ValueError(f"invalid E11-BASE3 bundle: {audit['failures']}")
    snapshot = g1.bundle_snapshot(bundle)
    value = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "status": "AWAITING_USER_APPROVAL",
        "authority": [
            "docs/analysis/e11-paper2-consolidation--codex-workorder.md §0/§1/Lane P2-B",
            "docs/analysis/e9-writeup-and-release--codex-workorder.md §0/§1/Lane P2",
            "docs/experiments/E10-P2-H16/preregistration.md",
            "docs/analysis/benchmark-credibility--adoption-norms.md P5/C2",
        ],
        "design": {
            "objective": (
                "test trivial full-context, bounded-runtime, and no-session-state "
                "baselines before making Paper 2 lifecycle-cost claims"
            ),
            "arms": list(ARMS),
            "session_length": SESSION_LENGTH,
            "paired_sessions": SESSION_COUNT,
            "reuse_factor": REUSE_FACTOR,
            "reuse_definition": bundle["schedule"]["definition"],
            "reuse_denominator": SESSION_COUNT * (SESSION_LENGTH - 1),
            "reuse_count": int(REUSE_FACTOR * SESSION_COUNT * (SESSION_LENGTH - 1)),
            "budget_pct": BUDGET_PCT,
            "granularity": "whole-document E5-G1/H16",
            "planned_valid_turns": PLANNED_VALID_TURNS,
            "schedule_seed": SCHEDULE_SEED,
            "same_task_schedule_for_all_arms": True,
            "window_completed_turns": WINDOW_COMPLETED_TURNS,
            "arm_contracts": {
                "full-history": {
                    "corpus_supply": "all 408 fixture artifacts / 35,128 fixture tokens at session start",
                    "cross_turn_state": "all prior task messages and answers retained",
                    "eviction": "none",
                    "retrieval": "none",
                    "provider_session": "one persistent thread per 16-turn session",
                },
                "sliding-window-compaction": {
                    "corpus_supply": "current turn frozen H16 BM25 plan",
                    "cross_turn_state": "most recent four completed task/evidence/answer turns replayed verbatim",
                    "eviction": "drop oldest completed turn when the four-turn window is full",
                    "retrieval": "frozen H16 BM25; no MCP",
                    "provider_session": "new isolated thread per turn with deterministic window replay",
                    "semantic_summary": "none; no extra compactor model call",
                },
                "stateless-rag": {
                    "corpus_supply": "current turn frozen H16 BM25 plan",
                    "cross_turn_state": "none",
                    "eviction": "all prior turn state absent by construction",
                    "retrieval": "frozen H16 BM25; no MCP",
                    "provider_session": "new isolated thread per turn",
                },
            },
            "stateless_rag_difference_from_h16_rag_bm25": (
                "H16 rag-bm25 used one persistent 16-turn provider conversation and "
                "therefore carried prior prompts, BM25 evidence, answers, and provider "
                "cache; stateless-rag creates a fresh isolated provider session for "
                "every turn and supplies only that turn's frozen BM25 evidence"
            ),
        },
        "reference_cell": {
            "experiment": "E10-P2-H16",
            "arms": list(REFERENCE_ARMS),
            "reuse_existing_control_rows": True,
            "new_control_model_calls": 0,
            "execution_batch_difference": (
                "E11 baseline rows are a later execution batch; comparisons are "
                "permitted only because fixture, schedule, reader pin, reasoning, "
                "auth mode, and accounting are exact and their hashes are frozen"
            ),
            "schedule_sha256": snapshot["sha256"],
            "config_self_sha256": H16_CONFIG_SELF_SHA256,
            **reference_hashes,
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
            "expected_model_spend_attempts_without_failure": PLANNED_VALID_TURNS,
            "model_spend_attempt_cap": MODEL_SPEND_ATTEMPT_CAP,
            "model_spend_definition": (
                "successful turn or any failed call with non-zero normalized model usage"
            ),
            "max_model_spend_attempts_per_turn": MAX_MODEL_SPEND_ATTEMPTS_PER_TURN,
            "infra_failure_attempt_cap": INFRA_FAILURE_ATTEMPT_CAP,
            "infra_failure_definition": "failed call with zero normalized model usage",
            "total_api_call_cap": TOTAL_API_CALL_CAP,
            "provider_probe_attempt_cap_separate": PROVIDER_PROBE_ATTEMPT_CAP,
            "infra_failures_are_append_only_and_do_not_refund_total_api_call_cap": True,
        },
        "analysis_contract": {
            "accounting": "F=fresh input; R=gross-fresh cache-read; C_w=F+wR",
            "cache_read_weights": list(CACHE_READ_WEIGHTS),
            "primary_horizon": 16,
            "secondary_horizons": list(HORIZONS),
            "primary_outputs": [
                "arm-by-position mean F, R, and gross for each new baseline",
                "paired-session cumulative C_w differences and ratios among all three new arms",
                "paired-session cumulative C_w differences and ratios versus each frozen H16 reference arm",
                "exact-answer correctness by arm and horizon",
            ],
            "uncertainty": (
                "paired-session percentile bootstrap, 10000 resamples, two-sided 95% CI"
            ),
            "quality_noninferiority": (
                "paired exact-answer difference; lower 95% CI must be >= -0.05"
            ),
            "quality_noninferiority_margin": QUALITY_NONINFERIORITY_MARGIN,
            "curve_fits": "descriptive positions 1..16 only; no extrapolation",
            "claim_scope": (
                "this frozen H16 fixture, schedule, reader pin, accounting, and "
                "observed positions 1..16 only"
            ),
            "forbidden_interpretations": [
                "no structural or asymptotic complexity claim",
                "no extrapolation beyond position 16",
                "no K-ARC performance-superiority claim",
                "no cross-runtime comparison",
                "do not mix H16 session-gross with any other cell or runtime axis",
            ],
        },
        "decision_contract": {
            "paper2_premise_collapse": {
                "label": "PAPER2_PREMISE_REWRITE_REQUIRED",
                "cost_condition": (
                    "at H=16, for full-history minus each of karc-full and rag-bm25, "
                    "the paired 95% CI upper bound is <0 for every w in {0,0.1,1}"
                ),
                "quality_condition": (
                    "full-history minus each reference arm has paired correctness "
                    "95% CI lower bound >= -0.05"
                ),
                "conjunction": "cost_condition AND quality_condition",
                "required_action": (
                    "Paper 2 may not retain a memory-system-necessity premise; rewrite "
                    "the thesis around the trivial full-history result and report it first"
                ),
            },
            "stateless_state_necessity_failure": {
                "label": "SESSION_STATE_NECESSITY_CLAIM_REMOVE",
                "condition": (
                    "at H=16 stateless-rag has paired 95% CI upper bound <0 versus "
                    "H16 rag-bm25 for every w and paired correctness lower bound >= -0.05"
                ),
                "required_action": (
                    "remove any claim that session state is necessary on this cell; "
                    "retain only accounting observations that survive"
                ),
            },
            "quality_guard": (
                "if a cost-cheaper arm fails the -5pp paired correctness margin, "
                "report its token curve but withhold a cost-optimal recommendation"
            ),
            "null_disposition": (
                "failure to trigger a rewrite rule is not evidence of K-ARC superiority"
            ),
            "adjudicator": "main/user only",
        },
        "stopping_contract": {
            "before_first_turn": [
                "stop until the user explicitly approves this E11-BASE3 execution and caps",
                "stop if Codex CLI is not exactly 0.144.5",
                "stop if gpt-5.6-luna/high under ChatGPT subscription cannot be selected; do not substitute",
                "stop on fixture, H16 schedule, schema, reference raw, or execution-source hash mismatch",
                "stop if the approved exact-pin probe reports a usage-limit or quota block",
            ],
            "during_run": [
                "stop globally on any usage-limit or quota error; do not switch provider, model, account, or runtime",
                "stop globally on reported-model or reasoning-pin mismatch",
                "stop globally on missing or internally inconsistent token telemetry",
                "stop globally on context overflow or runtime truncation; do not shorten, compact, or substitute the registered arm",
                "stop globally if any E11 arm records a K-ARC MCP search/get call",
                "stop globally if state-window/session-isolation instrumentation violates the registered arm contract",
                "stop when any registered attempt cap is reached",
            ],
            "task_errors": "grade and retain normalized metadata; task-answer error alone is not a stop",
        },
        "privacy_contract": {
            "persist": [
                "normalized token usage", "grade", "IDs", "counts", "hashes",
                "state-window counts", "failure category and hashed error digest",
            ],
            "forbid": sorted(FORBIDDEN_RAW_KEYS),
            "raw_is_append_only": True,
            "original_fixture_content_is_not_copied_to_experiment_raw": True,
            "transient_window_replay_is_never_persisted": True,
        },
        "fixture": {
            "root": "fixture/e4-v2",
            "corpus_seed": g1.CORPUS_SEED,
            "corpus_content_hash": bundle["manifest"]["corpus_content_hash"],
            "source_manifest_sha256": snapshot["schedule"]["source_e4_manifest_sha256"],
            "manifest_file_sha256": reference_hashes["fixture_manifest_file_sha256"],
            "artifact_count": len(bundle["manifest"]["artifacts"]),
            "corpus_tokens": sum(
                int(row["size_tok"]) for row in bundle["manifest"]["artifacts"].values()
            ),
            "budget_tokens": bundle["manifest"]["cell"]["budget_tokens"],
        },
        "schedule_sha256": snapshot["sha256"],
        "preparation_source_sha256": source_sha256,
        "preparation_base_git_hash": base_git_hash,
        "cost_plan": {
            "planned_valid_turns": PLANNED_VALID_TURNS,
            "expected_model_spend_attempts_without_failure": PLANNED_VALID_TURNS,
            "planning_gross_tokens": {
                "low_scenario": 57_836_510,
                "high_scenario": 72_541_596,
                "not_a_confidence_interval_or_cap": True,
            },
            "scenario_method": (
                "stateless proxy = H16 rag-bm25 position-1 mean x192 (2,346,160); "
                "sliding proxy = H16 rag-bm25 positions 1..4 plus position-5 mean "
                "repeated through position 16 (14,252,483); full-history bracketed "
                "by the two observed H16 arm gross totals (41,237,867..55,942,953)"
            ),
            "dollar_cost": None,
            "dollar_cost_note": (
                "ChatGPT subscription telemetry did not expose actual billed dollars; "
                "retry usage may exceed the planning scenarios but not the attempt caps"
            ),
        },
        "approval_gate": {
            "scientific_contract_edits_after_approval": "FORBIDDEN; amendment required",
            "execution_source_freeze": (
                "must be created after approval and before the first probe/model turn; "
                "the current runner is a non-executing skeleton"
            ),
            "required_user_approvals": [
                "576 planned valid turns and 720 model-spend attempt cap",
                "144 zero-token infra failures and 864 total API-call cap",
                "three exact arm operationalizations including W=4 and no compactor calls",
                "zero K-ARC MCP calls in all E11 arms",
                "separate exact-pin provider probe capped at 2 attempts",
                "reuse of frozen H16 control rows and their later-batch limitation",
                "planning token scenarios and unavailable exact dollar amount",
                "stop without substitution on usage-limit or any pin/hash mismatch",
            ],
        },
        "model_calls_at_freeze": 0,
        "embedding_calls_at_freeze": 0,
        "mcp_tool_calls_at_freeze": 0,
    }
    value["sha256"] = hash_json(value)
    return value


def validate_config(config: dict) -> dict:
    unsigned = dict(config)
    recorded = unsigned.pop("sha256", None)
    design = config.get("design", {})
    runtime = config.get("runtime_pin", {})
    attempts = config.get("attempt_contract", {})
    checks = {
        "self_hash": recorded == hash_json(unsigned),
        "schema": config.get("schema") == SCHEMA,
        "experiment": config.get("experiment") == EXPERIMENT,
        "awaiting_approval": config.get("status") == "AWAITING_USER_APPROVAL",
        "arms": design.get("arms") == list(ARMS),
        "same_h16_axes": (
            design.get("session_length") == SESSION_LENGTH
            and design.get("paired_sessions") == SESSION_COUNT
            and design.get("reuse_factor") == REUSE_FACTOR
            and design.get("budget_pct") == BUDGET_PCT
            and config.get("schedule_sha256") == H16_SCHEDULE_SHA256
        ),
        "window_four": design.get("window_completed_turns") == WINDOW_COMPLETED_TURNS,
        "reader_pin": runtime == {
            "codex_cli": CODEX_CLI_VERSION,
            "model": MODEL,
            "reasoning_effort": REASONING_EFFORT,
            "auth_mode": "ChatGPT subscription",
            "provider_model_availability": "UNKNOWN_WITHOUT_APPROVED_PROBE",
            "substitution": "FORBIDDEN",
            "mismatch_disposition": "STOP_WITHOUT_SUBSTITUTION",
        },
        "planned_valid_turns": attempts.get("planned_valid_turns") == PLANNED_VALID_TURNS,
        "model_spend_cap": attempts.get("model_spend_attempt_cap") == MODEL_SPEND_ATTEMPT_CAP,
        "infra_cap": attempts.get("infra_failure_attempt_cap") == INFRA_FAILURE_ATTEMPT_CAP,
        "total_cap": attempts.get("total_api_call_cap") == TOTAL_API_CALL_CAP,
        "probe_cap": attempts.get("provider_probe_attempt_cap_separate") == PROVIDER_PROBE_ATTEMPT_CAP,
        "h16_reference_self_hash": (
            config.get("reference_cell", {}).get("config_self_sha256")
            == H16_CONFIG_SELF_SHA256
        ),
        "calls_zero": (
            config.get("model_calls_at_freeze") == 0
            and config.get("embedding_calls_at_freeze") == 0
            and config.get("mcp_tool_calls_at_freeze") == 0
        ),
        "collapse_rule_frozen": (
            config.get("decision_contract", {})
            .get("paper2_premise_collapse", {})
            .get("label") == "PAPER2_PREMISE_REWRITE_REQUIRED"
        ),
        "no_substitution_stop": any(
            "usage-limit" in row and "do not switch" in row
            for row in config.get("stopping_contract", {}).get("during_run", [])
        ),
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "failures": [name for name, passed in checks.items() if not passed],
    }


def _expected_task_ids(bundle: dict) -> dict[tuple[str, int], str]:
    expected = {}
    for session in bundle["sessions"]:
        for item in session["tasks"]:
            position = int(item["task"]["session_task"])
            expected[(session["session_id"], position)] = item["task"]["task_id"]
    return expected


def audit_turn_rows(
    rows: Iterable[dict], *, bundle: dict | None = None, require_complete: bool,
) -> dict:
    """Audit mock or future normalized rows against caps, state, and privacy."""
    rows = list(rows)
    expected_tasks = _expected_task_ids(bundle) if bundle is not None else None
    seen: set[tuple[str, str, int]] = set()
    duplicates: list[tuple[str, str, int]] = []
    schema_mismatches = 0
    task_mismatches = 0
    token_mismatches = 0
    state_mismatches = 0
    privacy_keys: set[str] = set()
    mcp_calls = 0
    per_turn_attempt_violations = 0
    invalid_rows = 0
    model_spend_attempts = 0
    infra_failure_attempts = 0
    for row in rows:
        arm = row.get("arm")
        session_id = row.get("session_id")
        position = row.get("position")
        if (
            row.get("schema") != TURN_SCHEMA
            or arm not in ARMS
            or not isinstance(session_id, str)
            or not isinstance(position, int)
            or not 1 <= position <= SESSION_LENGTH
        ):
            schema_mismatches += 1
            invalid_rows += 1
            continue
        key = (arm, session_id, position)
        if key in seen:
            duplicates.append(key)
        seen.add(key)
        if expected_tasks is not None and row.get("task_id") != expected_tasks.get((session_id, position)):
            task_mismatches += 1
        fresh = row.get("api_input_tokens_no_cache")
        gross = row.get("api_input_tokens_with_cache")
        cached = row.get("api_cache_read_tokens")
        if (
            not all(isinstance(value, int) and value >= 0 for value in (fresh, gross, cached))
            or gross != fresh + cached
        ):
            token_mismatches += 1
        expected = expected_state(arm, position)
        if any(row.get(name) != value for name, value in expected.items()):
            state_mismatches += 1
        row_mcp = int(row.get("mcp_search_calls", 0)) + int(row.get("mcp_get_calls", 0))
        mcp_calls += row_mcp
        spend = row.get("model_spend_attempts", 0)
        infra = row.get("infra_failure_attempts", 0)
        if not isinstance(spend, int) or not 0 <= spend <= MAX_MODEL_SPEND_ATTEMPTS_PER_TURN:
            per_turn_attempt_violations += 1
        else:
            model_spend_attempts += spend
        if not isinstance(infra, int) or infra < 0:
            per_turn_attempt_violations += 1
        else:
            infra_failure_attempts += infra
        privacy_keys.update(FORBIDDEN_RAW_KEYS.intersection(row))
        if row.get("privacy") != "normalized-no-content":
            invalid_rows += 1
    expected_keys = {
        (arm, f"S{session:02d}", position)
        for arm in ARMS
        for session in range(1, SESSION_COUNT + 1)
        for position in range(1, SESSION_LENGTH + 1)
    }
    missing = expected_keys - seen if require_complete else set()
    unexpected = seen - expected_keys
    total_calls = model_spend_attempts + infra_failure_attempts
    checks = {
        "row_shape": invalid_rows == 0 and schema_mismatches == 0,
        "unique_keys": not duplicates,
        "coverage": not missing and not unexpected,
        "task_schedule": task_mismatches == 0,
        "token_identity": token_mismatches == 0,
        "state_contract": state_mismatches == 0,
        "mcp_zero": mcp_calls == 0,
        "privacy": not privacy_keys,
        "per_turn_attempt_cap": per_turn_attempt_violations == 0,
        "model_spend_cap": model_spend_attempts <= MODEL_SPEND_ATTEMPT_CAP,
        "infra_cap": infra_failure_attempts <= INFRA_FAILURE_ATTEMPT_CAP,
        "total_call_cap": total_calls <= TOTAL_API_CALL_CAP,
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "failures": [name for name, passed in checks.items() if not passed],
        "rows": len(rows),
        "expected_rows": PLANNED_VALID_TURNS if require_complete else None,
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "duplicate_keys": len(duplicates),
        "task_mismatches": task_mismatches,
        "token_accounting_mismatches": token_mismatches,
        "state_contract_mismatches": state_mismatches,
        "mcp_calls": mcp_calls,
        "privacy_forbidden_keys": sorted(privacy_keys),
        "model_spend_attempts": model_spend_attempts,
        "infra_failure_attempts": infra_failure_attempts,
        "total_api_calls": total_calls,
    }
