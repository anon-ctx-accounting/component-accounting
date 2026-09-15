"""Pre-execution contract for the E5-G1 Claude-runtime replication.

This module is intentionally model-free.  It normalizes provider usage,
constructs inspectable (but unexecuted) Claude CLI argument vectors, validates
approval records, and computes the post-smoke canary cap proposal.  Actual
provider execution must live behind the approval gates described in
``docs/analysis/e10-p2-xr-preregistration.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


EXPERIMENT_ID = "E10-P2-XR"
MODEL_PIN = "claude-sonnet-5"
CLI_PIN = "2.1.220 (Claude Code)"
SESSION_LENGTH = 8
REUSE_FACTOR = 0.5
BUDGET_PCT = 5
SMOKE_PAIRED_SESSIONS = 4
CANARY_PAIRED_SESSIONS = 12
SMOKE_VALID_TURNS = 64
CANARY_VALID_TURNS = 192
SMOKE_MODEL_SPEND_CAP = 80
CANARY_MODEL_SPEND_CAP = 240
EXECUTION_EPOCH = "tool-exposure-v4"
ARMS = ("karc-full", "rag-bm25")
ALLOWED_KARC_TOOLS = ("mcp__karc__search", "mcp__karc__get")
FORBIDDEN_NETWORK_TOOLS = ("WebSearch", "WebFetch")
FORBIDDEN_NATIVE_TOOLS = ("Bash", "Read", "Grep", "Glob")
KARC_FORBIDDEN_NATIVE_TOOLS = (
    "Agent", "AskUserQuestion", "Bash", "Edit", "EnterPlanMode",
    "ExitPlanMode", "Glob", "Grep", "LSP", "NotebookEdit", "Read", "Skill",
    "Task", "TaskOutput", "TaskStop", "TodoWrite", "WebFetch", "WebSearch",
    "ToolSearch", "Write",
)
PRIVACY_FORBIDDEN_KEYS = frozenset({
    "prompt", "output", "output_text", "transcript", "command",
    "tool_response", "credential", "credentials", "api_key", "oauth_token",
})


class ContractError(ValueError):
    """Raised before provider execution when a binding condition is unmet."""


def _token(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ContractError(f"{name} must be a non-negative integer")
    try:
        parsed = int(value or 0)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{name} must be a non-negative integer") from exc
    if parsed < 0:
        raise ContractError(f"{name} must be a non-negative integer")
    return parsed


def normalize_claude_usage(usage: Mapping[str, object]) -> dict[str, int]:
    """Return the only cross-runtime comparable input axis plus diagnostics.

    Claude's ``input_tokens`` is a provider-specific fresh bucket.  Cache
    creation and cache-read are separate buckets.  Their sum is gross logical
    input; only that sum may be compared with another runtime.
    """
    fresh = _token(usage.get("input_tokens"), "input_tokens")
    creation = _token(
        usage.get("cache_creation_input_tokens"),
        "cache_creation_input_tokens",
    )
    creation_detail = usage.get("cache_creation") or {}
    if not isinstance(creation_detail, Mapping):
        raise ContractError("cache_creation must be an object")
    c5 = _token(
        creation_detail.get("ephemeral_5m_input_tokens"),
        "ephemeral_5m_input_tokens",
    )
    c1 = _token(
        creation_detail.get("ephemeral_1h_input_tokens"),
        "ephemeral_1h_input_tokens",
    )
    if c5 or c1:
        if creation != c5 + c1:
            raise ContractError("Claude cache-creation bucket does not close")
    read = _token(
        usage.get("cache_read_input_tokens"),
        "cache_read_input_tokens",
    )
    output = _token(usage.get("output_tokens"), "output_tokens")
    return {
        "fresh_input_tokens_provider": fresh,
        "cache_creation_input_tokens_provider": creation,
        "cache_creation_5m_input_tokens_provider": c5,
        "cache_creation_1h_input_tokens_provider": c1,
        "cache_read_input_tokens_provider": read,
        "gross_input_tokens": fresh + creation + read,
        "output_tokens": output,
    }


def normalize_codex_usage(usage: Mapping[str, object]) -> dict[str, int]:
    """Normalize committed Codex raw without equating its fresh schema to Claude."""
    gross = _token(usage.get("input_tokens"), "input_tokens")
    cached = _token(usage.get("cached_input_tokens"), "cached_input_tokens")
    if cached > gross:
        raise ContractError("Codex cached_input_tokens exceeds gross input_tokens")
    return {
        "fresh_input_tokens_provider": gross - cached,
        "cache_read_input_tokens_provider": cached,
        "gross_input_tokens": gross,
        "output_tokens": _token(usage.get("output_tokens"), "output_tokens"),
    }


@dataclass(frozen=True)
class ClaudeRates:
    """Approved dollars per million tokens for one smoke execution window."""

    fresh: float
    cache_creation_5m: float
    cache_creation_1h: float
    cache_read: float
    output: float

    def __post_init__(self) -> None:
        if min(
            self.fresh,
            self.cache_creation_5m,
            self.cache_creation_1h,
            self.cache_read,
            self.output,
        ) < 0:
            raise ContractError("all price rates must be non-negative")


def priced_claude_cost(usage: Mapping[str, object], rates: ClaudeRates) -> float:
    normalized = normalize_claude_usage(usage)
    creation_unsplit = normalized["cache_creation_input_tokens_provider"]
    c5 = normalized["cache_creation_5m_input_tokens_provider"]
    c1 = normalized["cache_creation_1h_input_tokens_provider"]
    if creation_unsplit and not (c5 or c1):
        # The schema did not expose TTL detail.  Conservative execution policy:
        # price all creation at the larger approved creation rate.
        c1 = creation_unsplit
        creation_rate = max(rates.cache_creation_5m, rates.cache_creation_1h)
    else:
        creation_rate = rates.cache_creation_1h
    cost = (
        normalized["fresh_input_tokens_provider"] * rates.fresh
        + c5 * rates.cache_creation_5m
        + c1 * creation_rate
        + normalized["cache_read_input_tokens_provider"] * rates.cache_read
        + normalized["output_tokens"] * rates.output
    ) / 1_000_000
    return cost


def propose_canary_dollar_cap(pair_session_costs: Sequence[float]) -> dict[str, float]:
    """Conservative proposal shown to the user after the four-session smoke.

    The 12-session canary is three times the smoke cardinality.  The proposal
    takes the larger of the straight 3x projection and twelve copies of the
    most expensive observed paired session, then adds a 25% reserve.  It is a
    proposal only; an explicit user-approved value is still required.
    """
    if len(pair_session_costs) != SMOKE_PAIRED_SESSIONS:
        raise ContractError("cap proposal requires exactly four paired sessions")
    costs = [float(value) for value in pair_session_costs]
    if any((not math.isfinite(value)) or value < 0 for value in costs):
        raise ContractError("paired-session costs must be finite and non-negative")
    smoke_total = sum(costs)
    straight_projection = 3.0 * smoke_total
    max_session_envelope = CANARY_PAIRED_SESSIONS * max(costs, default=0.0)
    proposed = 1.25 * max(straight_projection, max_session_envelope)
    proposed = math.ceil(proposed * 100) / 100
    return {
        "smoke_total_usd": smoke_total,
        "straight_12_session_projection_usd": straight_projection,
        "max_session_12_session_envelope_usd": max_session_envelope,
        "reserve_multiplier": 1.25,
        "proposed_canary_cap_usd": proposed,
    }


def validate_approval(
    approval: Mapping[str, object], *, phase: str, expected_plan_sha256: str,
    expected_smoke_sha256: str | None = None,
) -> None:
    """Reject implicit, stale, or model/CLI-substituting approvals."""
    if phase not in {"smoke", "canary"}:
        raise ContractError(f"unknown approval phase {phase!r}")
    if approval.get("approved") is not True:
        raise ContractError(f"{phase} execution lacks explicit user approval")
    if approval.get("experiment_id") != EXPERIMENT_ID:
        raise ContractError("approval experiment_id mismatch")
    if approval.get("phase") != phase:
        raise ContractError("approval phase mismatch")
    if approval.get("plan_sha256") != expected_plan_sha256:
        raise ContractError("approval is stale relative to the frozen plan")
    if approval.get("model_pin") != MODEL_PIN:
        raise ContractError("model substitution is forbidden")
    if approval.get("cli_pin") != CLI_PIN:
        raise ContractError("CLI substitution is forbidden")
    cap = approval.get("dollar_cap_usd")
    if not isinstance(cap, (int, float)) or isinstance(cap, bool) or cap <= 0:
        raise ContractError("approval requires a positive dollar cap")
    expected_attempt_cap = (
        SMOKE_MODEL_SPEND_CAP if phase == "smoke" else CANARY_MODEL_SPEND_CAP
    )
    if approval.get("model_spend_attempt_cap") != expected_attempt_cap:
        raise ContractError("model-spend attempt cap mismatch")
    if approval.get("karc_mcp_allowed") is not True:
        raise ContractError("K-ARC MCP permission must be explicit")
    if approval.get("rag_mcp_allowed") is not False:
        raise ContractError("RAG arm must forbid every MCP server")
    if phase == "canary" and approval.get("smoke_sha256") != expected_smoke_sha256:
        raise ContractError("canary approval is stale relative to the smoke")


def _walk_keys(value: object) -> list[str]:
    keys: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            keys.append(str(key).lower())
            keys.extend(_walk_keys(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            keys.extend(_walk_keys(child))
    return keys


def validate_persisted_row(row: Mapping[str, object], *, arm: str) -> None:
    """Fail closed on privacy, model verification, gross closure, and RAG MCP."""
    forbidden = sorted(set(_walk_keys(row)) & PRIVACY_FORBIDDEN_KEYS)
    if forbidden:
        raise ContractError(f"privacy-forbidden raw keys: {forbidden}")
    if row.get("reported_model") != MODEL_PIN:
        raise ContractError("reported answer-producing model differs from exact pin")
    if row.get("model_verification") != "assistant-event-exact":
        raise ContractError("model pin was not verified from the assistant event")
    gross = _token(row.get("gross_input_tokens"), "gross_input_tokens")
    bucket_sum = sum(_token(row.get(name), name) for name in (
        "fresh_input_tokens_provider",
        "cache_creation_input_tokens_provider",
        "cache_read_input_tokens_provider",
    ))
    if gross != bucket_sum:
        raise ContractError("persisted gross input does not equal provider buckets")
    if arm == "rag-bm25" and _token(row.get("mcp_calls"), "mcp_calls") != 0:
        raise ContractError("RAG arm persisted a nonzero MCP count")


def validate_stage_ledger(*, phase: str, valid_turns: int,
                          model_spend_attempts: int) -> None:
    expected = {
        "smoke": (SMOKE_VALID_TURNS, SMOKE_MODEL_SPEND_CAP),
        "canary": (CANARY_VALID_TURNS, CANARY_MODEL_SPEND_CAP),
    }
    if phase not in expected:
        raise ContractError(f"unknown stage {phase!r}")
    planned_valid, cap = expected[phase]
    if valid_turns > planned_valid:
        raise ContractError("valid turns exceed the frozen stage cardinality")
    if model_spend_attempts > cap:
        raise ContractError("model-spend attempt cap exceeded")


def build_claude_session_argv(
    *, prompt: str, arm: str, session_id: str, resume: bool,
    mcp_config: str | Path, settings: str | Path, model: str = MODEL_PIN,
    max_agent_turns: int = 8,
) -> list[str]:
    """Build an auditable persistent-session argv; never invokes the CLI."""
    if arm not in ARMS:
        raise ContractError(f"unknown arm {arm!r}")
    if model != MODEL_PIN:
        raise ContractError("model substitution is forbidden")
    if not session_id:
        raise ContractError("session_id is required")
    allowed_tools = ",".join(ALLOWED_KARC_TOOLS) if arm == "karc-full" else ""
    denied = (KARC_FORBIDDEN_NATIVE_TOOLS if arm == "karc-full" else
              (*FORBIDDEN_NETWORK_TOOLS, *FORBIDDEN_NATIVE_TOOLS))
    argv = [
        "claude", "-p", prompt,
        "--model", model,
        "--output-format", "stream-json",
        "--verbose",
        "--max-turns", str(max_agent_turns),
        "--permission-mode", "bypassPermissions",
        "--disallowedTools", ",".join(denied),
    ]
    # Claude Code 2.1.220 defines --tools as the built-in tool set.  MCP tools
    # are exposed by the strict MCP config and permissioned with --allowedTools.
    # Retain the frozen RAG argv byte-for-byte; omit --tools only for K-ARC.
    if arm == "rag-bm25":
        argv += ["--tools", ""]
    argv += [
        "--allowedTools", allowed_tools,
        "--disable-slash-commands",
        "--no-chrome",
        "--strict-mcp-config",
        "--mcp-config", str(mcp_config),
        "--settings", str(settings),
        "--setting-sources", "",
    ]
    if resume:
        argv += ["--resume", session_id]
    else:
        argv += ["--session-id", session_id]
    return argv


def assert_rag_mcp_zero(transcript: Sequence[Mapping[str, object]]) -> None:
    for event in transcript:
        name = str(event.get("name") or "")
        if event.get("kind") == "tool_use" and name.startswith("mcp__"):
            raise ContractError("RAG arm emitted an MCP tool call")


def assert_karc_tool_surface(transcript: Sequence[Mapping[str, object]]) -> None:
    allowed = set(ALLOWED_KARC_TOOLS)
    for event in transcript:
        if event.get("kind") != "tool_use":
            continue
        name = str(event.get("name") or "")
        if name not in allowed:
            raise ContractError(f"K-ARC arm used forbidden tool {name!r}")
