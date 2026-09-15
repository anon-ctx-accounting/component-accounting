"""FR-H2: provider-neutral headless drivers + injectable mock.

The harness never binds to a concrete LLM. It drives a ``Driver`` that takes a
``DriverRequest`` (materialized cwd, prompt, pinned model, tool/permission
policy) and returns a ``DriverResult`` (final text, normalized transcript,
token usage, reported model, error class).

- ``ClaudeCliDriver`` shells out to ``claude -p`` (non-interactive) — the real
  path, exercised only during actual Stage 2 / E0-3 execution.
- ``CodexCliDriver`` shells out to ``codex exec --json --ephemeral`` inside a
  controlled ``CODEX_HOME``.  It normalizes Codex JSONL into the same result
  contract without persisting prompts, command strings, or model output.
- ``MockDriver`` wraps an injectable handler and is used for all development
  and the entire test suite. **No real LLM call happens under test** (project
  constraint); the CLI path is constructed and unit-tested for its argv/parse
  logic with a fake ``subprocess`` runner, never invoked against the API.

Model-ID pin (FR-H2, §8.3): ``claude -p`` cannot report the model *before*
generation, so the pin is enforced two ways — the ``--model`` flag is passed
to the CLI, and the reported model on the result is verified against the pin
by the harness (mismatch aborts the whole benchmark). ``preflight`` captures
``claude --version`` (the only real-CLI touch allowed outside execution).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# Network tools are disabled in every run (§8.3 "네트워크 도구 비활성").
DEFAULT_DISALLOWED_TOOLS = ("WebSearch", "WebFetch")

# P3 (E2-2 diagnosis): subscription usage-limit refusals are NOT retryable API
# faults — retrying burns attempts instantly (E2-2: 120 runs × 3 attempts lost
# in one contiguous block) and the only correct reaction is to pause the whole
# benchmark until the limit window resets. Textual signature of the refusal:
USAGE_LIMIT_RE = re.compile(
    r"(?i)(?:\b(?:usage|session|weekly|\d+\s*-?\s*hour)\s+limit\b|limit reached"
    r"|out of extra usage|resets? at? ?\d)")


@dataclass
class DriverRequest:
    prompt: str
    cwd: str
    model: str
    max_turns: int = 30
    # ``None`` = CLI default tool set; ``[]`` = no tools (closed-book single shot).
    allowed_tools: list[str] | None = None
    disallowed_tools: tuple[str, ...] = DEFAULT_DISALLOWED_TOOLS
    permission_mode: str = "bypassPermissions"
    mcp_config: str | None = None
    settings: str | None = None
    timeout_s: int = 300
    extra_args: tuple[str, ...] = ()
    runtime: str = "claude-code"
    config_overrides: tuple[str, ...] = ()
    environment: dict[str, str] = field(default_factory=dict)
    hook_mode: str = "observe"
    reasoning_effort: str = "high"


@dataclass
class DriverResult:
    ok: bool
    output_text: str = ""
    transcript: list[dict] = field(default_factory=list)  # normalized events
    usage: dict = field(default_factory=dict)  # raw token usage from the API
    reported_model: str | None = None
    num_turns: int | None = None
    # ``error_class`` ∈ {None, "api_error", "harness_error", "usage_limit"} —
    # api/harness errors are auto-retried by the scheduler (FR-H6);
    # ``usage_limit`` (P3) is NOT retried: it is a benchmark-wide pause/abort
    # signal. ``None`` = clean completion (grading decides pass/fail).
    error_class: str | None = None
    error_detail: str | None = None
    raw: dict | None = None
    usage_raw: dict = field(default_factory=dict)
    native_session_id: str | None = None
    model_verification: str | None = None
    driver_metadata: dict = field(default_factory=dict)
    # Amount the harness itself reports for the call.  Claude Code emits
    # ``total_cost_usd`` on the terminal ``result`` event and a per-model
    # ``modelUsage[*].costUSD`` breakdown.  Both are the harness's own
    # rate-card conversion of the recorded components, not a billed amount,
    # so they are kept verbatim and never used as a charge.
    harness_reported_cost_usd: float | None = None
    harness_reported_model_cost_usd: dict = field(default_factory=dict)

    def total_input_tokens(self, *, with_cache: bool) -> int:
        """FR-H4: cache-reflected vs cache-excluded input token views.

        ``with_cache=False`` counts only fresh (billed-at-full-rate) input
        tokens; ``with_cache=True`` adds cache-creation and cache-read tokens
        (the full context the model actually saw)."""
        u = self.usage or {}
        base = int(u.get("input_tokens", 0) or 0)
        if with_cache:
            base += int(u.get("cache_creation_input_tokens", 0) or 0)
            base += int(u.get("cache_read_input_tokens", 0) or 0)
        return base

    @property
    def output_tokens(self) -> int:
        return int((self.usage or {}).get("output_tokens", 0) or 0)

    @property
    def reasoning_tokens(self) -> int:
        return int((self.usage or {}).get("reasoning_output_tokens", 0) or 0)


class Driver:
    """Interface: ``run(request) -> DriverResult``."""

    def run(self, request: DriverRequest) -> DriverResult:  # pragma: no cover
        raise NotImplementedError


# --------------------------------------------------------------------------
# Mock driver (all tests / dry-runs)
# --------------------------------------------------------------------------
class MockDriver(Driver):
    """Injectable driver. ``handler(request) -> DriverResult`` produces every
    response, so a test can script task successes, leakage, API errors, etc.
    ``calls`` records each request for assertions."""

    def __init__(self, handler: Callable[[DriverRequest], DriverResult]):
        self._handler = handler
        self.calls: list[DriverRequest] = []

    def run(self, request: DriverRequest) -> DriverResult:
        self.calls.append(request)
        return self._handler(request)


# --------------------------------------------------------------------------
# Real Claude Code headless driver
# --------------------------------------------------------------------------
class ClaudeCliDriver(Driver):
    """``claude -p`` non-interactive driver (FR-H2).

    Uses ``--output-format stream-json --verbose`` so the transcript (tool_use
    blocks, needed by FR-H7 MCP-adoption) and the terminal ``result`` event
    (final text + usage + model) are both captured. ``_runner`` is injectable
    (defaults to ``subprocess.run``) so tests exercise argv assembly and output
    parsing against canned CLI output without any network call.
    """

    def __init__(self, binary: str = "claude", runner: Callable | None = None):
        self.binary = binary
        self._runner = runner or self._subprocess_runner

    # -- argv assembly ------------------------------------------------------
    def build_argv(self, request: DriverRequest) -> list[str]:
        argv = [
            self.binary,
            "-p", request.prompt,
            "--model", request.model,
            "--output-format", "stream-json",
            "--verbose",
            "--max-turns", str(request.max_turns),
            "--permission-mode", request.permission_mode,
        ]
        if request.disallowed_tools:
            argv += ["--disallowedTools", ",".join(request.disallowed_tools)]
        if request.allowed_tools is not None:
            # ``[]`` → empty allow list (no tools; closed-book single shot).
            argv += ["--allowedTools", ",".join(request.allowed_tools)]
        if request.mcp_config:
            argv += ["--mcp-config", request.mcp_config]
        if request.settings:
            argv += ["--settings", request.settings]
        argv += list(request.extra_args)
        return argv

    def _subprocess_runner(self, argv, cwd, timeout):
        return subprocess.run(
            argv, cwd=cwd, capture_output=True, text=True, timeout=timeout
        )

    def run(self, request: DriverRequest) -> DriverResult:
        argv = self.build_argv(request)
        try:
            proc = self._runner(argv, request.cwd, request.timeout_s)
        except subprocess.TimeoutExpired:
            return DriverResult(ok=False, error_class="api_error",
                                error_detail="claude cli timeout")
        except (OSError, subprocess.SubprocessError) as exc:
            return DriverResult(ok=False, error_class="harness_error",
                                error_detail=f"claude cli spawn failed: {exc}")
        return self.parse_output(proc.returncode, proc.stdout, proc.stderr)

    # -- stream-json parsing ------------------------------------------------
    def parse_output(self, returncode: int, stdout: str, stderr: str) -> DriverResult:
        events: list[dict] = []
        for line in (stdout or "").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue

        transcript: list[dict] = []
        output_text = ""
        usage: dict = {}
        harness_cost: float | None = None
        harness_model_cost: dict = {}
        reported_model: str | None = None
        num_turns: int | None = None
        result_error: str | None = None
        result_is_error = False

        for ev in events:
            etype = ev.get("type")
            if etype == "assistant":
                msg = ev.get("message", {})
                reported_model = msg.get("model", reported_model)
                for block in msg.get("content", []) or []:
                    btype = block.get("type")
                    if btype == "tool_use":
                        transcript.append({
                            "kind": "tool_use",
                            "name": block.get("name", ""),
                            "input": block.get("input", {}),
                        })
                    elif btype == "text":
                        transcript.append({"kind": "text", "text": block.get("text", "")})
                if msg.get("usage"):
                    usage = _merge_usage(usage, msg["usage"])
            elif etype == "result":
                output_text = ev.get("result", output_text) or output_text
                num_turns = ev.get("num_turns", num_turns)
                if ev.get("total_cost_usd") is not None:
                    harness_cost = ev["total_cost_usd"]
                for name, entry in (ev.get("modelUsage") or {}).items():
                    if isinstance(entry, dict) and entry.get("costUSD") is not None:
                        harness_model_cost[name] = entry["costUSD"]
                if ev.get("usage"):
                    usage = ev["usage"]  # terminal cumulative usage wins
                if ev.get("modelUsage"):
                    # The answer-producing model is authoritative (captured from
                    # the assistant message above). Claude Code may additionally
                    # invoke an auxiliary model (e.g. Haiku for background tasks),
                    # so ``modelUsage`` can carry several keys; use it only as a
                    # fallback when no assistant model was seen, and otherwise
                    # prefer the assistant model if it is among the usage keys.
                    keys = list(ev["modelUsage"].keys())
                    if reported_model is None:
                        reported_model = keys[0] if keys else reported_model
                    elif reported_model not in keys and keys:
                        # assistant model absent from usage breakdown → trust
                        # the assistant model as-is (do not overwrite).
                        pass
                if ev.get("is_error") or ev.get("subtype") not in (None, "success"):
                    result_error = ev.get("subtype") or "result_error"
                    result_is_error = bool(ev.get("is_error")) or result_is_error

        if returncode != 0 and not events:
            return DriverResult(ok=False, error_class="api_error",
                                error_detail=f"claude exit {returncode}: {stderr[:500]}")
        if result_error and result_error != "error_max_turns":
            # error_max_turns is a task outcome (agent gave up), not an API fault.
            # P3: usage-limit refusals pause the whole benchmark. Classification
            # requires the LIMIT MESSAGE TEXT — the bare is_error=true +
            # subtype "success" signature is shared by other terminal refusals
            # (observed in rerun-1: a transient Usage-Policy refusal carried the
            # same signature and must stay a retryable api_error, or one
            # stochastic safety false-positive stalls the run for an hour).
            # A textless refusal with that signature is still treated as a
            # limit (pausing is the safe default when there is nothing to
            # retry against); any texted non-limit refusal is api_error.
            text = output_text or ""
            if (USAGE_LIMIT_RE.search(text)
                    or (result_is_error and result_error == "success"
                        and not text.strip())):
                return DriverResult(
                    ok=False, error_class="usage_limit",
                    error_detail=(text or result_error)[:300],
                    raw={"events": events})
            return DriverResult(ok=False, error_class="api_error",
                                error_detail=(text[:300] or result_error),
                                raw={"events": events})
        return DriverResult(
            ok=True,
            output_text=output_text,
            transcript=transcript,
            usage=usage,
            reported_model=reported_model,
            num_turns=num_turns,
            usage_raw=usage,
            raw={"events": events},
            harness_reported_cost_usd=harness_cost,
            harness_reported_model_cost_usd=harness_model_cost,
        )

    # -- preflight ----------------------------------------------------------
    def preflight(self) -> dict:
        """Capture ``claude --version`` for the reproduction stamp. The only
        real-CLI touch permitted outside actual execution (smoke level)."""
        path = shutil.which(self.binary)
        info: dict = {"binary": self.binary, "resolved": path}
        if path is None:
            info["available"] = False
            return info
        try:
            proc = subprocess.run([self.binary, "--version"],
                                  capture_output=True, text=True, timeout=15)
            info["available"] = proc.returncode == 0
            info["version"] = (proc.stdout or proc.stderr).strip()
        except (OSError, subprocess.SubprocessError) as exc:
            info["available"] = False
            info["error"] = str(exc)
        return info


def _merge_usage(acc: dict, add: dict) -> dict:
    out = dict(acc)
    for k in ("input_tokens", "output_tokens", "cache_creation_input_tokens",
              "cache_read_input_tokens"):
        if k in add:
            out[k] = int(out.get(k, 0) or 0) + int(add.get(k, 0) or 0)
    return out


class ModelMismatch(RuntimeError):
    """Raised when a run's reported model differs from the pinned model
    (FR-H2 / §8.3). Aborts the benchmark — never silently continues."""


# --------------------------------------------------------------------------
# Real Codex CLI headless driver
# --------------------------------------------------------------------------
# Codex ``exec --json`` usage schemas.  The CLI version decides which token
# buckets exist, so the fresh axis does not mean the same thing across them.
#   v1 (<=0.144.x): usage has {input_tokens, cached_input_tokens, ...} only.
#       Upstream discards the provider's cache-write count at deserialization
#       (rust-v0.144.5 ``TokenUsage``), while OpenAI's ``input_tokens``
#       *includes* it.  Therefore ``input_tokens - cached`` is ``U + W``.
#   v2 (>=0.145.0, upstream 2edad72d / #33454): usage additionally carries
#       ``cache_write_input_tokens``, so ``U``, ``W``, ``R`` separate and a
#       four-component priced accounting becomes possible.
# ``gross = input_tokens`` under both schemas, so gross-axis numbers committed
# under v1 remain valid; only the fresh axis changes meaning.
CODEX_DRIVER_SCHEMA_V1 = "codex-exec-jsonl-0.144.x-v1"
CODEX_DRIVER_SCHEMA_V2 = "codex-exec-jsonl-0.145.x-v2"
CODEX_DRIVER_SCHEMA = CODEX_DRIVER_SCHEMA_V2
KNOWN_CODEX_DRIVER_SCHEMAS = (CODEX_DRIVER_SCHEMA_V1, CODEX_DRIVER_SCHEMA_V2)
CODEX_WRITE_TOKEN_KEY = "cache_write_input_tokens"
CODEX_FRESH_SEMANTICS = {
    CODEX_DRIVER_SCHEMA_V1: "uncached+cache_write",
    CODEX_DRIVER_SCHEMA_V2: "uncached",
}
_CODEX_KNOWN_ITEM_TYPES = {
    "agent_message", "reasoning", "command_execution", "file_change",
    "mcp_tool_call", "web_search", "plan_update", "todo_list", "error",
}


class CodexCliDriver(Driver):
    """``codex exec --json`` driver with controlled-home isolation.

    The subprocess runner is injectable and receives ``(argv, cwd, timeout,
    env)``.  Tests therefore exercise argv and parsing without network calls.
    """

    def __init__(self, binary: str = "codex", runner: Callable | None = None,
                 *, auth_source: str | Path | None = None,
                 temp_root: str | Path | None = None):
        self.binary = binary
        self._runner = runner or self._subprocess_runner
        self.auth_source = Path(auth_source) if auth_source else None
        self.temp_root = Path(temp_root) if temp_root else None

    def build_argv(self, request: DriverRequest) -> list[str]:
        argv = [
            self.binary, "exec",
            "--json", "--ephemeral", "--strict-config",
            "--model", request.model,
            "--sandbox", "read-only",
            "--dangerously-bypass-hook-trust",
            "--disable", "multi_agent",
            "-c", 'approval_policy="never"',
            "-c", f'model_reasoning_effort="{request.reasoning_effort}"',
            "-c", 'web_search="disabled"',
            "-c", "memories.use_memories=false",
            "-c", "memories.generate_memories=false",
        ]
        for value in request.config_overrides:
            argv += ["-c", value]
        argv += list(request.extra_args)
        argv += ["-C", request.cwd, request.prompt]
        return argv

    @staticmethod
    def _subprocess_runner(argv, cwd, timeout, env):
        return subprocess.run(
            argv, cwd=cwd, capture_output=True, text=True, timeout=timeout, env=env
        )

    def run(self, request: DriverRequest) -> DriverResult:
        from karc.bench.environment import ControlledCodexHome

        try:
            with ControlledCodexHome(
                auth_source=self.auth_source, temp_root=self.temp_root
            ) as home:
                env_overrides = dict(request.environment)
                env_overrides["KARC_CODEX_GUARD_MODE"] = request.hook_mode
                if env_overrides.get("KARC_CODEX_MAX_TOOL_CALLS"):
                    env_overrides["KARC_CODEX_GUARD_STATE"] = str(
                        home.path / "tool-count.txt"
                    )
                env = home.child_environment(env_overrides)
                argv = self.build_argv(request)
                try:
                    proc = self._runner(argv, request.cwd, request.timeout_s, env)
                except subprocess.TimeoutExpired:
                    return DriverResult(ok=False, error_class="api_error",
                                        error_detail="codex cli timeout")
                except (OSError, subprocess.SubprocessError) as exc:
                    return DriverResult(ok=False, error_class="harness_error",
                                        error_detail=f"codex cli spawn failed: {exc}")
                audit = home.read_audit_events()
                result = self.parse_output(proc.returncode, proc.stdout, proc.stderr,
                                           audit_events=audit)
                result.driver_metadata.update({
                    "schema": result.driver_metadata.get(
                        "usage_schema", CODEX_DRIVER_SCHEMA
                    ),
                    "controlled_home_manifest": home.manifest,
                    "hook_event_count": len(audit),
                    "guard_events": audit,
                })
                return result
        except RuntimeError as exc:
            return DriverResult(ok=False, error_class="harness_error",
                                error_detail=str(exc))

    @staticmethod
    def _mcp_name(item: dict) -> tuple[str, str | None, str | None]:
        server = item.get("server") or item.get("server_name")
        tool = item.get("tool") or item.get("tool_name")
        name = item.get("name")
        if (not server or not tool) and isinstance(name, str) and name.startswith("mcp__"):
            parts = name.split("__", 2)
            if len(parts) == 3:
                server, tool = parts[1], parts[2]
        logical = f"mcp__{server}__{tool}" if server and tool else (name or "mcp")
        return logical, server, tool

    @staticmethod
    def _command_text(item: dict) -> str:
        value = item.get("command")
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return " ".join(str(x) for x in value)
        return ""

    def parse_output(self, returncode: int, stdout: str, stderr: str,
                     *, audit_events: list[dict] | None = None) -> DriverResult:
        events: list[dict] = []
        malformed = 0
        for line in (stdout or "").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if isinstance(obj, dict):
                events.append(obj)
            else:
                malformed += 1

        transcript: list[dict] = []
        item_state: dict[str, dict] = {}
        output_text = ""
        usage_raw: dict = {}
        thread_id: str | None = None
        failed_detail: str | None = None
        unknown: list[str] = []
        completed_items = 0
        nonfatal_error_items = 0

        for ev in events:
            etype = ev.get("type")
            if etype == "thread.started":
                thread_id = ev.get("thread_id") or thread_id
            elif etype == "turn.started":
                continue
            elif etype in ("item.started", "item.updated", "item.completed"):
                item = ev.get("item") or {}
                if not isinstance(item, dict):
                    unknown.append(f"{etype}:non-object")
                    continue
                iid = str(item.get("id") or f"anon-{len(item_state)}")
                merged = {**item_state.get(iid, {}), **item}
                item_state[iid] = merged
                if etype != "item.completed":
                    continue
                completed_items += 1
                itype = merged.get("type")
                if itype not in _CODEX_KNOWN_ITEM_TYPES:
                    unknown.append(f"item:{itype}")
                    continue
                if itype == "agent_message":
                    text = merged.get("text") or ""
                    if isinstance(text, str):
                        output_text = text
                        transcript.append({"kind": "text", "text": text})
                elif itype == "error":
                    # ErrorItem is a documented non-fatal exec item.  Reduce
                    # it because diagnostics may contain paths or text.
                    message = merged.get("message")
                    message = message if isinstance(message, str) else ""
                    nonfatal_error_items += 1
                    transcript.append({
                        "kind": "observation", "name": "codex_error_item",
                        "message_length": len(message),
                        "message_sha256": hashlib.sha256(
                            message.encode("utf-8", errors="replace")
                        ).hexdigest(),
                        "item_id": iid,
                    })
                elif itype == "command_execution":
                    command = self._command_text(merged)
                    transcript.append({
                        "kind": "tool_use", "name": "Bash",
                        "input": {"command": command}, "item_id": iid,
                        "status": merged.get("status"),
                    })
                elif itype == "file_change":
                    changes = merged.get("changes") or merged.get("files") or []
                    transcript.append({
                        "kind": "tool_use", "name": "apply_patch",
                        "input": {"changes": changes}, "item_id": iid,
                        "status": merged.get("status"),
                    })
                elif itype == "mcp_tool_call":
                    name, server, tool = self._mcp_name(merged)
                    args = merged.get("arguments") or merged.get("input") or {}
                    transcript.append({
                        "kind": "tool_use", "name": name,
                        "logical_server": server, "logical_tool": tool,
                        "input": args if isinstance(args, dict) else {},
                        "item_id": iid, "status": merged.get("status"),
                    })
            elif etype == "turn.completed":
                usage_raw = ev.get("usage") or usage_raw
            elif etype == "turn.failed":
                err = ev.get("error")
                failed_detail = json.dumps(err, ensure_ascii=False) if err else "turn.failed"
            elif etype == "error":
                failed_detail = str(ev.get("message") or ev.get("error") or "error")
            else:
                unknown.append(str(etype))

        audit_events = audit_events or []
        for event in audit_events:
            if event.get("denied"):
                transcript.append({
                    "kind": "guard_denied",
                    "name": event.get("tool_name"),
                    "tool_use_id": event.get("tool_use_id"),
                })
        models = sorted({str(e.get("model")) for e in audit_events if e.get("model")})
        reported_model = models[0] if len(models) == 1 else None
        model_verification = "hook" if reported_model else "requested-only"
        if len(models) > 1:
            return DriverResult(
                ok=False, error_class="harness_error",
                error_detail=f"multiple hook models observed: {models}",
                native_session_id=thread_id, driver_metadata={"models": models},
            )

        if malformed or unknown:
            return DriverResult(
                ok=False, error_class="harness_error",
                error_detail=(f"Codex JSONL schema drift: malformed={malformed}, "
                              f"unknown={sorted(set(unknown))[:20]}"),
                native_session_id=thread_id, reported_model=reported_model,
                model_verification=model_verification,
                raw={"event_types": [e.get("type") for e in events],
                     "nonfatal_error_items": nonfatal_error_items},
            )

        error_text = failed_detail or stderr or ""
        if failed_detail or returncode != 0:
            cls = "usage_limit" if USAGE_LIMIT_RE.search(error_text) else "api_error"
            return DriverResult(
                ok=False, error_class=cls, error_detail=error_text[:500],
                reported_model=reported_model, native_session_id=thread_id,
                model_verification=model_verification,
                raw={"event_types": [e.get("type") for e in events],
                     "nonfatal_error_items": nonfatal_error_items},
            )
        if not events or not output_text or not usage_raw:
            return DriverResult(
                ok=False, error_class="harness_error",
                error_detail="Codex JSONL missing required final message or usage",
                reported_model=reported_model, native_session_id=thread_id,
                model_verification=model_verification,
                raw={"event_types": [e.get("type") for e in events],
                     "nonfatal_error_items": nonfatal_error_items},
            )

        total_in = int(usage_raw.get("input_tokens", 0) or 0)
        cached = int(usage_raw.get("cached_input_tokens", 0) or 0)
        # Version-aware cache-write bucket.  Absent (v1) is NOT the same claim
        # as observed-zero (v2): under v1 the write tokens are still inside
        # ``input_tokens``, merely unlabelled.
        has_write = usage_raw.get(CODEX_WRITE_TOKEN_KEY) is not None
        write = int(usage_raw.get(CODEX_WRITE_TOKEN_KEY, 0) or 0)
        usage_schema = CODEX_DRIVER_SCHEMA_V2 if has_write else CODEX_DRIVER_SCHEMA_V1
        if cached + write > total_in:
            return DriverResult(
                ok=False, error_class="harness_error",
                error_detail=(
                    "Codex usage buckets do not close: "
                    f"cached={cached} + write={write} > input_tokens={total_in}"
                ),
                reported_model=reported_model, native_session_id=thread_id,
                model_verification=model_verification,
                driver_metadata={"usage_schema": usage_schema},
                raw={"event_types": [e.get("type") for e in events],
                     "nonfatal_error_items": nonfatal_error_items},
            )
        normalized = {
            "input_tokens": max(0, total_in - cached - write),
            "cache_read_input_tokens": cached,
            "output_tokens": int(usage_raw.get("output_tokens", 0) or 0),
            "reasoning_output_tokens": int(
                usage_raw.get("reasoning_output_tokens", 0) or 0
            ),
        }
        if has_write:
            # Mirror the Anthropic bucket name so downstream priced accounting
            # is symmetric across providers (see e5_cross_runtime).
            normalized["cache_creation_input_tokens"] = write
        return DriverResult(
            ok=True, output_text=output_text, transcript=transcript,
            usage=normalized, usage_raw=usage_raw,
            reported_model=reported_model, num_turns=completed_items,
            native_session_id=thread_id, model_verification=model_verification,
            driver_metadata={
                "usage_schema": usage_schema,
                "fresh_semantics": CODEX_FRESH_SEMANTICS[usage_schema],
            },
            raw={"event_types": [e.get("type") for e in events],
                 "nonfatal_error_items": nonfatal_error_items},
        )

    def preflight(self) -> dict:
        path = shutil.which(self.binary)
        info: dict = {"binary": self.binary, "resolved": path,
                      "driver_schema": CODEX_DRIVER_SCHEMA}
        if path is None:
            info["available"] = False
            return info
        commands = {
            "version": [self.binary, "--version"],
            "exec_help": [self.binary, "exec", "--help"],
            "features": [self.binary, "features", "list"],
        }
        captured: dict[str, str] = {}
        try:
            for name, argv in commands.items():
                proc = subprocess.run(argv, capture_output=True, text=True, timeout=15)
                if proc.returncode != 0:
                    raise subprocess.SubprocessError(f"{name} exit {proc.returncode}")
                captured[name] = (proc.stdout or proc.stderr).strip()
        except (OSError, subprocess.SubprocessError) as exc:
            info.update(available=False, error=str(exc))
            return info
        help_text = captured["exec_help"]
        feature_text = captured["features"]
        required_flags = (
            "--json", "--model", "--sandbox", "--ephemeral",
            "--ignore-user-config", "--strict-config",
        )
        info.update({
            "available": True,
            "version": captured["version"],
            "required_flags": {f: f in help_text for f in required_flags},
            "hooks_feature_stable": bool(
                re.search(r"(?m)^hooks\s+stable\s+true$", feature_text)
            ),
        })
        info["contract_ok"] = (
            all(info["required_flags"].values()) and info["hooks_feature_stable"]
        )
        return info


class CodexPersistentSession:
    """A controlled Codex ``exec`` conversation spanning several turns.

    Unlike :class:`CodexCliDriver`, the initial invocation deliberately omits
    ``--ephemeral`` and later turns use ``codex exec resume`` inside the same
    temporary ``CODEX_HOME``.  The home (including Codex's on-disk rollout)
    exists only for the context-manager lifetime and is deleted on exit.  This
    is the minimum primitive needed to measure provider prompt-cache behavior
    across a session without persisting prompts or model output.
    """

    def __init__(
        self, request: DriverRequest, *, binary: str = "codex",
        runner: Callable | None = None, auth_source: str | Path | None = None,
        temp_root: str | Path | None = None,
    ):
        self.request = request
        self.driver = CodexCliDriver(
            binary=binary, runner=runner, auth_source=auth_source,
            temp_root=temp_root,
        )
        self._home = None
        self._environment: dict[str, str] | None = None
        self._thread_id: str | None = None
        self._audit_count = 0
        self._turn = 0

    @property
    def thread_id(self) -> str | None:
        return self._thread_id

    def _common_argv(self) -> list[str]:
        req = self.request
        argv = [
            "--json", "--strict-config",
            "--model", req.model,
            "--dangerously-bypass-hook-trust",
            "--disable", "multi_agent",
            "-c", 'approval_policy="never"',
            "-c", 'sandbox_mode="read-only"',
            "-c", f'model_reasoning_effort="{req.reasoning_effort}"',
            "-c", 'web_search="disabled"',
            "-c", "memories.use_memories=false",
            "-c", "memories.generate_memories=false",
        ]
        for value in req.config_overrides:
            argv += ["-c", value]
        argv += list(req.extra_args)
        return argv

    def build_initial_argv(self, prompt: str) -> list[str]:
        return [
            self.driver.binary, "exec", *self._common_argv(),
            "--sandbox", "read-only", "-C", self.request.cwd, prompt,
        ]

    def build_resume_argv(self, prompt: str) -> list[str]:
        if self._thread_id is None:
            raise RuntimeError("cannot resume before the initial Codex turn")
        return [
            self.driver.binary, "exec", "resume", *self._common_argv(),
            self._thread_id, prompt,
        ]

    def __enter__(self) -> "CodexPersistentSession":
        from karc.bench.environment import ControlledCodexHome

        self._home = ControlledCodexHome(
            auth_source=self.driver.auth_source,
            temp_root=self.driver.temp_root,
        )
        self._home.__enter__()
        overrides = dict(self.request.environment)
        overrides["KARC_CODEX_GUARD_MODE"] = self.request.hook_mode
        if overrides.get("KARC_CODEX_MAX_TOOL_CALLS"):
            overrides["KARC_CODEX_GUARD_STATE"] = str(
                self._home.path / "tool-count.txt"
            )
        self._environment = self._home.child_environment(overrides)
        return self

    def run_turn(self, prompt: str) -> DriverResult:
        if self._home is None or self._environment is None:
            raise RuntimeError("persistent Codex session is not active")
        argv = (self.build_initial_argv(prompt) if self._thread_id is None
                else self.build_resume_argv(prompt))
        try:
            proc = self.driver._runner(
                argv, self.request.cwd, self.request.timeout_s,
                self._environment,
            )
        except subprocess.TimeoutExpired:
            return DriverResult(ok=False, error_class="api_error",
                                error_detail="codex cli timeout")
        except (OSError, subprocess.SubprocessError) as exc:
            return DriverResult(ok=False, error_class="harness_error",
                                error_detail=f"codex cli spawn failed: {exc}")

        audit = self._home.read_audit_events()
        new_audit = audit[self._audit_count:]
        self._audit_count = len(audit)
        result = self.driver.parse_output(
            proc.returncode, proc.stdout, proc.stderr,
            audit_events=new_audit,
        )
        observed = result.native_session_id
        if self._thread_id is None and result.error_class is None:
            if observed is None:
                return DriverResult(
                    ok=False, error_class="harness_error",
                    error_detail="initial Codex turn did not report a thread id",
                    reported_model=result.reported_model,
                    model_verification=result.model_verification,
                )
            self._thread_id = observed
        elif (self._thread_id is not None and observed is not None
              and observed != self._thread_id):
            return DriverResult(
                ok=False, error_class="harness_error",
                error_detail="Codex resume returned a different thread id",
                reported_model=result.reported_model,
                model_verification=result.model_verification,
            )
        self._turn += 1
        result.driver_metadata.update({
            "schema": result.driver_metadata.get(
                "usage_schema", CODEX_DRIVER_SCHEMA
            ),
            "controlled_home_manifest": self._home.manifest,
            "hook_event_count": len(new_audit),
            "guard_events": new_audit,
            "persistent_session": True,
            "session_turn": self._turn,
        })
        return result

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._home is not None:
            self._home.__exit__(exc_type, exc, tb)
        self._home = None
        self._environment = None
