"""E20-CODEX-METERED: does the *same* Codex CLI report cache-write tokens when
the billing route is a metered API key instead of a ChatGPT subscription?

Why this probe exists
---------------------
E15 established that Codex CLI 0.145.0 on the **subscription** route reports
``cache_write_input_tokens = 0`` on 4/4 turns while ``cached_input_tokens``
grew from 7,936 to 54,272 inside one session.  E16 established that the
**metered** route reports write (cold 8,379 / warm 19).  E16 section 1.1 then
concluded "the only difference is the route" -- but strictly it was not: the
subscription measurement went through the Codex CLI and the metered
measurement went through a hand-written HTTP client.  A rival reading survives:
*the CLI implementation drops the field*.

This probe holds the client fixed (same binary, same argv builder, same parser,
same model, same multi-turn design) and changes only the billing route, so the
route attribution stops depending on the client.

Design, inherited from ``E15-CODEX-WRITE/scripts/multiturn_probe.py``
--------------------------------------------------------------------
One persistent ``codex exec`` session of several sequential turns, each turn
appending several hundred tokens of new coherent prose so that the accumulated
context provably grows.  Then:

  * any turn with ``cache_write_input_tokens > 0``  -> verdict **A**: with the
    client held fixed, the route alone produces write reporting;
  * all writes 0 while ``cached_input_tokens`` grows between turns of the same
    session -> verdict **B**: the route does not explain it alone, so E16
    section 1.1 must be weakened;
  * all writes 0 and ``cached`` never grows -> verdict **C**: inconclusive, and
    ``input_tokens`` is checked (not assumed) to see whether the session
    accumulated at all.

Per E16 section 1.2, ``write = 0`` on a fully warm call is legitimate, so the
zero alone is never read as evidence of omission -- only the *combination* of
zero write with growing ``cached`` is.

The turn text is coherent technical prose, not random vocabulary, because
E16 section 4 showed random-vocabulary fixtures draw refusals whose accounting
is distorted (creation billed, read never registered).

Proving the route
----------------
1. An isolated temporary ``CODEX_HOME`` is created with **no** ``auth.json``
   bridge -- the user's ``~/.codex`` is never linked, read or written.
2. Before any credential exists, ``codex login status`` and one real
   ``codex exec`` attempt are run.  The exec attempt must fail (401), which is
   the control showing the turns that follow are paid for by the injected key
   and not by a subscription entitlement.
3. ``codex login --with-api-key`` receives the key on **stdin** (never argv,
   never the child's environment), then ``codex login status`` records the
   auth type the CLI itself reports.
4. mtimes of the user's ``~/.codex`` paths are snapshotted before and after.

Privacy (R-9): only token counts, ids, hashes, lengths, key names, mtimes and
schema labels reach ``raw/``.  Prompt text, model output and transcripts are
never persisted; each turn's prose is reproducible from ``(round, turn)`` via
the constants in this file, and only its sha256 and length are recorded.  The
API key is read from ``os.environ`` and never printed, logged or stored; the
``codex login status`` line is classified, not stored, because the CLI prints a
partially masked key in it.

Pre-declared caps: 8 model-bearing turns total across all rounds, and an abort
if cumulative ``input_tokens`` would exceed ``MAX_CUMULATIVE_INPUT_TOKENS``.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
CELL = Path(__file__).resolve().parents[1]
BINARY = (
    "<HOME>/.local/opt/codex-0.145.0/node_modules/@openai/"
    "codex-darwin-arm64/vendor/aarch64-apple-darwin/bin/codex"
)
MODEL = "gpt-5.6-luna"
REASONING_EFFORT = "low"
TOTAL_TURN_CAP = 8
# gpt-5.6-luna public rates, USD per million tokens, from
# docs/analysis/openai-luna-rate-card.md as recorded in E18 config.yaml.
RATES = {"uncached": 0.20, "write": 0.25, "read": 0.02, "output": 1.20}
# 600k gross input is about $0.12 at the read/uncached rates above, far under
# the $1 ceiling; the guard exists to stop a runaway resume loop, not to bind.
MAX_CUMULATIVE_INPUT_TOKENS = 600_000

# Round -> (turns, prose sentences per turn).  ~20 tokens per generated
# sentence, so a 40-sentence block appends roughly 800 tokens of new context.
ROUNDS = {1: (4, 40), 2: (4, 40)}

USER_CODEX_PATHS = (
    "",
    "auth.json",
    "config.toml",
    "sessions",
    "packages/standalone/current",
)

sys.path.insert(0, str(REPO / "src"))

from karc.bench.driver import (  # noqa: E402
    CODEX_WRITE_TOKEN_KEY,
    CodexPersistentSession,
    DriverRequest,
)
from karc.bench.environment import ControlledCodexHome, _sha256  # noqa: E402

USAGE_KEYS_ALLOWED = {
    "input_tokens", "cached_input_tokens", CODEX_WRITE_TOKEN_KEY,
    "output_tokens", "reasoning_output_tokens", "total_tokens",
}

# Coherent prose about cache accounting.  Placeholders are filled with
# deterministic pseudo-measurements so the block reads as an engineering log
# and stays distinct per turn while remaining reproducible from (round, turn).
_SENTENCES = (
    "Segment {n} of the reader recorded {a} cached input tokens against {b} "
    "fresh input tokens, so the cached share of that segment was the larger of "
    "the two.",
    "The accountant treats the {a} tokens read from the cache as a separate "
    "bucket from the {b} tokens that were sent uncached, because the two are "
    "billed at different rates.",
    "When the prefix was replayed for the {n}th time the reader saw {a} tokens "
    "arrive from the cache and only {b} tokens charged at the ordinary input "
    "rate.",
    "A turn that appends {b} new tokens to a prefix of {a} tokens should, if "
    "the provider writes what it reads, report a write of about {b} rather "
    "than nothing at all.",
    "The ledger for segment {n} closes only if the reported input total of {c} "
    "equals the cached {a} plus the written {b} plus the remainder.",
    "Our earlier reading of segment {n} assumed the write bucket was empty, "
    "and the assumption survived precisely because {a} and {b} were never "
    "reported separately.",
    "Between the {n}th and the following segment the cached count moved from "
    "{a} to {c}, which means somebody put {b} tokens into the cache in the "
    "interval.",
    "For the purpose of this log the phrase fresh input means the {b} tokens "
    "that were neither read from the cache nor labelled as a write when "
    "segment {n} was measured.",
    "The rate card multiplies the {a} read tokens by a tenth and the {b} "
    "written tokens by five fourths, so the two buckets cannot be merged "
    "without changing the total.",
    "If the transport reports only {a} and omits {b}, the fresh figure we "
    "publish for segment {n} is an upper bound on the truly uncached amount "
    "and not the amount itself.",
    "Segment {n} is retained verbatim in this log because the later "
    "reconciliation needs the pair {a} and {b} and not their sum {c}.",
    "The reconciliation step for segment {n} compares the provider's total of "
    "{c} with our own sum and stops if the difference exceeds a single token.",
)


def build_prompt(round_no: int, turn: int, sentences: int) -> str:
    """Deterministic coherent-prose block plus a one-word ack instruction."""
    rng = random.Random(f"E20-CODEX-METERED/{round_no}/{turn}")
    body = []
    for i in range(sentences):
        template = _SENTENCES[(turn * 7 + i) % len(_SENTENCES)]
        a = rng.randrange(1_000, 60_000)
        b = rng.randrange(100, 9_000)
        line = template.format(n=f"{turn}.{i + 1}", a=f"{a:,}", b=f"{b:,}",
                               c=f"{a + b:,}")
        body.append(f"{turn}.{i + 1}. {line}")
    return (
        f"Section {turn} of an engineering log about prompt-cache accounting. "
        "Read it and keep it in context for the later sections. Do not "
        "summarize it, do not use any tool.\n"
        + "\n".join(body)
        + "\nAcknowledge this section with exactly: ok"
    )


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class MeteredCodexHome(ControlledCodexHome):
    """An isolated ``CODEX_HOME`` with no subscription auth bridge.

    ``ControlledCodexHome`` symlinks the user's ``auth.json`` and refuses to
    start without it.  Here the home deliberately starts with **no** credential
    so that (a) the no-credential control is real and (b) the only credential
    that ever exists inside the home is the metered API key the CLI itself
    writes via ``codex login --with-api-key``.  The hooks payload is kept
    byte-identical in shape to the parent class so model verification and the
    audit log behave exactly as in E15.
    """

    def __enter__(self) -> "MeteredCodexHome":
        root = str(self.temp_root) if self.temp_root else None
        self.path = Path(tempfile.mkdtemp(prefix="karc-codex-home-", dir=root))
        self.path.chmod(0o700)
        self.audit_path = self.path / "guard-events.jsonl"
        self.hooks_path = self.path / "hooks.json"
        command = shlex.join(
            [str(self.python_executable), "-m", "karc.bench.codex_guard"]
        )
        handler = {"type": "command", "command": command, "timeout": 10}
        hooks = {
            "description": "K-ARC reproducible benchmark guard (generated)",
            "hooks": {
                "SessionStart": [{"matcher": "startup", "hooks": [handler]}],
                "PreToolUse": [{"matcher": "*", "hooks": [handler]}],
                "PermissionRequest": [{"matcher": "*", "hooks": [handler]}],
                "PostToolUse": [{"matcher": "*", "hooks": [handler]}],
                "Stop": [{"hooks": [handler]}],
            },
        }
        self.hooks_path.write_text(
            json.dumps(hooks, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self.manifest = {
            "schema_version": 1,
            "auth_bridge": "none-metered-api-key-written-by-cli-login",
            "user_auth_json_linked": False,
            "generated_files": [
                {"name": "hooks.json", "sha256": _sha256(self.hooks_path)},
            ],
            "excluded": ["auth.json", "guard-events.jsonl", "tmp", "log"],
        }
        return self

    # -- route provenance helpers ------------------------------------------

    def _cli(self, argv: list[str], stdin_text: str | None = None):
        env = {k: os.environ[k] for k in ("PATH", "LANG", "TMPDIR", "TERM")
               if k in os.environ}
        env["CODEX_HOME"] = str(self.path)
        return subprocess.run(
            [BINARY, *argv], capture_output=True, text=True, timeout=120,
            env=env, input=stdin_text if stdin_text is not None else "",
        )

    def login_status(self) -> dict:
        """Classify the CLI's own auth report without storing the line.

        ``codex login status`` prints a partially masked API key, so only the
        classification, the line length and a digest are recorded.
        """
        proc = self._cli(["login", "status"])
        line = (proc.stdout or proc.stderr).strip()
        lowered = line.lower()
        if "api key" in lowered:
            kind = "api-key"
        elif "not logged in" in lowered:
            kind = "not-logged-in"
        elif "chatgpt" in lowered or "subscription" in lowered:
            kind = "chatgpt-subscription"
        else:
            kind = "unrecognized"
        return {
            "returncode": proc.returncode,
            "auth_type": kind,
            "line_length": len(line),
            "line_sha256": _sha256_text(line),
        }

    def login_with_api_key(self, api_key: str) -> dict:
        proc = self._cli(["login", "--with-api-key"], stdin_text=api_key)
        text = (proc.stdout or "") + (proc.stderr or "")
        return {
            "returncode": proc.returncode,
            "succeeded": proc.returncode == 0 and "success" in text.lower(),
            "auth_json_present_in_isolated_home": (
                self.path / "auth.json").is_file(),
        }


class MeteredCodexSession(CodexPersistentSession):
    """A persistent session bound to an externally managed metered home.

    The home must be entered (and the credential established) by the caller,
    because the no-credential control has to run inside the same home *before*
    the key is written.
    """

    def __init__(self, request: DriverRequest, *, home: MeteredCodexHome,
                 binary: str = BINARY):
        self.last_event_summary: dict = {}
        super().__init__(request, binary=binary, runner=self._runner_devnull)
        self._external_home = home

    def _runner_devnull(self, argv, cwd, timeout, env):
        # ``codex exec`` reads extra input from an open stdin; detach it so the
        # probe can never block on a terminal.  The stdout is also summarized
        # here because the driver's parser keeps only event *types*, and this
        # probe has to report whatever stop/finish reason the CLI emits.
        proc = subprocess.run(
            argv, cwd=cwd, capture_output=True, text=True, timeout=timeout,
            env=env, stdin=subprocess.DEVNULL,
        )
        self.last_event_summary = _event_summary(proc.stdout or "")
        return proc

    def __enter__(self) -> "MeteredCodexSession":
        self._home = self._external_home
        overrides = dict(self.request.environment)
        overrides["KARC_CODEX_GUARD_MODE"] = self.request.hook_mode
        self._environment = self._home.child_environment(overrides)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # The home outlives the session; the caller closes it.
        self._home = None
        self._environment = None


def _event_summary(stdout: str) -> dict:
    """Event-level summary used to hunt for a stop/finish reason.

    Codex ``exec --json`` has no documented stop-reason field, so instead of
    asserting one, every non-usage scalar on ``turn.completed`` is captured
    along with the event-type histogram.  Message bodies are reduced to
    digests (R-9).
    """
    types: dict[str, int] = {}
    completed_extra: dict = {}
    error_digests: list[str] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(ev, dict):
            continue
        etype = str(ev.get("type"))
        types[etype] = types.get(etype, 0) + 1
        if etype == "turn.completed":
            for key, value in ev.items():
                if key in ("type", "usage"):
                    continue
                if isinstance(value, (str, int, float, bool)) or value is None:
                    completed_extra[key] = value
                else:
                    completed_extra[key] = f"<{type(value).__name__}>"
        elif etype in ("error", "turn.failed"):
            payload = json.dumps(ev.get("error") or ev.get("message") or "",
                                 ensure_ascii=False)
            error_digests.append(_sha256_text(payload)[:16])
    return {
        "event_type_counts": types,
        "turn_completed_non_usage_fields": completed_extra,
        "error_event_digests": error_digests,
        "stop_reason_field_present": any(
            k in completed_extra for k in
            ("stop_reason", "finish_reason", "status", "reason")
        ),
    }


def _mtime_snapshot() -> dict:
    base = Path.home() / ".codex"
    out = {}
    for rel in USER_CODEX_PATHS:
        p = base / rel if rel else base
        try:
            st = p.lstat()
        except OSError:
            out[rel or "."] = None
            continue
        out[rel or "."] = {
            "mtime_ns": st.st_mtime_ns,
            "mtime_utc": datetime.fromtimestamp(
                st.st_mtime, timezone.utc).isoformat(timespec="seconds"),
        }
    return out


def _priced_usd(uncached: int, write: int, read: int, output: int) -> float:
    return round(
        uncached / 1e6 * RATES["uncached"]
        + write / 1e6 * RATES["write"]
        + read / 1e6 * RATES["read"]
        + output / 1e6 * RATES["output"],
        6,
    )


def main() -> int:
    round_no = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    if round_no not in ROUNDS:
        raise SystemExit(f"round must be one of {sorted(ROUNDS)}")
    turns, sentences = ROUNDS[round_no]
    spent_before = sum(ROUNDS[r][0] for r in sorted(ROUNDS) if r < round_no)
    if spent_before + turns > TOTAL_TURN_CAP:
        raise SystemExit("would exceed the pre-declared 8-turn cap")

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key.strip():
        raise SystemExit("OPENAI_API_KEY is not present in the environment")

    git_hash = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(REPO),
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    cli_version = subprocess.run(
        [BINARY, "--version"], capture_output=True, text=True, check=True,
    ).stdout.strip()

    user_codex_before = _mtime_snapshot()
    records: list[dict] = []
    control: dict = {}
    thread_id = None
    aborted_for_budget = False

    with tempfile.TemporaryDirectory(prefix="karc-e20-cwd-") as cwd:
        # Codex refuses to run outside a trusted (git) directory and the
        # persistent-session argv deliberately carries no repo-check bypass.
        subprocess.run(["git", "init", "--quiet"], cwd=cwd, check=True,
                       capture_output=True, text=True)
        request = DriverRequest(
            prompt="", cwd=cwd, model=MODEL, runtime="codex",
            reasoning_effort=REASONING_EFFORT, hook_mode="observe",
            timeout_s=600,
        )
        with MeteredCodexHome() as home:
            control["status_before_any_credential"] = home.login_status()
            with MeteredCodexSession(request, home=home) as session:
                # (1) control: identical argv, identical home, no credential.
                probe = session.run_turn(
                    "Reply with exactly: ok"
                )
                control["exec_without_credential"] = {
                    "ok": probe.ok,
                    "error_class": probe.error_class,
                    "unauthorized_in_error": "401" in (probe.error_detail or ""),
                    "usage_raw_empty": not (probe.usage_raw or {}),
                    "thread_id_assigned": session.thread_id is not None,
                    "event_summary": dict(session.last_event_summary),
                }
                # (2) inject the metered credential via stdin only.
                control["login_with_api_key"] = home.login_with_api_key(api_key)
                control["status_after_credential"] = home.login_status()
                if control["status_after_credential"]["auth_type"] != "api-key":
                    raise SystemExit(
                        "refusing to spend: CLI does not report api-key auth"
                    )
                if control["exec_without_credential"]["ok"]:
                    raise SystemExit(
                        "refusing to spend: the no-credential control succeeded, "
                        "so the isolation is not what this probe claims"
                    )

                cumulative_input = 0
                for turn in range(1, turns + 1):
                    prompt = build_prompt(round_no, turn, sentences)
                    result = session.run_turn(prompt)
                    usage_raw = dict(result.usage_raw or {})
                    meta = result.driver_metadata or {}
                    records.append({
                        "probe_turn": turn,
                        "observed_at_utc": datetime.now(timezone.utc).isoformat(
                            timespec="seconds"),
                        "prompt_chars": len(prompt),
                        "prompt_sha256": _sha256_text(prompt),
                        "ok": result.ok,
                        "error_class": result.error_class,
                        "error_detail": (result.error_detail or "")[:200] or None,
                        "reported_model": result.reported_model,
                        "model_verification": result.model_verification,
                        "usage_raw": usage_raw,
                        "usage_raw_unexpected_keys": sorted(
                            set(usage_raw) - USAGE_KEYS_ALLOWED),
                        "usage_normalized": dict(result.usage or {}),
                        "write_key_present": CODEX_WRITE_TOKEN_KEY in usage_raw,
                        "usage_schema": meta.get("usage_schema"),
                        "fresh_semantics": meta.get("fresh_semantics"),
                        "gross_input_tokens": result.total_input_tokens(
                            with_cache=True),
                        "fresh_input_tokens": result.total_input_tokens(
                            with_cache=False),
                        # session_turn counts the no-credential control as
                        # turn 1, so it is probe_turn + 1 throughout.
                        "session_turn": meta.get("session_turn"),
                        "persistent_session": meta.get("persistent_session"),
                        "hook_event_count": meta.get("hook_event_count"),
                        "event_types": sorted(set(
                            (result.raw or {}).get("event_types") or [])),
                        "event_summary": dict(session.last_event_summary),
                    })
                    print(json.dumps(records[-1]["usage_raw"],
                                     ensure_ascii=False), flush=True)
                    if not result.ok:
                        break
                    cumulative_input += int(
                        usage_raw.get("input_tokens", 0) or 0)
                    if cumulative_input > MAX_CUMULATIVE_INPUT_TOKENS:
                        aborted_for_budget = True
                        break
                thread_id = session.thread_id
            user_codex_after = _mtime_snapshot()

    ok_turns = [r for r in records if r["ok"]]
    cached = [int(r["usage_raw"].get("cached_input_tokens", 0) or 0)
              for r in ok_turns]
    writes = [int(r["usage_raw"].get(CODEX_WRITE_TOKEN_KEY, 0) or 0)
              for r in ok_turns]
    inputs = [int(r["usage_raw"].get("input_tokens", 0) or 0) for r in ok_turns]
    outputs = [int(r["usage_raw"].get("output_tokens", 0) or 0)
               for r in ok_turns]
    any_write = any(w > 0 for w in writes)
    cached_grew = any(b > a for a, b in zip(cached, cached[1:]))
    input_grew = any(b > a for a, b in zip(inputs, inputs[1:]))
    if any_write:
        verdict, outcome = "A", "metered-route-reports-writes"
    elif cached_grew:
        verdict, outcome = "B", "client-also-implicated-route-alone-insufficient"
    else:
        verdict, outcome = "C", "inconclusive"

    uncached_total = sum(
        max(0, i - c - w) for i, c, w in zip(inputs, cached, writes))
    priced = _priced_usd(uncached_total, sum(writes), sum(cached), sum(outputs))

    record = {
        "cell": "E20-CODEX-METERED",
        "probe": "metered-multiturn-cache-write",
        "round": round_no,
        "code_git_hash": git_hash,
        "binary_path": BINARY,
        "cli_version": cli_version,
        "model_requested": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "route": "metered OpenAI API key (isolated CODEX_HOME, no auth bridge)",
        "sandbox": "read-only",
        "planned_turns": turns,
        "prose_sentences_per_turn": sentences,
        "model_turns_spent_this_round": len(records),
        "model_turns_spent_cumulative": spent_before + len(records),
        "attempt_cap_model_turns": TOTAL_TURN_CAP,
        "thread_id": thread_id,
        "route_provenance": control,
        "user_codex_mtimes_before": user_codex_before,
        "user_codex_mtimes_after": user_codex_after,
        "user_codex_untouched": user_codex_before == user_codex_after,
        "isolated_home_manifest": {
            "auth_bridge": "none-metered-api-key-written-by-cli-login",
            "user_auth_json_linked": False,
        },
        "turns": records,
        "evaluation": {
            "cached_series": cached,
            "write_series": writes,
            "input_series": inputs,
            "output_series": outputs,
            "any_write_gt_zero": any_write,
            "cached_grew_between_turns": cached_grew,
            "input_grew_between_turns": input_grew,
            "verdict": verdict,
            "outcome": outcome,
            "aborted_for_budget": aborted_for_budget,
            "gross_input_tokens_total": sum(inputs),
            "uncached_input_tokens_total": uncached_total,
            "priced_usd_luna_public_card": priced,
        },
        "privacy": "token-counts-hashes-and-mtimes-only",
    }
    out = CELL / "raw" / f"metered-multiturn-round-{round_no}.json"
    out.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(json.dumps(record["evaluation"], ensure_ascii=False, indent=2))
    return 0 if all(r["ok"] for r in records) and records else 1


if __name__ == "__main__":
    raise SystemExit(main())
