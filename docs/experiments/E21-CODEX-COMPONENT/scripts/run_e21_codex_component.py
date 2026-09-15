"""E21-CODEX-COMPONENT — do the component split and the component-priced
ordering survive a THIRD harness?

E20 established that Codex CLI 0.145.0 on the metered route reports all four
buckets (``input_tokens``, ``cached_input_tokens``, ``cache_write_input_tokens``,
``output_tokens``), so a harness we previously believed could not carry
component accounting now can.  This cell runs the E19 measurement window
through that harness with the provider, the model and the rate card held fixed,
so the only thing that changes against E19 leg B is the harness.

    [warm-up]  karc arm only, S01..S08 x 8 turns    (never judged)
        v
    [verify]   resident injection tokens per turn, last vs previous session
        v
    [measure]  S09..S20 x 8 turns x 2 arms = 192 turns

Design fidelity to E19 (preregistration §2 "E19 leg B와 직접 비교 가능하도록"):

  * the SAME schedule build — seed 4352, 24 sessions of 8 turns — gated on the
    same three sha256 constants E19 committed, so the measurement window sees a
    bit-identical task sequence;
  * the SAME window — warm-up 8 sessions, measurement S09..S20;
  * the SAME arms, the SAME provider/model (``gpt-5.6-luna``), the SAME
    published rate card, the SAME statistics (session-unit paired percentile
    bootstrap, seed 1313, 10,000 resamples, common random numbers).

What is deliberately DIFFERENT is the harness, and with it two things E19 had to
adapt away and this cell restores to their native form:

  * ``AGENTS.md`` is auto-loaded by Codex as a project doc, so there is no
    ``--append-system-prompt`` substitute and the file stays in the workdir
    (E19 adaptation 1 is not needed here);
  * the MCP server is registered through Codex's own ``mcp_servers.*`` config
    overrides instead of the pi bridge's ``.mcp.json`` (E19 adaptation 2).

Both are produced by the committed, unmodified
``karc.bench.e5_cache_canary.materialize_session``, i.e. the Codex-native
materialization the E5-G1 canary used.

Instrumentation contract — measured, never assumed:

  * The INCLUSION RELATION is measured, not inherited.  Anthropic-through-pi
    reports ``input`` EXCLUDING the cache buckets; OpenAI raw reports it
    INCLUDING them; E20 §1.1 observed ``input = cached + write + fresh`` on this
    binary and route.  Both readings are computed per turn together with the
    component-magnitude ordering tests, and ``gross`` is DEFINED from whichever
    survives (see ``inclusion_probe``).
  * closure per turn: ``cached + write <= input`` and ``input == cached + write
    + fresh``; plus the total identity in its two candidate forms
    (``total == input + output`` vs ``total == input + output + reasoning``).
  * H9 cold-start assert, promoted to a first-class condition: no
    cache-eligible turn may report ``write == 0`` while reading the prefix it
    should have created.
  * H10 namespace: the workdir path AND the Codex thread id are both new — the
    thread id is provider-assigned per session and every workdir path lives
    under a namespace directory that no earlier execution used.  A retry gets
    its own workdir path too, so a discarded attempt cannot serve its cache to
    the retry (the E18 leg B attempt-2 failure mode).
  * H2 refusal detection is the three-signal disjunction the preregistration
    requires, because ``codex exec`` emits no stop reason at all (E20 §5):
    ``output_tokens == 0``, an ``error``/``turn.failed`` event, or the absence
    of an ``agent_message`` content item.
  * the MCP round trip is counted on three independent paths per unit: Codex
    ``mcp_tool_call`` items, the guard hook's ``PreToolUse`` events, and the
    server's own ``ingest_observations`` rows.
  * H11 metered-route proof: an isolated ``CODEX_HOME`` with NO auth bridge,
    the no-credential control run with the same home and the same argv, and the
    key delivered on ``codex login --with-api-key``'s stdin only.

Isolation: everything this cell writes lives under ``tmp/e21/``.  ``tmp/e17``,
``tmp/e17b``, ``tmp/e18`` and ``tmp/e19`` are never touched, the user's
``~/.codex`` is never read (``auth.json`` is not linked, copied or opened) and
only ``lstat`` mtimes of its paths are recorded.  The PATH ``codex`` (0.144.5)
and ``~/.codex/packages/standalone/current`` are left alone; this cell calls
0.145.0 by absolute path.

Privacy (R-9): rows carry counts, public ids, sha256 and usage numbers only.
Prompts, model output and transcripts are never persisted to ``raw/``.

Phases:
    preflight  binary identity, CLI version, PATH codex untouched (no model)
    mtime      lstat snapshot of the user's ~/.codex paths (no model)
    slices     warm-up / measurement / E19 task-slice identity (no model)
    prepare    materialize the 32 units (no model)
    authproof  H11 — no-credential control vs credential, same home/argv
    steady     §5 H12 steady-state verification from an input state quantity
    unit       run one (arm, session) unit of 8 turns
    wave       run one stage, ABBA submission order, parallel + TPM paced
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve()
CELL_DIR = HERE.parents[1]                 # docs/experiments/E21-CODEX-COMPONENT
REPO = HERE.parents[4]                     # repository root
if not (REPO / "pyproject.toml").exists() or not (REPO / "src" / "karc").is_dir():
    raise SystemExit(
        f"E21: repository root misresolved as {REPO} (E15 §8.4 off-by-one guard)")

sys.path.insert(0, str(REPO / "src"))

FIXTURE = REPO / "fixture" / "e4-v2"
RAW = CELL_DIR / "raw"
RUN = REPO / "tmp" / "e21"     # gitignored; NOT tmp/e17, e17b, e18, e19

# The 0.145.0 binary, by absolute path (E20).  The PATH `codex` is 0.144.5 and
# `~/.codex/packages/standalone/current` are deliberately left untouched.
BINARY = (
    "<HOME>/.local/opt/codex-0.145.0/node_modules/@openai/"
    "codex-darwin-arm64/vendor/aarch64-apple-darwin/bin/codex"
)
CODEX_VERSION = "codex-cli 0.145.0"
MODEL = "gpt-5.6-luna"
# Not fixed by the preregistration.  "low" follows E20 — the immediate
# predecessor on this binary and this billing route — rather than E5-G1's
# "high", so the reasoning-token component stays comparable to the cell that
# established the route.  Declared here before execution.
REASONING_EFFORT = "low"

# The E19 schedule, regenerated from the committed builder and gated on the
# three identity hashes E19 committed.  Identical constants => identical task
# sequence => the measurement window is the one E19 leg B measured.
SCHEDULE_SEED = 4352
SESSION_COUNT = 24
SESSION_LENGTH = 8
WARMUP_SESSIONS_N = 8
MEASURE_SESSIONS_N = 12
RESIDENT_BUDGET_TOKENS = 1756
STEADY_REL_TOLERANCE = 0.05
STEADY_MIN_UTILIZATION = 0.95
ARMS = ("karc-full", "rag-bm25")
SCHEDULE_SNAPSHOT_SHA256 = (
    "f6edbb9e33c63c935aade45961ec8995f121e50e3854713963508925df1c2a73")
SCHEDULE_SHA256 = (
    "acdec9a853ae051225ac111e4c336bae743fa52fad5264bcb1837f42693f225b")
TASKS_SHA256 = (
    "73706b12ea0cba1dcb46963a34da83d36201c942ac2958289ef0129e8d460d74")

# OpenAI cache minimum visible prefix (rate card §5.1).
OPENAI_CACHE_MIN_PREFIX = 1024
# Published gpt-5.6-luna rates, USD per million tokens
# (docs/analysis/openai-luna-rate-card.md, observed 2026-08-28).
RATES = {"input": 0.20, "write": 0.25, "read": 0.02, "output": 1.20}

# H6 ceiling: $2 for the whole cell, warm-up included.
BUDGET_CAP_USD = 2.0
BUDGET_STOP_USD = 1.6
LEDGER: Path | None = None

USER_CODEX_PATHS = ("", "auth.json", "config.toml", "sessions",
                    "packages/standalone/current")

TOKEN_KEYS = ("input_tokens", "cached_input_tokens", "cache_write_input_tokens",
              "output_tokens", "reasoning_output_tokens", "total_tokens")

# Codex emits documented NON-FATAL ``ErrorItem``s that the driver already
# treats as non-fatal.  Two of them appear on every turn of this cell and were
# identified by direct capture rather than guessed at: the hook-trust notice
# below (emitted twice per turn because this cell passes
# ``--dangerously-bypass-hook-trust`` for the benchmark guard) and, when the
# WebSocket transport is refused, a "Falling back ... to HTTPS transport"
# notice.  The first has a fixed body and therefore a stable digest; the second
# embeds a per-request cf-ray id, so it is classified by length instead.
HOOK_TRUST_NOTICE = (
    "`--dangerously-bypass-hook-trust` is enabled. Enabled hooks may run "
    "without review for this invocation."
)
HOOK_TRUST_NOTICE_SHA256 = hashlib.sha256(
    HOOK_TRUST_NOTICE.encode("utf-8")).hexdigest()

from karc.bench.driver import (  # noqa: E402
    CODEX_WRITE_TOKEN_KEY,
    CodexPersistentSession,
    DriverRequest,
)
from karc.bench.environment import ControlledCodexHome, _sha256  # noqa: E402

USAGE_KEYS_ALLOWED = set(TOKEN_KEYS)

_PRINT_LOCK = threading.Lock()
NAMESPACE = ""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def say(payload: dict) -> None:
    with _PRINT_LOCK:
        print(json.dumps(payload, ensure_ascii=False), flush=True)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def session_id(index: int) -> str:
    return f"S{index:02d}"


def warmup_sessions() -> list[int]:
    return list(range(1, WARMUP_SESSIONS_N + 1))


def measure_sessions() -> list[int]:
    first = WARMUP_SESSIONS_N + 1
    last = first + MEASURE_SESSIONS_N - 1
    if last > SESSION_COUNT:
        raise SystemExit("E21: measurement window overruns the built schedule")
    return list(range(first, last + 1))


def abba_order(sessions: list[int]) -> list[tuple[str, int]]:
    """E13 §5.0's ABBA submission order so neither arm is systematically
    earlier in wall time."""
    order: list[tuple[str, int]] = []
    for offset, index in enumerate(sessions):
        if offset % 2 == 0:
            order += [("karc-full", index), ("rag-bm25", index)]
        else:
            order += [("rag-bm25", index), ("karc-full", index)]
    return order


def ns_root() -> Path:
    return RUN / f"run{NAMESPACE}" if NAMESPACE else RUN / "run"


def unit_workroot(arm: str, sid: str, attempt: int) -> Path:
    """A retry gets its own workdir path: a discarded attempt must not be able
    to serve its provider-side cache to the attempt that replaces it
    (E18 §5, second discard)."""
    suffix = "" if attempt <= 1 else f"-a{attempt}"
    return ns_root() / "units" / f"{arm}--{sid}{suffix}"


# --- provider token-rate throttle (E19: parallelism 3 + 120k TPM) ------------

class TokenPacer:
    def __init__(self, tokens_per_minute: int) -> None:
        self.tpm = tokens_per_minute
        self._events: list[tuple[float, int]] = []
        self._lock = threading.Lock()
        self._peak = 40_000

    def _window(self, now: float) -> int:
        self._events = [(t, n) for t, n in self._events if now - t < 60.0]
        return sum(n for _, n in self._events)

    def wait(self) -> float:
        if self.tpm <= 0:
            return 0.0
        waited = 0.0
        while True:
            with self._lock:
                if self._window(time.time()) + self._peak <= self.tpm:
                    return waited
            time.sleep(1.0)
            waited += 1.0

    def record(self, tokens: int) -> None:
        if self.tpm <= 0:
            return
        with self._lock:
            self._events.append((time.time(), tokens))
            self._peak = max(self._peak, tokens)


PACER = TokenPacer(0)


class TurnFailure(RuntimeError):
    """A turn produced no usable model answer, so the unit's accounting is
    already unusable and the whole unit is abandoned."""


# ------------------------------------------------------- metered Codex home

class MeteredCodexHome(ControlledCodexHome):
    """An isolated ``CODEX_HOME`` with NO subscription auth bridge.

    Copied from E20 so the two cells share the harness: the parent class
    symlinks the user's ``auth.json`` and refuses to start without it, which
    would both break the no-credential control and read a credential this cell
    must not touch.  Here the home starts empty and the only credential that
    ever exists inside it is the metered API key that ``codex login
    --with-api-key`` writes from stdin.
    """

    def __enter__(self) -> "MeteredCodexHome":
        root = str(self.temp_root) if self.temp_root else None
        if root:
            Path(root).mkdir(parents=True, exist_ok=True)
        self.path = Path(tempfile.mkdtemp(prefix="karc-codex-home-", dir=root))
        self.path.chmod(0o700)
        self.audit_path = self.path / "guard-events.jsonl"
        self.hooks_path = self.path / "hooks.json"
        command = shlex.join(
            [str(self.python_executable), "-m", "karc.bench.codex_guard"])
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
            encoding="utf-8")
        self.manifest = {
            "schema_version": 1,
            "auth_bridge": "none-metered-api-key-written-by-cli-login",
            "user_auth_json_linked": False,
            "generated_files": [
                {"name": "hooks.json", "sha256": _sha256(self.hooks_path)}],
            "excluded": ["auth.json", "guard-events.jsonl", "tmp", "log"],
        }
        return self

    def _cli(self, argv: list[str], stdin_text: str | None = None):
        env = {k: os.environ[k] for k in ("PATH", "LANG", "TMPDIR", "TERM")
               if k in os.environ}
        env["CODEX_HOME"] = str(self.path)
        return subprocess.run(
            [BINARY, *argv], capture_output=True, text=True, timeout=180,
            env=env, input=stdin_text if stdin_text is not None else "")

    def login_status(self) -> dict:
        """Classify the CLI's own auth report.  ``codex login status`` prints a
        partially masked key, so the line itself is never stored."""
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
        return {"returncode": proc.returncode, "auth_type": kind,
                "line_length": len(line), "line_sha256": sha256_text(line)}

    def login_with_api_key(self, api_key: str) -> dict:
        proc = self._cli(["login", "--with-api-key"], stdin_text=api_key)
        text = (proc.stdout or "") + (proc.stderr or "")
        return {
            "returncode": proc.returncode,
            "succeeded": proc.returncode == 0 and "success" in text.lower(),
            "auth_json_present_in_isolated_home": (
                self.path / "auth.json").is_file(),
        }

    def rollout_usage(self, *, subtree: str | None = None) -> dict:
        """Numeric token fields Codex wrote into its own files inside the home.

        ``subtree`` restricts the walk.  The default walk is only used by the
        layout diagnostic; measurement uses ``subtree="sessions"`` because the
        home also contains a cloned Codex plugin marketplace whose
        ``plugin-eval/fixtures/observed-usage/responses.jsonl`` is a shipped
        FIXTURE, not provider traffic (raw/probe-home-layout.json).

        R-9: only the JSON key path and the integer value are extracted; no
        prompt, no message body, no command text ever leaves the home.
        """
        if self.path is None:
            return {"available": False}
        base = self.path if subtree is None else self.path / subtree
        if not base.exists():
            return {"available": False, "subtree": subtree}
        rows: list[dict] = []
        files = 0
        for path in sorted(base.rglob("*.jsonl")):
            if path.name == "guard-events.jsonl":
                continue
            files += 1
            for lineno, line in enumerate(
                    path.read_text(encoding="utf-8", errors="replace").splitlines()):
                line = line.strip()
                if not line.startswith("{") or "tokens" not in line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                found: dict[str, int] = {}
                _collect_token_fields(obj, "", found)
                if found:
                    rows.append({"file": path.name, "line": lineno,
                                 "fields": found})
        return {"available": True, "jsonl_files": files, "records": rows}

    def token_count_events(self) -> list[dict]:
        """Ordered per-API-CALL usage records from the isolated home's rollout.

        Codex writes a ``token_count`` event per model request carrying both
        ``last_token_usage`` (that request) and ``total_token_usage`` (the
        thread so far).  This is the second usage-acquisition route and the
        only route with per-call granularity: ``codex exec --json``'s
        ``turn.completed.usage`` on stdout carries the CUMULATIVE total, which
        was measured, not assumed (raw/probe-home-layout.json).
        """
        if self.path is None:
            return []
        out: list[dict] = []
        for path in sorted((self.path / "sessions").rglob("*.jsonl")):
            for line in path.read_text(encoding="utf-8",
                                       errors="replace").splitlines():
                line = line.strip()
                if not line.startswith("{") or "last_token_usage" not in line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                info = ((obj.get("payload") or {}).get("info") or {})
                last, total = info.get("last_token_usage"), info.get(
                    "total_token_usage")
                if not isinstance(last, dict) or not isinstance(total, dict):
                    continue
                out.append({
                    "last": {k: int(v) for k, v in last.items()
                             if isinstance(v, (int, float))},
                    "total": {k: int(v) for k, v in total.items()
                              if isinstance(v, (int, float))},
                })
        return out


def _collect_token_fields(node, prefix: str, out: dict) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if key in TOKEN_KEYS and isinstance(value, (int, float)):
                out[path] = int(value)
            else:
                _collect_token_fields(value, path, out)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            _collect_token_fields(value, f"{prefix}[{index}]", out)


class MeteredCodexSession(CodexPersistentSession):
    """A persistent Codex session bound to an externally managed metered home.

    The home is entered by the caller because the no-credential control has to
    run inside the same home BEFORE the key is written.  ``parse_output``,
    ``build_initial_argv`` and ``build_resume_argv`` are inherited unmodified,
    so this cell's argv builder and normalizer are byte-identical to E15's and
    E20's (E20 §3 harness equivalence).
    """

    def __init__(self, request: DriverRequest, *, home: MeteredCodexHome):
        self.last_event_summary: dict = {}
        super().__init__(request, binary=BINARY, runner=self._runner_devnull)
        self._external_home = home

    def _runner_devnull(self, argv, cwd, timeout, env):
        proc = subprocess.run(
            argv, cwd=cwd, capture_output=True, text=True, timeout=timeout,
            env=env, stdin=subprocess.DEVNULL)
        self.last_event_summary = _event_summary(proc.stdout or "")
        return proc

    def __enter__(self) -> "MeteredCodexSession":
        self._home = self._external_home
        overrides = dict(self.request.environment)
        overrides["KARC_CODEX_GUARD_MODE"] = self.request.hook_mode
        self._environment = self._home.child_environment(overrides)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._home = None
        self._environment = None


def _event_summary(stdout: str) -> dict:
    """Event-level summary (E20 §5).  ``codex exec`` documents no stop reason,
    so every non-usage scalar on ``turn.completed`` is captured instead of
    asserting one.  Message bodies are reduced to digests (R-9)."""
    types: dict[str, int] = {}
    completed_extra: dict = {}
    error_digests: list[str] = []
    item_types: dict[str, int] = {}
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
        if etype == "item.completed":
            item = ev.get("item") or {}
            if isinstance(item, dict):
                itype = str(item.get("type"))
                item_types[itype] = item_types.get(itype, 0) + 1
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
            error_digests.append(sha256_text(payload)[:16])
    return {
        "event_type_counts": types,
        "item_completed_type_counts": item_types,
        "turn_completed_non_usage_fields": completed_extra,
        "turn_completed_count": types.get("turn.completed", 0),
        "error_event_digests": error_digests,
        "stop_reason_field_present": any(
            k in completed_extra
            for k in ("stop_reason", "finish_reason", "status", "reason")),
        "cost_field_present": any(
            "cost" in str(k).lower() for k in completed_extra),
    }


# --------------------------------------------------------------- schedule

def build_bundle() -> dict:
    from karc.bench import e5_cache_canary as g1

    bundle = g1.prepare_bundle(
        REPO, FIXTURE, schedule_seed=SCHEDULE_SEED,
        session_count=SESSION_COUNT, session_length=SESSION_LENGTH)
    snapshot = g1.bundle_snapshot(bundle)
    if snapshot["sha256"] != SCHEDULE_SNAPSHOT_SHA256:
        raise SystemExit(
            "E21: regenerated schedule does not match E19's committed identity "
            f"({snapshot['sha256']} != {SCHEDULE_SNAPSHOT_SHA256})")
    if bundle["schedule"]["sha256"] != SCHEDULE_SHA256:
        raise SystemExit("E21: schedule sha256 mismatch")
    if bundle["schedule"]["tasks_sha256"] != TASKS_SHA256:
        raise SystemExit("E21: tasks sha256 mismatch")
    if int(bundle["manifest"]["cell"]["budget_tokens"]) != RESIDENT_BUDGET_TOKENS:
        raise SystemExit("E21: resident budget is not the preregistered 1,756")
    bundle["_snapshot"] = snapshot
    return bundle


def resident_of(bundle: dict, index: int) -> dict:
    """The resident set injected for one session, as an INPUT state quantity."""
    manifest = bundle["manifest"]
    session = bundle["sessions"][index - 1]
    resident = [v for v in session["initial_resident_versions"]
                if v in manifest["artifacts"]]
    tokens = sum(int(manifest["artifacts"][v]["size_tok"]) for v in resident)
    return {"session_index": index, "session_id": session["session_id"],
            "resident_versions": len(resident), "resident_tokens": tokens,
            "budget_tokens": RESIDENT_BUDGET_TOKENS,
            "budget_utilization": tokens / RESIDENT_BUDGET_TOKENS}


def slice_versions(bundle: dict, indices: list[int]) -> set[str]:
    out: set[str] = set()
    for index in indices:
        for item in bundle["sessions"][index - 1]["tasks"]:
            out.update(item["task"]["required_versions"])
    return out


# --------------------------------------------------------------- preflight

def phase_preflight() -> dict:
    def run(argv: list[str]) -> dict:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=60)
        return {"returncode": proc.returncode,
                "stdout": (proc.stdout or "").strip()[:400],
                "stderr_tail": (proc.stderr or "").strip().splitlines()[-2:]}

    absolute = run([BINARY, "--version"])
    path_codex = shutil.which("codex")
    path_version = run([path_codex, "--version"]) if path_codex else None
    standalone = Path.home() / ".codex" / "packages" / "standalone" / "current"
    try:
        link = os.readlink(standalone)
    except OSError:
        link = None
    from karc.bench.driver import CodexCliDriver
    contract = CodexCliDriver(binary=BINARY).preflight()
    return {
        "binary": BINARY,
        "binary_version": absolute,
        "binary_version_is_pinned": absolute["stdout"] == CODEX_VERSION,
        "path_codex": path_codex,
        "path_codex_version": path_version,
        "path_codex_left_alone": True,
        "user_standalone_current_symlink": link,
        "driver_preflight": contract,
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "python": sys.executable,
        "karc_command": str(Path(sys.executable).parent / "karc"),
    }


# ------------------------------------------------------------------- mtime

def phase_mtime() -> dict:
    """lstat only.  The user's ``auth.json`` is never opened."""
    base = Path.home() / ".codex"
    out: dict = {}
    for rel in USER_CODEX_PATHS:
        path = base / rel if rel else base
        try:
            st = path.lstat()
        except OSError:
            out[rel or "."] = None
            continue
        out[rel or "."] = {
            "mtime_ns": st.st_mtime_ns,
            "mtime_utc": datetime.fromtimestamp(
                st.st_mtime, timezone.utc).isoformat(timespec="seconds"),
            "size": st.st_size,
        }
    return {"paths": out, "observed_at_utc": now_utc(),
            "note": "lstat metadata only; contents never read"}


# ------------------------------------------------------------------ slices

def phase_slices() -> dict:
    bundle = build_bundle()
    warm, meas = warmup_sessions(), measure_sessions()
    warm_v, meas_v = slice_versions(bundle, warm), slice_versions(bundle, meas)
    e19_reuse = []
    for index in meas:
        session = bundle["sessions"][index - 1]
        e19_reuse.append({
            "session_id": session["session_id"],
            "reuse_turns": sum(1 for item in session["tasks"]
                               if item["task"]["e5_resident_reuse"]),
        })
    return {
        "warmup_sessions": [session_id(i) for i in warm],
        "measure_sessions": [session_id(i) for i in meas],
        "unused_sessions": [session_id(i) for i in range(1, SESSION_COUNT + 1)
                            if i not in warm and i not in meas],
        "distinct_required_versions": {"warmup": len(warm_v),
                                       "measure": len(meas_v)},
        "overlap_warmup_vs_measure": len(warm_v & meas_v),
        "schedule_identity": {
            "snapshot_sha256": bundle["_snapshot"]["sha256"],
            "schedule_sha256": bundle["schedule"]["sha256"],
            "tasks_sha256": bundle["schedule"]["tasks_sha256"],
            "equals_E19_committed_constants": True,
            "what_this_proves": "the measurement window's task sequence is "
                                "bit-identical to E19's, so the E19 leg B "
                                "contrast holds provider, model, rate card and "
                                "task sequence fixed and varies the harness",
        },
        # E19 §5 disclosed that its window's reuse mixture (4/7 sessions with 4
        # reuse turns, 3/7 with 3) is what leg A's gross margin leaned on.  The
        # same window is used here, so the same profile is recorded up front
        # instead of being discovered afterwards.
        "window_reuse_profile": e19_reuse,
        "resident_ramp": [resident_of(bundle, i)
                          for i in range(1, SESSION_COUNT + 1)],
    }


# ----------------------------------------------------------------- prepare

def _materialize_unit(arm: str, session: dict, manifest: dict,
                      attempt: int = 1) -> dict:
    """One (arm, session) unit, through the committed Codex-native
    ``materialize_session``.  Nothing about the materialization is adapted:
    AGENTS.md stays in the workdir (Codex auto-loads it) and the MCP server is
    registered with Codex's own ``mcp_servers.*`` overrides."""
    from karc.bench import e5_cache_canary as g1

    root = unit_workroot(arm, session["session_id"], attempt)
    if root.exists():
        shutil.rmtree(root)
    work = root / "work"
    materialized = g1.materialize_session(
        arm, work, fixture_root=FIXTURE, manifest=manifest,
        initial_resident_versions=session["initial_resident_versions"])
    resident = materialized["initial_resident_versions"]
    return {
        "arm": arm, "session_id": session["session_id"], "attempt": attempt,
        "workdir": str(work),
        "agents_sha256": materialized["agents_sha256"],
        "agents_bytes": materialized["agents_bytes"],
        "hook_mode": materialized["hook_mode"],
        "config_overrides": list(materialized["config_overrides"]),
        "initial_resident_versions": resident,
        "initial_resident_tokens": sum(
            int(manifest["artifacts"][v]["size_tok"]) for v in resident),
        "managed_artifacts": len(materialized["managed_versions"]),
        "index_db_bytes": ((work / ".karc" / "index.db").stat().st_size
                           if (work / ".karc" / "index.db").exists() else None),
    }


def phase_prepare() -> dict:
    bundle = build_bundle()
    manifest = bundle["manifest"]
    ns_root().mkdir(parents=True, exist_ok=True)
    units = []
    for arm in ARMS:
        for index in (warmup_sessions() if arm == "karc-full" else []) + measure_sessions():
            units.append(_materialize_unit(arm, bundle["sessions"][index - 1],
                                           manifest))
            say({"prepared": f"{units[-1]['arm']}/{units[-1]['session_id']}"})
    return {
        "harness": "codex-cli", "binary": BINARY,
        "codex_version": CODEX_VERSION, "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "schedule_seed": SCHEDULE_SEED, "session_count": SESSION_COUNT,
        "schedule_snapshot_sha256": bundle["_snapshot"]["sha256"],
        "schedule_sha256": bundle["schedule"]["sha256"],
        "tasks_sha256": bundle["schedule"]["tasks_sha256"],
        "fixture_manifest_sha256": manifest["manifest_sha256"],
        "retrieval_audit": bundle["retrieval_audit"],
        "cell": manifest["cell"],
        "namespace": NAMESPACE,
        "units": units,
    }


# --------------------------------------------------------------- authproof

def _driver_request(unit: dict) -> DriverRequest:
    environment = {
        "KARC_CODEX_FIXTURE_MANIFEST": str(FIXTURE / "manifest.json"),
        "KARC_CODEX_INJECTED_ARTIFACTS": json.dumps(
            unit["initial_resident_versions"], separators=(",", ":")),
        "KARC_CODEX_UNRESOLVED_POLICY": "deny",
    }
    return DriverRequest(
        prompt="", cwd=unit["workdir"], model=MODEL, runtime="codex",
        timeout_s=900, config_overrides=tuple(unit["config_overrides"]),
        hook_mode=unit["hook_mode"], reasoning_effort=REASONING_EFFORT,
        environment=environment)


def _read_api_key() -> str:
    """Read OPENAI_API_KEY out of ~/api-key.txt without ever printing it."""
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key
    path = Path.home() / "api-key.txt"
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[len("export "):]
        name, _, value = line.partition("=")
        if name.strip() == "OPENAI_API_KEY":
            return value.strip().strip('"').strip("'")
    raise SystemExit("E21: OPENAI_API_KEY not found in ~/api-key.txt")


def phase_authproof() -> dict:
    """H11 — prove the metered route with the same home and the same argv.

    E20 §2's construction, tightened as instructed: the key is delivered on
    ``codex login --with-api-key``'s stdin only, never in argv and never in the
    child environment.  The control spends one model turn on the success side
    so that the *only* difference between the failing and succeeding runs is the
    presence of the credential.  That turn is a probe, not a dataset row.
    """
    bundle = build_bundle()
    manifest = bundle["manifest"]
    unit = _materialize_unit("karc-full", bundle["sessions"][0], manifest,
                             attempt=90)
    request = _driver_request(unit)
    before = phase_mtime()
    key = _read_api_key()
    out: dict = {"probe_workdir": unit["workdir"]}
    with MeteredCodexHome(temp_root=ns_root() / "homes") as home:
        out["status_before_any_credential"] = home.login_status()
        with MeteredCodexSession(request, home=home) as session:
            probe = session.run_turn("Reply with exactly: ok")
            out["exec_without_credential"] = {
                "ok": probe.ok, "error_class": probe.error_class,
                "unauthorized_in_error": "401" in (probe.error_detail or ""),
                "usage_raw_empty": not (probe.usage_raw or {}),
                "thread_id_assigned": session.thread_id is not None,
                "event_summary": dict(session.last_event_summary),
            }
            out["login_with_api_key"] = home.login_with_api_key(key)
            out["status_after_credential"] = home.login_status()
            if out["status_after_credential"]["auth_type"] != "api-key":
                raise SystemExit("E21: CLI does not report api-key auth")
            if out["exec_without_credential"]["ok"]:
                raise SystemExit(
                    "E21: the no-credential control succeeded — isolation is "
                    "not what this cell claims")
            paid = session.run_turn("Reply with exactly: ok")
            usage = dict(paid.usage_raw or {})
            out["exec_with_credential"] = {
                "ok": paid.ok, "error_class": paid.error_class,
                "usage_raw": usage,
                "usage_raw_unexpected_keys": sorted(
                    set(usage) - USAGE_KEYS_ALLOWED),
                "write_key_present": CODEX_WRITE_TOKEN_KEY in usage,
                "usage_schema": (paid.driver_metadata or {}).get("usage_schema"),
                "reported_model": paid.reported_model,
                "model_verification": paid.model_verification,
                "thread_id_assigned": session.thread_id is not None,
                "event_summary": dict(session.last_event_summary),
            }
        out["rollout_token_count_events"] = home.token_count_events()
        out["rollout_usage_sessions_subtree"] = home.rollout_usage(
            subtree="sessions")
        out["home_manifest"] = dict(home.manifest)
    after = phase_mtime()
    out["user_codex_mtimes_before"] = before["paths"]
    out["user_codex_mtimes_after"] = after["paths"]
    out["user_codex_untouched"] = before["paths"] == after["paths"]
    out["argv_shape"] = {
        "initial_tail": _argv_shape(MeteredCodexSession, request),
        "note": "the failing and succeeding runs used the same home, the same "
                "request object and the same argv builder; only auth.json "
                "differs",
    }
    return out


def _argv_shape(_cls, request: DriverRequest) -> list[str]:
    """The argv the session builds, with the prompt and cwd elided (R-9)."""
    probe = CodexPersistentSession(request, binary=BINARY, runner=lambda *a: None)
    argv = probe.build_initial_argv("<prompt>")
    return [a if a != request.cwd else "<cwd>" for a in argv[1:]]


# ------------------------------------------------------------------ steady

def phase_steady() -> dict:
    """§5 H12 verification.  The judged quantity is the resident injection
    token count — an INPUT state quantity — never a cost ratio."""
    bundle = build_bundle()
    rows = []
    for index in warmup_sessions():
        row = resident_of(bundle, index)
        sid = row["session_id"]
        unit_file = ns_root() / "units" / f"karc-full--{sid}.json"
        row["executed"] = None
        if unit_file.exists():
            unit = json.loads(unit_file.read_text(encoding="utf-8"))
            row["executed"] = {
                "turns_executed": unit["turns_executed"],
                "thread_id_sha256": unit.get("thread_id_sha256"),
            }
        agents = Path(row_workdir(sid)) / "AGENTS.md"
        if agents.exists():
            row.update(_injected_resident_bytes(agents))
        rows.append(row)
    last, prev = rows[-1], rows[-2]
    rel = (abs(last["resident_tokens"] - prev["resident_tokens"])
           / prev["resident_tokens"]) if prev["resident_tokens"] else None
    util = last["budget_utilization"]
    within = rel is not None and rel <= STEADY_REL_TOLERANCE
    utilized = util >= STEADY_MIN_UTILIZATION
    reached = bool(within and utilized)
    measure_rows = [resident_of(bundle, i) for i in measure_sessions()]
    return {
        "warmup_sessions_run": WARMUP_SESSIONS_N,
        "per_session": rows,
        "criterion": {
            "definition": "last warm-up session's resident injection tokens "
                          "per turn within 5% of the previous session, and "
                          ">= 95% utilization of the 1,756-token budget",
            "last_session": last["session_id"],
            "previous_session": prev["session_id"],
            "resident_tokens_last": last["resident_tokens"],
            "resident_tokens_previous": prev["resident_tokens"],
            "relative_change": rel,
            "relative_change_within_tolerance": within,
            "budget_utilization": util,
            "utilization_at_least_95pct": utilized,
        },
        "steady_state_reached": reached,
        "H12_fired": not reached,
        "measurement_window_resident": measure_rows,
        "measurement_window_utilization_range": [
            min(r["budget_utilization"] for r in measure_rows),
            max(r["budget_utilization"] for r in measure_rows)],
        "all_warmup_turns_executed": all(
            (r["executed"] or {}).get("turns_executed") == SESSION_LENGTH
            for r in rows),
    }


def row_workdir(sid: str) -> str:
    return str(unit_workroot("karc-full", sid, 1) / "work")


def _injected_resident_bytes(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    inside = False
    body = 0
    blocks = 0
    for line in text.splitlines(keepends=True):
        if line.startswith("<!-- artifact:start "):
            inside = True
            blocks += 1
            continue
        if line.startswith("<!-- artifact:end "):
            inside = False
            continue
        if inside:
            body += len(line.encode("utf-8"))
    return {"agents_md_bytes": len(text.encode("utf-8")),
            "injected_artifact_blocks": blocks,
            "injected_artifact_bytes": body}


# ------------------------------------------------------------------- turns

COMPONENT_KEYS = ("input_tokens", "cached_input_tokens",
                  CODEX_WRITE_TOKEN_KEY, "output_tokens",
                  "reasoning_output_tokens", "total_tokens")


def _ints(usage: dict) -> dict:
    return {k: int(usage.get(k, 0) or 0) for k in COMPONENT_KEYS}


def turn_accounting(cumulative: dict, previous_cumulative: dict,
                    normalized: dict) -> dict:
    """Everything needed to DECIDE the accounting instead of assuming it.

    TWO things are measured here rather than inherited.

    1.  The REPORTING LEVEL.  ``codex exec --json``'s ``turn.completed.usage``
        is the CUMULATIVE thread total, not the turn's own usage.  That was
        established by direct comparison against the rollout's
        ``last_token_usage`` / ``total_token_usage`` pair inside the same home
        (raw/probe-home-layout.json), and it means the per-turn quantity is the
        DIFFERENCE of consecutive stdout reports.  Both the cumulative report
        and the difference are carried on every row.

    2.  The INCLUSION RELATION.  Two mutually exclusive readings exist across
        the cells measured so far.  EXCLUSIVE (pi, both providers): ``input`` is
        a fresh bucket and gross is ``input + read + write``.  INCLUSIVE
        (OpenAI raw, E16; Codex metered, E20 §1.1): ``input`` already contains
        the cache buckets and gross is ``input``.  The identity and the
        component-magnitude ordering are both recorded per turn, and gross is
        defined from whichever survives.
    """
    cum, prev = _ints(cumulative), _ints(previous_cumulative)
    delta = {k: cum[k] - prev[k] for k in cum}
    inp = delta["input_tokens"]
    cached = delta["cached_input_tokens"]
    write = delta[CODEX_WRITE_TOKEN_KEY]
    out = delta["output_tokens"]
    reasoning = delta["reasoning_output_tokens"]
    total = delta["total_tokens"]
    fresh = inp - cached - write
    total_key_present = "total_tokens" in cumulative
    return {
        # the per-turn quantities every judged number is built from
        "input_tokens": inp, "cached": cached, "write": write,
        "output": out, "reasoning": reasoning, "fresh": fresh,
        "total_tokens": total, "total_tokens_key_present": total_key_present,
        # the raw cumulative report, kept so the differencing is auditable
        "cumulative": cum,
        "cumulative_previous": prev,
        "monotone": all(delta[k] >= 0 for k in delta),
        # the driver's own normalization of the CUMULATIVE report, for the
        # record: it is correct per report but must not be summed over turns
        "driver_normalized_cumulative_fresh": int(
            normalized.get("input_tokens", 0) or 0),
        # inclusion-relation evidence, on the per-turn quantities
        "closure_input_eq_cached_plus_write_plus_fresh":
            inp == cached + write + fresh,
        "cached_plus_write_le_input": cached + write <= inp,
        "cached_gt_input": cached > inp,
        "write_gt_input": write > inp,
        "cached_plus_write_gt_input": cached + write > inp,
        "total_eq_input_plus_output": (total == inp + out
                                       if total_key_present else None),
        "total_eq_input_plus_output_plus_reasoning": (
            total == inp + out + reasoning if total_key_present else None),
        "reasoning_le_output": reasoning <= out,
        # the two candidate gross readings, both carried forward
        "gross_inclusive": inp,
        "gross_exclusive": inp + cached + write,
        "cold": cached == 0,
    }


def call_accounting(last: dict) -> dict:
    """Per-API-call accounting from one rollout ``last_token_usage`` record.

    This is the granularity E19 had through pi's per-message usage and the
    granularity the cache-eligibility and cold-start asserts need: the visible
    prefix of one request.
    """
    u = _ints(last)
    inp = u["input_tokens"]
    cached = u["cached_input_tokens"]
    write = u[CODEX_WRITE_TOKEN_KEY]
    fresh = inp - cached - write
    return {
        "input_tokens": inp, "cached": cached, "write": write,
        "output": u["output_tokens"], "reasoning": u["reasoning_output_tokens"],
        "total_tokens": u["total_tokens"], "fresh": fresh,
        "closure_input_eq_cached_plus_write_plus_fresh":
            inp == cached + write + fresh,
        "cached_plus_write_le_input": cached + write <= inp,
        "cached_gt_input": cached > inp,
        "cached_plus_write_gt_input": cached + write > inp,
        "total_eq_input_plus_output": u["total_tokens"] == inp + u["output_tokens"],
        "total_eq_input_plus_output_plus_reasoning":
            u["total_tokens"] == inp + u["output_tokens"]
            + u["reasoning_output_tokens"],
        "prefix_tokens": inp,
        "cache_eligible": inp >= OPENAI_CACHE_MIN_PREFIX,
        "sub_threshold": inp < OPENAI_CACHE_MIN_PREFIX,
        "cold": cached == 0,
        "no_cache_activity": cached == 0 and write == 0,
        # H9: write == 0 while the prefix is served from cache.  `prefix_grew`
        # separates "served a cache written outside this dataset" from "there
        # was simply nothing new to write".
        "cold_start_suspect": write == 0 and cached > 0,
    }


def priced_usd(fresh: int, write: int, read: int, output: int) -> float:
    return (fresh * RATES["input"] + write * RATES["write"]
            + read * RATES["read"] + output * RATES["output"]) / 1e6


def phase_unit(arm: str, index: int, *, attempt: int = 1,
               force: bool = False) -> dict:
    from karc.bench import e5_cache_canary as g1

    bundle = build_bundle()
    manifest = bundle["manifest"]
    session = bundle["sessions"][index - 1]
    sid = session["session_id"]
    out_path = ns_root() / "units" / f"{arm}--{sid}.json"
    failed_path = ns_root() / "units" / f"{arm}--{sid}.failed.json"
    if out_path.exists() and not force:
        return json.loads(out_path.read_text(encoding="utf-8"))
    if failed_path.exists():
        failed_path.unlink()

    # A (re)run always starts from a freshly materialized unit under a workdir
    # path this cell has never used: reusing a partially executed unit would
    # double-count the server's ingest_observations rows and could be served the
    # provider-side cache the abandoned attempt left behind.
    unit = _materialize_unit(arm, session, manifest, attempt=attempt)
    work = Path(unit["workdir"])
    request = _driver_request(unit)
    key = _read_api_key()

    rows: list[dict] = []
    thread_id = None
    guard_total = 0
    with MeteredCodexHome(temp_root=ns_root() / "homes") as home:
        auth = {"before": home.login_status(),
                "login": home.login_with_api_key(key)}
        auth["after"] = home.login_status()
        if auth["after"]["auth_type"] != "api-key":
            raise SystemExit(f"E21: {arm}/{sid} home is not on an api-key auth")
        previous_cumulative: dict = {}
        rollout_seen = 0
        with MeteredCodexSession(request, home=home) as driver:
            for item in session["tasks"]:
                position = int(item["task"]["session_task"])
                prompt = g1.task_prompt(arm, item, FIXTURE, manifest)
                paced = PACER.wait()
                started = time.time()
                result = driver.run_turn(prompt)
                elapsed = time.time() - started
                usage_raw = dict(result.usage_raw or {})
                meta = result.driver_metadata or {}
                guard = list(meta.get("guard_events") or [])
                guard_total += len(guard)
                summary = dict(driver.last_event_summary)
                acct = turn_accounting(usage_raw, previous_cumulative,
                                       dict(result.usage or {}))
                if usage_raw:
                    previous_cumulative = usage_raw
                # second acquisition route, per API call
                events = home.token_count_events()
                new_events = events[rollout_seen:]
                rollout_seen = len(events)
                calls = [call_accounting(e["last"]) for e in new_events]
                for index, call in enumerate(calls):
                    call["prefix_grew"] = (
                        index > 0
                        and call["prefix_tokens"] > calls[index - 1]["prefix_tokens"])
                route_sum = {
                    "input_tokens": sum(c["input_tokens"] for c in calls),
                    "cached": sum(c["cached"] for c in calls),
                    "write": sum(c["write"] for c in calls),
                    "output": sum(c["output"] for c in calls),
                    "reasoning": sum(c["reasoning"] for c in calls),
                }
                route_agrees = bool(calls) and all(
                    route_sum[k] == acct[k]
                    for k in ("input_tokens", "cached", "write", "output",
                              "reasoning"))
                rollout_cumulative_agrees = bool(new_events) and all(
                    int(new_events[-1]["total"].get(k, 0) or 0)
                    == int(usage_raw.get(k, 0) or 0)
                    for k in ("input_tokens", "cached_input_tokens",
                              CODEX_WRITE_TOKEN_KEY, "output_tokens"))
                tools = g1.tool_counts(result.transcript, guard)
                mcp_items = sum(
                    1 for e in result.transcript
                    if e.get("kind") == "tool_use"
                    and e.get("logical_server") == "karc")
                guard_mcp = sum(
                    1 for e in guard
                    if e.get("hook_event_name") == "PreToolUse"
                    and e.get("mcp_server") == "karc")
                agent_messages = int(
                    summary.get("item_completed_type_counts", {}).get(
                        "agent_message", 0))
                error_items = [e for e in result.transcript
                               if e.get("name") == "codex_error_item"]
                hook_notices = sum(
                    1 for e in error_items
                    if e.get("message_sha256") == HOOK_TRUST_NOTICE_SHA256)
                other_notices = [
                    {"message_length": e.get("message_length"),
                     "message_sha256_16": str(e.get("message_sha256"))[:16]}
                    for e in error_items
                    if e.get("message_sha256") != HOOK_TRUST_NOTICE_SHA256]
                answer = result.output_text or ""
                grade = (g1.grade_turn(item["task"], answer) if result.ok
                         else {"passed": False, "value_match": "error",
                               "format_ok": False, "abstained": False})
                row = {
                    "harness": "codex-cli", "codex_version": CODEX_VERSION,
                    "provider": "openai", "model": MODEL,
                    "reasoning_effort": REASONING_EFFORT,
                    "arm": arm, "session_id": sid, "position": position,
                    "attempt": attempt,
                    "task_id": item["task"]["task_id"],
                    "required_versions": item["task"]["required_versions"],
                    "reuse": bool(item["task"]["e5_resident_reuse"]),
                    "policy_resident_hit": bool(item["policy_resident_hit"]),
                    "rag_plan_tokens": int(item["rag_plan"]["tokens"]),
                    "prompt_sha256": sha256_text(prompt),
                    "prompt_chars": len(prompt),
                    "ok": result.ok,
                    "error_class": result.error_class,
                    "error_detail_digest": (
                        sha256_text(result.error_detail)[:16]
                        if result.error_detail else None),
                    "wall_seconds": round(elapsed, 2),
                    "throttle_wait_seconds": paced,
                    "usage_raw": usage_raw,
                    "usage_raw_unexpected_keys": sorted(
                        set(usage_raw) - USAGE_KEYS_ALLOWED),
                    "write_key_present": CODEX_WRITE_TOKEN_KEY in usage_raw,
                    "usage_schema": meta.get("usage_schema"),
                    "fresh_semantics": meta.get("fresh_semantics"),
                    "usage_normalized": dict(result.usage or {}),
                    "driver_gross_with_cache_cumulative":
                        result.total_input_tokens(with_cache=True),
                    "driver_fresh_no_cache_cumulative":
                        result.total_input_tokens(with_cache=False),
                    "accounting": acct,
                    "calls": calls,
                    "api_calls": len(calls),
                    "second_route": {
                        "rollout_token_count_events": len(new_events),
                        "per_call_sum": route_sum,
                        "per_call_sum_equals_stdout_delta": route_agrees,
                        "rollout_cumulative_equals_stdout": (
                            rollout_cumulative_agrees),
                    },
                    # H2: three-signal disjunction (codex exec emits no stop
                    # reason at all, E20 §5).
                    "refusal_signals": {
                        "output_tokens_zero": acct["output"] == 0,
                        "error_or_turn_failed_event": bool(
                            summary.get("event_type_counts", {}).get("error")
                            or summary.get("event_type_counts", {}).get(
                                "turn.failed")
                            or summary.get("error_event_digests")),
                        "content_block_absent": agent_messages == 0
                        or not answer.strip(),
                    },
                    # Non-fatal ErrorItems, classified rather than guessed.
                    # `unclassified_notices` empty on every turn means no
                    # transport fallback happened, i.e. the turn stayed on the
                    # WebSocket transport (E20 §7's first [U]).
                    "nonfatal_error_items": len(error_items),
                    "hook_trust_notices": hook_notices,
                    "unclassified_notices": other_notices,
                    "stop_reason_field_present": summary.get(
                        "stop_reason_field_present"),
                    "turn_completed_non_usage_fields": summary.get(
                        "turn_completed_non_usage_fields"),
                    "turn_completed_count": summary.get("turn_completed_count"),
                    "event_type_counts": summary.get("event_type_counts"),
                    "item_completed_type_counts": summary.get(
                        "item_completed_type_counts"),
                    "mcp_calls_codex_events": mcp_items,
                    "mcp_calls_guard_hook": guard_mcp,
                    "tool_counts": tools,
                    "hook_event_count": meta.get("hook_event_count"),
                    "session_turn": meta.get("session_turn"),
                    "persistent_session": meta.get("persistent_session"),
                    "reported_model": result.reported_model,
                    "model_verification": result.model_verification,
                    "priced_usd": priced_usd(acct["fresh"], acct["write"],
                                             acct["cached"], acct["output"]),
                    "grade_incidental": grade,
                }
                rows.append(row)
                PACER.record(acct["input_tokens"] + acct["output"])
                say({"unit": f"{arm}/{sid}", "position": position,
                     "ok": result.ok, "input": acct["input_tokens"],
                     "cached": acct["cached"], "write": acct["write"],
                     "out": acct["output"], "mcp": mcp_items,
                     "calls": len(calls), "route_ok": route_agrees,
                     "usd": round(row["priced_usd"], 6),
                     "s": round(elapsed, 1)})
                signals = row["refusal_signals"]
                if (not result.ok or any(signals.values())
                        or acct["input_tokens"] <= 0 or not acct["monotone"]):
                    detail = {
                        "unit": f"{arm}--{sid}", "position": position,
                        "attempt": attempt, "ok": result.ok,
                        "error_class": result.error_class,
                        "refusal_signals": signals,
                        "rows_before_failure": len(rows) - 1,
                    }
                    failed_path.parent.mkdir(parents=True, exist_ok=True)
                    failed_path.write_text(
                        json.dumps(detail, indent=2, sort_keys=True,
                                   ensure_ascii=False) + "\n", encoding="utf-8")
                    say({"unit": f"{arm}--{sid}", "abort": detail})
                    raise TurnFailure(json.dumps(detail))
            thread_id = driver.thread_id
        rollout_events = home.token_count_events()
        home_manifest = dict(home.manifest)

    db_observations = None
    db = work / ".karc" / "index.db"
    if db.exists():
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        db_observations = conn.execute(
            "select count(*) from ingest_observations "
            "where source_channel='mcp'").fetchone()[0]
        conn.close()

    result = {
        "arm": arm, "session_id": sid, "attempt": attempt,
        "workdir": unit["workdir"],
        "thread_id_sha256": (sha256_text(thread_id) if thread_id else None),
        "turns_requested": len(session["tasks"]),
        "turns_executed": len(rows),
        "harness": "codex-cli", "codex_version": CODEX_VERSION,
        "provider": "openai", "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "auth": auth,
        "home_manifest": home_manifest,
        "rollout_token_count_events": len(rollout_events),
        "rollout_final_total": (rollout_events[-1]["total"]
                                if rollout_events else None),
        "stdout_final_cumulative": (dict(rows[-1]["usage_raw"]) if rows
                                    else None),
        "component_totals_from_deltas": {
            key: sum(r["accounting"][key] for r in rows)
            for key in ("input_tokens", "cached", "write", "output",
                        "reasoning", "fresh")},
        "three_paths": {
            "codex_mcp_tool_call_items": sum(
                r["mcp_calls_codex_events"] for r in rows),
            "guard_hook_pretooluse_mcp": sum(
                r["mcp_calls_guard_hook"] for r in rows),
            "db_ingest_observations_mcp": db_observations,
        },
        "guard_events_total": guard_total,
        "rows": rows,
        "_meta": {"generated_at_utc": now_utc(), "namespace": NAMESPACE},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True,
                                   ensure_ascii=False) + "\n",
                        encoding="utf-8")
    return result


# -------------------------------------------------------------------- wave

def _ledger_add(usd: float) -> float:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    book = (json.loads(LEDGER.read_text(encoding="utf-8"))
            if LEDGER.exists() else {"usd": 0.0})
    book["usd"] = float(book.get("usd", 0.0)) + usd
    LEDGER.write_text(json.dumps(book, indent=2) + "\n", encoding="utf-8")
    return book["usd"]


def phase_wave(*, stage: str, parallel: int, force: bool,
               only: list[str] | None, unit_retries: int = 2) -> dict:
    if stage == "warmup":
        order = [("karc-full", i) for i in warmup_sessions()]
    elif stage == "measure":
        order = abba_order(measure_sessions())
    else:
        raise SystemExit(f"E21: unknown stage {stage!r}")
    if only is not None:
        order = [(arm, i) for arm, i in order if f"{arm}--S{i:02d}" in only]
    spent = {"usd": 0.0}
    lock = threading.Lock()
    aborted: list[str] = []
    discarded: list[dict] = []

    def work(job: tuple[str, int]) -> dict:
        arm, index = job
        with lock:
            if aborted:
                return {"skipped": f"{arm}--S{index:02d}", "reason": aborted[0]}
        attempt = 0
        while True:
            attempt += 1
            try:
                result = phase_unit(arm, index, attempt=attempt, force=force)
                break
            except TurnFailure as failure:
                with lock:
                    discarded.append({"unit": f"{arm}--S{index:02d}",
                                      "attempt": attempt,
                                      "detail": json.loads(str(failure))})
                if attempt > unit_retries:
                    with lock:
                        aborted.append(
                            f"unit {arm}--S{index:02d} failed {attempt} times")
                    return {"failed": f"{arm}--S{index:02d}",
                            "attempts": attempt, "detail": str(failure)}
                say({"retry": f"{arm}--S{index:02d}", "next_attempt": attempt + 1})
                time.sleep(30.0)
        result["attempts"] = attempt
        with lock:
            unit_usd = sum(float(r["priced_usd"]) for r in result["rows"])
            spent["usd"] += unit_usd
            cell_usd = _ledger_add(unit_usd)
            if cell_usd > BUDGET_STOP_USD:
                aborted.append(f"H6 guard: cell priced spend ${cell_usd:.4f} "
                               f"exceeded ${BUDGET_STOP_USD}")
            say({"wave": stage, "done": f"{arm}--{result['session_id']}",
                 "turns": result["turns_executed"],
                 "wave_usd": round(spent["usd"], 6),
                 "cell_usd": round(cell_usd, 6)})
        return result

    started = time.time()
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        results = list(pool.map(work, order))
    book = (json.loads(LEDGER.read_text(encoding="utf-8"))
            if LEDGER.exists() else {"usd": 0.0})
    return {
        "stage": stage, "parallel": parallel, "namespace": NAMESPACE,
        "submission_order": [f"{arm}--S{i:02d}" for arm, i in order],
        "units": len(order),
        "wall_seconds": round(time.time() - started, 1),
        "tokens_per_minute_target": PACER.tpm,
        "aborted": aborted,
        "discarded_attempts": discarded,
        "unit_attempts": {f"{r.get('arm')}--{r.get('session_id')}":
                          r.get("attempts")
                          for r in results if "arm" in r},
        "failed": [r for r in results if "failed" in r],
        "wave_priced_usd": round(spent["usd"], 6),
        "cell_priced_usd_running": round(float(book["usd"]), 6),
        "unit_summaries": [{
            "arm": r.get("arm"), "session_id": r.get("session_id"),
            "turns_executed": r.get("turns_executed"),
            "three_paths": r.get("three_paths"),
        } for r in results if "arm" in r],
        "skipped": [r for r in results if "skipped" in r],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["preflight", "mtime", "slices",
                                          "prepare", "authproof", "steady",
                                          "unit", "wave"])
    parser.add_argument("--arm", choices=ARMS, default=None)
    parser.add_argument("--session", type=int, default=None)
    parser.add_argument("--stage", choices=["warmup", "measure"], default=None)
    parser.add_argument("--parallel", type=int, default=3)
    parser.add_argument("--tpm", type=int, default=120_000)
    parser.add_argument("--unit-retries", type=int, default=2)
    parser.add_argument("--ns", default="",
                        help="execution namespace suffix; changes every workdir "
                             "path so a discarded execution's provider-side "
                             "cache cannot be served to this one")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--only", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    global PACER, NAMESPACE, LEDGER
    PACER = TokenPacer(args.tpm)
    NAMESPACE = args.ns
    LEDGER = RUN / "spend-ledger.json"
    RAW.mkdir(parents=True, exist_ok=True)

    if args.phase == "preflight":
        result, out = phase_preflight(), args.out or "preflight.json"
    elif args.phase == "mtime":
        result, out = phase_mtime(), args.out or "mtime-before.json"
    elif args.phase == "slices":
        result, out = phase_slices(), args.out or "slices.json"
    elif args.phase == "prepare":
        result, out = phase_prepare(), args.out or "prepare.json"
    elif args.phase == "authproof":
        result, out = phase_authproof(), args.out or "authproof.json"
    elif args.phase == "steady":
        result, out = phase_steady(), args.out or "steady.json"
    elif args.phase == "unit":
        result = phase_unit(args.arm, args.session, force=args.force)
        out = args.out or f"unit-{args.arm}-S{args.session:02d}.json"
    else:
        if args.stage is None:
            raise SystemExit("E21: wave requires --stage warmup|measure")
        only = args.only.split(",") if args.only else None
        result = phase_wave(stage=args.stage, parallel=args.parallel,
                            force=args.force, only=only,
                            unit_retries=args.unit_retries)
        out = args.out or f"wave-{args.stage}.json"
    result.setdefault("_meta", {})
    result["_meta"].update({"phase": args.phase, "cell": "E21-CODEX-COMPONENT",
                            "namespace": NAMESPACE,
                            "codex_version": CODEX_VERSION,
                            "generated_at_utc": now_utc()})
    (RAW / out).write_text(json.dumps(result, indent=2, sort_keys=True,
                                      ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(f"wrote {RAW / out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
