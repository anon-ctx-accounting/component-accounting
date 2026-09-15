"""E18-PI-REVERSAL — does the gross/priced ordering reversal reproduce on pi?

Two paired legs over one identical schedule (preregistration appendix A.1):

    leg A   provider anthropic, claude-sonnet-5, PI_CACHE_RETENTION=long
    leg B   provider openai,    gpt-5.6-luna,    PI_CACHE_RETENTION unset (B.3)

Everything else is shared: pi 0.84.3, the `fixture/e4-v2` corpus, the E10-P2-XR
canary schedule (seed 4200, 12 sessions x 8 turns), the two arms `karc-full`
(MCP through `integrations/pi/karc-mcp-bridge`) and `rag-bm25` (evidence in the
turn message, no tools), and the turn order.

Instrumentation contract, all of it measured rather than assumed:

  * gross = input + cacheRead + cacheWrite  (E17 §3 and E17B §2; pi's
    `usage.input` is already a fresh bucket on BOTH legs, so treating it like a
    provider raw `input_tokens` under-counts gross by ~5.5x).
  * closure assert per call: totalTokens == input + output + cacheRead +
    cacheWrite, and cacheWrite1h <= cacheWrite (leg A only; on leg B the key
    does not exist at all, E17B §4).
  * stop reason in two layers: pi's normalized `stopReason` and the provider's
    `rawStopReason`.  On leg B the provider layer is always "completed" and has
    no discriminating power, so refusal detection uses the H2' disjunction
    (output == 0, errorMessage present, content block absent).
  * the MCP round trip is counted on three independent paths per unit: pi's
    `tool_execution_start` events, the bridge audit JSONL, and the K-ARC
    server's own `ingest_observations` rows.
  * `--tools` typos are not validated by pi, so the tool names are read back out
    of pi's own registry by the model-free `registry` phase (E17 §7.2).

Isolation (E17 §6, mandatory rather than optional): pi prefers `auth.json` over
the environment, so without `PI_CODING_AGENT_DIR` redirection a user's OAuth
credential silently wins over the metered API key.  Every path this cell writes
lives under `tmp/e18/`; `tmp/e17` and `tmp/e17b` are never touched.

Privacy (R-9): raw rows carry counts, ids, sha256 and usage numbers only.
Prompts, model text and tool results never reach `raw/`.  The event streams are
kept under `tmp/e18/` (gitignored) so a parser bug does not require paying
twice, and are deleted at teardown.

Phases:
    mtime     snapshot the user's ~/.pi paths (no pi, no model)
    auth      model-free `pi auth check` inside the isolated home
    prepare   materialize 24 units for one leg (no pi, no model)
    registry  model-free registry dump through pi --mode rpc
    unit      run one (leg, arm, session) unit of 8 turns
    wave      run all 24 units of one leg, ABBA submission order, parallel
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve()
CELL_DIR = HERE.parents[1]                 # docs/experiments/E18-PI-REVERSAL
REPO = HERE.parents[4]                     # repository root
if not (REPO / "pyproject.toml").exists() or not (REPO / "src" / "karc").is_dir():
    raise SystemExit(f"E18: repository root misresolved as {REPO} (E15 §8.4 off-by-one guard)")

sys.path.insert(0, str(REPO / "src"))

FIXTURE = REPO / "fixture" / "e4-v2"
BRIDGE = REPO / "integrations" / "pi" / "karc-mcp-bridge" / "index.ts"
PROBE_EXT = CELL_DIR / "scripts" / "tool-registry-probe.ts"
RAW = CELL_DIR / "raw"
RUN = REPO / "tmp" / "e18"                 # gitignored; NOT tmp/e17, NOT tmp/e17b
PI_BIN = REPO / "tmp" / "vendor" / "piprobe" / "node_modules" / ".bin" / "pi"

PI_VERSION = "0.84.3"
SCHEDULE_SEED = 4200
SESSION_COUNT = 12
SESSION_LENGTH = 8
ARMS = ("karc-full", "rag-bm25")
TOOLS = ("karc_search", "karc_get")
THINKING = "off"

# The schedule this cell runs is the committed E10-P2-XR canary schedule,
# regenerated rather than copied so the task prompts (which the snapshot does
# not carry) come from the same deterministic builder.  The snapshot sha256 is
# asserted against the committed file, which is the identity gate.
XR_SCHEDULE = REPO / "docs" / "experiments" / "E10-P2-XR" / "canary" / "raw" / "schedule.json"

# ABBA submission order copied verbatim from E13 §5.0 (the baseline cell's
# executed block order), so neither arm is systematically earlier in wall time.
ABBA_ORDER = (
    ("karc-full", 1), ("rag-bm25", 1), ("rag-bm25", 2), ("karc-full", 2),
    ("karc-full", 3), ("rag-bm25", 3), ("rag-bm25", 4), ("karc-full", 4),
    ("karc-full", 5), ("rag-bm25", 5), ("rag-bm25", 6), ("karc-full", 6),
    ("karc-full", 7), ("rag-bm25", 7), ("rag-bm25", 8), ("karc-full", 8),
    ("karc-full", 9), ("rag-bm25", 9), ("rag-bm25", 10), ("karc-full", 10),
    ("karc-full", 11), ("rag-bm25", 11), ("rag-bm25", 12), ("karc-full", 12),
)

LEGS = {
    "A": {
        "provider": "anthropic",
        "model": "claude-sonnet-5",
        "retention": "long",          # preregistration §3
        "key_env": "KARC_ANTHROPIC_KEY",
        "provider_key_env": "ANTHROPIC_API_KEY",
    },
    "B": {
        "provider": "openai",
        "model": "gpt-5.6-luna",
        "retention": None,            # appendix B.3: default, unset
        "key_env": "KARC_OPENAI_KEY",
        "provider_key_env": "OPENAI_API_KEY",
    },
}

# OpenAI cache minimum prefix (docs/analysis/openai-luna-rate-card.md §5.1).
OPENAI_CACHE_MIN_PREFIX = 1024

# pi's own catalog rates, USD per million tokens, recovered by reverse solving
# pi's `usage.cost` from the four buckets on committed gate raw.  These price
# the CONTROL column only; the judged column uses the committed standard rate
# card (see analyze_e18.py).
PI_CATALOG_RATES = {
    "anthropic": {"input": 2.0, "output": 10.0, "read": 0.20,
                  "write_5m": 2.50, "write_1h": 4.00},
    "openai": {"input": 0.20, "output": 1.20, "read": 0.02,
               "write_5m": 0.25, "write_1h": 0.25},
}

# Standard-rate ceiling for the H6 budget guard ($15 for both legs together).
BUDGET_CAP_USD = 15.0
BUDGET_STOP_USD = 12.0

USER_PI_PATHS = (
    Path.home() / ".pi",
    Path.home() / ".pi" / "agent",
    Path.home() / ".pi" / "agent" / "auth.json",
    Path.home() / ".pi" / "agent" / "models-store.json",
)

_PRINT_LOCK = threading.Lock()

# --- provider token-rate throttle -------------------------------------------
# The OpenAI leg shares one org-wide tokens-per-minute budget (measured on the
# first leg-B attempt: "Limit 200000, Used 200000").  pi's own auto-retry gives
# up after 3 attempts and then emits an assistant message with
# stopReason "error" and an all-zero usage block, which is exactly the
# accounting distortion H2' exists to reject.  So the rate is governed here
# instead of being discovered by the provider.
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
                used = self._window(time.time())
                if used + self._peak <= self.tpm:
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
    """A turn came back with no model answer (transport error or refusal).

    Continuing the unit past this point would leave a user message with no
    assistant reply in the session, so the whole unit is abandoned and can only
    be retried after a full reset.
    """


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def say(payload: dict) -> None:
    with _PRINT_LOCK:
        print(json.dumps(payload, ensure_ascii=False), flush=True)


# Execution namespace.  A discarded execution leaves a live server-side prompt
# cache behind: OpenAI keeps a written prefix reusable for 30 minutes and pi
# routes it with prompt_cache_key = session id, so a re-run that reuses the same
# workdir paths and the same session ids is served that cache and reports
# `cacheWrite == 0` on calls that a clean run would have been charged creation
# for.  A namespace changes both the workdir path (which appears in pi's system
# prompt, so the prefix bytes differ) and the pi session id (so the cache key
# differs).  The effect is verified after the fact, not assumed: the crosscheck
# counts calls with cacheWrite == 0 and cacheRead > 0, which must be zero.
NAMESPACE = ""


def leg_root(leg: str) -> Path:
    return RUN / f"leg{leg}{NAMESPACE}"


def unit_dir(leg: str, arm: str, session_id: str) -> Path:
    return leg_root(leg) / "units" / f"{arm}--{session_id}"


def pi_env(leg: str, extra: dict[str, str] | None = None) -> dict[str, str]:
    """Isolated pi environment.  The user's ~/.pi is never read or written."""
    env = {
        "HOME": os.environ["HOME"],
        "PATH": os.environ["PATH"],
        "LANG": os.environ.get("LANG", "en_US.UTF-8"),
        "PI_CODING_AGENT_DIR": str(leg_root(leg) / "pi-home"),
        "PI_CODING_AGENT_SESSION_DIR": str(leg_root(leg) / "sessions"),
        "PI_OFFLINE": "1",
        "PI_SKIP_VERSION_CHECK": "1",
        "PI_TELEMETRY": "0",
    }
    if extra:
        env.update(extra)
    return env


def build_bundle() -> dict:
    """Rebuild the committed XR canary bundle and gate on its snapshot sha256."""
    from karc.bench import e5_cache_canary as g1

    bundle = g1.prepare_bundle(
        REPO, FIXTURE, schedule_seed=SCHEDULE_SEED,
        session_count=SESSION_COUNT, session_length=SESSION_LENGTH,
    )
    snapshot = g1.bundle_snapshot(bundle)
    committed = json.loads(XR_SCHEDULE.read_text(encoding="utf-8"))["sha256"]
    if snapshot["sha256"] != committed:
        raise SystemExit(
            "E18: regenerated schedule does not match the committed E10-P2-XR "
            f"canary ({snapshot['sha256']} != {committed})")
    bundle["_snapshot"] = snapshot
    return bundle


# ------------------------------------------------------------------ mtime

def phase_mtime() -> dict:
    return {"paths": {str(path): (
        {"exists": True, "mtime_ns": path.stat().st_mtime_ns,
         "size": path.stat().st_size if path.is_file() else None}
        if path.exists() else {"exists": False}) for path in USER_PI_PATHS}}


# ------------------------------------------------------------------- auth

def phase_auth(leg: str) -> dict:
    cfg = LEGS[leg]
    (leg_root(leg) / "pi-home").mkdir(parents=True, exist_ok=True)
    (leg_root(leg) / "sessions").mkdir(parents=True, exist_ok=True)
    results: dict = {"leg": leg, "provider": cfg["provider"]}
    for with_key in (False, True):
        extra = {}
        if with_key:
            key = os.environ.get(cfg["key_env"])
            if not key:
                results["with_env_key"] = {"skipped": f"{cfg['key_env']} unset"}
                continue
            extra[cfg["provider_key_env"]] = key
        proc = subprocess.run(
            [str(PI_BIN), "auth", "check", "--provider", cfg["provider"], "--json"],
            cwd=str(REPO), env=pi_env(leg, extra), text=True, capture_output=True,
            timeout=120)
        # `--credentials` is deliberately NOT passed: it prints the secret.
        results["with_env_key" if with_key else "no_env_key"] = {
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip()[:600],
            "stderr_tail": proc.stderr.strip().splitlines()[-3:],
        }
    auth_json = leg_root(leg) / "pi-home" / "auth.json"
    results["isolated_auth_json"] = {
        "exists": auth_json.exists(),
        "bytes": auth_json.stat().st_size if auth_json.exists() else None,
    }
    return results


# ---------------------------------------------------------------- prepare

def _materialize_unit(leg: str, arm: str, session: dict, manifest: dict) -> dict:
    """One (leg, arm, session) unit.  Faithful to e5_cache_canary's
    materialize_session, with two pi-specific adaptations recorded here:

      1. Codex auto-loads AGENTS.md as a project doc; pi does not (this cell
         runs with --no-context-files).  The same AGENTS.md bytes are therefore
         passed explicitly with --append-system-prompt, and the file is moved
         OUT of the workdir so it cannot also be discovered.
      2. The MCP server is reached through the pi bridge's `.mcp.json` instead
         of Codex `mcp_servers.*` config overrides.
    """
    from karc.bench import materialize as mzt
    from karc.bench import mcp_index
    from karc.bench import e5_cache_canary as g1

    root = unit_dir(leg, arm, session["session_id"])
    if root.exists():
        shutil.rmtree(root)
    work = root / "work"
    work.mkdir(parents=True)
    context = root / "context"
    context.mkdir()

    if arm == "rag-bm25":
        sha, nbytes = mzt._codex_agents(
            work, manifest, [], instruction=g1.RAG_SESSION_INSTRUCTION)
        resident: list[str] = []
        managed: list[str] = []
        index_bytes = None
    elif arm == "karc-full":
        shutil.copytree(FIXTURE / "repo", work, dirs_exist_ok=True)
        resident = [v for v in session["initial_resident_versions"]
                    if v in manifest["artifacts"]]
        sha, nbytes = mzt._codex_agents(
            work, manifest, resident, instruction=g1.KARC_SESSION_INSTRUCTION)
        managed = sorted(manifest["artifacts"],
                         key=lambda vid: manifest["artifacts"][vid]["path"])
        managed_dir = work / mzt.MANAGED_ROOT
        for version_id in managed:
            entry = manifest["artifacts"][version_id]
            source = work / entry["path"]
            target = managed_dir / entry["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.exists():
                shutil.move(str(source), str(target))
        index_db = work / ".karc" / "index.db"
        mcp_index.build_index(work, manifest, managed,
                              managed_root=mzt.MANAGED_ROOT, db_path=index_db)
        index_bytes = index_db.stat().st_size
        loader = (f"import sys;sys.path.insert(0,{str(REPO / 'src')!r});"
                  "from karc.cli import main;raise SystemExit(main())")
        (work / ".mcp.json").write_text(json.dumps({"mcpServers": {"karc": {
            "command": sys.executable,
            "args": ["-c", loader, "mcp", "serve", "--root", str(work),
                     "--db", str(index_db), "--runtime", "pi"],
        }}}, indent=2), encoding="utf-8")
    else:
        raise SystemExit(f"E18: unknown arm {arm!r}")

    # Adaptation 1: move AGENTS.md out of the workdir and use it as the
    # explicit system-prompt append source.
    agents = work / "AGENTS.md"
    system_append = context / "system-append.md"
    shutil.move(str(agents), str(system_append))

    return {
        "leg": leg, "arm": arm, "session_id": session["session_id"],
        "workdir": str(work),
        "system_append_path": str(system_append),
        "system_append_sha256": sha,
        "system_append_bytes": nbytes,
        "initial_resident_versions": resident,
        "initial_resident_tokens": sum(
            int(manifest["artifacts"][v]["size_tok"]) for v in resident),
        "managed_artifacts": len(managed),
        "index_db_bytes": index_bytes,
    }


def phase_prepare(leg: str) -> dict:
    bundle = build_bundle()
    manifest = bundle["manifest"]
    leg_root(leg).mkdir(parents=True, exist_ok=True)
    (leg_root(leg) / "pi-home").mkdir(parents=True, exist_ok=True)
    (leg_root(leg) / "sessions").mkdir(parents=True, exist_ok=True)
    (leg_root(leg) / "pi-home" / "settings.json").write_text(json.dumps({
        "defaultProjectTrust": "never",
        "quietStartup": True,
        "enableInstallTelemetry": False,
        "enableAnalytics": False,
        "compaction": {"enabled": False},
    }, indent=2), encoding="utf-8")

    units = []
    for arm in ARMS:
        for session in bundle["sessions"]:
            units.append(_materialize_unit(leg, arm, session, manifest))
            say({"prepared": units[-1]["arm"] + "/" + units[-1]["session_id"]})
    return {
        "leg": leg,
        "pi_version": PI_VERSION,
        "provider": LEGS[leg]["provider"],
        "model": LEGS[leg]["model"],
        "retention": LEGS[leg]["retention"],
        "schedule_snapshot_sha256": bundle["_snapshot"]["sha256"],
        "schedule_matches_committed_xr_canary": True,
        "committed_xr_schedule_sha256": json.loads(
            XR_SCHEDULE.read_text(encoding="utf-8"))["sha256"],
        "fixture_manifest_sha256": manifest["manifest_sha256"],
        "retrieval_audit": bundle["retrieval_audit"],
        "cell": manifest["cell"],
        "units": units,
    }


# --------------------------------------------------------------- registry

def _registry_probe(leg: str, *, arm: str, append: bool) -> dict:
    """One model-free pi start.  `rpc get_state` makes no model call, so this
    reads pi's own registry and system-prompt size for free."""
    cfg = LEGS[leg]
    root = unit_dir(leg, arm, "S01")
    work = root / "work"
    system_append = root / "context" / "system-append.md"
    if not work.is_dir():
        raise SystemExit("E18: run the prepare phase before registry")
    tag = f"{arm}-{'append' if append else 'noappend'}"
    out = leg_root(leg) / f"registry-probe-{tag}.json"
    if out.exists():
        out.unlink()
    audit = leg_root(leg) / f"registry-bridge-audit-{tag}.jsonl"
    if audit.exists():
        audit.unlink()
    argv = [str(PI_BIN), "--mode", "rpc",
            "--provider", cfg["provider"], "--model", cfg["model"],
            "--thinking", THINKING, "--no-extensions"]
    if arm == "karc-full":
        argv += ["-e", str(BRIDGE), "-e", str(PROBE_EXT),
                 "--no-builtin-tools", "--tools", ",".join(TOOLS)]
    else:
        argv += ["-e", str(PROBE_EXT), "--no-tools"]
    argv += ["--no-context-files", "--no-skills", "--no-prompt-templates",
             "--no-themes", "--no-approve", "--no-session"]
    if append:
        argv += ["--append-system-prompt", str(system_append)]
    extra = {"KARC_E18_PROBE_OUT": str(out),
             "KARC_PI_BRIDGE_AUDIT": str(audit),
             "KARC_PI_MCP_CONFIG": str(work / ".mcp.json"),
             "KARC_PI_MCP_SERVER": "karc"}
    key = os.environ.get(cfg["key_env"])
    if key:
        # No model call is made by rpc get_state; the key only lets pi resolve
        # the provider so the registry can be dumped at all.
        extra[cfg["provider_key_env"]] = key
    proc = subprocess.run(argv, cwd=str(work), env=pi_env(leg, extra),
                          input='{"type":"get_state"}\n', text=True,
                          capture_output=True, timeout=300)
    responses = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            responses.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    state = next((r.get("data", {}) for r in responses
                  if r.get("command") == "get_state"), {})
    probe = json.loads(out.read_text(encoding="utf-8")) if out.exists() else None
    return {
        "arm": arm, "append_system_prompt": append,
        "argv_tail": argv[1:],
        "returncode": proc.returncode,
        "stderr_tail": proc.stderr.strip().splitlines()[-5:],
        "rpc_event_types": sorted({str(r.get("type")) for r in responses}),
        "get_state": {
            "model": (state.get("model") or {}).get("id"),
            "provider": (state.get("model") or {}).get("provider"),
            "api": (state.get("model") or {}).get("api"),
            "autoCompactionEnabled": state.get("autoCompactionEnabled"),
        },
        "probe": probe,
        "bridge_audit_lines": (len(audit.read_text(encoding="utf-8").splitlines())
                               if audit.exists() else 0),
    }


def phase_registry(leg: str) -> dict:
    """Four model-free starts: each arm with and without the system-prompt
    append.  The with/without pair measures that --append-system-prompt <path>
    really injected the AGENTS.md bytes (pi resolves an existing path to its
    contents, core/resource-loader.js resolvePromptInput), which is the pi
    substitute for Codex's project-doc auto-load."""
    probes = [_registry_probe(leg, arm=arm, append=append)
              for arm in ARMS for append in (False, True)]
    by_key = {f"{p['arm']}--{'append' if p['append_system_prompt'] else 'noappend'}": p
              for p in probes}
    deltas = {}
    for arm in ARMS:
        no = by_key[f"{arm}--noappend"].get("probe") or {}
        yes = by_key[f"{arm}--append"].get("probe") or {}
        deltas[arm] = {
            "system_prompt_chars_noappend": no.get("system_prompt_chars"),
            "system_prompt_chars_append": yes.get("system_prompt_chars"),
            "delta_chars": (None if no.get("system_prompt_chars") is None
                            or yes.get("system_prompt_chars") is None
                            else yes["system_prompt_chars"] - no["system_prompt_chars"]),
            "active_tools": yes.get("active_tools"),
        }
    return {"leg": leg, "probes": probes,
            "system_prompt_injection": deltas}


# ------------------------------------------------------------------- turns

def parse_json_stream(lines: list[str]) -> dict:
    """Aggregate one `pi --mode json` run.  `usage` comes from message_end.
    `usage_keys` is retained because the presence of a key and a value of 0 are
    different facts (E16 §1.2)."""
    api_calls: list[dict] = []
    tool_starts: list[str] = []
    tool_ends: list[dict] = []
    event_types: dict[str, int] = {}
    errors: list[str] = []
    final_text: list[str] = []
    for line in lines:
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            errors.append("unparsable-line")
            continue
        etype = str(event.get("type"))
        event_types[etype] = event_types.get(etype, 0) + 1
        if etype == "tool_execution_start":
            tool_starts.append(str(event.get("toolName")))
        elif etype == "tool_execution_end":
            tool_ends.append({"tool": str(event.get("toolName")),
                              "isError": bool(event.get("isError"))})
        elif etype == "message_end":
            message = event.get("message") or {}
            if message.get("role") != "assistant":
                continue
            usage = message.get("usage") or {}
            blocks = message.get("content") or []
            text = "\n".join(str(b.get("text") or "") for b in blocks
                             if b.get("type") == "text")
            if text.strip():
                final_text.append(text)
            api_calls.append({
                "input": usage.get("input"),
                "output": usage.get("output"),
                "cacheRead": usage.get("cacheRead"),
                "cacheWrite": usage.get("cacheWrite"),
                "cacheWrite1h": usage.get("cacheWrite1h"),
                "cacheWrite1h_key_present": "cacheWrite1h" in usage,
                "reasoning": usage.get("reasoning"),
                "totalTokens": usage.get("totalTokens"),
                "usage_keys": sorted(usage.keys()),
                "pi_cost_total": (usage.get("cost") or {}).get("total"),
                "stopReason": message.get("stopReason"),
                "rawStopReason": message.get("rawStopReason"),
                "errorMessage_present": bool(message.get("errorMessage")),
                "content_block_types": [str(b.get("type")) for b in blocks],
                "responseModel": message.get("responseModel") or message.get("model"),
            })
        elif etype in ("extension_error", "error"):
            errors.append(etype)
    # `final_text` stays in memory only: it feeds the incidental grade and is
    # never written anywhere (R-9).
    return {"api_calls": api_calls, "tool_starts": tool_starts,
            "tool_ends": tool_ends, "event_types": event_types,
            "stream_errors": errors, "final_text": final_text}


def closure_of(call: dict, *, leg: str) -> dict:
    inp = int(call.get("input") or 0)
    out = int(call.get("output") or 0)
    read = int(call.get("cacheRead") or 0)
    write = int(call.get("cacheWrite") or 0)
    write1h = int(call.get("cacheWrite1h") or 0)
    total = int(call.get("totalTokens") or 0)
    prefix = inp + read + write            # == provider raw input_tokens
    return {
        "input": inp, "output": out, "cacheRead": read, "cacheWrite": write,
        "cacheWrite1h": write1h,
        "cacheWrite1h_key_present": bool(call.get("cacheWrite1h_key_present")),
        "totalTokens": total,
        "prefix_tokens": prefix,
        "gross_tokens": prefix,
        "closure_total_eq_four_sum": total == inp + out + read + write,
        "inclusive_hypothesis_total_eq_input_plus_output": total == inp + out,
        "input_ge_read_plus_write": inp >= read + write,
        "write1h_le_write": write1h <= write,
        "cache_eligible_openai": prefix >= OPENAI_CACHE_MIN_PREFIX,
        "sub_threshold_openai": prefix < OPENAI_CACHE_MIN_PREFIX,
        "cold": read == 0,
        "stopReason": call.get("stopReason"),
        "rawStopReason": call.get("rawStopReason"),
        "errorMessage_present": bool(call.get("errorMessage_present")),
        "content_block_count": len(call.get("content_block_types") or []),
        "pi_cost_total": call.get("pi_cost_total"),
        "pi_rate_card_modelled_usd": pi_modelled_cost(
            leg, inp, out, read, write, write1h),
    }


def pi_modelled_cost(leg: str, inp: int, out: int, read: int,
                     write: int, write1h: int) -> float:
    """Reverse the pi catalog rate card out of the four buckets."""
    rates = PI_CATALOG_RATES[LEGS[leg]["provider"]]
    write5m = max(0, write - write1h)
    return (inp * rates["input"] + out * rates["output"] + read * rates["read"]
            + write5m * rates["write_5m"] + write1h * rates["write_1h"]) / 1e6


def _sum(rows: list[dict], key: str) -> int:
    return sum(int(row.get(key) or 0) for row in rows)


def phase_unit(leg: str, arm: str, session_index: int, *,
               force: bool = False) -> dict:
    """Run 8 turns of one (leg, arm, session) unit through pi."""
    from karc.bench import e5_cache_canary as g1

    cfg = LEGS[leg]
    key = os.environ.get(cfg["key_env"])
    if not key:
        raise SystemExit(f"E18: {cfg['key_env']} is required to run turns")

    bundle = build_bundle()
    manifest = bundle["manifest"]
    session = bundle["sessions"][session_index - 1]
    session_id = session["session_id"]
    root = unit_dir(leg, arm, session_id)
    work = root / "work"
    system_append = root / "context" / "system-append.md"

    out_path = leg_root(leg) / "units" / f"{arm}--{session_id}.json"
    failed_path = leg_root(leg) / "units" / f"{arm}--{session_id}.failed.json"
    if out_path.exists() and not force:
        return json.loads(out_path.read_text(encoding="utf-8"))
    if failed_path.exists():
        failed_path.unlink()

    # A (re)run always starts from a freshly materialized unit and an empty
    # session store.  Reusing a partially executed unit would double-count the
    # bridge's ingest_observations rows (breaking the three-path check) and
    # would resume a conversation whose last turn has no assistant reply.
    _materialize_unit(leg, arm, session, manifest)
    for stale in (leg_root(leg) / "sessions").glob(
            f"*_e18{NAMESPACE}-{arm}-{session_id}.jsonl"):
        stale.unlink()

    pi_session_id = f"e18{NAMESPACE}-{arm}-{session_id}"
    audit = root / "bridge-audit.jsonl"
    if audit.exists():
        audit.unlink()
    extra = {cfg["provider_key_env"]: key}
    if arm == "karc-full":
        extra["KARC_PI_BRIDGE_AUDIT"] = str(audit)
        extra["KARC_PI_MCP_CONFIG"] = str(work / ".mcp.json")
        extra["KARC_PI_MCP_SERVER"] = "karc"
    if cfg["retention"] is not None:
        extra["PI_CACHE_RETENTION"] = cfg["retention"]

    streams = root / "streams"
    streams.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    audit_before = 0
    for item in session["tasks"]:
        position = int(item["task"]["session_task"])
        prompt = g1.task_prompt(arm, item, FIXTURE, manifest)
        argv = [str(PI_BIN), "-p", prompt, "--mode", "json",
                "--provider", cfg["provider"], "--model", cfg["model"],
                "--thinking", THINKING,
                "--no-extensions"]
        if arm == "karc-full":
            argv += ["-e", str(BRIDGE), "--no-builtin-tools",
                     "--tools", ",".join(TOOLS)]
        else:
            argv += ["--no-tools"]
        argv += ["--no-context-files", "--no-skills", "--no-prompt-templates",
                 "--no-themes", "--no-approve",
                 "--append-system-prompt", str(system_append),
                 "--session-dir", str(leg_root(leg) / "sessions"),
                 "--session-id", pi_session_id]
        paced = PACER.wait()
        started = time.time()
        proc = subprocess.run(argv, cwd=str(work), env=pi_env(leg, extra),
                              text=True, capture_output=True, timeout=1800)
        elapsed = time.time() - started
        (streams / f"turn{position:02d}.jsonl").write_text(proc.stdout,
                                                           encoding="utf-8")
        parsed = parse_json_stream(proc.stdout.splitlines())
        audit_lines = (audit.read_text(encoding="utf-8").splitlines()
                       if audit.exists() else [])
        audit_turn = [json.loads(line) for line in audit_lines[audit_before:]]
        audit_before = len(audit_lines)
        calls = parsed["api_calls"]
        closure = [closure_of(call, leg=leg) for call in calls]
        answer = parsed["final_text"][-1] if parsed["final_text"] else ""
        grade = g1.grade_turn(item["task"], answer)
        row = {
            "leg": leg, "provider": cfg["provider"], "model": cfg["model"],
            "retention": cfg["retention"], "arm": arm,
            "session_id": session_id, "position": position,
            "task_id": item["task"]["task_id"],
            "required_versions": item["task"]["required_versions"],
            "reuse": bool(item["task"]["e5_resident_reuse"]),
            "policy_resident_hit": bool(item["policy_resident_hit"]),
            "rag_plan_tokens": int(item["rag_plan"]["tokens"]),
            "prompt_sha256": sha256(prompt),
            "prompt_chars": len(prompt),
            "returncode": proc.returncode,
            "wall_seconds": round(elapsed, 2),
            "throttle_wait_seconds": paced,
            "api_call_count": len(calls),
            "usage_turn": {
                "input": _sum(calls, "input"),
                "output": _sum(calls, "output"),
                "cacheRead": _sum(calls, "cacheRead"),
                "cacheWrite": _sum(calls, "cacheWrite"),
                "cacheWrite1h": _sum(calls, "cacheWrite1h"),
                "reasoning": _sum(calls, "reasoning"),
                "totalTokens": _sum(calls, "totalTokens"),
                "pi_cost_total": round(sum(float(call.get("pi_cost_total") or 0.0)
                                           for call in calls), 10),
            },
            "gross_tokens": sum(c["prefix_tokens"] for c in closure),
            "stop_reasons": [call["stopReason"] for call in calls],
            "raw_stop_reasons": [call["rawStopReason"] for call in calls],
            "refusal_signals": {
                # H2' — leg A judges on the provider layer, leg B on the
                # disjunction because rawStopReason is always "completed".
                "output_tokens_zero": _sum(calls, "output") == 0,
                "any_call_output_zero": any(int(c.get("output") or 0) == 0
                                            for c in calls),
                "errorMessage_present": any(c["errorMessage_present"] for c in calls),
                "content_block_absent": any(len(c["content_block_types"]) == 0
                                            for c in calls),
                "raw_stop_refusal": any("refus" in str(c["rawStopReason"]).lower()
                                        for c in calls),
                "normalized_stop_error": any(str(c["stopReason"]) == "error"
                                             for c in calls),
            },
            "tool_execution_start_count": len(parsed["tool_starts"]),
            "tool_execution_start_names": parsed["tool_starts"],
            "tool_execution_end_errors": sum(1 for end in parsed["tool_ends"]
                                             if end["isError"]),
            "bridge_audit_lines_this_turn": len(audit_turn),
            "bridge_audit_outcomes": [entry.get("outcome") for entry in audit_turn],
            "bridge_audit_mcp_tools": [entry.get("mcpTool") for entry in audit_turn],
            "event_types": parsed["event_types"],
            "stream_errors": parsed["stream_errors"],
            "stderr_tail": proc.stderr.strip().splitlines()[-4:],
            "calls": closure,
            # Incidental only.  §7 of the preregistration forbids using the
            # accuracy axis; recorded so a silently broken arm is detectable.
            "grade_incidental": grade,
        }
        rows.append(row)
        PACER.record(row["gross_tokens"] + row["usage_turn"]["output"])
        say({"unit": f"{leg}/{arm}/{session_id}", "position": position,
             "rc": proc.returncode, "calls": len(calls),
             "gross": row["gross_tokens"],
             "pi_cost": row["usage_turn"]["pi_cost_total"],
             "tes": row["tool_execution_start_count"],
             "audit": row["bridge_audit_lines_this_turn"]})
        # Fail fast: an error stop reason, a missing answer or a zero-usage
        # call means this unit's accounting is already unusable.
        fatal = [c for c in closure
                 if str(c["stopReason"]) == "error"
                 or c["errorMessage_present"]
                 or int(c["totalTokens"] or 0) == 0]
        if proc.returncode != 0 or fatal:
            detail = {
                "unit": f"{leg}/{arm}/{session_id}", "position": position,
                "returncode": proc.returncode,
                "fatal_calls": len(fatal),
                "stop_reasons": row["stop_reasons"],
                "error_tail": proc.stderr.strip().splitlines()[-3:],
                "rows_before_failure": len(rows),
            }
            failed_path.parent.mkdir(parents=True, exist_ok=True)
            failed_path.write_text(json.dumps(detail, indent=2, sort_keys=True,
                                              ensure_ascii=False) + "\n",
                                   encoding="utf-8")
            say({"unit": f"{leg}/{arm}/{session_id}", "abort": detail})
            raise TurnFailure(json.dumps(detail))

    db_observations = None
    if arm == "karc-full":
        db = work / ".karc" / "index.db"
        if db.exists():
            conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            db_observations = conn.execute(
                "select count(*) from ingest_observations "
                "where source_channel='mcp'").fetchone()[0]
            conn.close()

    result = {
        "leg": leg, "arm": arm, "session_id": session_id,
        "pi_session_id": pi_session_id,
        "pi_session_id_sha256": sha256(pi_session_id),
        "turns_requested": len(session["tasks"]),
        "turns_executed": len(rows),
        "provider": cfg["provider"], "model": cfg["model"],
        "retention": cfg["retention"], "thinking": THINKING,
        "three_paths": {
            "pi_tool_execution_start": sum(r["tool_execution_start_count"]
                                           for r in rows),
            "bridge_audit_lines": audit_before,
            "db_ingest_observations_mcp": db_observations,
        },
        "rows": rows,
        "_meta": {"pi_version": PI_VERSION,
                  "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                    time.gmtime())},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True,
                                   ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def _standard_priced_usd(leg: str, rows: list[dict]) -> float:
    """Rough running spend at the committed standard rate cards, for the H6
    guard only.  The judged figures are produced by analyze_e18.py."""
    if LEGS[leg]["provider"] == "anthropic":
        p = {"in": 3.0, "out": 15.0, "read": 0.30, "w5": 3.75, "w1": 6.00}
    else:
        p = {"in": 0.20, "out": 1.20, "read": 0.02, "w5": 0.25, "w1": 0.25}
    total = 0.0
    for row in rows:
        u = row["usage_turn"]
        w1 = int(u["cacheWrite1h"] or 0)
        w5 = max(0, int(u["cacheWrite"] or 0) - w1)
        total += (u["input"] * p["in"] + u["output"] * p["out"]
                  + u["cacheRead"] * p["read"] + w5 * p["w5"]
                  + w1 * p["w1"]) / 1e6
    return total


def phase_wave(leg: str, *, parallel: int, force: bool,
               only: list[str] | None, unit_retries: int = 2) -> dict:
    order = [(arm, idx) for arm, idx in ABBA_ORDER
             if only is None or f"{arm}--S{idx:02d}" in only]
    spent = {"usd": 0.0}
    lock = threading.Lock()
    aborted: list[str] = []

    def work(job: tuple[str, int]) -> dict:
        arm, idx = job
        with lock:
            if aborted:
                return {"skipped": f"{arm}--S{idx:02d}", "reason": aborted[0]}
        attempts = 0
        while True:
            attempts += 1
            try:
                result = phase_unit(leg, arm, idx, force=force)
                break
            except TurnFailure as failure:
                if attempts > unit_retries:
                    with lock:
                        aborted.append(f"unit {arm}--S{idx:02d} failed "
                                       f"{attempts} times: {failure}")
                    return {"failed": f"{arm}--S{idx:02d}",
                            "attempts": attempts, "detail": str(failure)}
                say({"wave": leg, "retry": f"{arm}--S{idx:02d}",
                     "attempt": attempts + 1})
                time.sleep(30.0)
        result["attempts"] = attempts
        with lock:
            spent["usd"] += _standard_priced_usd(leg, result["rows"])
            if spent["usd"] > BUDGET_STOP_USD:
                aborted.append(f"H6 guard: running standard-priced spend "
                               f"${spent['usd']:.4f} exceeded ${BUDGET_STOP_USD}")
            say({"wave": leg, "done": f"{arm}--{result['session_id']}",
                 "turns": result["turns_executed"],
                 "running_standard_usd": round(spent["usd"], 6)})
        return result

    started = time.time()
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        results = list(pool.map(work, order))
    return {
        "leg": leg, "parallel": parallel,
        "submission_order": [f"{arm}--S{idx:02d}" for arm, idx in order],
        "units": len(order),
        "wall_seconds": round(time.time() - started, 1),
        "aborted": aborted,
        "tokens_per_minute_target": PACER.tpm,
        "unit_attempts": {f"{r.get('arm')}--{r.get('session_id')}": r.get("attempts")
                          for r in results if "arm" in r},
        "failed": [r for r in results if "failed" in r],
        "running_standard_priced_usd": round(spent["usd"], 6),
        "unit_summaries": [{
            "arm": r.get("arm"), "session_id": r.get("session_id"),
            "turns_executed": r.get("turns_executed"),
            "three_paths": r.get("three_paths"),
        } for r in results if "arm" in r],
        "skipped": [r for r in results if "skipped" in r],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["mtime", "auth", "prepare",
                                          "registry", "unit", "wave"])
    parser.add_argument("--leg", choices=sorted(LEGS), default=None)
    parser.add_argument("--arm", choices=ARMS, default=None)
    parser.add_argument("--session", type=int, default=None)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--tpm", type=int, default=0,
                        help="provider tokens-per-minute ceiling to respect "
                             "(0 disables the throttle)")
    parser.add_argument("--unit-retries", type=int, default=2)
    parser.add_argument("--ns", default="",
                        help="execution namespace suffix; changes the workdir "
                             "paths and the pi session ids so a re-run cannot be "
                             "served a discarded execution's provider-side cache")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--only", default=None,
                        help="comma list of arm--Sxx unit keys")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    global PACER, NAMESPACE
    PACER = TokenPacer(args.tpm)
    NAMESPACE = args.ns
    RAW.mkdir(parents=True, exist_ok=True)
    if args.phase == "mtime":
        result = phase_mtime()
        out = args.out or "mtime-before.json"
    elif args.phase == "auth":
        result = phase_auth(args.leg)
        out = args.out or f"auth-{args.leg}.json"
    elif args.phase == "prepare":
        result = phase_prepare(args.leg)
        out = args.out or f"prepare-{args.leg}.json"
    elif args.phase == "registry":
        result = phase_registry(args.leg)
        out = args.out or f"registry-{args.leg}.json"
    elif args.phase == "unit":
        result = phase_unit(args.leg, args.arm, args.session, force=args.force)
        out = args.out or f"unit-{args.leg}-{args.arm}-S{args.session:02d}.json"
    else:
        only = args.only.split(",") if args.only else None
        result = phase_wave(args.leg, parallel=args.parallel,
                            force=args.force, only=only,
                            unit_retries=args.unit_retries)
        out = args.out or f"wave-{args.leg}.json"
    result.setdefault("_meta", {})
    result["_meta"].update({"phase": args.phase, "pi_version": PI_VERSION,
                            "generated_at_utc": time.strftime(
                                "%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    (RAW / out).write_text(json.dumps(result, indent=2, sort_keys=True,
                                      ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(f"wrote {RAW / out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
