"""E19-STEADY-STATE — does the reversal hold once the resident set is full?

E18 could not establish the ordering reversal on its pooled statistics, and its
post-hoc per-session view is a range chosen after looking at the data, so it
cannot be a claim.  This cell removes the subset choice from the design: the
resident set is filled FIRST by a warm-up that is never judged, the arrival at
steady state is verified on an INPUT state quantity, and then every session of
the measurement window is measured and pooled into one test.

    [warm-up]  karc arm only, 8..12 sessions x 8 turns   (never judged)
        v
    [verify]   resident injection tokens per turn, last vs previous session
        v
    [measure]  12 sessions x 8 turns x 2 arms = 192 turns per leg

Two paired legs over one identical schedule:

    leg A   provider anthropic, claude-sonnet-5, PI_CACHE_RETENTION=long
    leg B   provider openai,    gpt-5.6-luna,    PI_CACHE_RETENTION unset

`rag-bm25` is not warmed up: it carries no resident state (the document store is
fixed at 408 artifacts) and its per-session totals were flat in E18.  The
asymmetry is intentional and declared in the preregistration §2/§7.

Task slices: the schedule is a NEW 24-session build (seed 4352) so that the
warm-up slice and the measurement slice are disjoint by construction, and so
that E19 does not re-run E18's task sequence.  Disjointness is measured, not
assumed, by the `slices` phase.

Instrumentation contract, inherited from E18 and all of it measured:

  * gross = input + cacheRead + cacheWrite  (E17 §3 and E17B §2; pi's
    `usage.input` is already a fresh bucket on BOTH legs, so treating it like a
    provider raw `input_tokens` under-counts gross by ~5.5x).
  * closure assert per call: totalTokens == input + output + cacheRead +
    cacheWrite, and cacheWrite1h <= cacheWrite (leg A only).
  * stop reason in two layers; on leg B the provider layer has no
    discriminating power, so refusal detection uses the H2' disjunction.
  * the MCP round trip is counted on three independent paths per unit.
  * `--tools` typos are not validated by pi, so the tool names are read back out
    of pi's own registry by the model-free `registry` phase.
  * H11 (promoted to a first-class hold condition after the E18 leg B attempt-2
    incident): no cache-eligible call may report cacheWrite == 0 while reading
    the prefix it should have created.
  * H12: the execution namespace changes BOTH the workdir path and the pi
    session id, so no provider-side cache written by another execution can be
    served to this one.

Isolation (E17 §6): every path this cell writes lives under `tmp/e19/`;
`tmp/e17`, `tmp/e17b` and `tmp/e18` are never touched.

Privacy (R-9): raw rows carry counts, ids, sha256 and usage numbers only.

Phases:
    mtime     snapshot the user's ~/.pi paths (no pi, no model)
    slices    warm-up / measurement / E18 task-slice overlap (no pi, no model)
    auth      model-free `pi auth check` inside the isolated home
    prepare   materialize 48 units for one leg (no pi, no model)
    registry  model-free registry dump through pi --mode rpc
    steady    §3 steady-state verification from executed warm-up units
    unit      run one (leg, arm, session) unit of 8 turns
    wave      run one stage of one leg, ABBA submission order, parallel
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
CELL_DIR = HERE.parents[1]                 # docs/experiments/E19-STEADY-STATE
REPO = HERE.parents[4]                     # repository root
if not (REPO / "pyproject.toml").exists() or not (REPO / "src" / "karc").is_dir():
    raise SystemExit(f"E19: repository root misresolved as {REPO} (E15 §8.4 off-by-one guard)")

sys.path.insert(0, str(REPO / "src"))

FIXTURE = REPO / "fixture" / "e4-v2"
BRIDGE = REPO / "integrations" / "pi" / "karc-mcp-bridge" / "index.ts"
PROBE_EXT = CELL_DIR / "scripts" / "tool-registry-probe.ts"
RAW = CELL_DIR / "raw"
RUN = REPO / "tmp" / "e19"                 # gitignored; NOT tmp/e17, e17b, e18
PI_BIN = REPO / "tmp" / "vendor" / "piprobe" / "node_modules" / ".bin" / "pi"

PI_VERSION = "0.84.3"
# A NEW schedule, not E18's.  24 sessions so that a warm-up of up to 12
# sessions can always be followed by a 12-session measurement window whose
# task slice is disjoint from it.  Seed chosen model-free before any
# execution by minimizing corpus overlap with E18 over seeds 4300..4399
# (config.yaml records the search); the criterion is outcome-independent.
SCHEDULE_SEED = 4352
SESSION_COUNT = 24
SESSION_LENGTH = 8
# preregistration §3: warm-up starts at 8 sessions and may grow to 12.
WARMUP_START = 8
WARMUP_MAX = 12
MEASURE_SESSIONS = 12
RESIDENT_BUDGET_TOKENS = 1756          # manifest cell budget, 5% of corpus
STEADY_REL_TOLERANCE = 0.05            # last vs previous warm-up session
STEADY_MIN_UTILIZATION = 0.95          # of RESIDENT_BUDGET_TOKENS
ARMS = ("karc-full", "rag-bm25")
TOOLS = ("karc_search", "karc_get")
THINKING = "off"

# The schedule is regenerated deterministically from the committed builder and
# gated on the snapshot sha256 recorded here BEFORE any execution.  E18 gated
# against the committed E10-P2-XR canary file; E19 uses a new schedule, so the
# identity gate is this constant plus config.yaml, both committed before the
# first model call.
SCHEDULE_SNAPSHOT_SHA256 = (
    "f6edbb9e33c63c935aade45961ec8995f121e50e3854713963508925df1c2a73")
SCHEDULE_SHA256 = (
    "acdec9a853ae051225ac111e4c336bae743fa52fad5264bcb1837f42693f225b")
TASKS_SHA256 = (
    "73706b12ea0cba1dcb46963a34da83d36201c942ac2958289ef0129e8d460d74")
# E18's schedule, read only to measure slice overlap (never re-analyzed here).
E18_SCHEDULE = REPO / "docs" / "experiments" / "E10-P2-XR" / "canary" / "raw" / "schedule.json"


def session_id(index: int) -> str:
    return f"S{index:02d}"


def warmup_sessions(count: int) -> list[int]:
    if not WARMUP_START <= count <= WARMUP_MAX:
        raise SystemExit(f"E19: warm-up must be {WARMUP_START}..{WARMUP_MAX} sessions")
    return list(range(1, count + 1))


def measure_sessions(warmup_count: int) -> list[int]:
    """The measurement window starts immediately after the warm-up window."""
    first = warmup_count + 1
    last = first + MEASURE_SESSIONS - 1
    if last > SESSION_COUNT:
        raise SystemExit("E19: measurement window overruns the built schedule")
    return list(range(first, last + 1))


def abba_order(sessions: list[int]) -> list[tuple[str, int]]:
    """E13 §5.0's ABBA submission order, generalized to any session list, so
    neither arm is systematically earlier in wall time."""
    order: list[tuple[str, int]] = []
    for offset, idx in enumerate(sessions):
        if offset % 2 == 0:
            order += [("karc-full", idx), ("rag-bm25", idx)]
        else:
            order += [("rag-bm25", idx), ("karc-full", idx)]
    return order


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
# card (see analyze_e19.py).
PI_CATALOG_RATES = {
    "anthropic": {"input": 2.0, "output": 10.0, "read": 0.20,
                  "write_5m": 2.50, "write_1h": 4.00},
    "openai": {"input": 0.20, "output": 1.20, "read": 0.02,
               "write_5m": 0.25, "write_1h": 0.25},
}

# H6' ceiling: $25 for the whole cell, warm-up included.  The guard is
# cell-global (a ledger under tmp/e19), not per-wave, because E19 runs four
# waves rather than two.
BUDGET_CAP_USD = 25.0
BUDGET_STOP_USD = 20.0
LEDGER = None                              # set in main(); RUN/"spend-ledger.json"

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
    """Rebuild the E19 schedule and gate on its snapshot sha256."""
    from karc.bench import e5_cache_canary as g1

    bundle = g1.prepare_bundle(
        REPO, FIXTURE, schedule_seed=SCHEDULE_SEED,
        session_count=SESSION_COUNT, session_length=SESSION_LENGTH,
    )
    snapshot = g1.bundle_snapshot(bundle)
    if snapshot["sha256"] != SCHEDULE_SNAPSHOT_SHA256:
        raise SystemExit(
            "E19: regenerated schedule does not match the committed identity "
            f"({snapshot['sha256']} != {SCHEDULE_SNAPSHOT_SHA256})")
    if bundle["schedule"]["sha256"] != SCHEDULE_SHA256:
        raise SystemExit("E19: schedule sha256 mismatch")
    if bundle["schedule"]["tasks_sha256"] != TASKS_SHA256:
        raise SystemExit("E19: tasks sha256 mismatch")
    if int(bundle["manifest"]["cell"]["budget_tokens"]) != RESIDENT_BUDGET_TOKENS:
        raise SystemExit("E19: resident budget is not the preregistered 1,756")
    bundle["_snapshot"] = snapshot
    return bundle


def resident_of(bundle: dict, index: int) -> dict:
    """The resident set injected for one session, as an INPUT state quantity.

    `initial_resident_versions` is produced by the model-free policy replay over
    the whole schedule, so it is the state the session STARTS from; it is fixed
    for the session and re-sent on every turn inside the system-prompt append.
    """
    manifest = bundle["manifest"]
    session = bundle["sessions"][index - 1]
    resident = [v for v in session["initial_resident_versions"]
                if v in manifest["artifacts"]]
    tokens = sum(int(manifest["artifacts"][v]["size_tok"]) for v in resident)
    return {
        "session_index": index,
        "session_id": session["session_id"],
        "resident_versions": len(resident),
        "resident_tokens": tokens,
        "budget_tokens": RESIDENT_BUDGET_TOKENS,
        "budget_utilization": tokens / RESIDENT_BUDGET_TOKENS,
    }


def slice_versions(bundle: dict, indices: list[int]) -> set[str]:
    out: set[str] = set()
    for index in indices:
        for item in bundle["sessions"][index - 1]["tasks"]:
            out.update(item["task"]["required_versions"])
    return out


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
        raise SystemExit(f"E19: unknown arm {arm!r}")

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
        "schedule_seed": SCHEDULE_SEED,
        "session_count": SESSION_COUNT,
        "schedule_snapshot_sha256": bundle["_snapshot"]["sha256"],
        "schedule_sha256": bundle["schedule"]["sha256"],
        "tasks_sha256": bundle["schedule"]["tasks_sha256"],
        "resident_ramp": [resident_of(bundle, i)
                          for i in range(1, SESSION_COUNT + 1)],
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
        raise SystemExit("E19: run the prepare phase before registry")
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
    extra = {"KARC_E19_PROBE_OUT": str(out),
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
        raise SystemExit(f"E19: {cfg['key_env']} is required to run turns")

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
            f"*_e19{NAMESPACE}-{arm}-{session_id}.jsonl"):
        stale.unlink()

    pi_session_id = f"e19{NAMESPACE}-{arm}-{session_id}"
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
    guard only.  The judged figures are produced by analyze_e19.py."""
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


# ------------------------------------------------------------ slices/steady

def phase_slices(warmup_count: int) -> dict:
    """Model-free: how far the three task slices actually are from disjoint.

    The preregistration §2 asks for warm-up / measurement / E18 slices that are
    mutually disjoint.  Within one schedule that is exact by construction (each
    session takes its own versioned anchor and its own stable artifacts).
    Against E18 it is NOT attainable: the fixture corpus is frozen and shared by
    every cell, so some artifacts recur.  This phase measures the residue rather
    than asserting a disjointness the corpus cannot provide.
    """
    bundle = build_bundle()
    warm = warmup_sessions(warmup_count)
    meas = measure_sessions(warmup_count)
    warm_v = slice_versions(bundle, warm)
    meas_v = slice_versions(bundle, meas)
    e18 = json.loads(E18_SCHEDULE.read_text(encoding="utf-8"))
    e18_v: set[str] = set()
    for session in e18["sessions"]:
        for task in session["tasks"]:
            e18_v.update(task["required_versions"])
    return {
        "warmup_sessions": [session_id(i) for i in warm],
        "measure_sessions": [session_id(i) for i in meas],
        "unused_sessions": [session_id(i) for i in range(1, SESSION_COUNT + 1)
                            if i not in warm and i not in meas],
        "distinct_required_versions": {
            "warmup": len(warm_v), "measure": len(meas_v),
            "e18_slice": len(e18_v),
        },
        "overlap": {
            "warmup_vs_measure": len(warm_v & meas_v),
            "warmup_vs_e18": len(warm_v & e18_v),
            "measure_vs_e18": len(meas_v & e18_v),
        },
        "task_sequence_identity": {
            "e19_schedule_sha256": bundle["schedule"]["sha256"],
            "e19_tasks_sha256": bundle["schedule"]["tasks_sha256"],
            "e18_schedule_sha256": e18["schedule"]["sha256"],
            "e18_tasks_sha256": e18["schedule"]["tasks_sha256"],
            "identical": bundle["schedule"]["tasks_sha256"] == e18["schedule"]["tasks_sha256"],
        },
        "note": "warm-up vs measurement disjointness is exact; residual overlap "
                "with E18 is a property of the frozen fixture corpus and is "
                "reported, not designed away",
    }


def _injected_resident_bytes(path: Path) -> dict:
    """Corroborating measurement: the bytes actually appended to pi's system
    prompt, and the part of them that is injected artifact content."""
    text = path.read_text(encoding="utf-8")
    total = len(text.encode("utf-8"))
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
    return {"system_append_bytes": total,
            "injected_artifact_blocks": blocks,
            "injected_artifact_bytes": body}


def phase_steady(leg: str, warmup_count: int) -> dict:
    """§3 steady-state verification.  The judged quantity is the resident
    injection token count (an INPUT state quantity), never a cost ratio."""
    bundle = build_bundle()
    rows = []
    for index in warmup_sessions(warmup_count):
        row = resident_of(bundle, index)
        sid = row["session_id"]
        unit_file = leg_root(leg) / "units" / f"karc-full--{sid}.json"
        executed = None
        if unit_file.exists():
            unit = json.loads(unit_file.read_text(encoding="utf-8"))
            executed = {"turns_executed": unit["turns_executed"],
                        "pi_session_id_sha256": unit["pi_session_id_sha256"]}
        append = (unit_dir(leg, "karc-full", sid) / "context"
                  / "system-append.md")
        row["executed"] = executed
        row["resident_tokens_per_turn_mean"] = float(row["resident_tokens"])
        if append.exists():
            row.update(_injected_resident_bytes(append))
        rows.append(row)

    last, prev = rows[-1], rows[-2]
    rel = (abs(last["resident_tokens"] - prev["resident_tokens"])
           / prev["resident_tokens"]) if prev["resident_tokens"] else None
    util = last["budget_utilization"]
    within = rel is not None and rel <= STEADY_REL_TOLERANCE
    utilized = util >= STEADY_MIN_UTILIZATION
    reached = bool(within and utilized)
    return {
        "leg": leg,
        "warmup_sessions_run": warmup_count,
        "per_session": rows,
        "criterion": {
            "definition": "last warm-up session's mean resident injection "
                          "tokens per turn within 5% of the previous session, "
                          "and >= 95% utilization of the 1,756-token budget",
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
        "H10_fired": (not reached) and warmup_count >= WARMUP_MAX,
        "all_turns_executed": all(
            (r["executed"] or {}).get("turns_executed") == SESSION_LENGTH
            for r in rows),
    }


# ------------------------------------------------------------------- wave

def _ledger_add(usd: float) -> float:
    """Cell-global H6' ledger.  E19 runs four waves, so a per-wave guard would
    not see the cell total."""
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    book = (json.loads(LEDGER.read_text(encoding="utf-8"))
            if LEDGER.exists() else {"usd": 0.0})
    book["usd"] = float(book.get("usd", 0.0)) + usd
    LEDGER.write_text(json.dumps(book, indent=2) + "\n", encoding="utf-8")
    return book["usd"]


def phase_wave(leg: str, *, stage: str, warmup_count: int, parallel: int,
               force: bool, only: list[str] | None,
               unit_retries: int = 2) -> dict:
    if stage == "warmup":
        # karc arm only (preregistration §2: rag-bm25 carries no resident state)
        order = [("karc-full", idx) for idx in warmup_sessions(warmup_count)]
    elif stage == "measure":
        order = abba_order(measure_sessions(warmup_count))
    else:
        raise SystemExit(f"E19: unknown stage {stage!r}")
    if only is not None:
        order = [(arm, idx) for arm, idx in order
                 if f"{arm}--S{idx:02d}" in only]
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
            unit_usd = _standard_priced_usd(leg, result["rows"])
            spent["usd"] += unit_usd
            cell_usd = _ledger_add(unit_usd)
            if cell_usd > BUDGET_STOP_USD:
                aborted.append(f"H6' guard: cell standard-priced spend "
                               f"${cell_usd:.4f} exceeded ${BUDGET_STOP_USD}")
            say({"wave": f"{leg}/{stage}", "done": f"{arm}--{result['session_id']}",
                 "turns": result["turns_executed"],
                 "wave_standard_usd": round(spent["usd"], 6),
                 "cell_standard_usd": round(cell_usd, 6)})
        return result

    started = time.time()
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        results = list(pool.map(work, order))
    book = (json.loads(LEDGER.read_text(encoding="utf-8"))
            if LEDGER.exists() else {"usd": 0.0})
    return {
        "leg": leg, "stage": stage, "parallel": parallel,
        "warmup_sessions_run": warmup_count,
        "submission_order": [f"{arm}--S{idx:02d}" for arm, idx in order],
        "units": len(order),
        "wall_seconds": round(time.time() - started, 1),
        "aborted": aborted,
        "tokens_per_minute_target": PACER.tpm,
        "unit_attempts": {f"{r.get('arm')}--{r.get('session_id')}": r.get("attempts")
                          for r in results if "arm" in r},
        "failed": [r for r in results if "failed" in r],
        "wave_standard_priced_usd": round(spent["usd"], 6),
        "cell_standard_priced_usd_running": round(float(book["usd"]), 6),
        "unit_summaries": [{
            "arm": r.get("arm"), "session_id": r.get("session_id"),
            "turns_executed": r.get("turns_executed"),
            "three_paths": r.get("three_paths"),
        } for r in results if "arm" in r],
        "skipped": [r for r in results if "skipped" in r],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["mtime", "slices", "auth", "prepare",
                                          "registry", "steady", "unit", "wave"])
    parser.add_argument("--leg", choices=sorted(LEGS), default=None)
    parser.add_argument("--arm", choices=ARMS, default=None)
    parser.add_argument("--session", type=int, default=None)
    parser.add_argument("--stage", choices=["warmup", "measure"], default=None)
    parser.add_argument("--warmup-sessions", type=int, default=WARMUP_START,
                        help="how many warm-up sessions the cell ran "
                             "(fixes the measurement window)")
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

    global PACER, NAMESPACE, LEDGER
    PACER = TokenPacer(args.tpm)
    NAMESPACE = args.ns
    LEDGER = RUN / "spend-ledger.json"
    RAW.mkdir(parents=True, exist_ok=True)
    if args.phase == "mtime":
        result = phase_mtime()
        out = args.out or "mtime-before.json"
    elif args.phase == "slices":
        result = phase_slices(args.warmup_sessions)
        out = args.out or "slices.json"
    elif args.phase == "auth":
        result = phase_auth(args.leg)
        out = args.out or f"auth-{args.leg}.json"
    elif args.phase == "prepare":
        result = phase_prepare(args.leg)
        out = args.out or f"prepare-{args.leg}.json"
    elif args.phase == "registry":
        result = phase_registry(args.leg)
        out = args.out or f"registry-{args.leg}.json"
    elif args.phase == "steady":
        result = phase_steady(args.leg, args.warmup_sessions)
        out = args.out or f"steady-{args.leg}.json"
    elif args.phase == "unit":
        result = phase_unit(args.leg, args.arm, args.session, force=args.force)
        out = args.out or f"unit-{args.leg}-{args.arm}-S{args.session:02d}.json"
    else:
        if args.stage is None:
            raise SystemExit("E19: wave requires --stage warmup|measure")
        only = args.only.split(",") if args.only else None
        result = phase_wave(args.leg, stage=args.stage,
                            warmup_count=args.warmup_sessions,
                            parallel=args.parallel,
                            force=args.force, only=only,
                            unit_retries=args.unit_retries)
        out = args.out or f"wave-{args.leg}-{args.stage}.json"
    result.setdefault("_meta", {})
    result["_meta"].update({"phase": args.phase, "pi_version": PI_VERSION,
                            "cell": "E19-STEADY-STATE",
                            "namespace": NAMESPACE,
                            "generated_at_utc": time.strftime(
                                "%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    (RAW / out).write_text(json.dumps(result, indent=2, sort_keys=True,
                                      ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(f"wrote {RAW / out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
