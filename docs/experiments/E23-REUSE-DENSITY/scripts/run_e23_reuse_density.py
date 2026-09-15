"""E23-REUSE-DENSITY — does the component signature survive a change of reuse density?

Derived from `docs/experiments/E19-STEADY-STATE/scripts/run_e19_steady_state.py`
(statistics, closure asserts, steady-state judgment and the E18 incident guards
are inherited verbatim; only the manipulated axis and the isolation root differ).

E19 measured the OpenAI leg at one workload point: reuse density rho = 0.50.
The adversarial review's structural objection is that the workload axis is a
single point, so this cell moves it BOTH ways inside the non-degenerate range:

    rho = 0.25   12 paired sessions x 8 turns x 2 arms = 192 turns
    rho = 0.75   12 paired sessions x 8 turns x 2 arms = 192 turns

Everything else is E19 leg B: pi 0.84.3, provider openai / gpt-5.6-luna, metered
route, arms `karc-full` (MCP via bridge) and `rag-bm25`, resident budget 5%,
schedule seed 4352 (E19's own seed, so rho is the ONLY changed build input),
steady-state window, rate card, statistics (seed 1313, 10,000 resamples, common
random numbers).

WHAT "rho" IS ON THIS HARNESS -- measured, not assumed
------------------------------------------------------
The preregistration §2 argues from `e4_fixture._schedule` / `RHO_GRID`.  That is
not the code path this harness takes, and the difference is recorded rather than
papered over:

  * `fixture/e4-v2` is built by `e4_fixture._schedule_v2` (`build_fixture_v2`),
    which takes NO rho argument; its `cell.rho = 0.5` is a label.
  * `e5_killgate._source_manifest` keeps that corpus and DISCARDS its task
    schedule, then `build_reuse_schedule(..., reuse_factor=r)` builds the
    session schedule.  `e5_killgate` states this in the manifest it emits:
    `engine_source_labels_non_control = {rho, sigma, "replay_cell compatibility
    only; E5 controls r, not rho/sigma"}`.
  * So the operative reuse-density knob is `reuse_factor` in
    `R_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)`.  The two preregistered values are in
    that grid and in `RHO_GRID`, so the chosen POINTS are unaffected.
  * The preregistration's substantive claim survives the translation: r = 0.0 in
    the operative path is degenerate in exactly the feared way (no reuse, no
    supersession, sigma undefined, every task `new-gold`, zero hot-supersession
    traps).  `phase_build` measures that as an explicit negative control.

Because `build_reuse_schedule` derives sigma from the built schedule
(`len(superseded_artifacts) / len(reused_artifacts)`), sigma is NOT an
independent knob here.  The replay engine's sigma label stays frozen at 0.3, but
the schedule's measured sigma moves with r.  Both numbers are recorded.

Pre-model gates specific to this cell (preregistration §7):
    H12  the built schedule's `gold_containment_rate` must be 1.0 for both rho
    H13  the built schedule must NOT be the degenerate zero-reuse branch

Instrumentation contract, inherited from E17/E17B/E18/E19 and all of it measured:
    gross = input + cacheRead + cacheWrite   (pi's `usage.input` is fresh-only)
    closure per call: totalTokens == input + output + cacheRead + cacheWrite
    stop reason in two layers (pi normalized + provider raw)
    MCP round trip counted on three independent paths, audit scoped per turn
    `--tools` names read back out of pi's own registry (typos are not validated)
    H9 cold-start assert as a first-class condition
    H10 namespace changes BOTH the workdir path and the pi session id, per rho

Isolation: every path this cell writes lives under `tmp/e23/`.  `tmp/e17`,
`tmp/e17b`, `tmp/e18`, `tmp/e19` and `tmp/e21` are never touched.

Privacy (R-9): raw rows carry counts, ids, sha256 and usage numbers only.

Phases:
    mtime     snapshot the user's ~/.pi paths (no pi, no model)
    build     H12/H13 gate, reuse profile, r=0 negative control (no pi, no model)
    steady    minimum warm-up window satisfying the E19 criterion (no model)
    auth      model-free `pi auth check` inside the isolated home
    prepare   materialize 48 units for one rho (no pi, no model)
    registry  model-free registry dump through pi --mode rpc
    unit      run one (rho, arm, session) unit of 8 turns
    wave      run the measurement window of one rho, ABBA order, parallel
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import statistics
import subprocess
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve()
CELL_DIR = HERE.parents[1]                 # docs/experiments/E23-REUSE-DENSITY
REPO = HERE.parents[4]                     # repository root
if not (REPO / "pyproject.toml").exists() or not (REPO / "src" / "karc").is_dir():
    raise SystemExit(f"E23: repository root misresolved as {REPO} (E15 §8.4 off-by-one guard)")

sys.path.insert(0, str(REPO / "src"))

FIXTURE = REPO / "fixture" / "e4-v2"
BRIDGE = REPO / "integrations" / "pi" / "karc-mcp-bridge" / "index.ts"
PROBE_EXT = CELL_DIR / "scripts" / "tool-registry-probe.ts"
RAW = CELL_DIR / "raw"
RUN = REPO / "tmp" / "e23"                 # gitignored; NOT e17/e17b/e18/e19/e21
PI_BIN = REPO / "tmp" / "vendor" / "piprobe" / "node_modules" / ".bin" / "pi"

PI_VERSION = "0.84.3"
PROVIDER = "openai"
MODEL = "gpt-5.6-luna"
RETENTION = None                           # E17B appendix: OpenAI leg default
KEY_ENV = "KARC_OPENAI_KEY"                # never the provider's own name
PROVIDER_KEY_ENV = "OPENAI_API_KEY"

# E19's schedule seed, reused unchanged so that rho is the ONLY altered build
# input.  No seed search is performed by this cell: a second free parameter
# would make the axis non-identifiable.
SCHEDULE_SEED = 4352
SESSION_COUNT = 24
SESSION_LENGTH = 8
WARMUP_START = 8
WARMUP_MAX = 12
MEASURE_SESSIONS = 12
RESIDENT_BUDGET_TOKENS = 1756          # manifest cell budget, 5% of corpus
BUDGET_PCT = 5
STEADY_REL_TOLERANCE = 0.05            # last vs previous warm-up session
STEADY_MIN_UTILIZATION = 0.95          # of RESIDENT_BUDGET_TOKENS
ARMS = ("karc-full", "rag-bm25")
TOOLS = ("karc_search", "karc_get")
THINKING = "off"

# The manipulated axis.  Tag -> reuse factor.  Both values are in
# e5_killgate.R_GRID and in e4_fixture.RHO_GRID, and both are non-zero.
RHOS: dict[str, float] = {"025": 0.25, "075": 0.75}
CONTROL_RHO = 0.50                     # E19 leg B's frozen value, for contrast
DEGENERATE_RHO = 0.0                   # H13 negative control, never executed

# Schedule identities, computed model-free from the committed builders BEFORE
# any model call and gated on here.  A mismatch means the builder or the frozen
# corpus moved, which invalidates the comparison with E19 leg B.
SCHEDULE_IDENTITY = {
    "025": {
        "snapshot_sha256": "5b91ca6eb26779d87b5d4183221e59c4c6b812e28fce0e0ddc93555e82246dfe",
        "schedule_sha256": "d5737bc7449e1d9595930f061fe6cf0501a39a860173c249ffd6fecd2158ebdf",
        "tasks_sha256": "6bdb731374207db63fd1eeec64de73ce75689c837aec6c837ee495da1e1d8e43",
        "manifest_sha256": "00971ff9992b837d2748ddcb27f5c53618490b5dfa38d69871b6f2b5b2f35077",
    },
    "075": {
        "snapshot_sha256": "0b63531feb12c6b98815754f6cce0bfbec747d482b0071bc7e8bbbe5ea7ac435",
        "schedule_sha256": "22feae85280e68365e2655df2e47ea5a86c90918dce3385dde823fa8548269e0",
        "tasks_sha256": "6ea879c480fa9e8172efc0aabd7addb7df9b1c7ff9a71b69721d2bc41480b258",
        "manifest_sha256": "1b3d2138ac2c265a31fb8d3bf856a375f72eca10d4ee1acb28705eb871cacf44",
    },
}
# E19 leg B's committed identity at rho = 0.50.  Re-derived here as a
# faithfulness check on this cell's re-implementation of `prepare_bundle`:
# the parameterized builder must reproduce E19's hashes exactly at r = 0.50.
E19_IDENTITY_AT_050 = {
    "snapshot_sha256": "f6edbb9e33c63c935aade45961ec8995f121e50e3854713963508925df1c2a73",
    "schedule_sha256": "acdec9a853ae051225ac111e4c336bae743fa52fad5264bcb1837f42693f225b",
    "tasks_sha256": "73706b12ea0cba1dcb46963a34da83d36201c942ac2958289ef0129e8d460d74",
}

# OpenAI cache minimum prefix (docs/analysis/openai-luna-rate-card.md §5.1).
OPENAI_CACHE_MIN_PREFIX = 1024

# pi's own catalog rates, USD per million tokens (E17B §6).  Control column only.
PI_CATALOG_RATES = {
    "openai": {"input": 0.20, "output": 1.20, "read": 0.02,
               "write_5m": 0.25, "write_1h": 0.25},
}

# H6 ceiling: $3 for the whole cell (preregistration §7; expected $0.3).
BUDGET_CAP_USD = 3.0
BUDGET_STOP_USD = 2.4
LEDGER = None                              # set in main(); RUN/"spend-ledger.json"

USER_PI_PATHS = (
    Path.home() / ".pi",
    Path.home() / ".pi" / "agent",
    Path.home() / ".pi" / "agent" / "auth.json",
    Path.home() / ".pi" / "agent" / "models-store.json",
)
# Isolation roots of every earlier pi/Codex cell.  Touching any of them is a
# hygiene failure, so their mtimes are snapshotted alongside the user's ~/.pi.
SIBLING_CELL_ROOTS = tuple(
    REPO / "tmp" / name for name in ("e17", "e17b", "e18", "e19", "e21"))

_PRINT_LOCK = threading.Lock()


class TokenPacer:
    """Provider token-rate throttle, verbatim from E19.

    The OpenAI leg shares one org-wide tokens-per-minute budget.  pi's own
    auto-retry gives up after 3 attempts and then emits an assistant message
    with stopReason "error" and an all-zero usage block, which is exactly the
    accounting distortion H2 exists to reject (E18 leg B attempt 1).  So the
    rate is governed here instead of being discovered by the provider.
    """

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
    """A turn came back with no model answer (transport error or refusal)."""


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def say(payload: dict) -> None:
    with _PRINT_LOCK:
        print(json.dumps(payload, ensure_ascii=False), flush=True)


def session_id(index: int) -> str:
    return f"S{index:02d}"


def warmup_sessions(count: int) -> list[int]:
    if not WARMUP_START <= count <= WARMUP_MAX:
        raise SystemExit(f"E23: warm-up must be {WARMUP_START}..{WARMUP_MAX} sessions")
    return list(range(1, count + 1))


def measure_sessions(warmup_count: int) -> list[int]:
    first = warmup_count + 1
    last = first + MEASURE_SESSIONS - 1
    if last > SESSION_COUNT:
        raise SystemExit("E23: measurement window overruns the built schedule")
    return list(range(first, last + 1))


def abba_order(sessions: list[int]) -> list[tuple[str, int]]:
    """E13 §5.0's ABBA submission order, so neither arm is systematically
    earlier in wall time."""
    order: list[tuple[str, int]] = []
    for offset, index in enumerate(sessions):
        if offset % 2 == 0:
            order += [("karc-full", index), ("rag-bm25", index)]
        else:
            order += [("rag-bm25", index), ("karc-full", index)]
    return order


# Execution namespace (H10).  A discarded execution leaves a live server-side
# prompt cache behind: OpenAI keeps a written prefix reusable for 30 minutes and
# pi routes it with prompt_cache_key = session id, so a re-run that reuses the
# same workdir paths and session ids is served that cache and reports
# `cacheWrite == 0` on calls a clean run would be charged creation for.  Both
# levers move, and they also differ BETWEEN the two rho values.
NAMESPACE = ""


def rho_root(tag: str) -> Path:
    return RUN / f"rho{tag}{NAMESPACE}"


def pi_session_name(tag: str, arm: str, sid: str) -> str:
    return f"e23r{tag}{NAMESPACE}-{arm}-{sid}"


def unit_dir(tag: str, arm: str, sid: str) -> Path:
    return rho_root(tag) / "units" / f"{arm}--{sid}"


def pi_env(tag: str, extra: dict[str, str] | None = None) -> dict[str, str]:
    """Isolated pi environment.  The user's ~/.pi is never read or written."""
    env = {
        "HOME": os.environ["HOME"],
        "PATH": os.environ["PATH"],
        "LANG": os.environ.get("LANG", "en_US.UTF-8"),
        "PI_CODING_AGENT_DIR": str(rho_root(tag) / "pi-home"),
        "PI_CODING_AGENT_SESSION_DIR": str(rho_root(tag) / "sessions"),
        "PI_OFFLINE": "1",
        "PI_SKIP_VERSION_CHECK": "1",
        "PI_TELEMETRY": "0",
    }
    if extra:
        env.update(extra)
    return env


# ------------------------------------------------------------------ bundle

def _prepare_bundle(reuse_factor: float, *, seed: int = SCHEDULE_SEED,
                    session_count: int = SESSION_COUNT,
                    session_length: int = SESSION_LENGTH) -> dict:
    """`e5_cache_canary.prepare_bundle` with `reuse_factor` opened up.

    The library function hard-codes `REUSE_FACTOR = 0.5`, and neither
    `e5_cache_canary` nor `e5_killgate` may be edited (every committed cell
    reproduces through them).  So the body is reproduced here with the one
    parameter this cell manipulates.  Faithfulness is not assumed: at r = 0.50
    the output must hash-match E19's committed identity, which `phase_build`
    checks (`E19_IDENTITY_AT_050`).
    """
    from karc.bench import e5_cache_canary as g1
    from karc.bench.e3_retrieval import bm25_plans, chunk_fixture
    from karc.bench.e4_replay import load_confirmed_config, replay_cell
    from karc.bench.e5_killgate import (
        ENGINE_RHO_LABEL, ENGINE_SIGMA_LABEL, _manifest_at_budget,
        build_reuse_schedule,
    )

    schedule, schedule_manifest, tasks = build_reuse_schedule(
        REPO, schedule_seed=seed, session_length=session_length,
        reuse_factor=reuse_factor, session_count=session_count,
        allow_unregistered_seed=True,
    )
    manifest = _manifest_at_budget(schedule_manifest, BUDGET_PCT)
    replay_summary, replay_rows = replay_cell(
        rho=ENGINE_RHO_LABEL, sigma=ENGINE_SIGMA_LABEL, budget_pct=BUDGET_PCT,
        confirmed_config=load_confirmed_config(REPO),
        fixture_manifest=manifest, fixture_tasks=tasks,
    )
    chunks = chunk_fixture(FIXTURE, manifest)
    plans = bm25_plans(chunks, tasks, manifest,
                       int(manifest["cell"]["budget_tokens"]))
    replay_by_task = {row["task_id"]: row for row in replay_rows}
    by_session: dict[str, list[dict]] = defaultdict(list)
    for task in tasks:
        by_session[task["session_id"]].append(task)
    sessions: list[dict] = []
    for sid, session_tasks in by_session.items():
        session_tasks.sort(key=lambda row: int(row["session_task"]))
        first = replay_by_task[session_tasks[0]["task_id"]]
        initial = list(first["arms"]["karc"]["working_set_versions"])
        rows = []
        for task in session_tasks:
            replay = replay_by_task[task["task_id"]]
            resident = list(replay["arms"]["karc"]["working_set_versions"])
            required = set(task["required_versions"])
            plan = plans[task["task_id"]]
            if not required <= set(plan["artifact_ids"]):
                raise AssertionError("BM25 budget-parity plan omitted gold evidence")
            rows.append({
                "task": task,
                "policy_resident_versions": resident,
                "policy_resident_hit": required <= set(resident),
                "initial_prefix_hit": required <= set(initial),
                "rag_plan": plan,
            })
        sessions.append({
            "schedule_seed": seed,
            "session_id": sid,
            "initial_resident_versions": initial,
            "tasks": rows,
        })
    if len(sessions) != session_count:
        raise AssertionError("session grouping changed cardinality")
    return {
        "schema": g1.SCHEMA,
        "schedule": schedule,
        "manifest": manifest,
        "tasks": tasks,
        "sessions": sessions,
        "replay_summary": replay_summary,
        "plans": plans,
        "retrieval_audit": {
            "gold_containment_rate": sum(
                set(task["required_versions"])
                <= set(plans[task["task_id"]]["artifact_ids"])
                for task in tasks) / len(tasks),
            "mean_budget_utilization": statistics.mean(
                float(plans[task["task_id"]]["budget_utilization"])
                for task in tasks),
            "rank_order_preserved": True,
            "plan_tokens_mean": statistics.mean(
                int(plans[task["task_id"]]["tokens"]) for task in tasks),
        },
    }


def build_bundle(tag: str) -> dict:
    """Rebuild one rho's schedule and gate on its committed identity."""
    from karc.bench import e5_cache_canary as g1

    if tag not in RHOS:
        raise SystemExit(f"E23: unknown rho tag {tag!r}")
    bundle = _prepare_bundle(RHOS[tag])
    snapshot = g1.bundle_snapshot(bundle)
    want = SCHEDULE_IDENTITY[tag]
    if snapshot["sha256"] != want["snapshot_sha256"]:
        raise SystemExit(
            f"E23: rho={RHOS[tag]} schedule does not match the committed "
            f"identity ({snapshot['sha256']} != {want['snapshot_sha256']})")
    if bundle["schedule"]["sha256"] != want["schedule_sha256"]:
        raise SystemExit(f"E23: rho={RHOS[tag]} schedule sha256 mismatch")
    if bundle["schedule"]["tasks_sha256"] != want["tasks_sha256"]:
        raise SystemExit(f"E23: rho={RHOS[tag]} tasks sha256 mismatch")
    if bundle["manifest"]["manifest_sha256"] != want["manifest_sha256"]:
        raise SystemExit(f"E23: rho={RHOS[tag]} manifest sha256 mismatch")
    if int(bundle["manifest"]["cell"]["budget_tokens"]) != RESIDENT_BUDGET_TOKENS:
        raise SystemExit("E23: resident budget is not the preregistered 1,756")
    if float(bundle["schedule"]["reuse_factor_measured"]) != RHOS[tag]:
        raise SystemExit("E23: measured reuse factor differs from the request")
    # H12 gate, re-measured on the object actually used.
    rate = float(bundle["retrieval_audit"]["gold_containment_rate"])
    if rate != 1.0:
        raise SystemExit(f"E23: H12 fired for rho={RHOS[tag]} "
                         f"(gold_containment_rate={rate})")
    # H13 gate, re-measured on the object actually used.
    structure = bundle["manifest"]["structure"]
    if int(structure["reused_occurrences"]) <= 0:
        raise SystemExit(f"E23: H13 fired for rho={RHOS[tag]} (zero reuse)")
    if int(structure["supersession_events"]) <= 0:
        raise SystemExit(f"E23: H13 fired for rho={RHOS[tag]} (no supersession)")
    if not bool(structure["sigma_defined"]):
        raise SystemExit(f"E23: H13 fired for rho={RHOS[tag]} (sigma undefined)")
    bundle["_snapshot"] = snapshot
    return bundle


def resident_of(bundle: dict, index: int) -> dict:
    """The resident set injected for one session, as an INPUT state quantity."""
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


def reuse_profile(bundle: dict) -> dict:
    """Per-session reuse counts and the schedule-wide reuse rate.

    E19 §5 disclosed that it fixed its window without inspecting the reuse mix.
    This cell measures the mix for every session before any model call, since
    reuse density is the manipulated axis.
    """
    per_session = {}
    for index, session in enumerate(bundle["sessions"], 1):
        tasks = session["tasks"]
        reuse = sum(1 for item in tasks if item["task"]["e5_resident_reuse"])
        per_session[session["session_id"]] = {
            "session_index": index,
            "tasks": len(tasks),
            "reuse_tasks": reuse,
            "eligible_followups": len(tasks) - 1,
            "reuse_rate_within_session": reuse / (len(tasks) - 1),
            "policy_resident_hits": sum(1 for item in tasks
                                        if item["policy_resident_hit"]),
            "hot_supersession_traps": sum(
                1 for item in tasks if item["task"]["forbidden_versions"]),
        }
    schedule = bundle["schedule"]
    structures: dict[str, int] = {}
    for task in bundle["tasks"]:
        structures[task["structure"]] = structures.get(task["structure"], 0) + 1
    return {
        "reuse_factor_requested": schedule["reuse_factor_requested"],
        "reuse_factor_measured": schedule["reuse_factor_measured"],
        "reuse_count": schedule["reuse_count"],
        "reuse_denominator_eligible_followups":
            schedule["reuse_denominator_eligible_followups"],
        "rho_measured_all_tasks": schedule["rho_measured_all_tasks"],
        "definition": schedule["definition"],
        "task_structure_histogram": structures,
        "per_session": per_session,
    }


def structure_facts(bundle: dict) -> dict:
    manifest_structure = bundle["manifest"]["structure"]
    keys = ("tasks", "sessions", "session_length", "reuse_opportunities",
            "reused_occurrences", "reused_doc_ids", "r_requested", "r_measured",
            "rho_measured_all_tasks", "supersession_events",
            "sigma_denominator_reused_doc_ids", "sigma_measured",
            "sigma_defined", "corrected_events", "validated_outcome_events")
    return {key: manifest_structure[key] for key in keys}


# ------------------------------------------------------------------ mtime

def _stat(path: Path) -> dict:
    if not path.exists():
        return {"exists": False}
    return {"exists": True, "mtime_ns": path.stat().st_mtime_ns,
            "size": path.stat().st_size if path.is_file() else None}


def phase_mtime() -> dict:
    return {
        "user_pi_paths": {str(p): _stat(p) for p in USER_PI_PATHS},
        "sibling_cell_roots": {str(p): _stat(p) for p in SIBLING_CELL_ROOTS},
    }


# ------------------------------------------------------------------- build

def phase_build() -> dict:
    """Model-free H12 / H13 gate plus the full reuse profile.

    Also re-derives r = 0.50 (E19's frozen value) to prove this cell's
    parameterized builder is faithful, and r = 0.0 to show the degenerate branch
    the preregistration §2 refuses to use.
    """
    from karc.bench import e5_cache_canary as g1
    from karc.bench import e5_killgate as kg
    from karc.bench import e4_fixture as f4

    out: dict = {
        "operative_axis": {
            "what": "e5_killgate.build_reuse_schedule(reuse_factor=r)",
            "grid": list(kg.R_GRID),
            "library_default_frozen_at": g1.REUSE_FACTOR,
            "e4_fixture_RHO_GRID": list(f4.RHO_GRID),
            "e4_fixture_schedule_branch_is_in_path": False,
            "why": ("fixture/e4-v2 is built by e4_fixture._schedule_v2, which "
                    "takes no rho argument; e5_killgate._source_manifest keeps "
                    "that corpus and discards its task schedule, then builds "
                    "the session schedule from reuse_factor. e5_killgate emits "
                    "engine_source_labels_non_control saying rho/sigma are "
                    "replay_cell compatibility labels and 'E5 controls r'."),
        },
        "schedule_seed": SCHEDULE_SEED,
        "seed_choice": ("E19 leg B's own seed 4352, reused unchanged so that "
                        "reuse density is the only altered build input"),
        "rho": {},
    }

    for tag, value in RHOS.items():
        bundle = build_bundle(tag)
        profile = reuse_profile(bundle)
        out["rho"][tag] = {
            "reuse_factor": value,
            "identity": {
                "snapshot_sha256": bundle["_snapshot"]["sha256"],
                "schedule_sha256": bundle["schedule"]["sha256"],
                "tasks_sha256": bundle["schedule"]["tasks_sha256"],
                "manifest_sha256": bundle["manifest"]["manifest_sha256"],
                "matches_committed": True,
            },
            "H12_gold_containment": {
                "gold_containment_rate":
                    bundle["retrieval_audit"]["gold_containment_rate"],
                "is_one": bundle["retrieval_audit"]["gold_containment_rate"] == 1.0,
                "fired": bundle["retrieval_audit"]["gold_containment_rate"] != 1.0,
                "mean_budget_utilization":
                    bundle["retrieval_audit"]["mean_budget_utilization"],
                "plan_tokens_mean": bundle["retrieval_audit"]["plan_tokens_mean"],
            },
            "H13_non_degenerate_branch": {
                "reused_occurrences": bundle["manifest"]["structure"]["reused_occurrences"],
                "reuse_gold_tasks": profile["task_structure_histogram"].get("reuse-gold", 0),
                "new_gold_tasks": profile["task_structure_histogram"].get("new-gold", 0),
                "supersession_events": bundle["manifest"]["structure"]["supersession_events"],
                "hot_supersession_trap_tasks": sum(
                    1 for task in bundle["tasks"] if task["forbidden_versions"]),
                "corrected_events": bundle["manifest"]["structure"]["corrected_events"],
                "validated_outcome_events":
                    bundle["manifest"]["structure"]["validated_outcome_events"],
                "sigma_defined": bundle["manifest"]["structure"]["sigma_defined"],
                "sigma_measured": bundle["manifest"]["structure"]["sigma_measured"],
                "degenerate_zero_reuse_branch": False,
                "fired": False,
            },
            "structure": structure_facts(bundle),
            "reuse_profile": profile,
            "resident_ramp": [resident_of(bundle, i)
                              for i in range(1, SESSION_COUNT + 1)],
            "cell": bundle["manifest"]["cell"],
            "fixture_manifest_sha256": bundle["manifest"]["manifest_sha256"],
        }

    # Faithfulness check: r = 0.50 must reproduce E19's committed identity.
    control = _prepare_bundle(CONTROL_RHO)
    control_snapshot = g1.bundle_snapshot(control)
    out["control_rho_050_reproduces_e19"] = {
        "reuse_factor": CONTROL_RHO,
        "snapshot_sha256": control_snapshot["sha256"],
        "schedule_sha256": control["schedule"]["sha256"],
        "tasks_sha256": control["schedule"]["tasks_sha256"],
        "matches_e19_snapshot":
            control_snapshot["sha256"] == E19_IDENTITY_AT_050["snapshot_sha256"],
        "matches_e19_schedule":
            control["schedule"]["sha256"] == E19_IDENTITY_AT_050["schedule_sha256"],
        "matches_e19_tasks":
            control["schedule"]["tasks_sha256"] == E19_IDENTITY_AT_050["tasks_sha256"],
        "structure": structure_facts(control),
        "reuse_profile": reuse_profile(control),
        "resident_ramp": [resident_of(control, i)
                          for i in range(1, SESSION_COUNT + 1)],
        "why": ("proves the parameterized re-implementation of prepare_bundle "
                "is byte-identical to the library path at the frozen value, so "
                "the rho = 0.25 / 0.75 builds differ only in reuse_factor"),
    }
    if not (out["control_rho_050_reproduces_e19"]["matches_e19_snapshot"]
            and out["control_rho_050_reproduces_e19"]["matches_e19_schedule"]
            and out["control_rho_050_reproduces_e19"]["matches_e19_tasks"]):
        raise SystemExit("E23: builder does not reproduce E19 leg B at r = 0.50")

    # H13 negative control: the branch the preregistration §2 refuses to use.
    from karc.bench.e5_killgate import build_reuse_schedule
    _, degenerate_manifest, degenerate_tasks = build_reuse_schedule(
        REPO, schedule_seed=SCHEDULE_SEED, session_length=SESSION_LENGTH,
        reuse_factor=DEGENERATE_RHO, session_count=SESSION_COUNT,
        allow_unregistered_seed=True)
    degenerate_structures: dict[str, int] = {}
    for task in degenerate_tasks:
        degenerate_structures[task["structure"]] = (
            degenerate_structures.get(task["structure"], 0) + 1)
    out["degenerate_control_rho_000"] = {
        "reuse_factor": DEGENERATE_RHO,
        "executed": False,
        "structure": {key: degenerate_manifest["structure"][key] for key in (
            "reused_occurrences", "reused_doc_ids", "supersession_events",
            "sigma_measured", "sigma_defined", "r_measured")},
        "task_structure_histogram": degenerate_structures,
        "hot_supersession_trap_tasks": sum(
            1 for task in degenerate_tasks if task["forbidden_versions"]),
        "why_excluded": ("preregistration §2: zero reuse also removes the "
                         "supersession dynamics, so a sign flip there could not "
                         "be attributed to reuse density. Measured here to show "
                         "the degeneracy is real on THIS harness, not only in "
                         "e4_fixture._schedule as §2 assumed."),
    }

    # Slice overlap between the two rho builds and against E19's r = 0.50, so a
    # shared-corpus artifact recurrence is reported rather than assumed away.
    tags = sorted(RHOS)
    per_tag_versions = {}
    for tag in tags:
        bundle = build_bundle(tag)
        per_tag_versions[tag] = slice_versions(
            bundle, list(range(1, SESSION_COUNT + 1)))
    control_versions = slice_versions(control, list(range(1, SESSION_COUNT + 1)))
    out["slice_overlap"] = {
        "distinct_required_versions": {
            **{tag: len(v) for tag, v in per_tag_versions.items()},
            "control_050": len(control_versions),
        },
        "overlap_025_vs_075": len(per_tag_versions["025"] & per_tag_versions["075"]),
        "overlap_025_vs_050": len(per_tag_versions["025"] & control_versions),
        "overlap_075_vs_050": len(per_tag_versions["075"] & control_versions),
        "note": ("the fixture corpus is frozen and shared by every cell, so "
                 "artifact recurrence across rho builds is structural and is "
                 "reported, not designed away (E19 §6.1)"),
    }
    return out


# ------------------------------------------------------------------ steady

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


def _steady_at(bundle: dict, tag: str, warmup_count: int) -> dict:
    rows = []
    for index in warmup_sessions(warmup_count):
        row = resident_of(bundle, index)
        sid = row["session_id"]
        append = unit_dir(tag, "karc-full", sid) / "context" / "system-append.md"
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
    return {
        "warmup_sessions": warmup_count,
        "per_session": rows,
        "last_session": last["session_id"],
        "previous_session": prev["session_id"],
        "resident_tokens_last": last["resident_tokens"],
        "resident_tokens_previous": prev["resident_tokens"],
        "relative_change": rel,
        "relative_change_within_tolerance": within,
        "budget_utilization": util,
        "utilization_at_least_95pct": utilized,
        "steady_state_reached": bool(within and utilized),
    }


def phase_steady(tag: str) -> dict:
    """E19 §3's criterion, applied per rho: the smallest warm-up window in
    8..12 whose last session's resident injection tokens are within 5% of the
    previous session's AND utilize >= 95% of the 1,756-token budget.

    The judged quantity is an INPUT state quantity, never a cost ratio.  The
    ramp is a deterministic function of the schedule and the model-free policy
    replay, so this is decided before any model call.
    """
    bundle = build_bundle(tag)
    attempts = [_steady_at(bundle, tag, count)
                for count in range(WARMUP_START, WARMUP_MAX + 1)]
    chosen = next((a for a in attempts if a["steady_state_reached"]), None)
    reached = chosen is not None
    warmup_count = chosen["warmup_sessions"] if reached else WARMUP_MAX
    window = measure_sessions(warmup_count) if reached else []
    profile = reuse_profile(bundle)
    window_rows = [resident_of(bundle, i) for i in window]
    return {
        "rho_tag": tag,
        "reuse_factor": RHOS[tag],
        "criterion": {
            "definition": ("last warm-up session's resident injection tokens "
                           "within 5% of the previous session, and >= 95% "
                           "utilization of the 1,756-token budget"),
            "search_range": [WARMUP_START, WARMUP_MAX],
            "quantity": "resident injection tokens per turn (input state)",
        },
        "attempts": attempts,
        "steady_state_reached": reached,
        "warmup_sessions_selected": warmup_count,
        "measurement_window": [session_id(i) for i in window],
        "measurement_window_resident": window_rows,
        "measurement_window_utilization_min": (
            min(r["budget_utilization"] for r in window_rows) if window_rows else None),
        "measurement_window_utilization_max": (
            max(r["budget_utilization"] for r in window_rows) if window_rows else None),
        "measurement_window_reuse_tasks": {
            session_id(i): profile["per_session"][session_id(i)]["reuse_tasks"]
            for i in window},
        "measurement_window_reuse_tasks_distinct": sorted({
            profile["per_session"][session_id(i)]["reuse_tasks"] for i in window}),
        "warmup_vs_measure_version_overlap": (
            len(slice_versions(bundle, warmup_sessions(warmup_count))
                & slice_versions(bundle, window)) if reached else None),
        "H11_fired": not reached,
    }


# ------------------------------------------------------------------- auth

def phase_auth(tag: str) -> dict:
    (rho_root(tag) / "pi-home").mkdir(parents=True, exist_ok=True)
    (rho_root(tag) / "sessions").mkdir(parents=True, exist_ok=True)
    results: dict = {"rho_tag": tag, "provider": PROVIDER}
    for with_key in (False, True):
        extra = {}
        if with_key:
            key = os.environ.get(KEY_ENV)
            if not key:
                results["with_env_key"] = {"skipped": f"{KEY_ENV} unset"}
                continue
            extra[PROVIDER_KEY_ENV] = key
        proc = subprocess.run(
            [str(PI_BIN), "auth", "check", "--provider", PROVIDER, "--json"],
            cwd=str(REPO), env=pi_env(tag, extra), text=True, capture_output=True,
            timeout=120)
        # `--credentials` is deliberately NOT passed: it prints the secret.
        results["with_env_key" if with_key else "no_env_key"] = {
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip()[:600],
            "stderr_tail": proc.stderr.strip().splitlines()[-3:],
        }
    auth_json = rho_root(tag) / "pi-home" / "auth.json"
    results["isolated_auth_json"] = {
        "exists": auth_json.exists(),
        "bytes": auth_json.stat().st_size if auth_json.exists() else None,
    }
    return results


# ---------------------------------------------------------------- prepare

def _materialize_unit(tag: str, arm: str, session: dict, manifest: dict) -> dict:
    """One (rho, arm, session) unit.  Verbatim E19, which is itself faithful to
    e5_cache_canary's materialize_session with two pi adaptations:

      1. Codex auto-loads AGENTS.md as a project doc; pi does not (this cell
         runs with --no-context-files).  The same AGENTS.md bytes are therefore
         passed explicitly with --append-system-prompt, and the file is moved
         OUT of the workdir so it cannot also be discovered.
      2. The MCP server is reached through the pi bridge's `.mcp.json`.
    """
    from karc.bench import materialize as mzt
    from karc.bench import mcp_index
    from karc.bench import e5_cache_canary as g1

    root = unit_dir(tag, arm, session["session_id"])
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
        raise SystemExit(f"E23: unknown arm {arm!r}")

    agents = work / "AGENTS.md"
    system_append = context / "system-append.md"
    shutil.move(str(agents), str(system_append))

    return {
        "rho_tag": tag, "arm": arm, "session_id": session["session_id"],
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


def phase_prepare(tag: str) -> dict:
    bundle = build_bundle(tag)
    manifest = bundle["manifest"]
    rho_root(tag).mkdir(parents=True, exist_ok=True)
    (rho_root(tag) / "pi-home").mkdir(parents=True, exist_ok=True)
    (rho_root(tag) / "sessions").mkdir(parents=True, exist_ok=True)
    (rho_root(tag) / "pi-home" / "settings.json").write_text(json.dumps({
        "defaultProjectTrust": "never",
        "quietStartup": True,
        "enableInstallTelemetry": False,
        "enableAnalytics": False,
        "compaction": {"enabled": False},
    }, indent=2), encoding="utf-8")

    units = []
    for arm in ARMS:
        for session in bundle["sessions"]:
            units.append(_materialize_unit(tag, arm, session, manifest))
            say({"prepared": units[-1]["arm"] + "/" + units[-1]["session_id"]})
    return {
        "rho_tag": tag,
        "reuse_factor": RHOS[tag],
        "pi_version": PI_VERSION,
        "provider": PROVIDER,
        "model": MODEL,
        "retention": RETENTION,
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
        "namespace": NAMESPACE,
        "pi_session_id_example": pi_session_name(tag, "karc-full", "S01"),
        "units": units,
    }


# --------------------------------------------------------------- registry

def _registry_probe(tag: str, *, arm: str, append: bool) -> dict:
    """One model-free pi start.  `rpc get_state` makes no model call, so this
    reads pi's own registry and system-prompt size for free."""
    root = unit_dir(tag, arm, "S01")
    work = root / "work"
    system_append = root / "context" / "system-append.md"
    if not work.is_dir():
        raise SystemExit("E23: run the prepare phase before registry")
    label = f"{arm}-{'append' if append else 'noappend'}"
    out = rho_root(tag) / f"registry-probe-{label}.json"
    if out.exists():
        out.unlink()
    audit = rho_root(tag) / f"registry-bridge-audit-{label}.jsonl"
    if audit.exists():
        audit.unlink()
    argv = [str(PI_BIN), "--mode", "rpc",
            "--provider", PROVIDER, "--model", MODEL,
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
    extra = {"KARC_E23_PROBE_OUT": str(out),
             "KARC_PI_BRIDGE_AUDIT": str(audit),
             "KARC_PI_MCP_CONFIG": str(work / ".mcp.json"),
             "KARC_PI_MCP_SERVER": "karc"}
    key = os.environ.get(KEY_ENV)
    if key:
        # No model call is made by rpc get_state; the key only lets pi resolve
        # the provider so the registry can be dumped at all.
        extra[PROVIDER_KEY_ENV] = key
    proc = subprocess.run(argv, cwd=str(work), env=pi_env(tag, extra),
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


def phase_registry(tag: str) -> dict:
    """Four model-free starts: each arm with and without the system-prompt
    append.  The with/without pair measures that --append-system-prompt really
    injected the AGENTS.md bytes, which is the pi substitute for Codex's
    project-doc auto-load."""
    probes = [_registry_probe(tag, arm=arm, append=append)
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
    return {"rho_tag": tag, "probes": probes, "system_prompt_injection": deltas}


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


def pi_modelled_cost(inp: int, out: int, read: int, write: int,
                     write1h: int) -> float:
    """Reverse the pi catalog rate card out of the four buckets."""
    rates = PI_CATALOG_RATES[PROVIDER]
    write5m = max(0, write - write1h)
    return (inp * rates["input"] + out * rates["output"] + read * rates["read"]
            + write5m * rates["write_5m"] + write1h * rates["write_1h"]) / 1e6


def closure_of(call: dict) -> dict:
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
        "pi_rate_card_modelled_usd": pi_modelled_cost(inp, out, read, write, write1h),
    }


def _sum(rows: list[dict], key: str) -> int:
    return sum(int(row.get(key) or 0) for row in rows)


def phase_unit(tag: str, arm: str, session_index: int, *,
               force: bool = False) -> dict:
    """Run 8 turns of one (rho, arm, session) unit through pi."""
    from karc.bench import e5_cache_canary as g1

    key = os.environ.get(KEY_ENV)
    if not key:
        raise SystemExit(f"E23: {KEY_ENV} is required to run turns")

    bundle = build_bundle(tag)
    manifest = bundle["manifest"]
    session = bundle["sessions"][session_index - 1]
    sid = session["session_id"]
    root = unit_dir(tag, arm, sid)
    work = root / "work"
    system_append = root / "context" / "system-append.md"

    out_path = rho_root(tag) / "units" / f"{arm}--{sid}.json"
    failed_path = rho_root(tag) / "units" / f"{arm}--{sid}.failed.json"
    if out_path.exists() and not force:
        return json.loads(out_path.read_text(encoding="utf-8"))
    if failed_path.exists():
        failed_path.unlink()

    # A (re)run always starts from a freshly materialized unit and an empty
    # session store (E18 §8.1).
    _materialize_unit(tag, arm, session, manifest)
    pi_session_id = pi_session_name(tag, arm, sid)
    for stale in (rho_root(tag) / "sessions").glob(f"*_{pi_session_id}.jsonl"):
        stale.unlink()

    audit = root / "bridge-audit.jsonl"
    if audit.exists():
        audit.unlink()
    extra = {PROVIDER_KEY_ENV: key}
    if arm == "karc-full":
        extra["KARC_PI_BRIDGE_AUDIT"] = str(audit)
        extra["KARC_PI_MCP_CONFIG"] = str(work / ".mcp.json")
        extra["KARC_PI_MCP_SERVER"] = "karc"
    if RETENTION is not None:
        extra["PI_CACHE_RETENTION"] = RETENTION

    streams = root / "streams"
    streams.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    audit_before = 0
    for item in session["tasks"]:
        position = int(item["task"]["session_task"])
        prompt = g1.task_prompt(arm, item, FIXTURE, manifest)
        argv = [str(PI_BIN), "-p", prompt, "--mode", "json",
                "--provider", PROVIDER, "--model", MODEL,
                "--thinking", THINKING, "--no-extensions"]
        if arm == "karc-full":
            argv += ["-e", str(BRIDGE), "--no-builtin-tools",
                     "--tools", ",".join(TOOLS)]
        else:
            argv += ["--no-tools"]
        argv += ["--no-context-files", "--no-skills", "--no-prompt-templates",
                 "--no-themes", "--no-approve",
                 "--append-system-prompt", str(system_append),
                 "--session-dir", str(rho_root(tag) / "sessions"),
                 "--session-id", pi_session_id]
        paced = PACER.wait()
        started = time.time()
        proc = subprocess.run(argv, cwd=str(work), env=pi_env(tag, extra),
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
        closure = [closure_of(call) for call in calls]
        answer = parsed["final_text"][-1] if parsed["final_text"] else ""
        grade = g1.grade_turn(item["task"], answer)
        row = {
            "rho_tag": tag, "reuse_factor": RHOS[tag],
            "provider": PROVIDER, "model": MODEL, "retention": RETENTION,
            "arm": arm, "session_id": sid, "position": position,
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
                # H2 — the OpenAI leg's provider layer is always "completed"
                # (E17B §5), so refusal detection is this disjunction.
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
            # Incidental only.  §8 of the preregistration forbids the accuracy
            # axis; recorded so a silently broken arm is detectable.
            "grade_incidental": grade,
        }
        rows.append(row)
        PACER.record(row["gross_tokens"] + row["usage_turn"]["output"])
        say({"unit": f"{tag}/{arm}/{sid}", "position": position,
             "rc": proc.returncode, "calls": len(calls),
             "gross": row["gross_tokens"],
             "pi_cost": row["usage_turn"]["pi_cost_total"],
             "tes": row["tool_execution_start_count"],
             "audit": row["bridge_audit_lines_this_turn"]})
        # Fail fast: an error stop reason, a missing answer or a zero-usage call
        # means this unit's accounting is already unusable.
        fatal = [c for c in closure
                 if str(c["stopReason"]) == "error"
                 or c["errorMessage_present"]
                 or int(c["totalTokens"] or 0) == 0]
        if proc.returncode != 0 or fatal:
            detail = {
                "unit": f"{tag}/{arm}/{sid}", "position": position,
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
            say({"unit": f"{tag}/{arm}/{sid}", "abort": detail})
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
        "rho_tag": tag, "reuse_factor": RHOS[tag],
        "arm": arm, "session_id": sid,
        "pi_session_id": pi_session_id,
        "pi_session_id_sha256": sha256(pi_session_id),
        "turns_requested": len(session["tasks"]),
        "turns_executed": len(rows),
        "provider": PROVIDER, "model": MODEL,
        "retention": RETENTION, "thinking": THINKING,
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


def _standard_priced_usd(rows: list[dict]) -> float:
    """Running spend at the committed OpenAI standard rate card, for the H6
    guard only.  The judged figures are produced by analyze_e23.py."""
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


# ------------------------------------------------------------------- wave

def _ledger_add(usd: float) -> float:
    """Cell-global H6 ledger over both rho waves."""
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    book = (json.loads(LEDGER.read_text(encoding="utf-8"))
            if LEDGER.exists() else {"usd": 0.0})
    book["usd"] = float(book.get("usd", 0.0)) + usd
    LEDGER.write_text(json.dumps(book, indent=2) + "\n", encoding="utf-8")
    return book["usd"]


def phase_wave(tag: str, *, sessions: list[int], parallel: int, force: bool,
               only: list[str] | None, unit_retries: int = 2) -> dict:
    order = abba_order(sessions)
    if only is not None:
        order = [(arm, idx) for arm, idx in order
                 if f"{arm}--S{idx:02d}" in only]
    spent = {"usd": 0.0}
    lock = threading.Lock()
    aborted: list[str] = []

    def work(job: tuple[str, int]) -> dict:
        arm, index = job
        with lock:
            if aborted:
                return {"skipped": f"{arm}--S{index:02d}", "reason": aborted[0]}
        attempts = 0
        while True:
            attempts += 1
            try:
                result = phase_unit(tag, arm, index, force=force)
                break
            except TurnFailure as failure:
                if attempts > unit_retries:
                    with lock:
                        aborted.append(f"unit {arm}--S{index:02d} failed "
                                       f"{attempts} times: {failure}")
                    return {"failed": f"{arm}--S{index:02d}",
                            "attempts": attempts, "detail": str(failure)}
                say({"wave": tag, "retry": f"{arm}--S{index:02d}",
                     "attempt": attempts + 1})
                time.sleep(30.0)
        result["attempts"] = attempts
        with lock:
            unit_usd = _standard_priced_usd(result["rows"])
            spent["usd"] += unit_usd
            cell_usd = _ledger_add(unit_usd)
            if cell_usd > BUDGET_STOP_USD:
                aborted.append(f"H6 guard: cell standard-priced spend "
                               f"${cell_usd:.4f} exceeded ${BUDGET_STOP_USD}")
            say({"wave": tag, "done": f"{arm}--{result['session_id']}",
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
        "rho_tag": tag, "reuse_factor": RHOS[tag], "parallel": parallel,
        "sessions": [session_id(i) for i in sessions],
        "submission_order": [f"{arm}--S{idx:02d}" for arm, idx in order],
        "units": len(order),
        "wall_seconds": round(time.time() - started, 1),
        "aborted": aborted,
        "tokens_per_minute_target": PACER.tpm,
        "namespace": NAMESPACE,
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
    parser.add_argument("phase", choices=["mtime", "build", "steady", "auth",
                                          "prepare", "registry", "unit", "wave"])
    parser.add_argument("--rho", choices=sorted(RHOS), default=None,
                        help="rho tag: 025 or 075")
    parser.add_argument("--arm", choices=ARMS, default=None)
    parser.add_argument("--session", type=int, default=None)
    parser.add_argument("--sessions", default=None,
                        help="comma list of 1-based session indices for a wave; "
                             "defaults to the steady-state window")
    parser.add_argument("--warmup-sessions", type=int, default=None,
                        help="override the warm-up count that fixes the window; "
                             "defaults to the steady phase's selection")
    parser.add_argument("--parallel", type=int, default=3)
    parser.add_argument("--tpm", type=int, default=0,
                        help="provider tokens-per-minute ceiling (0 disables)")
    parser.add_argument("--unit-retries", type=int, default=2)
    parser.add_argument("--ns", default="",
                        help="execution namespace suffix; changes the workdir "
                             "paths and the pi session ids so a re-run cannot "
                             "be served a discarded execution's cache")
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

    def window_for(tag: str) -> list[int]:
        if args.sessions:
            return [int(x) for x in args.sessions.split(",")]
        if args.warmup_sessions is not None:
            return measure_sessions(args.warmup_sessions)
        steady = phase_steady(tag)
        if not steady["steady_state_reached"]:
            raise SystemExit(f"E23: H11 fired for rho tag {tag}; not measured")
        return measure_sessions(steady["warmup_sessions_selected"])

    if args.phase == "mtime":
        result = phase_mtime()
        out = args.out or "mtime-before.json"
    elif args.phase == "build":
        result = phase_build()
        out = args.out or "build.json"
    elif args.phase == "steady":
        result = phase_steady(args.rho)
        out = args.out or f"steady-{args.rho}.json"
    elif args.phase == "auth":
        result = phase_auth(args.rho)
        out = args.out or f"auth-{args.rho}.json"
    elif args.phase == "prepare":
        result = phase_prepare(args.rho)
        out = args.out or f"prepare-{args.rho}.json"
    elif args.phase == "registry":
        result = phase_registry(args.rho)
        out = args.out or f"registry-{args.rho}.json"
    elif args.phase == "unit":
        result = phase_unit(args.rho, args.arm, args.session, force=args.force)
        out = args.out or f"unit-{args.rho}-{args.arm}-S{args.session:02d}.json"
    else:
        only = args.only.split(",") if args.only else None
        result = phase_wave(args.rho, sessions=window_for(args.rho),
                            parallel=args.parallel, force=args.force,
                            only=only, unit_retries=args.unit_retries)
        out = args.out or f"wave-{args.rho}.json"
    result.setdefault("_meta", {})
    result["_meta"].update({"phase": args.phase, "pi_version": PI_VERSION,
                            "cell": "E23-REUSE-DENSITY",
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
