"""E28-BUDGET-SWEEP — the resident-budget axis with the measurement window held fixed.

E27 raised the resident budget from 5% to 10% on E19 leg B's steady-state cell.
Because a larger budget fills more slowly, E19's steady-state criterion picked a
later warm-up and the measurement window moved S09-S20 -> S13-S24.  That cell's
contrast is therefore budget AND window position together, and it said so.

This cell removes the window from the contrast by moving to the cell where the
window is not chosen at all: E18-PI-REVERSAL runs the GROWTH window S01-S12 of a
12-session build with no warm-up, so the window is the whole schedule and cannot
shift with the budget.  Four cells are run against E18's two committed controls:

    cell        leg  provider / model          budget            control
    E28-O-025   B    openai gpt-5.6-luna       2.5%   878 tok    pi-O-g (E18 leg B)
    E28-O-10    B    openai gpt-5.6-luna       10%  3,512 tok    pi-O-g
    E28-O-20    B    openai gpt-5.6-luna       20%  7,025 tok    pi-O-g
    E28-A-10    A    anthropic claude-sonnet-5 10%  3,512 tok    pi-A-g (E18 leg A)

THIS FILE IS A THIN OVERRIDE of E18's runner.  Every phase body -- unit
materialization, argv construction, the closure asserts, the E18 cache-namespace
incident guards, the three-path MCP check, the ABBA submission order, the token
pacer and the H6 spend guard -- is imported from
`docs/experiments/E18-PI-REVERSAL/scripts/run_e18_pi_reversal.py`, which is the
code that produced both control cells.  Only these module attributes are
replaced:

    RUN                 tmp/e18            -> tmp/e28
    RAW                 E18's raw          -> this cell's raw
    NAMESPACE           ""                 -> the cell tag (H10: this moves both
                                              the workdir path and the pi session
                                              id, since E18 derives both from it)
    BUDGET_STOP_USD     12                 -> the batch's remaining allowance
    build_bundle        the c = 5% builder -> the same builder at this cell's
                                              budget, gated to reproduce E18's
                                              committed identity byte for byte
                                              when asked for c = 5%

BUDGET OVERRIDE PATH -- copied from E27, which copied it from E23
------------------------------------------------------------------
`e5_killgate._manifest_at_budget(manifest, pct)` recomputes exactly one number,
`cell.budget_tokens = int(total_corpus_tokens * pct / 100)`.  That number then
moves BOTH arms:

  * karc-full: the model-free policy replay admits a larger resident set, which
    is what `--append-system-prompt` injects.
  * rag-bm25: `e3_retrieval.bm25_plans(..., budget_tokens)` is budget-parity by
    construction, so the retrieval plan pasted into every turn grows with the
    same number.

The task schedule is NOT a function of the budget: `schedule_sha256` and
`tasks_sha256` are byte-identical at every budget measured here, which is gated
on (H14).  `e5_cache_canary.prepare_bundle` hard-codes BUDGET_PCT = 5 and may not
be edited (every committed cell reproduces through it), so its body is reproduced
in `_prepare_bundle` with that one parameter opened up -- exactly as E23 did for
the reuse factor.  Faithfulness is not assumed: at c = 5% the output must hash to
E18's committed identity, which `phase_build` checks.

THE FRACTIONAL 2.5% CELL -- a mechanical detail, recorded rather than hidden
---------------------------------------------------------------------------
`e4_replay.replay_cell` guards its explicit-manifest branch with
`int(cell["budget_pct"]) != budget_pct`, an int() cast that assumes an integer
percentage.  At 2.5% the manifest honestly carries `budget_pct: 2.5` and
`budget_tokens: 878`, so the guard is called with `int(2.5) == 2`.  The operative
value is unaffected -- `replay_cell` sets `PolicyConfig.c` from
`manifest["cell"]["budget_tokens"]`, i.e. 878, and uses `budget_pct` for nothing
else.  `phase_build` proves this rather than asserting it: it also builds the
genuine c = 2% bundle (702 tokens) and requires the 2.5% resident ramp to differ
from it.

Phases (E18's names and semantics):
    mtime     snapshot the user's ~/.pi paths and every sibling cell root
    build     identity + H12/H13/H14 gates + the c = 5% reproduction (no pi)
    auth      model-free `pi auth check` inside the isolated home
    prepare   materialize 24 units for one cell (no pi, no model)
    registry  model-free registry dump through pi --mode rpc
    unit      run one (arm, session) unit of 8 turns
    wave      run all 24 units, ABBA order, parallel

Privacy (R-9): raw rows carry counts, public ids, sha256 and usage numbers only.
API keys are read out of ~/api-key.txt by this process, never printed, never
placed in argv, never exported into the calling session, and reach only the
constructed pi child environment.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve()
CELL_DIR = HERE.parents[1]                 # docs/experiments/E28-BUDGET-SWEEP
REPO = HERE.parents[4]                     # repository root
if not (REPO / "pyproject.toml").exists() or not (REPO / "src" / "karc").is_dir():
    raise SystemExit(f"E28: repository root misresolved as {REPO} (E15 §8.4 guard)")

CELL = "E28-BUDGET-SWEEP"

# ---------------------------------------------------------------- E18 import

E18_RUNNER = (REPO / "docs" / "experiments" / "E18-PI-REVERSAL" / "scripts"
              / "run_e18_pi_reversal.py")
if not E18_RUNNER.exists():
    raise SystemExit(f"E28: E18 runner not found at {E18_RUNNER}")
_spec = importlib.util.spec_from_file_location("karc_e18_runner", E18_RUNNER)
E = importlib.util.module_from_spec(_spec)
sys.modules["karc_e18_runner"] = E
_spec.loader.exec_module(E)                # inserts REPO/src on sys.path
E18_RUNNER_SHA256 = hashlib.sha256(E18_RUNNER.read_bytes()).hexdigest()

FIXTURE = REPO / "fixture" / "e4-v2"
RUN = REPO / "tmp" / "e28"                 # gitignored; never tmp/e18 or a sibling
RAW = CELL_DIR / "raw"
LEDGER = RUN / "spend-ledger.json"

# E18's own schedule bundle: seed 4200, 12 sessions, 8 turns, reuse factor 0.50.
SCHEDULE_SEED = 4200
SESSION_COUNT = 12
SESSION_LENGTH = 8
REUSE_FACTOR = 0.50
WINDOW = [f"S{i:02d}" for i in range(1, SESSION_COUNT + 1)]   # the growth window
CONTROL_BUDGET_PCT = 5
CONTROL_BUDGET_TOKENS = 1756

# H6 -- the ceiling is for the BATCH of four cells, not per cell, so the guard
# reads a cumulative ledger.  Both figures are standard-rate-card dollars; the
# actually charged amount is at pi's catalog rates, which are <= standard on both
# providers, so a standard-dollar ceiling bounds real spend from above.
BATCH_CAP_USD = 10.0
BATCH_STOP_USD = 9.0

SIBLING_CELL_ROOTS = tuple(
    REPO / "tmp" / name
    for name in ("e17", "e17b", "e18", "e19", "e21", "e23", "e25", "e26", "e27"))

# ------------------------------------------------------------------- cells

CELLS = {
    "o025": {"name": "E28-O-025", "leg": "B", "budget_pct": 2.5,
             "budget_tokens": 878, "replay_pct": 2,
             "control": "pi-O-g (E18-PI-REVERSAL leg B)"},
    "o10": {"name": "E28-O-10", "leg": "B", "budget_pct": 10,
            "budget_tokens": 3512, "replay_pct": 10,
            "control": "pi-O-g (E18-PI-REVERSAL leg B)"},
    "o20": {"name": "E28-O-20", "leg": "B", "budget_pct": 20,
            "budget_tokens": 7025, "replay_pct": 20,
            "control": "pi-O-g (E18-PI-REVERSAL leg B)"},
    "a10": {"name": "E28-A-10", "leg": "A", "budget_pct": 10,
            "budget_tokens": 3512, "replay_pct": 10,
            "control": "pi-A-g (E18-PI-REVERSAL leg A)"},
}

# E18's committed identity at c = 5%, from docs/experiments/E18-PI-REVERSAL/raw/
# prepare-{A,B}.json and docs/experiments/E10-P2-XR/canary/raw/schedule.json.
E18_IDENTITY_AT_5PCT = {
    "snapshot_sha256": "a19dea499215f03560e18d2bd53249cb0c63fcec0fb8c0df0449b48c50840301",
    "schedule_sha256": "9944a77572e2d130a0150f865024785929771cd13a9264ab315b4bfa5f26eea3",
    "tasks_sha256": "618193d35103b819d83ce2d13dea40d7e3a65cbae5072266ab693613fe3b3584",
    "manifest_sha256": "0b663cef5e449eac060488136475082c191d7b0e950b01a2813ac70d2d8a3418",
}

# The per-cell bundle identities, computed model-free from the committed builders
# before any model call and registered in preregistration.md §3.
IDENTITY = {
    "o025": {
        "snapshot_sha256": "a7c7caf66e77974e3172fa43a476c717d2f8f097fd292617ad2846ce1a0664b7",
        "schedule_sha256": E18_IDENTITY_AT_5PCT["schedule_sha256"],
        "tasks_sha256": E18_IDENTITY_AT_5PCT["tasks_sha256"],
        "manifest_sha256": "11652d372f13b1d1cff6f5fa93971eee00e0995a5d047647d7e965e4a16c64ca",
    },
    "o10": {
        "snapshot_sha256": "7d16fa0f260d668e880b3164c51ac8da9420b08253d566ddf4818b880302bbed",
        "schedule_sha256": E18_IDENTITY_AT_5PCT["schedule_sha256"],
        "tasks_sha256": E18_IDENTITY_AT_5PCT["tasks_sha256"],
        "manifest_sha256": "5eece713507dcaae31c313c78fbf723e35ce8daccaff5fcdee0f94909f256ff0",
    },
    "o20": {
        "snapshot_sha256": "5ce38390128a7ede023fe41725c51f3274c4bfa863830a8753f3e5dd1497d719",
        "schedule_sha256": E18_IDENTITY_AT_5PCT["schedule_sha256"],
        "tasks_sha256": E18_IDENTITY_AT_5PCT["tasks_sha256"],
        "manifest_sha256": "e3dce84106730f4cafe1932cb45a574f2048b054273bb0d8a4211a09209078e4",
    },
}
IDENTITY["a10"] = IDENTITY["o10"]          # same budget, same model-free bundle

# The genuine c = 2% build, used only as the falsifier for the 2.5% replay guard.
IDENTITY_AT_2PCT = {
    "snapshot_sha256": "ccabdc7219a0d6650f11789e7585fd8b906aa4aa432f2be7abac14483d28a130",
    "budget_tokens": 702,
}

_CURRENT = "o10"
_BUNDLE_CACHE: dict[tuple, dict] = {}


# ------------------------------------------------------------------- bundle

def _prepare_bundle(budget_pct: float, replay_pct: int) -> dict:
    """`e5_cache_canary.prepare_bundle` with `budget_pct` opened up.

    The library function hard-codes BUDGET_PCT = 5 and may not be edited (every
    committed cell reproduces through it), so its body is reproduced here with
    the one parameter this cell manipulates.  E23 did the same for the reuse
    factor and E27 inherited it.  Faithfulness is gated, not assumed: at
    budget_pct = 5 the output must hash to E18's committed identity.
    """
    key = (budget_pct, replay_pct)
    if key in _BUNDLE_CACHE:
        return _BUNDLE_CACHE[key]

    from karc.bench import e5_cache_canary as g1
    from karc.bench.e3_retrieval import bm25_plans, chunk_fixture
    from karc.bench.e4_replay import load_confirmed_config, replay_cell
    from karc.bench.e5_killgate import (
        ENGINE_RHO_LABEL, ENGINE_SIGMA_LABEL, _manifest_at_budget,
        build_reuse_schedule,
    )

    schedule, schedule_manifest, tasks = build_reuse_schedule(
        REPO, schedule_seed=SCHEDULE_SEED, session_length=SESSION_LENGTH,
        reuse_factor=REUSE_FACTOR, session_count=SESSION_COUNT,
        allow_unregistered_seed=True,
    )
    manifest = _manifest_at_budget(schedule_manifest, budget_pct)
    replay_summary, replay_rows = replay_cell(
        rho=ENGINE_RHO_LABEL, sigma=ENGINE_SIGMA_LABEL, budget_pct=replay_pct,
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
            "schedule_seed": SCHEDULE_SEED,
            "session_id": sid,
            "initial_resident_versions": initial,
            "tasks": rows,
        })
    if len(sessions) != SESSION_COUNT:
        raise AssertionError("session grouping changed cardinality")
    bundle = {
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
    bundle["_snapshot"] = g1.bundle_snapshot(bundle)
    _BUNDLE_CACHE[key] = bundle
    return bundle


def bundle_for(tag: str) -> dict:
    """One cell's bundle, gated on its registered identity and H12/H13."""
    spec = CELLS[tag]
    bundle = _prepare_bundle(spec["budget_pct"], spec["replay_pct"])
    want = IDENTITY[tag]
    got = {
        "snapshot_sha256": bundle["_snapshot"]["sha256"],
        "schedule_sha256": bundle["schedule"]["sha256"],
        "tasks_sha256": bundle["schedule"]["tasks_sha256"],
        "manifest_sha256": bundle["manifest"]["manifest_sha256"],
    }
    for field, value in want.items():
        if got[field] != value:
            raise SystemExit(f"E28[{tag}]: {field} mismatch {got[field]} != {value}")
    if int(bundle["manifest"]["cell"]["budget_tokens"]) != spec["budget_tokens"]:
        raise SystemExit(f"E28[{tag}]: budget tokens are not the registered "
                         f"{spec['budget_tokens']}")
    if float(bundle["schedule"]["reuse_factor_measured"]) != REUSE_FACTOR:
        raise SystemExit(f"E28[{tag}]: reuse factor is not E18's 0.50")
    rate = float(bundle["retrieval_audit"]["gold_containment_rate"])
    if rate != 1.0:
        raise SystemExit(f"E28[{tag}]: H12 fired (gold_containment_rate={rate})")
    structure = bundle["manifest"]["structure"]
    if int(structure["reused_occurrences"]) <= 0:
        raise SystemExit(f"E28[{tag}]: H13 fired (zero reuse)")
    if int(structure["supersession_events"]) <= 0:
        raise SystemExit(f"E28[{tag}]: H13 fired (no supersession)")
    if not bool(structure["sigma_defined"]):
        raise SystemExit(f"E28[{tag}]: H13 fired (sigma undefined)")
    return bundle


def resident_of(bundle: dict, index: int) -> dict:
    """The resident set injected for one session, as an INPUT state quantity."""
    manifest = bundle["manifest"]
    session = bundle["sessions"][index - 1]
    resident = [v for v in session["initial_resident_versions"]
                if v in manifest["artifacts"]]
    tokens = sum(int(manifest["artifacts"][v]["size_tok"]) for v in resident)
    budget = int(manifest["cell"]["budget_tokens"])
    return {
        "session_index": index,
        "session_id": session["session_id"],
        "resident_versions": len(resident),
        "resident_tokens": tokens,
        "budget_tokens": budget,
        "budget_utilization": tokens / budget,
    }


def structure_facts(bundle: dict) -> dict:
    structure = bundle["manifest"]["structure"]
    return {key: structure[key] for key in sorted(structure)}


# ----------------------------------------------------------------- overrides

def _apply_overrides(tag: str) -> None:
    global _CURRENT
    _CURRENT = tag
    E.RUN = RUN
    E.RAW = RAW
    E.NAMESPACE = tag          # H10: moves leg_root AND the pi session id
    E.build_bundle = lambda: bundle_for(_CURRENT)


# -------------------------------------------------------------------- keys

# ~/api-key.txt names the Anthropic credential CLAUDE_API_KEY; pi expects it in
# the child environment as ANTHROPIC_API_KEY and E18's runner reads it out of
# KARC_ANTHROPIC_KEY on this process.
KEY_FILE_NAMES = {"A": "CLAUDE_API_KEY", "B": "OPENAI_API_KEY"}


def _read_api_key(leg: str) -> str:
    """Read one provider key out of ~/api-key.txt without ever printing it.

    E21's reader, reused.  The key is never passed in argv and never exported
    into the calling session's shell; it is set on this process only so that
    `E.pi_env` can place it in the pi child's constructed environment, which is
    the sole channel pi accepts a metered key on.
    """
    env_name = E.LEGS[leg]["key_env"]
    key = os.environ.get(env_name, "").strip()
    if key:
        return key
    wanted = KEY_FILE_NAMES[leg]
    path = Path.home() / "api-key.txt"
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[len("export "):]
        name, _, value = line.partition("=")
        if name.strip() == wanted:
            return value.strip().strip('"').strip("'")
    raise SystemExit(f"E28: {wanted} not found in ~/api-key.txt")


def _load_key(leg: str) -> None:
    os.environ[E.LEGS[leg]["key_env"]] = _read_api_key(leg)


# ------------------------------------------------------------------ ledger

def _ledger() -> dict:
    if LEDGER.exists():
        return json.loads(LEDGER.read_text(encoding="utf-8"))
    return {"cells": {}, "standard_priced_usd_total": 0.0}


def _ledger_write(tag: str, usd: float) -> dict:
    book = _ledger()
    book["cells"][tag] = round(float(usd), 6)
    book["standard_priced_usd_total"] = round(
        sum(book["cells"].values()), 6)
    book["batch_cap_usd"] = BATCH_CAP_USD
    book["batch_stop_usd"] = BATCH_STOP_USD
    book["currency"] = ("standard rate cards; pi charges at catalog rates which "
                        "are <= standard on both providers, so this bounds real "
                        "spend from above")
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text(json.dumps(book, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return book


# ------------------------------------------------------------------- mtime

def phase_mtime() -> dict:
    """E18's user-path snapshot, widened to every sibling cell root (H7)."""
    result = E.phase_mtime()
    result["sibling_cell_roots"] = {
        str(path): ({"exists": True, "mtime_ns": path.stat().st_mtime_ns}
                    if path.exists() else {"exists": False})
        for path in SIBLING_CELL_ROOTS}
    return result


# ------------------------------------------------------------------- build

def phase_build(tag: str) -> dict:
    """Model-free gates.  Nothing here calls pi or the provider.

    identity  this cell's bundle must hash to the registered IDENTITY
    H12       gold artifact containment rate must be 1.0
    H13       the schedule must keep reuse / supersession structure
    H14       the budget must not move the task sequence, and the same override
              path at c = 5% must reproduce E18 byte for byte
    2.5% only the replay must have used c = 878, proven by requiring its resident
              ramp to differ from the genuine c = 2% build
    """
    spec = CELLS[tag]
    bundle = bundle_for(tag)
    ramp = [resident_of(bundle, i) for i in range(1, SESSION_COUNT + 1)]

    control = _prepare_bundle(CONTROL_BUDGET_PCT, CONTROL_BUDGET_PCT)
    control_ramp = [resident_of(control, i) for i in range(1, SESSION_COUNT + 1)]
    faithful = {
        "budget_pct": CONTROL_BUDGET_PCT,
        "budget_tokens": int(control["manifest"]["cell"]["budget_tokens"]),
        "snapshot_sha256": control["_snapshot"]["sha256"],
        "schedule_sha256": control["schedule"]["sha256"],
        "tasks_sha256": control["schedule"]["tasks_sha256"],
        "manifest_sha256": control["manifest"]["manifest_sha256"],
        "why": ("the override path is e5_cache_canary.prepare_bundle's body with "
                "the budget opened up; at c = 5% it must be byte-identical to "
                "E18's committed bundle, which is what makes this cell a "
                "one-axis change from its control"),
    }
    for field, value in E18_IDENTITY_AT_5PCT.items():
        faithful[f"matches_e18_{field}"] = faithful[field] == value
    if not all(faithful[f"matches_e18_{f}"] for f in E18_IDENTITY_AT_5PCT):
        raise SystemExit("E28: the override path does not reproduce E18 at c = 5%")

    axis_purity = {
        "schedule_sha256_equal_across_budgets":
            bundle["schedule"]["sha256"] == control["schedule"]["sha256"],
        "tasks_sha256_equal_across_budgets":
            bundle["schedule"]["tasks_sha256"] == control["schedule"]["tasks_sha256"],
        "manifest_sha256_differs":
            bundle["manifest"]["manifest_sha256"]
            != control["manifest"]["manifest_sha256"],
        "snapshot_sha256_differs":
            bundle["_snapshot"]["sha256"] != control["_snapshot"]["sha256"],
        "structure_equal_across_budgets":
            structure_facts(bundle) == structure_facts(control),
        "budget_tokens": {
            "c05": int(control["manifest"]["cell"]["budget_tokens"]),
            "cell": int(bundle["manifest"]["cell"]["budget_tokens"]),
            "ratio": (int(bundle["manifest"]["cell"]["budget_tokens"])
                      / int(control["manifest"]["cell"]["budget_tokens"])),
        },
        "what_the_budget_moves": {
            "resident_versions_last_session": {
                "c05": control_ramp[-1]["resident_versions"],
                "cell": ramp[-1]["resident_versions"],
            },
            "resident_tokens_last_session": {
                "c05": control_ramp[-1]["resident_tokens"],
                "cell": ramp[-1]["resident_tokens"],
            },
            "rag_plan_tokens_mean": {
                "c05": control["retrieval_audit"]["plan_tokens_mean"],
                "cell": bundle["retrieval_audit"]["plan_tokens_mean"],
            },
            "note": ("both arms move together: the policy replay admits a larger "
                     "resident set and bm25_plans is budget-parity, so the "
                     "retrieval payload grows by the same number"),
        },
        "fired": False,
    }
    axis_purity["fired"] = not (
        axis_purity["schedule_sha256_equal_across_budgets"]
        and axis_purity["tasks_sha256_equal_across_budgets"]
        and axis_purity["structure_equal_across_budgets"]
        and axis_purity["manifest_sha256_differs"]
        and axis_purity["snapshot_sha256_differs"])
    if axis_purity["fired"]:
        raise SystemExit(f"E28[{tag}]: H14 fired — the budget moved the task sequence")

    fractional = None
    if float(spec["budget_pct"]) != int(spec["budget_pct"]):
        two = _prepare_bundle(2, 2)
        two_ramp = [resident_of(two, i) for i in range(1, SESSION_COUNT + 1)]
        differs = ([r["resident_tokens"] for r in ramp]
                   != [r["resident_tokens"] for r in two_ramp])
        fractional = {
            "what": ("e4_replay.replay_cell guards its explicit-manifest branch "
                     "with int(cell['budget_pct']) != budget_pct, so a "
                     "fractional percentage must be passed to that guard as "
                     "int(2.5) == 2. budget_pct is used for nothing else; "
                     "PolicyConfig.c comes from cell['budget_tokens']."),
            "manifest_budget_pct": bundle["manifest"]["cell"]["budget_pct"],
            "manifest_budget_tokens": int(bundle["manifest"]["cell"]["budget_tokens"]),
            "replay_pct_argument": spec["replay_pct"],
            "genuine_c02_snapshot_sha256": two["_snapshot"]["sha256"],
            "genuine_c02_matches_registered":
                two["_snapshot"]["sha256"] == IDENTITY_AT_2PCT["snapshot_sha256"],
            "genuine_c02_budget_tokens": int(two["manifest"]["cell"]["budget_tokens"]),
            "resident_ramp_differs_from_c02": differs,
            "reading": ("if the replay had silently used c = 702 the two ramps "
                        "would be identical; they are not, so the policy ran at "
                        "c = 878"),
            "fired": not differs,
        }
        if fractional["fired"]:
            raise SystemExit("E28[o025]: the 2.5% replay ran at the c = 2% budget")

    return {
        "cell": spec["name"],
        "tag": tag,
        "leg": spec["leg"],
        "control_cell": spec["control"],
        "manipulated_axis": {
            "name": "resident budget",
            "operative_knob": "e5_killgate._manifest_at_budget(manifest, budget_pct)",
            "value_measured": {"budget_pct": spec["budget_pct"],
                               "budget_tokens": spec["budget_tokens"]},
            "control_point": {"budget_pct": CONTROL_BUDGET_PCT,
                              "budget_tokens": CONTROL_BUDGET_TOKENS},
            "held_fixed": ["harness pi 0.84.3", "provider and model",
                           "arms karc-full / rag-bm25", "reuse factor 0.50",
                           "schedule seed 4200", "12-session build",
                           "session length 8", "rate card", "statistics",
                           "MEASUREMENT WINDOW S01-S12 (the whole build; no "
                           "warm-up, so the budget cannot move it)"],
            "not_held_fixed": [],
        },
        "measurement_window": WINDOW,
        "window_is_fixed_because": (
            "E18's design measures the growth window S01-S12 of a 12-session "
            "build with no warm-up. The window is the whole schedule, so unlike "
            "E27 (which inherited E19's steady-state criterion) no budget value "
            "can move it. That is the entire reason this batch uses E18's bundle "
            "rather than E19's."),
        "schedule_seed": SCHEDULE_SEED,
        "identity": {
            "snapshot_sha256": bundle["_snapshot"]["sha256"],
            "schedule_sha256": bundle["schedule"]["sha256"],
            "tasks_sha256": bundle["schedule"]["tasks_sha256"],
            "manifest_sha256": bundle["manifest"]["manifest_sha256"],
            "matches_registered": True,
        },
        "cell_parameters": bundle["manifest"]["cell"],
        "H12_gold_containment": {
            "gold_containment_rate": bundle["retrieval_audit"]["gold_containment_rate"],
            "is_one": bundle["retrieval_audit"]["gold_containment_rate"] == 1.0,
            "fired": bundle["retrieval_audit"]["gold_containment_rate"] != 1.0,
            "mean_budget_utilization":
                bundle["retrieval_audit"]["mean_budget_utilization"],
            "plan_tokens_mean": bundle["retrieval_audit"]["plan_tokens_mean"],
        },
        "H13_non_degenerate_branch": {
            "reused_occurrences": bundle["manifest"]["structure"]["reused_occurrences"],
            "supersession_events": bundle["manifest"]["structure"]["supersession_events"],
            "sigma_defined": bundle["manifest"]["structure"]["sigma_defined"],
            "sigma_measured": bundle["manifest"]["structure"]["sigma_measured"],
            "hot_supersession_trap_tasks": sum(
                1 for task in bundle["tasks"] if task["forbidden_versions"]),
            "fired": False,
        },
        "H14_axis_purity": axis_purity,
        "control_c05_reproduces_e18": faithful,
        "fractional_budget_pct_check": fractional,
        "structure": structure_facts(bundle),
        "resident_ramp": ramp,
        "resident_ramp_c05_control": control_ramp,
        "retrieval_audit": bundle["retrieval_audit"],
        "retrieval_audit_c05_control": control["retrieval_audit"],
        "imported_from": {
            "path": str(E18_RUNNER.relative_to(REPO)),
            "sha256": E18_RUNNER_SHA256,
        },
    }


# -------------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["mtime", "build", "auth", "prepare",
                                          "registry", "unit", "wave"])
    parser.add_argument("--cell", choices=sorted(CELLS), default=None)
    parser.add_argument("--arm", choices=E.ARMS, default=None)
    parser.add_argument("--session", type=int, default=None)
    parser.add_argument("--parallel", type=int, default=3)
    parser.add_argument("--tpm", type=int, default=0,
                        help="provider tokens-per-minute ceiling (0 disables)")
    parser.add_argument("--unit-retries", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--only", default=None,
                        help="comma list of arm--Sxx unit keys")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    E.PACER = E.TokenPacer(args.tpm)
    RAW.mkdir(parents=True, exist_ok=True)

    if args.phase == "mtime":
        _apply_overrides("o10")
        result = phase_mtime()
        out = args.out or "mtime-before.json"
        tag = None
        leg = None
    else:
        tag = args.cell
        if tag is None:
            raise SystemExit("E28: --cell is required for this phase")
        _apply_overrides(tag)
        leg = CELLS[tag]["leg"]
        if args.phase == "build":
            result = phase_build(tag)
            out = args.out or f"{tag}/build.json"
        elif args.phase == "auth":
            _load_key(leg)
            result = E.phase_auth(leg)
            out = args.out or f"{tag}/auth.json"
        elif args.phase == "prepare":
            result = E.phase_prepare(leg)
            out = args.out or f"{tag}/prepare.json"
        elif args.phase == "registry":
            _load_key(leg)
            result = E.phase_registry(leg)
            out = args.out or f"{tag}/registry.json"
        elif args.phase == "unit":
            _load_key(leg)
            result = E.phase_unit(leg, args.arm, args.session, force=args.force)
            out = args.out or f"{tag}/unit-{args.arm}-S{args.session:02d}.json"
        else:
            _load_key(leg)
            book = _ledger()
            already = float(book.get("standard_priced_usd_total", 0.0))
            remaining = BATCH_STOP_USD - already
            if remaining <= 0:
                raise SystemExit(
                    f"E28: batch spend guard — ${already:.4f} already booked, "
                    f"stop threshold is ${BATCH_STOP_USD}")
            E.BUDGET_STOP_USD = remaining
            only = args.only.split(",") if args.only else None
            result = E.phase_wave(leg, parallel=args.parallel, force=args.force,
                                  only=only, unit_retries=args.unit_retries)
            result["batch_spend_guard"] = {
                "already_booked_standard_usd": round(already, 6),
                "this_wave_allowance_standard_usd": round(remaining, 6),
                "batch_stop_usd": BATCH_STOP_USD,
                "batch_cap_usd": BATCH_CAP_USD,
                "ledger": _ledger_write(
                    tag, float(result.get("running_standard_priced_usd") or 0.0)),
            }
            out = args.out or f"{tag}/wave.json"

    result.setdefault("_meta", {})
    result["_meta"].update({
        "phase": args.phase, "cell": CELL,
        "cell_tag": tag,
        "cell_name": CELLS[tag]["name"] if tag else None,
        "leg": leg,
        "pi_version": E.PI_VERSION,
        "provider": E.LEGS[leg]["provider"] if leg else None,
        "model": E.LEGS[leg]["model"] if leg else None,
        "budget_pct": CELLS[tag]["budget_pct"] if tag else None,
        "budget_tokens": CELLS[tag]["budget_tokens"] if tag else None,
        "reuse_factor": REUSE_FACTOR,
        "namespace": E.NAMESPACE,
        "imported_runner_sha256": E18_RUNNER_SHA256,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    path = RAW / out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True,
                               ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
