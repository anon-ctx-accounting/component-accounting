"""E27-BUDGET-10PCT — does the component accounting survive a doubled resident budget?

All eight committed component-accounting cells hold the resident budget at
c = 5% of the corpus (1,756 of 35,128 tokens, `fixture/e4-v2/BUILD.json`).  The
most plausible reviewer objection is that 5% is a special value.  This cell is
the insurance measurement against that objection: it takes E19 leg B's cell
(pi x OpenAI, steady-state window; the paper calls it `pi-O-s`) and changes ONE
input, the resident budget, from 5% to 10% (3,512 tokens).

    harness    pi 0.84.3, isolated PI_CODING_AGENT_DIR under tmp/e27/
    provider   OpenAI gpt-5.6-luna, metered API key
    arms       karc-full (MCP via the pi bridge) and rag-bm25
    workload   reuse factor r = 0.50, schedule seed 4352, 24 sessions x 8 turns
    design     12 paired sessions x 8 turns x 2 arms = 192 turns
    rate card  OpenAI public rates observed 2026-08-28, identical to E19 leg B
    statistics session-unit paired percentile bootstrap, seed 1313, 10,000

THIS FILE IS A THIN OVERRIDE.  Every phase body — materialization, argv
construction, closure asserts, the E18 cache-namespace incident guards, the
steady-state judgment, the H6 spend ledger, the ABBA submission order and the
token pacer — is imported from E23's runner, which is itself E19 leg B's runner
with one axis opened up.  Only these module attributes are replaced:

    BUDGET_PCT               5 -> 10
    RESIDENT_BUDGET_TOKENS   1,756 -> 3,512
    RHOS                     {"b10": 0.50}   (one cell; r stays at E19's value)
    RUN / RAW / CELL_DIR     tmp/e27, this cell's raw
    SCHEDULE_IDENTITY        the c = 10% bundle identity
    rho_root / pi_session_name   a fresh execution namespace (H10)
    BUDGET_CAP_USD           3 -> 10 (this cell's preregistered ceiling)
    SIBLING_CELL_ROOTS       + tmp/e23

WHAT THE BUDGET KNOB TOUCHES -- measured, not assumed
-----------------------------------------------------
`e5_killgate._manifest_at_budget(manifest, pct)` recomputes only
`cell.budget_tokens = int(total_corpus_tokens * pct / 100)`.  Downstream that
one number moves BOTH arms symmetrically, which is the point of the axis:

  * karc-full: the model-free policy replay is run with `c = budget_tokens`, so
    the resident set it admits per session grows (20 versions / ~1,754 tokens at
    5%; 40 versions / ~3,508 tokens at 10%).  Those artifacts are what
    `--append-system-prompt` injects.
  * rag-bm25: `e3_retrieval.bm25_plans(..., budget_tokens)` is budget-parity by
    construction, so the retrieval plan pasted into each turn grows with the
    same number (plan tokens mean 1,729.1 at 5%; 3,451.1 at 10%).

The task schedule itself is NOT a function of the budget: `schedule_sha256` and
`tasks_sha256` at c = 10% are byte-identical to E19 leg B's.  That is gated on
(H14) and is what makes this a one-axis change, unlike E23 where moving rho
rebuilt the task sequence.

WHAT THE BUDGET KNOB ALSO MOVES -- the window, and it is disclosed
-----------------------------------------------------------------
The resident set fills more slowly under a larger budget, so E19's steady-state
criterion (last warm-up session within 5% of the previous session AND >= 95%
budget utilization) is first met later: warm-up 8 -> measurement window S09-S20
at c = 5%, warm-up 12 -> measurement window S13-S24 at c = 10%.  The window is
therefore NOT held fixed; it is a determined consequence of the axis.  The
preregistration records this as a confound the cell does not resolve, and
`phase_build` measures the session and task overlap between the two windows so
the size of the shift is a number rather than an argument.

Phases (identical names and semantics to E23, one cell instead of two rho):
    mtime     snapshot the user's ~/.pi paths and sibling cell roots (no pi)
    build     identity + H12/H13/H14 gates, both budgets, model-free
    steady    smallest warm-up in 8..12 meeting E19's criterion (no model)
    auth      model-free `pi auth check` inside the isolated home
    prepare   materialize 48 units (no pi, no model)
    registry  model-free registry dump through pi --mode rpc
    unit      run one (arm, session) unit of 8 turns
    wave      run the measurement window, ABBA order, parallel

Privacy (R-9): raw rows carry counts, public ids, sha256 and usage numbers only.
The API key is read out of ~/api-key.txt by this process, never printed, never
placed in argv, and reaches only the constructed pi child environment.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
CELL_DIR = HERE.parents[1]                 # docs/experiments/E27-BUDGET-10PCT
REPO = HERE.parents[4]                     # repository root
if not (REPO / "pyproject.toml").exists() or not (REPO / "src" / "karc").is_dir():
    raise SystemExit(f"E27: repository root misresolved as {REPO} (E15 §8.4 guard)")

CELL = "E27-BUDGET-10PCT"
TAG = "b10"                                # this cell's single unit-key tag
BUDGET_PCT = 10
RESIDENT_BUDGET_TOKENS = 3512              # int(35,128 * 10 / 100)
CONTROL_BUDGET_PCT = 5                     # E19 leg B / all eight cells
CONTROL_BUDGET_TOKENS = 1756
REUSE_FACTOR = 0.50                        # E19 leg B's value, held fixed

# ---------------------------------------------------------------- E23 import

E23_RUNNER = (REPO / "docs" / "experiments" / "E23-REUSE-DENSITY" / "scripts"
              / "run_e23_reuse_density.py")
if not E23_RUNNER.exists():
    raise SystemExit(f"E27: E23 runner not found at {E23_RUNNER}")
_spec = importlib.util.spec_from_file_location("karc_e23_runner", E23_RUNNER)
E = importlib.util.module_from_spec(_spec)
sys.modules["karc_e23_runner"] = E
_spec.loader.exec_module(E)                # inserts REPO/src on sys.path

E23_RUNNER_SHA256 = __import__("hashlib").sha256(E23_RUNNER.read_bytes()).hexdigest()

# ------------------------------------------------------------ the overrides

# The c = 10% bundle identity, computed model-free from the committed builders
# before any model call and gated on in `build_bundle`.  `schedule_sha256` and
# `tasks_sha256` are E19 leg B's own values: the budget does not touch the task
# sequence.  `snapshot` and `manifest` differ because they carry the resident
# sets and the budget-parity retrieval plans, which is the axis itself.
IDENTITY_AT_10PCT = {
    "snapshot_sha256": "4a36203e6fdf29aae53276502ba70b5bf1fab3b33167dbdabf88909a7144dd9d",
    "schedule_sha256": "acdec9a853ae051225ac111e4c336bae743fa52fad5264bcb1837f42693f225b",
    "tasks_sha256": "73706b12ea0cba1dcb46963a34da83d36201c942ac2958289ef0129e8d460d74",
    "manifest_sha256": "052b65cfc2e1543aa50131e24db05eea752a4376504819460846d33b0336682a",
}
# E19 leg B's committed identity at c = 5%.  The same override path must
# reproduce it exactly, which proves the only altered input is the budget.
E19_IDENTITY_AT_5PCT = {
    "snapshot_sha256": "f6edbb9e33c63c935aade45961ec8995f121e50e3854713963508925df1c2a73",
    "schedule_sha256": "acdec9a853ae051225ac111e4c336bae743fa52fad5264bcb1837f42693f225b",
    "tasks_sha256": "73706b12ea0cba1dcb46963a34da83d36201c942ac2958289ef0129e8d460d74",
    "manifest_sha256": "861e9114aa9d8b96c43bf671b3f079b517811ff35662af3e81cca18afe61baae",
}
# E19 leg B's measurement window, for the window-shift disclosure.
E19_WINDOW = [f"S{i:02d}" for i in range(9, 21)]
E19_WARMUP = [f"S{i:02d}" for i in range(1, 9)]

RUN = REPO / "tmp" / "e27"                 # gitignored; never e17..e23
RAW = CELL_DIR / "raw"
# H6: this cell's preregistered ceiling is $10 including warm-up.  The wave
# stops itself at $8 so the ceiling is never reached by accident.
BUDGET_CAP_USD = 10.0
BUDGET_STOP_USD = 8.0


def _root(tag: str) -> Path:
    """One execution root for this cell; the namespace suffix is an H10 lever."""
    return RUN / f"budget{BUDGET_PCT}{E.NAMESPACE}"


def _session_name(tag: str, arm: str, sid: str) -> str:
    """The second H10 lever: pi session ids (pi routes prompt_cache_key with
    them) share no prefix with any earlier cell's."""
    return f"e27b{BUDGET_PCT}{E.NAMESPACE}-{arm}-{sid}"


def _apply_overrides() -> None:
    E.BUDGET_PCT = BUDGET_PCT
    E.RESIDENT_BUDGET_TOKENS = RESIDENT_BUDGET_TOKENS
    E.RHOS = {TAG: REUSE_FACTOR}
    E.CONTROL_RHO = REUSE_FACTOR
    E.SCHEDULE_IDENTITY = {TAG: IDENTITY_AT_10PCT}
    E.CELL_DIR = CELL_DIR
    E.RAW = RAW
    E.RUN = RUN
    E.LEDGER = RUN / "spend-ledger.json"
    E.BUDGET_CAP_USD = BUDGET_CAP_USD
    E.BUDGET_STOP_USD = BUDGET_STOP_USD
    # Reuse E23's committed probe extension rather than duplicating it.
    E.PROBE_EXT = (REPO / "docs" / "experiments" / "E23-REUSE-DENSITY"
                   / "scripts" / "tool-registry-probe.ts")
    E.SIBLING_CELL_ROOTS = tuple(
        REPO / "tmp" / name for name in ("e17", "e17b", "e18", "e19", "e21", "e23"))
    E.rho_root = _root
    E.pi_session_name = _session_name


_apply_overrides()


def _at_budget(pct: int, tokens: int):
    """Temporarily point the imported machinery at another budget."""

    class _Ctx:
        def __enter__(self):
            self.pct, self.tok = E.BUDGET_PCT, E.RESIDENT_BUDGET_TOKENS
            E.BUDGET_PCT, E.RESIDENT_BUDGET_TOKENS = pct, tokens
            return self

        def __exit__(self, *exc):
            E.BUDGET_PCT, E.RESIDENT_BUDGET_TOKENS = self.pct, self.tok
            return False

    return _Ctx()


# ------------------------------------------------------------------- key

def _read_api_key() -> str:
    """Read OPENAI_API_KEY out of ~/api-key.txt without ever printing it.

    E21's reader, reused.  The key is never passed in argv and never exported
    into this session's shell environment; it is set on this process only so
    that `E.pi_env` can place it in the pi child's constructed environment,
    which is the sole channel pi accepts a metered key on.
    """
    key = os.environ.get(E.KEY_ENV, "").strip()
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
    raise SystemExit("E27: OPENAI_API_KEY not found in ~/api-key.txt")


def _load_key() -> bool:
    key = _read_api_key()
    if not key:
        return False
    os.environ[E.KEY_ENV] = key
    return True


# ------------------------------------------------------------------- build

def phase_build() -> dict:
    """Model-free gates.  Nothing here calls pi or the provider.

    Gates, all measured on the object actually used:
        identity  the c = 10% bundle must hash to IDENTITY_AT_10PCT
        H12       gold artifact containment rate must be 1.0
        H13       the schedule must keep reuse / supersession structure
        H14       the budget must not move the task sequence, and the same
                  override path at c = 5% must reproduce E19 leg B exactly
    """
    from karc.bench import e5_cache_canary as g1

    bundle = E.build_bundle(TAG)           # identity + H12 + H13 + budget gates
    snapshot = bundle["_snapshot"]
    profile = E.reuse_profile(bundle)
    ramp = [E.resident_of(bundle, i) for i in range(1, E.SESSION_COUNT + 1)]

    with _at_budget(CONTROL_BUDGET_PCT, CONTROL_BUDGET_TOKENS):
        control = E._prepare_bundle(REUSE_FACTOR)
        control_snapshot = g1.bundle_snapshot(control)
        control_ramp = [E.resident_of(control, i)
                        for i in range(1, E.SESSION_COUNT + 1)]

    faithful = {
        "budget_pct": CONTROL_BUDGET_PCT,
        "budget_tokens": int(control["manifest"]["cell"]["budget_tokens"]),
        "snapshot_sha256": control_snapshot["sha256"],
        "schedule_sha256": control["schedule"]["sha256"],
        "tasks_sha256": control["schedule"]["tasks_sha256"],
        "manifest_sha256": control["manifest"]["manifest_sha256"],
        "matches_e19_snapshot":
            control_snapshot["sha256"] == E19_IDENTITY_AT_5PCT["snapshot_sha256"],
        "matches_e19_schedule":
            control["schedule"]["sha256"] == E19_IDENTITY_AT_5PCT["schedule_sha256"],
        "matches_e19_tasks":
            control["schedule"]["tasks_sha256"] == E19_IDENTITY_AT_5PCT["tasks_sha256"],
        "matches_e19_manifest":
            control["manifest"]["manifest_sha256"]
            == E19_IDENTITY_AT_5PCT["manifest_sha256"],
        "why": ("the override path is the E23/E19 builder with the budget opened "
                "up; at c = 5% it must be byte-identical to E19 leg B, which is "
                "what makes the c = 10% build a one-axis change"),
    }
    if not all(faithful[k] for k in ("matches_e19_snapshot", "matches_e19_schedule",
                                     "matches_e19_tasks", "matches_e19_manifest")):
        raise SystemExit("E27: the override path does not reproduce E19 leg B at c = 5%")

    # H14 — axis purity.  The budget must not touch the task sequence.
    axis_purity = {
        "schedule_sha256_equal_across_budgets":
            bundle["schedule"]["sha256"] == control["schedule"]["sha256"],
        "tasks_sha256_equal_across_budgets":
            bundle["schedule"]["tasks_sha256"] == control["schedule"]["tasks_sha256"],
        "manifest_sha256_differs":
            bundle["manifest"]["manifest_sha256"]
            != control["manifest"]["manifest_sha256"],
        "snapshot_sha256_differs":
            snapshot["sha256"] != control_snapshot["sha256"],
        "budget_tokens": {
            "c05": int(control["manifest"]["cell"]["budget_tokens"]),
            "c10": int(bundle["manifest"]["cell"]["budget_tokens"]),
            "ratio": (int(bundle["manifest"]["cell"]["budget_tokens"])
                      / int(control["manifest"]["cell"]["budget_tokens"])),
        },
        "structure_equal_across_budgets":
            E.structure_facts(bundle) == E.structure_facts(control),
        "what_the_budget_moves": {
            "resident_versions_last_session": {
                "c05": control_ramp[-1]["resident_versions"],
                "c10": ramp[-1]["resident_versions"],
            },
            "resident_tokens_last_session": {
                "c05": control_ramp[-1]["resident_tokens"],
                "c10": ramp[-1]["resident_tokens"],
            },
            "rag_plan_tokens_mean": {
                "c05": control["retrieval_audit"]["plan_tokens_mean"],
                "c10": bundle["retrieval_audit"]["plan_tokens_mean"],
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
        and axis_purity["manifest_sha256_differs"])
    if axis_purity["fired"]:
        raise SystemExit("E27: H14 fired — the budget knob moved the task sequence")

    # Window overlap against E19 leg B, computed for every candidate warm-up so
    # the shift is a number before the window is chosen.
    candidates = {}
    for count in range(E.WARMUP_START, E.WARMUP_MAX + 1):
        window = E.measure_sessions(count)
        names = [E.session_id(i) for i in window]
        candidates[str(count)] = {
            "window": names,
            "sessions_shared_with_e19_leg_b": len(set(names) & set(E19_WINDOW)),
            "required_versions_in_window": len(E.slice_versions(bundle, window)),
            "overlap_with_e19_window_versions": len(
                E.slice_versions(bundle, window)
                & E.slice_versions(bundle, list(range(9, 21)))),
            "warmup_vs_window_version_overlap": len(
                E.slice_versions(bundle, E.warmup_sessions(count))
                & E.slice_versions(bundle, window)),
        }

    return {
        "cell": CELL,
        "manipulated_axis": {
            "name": "resident budget",
            "operative_knob": "e5_killgate._manifest_at_budget(manifest, budget_pct)",
            "grid_available_e5_killgate": [5, 10],
            "grid_available_e4_fixture": [1, 2, 5, 10, 20],
            "value_measured": BUDGET_PCT,
            "control_point_not_measured_here": CONTROL_BUDGET_PCT,
            "held_fixed": ["harness pi 0.84.3", "provider openai",
                           "model gpt-5.6-luna", "arms karc-full / rag-bm25",
                           "reuse factor 0.50", "schedule seed 4352",
                           "24-session build", "session length 8",
                           "rate card", "statistics", "12 sessions x 2 arms"],
            "not_held_fixed": ["measurement window position (see "
                               "window_candidates and the steady phase)"],
        },
        "schedule_seed": E.SCHEDULE_SEED,
        "identity": {
            "snapshot_sha256": snapshot["sha256"],
            "schedule_sha256": bundle["schedule"]["sha256"],
            "tasks_sha256": bundle["schedule"]["tasks_sha256"],
            "manifest_sha256": bundle["manifest"]["manifest_sha256"],
            "matches_committed": True,
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
            "reuse_gold_tasks": profile["task_structure_histogram"].get("reuse-gold", 0),
            "new_gold_tasks": profile["task_structure_histogram"].get("new-gold", 0),
            "hot_supersession_trap_tasks": sum(
                1 for task in bundle["tasks"] if task["forbidden_versions"]),
            "fired": False,
        },
        "H14_axis_purity": axis_purity,
        "control_c05_reproduces_e19_leg_b": faithful,
        "structure": E.structure_facts(bundle),
        "reuse_profile": profile,
        "resident_ramp_c10": ramp,
        "resident_ramp_c05": control_ramp,
        "retrieval_audit_c10": bundle["retrieval_audit"],
        "retrieval_audit_c05": control["retrieval_audit"],
        "window_candidates": candidates,
        "e19_leg_b_window": E19_WINDOW,
        "e19_leg_b_warmup": E19_WARMUP,
        "imported_from": {
            "path": str(E23_RUNNER.relative_to(REPO)),
            "sha256": E23_RUNNER_SHA256,
        },
    }


# ------------------------------------------------------------------ steady

def phase_steady() -> dict:
    """E19's criterion, unchanged, applied at c = 10%.

    The judged quantity is an INPUT state quantity — resident injection tokens
    per turn — never a cost ratio, and it is a deterministic function of the
    schedule and the model-free policy replay, so the window is fixed before any
    model call.  Only the budget denominator changed, so the same criterion picks
    a later warm-up.
    """
    steady = E.phase_steady(TAG)
    steady["budget_pct"] = BUDGET_PCT
    steady["budget_tokens"] = RESIDENT_BUDGET_TOKENS
    steady["e19_leg_b_window"] = E19_WINDOW
    steady["e19_leg_b_warmup_sessions"] = len(E19_WARMUP)
    window = steady["measurement_window"]
    steady["window_shift_vs_e19_leg_b"] = {
        "sessions_shared": sorted(set(window) & set(E19_WINDOW)),
        "sessions_shared_count": len(set(window) & set(E19_WINDOW)),
        "sessions_new": sorted(set(window) - set(E19_WINDOW)),
        "note": ("the window is a consequence of the axis, not a free choice: a "
                 "larger budget fills more slowly, so E19's criterion is first "
                 "met later. Disclosed as a confound, not resolved."),
    }
    steady["warmup_turns_executed"] = 0
    steady["warmup_turns_omitted_because"] = (
        "E19 §6.3 measured that executing warm-up turns is a no-op for the "
        "measurement window on this harness: the resident set is fixed by the "
        "schedule position through a model-free policy replay, and every pi unit "
        "has its own workdir and session id, so no harness or provider state "
        "carries over. E23 inherited the same decision.")
    return steady


# -------------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["mtime", "build", "steady", "auth",
                                          "prepare", "registry", "unit", "wave"])
    parser.add_argument("--arm", choices=E.ARMS, default=None)
    parser.add_argument("--session", type=int, default=None)
    parser.add_argument("--sessions", default=None,
                        help="comma list of 1-based session indices for a wave; "
                             "defaults to the steady-state window")
    parser.add_argument("--warmup-sessions", type=int, default=None)
    parser.add_argument("--parallel", type=int, default=3)
    parser.add_argument("--tpm", type=int, default=0,
                        help="provider tokens-per-minute ceiling (0 disables)")
    parser.add_argument("--unit-retries", type=int, default=2)
    parser.add_argument("--ns", default="",
                        help="execution namespace suffix; changes both the "
                             "workdir path and the pi session ids (H10)")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--only", default=None,
                        help="comma list of arm--Sxx unit keys")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    E.PACER = E.TokenPacer(args.tpm)
    E.NAMESPACE = args.ns
    _apply_overrides()
    RAW.mkdir(parents=True, exist_ok=True)

    def window() -> list[int]:
        if args.sessions:
            return [int(x) for x in args.sessions.split(",")]
        if args.warmup_sessions is not None:
            return E.measure_sessions(args.warmup_sessions)
        steady = phase_steady()
        if not steady["steady_state_reached"]:
            raise SystemExit("E27: H11 fired; not measured")
        return E.measure_sessions(steady["warmup_sessions_selected"])

    if args.phase == "mtime":
        result = E.phase_mtime()
        out = args.out or "mtime-before.json"
    elif args.phase == "build":
        result = phase_build()
        out = args.out or "build.json"
    elif args.phase == "steady":
        result = phase_steady()
        out = args.out or f"steady-{TAG}.json"
    elif args.phase == "auth":
        _load_key()
        result = E.phase_auth(TAG)
        out = args.out or f"auth-{TAG}.json"
    elif args.phase == "prepare":
        result = E.phase_prepare(TAG)
        out = args.out or f"prepare-{TAG}.json"
    elif args.phase == "registry":
        _load_key()
        result = E.phase_registry(TAG)
        out = args.out or f"registry-{TAG}.json"
    elif args.phase == "unit":
        _load_key()
        result = E.phase_unit(TAG, args.arm, args.session, force=args.force)
        out = args.out or f"unit-{TAG}-{args.arm}-S{args.session:02d}.json"
    else:
        _load_key()
        only = args.only.split(",") if args.only else None
        result = E.phase_wave(TAG, sessions=window(), parallel=args.parallel,
                              force=args.force, only=only,
                              unit_retries=args.unit_retries)
        out = args.out or f"wave-{TAG}.json"

    result.setdefault("_meta", {})
    result["_meta"].update({
        "phase": args.phase, "cell": CELL, "pi_version": E.PI_VERSION,
        "provider": E.PROVIDER, "model": E.MODEL,
        "budget_pct": BUDGET_PCT, "budget_tokens": RESIDENT_BUDGET_TOKENS,
        "reuse_factor": REUSE_FACTOR, "namespace": E.NAMESPACE,
        "imported_runner_sha256": E23_RUNNER_SHA256,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    (RAW / out).write_text(json.dumps(result, indent=2, sort_keys=True,
                                      ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {RAW / out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
