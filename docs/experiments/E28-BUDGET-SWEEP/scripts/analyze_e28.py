"""E28-BUDGET-SWEEP — crosscheck, statistics and the preregistered judgments.

The statistics are E18's, imported rather than copied: `resample_matrix`,
`percentile_ci`, `paired_bootstrap_mean`, `paired_bootstrap_ratio_of_sums`,
`priced_usd`, `per_session_components`, `leg_statistics`, `crosscheck`,
`load_units` and `write_turns` all come out of
`docs/experiments/E18-PI-REVERSAL/scripts/analyze_e18.py`, which is the code that
produced both control cells.  Session-unit paired percentile bootstrap, seed
1313, 10,000 resamples, one shared 10,000 x 12 index matrix so every paired
comparison uses common random numbers.

Emitted:

    raw/<tag>/turns.jsonl     one row per measurement turn, closure included
    raw/<tag>/crosscheck.json instrumentation contract + hold conditions
    raw/analysis.json         per cell: component_totals, delta karc-rag,
                              gross/priced ratio bootstraps, per-session ratios,
                              mu_w_star, the per-turn-position mechanism table,
                              the resident utilization trajectory; plus the
                              control contrast against E18 and E27 and the three
                              batch-level questions.

Preregistration §4 registers exactly one directional prediction per cell:

    B3  delta_cache_read(karc - rag) > 0  AND  delta_creation(karc - rag) < 0

B1 (gross ratio), B2 (priced ratio) and B4 (mu_w*) are measured and reported with
intervals and carry `predicted: false`; nothing here turns them into a verdict.
mu_w* is recomputed from each cell's own component totals.

No model call, no network.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
CELL_DIR = HERE.parents[1]
REPO = HERE.parents[4]
if not (REPO / "pyproject.toml").exists() or not (REPO / "src" / "karc").is_dir():
    raise SystemExit(f"E28: repository root misresolved as {REPO}")

CELL = "E28-BUDGET-SWEEP"
RAW = CELL_DIR / "raw"
RUN = REPO / "tmp" / "e28"
SESSIONS = [f"S{i:02d}" for i in range(1, 13)]
ARMS = ("karc-full", "rag-bm25")

CELLS = {
    "o025": {"name": "E28-O-025", "leg": "B", "budget_pct": 2.5,
             "budget_tokens": 878, "control": "pi-O-g", "control_leg": "B"},
    "o10": {"name": "E28-O-10", "leg": "B", "budget_pct": 10,
            "budget_tokens": 3512, "control": "pi-O-g", "control_leg": "B"},
    "o20": {"name": "E28-O-20", "leg": "B", "budget_pct": 20,
            "budget_tokens": 7025, "control": "pi-O-g", "control_leg": "B"},
    "a10": {"name": "E28-A-10", "leg": "A", "budget_pct": 10,
            "budget_tokens": 3512, "control": "pi-A-g", "control_leg": "A"},
}
ORDER = ["o025", "o10", "o20", "a10"]
CONTROL_BUDGET_PCT = 5
CONTROL_BUDGET_TOKENS = 1756
BATCH_CAP_USD = 10.0

E18_ANALYSIS = REPO / "docs" / "experiments" / "E18-PI-REVERSAL" / "raw" / "analysis.json"
E27_ANALYSIS = REPO / "docs" / "experiments" / "E27-BUDGET-10PCT" / "raw" / "analysis.json"

# ------------------------------------------------------------- E18 import

E18_ANALYZER = (REPO / "docs" / "experiments" / "E18-PI-REVERSAL" / "scripts"
                / "analyze_e18.py")
if not E18_ANALYZER.exists():
    raise SystemExit(f"E28: E18 analyzer not found at {E18_ANALYZER}")
_spec = importlib.util.spec_from_file_location("karc_e18_analyzer", E18_ANALYZER)
A = importlib.util.module_from_spec(_spec)
sys.modules["karc_e18_analyzer"] = A
_spec.loader.exec_module(A)
E18_ANALYZER_SHA256 = hashlib.sha256(E18_ANALYZER.read_bytes()).hexdigest()


def _point_cell(tag: str, units: list[dict], matrix) -> tuple[dict, dict]:
    """One cell's crosscheck and statistics, through E18's own functions."""
    spec = CELLS[tag]
    leg = spec["leg"]
    A.RUN = RUN
    A.RAW = RAW / tag
    A.RAW.mkdir(parents=True, exist_ok=True)
    A.NAMESPACE[leg] = tag
    check = A.crosscheck(leg, units)
    stats = A.leg_statistics(leg, units, matrix)
    return check, stats


def _judge(tag: str, stats: dict, check: dict) -> dict:
    """The preregistered judgments for one cell.  Only B3 is directional."""
    spec = CELLS[tag]
    gr = stats["gross_ratio_karc_over_rag"]
    pr = stats["priced_ratio_karc_over_rag"]
    gd = stats["paired_mean_session_gross_difference"]
    pd = stats["paired_mean_session_priced_usd_difference"]
    d = stats["component_delta_karc_minus_rag"]
    b3 = "PASS" if (d["cache_read"] > 0 and d["creation"] < 0) else "FAIL"
    mu_star = stats["reversal_threshold_mu_w_star"]
    mu_obs = stats["observed_mu_w"]
    return {
        "cell": spec["name"],
        "budget_pct": spec["budget_pct"],
        "budget_tokens": spec["budget_tokens"],
        "B1_gross_ratio": {
            "predicted": False,
            "quantity": "paired gross-input ratio karc/rag (ratio of session sums)",
            "ratio": gr["point"], "ci95": [gr["ci95_lo"], gr["ci95_hi"]],
            "ci_entirely_above_one": gr["entirely_above_one"],
            "ci_entirely_below_one": gr["entirely_below_one"],
            "excludes_parity": gr["entirely_above_one"] or gr["entirely_below_one"],
            "paired_mean_session_difference": gd["point"],
            "paired_mean_session_difference_ci95": [gd["ci95_lo"], gd["ci95_hi"]],
            "difference_ci_excludes_zero": gd["excludes_zero"],
        },
        "B2_priced_ratio": {
            "predicted": False,
            "quantity": "paired priced cost ratio karc/rag (ratio of session sums)",
            "ratio": pr["point"], "ci95": [pr["ci95_lo"], pr["ci95_hi"]],
            "ci_entirely_above_one": pr["entirely_above_one"],
            "ci_entirely_below_one": pr["entirely_below_one"],
            "excludes_parity": pr["entirely_above_one"] or pr["entirely_below_one"],
            "paired_mean_session_usd_difference": pd["point"],
            "paired_mean_session_usd_difference_ci95": [pd["ci95_lo"], pd["ci95_hi"]],
            "usd_difference_ci_excludes_zero": pd["excludes_zero"],
        },
        "B3_component_allocation_sign": {
            "predicted": True,
            "criterion": "delta_cache_read > 0 AND delta_creation < 0",
            "verdict": b3,
            "delta_cache_read": d["cache_read"],
            "delta_creation": d["creation"],
            "delta_uncached": d["uncached"],
            "delta_output": d["output"],
            "delta_gross": d["gross"],
        },
        "B4_break_even_cache_write_price": {
            "predicted": False,
            "mu_w_star": mu_star,
            "observed_mu_w": mu_obs,
            "distance_observed_minus_star": (None if mu_star is None
                                             else mu_obs - mu_star),
            "observed_exceeds_star": (None if mu_star is None
                                      else mu_obs > mu_star),
            "reading": ("mu_w* is the cache-write multiple at which the priced "
                        "ordering would flip, computed from this cell's own "
                        "component totals; the observed multiple is the actual "
                        "rate card's write/input ratio"),
        },
        "hold_fired": check["hold_fired"],
    }


def _component_paired_ci(units: list[dict], matrix) -> dict:
    """Per-component paired session-difference intervals (karc - rag).

    E18's `leg_statistics` reports the component deltas as window totals; the
    preregistration also asks for the interval on each, so the same bootstrap
    machinery is applied to the per-session paired differences here.
    """
    comp = A.per_session_components(units)
    out = {}
    for key in ("uncached", "creation", "cache_read", "output", "gross"):
        diffs = {s: float(comp["karc-full"][s][key]) - float(comp["rag-bm25"][s][key])
                 for s in SESSIONS}
        out[key] = A.paired_bootstrap_mean(diffs, matrix)
    return out


def _per_session_ratios(stats: dict) -> dict:
    gk = stats["per_session_gross"]["karc-full"]
    gr = stats["per_session_gross"]["rag-bm25"]
    pk = stats["per_session_priced_usd"]["karc-full"]
    pv = stats["per_session_priced_usd"]["rag-bm25"]
    return {s: {"gross_ratio": gk[s] / gr[s], "priced_ratio": pk[s] / pv[s]}
            for s in SESSIONS}


def _mechanism_table(stats: dict) -> dict:
    """Per turn position, per arm: cache creation and cache read means.

    The question this answers: does the retrieval arm pay creation on EVERY turn
    in proportion to the budget (its plan is new bytes each turn), while the
    resident arm pays it once at the head of the session (its resident set is
    constant within a session)?
    """
    out = {}
    for arm in ARMS:
        prof = stats["per_position_profile"][arm]
        out[arm] = {pos: {
            "creation_mean": prof[pos]["creation_mean"],
            "read_mean": prof[pos]["read_mean"],
            "gross_mean": prof[pos]["gross_mean"],
            "api_calls_mean": prof[pos]["api_calls_mean"],
        } for pos in sorted(prof, key=int)}
        first = out[arm]["1"]["creation_mean"]
        rest = [out[arm][str(p)]["creation_mean"] for p in range(2, 9)]
        out[arm]["_summary"] = {
            "creation_position_1": first,
            "creation_positions_2_to_8_mean": statistics.mean(rest),
            "creation_positions_2_to_8_min": min(rest),
            "creation_positions_2_to_8_max": max(rest),
            "creation_concentrated_at_head":
                first > 2.0 * statistics.mean(rest) if rest else None,
        }
    return out


def _resident_trajectory(tag: str) -> dict | None:
    path = RAW / tag / "build.json"
    if not path.exists():
        return None
    build = json.loads(path.read_text(encoding="utf-8"))
    return {
        "budget_tokens": CELLS[tag]["budget_tokens"],
        "per_session": [{
            "session_id": row["session_id"],
            "resident_versions": row["resident_versions"],
            "resident_tokens": row["resident_tokens"],
            "budget_utilization": row["budget_utilization"],
        } for row in build["resident_ramp"]],
        "control_c05_per_session": [{
            "session_id": row["session_id"],
            "resident_versions": row["resident_versions"],
            "resident_tokens": row["resident_tokens"],
            "budget_utilization": row["budget_utilization"],
        } for row in build["resident_ramp_c05_control"]],
        "final_utilization": build["resident_ramp"][-1]["budget_utilization"],
        "rag_plan_tokens_mean": build["retrieval_audit"]["plan_tokens_mean"],
        "note": ("a growth window: the resident set is not expected to fill. The "
                 "trajectory is an INPUT state quantity, fixed by the schedule "
                 "and the model-free policy replay before any model call."),
    }


def _controls() -> dict:
    """The two control cells, read out of E18's committed analysis.json."""
    if not E18_ANALYSIS.exists():
        return {}
    data = json.loads(E18_ANALYSIS.read_text(encoding="utf-8"))
    out = {}
    for leg, name in (("B", "pi-O-g"), ("A", "pi-A-g")):
        st = data["legs"][leg]
        out[name] = {
            "source": "docs/experiments/E18-PI-REVERSAL/raw/analysis.json",
            "leg": leg,
            "budget_pct": CONTROL_BUDGET_PCT,
            "budget_tokens": CONTROL_BUDGET_TOKENS,
            "window": "S01-S12 (growth)",
            "gross_ratio": st["gross_ratio_karc_over_rag"]["point"],
            "gross_ci95": [st["gross_ratio_karc_over_rag"]["ci95_lo"],
                           st["gross_ratio_karc_over_rag"]["ci95_hi"]],
            "priced_ratio": st["priced_ratio_karc_over_rag"]["point"],
            "priced_ci95": [st["priced_ratio_karc_over_rag"]["ci95_lo"],
                            st["priced_ratio_karc_over_rag"]["ci95_hi"]],
            "delta_cache_read": st["component_delta_karc_minus_rag"]["cache_read"],
            "delta_creation": st["component_delta_karc_minus_rag"]["creation"],
            "delta_uncached": st["component_delta_karc_minus_rag"]["uncached"],
            "delta_output": st["component_delta_karc_minus_rag"]["output"],
            "delta_gross": st["component_delta_karc_minus_rag"]["gross"],
            "mu_w_star": st["reversal_threshold_mu_w_star"],
            "observed_mu_w": st["observed_mu_w"],
            "component_totals": st["component_totals"],
            "priced_totals_usd": st["priced_totals_usd"],
            "per_session_gross": st["per_session_gross"],
            "per_session_priced_usd": st["per_session_priced_usd"],
            "per_position_profile": st["per_position_profile"],
        }
    return out


def _e27() -> dict:
    """E27's c = 10% steady-state cell, for batch question (iii)."""
    if not E27_ANALYSIS.exists():
        return {}
    data = json.loads(E27_ANALYSIS.read_text(encoding="utf-8"))
    st = data.get("statistics")
    if not isinstance(st, dict) or "gross_ratio_karc_over_rag" not in st:
        return {"unavailable": "could not locate E27's statistics block"}
    return {
        "source": "docs/experiments/E27-BUDGET-10PCT/raw/analysis.json",
        "budget_pct": 10, "budget_tokens": 3512,
        "window": "S13-S24 (steady state)",
        "schedule_seed": 4352, "sessions_built": 24,
        "component_delta_paired_ci": st.get("component_delta_paired_ci"),
        "gross_ratio": st["gross_ratio_karc_over_rag"]["point"],
        "gross_ci95": [st["gross_ratio_karc_over_rag"]["ci95_lo"],
                       st["gross_ratio_karc_over_rag"]["ci95_hi"]],
        "priced_ratio": st["priced_ratio_karc_over_rag"]["point"],
        "priced_ci95": [st["priced_ratio_karc_over_rag"]["ci95_lo"],
                        st["priced_ratio_karc_over_rag"]["ci95_hi"]],
        "delta_cache_read": st["component_delta_karc_minus_rag"]["cache_read"],
        "delta_creation": st["component_delta_karc_minus_rag"]["creation"],
        "mu_w_star": st["reversal_threshold_mu_w_star"],
        "component_totals": st["component_totals"],
    }


def _batch_questions(judged: dict, controls: dict, e27: dict) -> dict:
    """The three questions §4.1 registers as measured, not predicted."""
    openai_cells = [(2.5, "o025"), (10, "o10"), (20, "o20")]
    sweep = []
    if "pi-O-g" in controls:
        c = controls["pi-O-g"]
        sweep.append({"budget_pct": 5, "cell": "pi-O-g (control, E18 leg B)",
                      "gross_ratio": c["gross_ratio"], "gross_ci95": c["gross_ci95"],
                      "priced_ratio": c["priced_ratio"], "priced_ci95": c["priced_ci95"]})
    for pct, tag in openai_cells:
        if tag not in judged:
            continue
        j = judged[tag]
        sweep.append({
            "budget_pct": pct, "cell": j["cell"],
            "gross_ratio": j["B1_gross_ratio"]["ratio"],
            "gross_ci95": j["B1_gross_ratio"]["ci95"],
            "priced_ratio": j["B2_priced_ratio"]["ratio"],
            "priced_ci95": j["B2_priced_ratio"]["ci95"],
        })
    sweep.sort(key=lambda row: row["budget_pct"])

    q1 = {
        "question": ("where the gross ratio sits relative to 1.0 across the "
                     "budget sweep, and whether at c = 20% it passes beyond "
                     "parity"),
        "sweep_openai": sweep,
        "at_20pct": None,
    }
    if "o20" in judged:
        b1 = judged["o20"]["B1_gross_ratio"]
        q1["at_20pct"] = {
            "gross_ratio": b1["ratio"], "ci95": b1["ci95"],
            "ci_entirely_below_one": b1["ci_entirely_below_one"],
            "ci_entirely_above_one": b1["ci_entirely_above_one"],
            "reading": ("'passes beyond parity' means the 95% interval lies "
                        "wholly below 1.0, i.e. the resident arm buys fewer "
                        "gross tokens than retrieval, not merely as many"),
        }
    q2 = {
        "question": "whether the cost ratio stays below 1.0 at c = 2.5%",
        "at_025pct": None,
    }
    if "o025" in judged:
        b2 = judged["o025"]["B2_priced_ratio"]
        q2["at_025pct"] = {
            "priced_ratio": b2["ratio"], "ci95": b2["ci95"],
            "ci_entirely_below_one": b2["ci_entirely_below_one"],
            "control_priced_ratio": controls.get("pi-O-g", {}).get("priced_ratio"),
            "control_ci95": controls.get("pi-O-g", {}).get("priced_ci95"),
        }
    q3 = {
        "question": ("the difference between E27 (steady state, S13-S24, "
                     "c = 10%) and E28-O-10 with the window held fixed"),
        "caveat": ("the two cells also differ in schedule seed (4352 vs 4200) "
                   "and sessions built (24 vs 12), so this is read, not "
                   "causally attributed to the window"),
        "e27": e27,
        "e28_o10": None,
    }
    if "o10" in judged:
        j = judged["o10"]
        q3["e28_o10"] = {
            "window": "S01-S12 (growth)", "schedule_seed": 4200,
            "sessions_built": 12,
            "gross_ratio": j["B1_gross_ratio"]["ratio"],
            "gross_ci95": j["B1_gross_ratio"]["ci95"],
            "priced_ratio": j["B2_priced_ratio"]["ratio"],
            "priced_ci95": j["B2_priced_ratio"]["ci95"],
            "delta_cache_read": j["B3_component_allocation_sign"]["delta_cache_read"],
            "delta_creation": j["B3_component_allocation_sign"]["delta_creation"],
            "mu_w_star": j["B4_break_even_cache_write_price"]["mu_w_star"],
        }
        if e27 and "gross_ratio" in e27:
            q3["difference_e28_minus_e27"] = {
                "gross_ratio": j["B1_gross_ratio"]["ratio"] - e27["gross_ratio"],
                "priced_ratio": j["B2_priced_ratio"]["ratio"] - e27["priced_ratio"],
                "mu_w_star": (None if e27.get("mu_w_star") is None
                              else j["B4_break_even_cache_write_price"]["mu_w_star"]
                              - e27["mu_w_star"]),
            }
    return {"i_gross_ordering_across_budgets": q1,
            "ii_cost_ratio_at_2_5_percent": q2,
            "iii_window_fixed_vs_E27": q3}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", default=",".join(ORDER))
    args = parser.parse_args()
    tags = [t for t in args.cells.split(",") if t in CELLS]

    matrix = A.resample_matrix()
    stats: dict[str, dict] = {}
    checks: dict[str, dict] = {}
    judged: dict[str, dict] = {}
    for tag in tags:
        spec = CELLS[tag]
        leg = spec["leg"]
        A.RUN = RUN
        A.NAMESPACE[leg] = tag
        units = A.load_units(leg)
        if not units:
            print(f"{spec['name']}: no units, skipped")
            continue
        out_dir = RAW / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        A.RAW = out_dir
        # E18's write_turns names the file by leg; rename to this cell's shape.
        written = A.write_turns(leg, units)
        target = out_dir / "turns.jsonl"
        if written != target:
            target.write_bytes(written.read_bytes())
            written.unlink()
        check, st = _point_cell(tag, units, matrix)
        (out_dir / "crosscheck.json").write_text(
            json.dumps(check, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8")
        stale = out_dir / f"crosscheck-{leg}.json"
        if stale.exists():
            stale.unlink()
        st["component_delta_paired_ci"] = _component_paired_ci(units, matrix)
        checks[tag] = check
        stats[tag] = st
        judged[tag] = _judge(tag, st, check)
        judged[tag]["B3_component_allocation_sign"]["paired_ci"] = {
            "cache_read": st["component_delta_paired_ci"]["cache_read"],
            "creation": st["component_delta_paired_ci"]["creation"],
        }
        print(f"{spec['name']}: turns {check['turns_executed']}/"
              f"{check['turns_expected']} hold_fired={check['hold_fired']} "
              f"standard_usd={check['cost']['standard_rate_card_usd']} "
              f"B3={judged[tag]['B3_component_allocation_sign']['verdict']}")

    controls = _controls()
    e27 = _e27()
    total_standard = sum(checks[t]["cost"]["standard_rate_card_usd"] for t in checks)
    total_pi = sum(checks[t]["cost"]["pi_reported_usd"] for t in checks)

    analysis = {
        "cell": CELL,
        "preregistration": "docs/experiments/E28-BUDGET-SWEEP/preregistration.md",
        "cells": {tag: {
            "name": CELLS[tag]["name"],
            "leg": CELLS[tag]["leg"],
            "budget_pct": CELLS[tag]["budget_pct"],
            "budget_tokens": CELLS[tag]["budget_tokens"],
            "control": CELLS[tag]["control"],
            "statistics": stats[tag],
            "per_session_ratios": _per_session_ratios(stats[tag]),
            "mechanism_by_turn_position": _mechanism_table(stats[tag]),
            "resident_utilization_trajectory": _resident_trajectory(tag),
        } for tag in stats},
        "judgments": judged,
        "controls_c05": controls,
        "e27_steady_state_c10": e27,
        "batch_questions": _batch_questions(judged, controls, e27),
        "budget": {
            "batch_cap_usd": BATCH_CAP_USD,
            "standard_rate_card_total_usd": round(total_standard, 6),
            "pi_reported_total_usd": round(total_pi, 6),
            "within_cap": total_pi <= BATCH_CAP_USD
                          and total_standard <= BATCH_CAP_USD,
            "note": ("pi_reported is what was actually charged at pi's catalog "
                     "rates; standard is the judged rate card, which is >= "
                     "catalog on both providers"),
        },
        "accounting_contract": {
            "gross": "input + cacheRead + cacheWrite",
            "why": ("pi usage.input excludes the cache buckets on both legs "
                    "(E17 §3, E17B §2); treating it as provider raw input_tokens "
                    "under-counts gross by ~5.5x"),
        },
        "_meta": {
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "bootstrap_seed": A.BOOTSTRAP_SEED,
            "bootstrap_resamples": A.BOOTSTRAP_RESAMPLES,
            "imported_analyzer": str(E18_ANALYZER.relative_to(REPO)),
            "imported_analyzer_sha256": E18_ANALYZER_SHA256,
        },
    }
    (RAW / "analysis.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8")
    print(json.dumps(analysis["judgments"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
