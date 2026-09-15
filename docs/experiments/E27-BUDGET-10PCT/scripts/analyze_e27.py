"""E27-BUDGET-10PCT — crosscheck, statistics and the preregistered judgment.

The statistics are E23's, imported rather than copied: `resample_matrix`,
`percentile_ci`, `paired_bootstrap_mean`, `paired_bootstrap_ratio_of_sums`,
`priced_usd`, `per_session_components`, `rho_statistics`, `write_turns`,
`e19_leg_b_contrast` and `judge` all come out of
`docs/experiments/E23-REUSE-DENSITY/scripts/analyze_e23.py`, which is itself
E19's procedure verbatim (session-unit paired percentile bootstrap, seed 1313,
10,000 resamples, one shared index matrix so every paired comparison uses common
random numbers).  Only the cell's addressing is overridden — one cell instead of
two rho, `tmp/e27` instead of `tmp/e23`, this cell's raw directory.

Emitted, the same shapes E23 emitted:

    raw/turns-b10.jsonl      one row per measurement turn, closure included
    raw/crosscheck-b10.json  instrumentation contract + hold conditions H1..H14
    raw/analysis.json        component_totals, delta karc-rag, gross/priced
                             ratio bootstraps, per-session ratios, mu_w_star

What is NOT imported is `crosscheck`.  E23's version hard-codes its own pi
session-id prefix and its own sibling-cell list, and this cell has one extra
gate (H14, axis purity) and one condition E23 did not need to disclose (the
window shift).  So the bookkeeping is re-stated here, with the same measurements
in the same key names, while every judged number still comes from the imported
statistics.

Preregistration §4 registers exactly one directional prediction:

    B3  delta_cache_read(karc - rag) > 0  AND  delta_creation(karc - rag) < 0

B1 (gross ratio), B2 (priced ratio) and B4 (mu_w*) are measured and reported
with intervals and carry `predicted: false`; nothing here turns them into a
verdict.  mu_w* is recomputed from THIS cell's own component totals.

No model call, no network.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
CELL_DIR = HERE.parents[1]
REPO = HERE.parents[4]
if not (REPO / "pyproject.toml").exists() or not (REPO / "src" / "karc").is_dir():
    raise SystemExit(f"E27: repository root misresolved as {REPO}")

CELL = "E27-BUDGET-10PCT"
TAG = "b10"
BUDGET_PCT = 10
BUDGET_TOKENS = 3512
REUSE_FACTOR = 0.50
RAW = CELL_DIR / "raw"
RUN = REPO / "tmp" / "e27"
BUDGET_CAP_USD = 10.0
NAMESPACE = ""
ARMS = ("karc-full", "rag-bm25")
N_SESSIONS = 12
OPENAI_CACHE_MIN_PREFIX = 1024
SIBLING_CELLS = ("e17", "e17b", "e18", "e19", "e21", "e23")

# ------------------------------------------------------------- E23 import

E23_ANALYZER = (REPO / "docs" / "experiments" / "E23-REUSE-DENSITY" / "scripts"
                / "analyze_e23.py")
if not E23_ANALYZER.exists():
    raise SystemExit(f"E27: E23 analyzer not found at {E23_ANALYZER}")
_spec = importlib.util.spec_from_file_location("karc_e23_analyzer", E23_ANALYZER)
A = importlib.util.module_from_spec(_spec)
sys.modules["karc_e23_analyzer"] = A
_spec.loader.exec_module(A)
E23_ANALYZER_SHA256 = __import__("hashlib").sha256(
    E23_ANALYZER.read_bytes()).hexdigest()


def _root(tag: str) -> Path:
    return RUN / f"budget{BUDGET_PCT}{NAMESPACE}"


def _session_store_usage(tag: str, arm: str, session: str) -> list[dict] | None:
    """Second usage-acquisition route: pi's own session JSONL.  R-9: opened to
    read `usage` numbers, never to copy text.  E23's function with this cell's
    session-id pattern."""
    matches = sorted((_root(tag) / "sessions").glob(
        f"*_e27b{BUDGET_PCT}{NAMESPACE}-{arm}-{session}.jsonl"))
    if not matches:
        return None
    rows = []
    for line in matches[-1].read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        message = entry.get("message")
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        usage = message.get("usage")
        if not isinstance(usage, dict):
            continue
        rows.append({
            "input": usage.get("input"), "output": usage.get("output"),
            "cacheRead": usage.get("cacheRead"),
            "cacheWrite": usage.get("cacheWrite"),
            "cacheWrite1h": usage.get("cacheWrite1h"),
            "stopReason": message.get("stopReason"),
            "rawStopReason": message.get("rawStopReason"),
        })
    return rows


def _apply_overrides(window: list[str]) -> None:
    A.RHOS = {TAG: REUSE_FACTOR}
    A.RUN = RUN
    A.RAW = RAW
    A.CELL_DIR = CELL_DIR
    A.NAMESPACE = NAMESPACE
    A.BUDGET_CAP_USD = BUDGET_CAP_USD
    A.WINDOW = {TAG: list(window)}
    A.rho_root = _root
    A.session_store_usage = _session_store_usage


# ------------------------------------------------------------- crosscheck

def crosscheck(units: list[dict], steady: dict, build: dict | None) -> dict:
    """E23's instrumentation crosscheck, re-stated for this cell's namespace and
    extended with H14.  Every quantity is measured, none carried over."""
    sessions = A.WINDOW[TAG]
    calls = [c for unit in units for row in unit["rows"] for c in row["calls"]]
    rows = [row for unit in units for row in unit["rows"]]
    closure_fail = [c for c in calls if not c["closure_total_eq_four_sum"]]
    write1h_fail = [c for c in calls if not c["write1h_le_write"]]
    inclusive_true = [c for c in calls
                      if c["inclusive_hypothesis_total_eq_input_plus_output"]]
    discriminating = [c for c in calls if c["cacheRead"] + c["cacheWrite"] > 0]
    input_ge = [c for c in discriminating if c["input_ge_read_plus_write"]]

    per_unit = []
    jsonl_agree_all = True
    for unit in units:
        stdout_rows = [{
            "input": c["input"], "output": c["output"],
            "cacheRead": c["cacheRead"], "cacheWrite": c["cacheWrite"],
            "cacheWrite1h": (c["cacheWrite1h"] if c["cacheWrite1h_key_present"]
                             else None),
            "stopReason": c["stopReason"], "rawStopReason": c["rawStopReason"],
        } for row in unit["rows"] for c in row["calls"]]
        store = _session_store_usage(TAG, unit["arm"], unit["session_id"])
        fields = ["input", "output", "cacheRead", "cacheWrite", "cacheWrite1h",
                  "stopReason", "rawStopReason"]
        agree = (store is not None and len(store) == len(stdout_rows) and all(
            all(a.get(f) == b.get(f) for f in fields)
            for a, b in zip(store, stdout_rows)))
        jsonl_agree_all = jsonl_agree_all and agree
        tp = unit["three_paths"]
        if unit["arm"] == "karc-full":
            three_agree = (tp["pi_tool_execution_start"] == tp["bridge_audit_lines"]
                           == tp["db_ingest_observations_mcp"])
        else:
            three_agree = (tp["pi_tool_execution_start"] == 0
                           == tp["bridge_audit_lines"]
                           and tp["db_ingest_observations_mcp"] is None)
        per_unit.append({
            "arm": unit["arm"], "session_id": unit["session_id"],
            "turns_executed": unit["turns_executed"],
            "three_paths": tp, "three_paths_agree": three_agree,
            "session_store_usage_rows": (None if store is None else len(store)),
            "stdout_usage_rows": len(stdout_rows),
            "session_store_equals_stdout": agree,
        })

    refusal_rows = [{
        "arm": r["arm"], "session_id": r["session_id"], "position": r["position"],
        "signals": {k: v for k, v in r["refusal_signals"].items() if v},
    } for r in rows if (r["refusal_signals"]["output_tokens_zero"]
                        or r["refusal_signals"]["errorMessage_present"]
                        or r["refusal_signals"]["content_block_absent"]
                        or r["refusal_signals"]["raw_stop_refusal"]
                        or r["refusal_signals"]["normalized_stop_error"])]

    sub_threshold = [c for c in calls if c["sub_threshold_openai"]]
    eligible = [c for c in calls if c["cache_eligible_openai"]]
    eligible_dead = [c for c in eligible
                     if c["cacheRead"] == 0 and c["cacheWrite"] == 0]
    first_eligible_cold = []
    for unit in units:
        for row in unit["rows"]:
            for c in row["calls"]:
                if c["cacheRead"] == 0 and c["cache_eligible_openai"]:
                    first_eligible_cold.append({
                        "arm": unit["arm"], "session_id": unit["session_id"],
                        "position": row["position"],
                        "cacheWrite": c["cacheWrite"],
                        "prefix_tokens": c["prefix_tokens"]})
                    break
            else:
                continue
            break
    duplicate_request_hits = [{
        "arm": r["arm"], "session_id": r["session_id"], "position": r["position"],
        "call_index": i, "prefix_tokens": c["prefix_tokens"],
        "cacheRead": c["cacheRead"],
    } for r in rows for i, c in enumerate(r["calls"])
        if c["cacheWrite"] == 0 and c["cacheRead"] > 0]
    w1h_key_absent = [c for c in calls if not c["cacheWrite1h_key_present"]]

    residuals = [abs(float(c["pi_rate_card_modelled_usd"])
                     - float(c["pi_cost_total"] or 0.0)) for c in calls]

    turns_expected = len(sessions) * len(ARMS) * 8
    turns_done = len(rows)
    incomplete = [{"arm": u["arm"], "session_id": u["session_id"],
                   "turns_executed": u["turns_executed"]}
                  for u in units if u["turns_executed"] != 8]

    priced_total = sum(A.priced_usd(
        r["usage_turn"]["input"],
        max(0, r["usage_turn"]["cacheWrite"] - r["usage_turn"]["cacheWrite1h"]),
        r["usage_turn"]["cacheWrite1h"], r["usage_turn"]["cacheRead"],
        r["usage_turn"]["output"]) for r in rows)
    pi_total = sum(float(r["usage_turn"]["pi_cost_total"]) for r in rows)

    session_prefix = f"e27b{BUDGET_PCT}{NAMESPACE}-"
    ns_rows = [{
        "arm": unit["arm"], "session_id": unit["session_id"],
        "pi_session_id_prefix_ok": unit.get("pi_session_id", "").startswith(
            session_prefix),
        "pi_session_id_sha256": unit.get("pi_session_id_sha256"),
    } for unit in units]
    ns_bad = [r for r in ns_rows if not r["pi_session_id_prefix_ok"]]
    root = _root(TAG)
    sibling_roots = {name: str(REPO / "tmp" / name) for name in SIBLING_CELLS}

    hold = {
        "H1_three_path_mismatch": [u for u in per_unit
                                   if not u["three_paths_agree"]],
        "H2_refusal_or_zero_output": refusal_rows,
        "H3_closure_violation": len(closure_fail),
        "H4_first_eligible_cold_write_zero": [
            f for f in first_eligible_cold if f["cacheWrite"] == 0],
        "H5_session_completion_failure_rate": round(
            1.0 - turns_done / turns_expected, 6),
        "H6_priced_usd_this_cell": round(priced_total, 6),
        "H6_cap_usd": BUDGET_CAP_USD,
        "H7_user_pi_mtime": "judged outside this script (mtime phase)",
        "H8_eligible_call_with_no_cache_activity": len(eligible_dead),
        "H9_cold_start_misattribution_calls": len(duplicate_request_hits),
        "H10_namespace_collision": {
            "namespace": NAMESPACE,
            "pi_session_id_prefix": session_prefix,
            "units_with_wrong_prefix": len(ns_bad),
            "run_root": str(root),
            "run_root_under_tmp_e27": str(root).startswith(str(RUN)),
            "distinct_from_sibling_cells": all(
                not str(root).startswith(p) for p in sibling_roots.values()),
            "sibling_roots": sibling_roots,
        },
        "H11_steady_state_not_reached": bool(steady.get("H11_fired")),
        "H12_gold_containment_not_one": (
            None if not build else bool(build["H12_gold_containment"]["fired"])),
        "H13_degenerate_branch": (
            None if not build else bool(build["H13_non_degenerate_branch"]["fired"])),
        "H14_axis_purity_violated": (
            None if not build else bool(build["H14_axis_purity"]["fired"])),
    }
    fired = []
    if hold["H1_three_path_mismatch"]:
        fired.append("H1")
    if hold["H2_refusal_or_zero_output"]:
        fired.append("H2")
    if hold["H3_closure_violation"]:
        fired.append("H3")
    if hold["H4_first_eligible_cold_write_zero"]:
        fired.append("H4")
    if hold["H5_session_completion_failure_rate"] > 0.10:
        fired.append("H5")
    if hold["H6_priced_usd_this_cell"] > BUDGET_CAP_USD:
        fired.append("H6")
    if hold["H8_eligible_call_with_no_cache_activity"]:
        fired.append("H8")
    if hold["H9_cold_start_misattribution_calls"]:
        fired.append("H9")
    h10 = hold["H10_namespace_collision"]
    if (h10["units_with_wrong_prefix"] or not h10["run_root_under_tmp_e27"]
            or not h10["distinct_from_sibling_cells"]):
        fired.append("H10")
    if hold["H11_steady_state_not_reached"]:
        fired.append("H11")
    if hold["H12_gold_containment_not_one"]:
        fired.append("H12")
    if hold["H13_degenerate_branch"]:
        fired.append("H13")
    if hold["H14_axis_purity_violated"]:
        fired.append("H14")

    return {
        "cell": CELL, "budget_pct": BUDGET_PCT, "budget_tokens": BUDGET_TOKENS,
        "reuse_factor": REUSE_FACTOR,
        "turns_expected": turns_expected, "turns_executed": turns_done,
        "completion_rate": round(turns_done / turns_expected, 6),
        "incomplete_units": incomplete,
        "api_calls": len(calls),
        "inclusion_relation": {
            "discriminating_calls": len(discriminating),
            "exclusive_hypothesis_holds": sum(
                1 for c in discriminating if c["closure_total_eq_four_sum"]),
            "inclusive_hypothesis_holds": len(
                [c for c in discriminating
                 if c["inclusive_hypothesis_total_eq_input_plus_output"]]),
            "input_ge_read_plus_write_count": len(input_ge),
            "note": "gross = input + cacheRead + cacheWrite (E17 §3 / E17B §2); "
                    "measured here, not carried over",
        },
        "closure": {
            "total_eq_four_sum_pass": len(calls) - len(closure_fail),
            "total_eq_four_sum_fail": len(closure_fail),
            "write1h_le_write_fail": len(write1h_fail),
            "inclusive_true_anywhere": len(inclusive_true),
        },
        "cacheWrite1h": {
            "key_present_calls": len(calls) - len(w1h_key_absent),
            "key_absent_calls": len(w1h_key_absent),
            "note": "the OpenAI leg has no cacheWrite1h key at all (E17B §4)",
        },
        "openai_cache_threshold": {
            "min_prefix_tokens": OPENAI_CACHE_MIN_PREFIX,
            "sub_threshold_calls": len(sub_threshold),
            "sub_threshold_prefix_tokens": sorted(
                {c["prefix_tokens"] for c in sub_threshold}),
            "eligible_calls": len(eligible),
            "eligible_calls_with_no_cache_activity": len(eligible_dead),
            "first_eligible_cold_calls": first_eligible_cold,
        },
        "stop_reasons": {
            "normalized": dict(sorted({
                str(c["stopReason"]): sum(
                    1 for x in calls if str(x["stopReason"]) == str(c["stopReason"]))
                for c in calls}.items())),
            "provider_raw": dict(sorted({
                str(c["rawStopReason"]): sum(
                    1 for x in calls
                    if str(x["rawStopReason"]) == str(c["rawStopReason"]))
                for c in calls}.items())),
        },
        "per_unit": per_unit,
        "session_store_equals_stdout_all_units": jsonl_agree_all,
        "H9_cold_start_assert": {
            "what": "calls with cacheWrite == 0 and cacheRead > 0, i.e. served a "
                    "prefix written by an earlier identical request outside this "
                    "dataset (E18 §5 discard 2)",
            "calls": len(duplicate_request_hits),
            "tokens_mis_bucketed_as_read": sum(
                h["cacheRead"] for h in duplicate_request_hits),
            "by_arm": {arm: {
                "calls": sum(1 for h in duplicate_request_hits if h["arm"] == arm),
                "tokens": sum(h["cacheRead"] for h in duplicate_request_hits
                              if h["arm"] == arm)} for arm in ARMS},
            "clean": len(duplicate_request_hits) == 0,
            "detail": duplicate_request_hits,
        },
        "pi_rate_card_reverse_verification": {
            "calls": len(calls),
            "max_abs_residual_usd": max(residuals) if residuals else None,
            "exact_within_1e_9_usd": (max(residuals) < 1e-9) if residuals else None,
            "pi_catalog_usd_per_million": A.PI_CATALOG_RATES,
        },
        "cost": {
            "pi_reported_usd": round(pi_total, 6),
            "standard_rate_card_usd": round(priced_total, 6),
            "rate_card_source": A.RATE_CARD["source"],
        },
        "steady_state": steady,
        "measurement_window": {"sessions": sessions, "arms": list(ARMS)},
        "incidental_grade_diagnostic": {
            "note": "NOT a claim (preregistration §8 forbids the accuracy axis). "
                    "Recorded so that a silently non-functioning arm is "
                    "detectable.",
            "per_arm": {arm: {
                "passed": sum(1 for r in rows if r["arm"] == arm
                              and r["grade_incidental"]["passed"]),
                "n": sum(1 for r in rows if r["arm"] == arm),
                "passed_at_position_1": sum(
                    1 for r in rows if r["arm"] == arm and r["position"] == 1
                    and r["grade_incidental"]["passed"]),
                "n_position_1": sum(1 for r in rows if r["arm"] == arm
                                    and r["position"] == 1),
                "passed_positions_2_to_8": sum(
                    1 for r in rows if r["arm"] == arm and r["position"] > 1
                    and r["grade_incidental"]["passed"]),
                "n_positions_2_to_8": sum(1 for r in rows if r["arm"] == arm
                                          and r["position"] > 1),
            } for arm in ARMS},
        },
        "hold_conditions": hold,
        "hold_fired": fired,
        "_meta": {"generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                    time.gmtime())},
    }


# ------------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ns", default="")
    args = parser.parse_args()

    global NAMESPACE
    NAMESPACE = args.ns

    steady_path = RAW / f"steady-{TAG}.json"
    if not steady_path.exists():
        raise SystemExit("E27: run the steady phase first")
    steady = json.loads(steady_path.read_text(encoding="utf-8"))
    if not steady["steady_state_reached"]:
        raise SystemExit("E27: H11 fired; not measured")
    window = list(steady["measurement_window"])
    if len(window) != N_SESSIONS:
        raise SystemExit(f"E27: window is not {N_SESSIONS} sessions")

    build_path = RAW / "build.json"
    build = (json.loads(build_path.read_text(encoding="utf-8"))
             if build_path.exists() else None)

    _apply_overrides(window)

    units = A.load_units(TAG, window)
    if not units:
        raise SystemExit("E27: no unit files found; run the wave first")
    A.write_turns(TAG, units)
    check = crosscheck(units, steady, build)
    (RAW / f"crosscheck-{TAG}.json").write_text(
        json.dumps(check, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8")

    matrix = A.resample_matrix()
    stats = A.rho_statistics(TAG, units, matrix)
    if stats.get("incomplete"):
        raise SystemExit("E27: the measurement window is incomplete "
                         f"({json.dumps(stats.get('arms_present'))}); "
                         "no partial judgment is emitted")
    verdict = A.judge({TAG: stats}, {TAG: check})

    st = stats
    delta = st.get("component_delta_karc_minus_rag", {})
    hypotheses = {
        "naming_note": ("B1..B4 are the requesting brief's H1..H4, renamed so "
                        "they do not collide with hold conditions H1..H14"),
        "B1_gross_input_ratio": {
            "predicted_direction": False,
            "point": st["gross_ratio_karc_over_rag"]["point"],
            "ci95": [st["gross_ratio_karc_over_rag"]["ci95_lo"],
                     st["gross_ratio_karc_over_rag"]["ci95_hi"]],
            "entirely_above_one": st["gross_ratio_karc_over_rag"]["entirely_above_one"],
            "entirely_below_one": st["gross_ratio_karc_over_rag"]["entirely_below_one"],
        },
        "B2_priced_cost_ratio": {
            "predicted_direction": False,
            "point": st["priced_ratio_karc_over_rag"]["point"],
            "ci95": [st["priced_ratio_karc_over_rag"]["ci95_lo"],
                     st["priced_ratio_karc_over_rag"]["ci95_hi"]],
            "entirely_above_one": st["priced_ratio_karc_over_rag"]["entirely_above_one"],
            "entirely_below_one": st["priced_ratio_karc_over_rag"]["entirely_below_one"],
            "cost_parity_reference": 1.0,
        },
        "B3_component_allocation_sign": {
            "predicted_direction": True,
            "criterion": "delta_cache_read > 0 AND delta_creation < 0",
            "delta_cache_read": delta.get("cache_read"),
            "delta_creation": delta.get("creation"),
            "delta_uncached": delta.get("uncached"),
            "delta_output": delta.get("output"),
            "paired_ci": st.get("component_delta_paired_ci"),
            "verdict": (verdict.get(TAG) or {}).get(
                "S1_component_signature", {}).get("verdict"),
        },
        "B4_break_even_write_price": {
            "predicted_direction": False,
            "mu_w_star": st.get("reversal_threshold_mu_w_star"),
            "observed_mu_w": st.get("observed_mu_w"),
            "distance_to_actual": (
                None if st.get("reversal_threshold_mu_w_star") is None
                else st["observed_mu_w"] - st["reversal_threshold_mu_w_star"]),
            "formula": "mu_w* = -(dU + mu_r*dR + mu_o*dO) / dW, from this cell's "
                       "own component totals",
        },
    }

    e19 = A.e19_leg_b_contrast()
    analysis = {
        "cell": CELL,
        "preregistration": f"docs/experiments/{CELL}/preregistration.md",
        "manipulated_axis": {
            "name": "resident budget",
            "operative_knob": "e5_killgate._manifest_at_budget(manifest, budget_pct)",
            "value_measured": {"budget_pct": BUDGET_PCT,
                               "budget_tokens": BUDGET_TOKENS},
            "control_point": {"budget_pct": 5, "budget_tokens": 1756},
            "fixed": ["harness pi 0.84.3", "provider openai",
                      "model gpt-5.6-luna", "arms karc-full / rag-bm25",
                      "reuse factor 0.50", "schedule seed 4352",
                      "24-session build", "rate card", "statistics",
                      "12 sessions x 8 turns x 2 arms"],
            "not_fixed": ["measurement window position"],
        },
        "measurement_window": window,
        "window_shift_vs_e19_leg_b": steady.get("window_shift_vs_e19_leg_b"),
        "statistics": stats,
        "registered_hypotheses": hypotheses,
        "judgments": verdict,
        "e19_leg_b_contrast_c05": e19,
        "accounting_contract": {
            "gross": "input + cacheRead + cacheWrite",
            "why": "pi usage.input excludes cache buckets on both legs "
                   "(E17 §3, E17B §2); measured again in this cell",
        },
        "not_pooled_with_the_eight_cells": (
            "preregistration §8: a different capacity grid point, so it does not "
            "enter the abstract's range figures, §4's tables/figures or the "
            "claims ledger"),
        "_meta": {"generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                    time.gmtime()),
                  "bootstrap_seed": A.BOOTSTRAP_SEED,
                  "bootstrap_resamples": A.BOOTSTRAP_RESAMPLES,
                  "namespace": NAMESPACE,
                  "imported_analyzer": str(E23_ANALYZER.relative_to(REPO)),
                  "imported_analyzer_sha256": E23_ANALYZER_SHA256},
    }
    (RAW / "analysis.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8")
    print(f"turns {check['turns_executed']}/{check['turns_expected']} "
          f"hold_fired={check['hold_fired']} "
          f"usd={check['cost']['standard_rate_card_usd']}")
    print(json.dumps(hypotheses, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
