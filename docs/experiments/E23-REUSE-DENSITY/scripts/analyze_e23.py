"""E23-REUSE-DENSITY — crosscheck, statistics and the preregistered judgment.

Derived from `docs/experiments/E19-STEADY-STATE/scripts/analyze_e19.py`.  The
statistics are byte-for-byte the same procedure (session-unit paired percentile
bootstrap, seed 1313, 10,000 resamples, one shared index matrix so every paired
comparison uses common random numbers).  What changes is the axis: E19's two
provider legs become this cell's two reuse densities, and only ONE prediction is
registered.

Reads the unit files produced by run_e23_reuse_density.py (under tmp/e23, plus
the pi session stores for the second usage-acquisition route) and writes:

    raw/turns-<rho>.jsonl      one row per measurement turn, closure included
    raw/crosscheck-<rho>.json  instrumentation contract + HOLD conditions
    raw/analysis.json          statistics and the S1 judgment

No model call, no network.

Accounting contract, inherited and all of it measured:
    gross = input + cacheRead + cacheWrite   (E17 §3, E17B §2)
    priced = U*p_in + W_5m*p_w5 + W_1h*p_w1 + R*p_read + O*p_out

Preregistration §4 registers exactly one prediction:

    S1  for EACH rho:  delta_cache_read(karc - rag) > 0
                  AND  delta_creation(karc - rag) < 0

Preregistration §4 explicitly does NOT predict the direction of the gross ratio
or of the priced ratio, because changing rho moves both arms' call counts and
that interaction was never modelled.  Those two ratios and the USD difference
are therefore emitted as MEASUREMENTS with confidence intervals and carry
`predicted: false`; nothing in this file turns them into a verdict.

mu_w* is recomputed from THIS cell's own component totals, per rho.  No
threshold is carried over from another cell (E17B §6, E18 appendix A.3).

§5: an S1 FAIL at either rho is reported as-is; the data is not re-sliced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import time
from pathlib import Path

HERE = Path(__file__).resolve()
CELL_DIR = HERE.parents[1]
REPO = HERE.parents[4]
if not (REPO / "pyproject.toml").exists() or not (REPO / "src" / "karc").is_dir():
    raise SystemExit(f"E23: repository root misresolved as {REPO}")
RAW = CELL_DIR / "raw"
RUN = REPO / "tmp" / "e23"

RHOS = {"025": 0.25, "075": 0.75}
ARMS = ("karc-full", "rag-bm25")
# Measurement window per rho, set in main() from the committed steady phase.
WINDOW: dict[str, list[str]] = {}
WARMUP: dict[str, list[str]] = {}
NAMESPACE = ""

BOOTSTRAP_SEED = 1313
BOOTSTRAP_RESAMPLES = 10_000
CI_ALPHA = 0.05
N_SESSIONS = 12

# Committed standard OpenAI rate card, USD per million tokens.
# docs/analysis/openai-luna-rate-card.md, primary source developers.openai.com,
# rates observed 2026-08-28.  e5_cross_runtime.py is line-frozen (ClaudeRates
# only), so the card lives here, identical to E19's leg B card.
RATE_CARD = {"input": 0.20, "write_5m": 0.25, "write_1h": 0.25,
             "read": 0.02, "output": 1.20,
             "source": "docs/analysis/openai-luna-rate-card.md "
                       "(observed 2026-08-28); identical to E19 leg B"}
# pi's own catalog rates.  Control column only (E17B §6).
PI_CATALOG_RATES = {"input": 0.20, "output": 1.20, "read": 0.02,
                    "write_5m": 0.25, "write_1h": 0.25}
OPENAI_CACHE_MIN_PREFIX = 1024
BUDGET_CAP_USD = 3.0


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resample_matrix() -> list[list[int]]:
    """One shared 10,000 x 12 index matrix, so every paired comparison in this
    cell (and, being the same seed and shape, in E19) uses common random
    numbers."""
    rng = random.Random(BOOTSTRAP_SEED)
    return [[rng.randrange(N_SESSIONS) for _ in range(N_SESSIONS)]
            for _ in range(BOOTSTRAP_RESAMPLES)]


def percentile_ci(values: list[float]) -> tuple[float, float]:
    """E13's declared convention: floor(alpha/2*B) and ceil((1-alpha/2)*B)-1."""
    ordered = sorted(values)
    b = len(ordered)
    lo_i = int((CI_ALPHA / 2) * b)
    hi_i = int(-(-(1 - CI_ALPHA / 2) * b // 1)) - 1
    return ordered[lo_i], ordered[hi_i]


def paired_bootstrap_mean(per_session: dict[str, float], sessions: list[str],
                          matrix: list[list[int]]) -> dict:
    values = [per_session[s] for s in sessions]
    point = sum(values) / len(values)
    draws = [sum(values[i] for i in row) / len(row) for row in matrix]
    lo, hi = percentile_ci(draws)
    return {"point": point, "ci95_lo": lo, "ci95_hi": hi,
            "excludes_zero": (lo > 0.0) or (hi < 0.0)}


def paired_bootstrap_ratio_of_sums(numer: dict[str, float],
                                   denom: dict[str, float],
                                   sessions: list[str],
                                   matrix: list[list[int]]) -> dict:
    num = [numer[s] for s in sessions]
    den = [denom[s] for s in sessions]
    if sum(den) == 0:
        return {"point": None, "degenerate": True}
    point = sum(num) / sum(den)
    draws = []
    degenerate = 0
    for row in matrix:
        d = sum(den[i] for i in row)
        if d == 0:
            degenerate += 1
            continue
        draws.append(sum(num[i] for i in row) / d)
    lo, hi = percentile_ci(draws)
    return {"point": point, "ci95_lo": lo, "ci95_hi": hi,
            "entirely_above_one": lo > 1.0, "entirely_below_one": hi < 1.0,
            "usable_resamples": len(draws), "degenerate_resamples": degenerate}


def priced_usd(u: int, w5: int, w1: int, r: int, o: int) -> float:
    c = RATE_CARD
    return (u * c["input"] + w5 * c["write_5m"] + w1 * c["write_1h"]
            + r * c["read"] + o * c["output"]) / 1e6


# --------------------------------------------------------------- assembly

def rho_root(tag: str) -> Path:
    return RUN / f"rho{tag}{NAMESPACE}"


def load_units(tag: str, sessions: list[str],
               arms: tuple[str, ...] = ARMS) -> list[dict]:
    out = []
    for arm in arms:
        for session in sessions:
            path = rho_root(tag) / "units" / f"{arm}--{session}.json"
            if path.exists():
                out.append(json.loads(path.read_text(encoding="utf-8")))
    return out


def session_store_usage(tag: str, arm: str, session: str) -> list[dict] | None:
    """Second usage-acquisition route: pi's own session JSONL.  R-9: the file is
    opened to read `usage` numbers, never to copy text."""
    matches = sorted((rho_root(tag) / "sessions").glob(
        f"*_e23r{tag}{NAMESPACE}-{arm}-{session}.jsonl"))
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


def write_turns(tag: str, units: list[dict]) -> Path:
    path = RAW / f"turns-{tag}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for unit in units:
            for row in unit["rows"]:
                handle.write(json.dumps(row, sort_keys=True,
                                        ensure_ascii=False) + "\n")
    return path


# ------------------------------------------------------------- crosscheck

def crosscheck(tag: str, units: list[dict], steady: dict | None,
               build: dict | None) -> dict:
    sessions = WINDOW[tag]
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
        store = session_store_usage(tag, unit["arm"], unit["session_id"])
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

    # H2 refusal disjunction (E17B §5: the provider layer has no discriminating
    # power on the OpenAI leg).
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
    # H9 cold-start misattribution: a call charged cacheWrite == 0 while reading
    # a prefix it should have created has been served an entry written by an
    # execution outside this dataset (E18 §5, discard 2).
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

    priced_total = sum(priced_usd(
        r["usage_turn"]["input"],
        max(0, r["usage_turn"]["cacheWrite"] - r["usage_turn"]["cacheWrite1h"]),
        r["usage_turn"]["cacheWrite1h"], r["usage_turn"]["cacheRead"],
        r["usage_turn"]["output"]) for r in rows)
    pi_total = sum(float(r["usage_turn"]["pi_cost_total"]) for r in rows)

    # H10: the execution namespace must differ from every earlier execution, on
    # BOTH levers, and must also differ between the two rho values.
    session_prefix = f"e23r{tag}{NAMESPACE}-"
    ns_rows = [{
        "arm": unit["arm"], "session_id": unit["session_id"],
        "pi_session_id_prefix_ok": unit.get("pi_session_id", "").startswith(
            session_prefix),
        "pi_session_id_sha256": unit.get("pi_session_id_sha256"),
    } for unit in units]
    ns_bad = [r for r in ns_rows if not r["pi_session_id_prefix_ok"]]
    root = rho_root(tag)
    sibling_roots = {name: str(REPO / "tmp" / name)
                     for name in ("e17", "e17b", "e18", "e19", "e21")}

    build_rho = ((build or {}).get("rho") or {}).get(tag) or {}
    hold = {
        "H1_three_path_mismatch": [u for u in per_unit
                                   if not u["three_paths_agree"]],
        "H2_refusal_or_zero_output": refusal_rows,
        "H3_closure_violation": len(closure_fail),
        "H4_first_eligible_cold_write_zero": [
            f for f in first_eligible_cold if f["cacheWrite"] == 0],
        "H5_session_completion_failure_rate": round(
            1.0 - turns_done / turns_expected, 6),
        "H6_priced_usd_this_rho": round(priced_total, 6),
        "H7_user_pi_mtime": "judged outside this script (mtime phase)",
        "H8_eligible_call_with_no_cache_activity": len(eligible_dead),
        "H9_cold_start_misattribution_calls": len(duplicate_request_hits),
        "H10_namespace_collision": {
            "namespace": NAMESPACE,
            "pi_session_id_prefix": session_prefix,
            "units_with_wrong_prefix": len(ns_bad),
            "rho_root": str(root),
            "rho_root_under_tmp_e23": str(root).startswith(str(RUN)),
            "distinct_from_sibling_cells": all(
                not str(root).startswith(p) for p in sibling_roots.values()),
            "sibling_roots": sibling_roots,
        },
        "H11_steady_state_not_reached": (
            None if steady is None else bool(steady.get("H11_fired"))),
        "H12_gold_containment_not_one": (
            None if not build_rho
            else bool(build_rho["H12_gold_containment"]["fired"])),
        "H13_degenerate_branch": (
            None if not build_rho
            else bool(build_rho["H13_non_degenerate_branch"]["fired"])),
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
    if hold["H8_eligible_call_with_no_cache_activity"]:
        fired.append("H8")
    if hold["H9_cold_start_misattribution_calls"]:
        fired.append("H9")
    h10 = hold["H10_namespace_collision"]
    if (h10["units_with_wrong_prefix"] or not h10["rho_root_under_tmp_e23"]
            or not h10["distinct_from_sibling_cells"]):
        fired.append("H10")
    if hold["H11_steady_state_not_reached"]:
        fired.append("H11")
    if hold["H12_gold_containment_not_one"]:
        fired.append("H12")
    if hold["H13_degenerate_branch"]:
        fired.append("H13")

    return {
        "rho_tag": tag, "reuse_factor": RHOS[tag],
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
                    "dataset (E18 §5 discard 2). First-class condition (§7 H9).",
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
            "pi_catalog_usd_per_million": PI_CATALOG_RATES,
        },
        "cost": {
            "pi_reported_usd": round(pi_total, 6),
            "standard_rate_card_usd": round(priced_total, 6),
            "rate_card_source": RATE_CARD["source"],
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


# -------------------------------------------------------------- statistics

def per_session_components(units: list[dict]) -> dict:
    """{arm: {session: {component: value}}}"""
    out: dict[str, dict[str, dict[str, float]]] = {}
    for unit in units:
        arm = out.setdefault(unit["arm"], {})
        acc = {"uncached": 0, "creation": 0, "creation_1h": 0, "creation_5m": 0,
               "cache_read": 0, "output": 0, "gross": 0, "api_calls": 0,
               "tool_calls": 0, "turns": 0, "reuse_turns": 0,
               "policy_resident_hit_turns": 0}
        for row in unit["rows"]:
            u = row["usage_turn"]
            w1 = int(u["cacheWrite1h"] or 0)
            acc["uncached"] += u["input"]
            acc["creation"] += u["cacheWrite"]
            acc["creation_1h"] += w1
            acc["creation_5m"] += max(0, u["cacheWrite"] - w1)
            acc["cache_read"] += u["cacheRead"]
            acc["output"] += u["output"]
            acc["gross"] += row["gross_tokens"]
            acc["api_calls"] += row["api_call_count"]
            acc["tool_calls"] += row["tool_execution_start_count"]
            acc["turns"] += 1
            acc["reuse_turns"] += int(bool(row["reuse"]))
            acc["policy_resident_hit_turns"] += int(bool(row["policy_resident_hit"]))
        arm[unit["session_id"]] = acc
    return out


def rho_statistics(tag: str, units: list[dict], matrix: list[list[int]]) -> dict:
    sessions = WINDOW[tag]
    comp = per_session_components(units)
    if set(comp) != set(ARMS) or any(set(comp[a]) != set(sessions) for a in ARMS):
        return {"rho_tag": tag, "incomplete": True,
                "arms_present": {a: sorted(comp.get(a, {})) for a in ARMS}}

    def per_session(arm: str, key: str) -> dict[str, float]:
        return {s: float(comp[arm][s][key]) for s in sessions}

    def priced_per_session(arm: str) -> dict[str, float]:
        return {s: priced_usd(comp[arm][s]["uncached"], comp[arm][s]["creation_5m"],
                              comp[arm][s]["creation_1h"], comp[arm][s]["cache_read"],
                              comp[arm][s]["output"]) for s in sessions}

    gross_k = per_session("karc-full", "gross")
    gross_r = per_session("rag-bm25", "gross")
    priced_k, priced_r = priced_per_session("karc-full"), priced_per_session("rag-bm25")

    gross_ratio = paired_bootstrap_ratio_of_sums(gross_k, gross_r, sessions, matrix)
    priced_ratio = paired_bootstrap_ratio_of_sums(priced_k, priced_r, sessions, matrix)
    gross_diff = paired_bootstrap_mean(
        {s: gross_k[s] - gross_r[s] for s in sessions}, sessions, matrix)
    priced_diff = paired_bootstrap_mean(
        {s: priced_k[s] - priced_r[s] for s in sessions}, sessions, matrix)

    totals = {arm: {key: sum(comp[arm][s][key] for s in sessions)
                    for key in comp[arm][sessions[0]]} for arm in ARMS}
    delta = {key: totals["karc-full"][key] - totals["rag-bm25"][key]
             for key in totals["karc-full"]}

    # Per-component paired bootstrap, so the S1 signs carry an interval too.
    # S1 itself is judged on the component TOTALS, exactly as E19's Q3 was.
    component_ci = {}
    for key in ("cache_read", "creation", "uncached", "output"):
        k = per_session("karc-full", key)
        r = per_session("rag-bm25", key)
        component_ci[key] = paired_bootstrap_mean(
            {s: k[s] - r[s] for s in sessions}, sessions, matrix)

    c = RATE_CARD
    mu_r = c["read"] / c["input"]
    mu_o = c["output"] / c["input"]
    mu_w_5m = c["write_5m"] / c["input"]
    mu_w_1h = c["write_1h"] / c["input"]
    dU, dW, dR, dO = (delta["uncached"], delta["creation"],
                      delta["cache_read"], delta["output"])
    # dU + mu_w*dW + mu_r*dR + mu_o*dO = 0, solved from THIS rho's own totals.
    mu_w_star = None if dW == 0 else -(dU + mu_r * dR + mu_o * dO) / dW

    per_position = {}
    for arm in ARMS:
        arm_rows = [r for unit in units if unit["arm"] == arm for r in unit["rows"]]
        per_position[arm] = {str(pos): {
            "n": len([r for r in arm_rows if r["position"] == pos]),
            "gross_mean": statistics.mean(
                [r["gross_tokens"] for r in arm_rows if r["position"] == pos]),
            "creation_mean": statistics.mean(
                [r["usage_turn"]["cacheWrite"] for r in arm_rows
                 if r["position"] == pos]),
            "read_mean": statistics.mean(
                [r["usage_turn"]["cacheRead"] for r in arm_rows
                 if r["position"] == pos]),
            "output_mean": statistics.mean(
                [r["usage_turn"]["output"] for r in arm_rows if r["position"] == pos]),
            "api_calls_mean": statistics.mean(
                [r["api_call_count"] for r in arm_rows if r["position"] == pos]),
        } for pos in range(1, 9)}

    single_call_prefix = {}
    for arm in ARMS:
        prefixes = [c2["prefix_tokens"] for unit in units if unit["arm"] == arm
                    for r in unit["rows"] if r["api_call_count"] == 1
                    for c2 in r["calls"]]
        single_call_prefix[arm] = {
            "n": len(prefixes),
            "mean": statistics.mean(prefixes) if prefixes else None,
            "min": min(prefixes) if prefixes else None,
            "max": max(prefixes) if prefixes else None,
        }

    return {
        "rho_tag": tag, "reuse_factor": RHOS[tag],
        "measurement_window": sessions,
        "rate_card_usd_per_million": {k: v for k, v in c.items() if k != "source"},
        "rate_card_source": c["source"],
        "multipliers": {"mu_read": mu_r, "mu_output": mu_o,
                        "mu_write_1h": mu_w_1h, "mu_write_5m": mu_w_5m},
        "component_totals": totals,
        "component_delta_karc_minus_rag": delta,
        "component_delta_paired_ci": component_ci,
        "priced_totals_usd": {arm: sum(priced_per_session(arm).values())
                              for arm in ARMS},
        "gross_ratio_karc_over_rag": {**gross_ratio, "predicted": False,
                                      "why": "preregistration §4 does not "
                                             "predict the gross ordering"},
        "priced_ratio_karc_over_rag": {**priced_ratio, "predicted": False,
                                       "why": "preregistration §4 does not "
                                              "predict the priced ordering"},
        "paired_mean_session_gross_difference": {**gross_diff, "predicted": False},
        "paired_mean_session_priced_usd_difference": {**priced_diff,
                                                     "predicted": False},
        "reversal_threshold_mu_w_star": mu_w_star,
        "observed_mu_w": mu_w_5m,
        "mu_w_star_note": ("recomputed from THIS rho's own component totals; no "
                           "threshold is carried over from another cell "
                           "(E17B §6, E18 appendix A.3). Not a prediction."),
        "per_session_gross": {"karc-full": gross_k, "rag-bm25": gross_r},
        "per_session_priced_usd": {"karc-full": priced_k, "rag-bm25": priced_r},
        "per_session_reuse_turns": {
            arm: {s: comp[arm][s]["reuse_turns"] for s in sessions} for arm in ARMS},
        "per_session_tool_calls": {
            arm: {s: comp[arm][s]["tool_calls"] for s in sessions} for arm in ARMS},
        "per_position_profile": per_position,
        "single_api_call_turn_prefix_tokens": single_call_prefix,
        "bootstrap": {"method": "paired percentile bootstrap over sessions, "
                                "common random numbers",
                      "seed": BOOTSTRAP_SEED,
                      "resamples": BOOTSTRAP_RESAMPLES,
                      "ci": "two-sided 95% percentile",
                      "unit": "session", "n_sessions": len(sessions)},
    }


def e19_leg_b_contrast() -> dict:
    """E19 leg B's committed figures at rho = 0.50, read for the three-point
    contrast.  Read-only; E19's raw is never modified."""
    path = REPO / "docs" / "experiments" / "E19-STEADY-STATE" / "raw" / "analysis.json"
    if not path.exists():
        return {"available": False}
    data = json.loads(path.read_text(encoding="utf-8"))
    st = (data.get("legs") or {}).get("B")
    if not st or st.get("incomplete"):
        return {"available": False, "source": str(path.relative_to(REPO))}
    gr = st["gross_ratio_karc_over_rag"]
    pr = st["priced_ratio_karc_over_rag"]
    pd = st["paired_mean_session_priced_usd_difference"]
    gd = st["paired_mean_session_gross_difference"]
    return {
        "available": True,
        "source": str(path.relative_to(REPO)),
        "sha256": sha256_file(path),
        "reuse_factor": 0.50,
        "window": (data.get("measurement_window") or {}).get("measured_sessions"),
        "component_delta_karc_minus_rag": st["component_delta_karc_minus_rag"],
        "component_totals": st["component_totals"],
        "gross_ratio": gr["point"],
        "gross_ratio_ci95": [gr["ci95_lo"], gr["ci95_hi"]],
        "priced_ratio": pr["point"],
        "priced_ratio_ci95": [pr["ci95_lo"], pr["ci95_hi"]],
        "paired_mean_usd_difference": pd["point"],
        "paired_mean_usd_difference_ci95": [pd["ci95_lo"], pd["ci95_hi"]],
        "paired_mean_gross_difference": gd["point"],
        "paired_mean_gross_difference_ci95": [gd["ci95_lo"], gd["ci95_hi"]],
        "mu_w_star": st.get("reversal_threshold_mu_w_star"),
        "priced_totals_usd": st.get("priced_totals_usd"),
        "note": ("E19 leg B is the same harness, provider, model, arms, seed, "
                 "rate card and statistics; only the reuse factor differs. Its "
                 "figures are quoted, not recomputed."),
    }


def judge(stats: dict[str, dict], checks: dict[str, dict]) -> dict:
    verdict: dict = {}
    for tag in sorted(RHOS):
        st = stats.get(tag)
        if not st or st.get("incomplete"):
            verdict[tag] = {"status": "NOT RUN"}
            continue
        d = st["component_delta_karc_minus_rag"]
        s1 = "PASS" if (d["cache_read"] > 0 and d["creation"] < 0) else "FAIL"
        verdict[tag] = {
            "reuse_factor": RHOS[tag],
            "S1_component_signature": {
                "verdict": s1,
                "criterion": "delta_cache_read(karc - rag) > 0 AND "
                             "delta_creation(karc - rag) < 0",
                "delta_cache_read": d["cache_read"],
                "delta_creation": d["creation"],
                "delta_uncached": d["uncached"],
                "delta_output": d["output"],
                "delta_cache_read_positive": d["cache_read"] > 0,
                "delta_creation_negative": d["creation"] < 0,
                "paired_ci": st["component_delta_paired_ci"],
            },
            "measured_not_predicted": {
                "note": "preregistration §4 registers S1 only; the two ratios "
                        "and the USD difference are reported, not judged",
                "gross_ratio": st["gross_ratio_karc_over_rag"]["point"],
                "gross_ratio_ci95": [st["gross_ratio_karc_over_rag"]["ci95_lo"],
                                     st["gross_ratio_karc_over_rag"]["ci95_hi"]],
                "priced_ratio": st["priced_ratio_karc_over_rag"]["point"],
                "priced_ratio_ci95": [st["priced_ratio_karc_over_rag"]["ci95_lo"],
                                      st["priced_ratio_karc_over_rag"]["ci95_hi"]],
                "paired_mean_usd_difference":
                    st["paired_mean_session_priced_usd_difference"]["point"],
                "paired_mean_usd_difference_ci95": [
                    st["paired_mean_session_priced_usd_difference"]["ci95_lo"],
                    st["paired_mean_session_priced_usd_difference"]["ci95_hi"]],
                "paired_mean_gross_difference":
                    st["paired_mean_session_gross_difference"]["point"],
                "paired_mean_gross_difference_ci95": [
                    st["paired_mean_session_gross_difference"]["ci95_lo"],
                    st["paired_mean_session_gross_difference"]["ci95_hi"]],
                "mu_w_star": st["reversal_threshold_mu_w_star"],
                "observed_mu_w": st["observed_mu_w"],
            },
            "steady_state_reached": (checks[tag].get("steady_state") or {}).get(
                "steady_state_reached"),
            "hold_fired": checks[tag]["hold_fired"],
        }
    verdict["S1_overall"] = {
        "definition": "S1 holds at BOTH preregistered rho values",
        "verdict": ("HELD" if all(
            (verdict.get(t) or {}).get("S1_component_signature", {}).get("verdict")
            == "PASS" for t in sorted(RHOS)) else "NOT HELD"),
        "per_rho": {t: (verdict.get(t) or {}).get(
            "S1_component_signature", {}).get("verdict") for t in sorted(RHOS)},
    }
    total = sum(checks[t]["cost"]["standard_rate_card_usd"]
                for t in sorted(RHOS) if t in checks)
    verdict["H6_budget"] = {
        "cap_usd": BUDGET_CAP_USD,
        "standard_rate_card_total_usd": round(total, 6),
        "within_cap": total <= BUDGET_CAP_USD,
    }
    return verdict


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rhos", default="025,075")
    parser.add_argument("--ns", default="")
    args = parser.parse_args()
    tags = [t for t in args.rhos.split(",") if t in RHOS]

    global NAMESPACE
    NAMESPACE = args.ns

    build_path = RAW / "build.json"
    build = (json.loads(build_path.read_text(encoding="utf-8"))
             if build_path.exists() else None)

    matrix = resample_matrix()
    stats: dict[str, dict] = {}
    checks: dict[str, dict] = {}
    steadies: dict[str, dict] = {}
    for tag in tags:
        steady_path = RAW / f"steady-{tag}.json"
        if not steady_path.exists():
            raise SystemExit(f"E23: run the steady phase for rho tag {tag} first")
        steady = json.loads(steady_path.read_text(encoding="utf-8"))
        if not steady["steady_state_reached"]:
            raise SystemExit(f"E23: H11 fired for rho tag {tag}; not measured")
        steadies[tag] = steady
        WINDOW[tag] = list(steady["measurement_window"])
        WARMUP[tag] = [f"S{i:02d}"
                       for i in range(1, steady["warmup_sessions_selected"] + 1)]
        if len(WINDOW[tag]) != N_SESSIONS:
            raise SystemExit(f"E23: window for {tag} is not {N_SESSIONS} sessions")

    for tag in tags:
        units = load_units(tag, WINDOW[tag])
        if not units:
            continue
        write_turns(tag, units)
        check = crosscheck(tag, units, steadies[tag], build)
        checks[tag] = check
        (RAW / f"crosscheck-{tag}.json").write_text(
            json.dumps(check, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8")
        stats[tag] = rho_statistics(tag, units, matrix)
        print(f"rho {RHOS[tag]}: turns {check['turns_executed']}/"
              f"{check['turns_expected']} hold_fired={check['hold_fired']} "
              f"usd={check['cost']['standard_rate_card_usd']}")

    analysis = {
        "cell": "E23-REUSE-DENSITY",
        "preregistration": "docs/experiments/E23-REUSE-DENSITY/preregistration.md",
        "manipulated_axis": {
            "name": "reuse density",
            "operative_knob": "e5_killgate.build_reuse_schedule(reuse_factor=r)",
            "values": {t: RHOS[t] for t in sorted(RHOS)},
            "control_point": 0.50,
            "fixed": ["harness pi 0.84.3", "provider openai", "model gpt-5.6-luna",
                      "arms karc-full / rag-bm25", "resident budget 5%",
                      "schedule seed 4352", "rate card", "statistics",
                      "12 sessions x 8 turns x 2 arms"],
        },
        "measurement_window": {t: WINDOW.get(t) for t in tags},
        "rho": stats,
        "judgments": judge(stats, checks),
        "e19_leg_b_contrast_rho_050": e19_leg_b_contrast(),
        "accounting_contract": {
            "gross": "input + cacheRead + cacheWrite",
            "why": "pi usage.input excludes cache buckets on both legs "
                   "(E17 §3, E17B §2); measured again in this cell",
        },
        "not_pooled_with_the_six_cells": (
            "preregistration §6: this is a different workload point, so it does "
            "not enter the abstract's range figures or §4's tables 5/6"),
        "_meta": {"generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                    time.gmtime()),
                  "bootstrap_seed": BOOTSTRAP_SEED,
                  "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                  "namespace": NAMESPACE},
    }
    (RAW / "analysis.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8")
    print(json.dumps(analysis["judgments"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
