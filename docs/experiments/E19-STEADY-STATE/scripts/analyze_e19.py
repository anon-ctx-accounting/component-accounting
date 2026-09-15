"""E19-STEADY-STATE — crosscheck, statistics and the preregistered judgments.

Reads the unit files produced by run_e19_steady_state.py (under tmp/e19, plus
the pi session stores for the second usage-acquisition route) and writes:

    raw/turns-<leg>.jsonl      one row per MEASUREMENT turn, closure included
    raw/turns-warmup-<leg>.jsonl   warm-up turns, kept for audit, never judged
    raw/crosscheck-<leg>.json  instrumentation contract + HOLD conditions
    raw/analysis.json          statistics and the Q1..Q3 judgments

No model call, no network.  Deterministic: one shared 10,000 x 12 bootstrap
index matrix (seed 1313), common random numbers across every paired comparison,
exactly as E13-RECOMP and E18-PI-REVERSAL.

Accounting contract (all of it measured, inherited from E18):
    gross = input + cacheRead + cacheWrite
    priced = U*p_in + W_5m*p_w5 + W_1h*p_w1 + R*p_read + O*p_out

Judged prices are the committed standard rate cards.  pi's own `usage.cost` is a
control column.

The warm-up window is EXCLUDED from every judged quantity (preregistration §3);
its cost is reported separately and its only judged role is the steady-state
verification, which is computed by the runner's `steady` phase from an input
state quantity (resident injection tokens), never from a cost ratio.

Preregistration: docs/experiments/E19-STEADY-STATE/preregistration.md
    Q1  paired gross ratio karc/rag: 95% CI entirely ABOVE 1.0
    Q2  paired priced ratio: 95% CI entirely BELOW 1.0 AND paired mean USD
        difference CI excludes 0
    Q3  component signature: d_read > 0 and d_creation < 0
    REVERSAL = Q1 PASS and Q2 PASS.
§5: a Q1 FAIL is reported as-is; the data is not re-sliced.
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
    raise SystemExit(f"E19: repository root misresolved as {REPO}")
RAW = CELL_DIR / "raw"
RUN = REPO / "tmp" / "e19"

# The measurement window.  Set in main() from --warmup-sessions: the window
# starts immediately after the warm-up window and is 12 sessions long, so
# there is no subset choice left in the analysis.
WARMUP_COUNT = 8
SESSIONS: list[str] = [f"S{i:02d}" for i in range(9, 21)]
WARMUP: list[str] = [f"S{i:02d}" for i in range(1, 9)]
ARMS = ("karc-full", "rag-bm25")
LEGS = ("A", "B")
# Set per invocation with --ns-A/--ns-B (H12: a re-run must change both the
# workdir path and the pi session id).
NAMESPACE = {"A": "", "B": ""}

BOOTSTRAP_SEED = 1313
BOOTSTRAP_RESAMPLES = 10_000
CI_ALPHA = 0.05

# Committed standard rate cards, USD per million tokens.
#   leg A: the Anthropic claude-sonnet-5 card recovered exactly from the
#          E10-P2-XR canary raw by E13-RECOMP §4.1 (== the committed
#          ClaudeRates shape in src/karc/bench/e5_cross_runtime.py).
#   leg B: docs/analysis/openai-luna-rate-card.md, primary source
#          developers.openai.com, rates observed 2026-08-28 (appendix C.1/C.4).
#          e5_cross_runtime.py is line-frozen, so this card lives here.
RATE_CARDS = {
    "A": {"input": 3.00, "write_5m": 3.75, "write_1h": 6.00,
          "read": 0.30, "output": 15.00,
          "source": "E13-RECOMP §4.1 recovered Anthropic claude-sonnet-5 card"},
    "B": {"input": 0.20, "write_5m": 0.25, "write_1h": 0.25,
          "read": 0.02, "output": 1.20,
          "source": "docs/analysis/openai-luna-rate-card.md (observed 2026-08-28)"},
}
# pi's own catalog rates, USD per million tokens, recovered by reverse-solving
# pi's reported `usage.cost` from the four buckets (E17 §8, E17B §2 evidence 3).
# Control column only: the Anthropic entry is the introductory card that expires
# 2026-08-31, and its ratios are identical to standard, so no judgement moves.
PI_CATALOG_RATES = {
    "A": {"input": 2.00, "output": 10.00, "read": 0.20,
          "write_5m": 2.50, "write_1h": 4.00},
    "B": {"input": 0.20, "output": 1.20, "read": 0.02,
          "write_5m": 0.25, "write_1h": 0.25},
}
OPENAI_CACHE_MIN_PREFIX = 1024
BUDGET_CAP_USD = 25.0   # H6', warm-up included


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resample_matrix() -> list[list[int]]:
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(SESSIONS)
    return [[rng.randrange(n) for _ in range(n)] for _ in range(BOOTSTRAP_RESAMPLES)]


def percentile_ci(values: list[float]) -> tuple[float, float]:
    """E13's declared convention: floor(alpha/2*B) and ceil((1-alpha/2)*B)-1."""
    ordered = sorted(values)
    b = len(ordered)
    lo_i = int((CI_ALPHA / 2) * b)
    hi_i = int(-(-(1 - CI_ALPHA / 2) * b // 1)) - 1
    return ordered[lo_i], ordered[hi_i]


def paired_bootstrap_mean(per_session: dict[str, float],
                          matrix: list[list[int]]) -> dict:
    values = [per_session[s] for s in SESSIONS]
    point = sum(values) / len(values)
    draws = [sum(values[i] for i in row) / len(row) for row in matrix]
    lo, hi = percentile_ci(draws)
    return {"point": point, "ci95_lo": lo, "ci95_hi": hi,
            "excludes_zero": (lo > 0.0) or (hi < 0.0)}


def paired_bootstrap_ratio_of_sums(numer: dict[str, float],
                                   denom: dict[str, float],
                                   matrix: list[list[int]]) -> dict:
    num = [numer[s] for s in SESSIONS]
    den = [denom[s] for s in SESSIONS]
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


def priced_usd(leg: str, u: int, w5: int, w1: int, r: int, o: int) -> float:
    card = RATE_CARDS[leg]
    return (u * card["input"] + w5 * card["write_5m"] + w1 * card["write_1h"]
            + r * card["read"] + o * card["output"]) / 1e6


# --------------------------------------------------------------- assembly

def load_units(leg: str, sessions: list[str] | None = None,
               arms: tuple[str, ...] = ARMS) -> list[dict]:
    out = []
    for arm in arms:
        for session in (SESSIONS if sessions is None else sessions):
            path = (RUN / f"leg{leg}{NAMESPACE[leg]}" / "units"
                    / f"{arm}--{session}.json")
            if path.exists():
                out.append(json.loads(path.read_text(encoding="utf-8")))
    return out


def session_store_usage(leg: str, arm: str, session: str) -> list[dict] | None:
    """Second usage-acquisition route: pi's own session JSONL.  R-9: the file is
    opened to read `usage` numbers, never to copy text."""
    matches = sorted((RUN / f"leg{leg}{NAMESPACE[leg]}" / "sessions").glob(
        f"*_e19{NAMESPACE[leg]}-{arm}-{session}.jsonl"))
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


def write_turns(leg: str, units: list[dict], *, tag: str = "") -> Path:
    path = RAW / f"turns{tag}-{leg}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for unit in units:
            for row in unit["rows"]:
                handle.write(json.dumps(row, sort_keys=True,
                                        ensure_ascii=False) + "\n")
    return path


# ------------------------------------------------------------- crosscheck

def crosscheck(leg: str, units: list[dict], steady: dict | None,
               warmup_units: list[dict]) -> dict:
    calls = [c for unit in units for row in unit["rows"] for c in row["calls"]]
    rows = [row for unit in units for row in unit["rows"]]
    closure_fail = [c for c in calls if not c["closure_total_eq_four_sum"]]
    write1h_fail = [c for c in calls if not c["write1h_le_write"]]
    inclusive_true = [c for c in calls if c["inclusive_hypothesis_total_eq_input_plus_output"]]
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
        store = session_store_usage(leg, unit["arm"], unit["session_id"])
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

    # H2' refusal disjunction, applied to both legs (leg A additionally has the
    # discriminating provider layer).
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
    # H4'' first cache-eligible cold call per unit
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
    # Not a preregistered condition: an instrumentation assert added after the
    # leg B attempt-2 incident.  A call that extends the prefix but is charged
    # cacheWrite == 0 while reading the whole prefix has been served a cache
    # entry written by an earlier, identical request - i.e. by an execution that
    # is not part of this dataset.  Such a call mis-buckets creation as read, so
    # a clean run must have none.
    duplicate_request_hits = [{
        "arm": r["arm"], "session_id": r["session_id"], "position": r["position"],
        "call_index": i, "prefix_tokens": c["prefix_tokens"],
        "cacheRead": c["cacheRead"],
    } for r in rows for i, c in enumerate(r["calls"])
        if c["cacheWrite"] == 0 and c["cacheRead"] > 0]
    w1h_zero = [c for c in calls if int(c["cacheWrite1h"] or 0) == 0]
    w1h_key_absent = [c for c in calls if not c["cacheWrite1h_key_present"]]

    # rate card reverse verification against pi's own reported cost
    residuals = [abs(float(c["pi_rate_card_modelled_usd"])
                     - float(c["pi_cost_total"] or 0.0)) for c in calls]

    turns_expected = len(SESSIONS) * len(ARMS) * 8
    turns_done = len(rows)
    incomplete = [{"arm": u["arm"], "session_id": u["session_id"],
                   "turns_executed": u["turns_executed"]}
                  for u in units if u["turns_executed"] != 8]

    priced_total = sum(priced_usd(
        leg, r["usage_turn"]["input"],
        max(0, r["usage_turn"]["cacheWrite"] - r["usage_turn"]["cacheWrite1h"]),
        r["usage_turn"]["cacheWrite1h"], r["usage_turn"]["cacheRead"],
        r["usage_turn"]["output"]) for r in rows)
    pi_total = sum(float(r["usage_turn"]["pi_cost_total"]) for r in rows)

    # H12: the execution namespace must differ from every earlier execution.
    # Both levers are checked, because E18's leg B attempt 2 changed neither:
    # the pi session id prefix and the workdir path.
    session_prefix = f"e19{NAMESPACE[leg]}-"
    ns_rows = []
    for unit in units + warmup_units:
        pi_sid = unit.get("pi_session_id", "")
        ns_rows.append({
            "arm": unit["arm"], "session_id": unit["session_id"],
            "pi_session_id_prefix_ok": pi_sid.startswith(session_prefix),
            "pi_session_id_sha256": unit.get("pi_session_id_sha256"),
        })
    ns_bad = [r for r in ns_rows if not r["pi_session_id_prefix_ok"]]
    leg_dir = RUN / f"leg{leg}{NAMESPACE[leg]}"
    e18_dir = REPO / "tmp" / "e18"

    warm_priced = sum(priced_usd(
        leg, r["usage_turn"]["input"],
        max(0, r["usage_turn"]["cacheWrite"] - r["usage_turn"]["cacheWrite1h"]),
        r["usage_turn"]["cacheWrite1h"], r["usage_turn"]["cacheRead"],
        r["usage_turn"]["output"])
        for u in warmup_units for r in u["rows"])

    hold = {
        "H1_three_path_mismatch": [u for u in per_unit if not u["three_paths_agree"]],
        "H2_prime_refusal_or_zero_output": refusal_rows,
        "H3_closure_violation": len(closure_fail),
        "H4_legA_cacheWrite1h_zero_calls": (len(w1h_zero) if leg == "A" else None),
        "H4_prime_prime_first_eligible_cold_write_zero": [
            f for f in first_eligible_cold if f["cacheWrite"] == 0],
        "H5_session_completion_failure_rate": round(
            1.0 - turns_done / turns_expected, 6),
        "H6_prime_priced_usd_this_leg_measurement": round(priced_total, 6),
        "H6_prime_priced_usd_this_leg_warmup": round(warm_priced, 6),
        "H8_prime_eligible_call_with_no_cache_activity": len(eligible_dead),
        # H10 is decided by the runner's `steady` phase on an INPUT quantity.
        "H10_steady_state_not_reached": (
            None if steady is None else bool(steady.get("H10_fired"))),
        # H11: E18's post-hoc assert, promoted to a first-class hold condition.
        "H11_cold_start_misattribution_calls": len(duplicate_request_hits),
        "H11_cold_start_misattribution_calls_warmup": sum(
            1 for u in warmup_units for r in u["rows"] for c in r["calls"]
            if c["cacheWrite"] == 0 and c["cacheRead"] > 0),
        "H12_namespace_collision": {
            "namespace": NAMESPACE[leg],
            "pi_session_id_prefix": session_prefix,
            "units_with_wrong_prefix": len(ns_bad),
            "leg_root": str(leg_dir),
            "leg_root_under_tmp_e19": str(leg_dir).startswith(str(RUN)),
            "distinct_from_tmp_e18": not str(leg_dir).startswith(str(e18_dir)),
        },
    }
    fired = []
    if hold["H1_three_path_mismatch"]:
        fired.append("H1")
    if hold["H2_prime_refusal_or_zero_output"]:
        fired.append("H2'")
    if hold["H3_closure_violation"]:
        fired.append("H3")
    if leg == "A" and hold["H4_legA_cacheWrite1h_zero_calls"]:
        fired.append("H4")
    if hold["H4_prime_prime_first_eligible_cold_write_zero"]:
        fired.append("H4''")
    if hold["H5_session_completion_failure_rate"] > 0.10:
        fired.append("H5")
    if hold["H8_prime_eligible_call_with_no_cache_activity"]:
        fired.append("H8'")
    if hold["H10_steady_state_not_reached"]:
        fired.append("H10")
    if hold["H11_cold_start_misattribution_calls"]:
        fired.append("H11")
    h12 = hold["H12_namespace_collision"]
    if (h12["units_with_wrong_prefix"] or not h12["leg_root_under_tmp_e19"]
            or not h12["distinct_from_tmp_e18"]):
        fired.append("H12")

    return {
        "leg": leg,
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
            "note": "gross = input + cacheRead + cacheWrite (E17 §3 / E17B §2)",
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
            "value_zero_calls": len(w1h_zero),
            "equal_to_cacheWrite_calls": sum(
                1 for c in calls
                if int(c["cacheWrite1h"] or 0) == c["cacheWrite"]),
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
                str(c["stopReason"]): sum(1 for x in calls
                                          if str(x["stopReason"]) == str(c["stopReason"]))
                for c in calls}.items())),
            "provider_raw": dict(sorted({
                str(c["rawStopReason"]): sum(1 for x in calls
                                             if str(x["rawStopReason"]) == str(c["rawStopReason"]))
                for c in calls}.items())),
        },
        "per_unit": per_unit,
        "session_store_equals_stdout_all_units": jsonl_agree_all,
        "H11_cold_start_assert": {
            "what": "calls with cacheWrite == 0 and cacheRead > 0, i.e. served a "
                    "prefix written by an earlier identical request outside this "
                    "dataset. Preregistered as a first-class hold condition "
                    "for E19 (§7 H11).",
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
            "pi_catalog_usd_per_million": PI_CATALOG_RATES[leg],
            "what_this_checks": "pi's reported usage.cost is reproduced from the "
                                "four recorded buckets at the pi catalog rates; "
                                "an exact residual means the components and the "
                                "rate card are mutually consistent",
        },
        "cost": {
            "pi_reported_usd_measurement": round(pi_total, 6),
            "standard_rate_card_usd_measurement": round(priced_total, 6),
            "standard_rate_card_usd_warmup": round(warm_priced, 6),
            "standard_rate_card_usd_leg_total": round(
                priced_total + warm_priced, 6),
            "warmup_note": "warm-up cost is reported but enters no judgment "
                           "(preregistration §3)",
            "rate_card_source": RATE_CARDS[leg]["source"],
        },
        "steady_state": steady,
        "warmup_window": {
            "sessions": WARMUP,
            "arms": ["karc-full"],
            "units": len(warmup_units),
            "turns": sum(len(u["rows"]) for u in warmup_units),
        },
        "measurement_window": {"sessions": SESSIONS, "arms": list(ARMS)},
        "incidental_grade_diagnostic": {
            "note": "NOT a claim (preregistration §7 forbids the accuracy axis). "
                    "Recorded so that a silently non-functioning arm is "
                    "detectable, and because it independently reproduces the "
                    "baseline's position-1 floor (E13 §1.4).",
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
               "tool_calls": 0, "turns": 0}
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
        arm[unit["session_id"]] = acc
    return out


def leg_statistics(leg: str, units: list[dict], matrix: list[list[int]]) -> dict:
    comp = per_session_components(units)
    if set(comp) != set(ARMS) or any(set(comp[a]) != set(SESSIONS) for a in ARMS):
        return {"leg": leg, "incomplete": True,
                "arms_present": {a: sorted(comp.get(a, {})) for a in ARMS}}

    def per_session(arm: str, key: str) -> dict[str, float]:
        return {s: float(comp[arm][s][key]) for s in SESSIONS}

    def priced_per_session(arm: str) -> dict[str, float]:
        return {s: priced_usd(leg, comp[arm][s]["uncached"],
                             comp[arm][s]["creation_5m"],
                             comp[arm][s]["creation_1h"],
                             comp[arm][s]["cache_read"],
                             comp[arm][s]["output"]) for s in SESSIONS}

    gross_k, gross_r = per_session("karc-full", "gross"), per_session("rag-bm25", "gross")
    priced_k, priced_r = priced_per_session("karc-full"), priced_per_session("rag-bm25")

    gross_ratio = paired_bootstrap_ratio_of_sums(gross_k, gross_r, matrix)
    priced_ratio = paired_bootstrap_ratio_of_sums(priced_k, priced_r, matrix)
    gross_diff = paired_bootstrap_mean(
        {s: gross_k[s] - gross_r[s] for s in SESSIONS}, matrix)
    priced_diff = paired_bootstrap_mean(
        {s: priced_k[s] - priced_r[s] for s in SESSIONS}, matrix)

    totals = {arm: {key: sum(comp[arm][s][key] for s in SESSIONS)
                    for key in comp[arm][SESSIONS[0]]} for arm in ARMS}
    delta = {key: totals["karc-full"][key] - totals["rag-bm25"][key]
             for key in totals["karc-full"]}

    card = RATE_CARDS[leg]
    mu_r = card["read"] / card["input"]
    mu_o = card["output"] / card["input"]
    mu_w_1h = card["write_1h"] / card["input"]
    mu_w_5m = card["write_5m"] / card["input"]
    dU, dW, dR, dO = (delta["uncached"], delta["creation"],
                      delta["cache_read"], delta["output"])
    # dU + mu_w*dW + mu_r*dR + mu_o*dO = 0
    mu_w_star = (None if dW == 0
                 else -(dU + mu_r * dR + mu_o * dO) / dW)

    per_position = {}
    for arm in ARMS:
        rows = [r for unit in units if unit["arm"] == arm for r in unit["rows"]]
        per_position[arm] = {str(pos): {
            "n": len([r for r in rows if r["position"] == pos]),
            "gross_mean": statistics.mean(
                [r["gross_tokens"] for r in rows if r["position"] == pos]),
            "creation_mean": statistics.mean(
                [r["usage_turn"]["cacheWrite"] for r in rows if r["position"] == pos]),
            "read_mean": statistics.mean(
                [r["usage_turn"]["cacheRead"] for r in rows if r["position"] == pos]),
            "output_mean": statistics.mean(
                [r["usage_turn"]["output"] for r in rows if r["position"] == pos]),
            "api_calls_mean": statistics.mean(
                [r["api_call_count"] for r in rows if r["position"] == pos]),
        } for pos in range(1, 9)}

    single_call_prefix = {}
    for arm in ARMS:
        prefixes = [c["prefix_tokens"] for unit in units if unit["arm"] == arm
                    for r in unit["rows"] if r["api_call_count"] == 1
                    for c in r["calls"]]
        single_call_prefix[arm] = {
            "n": len(prefixes),
            "mean": statistics.mean(prefixes) if prefixes else None,
            "min": min(prefixes) if prefixes else None,
            "max": max(prefixes) if prefixes else None,
        }

    return {
        "leg": leg,
        "rate_card_usd_per_million": {k: v for k, v in card.items() if k != "source"},
        "rate_card_source": card["source"],
        "multipliers": {"mu_read": mu_r, "mu_output": mu_o,
                        "mu_write_1h": mu_w_1h, "mu_write_5m": mu_w_5m},
        "component_totals": totals,
        "component_delta_karc_minus_rag": delta,
        "priced_totals_usd": {arm: sum(priced_per_session(arm).values())
                              for arm in ARMS},
        "gross_ratio_karc_over_rag": gross_ratio,
        "priced_ratio_karc_over_rag": priced_ratio,
        "paired_mean_session_gross_difference": gross_diff,
        "paired_mean_session_priced_usd_difference": priced_diff,
        "reversal_threshold_mu_w_star": mu_w_star,
        "observed_mu_w": (mu_w_1h if leg == "A" else mu_w_5m),
        "reversal_predicted_by_threshold": (
            None if mu_w_star is None
            else ((mu_w_1h if leg == "A" else mu_w_5m) > mu_w_star
                  if dW < 0 else "threshold not a lower bound (dW >= 0)")),
        "per_session_gross": {"karc-full": gross_k, "rag-bm25": gross_r},
        "per_session_priced_usd": {"karc-full": priced_k, "rag-bm25": priced_r},
        "per_position_profile": per_position,
        "single_api_call_turn_prefix_tokens": single_call_prefix,
        "bootstrap": {"method": "paired percentile bootstrap over sessions, "
                                "common random numbers",
                      "seed": BOOTSTRAP_SEED,
                      "resamples": BOOTSTRAP_RESAMPLES,
                      "ci": "two-sided 95% percentile",
                      "unit": "session", "n_sessions": len(SESSIONS)},
    }


def e18_pooled_contrast() -> dict:
    """E18's POOLED judgments, read from its committed analysis for a side-by-side.

    Only the pooled figures E18 reported as its judgment are read.  E18 §4's
    post-hoc per-session regime observation is deliberately NOT read and NOT
    cited: E19 §8 forbids using it, and this cell judges independently.
    """
    path = REPO / "docs" / "experiments" / "E18-PI-REVERSAL" / "raw" / "analysis.json"
    if not path.exists():
        return {"available": False}
    data = json.loads(path.read_text(encoding="utf-8"))
    out = {"available": True, "source": str(path.relative_to(REPO)),
           "sha256": sha256_file(path),
           "excluded": "E18 §4 per-session regime observation (post hoc)"}
    for leg in LEGS:
        st = (data.get("legs") or {}).get(leg)
        if not st or st.get("incomplete"):
            continue
        gr = st["gross_ratio_karc_over_rag"]
        pr = st["priced_ratio_karc_over_rag"]
        out[leg] = {
            "pooled_gross_ratio": gr["point"],
            "pooled_gross_ratio_ci95": [gr["ci95_lo"], gr["ci95_hi"]],
            "pooled_gross_ci_crosses_one": not (gr["entirely_above_one"]
                                                or gr["entirely_below_one"]),
            "pooled_priced_ratio": pr["point"],
            "pooled_priced_ratio_ci95": [pr["ci95_lo"], pr["ci95_hi"]],
            "mu_w_star": st.get("reversal_threshold_mu_w_star"),
            "component_delta_karc_minus_rag": st["component_delta_karc_minus_rag"],
            "window": "E18 pooled all 12 of its sessions, warm-up included; "
                      "E19 pools 12 post-warm-up sessions only",
        }
    return out


def descriptive_window_profile(stats: dict[str, dict]) -> dict:
    """Descriptive only.  Per-session ratios INSIDE the single preregistered
    measurement window; no subset of them is judged, and no criterion attaches.
    """
    out: dict = {"caveat": "descriptive; the judgment is the pooled test over "
                           "the whole preregistered window, and no sub-range of "
                           "it is selected or tested"}
    for leg in LEGS:
        st = stats.get(leg)
        if not st or st.get("incomplete"):
            continue
        gk = st["per_session_gross"]["karc-full"]
        gr = st["per_session_gross"]["rag-bm25"]
        pk = st["per_session_priced_usd"]["karc-full"]
        pr = st["per_session_priced_usd"]["rag-bm25"]
        rows = {s: {"gross_ratio": gk[s] / gr[s], "priced_ratio": pk[s] / pr[s]}
                for s in SESSIONS}
        out[leg] = {
            "per_session": rows,
            "sessions_with_gross_ratio_above_one": sum(
                1 for s in SESSIONS if rows[s]["gross_ratio"] > 1.0),
            "sessions_with_priced_ratio_below_one": sum(
                1 for s in SESSIONS if rows[s]["priced_ratio"] < 1.0),
            "n_sessions": len(SESSIONS),
        }
    return out


def judge(stats: dict[str, dict], checks: dict[str, dict]) -> dict:
    verdict: dict = {}
    for leg in LEGS:
        st = stats.get(leg)
        if not st or st.get("incomplete"):
            verdict[leg] = {"status": "NOT RUN"}
            continue
        gr = st["gross_ratio_karc_over_rag"]
        pr = st["priced_ratio_karc_over_rag"]
        pd = st["paired_mean_session_priced_usd_difference"]
        gd = st["paired_mean_session_gross_difference"]
        d = st["component_delta_karc_minus_rag"]
        q1 = "PASS" if gr["entirely_above_one"] else "FAIL"
        q2_pass = bool(pr["entirely_below_one"] and pd["excludes_zero"])
        q2 = "PASS" if q2_pass else "FAIL"
        q3 = "PASS" if (d["cache_read"] > 0 and d["creation"] < 0) else "FAIL"
        verdict[leg] = {
            "Q1_gross_ordering_karc_more_expensive": {
                "verdict": q1,
                "criterion": "paired gross ratio 95% CI entirely above 1.0",
                "ratio": gr["point"], "ci95": [gr["ci95_lo"], gr["ci95_hi"]],
                "paired_mean_session_gross_difference": gd["point"],
                "paired_mean_session_gross_difference_ci95": [gd["ci95_lo"],
                                                              gd["ci95_hi"]],
            },
            "Q2_priced_reversal": {
                "verdict": q2,
                "criterion": "paired priced ratio 95% CI entirely below 1.0 AND "
                             "paired mean USD difference CI excludes 0",
                "priced_ratio": pr["point"],
                "priced_ratio_ci95": [pr["ci95_lo"], pr["ci95_hi"]],
                "priced_ratio_ci_entirely_below_one": pr["entirely_below_one"],
                "paired_mean_usd_difference": pd["point"],
                "paired_mean_usd_difference_ci95": [pd["ci95_lo"], pd["ci95_hi"]],
                "usd_difference_ci_excludes_zero": pd["excludes_zero"],
            },
            "Q3_component_signature": {
                "verdict": q3,
                "criterion": "delta_read > 0 AND delta_creation < 0",
                "delta_read": d["cache_read"], "delta_creation": d["creation"],
                "delta_uncached": d["uncached"], "delta_output": d["output"],
            },
            "REVERSAL": {
                "definition": "Q1 PASS and Q2 PASS",
                "verdict": "ESTABLISHED" if (q1 == "PASS" and q2 == "PASS")
                           else "NOT ESTABLISHED",
            },
            "reversal_threshold_mu_w_star": st["reversal_threshold_mu_w_star"],
            "observed_mu_w": st["observed_mu_w"],
            "steady_state_reached": (checks[leg].get("steady_state") or {}).get(
                "steady_state_reached"),
            "hold_fired": checks[leg]["hold_fired"],
        }
    total = sum(checks[leg]["cost"]["standard_rate_card_usd_leg_total"]
                for leg in LEGS if leg in checks)
    verdict["H6_prime_budget"] = {
        "cap_usd": BUDGET_CAP_USD,
        "standard_rate_card_total_usd_incl_warmup": round(total, 6),
        "within_cap": total <= BUDGET_CAP_USD,
    }
    return verdict


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legs", default="A,B")
    parser.add_argument("--ns-A", default="")
    parser.add_argument("--ns-B", default="")
    parser.add_argument("--warmup-sessions", type=int, default=8)
    args = parser.parse_args()
    legs = [leg for leg in args.legs.split(",") if leg in LEGS]
    NAMESPACE["A"] = getattr(args, "ns_A")
    NAMESPACE["B"] = getattr(args, "ns_B")

    global WARMUP_COUNT, SESSIONS, WARMUP
    WARMUP_COUNT = args.warmup_sessions
    if not 8 <= WARMUP_COUNT <= 12:
        raise SystemExit("E19: warm-up window must be 8..12 sessions")
    WARMUP = [f"S{i:02d}" for i in range(1, WARMUP_COUNT + 1)]
    SESSIONS = [f"S{i:02d}" for i in range(WARMUP_COUNT + 1, WARMUP_COUNT + 13)]

    matrix = resample_matrix()
    stats: dict[str, dict] = {}
    checks: dict[str, dict] = {}
    for leg in legs:
        units = load_units(leg)
        if not units:
            continue
        warmup_units = load_units(leg, sessions=WARMUP, arms=("karc-full",))
        steady_path = RAW / f"steady-{leg}.json"
        steady = (json.loads(steady_path.read_text(encoding="utf-8"))
                  if steady_path.exists() else None)
        write_turns(leg, units)
        if warmup_units:
            write_turns(leg, warmup_units, tag="-warmup")
        check = crosscheck(leg, units, steady, warmup_units)
        checks[leg] = check
        (RAW / f"crosscheck-{leg}.json").write_text(
            json.dumps(check, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8")
        stats[leg] = leg_statistics(leg, units, matrix)
        print(f"leg {leg}: turns {check['turns_executed']}/{check['turns_expected']} "
              f"hold_fired={check['hold_fired']} "
              f"measurement_usd={check['cost']['standard_rate_card_usd_measurement']} "
              f"warmup_usd={check['cost']['standard_rate_card_usd_warmup']}")

    analysis = {
        "cell": "E19-STEADY-STATE",
        "preregistration": "docs/experiments/E19-STEADY-STATE/preregistration.md",
        "measurement_window": {"warmup_sessions": WARMUP,
                               "measured_sessions": SESSIONS,
                               "arms": list(ARMS),
                               "note": "one preregistered window, pooled once; "
                                       "no sub-range is selected or tested"},
        "legs": stats,
        "judgments": judge(stats, checks),
        "e18_pooled_contrast": e18_pooled_contrast(),
        "descriptive_window_profile": descriptive_window_profile(stats),
        "accounting_contract": {
            "gross": "input + cacheRead + cacheWrite",
            "why": "pi usage.input excludes cache buckets on both legs "
                   "(E17 §3, E17B §2); treating it as provider raw input_tokens "
                   "under-counts gross by ~5.5x",
        },
        "_meta": {"generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                    time.gmtime()),
                  "bootstrap_seed": BOOTSTRAP_SEED,
                  "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                  "namespace": dict(NAMESPACE)},
    }
    (RAW / "analysis.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8")
    print(json.dumps(analysis["judgments"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
