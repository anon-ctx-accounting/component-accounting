"""E21-CODEX-COMPONENT — crosscheck, statistics and the preregistered R1..R3.

Reads the unit files produced by run_e21_codex_component.py (under tmp/e21) and
writes:

    raw/turns.jsonl            one row per MEASUREMENT turn
    raw/turns-warmup.jsonl     warm-up turns, kept for audit, never judged
    raw/crosscheck.json        instrumentation contract + hold conditions
    raw/analysis.json          statistics, R1..R3, and the E19 leg B contrast

No model call, no network.  Deterministic: one shared 10,000 x 12 bootstrap
index matrix (seed 1313), common random numbers across every paired
comparison, exactly as E13-RECOMP, E18-PI-REVERSAL and E19-STEADY-STATE.

Accounting contract, all of it measured by the runner:

  * ``turn.completed.usage`` on stdout is the CUMULATIVE thread total, so every
    per-turn component is the DIFFERENCE of consecutive reports.  The rollout's
    per-call ``last_token_usage`` records are the second acquisition route and
    must sum to the same per-turn numbers.
  * the inclusion relation is decided here from the recorded evidence, and
    ``gross`` is defined from the surviving reading rather than assumed.
  * priced = fresh*p_in + write*p_write + cached*p_read + output*p_out at the
    published gpt-5.6-luna card.

Preregistration (docs/experiments/E21-CODEX-COMPONENT/preregistration.md §3):
    R1  delta_read(karc-rag) > 0 AND delta_creation(karc-rag) < 0
    R2  paired priced ratio 95% CI entirely BELOW 1.0 AND paired mean USD
        difference CI excludes 0
    R3  mu_w_star computed from THIS cell's own component totals is < 1.25
The gross ordering is NOT predicted; it is measured and reported (§3).
§4: an R1 or R2 failure is reported as the result.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

HERE = Path(__file__).resolve()
CELL_DIR = HERE.parents[1]
REPO = HERE.parents[4]
if not (REPO / "pyproject.toml").exists() or not (REPO / "src" / "karc").is_dir():
    raise SystemExit(f"E21: repository root misresolved as {REPO}")
RAW = CELL_DIR / "raw"
RUN = REPO / "tmp" / "e21"

WARMUP = [f"S{i:02d}" for i in range(1, 9)]
SESSIONS = [f"S{i:02d}" for i in range(9, 21)]
ARMS = ("karc-full", "rag-bm25")
NAMESPACE = ""

BOOTSTRAP_SEED = 1313
BOOTSTRAP_RESAMPLES = 10_000
CI_ALPHA = 0.05

# Published gpt-5.6-luna card, USD per million tokens
# (docs/analysis/openai-luna-rate-card.md, observed 2026-08-28).  The same card
# E19 leg B used, so the two cells' priced numbers are comparable.
RATE_CARD = {"input": 0.20, "write": 0.25, "read": 0.02, "output": 1.20,
             "source": "docs/analysis/openai-luna-rate-card.md (2026-08-28)"}
PUBLISHED_MULTIPLIERS = {"mu_read": 0.10, "mu_write": 1.25, "mu_output": 6.00}
OPENAI_CACHE_MIN_PREFIX = 1024
BUDGET_CAP_USD = 2.0

E19_ANALYSIS = REPO / "docs" / "experiments" / "E19-STEADY-STATE" / "raw" / "analysis.json"


def ns_root() -> Path:
    return RUN / (f"run{NAMESPACE}" if NAMESPACE else "run")


def resample_matrix() -> list[list[int]]:
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(SESSIONS)
    return [[rng.randrange(n) for _ in range(n)]
            for _ in range(BOOTSTRAP_RESAMPLES)]


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
    draws, degenerate = [], 0
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


def priced_usd(fresh: int, write: int, read: int, output: int) -> float:
    return (fresh * RATE_CARD["input"] + write * RATE_CARD["write"]
            + read * RATE_CARD["read"] + output * RATE_CARD["output"]) / 1e6


# --------------------------------------------------------------- assembly

def load_units(sessions: list[str], arms: tuple[str, ...] = ARMS) -> list[dict]:
    out = []
    for arm in arms:
        for session in sessions:
            path = ns_root() / "units" / f"{arm}--{session}.json"
            if path.exists():
                out.append(json.loads(path.read_text(encoding="utf-8")))
    return out


def write_turns(units: list[dict], *, tag: str = "") -> Path:
    path = RAW / f"turns{tag}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for unit in units:
            for row in unit["rows"]:
                handle.write(json.dumps(row, sort_keys=True,
                                        ensure_ascii=False) + "\n")
    return path


# ------------------------------------------------------- inclusion relation

def inclusion_probe(rows: list[dict], calls: list[dict]) -> dict:
    """Decide the inclusion relation from the recorded evidence.

    Two readings compete.  EXCLUSIVE: ``input`` is a fresh bucket disjoint from
    the cache buckets, so gross = input + read + write and one expects
    ``cached > input`` on warm calls (the cache bucket dwarfs the few fresh
    tokens).  INCLUSIVE: ``input`` already contains them, so gross = input and
    one expects ``cached + write <= input`` with the residue equal to fresh.

    Both are checked on every turn AND on every API call, and the verdict is
    the conjunction of an identity and a magnitude ordering.
    """
    def summarize(items: list[dict], label: str) -> dict:
        n = len(items)
        closure = sum(1 for c in items
                      if c["closure_input_eq_cached_plus_write_plus_fresh"])
        le = sum(1 for c in items if c["cached_plus_write_le_input"])
        cached_gt = sum(1 for c in items if c["cached_gt_input"])
        cw_gt = sum(1 for c in items if c["cached_plus_write_gt_input"])
        # discriminating = the cache buckets are non-trivial, so the two
        # readings actually differ
        disc = [c for c in items if c["cached"] + c["write"] > 0]
        return {
            "level": label, "n": n, "discriminating": len(disc),
            "closure_input_eq_cached_plus_write_plus_fresh": closure,
            "cached_plus_write_le_input": le,
            "cached_gt_input": cached_gt,
            "cached_plus_write_gt_input": cw_gt,
            "inclusive_reading_holds": n > 0 and le == n and cached_gt == 0,
            "exclusive_reading_holds": n > 0 and cached_gt == n,
        }

    per_turn = summarize([r["accounting"] for r in rows], "turn (differenced)")
    per_call = summarize(calls, "api call (rollout)")
    total_present = [r for r in rows if r["accounting"]["total_tokens_key_present"]]
    call_total_in_out = sum(1 for c in calls if c["total_eq_input_plus_output"])
    call_total_plus_reason = sum(
        1 for c in calls if c["total_eq_input_plus_output_plus_reasoning"])
    verdict = ("inclusive" if per_call["inclusive_reading_holds"]
               and per_turn["inclusive_reading_holds"]
               else "exclusive" if per_call["exclusive_reading_holds"]
               else "undecided")
    return {
        "per_turn": per_turn, "per_call": per_call,
        "verdict": verdict,
        "gross_definition": (
            "gross = input_tokens (the visible prefix; already contains cached "
            "and write)" if verdict == "inclusive" else
            "gross = input_tokens + cached + write" if verdict == "exclusive"
            else "UNDECIDED — do not price"),
        "total_tokens_on_stdout_usage": {
            "turns_with_total_tokens_key": len(total_present),
            "turns": len(rows),
            "note": "codex exec's stdout usage block carries no total_tokens; "
                    "the rollout route does",
        },
        "total_identity_on_calls": {
            "total_eq_input_plus_output": call_total_in_out,
            "total_eq_input_plus_output_plus_reasoning": call_total_plus_reason,
            "n": len(calls),
            "conclusion": ("reasoning_output_tokens is INSIDE output_tokens; "
                           "output must not be double-counted"
                           if call_total_in_out == len(calls) and len(calls)
                           else "unresolved"),
        },
        "contrast_with_other_cells": {
            "pi_anthropic_and_pi_openai": "exclusive (E17 §3, E17B §2)",
            "openai_raw_http": "inclusive (E16 §1)",
            "codex_metered": "measured here",
        },
    }


# ------------------------------------------------------------- crosscheck

def crosscheck(units: list[dict], warmup_units: list[dict],
               steady: dict | None, authproof: dict | None,
               mtimes: tuple[dict | None, dict | None]) -> dict:
    rows = [r for unit in units for r in unit["rows"]]
    calls = [c for r in rows for c in r["calls"]]

    closure_fail_turn = [r for r in rows
                         if not r["accounting"][
                             "closure_input_eq_cached_plus_write_plus_fresh"]]
    closure_fail_call = [c for c in calls
                         if not c["closure_input_eq_cached_plus_write_plus_fresh"]]
    non_monotone = [r for r in rows if not r["accounting"]["monotone"]]

    # second acquisition route
    route_disagree = [
        {"arm": r["arm"], "session_id": r["session_id"],
         "position": r["position"], "second_route": r["second_route"]}
        for r in rows if not r["second_route"]["per_call_sum_equals_stdout_delta"]
        or not r["second_route"]["rollout_cumulative_equals_stdout"]]

    # H1 three paths
    per_unit = []
    for unit in units:
        tp = unit["three_paths"]
        if unit["arm"] == "karc-full":
            agree = (tp["codex_mcp_tool_call_items"]
                     == tp["db_ingest_observations_mcp"]
                     == tp["guard_hook_pretooluse_mcp"])
        else:
            agree = (tp["codex_mcp_tool_call_items"] == 0
                     == tp["guard_hook_pretooluse_mcp"]
                     and tp["db_ingest_observations_mcp"] is None)
        per_unit.append({
            "arm": unit["arm"], "session_id": unit["session_id"],
            "attempt": unit.get("attempt"),
            "turns_executed": unit["turns_executed"],
            "three_paths": tp, "three_paths_agree": agree,
            "thread_id_sha256": unit.get("thread_id_sha256"),
            "rollout_token_count_events": unit.get(
                "rollout_token_count_events"),
            "rollout_final_equals_stdout_final": (
                unit.get("rollout_final_total") is not None
                and unit.get("stdout_final_cumulative") is not None
                and all(int(unit["rollout_final_total"].get(k, 0) or 0)
                        == int(unit["stdout_final_cumulative"].get(k, 0) or 0)
                        for k in ("input_tokens", "cached_input_tokens",
                                  "cache_write_input_tokens", "output_tokens"))),
        })

    # H2 refusal disjunction
    refusal_rows = [{
        "arm": r["arm"], "session_id": r["session_id"],
        "position": r["position"],
        "signals": {k: v for k, v in r["refusal_signals"].items() if v},
    } for r in rows if any(r["refusal_signals"].values())]

    # cache thresholds and the cold-start assert, at API-call granularity
    eligible = [c for c in calls if c["cache_eligible"]]
    sub_threshold = [c for c in calls if c["sub_threshold"]]
    eligible_dead = [c for c in eligible if c["no_cache_activity"]]
    cold_start_hits = []
    first_eligible_cold = []
    for unit in units:
        found = False
        for row in unit["rows"]:
            for index, call in enumerate(row["calls"]):
                if call["cold_start_suspect"] and call["cache_eligible"]:
                    cold_start_hits.append({
                        "arm": unit["arm"], "session_id": unit["session_id"],
                        "position": row["position"], "call_index": index,
                        "prefix_tokens": call["prefix_tokens"],
                        "cached": call["cached"],
                        "prefix_grew": call.get("prefix_grew"),
                        "cached_over_prefix": (call["cached"]
                                               / call["prefix_tokens"]
                                               if call["prefix_tokens"] else None),
                    })
                if not found and call["cold"] and call["cache_eligible"]:
                    found = True
                    first_eligible_cold.append({
                        "arm": unit["arm"], "session_id": unit["session_id"],
                        "position": row["position"], "call_index": index,
                        "write": call["write"],
                        "prefix_tokens": call["prefix_tokens"]})
        # (loop continues; `found` only gates the FIRST record per unit)

    # non-fatal ErrorItem classification (E20 §7 transport [U])
    unclassified = [{"arm": r["arm"], "session_id": r["session_id"],
                     "position": r["position"],
                     "notices": r["unclassified_notices"]}
                    for r in rows if r.get("unclassified_notices")]

    turns_expected = len(SESSIONS) * len(ARMS) * 8
    turns_done = len(rows)
    incomplete = [{"arm": u["arm"], "session_id": u["session_id"],
                   "turns_executed": u["turns_executed"]}
                  for u in units if u["turns_executed"] != 8]

    priced_total = sum(float(r["priced_usd"]) for r in rows)
    warm_priced = sum(float(r["priced_usd"])
                      for u in warmup_units for r in u["rows"])

    # rate-card verification.  codex exec reports NO cost field, so E19's
    # inversion of a harness-reported cost is unavailable.  Two substitutes:
    #   (a) the published multipliers are recovered from the committed card;
    #   (b) aggregation invariance — pricing per turn and summing must equal
    #       pricing the component totals in one shot.
    recovered = {
        "mu_read": RATE_CARD["read"] / RATE_CARD["input"],
        "mu_write": RATE_CARD["write"] / RATE_CARD["input"],
        "mu_output": RATE_CARD["output"] / RATE_CARD["input"],
    }
    totals = {key: sum(r["accounting"][key] for r in rows)
              for key in ("fresh", "write", "cached", "output")}
    one_shot = priced_usd(totals["fresh"], totals["write"], totals["cached"],
                          totals["output"])
    cost_field_seen = any(
        r.get("turn_completed_non_usage_fields") for r in rows)

    # H10 namespace
    thread_ids = [u.get("thread_id_sha256") for u in units + warmup_units]
    workdirs = [u.get("workdir", "") for u in units + warmup_units]
    ns_ok = all(w.startswith(str(ns_root())) for w in workdirs)
    other_runs = [REPO / "tmp" / name for name in
                  ("e17", "e17b", "e18", "e19")]

    hold = {
        "H1_three_path_mismatch": [u for u in per_unit
                                   if not u["three_paths_agree"]],
        "H2_refusal_or_answerless_turns": refusal_rows,
        "H3_closure_violation_turns": len(closure_fail_turn),
        "H3_closure_violation_calls": len(closure_fail_call),
        "H3_non_monotone_cumulative_turns": len(non_monotone),
        "H4_first_eligible_cold_call_write_zero": [
            f for f in first_eligible_cold if f["write"] == 0],
        "H5_completion_failure_rate": round(1.0 - turns_done / turns_expected, 6),
        "H6_priced_usd_measurement": round(priced_total, 6),
        "H6_priced_usd_warmup": round(warm_priced, 6),
        "H6_priced_usd_cell_total": round(priced_total + warm_priced, 6),
        "H7_user_codex_mtimes_unchanged": (
            None if any(m is None for m in mtimes)
            else mtimes[0]["paths"] == mtimes[1]["paths"]),
        "H8_eligible_call_with_no_cache_activity": len(eligible_dead),
        "H9_cold_start_misattribution_calls": len(cold_start_hits),
        "H10_namespace": {
            "namespace": NAMESPACE,
            "all_workdirs_under_cell_root": ns_ok,
            "cell_root": str(ns_root()),
            "distinct_from_previous_cells": all(
                not str(ns_root()).startswith(str(p)) for p in other_runs),
            "thread_ids_distinct": len(set(thread_ids)) == len(thread_ids),
            "thread_ids": len(thread_ids),
        },
        "H11_metered_route_proved": (
            None if authproof is None else bool(
                authproof["exec_without_credential"]["ok"] is False
                and authproof["exec_without_credential"]["unauthorized_in_error"]
                and authproof["status_before_any_credential"]["auth_type"]
                == "not-logged-in"
                and authproof["status_after_credential"]["auth_type"] == "api-key"
                and authproof["exec_with_credential"]["ok"] is True)),
        "H12_steady_state_not_reached": (
            None if steady is None else bool(steady.get("H12_fired"))),
    }
    fired = []
    if hold["H1_three_path_mismatch"]:
        fired.append("H1")
    if hold["H2_refusal_or_answerless_turns"]:
        fired.append("H2")
    if (hold["H3_closure_violation_turns"] or hold["H3_closure_violation_calls"]
            or hold["H3_non_monotone_cumulative_turns"]):
        fired.append("H3")
    if hold["H4_first_eligible_cold_call_write_zero"]:
        fired.append("H4")
    if hold["H5_completion_failure_rate"] > 0.10:
        fired.append("H5")
    if hold["H6_priced_usd_cell_total"] > BUDGET_CAP_USD:
        fired.append("H6")
    if hold["H7_user_codex_mtimes_unchanged"] is False:
        fired.append("H7")
    if hold["H8_eligible_call_with_no_cache_activity"]:
        fired.append("H8")
    if hold["H9_cold_start_misattribution_calls"]:
        fired.append("H9")
    h10 = hold["H10_namespace"]
    if not (h10["all_workdirs_under_cell_root"]
            and h10["distinct_from_previous_cells"]
            and h10["thread_ids_distinct"]):
        fired.append("H10")
    if hold["H11_metered_route_proved"] is False:
        fired.append("H11")
    if hold["H12_steady_state_not_reached"]:
        fired.append("H12")

    return {
        "turns_expected": turns_expected, "turns_executed": turns_done,
        "completion_rate": round(turns_done / turns_expected, 6),
        "incomplete_units": incomplete,
        "api_calls": len(calls),
        "api_calls_per_arm": {arm: sum(r["api_calls"] for r in rows
                                       if r["arm"] == arm) for arm in ARMS},
        "inclusion_relation": inclusion_probe(rows, calls),
        "second_acquisition_route": {
            "what": "the isolated home's rollout token_count events, one per "
                    "API call, carrying last_token_usage and total_token_usage",
            "turns_with_agreement": len(rows) - len(route_disagree),
            "turns": len(rows),
            "disagreements": route_disagree,
            "units_whose_rollout_final_equals_stdout_final": sum(
                1 for u in per_unit if u["rollout_final_equals_stdout_final"]),
            "units": len(per_unit),
        },
        "openai_cache_threshold": {
            "min_prefix_tokens": OPENAI_CACHE_MIN_PREFIX,
            "eligible_calls": len(eligible),
            "sub_threshold_calls": len(sub_threshold),
            "sub_threshold_prefix_tokens": sorted(
                {c["prefix_tokens"] for c in sub_threshold}),
            "eligible_calls_with_no_cache_activity": len(eligible_dead),
            "first_eligible_cold_call_per_unit": first_eligible_cold,
        },
        "H9_cold_start_assert": {
            "what": "an API call with write == 0 while the prefix is served "
                    "from cache, i.e. served an entry written by an execution "
                    "outside this dataset (the assert that caught E18 leg B "
                    "attempt 2)",
            "calls": len(cold_start_hits),
            "clean": not cold_start_hits,
            "detail": cold_start_hits[:50],
        },
        "stop_reason": {
            "what": "codex exec emits no stop reason (E20 §5); H2 is the "
                    "three-signal disjunction instead",
            "turns_with_stop_reason_field": sum(
                1 for r in rows if r.get("stop_reason_field_present")),
            "turn_completed_non_usage_fields_nonempty": sum(
                1 for r in rows if r.get("turn_completed_non_usage_fields")),
            "cost_field_ever_present": cost_field_seen,
        },
        "nonfatal_error_items": {
            "what": "documented non-fatal ErrorItems.  Two per turn are the "
                    "--dangerously-bypass-hook-trust notice (fixed body, "
                    "stable digest).  A third, longer notice would be the "
                    "WebSocket -> HTTPS transport fallback.",
            "turns": len(rows),
            "total_items": sum(r.get("nonfatal_error_items", 0) for r in rows),
            "hook_trust_notices": sum(r.get("hook_trust_notices", 0)
                                      for r in rows),
            "turns_with_unclassified_notice": len(unclassified),
            "unclassified": unclassified[:20],
            "transport_stayed_on_websocket": not unclassified,
        },
        "rate_card_verification": {
            "harness_reported_cost_field": cost_field_seen,
            "note": "codex exec reports no cost, so E19's inversion of a "
                    "harness-reported cost is NOT available here.  The two "
                    "substitutes below are what this harness admits.",
            "published_multipliers": PUBLISHED_MULTIPLIERS,
            "recovered_from_committed_card": recovered,
            "multipliers_match": all(
                abs(recovered[k] - PUBLISHED_MULTIPLIERS[k]) < 1e-12
                for k in PUBLISHED_MULTIPLIERS),
            "cache_write_rule_1_25x_input": abs(
                RATE_CARD["write"] - 1.25 * RATE_CARD["input"]) < 1e-12,
            "aggregation_invariance": {
                "sum_of_per_turn_priced_usd": round(priced_total, 10),
                "priced_from_component_totals": round(one_shot, 10),
                "abs_residual_usd": abs(priced_total - one_shot),
                "exact_within_1e_9_usd": abs(priced_total - one_shot) < 1e-9,
            },
        },
        "per_unit": per_unit,
        "cost": {
            "priced_usd_measurement": round(priced_total, 6),
            "priced_usd_warmup": round(warm_priced, 6),
            "priced_usd_cell_total": round(priced_total + warm_priced, 6),
            "cap_usd": BUDGET_CAP_USD,
            "rate_card_source": RATE_CARD["source"],
            "warmup_note": "warm-up cost is reported but enters no judgment",
        },
        "steady_state": steady,
        "warmup_window": {
            "sessions": WARMUP, "arms": ["karc-full"],
            "units": len(warmup_units),
            "turns": sum(len(u["rows"]) for u in warmup_units)},
        "measurement_window": {"sessions": SESSIONS, "arms": list(ARMS)},
        "incidental_grade_diagnostic": {
            "note": "NOT a claim (preregistration §6 excludes the accuracy "
                    "axis).  Recorded so a silently non-functioning arm is "
                    "detectable and because it independently reproduces the "
                    "position-1 fixture floor.",
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
    out: dict[str, dict[str, dict[str, float]]] = {}
    for unit in units:
        arm = out.setdefault(unit["arm"], {})
        acc = {"uncached": 0, "creation": 0, "cache_read": 0, "output": 0,
               "reasoning": 0, "gross": 0, "api_calls": 0, "tool_calls": 0,
               "turns": 0}
        for row in unit["rows"]:
            a = row["accounting"]
            acc["uncached"] += a["fresh"]
            acc["creation"] += a["write"]
            acc["cache_read"] += a["cached"]
            acc["output"] += a["output"]
            acc["reasoning"] += a["reasoning"]
            acc["gross"] += a["gross_inclusive"]
            acc["api_calls"] += row["api_calls"]
            acc["tool_calls"] += row["mcp_calls_codex_events"]
            acc["turns"] += 1
        arm[unit["session_id"]] = acc
    return out


def statistics_block(units: list[dict], matrix: list[list[int]]) -> dict:
    comp = per_session_components(units)
    if set(comp) != set(ARMS) or any(set(comp[a]) != set(SESSIONS)
                                     for a in ARMS):
        return {"incomplete": True,
                "arms_present": {a: sorted(comp.get(a, {})) for a in ARMS}}

    def per_session(arm: str, key: str) -> dict[str, float]:
        return {s: float(comp[arm][s][key]) for s in SESSIONS}

    def priced_per_session(arm: str) -> dict[str, float]:
        return {s: priced_usd(comp[arm][s]["uncached"], comp[arm][s]["creation"],
                              comp[arm][s]["cache_read"], comp[arm][s]["output"])
                for s in SESSIONS}

    gross_k = per_session("karc-full", "gross")
    gross_r = per_session("rag-bm25", "gross")
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

    mu_r = RATE_CARD["read"] / RATE_CARD["input"]
    mu_o = RATE_CARD["output"] / RATE_CARD["input"]
    mu_w = RATE_CARD["write"] / RATE_CARD["input"]
    dU, dW, dR, dO = (delta["uncached"], delta["creation"],
                      delta["cache_read"], delta["output"])
    # dU + mu_w*dW + mu_r*dR + mu_o*dO = 0, solved from THIS cell's totals
    mu_w_star = None if dW == 0 else -(dU + mu_r * dR + mu_o * dO) / dW

    per_position = {}
    for arm in ARMS:
        rows = [r for unit in units if unit["arm"] == arm for r in unit["rows"]]
        per_position[arm] = {str(pos): {
            "n": len([r for r in rows if r["position"] == pos]),
            "gross_mean": statistics.mean(
                [r["accounting"]["gross_inclusive"] for r in rows
                 if r["position"] == pos]),
            "creation_mean": statistics.mean(
                [r["accounting"]["write"] for r in rows if r["position"] == pos]),
            "read_mean": statistics.mean(
                [r["accounting"]["cached"] for r in rows if r["position"] == pos]),
            "output_mean": statistics.mean(
                [r["accounting"]["output"] for r in rows if r["position"] == pos]),
            "api_calls_mean": statistics.mean(
                [r["api_calls"] for r in rows if r["position"] == pos]),
        } for pos in range(1, 9)}

    # harness fixed cost: the cold first API call of every unit is the whole
    # standing prefix (Codex base prompt + AGENTS.md + task 1), so it measures
    # what the harness charges before any conversation exists.
    cold_prefix = {}
    single_call_prefix = {}
    for arm in ARMS:
        firsts = [unit["rows"][0]["calls"][0]["prefix_tokens"]
                  for unit in units if unit["arm"] == arm and unit["rows"]
                  and unit["rows"][0]["calls"]]
        cold_prefix[arm] = {
            "n": len(firsts),
            "mean": statistics.mean(firsts) if firsts else None,
            "min": min(firsts) if firsts else None,
            "max": max(firsts) if firsts else None}
        prefixes = [c["prefix_tokens"] for unit in units
                    if unit["arm"] == arm for r in unit["rows"]
                    if r["api_calls"] == 1 for c in r["calls"]]
        single_call_prefix[arm] = {
            "n": len(prefixes),
            "mean": statistics.mean(prefixes) if prefixes else None,
            "min": min(prefixes) if prefixes else None,
            "max": max(prefixes) if prefixes else None}

    return {
        "rate_card_usd_per_million": {k: v for k, v in RATE_CARD.items()
                                      if k != "source"},
        "rate_card_source": RATE_CARD["source"],
        "multipliers": {"mu_read": mu_r, "mu_write": mu_w, "mu_output": mu_o},
        "component_totals": totals,
        "component_delta_karc_minus_rag": delta,
        "priced_totals_usd": {arm: sum(priced_per_session(arm).values())
                              for arm in ARMS},
        "gross_ratio_karc_over_rag": gross_ratio,
        "priced_ratio_karc_over_rag": priced_ratio,
        "paired_mean_session_gross_difference": gross_diff,
        "paired_mean_session_priced_usd_difference": priced_diff,
        "reversal_threshold_mu_w_star": mu_w_star,
        "observed_mu_w": mu_w,
        "mu_w_star_below_observed": (None if mu_w_star is None
                                     else mu_w_star < mu_w),
        "per_session_gross": {"karc-full": gross_k, "rag-bm25": gross_r},
        "per_session_priced_usd": {"karc-full": priced_k, "rag-bm25": priced_r},
        "per_position_profile": per_position,
        "cold_first_call_prefix_tokens": cold_prefix,
        "single_api_call_turn_prefix_tokens": single_call_prefix,
        "bootstrap": {"method": "paired percentile bootstrap over sessions, "
                                "common random numbers",
                      "seed": BOOTSTRAP_SEED,
                      "resamples": BOOTSTRAP_RESAMPLES,
                      "ci": "two-sided 95% percentile",
                      "unit": "session", "n_sessions": len(SESSIONS)},
    }


def e19_leg_b_contrast(stats: dict) -> dict:
    """Provider, model, rate card and task sequence fixed; harness varied.

    Only E19's committed leg B figures are read.  E19's own gross definition is
    ``pi.input + cacheRead + cacheWrite`` and this cell's is ``input_tokens``,
    which under the measured inclusive reading is ``fresh + write + cached`` —
    the SAME formula in different fields, so the two columns are comparable in
    construction rather than merely in name.
    """
    if not E19_ANALYSIS.exists() or stats.get("incomplete"):
        return {"available": False}
    e19 = json.loads(E19_ANALYSIS.read_text(encoding="utf-8"))
    leg = (e19.get("legs") or {}).get("B")
    if not leg:
        return {"available": False}
    rows: dict = {"available": True,
                  "source": str(E19_ANALYSIS.relative_to(REPO)),
                  "held_fixed": ["provider openai", "model gpt-5.6-luna",
                                 "published luna rate card",
                                 "schedule seed 4352 / identical tasks_sha256",
                                 "measurement window S09..S20",
                                 "arms karc-full and rag-bm25",
                                 "statistics: session bootstrap seed 1313"],
                  "varied": ["harness: pi 0.84.3 -> Codex CLI 0.145.0"],
                  "gross_formula": {
                      "E19_leg_B": "pi input(fresh) + cacheRead + cacheWrite",
                      "E21": "codex input_tokens == fresh + write + cached",
                      "same_construction": True}}
    for arm in ARMS:
        e19t, e21t = leg["component_totals"][arm], stats["component_totals"][arm]
        rows[arm] = {
            "E19_leg_B": {"gross": e19t["gross"], "uncached": e19t["uncached"],
                          "creation": e19t["creation"],
                          "cache_read": e19t["cache_read"],
                          "output": e19t["output"],
                          "api_calls": e19t["api_calls"],
                          "tool_calls": e19t["tool_calls"]},
            "E21": {"gross": e21t["gross"], "uncached": e21t["uncached"],
                    "creation": e21t["creation"],
                    "cache_read": e21t["cache_read"],
                    "output": e21t["output"],
                    "api_calls": e21t["api_calls"],
                    "tool_calls": e21t["tool_calls"]},
        }
        for key in ("gross", "uncached", "creation", "cache_read", "output"):
            base = e19t[key]
            rows[arm].setdefault("E21_over_E19", {})[key] = (
                e21t[key] / base if base else None)
        for key in ("uncached", "creation", "cache_read", "output"):
            rows[arm].setdefault("share_of_gross", {})[key] = {
                "E19_leg_B": e19t[key] / e19t["gross"] if e19t["gross"] else None,
                "E21": e21t[key] / e21t["gross"] if e21t["gross"] else None}
    rows["judgments"] = {
        "E19_leg_B": {
            "gross_ratio": leg["gross_ratio_karc_over_rag"]["point"],
            "gross_ratio_ci95": [leg["gross_ratio_karc_over_rag"]["ci95_lo"],
                                 leg["gross_ratio_karc_over_rag"]["ci95_hi"]],
            "priced_ratio": leg["priced_ratio_karc_over_rag"]["point"],
            "priced_ratio_ci95": [leg["priced_ratio_karc_over_rag"]["ci95_lo"],
                                  leg["priced_ratio_karc_over_rag"]["ci95_hi"]],
            "priced_totals_usd": leg["priced_totals_usd"],
            "mu_w_star": leg["reversal_threshold_mu_w_star"],
            "component_delta_karc_minus_rag": leg[
                "component_delta_karc_minus_rag"],
        },
        "E21": {
            "gross_ratio": stats["gross_ratio_karc_over_rag"]["point"],
            "gross_ratio_ci95": [stats["gross_ratio_karc_over_rag"]["ci95_lo"],
                                 stats["gross_ratio_karc_over_rag"]["ci95_hi"]],
            "priced_ratio": stats["priced_ratio_karc_over_rag"]["point"],
            "priced_ratio_ci95": [stats["priced_ratio_karc_over_rag"]["ci95_lo"],
                                  stats["priced_ratio_karc_over_rag"]["ci95_hi"]],
            "priced_totals_usd": stats["priced_totals_usd"],
            "mu_w_star": stats["reversal_threshold_mu_w_star"],
            "component_delta_karc_minus_rag": stats[
                "component_delta_karc_minus_rag"],
        },
    }
    rows["harness_fixed_cost"] = {
        "what": "the standing prefix a harness charges before any conversation "
                "exists, i.e. the cold first API call of a unit",
        "E21_cold_first_call_prefix_tokens": stats["cold_first_call_prefix_tokens"],
        "E19_leg_B_single_call_turn_prefix_tokens": leg[
            "single_api_call_turn_prefix_tokens"],
        "E21_single_call_turn_prefix_tokens": stats[
            "single_api_call_turn_prefix_tokens"],
        "note": "E19 reported the prefix of single-API-call turns; both are "
                "given so the comparison is like for like",
        "codex_own_standing_prompt": _codex_bare_prefix(),
    }
    return rows


def _codex_bare_prefix() -> dict:
    """The Codex CLI's OWN standing prompt, isolated.

    The layout diagnostic ran a turn in a bare git workdir with no AGENTS.md,
    no MCP server and a ~20-token user message, so its cold prefix is almost
    entirely the harness's own system prompt.  That makes the harness fixed
    cost a measured quantity here rather than a residual inferred from arm
    totals.
    """
    path = RAW / "probe-home-layout.json"
    if not path.exists():
        return {"available": False}
    probe = json.loads(path.read_text(encoding="utf-8"))
    first = (probe.get("turns") or [{}])[0].get("usage_raw") or {}
    return {
        "available": True,
        "source": "raw/probe-home-layout.json",
        "cold_prefix_tokens": int(first.get("input_tokens", 0) or 0),
        "cache_write_on_that_call": int(
            first.get("cache_write_input_tokens", 0) or 0),
        "conditions": "bare git workdir, no AGENTS.md, no MCP server, "
                      "reasoning_effort low, ~20-token user message",
    }


def descriptive_window_profile(stats: dict) -> dict:
    """Descriptive only.  Per-session ratios INSIDE the single preregistered
    window; no sub-range is judged and no criterion attaches to any of it.

    It is reported because R2 failed and a failure that is only reported as a
    pooled number tells the reader nothing about where it came from.  The
    window's reuse mixture was recorded in raw/slices.json BEFORE execution
    (E19 §5 disclosed the same profile only afterwards), so relating the
    per-session spread to it is not a post-hoc slice.
    """
    if stats.get("incomplete"):
        return {"available": False}
    gk = stats["per_session_gross"]["karc-full"]
    gr = stats["per_session_gross"]["rag-bm25"]
    pk = stats["per_session_priced_usd"]["karc-full"]
    pr = stats["per_session_priced_usd"]["rag-bm25"]
    rows = {s: {"gross_ratio": gk[s] / gr[s], "priced_ratio": pk[s] / pr[s]}
            for s in SESSIONS}
    return {
        "caveat": "descriptive; the judgment is the pooled test over the whole "
                  "preregistered window, and no sub-range of it is selected, "
                  "tested or re-pooled",
        "per_session": rows,
        "sessions_with_gross_ratio_above_one": sum(
            1 for s in SESSIONS if rows[s]["gross_ratio"] > 1.0),
        "sessions_with_priced_ratio_below_one": sum(
            1 for s in SESSIONS if rows[s]["priced_ratio"] < 1.0),
        "n_sessions": len(SESSIONS),
        "priced_ratio_min": min(rows[s]["priced_ratio"] for s in SESSIONS),
        "priced_ratio_max": max(rows[s]["priced_ratio"] for s in SESSIONS),
    }


def harness_fixed_cost_attribution(stats: dict) -> dict:
    """How much of the read-component delta the harness's own preamble explains.

    Codex re-sends the whole visible prefix on every API call, and the karc arm
    makes more API calls than the rag arm because each MCP round trip is one.
    The Codex standing prompt was measured in isolation (bare git workdir, no
    AGENTS.md, no MCP server), so the extra calls carry a KNOWN floor of
    harness-owned tokens.  Multiplying that floor by the extra calls gives a
    LOWER BOUND on the part of the read delta that is harness overhead rather
    than anything about either arm's knowledge strategy.
    """
    if stats.get("incomplete"):
        return {"available": False}
    probe = _codex_bare_prefix()
    if not probe.get("available"):
        return {"available": False}
    delta = stats["component_delta_karc_minus_rag"]
    extra_calls = delta["api_calls"]
    floor = probe["cold_prefix_tokens"]
    lower_bound = extra_calls * floor
    return {
        "available": True,
        "codex_standing_prompt_tokens": floor,
        "extra_api_calls_karc_minus_rag": extra_calls,
        "read_delta_karc_minus_rag": delta["cache_read"],
        "harness_preamble_lower_bound_tokens": lower_bound,
        "share_of_read_delta": (lower_bound / delta["cache_read"]
                                if delta["cache_read"] else None),
        "why_a_lower_bound": "those extra calls re-read the whole accumulated "
                             "prefix, not only the standing prompt, so the "
                             "true harness-attributable share is larger",
        "caveat": "E19 never measured pi's standing prompt in isolation, so no "
                  "like-for-like number exists on the pi side [U]",
    }


def judge(stats: dict, checks: dict) -> dict:
    if stats.get("incomplete"):
        return {"status": "NOT RUN", "arms_present": stats.get("arms_present")}
    d = stats["component_delta_karc_minus_rag"]
    pr = stats["priced_ratio_karc_over_rag"]
    pd = stats["paired_mean_session_priced_usd_difference"]
    gr = stats["gross_ratio_karc_over_rag"]
    gd = stats["paired_mean_session_gross_difference"]
    mu_w_star = stats["reversal_threshold_mu_w_star"]
    r1 = "PASS" if (d["cache_read"] > 0 and d["creation"] < 0) else "FAIL"
    r2 = "PASS" if (pr["entirely_below_one"] and pd["excludes_zero"]) else "FAIL"
    r3 = ("PASS" if (mu_w_star is not None
                     and mu_w_star < stats["observed_mu_w"]) else "FAIL")
    return {
        "R1_component_signature_sign_preserved": {
            "verdict": r1,
            "criterion": "delta_read > 0 AND delta_creation < 0",
            "delta_read": d["cache_read"], "delta_creation": d["creation"],
            "delta_uncached": d["uncached"], "delta_output": d["output"],
        },
        "R2_component_priced_ordering": {
            "verdict": r2,
            "criterion": "paired priced ratio 95% CI entirely below 1.0 AND "
                         "paired mean USD difference CI excludes 0",
            "priced_ratio": pr["point"],
            "priced_ratio_ci95": [pr["ci95_lo"], pr["ci95_hi"]],
            "priced_ratio_ci_entirely_below_one": pr["entirely_below_one"],
            "paired_mean_usd_difference": pd["point"],
            "paired_mean_usd_difference_ci95": [pd["ci95_lo"], pd["ci95_hi"]],
            "usd_difference_ci_excludes_zero": pd["excludes_zero"],
        },
        "R3_reversal_threshold_below_actual_rate": {
            "verdict": r3,
            "criterion": "mu_w_star from THIS cell's own component totals < 1.25",
            "mu_w_star": mu_w_star,
            "observed_mu_w": stats["observed_mu_w"],
            "note": "computed from this cell's totals only; no threshold is "
                    "carried over from another cell (E18's failure mode)",
        },
        "gross_ordering_measured_not_predicted": {
            "status": "REPORTED ONLY — the preregistration §3 declines to "
                      "predict the gross direction",
            "gross_ratio": gr["point"],
            "gross_ratio_ci95": [gr["ci95_lo"], gr["ci95_hi"]],
            "ci_entirely_above_one": gr["entirely_above_one"],
            "ci_entirely_below_one": gr["entirely_below_one"],
            "paired_mean_session_gross_difference": gd["point"],
            "paired_mean_session_gross_difference_ci95": [gd["ci95_lo"],
                                                          gd["ci95_hi"]],
        },
        "inclusion_relation_verdict": checks["inclusion_relation"]["verdict"],
        "steady_state_reached": (checks.get("steady_state") or {}).get(
            "steady_state_reached"),
        "hold_fired": checks["hold_fired"],
        "cost": checks["cost"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ns", default="")
    args = parser.parse_args()
    global NAMESPACE
    NAMESPACE = args.ns

    units = load_units(SESSIONS)
    warmup_units = load_units(WARMUP, arms=("karc-full",))
    if not units:
        raise SystemExit("E21: no measurement units found")
    write_turns(units)
    if warmup_units:
        write_turns(warmup_units, tag="-warmup")

    steady_path = RAW / "steady.json"
    steady = (json.loads(steady_path.read_text(encoding="utf-8"))
              if steady_path.exists() else None)
    auth_path = RAW / "authproof.json"
    authproof = (json.loads(auth_path.read_text(encoding="utf-8"))
                 if auth_path.exists() else None)
    before = RAW / "mtime-before.json"
    after = RAW / "mtime-after.json"
    mtimes = (json.loads(before.read_text(encoding="utf-8"))
              if before.exists() else None,
              json.loads(after.read_text(encoding="utf-8"))
              if after.exists() else None)

    checks = crosscheck(units, warmup_units, steady, authproof, mtimes)
    (RAW / "crosscheck.json").write_text(
        json.dumps(checks, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8")

    matrix = resample_matrix()
    stats = statistics_block(units, matrix)
    analysis = {
        "cell": "E21-CODEX-COMPONENT",
        "preregistration": "docs/experiments/E21-CODEX-COMPONENT/preregistration.md",
        "harness": {"name": "codex-cli", "version": "codex-cli 0.145.0",
                    "route": "metered OpenAI API key",
                    "provider": "openai", "model": "gpt-5.6-luna",
                    "reasoning_effort": "low"},
        "measurement_window": {"warmup_sessions": WARMUP,
                               "measured_sessions": SESSIONS,
                               "arms": list(ARMS),
                               "note": "one preregistered window, pooled once; "
                                       "no sub-range is selected or tested"},
        "statistics": stats,
        "judgments": judge(stats, checks),
        "e19_leg_b_contrast": e19_leg_b_contrast(stats),
        "descriptive_window_profile": descriptive_window_profile(stats),
        "harness_fixed_cost_attribution": harness_fixed_cost_attribution(stats),
        "accounting_contract": {
            "reporting_level": "codex exec turn.completed.usage is the "
                               "CUMULATIVE thread total; per-turn components "
                               "are consecutive differences (measured, see "
                               "raw/probe-home-layout.json)",
            "gross": checks["inclusion_relation"]["gross_definition"],
            "priced": "fresh*0.20 + write*0.25 + cached*0.02 + output*1.20 "
                      "per million",
        },
        "_meta": {"generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                    time.gmtime()),
                  "bootstrap_seed": BOOTSTRAP_SEED,
                  "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                  "namespace": NAMESPACE},
    }
    (RAW / "analysis.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8")
    print(f"turns {checks['turns_executed']}/{checks['turns_expected']} "
          f"hold_fired={checks['hold_fired']} "
          f"cell_usd={checks['cost']['priced_usd_cell_total']}")
    print(json.dumps(analysis["judgments"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
