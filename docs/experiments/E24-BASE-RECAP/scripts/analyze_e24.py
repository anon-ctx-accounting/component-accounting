#!/usr/bin/env python3
"""E24-BASE-RECAP 판정 — 사전등록서 §4의 P1~P5.

추정량은 새로 쓰지 않고 동결 cell의 것을 가져다 쓴다.  부트스트랩 헬퍼는
E13-RECOMP 의 것을 그대로 import 하고(시드 1313 · 10,000회 · 공통 난수),
성분 총계와 비는 E18 의 ``baseline_profile`` 과 같은 식으로 계산한다.
모델 호출 0.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
CELL = HERE.parents[1]
REPO = HERE.parents[4]
assert (REPO / "src" / "karc").is_dir(), f"repo root 오판: {REPO}"

RAW = CELL / "run" / "raw"
FROZEN_TURNS = REPO / "docs/experiments/E10-P2-XR/canary/raw/turns.jsonl"
ARMS = ("karc-full", "rag-bm25")

# 공개 요금표 (USD/1M) — 동결 cell 이 rate_card_equivalent_usd 계산에 쓴 값과 같다.
CARD = {"input": 3.00, "write_5m": 3.75, "write_1h": 6.00,
        "read": 0.30, "output": 15.00}
MU_R = CARD["read"] / CARD["input"]
MU_O = CARD["output"] / CARD["input"]
MU_W_1H = CARD["write_1h"] / CARD["input"]

# E13 의 부트스트랩을 그대로 쓴다.
_spec = importlib.util.spec_from_file_location(
    "e13_recomp", REPO / "docs/experiments/E13-RECOMP/scripts/e13_recomp.py")
e13 = importlib.util.module_from_spec(_spec)
sys.modules["e13_recomp"] = e13
_spec.loader.exec_module(e13)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows_of(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()
            if x.strip()]


def totals(rows: list[dict]) -> dict:
    out = {}
    for arm in ARMS:
        unit = [r for r in rows if r["arm"] == arm]
        out[arm] = {
            "turns": len(unit),
            "uncached": sum(r["fresh_input_tokens_provider"] for r in unit),
            "creation": sum(r["cache_creation_input_tokens_provider"] for r in unit),
            "creation_1h": sum(r["cache_creation_1h_input_tokens_provider"] for r in unit),
            "creation_5m": sum(r["cache_creation_5m_input_tokens_provider"] for r in unit),
            "cache_read": sum(r["cache_read_input_tokens_provider"] for r in unit),
            "output": sum(r["output_tokens"] for r in unit),
            "gross": sum(r["gross_input_tokens"] for r in unit),
            "priced_usd": sum(r["rate_card_equivalent_usd"] for r in unit),
            "tool_calls": sum(r["tool_calls"] for r in unit),
        }
        out[arm]["api_calls"] = out[arm]["turns"] + out[arm]["tool_calls"]
        out[arm]["mean_prefix_tokens_per_call"] = (
            out[arm]["gross"] / out[arm]["api_calls"])
        out[arm]["api_calls_per_turn"] = out[arm]["api_calls"] / out[arm]["turns"]
    return out


def per_session(rows: list[dict], sessions: list[str], arm: str, field: str) -> dict:
    agg = {s: 0.0 for s in sessions}
    for r in rows:
        if r["arm"] == arm:
            agg[r["session_id"]] += r[field]
    return agg


def profile(rows: list[dict], sessions: list[str], matrix) -> dict:
    tot = totals(rows)
    delta = {k: tot["karc-full"][k] - tot["rag-bm25"][k]
             for k in tot["karc-full"]}
    gk = per_session(rows, sessions, "karc-full", "gross_input_tokens")
    gr = per_session(rows, sessions, "rag-bm25", "gross_input_tokens")
    pk = per_session(rows, sessions, "karc-full", "rate_card_equivalent_usd")
    pr = per_session(rows, sessions, "rag-bm25", "rate_card_equivalent_usd")
    dW = delta["creation"]
    mu_w_star = (None if dW == 0 else
                 -(delta["uncached"] + MU_R * delta["cache_read"]
                   + MU_O * delta["output"]) / dW)
    per_position = {}
    for arm in ARMS:
        unit = [r for r in rows if r["arm"] == arm]
        per_position[arm] = {str(pos): {
            "n": len([r for r in unit if r["position"] == pos]),
            "gross_mean": (sum(r["gross_input_tokens"] for r in unit
                               if r["position"] == pos)
                           / max(1, len([r for r in unit if r["position"] == pos]))),
        } for pos in sorted({r["position"] for r in unit})}
    return {
        "component_totals": tot,
        "per_position_profile": per_position,
        "component_delta_karc_minus_rag": delta,
        "gross_ratio": tot["karc-full"]["gross"] / tot["rag-bm25"]["gross"],
        "priced_ratio": tot["karc-full"]["priced_usd"] / tot["rag-bm25"]["priced_usd"],
        "gross_ratio_boot": e13.paired_bootstrap_ratio_of_sums(gk, gr, matrix),
        "priced_ratio_boot": e13.paired_bootstrap_ratio_of_sums(pk, pr, matrix),
        "paired_mean_session_priced_usd_difference": e13.paired_bootstrap(
            {s: pk[s] - pr[s] for s in sessions}, matrix),
        "paired_mean_session_gross_difference": e13.paired_bootstrap(
            {s: float(gk[s] - gr[s]) for s in sessions}, matrix),
        "reversal_threshold_mu_w_star": mu_w_star,
        "observed_mu_w": MU_W_1H,
        "per_session_ratios": {s: {"gross_ratio": gk[s] / gr[s],
                                   "priced_ratio": pk[s] / pr[s]}
                               for s in sessions},
    }


def p1_cost_inversion(rows: list[dict]) -> dict:
    """harness 보고 금액이 기록된 네 성분에서 복원되는가."""
    residuals, missing, detail = [], 0, []
    for r in rows:
        reported = r.get("harness_reported_cost_usd")
        if reported is None:
            missing += 1
            continue
        model = (r["fresh_input_tokens_provider"] * CARD["input"]
                 + r["cache_creation_1h_input_tokens_provider"] * CARD["write_1h"]
                 + r["cache_creation_5m_input_tokens_provider"] * CARD["write_5m"]
                 + r["cache_read_input_tokens_provider"] * CARD["read"]
                 + r["output_tokens"] * CARD["output"]) / 1e6
        residuals.append(abs(model - reported))
        detail.append({"arm": r["arm"], "session_id": r["session_id"],
                       "position": r["position"], "reported": reported,
                       "modelled": model, "abs_residual": abs(model - reported)})
    detail.sort(key=lambda d: -d["abs_residual"])
    n = len(rows)
    return {
        "turns": n,
        "turns_with_reported_cost": n - missing,
        "missing": missing,
        "missing_share": (missing / n) if n else None,
        "max_abs_residual_usd": max(residuals) if residuals else None,
        "exact_within_1e_9_usd": (max(residuals) < 1e-9) if residuals else None,
        "reported_total_usd": sum(r["harness_reported_cost_usd"] or 0.0 for r in rows),
        "our_priced_total_usd": sum(r["rate_card_equivalent_usd"] for r in rows),
        "worst_five": detail[:5],
        "what_this_checks":
            "the amount the harness reports for each call is reproduced from the "
            "four recorded components at the published card.  The amount is the "
            "harness's own rate-card conversion, not a billed charge.",
    }


def attach_cost(turns: list[dict], attempts: list[dict]) -> dict:
    """attempt 행의 금액을 turn 행에 붙인다.

    동결 실행기는 turn 행을 attempt 에서 고른 키만 인라인으로 복사해 만들고
    ``validate_persisted_row`` 로 검증하므로, wrapper 가 attempt 에 붙인 금액은
    turn 행에 자동으로 옮겨가지 않는다.  두 파일을
    (arm, session_id, session_attempt, position) 으로 join 하며, join 이 옳은지는
    성분 다섯 개가 양쪽에서 일치하는지로 교차검증한다.
    """
    def key(r: dict) -> tuple:
        return (r["arm"], r["session_id"], r["session_attempt"], r["position"])

    index: dict[tuple, dict] = {}
    for row in attempts:
        k = key(row)
        if k in index:
            raise SystemExit(f"attempt 키 중복: {k}")
        index[k] = row
    crosscheck = ("fresh_input_tokens_provider", "cache_read_input_tokens_provider",
                  "cache_creation_input_tokens_provider", "gross_input_tokens",
                  "output_tokens", "rate_card_equivalent_usd")
    joined = mismatched = 0
    for turn in turns:
        src = index.get(key(turn))
        if src is None:
            raise SystemExit(f"join 실패: {key(turn)}")
        if any(src[f] != turn[f] for f in crosscheck):
            mismatched += 1
            continue
        turn["harness_reported_cost_usd"] = src.get("harness_reported_cost_usd")
        turn["harness_reported_model_cost_usd"] = src.get(
            "harness_reported_model_cost_usd") or {}
        joined += 1
    return {"turns": len(turns), "joined": joined, "component_mismatched": mismatched,
            "key": "arm + session_id + session_attempt + position",
            "crosscheck_fields": list(crosscheck)}


def main() -> int:
    turns_path = RAW / "turns.jsonl"
    rows = rows_of(turns_path)
    join = attach_cost(rows, rows_of(RAW / "attempts.jsonl"))
    if join["component_mismatched"]:
        raise SystemExit(f"join 교차검증 실패: {join}")
    sessions = sorted({r["session_id"] for r in rows})
    matrix = e13.resample_matrix()
    assert len(sessions) == len(e13.SESSIONS), (
        f"세션 수 불일치: {len(sessions)} vs {len(e13.SESSIONS)}")
    # 부트스트랩 행렬은 E13 의 SESSIONS 순서를 전제한다.
    assert sessions == list(e13.SESSIONS), "세션 ID 집합이 동결 cell과 다르다"

    now = profile(rows, sessions, matrix)
    frozen_rows = rows_of(FROZEN_TURNS)
    frozen = profile(frozen_rows, sessions, matrix)
    p1 = p1_cost_inversion(rows)

    def within(a: float, b: float, tol: float) -> bool:
        return abs(a - b) / abs(b) <= tol

    d_now, d_fr = (now["component_delta_karc_minus_rag"],
                   frozen["component_delta_karc_minus_rag"])
    judgments = {
        "P1_harness_cost_inverts": bool(p1["exact_within_1e_9_usd"]),
        "P2_component_sign_preserved": bool(
            d_now["cache_read"] > 0 and d_now["creation"] < 0),
        "P3_priced_ci_below_one": bool(now["priced_ratio_boot"]["ci95_hi"] < 1.0),
        "P4_gross_ci_above_one": bool(now["gross_ratio_boot"]["ci95_lo"] > 1.0),
        "P5_prefix_within_20pct": {
            arm: within(now["component_totals"][arm]["mean_prefix_tokens_per_call"],
                        frozen["component_totals"][arm]["mean_prefix_tokens_per_call"],
                        0.20) for arm in ARMS},
    }
    judgments["P5_pass"] = all(judgments["P5_prefix_within_20pct"].values())

    holds = {
        "H1_model_mismatch": sum(
            1 for r in rows if r.get("model_verification") != "assistant-event-exact"),
        "H2_refusals": sum(1 for r in rows
                           if str(r.get("failure_class")) == "refusal"),
        "H6_gross_closure_failures": sum(
            1 for r in rows if r["gross_input_tokens"] != (
                r["fresh_input_tokens_provider"]
                + r["cache_creation_input_tokens_provider"]
                + r["cache_read_input_tokens_provider"])),
        "H7_cost_missing_over_5pct": bool(
            p1["missing_share"] is not None and p1["missing_share"] > 0.05),
        "valid_turns": len(rows),
    }

    out = {
        "_meta": {"cell": "E24-BASE-RECAP", "model_calls": 0,
                  "turns_sha256": sha(turns_path),
                  "frozen_turns_sha256": sha(FROZEN_TURNS),
                  "estimators": "E13-RECOMP resample_matrix / paired_bootstrap"
                                " / paired_bootstrap_ratio_of_sums (seed "
                                f"{e13.BOOTSTRAP_SEED}, {e13.BOOTSTRAP_RESAMPLES})",
                  "rate_card_usd_per_million": CARD},
        "attempts_to_turns_join": join,
        "P1_cost_inversion": p1,
        "this_run": now,
        "frozen_cell_recomputed": frozen,
        "contrast": {
            "gross_ratio": {"this": now["gross_ratio"], "frozen": frozen["gross_ratio"]},
            "priced_ratio": {"this": now["priced_ratio"], "frozen": frozen["priced_ratio"]},
            "mu_w_star": {"this": now["reversal_threshold_mu_w_star"],
                          "frozen": frozen["reversal_threshold_mu_w_star"]},
            "mean_prefix_tokens_per_call": {
                arm: {"this": now["component_totals"][arm]["mean_prefix_tokens_per_call"],
                      "frozen": frozen["component_totals"][arm]["mean_prefix_tokens_per_call"]}
                for arm in ARMS},
            "component_delta": {k: {"this": d_now[k], "frozen": d_fr[k]}
                                for k in ("cache_read", "creation", "uncached", "output")},
        },
        "judgments": judgments,
        "hold_conditions": holds,
    }
    (RAW / "analysis.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"judgments": judgments, "hold_conditions": holds,
                      "contrast": out["contrast"]}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
