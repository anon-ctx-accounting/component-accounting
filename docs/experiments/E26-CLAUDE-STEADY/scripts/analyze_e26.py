#!/usr/bin/env python3
"""E26 판정 — E24 의 분석기를 이 cell 의 창에 적용한다.

추정량을 새로 쓰지 않는다.  E24 의 profile/attach_cost/p1_cost_inversion 을 그대로
가져오고, E13 의 부트스트랩(시드 1313 · 10,000회 · 공통 난수)을 쓴다.  다른 것은 세션
이름이 S09..S20 이라는 점뿐이므로 E13 의 SESSIONS 를 이 창으로 바꿔 끼운다 — 부트스트랩
행렬은 인덱스만 쓰므로 이름이 무엇이든 같은 재표집이다.  모델 호출 0.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
CELL = HERE.parents[1]
REPO = HERE.parents[4]
assert (REPO / "src" / "karc").is_dir(), f"repo root 오판: {REPO}"
assert CELL.name == "E26-CLAUDE-STEADY", f"cell root 오판: {CELL}"

spec = importlib.util.spec_from_file_location(
    "analyze_e24", REPO / "docs/experiments/E24-BASE-RECAP/scripts/analyze_e24.py")
a24 = importlib.util.module_from_spec(spec)
sys.modules["analyze_e24"] = a24
spec.loader.exec_module(a24)

RAW = CELL / "run" / "raw"
BASE_TURNS = REPO / "docs/experiments/E24-BASE-RECAP/run/raw/turns.jsonl"


def main() -> int:
    rows = a24.rows_of(RAW / "turns.jsonl")
    join = a24.attach_cost(rows, a24.rows_of(RAW / "attempts.jsonl"))
    if join["component_mismatched"]:
        raise SystemExit(f"join 교차검증 실패: {join}")
    sessions = sorted({r["session_id"] for r in rows})
    assert len(sessions) == 12 and sessions[0] == "S09" and sessions[-1] == "S20", sessions
    a24.e13.SESSIONS = sessions          # 창 이름만 바꿔 끼운다
    matrix = a24.e13.resample_matrix()
    now = a24.profile(rows, sessions, matrix)
    p1 = a24.p1_cost_inversion(rows)

    # 같은 harness·provider 의 성장 구간 cell (E24) 과의 창 대조
    base_rows = a24.rows_of(BASE_TURNS)
    base_sessions = sorted({r["session_id"] for r in base_rows})
    a24.e13.SESSIONS = base_sessions
    base = a24.profile(base_rows, base_sessions, a24.e13.resample_matrix())

    d = now["component_delta_karc_minus_rag"]
    out = {
        "cell": "E26-CLAUDE-STEADY",
        "preregistration": "docs/experiments/E26-CLAUDE-STEADY/preregistration.md",
        "harness": {"name": "claude-code", "version": "2.1.220",
                    "route": "Anthropic subscription (OAuth)",
                    "provider": "anthropic", "model": "claude-sonnet-5"},
        "measurement_window": {
            "sessions": sessions, "warmup_executed": False,
            "note": "steady-state window of the seed-4352 24-session build, the window "
                    "E19 both legs and E21 measured. Steady state comes from the schedule "
                    "position (resident set fixed by model-independent policy replay), not "
                    "from executing warm-up turns.",
        },
        "_meta": {"model_calls": 0,
                  "turns_sha256": a24.sha(RAW / "turns.jsonl"),
                  "estimators": "E24 profile with E13 bootstrap "
                                f"(seed {a24.e13.BOOTSTRAP_SEED}, "
                                f"{a24.e13.BOOTSTRAP_RESAMPLES})",
                  "rate_card_usd_per_million": a24.CARD},
        "attempts_to_turns_join": join,
        "P2_cost_inversion": p1,
        "this_run": now,
        "growth_window_contrast_e24": {
            "held_fixed": ["harness claude-code 2.1.220", "provider anthropic",
                           "model claude-sonnet-5", "published Anthropic rate card",
                           "arms karc-full and rag-bm25",
                           "statistics: session bootstrap seed 1313"],
            "varied": ["schedule window: seed4200 S01-S12 (growth) -> "
                       "seed4352 S09-S20 (steady state)"],
            "e24": {k: base[k] for k in ("gross_ratio", "priced_ratio",
                                         "gross_ratio_boot", "priced_ratio_boot",
                                         "reversal_threshold_mu_w_star")},
            "this_cell": {k: now[k] for k in ("gross_ratio", "priced_ratio",
                                              "gross_ratio_boot", "priced_ratio_boot",
                                              "reversal_threshold_mu_w_star")},
        },
        "judgments": {
            "P1_component_sign_preserved": bool(d["cache_read"] > 0 and d["creation"] < 0),
            "P2_harness_cost_inverts": bool(p1["exact_within_1e_9_usd"]),
            "P3_priced_ci_excludes_parity": bool(now["priced_ratio_boot"]["ci95_hi"] < 1.0),
            "P4_gross_ci_excludes_parity": bool(now["gross_ratio_boot"]["ci95_lo"] > 1.0),
            "threshold_predicts_direction": {
                "mu_w_star": now["reversal_threshold_mu_w_star"],
                "actual_mu_w": now["observed_mu_w"],
                "predicts_managed_arm_cheaper":
                    now["reversal_threshold_mu_w_star"] < now["observed_mu_w"],
                "observed_managed_arm_cheaper": now["priced_ratio"] < 1.0,
                "agree": ((now["reversal_threshold_mu_w_star"] < now["observed_mu_w"])
                          == (now["priced_ratio"] < 1.0)),
            },
        },
        "hold_conditions": {
            "H1_model_mismatch": sum(
                1 for r in rows if r.get("model_verification") != "assistant-event-exact"),
            "H2_refusals": sum(1 for r in rows if str(r.get("failure_class")) == "refusal"),
            "H6_gross_closure_failures": sum(
                1 for r in rows if r["gross_input_tokens"] != (
                    r["fresh_input_tokens_provider"]
                    + r["cache_creation_input_tokens_provider"]
                    + r["cache_read_input_tokens_provider"])),
            "H7_first_call_read_without_write": sum(
                1 for r in rows if r["cache_creation_input_tokens_provider"] == 0
                and r["cache_read_input_tokens_provider"] > 0),
            "valid_turns": len(rows),
            "accuracy": {arm: f"{sum(1 for r in rows if r['arm'] == arm and r.get('passed'))}"
                              f"/{sum(1 for r in rows if r['arm'] == arm)}"
                         for arm in a24.ARMS},
        },
    }
    (RAW / "analysis.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"judgments": out["judgments"],
                      "hold_conditions": out["hold_conditions"]},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
