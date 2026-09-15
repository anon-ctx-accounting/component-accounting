#!/usr/bin/env python3
"""E26-CLAUDE-STEADY — 동결 canary 실행기를 그대로 쓰고 schedule 창만 바꾼다.

측정 코드를 복사하지 않는다.  `scripts/run_e10_xr_canary.py` 를 모듈로 가져와
schedule 상수·창·게이트 값과 출력 경로만 덮어쓰며, driver·argv·보류 조건은 동결
실행기의 것이다.  사전등록서 §2·§3 참조.

warm-up 턴은 돌리지 않는다.  상주 집합은 정책 replay 가 정하므로 정상 상태는 창
위치가 만들고, warm-up 을 돌리면 이 harness 의 세션 간 공유 prefix 가 provider 캐시에
미리 써져 다른 일곱 cell 과 회계 경계가 달라진다(E24 §4).
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

FROZEN = REPO / "scripts" / "run_e10_xr_canary.py"
spec = importlib.util.spec_from_file_location("e10_xr_canary", FROZEN)
canary = importlib.util.module_from_spec(spec)
sys.modules["e10_xr_canary"] = canary
spec.loader.exec_module(canary)

MEASURE_WINDOW = [f"S{i:02d}" for i in range(9, 21)]

# --- 창과 게이트 (사전등록서 §2·§3) ---
canary.SCHEDULE_SEED = 4352
canary.PAIRED_SESSIONS = 24                 # 빌드 규모.  측정은 아래에서 창으로 자른다
canary.VALID_TURN_TARGET = 192              # 12 세션 x 8 turn x 2 arm
canary.EXPECTED_SCHEDULE_SHA256 = (
    "f6edbb9e33c63c935aade45961ec8995f121e50e3854713963508925df1c2a73")
canary.CONFIG = CELL / "pre-execution-config.json"
canary.APPROVAL = CELL / "approvals" / "e26.json"

# --- 출력 경로 ---
canary.CANARY_ROOT = CELL / "run"
canary.RAW = CELL / "run" / "raw"
canary.RAW.mkdir(parents=True, exist_ok=True)

_frozen_attempt = canary._attempt


def _attempt(result, **kwargs) -> dict:
    """harness 가 보고한 금액을 함께 기록한다 (E24 에서 driver 에 추가한 필드)."""
    row = _frozen_attempt(result, **kwargs)
    row["schema"] = "e26-claude-steady-attempt-v1"
    row["harness_reported_cost_usd"] = result.harness_reported_cost_usd
    row["harness_reported_model_cost_usd"] = dict(
        result.harness_reported_model_cost_usd or {})
    return row


canary._attempt = _attempt

_frozen_verify = canary._verify_freeze


def _verify_freeze():
    """전체 24세션 bundle 로 해시를 검증한 뒤 측정 창 S09..S20 만 남긴다.

    해시 검증은 원본 bundle 로 수행하고, 자르는 것은 그 다음이다.  warm-up 세션
    S01..S08 은 실행 대상에서 제외되며 상주 집합은 replay 가 이미 그 위치까지
    반영해 두었다.
    """
    freeze, bundle, approval = _frozen_verify()
    bundle = dict(bundle)
    kept = [s for s in bundle["sessions"] if s["session_id"] in MEASURE_WINDOW]
    assert len(kept) == 12, f"측정 창을 찾지 못했다: {len(kept)}개"
    bundle["sessions"] = kept
    return freeze, bundle, approval


canary._verify_freeze = _verify_freeze


def main() -> int:
    """cell 신원은 별도 파일에 적는다.

    처음에는 산출물의 `phase == "canary"` 인 문서에 cell 이름을 덧붙였는데,
    `pre-canary-freeze.json` 이 그 조건에 걸린다.  그 파일은 자기 내용의 canonical
    해시를 `sha256` 필드로 들고 있고 실행 단계가 그것을 재계산해 대조하므로, 키를
    하나라도 더하면 검증이 깨진다.  실제로 깨졌고 가드가 실행을 막았다.  서명된
    산출물은 건드리지 않고 신원은 옆에 적는다.
    """
    (canary.RAW / "cell.json").write_text(
        json.dumps({
            "cell": "E26-CLAUDE-STEADY",
            "apparatus": "scripts/run_e10_xr_canary.py, imported unmodified",
            "overridden": {
                "SCHEDULE_SEED": canary.SCHEDULE_SEED,
                "build_sessions": canary.PAIRED_SESSIONS,
                "measure_window": MEASURE_WINDOW,
                "EXPECTED_SCHEDULE_SHA256": canary.EXPECTED_SCHEDULE_SHA256,
                "CONFIG": str(canary.CONFIG.relative_to(REPO)),
                "APPROVAL": str(canary.APPROVAL.relative_to(REPO)),
            },
            "warmup_executed": False,
            "why_no_warmup": ("steady state comes from the schedule position because the "
                              "resident set is fixed by model-independent policy replay "
                              "(E19 report 7.3); executing warm-up would pre-write this "
                              "harness's cross-session shared prefix into the provider "
                              "cache and break the accounting boundary the other cells "
                              "share (E24 section 4)"),
        }, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return canary.main()


if __name__ == "__main__":
    raise SystemExit(main())
