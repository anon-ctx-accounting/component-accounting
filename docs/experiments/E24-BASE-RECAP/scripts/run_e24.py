#!/usr/bin/env python3
"""E24-BASE-RECAP — 기준 성분 cell 재측정.

동결 실행기 ``scripts/run_e10_xr_canary.py``를 **모듈로 가져와 그대로 실행**하고
출력 경로만 이 cell로 돌린다.  측정 코드를 복사하지 않는 이유는, 복사하면 경로
편집 과정에서 측정 자체가 달라질 수 있기 때문이다.  이 wrapper가 바꾸는 것은
출력 위치와 attempt 행에 붙는 금액 필드 둘뿐이다.

사전등록서 §3 참조.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
CELL = HERE.parents[1]
REPO = HERE.parents[4]
assert (REPO / "src" / "karc").is_dir(), f"repo root 오판: {REPO}"
assert CELL.name == "E24-BASE-RECAP", f"cell root 오판: {CELL}"

FROZEN = REPO / "scripts" / "run_e10_xr_canary.py"
spec = importlib.util.spec_from_file_location("e10_xr_canary", FROZEN)
canary = importlib.util.module_from_spec(spec)
sys.modules["e10_xr_canary"] = canary
spec.loader.exec_module(canary)

# 출력만 이 cell로 돌린다.  CONFIG·APPROVAL·SMOKE_* 는 동결 cell의 것을 그대로
# 가리켜, fixture·schedule·승인 해시 검증이 동결 실행기 자신의 것으로 걸린다.
canary.CANARY_ROOT = CELL / "run"
canary.RAW = CELL / "run" / "raw"
canary.RAW.mkdir(parents=True, exist_ok=True)

_frozen_attempt = canary._attempt


def _attempt(result, **kwargs) -> dict:
    """동결 실행기의 attempt 행에 harness 보고 금액을 덧붙인다.

    ``actual_subscription_billed_usd`` 는 동결 실행기가 None 으로 하드코딩한
    자리다.  구독 경로에서 그 이름이 뜻하는 값(증분 청구액)은 여전히 관측되지
    않으므로 이름을 바꾸지 않고 그대로 두고, harness 가 보고하는 요금표 환산값을
    별도 필드로 기록한다.
    """
    row = _frozen_attempt(result, **kwargs)
    row["schema"] = "e24-base-recap-attempt-v1"
    row["harness_reported_cost_usd"] = result.harness_reported_cost_usd
    row["harness_reported_model_cost_usd"] = dict(
        result.harness_reported_model_cost_usd or {})
    return row


canary._attempt = _attempt

if __name__ == "__main__":
    raise SystemExit(canary.main())
