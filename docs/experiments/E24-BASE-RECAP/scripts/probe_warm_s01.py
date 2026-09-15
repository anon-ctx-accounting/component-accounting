#!/usr/bin/env python3
"""S01 만 다시 돌려 캐시 온도가 성분 배분을 가르는지 실증하는 probe.

E24 는 cell 의 첫 호출에서 공유 prefix 를 썼고 동결 cell 은 그것을 읽었다.  그
차이가 캐시 온도 때문이라면, E24 직후에 같은 unit 을 다시 돌리면 prefix 가 이미
써져 있으므로 동결 cell 의 배분이 재현되어야 한다.

이것은 probe 이며 cell 이 아니다.  판정 수치를 만들지 않고, 어떤 주장도 이 데이터
위에 세우지 않는다 — 두 run 의 차이가 캐시 온도로 설명되는지만 확인한다.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
CELL = HERE.parents[1]
REPO = HERE.parents[4]
assert (REPO / "src" / "karc").is_dir(), f"repo root 오판: {REPO}"

spec = importlib.util.spec_from_file_location(
    "e10_xr_canary", REPO / "scripts" / "run_e10_xr_canary.py")
canary = importlib.util.module_from_spec(spec)
sys.modules["e10_xr_canary"] = canary
spec.loader.exec_module(canary)

canary.CANARY_ROOT = CELL / "probe-warm-s01"
canary.RAW = CELL / "probe-warm-s01" / "raw"
canary.RAW.mkdir(parents=True, exist_ok=True)

_frozen_attempt = canary._attempt


def _attempt(result, **kwargs) -> dict:
    row = _frozen_attempt(result, **kwargs)
    row["schema"] = "e24-probe-warm-s01-attempt-v1"
    row["harness_reported_cost_usd"] = result.harness_reported_cost_usd
    return row


canary._attempt = _attempt

_frozen_verify = canary._verify_freeze


def _verify_freeze():
    """전체 schedule 해시를 검증한 뒤 S01 만 남긴다.

    해시 검증은 원본이 온전한 bundle 로 수행하고, 잘라내는 것은 그 다음이다.
    """
    freeze, bundle, approval = _frozen_verify()
    bundle = dict(bundle)
    bundle["sessions"] = [s for s in bundle["sessions"] if s["session_id"] == "S01"]
    assert len(bundle["sessions"]) == 1, "S01 을 찾지 못했다"
    return freeze, bundle, approval


canary._verify_freeze = _verify_freeze

if __name__ == "__main__":
    raise SystemExit(canary.main())
