#!/usr/bin/env python3
"""E25-CODEX-BASEWINDOW — E21 실행기를 그대로 쓰고 schedule 창만 바꾼다.

측정 코드를 복사하지 않는 이유는 복사 과정에서 측정이 달라질 수 있기 때문이다.
E21 모듈을 가져와 schedule 상수 셋과 출력 경로만 덮어쓰며, 그 외 argv·판정·보류
조건은 E21 의 것이다.  사전등록서 §2·§3 참조.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
CELL = HERE.parents[1]
REPO = HERE.parents[4]
assert (REPO / "src" / "karc").is_dir(), f"repo root 오판: {REPO}"
assert CELL.name == "E25-CODEX-BASEWINDOW", f"cell root 오판: {CELL}"

E21 = REPO / "docs/experiments/E21-CODEX-COMPONENT/scripts/run_e21_codex_component.py"
spec = importlib.util.spec_from_file_location("run_e21", E21)
e21 = importlib.util.module_from_spec(spec)
sys.modules["run_e21"] = e21
spec.loader.exec_module(e21)

# --- 창만 바꾼다 (사전등록서 §2) ---
e21.SCHEDULE_SEED = 4200
e21.SESSION_COUNT = 12
e21.WARMUP_SESSIONS_N = 0
e21.MEASURE_SESSIONS_N = 12
# 이 빌드의 신원 해시.  등록 시점에 모델 호출 0 으로 계산해 사전등록서에 적었다.
e21.SCHEDULE_SNAPSHOT_SHA256 = (
    "a19dea499215f03560e18d2bd53249cb0c63fcec0fb8c0df0449b48c50840301")
e21.SCHEDULE_SHA256 = (
    "9944a77572e2d130a0150f865024785929771cd13a9264ab315b4bfa5f26eea3")
e21.TASKS_SHA256 = (
    "618193d35103b819d83ce2d13dea40d7e3a65cbae5072266ab693613fe3b3584")

# --- 출력 경로 ---
e21.CELL_DIR = CELL
e21.RAW = CELL / "raw"
e21.RUN = REPO / "tmp" / "e25"
e21.RAW.mkdir(parents=True, exist_ok=True)

# --- warm-up 이 없으므로 정상 상태 단계는 돌지 않는다 ---
def phase_steady() -> dict:
    """이 cell 은 정상 상태를 주장하지 않는다.

    E21 의 단계는 마지막 warm-up 세션과 그 앞 세션의 상주 주입을 비교하는데
    warm-up 이 0 이면 비교할 행이 없다.  이 창의 상주 집합은 S01 에서 자라기
    시작하므로 base·E18 과 같은 성장 구간이며, 그 사실을 기록만 한다.
    """
    bundle = e21.build_bundle()["bundle"] if hasattr(e21, "build_bundle") else None
    return {
        "claimed": False,
        "why": "warm-up sessions = 0; this cell measures the growth-region window "
               "S01-S12 that base and E18 measured, not a steady-state window.",
        "warmup_sessions_run": 0,
        "measure_sessions": [e21.session_id(i) for i in e21.measure_sessions()],
        "bundle_present": bundle is not None,
    }


e21.phase_steady = phase_steady

def main() -> int:
    """E21 의 main 을 그대로 쓰되 산출물의 cell 이름만 이 cell 로 고친다."""
    rc = e21.main()
    for path in sorted(e21.RAW.glob("*.json")):
        import json
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        meta = doc.get("_meta")
        if isinstance(meta, dict) and meta.get("cell") == "E21-CODEX-COMPONENT":
            meta["cell"] = "E25-CODEX-BASEWINDOW"
            meta["derived_from"] = "E21 runner with schedule window overridden"
            path.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n",
                            encoding="utf-8")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
