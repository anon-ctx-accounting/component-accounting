#!/usr/bin/env python3
"""E25 판정 — E21 분석기를 그대로 쓰고 창과 대조 상대만 바꾼다.

E21 의 분석기는 측정 창 S09..S20 과 E19 leg B 대조를 상수로 들고 있다. 이 cell 은
창이 S01..S12 이고 자연스러운 대조 상대가 같은 창·같은 provider 의 E18 leg B 이므로
그 둘만 덮어쓴다. 추정량·부트스트랩·보류 조건은 E21 의 것이다. 모델 호출 0.
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
assert CELL.name == "E25-CODEX-BASEWINDOW", f"cell root 오판: {CELL}"

SRC = REPO / "docs/experiments/E21-CODEX-COMPONENT/scripts/analyze_e21.py"
spec = importlib.util.spec_from_file_location("analyze_e21", SRC)
a21 = importlib.util.module_from_spec(spec)
sys.modules["analyze_e21"] = a21
spec.loader.exec_module(a21)

a21.CELL_DIR = CELL
a21.RAW = CELL / "raw"
a21.RUN = REPO / "tmp" / "e25"          # unit 산출물 위치 (실행기와 같은 경로)
a21.WARMUP = []                                   # 이 cell 은 warm-up 이 없다
a21.SESSIONS = [f"S{i:02d}" for i in range(1, 13)]  # S01..S12

E18_ANALYSIS = REPO / "docs/experiments/E18-PI-REVERSAL/raw/analysis.json"


def e18_leg_b_contrast(stats: dict) -> dict:
    """같은 창·같은 provider 에서 harness 만 다른 대조.

    E18 leg B 는 pi 0.84.3 × OpenAI 를 seed 4200 의 S01..S12 에서 측정했고 이 cell 은
    같은 창을 Codex CLI 로 측정한다.  E21 이 E19 leg B 에 대해 하는 것과 같은 형태다.
    """
    if not E18_ANALYSIS.exists() or stats.get("incomplete"):
        return {"available": False}
    leg = (json.loads(E18_ANALYSIS.read_text(encoding="utf-8")).get("legs") or {}).get("B")
    if not leg:
        return {"available": False}
    out = {
        "available": True,
        "source": str(E18_ANALYSIS.relative_to(REPO)),
        "held_fixed": ["provider openai", "model gpt-5.6-luna",
                       "published luna rate card",
                       "schedule seed 4200 / identical snapshot_sha256",
                       "measurement window S01..S12 (no warm-up)",
                       "arms karc-full and rag-bm25",
                       "statistics: session bootstrap seed 1313"],
        "varied": ["harness: pi 0.84.3 -> Codex CLI 0.145.0"],
    }
    for key in ("gross_ratio_karc_over_rag", "priced_ratio_karc_over_rag",
                "component_delta_karc_minus_rag", "component_totals",
                "reversal_threshold_mu_w_star"):
        if key in leg:
            out.setdefault("e18_leg_b", {})[key] = leg[key]
    for key in ("gross_ratio_karc_over_rag", "priced_ratio_karc_over_rag",
                "component_delta_karc_minus_rag", "component_totals",
                "reversal_threshold_mu_w_star"):
        if key in stats:
            out.setdefault("this_cell", {})[key] = stats[key]
    return out


a21.e19_leg_b_contrast = e18_leg_b_contrast


def main() -> int:
    rc = a21.main()
    path = a21.RAW / "analysis.json"
    if path.exists():
        doc = json.loads(path.read_text(encoding="utf-8"))
        doc["cell"] = "E25-CODEX-BASEWINDOW"
        doc["preregistration"] = "docs/experiments/E25-CODEX-BASEWINDOW/preregistration.md"
        doc["e18_leg_b_contrast"] = doc.pop("e19_leg_b_contrast", {"available": False})
        win = doc.get("measurement_window")
        if isinstance(win, dict):
            win["note"] = ("one preregistered window, pooled once; no sub-range is "
                           "selected or tested. This is the growth-region window that "
                           "base and E18 measured, not a steady-state window; steady "
                           "state is not claimed for this cell.")
        path.write_text(json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    for p in sorted(a21.RAW.glob("*.json")):
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        meta = doc.get("_meta")
        if isinstance(meta, dict) and meta.get("cell") == "E21-CODEX-COMPONENT":
            meta["cell"] = "E25-CODEX-BASEWINDOW"
            p.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n",
                         encoding="utf-8")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
