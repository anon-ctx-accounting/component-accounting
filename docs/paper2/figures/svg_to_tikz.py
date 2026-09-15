#!/usr/bin/env python3
"""커밋된 그림 SVG를 TikZ로 옮겨 조판본에 PDF로 들어가게 한다.

왜 변환기를 쓰지 않는가.  이 저장소는 stdlib 전용이고(`pyproject.toml`의
"dev 의존성은 pytest만 허용") cairosvg·svglib 계열은 새 의존성이거나 시스템
라이브러리를 요구한다.  TikZ로 내보내면 LaTeX 자신이 그림을 조판하므로 변환기가
필요 없고, 본문과 같은 폰트로 글자가 놓이며, `standalone` 문서로 컴파일하면 그림
하나당 PDF 파일이 나온다.

레이아웃의 단일 출처는 여전히 `make_figures.py`가 만드는 SVG다.  이 스크립트는 그
SVG를 읽어 좌표계만 뒤집어 옮기며, 자체 수치를 만들지 않는다.

사용법
    python docs/paper2/figures/svg_to_tikz.py                # 논문에 실린 두 그림
    python docs/paper2/figures/svg_to_tikz.py --width-pt 505 # 두 단 폭
    pdflatex -output-directory=docs/paper2/figures docs/paper2/figures/accounting-forest.tex
"""
from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
NS = "{http://www.w3.org/2000/svg}"

# `svg_document`의 <style> 블록과 같은 값이어야 한다.  클래스를 쓰는 그림과
# font-size 를 직접 적는 그림이 섞여 있으므로 양쪽을 모두 해석한다.
BASE_FILL = "#17212b"
CLASS_STYLE = {
    "title": {"font-size": 18.0, "bold": True},
    "panel": {"font-size": 15.0, "bold": True},
    "axis": {"font-size": 11.0},
    "note": {"font-size": 10.5, "fill": "#59636e"},
    "legend": {"font-size": 11.0},
}
# 논문에 실린 그림만 변환한다.  나머지 SVG는 본문에서 인용되지 않는다.
PAPER_FIGURES = ("accounting-forest.svg", "baseline-first.svg")

LATEX_ESCAPE = {
    "\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "$": r"\$", "&": r"\&",
    "#": r"\#", "^": r"\textasciicircum{}", "_": r"\_", "%": r"\%",
    "~": r"\textasciitilde{}",
}
UNICODE_MAP = {"\u2014": "---", "\u2013": "--", "\u2212": "$-$", "\u00d7": r"$\times$",
               "\u2264": r"$\leq$", "\u2265": r"$\geq$", "\u00b5": r"$\mu$"}


def tex_text(value: str) -> str:
    out = []
    for ch in value:
        if ch in LATEX_ESCAPE:
            out.append(LATEX_ESCAPE[ch])
        elif ch in UNICODE_MAP:
            out.append(UNICODE_MAP[ch])
        elif ord(ch) > 127:
            raise SystemExit(f"변환하지 못한 비ASCII 문자 {ch!r} — UNICODE_MAP에 추가하라")
        else:
            out.append(ch)
    return "".join(out)


class Colors:
    """hex 색을 \\definecolor 이름으로 모은다."""

    def __init__(self, prefix: str) -> None:
        self.seen: dict[str, str] = {}
        self.prefix = prefix

    def name(self, value: str | None, default: str = BASE_FILL) -> str:
        raw = (value or default).strip()
        if not re.fullmatch(r"#[0-9a-fA-F]{6}", raw):
            raise SystemExit(f"지원하지 않는 색 표기 {raw!r}")
        key = raw[1:].upper()
        if key not in self.seen:
            self.seen[key] = f"{self.prefix}c{len(self.seen)}"
        return self.seen[key]

    def definitions(self) -> list[str]:
        return [f"\\definecolor{{{n}}}{{HTML}}{{{k}}}" for k, n in self.seen.items()]


ANCHOR = {"start": "base west", "middle": "base", "end": "base east", None: "base west"}


def convert(path: Path, width_pt: float,
            font_package: str) -> tuple[str, str]:
    root = ET.parse(path).getroot()
    vb = [float(v) for v in (root.get("viewBox") or "").split()]
    if len(vb) != 4:
        raise SystemExit(f"{path.name}: viewBox 를 읽지 못했다")
    _, _, w, h = vb
    f = width_pt / w
    colors = Colors(re.sub(r"[^a-zA-Z]", "", path.stem)[:12] or "fig")
    body: list[str] = []

    # 내용의 실제 y 범위.  캔버스 하단 여백을 그대로 옮기면 조판에서 빈 줄로 남는다.
    content_y: list[float] = []
    for el in root.iter():
        tag = el.tag.replace(NS, "")
        if tag == "text":
            content_y.append(float(el.get("y")))
        elif tag == "line":
            content_y += [float(el.get("y1")), float(el.get("y2"))]
        elif tag == "circle":
            r = float(el.get("r"))
            content_y += [float(el.get("cy")) - r, float(el.get("cy")) + r]
    # 글자 하강부와 선 두께를 위해 소폭만 남긴다.
    bottom = min(h, max(content_y) + 6.0) if content_y else h

    def X(v: float) -> str:
        return f"{v * f:.3f}pt"

    def Y(v: float) -> str:
        return f"{(h - v) * f:.3f}pt"

    def stroke_opts(el: ET.Element) -> str:
        opts = [colors.name(el.get("stroke"))]
        sw = float(el.get("stroke-width", 1.0)) * f
        opts.append(f"line width={sw:.3f}pt")
        dash = el.get("stroke-dasharray")
        if dash:
            parts = [float(x) * f for x in re.split(r"[ ,]+", dash.strip()) if x]
            if len(parts) == 1:
                parts *= 2
            opts.append("dash pattern="
                        + " ".join(f"on {a:.2f}pt off {b:.2f}pt"
                                   for a, b in zip(parts[::2], parts[1::2])))
        return ", ".join(opts)

    for el in root.iter():
        tag = el.tag.replace(NS, "")
        if tag in ("svg", "title", "desc", "style", "g"):
            continue
        if tag == "rect":
            def length(value: str | float, span: float) -> float:
                """`100%` 같은 백분율은 캔버스 크기에 상대적이다."""
                text = str(value).strip()
                return span * float(text[:-1]) / 100.0 if text.endswith("%") else float(text)

            x = length(el.get("x", 0.0), w)
            y = length(el.get("y", 0.0), h)
            rw = length(el.get("width"), w)
            rh = length(el.get("height"), h)
            body.append(f"\\fill[{colors.name(el.get('fill'))}] "
                        f"({X(x)},{Y(y + rh)}) rectangle ({X(x + rw)},{Y(y)});")
        elif tag == "line":
            body.append(f"\\draw[{stroke_opts(el)}] "
                        f"({X(float(el.get('x1')))},{Y(float(el.get('y1')))}) -- "
                        f"({X(float(el.get('x2')))},{Y(float(el.get('y2')))});")
        elif tag == "polyline":
            pts = [float(v) for v in re.split(r"[ ,]+", el.get("points").strip()) if v]
            chain = " -- ".join(f"({X(a)},{Y(b)})"
                                for a, b in zip(pts[::2], pts[1::2]))
            body.append(f"\\draw[{stroke_opts(el)}] {chain};")
        elif tag == "circle":
            cx, cy, r = (float(el.get("cx")), float(el.get("cy")), float(el.get("r")))
            cmd = "\\fill" if el.get("stroke") is None else "\\filldraw"
            opts = [colors.name(el.get("fill"))]
            if el.get("stroke") is not None:
                opts.append(f"draw={colors.name(el.get('stroke'))}")
                opts.append(f"line width={float(el.get('stroke-width', 1.0)) * f:.3f}pt")
            body.append(f"{cmd}[{', '.join(opts)}] ({X(cx)},{Y(cy)}) "
                        f"circle[radius={r * f:.3f}pt];")
        elif tag == "text":
            style = dict(CLASS_STYLE.get(el.get("class", ""), {}))
            size = float(el.get("font-size", style.get("font-size", 11.0)))
            fill = el.get("fill") or style.get("fill") or BASE_FILL
            bold = style.get("bold") or el.get("font-weight") == "700"
            italic = el.get("font-style") == "italic"
            content = tex_text((el.text or "").strip())
            if bold:
                content = f"\\bfseries {content}"
            if italic:
                content = f"\\itshape {content}"
            pt = size * f
            opts = [f"anchor={ANCHOR[el.get('text-anchor')]}", colors.name(fill),
                    f"font=\\fontsize{{{pt:.2f}pt}}{{{pt * 1.2:.2f}pt}}\\selectfont",
                    "inner sep=0pt", "outer sep=0pt"]
            tr = el.get("transform")
            if tr:
                m = re.fullmatch(r"rotate\(\s*(-?[\d.]+)\s+([\d.]+)\s+([\d.]+)\s*\)", tr)
                if not m:
                    raise SystemExit(f"지원하지 않는 transform {tr!r}")
                ang, cx, cy = (float(m.group(1)), float(m.group(2)), float(m.group(3)))
                # `rotate around` 는 노드 옵션에서 국소 좌표계에 적용되어 노드가 (cx,cy)만큼
                # 튕겨 나간다.  회전 중심이 텍스트 앵커 자체일 때만 지원하고, SVG(y 아래) →
                # TikZ(y 위) 변환이므로 각의 부호를 뒤집어 노드 자체를 돌린다.
                tx, ty = float(el.get("x")), float(el.get("y"))
                if abs(cx - tx) > 0.05 or abs(cy - ty) > 0.05:
                    raise SystemExit(f"rotate 중심이 텍스트 앵커와 다르다: {tr!r} vs ({tx},{ty})")
                opts.append(f"rotate={-ang:g}")
            body.append(f"\\node[{', '.join(opts)}] at "
                        f"({X(float(el.get('x')))},{Y(float(el.get('y')))}) {{{content}}};")
        else:
            raise SystemExit(f"{path.name}: 처리하지 않은 요소 <{tag}>")

    head = [
        "% 자동 생성물 — 손으로 고치지 말 것.",
        f"% 출처: {path.name} (docs/paper2/figures/make_figures.py 생성)",
        f"% 생성: python docs/paper2/figures/svg_to_tikz.py --width-pt {width_pt:g}",
    ]
    body_tex = "\n".join([
        *head,
        "% 본문 삽입용. 논문 문서에서 \\input 하면 글자가 본문 폰트로 조판된다.",
        *colors.definitions(),
        "\\begin{tikzpicture}[baseline]",
        f"\\useasboundingbox ({X(0)},{Y(bottom)}) rectangle ({X(w)},{Y(0)});",
        *body,
        "\\end{tikzpicture}",
        "",
    ])
    font = f"\\usepackage{{{font_package}}}" if font_package else "% 폰트 패키지 없음"
    standalone_tex = "\n".join([
        *head,
        "% 단독 PDF: cd docs/paper2/figures && pdflatex "
        f"{path.stem}.tex   (\\input 이 같은 디렉터리를 보므로 그 안에서 실행한다)",
        "\\documentclass[tikz,border=0pt]{standalone}",
        "\\usepackage{tikz}",
        "\\usepackage[T1]{fontenc}",
        font,
        "\\begin{document}",
        f"\\input{{{path.stem}-body.tex}}",
        "\\end{document}",
        "",
    ])
    return standalone_tex, body_tex


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width-pt", type=float, default=240.0,
                    help="목표 폭(pt). 기본 240 = USENIX 한 단 근사, 두 단은 505.")
    ap.add_argument("--font-package", default="newtxtext",
                    help="단독 PDF 의 본문 폰트 패키지. USENIX 는 Times 계열이다. "
                         "설치되어 있지 않으면 times 로 바꾸거나 빈 문자열을 준다.")
    ap.add_argument("svg", nargs="*", help="비우면 논문에 실린 두 그림")
    args = ap.parse_args()
    targets = [Path(s) for s in args.svg] or [HERE / n for n in PAPER_FIGURES]
    for svg in targets:
        standalone, body = convert(svg, args.width_pt, args.font_package)
        svg.with_suffix(".tex").write_text(standalone, encoding="utf-8")
        (svg.parent / f"{svg.stem}-body.tex").write_text(body, encoding="utf-8")
        print(f"  {svg.name} -> {svg.stem}.tex + {svg.stem}-body.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
