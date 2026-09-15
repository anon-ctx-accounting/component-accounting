#!/usr/bin/env python3
"""Render Paper 2 SVG figures from committed E8-D5 and E10 summaries.

This script performs presentation-only transformations.  It does not invoke a
model, rerun a benchmark, alter experiment data, or fit a new curve.
"""

from __future__ import annotations

import csv
import html
import json
import math
from pathlib import Path
from typing import Iterable


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
D5 = REPO / "docs" / "experiments" / "E8-D5" / "raw"
H16_SUMMARY = (
    REPO / "docs" / "experiments" / "E10-P2-H16" / "run" / "raw" / "summary.json"
)
BASE3_SUMMARY = (
    REPO / "docs" / "experiments" / "E11-BASE3" / "run" / "raw" / "summary.json"
)
E18_ANALYSIS = (
    REPO / "docs" / "experiments" / "E18-PI-REVERSAL" / "raw" / "analysis.json"
)
E19_ANALYSIS = (
    REPO / "docs" / "experiments" / "E19-STEADY-STATE" / "raw" / "analysis.json"
)
E21_ANALYSIS = (
    REPO / "docs" / "experiments" / "E21-CODEX-COMPONENT" / "raw" / "analysis.json"
)
E23_ANALYSIS = (
    REPO / "docs" / "experiments" / "E23-REUSE-DENSITY" / "raw" / "analysis.json"
)
E24_ANALYSIS = (
    REPO / "docs" / "experiments" / "E24-BASE-RECAP" / "run" / "raw" / "analysis.json"
)
E25_ANALYSIS = (
    REPO / "docs" / "experiments" / "E25-CODEX-BASEWINDOW" / "raw" / "analysis.json"
)
E26_ANALYSIS = (
    REPO / "docs" / "experiments" / "E26-CLAUDE-STEADY" / "run" / "raw" / "analysis.json"
)
E28_ANALYSIS = (
    REPO / "docs" / "experiments" / "E28-BUDGET-SWEEP" / "raw" / "analysis.json"
)
BASE3_CORRECTED = (
    REPO
    / "docs"
    / "experiments"
    / "E22-CODEX-CUMULATIVE"
    / "raw"
    / "recompute.json"
)
XR_SUMMARY = (
    REPO
    / "docs"
    / "experiments"
    / "E10-P2-XR"
    / "canary"
    / "raw"
    / "summary.json"
)

INK = "#17212b"
GRID = "#dce2e8"
BLUE = "#2474b5"
ORANGE = "#d46b28"
GREEN = "#2a8a68"
PURPLE = "#7656a3"
GREY = "#69737d"


def read_csv(name: str) -> list[dict[str, str]]:
    with (D5 / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def line(x1: float, y1: float, x2: float, y2: float, **attrs: object) -> str:
    parts = {
        "x1": f"{x1:.2f}",
        "y1": f"{y1:.2f}",
        "x2": f"{x2:.2f}",
        "y2": f"{y2:.2f}",
        **attrs,
    }
    return "<line " + " ".join(f'{k}="{esc(v)}"' for k, v in parts.items()) + "/>"


def text(x: float, y: float, value: object, **attrs: object) -> str:
    parts = {"x": f"{x:.2f}", "y": f"{y:.2f}", **attrs}
    return (
        "<text "
        + " ".join(f'{k}="{esc(v)}"' for k, v in parts.items())
        + f">{esc(value)}</text>"
    )


def polyline(points: Iterable[tuple[float, float]], **attrs: object) -> str:
    encoded = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    parts = {"points": encoded, "fill": "none", **attrs}
    return (
        "<polyline "
        + " ".join(f'{k}="{esc(v)}"' for k, v in parts.items())
        + "/>"
    )


def circle(x: float, y: float, radius: float = 3.2, **attrs: object) -> str:
    parts = {"cx": f"{x:.2f}", "cy": f"{y:.2f}", "r": radius, **attrs}
    return (
        "<circle "
        + " ".join(f'{k}="{esc(v)}"' for k, v in parts.items())
        + "/>"
    )


def svg_document(
    width: int,
    height: int,
    body: list[str],
    title_value: str,
    source_desc: str = "Generated only from committed docs/experiments/E8-D5/raw CSV data.",
) -> str:
    return "\n".join(
        [
            (
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
                f'height="{height}" viewBox="0 0 {width} {height}" '
                'role="img" aria-labelledby="title desc">'
            ),
            f"<title id=\"title\">{esc(title_value)}</title>",
            f'<desc id="desc">{esc(source_desc)}</desc>',
            "<style>",
            "text { font-family: Inter, Arial, sans-serif; fill: #17212b; }",
            ".title { font-size: 18px; font-weight: 700; }",
            ".panel { font-size: 15px; font-weight: 700; }",
            ".axis { font-size: 11px; }",
            ".note { font-size: 10.5px; fill: #59636e; }",
            ".legend { font-size: 11px; }",
            "</style>",
            *body,
            "</svg>",
            "",
        ]
    )


def render_turn_curves() -> None:
    rows = read_csv("turn-token-curve.csv")
    width, height = 900, 430
    body = [
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        text(
            32,
            28,
            "Turn-position input: initial level and subsequent growth",
            **{"class": "title"},
        ),
    ]

    styles = {
        ("E5-G1", "karc-full"): (BLUE, "", "K-ARC whole-doc"),
        ("E5-G1", "rag-bm25"): (ORANGE, "", "RAG (G1)"),
        ("E5-G2", "karc-chunk"): (GREEN, "7 4", "K-ARC chunk"),
        ("E5-G2", "rag-bm25"): (PURPLE, "7 4", "RAG (G2)"),
    }

    panels = [
        ("fresh_input_mean", "A  Fresh input", 55, 78, 370, 260, 40_000),
        ("gross_input_mean", "B  Gross logical input", 490, 78, 370, 260, 300_000),
    ]

    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row["experiment"], row["display_arm"]), []).append(row)

    for field, label, left, top, plot_w, plot_h, y_max in panels:
        body.append(text(left, top - 14, label, **{"class": "panel"}))
        for tick in range(6):
            value = y_max * tick / 5
            y = top + plot_h - plot_h * tick / 5
            body.append(line(left, y, left + plot_w, y, stroke=GRID, **{"stroke-width": 1}))
            body.append(
                text(
                    left - 8,
                    y + 4,
                    f"{value / 1000:.0f}k",
                    **{"class": "axis", "text-anchor": "end"},
                )
            )
        body.append(line(left, top, left, top + plot_h, stroke=INK, **{"stroke-width": 1.2}))
        body.append(
            line(
                left,
                top + plot_h,
                left + plot_w,
                top + plot_h,
                stroke=INK,
                **{"stroke-width": 1.2},
            )
        )
        for position in range(1, 9):
            x = left + (position - 1) * plot_w / 7
            body.append(
                text(
                    x,
                    top + plot_h + 18,
                    position,
                    **{"class": "axis", "text-anchor": "middle"},
                )
            )
        body.append(
            text(
                left + plot_w / 2,
                top + plot_h + 37,
                "Turn position",
                **{"class": "axis", "text-anchor": "middle"},
            )
        )
        body.append(
            text(
                left - 42,
                top + plot_h / 2,
                "tokens / turn",
                **{
                    "class": "axis",
                    "text-anchor": "middle",
                    "transform": f"rotate(-90 {left - 42:.2f} {top + plot_h / 2:.2f})",
                },
            )
        )

        for key, series in grouped.items():
            color, dash, _ = styles[key]
            series.sort(key=lambda item: int(item["turn_position"]))
            points = []
            for row in series:
                x = left + (int(row["turn_position"]) - 1) * plot_w / 7
                y = top + plot_h - float(row[field]) * plot_h / y_max
                points.append((x, y))
            attrs: dict[str, object] = {"stroke": color, "stroke-width": 2.4}
            if dash:
                attrs["stroke-dasharray"] = dash
            body.append(polyline(points, **attrs))
            for x, y in points:
                body.append(circle(x, y, fill=color, stroke="#ffffff", **{"stroke-width": 0.8}))

    legend_y = 395
    legend_x = 95
    for index, key in enumerate(styles):
        color, dash, label = styles[key]
        x = legend_x + index * 185
        attrs = {"stroke": color, "stroke-width": 2.4}
        if dash:
            attrs["stroke-dasharray"] = dash
        body.append(line(x, legend_y - 4, x + 24, legend_y - 4, **attrs))
        body.append(text(x + 30, legend_y, label, **{"class": "legend"}))

    body.append(
        text(
            450,
            422,
            "Means over 12 sessions at each position; lines connect observations, not extrapolations.",
            **{"class": "note", "text-anchor": "middle"},
        )
    )
    (HERE / "turn-input-curves.svg").write_text(
        svg_document(width, height, body, "Turn-position fresh and gross input curves"),
        encoding="utf-8",
    )


def render_carry_break_even() -> None:
    rows = read_csv("carry-break-even.csv")
    width, height = 760, 430
    left, top, plot_w, plot_h = 80, 65, 620, 280
    y_min, y_max = -1_500.0, 1_800.0
    body = [
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        text(
            28,
            28,
            "Carry-price sensitivity: K-ARC minus RAG priced input",
            **{"class": "title"},
        ),
    ]

    variants = {
        "primary_asymmetric": (GREY, "3 3", "asymmetric primary"),
        "symmetric_w0": (BLUE, "", "symmetric w=0"),
        "symmetric_w0.1": (GREEN, "", "symmetric w=0.1"),
        "symmetric_w1": (ORANGE, "", "symmetric w=1"),
    }

    def x_map(value: float) -> float:
        return left + value * plot_w

    def y_map(value: float) -> float:
        return top + (y_max - value) * plot_h / (y_max - y_min)

    for value in (-1_500, -1_000, -500, 0, 500, 1_000, 1_500):
        y = y_map(value)
        body.append(
            line(
                left,
                y,
                left + plot_w,
                y,
                stroke=INK if value == 0 else GRID,
                **{"stroke-width": 1.4 if value == 0 else 1},
            )
        )
        body.append(
            text(left - 9, y + 4, f"{value:,}", **{"class": "axis", "text-anchor": "end"})
        )
    body.append(line(left, top, left, top + plot_h, stroke=INK, **{"stroke-width": 1.2}))
    body.append(
        line(
            left,
            top + plot_h,
            left + plot_w,
            top + plot_h,
            stroke=INK,
            **{"stroke-width": 1.2},
        )
    )
    for value in (0, 0.25, 0.5, 0.75, 1):
        x = x_map(value)
        body.append(
            text(
                x,
                top + plot_h + 19,
                f"{value:g}",
                **{"class": "axis", "text-anchor": "middle"},
            )
        )
    body.append(
        text(
            left + plot_w / 2,
            top + plot_h + 39,
            "Reuse r",
            **{"class": "axis", "text-anchor": "middle"},
        )
    )
    body.append(
        text(
            20,
            top + plot_h / 2,
            "tokens / task",
            **{
                "class": "axis",
                "text-anchor": "middle",
                "transform": f"rotate(-90 20 {top + plot_h / 2:.2f})",
            },
        )
    )

    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["variant"], []).append(row)
    for variant, series in grouped.items():
        color, dash, _ = variants[variant]
        series.sort(key=lambda item: float(item["reuse_r"]))
        points = [
            (x_map(float(row["reuse_r"])), y_map(float(row["karc_minus_rag"])))
            for row in series
        ]
        attrs: dict[str, object] = {"stroke": color, "stroke-width": 2.5}
        if dash:
            attrs["stroke-dasharray"] = dash
        body.append(polyline(points, **attrs))
        for x, y in points:
            body.append(circle(x, y, fill=color, stroke="#ffffff", **{"stroke-width": 0.8}))

    body.append(
        line(
            x_map(0.596994997024),
            y_map(0) - 18,
            x_map(0.596994997024),
            y_map(0) + 18,
            stroke=ORANGE,
            **{"stroke-width": 1.5, "stroke-dasharray": "2 2"},
        )
    )
    body.append(
        text(
            x_map(0.596994997024) + 6,
            y_map(0) - 22,
            "symmetric w=1: interpolated r≈0.597",
            **{"class": "note"},
        )
    )

    legend_y = 392
    for index, variant in enumerate(variants):
        color, dash, label = variants[variant]
        x = 40 + index * 175
        attrs = {"stroke": color, "stroke-width": 2.5}
        if dash:
            attrs["stroke-dasharray"] = dash
        body.append(line(x, legend_y - 4, x + 22, legend_y - 4, **attrs))
        body.append(text(x + 28, legend_y, label, **{"class": "legend"}))
    body.append(
        text(
            width / 2,
            422,
            "Below zero favors K-ARC under the named accounting; interpolation is within observed brackets only.",
            **{"class": "note", "text-anchor": "middle"},
        )
    )
    (HERE / "carry-price-sensitivity.svg").write_text(
        svg_document(width, height, body, "Carry-price and reuse sensitivity"),
        encoding="utf-8",
    )


def render_cross_runtime() -> None:
    rows = read_csv("cross-runtime-token.csv")
    wanted = [row for row in rows if row["arm"] in {"static-full", "karc"}]
    by_runtime: dict[str, dict[str, dict[str, str]]] = {}
    for row in wanted:
        by_runtime.setdefault(row["runtime"], {})[row["arm"]] = row

    width, height = 650, 420
    left, top, plot_w, plot_h = 75, 70, 520, 260
    body = [
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        text(
            28,
            28,
            "Cross-runtime direction on one axis: gross logical input",
            **{"class": "title"},
        ),
    ]
    for tick in range(6):
        value = tick / 5
        y = top + plot_h - value * plot_h
        body.append(line(left, y, left + plot_w, y, stroke=GRID, **{"stroke-width": 1}))
        body.append(
            text(
                left - 9,
                y + 4,
                f"{value:.1f}",
                **{"class": "axis", "text-anchor": "end"},
            )
        )
    body.append(line(left, top, left, top + plot_h, stroke=INK, **{"stroke-width": 1.2}))
    body.append(
        line(
            left,
            top + plot_h,
            left + plot_w,
            top + plot_h,
            stroke=INK,
            **{"stroke-width": 1.2},
        )
    )

    centers = [210, 465]
    for center, runtime in zip(centers, ("Claude Code", "Codex")):
        static = by_runtime[runtime]["static-full"]
        karc = by_runtime[runtime]["karc"]
        static_total = float(static["gross_input_tokens"])
        ratio = float(karc["gross_input_tokens"]) / static_total
        saving = float(karc["gross_saving_vs_static"])
        for x, value, color, label in (
            (center - 44, 1.0, GREY, "static-full"),
            (center + 22, ratio, BLUE, "K-ARC"),
        ):
            bar_w = 48
            y = top + plot_h - value * plot_h
            body.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w}" '
                f'height="{value * plot_h:.2f}" fill="{color}"/>'
            )
            body.append(
                text(
                    x + bar_w / 2,
                    y - 7,
                    f"{value:.3f}",
                    **{"class": "axis", "text-anchor": "middle"},
                )
            )
            body.append(
                text(
                    x + bar_w / 2,
                    top + plot_h + 18,
                    label,
                    **{"class": "axis", "text-anchor": "middle"},
                )
            )
        body.append(
            text(
                center,
                top + plot_h + 43,
                runtime,
                **{"class": "panel", "text-anchor": "middle"},
            )
        )
        body.append(
            text(
                center + 46,
                top + plot_h - ratio * plot_h - 23,
                f"−{saving * 100:.2f}%",
                **{"class": "legend", "text-anchor": "middle"},
            )
        )

    body.append(
        text(
            20,
            top + plot_h / 2,
            "gross input / static-full",
            **{
                "class": "axis",
                "text-anchor": "middle",
                "transform": f"rotate(-90 20 {top + plot_h / 2:.2f})",
            },
        )
    )
    body.append(
        text(
            width / 2,
            405,
            "Direction check only: model, tokenizer, CLI, turns, prompt assembly, and cache schema differ.",
            **{"class": "note", "text-anchor": "middle"},
        )
    )
    (HERE / "cross-runtime-gross.svg").write_text(
        svg_document(width, height, body, "Cross-runtime gross input direction"),
        encoding="utf-8",
    )


def render_horizon_reader_contrast() -> None:
    """Render separate H16 and Claude-reader panels without cross-cell scaling."""

    h16 = json.loads(H16_SUMMARY.read_text(encoding="utf-8"))
    xr = json.loads(XR_SUMMARY.read_text(encoding="utf-8"))
    no_h16_crossover = (
        h16["first_per_turn_gross_crossover"] is None
        and h16["first_cumulative_gross_crossover"] is None
    )
    h16_crossover_note = (
        "No observed per-turn or cumulative gross crossover through position 16."
        if no_h16_crossover
        else "A gross crossover is present in the committed H16 summary."
    )
    xr_ratio = float(
        xr["primary_direction"]["karc_to_rag_gross_input_per_task_ratio"]
    )
    xr_karc_hit = float(xr["by_arm"]["karc-full"]["provider_cache_hit_rate"])
    xr_rag_hit = float(xr["by_arm"]["rag-bm25"]["provider_cache_hit_rate"])
    width, height = 940, 470
    body = [
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        text(
            28,
            28,
            "Observed horizon and independent reader contrast",
            **{"class": "title"},
        ),
    ]

    # Panel A: H16 Codex cell. Only the committed per-position gross means are
    # connected; no curve is fitted or extended beyond the final observation.
    left, top, plot_w, plot_h = 65, 92, 520, 285
    y_max = 600_000.0
    body.append(text(left, 55, "A  E10-P2-H16 | Codex | 16-turn cell", **{"class": "panel"}))
    body.append(
        text(
            left,
            73,
            "gross input per turn; observed positions only",
            **{"class": "note"},
        )
    )
    for tick in range(7):
        value = y_max * tick / 6
        y = top + plot_h - plot_h * tick / 6
        body.append(line(left, y, left + plot_w, y, stroke=GRID, **{"stroke-width": 1}))
        body.append(
            text(
                left - 8,
                y + 4,
                f"{value / 1000:.0f}k",
                **{"class": "axis", "text-anchor": "end"},
            )
        )
    body.append(line(left, top, left, top + plot_h, stroke=INK, **{"stroke-width": 1.2}))
    body.append(
        line(
            left,
            top + plot_h,
            left + plot_w,
            top + plot_h,
            stroke=INK,
            **{"stroke-width": 1.2},
        )
    )
    for position in (1, 4, 8, 12, 16):
        x = left + (position - 1) * plot_w / 15
        body.append(
            text(
                x,
                top + plot_h + 18,
                position,
                **{"class": "axis", "text-anchor": "middle"},
            )
        )
    body.append(
        text(
            left + plot_w / 2,
            top + plot_h + 37,
            "Turn position",
            **{"class": "axis", "text-anchor": "middle"},
        )
    )

    h16_styles = {
        "karc-full": (BLUE, "K-ARC"),
        "rag-bm25": (ORANGE, "RAG"),
    }
    for arm, (color, _) in h16_styles.items():
        series = h16["by_position"][arm]
        points = []
        for position in range(1, 17):
            x = left + (position - 1) * plot_w / 15
            y = top + plot_h - float(series[str(position)]["gross_mean"]) * plot_h / y_max
            points.append((x, y))
        body.append(polyline(points, stroke=color, **{"stroke-width": 2.5}))
        for x, y in points:
            body.append(circle(x, y, fill=color, stroke="#ffffff", **{"stroke-width": 0.8}))
    for index, (_, (color, label)) in enumerate(h16_styles.items()):
        x = left + 118 + index * 145
        body.append(line(x, 427, x + 24, 427, stroke=color, **{"stroke-width": 2.5}))
        body.append(text(x + 30, 431, label, **{"class": "legend"}))
    body.append(
        text(
            left + plot_w / 2,
            451,
            h16_crossover_note,
            **{"class": "note", "text-anchor": "middle"},
        )
    )

    # Panel B: the independent Claude reader cell. This panel has its own
    # origin, scale, metric, and runtime label so its bar heights are never a
    # cross-runtime comparison with Panel A.
    b_left, b_top, b_w, b_h = 650, 92, 240, 285
    b_y_max = 75_000.0
    body.append(text(b_left, 55, "B  E10-P2-XR", **{"class": "panel"}))
    body.append(
        text(
            b_left,
            73,
            "Claude | 8-turn | gross/task | own axis",
            **{"class": "note"},
        )
    )
    for tick in range(4):
        value = b_y_max * tick / 3
        y = b_top + b_h - b_h * tick / 3
        body.append(line(b_left, y, b_left + b_w, y, stroke=GRID, **{"stroke-width": 1}))
        body.append(
            text(
                b_left - 8,
                y + 4,
                f"{value / 1000:.0f}k",
                **{"class": "axis", "text-anchor": "end"},
            )
        )
    body.append(line(b_left, b_top, b_left, b_top + b_h, stroke=INK, **{"stroke-width": 1.2}))
    body.append(
        line(
            b_left,
            b_top + b_h,
            b_left + b_w,
            b_top + b_h,
            stroke=INK,
            **{"stroke-width": 1.2},
        )
    )
    xr_styles = {
        "karc-full": (BLUE, "K-ARC"),
        "rag-bm25": (ORANGE, "RAG"),
    }
    for index, (arm, (color, label)) in enumerate(xr_styles.items()):
        value = float(xr["by_arm"][arm]["gross_input_per_task"])
        x = b_left + 40 + index * 105
        y = b_top + b_h - value * b_h / b_y_max
        body.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="55" '
            f'height="{value * b_h / b_y_max:.2f}" fill="{color}"/>'
        )
        body.append(
            text(
                x + 27.5,
                y - 7,
                f"{value:,.1f}",
                **{"class": "axis", "text-anchor": "middle"},
            )
        )
        body.append(
            text(
                x + 27.5,
                b_top + b_h + 18,
                label,
                **{"class": "axis", "text-anchor": "middle"},
            )
        )
    body.append(
        text(
            b_left + b_w / 2,
            425,
            f"K/R gross/task = {xr_ratio:.4f}×",
            **{"class": "legend", "text-anchor": "middle"},
        )
    )
    body.append(
        text(
            b_left + b_w / 2,
            443,
            f"Claude-internal hit: {xr_karc_hit:.2%} / {xr_rag_hit:.2%}",
            **{"class": "note", "text-anchor": "middle"},
        )
    )
    body.append(
        text(
            width / 2,
            466,
            "Separate cells, axes, and estimands; panel heights are not cross-runtime magnitudes.",
            **{"class": "note", "text-anchor": "middle"},
        )
    )
    (HERE / "horizon-reader-contrast.svg").write_text(
        svg_document(
            width,
            height,
            body,
            "H16 gross horizon and independent Claude reader contrast",
            (
                "Generated only from committed "
                "docs/experiments/E10-P2-H16/run/raw/summary.json and "
                "docs/experiments/E10-P2-XR/canary/raw/summary.json."
            ),
        ),
        encoding="utf-8",
    )



def render_accounting_forest() -> None:
    """Two accountings across the eight component configurations, plus the reuse-density axis.

    Each row is one cell; the two markers are its paired gross ratio and its
    paired four-component ratio with 95% intervals.  The vertical rule is cost
    parity.  Every number is read from a committed analysis.json, so the figure
    carries no derived quantity of its own.  The lower panel varies reuse
    density and is kept visually separate because it is a different workload
    point that does not enter the six-cell range figures.
    """
    e18 = json.loads(E18_ANALYSIS.read_text(encoding="utf-8"))
    e19 = json.loads(E19_ANALYSIS.read_text(encoding="utf-8"))
    e21 = json.loads(E21_ANALYSIS.read_text(encoding="utf-8"))
    e23 = json.loads(E23_ANALYSIS.read_text(encoding="utf-8"))
    e24 = json.loads(E24_ANALYSIS.read_text(encoding="utf-8"))
    e25 = json.loads(E25_ANALYSIS.read_text(encoding="utf-8"))
    e26 = json.loads(E26_ANALYSIS.read_text(encoding="utf-8"))

    def triple(node: dict) -> tuple[float, float, float]:
        return float(node["point"]), float(node["ci95_lo"]), float(node["ci95_hi"])

    base = e24["this_run"]
    rows = [
        ("base",
         triple(base["gross_ratio_boot"]),
         triple(base["priced_ratio_boot"]), True),
    ]
    for tag, label in (("A", "pi x Anthropic"), ("B", "pi x OpenAI")):
        leg = e18["legs"][tag]
        rows.append((f"E18-{tag}",
                     triple(leg["gross_ratio_karc_over_rag"]),
                     triple(leg["priced_ratio_karc_over_rag"]), True))
    for tag, label in (("A", "pi x Anthropic"), ("B", "pi x OpenAI")):
        leg = e19["legs"][tag]
        rows.append((f"E19-{tag}",
                     triple(leg["gross_ratio_karc_over_rag"]),
                     triple(leg["priced_ratio_karc_over_rag"]), True))
    st = e21["statistics"]
    rows.append(("E21",
                 triple(st["gross_ratio_karc_over_rag"]),
                 triple(st["priced_ratio_karc_over_rag"]), True))
    st25 = e25["statistics"]
    rows.append(("E25",
                 triple(st25["gross_ratio_karc_over_rag"]),
                 triple(st25["priced_ratio_karc_over_rag"]), True))
    rows.append(("E26",
                 triple(e26["this_run"]["gross_ratio_boot"]),
                 triple(e26["this_run"]["priced_ratio_boot"]), True))
    # 논문 이름(harness-provider-window)으로 바꾸고 harness별로 정렬한다.
    # 매핑: docs/paper2/configuration-names.md
    names = {"base": "CC-A-g", "E26": "CC-A-s", "E18-A": "pi-A-g", "E18-B": "pi-O-g",
             "E19-A": "pi-A-s", "E19-B": "pi-O-s", "E25": "Cx-O-g", "E21": "Cx-O-s"}
    order = ["CC-A-g", "CC-A-s", "pi-A-g", "pi-O-g", "pi-A-s", "pi-O-s", "Cx-O-g", "Cx-O-s"]
    rows = sorted(((names[r[0]],) + tuple(r[1:]) for r in rows),
                  key=lambda r: order.index(r[0]))

    lower = []
    for tag, label in (("025", "reuse 0.25"), ("075", "reuse 0.75")):
        node = e23["rho"][tag]
        lower.append((f"{label}",
                      triple(node["gross_ratio_karc_over_rag"]),
                      triple(node["priced_ratio_karc_over_rag"]), True))
    # 상주 예산 축 (pi-O-g 구성, 성장 창 S01-S12, 예산만 변경).  20%·Anthropic 10% 는
    # 오독 위험과 x축 하한(0.3) 절단 때문에 넣지 않는다 — docs/paper2/configuration-names.md
    e28 = json.loads(E28_ANALYSIS.read_text(encoding="utf-8"))
    for tag, label in (("o025", "budget 0.025 g"), ("o10", "budget 0.10 g")):
        st = e28["cells"][tag]["statistics"]
        lower.append((label,
                      triple(st["gross_ratio_karc_over_rag"]),
                      triple(st["priced_ratio_karc_over_rag"]), True))

    # 폭 505pt(두 단)로 내보내면 배율이 0.90 이 되어 11px 글자가 본문과 비슷한
    # 9.9pt 로 놓인다.  제목·부제·주석은 캡션으로 옮겼으므로 캔버스에 두지 않는다.
    width, height = 560, 282
    left, right = 80.0, 548.0
    top = 30.0
    step = 18.0
    x_lo, x_hi = math.log10(0.3), math.log10(3.0)

    def x_map(value: float) -> float:
        v = max(value, 0.3)
        return left + (math.log10(v) - x_lo) / (x_hi - x_lo) * (right - left)

    body = ['<rect width="100%" height="100%" fill="#ffffff"/>']

    for tick in (0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0):
        x = x_map(tick)
        body.append(line(x, top - 8, x, top + step * (len(rows) + len(lower) + 1) + 2,
                         stroke=GRID, **{"stroke-width": 1.0}))
        body.append(text(x, top - 13, f"{tick:g}",
                         **{"font-size": 11, "fill": GREY, "text-anchor": "middle"}))
    parity = x_map(1.0)
    body.append(line(parity, top - 8, parity,
                     top + step * (len(rows) + len(lower) + 1) + 2,
                     stroke=INK, **{"stroke-width": 1.4, "stroke-dasharray": "4 3"}))

    def draw(items: list, y0: float) -> float:
        y = y0
        for label, gross, priced, has_ci in items:
            body.append(text(left - 8, y + 4, label,
                             **{"font-size": 11, "fill": INK, "text-anchor": "end"}))
            for (point, lo, hi), colour, dy in ((gross, ORANGE, -4.0), (priced, BLUE, 4.0)):
                if has_ci and hi > lo:
                    body.append(line(x_map(lo), y + dy, x_map(hi), y + dy,
                                     stroke=colour, **{"stroke-width": 2.0}))
                    for end in (lo, hi):
                        body.append(line(x_map(end), y + dy - 2.6, x_map(end), y + dy + 2.6,
                                         stroke=colour, **{"stroke-width": 1.4}))
                body.append(circle(x_map(point), y + dy, 2.8, fill=colour))
            y += step
        return y

    y = draw(rows, top + 6)
    body.append(line(left - 62, y - 3, right, y - 3, stroke=GRID, **{"stroke-width": 1.2}))
    y = draw(lower, y + 7)

    legend_y = y + 13
    body.append(circle(left + 4, legend_y - 4, 2.8, fill=ORANGE))
    body.append(text(left + 13, legend_y, "gross input tokens",
                     **{"font-size": 11, "fill": INK}))
    body.append(circle(left + 150, legend_y - 4, 2.8, fill=BLUE))
    body.append(text(left + 159, legend_y, "component-weighted cost",
                     **{"font-size": 11, "fill": INK}))

    (HERE / "accounting-forest.svg").write_text(
        svg_document(width, height, body,
                     "Two accountings across cells and reuse densities"),
        encoding="utf-8",
    )


def render_accounting_scatter() -> None:
    """Two accountings of the same turns as a quadrant scatter (paper Figure 1).

    x = paired gross-input ratio, y = paired component-weighted cost ratio, both
    managed over retrieval on one log scale; crosses are 95% intervals.  The two
    dashed lines are parity, so the lower-right quadrant is "more tokens, lower
    cost".  The grey diagonal is y = x: below it the managed arm pays less
    component-weighted cost per gross input token.  Filled circles are the eight
    component configurations; open circles (reuse density) and open squares
    (resident budget) are workload points varied from pi-O-s and pi-O-g, joined
    to those cells by dotted paths and not pooled with the eight.  Every number
    is read from a committed analysis.json.  Names follow
    docs/paper2/configuration-names.md.
    """
    e18 = json.loads(E18_ANALYSIS.read_text(encoding="utf-8"))
    e19 = json.loads(E19_ANALYSIS.read_text(encoding="utf-8"))
    e21 = json.loads(E21_ANALYSIS.read_text(encoding="utf-8"))
    e23 = json.loads(E23_ANALYSIS.read_text(encoding="utf-8"))
    e24 = json.loads(E24_ANALYSIS.read_text(encoding="utf-8"))
    e25 = json.loads(E25_ANALYSIS.read_text(encoding="utf-8"))
    e26 = json.loads(E26_ANALYSIS.read_text(encoding="utf-8"))
    e28 = json.loads(E28_ANALYSIS.read_text(encoding="utf-8"))
    G, P = "gross_ratio_karc_over_rag", "priced_ratio_karc_over_rag"

    def T(node: dict) -> tuple[float, float, float]:
        return float(node["point"]), float(node["ci95_lo"]), float(node["ci95_hi"])

    eight = [
        ("CC-A-g", T(e24["this_run"]["gross_ratio_boot"]), T(e24["this_run"]["priced_ratio_boot"])),
        ("CC-A-s", T(e26["this_run"]["gross_ratio_boot"]), T(e26["this_run"]["priced_ratio_boot"])),
        ("pi-A-g", T(e18["legs"]["A"][G]), T(e18["legs"]["A"][P])),
        ("pi-O-g", T(e18["legs"]["B"][G]), T(e18["legs"]["B"][P])),
        ("pi-A-s", T(e19["legs"]["A"][G]), T(e19["legs"]["A"][P])),
        ("pi-O-s", T(e19["legs"]["B"][G]), T(e19["legs"]["B"][P])),
        ("Cx-O-g", T(e25["statistics"][G]), T(e25["statistics"][P])),
        ("Cx-O-s", T(e21["statistics"][G]), T(e21["statistics"][P])),
    ]
    # 20%·Anthropic 10% 예산 cell은 넣지 않는다 — 비교 대상만 비싸진 점이라 "큰 상주가
    # 유리"로 오독되고, cost 구간이 y축 하한 0.3 아래로 잘린다.
    work = [
        ("reuse 0.25", T(e23["rho"]["025"][G]), T(e23["rho"]["025"][P])),
        ("reuse 0.75", T(e23["rho"]["075"][G]), T(e23["rho"]["075"][P])),
        ("budget 2.5%", T(e28["cells"]["o025"]["statistics"][G]), T(e28["cells"]["o025"]["statistics"][P])),
        ("budget 10%", T(e28["cells"]["o10"]["statistics"][G]), T(e28["cells"]["o10"]["statistics"][P])),
    ]
    for _, g, c in eight + work:      # 축 안에 들어오는지 — 절단 금지
        assert 0.5 <= g[1] and g[2] <= 3.0 and 0.3 <= c[1] and c[2] <= 1.5, (_, g, c)

    WHITE = "#ffffff"
    W = 240                                   # 단일 단 폭 → svg_to_tikz --width-pt 240 에서 배율 1.0
    px0, px1 = 30.0, 232.0
    xr, yr = (0.5, 3.0), (0.3, 1.5)
    ppd = (px1 - px0) / (math.log10(xr[1]) - math.log10(xr[0]))   # 두 축 같은 pt/decade
    py0 = 6.0
    py1 = py0 + (math.log10(yr[1]) - math.log10(yr[0])) * ppd

    def X(v: float) -> float:
        return px0 + (math.log10(v) - math.log10(xr[0])) * ppd

    def Y(v: float) -> float:
        return py1 - (math.log10(v) - math.log10(yr[0])) * ppd

    def rect(x: float, y: float, w: float, h: float, fill: str) -> str:
        return f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" fill="{fill}"/>'

    def open_square(x: float, y: float, half: float) -> list[str]:
        pts = [(x - half, y - half), (x + half, y - half), (x + half, y + half),
               (x - half, y + half), (x - half, y - half)]
        return [rect(x - half, y - half, 2 * half, 2 * half, WHITE),
                polyline(pts, stroke=INK, **{"stroke-width": 0.9})]

    H = int(py1 + 26)
    body = [rect(0, 0, W, H, WHITE)]
    body.append(line(px0, py1, px1, py1, stroke=GREY, **{"stroke-width": 0.6}))
    body.append(line(px0, py0, px0, py1, stroke=GREY, **{"stroke-width": 0.6}))
    for t in (0.5, 0.7, 1.0, 1.5, 2.0, 3.0):
        body.append(line(X(t), py1, X(t), py1 + 2.5, stroke=GREY, **{"stroke-width": 0.6}))
        body.append(text(X(t), py1 + 10, f"{t:g}",
                         **{"font-size": 7.5, "fill": GREY, "text-anchor": "middle"}))
    for t in (0.3, 0.5, 0.7, 1.0, 1.5):
        body.append(line(px0 - 2.5, Y(t), px0, Y(t), stroke=GREY, **{"stroke-width": 0.6}))
        body.append(text(px0 - 4.5, Y(t) + 2.6, f"{t:g}",
                         **{"font-size": 7.5, "fill": GREY, "text-anchor": "end"}))
    body.append(text((px0 + px1) / 2, py1 + 21, "gross-input ratio (managed / retrieval)",
                     **{"font-size": 8, "fill": INK, "text-anchor": "middle"}))
    ymid = (py0 + py1) / 2
    body.append(text(8, ymid, "component-weighted cost ratio",
                     **{"font-size": 8, "fill": INK, "text-anchor": "middle",
                        "transform": f"rotate(-90 8 {ymid:.2f})"}))
    # y = x : 두 비가 같은 선.  아래쪽 = 상주 관리 arm 의 gross 토큰당 성분 가격이 더 낮다.
    body.append(line(X(0.5), Y(0.5), X(1.5), Y(1.5), stroke=GRID, **{"stroke-width": 0.9}))
    body.append(text(X(1.5) + 2, Y(1.5) - 3, "equal ratios",
                     **{"font-size": 6.5, "fill": GREY, "font-style": "italic"}))
    body.append(line(X(1.0), py0, X(1.0), py1, stroke=INK,
                     **{"stroke-width": 0.8, "stroke-dasharray": "3 2"}))
    body.append(line(px0, Y(1.0), px1, Y(1.0), stroke=INK,
                     **{"stroke-width": 0.8, "stroke-dasharray": "3 2"}))
    body.append(text(px0 + 4, Y(1.0) - 4, "fewer tokens, higher cost: none",
                     **{"font-size": 7, "fill": GREY, "font-style": "italic"}))
    body.append(text(px1 - 3, py1 - 5, "more tokens, lower cost",
                     **{"font-size": 7, "fill": GREY, "font-style": "italic",
                        "text-anchor": "end"}))

    pts = {name: (g, c) for name, g, c in eight + work}
    for a, base_name, b in (("reuse 0.25", "pi-O-s", "reuse 0.75"),
                            ("budget 2.5%", "pi-O-g", "budget 10%")):
        for u, v in ((a, base_name), (base_name, b)):
            body.append(line(X(pts[u][0][0]), Y(pts[u][1][0]),
                             X(pts[v][0][0]), Y(pts[v][1][0]),
                             stroke=GREY, **{"stroke-width": 0.7, "stroke-dasharray": "1.2 1.6"}))
    for name, g, c in eight + work:   # 95% 구간: 가로 = gross, 세로 = cost
        body.append(line(X(g[1]), Y(c[0]), X(g[2]), Y(c[0]), stroke=INK, **{"stroke-width": 0.6}))
        body.append(line(X(g[0]), Y(c[1]), X(g[0]), Y(c[2]), stroke=INK, **{"stroke-width": 0.6}))
    for name, g, c in eight:
        body.append(circle(X(g[0]), Y(c[0]), 2.3, fill=INK))
    for name, g, c in work:
        if name.startswith("reuse"):
            body.append(circle(X(g[0]), Y(c[0]), 2.3, fill=WHITE, stroke=INK,
                               **{"stroke-width": 0.9}))
        else:
            body.extend(open_square(X(g[0]), Y(c[0]), 2.3))
    # 라벨 오프셋은 손으로 배치한 값이다 (데이터 동결; 충돌 검사는 렌더로).
    lab = {
        "CC-A-g": (0, 16.5, "middle"), "CC-A-s": (-3, -6, "end"),
        "pi-A-g": (5, -4, "start"), "pi-O-g": (5, 10, "start"),
        "pi-A-s": (5.5, 2.5, "start"), "pi-O-s": (-4.5, -3.5, "end"),
        "Cx-O-g": (-13.5, 1, "end"), "Cx-O-s": (3, 10, "start"),
        "reuse 0.25": (4, 9.5, "start"), "reuse 0.75": (-6, 2.5, "end"),
        "budget 2.5%": (0, -14, "middle"), "budget 10%": (-3.5, 8.5, "end"),
    }
    for name, g, c in eight + work:
        dx, dy, anc = lab[name]
        body.append(text(X(g[0]) + dx, Y(c[0]) + dy, name,
                         **{"font-size": 7, "fill": INK, "text-anchor": anc}))
    lx, ly = px0 + 6, py0 + 12
    body.append(circle(lx, ly - 2.5, 2.3, fill=INK))
    body.append(text(lx + 6, ly, "component configuration", **{"font-size": 7, "fill": INK}))
    body.append(circle(lx, ly + 7.5, 2.3, fill=WHITE, stroke=INK, **{"stroke-width": 0.9}))
    body.append(text(lx + 6, ly + 10, "reuse density", **{"font-size": 7, "fill": INK}))
    body.extend(open_square(lx, ly + 17.5, 2.3))
    body.append(text(lx + 6, ly + 20, "resident budget", **{"font-size": 7, "fill": INK}))

    (HERE / "accounting-scatter.svg").write_text(
        svg_document(W, H, body, "Two accountings of the same turns, quadrant scatter"),
        encoding="utf-8",
    )


def render_baseline_first() -> None:
    """Five-arm cost/quality scatter for the frozen H16 E11-BASE3 cell.

    Cost is arm-total gross input on a log axis; quality is exact-answer
    correctness.  Every point comes from one frozen cell measured on one
    runtime, so the figure carries no cross-cell or cross-runtime comparison.
    """
    summary = json.loads(BASE3_SUMMARY.read_text(encoding="utf-8"))
    # Gross totals come from E22, which differenced the thread-cumulative Codex
    # reporting that the committed summary sums naively.  Correctness is per-turn
    # pass counts, so it is read from the committed summary unchanged.
    corrected = json.loads(BASE3_CORRECTED.read_text(encoding="utf-8"))["arms"]
    arms = [
        ("stateless-rag", "stateless RAG", PURPLE),
        ("sliding-window-compaction", "sliding-window compaction", GREEN),
        ("rag-bm25", "stateful RAG", ORANGE),
        ("karc-full", "managed memory", BLUE),
        ("full-history", "full history", GREY),
    ]
    points = []
    for key, label, colour in arms:
        entry = summary["by_arm"][key]
        gross = float(corrected[key]["corrected"]["gross"])
        points.append((label, colour, gross, float(entry["correct_rate"])))

    width, height = 900, 470
    left, right, top, bottom = 96.0, 852.0, 84.0, 352.0
    x_lo, x_hi = math.log10(2e6), math.log10(2e7)
    y_lo, y_hi = 0.925, 0.945

    def x_map(gross: float) -> float:
        return left + (math.log10(gross) - x_lo) / (x_hi - x_lo) * (right - left)

    def y_map(rate: float) -> float:
        return bottom - (rate - y_lo) / (y_hi - y_lo) * (bottom - top)

    body = ['<rect width="100%" height="100%" fill="#ffffff"/>']
    body.append(
        text(
            left - 6,
            36,
            "Cost and quality of five arms in one frozen cell",
            **{"class": "title"},
        )
    )
    body.append(
        text(
            left - 6,
            56,
            "E11-BASE3 | Codex | frozen H16 cell | 12 paired sessions x 16 turns",
            **{"class": "note"},
        )
    )

    for gross in (2e6, 3e6, 5e6, 7e6, 1e7, 1.5e7, 2e7):
        x = x_map(gross)
        body.append(line(x, top, x, bottom, stroke=GRID, **{"stroke-width": 1}))
        body.append(
            text(
                x,
                bottom + 18,
                f"{gross / 1e6:.0f}M",
                **{"class": "axis", "text-anchor": "middle"},
            )
        )
    for pct in (92.5, 93.0, 93.5, 94.0, 94.5):
        y = y_map(pct / 100.0)
        body.append(line(left, y, right, y, stroke=GRID, **{"stroke-width": 1}))
        body.append(
            text(
                left - 10,
                y + 4,
                f"{pct:.1f}%",
                **{"class": "axis", "text-anchor": "end"},
            )
        )

    body.append(line(left, bottom, right, bottom, stroke=INK, **{"stroke-width": 1.4}))
    body.append(line(left, top, left, bottom, stroke=INK, **{"stroke-width": 1.4}))
    body.append(
        text(
            (left + right) / 2,
            bottom + 40,
            "arm-total gross input tokens, log scale (cheaper to the left)",
            **{"class": "axis", "text-anchor": "middle"},
        )
    )
    mid_y = (top + bottom) / 2
    body.append(
        text(
            left - 66,
            mid_y,
            "exact-answer correctness",
            **{
                "class": "axis",
                "text-anchor": "middle",
                "transform": f"rotate(-90 {left - 66:.2f} {mid_y:.2f})",
            },
        )
    )

    layout = {
        "stateless RAG": (-6, -18, "start"),
        "sliding-window compaction": (0, -18, "middle"),
        "stateful RAG": (-10, 30, "end"),
        "managed memory": (10, -18, "start"),
        "full history": (0, 28, "end"),
    }
    for label, colour, gross, rate in points:
        x, y = x_map(gross), y_map(rate)
        body.append(
            circle(x, y, 6.5, fill=colour, stroke="#ffffff", **{"stroke-width": 1.6})
        )
        dx, dy, anchor = layout[label]
        body.append(
            text(
                x + dx,
                y + dy,
                f"{label} \u2014 {gross / 1e6:,.1f}M, {rate * 100:.2f}%",
                fill=colour,
                **{"class": "legend", "text-anchor": anchor},
            )
        )

    notes = (
        "Both stateful arms and full history sit 1.5x to 6.9x to the right of the "
        "near-stateless arms at comparable correctness.",
        "This reproduces a previously reported quality-cost direction in an agentic-coding "
        "domain; it claims no novelty and ranks no memory architecture.",
        "One frozen cell, one runtime: no cross-cell or cross-runtime magnitude is implied.",
    )
    for index, note in enumerate(notes):
        body.append(text(left - 6, bottom + 64 + index * 16, note, **{"class": "note"}))

    out = HERE / "baseline-first.svg"
    out.write_text(
        svg_document(
            width,
            height,
            body,
            "Five-arm cost and quality in the frozen H16 BASE3 cell",
            source_desc=(
                "Generated only from committed "
                "docs/experiments/E11-BASE3/run/raw/summary.json."
            ),
        ),
        encoding="utf-8",
    )


def main() -> None:
    render_baseline_first()
    render_accounting_forest()
    render_accounting_scatter()
    render_turn_curves()
    render_horizon_reader_contrast()
    render_carry_break_even()
    render_cross_runtime()


if __name__ == "__main__":
    main()
