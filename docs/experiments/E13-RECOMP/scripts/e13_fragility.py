#!/usr/bin/env python3
"""E13-RECOMP fragility analysis of the condition-2 cost-ordering reversal.

Zero model / network / embedding / fixture / benchmark activity.  Every input is
a file already committed to this repository.  The bootstrap machinery is imported
from `e13_recomp.py` (seed 1313, 10,000 resamples, common random numbers), so the
session-level intervals here are directly comparable with E13 report §4.2 and the
output is byte-reproducible.

Items:
  A  position-1 exclusion (positions 2-8 only): both accountings, both CIs,
     both flip thresholds, and how the margins move.
  B  leave-one-session-out (12 folds) on the priced ordering, with the gross
     ordering as a control.
  C  token-level perturbation bounds that would kill the reversal.
  D  1-hour-TTL provenance audit over committed sources only (mechanical grep).
  E  inventory of every other committed cell on which the four-component
     accounting is computable.

Usage:
    PYTHONPATH=$PWD/src .venv/bin/python \
        docs/experiments/E13-RECOMP/scripts/e13_fragility.py

Writes docs/experiments/E13-RECOMP/raw/fragility.json.
"""

from __future__ import annotations

import importlib.util
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
E13 = HERE.parent
REPO = E13.parents[2]
RAW_OUT = E13 / "raw"

XR = REPO / "docs" / "experiments" / "E10-P2-XR" / "canary"
XR_SMOKE = REPO / "docs" / "experiments" / "E10-P2-XR" / "smoke"


def _load_e13():
    """Import the E13 generator as a module without running its main()."""
    spec = importlib.util.spec_from_file_location("e13_recomp", HERE / "e13_recomp.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


E13M = _load_e13()

SESSIONS = E13M.SESSIONS
BOOTSTRAP_SEED = E13M.BOOTSTRAP_SEED
BOOTSTRAP_RESAMPLES = E13M.BOOTSTRAP_RESAMPLES
P_IN = E13M.P_IN
MU_W_1H = E13M.MU_CREATION_1H  # 2.0
MU_W_5M = E13M.MU_CREATION_5M  # 1.25
MU_R = E13M.MU_READ  # 0.1
MU_O = E13M.MU_OUT  # 5.0
ARMS = ("karc-full", "rag-bm25")

# Declared before any fragility number was inspected: the perturbation is applied
# to one component of one arm at a time, holding every other counter fixed.
PERTURBATION_TARGETS = (
    ("karc-full", "R", "add"),  # more cache-read for karc
    ("rag-bm25", "W", "remove"),  # less cache-creation for rag
)

# Mechanical Item-D audit: any of these tokens appearing in a *harness-authored*
# file would mean the 1-hour TTL was our configuration choice.  Fixture corpora
# are excluded because their synthetic documents contain unrelated
# `*_cache_ttl_s` config values.
TTL_SETTING_PATTERNS = (
    r"cache_control",
    r'ttl\s*[=:]\s*["\']1h',
    r'"ttl"',
    r"ANTHROPIC_[A-Z_]*CACHE",
    r"CLAUDE_CODE_[A-Z_]*CACHE",
)
TTL_AUDIT_ROOTS = (
    "src/karc",
    "scripts",
    "docs/experiments/E10-P2-XR",
)
TTL_AUDIT_SUFFIXES = (".py", ".json", ".jsonl", ".md", ".yaml", ".yml", ".toml")


# --- helpers -----------------------------------------------------------------
def resample_matrix(width: int, seed: int = BOOTSTRAP_SEED) -> list[list[int]]:
    """Index matrix of the declared E13 shape, widened/narrowed to `width`.

    For width == 12 this is byte-identical to E13's own matrix, so item A is
    directly comparable with report §4.2.  For width == 11 one matrix is built
    once and shared by all 12 leave-one-out folds (common random numbers across
    folds).
    """
    rng = random.Random(seed)
    return [
        [rng.randrange(width) for _ in range(width)]
        for _ in range(BOOTSTRAP_RESAMPLES)
    ]


def paired_bootstrap(values: list[float], matrix: list[list[int]]) -> dict:
    point = sum(values) / len(values)
    draws = [sum(values[i] for i in row) / len(row) for row in matrix]
    lo, hi = E13M.percentile_ci(draws)
    return {
        "point": point,
        "ci95_lo": lo,
        "ci95_hi": hi,
        "excludes_zero": (lo > 0.0) or (hi < 0.0),
    }


def paired_ratio(num: list[float], den: list[float], matrix: list[list[int]]) -> dict:
    total_den = sum(den)
    assert total_den != 0, "NOT-COMPUTABLE: zero denominator in paired ratio"
    draws = []
    degenerate = 0
    for row in matrix:
        d = sum(den[i] for i in row)
        if d == 0:
            degenerate += 1
            continue
        draws.append(sum(num[i] for i in row) / d)
    lo, hi = E13M.percentile_ci(draws)
    return {
        "point": sum(num) / total_den,
        "ci95_lo": lo,
        "ci95_hi": hi,
        "entirely_below_one": hi < 1.0,
        "entirely_above_one": lo > 1.0,
        "degenerate_resamples": degenerate,
        "usable_resamples": len(draws),
    }


def component_table(rows: list[dict], sessions: list[str]) -> dict:
    """Per-arm, per-session component sums over the supplied turn rows."""
    fields = {
        "U": "fresh_input_tokens_provider",
        "W1h": "cache_creation_1h_input_tokens_provider",
        "W5m": "cache_creation_5m_input_tokens_provider",
        "R": "cache_read_input_tokens_provider",
        "O": "output_tokens",
        "G": "gross_input_tokens",
        "USD": "rate_card_equivalent_usd",
    }
    out = {
        arm: {key: {s: 0.0 for s in sessions} for key in fields} for arm in ARMS
    }
    for row in rows:
        arm = row["arm"]
        if arm not in out:
            continue
        for key, field in fields.items():
            out[arm][key][row["session_id"]] += row[field]
    return out


def totals_of(comp: dict, sessions: list[str]) -> dict:
    return {
        arm: {key: sum(series[s] for s in sessions) for key, series in per.items()}
        for arm, per in comp.items()
    }


def deltas_of(tot: dict) -> dict:
    k, r = tot["karc-full"], tot["rag-bm25"]
    return {
        "uncached": k["U"] - r["U"],
        "creation": (k["W1h"] + k["W5m"]) - (r["W1h"] + r["W5m"]),
        "cache_read": k["R"] - r["R"],
        "output": k["O"] - r["O"],
    }


def flip_thresholds(d: dict) -> dict:
    """Closed-form flip boundaries of dU + mu_w*dW + mu_r*dR + mu_o*dO = 0."""
    out = {}
    for mu_o in (5.0, 0.0):
        mu_r_flip = -(d["uncached"] + MU_W_1H * d["creation"] + mu_o * d["output"]) / d[
            "cache_read"
        ]
        mu_w_flip = -(d["uncached"] + MU_R * d["cache_read"] + mu_o * d["output"]) / d[
            "creation"
        ]
        out[f"mu_o={mu_o}"] = {
            "mu_r_flip_at_observed_mu_w_2_0": mu_r_flip,
            "observed_mu_r": MU_R,
            "mu_r_headroom_pct": (mu_r_flip / MU_R - 1.0) * 100.0,
            "karc_cheaper_at_observed_mu_r": MU_R < mu_r_flip,
            "mu_w_flip_at_observed_mu_r_0_1": mu_w_flip,
            "observed_mu_w_1h": MU_W_1H,
            "mu_w_headroom_pct": (MU_W_1H / mu_w_flip - 1.0) * 100.0,
            "reversal_holds_at_1h_weight": MU_W_1H > mu_w_flip,
            "reversal_holds_at_5m_weight_1_25": MU_W_5M > mu_w_flip,
        }
    return out


def priced_margin_input_equivalent(d: dict) -> float:
    """karc - rag in uncached-input-equivalent tokens at the observed rate card."""
    return (
        d["uncached"]
        + MU_W_1H * d["creation"]
        + MU_R * d["cache_read"]
        + MU_O * d["output"]
    )


# --- main --------------------------------------------------------------------
def main() -> int:
    xr_turns = E13M.read_jsonl(XR / "raw" / "turns.jsonl")
    assert len(xr_turns) == 192, f"NEEDS-DATA: expected 192 XR turns, got {len(xr_turns)}"
    rate_card = E13M.verify_rate_card(xr_turns)
    assert rate_card["exact_within_1e_9_usd"], "NEEDS-DATA: rate card not recovered"

    inputs = {
        "docs/experiments/E10-P2-XR/canary/raw/turns.jsonl": E13M.sha256_file(
            XR / "raw" / "turns.jsonl"
        ),
        "docs/experiments/E10-P2-XR/canary/raw/schedule.json": E13M.sha256_file(
            XR / "raw" / "schedule.json"
        ),
        "docs/experiments/E10-P2-XR/smoke/raw/turns.jsonl": E13M.sha256_file(
            XR_SMOKE / "raw" / "turns.jsonl"
        ),
        "docs/experiments/E10-P2-XR/smoke/raw/summary.json": E13M.sha256_file(
            XR_SMOKE / "raw" / "summary.json"
        ),
        "docs/experiments/E10-P2-XR/smoke/raw/validation.json": E13M.sha256_file(
            XR_SMOKE / "raw" / "validation.json"
        ),
        "docs/experiments/E13-RECOMP/raw/recomp.json": E13M.sha256_file(
            E13 / "raw" / "recomp.json"
        ),
    }

    matrix12 = resample_matrix(12)
    matrix11 = resample_matrix(11)

    # ============== shared: full cell and positions-2-8 cell =================
    def build(rows: list[dict], label: str) -> dict:
        comp = component_table(rows, SESSIONS)
        tot = totals_of(comp, SESSIONS)
        d = deltas_of(tot)
        usd_k = [comp["karc-full"]["USD"][s] for s in SESSIONS]
        usd_r = [comp["rag-bm25"]["USD"][s] for s in SESSIONS]
        g_k = [comp["karc-full"]["G"][s] for s in SESSIONS]
        g_r = [comp["rag-bm25"]["G"][s] for s in SESSIONS]
        return {
            "label": label,
            "n_turns": len(rows),
            "component_totals": tot,
            "component_deltas_karc_minus_rag": d,
            "gross_ordering": {
                "karc_gross": tot["karc-full"]["G"],
                "rag_gross": tot["rag-bm25"]["G"],
                "karc_over_rag": tot["karc-full"]["G"] / tot["rag-bm25"]["G"],
                "paired_ratio": paired_ratio(g_k, g_r, matrix12),
                "paired_mean_diff_tokens": paired_bootstrap(
                    [a - b for a, b in zip(g_k, g_r)], matrix12
                ),
            },
            "priced_ordering": {
                "karc_usd": tot["karc-full"]["USD"],
                "rag_usd": tot["rag-bm25"]["USD"],
                "karc_minus_rag_usd": tot["karc-full"]["USD"] - tot["rag-bm25"]["USD"],
                "karc_over_rag": tot["karc-full"]["USD"] / tot["rag-bm25"]["USD"],
                "karc_cheaper_pct": (1.0 - tot["karc-full"]["USD"] / tot["rag-bm25"]["USD"])
                * 100.0,
                "paired_ratio": paired_ratio(usd_k, usd_r, matrix12),
                "paired_mean_diff_usd": paired_bootstrap(
                    [a - b for a, b in zip(usd_k, usd_r)], matrix12
                ),
            },
            "ordering_reverses": tot["karc-full"]["USD"] < tot["rag-bm25"]["USD"],
            "flip_thresholds": flip_thresholds(d),
            "per_session": {
                "usd_karc": {s: comp["karc-full"]["USD"][s] for s in SESSIONS},
                "usd_rag": {s: comp["rag-bm25"]["USD"][s] for s in SESSIONS},
                "gross_karc": {s: comp["karc-full"]["G"][s] for s in SESSIONS},
                "gross_rag": {s: comp["rag-bm25"]["G"][s] for s in SESSIONS},
            },
        }

    full = build(xr_turns, "all positions 1-8 (E13 §4.2 baseline)")
    pos28_rows = [r for r in xr_turns if r["position"] >= 2]
    pos28 = build(pos28_rows, "positions 2-8 only (position 1 excluded)")

    # cross-check that the reproduced baseline still equals the committed E13 raw
    committed = json.loads((E13 / "raw" / "recomp.json").read_text(encoding="utf-8"))
    c4 = committed["item4_four_component_pricing"]
    baseline_agrees = (
        abs(full["priced_ordering"]["karc_usd"] - c4["priced_ordering_observed_rate_card"]["karc_usd"]) < 1e-9
        and abs(full["priced_ordering"]["rag_usd"] - c4["priced_ordering_observed_rate_card"]["rag_usd"]) < 1e-9
        and abs(
            full["priced_ordering"]["paired_ratio"]["ci95_lo"]
            - c4["priced_ordering_observed_rate_card"]["paired_ratio"]["ci95_lo"]
        )
        < 1e-12
        and abs(
            full["priced_ordering"]["paired_ratio"]["ci95_hi"]
            - c4["priced_ordering_observed_rate_card"]["paired_ratio"]["ci95_hi"]
        )
        < 1e-12
    )
    assert baseline_agrees, "NEEDS-DATA: baseline does not reproduce committed E13 §4.2"

    # position-1 share of each arm's creation, the reason item A exists at all
    pos1_rows = [r for r in xr_turns if r["position"] == 1]
    pos1_share = {}
    for arm in ARMS:
        arm_all = [r for r in xr_turns if r["arm"] == arm]
        arm_p1 = [r for r in pos1_rows if r["arm"] == arm]
        creation_all = sum(
            r["cache_creation_1h_input_tokens_provider"]
            + r["cache_creation_5m_input_tokens_provider"]
            for r in arm_all
        )
        creation_p1 = sum(
            r["cache_creation_1h_input_tokens_provider"]
            + r["cache_creation_5m_input_tokens_provider"]
            for r in arm_p1
        )
        read_all = sum(r["cache_read_input_tokens_provider"] for r in arm_all)
        read_p1 = sum(r["cache_read_input_tokens_provider"] for r in arm_p1)
        usd_all = sum(r["rate_card_equivalent_usd"] for r in arm_all)
        usd_p1 = sum(r["rate_card_equivalent_usd"] for r in arm_p1)
        pos1_share[arm] = {
            "creation_tokens_position_1": creation_p1,
            "creation_tokens_all": creation_all,
            "creation_share_pct": creation_p1 / creation_all * 100.0,
            "cache_read_tokens_position_1": read_p1,
            "cache_read_share_pct": read_p1 / read_all * 100.0,
            "usd_position_1": usd_p1,
            "usd_share_pct": usd_p1 / usd_all * 100.0,
            "mean_creation_per_turn_position_1": creation_p1 / len(arm_p1),
            "mean_creation_per_turn_positions_2_8": (creation_all - creation_p1)
            / (len(arm_all) - len(arm_p1)),
        }

    item_a = {
        "question": (
            "Does the two-accounting reversal survive excluding the defective "
            "position-1 turn, and how do the flip margins move?"
        ),
        "position_1_defect_reference": "E13 report §1.4 (arm- and runtime-independent workload defect)",
        "position_1_component_share": pos1_share,
        "baseline_all_positions": full,
        "positions_2_to_8": pos28,
        "reversal_survives_position_1_exclusion": pos28["ordering_reverses"],
        "margin_movement": {
            "mu_r_flip_mu_o5_baseline": full["flip_thresholds"]["mu_o=5.0"][
                "mu_r_flip_at_observed_mu_w_2_0"
            ],
            "mu_r_flip_mu_o5_pos28": pos28["flip_thresholds"]["mu_o=5.0"][
                "mu_r_flip_at_observed_mu_w_2_0"
            ],
            "mu_r_headroom_pct_baseline": full["flip_thresholds"]["mu_o=5.0"][
                "mu_r_headroom_pct"
            ],
            "mu_r_headroom_pct_pos28": pos28["flip_thresholds"]["mu_o=5.0"][
                "mu_r_headroom_pct"
            ],
            "mu_w_flip_mu_o5_baseline": full["flip_thresholds"]["mu_o=5.0"][
                "mu_w_flip_at_observed_mu_r_0_1"
            ],
            "mu_w_flip_mu_o5_pos28": pos28["flip_thresholds"]["mu_o=5.0"][
                "mu_w_flip_at_observed_mu_r_0_1"
            ],
            "mu_w_headroom_pct_baseline": full["flip_thresholds"]["mu_o=5.0"][
                "mu_w_headroom_pct"
            ],
            "mu_w_headroom_pct_pos28": pos28["flip_thresholds"]["mu_o=5.0"][
                "mu_w_headroom_pct"
            ],
            "priced_ratio_baseline": full["priced_ordering"]["karc_over_rag"],
            "priced_ratio_pos28": pos28["priced_ordering"]["karc_over_rag"],
        },
    }

    # ====================== ITEM B: leave-one-session-out ====================
    def loso(cell: dict, axis: str) -> dict:
        """axis in {'usd', 'gross'}; returns per-fold point ratios plus CIs."""
        key_k = "usd_karc" if axis == "usd" else "gross_karc"
        key_r = "usd_rag" if axis == "usd" else "gross_rag"
        whole_cell_ratio = (
            cell["priced_ordering"]["karc_over_rag"]
            if axis == "usd"
            else cell["gross_ordering"]["karc_over_rag"]
        )
        folds = []
        for dropped in SESSIONS:
            kept = [s for s in SESSIONS if s != dropped]
            num = [cell["per_session"][key_k][s] for s in kept]
            den = [cell["per_session"][key_r][s] for s in kept]
            ratio = paired_ratio(num, den, matrix11)
            diff = paired_bootstrap([a - b for a, b in zip(num, den)], matrix11)
            folds.append(
                {
                    "dropped_session": dropped,
                    "n_sessions": len(kept),
                    "karc_total": sum(num),
                    "rag_total": sum(den),
                    "karc_over_rag": ratio["point"],
                    "karc_cheaper": sum(num) < sum(den),
                    "paired_ratio_ci95": [ratio["ci95_lo"], ratio["ci95_hi"]],
                    "ratio_ci_entirely_below_one": ratio["entirely_below_one"],
                    "ratio_ci_entirely_above_one": ratio["entirely_above_one"],
                    "paired_mean_diff": diff["point"],
                    "diff_ci95": [diff["ci95_lo"], diff["ci95_hi"]],
                    "diff_ci_excludes_zero": diff["excludes_zero"],
                }
            )
        preserved = sum(1 for f in folds if f["karc_cheaper"])
        ci_preserved = sum(1 for f in folds if f["ratio_ci_entirely_below_one"])
        return {
            "folds": folds,
            "folds_total": len(folds),
            "folds_karc_cheaper": preserved,
            "folds_with_ratio_ci_entirely_below_one": ci_preserved,
            "whole_cell_ratio": whole_cell_ratio,
            "ratio_min": min(f["karc_over_rag"] for f in folds),
            "ratio_max": max(f["karc_over_rag"] for f in folds),
            "ratio_span": max(f["karc_over_rag"] for f in folds)
            - min(f["karc_over_rag"] for f in folds),
            "most_influential_session": max(
                folds,
                key=lambda f: abs(f["karc_over_rag"] - whole_cell_ratio),
            )["dropped_session"],
            "largest_single_session_ratio_shift": max(
                abs(f["karc_over_rag"] - whole_cell_ratio) for f in folds
            ),
        }

    item_b = {
        "question": (
            "The unit of pairing is the session (n=12). Does the priced ordering "
            "survive every leave-one-session-out fold?"
        ),
        "method": (
            "One shared 10,000 x 11 index matrix from the declared seed 1313 is "
            "reused by all 12 folds (common random numbers across folds); the "
            "construction is identical to E13's 12-wide matrix."
        ),
        "priced_all_positions": loso(full, "usd"),
        "gross_all_positions_control": loso(full, "gross"),
        "priced_positions_2_to_8": loso(pos28, "usd"),
        "gross_positions_2_to_8_control": loso(pos28, "gross"),
    }
    item_b["reversal_preserved_in_all_12_folds"] = (
        item_b["priced_all_positions"]["folds_karc_cheaper"] == 12
    )
    item_b["gross_ordering_robust_in_all_12_folds"] = (
        item_b["gross_all_positions_control"]["folds_karc_cheaper"] == 0
    )

    # ================= ITEM C: token-level perturbation bounds ==============
    def perturbations(cell: dict) -> dict:
        d = cell["component_deltas_karc_minus_rag"]
        tot = cell["component_totals"]
        margin_tokens = priced_margin_input_equivalent(d)  # negative = karc cheaper
        assert margin_tokens < 0, "NOT-COMPUTABLE: no reversal to perturb in this cell"
        need = -margin_tokens  # positive input-equivalent tokens to close the gap
        rows = []
        for arm, comp_key, direction in PERTURBATION_TARGETS:
            weight = MU_R if comp_key == "R" else MU_W_1H
            tokens = need / weight
            observed_component = (
                tot[arm]["R"]
                if comp_key == "R"
                else tot[arm]["W1h"] + tot[arm]["W5m"]
            )
            rows.append(
                {
                    "arm": arm,
                    "component": "cache_read" if comp_key == "R" else "cache_creation",
                    "direction": direction,
                    "weight_applied": weight,
                    "tokens_required": tokens,
                    "observed_component_total": observed_component,
                    "pct_of_arm_component_total": tokens / observed_component * 100.0,
                    "observed_arm_gross_input": tot[arm]["G"],
                    "pct_of_arm_gross_input": tokens / tot[arm]["G"] * 100.0,
                    "tokens_per_turn_equivalent": tokens / (cell["n_turns"] / 2),
                    "tokens_per_session_equivalent": tokens / 12.0,
                    "feasible_within_observed_component": tokens <= observed_component,
                }
            )
        return {
            "priced_margin_usd": cell["priced_ordering"]["karc_minus_rag_usd"],
            "priced_margin_input_equivalent_tokens": margin_tokens,
            "closing_requirement_input_equivalent_tokens": need,
            "perturbations": rows,
        }

    item_c = {
        "question": (
            "How many tokens of composition change, holding everything else "
            "fixed, would remove the reversal?"
        ),
        "declared_targets": [
            {"arm": a, "component": c, "direction": d}
            for a, c, d in PERTURBATION_TARGETS
        ],
        "all_positions": perturbations(full),
        "positions_2_to_8": perturbations(pos28),
    }

    # ================= ITEM D: 1-hour TTL provenance audit ==================
    hits = []
    files_scanned = 0
    for root in TTL_AUDIT_ROOTS:
        base = REPO / root
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.suffix not in TTL_AUDIT_SUFFIXES:
                continue
            if "__pycache__" in path.parts:
                continue
            files_scanned += 1
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for pattern in TTL_SETTING_PATTERNS:
                for match in re.finditer(pattern, text):
                    line = text.count("\n", 0, match.start()) + 1
                    hits.append(
                        {
                            "path": str(path.relative_to(REPO)),
                            "pattern": pattern,
                            "line": line,
                        }
                    )

    # Where does 1h appear in our own source at all?  Classify each occurrence.
    provenance_sites = []
    for rel in (
        "src/karc/bench/e5_cross_runtime.py",
        "scripts/run_e10_xr_canary.py",
    ):
        text = (REPO / rel).read_text(encoding="utf-8")
        for match in re.finditer(r"[A-Za-z_0-9\.]*1h[A-Za-z_0-9]*", text):
            provenance_sites.append(
                {
                    "path": rel,
                    "line": text.count("\n", 0, match.start()) + 1,
                    "symbol": match.group(0),
                }
            )

    ttl_5m_total = sum(r["cache_creation_5m_input_tokens_provider"] for r in xr_turns)
    ttl_1h_total = sum(r["cache_creation_1h_input_tokens_provider"] for r in xr_turns)

    # Cross-experiment invariance check: if 1h were a per-experiment harness
    # choice, cells run under different CLI builds could differ.  Scan every
    # committed Claude cell for its CLI version stamp and its TTL split.
    cli_pattern = re.compile(r"2\.1\.\d+")
    ttl_by_cell = []
    for cell_dir in sorted((REPO / "docs" / "experiments").iterdir()):
        if not cell_dir.is_dir():
            continue
        versions: set[str] = set()
        c1 = c5 = 0
        rows_seen = 0
        for path in sorted(cell_dir.rglob("*")):
            if not path.is_file() or path.suffix not in (".jsonl", ".json", ".yaml", ".md"):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            versions.update(cli_pattern.findall(text))
            if path.suffix != ".jsonl" or "cache_creation" not in text:
                continue
            for line in text.splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                usage = row.get("api_usage_raw")
                if isinstance(usage, dict) and "cache_creation" in usage:
                    detail = usage["cache_creation"]
                    c1 += detail.get("ephemeral_1h_input_tokens", 0)
                    c5 += detail.get("ephemeral_5m_input_tokens", 0)
                    rows_seen += 1
                elif "cache_creation_1h_input_tokens_provider" in row:
                    c1 += row["cache_creation_1h_input_tokens_provider"]
                    c5 += row["cache_creation_5m_input_tokens_provider"]
                    rows_seen += 1
        if rows_seen:
            ttl_by_cell.append(
                {
                    "cell": cell_dir.name,
                    "claude_cli_versions_stamped": sorted(versions),
                    "rows_with_creation_detail": rows_seen,
                    "cache_creation_1h_tokens": c1,
                    "cache_creation_5m_tokens": c5,
                    "pct_1h": (c1 / (c1 + c5) * 100.0) if (c1 + c5) else None,
                }
            )
    all_cells_100pct_1h = all(
        cell["cache_creation_5m_tokens"] == 0 for cell in ttl_by_cell
    )
    distinct_cli_versions = sorted(
        {v for cell in ttl_by_cell for v in cell["claude_cli_versions_stamped"]}
    )

    item_d = {
        "question": (
            "Was the 1-hour cache TTL our harness configuration choice or the "
            "runtime/provider default?"
        ),
        "scope": "committed repository sources only; no network access was used",
        "audit": {
            "roots_scanned": list(TTL_AUDIT_ROOTS),
            "files_scanned": files_scanned,
            "patterns": list(TTL_SETTING_PATTERNS),
            "ttl_setting_hits": hits,
            "n_ttl_setting_hits": len(hits),
        },
        "occurrences_of_1h_in_harness_source": provenance_sites,
        "observed_ttl_split_in_raw": {
            "cache_creation_1h_tokens": ttl_1h_total,
            "cache_creation_5m_tokens": ttl_5m_total,
            "pct_1h": ttl_1h_total / (ttl_1h_total + ttl_5m_total) * 100.0,
        },
        "cross_experiment_invariance": {
            "per_cell": ttl_by_cell,
            "distinct_claude_cli_versions_on_file": distinct_cli_versions,
            "all_committed_claude_cells_are_100pct_1h": all_cells_100pct_1h,
            "reading": (
                "Every committed Claude cell writes 100% 1-hour-TTL entries "
                "across every CLI build on file, with no TTL setting in any "
                "harness file. A per-experiment configuration choice would not "
                "be expected to hold invariantly across independently written "
                "runners and CLI builds."
            ),
        },
        "harness_reads_provider_reported_bucket": {
            "function": "karc.bench.e5_cross_runtime.normalize_claude_usage",
            "provider_fields_read": [
                "cache_creation.ephemeral_1h_input_tokens",
                "cache_creation.ephemeral_5m_input_tokens",
            ],
            "note": (
                "The harness only reads the split the provider reports; it never "
                "requests a TTL."
            ),
        },
        "harness_declares_prices_not_ttl": {
            "site": "scripts/run_e10_xr_canary.py:64-66 (xr.ClaudeRates)",
            "declared": {
                "fresh_usd_per_mtok": 3.0,
                "cache_creation_5m_usd_per_mtok": 3.75,
                "cache_creation_1h_usd_per_mtok": 6.0,
                "cache_read_usd_per_mtok": 0.30,
                "output_usd_per_mtok": 15.0,
            },
            "note": (
                "These are prices attached to whichever bucket the provider "
                "reports, not a request-side TTL selection."
            ),
        },
        "verdict": (
            "NOT-OUR-HARNESS-CONFIGURATION (established); "
            "PROVIDER-DEFAULT: NOT-COMPUTABLE from committed sources"
        ),
        "verdict_detail": (
            "No committed harness file sets a cache TTL: 0 request-side "
            "cache_control / ttl / cache-env settings across the scanned roots. "
            "The 1-hour bucket is only ever read back from the provider's own "
            "usage payload, and the request that selected it was constructed "
            "inside Claude Code CLI 2.1.220, which is not vendored here. "
            "Whether the CLI treats 1h as its default, derives it from the "
            "subscription tier, or negotiates it per request cannot be decided "
            "from this repository."
        ),
        "file_that_would_settle_it": (
            "A recorded request body from Claude Code CLI 2.1.220 showing the "
            "cache_control block it sends (e.g. an ANTHROPIC_BASE_URL proxy "
            "capture, or the CLI's own version-pinned source). No such file "
            "exists in this repository, and R-9 privacy forbids storing request "
            "payloads, so producing it requires an explicit new instrumented run "
            "under an amended privacy scope."
        ),
        "corroborating_but_out_of_repo": (
            "The bundled claude-api reference states the API-level default is the "
            "5-minute ephemeral cache and that 1h requires an explicit "
            "cache_control.ttl, which is consistent with the runtime (not the "
            "provider) having selected 1h. That reference is not a committed "
            "repository source and is therefore recorded separately from the "
            "verdict above."
        ),
    }

    # ============ ITEM E: other committed four-component cells ==============
    def scan_four_component_candidates() -> list[dict]:
        found = []
        exp_root = REPO / "docs" / "experiments"
        for path in sorted(exp_root.rglob("*.jsonl")):
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            if "cache_creation" not in text:
                continue
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
            if not rows:
                continue
            per_turn = "cache_creation_1h_input_tokens_provider" in rows[0]
            nested = isinstance(rows[0].get("api_usage_raw"), dict) and (
                "cache_creation" in rows[0]["api_usage_raw"]
            )
            if not (per_turn or nested):
                continue
            # str() the keys: some rows carry a null model/arm and the JSON
            # writer sorts keys, which cannot order str against None.
            models = Counter(str(r.get("reported_model")) for r in rows)
            arms = Counter(str(r.get("arm")) for r in rows)
            if per_turn:
                c1 = sum(r["cache_creation_1h_input_tokens_provider"] for r in rows)
                c5 = sum(r["cache_creation_5m_input_tokens_provider"] for r in rows)
            else:
                c1 = sum(
                    (r.get("api_usage_raw") or {})
                    .get("cache_creation", {})
                    .get("ephemeral_1h_input_tokens", 0)
                    for r in rows
                )
                c5 = sum(
                    (r.get("api_usage_raw") or {})
                    .get("cache_creation", {})
                    .get("ephemeral_5m_input_tokens", 0)
                    for r in rows
                )
            found.append(
                {
                    "path": str(path.relative_to(REPO)),
                    "rows": len(rows),
                    "creation_split_granularity": "per-turn" if per_turn else "per-run (nested api_usage_raw)",
                    "reported_models": dict(models),
                    "arms": dict(arms),
                    "has_session_position_pairing": "session_id" in rows[0]
                    and "position" in rows[0],
                    "cache_creation_1h_tokens": c1,
                    "cache_creation_5m_tokens": c5,
                    "four_component_accounting_computable": (c1 + c5) > 0,
                }
            )
        return found

    candidates = scan_four_component_candidates()

    # The XR smoke is the only other cell with the same two arms and a per-turn
    # creation split.  Its eligibility rule is taken verbatim from the committed
    # validation record, not invented here.
    smoke_rows = E13M.read_jsonl(XR_SMOKE / "raw" / "turns.jsonl")
    smoke_validation = json.loads(
        (XR_SMOKE / "raw" / "validation.json").read_text(encoding="utf-8")
    )
    smoke_summary = json.loads(
        (XR_SMOKE / "raw" / "summary.json").read_text(encoding="utf-8")
    )

    def smoke_eligible(row: dict) -> bool:
        if row["arm"] == "karc-full":
            return row.get("execution_epoch") == "tool-exposure-v4"
        return True

    smoke_sel = [r for r in smoke_rows if smoke_eligible(r)]
    assert len(smoke_sel) == smoke_validation["eligible_turn_rows"], (
        "NEEDS-DATA: reconstructed smoke eligibility does not match the "
        f"committed count {smoke_validation['eligible_turn_rows']}"
    )
    smoke_sessions = sorted({r["session_id"] for r in smoke_sel})
    smoke_comp = component_table(smoke_sel, smoke_sessions)
    smoke_tot = totals_of(smoke_comp, smoke_sessions)
    for arm in ARMS:
        ref = smoke_summary["by_arm"][arm]
        assert abs(smoke_tot[arm]["USD"] - ref["rate_card_equivalent_usd"]) < 1e-9
        assert smoke_tot[arm]["G"] == ref["gross_input_tokens"]
        assert (
            smoke_tot[arm]["W1h"] + smoke_tot[arm]["W5m"]
            == ref["cache_creation_input_tokens_provider"]
        )
    smoke_d = deltas_of(smoke_tot)

    item_e = {
        "question": (
            "Is there any second committed four-component data point that could "
            "independently check the reversal?"
        ),
        "committed_candidates": candidates,
        "second_cell_with_the_same_two_arms": {
            "cell": "E10-P2-XR smoke",
            "path": "docs/experiments/E10-P2-XR/smoke/raw/turns.jsonl",
            "size": {
                "persisted_turn_rows": len(smoke_rows),
                "eligible_turn_rows": len(smoke_sel),
                "paired_sessions": smoke_summary["complete_paired_sessions"],
                "turns_per_session": 8,
                "eligible_turns_per_arm": 32,
            },
            "eligibility_rule_source": (
                "docs/experiments/E10-P2-XR/smoke/raw/validation.json "
                "eligible_composition (tool_exposure_v4_karc 32, "
                "legacy_rag_s01_s02 16, tool_exposure_v4_rag_s03_s04 16); "
                "9 legacy karc rows are recorded invalid"
            ),
            "reconstruction_cross_check": "matches summary.json by_arm exactly",
            "component_totals": smoke_tot,
            "component_deltas_karc_minus_rag": smoke_d,
            "gross_karc_over_rag": smoke_tot["karc-full"]["G"] / smoke_tot["rag-bm25"]["G"],
            "priced_karc_usd": smoke_tot["karc-full"]["USD"],
            "priced_rag_usd": smoke_tot["rag-bm25"]["USD"],
            "priced_karc_over_rag": smoke_tot["karc-full"]["USD"]
            / smoke_tot["rag-bm25"]["USD"],
            "priced_karc_cheaper_pct": (
                1.0 - smoke_tot["karc-full"]["USD"] / smoke_tot["rag-bm25"]["USD"]
            )
            * 100.0,
            "ordering_reverses": smoke_tot["karc-full"]["USD"]
            < smoke_tot["rag-bm25"]["USD"],
            "flip_thresholds": flip_thresholds(smoke_d),
            "ttl_split": {
                "cache_creation_1h_tokens": smoke_tot["karc-full"]["W1h"]
                + smoke_tot["rag-bm25"]["W1h"],
                "cache_creation_5m_tokens": smoke_tot["karc-full"]["W5m"]
                + smoke_tot["rag-bm25"]["W5m"],
            },
            "independence_limits": [
                "same experiment family, fixture, runtime, model and rate card as the canary",
                "4 paired sessions, not 12; no paired CI is reported here",
                "the rag arm mixes two execution epochs (S01-S02 legacy, S03-S04 tool-exposure-v4)",
                "karc S01-S02 are retry attempts; 9 legacy karc rows are excluded as invalid",
                "the canary approval records smoke_rows_included=false, so this is a separate cell, never pooled",
            ],
        },
        "verdict": (
            "One other four-component cell with the same two arms exists on file "
            "(the XR smoke, 4 paired sessions x 8 turns). It is a smaller, "
            "earlier, epoch-mixed execution of the same workload on the same "
            "runtime, not an independent replication."
        ),
    }

    payload = {
        "experiment_id": "E13-RECOMP",
        "analysis": "fragility of the condition-2 cost-ordering reversal",
        "model_calls": 0,
        "new_experiments": 0,
        "new_fixtures": 0,
        "network_access": False,
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "resamples": BOOTSTRAP_RESAMPLES,
            "unit": "session",
            "ci": "two-sided 95% percentile",
            "common_random_numbers": True,
            "matrix_widths_used": [12, 11],
            "note": (
                "the width-12 matrix is byte-identical to E13's, so item A is "
                "directly comparable with report §4.2; the width-11 matrix is "
                "shared by all leave-one-out folds"
            ),
        },
        "input_sha256": inputs,
        "rate_card_recovery": rate_card,
        "baseline_reproduces_committed_e13": baseline_agrees,
        "item_a_position_1_exclusion": item_a,
        "item_b_leave_one_session_out": item_b,
        "item_c_token_perturbation_bounds": item_c,
        "item_d_one_hour_ttl_provenance": item_d,
        "item_e_second_four_component_cell": item_e,
    }

    RAW_OUT.mkdir(parents=True, exist_ok=True)
    (RAW_OUT / "fragility.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("A reversal survives position-1 exclusion:", pos28["ordering_reverses"])
    print(
        "  priced ratio 1-8 %.4f -> 2-8 %.4f; mu_w headroom %.2f%% -> %.2f%%"
        % (
            full["priced_ordering"]["karc_over_rag"],
            pos28["priced_ordering"]["karc_over_rag"],
            full["flip_thresholds"]["mu_o=5.0"]["mu_w_headroom_pct"],
            pos28["flip_thresholds"]["mu_o=5.0"]["mu_w_headroom_pct"],
        )
    )
    print(
        "B priced folds karc-cheaper: %d/12 (CI below 1.0 in %d/12); gross control karc-cheaper %d/12"
        % (
            item_b["priced_all_positions"]["folds_karc_cheaper"],
            item_b["priced_all_positions"]["folds_with_ratio_ci_entirely_below_one"],
            item_b["gross_all_positions_control"]["folds_karc_cheaper"],
        )
    )
    for row in item_c["all_positions"]["perturbations"]:
        print(
            "C %s %s %s %.0f tokens = %.2f%% of its own component, %.2f%% of arm gross"
            % (
                row["arm"],
                row["direction"],
                row["component"],
                row["tokens_required"],
                row["pct_of_arm_component_total"],
                row["pct_of_arm_gross_input"],
            )
        )
    print("D", item_d["verdict"])
    print(
        "E second cell reverses: %s (karc $%.4f vs rag $%.4f, %d eligible turns)"
        % (
            item_e["second_cell_with_the_same_two_arms"]["ordering_reverses"],
            smoke_tot["karc-full"]["USD"],
            smoke_tot["rag-bm25"]["USD"],
            len(smoke_sel),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
