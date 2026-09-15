#!/usr/bin/env python3
"""E13-RECOMP: zero-cost recomputation over committed E10/E11 raw artifacts.

No model, network, embedding, fixture-generation, or benchmark execution.  Every
input is a file already committed to the repository.  Output is deterministic:
the bootstrap draws come from one explicitly seeded index matrix that is shared
by every paired comparison (common random numbers), so re-running the script
reproduces byte-identical JSON.

Usage:
    PYTHONPATH=$PWD/src .venv/bin/python \
        docs/experiments/E13-RECOMP/scripts/e13_recomp.py

Writes docs/experiments/E13-RECOMP/raw/*.json.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
E13 = HERE.parent
REPO = E13.parents[2]
RAW_OUT = E13 / "raw"

FIXTURE_TASKS = REPO / "fixture" / "e4-v2" / "tasks.json"
BASE3 = REPO / "docs" / "experiments" / "E11-BASE3"
H16 = REPO / "docs" / "experiments" / "E10-P2-H16"
XR = REPO / "docs" / "experiments" / "E10-P2-XR" / "canary"

# --- Preregistered / declared analysis constants -------------------------------
BOOTSTRAP_SEED = 1313
BOOTSTRAP_RESAMPLES = 10_000
CI_ALPHA = 0.05  # two-sided 95%

# Observed Anthropic Sonnet-5 rate card, recovered exactly from the committed
# per-turn `rate_card_equivalent_usd` field (see verify_rate_card()).  Units are
# USD per token; the *_MULT values are the multiples of the uncached input rate.
P_IN = 3.0e-6
P_CREATION_1H = 6.0e-6
P_CREATION_5M = 3.75e-6
P_READ = 0.30e-6
P_OUT = 15.0e-6
MU_CREATION_1H = P_CREATION_1H / P_IN  # 2.0
MU_CREATION_5M = P_CREATION_5M / P_IN  # 1.25
MU_READ = P_READ / P_IN  # 0.1
MU_OUT = P_OUT / P_IN  # 5.0

# Declared sweep grid for item 4.  Fixed before looking at the flip location.
GRID_MU_W = [0.0, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5]
GRID_MU_R = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1.0]
GRID_MU_O = [0.0, 5.0]

# Published cache-read carry weights found in the committed record only.  Each
# entry is (label, source_path, source_location, weight).  No web access.
PUBLISHED_READ_WEIGHTS = [
    (
        "Wolff & Bennati, IEEE COMPSAC 2026 (arXiv:2601.07978)",
        "docs/analysis/paper2-refs-pricing-and-regime.md",
        "table row: LLM input $0.15/M vs LLM cached input $0.075/M",
        0.075 / 0.15,
    ),
    (
        "Weinberger & Hozez, arXiv:2607.12161 v4",
        "docs/analysis/paper2-refs-pricing-and-regime.md",
        "table row: mu_w = 1.25, mu_r = 0.1",
        0.1,
    ),
    (
        "The Harness Effect, arXiv:2607.06906 v1",
        "docs/analysis/paper2-citations-and-venues.md",
        "table row: effective input price p_in^eff = p_in(1 - h(1 - kappa)), kappa ~ 0.1",
        0.1,
    ),
    (
        "Beyond the Context Window, arXiv:2603.04814",
        "docs/analysis/paper2-citations-and-venues.md",
        "table row: cached tokens discounted 90%",
        0.1,
    ),
    (
        "Anthropic rate card recovered from E10-P2-XR canary raw",
        "docs/experiments/E10-P2-XR/canary/raw/turns.jsonl",
        "rate_card_equivalent_usd solved exactly: $0.30/M read vs $3.00/M input",
        0.1,
    ),
]
PUBLISHED_CREATION_WEIGHTS = [
    (
        "Weinberger & Hozez, arXiv:2607.12161 v4 (5-minute TTL)",
        "docs/analysis/paper2-refs-pricing-and-regime.md",
        "table row: mu_w = 1.25",
        1.25,
    ),
    (
        "Anthropic rate card recovered from E10-P2-XR canary raw (1-hour TTL)",
        "docs/experiments/E10-P2-XR/canary/raw/turns.jsonl",
        "rate_card_equivalent_usd solved exactly: $6.00/M 1h creation vs $3.00/M input",
        2.0,
    ),
]

SESSIONS = [f"S{i:02d}" for i in range(1, 13)]


# --- helpers ------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def resample_matrix() -> list[list[int]]:
    """One shared 10,000 x 12 index matrix. Common random numbers everywhere."""
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(SESSIONS)
    return [[rng.randrange(n) for _ in range(n)] for _ in range(BOOTSTRAP_RESAMPLES)]


def percentile_ci(values: list[float]) -> tuple[float, float]:
    """Percentile CI, declared convention: floor(alpha/2*B) and ceil((1-alpha/2)*B)-1."""
    ordered = sorted(values)
    b = len(ordered)
    lo_i = int((CI_ALPHA / 2) * b)
    hi_i = int(-(-(1 - CI_ALPHA / 2) * b // 1)) - 1
    return ordered[lo_i], ordered[hi_i]


def exact_sign_test_p(pos: int, neg: int) -> float | None:
    """Two-sided exact sign test on discordant pairs only (ties dropped)."""
    n = pos + neg
    if n == 0:
        return None
    from math import comb

    k = min(pos, neg)
    tail = sum(comb(n, i) for i in range(0, k + 1))
    return min(1.0, 2.0 * tail / (2**n))


def paired_bootstrap(
    per_session: dict[str, float], matrix: list[list[int]]
) -> dict[str, float]:
    """Percentile bootstrap of the mean of a per-session paired quantity."""
    values = [per_session[s] for s in SESSIONS]
    point = sum(values) / len(values)
    draws = [sum(values[i] for i in row) / len(row) for row in matrix]
    lo, hi = percentile_ci(draws)
    return {
        "point": point,
        "ci95_lo": lo,
        "ci95_hi": hi,
        "excludes_zero": (lo > 0.0) or (hi < 0.0),
    }


def paired_bootstrap_ratio_of_sums(
    numer: dict[str, float], denom: dict[str, float], matrix: list[list[int]]
) -> dict[str, float] | dict[str, object]:
    """Percentile bootstrap of sum(numer)/sum(denom) over resampled sessions."""
    num = [numer[s] for s in SESSIONS]
    den = [denom[s] for s in SESSIONS]
    total_den = sum(den)
    if total_den == 0:
        return {"point": None, "ci95_lo": None, "ci95_hi": None, "degenerate": True}
    point = sum(num) / total_den
    draws = []
    degenerate = 0
    for row in matrix:
        d = sum(den[i] for i in row)
        if d == 0:
            degenerate += 1
            continue
        draws.append(sum(num[i] for i in row) / d)
    lo, hi = percentile_ci(draws)
    return {
        "point": point,
        "ci95_lo": lo,
        "ci95_hi": hi,
        "degenerate_resamples": degenerate,
        "usable_resamples": len(draws),
    }


# --- Item 0: task-type join ----------------------------------------------------
def build_task_type_map() -> dict[tuple[str, ...], str]:
    tasks = json.loads(FIXTURE_TASKS.read_text(encoding="utf-8"))
    grouped: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for task in tasks:
        grouped[tuple(task["required_versions"])].add(task["task_type"])
    ambiguous = {k: sorted(v) for k, v in grouped.items() if len(v) > 1}
    assert not ambiguous, f"required_versions -> task_type is not a function: {ambiguous}"
    return {k: next(iter(v)) for k, v in grouped.items()}


def join_schedule(schedule_path: Path, type_map: dict) -> dict[tuple[str, int], str]:
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    out: dict[tuple[str, int], str] = {}
    unmatched = []
    for session in schedule["sessions"]:
        for task in session["tasks"]:
            key = tuple(task["required_versions"])
            if key not in type_map:
                unmatched.append((session["session_id"], task["position"]))
                continue
            out[(session["session_id"], task["position"])] = type_map[key]
    assert not unmatched, f"NEEDS-DATA: unmatched schedule turns {unmatched}"
    return out


# --- Item 4 sanity: recover the rate card exactly ------------------------------
def verify_rate_card(xr_turns: list[dict]) -> dict:
    residuals = []
    for row in xr_turns:
        modelled = (
            row["fresh_input_tokens_provider"] * P_IN
            + row["cache_creation_1h_input_tokens_provider"] * P_CREATION_1H
            + row["cache_creation_5m_input_tokens_provider"] * P_CREATION_5M
            + row["cache_read_input_tokens_provider"] * P_READ
            + row["output_tokens"] * P_OUT
        )
        residuals.append(abs(modelled - row["rate_card_equivalent_usd"]))
    return {
        "n_turns": len(xr_turns),
        "max_abs_residual_usd": max(residuals),
        "exact_within_1e_9_usd": max(residuals) < 1e-9,
        "recovered_rate_card_usd_per_million": {
            "uncached_input": P_IN * 1e6,
            "cache_creation_1h": P_CREATION_1H * 1e6,
            "cache_creation_5m": P_CREATION_5M * 1e6,
            "cache_read": P_READ * 1e6,
            "output": P_OUT * 1e6,
        },
        "multiples_of_uncached_input": {
            "mu_creation_1h": MU_CREATION_1H,
            "mu_creation_5m": MU_CREATION_5M,
            "mu_read": MU_READ,
            "mu_output": MU_OUT,
        },
    }


# --- main ---------------------------------------------------------------------
def main() -> int:
    matrix = resample_matrix()
    type_map = build_task_type_map()

    base3_sched = BASE3 / "raw" / "schedule.json"
    h16_sched = H16 / "raw" / "schedule.json"
    xr_sched = XR / "raw" / "schedule.json"

    inputs = {
        "fixture/e4-v2/tasks.json": sha256_file(FIXTURE_TASKS),
        "docs/experiments/E11-BASE3/raw/schedule.json": sha256_file(base3_sched),
        "docs/experiments/E10-P2-H16/raw/schedule.json": sha256_file(h16_sched),
        "docs/experiments/E10-P2-XR/canary/raw/schedule.json": sha256_file(xr_sched),
        "docs/experiments/E11-BASE3/run/raw/turns.jsonl": sha256_file(
            BASE3 / "run" / "raw" / "turns.jsonl"
        ),
        "docs/experiments/E10-P2-H16/run/raw/turns.jsonl": sha256_file(
            H16 / "run" / "raw" / "turns.jsonl"
        ),
        "docs/experiments/E10-P2-XR/canary/raw/turns.jsonl": sha256_file(
            XR / "raw" / "turns.jsonl"
        ),
        "docs/experiments/E10-P2-XR/canary/raw/attempts.jsonl": sha256_file(
            XR / "raw" / "attempts.jsonl"
        ),
    }

    # ---- schedule identity gate -------------------------------------------
    same_schedule = (
        inputs["docs/experiments/E11-BASE3/raw/schedule.json"]
        == inputs["docs/experiments/E10-P2-H16/raw/schedule.json"]
    )
    schedule_gate = {
        "base3_schedule_sha256": inputs["docs/experiments/E11-BASE3/raw/schedule.json"],
        "h16_schedule_sha256": inputs["docs/experiments/E10-P2-H16/raw/schedule.json"],
        "xr_schedule_sha256": inputs[
            "docs/experiments/E10-P2-XR/canary/raw/schedule.json"
        ],
        "base3_equals_h16_bytewise": same_schedule,
        "turns_per_session": {
            "base3": sorted(
                {
                    len(s["tasks"])
                    for s in json.loads(base3_sched.read_text())["sessions"]
                }
            ),
            "h16": sorted(
                {len(s["tasks"]) for s in json.loads(h16_sched.read_text())["sessions"]}
            ),
            "xr": sorted(
                {len(s["tasks"]) for s in json.loads(xr_sched.read_text())["sessions"]}
            ),
        },
        "five_arms_share_one_task_type_assignment": same_schedule,
        "xr_shares_that_assignment": inputs[
            "docs/experiments/E10-P2-XR/canary/raw/schedule.json"
        ]
        == inputs["docs/experiments/E11-BASE3/raw/schedule.json"],
    }
    assert same_schedule, "NEEDS-DATA: BASE3 and H16 schedules differ"

    h16_join = join_schedule(base3_sched, type_map)
    xr_join = join_schedule(xr_sched, type_map)
    assert len(h16_join) == 192, len(h16_join)
    assert len(xr_join) == 96, len(xr_join)
    join_report = {
        "distinct_required_versions_keys_in_fixture": len(type_map),
        "fixture_tasks": len(json.loads(FIXTURE_TASKS.read_text(encoding="utf-8"))),
        "ambiguous_required_versions_keys": 0,
        "h16_cell_turns_joined": len(h16_join),
        "h16_cell_task_type_counts": dict(sorted(Counter(h16_join.values()).items())),
        "xr_cell_turns_joined": len(xr_join),
        "xr_cell_task_type_counts": dict(sorted(Counter(xr_join.values()).items())),
        "join_key": "schedule.sessions[].tasks[].required_versions -> fixture task_type",
        "note": (
            "schedule task_id values (E5-Sxx-Tyy) are synthetic and do not appear in "
            "fixture/e4-v2/tasks.json, so required_versions is the only join key"
        ),
    }

    # ---- load turns -------------------------------------------------------
    base3_turns = read_jsonl(BASE3 / "run" / "raw" / "turns.jsonl")
    h16_turns = read_jsonl(H16 / "run" / "raw" / "turns.jsonl")
    xr_turns = read_jsonl(XR / "raw" / "turns.jsonl")
    xr_attempts = read_jsonl(XR / "raw" / "attempts.jsonl")

    h16cell = base3_turns + h16_turns  # 5 arms on the identical 12x16 schedule
    assert len(h16cell) == 960, len(h16cell)
    arms5_decl = [
        "sliding-window-compaction",
        "karc-full",
        "rag-bm25",
        "full-history",
        "stateless-rag",
    ]
    assert set(arms5_decl) == {r["arm"] for r in h16cell}

    # ================= ITEM 1: task-type stratified accuracy ================
    def stratify(turns: list[dict], join: dict) -> dict:
        table: dict[str, dict[str, list[int]]] = defaultdict(
            lambda: defaultdict(lambda: [0, 0])
        )
        for row in turns:
            ttype = join[(row["session_id"], row["position"])]
            cell = table[row["arm"]][ttype]
            cell[1] += 1
            cell[0] += 1 if row["passed"] else 0
        out = {}
        for arm in sorted(table):
            overall_k = sum(v[0] for v in table[arm].values())
            overall_n = sum(v[1] for v in table[arm].values())
            out[arm] = {
                "overall": {
                    "k": overall_k,
                    "n": overall_n,
                    "pct": 100.0 * overall_k / overall_n,
                },
                "by_task_type": {
                    t: {
                        "k": table[arm][t][0],
                        "n": table[arm][t][1],
                        "pct": 100.0 * table[arm][t][0] / table[arm][t][1],
                    }
                    for t in sorted(table[arm])
                },
            }
        return out

    # --- supplement: position stratification and the prime-turn floor -------
    # Discovered while re-deriving the join: nearly every failure in every arm
    # sits at position 1.  Quantify it rather than leave it implicit, because it
    # decides how much information the task-type table can carry.
    fixture_tasks = json.loads(FIXTURE_TASKS.read_text(encoding="utf-8"))
    struct_map: dict[tuple[str, ...], str] = {}
    for task in fixture_tasks:
        struct_map.setdefault(tuple(task["required_versions"]), task["structure"])

    def position_split(turns: list[dict], arms: list[str], first_only: int = 1) -> dict:
        out = {}
        for arm in arms:
            sub = [r for r in turns if r["arm"] == arm]
            p1 = [r for r in sub if r["position"] <= first_only]
            rest = [r for r in sub if r["position"] > first_only]
            out[arm] = {
                "position_1": {
                    "k": sum(1 for r in p1 if r["passed"]),
                    "n": len(p1),
                },
                "position_ge_2": {
                    "k": sum(1 for r in rest if r["passed"]),
                    "n": len(rest),
                    "pct": 100.0 * sum(1 for r in rest if r["passed"]) / len(rest),
                },
                "failures_total": sum(1 for r in sub if not r["passed"]),
                "failures_at_position_1": sum(
                    1 for r in sub if not r["passed"] and r["position"] == 1
                ),
            }
        return out

    # per-item failure overlap across the five arms
    fail_sets = {
        arm: {
            (r["session_id"], r["position"])
            for r in h16cell
            if r["arm"] == arm and not r["passed"]
        }
        for arm in arms5_decl
    }
    union = set().union(*fail_sets.values())
    shared_by_all = set.intersection(*fail_sets.values())
    discriminating = sorted(union - shared_by_all)

    prime_structures = {
        s: struct_map[
            tuple(
                next(
                    t
                    for t in json.loads(base3_sched.read_text())["sessions"]
                    if t["session_id"] == s
                )["tasks"][0]["required_versions"]
            )
        ]
        for s in SESSIONS
    }
    # does each position-1 retrieval plan also carry another version of the gold
    # artifact?  That is the leakage that makes the "current value" answer wrong.
    leak = {}
    for session in json.loads(base3_sched.read_text())["sessions"]:
        task = session["tasks"][0]
        gold = task["required_versions"][0]
        base = gold.rsplit("-v", 1)[0]
        others = [
            a
            for a in task["rag_artifact_ids"]
            if a.startswith(base + "-v") and a != gold
        ]
        leak[session["session_id"]] = {
            "gold_version": gold,
            "other_versions_of_same_artifact_in_plan": others,
        }

    item1_position = {
        "h16_cell": position_split(h16cell, arms5_decl),
        "xr_cell": position_split(xr_turns, ["karc-full", "rag-bm25"]),
        "union_of_failed_items_across_5_arms": len(union),
        "items_failed_by_all_5_arms": len(shared_by_all),
        "items_failed_by_all_5_arms_all_at_position_1": all(
            p == 1 for _, p in shared_by_all
        ),
        "discriminating_items": [
            {
                "session": s,
                "position": p,
                "task_type": h16_join[(s, p)],
                "arms_failing": sorted(a for a in arms5_decl if (s, p) in fail_sets[a]),
            }
            for s, p in discriminating
        ],
        "n_discriminating_items": len(discriminating),
        "position_1_task_structures": prime_structures,
        "position_1_task_types": dict(
            sorted(Counter(h16_join[(s, 1)] for s in SESSIONS).items())
        ),
        "position_1_retrieval_plan_leakage": leak,
        "position_1_plans_carrying_a_newer_version_of_the_gold_artifact": sum(
            1 for v in leak.values() if v["other_versions_of_same_artifact_in_plan"]
        ),
        "position_1_value_match_by_arm": {
            arm: dict(
                sorted(
                    Counter(
                        r["value_match"]
                        for r in h16cell
                        if r["arm"] == arm and r["position"] == 1
                    ).items()
                )
            )
            for arm in arms5_decl
        },
        "position_1_format_ok_by_arm": {
            arm: sum(
                1
                for r in h16cell
                if r["arm"] == arm and r["position"] == 1 and r["format_ok"]
            )
            for arm in arms5_decl
        },
        "diagnosis": (
            "Every position-1 task is a `*-prime` turn whose gold answer is the OLDER "
            "version, while the position-1 retrieval plan already contains a NEWER "
            "version of the same artifact (12/12 sessions). src/karc/bench/"
            "e5_cache_canary.py:task_prompt injects every rag_plan artifact body "
            "unfiltered, and the prompt asks for `the current value`, so answering the "
            "newer version is graded value_match=other (grade_answer only labels "
            "`stale` against task['stale_values'], which is empty for prime turns). "
            "This floor is arm-independent and runtime-independent."
        ),
    }

    item1 = {
        "h16_cell_5_arms": stratify(h16cell, h16_join),
        "xr_cell_2_arms_separate_schedule": stratify(xr_turns, xr_join),
        "position_stratification_supplement": item1_position,
        "stale_trap_types": ["corrected", "plain-stale"],
        "history_rewarding_types": ["rehabilitation"],
        "grouping_note": (
            "BASE3 (full-history, sliding-window-compaction, stateless-rag) and "
            "E10-P2-H16 (karc-full, rag-bm25) run the byte-identical 12x16 schedule, "
            "so the five arms share one task-type assignment. The E10-P2-XR canary is a "
            "different 12x8 schedule (different sha256, seed 4200) and is reported "
            "separately; it is never pooled with the five arms."
        ),
    }

    # ================= ITEM 2: paired accuracy uncertainty ==================
    def per_session_acc(turns: list[dict], arm: str) -> dict[str, float]:
        agg: dict[str, list[int]] = {s: [0, 0] for s in SESSIONS}
        for row in turns:
            if row["arm"] != arm:
                continue
            agg[row["session_id"]][1] += 1
            agg[row["session_id"]][0] += 1 if row["passed"] else 0
        return {s: 100.0 * agg[s][0] / agg[s][1] for s in SESSIONS}

    arms5 = arms5_decl
    acc_by_arm = {a: per_session_acc(h16cell, a) for a in arms5}
    near_stateless = {"sliding-window-compaction", "stateless-rag"}
    stateful = {"karc-full", "rag-bm25", "full-history"}

    pairs = []
    for i, left in enumerate(arms5):
        for right in arms5[i + 1 :]:
            diff = {s: acc_by_arm[left][s] - acc_by_arm[right][s] for s in SESSIONS}
            boot = paired_bootstrap(diff, matrix)
            n_sess_left = sum(
                1 for r in h16cell if r["arm"] == left and r["session_id"] == "S01"
            )
            pairs.append(
                {
                    "left": left,
                    "right": right,
                    "sessions_left_better": sum(1 for s in SESSIONS if diff[s] > 0),
                    "sessions_right_better": sum(1 for s in SESSIONS if diff[s] < 0),
                    "sessions_tied": sum(1 for s in SESSIONS if diff[s] == 0),
                    "exact_sign_test_two_sided_p": exact_sign_test_p(
                        sum(1 for s in SESSIONS if diff[s] > 0),
                        sum(1 for s in SESSIONS if diff[s] < 0),
                    ),
                    "left_k_of_n": f"{sum(1 for r in h16cell if r['arm'] == left and r['passed'])}/192",
                    "right_k_of_n": f"{sum(1 for r in h16cell if r['arm'] == right and r['passed'])}/192",
                    "mean_pp_diff": boot["point"],
                    "ci95_pp": [boot["ci95_lo"], boot["ci95_hi"]],
                    "distinguishable_from_zero": boot["excludes_zero"],
                    "cross_arm_type": (
                        "near-stateless vs stateful"
                        if (left in near_stateless) != (right in near_stateless)
                        else "within-group"
                    ),
                    "turns_per_session": n_sess_left,
                }
            )

    # Non-inferiority reading against declared margins (declared in this report,
    # not preregistered upstream; reported as descriptive, not as a test).
    ni_margins_pp = [2.0, 5.0]
    ni = []
    for pair in pairs:
        for margin in ni_margins_pp:
            ni.append(
                {
                    "left": pair["left"],
                    "right": pair["right"],
                    "margin_pp": margin,
                    "ci95_lo_pp": pair["ci95_pp"][0],
                    "non_inferior": pair["ci95_pp"][0] > -margin,
                }
            )

    item2 = {
        "unit": "session (12 paired sessions, 16 turns each)",
        "method": "paired percentile bootstrap over sessions, common random numbers",
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": BOOTSTRAP_SEED,
        "ci": "two-sided 95% percentile",
        "arm_overall_pct": {
            a: {
                "k": sum(1 for r in h16cell if r["arm"] == a and r["passed"]),
                "n": 192,
                "pct": 100.0
                * sum(1 for r in h16cell if r["arm"] == a and r["passed"])
                / 192,
            }
            for a in arms5
        },
        "pairs": pairs,
        "non_inferiority_declared_margins": ni,
        "any_pair_distinguishable_from_zero": any(
            p["distinguishable_from_zero"] for p in pairs
        ),
        "near_stateless_vs_stateful_pairs_distinguishable": [
            f"{p['left']} vs {p['right']}"
            for p in pairs
            if p["cross_arm_type"] == "near-stateless vs stateful"
            and p["distinguishable_from_zero"]
        ],
        "pairing_caveat": (
            "BASE3 and E10-P2-H16 are separate executions of the byte-identical "
            "schedule; session_id pairing across the two files is a same-task pairing, "
            "not a same-execution pairing"
        ),
    }

    # ================= ITEM 3: break-even carry price w* ====================
    def per_session_sum(turns: list[dict], arm: str, field: str) -> dict[str, float]:
        agg = {s: 0.0 for s in SESSIONS}
        for row in turns:
            if row["arm"] == arm:
                agg[row["session_id"]] += row[field]
        return agg

    fk = per_session_sum(h16_turns, "karc-full", "api_input_tokens_no_cache")
    rk = per_session_sum(h16_turns, "karc-full", "api_cache_read_tokens")
    fr = per_session_sum(h16_turns, "rag-bm25", "api_input_tokens_no_cache")
    rr = per_session_sum(h16_turns, "rag-bm25", "api_cache_read_tokens")

    # C_w(karc) = C_w(rag)  <=>  w * (R_rag - R_karc) = F_karc - F_rag
    num = {s: fk[s] - fr[s] for s in SESSIONS}  # F_karc - F_rag
    den = {s: rr[s] - rk[s] for s in SESSIONS}  # R_rag - R_karc
    wstar = paired_bootstrap_ratio_of_sums(num, den, matrix)

    weights_on_file = sorted({w for _, _, _, w in PUBLISHED_READ_WEIGHTS})
    published_range = [min(weights_on_file), max(weights_on_file)]

    item3 = {
        "definition": "C_w = sum(F + w*R); w* solves C_w(karc-full) = C_w(rag-bm25)",
        "totals": {
            "karc_full_fresh": sum(fk.values()),
            "karc_full_cache_read": sum(rk.values()),
            "karc_full_gross": sum(fk.values()) + sum(rk.values()),
            "rag_bm25_fresh": sum(fr.values()),
            "rag_bm25_cache_read": sum(rr.values()),
            "rag_bm25_gross": sum(fr.values()) + sum(rr.values()),
            "fresh_diff_karc_minus_rag": sum(fk.values()) - sum(fr.values()),
            "cache_read_diff_rag_minus_karc": sum(rr.values()) - sum(rk.values()),
        },
        "w_star": wstar,
        "w_star_in_unit_interval": (
            0.0 <= wstar["point"] <= 1.0 if wstar["point"] is not None else None
        ),
        "w_star_ci_entirely_below_zero": (
            wstar["ci95_hi"] < 0.0 if wstar["ci95_hi"] is not None else None
        ),
        "interpretation": (
            "karc-full spends strictly more fresh AND strictly more cache-read than "
            "rag-bm25 in this cell, so C_w(karc) > C_w(rag) for every w >= 0 and the "
            "algebraic root is negative: there is no break-even carry price in [0,1]"
        ),
        "published_read_weights_on_file": [
            {
                "source": label,
                "path": path,
                "location": loc,
                "cache_read_weight": weight,
            }
            for label, path, loc, weight in PUBLISHED_READ_WEIGHTS
        ],
        "published_read_weight_range_on_file": published_range,
        "w_star_inside_published_range": (
            published_range[0] <= wstar["point"] <= published_range[1]
            if wstar["point"] is not None
            else None
        ),
        "w_star_ci_entirely_below_smallest_published_weight": (
            wstar["ci95_hi"] < published_range[0]
            if wstar["ci95_hi"] is not None
            else None
        ),
    }

    # ================= ITEM 4: four-component priced accounting =============
    rate_card = verify_rate_card(xr_turns)

    def xr_session_component(arm: str, field: str) -> dict[str, float]:
        agg = {s: 0.0 for s in SESSIONS}
        for row in xr_turns:
            if row["arm"] == arm:
                agg[row["session_id"]] += row[field]
        return agg

    comp = {}
    for arm in ("karc-full", "rag-bm25"):
        comp[arm] = {
            "U": xr_session_component(arm, "fresh_input_tokens_provider"),
            "W1h": xr_session_component(
                arm, "cache_creation_1h_input_tokens_provider"
            ),
            "W5m": xr_session_component(
                arm, "cache_creation_5m_input_tokens_provider"
            ),
            "R": xr_session_component(arm, "cache_read_input_tokens_provider"),
            "O": xr_session_component(arm, "output_tokens"),
            "G": xr_session_component(arm, "gross_input_tokens"),
            "USD": xr_session_component(arm, "rate_card_equivalent_usd"),
        }

    totals = {
        arm: {k: sum(v.values()) for k, v in comp[arm].items()} for arm in comp
    }

    def priced(arm: str, session: str, mu_w: float, mu_r: float, mu_o: float) -> float:
        c = comp[arm]
        return (
            c["U"][session]
            + mu_w * (c["W1h"][session] + c["W5m"][session])
            + mu_r * c["R"][session]
            + mu_o * c["O"][session]
        )

    sweep = []
    for mu_o in GRID_MU_O:
        for mu_w in GRID_MU_W:
            for mu_r in GRID_MU_R:
                k = sum(priced("karc-full", s, mu_w, mu_r, mu_o) for s in SESSIONS)
                r = sum(priced("rag-bm25", s, mu_w, mu_r, mu_o) for s in SESSIONS)
                sweep.append(
                    {
                        "mu_w": mu_w,
                        "mu_r": mu_r,
                        "mu_o": mu_o,
                        "karc_input_equivalent_tokens": k,
                        "rag_input_equivalent_tokens": r,
                        "karc_minus_rag": k - r,
                        "ordering": "karc<rag" if k < r else ("karc>rag" if k > r else "tie"),
                        "reverses_gross_ordering": k < r,
                    }
                )

    # Closed-form flip boundary in mu_r for each mu_w (mu_o fixed per row).
    dU = totals["karc-full"]["U"] - totals["rag-bm25"]["U"]
    dW = (totals["karc-full"]["W1h"] + totals["karc-full"]["W5m"]) - (
        totals["rag-bm25"]["W1h"] + totals["rag-bm25"]["W5m"]
    )
    dR = totals["karc-full"]["R"] - totals["rag-bm25"]["R"]
    dO = totals["karc-full"]["O"] - totals["rag-bm25"]["O"]
    boundary = []
    for mu_o in GRID_MU_O:
        for mu_w in GRID_MU_W:
            # dU + mu_w*dW + mu_r*dR + mu_o*dO = 0
            mu_r_crit = -(dU + mu_w * dW + mu_o * dO) / dR
            boundary.append(
                {
                    "mu_o": mu_o,
                    "mu_w": mu_w,
                    "mu_r_flip": mu_r_crit,
                    "karc_cheaper_when": "mu_r < mu_r_flip" if dR > 0 else "mu_r > mu_r_flip",
                    "observed_mu_r": MU_READ,
                    "karc_cheaper_at_observed_mu_r": (
                        MU_READ < mu_r_crit if dR > 0 else MU_READ > mu_r_crit
                    ),
                    "headroom_ratio_observed_over_flip": MU_READ / mu_r_crit
                    if mu_r_crit
                    else None,
                }
            )

    # Paired session-level uncertainty on the observed rate card.
    usd_diff = {
        s: comp["karc-full"]["USD"][s] - comp["rag-bm25"]["USD"][s] for s in SESSIONS
    }
    usd_boot = paired_bootstrap(usd_diff, matrix)
    usd_ratio = paired_bootstrap_ratio_of_sums(
        comp["karc-full"]["USD"], comp["rag-bm25"]["USD"], matrix
    )
    gross_diff = {
        s: comp["karc-full"]["G"][s] - comp["rag-bm25"]["G"][s] for s in SESSIONS
    }
    gross_boot = paired_bootstrap(gross_diff, matrix)
    gross_ratio = paired_bootstrap_ratio_of_sums(
        comp["karc-full"]["G"], comp["rag-bm25"]["G"], matrix
    )

    item4 = {
        "runtime": "claude-sonnet-5 via Claude Code 2.1.220, E10-P2-XR canary, 12x8",
        "rate_card_recovery": rate_card,
        "component_totals": totals,
        "declared_grid": {
            "mu_w": GRID_MU_W,
            "mu_r": GRID_MU_R,
            "mu_o": GRID_MU_O,
            "note": "grid declared in the script before the flip location was inspected",
        },
        "sweep": sweep,
        "flip_boundary_closed_form": boundary,
        "component_deltas_karc_minus_rag": {
            "uncached": dU,
            "creation": dW,
            "cache_read": dR,
            "output": dO,
        },
        "gross_ordering": {
            "karc_gross": totals["karc-full"]["G"],
            "rag_gross": totals["rag-bm25"]["G"],
            "karc_over_rag": totals["karc-full"]["G"] / totals["rag-bm25"]["G"],
            "paired_mean_diff_tokens": gross_boot,
            "paired_ratio": gross_ratio,
        },
        "priced_ordering_observed_rate_card": {
            "karc_usd": totals["karc-full"]["USD"],
            "rag_usd": totals["rag-bm25"]["USD"],
            "karc_minus_rag_usd": totals["karc-full"]["USD"]
            - totals["rag-bm25"]["USD"],
            "karc_over_rag": totals["karc-full"]["USD"] / totals["rag-bm25"]["USD"],
            "paired_mean_diff_usd": usd_boot,
            "paired_ratio": usd_ratio,
        },
        "ordering_reverses_at_observed_rate_card": totals["karc-full"]["USD"]
        < totals["rag-bm25"]["USD"],
        "creation_weight_threshold_for_reversal": [
            {
                "mu_o": mu_o,
                "mu_r": MU_READ,
                "mu_w_threshold": -(dU + MU_READ * dR + mu_o * dO) / dW,
                "reversal_requires": "mu_w > threshold",
                "observed_mu_w_1h": MU_CREATION_1H,
                "reversal_holds_at_1h_weight": MU_CREATION_1H
                > -(dU + MU_READ * dR + mu_o * dO) / dW,
                "reversal_holds_at_5m_weight_1_25": MU_CREATION_5M
                > -(dU + MU_READ * dR + mu_o * dO) / dW,
            }
            for mu_o in GRID_MU_O
        ],
        "reversal_contingency_note": (
            "Both arms wrote 100% 1-hour-TTL cache entries (W5m = 0 for all 192 turns), "
            "so mu_w = 2.0 is the price of what was actually executed. Re-pricing the "
            "same token counts at the 5-minute weight mu_w = 1.25 is a pricing "
            "counterfactual only: a 5-minute TTL would also change the token counts, "
            "which this recomputation cannot model."
        ),
        "published_creation_weights_on_file": [
            {
                "source": label,
                "path": path,
                "location": loc,
                "cache_creation_weight": weight,
            }
            for label, path, loc, weight in PUBLISHED_CREATION_WEIGHTS
        ],
    }

    # ================= ITEM 5: mechanism decomposition ======================
    # (0) execution block order, from file order of the committed attempts log.
    block_seq: list[tuple[str, str]] = []
    for row in xr_attempts:
        key = (row["arm"], row["session_id"])
        if not block_seq or block_seq[-1] != key:
            block_seq.append(key)
    labels = [("K" if a == "karc-full" else "R") + s[1:] for a, s in block_seq]
    prev_same_arm = {
        block_seq[i]: (block_seq[i - 1][0] == block_seq[i][0]) if i else False
        for i in range(len(block_seq))
    }
    block_index = {key: i + 1 for i, key in enumerate(block_seq)}
    mean_block_index = {
        arm: sum(i for k, i in block_index.items() if k[0] == arm)
        / sum(1 for k in block_index if k[0] == arm)
        for arm in ("karc-full", "rag-bm25")
    }

    # (a) multi-call attribution.
    karc = [r for r in xr_turns if r["arm"] == "karc-full"]
    by_calls: dict[int, list[dict]] = defaultdict(list)
    for row in karc:
        by_calls[row["mcp_calls"]].append(row)
    call_groups = {
        str(c): {
            "n_turns": len(rows),
            "mean_cache_read": sum(r["cache_read_input_tokens_provider"] for r in rows)
            / len(rows),
            "mean_cache_creation": sum(
                r["cache_creation_input_tokens_provider"] for r in rows
            )
            / len(rows),
            "api_calls_implied": c + 1,
            "mean_cache_read_per_api_call": (
                sum(r["cache_read_input_tokens_provider"] for r in rows)
                / len(rows)
                / (c + 1)
            ),
            "positions": sorted({r["position"] for r in rows}),
        }
        for c, rows in sorted(by_calls.items())
    }
    # position-matched contrast: positions where both 0-call and 2-call turns exist
    matched = []
    for pos in sorted({r["position"] for r in karc}):
        zero = [r for r in karc if r["position"] == pos and r["mcp_calls"] == 0]
        two = [r for r in karc if r["position"] == pos and r["mcp_calls"] == 2]
        if zero and two:
            mz = sum(r["cache_read_input_tokens_provider"] for r in zero) / len(zero)
            mt = sum(r["cache_read_input_tokens_provider"] for r in two) / len(two)
            matched.append(
                {
                    "position": pos,
                    "n_single_call": len(zero),
                    "n_three_call": len(two),
                    "mean_read_single_call": mz,
                    "mean_read_three_call": mt,
                    "ratio": mt / mz,
                }
            )
    total_read = sum(r["cache_read_input_tokens_provider"] for r in karc)
    # equal-prefix-per-call allocation: a turn with c tool calls issues c+1 API
    # calls, so c/(c+1) of its cache-read is the round-trip re-read.
    roundtrip = sum(
        r["cache_read_input_tokens_provider"] * r["mcp_calls"] / (r["mcp_calls"] + 1)
        for r in karc
    )
    read_on_multicall_turns = sum(
        r["cache_read_input_tokens_provider"] for r in karc if r["mcp_calls"] > 0
    )
    item5a = {
        "karc_mcp_calls_total": sum(r["mcp_calls"] for r in karc),
        "rag_mcp_calls_total": sum(r["mcp_calls"] for r in xr_turns if r["arm"] == "rag-bm25"),
        "karc_turns_with_tool_calls": sum(1 for r in karc if r["mcp_calls"] > 0),
        "karc_turns_total": len(karc),
        "cache_read_total": total_read,
        "cache_read_on_multicall_turns": read_on_multicall_turns,
        "share_of_read_on_multicall_turns": read_on_multicall_turns / total_read,
        "roundtrip_attributed_read_equal_prefix_model": roundtrip,
        "share_attributed_to_roundtrips": roundtrip / total_read,
        "by_tool_call_count": call_groups,
        "position_matched_contrast": matched,
        "predicted_ratio_under_equal_prefix_model": 3.0,
        "observed_pooled_ratio_three_call_over_single_call": (
            call_groups["2"]["mean_cache_read"] / call_groups["0"]["mean_cache_read"]
        ),
        "position_matched_ratio_mean": sum(m["ratio"] for m in matched) / len(matched),
        "position_matched_ratio_range": [
            min(m["ratio"] for m in matched),
            max(m["ratio"] for m in matched),
        ],
        "within_turn_growth_probe": {
            "mean_read_per_api_call_single_call_turns": call_groups["0"][
                "mean_cache_read_per_api_call"
            ],
            "mean_read_per_api_call_three_call_turns": call_groups["2"][
                "mean_cache_read_per_api_call"
            ],
            "pct_change": 100.0
            * (
                call_groups["2"]["mean_cache_read_per_api_call"]
                / call_groups["0"]["mean_cache_read_per_api_call"]
                - 1.0
            ),
            "reading": (
                "per-API-call cache-read is essentially unchanged between 1-call and "
                "3-call turns, so the extra volume on multi-call turns is repetition of "
                "the same prefix, not within-turn prefix growth"
            ),
        },
        "counterfactual_single_call_karc_read_tokens": total_read - roundtrip,
        "counterfactual_single_call_karc_over_rag_read": (total_read - roundtrip)
        / sum(r["cache_read_input_tokens_provider"] for r in xr_turns if r["arm"] == "rag-bm25"),
        "observed_karc_over_rag_read": total_read
        / sum(r["cache_read_input_tokens_provider"] for r in xr_turns if r["arm"] == "rag-bm25"),
    }

    # (b) resident-set fill vs position-1 growth.
    xr_schedule = json.loads(xr_sched.read_text(encoding="utf-8"))
    resident = {
        s["session_id"]: s["initial_resident_tokens"] for s in xr_schedule["sessions"]
    }
    resident_versions = {
        s["session_id"]: len(s["initial_resident_versions"])
        for s in xr_schedule["sessions"]
    }
    budget = xr_schedule["cell"]["budget_tokens"]
    pos1 = {}
    for arm in ("karc-full", "rag-bm25"):
        pos1[arm] = {
            r["session_id"]: r
            for r in xr_turns
            if r["arm"] == arm and r["position"] == 1
        }
    plateau_from = None
    for i, s in enumerate(SESSIONS):
        if resident[s] >= budget * 0.98 and plateau_from is None:
            plateau_from = s
    item5b = {
        "budget_tokens": budget,
        "initial_resident_tokens_by_session": resident,
        "initial_resident_versions_by_session": resident_versions,
        "resident_plateau_first_session": plateau_from,
        "karc_position1": {
            s: {
                "cache_read": pos1["karc-full"][s]["cache_read_input_tokens_provider"],
                "cache_creation": pos1["karc-full"][s][
                    "cache_creation_input_tokens_provider"
                ],
                "mcp_calls": pos1["karc-full"][s]["mcp_calls"],
            }
            for s in SESSIONS
        },
        "rag_position1": {
            s: {
                "cache_read": pos1["rag-bm25"][s]["cache_read_input_tokens_provider"],
                "cache_creation": pos1["rag-bm25"][s][
                    "cache_creation_input_tokens_provider"
                ],
            }
            for s in SESSIONS
        },
        "karc_pos1_creation_growth_S01_to_S06": pos1["karc-full"]["S06"][
            "cache_creation_input_tokens_provider"
        ]
        - pos1["karc-full"]["S01"]["cache_creation_input_tokens_provider"],
        "resident_growth_S01_to_S06": resident["S06"] - resident["S01"],
        "karc_pos1_creation_spread_S06_to_S12": max(
            pos1["karc-full"][s]["cache_creation_input_tokens_provider"]
            for s in SESSIONS[5:]
        )
        - min(
            pos1["karc-full"][s]["cache_creation_input_tokens_provider"]
            for s in SESSIONS[5:]
        ),
        "rag_pos1_read_constant": len(
            {pos1["rag-bm25"][s]["cache_read_input_tokens_provider"] for s in SESSIONS}
        )
        == 1,
        "rag_pos1_read_value": pos1["rag-bm25"]["S01"][
            "cache_read_input_tokens_provider"
        ],
        "rag_pos1_creation_spread_all_sessions": max(
            pos1["rag-bm25"][s]["cache_creation_input_tokens_provider"]
            for s in SESSIONS
        )
        - min(
            pos1["rag-bm25"][s]["cache_creation_input_tokens_provider"]
            for s in SESSIONS
        ),
        "karc_pos1_read_growth_S01_to_S06": pos1["karc-full"]["S06"][
            "cache_read_input_tokens_provider"
        ]
        - pos1["karc-full"]["S01"]["cache_read_input_tokens_provider"],
        "karc_pos1_read_spread_S06_to_S12": max(
            pos1["karc-full"][s]["cache_read_input_tokens_provider"]
            for s in SESSIONS[5:]
        )
        - min(
            pos1["karc-full"][s]["cache_read_input_tokens_provider"]
            for s in SESSIONS[5:]
        ),
        "prefix_inflation_creation_tokens_per_resident_token_S01_to_S06": (
            pos1["karc-full"]["S06"]["cache_creation_input_tokens_provider"]
            - pos1["karc-full"]["S01"]["cache_creation_input_tokens_provider"]
        )
        / (resident["S06"] - resident["S01"]),
        "monotone_through_S06": all(
            pos1["karc-full"][SESSIONS[i]]["cache_creation_input_tokens_provider"]
            < pos1["karc-full"][SESSIONS[i + 1]]["cache_creation_input_tokens_provider"]
            for i in range(5)
        )
        and all(
            pos1["karc-full"][SESSIONS[i]]["cache_read_input_tokens_provider"]
            < pos1["karc-full"][SESSIONS[i + 1]]["cache_read_input_tokens_provider"]
            for i in range(5)
        ),
        "budget_utilization_at_plateau": resident["S06"] / budget,
        "reading": (
            "karc position-1 creation and cache-read are both strictly monotone through "
            "S06 and then flat, and initial_resident_tokens is strictly monotone through "
            "S06 and then flat at ~98.8% of the 1,756-token budget. rag-bm25, which "
            "carries no resident set, shows exactly constant position-1 cache-read across "
            "all 12 sessions. The growth-then-plateau is therefore consistent with the "
            "resident set filling to the token budget."
        ),
    }

    # (c) adjacency sensitivity.
    adj_karc = [s for s in SESSIONS if prev_same_arm[("karc-full", s)]]
    non_adj_karc = [s for s in SESSIONS if not prev_same_arm[("karc-full", s)]]
    adj_rag = [s for s in SESSIONS if prev_same_arm[("rag-bm25", s)]]
    plateau = SESSIONS[5:]  # S06..S12, resident set already at budget

    def mean_over(sessions, arm, field):
        vals = [pos1[arm][s][field] for s in sessions]
        return sum(vals) / len(vals) if vals else None

    adj_p = [s for s in plateau if s in adj_karc]
    non_p = [s for s in plateau if s in non_adj_karc]
    contrast = {}
    for field in (
        "cache_read_input_tokens_provider",
        "cache_creation_input_tokens_provider",
    ):
        a = mean_over(adj_p, "karc-full", field)
        b = mean_over(non_p, "karc-full", field)
        contrast[field] = {
            "adjacent_sessions": adj_p,
            "non_adjacent_sessions": non_p,
            "mean_adjacent": a,
            "mean_non_adjacent": b,
            "delta_adjacent_minus_non": a - b,
            "delta_as_pct_of_non_adjacent": 100.0 * (a - b) / b,
        }
    # translate the largest observed position-1 adjacency delta into USD and
    # compare it against the priced karc-vs-rag margin.
    creat_delta = contrast["cache_creation_input_tokens_provider"][
        "delta_adjacent_minus_non"
    ]
    read_delta = contrast["cache_read_input_tokens_provider"]["delta_adjacent_minus_non"]
    worst_case_usd = abs(creat_delta) * P_CREATION_1H * len(adj_karc) + abs(
        read_delta
    ) * P_READ * len(adj_karc)
    margin_usd = abs(totals["rag-bm25"]["USD"] - totals["karc-full"]["USD"])
    item5c = {
        "execution_block_sequence": labels,
        "block_index": {f"{'K' if a == 'karc-full' else 'R'}{s[1:]}": i for (a, s), i in block_index.items()},
        "first_executed_block": labels[0],
        "mean_block_index_by_arm": mean_block_index,
        "karc_sessions_preceded_by_karc": adj_karc,
        "rag_sessions_preceded_by_rag": adj_rag,
        "plateau_only_contrast": contrast,
        "adjacency_worst_case_usd_if_all_karc_adjacency_removed": worst_case_usd,
        "priced_margin_usd_rag_minus_karc": margin_usd,
        "adjacency_bound_as_share_of_margin": worst_case_usd / margin_usd,
        "decidable": True,
        "verdict": (
            "The ordering is NOT sensitive to session adjacency within the bound "
            "measurable from committed data. A literal counterfactual re-ordering is "
            "NOT-COMPUTABLE from committed data (no wall-clock timestamps are "
            "recorded, per R-9 privacy), but the observed adjacent-vs-non-adjacent "
            "position-1 contrast bounds any adjacency effect far below the priced "
            "margin."
        ),
        "what_would_decide_a_literal_counterfactual": (
            "a re-run of the same 12x8 canary under a blocked order (all karc blocks "
            "then all rag blocks, and the reverse) with per-turn timestamps recorded, "
            "so cache-TTL expiry between blocks is observable"
        ),
    }

    item5 = {"a_multi_call_attribution": item5a, "b_resident_fill": item5b, "c_adjacency": item5c}

    # ================= preregistered decision rule ==========================
    cond1_pass = bool(
        item3["w_star_inside_published_range"] and item3["w_star_in_unit_interval"]
    )
    cond2_pass = bool(item4["ordering_reverses_at_observed_rate_card"])

    # condition 3: paired accuracy must support the near-stateless advantage AND
    # the stratification must show it is not confined to stale-trap types.
    cross_sig = item2["near_stateless_vs_stateful_pairs_distinguishable"]
    cond3_accuracy_supported = len(cross_sig) > 0
    cond3_pass = bool(cond3_accuracy_supported)  # second clause only reachable if first holds

    # Report the stratification clause mechanically even though clause 1 fails,
    # so the record shows it was evaluated rather than skipped.
    strat = item1["h16_cell_5_arms"]
    stratification_clause = {
        "types_with_zero_between_arm_variation": [
            t
            for t in ("corrected", "harmful", "plain-stale", "rehabilitation")
            if len({strat[a]["by_task_type"][t]["k"] for a in arms5_decl}) == 1
        ],
        "sliding_advantage_over_karc_by_type": {
            t: strat["sliding-window-compaction"]["by_task_type"][t]["k"]
            - strat["karc-full"]["by_task_type"][t]["k"]
            for t in ("corrected", "harmful", "plain-stale", "rehabilitation")
        },
        "stateless_rag_delta_vs_karc_by_type": {
            t: strat["stateless-rag"]["by_task_type"][t]["k"]
            - strat["karc-full"]["by_task_type"][t]["k"]
            for t in ("corrected", "harmful", "plain-stale", "rehabilitation")
        },
        "advantage_confined_to_stale_trap_types": None,
        "reading": (
            "Not evaluable as designed: `corrected` is 75/76 for all five arms, and once "
            "position 1 is excluded four of five arms are at 100% in every task type. The "
            "only near-stateless gain is one `harmful` turn for sliding-window-compaction; "
            "stateless-rag is worse than the stateful arms in `harmful` (-1) and in the "
            "stale-trap type `plain-stale` (-1), which is the opposite of the "
            "dropping-history-is-rewarded prediction. `rehabilitation` shows no penalty "
            "for the near-stateless arms (both at 44/44) but also no signal, since four of "
            "five arms are at ceiling there."
        ),
    }

    conditions = {
        "condition_1_break_even_w_star_in_published_range": {
            "pass": cond1_pass,
            "w_star": item3["w_star"]["point"],
            "w_star_ci95": [item3["w_star"]["ci95_lo"], item3["w_star"]["ci95_hi"]],
            "published_range_on_file": published_range,
            "reason": (
                "w* is negative, hence outside [0,1] and outside the published range"
                if not cond1_pass
                else "w* inside published range"
            ),
        },
        "condition_2_four_component_ordering_reversal": {
            "pass": cond2_pass,
            "gross_ordering": "karc>rag",
            "priced_ordering": "karc<rag" if cond2_pass else "karc>rag",
            "karc_usd": totals["karc-full"]["USD"],
            "rag_usd": totals["rag-bm25"]["USD"],
        },
        "condition_3_accuracy_plus_stratification": {
            "pass": cond3_pass,
            "near_stateless_vs_stateful_pairs_distinguishable_from_zero": cross_sig,
            "stratification_clause": stratification_clause,
            "reason": (
                "no near-stateless-vs-stateful paired accuracy CI excludes zero, so the "
                "first clause of condition 3 fails and the stratification clause is "
                "not reached"
                if not cond3_pass
                else "first clause holds; see report for the stratification clause"
            ),
        },
    }
    supported = [k for k, v in conditions.items() if v["pass"]]
    verdict = (
        f"E13 VERDICT: FAST-SUPPORTED (condition {supported[0].split('_')[1]})"
        if supported
        else "E13 VERDICT: HOLD"
    )

    payload = {
        "schema": "e13-recomp-v1",
        "model_calls": 0,
        "new_experiments": 0,
        "new_fixtures": 0,
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "resamples": BOOTSTRAP_RESAMPLES,
            "unit": "session",
            "ci": "two-sided 95% percentile",
            "common_random_numbers": True,
        },
        "input_sha256": inputs,
        "schedule_gate": schedule_gate,
        "task_type_join": join_report,
        "item1_task_type_stratified_accuracy": item1,
        "item2_paired_accuracy_uncertainty": item2,
        "item3_break_even_carry_price": item3,
        "item4_four_component_pricing": item4,
        "item5_mechanism_decomposition": item5,
        "preregistered_conditions": conditions,
        "verdict": verdict,
    }

    RAW_OUT.mkdir(parents=True, exist_ok=True)
    (RAW_OUT / "recomp.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(verdict)
    for name, cond in conditions.items():
        print(f"  {name}: {'PASS' if cond['pass'] else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
