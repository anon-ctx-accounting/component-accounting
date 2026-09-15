"""Execution-only ledger and analysis for frozen E10-P2-H16."""

from __future__ import annotations

import math
import random
import statistics
import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from karc.bench import e10_horizon_canary as plan
from karc.bench.e5_runtime_accounting import (
    is_model_spend,
    ledger_from_rows,
    result_usage,
)


EXECUTION_SCHEMA = "e10-p2-h16-execution-freeze-v1"
SUMMARY_SCHEMA = "e10-p2-h16-summary-v1"
BOOTSTRAP_REPS = 10_000


@dataclass(frozen=True)
class AttemptCharge:
    model_spend: int
    infra_failure: int
    usage: dict[str, int]


class HardAttemptBudget:
    """Atomically enforce model-spend, zero-token-infra, and total caps."""

    def __init__(self, rows: Iterable[dict] = ()):
        prior = ledger_from_rows(rows)
        self.model_spend_attempts = prior["model_spend_attempts"]
        self.infra_failure_attempts = prior["infra_failure_attempts"]
        self.api_calls_total = prior["api_calls_total"]
        self.in_flight = 0
        self._lock = threading.Lock()
        if not self._within_caps():
            raise ValueError("historical attempts exceed frozen caps")

    def _within_caps(self) -> bool:
        return (
            self.model_spend_attempts <= plan.MODEL_SPEND_ATTEMPT_CAP
            and self.infra_failure_attempts <= plan.INFRA_FAILURE_ATTEMPT_CAP
            and self.api_calls_total <= plan.TOTAL_API_CALL_CAP
        )

    def reserve(self) -> bool:
        with self._lock:
            if (
                self.model_spend_attempts + self.in_flight
                >= plan.MODEL_SPEND_ATTEMPT_CAP
                or self.infra_failure_attempts + self.in_flight
                >= plan.INFRA_FAILURE_ATTEMPT_CAP
                or self.api_calls_total + self.in_flight
                >= plan.TOTAL_API_CALL_CAP
            ):
                return False
            self.in_flight += 1
            return True

    def settle(self, result: object) -> AttemptCharge:
        usage = result_usage(result)
        spend = int(is_model_spend(result))
        with self._lock:
            if self.in_flight <= 0:
                raise RuntimeError("attempt settled without reservation")
            self.in_flight -= 1
            self.api_calls_total += 1
            self.model_spend_attempts += spend
            self.infra_failure_attempts += 1 - spend
            if not self._within_caps():
                raise RuntimeError("attempt settlement exceeded a frozen cap")
        return AttemptCharge(spend, 1 - spend, usage)

    def snapshot(self) -> dict[str, int | bool]:
        with self._lock:
            return {
                "model_spend_attempt_cap": plan.MODEL_SPEND_ATTEMPT_CAP,
                "infra_failure_attempt_cap": plan.INFRA_FAILURE_ATTEMPT_CAP,
                "total_api_call_cap": plan.TOTAL_API_CALL_CAP,
                "model_spend_attempts": self.model_spend_attempts,
                "infra_failure_attempts": self.infra_failure_attempts,
                "api_calls_total": self.api_calls_total,
                "in_flight": self.in_flight,
                "within_caps": self._within_caps(),
            }


def latest_complete(rows: Iterable[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(
            row["arm"], row["session_id"], int(row.get("session_attempt", 0)),
        )].append(row)
    expected = set(range(1, plan.SESSION_LENGTH + 1))
    complete: dict[tuple[str, str], tuple[int, list[dict]]] = {}
    for (arm, sid, attempt), unit in grouped.items():
        valid = [row for row in unit if row["failure_class"] in {"ok", "task"}]
        if len(valid) != plan.SESSION_LENGTH:
            continue
        if {int(row["position"]) for row in valid} != expected:
            continue
        key = (arm, sid)
        if key not in complete or attempt > complete[key][0]:
            complete[key] = (attempt, valid)
    return sorted(
        [row for _attempt, unit in complete.values() for row in unit],
        key=lambda row: (row["session_id"], row["arm"], row["position"]),
    )


def complete_session_keys(rows: Iterable[dict]) -> set[tuple[str, str]]:
    return {(row["arm"], row["session_id"]) for row in latest_complete(rows)}


def next_session_attempt(rows: Iterable[dict], arm: str, sid: str) -> int:
    attempts = [
        int(row.get("session_attempt", 0)) for row in rows
        if row.get("arm") == arm and row.get("session_id") == sid
    ]
    return max(attempts, default=-1) + 1


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    lo, hi = math.floor(position), math.ceil(position)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - position) + ordered[hi] * (position - lo)


def _bootstrap_ci(values: list[float], seed: int) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(seed)
    means = [
        statistics.mean(values[rng.randrange(len(values))] for _ in values)
        for _ in range(BOOTSTRAP_REPS)
    ]
    return [_quantile(means, 0.025), _quantile(means, 0.975)]


def _fit(xs: list[float], ys: list[float], degree: int) -> dict:
    if degree == 1:
        n = len(xs)
        sx, sy = sum(xs), sum(ys)
        denominator = n * sum(x * x for x in xs) - sx * sx
        slope = (n * sum(x * y for x, y in zip(xs, ys)) - sx * sy) / denominator
        coefficients = [(sy - slope * sx) / n, slope]
    else:
        matrix = [[sum(x ** (i + j) for x in xs) for j in range(3)] for i in range(3)]
        target = [sum(y * x ** i for x, y in zip(xs, ys)) for i in range(3)]
        for index in range(3):
            pivot = max(range(index, 3), key=lambda row: abs(matrix[row][index]))
            matrix[index], matrix[pivot] = matrix[pivot], matrix[index]
            target[index], target[pivot] = target[pivot], target[index]
            scale = matrix[index][index]
            matrix[index] = [value / scale for value in matrix[index]]
            target[index] /= scale
            for row in range(3):
                if row == index:
                    continue
                factor = matrix[row][index]
                matrix[row] = [
                    matrix[row][column] - factor * matrix[index][column]
                    for column in range(3)
                ]
                target[row] -= factor * target[index]
        coefficients = target
    predictions = [
        sum(value * x ** power for power, value in enumerate(coefficients))
        for x in xs
    ]
    mean = statistics.mean(ys)
    residual = sum((y - fitted) ** 2 for y, fitted in zip(ys, predictions))
    total = sum((y - mean) ** 2 for y in ys)
    return {
        "coefficients_ascending": coefficients,
        "r2": 1 - residual / total if total else 1.0,
    }


def _row_audit(rows: list[dict]) -> dict:
    forbidden = sorted({
        key for row in rows for key in row if key in plan.FORBIDDEN_RAW_KEYS
    })
    rag_mcp = sum(
        int(row.get("mcp_search_calls", 0)) + int(row.get("mcp_get_calls", 0))
        for row in rows if row["arm"] == "rag-bm25"
    )
    token_mismatch = sum(
        row["api_input_tokens_no_cache"] + row["api_cache_read_tokens"]
        != row["api_input_tokens_with_cache"] for row in rows
    )
    return {
        "pass": not forbidden and rag_mcp == 0 and token_mismatch == 0,
        "privacy_forbidden_keys": forbidden,
        "rag_karc_mcp_calls": rag_mcp,
        "token_accounting_mismatches": token_mismatch,
    }


def summarize(rows: Iterable[dict]) -> dict:
    all_rows = list(rows)
    valid = latest_complete(all_rows)
    by_position: dict[str, dict[str, dict]] = {arm: {} for arm in plan.ARMS}
    for arm in plan.ARMS:
        for position in range(1, plan.SESSION_LENGTH + 1):
            unit = [
                row for row in valid
                if row["arm"] == arm and int(row["position"]) == position
            ]
            by_position[arm][str(position)] = {
                "turns": len(unit),
                "fresh_mean": statistics.mean(
                    row["api_input_tokens_no_cache"] for row in unit
                ) if unit else None,
                "cache_read_mean": statistics.mean(
                    row["api_cache_read_tokens"] for row in unit
                ) if unit else None,
                "gross_mean": statistics.mean(
                    row["api_input_tokens_with_cache"] for row in unit
                ) if unit else None,
                "correct_rate": statistics.mean(bool(row["passed"]) for row in unit)
                if unit else None,
            }
    session_arm: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in valid:
        session_arm[(row["session_id"], row["arm"])].append(row)
    paired = {}
    for horizon in plan.HORIZONS:
        for weight in plan.CACHE_READ_WEIGHTS:
            differences, ratios = [], []
            for session in range(1, plan.SESSION_COUNT + 1):
                sid = f"S{session:02d}"
                costs = {}
                for arm in plan.ARMS:
                    unit = [
                        row for row in session_arm.get((sid, arm), [])
                        if row["position"] <= horizon
                    ]
                    if len(unit) == horizon:
                        costs[arm] = sum(
                            row["api_input_tokens_no_cache"]
                            + weight * row["api_cache_read_tokens"] for row in unit
                        )
                if len(costs) == 2:
                    differences.append(costs["karc-full"] - costs["rag-bm25"])
                    ratios.append(costs["karc-full"] / costs["rag-bm25"])
            key = f"H{horizon}_w{weight:g}"
            seed = 7100 + horizon * 10 + int(weight * 10)
            paired[key] = {
                "paired_sessions": len(differences),
                "karc_minus_rag_mean": statistics.mean(differences)
                if differences else None,
                "karc_minus_rag_ci95": _bootstrap_ci(differences, seed),
                "karc_over_rag_mean": statistics.mean(ratios) if ratios else None,
                "karc_over_rag_ci95": _bootstrap_ci(ratios, seed + 1000),
            }
    fits = {}
    for arm in plan.ARMS:
        xs = [float(position) for position in range(1, plan.SESSION_LENGTH + 1)]
        ys = [
            by_position[arm][str(position)]["gross_mean"]
            for position in range(1, plan.SESSION_LENGTH + 1)
        ]
        if all(value is not None for value in ys):
            fits[arm] = {
                "linear": _fit(xs, ys, 1),
                "quadratic": _fit(xs, ys, 2),
            }
    per_turn = next((
        position for position in range(1, plan.SESSION_LENGTH + 1)
        if by_position["karc-full"][str(position)]["gross_mean"] is not None
        and by_position["karc-full"][str(position)]["gross_mean"]
        <= by_position["rag-bm25"][str(position)]["gross_mean"]
    ), None)
    cumulative = None
    left_total = right_total = 0.0
    for position in range(1, plan.SESSION_LENGTH + 1):
        left = by_position["karc-full"][str(position)]["gross_mean"]
        right = by_position["rag-bm25"][str(position)]["gross_mean"]
        if left is None or right is None:
            break
        left_total += left
        right_total += right
        if cumulative is None and left_total <= right_total:
            cumulative = position
    arm_summary = {}
    for arm in plan.ARMS:
        unit = [row for row in valid if row["arm"] == arm]
        arm_summary[arm] = {
            "turns": len(unit),
            "fresh_total": sum(row["api_input_tokens_no_cache"] for row in unit),
            "cache_read_total": sum(row["api_cache_read_tokens"] for row in unit),
            "gross_total": sum(row["api_input_tokens_with_cache"] for row in unit),
            "output_total": sum(row["api_output_tokens"] for row in unit),
            "correct_rate": statistics.mean(bool(row["passed"]) for row in unit)
            if unit else None,
        }
    complete_keys = complete_session_keys(all_rows)
    paired_sessions = sum(
        all((arm, f"S{session:02d}") in complete_keys for arm in plan.ARMS)
        for session in range(1, plan.SESSION_COUNT + 1)
    )
    value = {
        "schema": SUMMARY_SCHEMA,
        "complete": len(valid) == plan.PLANNED_VALID_TURNS,
        "valid_turns": len(valid),
        "expected_turns": plan.PLANNED_VALID_TURNS,
        "complete_paired_sessions": paired_sessions,
        "by_arm": arm_summary,
        "by_position": by_position,
        "paired_priced_cost": paired,
        "gross_curve_fits": fits,
        "first_per_turn_gross_crossover": per_turn,
        "first_cumulative_gross_crossover": cumulative,
        "valid_row_audit": _row_audit(valid),
    }
    value["sha256"] = plan.hash_json(value)
    return value
