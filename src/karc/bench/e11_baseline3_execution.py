"""Execution ledger and frozen analysis for approved E11-BASE3."""

from __future__ import annotations

import math
import random
import statistics
import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from karc.bench import e10_horizon_execution as h16_execution
from karc.bench import e11_baseline3 as plan
from karc.bench.e5_runtime_accounting import (
    is_model_spend,
    ledger_from_rows,
    result_usage,
)


EXECUTION_SCHEMA = "e11-base3-execution-freeze-v1"
SUMMARY_SCHEMA = "e11-base3-summary-v1"
BOOTSTRAP_SEED = 11130


@dataclass(frozen=True)
class AttemptCharge:
    model_spend: int
    infra_failure: int
    usage: dict[str, int]


class HardAttemptBudget:
    """Atomically enforce the three approved E11 attempt caps."""

    def __init__(self, rows: Iterable[dict] = ()):
        prior = ledger_from_rows(rows)
        self.model_spend_attempts = prior["model_spend_attempts"]
        self.infra_failure_attempts = prior["infra_failure_attempts"]
        self.api_calls_total = prior["api_calls_total"]
        self.in_flight = 0
        self._lock = threading.Lock()
        if not self._within_caps():
            raise ValueError("historical attempts exceed E11 frozen caps")

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
                raise RuntimeError("attempt settlement exceeded E11 frozen cap")
        return AttemptCharge(spend, 1 - spend, usage)

    def snapshot(self) -> dict:
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
        if row.get("arm") not in plan.ARMS:
            continue
        grouped[(
            row["arm"], row["session_id"], int(row.get("session_attempt", 0)),
        )].append(row)
    expected = set(range(1, plan.SESSION_LENGTH + 1))
    complete: dict[tuple[str, str], tuple[int, list[dict]]] = {}
    for (arm, sid, attempt), unit in grouped.items():
        valid = [row for row in unit if row.get("failure_class") in {"ok", "task"}]
        if len(valid) != plan.SESSION_LENGTH:
            continue
        if {int(row["position"]) for row in valid} != expected:
            continue
        key = (arm, sid)
        if key not in complete or attempt > complete[key][0]:
            complete[key] = (attempt, valid)
    return sorted(
        [row for _attempt, unit in complete.values() for row in unit],
        key=lambda row: (row["session_id"], row["arm"], int(row["position"])),
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
        for _ in range(plan.BOOTSTRAP_REPS)
    ]
    return [_quantile(means, 0.025), _quantile(means, 0.975)]


def _session_rows(rows: Iterable[dict]) -> dict[tuple[str, str], list[dict]]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["session_id"], row["arm"])].append(row)
    for unit in grouped.values():
        unit.sort(key=lambda row: int(row["position"]))
    return grouped


def _comparison(
    grouped: dict[tuple[str, str], list[dict]],
    *,
    left: str,
    right: str,
    horizon: int,
    weight: float,
    seed: int,
) -> dict:
    differences = []
    ratios = []
    for session in range(1, plan.SESSION_COUNT + 1):
        sid = f"S{session:02d}"
        costs = {}
        for arm in (left, right):
            unit = [
                row for row in grouped.get((sid, arm), [])
                if int(row["position"]) <= horizon
            ]
            if len(unit) == horizon:
                costs[arm] = sum(
                    int(row["api_input_tokens_no_cache"])
                    + weight * int(row["api_cache_read_tokens"])
                    for row in unit
                )
        if len(costs) == 2 and costs[right] > 0:
            differences.append(costs[left] - costs[right])
            ratios.append(costs[left] / costs[right])
    return {
        "left": left,
        "right": right,
        "horizon": horizon,
        "cache_read_weight": weight,
        "paired_sessions": len(differences),
        "left_minus_right_mean": statistics.mean(differences) if differences else None,
        "left_minus_right_ci95": _bootstrap_ci(differences, seed),
        "left_over_right_mean": statistics.mean(ratios) if ratios else None,
        "left_over_right_ci95": _bootstrap_ci(ratios, seed + 1),
    }


def _quality_comparison(
    grouped: dict[tuple[str, str], list[dict]],
    *,
    left: str,
    right: str,
    horizon: int,
    seed: int,
) -> dict:
    differences = []
    for session in range(1, plan.SESSION_COUNT + 1):
        sid = f"S{session:02d}"
        left_rows = [
            row for row in grouped.get((sid, left), [])
            if int(row["position"]) <= horizon
        ]
        right_rows = [
            row for row in grouped.get((sid, right), [])
            if int(row["position"]) <= horizon
        ]
        if len(left_rows) == len(right_rows) == horizon:
            differences.append(
                statistics.mean(bool(row["passed"]) for row in left_rows)
                - statistics.mean(bool(row["passed"]) for row in right_rows)
            )
    return {
        "left": left,
        "right": right,
        "horizon": horizon,
        "paired_sessions": len(differences),
        "left_minus_right_mean": statistics.mean(differences) if differences else None,
        "left_minus_right_ci95": _bootstrap_ci(differences, seed),
        "noninferiority_margin": -plan.QUALITY_NONINFERIORITY_MARGIN,
    }


def _cost_dominates(comparisons: dict, left: str, right: str) -> bool:
    for weight in plan.CACHE_READ_WEIGHTS:
        key = f"{left}__minus__{right}__H16_w{weight:g}"
        ci = comparisons.get(key, {}).get("left_minus_right_ci95")
        if ci is None or ci[1] >= 0:
            return False
    return True


def _quality_noninferior(quality: dict, left: str, right: str) -> bool:
    key = f"{left}__minus__{right}__H16"
    ci = quality.get(key, {}).get("left_minus_right_ci95")
    return ci is not None and ci[0] >= -plan.QUALITY_NONINFERIORITY_MARGIN


def _session_identity_audit(rows: list[dict]) -> dict:
    grouped = _session_rows(rows)
    failures = []
    for session in range(1, plan.SESSION_COUNT + 1):
        sid = f"S{session:02d}"
        for arm in plan.ARMS:
            unit = grouped.get((sid, arm), [])
            identities = {row.get("native_session_sha256") for row in unit}
            identities.discard(None)
            expected = 1 if arm == "full-history" else plan.SESSION_LENGTH
            if len(unit) != plan.SESSION_LENGTH or len(identities) != expected:
                failures.append({
                    "arm": arm,
                    "session_id": sid,
                    "turns": len(unit),
                    "unique_provider_sessions": len(identities),
                    "expected_provider_sessions": expected,
                })
    return {
        "pass": not failures,
        "failures": failures,
        "contract": (
            "full-history=1 provider thread/session; sliding/stateless=16 fresh "
            "provider threads/session"
        ),
    }


def summarize(new_rows: Iterable[dict], reference_rows: Iterable[dict]) -> dict:
    """Apply the preregistered five-arm accounting and rewrite rules."""
    all_new = list(new_rows)
    valid_new = latest_complete(all_new)
    valid_reference = h16_execution.latest_complete(reference_rows)
    combined = [*valid_new, *valid_reference]
    grouped = _session_rows(combined)
    new_audit = plan.audit_turn_rows(valid_new, require_complete=True)
    local_tool_calls = sum(int(row.get("local_tool_calls", 0)) for row in valid_new)
    session_identity_audit = _session_identity_audit(valid_new)
    reference_complete = len(valid_reference) == 384
    all_arms = [*plan.ARMS, *plan.REFERENCE_ARMS]
    by_arm = {}
    for arm in all_arms:
        unit = [row for row in combined if row["arm"] == arm]
        by_arm[arm] = {
            "turns": len(unit),
            "fresh_total": sum(int(row["api_input_tokens_no_cache"]) for row in unit),
            "cache_read_total": sum(int(row["api_cache_read_tokens"]) for row in unit),
            "gross_total": sum(int(row["api_input_tokens_with_cache"]) for row in unit),
            "output_total": sum(int(row["api_output_tokens"]) for row in unit),
            "correct_rate": (
                statistics.mean(bool(row["passed"]) for row in unit) if unit else None
            ),
        }
    by_position = {}
    for arm in plan.ARMS:
        by_position[arm] = {}
        for position in range(1, plan.SESSION_LENGTH + 1):
            unit = [
                row for row in valid_new
                if row["arm"] == arm and int(row["position"]) == position
            ]
            by_position[arm][str(position)] = {
                "turns": len(unit),
                "fresh_mean": statistics.mean(
                    int(row["api_input_tokens_no_cache"]) for row in unit
                ) if unit else None,
                "cache_read_mean": statistics.mean(
                    int(row["api_cache_read_tokens"]) for row in unit
                ) if unit else None,
                "gross_mean": statistics.mean(
                    int(row["api_input_tokens_with_cache"]) for row in unit
                ) if unit else None,
                "correct_rate": statistics.mean(bool(row["passed"]) for row in unit)
                if unit else None,
            }
    pairs = [
        ("full-history", "sliding-window-compaction"),
        ("full-history", "stateless-rag"),
        ("sliding-window-compaction", "stateless-rag"),
        *[(arm, ref) for arm in plan.ARMS for ref in plan.REFERENCE_ARMS],
    ]
    comparisons = {}
    counter = 0
    for left, right in pairs:
        for horizon in plan.HORIZONS:
            for weight in plan.CACHE_READ_WEIGHTS:
                key = f"{left}__minus__{right}__H{horizon}_w{weight:g}"
                comparisons[key] = _comparison(
                    grouped,
                    left=left,
                    right=right,
                    horizon=horizon,
                    weight=weight,
                    seed=BOOTSTRAP_SEED + counter * 2,
                )
                counter += 1
    quality = {}
    for index, (left, right) in enumerate(pairs):
        key = f"{left}__minus__{right}__H16"
        quality[key] = _quality_comparison(
            grouped,
            left=left,
            right=right,
            horizon=16,
            seed=BOOTSTRAP_SEED + 5000 + index,
        )
    full_cost = all(
        _cost_dominates(comparisons, "full-history", ref)
        for ref in plan.REFERENCE_ARMS
    )
    full_quality = all(
        _quality_noninferior(quality, "full-history", ref)
        for ref in plan.REFERENCE_ARMS
    )
    stateless_cost = _cost_dominates(comparisons, "stateless-rag", "rag-bm25")
    stateless_quality = _quality_noninferior(
        quality, "stateless-rag", "rag-bm25",
    )
    complete = (
        len(valid_new) == plan.PLANNED_VALID_TURNS
        and new_audit["pass"]
        and local_tool_calls == 0
        and session_identity_audit["pass"]
        and reference_complete
    )
    value = {
        "schema": SUMMARY_SCHEMA,
        "complete": complete,
        "valid_new_turns": len(valid_new),
        "planned_new_turns": plan.PLANNED_VALID_TURNS,
        "complete_new_sessions": len(complete_session_keys(all_new)),
        "reference_turns": len(valid_reference),
        "reference_complete": reference_complete,
        "new_row_audit": new_audit,
        "local_tool_calls": local_tool_calls,
        "session_identity_audit": session_identity_audit,
        "by_arm": by_arm,
        "by_position": by_position,
        "paired_cost": comparisons,
        "paired_quality": quality,
        "decision_conditions": {
            "PAPER2_PREMISE_REWRITE_REQUIRED": {
                "cost_condition": full_cost,
                "quality_condition": full_quality,
                "triggered": complete and full_cost and full_quality,
            },
            "SESSION_STATE_NECESSITY_CLAIM_REMOVE": {
                "cost_condition": stateless_cost,
                "quality_condition": stateless_quality,
                "triggered": complete and stateless_cost and stateless_quality,
            },
            "adjudication": "PENDING_MAIN_USER",
        },
        "scope": (
            "frozen H16 fixture/schedule/reader/accounting and positions 1..16 only"
        ),
    }
    value["sha256"] = plan.hash_json(value)
    return value
