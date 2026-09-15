"""E5 model-spend accounting and infrastructure recovery policy.

The experiment cap bounds calls that reached a valid model turn (or otherwise
reported non-zero model usage).  A zero-token infrastructure failure is still
audited as an API call, but it refunds its provisional spend reservation.

This module is deliberately runtime-agnostic and stdlib-only.  Sleep and model
execution remain in the runner; the policy only returns deterministic recovery
directives so mock tests never contact a provider.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Iterable

from karc.bench.checkpoint import parse_reset_wait_s


MODEL_COMPLETE = frozenset({"ok", "task"})


def _nonnegative(value: object) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def result_usage(result: object) -> dict[str, int]:
    """Return normalized usage without retaining provider payloads."""
    gross = _nonnegative(result.total_input_tokens(with_cache=True))
    fresh = _nonnegative(result.total_input_tokens(with_cache=False))
    return {
        "fresh_input_tokens": fresh,
        "gross_input_tokens": gross,
        "output_tokens": _nonnegative(getattr(result, "output_tokens", 0)),
        "reasoning_tokens": _nonnegative(getattr(result, "reasoning_tokens", 0)),
    }


def is_model_spend(result: object) -> bool:
    """True when a call must debit the model-spend attempt cap.

    Successful turns debit even if a mock/runtime omits usage.  Failed calls
    debit only if non-zero model usage proves that inference work occurred.
    """
    usage = result_usage(result)
    return getattr(result, "error_class", None) is None or any(usage.values())


@dataclass(frozen=True)
class AttemptCharge:
    model_spend: int
    infra_failure: int
    usage: dict[str, int]


def infer_row_accounting(row: dict) -> tuple[int, int, int]:
    """Return ``(calls, model_spend, infra)`` for new or legacy raw rows."""
    calls = _nonnegative(row.get("api_attempts", 1))
    if "model_spend_attempts" in row or "infra_failure_attempts" in row:
        spend = _nonnegative(row.get("model_spend_attempts"))
        infra = _nonnegative(row.get("infra_failure_attempts"))
        if spend + infra > calls:
            raise ValueError("row attempt accounting exceeds API calls")
        infra += calls - spend - infra
        return calls, spend, infra
    usage = sum(_nonnegative(row.get(name)) for name in (
        "api_input_tokens_no_cache", "api_input_tokens_with_cache",
        "api_output_tokens", "api_reasoning_tokens",
    ))
    spend = int(row.get("failure_class") in MODEL_COMPLETE or usage > 0)
    return calls, spend, max(0, calls - spend)


def ledger_from_rows(rows: Iterable[dict]) -> dict[str, int]:
    calls = spend = infra = 0
    for row in rows:
        row_calls, row_spend, row_infra = infer_row_accounting(row)
        calls += row_calls
        spend += row_spend
        infra += row_infra
    return {
        "api_calls_total": calls,
        "model_spend_attempts": spend,
        "infra_failure_attempts": infra,
    }


class ModelSpendBudget:
    """Thread-safe provisional cap reservation with infra-failure refunds."""

    def __init__(self, cap: int, *, model_spend_attempts: int = 0,
                 infra_failure_attempts: int = 0, api_calls_total: int = 0):
        if cap <= 0 or min(model_spend_attempts, infra_failure_attempts,
                           api_calls_total) < 0:
            raise ValueError("attempt budget values must be non-negative; cap positive")
        if model_spend_attempts > cap:
            raise ValueError("historical model spend exceeds cap")
        self.cap = int(cap)
        self.model_spend_attempts = int(model_spend_attempts)
        self.infra_failure_attempts = int(infra_failure_attempts)
        self.api_calls_total = int(api_calls_total)
        self.in_flight = 0
        self._condition = threading.Condition()

    @classmethod
    def from_rows(cls, cap: int, rows: Iterable[dict]) -> "ModelSpendBudget":
        ledger = ledger_from_rows(rows)
        return cls(cap, **ledger)

    def reserve(self) -> bool:
        with self._condition:
            if self.model_spend_attempts + self.in_flight >= self.cap:
                return False
            self.in_flight += 1
            self.api_calls_total += 1
            return True

    def reserve_when_available(self, timeout_s: float | None = None) -> bool:
        """Wait for a provisional slot to settle; False only on true cap/timeout."""
        with self._condition:
            ready = self._condition.wait_for(
                lambda: (
                    self.model_spend_attempts >= self.cap
                    or self.model_spend_attempts + self.in_flight < self.cap
                ),
                timeout=timeout_s,
            )
            if not ready or self.model_spend_attempts >= self.cap:
                return False
            self.in_flight += 1
            self.api_calls_total += 1
            return True

    def settle(self, result: object) -> AttemptCharge:
        usage = result_usage(result)
        spend = int(is_model_spend(result))
        with self._condition:
            if self.in_flight <= 0:
                raise RuntimeError("attempt settled without a reservation")
            self.in_flight -= 1
            if spend:
                self.model_spend_attempts += 1
            else:
                self.infra_failure_attempts += 1
            self._condition.notify_all()
        return AttemptCharge(
            model_spend=spend,
            infra_failure=1 - spend,
            usage=usage,
        )

    def snapshot(self) -> dict[str, int]:
        with self._condition:
            return {
                "model_spend_attempt_cap": self.cap,
                "model_spend_attempts": self.model_spend_attempts,
                "infra_failure_attempts": self.infra_failure_attempts,
                "api_calls_total": self.api_calls_total,
                "in_flight": self.in_flight,
                "remaining_model_spend_attempts": (
                    self.cap - self.model_spend_attempts - self.in_flight
                ),
            }


@dataclass(frozen=True)
class ResumeDirective:
    error_class: str
    delay_s: float
    wait_source: str
    global_window: bool


@dataclass(frozen=True)
class InfraResumePolicy:
    transient_backoff_base_s: float = 1.0
    transient_backoff_max_s: float = 60.0
    usage_fallback_wait_s: int = 3600
    usage_max_wait_s: int = 6 * 3600
    max_auto_resumes: int = 3

    def directive(self, error_class: str | None, detail: str | None,
                  retry_index: int) -> ResumeDirective | None:
        """Return a bounded backoff/window directive for infra failures."""
        if retry_index < 0 or retry_index >= self.max_auto_resumes:
            return None
        if error_class == "api_error":
            delay = min(
                self.transient_backoff_max_s,
                self.transient_backoff_base_s * (2 ** retry_index),
            )
            return ResumeDirective(
                error_class="api_error", delay_s=delay,
                wait_source="exponential-backoff", global_window=False,
            )
        if error_class == "usage_limit":
            delay, source = parse_reset_wait_s(
                detail, fallback_s=self.usage_fallback_wait_s,
                max_s=self.usage_max_wait_s,
            )
            return ResumeDirective(
                error_class="usage_limit", delay_s=delay,
                wait_source=source, global_window=True,
            )
        return None
