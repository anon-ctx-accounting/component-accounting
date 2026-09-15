"""Typed, model-free inputs for the E6 activation diagnostic.

Policy-visible data and measurement-only data deliberately use different
types.  A replay arm can only receive :class:`PolicyQueryView`; query text,
gold answers, and held-out metadata remain inside :class:`MeasurementOnly`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Iterable, Mapping


class LeakageError(ValueError):
    """Raised when evaluation-only information enters a policy input."""


class MechanismFamily(str, Enum):
    RECENCY = "recency-only"
    FREQUENCY = "frequency-only"
    VALIDITY = "validity-only"
    RETRIEVAL = "retrieval-only"


class SignalKind(str, Enum):
    REFERENCE = "reference"
    SUPERSEDES = "supersedes"
    RETRIEVAL = "retrieval"


class SignalSource(str, Enum):
    INGEST_NATIVE = "ingest-native"
    ACCESS_NATIVE = "access-native"
    RETRIEVAL_NATIVE = "retrieval-native"


_FORBIDDEN_POLICY_KEYS = frozenset(
    {
        "answer",
        "answers",
        "gold",
        "gold_answer",
        "gold_answers",
        "gold_memory_state",
        "gold_operation",
        "gold_provenance",
        "held_out",
        "heldout",
        "heldout_metadata",
        "judge_rubric",
        "query",
        "query_text",
        "question",
        "question_text",
        "target_fact",
    }
)


def _normalise_key(key: object) -> str:
    return str(key).strip().casefold().replace("-", "_")


def assert_policy_mapping_safe(value: Any, path: str = "policy") -> None:
    """Recursively reject evaluation fields from an untrusted policy mapping."""

    if isinstance(value, MeasurementOnly):
        raise LeakageError(f"{path}: MeasurementOnly cannot enter policy input")
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalised = _normalise_key(key)
            if normalised in _FORBIDDEN_POLICY_KEYS:
                raise LeakageError(f"{path}.{key}: evaluation-only field")
            assert_policy_mapping_safe(child, f"{path}.{key}")
    elif isinstance(value, (tuple, list, set, frozenset)):
        for index, child in enumerate(value):
            assert_policy_mapping_safe(child, f"{path}[{index}]")


@dataclass(frozen=True)
class NativeSignal:
    """An ingest-legal signal available before evaluation."""

    kind: SignalKind
    artifact_ids: tuple[str, ...]
    source: SignalSource

    def __post_init__(self) -> None:
        if not self.artifact_ids or any(not item for item in self.artifact_ids):
            raise ValueError("native signals require non-empty artifact ids")
        allowed = {
            SignalKind.REFERENCE: SignalSource.ACCESS_NATIVE,
            SignalKind.SUPERSEDES: SignalSource.INGEST_NATIVE,
            SignalKind.RETRIEVAL: SignalSource.RETRIEVAL_NATIVE,
        }
        if allowed[self.kind] is not self.source:
            raise LeakageError(
                f"{self.kind.value} requires {allowed[self.kind].value}, "
                f"not {self.source.value}"
            )


@dataclass(frozen=True)
class IngestionRecord:
    """One official ingestion unit."""

    scope_id: str
    artifact_id: str
    version: str
    size_tok: int
    native_signals: tuple[NativeSignal, ...] = ()

    def __post_init__(self) -> None:
        if not self.scope_id or not self.artifact_id or not self.version:
            raise ValueError("scope_id, artifact_id, and version are required")
        if self.size_tok <= 0:
            raise ValueError("size_tok must be positive")
        if any(not isinstance(signal, NativeSignal) for signal in self.native_signals):
            raise LeakageError("native_signals must contain NativeSignal values")


@dataclass(frozen=True)
class MeasurementOnly:
    """Evaluation metadata that is never passed to a replay arm."""

    category: str | None = None
    query_text: str | None = None
    gold_answer: str | None = None
    heldout_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "heldout_metadata",
            MappingProxyType(dict(self.heldout_metadata)),
        )


@dataclass(frozen=True)
class PolicyQueryView:
    """The complete query-time object visible to a policy."""

    scope_id: str
    query_id: str
    after_ingest: int
    native_signals: tuple[NativeSignal, ...] = ()

    def __post_init__(self) -> None:
        if not self.scope_id or not self.query_id:
            raise ValueError("scope_id and query_id are required")
        if self.after_ingest < 0:
            raise ValueError("after_ingest must be non-negative")
        if any(not isinstance(signal, NativeSignal) for signal in self.native_signals):
            raise LeakageError("native_signals must contain NativeSignal values")

    @classmethod
    def from_untrusted(cls, value: Mapping[str, Any]) -> "PolicyQueryView":
        """Build a policy view only after recursively linting the mapping."""

        assert_policy_mapping_safe(value)
        allowed = {"scope_id", "query_id", "after_ingest", "native_signals"}
        unknown = set(value) - allowed
        if unknown:
            raise LeakageError(f"unknown policy fields: {sorted(unknown)}")
        return cls(**value)


@dataclass(frozen=True)
class QueryPoint:
    """A policy-visible query view paired with protected measurements."""

    policy: PolicyQueryView
    measurement: MeasurementOnly = field(default_factory=MeasurementOnly)


@dataclass(frozen=True)
class CapacityCell:
    label: str
    capacity: int
    is_base: bool = False
    legacy_regression: bool = False

    def __post_init__(self) -> None:
        if not self.label or self.capacity <= 0:
            raise ValueError("capacity cells require a label and positive capacity")


def ensure_complete_family_set(families: Iterable[MechanismFamily]) -> None:
    observed = tuple(families)
    expected = tuple(MechanismFamily)
    if len(observed) != len(expected) or set(observed) != set(expected):
        raise ValueError(
            "all four preregistered families must be reported exactly once"
        )
