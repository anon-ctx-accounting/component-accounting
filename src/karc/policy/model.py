"""Shared value types for policies and the replay runner.

Event model = the 8 canonical telemetry event types (data-model §3.3) plus
one replay-only meta input ``validity_changed``: in the live system validity
transitions (STALE via supersede / valid_to) flow from the registry
(cache-policy §4.2 "Staleness 판정 자체는 Policy Engine이 하지 않는다 —
bi-temporal metadata를 읽기만 한다"), so the replay stream carries them as an
explicit input item rather than as one of the 8 telemetry events.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

REFERENCE_EVENT_TYPES = ("loaded", "read", "cited", "applied")
QUALITY_EVENT_TYPES = ("validated", "corrected", "conflicted")
EVENT_TYPES = (
    "discovered",
    *REFERENCE_EVENT_TYPES,
    *QUALITY_EVENT_TYPES,
    "validity_changed",  # replay-only registry feed (see module docstring)
)

VALIDITIES = ("VALID", "SUSPECT", "STALE", "INVALIDATED")


def iso_to_epoch(ts: str) -> float:
    """ISO8601 (Z suffix ok) → epoch seconds. Deterministic."""
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


@dataclass(frozen=True)
class Event:
    """One policy input. ``load_class`` follows D2 (preload|policy|on_demand)."""

    event_id: str
    event_type: str
    occurred_at: str  # ISO8601; canonical order = (occurred_at, event_id)
    artifact_id: str
    task_id: str | None = None
    session_id: str | None = None
    scope_id: str | None = None
    load_class: str | None = None  # loaded only: preload | policy | on_demand
    token_cost: int | None = None
    new_validity: str | None = None  # validity_changed only
    origin: str = "script"  # script | outcome-injector | trace

    @property
    def epoch(self) -> float:
        return iso_to_epoch(self.occurred_at)

    def is_reference(self, preload_as_hit: bool = False) -> bool:
        """Reference track membership per cache-policy §2.1/§2.2.

        ``preload_as_hit`` is the V4 negative-control ablation: it moves
        loaded(preload|policy) into the reference track (the design's own
        derivation §2.2 predicts this degenerates to static preload).
        """
        if self.event_type in ("read", "cited", "applied"):
            return True
        if self.event_type == "loaded":
            if self.load_class == "on_demand":
                return True
            return preload_as_hit
        return False

    def reference_key(self) -> str:
        """W_REF key for a reference-track event."""
        if self.event_type == "loaded":
            # preload-as-hit ablation reuses the weakest weight (0.5); the
            # design assigns no weight to preload loads, so the weakest
            # reference weight is the conservative stand-in.
            return "loaded_on_demand"
        return self.event_type


@dataclass
class ArtifactMeta:
    """Registry view the policies read (size/critical/pinned are static in
    replay; validity is dynamic and lives inside the policy state)."""

    artifact_id: str
    size_tok: int
    path: str = ""
    critical: bool = False
    pinned: bool = False
    preload: bool = False
    language: str = "en"
    tags: tuple[str, ...] = ()
    byte_size: int | None = None  # for the Q9 size-fallback chain

    def resolved_size(self, mode: str = "token_cost") -> int:
        """FR-R5 size fallback chain: token_cost → (tokenizer: none in a
        stdlib-only build) → bytes/4, ko bytes/2.5. ``mode='bytes'`` forces
        the byte heuristic even when token_cost exists (Q9 forced-fallback).
        """
        if mode == "bytes":
            b = self.byte_size
            if b is None:  # reconstruct bytes from tokens with the same ratio
                b = int(self.size_tok * (2.5 if self.language == "ko" else 4))
            divisor = 2.5 if self.language == "ko" else 4.0
            return max(1, int(b / divisor))
        return max(1, int(self.size_tok))


@dataclass
class Transition:
    """One explainable list transition (V11: rule id + evidence + p)."""

    seq: int
    artifact_id: str
    from_list: str | None
    to_list: str | None
    rule_id: str
    evidence_event_id: str | None
    p_before: float | None = None
    p_after: float | None = None

    def as_dict(self) -> dict:
        return {
            "seq": self.seq,
            "artifact_id": self.artifact_id,
            "from": self.from_list,
            "to": self.to_list,
            "rule_id": self.rule_id,
            "evidence": self.evidence_event_id,
            "p_before": self.p_before,
            "p_after": self.p_after,
        }


@dataclass
class Recommendation:
    """Advisory output (never auto-executed; R-7 recommendation-only)."""

    action: str  # split | invalidate_or_correct | cold_transition | contradiction_review | unpin_review
    artifact_id: str
    rule_id: str
    rationale: str
    requires_approval: bool = True
    evidence: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "action": self.action,
            "artifact_id": self.artifact_id,
            "rule_id": self.rule_id,
            "rationale": self.rationale,
            "requires_approval": self.requires_approval,
            "evidence": list(self.evidence),
        }
