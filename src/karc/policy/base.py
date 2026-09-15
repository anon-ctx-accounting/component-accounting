"""Policy plugin interface (FR-R1): ``on_event(event) → 상태``.

Every policy — K-ARC, its ablation variants, and the classic baselines —
implements this interface so the replay runner treats them uniformly
(§8.4 fairness: same budget, same event stream).
"""

from __future__ import annotations

import hashlib
import json

from karc.policy.model import ArtifactMeta, Event, Recommendation, Transition


class Policy:
    """Deterministic (I9) event-fold policy."""

    name: str = "abstract"

    def __init__(self, registry: dict[str, ArtifactMeta]):
        self.registry = registry
        self.transitions: list[Transition] = []
        self.recommendations: list[Recommendation] = []
        self._seen_events: set[str] = set()

    # ---- core contract -----------------------------------------------------
    def on_event(self, e: Event) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def working_set(self) -> dict[str, int]:
        """Supplied set at this instant: artifact_id → accounted tokens.

        For ARC-family policies this is T1∪T2∪Pinned; for the classics it is
        their resident set. Ghosts/OVERSIZE are metadata, never supplied.
        """
        raise NotImplementedError  # pragma: no cover - abstract

    def is_resident(self, artifact_id: str) -> bool:
        """Membership test for hit accounting (metric layer)."""
        return artifact_id in self.working_set()

    # ---- shared helpers ----------------------------------------------------
    def dedup_insert(self, event_id: str) -> bool:
        """Idempotency under at-least-once delivery (§5.2)."""
        if event_id in self._seen_events:
            return False
        self._seen_events.add(event_id)
        return True

    def resolved_size(self, artifact_id: str, size_mode: str = "token_cost") -> int:
        meta = self.registry.get(artifact_id)
        if meta is None:
            # Untracked artifact (e.g. real-trace artifact without metadata):
            # register a provisional entry with the corpus-median default.
            meta = ArtifactMeta(artifact_id=artifact_id, size_tok=800)
            self.registry[artifact_id] = meta
        return meta.resolved_size(size_mode)

    def state_hash(self) -> str:
        """FR-R2 reproduction hash over the canonical state serialization."""
        payload = json.dumps(self._state_payload(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _state_payload(self) -> dict:  # pragma: no cover - abstract
        raise NotImplementedError

    def summary(self) -> dict:
        """Counter snapshot for the run summary. Policies extend this."""
        recs: dict[str, int] = {}
        for r in self.recommendations:
            recs[r.action] = recs.get(r.action, 0) + 1
        return {
            "policy": self.name,
            "transitions": len(self.transitions),
            "recommendations": recs,
        }
