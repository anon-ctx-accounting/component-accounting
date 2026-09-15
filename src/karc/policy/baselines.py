"""Baseline policies (FR-R1, experiment-design §5.1 tier 1+2).

All baselines consume the same event stream as K-ARC and are metered in the
same token budget c (§8.4 fairness). Classic policies react to the reference
track only — ignoring outcome/validity events is their definition, not an
unfairness (§8.4-2). All are deterministic (I9) and expose ``state_hash``.

Plain ARC is built as a preset of the K-ARC engine (see ``plain_arc_config``)
so that token metering — acct(), ghost bounds, N_GHOST_MIN floor — is
identical by construction ("token-budget 계량은 K-ARC와 동일 조건", INT-10):
coalescing/gates/validity/tail-window/advisory layer are switched off, which
reduces the engine to original-ARC semantics in token form.
"""

from __future__ import annotations

from collections import OrderedDict

from karc.policy.base import Policy
from karc.policy.config import PolicyConfig
from karc.policy.model import ArtifactMeta, Event


def plain_arc_config(cfg: PolicyConfig) -> PolicyConfig:
    """K-ARC engine preset that degenerates to original ARC (token-budget)."""
    return cfg.replace(
        coalesce_tasks=False,  # 2nd hit in T1 promotes immediately (원 논문)
        without_outcome=True,
        without_validity_gate=True,
        alpha=1.0,  # only the physical constraint size_tok > c blocks residency
        k_tail=1,  # tail window of 1 = pure LRU victim
        split_recommendations=False,
        critical_cold_approval=False,  # plain ARC has no safety layer
        q5a_deny_revive_on_gate_fail=False,
        q7_discovered_ghost_recency=False,
        q8_validated_ghost_recency=False,
        q11_tail_posterior_tiebreak=False,
        preload_as_hit=False,
    )


class StaticPreloadPolicy(Policy):
    """Baseline 1: the preload set, truncated deterministically to budget c.

    Truncation rule (§8.4-1): order by directory depth ascending, then file
    name lexicographic; take the maximal prefix whose token sum fits c
    ("절단" = prefix cut — an artifact that does not fit stops the fill).
    """

    name = "static"

    def __init__(self, config: PolicyConfig, registry: dict[str, ArtifactMeta]):
        super().__init__(registry)
        self.config = config
        ordered = sorted(
            (m for m in registry.values() if m.preload),
            key=lambda m: (m.path.count("/"), m.path, m.artifact_id),
        )
        self._ws: dict[str, int] = {}
        used = 0
        for m in ordered:
            size = m.resolved_size(config.size_mode)
            if used + size > config.c:
                break
            self._ws[m.artifact_id] = size
            used += size

    def on_event(self, e: Event) -> None:
        self.dedup_insert(e.event_id)  # static: no dynamics at all

    def working_set(self) -> dict[str, int]:
        return dict(self._ws)

    def _state_payload(self) -> dict:
        return {"policy": self.name, "ws": sorted(self._ws.items()), "c": self.config.c}


class _ResidentSetPolicy(Policy):
    """Shared machinery for FIFO/LRU/LFU: token-budgeted resident dict."""

    def __init__(self, config: PolicyConfig, registry: dict[str, ArtifactMeta]):
        super().__init__(registry)
        self.config = config
        self.resident: OrderedDict[str, int] = OrderedDict()  # order defined per policy
        self.counters = {
            "admissions": 0,
            "evictions": 0,
            "hits": 0,
            "rejected_oversize": 0,
            # E1-3 panel metric ("critical eviction count"): classic policies
            # have no safety layer, so an eviction of a critical artifact is a
            # loss. Counting only — no behavioral change.
            "critical_evictions": 0,
        }

    def on_event(self, e: Event) -> None:
        if not self.dedup_insert(e.event_id):
            return
        if not e.is_reference():
            return
        a = e.artifact_id
        size = self.resolved_size(a, self.config.size_mode)
        if a in self.resident:
            self.counters["hits"] += 1
            self._on_hit(a)
            return
        if size > self.config.c:
            self.counters["rejected_oversize"] += 1
            return  # cannot ever fit — physical constraint, logged
        while self.resident and self._tok() + size > self.config.c:
            victim = self._pick_victim()
            self.resident.pop(victim)
            self.counters["evictions"] += 1
            vm = self.registry.get(victim)
            if vm is not None and vm.critical:
                self.counters["critical_evictions"] += 1
        self._admit(a, size)
        self.counters["admissions"] += 1

    def _tok(self) -> int:
        return sum(self.resident.values())

    def working_set(self) -> dict[str, int]:
        return dict(self.resident)

    def _state_payload(self) -> dict:
        return {
            "policy": self.name,
            "resident": [[a, t] for a, t in self.resident.items()],
            "extra": self._extra_state(),
            "c": self.config.c,
        }

    def summary(self) -> dict:
        out = super().summary()
        out.update(self.counters)
        return out

    # hooks -------------------------------------------------------------
    def _on_hit(self, a: str) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def _admit(self, a: str, size: int) -> None:
        self.resident[a] = size  # appended at the end

    def _pick_victim(self) -> str:  # pragma: no cover - abstract
        raise NotImplementedError

    def _extra_state(self) -> list:
        return []


class FIFOPolicy(_ResidentSetPolicy):
    """Insertion-order eviction; hits do not reorder (SOLAR diagnostic)."""

    name = "fifo"

    def _on_hit(self, a: str) -> None:
        pass  # FIFO: access does not change order

    def _pick_victim(self) -> str:
        return next(iter(self.resident))  # oldest insertion


class LRUPolicy(_ResidentSetPolicy):
    name = "lru"

    def _on_hit(self, a: str) -> None:
        self.resident.move_to_end(a)  # end = MRU

    def _pick_victim(self) -> str:
        return next(iter(self.resident))  # front = LRU


class LFUPolicy(_ResidentSetPolicy):
    """Exact frequency counts (no decay); ties broken by LRU order."""

    name = "lfu"

    def __init__(self, config: PolicyConfig, registry: dict[str, ArtifactMeta]):
        super().__init__(config, registry)
        self.freq: dict[str, int] = {}

    def _on_hit(self, a: str) -> None:
        self.freq[a] = self.freq.get(a, 0) + 1
        self.resident.move_to_end(a)  # recency for tie-break

    def _admit(self, a: str, size: int) -> None:
        super()._admit(a, size)
        self.freq[a] = self.freq.get(a, 0) + 1

    def _pick_victim(self) -> str:
        # min frequency; ties → least recently used (front-most in order)
        order = {a: i for i, a in enumerate(self.resident)}
        return min(self.resident, key=lambda x: (self.freq.get(x, 0), order[x]))

    def _extra_state(self) -> list:
        return sorted(self.freq.items())
