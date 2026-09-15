"""Policy factory — maps CLI/runner names to configured policy instances."""

from __future__ import annotations

from karc.policy.baselines import (
    FIFOPolicy,
    LFUPolicy,
    LRUPolicy,
    StaticPreloadPolicy,
    plain_arc_config,
)
from karc.policy.config import PolicyConfig
from karc.policy.karc import KArcPolicy
from karc.policy.model import ArtifactMeta

POLICY_NAMES = (
    "karc",
    "karc-no-outcome",  # #9 ablation (E1-5)
    "karc-no-validity-gate",  # A2/#10 ablation
    "karc-preload-as-hit",  # V4 negative control
    "arc",  # plain ARC (원 논문, token-budget 계량 동일)
    "static",
    "fifo",
    "lru",
    "lfu",
)


def make_policy(name: str, config: PolicyConfig, registry: dict[str, ArtifactMeta]):
    name = name.strip().lower()
    if name == "karc":
        return KArcPolicy(config, registry)
    if name == "karc-no-outcome":
        p = KArcPolicy(config.replace(without_outcome=True), registry)
        p.name = name
        return p
    if name == "karc-no-validity-gate":
        p = KArcPolicy(config.replace(without_validity_gate=True), registry)
        p.name = name
        return p
    if name == "karc-preload-as-hit":
        p = KArcPolicy(config.replace(preload_as_hit=True), registry)
        p.name = name
        return p
    if name == "arc":
        p = KArcPolicy(plain_arc_config(config), registry)
        p.name = "arc"
        return p
    if name == "static":
        return StaticPreloadPolicy(config, registry)
    if name == "fifo":
        return FIFOPolicy(config, registry)
    if name == "lru":
        return LRUPolicy(config, registry)
    if name == "lfu":
        return LFUPolicy(config, registry)
    raise ValueError(f"unknown policy {name!r}; known: {', '.join(POLICY_NAMES)}")
