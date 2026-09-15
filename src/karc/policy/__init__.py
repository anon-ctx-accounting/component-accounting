"""K-ARC policy engine package (M1).

Implements cache-policy--arc-adaptation.md §5 (K-ARC(c)) plus the baseline
policies required by experiment-design--mvp-plan.md §5.1 / FR-R1.
"""

from karc.policy.base import Policy
from karc.policy.config import PolicyConfig
from karc.policy.factory import POLICY_NAMES, make_policy
from karc.policy.karc import InvariantViolation, KArcPolicy
from karc.policy.model import ArtifactMeta, Event, Recommendation, Transition

__all__ = [
    "ArtifactMeta",
    "Event",
    "InvariantViolation",
    "KArcPolicy",
    "POLICY_NAMES",
    "Policy",
    "PolicyConfig",
    "Recommendation",
    "Transition",
    "make_policy",
]
