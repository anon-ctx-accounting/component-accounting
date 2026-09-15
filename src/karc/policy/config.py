"""Policy configuration — every §5.1 parameter plus the Q-switches and the
ablation flags E1-2 sweeps (Q1~Q9/Q11) and E1-5/E1-3 toggle.

All fields are plain data so a config is JSON-serializable and hashes into
the run's reproduction stamp (FR-R9).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields

POLICY_VERSION = "karc-policy-0.1.0"

DEFAULT_W_REF = {
    "loaded_on_demand": 0.5,
    "read": 1.0,
    "cited": 1.5,
    "applied": 2.0,
}


@dataclass
class PolicyConfig:
    # ---- capacity (§5.1) ----
    c: int = 8000  # adaptive token budget (T1+T2 resident cap)
    c_pin: int = 0  # pinned-only budget (metering/warning only, §4.3)
    alpha: float = 0.25  # oversize threshold: size_tok > alpha*c → no residency
    q_min: float = 0.5  # promotion gate threshold (Q4)
    n_ghost_min: int = 8  # ghost entry-count floor (§3.5, Q2)
    k_tail: int = 8  # validity-aware eviction tail window (entries)
    w_ref: dict = field(default_factory=lambda: dict(DEFAULT_W_REF))  # Q1
    half_life_days: float = 14.0  # F exponential decay
    p_init_frac: float = 0.0  # Q6: p0 = p_init_frac * c (0 = original ARC)
    fallback_window_min: float = 30.0  # Q3: task_id-missing coalescing window
    harmful_n_corr: int = 2  # harmful(a) = n_corr >= this and q < harmful_q_max
    harmful_q_max: float = 1.0 / 3.0

    # ---- open-question switches (E1-2 grid) ----
    q5a_deny_revive_on_gate_fail: bool = False  # Q5a: True → deny revival (stay ghost)
    # Q5a 3rd option (Amendment A-3(b), E1-2e "deny-no-adapt"): on a
    # gate-failed ghost hit, additionally skip the p adaptation — probes the
    # concern that repeated hits on harmful ghosts pollute p. Only meaningful
    # together with q5a_deny_revive_on_gate_fail=True; ignored otherwise.
    q5a_no_adapt_on_gate_fail: bool = False
    q7_discovered_ghost_recency: bool = False  # Q7: discovered refreshes ghost recency
    q8_validated_ghost_recency: bool = False  # Q8: validated refreshes ghost recency
    q11_tail_posterior_tiebreak: bool = False  # Q11: deterministic q-ranking in tail window

    # ---- Q12 shift-relief p-floor (Amendment A-6, conditional dev-seed) ----
    # Detection: over the trailing ``q12_window`` tasks, if the summed B1
    # ghost-hit accounted-token mass ≥ ``q12_theta``·c, boost p ← max(p, c/2)
    # (re-appliable, capped at c). Default off — flag-gated so the confirmed
    # config and every prior experiment stay byte-identical. Logged with the
    # detection basis (contributing ghost-hit artifacts).
    q12_shift_relief: bool = False
    q12_window: int = 5
    q12_theta: float = 0.10

    # ---- ablation / variant flags (FR-R1) ----
    without_outcome: bool = False  # #9: ignore validated/corrected effects
    without_validity_gate: bool = False  # A2/#10: validity plays no policy role
    preload_as_hit: bool = False  # V4 negative control

    # ---- structural switches (baseline presets; not part of the Q grid) ----
    coalesce_tasks: bool = True  # False → plain-ARC immediate T1→T2 on 2nd hit
    critical_cold_approval: bool = True  # False → plain policies auto-transition
    split_recommendations: bool = True  # plain ARC has no advisory layer

    # ---- runner/debug ----
    debug_invariants: bool = True  # FR-R6: per-event I1~I9 assertions
    size_mode: str = "token_cost"  # FR-R5: token_cost | bytes (Q9 forced fallback)

    def __post_init__(self) -> None:
        if self.c <= 0:
            raise ValueError("c must be positive")
        if not (0 < self.alpha <= 1.0):
            raise ValueError("alpha must be in (0, 1]")
        if self.size_mode not in ("token_cost", "bytes"):
            raise ValueError(f"unknown size_mode {self.size_mode!r}")

    # -- helpers -------------------------------------------------------------
    def as_dict(self) -> dict:
        return asdict(self)

    def canonical_json(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True, ensure_ascii=False)

    def replace(self, **kw) -> "PolicyConfig":
        d = self.as_dict()
        d.update(kw)
        return PolicyConfig(**d)

    @classmethod
    def from_overrides(cls, base: "PolicyConfig | None" = None, **kw) -> "PolicyConfig":
        cfg = base or cls()
        valid = {f.name for f in fields(cls)}
        unknown = set(kw) - valid
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cfg.replace(**kw)
