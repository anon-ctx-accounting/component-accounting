"""Fold canonical events → live ARC state, recommendation queue, gates.

Reuses ``karc.replay.trace._load_registry`` / ``_load_events`` and
``karc.policy.make_policy`` so the fold is byte-for-byte the replay path
(data-model §8; the M3 task's "동일 fold 코드 경로" requirement).

Two gates decide whether demote/archive-class recommendations are emitted:

- **Observation window** (cache-policy §6.4): before 14 days AND 50 tasks of
  history, archive/cold recommendations are suppressed ("사용 빈도 0은 중요도
  0이 아니다"). Pin-suggestions and safety recs (harmful/stale, oversize) are
  NOT suppressed.
- **Coverage gate** (cache-policy §6.5-d): when telemetry coverage (sessions
  that produced ≥1 canonical event, over sessions observed) falls below
  threshold, demote/archive recs freeze so an empty queue is never misread as
  "healthy". This was N/A in the replay suite (V9); the definition here is a
  conservative session-heartbeat proxy, documented as such.

The five review-queue types (product-ux §1.3 F-1) are the only human-approval
recommendations: archive, pin-suggestion, split, cold-transition,
invalidate_or_correct. Promote/demote are never queued (auto, observed via
`why`/`log`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from karc.policy import PolicyConfig, make_policy
from karc.policy.model import iso_to_epoch
from karc.replay.trace import _load_events, _load_registry, resolve_budget
from karc.util import utc_now_iso

# Confirmed K-ARC config (E1-2 parameter suite, confirmed-config.json). c
# (budget) is not part of the confirmed set — it is resolved per-scope below.
CONFIRMED_CONFIG = dict(
    c_pin=0,
    alpha=0.25,
    q_min=0.5,
    n_ghost_min=8,
    k_tail=8,
    w_ref={"loaded_on_demand": 1.0, "read": 1.0, "cited": 1.0, "applied": 1.0},
    half_life_days=14.0,
    p_init_frac=0.0,
    fallback_window_min=30.0,
    harmful_n_corr=2,
    harmful_q_max=1.0 / 3.0,
    q5a_deny_revive_on_gate_fail=False,
    q5a_no_adapt_on_gate_fail=False,
    q7_discovered_ghost_recency=False,
    q8_validated_ghost_recency=False,
    q11_tail_posterior_tiebreak=False,
    without_outcome=False,
    without_validity_gate=False,
    preload_as_hit=False,
    coalesce_tasks=True,
    critical_cold_approval=True,
    split_recommendations=True,
    size_mode="token_cost",
)

DEFAULT_BUDGET = "25pct"
OBS_WINDOW_DAYS = 14.0
OBS_WINDOW_TASKS = 50
COVERAGE_MIN = 0.60
COVERAGE_MIN_SESSIONS = 3  # below this, don't trust the ratio (→ no freeze)

# Path/name patterns that mark likely-critical docs (product-ux §5-2). Not
# suppressed by the observation window — pin-suggestion is valid from day 0.
PIN_PATTERNS = (
    "runbook", "incident", "security", "recovery", "disaster",
    "oncall", "on-call", "postmortem", "post-mortem", "escalation", "sev",
)

# The five review-queue actions and their demote/archive-class membership.
DEMOTE_ARCHIVE_CLASS = {"archive", "cold_transition"}


@dataclass
class RecItem:
    action: str
    artifact_id: str
    name: str
    rule_id: str
    rationale: str
    risk: str
    suppressed_reason: str | None = None  # non-None → held back by a gate
    rec_id: str | None = None  # set once persisted to the recommendations table

    def as_dict(self) -> dict:
        return {
            "rec_id": self.rec_id,
            "action": self.action,
            "artifact_id": self.artifact_id,
            "name": self.name,
            "rule_id": self.rule_id,
            "rationale": self.rationale,
            "risk": self.risk,
            "suppressed_reason": self.suppressed_reason,
        }


@dataclass
class LiveState:
    scope_id: str
    config: PolicyConfig
    policy: object
    registry: dict
    n_events: int
    n_tasks: int
    days_observed: float
    in_observation_window: bool
    coverage: float | None
    coverage_frozen: bool
    total_sessions: int
    sessions_with_events: int
    active: list[RecItem] = field(default_factory=list)
    suppressed: list[RecItem] = field(default_factory=list)

    def empty_queue_reason(self) -> str | None:
        """3-state empty-queue diagnosis (product-ux §2.1). None → non-empty."""
        if self.active:
            return None
        if self.coverage_frozen:
            return "coverage_frozen"
        if self.in_observation_window or self.n_events == 0:
            return "observation_window"
        return "healthy"


def _name(registry, art_id):
    m = registry.get(art_id)
    if m is None:
        return art_id
    return m.path or art_id


def _pin_pattern_hit(name: str) -> str | None:
    low = name.lower()
    for pat in PIN_PATTERNS:
        if pat in low:
            return pat
    return None


def compute(conn, db_path, scope_id, budget=DEFAULT_BUDGET, config_overrides=None):
    """Fold events for ``scope_id`` and synthesize the recommendation queue."""
    registry, _prov = _load_registry(conn, scope_id)
    events = _load_events(conn, scope_id)
    corpus_tokens = sum(m.size_tok for m in registry.values())
    budget_tokens = resolve_budget(budget, max(1, corpus_tokens))
    cfg = PolicyConfig.from_overrides(
        PolicyConfig(c=budget_tokens), **{**CONFIRMED_CONFIG, **(config_overrides or {})}
    )
    policy = make_policy("karc", cfg, registry)
    for e in events:
        policy.on_event(e)

    # ---- observation window (§6.4) ----
    now = iso_to_epoch(utc_now_iso())
    if events:
        days_observed = max(0.0, (now - iso_to_epoch(events[0].occurred_at)) / 86400.0)
    else:
        days_observed = 0.0
    n_tasks = len({e.task_id for e in events if e.task_id})
    in_window = days_observed < OBS_WINDOW_DAYS and n_tasks < OBS_WINDOW_TASKS

    # ---- coverage gate (§6.5-d) — session-heartbeat proxy ----
    total_sessions = conn.execute(
        "SELECT COUNT(*) FROM sessions WHERE scope_id = ?", (scope_id,)
    ).fetchone()[0]
    sessions_with_events = conn.execute(
        "SELECT COUNT(DISTINCT session_id) FROM events "
        "WHERE scope_id = ? AND session_id IS NOT NULL",
        (scope_id,),
    ).fetchone()[0]
    coverage = (
        (sessions_with_events / total_sessions) if total_sessions else None
    )
    coverage_frozen = (
        coverage is not None
        and total_sessions >= COVERAGE_MIN_SESSIONS
        and coverage < COVERAGE_MIN
    )

    ls = LiveState(
        scope_id=scope_id,
        config=cfg,
        policy=policy,
        registry=registry,
        n_events=len(events),
        n_tasks=n_tasks,
        days_observed=days_observed,
        in_observation_window=in_window,
        coverage=coverage,
        coverage_frozen=coverage_frozen,
        total_sessions=total_sessions,
        sessions_with_events=sessions_with_events,
    )
    _build_queue(conn, scope_id, policy, registry, ls)
    return ls


def _build_queue(conn, scope_id, policy, registry, ls: LiveState) -> None:
    gate_reason = None
    if ls.coverage_frozen:
        gate_reason = "coverage_frozen"
    elif ls.in_observation_window:
        gate_reason = "observation_window"

    def add(action, art_id, rule_id, rationale, risk):
        item = RecItem(action, art_id, _name(registry, art_id), rule_id, rationale, risk)
        if action in DEMOTE_ARCHIVE_CLASS and gate_reason is not None:
            item.suppressed_reason = gate_reason
            ls.suppressed.append(item)
        else:
            ls.active.append(item)

    # (1) pin-suggestion — pattern based, never suppressed (§5-2)
    already_pinned = {a for a, m in registry.items() if getattr(m, "pinned", False)}
    for art_id, m in sorted(registry.items()):
        if art_id in already_pinned:
            continue
        pat = _pin_pattern_hit(m.path or "")
        if pat:
            add("pin", art_id, "pin-suggestion:pattern",
                f"matches critical path pattern '{pat}'", "none")

    # (2) archive — non-critical artifacts that fell out of the ghost lists
    #     (policy.cold_candidates). Critical ones surface as cold_transition.
    for art_id in sorted(getattr(policy, "cold_candidates", set())):
        m = registry.get(art_id)
        if m is not None and getattr(m, "critical", False):
            continue
        add("archive", art_id, "cold-candidate",
            "dormant: dropped from the working set and its recency history "
            "(archivable — snapshot preserved, restorable)", "none")

    # (3–5) policy recommendations mapped to the five review types
    action_map = {
        "split": ("split", "none"),
        "cold_transition": ("cold_transition", "high"),
        "invalidate_or_correct": ("invalidate_or_correct", "medium"),
    }
    for rec in getattr(policy, "recommendations", []):
        mapped = action_map.get(rec.action)
        if mapped is None:
            continue  # contradiction_review / unpin_review: not in the 5 types
        action, risk = mapped
        add(action, rec.artifact_id, rec.rule_id, rec.rationale, risk)


def artifact_report(ls: LiveState, conn, artifact_id: str) -> dict | None:
    """`k-arc why` data: 6-component breakdown + transitions + blocked_reason."""
    policy = ls.policy
    m = ls.registry.get(artifact_id)
    if m is None:
        return None
    stat = policy.stat(artifact_id) if hasattr(policy, "stat") else None
    list_name = policy.list_of(artifact_id) if hasattr(policy, "list_of") else None
    if artifact_id in getattr(policy, "pinned_ids", set()):
        list_name = "PINNED"
    q = policy.q(artifact_id) if hasattr(policy, "q") else None
    # recent transitions for this artifact (from the folded policy log)
    transitions = [
        t.as_dict() for t in getattr(policy, "transitions", []) if t.artifact_id == artifact_id
    ][-10:]
    report = {
        "artifact_id": artifact_id,
        "name": m.path or artifact_id,
        "list": list_name,
        "size_tokens": m.size_tok,
        "critical": getattr(m, "critical", False),
        "pinned": getattr(m, "pinned", False),
        "cold": artifact_id in getattr(policy, "cold_candidates", set()),
    }
    if stat is not None:
        report.update({
            "f_score": round(stat.F, 4),
            "n_val": stat.n_val,
            "n_corr": stat.n_corr,
            "q": round(q, 4) if q is not None else None,
            "validity": stat.validity,
            "last_ref_task": stat.last_ref_task,
            "blocked_reason": stat.blocked_reason,
            "discovered_count": stat.discovered_count,
            "preload_count": stat.preload_count,
        })
    report["recent_transitions"] = transitions
    return report
