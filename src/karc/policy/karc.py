"""K-ARC(c) policy engine — faithful implementation of cache-policy §5.

Pseudocode fidelity: function names/branching mirror §5.2 (``on_event``),
§5.3 (``on_reference`` / ``ghost_hit`` / ``cache_miss`` / ``evict_until_fits``
/ ``pick_victim_tail`` / ``enforce_ghost_bounds`` / ``to_cold`` /
``on_negative_quality``). Invariants I1~I9 (§5.4) are runtime assertions
(FR-R6) checked after every processed event when ``config.debug_invariants``.

Interpretation decisions (ambiguities resolved conservatively; also listed in
the completion report — no silent redesign):

INT-1  ``evict_until_fits``: when the selected source list is empty the other
       list is used; if both resident lists are empty the loop stops (only
       reachable because acct(a) ≤ α·c ≤ c). Fallbacks are logged with a
       distinct rule id.
INT-2  Directory-leaving paths other than ghost truncation — the rare
       Case IV-A direct T1-tail delete (original ARC's "B1 empty" path) —
       also route through ``to_cold`` (§3.5 "ghost 탈락 → COLD_CANDIDATE"
       extended conservatively; preserves I8's critical protection there
       too). The direct path also runs when the IV-A ghost trim exhausts B1
       without making room (token sizes make this reachable, unlike the
       count-based original) so that I3 stays tight.
INT-3  "상시 split recommendation" (§3.2): one pending recommendation per
       (action, artifact); repeats do not re-issue while pending.
INT-4  SUSPECT (conflicted) is NOT auto-cleared by ``validated``: validity's
       truth source is the registry (§4.2), so only an explicit
       ``validity_changed`` input resolves it.
INT-5  Ghost hits revive regardless of the coalescing new-task test — the
       §5.3 pseudocode takes every reference-track event on a ghost as a
       ghost hit; §2.4's "자동으로 서로 다른 task" note is treated as
       justification, not an extra condition.
INT-6  Q5a=deny: the ghost hit still adapts p and refreshes the ghost's MRU
       position, but no revival occurs (the demand happened; only residency
       is denied).
INT-7  Q11 tie-break posterior = q(a), i.e. Beta(1+n_val, 1+n_corr) mean —
       the only deterministic posterior the engine maintains (§10.4 requires
       determinism; SOLAR's reuse posterior needs signals K-ARC rejects).
INT-8  task_id-missing fallback (Q3): a reference with task_id=None counts as
       a new distinct task iff the artifact's previous reference happened in
       a different session OR ≥ 30min earlier (rolling window).
INT-9  Budget change (§6.6): residents that became oversize under the new c
       move to the OVERSIZE side table; ghost entries that became oversize
       are dropped via ``to_cold`` (oversize must not sit in B, §3.5); then
       p is clamped and bounds re-enforced.
INT-11 c_pin is metering only: exceeding it issues a single ``unpin_review``
       recommendation, never an automatic demotion (안전 원칙 10.4).
INT-16 ``on_negative_quality``/STALE demotion insert into B1 and then call
       ``enforce_ghost_bounds`` so I3/I4 hold at event end (§5.3 omits the
       call; §5.4 requires the invariant at every event boundary).
INT-17 I7 is enforced per the doc's own timing qualifier ("quality 이벤트
       처리 직후 기준"), read per-artifact — see ``check_invariants``.
INT-18 §5.3's Case IV-B ``elif`` is generalized to an independent check: the
       original's exclusivity is a count-arithmetic property that variable
       token sizes break (IV-A's direct T1 delete can free fewer tokens
       than the admission needs, leaving the directory over 2c). I4 wins —
       see ``cache_miss``.
"""

from __future__ import annotations

from collections import OrderedDict

from karc.policy.base import Policy
from karc.policy.config import POLICY_VERSION, PolicyConfig
from karc.policy.model import ArtifactMeta, Event, Recommendation, Transition


class InvariantViolation(AssertionError):
    """I1~I9 violation. Carries reproduction info (FR-R6)."""

    def __init__(self, rule: str, detail: str, event: Event | None, state: dict):
        super().__init__(f"{rule}: {detail}")
        self.rule = rule
        self.detail = detail
        self.event = event
        self.state = state

    def dump(self) -> dict:
        return {
            "rule": self.rule,
            "detail": self.detail,
            "event": None if self.event is None else self.event.__dict__,
            "state": self.state,
        }


class _ArtStat:
    """Per-artifact dynamic metadata (§5.1 per-artifact block)."""

    __slots__ = (
        "F",
        "f_updated_epoch",
        "n_val",
        "n_corr",
        "validity",
        "last_ref_task",
        "last_ref_epoch",
        "last_ref_session",
        "blocked_reason",
        "discovered_count",
        "preload_count",
    )

    def __init__(self) -> None:
        self.F = 0.0
        self.f_updated_epoch: float | None = None
        self.n_val = 0
        self.n_corr = 0
        self.validity = "VALID"
        self.last_ref_task: str | None = None
        self.last_ref_epoch: float | None = None
        self.last_ref_session: str | None = None
        self.blocked_reason: str | None = None
        self.discovered_count = 0
        self.preload_count = 0


class KArcPolicy(Policy):
    """Scope-level K-ARC(c) instance (§5.6: one instance per scope)."""

    name = "karc"
    policy_version = POLICY_VERSION

    def __init__(self, config: PolicyConfig, registry: dict[str, ArtifactMeta]):
        super().__init__(registry)
        self.config = config
        # ---- ARC state (§5.1). OrderedDict: first item = MRU. ----
        self.t1: OrderedDict[str, int] = OrderedDict()  # id -> acct tokens
        self.t2: OrderedDict[str, int] = OrderedDict()
        self.b1: OrderedDict[str, int] = OrderedDict()
        self.b2: OrderedDict[str, int] = OrderedDict()
        self.oversize: dict[str, int] = {}  # side table: id -> observed refs
        self.p: float = config.p_init_frac * config.c
        self.stats: dict[str, _ArtStat] = {}
        self.pinned_ids: set[str] = {a for a, m in registry.items() if m.pinned}
        # lifecycle bookkeeping (registry metadata is永久 — we only tag)
        self.cold_candidates: set[str] = set()
        self.cold_ever: set[str] = set()  # for archive-regret metrics
        self._pending_recs: set[tuple[str, str]] = set()
        self._seq = 0
        # ---- Q12 shift-relief bookkeeping (Amendment A-6, flag-gated) ----
        self._q12_task: str | None = None
        self._q12_cur_acct: int = 0
        self._q12_cur_arts: list[str] = []
        self._q12_window: list[tuple[int, list[str]]] = []  # (acct_sum, arts)
        self._q12_boosts: int = 0
        # counters
        self.counters = {
            "ghost_hits": 0,
            "ghost_hits_b1": 0,
            "ghost_hits_b2": 0,
            "promotions_t1_t2": 0,
            "revivals_t2": 0,
            "revivals_quarantine_t1": 0,
            "revivals_denied": 0,
            "admissions": 0,
            "evictions": 0,
            "critical_resident_evictions": 0,
            "cold_auto": 0,
            "cold_auto_critical": 0,  # must stay 0 when critical_cold_approval
            "restores_from_cold": 0,
            "gate_blocks": 0,
            "demotions_t2_t1": 0,
            "demotions_to_ghost": 0,
            "dedup_skipped": 0,
        }
        self._warned_pin_budget = False

    # ======================================================================
    # helpers (§5.1)
    # ======================================================================
    def size_tok(self, a: str) -> int:
        return self.resolved_size(a, self.config.size_mode)

    def acct(self, a: str) -> int:
        return min(self.size_tok(a), int(self.config.alpha * self.config.c))

    def _tok(self, lst: OrderedDict[str, int]) -> int:
        return sum(lst.values())

    def tok_t1(self) -> int:
        return self._tok(self.t1)

    def tok_t2(self) -> int:
        return self._tok(self.t2)

    def tok_b1(self) -> int:
        return self._tok(self.b1)

    def tok_b2(self) -> int:
        return self._tok(self.b2)

    def tok_all(self) -> int:
        return self.tok_t1() + self.tok_t2() + self.tok_b1() + self.tok_b2()

    def stat(self, a: str) -> _ArtStat:
        s = self.stats.get(a)
        if s is None:
            s = _ArtStat()
            self.stats[a] = s
        return s

    def q(self, a: str) -> float:
        s = self.stat(a)
        if self.config.without_outcome:
            return 0.5  # Laplace prior — outcome signals are ignored (#9)
        return (1 + s.n_val) / (2 + s.n_val + s.n_corr)

    def harmful(self, a: str) -> bool:
        if self.config.without_outcome:
            return False
        s = self.stat(a)
        return s.n_corr >= self.config.harmful_n_corr and self.q(a) < self.config.harmful_q_max

    def is_oversize(self, a: str) -> bool:
        return self.size_tok(a) > self.config.alpha * self.config.c

    def validity(self, a: str) -> str:
        return self.stat(a).validity

    def gate_promote(self, a: str) -> tuple[bool, str]:
        """(passes, rule) — rule explains the first failing condition (§4.1)."""
        if not self.config.without_validity_gate and self.validity(a) != "VALID":
            return False, f"gate:validity={self.validity(a)}"
        if self.q(a) < self.config.q_min:
            return False, f"gate:q={self.q(a):.3f}<{self.config.q_min}"
        if self.is_oversize(a):
            return False, "gate:oversize"
        if self.harmful(a):
            return False, "gate:harmful"
        return True, "gate:pass"

    def _log(
        self,
        a: str,
        from_list: str | None,
        to_list: str | None,
        rule: str,
        evidence: str | None,
        p_before: float | None = None,
    ) -> None:
        self._seq += 1
        self.transitions.append(
            Transition(
                seq=self._seq,
                artifact_id=a,
                from_list=from_list,
                to_list=to_list,
                rule_id=f"{self.policy_version}:{rule}",
                evidence_event_id=evidence,
                p_before=p_before,
                p_after=self.p,
            )
        )

    def _recommend(
        self, action: str, a: str, rule: str, rationale: str, requires_approval: bool = True
    ) -> None:
        key = (action, a)
        if key in self._pending_recs:
            return  # INT-3: keep one pending recommendation per (action, artifact)
        self._pending_recs.add(key)
        self.recommendations.append(
            Recommendation(
                action=action,
                artifact_id=a,
                rule_id=f"{self.policy_version}:{rule}",
                rationale=rationale,
                requires_approval=requires_approval,
            )
        )

    def list_of(self, a: str) -> str | None:
        if a in self.t1:
            return "T1"
        if a in self.t2:
            return "T2"
        if a in self.b1:
            return "B1"
        if a in self.b2:
            return "B2"
        if a in self.oversize:
            return "OVERSIZE"
        return None

    def working_set(self) -> dict[str, int]:
        ws = dict(self.t2)
        ws.update(self.t1)
        for a in sorted(self.pinned_ids):
            ws[a] = self.size_tok(a)
        return ws

    def is_resident(self, artifact_id: str) -> bool:
        return (
            artifact_id in self.t1
            or artifact_id in self.t2
            or artifact_id in self.pinned_ids
        )

    # ======================================================================
    # §5.2 on_event (top level)
    # ======================================================================
    def on_event(self, e: Event) -> None:
        if not self.dedup_insert(e.event_id):  # at-least-once idempotency
            self.counters["dedup_skipped"] += 1
            return
        a = e.artifact_id  # registry.resolve: identity (no splits in M1, R-7)
        try:
            self._handle(a, e)
        finally:
            if self.config.debug_invariants:
                self.check_invariants(e)

    def _handle(self, a: str, e: Event) -> None:
        cfg = self.config
        s = self.stat(a)

        if e.event_type == "validity_changed":
            self.on_validity_change(a, e.new_validity or "VALID", e)
            return

        if a in self.pinned_ids:
            self._record_stats_only(a, e)  # pin is outside the ARC lists (§4.3)
            return

        if e.event_type == "validated":
            if not cfg.without_outcome:
                s.n_val += 1
            if cfg.q8_validated_ghost_recency:
                self._refresh_ghost_recency(a, e, "q8:validated-ghost-recency")
            return
        if e.event_type == "corrected":
            if cfg.without_outcome:
                return
            s.n_corr += 1
            self.on_negative_quality(a, e)
            return
        if e.event_type == "conflicted":
            # validity=SUSPECT freezes promotion until resolved (INT-4)
            if not cfg.without_validity_gate:
                s.validity = "SUSPECT"
                self._recommend(
                    "contradiction_review",
                    a,
                    "conflicted",
                    "conflicted event: validity=SUSPECT until human review",
                )
            return
        if e.event_type == "discovered":
            s.discovered_count += 1  # funnel only — not a hit (§2.2)
            if cfg.q7_discovered_ghost_recency:
                self._refresh_ghost_recency(a, e, "q7:discovered-ghost-recency")
            return
        if e.event_type == "loaded" and e.load_class in ("preload", "policy"):
            s.preload_count += 1  # funnel only — not a hit (§2.2)
            if not cfg.preload_as_hit:
                return
            # fall through: V4 negative-control ablation treats this as a hit

        if not e.is_reference(cfg.preload_as_hit):
            return

        if cfg.q12_shift_relief:
            self._q12_on_reference(e)  # task-boundary shift detection (A-6)

        # ---- reference track: loaded(on_demand) | read | cited | applied ----
        w = cfg.w_ref.get(e.reference_key(), 1.0)
        s.F = s.F * self._decay_factor(s, e.epoch) + w
        s.f_updated_epoch = e.epoch
        new_task = self._is_new_task(s, e)
        s.last_ref_task = e.task_id
        s.last_ref_epoch = e.epoch
        s.last_ref_session = e.session_id
        self.on_reference(a, e, new_task)

    def _decay_factor(self, s: _ArtStat, now_epoch: float) -> float:
        if s.f_updated_epoch is None:
            return 0.0
        dt_days = max(0.0, (now_epoch - s.f_updated_epoch) / 86400.0)
        return 0.5 ** (dt_days / self.config.half_life_days)

    def _is_new_task(self, s: _ArtStat, e: Event) -> bool:
        if not self.config.coalesce_tasks:
            return True  # plain-ARC preset: every reference is distinct
        if e.task_id is not None:
            return e.task_id != s.last_ref_task
        # INT-8 / Q3 fallback: (runtime, session, 30-min rolling window)
        if s.last_ref_epoch is None:
            return True
        if e.session_id is not None and e.session_id != s.last_ref_session:
            return True
        return (e.epoch - s.last_ref_epoch) >= self.config.fallback_window_min * 60.0

    def _q12_on_reference(self, e: Event) -> None:
        """A-6 shift-relief: at each task boundary, if the summed B1 ghost-hit
        accounted-token mass over the trailing ``q12_window`` tasks reaches
        ``q12_theta``·c, floor p at c/2 (``p ← max(p, c/2)``, capped at c).
        Detection is keyed on reference events (where task work happens); the
        just-completed task is included in the trailing window."""
        cfg = self.config
        tid = e.task_id
        if tid is None or tid == self._q12_task:
            return
        if self._q12_task is not None:  # close the just-completed task's bucket
            self._q12_window.append((self._q12_cur_acct, self._q12_cur_arts))
            if len(self._q12_window) > cfg.q12_window:
                self._q12_window.pop(0)
            mass = sum(acct for acct, _ in self._q12_window)
            if mass >= cfg.q12_theta * cfg.c and self.p < cfg.c / 2.0:
                p_before = self.p
                self.p = min(float(cfg.c), cfg.c / 2.0)
                self._q12_boosts += 1
                contrib = [x for _, arts in self._q12_window for x in arts]
                self._log(
                    e.artifact_id, None, None,
                    f"q12:shift-relief-p-floor(mass={mass},w={len(self._q12_window)},"
                    f"contrib={contrib[:8]})",
                    e.event_id, p_before,
                )
        self._q12_task = tid
        self._q12_cur_acct = 0
        self._q12_cur_arts = []

    def _record_stats_only(self, a: str, e: Event) -> None:
        """Pinned artifacts: collect statistics, never move lists (§4.3)."""
        s = self.stat(a)
        if e.event_type == "validated" and not self.config.without_outcome:
            s.n_val += 1
        elif e.event_type == "corrected" and not self.config.without_outcome:
            s.n_corr += 1
        elif e.event_type == "discovered":
            s.discovered_count += 1
        elif e.is_reference(self.config.preload_as_hit):
            w = self.config.w_ref.get(e.reference_key(), 1.0)
            s.F = s.F * self._decay_factor(s, e.epoch) + w
            s.f_updated_epoch = e.epoch
        # INT-11: c_pin is metering only — warn once via recommendation
        if self.config.c_pin and not self._warned_pin_budget:
            pin_tok = sum(self.size_tok(x) for x in self.pinned_ids)
            if pin_tok > self.config.c_pin:
                self._warned_pin_budget = True
                self._recommend(
                    "unpin_review",
                    a,
                    "pin-budget-exceeded",
                    f"tok(pinned)={pin_tok} > c_pin={self.config.c_pin}",
                )

    def _refresh_ghost_recency(self, a: str, e: Event, rule: str) -> None:
        for lst, name in ((self.b1, "B1"), (self.b2, "B2")):
            if a in lst:
                lst.move_to_end(a, last=False)
                self._log(a, name, name, rule, e.event_id)
                return

    # ======================================================================
    # §5.3 reference handling / REPLACE / eviction
    # ======================================================================
    def on_reference(self, a: str, e: Event, new_task: bool) -> None:
        if a in self.t1 or a in self.t2:  # ---- Case I: hit
            if a in self.t2 or not new_task:
                lst = self.t2 if a in self.t2 else self.t1
                lst.move_to_end(a, last=False)
            else:
                ok, rule = self.gate_promote(a)
                if ok:  # T1, 2nd distinct task
                    self.t1.pop(a)
                    self._insert_mru(self.t2, a)
                    self.counters["promotions_t1_t2"] += 1
                    self._log(a, "T1", "T2", "two-distinct-task-refs", e.event_id)
                else:
                    self.t1.move_to_end(a, last=False)
                    self.stat(a).blocked_reason = rule
                    self.counters["gate_blocks"] += 1
                    self._log(a, "T1", "T1", f"promotion-blocked:{rule}", e.event_id)
        elif a in self.b1:  # ---- Case II
            self.ghost_hit(a, e, src_name="B1")
        elif a in self.b2:  # ---- Case III
            self.ghost_hit(a, e, src_name="B2")
        else:  # ---- Case IV
            self.cache_miss(a, e)

    def ghost_hit(self, a: str, e: Event, src_name: str) -> None:
        cfg = self.config
        src = self.b1 if src_name == "B1" else self.b2
        s_x = self.acct(a)
        p_before = self.p
        # Gate first: gate_promote reads validity/q/size only (never p or the
        # lists), so evaluating it before the p update changes nothing for the
        # isolate/deny options and enables A-3(b)'s deny-no-adapt option.
        ok, rule = self.gate_promote(a)
        deny = (not ok) and cfg.q5a_deny_revive_on_gate_fail
        no_adapt = deny and cfg.q5a_no_adapt_on_gate_fail
        if not no_adapt:
            if src_name == "B1":  # recency demand proven → grow T1 target (§3.4)
                ratio = self.tok_b2() / max(self.tok_b1(), 1)
                self.p = min(cfg.c, self.p + max(s_x, s_x * ratio))
            else:  # frequency demand proven → shrink T1 target
                ratio = self.tok_b1() / max(self.tok_b2(), 1)
                self.p = max(0.0, self.p - max(s_x, s_x * ratio))
        self.counters["ghost_hits"] += 1
        self.counters["ghost_hits_b1" if src_name == "B1" else "ghost_hits_b2"] += 1
        if cfg.q12_shift_relief and src_name == "B1":  # A-6: accrue shift mass
            self._q12_cur_acct += s_x
            self._q12_cur_arts.append(a)

        if deny:
            # INT-6 (Q5a=deny): ghost MRU refreshed, no revival. p adapted
            # unless deny-no-adapt (A-3(b)) is on.
            src.move_to_end(a, last=False)
            self.stat(a).blocked_reason = rule
            self.counters["revivals_denied"] += 1
            deny_rule = "revive-denied-no-adapt" if no_adapt else "revive-denied"
            self._log(a, src_name, src_name, f"{deny_rule}:{rule}", e.event_id, p_before)
            return

        src.pop(a)
        self.evict_until_fits(self.size_tok(a), demand_from_b2=(src_name == "B2"), evidence=e)
        if ok:
            self._insert_mru(self.t2, a)
            self.counters["revivals_t2"] += 1
            self._log(a, src_name, "T2", f"revive-{src_name}->T2", e.event_id, p_before)
        else:
            self._insert_mru(self.t1, a)
            self.stat(a).blocked_reason = rule
            self.counters["revivals_quarantine_t1"] += 1
            self.counters["gate_blocks"] += 1
            self._log(a, src_name, "T1", f"revive-quarantine:{rule}", e.event_id, p_before)
        self._restore_content_if_cold(a, e)

    def cache_miss(self, a: str, e: Event) -> None:
        cfg = self.config
        if self.is_oversize(a):
            self.oversize[a] = self.oversize.get(a, 0) + 1
            if cfg.split_recommendations:
                self._recommend(
                    "split",
                    a,
                    "oversize-guard",
                    f"size_tok={self.size_tok(a)} > alpha*c={cfg.alpha * cfg.c:.0f}",
                )
            self._log(a, None, "OVERSIZE", "oversize-guard", e.event_id)
            self._restore_content_if_cold(a, e)
            return  # supplied on demand but never resident (§3.2/§6.1)

        need = self.acct(a)
        if self.tok_t1() + self.tok_b1() + need > cfg.c:  # Case IV-A: L1 saturated
            if len(self.b1) > 0:
                self._trim_ghost_lru(
                    self.b1,
                    "B1",
                    until=lambda: self.tok_t1() + self.tok_b1() + need <= cfg.c,
                    evidence=e,
                )
            if len(self.b1) == 0 and self.tok_t1() + need > cfg.c:
                # original ARC's rare direct-delete path (B1 empty); INT-2
                self._evict_t1_tail_direct(need, e)
        # Case IV-B: directory saturated. INT-18: the original's exclusive
        # if/elif relies on count arithmetic (deleting one directory slot in
        # IV-A always makes room); with variable token sizes IV-A's direct
        # T1 delete may free fewer tokens than `need`, so the directory
        # bound (I4) must be checked independently — conservative reading:
        # trim more ghost history rather than let the directory exceed 2c.
        if self.tok_all() + need > 2 * cfg.c:
            self._trim_ghost_lru(
                self.b2, "B2", until=lambda: self.tok_all() + need <= 2 * cfg.c, evidence=e
            )
        self.evict_until_fits(self.size_tok(a), demand_from_b2=False, evidence=e)
        self._insert_mru(self.t1, a)
        self.counters["admissions"] += 1
        self._log(a, None, "T1", "admit->T1", e.event_id)
        self._restore_content_if_cold(a, e)

    def evict_until_fits(self, need: int, demand_from_b2: bool, evidence: Event | None) -> None:
        """REPLACE(x, p) generalized to tokens (§3.4/§5.3)."""
        cfg = self.config
        need = min(need, int(cfg.alpha * cfg.c))  # resident cost is acct-capped
        while self.tok_t1() + self.tok_t2() + need > cfg.c:
            from_t1 = self.tok_t1() > 0 and (
                self.tok_t1() > self.p or (demand_from_b2 and self.tok_t1() >= self.p)
            )
            src, src_name, dst, dst_name = (
                (self.t1, "T1", self.b1, "B1") if from_t1 else (self.t2, "T2", self.b2, "B2")
            )
            rule = "evict:tok(T1)>p" if from_t1 else "evict:tok(T1)<=p"
            if not src:  # INT-1: fallback when the selected list is empty
                if from_t1:
                    src, src_name, dst, dst_name = self.t2, "T2", self.b2, "B2"
                else:
                    src, src_name, dst, dst_name = self.t1, "T1", self.b1, "B1"
                rule += ":fallback-empty-src"
                if not src:
                    break  # both resident lists empty; need ≤ α·c ≤ c holds
            v = self.pick_victim_tail(src)
            src.pop(v)
            self._insert_mru(dst, v)  # ghost_of(v): metadata-only entry
            self.counters["evictions"] += 1
            if self.registry.get(v) is not None and self.registry[v].critical:
                self.counters["critical_resident_evictions"] += 1
            ev_id = evidence.event_id if evidence else None
            self._log(v, src_name, dst_name, rule, ev_id)
        self.enforce_ghost_bounds(evidence)

    def pick_victim_tail(self, src: OrderedDict[str, int]) -> str:
        """§3.2(3): validity ladder inside the LRU tail window only."""
        cfg = self.config
        keys = list(src.keys())  # MRU first → tail is the end
        window = keys[-cfg.k_tail :][::-1]  # LRU-order within the window
        ladder = []
        if not cfg.without_outcome:
            ladder.append(self.harmful)
        if not cfg.without_validity_gate:
            ladder.append(lambda x: self.validity(x) == "STALE")
            ladder.append(lambda x: self.validity(x) == "SUSPECT")
        for pref in ladder:
            for v in window:
                if pref(v):
                    return v
        if cfg.q11_tail_posterior_tiebreak:
            # INT-7: deterministic posterior-mean ranking (lowest q evicted;
            # ties resolved by LRU order = first occurrence in window).
            return min(window, key=lambda x: (self.q(x), window.index(x)))
        return keys[-1]  # pure LRU

    def enforce_ghost_bounds(self, evidence: Event | None = None) -> None:
        cfg = self.config
        while self.tok_t1() + self.tok_b1() > cfg.c and len(self.b1) > cfg.n_ghost_min:
            self._trim_lru_one(self.b1, "B1", evidence)
        while self.tok_all() > 2 * cfg.c and len(self.b2) > cfg.n_ghost_min:
            self._trim_lru_one(self.b2, "B2", evidence)

    def _trim_lru_one(self, lst: OrderedDict[str, int], name: str, evidence: Event | None) -> None:
        a, _ = lst.popitem(last=True)
        self._log(a, name, None, "ghost-truncate", evidence.event_id if evidence else None)
        self.to_cold(a)

    def _trim_ghost_lru(self, lst, name: str, until, evidence: Event | None) -> None:
        while lst and not until():
            self._trim_lru_one(lst, name, evidence)

    def _evict_t1_tail_direct(self, need: int, evidence: Event | None) -> None:
        """Case IV-A with B1 empty: delete T1 LRU without ghosting (INT-2)."""
        while self.t1 and self.tok_t1() + need > self.config.c:
            v = self.pick_victim_tail(self.t1)
            self.t1.pop(v)
            self.counters["evictions"] += 1
            if self.registry.get(v) is not None and self.registry[v].critical:
                self.counters["critical_resident_evictions"] += 1
            self._log(v, "T1", None, "evict:IV-A-direct(B1-empty)",
                      evidence.event_id if evidence else None)
            self.to_cold(v)

    def to_cold(self, a: str) -> None:
        meta = self.registry.get(a)
        critical = meta is not None and meta.critical
        if critical and self.config.critical_cold_approval:
            # §4.3: no automatic transition path exists for critical (I8)
            self._recommend(
                "cold_transition",
                a,
                "critical-ghost-drop",
                "critical artifact left the ghost lists; Cold transition "
                "requires human approval",
                requires_approval=True,
            )
            return
        self.cold_candidates.add(a)
        self.cold_ever.add(a)
        self.counters["cold_auto"] += 1
        if critical:
            self.counters["cold_auto_critical"] += 1

    def _restore_content_if_cold(self, a: str, e: Event) -> None:
        if a in self.cold_candidates:
            self.cold_candidates.discard(a)
            self.counters["restores_from_cold"] += 1
            self._log(a, None, None, "restore-from-cold", e.event_id)

    def _insert_mru(self, lst: OrderedDict[str, int], a: str) -> None:
        lst[a] = self.acct(a)
        lst.move_to_end(a, last=False)

    # ======================================================================
    # §4.2 negative quality path
    # ======================================================================
    def on_negative_quality(self, a: str, e: Event) -> None:
        stale = (not self.config.without_validity_gate) and self.validity(a) in (
            "STALE",
            "INVALIDATED",
        )
        if self.harmful(a) or stale:
            lst_name = self.list_of(a)
            if lst_name in ("T1", "T2"):
                lst = self.t1 if lst_name == "T1" else self.t2
                lst.pop(a)
                self._insert_mru(self.b1, a)
                self.counters["demotions_to_ghost"] += 1
                self._log(a, lst_name, "B1", "demote:harmful|stale", e.event_id)
                self.enforce_ghost_bounds(e)  # INT-16
            self._recommend(
                "invalidate_or_correct",
                a,
                "harmful|stale",
                f"n_corr={self.stat(a).n_corr}, n_val={self.stat(a).n_val}, "
                f"q={self.q(a):.3f}",
            )
        elif a in self.t2:
            self.t2.pop(a)
            self._insert_mru(self.t1, a)
            self.counters["demotions_t2_t1"] += 1
            self._log(a, "T2", "T1", "demote:corrected-once", e.event_id)

    def on_validity_change(self, a: str, new_validity: str, e: Event | None) -> None:
        """Registry-fed validity transition (§4.2 row 3)."""
        s = self.stat(a)
        old = s.validity
        s.validity = new_validity
        ev_id = e.event_id if e else None
        if self.config.without_validity_gate:
            return
        if new_validity in ("STALE", "INVALIDATED") and old not in ("STALE", "INVALIDATED"):
            lst_name = self.list_of(a)
            if lst_name in ("T1", "T2"):
                lst = self.t1 if lst_name == "T1" else self.t2
                lst.pop(a)
                self._insert_mru(self.b1, a)
                self.counters["demotions_to_ghost"] += 1
                self._log(a, lst_name, "B1", f"demote:validity={new_validity}", ev_id)
                self.enforce_ghost_bounds(e)  # INT-16
            self._recommend(
                "invalidate_or_correct",
                a,
                f"validity-{new_validity.lower()}",
                f"validity transitioned {old}->{new_validity}",
            )

    # ======================================================================
    # §6.6 budget change
    # ======================================================================
    def set_budget(self, new_c: int, evidence: Event | None = None) -> None:
        if new_c <= 0:
            raise ValueError("c must be positive")
        self.config = self.config.replace(c=new_c)
        self.p = min(self.p, float(new_c))  # clamp (§6.6)
        ev_id = evidence.event_id if evidence else None
        # INT-9: entries that became oversize under the new c leave the lists.
        for lst, name in ((self.t1, "T1"), (self.t2, "T2")):
            for a in [x for x in lst if self.is_oversize(x)]:
                lst.pop(a)
                self.oversize[a] = self.oversize.get(a, 0)
                self._log(a, name, "OVERSIZE", "budget-change:now-oversize", ev_id)
                if self.config.split_recommendations:
                    self._recommend(
                        "split", a, "oversize-guard",
                        f"size_tok={self.size_tok(a)} > alpha*c={self.config.alpha * new_c:.0f}",
                    )
        for lst, name in ((self.b1, "B1"), (self.b2, "B2")):
            for a in [x for x in lst if self.is_oversize(x)]:
                lst.pop(a)
                self._log(a, name, None, "budget-change:ghost-now-oversize", ev_id)
                self.to_cold(a)
        # re-account resident/ghost entries (acct depends on c), then shrink
        for lst in (self.t1, self.t2, self.b1, self.b2):
            for a in lst:
                lst[a] = self.acct(a)
        self.evict_until_fits(0, demand_from_b2=False, evidence=evidence)
        if self.config.debug_invariants:
            self.check_invariants(evidence)

    # ======================================================================
    # §5.4 invariants (FR-R6)
    # ======================================================================
    def check_invariants(self, e: Event | None) -> None:
        cfg = self.config
        state = None

        def fail(rule: str, detail: str) -> None:
            raise InvariantViolation(rule, detail, e, self._state_payload())

        # I1 disjoint
        sets = {
            "T1": set(self.t1),
            "T2": set(self.t2),
            "B1": set(self.b1),
            "B2": set(self.b2),
            "OVERSIZE": set(self.oversize),
            "PINNED": set(self.pinned_ids),
        }
        names = list(sets)
        for i, n1 in enumerate(names):
            for n2 in names[i + 1 :]:
                inter = sets[n1] & sets[n2]
                if inter:
                    fail("I1", f"{n1}∩{n2}={sorted(inter)[:4]}")
        # I2 resident budget
        if self.tok_t1() + self.tok_t2() > cfg.c:
            fail("I2", f"tok(T1)+tok(T2)={self.tok_t1() + self.tok_t2()} > c={cfg.c}")
        # I3 L1 bound (N_GHOST_MIN floor exception, excess ≤ N_GHOST_MIN·α·c)
        excess1 = self.tok_t1() + self.tok_b1() - cfg.c
        if excess1 > 0:
            if len(self.b1) > cfg.n_ghost_min:
                fail("I3", f"L1 over budget by {excess1} with |B1|={len(self.b1)} > floor")
            if excess1 > cfg.n_ghost_min * cfg.alpha * cfg.c:
                fail("I3", f"floor excess {excess1} > N_GHOST_MIN*alpha*c")
        # I4 directory bound (same floor exception)
        excess2 = self.tok_all() - 2 * cfg.c
        if excess2 > 0:
            if len(self.b2) > cfg.n_ghost_min:
                fail("I4", f"directory over 2c by {excess2} with |B2|={len(self.b2)} > floor")
            if excess2 > cfg.n_ghost_min * cfg.alpha * cfg.c:
                fail("I4", f"floor excess {excess2} > N_GHOST_MIN*alpha*c")
        # I5 p range
        if not (0.0 <= self.p <= cfg.c + 1e-9):
            fail("I5", f"p={self.p} outside [0, c={cfg.c}]")
        # I6 recoverability: every tracked entry resolves in the registry
        for name in ("T1", "T2", "B1", "B2"):
            for a in sets[name]:
                if a not in self.registry:
                    fail("I6", f"{name} entry {a} has no registry metadata")
        # I7 validity hygiene — INT-17: checked per the doc's own timing
        # qualifier "(quality 이벤트 처리 직후 기준)", read per-artifact: the
        # handler of a quality/validity event must never leave THAT artifact
        # resident while stale/harmful. A blanket at-all-times reading would
        # contradict the §5.3 quarantine revival path (Q5a option 1), which
        # deliberately re-admits gate-failed ghosts into T1. Variant-aware:
        # each intervention only exists when its axis is enabled.
        if e is not None and e.event_type in (
            "validated",
            "corrected",
            "conflicted",
            "validity_changed",
        ):
            a = e.artifact_id
            if a in sets["T1"] or a in sets["T2"]:
                if not cfg.without_validity_gate and self.validity(a) in (
                    "STALE",
                    "INVALIDATED",
                ):
                    fail("I7", f"{a} validity={self.validity(a)} resident")
                if not cfg.without_outcome and self.harmful(a):
                    fail("I7", f"{a} harmful but resident")
        # I8 safety: no automatic critical cold transition; pins never listed
        if cfg.critical_cold_approval and self.counters["cold_auto_critical"] > 0:
            fail("I8", "automatic critical ghost->Cold transition occurred")
        # oversize residency (§6.1: no list may hold an oversize artifact)
        for name in ("T1", "T2", "B1", "B2"):
            for a in sets[name]:
                if self.is_oversize(a):
                    fail("I1", f"oversize artifact {a} present in {name}")
        # I9 (determinism) is validated by replaying the stream twice and
        # comparing state hashes — not assertable per event.
        _ = state

    # ======================================================================
    # state serialization (FR-R2)
    # ======================================================================
    def _state_payload(self) -> dict:
        def dump_list(lst: OrderedDict[str, int]) -> list:
            return [[a, tok] for a, tok in lst.items()]

        art = {}
        for a in sorted(self.stats):
            s = self.stats[a]
            art[a] = [
                round(s.F, 9),
                s.n_val,
                s.n_corr,
                s.validity,
                s.last_ref_task,
                s.blocked_reason,
            ]
        return {
            "policy": self.name,
            "policy_version": self.policy_version,
            "p": round(self.p, 6),
            "c": self.config.c,
            "T1": dump_list(self.t1),
            "T2": dump_list(self.t2),
            "B1": dump_list(self.b1),
            "B2": dump_list(self.b2),
            "OVERSIZE": sorted(self.oversize),
            "PINNED": sorted(self.pinned_ids),
            "cold_candidates": sorted(self.cold_candidates),
            "artifacts": art,
        }

    def summary(self) -> dict:
        out = super().summary()
        out.update(self.counters)
        out["p"] = round(self.p, 3)
        out["tok_t1"] = self.tok_t1()
        out["tok_t2"] = self.tok_t2()
        out["tok_b1"] = self.tok_b1()
        out["tok_b2"] = self.tok_b2()
        out["oversize_entries"] = len(self.oversize)
        out["cold_candidates"] = len(self.cold_candidates)
        if self.config.q12_shift_relief:  # additive; off → summary unchanged
            out["q12_boosts"] = self._q12_boosts
        return out
