"""Workload script generator — experiment-design §6.3 family 3종 (FR-R3).

Deterministic function of (family, seed, corpus, noise params). Each task is
a reference-sequence template expanded with seeded sampling, then noise is
applied: within-task order shuffle, event drop 0~50% (per-event hash so drop
sets nest monotonically across rates — V9), task_id 누락률.

The generator returns both the noisy event stream (what policies observe)
and the ground-truth reference plan per task (what actually happened) —
the outcome injector (§6.5) works from the ground truth (INT-15: telemetry
loss must not change the world, only its observation).

Segment map (exposed in ``Workload.meta`` for the V-suite):
- F-A (50 tasks, project A): hot repetition; pure distractor scan tasks
  20–22 (V1); retry-loop tasks 10 and 30 (V10).
- F-B (110 tasks): A 0–49 → B 50–99 → A return 100–109 (V2 shift at 50);
  preload noise at every session start; cyclic segment 70–84 over a set of
  ≈1.2×budget tokens (V8).
- F-C (60 tasks, adversarial): useless preload every session (V4); harmful
  docs with scripted corrected×2 (V5); superseded v1→v2 switch at task 25;
  conflict pairs at tasks 30–31; rare-critical at task 2 re-appearing at
  task 58 (V6); oversize reads (V7); no-locality one-shot segment 45–57
  (V12).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from karc.policy.model import Event
from karc.replay.corpus import Corpus, _hash_int, _unit, generate_corpus

FAMILIES = ("f-a", "f-b", "f-c")
SESSION_TASKS = 5
TASK_GAP_MIN = 45.0
_T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _ts(minutes: float) -> str:
    t = _T0 + timedelta(minutes=minutes)
    return t.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


@dataclass
class TaskSpec:
    idx: int
    task_id: str
    session_id: str
    segment: str
    required: list[str] = field(default_factory=list)
    forbidden: list[str] = field(default_factory=list)
    refs: list[tuple[str, str]] = field(default_factory=list)  # ground truth (etype, artifact)
    events: list[Event] = field(default_factory=list)  # post-noise observed stream


@dataclass
class Workload:
    family: str
    seed: int
    corpus: Corpus
    tasks: list[TaskSpec]
    meta: dict

    @property
    def n_events(self) -> int:
        return sum(len(t.events) for t in self.tasks)


class _Builder:
    def __init__(self, family: str, seed: int, corpus: Corpus):
        self.family = family
        self.seed = seed
        self.corpus = corpus
        self.tasks: list[TaskSpec] = []
        self._eid = 0
        # (etype, artifact, load_class, new_validity, droppable)
        self._pending: list[tuple] = []

    def rng(self, *key) -> random.Random:
        return random.Random(_hash_int(self.seed, self.family, *key))

    def task(self, idx: int, segment: str) -> TaskSpec:
        t = TaskSpec(
            idx=idx,
            task_id=f"T{idx:03d}",
            session_id=f"S{idx // SESSION_TASKS:03d}",
            segment=segment,
        )
        self.tasks.append(t)
        self._pending = []
        return t

    def emit(self, etype: str, artifact: str, load_class=None, new_validity=None,
             droppable: bool = True) -> None:
        self._pending.append((etype, artifact, load_class, new_validity, droppable))

    def ref(self, t: TaskSpec, etype: str, artifact: str, load_class=None) -> None:
        t.refs.append((etype, artifact))
        self.emit(etype, artifact, load_class=load_class)

    def preload_block(self, t: TaskSpec) -> None:
        if t.idx % SESSION_TASKS == 0:
            for a in self.corpus.groups["preload_set"]:
                self.emit("loaded", a, load_class="preload")

    def finish_task(self, t: TaskSpec, drop_rate: float, shuffle: bool,
                    task_id_missing_rate: float) -> None:
        items = list(self._pending)
        # split off non-shufflable/non-droppable registry feeds (validity_changed)
        head = [it for it in items if it[0] == "validity_changed"]
        body = [it for it in items if it[0] != "validity_changed"]
        # assign stable event ids BEFORE noise so drop sets nest across rates
        head_ids = [self._next_id() for _ in head]
        body_ids = [self._next_id() for _ in body]
        if shuffle:
            order = list(range(len(body)))
            self.rng("shuffle", t.idx).shuffle(order)
            body = [body[i] for i in order]
            body_ids = [body_ids[i] for i in order]
        kept: list[tuple[str, tuple]] = [(i, it) for i, it in zip(head_ids, head)]
        for eid, it in zip(body_ids, body):
            if drop_rate > 0 and it[4] and _unit(self.seed, "drop", eid) < drop_rate:
                continue
            kept.append((eid, it))
        base_min = t.idx * TASK_GAP_MIN
        for j, (eid, (etype, artifact, load_class, new_validity, _)) in enumerate(kept):
            task_id: str | None = t.task_id
            if (
                task_id_missing_rate > 0
                and etype != "validity_changed"
                and _unit(self.seed, "tidmiss", eid) < task_id_missing_rate
            ):
                task_id = None
            t.events.append(
                Event(
                    event_id=eid,
                    event_type=etype,
                    occurred_at=_ts(base_min + j * (5.0 / 60.0)),
                    artifact_id=artifact,
                    task_id=task_id,
                    session_id=t.session_id,
                    scope_id="fixture",
                    load_class=load_class,
                    new_validity=new_validity,
                )
            )

    def _next_id(self) -> str:
        self._eid += 1
        return f"{self.family}.s{self.seed}.e{self._eid:06d}"


def _hot_work(b: _Builder, t: TaskSpec, hot: list[str], nmin=3, nmax=6) -> None:
    """Zipf-weighted hot-doc work: reads + occasional cited/applied."""
    rng = b.rng("hotwork", t.idx)
    n = rng.randint(nmin, nmax)
    weights = [1.0 / (i + 1) for i in range(len(hot))]
    chosen: list[str] = []
    pool = list(hot)
    w = list(weights)
    for _ in range(min(n, len(pool))):
        a = rng.choices(pool, weights=w, k=1)[0]
        i = pool.index(a)
        pool.pop(i)
        w.pop(i)
        chosen.append(a)
    for k, a in enumerate(chosen):
        b.ref(t, "read", a)
        if k == 0 and rng.random() < 0.30:
            b.ref(t, "cited", a)
        elif rng.random() < 0.15:
            b.ref(t, "applied", a)
    t.required.extend(chosen)


def _chain_work(b: _Builder, t: TaskSpec, chains: list[list[str]]) -> None:
    rng = b.rng("chain", t.idx)
    chain = chains[rng.randrange(len(chains))]
    for a in chain:
        b.ref(t, "read", a)
    t.required.extend(chain)


# ---------------------------------------------------------------------------
# families
# ---------------------------------------------------------------------------
def _build_fa(b: _Builder, noise: dict) -> dict:
    g = b.corpus.groups
    hot = g["hot_a"]
    scan_pool = list(g["scan_pool"])
    scan_tasks = (20, 22)
    retry_tasks = (10, 30)
    scan_cursor = 0
    for idx in range(50):
        if scan_tasks[0] <= idx <= scan_tasks[1]:
            t = b.task(idx, "scan")
            for _ in range(6):  # pure one-shot sweep (V1)
                if scan_cursor >= len(scan_pool):
                    break
                b.ref(t, "read", scan_pool[scan_cursor])
                scan_cursor += 1
        elif idx in retry_tasks:
            t = b.task(idx, "retry")
            a = hot[0]
            for _ in range(20):  # retry loop (V10)
                b.ref(t, "read", a)
            t.required.append(a)
        else:
            t = b.task(idx, "hot")
            _hot_work(b, t, hot)
            if idx % 7 == 3:
                _chain_work(b, t, g["chains_a"])
        b.finish_task(t, **noise)
    return {
        "scan_tasks": list(scan_tasks),
        "retry_tasks": list(retry_tasks),
        "retry_artifact": hot[0],
        "invalid_from": {},
    }


def _cycle_set(corpus: Corpus, budget_tokens: int) -> list[str]:
    g = corpus.groups
    pool = list(g["hot_b"]) + [a for ch in g["chains_b"] for a in ch] + [
        a for a in g["distractors_b"] if a not in set(g["fresh_pool"])
    ] + list(g["distractors_shared"])
    target = 1.2 * budget_tokens
    out, tok = [], 0
    for a in pool:
        if tok >= target:
            break
        out.append(a)
        tok += corpus.artifacts[a].size_tok
    return out


def _build_fb(b: _Builder, noise: dict, budget_tokens: int) -> dict:
    g = b.corpus.groups
    cycle = _cycle_set(b.corpus, budget_tokens)
    cycle_tasks = (70, 84)
    shift_idx, return_idx = 50, 100
    cursor = 0
    for idx in range(110):
        if idx < shift_idx:
            t = b.task(idx, "phase-a")
            b.preload_block(t)
            _hot_work(b, t, g["hot_a"])
        elif cycle_tasks[0] <= idx <= cycle_tasks[1]:
            t = b.task(idx, "cycle")  # V8: round-robin over ≈1.2c tokens
            b.preload_block(t)
            for _ in range(5):
                a = cycle[cursor % len(cycle)]
                cursor += 1
                b.ref(t, "read", a)
                t.required.append(a)
        elif idx < return_idx:
            t = b.task(idx, "phase-b")
            b.preload_block(t)
            _hot_work(b, t, g["hot_b"])
            if idx % 9 == 5:
                _chain_work(b, t, g["chains_b"])
        else:
            t = b.task(idx, "return-a")
            b.preload_block(t)
            _hot_work(b, t, g["hot_a"])
        b.finish_task(t, **noise)
    return {
        "shift_task_idx": shift_idx,
        "return_task_idx": return_idx,
        "cycle_tasks": list(cycle_tasks),
        "cycle_set_tokens": sum(b.corpus.artifacts[a].size_tok for a in cycle),
        "invalid_from": {},
    }


def _build_fc(b: _Builder, noise: dict) -> dict:
    g = b.corpus.groups
    hot = g["hot_a"]
    harmful = g["harmful"]
    sup_pairs = g["superseded_pairs_a"]
    conflict_pairs = g["conflict_pairs_a"]
    fresh = list(g["fresh_pool"])
    critical_late = g["critical_a"][0]
    oversize_a = [a for a in g["oversize"] if a.startswith("a-")]
    supersede_idx = 25
    nolocality = (45, 57)
    # V5 requires zero T2 promotions for harmful docs: the first corrected
    # must precede the doc's 2nd distinct-task reference, so corrections land
    # in the first read task (0) and a later one (8) → harmful at task 8.
    harmful_corrected_tasks = (0, 8)
    fresh_cursor = 0

    # --- Amendment A-5(b): rehabilitation scenario (C11′) ---------------
    # rehab docs are good (not forbidden/harmful). Operationalization note
    # (fixed here, before any run): A-5(b) states the rehab doc is "required"
    # early, receives corrected×1 producing "q=1/3, gate blocked", then is
    # "required" again ≥5 tasks later. The §6.5 injector emits ``validated``
    # on required∩supplied for every SUCCESS task, so early *required*
    # references would push n_val>0 before the correction and the charter's
    # own "q=1/3" (n_val=0, n_corr=1) state — the state that makes validated
    # decision-changing and α-sensitive, i.e. the entire point of the C11′
    # redesign — could not hold. So the early phase uses *read* references
    # (establish T2 residency, emit no validated) and the correction lands at
    # n_val=0; the ≥5 later references are *required* so the injector emits
    # the α-subsampled validated that drives rehabilitation. Deviation from
    # the literal "required (early)" is documented in the rerun report.
    rehab = list(g.get("rehab") or [])
    rehab_read_tasks = {1, 3, 6, 9, 12, 15, 19}   # promote to T2, NOT required
    rehab_corrected_task = 20                      # corrected×1 at n_val=0
    rehab_required_tasks = {26, 28, 32, 34, 36, 38, 40}  # ≥5 required → validated

    invalid_from = {v1: supersede_idx for v1, _ in sup_pairs}

    for idx in range(60):
        seg = "adversarial"
        if nolocality[0] <= idx <= nolocality[1]:
            seg = "no-locality"
        t = b.task(idx, seg)
        b.preload_block(t)  # useless preload 상시 (V4)

        if idx == supersede_idx:
            for v1, v2 in sup_pairs:  # registry feed: v1 → STALE
                b.emit("validity_changed", v1, new_validity="STALE", droppable=False)

        if seg == "no-locality":
            for _ in range(4):  # V12: brand-new one-shot artifacts only
                if fresh_cursor >= len(fresh):
                    break
                b.ref(t, "read", fresh[fresh_cursor])
                fresh_cursor += 1
            b.finish_task(t, **noise)
            t.forbidden = [v1 for v1, _ in sup_pairs] if idx >= supersede_idx else []
            continue

        _hot_work(b, t, hot, nmin=2, nmax=4)

        # superseded usage: v1 before the switch, v2 after (V4/V5 배경 하중)
        pair = sup_pairs[idx % len(sup_pairs)]
        if idx < supersede_idx:
            b.ref(t, "read", pair[0])
            t.required.append(pair[0])
        else:
            b.ref(t, "read", pair[1])
            t.required.append(pair[1])
        t.forbidden = [v1 for v1, _ in sup_pairs] if idx >= supersede_idx else []

        # harmful docs: high read frequency + scripted corrected×2 (V5)
        if idx <= 40 and idx % 2 == 0:
            for h in harmful:
                b.ref(t, "read", h)
        if idx in harmful_corrected_tasks:
            for h in harmful:
                b.emit("corrected", h)

        # rehab docs (A-5(b)): early reads (→T2, no validated), one corrected
        # at n_val=0 (→q=1/3, demote+block), then ≥5 required tasks whose
        # success drives the α-injected validated that rehabilitates them.
        if rehab and idx in rehab_read_tasks:
            for r in rehab:
                b.ref(t, "read", r)
        if rehab and idx == rehab_corrected_task:
            for r in rehab:
                b.emit("corrected", r)
        if rehab and idx in rehab_required_tasks:
            for r in rehab:
                b.ref(t, "read", r)
                t.required.append(r)

        # conflict pairs referenced together → conflicted (V-suite F-C spec)
        if idx in (30, 31):
            x, y = conflict_pairs[idx - 30]
            b.ref(t, "read", x)
            b.ref(t, "read", y)
            b.emit("conflicted", x)
            b.emit("conflicted", y)

        # rare-critical: once early, re-needed at the very end (V6)
        if idx == 2:
            b.ref(t, "read", critical_late)
            b.ref(t, "applied", critical_late)
            t.required.append(critical_late)
        if idx == 58:
            b.ref(t, "read", critical_late)
            t.required.append(critical_late)

        # oversize long docs (V7)
        if idx in (5, 15, 35):
            a = oversize_a[idx % len(oversize_a)]
            b.ref(t, "read", a)
            t.required.append(a)

        b.finish_task(t, **noise)

    return {
        "supersede_task_idx": supersede_idx,
        "nolocality_tasks": list(nolocality),
        "harmful": list(harmful),
        "critical_late": critical_late,
        "critical_late_tasks": [2, 58],
        "conflict_tasks": [30, 31],
        "oversize_read_tasks": [5, 15, 35],
        "rehab": list(rehab),
        "rehab_read_tasks": sorted(rehab_read_tasks),
        "rehab_corrected_task": rehab_corrected_task,
        "rehab_required_tasks": sorted(rehab_required_tasks),
        "invalid_from": invalid_from,
    }


def generate_workload(
    family: str,
    seed: int,
    corpus: Corpus | None = None,
    budget_tokens: int | None = None,
    drop_rate: float = 0.0,
    shuffle: bool = False,
    task_id_missing_rate: float = 0.0,
) -> Workload:
    family = family.strip().lower().replace("_", "-")
    if family in ("fa", "fb", "fc"):
        family = f"f-{family[1]}"
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}; known: {FAMILIES}")
    if not (0.0 <= drop_rate <= 0.5):
        raise ValueError("drop_rate must be within 0~0.5 (§6.3)")
    corpus = corpus or generate_corpus(seed)
    if budget_tokens is None:
        budget_tokens = int(corpus.total_tokens * 0.10)
    b = _Builder(family, seed, corpus)
    noise = {
        "drop_rate": drop_rate,
        "shuffle": shuffle,
        "task_id_missing_rate": task_id_missing_rate,
    }
    if family == "f-a":
        meta = _build_fa(b, noise)
    elif family == "f-b":
        meta = _build_fb(b, noise, budget_tokens)
    else:
        meta = _build_fc(b, noise)
    meta.update(
        {
            "family": family,
            "seed": seed,
            "noise": noise,
            "budget_tokens_hint": budget_tokens,
            "n_tasks": len(b.tasks),
        }
    )
    return Workload(family=family, seed=seed, corpus=corpus, tasks=b.tasks, meta=meta)
