"""P4: checkpoint resume + usage-limit window management for benchmark runs.

E2-2's first execution lost 120/300 runs to a subscription usage limit
(contiguous schedule block, 3 attempts each burned instantly — report §4.2).
This module makes a run resumable and limit-aware:

1. **Checkpoint resume** — ``completed_run_ids`` reads an existing
   ``runs.jsonl`` and returns the run_ids whose *latest* row is validly
   complete (``failure_class`` ∈ {"ok", "task"}); ``pending_units`` filters the
   deterministic ``schedule(spec)`` grid down to everything else
   (``usage_limit`` / ``api_error`` / never-run). The schedule itself stays a
   pure function of the spec seed, so the interleaved order — and therefore the
   §8.3 time-effect control — is reproducible across resumes; a resume simply
   skips the completed prefix subset. run_id rules are unchanged.

2. **Usage-limit window loop** — ``RunLoop`` drives ``harness.run_unit`` with
   rolling submission (a cap check runs before every submission). When a unit
   comes back ``usage_limit``: stop submitting, drain in-flight, close the
   execution *window*, wait until the limit resets (parsed from the refusal
   message via ``parse_reset_wait_s``; conservative fallback otherwise), then
   confirm recovery with a caller-supplied probe before opening the next
   window and re-queuing the limit-hit units (same run_id — the later row
   supersedes; every row still carries provenance: ``exec_window`` and
   ``resumed``). Window boundaries (stop/resume timestamps, wait, probe count)
   are recorded for the report (R-16 §6).

Everything here is LLM-free; the probe and sleep are injected callables so the
whole loop is unit-testable with mocks.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

VALID_COMPLETE = ("ok", "task")


# --------------------------------------------------------------------------
# 1. checkpoint computation (pure)
# --------------------------------------------------------------------------
def load_rows(runs_jsonl: str | Path) -> list[dict]:
    p = Path(runs_jsonl)
    if not p.exists():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # torn tail line from an interrupted write
    return rows


def latest_by_run_id(rows: list[dict]) -> dict[str, dict]:
    """Later row wins — a re-executed unit (same run_id, next window)
    supersedes its earlier usage_limit/api_error row for analysis."""
    out: dict[str, dict] = {}
    for r in rows:
        rid = r.get("run_id")
        if rid:
            out[rid] = r
    return out


def completed_run_ids(rows: list[dict]) -> set[str]:
    return {rid for rid, r in latest_by_run_id(rows).items()
            if r.get("failure_class") in VALID_COMPLETE}


def pending_units(spec, units, run_id_fn) -> list:
    """Filter the deterministic schedule down to units without a valid
    completed row. ``units`` is the full ``schedule(spec)`` output (order
    preserved); ``run_id_fn(arm, task, rep) -> run_id`` matches the harness
    rule. Returns the sub-list in original interleaved order."""
    done = spec if isinstance(spec, set) else set(spec)
    return [(arm, task, rep) for arm, task, rep in units
            if run_id_fn(arm, task, rep) not in done]


# --------------------------------------------------------------------------
# 2. reset-time parsing
# --------------------------------------------------------------------------
_RESET_RE = re.compile(
    r"(?i)resets?\s*(?:at\s*)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?")

MARGIN_S = 120  # start a little after the stated reset


def parse_reset_wait_s(detail: str | None, *, now: datetime | None = None,
                       fallback_s: int = 3600,
                       max_s: int = 6 * 3600) -> tuple[int, str]:
    """Seconds to wait before resuming, from a limit-refusal message like
    "5-hour limit reached ∙ resets 3am". Returns (seconds, source) where
    source ∈ {"parsed", "fallback"}. Unparseable/absent/absurd → conservative
    ``fallback_s``. The parsed clock time is interpreted in *local* time (the
    CLI prints local clock times) as the next future occurrence."""
    if not detail:
        return fallback_s, "fallback"
    m = _RESET_RE.search(detail)
    if not m:
        return fallback_s, "fallback"
    hour = int(m.group(1))
    minute = int(m.group(2) or 0)
    ampm = (m.group(3) or "").lower()
    if ampm == "pm" and hour != 12:
        hour += 12
    elif ampm == "am" and hour == 12:
        hour = 0
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return fallback_s, "fallback"
    now = now if now is not None else datetime.now().astimezone()
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    wait = int((target - now).total_seconds()) + MARGIN_S
    if wait > max_s:  # implausibly far → treat as unparseable
        return fallback_s, "fallback"
    return wait, "parsed"


# --------------------------------------------------------------------------
# 3. resumable run loop with usage-limit windows
# --------------------------------------------------------------------------
@dataclass
class Window:
    window: int
    started_at: str
    stopped_at: str | None = None
    stop_reason: str | None = None
    wait_s: int | None = None
    wait_source: str | None = None
    probe_attempts: int = 0

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RunLoop:
    """Rolling-submission executor with usage-limit pause/probe/resume.

    - ``harness``: provides ``run_unit(arm, task, rep) -> RunOutcome``.
    - ``units``: pending (arm, task, rep) list in schedule order.
    - ``on_outcome(outcome, window, resumed) -> bool``: caller sink (writes the
      row, tracks cost/attempts); return False to abort the whole run (cap).
    - ``probe_fn() -> bool``: cheap recovery check, called after each wait;
      True = limit lifted. Injected so tests never make an LLM call.
    - ``sleep_fn``: injected for tests.
    - ``max_windows``: hard stop on pathological limit loops.
    """
    harness: object
    units: list
    on_outcome: object
    probe_fn: object
    concurrency: int = 1
    sleep_fn: object = time.sleep
    fallback_wait_s: int = 3600
    max_windows: int = 8
    windows: list[Window] = field(default_factory=list)
    aborted: dict | None = None

    def run(self) -> dict:
        from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

        queue = list(self.units)
        resumed_ids: set[str] = set()
        window = Window(window=len(self.windows), started_at=_now_iso())
        self.windows.append(window)

        while queue and self.aborted is None:
            limit_hits: list = []
            with ThreadPoolExecutor(max_workers=self.concurrency) as ex:
                futs = {}
                stop_submitting = False

                def submit_next() -> bool:
                    if stop_submitting or not queue:
                        return False
                    unit = queue.pop(0)
                    futs[ex.submit(self.harness.run_unit, *unit)] = unit
                    return True

                for _ in range(min(self.concurrency, len(queue))):
                    submit_next()
                while futs:
                    done, _pending = wait(list(futs), return_when=FIRST_COMPLETED)
                    for f in done:
                        unit = futs.pop(f)
                        outcome = f.result()
                        keep_going = self.on_outcome(
                            outcome, window.window,
                            outcome.run_id in resumed_ids)
                        if outcome.failure_class == "usage_limit":
                            limit_hits.append((unit, outcome))
                            stop_submitting = True
                        if not keep_going:
                            self.aborted = self.aborted or {
                                "reason": "caller_stop", "window": window.window}
                            stop_submitting = True
                    if not stop_submitting:
                        while len(futs) < self.concurrency and submit_next():
                            pass

            if self.aborted is not None:
                break
            if not limit_hits:
                break  # queue drained cleanly

            # ---- window transition: wait → probe → resume ----------------
            unit_list, outcomes = zip(*limit_hits)
            detail = ""
            for o in outcomes:
                for a in reversed(o.attempts):
                    if a.get("error_class") == "usage_limit" and a.get("error_detail"):
                        detail = a["error_detail"]
                        break
                if detail:
                    break
            window.stopped_at = _now_iso()
            window.stop_reason = "usage_limit"
            wait_s, source = parse_reset_wait_s(
                detail, fallback_s=self.fallback_wait_s)
            window.wait_s, window.wait_source = wait_s, source

            if len(self.windows) >= self.max_windows:
                self.aborted = {"reason": "max_windows",
                                "windows": len(self.windows)}
                break

            self.sleep_fn(wait_s)
            probes = 0
            while True:
                probes += 1
                if self.probe_fn():
                    break
                if probes >= 3:
                    self.aborted = {"reason": "probe_never_recovered",
                                    "window": window.window}
                    break
                self.sleep_fn(self.fallback_wait_s)
            window.probe_attempts = probes
            if self.aborted is not None:
                break

            # re-queue limit-hit units at the front (same run_id — the later
            # row supersedes) and open the next window.
            for unit, o in limit_hits:
                resumed_ids.add(o.run_id)
            queue = list(unit_list) + queue
            window = Window(window=len(self.windows), started_at=_now_iso())
            self.windows.append(window)

        if self.windows and self.windows[-1].stopped_at is None:
            self.windows[-1].stopped_at = _now_iso()
            self.windows[-1].stop_reason = (
                self.aborted["reason"] if self.aborted else "drained")
        return {"windows": [w.as_dict() for w in self.windows],
                "aborted": self.aborted}
