"""E15-CODEX-WRITE support check (no model spend): is Codex ``turn.completed``
usage per-invocation, or cumulative over the session?

The multi-turn probe's verdict rests on per-turn ``cached_input_tokens`` growing
inside one session.  If the CLI instead reported session-cumulative totals,
growth would be arithmetically trivial and the verdict would not follow.  This
falsifies the cumulative reading from *already committed* raw, read-only:

Under a cumulative reading, a turn's own input equals the increment over the
previous turn.  A session that accumulates context can never have a later turn
whose own input is smaller than turn 1's -- turn N's prompt contains turn 1's.
E10-P2-H16 ``karc-full`` sessions show exactly that (turn-2 increment far below
turn-1 total), so the cumulative reading is false and the values are per-turn.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
CELL = Path(__file__).resolve().parents[1]
SOURCE = REPO / "docs/experiments/E10-P2-H16/run/raw/turns.jsonl"


def main() -> int:
    rows = [json.loads(line) for line in SOURCE.read_text().splitlines() if line]
    sessions: dict[tuple, list[dict]] = {}
    for r in rows:
        sessions.setdefault((r["arm"], r["session_id"]), []).append(r)

    checked = 0
    monotone_gross = 0
    cumulative_violations = 0
    example = None
    for key, turns in sessions.items():
        turns.sort(key=lambda r: r["position"])
        if len(turns) < 3:
            continue
        checked += 1
        gross = [r["api_input_tokens_with_cache"] for r in turns]
        if all(b >= a for a, b in zip(gross, gross[1:])):
            monotone_gross += 1
        increments = [b - a for a, b in zip(gross, gross[1:])]
        # Cumulative reading => increments are the per-turn own inputs.
        if any(inc < gross[0] for inc in increments):
            cumulative_violations += 1
            if example is None:
                example = {
                    "arm": key[0],
                    "gross_series_first6": gross[:6],
                    "increment_series_first5": increments[:5],
                    "turn1_gross": gross[0],
                }

    out = {
        "cell": "E15-CODEX-WRITE",
        "check": "codex-turn-usage-is-per-invocation-not-cumulative",
        "source_committed_raw": str(SOURCE.relative_to(REPO)),
        "source_recomputed": False,
        "model_turns_spent": 0,
        "sessions_with_at_least_3_turns": checked,
        "sessions_with_monotone_gross": monotone_gross,
        "sessions_falsifying_cumulative_reading": cumulative_violations,
        "example": example,
        "conclusion": (
            "cumulative reading falsified" if cumulative_violations
            else "NEEDS-DATA: cumulative reading not falsified by this source"
        ),
    }
    path = CELL / "raw" / "multiturn-usage-semantics-check.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
