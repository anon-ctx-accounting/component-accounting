"""E15-CODEX-WRITE multi-turn probe: does the ChatGPT-subscription Codex route
report cache-*write* token counts at all?

Design (this is the point of the probe -- it does not chase a cold prefix):
one persistent ``codex exec`` session of several sequential turns, each turn
adding a few hundred to a few thousand tokens of new content so the accumulated
context provably grows.  Then:

  * any turn with ``cache_write_input_tokens > 0``  -> the route reports writes;
  * all writes 0 but ``cached_input_tokens`` grows between turns of the same
    session -> the provider demonstrably wrote new content into the cache
    between those turns while reporting write = 0, i.e. the route OMITS writes.
    That is a decisive negative and needs no cold prefix.
  * all writes 0 and ``cached`` never grows -> inconclusive; the session may not
    have accumulated context, which is checked against ``input_tokens``, not
    assumed.

Privacy (R-9): only token counts, ids, hashes, lengths and schema labels are
written to ``raw/``.  Prompt text, model output and the transcript are never
persisted; the generated filler is reproducible from ``(round, turn)`` via the
constants in this file, and only its sha256 and length are recorded.

Pre-declared attempt cap: 8 model-bearing turns total across all rounds.
"""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
CELL = Path(__file__).resolve().parents[1]
BINARY = (
    "<HOME>/.local/opt/codex-0.145.0/node_modules/@openai/"
    "codex-darwin-arm64/vendor/aarch64-apple-darwin/bin/codex"
)
MODEL = "gpt-5.6-luna"
REASONING_EFFORT = "low"
TOTAL_TURN_CAP = 8

# Round -> (turns, filler lines per turn).  ~14-18 tokens per generated line,
# so round 1 adds roughly 1.5k tokens of new context per turn and round 2 (only
# run if round 1 is inconclusive) roughly 4k.
ROUNDS = {1: (4, 100), 2: (4, 260)}

_WORDS = (
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima "
    "mike november oscar papa quebec romeo sierra tango uniform victor whiskey "
    "xray yankee zulu"
).split()

sys.path.insert(0, str(REPO / "src"))

from karc.bench.driver import (  # noqa: E402
    CODEX_WRITE_TOKEN_KEY,
    CodexPersistentSession,
    DriverRequest,
)

USAGE_KEYS_ALLOWED = {
    "input_tokens", "cached_input_tokens", CODEX_WRITE_TOKEN_KEY,
    "output_tokens", "reasoning_output_tokens", "total_tokens",
}


def build_prompt(round_no: int, turn: int, lines: int) -> str:
    """Deterministic filler block plus a one-word ack instruction."""
    rng = random.Random(f"E15-CODEX-WRITE/{round_no}/{turn}")
    body = []
    for i in range(lines):
        words = " ".join(rng.choice(_WORDS) for _ in range(6))
        token = "".join(rng.choice("0123456789abcdef") for _ in range(10))
        body.append(f"R{round_no}T{turn}-{i:04d} {token} {words} {rng.randrange(10**6)}")
    return (
        f"Ledger block {turn} of an append-only audit log. Store it for later "
        "reference. Do not summarize, do not use any tool.\n"
        + "\n".join(body)
        + "\nAcknowledge this block with exactly: ok"
    )


def main() -> int:
    round_no = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    if round_no not in ROUNDS:
        raise SystemExit(f"round must be one of {sorted(ROUNDS)}")
    turns, lines = ROUNDS[round_no]
    spent_before = sum(ROUNDS[r][0] for r in sorted(ROUNDS) if r < round_no)
    if spent_before + turns > TOTAL_TURN_CAP:
        raise SystemExit("would exceed the pre-declared 8-turn cap")

    git_hash = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(REPO),
        capture_output=True, text=True, check=True,
    ).stdout.strip()

    records: list[dict] = []
    thread_id = None
    with tempfile.TemporaryDirectory(prefix="karc-e15mt-cwd-") as cwd:
        # Codex refuses to run outside a trusted (git) directory, and the
        # persistent-session argv deliberately carries no repo-check bypass.
        subprocess.run(["git", "init", "--quiet"], cwd=cwd, check=True,
                       capture_output=True, text=True)
        request = DriverRequest(
            prompt="", cwd=cwd, model=MODEL, runtime="codex",
            reasoning_effort=REASONING_EFFORT, hook_mode="observe",
            timeout_s=600,
        )
        with CodexPersistentSession(request, binary=BINARY) as session:
            for turn in range(1, turns + 1):
                prompt = build_prompt(round_no, turn, lines)
                result = session.run_turn(prompt)
                usage_raw = dict(result.usage_raw or {})
                meta = result.driver_metadata or {}
                records.append({
                    "turn": turn,
                    "observed_at_utc": datetime.now(timezone.utc).isoformat(
                        timespec="seconds"),
                    "prompt_chars": len(prompt),
                    "prompt_sha256": hashlib.sha256(
                        prompt.encode("utf-8")).hexdigest(),
                    "ok": result.ok,
                    "error_class": result.error_class,
                    "error_detail": result.error_detail,
                    "reported_model": result.reported_model,
                    "model_verification": result.model_verification,
                    "usage_raw": usage_raw,
                    "usage_raw_unexpected_keys": sorted(
                        set(usage_raw) - USAGE_KEYS_ALLOWED),
                    "usage_normalized": dict(result.usage or {}),
                    "write_key_present": CODEX_WRITE_TOKEN_KEY in usage_raw,
                    "usage_schema": meta.get("usage_schema"),
                    "fresh_semantics": meta.get("fresh_semantics"),
                    "gross_input_tokens": result.total_input_tokens(with_cache=True),
                    "fresh_input_tokens": result.total_input_tokens(with_cache=False),
                    "session_turn": meta.get("session_turn"),
                    "persistent_session": meta.get("persistent_session"),
                    "hook_event_count": meta.get("hook_event_count"),
                    "event_types": sorted(set(
                        (result.raw or {}).get("event_types") or [])),
                })
                print(json.dumps(records[-1]["usage_raw"], ensure_ascii=False),
                      flush=True)
                if not result.ok:
                    break
            thread_id = session.thread_id

    ok_turns = [r for r in records if r["ok"]]
    cached = [int(r["usage_raw"].get("cached_input_tokens", 0) or 0)
              for r in ok_turns]
    writes = [int(r["usage_raw"].get(CODEX_WRITE_TOKEN_KEY, 0) or 0)
              for r in ok_turns]
    inputs = [int(r["usage_raw"].get("input_tokens", 0) or 0) for r in ok_turns]
    any_write = any(w > 0 for w in writes)
    cached_grew = any(b > a for a, b in zip(cached, cached[1:]))
    input_grew = any(b > a for a, b in zip(inputs, inputs[1:]))
    if any_write:
        outcome = "route-reports-writes"
    elif cached_grew:
        outcome = "route-omits-writes"
    else:
        outcome = "inconclusive"

    record = {
        "cell": "E15-CODEX-WRITE",
        "probe": "multiturn-cache-write",
        "round": round_no,
        "code_git_hash": git_hash,
        "binary_path": BINARY,
        "cli_version": "codex-cli 0.145.0",
        "model_requested": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "route": "chatgpt-subscription (auth.json bridge, controlled CODEX_HOME)",
        "sandbox": "read-only",
        "planned_turns": turns,
        "filler_lines_per_turn": lines,
        "model_turns_spent_this_round": len(records),
        "model_turns_spent_cumulative": spent_before + len(records),
        "attempt_cap_model_turns": TOTAL_TURN_CAP,
        "thread_id": thread_id,
        "turns": records,
        "evaluation": {
            "cached_series": cached,
            "write_series": writes,
            "input_series": inputs,
            "any_write_gt_zero": any_write,
            "cached_grew_between_turns": cached_grew,
            "input_grew_between_turns": input_grew,
            "outcome": outcome,
        },
        "privacy": "token-counts-and-hashes-only",
    }
    out = CELL / "raw" / f"multiturn-round-{round_no}.json"
    out.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(json.dumps(record["evaluation"], ensure_ascii=False, indent=2))
    return 0 if all(r["ok"] for r in records) and records else 1


if __name__ == "__main__":
    raise SystemExit(main())
