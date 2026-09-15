"""E15-CODEX-WRITE smoke: one trivial Codex turn on the ChatGPT-subscription
route through a >=0.145.0 binary, to observe ``cache_write_input_tokens``.

Privacy (R-9): only token counts, schema labels, ids and hashes are written to
``raw/``.  The prompt is a constant in this script; the model's output text and
the transcript are never persisted.  Attempt cap: 2 turns (pre-declared).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CELL = Path(__file__).resolve().parents[1]
BINARY = (
    "<HOME>/.local/opt/codex-0.145.0/node_modules/@openai/"
    "codex-darwin-arm64/vendor/aarch64-apple-darwin/bin/codex"
)
MODEL = "gpt-5.6-luna"
REASONING_EFFORT = "low"
PROMPT = "Reply with exactly: ok"
# Attempt 1 landed on an already-warm prefix (cache_read 6912 / write 0), so it
# was not a cache-writing turn.  Attempt 2 perturbs the *front* of the cached
# prefix (developer instructions precede tools and messages) to force a miss,
# which is the only shape under which a write must occur.
ATTEMPT2_OVERRIDES = (
    'developer_instructions="E15-CODEX-WRITE cache-write probe marker"',
)

sys.path.insert(0, str(REPO / "src"))

from karc.bench.driver import (  # noqa: E402
    CODEX_WRITE_TOKEN_KEY,
    CodexCliDriver,
    DriverRequest,
)

USAGE_KEYS_ALLOWED = {
    "input_tokens", "cached_input_tokens", CODEX_WRITE_TOKEN_KEY,
    "output_tokens", "reasoning_output_tokens", "total_tokens",
}


def main() -> int:
    attempt = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    if attempt not in (1, 2):
        raise SystemExit("attempt cap is 2 turns")
    git_hash = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(REPO),
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    with tempfile.TemporaryDirectory(prefix="karc-e15-cwd-") as cwd:
        # Codex refuses to run outside a trusted (git) directory; the harness
        # materializer does the same ``git init`` for every benchmark cwd.
        subprocess.run(["git", "init", "--quiet"], cwd=cwd, check=True,
                       capture_output=True, text=True)
        driver = CodexCliDriver(binary=BINARY)
        request = DriverRequest(
            prompt=PROMPT, cwd=cwd, model=MODEL, runtime="codex",
            reasoning_effort=REASONING_EFFORT, hook_mode="observe",
            timeout_s=300,
            config_overrides=ATTEMPT2_OVERRIDES if attempt == 2 else (),
        )
        result = driver.run(request)
    usage_raw = dict(result.usage_raw or {})
    unexpected = sorted(set(usage_raw) - USAGE_KEYS_ALLOWED)
    record = {
        "cell": "E15-CODEX-WRITE",
        "attempt": attempt,
        "observed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code_git_hash": git_hash,
        "binary_path": BINARY,
        "cli_version": driver.preflight().get("version"),
        "model_requested": MODEL,
        "model_reported": result.reported_model,
        "model_verification": result.model_verification,
        "reasoning_effort": REASONING_EFFORT,
        "prefix_perturbed": attempt == 2,
        "route": "chatgpt-subscription (auth.json bridge, controlled CODEX_HOME)",
        "ok": result.ok,
        "error_class": result.error_class,
        "error_detail": result.error_detail,
        "usage_raw": usage_raw,
        "usage_raw_unexpected_keys": unexpected,
        "usage_normalized": dict(result.usage or {}),
        "write_key_present": CODEX_WRITE_TOKEN_KEY in usage_raw,
        "usage_schema": result.driver_metadata.get("usage_schema"),
        "fresh_semantics": result.driver_metadata.get("fresh_semantics"),
        "gross_input_tokens": result.total_input_tokens(with_cache=True),
        "fresh_input_tokens": result.total_input_tokens(with_cache=False),
        "event_types": sorted(set((result.raw or {}).get("event_types") or [])),
        "hook_event_count": result.driver_metadata.get("hook_event_count"),
    }
    out = CELL / "raw" / f"smoke-attempt-{attempt}.json"
    out.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
