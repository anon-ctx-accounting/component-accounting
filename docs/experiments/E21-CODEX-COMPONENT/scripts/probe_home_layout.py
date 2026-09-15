"""E21 diagnostic — what is inside an isolated CODEX_HOME after a turn?

The H11 authproof run found a ``responses.jsonl`` in the isolated home carrying
three records with token counts (input 180/152/164, output 96/74/88) that do
NOT appear in any ``turn.completed`` usage block.  Either the CLI is making
auxiliary model calls whose billing this cell would silently omit, or the file
is not provider traffic at all.  Guessing is not allowed, so this probe
measures it: two trivial turns in a bare git workdir (no MCP server, no
AGENTS.md), then a structural dump of the home.

R-9: only file names, sizes, per-line JSON KEY NAMES, discriminator values that
are short enum-like strings, and numeric token fields are recorded.  No prompt,
no model output, no command text.

Cost: two turns of a ~20-token prompt.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
CELL = HERE.parents[1]
REPO = HERE.parents[4]
if not (REPO / "pyproject.toml").exists():
    raise SystemExit(f"E21 probe: repo root misresolved as {REPO}")
sys.path.insert(0, str(REPO / "src"))

import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "e21run", HERE.parent / "run_e21_codex_component.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)

from karc.bench.driver import DriverRequest  # noqa: E402

ENUM_KEYS = ("type", "role", "kind", "record_type", "event", "model",
             "status", "name", "source")
MAX_ENUM_LEN = 64


def describe(obj, prefix: str = "") -> list[str]:
    keys: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            keys.append(path)
            if isinstance(value, (dict, list)):
                keys += describe(value, path)
    elif isinstance(obj, list):
        if obj:
            keys += describe(obj[0], f"{prefix}[]")
    return keys


def main() -> int:
    root = REPO / "tmp" / "e21" / "probe"
    root.mkdir(parents=True, exist_ok=True)
    work = root / "work"
    if work.exists():
        import shutil
        shutil.rmtree(work)
    work.mkdir(parents=True)
    subprocess.run(["git", "init", "--quiet"], cwd=work, check=True,
                   capture_output=True, text=True)
    request = DriverRequest(
        prompt="", cwd=str(work), model=run.MODEL, runtime="codex",
        timeout_s=300, reasoning_effort=run.REASONING_EFFORT,
        hook_mode="observe")
    key = run._read_api_key()
    turns = []
    layout: list[dict] = []
    with run.MeteredCodexHome(temp_root=root / "homes") as home:
        home.login_with_api_key(key)
        with run.MeteredCodexSession(request, home=home) as session:
            for index in (1, 2):
                result = session.run_turn(
                    f"Turn {index}. Reply with exactly: ok")
                turns.append({
                    "turn": index, "ok": result.ok,
                    "usage_raw": dict(result.usage_raw or {}),
                    "event_summary": dict(session.last_event_summary),
                })
        for path in sorted(home.path.rglob("*")):
            if path.is_dir():
                continue
            rel = str(path.relative_to(home.path))
            entry = {"path": rel, "bytes": path.stat().st_size}
            if path.suffix == ".jsonl" and path.name != "guard-events.jsonl":
                lines = []
                for lineno, line in enumerate(
                        path.read_text(encoding="utf-8",
                                       errors="replace").splitlines()):
                    line = line.strip()
                    if not line.startswith("{"):
                        lines.append({"line": lineno, "not_json": True})
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        lines.append({"line": lineno, "unparsable": True})
                        continue
                    tokens: dict = {}
                    run._collect_token_fields(obj, "", tokens)
                    enums = {}
                    for key_name in ENUM_KEYS:
                        value = obj.get(key_name) if isinstance(obj, dict) else None
                        if isinstance(value, str) and len(value) <= MAX_ENUM_LEN:
                            enums[key_name] = value
                    lines.append({
                        "line": lineno,
                        "key_paths": sorted(set(describe(obj)))[:80],
                        "enums": enums,
                        "token_fields": tokens,
                    })
                entry["records"] = lines
            layout.append(entry)
        rollout = home.rollout_usage()
    record = {
        "cell": "E21-CODEX-COMPONENT",
        "probe": "codex-home-layout",
        "binary": run.BINARY, "codex_version": run.CODEX_VERSION,
        "model": run.MODEL, "reasoning_effort": run.REASONING_EFFORT,
        "workdir": str(work),
        "note": "bare git workdir: no MCP server, no AGENTS.md, no guard mode "
                "beyond observe",
        "turns": turns,
        "home_layout": layout,
        "rollout_usage": rollout,
        "privacy": "file-names-key-paths-enums-and-token-counts-only",
        "generated_at_utc": run.now_utc(),
    }
    out = CELL / "raw" / "probe-home-layout.json"
    out.write_text(json.dumps(record, ensure_ascii=False, indent=2,
                              sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
