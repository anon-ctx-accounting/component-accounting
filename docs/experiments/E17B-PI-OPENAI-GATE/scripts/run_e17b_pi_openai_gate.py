"""E17B-PI-OPENAI-GATE — does pi's OpenAI leg surface 4-component usage per turn?

Derived from docs/experiments/E17-PI-GATE/scripts/run_e17_pi_gate.py (that cell
GATE-PASSed the Anthropic leg).  Only the provider leg, the scratch directory and
the closure measurement differ; the isolation discipline, the prompt shape and
the per-turn audit scoping are carried over unchanged.

Single gate question:
  (a) at least 4 consecutive turns complete on pi's OpenAI leg,
  (b) input / output / cacheRead / cacheWrite are recoverable per turn,
  (c) a cold call reports cacheWrite > 0,
  (d) the inclusion relation of `usage.input` is decided BY MEASUREMENT.

On (d): the three sources known to this project disagree with each other
(E17 §3).  Anthropic raw `input_tokens` excludes cache buckets, OpenAI raw
`input_tokens` includes them (E16 §1), pi's Anthropic `usage.input` excludes
them.  pi's OpenAI leg is a fourth, independent case, so nothing is assumed:
`totalTokens` is passed through from the provider's `total_tokens`, therefore
  inclusive hypothesis   totalTokens == input + output
  exclusive hypothesis   totalTokens == input + output + cacheRead + cacheWrite
are mutually exclusive whenever cacheRead + cacheWrite > 0, and both are
evaluated on every call.

Phases (`python run_e17b_pi_openai_gate.py <phase>`):
  mtime     snapshot the user's ~/.pi paths (no pi, no model)
  auth      model-free: `pi auth check --provider openai --json` inside the
            isolated pi home, to prove the user's auth.json is invisible
  prepare   materialize the managed workdir + .mcp.json (no model, no pi)
  registry  model-free: load the bridge in pi --mode rpc and dump pi's own
            tool registry via the E17B probe extension
  smoke     N turns of one fixture session through `pi -p --mode json`
            (requires KARC_OPENAI_KEY in the environment)

Privacy (R-9): raw rows carry counts, ids, sha256 and usage numbers only.
Prompt text, model text and tool-result text are never written to raw/.
The API key is never written to raw/, stdout or the session store.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
CELL_DIR = HERE.parents[1]                 # docs/experiments/E17B-PI-OPENAI-GATE
REPO = HERE.parents[4]                     # repository root
if not (REPO / "pyproject.toml").exists() or not (REPO / "src" / "karc").is_dir():
    raise SystemExit(f"E17B: repository root misresolved as {REPO} (E15 §8.4 off-by-one guard)")

FIXTURE = REPO / "fixture" / "e4-v2"
BRIDGE = REPO / "integrations" / "pi" / "karc-mcp-bridge" / "index.ts"
PROBE_EXT = CELL_DIR / "scripts" / "tool-registry-probe.ts"
RAW = CELL_DIR / "raw"
RUN = REPO / "tmp" / "e17b"                # gitignored scratch; NOT tmp/e17
PI_BIN = REPO / "tmp" / "vendor" / "piprobe" / "node_modules" / ".bin" / "pi"

PROVIDER = "openai"
# Pinned to E16 §1's model so the baseline ("the provider does send
# cache_write_tokens: cold 8,379/8,381, warm 19") transfers without a model
# confound.  pi routes provider `openai` through api/openai-responses.js, i.e.
# POST https://api.openai.com/v1/responses — the same endpoint E16 used.
MODEL = "gpt-5.6-luna"
THINKING = "off"
FIXTURE_SESSION = "S000"
SESSION_LENGTH = 8
TOOLS = ("karc_search", "karc_get")

# Paths of the user's real pi state that must stay untouched (isolation proof).
USER_PI_PATHS = (
    Path.home() / ".pi",
    Path.home() / ".pi" / "agent",
    Path.home() / ".pi" / "agent" / "auth.json",
    Path.home() / ".pi" / "agent" / "models-store.json",
)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def pi_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Isolated pi environment. The user's ~/.pi is never read or written:
    PI_CODING_AGENT_DIR redirects config+auth.json, PI_CODING_AGENT_SESSION_DIR
    redirects session storage. Both live under tmp/e17b (gitignored).

    E17 §6: pi prefers auth.json over env, so without this redirect a user's
    OAuth credential would silently win over the metered API key and the run
    would bill a subscription route instead."""
    env = {
        "HOME": os.environ["HOME"],
        "PATH": os.environ["PATH"],
        "LANG": os.environ.get("LANG", "en_US.UTF-8"),
        "PI_CODING_AGENT_DIR": str(RUN / "pi-home"),
        "PI_CODING_AGENT_SESSION_DIR": str(RUN / "sessions"),
        "PI_OFFLINE": "1",
        "PI_SKIP_VERSION_CHECK": "1",
        "PI_TELEMETRY": "0",
        "KARC_PI_MCP_CONFIG": str(RUN / "work" / ".mcp.json"),
        "KARC_PI_MCP_SERVER": "karc",
    }
    if extra:
        env.update(extra)
    return env


# ------------------------------------------------------------------ mtime

def phase_mtime() -> dict:
    """Snapshot the user's pi state. Compared before/after the model phase to
    show the run never touched ~/.pi (E17 §6 did the same for the Anthropic leg)."""
    return {"paths": {str(path): (
        {"exists": True, "mtime_ns": path.stat().st_mtime_ns,
         "size": path.stat().st_size if path.is_file() else None}
        if path.exists() else {"exists": False}) for path in USER_PI_PATHS}}


# ------------------------------------------------------------------- auth

def phase_auth() -> dict:
    """Model-free: in the isolated home, does pi see any OpenAI credential on
    disk?  It must not — the only credential in this cell arrives via env."""
    (RUN / "pi-home").mkdir(parents=True, exist_ok=True)
    (RUN / "sessions").mkdir(parents=True, exist_ok=True)
    results = {}
    for with_key in (False, True):
        extra = {}
        if with_key:
            key = os.environ.get("KARC_OPENAI_KEY")
            if not key:
                results["with_env_key"] = {"skipped": "KARC_OPENAI_KEY unset"}
                continue
            extra["OPENAI_API_KEY"] = key
        proc = subprocess.run(
            [str(PI_BIN), "auth", "check", "--provider", PROVIDER, "--json"],
            cwd=str(REPO), env=pi_env(extra), text=True, capture_output=True,
            timeout=120)
        # `--credentials` is deliberately NOT passed: it would print the secret.
        results["with_env_key" if with_key else "no_env_key"] = {
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip()[:600],
            "stderr_tail": proc.stderr.strip().splitlines()[-3:],
        }
    auth_json = RUN / "pi-home" / "auth.json"
    results["isolated_auth_json"] = {
        "exists": auth_json.exists(),
        "bytes": auth_json.stat().st_size if auth_json.exists() else None,
    }
    return results


# ---------------------------------------------------------------- prepare

def phase_prepare() -> dict:
    """Managed-memory (K-ARC MCP) arm materialization, identical to E17."""
    sys.path.insert(0, str(REPO / "src"))
    from karc.bench import mcp_index

    manifest = json.loads((FIXTURE / "manifest.json").read_text(encoding="utf-8"))
    work = RUN / "work"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    shutil.copytree(FIXTURE / "repo", work, dirs_exist_ok=True)

    managed = sorted(manifest["artifacts"],
                     key=lambda vid: manifest["artifacts"][vid]["path"])
    managed_root = work / ".karc" / "managed"
    for version in managed:
        relative = manifest["artifacts"][version]["path"]
        source = work / relative
        target = managed_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.exists():
            shutil.move(str(source), str(target))
    index = work / ".karc" / "index.db"
    mcp_index.build_index(work, manifest, managed,
                          managed_root=".karc/managed", db_path=index)

    loader = (f"import sys;sys.path.insert(0,{str(REPO / 'src')!r});"
              "from karc.cli import main;raise SystemExit(main())")
    mcp_cfg = {"mcpServers": {"karc": {
        "command": sys.executable,
        "args": ["-c", loader, "mcp", "serve", "--root", str(work),
                 "--db", str(index), "--runtime", "pi"],
    }}}
    (work / ".mcp.json").write_text(json.dumps(mcp_cfg, indent=2), encoding="utf-8")

    (RUN / "pi-home").mkdir(parents=True, exist_ok=True)
    (RUN / "sessions").mkdir(parents=True, exist_ok=True)
    (RUN / "pi-home" / "settings.json").write_text(json.dumps({
        "defaultProjectTrust": "never",
        "quietStartup": True,
        "enableInstallTelemetry": False,
        "enableAnalytics": False,
        "compaction": {"enabled": False},
    }, indent=2), encoding="utf-8")

    return {
        "workdir": str(work),
        "managed_artifacts": len(managed),
        "index_db_bytes": index.stat().st_size,
        "mcp_config_sha256": sha256((work / ".mcp.json").read_text(encoding="utf-8")),
        "fixture_manifest_sha256": manifest["manifest_sha256"],
    }


# --------------------------------------------------------------- registry

def phase_registry() -> dict:
    """Model-free: start pi in rpc mode on the OpenAI leg with the bridge + the
    E17B probe, ask for get_state, and read pi's own tool registry out of the
    probe file.  Guards against the `--tools` typo class (E17 §7.2)."""
    out = RUN / "registry-probe.json"
    if out.exists():
        out.unlink()
    audit = RUN / "registry-bridge-audit.jsonl"
    if audit.exists():
        audit.unlink()
    argv = [
        str(PI_BIN), "--mode", "rpc",
        "--provider", PROVIDER, "--model", MODEL, "--thinking", THINKING,
        "--no-extensions", "-e", str(BRIDGE), "-e", str(PROBE_EXT),
        "--no-builtin-tools", "--tools", ",".join(TOOLS),
        "--no-context-files", "--no-skills", "--no-prompt-templates", "--no-themes",
        "--no-approve", "--no-session",
    ]
    extra = {"KARC_E17B_PROBE_OUT": str(out), "KARC_PI_BRIDGE_AUDIT": str(audit)}
    key = os.environ.get("KARC_OPENAI_KEY")
    if key:
        # No model call is made in rpc get_state; the key only lets pi resolve
        # the provider so the registry can be dumped at all.
        extra["OPENAI_API_KEY"] = key
    proc = subprocess.run(argv, cwd=str(RUN / "work"), env=pi_env(extra),
                          input='{"type":"get_state"}\n', text=True,
                          capture_output=True, timeout=180)
    responses = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            responses.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    state = next((r.get("data", {}) for r in responses
                  if r.get("command") == "get_state"), {})
    probe = json.loads(out.read_text(encoding="utf-8")) if out.exists() else None
    return {
        "argv_tail": argv[1:],
        "returncode": proc.returncode,
        "stderr_tail": proc.stderr.strip().splitlines()[-5:],
        "rpc_event_types": sorted({str(r.get("type")) for r in responses}),
        "get_state": {
            "model": (state.get("model") or {}).get("id"),
            "provider": (state.get("model") or {}).get("provider"),
            "api": (state.get("model") or {}).get("api"),
            "autoCompactionEnabled": state.get("autoCompactionEnabled"),
            "messageCount": state.get("messageCount"),
        },
        "probe": probe,
        "bridge_audit_lines": (len(audit.read_text(encoding="utf-8").splitlines())
                               if audit.exists() else 0),
    }


# ------------------------------------------------------------------ smoke

def turn_prompt(task: dict, position: int) -> str:
    """Coherent technical prose (E16 §4: incoherent prefixes draw refusals).
    Byte-identical shape to E17's prompt so the two legs stay comparable."""
    return (
        f"Benchmark task {position}/{SESSION_LENGTH}. {task['prompt']} "
        "Return only the requested assignment. Use knowledge from earlier turns "
        "of this session when it already contains the fact; otherwise use only "
        "the K-ARC tools karc_search and karc_get to find and read it."
    )


def parse_json_stream(lines: list[str]) -> dict:
    """Aggregate one pi --mode json run. `usage` is read from message_end
    (authoritative final message, docs/json.md); no closure relation between
    fields is assumed here.  `usage_keys` is retained because the *presence* of
    a key and a value of 0 are different facts (E16 §1.2)."""
    api_calls: list[dict] = []
    tool_starts: list[str] = []
    tool_ends: list[dict] = []
    event_types: dict[str, int] = {}
    errors: list[str] = []
    for line in lines:
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            errors.append("unparsable-line")
            continue
        etype = str(event.get("type"))
        event_types[etype] = event_types.get(etype, 0) + 1
        if etype == "tool_execution_start":
            tool_starts.append(str(event.get("toolName")))
        elif etype == "tool_execution_end":
            tool_ends.append({"tool": str(event.get("toolName")),
                              "isError": bool(event.get("isError"))})
        elif etype == "message_end":
            message = event.get("message") or {}
            if message.get("role") != "assistant":
                continue
            usage = message.get("usage") or {}
            api_calls.append({
                "input": usage.get("input"),
                "output": usage.get("output"),
                "cacheRead": usage.get("cacheRead"),
                "cacheWrite": usage.get("cacheWrite"),
                "cacheWrite1h": usage.get("cacheWrite1h"),
                "cacheWrite1h_key_present": "cacheWrite1h" in usage,
                "reasoning": usage.get("reasoning"),
                "totalTokens": usage.get("totalTokens"),
                "usage_keys": sorted(usage.keys()),
                "cost_total": (usage.get("cost") or {}).get("total"),
                "cost_cacheWrite": (usage.get("cost") or {}).get("cacheWrite"),
                "stopReason": message.get("stopReason"),
                "rawStopReason": message.get("rawStopReason"),
                "endTurn": message.get("endTurn"),
                "errorMessage_present": bool(message.get("errorMessage")),
                "responseModel": message.get("responseModel") or message.get("model"),
                "content_block_types": [str(block.get("type"))
                                        for block in (message.get("content") or [])],
            })
        elif etype in ("extension_error", "error"):
            errors.append(etype)
    return {"api_calls": api_calls, "tool_starts": tool_starts,
            "tool_ends": tool_ends, "event_types": event_types,
            "stream_errors": errors}


def closure_of(call: dict) -> dict:
    """Decide the inclusion relation from numbers only — never from docs."""
    inp = int(call.get("input") or 0)
    out = int(call.get("output") or 0)
    read = int(call.get("cacheRead") or 0)
    write = int(call.get("cacheWrite") or 0)
    total = int(call.get("totalTokens") or 0)
    return {
        "input": inp, "output": out, "cacheRead": read, "cacheWrite": write,
        "totalTokens": total,
        "input_ge_read_plus_write": inp >= read + write,
        # Mutually exclusive whenever read + write > 0:
        "inclusive_hypothesis_total_eq_input_plus_output": total == inp + out,
        "exclusive_hypothesis_total_eq_four_sum": total == inp + out + read + write,
        "discriminating": (read + write) > 0,
        "cacheWrite1h_le_cacheWrite": int(call.get("cacheWrite1h") or 0) <= write,
    }


def _sum(rows: list[dict], key: str) -> int:
    return sum(int(row.get(key) or 0) for row in rows)


def phase_smoke(*, regime: str, turns: int, session_tag: str) -> dict:
    key = os.environ.get("KARC_OPENAI_KEY")
    if not key:
        raise SystemExit("E17B: KARC_OPENAI_KEY is required for the smoke phase")
    tasks = [task for task in json.loads((FIXTURE / "tasks.json").read_text(encoding="utf-8"))
             if task["session_id"] == FIXTURE_SESSION]
    tasks.sort(key=lambda task: task["seq"])
    if len(tasks) < turns:
        raise SystemExit(f"E17B: fixture session {FIXTURE_SESSION} has {len(tasks)} tasks")

    # A never-before-used session id is what makes turn 1 cold on this leg:
    # pi sends `prompt_cache_key = clampOpenAIPromptCacheKey(sessionId)`
    # (pi-ai dist/api/openai-responses.js:219), so a fresh id cannot hit an
    # existing cache bucket.  The regime-specific system-prompt suffix splits
    # the prefix bytes as well (E17 §7.1).
    session_id = f"e17b-{session_tag}"
    audit = RUN / f"bridge-audit-{session_tag}.jsonl"
    if audit.exists():
        audit.unlink()
    extra = {"OPENAI_API_KEY": key, "KARC_PI_BRIDGE_AUDIT": str(audit)}
    if regime == "long":
        # On this leg PI_CACHE_RETENTION=long means prompt_cache_retention:"24h",
        # not Anthropic's ttl:"1h" (openai-responses.js:64).
        extra["PI_CACHE_RETENTION"] = "long"
    elif regime != "short":
        raise SystemExit(f"E17B: unknown retention regime {regime!r}")

    rows = []
    audit_before = 0
    for position, task in enumerate(tasks[:turns], start=1):
        prompt = turn_prompt(task, position)
        argv = [
            str(PI_BIN), "-p", prompt, "--mode", "json",
            "--provider", PROVIDER, "--model", MODEL, "--thinking", THINKING,
            "--no-extensions", "-e", str(BRIDGE),
            "--no-builtin-tools", "--tools", ",".join(TOOLS),
            "--no-context-files", "--no-skills", "--no-prompt-templates", "--no-themes",
            "--no-approve",
            "--append-system-prompt", f"E17B accounting regime: retention={regime}.",
            "--session-dir", str(RUN / "sessions"), "--session-id", session_id,
        ]
        started = time.time()
        proc = subprocess.run(argv, cwd=str(RUN / "work"), env=pi_env(extra),
                              text=True, capture_output=True, timeout=900)
        elapsed = time.time() - started
        # Transient safety net against paying twice for a parser bug: the raw
        # event stream is kept under tmp/e17b (gitignored) so it can be
        # re-parsed, and is deleted at teardown. It never enters raw/ (R-9).
        streams = RUN / "streams"
        streams.mkdir(parents=True, exist_ok=True)
        (streams / f"{session_tag}-turn{position}.jsonl").write_text(
            proc.stdout, encoding="utf-8")
        parsed = parse_json_stream(proc.stdout.splitlines())
        audit_lines = (audit.read_text(encoding="utf-8").splitlines()
                       if audit.exists() else [])
        audit_turn = [json.loads(line) for line in audit_lines[audit_before:]]
        audit_before = len(audit_lines)
        calls = parsed["api_calls"]
        row = {
            "regime": regime,
            "position": position,
            "task_id": task["task_id"],
            "prompt_sha256": sha256(prompt),
            "prompt_chars": len(prompt),
            "returncode": proc.returncode,
            "wall_seconds": round(elapsed, 2),
            "api_calls": calls,
            "api_call_count": len(calls),
            "usage_turn": {
                "input": _sum(calls, "input"),
                "output": _sum(calls, "output"),
                "cacheRead": _sum(calls, "cacheRead"),
                "cacheWrite": _sum(calls, "cacheWrite"),
                "cacheWrite1h": _sum(calls, "cacheWrite1h"),
                "reasoning": _sum(calls, "reasoning"),
                "cost_total": round(sum(float(call.get("cost_total") or 0.0)
                                        for call in calls), 6),
            },
            "stop_reasons": [call["stopReason"] for call in calls],
            "raw_stop_reasons": [call["rawStopReason"] for call in calls],
            "tool_execution_start_count": len(parsed["tool_starts"]),
            "tool_execution_start_names": parsed["tool_starts"],
            "tool_execution_end_errors": sum(1 for end in parsed["tool_ends"]
                                             if end["isError"]),
            "bridge_audit_lines_this_turn": len(audit_turn),
            "bridge_audit_outcomes": [entry.get("outcome") for entry in audit_turn],
            "bridge_audit_mcp_tools": [entry.get("mcpTool") for entry in audit_turn],
            "event_types": parsed["event_types"],
            "stream_errors": parsed["stream_errors"],
            "stderr_tail": proc.stderr.strip().splitlines()[-4:],
        }
        row["closure"] = [closure_of(call) for call in calls]
        rows.append(row)
        print(json.dumps({k: row[k] for k in (
            "regime", "position", "returncode", "api_call_count", "usage_turn",
            "stop_reasons", "raw_stop_reasons", "tool_execution_start_count",
            "bridge_audit_lines_this_turn")}, ensure_ascii=False), flush=True)
        if proc.returncode != 0:
            print("E17B: pi exited non-zero; stopping this regime", flush=True)
            break
    return {"regime": regime, "session_id_sha256": sha256(session_id),
            "turns_requested": turns, "turns_executed": len(rows),
            "model": MODEL, "provider": PROVIDER, "thinking": THINKING,
            "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["mtime", "auth", "prepare", "registry", "smoke"])
    parser.add_argument("--regime", default="short", choices=["short", "long"])
    parser.add_argument("--turns", type=int, default=5)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    if args.phase == "mtime":
        result = phase_mtime()
        out = args.out or "mtime.json"
    elif args.phase == "auth":
        result = phase_auth()
        out = args.out or "auth-check.json"
    elif args.phase == "prepare":
        result = phase_prepare()
        out = args.out or "prepare.json"
    elif args.phase == "registry":
        result = phase_registry()
        out = args.out or "registry-probe.json"
    else:
        tag = args.tag or args.regime
        result = phase_smoke(regime=args.regime, turns=args.turns, session_tag=tag)
        out = args.out or f"smoke-{tag}.json"
    result["_meta"] = {"phase": args.phase, "pi_version": "0.84.3",
                       "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                         time.gmtime())}
    (RAW / out).write_text(json.dumps(result, indent=2, sort_keys=True,
                                      ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {RAW / out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
