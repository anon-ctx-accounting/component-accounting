"""Runtime detection + init plan (managed MCP registration & AGENTS.md block).

Universal floor per integration-options §1/§2: register the ``karc`` MCP server
with each detected runtime and install a runtime-neutral managed instruction
block. All planning is pure (no writes) so ``init --dry-run`` and the diff
preview show exactly what would change; ``apply_*`` performs idempotent,
marker-guarded writes.

Design points:
- Managed block text uses the *logical* server/tool names ("the `karc` MCP
  server's `search` and `get` tools") — never a runtime-specific prefix, which
  would be a false instruction in another runtime (§2.1-F).
- Idempotency via ``<!-- k-arc:managed:start/end -->`` markers.
- Hermes first-match warning: if ``.hermes.md`` exists, the AGENTS.md block
  will not be seen by Hermes (§2.1); we surface the choice instead of silently
  failing.
- stdlib only: Claude Code ``.mcp.json`` is JSON (safe merge); Hermes YAML has
  no stdlib parser, so its MCP snippet is emitted for the user to add.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

MANAGED_START = "<!-- k-arc:managed:start (do not edit inside this block) -->"
MANAGED_END = "<!-- k-arc:managed:end -->"

MANAGED_BODY = (
    "When project knowledge is managed by K-ARC, use the `karc` MCP server's "
    "`search` and `get` tools to discover and read it. Treat K-ARC results as "
    "project-local evidence and verify task outcomes normally."
)

CODEX_TOML_START = "# k-arc:managed:start"
CODEX_TOML_END = "# k-arc:managed:end"
CODEX_HOOK_IDENTITY = "karc hook --runtime codex"


def managed_block() -> str:
    return f"{MANAGED_START}\n{MANAGED_BODY}\n{MANAGED_END}\n"


@dataclass
class RuntimeInfo:
    name: str
    key: str
    found: bool
    version: str | None = None
    config_path: str | None = None
    note: str | None = None

    def as_dict(self) -> dict:
        return {
            "name": self.name, "key": self.key, "found": self.found,
            "version": self.version, "config_path": self.config_path, "note": self.note,
        }


def _version(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    text = (out.stdout or out.stderr or "").strip()
    return text.splitlines()[0] if text else None


def detect_runtimes(home: str | None = None) -> list[RuntimeInfo]:
    """Read-only detection (which/--version + config file existence)."""
    home = home or os.path.expanduser("~")
    infos: list[RuntimeInfo] = []

    claude_bin = shutil.which("claude")
    cc_cfg = os.path.join(home, ".claude.json")
    infos.append(RuntimeInfo(
        "Claude Code", "claude-code", claude_bin is not None,
        version=_version(["claude", "--version"]) if claude_bin else None,
        config_path=cc_cfg if os.path.exists(cc_cfg) else None,
    ))

    hermes_bin = shutil.which("hermes")
    h_cfg = os.path.join(home, ".hermes", "config.yaml")
    infos.append(RuntimeInfo(
        "Hermes Agent", "hermes", hermes_bin is not None,
        version=_version(["hermes", "--version"]) if hermes_bin else None,
        config_path=h_cfg if os.path.exists(h_cfg) else None,
        note="loads only the first match of .hermes.md > AGENTS.md > CLAUDE.md",
    ))

    codex_bin = shutil.which("codex")
    infos.append(RuntimeInfo(
        "Codex CLI", "codex", codex_bin is not None,
        version=_version(["codex", "--version"]) if codex_bin else None,
        config_path=os.path.join(home, ".codex", "config.toml"),
    ))

    oc_bin = shutil.which("opencode")
    infos.append(RuntimeInfo(
        "OpenCode", "opencode", oc_bin is not None,
        version=_version(["opencode", "--version"]) if oc_bin else None,
    ))
    return infos


# ---------------------------------------------------------------------------
# MCP registration (Claude Code .mcp.json)
# ---------------------------------------------------------------------------

def mcp_server_entry(root: str, db_path: str) -> dict:
    return {
        "command": "karc",
        "args": ["mcp", "serve", "--root", root, "--db", db_path],
    }


def plan_claude_mcp(root: str, db_path: str) -> dict:
    """What `.mcp.json` change init would make (idempotent)."""
    mcp_path = os.path.join(root, ".mcp.json")
    existing = {}
    if os.path.exists(mcp_path):
        try:
            existing = json.loads(Path(mcp_path).read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing = {}
    already = "karc" in (existing.get("mcpServers") or {})
    return {
        "path": mcp_path,
        "already_configured": already,
        "entry": mcp_server_entry(root, db_path),
    }


def apply_claude_mcp(root: str, db_path: str) -> dict:
    plan = plan_claude_mcp(root, db_path)
    if plan["already_configured"]:
        return {"path": plan["path"], "changed": False, "reason": "already configured"}
    mcp_path = Path(plan["path"])
    data = {}
    if mcp_path.exists():
        try:
            data = json.loads(mcp_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            data = {}
    data.setdefault("mcpServers", {})["karc"] = plan["entry"]
    mcp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"path": str(mcp_path), "changed": True}


def hermes_mcp_snippet(root: str, db_path: str) -> str:
    return (
        "mcp_servers:\n"
        "  karc:\n"
        "    command: karc\n"
        f"    args: [\"mcp\", \"serve\", \"--root\", \"{root}\", \"--db\", \"{db_path}\"]\n"
    )


# ---------------------------------------------------------------------------
# Codex project config + hooks (marker/identity-owned, user content preserved)
# ---------------------------------------------------------------------------
def _karc_binary() -> str:
    # Do not resolve the venv Python symlink: its target usually lives in the
    # shared interpreter installation, while the console script is next to the
    # symlink in the active environment.
    names = ("karc.exe", "karc") if os.name == "nt" else ("karc",)
    dirs = (Path(sys.executable).parent, Path(sys.prefix) / "bin",
            Path(sys.prefix) / "Scripts")
    for directory in dirs:
        for name in names:
            candidate = directory / name
            if candidate.is_file():
                return str(candidate.absolute())
    return shutil.which("karc") or "karc"


def codex_mcp_block(root: str, db_path: str, *, command: str | None = None) -> str:
    root = os.path.realpath(root)
    db_path = os.path.realpath(db_path)
    command = command or _karc_binary()
    args = ["mcp", "serve", "--root", root, "--db", db_path, "--runtime", "codex"]
    return (
        f"{CODEX_TOML_START}\n"
        "[mcp_servers.karc]\n"
        f"command = {json.dumps(command)}\n"
        f"args = {json.dumps(args)}\n"
        "required = true\n"
        "startup_timeout_sec = 10\n"
        "tool_timeout_sec = 60\n"
        f"{CODEX_TOML_END}\n"
    )


def _replace_marker_block(text: str, block: str) -> tuple[str | None, str | None]:
    n_start, n_end = text.count(CODEX_TOML_START), text.count(CODEX_TOML_END)
    if n_start != n_end or n_start > 1:
        return None, "malformed K-ARC marker block"
    if n_start == 1:
        pattern = re.compile(
            rf"(?ms)^{re.escape(CODEX_TOML_START)}$.*?"
            rf"^{re.escape(CODEX_TOML_END)}[^\n]*(?:\n|$)"
        )
        if not pattern.search(text):
            return None, "malformed K-ARC marker boundaries"
        return pattern.sub(block, text), None
    prefix = text
    if prefix and not prefix.endswith("\n"):
        prefix += "\n"
    if prefix:
        prefix += "\n"
    return prefix + block, None


def plan_codex_mcp(root: str, db_path: str, *, command: str | None = None) -> dict:
    path = Path(root) / ".codex" / "config.toml"
    exists = path.exists()
    try:
        current = path.read_text(encoding="utf-8") if exists else ""
    except OSError as exc:
        return {"path": str(path), "conflict": True, "reason": str(exc)}
    try:
        parsed = tomllib.loads(current) if current.strip() else {}
    except tomllib.TOMLDecodeError as exc:
        return {"path": str(path), "conflict": True,
                "reason": f"invalid TOML: {exc}", "current": current}
    has_marker = CODEX_TOML_START in current
    owned_table = ((parsed.get("mcp_servers") or {}).get("karc"))
    if owned_table is not None and not has_marker:
        return {
            "path": str(path), "conflict": True,
            "reason": "user-owned [mcp_servers.karc] exists outside K-ARC markers",
            "current": current,
        }
    block = codex_mcp_block(root, db_path, command=command)
    proposed, error = _replace_marker_block(current, block)
    if error:
        return {"path": str(path), "conflict": True, "reason": error,
                "current": current}
    # Validate the exact planned file before it can be applied.
    try:
        tomllib.loads(proposed or "")
    except tomllib.TOMLDecodeError as exc:  # pragma: no cover - defensive
        return {"path": str(path), "conflict": True,
                "reason": f"planned TOML invalid: {exc}", "current": current}
    return {
        "path": str(path), "conflict": False, "current": current,
        "proposed": proposed, "already_configured": proposed == current,
        "entry": {"command": command or _karc_binary(),
                  "root": os.path.realpath(root), "db": os.path.realpath(db_path)},
    }


def apply_codex_mcp(root: str, db_path: str, *, command: str | None = None) -> dict:
    plan = plan_codex_mcp(root, db_path, command=command)
    if plan.get("conflict"):
        return {"path": plan["path"], "changed": False, "conflict": True,
                "reason": plan["reason"]}
    if plan["already_configured"]:
        return {"path": plan["path"], "changed": False,
                "reason": "already configured"}
    path = Path(plan["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(plan["proposed"], encoding="utf-8")
    return {"path": str(path), "changed": True}


def codex_hook_command(db_path: str, *, command: str | None = None) -> str:
    import shlex
    return shlex.join([
        command or _karc_binary(), "hook", "--runtime", "codex",
        "--db", os.path.realpath(db_path),
    ])


def codex_hook_groups(db_path: str, *, command: str | None = None) -> dict[str, dict]:
    handler = {
        "type": "command", "command": codex_hook_command(db_path, command=command),
        "timeout": 10, "statusMessage": "Recording K-ARC telemetry",
    }
    return {
        "SessionStart": {"matcher": "startup|resume|clear|compact", "hooks": [handler]},
        "PreToolUse": {"matcher": "Bash|apply_patch|mcp__karc__.*", "hooks": [handler]},
        "PermissionRequest": {"matcher": "Bash|apply_patch|mcp__karc__.*", "hooks": [handler]},
        "PostToolUse": {"matcher": "Bash|apply_patch|mcp__karc__.*", "hooks": [handler]},
        "Stop": {"hooks": [handler]},
    }


def _is_karc_codex_handler(handler: object) -> bool:
    return isinstance(handler, dict) and CODEX_HOOK_IDENTITY in str(handler.get("command", ""))


def plan_codex_hooks(root: str, db_path: str, *, command: str | None = None) -> dict:
    path = Path(root) / ".codex" / "hooks.json"
    try:
        current_text = path.read_text(encoding="utf-8") if path.exists() else ""
        data = json.loads(current_text) if current_text.strip() else {}
    except (OSError, json.JSONDecodeError) as exc:
        return {"path": str(path), "conflict": True,
                "reason": f"invalid hooks JSON: {exc}"}
    if not isinstance(data, dict) or not isinstance(data.get("hooks", {}), dict):
        return {"path": str(path), "conflict": True,
                "reason": "hooks.json must contain an object-valued hooks field"}
    proposed = json.loads(json.dumps(data))
    proposed.setdefault("description", "Project lifecycle hooks")
    hooks = proposed.setdefault("hooks", {})
    desired = codex_hook_groups(db_path, command=command)
    for event, group in desired.items():
        groups = hooks.setdefault(event, [])
        if not isinstance(groups, list):
            return {"path": str(path), "conflict": True,
                    "reason": f"hooks.{event} must be an array"}
        replaced = False
        for existing in groups:
            handlers = existing.get("hooks", []) if isinstance(existing, dict) else []
            for idx, handler in enumerate(handlers if isinstance(handlers, list) else []):
                if _is_karc_codex_handler(handler):
                    existing["matcher"] = group.get("matcher", existing.get("matcher"))
                    if "matcher" not in group:
                        existing.pop("matcher", None)
                    handlers[idx] = group["hooks"][0]
                    replaced = True
        if not replaced:
            groups.append(group)
    proposed_text = json.dumps(proposed, ensure_ascii=False, indent=2) + "\n"
    normalized_current = (
        json.dumps(data, ensure_ascii=False, indent=2) + "\n" if current_text else ""
    )
    return {
        "path": str(path), "conflict": False, "current": current_text,
        "proposed": proposed_text,
        "already_configured": bool(current_text) and proposed_text == normalized_current,
    }


def apply_codex_hooks(root: str, db_path: str, *, command: str | None = None) -> dict:
    plan = plan_codex_hooks(root, db_path, command=command)
    if plan.get("conflict"):
        return {"path": plan["path"], "changed": False, "conflict": True,
                "reason": plan["reason"]}
    if plan["already_configured"]:
        return {"path": plan["path"], "changed": False,
                "reason": "already configured"}
    path = Path(plan["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(plan["proposed"], encoding="utf-8")
    return {"path": str(path), "changed": True}


def probe_codex(binary: str = "codex", runner=None) -> dict:
    """Capability-first Codex probe; version is reproduction metadata only."""
    runner = runner or subprocess.run
    resolved = shutil.which(binary)
    out = {"binary": binary, "resolved": resolved, "available": resolved is not None}
    if resolved is None:
        out["status"] = "unsupported"
        return out
    commands = {
        "version": [binary, "--version"],
        "exec_help": [binary, "exec", "--help"],
        "features": [binary, "features", "list"],
    }
    captured = {}
    try:
        for name, argv in commands.items():
            proc = runner(argv, capture_output=True, text=True, timeout=15)
            if proc.returncode != 0:
                raise subprocess.SubprocessError(f"{name} exit {proc.returncode}")
            captured[name] = (proc.stdout or proc.stderr or "").strip()
    except (OSError, subprocess.SubprocessError) as exc:
        out.update(status="unsupported", error=str(exc))
        return out
    flags = ("--json", "--model", "--sandbox", "--ephemeral",
             "--ignore-user-config", "--strict-config")
    out.update({
        "version": captured["version"].splitlines()[-1] if captured["version"] else None,
        "flags": {flag: flag in captured["exec_help"] for flag in flags},
        "hooks": ("stable" if re.search(
            r"(?m)^hooks\s+stable\s+true$", captured["features"]
        ) else "unavailable"),
    })
    out["status"] = (
        "ok" if all(out["flags"].values()) and out["hooks"] == "stable"
        else "unsupported"
    )
    return out


# ---------------------------------------------------------------------------
# Managed AGENTS.md block + CLAUDE.md @import
# ---------------------------------------------------------------------------

def plan_agents_block(root: str) -> dict:
    agents = os.path.join(root, "AGENTS.md")
    claude = os.path.join(root, "CLAUDE.md")
    hermes_md = os.path.join(root, ".hermes.md")
    agents_text = Path(agents).read_text(encoding="utf-8") if os.path.exists(agents) else None
    claude_text = Path(claude).read_text(encoding="utf-8") if os.path.exists(claude) else None
    agents_has = agents_text is not None and MANAGED_START in agents_text
    claude_has_import = claude_text is not None and "@AGENTS.md" in claude_text
    return {
        "agents_path": agents,
        "agents_exists": agents_text is not None,
        "agents_has_block": agents_has,
        "claude_path": claude,
        "claude_exists": claude_text is not None,
        "claude_has_import": claude_has_import,
        "block": managed_block(),
        "hermes_md_present": os.path.exists(hermes_md),
        "hermes_md_path": hermes_md,
    }


def apply_agents_block(root: str, append_to_hermes: bool = False,
                       bridge_claude: bool = True) -> dict:
    plan = plan_agents_block(root)
    changes = []
    # AGENTS.md
    agents = Path(plan["agents_path"])
    if not plan["agents_has_block"]:
        prefix = ""
        if plan["agents_exists"]:
            cur = agents.read_text(encoding="utf-8")
            prefix = cur if cur.endswith("\n") else cur + "\n"
            prefix += "\n"
        agents.write_text(prefix + managed_block(), encoding="utf-8")
        changes.append({"path": str(agents), "changed": True})
    else:
        changes.append({"path": str(agents), "changed": False, "reason": "block present"})
    # CLAUDE.md bridge is needed only when Claude Code is selected. Codex
    # consumes AGENTS.md natively and Codex-only init must not mutate it.
    if bridge_claude:
        claude = Path(plan["claude_path"])
        if not plan["claude_has_import"]:
            cur = claude.read_text(encoding="utf-8") if plan["claude_exists"] else ""
            if cur and not cur.endswith("\n"):
                cur += "\n"
            claude.write_text(cur + "@AGENTS.md\n", encoding="utf-8")
            changes.append({"path": str(claude), "changed": True})
        else:
            changes.append({"path": str(claude), "changed": False,
                            "reason": "import present"})
    # Hermes first-match: optionally append the block to .hermes.md
    if plan["hermes_md_present"] and append_to_hermes:
        hpath = Path(plan["hermes_md_path"])
        cur = hpath.read_text(encoding="utf-8")
        if MANAGED_START not in cur:
            if not cur.endswith("\n"):
                cur += "\n"
            hpath.write_text(cur + "\n" + managed_block(), encoding="utf-8")
            changes.append({"path": str(hpath), "changed": True})
    return {"changes": changes}
