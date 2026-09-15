"""Controlled Codex homes for reproducible benchmark execution.

The benchmark must not inherit a user's Codex config, hooks, rules, plugins,
skills, memories, or rollout state.  ``ControlledCodexHome`` creates a fresh
``CODEX_HOME`` per invocation and bridges only the existing authentication
file through a symlink.  The link target and credential contents are never
copied, hashed, or included in result packages.

The generated ``hooks.json`` contains only the vetted benchmark guard.  It is
safe to run with ``--dangerously-bypass-hook-trust`` because both the command
and this module's source are part of the frozen experiment commit.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


_ENV_ALLOWLIST = (
    "PATH",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
    "TMPDIR",
    "TMP",
    "TEMP",
    "TERM",
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def default_auth_source() -> Path:
    """Return the active user's Codex auth file without reading its contents."""
    configured = os.environ.get("CODEX_HOME")
    base = Path(configured).expanduser() if configured else Path.home() / ".codex"
    return base / "auth.json"


@dataclass
class ControlledCodexHome:
    """Context manager for one isolated, ephemeral Codex invocation."""

    auth_source: Path | None = None
    python_executable: str = sys.executable
    temp_root: Path | None = None
    extra_environment: dict[str, str] = field(default_factory=dict)
    path: Path | None = field(init=False, default=None)
    audit_path: Path | None = field(init=False, default=None)
    hooks_path: Path | None = field(init=False, default=None)
    manifest: dict = field(init=False, default_factory=dict)

    def __enter__(self) -> "ControlledCodexHome":
        root = str(self.temp_root) if self.temp_root else None
        self.path = Path(tempfile.mkdtemp(prefix="karc-codex-home-", dir=root))
        self.path.chmod(0o700)
        self.audit_path = self.path / "guard-events.jsonl"
        self.hooks_path = self.path / "hooks.json"

        auth = Path(self.auth_source) if self.auth_source else default_auth_source()
        if not auth.is_file():
            raise RuntimeError(f"Codex auth bridge unavailable: {auth}")
        os.symlink(auth, self.path / "auth.json")

        command = shlex.join(
            [str(self.python_executable), "-m", "karc.bench.codex_guard"]
        )
        handler = {"type": "command", "command": command, "timeout": 10}
        hooks = {
            "description": "K-ARC reproducible benchmark guard (generated)",
            "hooks": {
                "SessionStart": [{"matcher": "startup", "hooks": [handler]}],
                "PreToolUse": [{"matcher": "*", "hooks": [handler]}],
                "PermissionRequest": [{"matcher": "*", "hooks": [handler]}],
                "PostToolUse": [{"matcher": "*", "hooks": [handler]}],
                "Stop": [{"hooks": [handler]}],
            },
        }
        self.hooks_path.write_text(
            json.dumps(hooks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        self.manifest = {
            "schema_version": 1,
            "auth_bridge": "read-only-source-symlink",
            "generated_files": [
                {"name": "hooks.json", "sha256": _sha256(self.hooks_path)},
            ],
            "excluded": ["auth.json", "guard-events.jsonl", "tmp"],
        }
        return self

    def child_environment(self, overrides: dict[str, str] | None = None) -> dict[str, str]:
        if self.path is None or self.audit_path is None:
            raise RuntimeError("ControlledCodexHome is not active")
        env = {k: os.environ[k] for k in _ENV_ALLOWLIST if k in os.environ}
        env.update(self.extra_environment)
        env.update(overrides or {})
        env["CODEX_HOME"] = str(self.path)
        env["KARC_CODEX_GUARD_LOG"] = str(self.audit_path)
        env.setdefault("KARC_CODEX_GUARD_MODE", "observe")
        return env

    def read_audit_events(self) -> list[dict]:
        if self.audit_path is None or not self.audit_path.exists():
            return []
        out: list[dict] = []
        for line in self.audit_path.read_text(encoding="utf-8").splitlines():
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                out.append(obj)
        return out

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.path is not None:
            shutil.rmtree(self.path, ignore_errors=True)
