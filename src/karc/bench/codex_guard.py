"""Vetted Codex benchmark hook.

Modes are selected through ``KARC_CODEX_GUARD_MODE``:

``observe``
    Record reduced hook metadata and never influence a tool call.
``deny-all``
    Deny every local tool call.  This operationalizes the closed-book E0C-3
    arm because Codex has no public max-turns/allowed-tools flag.
``deny-managed``
    Deny native access to ``.karc/managed`` while always allowing K-ARC MCP
    calls.  This is the E2C-2 C2 condition.

The audit log contains identifiers, tool names, hashes, and decisions only.
Prompts, commands, file bodies, tool responses, and credentials are excluded.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import fcntl
from datetime import datetime, timezone
from pathlib import Path

MAX_STDIN_BYTES = 1024 * 1024
MANAGED_REL = ".karc/managed"


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def _squash(value: str) -> str:
    """Normalize common quoting/concatenation evasions without executing input."""
    return re.sub(r"[\s'\"`\\+]+", "", value).lower()


def _targets_managed(tool_input, cwd: str | None) -> bool:
    absolute = str((Path(cwd or os.getcwd()) / MANAGED_REL).resolve())
    needles = (_squash(MANAGED_REL), _squash(absolute))
    for value in _strings(tool_input):
        compact = _squash(value)
        if any(needle in compact for needle in needles):
            return True
    return False


def _logical_mcp(tool_name: str) -> tuple[str | None, str | None]:
    if not tool_name.startswith("mcp__"):
        return None, None
    parts = tool_name.split("__", 2)
    if len(parts) != 3:
        return None, None
    return parts[1], parts[2]


def _content_access_kind(tool_name: str, haystack: str) -> str | None:
    """Classify a tool request without retaining its command text.

    ``rg --files`` is path discovery, not document-content exposure.  The
    distinction matters because the old resolver counted every Bash ``rg`` as
    an unresolved document read even when it returned filenames only.
    """
    if tool_name in {"Read", "Grep"}:
        return "document-content"
    if tool_name != "Bash":
        return None
    compact = haystack.lower()
    if re.search(r"(?:^|[;&|]\s*)rg\s+--files(?:\s|$)", compact):
        return "path-listing"
    markers = ("cat ", "rg ", "grep ", "sed ", "head ", "tail ",
               "read_file", "open_file")
    return "document-content" if any(marker in compact for marker in markers) else None


def _injected_versions(raw: str | None, artifacts: dict) -> list[str]:
    if not raw:
        return []
    try:
        values = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(values, list):
        return []
    return sorted({value for value in values
                   if isinstance(value, str) and value in artifacts})


def _document_resolution(tool_input, tool_name: str, manifest_path: str | None,
                         seq: int | None, injected_raw: str | None = None
                         ) -> tuple[list[dict], int, str | None]:
    """Resolve E4 document IDs before input strings are reduced to hashes.

    Exact paths and fixture fact keys are both accepted.  The latter covers a
    content-returning `rg/grep` query whose command names the key rather than
    the result path.  Only IDs and access metadata leave this function.
    """
    if not manifest_path or not tool_input:
        return [], 0, None
    try:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return [], 1, "manifest-unavailable"
    strings = list(_strings(tool_input))
    haystack = "\n".join(strings)
    access_kind = _content_access_kind(tool_name, haystack)
    if access_kind != "document-content":
        return [], 0, access_kind
    resolved: dict[str, dict] = {}
    artifacts = manifest.get("artifacts", {})

    def add(version_id: str, source: str) -> None:
        entry = artifacts[version_id]
        resolved[version_id] = {
            "artifact_id": entry.get("artifact_id"),
            "version_id": version_id,
            "access_type": tool_name or "unknown",
            "resolution_source": source,
            "seq": seq,
        }

    for version_id, entry in artifacts.items():
        path = str(entry.get("path") or "")
        fact_key = str(entry.get("fact_key") or "")
        parent = str(Path(path).parent) if path else ""
        if path and path in haystack:
            add(version_id, "manifest-path")
        elif fact_key and fact_key in haystack:
            add(version_id, "fact-key")
        elif parent and parent in haystack:
            # Shell glob and directory-form reads name a manifest directory,
            # not necessarily a full version filename.  Resolve every version
            # inside that named directory rather than losing the doc-ID join.
            add(version_id, "manifest-directory")

    agents_named = bool(re.search(r"(?:^|[/\s'\"])AGENTS\.md(?:$|[\s'\"])",
                                  haystack))
    if agents_named:
        for version_id in _injected_versions(injected_raw, artifacts):
            add(version_id, "injected-agents-context")

    unresolved = int(not resolved)
    return [resolved[key] for key in sorted(resolved)], unresolved, access_kind


def evaluate(payload: dict, mode: str, *, tool_ordinal: int | None = None,
             tool_limit: int | None = None,
             limit_denied: bool = False) -> tuple[bool, dict]:
    event = str(payload.get("hook_event_name") or "unknown")
    tool = str(payload.get("tool_name") or "")
    server, mcp_tool = _logical_mcp(tool)
    denied = False
    if event == "PreToolUse":
        if mode == "deny-all":
            denied = True
        elif mode == "deny-managed" and server == "karc":
            denied = False
        elif mode == "deny-managed":
            denied = _targets_managed(payload.get("tool_input"), payload.get("cwd"))
        if mode != "deny-all" and limit_denied:
            denied = True

    strings = list(_strings(payload.get("tool_input")))
    resolved_documents, unresolved, access_kind = _document_resolution(
        payload.get("tool_input"), tool,
        os.environ.get("KARC_CODEX_FIXTURE_MANIFEST"), tool_ordinal,
        os.environ.get("KARC_CODEX_INJECTED_ARTIFACTS"),
    ) if event == "PreToolUse" else ([], 0, None)
    document_policy = os.environ.get("KARC_CODEX_UNRESOLVED_POLICY", "observe")
    blocked_unresolved = 0
    if event == "PreToolUse" and unresolved and document_policy == "deny":
        # An unmatched content request cannot create an unobservable exposure:
        # deny it symmetrically for every arm and retain only the reduced count.
        denied = True
        blocked_unresolved = unresolved
        unresolved = 0
    reduced = {
        "schema_version": 1,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "hook_event_name": event,
        "session_id": payload.get("session_id"),
        "turn_id": payload.get("turn_id"),
        "tool_use_id": payload.get("tool_use_id"),
        "tool_name": tool or None,
        "mcp_server": server,
        "mcp_tool": mcp_tool,
        "model": payload.get("model"),
        "mode": mode,
        "denied": denied,
        "tool_ordinal": tool_ordinal,
        "tool_limit": tool_limit,
        "limit_denied": limit_denied,
        "permission_decision": (
            "allow" if event == "PermissionRequest" and server == "karc"
            and mode != "deny-all" else None
        ),
        "input_hashes": sorted({_hash(v) for v in strings}),
        "input_string_count": len(strings),
        "resolved_documents": resolved_documents,
        "unresolved_document_accesses": unresolved,
        "blocked_unresolved_document_accesses": blocked_unresolved,
        "document_access_kind": access_kind,
        "document_access_policy": document_policy,
        "reduced": True,
    }
    return denied, reduced


def _reserve_tool_slot(payload: dict, limit_text: str | None,
                       state_path: str | None) -> tuple[int | None, int | None, bool]:
    """Atomically number PreToolUse events and deny only calls after limit."""
    if payload.get("hook_event_name") != "PreToolUse" or not limit_text or not state_path:
        return None, None, False
    try:
        limit = int(limit_text)
    except (TypeError, ValueError):
        return None, None, False
    if limit < 0:
        return None, None, False
    path = Path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        raw = os.read(fd, 64).decode("ascii", errors="ignore").strip()
        ordinal = (int(raw) if raw.isdigit() else 0) + 1
        os.lseek(fd, 0, os.SEEK_SET)
        os.ftruncate(fd, 0)
        os.write(fd, str(ordinal).encode("ascii"))
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
    return ordinal, limit, ordinal > limit


def _append(path: str | None, row: dict) -> None:
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, line.encode("utf-8"))
    finally:
        os.close(fd)


def main() -> int:
    try:
        raw = sys.stdin.buffer.read(MAX_STDIN_BYTES + 1)
        if len(raw) > MAX_STDIN_BYTES:
            return 0
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            return 0
        mode = os.environ.get("KARC_CODEX_GUARD_MODE", "observe")
        ordinal, limit, limit_denied = _reserve_tool_slot(
            payload, os.environ.get("KARC_CODEX_MAX_TOOL_CALLS"),
            os.environ.get("KARC_CODEX_GUARD_STATE"),
        )
        denied, row = evaluate(payload, mode, tool_ordinal=ordinal,
                               tool_limit=limit, limit_denied=limit_denied)
        _append(os.environ.get("KARC_CODEX_GUARD_LOG"), row)
        if denied:
            print(json.dumps({
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": "Blocked by the K-ARC benchmark guard.",
                }
            }))
        elif (payload.get("hook_event_name") == "PermissionRequest"
              and row.get("permission_decision") == "allow"):
            # Approve only the local, read-only K-ARC benchmark MCP.  Every
            # other request remains under Codex's normal approval policy.
            print(json.dumps({
                "hookSpecificOutput": {
                    "hookEventName": "PermissionRequest",
                    "decision": {"behavior": "allow"},
                }
            }))
        elif payload.get("hook_event_name") == "Stop":
            # Stop requires JSON on stdout when a command hook exits zero.
            print("{}")
    except Exception:
        # A benchmark observation hook must never fail or alter the agent.
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
