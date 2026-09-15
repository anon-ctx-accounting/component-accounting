"""Claude Code transcript (JSONL) parser — streaming, read-only.

Observed format (measured 2026-07-17 on ~/.claude/projects, Claude Code
v2.1.211/2.1.212, 266 files / ~199MB; see telemetry doc §2.1):

- Layout: ``<projects-root>/<encoded-project-dir>/<session-uuid>.jsonl`` is a
  top-level (main) session transcript. Subagent transcripts live in
  ``<encoded-project-dir>/<session-uuid>/subagents/agent-<agentId>.jsonl``.
  Subagent lines carry the *parent* ``sessionId`` plus their own ``agentId``
  and ``isSidechain: true`` (measured).
- Line ``type`` values observed: ``user``, ``assistant``, ``system``,
  ``attachment``, ``queue-operation``, ``last-prompt``, ``ai-title``,
  ``agent-name``, ``mode``, ``permission-mode``, ``file-history-snapshot``,
  ``file-history-delta``, ``frame-link``, ``summary`` (older versions).
- Common per-line metadata: ``parentUuid``, ``isSidechain``, ``uuid``,
  ``timestamp`` (ISO8601 Z), ``cwd``, ``sessionId``, ``version``,
  ``gitBranch``, ``promptId``.

Task boundary heuristic (R-4: "최상위의 진짜 사용자 prompt turn").
A line is a REAL user prompt turn iff ALL of:

  1. ``type == 'user'``
  2. ``isSidechain`` is falsy — sidechain/subagent "user" lines are prompts
     injected by the parent agent, not the human.
  3. ``isMeta`` is falsy — meta lines are caveats/command wrappers.
  4. ``message.content`` is a string, or a list that contains NO
     ``tool_result`` block (tool results come back as user-role lines);
     list text is the concatenation of its ``text`` blocks.
  5. The text does not start with a system-injected tag. Measured injected
     prefixes: ``<task-notification>`` (background agent completion),
     ``<command-name>`` / ``<local-command-stdout>`` /
     ``<local-command-caveat>`` (slash-command wrappers),
     ``<system-reminder>``, and ``[Request interrupted`` (interrupt marker).
  6. Not a compact-continuation summary (``isCompactSummary`` or text
     starting with "This session is being continued").

Privacy (R-9, absolute): this parser NEVER retains prompt/response text or
file contents. User turns keep only identifiers/timestamp/length; tool calls
keep only tool name, ids, paths and size metadata (content-bearing input
fields such as ``content``/``new_string``/``old_string`` are reduced to
lengths).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

RUNTIME = "claude-code"

# Tools whose calls we observe. Only Read produces canonical reference
# events in M0 (E0-1 reference track); the rest are metadata-only
# observations (task instruction: "참고용").
REFERENCE_TOOLS = {"Read"}
METADATA_TOOLS = {"Glob", "Grep", "Edit", "Write", "MultiEdit", "NotebookEdit"}
CAPTURED_TOOLS = REFERENCE_TOOLS | METADATA_TOOLS

# Content-bearing input fields that must never be stored (R-9).
_CONTENT_FIELDS = ("content", "new_string", "old_string", "edits", "new_source")

_INJECTED_PREFIXES = (
    "<task-notification>",
    "<command-name>",
    "<local-command-stdout>",
    "<local-command-caveat>",
    "<system-reminder>",
    "[Request interrupted",
)
_CONTINUATION_PREFIX = "This session is being continued"


@dataclass
class UserTurn:
    """A real (human) top-level user prompt turn — task boundary candidate."""

    native_session_id: str
    uuid: str | None
    prompt_id: str | None
    timestamp: str
    cwd: str | None
    text_len: int
    source_file: str
    source_line: int


@dataclass
class ToolCall:
    """A captured tool_use block from an assistant line."""

    native_session_id: str
    agent_id: str | None  # None for main-session lines
    is_sidechain: bool
    tool_use_id: str | None
    tool_name: str
    file_path: str | None  # raw, as recorded by the runtime
    input_meta: dict  # reduced, content-stripped metadata
    timestamp: str
    cwd: str | None
    source_file: str
    source_line: int


@dataclass
class SessionInfo:
    native_session_id: str
    scope_cwd: str | None  # first observed cwd (session start dir)
    transcript_path: str
    runtime_version: str | None = None
    git_branch: str | None = None
    started_at: str | None = None
    ended_at: str | None = None


@dataclass
class ParseStats:
    files: int = 0
    lines: int = 0
    parse_errors: int = 0
    error_files: dict = field(default_factory=dict)  # file -> error count
    user_lines: int = 0
    real_user_turns: int = 0
    tool_calls_by_name: dict = field(default_factory=dict)


@dataclass
class ParseResult:
    sessions: list[SessionInfo] = field(default_factory=list)
    user_turns: list[UserTurn] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    stats: ParseStats = field(default_factory=ParseStats)


def _text_of_user_content(content) -> str | None:
    """Extract the text of a user message; None if it is a tool_result turn."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        blocks = [b for b in content if isinstance(b, dict)]
        if any(b.get("type") == "tool_result" for b in blocks):
            return None
        return " ".join(b.get("text", "") for b in blocks if b.get("type") == "text")
    return None


def is_real_user_turn(obj: dict) -> bool:
    """R-4 heuristic — see module docstring for the measured rules."""
    if obj.get("type") != "user":
        return False
    if obj.get("isSidechain") or obj.get("isMeta"):
        return False
    if obj.get("isCompactSummary"):
        return False
    text = _text_of_user_content((obj.get("message") or {}).get("content"))
    if text is None:
        return False
    stripped = text.lstrip()
    if not stripped:
        return False
    if any(stripped.startswith(p) for p in _INJECTED_PREFIXES):
        return False
    if stripped.startswith(_CONTINUATION_PREFIX):
        return False
    return True


def _reduce_tool_input(name: str, tool_input: dict) -> tuple[str | None, dict]:
    """Return (file_path, reduced_meta). Strips all content-bearing fields."""
    if not isinstance(tool_input, dict):
        return None, {}
    file_path = tool_input.get("file_path")
    if not isinstance(file_path, str):
        file_path = tool_input.get("notebook_path")
        if not isinstance(file_path, str):
            file_path = None
    meta: dict = {}
    for key in ("pattern", "path", "glob", "offset", "limit", "replace_all"):
        if key in tool_input and isinstance(tool_input[key], (str, int, bool)):
            meta[key] = tool_input[key]
    for key in _CONTENT_FIELDS:
        if key in tool_input:
            val = tool_input[key]
            try:
                meta[f"{key}_len"] = len(val)
            except TypeError:
                meta[f"{key}_len"] = -1
    return file_path, meta


def parse_transcript_file(path: str | Path, result: ParseResult) -> None:
    """Parse one transcript JSONL file, streaming line by line.

    Never loads the whole file (single files can be tens of MB or more).
    Appends findings into ``result``.
    """
    path = Path(path)
    stats = result.stats
    stats.files += 1
    session: SessionInfo | None = None
    first_cwd: str | None = None
    min_ts: str | None = None
    max_ts: str | None = None
    native_sid: str | None = None
    runtime_version: str | None = None
    git_branch: str | None = None

    with open(path, encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            stats.lines += 1
            try:
                obj = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                stats.parse_errors += 1
                stats.error_files[str(path)] = stats.error_files.get(str(path), 0) + 1
                continue
            if not isinstance(obj, dict):
                stats.parse_errors += 1
                stats.error_files[str(path)] = stats.error_files.get(str(path), 0) + 1
                continue

            sid = obj.get("sessionId")
            if native_sid is None and isinstance(sid, str):
                native_sid = sid
            cwd = obj.get("cwd")
            if first_cwd is None and isinstance(cwd, str):
                first_cwd = cwd
            if runtime_version is None and isinstance(obj.get("version"), str):
                runtime_version = obj["version"]
            if git_branch is None and isinstance(obj.get("gitBranch"), str):
                git_branch = obj["gitBranch"]
            ts = obj.get("timestamp")
            if isinstance(ts, str):
                if min_ts is None or ts < min_ts:
                    min_ts = ts
                if max_ts is None or ts > max_ts:
                    max_ts = ts

            t = obj.get("type")
            if t == "user":
                stats.user_lines += 1
                if is_real_user_turn(obj) and isinstance(sid, str) and isinstance(ts, str):
                    text = _text_of_user_content((obj.get("message") or {}).get("content")) or ""
                    result.user_turns.append(
                        UserTurn(
                            native_session_id=sid,
                            uuid=obj.get("uuid"),
                            prompt_id=obj.get("promptId"),
                            timestamp=ts,
                            cwd=cwd if isinstance(cwd, str) else None,
                            text_len=len(text),
                            source_file=str(path),
                            source_line=line_no,
                        )
                    )
                    stats.real_user_turns += 1
            elif t == "assistant":
                message = obj.get("message") or {}
                content = message.get("content")
                if not isinstance(content, list):
                    continue
                for block in content:
                    if not isinstance(block, dict) or block.get("type") != "tool_use":
                        continue
                    name = block.get("name")
                    if name not in CAPTURED_TOOLS:
                        continue
                    if not (isinstance(sid, str) and isinstance(ts, str)):
                        continue
                    file_path, meta = _reduce_tool_input(name, block.get("input") or {})
                    result.tool_calls.append(
                        ToolCall(
                            native_session_id=sid,
                            agent_id=obj.get("agentId"),
                            is_sidechain=bool(obj.get("isSidechain")),
                            tool_use_id=block.get("id"),
                            tool_name=name,
                            file_path=file_path,
                            input_meta=meta,
                            timestamp=ts,
                            cwd=cwd if isinstance(cwd, str) else None,
                            source_file=str(path),
                            source_line=line_no,
                        )
                    )
                    stats.tool_calls_by_name[name] = stats.tool_calls_by_name.get(name, 0) + 1

    if native_sid is not None:
        session = SessionInfo(
            native_session_id=native_sid,
            scope_cwd=first_cwd,
            transcript_path=str(path),
            runtime_version=runtime_version,
            git_branch=git_branch,
            started_at=min_ts,
            ended_at=max_ts,
        )
        result.sessions.append(session)


def iter_project_files(project_dir: str | Path) -> Iterator[tuple[Path, bool]]:
    """Yield (path, is_subagent) for all transcript files of one project dir.

    Main sessions: ``<project_dir>/<uuid>.jsonl``.
    Subagents:     ``<project_dir>/<uuid>/subagents/agent-*.jsonl``.
    """
    project_dir = Path(project_dir)
    for p in sorted(project_dir.glob("*.jsonl")):
        yield p, False
    for p in sorted(project_dir.glob("*/subagents/*.jsonl")):
        yield p, True


def parse_project_dir(project_dir: str | Path) -> ParseResult:
    """Parse all transcripts (main + subagent) of one encoded project dir.

    Subagent files carry the parent ``sessionId`` (measured), so their
    SessionInfo entries duplicate the main session's. One SessionInfo is kept
    per native_session_id, preferring the main-file entry; a subagent-only
    entry survives only when its parent main file is absent (orphan).
    """
    result = ParseResult()
    chosen: dict[str, SessionInfo] = {}
    from_main: set[str] = set()
    for path, is_subagent in iter_project_files(project_dir):
        before = len(result.sessions)
        parse_transcript_file(path, result)
        for s in result.sessions[before:]:
            sid = s.native_session_id
            if not is_subagent:
                chosen[sid] = s
                from_main.add(sid)
            elif sid not in from_main and sid not in chosen:
                chosen[sid] = s
    result.sessions = list(chosen.values())
    return result
