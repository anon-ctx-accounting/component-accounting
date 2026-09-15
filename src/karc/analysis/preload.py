"""E0-2 preload waste metrics (experiment plan §4 E0-2, claim C10).

Preload identification rule — MEASURED on the 3-machine dataset (2026-07-17,
70 sessions of E0-1 rerun-1; Claude Code v2.1.148..2.1.212):

The experiment plan identifies the preload set via the ``InstructionsLoaded``
hook's ``load_reason`` (S-3), which does not exist in batch transcripts. The
known candidate — a ``<system-reminder>`` block with "Contents of <path>"
headers injected into the session's first user message — was measured and is
ABSENT from this dataset (0/70 main sessions; system-prompt-side CLAUDE.md /
claudeMd context injection is not persisted to transcripts). No user-level or
project-level CLAUDE.md exists on disk where verifiable (local: all scope
roots + ~/.claude; remote: both machines' ~/.claude archives).

What IS observably auto-injected into the context, session-start or
mid-session, are ``attachment`` lines. Five attachment categories constitute
the transcript-observable preload set (fixed BEFORE judgment):

  category  attachment.type          item key          token source
  --------  -----------------------  ----------------  --------------------
  memory    nested_memory            file path         content.content
  skill     skill_listing            skill name        its listing line(s)
  agent     agent_listing_delta      agent type        its addedLines line(s)
  tool      deferred_tools_delta     tool name         its addedLines line
  mcp       mcp_instructions_delta   server name       its addedBlocks block

``memory`` is the only CLAUDE.md-계열 (spec-원형) category; the others are
runtime-injected instruction/listing content — the actual preload mass in
this environment. Non-instruction attachments (task_reminder, file =
user @-mention, edited_text_file, IDE state, ...) are EXCLUDED from the
preload set and reported in the census.

Downstream-usage rules (fixed BEFORE judgment; evidence must occur at
timestamp >= the item's first injection, anywhere in the session = main
transcript + its subagent transcripts):

  strict (usage lower bound -> unused UPPER bound):
    memory: Read tool_use whose normalized file_path == item path
    skill:  Skill tool_use (input.skill == name), slash-command invocation
            (<command-name>/name</command-name>), or skill base-dir load
            marker ("Base directory for this skill: ...")
    agent:  Agent/Task tool_use with input.subagent_type == type
    tool:   tool_use with that tool name
    mcp:    tool_use named mcp__<sanitized server>__*

  broad (usage upper bound -> unused LOWER bound = the gate metric):
    strict PLUS
    memory: any captured tool path arg == item path; item path substring
            re-occurring in any later user/assistant message (prompts,
            tool_use inputs incl. Bash commands, tool_results)
    skill:  any tool path arg containing /skills/<name>/
    tool:   ToolSearch input.query mentioning the tool name (word boundary)
    agent/mcp: same as strict (no structurally attributable weaker signal)

Token estimation: existing heuristic chain (experiment plan FR-R5) —
UTF-8 bytes/4, Korean (Hangul) bytes/2.5, per character class, rounded.

Judgment thresholds (experiment plan E0-2, pre-registered — DO NOT MODIFY):
  unused-ratio LOWER bound >= 30%  -> savings exist, keep C2's 30% target;
  10%..30%                         -> lower target to measured x 0.7
                                      (Amendments 기록 대상으로 보고);
  < 10%                            -> hypothesis review, hold Stage 2.
The gate metric is the POOLED TOKEN-WEIGHTED unused lower bound across all
measurable sessions (fixed before judgment); session-weighted means and
per-machine/per-category decompositions are reported alongside.

Privacy (R-9, absolute): this module never retains preload content. Items
carry only category, key (path/name), token counts, injection counts,
timestamps, and evidence counters.
"""

from __future__ import annotations

import glob
import json
import os
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

# Pre-registered thresholds (experiment plan E0-2) — DO NOT CHANGE.
UNUSED_LB_KEEP = 0.30
UNUSED_LB_REVIEW = 0.10
TARGET_ADJUST_FACTOR = 0.7

PRELOAD_ATTACHMENT_CATEGORY = {
    "nested_memory": "memory",
    "skill_listing": "skill",
    "agent_listing_delta": "agent",
    "deferred_tools_delta": "tool",
    "mcp_instructions_delta": "mcp",
}

AGENT_SPAWN_TOOLS = {"Agent", "Task"}
PATH_ARG_KEYS = ("file_path", "notebook_path", "path")

_COMMAND_NAME_RE = re.compile(r"<command-name>/?([^<\s]+)</command-name>")
_SKILL_BASE_DIR_RE = re.compile(r"Base directory for this skill:\s*(\S+)")


# ---------------------------------------------------------------------------
# Token estimation heuristic (bytes/4, Korean bytes/2.5)
# ---------------------------------------------------------------------------

def _is_hangul(ch: str) -> bool:
    cp = ord(ch)
    return (
        0xAC00 <= cp <= 0xD7A3      # Hangul syllables
        or 0x1100 <= cp <= 0x11FF   # Hangul Jamo
        or 0x3130 <= cp <= 0x318F   # Hangul compatibility Jamo
        or 0xA960 <= cp <= 0xA97F   # Hangul Jamo extended-A
        or 0xD7B0 <= cp <= 0xD7FF   # Hangul Jamo extended-B
    )


def estimate_tokens(text: str) -> int:
    """bytes/4 heuristic with Korean bytes/2.5 (experiment plan FR-R5)."""
    ko_bytes = 0
    other_bytes = 0
    for ch in text:
        n = len(ch.encode("utf-8"))
        if _is_hangul(ch):
            ko_bytes += n
        else:
            other_bytes += n
    return int(round(ko_bytes / 2.5 + other_bytes / 4.0))


def _line_tokens(line: str) -> int:
    # Rendered listing lines are newline-separated; count the separator.
    return estimate_tokens(line + "\n")


# ---------------------------------------------------------------------------
# Session-level extraction
# ---------------------------------------------------------------------------

@dataclass
class PreloadItem:
    category: str
    key: str
    tokens: int = 0
    injections: int = 0
    first_injected_at: str | None = None
    used_strict: bool = False
    used_broad: bool = False
    evidence: dict = field(default_factory=dict)  # kind -> count (R-9: counts only)

    def mark(self, kind: str, ts: str, *, strict: bool) -> None:
        if self.first_injected_at is None or ts < self.first_injected_at:
            return  # evidence must be downstream of the first injection
        self.evidence[kind] = self.evidence.get(kind, 0) + 1
        self.used_broad = True
        if strict:
            self.used_strict = True


@dataclass
class SessionPreload:
    items: dict = field(default_factory=dict)  # (category, key) -> PreloadItem
    envelope_tokens: dict = field(default_factory=dict)  # category -> tokens
    attachment_census: dict = field(default_factory=dict)  # attachment.type -> count
    lines: int = 0
    parse_errors: int = 0

    # -- item bookkeeping ---------------------------------------------------

    def _item(self, category: str, key: str) -> PreloadItem:
        k = (category, key)
        if k not in self.items:
            self.items[k] = PreloadItem(category=category, key=key)
        return self.items[k]

    def add_injection(self, category: str, key: str, tokens: int, ts: str | None) -> None:
        it = self._item(category, key)
        it.tokens += tokens
        it.injections += 1
        if ts is not None and (it.first_injected_at is None or ts < it.first_injected_at):
            it.first_injected_at = ts

    def add_envelope(self, category: str, tokens: int) -> None:
        if tokens:
            self.envelope_tokens[category] = self.envelope_tokens.get(category, 0) + tokens

    # -- metrics --------------------------------------------------------

    def metrics(self) -> dict:
        env = sum(self.envelope_tokens.values())
        item_tokens = sum(i.tokens for i in self.items.values())
        total = item_tokens + env
        # Envelope/unattributable listing text is counted in the denominator
        # and treated as USED in both bounds (conservative for the gate).
        used_strict = env + sum(i.tokens for i in self.items.values() if i.used_strict)
        used_broad = env + sum(i.tokens for i in self.items.values() if i.used_broad)
        return {
            "preload_tokens": total,
            "envelope_tokens": env,
            "used_strict_tokens": used_strict,
            "used_broad_tokens": used_broad,
            "unused_lb": (1.0 - used_broad / total) if total else None,
            "unused_ub": (1.0 - used_strict / total) if total else None,
            "n_items": len(self.items),
        }


def _norm_path(p: str, cwd: str | None) -> str:
    """Lexical normalization for path matching (no realpath — see E0-1 §6)."""
    p = os.path.expanduser(p)
    if not os.path.isabs(p) and cwd:
        p = os.path.join(cwd, p)
    return unicodedata.normalize("NFC", os.path.normpath(p))


def _mcp_tool_prefix(server_name: str) -> str:
    return "mcp__" + re.sub(r"[^A-Za-z0-9]", "_", server_name) + "__"


def _name_matches(invoked: str | None, key: str) -> bool:
    if not isinstance(invoked, str):
        return False
    return invoked == key or invoked.split(":")[-1] == key.split(":")[-1]


_ITEM_LINE_RE = re.compile(r"^- ([^\s:]+):")


def _split_listing(lines: list[str], names: list[str]) -> tuple[dict, int]:
    """Attribute listing lines to items: a line matching '- <name>:' opens
    that item; other lines continue the current item; leading unattributable
    lines count as envelope tokens.

    When ``names`` is provided, only listed names open items (guards against
    '- word: ...' bullets inside a description). Some attachments carry an
    empty ``names`` list (measured: skill_listing on v2.1.14x subagents) —
    then any '- <name>:' line opens an item.

    Returns ({name: tokens}, envelope_tokens).
    """
    known = {n for n in names if isinstance(n, str)}
    per_item: dict = {}
    envelope = 0
    current: str | None = None
    for line in lines:
        m = _ITEM_LINE_RE.match(line)
        if m and (not known or m.group(1) in known):
            current = m.group(1)
        if current is None:
            envelope += _line_tokens(line)
        else:
            per_item[current] = per_item.get(current, 0) + _line_tokens(line)
    return per_item, envelope


def _ingest_attachment(sp: SessionPreload, att: dict, ts: str | None) -> None:
    at = att.get("type")
    sp.attachment_census[at] = sp.attachment_census.get(at, 0) + 1
    category = PRELOAD_ATTACHMENT_CATEGORY.get(at)
    if category is None:
        return

    if at == "nested_memory":
        content = att.get("content")
        path = att.get("path")
        if isinstance(content, dict):
            path = path or content.get("path")
            text = content.get("content")
            if not isinstance(text, str):
                text = json.dumps(content, ensure_ascii=False)
        elif isinstance(content, str):
            text = content
        else:
            text = ""
        if isinstance(path, str) and path:
            sp.add_injection("memory", _norm_path(path, None), estimate_tokens(text), ts)
        else:
            sp.add_envelope("memory", estimate_tokens(text))

    elif at == "skill_listing":
        content = att.get("content")
        names = att.get("names") if isinstance(att.get("names"), list) else []
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False) if content else ""
        per_item, envelope = _split_listing(content.split("\n"), names)
        for name, tokens in per_item.items():
            sp.add_injection("skill", name, tokens, ts)
        sp.add_envelope("skill", envelope)

    elif at == "agent_listing_delta":
        lines = att.get("addedLines")
        types = att.get("addedTypes") if isinstance(att.get("addedTypes"), list) else []
        if not isinstance(lines, list):
            lines = str(lines).split("\n") if lines else []
        per_item, envelope = _split_listing([str(x) for x in lines], types)
        for name, tokens in per_item.items():
            sp.add_injection("agent", name, tokens, ts)
        sp.add_envelope("agent", envelope)

    elif at == "deferred_tools_delta":
        lines = att.get("addedLines")
        if not isinstance(lines, list):
            lines = str(lines).split("\n") if lines else []
        for line in lines:
            line = str(line)
            sp.add_injection("tool", line.strip(), _line_tokens(line), ts)

    elif at == "mcp_instructions_delta":
        blocks = att.get("addedBlocks")
        names = att.get("addedNames") if isinstance(att.get("addedNames"), list) else []
        if not isinstance(blocks, list):
            blocks = [blocks] if blocks else []
        for i, block in enumerate(blocks):
            text = block if isinstance(block, str) else json.dumps(block, ensure_ascii=False)
            key = names[i] if i < len(names) and isinstance(names[i], str) else None
            if key is None:
                m = re.match(r"##\s+(.+)", text)
                key = m.group(1).strip() if m else f"block-{i}"
            sp.add_injection("mcp", key, estimate_tokens(text), ts)


# ---------------------------------------------------------------------------
# Evidence pass
# ---------------------------------------------------------------------------

def _iter_jsonl(path: Path, sp: SessionPreload):
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sp.lines += 1
            try:
                obj = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                sp.parse_errors += 1
                continue
            if isinstance(obj, dict):
                yield obj


def _collect_items(files: list[Path], sp: SessionPreload) -> None:
    for path in files:
        for obj in _iter_jsonl(path, sp):
            if obj.get("type") != "attachment":
                continue
            att = obj.get("attachment")
            if isinstance(att, dict):
                ts = obj.get("timestamp")
                _ingest_attachment(sp, att, ts if isinstance(ts, str) else None)


def _tool_use_evidence(sp: SessionPreload, name: str, tool_input: dict, ts: str, cwd: str | None) -> None:
    # agent spawn
    if name in AGENT_SPAWN_TOOLS:
        st = tool_input.get("subagent_type")
        for (cat, key), it in sp.items.items():
            if cat == "agent" and isinstance(st, str) and st == key:
                it.mark("agent_spawn", ts, strict=True)
    # skill invocation
    if name == "Skill":
        sk = tool_input.get("skill")
        for (cat, key), it in sp.items.items():
            if cat == "skill" and _name_matches(sk, key):
                it.mark("skill_invocation", ts, strict=True)
    # deferred tool call (any tool_use name match) + ToolSearch mention
    for (cat, key), it in sp.items.items():
        if cat == "tool":
            if name == key:
                it.mark("tool_call", ts, strict=True)
            elif name == "ToolSearch":
                q = tool_input.get("query")
                if isinstance(q, str) and re.search(
                    rf"(?<![A-Za-z0-9_]){re.escape(key)}(?![A-Za-z0-9_])", q
                ):
                    it.mark("toolsearch_mention", ts, strict=False)
        elif cat == "mcp" and name.startswith(_mcp_tool_prefix(key)):
            it.mark("mcp_tool_call", ts, strict=True)
    # path args -> memory / skill paths
    paths = []
    for k in PATH_ARG_KEYS:
        v = tool_input.get(k)
        if isinstance(v, str) and v:
            paths.append(_norm_path(v, cwd))
    if paths:
        for (cat, key), it in sp.items.items():
            if cat == "memory":
                for p in paths:
                    if p == key:
                        it.mark("read_same_path" if name == "Read" else "tool_path_match",
                                ts, strict=(name == "Read"))
            elif cat == "skill":
                frag = "/skills/" + key.split(":")[-1] + "/"
                if any(frag in p for p in paths):
                    it.mark("skill_path_read", ts, strict=False)


def _text_evidence(sp: SessionPreload, obj: dict, ts: str) -> None:
    """Broad/strict evidence from user/assistant message text.

    The full message content (prompts, tool_use inputs incl. Bash commands,
    tool_results) is serialized once; only match booleans are retained (R-9).
    """
    message = obj.get("message")
    if not isinstance(message, dict):
        return
    blob = json.dumps(message.get("content"), ensure_ascii=False)
    if obj.get("type") == "user":
        for m in _COMMAND_NAME_RE.finditer(blob):
            cmd = m.group(1)
            for (cat, key), it in sp.items.items():
                if cat == "skill" and _name_matches(cmd, key):
                    it.mark("command_invocation", ts, strict=True)
        for m in _SKILL_BASE_DIR_RE.finditer(blob):
            base = os.path.basename(m.group(1).rstrip("/"))
            for (cat, key), it in sp.items.items():
                if cat == "skill" and _name_matches(base, key):
                    it.mark("skill_base_dir", ts, strict=True)
    for (cat, key), it in sp.items.items():
        if cat == "memory" and key in blob:
            it.mark("text_mention", ts, strict=False)


def _collect_evidence(files: list[Path], sp: SessionPreload) -> None:
    for path in files:
        for obj in _iter_jsonl(path, sp):
            t = obj.get("type")
            if t not in ("user", "assistant"):
                continue
            ts = obj.get("timestamp")
            if not isinstance(ts, str):
                continue
            cwd = obj.get("cwd") if isinstance(obj.get("cwd"), str) else None
            if t == "assistant":
                content = (obj.get("message") or {}).get("content")
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_use":
                            name = block.get("name")
                            tool_input = block.get("input")
                            if isinstance(name, str) and isinstance(tool_input, dict):
                                _tool_use_evidence(sp, name, tool_input, ts, cwd)
            _text_evidence(sp, obj, ts)


def session_files(transcript_path: str) -> list[Path]:
    """Main transcript + its subagent transcripts (parent sessionId)."""
    main = Path(transcript_path)
    subs = sorted(
        Path(p) for p in glob.glob(os.path.join(str(main)[: -len(".jsonl")], "subagents", "*.jsonl"))
    )
    return [main] + subs


def process_session(files: list[str | Path]) -> SessionPreload:
    """Two sub-passes: (A) preload items from attachments, (B) usage evidence.

    Pass B needs the full item set because evidence in the main transcript
    may refer to items injected in a subagent context and vice versa; the
    per-item first-injection timestamp guard keeps causality.
    """
    paths = [Path(f) for f in files]
    sp = SessionPreload()
    _collect_items(paths, sp)
    lines_a = sp.lines  # pass A and B traverse the same lines; report once
    _collect_evidence(paths, sp)
    sp.lines = lines_a
    return sp


# ---------------------------------------------------------------------------
# Judgment (pre-registered E0-2 criteria — DO NOT MODIFY)
# ---------------------------------------------------------------------------

def judge(unused_lb: float | None, unused_ub: float | None) -> dict:
    if unused_lb is None:
        return {"verdict": "no-data", "criteria": "pre-registered E0-2"}
    if unused_lb >= UNUSED_LB_KEEP:
        verdict, action = "savings-exist", "C2의 30% 절감 목표 유지"
        adjusted = None
    elif unused_lb >= UNUSED_LB_REVIEW:
        verdict = "target-adjust"
        adjusted = round(unused_lb * TARGET_ADJUST_FACTOR, 4)
        action = f"절감 목표를 실측 하한 x {TARGET_ADJUST_FACTOR} = {adjusted:.1%}로 하향 (Amendments 기록 대상)"
    else:
        verdict, action = "hypothesis-review", "C10 가설 재검토, Stage 2 착수 보류 (brief v0.2 협의)"
        adjusted = None
    return {
        "unused_lb": unused_lb,
        "unused_ub": unused_ub,
        "thresholds": {"keep": UNUSED_LB_KEEP, "review": UNUSED_LB_REVIEW},
        "verdict": verdict,
        "action": action,
        "adjusted_target": adjusted,
    }


# ---------------------------------------------------------------------------
# DB-driven analysis over the E0-1 session universe
# ---------------------------------------------------------------------------

def _machine_of(transcript_path: str) -> str:
    if "claude-task-data/host-a/" in transcript_path:
        return "host-a"
    if "claude-task-data/host-b/" in transcript_path:
        return "host-b"
    return "local"


def _pool(rows: list[dict]) -> dict:
    total = sum(r["preload_tokens"] for r in rows)
    strict = sum(r["used_strict_tokens"] for r in rows)
    broad = sum(r["used_broad_tokens"] for r in rows)
    lbs = [r["unused_lb"] for r in rows if r["unused_lb"] is not None]
    ubs = [r["unused_ub"] for r in rows if r["unused_ub"] is not None]
    return {
        "sessions": len(rows),
        "preload_tokens": total,
        "used_strict_tokens": strict,
        "used_broad_tokens": broad,
        "unused_lb_pooled": (1.0 - broad / total) if total else None,
        "unused_ub_pooled": (1.0 - strict / total) if total else None,
        "unused_lb_session_mean": (sum(lbs) / len(lbs)) if lbs else None,
        "unused_ub_session_mean": (sum(ubs) / len(ubs)) if ubs else None,
    }


def analyze_all(conn, active_cutoff: str | None = None) -> dict:
    """Run E0-2 over every session in the DB (the E0-1 rerun-1 universe).

    ``active_cutoff``: ISO8601 UTC — sessions whose transcript (or subagent)
    files were modified at/after this instant are excluded as right-censored
    (still active during/after ingest freeze; includes the session that runs
    this very experiment). Exclusions are fully reported.
    """
    import datetime

    cutoff_ts = None
    if active_cutoff:
        cutoff_ts = datetime.datetime.fromisoformat(
            active_cutoff.replace("Z", "+00:00")
        ).timestamp()

    sessions = conn.execute(
        """
        SELECT s.session_id, s.native_session_id, s.transcript_path,
               COALESCE(sc.root_path, '') AS scope_root
        FROM sessions s LEFT JOIN scopes sc ON sc.scope_id = s.scope_id
        ORDER BY s.session_id
        """
    ).fetchall()

    session_rows: list[dict] = []
    item_rows: list[dict] = []
    excluded: dict = {"missing-transcript": [], "active-right-censored": [], "no-observable-preload": []}
    census_total: dict = {}
    lines = 0
    parse_errors = 0

    for session_id, native_sid, tpath, scope_root in sessions:
        machine = _machine_of(tpath)
        files = session_files(tpath)
        files = [f for f in files if f.is_file()]
        if not files or not Path(tpath).is_file():
            excluded["missing-transcript"].append({"session_id": session_id, "machine": machine})
            continue
        if cutoff_ts is not None and any(os.path.getmtime(f) >= cutoff_ts for f in files):
            excluded["active-right-censored"].append(
                {"session_id": session_id, "machine": machine, "transcript_path": tpath}
            )
            continue
        sp = process_session(files)
        lines += sp.lines
        parse_errors += sp.parse_errors
        for at, n in sp.attachment_census.items():
            census_total[at] = census_total.get(at, 0) + n
        if not sp.items:
            excluded["no-observable-preload"].append(
                {"session_id": session_id, "machine": machine, "transcript_path": tpath,
                 "lines": sp.lines}
            )
            continue
        m = sp.metrics()
        m.update(
            session_id=session_id,
            native_session_id=native_sid,
            machine=machine,
            scope_root=scope_root,
            files=len(files),
        )
        session_rows.append(m)
        for (cat, key), it in sorted(sp.items.items()):
            item_rows.append(
                {
                    "session_id": session_id,
                    "machine": machine,
                    "category": cat,
                    "key": key,
                    "tokens": it.tokens,
                    "injections": it.injections,
                    "first_injected_at": it.first_injected_at,
                    "used_strict": it.used_strict,
                    "used_broad": it.used_broad,
                    "evidence": it.evidence,
                }
            )

    per_machine = {}
    for mname in ("local", "host-a", "host-b"):
        rows = [r for r in session_rows if r["machine"] == mname]
        if rows:
            per_machine[mname] = _pool(rows)

    per_category: dict = {}
    for cat in ("memory", "skill", "agent", "tool", "mcp"):
        rows = [r for r in item_rows if r["category"] == cat]
        if not rows:
            continue
        total = sum(r["tokens"] for r in rows)
        strict = sum(r["tokens"] for r in rows if r["used_strict"])
        broad = sum(r["tokens"] for r in rows if r["used_broad"])
        per_category[cat] = {
            "items": len(rows),
            "items_used_strict": sum(1 for r in rows if r["used_strict"]),
            "items_used_broad": sum(1 for r in rows if r["used_broad"]),
            "tokens": total,
            "unused_lb_pooled": (1.0 - broad / total) if total else None,
            "unused_ub_pooled": (1.0 - strict / total) if total else None,
        }

    pooled = _pool(session_rows)
    return {
        "experiment": "E0-2",
        "session_universe": len(sessions),
        "measurable_sessions": len(session_rows),
        "excluded": {k: v for k, v in excluded.items()},
        "excluded_counts": {k: len(v) for k, v in excluded.items()},
        "lines_parsed": lines,
        "parse_errors": parse_errors,
        "attachment_census": dict(sorted(census_total.items(), key=lambda kv: -kv[1])),
        "pooled": pooled,
        "per_machine": per_machine,
        "per_category": per_category,
        "judgment": judge(pooled["unused_lb_pooled"], pooled["unused_ub_pooled"]),
        "judgment_memory_only": judge(
            per_category.get("memory", {}).get("unused_lb_pooled"),
            per_category.get("memory", {}).get("unused_ub_pooled"),
        ) if "memory" in per_category else None,
        "parameters": {
            "active_cutoff": active_cutoff,
            "token_estimator": "bytes_div4_ko_div2.5",
            "gate_metric": "pooled token-weighted unused_lb over measurable sessions",
            "preload_categories": sorted(set(PRELOAD_ATTACHMENT_CATEGORY.values())),
            "thresholds": {"keep": UNUSED_LB_KEEP, "review": UNUSED_LB_REVIEW,
                           "adjust_factor": TARGET_ADJUST_FACTOR},
        },
        "_session_rows": session_rows,
        "_item_rows": item_rows,
    }
