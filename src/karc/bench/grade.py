"""FR-H3 deterministic grader, FR-H8 leakage/canary check, FR-H7 MCP adoption.

All three are pure functions over the driver output + transcript — no LLM
judgment (FR-H3: "LLM 판정 없음"). The grader mirrors the fixture's own
``nonce.answer_regex`` semantics used to build ground truth, so a run is
graded by exactly the rule the answer was constructed to satisfy.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field

NATIVE_READ_TOOLS = ("Read", "Grep", "Glob", "Bash")


@dataclass
class GradeResult:
    passed: bool
    detail: str = ""


def grade(expected_answer: dict, output_text: str, *, cwd: str | None = None,
          timeout_s: int = 30) -> GradeResult:
    """Grade ``output_text`` against a task's ``expected_answer`` spec
    (type ∈ {regex, exact, test-command}), temperature-0 / deterministic."""
    kind = expected_answer.get("type", "regex")
    value = expected_answer.get("value", "")
    text = output_text or ""
    if kind == "regex":
        return GradeResult(bool(re.search(value, text)), "regex")
    if kind == "exact":
        return GradeResult(value.strip() in text, "exact")
    if kind == "test-command":
        try:
            proc = subprocess.run(value, shell=True, cwd=cwd, capture_output=True,
                                  text=True, timeout=timeout_s)
            return GradeResult(proc.returncode == 0,
                               f"test-command rc={proc.returncode}")
        except (OSError, subprocess.SubprocessError) as exc:
            return GradeResult(False, f"test-command error: {exc}")
    return GradeResult(False, f"unknown expected_answer.type {kind!r}")


def classify_answer(task: dict, output_text: str) -> str:
    """E3 deterministic answer class: correct | stale | other-wrong.

    Correct takes precedence because it is the primary success contract.  A
    non-passing answer is stale only when it matches the task's frozen v1
    answer regex; near-miss drafts and all other text remain ``other-wrong``.
    """
    if grade(task["expected_answer"], output_text).passed:
        return "correct"
    stale = task.get("stale_answer")
    if stale and grade(stale, output_text).passed:
        return "stale"
    return "other-wrong"


# --------------------------------------------------------------------------
# FR-H8: leakage / canary regression
# --------------------------------------------------------------------------
def all_canaries(manifest: dict) -> set[str]:
    return {e["canary"] for e in manifest["artifacts"].values()}


def transcript_text(transcript: list[dict]) -> str:
    parts = []
    for ev in transcript or []:
        if ev.get("kind") == "text":
            parts.append(ev.get("text", ""))
    return "\n".join(parts)


def detect_leakage(output_text: str, transcript: list[dict],
                   canaries: set[str]) -> list[str]:
    """Return the sorted list of canary UUIDs that appear in the model's
    output or transcript. Any hit ⇒ leakage (closed-book: the model produced
    a fixture-only UUID it could only know from pre-training contamination)."""
    haystack = (output_text or "") + "\n" + transcript_text(transcript)
    found = {c for c in canaries if c in haystack}
    return sorted(found)


# --------------------------------------------------------------------------
# FR-H7: MCP adoption (E2-2)
# --------------------------------------------------------------------------
@dataclass
class Adoption:
    mcp_calls: int = 0
    native_managed_calls: int = 0
    native_other_calls: int = 0
    denied_native_attempts: int = 0
    per_tool: dict = field(default_factory=dict)

    @property
    def denominator(self) -> int:
        return self.mcp_calls + self.native_managed_calls

    @property
    def adoption_rate(self) -> float | None:
        d = self.denominator
        return (self.mcp_calls / d) if d else None

    def as_dict(self) -> dict:
        return {
            "mcp_calls": self.mcp_calls,
            "native_managed_calls": self.native_managed_calls,
            "native_other_calls": self.native_other_calls,
            "denied_native_attempts": self.denied_native_attempts,
            "adoption_rate": self.adoption_rate,
            "per_tool": self.per_tool,
        }


def _tool_targets_managed(inp: dict, managed_prefixes: tuple[str, ...]) -> bool:
    """Does a native Read/Grep/Glob tool call target the managed path?"""
    candidates = []
    for k in ("file_path", "path", "pattern", "glob", "command"):
        v = inp.get(k)
        if isinstance(v, str):
            candidates.append(v)
    return any(pref.rstrip("/") in c or c.startswith(pref) or ("/" + pref) in c
               for c in candidates for pref in managed_prefixes)


def mcp_adoption(transcript: list[dict], managed_prefixes: tuple[str, ...]) -> Adoption:
    """Count ``mcp__karc__*`` tool calls vs native Read/Grep/Glob calls that
    target the managed path (FR-H7). Adoption = mcp / (mcp + native-on-managed).
    Native reads of unmanaged files are reported but excluded from the ratio."""
    a = Adoption()
    for ev in transcript or []:
        if ev.get("kind") == "guard_denied":
            a.denied_native_attempts += 1
            continue
        if ev.get("kind") != "tool_use":
            continue
        name = ev.get("name", "")
        a.per_tool[name] = a.per_tool.get(name, 0) + 1
        if (ev.get("logical_server") == "karc"
                and ev.get("logical_tool") in ("search", "get")):
            a.mcp_calls += 1
        elif name.startswith("mcp__karc__"):
            a.mcp_calls += 1
        elif name in NATIVE_READ_TOOLS:
            if str(ev.get("status") or "").lower() in ("failed", "error"):
                continue
            if _tool_targets_managed(ev.get("input", {}) or {}, managed_prefixes):
                a.native_managed_calls += 1
            else:
                a.native_other_calls += 1
    return a
