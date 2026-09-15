"""Project artifact scan + preload-seed import (init step [2/5]).

Registers in-scope knowledge files (path + hash + metadata only — R-9, never
content) so ``list``/``pin``/pin-suggestion work from day 0 (product-ux §2.1
[2/5]), and imports the existing static preload set as a working-set seed at
F=0 (cache-policy §6.4: "preloading alone never counts").
"""

from __future__ import annotations

import os
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from karc.ingest import pipeline
from karc.util import new_ulid, utc_now_iso

_DOC_EXT = {".md", ".markdown", ".rst", ".txt", ".adoc"}
_SKIP_DIRS = {
    ".git", ".venv", "node_modules", "__pycache__", ".karc", ".pytest_cache",
    "claude-task-data", ".obsidian", ".hermes", "site-packages", "dist", "build",
}
_IMPORT_RE = re.compile(r"^@([^\s]+)", re.MULTILINE)


@dataclass
class ScanResult:
    root: str
    by_type: dict = field(default_factory=dict)
    total: int = 0
    preload_files: list = field(default_factory=list)
    registered: int = 0

    def as_dict(self) -> dict:
        return {
            "root": self.root, "total": self.total, "by_type": self.by_type,
            "preload_count": len(self.preload_files), "registered": self.registered,
        }


def _classify(path: str) -> str:
    low = path.lower()
    if "/.claude/rules/" in low or low.endswith(".rules.md"):
        return "rule"
    base = os.path.basename(low)
    if base in ("skill.md", "skills.md") or "/skills/" in low:
        return "skill"
    if base in ("claude.md", "agents.md", ".hermes.md", "memory.md"):
        return "instruction"
    ext = os.path.splitext(base)[1]
    if ext in _DOC_EXT:
        return "document"
    return "other"


def find_artifacts(root: str) -> list[tuple[str, str]]:
    """Return [(abs_path, artifact_type)] for in-scope knowledge files."""
    out: list[tuple[str, str]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for fn in filenames:
            if fn.startswith("._"):
                continue  # macOS AppleDouble sidecar, not a real artifact
            ext = os.path.splitext(fn)[1].lower()
            if ext not in _DOC_EXT:
                continue
            p = os.path.join(dirpath, fn)
            out.append((p, _classify(p)))
    return sorted(out)


def preload_seed(root: str) -> list[str]:
    """CLAUDE.md (or AGENTS.md) plus its @-imports — the static preload set."""
    seed: list[str] = []
    for base in ("CLAUDE.md", "AGENTS.md"):
        p = os.path.join(root, base)
        if not os.path.isfile(p):
            continue
        seed.append(p)
        try:
            text = Path(p).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for m in _IMPORT_RE.finditer(text):
            imp = m.group(1)
            ip = imp if os.path.isabs(imp) else os.path.join(root, imp)
            if os.path.isfile(ip):
                seed.append(os.path.realpath(ip))
    # de-dup preserving order
    seen, uniq = set(), []
    for p in seed:
        rp = os.path.realpath(p)
        if rp not in seen:
            seen.add(rp)
            uniq.append(rp)
    return uniq


def scan_counts(root: str) -> ScanResult:
    """Dry-run counts (no DB writes)."""
    res = ScanResult(root=os.path.realpath(root))
    for _p, atype in find_artifacts(root):
        res.by_type[atype] = res.by_type.get(atype, 0) + 1
        res.total += 1
    res.preload_files = preload_seed(root)
    return res


def scan_and_register(
    conn: sqlite3.Connection, db_path: str, scope_id: str, root: str
) -> ScanResult:
    """Register in-scope artifacts and import the preload seed (F=0 funnel
    loaded(preload) events)."""
    res = ScanResult(root=os.path.realpath(root))
    stats = pipeline.IngestStats()
    preload = set(preload_seed(root))
    for p, atype in find_artifacts(root):
        resolved, existed, inode = pipeline.normalize_path(p, root)
        scope_rows = pipeline._scope_table(conn)
        sid = pipeline._assign_scope(scope_rows, resolved)
        if sid is None:
            continue
        artifact_id = pipeline._resolve_artifact(
            conn, resolved, p, sid, utc_now_iso(), existed, inode, stats
        )
        res.by_type[atype] = res.by_type.get(atype, 0) + 1
        res.total += 1
        res.registered += 1
        if resolved in preload:
            _seed_preload_event(conn, sid, artifact_id)
            res.preload_files.append(resolved)
    return res


def _seed_preload_event(conn: sqlite3.Connection, scope_id: str, artifact_id: str) -> None:
    """A single loaded(preload) funnel event marking the artifact as part of
    the static preload set — never counted as a hit (§2.2)."""
    now = utc_now_iso()
    dedup = f"seed:{scope_id}:preload:{artifact_id}"
    conn.execute(
        """INSERT OR IGNORE INTO events
             (event_id, dedup_key, event_type, occurred_at, scope_id, runtime,
              artifact_id, load_reason, load_class, confidence, source_channel,
              schema_version)
           VALUES (?, ?, 'loaded', ?, ?, 'karc-init', ?, 'session_start', 'preload',
                   'inferred', 'manual', 1)""",
        (new_ulid(), dedup, now, scope_id, artifact_id),
    )
