"""E5-G2 semantic-chunk canary core.

The module keeps E5-G1's schedule, RAG supply, persistent-session accounting,
and statistics.  It changes only the K-ARC policy/MCP artifact from a whole
document version to a deterministic semantic chunk version.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import random
import re
import sqlite3
import statistics
import unicodedata
from pathlib import Path
from typing import Iterable, Sequence

from karc.bench import e5_cache_canary as g1
from karc.bench import materialize as mzt
from karc.bench.e4_replay import load_confirmed_config, replay_cell
from karc.db import connection
from karc.mcp.server import ensure_scope
from karc.util import utc_now_iso


SCHEMA = "e5-g2-chunk-canary-v1"
ARMS = g1.ARMS
CORPUS_SEED = g1.CORPUS_SEED
REUSE_FACTOR = g1.REUSE_FACTOR
SESSION_LENGTH = g1.SESSION_LENGTH
BUDGET_PCT = g1.BUDGET_PCT
MODEL = g1.MODEL
REASONING_EFFORT = g1.REASONING_EFFORT
MAX_TURN_RETRIES = g1.MAX_TURN_RETRIES
SMOKE_SESSIONS = 4
SMOKE_SCHEDULE_SEED = 4300
SMOKE_ATTEMPT_CAP = 80
CANARY_SCHEDULE_SEED = 4200  # exact E5-G1 canary schedule for direct delta
CANARY_SESSION_CHOICES = (8, 12)
COST_MARGIN_CANDIDATES = (0.05, 0.10)
ACCURACY_MARGIN_CANDIDATES = (0.05, 0.10)
SOFT_MAX_TOKENS = 64
SPLITTER_SEED = 5200
BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 5201
Z_ALPHA = g1.Z_ALPHA
Z_POWER = g1.Z_POWER


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode()


def hash_json(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text.encode("utf-8")) / 4.0))


def _slug(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode(
        "ascii", errors="ignore",
    ).decode().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", normalized).strip("-")
    return slug[:48] or "section"


def _paragraph_atoms(lines: Sequence[str]) -> list[str]:
    atoms: list[str] = []
    current: list[str] = []
    fenced = False
    for line in lines:
        if line.lstrip().startswith("```"):
            if current and not fenced:
                atoms.append("".join(current).strip("\n"))
                current = []
            current.append(line)
            fenced = not fenced
            if not fenced:
                atoms.append("".join(current).strip("\n"))
                current = []
            continue
        if not fenced and not line.strip():
            if current:
                atoms.append("".join(current).strip("\n"))
                current = []
            continue
        current.append(line)
    if current:
        atoms.append("".join(current).strip("\n"))
    return [atom for atom in atoms if atom]


def _pack_atoms(prefix: str, atoms: Sequence[str], soft_max_tokens: int) -> list[str]:
    values = list(atoms) or ([""] if prefix else [])
    packed: list[str] = []
    current = prefix.strip("\n")
    for atom in values:
        candidate = "\n\n".join(part for part in (current, atom) if part)
        if current and atom and _estimate_tokens(candidate) > soft_max_tokens:
            packed.append(current.rstrip() + "\n")
            current = atom
        else:
            current = candidate
    if current:
        packed.append(current.rstrip() + "\n")
    return packed


def _markdown_units(text: str, soft_max_tokens: int) -> list[dict]:
    lines = text.splitlines(keepends=True)
    sections: list[tuple[tuple[str, ...], str, list[str]]] = []
    heading_stack: list[str] = []
    heading_line = ""
    body: list[str] = []
    fenced = False

    def flush() -> None:
        nonlocal heading_line, body
        if heading_line or any(line.strip() for line in body):
            sections.append((tuple(heading_stack), heading_line, list(body)))
        heading_line = ""
        body = []

    for line in lines:
        if line.lstrip().startswith("```"):
            fenced = not fenced
        match = None if fenced else re.match(r"^(#{1,6})\s+(.+?)\s*$", line.rstrip("\n"))
        if match:
            flush()
            level = len(match.group(1))
            heading_stack[:] = heading_stack[: level - 1]
            heading_stack.append(_slug(match.group(2)))
            heading_line = line.rstrip("\n")
        else:
            body.append(line)
    flush()
    if not sections:
        sections = [((), "", lines)]

    units: list[dict] = []
    for path, heading, section_lines in sections:
        chunks = _pack_atoms(heading, _paragraph_atoms(section_lines), soft_max_tokens)
        for text_chunk in chunks:
            units.append({
                "kind": "markdown-section",
                "heading_path": list(path),
                "text": text_chunk,
            })
    return units


def _python_units(text: str, soft_max_tokens: int) -> list[dict]:
    lines = text.splitlines(keepends=True)
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return _prose_units(text, soft_max_tokens)
    nodes = [node for node in tree.body if isinstance(
        node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
    )]
    if not nodes:
        return _prose_units(text, soft_max_tokens)
    atoms: list[tuple[str, str]] = []
    cursor = 0
    for node in nodes:
        start = max(0, int(node.lineno) - 1)
        decorators = [int(value.lineno) - 1 for value in getattr(node, "decorator_list", [])]
        start = min([start, *decorators])
        if start > cursor and any(line.strip() for line in lines[cursor:start]):
            atoms.append(("module", "".join(lines[cursor:start])))
        end = int(getattr(node, "end_lineno", node.lineno))
        atoms.append((f"{node.__class__.__name__.lower()}:{node.name}", "".join(lines[start:end])))
        cursor = end
    if cursor < len(lines) and any(line.strip() for line in lines[cursor:]):
        atoms.append(("module-tail", "".join(lines[cursor:])))
    return [
        {"kind": "python-definition", "heading_path": [name],
         "text": body.rstrip() + "\n"}
        for name, body in atoms if body.strip()
    ]


def _prose_units(text: str, soft_max_tokens: int) -> list[dict]:
    return [
        {"kind": "prose-paragraph", "heading_path": [], "text": value}
        for value in _pack_atoms("", _paragraph_atoms(text.splitlines(keepends=True)),
                                 soft_max_tokens)
    ]


def semantic_split(path: str, text: str, *, soft_max_tokens: int = SOFT_MAX_TOKENS,
                   seed: int = SPLITTER_SEED) -> list[dict]:
    """Split only at semantic boundaries; ``seed`` is frozen provenance.

    No random operation or fixed-byte/page cut is used.  An indivisible
    paragraph/definition may exceed the soft maximum.
    """
    if soft_max_tokens <= 0:
        raise ValueError("soft_max_tokens must be positive")
    suffix = Path(path).suffix.lower()
    if suffix in {".md", ".markdown", ".mdx"}:
        raw = _markdown_units(text, soft_max_tokens)
    elif suffix == ".py":
        raw = _python_units(text, soft_max_tokens)
    else:
        raw = _prose_units(text, soft_max_tokens)
    units = []
    for index, value in enumerate(raw):
        heading = "/".join(value["heading_path"]) or "root"
        boundary_key = f"{heading}::b{index:03d}"
        body = value["text"]
        units.append({
            **value,
            "boundary_index": index,
            "boundary_key": boundary_key,
            "bytes": len(body.encode("utf-8")),
            "estimated_tokens": _estimate_tokens(body),
            "sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
            "splitter_seed": seed,
        })
    if (not units or re.findall(r"\S+", "\n".join(unit["text"] for unit in units))
            != re.findall(r"\S+", text)):
        raise AssertionError(f"semantic splitter lost or reordered content: {path}")
    return units


def _allocate_tokens(total: int, byte_sizes: Sequence[int]) -> list[int]:
    if not byte_sizes or total < len(byte_sizes):
        raise ValueError("source token count cannot cover semantic chunks")
    remaining = total - len(byte_sizes)
    weight = sum(byte_sizes) or len(byte_sizes)
    exact = [remaining * size / weight for size in byte_sizes]
    extra = [math.floor(value) for value in exact]
    left = remaining - sum(extra)
    order = sorted(range(len(byte_sizes)), key=lambda i: (-(exact[i] - extra[i]), i))
    for index in order[:left]:
        extra[index] += 1
    result = [1 + value for value in extra]
    if sum(result) != total:
        raise AssertionError("chunk token allocation drift")
    return result


def build_chunk_manifest(source_manifest: dict, fixture_root: str | Path) -> tuple[dict, dict[str, str]]:
    fixture_root = Path(fixture_root)
    artifacts: dict[str, dict] = {}
    texts: dict[str, str] = {}
    by_source: dict[str, list[str]] = {}
    boundary_lookup: dict[tuple[str, str], str] = {}
    for source_version, entry in sorted(source_manifest["artifacts"].items()):
        body = (fixture_root / "repo" / entry["path"]).read_text(encoding="utf-8")
        units = semantic_split(entry["path"], body)
        allocations = _allocate_tokens(
            int(entry["size_tok"]), [int(unit["bytes"]) for unit in units],
        )
        chunk_versions = []
        for unit, size_tok in zip(units, allocations):
            boundary = unit["boundary_key"]
            chunk_id = f"{entry['artifact_id']}::{boundary}"
            chunk_version = f"{source_version}::{boundary}"
            suffix = Path(entry["path"]).suffix or ".txt"
            rel = (
                f"chunks/{entry['artifact_id']}/"
                f"{hashlib.sha256(chunk_version.encode()).hexdigest()[:16]}{suffix}"
            )
            artifacts[chunk_version] = {
                "artifact_id": chunk_id,
                "version_id": chunk_version,
                "doc_id": entry["artifact_id"],
                "doc_version_id": source_version,
                "source_path": entry["path"],
                "path": rel,
                "boundary_index": unit["boundary_index"],
                "boundary_key": boundary,
                "heading_path": unit["heading_path"],
                "kind": unit["kind"],
                "sha256": unit["sha256"],
                "bytes": unit["bytes"],
                "size_tok": size_tok,
                "critical": bool(entry.get("critical", False)),
                "pinned": bool(entry.get("pinned", False)),
                "valid_from_seq": entry.get("valid_from_seq"),
                "superseded_at_seq": entry.get("superseded_at_seq"),
                "is_current_at_end": entry.get("is_current_at_end"),
                "superseded_by_version_id": None,
                "supersedes_version_id": None,
            }
            texts[chunk_version] = unit["text"]
            chunk_versions.append(chunk_version)
            boundary_lookup[(source_version, boundary)] = chunk_version
        by_source[source_version] = chunk_versions

    for source_version, entry in source_manifest["artifacts"].items():
        newer = entry.get("superseded_by_version_id")
        older = entry.get("supersedes_version_id")
        for chunk_version in by_source[source_version]:
            boundary = artifacts[chunk_version]["boundary_key"]
            if newer:
                artifacts[chunk_version]["superseded_by_version_id"] = boundary_lookup.get(
                    (newer, boundary),
                )
            if older:
                artifacts[chunk_version]["supersedes_version_id"] = boundary_lookup.get(
                    (older, boundary),
                )

    groups = {}
    for name, members in source_manifest.get("groups", {}).items():
        expanded = []
        for member in members:
            expanded.extend(by_source.get(member, []))
        groups[name] = expanded
    total_source = sum(int(row["size_tok"]) for row in source_manifest["artifacts"].values())
    total_chunk = sum(int(row["size_tok"]) for row in artifacts.values())
    if total_source != total_chunk:
        raise AssertionError("chunk corpus changed registered token total")
    manifest = {
        **{key: value for key, value in source_manifest.items()
           if key not in {"artifacts", "groups", "manifest_sha256", "structure"}},
        "experiment": "E5-G2",
        "schema_version": 1,
        "artifacts": artifacts,
        "groups": groups,
        "source_to_chunks": by_source,
        "cell": dict(source_manifest["cell"]),
        "structure": {
            **source_manifest["structure"],
            "source_document_versions": len(source_manifest["artifacts"]),
            "chunk_versions": len(artifacts),
            "stable_chunk_identities": len({row["artifact_id"] for row in artifacts.values()}),
            "soft_max_tokens": SOFT_MAX_TOKENS,
            "splitter_seed": SPLITTER_SEED,
            "split_kinds": sorted({row["kind"] for row in artifacts.values()}),
            "critical_source_versions": sum(
                bool(row.get("critical")) for row in source_manifest["artifacts"].values()
            ),
            "critical_chunk_versions": sum(bool(row["critical"]) for row in artifacts.values()),
            "token_total_preserved": total_source,
        },
    }
    content_free = {
        "schema": SCHEMA,
        "splitter": {
            "strategy": "semantic-boundary",
            "markdown": "heading-path section with paragraph/fenced-code sub-boundaries",
            "python": "AST top-level function/class boundaries",
            "prose": "blank-line paragraph boundaries",
            "soft_max_tokens": SOFT_MAX_TOKENS,
            "oversize_rule": "retain indivisible semantic unit; never fixed-page split",
            "seed": SPLITTER_SEED,
        },
        "source_manifest_sha256": source_manifest["manifest_sha256"],
        "artifacts": {
            key: {name: value for name, value in row.items() if name != "text"}
            for key, row in sorted(artifacts.items())
        },
        "groups": groups,
        "structure": manifest["structure"],
    }
    content_free["sha256"] = hash_json(content_free)
    manifest["chunk_manifest_sha256"] = content_free["sha256"]
    manifest["manifest_sha256"] = hash_json({
        **manifest, "chunk_text_sha256": hash_json(texts),
    })
    manifest["content_free_chunk_manifest"] = content_free
    return manifest, texts


def _gold_chunks(task: dict, source_version: str, chunk_manifest: dict,
                 chunk_texts: dict[str, str]) -> list[str]:
    fact = task["answer_fact"]
    candidates = chunk_manifest["source_to_chunks"][source_version]
    exact = [value for value in candidates
             if fact["key"] in chunk_texts[value] and fact["value"] in chunk_texts[value]]
    if len(exact) != 1:
        raise AssertionError((task["task_id"], source_version, exact))
    return exact


def map_tasks_to_chunks(tasks: Sequence[dict], chunk_manifest: dict,
                        chunk_texts: dict[str, str]) -> list[dict]:
    mapped = []
    for task in tasks:
        required = [chunk for source in task["required_versions"]
                    for chunk in _gold_chunks(task, source, chunk_manifest, chunk_texts)]
        forbidden = [chunk for source in task["forbidden_versions"]
                     for chunk in chunk_manifest["source_to_chunks"][source]]
        pre_events = []
        for event in task["pre_events"]:
            for chunk in chunk_manifest["source_to_chunks"][event["version_id"]]:
                pre_events.append({**event, "version_id": chunk})
        mapped.append({
            **task,
            "required_versions": required,
            "required_artifacts": [chunk_manifest["artifacts"][v]["artifact_id"]
                                   for v in required],
            "forbidden_versions": forbidden,
            "pre_events": pre_events,
            "source_required_versions": list(task["required_versions"]),
            "source_forbidden_versions": list(task["forbidden_versions"]),
        })
    return mapped


def prepare_bundle(repo_root: str | Path, fixture_root: str | Path, *,
                   schedule_seed: int, session_count: int) -> dict:
    base = g1.prepare_bundle(
        repo_root, fixture_root, schedule_seed=schedule_seed,
        session_count=session_count,
    )
    chunk_manifest, chunk_texts = build_chunk_manifest(base["manifest"], fixture_root)
    chunk_tasks = map_tasks_to_chunks(base["tasks"], chunk_manifest, chunk_texts)
    replay_summary, replay_rows = replay_cell(
        rho=g1.ENGINE_RHO_LABEL, sigma=g1.ENGINE_SIGMA_LABEL,
        budget_pct=BUDGET_PCT, confirmed_config=load_confirmed_config(repo_root),
        fixture_manifest=chunk_manifest, fixture_tasks=chunk_tasks,
    )
    replay_by_task = {row["task_id"]: row for row in replay_rows}
    source_task = {task["task_id"]: task for task in base["tasks"]}
    chunk_task = {task["task_id"]: task for task in chunk_tasks}
    sessions = []
    for session in base["sessions"]:
        rows = []
        first_id = session["tasks"][0]["task"]["task_id"]
        initial = list(replay_by_task[first_id]["arms"]["karc"]["working_set_versions"])
        for base_item in session["tasks"]:
            task_id = base_item["task"]["task_id"]
            policy = list(replay_by_task[task_id]["arms"]["karc"]["working_set_versions"])
            required = set(chunk_task[task_id]["required_versions"])
            rows.append({
                "task": source_task[task_id],
                "chunk_task": chunk_task[task_id],
                "policy_resident_versions": policy,
                "policy_resident_hit": required <= set(policy),
                "initial_prefix_hit": required <= set(initial),
                "rag_plan": base_item["rag_plan"],
            })
        sessions.append({
            "schedule_seed": session["schedule_seed"],
            "session_id": session["session_id"],
            "initial_resident_versions": initial,
            "tasks": rows,
        })
    return {
        **base,
        "schema": SCHEMA,
        "doc_manifest": base["manifest"],
        "manifest": chunk_manifest,
        "chunk_texts": chunk_texts,
        "chunk_tasks": chunk_tasks,
        "sessions": sessions,
        "chunk_replay_summary": replay_summary,
        "chunk_replay_rows": replay_rows,
        "chunk_policy_audit": {
            "cold_auto_critical": replay_summary["arms"]["karc"]["cold_auto_critical"],
            "critical_resident_evictions": replay_summary["arms"]["karc"]["critical_resident_evictions"],
            "chunk_transition_count": len(replay_summary["karc_transition_audit"]),
        },
    }


def bundle_snapshot(bundle: dict) -> dict:
    sessions = []
    manifest = bundle["manifest"]
    for session in bundle["sessions"]:
        sessions.append({
            "schedule_seed": session["schedule_seed"],
            "session_id": session["session_id"],
            "initial_resident_chunk_versions": session["initial_resident_versions"],
            "initial_resident_tokens": sum(
                int(manifest["artifacts"][v]["size_tok"])
                for v in session["initial_resident_versions"]
            ),
            "tasks": [{
                "task_id": item["task"]["task_id"],
                "position": item["task"]["session_task"],
                "source_required_versions": item["task"]["required_versions"],
                "required_chunk_versions": item["chunk_task"]["required_versions"],
                "reuse": bool(item["task"]["e5_resident_reuse"]),
                "policy_resident_hit": item["policy_resident_hit"],
                "initial_prefix_hit": item["initial_prefix_hit"],
                "rag_artifact_ids": item["rag_plan"]["artifact_ids"],
                "rag_tokens": item["rag_plan"]["tokens"],
            } for item in session["tasks"]],
        })
    value = {
        "schema": SCHEMA,
        "schedule": {
            key: bundle["schedule"][key] for key in (
                "schedule_seed", "sessions", "session_length",
                "reuse_factor_requested", "reuse_factor_measured", "reuse_count",
                "reuse_denominator_eligible_followups", "corpus_content_hash",
                "source_e4_manifest_sha256", "tasks_sha256", "sha256",
            )
        },
        "cell": bundle["manifest"]["cell"],
        "chunk_manifest_sha256": bundle["manifest"]["chunk_manifest_sha256"],
        "chunk_manifest": bundle["manifest"]["content_free_chunk_manifest"],
        "retrieval_audit": bundle["retrieval_audit"],
        "chunk_policy_audit": bundle["chunk_policy_audit"],
        "sessions": sessions,
    }
    value["sha256"] = hash_json(value)
    return value


def _write_agents(workdir: Path, manifest: dict, texts: dict[str, str],
                  initial: Sequence[str]) -> tuple[str, int]:
    parts = ["# K-ARC chunk benchmark context", "", g1.KARC_SESSION_INSTRUCTION,
             "", "## Injected semantic chunks", ""]
    for version in sorted(initial, key=lambda v: (
        manifest["artifacts"][v]["source_path"],
        manifest["artifacts"][v]["boundary_index"],
    )):
        entry = manifest["artifacts"][version]
        parts += [
            f"<!-- chunk:start id={entry['artifact_id']} version={version} "
            f"source={entry['source_path']} boundary={entry['boundary_key']} -->",
            texts[version].rstrip("\n"),
            "<!-- chunk:end -->", "",
        ]
    value = "\n".join(parts).rstrip() + "\n"
    raw = value.encode("utf-8")
    (workdir / "AGENTS.md").write_text(value, encoding="utf-8")
    return hashlib.sha256(raw).hexdigest(), len(raw)


def _active_source_versions(doc_manifest: dict, seq: int) -> list[str]:
    values = []
    for version, entry in doc_manifest["artifacts"].items():
        start = entry.get("valid_from_seq")
        stop = entry.get("superseded_at_seq")
        if start is not None and int(start) <= seq and (stop is None or int(stop) > seq):
            values.append(version)
    return sorted(values)


def _chunk_path(workdir: Path, entry: dict) -> Path:
    return workdir / mzt.MANAGED_ROOT / entry["path"]


def _insert_live_chunk(conn: sqlite3.Connection, scope_id: str, workdir: Path,
                       entry: dict, *, observed_at: str) -> None:
    artifact_id = entry["artifact_id"]
    version_id = entry["version_id"]
    path = _chunk_path(workdir, entry)
    raw = path.read_bytes()
    canonical = unicodedata.normalize("NFC", str(path.resolve()))
    row = conn.execute(
        "SELECT current_version_id FROM artifacts WHERE artifact_id = ?", (artifact_id,),
    ).fetchone()
    if row and row[0] == version_id:
        return
    if row and row[0]:
        conn.execute(
            "UPDATE versions SET invalidated_at = ? WHERE version_id = ? AND invalidated_at IS NULL",
            (observed_at, row[0]),
        )
    if not row:
        conn.execute(
            "INSERT INTO artifacts (artifact_id, scope_id, artifact_type, logical_name, "
            "canonical_path, criticality, pinned, lifecycle_state, heading_anchors) "
            "VALUES (?, ?, 'document', ?, ?, ?, ?, 'active', ?)",
            (artifact_id, scope_id,
             f"{entry['doc_id']} {entry['boundary_key']}", canonical,
             "critical" if entry["critical"] else "normal", int(entry["pinned"]),
             json.dumps(entry["heading_path"], ensure_ascii=False)),
        )
    existing = conn.execute(
        "SELECT 1 FROM versions WHERE version_id = ?", (version_id,),
    ).fetchone()
    if not existing:
        conn.execute(
            "INSERT INTO versions (version_id, artifact_id, content_hash, size_bytes, "
            "size_tokens, token_estimator, observed_at, source_channel) "
            "VALUES (?, ?, ?, ?, ?, 'measured', ?, 'fixture-chunk')",
            (version_id, artifact_id, hashlib.sha256(raw).hexdigest(), len(raw),
             int(entry["size_tok"]), observed_at),
        )
    if row and row[0] and row[0] != version_id:
        conn.execute(
            "UPDATE versions SET superseded_by_version_id = ?, supersede_reason = 'new_version' "
            "WHERE version_id = ?", (version_id, row[0]),
        )
    conn.execute(
        "UPDATE artifacts SET current_version_id = ?, canonical_path = ?, "
        "lifecycle_state = 'active', updated_at = ? WHERE artifact_id = ?",
        (version_id, canonical, observed_at, artifact_id),
    )


def build_chunk_index(workdir: Path, manifest: dict, active_source_versions: Sequence[str],
                      db_path: Path) -> str:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = connection.connect(str(db_path))
    connection.migrate(conn, db_path=str(db_path))
    conn.execute("BEGIN IMMEDIATE")
    try:
        scope_id = ensure_scope(conn, str(workdir))
        now = utc_now_iso()
        for source in active_source_versions:
            for version in manifest["source_to_chunks"][source]:
                _insert_live_chunk(conn, scope_id, workdir, manifest["artifacts"][version],
                                   observed_at=now)
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    return scope_id


def activate_document_version(db_path: str | Path, workdir: str | Path, manifest: dict,
                              source_version: str) -> int:
    workdir = Path(workdir)
    conn = connection.connect(str(db_path))
    conn.execute("BEGIN IMMEDIATE")
    changed = 0
    try:
        scope_id = ensure_scope(conn, str(workdir))
        now = utc_now_iso()
        new_versions = manifest["source_to_chunks"][source_version]
        new_ids = {manifest["artifacts"][value]["artifact_id"] for value in new_versions}
        doc_id = manifest["artifacts"][new_versions[0]]["doc_id"]
        rows = conn.execute(
            "SELECT artifact_id, current_version_id FROM artifacts WHERE scope_id = ?",
            (scope_id,),
        ).fetchall()
        for artifact_id, current in rows:
            if artifact_id.startswith(doc_id + "::") and artifact_id not in new_ids:
                if current:
                    conn.execute(
                        "UPDATE versions SET invalidated_at = ? WHERE version_id = ? "
                        "AND invalidated_at IS NULL", (now, current),
                    )
                conn.execute(
                    "UPDATE artifacts SET current_version_id = NULL, lifecycle_state = "
                    "'invalidated', updated_at = ? WHERE artifact_id = ?",
                    (now, artifact_id),
                )
                changed += 1
        for version in new_versions:
            entry = manifest["artifacts"][version]
            before = conn.execute(
                "SELECT current_version_id FROM artifacts WHERE artifact_id = ?",
                (entry["artifact_id"],),
            ).fetchone()
            _insert_live_chunk(conn, scope_id, workdir, entry, observed_at=now)
            changed += int(not before or before[0] != version)
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    return changed


def materialize_session(arm: str, workdir: str | Path, *, fixture_root: str | Path,
                        bundle: dict, initial_resident_versions: Sequence[str],
                        initial_seq: int) -> dict:
    workdir = Path(workdir)
    if arm == "rag-bm25":
        return g1.materialize_session(
            arm, workdir, fixture_root=fixture_root,
            manifest=bundle["doc_manifest"], initial_resident_versions=[],
        )
    if arm != "karc-full":
        raise ValueError(arm)
    workdir.mkdir(parents=True, exist_ok=True)
    g1._init_git(workdir)
    manifest = bundle["manifest"]
    texts = bundle["chunk_texts"]
    sha, nbytes = _write_agents(
        workdir, manifest, texts, initial_resident_versions,
    )
    for version, entry in manifest["artifacts"].items():
        target = _chunk_path(workdir, entry)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(texts[version], encoding="utf-8")
    index_db = workdir / ".karc" / "index.db"
    active = _active_source_versions(bundle["doc_manifest"], initial_seq)
    build_chunk_index(workdir, manifest, active, index_db)
    overrides = [*mzt._codex_mcp_overrides(workdir, index_db),
                 f"project_doc_max_bytes={nbytes + 4096}"]
    return {
        "workdir": workdir,
        "hook_mode": "deny-managed",
        "config_overrides": tuple(overrides),
        "agents_sha256": sha,
        "agents_bytes": nbytes,
        "managed_versions": sorted(manifest["artifacts"]),
        "initial_resident_versions": list(initial_resident_versions),
        "chunk_index_db": str(index_db),
        "active_source_versions": active,
    }


def apply_task_version_updates(materialized: dict, item: dict, bundle: dict) -> int:
    if not materialized.get("chunk_index_db"):
        return 0
    changed = 0
    doc_manifest = bundle["doc_manifest"]
    for event in item["task"]["pre_events"]:
        if event["event_type"] != "validity_changed":
            continue
        newer = doc_manifest["artifacts"][event["version_id"]].get(
            "superseded_by_version_id",
        )
        if newer:
            changed += activate_document_version(
                materialized["chunk_index_db"], materialized["workdir"],
                bundle["manifest"], newer,
            )
    return changed


def task_prompt(arm: str, item: dict, fixture_root: str | Path,
                doc_manifest: dict) -> str:
    return g1.task_prompt(arm, item, fixture_root, doc_manifest)


def grade_turn(task: dict, output_text: str) -> dict:
    return g1.grade_turn(task, output_text)


def error_digest(detail: str | None) -> dict:
    return g1.error_digest(detail)


def tool_counts(transcript: Iterable[dict], guard_events: Iterable[dict]) -> dict:
    return g1.tool_counts(transcript, guard_events)


def summarize(rows: Sequence[dict], expected_sessions: int) -> dict:
    summary = g1.summarize(rows, expected_sessions)
    valid = g1._latest_complete(rows)
    by_key: dict[tuple[int, str, str], list[dict]] = {}
    for row in valid:
        by_key.setdefault((int(row["schedule_seed"]), row["session_id"], row["arm"]), []).append(row)
    differences = []
    for seed, sid in sorted({(key[0], key[1]) for key in by_key}):
        karc = by_key.get((seed, sid, "karc-full"))
        rag = by_key.get((seed, sid, "rag-bm25"))
        if karc and rag:
            differences.append(
                statistics.mean(float(row["passed"]) for row in karc)
                - statistics.mean(float(row["passed"]) for row in rag)
            )
    for pair, difference in zip(summary["paired_session_metrics"], differences):
        pair["accuracy_difference"] = difference
    summary.update({
        "schema": SCHEMA,
        "runtime_isolation": {
            "isolated_codex_home_per_process": True,
            "multi_agent_disabled": True,
            "rag_mcp_search_calls": sum(
                int(row["mcp_search_calls"]) for row in valid
                if row["arm"] == "rag-bm25"
            ),
            "rag_mcp_get_calls": sum(
                int(row["mcp_get_calls"]) for row in valid
                if row["arm"] == "rag-bm25"
            ),
        },
        "accuracy_difference": {
            "point": statistics.mean(differences) if differences else None,
            "ci_lower_one_sided_95": g1.bootstrap_lower(
                differences, reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED + 2,
            ) if differences else None,
            "sample_sd": statistics.stdev(differences) if len(differences) > 1 else 0,
        },
    })
    return summary


def _required_sessions(sd: float, distance: float) -> int:
    distance = max(abs(distance), 0.025)
    return math.ceil(((Z_ALPHA + Z_POWER) * sd / distance) ** 2)


def choose_canary_freeze(smoke: dict, *, code_git_hash: str,
                         source_sha256: dict[str, str]) -> dict:
    if not smoke.get("complete"):
        raise ValueError("cannot freeze canary from incomplete smoke")
    reuse_point = float(smoke["reuse_fresh_saving"]["point"])
    cost_margin = COST_MARGIN_CANDIDATES[0]
    for candidate in COST_MARGIN_CANDIDATES:
        if reuse_point >= 2 * candidate:
            cost_margin = candidate
    n_reuse = _required_sessions(
        float(smoke["reuse_fresh_saving"]["sample_sd"]), reuse_point - cost_margin,
    )
    amortized_point = float(smoke["amortized_fresh_saving"]["point"])
    n_amortized = _required_sessions(
        float(smoke["amortized_fresh_saving"]["sample_sd"]), amortized_point,
    )
    accuracy_sd = float(smoke["accuracy_difference"]["sample_sd"])
    accuracy_requirements = {
        str(margin): _required_sessions(accuracy_sd, margin)
        for margin in ACCURACY_MARGIN_CANDIDATES
    }
    feasible = [margin for margin in ACCURACY_MARGIN_CANDIDATES
                if accuracy_requirements[str(margin)] <= max(CANARY_SESSION_CHOICES)]
    accuracy_margin = feasible[0] if feasible else max(ACCURACY_MARGIN_CANDIDATES)
    n_accuracy = accuracy_requirements[str(accuracy_margin)]
    uncapped = max(n_reuse, n_amortized, n_accuracy, min(CANARY_SESSION_CHOICES))
    choices = [value for value in CANARY_SESSION_CHOICES if value >= uncapped]
    sessions = choices[0] if choices else max(CANARY_SESSION_CHOICES)
    planned = sessions * len(ARMS) * SESSION_LENGTH
    value = {
        "schema": "e5-g2-canary-freeze-v1",
        "authority": "docs/analysis/experiment-design--e5-fair-comparison.md §3c A-E5-1",
        "created_after_smoke_before_canary": True,
        "cell": {"reuse_factor": REUSE_FACTOR, "session_length": SESSION_LENGTH,
                 "budget_pct": BUDGET_PCT},
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "runtime": "Codex CLI",
        "schedule_seed": CANARY_SCHEDULE_SEED,
        "session_count": sessions,
        "session_count_choices_pre_smoke": list(CANARY_SESSION_CHOICES),
        "right_size": {
            "method": "paired-session normal approximation, one-sided alpha=.05, power=.80",
            "n_reuse": n_reuse,
            "n_amortized": n_amortized,
            "n_accuracy": n_accuracy,
            "accuracy_candidates": accuracy_requirements,
            "uncapped_max": uncapped,
            "selected": sessions,
            "cap": max(CANARY_SESSION_CHOICES),
            "cap_binding": uncapped > max(CANARY_SESSION_CHOICES),
        },
        "reuse_margin": cost_margin,
        "accuracy_noninferiority_margin": accuracy_margin,
        "cost_margin_selection_rule": (
            "largest of {5%,10%} no greater than half smoke reuse-saving; fallback 5%"
        ),
        "accuracy_margin_selection_rule": (
            "smallest of {5pp,10pp} whose zero-difference paired-session variance "
            "right-size fits registered max 12; fallback 10pp"
        ),
        "decision_rule": {
            "ci": "paired-session bootstrap 10,000, deterministic one-sided 95% lower",
            "CONFIRM": (
                "reuse fresh-saving CI lower >= cost margin AND amortized fresh-saving "
                "CI lower >= 0 AND overall accuracy-difference CI lower >= -accuracy margin"
            ),
            "REFUTE": "CONFIRM conjunction not met",
            "multi_hop": "not measured under A-E5-1",
        },
        "planned_turns": planned,
        "attempt_cap": planned + max(16, planned // 4),
        "max_turn_retries": MAX_TURN_RETRIES,
        "bootstrap_reps": BOOTSTRAP_REPS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "smoke_summary_sha256": hash_json(smoke),
        "execution_code_git_hash": code_git_hash,
        "source_sha256": source_sha256,
    }
    value["sha256"] = hash_json(value)
    return value


def canary_verdict(summary: dict, freeze: dict, retrieval_audit: dict) -> dict:
    complete = bool(summary.get("complete"))
    reuse_lower = summary["reuse_fresh_saving"]["ci_lower_one_sided_95"]
    amortized_lower = summary["amortized_fresh_saving"]["ci_lower_one_sided_95"]
    accuracy_lower = summary["accuracy_difference"]["ci_lower_one_sided_95"]
    reuse_pass = complete and reuse_lower is not None and reuse_lower >= freeze["reuse_margin"]
    amortized_pass = complete and amortized_lower is not None and amortized_lower >= 0
    accuracy_pass = (
        complete and accuracy_lower is not None
        and accuracy_lower >= -float(freeze["accuracy_noninferiority_margin"])
    )
    status = "INCOMPLETE" if not complete else (
        "CONFIRM" if reuse_pass and amortized_pass and accuracy_pass else "REFUTE"
    )
    ops = summary["karc_operations"]
    rag = summary["by_arm"].get("rag-bm25", {})
    strong = (
        retrieval_audit["gold_containment_rate"] == 1
        and retrieval_audit["mean_budget_utilization"] >= 0.8
        and retrieval_audit["rank_order_preserved"]
    )
    value = {
        "schema": "e5-g2-verdict-v1",
        "computed_status": status,
        "decision_owner": "main session",
        "runtime_scope": "Codex CLI 0.144.x / gpt-5.6-luna subscription runtime",
        "complete": complete,
        "amendment": "A-E5-1; multi-hop not measured or binding",
        "conjuncts": {
            "reuse_margin": {"margin": freeze["reuse_margin"],
                             "ci_lower": reuse_lower, "pass": reuse_pass},
            "amortized_real_cache": {"threshold": 0,
                                     "ci_lower": amortized_lower, "pass": amortized_pass},
            "overall_accuracy_noninferiority": {
                "margin": freeze["accuracy_noninferiority_margin"],
                "ci_lower": accuracy_lower, "pass": accuracy_pass,
            },
        },
        "stage2_disposition": (
            "cost axis revived candidate; main confirmation required" if status == "CONFIRM"
            else "quality-at-capacity alignment recommended" if status == "REFUTE"
            else "no decision"
        ),
        "retrieval_crippling_self_audit": {
            "pass": strong,
            "gold_containment_rate": retrieval_audit["gold_containment_rate"],
            "mean_budget_utilization": retrieval_audit["mean_budget_utilization"],
            "bm25_rank_order_preserved": retrieval_audit["rank_order_preserved"],
            "rag_cache_credit_observed": summary["rag_cache_credit_observed"],
            "rag_cache_read_per_task": rag.get("cache_read_per_task"),
        },
        "mcp_routing": {
            "channel_adoption_rate": ops["mcp_channel_adoption_rate"],
            "native_managed_attempts": ops["native_managed_attempts"],
            "native_managed_denied": ops["native_managed_denied"],
        },
        "freeze_sha256": freeze["sha256"],
    }
    value["sha256"] = hash_json(value)
    return value
