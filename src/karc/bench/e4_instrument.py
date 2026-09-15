"""E4 lifecycle/exposure instrumentation (deterministic, content-free).

Only fixture document/version identifiers are persisted.  Command text and
document contents are deliberately outside this interface (R-9).
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True, order=True)
class DocumentRef:
    artifact_id: str
    version_id: str


@dataclass(frozen=True)
class FetchAccess:
    artifact_id: str | None
    version_id: str | None
    access_type: str
    seq: int
    unresolved: bool = False
    listing: bool = False

    @property
    def exposed(self) -> bool:
        return not self.unresolved and not self.listing


@dataclass(frozen=True)
class LifecycleClassification:
    lifecycle: str | None
    stale_exposure: bool
    source_category: str
    evidence_complete: bool
    answer_class: str


def resolve_fetch_access(
    manifest: dict, path: str, *, access_type: str, seq: int,
    listing: bool = False,
) -> FetchAccess:
    """Resolve a file path before any surrounding command is hashed.

    Directory listings are observable accesses but never evidence exposure;
    a partial read still exposes the whole document conservatively.
    """
    normalized = path.removeprefix("./")
    for version_id, row in manifest["artifacts"].items():
        if normalized == row["path"] or normalized.endswith("/" + row["path"]):
            return FetchAccess(
                artifact_id=row["artifact_id"], version_id=version_id,
                access_type=access_type, seq=seq, listing=listing,
            )
    return FetchAccess(None, None, access_type, seq, unresolved=True,
                       listing=listing)


def refs_for_versions(manifest: dict, versions: Iterable[str]) -> set[DocumentRef]:
    refs: set[DocumentRef] = set()
    for version_id in versions:
        row = manifest["artifacts"].get(version_id)
        if row is None:
            raise KeyError(f"unresolved version id: {version_id}")
        refs.add(DocumentRef(row["artifact_id"], version_id))
    return refs


def exposure_set(
    manifest: dict, *, injected: Iterable[str] = (),
    working_set: Iterable[str] = (), fetches: Iterable[FetchAccess] = (),
    require_resolved: bool = True,
) -> set[DocumentRef]:
    """E(t,a,r) = injected union working-set union allowed fetch."""
    accesses = list(fetches)
    unresolved = sum(access.unresolved for access in accesses)
    if require_resolved and unresolved:
        raise AssertionError(f"unresolved fetch paths: {unresolved}")
    exposed_fetches = [access.version_id for access in accesses if access.exposed]
    return refs_for_versions(
        manifest, [*injected, *working_set,
                   *(v for v in exposed_fetches if v is not None)],
    )


def source_category(required: set[DocumentRef], stale: set[DocumentRef],
                    exposure: set[DocumentRef]) -> str:
    has_current = bool(required & exposure)
    has_stale = bool(stale & exposure)
    if has_current and has_stale:
        return "both"
    if has_current:
        return "current_only"
    if has_stale:
        return "stale_only"
    return "neither"


def classify_lifecycle(
    *, required: set[DocumentRef], stale: set[DocumentRef],
    exposure: set[DocumentRef], value_match: str, format_ok: bool,
    abstained: bool, format_is_failure: bool = False,
) -> LifecycleClassification:
    """Apply §4.4's fixed precedence; L4 is an orthogonal boolean."""
    evidence_complete = required <= exposure
    stale_exposure = bool(stale & exposure)
    if abstained:
        lifecycle = "L6"
        answer_class = "abstain"
    elif value_match == "stale":
        lifecycle = "L5"
        answer_class = "stale-answer"
    elif value_match == "expected":
        answer_class = "correct"
        lifecycle = "L3" if format_is_failure and not format_ok else None
    elif not evidence_complete:
        lifecycle = "L1"
        answer_class = "other-wrong"
    else:
        lifecycle = "L2"
        answer_class = "other-wrong"
    return LifecycleClassification(
        lifecycle=lifecycle, stale_exposure=stale_exposure,
        source_category=source_category(required, stale, exposure),
        evidence_complete=evidence_complete, answer_class=answer_class,
    )


def classification_hash(rows: Iterable[LifecycleClassification | dict]) -> str:
    normalized = [asdict(row) if isinstance(row, LifecycleClassification) else row
                  for row in rows]
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _load_latest_runs(path: Path) -> list[dict]:
    latest: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("run_id") and row.get("failure_class") in {"ok", "task"}:
                latest[row["run_id"]] = row
    return list(latest.values())


def e3_rerun1_regression(repo_root: str | Path) -> dict:
    """Recompute the observable A4a/A8 joins from frozen E3 raw.

    This is a validation-only read.  E3 rows/verdicts are never rewritten.
    Search-only remains N/A because E3 hashed the command before path
    resolution, exactly the asymmetry E4 fixes.
    """
    root = Path(repo_root)
    raw = root / "docs/experiments/E3-comparative/rerun-1/grid/raw"
    fixture = root / "fixture/e3-rerun-1"
    runs = _load_latest_runs(raw / "runs.jsonl")
    working = _load_json(raw / "working_sets.json")["policy"]
    selected = _load_json(raw / "grid_tasks.json")["tasks"]
    tasks_all = {row["task_id"]: row for row in _load_json(fixture / "tasks.json")}
    tasks = {row["task_id"]: tasks_all[row["task_id"]] for row in selected}
    retrieval = _load_json(fixture / "retrieval/plans.json")
    manifest = _load_json(fixture / "manifest.json")
    all_artifacts = set(manifest["artifacts"])

    def evidence(task: dict) -> set[str]:
        return set(task.get("evidence_chain") or task.get("required_artifacts") or [])

    def gold(task: dict) -> str:
        return str((task.get("answer_fact") or {}).get("artifact") or task["chain"][-1])

    def stale(task: dict) -> set[str]:
        return set(task.get("forbidden_artifacts") or [])

    sources: dict[tuple[str, str], set[str] | None] = {}
    for task_id in tasks:
        for arm in ("karc", "classic", "recency"):
            sources[(arm, task_id)] = set(working[arm]["by_task"][task_id])
        for arm in ("rag-bm25", "rag-embed"):
            sources[(arm, task_id)] = set(retrieval[arm][task_id]["artifact_ids"])
        sources[("static-full", task_id)] = all_artifacts
        sources[("search-only", task_id)] = None

    a4a: dict[str, dict] = {}
    for arm in ("karc", "classic", "recency"):
        failed = [r for r in runs if r["arm"] == arm and not r["passed"]]
        present = sum(evidence(tasks[r["task_id"]]) <= (sources[(arm, r["task_id"])] or set())
                      for r in failed)
        a4a[arm] = {"failed_runs": len(failed), "all_evidence_resident": present,
                    "any_evidence_missing": len(failed) - present}

    stale_tasks = [task_id for task_id, task in tasks.items()
                   if stale(task) or task.get("forbidden_stale")]
    a8: dict[str, dict] = {}
    for arm in ("karc", "classic", "recency", "rag-bm25", "rag-embed",
                "static-full", "search-only"):
        counts: Counter[str] = Counter()
        for task_id in stale_tasks:
            source = all_artifacts if arm == "search-only" else (sources[(arm, task_id)] or set())
            current = gold(tasks[task_id]) in source
            old = bool(stale(tasks[task_id]) & source)
            counts["both" if current and old else "current_only" if current
                   else "stale_only" if old else "neither"] += 1
        a8[arm] = {
            "categories": dict(sorted(counts.items())),
            "actual_fetch_observable": arm != "search-only",
        }
    stale_answers = sum(r.get("answer_class") == "stale" for r in runs
                        if r["task_id"] in set(stale_tasks))
    result = {"A4a": a4a, "A8": a8, "stale_answers_total": stale_answers,
              "search_only": "N/A", "stale_task_count": len(stale_tasks)}
    result["sha256"] = hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return result
