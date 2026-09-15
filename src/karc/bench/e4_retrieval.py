"""Frozen E4 retrieval-plan builders and model-stage supply interfaces."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

from karc.bench.e3_retrieval import (
    BM25_B,
    BM25_K1,
    BM25Index,
    Chunk,
    bm25_plans,
    dense_plans,
    select_budget,
)
from karc.bench.e4_fixture import EMBED_MODEL_ID, EMBED_REVISION


RAG_VA_RULES = {
    "freshness_filter": "exclude not-yet-valid and superseded-at-or-before-task-seq",
    "conflict_abstain": (
        "group by artifact_id; if multiple unresolved current versions remain, "
        "keep a unique greatest valid_from_seq, otherwise exclude the group"
    ),
    "metadata_fields": ["artifact_id", "version_id", "valid_from_seq",
                        "superseded_at_seq"],
    "tuning": "none",
}


def in_memory_chunks(entries_with_text: dict[str, dict]) -> list[Chunk]:
    """One deterministic document-first chunk per compact E4 document."""
    return [
        Chunk(chunk_id=f"{row['path']}::000", artifact_id=version_id,
              path=row["path"], text=row["text"], size_tok=int(row["size_tok"]))
        for version_id, row in sorted(entries_with_text.items(),
                                      key=lambda item: item[1]["path"])
    ]


def _active(entry: dict, seq: int) -> bool:
    valid_from = entry.get("valid_from_seq")
    superseded_at = entry.get("superseded_at_seq")
    return (valid_from is not None and int(valid_from) <= seq
            and (superseded_at is None or int(superseded_at) > seq))


def version_aware_ranked(
    ranked: Iterable[tuple[float, Chunk]], manifest: dict, *, task_seq: int,
) -> tuple[list[tuple[float, Chunk]], dict]:
    """Apply B8 freshness and conflict rules before the common selector."""
    fresh = [(score, chunk) for score, chunk in ranked
             if _active(manifest["artifacts"][chunk.artifact_id], task_seq)]
    by_artifact: dict[str, list[tuple[float, Chunk]]] = defaultdict(list)
    for item in fresh:
        by_artifact[manifest["artifacts"][item[1].artifact_id]["artifact_id"]].append(item)
    allowed: set[str] = set()
    unresolved: list[str] = []
    conflicts = 0
    for artifact_id, candidates in by_artifact.items():
        versions = {item[1].artifact_id for item in candidates}
        if len(versions) == 1:
            allowed.update(versions)
            continue
        conflicts += 1
        starts = [(manifest["artifacts"][version].get("valid_from_seq"), version)
                  for version in versions]
        greatest = max(value for value, _ in starts if value is not None)
        winners = [version for value, version in starts if value == greatest]
        if len(winners) == 1:
            allowed.add(winners[0])
        else:
            unresolved.append(artifact_id)
    filtered = [(score, chunk) for score, chunk in fresh
                if chunk.artifact_id in allowed]
    return filtered, {
        "freshness_excluded": len(list(ranked)) - len(fresh)
        if isinstance(ranked, Sequence) else None,
        "conflict_groups": conflicts,
        "unresolved_conflict_groups": sorted(unresolved),
    }


def rag_va_plans(chunks: Sequence[Chunk], tasks: Sequence[dict], manifest: dict,
                 budget_tokens: int) -> dict[str, dict]:
    index = BM25Index(chunks, k1=BM25_K1, b=BM25_B)
    plans: dict[str, dict] = {}
    for task in tasks:
        ranked = index.rank(task["prompt"])
        filtered, audit = version_aware_ranked(ranked, manifest,
                                                task_seq=int(task["seq"]))
        plan = select_budget(filtered, manifest, budget_tokens)
        plan["version_audit"] = audit
        plans[task["task_id"]] = plan
    return plans


def deterministic_retrieval_plans(
    chunks: Sequence[Chunk], tasks: Sequence[dict], manifest: dict,
    budget_tokens: int, *, dense_score_rows: Sequence[Sequence[float]] | None = None,
) -> dict[str, dict]:
    """Build no-model BM25/VA plans; accept pinned dense scores when present."""
    plans: dict[str, dict] = {
        "rag-bm25": bm25_plans(chunks, tasks, manifest, budget_tokens),
        "rag-va": rag_va_plans(chunks, tasks, manifest, budget_tokens),
    }
    if dense_score_rows is None:
        plans["rag-embed"] = {
            "status": "interface-only; pinned local embedding build required",
            "model_id": EMBED_MODEL_ID,
            "revision": EMBED_REVISION,
            "budget_tokens": budget_tokens,
        }
    else:
        plans["rag-embed"] = dense_plans(
            chunks, tasks, manifest, budget_tokens, dense_score_rows,
        )
    return plans


@dataclass(frozen=True)
class ModelStageSupplyInterface:
    arm: str
    supply_sources: tuple[str, ...]
    deterministic_rule: str
    status: str = "interface-only-model-stage-not-started"


MODEL_STAGE_INTERFACES = {
    "static-full": ModelStageSupplyInterface(
        "static-full", ("all-current-and-stale-corpus-documents",),
        "inject full fixture corpus without budget truncation",
    ),
    "solar-adm": ModelStageSupplyInterface(
        "solar-adm", ("utility-admission-working-set",),
        "frozen Bayesian utility admission reconstruction; fidelity-limited comparator",
    ),
    "karc+va": ModelStageSupplyInterface(
        "karc+va", ("karc-working-set", "rag-va-miss-fallback"),
        "supply K-ARC residents then version-aware BM25 on a required-evidence miss",
    ),
}


def retrieval_config() -> dict:
    return {
        "rag-bm25": {"k1": BM25_K1, "b": BM25_B, "tuning": "none"},
        "rag-embed": {"model_id": EMBED_MODEL_ID,
                      "revision": EMBED_REVISION, "tuning": "none"},
        "rag-va": dict(RAG_VA_RULES),
        "model_stage_interfaces": {
            key: value.__dict__ for key, value in MODEL_STAGE_INTERFACES.items()
        },
    }
