"""Frozen, untuned E3 BM25 and local dense retrieval.

Both retrievers consume the same document-first chunks and use the same
artifact-token budget selector.  Ties are resolved by chunk id, making the
entire mapping from (fixture, query, budget) to injected artifact ids stable.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

BM25_K1 = 1.5
BM25_B = 0.75
CHUNK_MAX_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 0
DENSE_DEVICE = "cpu"
DENSE_BATCH_SIZE = 8

_TERM_RE = re.compile(r"[\w]+(?:[-_.][\w]+)*", re.UNICODE)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    artifact_id: str
    path: str
    text: str
    size_tok: int


def _estimate_tokens(text: str, language: str = "en") -> int:
    raw = len(text.encode("utf-8"))
    return max(1, math.ceil(raw / (2.5 if language == "ko" else 4.0)))


def _split_paragraph(paragraph: str, max_tokens: int, language: str) -> list[str]:
    """Deterministic fallback for a paragraph larger than the fixed limit."""
    words = paragraph.split()
    if not words:
        return []
    pieces: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join([*current, word])
        if current and _estimate_tokens(candidate, language) > max_tokens:
            pieces.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        pieces.append(" ".join(current))
    return pieces


def chunk_fixture(fixture_root: str | Path, manifest: dict,
                  *, max_tokens: int = CHUNK_MAX_TOKENS) -> list[Chunk]:
    """Paragraph-boundary, document-first chunks; no cross-document chunks."""
    root = Path(fixture_root) / "repo"
    chunks: list[Chunk] = []
    for aid, entry in sorted(manifest["artifacts"].items(),
                             key=lambda item: item[1]["path"]):
        text = (root / entry["path"]).read_text(encoding="utf-8")
        language = entry.get("language", "en")
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        units: list[str] = []
        for paragraph in paragraphs:
            if _estimate_tokens(paragraph, language) <= max_tokens:
                units.append(paragraph)
            else:
                units.extend(_split_paragraph(paragraph, max_tokens, language))
        current: list[str] = []
        doc_chunks: list[str] = []
        for unit in units:
            candidate = "\n\n".join([*current, unit])
            if current and _estimate_tokens(candidate, language) > max_tokens:
                doc_chunks.append("\n\n".join(current))
                current = [unit]
            else:
                current.append(unit)
        if current:
            doc_chunks.append("\n\n".join(current))
        for index, body in enumerate(doc_chunks):
            chunks.append(Chunk(
                chunk_id=f"{entry['path']}::{index:03d}", artifact_id=aid,
                path=entry["path"], text=body,
                size_tok=_estimate_tokens(body, language),
            ))
    return chunks


def chunks_sha256(chunks: Sequence[Chunk]) -> str:
    payload = [{**asdict(c), "text_sha256": hashlib.sha256(
        c.text.encode("utf-8")).hexdigest()} for c in chunks]
    for row in payload:
        row.pop("text")
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def tokenize(text: str) -> list[str]:
    return [value.lower() for value in _TERM_RE.findall(text)]


class BM25Index:
    def __init__(self, chunks: Sequence[Chunk], *, k1: float = BM25_K1,
                 b: float = BM25_B):
        self.chunks = list(chunks)
        self.k1 = float(k1)
        self.b = float(b)
        self._tf: list[dict[str, int]] = []
        self._lengths: list[int] = []
        df: dict[str, int] = {}
        for chunk in self.chunks:
            counts: dict[str, int] = {}
            for term in tokenize(chunk.text):
                counts[term] = counts.get(term, 0) + 1
            self._tf.append(counts)
            length = sum(counts.values())
            self._lengths.append(length)
            for term in counts:
                df[term] = df.get(term, 0) + 1
        self._df = df
        self._avgdl = sum(self._lengths) / len(self._lengths) if self._lengths else 0.0

    def rank(self, query: str) -> list[tuple[float, Chunk]]:
        terms = set(tokenize(query))
        n_docs = len(self.chunks)
        ranked: list[tuple[float, Chunk]] = []
        for i, chunk in enumerate(self.chunks):
            score = 0.0
            dl = self._lengths[i]
            for term in terms:
                freq = self._tf[i].get(term, 0)
                if not freq:
                    continue
                df = self._df[term]
                idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
                norm = freq + self.k1 * (
                    1.0 - self.b + self.b * dl / (self._avgdl or 1.0)
                )
                score += idf * (freq * (self.k1 + 1.0)) / norm
            ranked.append((score, chunk))
        ranked.sort(key=lambda item: (-item[0], item[1].chunk_id))
        return ranked


def select_budget(ranked: Iterable[tuple[float, Chunk]], manifest: dict,
                  budget_tokens: int) -> dict:
    """Select whole artifacts within c*, deduplicating multi-chunk documents."""
    selected: list[str] = []
    selected_chunks: list[str] = []
    seen: set[str] = set()
    used = 0
    scored: list[dict] = []
    for score, chunk in ranked:
        if score <= 0.0:
            continue
        aid = chunk.artifact_id
        if aid in seen:
            continue
        seen.add(aid)
        cost = int(manifest["artifacts"][aid]["size_tok"])
        if used + cost > budget_tokens:
            continue
        selected.append(aid)
        selected_chunks.append(chunk.chunk_id)
        used += cost
        scored.append({"artifact_id": aid, "chunk_id": chunk.chunk_id,
                       "score": float(score), "size_tok": cost})
    return {
        "artifact_ids": selected, "chunk_ids": selected_chunks,
        "tokens": used, "budget_tokens": budget_tokens,
        "budget_utilization": used / budget_tokens if budget_tokens else 0.0,
        "ranked_selected": scored,
    }


def bm25_plans(chunks: Sequence[Chunk], tasks: Sequence[dict], manifest: dict,
               budget_tokens: int) -> dict[str, dict]:
    index = BM25Index(chunks)
    return {task["task_id"]: select_budget(index.rank(task["prompt"]), manifest,
                                             budget_tokens)
            for task in tasks}


def dense_rank(chunks: Sequence[Chunk], scores: Sequence[float]) -> list[tuple[float, Chunk]]:
    if len(chunks) != len(scores):
        raise ValueError("dense score count differs from chunk count")
    ranked = [(float(score), chunk) for score, chunk in zip(scores, chunks)]
    ranked.sort(key=lambda item: (-item[0], item[1].chunk_id))
    return ranked


def dense_plans(chunks: Sequence[Chunk], tasks: Sequence[dict], manifest: dict,
                budget_tokens: int, score_rows: Sequence[Sequence[float]]) -> dict[str, dict]:
    if len(tasks) != len(score_rows):
        raise ValueError("dense query score count differs from task count")
    return {
        task["task_id"]: select_budget(dense_rank(chunks, scores), manifest,
                                         budget_tokens)
        for task, scores in zip(tasks, score_rows)
    }


def build_dense_vectors(chunks: Sequence[Chunk], tasks: Sequence[dict], *,
                        model_id: str, revision: str,
                        device: str = DENSE_DEVICE,
                        batch_size: int = DENSE_BATCH_SIZE,
                        progress: bool = True):
    """Load the pinned local SentenceTransformer and return normalized arrays."""
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    torch.manual_seed(3000)
    torch.set_num_threads(4)
    model = SentenceTransformer(
        model_id, revision=revision, device=device, trust_remote_code=False,
    )
    doc_vectors = model.encode(
        [chunk.text for chunk in chunks], batch_size=batch_size,
        show_progress_bar=progress, convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32, copy=False)
    query_vectors = model.encode(
        [task["prompt"] for task in tasks], batch_size=batch_size,
        show_progress_bar=progress, convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32, copy=False)
    scores = query_vectors @ doc_vectors.T
    versions = {}
    for package in ("sentence-transformers", "transformers", "torch", "numpy"):
        versions[package] = importlib.metadata.version(package)
    return doc_vectors, query_vectors, scores, versions


def write_chunks(path: str | Path, chunks: Sequence[Chunk]) -> None:
    value = [asdict(chunk) for chunk in chunks]
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n",
                          encoding="utf-8")


def read_chunks(path: str | Path) -> list[Chunk]:
    return [Chunk(**row) for row in json.loads(Path(path).read_text(encoding="utf-8"))]


def build_retrieval_artifacts(fixture_root: str | Path, manifest: dict,
                              tasks: Sequence[dict], *, budget_tokens: int,
                              model_id: str, revision: str,
                              encode_fn: Callable | None = None,
                              progress: bool = True) -> dict:
    """Build and persist both retrieval plans plus dense raw vectors."""
    import numpy as np

    fixture = Path(fixture_root)
    out = fixture / "retrieval"
    out.mkdir(parents=True, exist_ok=True)
    chunks = chunk_fixture(fixture, manifest)
    write_chunks(out / "chunks.json", chunks)
    bm25 = bm25_plans(chunks, tasks, manifest, budget_tokens)
    builder = encode_fn or build_dense_vectors
    doc_vectors, query_vectors, scores, versions = builder(
        chunks, tasks, model_id=model_id, revision=revision,
        device=DENSE_DEVICE, batch_size=DENSE_BATCH_SIZE, progress=progress,
    )
    doc_vectors = np.asarray(doc_vectors, dtype=np.float32)
    query_vectors = np.asarray(query_vectors, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    embed = dense_plans(chunks, tasks, manifest, budget_tokens, scores)
    np.save(out / "document_vectors.npy", doc_vectors, allow_pickle=False)
    np.save(out / "query_vectors.npy", query_vectors, allow_pickle=False)
    plans = {"rag-bm25": bm25, "rag-embed": embed}
    (out / "plans.json").write_text(
        json.dumps(plans, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    vector_hashes = {}
    for name in ("document_vectors.npy", "query_vectors.npy"):
        vector_hashes[name] = hashlib.sha256((out / name).read_bytes()).hexdigest()
    metadata = {
        "schema_version": 1,
        "chunks": {"strategy": "document-first-paragraph", "max_tokens": CHUNK_MAX_TOKENS,
                   "overlap_tokens": CHUNK_OVERLAP_TOKENS, "count": len(chunks),
                   "sha256": chunks_sha256(chunks)},
        "budget": {"tokens": budget_tokens, "selector": "whole-artifact-greedy"},
        "bm25": {"k1": BM25_K1, "b": BM25_B, "tokenizer": "unicode-word-v1"},
        "embedder": {
            "model_id": model_id, "revision": revision, "device": DENSE_DEVICE,
            "batch_size": DENSE_BATCH_SIZE, "normalize_embeddings": True,
            "trust_remote_code": False, "dtype": "float32", "versions": versions,
        },
        "vector_hashes": vector_hashes,
        "plans_sha256": hashlib.sha256((out / "plans.json").read_bytes()).hexdigest(),
    }
    (out / "config.json").write_text(
        json.dumps(metadata, indent=1, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata
