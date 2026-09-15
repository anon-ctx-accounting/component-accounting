"""Stage 2 fixture text materializer (experiment-design §6.1~§6.2, M4a).

Renders the *metadata* corpus of ``replay.corpus`` into an actual
nonce-domain markdown tree. Deterministic in the corpus seed; the manifest
(ground truth: per-doc facts, canaries, supersede/draft/chain relations,
file hashes) lives OUTSIDE the agent-visible ``repo/`` root.

Content rules (§6.2):
- nonce system/component names with hex suffixes → not guessable, not real
- every document carries exactly two namespaced config facts; the fact key
  embeds a doc-scoped token, so a (key, value) pair can only appear in its
  owner (and, with a *different* value, in its superseded/conflict twin)
- every document carries a unique canary tag (E0-3 regression check)
- documents are padded with nonce filler sentences to ≈ ``meta.byte_size``
  so the materialized tree matches the frozen Stage 1 token accounting
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from karc.replay.corpus import Corpus

from . import nonce

# skeleton (headers, canary, two fact lines) can exceed tiny byte targets;
# tests enforce bytes ∈ [0.9·target, max(1.35·target, target+300)]
PAD_STOP = 0.97


def _role_map(corpus: Corpus) -> dict[str, dict]:
    g = corpus.groups
    roles: dict[str, dict] = {}

    def put(aid: str, role: str, **extra):
        roles[aid] = {"role": role, **extra}

    for proj in ("a", "b"):
        for aid in g[f"hot_{proj}"]:
            put(aid, "hot", project=proj)
        for aid in g[f"critical_{proj}"]:
            put(aid, "critical", project=proj)
        for v1, v2 in g[f"superseded_pairs_{proj}"]:
            put(v1, "superseded_v1", project=proj, superseded_by=v2)
            put(v2, "superseded_v2", project=proj, supersedes=v1)
        for x, y in g[f"conflict_pairs_{proj}"]:
            put(x, "conflict_x", project=proj, draft=y)
            put(y, "conflict_y", project=proj, ratified=x)
        for chain in g[f"chains_{proj}"]:
            for j, aid in enumerate(chain):
                nxt = chain[j + 1] if j + 1 < len(chain) else None
                put(aid, "chain", project=proj, chain=list(chain), pos=j, next=nxt)
        for aid in g[f"distractors_{proj}"]:
            put(aid, "distractor", project=proj)
    for aid in g["oversize"]:
        # oversize ids (``{proj}-long-{k}``) are in no other group, so create
        # the entry outright; project is the id prefix (F-C uses it for the
        # nonce system name only).
        put(aid, "oversize", project=aid.split("-", 1)[0])
    for aid in g["useless_preload"]:
        put(aid, "useless_preload", project="shared")
    for aid in g["shared_hot"]:
        put(aid, "hot", project="shared")
    for aid in g["distractors_shared"]:
        put(aid, "distractor", project="shared")
    return roles


def _facts_for(corpus: Corpus, roles: dict[str, dict]) -> dict[str, list[dict]]:
    """Two facts per document. Superseded v1 / conflict y reuse their twin's
    keys with deterministically different values (same doc, older/draft
    numbers) — answer uniqueness is checked in tests, not assumed."""
    seed = corpus.seed
    facts: dict[str, list[dict]] = {}
    for aid, info in roles.items():
        if info["role"] == "superseded_v1":
            twin = info["superseded_by"]
            base = [nonce.fact_for(seed, twin, 0), nonce.fact_for(seed, twin, 1)]
            facts[aid] = [
                {**f, "value": nonce.alt_value(seed, f, f"v1-{aid}")} for f in base
            ]
        elif info["role"] == "conflict_y":
            twin = info["ratified"]
            base = [nonce.fact_for(seed, twin, 0), nonce.fact_for(seed, twin, 1)]
            facts[aid] = [
                {**f, "value": nonce.alt_value(seed, f, f"draft-{aid}")} for f in base
            ]
        else:
            facts[aid] = [nonce.fact_for(seed, aid, 0), nonce.fact_for(seed, aid, 1)]
    return facts


def _title(seed: int, aid: str, info: dict, lang: str) -> str:
    name = nonce.sys_name(seed, f"sys-{info.get('project', 'shared')}")
    comp = nonce.sys_name(seed, f"comp-{aid}")
    if lang == "ko":
        name_ko = nonce.sys_name_ko(seed, f"sys-{info.get('project', 'shared')}")
        return {
            "hot": f"{name_ko} {comp} 구성 가이드",
            "critical": f"{name_ko} 장애 복구 runbook — {comp}",
            "superseded_v1": f"{name_ko} 정책 규격 — {comp} (v1)",
            "superseded_v2": f"{name_ko} 정책 규격 — {comp} (v2)",
            "conflict_x": f"{name_ko} 확정 사양 — {comp}",
            "conflict_y": f"{name_ko} 검토 초안 — {comp}",
            "chain": f"{name_ko} 배포 절차 {info.get('pos', 0) + 1}단계 — {comp}",
            "oversize": f"{name_ko} {comp} 통합 매뉴얼",
            "useless_preload": f"{name_ko} 구형 메모 — {comp}",
            "distractor": f"{name_ko} 작업 노트 — {comp}",
        }[info["role"]]
    return {
        "hot": f"{name} {comp} configuration guide",
        "critical": f"{name} incident recovery runbook — {comp}",
        "superseded_v1": f"{name} policy spec — {comp} (v1)",
        "superseded_v2": f"{name} policy spec — {comp} (v2)",
        "conflict_x": f"{name} ratified spec — {comp}",
        "conflict_y": f"{name} review draft — {comp}",
        "chain": f"{name} rollout procedure step {info.get('pos', 0) + 1} — {comp}",
        "oversize": f"{name} unified manual — {comp}",
        "useless_preload": f"{name} legacy memo — {comp}",
        "distractor": f"{name} working note — {comp}",
    }[info["role"]]


def _role_section(corpus: Corpus, aid: str, info: dict, lang: str) -> list[str]:
    role, ko = info["role"], lang == "ko"
    out: list[str] = []
    if role == "superseded_v2":
        old = corpus.artifacts[info["supersedes"]].path
        out.append(("이 문서는 `%s`(v1)를 대체한다. v1의 수치는 더 이상 유효하지 않다."
                    if ko else
                    "This document supersedes `%s` (v1); v1 values are no longer valid.")
                   % old)
    elif role == "superseded_v1":
        out.append("이 규격은 개정 검토 중이다." if ko
                   else "This spec revision is under review.")
    elif role == "conflict_y":
        rat = corpus.artifacts[info["ratified"]].path
        out.append(("검토용 초안 — 확정 수치는 `%s`를 따른다."
                    if ko else
                    "Review draft — the ratified values live in `%s`.")
                   % rat)
    elif role == "chain" and info.get("next"):
        nxt = corpus.artifacts[info["next"]].path
        out.append(("확정된 파라미터와 다음 단계는 `%s`에 기록되어 있다."
                    if ko else
                    "The ratified parameters for the next step are recorded in `%s`.")
                   % nxt)
    elif role == "critical":
        out.append("## Recovery procedure" if not ko else "## 복구 절차")
        steps = (["게이트웨이를 격리한다", "아래 구성값으로 재기동한다",
                  "복제 상태를 확인한다"] if ko else
                 ["Isolate the gateway.", "Restart with the configuration below.",
                  "Verify replication state."])
        out.extend(f"{i + 1}. {s}" for i, s in enumerate(steps))
    return out


def _render(corpus: Corpus, aid: str, info: dict, facts: list[dict]) -> str:
    meta = corpus.artifacts[aid]
    seed, lang = corpus.seed, meta.language
    ko = lang == "ko"
    parts = [f"# {_title(seed, aid, info, lang)}", "",
             f"Reference tag: `{nonce.canary(seed, aid)}`", ""]
    parts.append(nonce.filler_sentence(seed, aid, 9000, lang))
    parts.append("")
    if info["role"] == "oversize":
        parts.append("## Quick reference" if not ko else "## 핵심 참조 (요약)")
        parts.append("")
    parts.append("## Configuration" if not ko else "## 구성값")
    parts.append("")
    parts.append("```")
    parts.extend(nonce.fact_line(f) for f in facts)
    parts.append("```")
    parts.append("")
    sec = _role_section(corpus, aid, info, lang)
    if sec:
        parts.extend(sec)
        parts.append("")
    parts.append("## Notes" if not ko else "## 비고")
    parts.append("")
    body = "\n".join(parts)
    target = meta.byte_size or meta.size_tok * 4
    i = 0
    section_every = 14  # oversize docs get periodic filler section headers
    while len(body.encode("utf-8")) < target * PAD_STOP:
        extra = ""
        if info["role"] == "oversize" and i % section_every == 0:
            extra = f"\n## Section {i // section_every + 2}\n\n"
        body += extra + nonce.filler_sentence(seed, aid, i, lang) + "\n"
        i += 1
    return body + "\n"


def materialize(corpus: Corpus, out_root: Path) -> dict:
    """Write the fixture tree to ``out_root/repo`` and return the manifest."""
    roles = _role_map(corpus)
    facts = _facts_for(corpus, roles)
    repo = out_root / "repo"
    entries = {}
    for aid, meta in sorted(corpus.artifacts.items()):
        info = roles[aid]
        text = _render(corpus, aid, info, facts[aid])
        p = repo / meta.path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        raw = text.encode("utf-8")
        entries[aid] = {
            "artifact_id": aid,
            "path": meta.path,
            "role": info["role"],
            "language": meta.language,
            "critical": meta.critical,
            "preload": meta.preload,
            "size_tok": meta.size_tok,
            "byte_target": meta.byte_size,
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "canary": nonce.canary(corpus.seed, aid),
            "facts": facts[aid],
            "relations": {k: v for k, v in info.items()
                          if k in ("superseded_by", "supersedes", "draft",
                                   "ratified", "chain", "pos", "next")},
        }
    manifest = {
        "seed": corpus.seed,
        "corpus_content_hash": corpus.content_hash(),
        "n_artifacts": len(entries),
        "artifacts": entries,
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    (out_root / "manifest.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest
