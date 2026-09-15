"""Synthetic fixture corpus — metadata only (experiment-design §6.1).

Stage 1 replay needs artifact *metadata* (id, size_tok, type tags); the
actual nonce-domain text corpus is a Stage 2 deliverable and is deliberately
NOT generated here. Composition follows §6.1:

- 140 artifacts: project A 60 + project B 60 + shared 20
- per project: hot 15, rare-critical 4, superseded pairs 4 (=8 artifacts),
  conflict pairs 2 (=4 artifacts), chains (A: 3×3, B: 2×3), hot-section
  long docs 2 (these four are the §6.1 oversize docs, 8k~30k), distractors
  fill the remainder
- shared 20: useless_preload 6 (V4 targets) + shared-hot 6 + distractor 8
- sizes: log-normal median 800 / p90 ≈ 4k (σ = ln5/1.2816), clipped to
  [60, 6000] for non-oversize docs
- language: ~20% Korean (Q9 requirement)
- preload set: useless_preload 6 + shared-hot 6 + 5 hot per project
  + superseded v1 4 (Amendment A-4, below)

Amendment A-4 (2026-07-18, post-hoc corpus amendment — experiment-design
"Amendments" A-4): the preload set additionally contains the v1 documents of
**4 of the 8 superseded pairs — all of project A** (``a-sup-{0..3}-v1``), so
that a static preload has real stale exposure once the supersede event fires
(E0-2: real-world preloads go stale). Spec frozen before the rerun:

- placement: project A pairs only, because the supersede event exists only
  in family F-C, which uses ``superseded_pairs_a`` (v1→STALE at task 25);
  project-B v1 docs are never referenced or invalidated by any Stage 1
  family, so preloading them would create no exposure. F-A/F-B have no
  supersede events → their static stale stays 0 (realistic, per A-4).
- count: 4 of 8 pairs (half) — in reality only part of a preload goes stale.
- path convention: identical to the hot-doc preload promotion
  (``preload/a-<artifact_id>.md``), which places them after ``a-a-hot-*``
  and before ``b-b-hot-*`` in the §8.4-1 static truncation order.
- invariant: artifact count (140), type mix, size distribution, language
  mix, corpus total tokens, and the supersede timing (F-C task 25) are all
  unchanged; preload set grows 22 → 26 artifacts and ``preload_tokens``
  grows by the four v1 sizes (seed-dependent). Group ``preload_stale_v1``
  records the membership; the corpus content hash is re-frozen per run.

Everything is a deterministic function of ``seed`` (FR-R3); the corpus hash
goes into the run's reproduction stamp (FR-R9).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field

from karc.policy.model import ArtifactMeta

SIGMA = math.log(5) / 1.2816  # p90/median = 5 → σ ≈ 1.2557
MEDIAN_TOK = 800
CLIP_LO, CLIP_HI = 60, 6000


def _hash_int(*parts) -> int:
    h = hashlib.sha256(":".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(h[:16], 16)


def _unit(*parts) -> float:
    """Deterministic uniform in [0,1) keyed by parts (stable across runs)."""
    return _hash_int(*parts) / float(1 << 64)


def _lognormal_size(seed: int, name: str) -> int:
    # Box-Muller from two deterministic uniforms
    u1 = max(_unit(seed, name, "u1"), 1e-12)
    u2 = _unit(seed, name, "u2")
    z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    size = int(MEDIAN_TOK * math.exp(SIGMA * z))
    return max(CLIP_LO, min(CLIP_HI, size))


@dataclass
class Corpus:
    seed: int
    artifacts: dict[str, ArtifactMeta] = field(default_factory=dict)
    groups: dict[str, list] = field(default_factory=dict)
    # groups keys: hot_a, hot_b, critical_a, critical_b, superseded_pairs_a,
    # superseded_pairs_b (list of (v1,v2)), conflict_pairs_a/_b, chains_a/_b
    # (list of [d1,d2,d3]), oversize, useless_preload, shared_hot,
    # distractors_a, distractors_b, distractors_shared, fresh_pool, harmful,
    # preload_set

    @property
    def total_tokens(self) -> int:
        return sum(m.size_tok for m in self.artifacts.values())

    @property
    def preload_tokens(self) -> int:
        return sum(self.artifacts[a].size_tok for a in self.groups["preload_set"])

    def registry(self) -> dict[str, ArtifactMeta]:
        return dict(self.artifacts)

    def content_hash(self) -> str:
        payload = json.dumps(
            {
                "seed": self.seed,
                "artifacts": {
                    a: [m.size_tok, m.path, m.critical, m.preload, m.language, list(m.tags)]
                    for a, m in sorted(self.artifacts.items())
                },
                "groups": {k: sorted(map(str, v)) for k, v in self.groups.items()},
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _add(
    corpus: Corpus,
    art_id: str,
    size: int,
    path: str,
    tags: tuple[str, ...],
    critical: bool = False,
    preload: bool = False,
) -> str:
    lang = "ko" if _unit(corpus.seed, art_id, "lang") < 0.20 else "en"
    corpus.artifacts[art_id] = ArtifactMeta(
        artifact_id=art_id,
        size_tok=size,
        path=path,
        critical=critical,
        preload=preload,
        language=lang,
        tags=tags,
        byte_size=int(size * (2.5 if lang == "ko" else 4)),
    )
    return art_id


def generate_corpus(seed: int, rehab: bool = False) -> Corpus:
    """Deterministic metadata corpus for ``seed``.

    ``rehab`` (Amendment A-5(b), 2026-07-18): when True, add a ``rehab`` group
    of 4 good documents (``b-dis-15..18`` — in no forbidden/harmful list, and
    otherwise unreferenced by any base family) for the C11′ rehabilitation
    scenario. This adds NO artifacts and changes NO sizes/total tokens — only
    ``groups`` gains the ``rehab`` key, so the content hash changes (a new,
    frozen fixture on top of the A-4 corpus) while every rehab=False caller
    (E1-1..E1-4) stays byte-identical. F-C's workload scripts the rehab
    corrected×1 + re-required references only when this group is present.
    """
    c = Corpus(seed=seed)
    g = c.groups

    def sz(name: str) -> int:
        return _lognormal_size(seed, name)

    for proj in ("a", "b"):
        g[f"hot_{proj}"] = [
            _add(c, f"{proj}-hot-{i:02d}", sz(f"{proj}-hot-{i}"),
                 f"docs/{proj}/hot-{i:02d}.md", ("hot",))
            for i in range(15)
        ]
        g[f"critical_{proj}"] = [
            _add(c, f"{proj}-crit-{i}", sz(f"{proj}-crit-{i}"),
                 f"docs/{proj}/runbooks/recovery-{i}.md", ("critical",), critical=True)
            for i in range(4)
        ]
        pairs = []
        for i in range(4):
            v1 = _add(c, f"{proj}-sup-{i}-v1", sz(f"{proj}-sup-{i}-v1"),
                      f"docs/{proj}/spec/policy-{i}-v1.md", ("superseded_v1",))
            v2 = _add(c, f"{proj}-sup-{i}-v2", sz(f"{proj}-sup-{i}-v2"),
                      f"docs/{proj}/spec/policy-{i}-v2.md", ("superseded_v2",))
            pairs.append((v1, v2))
        g[f"superseded_pairs_{proj}"] = pairs
        cpairs = []
        for i in range(2):
            x = _add(c, f"{proj}-conf-{i}-x", sz(f"{proj}-conf-{i}-x"),
                     f"docs/{proj}/notes/fact-{i}-x.md", ("conflict",))
            y = _add(c, f"{proj}-conf-{i}-y", sz(f"{proj}-conf-{i}-y"),
                     f"docs/{proj}/notes/fact-{i}-y.md", ("conflict",))
            cpairs.append((x, y))
        g[f"conflict_pairs_{proj}"] = cpairs
        n_chains = 3 if proj == "a" else 2
        chains = []
        for i in range(n_chains):
            chain = [
                _add(c, f"{proj}-chain-{i}-{j}", sz(f"{proj}-chain-{i}-{j}"),
                     f"docs/{proj}/guide/step-{i}-{j}.md", ("chain",))
                for j in range(3)
            ]
            chains.append(chain)
        g[f"chains_{proj}"] = chains

    # hot-section long docs = the §6.1 oversize four (8k~30k tokens)
    g["oversize"] = []
    for k, proj in enumerate(("a", "a", "b", "b")):
        size = 8000 + int(_unit(seed, f"oversize-{k}") * 22000)
        g["oversize"].append(
            _add(c, f"{proj}-long-{k}", size, f"docs/{proj}/reference/manual-{k}.md",
                 ("hot_section", "oversize"))
        )

    # distractors per project (fills project count to 60)
    n_da = 60 - (15 + 4 + 8 + 4 + 9 + 2)  # = 18
    n_db = 60 - (15 + 4 + 8 + 4 + 6 + 2)  # = 21
    g["distractors_a"] = [
        _add(c, f"a-dis-{i:02d}", sz(f"a-dis-{i}"), f"docs/a/misc/note-{i:02d}.md",
             ("distractor",))
        for i in range(n_da)
    ]
    g["distractors_b"] = [
        _add(c, f"b-dis-{i:02d}", sz(f"b-dis-{i}"), f"docs/b/misc/note-{i:02d}.md",
             ("distractor",))
        for i in range(n_db)
    ]

    # shared 20: useless_preload 6 + shared-hot 6 + shared distractor 8
    g["useless_preload"] = [
        _add(c, f"s-useless-{i}", sz(f"s-useless-{i}"), f"preload/legacy-{i}.md",
             ("useless_preload",), preload=True)
        for i in range(6)
    ]
    g["shared_hot"] = [
        _add(c, f"s-hot-{i}", sz(f"s-hot-{i}"), f"preload/common-{i}.md",
             ("hot", "shared"), preload=True)
        for i in range(6)
    ]
    g["distractors_shared"] = [
        _add(c, f"s-dis-{i}", sz(f"s-dis-{i}"), f"docs/shared/misc/note-{i}.md",
             ("distractor", "shared"))
        for i in range(8)
    ]

    # preload flags for 5 hot docs per project (import-chain approximation)
    for proj in ("a", "b"):
        for a in g[f"hot_{proj}"][:5]:
            m = c.artifacts[a]
            c.artifacts[a] = ArtifactMeta(
                artifact_id=m.artifact_id, size_tok=m.size_tok,
                path=f"preload/{proj}-{m.artifact_id}.md", critical=m.critical,
                pinned=m.pinned, preload=True, language=m.language, tags=m.tags,
                byte_size=m.byte_size,
            )
    # Amendment A-4: v1 docs of all 4 project-A superseded pairs join the
    # preload set (same path convention as the hot-doc promotion above).
    # F-C's supersede event (task 25) turns exactly these STALE, giving the
    # static preload real stale exposure (spec rationale in the docstring).
    g["preload_stale_v1"] = []
    for v1, _v2 in g["superseded_pairs_a"]:
        m = c.artifacts[v1]
        c.artifacts[v1] = ArtifactMeta(
            artifact_id=m.artifact_id, size_tok=m.size_tok,
            path=f"preload/a-{m.artifact_id}.md", critical=m.critical,
            pinned=m.pinned, preload=True, language=m.language, tags=m.tags,
            byte_size=m.byte_size,
        )
        g["preload_stale_v1"].append(v1)
    g["preload_set"] = (
        list(g["useless_preload"]) + list(g["shared_hot"])
        + g["hot_a"][:5] + g["hot_b"][:5] + list(g["preload_stale_v1"])
    )

    # role assignments used by F-C (deterministic slices, no extra artifacts)
    g["harmful"] = g["distractors_a"][:2]  # scripted corrected×2 targets (V5)
    g["fresh_pool"] = g["distractors_b"][:15]  # V12 no-locality one-shots
    g["scan_pool"] = g["distractors_a"][2:] + g["distractors_shared"]  # V1 sweep

    if rehab:  # Amendment A-5(b): C11′ rehabilitation docs (see docstring)
        # b-dis-15..18: good (distractor tag → never forbidden/harmful) and
        # unused by any base family (fresh_pool = distractors_b[:15]).
        g["rehab"] = list(g["distractors_b"][15:19])

    assert len(c.artifacts) == 140, f"corpus size {len(c.artifacts)} != 140"
    return c
