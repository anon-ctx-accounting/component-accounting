"""Stage 2 task definitions (experiment-design §6.5, E2-1/E2-2 input).

60 tasks (main 40 + spare 20, spares = every ``seq % 3 == 0`` so replacements
span all phases/types) over the materialized fixture, sequenced A → B →
return-A so the policy state accumulates across a workload shift (F-B flavor)
with supersede/conflict pressure (F-C flavor).

Ground truth is constructed, not judged: each task's answer is a namespaced
config fact placed by ``corpus_text`` into exactly its required document(s);
tests assert the regex hits required docs and nothing else. Grading is
deterministic (regex), per R-17.
"""

from __future__ import annotations

import json
from pathlib import Path

from karc.replay.corpus import Corpus

from . import nonce


def _prompt(corpus: Corpus, aid: str, fact: dict, role: str, extra: dict) -> str:
    meta = corpus.artifacts[aid]
    seed, ko = corpus.seed, meta.language == "ko"
    comp = nonce.sys_name(seed, f"comp-{aid}")
    key = fact["key"]
    form = (f"답은 정확히 `{key} = <값>` 형식의 한 줄로만 출력하라."
            if ko else
            f"Answer with a single line in exactly this form: `{key} = <value>`.")
    if role == "superseded_v2":
        q = (f"{comp} 정책 규격에서 현재 유효한(최신 개정판의) `{key}` 값을 찾아라. "
             "구판이 존재할 수 있으니 대체 관계를 확인하라."
             if ko else
             f"Find the currently valid (latest-revision) value of `{key}` in the "
             f"{comp} policy spec. An older revision may exist — check supersession.")
    elif role == "conflict_x":
        q = (f"{comp} 사양에서 확정(ratified)된 `{key}` 값을 찾아라. "
             "검토 초안이 존재할 수 있으니 확정본을 사용하라."
             if ko else
             f"Find the ratified value of `{key}` for {comp}. A review draft may "
             "exist — use the ratified spec.")
    elif role == "chain":
        start = corpus.artifacts[extra["chain"][0]].path
        q = (f"`{start}`에서 시작하는 배포 절차를 끝까지 따라가 최종 단계의 "
             f"`{key}` 값을 찾아라."
             if ko else
             f"Follow the rollout procedure starting at `{start}` to its final "
             f"step and find the value of `{key}`.")
    elif role == "critical":
        q = (f"{comp} 장애 복구 runbook의 재기동 구성에서 `{key}` 값을 찾아라."
             if ko else
             f"In the incident recovery runbook for {comp}, find the restart "
             f"configuration value of `{key}`.")
    else:  # hot / oversize / shared hot
        q = (f"{comp} 문서에서 `{key}` 값을 찾아라."
             if ko else
             f"Find the configured value of `{key}` in the {comp} documentation.")
    return f"{q} {form}"


def _task(corpus: Corpus, seq: int, aid: str, fact_idx: int, role: str,
          manifest: dict, required: list[str], forbidden: list[str],
          phase: str, extra: dict | None = None) -> dict:
    extra = extra or {}
    fact = manifest["artifacts"][aid]["facts"][fact_idx]
    proj = manifest["artifacts"][aid]["path"].split("/")[1] if "/" in \
        manifest["artifacts"][aid]["path"] else "shared"
    return {
        "task_id": f"S2-{seq:03d}",
        "seq": seq,
        "tier": "spare" if seq % 3 == 0 else "main",
        "phase": phase,
        "scope": {"a": "project-a", "b": "project-b"}.get(proj, "shared"),
        "prompt": _prompt(corpus, aid, fact, role, extra),
        "required_artifacts": required,
        "forbidden_stale": forbidden,
        "chain": extra.get("chain", []),
        "critical_involved": [aid] if role == "critical" else [],
        "expected_answer": {"type": "regex", "value": nonce.answer_regex(fact)},
        "answer_fact": {"artifact": aid, "fact_idx": fact_idx,
                        "key": fact["key"], "value": fact["value"]},
        "outcome_rule": "deterministic-v1",
    }


def generate_tasks(corpus: Corpus, manifest: dict) -> list[dict]:
    g = corpus.groups
    tasks: list[dict] = []
    seq = 0

    def add(aid, role, required=None, forbidden=(), fact_idx=0, phase="A", extra=None):
        nonlocal seq
        seq += 1
        tasks.append(_task(corpus, seq, aid, fact_idx, role, manifest,
                           required or [aid], list(forbidden), phase, extra))

    # ---- Phase A (seq 1..24) -------------------------------------------
    for aid in g["hot_a"][:10]:
        add(aid, "hot", phase="A")
    for chain in g["chains_a"]:
        add(chain[-1], "chain", required=list(chain), phase="A",
            extra={"chain": list(chain)})
    for aid in g["critical_a"]:
        add(aid, "critical", phase="A")
    for aid in g["oversize"][:2]:
        add(aid, "hot", phase="A")
    for v1, v2 in g["superseded_pairs_a"]:
        add(v2, "superseded_v2", required=[v2], forbidden=[v1], phase="A")
    x, _y = g["conflict_pairs_a"][0]
    add(x, "conflict_x", phase="A")

    # ---- Phase B (seq 25..48) ------------------------------------------
    for aid in g["hot_b"][:10]:
        add(aid, "hot", phase="B")
    for chain in g["chains_b"]:
        add(chain[-1], "chain", required=list(chain), phase="B",
            extra={"chain": list(chain)})
    for aid in g["critical_b"]:
        add(aid, "critical", phase="B")
    for aid in g["oversize"][2:]:
        add(aid, "hot", phase="B")
    for v1, v2 in g["superseded_pairs_b"]:
        add(v2, "superseded_v2", required=[v2], forbidden=[v1], phase="B")
    x, _y = g["conflict_pairs_b"][0]
    add(x, "conflict_x", phase="B")
    add(g["shared_hot"][0], "hot", phase="B")

    # ---- Phase A-return (seq 49..60) -----------------------------------
    for aid in g["hot_a"][10:15]:
        add(aid, "hot", phase="A-return")
    for aid in g["hot_a"][:3]:
        add(aid, "hot", fact_idx=1, phase="A-return")
    x, _y = g["conflict_pairs_a"][1]
    add(x, "conflict_x", phase="A-return")
    x, _y = g["conflict_pairs_b"][1]
    add(x, "conflict_x", phase="A-return")
    add(g["shared_hot"][1], "hot", phase="A-return")
    add(g["shared_hot"][2], "hot", phase="A-return")

    assert len(tasks) == 60, f"task count {len(tasks)} != 60"
    assert sum(1 for t in tasks if t["tier"] == "main") == 40
    return tasks


# ---- YAML emission (stdlib-only; every scalar JSON-quoted → valid YAML) ----

def _yaml_scalar(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return json.dumps(v)
    return json.dumps(v, ensure_ascii=False)


def _yaml_block(d: dict, indent: str) -> list[str]:
    lines = []
    for k, v in d.items():
        if isinstance(v, dict):
            lines.append(f"{indent}{k}:")
            lines.extend(_yaml_block(v, indent + "  "))
        elif isinstance(v, list):
            if not v:
                lines.append(f"{indent}{k}: []")
            else:
                lines.append(f"{indent}{k}:")
                lines.extend(f"{indent}  - {_yaml_scalar(x)}" for x in v)
        else:
            lines.append(f"{indent}{k}: {_yaml_scalar(v)}")
    return lines


def write_tasks(tasks: list[dict], out_root: Path) -> None:
    (out_root / "tasks.json").write_text(
        json.dumps(tasks, indent=1, sort_keys=False, ensure_ascii=False),
        encoding="utf-8",
    )
    lines = ["# Stage 2 tasks (generated — see bench/tasks.py; canonical: tasks.json)"]
    for t in tasks:
        block = _yaml_block(t, "  ")
        first = block[0].lstrip()
        lines.append(f"- {first}")
        lines.extend(block[1:])
    (out_root / "tasks.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
