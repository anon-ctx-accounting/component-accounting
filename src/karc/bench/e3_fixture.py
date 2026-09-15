"""Deterministic E3 comparative-benchmark fixture (seed 3000).

The fixture is deliberately separate from the Stage-2 corpus generator.  It
authors the complete 440-task pool required by E3 rev2 and materializes an
agent-visible repository plus an external ground-truth manifest.

Each route group contains a five-document chain, a superseded v1 final, and a
near-miss draft.  Five task fields share the same chain, yielding 88 groups ×
5 tasks = 440 tasks.  The v1 path sorts before v2 and repeats the query terms,
which fixes the stale ranking trap without tuning either retriever.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

from karc.policy.model import ArtifactMeta
from karc.replay.corpus import Corpus

from . import nonce

SEED = 3000
DEV_SEED = 3100
BASELINE_PROFILE = "baseline-v1"
RECALIBRATION_PROFILE_V1 = "recal-v1"
RECALIBRATION_PROFILE_V2 = "recal-v2"
RECALIBRATION_PROFILE = "recal-v3"
RECALIBRATION_PROFILES = (
    RECALIBRATION_PROFILE_V1,
    RECALIBRATION_PROFILE_V2,
    RECALIBRATION_PROFILE,
)
N_GROUPS = 88
TASKS_PER_GROUP = 5
N_TASKS = N_GROUPS * TASKS_PER_GROUP
N_NOISE = 40
MAX_TURNS = 8
EMBED_MODEL_ID = "BAAI/bge-m3"
EMBED_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
FIELD_NAMES = ("relay_code", "window_code", "shard_code", "ledger_code", "beacon_code")


def _sha(*parts: object) -> str:
    return hashlib.sha256(":".join(map(str, parts)).encode("utf-8")).hexdigest()


def _value(seed: int, group: int, slot: int, version: str) -> str:
    if version == "v2":
        return "E3-" + "".join(
            _sha(seed, group, slot, version, step)[:3].upper()
            for step in range(1, 6)
        )
    return f"E3-{_sha(seed, group, slot, version)[:15].upper()}"


def _fragments(seed: int, group: int, position: int) -> list[str]:
    return [_sha(seed, group, slot, "v2", position)[:3].upper()
            for slot in range(TASKS_PER_GROUP)]


def _route(seed: int, group: int) -> str:
    return f"route-{_sha(seed, group, 'route')[:10]}"


def _request_id(seed: int, group: int, slot: int) -> str:
    return f"REQ-{_sha(seed, group, slot, 'request')[:12].upper()}"


def _artifact_language(index: int) -> str:
    # Exact, deterministic 20% at every complete block of five.
    return "ko" if index % 5 == 4 else "en"


def _doc_text(*, seed: int, aid: str, title: str, language: str, body: list[str]) -> str:
    heading = "운영 문서" if language == "ko" else "Operations document"
    filler = nonce.filler_sentence(seed, aid, 0, language)
    return "\n".join([
        f"# {title}", "", f"{heading}.", "",
        f"Reference tag: `{nonce.canary(seed, aid)}`", "",
        *body, "", "## Context", "", filler, nonce.filler_sentence(seed, aid, 1, language), "",
    ])


def _fragment_lines(position: int, fragments: list[str]) -> list[str]:
    return ["```", *[f"{name}_part_{position} = {value}"
                      for name, value in zip(FIELD_NAMES, fragments)], "```"]


def _entry_body(route: str, next_path: str, language: str,
                fragments: list[str]) -> list[str]:
    if language == "ko":
        return [
            f"이 문서는 `{route}`의 승인된 진입점이다.",
            f"다음 단계: `{next_path}`.",
            "최종 값은 각 단계 조각을 1→5 순서로 이어 붙인다.",
            *_fragment_lines(1, fragments),
        ]
    return [
        f"This is the approved entry point for `{route}`.",
        f"Next step: `{next_path}`.",
        "Concatenate the fragment from every link in order 1→5 to form the final value.",
        *_fragment_lines(1, fragments),
    ]


def _middle_body(route: str, position: int, next_path: str, language: str,
                 fragments: list[str]) -> list[str]:
    if language == "ko":
        return [
            f"`{route}` 승인 체인 {position}/5.",
            f"다음 단계: `{next_path}`.",
            "아래 조각을 최종 값의 같은 순번 위치에 사용한다.",
            *_fragment_lines(position, fragments),
        ]
    return [
        f"Approval chain {position}/5 for `{route}`.",
        f"Next step: `{next_path}`.",
        "Use the fragment below in this numbered position of the final value.",
        *_fragment_lines(position, fragments),
    ]


def _final_body(route: str, values: list[str], language: str, *, role: str,
                current_path: str | None = None) -> list[str]:
    lines: list[str]
    if role == "current":
        lines = ([f"`{route}` 승인 체인 5/5. 이 v2 문서만 현재 유효하다."]
                 if language == "ko" else
                 [f"Approval chain 5/5 for `{route}`. Only this v2 document is current."])
    elif role == "stale":
        warning = (f"폐기된 v1. 현재 값은 `{current_path}`에서 확인한다."
                   if language == "ko" else
                   f"SUPERSEDED v1. Current values live in `{current_path}`.")
        # Repetition is fixed before observation and intentionally puts v1
        # ahead in lexical/BM25 ranking for the route query.
        lines = [f"{route} {route} {route} legacy final configuration", warning]
    else:
        warning = (f"유사 이름의 미승인 초안. `{route}`에 사용하지 않는다."
                   if language == "ko" else
                   f"Near-name unapproved draft. Do not use for `{route}`.")
        lines = [f"{route} final configuration draft", warning]
    if role == "current":
        lines += _fragment_lines(5, values)
    else:
        lines += ["```", *[f"{name} = {value}" for name, value in zip(FIELD_NAMES, values)], "```"]
    return lines


def _ledger_body(route: str, entry_path: str, request_ids: list[str],
                 language: str) -> list[str]:
    """Arm-neutral discovery hop used only by the recalibration profile."""
    if language == "ko":
        header = [
            f"`{route}` 운영 요청의 현재 승인 레지스트리다.",
            "상태가 ACTIVE인 행만 유효하다. RETIRED·DRAFT 행은 무시한다.",
            f"ACTIVE release=v2, 승인 체인 진입점=`{entry_path}`.",
            "체인 문서의 동일 필드 `_part_1`…`_part_5`를 순서대로 결합하고 `E3-`를 붙인다.",
        ]
    else:
        header = [
            f"Current approved request registry for `{route}`.",
            "Only rows marked ACTIVE are valid; ignore RETIRED and DRAFT rows.",
            f"ACTIVE release=v2, approval-chain entry=`{entry_path}`.",
            "Join `_part_1`…`_part_5` for the named field in chain order and prefix `E3-`.",
        ]
    rows = [
        f"- request={request_id}; status=ACTIVE; field={field}; release=v2"
        for request_id, field in zip(request_ids, FIELD_NAMES)
    ]
    # The final formatting distinction is fixed by group/slot, not by an arm
    # outcome. Exactly one task per group is the specificity tier.
    rows += [
        f"- canonical={request_ids[slot]}: {FIELD_NAMES[slot]} = <value>"
        + (" | release = v2" if slot == (int(_sha(route, 'tier')[:2], 16) % TASKS_PER_GROUP)
           else "")
        for slot in range(TASKS_PER_GROUP)
    ]
    return [*header, "", *rows]


def _manifest_hash(value: dict) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def build_fixture(out_root: str | Path, *, seed: int = SEED,
                  embed_revision: str = EMBED_REVISION,
                  profile: str = BASELINE_PROFILE) -> tuple[Corpus, dict, list[dict], dict]:
    """Build the E3 repository, manifest, task pool, and BUILD record."""
    if profile not in (BASELINE_PROFILE, *RECALIBRATION_PROFILES):
        raise ValueError(f"unknown E3 difficulty profile: {profile}")
    out = Path(out_root)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite versioned E3 fixture: {out}")
    repo = out / "repo"
    repo.mkdir(parents=True)

    entries: dict[str, dict] = {}
    tasks: list[dict] = []
    groups: dict[str, list] = {"chains": [], "stale_v1": [], "near_miss": [],
                              "noise": [], "preload_set": []}
    if profile in RECALIBRATION_PROFILES:
        groups["request_ledgers"] = []
    doc_index = 0

    def add(aid: str, path: str, role: str, text: str, language: str,
            relations: dict | None = None, facts: list[dict] | None = None) -> None:
        nonlocal doc_index
        target = repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        raw = text.encode("utf-8")
        size_tok = max(1, math.ceil(len(raw) / (2.5 if language == "ko" else 4.0)))
        entries[aid] = {
            "artifact_id": aid, "path": path, "role": role,
            "language": language, "critical": False, "preload": True,
            "size_tok": size_tok, "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "canary": nonce.canary(seed, aid), "facts": facts or [],
            "relations": relations or {},
        }
        groups["preload_set"].append(aid)
        doc_index += 1

    for group in range(N_GROUPS):
        route = _route(seed, group)
        prefix = f"docs/routes/{group:03d}"
        stale_id = f"e3-g{group:03d}-v1"
        current_id = f"e3-g{group:03d}-v2"
        near_id = f"e3-g{group:03d}-near"
        chain_ids = [f"e3-g{group:03d}-s{i}" for i in range(1, 5)] + [current_id]
        paths = [f"{prefix}/{10 + i * 10:02d}-step-{i + 1}.md" for i in range(4)]
        stale_path = f"{prefix}/00-legacy-v1.md"
        near_path = f"{prefix}/01-near-miss-draft.md"
        ledger_id = f"e3-g{group:03d}-ledger"
        ledger_path = f"{prefix}/05-request-registry.md"
        current_path = f"{prefix}/99-current-v2.md"
        chain_paths = paths + [current_path]
        request_ids = [_request_id(seed, group, slot) for slot in range(TASKS_PER_GROUP)]
        current_values = [_value(seed, group, slot, "v2") for slot in range(TASKS_PER_GROUP)]
        stale_values = [_value(seed, group, slot, "v1") for slot in range(TASKS_PER_GROUP)]
        near_values = [_value(seed, group, slot, "near") for slot in range(TASKS_PER_GROUP)]

        # Add v1 first and give it a lexically earlier path: both directory
        # enumeration and fixed BM25 tie-breaking expose it before v2.
        lang = _artifact_language(doc_index)
        stale_body = _final_body(route, stale_values, lang, role="stale",
                                 current_path=current_path)
        if profile in RECALIBRATION_PROFILES:
            stale_body += ["", "RETIRED request aliases: " + " ".join(request_ids) + "."]
        add(stale_id, stale_path, "superseded_v1",
            _doc_text(seed=seed, aid=stale_id, title=f"{route} legacy v1", language=lang,
                      body=stale_body), lang,
            relations={"superseded_by": current_id},
            facts=[{"key": k, "value": v} for k, v in zip(FIELD_NAMES, stale_values)])
        groups["stale_v1"].append(stale_id)

        lang = _artifact_language(doc_index)
        near_body = _final_body(route, near_values, lang, role="near")
        if profile in RECALIBRATION_PROFILES:
            near_body += ["", "DRAFT request aliases: " + " ".join(request_ids) + "."]
        add(near_id, near_path, "near_miss",
            _doc_text(seed=seed, aid=near_id, title=f"{route} near-match draft", language=lang,
                      body=near_body), lang,
            relations={"near_miss_for": current_id},
            facts=[{"key": k, "value": v} for k, v in zip(FIELD_NAMES, near_values)])
        groups["near_miss"].append(near_id)

        if profile in RECALIBRATION_PROFILES:
            lang = _artifact_language(doc_index)
            add(ledger_id, ledger_path, "request_registry",
                _doc_text(seed=seed, aid=ledger_id, title=f"{route} request registry",
                          language=lang,
                          body=_ledger_body(route, paths[0], request_ids, lang)), lang,
                relations={"entry": chain_ids[0], "release": "v2"},
                facts=[{"key": request_id, "value": field}
                       for request_id, field in zip(request_ids, FIELD_NAMES)])
            groups["request_ledgers"].append(ledger_id)

        for pos, (aid, path) in enumerate(zip(chain_ids[:-1], chain_paths[:-1])):
            lang = _artifact_language(doc_index)
            fragments = _fragments(seed, group, pos + 1)
            body = (_entry_body(route, chain_paths[pos + 1], lang, fragments) if pos == 0
                    else _middle_body(route, pos + 1, chain_paths[pos + 1], lang, fragments))
            add(aid, path, "chain",
                _doc_text(seed=seed, aid=aid, title=f"{route} approval step {pos + 1}",
                          language=lang, body=body), lang,
                relations={"chain": chain_ids, "position": pos + 1,
                           "next": chain_ids[pos + 1]},
                facts=[{"key": f"{k}_part_{pos + 1}", "value": v}
                       for k, v in zip(FIELD_NAMES, fragments)])

        lang = _artifact_language(doc_index)
        add(current_id, current_path, "superseded_v2",
            _doc_text(seed=seed, aid=current_id, title=f"{route} current v2", language=lang,
                      body=_final_body(route, _fragments(seed, group, 5), lang,
                                       role="current")), lang,
            relations={"chain": chain_ids, "position": 5, "supersedes": stale_id},
            facts=[{"key": f"{k}_part_5", "value": v}
                   for k, v in zip(FIELD_NAMES, _fragments(seed, group, 5))])
        groups["chains"].append(chain_ids)

        for slot in range(TASKS_PER_GROUP):
            seq = group * TASKS_PER_GROUP + slot + 1
            task_lang = "ko" if seq % 5 == 0 else "en"
            key = FIELD_NAMES[slot]
            hard_slot = int(_sha(route, "tier")[:2], 16) % TASKS_PER_GROUP
            specificity = slot == hard_slot
            if profile in RECALIBRATION_PROFILES:
                if profile in (RECALIBRATION_PROFILE_V2, RECALIBRATION_PROFILE) and not specificity:
                    prompt = (
                        (f"`{ledger_path}`의 `{route}` 현재 승인 레지스트리에서 운영 요청 "
                         f"`{request_ids[slot]}`을 찾고, 그 레지스트리의 진입점부터 5단계 체인을 "
                         "따라 동일 필드 조각을 순서대로 결합하라. 레지스트리에 지정된 canonical "
                         "형식을 정확히 지켜 한 줄만 답하라.")
                        if task_lang == "ko" else
                        (f"Open `{ledger_path}`, resolve operations request `{request_ids[slot]}` in "
                         f"the current approved `{route}` registry, then follow its entry through the "
                         "five-document chain and join the named field fragments in order. Return "
                         "exactly one line in the canonical format specified by the registry.")
                    )
                else:
                    prompt = (
                        f"운영 요청 `{request_ids[slot]}`을 현재 승인 레지스트리 기준으로 해결하라. "
                        "요청 레지스트리에 지정된 canonical 형식을 정확히 지켜 한 줄만 답하라."
                        if task_lang == "ko" else
                        f"Resolve operations request `{request_ids[slot]}` under the current approved "
                        "registry. Return exactly one line in the canonical format specified by that registry."
                    )
                suffix = r"\s*\|\s*release\s*=\s*v2" if specificity else ""
                if profile == RECALIBRATION_PROFILE and not specificity:
                    expected_pattern = (
                        rf"{re.escape(key)}\s*=\s*{re.escape(current_values[slot])}(?![A-Z0-9])"
                    )
                    stale_pattern = (
                        rf"{re.escape(key)}\s*=\s*{re.escape(stale_values[slot])}(?![A-Z0-9])"
                    )
                else:
                    expected_pattern = (
                        rf"\A\s*{re.escape(key)}\s*=\s*{re.escape(current_values[slot])}"
                        rf"{suffix}\s*\Z"
                    )
                    stale_pattern = (
                        rf"\A\s*{re.escape(key)}\s*=\s*{re.escape(stale_values[slot])}"
                        rf"{suffix}\s*\Z"
                    )
                required = [ledger_id, *chain_ids]
                outcome_rule = f"deterministic-e3-{profile}"
            else:
                prompt = (
                    f"`{paths[0]}`에서 시작해 `{route}`의 5단계 승인 체인을 끝까지 따라가라. "
                    f"각 문서의 `{key}_part_1`부터 `_part_5`까지를 순서대로 붙이고 `E3-`를 앞에 붙여라. "
                    f"정확히 `{key} = <value>` 한 줄로 답하라. "
                    "legacy v1과 미승인 near-match 초안 값은 사용하지 마라."
                    if task_lang == "ko" else
                    f"Start at `{paths[0]}` and follow all five documents in the `{route}` approval "
                    f"chain. Concatenate `{key}_part_1` through `_part_5` in order, prefix the result "
                    f"with `E3-`, and answer with exactly one line: "
                    f"`{key} = <value>`. Do not use the legacy v1 or near-match draft value."
                )
                expected_pattern = (
                    rf"{re.escape(key)}\s*=\s*{re.escape(current_values[slot])}(?![A-Z0-9])"
                )
                stale_pattern = (
                    rf"{re.escape(key)}\s*=\s*{re.escape(stale_values[slot])}(?![A-Z0-9])"
                )
                required = chain_ids
                outcome_rule = "deterministic-e3-v1"
            task = {
                "task_id": f"E3-{seq:04d}", "seq": seq, "tier": "pool",
                "phase": f"G{group:03d}", "scope": route, "language": task_lang,
                "prompt": prompt, "required_artifacts": required,
                "forbidden_artifacts": [stale_id], "forbidden_stale": stale_values[slot],
                "near_miss_artifact": near_id, "near_miss_value": near_values[slot],
                "chain": chain_ids, "max_turns": MAX_TURNS,
                "expected_answer": {
                    "type": "regex",
                    "value": expected_pattern,
                },
                "answer_fact": {"artifact": current_id, "key": key,
                                "value": current_values[slot]},
                "stale_answer": {"type": "regex",
                                 "value": stale_pattern},
                "near_miss_answer": {"type": "regex",
                                     "value": rf"{re.escape(key)}\s*=\s*{re.escape(near_values[slot])}(?![A-Z0-9])"},
                "outcome_rule": outcome_rule,
            }
            if profile in RECALIBRATION_PROFILES:
                task.update({
                    "difficulty_profile": profile,
                    "difficulty_tier": "specificity" if specificity else "standard",
                    "request_id": request_ids[slot],
                    "evidence_chain": [ledger_id, *chain_ids],
                })
            tasks.append(task)

    for i in range(N_NOISE):
        aid = f"e3-noise-{i:03d}"
        path = f"docs/reference/noise-{i:03d}.md"
        lang = _artifact_language(doc_index)
        body = (["이 문서는 일반적인 배경 정보이며 승인 체인의 일부가 아니다."]
                if lang == "ko" else
                ["This is general background material and is not part of an approval chain."])
        add(aid, path, "noise",
            _doc_text(seed=seed, aid=aid, title=f"background note {i:03d}",
                      language=lang, body=body), lang)
        groups["noise"].append(aid)

    corpus_payload = "\n".join(
        f"{entry['path']}\0{entry['sha256']}" for entry in sorted(entries.values(),
                                                                  key=lambda e: e["path"])
    )
    corpus_hash = hashlib.sha256(corpus_payload.encode("utf-8")).hexdigest()
    manifest = {
        "schema_version": 1, "experiment": "E3-comparative", "seed": seed,
        "corpus_content_hash": corpus_hash, "n_artifacts": len(entries),
        "artifacts": entries,
        "fixture_contract": {
            "n_tasks": N_TASKS, "n_groups": N_GROUPS, "chain_length": 5,
            "language_target": {"en": 0.8, "ko": 0.2}, "max_turns": MAX_TURNS,
            "stale_rank_rule": "v1 path lexically before v2; v1 route term repeated 3x",
        },
        "embedder": {"model_id": EMBED_MODEL_ID, "revision": embed_revision},
    }
    if profile in RECALIBRATION_PROFILES:
        manifest["fixture_contract"].update({
            "difficulty_profile": profile,
            "calibration_role": "dev" if seed == DEV_SEED else "official",
            "discovery_hops": 1,
            "required_artifact_count": 6,
        })
    manifest["manifest_sha256"] = _manifest_hash(manifest)
    tasks_hash = hashlib.sha256(json.dumps(
        tasks, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()

    corpus = Corpus(seed=seed)
    corpus.groups = groups
    for aid, entry in entries.items():
        corpus.artifacts[aid] = ArtifactMeta(
            artifact_id=aid, size_tok=entry["size_tok"], path=entry["path"],
            preload=True, language=entry["language"], tags=(entry["role"],),
            byte_size=entry["bytes"],
        )

    build = {
        "seed": seed, "n_artifacts": len(entries), "n_tasks": len(tasks),
        "n_english_tasks": sum(t["language"] == "en" for t in tasks),
        "n_korean_tasks": sum(t["language"] == "ko" for t in tasks),
        "total_tokens": corpus.total_tokens,
        "corpus_content_hash": corpus_hash,
        "manifest_sha256": manifest["manifest_sha256"], "tasks_sha256": tasks_hash,
        "embedder": manifest["embedder"],
        "status": "draft — freeze only after G-E3a passes",
    }
    if profile in RECALIBRATION_PROFILES:
        build["difficulty_profile"] = profile
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out / "tasks.json").write_text(
        json.dumps(tasks, indent=1, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    (out / "groups.json").write_text(
        json.dumps(groups, indent=1, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out / "BUILD.json").write_text(
        json.dumps(build, indent=1, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    return corpus, manifest, tasks, build


def corpus_from_manifest(manifest: dict, groups: dict) -> Corpus:
    """Reconstruct policy metadata from a frozen E3 manifest."""
    corpus = Corpus(seed=int(manifest["seed"]), groups={k: list(v) for k, v in groups.items()})
    for aid, entry in manifest["artifacts"].items():
        corpus.artifacts[aid] = ArtifactMeta(
            artifact_id=aid, size_tok=int(entry["size_tok"]), path=entry["path"],
            preload=bool(entry.get("preload")), language=entry.get("language", "en"),
            tags=(entry.get("role", "unknown"),), byte_size=int(entry["bytes"]),
        )
    return corpus
