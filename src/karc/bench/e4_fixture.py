"""Deterministic E4 controlled fixture generator (seed 4000).

The corpus is fixed across the preregistered rho x sigma x budget grid.  Only
the task/event schedule and the token budget vary.  The generator records the
measured locality/stale structure; requested values are labels, not asserted
outcomes.  No model is called here.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from karc.policy.model import ArtifactMeta
from karc.replay.corpus import Corpus

SEED = 4000
N_TASKS = 80
N_STABLE_DOCS = 112
N_VERSIONED_DOCS = 12
V2_N_TASKS = 640
V2_N_CYCLES = 40
V2_N_STABLE_DOCS = 232
V2_N_VERSIONED_DOCS = 88
MAX_TURNS = 8
HOT_WINDOW_TASKS = 3
RHO_GRID = (0.0, 0.25, 0.5, 0.75)
SIGMA_GRID = (0.0, 0.15, 0.3)
BUDGET_GRID = (1, 2, 5, 10, 20)
PRIMARY_CELL = (0.5, 0.3, 5)
EMBED_MODEL_ID = "BAAI/bge-m3"
EMBED_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"


def _sha(*parts: object) -> str:
    return hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()


def _value(seed: int, doc_index: int, version: int) -> str:
    raw = _sha(seed, doc_index, version, "value").upper()
    return f"E4-{raw[:4]}-{raw[4:8]}"


def _canary(seed: int, version_id: str) -> str:
    raw = _sha(seed, version_id, "canary")
    return f"e4c-{raw[:8]}-{raw[8:12]}-{raw[12:16]}-{raw[16:28]}"


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode()


def _hash_json(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _validate_grid(rho: float, sigma: float, budget_pct: int) -> None:
    if rho not in RHO_GRID:
        raise ValueError(f"rho must be one of {RHO_GRID}")
    if sigma not in SIGMA_GRID:
        raise ValueError(f"sigma must be one of {SIGMA_GRID}")
    if budget_pct not in BUDGET_GRID:
        raise ValueError(f"budget_pct must be one of {BUDGET_GRID}")


def _doc_text(seed: int, doc_id: str, version_id: str, fact_key: str,
              value: str, *, current: bool, supersedes: str | None) -> str:
    # v1 intentionally does not advertise that it will be superseded.  The
    # relationship lives in metadata and on v2 only (§3.3 confusable trap).
    title = f"Operations registry entry {doc_id}"
    lines = [
        f"# {title}", "", f"Record key: `{fact_key}`.",
        f"Current operational value: `{value}`.",
        "Use this value when a request asks for the current registry entry.",
        f"Reference tag: `{_canary(seed, version_id)}`.",
        "", "The value has the canonical E4-XXXX-XXXX shape and was generated "
        "for the controlled benchmark.",
    ]
    if current and supersedes:
        lines += ["", f"Version metadata declares that this release supersedes `{supersedes}`."]
    return "\n".join(lines) + "\n"


def _corpus(seed: int, *, n_stable: int = N_STABLE_DOCS,
            n_versioned: int = N_VERSIONED_DOCS) -> tuple[Corpus, dict[str, dict], dict[str, list]]:
    corpus = Corpus(seed=seed)
    entries: dict[str, dict] = {}
    groups: dict[str, list] = {
        "stable": [], "versioned_v1": [], "versioned_v2": [],
        "preload_set": [], "correction_targets": [],
        "supersession_candidates": [], "ghost_cycle": [],
    }

    def add(doc_index: int, version: int, *, versioned: bool) -> str:
        doc_id = f"e4-doc-{doc_index:03d}"
        version_id = f"{doc_id}-v{version}"
        fact_key = f"registry_{doc_index:03d}"
        value = _value(seed, doc_index, version)
        current = not versioned or version == 2
        v1 = f"{doc_id}-v1" if version == 2 else None
        text = _doc_text(seed, doc_id, version_id, fact_key, value,
                         current=current, supersedes=v1)
        raw = text.encode()
        # Stable near-equal sizes make a percentage budget an interpretable
        # resident-document budget without changing the token policy.
        size_tok = max(1, math.ceil(len(raw) / 4.0))
        path = (
            f"docs/registry/{doc_index:03d}/00-release-v1.md"
            if versioned and version == 1 else
            f"docs/registry/{doc_index:03d}/99-release-v2.md"
            if versioned else f"docs/registry/{doc_index:03d}/50-entry.md"
        )
        entry = {
            "artifact_id": doc_id,
            "version_id": version_id,
            "path": path,
            "fact_key": fact_key,
            "value": value,
            "version_number": version,
            # The concrete validity timeline is cell-specific and is filled
            # after the task schedule is built.
            "is_current_at_end": current,
            "valid_from_seq": 1 if version == 1 else None,
            "superseded_at_seq": None,
            "supersedes_version_id": v1,
            "superseded_by_version_id": (
                f"{doc_id}-v2" if versioned and version == 1 else None
            ),
            "size_tok": size_tok,
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "canary": _canary(seed, version_id),
            "text": text,
        }
        entries[version_id] = entry
        corpus.artifacts[version_id] = ArtifactMeta(
            artifact_id=version_id, size_tok=size_tok, path=path,
            preload=True, tags=("versioned" if versioned else "stable",),
            byte_size=len(raw),
        )
        groups["preload_set"].append(version_id)
        return version_id

    for i in range(n_stable):
        groups["stable"].append(add(i, 1, versioned=False))
    for i in range(n_stable, n_stable + n_versioned):
        groups["versioned_v1"].append(add(i, 1, versioned=True))
        groups["versioned_v2"].append(add(i, 2, versioned=True))
    groups["correction_targets"] = groups["stable"][:3]
    groups["supersession_candidates"] = groups["versioned_v1"][:6]
    groups["ghost_cycle"] = groups["stable"][10:18]
    corpus.groups = groups
    return corpus, entries, groups


def _pre_event(kind: str, version_id: str, *, new_validity: str | None = None) -> dict:
    row = {"event_type": kind, "version_id": version_id, "origin": "script"}
    if new_validity is not None:
        row["new_validity"] = new_validity
    return row


def _task_row(seq: int, required: str, entries: dict[str, dict], *,
              pre_events: Iterable[dict] = (), forbidden: Iterable[str] = (),
              outcome_inject: Iterable[str] = (),
              structure: str = "fill", task_type: str | None = None,
              outcome_targets: Iterable[str] = ()) -> dict:
    entry = entries[required]
    stale_values = [entries[v]["value"] for v in forbidden]
    return {
        "task_id": f"E4-{seq:04d}",
        "seq": seq,
        "session_id": f"S{(seq - 1) // 8:03d}",
        "language": "en",
        "difficulty_tier": "controlled",
        "structure": structure,
        "task_type": task_type or structure,
        "required_artifacts": [entry["artifact_id"]],
        "required_versions": [required],
        "forbidden_versions": list(forbidden),
        "pre_events": list(pre_events),
        # These labels are emitted by the deterministic outcome-injector
        # after the task.  They are never part of the scripted pre-event log.
        "outcome_inject": list(outcome_inject),
        "outcome_targets": list(outcome_targets),
        "max_turns": MAX_TURNS,
        "prompt": (
            f"Return the current value for `{entry['fact_key']}` as exactly "
            f"`{entry['fact_key']} = <value>`."
        ),
        "answer_fact": {
            "artifact_id": entry["artifact_id"], "version_id": required,
            "key": entry["fact_key"], "value": entry["value"],
        },
        "stale_values": stale_values,
        # Format is a structural secondary metric.  It deliberately accepts
        # any syntactically valid E4 value, including a stale one, so value
        # correctness and formatting remain independent (§3.5-3, A-B3).
        "format_regex": rf"\A\s*{re.escape(entry['fact_key'])}\s*=\s*"
                        r"E4-[0-9A-F]{4}-[0-9A-F]{4}\s*\Z",
    }


def _schedule(seed: int, rho: float, sigma: float, entries: dict[str, dict],
              groups: dict[str, list]) -> tuple[list[dict], dict]:
    if rho == 0.0:
        tasks = [_task_row(i + 1, groups["stable"][20 + i], entries,
                           structure="rho-zero-anchor") for i in range(N_TASKS)]
        return tasks, _structure_stats(tasks, sigma, 0, HOT_WINDOW_TASKS)

    tasks: list[dict] = []

    def add(required: str, *, pre=(), forbidden=(), outcome=(), structure="core") -> None:
        tasks.append(_task_row(len(tasks) + 1, required, entries,
                               pre_events=pre, forbidden=forbidden,
                               outcome_inject=outcome,
                               structure=structure))

    rehab, harmful, anchor = groups["correction_targets"]
    add(rehab, structure="rehab-prime")
    add(rehab, structure="rehab-promote")
    add(harmful, structure="harmful-prime")
    add(harmful, structure="harmful-promote")
    add(anchor, structure="anchor-prime")
    add(anchor, structure="anchor-promote")
    add(anchor, pre=[_pre_event("corrected", rehab)], structure="rehab-correct")
    add(rehab, outcome=("validated",), structure="rehab-validated-by-injector")
    add(rehab, structure="rehab-repromote")
    add(rehab, pre=[_pre_event("corrected", harmful)], structure="harmful-correct-1")
    add(rehab, pre=[_pre_event("corrected", harmful)], forbidden=[harmful],
        structure="harmful-correct-2-forbidden")

    # The denominator is fixed before selecting supersessions.  Fill reuse
    # only revisits these already-reused IDs, so this remains the denominator.
    reuse_population = 17 if rho >= 0.5 else 9
    n_superseded = min(6, int(round(sigma * reuse_population)))
    for index, v1 in enumerate(groups["supersession_candidates"]):
        add(v1, structure="supersession-prime")
        add(v1, structure="supersession-promote")
        if index < n_superseded:
            v2 = entries[v1]["superseded_by_version_id"]
            add(v2, pre=[_pre_event("validity_changed", v1,
                                    new_validity="STALE")], forbidden=[v1],
                structure="hot-supersession-trap")

    if rho >= 0.5:
        for version_id in groups["ghost_cycle"]:
            add(version_id, structure="ghost-prime")
            add(version_id, structure="ghost-promote")
        add(groups["ghost_cycle"][0], structure="ghost-b2-revisit")
        add(groups["ghost_cycle"][1], structure="ghost-b2-revisit")
        # Harmful is in B1 for K-ARC; this is an explicit B1 ghost hit.  It is
        # last among core structures so later answer exposure is not claimed.
        add(harmful, structure="ghost-b1-revisit")

    target_reuse = int(round(rho * N_TASKS))
    seen_docs: set[str] = set()
    current_reuse = 0
    for task in tasks:
        doc_id = task["required_artifacts"][0]
        current_reuse += int(doc_id in seen_docs)
        seen_docs.add(doc_id)
    if current_reuse > target_reuse:
        raise AssertionError("scripted E4 core exceeds requested locality")

    # Fill only from artifacts that the core has actually referenced.  Keep
    # the most recently required version so every selected fill is a real
    # distinct-task reuse occurrence (including activated v2 versions).
    reusable_by_doc: dict[str, str] = {}
    for task in tasks:
        reusable_by_doc[task["required_artifacts"][0]] = task["required_versions"][0]
    reusable = list(reusable_by_doc.values())
    new_cursor = 40
    while len(tasks) < N_TASKS:
        need = target_reuse - current_reuse
        remaining = N_TASKS - len(tasks)
        if need > 0 and (need >= remaining or (len(tasks) + seed) % 3 != 0):
            version_id = reusable[(len(tasks) + seed) % len(reusable)]
            add(version_id, structure="rho-fill-reuse")
            current_reuse += 1
        else:
            version_id = groups["stable"][new_cursor]
            new_cursor += 1
            add(version_id, structure="rho-fill-new")
    if current_reuse != target_reuse:
        raise AssertionError((current_reuse, target_reuse))
    return tasks, _structure_stats(tasks, sigma, n_superseded, HOT_WINDOW_TASKS)


def _schedule_v2(seed: int, entries: dict[str, dict],
                 groups: dict[str, list]) -> tuple[list[dict], dict]:
    """Build the A-B3 balanced 640-task primary-cell pool.

    Forty identical-shape, disjoint-document cycles prevent any one curation
    type from dominating the pool.  Each cycle has exactly eight cross-task
    reuse occurrences.  Every cycle contains one ordinary hot supersession;
    eight cycles contain a second one, giving measured rho=.5 and sigma=.3.
    Harmful replacement is driven only by corrected outcome events (not the
    validity event), preserving the karc/karc-no-outcome attribution path.
    """
    stable = iter(groups["stable"])
    versioned = iter(zip(groups["versioned_v1"], groups["versioned_v2"]))
    tasks: list[dict] = []
    trap_candidates: list[str] = []
    harmful_targets: list[str] = []
    rehabilitation_targets: list[str] = []
    correction_targets: list[str] = []

    def add(required: str, *, pre=(), forbidden=(), outcome=(),
            structure: str, task_type: str, targets=()) -> None:
        tasks.append(_task_row(
            len(tasks) + 1, required, entries, pre_events=pre,
            forbidden=forbidden, outcome_inject=outcome,
            structure=structure, task_type=task_type,
            outcome_targets=targets,
        ))

    for cycle in range(V2_N_CYCLES):
        extra_supersession = cycle % 5 == 0

        # Corrected-on-T2 structure.  Eight cycles also carry the additional
        # hot supersession needed for sigma=.3; the ordinary cycles use a new
        # anchor in the fourth slot so locality remains exactly rho=.5.
        if extra_supersession:
            corrected, corrected_v2 = next(versioned)
            trap_candidates.append(corrected)
        else:
            corrected, corrected_v2 = next(stable), None
        corrected_anchor = next(stable)
        correction_targets.append(corrected)
        add(corrected, structure="corrected-prime", task_type="corrected",
            targets=(corrected,))
        add(corrected, structure="corrected-promote", task_type="corrected",
            targets=(corrected,))
        add(corrected_anchor, pre=(_pre_event("corrected", corrected),),
            structure="corrected-demote", task_type="corrected",
            targets=(corrected,))
        if extra_supersession:
            add(corrected_v2,
                pre=(_pre_event("validity_changed", corrected,
                                new_validity="STALE"),),
                forbidden=(corrected,), structure="corrected-hot-supersession",
                task_type="corrected", targets=(corrected,))
        else:
            add(next(stable), structure="corrected-post-demotion",
                task_type="corrected", targets=(corrected,))

        # Two corrections make the resident v1 harmful for K-ARC.  The
        # ablation ignores both corrections, so the v1 supply can persist when
        # the same fact's replacement v2 becomes the required answer.
        harmful, harmful_v2 = next(versioned)
        harmful_anchor = next(stable)
        harmful_targets.append(harmful)
        add(harmful, structure="harmful-prime", task_type="harmful",
            targets=(harmful,))
        add(harmful, structure="harmful-promote", task_type="harmful",
            targets=(harmful,))
        add(harmful_anchor, pre=(_pre_event("corrected", harmful),),
            structure="harmful-correct-1", task_type="harmful",
            targets=(harmful,))
        add(harmful_v2, pre=(_pre_event("corrected", harmful),),
            forbidden=(harmful,), structure="harmful-current-forbidden",
            task_type="harmful", targets=(harmful,))

        # Rehabilitation has one correction followed by a validated signal
        # emitted only by the outcome injector.  Normal cycles revisit the
        # target after validation; extra-supersession cycles use a new anchor
        # here to offset their additional corrected-v2 reuse occurrence.
        rehab = next(stable)
        rehab_anchor = next(stable)
        rehabilitation_targets.append(rehab)
        add(rehab, structure="rehab-prime", task_type="rehabilitation",
            targets=(rehab,))
        add(rehab, structure="rehab-promote", task_type="rehabilitation",
            targets=(rehab,))
        add(rehab_anchor, pre=(_pre_event("corrected", rehab),),
            structure="rehab-correct", task_type="rehabilitation",
            targets=(rehab,))
        add(rehab, outcome=("validated",),
            structure="rehab-validated-by-injector",
            task_type="rehabilitation", targets=(rehab,))
        add(next(stable) if extra_supersession else rehab,
            structure=("rehab-balance-anchor" if extra_supersession
                       else "rehab-post-validated"),
            task_type="rehabilitation", targets=(rehab,))

        # Ordinary confusable stale trap.  v1 is hot/T2 before the metadata
        # validity transition, and v2 is the required current value.
        stale, stale_v2 = next(versioned)
        trap_candidates.append(stale)
        add(stale, structure="plain-stale-prime", task_type="plain-stale")
        add(stale, structure="plain-stale-promote", task_type="plain-stale")
        add(stale_v2,
            pre=(_pre_event("validity_changed", stale,
                            new_validity="STALE"),),
            forbidden=(stale,), structure="hot-supersession-trap",
            task_type="plain-stale")

    if len(tasks) != V2_N_TASKS:
        raise AssertionError((len(tasks), V2_N_TASKS))
    try:
        next(stable)
        raise AssertionError("unused v2 stable artifacts")
    except StopIteration:
        pass
    try:
        next(versioned)
        raise AssertionError("unused v2 versioned artifacts")
    except StopIteration:
        pass

    groups["trap_candidates"] = trap_candidates
    groups["correction_targets"] = correction_targets
    groups["harmful_targets"] = harmful_targets
    groups["rehabilitation_targets"] = rehabilitation_targets
    structure = _structure_stats(
        tasks, PRIMARY_CELL[1], len(trap_candidates), HOT_WINDOW_TASKS,
        sigma_denominator=V2_N_CYCLES * 4,
    )
    structure.update({
        "profile": "balanced-pool-v2",
        "cycles": V2_N_CYCLES,
        "unique_trap_candidates": len(set(trap_candidates)),
        "harmful_target_docs": len(set(harmful_targets)),
        "rehabilitation_target_docs": len(set(rehabilitation_targets)),
        "format_rule": "fact-key = E4-XXXX-XXXX; expected value not embedded",
    })
    return tasks, structure


def _structure_stats(tasks: list[dict], sigma_requested: float,
                     superseded: int, hot_window: int, *,
                     sigma_denominator: int | None = None) -> dict:
    seen: set[str] = set()
    reused_occurrences = 0
    reused_docs: set[str] = set()
    distinct_refs: dict[str, set[str]] = {}
    for task in tasks:
        for doc_id in task["required_artifacts"]:
            if doc_id in seen:
                reused_occurrences += 1
                reused_docs.add(doc_id)
            seen.add(doc_id)
            distinct_refs.setdefault(doc_id, set()).add(task["task_id"])
    eligible = sigma_denominator
    if eligible is None:
        eligible = 17 if any(t["structure"].startswith("ghost") for t in tasks) else 9
        if not reused_docs:
            eligible = 0
    type_counts: dict[str, int] = {}
    for task in tasks:
        kind = str(task.get("task_type", task.get("structure", "unknown")))
        type_counts[kind] = type_counts.get(kind, 0) + 1
    return {
        "tasks": len(tasks),
        "required_occurrences": sum(len(t["required_artifacts"]) for t in tasks),
        "distinct_task_references": sum(len(v) for v in distinct_refs.values()),
        "reused_occurrences": reused_occurrences,
        "reused_doc_ids": len(reused_docs),
        "rho_measured": reused_occurrences / len(tasks) if tasks else 0.0,
        "sigma_requested": sigma_requested,
        "sigma_denominator_reused_doc_ids": eligible,
        "supersession_events": superseded,
        "sigma_measured": superseded / eligible if eligible else 0.0,
        "hot_window_tasks": hot_window,
        "corrected_events": sum(
            e["event_type"] == "corrected" for t in tasks for e in t["pre_events"]
        ),
        "validated_outcome_events": sum(
            event == "validated" for task in tasks
            for event in task.get("outcome_inject", [])
        ),
        "task_type_counts": dict(sorted(type_counts.items())),
        "max_task_type_share": (max(type_counts.values()) / len(tasks)
                                if tasks and type_counts else 0.0),
    }


def generate_fixture(*, seed: int = SEED, rho: float = PRIMARY_CELL[0],
                     sigma: float = PRIMARY_CELL[1],
                     budget_pct: int = PRIMARY_CELL[2]) -> tuple[Corpus, dict, list[dict], dict]:
    _validate_grid(rho, sigma, budget_pct)
    corpus, entries_with_text, groups = _corpus(seed)
    tasks, structure = _schedule(seed, rho, sigma, entries_with_text, groups)
    # Version state comes from explicit metadata/events, never filenames.
    # A v2 not activated in this cell remains future metadata and its v1
    # remains current for the whole replay.
    activated_v2: set[str] = set()
    for task in tasks:
        for event in task["pre_events"]:
            if event["event_type"] != "validity_changed":
                continue
            v1 = event["version_id"]
            v2 = entries_with_text[v1]["superseded_by_version_id"]
            activated_v2.add(v2)
            entries_with_text[v1]["is_current_at_end"] = False
            entries_with_text[v1]["superseded_at_seq"] = task["seq"]
            entries_with_text[v2]["valid_from_seq"] = task["seq"]
    for v1 in groups["versioned_v1"]:
        v2 = entries_with_text[v1]["superseded_by_version_id"]
        if v2 not in activated_v2:
            entries_with_text[v1]["is_current_at_end"] = True
            entries_with_text[v2]["is_current_at_end"] = False
    entries = {key: {k: v for k, v in row.items() if k != "text"}
               for key, row in entries_with_text.items()}
    corpus_hash = hashlib.sha256("\n".join(
        f"{row['path']}\0{row['sha256']}" for row in
        sorted(entries.values(), key=lambda item: item["path"])
    ).encode()).hexdigest()
    budget_tokens = max(1, int(corpus.total_tokens * budget_pct / 100.0))
    manifest = {
        "schema_version": 1,
        "experiment": "E4-common",
        "seed": seed,
        "cell": {"rho": rho, "sigma": sigma, "budget_pct": budget_pct,
                 "budget_tokens": budget_tokens},
        "corpus_content_hash": corpus_hash,
        "n_artifacts": len(entries),
        "artifacts": entries,
        "groups": groups,
        "structure": structure,
        "fixture_contract": {
            "status": "draft-candidate-not-frozen",
            "n_tasks": N_TASKS,
            "max_turns": MAX_TURNS,
            "locality_unit": "stable artifact_id across version_id",
            "version_metadata_not_filename": True,
            "validated_origin": "outcome-injector-only",
            "oracle_tier_guard_scaffold": {"tier": "controlled", "minimum": 0.70},
        },
        "embedder": {"model_id": EMBED_MODEL_ID, "revision": EMBED_REVISION},
    }
    manifest["manifest_sha256"] = _hash_json(manifest)
    build = {
        "seed": seed,
        "cell": manifest["cell"],
        "n_artifacts": len(entries),
        "n_tasks": len(tasks),
        "total_tokens": corpus.total_tokens,
        "corpus_content_hash": corpus_hash,
        "manifest_sha256": manifest["manifest_sha256"],
        "tasks_sha256": _hash_json(tasks),
        "structure": structure,
        "status": "draft candidate; freeze prohibited until canary",
    }
    return corpus, manifest, tasks, build


def generate_fixture_v2(*, seed: int = SEED,
                        rho: float = PRIMARY_CELL[0],
                        sigma: float = PRIMARY_CELL[1],
                        budget_pct: int = PRIMARY_CELL[2]) -> tuple[Corpus, dict, list[dict], dict]:
    """Generate the A-B3 balanced E4b expansion candidate.

    The expansion is deliberately primary-cell-only: the sequential grid and
    E4a transect remain out of scope until the re-smoke checkpoint is reviewed.
    """
    _validate_grid(rho, sigma, budget_pct)
    if (rho, sigma, budget_pct) != PRIMARY_CELL:
        raise ValueError("e4-v2 pool expansion is frozen to the E4b primary cell")
    corpus, entries_with_text, groups = _corpus(
        seed, n_stable=V2_N_STABLE_DOCS, n_versioned=V2_N_VERSIONED_DOCS,
    )
    tasks, structure = _schedule_v2(seed, entries_with_text, groups)

    # Resolve both validity-driven and outcome-driven replacements in the
    # registry metadata.  Only the former emits validity_changed to policies;
    # the latter is intentionally attributable to corrected outcome signals.
    activated: dict[str, tuple[int, str]] = {}
    for task in tasks:
        for event in task["pre_events"]:
            if event["event_type"] != "validity_changed":
                continue
            v1 = event["version_id"]
            activated[entries_with_text[v1]["superseded_by_version_id"]] = (
                int(task["seq"]), "validity",
            )
        if task["structure"] == "harmful-current-forbidden":
            v2 = task["required_versions"][0]
            v1 = task["forbidden_versions"][0]
            if entries_with_text[v1]["superseded_by_version_id"] != v2:
                raise AssertionError("harmful replacement must be a version pair")
            activated[v2] = (int(task["seq"]), "outcome-corrected")
    for v1 in groups["versioned_v1"]:
        v2 = entries_with_text[v1]["superseded_by_version_id"]
        if v2 in activated:
            seq, channel = activated[v2]
            entries_with_text[v1]["is_current_at_end"] = False
            entries_with_text[v1]["superseded_at_seq"] = seq
            entries_with_text[v1]["replacement_channel"] = channel
            entries_with_text[v2]["valid_from_seq"] = seq
            entries_with_text[v2]["replacement_channel"] = channel
        else:
            entries_with_text[v1]["is_current_at_end"] = True
            entries_with_text[v2]["is_current_at_end"] = False

    entries = {key: {k: v for k, v in row.items() if k != "text"}
               for key, row in entries_with_text.items()}
    corpus_hash = hashlib.sha256("\n".join(
        f"{row['path']}\0{row['sha256']}" for row in
        sorted(entries.values(), key=lambda item: item["path"])
    ).encode()).hexdigest()
    budget_tokens = max(1, int(corpus.total_tokens * budget_pct / 100.0))
    manifest = {
        "schema_version": 2,
        "experiment": "E4-common",
        "fixture_version": "e4-draft-v2",
        "generator_profile": "balanced-pool-v2",
        "seed": seed,
        "cell": {"rho": rho, "sigma": sigma, "budget_pct": budget_pct,
                 "budget_tokens": budget_tokens},
        "corpus_content_hash": corpus_hash,
        "n_artifacts": len(entries),
        "artifacts": entries,
        "groups": groups,
        "structure": structure,
        "fixture_contract": {
            "status": "draft-candidate-not-frozen",
            "fixture_version": "e4-draft-v2",
            "n_tasks": V2_N_TASKS,
            "max_turns": MAX_TURNS,
            "locality_unit": "stable artifact_id across version_id",
            "version_metadata_not_filename": True,
            "validated_origin": "outcome-injector-only",
            "task_type_balance": "single type <= 40%",
            "unique_trap_minimum": 20,
            "grader": {
                "primary": "value_match == expected",
                "format_secondary": True,
                "format_expected_value_embedded": False,
            },
            "oracle_tier_guard_scaffold": {
                "tier": "controlled", "minimum": 0.70,
            },
        },
        "embedder": {"model_id": EMBED_MODEL_ID, "revision": EMBED_REVISION},
    }
    manifest["manifest_sha256"] = _hash_json(manifest)
    build = {
        "seed": seed,
        "generator_profile": "balanced-pool-v2",
        "cell": manifest["cell"],
        "n_artifacts": len(entries),
        "n_tasks": len(tasks),
        "total_tokens": corpus.total_tokens,
        "corpus_content_hash": corpus_hash,
        "manifest_sha256": manifest["manifest_sha256"],
        "tasks_sha256": _hash_json(tasks),
        "structure": structure,
        "status": "draft candidate; freeze prohibited until canary and trap probe",
    }
    return corpus, manifest, tasks, build


def _write_fixture(out: Path, corpus: Corpus, manifest: dict,
                   tasks: list[dict], build: dict) -> None:
    profile = manifest.get("generator_profile", "draft-v1")
    if profile == "balanced-pool-v2":
        _unused, entries_with_text, _groups = _corpus(
            int(manifest["seed"]), n_stable=V2_N_STABLE_DOCS,
            n_versioned=V2_N_VERSIONED_DOCS,
        )
    else:
        _unused, entries_with_text, _groups = _corpus(int(manifest["seed"]))
    repo = out / "repo"
    repo.mkdir(parents=True)
    for version_id, entry in entries_with_text.items():
        target = repo / entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(entry["text"], encoding="utf-8")
        if hashlib.sha256(target.read_bytes()).hexdigest() != manifest["artifacts"][version_id]["sha256"]:
            raise AssertionError(version_id)
    for name, value in (("manifest.json", manifest), ("tasks.json", tasks),
                        ("groups.json", manifest["groups"]), ("BUILD.json", build)):
        (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=1,
                                            sort_keys=True) + "\n", encoding="utf-8")


def build_fixture(out_root: str | Path, **kwargs):
    out = Path(out_root)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite E4 candidate fixture: {out}")
    corpus, manifest, tasks, build = generate_fixture(**kwargs)
    _write_fixture(out, corpus, manifest, tasks, build)
    return corpus, manifest, tasks, build


def build_fixture_v2(out_root: str | Path, **kwargs):
    out = Path(out_root)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite E4 candidate fixture: {out}")
    corpus, manifest, tasks, build = generate_fixture_v2(**kwargs)
    _write_fixture(out, corpus, manifest, tasks, build)
    return corpus, manifest, tasks, build


@dataclass(frozen=True)
class E4Grade:
    value_match: str
    format_ok: bool
    abstained: bool

    @property
    def success(self) -> bool:
        return self.value_match == "expected"


def grade_answer(task: dict, output_text: str) -> E4Grade:
    """Value and format are intentionally independent (§3.5)."""
    text = (output_text or "").strip()
    abstained = not text or text.upper() in {"ABSTAIN", "UNKNOWN", "N/A"}
    expected = str(task["answer_fact"]["value"])
    if expected in text:
        value_match = "expected"
    elif any(str(value) in text for value in task.get("stale_values", [])):
        value_match = "stale"
    else:
        value_match = "other"
    return E4Grade(value_match=value_match,
                   format_ok=bool(re.fullmatch(task["format_regex"], text)),
                   abstained=abstained)
