"""Pure-Python E4 working-set replay over the preregistered 60-cell grid."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

from karc.bench.e4_fixture import generate_fixture
from karc.bench.e4_instrument import exposure_set, refs_for_versions
from karc.policy import PolicyConfig, make_policy
from karc.policy.model import ArtifactMeta, Event
from karc.replay.corpus import Corpus


WORKING_SET_ARMS = ("fifo", "lru", "classic", "karc", "karc-no-outcome")


def _ts(seq: int, offset: int) -> str:
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return (base + timedelta(minutes=seq * 10, seconds=offset)).isoformat().replace("+00:00", "Z")


def _event(task: dict, payload: dict, event_index: int, *, origin: str | None = None) -> Event:
    return Event(
        event_id=f"e4.{task['task_id']}.{event_index:03d}.{payload['event_type']}",
        event_type=payload["event_type"], occurred_at=_ts(int(task["seq"]), event_index),
        artifact_id=payload["version_id"], task_id=task["task_id"],
        session_id=task["session_id"], scope_id="E4-controlled",
        new_validity=payload.get("new_validity"),
        origin=origin or payload.get("origin", "script"),
    )


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _transition_counts(policy) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for transition in policy.transitions:
        src, dst = transition.from_list, transition.to_list
        if src == "T1" and dst == "T2":
            counts["t1_to_t2"] += 1
        if src in {"T1", "T2"} and dst in {"B1", "B2"}:
            counts["resident_to_b"] += 1
        if src == "B1" and dst in {"T1", "T2"}:
            counts["b1_revival"] += 1
        if src == "B2" and dst in {"T1", "T2"}:
            counts["b2_revival"] += 1
        if src == "T2" and dst == "T1" and "corrected" in transition.rule_id:
            counts["corrected_t2_to_t1"] += 1
    return {key: counts.get(key, 0) for key in (
        "t1_to_t2", "resident_to_b", "b1_revival", "b2_revival",
        "corrected_t2_to_t1",
    )}


def load_confirmed_config(repo_root: str | Path) -> dict:
    path = Path(repo_root) / "docs/experiments/E1-2-parameter-suite/confirmed-config.json"
    return json.loads(path.read_text(encoding="utf-8"))["config"]


def replay_cell(*, rho: float, sigma: float, budget_pct: int,
                confirmed_config: dict, fixture_manifest: dict | None = None,
                fixture_tasks: list[dict] | None = None) -> tuple[dict, list[dict]]:
    if (fixture_manifest is None) != (fixture_tasks is None):
        raise ValueError("fixture_manifest and fixture_tasks must be supplied together")
    if fixture_manifest is None:
        corpus, manifest, tasks, build = generate_fixture(
            rho=rho, sigma=sigma, budget_pct=budget_pct,
        )
    else:
        manifest = fixture_manifest
        tasks = fixture_tasks or []
        cell = manifest["cell"]
        if (float(cell["rho"]), float(cell["sigma"]), int(cell["budget_pct"])) != (
                rho, sigma, budget_pct):
            raise ValueError("explicit fixture cell differs from replay request")
        corpus = Corpus(seed=int(manifest["seed"]))
        for version_id, entry in manifest["artifacts"].items():
            corpus.artifacts[version_id] = ArtifactMeta(
                artifact_id=version_id, size_tok=int(entry["size_tok"]),
                path=entry["path"], preload=True,
                critical=bool(entry.get("critical", False)),
                pinned=bool(entry.get("pinned", False)),
                tags=("versioned" if entry.get("superseded_by_version_id")
                      or entry.get("supersedes_version_id") else "stable",),
                byte_size=int(entry["bytes"]),
            )
        corpus.groups = manifest["groups"]
        build = {
            "tasks_sha256": hashlib.sha256(json.dumps(
                tasks, ensure_ascii=False, sort_keys=True,
                separators=(",", ":"),
            ).encode()).hexdigest(),
            "structure": manifest["structure"],
        }
    overrides = dict(confirmed_config)
    overrides["c"] = int(manifest["cell"]["budget_tokens"])
    config = PolicyConfig.from_overrides(PolicyConfig(), **overrides)
    policies = {
        arm: make_policy("arc" if arm == "classic" else arm, config,
                         corpus.registry())
        for arm in WORKING_SET_ARMS
    }
    task_rows: list[dict] = []
    hot_counts = {"corrected_t2_resident": 0, "supersession_t2_resident": 0}
    curation_audit: list[dict] = []
    previous_transition_lengths = {arm: 0 for arm in WORKING_SET_ARMS}

    for task in tasks:
        event_index = 0
        # Registry/outcome inputs land before the answer-time snapshot.
        for payload in task["pre_events"]:
            event_index += 1
            karc = policies["karc"]
            before_list = getattr(karc, "list_of")(payload["version_id"])
            before_q = getattr(karc, "q")(payload["version_id"])
            if before_list == "T2":
                if payload["event_type"] == "corrected":
                    hot_counts["corrected_t2_resident"] += 1
                elif payload["event_type"] == "validity_changed":
                    hot_counts["supersession_t2_resident"] += 1
            event = _event(task, payload, event_index)
            for policy in policies.values():
                policy.on_event(event)
            curation_audit.append({
                "task_id": task["task_id"], "event_type": payload["event_type"],
                "version_id": payload["version_id"], "origin": event.origin,
                "karc_list_before": before_list,
                "karc_list_after": getattr(karc, "list_of")(payload["version_id"]),
                "karc_q_before": before_q,
                "karc_q_after": getattr(karc, "q")(payload["version_id"]),
            })

        snapshots = {arm: set(policy.working_set())
                     for arm, policy in policies.items()}
        required = refs_for_versions(manifest, task["required_versions"])
        stale = refs_for_versions(manifest, task["forbidden_versions"])
        per_arm: dict[str, dict] = {}
        for arm, versions in snapshots.items():
            exposure = exposure_set(manifest, working_set=versions)
            new_transitions = len(policies[arm].transitions) - previous_transition_lengths[arm]
            previous_transition_lengths[arm] = len(policies[arm].transitions)
            per_arm[arm] = {
                "working_set_versions": sorted(versions),
                "working_set_doc_ids": sorted({ref.artifact_id for ref in exposure}),
                "working_set_tokens": sum(corpus.registry()[v].size_tok for v in versions),
                "required_exposed": required <= exposure,
                "stale_exposed": bool(stale & exposure),
                "transitions_since_previous_snapshot": new_transitions,
                "actual_fetch_observable": True,
                "unresolved": 0,
            }
        jaccard = _jaccard(snapshots["karc"], snapshots["classic"])
        classic_stale_only = (per_arm["classic"]["stale_exposed"]
                              and not per_arm["karc"]["stale_exposed"])
        task_rows.append({
            "cell": manifest["cell"], "task_id": task["task_id"],
            "seq": task["seq"], "structure": task["structure"],
            "required_versions": task["required_versions"],
            "forbidden_versions": task["forbidden_versions"],
            "jaccard_karc_classic": jaccard,
            "classic_stale_karc_clean": classic_stale_only,
            "arms": per_arm,
        })

        # All arms consume the same exogenous read stream after the snapshot.
        for version_id in task["required_versions"]:
            event_index += 1
            event = _event(task, {"event_type": "read", "version_id": version_id},
                           event_index)
            for policy in policies.values():
                policy.on_event(event)
        # Positive quality signals can only originate here, never in script.
        for outcome_type in task["outcome_inject"]:
            for version_id in task["required_versions"]:
                event_index += 1
                event = _event(
                    task, {"event_type": outcome_type, "version_id": version_id},
                    event_index, origin="outcome-injector",
                )
                for policy in policies.values():
                    policy.on_event(event)
                karc = policies["karc"]
                curation_audit.append({
                    "task_id": task["task_id"], "event_type": outcome_type,
                    "version_id": version_id, "origin": event.origin,
                    "karc_list_after": getattr(karc, "list_of")(version_id),
                    "karc_q_after": getattr(karc, "q")(version_id),
                })

    arm_summary: dict[str, dict] = {}
    for arm, policy in policies.items():
        rows = [row["arms"][arm] for row in task_rows]
        summary = policy.summary()
        summary.update({
            "coverage_tasks": sum(row["required_exposed"] for row in rows),
            "stale_exposure_tasks": sum(row["stale_exposed"] for row in rows),
            "working_set_token_task_sum": sum(row["working_set_tokens"] for row in rows),
            "working_set_token_task_mean": sum(row["working_set_tokens"] for row in rows) / len(rows),
            "final_state_hash": policy.state_hash(),
            "transition_counts": _transition_counts(policy),
        })
        arm_summary[arm] = summary
    cell = {
        "schema_version": 1,
        "cell": manifest["cell"],
        "fixture": {"manifest_sha256": manifest["manifest_sha256"],
                    "tasks_sha256": build["tasks_sha256"],
                    "structure": build["structure"]},
        "events": {
            "corrected": build["structure"]["corrected_events"],
            "supersession": build["structure"]["supersession_events"],
            **hot_counts,
        },
        "arms": arm_summary,
        "jaccard": {
            "divergent_tasks": sum(row["jaccard_karc_classic"] < 1.0 for row in task_rows),
            "divergence_rate": sum(row["jaccard_karc_classic"] < 1.0 for row in task_rows) / len(task_rows),
            "minimum": min(row["jaccard_karc_classic"] for row in task_rows),
            "mean": sum(row["jaccard_karc_classic"] for row in task_rows) / len(task_rows),
        },
        "classic_stale_karc_clean_tasks": sum(
            row["classic_stale_karc_clean"] for row in task_rows
        ),
        "validated_origin_contract": {
            "scripted_validated_events": sum(
                event["event_type"] == "validated" for task in tasks
                for event in task["pre_events"]
            ),
            "outcome_injector_validated_events": sum(
                label == "validated" for task in tasks for label in task["outcome_inject"]
            ),
        },
        "curation_audit": curation_audit,
        "karc_transition_audit": [
            {
                **transition.as_dict(),
                "artifact_id": manifest["artifacts"][transition.artifact_id]["artifact_id"],
                "version_id": transition.artifact_id,
            }
            for transition in policies["karc"].transitions
        ],
    }
    cell["sha256"] = hashlib.sha256(
        json.dumps(cell, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return cell, task_rows


def replay_grid(*, confirmed_config: dict) -> tuple[list[dict], list[dict]]:
    from karc.bench.e4_fixture import BUDGET_GRID, RHO_GRID, SIGMA_GRID

    cells: list[dict] = []
    tasks: list[dict] = []
    for rho in RHO_GRID:
        for sigma in SIGMA_GRID:
            for budget_pct in BUDGET_GRID:
                cell, rows = replay_cell(
                    rho=rho, sigma=sigma, budget_pct=budget_pct,
                    confirmed_config=confirmed_config,
                )
                cells.append(cell)
                tasks.extend(rows)
    return cells, tasks
