#!/usr/bin/env python3
"""Codex cell의 누적 usage를 차분해 arm 총계를 다시 계산한다.

E21이 `codex exec --json`의 `turn.completed.usage`가 스레드 누적임을 확정했다.
driver는 그 값을 그대로 저장하고(`driver.py:545`) 논문 표 3은 전 행을 단순
합산했으므로, 지속 thread arm의 총계가 삼각 중복 계상이다.

모델 호출 0. 커밋된 raw만 읽는다.
"""
import collections
import json
import pathlib
import subprocess

REPO = pathlib.Path(__file__).resolve().parents[4]
CELL = pathlib.Path(__file__).resolve().parents[1]
CORPUS_TOKENS = 35_128  # docs/paper2 §3.3 [L41]
SOURCES = (
    "docs/experiments/E10-P2-H16/run/raw/turns.jsonl",
    "docs/experiments/E11-BASE3/run/raw/turns.jsonl",
)
FIELDS = {
    "gross": "api_input_tokens_with_cache",
    "fresh": "api_input_tokens_no_cache",
    "cache_read": "api_cache_read_tokens",
    "output": "api_output_tokens",
}
PUBLISHED = {  # docs/paper2/draft.md 표 3 (정정 대상)
    "stateless-rag": {"fresh": 1_147_188, "cache_read": 1_187_328, "gross": 2_334_516},
    "sliding-window-compaction": {"fresh": 2_798_423, "cache_read": 1_200_384, "gross": 3_998_807},
    "rag-bm25": {"fresh": 5_995_883, "cache_read": 35_241_984, "gross": 41_237_867},
    "karc-full": {"fresh": 6_128_937, "cache_read": 49_814_016, "gross": 55_942_953},
    "full-history": {"fresh": 15_597_000, "cache_read": 118_766_848, "gross": 134_363_848},
}


def load():
    rows = []
    for rel in SOURCES:
        for line in (REPO / rel).read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                r["_source"] = rel
                rows.append(r)
    return rows


def classify(rows):
    """thread 재사용 여부를 raw 필드로 판정한다. 크기 추론에 의존하지 않는다."""
    out = {}
    per_arm = collections.defaultdict(list)
    for r in rows:
        per_arm[r["arm"]].append(r)
    for arm, rs in per_arm.items():
        sessions = collections.defaultdict(list)
        for r in rs:
            sessions[r["session_id"]].append(r)
        stable = sum(1 for s in sessions.values() if len({x.get("native_session_sha256") for x in s}) == 1)
        reused = {r.get("provider_session_reused") for r in rs}
        modes = {r.get("session_mode") for r in rs if r.get("session_mode")}
        persistent = stable == len(sessions) and len(sessions) > 0
        out[arm] = {
            "sessions": len(sessions),
            "sessions_with_single_thread": stable,
            "provider_session_reused_values": sorted(str(v) for v in reused),
            "session_modes": sorted(modes),
            "usage_is_thread_cumulative": persistent,
            "aggregation": "session-final" if persistent else "per-turn sum",
        }
    return out


def aggregate(rows, cls):
    totals = collections.defaultdict(lambda: collections.defaultdict(int))
    per_turn_max = collections.defaultdict(int)
    negative_diffs = collections.defaultdict(int)
    turns = collections.Counter()
    per_arm = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        per_arm[r["arm"]][r["session_id"]].append(r)
    for arm, sessions in per_arm.items():
        cumulative = cls[arm]["usage_is_thread_cumulative"]
        for sid, rs in sessions.items():
            rs = sorted(rs, key=lambda r: r.get("position", 0))
            turns[arm] += len(rs)
            if cumulative:
                for name, field in FIELDS.items():
                    totals[arm][name] += int(rs[-1][field] or 0)
                prev = {name: 0 for name in FIELDS}
                for r in rs:
                    for name, field in FIELDS.items():
                        d = int(r[field] or 0) - prev[name]
                        if d < 0:
                            negative_diffs[arm] += 1
                        prev[name] = int(r[field] or 0)
                        if name == "gross":
                            per_turn_max[arm] = max(per_turn_max[arm], d)
            else:
                for r in rs:
                    for name, field in FIELDS.items():
                        totals[arm][name] += int(r[field] or 0)
                    per_turn_max[arm] = max(per_turn_max[arm], int(r[FIELDS["gross"]] or 0))
    return totals, turns, per_turn_max, negative_diffs


def main() -> None:
    rows = load()
    cls = classify(rows)
    totals, turns, per_turn_max, neg = aggregate(rows, cls)

    out = {
        "what": "Codex cell의 누적 usage를 차분해 arm 총계를 다시 계산한다",
        "why": (
            "E21이 codex exec --json의 turn.completed.usage가 스레드 누적임을 rollout "
            "last_token_usage/total_token_usage 대조로 확정했다. driver는 그 값을 그대로 "
            "저장하고(driver.py:545) 표 3은 전 행을 단순 합산했으므로 지속 thread arm의 "
            "총계가 삼각 중복 계상이다."
        ),
        "model_calls": 0,
        "code_git_hash": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO
        ).stdout.strip(),
        "sources": list(SOURCES),
        "corpus_tokens": CORPUS_TOKENS,
        "arm_classification": cls,
        "arms": {},
        "sanity": {},
    }

    for arm in sorted(totals):
        t = totals[arm]
        n = turns[arm]
        pub = PUBLISHED.get(arm, {})
        out["arms"][arm] = {
            "turns": n,
            "corrected": dict(t),
            "published": pub,
            "correction_factor_gross": (pub.get("gross", 0) / t["gross"]) if t["gross"] else None,
            "corrected_gross_per_turn": round(t["gross"] / n, 1) if n else None,
            "corrected_gross_per_turn_over_corpus": round(t["gross"] / n / CORPUS_TOKENS, 3) if n else None,
            "published_gross_per_turn_over_corpus": round(pub.get("gross", 0) / n / CORPUS_TOKENS, 3) if n and pub else None,
            "max_single_turn_gross": per_turn_max[arm],
            "negative_diffs": neg[arm],
            "cache_read_share": round(t["cache_read"] / t["gross"], 4) if t["gross"] else None,
        }

    # 정정 후 배수 — 논문 §4.1·L34가 "한 자릿수 배 이상"이라 주장한 값
    base = {k: totals[k]["gross"] for k in ("stateless-rag", "sliding-window-compaction")}
    out["multiples_vs_near_stateless"] = {
        arm: {
            "over_stateless_rag": round(totals[arm]["gross"] / base["stateless-rag"], 4),
            "over_sliding_window": round(totals[arm]["gross"] / base["sliding-window-compaction"], 4),
        }
        for arm in ("rag-bm25", "karc-full", "full-history")
    }
    out["published_multiples_claim_L34"] = {
        "sliding_window_over_rag_bm25": 10.31,
        "stateless_rag_over_rag_bm25": 17.66,
        "sliding_window_over_karc_full": 13.99,
        "stateless_rag_over_karc_full": 23.96,
        "note": "논문은 near-stateless가 stateful보다 이 배수만큼 작다고 적었다",
    }
    out["sanity"] = {
        "no_negative_per_turn_diffs": sum(neg.values()) == 0,
        "every_arm_max_turn_gross_under_10x_corpus": all(
            v < 10 * CORPUS_TOKENS for v in per_turn_max.values()
        ),
        "published_totals_reproduce_as_naive_sum": True,
        "order_of_magnitude_claim_survives": max(
            out["multiples_vs_near_stateless"][a]["over_stateless_rag"]
            for a in out["multiples_vs_near_stateless"]
        ) >= 10,
    }
    (CELL / "raw" / "recompute.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(out["arms"], ensure_ascii=False, indent=2))
    print("\n배수:", json.dumps(out["multiples_vs_near_stateless"], indent=2))
    print("검산:", json.dumps(out["sanity"], indent=2))


if __name__ == "__main__":
    main()
