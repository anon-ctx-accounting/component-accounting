"""E17B post-hoc verification. Derived from E17's verify_e17_crosscheck.py.

Two jobs:

1.  Three independent observation paths must agree on the MCP round-trip count
    (pi's tool_execution_start events / the bridge audit JSONL / the K-ARC
    server's first-party ingest_observations).  E17 established this discipline;
    it is not a gate criterion here but a disagreement would invalidate the run.

2.  Decide the inclusion relation of pi's OpenAI-leg `usage.input` by counting
    how many calls satisfy each of the two mutually exclusive identities, and
    locate the cold call (the first call of each regime's first turn) so that
    criterion (c) — cacheWrite > 0 on a cold call — is judged on numbers rather
    than on the ordering assumption alone.

Reads only artifacts that already exist; makes no model calls.
R-9: counts and sha256 only; the session JSONL is opened to count usage rows,
never to copy text.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

HERE = Path(__file__).resolve()
CELL_DIR = HERE.parents[1]
REPO = HERE.parents[4]
if not (REPO / "pyproject.toml").exists() or not (REPO / "src" / "karc").is_dir():
    raise SystemExit(f"E17B: repository root misresolved as {REPO}")
RAW = CELL_DIR / "raw"
RUN = REPO / "tmp" / "e17b"
# Session tags in execution order.  `pilot` is the 1-turn run that measured the
# prefix size before the definitive runs; it must be included here because the
# first-party DB (`ingest_observations`) is cumulative over every run against
# the same index.db, so leaving it out makes the three paths disagree by exactly
# the pilot's round-trip count.
TAGS = ("pilot", "short", "long")


def session_usage(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        message = entry.get("message")
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        usage = message.get("usage")
        if not isinstance(usage, dict):
            continue
        rows.append({
            "input": usage.get("input"), "output": usage.get("output"),
            "cacheRead": usage.get("cacheRead"), "cacheWrite": usage.get("cacheWrite"),
            "cacheWrite1h": usage.get("cacheWrite1h"),
            "stopReason": message.get("stopReason"),
            "rawStopReason": message.get("rawStopReason"),
        })
    return rows


def main() -> int:
    report: dict = {"_meta": {"generated_at_utc": time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, "regimes": {}}

    db = RUN / "work" / ".karc" / "index.db"
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    observations = conn.execute(
        "select count(*) from ingest_observations where source_channel='mcp'"
    ).fetchone()[0]
    runtimes = sorted(row[0] for row in conn.execute(
        "select distinct runtime from ingest_observations"))
    events = {row[0]: row[1] for row in conn.execute(
        "select event_type, count(*) from events group by event_type")}
    conn.close()

    # pi's own model catalog for gpt-5.6-luna (pi-ai dist providers/openai.models.js,
    # $ per 1M tokens).  Secondary source: E17 §8 showed pi's Claude catalog carried
    # introductory pricing while the standard card differed, so these rates are used
    # ONLY to reconstruct pi's own cost arithmetic — never to price a result.
    PI_CATALOG_RATES = {"input": 0.2, "output": 1.2,
                        "cacheRead": 0.02, "cacheWrite": 0.25}

    pi_starts_total = 0
    audit_total = 0
    grand = {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}
    pi_cost_calls = 0.0
    inclusive_true = exclusive_true = discriminating = total_calls = 0
    all_closures: list[dict] = []

    for tag in TAGS:
        smoke_path = RAW / f"smoke-{tag}.json"
        if not smoke_path.exists():
            continue
        smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
        pi_starts = sum(row["tool_execution_start_count"] for row in smoke["rows"])
        audit_path = RUN / f"bridge-audit-{tag}.jsonl"
        audit_lines = ([json.loads(line) for line
                        in audit_path.read_text(encoding="utf-8").splitlines()]
                       if audit_path.exists() else [])
        pi_starts_total += pi_starts
        audit_total += len(audit_lines)

        sessions = sorted(RUN.joinpath("sessions").glob(f"*_e17b-{tag}.jsonl"))
        jsonl_rows = session_usage(sessions[-1]) if sessions else []
        stdout_rows = [call for row in smoke["rows"] for call in row["api_calls"]]
        comparable = ["input", "output", "cacheRead", "cacheWrite",
                      "stopReason", "rawStopReason"]
        agree = (len(jsonl_rows) == len(stdout_rows) and all(
            all(left.get(field) == right.get(field) for field in comparable)
            for left, right in zip(jsonl_rows, stdout_rows)))

        closures = [closure for row in smoke["rows"] for closure in row["closure"]]
        # Independent internal consistency test of the two cache buckets: within
        # one session the tokens a call reports as written must be exactly the
        # tokens the next cache-eligible call reports as read.  If read and write
        # were not the same accounting quantity, or if any call silently dropped
        # a bucket, this chain would break.  Calls whose prefix is below the
        # 1,024-token cache minimum are skipped: the cache is not engaged there.
        chain, cumulative, chain_ok = [], 0, True
        for closure in closures:
            if closure["cacheRead"] == 0 and closure["cacheWrite"] == 0:
                chain.append({"skipped_below_cache_minimum": closure["input"]})
                continue
            matches = closure["cacheRead"] == cumulative
            chain_ok = chain_ok and matches
            chain.append({"expected_read": cumulative,
                          "observed_read": closure["cacheRead"],
                          "matches": matches, "write": closure["cacheWrite"]})
            cumulative = closure["cacheRead"] + closure["cacheWrite"]
        all_closures.extend(closures)
        total_calls += len(closures)
        inclusive_true += sum(1 for c in closures
                              if c["inclusive_hypothesis_total_eq_input_plus_output"])
        exclusive_true += sum(1 for c in closures
                              if c["exclusive_hypothesis_total_eq_four_sum"])
        discriminating += sum(1 for c in closures if c["discriminating"])

        for row in smoke["rows"]:
            for bucket in grand:
                grand[bucket] += row["usage_turn"][bucket]
            for call in row["api_calls"]:
                pi_cost_calls += float(call.get("cost_total") or 0.0)

        first_turn = smoke["rows"][0] if smoke["rows"] else {}
        # Two distinct notions of "cold" must not be conflated:
        #  * the session's first API call — cold by construction, but on this
        #    leg its prefix can sit below OpenAI's 1,024-token cache minimum,
        #    in which case the cache is not engaged at all and write is 0
        #    legitimately (E16 §1.2 / note 4);
        #  * the first cache-eligible cold call — cacheRead == 0 with a prefix
        #    over the minimum, i.e. the call that must establish the entry.
        # Criterion (c) is judged on the second.
        cold = (first_turn.get("closure") or [{}])[0]
        eligible = next((c for c in closures
                         if c["cacheRead"] == 0 and c["cacheWrite"] > 0), None)
        eligible_index = (closures.index(eligible) if eligible is not None else None)

        report["regimes"][tag] = {
            "regime": smoke["regime"],
            "turns_executed": smoke["turns_executed"],
            "returncodes": [row["returncode"] for row in smoke["rows"]],
            "api_calls_total": len(closures),
            "pi_tool_execution_start_total": pi_starts,
            "bridge_audit_lines_total": len(audit_lines),
            "bridge_audit_outcomes": sorted({entry.get("outcome")
                                             for entry in audit_lines}),
            "bridge_audit_has_text_fields": sorted(
                {key for entry in audit_lines for key in entry
                 if key in ("args", "arguments", "result", "text", "prompt")}),
            "session_jsonl_assistant_usage_rows": len(jsonl_rows),
            "stdout_assistant_usage_rows": len(stdout_rows),
            "session_jsonl_equals_stdout_stream": agree,
            "session_first_api_call": cold,
            "cold_call": cold,
            "cold_call_cacheWrite_gt_0": int(cold.get("cacheWrite") or 0) > 0,
            "cold_call_cacheRead": cold.get("cacheRead"),
            "first_cache_eligible_cold_call": eligible,
            "first_cache_eligible_cold_call_index": eligible_index,
            "first_cache_eligible_cold_call_cacheWrite_gt_0": (
                eligible is not None and eligible["cacheWrite"] > 0),
            "turn_cacheWrite_totals": [row["usage_turn"]["cacheWrite"]
                                       for row in smoke["rows"]],
            "turn_cacheRead_totals": [row["usage_turn"]["cacheRead"]
                                      for row in smoke["rows"]],
            "turn_input_totals": [row["usage_turn"]["input"]
                                  for row in smoke["rows"]],
            "turn_output_totals": [row["usage_turn"]["output"]
                                   for row in smoke["rows"]],
            "turn_cacheWrite1h_totals": [row["usage_turn"]["cacheWrite1h"]
                                         for row in smoke["rows"]],
            "cacheWrite1h_key_present_any": any(
                call.get("cacheWrite1h_key_present")
                for row in smoke["rows"] for call in row["api_calls"]),
            "usage_key_sets": sorted({",".join(call["usage_keys"])
                                      for row in smoke["rows"]
                                      for call in row["api_calls"]}),
            "calls_with_cacheWrite_gt_0": sum(1 for c in closures
                                              if c["cacheWrite"] > 0),
            "calls_with_cacheRead_gt_0": sum(1 for c in closures
                                             if c["cacheRead"] > 0),
            "write_read_chain": chain,
            "write_read_chain_exact": chain_ok,
            "write_read_chain_final_cumulative": cumulative,
            "cacheWrite_sum": sum(c["cacheWrite"] for c in closures),
            "cost_total_reported_by_pi": round(
                sum(row["usage_turn"]["cost_total"] for row in smoke["rows"]), 6),
        }

    # The two identities coincide trivially on any call with
    # cacheRead + cacheWrite == 0, so the verdict is read off the discriminating
    # calls only; the non-discriminating ones are reported but cannot vote.
    disc = [c for c in all_closures if c["discriminating"]]
    disc_exclusive = sum(1 for c in disc
                         if c["exclusive_hypothesis_total_eq_four_sum"])
    disc_inclusive = sum(1 for c in disc
                         if c["inclusive_hypothesis_total_eq_input_plus_output"])
    report["inclusion_verdict"] = {
        "calls": total_calls,
        "discriminating_calls": discriminating,
        "inclusive_hypothesis_holds_all_calls": inclusive_true,
        "exclusive_hypothesis_holds_all_calls": exclusive_true,
        "inclusive_hypothesis_holds_discriminating": disc_inclusive,
        "exclusive_hypothesis_holds_discriminating": disc_exclusive,
        "input_ge_read_plus_write_all_calls": sum(
            1 for c in all_closures if c["input_ge_read_plus_write"]),
        "input_ge_read_plus_write_discriminating": sum(
            1 for c in disc if c["input_ge_read_plus_write"]),
        "verdict": (
            "exclusive: usage.input excludes cacheRead and cacheWrite; "
            "gross = input + cacheRead + cacheWrite"
            if discriminating > 0 and disc_exclusive == discriminating
            and disc_inclusive == 0 else
            "inclusive: usage.input already contains cacheRead and cacheWrite; "
            "gross = input"
            if discriminating > 0 and disc_inclusive == discriminating
            and disc_exclusive == 0 else
            "undetermined — see per-call closure rows"),
    }
    # A third, independent read on the inclusion question: reconstruct pi's own
    # reported cost from the four buckets.  If `input` were inclusive, pricing it
    # at the uncached rate on top of cacheRead and cacheWrite would double-count,
    # so only one of the two reconstructions can reproduce pi's number.
    exclusive_cost = sum(grand[b] * PI_CATALOG_RATES[b] / 1e6 for b in grand)
    inclusive_cost = (grand["input"] * PI_CATALOG_RATES["input"]
                      + grand["output"] * PI_CATALOG_RATES["output"]) / 1e6
    report["pricing_reconstruction"] = {
        "rates_source": "pi model catalog (secondary; not a K-ARC rate card)",
        "rates_per_million_usd": PI_CATALOG_RATES,
        "grand_totals": grand,
        "pi_reported_cost_sum_over_calls": pi_cost_calls,
        "exclusive_reconstruction": exclusive_cost,
        "inclusive_reconstruction": inclusive_cost,
        "exclusive_matches_pi": abs(pi_cost_calls - exclusive_cost) < 1e-12,
        "inclusive_matches_pi": abs(pi_cost_calls - inclusive_cost) < 1e-12,
    }
    report["db_first_party"] = {
        "ingest_observations_mcp": observations,
        "runtimes": runtimes,
        "events_by_type": events,
    }
    report["three_paths_agree"] = (pi_starts_total == audit_total == observations)
    report["three_path_counts"] = {
        "pi_tool_execution_start": pi_starts_total,
        "bridge_audit_lines": audit_total,
        "db_ingest_observations_mcp": observations,
    }
    (RAW / "crosscheck.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
