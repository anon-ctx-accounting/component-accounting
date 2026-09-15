# E11-BASE3

This configuration measures three additional baseline strategies against the frozen horizon controls. It supplies the 16-turn auxiliary baseline comparison and retains the original experiment identifiers used by the correction analysis. See [CONFIGURATIONS.md](../../../CONFIGURATIONS.md) for the mapping.

The source checklist dates closure to 2026-08-04 KST; an execution date in UTC is not retained. Codex CLI 0.144.5 used OpenAI gpt-5.6-luna with reasoning effort high. Twelve paired sessions of 16 turns produced 576 new valid turns across full-history, sliding-window-compaction, and stateless-rag. The K-ARC (data key `karc-full`) and rag-bm25 controls come from E10-P2-H16, using the same fixture and task schedule. The new arms prohibit MCP and local tool calls.

run/raw/turns.jsonl contains the three new arms' counters, grades, and provider-session metadata. run/raw/summary.json preserves historical comparisons. raw/schedule.json and verification.json record task identity and checks. E22-CODEX-CUMULATIVE/raw/recompute.json supplies corrected cumulative-counter totals; E13 contains the paired historical reanalysis.

This configuration was preregistered before execution. The report records 576 model-spend attempts and 26 infrastructure attempts. The controls were measured in an earlier batch. Historical summaries are preserved with their cumulative-counter interpretation, and the first-turn fixture defect remains in the answer records.

Recompute the released numerical checks from the repository root:

```sh
python3 scripts/verify.py --experiment E11-BASE3
```
