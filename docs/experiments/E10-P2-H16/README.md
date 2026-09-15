# E10-P2-H16

This configuration extends the horizon measurement to 16 turns per session and compares resident memory with retrieval. It is supporting horizon evidence and provides the frozen control arms used by the 16-turn auxiliary baseline comparison. See [CONFIGURATIONS.md](../../../CONFIGURATIONS.md) for the mapping.

An execution date in UTC is not retained. Codex CLI 0.144.5 used OpenAI gpt-5.6-luna through the subscription route with reasoning effort high. The design contains 12 paired sessions × 16 turns × 2 arms, K-ARC (data key `karc-full`) and rag-bm25, giving 384 valid turns. Reuse is 0.5, budget is 5%, and the resident representation uses whole documents. Both arms retain provider threads within sessions.

run/raw/turns.jsonl contains normalized usage, grades, session identifiers, and tool counts. run/raw/summary.json preserves the historical arm and position aggregates. raw/schedule.json contains task order; raw/verification.json and freeze files record execution checks. There is no analysis.json. E22-CODEX-CUMULATIVE contains the later counter correction.

This configuration was preregistered before execution. The report records 384 model-spend attempts and 4 infrastructure attempts. Historical summaries use cumulative stdout counters and remain unchanged. The first-turn fixture defect persists; corrected per-turn accounting must use successive counter differences.

Recompute the released numerical checks from the repository root:

```sh
python3 scripts/verify.py --experiment E10-P2-H16
```
