# E22-CODEX-CUMULATIVE

Codex CLI reports a per-thread cumulative input counter on stdout. Summing successive reports instead of differencing them counts earlier turns repeatedly. This offline reanalysis recomputes the affected totals; Table 2 of the paper prints the corrected values in integer tokens. It adds no new model configuration. See [CONFIGURATIONS.md](../../../CONFIGURATIONS.md) for the mapping.

A reanalysis date of 2026-09-08 is recorded; a UTC timestamp is not retained in these records. No model calls were made. Inputs are Codex CLI 0.144.5 measurements of OpenAI gpt-5.6-luna from E10-P2-H16 and E11-BASE3. Each arm has 12 sessions of 16 turns. Arms are K-ARC (data key `karc-full`), rag-bm25, full-history, sliding-window-compaction, and stateless-rag; the first three retain provider threads, while the latter two rebuild them.

raw/recompute.json contains thread-based arm classifications, naive and corrected totals, correction factors, and consistency checks. scripts/recompute.py reads the included upstream turns.jsonl files and uses final session counters for persistent threads. The original upstream rows and summaries remain unchanged.

This is a post-hoc correction rather than a newly preregistered measurement. No model runs were discarded or repeated for it. The correction also changes the interpretation of per-turn-position figures. Answer counts are unchanged, including the inherited first-turn fixture limitation.

Recompute the released numerical checks from the repository root:

```sh
python3 scripts/verify.py --experiment E22-CODEX-CUMULATIVE
```
