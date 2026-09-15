# 16-turn auxiliary baselines
Data: [E11-BASE3](../../docs/experiments/E11-BASE3/).
Harness: Codex CLI 0.144.5; provider/model: OpenAI / gpt-5.6-luna.
The auxiliary baseline run reported in Table 2 of the paper: 12 sessions × 16 turns on a 16-turn variant of the fixture, with three additional baseline arms (full-history, sliding-window-compaction, stateless-rag) plus the two primary arms carried over as controls.
Table 2 prints exact integer token totals, including gross input, corrected in [E22-CODEX-CUMULATIVE](../../docs/experiments/E22-CODEX-CUMULATIVE/).
