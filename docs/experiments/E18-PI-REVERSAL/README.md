# E18-PI-REVERSAL

This configuration measures component accounting in pi across paired Anthropic and OpenAI legs. Leg A supplies pi-A-g and leg B supplies pi-O-g. See [CONFIGURATIONS.md](../../../CONFIGURATIONS.md) for the name-to-leg mapping.

Wave records are dated 2026-08-27 UTC; provenance separately lists 2026-08-28. Both legs used pi 0.84.3 with `--thinking off`. Anthropic used claude-sonnet-5 with long retention; OpenAI used gpt-5.6-luna with default retention. Each growth window S01–S12 contains 12 paired sessions × 8 turns × 2 arms, K-ARC (data key `karc-full`) and rag-bm25, totaling 192 turns per leg. There is no warm-up window.

raw/analysis.json contains leg-specific component totals, paired intervals, thresholds, gate results, and labeled post-hoc decomposition. raw/turns-A.jsonl and raw/turns-B.jsonl preserve per-turn and per-call usage. Crosscheck, auth, and provenance JSON files preserve acquisition and execution evidence.

This configuration was preregistered before execution. Leg B discarded two attempts and retained the third: a rate-limit failure was followed by cache contamination. Discarded-attempt evidence is retained separately. Both legs and both arms scored 0/12 at position 1 and 84/84 at positions 2–8, the recorded fixture defect.

Recompute the released numerical checks from the repository root:

```sh
python3 scripts/verify.py --experiment E18-PI-REVERSAL
```
