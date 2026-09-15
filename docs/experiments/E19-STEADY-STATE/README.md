# E19-STEADY-STATE

This confirmatory configuration measures component accounting after the resident set reaches the registered steady-state condition. Leg A supplies pi-A-s and leg B supplies pi-O-s. See [CONFIGURATIONS.md](../../../CONFIGURATIONS.md) for the name-to-leg mapping.

Execution wave records are dated 2026-08-28 UTC. Both legs used pi 0.84.3 with `--thinking off`: Anthropic claude-sonnet-5 in leg A and OpenAI gpt-5.6-luna in leg B. Measurement covers S09–S20, with 12 paired sessions × 8 turns × 2 arms per leg. The arms are K-ARC (data key `karc-full`) and rag-bm25. Each leg also executed 8 warm-up sessions for karc-full, outside measurement.

raw/analysis.json contains component totals, bootstrap intervals, thresholds, steady-state checks, and the pooled E18 contrast. raw/turns-A.jsonl and raw/turns-B.jsonl are measurement inputs; turns-warmup files are separate. raw/window-reuse-profile.json records the measured window's reuse composition.

This configuration was preregistered before execution. No runs were discarded or retried. Residual task overlap with E18-PI-REVERSAL and differing reuse composition within the measurement window are recorded. Both arms in both legs retain the first-turn fixture defect: position 1 is 0/12 and positions 2–8 are 84/84.

Recompute the released numerical checks from the repository root:

```sh
python3 scripts/verify.py --experiment E19-STEADY-STATE
```
