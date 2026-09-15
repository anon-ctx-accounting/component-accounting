# E27-BUDGET-10PCT

This configuration measures component accounting after increasing the resident and retrieval budgets from 5% to 10%. It supplies auxiliary group budget-steady-10; E19 leg B is the recorded reference at the smaller budget. See [CONFIGURATIONS.md](../../../CONFIGURATIONS.md) for the mapping.

Execution wave records are dated 2026-09-10 UTC. Pi 0.84.3 with `--thinking off` used OpenAI gpt-5.6-luna with K-ARC (data key `karc-full`) and rag-bm25. Seed 4352 selected the steady-state window S13–S24, with 12 paired sessions × 8 turns × 2 arms, totaling 192 turns. The budget is 3,512 tokens. Twelve schedule warm-up positions were selected, but warm-up model turns were not executed.

raw/analysis.json contains component totals, bootstrap intervals, thresholds, and the reference comparison. raw/turns-b10.jsonl contains retained turn and call usage. raw/steady-b10.json records the utilization checks used to select the window; build.json records budget and schedule construction.

This configuration was preregistered before execution. No measurement runs were discarded or retried. Budget and measurement-window position both differ from the reference. Both arms scored 84/96, including 0/12 at position 1, preserving the documented first-turn fixture defect.

Recompute the released numerical checks from the repository root:

```sh
python3 scripts/verify.py --experiment E27-BUDGET-10PCT
```
