# E23-REUSE-DENSITY

This configuration varies reuse density while retaining the component-accounting measurement procedure. It belongs to auxiliary group reuse-density: tags 025 and 075 correspond to nominal reuse 0.25 and 0.75, with E19 leg B supplying the 0.50 comparison. See [CONFIGURATIONS.md](../../../CONFIGURATIONS.md) for the name-to-leg mapping.

Execution wave records are dated 2026-09-07 UTC. Both endpoints used pi 0.84.3 with `--thinking off`, OpenAI gpt-5.6-luna, K-ARC (data key `karc-full`) and rag-bm25, and a 5% budget. Each endpoint contains 12 paired sessions × 8 turns × 2 arms, totaling 192 turns. Selected windows are S09–S20 and S12–S23 after schedule warm-up positions 8 and 11; warm-up turns were not executed.

raw/analysis.json contains per-density component totals, paired intervals, and the reference contrast. raw/turns-025.jsonl and turns-075.jsonl are the retained measurements. build.json and steady files document schedule reconstruction, resident utilization, and task separation.

This configuration was preregistered before execution. The report records no discarded runs, failures, or interrupted measurements. A pilot wave is recorded separately. The first-turn fixture defect remains, and answer grades are incidental. The two density endpoints use different selected windows, as documented in the steady-state records.

Recompute the released numerical checks from the repository root:

```sh
python3 scripts/verify.py --experiment E23-REUSE-DENSITY
```
