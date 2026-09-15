# E28-BUDGET-SWEEP

This batch measures component accounting at several budgets while holding the growth window fixed. Auxiliary group budget-sweep maps o025, o10, and o20 to E28-O-025, E28-O-10, and E28-O-20; a10 maps to E28-A-10. See [CONFIGURATIONS.md](../../../CONFIGURATIONS.md) for the mapping.

Execution wave records are dated 2026-09-13 UTC. Pi 0.84.3 with `--thinking off` used OpenAI gpt-5.6-luna for budgets 2.5%, 10%, and 20%, and Anthropic claude-sonnet-5 for 10%. Each configuration measures S01–S12 with 12 paired sessions × 8 turns × K-ARC (data key `karc-full`) and rag-bm25: 192 turns per configuration, 768 total. No warm-up model turns were executed.

raw/analysis.json contains totals for each budget configuration (JSON key `cells`), paired intervals, thresholds, and comparisons with E18 and E27. Each raw subdirectory contains turns.jsonl, build.json, and execution checks. raw/o20/h9-incident.json documents the retained unit's cache-contamination investigation.

This configuration was preregistered before execution. E28-O-20 first retried a failed unit, then triggered H9 because that retry reused a cached prefix. The affected karc-full S11 unit was remeasured after the retention window, at 14:36 UTC. Final measurements retain that replacement. Both arms show the first-turn fixture defect, with position 1 at 0/12.

Recompute the released numerical checks from the repository root:

```sh
python3 scripts/verify.py --experiment E28-BUDGET-SWEEP
```
