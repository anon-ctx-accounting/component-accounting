# E25-CODEX-BASEWINDOW

This configuration measures metered Codex CLI component accounting in the growth window used by the original provider comparison. It supplies configuration Cx-O-g, complementing the steady-window Cx-O-s measurement. See [CONFIGURATIONS.md](../../../CONFIGURATIONS.md) for the name-to-leg mapping.

Execution wave records are dated 2026-09-10 UTC. Codex CLI 0.145.0 used OpenAI gpt-5.6-luna, reasoning effort low, and isolated API-key credentials. Seed 4200 selects S01–S12: 12 paired sessions × 8 turns × 2 arms, K-ARC (data key `karc-full`) and rag-bm25. All 192 measurement turns completed, with 899 reasoning tokens across both arms. No warm-up was performed; the resident set grows across the selected schedule.

raw/analysis.json contains component totals, gross and priced bootstrap intervals, thresholds, and the E18 leg B comparison. raw/turns.jsonl provides cumulative counters, turn differences, and per-call usage from the session transcript. raw/slices.json and prepare.json document schedule selection; authproof.json records the keyless control and metered route.

This configuration was preregistered before execution. The report records no discarded measurement runs and no cold-start cache misassignment. Answer grades remain diagnostic and the corpus's first-turn limitation is not repaired. This is a growth-window run, not a claim that provider caches began uniformly cold.

Recompute the released numerical checks from the repository root:

```sh
python3 scripts/verify.py --experiment E25-CODEX-BASEWINDOW
```
