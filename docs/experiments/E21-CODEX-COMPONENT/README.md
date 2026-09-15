# E21-CODEX-COMPONENT

This configuration measures four-component accounting through the metered Codex CLI in the steady resident-state window. It supplies configuration Cx-O-s, with E19 leg B providing the recorded pi comparison. See [CONFIGURATIONS.md](../../../CONFIGURATIONS.md) for the name-to-leg mapping.

Execution wave records are dated 2026-09-07 UTC. Codex CLI 0.145.0 used OpenAI gpt-5.6-luna with reasoning effort low and an isolated API-key route. Measurement covers S09–S20: 12 paired sessions × 8 turns × 2 arms, K-ARC (data key `karc-full`) and rag-bm25, totaling 192 turns. Eight managed-arm warm-up sessions were executed separately. The two measured arms record 1,009 reasoning tokens in total.

raw/analysis.json contains component totals, paired intervals, thresholds, and harness comparisons. raw/turns.jsonl preserves cumulative stdout counters, per-turn differences, and per-call usage events from the session transcript; turns-warmup.jsonl is separate. raw/authproof.json and crosscheck.json record route isolation and acquisition checks.

This configuration was preregistered before execution. One warm-up unit was rerun; the measurement dataset had no discarded runs. The first-turn fixture defect persisted for K-ARC, while retrieval's observed answer distribution differed. raw/probe-home-layout.json is excluded because it records an operator home-directory layout. The registered result retains all measurement sessions, including S19.

Recompute the released numerical checks from the repository root:

```sh
python3 scripts/verify.py --experiment E21-CODEX-COMPONENT
```
