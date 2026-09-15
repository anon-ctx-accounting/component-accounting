# E24-BASE-RECAP

This configuration recaptures the Claude Code growth-window measurement while retaining the harness-reported monetary fields needed for rate-card reconstruction. It supplies configuration CC-A-g; the earlier frozen run is retained separately under E10-P2-XR. See [CONFIGURATIONS.md](../../../CONFIGURATIONS.md) for the name-to-leg mapping.

A execution date of 2026-09-09 is recorded; a UTC timestamp is not retained in these records. Claude Code 2.1.220 used Anthropic claude-sonnet-5 through the subscription route. The growth window is S01–S12, with 12 paired sessions, 8 turns per session, and K-ARC (data key `karc-full`) versus rag-bm25: 192 valid turns. No warm-up window was executed.

run/raw/analysis.json contains component totals, paired intervals, rate-card reconstruction, and comparisons with the frozen run. run/raw/turns.jsonl contains retained turn counts; run/raw/attempts.jsonl preserves total_cost_usd, joined to those turns. probe-warm-s01/ contains the separate cache-state probe.

This configuration was preregistered before execution. The cache-state probe is separate from the reported measurement. Both arms scored 84/96; the first-turn fixture defect remains. Harness-reported dollars are rate-card equivalents, not subscription invoices.

Recompute the released numerical checks from the repository root:

```sh
python3 scripts/verify.py --experiment E24-BASE-RECAP
```
