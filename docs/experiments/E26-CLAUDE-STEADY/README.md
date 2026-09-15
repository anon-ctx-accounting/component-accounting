# E26-CLAUDE-STEADY

This configuration measures Claude Code component accounting in the steady resident-state window. It supplies configuration CC-A-s and pairs with CC-A-g in the configuration index. See [CONFIGURATIONS.md](../../../CONFIGURATIONS.md) for the name-to-leg mapping.

A execution date of 2026-09-10 is recorded; a UTC timestamp is not retained in these records. Claude Code 2.1.220 used Anthropic claude-sonnet-5 through the subscription route. Seed 4352 selected S09–S20 from the schedule. There are 12 paired sessions of 8 turns, with K-ARC (data key `karc-full`) and rag-bm25, giving 192 valid turns. Resident state came from policy replay at the selected schedule positions; warm-up turns were not executed.

run/raw/analysis.json contains component totals, paired gross and priced intervals, monetary reconstruction, and the growth-window contrast. run/raw/turns.jsonl records retained turns, while run/raw/attempts.jsonl supplies harness-reported cost fields. run/raw/schedule.json preserves the selected task schedule.

This configuration was preregistered before execution. No measurement replacement is reported. Both arms scored 84/96, retaining the first-turn fixture defect. The record distinguishes steady resident state from provider-cache warming; reported monetary amounts are not subscription invoices.

Recompute the released numerical checks from the repository root:

```sh
python3 scripts/verify.py --experiment E26-CLAUDE-STEADY
```
