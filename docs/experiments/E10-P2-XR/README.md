# E10-P2-XR

This frozen Claude Code measurement records cross-runtime cache accounting for resident memory and retrieval. It is the original supporting growth-window record; the later monetary recapture supplies CC-A-g under E24-BASE-RECAP. See [CONFIGURATIONS.md](../../../CONFIGURATIONS.md) for the mapping.

An execution date in UTC is not retained. Claude Code 2.1.220 used Anthropic claude-sonnet-5 through the subscription route. The retained measurement covers S01–S12, with 12 paired sessions × 8 turns × 2 arms, K-ARC (data key `karc-full`) and rag-bm25, totaling 192 valid turns. The `canary/` subdirectory holds that measurement and `smoke/` holds the pre-execution validation runs that preceded it; the two are kept separate. Cross-runtime comparisons use gross input, while the per-configuration records preserve fresh input, creation, cache read, and output.

canary/raw/turns.jsonl contains retained per-turn components; attempts.jsonl contains attempt-level evidence. canary/raw/summary.json and validation.json store aggregates and gate checks. The `smoke/` directory retains probe and recovery data; approvals and freeze JSON files preserve the execution contract. There is no analysis.json here; E13-RECOMP recomputes the component statistics.

This configuration was preregistered before execution. Authentication, tool-policy, and exposure recoveries from the validation runs are retained as separate records rather than pooled with the measurement. Both arms scored 84/96, including the first-turn fixture defect. Subscription billed amounts were not observed.

Recompute the released numerical checks from the repository root:

```sh
python3 scripts/verify.py --experiment E10-P2-XR
```
