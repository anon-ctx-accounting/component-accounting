# E13-RECOMP

This offline analysis recomposes committed measurements under alternative accounting rules and examines uncertainty, task-type grades, and sensitivity. It is supporting evidence for the original Claude Code measurement and the auxiliary baseline comparison. See [CONFIGURATIONS.md](../../../CONFIGURATIONS.md) for the mapping.

A reanalysis date in UTC is not retained. No model calls, new experiments, or new fixtures were used. Inputs are the included E10-P2-H16 and E11-BASE3 Codex CLI 0.144.5 records for OpenAI gpt-5.6-luna, plus E10-P2-XR Claude Code 2.1.220 records for Anthropic claude-sonnet-5. The Codex sessions contain 16 turns and the Claude sessions 8 turns, with 12 paired sessions per comparison. Comparisons include the two resident/retrieval arms and the three additional auxiliary baseline arms.

raw/recomp.json contains seeded bootstrap results, component pricing, task-type joins, and registered conditions. raw/fragility.json preserves additional sensitivity and audit results. scripts/e13_recomp.py reads included schedules and turn records; e13_fragility.py also scans repository documents and code.

The recomposition conditions were preregistered before the analysis. No model runs were discarded or repeated here. The first-turn fixture defect is explicitly recorded. Document-scanning fragility audits depend on which documents a checkout contains and are outside make verify; the later E22 cumulative-counter correction remains separate.

Recompute the released numerical checks from the repository root:

```sh
python3 scripts/verify.py --experiment E13-RECOMP
```
