# E15-CODEX-WRITE

This probe measures whether the subscription Codex CLI exposes a cache-write field and separates an absent field from a reported zero. It is supporting telemetry evidence rather than a named main configuration. See [CONFIGURATIONS.md](../../../CONFIGURATIONS.md) for the mapping.

Raw observations are dated 2026-08-27 UTC. Codex CLI 0.145.0 used OpenAI gpt-5.6-luna through the subscription route; the existing 0.144.5 installation remained separate. The record includes two smoke turns and a subsequent four-turn persistent-session probe. These are instrumentation prompts, not the paired S01–S12 workload, and have no karc-full versus rag-bm25 treatment comparison.

raw/smoke-attempt-1.json and smoke-attempt-2.json retain field presence, normalized usage, model verification, and errors. raw/multiturn-round-1.json preserves successive stdout reports. raw/multiturn-usage-semantics-check.json records the original interpretation audit; the later cumulative-counter correction is retained in E22-CODEX-CUMULATIVE.

These probe records are a telemetry probe rather than a preregistered measurement. An additional harness failure without a model call is recorded. Historical usage interpretations remain in raw data and are not silently rewritten. The paired-workload first-turn fixture defect is not a grading condition for these probes.

Inspect the retained probe evidence from the repository root:

```sh
python3 -m json.tool docs/experiments/E15-CODEX-WRITE/raw/multiturn-round-1.json
```
