# E20-CODEX-METERED

This probe measures cache-write reporting with the Codex CLI client fixed and the credential route changed to metered API access. It is supporting evidence for the later Cx-O-g and Cx-O-s configurations. See [CONFIGURATIONS.md](../../../CONFIGURATIONS.md) for the mapping.

Raw observations are dated 2026-09-01 UTC. Codex CLI 0.145.0 used OpenAI gpt-5.6-luna with reasoning effort low. Four successful model turns ran in one persistent session after a keyless control failed in the same isolated home. This is a prose-based instrumentation probe, without a paired S01–S12 window or karc-full versus rag-bm25 arms.

raw/metered-multiturn-round-1.json contains route-provenance checks, successive raw and normalized usage reports, field-presence checks, and the original evaluation. The retained script shows how the isolated API-key login and keyless control were performed. There is no analysis.json. Later records in E21 and E22 establish that stdout counters are cumulative.

These probe records are a telemetry probe rather than a preregistered measurement. The keyless control precedes the measured turns and is recorded separately from their successful usage. No paired-fixture accuracy claim is made; the first-turn fixture defect is not a grading condition for this probe.

Inspect the retained probe evidence from the repository root:

```sh
python3 -m json.tool docs/experiments/E20-CODEX-METERED/raw/metered-multiturn-round-1.json
```
