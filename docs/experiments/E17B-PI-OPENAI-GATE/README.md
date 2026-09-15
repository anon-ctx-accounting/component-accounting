# E17B-PI-OPENAI-GATE

This gate measures whether pi's OpenAI leg preserves per-turn usage components and observable cache writes. It is supporting evidence for the later pi-O-g and pi-O-s configurations, rather than a paired configuration itself. See [CONFIGURATIONS.md](../../../CONFIGURATIONS.md) for the mapping.

The raw short and long probe records are dated 2026-08-27 UTC. Pi 0.84.3 used OpenAI gpt-5.6-luna with thinking off and isolated metered credentials. The short regime contains 5 turns and 11 calls; long contains 2 turns and 4 calls. A separate pilot contributes the remaining calls to the recorded total of 18. There is no paired measurement window or retrieval control arm.

raw/smoke-short.json, smoke-long.json, and smoke-pilot.json contain usage and stop signals. raw/crosscheck.json records acquisition-route agreement, inclusion identities, and round-trip counts. raw/registry-probe.json records tool registration. There is no analysis.json; the gate is described by these probe and crosscheck records.

These gate records are a telemetry probe rather than a preregistered measurement. The records include an added pilot and a reduced long regime. Calls below the cache threshold are retained. The paired-workload first-turn accuracy defect is not this gate's adjudication target.

Recompute the released numerical checks from the repository root:

```sh
python3 scripts/verify.py --experiment E17B-PI-OPENAI-GATE
```
