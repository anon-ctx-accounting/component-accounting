# Configurations

Data retain the original experiment paths and internal identifiers.

**Arms.** Every main component configuration pairs two arms. The managed arm is K-ARC, the
resident-set manager described in the paper; its data key is `karc-full`. The
retrieval arm is budget-matched BM25 retrieval; its data key is `rag-bm25`.
Each primary-pair session starts a new thread, and both arms retain history
within that session. The auxiliary full-history arm also retains its thread;
sliding-window-compaction and stateless-rag start a new thread on every turn.
These auxiliary names match both Table 2 and the data keys.

**Legs.** Two directories each hold one run executed on two provider routes. The
route is recorded as leg A or leg B, and each leg is one configuration.

## Main component-accounting configurations

| Name | Data | Harness | Provider / model | Window | Warm-up outside window | Retained size | Leg |
|---|---|---|---|---|---|---|---|
| CC-A-g | [E24-BASE-RECAP](docs/experiments/E24-BASE-RECAP/) | Claude Code 2.1.220 | Anthropic / claude-sonnet-5 | growth S01–S12 | none | 12 × 8 × 2 arms | — |
| CC-A-s | [E26-CLAUDE-STEADY](docs/experiments/E26-CLAUDE-STEADY/) | Claude Code 2.1.220 | Anthropic / claude-sonnet-5 | steady S09–S20 (seed 4352) | none; selected by schedule position | 12 × 8 × 2 arms | — |
| pi-A-g | [E18-PI-REVERSAL](docs/experiments/E18-PI-REVERSAL/) | pi 0.84.3 | Anthropic / claude-sonnet-5 | growth S01–S12 | none | 12 × 8 × 2 arms | A |
| pi-O-g | [E18-PI-REVERSAL](docs/experiments/E18-PI-REVERSAL/) | pi 0.84.3 | OpenAI / gpt-5.6-luna | growth S01–S12 | none | 12 × 8 × 2 arms | B |
| pi-A-s | [E19-STEADY-STATE](docs/experiments/E19-STEADY-STATE/) | pi 0.84.3 | Anthropic / claude-sonnet-5 | steady S09–S20 | 8 sessions / 64 turns | 12 × 8 × 2 arms | A |
| pi-O-s | [E19-STEADY-STATE](docs/experiments/E19-STEADY-STATE/) | pi 0.84.3 | OpenAI / gpt-5.6-luna | steady S09–S20 | 8 sessions / 64 turns | 12 × 8 × 2 arms | B |
| Cx-O-g | [E25-CODEX-BASEWINDOW](docs/experiments/E25-CODEX-BASEWINDOW/) | Codex CLI 0.145.0 | OpenAI / gpt-5.6-luna | growth S01–S12 (seed 4200) | none | 12 × 8 × 2 arms | — |
| Cx-O-s | [E21-CODEX-COMPONENT](docs/experiments/E21-CODEX-COMPONENT/) | Codex CLI 0.145.0 | OpenAI / gpt-5.6-luna | steady S09–S20 | 8 sessions / 64 turns | 12 × 8 × 2 arms | — |

## Auxiliary groups

| Name | Data | Harness | Provider / model | Window, sessions, and interpretation |
|---|---|---|---|---|
| 16-turn auxiliary baselines | [E11-BASE3](docs/experiments/E11-BASE3/) | Codex CLI 0.144.5 | OpenAI / gpt-5.6-luna | The auxiliary baseline run reported in Table 2 of the paper: 12 sessions × 16 turns on a 16-turn variant of the fixture, with three additional baseline arms (full-history, sliding-window-compaction, stateless-rag) plus the two primary arms carried over as controls. Table 2 prints exact integer token totals corrected in E22-CODEX-CUMULATIVE; uncached input and creation remain combined |
| reuse-density | [E23-REUSE-DENSITY](docs/experiments/E23-REUSE-DENSITY/) | pi 0.84.3 | OpenAI / gpt-5.6-luna | nominal reuse 0.25 / 0.75 relative to pi-O-s; schedule warm-up positions 8 / 11; warm-up turns not executed; 12 paired measurement sessions per endpoint |
| budget-steady-10 | [E27-BUDGET-10PCT](docs/experiments/E27-BUDGET-10PCT/) | pi 0.84.3 | OpenAI / gpt-5.6-luna | resident and BM25 budgets 10%; 12 schedule warm-up positions; warm-up turns not executed; 12 paired measurement sessions in steady S13–S24 |
| budget-sweep | [E28-BUDGET-SWEEP](docs/experiments/E28-BUDGET-SWEEP/) | pi 0.84.3 | OpenAI / gpt-5.6-luna; Anthropic / claude-sonnet-5 | growth S01–S12; 12 paired sessions per configuration; OpenAI 2.5% / 10% / 20%, Anthropic 10%; no warm-up |

## Supporting experiments

| Experiment | Description |
|---|---|
| [E10-P2-XR](docs/experiments/E10-P2-XR/) | The original frozen Claude Code / Anthropic growth measurement, reported in the paper as a cache-state sensitivity; later remeasured as CC-A-g in E24-BASE-RECAP. |
| [E13-RECOMP](docs/experiments/E13-RECOMP/) | Component accounting reconstruction and sensitivity evidence. |
| [E22-CODEX-CUMULATIVE](docs/experiments/E22-CODEX-CUMULATIVE/) | Codex CLI reports a per-thread cumulative input counter. Summing those reports instead of differencing successive ones counts earlier turns repeatedly. This directory recomputes the affected totals, including the auxiliary baseline run. |
| [E15-CODEX-WRITE](docs/experiments/E15-CODEX-WRITE/) | Probe of exposed cache-write telemetry. |
| [E16-METERED-WRITE](docs/experiments/E16-METERED-WRITE/) | Metered cache-write evidence. |
| [E20-CODEX-METERED](docs/experiments/E20-CODEX-METERED/) | Codex metered telemetry evidence. |
| [E17B-PI-OPENAI-GATE](docs/experiments/E17B-PI-OPENAI-GATE/) | Gate records for the pi / OpenAI runtime. |
| [E10-P2-H16](docs/experiments/E10-P2-H16/) | Sensitivity measurement on the 16-turn variant of the fixture; an input to the corrected auxiliary baseline totals. |

The pi measurements use `--thinking off`. The main Codex CLI measurements use
low reasoning effort; the older 16-turn auxiliary measurements use high effort.
See the [accounting contract](docs/accounting-contract.md) for the recorded
reasoning totals and the sensitivity calculation.

The g/s distinction describes replay-derived resident snapshots, not cold/warm provider caches.
A provider change also changes the model. Naming does not establish a single-axis causal comparison.
Table sizes describe retained windows. Discarded attempts and reruns are described in the experiment summaries and retained raw records.
