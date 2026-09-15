# Accounting contract and declared inputs

U, W, R and O are non-overlapping uncached input, cache creation, cache read
and output. Gross input is G = U + W + R. Component-weighted cost in uncached-input
units is P = U + mu_w W + mu_r R + mu_o O. Separate 5-minute and 1-hour creation
buckets are retained and priced once. Reasoning already included in output is
not added again.

| Harness | Native inclusion and aggregation |
|---|---|
| Claude Code | input_tokens is U; cache_creation_input_tokens is W; cache_read_input_tokens is R. Retain TTL subfields and sum the disjoint input categories for G. |
| pi | input is U; cacheWrite is W; cacheRead is R. cacheWrite1h is a subset of cacheWrite; their difference is 5-minute creation. Sum calls within a turn. totalTokens = U + W + R + O. |
| Codex CLI | Difference cumulative counters within each arm/session thread first. input_tokens is G; cached_input_tokens is R; cache_write_input_tokens is W; U = G - R - W. Compare differences with per-call usage events in the session transcript. |

To resolve inclusion, let I denote the harness input field and T its reported
total. Under the exclusive reading, I = U and T = I + W + R + O; under the
inclusive reading, I = U + W + R and T = I + O. When W = R = 0, both readings
coincide. The pi OpenAI gate has 13 calls with cache traffic: all satisfy the
exclusive reading and none the inclusive reading. The verifier rebuilds
those identities from the retained call records.

Older records for the 16-turn auxiliary baselines combine uncached input and creation. Persistent-thread
totals are recovered by differencing successive reports; fresh-thread baselines
are summed per turn. The combined field cannot support full component pricing. Table 2 reports
all input totals in integer tokens, including gross input.

For managed-minus-retrieval totals, mu_w_star =
-(delta_U + mu_r delta_R + mu_o delta_O) / delta_W. This changes the price applied
to a fixed trace, not the behavior of the measured model.

## Recorded reasoning settings

The pi runs use `--thinking off`; their main measurement ledgers record zero
reasoning tokens. Codex CLI uses low effort and records 899 reasoning tokens
in Cx-O-g and 1,009 in Cx-O-s across both arms. These tokens are already part
of output. The verifier differences the Codex cumulative reasoning counters
within each thread and checks them against the session transcript; pi
reasoning totals come from its retained turn ledgers.

Removing reasoning from output in both Cx-O-s arms, while keeping the same
reference rates and recorded input components, moves its cost ratio from
0.9944 to 0.9912. This records the bounded contribution of the settings
difference; the matched-workload comparison does not isolate a causal
harness effect.

## Frozen prices

These are the reference snapshots recorded for 2026-08-28, in USD per million
tokens. They are declared inputs, not assertions about current prices.

| Card | U | W, 5 minutes | W, 1 hour | R | O |
|---|---:|---:|---:|---:|---:|
| Anthropic reference | 3.00 | 3.75 | 6.00 | 0.30 | 15.00 |
| OpenAI reference | 0.20 | 0.25 | 0.25 | 0.02 | 1.20 |
| pi Anthropic catalog | 2.00 | 2.50 | 4.00 | 0.20 | 10.00 |
| pi OpenAI catalog | 0.20 | 0.25 | 0.25 | 0.02 | 1.20 |

The read multiple is 0.10, creation multiples are 1.25 and 2.00, and output
multiples are 5.0 for Anthropic and 6.0 for OpenAI. Creation costs 12.5 or 20
times a read. Pi's Anthropic catalog is two thirds of the reference absolute
level with identical multiples, preserving paired ratios. Main Anthropic writes
use the 1-hour bucket. Amount reconstruction uses the harness's card; matching
its computed amount is not independent invoice verification or a statement of
actual subscription charges. Six main configurations supply amounts, with
a largest reconstructed residual of 3 × 10^-17 USD at the paper's displayed
precision. Codex supplies no amount for that check.

## Experimental and statistical inputs

| Input | Declared value |
|---|---|
| Main configurations | Eight; versions and models in [CONFIGURATIONS.md](../CONFIGURATIONS.md) |
| Component windows | 12 paired sessions, 8 turns per arm/session |
| 16-turn auxiliary baselines | 12 sessions, 16 turns per arm/session |
| Main matched budget | 1,756 tokens, nominal 5% |
| Additional budgets | 2.5%, 10%, 20%; 878, 3,512, 7,025 tokens |
| Nominal reuse | 0.50; auxiliary endpoints 0.25 and 0.75 |
| Growth / main steady windows | S01–S12 / S09–S20 |
| Reuse endpoint windows | S09–S20 / S12–S23 |
| Steady 10% budget window | S13–S24 |
| Paired bootstrap | 10,000 resamples; seed 1313; shared session-index matrix |
| Confidence convention | Two-sided 95%; sorted draw indices 250 and 9749, zero based |
| Steady warm-up criterion | Resident-volume change at most 5%; budget utilization at least 95% |

The fixture's actual document, task, token and supersession counts are checked
from its manifest and tasks. Session counts, turn positions, nominal settings,
prices, build/model identifiers, dates and statistical settings are inputs,
not separate catalog assertions. Growth-window capacity uses whole documents:
a snapshot is full when its remaining budget cannot fit the smallest fixture
document. This differs from the steady warm-up utilization criterion.

Reference records: [E19](experiments/E19-STEADY-STATE/raw/analysis.json),
[E21](experiments/E21-CODEX-COMPONENT/raw/analysis.json),
[E24](experiments/E24-BASE-RECAP/run/raw/analysis.json), and
[corrected auxiliary baseline totals](experiments/E22-CODEX-CUMULATIVE/raw/recompute.json).
