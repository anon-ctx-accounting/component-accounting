# Persistent-context component accounting artifact

## What this is

This artifact contains paired observations of K-ARC, the resident-set manager (data key `karc-full`)
and a persistent retrieval comparator (rag-bm25), collected across Claude Code,
pi and Codex CLI with Anthropic and OpenAI models. It reconstructs uncached input,
cache creation, cache read and output from the recorded counters, then compares
gross input with component-weighted cost. The release supports anonymous peer
review. The verifier recomputes the measured numbers printed in the paper from
released records.

## Quick start

Use Python 3.12 or newer and make. No package installation, API key, provider
account or network access is needed for these commands:

```sh
make verify
make check
make test
```

`make verify` checks 214 printed quantities and reports expected, recomputed,
difference and PASS/FAIL for each. Any mismatch produces a nonzero exit code.
During preparation it took 0.540 seconds with Python 3.14.6; allow a few minutes
on other machines. `make check` scans anonymity patterns, checks file integrity,
and checks anonymous commit metadata. Git is needed for the last check in a
Git checkout; a source zip explicitly skips it because the zip has no metadata.
`make test` runs the standard-library unittest suite.

To regenerate the paper's scatter figure, run `make figures`. See
[reproduction](docs/reproduction.md) for outputs and verification options.

## Layout

```text
README.md  CONFIGURATIONS.md  LICENSE  LICENSE-DATA  Makefile  pyproject.toml
paper-numbers.json          printed quantities, provenance and declared inputs
integrity.json              registered anonymization changes and hashes
configurations/             twelve configuration/group guides
docs/accounting-contract.md component definitions and frozen reference prices
docs/reproduction.md        recorded-data checks and live remeasurement scope
docs/experiments/           eighteen original experiment directories
docs/paper2/figures/        original generators and the generated scatter figure
fixture/e4-v2/              frozen synthetic workload
src/karc/                  resident-set manager and original harness code
scripts/                   verifier, anonymity, integrity and metadata checks
tests/                     artifact tests
```

## Configurations

Every main configuration contains 12 paired sessions of 8 turns per arm. The
following paths retain their original experiment IDs; no data are duplicated
in the configuration guides.

| Name | Recorded inputs | Harness / provider | Window / leg |
|---|---|---|---|
| CC-A-g | [E24-BASE-RECAP](docs/experiments/E24-BASE-RECAP/) | Claude Code / Anthropic | S01–S12 |
| CC-A-s | [E26-CLAUDE-STEADY](docs/experiments/E26-CLAUDE-STEADY/) | Claude Code / Anthropic | S09–S20 |
| pi-A-g | [E18-PI-REVERSAL](docs/experiments/E18-PI-REVERSAL/) | pi / Anthropic | S01–S12, A |
| pi-O-g | [E18-PI-REVERSAL](docs/experiments/E18-PI-REVERSAL/) | pi / OpenAI | S01–S12, B |
| pi-A-s | [E19-STEADY-STATE](docs/experiments/E19-STEADY-STATE/) | pi / Anthropic | S09–S20, A |
| pi-O-s | [E19-STEADY-STATE](docs/experiments/E19-STEADY-STATE/) | pi / OpenAI | S09–S20, B |
| Cx-O-g | [E25-CODEX-BASEWINDOW](docs/experiments/E25-CODEX-BASEWINDOW/) | Codex CLI / OpenAI | S01–S12 |
| Cx-O-s | [E21-CODEX-COMPONENT](docs/experiments/E21-CODEX-COMPONENT/) | Codex CLI / OpenAI | S09–S20 |

[CONFIGURATIONS.md](CONFIGURATIONS.md) gives exact harness/model versions,
warm-up conditions, the auxiliary 16-turn baselines, reuse/budget groups and supporting probes.
Growth/steady describes replay-derived resident snapshots. Physical provider
cache temperature and other execution conditions are separately recorded.

## What can and cannot be reproduced

Recomputation from the retained records is deterministic: the verifier rebuilds
component totals, printed derivatives and paired bootstrap intervals. Ratios
use paired arm sums, with 10,000 common resamples, seed 1313 and sorted draw
indices 250 and 9749. Counter inclusion and cumulative differencing are described
in the [accounting contract](docs/accounting-contract.md).

Live remeasurement incurs provider charges and depends on historical harness
versions, model availability, reference prices, cache initialization and provider
behavior. [Reproduction](docs/reproduction.md) gives recorded time/cost examples
for planning and explains which original scripts require additional temporary
inputs. No live calls are made by verify, check, test or figures.

## What is included

Raw measurements for every configuration the paper reports, the analysis and
execution scripts that produced them, the resident-set manager source, the frozen
fixture, and an English summary for each experiment directory. `verify.py`
recomputes the measured numbers printed in the paper from these records.

The E15/E16/E20 telemetry probes record the counter behavior behind the choice of
a metered Codex CLI route. E10-P2-XR and E10-P2-H16 are retained because printed
numbers are recomputed from those records.

## Known gaps

- The frozen fixture has a first-turn old-gold/newer-revision conflict. Retained
  grades describe that fixture; they do not establish quality equivalence or
  constitute regrading on a corrected workload. Original answer text needed for
  full regrading is not present in the retained main ledgers.
- E18 leg B discarded two attempts and retained the third. E21 reran one
  warm-up unit. E28-O-20 remeasured the H9-affected unit. Their experiment
  summaries and raw records disclose these histories.
- Personal path prefixes and the private hosts in `preload.py` are replaced
  with placeholders; `integrity.json` records every such change. The preload
  code path requires transcript data that these records do not contain.
- Some source comments are in the authors' working language. Original tree-audit
  scripts also scan repository documents, so their results depend on which documents a checkout contains;
  they are outside `make verify`. Historical provenance paths remain in raw
  records and comments.
- Reconstructed amounts use frozen reference or harness catalog prices. They
  are distinct from independent invoice verification or subscription charges.

## Integrity

Original sha256.json manifests remain byte-identical to their source files.
All 67 applied anonymization changes are listed in integrity.json with separate
source and artifact hashes: 3 are covered by the preserved manifests and 64
are outside their coverage. Fourteen references to excluded Markdown files are
explicitly counted as SKIPPED; all other manifest checks pass. The excluded E11
approval checklist has no registry entry. `make check` validates this chain.

## License

Code is licensed under [Apache License 2.0](LICENSE). Data and documentation are
licensed under [Creative Commons Attribution 4.0 International](LICENSE-DATA).
Both use the anonymous attribution “The Authors” for this review release.

## Anonymity

This repository is anonymized for double-blind review.
