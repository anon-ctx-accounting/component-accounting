# Reproduction

## Recompute the recorded measurements

Use Python 3.12 or newer and make from the repository root. No package
installation, API key or network connection is required. The supplied
pyproject.toml retains the original package configuration; its optional research
dependency groups are not used by these artifact commands.

```sh
make verify
make check
make test
```

The preparation run checked all 214 quantities in 0.540 seconds using Python
3.14.6. Runtime depends on the machine; the target is under five minutes.
Expected and recomputed values, absolute differences and PASS/FAIL are printed
for every item. Any numerical mismatch or unhandled quantity makes verification
fail. Raw inputs are never rewritten by these commands. To save results outside
the repository or restrict the printed checks to one experiment:

```sh
make verify VERIFY_ARGS=--json-output=../verify-results.json
python3 -B scripts/verify.py --experiment E19-STEADY-STATE
```

The catalog covers measured quantities printed in the submitted text, tables and
figure captions, and their printed derivatives. Prices, versions, dates and
experimental/statistical dimensions are declared inputs in
[accounting-contract.md](accounting-contract.md). Preparation recorded the
submission TeX hash and classified every extracted numerical occurrence; the
TeX and private reverse audit are not distributed. Repeat that preparation if
the submission changes.

Ratios use arm totals over paired sessions. One shared deterministic index matrix
supplies 10,000 draws with seed 1313. The only percentile rule uses lower index
int(0.025 B) and upper index ceil(0.975 B)-1: 250 and 9749, zero based. These
intervals reweight the recorded schedule and do not capture repeated provider
runs. Exact counts/totals require equality; rounded values use half the last
displayed unit after the stated unit conversion.

## Integrity and commit metadata

`make check` runs the anonymity scan, original-manifest/registered-change check
and Git metadata check. The first two read every applicable working-tree file
and report excluded-manifest references separately. A Git checkout must contain
one anonymous commit on main, with no tags or other branches and the fixed
release author/committer dates. A clone may include remote tracking copies of
main. When .git is absent, as in the source zip, the metadata step explicitly
reports SKIPPED. The preparation gate instead uses
`python3 -B scripts/check_git_metadata.py --require-git` and must pass before
the archive is created. See [the root README](../README.md#integrity).

## Regenerate the paper figure

```sh
make figures
```

This imports the unchanged make_figures.py, invokes only
render_accounting_scatter, and converts that SVG at width 240 points with the
unchanged svg_to_tikz.py. It produces accounting-scatter.svg,
accounting-scatter.tex and accounting-scatter-body.tex under docs/paper2/figures.
The other renderers target data absent from this artifact and are unrelated to
the paper figure. Generating SVG/TikZ sources needs no TeX installation;
compiling the standalone figure needs the packages named in its generated
preamble.

## Live remeasurement and planning costs

New provider calls are paid. The following recorded waves provide starting
estimates for time and cost under the historical conditions. Amounts are
component-weighted reference USD from the named wave records; they do not
establish an independently verified invoice or current provider pricing.
Each row links to its wall_seconds and wave/running priced USD fields.

| Recorded wave | Wall time, seconds | Reference USD | Evidence |
|---|---:|---:|---|
| pi-A-s warm-up | 55.8 | 0.696936 | [E19 A warm-up](experiments/E19-STEADY-STATE/raw/wave-A-warmup.json) |
| pi-A-s measurement | 99.9 | 4.019379 | [E19 A measurement](experiments/E19-STEADY-STATE/raw/wave-A-measure.json) |
| pi-O-s warm-up | 400.4 | 0.033742 | [E19 B warm-up](experiments/E19-STEADY-STATE/raw/wave-B-warmup.json) |
| pi-O-s measurement | 1,416.7 | 0.148656 | [E19 B measurement](experiments/E19-STEADY-STATE/raw/wave-B-measure.json) |
| Cx-O-s warm-up | 742.0 | 0.066159 | [E21 warm-up](experiments/E21-CODEX-COMPONENT/raw/wave-warmup.json) |
| Cx-O-s measurement | 2,311.1 | 0.234128 | [E21 measurement](experiments/E21-CODEX-COMPONENT/raw/wave-measure.json) |
| Cx-O-g measurement | 2,241.3 | 0.226404 | [E25 measurement](experiments/E25-CODEX-BASEWINDOW/raw/wave-measure.json) |
| OpenAI growth budget 2.5% | 978.5 | 0.093401 | [E28 O-025](experiments/E28-BUDGET-SWEEP/raw/o025/wave.json) |
| OpenAI growth budget 10% | 1,800.8 | 0.226216 | [E28 O-10](experiments/E28-BUDGET-SWEEP/raw/o10/wave.json) |
| OpenAI growth budget 20%, initial wave | 3,456.7 | 0.378996 | [E28 O-20](experiments/E28-BUDGET-SWEEP/raw/o20/wave.json) |
| Anthropic growth budget 10% | 86.2 | 6.564240 | [E28 A-10](experiments/E28-BUDGET-SWEEP/raw/a10/wave.json) |

These rows are individual historical waves. Setup, pilot probes, discarded
attempts, retention waits and later replacement units can add time and cost;
the E28-O-20 initial wave excludes its subsequent H9 replacement. The OpenAI
waves include recorded pacing delays. Full rerun cost cannot be bounded from
these examples alone. The [accounting contract](accounting-contract.md) separates
the reference card from the pi catalog card. Historical CLI/model names,
versions, windows and warm-up choices are in [CONFIGURATIONS.md](../CONFIGURATIONS.md).

For a fresh run, first preserve this release and work in a separate writable
copy. Inspect that experiment's README, configuration and run script; reconstruct
its fixture/schedule and any missing temporary inputs, and supply historical
harness binaries and an isolated authenticated provider environment. The
original scripts under each experiment directory describe the CLI arguments and
preflight checks. The package can be imported directly with `PYTHONPATH=src`.
Path placeholders such as `<REPO>` and `<HOME>` must be resolved for the new local
execution without treating those changes as original evidence. Use fresh run
namespaces and retain the recorded counter interpretation, rate cards, session
pairing, pacing and cache initialization settings. Store new outputs separately
and label them as a new measurement; exact tokens and provider cache behavior
are not deterministic across fresh runs. No remeasurement was performed during
artifact preparation.

## Limits of the retained scripts and evidence

Some original analysis scripts need excluded temporary inputs or overwrite raw
analysis JSON. Use the new verifier for the read-only numerical check. The
retained E15/E16/E20 probes support the metered-route choice without contributing
printed numeric targets; their READMEs show how to inspect their evidence.

Tree-audit scripts such as docs/experiments/E13-RECOMP/scripts/e13_fragility.py
also inspect Markdown and code, so their results depend on which documents a
checkout contains; make verify does not invoke them. Raw JSON
and comments retain paths such as preregistration.md and docs/analysis/...md as
historical provenance. Those paths are not runtime inputs of the released
verifier or the corresponding measurement path; the tree-audit scripts are the
separate case that can actually read documents.

The fixture's first-turn defect and absent answer text prevent treating these
records as a regraded corrected-workload result. E18 leg B's discarded attempts,
E21's warm-up replacement and E28's H9 remeasurement remain part of the disclosed
history. Fresh runs must record their own exclusions and provider behavior.
