<!-- kind: how-to -->
# Cut a release

This how-to is for the maintainer who releases polars-online: it takes a
checked build on `main` to a verified install from PyPI, in ten steps.

**A release is dispatched, never tagged by hand.** Dispatched on `main`
with `publish` on, `.github/workflows/release.yml` tests everything first.
It runs CI on all three OSes, and builds every wheel and the command-line
binaries. It runs the state hand-off between OSes, and the suite on the
newest Polars, on the floor of the Polars range and on the newest NumPy.
Then it waits for approval and uploads to PyPI. Only after the upload does
it create the tag `v<version>` on the sha it tested, and the GitHub
release. [Release readiness](../RELEASE-READINESS.md) describes each job
and what it holds back.

**A release is not rehearsed, not even after a change to `release.yml`**
(the maintainer, 2026-11-06). A push to `main` runs CI on all three OSes
and the instruction budgets, and a publishing run tests everything again
before it asks for approval. It stops before the upload, and creates no
tag, when any job fails. A failed release is fixed on `main` and
dispatched again, and no version is used up. The maintainer's word
dispatches a release, and the PyPI approval is always the maintainer's
click.

**Nothing is committed after a release.** What the next release compares
itself with is read from this one where it was published: its state files
from its wheels on PyPI ([State files across
releases](../VERSIONS.md#state-files-across-releases)), its measurements
from an asset of its GitHub release ([The budgets
baseline](../RELEASE-READINESS.md#the-budgets-baseline)).

## Prerequisites

A release needs these before step 1:

- A clean checkout of `main`, with every change of the release merged and
  its entries under the CHANGELOG's `[Unreleased]`.
- The `gh` command line, signed in with the right to dispatch a workflow.
- The approval of the `Pypi` environment's required reviewer, the owner,
  for step 9.
- A network connection: step 1 installs the last release from PyPI.
- The version number, chosen by [What each kind of release may
  change](../VERSIONS.md#what-each-kind-of-release-may-change). Widening
  the Polars range, for one, is a minor release ([Move the Polars
  range](move-the-polars-range.md)).

## Steps

The first six steps are checks and edits made before the commit, and the
last four push, dispatch, approve and verify.

1. **Compare every output with the last release's,** with
   `uv run python scripts/compare_release.py`. Every difference it prints
   is declared in the CHANGELOG, or its cause is pinned by a test
   ([Compare with the last release, bit for
   bit](#compare-with-the-last-release-bit-for-bit)).
2. **Benchmark this build against the last release, and record its
   throughput.**
   - Run `uv run python scripts/benchmark.py` and
     `uv run python scripts/wide_benchmark.py` under the last release and
     under this build, in turn, with the machine's load logged. Run the
     last release both built here from its tag and as its wheel from PyPI,
     since a wheel built on another machine reads a few percent apart on
     its own. The wide benchmark reaches the solves and products
     `benchmark.py`'s 50 features never do: before 1.1.0 it found a
     1,000-feature `ewridge` at 0.56 of 1.0.1's speed with every
     `benchmark.py` row unchanged. Fix a slowdown past the runs' scatter,
     or declare it in the CHANGELOG.
   - Run this build's `benchmark.py --markdown` three times or more, and
     take the best run for each row. It replaces the rows of [Rows a
     second for each
     model](../PERFORMANCE.md#rows-a-second-for-each-model) in
     `docs/PERFORMANCE.md`, and goes into a dated section of that page with
     the date, the commit, the machine and the load. Check the prose around
     the table against the new rows, and measure the headline's share of
     the bank by width again. A test holds the table's date to no older
     than the previous release, since a table nobody re-ran sat three
     releases stale, past its own stated noise.
   - Read the `Budgets` run of the last push: it reports each instruction
     count's drift since the last release, read from that release's
     measurements. A case that moved more than 10% either way needs a
     `[[signoff]]` for this version in `budgets.toml`, or the release's
     `budgets` job fails ([Budgets that only
     tighten](../TESTING.md#budgets-that-only-tighten)).
3. **Set the version in its six places:** `pyproject.toml`, `Cargo.toml`
   three times, `python/polars_online/__init__.py`, and
   `docs/VALIDATION.md`, regenerated with
   `uv run python scripts/validate.py > docs/VALIDATION.md`. Then refresh
   `Cargo.lock` and `uv.lock`.
   `uv run --no-project --python 3.12 python scripts/release_version.py --publish`
   says whether they agree and the tag is new.
4. **Promote the CHANGELOG's `[Unreleased]` to `## [X.Y.Z] — <date>`.**
   The tag's message and the GitHub release are that section.
   `release_version.py --publish` refuses to publish while anything is left
   under `[Unreleased]`.
5. **For a minor release, update the example pin.** The pin under [This
   package's own versioning](../VERSIONS.md#this-packages-own-versioning)
   in `docs/VERSIONS.md` names the new minor, as `~=0.13.0` names the 0.13
   series. `release_version.py` refuses a pin on another minor. The
   README's install *size* row needs an edit only if a wheel's size moved
   out of the range it states. The release's `wheel-size` job holds every
   wheel it builds to the row, and stops before the upload if one is
   outside ([The README's size
   row](../RELEASE-READINESS.md#the-readmes-size-row)).
6. **Run the full suite,** with `bash scripts/gate.sh --extended`, unpiped,
   until its last line says `gate: PASS` with the extended tier.
7. **Commit on `main` itself and push,** not on a branch. Wait for CI on
   the pushed sha to pass.
8. **Dispatch the release,** with
   `gh workflow run release.yml --ref main -f publish=true`. Never push a
   tag by hand. Push no change to `.github/workflows` until the tag exists:
   the tag job's token may not create a ref whose workflow files differ
   from `main`'s.
9. **Approve the `publish to PyPI` job,** in the `Pypi` environment, once
   every job is green ([The release
   gate](../RELEASE-READINESS.md#the-release-gate)). The tag and the GitHub
   release follow the upload.
10. **Read the `verify` job's three legs.** After the upload, on Linux,
    macOS and Windows, it installs `polars-online==<version>` from PyPI into
    a fresh environment, retrying for up to ten minutes while the index
    catches up. It checks that the package came from there, that
    `po.__version__`, `po.native_version()` and `po.schema_version()` are
    the release's, and that an `ewridge` fit over a parquet file finds its
    line's slope (`scripts/verify_release.py`).

## Check the result

The release is done when these hold:

- The three `verify` legs are green.
- The GitHub release carries the wheels, the sdist, the binaries and
  `budgets-baseline.json`, with the wheels' exact sizes under the
  CHANGELOG's section in its notes.

## If a job fails

**Before the approval, fix the cause on `main` and dispatch again.** No
version is used up, and no tag exists. **After the upload, re-run the
failed `tag`, `release` or `verify` job:** the tag goes on the same sha,
and the release is finished where it stopped.

**Some jobs run only in a release, so a release is where they first
fail:**

| job | what it makes or runs |
|---|---|
| `sdist` | the source distribution |
| `build` | every wheel and the command-line binaries |
| `write-state`, `read-state` | the state written on one OS and read on the others |
| `next-polars`, `floor-polars`, `next-numpy` | the suite on the newest Polars, on the floor of the Polars range and on the newest NumPy |
| `baseline` | the instruction baseline |
| `wheel-size` | the wheels' sizes |

Run the floor of the Polars range
locally before the dispatch, in a scratch environment, to find its
failures first. The three jobs after the upload, `tag`, `release` and
`verify`, are tested by `tests/test_release_packaging.py`, which runs the
scripts of the first two against a local remote and a stand-in for `gh`.

## Compare with the last release, bit for bit

**Before a release, every output is compared with the last release's, bit
for bit.** A test with a tolerance lets a small numeric change through,
and a change to the numbers a model returns makes a release a minor.
`uv run python scripts/compare_release.py` installs the newest release
from PyPI into a cached venv under `.cache/release-compare/`. It runs the
workload in `scripts/release_probe.py` under both builds, over 400 rows
with groups, sessions, gaps, nulls and zero weights.

**The workload's specs hold every model, every output a regression can
emit, and the settings that take a path of their own.** Every output is
every field name the memory budgets' fullest workload writes. The settings
are these:

- a temporal clock with duration parameters
- the blocked Gram on two threads
- feature sets, and a coefficient prior
- `kalman`'s shared `P`, reversion and given noise
- every `sgd` loss, `ftrl`'s squared loss and `pa`'s modes
- the windows of `lasso`, `marginal` and `ew_class`
- a formula target, and a drift reset
- the solve and coefficient cadences
- a reset at a step back

`tests/test_release_compare.py` holds the workload to those field names
and those paths.

**The script compares each spec's output fields, and the frame
`closed_groups` drains, bit by bit.** What it does not run is outside the
comparison: the window operators (`po.ewm_mean` and the rest,
`with_windows`), `po.gram`, `po.gbdt`, `refresh_time`, a bank's
`threads=`, and every combination of settings beyond the ones named. It
prints each field that differs, with the first row where it does, and
exits 1. Each run reports which package it imported, so the comparison
cannot compare a build with itself. A spec the old release refuses is
listed as not comparable, not as a difference. `--against 0.9.1` names
another release, and `--report` prints without failing.

**Declare every difference, or treat it as a regression.** A declared one is in
the CHANGELOG, and makes the release a minor. An undeclared one is a
regression: find its cause, and add a test that pins it, before the
release goes out. CI runs the same comparison on every push, report only,
so a change shows at the commit that made it.

**A comparison fails where it should.** Measured on 2026-09-24, against
0.8.0, 112 fields differed, all from 0.9.0's declared readiness gates. On
2026-11-03, 1.4.0's build against 1.3.0 compared 1,283 fields of 72
specs, and 637 differed, each where its CHANGELOG declares a change. The
specs that set a keyword the old release does not take, such as
`huber_refit_rows` against 1.3.0, are not comparable. That comparison
corrected two CHANGELOG entries: the co-moments' sizes, which the released
states had read lower, and the readiness gate's reach into `se_coef`.
