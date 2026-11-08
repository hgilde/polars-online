<!-- kind: how-to -->
# Cut a release

This how-to is for the maintainer who releases polars-online: it takes a
checked build on `main` to a verified install from PyPI.

**A release is dispatched, never tagged by hand, and never rehearsed.**
The dispatched run tests everything before it asks for approval, uploads
to PyPI only once approved, and creates the tag after the upload. The
maintainer's word dispatches a release, and the PyPI approval is always
the maintainer's click. [Release readiness](../RELEASE-READINESS.md)
describes each job, what it holds back and why no rehearsal is needed.

## Prerequisites

A release needs these before step 1:

- A clean checkout of `main`, with every change of the release merged and
  its entries under the CHANGELOG's `[Unreleased]`.
- This checkout's build installed in its environment, with
  `uv run maturin develop --release -m crates/online-py/Cargo.toml`: the
  comparison and the benchmarks import whichever `polars_online` the
  environment holds.
- The `gh` command line, signed in with the right to dispatch a workflow.
- The approval of the `Pypi` environment's required reviewer, the owner,
  for step 15.
- A network connection: step 1 installs the last release from PyPI.
- The version number, chosen by [What each kind of release may
  change](../VERSIONS.md#what-each-kind-of-release-may-change). Widening
  the Polars range, for one, is a minor release ([Move the Polars
  range](move-the-polars-range.md)).

## Steps

Nothing leaves your checkout before step 13.

1. **Compare every output with the last release's,** with
   `uv run python scripts/compare_release.py`. It exits 0 when every
   field is the same to the bit, and otherwise prints each field that
   differs with the first row where it does. Declare each difference in
   the CHANGELOG, or find its cause and pin it with a test ([Compare with
   the last release, bit for
   bit](../RELEASE-READINESS.md#compare-with-the-last-release-bit-for-bit)).
2. **Benchmark this build against the last release.** Run this
   checkout's `scripts/benchmark.py` and `scripts/wide_benchmark.py` under
   three builds in turn, with the machine's load logged: this build, the
   last release from PyPI, and the last release built here. A wheel built
   on another machine reads a few percent apart on its own. Each of the
   last release's two gets an environment of its own, so this checkout's
   build stays installed:
   1. Install the release from PyPI: `uv venv <scratch> --python 3.12`,
      then `uv pip install --python <scratch>
      "polars-online[numpy]==<last>"`. The scripts import NumPy, which the
      `numpy` extra brings.
   2. Build the release here: `git worktree add <tag-dir> v<last>`, then,
      inside it, `uv sync` and `uv run maturin develop --release -m
      crates/online-py/Cargo.toml`. Its environment is `<tag-dir>/.venv`.
   3. Run this build with `uv run python scripts/benchmark.py`, and each
      of the others with its own interpreter, as `<env>/bin/python
      scripts/benchmark.py` (`<env>\Scripts\python` on Windows). Run
      `wide_benchmark.py` the same way.
   4. Remove the two, with `git worktree remove <tag-dir>` and by deleting
      `<scratch>`.

   The wide benchmark reaches the solves and products that
   `benchmark.py`'s 50 features never do, where a change can slow a wide
   fit and move no `benchmark.py` row. Fix a slowdown past the runs'
   scatter, or declare it in the CHANGELOG.
3. **Record this build's throughput.** Run
   `uv run python scripts/benchmark.py --markdown` three times or more, and
   take the best run for each row. Paste the rows into [Rows a second for
   each model](../PERFORMANCE.md#rows-a-second-for-each-model) in
   `docs/PERFORMANCE.md`, and set its `runs` row's date, commit and load.
   Add a dated section, headed with the date and naming the commit, to the
   performance log, which `docs/README.md` lists among the records. A test
   holds the table's date to no older than the previous release, and the
   commit to the dated section.
4. **Sign off each instruction count that moved more than 10%.** The
   `Budgets` run of the last push reports each count's drift since the
   last release, read from that release's measurements. Add a
   `[[signoff]]` to `budgets.toml` for each case that moved more than 10%
   either way, or the release's `budgets` job fails ([Budgets that only
   tighten](../TESTING.md#budgets-that-only-tighten)). Each gives the
   case's `key` as the report names it, this release's `version` and a
   `reason`.
5. **Set the version in its six places:** `pyproject.toml`, `Cargo.toml`
   three times, `python/polars_online/__init__.py`, and
   `docs/VALIDATION.md`, regenerated with
   `uv run python scripts/validate.py > docs/VALIDATION.md`.
6. **Refresh the two locks** with `cargo update --workspace` and
   `uv lock`. Each then names the new version.
7. **Promote the CHANGELOG's `[Unreleased]` to `## [X.Y.Z] — <date>`,**
   with the date as `YYYY-MM-DD` and an em dash between. The tag's message
   and the GitHub release are that section.
8. **For a minor release, update the example pin.** The pin under [This
   package's own versioning](../VERSIONS.md#this-packages-own-versioning)
   in `docs/VERSIONS.md` names the new minor, as `~=1.5.0` names the 1.5
   series.
9. **If a wheel's size moved out of the README's install *size* row,
   correct the row.** The release's `wheel-size` job holds every wheel it
   builds to the row, and stops before the upload if one is outside ([The
   README's size row](../RELEASE-READINESS.md#the-readmes-size-row)). To
   read this platform's wheel before then, build it with
   `uv run maturin build --release` and run
   `python3 scripts/wheel_size.py target/wheels/<wheel>`, which exits 1
   when it is outside the row. A wheel of another platform shows only in
   that job, and its fix is a new dispatch.
10. **Check the version,** with
    `uv run --no-project --python 3.12 python scripts/release_version.py --publish`.
    It prints `version=`, `tag=` and `schema=` and exits 0 when all of
    these hold, and otherwise prints what to fix:
    - the six places agree;
    - the CHANGELOG has the version's section, and nothing under
      `[Unreleased]`;
    - the example pin is on the version's minor;
    - the tag is new;
    - no spec or kind in `tests/test_released_state.py`'s
      `PARAMETER_INTRODUCED` or `INTRODUCED` names a newer version.
11. **Run the suite on the floor of the Polars range, in a scratch copy of
    the checkout,** since `uv sync --upgrade-package` rewrites `uv.lock`.
    Make the copy with `git worktree add <dir> HEAD`. It holds the last
    commit, without the uncommitted edits of steps 1 to 10. In the shell
    that runs there, set `UV_PYTHON=3.12`, the Python the release's leg
    runs. Read the
    floor with `python3 scripts/polars_floor.py`, then run
    `uv sync --upgrade-package "polars==<floor>" --upgrade-package "polars-runtime-32==<floor>"`,
    `uv run maturin develop --release -m crates/online-py/Cargo.toml` and
    `uv run pytest -q -m "not soak and not pins and not oracle_grid"`, as
    the release's `floor-polars` job does. It passes, or shows first what
    the release would fail on. Remove the copy with `git worktree remove
    <dir>`.
12. **Run the full suite,** with `bash scripts/gate.sh --extended`,
    unpiped, until its last line says `gate: PASS` with the extended tier.
13. **Commit on `main` itself and push,** not on a branch. Wait for CI on
    the pushed sha to pass.
14. **Dispatch the release,** with
    `gh workflow run release.yml --ref main -f publish=true`. Never push a
    tag by hand, except as [If a job fails](#if-a-job-fails) says, after
    the upload. Push no change to `.github/workflows` until the tag
    exists: the tag job's token may not create a ref whose workflow files
    differ from `main`'s.
15. **Approve the `publish to PyPI` job,** in the `Pypi` environment, once
    every blocking job is green ([The release
    gate](../RELEASE-READINESS.md#the-release-gate)). The tag and the
    GitHub release follow the upload.
16. **Read the `verify` job's three legs.** After the upload, on Linux,
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
version is used up, and no tag exists.

**After the upload, re-run a failed `release` job whatever `main` holds,
and a failed `tag` job unless `main`'s workflow files have changed since
the tested sha.** The `release` job creates no ref: it makes the GitHub
release on the tag the `tag` job pushed. A re-run of `tag` puts the tag on
the same sha, and the release follows it. If `main`'s `.github/workflows`
differ from the tested sha's, the tag job's token is refused on every
re-run. Then make the tag and the release by hand, as the jobs do. Push
the tag with your own credential. A token needs the `workflow` scope, for
the same reason the job's is refused.

1. **Download the run's files,** with
   `gh run download <run-id> -D artifacts`.
2. **Write the CHANGELOG's section for the version to `notes.md`,** under
   the line `polars-online <version>`.
3. **Tag the tested sha,** with
   `git tag -a --cleanup=verbatim v<version> -F notes.md <sha>`, and push
   the tag.
4. **Create the GitHub release** with
   `gh release create v<version> <files> --verify-tag --title v<version> --notes-file <notes>`.
   The files are the run's wheels and sdist, and
   `artifacts/budgets-baseline/budgets-baseline.json`. They are also each
   command-line binary, renamed `<name>-<target>` and keeping `.exe`, as
   the `release` job names it. The `<target>` is the suffix of the
   binary's directory, `artifacts/cli-<target>`. The notes are `notes.md`, a blank line, and the
   run's `artifacts/wheel-sizes/sizes.md`.

**A failed `verify` leg is re-run while PyPI's index lags.** A failure
that is not the index is a release to fix with the next version, since
the upload cannot be undone.
