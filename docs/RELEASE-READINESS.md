# Release readiness and API stability

This document is for whoever cuts a release of polars-online, or asks which
Polars versions it promises and which were only measured. It says how a
release is cut and what the release workflow checks. It then says which
versions of Polars and NumPy are promised, and how the API is kept stable.
Its last two sections are records: what CI cost while the repository was
private, and how it went public.

**Each dated section reads as of its date.** Where a dated figure has gone
stale, it stays, and the present one follows it with its evidence.

| section | what it covers |
|---|---|
| [Cutting a release](#cutting-a-release) | [the steps](#the-steps-of-a-release) · [the release gate](#the-release-gate) · [rehearse before tagging](#rehearse-before-tagging) · [compare with the last release](#compare-with-the-last-release-bit-for-bit) · [the README on PyPI](#the-readmes-links-on-pypi) · [the tests that hold the workflow](#the-tests-that-hold-the-workflow) · [how the workflow came to this](#how-the-release-workflow-came-to-this) |
| [Which Polars versions are promised](#which-polars-versions-are-promised) | [the floor and the ceiling](#the-floor-and-the-ceiling) · [which interfaces carry a promise](#which-interfaces-carry-a-promise) · [NumPy](#numpy-the-one-optional-dependency) · [raising the ceiling](#raising-the-ceiling-to-a-new-major) · [the pin, and the two copies of Polars](#the-polars-pin-and-the-two-copies-of-polars-2026-08-31) · [2.0.0rc1, measured](#polars-200rc1-measured-2026-09-10) · [the expression plugin](#the-expression-plugin-as-recorded) |
| [Keeping the API stable](#keeping-the-api-stable) | [the policy](#the-policy) · [what not to do](#what-not-to-do) · [what the API is](#what-the-api-actually-is) · [the snapshot test](#s--the-mechanism-one-api-snapshot-test--done) · [the usability round](#api-usability-round-pre-users-window) · [the 2026-09 batch](#the-2026-09-batch-what-it-added-to-the-surface-tasks-4556) |
| [CI cost while the repo was private](#ci-cost-while-the-repo-was-private) | [the policy, as the test holds it](#the-policy-as-the-test-holds-it) · [the quota, spent on day one](#actions-quota-exhausted-on-day-one-2026-08-31) · [where the 80 minutes went](#ci-cost-where-the-80-minutes-went) · [what was unexpectedly expensive](#what-was-unexpectedly-expensive-2026-08-31) |
| [Going public, as recorded](#going-public-as-recorded) | [the suggested order](#suggested-order) · [R1–R6](#before-going-public-r1r6) · [the open-source sweep](#open-source-preparation-the-full-sweep-2026-08-31) · [going public](#going-public-2026-08-31) · [repository settings](#repository-settings-as-recorded) |

## Cutting a release

**A release is dispatched, never tagged by hand.** Dispatched on `main`
with `publish` on, `.github/workflows/release.yml` tests everything first.
It runs CI on all three OSes and builds every wheel and the command-line
binaries. It runs the state hand-off between OSes, and the suite on the
newest Polars, on the floor of the Polars range and on the newest NumPy.
Then it waits for approval and uploads to PyPI. Only after the upload does
it create the tag `v<version>` on the sha it tested, and the GitHub
release. With `publish` off, the same run is
the rehearsal, and it tags and uploads nothing.

**A tag pushed by hand starts nothing, and it uses up its version.**
Release tags cannot be moved or deleted, and
`scripts/release_version.py --publish` refuses a version whose tag exists.
So a version tagged by hand can never be released.

Three things make a release safe. A gate stands in front of the one step
nothing can undo, the upload. The tag exists only once the release does.
And a rehearsal finds a broken workflow before a publishing run does.

### The steps of a release

In order:

| step | how |
|---|---|
| 1. compare | `uv run python scripts/compare_release.py`; every difference declared in the CHANGELOG, or its cause pinned by a test ([Compare with the last release](#compare-with-the-last-release-bit-for-bit)) |
| 2. benchmark | `uv run python scripts/benchmark.py` under the last release's wheel from PyPI and under this build, in turn, with the machine's load logged. A slowdown past the runs' scatter is fixed, or declared in the CHANGELOG. `docs/PERFORMANCE.md` §26 found two slowdowns 0.11.0 had shipped, and §31 ran this step before 0.13.0 |
| 3. released states | every release that added a schema is in `RELEASES` in `tests/test_released_state.py` ([A released build's state files](#a-released-builds-state-files)) |
| 4. version | the version in six places: `pyproject.toml`, `Cargo.toml` three times, `python/polars_online/__init__.py`, and `docs/VALIDATION.md`, regenerated with `uv run python scripts/validate.py > docs/VALIDATION.md`; then `Cargo.lock` and `uv.lock` refreshed. `uv run --no-project --python 3.12 python scripts/release_version.py --publish` says whether they agree and the tag is new |
| 5. changelog | `[Unreleased]` promoted to `## [X.Y.Z] — <date>`; the tag's message and the GitHub release are that section. `release_version.py --publish` refuses to publish while anything is left under `[Unreleased]` |
| 6. README | for a minor release, the example pin under the README's *This package's own versioning* names the new minor, as `~=0.13.0` names the 0.13 series. `release_version.py` refuses a pin on another minor, rehearsing too |
| 7. local gate | `bash scripts/gate.sh --extended`, the full suite, unpiped, until its last line says `gate: PASS` with the extended tier |
| 8. commit and push | on `main` itself, not on a branch; then wait for CI on the pushed sha |
| 9. release | dispatch `release.yml` on `main` with `publish` on (`gh workflow run release.yml --ref main -f publish=true`); never push a tag by hand. No rehearsal first: this run does everything a rehearsal does before it asks for approval, and a failure there leaves nothing to undo -- dispatch again ([Rehearse before tagging](#rehearse-before-tagging)) |
| 10. approve | the `publish to PyPI` job, in the `Pypi` environment, once every job is green ([The release gate](#the-release-gate)); the tag and the GitHub release follow the upload |
| 11. verify | install from PyPI into a clean venv outside the repository, and check `po.__file__`, the three versions the package reports (`po.__version__`, `po.native_version()` and `po.schema_version()`) and a fit |

Which version number a change needs is in [The policy](#the-policy).
Widening the Polars range is a minor release, as step 5 of
[Raising the ceiling to a new major](#raising-the-ceiling-to-a-new-major)
says.

### The release gate

**The one human step in a release is approving the PyPI upload, the one
step that cannot be undone.** The upload is the `publish to PyPI` job,
which declares the `Pypi` environment; `release.yml` spells it
`environment: pypi`. That environment has the owner as a required reviewer,
with self-review allowed, since with a single reviewer, forbidding it would
deadlock every release. The click is a button on the run page, and GitHub
emails when one is waiting.

**The job asks for approval only after every job it needs is green.** Two
pytest markers appear in the table and again below. `soak` marks the
opt-in long runs over millions of rows. `pins` marks the three tests that
assert this repository's own Polars pin, which cannot pass once Polars has
moved (`pyproject.toml`).

| job in `release.yml` | what it checks or makes | holds back the upload |
|---|---|---|
| `version` | the version agrees in its six places, the CHANGELOG has its section, and, publishing, the tag is new and the run is on `main` | yes, and it runs before anything is built |
| `ci` | the whole of `ci.yml`, called: lint, and the Rust and Python suites on Linux, macOS and Windows | yes |
| `sdist` | the source distribution | yes |
| `build` | all six wheels, and the command-line binaries on the five targets that build one, all but musl. The Linux binary is built in the wheels' manylinux2014 image, and refused above glibc 2.17 | yes |
| `write-state`, then `read-state` | the cross-OS state hand-off: a state written on macOS is loaded, and its stream continued, on Windows and Linux | yes |
| `next-polars` | the Python suite, less its `soak` and `pins` tests, in two legs: on the newest Polars the declared range admits, and on the next major | the first leg, the blocking one; the second is advisory |
| `floor-polars` | the same suite on the floor of the declared range, exactly the py-polars and runtime package `pyproject.toml`'s floor names, read by `scripts/polars_floor.py`; a test that needs a newer Polars skips there by version (`tests/polars_version.py`) | yes |
| `next-numpy` | the same suite in two legs, with only NumPy upgraded: on the newest NumPy, which the optional extra's `numpy>=1.26` admits, and on NumPy's next release candidate | the first leg only; the second is advisory |
| `publish to PyPI` | the upload | it is the gate |
| `tag`, then `release` | the tag `v<version>` on the tested sha, annotated with the CHANGELOG's section, then the GitHub release page with the wheels, the sdist and the command-line binaries | no: they run after the upload |

**Everything before the approval can be retried, and nothing after it can
be undone.** That is why the gate sits where it does: in front of the one
step nothing can undo, and in front of nothing else.

| side of the gate | what happens there | can it be undone |
|---|---|---|
| before | dispatching a rehearsal or a release, re-running a failed job, cancelling a run, declining the approval | yes: all of it can be retried, none of it reaches a user, and none of it creates a tag |
| after | the upload, then the tag and the GitHub release | no: PyPI never allows a version number to be reused, even after a yank, and a `v*` tag cannot be moved or deleted. A failed `tag` or `release` job can still be re-run, and tags the same sha |

**A tag that turns out to be wrong is *not* fixed by moving it.** The
`release tags are immutable` ruleset covers `refs/tags/v*` with `deletion`
and `update` rules and an empty bypass list, so the owner is bound too. It
allows *creating* a tag, which is all the workflow does. The fix ships as
the next version, which agrees with PyPI's rule rather than fighting it:
version numbers are single-use on both sides. `v0.11.0` is such a tag: it
names a commit whose CI failed, and nothing was ever published from it.

**One thing is deliberately *not* gated: the Pages deploy.** The deploy in
`ci.yml` republishes the API reference on every push to `main`, and is as
reversible as the next push. `github-pages` briefly carried a required
reviewer on 2026-09-06, and no longer does. The release's call of `ci.yml`
grants the deploy's scopes, since a called workflow may ask for no more
than its caller grants. The deploy runs on a push alone, so under a release
it is skipped (`tests/test_release_workflow.py`).

### Rehearse before tagging

**Rehearse a change to `release.yml` with `publish` off, because a run uses
the workflow file at the commit it runs on.** A fault in the workflow would
otherwise ship inside the released commit. A rehearsal runs the same jobs,
CI included, and tags and uploads nothing. Only publishing is held to
`main`, by the `version` job, so a rehearsal can also try a branch's
workflow before it is merged.

**A release needs no rehearsal first.** Since 2026-09-27 a publishing run
stops before the upload, and creates no tag, when any job fails. A failed
release then means a rerun, not a lost version: its own run is a rehearsal
up to the approval. Neither run can try the two jobs after the upload,
`tag` and `release`, since a rehearsal skips them. If one fails, re-running
it tags the same sha.

The rehearsals of 2026-09-03 found six faults in the workflow, listed in
[The six faults of 2026-09-03](#the-six-faults-of-2026-09-03).

### Compare with the last release, bit for bit

**Before a release, every output is compared with the last release's, bit
for bit.** A test with a tolerance lets a small numeric change through,
and a change to the numbers a model returns makes a release a minor.
`uv run python scripts/compare_release.py` installs the newest release
from PyPI into a cached venv under `.cache/release-compare/`. It runs the
workload in `scripts/release_probe.py` under both builds: 30 specs, every
model among them, over 400 rows with groups, sessions, gaps, nulls and
zero weights. It then compares every output field, and the frame
`closed_groups` drains, bit by bit. It prints each field that differs,
with the first row where it does, and exits 1. Each run reports which
package it imported, so the comparison cannot compare a build with itself.
A spec the old release refuses is listed as not comparable, not as a
difference. `--against 0.9.1` names another release, and `--report`
prints without failing.

**A difference is declared or it is a regression.** A declared one is in
the CHANGELOG, and makes the release a minor. An undeclared one is a
regression: find its cause, and add a test that pins it, before the
release goes out. CI runs the same comparison on every push, report only,
so a change shows at the commit that made it.

Measured when it was written, on 2026-09-24: this build against 0.10.0 and
against 0.9.1, all 291 fields identical. Against 0.8.0, 112 fields differ,
all from 0.9.0's declared readiness gates, which is the evidence it can
fail.

#### A released build's state files

**A state file written by any release so far is refused, with a message
naming the way out: refit from the input.** This build's bank reads the
schemas from `MIN_BANK_SCHEMA_VERSION` (`crates/online-polars/src/bank.rs`)
to its own, `SCHEMA_VERSION` (`crates/online-core/src/lib.rs`), which
`po.schema_version()` gives. The releases wrote older ones: schema 14
(0.10.0), 17 (0.11.1), 19 (0.12.0) and 20 (0.13.0).

`tests/test_released_state.py` holds each release it lists to that. It
installs the release into the comparison's cache, and runs
`scripts/release_probe.py --states` under it: the workload's 30 specs,
fitted on the first half of the stream and saved. This build must then
refuse each file by its version, naming "refit it from its input". It runs
in the suite, and is skipped offline, since it downloads the releases (hard
rule 1 in `CLAUDE.md`).

**Add each release that adds a schema to `RELEASES` there.** It lists
every release from 0.10.0 to 0.13.0. On 2026-10-03 each release saved all
30 specs, and this build refused every file by its version.

**From 1.0, a file a 1.x release wrote loads in every later 1.x and goes on
as this build's own** (task 198; decision D1). The same test holds a 1.x
release's files to that: this build loads each with its own spec, fits the
second half beside a bank of its own that fitted both halves, and every
float, the closed groups, the coefficients and `marginal`'s table agree.
Until 1.0.0 is released it runs on the states this build writes itself, so
the check has run before the first release that must honour it. **On
release day, add 1.0.0 to `RELEASES`**, and every later release with it.

**The frozen fixtures are the in-repo mechanism; the released-state test is
the one against PyPI.** A state of every `ModelState` variant, a bank file,
a `with_windows` state and a `refresh_time` state are frozen at each schema
this build loads, from `MIN_SCHEMA_VERSION` and `MIN_BANK_SCHEMA_VERSION` to
`SCHEMA_VERSION` (both minimums 44 since tasks 194-202), and each is held to
loading, going on to the bit and saving its bytes again
(`crates/online-core/tests/state_fixtures.rs`,
`crates/online-polars/tests/state_fixtures.rs`). Before 1.0 a layout change
regenerates them (`PRINT_STATE_FIXTURES=1`) and raises both minimums; from
1.0 it keeps the previous set beside a loader for it.

Until 2026-09-28 the test held a released file to loading, and the model
to going on as this build's would. This build fitted the second half
beside a bank of its own that saw both halves, and compared every float,
the coefficients, the closed groups and `marginal`'s table. Measured
2026-09-27: 0.11.1's files, at schema 17, went on bit for bit. The files of
0.10.0, at schema 14, went on within 2.2e-13 relative, since schema 14
carries no low parts for the means and they restart at zero. Task 120 then
removed the `on_clock_reset` setting every one of those files names, and
the user said "Do not worry about old specs".

### The README's links on PyPI

**The workflow rewrites the README's relative links for PyPI.** PyPI shows
the README as the project page, where a relative link to another file
resolves against pypi.org and is a 404. So both jobs that build a package
first run `scripts/pypi_readme.py --ref <the tag the run will create>`, or
the sha in a rehearsal. It points each such link at the file on GitHub as
it is at the released commit. The README in the repository keeps its
relative links.

### The tests that hold the workflow

**Three test files hold the workflows to the shape this section
describes.** They run in the suite, so a change to a workflow that breaks
the shape fails before it reaches a release:

| test | what it holds |
|---|---|
| `tests/test_release_workflow.py` | a dispatch is the only trigger, with `publish` off unless asked; the version is checked before anything is built; CI runs inside the release on every OS, and its Pages grant is inert; the upload waits for every job, and the tag for the upload, on the tested sha; the NumPy legs; the floor leg, read from `pyproject.toml`, and the canary's monthly run of it; the floor Python in every leg that moves a dependency; the Linux CLI built in the manylinux2014 image and held to glibc 2.17; only the release jobs write; and `scripts/release_version.py`'s rules |
| `tests/test_release_packaging.py` | the release job's step that collects the files, run for real: each binary named by its target, the Windows `.exe` suffix kept, and no binary sent to PyPI |
| `tests/test_ci_cost_policy.py` | every workflow bounded by timeouts and a concurrency group, releases queueing rather than cancelling, and the rest of the CI policy ([The policy, as the test holds it](#the-policy-as-the-test-holds-it)) |

### How the release workflow came to this

These are records, each read as of its date.

#### Why the tag comes last (2026-09-27)

Until then a release started when a `v*` tag was pushed. 0.11.0 was tagged
on a branch whose 21 commits had never run on Linux or Windows. CI failed
there on the tag, the publish jobs were skipped, and the tag could not be
moved ([The release gate](#the-release-gate)). So the release now runs
everything first, CI included, and a version whose tag exists is refused
before anything is built (`scripts/release_version.py`).

The gate's list of jobs grew with the workflow. On 2026-09-06 it read
`needs: [sdist, build, read-state]`. `next-polars`, which tests the newest
Polars at the release, joined it on 2026-09-10 (`e6da97c`). `version` and
`ci` joined on 2026-09-27, and `next-numpy` on 2026-09-30.

#### The six faults of 2026-09-03

Six faults were found on 2026-09-03, and every fix is in the commit that was
tagged. The first was found before the first rehearsal ran, and is one a
rehearsal would have caught. The rehearsals found the other five. The fourth
rehearsal, with every earlier fix in, passed the state hand-off on both OSes
and found the sixth.

| # | where | what failed | why | the fix |
|---|---|---|---|---|
| 1 | the Intel macOS wheel | `release.yml` named the `macos-13` runner | GitHub has retired it: a tag would have built five wheels and published none | the label is `macos-15-intel` now |
| 2 | the aarch64 Linux wheel | cross-compiled from x64 in `manylinux2014-cross:aarch64`, it failed in `ring` 0.17, reached through polars-io's cloud feature | that image's GCC 4.8.5 does not define `__ARM_ARCH` (briansmith/ring#1728) | built the way polars builds its own: natively on GitHub's `ubuntu-24.04-arm` runner, where maturin-action picks the `manylinux2014_aarch64` image with a current GCC. The tag stays `manylinux_2_17`, and the native runner means the aarch64 CLI binary now ships too |
| 3 | the x64 Linux job | the host `cargo build` of the CLI could not execute its compiler wrapper | maturin-action exports `RUSTC_WRAPPER=sccache` into the job, but installs sccache inside the manylinux container | the CLI step clears the wrapper |
| 4 | the Intel macOS job | it hit the 90-minute timeout | 26 minutes of it were `uv sync` building the extension into a venv nothing in the job uses: maturin-action brings its own Python, and the CLI has no pyo3 in its tree | that step is gone, the timeout is 120, and rust-cache saves on failure, as it does in `ci.yml` |
| 5 | both `read state` jobs | `handoff.state: No such file or directory` | the artifact lands in the workspace root, but cargo runs an integration test with the *package* root as its working directory (`crates/online-polars/`, per the cargo reference) | the read side uses `${{ github.workspace }}`, as the write side already did; reproduced locally both ways before the fix |
| 6 | the Linux wheel jobs | the host `cargo build` for the CLI failed on `target/release/.cargo-build-lock` (Permission denied), and rust-cache's post step could not tar the tree | the wheel is built inside maturin-action's manylinux container, which runs as root over the bind-mounted workspace, so `target/` comes back root-owned | a `sudo chown -R` of `target` between the two steps fixes both |

**The fifth matters most, because it stopped the check of hard rule 5
itself.** That rule, in `CLAUDE.md`, says a state file loads on both OSes,
and the `read state` jobs are where a release checks it. The state jobs
also lost their `uv sync` (17--23 minutes each): `online-polars` has no
pyo3 in its tree, and the test is pure Rust. The sixth is why the Linux
jobs had never once restored a cache. Rehearsal one never reached it,
because the sccache wrapper failed first and hid it.

#### The Linux CLI's glibc floor (2026-09-27)

**0.11.1's Linux CLI binaries needed glibc 2.39, the build runner's own.**
Measured on 0.11.1's release assets on 2026-09-27, from each binary's
version requirements (`.gnu.version_r`), for x86_64 and aarch64 alike:

| requirement | from | binding |
|---|---|---|
| `GLIBC_2.34` | `__libc_start_main` and the pthread functions | hard |
| `GLIBC_2.35` | `hypot`, in libm | hard |
| `GLIBC_2.39` | `pidfd_getpid` and `pidfd_spawnp`, which Rust's standard library uses to spawn a process | weak symbols, but the version requirement is not marked weak, so the loader refuses the binary without it |

So that CLI started on Ubuntu 24.04, Debian 13 or newer, and not on Ubuntu
22.04 (2.35), Debian 12 (2.36) or RHEL 9 (2.34). This was read from the
files, not run on an older glibc: the machine that measured it has no
container runtime. The binaries were built on the runner's own image, not
in the manylinux2014 container the wheels come from (the wheels are
`manylinux_2_17`). So the floor followed the image, and would have risen
when `ubuntu-latest` moved to 26.04.

**Since 0.12.0 the Linux CLI is built in that manylinux2014 image, at
glibc 2.17** (the user's decision, 2026-09-28; [PLAN](PLAN.md) task 115
(i)). `release.yml`'s build job runs `cargo build` in
`quay.io/pypa/manylinux2014_<arch>`, the image for the runner's own
architecture. Rust is installed inside it as maturin-action installs it
there. It is a `docker run` step, since the image's glibc is too old for
the actions' JavaScript to run as a job container. `scripts/glibc_floor.py`
then reads the binary's version requirements and fails the job above 2.17,
before anything is uploaded. On 0.11.1's x86_64 binary it reports 2.39 and
refuses it. The 0.12.0 rehearsal was the first run to build it (run
36448809775 on `36e468c`). The binaries need glibc 2.16 on x86_64 and 2.17
on aarch64, and 0.12.0 shipped them.

## Which Polars versions are promised

**The declared range, `polars>=1.34.0,<3`, is this project's measured
claim, and Polars does not promise it.** It rests on the measurements in
this section:

| part | what it rests on | where |
|---|---|---|
| the floor, 1.34.0 | `LazyFrame.collect_batches`, which py-polars added in 1.34.0; the suite on 1.34.0 in the blocking `floor-polars` leg of every release and in the canary each month; `tests/test_scaffold.py` pins the declared floor | [The floor and the ceiling](#the-floor-and-the-ceiling) |
| `ModelBank` alone, from 1.28.1 | `PySeries._export`, measured on 17 py-polars releases | [the matrix](#statically-linking-polars-will-break-users-on-other-versions) |
| the ceiling, `<3` | the Python suite on `2.0.0rc1`, and the blocking leg of `release.yml` on the newest version in range, run in every dispatched release before any tag exists | [Polars 2.0.0rc1, measured](#polars-200rc1-measured-2026-09-10) |
| what Polars itself promises | nothing for `ModelBank`, the IO plugin or the serialized form of an expression; the Arrow PyCapsule output is Arrow's contract | [Which interfaces carry a promise](#which-interfaces-carry-a-promise) |
| the Rust `polars` | 0.55.2, pinned by `Cargo.toml` and linked into the wheel, so it never meets the user's | [the matrix](#statically-linking-polars-will-break-users-on-other-versions) |
| the `online` CLI | the Rust `polars` 0.55.2 alone: it never touches py-polars | [The other failure](#the-other-failure-is-the-canarys-own-hygiene) |
| NumPy, the optional extra | `numpy>=1.26`: the newest NumPy blocks a release | [NumPy, the one optional dependency](#numpy-the-one-optional-dependency) |

### The floor and the ceiling

**The floor is 1.34.0, the release that added `LazyFrame.collect_batches`.**
Every surface that reads a query in chunks reads with it: `ModelBank.fit(lf)`
and `fit_predict_batches(lf)`, `lf.online.fit_predict` and
`lf.online.predict`, `with_windows` in each of its forms, and
`po.stream.refresh_time`. `ModelBank` alone works from 1.28.1, as
[the matrix](#statically-linking-polars-will-break-users-on-other-versions)
shows. `tests/test_scaffold.py` pins the declared floor, so a change to
either has to change both.

**Every release runs the whole suite on the floor, and the canary runs it
on the 1st of each month** (review 2026-10-06, AP7 and CI2). The
newest-Polars runs prove the ceiling and say nothing of the floor, and the
whole suite had last run on 1.34.0 on 2026-09-27, before the window
operators were built. `scripts/polars_floor.py` reads the floor from
`pyproject.toml` for both runs, so a raised floor moves them with it. Only
polars and its runtime package move, and the run refuses to go on unless
the installed py-polars is the floor. A test that needs a newer Polars
skips there, naming the version and why, through `needs_polars` in
`tests/polars_version.py`. It refuses a version at or below the floor,
where it would skip nothing, and one above the pin, where it would hide the
test from CI's own runs.

| run | when | a red run |
|---|---|---|
| `floor-polars` in `release.yml` | every dispatched release, rehearsals included, before any tag exists | withholds the release until the code is fixed or the floor raised |
| `floor-polars` in `polars-canary.yml` | the 1st of each month, and on demand | is the notification, and withholds nothing |

**How the floor was found (2026-09-02).** The declaration that day read
`polars>=1.34.0,<2`. The matrix measured `ModelBank` and the expression
plugin, and those do work from 1.28.1. But `po.run` over a path or a plan
read with `LazyFrame.collect_batches`, and so did the streaming query form
`lf.online.fit_predict` (E32 and E33 in `docs/ENHANCEMENTS.md`). py-polars
added `collect_batches` in **1.34.0**. On 1.28.1–1.33 both failed with
`AttributeError: 'LazyFrame' object has no attribute 'collect_batches'`, in
24 of 27 runner tests. The canary, the weekly job that runs the suite on the
newest Polars, could not see this, since it tests nothing older. It was
found while checking E33 against the floor, and the floor now says what the
package needs. The whole suite (1037 tests) passed on 1.34.0, 1.38.1 and
1.44.1 with identical numbers.

**The whole suite on 1.34.0 again, 2026-09-27: one bug in the package,
fixed.** A venv with the locked dev and docs groups, `polars` and
`polars-runtime-32` at 1.34.0, and this build's wheel ran 3,425 tests:
3,408 passed and 17 failed.

| failures | cause | now |
|---|---|---|
| 4 | `po.eval.seqtest` with `by`, the README's example among them: a window inside a window, which 1.34.0 refuses ("window expression not allowed in aggregation") | fixed: two passes, the same arithmetic; all 208 of those files' tests pass on 1.34.0 |
| 1 | `scripts/compare_release.py` passed `explode(empty_as_null=True)`, a keyword 1.34.0 has not got | fixed: the keyword where it exists |
| 8 | `pl.scan_arrow_c_stream`, in py-polars from 1.43.0: the DuckDB, ADBC and pyarrow paths' tests and examples | the docs say 1.43.0 ([ARROW-SOURCES](ARROW-SOURCES.md)); the package itself only names it in a warning |
| 2 | durations: 1.34.0's parser refuses a leading `+`, which the text oracle compares against, and its `pl.duration` wraps past an i64 before the package can see it | recorded: the promise that a `pl.duration` past 292 years is refused holds on a newer polars |
| 2 | the development environment's version pin (`tests/test_scaffold.py`) and `docs/VALIDATION.md`, which records 1.44.2 | expected |

`po.run` has gone since, and the floor has not moved. It left Python in
task 83 (`7d23a80`, 2026-09-17), and the command line kept the runner.

**The whole suite on 1.34.0 again, 2026-10-07: nothing in the package
failed.** A venv made the same way, with this build's extension, at
`4d6ae0f` with task 199's changes, ran 4,711 tests, and 24 failed on
Polars' version. Each was then run on sixteen releases between 1.34.0
and 1.44.2, every minor among them, to find where it first passes. 1.35.0
and 1.36.0 cannot be installed, since their runtime packages are not on
PyPI. Every failure is a test's reference, or an
example, that needs a newer Polars:

| tests | first passes on | why |
|---|---|---|
| 9: the DuckDB, ADBC and pyarrow paths' tests, both database examples and the README's database block | 1.43.0 | `pl.scan_arrow_c_stream`, as the README says |
| `test_at_a_repeated_stamp_every_row_carries_the_stamps_total` | 1.43.0 | its reference, `Expr.ewm_sum_by` |
| 10: `test_which_rows_a_window_holds_is_polars_rolling` under `left` and `none`, and all 8 cases of its long random stream | 1.41.1 | its reference, `rolling_sum_by`, refuses a column with a null before it, or gives null for an empty window where it now gives 0, and a value where it now gives null |
| `test_a_pl_duration_past_an_i64_is_refused_by_name_and_never_wraps` | 1.38.0 | `pl.duration` wraps past an i64 inside Polars, before the package sees it, as on 2026-09-27 |
| `test_a_row_on_the_far_edge_counts_but_weighs_nothing` | 1.37.0 | its reference, `rolling_mean_by` over an integer index, refuses a column with a null |
| `test_what_is_not_element_wise_is_refused_by_name` | 1.36.1 | the package refuses `.over()` by the node's name, which Polars spells `Window` before 1.36.1 and `Over` from it |
| `test_the_text_means_what_polars_reads_it_as` | 1.35.1 | its reference, `dt.offset_by`, refuses a leading `+`, as on 2026-09-27 |

The refusal of `.over()` holds on the floor, so that test now expects the
name the installed Polars gives, and runs. Every other is marked with
`needs_polars` and the version in the table. With the marks, the suite on
1.34.0 passed: 4,687 tests passed and 25 skipped, 23 of them by version
and two that test a Windows-only path spelling.

**The ceiling, `<3`, is a bet that the interface holds through 2.x.** On
2026-09-02 it was `<2`, a bet on 1.x. Since 0.5.1 it is `<3`, made on the
measurements in
[Polars 2.0.0rc1, measured](#polars-200rc1-measured-2026-09-10). Two runs
hedge it:

| hedge | when | what it runs | a red run |
|---|---|---|---|
| the blocking leg of `release.yml`'s `next-polars` | every dispatched release, rehearsals included, before any tag exists | the Python suite on the newest version the range admits | withholds the release until the range is capped or the code fixed |
| `polars-canary.yml`, the canary | weekly, on Mondays, and on demand | the Rust tests and the Python suite on the newest py-polars, prereleases included, with the range dropped | is the notification, and withholds nothing |

A third hedge went with task 85. The expression plugin's ABI was
version-negotiated, and refused to load rather than misbehave. On the paths
that are left, a break is a loud failure rather than a negotiated one. A
missing `PySeries._export` is a clean `AttributeError` before any data
moves.

### Which interfaces carry a promise

**The range is measured, and the interfaces it rides on carry different
promises, mostly none.** This came from reading what Polars actually
promises:

| interface | used by | what Polars promises | a break would show as |
|---|---|---|---|
| pyo3-polars' extension types, `PyDataFrame`/`PySeries` | `ModelBank` | no stability guarantee beyond the latest definitions working with the latest version | a clean `AttributeError` before any data moves |
| the IO plugin, `polars.io.plugins.register_io_source` | `lf.online.fit_predict`, `lf.online.predict`, `with_windows` and `po.stream.refresh_time` | documented in the user guide, but decorated `@unstable` in polars (a warning only under `POLARS_WARN_UNSTABLE`); its contract is partly unwritten | a wrong row count, not a crash |
| an expression's serialized form, `expr.meta.serialize(format="json")` | the window formulas, read into this library's own tree when a spec or a `with_windows` call takes them | nothing: the form is unstable across Polars versions | an unknown node, refused by name when the formula is read |
| the Arrow PyCapsule interface | `ModelBank.fit_predict_arrow` and `predict_arrow` (task 86) | nothing: its contract is Arrow's rather than polars', and it covers the output side only | |

**`ModelBank` carries no guarantee.** The pyo3-polars README says the
`PyDataFrame`/`PySeries` types "are however only provided for convenience
and **do not have stability guarantees beyond that the latest definitions
should work for the latest version of Polars**". That is exactly the
`PySeries._export` call whose absence sets the 1.28.1 floor of the matrix.

**The unwritten part of the IO plugin's contract is its pushdowns.** Polars
pushes a projection, a predicate and a slice into a Python source, and does
not re-apply any of them afterwards. So the source honours all three itself
(`python/polars_online/_frame.py`, and `stream.py` for `with_windows` and
`refresh_time`). `tests/test_frame.py` checks every pushdown against the
collected frame, in both orders. A change in that contract would show as a
wrong row count, not a crash, which is why those tests exist and why the
canary runs them.

**The serialized form is read narrowly, because it is not stable.** Seven
node kinds are read, when the formula is taken, under the caller's own
Polars, and an unknown node is refused by name. Python and Rust then
rebuild the formula through Polars' public builders
(`python/polars_online/_formula.py`).

**The range stays, because it is measured**: the numbers are identical
across the releases, and the failure mode is a loud `AttributeError` rather
than a wrong answer. But it is this project's empirical claim, not Polars'.
A `ModelBank` break on a new Polars is expected maintenance: check that path
first, and the IO-plugin tests after it.

### NumPy, the one optional dependency

**The newest NumPy blocks a release, and its next release candidate is
advisory.** NumPy is needed only by the `numpy` extra,
`polars-online[numpy]`, for `po.sim`, `po.corr`, `po.gram` and
`ModelBank.gram`. Every other job runs on the locked NumPy. So
`release.yml`'s `next-numpy` runs the Python suite with only NumPy
upgraded, in two legs:

| leg | resolves to | blocks the publish |
|---|---|---|
| the newest NumPy | the newest stable that `numpy>=1.26` admits; the extra has no ceiling | yes |
| NumPy's next release candidate | a release candidate while one is out, and the newest stable again otherwise | no |

The canary's second job, `next-numpy`, runs the advisory leg every week.
Only NumPy moves in these runs, so a red one names it.

**The floor, `numpy>=1.26`, follows the oldest Python this package
supports: NumPy's first wheels for Python 3.12 are 1.26.0.**
`tests/test_release_workflow.py` holds the floor to that first wheel. No
run on NumPy 1.26 is recorded.
With no ceiling, a NumPy that breaks this library blocks
every release until it is fixed or capped. The Polars range makes the same
trade (`docs/PLAN.md` task 142).

### Raising the ceiling to a new major

The ceiling is `<3`, raised from `<2` in 0.5.1, so a py-polars 3.0 is
excluded until this is done again. The steps, in order:

1. **The canary and the release job's advisory leg are already testing
   it.** Both unpin, and both allow prereleases, so a release candidate of
   the next major is exercised the week it appears. Read the last run
   before starting. Raising the ceiling is also what moves that major under
   the *blocking* leg, since that leg tests the range as declared.
2. **Check the Rust side separately.** py-polars' major and the `polars`
   crate's version are independent. As of 2026-09-10, py-polars is at
   2.0.0rc1, while the newest crate is 0.55.2, the one `Cargo.toml` pins. A
   Python major on its own needs no Rust change. The wheel carries its own
   statically linked copy, and the two never meet: data crosses on the
   Arrow C Data Interface.
3. **Widen the range in `pyproject.toml`**, today `polars>=1.34.0,<3`, and
   let the lock resolve. That is the only required source change if steps
   1–2 are clean.
4. **Re-run `docs/VALIDATION.md`** with
   `uv run python scripts/validate.py > docs/VALIDATION.md`. Its header
   records the Polars version it was generated with, which is why its test
   is marked `pins`.
5. **Ship it as a minor**, not a patch. Widening the Polars range is a
   minor release by this package's own rule (the README, *This package's
   own versioning*).

**`<3` was raised on what the 2.0 candidate measured, not on a guess that
2.x keeps the interface.** The whole suite passes with the same numbers as
on 1.44. All three interfaces of the time work: the expression plugin,
`ModelBank` and the IO plugin. `LazyFrame.collect_batches`, the floor, is
unchanged. One behaviour moved in our favour: a query that fails *after*
the bank now stops the source instead of draining it, so `save_state` is
not written on a long stream. That narrows the gap `docs/STATE-WORKFLOW.md`
calls R6. The measurements are below, in *Polars 2.0.0rc1, measured*.

**It was raised on `2.0.0rc1`, before 2.0.0 final was on PyPI, and that was
deliberate.** It changes nothing for a user today, because installers do
not resolve to a release candidate. So every user still gets the newest 1.x
until 2.0.0 ships, and then gets it without waiting on a release of ours.
The blocking leg is what covers the difference between the candidate and
the final.

### The Polars pin, and the two copies of Polars (2026-08-31)

Two related worries, both worth answering with evidence rather than
architecture talk: both sound alarming, and only one is real.

#### "Statically linking Polars will break users on other versions"

**It does the opposite, and the range is measured.** Every wheel carries its
own Rust Polars 0.55.2, which never meets the user's. Data crosses between
the two on the **Arrow C Data Interface**, a language-independent ABI whose
carrier is `SeriesExport`, a `#[repr(C)]` struct of
`ArrowSchema`/`ArrowArray` pointers.

One wheel was tested against 17 py-polars releases, through both entry
points of the time, checking values and not just the absence of exceptions:

| py-polars | `ModelBank` | expression plugin |
|---|---|---|
| 1.26.0 | `AttributeError` | `ComputeError: error loading dynamic library` |
| 1.27.1 | `AttributeError` | OK |
| 1.28.0 | *bug in polars itself* (`NameError: PySeries`) | — |
| **1.28.1 – 1.44.1** | **OK** | **OK** |

The numbers are identical everywhere it works, and both failure modes are
clean named exceptions: no segfault, and no silent wrong answer. The floor
of *these two* is `PySeries._export`, which pyo3-polars calls and py-polars
added in 1.28.1.

**So the exact pin was protecting nothing the ABI does not already
protect,** and it blocked resolution for anyone wanting a different polars.
Reproduce with:

```sh
uv venv /tmp/v && uv pip install --python /tmp/v/bin/python --no-deps dist/*.whl
uv pip install --python /tmp/v/bin/python 'polars==1.34.0'
```

#### "A frame allocated by one binary and freed by the other"

**A `SeriesExport` carries a `release` callback into the binary that
produced it, so each side frees its own memory with its own allocator.**
This is the genuinely dangerous version of the question, and the reason the
C Data Interface is the right mechanism. No Rust `DataFrame`, no `Drop` impl
and no raw buffer ownership ever crosses. This was confirmed in `polars-ffi`
0.55.2's source, where `import_series` goes through Arrow's `import_array`.
It was also stress-tested, and came back clean: 300 round-trips of
5,000-row frames, plus 50 outputs deliberately outliving the inputs they
came from.

**It did surface a real gap, and a bigger one than first written up here:
we were not installing `PolarsAllocator`.** pyo3-polars ships it to route a
plugin's allocations through py-polars' own allocator, via its
`polars.polars._allocator` capsule. This was first recorded as a
performance nicety. It is not one: Polars' own canonical plugin example
opens with

```rust
use pyo3_polars::PolarsAllocator;
#[global_allocator]
static ALLOC: PolarsAllocator = PolarsAllocator::new();
```

so it is **part of the prescribed setup**. It is the piece that keeps
allocation coherent between the two copies of Polars, and a large part of
why a statically linked plugin is safe rather than a latent double-free. We
had been running on a second, independent heap. Installing it also
**gained 5–43% throughput**, which was not the reason for doing it: +43% at
k=5, +16% at k=20 and +5% at k=50 (`docs/PERFORMANCE.md` §6).

#### "Why not dynamically link Polars?"

**Nothing links dynamically against Polars' Rust API, but three Rust→Rust
bindings were resolved at runtime, all of them across a C ABI.** This is
worth stating precisely, because "there is no dynamic link" is too strong,
and the truth is more interesting. One of the three went with the
expression plugin in task 85.

**What is *not* dynamically linked, verified on both sides:**

| binary | what it exports or links | Polars symbols |
|---|---|---|
| py-polars' runtime, a 189 MB binary | **33 dynamic symbols**: three `PyInit_*` entry points, plus incidental C symbols from vendored blake3 and crc libraries | **zero** Polars Rust symbols: there is nothing to bind to |
| our extension, a `cdylib`, which statically links its whole Rust graph by construction | `otool -L` shows it linking only system frameworks, libc++, libiconv and libSystem. Of its 298 undefined symbols, 104 are the CPython C API, 27 macOS frameworks, and the rest libc/libc++ | **none** is Polars- or Arrow-related |

**And it could not work anyway, because Rust has no stable ABI.** Symbol
names carry a hash over the compiler version, crate version, features and
the entire dependency graph. Binding to them would demand a toolchain and
dependency graph bit-identical to the py-polars wheel's. It would break on
every polars patch release and every rustc bump, which is *far* tighter
coupling than static linking. The 1.28.1–1.44.1 range above exists precisely
because we do not do it.

**What *is* resolved dynamically, Rust to Rust, goes through C-compatible
function pointers:**

| # | binding | how it works | today |
|---|---|---|---|
| 1 | **the plugin entry points** | py-polars `dlopen`s our `.so` and `dlsym`s `_polars_plugin_online_run`, `_polars_plugin_field_online_run`, `_polars_plugin_get_version` and `_polars_plugin_get_last_error_message`. That is py-polars calling into our Rust at runtime | gone with the expression plugin in task 85 |
| 2 | **the `release` callbacks** inside every `SeriesExport` | a function pointer into the *producing* binary, invoked by the consumer. This is what makes two copies of Polars safe rather than a double-free | in use |
| 3 | **the allocator capsule** | `PyCapsule_Import("polars.polars._allocator")` hands us a struct of four function pointers into py-polars' binary, which then serve every allocation this extension makes | in use, and the one fragile part (below) |

So the dynamic interop is real, but at the C level, where there is an ABI
to rely on, instead of at the Rust level, where there is not.

**The plugin's binding was a designed, versioned contract.** It is worth
spelling out, because "they dlsym a C function" undersells it. The
expression plugin that used it was removed in task 85, so this is the
record of how it worked:

| part | what it was |
|---|---|
| the version | `polars_ffi` exports `MAJOR: u16 = 0` and `MINOR: u16 = 1`, and `_polars_plugin_get_version()` packs them into a `u32` as `(major << 16) \| minor` |
| the loader | `polars-plan` reads that version *before making any other call*, and **branches on MAJOR**: `if major == 0 { use polars_ffi::version_0::*; … }`. The module is literally named `version_0`. When the calling convention changes, polars adds `version_1` and a second branch, and plugins built against the old one keep working. Multi-version support is built into the loader |
| the call | pure C: `extern "C" fn(*const SeriesExport, usize, *const u8, usize, *mut SeriesExport, *const CallerContext)` |
| the payload | the **Arrow C Data Interface**: a formal cross-language specification, with its own stability guarantees and ownership protocol. It is the same one DuckDB, pyarrow and cuDF interoperate through |

That is the textbook way to cross a binary boundary you cannot recompile:
drop to C, version the contract explicitly, and let each side own its own
memory. The 1.28.1–1.44.1 range is the evidence that it works.

**What Rust genuinely cannot offer here, and why nobody ships this
differently.** Rust has no stable ABI, and its symbol names say so out loud.
Here is one method from our own crate:

```
_RNCNCNvMs_NtCskWXLaxruzD7_11online_core6kalmanNtB8_6Kalman8pred_var00Ba_
                 ^^^^^^^^^^^^ crate disambiguator
```

`CskWXLaxruzD7_` is a hash over the crate's name, version, features and
`-C metadata`. Change any of them, and every one of the 3,097 mangled
symbols in that one small rlib gets a new name. `crate-type = ["dylib"]`
*does* exist, and does preserve the Rust ABI: rustc uses it for
`librustc_driver.so`. But it requires the identical compiler and dependency
graph on both sides, which is why rustup ships matched toolchain components
instead of letting you mix them. It is workable inside one build, and not
workable between two wheels published months apart by different people.

**The one part that genuinely is fragile is binding 3, and it is
pyo3-polars' own choice rather than anything about the plugin ABI.**
`polars.polars` is *not* an importable submodule: `importlib.import_module`
raises. It is an **attribute** of the `polars` package, aliasing the real
runtime module, currently `_polars_runtime_32._polars_runtime`.
`PyCapsule_Import` walks dotted names by import-then-getattr, which is the
only reason the lookup succeeds. And `PolarsAllocator` **falls back to the
system allocator without erroring** when it fails, so a rename would
silently lose 43% throughput. `test_scaffold.py` asserts the resolution path
for exactly that reason.

### Polars 2.0.0rc1, measured (2026-09-10)

`polars 2.0.0rc1` is on PyPI, and it is the only 2.x release; stable is
1.44.2. **The suite was built and run against it, and neither failure it
found is a break in the library.** The run used a throwaway worktree with
its own `CARGO_TARGET_DIR`. It pinned the Python dependency, then ran
`uv sync`, `maturin develop --release`, and
`pytest -m "not soak and not pins"`.

**Result: 2,438 passed, 2 failed, 3 skipped.**

**There is no Rust-side 2.0.** crates.io's newest `polars` is still 0.55.2,
the version `Cargo.toml` pins. py-polars 2.0 is a Python-side major only, so
moving the ceiling is a `pyproject.toml` change, not a Rust upgrade. That is
the split the canary's own comment predicts, and the reason the wheel's
statically linked copy never meets the user's.

**All three interfaces of the time work.** The expression plugin loads,
since its ABI handshake passes, and it still warns.
`PySeries._export`/`_import` are both present, so `ModelBank` works, and the
IO plugin runs under `collect()` and `sink_parquet()`.
`LazyFrame.collect_batches`, the floor, has an identical signature in 1.44.1
and 2.0.0rc1, `chunk_size` and `maintain_order` included.

#### R6 narrows, as `docs/STATE-WORKFLOW.md` said it might

**`docs/STATE-WORKFLOW.md` §4 states its rule R6, the one gap in the
plan's state workflow.** Polars drains a Python source before surfacing a
later node's error. So a query that fails *after* the bank still writes
`save_state` with the whole stream's state, while its own output is
missing. The stability note in that document's §6 says "R6 narrows if
polars ever stops it". That R6 is the state workflow's; this document's
own R6 is the history scan.

**On 2.0.0rc1 it does narrow, and the narrowing is length-dependent.**
Measured with `chunk_size=500` and a failing downstream cast:

| rows | chunks | 1.44.1 writes the state? | 2.0.0rc1 |
|---:|---:|---|---|
| 4,000 | 8 | yes | yes |
| 40,000 | 80 | yes | **no** |

So 2.0 propagates the cancellation to the source once the stream is long
enough that the failure lands before the source is exhausted. A short
stream is still drained.
`tests/test_frame.py::test_a_run_that_does_not_reach_the_end_writes_nothing`
asserted the 1.x behaviour at 40,000 rows, and was the first of the two
failures. The test was right about 1.x, and needed a version-aware
assertion to hold on both.

**That is a narrowing of a documented hazard, not a regression.** The
dangerous case is the state being written when the output was not, and 2.0
does that less often. The `online` CLI remains the transactional call
either way.

#### The other failure is the canary's own hygiene

`tests/test_validation_doc.py` regenerates `docs/VALIDATION.md` and compares
it. The document's header line records the Polars version it was generated
with: `1.44.1` committed, against `2.0.0-rc.1` produced, so the test is
version-sensitive by construction. It carried no `pins` marker, so the
canary's `-m "not soak and not pins"` did not deselect it. Marking it `pins`
would make the canary's signal mean only "Polars broke us", which is what
that job exists for.

**Both failures are closed since `e6da97c`, so the paragraphs above record
the argument, and this one the state.** `tests/test_validation_doc.py` now
carries `@pytest.mark.pins`, and the canary deselects it.
`test_a_run_that_does_not_reach_the_end_writes_nothing` became version-aware
about the R6 narrowing, rather than asserting 1.x alone. Re-measured on
2026-09-18 against `2.0.0-rc.1`, still the only 2.x on PyPI with stable at
1.44.2: **2,679 passed, 2 skipped, 5 deselected, 0 failed.**

**Every test that *can* run on 2.0 did.** The first pass of that measurement
read 2,651 passed and **30 skipped**, and the 28 extra skips were not a
result. They were `pandas`, `statsmodels`, `filterpy` and
`bayesian_changepoint_detection` missing from the throwaway venv, so the
oracle comparisons never executed. Installing them turns those 28 into
passes, and leaves two skips, neither about polars: `test_golden_pipeline`
("regeneration only") and `test_semantics_all_models` ("holt has no
features").

The 5 deselected are the 3 `pins` and 2 `soak`. `pins` asserts properties
*of the pin itself*: `pl.__version__ == BUILT_AGAINST`, and the version
recorded in a generated document. So it cannot pass on an unpinned run, by
construction, and that is the canary's design rather than a gap.

The `online` CLI is unaffected either way: it links the Rust `polars`
0.55.2 and never touches py-polars, so a py-polars major cannot reach it.

#### What 2.0 adds that this library could use

`polars.io.plugins.register_io_source` gained two keywords:

```
1.44.1: (io_source, *, schema, validate_schema=False, is_pure=False)
2.0rc1: (io_source, *, schema, validate_schema=False, is_pure=False,
                       explain_name=None, explain_detail=None)
```

**`explain_name` / `explain_detail` put the bank's own name and its specs
into `lf.explain()`.** Without them, a plan with a bank in it shows a
generic Python source. On 2026-09-10 `_frame.py` passed neither. They are
additive and 2.0-only, so they needed a capability check rather than a
version bump. **Done the same day** in `e6da97c`: `_frame.py` passes both
when the installed polars takes them, and a polars without them gets the
plan it had before.

**`is_pure` is not new: it is in 1.44.1 too.** On 2026-09-10 this library
did not set it, although `_frame.py`'s module docstring documents exactly
that property. It was worth deciding on its own merits. Purity is polars'
licence to cache or skip a re-execution, and the rule R2 of
`docs/STATE-WORKFLOW.md` measured how often the source runs. **Decided the
same day in task 77** (`4c545af`): `_frame.py` declares the source pure
exactly when a run has nothing to write, with no `save_state` and no
`closed_groups` file.

#### What this does not settle

The run was one platform (macOS arm64), one Python (3.14), and a release
candidate.

**Superseded on 2026-09-10: `<2` became `<3` in 0.5.1.** What changed the
judgment was not new evidence about the candidate, but where the risk of
being wrong now lands. `release.yml`'s blocking leg runs the suite on the
newest version the declared range admits, in every dispatched release,
before any tag exists. So from 0.5.1 on, 2.0.0 final is tested before any
wheel is published under a range that includes it. The paragraph above was
written when nothing checked the top of the range at release time.
Widening then would have meant declaring support and finding out
afterwards. See
[Raising the ceiling to a new major](#raising-the-ceiling-to-a-new-major),
above.

#### Polars 2.0.0 final, and the pin moved (2026-10-08)

**The canary passed on 2.0.0 final, and the development environment moved
to it the same day (review round 4, D9).** The canary run of 2026-10-08
(run 37718689527, on `9b23ae6`) built the wheel and ran the suite on
py-polars 2.0.0: 4,920 passed, 1 failed. The failure was the floor-leg
workflow test, which reads the pin's form that the canary's unpin step
rewrites; it carries the `pins` marker since. The pin then moved:
`uv.lock` (polars and polars-runtime-32, 1.44.2 to 2.0.0, no other
package), `BUILT_AGAINST` in `tests/test_scaffold.py`, the README's matrix
row, and `docs/VALIDATION.md`, regenerated on 2.0.0 with Python 3.12.13.
That document changed in its header and one timing, and no number moved.
The goldens (`tests/test_golden_pipeline.py`,
`crates/online-core/tests/golden.rs`) passed unchanged, and
`docs/REGIMES.md`'s sections 1 and 5 to 9 held to their experiments.

**The Rust crate does not move, and the declared range does not either.**
crates.io's newest `polars` was still 0.55.2 that day (`cargo search`),
the one `Cargo.toml` pins, with `pyo3-polars` 0.28. py-polars 2.0 is a
Python-side major, and the wheel's statically linked copy never meets the
user's ([the pin, and the two copies of
Polars](#the-polars-pin-and-the-two-copies-of-polars-2026-08-31)). So the
Rust tests cannot see a py-polars release, and `polars>=1.34.0,<3` admits
2.0.0 as it stood; the floor leg still runs on 1.34.0.

### The expression plugin, as recorded

**The expression plugin, `pl.col(..).online.<model>(..)`, was removed in
task 85 (2026-09-17), because Polars hands a stateful user expression its
whole column.** It was the supported mechanism, with a MAJOR/MINOR
handshake the loader checks before its first call. So a break showed as a
refusal to load, rather than misbehaviour.

**The one path that carried a guarantee was also the one that could not
stream.** The measurements above stand as the record of what it did, and
nothing in the current guidance depends on it. The **Arrow PyCapsule
interface** now takes its place in the guarantee story. It covers the
output side only, so a call still crosses the `PyDataFrame` boundary on the
way in.

**2026-09-03: the expression plugin warned on every use** (`docs/PLAN.md`
§6, task 19). It shipped as before, and everything above still held for it.
What changed was that `pl.col(..).online.<model>(..)` issued
`InMemoryExpressionWarning` naming `lf.online.fit_predict`, because it was
the one O(data) form. Nothing here moved: the floor and ceiling were set by
`collect_batches`, not by the plugin, and the canary exercised all three
paths. For a few hours the same task had the plugin out of the wheel,
behind a cargo feature. That was reverted before release, so no wheel
published at the time lacked it.

## Keeping the API stable

**A change to the public API can break a user without raising anything.**
So a policy says what counts as API and what a change to it needs, and one
snapshot test turns every change into a diff someone has to accept. This
section gives the policy first, then what the API is and how each part is
caught. Its last two subsections are records of 2026-08-31 and of 2026-09.

### The policy

**What the table calls stable is API, and a breaking change to it needs a
new minor version before 1.0 and a new major after.** The README carries
the part a user needs, under *This package's own versioning*: the same
table, the release each kind of change needs, and the state-file promise.
The rest of the policy lives here, and CONTRIBUTING carries none of it.

| | what it covers |
|---|---|
| **stable** | everything in `__all__`, the `spec.*` constructors, their keyword names and their defaults (those that resolve in Rust included), the helper modules' functions but those of `po.sim` and `po.corr`, the `.online` namespace, output field names, the column names and dtypes of the frames the bank returns, the columns and dtypes of the closed-groups frame, the two columns `po.stream.embargo` adds, the words each string-valued parameter takes, the TOML config keys, the CLI's flags and its exit status, and the environment variables the shipped code reads |
| **unstable** | the file format of a window run's state; the written form of a formula target, in a TOML config and in a state file; the Arrow export, `fit_predict_arrow`, `predict_arrow` and `ArrowStruct`; and the modules `po.sim` and `po.corr`. Each carries a docstring label, and warns with `UnstableWarning` when used with `POLARS_ONLINE_WARN_UNSTABLE=1` set, as Polars warns under its own `POLARS_WARN_UNSTABLE` (task 198). An unstable part can change in any minor release, with a CHANGELOG entry |
| **stable by schema** | the state file format, versioned msgpack, which loads on every OS. From 1.0, a file a 1.x build wrote loads in every later 1.x. Before 1.0, a change of layout raises the minimum schema rather than adding a loader, so an older file is refused by its version (the user's waiver of hard rule 5, 2026-09-14 and 2026-09-28) |
| **not stable** | anything underscore-prefixed, apart from `po.stream.embargo`'s `_online_role` and `_online_role_weight`: a user passes the second to a spec's `weight` and filters on the first, so both are stable; the Rust crates, which the going-public item [R3](#r3--rust-crates-not-published) decided not to publish; and the exact numeric output, which depends on the polars version and the platform, within the 1e-12 tolerance `test_golden_pipeline.py` pins |

**Each part is unstable for a reason that a 1.0 promise would make
costly.** The window run's state reached version 7 in nine rounds of one
week. The formula tree is read from `expr.meta.serialize`, whose form
Polars calls unstable across its versions. The Arrow export's input side
still arrives as a Polars frame, and its import is unbuilt (task 86).
`po.sim` and `po.corr` stand beside the bank: neither reads its outputs,
as `po.eval` and `po.gram` do. `po.sim`'s frames also rest on NumPy's
`Generator`, which promises no stream across NumPy versions (review
2026-10-06, AP12, YB11 and YB17).

**Each stable part is pinned by a section of `tests/api_surface.txt`,
except the exit status, which a test holds.** A change to one is a diff in
that section ([the mechanism](#s--the-mechanism-one-api-snapshot-test--done)).
An unstable name that stands in a section -- `po.sim`, `po.corr`,
`ArrowStruct`, `fit_predict_arrow` and `predict_arrow` -- is pinned with the
label `# unstable` on its line, so the snapshot says which of its lines
carry no promise (review round 5, E7):

| stable part | pinned by |
|---|---|
| `__all__`, each function with its signature | `[package]` |
| the constructors' keyword names and the defaults they set in Python | `[common parameters]`, `[spec constructors]` |
| the defaults that resolve in Rust, `min_weight`'s per-kind rule included | `[resolved defaults]`, read off the bank's own build of each kind (`_polars_online.resolved_defaults`); `tests/test_spec_defaults.py` holds the README's warm-up tables to it |
| `ModelBank`'s methods, `__init__` included | `[ModelBank]` |
| the `.online` namespace | `[frame namespaces]` |
| the frames the bank returns: `groups()`, `summary()`, `describe()`, `coef()`, `last_row()`, `marginal()`, `gram()`'s keys, and `po.spec`'s `output_index()`, `coef_index()`, `coef_fields()` | `[frame columns]`, each column with its dtype |
| the closed-groups frame's columns | `[closed_groups columns]`, each column with its dtype, in order |
| the helper modules' functions, each with its signature | `[helper modules]` |
| `po.stream.embargo`'s two columns, `_online_role` and `_online_role_weight` | `[helper modules]`, as the default of `embargo`'s `role`, the first column's name, which the second's extends; `tests/test_label_delay.py` reads both |
| output field names | `[output field grammar]` |
| the words each string-valued parameter takes, and `withheld_reason`'s | `[enum values]`, read from each parameter's refusal of a word it does not take |
| the TOML config keys: the run config's, a spec's, and each model type's | `[toml keys]`, read from serde's refusal of a key a table has not got |
| the CLI's flags, and whether each takes a value | `[cli flags]`, from `online --help` |
| the CLI's exit status: 0, 1 for a refusal or a run error, 2 for a usage error | `docs/RUNNER.md`'s *Exit status* and a CLI test (tasks 191 and 187), not the snapshot |
| the environment variables, `POLARS_ONLINE_MAX_THREADS` and `ONLINE_TIMING` among them | `[env vars]`, from every read in `crates/*/src` and the package |

**From 1.0, a changed default or meaning needs a major release, and a fix
to numbers that were wrong a minor one** (decision D3 of `docs/PLAN.md`
§18, 2026-10-06). A number is wrong when it disagrees with the model's stated
definition, the update equations its docstring gives. Changing a default
is breaking, because it changes results silently:

| a change | before 1.0 | from 1.0 |
|---|---|---|
| a fix that moves no number | patch | patch |
| a fix to numbers that were wrong against the model's stated definition | minor | minor, declared in the CHANGELOG with the difference `scripts/compare_release.py` measures against the last release ([Compare with the last release](#compare-with-the-last-release-bit-for-bit)) |
| a new model, parameter, output or function | minor | minor |
| a stable name renamed | minor: the old name is refused, naming the new one (task 144's rule) | minor: the old name is forwarded to the new one with a `PolarsOnlineDeprecationWarning` (a spec parameter by the table, anything else by a forwarding of its own, below), and removed at the next major |
| a default or a meaning changed, or a stable name removed | minor, with a CHANGELOG entry saying what moved and by how much | major, with the same entry |
| a change to an unstable part | minor | minor, with a CHANGELOG entry |

**From 1.0, a rename is a deprecation, not a refusal** (decision D2).
`polars_online.PolarsOnlineDeprecationWarning`, a `DeprecationWarning`,
names the new spelling. A forwarding table carries each renamed spec
parameter to its new one, at any depth of a spec (task 198): in Python a
spec builder's keyword and a spec dict's key, `with_windows`' `like=` spec
included (`_DEPRECATED` in `python/polars_online/_warnings.py`), and in
Rust a TOML key (`DEPRECATED` in `crates/online-polars/src/spec.rs`). The
table is
empty at 1.0: every rename made before it stays refused, by
`_RENAMED` in `python/polars_online/_spec.py` and `RENAMED` in
`crates/online-polars/src/spec.rs`. A name enters the forwarding table in
the minor release that renames it, and leaves it at the next major.

**The table forwards spec parameters, and nothing else** (review round 5,
D4). A keyword of any other function, a function, a word a parameter
takes, a command-line flag, an environment variable, an output field or a
frame column renamed after 1.0 does not go through it. Each ships its own
forwarding in the release that renames it: `forward_deprecated` at the
top of the function whose keyword moved, a stub that warns and calls the
new function, an alias for the word, flag or variable, or a second column
beside the new one. It warns with the same `PolarsOnlineDeprecationWarning`
and is kept until the next major. Nothing of this kind is built at 1.0,
since nothing has been renamed after it; `renamed_function` in
`python/polars_online/_renamed.py` is the refusing stub a pre-1.0 rename
keeps, not a forwarding one.

**From 1.0, a state file a 1.x build wrote loads in every later 1.x**
(decision D1). The schema 1.0.0 ships is the floor: `MIN_SCHEMA_VERSION` and
`MIN_BANK_SCHEMA_VERSION` stay at it for the life of 1.x. Every later
change of layout ships a loader for the layout before it, and keeps the
frozen fixtures of that layout. Each fixture is checked three ways (task
198; review round 5, D1). The current schema's fixtures load, continue to
the bit, and re-save byte for byte. A previous schema's load through the
loader, continue to the bit, and re-save as the current schema's fixture
bytes of the same case: the loader turns a state into exactly what this
build writes from the same rows. The two unstable formats, a window run's
state and a formula target inside a bank's, are outside the promise.
Before 1.0, refit a saved state from its input after upgrading across a
release.

**From 1.0, raising the Polars floor is a minor release, capping the range
below a broken Polars a patch, and dropping a Polars major a major**
(decision D4). A resolver that installs the new release upgrades Polars with it,
inside the same Polars major, so a raised floor asks no code to change. A
user who must stay on an older Polars stays on the older minor. The
newest-Polars runs prove the ceiling, and only the floor runs prove the
floor ([The floor and the ceiling](#the-floor-and-the-ceiling)):

| a change to the Polars range | before 1.0 | from 1.0 |
|---|---|---|
| widening it to admit a newer Polars | minor | minor |
| raising the floor | minor | minor |
| capping it below a Polars that broke this library | patch | patch |
| dropping a Polars major version | minor | major |

**A user who wants stability pins the current minor, before 1.0 and
after.** Before 1.0 the minor carries breaking changes. From 1.0 it carries
none, but it can carry a declared fix to numbers that were wrong, so
`~=1.0` keeps code working and `~=1.4.0` holds the numbers as well. The
README's example pin names the current minor, as `~=0.13.0` names the
0.13 series, and a minor release updates that example
([step 6](#the-steps-of-a-release)). The current release is the version
`pyproject.toml` names.

### What not to do

| do not | why |
|---|---|
| `__getattr__`-guard the submodules | `polars_online._spec` is importable, and someone will reach into it; that is what the underscore is for. Enforcing it would take more work than the honesty is worth |
| freeze the API before 0.1.0 ships | the snapshot test is for detecting change, not preventing it, and before there are any users is exactly when changing names is cheap. 0.1.0 shipped on 2026-09-03 |
| add `cargo semver-checks` | it is the right tool for a published Rust API, and pure overhead otherwise. Add it only if the decision of the going-public item [R3](#r3--rust-crates-not-published) is reversed and the crates are published |

### What the API actually is

The API is bigger than `__all__` suggests. The proposal of 2026-08-31 found
four surfaces, in descending order of how easily a change could break
someone. Each is described as it stands, with its dated figures.

**1. Output field names, the largest surface.**

```
pred_y__r0.000001@h100    resid_y__r0.5@h100    abs_resid_q0.95_y0    weight_sum@h500
```

Users write `out["m"].struct.field("pred_y")`. These strings are generated
by `format!` over ridge values, half-life suffixes, feature-set labels,
quantile levels and target names, and **every one of them is API**. Adding
an output, renaming a suffix, or changing how a float renders silently
breaks downstream code, and the failure surfaces as a `StructFieldNotFound`
in someone else's pipeline. On 2026-08-31 exactly *one* spec shape was
pinned (`test_exact_field_names_for_a_grid_spec`, then 52 fields, now 58,
still in `tests/test_portability.py`). Today the snapshot test below pins
28 spec shapes. `test_names_match_the_realized_struct_for_every_model`
checks, in 40 cases, that the fields a spec declares are the fields its
output has.

**2. Twenty-one spec constructors and ~200 named keyword parameters, plus
the helper modules**, as counted on 2026-09-06. All are keyword-only, so
positional order is not API, but every *name* is. On that date 29 of the
parameters were the `CommonKwargs` every spec shares. Today
`python/polars_online/_kwargs.py` declares them, and the snapshot's
`[common parameters]` lists them once. The constructors are still
twenty-one. The four helper modules named then were `po.corr`, `po.sim`,
`po.gram` and `po.prep` (`po.stream` since task 105), whose functions are
read the same way. `tests/api_surface.txt` pins the constructors and their
keyword names. It pins every helper module the package has, read from the
package's directory: today `corr`, `eval`, `gram`, `ops`, `sim` and
`stream` (`tests/test_api_surface.py`, since task 109). Since task 190 it
pins each of their functions with its signature, keyword names and defaults
included, as it does every function in `__all__` and `ModelBank.__init__`;
before, it pinned their names alone.

**3. Defaults, which are API in the worst way.** Changing one does not
raise: it silently changes users' numbers, which is worse than breaking
them. Every default a builder sets in Python is in the snapshot, which
renders keyword names with their defaults. Many resolve in Rust from
`None`, so a constructor's signature shows `None` for them. Until task 190
nothing pinned most of them. Now the snapshot's `[resolved defaults]` reads
each off the bank's own build of every kind's least spec: `Spec::check`,
then the constructor of every group's stream, through
`_polars_online.resolved_defaults`. It renders one line per kind and field:
the model's configuration, what the model derives from a field left unset
(`bocpd`'s priors, `rcov`'s ring and window, `marginal`'s bin budget), the
stream's gates and diagnostics' settings, and the clock policy. Variants
add the defaults that only an option reads, such as `sgd`'s `huber_delta`
under `loss = "huber"`. Pinned by:

| default | set in | pinned by |
|---|---|---|
| every default a builder leaves to Rust, `min_weight`'s per-kind rule included | Rust | the snapshot's `[resolved defaults]`; `tests/test_spec_defaults.py` holds the README's warm-up tables to it |
| the solve cadence of `ewridge`, `lasso`, `huber` and `quantile`: by weight since 0.13.0, `half_life/50` of clock in steady state | Rust | `[resolved defaults]` (`solve_every`, `solve_share`); `tests/test_solve_cadence.py` measures it |
| the clock settings, the readiness gates, and `min_weight`'s count floor | Rust | `[resolved defaults]`; `crates/online-polars/tests/spec_defaults.rs` |
| `standardize`: true for `kalman`, false for `ewridge`, `huber`, `quantile` and `sgd`; `lasso` has no such keyword | Python | the snapshot |
| `average_eta = 1.0` | Rust | `[resolved defaults]`; `spec_defaults.rs` |
| `clip_gradient = 1e3`, for `sgd` | Rust | `[resolved defaults]`; `spec_defaults.rs`; `tests/test_sgd.py` checks that the default clip keeps a Poisson fit stable |

`docs/VALIDATION.md` justifies the defaults it measures, and
`test_validation_doc.py` pins that they are still the measured optimum.

**4. State files and the TOML config.** A state file is versioned msgpack,
with `SCHEMA_VERSION` plus a bank `format_version`. `SCHEMA_VERSION`
(`crates/online-core/src/lib.rs`, and `po.schema_version()` from Python)
moves with every layout change. The bank's `format_version` is 2, or 3 when
a spec carries a clock parameter as a duration. A bank file older than
`MIN_BANK_SCHEMA_VERSION` (`crates/online-polars/src/bank.rs`) is refused
by its version, and that is every file a release has written so far. The
models' own states name a floor of their own, `MIN_SCHEMA_VERSION`
(`crates/online-core/src/lib.rs`). A window run's state, from
`with_windows`, has its own version, `WINDOWS_VERSION`
(`crates/online-polars/src/windows_frame.rs`). A change to it moves the
bank's schema too, and a test pairs the two numbers (`windows_frame.rs`).
The state of `po.stream.refresh_time` has its own version,
`REFRESH_VERSION` (`crates/online-polars/src/refresh.rs`). The exception to
hard rule 5, and the reason for each raise of a minimum, are recorded
beside `MIN_SCHEMA_VERSION` and `MIN_BANK_SCHEMA_VERSION`.

**The proposal held this surface up as the model the rest should follow.**
It had a frozen fixture per schema (`state_v1.rs`, `state_schema2.rs`,
`state_schema3.rs`, `state_schema4.rs`, `state_schema5.rs`), and a
documented rule (hard rule 5) to keep a previous-version loader. Each
fixture asked only what a converter must preserve: it loads, it continues
the stream to the bit, and it re-saves byte-identically while the writer is
unchanged. So the last of those became the upgrade test the moment the
writer moved, which is exactly what happened to schema 3 at task 40.

**Pre-1.0, the loader rule is suspended, and the fixtures are gone.** They
went in the naming pass of 2026-09-07 (`6124d0b`), after which no older
state could be read. On 2026-09-25 `SCHEMA_VERSION` was 16 and
`MIN_SCHEMA_VERSION` 14, so 14 and 15 loaded through the named encoding's
defaults. The constants named above hold today's numbers.

### S — The mechanism: one API snapshot test — **done**

**Every API change becomes a reviewable diff.** `tests/test_api_surface.py`
renders the entire public surface to text, and compares it with a
checked-in file:

```
tests/test_api_surface.py   ->  compares against  tests/api_surface.txt
```

A rename shows up as `- pred_y__r1e-06 / + pred_y__r0.000001` in the PR,
where a human decides whether it is intended, and whether it needs a major
version. Accidental changes fail, and deliberate ones are visible and
deliberate. It is the same trick `test_golden_pipeline.py` plays on the
numbers, applied to the names. That is why the proposal chose this shape
rather than more unit tests.

Regenerate with `UPDATE_API_SURFACE=1` set:
`UPDATE_API_SURFACE=1 pytest tests/test_api_surface.py`. It was verified to
fail loudly on a simulated rename, with a unified diff and "needs a version
bump" in the message.

What the snapshot renders, as the proposal asked for it, as it was built on
2026-08-31, and today:

| what it renders | the proposal | built, 2026-08-31 | today |
|---|---|---|---|
| public symbols | every public symbol in `polars_online` and `polars_online.spec` | every public symbol | each function with its signature (task 190) |
| constructors | every spec constructor's **full signature including default values** | every constructor signature *including defaults*, the shared `**common` parameters listed once, explicitly | |
| the namespace | the expression-namespace method list | the expression namespace | the `lf.online` and `df.online` namespaces; the expression namespace went with task 85 |
| `ModelBank` | | | its name and signature, `__init__`'s since task 190 |
| helper modules | | | every one: `corr`, `eval`, `gram`, `ops`, `sim` and `stream` (`prep` until task 105), each function with its signature since task 190 |
| output fields | `output_fields()` for a canonical matrix of ~12 spec shapes: each model, plus grids, feature sets, multi-target, and every `emit_*` combination | `output_fields()` across a 14-case matrix covering every model, the full grid/emit combinations, and the float-rendering extremes | 28 cases |
| the closed-groups columns | | | each kind's columns, in order (since 2026-09-24) |
| the defaults that resolve in Rust | | | `[resolved defaults]`: every kind's configuration as the bank builds it from its least spec, and the defaults only an option reads (task 190) |
| the frames' columns | | | `[frame columns]`: each frame's columns and dtypes, in order (task 190) |
| the words a string parameter takes | | | `[enum values]`: every string-valued parameter, and `withheld_reason` (task 190) |
| the TOML config keys | | | `[toml keys]`: the run config's, a spec's, and each model type's (task 190) |
| the CLI's flags | | | `[cli flags]`: the usage line and every flag, with the value it takes (task 190) |
| the environment variables | | | `[env vars]`: every read in `crates/*/src` and the package (task 190) |
| versions | `SCHEMA_VERSION` and the bank `format_version` | `schema_version` | |
| length of `tests/api_surface.txt` | | 416 lines | as `wc -l` counts it; 904 lines on 2026-09-24, 1,156 on 2026-10-06 |

**Building it found two things, and fixed both:**

| found | what was done |
|---|---|
| a legal `ridge = 1e-300` produced a **311-character field name**, because Rust's float `Display` never uses scientific notation | numbers outside `[1e-6, 1e7)` now render compactly (`1e-300`), chosen so no existing name changed. The rendering is centralized in one `num_label()` function, instead of seven scattered `format!` sites |
| a grammar ambiguity: a target literally named `y__r0.5` renders identically to a grid field | documented in the README, with a duplicate-name tripwire in `Bank::new`, rather than banned, since it cannot corrupt a struct |

The snapshot subsumes the single-shape field-name test. It would have
caught the E23 declared-vs-realized divergence (`docs/ENHANCEMENTS.md`)
from the other direction.

### API usability round (pre-users window)

The API was reviewed by using it as a naive user. The findings, and what
was done:

| finding | what was done |
|---|---|
| **String construction was the API's worst ergonomic.** Reaching a grid slot meant hand-building `pred_y__r0.5@h500`, that is, mentally reimplementing the float rendering | **Fixed**: `spec.output_index()` returns every field with the machine values its name encodes (kind, target, half-life/lam, ridge, feature_set, lasso λ, quantile level, ew_cov columns). It is produced by the same Rust code that renders the names, and selection becomes a Polars filter |
| **`coef` was an unmapped flat list** | **Fixed**: `spec.coef_index()` maps each position to (target, combo, term). It is derived from `output_index`, so it cannot drift, and was verified by recovering known coefficients by position |
| **Our own `eval.unpack` parsed names heuristically** ("longest match wins") | **Fixed**: an optional `spec=` argument resolves slot→target exactly, through the index. The heuristic remains only for callers with a frame but no spec |
| **`min_weight` defaults sensibly**: the first prediction comes at ~k+2 | verified; nothing to do |

**Reviewed and deliberately kept:**

| kept | rather than | why |
|---|---|---|
| `fit_predict`'s name | | sklearn readers may expect in-sample fit-then-predict. Ours is predict-then-update, which is strictly better for them. A docstring note suffices, and the fused call *is* the out-of-sample guarantee |
| the `emit_*` booleans, seven then and nine today (`tests/api_surface.txt`) | a stringly `outputs=[...]` list | they are discoverable in signatures |
| spec dicts | spec objects | they are JSON-ready and printable, and validation already happens at construction |
| wide struct output | a native long format | `eval.unpack` already provides long form on demand |

**A new optional output is switched on in one of the three shapes already
in use, never a fourth** (review 2026-10-06, AP14). Each shape says what
the setting is:

| the switch | when | as in |
|---|---|---|
| a boolean `emit_*` keyword | the output is on or off, and nothing else | `emit_sigma=True` |
| a list | the list is also the setting: which of several outputs, or at which values | `ew_cov(stats=["mean", "corr"])`, `resid_quantiles=[0.5, 0.9]` |
| a level | the number is also the setting | `conformal=0.9`, the coverage asked for |

### The 2026-09 batch: what it added to the surface (tasks 45–56)

The batch added five models (`deco`, `rcov`, `hmm`, `corrchange`, `bocpd`),
and four helper modules' worth of new functions: `po.corr` with twenty,
`po.sim.regimes`, `po.prep.refresh_time` and `po.gram.from_row`. It also
added `group_close` and the closed-group queue on every spec, and lagged
co-moments on `ew_cov`. In API terms:

| what the batch added | how the API held it |
|---|---|
| **`SCHEMA_VERSION` 5** | `state_schema5.rs` froze it: a bank with an undrained closed row, an `ew_cov` with a partly filled lag ring, and one of each new model mid-stream. `state_schema4.rs` still loaded, continued to the bit and re-saved as 5, which was hard rule 5 discharged. Both fixtures have since gone, and the schema has moved on many times since: `po.schema_version()` gives today's ([What the API actually is](#what-the-api-actually-is)) |
| **every new output field name** | pinned by `tests/api_surface.txt`, which then gained a `[helper modules]` section, and by `tests/test_golden_pipeline.py`, which fixes the numbers of a twenty-five-spec bank at three rows |
| **three new defaults, measured rather than chosen** | `bocpd`'s `robust_beta = 0.1` (above ~0.2 nothing is ever detected), `bocpd`'s `emission = "diag"`, and `rcov`'s automatic bandwidth. The measurements are in `docs/PLAN.md` task 55 and `docs/ENHANCEMENTS.md` E61 for `bocpd`, and in E57 for `rcov`. Changing one of them is a breaking change, by [the policy](#the-policy) |
| **no new Polars interface** | two of the three streaming paths of the time still carried no guarantee (CLAUDE.md rule 13): `ModelBank` and the IO plugin. The batch added no polars API dependency beyond `LazyFrame.collect_batches`, which is already the floor |
| **the closed-group frame's *column* names** | pinned since 2026-09-24 (`spec`, `group`, `session`, `rcov`, `rcov_bandwidth_used`, …). They are as much API as an output field is, and until then only `test_closed_groups.py` read them by name. `tests/api_surface.txt` now records them in order, as `[closed_groups columns]`: the shared columns, and each kind's full list, with `ew_cov`'s PCA block and `marginal`'s lag and bin blocks switched on |

## CI cost while the repo was private

**The CI cost policy still holds, and `tests/test_ci_cost_policy.py`
enforces it by reading the workflows.** While the repository was private,
Actions minutes were metered, and not evenly: macOS billed at 10x and
Windows at 2x. Going public removed the constraint, since Actions is
unmetered on public repositories. The matrix still widens by itself on a
public repository, and would narrow again on a private one. The records
below are how the month's minutes ran out on day one, where they went, and
what the fixes changed.

### The policy, as the test holds it

`tests/test_ci_cost_policy.py` parses every workflow, in 65 cases over
twelve classes, counted on 2026-10-08:

| class | what it asserts |
|---|---|
| `TestEveryJobIsBounded` | every job in every workflow has a timeout of at most two hours, and every workflow a concurrency group; superseded runs are cancelled, and releases queue instead. The one longer job is the weekly mutation shard, at four hours, on Linux and only while public or by hand |
| `TestTheMatrixDefaultsToCheap` | lint stays on Linux and never builds the extension; the expensive runners are opt-in; the visibility test fails safe on a missing field; macOS is reachable only by a manual dispatch or the repository being public |
| `TestStepOrderingThatHasAlreadyBrokenCI` | disk is freed before the cache is restored, rustflags are set before anything compiles, and the cache survives a failing job |
| `TestDocOnlyPushesAreFree` | CI has no paths filter on a push or a pull request; the benchmark, which reports and never gates, skips doc-only pushes |
| `TestPythonVersions` | every Python the package declares runs on Linux, and the floor and the newest on every OS; the release comparison reports once and never gates; the API reference is built and published once |
| `TestMutationTesting` | the mutation run over changed lines gates every push and pull request, with one report, in as many shards as its mutants need: a first job lists them, with the scope, the diff and the pinned cargo-mutants the shards use, and takes a shard for every forty, at most four-fifths of the slowest shard measured, and the report expects that many; each push's pass has a concurrency group of its own, so no later push cancels or replaces it, while a pull request's newer commit cancels its older; the weekly pass runs only while public or by hand, and reports its survivors; every run skips the doctests, stops a mutant at ten times the baseline, stops itself inside its job, and lists the mutants it was given, and its report fails on a run that tested fewer or a shard that sent nothing |
| `TestTheRustTestsLinkNoPython` | every workflow and the local gate leave `online-py` out of `cargo test`, so no test binary links libpython |
| `TestTheLinuxPrepIsOneAction` | the step that frees the disk and swaps in `lld` is one composite action, called after the checkout and on Linux alone |
| `TestActionsArePinnedToCommits` | every external action is pinned to a commit, with the version it names in a comment |
| `TestTheDeclaredRustVersionBuildsTheLock` | the workspace's `rust-version` covers every locked dependency's own |
| `TestTheDeclaredRustVersionCompiles` | a weekly Linux job, `msrv.yml`, checks the whole workspace on exactly the `rust-version` Cargo.toml declares, from the lock as committed |
| `TestEveryDeselectedMarkerRunsSomewhere` | each pytest marker the default run leaves out is asked for by some workflow |

A comment could not stop the 830-minute mistake recorded below; that test
would have.

### Actions quota: exhausted on day one (2026-08-31)

**A push on 2026-08-31 did not run, and the cause was the Actions
allowance, not the workflow.** The commit it pushed was `1a52267`, which
is not a commit in this repository's history. All four jobs failed in ~2s
with *"The job was not started because recent account payments have failed
or your spending limit needs to be increased."*

The billed minutes, summed from job durations across every run this repo
had ever had, all of them that day, with GitHub's per-job round-up and OS
multipliers:

| runner | jobs | raw min | x | billed |
|---|---:|---:|---:|---:|
| windows-latest | 12 | 565 | 2 | **1130** |
| macos-latest | 4 | 83 | 10 | **830** |
| ubuntu-latest | 14 | 310 | 1 | 310 |
| | | | | **2270** vs a 2,000/mo allowance |

Two things went wrong.

**macOS leaked onto pull requests.** The matrix read
`event_name == 'push' && [ubuntu,windows] || [ubuntu,macos,windows]`, so
anything that was not a push, every PR, got the full matrix. One dependabot
PR spent 830 minutes, 37% of the month, on four macOS jobs. Meanwhile the
comment above the matrix claimed macOS ran "weekly, on demand, and on
release tags". It was rewritten as opt-in on `schedule`/`workflow_dispatch`.
Today a schedule adds Windows alone, and macOS runs on a manual dispatch or
a public repository ([the policy](#the-policy-as-the-test-holds-it)).

**Windows ran cold every time**: 12 jobs, averaging 47 raw minutes each.
[CI cost: where the 80 minutes went](#ci-cost-where-the-80-minutes-went)
says why. `cache-on-failure` should cut this sharply, but it had not been
observed yet: no run since had been allowed to start.

The allowance resets at the start of the billing month. The structural fix
was going public, because Actions is unmetered on public repositories, which
removes this constraint entirely. It was already the plan for v0.1.0, and
the repository went public on 2026-09-02.

### CI cost: where the 80 minutes went

Measured from step timings on runs 33398135931 and 33406764631, not guessed.
A cold `test` job, step by step:

| step | Windows | Linux |
|---|---|---|
| `uv sync` (compiles the extension, release) | 39m | 18m |
| `cargo test` (compiles the workspace, debug) | 29m | 10m |
| `maturin develop --release` | **31s** | **18m** |
| `pytest` | 11m | 5m |

Three findings:

1. **The cache never existed.** Every Windows run logged `No cache found.`
   rust-cache skips its save step when the job fails, and the Windows job
   had never once passed. So each red run discarded ~70 minutes of
   compilation, and the next started cold. `cache-on-failure: true` breaks
   the cycle.
2. **Linux compiled release twice.** The `lld` step wrote rustflags into
   `.cargo/config.toml` *after* `uv sync` had already built the extension.
   Rustflags are part of cargo's fingerprint, so `maturin develop --release`
   rebuilt everything: 18 minutes, against Windows' 31 seconds for the same
   step with a valid fingerprint. The step moved before `uv sync`.
3. **Two full compiles are inherent, not waste.** `uv sync` produces the
   release extension that pytest imports, and `cargo test` the debug test
   binaries. Only caching helps here.

Defender exclusions were added for the Windows build. They are best-effort
and cannot fail the job, and their effect is confounded with the cache
landing in the same run.

### What was unexpectedly expensive (2026-08-31)

A full review of every cost driver after the quota ran out, ordered by what
each actually cost. The numbers are observed job durations, with GitHub's
per-job round-up and the OS multipliers (macOS 10x, Windows 2x).

| # | Cause | Cost | Status |
|---|---|---|---|
| 1 | macOS on every pull request — the matrix's *fallback* branch was the expensive one | 830 min (37%) | fixed, and pinned by a test |
| 2 | rust-cache never saved: it skips its post step on failure, and the Windows job never passed | ~70 min/run recompiled | `cache-on-failure: true` |
| 3 | `lint` ran on three OSes to check formatting | 220 min | Linux only, always |
| 4 | `lint`'s `uv sync` built the extension it never imports | 18 min Linux, 39 Windows, per run | `--no-install-project` |
| 5 | Linux compiled release twice: `lld` rustflags written *after* `uv sync` invalidated its build | 18 min/run | linker step moved before the sync |
| 6 | No `timeout-minutes` on any job in any workflow | up to 720 billed min for one hung Windows job | every job bounded |
| 7 | No concurrency group on `polars-canary` or `release` | pays for superseded runs | added (release *queues*, see below) |
| 8 | A cancelled run still bills for what it used | 55 min run cancelled at 12:44 was billed | inherent; `cancel-in-progress` limits how often |

Two things were examined and deliberately left alone:

| left alone | why |
|---|---|
| two full compiles per `test` job, finding 3 above | different profiles, and genuinely different artifacts, so only caching helps |
| `release.yml` queues superseded runs rather than cancelling them | it is the one workflow where cancelling costs more than it saves: the run being cancelled may be midway through uploading wheels or publishing to PyPI |

**Watch out for `release.yml` (2026-08-31, while the repo was private).** It
was tag-triggered then, and untouched by the policy above: three macOS jobs
plus Windows, across a six-target wheel matrix. At 10x, one release tag
could plausibly cost most of a month's allowance while the repo was
private. Hence the rule of the day: do not cut a release tag before going
public. The repository went public on 2026-09-02, and 0.1.0 was released
on 2026-09-03.

#### The ubuntu `test` failure was not infrastructure

**Twice the Ubuntu test job died ~90 seconds in, *inside*
`Swatinem/rust-cache`, and it was not a flaky runner.** Each time there was
no step conclusion, and a log blob that 404s. It looked like a flaky runner,
but the check-runs annotations API had the real error:

```
System.IO.IOException: No space left on device
```

The runner could not write its own diagnostic log, which is why no log was
ever uploaded. The Ubuntu image starts with 9.3 GB free, and the saved
`v0-rust-test-Linux-x64` cache is 1 GB compressed and expands well past
that. The step that recovers 21 GB ran *after* the cache restore.

It only started failing once a cache existed to restore: every earlier run
logged `No cache found.` and sailed past. So the symptom appeared at the
exact moment the cache fix started working, which is what made it look
unrelated.

**The fix moved the disk-freeing step ahead of the cache restore.** That
step now has two independent reasons to be where it is, and
`tests/test_ci_cost_policy.py` pins both: disk before the restore, and
rustflags before any compile.

#### What a push cost after the fixes

Extrapolated from the observed timings, for a private-repo push:

| | before | after |
|---|---:|---:|
| lint | 3 OSes, full build — ~235 billed | Linux, no build — **2** |
| test | 3 OSes — ~706 billed | Linux — **26** |
| **per push** | **~940** | **28** |

All measured, on the first fully green run (33419911742). **A 34x
reduction**: roughly 107 pushes a month against Pro's 3,000, where the
config this replaced allowed three.

Where the two fixes landed, step by step:

| step | before | after |
|---|---:|---:|
| `free disk` | ran after the cache | 14 GB → 34 GB free, *then* `Cache restored successfully` |
| `uv sync` (lint) | 18 min | **3 s** |
| `cargo test` | 9m46s cold | **4m56s** warm |
| `maturin develop --release` | 18 min | **3 s** |

**`maturin develop --release` going from 18 minutes to 3 seconds is the
rustflags fingerprint fix, exactly as predicted.** It now reuses the release
build `uv sync` already did, instead of invalidating and repeating it.

**The remaining cost is `uv sync` in the `test` job, at 13m29s, which builds
that release extension.** rust-cache deliberately discards workspace crates'
own artifacts to keep the cache small, so our four crates recompile each
run. `cache-workspace-crates` would address it, but was not pursued,
because 28 billed minutes a push is no longer the constraint.

## Going public, as recorded

This section is the record of going public, from the proposal of 2026-08-31
to the settings of 2026-09-06. On 2026-09-06 the document's status read
**done, and kept as the record**: 0.1.0 had been released on 2026-09-03 and
0.1.1 on 2026-09-04, and 0.2.0 was being released. The decision to go public
was taken on 2026-08-31, and the repository went public on 2026-09-02, when
it was deleted and recreated public.

Its R1–R6 are this document's own items. `docs/STATE-WORKFLOW.md` numbers
its rules R1–R7, and `docs/PLAN.md` §14 its review rounds R1 to R9; each
is named with its document where this one cites it.

### Suggested order

The proposal's order, as planned on 2026-08-31:

1. **R1, R2, R3**: small, mechanical, and all three are things a reviewer
   of a public repo checks in the first minute.
2. **The API snapshot test**: before 0.1.0, while renames are still free.
3. **The policy paragraphs** in README and CONTRIBUTING.
4. **R6** scan, **R5** green CI, then **R4** settings, then flip visibility.

**Steps 1 and 2 and the R6 scan happened in this order on 2026-08-31, and
R5's green CI came before the repository went public.** Three things went
otherwise. The policy was written in this document, under
[The policy](#the-policy), and not in the README or CONTRIBUTING. The R4
settings needed the web UI, and on 2026-09-03 the REST API set some of them
instead. And instead of a flip of visibility, the repository was deleted
and recreated public on 2026-09-02
([Going public (2026-08-31)](#going-public-2026-08-31)).

### Before going public: R1–R6

Ordered by "would embarrass us if a stranger found it first".

**Already done when the proposal was written:** Apache-2.0 `LICENSE`,
`CONTRIBUTING.md`, `SECURITY.md`, `CHANGELOG.md`, issue and PR templates,
and an outward-facing README. So were `release.yml`, with wheels for six
platforms plus an sdist and PyPI trusted publishing, and the name
`polars-online`, verified free. The test suite is the repo's strongest
argument, with golden and hardening layers: ~640 Rust + ~2,200 pytest,
counted on 2026-09-06; 976 Rust tests and 3,265 pytest cases on 2026-09-25
(docs/TESTING.md).

| ID | item | status |
|---|---|---|
| R1 | least-privilege workflow permissions | done |
| R2 | pin actions to commit SHAs | done, with one deliberate exception |
| R3 | Rust crates: not published | done |
| R4 | repository settings | needed the web UI on 2026-08-31; partly done through the REST API on 2026-09-03 ([Repository settings, as recorded](#repository-settings-as-recorded)) |
| R5 | green CI on all three platforms first | done |
| R6 | history scan | done, and it found something |

#### R1 — Least-privilege workflow permissions

None of the four workflows set a top-level `permissions:` block, so every
job inherited the repository default. `release.yml` correctly narrowed two
jobs (`contents: write`, `id-token: write`), but the rest ran wider than
they needed.

**On a public repo this matters more than on a private one.** Any fork's PR
can trigger workflows, and a token with more scope than the job needs is the
standard supply-chain foothold.

The fix: `permissions: {}` at the top of every workflow, granting per job
only what that job uses. That is `contents: read` for checkout, and the two
`release.yml` grants that already existed.

#### R2 — Pin actions to commit SHAs

All nine third-party actions were pinned to mutable tags (`@v5`, `@v2`,
`@release/v1`), and a tag can be repointed by whoever owns the action. For a
private repo this is a low risk, but for a public one publishing signed
artifacts to PyPI, it is the thing supply-chain audits look for first.

The fix: pin to full SHAs with the version in a trailing comment, and add
Dependabot (`.github/dependabot.yml`, `package-ecosystem: github-actions`)
so the pins still get updated. Pinning without automation just means stale
actions.

**One action stays on a tag, on purpose: `dtolnay/rust-toolchain@stable`.**
Pinning it would freeze the compiler version, which is the opposite of what
that action is for. The commit that did R2 says so (`307543f`).

#### R3 — Rust crates: not published

No crate set `publish`, so `cargo publish` would happily have pushed
`online-core`, `online-polars`, `online-cli` and `online-py` to crates.io.
That is four more public APIs to maintain, and `online-py` in particular is
meaningless outside the wheel.

**Decided: Python only.** All four carry `publish = false`, and that was
verified rather than assumed. The wheel builds from path dependencies, and a
clean-venv install of the sdist compiles the Rust from source. It ran both
the `ModelBank` and the expression plugin, so nothing needs to reach
crates.io. Revisit for `online-core` alone if it is ever wanted standalone.

**That verification found a real packaging defect: the sdist shipped with
no `LICENSE` at all.** Apache-2.0 requires the licence to accompany a
distribution, and packagers check for it. The fix is PEP 639 `license-files`
under `[project]`; note that `[tool.maturin] license-files` is silently
ignored. The licence now lands in the sdist, and in the wheel's
`.dist-info/licenses/LICENSE`.

#### R4 — Repository settings

**R4 asked for three settings: branch protection on `main`, the
description and topics, and private vulnerability reporting.** Branch
protection was to require CI to pass and forbid a force-push, since without
it nothing catches a bad push. The topics, `polars`, `online-learning`,
`streaming`, `regression` and `rust`, are how anyone finds the repository.
`SECURITY.md` already told people to use private reporting. The working
rule that followed was to fast-forward `main` only from a branch whose CI
run was green; a release, by [its steps](#the-steps-of-a-release), is
committed on `main` itself.

On 2026-08-31 these needed the web UI and were not scriptable from here.
On 2026-09-03 the REST API set the description and topics, and a ruleset
that stops `main` being force-pushed or deleted. Private vulnerability
reporting was left to the owner.
[Repository settings, as recorded](#repository-settings-as-recorded) has
each one's full record.

#### R5 — Green CI on all three platforms first

Do not make it public with a red badge. When this was written, Windows and
Linux fixes were in flight, and macOS was green on the first run. All three
were green before the repository went public, and every release since has
waited for them.

#### R6 — History scan

**No key, token or credential pattern anywhere in the history**, and no
occurrence of the maintainer's personal email in any file's content or any
commit's diff.

**It did find one thing worth acting on: two *local* refs still held the 100
pre-rewrite commits authored under the maintainer's personal address.** They
were `backup-before-email-rewrite` and filter-branch's
`refs/original/refs/heads/main`. The address is not reproduced here: this
file was about to be public, and writing the address down would undo the
removal it describes. The refs were not pushed, and their content was
byte-identical to `main`. So their only remaining property was the email
that had been deliberately removed, and `git push --all` or `--mirror`
would have published exactly that. Both were deleted and the reflog
expired, so no ref now carries the address.

### Open-source preparation: the full sweep (2026-08-31)

Everything checked or done beyond the two standing items of the day: delete
and recreate the repository as public, and cut no release tag while it was
private.

**The license question, answered: Apache-2.0 is among the most
corporate-friendly licenses there is.** Closed-source use, modification, and
internal or commercial redistribution are all permitted, with no obligation
to share source. A company using it must keep the license text and copyright
notices with any *redistribution*, and state significant changes if it ships
modified copies. Purely internal use triggers nothing in practice. Its §3
patent grant is the reason many corporate counsel *prefer* it to MIT. Every
contributor grants an express patent license, with termination only for a
party that sues over the covered code. It appears on effectively every
corporate allowlist, unlike GPL/AGPL (commonly banned) or MPL (often
case-by-case review).

**The license of the *artifact* matters as much as the repo's, because the
wheel statically links its whole Rust dependency tree.** An audit via
`cargo metadata` found **453 external crates, and zero copyleft
obligations**. Everything is MIT/Apache-2.0/BSD/ISC/Zlib/Unicode/BSL-1.0,
where BSL is Boost, not Business Source. The only crate mentioning LGPL,
`r-efi`, is `MIT OR Apache-2.0 OR LGPL`, and OR means MIT applies. The
single Python runtime dependency, `polars`, is MIT.

**Deliberately, there is no NOTICE file.** Apache-2.0 §4(d) makes a NOTICE
file propagate to every downstream redistribution. Not shipping one is a
kindness to corporate users, and the LICENSE carries the attribution.

**Done in this sweep:**

| item | what was done |
|---|---|
| PyPI metadata | classifiers, keywords, and Changelog and Issues URLs filled in. There is no `License ::` classifier, because PEP 639 deprecates mixing it with `license`. `twine check` passes on both sdist and wheel, and the wheel carries `dist-info/licenses/LICENSE` |
| the name | **free on PyPI**, in both spellings, checked 2026-08-31 |
| `CONTRIBUTING.md` | states inbound = outbound (Apache-2.0 §5): no CLA, no DCO bot, and the PR is the record |
| `CITATION.cff` | added; GitHub renders a "Cite this repository" button |
| LICENSE | verified as the canonical verbatim text. The `[yyyy] [name]` on its line 189 is the appendix's how-to-apply instructions, not an unfilled template |

**Remaining, in the web UI after recreation**, with nothing else blocking.
Each one's later record is in
[Repository settings, as recorded](#repository-settings-as-recorded).

| # | what remained | what it was for, and what became of it |
|---|---|---|
| 1 | Settings → Code security: Dependabot alerts and security updates, secret scanning and push protection, and **private vulnerability reporting** | private reporting is what both SECURITY.md and CODE_OF_CONDUCT.md point at |
| 2 | description and topics; branch protection on `main`, last | |
| 3 | README badges (CI, license) | they only render once public, so add them then |
| 4 | **a weekly native leak check**, `docs/PLAN.md` task 18 | done 2026-09-03: `.github/workflows/leakcheck.yml`, Mondays and on demand, on ubuntu and macOS, with a control leak that must be caught. The "clean today" baseline this item rested on turned out to be a blind check, and task 18 has the numbers |

### Going public (2026-08-31)

**Decided on 2026-08-31: go public now, rather than at v0.1.0.** The Actions
quota forced the timing
([Actions quota: exhausted on day one](#actions-quota-exhausted-on-day-one-2026-08-31)),
but the repo was already prepared for it (R1-R3, R6).

The pre-flight audit, all of it verified rather than assumed:

| check | result |
|---|---|
| credentials | **none** in HEAD, or in any of the 1,528 objects in history |
| data files (hard rule 1) | **none ever committed**: 282 distinct paths have existed across all history, and none is a `.csv`/`.parquet`/`.npy`/etc. |
| authors and committers | **every one is a noreply address.** The earlier rewrite holds, and the two local refs that still carried the old one are gone |
| the maintainer's personal address | **one leak found and scrubbed**, below |
| the crates and the sdist | `publish = false` on all four crates, and LICENSE ships in the sdist |
| machine-local files | only `.vscode/settings.json` and the (untracked, gitignored) `mutants.out/` show up, and the former is portable (`${env:HOME}`) |
| community health files | `CODE_OF_CONDUCT.md` added, the last one missing. Its enforcement channel is GitHub private reporting, deliberately not an email, for the same reason as `SECURITY.md` |

**The leak: this very file quoted the maintainer's personal address**, in
the paragraph describing its removal from history. Redacting HEAD was not
enough, because the address lived in one commit's diff *and* its message.
So the eleven commits from that point to the tip were rewritten and
force-pushed. The tree hash of the tip is byte-identical before and after,
which is the proof that nothing but the address changed. The old commit is
deliberately not named here. A public file saying "the address is in commit
`abc1234`" is a signpost to it, which is the same mistake in a different
form.

**What a rewrite cannot reach.** A commit stays retrievable by full SHA for
as long as *any* ref on the remote holds it. Pull-request refs
(`refs/pull/N/head`) are kept by GitHub even after the branch is deleted and
the PR is closed. Getting those collected requires GitHub Support, or, on a
repository this young, deleting and recreating it.

**CI changes that go with it:** the full three-OS matrix on every push and
pull request, and `paths-ignore` deleted. That flag does not come back, even
as an optimisation. If CI becomes a required status check, a doc-only pull
request would never run it, and so never become mergeable.

**Done, 2026-09-02: the repository was deleted and recreated public, and 163
commits pushed to the empty repo.** What came back by itself: every
community health file, `dependabot.yml`, all four workflows, and the matrix,
which widens on `github.event.repository.private` without being touched.
What did **not** come back were the repository settings, which are not files;
[Repository settings, as recorded](#repository-settings-as-recorded) lists
them. Actions history and the open dependabot pull requests did not come
back either, and are disposable: dependabot opened a fresh one within a
minute of the push.

**The `pypi` environment that `release.yml` names (`environment: pypi`) must
exist again *before any tag is pushed*,** or the publish job fails after the
wheels are built. PyPI's trusted-publisher configuration is keyed by owner /
repository / workflow / environment *names*. So recreating the repository
under the same name leaves it valid: there is nothing to redo on PyPI's
side, but the environment is not optional.

**Done, 2026-09-03, through the REST API rather than the settings pages:**
Dependabot alerts and security updates, the description, homepage and
topics, and two rulesets, each recorded in the table below. Four things were
left to the owner, each of them a decision rather than a setting. They were
the `pypi` environment, private vulnerability reporting, the PyPI pending
publisher, and a `workflow_dispatch` rehearsal of `release.yml` on `main`.
The rehearsal ran the same day, and what it found is in
[The six faults of 2026-09-03](#the-six-faults-of-2026-09-03).

**The first push also proved the `paths-ignore` warning above, in the
cheapest possible way.** It ended in two documentation commits, the filter
matched them, and **no CI ran at all** on the 163 commits. The flag is gone,
and `benchmark.yml` gained a `push:` trigger, keeping a paths filter of its
own. It is reported-never-gating, so a filter there cannot block a merge.

### Repository settings, as recorded

Every repository setting this document tracks, with what was recorded of it
and when:

| setting | the record |
|---|---|
| branch protection on `main` | 2026-08-31 (R4): require CI to pass, no force-push. 2026-09-02: did not come back with the recreation. 2026-09-03: a ruleset, so `main` cannot be force-pushed or deleted |
| `v*` tags | 2026-09-03: a ruleset, so `v*` tags cannot be moved or deleted once they exist, with no bypass list; the owner is bound too, and disabling the ruleset is the only way round it. 2026-09-06: [the release gate](#the-release-gate) records the `release tags are immutable` ruleset over `refs/tags/v*`, with `deletion` and `update` rules and an empty bypass list |
| description, homepage and topics | 2026-08-31 (R4): description and topics. 2026-09-02: did not come back. 2026-09-03: the description, the homepage (the Pages site) and the topics set |
| Dependabot alerts and security updates | 2026-08-31: remaining, under Settings → Code security. 2026-09-02: did not come back. 2026-09-03: re-enabled. 2026-10-06: security updates on; they opened #4 (urllib3, `uv.lock`) and #5 (rustls, `Cargo.lock`), both merged that day |
| secret scanning and push protection | 2026-08-31: remaining, under Settings → Code security. 2026-10-06: both on, as `security_and_analysis` in `gh api repos/hgilde/polars-online` reports |
| private vulnerability reporting | 2026-08-31 (R4 and the sweep): enable it; it is the enforcement channel `SECURITY.md` and `CODE_OF_CONDUCT.md` both point at. 2026-09-02: did not come back. 2026-09-03: left to the owner. 2026-10-06: on (`gh api repos/hgilde/polars-online/private-vulnerability-reporting`) |
| CodeQL code scanning | 2026-10-06: GitHub's default setup, not a workflow in `.github/workflows/`: actions, Python and Rust, weekly, configured on 2026-09-07 (`gh api repos/hgilde/polars-online/code-scanning/default-setup`) |
| the `pypi` environment | 2026-09-02: must exist again before any tag is pushed. 2026-09-03: left to the owner, with the owner as required reviewer and a `v*` tag rule. 2026-09-06: the `Pypi` environment, with the owner as a required reviewer and self-review allowed |
| the PyPI pending publisher | 2026-09-03: left to the owner, as `polars-online` / `hgilde` / `polars-online` / `release.yml` / `pypi` |
| README badges (CI, license) | 2026-08-31: they only render once public, so add them then |
| the `github-pages` environment | 2026-09-06: briefly carried a required reviewer, and no longer does |
