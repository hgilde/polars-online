<!-- kind: reference -->
# Release readiness

This page is the reference, for a contributor who changes the release
workflow or cuts a release, on what a release of polars-online is held to
before anything reaches PyPI. That is the release gate, the README as PyPI
shows it, the budgets baseline and the tests that hold the workflow.

**A release is tested on every OS, and at both ends of the Polars range,
before anything is uploaded.** A release is dispatched, never tagged by
hand, and three things make it safe. A gate stands in front of the one
step nothing can undo, the upload. The tag exists only once the release
does. And a run that fails anywhere stops before the upload, so a broken
release is fixed and dispatched again, and no version is lost. [Cut a
release](contributing/cut-a-release.md) has the steps, and [Versions and
the Polars pin](VERSIONS.md) the range of Polars a release declares and
what each kind of release may change.

## The release gate

**The one human step in a release is approving the PyPI upload, the one
step that cannot be undone.** The upload is the `publish to PyPI` job,
which declares the `Pypi` environment (`release.yml` spells it
`environment: pypi`). That environment has the owner as a required
reviewer, with self-review allowed: with a single reviewer, forbidding
self-review would deadlock every release. The click is a button on the run
page, and GitHub emails when one is waiting.

**The job asks for approval only after every job it needs is green.** The
legs that move a dependency leave out two pytest markers. `soak` marks the
opt-in long runs over millions of rows. `pins` marks the tests that
assert this repository's own Polars pin, which cannot pass once Polars has
moved (`pyproject.toml`).

| job in `release.yml` | what it checks or makes | holds back the upload |
|---|---|---|
| `version` | the version agrees in its six places, the CHANGELOG has its section, and, publishing, the tag is new and the run is on `main` | yes, and it runs before anything is built |
| `ci` | the whole of `ci.yml`, called: lint, and the Rust and Python suites on Linux, macOS and Windows | yes |
| `budgets` | the whole of `budgets.yml`, called: every instruction count against its budget, and each case's drift since the last release against the sign-off | yes |
| `baseline` | the run's instruction counts and each OS's floor-Python memory, merged into `budgets-baseline.json` for the release's assets; refused if a budgeted quantity is missing or two legs disagree ([The budgets baseline](#the-budgets-baseline)) | yes |
| `sdist` | the source distribution | yes |
| `build` | all six wheels, and the command-line binaries on the five targets that build one, all but musl. The Linux binary is built in the wheels' manylinux2014 image, and refused above glibc 2.17 | yes |
| `wheel-size` | every wheel against the README's install *size* row, and their exact sizes for the release's notes ([The README's size row](#the-readmes-size-row)) | yes |
| `write-state`, then `read-state` | the cross-OS state hand-off: a state written on macOS is loaded, and its stream continued, on Windows and Linux | yes |
| `next-polars` | the Python suite, less its `soak` and `pins` tests, in two legs: on the newest Polars the declared range admits, and on the next major | the first leg, the blocking one; the second is advisory |
| `floor-polars` | the same suite on the floor, 1.37.0, of the declared range: exactly the py-polars and runtime package that `pyproject.toml`'s floor names, read by `scripts/polars_floor.py`. A test that needs a newer Polars skips there by version (`tests/polars_version.py`) | yes |
| `next-numpy` | the same suite in two legs, with only NumPy upgraded: on the newest NumPy, which the optional extra's `numpy>=1.26` admits, and on NumPy's next release candidate | the first leg only; the second is advisory |
| `publish to PyPI` | the upload | it is the gate |
| `tag`, then `release` | the tag `v<version>` on the tested sha, annotated with the CHANGELOG's section, then the GitHub release page with the wheels, the sdist, the command-line binaries and `budgets-baseline.json`, its notes the CHANGELOG's section and the wheels' exact sizes. Each tries three times, with a growing pause, and a retry after an attempt that landed succeeds: the tag on the remote at the tested sha is success, at another sha a failure, and a release that exists is finished rather than made again | no: they run after the upload |
| `verify` | `polars-online==<version>` installed from PyPI on Linux, macOS and Windows, its versions and schema checked and one fit run ([step 10](contributing/cut-a-release.md#steps)) | no: it reports on what was uploaded |

**Everything before the approval can be retried, and nothing after it can
be undone.** That is why the gate sits where it does: in front of the one
step nothing can undo, and in front of nothing else.

| side of the gate | what happens there | can it be undone |
|---|---|---|
| before | dispatching a release, re-running a failed job, canceling a run, declining the approval | yes: all of it can be retried, none of it reaches a user, and none of it creates a tag |
| after | the upload, then the tag, the GitHub release and the check of what PyPI serves | no: PyPI never allows a version number to be reused, even after a yank, and a `v*` tag cannot be moved or deleted. A failed `tag`, `release` or `verify` job can still be re-run: the tag goes on the same sha, and the release is finished where it stopped |

**A tag that turns out to be wrong is *not* fixed by moving it.** The
`release tags are immutable` ruleset covers `refs/tags/v*` with `deletion`
and `update` rules and an empty bypass list, so the owner is bound too
([Repository settings](contributing/repository-settings.md)). It allows
*creating* a tag, which is all the workflow does. The fix ships as the
next version. That agrees with PyPI's rule: version numbers are single-use
on both sides. `v0.11.0` is such a tag: it names a commit whose CI failed,
and nothing was ever published from it.

**A tag pushed by hand starts nothing, and it uses up its version.**
Release tags cannot be moved or deleted, and
`scripts/release_version.py --publish` refuses a version whose tag exists.
So a version tagged by hand can never be released.

**One deploy deliberately waits for no approval: the Pages deploy.** The deploy in
`ci.yml` republishes the API reference on every push to `main`, and is as
reversible as the next push. The `github-pages` environment carries no
required reviewer. The release's call of `ci.yml` grants the deploy's
scopes, since a called workflow may ask for no more than its caller
grants. The deploy runs on a push alone, so under a release it is skipped
(`tests/test_release_workflow.py`).

## The README's links on PyPI

**The workflow rewrites the README's relative links for PyPI.** PyPI shows
the README as the project page, where a relative link to another file
resolves against pypi.org and is a 404. So both jobs that build a package
first run `scripts/pypi_readme.py --ref <the tag the run will create>`, or
the sha in a run with `publish` off. It points each such link at the file
on GitHub as it is at the released commit. The README in the repository
keeps its relative links.

## The README's size row

**The README states one install size for every platform, and every
release holds its wheels to it before the upload.** The row reads "*A*
to *B* MB to download and *C* to *D* MB installed, by platform", and
names no version. A wheel's download is its file's size. What it installs
is the sum of its members' unpacked sizes. Each is in megabytes of 10^6
bytes, rounded to the whole number the row states. The release's
`wheel-size` job measures every wheel the build matrix made, fails if one
falls outside either range, and keeps each wheel's exact sizes, which the
release job writes into the GitHub release's notes
(`scripts/wheel_size.py`). So the row is edited before a release, by the
change that moved a wheel out of it, and only then. A row that names no
release needs no refresh after one.

## The budgets baseline

**The drift each budget run reports is read from the last release's own
measurements, which the release publishes beside its wheels.** The
instruction counts and the memory budgets hold each push to a fixed number
([Budgets that only tighten](TESTING.md#budgets-that-only-tighten)). Every
run also reports how far each case moved since the last release, and a
release fails on a case that moved more than 10% either way without a
sign-off for it.

**Those release numbers are the release's own run's.** The `baseline` job
merges the counts its `budgets` call made and the memory each OS's `ci`
leg measured at the floor Python into one file, `budgets-baseline.json`,
with the version and the sha. It refuses a measurement that is missing
for any budgeted quantity, or that two legs disagree on, before the
upload, while the release can still be dispatched again. The release job
attaches the file to the GitHub release.

**Every later run of `scripts/budgets.py` reads it from the newest
release,** by GitHub's `releases/latest/download/budgets-baseline.json`,
which names no version, and needs no token, since the repository is
public. CI fails when it cannot read it, so a missing asset cannot quietly
turn the drift and the sign-off off. Elsewhere, the report says it read
no baseline and the budgets are held as ever.

## The tests that hold the workflow

**Three test files hold the workflows' shape, and they run in the suite.**
So a change to a workflow that breaks the shape fails before it reaches a
release:

| test | what it holds |
|---|---|
| `tests/test_release_workflow.py` | a dispatch is the only trigger, with `publish` off unless asked; the version is checked before anything is built; CI runs inside the release on every OS, and its Pages grant is inert; the upload waits for every job, and the tag for the upload, on the tested sha; the NumPy legs; the floor leg, read from `pyproject.toml`, and the canary's monthly run of it; the floor Python in every leg that moves a dependency; the Linux command-line binary built in the manylinux2014 image and held to glibc 2.17; the budgets baseline merged and attached; every wheel held to the README's size row; the tag and the release each tried three times; the `verify` job's install from PyPI on every OS; only the release jobs write; and `scripts/release_version.py`'s rules |
| `tests/test_release_packaging.py` | the release job's steps run for real: the files it collects, each binary named by its target, the Windows `.exe` suffix kept, and no binary sent to PyPI; the tag's push against a local remote, which lands once, succeeds when the tag is there on the tested sha, fails when it is on another and gives up after three tries; the release's creation against a stand-in for `gh`, finished rather than repeated when an attempt landed; and its notes |
| `tests/test_ci_cost_policy.py` | every workflow bounded by timeouts and a concurrency group, releases queueing rather than canceling, and the rest of the CI policy ([The policy, as the test holds it](TESTING.md#the-policy-as-the-test-holds-it)) |
