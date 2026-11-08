<!-- kind: reference -->
# Release readiness

This page is the reference, for a contributor who changes the release
workflow or cuts a release, on what a release of polars-online is held to
before anything reaches PyPI.

**A release is tested on every OS, and at both ends of the Polars range,
before anything is uploaded.** A release is dispatched, never tagged by
hand, and three things make it safe. A gate stands in front of the one
step nothing can undo, the upload. The tag exists only once the release
does. And a run that fails any blocking job stops before the upload, so a
broken release is fixed and dispatched again, and no version is lost.
[Cut a release](contributing/cut-a-release.md) has the steps, and
[Versions and the Polars pin](VERSIONS.md) the range of Polars a release
declares and what each kind of release may change.

**Nothing is committed after a release.** What the next release compares
itself with is read from this one where it was published: its state files
from its wheels on PyPI ([State files across
releases](VERSIONS.md#state-files-across-releases)), its measurements from
an asset of its GitHub release ([The budgets
baseline](#the-budgets-baseline)).

## The release gate

**The one human step in a release is approving the PyPI upload, the one
step that cannot be undone.** The upload is the `publish to PyPI` job,
which declares the `Pypi` environment (`release.yml` spells it
`environment: pypi`). That environment has the owner as a required
reviewer, with self-review allowed: with a single reviewer, forbidding
self-review would deadlock every release. The click is a button on the run
page, and GitHub emails when one is waiting.

**The job asks for approval only after every job it needs is green.** The
legs that move a dependency leave out three pytest markers. `soak` marks
the opt-in long runs over millions of rows. `pins` marks the tests that
assert this repository's own Polars pin, which cannot pass once Polars has
moved (`pyproject.toml`). `oracle_grid` marks the heavy grids checked
against third-party oracles, which `oracles.yml` runs on every push.

| job in `release.yml` | what it checks or makes | holds back the upload |
|---|---|---|
| `version` | the version agrees in its six places; the CHANGELOG has its section; the example pin in `docs/VERSIONS.md` is on the version's minor. Publishing, it also checks that nothing is left under `[Unreleased]`, that no spec or kind in `tests/test_released_state.py`'s `PARAMETER_INTRODUCED` or `INTRODUCED` names a newer version, that the tag is new and that the run is on `main` | yes, and the wheels and the sdist are built only after it |
| `ci` | the whole of `ci.yml`, called: lint, and the Rust and Python suites on Linux, macOS and Windows | yes |
| `budgets` | the whole of `budgets.yml`, called: every instruction count against its budget, and each case's drift since the last release against the sign-off | yes |
| `baseline` | the run's instruction counts and each OS's floor-Python memory, merged into `budgets-baseline.json` for the release's assets; refused if a budgeted quantity is missing or two legs disagree ([The budgets baseline](#the-budgets-baseline)) | yes |
| `sdist` | the source distribution | yes |
| `build` | all six wheels, and the command-line binaries on the five targets that build one, all but musl. The Linux binaries are built in the wheels' manylinux2014 image, and refused above glibc 2.17 | yes |
| `wheel-size` | every wheel against the README's install *size* row, and their exact sizes for the release's notes ([The README's size row](#the-readmes-size-row)) | yes |
| `write-state`, then `read-state` | the cross-OS state hand-off: a state written on macOS is loaded, and its stream continued, on Windows and Linux | yes |
| `next-polars` | the Python suite, less its `soak`, `pins` and `oracle_grid` tests, in two legs: on the newest Polars the declared range admits, and on the next major | only the first leg (the second is advisory) |
| `floor-polars` | the same suite on the floor, 1.37.0, of the declared range: exactly the py-polars and runtime package that `pyproject.toml`'s floor names, read by `scripts/polars_floor.py`. A test that needs a newer Polars skips there by version (`tests/polars_version.py`) | yes |
| `next-numpy` | the same suite in two legs, with only NumPy upgraded: on the newest NumPy, which the optional extra's `numpy>=1.26` admits, and on NumPy's next release candidate | only the first leg (the second is advisory) |
| `publish to PyPI` | the upload | it is the gate |
| `tag`, then `release` | the tag `v<version>` on the tested sha, annotated with the CHANGELOG's section, then the GitHub release page with the wheels, the sdist, the command-line binaries and `budgets-baseline.json`, its notes the CHANGELOG's section and the wheels' exact sizes | no: they run after the upload |
| `verify` | `polars-online==<version>` installed from PyPI on Linux, macOS and Windows, its versions and schema checked and one fit run ([the last step](contributing/cut-a-release.md#steps)) | no: it reports on what was uploaded |

**The `tag` and `release` jobs each try three times, with a growing pause,
and a retry after an attempt that landed succeeds.** A push or a creation
whose answer was lost may have landed, so after each failure the job asks
the remote. The tag there at the tested sha is success, and at another sha
a failure no retry can fix. A release that exists is finished rather than
made again.

**Everything before the approval can be retried, and nothing after it can
be undone.** That is why the gate sits where it does: in front of the one
step nothing can undo, and in front of nothing else.

| side of the gate | what happens there | can it be undone |
|---|---|---|
| before | dispatching a release, re-running a failed job, canceling a run, declining the approval | yes: all of it can be retried, none of it reaches a user, and none of it creates a tag |
| after | the upload, then the tag, the GitHub release and the check of what PyPI serves | no: PyPI never allows a version number to be reused, even after a yank, and a `v*` tag cannot be moved or deleted. A failed `tag` or `release` job can be re-run, unless `main`'s workflow files changed since the tested sha. [If a job fails](contributing/cut-a-release.md#if-a-job-fails) has both cases |

**A tag that turns out to be wrong is *not* fixed by moving it.** The
`release tags are immutable` ruleset covers `refs/tags/v*` with `deletion`
and `update` rules and an empty bypass list, so the owner is bound too
([Repository settings](contributing/repository-settings.md)). It allows
*creating* a tag, which is all the workflow does. The fix ships as the
next version. That agrees with PyPI's rule: version numbers are single-use
on both sides.

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

## No rehearsal

**A release is not rehearsed, not even after a change to `release.yml`.**
A push to `main` runs CI on all three OSes and the instruction budgets, and
a publishing run tests everything again before it asks for approval. It
stops before the upload, and creates no tag, when any blocking job fails.
A failed release is fixed on `main` and dispatched again, and no version
is used up.

**Some jobs run only in a release, so a release is where they first
fail:**

| job | what it makes or runs |
|---|---|
| `version` | the version's checks, as publishing asks them |
| `sdist` | the source distribution |
| `build` | every wheel and the command-line binaries |
| `wheel-size` | the wheels' sizes |
| `write-state`, `read-state` | the state written on one OS and read on the others |
| `next-polars`, `floor-polars`, `next-numpy` | the suite on the newest Polars, on the floor of the Polars range and on the newest NumPy |
| `baseline` | the instruction counts and each OS's memory, merged into one file |
| `publish to PyPI` | the upload |

The floor of the Polars range can be run locally before the dispatch
([Cut a release](contributing/cut-a-release.md), step 11). The jobs after
the upload are tested in the suite: `tests/test_release_packaging.py` runs
the scripts of `tag` and `release` against a local remote and a stand-in
for `gh`, and `tests/test_release_workflow.py` holds the `verify` job's
install and its script.

## The runs that watch the dependencies

**Two kinds of run look for a break, in Polars and in NumPy, the one
optional dependency** (`polars-online[numpy]`, `numpy>=1.26`). The canary
reports one, and the legs every release runs before it publishes withhold
the release. Each leg of `release.yml` runs in every dispatched release,
before any tag exists:

| run | when | what it resolves to | a red run |
|---|---|---|---|
| `release.yml`'s `next-polars`, the newest in-range leg | every dispatched release | the newest stable inside `<3` | withholds the release until the range is capped or the code fixed |
| `release.yml`'s `floor-polars` | every dispatched release | exactly the floor `pyproject.toml` declares, 1.37.0, with its runtime package | withholds the release until the code is fixed or the floor moved up |
| `release.yml`'s `next-polars`, the next-major leg | every dispatched release | unpinned, prereleases allowed | is advisory, and withholds nothing |
| `release.yml`'s `next-numpy`, the newest NumPy | every dispatched release | the newest NumPy | withholds the release |
| `release.yml`'s `next-numpy`, NumPy's next release candidate | every dispatched release | its next release candidate | is advisory, and withholds nothing |
| `polars-canary.yml`, the canary | weekly, on Mondays, and on demand | the newest py-polars, prereleases included, with the range dropped, for the Rust tests and the Python suite | is the notification, and withholds nothing |
| the canary's `next-numpy` | weekly, on Mondays, and on demand | NumPy's next release candidate | is the notification, and withholds nothing |
| the canary's `floor-polars` | the 1st of each month, and on demand | the floor, 1.37.0, as every release runs it | is the notification, and withholds nothing |

**The canary ([`polars-canary.yml`](../.github/workflows/polars-canary.yml))
drops the range from `pyproject.toml` and installs the newest py-polars,
prereleases included.** It builds the wheel as CI does, then runs the
suite, all but its `soak`, `pins` and `oracle_grid` tests. Only Polars
moves in that run, so a red canary means Polars broke this library. Only
NumPy moves in the NumPy runs, the release's and the canary's, so a red
one names NumPy. Only the canary's monthly floor run reads the declared
range.

**The newest in-range leg blocks the publish, because `<3` admits every
1.x and 2.x.** A resolver can hand someone a Polars newer than the one the
wheel was built against, the day after that Polars ships. A pass on the
pinned version alone would not show that the range holds. The next-major
leg is early warning for [a new
major](contributing/move-the-polars-range.md#move-the-ceiling-up-to-a-new-major).

**The floor leg blocks the publish too, because a run on the newest Polars
says nothing about the oldest one the range admits.** A resolver keeps a
user's Polars while it is inside the range, so a change that needs a newer
Polars than the floor would break that user's install. The newest-Polars
runs prove the ceiling, and only the floor runs prove the floor.

## The state files of earlier releases

**`tests/test_released_state.py` holds every release's state files to the
promise [State files across
releases](VERSIONS.md#state-files-across-releases) makes.** It installs
each release into the comparison's cache, and runs
`scripts/release_probe.py --states` under it: each spec of the workload is
fitted on the first half of the stream and saved. This build must refuse
each 0.x file by its version, with the message to "fit it on its input".
It loads each 1.x file with its own spec, and fits the second half beside
a bank of its own that fitted both halves. Every float, the closed groups,
the coefficients and `marginal`'s table must agree to the bit, apart from
the cases VERSIONS lists and the specs whose numbers a CHANGELOG entry
moved since that release, which the test names with the entry. It runs in
the suite, and is skipped offline, since it downloads the releases.

**The 1.x releases are read from PyPI, not listed by hand.** The four
releases before 1.0 are named in the test. Every 1.x release is read from
PyPI's JSON description of the package when the suite is collected, every
final version not yanked, so a release is held from the moment it is
uploaded, with no edit to the test. A spec that needs a parameter newer
than a release is entered in `PARAMETER_INTRODUCED`, and a new kind in
`INTRODUCED` ([Add a spec
parameter](contributing/change-the-api.md#add-a-spec-parameter)).

**Frozen states hold the format inside the repository, and the
released-state test holds it against what PyPI shipped.** A state of every
`ModelState` variant, a bank file, a `with_windows` state and a
`refresh_time` state are frozen at each schema this build loads, from
`MIN_SCHEMA_VERSION` and `MIN_BANK_SCHEMA_VERSION` to `SCHEMA_VERSION`
(`crates/online-core/tests/state_fixtures.rs`,
`crates/online-polars/tests/state_fixtures.rs`). For 1.0.0 all three
constants stood at 61. [The versions a state file
carries](#the-versions-a-state-file-carries) names where each version is
kept.

## What the API snapshot pins

**Each stable part is pinned by a section of `tests/api_surface.txt`,
except the command line's exit status, which a test holds.** The unstable
names that stand in a section, `po.sim`, `po.corr`, `po.gbdt`,
`ArrowStruct`, `fit_predict_arrow` and `predict_arrow`, each carry the
label `# unstable` on their line, so the snapshot says which of its lines
carry no promise:

| stable part | pinned by |
|---|---|
| `__all__`, each function with its signature | `[package]` |
| the constructors' keyword names and the defaults they set in Python | `[common parameters]`, which lists once the keywords every spec shares (`CommonKwargs` in `python/polars_online/_kwargs.py`), and `[spec constructors]` |
| the defaults that resolve in Rust, `min_weight`'s per-kind rule included | `[resolved defaults]`, read off the bank's own build of each kind (`_polars_online.resolved_defaults`); `tests/test_spec_defaults.py` holds MODELS.md's warm-up tables to it |
| `ModelBank`'s methods, `__init__` included | `[ModelBank]` |
| the `.online` namespace | `[frame namespaces]` |
| the frames the bank returns: `groups()`, `summary()`, `describe()`, `coef()`, `last_row()`, `marginal()`, `audit()`, `check()`, `solve_failures()`, `gram()`'s keys, and `po.spec`'s `output_index()`, `coef_index()`, `coef_fields()` | `[frame columns]`, each column with its dtype |
| the closed-groups frame's columns | `[closed_groups columns]`, each column with its dtype, in order |
| the helper modules' functions, each with its signature | `[helper modules]`, every module the package's directory holds: today `corr`, `eval`, `gbdt`, `gram`, `ops`, `sim` and `stream` |
| `po.stream.embargo`'s two columns, `_online_role` and `_online_role_weight` | `[helper modules]`, as the default of `embargo`'s `role`, the first column's name, which the second's extends; `tests/test_label_delay.py` reads both |
| output field names and their dtypes | `[output field grammar]`, 29 spec shapes, one with every `emit_*` turned on |
| the words each string-valued parameter takes, and `withheld_reason`'s | `[enum values]`, read from each parameter's refusal of a word it does not take |
| the TOML config keys: the run config's, a spec's, and each model type's | `[toml keys]`, read from serde's refusal of a key a table has not got |
| the command line's flags, and whether each takes a value | `[cli flags]`, from `polars-online --help` |
| the command line's exit status: 0, 1 for a refusal or a run error, 2 for a usage error | [*Exit status*](RUNNER.md#exit-status) and the command line's test `the_exit_status_is_0_for_a_run_1_for_a_refusal_and_2_for_usage` (`crates/online-cli/tests/run.rs`), not the snapshot |
| the environment variables, `POLARS_ONLINE_MAX_THREADS` and `POLARS_ONLINE_TIMING` among them | `[env vars]`, from every read in `crates/*/src` and the package |

**A default that resolves in Rust is pinned where the bank builds it.**
A builder passes `None` for it, so a constructor's signature shows `None`.
The snapshot's `[resolved defaults]` reads each off the bank's own build
of every kind's least spec: `Spec::check`, then the constructor of every
group's stream, through `_polars_online.resolved_defaults`. It renders one
line per kind and field. The lines cover the model's configuration, what
the model derives from a field left unset (`bocpd`'s priors, `rcov`'s ring
and window, `marginal`'s bin budget), the stream's gates and diagnostics'
settings, and the clock policy. Variants add the defaults that only an
option reads, such as `sgd`'s `huber_delta` under `loss = "huber"`. Other
tests hold some of these defaults to what they are for:

| default | set in | pinned by |
|---|---|---|
| every default a builder leaves to Rust, `min_weight`'s per-kind rule included | Rust | the snapshot's `[resolved defaults]`; `tests/test_spec_defaults.py` holds MODELS.md's warm-up tables to it |
| the solve cadence of `ewridge`, `lasso`, `huber` and `quantile`: by weight, `half_life/50` of clock in steady state | Rust | `[resolved defaults]` (`solve_every`, `solve_share`); `tests/test_solve_cadence.py` measures it |
| the clock settings, the readiness gates, and `min_weight`'s count floor | Rust | `[resolved defaults]`; `crates/online-polars/tests/spec_defaults.rs` |
| `standardize`: true for `kalman`, `sgd`, `pa`, `kmeans` and `micro`, false for `ewridge`, `huber` and `quantile`; `lasso` has no such keyword | Python | the snapshot |
| `average_eta = 1.0` | Rust | `[resolved defaults]`; `spec_defaults.rs` |
| `clip_gradient = 1e3`, for `sgd` | Rust | `[resolved defaults]`; `spec_defaults.rs`; `tests/test_sgd.py` checks that the default clip keeps a Poisson fit stable |

`docs/VALIDATION.md` justifies the defaults it measures, and
`tests/test_validation_doc.py` pins that they are still the measured
optimum.

**An output field's name is rendered in one place.** The names are
generated by `format!` over ridge values, half-life suffixes, feature-set
labels, quantile levels and target names. `num_label()` in
`crates/online-polars/src/spec.rs` is the only place a number in a name is
rendered. A number outside `[1e-6, 1e7)` renders compactly, as `1e-300`.
Rust's float `Display` never uses scientific notation, so a legal
`ridge = 1e-300` would otherwise make a 311-character name. A target literally named
`y__r0.5` renders like a grid field, so `Spec::check` refuses a spec
whose two outputs render to one field name, and `Bank::new` checks again. `test_names_match_the_realized_struct_for_every_model`
(`tests/test_portability.py`) checks, in 40 cases, that the fields a spec
declares are the fields its output has.

### The versions a state file carries

**A state file is versioned msgpack, with a schema version and a bank
`format_version`.** `SCHEMA_VERSION` (`crates/online-core/src/lib.rs`, and
`po.schema_version()` from Python) moves with every layout change. The
bank's `format_version` is 2, or 3 when a spec carries a clock parameter
as a duration. The bank refuses a file older than
`MIN_BANK_SCHEMA_VERSION` (`crates/online-polars/src/bank.rs`) by its
version, and the models' own states have a floor of their own,
`MIN_SCHEMA_VERSION` (`crates/online-core/src/lib.rs`). A window run's
state, from `with_windows`, has its own version, `WINDOWS_VERSION`
(`crates/online-polars/src/windows_frame.rs`), and a change to it moves
the bank's schema too. The state of `po.stream.refresh_time` has its own
version, `REFRESH_VERSION` (`crates/online-polars/src/refresh.rs`).

### What is deliberately not done

**Three measures look like stability, and would take more work here than
they are worth:**

| do not | why |
|---|---|
| `__getattr__`-guard the submodules | `polars_online._spec` is importable, and someone will reach into it; that is what the underscore is for. Enforcing it would take more work than keeping users out of private modules is worth |
| freeze the API to prevent change | the snapshot test is for detecting change, not preventing it |
| add `cargo semver-checks` | it is the right tool for a published Rust API, and pure overhead otherwise. The Rust crates carry `publish = false` in their `Cargo.toml` and are not API; add it only if they are ever published |

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
| `tests/test_release_workflow.py` | a dispatch is the only trigger, with `publish` off unless asked; the version is checked before the wheels and the sdist are built; CI runs inside the release on every OS, and its Pages grant is inert; the upload waits for every job, and the tag for the upload, on the tested sha; the NumPy legs; the floor leg, read from `pyproject.toml`, and the canary's monthly run of it; the floor Python in every leg that moves a dependency; the Linux command-line binary built in the manylinux2014 image and held to glibc 2.17; the budgets baseline merged and attached; every wheel held to the README's size row; the tag and the release each tried three times; the `verify` job's install from PyPI on every OS; only the release jobs write; `scripts/release_version.py`'s rules, its refusal of a version older than an `INTRODUCED` or `PARAMETER_INTRODUCED` entry among them; and the throughput table's date, no older than the previous release |
| `tests/test_release_packaging.py` | the release job's steps run for real: the files it collects, each binary named by its target, the Windows `.exe` suffix kept, and no binary sent to PyPI; the tag's push against a local remote, which lands once, succeeds when the tag is there on the tested sha, fails when it is on another and gives up after three tries; the release's creation against a stand-in for `gh`, finished rather than repeated when an attempt landed; and its notes |
| `tests/test_ci_cost_policy.py` | every workflow bounded by timeouts and a concurrency group, releases queueing rather than canceling, and the rest of the CI policy ([The policy, as the test holds it](TESTING.md#the-policy-as-the-test-holds-it)) |

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
every field name the memory budgets' fullest workload writes.
`tests/test_release_compare.py` holds the workload to those field names,
and to each path in its `PATHS`, by the keyword that takes it:

| path | the keyword or setting |
|---|---|
| a temporal clock, with duration parameters | `clock`, and `half_life` as a duration |
| the blocked Gram on two threads | `gram_block_rows`, `gram_threads` |
| targets that go missing on different rows | `target_gaps="pairwise"` |
| a formula target | a window formula in `targets` |
| feature sets and a coefficient prior | `feature_sets`, `coef_prior` |
| `kalman`'s settings | `revert_half_life`, `share_p=True`, `q` with `obs_var`, `standardize=True`, a prior sized past `min_weight` |
| a drift reset | `drift_action="reset"` |
| the solve and coefficient cadences | `max_rows_between_solves`, `max_rows_between_coefs` |
| a reset at a step back | `restart_after_step_back` |
| decay given per row, as `lam` | `lam` |
| windows | `closed="both"`; the windows of `lasso` and `ew_class`; `marginal`'s `window_lags=True` |
| `marginal`'s bins | `bin_edges` |
| closing a group | `group_close="session"` |
| the gradient models | `ftrl`'s squared loss; `pa`'s modes `pa1` and `pa2`; `sgd`'s `adagrad` schedule and its losses `huber`, `quantile`, `epsilon_insensitive`, `poisson` and `logistic` |

**The script compares each spec's output fields, and the frame
`closed_groups` drains, bit by bit.** What it does not run is outside the
comparison. That is the window operators (`po.ewm_mean` and the rest,
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
