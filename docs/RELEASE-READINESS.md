# Release readiness and API stability

This document is for whoever cuts a release of polars-online, or asks which
Polars versions it promises and which were only measured. It began on
2026-08-31 as a proposal with two questions: what was left before the
repository went public, and what makes its API safe to promise. It has since
gained the release gate, the Polars measurements, and the records of going
public and of CI cost.

**Each section is dated where it was added, so read it as of its date.**
Where a dated figure has gone stale, it stays, and the present one follows it
with its evidence.

| section | what it covers |
|---|---|
| [Cutting a release](#cutting-a-release) | [the release gate](#the-release-gate-2026-09-06) · [rehearse before tagging](#rehearse-before-tagging) |
| [Which Polars versions are promised](#which-polars-versions-are-promised) | [the pin, and the two copies of Polars](#the-polars-pin-and-the-two-copies-of-polars-2026-08-31) · [2.0.0rc1, measured](#polars-200rc1-measured-2026-09-10) · [raising the ceiling](#raising-the-ceiling-to-a-new-major) |
| [Keeping the API stable](#keeping-the-api-stable) | [what the API is](#what-the-api-actually-is) · [the snapshot test](#s--the-mechanism-one-api-snapshot-test--done) · [the policy](#policy-to-write-down) · [what not to do](#what-not-to-do) · [the usability round](#api-usability-round-pre-users-window) · [the 2026-09 batch](#the-2026-09-batch-what-it-added-to-the-surface-tasks-4556) |
| [CI cost while the repo was private](#ci-cost-while-the-repo-was-private) | [the quota, spent on day one](#actions-quota-exhausted-on-day-one-2026-08-31) · [where the 80 minutes went](#ci-cost-where-the-80-minutes-went) · [what was unexpectedly expensive](#what-was-unexpectedly-expensive-2026-08-31) |
| [Going public, as recorded](#going-public-as-recorded) | [the suggested order](#suggested-order) · [R1–R6](#before-going-public-r1r6) · [the open-source sweep](#open-source-preparation-the-full-sweep-2026-08-31) · [going public](#going-public-2026-08-31) · [repository settings](#repository-settings-as-recorded) |

## Cutting a release

A release starts when a `v*` tag is pushed, and
`.github/workflows/release.yml` does the rest. Two things make it safe: a
gate in front of the one step nothing can undo, and a rehearsal that finds a
broken workflow before a tag does. Which version number a change needs is in
[Policy to write down](#policy-to-write-down). Widening the Polars range is
a minor release, as step 5 of
[Raising the ceiling to a new major](#raising-the-ceiling-to-a-new-major)
says.

### The release gate (2026-09-06)

**The one human step in a release is approving the PyPI upload, and it is
the only step that cannot fail.** The upload is the `publish to PyPI` job,
which declares the `Pypi` environment; `release.yml` spells it
`environment: pypi`. That environment has the owner as a required reviewer,
with self-review allowed, since with a single reviewer, forbidding it would
deadlock every release. The click is a button on the run page, and GitHub
emails when one is waiting.

**The job asks for approval only after every job it needs is green.** On
2026-09-06 that list read `needs: [sdist, build, read-state]`, and since
2026-09-10 (`e6da97c`) it also names `next-polars`, which tests the newest
Polars at the tag:

| job in `release.yml` | what it checks or makes | holds back the upload |
|---|---|---|
| `sdist` | the source distribution | yes |
| `build` | all six wheels, and the command-line binaries | yes |
| `write-state`, then `read-state` | the cross-OS state hand-off: a state written on macOS is loaded, and its stream continued, on Windows and Linux | yes |
| `next-polars` | the whole suite, in two legs: on the newest Polars the declared range admits, and on the next major | the first leg only; the second is advisory |
| `release` | the GitHub release page, with the wheels and CLI binaries attached | no: it runs while the approval waits |
| `publish to PyPI` | the upload | it is the gate |

**Everything before the approval can be retried, and nothing after it
can.** That is why the gate sits where it does: in front of the one step
nothing can undo, and in front of nothing else.

| side of the gate | what happens there | can it be undone |
|---|---|---|
| before | dispatching a rehearsal, re-running a failed wheel, cancelling a run, pushing the tag | yes: all of it is retryable, and none of it reaches a user |
| after | nothing: the upload is the last step | no: PyPI never allows a version number to be reused, even after a yank |

**A tag that turns out to be wrong is *not* fixed by moving it.** The
`release tags are immutable` ruleset covers `refs/tags/v*` with `deletion`
and `update` rules and an empty bypass list, so the owner is bound too. A
bad tag is left unapproved, and the fix ships as the next version. That
agrees with PyPI's rule rather than fighting it: version numbers are
single-use on both sides.

**Two things are deliberately *not* gated.** The GitHub release job runs
before the approval, so the wheels and CLI binaries are attached to the
release page while the decision is being made. It can be deleted and
recreated if wrong. The Pages deploy in `ci.yml` republishes the API
reference on every push to `main`, and is as reversible as the next push.
`github-pages` briefly carried a required reviewer on 2026-09-06, and no
longer does.

### Rehearse before tagging

**Rehearse `release.yml` before pushing a tag, because a tag runs the
workflow file *at the tag*.** A fault in the workflow therefore ships inside
the tagged commit. A `workflow_dispatch` rehearsal of `release.yml` on
`main` builds every wheel and publishes nothing. The rehearsal is not
optional.

Six faults were found on 2026-09-03, and every fix is in the tagged commit.
The first was found before the first rehearsal ran, and is one a rehearsal
would have caught. The rehearsals found the other five. The fourth, with
the fixes before it in, passed the state hand-off on both OSes and found
the sixth.

| # | where | what failed | why | the fix |
|---|---|---|---|---|
| 1 | the Intel macOS wheel | `release.yml` named the `macos-13` runner | GitHub has retired it: a tag would have built five wheels and published none | the label is `macos-15-intel` now |
| 2 | the aarch64 Linux wheel | cross-compiled from x64 in `manylinux2014-cross:aarch64`, it failed in `ring` 0.17, reached through polars-io's cloud feature | that image's GCC 4.8.5 does not define `__ARM_ARCH` (briansmith/ring#1728) | built the way polars builds its own: natively on GitHub's `ubuntu-24.04-arm` runner, where maturin-action picks the `manylinux2014_aarch64` image with a current GCC. The tag stays `manylinux_2_17`, and the native runner means the aarch64 CLI binary now ships too |
| 3 | the x64 Linux job | the host `cargo build` of the CLI could not execute its compiler wrapper | maturin-action exports `RUSTC_WRAPPER=sccache` into the job, but installs sccache inside the manylinux container | the CLI step clears the wrapper |
| 4 | the Intel macOS job | it hit the 90-minute timeout | 26 minutes of it were `uv sync` building the extension into a venv nothing in the job uses: maturin-action brings its own Python, and the CLI has no pyo3 in its tree | that step is gone, the timeout is 120, and rust-cache saves on failure, as it does in `ci.yml` |
| 5 | both `read state` jobs | `handoff.state: No such file or directory` | the artifact lands in the workspace root, but cargo runs an integration test with the *package* root as its working directory (`crates/online-polars/`, per the cargo reference) | the read side uses `${{ github.workspace }}`, as the write side already did; reproduced locally both ways before the fix |
| 6 | the Linux wheel jobs | the host `cargo build` for the CLI failed on `target/release/.cargo-build-lock` (Permission denied), and rust-cache's post step could not tar the tree | the wheel is built inside maturin-action's manylinux container, which runs as root over the bind-mounted workspace, so `target/` comes back root-owned | a `sudo chown -R` of `target` between the two steps fixes both |

**The fifth matters most, because it is rule 5 itself.** The state jobs
also lost their `uv sync` (17--23 minutes each): `online-polars` has no pyo3
in its tree, and the test is pure Rust. The sixth is why the Linux jobs had
never once restored a cache. Rehearsal one never reached it, because the
sccache wrapper failed first and hid it.

**The rehearsals do not measure the glibc floor of the Linux CLI binaries.**
Those are built on the runner's own glibc (Ubuntu 24.04, 2.39), not in the
manylinux2014 container the wheels come from. So their glibc floor is
whatever the toolchain emits, and nothing checks it. The wheels are
`manylinux_2_17`. A deployment on an older glibc builds the CLI from a
checkout, or the release job grows a container step for it.

## Which Polars versions are promised

**The declared range, `polars>=1.34.0,<3`, is measured, and Polars does not
promise it.** It is this project's empirical claim, resting on the
measurements in this section:

| part | what it rests on | where |
|---|---|---|
| the floor, 1.34.0 | `LazyFrame.collect_batches`, which py-polars added in 1.34.0; `tests/test_scaffold.py` pins the declared floor | [The floor and the ceiling](#the-floor-and-the-ceiling) |
| `ModelBank` alone, from 1.28.1 | `PySeries._export`, measured on 17 py-polars releases | [the matrix](#statically-linking-polars-will-break-users-on-other-versions) |
| the ceiling, `<3` | the whole suite on `2.0.0rc1`, and, at every tag, the blocking leg of `release.yml` on the newest version in range | [Polars 2.0.0rc1, measured](#polars-200rc1-measured-2026-09-10) |
| what Polars itself promises | nothing for `ModelBank` or the IO plugin; the Arrow PyCapsule output is Arrow's contract | [Which interfaces carry a promise](#which-interfaces-carry-a-promise) |
| the Rust `polars` | 0.55.2, pinned by `Cargo.toml` and linked into the wheel, so it never meets the user's | [the matrix](#statically-linking-polars-will-break-users-on-other-versions) |
| the `online` CLI | the Rust `polars` 0.55.2 alone: it never touches py-polars | [The other failure](#the-other-failure-is-the-canarys-own-hygiene) |

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

#### The floor and the ceiling

**The declared floor is 1.34.0, not 1.28.1 (2026-09-02).** The declaration
that day read `polars>=1.34.0,<2`. The table measured `ModelBank` and the
plugin, and those do work from 1.28.1. But `po.run` over a path or a plan
read with `LazyFrame.collect_batches` (E32), and so did the streaming query
form `lf.online.fit_predict` (E33). py-polars added `collect_batches` in
**1.34.0**. On 1.28.1–1.33 both failed with
`AttributeError: 'LazyFrame' object has no attribute 'collect_batches'`, in
24 of 27 runner tests. The canary tests the latest polars only, so it could
not see this. It was found while checking E33 against the floor, and the
floor now says what the package needs. The whole suite (1037 tests) passed
on 1.34.0, 1.38.1 and 1.44.1 with identical numbers.

**The floor has not moved since, though `po.run` has gone.** It left Python
in task 83 (`7d23a80`, 2026-09-17), and the command line kept the runner.
Today `ModelBank.fit(lf)`, `fit_predict_batches(lf)` and
`lf.online.fit_predict` read with `collect_batches`, as the note on the
range in `pyproject.toml` says. `tests/test_scaffold.py` pins the declared
floor, so a change to either has to change both.

**The ceiling is a bet that the interface holds.** On 2026-09-02 it was
`<2`, a bet on 1.x, and since 0.5.1 it is `<3`, a bet on the rest of 2.x,
made on the measurements in
[Polars 2.0.0rc1, measured](#polars-200rc1-measured-2026-09-10).
Two hedges back it, as the note in `pyproject.toml` records:
`polars-canary.yml` runs the suite against the latest polars weekly. `release.yml` runs it again at every tag, on the
newest version the range admits, and blocks the release if it fails. A third
hedge went with task 85: the expression plugin's ABI was version-negotiated,
and refused to load rather than misbehave. What remains on the paths that
are left is a loud failure rather than a negotiated one. A missing
`PySeries._export` is a clean `AttributeError` before any data moves.

#### Which interfaces carry a promise

**The matrix is measured, and the interfaces it rides on do not carry the
same weight of guarantee.** This correction came from reading what Polars
actually promises:

| interface | used by | what Polars promises | a break would show as |
|---|---|---|---|
| the expression plugin, **removed** in task 85 (2026-09-17) | `pl.col(..).online.<model>(..)` | it was the supported mechanism, with a MAJOR/MINOR handshake the loader checks before its first call | a refusal to load, rather than misbehaviour |
| pyo3-polars' extension types, `PyDataFrame`/`PySeries` | `ModelBank` | no stability guarantee beyond the latest definitions working with the latest version | a clean `AttributeError` before any data moves |
| the IO plugin, `polars.io.plugins.register_io_source` | `lf.online.fit_predict` | documented in the user guide, but decorated `@unstable` in polars (a warning only under `POLARS_WARN_UNSTABLE`); its contract is partly unwritten | a wrong row count, not a crash |
| the Arrow PyCapsule interface | `ModelBank.fit_predict_arrow` (task 86) | nothing: its contract is Arrow's rather than polars', and it covers the output side only | |

**The one path that carried a guarantee was also the one that could not
stream.** The expression plugin was removed because polars hands a stateful
user expression its whole column. The measurements above stand as the
record of what it did, and nothing below depends on it. The **Arrow
PyCapsule interface** now takes its place in the guarantee story. It covers
the output side only, so a call still crosses the `PyDataFrame` boundary on
the way in.

**`ModelBank` carries no guarantee.** The pyo3-polars README says the
`PyDataFrame`/`PySeries` types "are however only provided for convenience
and **do not have stability guarantees beyond that the latest definitions
should work for the latest version of Polars**". That is exactly the
`PySeries._export` call whose absence sets the 1.28.1 floor of the table.

**The unwritten part of the IO plugin's contract is its pushdowns.** Polars
pushes a projection, a predicate and a slice into a Python source, and does
not re-apply any of them afterwards, so the source honours all three itself
(`python/polars_online/_frame.py`). `tests/test_frame.py` checks every
pushdown against the collected frame, in both orders. A change in that
contract would show as a wrong row count, not a crash, which is why those
tests exist and why the canary runs them.

**The range stays, because it is measured**: the numbers are identical
across the releases, and the failure mode is a loud `AttributeError` rather
than a wrong answer. But it is this project's empirical claim, not Polars'.
A `ModelBank` break on a new Polars is expected maintenance: check that path
first, and the IO-plugin tests after it.

**2026-09-03: the expression plugin warned on every use** (`docs/PLAN.md`
§6, task 19). It shipped as before, and everything above still held for it.
What changed was that `pl.col(..).online.<model>(..)` issued
`InMemoryExpressionWarning` naming `lf.online.fit_predict`, because it was
the one O(data) form. Nothing here moved: the floor and ceiling were set by
`collect_batches`, not by the plugin, and the canary exercised all three
paths. For a few hours the same task had the plugin out of the wheel,
behind a cargo feature. That was reverted before release, so no wheel
published at the time lacked it.

#### "A frame allocated by one binary and freed by the other"

This is the genuinely dangerous version of the question, and the reason the
C Data Interface is the right mechanism. **A `SeriesExport` carries a
`release` callback into the binary that produced it, so each side frees its
own memory with its own allocator.** No Rust `DataFrame`, no `Drop` impl and
no raw buffer ownership ever crosses. This was confirmed in `polars-ffi`
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

- `polars_ffi` exports `MAJOR: u16 = 0` and `MINOR: u16 = 1`, and
  `_polars_plugin_get_version()` packs them into a `u32` as
  `(major << 16) | minor`.
- The loader in `polars-plan` reads that *before making any other call*,
  and **branches on MAJOR**:
  `if major == 0 { use polars_ffi::version_0::*; … }`. The module is
  literally named `version_0`. When the calling convention changes, polars
  adds `version_1` and a second branch, and plugins built against the old
  one keep working. Multi-version support is built into the loader.
- The call itself is pure C:
  `extern "C" fn(*const SeriesExport, usize, *const u8, usize, *mut SeriesExport, *const CallerContext)`.
- The payload is the **Arrow C Data Interface**: a formal cross-language
  specification, with its own stability guarantees and ownership protocol.
  It is the same one DuckDB, pyarrow and cuDF interoperate through.

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
its own `CARGO_TARGET_DIR`, the way `polars-canary.yml` does it. It pinned
the Python dependency, then ran `uv sync`, `maturin develop --release`, and
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

**`docs/STATE-WORKFLOW.md` §4 states R6, the one gap in the plan's state
workflow.** Polars drains a Python source before surfacing a later node's
error. So a query that fails *after* the bank still writes `save_state` with
the whole stream's state, while its own output is missing. The stability
note in that document's §6 says "R6 narrows if polars ever stops it". That
R6 is the state workflow's; this document's own R6 is the history scan.

**On 2.0.0rc1 it does narrow, and the narrowing is length-dependent.**
Measured with `chunk_rows=500` and a failing downstream cast:

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
licence to cache or skip a re-execution, and `docs/STATE-WORKFLOW.md` R2
measured how often the source runs. **Decided the same day in task 77**
(`4c545af`): `_frame.py` declares the source pure exactly when a run has
nothing to write, with no `save_state` and no `closed_groups` file.

#### What this does not settle

The run was one platform (macOS arm64), one Python (3.14), and a release
candidate.

**Superseded on 2026-09-10: `<2` became `<3` in 0.5.1.** What changed the
judgment was not new evidence about the candidate, but where the risk of
being wrong now lands. `release.yml`'s blocking leg runs the whole suite,
at the tag, on the newest version the declared range admits. So from 0.5.1
on, 2.0.0 final is tested before any wheel is published under a range that
includes it. The paragraph above was written when nothing checked the top
of the range at release time. Widening then would have meant declaring
support and finding out afterwards. See
[Raising the ceiling to a new major](#raising-the-ceiling-to-a-new-major),
below.

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
calls R6. The measurements are above, in *Polars 2.0.0rc1, measured*.

**It was raised on `2.0.0rc1`, before 2.0.0 final was on PyPI, and that was
deliberate.** It changes nothing for a user today, because installers do
not resolve to a release candidate. So every user still gets the newest 1.x
until 2.0.0 ships, and then gets it without waiting on a release of ours.
The blocking leg is what covers the difference between the candidate and
the final.

## Keeping the API stable

A change to the public API can break a user without raising anything. This
section says what the API is, how a change to it is caught, and the policy
for making one. It is the proposal of 2026-08-31, with dated notes where the
code has moved since.

### What the API actually is

The API is bigger than `__all__` suggests. The proposal found four surfaces,
in descending order of how easily a change could break someone.

**1. Output field names, the largest and least guarded.**

```
pred_y__r0.000001@h100    resid_y__r0.5@h100    absresid_q0.95_y0    n_eff@h500
```

Users write `out["m"].struct.field("pred_y")`. These strings are generated
by `format!` over ridge values, halflife suffixes, feature-set labels,
quantile levels and target names, and **every one of them is API**. Adding
an output, renaming a suffix, or changing how a float renders silently
breaks downstream code, and the failure surfaces as a `StructFieldNotFound`
in someone else's pipeline. On 2026-08-31 exactly *one* spec shape was
pinned (`test_exact_field_names_for_a_grid_spec`, 52 fields, still in
`tests/test_portability.py`). The snapshot test below pins 28 today.

**2. Twenty-one spec constructors and ~200 named keyword parameters, plus
the helper modules**, as counted on 2026-09-06. All are keyword-only, so
positional order is not API, but every *name* is. On that date 29 of the
parameters were the `CommonKwargs` every spec shares; today it holds 33
(`python/polars_online/_kwargs.py`). The four helper modules named then were
`po.corr`, `po.sim`, `po.gram` and `po.prep`, whose functions are read the
same way. `tests/api_surface.txt` pins the constructors and their keyword
names. Of the helper modules it pins `corr`, `eval`, `gram` and `prep`
(`tests/test_api_surface.py`), so `po.sim`'s one function, `regimes`, is not
pinned.

**3. Defaults, which are API in the worst way.** They are
`solve_every = halflife/50`, `standardize` true for lasso and false for
ridge, `average_eta = 1.0` and `clip_gradient = 1e3`. Changing one of these
does not raise: it silently changes users' numbers, which is worse than
breaking them. `docs/VALIDATION.md` justifies them, and
`test_validation_doc.py` pins that they are still the measured optimum, but
nothing pins the *values* as a contract.

**4. State files and the TOML config, already the best guarded.** A state
file is versioned msgpack, with `SCHEMA_VERSION` plus a bank
`format_version`. The proposal held this up as the model the rest should
follow. It had a frozen fixture per schema (`state_v1.rs`,
`state_schema2.rs`, `state_schema3.rs`, `state_schema4.rs`,
`state_schema5.rs`), and a documented rule (hard rule 5) to keep a
previous-version loader. Each fixture asked only what a converter must
preserve: it loads, it continues the stream to the bit, and it re-saves
byte-identically while the writer is unchanged. So the last of those became
the upgrade test the moment the writer moved, which is exactly what happened
to schema 3 at task 40.

**Pre-1.0, the loader rule is suspended, and the fixtures are gone.** They
went in the naming pass of 2026-09-07 (`6124d0b`), after which no older
state could be read. Today `SCHEMA_VERSION` and `MIN_SCHEMA_VERSION` are both
13, so a state written in an older layout is refused on its version. The
exception to hard rule 5, and the reason for each raise of the minimum, are
recorded beside `MIN_SCHEMA_VERSION` in `crates/online-core/src/lib.rs`.

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
| public symbols | every public symbol in `polars_online` and `polars_online.spec` | every public symbol | |
| constructors | every spec constructor's **full signature including default values** | every constructor signature *including defaults*, the shared `**common` parameters listed once, explicitly | |
| the namespace | the expression-namespace method list | the expression namespace | the `lf.online` and `df.online` namespaces; the expression namespace went with task 85 |
| `ModelBank` | | | its name and signature |
| helper modules | | | `corr`, `eval`, `gram` and `prep`, since the 2026-09 batch |
| output fields | `output_fields()` for a canonical matrix of ~12 spec shapes: each model, plus grids, feature sets, multi-target, and every `emit_*` combination | `output_fields()` across a 14-case matrix covering every model, the full grid/emit combinations, and the float-rendering extremes | 28 cases |
| versions | `SCHEMA_VERSION` and the bank `format_version` | `schema_version` | |
| length of `tests/api_surface.txt` | | 416 lines | 904 lines |

**Building it found two things, and fixed both:**

| found | what was done |
|---|---|
| a legal `ridge = 1e-300` produced a **311-character field name**, because Rust's float `Display` never uses scientific notation | numbers outside `[1e-6, 1e7)` now render compactly (`1e-300`), chosen so no existing name changed. The rendering is centralized in one `num_label()` function, instead of seven scattered `format!` sites |
| a grammar ambiguity: a target literally named `y__r0.5` renders identically to a grid field | documented in the README, with a duplicate-name tripwire in `Bank::new`, rather than banned, since it cannot corrupt a struct |

The snapshot subsumes the single-shape field-name test, and would have
caught the E23 declared-vs-realized divergence from the other direction.

### Policy to write down

**The proposal meant this policy to go, short, into the README and
CONTRIBUTING, and it is written here instead.** The README carries only its
versioning rules, under *This package's own versioning*, and CONTRIBUTING
carries none of it.

| | what it covers |
|---|---|
| **stable** | everything in `__all__`, the `spec.*` constructors and their keyword names, the `.online` namespace, output field names, the state file format, and the TOML config keys |
| **not stable** | anything underscore-prefixed; the Rust crates, which R3 decided not to publish; and the exact numeric output, which depends on the polars version and the platform, within the 1e-12 tolerance `test_golden_pipeline.py` pins |

**Changing a default is a breaking change, because it changes results
silently.** It needs a minor bump pre-1.0, a major bump after, and a
CHANGELOG entry saying what moved and by how much.

**Pre-1.0, the minor version carries breaking changes**, so a user who
wants stability pins the current minor. This line recorded `~=0.7.0` as that
pin; the current release is the version `pyproject.toml` names.

### What not to do

- **Do not `__getattr__`-guard the submodules.** `polars_online._spec` is
  importable, and someone will reach into it; that is what the underscore
  is for. Enforcing it would take more work than the honesty is worth.
- **Do not freeze the API before 0.1.0 ships.** The snapshot test is for
  detecting change, not preventing it. Before there are any users is
  exactly when changing names is cheap. (0.1.0 shipped on 2026-09-03.)
- **Do not add `cargo semver-checks`** unless R3 says the crates are
  published. It is the right tool for a published Rust API, and pure
  overhead otherwise.

### API usability round (pre-users window)

The API was reviewed by using it as a naive user. The findings, and what
was done:

| finding | what was done |
|---|---|
| **String construction was the API's worst ergonomic.** Reaching a grid slot meant hand-building `pred_y__r0.5@h500`, that is, mentally reimplementing the float rendering | **Fixed**: `spec.output_index()` returns every field with the machine values its name encodes (kind, target, halflife/lam, ridge, feature_set, lasso λ, quantile level, ew_cov columns). It is produced by the same Rust code that renders the names, and selection becomes a Polars filter |
| **`coef` was an unmapped flat list** | **Fixed**: `spec.coef_index()` maps each position to (target, combo, term). It is derived from `output_index`, so it cannot drift, and was verified by recovering known coefficients by position |
| **Our own `eval.unpack` parsed names heuristically** ("longest match wins") | **Fixed**: an optional `spec=` argument resolves slot→target exactly, through the index. The heuristic remains only for callers with a frame but no spec |
| **`min_periods` defaults sensibly**: the first prediction comes at ~k+2 | verified; nothing to do |

**Reviewed and deliberately kept:**

| kept | rather than | why |
|---|---|---|
| `fit_predict`'s name | | sklearn readers may expect in-sample fit-then-predict. Ours is predict-then-update, which is strictly better for them. A docstring note suffices, and the fused call *is* the out-of-sample guarantee |
| the eight `emit_*` booleans | a stringly `outputs=[...]` list | they are discoverable in signatures |
| spec dicts | spec objects | they are JSON-ready and printable, and validation already happens at construction |
| wide struct output | a native long format | `eval.unpack` already provides long form on demand |

### The 2026-09 batch: what it added to the surface (tasks 45–56)

The batch added five models (`deco`, `rcov`, `hmm`, `corrchange`, `bocpd`),
and four helper modules' worth of new functions: `po.corr` with twenty,
`po.sim.regimes`, `po.prep.refresh_time` and `po.gram.from_row`. It also
added `group_close` and the closed-group queue on every spec, and lagged
co-moments on `ew_cov`. In API terms:

- **`SCHEMA_VERSION` became 5, and `state_schema5.rs` froze it**: a bank
  with an undrained closed row, an `ew_cov` with a partly filled lag ring,
  and one of each new model mid-stream. `state_schema4.rs` still loaded,
  continued to the bit and re-saved as 5, which was hard rule 5 discharged.
  Both fixtures have since gone, and the schema is 13
  ([What the API actually is](#what-the-api-actually-is)).
- **Every new output field name is pinned** by `tests/api_surface.txt`,
  which then gained a `[helper modules]` section, and by
  `tests/test_golden_pipeline.py`, which fixes the numbers of a
  twenty-five-spec bank at three rows.
- **Three of the new defaults are measured, not chosen**: `bocpd`'s
  `robust_beta = 0.1` (above ~0.2 nothing is ever detected), `bocpd`'s
  `emission = "diag"`, and `rcov`'s automatic bandwidth. The measurements
  are in `docs/PLAN.md` task 55 and `docs/ENHANCEMENTS.md` E61 for `bocpd`,
  and in E57 for `rcov`. Changing one of them is a breaking change, by
  [the policy](#policy-to-write-down).
- **Two of the three streaming paths still carry no guarantee** (CLAUDE.md
  rule 13): `ModelBank` and the IO plugin. Nothing in this batch changes
  that, and it adds no new polars API dependency beyond
  `LazyFrame.collect_batches`, which is already the floor.
- **The closed-group frame's *column* names are pinned since 2026-09-24**
  (`spec`, `group`, `session`, `rcov`, `bandwidth_used`, …). They are as
  much API as an output field is, and until then only
  `test_closed_groups.py` read them by name. `tests/api_surface.txt` now
  records them in order, as `[closed_groups columns]`: the shared columns,
  and each kind's full list, with `ew_cov`'s PCA block and `marginal`'s lag
  and bin blocks switched on.

## CI cost while the repo was private

While the repository was private, Actions minutes were metered, and not
evenly: macOS billed at 10x and Windows at 2x. These records are how the
month's minutes ran out on day one, where they went, and what the fixes
changed. Going public removed the constraint, since Actions is unmetered on
public repositories. The policy that came out of it still holds, enforced by
`tests/test_ci_cost_policy.py`
([What a push costs now](#what-a-push-costs-now)).

### Actions quota: exhausted on day one (2026-08-31)

**The push of `1a52267` did not run, and the cause was the Actions
allowance, not the workflow.** All four jobs failed in ~2s with *"The job
was not started because recent account payments have failed or your
spending limit needs to be increased."*

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
release tags". It is now written as opt-in on `schedule`/`workflow_dispatch`.

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

- **Two full compiles per `test` job**, finding 3 above: different
  profiles, and genuinely different artifacts, so only caching helps.
- **`release.yml` does not cancel superseded runs; it queues instead.** It
  is the one workflow where cancelling costs more than it saves: the run
  being cancelled may be midway through uploading wheels or publishing to
  PyPI.

**Watch out for `release.yml` (2026-08-31, while the repo was private).** It
is tag-triggered, and untouched by the policy above: three macOS jobs plus
Windows, across a six-target wheel matrix. At 10x, one release tag could
plausibly cost most of a month's allowance while the repo is private. Hence
the rule of the day: do not cut a release tag before going public. The
repository went public on 2026-09-02, and 0.1.0 was released on 2026-09-03.

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

#### What a push costs now

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

**The policy is enforced by `tests/test_ci_cost_policy.py`, not by
comments.** It parses the workflows and asserts four things:

- the matrix fallback is the cheap branch;
- lint stays on Linux and never builds the extension;
- every job has a timeout;
- macOS is reachable only by dispatch, schedule, or the repo being public.

A comment could not stop the 830-minute mistake; that test would have.

## Going public, as recorded

This section is the record of going public, from the proposal of 2026-08-31
to the settings of 2026-09-06. On 2026-09-06 the document's status read
**done, and kept as the record**: 0.1.0 had been released on 2026-09-03 and
0.1.1 on 2026-09-04, and 0.2.0 was being released. The decision to go public
was taken on 2026-08-31, and the repository went public on 2026-09-02, when
it was deleted and recreated public.

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
[Policy to write down](#policy-to-write-down), and not in the README or
CONTRIBUTING. The R4 settings needed the web UI, and on 2026-09-03 the REST
API set some of them instead. And instead of a flip of visibility, the
repository was deleted and recreated public on 2026-09-02
([Going public (2026-08-31)](#going-public-2026-08-31)).

### Before going public: R1–R6

Ordered by "would embarrass us if a stranger found it first".

**Already done when the proposal was written:** Apache-2.0 `LICENSE`,
`CONTRIBUTING.md`, `SECURITY.md`, `CHANGELOG.md`, issue and PR templates,
and an outward-facing README. So were `release.yml`, with wheels for six
platforms plus an sdist and PyPI trusted publishing, and the name
`polars-online`, verified free. The test suite is the repo's strongest
argument, with golden and hardening layers: ~640 Rust + ~2,200 pytest,
counted on 2026-09-06. The README gives about 650 Rust tests and 2,200
pytest cases today.

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

On 2026-08-31 these needed the web UI and were not scriptable from here.
On 2026-09-03 the REST API set the description and topics, and a ruleset
that stops `main` being force-pushed or deleted. Private vulnerability
reporting was left to the owner.
[Repository settings, as recorded](#repository-settings-as-recorded) has
each one's full record.

- **Branch protection on `main`: require CI to pass, no force-push.**
  Without it, a bad push cannot be caught by anything. The working rule
  since has been to fast-forward `main` only from a branch whose CI run is
  green.
- **Description and topics** (`polars`, `online-learning`, `streaming`,
  `regression`, `rust`): this is how anyone finds it.
- **Private vulnerability reporting**, which `SECURITY.md` already tells
  people to use.

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

1. Settings → Code security: Dependabot alerts and security updates, secret
   scanning and push protection, and **private vulnerability reporting**,
   which both SECURITY.md and CODE_OF_CONDUCT.md point at.
2. Description and topics; branch protection on `main`, last.
3. README badges (CI, license): they only render once public, so add them
   then.
4. **Weekly native leak check**, `docs/PLAN.md` task 18. Done 2026-09-03:
   `.github/workflows/leakcheck.yml`, Mondays and on demand, on ubuntu and
   macOS, with a control leak that must be caught. The "clean today"
   baseline this item rested on turned out to be a blind check, and task 18
   has the numbers.

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
[Rehearse before tagging](#rehearse-before-tagging).

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
| `v*` tags | 2026-09-03: a ruleset, so `v*` tags cannot be moved or deleted once they exist, with no bypass list; the owner is bound too, and disabling the ruleset is the only way round it. 2026-09-06: [the release gate](#the-release-gate-2026-09-06) records the `release tags are immutable` ruleset over `refs/tags/v*`, with `deletion` and `update` rules and an empty bypass list |
| description, homepage and topics | 2026-08-31 (R4): description and topics. 2026-09-02: did not come back. 2026-09-03: the description, the homepage (the Pages site) and the topics set |
| Dependabot alerts and security updates | 2026-08-31: remaining, under Settings → Code security. 2026-09-02: did not come back. 2026-09-03: re-enabled |
| secret scanning and push protection | 2026-08-31: remaining, under Settings → Code security |
| private vulnerability reporting | 2026-08-31 (R4 and the sweep): enable it; it is the enforcement channel `SECURITY.md` and `CODE_OF_CONDUCT.md` both point at. 2026-09-02: did not come back. 2026-09-03: left to the owner |
| the `pypi` environment | 2026-09-02: must exist again before any tag is pushed. 2026-09-03: left to the owner, with the owner as required reviewer and a `v*` tag rule. 2026-09-06: the `Pypi` environment, with the owner as a required reviewer and self-review allowed |
| the PyPI pending publisher | 2026-09-03: left to the owner, as `polars-online` / `hgilde` / `polars-online` / `release.yml` / `pypi` |
| README badges (CI, license) | 2026-08-31: they only render once public, so add them then |
| the `github-pages` environment | 2026-09-06: briefly carried a required reviewer, and no longer does |
