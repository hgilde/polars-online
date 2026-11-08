<!-- kind: record -->
# The release history

*Status, 2026-11-08: a record. How the release workflow, the Polars range, the public API and the public repository came to be as they are, moved here whole from `docs/RELEASE-READINESS.md`: the workflow's faults and fixes, the suite on the Polars floor, the reasoning behind the two copies of Polars, Polars 2.0 measured, the removed expression plugin, the API's usability rounds, what CI cost while the repository was private, and going public. Each part reads as of its date. `docs/RELEASE-READINESS.md` holds what a release is held to now, `docs/VERSIONS.md` the Polars range and the API promise, and `docs/contributing/` the steps.*

### How the release workflow came to this

**The workflow's shape came from three incidents, each recorded as of its
date.**

#### Why the tag comes last (2026-09-27)

Until then a release started when a `v*` tag was pushed. 0.11.0 was tagged
on a branch whose 21 commits had never run on Linux or Windows. CI failed
there on the tag, the publish jobs were skipped, and the tag could not be
moved ([The release gate](../RELEASE-READINESS.md#the-release-gate)). So the release now runs
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
glibc 2.17** (the user's decision, 2026-09-28). `release.yml`'s build job runs `cargo build` in
`quay.io/pypa/manylinux2014_<arch>`, the image for the runner's own
architecture. Rust is installed inside it as maturin-action installs it
there. It is a `docker run` step, since the image's glibc is too old for
the actions' JavaScript to run as a job container. `scripts/glibc_floor.py`
then reads the binary's version requirements and fails the job above 2.17,
before anything is uploaded. On 0.11.1's x86_64 binary it reports 2.39 and
refuses it. The 0.12.0 rehearsal was the first run to build it (run
36448809775 on `36e468c`). The binaries need glibc 2.16 on x86_64 and 2.17
on aarch64, and 0.12.0 shipped them.

#### The suite on the floor, as recorded

**How the floor was found (2026-09-02).** The declaration that day read
`polars>=1.34.0,<2`. The matrix measured `ModelBank` and the expression
plugin, and those do work from 1.28.1. But `po.run` over a path or a plan
read with `LazyFrame.collect_batches`, and so did the streaming query form
`lf.online.fit_predict`. py-polars added `collect_batches` in **1.34.0**.
On 1.28.1–1.33 both failed with
`AttributeError: 'LazyFrame' object has no attribute 'collect_batches'`, in
24 of 27 runner tests. The canary, the weekly job that runs the suite on the
newest Polars, could not see this, since it tests nothing older. It was
found while checking `lf.online.fit_predict` against the floor, and the
floor now says what the package needs. The whole suite (1037 tests) passed
on 1.34.0, 1.38.1 and 1.44.1 with identical numbers.

**The whole suite on 1.34.0 again, 2026-09-27: one bug in the package,
fixed.** A venv with the locked dev and docs groups, `polars` and
`polars-runtime-32` at 1.34.0, and this build's wheel ran 3,425 tests:
3,408 passed and 17 failed.

| failures | cause | now |
|---|---|---|
| 4 | `po.eval.seqtest` with `by`, the README's example among them: a window inside a window, which 1.34.0 refuses ("window expression not allowed in aggregation") | fixed: two passes, the same arithmetic; all 208 of those files' tests pass on 1.34.0 |
| 1 | `scripts/compare_release.py` passed `explode(empty_as_null=True)`, a keyword 1.34.0 has not got | fixed: the keyword where it exists |
| 8 | `pl.scan_arrow_c_stream`, in py-polars from 1.43.0: the DuckDB, ADBC and pyarrow paths' tests and examples | the README says 1.43.0; the package itself only names it in a warning |
| 2 | durations: 1.34.0's parser refuses a leading `+`, which the text oracle compares against, and its `pl.duration` wraps past an i64 before the package can see it | recorded: the promise that a `pl.duration` past 292 years is refused holds on a newer polars |
| 2 | the development environment's version pin (`tests/test_scaffold.py`) and `docs/VALIDATION.md`, which records 1.44.2 | expected |

`po.run` has gone since, and the floor has not moved. It left Python on
2026-09-17 (`7d23a80`, released in 0.7.0), and the command line kept the
runner.

**The whole suite on 1.34.0 again, 2026-10-07: nothing in the package
failed.** A venv made the same way, with this build's extension, at
`4d6ae0f`, with the changes that added the floor leg, ran 4,711 tests, and
24 failed on Polars' version. Each was then run on sixteen releases between
1.34.0 and 1.44.2, every minor among them, to find where it first passes.
1.35.0 and 1.36.0 cannot be installed, since their runtime packages are not
on PyPI. Every failure is a test's reference, or an example, that needs a
newer Polars:

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
expression plugin, removed on 2026-09-17.

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
| 1 | **the plugin entry points** | py-polars `dlopen`s our `.so` and `dlsym`s `_polars_plugin_online_run`, `_polars_plugin_field_online_run`, `_polars_plugin_get_version` and `_polars_plugin_get_last_error_message`. That is py-polars calling into our Rust at runtime | gone with the expression plugin, on 2026-09-17 |
| 2 | **the `release` callbacks** inside every `SeriesExport` | a function pointer into the *producing* binary, invoked by the consumer. This is what makes two copies of Polars safe rather than a double-free | in use |
| 3 | **the allocator capsule** | `PyCapsule_Import("polars.polars._allocator")` hands us a struct of four function pointers into py-polars' binary, which then serve every allocation this extension makes | in use, and the one fragile part (below) |

So the dynamic interop is real, but at the C level, where there is an ABI
to rely on, instead of at the Rust level, where there is not.

**The plugin's binding was a designed, versioned contract.** It is worth
spelling out, because "they dlsym a C function" undersells it. The
expression plugin that used it was removed on 2026-09-17, so this is the
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

#### R6 narrows, as `docs/records/STATE-WORKFLOW-RESEARCH.md` said it might

**`docs/records/STATE-WORKFLOW-RESEARCH.md` §4 states its rule R6, the one gap in the
state workflow.** Polars drains a Python source before surfacing a
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
asserted the 1.x behavior at 40,000 rows, and was the first of the two
failures. The test was right about 1.x, and needed a version-aware
assertion to hold on both.

**That is a narrowing of a documented hazard, not a regression.** The
dangerous case is the state being written when the output was not, and 2.0
does that less often. The `polars-online` CLI remains the transactional call
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

The `polars-online` CLI is unaffected either way: it links the Rust `polars`
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
license to cache or skip a re-execution, and the rule R2 of
`docs/records/STATE-WORKFLOW-RESEARCH.md` measured how often the source runs. **Decided the
same day** (`4c545af`): `_frame.py` declares the source pure
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
[Raising the ceiling to a new major](../contributing/move-the-polars-range.md#move-the-ceiling-up-to-a-new-major),
above.

#### Polars 2.0.0 final, and the pin moved (2026-10-08)

**The canary passed on 2.0.0 final, and the development environment moved
to it the same day.** The canary run of 2026-10-08
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
Rust tests cannot see a py-polars release, and `polars>=1.34.0,<3`, the
range then, admitted 2.0.0 as it stood; the floor leg ran on 1.34.0.

### The expression plugin, as recorded

**The expression plugin, `pl.col(..).online.<model>(..)`, was removed on
2026-09-17, in 0.7.0, because Polars hands a stateful user expression its
whole column.** It was the supported mechanism, with a MAJOR/MINOR
handshake the loader checks before its first call. So a break showed as a
refusal to load, rather than misbehaviour.

**The one path that carried a guarantee was also the one that could not
stream.** The measurements above stand as the record of what it did, and
nothing in the current guidance depends on it. The **Arrow PyCapsule
interface** now takes its place in the guarantee story. It covers the
output side only, so a call still crosses the `PyDataFrame` boundary on the
way in.

**2026-09-03: the expression plugin warned on every use.** It shipped as before, and everything above still held for it.
What changed was that `pl.col(..).online.<model>(..)` issued
`InMemoryExpressionWarning` naming `lf.online.fit_predict`, because it was
the one O(data) form. Nothing here moved: the floor and ceiling were set by
`collect_batches`, not by the plugin, and the canary exercised all three
paths. For a few hours the same change had the plugin out of the wheel,
behind a cargo feature. That was reverted before release, so no wheel
published at the time lacked it.

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
| the `emit_*` booleans, seven then and seventeen today (`tests/api_surface.txt`) | a stringly `outputs=[...]` list | they are discoverable in signatures |
| spec dicts | spec objects | they are JSON-ready and printable, and validation already happens at construction |
| wide struct output | a native long format | `eval.unpack` already provides long form on demand |

**A new optional output is switched on in one of the three shapes already
in use, never a fourth.** Each shape says what
the setting is:

| the switch | when | as in |
|---|---|---|
| a boolean `emit_*` keyword | the output is on or off, and nothing else | `emit_sigma=True` |
| a list | the list is also the setting: which of several outputs, or at which values | `ew_cov(stats=["mean", "corr"])`, `resid_quantiles=[0.5, 0.9]` |
| a level | the number is also the setting | `conformal=0.9`, the coverage asked for |

### What the correlation and regime models added to the surface (2026-09)

A batch of work in 2026-09 added five models (`deco`, `rcov`, `hmm`, `corrchange`, `bocpd`),
and four helper modules' worth of new functions: `po.corr` with twenty,
`po.sim.regimes`, `po.prep.refresh_time` and `po.gram.from_row`. It also
added `group_close` and the closed-group queue on every spec, and lagged
co-moments on `ew_cov`. In API terms:

| what the batch added | how the API held it |
|---|---|
| **`SCHEMA_VERSION` 5** | `state_schema5.rs` froze it: a bank with an undrained closed row, an `ew_cov` with a partly filled lag ring, and one of each new model mid-stream. `state_schema4.rs` still loaded, continued to the bit and re-saved as 5, which was hard rule 5 discharged. Both fixtures have since gone, and the schema has moved on many times since: `po.schema_version()` gives today's ([What the API actually is](../VERSIONS.md#what-the-api-is)) |
| **every new output field name** | pinned by `tests/api_surface.txt`, which then gained a `[helper modules]` section, and by `tests/test_golden_pipeline.py`, which fixes the numbers of a twenty-five-spec bank at three rows |
| **three new defaults, each with its reason: one measured, one a cost, one a published rule** | `bocpd`'s `robust_beta = 0.1` (measured: above ~0.2 nothing is ever detected), `bocpd`'s `emission = "diag"`, and `rcov`'s automatic bandwidth. `bocpd`'s robust weighting passes the test of a 20-σ row, which restarts the plain run and does not move the robust one, and it trades detection for robustness above about 0.2. Its diagonal emission takes `O(runs·d)` work a row, against the full pair's `O(runs·d³)`, which factors each run's scale every row. `rcov`'s bandwidth follows the published rule of Barndorff-Nielsen, Hansen, Lunde and Shephard (2011), `H = ⌈c* ξ̂^{4/5} n^{3/5}⌉`, from a dense and a sparse realized variance. Changing one of them is a breaking change, by [the policy](../VERSIONS.md#what-each-kind-of-release-may-change) |
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
a public repository ([the policy](../TESTING.md#the-policy-as-the-test-holds-it)).

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
| `release.yml` queues superseded runs rather than canceling them | it is the one workflow where canceling costs more than it saves: the run being cancelled may be midway through uploading wheels or publishing to PyPI |

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

Its R1–R6 are this document's own items. `docs/records/STATE-WORKFLOW-RESEARCH.md` numbers
its rules R1–R7, and each of those is named with its document where this
one cites it.

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
[The policy](../VERSIONS.md#what-each-kind-of-release-may-change), and not in the README or CONTRIBUTING. The R4
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
no `LICENSE` at all.** Apache-2.0 requires the license to accompany a
distribution, and packagers check for it. The fix is PEP 639 `license-files`
under `[project]`; note that `[tool.maturin] license-files` is silently
ignored. The license now lands in the sdist, and in the wheel's
`.dist-info/licenses/LICENSE`.

#### R4 — Repository settings

**R4 asked for three settings: branch protection on `main`, the
description and topics, and private vulnerability reporting.** Branch
protection was to require CI to pass and forbid a force-push, since without
it nothing catches a bad push. The topics, `polars`, `online-learning`,
`streaming`, `regression` and `rust`, are how anyone finds the repository.
`SECURITY.md` already told people to use private reporting. The working
rule that followed was to fast-forward `main` only from a branch whose CI
run was green; a release, by [its steps](../contributing/cut-a-release.md#steps), is
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
| 4 | **a weekly native leak check** | done 2026-09-03: `.github/workflows/leakcheck.yml`, Mondays and on demand, on ubuntu and macOS, with a control leak that must be caught. The "clean today" baseline this item rested on turned out to be a blind check: `leaks` does not walk the arenas pymalloc hands Python objects out of, and reported 0 leaks against a deliberate 128 MB leak. With `PYTHONMALLOC=malloc` and a differential count, the real workload grew 63 blocks (3.7 KB), and the control 1,968 blocks (142 KB) |

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
| `v*` tags | 2026-09-03: a ruleset, so `v*` tags cannot be moved or deleted once they exist, with no bypass list; the owner is bound too, and disabling the ruleset is the only way round it. 2026-09-06: [the release gate](../RELEASE-READINESS.md#the-release-gate) records the `release tags are immutable` ruleset over `refs/tags/v*`, with `deletion` and `update` rules and an empty bypass list |
| description, homepage and topics | 2026-08-31 (R4): description and topics. 2026-09-02: did not come back. 2026-09-03: the description, the homepage (the Pages site) and the topics set |
| Dependabot alerts and security updates | 2026-08-31: remaining, under Settings → Code security. 2026-09-02: did not come back. 2026-09-03: re-enabled. 2026-10-06: security updates on; they opened #4 (urllib3, `uv.lock`) and #5 (rustls, `Cargo.lock`), both merged that day |
| secret scanning and push protection | 2026-08-31: remaining, under Settings → Code security. 2026-10-06: both on, as `security_and_analysis` in `gh api repos/hgilde/polars-online` reports |
| private vulnerability reporting | 2026-08-31 (R4 and the sweep): enable it; it is the enforcement channel `SECURITY.md` and `CODE_OF_CONDUCT.md` both point at. 2026-09-02: did not come back. 2026-09-03: left to the owner. 2026-10-06: on (`gh api repos/hgilde/polars-online/private-vulnerability-reporting`) |
| CodeQL code scanning | 2026-10-06: GitHub's default setup, not a workflow in `.github/workflows/`: actions, Python and Rust, weekly, configured on 2026-09-07 (`gh api repos/hgilde/polars-online/code-scanning/default-setup`) |
| the `pypi` environment | 2026-09-02: must exist again before any tag is pushed. 2026-09-03: left to the owner, with the owner as required reviewer and a `v*` tag rule. 2026-09-06: the `Pypi` environment, with the owner as a required reviewer and self-review allowed |
| the PyPI pending publisher | 2026-09-03: left to the owner, as `polars-online` / `hgilde` / `polars-online` / `release.yml` / `pypi` |
| README badges (CI, license) | 2026-08-31: they only render once public, so add them then |
| the `github-pages` environment | 2026-09-06: briefly carried a required reviewer, and no longer does |
