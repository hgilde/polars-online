# Test coverage and testing improvements

Status as of 2026-09-06: **about 650 Rust tests and 1,250 pytest functions**
(some 2,200 cases, plus 2 opt-in soak tests), all green, run in CI on three
OSes on every push. Counted again on 2026-09-24: **831 Rust tests and
1,564 pytest functions** (2,838 cases), plus the same 2 opt-in soak tests.
`cargo test --workspace -- --list` and `pytest --collect-only` did the
counting. The coverage figures below are from the 2026-08-30 run.

This document assesses what the tests actually prove, then lists concrete
improvements. Its emphasis is on edge cases, and on comparing behavior against
reference implementations, including [river](https://riverml.xyz). It is for
a reader who has seen the README's summary of the suite and wants the
evidence: what each part proves, what it has found, and where it is thin.

Each improvement is an entry with an ID, such as T-E9, and the code and the
tests cite those IDs. They also cite the lettered sections that hold the
entries, such as section C. Rows that say **blocked** or **never executed**
were written before the repo had CI and are kept as the record.
[What is left](#what-is-left) says what cleared them.

| section | what it holds |
|---|---|
| [What the suite proves](#what-the-suite-proves) | [the eight test classes](#the-eight-test-classes) · [against reference implementations](#against-reference-implementations) · [beyond the eight classes](#beyond-the-eight-classes) |
| [What it has found](#what-it-has-found) | [defects, and where each is told](#defects-and-where-each-is-told) · [differences from river that are not bugs](#differences-from-river-that-are-not-bugs) |
| [Where it is thin, and what is left](#where-it-is-thin-and-what-is-left) | [measured coverage](#measured-coverage) · [open entries](#open-entries) · [mutation survivors](#mutation-survivors) · [what is left](#what-is-left), which is current |
| [How the suite looks for defects](#how-the-suite-looks-for-defects) | [an oracle, not a golden number](#an-oracle-not-a-golden-number) · [what the mutation run actually found](#what-the-mutation-run-actually-found) · [FFI memory and crash safety](#ffi-memory-and-crash-safety-2026-08-31), the crash-safety audit |
| [The entries, by ID](#the-entries-by-id) | [A. our own oracles](#a-close-the-oracle-gaps-our-own-references) · [B. river](#b-cross-checks-against-river) · [C. edge cases](#c-edge-case-matrix) · [D. Windows](#d-windows-and-cross-platform) · [E. infrastructure](#e-infrastructure) |

The words this ledger uses:

| word | what it means here |
|---|---|
| oracle | a reference a model cannot share a bug with; its four forms are in [An oracle, not a golden number](#an-oracle-not-a-golden-number) |
| golden number | an exact expected output from a fixed stream, embedded in a test. It pins the arithmetic against any change, but it is not an oracle |
| mutant | one small change `cargo mutants` makes to the source, such as a flipped operator or a function body replaced with a constant, before it reruns the tests. A mutant is *caught* when some test fails, and *missed*, a survivor, when every test still passes (`scripts/mutants.sh`) |
| equivalent mutant | a mutant that no test can kill |
| IC | the correlation of prediction with target, `po.eval`'s `ic` |
| P1, P2, P3 | in an entry's P column, its priority: **P1** closes a PLAN promise or covers a found defect, **P2** is meaningful new assurance, **P3** is infrastructure |
| P1–P11 elsewhere | the performance items of `docs/PERFORMANCE.md` §3 |
| E numbers; C, T, U and X numbers | entries in `docs/ENHANCEMENTS.md`; entries in `docs/IMPROVEMENTS.md` |
| hard rule N | the numbered hard rules in `CLAUDE.md` |
| the ten regression models | the models the per-model sweeps run: `ewridge`, `rls`, `lasso`, `kalman`, `huber`, `quantile`, `sgd`, `pa`, `ftrl` and `holt`, which `REGRESSIONS` in `tests/test_model_registry.py` lists |

## What the suite proves

The scorecard is against the eight test classes of `docs/PLAN.md` §9. The
parts after it test what those classes do not name.

### The eight test classes

| class | status | what holds it |
|---|---|---|
| 1. Oracle agreement | **Mostly done.** | each model against a reference it cannot share a bug with, in the [table below](#against-reference-implementations). Open: a numpy `lasso_ref` for the lasso's *pred* path (T-A2) |
| 2. Chunk invariance | Done | bitwise at the bank (1/7/100 chunks) and CLI (`chunk_rows` sweep) levels; save/load mid-stream identical. The `coef` field is correctly excluded: it is chunk-dependent by design |
| 3. Out-of-sample by construction | Done | IC ≈ 0 on pure-noise targets asserted for ewridge, kalman, huber, ftrl; lasso selection prefers the all-zero penalty on noise; robust reweighting proven to use the *prior* residual |
| 4. Clock semantics | Done | cap, negative-delta (`max`/`zero`/`reset_state`), session gap and reset, first row, row-count clock, skipped-row decay folding, per-group independence |
| 5. Null policy & warmup | Done | feature/target/weight nulls and `min_periods`, for all ten regression models (T-A5) |
| 6. Arrow ≡ Polars output | **Retired, and replaced.** | the Arrow path, below |
| 6b. `predict` ≡ `fit_predict` of the next row | Done (E31) | `tests/test_predict.py`; `crates/online-core/tests/model_contract.rs` |
| 6c. Runner ≡ bank, every source and format | Done (E32) | `crates/online-polars/tests/runner.rs`; `run_online` in `tests/conftest.py`; `tests/test_bank_ergonomics.py` |
| 7. Cross-platform state | Done | the macOS→Windows/Linux artifact hand-off in `release.yml`, run for every release since 0.1.0, and `ci.yml` loading states on all three OSes on every push. It was defined but never executed before the repo had a remote |
| 8. Benchmark | Done | `scripts/benchmark.py`, numbers in README |

**6. Arrow ≡ Polars output: retired, and replaced.** This row held the
expression form to the bank. Task 85 removed that form (2026-09-17), and
`tests/test_expr.py` with it, so the equivalence it named no longer has two
sides. The Arrow path took its place. `tests/test_arrow_capsule.py` holds
`fit_predict_arrow`'s struct to `fit_predict`'s struct column, field for
field and null for null. The nested `coef` list is included, and nulls are
compared rather than skipped. `crates/online-polars/tests/arrow_chunk.rs`
feeds a chunk built by hand from Arrow arrays, with no frame anywhere, and
holds it to the same output.

**6b. `predict` ≡ `fit_predict` of the next row (E31).**
`tests/test_predict.py` holds row `i` of `predict(df)` to row 0 of
`fit_predict(df.slice(i, 1))` on a fresh clone, field for field. It does so
for all ten regression models with every diagnostic on, and across every
session and clock policy. `crates/online-core/tests/model_contract.rs` holds
each model's `predict` to its `step`, row by row.

**6c. Runner ≡ bank, every source and format (E32).**
`crates/online-polars/tests/runner.rs` holds every input format (parquet,
ipc, csv, ndjson), every output format, `Input::Batches` ≡ a plan, and the
failure paths to the numbers `ModelBank` gives on the same rows. CSV's
flattened columns and JSON-text `coef` decode bit-exact. The `online` command
line is held to the same numbers through `run_online` in `tests/conftest.py`
(`test_closed_groups`, `test_hardening`, `test_predict`, `test_no_output`,
`test_label_delay`), and every `sh` block of `docs/RUNNER.md` runs. The
Python-side sources of the old runner, a path, a `LazyFrame`, a `DataFrame`
and an iterator, went with it in task 83. What a `ModelBank` takes now is held
in `tests/test_bank_ergonomics.py`: `fit`, `fit_predict_batches`, over a
plan, a frame or an iterator, and `chunk_rows`. A format is bound in the
tests, never in the API.

### Against reference implementations

**Do we compare edge cases against reference implementations?** Yes, for most
of the surface. Clock semantics, null policy and warmup are cross-checked
against the numpy oracles along the `ewridge`, `rls` and `kalman` paths. The
lasso is checked against its own optimality conditions. `tests/test_river.py`
compares FTRL, EW moments, quantile and Huber behavior against river. Huber,
quantile and FTRL also have numpy references at the *numeric* level (T-A3,
T-A4).

| model | held against | agreement | entry |
|---|---|---|---|
| `ewridge`, `rls` | `tests/reference.py`, incl. multi-target, standardize, `lam` decay, row-count clock | 1e-9 | |
| `rls` | `rls ≡ ewridge(ridge_decay, solve_every=1)` | <1e-9 | |
| Kalman | `kalman_ref`, across every configuration | ~1e-15 | T-A1 |
| the lasso | its KKT conditions, rather than a ported solver | the conditions hold | T-A2 |
| Huber, quantile | `robust_ref` | ~1e-13 | T-A3 |
| FTRL | `ftrl_ref` | ~1e-16 | T-A4 |
| FTRL | `river.optim.FTRLProximal`, row for row | 1e-12 | T-R1 |
| Kalman(q=0, fixed `obs_var`, `standardize=False`) | `river.linear_model.BayesianLinearRegression` | 3.6e-15 | T-R2 |
| `EwCov` | `river.stats.Mean` / `Var` / `Cov` / `PearsonCorr` | 1e-9 | T-R3 |
| EW mean/var | `river.stats.EWMean` / `EWVar` | in the limit | T-R4 |
| quantile | `river.stats.Quantile` | statistical | T-R5 |
| Huber | `river.linear_model.LinearRegression(loss=optim.losses.Huber)` | statistical | T-R6 |

### Beyond the eight classes

| part | test | what it checks |
|---|---|---|
| [production hardening round 3](#production-hardening-round-3-vs-rivers-own-battery) | `tests/test_production_hardening.py` | river's battery of checks, sklearn's estimator invariances and statsmodels' oracle convention |
| [post-refactor hardening](#post-refactor-hardening-p1p8-review) | `tests/test_hardening.py` | parameter ranges at their edges, and row counts large enough to expose stride bugs |
| [the validated defaults](#the-validated-defaults-are-still-the-measured-ones) | `tests/test_validation_doc.py` | `docs/VALIDATION.md`, regenerated and compared |
| [the declared schema](#the-declared-schema-is-checked-for-every-model) | `test_names_match_the_realized_struct_for_every_model` | the declared field names against the struct the bank produces |
| [hard rule 1](#hard-rule-1-is-enforced-not-remembered) | `tests/test_repo_hygiene.py` | no data file, large file or generated output is tracked |
| [the examples](#examples-are-executed) | `tests/test_examples.py` | everything under `examples/` runs unmodified |
| [memory and crash safety](#ffi-memory-and-crash-safety-2026-08-31) | `tests/test_ffi_memory.py`, `scripts/leakcheck.sh` | no leak and no crash across the FFI |

#### Production hardening round 3 (vs river's own battery)

**Done: it found three defects and one doc-API mismatch.** The round was
calibrated against river 0.26.1's `river.checks` (37 checks), sklearn's
estimator invariances and statsmodels' oracle convention. It lives in
`tests/test_production_hardening.py`: 38 tests when written, and 168 collected
cases on 2026-09-24.

| finding | what it was | what holds now |
|---|---|---|
| **(1) a target listed as its own feature was accepted** | corr(pred, y) = 1.000000: perfect leakage, through the door hard rule 2 does not guard | a validation error for every model, whose message points at a lagged copy |
| **(2) duplicate feature/target names were accepted** | a coefficient silently split across identical slots, on an exactly singular system | rejected |
| **(3) the finite-or-null contract filtered only NaN** | an exactly ±inf prediction from a diverged model would have reached the user | the boundary checks `is_finite` |
| **(4)** `clip_gradient=inf` is the documented way to disable clipping | the JSON layer refused it, while `halflife=inf` worked | `clip_gradient` uses the same `Num` type |

The round also implemented the river checks that apply and that the suite
lacked:

| check | what it asserts |
|---|---|
| **feature-order invariance** | spec order and frame order, plus extra-columns tolerance |
| **pickling** | `__reduce__` via `save_bytes`, so pickle and `copy.deepcopy` resume bit-exactly. It needed `#[pyclass(module=...)]`, since pickle cannot name a class that claims to live in `builtins` |
| all-null columns | |
| first-row outliers | with the washout horizon stated: ~80 halflives, inherent to EW accumulators |
| a 15-case extreme-parameter sweep | p0 at 1e±12, τ at 0.01/0.99, and deliberate SGD divergence, pinning finite-or-null |
| state-byte corruption | every magic-string byte must detect; any flip anywhere must fail cleanly, never panic |
| concurrent `fit_predict` from two threads | a safe error, and the object usable after |
| unicode/space column names | through save/load |
| README python blocks | compiling. Since IMPROVEMENTS T5 they are *run*, each in its own copy of a namespace holding what the prose has introduced by that point. That found the README's `holt` example refused by the library's own validation |

River checks *not* adopted, with reasons:

| check | why not |
|---|---|
| clone/repr/params | our specs are plain dicts |
| emerging/disappearing features | a fixed schema is a design decision, pinned as a clear error instead |

#### Post-refactor hardening (P1–P8 review)

**Done, with zero new defects: every path held, including the never-executed
one.** This was a second full review after the performance rewrite. It aimed
at the two blind spots the request named. One was parameter ranges tested
only in their comfortable middles. The other was row counts too small to
expose stride bugs in the new flat slot-major buffers. It lives in
`tests/test_hardening.py`: 18 tests (~4.5 s) when written, and 20 collected
cases on 2026-09-24.

| attack | what must hold |
|---|---|
| a kitchen-sink stream at **30k rows with every output enabled at once**: 156+ fields from two instances × ridge grid × feature sets × two targets, sigma/z/drift/metrics/autocorr/quantiles/selected/averaged, groups, sessions, weights, nulls, clock gaps | the same digest across chunkings incl. row-at-a-time, across a mid-stream save/load, and across `POLARS_ONLINE_MAX_THREADS` 1 vs 8 |
| the **coupled drift path** (grid + `drift_action="reset"`), which P2 added and nothing executed | a break in either instance resets both; the row-major path is chunk-invariant; it equals the parallel path when nothing fires |
| parameter edges: halflife 1e-3 and inf, k=64, quantile levels 0.001/0.999 | halflife 1e-3 and inf give their exact limits |
| **weight-scale invariance at 1e±6** | all weights ×c changes nothing but `n_eff`: the test that the mean form is real |
| twelve targets with per-target warmup | each lands on the exact ceil(threshold) row |
| `coef_every` 0/1/997 | across chunk boundaries |
| P6's reader thread, on a corrupt file and on a mid-stream bank error | a clean exception, with no deadlock on either side of the channel |
| the P5 spec cache | two specs on one thread stay distinct |

The soak (10M rows) was re-run after the refactor, and passes.

#### The validated defaults are still the measured ones

**Done.** `tests/test_validation_doc.py` regenerates `docs/VALIDATION.md`
and compares it to the committed copy, with timings normalized away. So the
numbers the shipped defaults were chosen from cannot silently stop being
true. It was re-run after the whole enhancement backlog, including E11b's
centered moments, and **every number is unchanged**. That is itself the
result: the numerics work did not move the measured optima. A second
assertion checks the script still runs all five experiments, so the
comparison cannot pass on a document that no longer justifies anything.

#### The declared schema is checked for every model

**Done.** `test_names_match_the_realized_struct_for_every_model` extends the
E23 guard from `ewridge` alone to all ten regression models, times four output
combinations (three when written), plus `ew_cov` separately. The optional outputs are
assembled in the stream layer and so generalize. But each model contributes
its own prediction and coefficient slots, and `sgd`, `pa`, `holt` and
`ew_cov` all postdate the original test.

#### Hard rule 1 is enforced, not remembered

**Done.** `tests/test_repo_hygiene.py` fails if a data file, a large file, or
generated tool output is tracked. It also fails if `.cache/`, `target/` or
`mutants.out/` stop being gitignored, or if a file the build needs is missing
from what `git archive` (a fresh clone, an sdist) would produce. It was
written after 136 files of `cargo mutants` output sat tracked for several
commits, swept in by a `git add -A`, with nothing complaining. It was
verified to fire, not just to pass.

#### Examples are executed

**Done.** `tests/test_examples.py` runs everything under `examples/`
unmodified. The Pathway operator example runs end to end, on its plain-batch
path, since Pathway is BSL and not a dependency. That is asserted by checking
it appears in no dependency group. `examples/bank.toml` goes through the real
CLI for `--dry-run`, a full run, and `--resume` from the state the run wrote.
A documented example that no longer works is worse than none, and until this
test nothing ran either file.

**It found a documentation defect.** The README's chunk-invariance guarantee
said "bit-identical output" with no exception. But `coef` is snapshotted on
each chunk's last row as well as every `coef_every` rows, so smaller chunks
report it more often. The guarantee now names that exception, and every
computed field is still bit-identical.

## What it has found

### Defects, and where each is told

| finding | found by | told in |
|---|---|---|
| six errors in the 2026-09 batch: `deco`, `hmm`, `corrchange` twice, `bocpd` and `rcov` | oracles written from the paper | [An oracle, not a golden number](#an-oracle-not-a-golden-number) |
| a zero-weight row at the head of a stream permanently disabled `ewridge` and `lasso` | the mutation run | [What the mutation run actually found](#what-the-mutation-run-actually-found) |
| `sgd` and `pa` reported `n_eff` with the current row's decay already applied | the mutation run, and T-A5's sweep | the same, and T-A5 in [A](#a-close-the-oracle-gaps-our-own-references) |
| `EwCovModel::n_targets` returned 1; `blend_toward_long_run` lost its doc comment; a `coef_prior` misconfiguration gave a garbled message | the mutation run | [What the mutation run actually found](#what-the-mutation-run-actually-found) |
| the robust models reported `n_eff` as the sum of IRLS weights | T-A5 | [A](#a-close-the-oracle-gaps-our-own-references) |
| a finite negative weight turned every later prediction null | writing this document (T-E1) | [C](#c-edge-case-matrix) |
| a null group key and a group named `"<null>"` shared one stream | writing this document (T-E2) | [C](#c-edge-case-matrix) |
| a unit-variance feature on a 1e6 offset was silently dropped | T-E9 | [C](#c-edge-case-matrix) |
| `halflife=600` on a microsecond clock silently meant 600 µs | T-E10 | [C](#c-edge-case-matrix) |
| a target listed as its own feature was accepted, and so were duplicate names; a ±inf prediction could pass; `clip_gradient=inf` was refused | production hardening round 3 | [Production hardening round 3](#production-hardening-round-3-vs-rivers-own-battery) |
| the README's `holt` example was refused by the library's own validation | running the README's python blocks (IMPROVEMENTS T5) | the same |
| the README's chunk-invariance guarantee named no exception for `coef` | running the examples | [Examples are executed](#examples-are-executed) |
| three `online-cli` tests wrote Windows paths into a TOML basic string; the parse error said nothing about paths | the first Windows CI run | T-W3b in [D](#d-windows-and-cross-platform) |
| nine pytest failures on Windows, all test bugs and no library bug | the first Windows CI runs | T-W1 in [D](#d-windows-and-cross-platform) |
| a `UnicodeEncodeError` in `examples/pathway_integration.py` | CI on Windows | [What is left](#what-is-left) |
| two Windows gaps in `.vscode/settings.json` | writing `scripts/env.ps1` | T-W9 in [D](#d-windows-and-cross-platform) |
| a String feature column was silently parsed back to f64 | the FFI audit | [FFI memory and crash safety](#ffi-memory-and-crash-safety-2026-08-31) |

### Differences from river that are not bugs

The river work found two convention differences worth knowing about. Neither
is a bug in either library, but code that assumes they agree during warmup is
wrong.

| difference | ours | river's | entry |
|---|---|---|---|
| when FTRL predicts | recomputed from `z` at prediction time, per McMahan Algorithm 1 | `LogisticRegression` predicts with the previous step's proximal weights, one proximal step behind the paper | T-R1 |
| the EW mean | *bias-corrected*: divides by the accumulated weight, so exact from the first row | `EWMean` is un-normalized, seeded at its first value, and stays anchored near that seed during warmup | T-R4 |

## Where it is thin, and what is left

### Measured coverage

Measured with `./scripts/coverage.sh` on 2026-08-30: **96% of the Python
package**, and **75% region / 73% line** of the Rust workspace. It is
reported, never gating (T-D4).

**The Rust figure understates reality.** `cargo llvm-cov` only sees what
`cargo test` runs. `online-py` (0%) and much of `online-polars` are exercised
by the pytest suite through the compiled extension, which is invisible to the
Rust instrumentation. The genuinely thin spots it does reveal are two.
`online-cli/src/main.rs` is argument plumbing, covered instead by the CLI
integration tests through `run_config`. `online-core/src/robust.rs` is at
75%.

### Open entries

What the entries themselves still call open:

| entry | still open | why it matters |
|---|---|---|
| T-A2 | a numpy CD `lasso_ref` for the lasso's *pred* path | it would catch schedule and warm-start bugs the KKT check cannot see |
| T-W3 | path resolution on a real Windows runner | escaped Windows-style paths and paths with spaces are tested through the CLI on any OS; the entry is marked only partly done |
| T-D4 | making the mutation pass periodic in CI | no workflow under `.github/workflows/` runs `scripts/mutants.sh` |

### Mutation survivors

Run 3 of the mutation pass left **217 missed** in `online-core`: 8.3%
surviving, down from 31% at the first-ever pass. Most are in `lasso.rs` (50),
`stats.rs` (45) and `ewridge.rs` (42). The 45 in `stats.rs` are in
`P2Quantile` and `SlotMetrics`, loops whose mutations spin, and they are the
obvious next batch. [Where the 217 stand now](#where-the-217-stand-now) has
the count by file.

### What is left

**The original blocker is gone.** It was that *nothing has ever been pushed,
no CI job has ever run on any platform*. As of 2026-08-31 the repo is pushed, and
CI has run on all three platforms.

Cleared:

| entry | what cleared it |
|---|---|
| **T-D1** | The workflows have run. What they claimed is now measured, and three of the claims were wrong: see below the table. |
| **T-W1** | `cargo test`, `maturin develop` and pytest have all executed on Windows: **712 passed, 1 failed** on `d6158aa`, down from 9 failures. Every one of the nine was a test bug, not a library bug: cp1252 encoding, `str(WindowsPath)` backslashes, a hardcoded POSIX `PATH`, a bash-only test. |
| **T-W7** | The 126 committed golden pipeline outputs compared at 1e-12 on Windows, so the golden comparison is genuinely cross-platform now. |
| **T-W5**, **T-W3b**, **T-W8**, **T-W2** | All executed as part of that run. |

The three claims that proved wrong were the Linux `ld` SIGBUS (disk, not
memory), the cache that never saved, and the disk exhaustion *inside* the
cache restore. The SIGBUS is told in the comment above the `test` job in
`.github/workflows/ci.yml`; the other two are in `docs/RELEASE-READINESS.md`.
`d6158aa` is not a commit in this repository's history, perhaps a CI merge
commit.

Since cleared (2026-09-06):

**The tenth Windows failure** was a `UnicodeEncodeError` in
`examples/pathway_integration.py`. It was fixed and pinned by a test. The cost
policy had taken Windows off push while the repo was private (see the COST
POLICY comment in `ci.yml`). The repo is public, every push runs all three
OSes, and Windows has been green on every release since 0.1.0.

**PyPI.** `polars-online` is published there, 0.1.0 on 2026-09-03 and 0.1.1
on 2026-09-04, through the trusted-publisher `release.yml`. The Polars pin
question is settled as `polars>=1.34.0,<3`: see "The Polars pin" in
`docs/RELEASE-READINESS.md`.

Two things are worth doing periodically rather than once:

| run | when | what to watch for |
|---|---|---|
| **`./scripts/mutants.sh`** | after a batch of feature work | a cluster of survivors in one function, which almost always means its only oracle lives in the Python suite. The runs so far are described in [What the mutation run actually found](#what-the-mutation-run-actually-found) |
| **`./scripts/coverage.sh`** | periodically | reported and never gating |

## How the suite looks for defects

### An oracle, not a golden number

**The 2026-09 batch (tasks 45–56) added five models and four helper
modules**, and its testing pattern is worth naming, because it is what found
the defects. **Every recursion got a longhand oracle written from the paper,
not from the code**, and the oracle was allowed to disagree. It did, six
times:

| model | what the oracle found |
|---|---|
| `deco` | its `rho` was not `ew_cov`'s `corr` |
| `hmm` | its Π seeding was not the prior mean |
| `corrchange` | its scalar CUSUM was identically zero |
| `bocpd` | it implemented Algorithm 1's line 6 with the new run holding one row of the old regime |
| `rcov` | its pre-averaging window was off by one |
| `corrchange` | its size study was comparing Gaussian draws against a `t₅` table |

Each is recorded in `docs/PLAN.md` §11a, with what it measured. **An oracle
written *from the implementation* agrees with it by construction.** That is
the one that is easy to get wrong, and it is how `bocpd`'s line 6 survived
twelve unit tests.

The mutation work below followed the same rule: add an *oracle*, not a golden
number. An oracle took one of four forms:

| form | example |
|---|---|
| the recursion written out longhand beside the implementation | Holt, Page-Hinkley, `sigma2` |
| an equivalent model configured a different way | the slow twin against a standalone model at `long_halflife`; the standardized solve against the plain one at zero penalty |
| the optimality conditions of the problem being solved | the lasso's KKT conditions |
| the definition of the statistic | `read` against a recomputation from the raw rows |

### What the mutation run actually found

`scripts/mutants.sh` runs `cargo mutants` over `online-core`. It makes one
small change to the source, rebuilds, and reruns the tests. A missed mutant
means every test still passed with the code deliberately broken: a gap in the
tests, not a bug in the code. The passes so far:

| pass | mutants | missed | timeouts |
|---|---|---|---|
| the first-ever pass (T-D4) | 1645 | 517, with 1104 caught and 24 unviable | |
| run 1, before the follow-up work (T-D5) | 2616 | 501 | 175 |
| run 2, after it, under load | | 104 as reported, which was wrong | 425 |
| run 3 (`--iterate`), 14 min on an idle machine | 690 | **217**, the honest current figure | 18 |

The headline number (501 of 2616 mutants surviving) is less interesting than
its shape. Grouped by function, the survivors were:

| function | missed | why |
|---|---|---|
| shape and state accessors (`n_eff`, `n_targets`, `n_features`, `sigma2`, `coefficients`, `kind`) | 81 | asserted nowhere in Rust, in any model |
| `<Lasso as OnlineModel>::step` + `Lasso::solve` + `standardized` | 66 | KKT verification lives in `tests/test_oracles.py` |
| `EwRidge::blend_toward_long_run` | 54 | `session_shrink` is tested only from Python |
| `<EwRidge as OnlineModel>::step` + `solve` + `run_solve` | 54 | the slow twin, `sigma2`, and the solve schedule |
| `EwRidge::solve_standardized` | 53 | the `add_intercept = false` branch had no test at all |
| `*Cfg::validate` (seven models) | 34 | rejections are asserted in `tests/test_edge_cases.py` |
| `EwCovModel::read` / `labels` / `n_outputs` | 26 | `ew_cov` is reachable only through the Polars layer |
| `<Holt as OnlineModel>::step` | 19 | brand new, and its tests checked outcomes not arithmetic |
| `Kalman::pred_var` + `<Kalman as OnlineModel>::step` | 17 | surfaced only as `emit_sigma` / spec options |
| `EwCov::precision` / `with_precision_prior` / `partial_corr` | 15 | E2's precision matrix, solved on demand and exposed only as a statistic (C5) |

#### The mutation blind spot

**One cause explains nearly all of it: `cargo mutants` runs `cargo test`, so
everything proven only by the pytest suite through the compiled extension is
invisible to it.** That is already noted in `scripts/mutants.sh` as the
reason for scoping the run to `online-core`. But the same blind spot applies
*inside* `online-core`, wherever a feature's only oracle is a Python test.

Eight commits closed it, taking `online-core`'s Rust tests from 151 to 218.
The rule they followed was to add an oracle in one of the four forms
[above](#an-oracle-not-a-golden-number), not a golden number.

#### Five defects the behavioural tests were happy with

Five real defects surfaced in the process, all in code the behavioural tests
were happy with. **The most serious was a zero-weight row at the head of a
stream, which permanently disabled `ewridge` and `lasso`.** Their per-target
mean-form update computes `a = lam·wj / (lam·wj + w)`, which is 0/0 when
nothing has ever carried weight. The NaN never washed out: `wj` stayed NaN,
`NaN > 0.0` is false, and the model silently stopped predicting for the rest
of the stream. Every other model already guarded it, and so did
`EwCov::update` two lines away. It was found indirectly. A Rust unit test for
the analogous guard in `blend_toward_long_run` failed, which pointed at the
same shape in `step`. `tests/test_edge_cases.py::TestWeights` now checks it
for all ten regression models.

The other four:

| defect | what it did |
|---|---|
| `sgd` and `pa` reported `n_eff` with the current row's decay already applied | `min_periods` meant a different number of rows for them than for every other model |
| `EwCovModel::n_targets` returned 1 | for a model that regresses nothing |
| `blend_toward_long_run` had lost its doc comment | to a `#[cfg(test)]` helper inserted between the comment and the function |
| `coef_prior` misconfiguration | reported "coef_prior must be 1 vectors of length 3" |

**Equivalent mutants were left alone deliberately: they cannot be killed by
any test.** `Ftrl::weight`'s `zz < 0.0` sign branch is only reachable when
`zz == 0`, which the `|zz| <= l1` guard above it has already returned on. In
Holt's `beta > 0.0 && d_clock > 0.0`, the second test can never decide,
because beta is `1 - 0.5^(d/halflife)`, zero exactly when d is. Neither says
so in the source today. Holt's note stood until its rewrite in task 80
(`ddc9d91`) removed `beta`, and the branch with it. FTRL's branch, now in
`weight_of` in `crates/online-core/src/ftrl.rs`, never carried such a note.

#### Do not run anything else while a mutation pass is going

**Let the machine be idle.** A mutation pass and a build loop cannot share a
laptop without corrupting the classification. Run 2 reported 104 survivors
and 425 timeouts, and this document said 104 for several commits. It was
wrong by a factor of two. Run 3 (`--iterate`) re-tested exactly those on an
otherwise idle machine, and found **132 of the "timeouts" were ordinary
survivors**. The tests had passed, and had only taken longer than the
ceiling. `./scripts/gate.sh` was running concurrently: cargo build, cargo
test, maturin, pytest, repeatedly, on the same four cores. Because
cargo-mutants counts a timeout separately from a miss, the contention did not
look like an error. It looked like good news.

**Treat a large timeout count as a failed run, not a result.** Run 1 had 175
timeouts of 2616. Run 2, under load, had 425. Run 3, idle, had 18 of 690.
Above a few percent, the numbers underneath are not trustworthy.

`--minimum-test-timeout 10` in `scripts/mutants.sh` makes a pass faster, but
*more* sensitive to this, not less. It is right for an idle machine and wrong
for a busy one.

#### Where the 217 stand now

| file | run 1 | run 3 |
|---|---|---|
| lasso.rs | 83 | 50 |
| stats.rs | 5 | 45 |
| ewridge.rs | 178 | 42 |
| robust.rs | 16 | 27 |
| sgd.rs | 15 | 15 |
| kalman.rs | 32 | 10 |
| ewcov.rs | 75 | 10 |
| everything else | 97 | 18 |

Three files went *up*, which is the same measurement artifact in reverse:
their run-1 figures were themselves depressed by timeouts. `stats.rs` is the
clearest case. Its 45 survivors are in `P2Quantile` and `SlotMetrics`, both
of which are loops whose mutations spin. They are the obvious next batch of
work, and the same rule applies: add an oracle, not a golden number.

### FFI memory and crash safety (2026-08-31)

Two copies of Polars live in this process, and data crosses on the Arrow C
Data Interface. There a `SeriesExport` carries a `release` callback back into
the binary that produced it. Nothing else in the suite would notice if that
contract were broken: a leak is invisible, and a double-free is a crash that
takes pytest with it. `tests/test_ffi_memory.py` covers it: 16 tests (~7 s)
when written, and 13 since task 85 (`6d9b983`) removed the expression
plugin's four and `test_many_tiny_groups` took the place of one (`b9e4977`).

**The assertion is "plateaus", not "does not grow."** Allocators do not
return pages eagerly, rayon spawns workers lazily, and Polars cached the
loaded plugin while there was one. This was measured on the plugin, before
task 85 removed it.
Our `.over()` cost a one-time **+6 MB** and was then flat across 1,800
iterations (per-block deltas +0.7, −0.9, +1.1, −0.3, −0.3). Native Polars
`.over()`, by contrast, slowly *returns* memory. A naive "RSS must not grow"
test would have failed on that step forever. So `assert_plateaus` compares
the later blocks against each other: a step passes, a slope fails. When
written, it discarded the first block. Since 2026-09-08 it finds the end of
the ramp instead. Blocks of 120 iterations run until two in a row each grow
by less than 4 KB per iteration. Only then are five more measured.

**The statistic is the median block-to-block gap**: five measured blocks of
120, so four gaps, and three when written. Comparing the tail's first mark
against its last cannot tell a late step from a slope. On 2026-09-06 that
comparison failed `main` on marks of [361112, 364160, 364160, 367644] KB.
Two blocks were identical to the page, then came one 3.4 MB step. It
reported 14.5 KB/iter, on a tree whose previous run was green. The median gap
reads the same trace as 0.0 KB/iter, and still reads a sustained 14 KB/iter
leak as 14. A heuristic that cries wolf on a release day is worse than a
looser one that does not.

What the leak tests cover:

| case | test |
|---|---|
| repeated `fit_predict` | `test_repeated_fit_predict` |
| bank churn | `test_bank_churn` |
| fifty groups of thirty rows, through the bank's own group column; when written, `.over()` across groups | `test_many_tiny_groups` |
| multi-chunk inputs: a chunked Series exports one `ArrowArray` per chunk, each needing release | `test_multi_chunk_input` |
| sliced frames that share their parent's buffers | `test_sliced_frame_sharing_buffers` |
| the bank's error path; **both error paths** when written, before the plugin's went with task 85 | `test_the_bank_error_path_still_releases` |
| outputs outliving their inputs | `test_output_outliving_its_input` |
| state round-trips | `test_state_round_trip` |

Crash cases run in a **subprocess with `faulthandler`**, so a segfault is a
failed assertion with a native traceback instead of a dead test session:

| case | test |
|---|---|
| GC of frames mid-flight | `test_gc_of_frames_mid_flight` |
| repeated reference/dereference with refcount checks | `test_repeated_reference_and_dereference` |
| reference cycles holding a bank | `test_reference_cycles_are_collectable` |
| empty/single-row/all-null frames | `test_empty_and_degenerate_frames` |
| pickle round-trips | `test_pickled_bank_survives_a_round_trip` |

When written, the crash cases also held both FFI paths interleaved while each
held the other's exports. That test went with the plugin in task 85.

**`scripts/leakcheck.sh` goes further than RSS can.** It runs `leaks` on
macOS, which walks the malloc zones for unreachable blocks, and valgrind on
Linux. Two limits each took a deliberate leak to find (2026-09-03):

| limit | what it means |
|---|---|
| it sees Python objects only under `PYTHONMALLOC=malloc` | pymalloc's arenas are mmap'd. Against a 128 MB refcount leak the default allocator reported "0 leaks", which is what the earlier "clean" result here was |
| it cannot see anything allocated through polars' allocator | that is every Rust-side allocation, so that side stays with `test_ffi_memory.py` |

The count is differential, 1 iteration against 1000, since the interpreter
leaves ~11k blocks unreachable at exit regardless. The script has a control
mode that leaks one object per iteration and must fail. It runs weekly in CI
(`leakcheck.yml`), on both platforms, control included. It is not in
`gate.sh`, because it needs a live process and valgrind is ~50x slower. Run
it after touching `crates/online-py` or the extraction path.

Two findings worth recording, neither a leak:

| finding (2026-08-31) | what became of it |
|---|---|
| The error paths are clean. The sharper one was the plugin's: the engine had already exported the inputs when our function returned an error, so releasing them happened on a path that only runs when something has gone wrong. | The plugin went with task 85; the bank's error path is still tested. |
| A **String feature column was silently parsed back to f64**, and a non-numeric one became all-nulls rather than raising. That was consistent with the null policy and not a memory issue. But `pytest.raises` on a String column did not fire, and a Categorical was the dtype genuinely refused. | Since IMPROVEMENTS U2 (`667bac1`, 2026-09-01) a String column is refused by name ("has dtype str; it must be numeric"), pinned by `test_a_string_column_is_refused_not_cast_to_null` in `tests/test_error_messages.py`. |

## The entries, by ID

Priorities: **P1** = closes a PLAN promise or covers a found defect; **P2** =
meaningful new assurance; **P3** = infrastructure. A struck-through
priority, such as ~~P1~~ **done**, marks an entry that is finished. Each
section has one table, with one row per ID, and a paragraph under it for an
entry whose detail does not fit a cell.

### A. Close the oracle gaps (our own references)

| # | P | improvement | what it found or still lacks |
|---|---|---|---|
| T-A1 | ~~P1~~ **done** | **`kalman_ref` in `tests/reference.py`**: agreement to 1e-9 (observed max 1.1e-15) | three load-bearing subtleties |
| T-A2 | ~~P1~~ **done** | **Lasso KKT verification**, `tests/test_oracles.py::TestLassoOptimality` | still lacks a numpy CD `lasso_ref` for the *pred* path |
| T-A3 | ~~P2~~ **done** | **`robust_ref`** covers both Huber and quantile; agreement ~1e-13 | two load-bearing details |
| T-A4 | ~~P2~~ **done** | **`ftrl_ref`**; agreement ~1e-16 | when the decay is applied |
| T-A5 | ~~P2~~ **done** | null policy, warmup, clock semantics and the universal invariants over all ten regression models, `tests/test_semantics_all_models.py` | **found the robust `n_eff` defect**, and `sgd`/`pa`'s |

**T-A1.** `kalman_ref` is a plain numpy predict/update recursion, mirroring
the standardization-from-prior-stats scheme. It agrees to 1e-9 (observed max
1.1e-15) across scalar and per-factor `coef_halflife`, `inf` pinning,
explicit `q` and fixed `obs_var`/`p0`. It also covers multi-target with and
without `share_p`, the null policy and the null-target path, and
`add_intercept=False`. Writing it confirmed several subtleties are
load-bearing: scales come from the stats
*before* the row, `Q·Δclock` is applied once per shared `P`, and the
innovation variance carries `σ²/w`.

**T-A2.** `tests/test_oracles.py::TestLassoOptimality` checks stationarity
and the subgradient conditions at the emitted coefficient snapshot, on the
model's own standardized statistics rather than a ported solver.
`g_i = c_i − (Cb)_i − l2·b_i` must equal `l1·sign(b_i)` where `b_i ≠ 0`, and
satisfy `|g_i| ≤ l1` where it is zero. It runs across λ ∈ {0, 0.01, 0.1} ×
`l1_ratio` ∈ {1.0, 0.5}, plus sparsity monotonicity along the path and the
intercept identity. A numpy CD `lasso_ref` for the *pred* path is still open.
It would catch schedule and warm-start bugs the KKT check cannot see.

**T-A3.** `robust_ref` is in `tests/reference.py`. It agrees to ~1e-13
across `huber_delta` ∈ {0.5, 1.5, 10}, τ ∈ {0.1, 0.5, 0.9}, nulls,
multi-target and standardized solves. Two details proved load-bearing while
writing it. The robust weight scales the accumulator update but **not** the
`sigma2_j` update, since otherwise the scale estimate shrinks itself. And a
zero robust weight still decays the accumulator.

**T-A4.** `ftrl_ref` agrees to ~1e-16 with and without clock decay, across
`l1` ∈ {0, 0.5, 5}, custom α/β/l2, no-intercept and null targets. The decay
is applied to `n` and `z` *before* the proximal weights are computed, so a
row's prediction already reflects its own elapsed clock.

**T-A5.** `tests/test_semantics_all_models.py` runs the null policy, warmup,
clock semantics and the universal invariants (chunk invariance, save/load,
group independence, no non-finite outputs) against all ten regression
models. That was seven when written; `sgd`, `pa` and `holt` joined when they
landed. The genuinely model-specific deviations are named in the module
docstring rather than skipped silently (`rls` predict-only on any null
target; `lasso` slot naming; `ftrl` probabilities). It found two defects.

- **The robust models reported `n_eff` as the sum of *IRLS weights* rather
  than observations.** A quantile spec showed `n_eff ≈ 1001` after three
  rows, since quantile weights reach `2/quantile_eps` ≈ 2000×, so
  `min_periods` was effectively inert. `Robust` now tracks a raw-weight
  observation count for `n_eff`/`min_periods`, while the accumulators keep
  using the robust weights.
- **`sgd` and `pa` reported `n_eff` with the current row's decay already
  applied**, found as soon as the sweep reached them. Every other model
  reports the weight before the row's update and before its decay, so
  `min_periods` meant a slightly different number of rows depending on the
  model. Both now follow the documented convention.

### B. Cross-checks against river

`tests/test_river.py` holds these, with river as a dev-dependency. The module
skips cleanly when river is not installed, the same pattern as the offline
skip. There are two tiers: **exact**, at a tolerance of ~1e-12 with the
configuration pinned so the algorithms coincide, and **statistical**, tail
agreement after warmup with the tolerance stated per test.

| # | P | tier | comparison | agreement |
|---|---|---|---|---|
| T-R1 | ~~P1~~ **done** | exact | **`ftrl` ≡ `river.optim.FTRLProximal`**, on the state recursion | 1e-12, row for row, with and without L1 |
| T-R2 | ~~P2~~ **done** | exact | **Kalman(q=0, fixed `obs_var`, `standardize=False`) ≡ `river.linear_model.BayesianLinearRegression`** | 3.6e-15, across three (alpha, beta) settings |
| T-R3 | ~~P2~~ **done** | exact | **`EwCov` ≡ `river.stats.Mean` / `Var` / `Cov` / `PearsonCorr`**, with no decay | 1e-9 |
| T-R4 | ~~P2~~ **done** | statistical | **EW mean/var vs `river.stats.EWMean` / `EWVar`** | they converge in the limit, and disagree sharply during warmup |
| T-R5 | ~~P3~~ **done** | statistical | **Quantile regression (intercept-only) vs `river.stats.Quantile`** (P² algorithm) at τ ∈ {0.25, 0.5, 0.75} | both land near the empirical quantile |
| T-R6 | ~~P3~~ **done** | statistical | **Huber vs `river.linear_model.LinearRegression(loss=optim.losses.Huber)`** under 3% contamination | both beat least squares on the clean slope, and agree with each other |

**T-R1.** The comparison is at the level of the state recursion. Driven by
the same gradient sequence, river's `optim.FTRLProximal` and our model agree
on the z/n recursion to 1e-12, row for row, with and without L1. Comparing
the two *models* end to end is not exact. **River's `LogisticRegression`
predicts with the previous step's proximal weights**, while we follow McMahan
Algorithm 1 and recompute from `z` at prediction time. That is a real
semantic difference, and the ordering difference is itself pinned by a test
rather than left as a surprise.

**T-R2.** `TestKalmanIsBayesianLinearRegression` maps `p0 = 1/alpha` and
`obs_var = 1/beta`, and matches exactly, to 3.6e-15, across three (alpha,
beta) settings. It includes a guard that turning standardization on breaks
the match, so the switch cannot silently become a no-op. The class was
deleted in `509c6cf` and restored in task 90 (2026-09-23).

**T-R3.** The agreement is exact (1e-9) with no decay, reached via the new
`ew_cov` surface (ENHANCEMENTS E1). The final test in that class quantified
where the two diverged: at a 1e9 offset, river's Welford form was still exact
while our raw-moment form had lost the variance entirely. That was the gap
E11b closed (`509c6cf`). `EwCov` keeps centered co-moments, and the final
test is now `test_matches_river_even_on_a_large_offset`, which holds the two
to agreement at such offsets.

**T-R4.** The mapping is `fading_factor = 1 − 0.5^(1/halflife)`, and it found
a second convention difference. Ours is the *bias-corrected* weighted mean:
it divides by the accumulated weight, so it is the exact weighted mean from
row 1. River's `EWMean` is the un-normalized EWMA `m += f·(x − m)`, seeded at
its first value, which stays anchored near that seed during warmup. Both the
closed-form match and the convergence in the limit are asserted.

**T-R5 and T-R6**, the statistical tier. Our IRLS quantile lands near the
empirical quantile alongside river's P² estimator. Our Huber and river's
SGD-Huber both beat least squares under 3% contamination, and agree with each
other.

### C. Edge-case matrix

Findings first: T-E1 and T-E2 were live defects, both verified against the
build of the time when this document was written, and both are fixed.

| # | P | case | what holds now |
|---|---|---|---|
| T-E1 | ~~P1~~ **done** | **Negative weight value** | **Was a defect**, now fixed: a finite negative weight is an error naming the column, value and row |
| T-E2 | ~~P1~~ **done** | **Null group key vs a group literally named `"<null>"`** | **Was a defect**, now fixed: `GroupKey(Option<String>)` keeps them apart |
| T-E3 | ~~P1~~ **done** | ±inf in features / targets / weight / clock | pinned by `TestNonFinite`, plus outputs are never non-finite |
| T-E4 | ~~P1~~ **done** | Mis-ordered chunks (clock goes backwards across a chunk boundary within a group) | both halves covered: the absorbing policies, and `on_clock_reset="error"` |
| T-E5 | ~~P2~~ **done** | Degenerate solves in the **plain** path | finite outputs throughout, with `solve_failures` observable |
| T-E6 | ~~P2~~ **done** | Duplicate clock values, `max_dclock = 0`, `halflife` far below the typical Δ | pinned; no NaN leaks under extreme decay |
| T-E7 | ~~P2~~ **done** | Minimal shapes | empty chunks, single-row groups and a one-feature/one-target spec all behave |
| T-E8 | ~~P2~~ **done** | Non-string group and session columns, null session values | a null session value **is** its own session |
| T-E9 | ~~P2~~ **done; found a defect, then removed the limit** | **Large-offset cancellation** | slope-recovery error at a 1e6 offset **2.0e-03 → 6.8e-10** |
| T-E10 | ~~P2~~ **done, decision taken** | Datetime-typed clock columns | read in their own nanoseconds, with clock parameters as durations; every duration unit, in each form, against every column unit |
| T-E11 | ~~P3~~ **done** | Long-stream soak: 10⁷ rows through one state | `n_eff` bounded, the fit accurate, the state under 4KB; opt-in |
| T-E12 | ~~P3~~ **done** | Pending-delta across a save/load boundary; session change on a group's first row | both targeted |
| T-E13 | ~~P2~~ **done** (2026-09-04) | The chunk plan's group layout, per-field assembly and parallel extract (P9–P11) | the layout is invisible; no defect found |

**T-E1.** `EwCov::update` silently no-opped when `λW + w ≤ 0`, while the
per-target `r_j` update ran with a negative denominator. In practice every
later prediction went null. Finite negative weights are now rejected at
extraction, naming the column, value and row. Non-finite ones skip the row
like any other non-finite input, uniformly with non-finite features. `w = 0`
is a legal pure-decay row, and `EwCov::update` also debug-asserts the
contract. `tests/test_edge_cases.py::TestWeights` covers all ten regression
models (all seven when written) and the `w = 0` pure-decay case.

**T-E2.** Both mapped to the string `"<null>"` and shared one stream:
`n_eff` was verified to accumulate across them. `GroupKey(Option<String>)`
replaces the `"<null>"` string sentinel, so a null group is structurally
distinct from a group named `"<null>"`. Bank files gained a `format_version`,
2 then. It is still 2 for most files, and since task 88 a bank whose specs
carry a duration writes 3. Version 1 files still load, because the key
serializes transparently as its inner `Option`. `TestGroupKeys` covers it,
including save/load and integer group columns.

**T-E3.** `TestNonFinite` pins ±inf/NaN in features, targets, weights and the
clock. A non-finite feature or weight skips the row, and the clock still
advances. A non-finite target is predict-only, and a non-finite or null clock
errors loudly. A fuzz-ish test adds that outputs are never non-finite.

**T-E4.** `TestClockOrdering` pins the absorbing policies as they are: a
backwards delta across a chunk boundary goes through `on_clock_reset`,
indistinguishable from real data. The test sets `min_backwards_jump=0.0` for
that. By default a backwards jump smaller than `min_backwards_jump`, which
defaults to `max_dclock`, is refused as out-of-order rows, whatever
`on_clock_reset` says. It also guards that correctly ordered chunking stays
invariant. The strict mode, `on_clock_reset="error"`
(ENHANCEMENTS E3), is implemented, and catches a mis-sorted chunk boundary
loudly.

**T-E5.** The degenerate solves are collinear and constant features in the
plain path. Exactly collinear features drive the jitter fallback (107
jittered solves over 200 rows), with finite outputs throughout. A real ridge
removes the need for jitter entirely, and non-solving models report 0. It
required implementing ENHANCEMENTS E5 first: `Bank::solve_failures()` /
`ModelBank.solve_failures()` expose the count per spec and group.

**T-E6.** Duplicate clock values are zero deltas, so no decay.
`max_dclock = 0` disables decay entirely. A halflife far below the delta
makes every row effectively the first (`n_eff → 1`), and no NaN leaks under
extreme decay.

**T-E7.** An empty chunk is accepted, and returns an empty frame with the
output column. An empty chunk *between* real chunks changes nothing.
Single-row groups, a group appearing in only one chunk, and a
one-feature/one-target spec all behave.

**T-E8.** Integer and categorical group columns, and integer session columns,
are covered. A null session value **is** its own session: `a → null` and
`null → a` both count as changes, and `null → null` does not. That was
previously undocumented; it is now pinned.

**T-E9.** The first finding was a threshold. The zero-variance drop threshold
was `1e-10 × raw second moment`, ~450,000× the real noise floor. So a
unit-variance feature on a **1e6 offset was silently dropped with coefficient
0**, at an ordinary financial scale. Then the underlying cause was removed
(ENHANCEMENTS E11b): `EwCov` keeps centered co-moments, and the solves read
them directly. Slope-recovery error at a 1e6 offset went **2.0e-03 →
6.8e-10**. Offsets of 1e8/1e10 went from dropped entirely to 2.5e-08 /
4.8e-07. The tests pin the new range, and keep the old numbers in comments so
a regression is obvious.

**T-E10.** This was a silent trap. A temporal clock cast to f64 exposes its
internal representation: the same 60 seconds is 60e3 / 60e6 / 60e9 units for
`Datetime(ms/us/ns)`, and 1 unit per day for `Date`. So `halflife=600` on a
microsecond column silently meant 600 µs, decaying every row to nothing and
producing plausible-looking garbage with no error. **Decision: reject
(2026-08-30), then durations (task 88, 2026-09-23).** A temporal clock is
read in its own integer nanoseconds and takes its clock parameters as
durations; the gap between two rows is taken in integers before it becomes
seconds, so a nanosecond timestamp keeps its nanoseconds whatever the
stream's age (2026-09-24). A plain number on it is refused, naming the column,
the parameter and both fixes (a duration, or `dt.epoch`).
`tests/test_temporal_clock.py` is the trap turned into a pass: the same
instants as `Datetime(ms/us/ns)` give a float clock's numbers to the bit, and
`TestEveryUnitAgainstEveryColumn` holds every duration unit, in each form
that can write it (text, `pl.duration`, `timedelta`), against every temporal
column kind and unit, a zone-aware one among them: 192 cases, none skipped --
the exact recursion where the column can express the unit, a refusal by name
where a cap or a threshold is finer than the column's step. Numeric clocks (int and float) are unchanged, and are
asserted to agree with each other.

**T-E11.** 10M rows go through one state in ~6.5s. `n_eff` stays bounded and
does not drift between the start and end of the stream, the coefficients are
still accurate, and resume is still exact. A 2M-row state serializes to under
4KB: memory is O(state), not O(data). It is opt-in, via `pytest -m soak`.

**T-E12.** Splitting a stream exactly after a skipped row and resuming from
state reproduces the unbroken run. A group's first row is treated as first,
even when it also changes session.

**T-E13.** `crates/online-polars/tests/chunk_plan.rs` runs one interleaved
fixture on both sides of `PAR_MIN_ROWS`, at 64, 4095, 4096, 4097 and 12305
rows. The fixture has 40 groups, a null key, two one-row groups, nulls in
every input, sessions, and an ungrouped spec beside the grouped ones.

| claim | how it is held |
|---|---|
| the layout is invisible | the same rows interleaved, sorted by group, and each group alone give the same output and the same state bytes |
| the key type does not matter | a String key and an Int32 key give the same everything |
| a column's arrow chunking does not matter | a column split into arrow chunks of 1 to 9000 rows, null-free pieces included, reads the same |
| other numeric dtypes read like Float64 | Int8, UInt8, Int32, Int64, Float32, Boolean and a `Null` column |
| the threshold is not a seam | chunk sizes around it |
| an error names the frame row, not the layout position | in the clock, weight and state readers, under a permuted layout; a refused chunk leaves the bank untouched, and the readers' errors keep their fixed precedence |
| `predict` skips unseen groups | in any layout, without touching the bank |

**It found no defect.** The tests were then mutated three ways, and each
mutation was caught: `source_row` ignoring the layout, and the fast path and
`gathered` skipping the gather.

### D. Windows and cross-platform

Windows is a **stated deployment target** (CLAUDE.md: "Dev on macOS (arm64);
deploy on macOS and Windows"). When this section was written it was the
least-verified part of the project: no CI job had executed there. CI runs
there on every push now, but "run CI" is not itself a test plan. These are the specific
things Windows can break that macOS never will, and what each found. The P
column is as first written. The last column says what has cleared since CI
first ran on Windows, as [What is left](#what-is-left) records it.

| # | P | case | since CI ran on Windows |
|---|---|---|---|
| T-W1 | ~~P1~~ **done** | **Run `ci.yml` on `windows-latest` at all** | runs on every push since 2026-08-31 |
| T-W2 | **P1** | **Cross-OS state hand-off** (`release.yml`: write on macOS, load on Windows/Linux) | executed on Windows CI; the hand-off has run for every release since 0.1.0 |
| T-W3 | P1 (partly) | **Path handling through the CLI**: escaped Windows-style paths and paths with spaces | |
| T-W3b | ~~P1~~ **found a real bug, fixed** | TOML path escaping in the CLI tests | executed on Windows CI |
| T-W3b (original) | P1 | **Path handling through the CLI**, as first written: backslash separators, drive letters, UNC paths, and spaces in paths, in both the TOML `input`/`output`/`load_state`/`save_state` fields and the `--input`/`--output` overrides | |
| T-W4 | ~~P2~~ **mitigated + tested locally** | **CRLF line endings in the TOML config** | |
| T-W5 | ~~P2~~ **tested locally** | **Binary/artifact naming**: `online.exe` vs `online` | executed on Windows CI |
| T-W6 | ~~P2~~ **pinned locally** | **Float formatting in output field names** | |
| T-W7 | P2 (**mechanism built**) | **Numeric reproducibility across OS/CPU** | compared at 1e-12 on Windows |
| T-W8 | P3 | **Filesystem behavior**: case-insensitivity, `MAX_PATH`, file locking on rewrite | executed on Windows CI |
| T-W9 | ~~P3~~ **done** | **`scripts/env.sh` has no Windows equivalent** | |

**T-W1.** Written when `cargo test --workspace`, `maturin develop` and the
pytest suite had never executed on Windows. They have since 2026-08-31, on
every push. The first run found nine test bugs and no library bug.

**Windows portability of the *tests*: the first Windows run found 9, all in
tests, none in the library.** After the TOML fix cleared the Rust side, the
pytest suite failed nine ways on Windows. The failures fell in four classes,
every one a test making a Unix assumption:

| class | what failed | the fix |
|---|---|---|
| **(1) encoding** | `Path.read_text()` and `subprocess(text=True)` default to the locale codec, which is cp1252 on Windows. It cannot decode the box-drawing characters polars prints or the arrows in our own README (`UnicodeDecodeError: 'charmap' codec ... byte 0x90`) | explicit `encoding="utf-8"`, suite-wide in 7 files rather than only at the reported sites |
| **(2) path separators** | `test_a_clean_checkout_has_what_the_build_needs` compared `str(WindowsPath(...))` (backslashes) against git's forward-slash output, and declared a tracked file missing | `as_posix()` |
| **(3) environment** | the thread-determinism subprocess passed a hardcoded `PATH=/usr/bin:/bin`, leaving the child with no resolvable interpreter | it inherits `os.environ` |
| **(4) shell** | `test_release_packaging` executes a `run:` block that only ever runs on the workflow's ubuntu job, and Git Bash's `find` differs enough to fail for reasons that say nothing about the workflow | skipped on Windows, where Linux and macOS still cover it |

That the library itself passed all 718 tests on Windows first time is the
result worth recording.

**T-W2.** PLAN §9 class 7 and hard rule 5. As first written: the msgpack
payload has no host-dependent parts *by construction*, and `save_bytes` is
asserted deterministic locally, but that is an argument, not a test.

**T-W3.** Escaped Windows-style paths and paths with spaces are now tested
through the CLI on any OS, and round-trip. Actual resolution on Windows still
needs a runner.

**T-W3b.** The first Windows CI run ever attempted failed exactly here. Three
`online-cli` tests died on `toml::de::Error`, "too few unicode value digits",
because the test interpolated `C:\Users\runner\...` straight into a TOML
*basic* string, where `\U` begins a unicode escape. TOML was right and the
caller was wrong. That makes it the same trap any Windows user hand-writing a
config falls into, with an error message that says nothing about paths. It
was fixed in three places. The test escapes properly. The CLI's parse error
now names all three valid forms (literal string, doubled backslashes, forward
slashes) whenever the config contains a backslash. And the README documents
it. Two new tests run **on every OS**, because the mistake is about the
config text rather than the host. They are
`test_an_unescaped_windows_path_is_rejected_with_a_usable_hint`, and a
parametrized check that each recommended form actually parses and runs. Those
would have caught this on Linux, before a Windows runner existed. Drive
letters and UNC *resolution* still need a real Windows runner.

**T-W3b (original)**, as first written. `PlRefPath::try_from_pathbuf`
normalizes Windows paths (polars has explicit `normalize_windows_path`
logic). TOML string escaping means `"C:\data\x.parquet"` needs doubling or a
literal string. Neither is exercised.

**T-W4.** A `.gitattributes` now normalizes line endings to LF on checkout,
so a Windows checkout cannot introduce CRLF, and the CLI is tested against
both CRLF and LF configs. As first written: the repo has no
`.gitattributes`, so git may check out configs with CRLF on Windows. `toml`
handles it, but the example config and any doc snippets should be proven to
parse as checked out.

**T-W5.** `tests/test_release_packaging.py` stages what `download-artifact`
leaves behind: one directory per matrix job, two of them holding a file
called plain `online`. It then runs the workflow's own bash against it. The
shell is *extracted from `release.yml`*, not copied, so editing the workflow
changes what the test runs. That was verified by breaking the workflow and
watching three tests fail. Pinned: exactly the Windows artifact keeps `.exe`,
the two unix binaries do not collapse onto one name, contents are copied not
just renamed, and the PyPI job's `dist/` collects wheels only. What still
needed a runner, as first written, is whether the Windows job produces
`online.exe` in the first place.

**T-W6.** The exact field-name list for a grid spec is asserted, so a
platform divergence fails loudly: 58 fields, and 52 when written. Task 87
(`3e0b359`) added `settled_frac`, `withheld_reason` and `support_coef` at
each of the grid's two halflives. The list embeds formatted floats. Combo
labels are built with `format!("{r}")` on f64 (e.g. `pred_y__r0.000001`).
Rust's float `Display` is locale-independent, so this *should* be identical
everywhere. But the field names are part of the public schema: users index
the output struct by them. So a divergence would silently break every caller
reading a field by name. Hence: assert the exact field-name list on every OS.

**T-W7.** `tests/test_golden_pipeline.py` commits 502 outputs from a fixed
stream through a 25-spec bank. Of those, 156 are read at each of three rows:
25, 60 and 119. The other 34 are not per row. They are `marginal`'s 20 pair
statistics, read at the end, `corrchange`'s 4 statistics, and `rcov`'s 10
closed-group values. When written it committed 135 outputs from an
eleven-spec bank at three rows each, and 126 until IMPROVEMENTS X2 added
`ftrl`, the one model it had never pinned. It compares with a 1e-12 relative tolerance. `golden.rs` already
pinned the Rust core; this pins the *Polars* layer, which nothing did:
extraction, per-group fan-out, diagnostics, struct assembly. Locally the
agreement is exact. The tolerance is what "the same answer on another
platform" is allowed to mean. Different LLVM vectorization and BLAS paths can
reorder floating-point operations, while a genuinely divergent algorithm
shows up far above it. It was verified to bite at 1e-11 and to tolerate
1e-14. The comparison became a cross-platform one the first time CI ran, with
no further work.

**T-W8**, as first written. The bank writes state with `std::fs::write` and
the runner opens the output parquet with `File::create`. A still-open reader
on Windows makes rewriting fail where POSIX allows it. That is relevant to
`--resume` loops. Both writes have since moved to a temporary file and a
rename (IMPROVEMENTS C6, `crates/online-polars/src/atomic.rs`).

**T-W9.** `scripts/env.ps1` was added, dot-sourced (`. .\scripts\env.ps1`).
Writing it surfaced two real Windows gaps in `.vscode/settings.json` that the
shell script alone would not have. The Windows `PATH` entry omitted uv's
`%USERPROFILE%\.local\bin`. And `rust-analyzer.cargo.extraEnv` hardcoded
`PYO3_PYTHON` to `.venv/bin/python`, which does not exist on Windows
(`.venv\Scripts\python.exe`); that key cannot be made OS-specific the way
`terminal.integrated.env.*` can. The fix was to set `VIRTUAL_ENV` instead.
`pyo3-build-config`'s `get_env_interpreter` resolves it with
`venv_interpreter(dir, cfg!(windows))`, so one value is correct on all three
platforms.

### E. Infrastructure

| # | P | improvement | where it stands |
|---|---|---|---|
| T-D1 | ~~P1~~ **done** (2026-08-31) | **Actually run the workflows once.** | the workflows have run; [What is left](#what-is-left) has the results |
| T-D2 | ~~P2~~ **done** | **Property-based testing** (hypothesis) | `tests/test_properties.py` |
| T-D3 | ~~P2~~ **done** | Determinism across parallelism | `tests/test_portability.py` |
| T-D4 | ~~P3~~ **done** | Coverage, and **mutation testing** | reported, not gating; making the mutation pass periodic in CI is still open |
| T-D5 | | Mutation re-run | **done, three passes** |

**T-D1.** Until the workflows ran, T-W1/T-W2 and the wheel builds were
untested claims. It *was blocked on GitHub credentials on this machine*: no
keychain entry for github.com, no SSH key, no `GH_TOKEN`, and `gh` was not
installed, so `git push` could not authenticate. The unblock was any of
`gh auth login` / an SSH key / a PAT, and the token needed the **`workflow`
scope**, since this push added `.github/workflows/`. The repo has been pushed
since 2026-08-31, and CI runs on Windows on every push; the Windows results
are in [What is left](#what-is-left).

**T-D2.** `tests/test_properties.py` uses hypothesis to generate adversarial
streams: mixed nulls, duplicate/long-gap clocks, ±1e8 values, zero weights,
tiny groups. It asserts the universal invariants for all ten regression
models. They are chunk invariance under any chunk size, save/load
transparency at any split, outputs finite-or-null, no `n_eff` reported by a
skipped row, and group independence. The strongest is that **changing a
row's own target never changes that row's own prediction**: out-of-sample by
construction, hard rule 2. A Rust-side `proptest` pass on `online-core`
remains possible, but is largely redundant now.

**T-D3.** `tests/test_portability.py` runs the bank in subprocesses at
`POLARS_ONLINE_MAX_THREADS=1` and `=8`, and requires identical output, every
field compared exactly. Since 2026-09-04 it runs on both sides of
`PAR_MIN_ROWS`: 400 rows over 6 groups, and 5000 over 37, which exercises
the parallel extract and assembly. Both use a halflife grid with every
optional output on.

**T-D4.** Coverage: `scripts/coverage.sh` reports 96% Python and 75%/73%
Rust ([the caveat](#measured-coverage)), and CI reports the Python figure,
non-gating. **Mutation testing** (`scripts/mutants.sh`) was run in full over
`online-core`: **1645 mutants, 517 missed / 1104 caught / 24 unviable**. The
misses concentrated exactly where the Rust unit tests lean on the *Python*
oracle suite, which `cargo test` cannot see: `robust.rs` 68% missed,
`kalman.rs` 38%, `ewridge.rs` 36%. The fix was
`crates/online-core/tests/golden.rs`: one fixed 60-row stream per model with
the exact expected predictions embedded, which pins the arithmetic against
any mutation. Measured on the worst file, **`robust.rs` went from 162 missed
/ 77 caught to 42 / 197**, a 74% reduction from one test. The residue is
mostly accessors (`n_features -> 0`) and validation-branch comparisons, which
are low value. Re-running the full pass for a new headline number is done
(T-D5). Making it periodic in CI is still open: no workflow under
`.github/workflows/` runs it.

**2026-09-02:** `golden.rs` had signatures for seven of the eleven kinds.
`sgd`, `pa`, `holt` and `ew_cov`, the four with longhand-recursion oracles in
their own modules rather than numpy references, now have one each, and `sgd`
two. `test_model_registry::test_the_core_golden_file_pins_every_model` holds
the file to `KINDS`. Each new pin was mutated once (the Huber clamp, the
PA-II damping, the trend smoothing, the variance floor), and each caught its
own. The same day, `test_every_builder_has_a_per_model_test_file` gave the
per-model test file its check, EXTENDING step 11 (step 13 when written). It
also gave `ewridge`/`rls` their own `test_<model>.py`, out of `test_bank.py`.

**T-D5.** Three passes, all in the
[table of passes](#what-the-mutation-run-actually-found). Run 1, before the
follow-up work, found **2616 mutants, 501 missed**. Run 2, after it, reported
104 missed, a number that was wrong
([why](#do-not-run-anything-else-while-a-mutation-pass-is-going)). Run 3
(`--iterate`, 690 mutants in 14 min on an idle machine) found **217 missed**,
the honest current figure. That is 8.3% surviving, down from 31% (517/1645) at
the first-ever pass, despite the crate having grown by 60%. The misses were not
scattered. They clustered almost perfectly on the code whose *only* tests live
in `tests/*.py`, because `cargo mutants` runs `cargo test` and cannot see the
Python suite. Eight commits of Rust-side oracles followed; see
[What the mutation run actually found](#what-the-mutation-run-actually-found).
