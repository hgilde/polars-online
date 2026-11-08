<!-- kind: record -->
# The testing log

*Status, 2026-11-08: a record. What the test suite found and when, the hardening rounds beyond the eight test classes, how the mutation runs and the crash-safety audit looked for defects, and the improvement entries by ID (sections A to E, entries such as T-D4 and T-W9, which the code and tests cite), moved here whole from `docs/TESTING.md`. Each part reads as of its date; rows marked **blocked** or **never executed** were written before the repository had CI, and *What the first CI runs cleared* says what cleared them. `docs/TESTING.md` holds what the suite proves now and where it is thin.*

### The suite's size, counted

| counted | Rust tests | pytest |
|---|---|---|
| 2026-09-06 | about 650 | 1,250 functions, some 2,200 cases |
| 2026-09-24 | 831 | 1,564 functions, 2,838 cases |
| 2026-09-25 | 976 | 3,265 cases |
| 2026-10-03 | 1,215 | 3,840 cases |
| 2026-10-11 | 2,207 | 6,262 cases |
| 2026-10-24 | 2,653 | 7,211 cases |

The counts before 2026-10-03 used `cargo test --workspace -- --list`, which
also lists `online-py`, a crate with no tests. The soak tests were added to
the opt-in set on two dates: the long-run stability cases and the
contamination sweep on 2026-10-25, and the worst-seed grids of the
clustering and changepoint models on 2026-10-26.

### Stories TESTING.md told, moved here on 2026-11-08

The rewrite of 2026-11-08 kept TESTING.md to each rule and its reason in the present tense. The events below are what those rules came from, as TESTING.md told them then.

#### The oracles that found the 0.2.0 batch's defects

**An oracle is written from the paper or the definition, never from the
code it checks.** An oracle written *from the implementation* agrees with it
by construction. That is the one that is easy to get wrong, and it is how
`bocpd`'s line 6 survived twelve unit tests.

The batch that added `deco`, `hmm`, `corrchange`, `bocpd` and `rcov`, with
four helper modules, shipped in 0.2.0 on 2026-09-06. Its testing pattern is
what found its defects. Every recursion got a
longhand oracle written from the paper, and the oracle was allowed to
disagree. It did, six times:

| model | what the oracle found |
|---|---|
| `deco` | its `rho` was not `ew_cov`'s `corr` |
| `hmm` | its Π seeding was not the prior mean |
| `corrchange` | its scalar CUSUM was identically zero |
| `bocpd` | it implemented Algorithm 1's line 6 with the new run holding one row of the old regime |
| `rcov` | its pre-averaging window was off by one |
| `corrchange` | its size study was comparing Gaussian draws against a `t₅` table |

#### The libraries taken for tests, as TESTING.md told it

An open license is one the Open Source Initiative approves, or CC0, named in
the metadata as an SPDX expression or an "OSI Approved" classifier. A library
whose metadata names none is listed with the license read at its source and
where: `bayesian-changepoint-detection`'s wheel declares none, and its
repository's `LICENSE` is MIT. A library that must not be in every
contributor's environment, for its platforms, gets a group of its own and a
CI job of its own. For rule 3, `tests/child.py` puts
`tests/_site/sitecustomize.py`, which installs the same finder, on the path
of every child interpreter a test spawns. Only
`tests/test_pyarrow_interop.py`'s children see pyarrow. The children are
covered because on 2026-09-25 the examples' "nothing needs pyarrow" was
found running with pyarrow importable. The gate checks the lock with
`uv lock --check`.

**scikit-learn is taken** (2026-09-25). `tests/test_sgd.py`
compares `sgd` with `SGDRegressor` live. `tests/test_second_opinion.py` holds `huber`
to `LinearRegression` at `huber_delta = 1e9`, the limit at which no row is
down-weighted, and to `HuberRegressor` under outliers. It holds
`marginal`'s best split to a `DecisionTreeRegressor` stump, and every
`ewridge` solve to `Ridge`.

**Vowpal Wabbit is taken** (the user, 2026-09-29: "find an alternative
oracle for ftrl that supports more options"). Its `--ftrl`
runs the recursion `ftrl.rs` states, and it reaches options river's
comparison does not: the squared loss, row weights, the intercept and the
model's own predictions. `TestFtrlIsVowpalWabbits` holds `pred` and `coef`
to it on every row, to 1e-5, since VW computes in single precision. The
wheel has no dependencies and ships for Python 3.12 to 3.14 on all three
operating systems, so it sits in the dev group like the rest. None of the
libraries checked forgets as a half-life does (river, VW, and Keras's
`Ftrl`), so a finite half-life stays with `ftrl_ref`.

**hmmlearn and padasip are taken** (2026-10-06). hmmlearn's
`GaussianHMM` holds `hmm`'s filter at fixed parameters, row by row.
padasip's `FilterRLS` holds `rls`. hmmlearn ships wheels up to CPython 3.13,
so the 3.14 legs build its sdist, a small C++ extension. padasip is pure
Python. In the same round scikit-learn's `SGDRegressor` and `SGDClassifier`
took on `sgd`'s losses one row at a time, and filterpy the `kalman` settings
`kalman_ref` alone held.

**Pathway is parked.** It would run the Pathway half of
`examples/pathway_integration.py`, but it is under the Business Source
License, which rule 1 parks. `test_pathway_is_not_a_dependency` keeps it
out of `pyproject.toml` until the user decides.

#### A forward mean's defect, found by two oracles

The brute force and the identity both found a defect while the operators
were built: a forward mean counted rows that held a value from outside its
window.

#### The window run's nine rounds of checks

The window run's resume rules came out of nine rounds of checks on
2026-10-03, R1 to R9. A test pins each finding, and its docstring names the
finding by its round, such as R5-C1.

#### What ran at once, and what happened

| what ran at once | load on 14 cores | what happened |
|---|---|---|
| eight agents, with the other user's jobs | 31 | every build and test slowed in proportion |
| five essentials gates | 78 | gates of about 300 s took 500 to 2,000 s |
| four `cargo mutants --in-diff` runs | 88, for hours | the kernel panicked (`watchdog timeout: no checkins from watchdogd in 90 seconds`) and the Mac rebooted |
| three agents resumed straight after that reboot | | the same panic, eight hours later |

#### Why several branches share one gate

**Several branches in flight share one essentials gate.** A gate per
branch cost twice over when branches ran side by side. Each gate builds
and links the release extension with every core, so five at once took 500
to 2,000 s each where one alone takes about 300. And every branch that
landed made the others merge `main` and gate again, most often after a
clean merge.

#### The cache eviction that failed the oracle run

The repository's caches sit at GitHub's 10 GB limit, most of it Rust's build caches, and
GitHub then evicts the least recently used entry; that took the 0.9 MB
dataset entry just as Fränti's server stopped answering, and the oracle
run failed.

#### Fränti's server and the skipped guards

Fränti's server timed out from the runners on nearly every
leg of the 25 CI runs to 2026-11-05, and the two oracle guards that read
its sets skipped on each. ... Among hundreds of tests a skip is easy to miss: those two guards skipped on every such leg while the run passed.

#### What the second-opinion twins found

The twins found two of the five
single-seed bounds narrower than the seeds' spread, and hold the measured
spread. The third was a defect: `marginal`'s windowed lag correlations
were centered at the running mean as it stood, 0.165 from `acf` over the 50
seeds, and are now held to their definition to rounding.

#### The ledger's first CI run

On the ledger's first CI run each shard fetched its own sets from
a cold cache, one shard's download of Fränti's S4 failed, and its grid
skipped as offline. Its three cases went unmeasured, and the ledger job
failed on them.

#### The mutation triage, newest first, and the equivalents it found

**Equivalent mutants are left alone deliberately: no test can kill them.**
Two from the first passes show the kind. Flipping `Ftrl::weight`'s
`zz < 0.0` sign test to `<=` changes it only at `zz == 0`, which the
`|zz| <= l1` guard above it has already returned on. In Holt's
`beta > 0.0 && d_clock > 0.0`, the second test can never decide, because
beta is `1 - 0.5^(d/half_life)`, zero exactly when d is. Holt's note in the
source stood until a rewrite of `holt` (`ddc9d91`) removed `beta`, and the
branch with it. FTRL's branch, now in `Ftrl::weight_of` in
`crates/online-core/src/ftrl.rs`, is recorded in
`scripts/mutants_equivalent.toml` with that reason.

**The last whole-crate count recorded here is run 3's, on 2026-08-30: 217
missed**, 8.3% surviving, down from 31% at the first-ever pass. Most were in
`lasso.rs` (50), `stats.rs` (45) and `ewridge.rs` (42), and each of those
files has since been triaged until every mutant in it is caught or recorded
with its reason. The triage, newest first:

| scope | date | missed at the start | at the end |
|---|---|---|---|
| the 698 mutants the push to `103d721` left untested at the job's time limit, and the lines two batches of fixes changed that day, in `kalman.rs`, `sgd.rs`, `pa.rs`, `rls.rs`, `robust.rs`, `stats.rs`, `lasso.rs` and the cluster summaries | 2026-10-08 | 71, run locally on the essentials tier | 55 caught by twenty-one new tests, 14 equivalent, 2 tolerated (rounding); and one of the equivalents the triage of `103d721`'s changed lines listed, `ewridge`'s `coef_variance` guard, was not: `Q ≤ W²` fails in doubles once `W²` underflows while `λ > 1/2` keeps `Q` at `2^-1074`, and a test now catches it |
| the lines the push to `103d721` changed, in `kalman.rs`, `ewridge.rs`, `clock.rs` and `lasso.rs` | 2026-10-08 | 105, run 37834489097 | 86 caught by ten new tests, 19 equivalent; two of those went with the code that left a fit with no kept column NaN and `inf` live, where a resumed stream read its documented values |
| `lasso.rs` and `ewridge.rs`, afresh | 2026-09-27 | 104 of 906, and 152 timed out | every mutant caught or recorded, after two `--iterate` rounds |
| the lines a batch of performance fixes changed in `gaps.rs` and `ewlagcov.rs` | 2026-09-27 | 43 viable | 42 caught, the other equivalent |
| `stats.rs`, on its own | 2026-09-24 | 48 of 285, no timeouts | 5: one caught since, and four equivalent |

**`lasso.rs` and `ewridge.rs`.** The tests that closed them lean on
third-party oracles where there is one. Every `ewridge` solve is held to its
closed form, solved by `faer` (already a dependency, and not the model's own
Cholesky). It is held to scikit-learn's `Ridge` too, in
`tests/test_second_opinion.py`, 20 cases to 1e-8. `lasso` without an
intercept and no penalty is held to least squares through the origin, again
by `faer`. The rest hold a definition written out:

| quantity | held to |
|---|---|
| the row leverage, and each coefficient's support share | their definitions, per target and ridge |
| the window's Kish count and residual spread | the rows inside the window |
| the solve schedule | a count, row by row |
| a zero-weight copy of every row | no change, to the bit |
| a saved state | its bytes, since an in-memory `State` clones the kept factors and hid their rebuild |

**`stats.rs`.** Rerun on its own, it left 48 of 285 missed: 30 in
`P2Quantile`, 12 in `EwAutoCorr` and 6 in `SlotMetrics`, with no timeouts.
Oracle tests closed them:

| survivors in | held to |
|---|---|
| `P2Quantile` | the algorithm box of Jain & Chlamtac (1985), marker by marker, written from the paper, on a continuous stream and a discrete one, where markers tie |
| `EwAutoCorr` | a regime switch, invariance under a shift and a scale, a zero co-moment before the first pair, and `same_shape`'s bounds |
| `SlotMetrics` | `weight_sum`'s definition and the strict 0.5 threshold |

After them, 5 of 285 were missed. The discrete stream catches one, the tie
at the minimum, and the other four were equivalent mutants. Three were in
`P2Quantile`. One was `d_sign > 0` against `>=`, where `d_sign` is only ever
±1. The other two were the two `<` of the parabolic order check against
`<=`, which differ only when the prediction lands exactly on a neighbor.
On 2026-10-02 `EwQuantile` replaced `P2Quantile`, which never forgot. It is held
to the exponentially weighted quantile's definition row by row, and the three
equivalents went with `P2Quantile`. The fourth is still listed: `SlotMetrics::update`'s
`denom > 0` against `>=`, where a scored row's `w > 0` keeps `denom`
positive.

### Beyond the eight classes

These parts test what the eight classes do not name: first how the library
behaves, then the files the repository promises.

| part | test | what it checks |
|---|---|---|
| [production hardening round 3](#production-hardening-round-3-vs-rivers-own-battery) | `tests/test_production_hardening.py` | river's battery of checks, sklearn's estimator invariances and statsmodels' oracle convention |
| [post-refactor hardening](#post-refactor-hardening-p1p8-review) | `tests/test_hardening.py` | parameter ranges at their edges, and row counts large enough to expose stride bugs |
| [the validated defaults](#the-validated-defaults-are-still-the-measured-ones) | `tests/test_validation_doc.py` | `docs/VALIDATION.md`, regenerated and compared |
| [the declared schema](#the-declared-schema-is-checked-for-every-model) | `test_names_match_the_realized_struct_for_every_model`; `tests/test_every_kind.py` | the declared field names against the struct the bank produces |
| every model in every list | `tests/test_model_registry.py` | each per-model list against the Rust registry: the builders, the sweeps, the API snapshot, the golden files, each model's own test file, the README's model headings, the release comparison's workload and `_spec.UNSUPERVISED` |
| [hard rule 1](#hard-rule-1-is-enforced-not-remembered) | `tests/test_repo_hygiene.py` | no data file, large file or generated output is tracked |
| [the examples](#examples-are-executed) | `tests/test_examples.py` | everything under `examples/` runs unmodified |
| [memory and crash safety](#ffi-memory-and-crash-safety-2026-08-31) | `tests/test_ffi_memory.py`, `scripts/leakcheck.sh` | no leak and no crash across the FFI |
| the shared contract | `crates/online-core/tests/model_contract.rs` | every model's shape accessors, its `n_eff` against one recursion (hard rule 8), its prediction slots and its state round trip |
| a stopped feature or target (2026-09-24) | `crates/online-core/tests/held_values.rs` | every model that centers a feature or target keeps what exact arithmetic gives for 150 half-lives after it stops, at levels from 0 to 1e12 and with no decay: the slope it learned, a spread that keeps decaying, and fits that do not depend on the level. It has 21 tests. Of the fourteen it had when written, twelve fail with each running mean held as one plain double, where the models now carry it as a pair of doubles, to twice a double's precision; the other two are contracts any design must keep: a row of weight 0 changes nothing, and a state saved part-way resumes to the bit |
| Arrow with pyarrow (2026-09-24) | `tests/test_pyarrow_interop.py` | pyarrow 25.0.1 reads the Arrow output through `pa.array`, `pa.chunked_array`, `pa.record_batch` and `pa.table` with `fit_predict`'s values and dtypes; a reader streams into a bank with the whole frame's numbers; the export validates in full, zero rows and an all-null field cross, two specs are two exports, and a requested schema other than the struct's own raises inside pyarrow 25.0.1's cast path (its bug, `array.pxi:321`) rather than coming back silently cast. pyarrow is a test-only dependency: `tests/conftest.py` makes it unimportable in the rest of the suite, and `tests/child.py` in the child interpreters the tests spawn, which so run as a user without it does |
| a weight's scale (2026-10-02) | `tests/test_weight_scale.py` | every spec of the release probe's workload, at its weights and at a hundred times them with `min_weight` scaled alike: only the specs whose docs say a weight counts on the sum scale move |
| row order (2026-09-17; its exemptions narrowed 2026-09-29) | `tests/test_order_hazards.py` | a query whose row order is unspecified is warned about before a bank reads it, with an `OrderNotGuaranteedWarning` naming the step and the fix; `test_what_is_not_order_free` holds the specs `ModelBank.fit` exempts from the warning to those whose state cannot depend on the order |
| a delayed label at a break (2026-10-02) | `TestBreaksCountElapsedTime` and `TestBreaksOnSkippedRows` in `tests/test_label_delay.py` | a row is learned once its delay has passed in elapsed time, counted from the frame, and a break releases nothing early |
| an integer clock (2026-10-07) | `tests/test_integer_clock.py`, `an_integer_*` in `crates/online-core/src/clock.rs`, `an_integer_clock_decides_its_edges_in_integers` in `crates/online-polars/src/windows.rs` | an `Int64` clock of epoch nanoseconds near 1.79e18, where a double resolves 256, against the raw integers: its steps, `settled_frac`, a `coef_every` cadence, a model window's edge, an embargo's release, a step back of 1 and one of exactly `restart_after_step_back`, a gap of exactly `gap_cap`, the window operators against Polars' `rolling_sum_by` and `po.increment` of an integer input; the same stream shifted to 0 as a float clock gives the same output to the bit; the frames give the clock in its own integer dtype |
| the clocks a row reports (2026-10-02) | `tests/test_clocks.py` | `emit_clocks`' `scored_clock`, the clock a row was scored at, and `learned_clock`, the clock of the newest row learned by then: counted from the frame, and exact to the nanosecond on a `Datetime` clock |
| `kalman` at any row spacing (2026-10-03) | `crates/online-core/tests/kalman_clock.rs` | `coef_half_life` is a half-life on the clock: a slope step takes about the same clock time to learn at rows 1, 0.25 and 0.04 apart, where the old form parted by a factor of five, and a gap row adds exactly `q d²` to `P` |
| [the math library's last bits](../TESTING.md#the-math-librarys-last-bits) | the bound helpers in `crates/online-core/src/oracle.rs`, and the comments beside each oracle comparison | each oracle comparison held to a bound derived from both sides' rounding, a few ulps for each call to `exp`, `ln`, `exp2` or `powf` among the terms; each value pinned to its bits on a stream that calls none of them |
| the solve cadence | `tests/test_solve_cadence.py` | by default, `ewridge`, `lasso`, `huber` and `quantile` solve again once the weight learned since the last solve reaches `ln 2 / 50` of the weight the fit holds, so a long half-life still re-solves |
| [how cost grows](#how-cost-grows-is-held-not-only-what-it-is) | `crates/online-bench/tests/scaling.rs`; the `scale_*` cases in `crates/online-bench/benches/instructions.rs`; `tests/test_scaling.py` | each surface's cost at two sizes of each size it has, held to the growth its algorithm declares: a row's time and allocations, the saved state and its heap natively, a row's instructions on CI |

The files the repository promises are held the same way:

| contract | held by | what it holds |
|---|---|---|
| the public API | `tests/test_api_surface.py` | every name, default, signature and output field name, against the snapshot `tests/api_surface.txt`, so a change is a reviewable diff; since 2026-10-07 also the defaults that resolve in Rust, the frames' columns and dtypes, the TOML keys, the CLI's flags, the environment variables, and the words each string-valued parameter takes |
| the README's warm-up defaults | `tests/test_spec_defaults.py` | the `min_weight` table's rule for every model, at two feature counts and with and without an intercept, and the readiness gates' defaults, against what the bank resolves |
| the names that follow Polars (2026-10-02) | `tests/test_renames.py` | an old parameter is refused naming the new one, from a builder and from a spec dict; an old output name is gone |
| a released state | `tests/test_released_state.py` | the files each release wrote, from 0.10.0 on, each from its wheel on PyPI, the four before 1.0 listed in the test and every 1.x release read from PyPI's own list, so a release is held from its upload with no edit: a 0.x release's are refused by their schema version, and a 1.x release's load and go on as this build's own; offline, the test skips |
| a released output | `scripts/compare_release.py` | the release workload's outputs, every field name a model writes among them, against the newest release's, bit for bit: a report in CI, and a step of each release |
| the documentation | `tests/test_production_hardening.py`, `tests/test_api_links.py`, `tests/test_doc_structure.py`, `tests/test_llms_txt.py`, and Sphinx with `-W` | every README python block and docstring example runs; every link into the API reference resolves; every Markdown file git tracks passes `docs/WRITING.md`'s structure checks; `llms.txt`'s model names and links are current; every docstring is valid reStructuredText |
| docstring text | `test_weight_scale.py::test_the_docs_say_what_each_unit_is`; `test_temporal_clock.py::TestEveryClockParameterTakesADuration` | whole phrases of `ftrl`'s and `micro`'s docstrings, a line wrap included; "clock units" in the entry of each parameter measured in them. A docstring pass keeps both (`docs/WRITING.md` §6) |
| the workflows | `tests/test_release_workflow.py`, `tests/test_release_packaging.py` (T-W5), `tests/test_ci_cost_policy.py` | the release tags what it published, after everything ran; the release artifacts' names; the CI matrix and its timeouts |

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
| **(4)** `clip_gradient=inf` is the documented way to disable clipping | the JSON layer refused it, while `half_life=inf` worked | `clip_gradient` uses the same `Num` type |

The round also implemented the river checks that apply and that the suite
lacked:

| check | what it asserts |
|---|---|
| **feature-order invariance** | spec order and frame order, plus extra-columns tolerance |
| **pickling** | `__reduce__` via `save_bytes`, so pickle and `copy.deepcopy` resume bit-exactly. It needed `#[pyclass(module=...)]`, since pickle cannot name a class that claims to live in `builtins` |
| all-null columns | |
| first-row outliers | with the washout horizon stated: ~80 half-lives, inherent to EW accumulators |
| a 15-case extreme-parameter sweep | p0 at 1e±12, τ at 0.01/0.99, and deliberate SGD divergence, pinning finite-or-null |
| state-byte corruption | every magic-string byte must detect; any flip anywhere must fail cleanly, never panic |
| concurrent `fit_predict` from two threads | a safe error, and the object usable after |
| unicode/space column names | through save/load |
| README python blocks | compiling, when the round was written. They are now *run*, each in its own copy of a namespace holding what the prose has introduced by that point. That found the README's `holt` example refused by the library's own validation |

River checks *not* adopted, with reasons:

| check | why not |
|---|---|
| clone/repr/params | our specs are plain dicts |
| emerging/disappearing features | a fixed schema is a design decision, pinned as a clear error instead |

#### Post-refactor hardening (P1–P8 review)

**Done, with zero new defects: every path held, including the never-executed
one.** This was a second full review after the performance rewrite. It aimed
at two blind spots. One was parameter ranges tested only in their
comfortable middles. The other was row counts too small to expose stride
bugs in the new flat slot-major buffers. It lives in `tests/test_hardening.py`:
18 tests (~4.5 s) when written, and 20 collected cases on 2026-09-24.

| attack | what must hold |
|---|---|
| a kitchen-sink stream at **30k rows with every output enabled at once**: 156+ fields from two instances × ridge grid × feature sets × two targets, sigma/z/drift/metrics/autocorr/quantiles/selected/averaged, groups, sessions, weights, nulls, clock gaps | the same digest across chunkings incl. row-at-a-time, across a mid-stream save/load, and across `POLARS_ONLINE_MAX_THREADS` 1 vs 8 |
| the **coupled drift path** (grid + `drift_action="reset"`), which P2 added and nothing executed | a break in either instance resets both; the row-major path is chunk-invariant; it equals the parallel path when nothing fires |
| parameter edges: half-life 1e-3 and inf, k=64, quantile levels 0.001/0.999 | half-life 1e-3 and inf give their exact limits |
| **weight-scale invariance at 1e±6** | all weights ×c changes nothing but `weight_sum`: the test that the accumulators are in the mean form. `tests/test_weight_scale.py` runs it over every kind, apart from the cases each model's docs name |
| twelve targets with per-target warmup | each lands on the exact ceil(threshold) row |
| the `coef` cadence: unset, `coef_every` 0 and 997 clock units, `max_rows_between_coefs` 997 | across chunk boundaries: each group's last accepted row in each chunk unset, the same rows under a cadence |
| P6's reader thread, on a corrupt file and on a mid-stream bank error | a clean exception, with no deadlock on either side of the channel |
| the P5 spec cache | two specs on one thread stay distinct |

The soak (10M rows) was re-run after the refactor, and passes.

#### The validated defaults are still the measured ones

**Done.** `tests/test_validation_doc.py` regenerates `docs/VALIDATION.md`
and compares it to the committed copy, with timings normalized away. So the
numbers the shipped defaults were chosen from cannot silently stop being
true. It was re-run after the whole enhancement backlog, including the switch
to centered co-moments in `EwCov`, and **every number is unchanged**. That is itself the
result: the numerics work did not move the measured optima. A second
assertion checks the script still runs all five experiments, so the
comparison cannot pass on a document that no longer justifies anything.

#### The declared schema is checked for every model

**Done.** A guard written with the residual quantiles and autocorrelations
held `ewridge`'s declared field names to the struct the bank produces.
`test_names_match_the_realized_struct_for_every_model` extends it to all ten
regression models, times four output combinations (three when written),
plus `ew_cov` separately. The optional
outputs are assembled in the stream layer and so generalize. But each model
contributes its own prediction and coefficient slots, and `sgd`, `pa`, `holt`
and `ew_cov` all postdate the original test. `tests/test_every_kind.py`
holds every kind's output struct to `po.spec.output_fields`,
parametrised over the registry's `MINIMAL`, so a new kind is checked the day
it is registered.

#### Hard rule 1 is enforced, not remembered

**Done.** `tests/test_repo_hygiene.py` fails if a data file, a large file, or
generated tool output is tracked. It also fails if `.cache/`, `target/` or
`mutants.out/` stop being gitignored, or if a file the build needs is missing
from what `git archive` (a fresh clone, an sdist) would produce. It was
written after 136 files of `cargo mutants` output sat tracked for several
commits, swept in by a `git add -A`, with nothing complaining. It was
verified to fire as well as to pass.

#### How cost grows is held, not only what it is

**A budget catches a workload that got slower; a ratio of two sizes catches
one that grows the wrong way.** A budget in `budgets.toml` fixes the cost of
one workload at one size and compares each run with it. A cost that grows
with a size the workload holds small is invisible to it, however long it has
been there: `with_windows(group=...)` walked every group the stream had seen
on every row in every release from 1.0.0, while every window test used a
handful of groups and the one budget case with 1,000 groups set its number
with the walk already in it. So each surface also declares how a row's work
and its state grow with each size it has, and
`crates/online-bench/tests/scaling.rs` measures every declaration at two
sizes, `n` and usually `8n`, and fails where the ratio exceeds the declared
growth. No earlier run's number is read, so a defect present from the first
day fails on the first run. docs/PERFORMANCE.md's
[table](../PERFORMANCE.md#how-cost-grows-with-size) gives every cell, its
reason and its measured ratios.

**Natively the counts are exact and time is a backstop.** An allocation
count, the saved state's bytes and the heap freed on drop are the same on
every run, so they are held within 25% of the declared growth. A block's
allocations are counted on every thread but the test harness's. The
measured work also runs on threads it does not name: the bank's pool,
Polars' pool, the runner's reader. So the count cannot follow the work by
thread, but the harness's threads are known. The main thread of libtest
reports each finished test and starts the next while another test measures. A
test's own thread formats its verdict while the next cell runs. Counted
with the rest, they moved a cell that allocates nothing to 4 allocations
over its blocks when the binary's tests ran side by side. That is past what
a ratio of O(1) allows. A count may also read 4 allocations a block past
its declared growth, whatever its ratio. That is what no mark reaches: the
harness allocates on a new test's thread before the test's first line can
mark it as the harness's. A growth the check is for adds one a row, at
least 200 a block. Time is held
within a factor of `step^0.5`, 2.8 at a step of 8, the least of three blocks
taken in turn at the two sizes. It is the process's CPU time where the
platform reads it, and the wall clock's on Windows. On a loaded machine a
thread waits its turn for a core. That stretches a long block and may
leave a short one whole, so the wall clock's ratio of two sizes grows with
the load. At a load of about 50 on 14 cores, every run judged by the wall
clock failed cells of the right growth, and measuring them again did not
pass them. Judged by CPU time, no cell failed in thirteen runs at loads
of 25-99; the highest read 0.89 of its bound. On CI,
Callgrind counts the instructions of
the cells in `[scaling.cells]` of `budgets.toml` at both sizes, and
`scripts/budgets.py check` holds their ratio within one run to 10% past the
declared growth. It counts under a stopped wall clock, so two runs of one
binary agree to 0.001%: at the machine's clock, the hash seed Polars' maps
draw from the time once a process moved a bank cell's count by up to 2%
between runs, past the quarter of the 2% tolerance that two runs may differ
by ([TESTING.md](../TESTING.md#budgets-that-only-tighten)). Only the
counted process sees the stopped clock; the tests and the library run at
the machine's.

**A cell that fails is a finding, held in the extended tier until it is
fixed.** It keeps its declaration and its test is marked `#[ignore =
"extended: scaling finding <id>"]`, so the essentials stay green while the
fix is dispatched, and the extended tier, which every push runs, fails until
the fix lands. A declaration changes only where the algorithm's own growth is
higher by design, with the reason in the table. Each kind has a cell for the
rows a stream has already seen and one for a size of its own
(`every_kind_has_its_cells`), so a new model joins the table with its first
test.

#### Examples are executed

**Done.** `tests/test_examples.py` runs everything under `examples/`
unmodified. The Pathway operator example runs end to end, on its plain-batch
path, since Pathway is BSL and not a dependency. That is asserted by checking
it appears in no dependency group. The cursor examples,
`examples/duckdb_cursors.py` and `examples/adbc_cursors.py`, run against
real DuckDB and SQLite databases. `examples/bank.toml` goes through the real
CLI for `--dry-run`, a full run, and `--load-state` from the state the run wrote.
A documented example that no longer works is worse than none, and until this
test nothing ran either file.

**It found a documentation defect.** The README's chunk-invariance guarantee
said "bit-identical output" and set nothing apart. But `coef` is snapshotted
on each chunk's last row as well as every `coef_every` rows, so smaller
chunks report it more often. The guarantee now sets `coef` apart, and every
computed field is still bit-identical.

## What it has found

**The suite has found defects in the library, in its documents and in its
own tests, and two differences from river that are not bugs.**

### Defects, and where each is told

Each row names what found the defect, and where it is told in full or the test that now holds it.

| finding | found by | told in, or held by |
|---|---|---|
| six errors in the 2026-09 batch: `deco`, `hmm`, `corrchange` twice, `bocpd` and `rcov` | oracles written from the paper | [An oracle, not a golden number](../TESTING.md#an-oracle-not-a-golden-number) |
| a zero-weight row at the head of a stream permanently disabled `ewridge` and `lasso` | the mutation run | [What the mutation run actually found](#what-the-mutation-run-actually-found) |
| `sgd` and `pa` reported `weight_sum` with the current row's decay already applied | the mutation run, and T-A5's sweep | the same, and T-A5 in [A](#a-close-the-oracle-gaps-our-own-references) |
| `EwCovModel::n_targets` returned 1; `blend_toward_long_run` lost its doc comment; a `coef_prior` misconfiguration gave a garbled message | the mutation run | [What the mutation run actually found](#what-the-mutation-run-actually-found) |
| the robust models reported `weight_sum` as the sum of IRLS weights | T-A5 | [A](#a-close-the-oracle-gaps-our-own-references) |
| a finite negative weight turned every later prediction null | writing this document (T-E1) | [C](#c-edge-case-matrix) |
| a null group key and a group named `"<null>"` shared one stream | writing this document (T-E2) | [C](#c-edge-case-matrix) |
| a unit-variance feature on a 1e6 offset was silently dropped | T-E9 | [C](#c-edge-case-matrix) |
| `half_life=600` on a microsecond clock silently meant 600 µs | T-E10 | [C](#c-edge-case-matrix) |
| a target listed as its own feature was accepted, and so were duplicate names; a ±inf prediction could pass; `clip_gradient=inf` was refused | production hardening round 3 | [Production hardening round 3](#production-hardening-round-3-vs-rivers-own-battery) |
| the README's `holt` example was refused by the library's own validation | running the README's python blocks | the same |
| the README's chunk-invariance guarantee did not set `coef` apart | running the examples | [Examples are executed](#examples-are-executed) |
| three `online-cli` tests wrote Windows paths into a TOML basic string; the parse error said nothing about paths | the first Windows CI run | T-W3b in [D](#d-windows-and-cross-platform) |
| nine pytest failures on Windows, all test bugs and no library bug | the first Windows CI runs | T-W1 in [D](#d-windows-and-cross-platform) |
| a `UnicodeEncodeError` in `examples/pathway_integration.py` | CI on Windows | [What the first CI runs cleared](#what-the-first-ci-runs-cleared-2026-08-31-to-2026-09-06) |
| two Windows gaps in `.vscode/settings.json` | writing `scripts/env.ps1` | T-W9 in [D](#d-windows-and-cross-platform) |
| a String feature column was silently parsed back to f64 | the FFI audit | [FFI memory and crash safety](#ffi-memory-and-crash-safety-2026-08-31) |
| a feature constant inside a `window_size` read as the subtraction's remainder, which a windowed lasso at a zero penalty divided by itself (predictions of 1e55) | the oracles of 2026-09-24 | `tests/test_oracles_lasso_paths.py` |
| the lasso's `penalty_selected` under a `window_size` read errors outside it, four ways | the same oracles, then a Rust one | `tests/test_oracles_lasso_paths.py` |
| `holt` reported `coef` as `[0, 0]` before a target's first observation | the same oracles | `tests/test_oracles_gradient_paths.py` |
| a CLI test skipped on Windows for want of `online.exe`, and wrote Windows paths into a TOML basic string | reading the first Windows run's skips | `toml_config` in `tests/test_closed_groups.py`, which writes paths POSIX-style |
| a feature or target that stops moving: its running mean stopped `1/(2b)` rounding steps short, and every variance and slope centered on it read that gap (a lasso slope of -4.7e3 at a level of 1e8) | measuring the windowed lasso's case of 2026-09-24 outside a window | `crates/online-core/tests/held_values.rs` ([a stopped feature or target](#beyond-the-eight-classes)) |
| a windowed `marginal` pair kept the remainder a constant feature leaves inside a window, for a slot held over the window, and `beta` divided it by itself | the sweep of every running mean that followed | `marginal_reports_a_stopped_feature_as_it_is` in `crates/online-core/tests/held_values.rs` |
| the clusters' metric weight `1 / var` grew without bound as a quiet feature's variance decayed, and a feature that stopped left a rounding artifact in it | the same sweep, then the prototype's streams | `kmeans_assigns_the_same_at_every_level` and `micro_assigns_the_same_at_every_level` in `crates/online-core/tests/held_values.rs` |
| the window's held-feature rule compared the run's decayed weight with the window's to 1e-12, and the two drifted past it under a long window and a long half-life (1.6e-12 at a half-life of 1e6, fifty thousand rows of each) | a second reading of the windowed fixes on 2026-09-25; a simulation, then `a_long_window_under_a_long_halflife_still_reads_a_held_feature` on the old rule | that test, in `crates/online-core/src/window.rs` |
| the lasso's windowed selection fell back on whole-history errors through an empty window | the same reading | `tests/test_oracles_lasso_paths.py` |
| `corrchange`'s accurate `D̂` reached its true 0 on a collinear pair, and the numerator's rounding divided by it flagged every span of a derived column | a second reading of `corrchange`'s fix on 2026-09-25; `a_collinear_pair_has_no_verdict` | that test, in `crates/online-core/src/corrchange.rs` |
| `corrchange`'s long-run standard deviation took a span's variance as `E[x²] − E[x]²`: 1.2e-5 of itself off at a level of 1e5, NaN at 1e8, so the monitor flagged nothing there | the same sweep; a level-invariance test on deviations that are exact multiples of 2⁻²⁰ | `the_long_run_sd_is_free_of_the_columns_level`, in `crates/online-core/src/corrchange.rs` |
| `ModelBank.fit` exempted `huber` and `lasso` from the row-order warning, though their state depends on the order: with one row in ten lifted by 5, a shuffle moves `huber`'s coefficients by 1.05e-02, and shuffles moved the penalty `lasso` selects from 0.01 to 0.1 and to 0.001 | the analysis behind the README rewrite of 2026-09-29; `test_what_is_not_order_free`, which failed for both on the old list | that test, in `tests/test_order_hazards.py` |
| the bound that stands in for the noise gate's exact ratio, where a solve goes unread, would have passed a row whose share was NaN: an inverse that overflowed, with no ridge, at features near `1e-155` fitted through the origin | the test written for it, which failed before the guard | `a_share_that_is_not_a_number_withholds_whichever_way_the_gate_reads`, in `crates/online-core/src/ewridge/tests.rs` |
| a forward window's mean counted rows holding the row's own value or one its stamp excludes; it now starts at the first in-window row with a value of its own | the brute force and the time-reversal identity, both, while the operators were built | [The window operators and formula targets](../TESTING.md#the-window-operators-and-formula-targets) |
| a held value cut at a window's edge subtracted the boundary row's whole mass, which at a row exactly a window old cancelled to rounding dust: a `"left"` mean read 6.27 where the definition gave 6.34 | the definition, while the operators were built | the brute force, `windows.rs::every_operator_matches_its_definition` |
| parameters in clock units acted per row: `emit_drift` flagged a 30-unit burst at four rows a unit and not at one, the residual quantiles never forgot (1.55 where the recent quantile was 0.166), and `resid_autocorr_lag` paired residuals across a break | measuring 0.13.0, and reading the code | `tests/test_diagnostics.py` |
| `kalman`'s `coef_half_life` acted as about `h · sqrt(d)` for rows `d` apart: a slope step took 74, 38 and 13 clock units to learn at rows 1, 0.25 and 0.04 apart | measuring 0.13.0 | `crates/online-core/tests/kalman_clock.rs` |
| `label_delay`, now `embargo`, learned labels early at a break: with a 10-unit delay and a 5-unit cap, an 8-unit gap learned two labels early, and a session change with no gap learned nine | the design of formula targets, which first capped the delay at the gap cap for it; four count tests that failed on the old build | `TestBreaksCountElapsedTime` and `TestBreaksOnSkippedRows` in `tests/test_label_delay.py` |
| the findings of nine rounds of checks over the window operators and formula targets, each pinned by a test that failed before its fix | the rounds R1 to R9, on 2026-10-03 | the docstring of each pinning test, which names its finding, such as R5-C1 |

### Differences from river that are not bugs

The river work found two convention differences worth knowing about. Neither
is a bug in either library, but code that assumes they agree during warmup is
wrong.

| difference | ours | river's | entry |
|---|---|---|---|
| when FTRL predicts | recomputed from `z` at prediction time, per McMahan Algorithm 1 | `LogisticRegression` predicts with the previous step's proximal weights, one proximal step behind the paper | T-R1 |
| the EW mean | *bias-corrected*: divides by the accumulated weight, so exact from the first row | `EWMean` is un-normalized, seeded at its first value, and stays anchored near that seed during warmup | T-R4 |

## How the suite looks for defects

The references and invariants above check what the code computes. Two
methods look for what they miss. Mutation testing finds code that no test
would notice breaking, and the FFI audit finds memory faults that no
assertion sees.

### What the mutation run actually found

`scripts/mutants.sh` runs `cargo mutants` over `online-core`. It makes one
small change to the source, rebuilds, and reruns the tests. A missed mutant
means every test still passed with the code deliberately broken: a gap in the
tests rather than a bug in the code. Six passes have run, the last two over
one or two files, and the survivors left today are counted in [Mutation
survivors](../TESTING.md#mutation-survivors):

| pass | date | mutants | missed | timeouts |
|---|---|---|---|---|
| the first-ever pass (T-D4) | 2026-08-30 | 1645 | 517, with 1104 caught and 24 unviable | |
| run 1, before the follow-up work (T-D5) | 2026-08-30 | 2616 | 501 | 175 |
| run 2, after it, under load | 2026-08-30 | | 104 as reported, which was wrong | 425 |
| run 3 (`--iterate`), 14 min on an idle machine | 2026-08-30 | 690 | **217**, the last whole-crate figure recorded here | 18 |
| `stats.rs` on its own | 2026-09-24 | 285 | 48, then 5 | 0 |
| `lasso.rs` and `ewridge.rs` afresh, 3 h | 2026-09-27 | 906 | 104, then 0 not caught or recorded after two `--iterate` rounds | 152, then 0 |

The headline number (501 of 2616 mutants surviving) is less interesting than
its shape. Grouped by function, run 1's survivors were:

| function | missed | why |
|---|---|---|
| shape and state accessors (`weight_sum`, `n_targets`, `n_features`, `sigma2`, `coefficients`, `kind`) | 81 | asserted nowhere in Rust, in any model |
| `<Lasso as OnlineModel>::step` + `Lasso::solve` + `standardized` | 66 | KKT verification lives in `tests/test_oracles.py` |
| `EwRidge::blend_toward_long_run` | 54 | `session_shrink` is tested only from Python |
| `<EwRidge as OnlineModel>::step` + `solve` + `run_solve` | 54 | the slow twin, `sigma2`, and the solve schedule |
| `EwRidge::solve_standardized` | 53 | the `fit_intercept = false` branch had no test at all |
| `*Cfg::validate` (seven models) | 34 | rejections are asserted in `tests/test_edge_cases.py` |
| `EwCovModel::read` / `labels` / `n_outputs` | 26 | `ew_cov` is reachable only through the Polars layer |
| `<Holt as OnlineModel>::step` | 19 | brand new, and its tests checked outcomes not arithmetic |
| `Kalman::pred_var` + `<Kalman as OnlineModel>::step` | 17 | surfaced only as `emit_sigma` / spec options |
| `EwCov::precision` / `with_precision_prior` / `partial_corr` | 15 | `ew_cov`'s precision matrix, solved on demand and exposed only as a statistic |

#### The mutation blind spot

**One cause explains nearly all of it: `cargo mutants` runs `cargo test`, so
everything proven only by the pytest suite through the compiled extension is
invisible to it.** That is already noted in `scripts/mutants.sh` as the
reason for scoping the run to `online-core`. But the same blind spot applies
*inside* `online-core`, wherever a feature's only oracle is a Python test.

Eight commits closed it, taking `online-core`'s Rust tests from 151 to 218.
Each added an oracle in one of the four forms
[in TESTING.md](../TESTING.md#an-oracle-not-a-golden-number), rather than a golden number.

#### Five defects the behavioural tests were happy with

Five real defects surfaced in the process, all in code the behavioural tests
were happy with.

**The most serious was a zero-weight row at the head of a stream, which
permanently disabled `ewridge` and `lasso`.** Their per-target mean-form
update computes `a = lam·wj / (lam·wj + w)`, which is 0/0 when nothing has
ever carried weight. The NaN never washed out: `wj` stayed NaN, `NaN > 0.0`
is false, and the model silently stopped predicting for the rest of the
stream. Every other model already guarded it, and so did `EwCov::update` two
lines away. It was found indirectly. A Rust unit test for the analogous guard
in `blend_toward_long_run` failed, which pointed at the same shape in `step`.
`tests/test_edge_cases.py::TestWeights` now checks it for all ten regression
models.

The other four:

| defect | what it did |
|---|---|
| `sgd` and `pa` reported `weight_sum` with the current row's decay already applied | `min_weight` meant a different number of rows for them than for every other model (T-A5 has the detail) |
| `EwCovModel::n_targets` returned 1 | for a model that regresses nothing |
| `blend_toward_long_run` had lost its doc comment | to a `#[cfg(test)]` helper inserted between the comment and the function |
| `coef_prior` misconfiguration | reported "coef_prior must be 1 vectors of length 3" |

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

**Treat a large timeout count as a failed run.** Run 1 had 175 timeouts of
2616. Run 2, under load, had 425. Run 3, idle, had 18 of 690. Above a few
percent, the numbers underneath are not trustworthy.

A shorter limit makes a pass faster, but *more* sensitive to this, and
it hides survivors as timeouts. Since 2026-10-04 a mutant stops at ten times
the baseline's test run, in CI and in `scripts/mutants.sh`, after three
times hid most of a pass's survivors.

#### Where the 217 stood after run 3

Run 3's survivors by file, beside run 1's:

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
clearest case. Its 45 survivors were in `P2Quantile` and `SlotMetrics`, both
of which are loops whose mutations spin. They were the obvious next batch,
closed on 2026-09-24 by the same rule, an oracle rather than a golden number
([Mutation survivors](../TESTING.md#mutation-survivors)).

### FFI memory and crash safety (2026-08-31)

Two copies of Polars live in this process, and data crosses on the Arrow C
Data Interface. There a `SeriesExport` carries a `release` callback back into
the binary that produced it. Nothing else in the suite would notice if that
contract were broken: a leak is invisible, and a double-free is a crash that
takes pytest with it. `tests/test_ffi_memory.py` covers it: 16 tests (~7 s)
when written, and 13 since the expression plugin's removal (`6d9b983`, in
0.7.0) took its four and `test_many_tiny_groups` took the place of one
(`b9e4977`).

**The assertion is that memory plateaus, rather than that it never grows.**
Allocators do not return pages eagerly, rayon spawns workers lazily, and
Polars cached the loaded plugin while there was one. This was measured on the
plugin, before its removal. Our `.over()` took a one-time **+6 MB**
and was then flat across 1,800 iterations (per-block deltas +0.7, −0.9, +1.1,
−0.3, −0.3). Native Polars `.over()`, by contrast, slowly *returns* memory. A
naive "RSS must not grow" test would have failed on that step forever. So
`assert_plateaus` compares the later blocks against each other: a step
passes, a slope fails. When written, it discarded the first block. Since
2026-09-08 it finds the end of the ramp instead. Blocks of 120 iterations run
until two in a row each grow by less than 4 KB per iteration. Only then are
five more measured.

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
| the bank's error path; **both error paths** when written, before the plugin's was removed | `test_the_bank_error_path_still_releases` |
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
held the other's exports. That test went with the plugin.

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
| The error paths are clean. The sharper one was the plugin's: the engine had already exported the inputs when our function returned an error, so releasing them happened on a path that only runs when something has gone wrong. | The plugin was removed in 0.7.0; the bank's error path is still tested. |
| A **String feature column was silently parsed back to f64**, and a non-numeric one became all-nulls rather than raising. That was consistent with the null policy and not a memory issue. But `pytest.raises` on a String column did not fire, and a Categorical was the dtype genuinely refused. | Since `667bac1` (2026-09-01) a String column is refused by name ("has dtype str; it must be numeric"), pinned by `test_a_string_column_is_refused_not_cast_to_null` in `tests/test_error_messages.py`. |

## The entries, by ID

Priorities: **P1** = makes good a promise the library states, or covers a found defect; **P2** =
meaningful new assurance; **P3** = infrastructure. A struck-through
priority, such as ~~P1~~ **done**, marks an entry that is finished. Each
section has one table, with one row per ID, and a paragraph under it for an
entry whose detail does not fit a cell. The record of what the first CI runs
cleared closes the section.

### A. Close the oracle gaps (our own references)

| # | P | improvement | what it found or still lacks |
|---|---|---|---|
| T-A1 | ~~P1~~ **done** | **`kalman_ref` in `tests/reference.py`**: agreement to 1e-9 (observed max 1.1e-15) | three load-bearing subtleties |
| T-A2 | ~~P1~~ **done** | **Lasso KKT verification**, `tests/test_oracles.py::TestLassoOptimality`, and the *pred* path, `TestLassoPredPath` | both, since 2026-09-24 |
| T-A3 | ~~P2~~ **done** | **`robust_ref`** covers both Huber and quantile; agreement ~1e-13 | two load-bearing details |
| T-A4 | ~~P2~~ **done** | **`ftrl_ref`**; agreement ~1e-16 | when the decay is applied |
| T-A5 | ~~P2~~ **done** | null policy, warmup, clock semantics and the universal invariants over all ten regression models, `tests/test_semantics_all_models.py` | **found the robust `weight_sum` defect**, and `sgd`/`pa`'s |

**T-A1.** `kalman_ref` is a plain numpy predict/update recursion, mirroring
the standardization-from-prior-stats scheme. It agrees to 1e-9 (observed max
1.1e-15) across scalar and per-factor `coef_half_life`, `inf` pinning,
explicit `q` and fixed `obs_var`/`p0`. It also covers multi-target with and
without `share_p`, the null policy and the null-target path, and
`fit_intercept=False`. Writing it confirmed several subtleties are
load-bearing: scales come from the stats *before* the row, `Q·Δclock²` is
applied once per shared `P`, and the innovation variance carries `σ²/w`.

**T-A2.** `tests/test_oracles.py::TestLassoOptimality` checks stationarity
and the subgradient conditions at the emitted coefficient snapshot, on the
model's own standardized statistics rather than a ported solver.
`g_i = c_i − (Cb)_i − l2·b_i` must equal `l1·sign(b_i)` where `b_i ≠ 0`, and
satisfy `|g_i| ≤ l1` where it is zero. It runs across λ ∈ {0, 0.01, 0.1} ×
`l1_ratio` ∈ {1.0, 0.5}, plus sparsity monotonicity along the path and the
intercept identity. It sees one snapshot, the last solve's. So since
2026-09-24 `tests/test_oracles.py::TestLassoPredPath` also holds every row
to `reference.lasso_ref`, a cyclic coordinate descent written from the
objective (Friedman, Hastie & Tibshirani 2010). It runs from zero to 1e-14,
so no warm start can change its answer. It follows the documented schedule:
`solve_every` and its default by weight,
`max_rows_between_solves`, the forced first solve at `min_weight`, the decay,
a capped gap, skipped and zero-weight rows. It compares every path point's
`pred` and `resid`, `weight_sum`, every `coef` row, where `coef` is null, and
which coefficients the L1 zeroed. Measured: pred 1.7e-14, coef 1.2e-12,
`weight_sum` exact; at the library's own `tol` and `max_iter`, pred 1.7e-10.
Each tolerance is 100 times that. Each of seven bugs seeded into the
reference fails it: a solve a row late (pred moves 0.17-0.57), an in-sample
pred, no cap, and `>` for `>=` at the cadence. The other three are
zero-weight rows not counted as rows, skipped rows counted, and no forced
first solve. It would catch schedule and warm-start bugs the KKT check cannot
see. It also found the docstring calling `c` the feature-target correlations;
it is `cov(x_i, y) / s_i`, so the L1 threshold is in the target's units, and
the docstring now says so. The paths it leaves out are held by
`tests/test_oracles_lasso_paths.py` against `reference_paths.lasso_paths_ref`,
since 2026-09-24: null targets, several targets, `target_gaps`,
`window_size`, `fit_intercept=False` and `penalty_selected`.

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
target; `lasso` slot naming; `ftrl` probabilities). It found two defects:

| defect | what it did | what holds now |
|---|---|---|
| **the robust models reported `weight_sum` as the sum of *IRLS weights* rather than observations** | a quantile spec showed `weight_sum ≈ 1001` after three rows, since quantile weights reach `2/quantile_eps` ≈ 2000×, so `min_weight` was effectively inert | `Robust` tracks a raw-weight observation count for `weight_sum`/`min_weight`, while the accumulators keep using the robust weights |
| **`sgd` and `pa` reported `weight_sum` with the current row's decay already applied**, found as soon as the sweep reached them | every other model reports the weight before the row's update and before its decay, so `min_weight` meant a slightly different number of rows depending on the model | both follow the documented convention |

### B. Cross-checks against river

`tests/test_river.py` holds these, with river as a dev-dependency. It
imports river plainly, so a missing river fails the module rather than
skipping it ([Libraries the package does not depend on](../TESTING.md#libraries-the-package-does-not-depend-on)).
There are two tiers: **exact**, at a tolerance of ~1e-12 with the
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
rather than left as a surprise. Vowpal Wabbit's `--ftrl` recomputes its
weights after each update, as Algorithm 1 does, so the end-to-end
comparison is with it, in `tests/test_second_opinion.py`.

**T-R2.** `TestKalmanIsBayesianLinearRegression` maps `obs_var = 1/beta`
and `p0 = beta/alpha`, the prior `p0 * obs_var = 1/alpha` (since 2026-10-06), and
matches exactly, to 3.6e-15, across three (alpha,
beta) settings. It includes a guard that turning standardization on breaks
the match, so the switch cannot silently become a no-op. The class was
deleted in `509c6cf` and restored on 2026-09-23.

**T-R3.** The agreement is exact (1e-9) with no decay, reached via the new
`ew_cov` surface. The final test in that class quantified
where the two diverged: at a 1e9 offset, river's Welford form was still exact
while our raw-moment form had lost the variance entirely. That was the gap
the switch to centered co-moments closed (`509c6cf`). `EwCov` keeps centered co-moments, and the final
test is now `test_matches_river_even_on_a_large_offset`, which holds the two
to agreement at such offsets.

**T-R4.** The mapping is `fading_factor = 1 − 0.5^(1/half_life)`, and it found
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
| T-E4 | ~~P1~~ **done** | Mis-ordered chunks (clock goes backwards across a chunk boundary within a group) | both halves covered: a restart at a step back, and the refusal with `restart_after_step_back` unset |
| T-E5 | ~~P2~~ **done** | Degenerate solves in the **plain** path | finite outputs throughout, with `solve_failures` observable |
| T-E6 | ~~P2~~ **done** | Duplicate clock values, `gap_cap = 0` (refused since 2026-09-28), `half_life` far below the typical Δ | pinned; no NaN leaks under extreme decay |
| T-E7 | ~~P2~~ **done** | Minimal shapes | empty chunks, single-row groups and a one-feature/one-target spec all behave |
| T-E8 | ~~P2~~ **done** | Non-string group and session columns, null session values | a null session value **is** its own session |
| T-E9 | ~~P2~~ **done; found a defect, then removed the limit** | **Large-offset cancellation** | slope-recovery error at a 1e6 offset **2.0e-03 → 6.8e-10** |
| T-E10 | ~~P2~~ **done, decision taken** | Datetime-typed clock columns | read in their own nanoseconds, with clock parameters as durations; every duration unit, in each form, against every column unit |
| T-E11 | ~~P3~~ **done** | Long-stream soak: 10⁷ rows through one state | `weight_sum` bounded, the fit accurate, the state under 4KB; opt-in |
| T-E12 | ~~P3~~ **done** | The pending delta across a save/load boundary; session change on a group's first row | both targeted |
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
`weight_sum` was verified to accumulate across them. `GroupKey(Option<String>)`
replaces the `"<null>"` string sentinel, so a null group is structurally
distinct from a group named `"<null>"`. `TestGroupKeys` covers it, including
save/load and integer group columns. Bank files gained a `format_version`, 2
then. It is still 2 for most files, and since 2026-09-23 a bank whose specs
carry a duration writes 3. Version 1 files loaded then, because the key
serializes transparently as its inner `Option`. Today a bank file below
`MIN_BANK_SCHEMA_VERSION` (`crates/online-polars/src/bank.rs`) is refused by
its schema version, and every version 1 file is older than that.

**T-E3.** `TestNonFinite` pins ±inf/NaN in features, targets, weights and the
clock. A non-finite feature or weight skips the row, and the clock still
advances. A non-finite target is predict-only, and a non-finite or null clock
errors loudly. A fuzz-ish test adds that outputs are never non-finite.

**T-E4.** `TestClockOrdering` pins what a step back does: a backwards delta
across a chunk boundary goes through `restart_after_step_back`,
indistinguishable from real data. The test sets `restart_after_step_back=0.0`
for that, so every step back restarts. A step back no larger than
`restart_after_step_back` is refused as a late row, and with it unset, the
default, every step back is (since 2026-09-28), which catches a
mis-sorted chunk boundary loudly. It also guards that correctly ordered
chunking stays invariant.

**T-E5.** The degenerate solves are collinear and constant features in the
plain path. Exactly collinear features drive the jitter fallback (107
jittered solves over 200 rows), with finite outputs throughout. A real ridge
removes the need for jitter entirely, and non-solving models report 0. It
required surfacing the solve failures first: `Bank::solve_failures()` /
`ModelBank.solve_failures()` expose the count per spec and group.

**T-E6.** Duplicate clock values are zero deltas, so no decay.
`gap_cap = 0` disabled decay entirely until 2026-09-28, when it was
refused: no decay is `half_life = "inf"`. A half-life far below the delta
makes every row effectively the first (`weight_sum → 1`), and no NaN leaks
under extreme decay. A zero-weight row whose decay underflows forgets the
history as the decay does, in every model
(`crates/online-core/tests/model_contract.rs`).

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
0**, at an ordinary financial scale. Then the underlying cause was removed:
`EwCov` keeps centered co-moments, and the solves read
them directly. Slope-recovery error at a 1e6 offset went **2.0e-03 →
6.8e-10**. Offsets of 1e8/1e10 went from dropped entirely to 2.5e-08 /
4.8e-07. The tests pin the new range, and keep the old numbers in comments so
a regression is obvious.

**T-E10.** This was a silent trap. A temporal clock cast to f64 exposes its
internal representation: the same 60 seconds is 60e3 / 60e6 / 60e9 units for
`Datetime(ms/us/ns)`, and 1 unit per day for `Date`. So `half_life=600` on a
microsecond column silently meant 600 µs, decaying every row to nothing and
producing plausible-looking garbage with no error.

**Decision: reject (2026-08-30), then durations (2026-09-23).** A
temporal clock is read in its own integer nanoseconds and takes its clock
parameters as durations. The gap between two rows is taken in integers
before it becomes seconds, so a nanosecond timestamp keeps its nanoseconds
whatever the stream's age (2026-09-24). A plain number on it is refused,
naming the column, the parameter and both fixes (a duration, or `dt.epoch`).
`tests/test_temporal_clock.py` is the trap turned into a pass. The same
instants as `Datetime(ms/us/ns)` give a float clock's numbers to the bit.
`TestEveryUnitAgainstEveryColumn` holds every duration unit against every
temporal column kind and unit, a zone-aware one among them: 192 cases, none
skipped. Each unit is written in each form that can write it (text,
`pl.duration`, `timedelta`). Where the column can express the unit, the test
wants the exact recursion; where a cap or a threshold is finer than the
column's step, a refusal by name. Numeric clocks (int and float) are
unchanged, and are asserted to agree with each other.

**T-E11.** 10M rows go through one state in ~6.5s (2.4s for both soak tests
on 2026-10-05). `weight_sum` stays bounded and does not drift between the start
and end of the stream, the coefficients are still accurate, and resume is
still exact. A 2M-row state serializes to under 4KB: memory is O(state)
rather than O(data). It is opt-in, via `pytest -m soak`, and the weekly
leak-check workflow runs it. Until 2026-10-05 no workflow did, and the
resume test had failed unseen since a step back was first refused, on
2026-09-28: its tail restarted the clock at 0.

**T-E12.** Splitting a stream exactly after a skipped row and resuming from
state reproduces the unbroken run, so the pending delta survives the state
file. A group's first row is treated as first, even when it also changes
session.

**T-E13.** `crates/online-polars/tests/chunk_plan.rs` runs one interleaved
fixture on both sides of `PAR_MIN_ROWS`, the 4,096-row threshold from which a
chunk's columns are read in parallel. It runs at 64, 4095, 4096, 4097 and
12305 rows. The fixture has 40 groups, a null key, two one-row groups, nulls
in every input, sessions, and an ungrouped spec beside the grouped ones.

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
there at every push to `main` and pull request now, but "run CI" is not
itself a test plan. These are the specific things Windows can break that
macOS never will, and what each found. The P column is as first written. The
last column says what has cleared since CI first ran on Windows, as
[the record of the first CI runs](#what-the-first-ci-runs-cleared-2026-08-31-to-2026-09-06)
has it.

| # | P | case | since CI ran on Windows |
|---|---|---|---|
| T-W1 | ~~P1~~ **done** | **Run `ci.yml` on `windows-latest` at all** | runs on every push since 2026-08-31 |
| T-W2 | ~~P1~~ **done** | **Cross-OS state hand-off** (`release.yml`: write on macOS, load on Windows/Linux) | executed on Windows CI; the hand-off has run for every release since 0.1.0 |
| T-W3 | ~~P1 (partly)~~ **done 2026-09-24** | **Path handling through the CLI**: escaped Windows-style paths and paths with spaces | drive letters and the `\\?\` extended-length form pass on both Windows CI legs, Python 3.12 and 3.14 |
| T-W3b | ~~P1~~ **found a real bug, fixed** | TOML path escaping in the CLI tests | executed on Windows CI |
| T-W3b (original) | superseded by T-W3 (done 2026-09-24); UNC share paths stay untested | **Path handling through the CLI**, as first written: backslash separators, drive letters, UNC paths, and spaces in paths, in both the TOML `input`/`output`/`load_state`/`save_state` fields and the `--input`/`--output` overrides | |
| T-W4 | ~~P2~~ **mitigated + tested locally** | **CRLF line endings in the TOML config** | |
| T-W5 | ~~P2~~ **tested locally** | **Binary/artifact naming**: `polars-online.exe` vs `polars-online` | executed on Windows CI |
| T-W6 | ~~P2~~ **pinned locally** | **Float formatting in output field names** | |
| T-W7 | ~~P2~~ **done** | **Numeric reproducibility across OS/CPU** | compared at 1e-12 on Windows |
| T-W8 | P3, **partly done**: file locking (the atomic rewrite in `atomic.rs`) and long paths (`\\?\`) are tested; case-insensitivity is not | **Filesystem behavior**: case-insensitivity, `MAX_PATH`, file locking on rewrite | executed on Windows CI |
| T-W9 | ~~P3~~ **done** | **`scripts/env.sh` has no Windows equivalent** | |

**T-W1.** Written when `cargo test --workspace`, `maturin develop` and the
pytest suite had never executed on Windows. They have run there since
2026-08-31, at every push to `main` and pull request.

**The first Windows run found nine failures, all in the tests and none in
the library.** After T-W3b's TOML fix cleared the Rust side, the pytest
suite failed nine ways on Windows. The failures fell in four classes, every
one a test making a Unix assumption:

| class | what failed | the fix |
|---|---|---|
| **(1) encoding** | `Path.read_text()` and `subprocess(text=True)` default to the locale codec, which is cp1252 on Windows. It cannot decode the box-drawing characters polars prints or the arrows in our own README (`UnicodeDecodeError: 'charmap' codec ... byte 0x90`) | explicit `encoding="utf-8"`, suite-wide in 7 files rather than only at the reported sites |
| **(2) path separators** | `test_a_clean_checkout_has_what_the_build_needs` compared `str(WindowsPath(...))` (backslashes) against git's forward-slash output, and declared a tracked file missing | `as_posix()` |
| **(3) environment** | the thread-determinism subprocess passed a hardcoded `PATH=/usr/bin:/bin`, leaving the child with no resolvable interpreter | it inherits `os.environ` |
| **(4) shell** | `test_release_packaging` executes a `run:` block that only ever runs on the workflow's ubuntu job, and Git Bash's `find` differs enough to fail for reasons that say nothing about the workflow | skipped on Windows, where Linux and macOS still cover it |

The library passed 709 of the 718 tests on that first run, and none of the
nine failures was its own (the message of `fbc6f8f`, which fixed them). That
is the result worth recording. A later run the same day, on `d6158aa`, passed
712 and failed one
([the record](#what-the-first-ci-runs-cleared-2026-08-31-to-2026-09-06)).

**T-W2.** Class 7 of [the eight test classes](../TESTING.md#the-eight-test-classes), and hard rule 5. As first written: the msgpack
payload has no host-dependent parts *by construction*, and `save_bytes` is
asserted deterministic locally, but that is an argument rather than a test.

**T-W3.** Escaped Windows-style paths and paths with spaces are tested
through the CLI on any OS, and round-trip. Resolution on Windows runs on the
Windows CI leg, which every push to `main` has had since 2026-08-31. There,
the three path forms of `test_each_documented_windows_path_form_parses` and
`test_absolute_paths_in_every_field_and_flag` write the runner's real
`C:\...` temporary paths, so drive letters resolve on a real Windows host.
`test_an_extended_length_path_resolves_on_windows` (Windows only) repeats
the same run with the UNC-style `\\?\C:\...` spelling. It does so in the
config's `input`, `output` and `save_state`, and in the `--input`,
`--output`, `--save-state` and `--load-state` flags. Polars strips the prefix
(`normalize_windows_path` in polars-utils). A network share,
`\\server\share\...`, is out of a runner's reach and stays untested. The
Windows-only test first ran on the push of `7d9a1c0` (2026-09-24), and passed
on both Windows legs, Python 3.12 and 3.14.

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
called plain `polars-online`. It then runs the workflow's own bash against it. The
shell is *extracted from `release.yml`* rather than copied, so editing the
workflow changes what the test runs. That was verified by breaking the
workflow and watching three tests fail. Pinned: exactly the Windows artifact
keeps `.exe`, the two unix binaries do not collapse onto one name, and
contents are copied as well as renamed. The PyPI job's `dist/` takes the
packages, the wheels and the sdist, and no CLI binary. What still needed a
runner, as first written, is whether the Windows job produces `polars-online.exe`
in the first place.

**T-W6.** The exact field-name list for a grid spec is asserted, so a
platform divergence fails loudly: 58 fields, and 52 when written. The
readiness fields (`3e0b359`, 0.9.0) added `settled_frac`, `withheld_reason` and `support_coef` at
each of the grid's two half-lives. The list embeds formatted floats. Combo
labels are built with `format!("{r}")` on f64 (e.g. `pred_y__r0.000001`).
Rust's float `Display` is locale-independent, so this *should* be identical
everywhere. But the field names are part of the public schema: users index
the output struct by them. So a divergence would silently break every caller
reading a field by name. Hence: assert the exact field-name list on every OS.

**T-W7.** `tests/test_golden_pipeline.py` commits 505 outputs from a fixed
stream through a 25-spec bank. Of those, 157 are read at each of three rows:
25, 60 and 119. The other 34 are not per row. They are `marginal`'s 20 pair
statistics, read at the end, `corrchange`'s 4 statistics, and `rcov`'s 10
closed-group values. When written it committed 135 outputs from an
eleven-spec bank at three rows each. It held 126 until it took in `ftrl`,
the one model it had never pinned, and 502 on 2026-09-26. It compares
with a 1e-12 relative tolerance. `golden.rs` already pinned the Rust core;
this pins the *Polars* layer, which nothing did: extraction, per-group
fan-out, diagnostics, struct assembly. Locally the agreement is exact. The
tolerance is what "the same answer on another platform" is allowed to mean.
Different LLVM vectorization and BLAS paths can reorder floating-point
operations, while a genuinely divergent algorithm shows up far above it. It
was verified to bite at 1e-11 and to tolerate 1e-14. The comparison became a
cross-platform one the first time CI ran, with no further work.

**T-W8**, as first written. The bank writes state with `std::fs::write` and
the runner opens the output parquet with `File::create`. A still-open reader
on Windows makes rewriting fail where POSIX allows it. That is relevant to
`--load-state` loops. Both writes have since moved to a temporary file and a
rename (`crates/online-polars/src/atomic.rs`).

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
| T-D1 | ~~P1~~ **done** (2026-08-31) | **Actually run the workflows once.** | the workflows have run; [the record](#what-the-first-ci-runs-cleared-2026-08-31-to-2026-09-06) has the results |
| T-D2 | ~~P2~~ **done** | **Property-based testing** (hypothesis; proptest in Rust since 2026-09-25) | `tests/test_properties.py`; `crates/online-core/tests/model_contract.rs`, module `generated` |
| T-D3 | ~~P2~~ **done** | Determinism across parallelism | `tests/test_portability.py` |
| T-D4 | ~~P3~~ **done** | Coverage, and **mutation testing** | coverage reported, not gating; mutation testing in CI since 2026-09-24 (`mutants.yml`) |
| T-D5 | | Mutation re-run | **done, three passes** |

**T-D1.** Until the workflows ran, T-W1/T-W2 and the wheel builds were
untested claims. It *was blocked on GitHub credentials on this machine*: no
keychain entry for github.com, no SSH key, no `GH_TOKEN`, and `gh` was not
installed, so `git push` could not authenticate. The unblock was any of
`gh auth login` / an SSH key / a PAT, and the token needed the **`workflow`
scope**, since this push added `.github/workflows/`. The repo has been pushed
since 2026-08-31, and CI runs on Windows at every push to `main` and pull
request. The first runs' results are in
[the record](#what-the-first-ci-runs-cleared-2026-08-31-to-2026-09-06), and
the Windows failures in T-W1.

**T-D2.** `tests/test_properties.py` uses hypothesis to generate adversarial
streams: mixed nulls, duplicate/long-gap clocks, ±1e8 values, zero weights,
tiny groups. It asserts the universal invariants for all ten regression
models. They are chunk invariance under any chunk size, save/load
transparency at any split, outputs finite-or-null, no field at all
reported by a skipped row, and group independence. The strongest is that **changing a
row's own target never changes that row's own prediction**: out-of-sample by
construction, hard rule 2.

**T-D2, extended 2026-09-24.** `tests/test_properties_temporal.py` adds 13
property tests, 41 cases on 2026-10-03. They cover duration text: round trip over the whole
i64 range, polars' own parser as the oracle, overflow, padding and inner
spaces, the three forms. They cover temporal clocks too: chunk invariance,
save and load at any row, nanosecond exactness years into a stream, a
delayed label. They found four bugs, all fixed. Duration text accepted a
space after the sign, a `pl.duration` past 292 years wrapped, and a
`timedelta` that long was refused without a name. And `embargo` broke chunk
invariance in `settled_frac`'s last bit.

**T-D2 in Rust, 2026-09-25.** `model_contract.rs`'s module
`generated` holds every kind (21 then, `audit` the 22nd since 2026-10-08) to three clauses of the contract, over streams
`proptest` generates and shrinks. First, `predict_with` is the step without the
update. Second, a state saved and restored at any row continues exactly as
the model that was not. Third, nothing is infinite, with values up to `1e50`,
repeats, absent targets, zero and uneven weights, and clock gaps. The suite
runs 128 streams a model; a run of 2,000 a model found nothing.

**T-D3.** `tests/test_portability.py` runs the bank in subprocesses at
`POLARS_ONLINE_MAX_THREADS=1` and `=8`, and requires identical output, every
field compared exactly. Since 2026-09-04 it runs on both sides of
`PAR_MIN_ROWS` ([T-E13](#c-edge-case-matrix)): 400 rows over 6 groups, and
5000 over 37, which exercises the parallel extract and assembly. Both use a
half-life grid with every optional output on.

**T-D4.** Coverage: `scripts/coverage.sh` reported 96% Python and
93.9%/92.6% Rust on 2026-09-27, and 75%/73% Rust when this entry was written
([the caveat](../TESTING.md#measured-coverage)). CI reports the Python figure, never
gating. **Mutation testing** (`scripts/mutants.sh`) was first run in full
over `online-core` on 2026-08-30: the first row of the
[table of passes](#what-the-mutation-run-actually-found). The misses
concentrated exactly where the Rust unit tests lean on the *Python* oracle
suite, which `cargo test` cannot see: `robust.rs` 68% missed, `kalman.rs`
38%, `ewridge.rs` 36%. The fix was `crates/online-core/tests/golden.rs`: one
fixed 60-row stream per model with the exact expected predictions embedded,
which pins the arithmetic against any mutation. Measured on the worst file,
**`robust.rs` went from 162 missed / 77 caught to 42 / 197**, a 74%
reduction from one test. The residue is mostly accessors (`n_features -> 0`)
and validation-branch comparisons, which are low value. T-D5 re-ran the full
pass.

**In CI since 2026-09-24** (`.github/workflows/mutants.yml`). Every push to
`main` and pull request runs cargo-mutants over the lines it changed. It
fails on a survivor that `scripts/mutants_equivalent.toml` does not list:
new code should come with a test that would notice it breaking.

**The changed lines' pass tests every mutant it lists, however large the
push.** The repository is public, so GitHub's runners cost it
nothing (the user, 2026-10-08: "We don't pay for minutes being open
source"). A first job lists the change's mutants, which builds nothing.
`scripts/mutants_shards.py` then takes a shard for every thirty (forty
until 2026-10-09), at least one and at most 256, GitHub's limit on a matrix. Ten fixed shards, from
2026-10-07, could not hold a large push. The push of `103d721` listed 1,346
mutants. Each shard stopped at its 100 minutes having tested 51 to 99 of
its 135, and 698 went untested. At that run's slowest rate, 114 s a
mutant, forty take 79 of the 100 minutes. The account runs 20 jobs at a
time, CI's among them, so a pass of more shards runs in waves. That push's
34 would take two, about two and a half hours. Each push to `main` has a
concurrency group of its own, by its commit, so no later push cancels its
pass or replaces it while it waits: the next pass lists only its own diff. On
2026-10-09 the share became thirty and each shard's stop 160 minutes (the
job's limit 180): the push of `f54d1e5`'s slowest shard of forty took 83 of
its 100 minutes, and the Rust suite every mutant runs had grown from 146 s
to 209 s (the user: "Yes mutants sharding"). On 2026-10-11 each shard's stop
became 300 minutes inside a limit of 330, three times the push of
`5cb8f7e`'s slowest shard (98 minutes), and the weekly shard's 325 inside
350.

**A mutant cannot take its runner down.** Each shard's whole `cargo
mutants` run is inside one transient systemd scope whose cgroup holds at
most 12 GiB of a runner's 16, with no swap (`scripts/memory_scope.sh`, the
workflow's `MUTANTS_MEMORY_MAX`). A mutant whose loop keeps allocating makes
its test the scope's largest process, the kernel's OOM killer kills it,
nextest counts the test failed, and the mutant is caught. The scope's
`OOMPolicy=continue` keeps cargo-mutants running, and the runner's own
agent, outside the scope, keeps the rest of the memory. Before any cap, the
push of `5cb8f7e` lost 8 of its 111 shards to "The runner has received a
shutdown signal", every one of them dealt `audit.rs`'s `Moments` mutants,
and their outcomes never reached the report.

**The first cap capped the wrong number.** It ran each test binary under
`prlimit --data`, 6 GiB for the changed lines' two mutants at a time and 3
GiB for the weekly pass's four, sized from a resident peak of 497 MiB for
online-core's largest test binary, measured on macOS. `RLIMIT_DATA` counts
a process's private writable virtual memory, which glibc keeps after a free
in its arenas and retained heap, not what is resident. On Linux the suite
crossed 6 GiB of it with far less resident: the baseline failed in 4 of the
32 shards of run 38205650365 (`memory allocation of 545259520 bytes
failed`), depending on which tests shared a process. The scope's `MemoryMax`
counts resident memory, page cache included, which the kernel reclaims
before it kills anything. Every run of the workflow proves it on its own
runner in the `memory-cap` job (`scripts/memory_scope_check.py`): a process
allocating without end inside a scope of the shards' cap is killed within
the cap's last quarter, and the process that started it goes on.

**The scope carries the job's whole environment, and the build copies sit
on the runner's disk.** The scope is made with `sudo`, which resets the
environment, and ubuntu-26.04's sudo-rs ignores `sudo -E`; so
`memory_scope.sh` writes the caller's environment to a file and exports it
again inside, and the `memory-cap` job checks a variable set outside
arrives. cargo-mutants copies the tree into `TMPDIR` (else `/tmp`), once a
job, and builds each copy cold. On ubuntu-26.04 `/tmp` is a RAM-backed
tmpfs with a per-user quota, charged to the scope's memory: the push of
`85da91e` lost 23 of its 24 shards to `Disk quota exceeded (os error 122)`,
each after one to six mutants, while the root disk had 109 GB free. Each
step now sets `TMPDIR` to a directory under the job's `RUNNER_TEMP`, on that
disk, and prints the disk before the run and the most it used. A shard's
use is one build copy a job whatever the diff's size: a larger diff makes
more shards, not larger ones. A pass that failed this way cannot be rerun
under the fixed workflow, since a rerun keeps its run's own workflow file,
and the next push lists only its own diff; so a dispatch given a `base`
(`gh workflow run mutants.yml --ref main -f base=<commit>`) runs the
changed-lines pass over the diff from that commit to the ref's head, and a
dispatch without one runs the weekly pass.

**Each mutant's tests run under cargo-nextest**, a process per test, as the
gate's do: the full Rust suite took 84 s under nextest against `cargo
test`'s 182 s on the development machine.
A pull request's runs share one group, and a newer commit cancels the
older, since it lists the whole pull request's diff again.

The whole of `online-core`, 11,138 mutants on 2026-10-03, runs weekly in
ninety-six shards, sixteen at a time. The weekly pass runs on its schedule only while
the repository is public, and by hand. Its survivors are reported through
`scripts/mutants_report.py` and never gate.

**Every run is sized to finish, and says when it did not.** It
skips the doctests, which took 39 of the 67 seconds a test run of
`online-core` cost, and stops a mutant at ten times the baseline's test
run. It stops itself inside its job's limit and lists the mutants it was
given, so the report fails on a run that tested fewer, or a shard that sent
nothing.

**A survivor needs room in the time limit.** The baseline runs alone, but a
mutant shares the runner's cores with the other jobs, and a survivor runs
the whole suite. Nineteen survivors took 1.7 to 6.3 times the solo baseline
at fourteen jobs on fourteen cores. The first full pass, at three times,
reported 11 survivors and 1,588 timeouts. Of 25 of those timeouts rerun
with room, 19 survived, 4 were caught and 2 still hung.

The changed-lines scope is `online-core` and `online-polars/src/span.rs`,
chosen by measurement. cargo-mutants runs `cargo test` only, and the pytest
suite, which covers most of `online-polars` through the extension, is
invisible to it. On `span.rs`, whose duration parser its own Rust tests pin,
55 of the 60 viable mutants were caught. Of the 5 survivors, 4 were real
gaps, now closed by Rust tests (the longest duration that parses, a span's
`Display`, and the two deserialisers' messages), and 1 is equivalent. On a
one-in-ten sample of `spec.rs`, 16 of 39 viable mutants survived, nearly all
of them validation that pytest covers, so it stays out of scope.

`scripts/mutants_equivalent.toml` names each mutant no test can catch by
file, function, mutation and code on its line. So an entry follows its line
when code above it moves, and lapses when the line changes;
`tests/test_mutants_report.py` checks every entry still finds its line.

**2026-09-02:** `golden.rs` had signatures for seven of the eleven kinds.
`sgd`, `pa`, `holt` and `ew_cov`, the four with longhand-recursion oracles in
their own modules rather than numpy references, now have one each, and `sgd`
two. `test_model_registry::test_the_core_golden_file_pins_every_model` holds
the file to every kind the extension reports, `_native.model_kinds()`. Each
new pin was mutated once
(the Huber clamp, the PA-II damping, the trend smoothing, the variance
floor), and each caught its own. The same day,
`test_every_builder_has_a_per_model_test_file` gave the per-model test file
its check, EXTENDING step 11 (step 13 when written). It also gave
`ewridge`/`rls` their own `test_<model>.py`, out of `test_bank.py`.

**T-D5.** Three passes on 2026-08-30, all in the
[table of passes](#what-the-mutation-run-actually-found). Run 2's 104 missed
was wrong ([why](#do-not-run-anything-else-while-a-mutation-pass-is-going)).
Run 3's 217 is the last whole-crate figure recorded here
([Mutation survivors](../TESTING.md#mutation-survivors)). It came despite the crate
having grown by 60% since the first-ever pass. The misses were not
scattered. They clustered almost perfectly on the code
whose *only* tests live in `tests/*.py`, because `cargo mutants` runs
`cargo test` and cannot see the Python suite. Eight commits of Rust-side
oracles followed; see
[What the mutation run actually found](#what-the-mutation-run-actually-found).

### What the first CI runs cleared (2026-08-31 to 2026-09-06)

This is the record of the first CI runs. Its last paragraph adds the newest
release.

**The original blocker is gone.** It was that *nothing has ever been pushed,
no CI job has ever run on any platform*. As of 2026-08-31 the repo is pushed,
and CI has run on all three platforms. Cleared:

| entry | what cleared it |
|---|---|
| **T-D1** | The workflows have run. What they claimed is now measured, and three of the claims were wrong: see below the table. |
| **T-W1** | `cargo test`, `maturin develop` and pytest have all executed on Windows: **712 passed, 1 failed** on `d6158aa`, down from 9 failures. Every one of the nine was a test bug, not a library bug: T-W1 in [D](#d-windows-and-cross-platform) has the four classes. |
| **T-W7** | The 126 committed golden pipeline outputs compared at 1e-12 on Windows, so the golden comparison is genuinely cross-platform now. |
| **T-W5**, **T-W3b**, **T-W8**, **T-W2** | All executed as part of that run. |

The three claims that proved wrong were the Linux `ld` SIGBUS (a full disk
rather than memory), the cache that never saved, and the disk exhaustion *inside* the
cache restore. The SIGBUS is told in the comment above the `test` job in
`.github/workflows/ci.yml`; the other two are in `docs/RELEASE-READINESS.md`.
`d6158aa` is not a commit in this repository's history, perhaps a CI merge
commit.

**Since cleared, as recorded on 2026-09-06.** The tenth Windows failure was a
`UnicodeEncodeError` in `examples/pathway_integration.py`. It was fixed and
pinned by a test. Windows had been off the push matrix while the repo was
private (the COST POLICY comment in `ci.yml`). The repo is public, every push
runs all three OSes, and Windows has been green on every release since 0.1.0.

**PyPI.** `polars-online` is published there, 0.1.0 on 2026-09-03 and 0.1.1
on 2026-09-04, through the trusted-publisher `release.yml`. The Polars pin
question is settled as `polars>=1.37.0,<3`: see "The Polars pin" in
`docs/RELEASE-READINESS.md`. Every release since, with its date, is a
section of `CHANGELOG.md`. A release is dispatched on `main`, and the workflow
tags only what it published, after everything ran
(`tests/test_release_workflow.py`).
