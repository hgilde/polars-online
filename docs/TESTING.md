<!-- kind: reference -->
# The test suite

This page is the reference, for a contributor changing the library, on
what the test suite proves and which tests hold each guarantee. [How the library is
tested](explanation/how-the-library-is-tested.md) explains the same
guarantees for a user deciding whether to trust the numbers.

**The suite has 2,653 Rust tests and 7,211 pytest cases, counted on
2026-10-24.** The Rust count is the lines ending in `: test` that
`cargo test --workspace --exclude online-py -- --list` prints, the
extended tests and the doctests included; the pytest count is what
`pytest --collect-only -q` reports collected, the soak tests aside. Each
model is held to a reference it cannot share a bug with. The bank is held
to the invariants it promises: the same numbers whether a stream arrives in
one chunk or many, runs on one thread or eight, or is saved and resumed
part-way. The suite runs on three
operating systems at every push to `main` and every pull request
([Where the suite runs](#where-the-suite-runs)). The coverage figures are
from 2026-09-27 ([Measured coverage](#measured-coverage)).

**The 45 opt-in soak tests are outside every count.** They are the 10M-row soak's two, and eight long-run stability cases at
1e8 rows each (`tests/test_long_run_soak.py`, about ten minutes). Beside
them sit the nine cells of the robust models' contamination sweep
(`tests/test_convergence_soak.py`), and the 26 worst-seed grids of the clustering and changepoint models' own
parameters (`tests/test_family_f_grids.py`, about 15 seconds, beside five
cheap cases in the essentials). They run with `pytest -m soak`, and the
weekly leak-check workflow runs them.

| section | what it holds |
|---|---|
| [Terminology](#terminology) | the words this page uses |
| [What the suite proves](#what-the-suite-proves) | [the eight test classes](#the-eight-test-classes) · [against reference implementations](#against-reference-implementations) · [an oracle, not a golden number](#an-oracle-not-a-golden-number) · [libraries the package does not depend on](#libraries-the-package-does-not-depend-on) · [the window operators and formula targets](#the-window-operators-and-formula-targets) · [resuming a run](#resuming-a-run) · [what the README's guarantees are held to](#what-the-readmes-guarantees-are-held-to) |
| [How the suite runs](#how-the-suite-runs) | [two tiers](#two-tiers) · [iterating](#iterating) · [worktree slots](#worktree-slots) · [caches](#caches-and-when-to-clear-them) · [where the suite runs](#where-the-suite-runs) · [downloaded data in CI](#downloaded-data-in-ci) · [the machines and the CPU paths](#the-machines-and-the-cpu-paths) |
| [Numbers that only tighten](#numbers-that-only-tighten) | [the oracle ledger](#the-oracle-ledger) · [budgets that only tighten](#budgets-that-only-tighten) |
| [Where it is thin, and what is left](#where-it-is-thin-and-what-is-left) | [what is left](#what-is-left) · [measured coverage](#measured-coverage) · [mutation survivors](#mutation-survivors) |

## Terminology

**Each word below has one meaning on this page,** and the
[glossary](GLOSSARY.md) defines the library's own words, such as bank,
spec and clock:

| word | what it means here |
|---|---|
| oracle | a reference a model cannot share a bug with; its four forms are in [An oracle, not a golden number](#an-oracle-not-a-golden-number) |
| golden number | an exact expected output from a fixed stream, embedded in a test. It pins the arithmetic against any change, but it is not an oracle |
| second opinion | another library's computation of the same quantity, held beside the model's in `tests/test_second_opinion.py` (the diagnostics' in `tests/test_second_opinion_diagnostics.py`, split from it when it grew past 250 KB) or `tests/test_river.py` |
| mutant | one small change `cargo mutants` makes to the source, such as a flipped operator or a function body replaced with a constant, before it reruns the tests. A mutant is *caught* when some test fails, and *missed*, a survivor, when every test still passes (`scripts/mutants.sh`) |
| equivalent mutant | a mutant that no test can kill: on every input the code can receive, the mutated line computes what the original does |
| tolerated mutant | a mutant that differs from the original only at an exact tie or by less than the computation's own error, so no test a correct implementation passes can kill it; listed apart, not as equivalent ([Mutation survivors](#mutation-survivors)) |
| IC | the correlation of prediction with target, `po.eval`'s `ic` |
| hard rule N | the numbered hard rules in `CLAUDE.md` |
| a model kind | one of the 22 model types a spec can name; `MINIMAL` in `tests/test_model_registry.py` holds one spec of each |
| the ten regression models | the models the per-model sweeps run: `ewridge`, `rls`, `lasso`, `kalman`, `huber`, `quantile`, `sgd`, `pa`, `ftrl` and `holt`, which `REGRESSIONS` in `tests/test_model_registry.py` lists |
| the window operators | exponentially weighted means, variances, sums, rates and quantiles of a column along the clock, with or without a hard window, which `po.stream.with_windows` runs over a stream: `po.ewm_mean`, `po.ewm_sum`, `po.ewm_rate`, `po.ewm_quantile`, `po.ewm_var` and `po.ewm_std` look back, and the first four look ahead under `reverse=True` |
| a formula target | a spec's target written as an expression over a window operator that looks ahead, such as `po.ewm_mean("mid", ...) - pl.col("mid")`, which the bank resolves once the window has closed |
| the column form | the same window expression written as a column by `with_windows(like=spec)` and fed back as a plain target under the same `embargo`: what a formula target is held to |
| the brute force | `brute()` in `crates/online-polars/src/windows.rs`, which computes each row's window from the definition by scanning every row of its stretch, with no running sums |
| the time-reversal identity | a forward window over a stream is a backward one over the same stream reversed in time, with the row itself left out, so a forward `closed="right"` is a backward `"left"` |
| the mean form | running sums kept as weighted means, such as `S = Σ w zz' / W`, rather than weighted sums, so a factor common to every weight cancels |
| the slow twin | `ewridge`'s second set of running sums, at `long_half_life`, which `session_shrink` blends toward on a session change |
| the pending delta | the clock step of a skipped row, held and folded into the next accepted row's decay |
| the test gate | `scripts/gate.sh`, which runs the lints and the tests of one tier; the essentials test gate is its default tier, and `--extended` runs the full suite |
| the lints | the test gate's checks other than its tests: `cargo fmt`, `clippy -D warnings`, `uv lock --check`, `ruff`, `mypy` and Sphinx with `-W` |
| a worktree slot | a git worktree kept between tasks under `.claude/slots/<n>`, with its own builds, which a task takes with `scripts/slot.sh take` and gives back once its branch is merged; not the glossary's *slot*, a prediction a spec writes per row |
| a wave | the finished branches merged together onto `main` behind one essentials test gate; a wave of one branch skips it, and runs only the full suite before its push |

## What the suite proves

**The suite holds each model to a reference it cannot share a bug with, and
the bank to the invariants it promises.** The design set out eight classes
of test, from agreement with an oracle to a benchmark. The window operators
and resumed runs came after the classes, and are held the same way. Other
tests hold the library at its edges, and hold the files the repository
promises, such as the API snapshot and the examples. The essentials gate
each commit, and everything runs before a push and in CI on three operating
systems.

### The eight test classes

**Each of the eight classes holds, apart from the sixth, which the Arrow
path replaces.** Each class is a property every model, or the bank,
must have:

| class | what holds it |
|---|---|
| 1. Oracle agreement | each model against a reference it cannot share a bug with, in [Against reference implementations](#against-reference-implementations); each window operator against its definition, Polars and the time-reversal identity ([The window operators and formula targets](#the-window-operators-and-formula-targets)), and the lasso's *pred* path against a numpy `lasso_ref` |
| 2. Chunk invariance | bitwise at the bank (1/7/100 chunks) and command-line (`chunk_size` sweep) levels, in the window core at chunks of 1, 2, 3, 7, 50 and 499 rows, and for formula targets at 1, 7 and 600 chunks; save/load mid-stream identical. Which rows carry `coef`, and `support_coef` and `se_coef` beside it, is excluded: a reporting cadence, chunk-dependent by design |
| 3. Out-of-sample by construction | IC ≈ 0 on pure-noise targets asserted for ewridge, kalman, huber, ftrl; lasso selection prefers the all-zero penalty on noise; robust reweighting proven to use the *prior* residual |
| 4. Clock semantics | cap, a step back (refused, or a reset past `restart_after_step_back`), session gap and reset, first row, row-count clock, skipped-row decay folding, per-group independence |
| 5. Null policy & warm-up | feature/target/weight nulls and `min_weight`, for all ten regression models |
| 6. Arrow ≡ Polars output, retired and replaced | the Arrow path, after the table |
| 6b. `predict` ≡ `fit_predict` of the next row | `tests/test_predict.py`; `crates/online-core/tests/model_contract.rs` |
| 6c. Runner ≡ bank, every source and format | `crates/online-polars/tests/runner.rs`; `run_online` in `tests/conftest.py`; `tests/test_bank_ergonomics.py` |
| 7. Cross-platform state | the macOS→Windows/Linux artifact hand-off in `release.yml`, run at every release since 0.1.0 and at no push: a bank of every kind, with a `Datetime` clock, a window, an embargo, a closed group and a formula target beside them, and a `refresh_time` and a `with_windows` state, each loaded, continued and re-saved to its own bytes (`crates/online-polars/tests/state_portability.rs`). At every push `ci.yml` saves and loads states on each of the three OSes, each reading its own |
| 7a. Frozen states | one state of every `ModelState` variant mid-stream, with what it keeps (a window, lags, bins, a warm-up buffer, far rows, the clusterers' held start), and a bank file, a `with_windows` state and a `refresh_time` state, embedded as bytes with the rows that follow and what they reported; each loads, goes on to the bit on the platform that wrote it (to the golden tolerance elsewhere, libm's last bit) and saves its bytes again (`crates/online-core/tests/state_fixtures.rs`, `crates/online-polars/tests/state_fixtures.rs`). Every schema from the minimum to `SCHEMA_VERSION` must have its set, so a layout change cannot land without one; `PRINT_STATE_FIXTURES=1` regenerates them |
| 8. Benchmark | `scripts/benchmark.py`, numbers in README; `benchmark.yml` reports them on every push to `main` that can move them, never gating |

**6. Arrow ≡ Polars output: retired, and replaced.** The class compared a
form of the library that no longer exists, so the Arrow path holds its
place. `tests/test_arrow_capsule.py` holds
`fit_predict_arrow`'s struct to `fit_predict`'s output column, field for
field and null for null. The nested `coef` list is included, and nulls are
compared rather than skipped. `crates/online-polars/tests/arrow_chunk.rs`
feeds a chunk built by hand from Arrow arrays, with no frame anywhere, and
holds it to the same output.

**6b. `predict` ≡ `fit_predict` of the next row.**
`tests/test_predict.py` holds row `i` of `predict(df)` to row 0 of
`fit_predict(df.slice(i, 1))` on a fresh clone, field for field. It does so
for all ten regression models with every diagnostic on, and across every
session and clock policy. `crates/online-core/tests/model_contract.rs` holds
each model's `predict` to its `step`, row by row.

**6c. Runner ≡ bank, every source and format.**
`crates/online-polars/tests/runner.rs` holds every input format (parquet,
ipc, csv, ndjson), every output format, `Input::Batches` ≡ a query, and the
failure paths to the numbers `ModelBank` gives on the same rows. CSV's
flattened columns and JSON-text `coef` decode bit-exact. The `polars-online` command
line is held to the same numbers through `run_online` in `tests/conftest.py`
(`test_closed_groups`, `test_hardening`, `test_predict`, `test_no_output`,
`test_label_delay`), and every `sh` block of `docs/RUNNER.md` runs. What a
`ModelBank` takes is held
in `tests/test_bank_ergonomics.py`: `fit`, `fit_predict_batches`, over a
query, a frame or an iterator, and `chunk_size`. A format is bound in the
tests, never in the API.

### Against reference implementations

**Each model is held to a reference it cannot share a bug with, edge cases
included.** The numpy references in `tests/reference.py` replay each model's
documented recursion, with clock semantics, null policy and warm-up. Those in
`tests/reference_paths.py` recompute every statistic from the raw rows at
each solve, so they share neither the core's recursions nor its schedule. The
lasso is also checked against its own optimality conditions. Wherever another
library computes the same quantity, that library is a second opinion: river
in `tests/test_river.py`, and the rest
in `tests/test_second_opinion.py`, Vowpal Wabbit among them.

| model | held against | agreement |
|---|---|---|
| `ewridge`, `rls` | `tests/reference.py`, incl. multi-target, standardize, `lam` decay, row-count clock | 1e-9 |
| `ewridge` | every solve against scikit-learn's `Ridge` (`TestEwRidgeIsSklearnsRidge`, 20 cases), and in Rust against its closed form solved by `faer` (`every_solve_is_its_closed_form_across_targets_and_ridges`) | 1e-8, relative; 1e-9 |
| `ewridge`'s grids, windows, sessions and `session_shrink`, schedule, no intercept | `reference_paths.ewridge_paths_ref` (`tests/test_oracles_ewridge_paths.py`), recomputing every statistic from the raw rows at each solve, as `lasso_paths_ref` does | pred 1.1e-14 |
| `rls` | `rls ≡ ewridge(ridge_scale="sum", solve_every=1)` | <1e-9 |
| `rls` with several targets, `coef_prior`, no intercept; `kalman` with several targets and nulls | `rls_paths_ref`; `kalman_ref` (`tests/test_oracles_rls_kalman_paths.py`) | pred 5.8e-15; 6.1e-15 |
| Kalman | `kalman_ref`, across every configuration | ~1e-15 |
| `kalman` | filterpy's `KalmanFilter`, across a zero-weight row (`TestKalmanZeroWeightRow`) and with a mean-reverting transition (`TestAMeanRevertingKalmanIsFilterpy`) | 1e-9 |
| `kalman` standardized, the default | filterpy's `KalmanFilter` fed the rows standardized in numpy from the documented pre-row moments (`TestAStandardizedKalmanIsFilterpy`), and `coef[t]·x[t+1] = pred[t+1]` asserted directly | 1e-9; 1e-12 |
| Kalman(q=0, fixed `obs_var`, `standardize=False`) | `river.linear_model.BayesianLinearRegression` | 3.6e-15 |
| the lasso | its KKT conditions, rather than a ported solver; and `lasso_ref`, a coordinate descent from zero on the documented schedule, for every row's *pred* | the conditions hold; pred ~1e-14 |
| the lasso's targets, `target_gaps`, window, selection, no intercept | `reference_paths.lasso_paths_ref` (`tests/test_oracles_lasso_paths.py`): every statistic recomputed from the raw rows at each solve, so independent of the core's recursions; `penalty_selected` under a window and with a `min_weight` list, and a window down to one row of a target | pred 4.5e-14 |
| Huber, quantile | `robust_ref` | ~1e-13 |
| `huber` | scikit-learn's `LinearRegression` at `huber_delta = 1e9`, where no row is down-weighted; `HuberRegressor`, a batch fit of the same objective, with one row in fifty a gross error, and one in ten | 1e-9; 0.03 (0.0016 measured, at most 0.011 over eight seeds, at the default `huber_delta = 1.345` and `huber_scale = "mad"` against scikit-learn's 1.35), where least squares is 0.38 to 0.45 off; 0.15 at one in ten (0.031 to 0.092), where the RMS scale is 0.84 to 1.0 off |
| `huber`'s MAD scale | statsmodels' `RLM(HuberT(1.345))` with `scale_est=mad`, iterated to 1e-10, on a stream at a level of 1,000 with no decay and one-sided outliers on 1% and 5% of the rows (`TestHuberMadAgainstStatsmodels`); `robust_ref` with the sketch's buckets written from their definition (`EwMedian`), both scales, a scale memory of its own | 0.01 on the slope (0.0022 and 0.0065 measured, seeds 0-4), where the RMS scale is 0.05 and 0.64 off; ~1e-13 |
| `quantile` | statsmodels' `QuantReg`, a batch fit of the same objective, after 20,000 rows (`TestQuantileIsQuantReg`) | 0.05 at the median and 0.08 at τ = 0.9 (0.005 and 0.006 measured) |
| `quantile`'s every fit | the stationarity condition of the smoothed check loss it solves, rebuilt from the rows by the module doc's rules, band and nudge rows alike (`TestTheQuantileFitsDefinition`) | 1e-10 of the gradient's scale (1e-15 measured) |
| Huber | `river.linear_model.LinearRegression(loss=optim.losses.Huber)` | statistical |
| quantile | `river.stats.Quantile` | statistical |
| FTRL | `ftrl_ref` | ~1e-16 |
| `ftrl`'s targets and decay, `pa`, `sgd`, `holt` | `ftrl_ref`; the docstrings' update equations, written out (`pa_ref`, `sgd_ref`, `holt_ref`), in `tests/test_oracles_gradient_paths.py` | pred 8.2e-16 |
| FTRL | `river.optim.FTRLProximal`, row for row | 1e-12 |
| FTRL end to end: both losses, the intercept, row weights with zeros, null targets, two targets, `l1` with `l2` | Vowpal Wabbit's `--ftrl`, `pred` and `coef` on every row | 1e-5, its single precision (measured 2.6e-6) |
| `pa` | river's `PARegressor`, without an intercept, at unit weight and at `eps = 0`, river's tube being in the target's units and ours in the target's own EW std (`TestPassiveAggressiveIsRivers`) | 1e-12 |
| `sgd` | scikit-learn's `SGDRegressor`, one per group (`tests/test_sgd.py`) | R² within 0.03 |
| `sgd`'s `l2` and `clip_gradient` | `sgd_ref` with the ridge on the slopes and the per-coordinate clip, on a case where the clip binds on more than 1,000 coordinates under every schedule (`test_the_clip_and_the_ridge_bind`) | 1e-12, relative |
| `holt` | statsmodels' `Holt` once the weight saturates, and its state-space `ExponentialSmoothing` across a missing observation (`TestHoltAcrossAMissingObservation`); without decay, statsmodels' `WLS` on `(1, t)` before every row (`TestHoltWithoutDecayIsTheWeightedLineFit`); without a trend, statsmodels' `DescrStatsW` and pandas' `ewm(times=)` (`TestALevelOnlyHoltIsAnEwMean`) | 1e-12 and 1e-9; 1e-12; 1e-12 and 1e-8 |
| the plain `sigma` and `zscore` | the weighted EW root mean square of the residuals before the row (`tests/test_oracles_sigma.py`) | 8.1e-16 |
| `EwCov` | `river.stats.Mean` / `Var` / `Cov` / `PearsonCorr` | 1e-9 |
| `ew_cov`'s `mean`, `var`, `cov` and `corr` | pandas' `ewm` on a row-count clock, at offsets up to 1e8 (`TestTheEwMomentsArePandas`) | 1e-12 |
| EW mean/var | `river.stats.EWMean` / `EWVar` | in the limit |
| `ew_cov`'s lagged co-moments | the unrolled sum over the raw rows at the last row, AR(1) data with zero weights (`test_the_last_row_is_the_unrolled_sum_over_the_raw_rows`) | 1e-9 |
| `ew_class`; `ew_cov`'s `mahal` and components, under a window | scipy's `multivariate_normal` and `mahalanobis`, and numpy's `eigh`, on the rows the window keeps (`TestWindowedGaussian`) | 1e-9 |
| `marginal`'s bins and best split | scipy's `binned_statistic`; a scikit-learn `DecisionTreeRegressor` stump, whose gain bounds the split's from above (`TestTheBinsAgainstScipyAndAStump`) | 1e-9 |
| `marginal`'s serial count | statsmodels' `acf` and `weights_bartlett` (`TestTheBartlettSerialFactor`) | 0.02 |
| `bocpd` | the `bayesian_changepoint_detection` package, at levels up to 1e8 (`TestBocpdAtALevel`) | 1e-9, and `run_mode` exactly |
| `audit`'s column statistics | scipy's `skew` and `kurtosis`, statsmodels' `adfuller` with no lags and a constant, numpy's lag-1 `corrcoef`, `median` and median absolute deviation (`TestAuditIsScipyAndStatsmodels`) | 1e-8; the median and MAD exact from exact counts, and within ranks 49% to 51% (48% to 52% for the MAD) from the digest |
| `kmeans`, `micro` | `tests/reference_cluster.py`, a transcription of the Rust held bit for bit: a regression check, which can share a mistake with the code | bit for bit |
| `kmeans`, `micro`, independently | `TestDefinitions` in `tests/test_kmeans.py` and `tests/test_micro.py`, from the module docs: each checkpoint's centers, weights and radii recomputed from the raw rows assigned to them at their decayed weights; `kmeans`' far, merge and dead decisions and `micro`'s admission, row by row against the exported state; scikit-learn's `KMeans` on 100,000 rows without decay | 1e-9; the partition at ARI 0.99995 and the centers to 5.4e-4 |
| `deco`, `hmm`, `corrchange`, `bocpd`, `rcov` | a longhand oracle written from each paper ([An oracle, not a golden number](#an-oracle-not-a-golden-number)) | 1e-12 to 1e-9: `deco`'s pair sum, `hmm`'s probabilities, `corrchange`'s statistic and `bocpd`'s posterior to 1e-12; `hmm`'s log-likelihood and `rcov`'s three estimators to 1e-9, relative |
| a target with gaps | numpy's `lstsq` on the target's own rows, pandas' pairwise `cov`, and statsmodels' `WLS`, ridge and elastic net (`TestATargetWithGaps`) | 1e-10 to 1e-8; the elastic net 1e-7 |
| `kalman`'s explicit `q`, `obs_var` with `p0`, a `coef_half_life` per coefficient with `inf` pinning, and no intercept under `standardize` | filterpy's `KalmanFilter`, the last on rows scaled by each feature's EW root mean square from numpy (`TestKalmanSettingsAreFilterpy`) | 1e-11 (1.0e-14 measured) |
| `sgd`'s squared, epsilon-insensitive (at `eps = 0`, its tube being in the target's own EW std) and logistic losses under `"constant"`, weighted or not; `"inv_scaling"` at unit weights; the Huber cut is `sgd_ref`'s | scikit-learn's `SGDRegressor` and `SGDClassifier(loss="log_loss")`, `partial_fit` one row at a time (`TestSgdIsScikitLearnsSgd`) | 1e-12 (1.0e-15 measured) |
| `hmm` at fixed parameters | hmmlearn's `GaussianHMM`: the filtered and predicted probabilities, and each row's log-likelihood (`TestHmmIsHmmlearns`) | 1e-10 (1.4e-13 measured) |
| `rls` | padasip's `FilterRLS`, at unit weights on a row clock (`TestRlsIsPadasips`) | 1e-11 (1.6e-14 measured) |
| `seqtest` | `reference.seqtest_ref`, the paper's betting recursion in scalar code, and its closed form where the clip never binds (`tests/test_seqtest.py`) | 1e-12, relative |
| `emit_drift` | river's `PageHinkley` at `half_life = inf` (`TestAZeroWeightRowIsNotSeen`) | flag for flag |
| a prediction under `embargo` | river's progressive validation with `delay` (`TestLabelDelayFoldsWhatWasScored`) | 1e-12 |
| `r2` and `coverage` under `embargo` | scikit-learn's `r2_score` over the matured rows; the share of matured rows inside their interval | 1e-9 |
| `emit_metrics` and `emit_autocorr` | a row's own outcome moved across zero, which moves none of the row's own fields, to the bit; on a logistic fit, scikit-learn's `accuracy_score` and `brier_score_loss` and scipy's `pointbiserialr` (`TestStreamingMetrics`) | to the bit; 1e-9 |
| `resid_quantiles`, `mahal_quantiles` | the EW quantile by its definition, within the sketch's `tanh(1/128)` (`tests/test_diagnostics.py`); numpy's `quantile` of the past `mahal` scores (`TestMahalQuantiles`) | 3% (1.4e-3 measured); 10% |
| the `session_shrink` blend | numpy's `cov(aweights=)` of the rows at their blended weights (`TestSessionShrinkBlend`) | 1e-12 |
| `conformal` | **definition only**: a longhand replay of the recursion over the bank's own `pred`, `resid` and `sigma`, and the telescoped coverage bound (`tests/test_conformal.py`); no library computes an adaptive conformal radius | bit for bit |
| `huber` and `quantile` under a finite half-life, with nulls, several targets, `standardize` or zero weights; `ftrl` under a half-life; `sgd`'s Poisson loss and AdaGrad schedule; `ew_class` under decay; `EwQuantile` | **definition only**: `robust_ref`, `ftrl_ref`, `sgd_ref` and the definitions in `tests/test_diagnostics.py` and `tests/test_ew_class.py`. No library checked forgets as a half-life does, and scikit-learn's SGD has neither the Poisson loss nor AdaGrad | as each such model's row of the table gives |
| `micro`, `seqtest`, `deco`, `corrchange`, `rcov` | **definition only**: the longhand oracles of [An oracle, not a golden number](#an-oracle-not-a-golden-number), each written from its paper. river's `DenStream` takes another macro step, so it could agree only statistically, and no Python library computes the other four | as each such model's row of the table gives |
| the window operators | the brute force; the time-reversal identity; Polars' `ewm_mean_by` and `ewm_sum_by`; Polars' `rolling_sum_by`, for which rows a window holds ([The window operators and formula targets](#the-window-operators-and-formula-targets)) | 1e-9; 1e-9; 1e-9; exactly |
| `po.ewm_quantile`, looking back with and without a window, and looking ahead | numpy's weighted quantile, `np.quantile(..., weights=, method="inverted_cdf")`, over the rows each window holds at the weights `po.ewm_sum` gives them; under even weights, statsmodels' MAD about 0 of the last 101 rows (`tests/test_ewm_quantile.py`) | `tanh(1/128)`, relative, the sketch's accuracy |
| a formula target | its column form | bit for bit, every prediction |

### An oracle, not a golden number

**An oracle is written from the paper or the definition, never from the
code it checks.** An oracle written *from the implementation* agrees with it
by construction, so it cannot find a defect in it. An oracle written from
the paper is allowed to disagree, and where it does, either the code or the
oracle is wrong, and the test says which to look at.

**A test adds an oracle rather than a golden number.** The tests that close
[mutation survivors](#mutation-survivors) follow the same rule, and so do
the window operators' tests. An oracle takes one of four forms:

| form | example |
|---|---|
| the recursion written out longhand beside the implementation | Holt, Page-Hinkley, `sigma2` |
| an equivalent model configured a different way | the slow twin against a standalone model at `long_half_life`; the standardized solve against the plain one at zero penalty; a forward window against a backward one over the reversed stream |
| the optimality conditions of the problem being solved | the lasso's KKT conditions |
| the definition of the statistic | `read` against a recomputation from the raw rows; each window operator against the brute force |

### Libraries the package does not depend on

**A test may use any library the package does not depend on, when the test
needs it.** The user decided this on 2026-09-24: "packages that we do not
want to depend on are fine if needed to test with other libraries". The
package depends on polars alone,
with numpy as an extra, and that does not change. What a test needs is a
separate question, and three kinds of need have come up:

| need | libraries | example |
|---|---|---|
| an oracle, computed independently of this library | numpy, pandas, scipy | pandas' `ewm(times=)` against the temporal clock |
| a second opinion from another implementation | river, statsmodels, filterpy, bayesian-changepoint-detection, scikit-learn, Vowpal Wabbit, hmmlearn, padasip | `river.optim.FTRLProximal`, row for row; `HuberRegressor` beside `huber`; `GaussianHMM` beside `hmm` |
| interop, the other library reading a bank's output or feeding one | pyarrow, duckdb, the ADBC SQLite driver | `pa.table(s)` on `fit_predict_arrow`'s output |

**An oracle comes from a third-party library wherever one computes the
thing checked.** An oracle written here can share a mistake with the code it
checks, and a library's cannot. So before writing a longhand, look for a
library that computes the same quantity. Write by hand only the definition,
from the paper, such as the normal equations a fit must solve. The user set
the rule on 2026-09-27, while a Gaussian elimination was being written by
hand as a ridge oracle: "Is there no third party oracle library?"

| where the test is | library oracles | example |
|---|---|---|
| Rust, which `cargo mutants` runs | `faer`, already an `online-core` dependency, through `crates/online-core/src/oracle.rs`: its LU and eigensolver, never the model's own Cholesky in `solve.rs` | `every_solve_is_its_closed_form_across_targets_and_ridges` solves the normal equations with `oracle::solve`, `faer`'s LU |
| Python | scikit-learn, scipy, statsmodels, pandas, numpy | `TestEwRidgeIsSklearnsRidge` holds every `ewridge` solve to `sklearn.linear_model.Ridge`, 20 cases to 1e-8 |

**A mutant can be killed only from Rust**, because `cargo mutants` runs
`cargo test` and never the pytest suite. Where a library oracle exists only
in Python, write both: the Rust test with `faer` doing the linear algebra,
and the library's second opinion in `tests/test_second_opinion.py`. Where no
library computes the quantity, the oracle is written from the paper, never
from the code it checks.

Four rules govern such a library:

| rule | why | checked by |
|---|---|---|
| **1. Declare it in the dev group, by name, under an open license.** | A library that arrives through another package's dependencies breaks tests that never name it when that package goes, as pandas and scipy would if they came in only through statsmodels. A library under source-available or commercial terms is parked, not used (the user, 2026-09-25: "enable every unlicensed library in tests and park using licensed libraries") | `tests/test_dependency_policy.py`: every library a test imports is declared in the dev group, and every library of the dev and docs groups, and every Rust crate's dev-dependency, names an open license in its metadata; the test gate checks the lock with `uv lock --check` |
| **2. Import it plainly.** | The dev group is installed wherever the suite runs, so `pytest.importorskip` could only turn a broken environment into a skip. A missing library fails the test | the same file: no test calls `importorskip`, nor skips behind `except ImportError` |
| **3. Keep it out of the package's reach when its presence changes other libraries.** | pyarrow is the case: pandas and duckdb take other paths when it is importable, and a use of it inside the package would pass the suite | `tests/conftest.py` makes it unimportable; `test_this_session_runs_without_pyarrow` and `test_child_interpreters_run_without_pyarrow_too` check that it is |
| **4. Never add it to the package's own dependencies.** | Those stay `polars` | `tests/test_dependency_policy.py`: the package depends on polars alone, and its one extra on numpy alone |

**Rule 1 counts a license as open when the Open Source Initiative
approves it, or it is CC0,** named in the metadata as an SPDX expression or an "OSI Approved" classifier.
A library whose metadata names none is listed with the license read at its
source and where: `bayesian-changepoint-detection`'s wheel declares none,
and its repository's `LICENSE` is MIT.

**A library that must not be in every contributor's environment, for its
platforms, gets a group of its own and a CI job of its own.**

**Rule 3 reaches the child interpreters a test spawns.** `tests/child.py`
puts `tests/_site/sitecustomize.py`, which installs the same finder, on the
path of every child interpreter a test spawns, so an example run in a child
cannot see pyarrow either. Only `tests/test_pyarrow_interop.py`'s children
see pyarrow.

**Each library a second opinion takes holds what no other oracle reaches:**

| library | what it holds | where |
|---|---|---|
| scikit-learn | `sgd` against `SGDRegressor`, live; `huber` against `LinearRegression` at `huber_delta = 1e9`, the limit at which no row is down-weighted, and against `HuberRegressor` under outliers; `marginal`'s best split against a `DecisionTreeRegressor` stump; every `ewridge` solve against `Ridge`; `sgd`'s losses one row at a time against `SGDRegressor` and `SGDClassifier` | `tests/test_sgd.py`, `tests/test_second_opinion.py` |
| Vowpal Wabbit | `ftrl`'s `pred` and `coef` on every row, to 1e-5, since VW computes in single precision. Its `--ftrl` runs the recursion `ftrl.rs` states, and reaches options river's comparison does not: the squared loss, row weights, the intercept and the model's own predictions. Its wheel has no dependencies and ships for Python 3.12 to 3.14 on all three operating systems | `TestFtrlIsVowpalWabbits` |
| hmmlearn | `hmm`'s filter at fixed parameters, row by row. It ships wheels up to CPython 3.13, so the 3.14 legs build its sdist, a small C++ extension | `TestHmmIsHmmlearns` |
| padasip | `rls`, through its `FilterRLS`; pure Python | `TestRlsIsPadasips` |
| filterpy | the `kalman` settings, through its `KalmanFilter` | `TestKalmanSettingsAreFilterpy` |
| Pathway, parked | it would run the Pathway half of `examples/pathway_integration.py`, but it is under the Business Source License, which rule 1 parks | `test_pathway_is_not_a_dependency` keeps it out of `pyproject.toml` until the user decides |

None of the libraries checked forgets as a half-life does (river, VW, and
Keras's `Ftrl`), so `ftrl` under a finite half-life stays with `ftrl_ref`.

### The window operators and formula targets

No library computes a windowed or forward form, so the oracles here are
the definition and an identity, with Polars where it has the same operator.

**Each window operator is held three ways: to the definition, to Polars'
own functions where Polars has the operator, and to the time-reversal
identity.**

| oracle | what it checks | held by |
|---|---|---|
| the brute force | every operator, a mean, a sum and a rate, on nine kernels: each direction, every `closed`, finite and infinite half-lives. Five streams of 400 rows carry gaps, repeated stamps, sessions, nulls and groups. To 1e-9 | `windows.rs::every_operator_matches_its_definition` |
| a loop from the definition, in Python | the mean, the sum and the rate, each looking back and looking ahead, under each `closed`, to 1e-9 | `test_windows.py::test_each_operator_matches_the_definition` |
| Polars' own functions | the backward mean and sum, with no window, against `ewm_mean_by` and `ewm_sum_by` (1.44.1 on) on a `Datetime` clock, to 1e-9; which rows a window holds under each `closed`, against `rolling_sum_by`, exactly | `test_polars_computes_the_same_mean_and_sum`; `test_which_rows_a_window_holds_is_polars_rolling` |
| the time-reversal identity | a forward window against a backward one over the reversed stream, to 1e-9 | `windows.rs::a_forward_window_is_a_backward_one_over_the_reversed_stream`, and `test_windows.py`'s test of the same name |
| chunking | chunks of 1, 2, 3, 7, 50 and 499 rows give one chunk's bits | `windows.rs::chunking_changes_no_bit` |
| a public day of data | one symbol-day of Binance futures quotes and trades through four formulas, chunked; two recipes for one VWAP agree to 1e-6 | `test_the_real_day_runs_and_the_recipes_agree_where_they_should` |

**Each oracle reaches what the others cannot.** The brute force and the
identity check every row against the definition, so a forward window that
counts a row from outside its stretch fails both. The day of
data is ETCUSDT on 2024-01-02, from data.binance.vision.
`tests/data.py::public_quotes_and_trades` downloads it once and caches it
under `.cache/microstructure`, and offline that one test skips.

**A formula target is held to its column form, bit for bit on every
prediction.** The same window expression, written as a column and fed back
as a plain target under the same `embargo`, gives the same prediction on
every row:

| case | held by |
|---|---|
| chunked and whole: 1, 7 and 600 chunks in Rust, and `chunk_size` of 1 and 97 in Python | `formula_targets.rs::a_formula_target_is_the_column_form_fed_back_under_the_embargo`; `test_formula_targets.py::test_a_formula_target_is_the_column_form_fed_back` |
| through every clock event: a gap past the cap, a session change, a step back the policy resets on, groups resetting together, rows at one stamp, a row exactly one window later, a run of skipped rows longer than the cap, and the input ending inside a window | `test_parity_through_every_clock_event` |

### Resuming a run

**A run resumed from its state gives the output of the run that never
stopped, and a state refuses an input or a version it does not belong to.**
A bank's state and a window run's can each be saved to a file and loaded
back. A window run saved under a slice, such as `head(n)`, also records how
much of its input it consumed. Each test's docstring states the case it
holds:

| what is resumed | the claim | held by |
|---|---|---|
| a bank of any kind | saved mid-stream and loaded, it goes on exactly as the one that was not | `tests/test_every_kind.py`; `model_contract.rs`'s generated streams, at any row; a `quantile` fit holding a moved band factor, at 101 save points (`robust.rs`) and at every row from 990 to 1010 through the bank (`test_robust.py::TestTheBandFactorIsState`) |
| a bank on input that overlaps its state | `ModelBank.skip_learned` keeps each row after its group's last clock, so the rerun learns each row once | `tests/test_skip_learned.py` |
| a formula target with a window open | a state saved mid-window resumes as one run | `a_state_saved_mid_window_resumes_as_one_run`, in `formula_targets.rs` and in `test_formula_targets.py` |
| a window run, saved at a row | a save and load at row 1, 2, 17, 33 or 59 is one run | `test_a_save_and_load_at_every_row_is_one_run` |
| a chain of window runs under a slice | six sliced runs, each resumed from the last state and saved again, then the rest, give one run's output, at `chunk_size` of 1, 7 and 100,000 | `test_a_state_saved_under_a_slice_resumes_on_the_same_input`, and its Rust twin in `windows_frame.rs` |
| a window state and another input | the state knows its input. Another input is refused by name, and the next file goes on: with a clock column and without one, and from a state that holds no rows | the identity tests in `test_windows.py` and `windows_frame.rs`, such as `test_a_state_saved_under_a_slice_refuses_another_input` and `test_without_a_clock_the_next_file_begins_with_a_new_session` |
| a state of another version | refused by its version, never misread | `windows_frame.rs::a_windows_state_version_moves_the_banks_schema_with_it`; `test_a_windows_state_of_another_version_is_refused_by_its_version`; `tests/test_released_state.py`, on the files each release wrote, every 1.x release read from PyPI |

The first test of the last row is a tripwire. It asserts the windows state's
version and the bank's schema version together, because a bank file embeds a
window core for each formula target. So neither can move without the other.

### What the README's guarantees are held to

**Each guarantee in the README's [What you can rely
on](../README.md#what-you-can-rely-on) is held by a reference, an
invariant, generated streams, fixed numbers, a deliberate attack or a file
the repository promises.** Each model's references, with the agreement each
reaches, are in [Against reference
implementations](#against-reference-implementations). In the Rust tests the
core's hand-written solves are held to `faer`'s ([Libraries the package
does not depend on](#libraries-the-package-does-not-depend-on)).

**Invariants, for every model**, checked at the bank and, where they apply,
at the command line:

| invariant | checked as |
|---|---|
| chunk invariance | one chunk, seven, a hundred, one row at a time, and with a save and load in the middle ([class 2](#the-eight-test-classes)) |
| thread invariance | 1 thread against 8, bit for bit (`tests/test_portability.py`, `tests/test_hardening.py`) |
| group independence | a group's numbers do not depend on what else is in the bank ([class 4](#the-eight-test-classes)) |
| the paths agree | runner ≡ bank for every input source and format; the Arrow output ≡ the Polars one, field for field and null for null ([classes 6 and 6c](#the-eight-test-classes)) |
| `predict` ≡ `fit_predict` | of the next row, field for field, with every diagnostic on ([class 6b](#the-eight-test-classes)) |
| stream semantics | the null policy, warm-up, and the clock ([classes 4 and 5](#the-eight-test-classes)) |
| `weight_sum` | the same recursion in every model (`crates/online-core/tests/model_contract.rs`) |
| a window run resumed | a chain of runs saved under a slice and resumed gives one run's output, at chunks of 1, 7 and 100,000 rows, and another input is refused ([Resuming a run](#resuming-a-run)) |

**Generated streams.** In the Rust tests, `proptest` drives generated
streams through every model against the same contract. Hypothesis
generates adversarial streams, with mixed nulls, duplicate and long-gap
clocks, values at ±1e8, zero weights and tiny groups. It asserts the
strongest invariant: **changing a row's own target never changes that
row's own prediction.** An information coefficient (IC) of about 0 on
pure-noise targets checks the same thing from the other side
([class 3](#the-eight-test-classes)). The Hypothesis properties are in
`tests/test_properties.py`.

**Fixed numbers.** Each model has one golden stream in the Rust core
(`crates/online-core/tests/golden.rs`). The whole pipeline, from reading the columns through the models and
the diagnostics to assembling each spec's output, is pinned to fixed output
and compared on every operating system (`tests/test_golden_pipeline.py`). So a divergence in Polars'
vectorized paths on another CPU would show.

**Hardening.** What the suite does to a bank on purpose, held in
`tests/test_production_hardening.py`, `tests/test_hardening.py` and, across
the boundary with Polars, `tests/test_ffi_memory.py` and
`scripts/leakcheck.sh`:

| attack | what must hold |
|---|---|
| everything at once | a 30k-row stream with every output switched on, compared by digest across chunkings, a mid-stream save and load, and thread counts |
| weight scale | all weights ×1e±6, with `min_weight` scaled alike, change nothing but `weight_sum` |
| parameter edges | `half_life` from `1e-3` to `inf` |
| a corrupt state file | any byte flipped fails cleanly, and never panics |
| concurrent misuse | two threads calling `fit_predict` at once get a clean error |
| copying a bank | `pickle` and `copy.deepcopy` resume bit-exactly |
| across the FFI | memory safety where two copies of Polars share one process |
| sustained load | a 10M-row soak, and 1e8-row stability cases against recomputations from scratch, opt-in with `pytest -m soak` |

**Contracts that are files.** What is held to a file, and how:

| contract | how it is held |
|---|---|
| the public API: every name, default and signature, and every output field name | a checked-in snapshot (`tests/api_surface.txt`), so a change is a reviewable diff (`tests/test_api_surface.py`) |
| every python block in the README, and every example in the API reference | each one runs |
| some docstring text | pinned where a test reads it |
| everything under `examples/` | runs unmodified: the TOML through the real command line, the Pathway example's operator over plain batches, and the cursor examples against real DuckDB and SQLite databases (`tests/test_examples.py`) |
| the state files every release from 0.10.0 on wrote | a 0.x release's refused by their schema, and a 1.x release's loaded and continued, as the CHANGELOG says (`tests/test_released_state.py`) |
| `docs/VALIDATION.md`, where the defaults were chosen | regenerated and compared, so the numbers behind them cannot silently stop being true (`tests/test_validation_doc.py`) |
| no data files in the repository | tests generate or download their own data, and a data file, a large file or generated output that gets tracked fails a test (`tests/test_repo_hygiene.py`) |

## How the suite runs

**The essentials gate each commit while a task is in progress, the whole
suite runs before every push, and CI runs it on Linux, macOS and Windows
at every push to `main` and every pull request.** The test gate is
`scripts/gate.sh`, which runs the lints and the tests of one tier. With
several branches in flight, each task works in a worktree slot (a git
worktree kept between tasks, with its own builds) and runs only the tests
its change touches, and the build caches the runs leave are cleared after
each push. The suite's data comes from
the network, which CI keeps copies of, and its last bits depend on the
machine it runs on.

### Two tiers

**The essentials gate each commit while a task is in progress, and the
full suite runs before every push, always.** A contributor working alone
runs the essentials test gate, `scripts/gate.sh`, on each commit. Where
several branches are in flight in worktree slots, each task runs
`scripts/quick.sh` instead, and one essentials test gate runs on each wave
of two or more merged branches ([Iterate on a
change](contributing/iterate-on-a-change.md)). The user set the purpose on
2026-10-08: "We will always run the full tests before a push, the idea of
the essential tests is to improve the speed of iteration while developing
many steps." A push never rides on the essentials, whatever they said.

| when | what runs | tier |
|---|---|---|
| each commit while a task is in progress, working alone | `./scripts/gate.sh` | the essentials |
| each change inside a task, with other branches in flight | `./scripts/quick.sh`, then one `./scripts/gate.sh` on each wave of two or more merged branches | the tests mapped to the changed paths; then the essentials |
| before every push, always | `./scripts/gate.sh --extended` | every test but the soak tests and the oracle grids |
| every push and pull request | `ci.yml` on Linux, macOS and Windows | every test but the soak tests and the oracle grids, and the essentials nowhere |
| every push and pull request | `mutants.yml`, its pass over the changed lines | every Rust test, the ignored ones included, against each mutant of the changed lines |
| every push and pull request | `budgets.yml`, on Linux | the instruction counts, against `budgets.toml` ([Budgets that only tighten](#budgets-that-only-tighten)) |
| every push to `main`, weekly, and on dispatch | `oracles.yml`, sharded across parallel jobs on Linux | the oracle grids (`-m oracle_grid`), each case held to the oracle ledger ([The oracle ledger](#the-oracle-ledger)) |
| before a release | `release.yml`, which runs `ci.yml` and `budgets.yml` | as those two |
| on their schedules | `polars-canary.yml`; `mutants.yml`'s weekly pass | every test but the soak tests, the oracle grids and the `pins` tests; every Rust test, the ignored ones included, against each mutant of all of `online-core` |

**A test is in the essentials unless it is slow, reaches the network or
sweeps.** The essentials
hold a test of every hard rule, every golden and the contract tests. They
hold the refusals and renames, the API snapshot and the document-structure
tests too, and each module's own unit tests, each in its fastest form that
can still fail. A test moves to the extended tier when it takes a second or
more or needs the network or downloaded data. It moves too when it re-runs
a document's experiments ([REGIMES.md](REGIMES.md),
[VALIDATION.md](VALIDATION.md)), measures memory or threads,
sweeps a grid, or runs a third-party oracle over a long stream. The only
test of a hard rule never moves whole: a reduced form, with fewer rows or
cases, stays in the essentials, and the full form moves. A golden stays
whole however slow, since fewer rows are other numbers. The `soak` tests
stay opt-in.

**The oracle grids run on CI, not in the test gate.** A grid that holds a model
to a third-party oracle over 50 seeds or orders, with batch fits over long
streams or downloaded datasets, is marked
`@pytest.mark.oracle_grid(reason="...")`. The user set the rule on
2026-10-25: "we do not want to run more very long tests in the gate as it's
already long". The convergence grids took 22 minutes in four processes, more
than the rest of `gate.sh --extended` together. No local run takes them: `addopts` and both tiers of `gate.sh` leave them
out. `.github/workflows/oracles.yml` runs them on every push to `main`, one job a
module or a model, in parallel (CI's minutes are free; its limits are 20
jobs at once and six hours a job). Read its conclusion after each push with
the other workflows'. `uv run pytest -m oracle_grid` runs them here, and
`pytest -m oracle_grid tests/test_convergence.py -k quantile` one model's.
Each such module keeps one cheap representative cell per class in the
essentials, so the code paths stay covered locally.

**How a test says its tier:**

| where | extended | essentials |
|---|---|---|
| a pytest test | `@pytest.mark.extended(reason="...")` on the test, on one of its parameters, or as its module's `pytestmark` | no mark |
| a pytest module | `TIER = "extended"`, or `TIER = "mixed"` when only some of its tests are (or are `oracle_grid`); `TIER = "oracle_grid"` when its `pytestmark` is | `TIER = "essential"` |
| a Rust test | `#[ignore = "extended: <reason>"]` under its `#[test]` | no `#[ignore]` |
| a Hypothesis property | the `extended` profile: the count its file names, 30 by default | the `essential` profile: at most 10 examples, or the smaller count its file names |
| a proptest property | proptest's own 256 scaled by each property: 128 to 2,048 | `PROPTEST_CASES=32`, an eighth: 16 to 256 |

**A plain `pytest` runs both tiers; a plain `cargo test` runs the
essentials.** `uv run pytest` is the full local run. Its `addopts` leave
out the soak and the oracle grids, and the Hypothesis profile is `extended` unless
`HYPOTHESIS_PROFILE=essential` is set (`tests/tiers.py`). cargo skips an
ignored test, so `cargo test` alone runs the Rust essentials, and
`cargo test -- --include-ignored` is the full Rust run. `gate.sh
--extended`, CI, the release, the canary and `scripts/coverage.sh` all pass
it. The mutation runs in CI test each mutant under cargo-nextest and pass
`--cargo-test-arg=--run-ignored=all`, nextest's form, and
`scripts/mutants.sh` passes `-- -- --include-ignored` to `cargo test`, so a
mutant only an extended test catches is not counted a survivor. The essentials test gate
runs pytest with `-m "not extended and not soak and not oracle_grid"`, the
`essential` profile and `PROPTEST_CASES=32`.

**Every hard rule a test can hold has tests in the essentials.**
`tests/test_tiers.py` reads this table and fails if a test it names is
missing or extended:

| hard rule | essentials tests |
|---|---|
| 1, no data files | `tests/test_repo_hygiene.py::test_no_data_files_are_tracked` |
| 2, out-of-sample | `tests/test_bank.py::TestOutOfSample::test_noise_target_has_no_ic`, `tests/test_properties.py::TestUniversalProperties::test_prediction_never_depends_on_the_current_target`, `crates/online-core/tests/model_contract.rs::ewridge_predict_is_the_step` and each model's beside it |
| 3, chunk invariance | `tests/test_bank.py::TestChunkInvariance::test_chunked_equals_single`, `tests/test_semantics_all_models.py::TestUniversalInvariants::test_chunk_invariance`, `crates/online-polars/tests/bank.rs::chunk_invariance`, `crates/online-polars/tests/summary.rs::chunking_cannot_move_a_bit` |
| 5, frozen fixtures | `crates/online-core/tests/state_fixtures.rs::every_fixture_goes_on_to_the_bit`, `crates/online-core/tests/state_fixtures.rs::every_fixture_saves_its_bytes_again`, `crates/online-polars/tests/state_fixtures.rs::every_fixture_loads_goes_on_to_the_bit_and_saves_its_bytes_again` |
| 8, `weight_sum` | `crates/online-core/tests/model_contract.rs::ewridge` and each model's probe beside it, which hold the accessor and the reported weight to the one recursion; `tests/test_semantics_all_models.py::TestWarmup::test_n_eff_is_reported_before_the_update` |
| 9, a zero weight | `crates/online-core/tests/model_contract.rs::ewridge` and each model's probe beside it, which run `zero_weight_rows_only_advance_the_clock`; `tests/test_edge_cases.py::TestWeights::test_zero_weight_is_a_pure_decay_row` |

The build holds rule 4 and rule 6's ban on `unsafe`: `online-core`'s manifest has no Polars
dependency and forbids `unsafe_code`. Rule 7 is a practice.

**The budget is time, so the test gate prints it rather than a test checking
it.** The essentials aim at pytest under 90 s and `cargo test`'s run under
60 s at this machine's usual load. On 2026-10-08, at a load of 4 to 5 on 14
cores, pytest's essentials took 72 s against the full suite's 196 s, and
`cargo test`'s ran 29 s against 146 s. `gate.sh` names its tier on its
first and last lines and prints each step's time, so a slow test that
creeps into the essentials shows where they run. CI runs only the full tier,
which every push runs anyway, so it spends no minutes timing the
essentials (the user, 2026-10-08). Each test
that moved gives its reason in its mark.

### Iterating

**[Iterate on a change](contributing/iterate-on-a-change.md) is the
how-to.** Working alone, a contributor runs the essentials test gate,
`scripts/gate.sh`, on each commit. With several branches in flight, each
task runs `scripts/quick.sh` after each change and `scripts/quick.sh
--extended` and the lints at its end, and one essentials test gate runs on
each wave of two or more merged branches ([Merge a wave of
branches](contributing/merge-a-wave-of-branches.md)).
`scripts/gate.sh --extended` runs before every push. The development Mac
these figures come from has 14 cores, shared with another user's jobs.

#### What `scripts/quick.sh` runs

**`scripts/quick.sh` runs the tests that cover the paths a branch changed,
and nothing else.** It reads the paths from `git diff` against the
merge-base with `main`, with uncommitted edits and new files included. A
source change in a crate the extension links (`online-core`,
`online-polars`, `online-py`) rebuilds the extension before pytest, which
would otherwise test the old binary. A change in a crate with Rust tests
(`online-core`, `online-polars`, `online-cli`, `online-bench`) runs that
crate's `cargo test`. Any source change in `online-core` also runs
`online-polars`' state fixture and portability tests. A bank's frozen
fixture holds every model's state, and it moves with a model's arithmetic
while every `online-core` test passes. A Python module runs the test files
named after it, and any path runs the test files that name it. A model's
source and a test file that registers oracle groups also run [the oracle
ledger](#the-oracle-ledger)'s hash guard for the groups they can move,
which takes seconds. `quick.sh`'s verdict says it covers the mapped tests
only, because a change can break a test the map does not reach.

**Inside a task, the extension is built with the `dev-wheel` profile,
which takes seconds after an edit where `release` takes minutes.** The
profile is `release` with incremental compilation and 256 codegen units,
and without thin LTO, so a rebuild redoes only what changed. `quick.sh`
builds with it, and so can `uv run maturin develop --profile dev-wheel -m
crates/online-py/Cargo.toml`.

| the extension's build | `release` | `dev-wheel` |
|---|---|---|
| from nothing | 241 s (load 10) | 130 s (load 5 to 18) |
| after a new function in `ewridge.rs` | 189 s (load 8 to 11) | 4.5 s (load 14 to 24) |
| after an edit inside `solve()`, which every model's fit calls | 189 s | 37 s (load 9 to 15) |
| the Python essentials on it, one process | 116 s | 118 to 123 s |

**A `dev-wheel` build computes the same bits as a `release` build.** Rust
fuses no floating-point operations and reorders none unless the code asks,
so neither LTO nor the number of codegen units can move a result. That was
checked rather than assumed. `scripts/release_probe.py` runs every spec
over a stream built to reach every path. It gave 297 output columns over
400 rows identical in every bit under both builds, and its 31 saved states
were identical byte for byte. The whole Python suite passed on the dev
build, the goldens and the saved-state tests among them. The test gate, CI
and the shipped wheel still build `release`, because what ships is what a
push is tested on.

#### How the test gate runs

**At most two test gates run at once on the development Mac.**
`scripts/gate.sh` takes one of `POLARS_ONLINE_GATES` (default 2) tokens,
directories in the repository's common git directory holding their test
gate's pid. It waits for one when all are held, prints that it waited and
for how long, and starts its timing when it has a token. A token whose
test gate has died is taken back. CI runs one test gate per job and takes
none.

**The test gate runs the tests in parallel.** cargo-nextest runs each Rust
test as a process of its own, and pytest-xdist runs the Python tests in
several processes. Both run every test that the serial run runs, with the
same results (measured on 2026-10-11):

| tests | one process | in parallel |
|---|---|---|
| the Rust essentials | 19 s, `cargo test` | 12 s: 9.5 s under nextest and 2.6 s for the 5 doctests |
| every Rust test | 182 s | 78 to 86 s under nextest, and the doctests |
| the Python essentials | 116 to 123 s | 30 s in 8 processes (39 s in 4) |
| every Python test | 302 to 318 s | 134 s: 59 s in 8 processes, then 75 s for the 20 `serial` tests of that day alone |

**The test gate starts as many processes as the machine has cores free,
from 2 to 8.** It reads that as the core count less the last minute's load
average (`scripts/workers.py`), so it does not crowd out another user's
jobs, and `GATE_WORKERS=4` overrides it. Past 8, more processes add little
speed and crowd the machine: 12 took the Python essentials from 30 s to
27 s and the load average from 23 to 51. The load average counts the test
gate's own last run too, so a test gate started right after another runs
in fewer processes. The test gate runs `cargo test --doc` beside nextest,
which runs no doctests. Where nextest is not installed, it runs `cargo
test`. `scripts/gate.sh --serial` runs every test in one process.

**A test that reads its process's memory runs alone, after the rest.** The
memory tests of `tests/test_ffi_memory.py`, `tests/test_gbdt.py` and
`tests/test_budgets_memory.py` carry the `serial` mark, 23 tests (counted
2026-11-08). Memory that other processes take can make macOS compress or
swap pages, which moves the resident size these tests read. The test
gate's parallel run leaves them out, and a second run takes them in one
process. Run that way, each leak check still fails a planted leak: keeping
a bank's every output read 155 KB a call, and keeping every tree-model
build 1,505 KB a build, while the loop without the leak passed. The leak
checks run before the tests that read a 2 GB source. Those leave the
allocator so much freed memory that a planted leak of the tree-model
builds goes unseen after them.

**A test that writes outside its own temporary directory must write a whole
file in one step.** Several processes share `.cache/`, where the downloads
land. A download is written to a file named for the writing process and
renamed into place (`tests/data.py::write_atomically`), so a reader sees no
file or the whole one. The released versions' environments, which
`scripts/compare_release.py` builds, are built under a per-process name and
renamed the same way.

**A compiler cache does not speed up a new worktree, so it is off unless
asked for.** sccache, turned on with `POLARS_ONLINE_SCCACHE=1` in
`scripts/env.sh`, caches each crate rustc compiles, keyed by its inputs. In
a checkout whose path it had seen, a build from nothing took 33 s for the
Rust tests and 91 s for the `release` extension, against 66 s and 241 s
without it. A new worktree has a new path, and from a new path a third of
the crates missed. A crate that includes a file its build script wrote
names that file by its absolute path, which differs per checkout. Every
crate built on such a crate misses with it, serde and Polars among them.
The misses took longer than the hits saved: 92 s and 519 s, slower than no
cache. CI never turns it on (`tests/test_dev_loop.py`).

#### Keep the machine's load down

**At most two agents build at once on the development Mac, three while
the 1-minute load is under about 20, each held to six build jobs and four
test processes.** An agent here is any process that builds and tests a
checkout, a contributor's or an automated worker's. Each worktree has its
own `target/` and compiles Polars there, which takes 66 s for the Rust
tests and 130 s for the `dev-wheel` extension from nothing, at a load near
5. More builds at once do not finish sooner, and past a point they bring
the machine down. Five essentials test gates at once ran at a load of 78
on 14 cores, and took 500 to 2,000 s each where one alone takes about 300.
Four `cargo mutants --in-diff` runs at once held a load of 88 for hours,
and the kernel panicked (`watchdog timeout: no checkins from watchdogd in
90 seconds`), measured 2026-10-27 and 2026-10-28. A `cargo mutants` run
builds and tests every mutant, each time with every core, so two at once
already saturate the machine. So, on that machine:

- **Two agents at most, three while the load is low.** A load under about
  20 leaves room for a third build, so a third starts only then, and none
  starts above it. When the agents' work is mostly builds and tests rather
  than reading and writing, run one at a time.
- **Each agent exports `CARGO_BUILD_JOBS=6`.** It runs pytest at `-n 4` at
  most, and the test gate at `GATE_WORKERS=4`.
- **A local `cargo mutants --in-diff` runs alone.** It takes the lock
  directory `.claude/coord/mutants.lock` (`mkdir` it, `rmdir` it after), and
  any other waits. CI's changed-lines pass on the push checks the same lines
  on GitHub's machines, so a local run is rare.
- **Check `uptime` before starting a build.** Wait while the 1-minute load is
  above about 20.
- **After a crash, look before resuming.** Read
  `/Library/Logs/DiagnosticReports/panic-full-*` for the cause, and remove a
  stale `mutants.lock` that a killed run left.

### Worktree slots

**A task works in a worktree slot, a git worktree kept between tasks with
its own builds, so it rebuilds only what it changes.** [Use a worktree slot](contributing/use-a-worktree-slot.md) is
the how-to: `scripts/slot.sh take`, `give`, `list` and `overlap`, and what
a warm slot saves.

### Caches and when to clear them

**Clear the build caches after each push wave, or when free disk falls
below 100 GB, with `scripts/clear_caches.sh --worktrees`.** [Clear the
caches](contributing/clear-the-caches.md) is the how-to, with every cache
the builds and the tests fill, its size, and what deleting it means for the
next build.

### Where the suite runs

**Every push to `main` and every pull request runs the suite on Linux,
macOS and Windows.** Locally, the test gate runs the essentials while a
task is in progress and every test before every push ([Two
tiers](#two-tiers)). Each workflow under `.github/workflows/` runs on its
own schedule:

| when | where | what runs |
|---|---|---|
| each commit while a task is in progress, working alone, and once on each wave of two or more merged branches | `./scripts/gate.sh` | `cargo fmt`, `clippy -D warnings`, the Rust tests' essentials (under cargo-nextest where it is installed), `uv lock --check`, `ruff`, `mypy`, the extension's build, `pytest`'s essentials in parallel, and Sphinx with `-W` |
| before every push, always | `./scripts/gate.sh --extended` | the same checks with every test but the soak tests and the oracle grids |
| every push to `main` and pull request | `ci.yml` | the suite on Linux at Python 3.12, 3.13 and 3.14, on macOS at 3.12 and 3.14, and on two Windows builds, Server 2025 and Server 2022, at 3.12 and 3.14, each leg first printing the machine it ran on ([The machines and the CPU paths](#the-machines-and-the-cpu-paths)); on Linux, the format, lint, type and documentation checks (`cargo fmt`, `clippy`, `ruff`, `mypy` and Sphinx); the release comparison's outputs against the newest release's, bit for bit, as a report; the Python coverage, as a report |
| every push to `main` and pull request | `mutants.yml` | mutation testing of the lines the change touched in `online-core` and `online-polars/src/span.rs`, in a shard for every thirty mutants the change lists, with one report. The report fails on a mutant no test catches, unless `scripts/mutants_equivalent.toml` lists it |
| every push to `main`, weekly, and by hand | `oracles.yml` | the oracle grids (`-m oracle_grid`): the convergence, learning-rate, truth-recovery and long-run grids against third-party batch fits and downloaded datasets, in twelve shards dealt round-robin over the tests (`--oracle-shard`), on Linux; no local run takes them ([Two tiers](#two-tiers)). A last job holds every case each shard recorded to the oracle ledger; the weekly run downloads the datasets afresh ([The oracle ledger](#the-oracle-ledger)) |
| every push to `main` and pull request, and by hand | `budgets.yml` | the instruction counts under Callgrind, against `budgets.toml` ([Budgets that only tighten](#budgets-that-only-tighten)) |
| every push to `main` but a docs-only one | `benchmark.yml` | throughput, into the job summary and an artifact, never gating |
| every release | `release.yml` | the whole of `ci.yml` and of `budgets.yml`; a state written on macOS and continued on Windows and Linux; the suite on the newest Polars the range admits, on the floor of the range and on the newest NumPy, all blocking; the suite on the next Polars major and on NumPy's next release candidate, both advisory |
| weekly | `ci.yml`, `polars-canary.yml`, `leakcheck.yml`, `benchmark.yml`, `mutants.yml`, `msrv.yml`, `advisories.yml`, `cpu-paths.yml` | the suite again; the suite on the newest py-polars, release candidates included, and on NumPy's next release candidate; the leak check across the boundary with Polars, on Linux and macOS, its control included, and the soak tests; throughput; mutation testing of all of `online-core`, as a report; `cargo check` of the workspace on the Rust its `rust-version` declares; both lock files against OSV's published advisories (`scripts/advisories.py`), failing on a vulnerability in a crate the wheel or the command line links and reporting the rest; on Windows, the numeric tests down each CPU path in turn, under Intel SDE and with OpenBLAS's and NumPy's paths pinned ([The machines and the CPU paths](#the-machines-and-the-cpu-paths)) |
| monthly | `polars-canary.yml` | the suite on the floor of the Polars range, the version `pyproject.toml` declares |

**The canary and the release's Polars and NumPy legs leave out the tests
marked `pins`.** Those assert this repo's own Polars pins, so they cannot
pass on another Polars.

**On the floor of the Polars range, a test that needs a newer Polars
skips,** naming the version it needs and why (`tests/polars_version.py`).

**Tests generate or download their own data (hard rule 1),** as [Downloaded
data in CI](#downloaded-data-in-ci) describes.

#### The policy, as the test holds it

`tests/test_ci_cost_policy.py` parses every workflow, in 94 cases over
thirteen classes, counted on 2026-11-08:

| class | what it asserts |
|---|---|
| `TestEveryJobIsBounded` | every job in every workflow has a timeout of at most two hours, and every workflow a concurrency group; superseded runs are canceled, and releases queue instead. The longer jobs are the mutation shards, on Linux: the changed lines' at 330 minutes on every push, the weekly pass's at 350 and only while public or by hand; and `ci.yml`'s test legs at 300 minutes while the repository is public, since Windows Server 2022's two-thread legs take 2.4 to 2.9 hours |
| `TestTheMatrixDefaultsToCheap` | lint stays on Linux and never builds the extension; the expensive runners are opt-in; the visibility test fails safe on a missing field; macOS is reachable only by a manual dispatch or the repository being public |
| `TestWindowsIsTwoImagesAndEveryJobNamesItsMachine` | the test matrix runs Windows Server 2022 beside the newest wherever it runs Windows, and every job that can land on Windows prints its machine first, so a failure names it |
| `TestStepOrderingThatHasAlreadyBrokenCI` | disk is freed before the cache is restored, rustflags are set before anything compiles, and the cache survives a failing job |
| `TestDocOnlyPushesAreFree` | CI has no paths filter on a push or a pull request; the benchmark, which reports and never gates, skips doc-only pushes |
| `TestPythonVersions` | every Python the package declares runs on Linux, and the floor and the newest on every OS; the release comparison reports once and never gates; the API reference is built and published once |
| `TestMutationTesting` | the mutation run over changed lines gates every push and pull request, with one report, in as many shards as its mutants need: a first job lists them, with the scope, the diff and the pinned cargo-mutants the shards use, and takes a shard for every thirty, at most four-fifths of the slowest shard measured, each stopped at 300 minutes (three times the slowest shard measured, 98) with every run inside a memory scope whose cgroup the kernel holds to 12 GiB, so a runaway mutant is killed alone and cannot shut the runner down (a job proves it on every run), each mutant tested under cargo-nextest with the extended tier, and the report expects that many; each push's pass has a concurrency group of its own, so no later push cancels or replaces it, while a pull request's newer commit cancels its older; the weekly pass runs only while public or by hand, and reports its survivors; every run skips the doctests, stops a mutant at ten times the baseline, stops itself inside its job, and lists the mutants it was given, and its report fails on a run that tested fewer or a shard that sent nothing |
| `TestTheRustTestsLinkNoPython` | every workflow and the local test gate leave `online-py` out of `cargo test`, so no test binary links libpython |
| `TestTheLinuxPrepIsOneAction` | the step that frees the disk and swaps in `lld` is one composite action, called after the checkout and on Linux alone |
| `TestActionsArePinnedToCommits` | every external action is pinned to a commit, with the version it names in a comment |
| `TestTheDeclaredRustVersionBuildsTheLock` | the workspace's `rust-version` covers every locked dependency's own |
| `TestTheDeclaredRustVersionCompiles` | a weekly Linux job, `msrv.yml`, checks the whole workspace on exactly the `rust-version` Cargo.toml declares, from the lock as committed |
| `TestEveryDeselectedMarkerRunsSomewhere` | each pytest marker the default run leaves out is asked for by some workflow |

### Downloaded data in CI

**No dataset is in the repository, and none is published anew.** The
public sets stay at their sources, and CI keeps a copy of each where only
this repository's workflows can read it. Downloads are cached under
`.cache/`, and a test that needs one skips offline: the public intraday
data (`tests/data.py::public_intraday`), the released wheels, the day of
quotes and trades, and the public benchmark sets. Three copies stand
between a CI run and a server that is down.

1. **The Actions cache**, keyed on the list of sets alone:
   `uv run python scripts/oracles.py datasets [--suite] --digest` hashes
   the sets the ledger's groups name (and with `--suite`, the sets the rest
   of the suite reads, `tests/public_benchmarks.py`'s `SUITE`). The key does
   not hash `oracles.toml`, which every ledger `apply` rewrites, so a
   measured number moving leaves the key as it was. A loader that changes what it writes to
   `.cache/` must change its file name too, as it must for a developer's
   own `.cache/`, which nothing clears either.
2. **The `oracle-datasets` artifact**, on a cache miss. The repository's
   caches sit at GitHub's 10 GB limit, most of it Rust's build caches, and
   GitHub then evicts the least recently used entry, which can be the
   datasets' entry. An artifact is kept for its retention period (90 days)
   whatever the cache does. `.github/actions/last-datasets` copies the
   newest unexpired one, from a run of this repository's `main`, into
   `.cache/`, file by file, leaving every file already there. It holds the
   ledger's sets, which the oracle job uploads every run. The action reads
   the artifact and publishes nothing.
3. **`oracles.py fetch`**, which downloads only what is still missing: a
   set already in `.cache/` is read there. Under CI (`CI` set) a download
   is tried five times, 5, 10, 20 and 40 seconds apart, so a server down
   for about a minute and a half is waited out. Elsewhere it is tried three times, 2
   and 4 seconds apart, so an offline run gives up within seconds
   (`tests/data.py::download_pauses`). Once a source has failed every
   round, a later set from it is named as missed without waiting again.

Where each runs:

| job | cache | on a miss | then | a set it cannot get |
|---|---|---|---|---|
| `oracles.yml`, `datasets`, a push or by hand | `oracle-data-<digest>`, the ledger's sets | the artifact | `fetch`, three rounds | fails the job by name, before any grid runs |
| `oracles.yml`, `datasets`, the weekly sweep | not read | — | `fetch` afresh first, so a set changed at its source is measured; a set an outage withholds is filled from the artifact, with a warning naming it, and fetched again | fails the job by name, before any grid runs (only when the artifact lacks it too) |
| `ci.yml`, `test`, every leg | `test-data-<OS>-<digest>`, the ledger's sets and `SUITE` (Binance's days and the day of quotes and trades, NIST's NumAcc sets) | the artifact | `fetch --suite`, one round | a warning naming it, and each test that reads it fails, naming it |

The cache is saved where it missed, and in `ci.yml` only once every set is
in hand, so an entry is never short of one. Binance's days and the quotes
are in no artifact (only the ledger's sets are published), so the CI cache
is their one copy: on a miss with Binance down, their tests fail. A test that reads a new public set adds it to
`SUITE` (or to its ledger group's `dataset`), or CI fetches it only when
the test runs and keeps no copy.

**Where the datasets were restored, a test skipped as offline fails.** A
test whose set cannot be downloaded skips, so a desk with no network still
runs the suite. In `ci.yml`'s `test` job and in each oracle grid's shard
the sets are in place before the tests start, so there such a skip means all three copies, the cache, the artifact
and `fetch`, failed. Among hundreds of tests a skip is easy to miss, and the run
would pass without its check.
The two steps set `POLARS_ONLINE_DATASETS_RESTORED=1`, and
`tests/offline_skips.py` then turns a skip whose reason begins with the
loaders' `offline: could not download ` into a failure that names the set
(a skip in a fixture shows as an error). `CI` alone does not turn it on:
the canary, the release's Polars and NumPy legs and the CPU-path runs also
run the suite under `CI` and restore nothing, so there an outage at a
source is a skip, as on a desk. A released wheel or PyPI's list of
releases, of which no job keeps a copy, skips everywhere. An oracle grid
fails here only on a set its ledger group does not name, which the
`datasets` job never fetched. Naming the set in the group's `dataset`
mends it.

### The machines and the CPU paths

Two things a CI leg does not choose decide the last bits of a numeric test.

- **The math library.** On Windows, `exp`, `ln`, `exp2`, `powf` and
  `exp_m1` come from `ucrtbase.dll`, for Rust, CPython and NumPy alike. That
  DLL is part of the operating system and changes with the Windows build.
  glibc and Apple's library are fixed per runner image. A Rust bound
  measured at 0.9 of its limit on glibc and Apple's `exp` came out at 1.7 on
  Windows Server 2025.
- **The CPU.** A runner label is a pool of machines: `windows-latest` has
  handed out AMD EPYC 7763s without AVX-512 and Intel Xeons with it, each
  with its own L3. The product kernels of faer on x86-64
  (`private-gemm-x86`),
  OpenBLAS and NumPy each choose their code from the CPU's instruction sets
  when they start, and the product kernels choose their blocking from its
  cache sizes too. `private-gemm-x86` 0.1.20 gave wrong products at widths
  that depended on the L3, so only the runners with that L3 failed.

So CI does three things about them.

| what | where | what it gives |
|---|---|---|
| two Windows builds | `ci.yml` runs every Windows leg on `windows-latest` (Server 2025) and `windows-2022` (Server 2022), at both Pythons | two `ucrtbase.dll`s for each bound to hold against |
| the machine, first | every job that can run on Windows, and every leg of `ci.yml`'s test job, starts with `.github/actions/machine-report` | the CPU model, its L2 and L3, whether it has AVX2, AVX-512F and FMA, the logical cores, the OS build and its math library (`ucrtbase.dll`'s file version, glibc's version), in the log and the job summary, so a failure names its machine |
| the CPU paths, forced | `cpu-paths.yml`, weekly and by hand, on `windows-latest` | each path in turn, instead of the ones the pool happens to hand out |

`cpu-paths.yml` has two jobs. Every leg first shows that the path it asked
for is the one in effect, and fails if not. Without that check, a green run
could be the runner's own path passing under another name.

**online-core under Intel SDE.** Intel's Software Development Emulator runs
a program as if on another chip: it answers the CPUID instruction as that
chip would and executes, or refuses, the instructions that chip has. Each
leg runs online-core's unit tests in every module whose tests run faer's
products and factorizations or compare against `crate::oracle`, the list in
the job's `SDE_TESTS`. `tests/test_cpu_paths.py` fails when a module that
does either is missing from it. The tests are a release build with the
debug build's assertions and overflow checks on, so the tests that expect an
assertion still pass. Natively the 910 tests took 6.1 seconds on 2026-10-24, on an Apple
M4 Pro at a load of 30 on its 14 cores. SDE runs code 10 to 100 times
slower, slowest where it emulates AVX-512 on a host without it.

| leg | what the tests see |
|---|---|
| Sandy Bridge (`-snb`) | AVX without AVX2 or FMA, so the product kernels fall back from `private-gemm-x86` to the `gemm` crate; a 6 MiB L3 |
| Haswell (`-hsw`) | AVX2 and FMA; a 6 MiB L3 |
| Ice Lake server (`-icx`) | AVX-512; a 38.5 MiB L3 |
| Sapphire Rapids (`-spr`) | AVX-512; a 105 MiB L3 |
| Haswell with a 2 MiB L3 | the Haswell CPUID with its L3 resized (`-cpuid_in`), the size at which `private-gemm-x86` 0.1.20 failed the order-1,500 solve test |
| Ice Lake server with a 2 MiB L3 | the same, with AVX-512 |

The proof is `scripts/cpu_probe.rs`, a one-file Rust program built in the
job. It prints what `is_x86_feature_detected!` says of SSE3, AVX, AVX2, FMA
and AVX-512F, and the cache sizes CPUID's leaf 4 reports, which is what
`private-gemm-x86` reads on Windows. `scripts/cpu_paths.py probe` runs it
on the host, then under SDE, and fails unless SDE's run reports the leg's
`expect`. SDE runs with `-update_cpuid_from_host 0`, so the cache sizes are
the definition file's, not the host's. The test run then counts its tests:
it lists them on the host, runs them under SDE, and fails unless every
listed test passed. A filter that matched nothing would otherwise pass.

SDE is free of charge for any use under the Intel Simplified Software
License, which allows it to be redistributed unmodified. The workflow takes
it through the `petarpetrovt/setup-sde` action, pinned to a commit, which
fetches Intel's kit from the action's own release. The job then checks the
tarball against the SHA-256 on Intel's download page, so what runs is
Intel's kit. The action's README says that using it means agreeing to SDE's
license, which the kit carries.

**The oracle tests with OpenBLAS's and NumPy's paths pinned.** The tests
marked `oracle` hold this library's numbers to another computation, and that
computation's last bits follow the CPU too: scipy, statsmodels,
scikit-learn, `numpy.linalg` (OpenBLAS) and the NumPy references in
`tests/reference*.py`. `tests/conftest.py` gives the mark to every test in a
module that imports one of those or calls `numpy.linalg`
(`tests/oracle_marks.py`), so the mark cannot fall behind the tests. It
leaves out a test that runs the command line, which compares the binary with
the bank rather than with an oracle. Each leg runs
`pytest -m "oracle and not soak and not oracle_grid"` with two variables set (the oracle grids run in `oracles.yml`):

| leg | `OPENBLAS_CORETYPE` | NumPy's loops |
|---|---|---|
| Prescott | SSE3 kernels | the baseline, X86_V2 (SSE4.2) |
| SandyBridge | AVX kernels | the baseline |
| Haswell, under SDE as an Intel Haswell | AVX2 and FMA kernels | up to X86_V3 (AVX2, FMA) |

`OPENBLAS_CORETYPE` picks the kernels in NumPy's copy of OpenBLAS and in
SciPy's. `NPY_DISABLE_CPU_FEATURES` takes NumPy's dispatch targets away.
NumPy 2.5 dispatches X86_V3, X86_V4, AVX512_ICL and AVX512_SPR above its
X86_V2 baseline. It refuses to disable a baseline feature, with an error,
and only warns of a name it does not dispatch, so `scripts/cpu_paths.py`
reads the wheel's own dispatch list and disables every target above the
leg's. `scripts/cpu_paths.py check` then fails unless threadpoolctl reports
the named core in both copies of OpenBLAS and NumPy reports each disabled
target off, with that warning made an error.

One core reports another name. NumPy's and SciPy's OpenBLAS on Windows are
built with five cores of kernels: Prescott, Nehalem, Sandybridge, Haswell
and SkylakeX. In such a build OpenBLAS makes each core left out an alias of
one kept, Katmai of Prescott among them, and its name lookup tests Katmai
first, so forced Prescott kernels report "Katmai". A runner's own choice is
Haswell or SkylakeX, so the check still fails if the forced core did not
take effect.

The Haswell leg runs under Intel SDE posing as an Intel Haswell. OpenBLAS
0.3.34, NumPy's copy, ends its parameter setup with a block for Zen 4 and 5. On a CPU whose CPUID names AMD, with AVX-512, a 1 MiB L2 and an L3
a multiple of 32 MiB, it increases the `k` step of a double-precision product to
512, in whatever table is set up, a forced one included. The Haswell kernel
copies its panel into a stack buffer sized for a `k` of 128 under the Windows ABI. So on such a host a forced Haswell table overruns
it, and the process dies of an access violation, as `np.cov` of 2,000 rows did twice on
an AMD EPYC 9V74. That is a defect in OpenBLAS 0.3.34, reported upstream as
OpenMathLib/OpenBLAS#6013 and #6021 and unfixed in a release as of
2026-10-25. The block reads an L2 size from `OPENBLAS_L2_SIZE` first, but
not on Windows: there OpenBLAS's `readenv_atoi` returns 0 for any variable
that is set, since `GetEnvironmentVariable` returns the value's length and
the code reads a nonzero result as failure. Under SDE the CPUID names Intel
on any host, so the block cannot fire and the forced table keeps its own
blocking everywhere. SDE follows the processes the interpreter starts,
pytest's workers among them, and the leg's check fails unless NumPy reports
AVX2 present and AVX-512 absent, which on a host with AVX-512 shows the
interpreter itself ran under SDE. Without a forced core OpenBLAS picks
Cooperlake or SkylakeX on those hosts, kernels the block suits, so the
package's users and the suite's other runs are not affected.

No leg forces AVX-512 kernels. OpenBLAS runs a forced core type without
checking the CPU (`force_coretype` in its `driver/others/dynamic.c`), so
SkylakeX on one of the pool's AMD machines would stop at an illegal
instruction. Running these tests under SDE instead would make that leg
independent of the machine, but it has not been tried. Until then the pool's
Intel machines run those kernels in `ci.yml`, and the machine report says
when.

#### The math library's last bits

**No test takes a blanket tolerance for the platform: each kind of test
treats the math library's last bits its own way.** `exp`, `ln`, `exp2` and
`powf` differ in their last bits between Apple's library, glibc and each
Windows build's `ucrtbase.dll`. online-core calls the platform's own. A
pure-Rust copy, the `libm` crate, would give the same bits everywhere, but
measured 1.4 times as slow for `exp` and `exp2` and 3.5 to 4.5 times for
`powf`, and was declined. So a test is one of three kinds:

| kind | what it compares | how it is held |
|---|---|---|
| a build against itself | one chunk against many, one thread against eight, a state saved and continued | exactly: both sides call the same library on the same machine |
| a value pinned to its bits | a digest or a frozen number from a fixed stream | a stream that calls none of those functions, such as a decay given as a literal `lam` on a clock that steps by 1, so the pin holds on every platform |
| this library against an oracle | another library, or a definition computed another way | a bound derived from both computations' rounding: a few ulps for each call to one of those functions, times the formula's amplification, measured on all three platforms and written beside the test |

**A Rust oracle comparison is held to a derived bound, from the helpers in
`crates/online-core/src/oracle.rs`.** A fixed tolerance such as `1e-9` is
too tight for an ill-conditioned case, or far too loose for a
well-conditioned one, where a kernel a hundred times past its rounding
still passes. The helpers are the standard results from Higham's
*Accuracy and Stability of Numerical Algorithms*:

| helper | the bound |
|---|---|
| `U`, `LIBM` | the unit roundoff `2⁻⁵³`, and the four units of it a call to the platform's `ln`, `exp`, `exp2` or `powf` may be off: two ulps. Apple's and glibc's are within one, and the second is left for `ucrtbase.dll`, which has not been measured to the ulp |
| `gamma(m)` | `γ_m = m·u / (1 − m·u)`, what `m` roundings in a row can move a result by, relative |
| `dot_error`, `running_dot` | the most a dot product's rounding can move it: a priori from the terms, or read off the run from its partial sums, which is smaller where the terms cancel |
| `condition` | the 2-norm condition number of a symmetric positive definite matrix, each eigenvalue moved by its own error bound the way that makes it larger |
| `forward_bound(κ, η)` | a backward-stable solve's normwise forward error, `2κη / (1 − κη)` |

**Each helper rounds its bound up, so it holds as computed.** The rounding is by
`up(m) = 1 + γ_{m+4}` for a bound evaluated in `m` roundings, so it is no
smaller than the exact bound. On a CPU without FMA, a 3 × 3 Hilbert solve
made an error equal to its bound in exact arithmetic, and the bound
evaluated in floating point came out 14 units of `u` below it.

Each test states which helpers it uses and how it counts its terms, with
the headroom it measured in a comment. About 32 comparisons sit at 0.01 to
0.31 of their bounds, or agree to the bit, so a kernel 100 times past its
rounding fails them. Eleven that a long accumulation dominates sit at 1e-4
to 1e-2 of their bounds: the worst case per step grows with the row count,
while the realized error grows with its square root. Even so, each is 10³
to 10⁶ times tighter than the fixed tolerance it replaced. A Python oracle
comparison whose two sides pass through those functions carries a bound of
the same kind, derived in a comment beside it.

## Numbers that only tighten

**Two files hold measured numbers that a change may lower but not increase
without a stated reason:** `oracles.toml` holds each oracle grid's error
against its oracle, and `budgets.toml` holds each hot path's instruction
count and each output's memory. Each number is compared with the file,
never with the previous run, so a run of small drifts adds up against one
number instead of each passing against the last.

### The oracle ledger

**Each oracle grid's actual error is committed, and it only tightens.** A
grid's assertion is the measurement with headroom, often half as much
again. A change that moves a model's error against its oracle from 0.19 to
0.27 passes a bound of 0.3, and the next change's move from 0.27 to 0.29
passes it too. Each step looks harmless against the bound, and together
they drift far. So `oracles.toml`, the ledger, holds each case's error as
measured. Every run is held to those fixed numbers, never to the run
before, as `budgets.toml` holds the speed. The user set the aim on
2026-10-25: "a way not to drift from all the oracles and data sets for
every parameter test", without more long tests in the test gate.

| file | holds |
|---|---|
| `oracles.toml` | each case's worst value and 95th-percentile value, each group's oracle libraries at the versions measured, its tests and its dataset, and the `[[raised]]` entries |
| `oracle_hashes.toml` | a hash of this library's outputs on each group's inputs, with the platform it was written on |
| `tests/ledger.py` | `ledger.guard` and `ledger.record`, the two calls a grid makes, and the hash |
| `scripts/oracles.py` | `check`, `apply`, `raises` and `rerun` |
| `tests/test_oracle_ledger.py` | the hash guard, and the rules: the ledger well formed, every grid of a registered module recording, every module with a grid registered or listed as waiting |

**A grid registers in two calls,** `ledger.guard` at module level and
`ledger.record` in the test. [Register an oracle
grid](contributing/register-an-oracle-grid.md) shows both, and how a
group's id is formed. Every module with a grid
registers. One that could not yet would be listed in `tests/ledger.py`'s
`PENDING` with the reason it waits, and the rules test fails when a module
with a grid is in neither. The rules test finds a grid by its mark on the
test, its class or a parameter, directly or through a module-level name
such as `marks=_GRID`.

**A test that records carries the `oracle_grid` mark.** `rerun` and CI run
nothing else, and a case of the ledger that CI's complete run did not
measure fails its check. So a single deterministic case whose bound has
room to drift in joins the grids' tier to be held. NIST's certified regressions are such
cases, whose bounds are the theory's and sit up to six hundred times
above the error each solve reaches, and TCPD's annotated series are too, whose F1
bounds sit 0.05 under the measured. The test gate holds their outputs by the
hash instead. A single-seed second opinion in the essentials whose bound
has room, and that no grid covers, keeps its cheap check and gains a twin
in `tests/test_second_opinion_grids.py`: the same design over 50 seeds,
against the same oracle, recorded. Of the five
single-seed bounds, two were narrower than the seeds' spread, and their
twins hold the measured spread. A third twin holds `marginal`'s windowed
lag correlations to their definition, to rounding.

**A case is judged by its gap from its ideal.** The gap is `|value -
ideal|`: the ideal is 0 for an error and 1 for a score such as an ARI, a
share or an F1. The ledger keeps the value of the seed with the largest gap
and of the seed at the nearest-rank 95th percentile, both in the test's own
units. `check` fails on five things: a case past the ledger, one to
tighten, an oracle library at another version, a case not measured, and a
case measured over fewer values than the ledger's `n`. [Update the oracle
ledger](contributing/update-the-oracle-ledger.md#act-on-each-verdict) gives
each with its remedy.

A case is also past the ledger, whole, when the test's `ideal` is not the
ledger's or its `floor=` is wider than the ledger's: the ledger's floor
alone judges, and a wider one is an increase. `apply` writes nothing of such
a case, nor of one with fewer values; it writes the test's floor only for
a case new to the ledger.

A case new to the ledger is reported and fails nothing. The grids are
deterministic, so one platform reads the same numbers on every run. A count
is held exactly: one row more of delay on the worst seed is a regression.
Two per cent of a count under 50 is less than one, but of a delay of 333
rows it is six, so a grid records a count with `exact=True`, and the case
carries a `tolerance` of 0. A case whose oracle agrees to round-off carries
a `floor`, the gap below which a difference counts as round-off; a gap
under it is judged as the floor. Each floor is about four times the
largest error measured on macOS arm64 and on two of CI's Linux x86-64
runs. Two runs are needed because GitHub's runners vary in CPU and faer
picks its SIMD kernel by the CPU at run time: the same code read some
round-off cases up to 7 times apart on two Linux runs (2026-10-26 and
2026-10-27).

**The numbers are CI's, measured on Linux x86-64.** `[meta] platform`
names the platform, and `check` judges every push there, so the whole 2%
is room for drift on the platform that judges. The ledger was first
written on a developer's Mac, and CI's first run against the same commit
(2026-10-26, an M4 Pro against GitHub's x86-64 runners) measured how far
the two platforms part. Of 1,495 cases, every statistical case but one
agreed within 1.8%, and that one, a quantile regression's distance from
statsmodels' `QuantReg` in its standard errors, by 2.9%. The round-off
cases parted by up to 3.7 times: the regularized prior's fits, the NIST
Grams and Wampler's polynomials. So the ledger was re-based on CI's
`measured.json`, and every floor increased to cover both platforms. A run on
another platform than the ledger's, such as `rerun` on a Mac, is held to
`tolerance_elsewhere` (5%) past the ledger. It writes none of its numbers,
since they are not the ones the ledger holds: a gain it sees is reported,
not applied, and a case new to the ledger gets its table without numbers,
which `apply` on CI's next `measured.json` fills. A version of the file
that moves `platform` may move each number by up to the tolerance under a
`[[rebased]]` entry, which gives the reason and a CHANGELOG line. A larger
move, or a wider floor, still needs its own `[[raised]]` entry.

**An increase needs its reason and a CHANGELOG line.** `scripts/oracles.py
raises` reads every committed version of `oracles.toml` on `main`'s first
parents (a merge's result is one version, so the dates of a branch's
commits cannot reorder them) and fails on any number that moved away from
its ideal without a `[[raised]]` entry. A wider `tolerance` or `floor`
counts too: a case with no `floor` has 0 and one with no `tolerance` the
file's, so adding either is an increase. So does a case's `ideal` moved either
way, since moving it to the measured value would read as a tightening, and
a smaller `n`. The entry names the key, the old and new value, the date,
the reason, and a `changelog` text the CHANGELOG contains. A worse error
against an oracle is a *Changed* or *Fixed* matter, so any section serves.
A number the history holds and the file no longer has -- a case or group
dropped or renamed -- fails until a `[[retired]]` entry names it (a case's
key or a group's), with the date and the reason. `apply` never writes a higher number.

**CI runs the grids and the check.** Each of `oracles.yml`'s twelve shards
records its cases and marks itself finished only when every test passed.
The last job, `the oracle ledger`, needs no project environment: the script
uses the standard library alone. It merges the shards and runs `raises`,
then `check` with `--expect-shards 12`. A missing case fails only when every
shard finished, since a failed shard already fails the run. It uploads
`measured.json` for `apply`. Every push to `main` runs every case, so a bump
of an oracle library in `uv.lock` is measured on the push that carries it.
The weekly run measures everything when nothing was pushed. It downloads
each dataset afresh rather than reading the cache, so a set that changed at
its source is measured too.

**One job fetches every dataset before the grids run.** `oracles.py fetch`
downloads every set the ledger's groups name in their `dataset` (Rdatasets,
NIST StRD, LIBSVM, Fränti's clustering sets, TCPD), retrying each, and
fails naming any it cannot get. The shards read its files and download
nothing, so one shard's failed download cannot leave its grid skipped as
offline and its cases unmeasured. The shards also print each skip's reason (`-rfEsxX`).
How that job survives an outage at a source is
[Downloaded data in CI](#downloaded-data-in-ci).

**The test gate keeps a cheap guard instead of the grids.** For each group,
`tests/test_oracle_ledger.py` calls the group's callable and hashes this
library's outputs, with the source of the tests that record the group. Each
test brings with it every module-level helper and constant it reaches by
name, its docstrings dropped. The hash is compared with
`oracle_hashes.toml`, and the group's installed oracle versions with the
ledger's. While both hold, the comparison still holds: the oracle's answer
is fixed by its pinned version, the inputs by the hash, the test by its
source. When either moves, the test fails naming the groups and the
`scripts/oracles.py rerun` command that reruns them, which [Update the
oracle ledger](contributing/update-the-oracle-ledger.md) walks through. A
hash changes only after a passing run of its oracles. Sixty-seven groups in nineteen modules hold 777 cases (counted
2026-11-08). A
`rerun` of 38 of them, the convergence grids among them, took 10.6 minutes
here (2026-10-26, eight processes, at a load of 5 to 35 on 14 cores); the
first 21 groups' grids took two. A group whose callable is cheap hashes
every seed of its grid. One that is not hashes the first seeds' or
streams' every output field, which any change to the model's arithmetic
moves as it moves the rest. The convergence groups, one a registration,
run the first seed of each scenario its grids run, at 2,000 rows.

**The guard costs seconds.** Measured 2026-10-26 on this machine, at a
load of about 6 on 14 cores: the 54 groups' callables took 5.8 s in one
process, the slowest 0.69 s, and the guard's tests 8.6 s run serially,
imports included. The first 21 groups' took 4.7 s and 6.0 s. The five
second-opinion twins added 0.37 s of callables and 0.9 s of imports. The
test gate spreads them over its workers.

**The hash survives the last bits and catches a real change.** Each float
is rounded to 24 significant bits, a float32's mantissa, about 6e-8
relative. Values under 2^-40 of their array's largest are set to 0 first,
since round-off about zero has no stable sign or exponent. Swapping the
platform's math library for the pure-Rust `libm` crate moved 102 of the
release comparison's 1,299 output fields, by at most 1.7e-11 relative
(measured 2026-10-24; [The math library's last
bits](#the-math-librarys-last-bits)). So a value lands on the other side
of a rounding boundary with a chance of about 3e-4 at that worst, and of
about 2e-8 for the usual 1e-15: across a group's hundred thousand values,
about one hash in five hundred. Rounding to bits rather than
to decimal places puts the boundaries at odd multiples of 2^-25 relative,
never at 1.0, 0.5 or a power of ten, where exact values sit. A statistical
case's gap moves by its 2% only when the outputs move by far more than the
quantum. A group whose oracle agrees to round-off cannot be hashed at a
precision another platform shares, since its errors are the last bits. It
keeps every bit (`bits=52`), so a change the ledger would see moves its
hash. Measured 2026-10-26 by changing `hmm`'s ridge and rebuilding: 1e-6
larger moved the one `bits=52` group that fits `hmm` and no other hash,
since the 24-bit group's states and means moved by less than its quantum. A ridge 1e-3 larger moved both `hmm` groups and none of the other nineteen.

**The guard enforces where its hashes were written, off CI.** That is the
system and the processor, such as `darwin-arm64`. Python's version is not
part of it: on each of CI's systems, Python 3.12, 3.13 and 3.14 gave every
group the same hash. CI runs the grids themselves on every push, so there,
and on any other platform, a moved hash is a warning in the job summary.

**One set of hashes serves every platform for the 24-bit groups, and none
for the whole ones.** On the first push with the guard (2026-10-26), every
one of the 42 groups hashed at 24 bits held on Linux x86-64, Windows
Server 2022 and 2025 and GitHub's arm64 Macs, under each Python, but one. That one was `online_diagnostics.cusum_size`
on Windows Server 2022, a value on a rounding boundary. The 17 groups hashed whole (`bits=52`) moved where the
processor differs: 12 on Linux, 8 on Windows Server 2025 and 15 on 2022,
and 3 on GitHub's arm64 Macs against the M4 Pro the hashes were written
on. Those 3 are the groups that solve a Gram through `po.gram`. A per-platform set could not keep the guard's promise, that a hash
moves only after a passing run of its oracles on that platform, since only
Linux runs the grids, and only on CI. A coarser rounding would hide what a
whole hash is for, a change in the last bits a round-off case's ledger
sees. So a whole hash is compared only where the guard enforces, and
elsewhere only the 24-bit hashes are reported.

**`scripts/quick.sh` runs the guard for what a branch changed.** A model's
source moves the groups whose id names the module, or one of its aliases,
as a word: `kmeans.rs` the `kmeans_*` groups. A source module that names
none, such as `solve.rs`, moves every group. A test file moves its own
groups.

### Budgets that only tighten

**Every push and pull request is held to a fixed number per hot path and
per output field, kept in `budgets.toml` at the repository's root.** Each
measurement is compared with its budget, never with the previous run, so a
run of small slowdowns adds up against one number instead of each passing
against the last. There are three kinds:

| kind | what is measured | where it runs | tolerance |
|---|---|---|---|
| `instructions` | the instructions one block of 1,000 rows executes, after 2,000 rows of warm-up, counted by Callgrind. The cases are the lists in `crates/online-bench/src/lib.rs`: a step of each of the 22 models at a standard width, and beside them the costlier options (`ewridge` at 200 features, `ewridge` at 50 features with 10 targets folded into a 64-row block's merge, the refit of the first rows of `huber`, `sgd`'s Huber loss and `quantile`, `huber`'s feature sets and `ridge_scale="sum"`, `sgd`'s Huber loss, `bocpd`'s full run, `ew_cov`'s lags, `marginal`'s Huber bins, and `corrchange`'s monitor and window); the clock's step on a float and on a nanosecond clock; `Bank::fit_predict` around `seqtest` and around `ewridge`, as one stream and as 1,000 groups of one row each, and around a window target that holds 1,000 rows back; five window cases through `WindowsRun::feed` (a mean, a sum and two quantiles, one within a window and one running, and a sum over 10,000 groups); and `Bank::fit_predict` of 8,192 rows of `ew_cov`'s 230 statistics on a pool of one thread | `.github/workflows/budgets.yml`, on Linux | 2% |
| `memory` | the bytes per row of every output field of every model with every output it has switched on, over 20,000 rows, from `estimated_size()` | `tests/test_budgets_memory.py`, on every CI leg | 2% |
| `rss` | the peak resident memory a standard run adds over its process's baseline, per platform, in MiB, for six runs: `ewridge` with ten targets over 200,000 rows; `ew_cov`'s 230 moments a row over 200,000; an eager `predict` of 1,000 features over 20,000 rows; a windowed `ewridge` (`window_size` 20,000) over 200,000; `bocpd` holding up to 200 runs over 50,000; and five million rows streamed from one parquet file to another. The streamed run has Polars reading one row group ahead on four threads, and the allocator giving freed pages back at once (`_RJEM_MALLOC_CONF=dirty_decay_ms:0,muzzy_decay_ms:0`, `MIMALLOC_PURGE_DELAY=0`), so its peak is the live memory. A control reads the same rows whole before the plan, and must exceed the streamed run's budget | the same file, in the extended tier: held on CI at each OS's floor Python, on Windows on `windows-latest` alone, and reported everywhere else, since the interpreter alone moved macOS's peak 18-20% between 3.12 and 3.14, the machine moved the streaming run's 23%, and Windows Server 2022 and Server 2025 read the six runs up to 8% apart on one build, so the two images would each tighten the budget below the other | 30% over: two runs of one build on one CI runner read 14-18% apart; tightened at 7.5% under |

**An instruction count can gate where a timing cannot.** Two release builds
of one source measured up to 19% apart in wall time, from where the linker
placed the code alone, and a shared runner adds its load on top. Two runs
of one binary under Callgrind count the same instructions, so the workflow
runs every case twice and fails if the two differ by more than a quarter of
the tolerance. What the 2% absorbs is the inlining an unrelated change
moves. They count the same only under a stopped wall clock, which the
counting step alone runs under (libfaketime, preloaded). Polars' hash maps draw their seed once a
process from the time since the epoch and a few addresses. The seed decides where each key lands, and so the order in which
a table grows and frees and how far the allocator searches for a block. At
the machine's clock two runs counted a bank case up to 2% apart, and turning
address randomization off did not help; with the clock stopped they agree
within 0.001%. The library keeps its seeds, since a fixed one would let
keys chosen to collide slow a bank down; only the counted process sees the
stopped clock. The stopped clock is a seed like any other, so the counts it
gives differ from the machine clock's once, and the budgets were re-applied
from its first run. A count does not see a cache miss, so `scripts/benchmark.py` still
measures throughput. A count also belongs to the compiler that built it,
the image it ran on and the harness that counted it, so `budgets.toml`
names all three in `[meta]`. Changing one re-measures every case.

**The rules, as `scripts/budgets.py` applies them:**

- A measurement over its budget by more than the tolerance fails. Find what
  made it slower or larger. If the change is worth having at that speed
  or size, record the higher budget in the same change, with a `[[raised]]` entry that gives
  the key, the old and new number, the date and the reason, and quotes a
  CHANGELOG *Performance* entry. `scripts/budgets.py raises` holds every
  increase in the file's git history to such an entry, and a wider
  tolerance counts as an increase.
- A measurement under its budget by more than the tolerance fails too, and
  names the number to tighten the budget to. The change that made a gain
  then keeps it. `scripts/budgets.py apply measured.json` writes the
  tightened budgets from a run's measurements. A resident-memory budget is
  tightened past a quarter of its tolerance, 7.5% under. Tightened only
  past the whole 30%, a budget could settle 43% above the run that set it,
  and a later run could pass 86% above that run; at 7.5% the most a later
  run passes by is 41%.
- `scripts/budgets.py raises` holds each number to the last one the file's
  history gave its key, so a budget deleted in one commit and written back
  higher in a later one is an increase like any other.
- A case with no budget is reported with its measurement and fails nothing,
  which is how a new case gets its first budget. A measuring run, a dispatch
  of the workflow with `measure` or `POLARS_ONLINE_BUDGETS=measure` for the
  tests, reports everything and fails nothing.
- The last release's own measurements are the baseline every run's drift
  is read from. A release merges its run's instruction counts and each
  OS's memory into one file, `budgets-baseline.json`, and attaches it to
  its GitHub release as an asset, so nothing is committed after a release
  to record them. Every run then reads the newest release's copy, by a URL
  that names no version (`https://github.com/hgilde/polars-online/releases/latest/download/budgets-baseline.json`),
  and reports each case's drift since. A release fails on a case that
  moved more than 10% either way unless `budgets.toml` holds a
  `[[signoff]]` for it at that version. A push only warns of it, and only
  a sign-off for a release after the baseline's silences the warning. CI
  fails when it cannot read the baseline, since a lost asset would
  otherwise switch the drift off without a word; run elsewhere,
  `scripts/budgets.py check` says it read none and holds the budgets as
  ever. `--baseline` names a local copy: the asset, or the directory
  `gh release download --pattern budgets-baseline.json` left it in.

**Valgrind does not run on macOS on Apple silicon, so the counts come from
CI.** `cargo test -p online-bench` runs every case natively on any platform
and checks that each reaches the path it is named for. On Linux, `cargo
install gungraun-runner --version 0.20.0` (the version `Cargo.lock` holds)
and then `cargo bench -p online-bench --bench instructions` prints the
counts. Elsewhere, a Linux container with Valgrind installed runs the same
two commands. The workflow keeps each run's counts as an artifact.

## Where it is thin, and what is left

**Case-insensitive file names on Windows have no test, and the newest
code sits outside the mutation scope.** Two measures find where the tests are thin. Coverage counts the
code the tests reach, and mutation testing finds code no test would notice
breaking.

### What is left

Each thin spot, why it is thin, and where it stands:

| what | why it is thin | where it stands |
|---|---|---|
| case-insensitive file names on Windows | no test covers it; file locking on a state's rewrite and long paths (`\\?\`) are tested | open since 2026-09-25 |
| a network share path, `\\server\share\...` | out of a CI runner's reach | untested; drive letters, the `\\?\` extended-length form and paths with spaces pass on both Windows legs |
| `windows.rs`, `windows_frame.rs`, `formula.rs`, `resolvers.rs` and `targets.rs`, the newest code in `crates/online-polars/src/` | outside the mutation scope, which is `online-core` and `span.rs`, as the rest of `online-polars` is; the first four postdate the coverage run of 2026-09-27 | held by the oracles and resume tests of [The window operators and formula targets](#the-window-operators-and-formula-targets), each test's docstring stating the case it holds |
| the weekly mutation pass over all of `online-core` | the first to finish, on 2026-10-04, ran at a time limit that counted most survivors as timeouts, so its 11 survivors are a floor | the limit corrected on 2026-10-04, to ten times the baseline's test run; a pass at the new limit is the next count; [Mutation survivors](#mutation-survivors) |

Two things are worth doing periodically rather than once:

| run | when | what to watch for |
|---|---|---|
| **`./scripts/mutants.sh`** | after a batch of feature work, or with `--in-diff` on a branch | a cluster of survivors in one function, which almost always means its only oracle lives in the Python suite. |
| **`./scripts/coverage.sh`** | periodically | reported and never gating |

### Measured coverage

Measured with `./scripts/coverage.sh` on 2026-09-27: **96% of the Python
package** (2,139 statements, 83 missed), and **93.9% region / 92.6% line**
of the Rust workspace. The Rust figures were 75% and 73% on 2026-08-30. It
is reported, never gating, and Rust's figure is not in CI. It sees
`cargo test` alone, so it would read `online-py` at 0%, and it would add an
instrumented build to every run. The mutation jobs, on every push and
weekly, are the sharper measure of the same thing.

**The Rust figure understates reality.** `cargo llvm-cov` only sees what
`cargo test` runs. `online-py` (0%) and much of `online-polars` are exercised
by the pytest suite through the compiled extension, which is invisible to the
Rust instrumentation. The thin spot it does reveal is
`online-cli/src/main.rs`, argument plumbing covered instead by the command line's
integration tests through `run_config`. `online-core/src/robust.rs`, the
other thin spot on 2026-08-30 at 75%, is at 98%.

### Mutation survivors

**The lines each change touches have been held to no unlisted survivor since
2026-09-24.** `mutants.yml` runs `cargo mutants` over the changed lines of
`online-core` and `online-polars/src/span.rs`, on every push to `main` and
every pull request. It fails on a survivor that
`scripts/mutants_equivalent.toml` does not list. That file holds 352
entries (2026-11-07), each an equivalent mutant -- a few
name two that share a line -- with the reason no input can tell it from
the original. The weekly pass over all of `online-core`, 11,138 mutants on
2026-10-03, reports its survivors without failing on them.

**A survivor a test could tell apart only by a difference no caller can act
on is tolerated, and listed apart** (2026-10-05).
`scripts/mutants_tolerated.toml` names 75, in 70 entries, each with its kind
and the measured size of the difference, or the loop it makes endless. The report shows them by kind and
fails on none of them, in either job. The kinds:

| kind | what it is |
|---|---|
| rounding | a difference in the last bit |
| below the computation's own error | most are `boundary`'s solver stopping a solve that has converged, under 1e-7 where its grid error is 3e-5 |
| an exact tie | one no input can be built to reach |
| a release-only guard | a guard only a release build reaches, past a `debug_assert!` |
| past the input bound | a weight past the bound on what an input may be |
| too slow to test | a path one call of which costs 30 s in a debug build |
| an endless loop | a timeout, not a survivor: an iteration that changes nothing repeats for ever, so the test hangs at the time limit, and only a bound on the iterations in the code itself would make it fail |

An entry matches as an equivalent does. It may name the `columns` it covers
or take in the line before, so it cannot cover a twin that a test catches,
on the same line or in the same function.

**A tie or a difference below the computation's own error is tolerated, not
equivalent.** An equivalent mutant computes what the original does on every
input. A tie mutant does not: at `|r| = h` exactly, `<` and `<=` send the row
to different arms. But the two arms agree there in exact arithmetic, because
the function is continuous at its own breakpoint, so the original's choice
of arm is a convention, not a fact about the answer. A test that told the two
apart would hold the code to one side of a boundary the specification leaves
open, and an implementation that took the other side would be just as
correct and fail it. And the input that reaches the tie is an exact equality
of two continuous statistics, a residual against a scale times a width,
which no stream of real data can be built to hit. A below-error mutant is the
same argument in size rather than position. The original is only correct to within its
own rounding or stopping bound, so any test must allow that bound,
and a mutant that lands inside it passes every test the original passes. A
test tight enough to fail the mutant would fail a correct implementation that
rounded differently, such as the same code on another CPU's kernels. Each
such entry gives its `kind`, the reason, and for `below-error` the measured
size of the difference against the bound, so the claim can be checked, and
the report lists them by kind rather than hiding them.

**Equivalent mutants are left alone deliberately: no test can kill them.**
Flipping `Ftrl::weight_of`'s `zz < 0.0` sign test to `<=` changes it only at
`zz == 0`, which the `|zz| <= l1` guard before it has already returned on,
so `scripts/mutants_equivalent.toml` records it with that reason.
`SlotMetrics::update_with`'s `denom > 0` against `>=` is another: a scored
row's `w > 0` keeps `denom` positive.

**`lasso.rs`, `stats.rs` and `ewridge.rs`, the files that once held most
survivors, have every mutant caught or recorded with its reason.** The
tests that close them lean on third-party oracles where there is one. Every
`ewridge` solve is held to its closed form, solved by `faer` (already a
dependency, and not the model's own Cholesky). It is held to scikit-learn's
`Ridge` too, in `tests/test_second_opinion.py`, 20 cases to 1e-8. `lasso`
without an intercept and no penalty is held to least squares through the
origin, again by `faer`. The rest hold a definition written out:

| quantity | held to |
|---|---|
| the row leverage, and each coefficient's support share | their definitions, per target and ridge |
| the window's Kish count and residual spread | the rows inside the window |
| the solve schedule | a count, row by row |
| a zero-weight copy of every row | no change, to the bit |
| a saved state | its bytes, since an in-memory `State` clones the kept factors and would hide their rebuild |
| `EwQuantile` | the exponentially weighted quantile's definition, row by row |
| `EwAutoCorr` | a regime switch, invariance under a shift and a scale, a zero co-moment before the first pair, and `same_shape`'s bounds |
| `SlotMetrics` | `weight_sum`'s definition and the strict 0.5 threshold |
