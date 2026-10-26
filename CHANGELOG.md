# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses
[semantic versioning](https://semver.org/). From 1.0, a breaking change to the
stable API takes a major version, and a minor version may also correct numbers
that were wrong, or move their last bits within rounding (a sum in another
order, another kernel), each declared with its size. Before 1.0, the minor
version carried breaking changes, and any change to the numbers a model
returns.

## [Unreleased]

### Changed

- `po.eval.window_metrics` on a `Datetime` or `Date` clock makes the windows
  Polars' `group_by_dynamic` makes of the same rows, reading `every` as
  Polars reads it: nanoseconds to hours are elapsed time, so across a
  change of a time zone's offset a `"5h"` window holds five hours and
  `"24h"` holds 24; days, weeks, months, quarters and years follow the wall
  calendar. Its windows' edges used to come from `dt.truncate` in local wall
  time, so a `"5h"` window across a change held four hours and `"24h"` was
  read as `"1d"`. A month, quarter or year (`"1mo"`, `"1q"`, `"1y"`) is now
  taken on these clocks; a `Duration` clock, which has no calendar, still
  refuses one. Naive `Datetime`, numeric and `Duration` clocks get the same
  windows as before (task 278).
- `withheld_reason` has a fifth value, `ill_conditioned`, appended after
  `below_warm_rows`, for a row predicted from a fit withheld because its
  normal equations kept no digit (task 278). Every existing name keeps its
  position, so a saved state's last row reads as it did. The noise gate
  reads such a fit's error inflation as not a number and does not name the
  row in its place.
- `po.gram.solve` with a grid of ridges rides one eigendecomposition only
  where the system's diagonal spreads by 4 or less (under `standardize`, or
  columns of like scale); elsewhere each ridge takes its own solve of the
  scaled system: on columns from 1e-3 to 1e3 a grid of fifty took 1.4 ms at
  50 columns and 8.4 ms at 200, against 0.45 and 2.7, and a single ridge
  above 0 takes less time than before (task 278).
- `init="k-means++"`, and the starts of `init="lloyd"`, are greedy
  k-means++, scikit-learn's rule: each centre after the first is the best of
  `2 + ⌊ln k⌋` candidates by the inertia it leaves (task 278). On Fränti's
  benchmark sets in 50 orders each, the worst order's inertia over the
  optimum fell from 2.19 to 1.001 on S1, from 1.29 to 1.001 on S2 and from
  1.56 / 1.51 / 1.71 to 1.30 / 1.24 / 1.50 on A1-A3 (median 1.26-1.35 to
  1.01-1.10); no S1 or S2 order now misses a cluster, where 10 and 13 did.
  Seeded centres move, and where they do not, their numbering may change.
- `hmm`'s `warm_rows` defaults to 1,000 rows (or `n_components` where that
  is more), where it was 50 (task 278): the seeding places states where the
  buffered rows are, and 50 rows hold one or two regimes. On three regimes
  lasting about 50 rows each, the worst of 50 seeds put 0.46 of the rows in
  their true state; it now puts 0.95. Each group's first 1,000 rows report
  null and the buffer holds them until seeding; a group with fewer rows never
  seeds and reports null throughout.
- Documentation: `ftrl` states how fast it follows a break under a
  half-life (task 278). AdaGrad's normalised step moves a settled
  coefficient at most about `2·alpha·√(1 − λ)` a row, however large the
  error, so a break that moves a coefficient by D takes about
  `D / (2·alpha·√(1 − λ))` rows (measured 0.011 a row against 0.012 at
  `alpha=0.1`, `half_life=200`). A larger `alpha` follows sooner and settles
  noisier: after a change of regime that moved the intercept by 46, the fit
  was 34.4, 18.4 and 2.6 off four half-lives later at `alpha` 0.1, 0.3 and 1,
  and the settled fit 0.17, 0.24 and 0.43 off. Under a half-life the fixed
  prior holds a logistic fit up to 0.41 from the decayed maximum likelihood at
  the defaults, against 0.17 at `beta=0.1, l2=0`.
- `marginal`'s `min_settled_frac` does what it says (task 278): while the
  stream is less settled than the fraction, `ModelBank.marginal()` and the
  `pair_*` columns of `closed_groups` report `corr`, `beta`, `t_stat`,
  `t_stat_serial` and the split columns as null, as `min_weight` does; the
  moments are reported either way. It was accepted and changed nothing;
  output changes only for specs that set it.
- `rcov`'s `min_weight` withholds a closed block's estimate (task 278):
  each row weighs 0 or 1, so the block's `weight_sum` counts its returns;
  below the floor `rcov` and `rcorr` are null, as for a block too short to
  estimate from, and the other fields are unchanged. It was accepted and
  changed nothing; output changes only for specs that set it.
- `ewridge(session_shrink=f)`'s blend no longer forces a solve and
  restarts the solve schedule at a session change (task 278): it moves the
  moments only, and the next scheduled solve reads them, as the docstring
  promised. The output is continuous in `f` and `f=0` is no `session_shrink`
  to the bit; before, `f=1e-12` differed from `f=0` by up to 0.48 in a
  prediction (1.43 under `solve_every=50`, 3,000 rows), now by 1.3e-12. The
  rows from a session's start to the next scheduled solve are predicted from
  the fit before the blend; `max_rows_between_solves=1` makes every row after
  the session's first read it.
- A number or a duration given as `session_gap` to `po.stream.with_windows`
  is deprecated, because it has no effect there: every window ends at a
  session change, so no window measures the clock's step across one (task
  278). A session no longer needs a `session_gap` in `with_windows`, and a
  number raises a `PolarsOnlineDeprecationWarning` until the next major
  version. `session_gap="reset"` is kept: it discards, rather than cuts, the
  windows looking ahead that a session change leaves open, and changes
  nothing else. A windows state saved with a numeric `session_gap` resumes
  the call with any other number or none. A spec's `session_gap`, which sets
  its decay across the change, is unchanged, and `like=spec` lends it without
  a warning.
- The weekly CPU-paths workflow's forced-Haswell oracle leg runs under
  Intel SDE posing as an Intel Haswell (task 273). OpenBLAS 0.3.34 resizes
  a forced core's blocking on AMD Zen 4 and 5 hosts with AVX-512, which
  overran the Haswell kernel's stack buffer on Windows
  (OpenMathLib/OpenBLAS#6013 and #6021); `OPENBLAS_L2_SIZE` cannot prevent
  it there, since OpenBLAS's Windows build reads any variable that is set as
  0. Under SDE the CPU reports Intel on every host, so the block never
  fires; proven on an AMD EPYC 9V74 runner, all 2,345 oracle tests passing
  with NumPy reporting no AVX-512. The `OPENBLAS_L2_SIZE` setting is
  removed.

### Deprecated

- `theta` given beside `preavg_rows` to `po.spec.rcov` is deprecated: it has
  no effect, because `preavg_rows` fixes the pre-averaging window, which is
  all `theta` sets. It raises a `PolarsOnlineDeprecationWarning` naming the
  caller's line and is accepted and ignored until the next major version
  (task 278).

### Fixed

- A weight or target of finite magnitude above `1e100` is read as missing,
  as the README says, instead of being refused (task 278, D-WB): a weight
  of `-1e101` was refused as "a negative value" on every model that takes a
  weight, `±1e101` by `rcov` as neither 0 nor 1, a target of `±1e101` under
  `strict_binary` and `-1e101` under `loss="poisson"`, and a `bocpd`
  `hazard_col` value of `-1e101` as not above 1 rather than falling back to
  `hazard`. Each check now reads only usable values; a usable value that
  breaks its rule is still refused by row, at `±1e100` itself too.
- `po.eval.diebold_mariano` and `po.eval.clark_west` with `half_life`, and
  `po.eval.seqtest`, refuse a `name` the frame already has, naming the
  column; Polars' `with_columns` used to replace that column -- the target,
  or a spec's output -- without a word. With `half_life`, a frame holding a
  column `dm_<target>` or `cw_<target>` had that column's values returned as
  the statistic and lost the column, and a `name` spelled like one of the
  functions' working columns returned no result column at all; both now give
  the right statistic and keep the frame's columns (task 278).
- `deco` with `blocks` reported a finite `loglik` on rows whose block
  levels no correlation matrix holds, or holds only to rounding -- warm-up
  rows where the levels rest on one or two rows. The value was meaningless
  (-4.9e5 to -8.8e18) and depended on the order the blocks were listed in
  (-2.3e8 in one order, -2.6e12 in the other on one stream). `loglik` is now
  null where the `K x K` system the density solves factorizes only with a
  shift on its diagonal, or where its condition passes the point at which
  the solve's forward-error bound reaches 1 (`3K·γ_{3K+1}·‖H⁻¹‖₁ ≥ 1`).
  Every other row's `loglik` is unchanged to the bit (218,872 rows over 183
  test streams); one block is unaffected (task 278).
- The refusal of an `ew_class` label listed among its features calls it the
  label, not the target.
- A negative `block_rows` or `preavg_rows` given to `po.spec.rcov` is
  refused with the real floor -- 1 for `block_rows`, 2 (3 under `psd=False`)
  for `preavg_rows` -- rather than ">= 0".
- `po.spec.corrchange`'s refusal of `n_perm` above 2^20 says why: the draws
  are held to take their quantile, and each is a pass over both blocks.
- The docs now state that `ew_cov`'s `lag_corr` reads 0.0, not null, until a
  group's `l`-th learned row; that `audit` records the clock as fed, so a
  numeric `session_gap` changes nothing it counts; that `corrchange`'s window
  kind cannot flag on its first report after the windows fill or refill;
  that `hmm` with `transition_prior=0` and `learn=True` can lock out an
  extremely separated state; and, in `po.stream.with_windows`'s `like=`,
  that a skipped row is one whose feature or weight is null, NaN, infinite
  or past 1e100 (task 278).
- The windowed CUSUM and CUSUM of squares (`emit_breaks` with a memory)
  are divided by their null's share of the plain variance as far as their
  sums have filled, not by the filled share from the first row (task 278,
  D-B5). A young sum holds little of what the fit's mean spreads a level
  over, so its variance is nearly the plain sum's, and it read too wide
  until its sums had filled their memory. On clean Gaussian streams beside
  `ewridge` at a half-life of 200, the share of the first 150 rows after the
  diagnostics' readiness past 1.96 fell from 6.5%-34% (`cusum`, memories 50
  to 800) and 11%-16% (`cusum_sq`) to 6.1%-6.4% and 6.2%-6.7%; 300 to 600
  rows in, from up to 23% and 15% to 5.0%-5.9%; filled, it is about 5% as
  before. `sgd`'s CUSUM of squares read 15% there and now 6.1%. The share
  is read from the sums' own weights, so no saved state changes; values
  move only while the sums are young, by up to a factor of 0.71 on the
  first row at the default memory, and once filled equal the old values to
  the bit with unit weights and within 1e-6 under uneven weights.
- `ewridge` withholds a fit whose normal equations are too nearly singular
  to keep a digit, and says so (task 278). Past a condition of `1/u` a
  Cholesky factorization does not always fail -- rounding can leave the
  matrix positive definite -- and the coefficients then came out wrong by
  more than their own size with nothing said: two features equal to within
  `1e-11` of their spread gave slopes of -364.1 and 363.1 for a truth of 1
  and -2, silently on 19 of 40 seeds, and NIST's Filip kept no correct
  digit. Each factorization that needs no jitter now estimates `‖H⁻¹‖₁` of
  the system scaled by its diagonal (Hager and Higham's estimator, LAPACK's
  `dlacn2`, from the factor in `O(k²)`, behind a comparison-matrix bound
  that settles most systems at one solve's cost); where the solve's
  forward-error bound reaches 1, `3 k γ_{3k+1} ‖H⁻¹‖₁ ≥ 1`, the fit is
  withheld -- `coef`, `support_coef` and the predictions null -- and counted
  in `solve_failures`. NIST's solvable designs sit seven orders of magnitude
  inside the limit, and every other solve is unchanged to the bit.
- `po.gram.solve(standardize=False)` is accurate whatever the features'
  units (task 278): it diagonalized the raw Gram with `eigh`, whose
  eigenvalues are accurate only relative to the largest, so features from
  1e-10 to 1e10 put the coefficients off by up to 8.9 times their size and
  NIST's Longley kept 6.4 digits; the system is now scaled by its diagonal
  first (9.4e-14 at most, 12.0 digits on Longley).
- A negative `jitter`, `noise_stride` or `iv_stride` given to
  `po.spec.rcov` is refused as needing to be at least 1, the real floor; it
  was refused as needing to be at least 0, and then 0 was refused as well
  (task 278).
- `lasso`'s `tol` was in the target's units (task 278): the coordinate
  descent stopped when no standardized coefficient moved by `tol`, and those
  coefficients are in the target's units, so a target near 1e-9 met the
  default after a sweep with no descent counted unconverged (0.20 of the
  target's spread off at y×2^-30, 1e-3 at 2^-20) and a target in the
  millions could not meet it above its rounding. A sweep now ends the descent
  when no coefficient moves by `tol` times the target's spread (its standard
  deviation, or its root mean square without an intercept): a target and
  penalties scaled by a power of two give the same fit scaled, to the bit,
  with the same `solve_failures`. `po.gram.lasso_path` follows the same rule.
  Outputs move only where the target's spread is not 1, by at most 7.8e-10
  in a prediction or coefficient on the release probe. A `lasso` state saved
  by 1.0-1.3 loads and goes on within 1e-9 of this build's own, not to the
  bit.
- `po.gram.lasso_path` returned a penalty whose descent ran out of its
  `max_iter` sweeps with nothing to say so (at a condition of 4e8 and a
  penalty of 0.001, 5 and 1,000 sweeps both left a KKT violation of 7.8e-9
  against the rule's 2e-10); it now raises a `ReadinessWarning` naming each
  such penalty (task 278).
- `kmeans` and `hmm`: the default seeding, `init="lloyd"`, runs Lloyd's
  passes from each of its ten starts until no row changes centre, at most
  300 passes, where it stopped after ten wherever they had reached (task
  278). A start stopped at the cap is counted in `solve_failures`, which
  `kmeans` now reports. On eight overlapping blobs in two dimensions (500
  rows), 38 of 50 seeds kept centres that were not a fixed point, up to 0.12
  noise standard deviations off, and nothing said so. A model seeded from the
  same rows may now number its clusters or states differently even where it
  finds the same centres, so read labels by their centres, not their numbers.
  A `kmeans` state saved before it seeded seeds by the new rule after
  loading; one saved after seeding goes on exactly as before.
- These values are refused, each with the reason in the message (task
  278): `min_weight=inf`, which withheld every row for good while the model
  kept learning; a `marginal` `bin_edges` that gives no feature an edge, such
  as `[[]]` (one feature without edges beside others that have them is still
  taken, and reports a single bin); a `kmeans` `dead_frac` of 1 or more,
  which called a centre dead at every check; `emit_averaged` with one slot
  per target, as `emit_selected` already was; a `corrchange` `bandwidth`
  above `window_rows`, which adds no lag and drives the long-run variance
  toward 0 (on independent noise the statistic's median went from 1.8 at 20
  to 12.5 at 1,000). A state saved with any of these values still loads.
- `bocpd`'s refusal of `warm_rows` beside both priors names the keywords the
  builder takes, `mean_prior` and `covariance_prior` (task 278).
- `drift_action="reset"` starts the model over exactly (task 278): from
  the row after the one that fired, every output field equals what a fresh
  bank fed those rows returns. That row read the clock step since the row
  that fired, which a fresh stream's first row does not have, so the rebuilt
  model decayed by it: `rls`'s prior moved predictions by up to 2.5e-3 at
  `delta=1` and 8.0e-3 at `delta=100` (3,000 rows, `half_life=200`),
  `kalman` with a fixed process noise by up to 4.7e-3, every model's
  `settled_frac` counted the extra step, and the diagnostics began folding
  residuals a row early (`calibration_slope` up to 0.86 apart on `kalman`
  with a mean-reverting slot). `learned_clock` on that row showed the clock
  of the row that fired; it is null, as documented.
- `po.stream.with_windows` refuses a `session_gap` without `session` and a
  `restart_after_step_back` without `clock`, as a spec does; each was
  accepted before and did nothing (task 278).

### Performance

- `deco`'s blocked `loglik` check costs about 500 to 1,100 instructions a
  row with 2 to 5 blocks, 3-5% of the row; one block pays nothing (task 278).
- Reading the windowed CUSUMs' young share costs about 60-90 ns a slot and
  row under `emit_breaks`, about 3% of such a row (task 278, D-B5).
- `ewridge`'s condition estimate costs 4.3% at 5 features (135.9 -> 141.7
  ns a row), 7.6% at 20 (288.9 -> 311.0) and 1.8% for the 50-feature,
  10-target blocked case solving every 400 rows (556.1 -> 565.9), measured
  natively, best of seven runs interleaved with the previous build (task
  278).
- Seeding with `init="lloyd"` at 500 rows, 8 centres and 400 features
  takes 56 ms on separated blobs (from 97) and 91 ms on overlapping ones
  (from 98); `init="k-means++"` takes 3.6 ms there, from 1.9 (task 278).

## [1.3.0] — 2026-10-24

### Added

- **`huber_scale` on `po.spec.huber` and `po.spec.sgd`** (`"mad"`, the
  default, or `"rms"`; on `sgd` it applies to the Huber loss alone and is
  refused by name beside another loss), and **`huber_scale_half_life`**,
  the scale's own memory, a number or a duration and a clock parameter,
  defaulting to the spec's `half_life` (task 271). A robust scale treats
  every row after a level shift as an outlier until its median has moved:
  with a fit half-life of 500 rows and a +20 shift, the mean absolute
  error over rows 200-400 after the shift is 19.5 at the fit's half-life
  and 14.6 at `huber_scale_half_life=100`, against least squares' 13.3.
- **`po.spec.huber(feature_sets=...)`**: named subsets of the features,
  each fitted from the same per-target sums and reported as
  `pred_<t>__<set>`, as `ewridge`'s are (task 272). A row's Huber weight and
  the residual scale come from the largest set's residual: the sets share
  one set of sums per target, which holds a row at one weight. The largest
  set is the plain Huber fit on its features to the bit when it holds every
  feature, or under the same explicit `min_weight` (the default counts the
  spec's whole feature list, so a narrower largest set starts predicting
  later, and the two fits part until the decay forgets the first rows: by
  up to 0.17 early and 1e-4 after 1,200 rows at `half_life=300` on the test
  stream); a smaller set is the weighted least squares fit at those
  weights. Sets of 5, 10 and 20 features cost about
  60% more a row than one fit (443 against 272 ns at 20 features). A
  `huber` with no sets fills its data shares only when they are read; its
  row is about 1.5% dearer than before sets and the emitted scale (252
  against 249 ns at 20 features, a solve every 20 rows).
- **`po.spec.huber(ridge_scale="sum")`**: the decaying prior on the sum
  scale, as for `ewridge`; at `huber_delta=inf` it gives `rls`'s
  predictions within 1e-9, so it is the Huber form of `rls` (task 272).
- **`po.spec.huber(emit_huber_scale=True)`** writes `huber_scale_<t>`, the
  scale each row's Huber cut is drawn in, read before the row's own
  residual enters it, so it can serve as a causal weight or scoring scale
  (task 272). It is null on a row the model does not read, one with a null
  or unusable feature.
- **`marginal(bin_stat="huber")`**: each bin's location is Huber's
  M-estimate of location, learned in one pass (task 272). A row counts for
  `w·min(1, c/|e|)` against the bin's location before it, with the cut
  `c = bin_huber_delta · σ_t` (default 1.345) and `σ_t` the target's
  exponentially weighted MAD about its own robust location, so one spike
  moves a thin bin by at most the cut times its share, where it moves the
  mean by its whole size times its share. Against statsmodels' converged
  M-location per bin on 20,000 rows in 16 bins it lands within 0.001,
  0.002 and 0.03 of the target's scale at 1%, 5% and 12% spikes, where the
  mean is off by 0.45, 1.5 and 3. `bin_n` stays the rows' weight;
  `bin_var_y` is the clipped mean square and `split_gain` reads both. The
  default `"mean"` is unchanged to the bit, as is `bin_huber_delta=inf`.
  Under `"huber"` a cell keeps 40 bytes instead of 32 and a row runs at
  0.75 to 0.9 times the mean's speed.
- **`po.ewm_quantile`**: the exponentially weighted quantile of a column
  over the window operators' windows, looking back or ahead with
  `reverse=True`, with or without a `window_size` (task 272). Each row is
  weighed as `po.ewm_sum` weighs it, and the result is within 0.78% of
  numpy's weighted `inverted_cdf` quantile. A hard window holds 24 bytes a
  row per operator: a row leaving restores the weights it kept, with no
  subtraction. The median of `|r|` over 0.6745 is a robust scale (the
  EW-MAD); WINDOWS.md gives the recipe and the rule for a scale of a target
  used as a weight (leave the label out until it matures). It costs 1.7 to
  1.9 times `ewm_mean` a row. The `with_windows` state is version 10 and
  still reads version 9.
- `docs/GRAM.md`, *Huber regression on a Gram*: Huber's algorithm H on one
  shared Gram, a Polars query each pass, reaching statsmodels'
  `RLM(scale_est="mad")` within 1.2e-6 on the slope in ten passes and
  3e-7 from twenty on (the 0.6745 the recipe rounds Φ⁻¹(3/4) to); why weighting only
  the cross-moments is not weighted least squares (slope −59.4 where the
  fit is 2.04), and why to start from a fit that has the target's level.
  GBDT.md names LightGBM's piecewise `huber` and XGBoost's smooth
  pseudo-Huber `reg:pseudohubererror`, with their slope set from the MAD
  (task 272).
- **`--output-layout flat`** on the command line (TOML
  `output_layout = "flat"`) writes each spec's output as `<spec>.<field>`
  columns, as a CSV already gets, in parquet, IPC and NDJSON; the default
  stays one struct column per spec (task 270). Polars' parquet writer
  encodes a struct's fields at several times the cost of plain columns: an
  `ew_cov` over 20 series (233 fields) ran through the command line at
  7,697 ns a row writing the struct to parquet and 1,274 flat. The frame
  `fit_predict` docstring, the README and RUNNER.md say to unnest a wide
  spec before `sink_parquet` (2,785 against 365 ns a row for the same 233
  fields).
- **`OnlineModel::step_into`**, a step that writes into a `Step` the caller
  keeps; its default goes through `step` (task 270).
- A weekly workflow checks `Cargo.lock` and `uv.lock` against OSV's
  published advisories, failing on a vulnerability in a crate the wheel or
  the command line links and reporting the rest (task 270).
- CI's test matrix runs every Windows leg on Windows Server 2022 as well
  as Server 2025, at Python 3.12 and 3.14: on Windows `exp`, `log` and
  `pow` come from `ucrtbase.dll`, which ships with the operating system, so
  a second Windows build holds each numeric bound to a second math library.
  Every CI job that can run on Windows, and every leg of the test matrix,
  first prints the machine it ran on (CPU model, L2 and L3, AVX2, AVX-512F
  and FMA, logical cores, OS build, and `ucrtbase.dll`'s or glibc's
  version), so a failure names its machine (task 273).
- A weekly workflow, `cpu-paths.yml`, tests the CPU paths one at a time
  rather than whichever machine the runner pool hands out: online-core's
  numeric tests (910 in the modules that run faer's products or compare
  with the oracle) under Intel SDE posing as Sandy Bridge, Haswell, Ice
  Lake server and Sapphire Rapids, and as Haswell and Ice Lake with a 2 MiB
  L3, the size at which `private-gemm-x86` 0.1.20 gave wrong products; and
  the Python tests marked `oracle` with OpenBLAS's kernels pinned to
  Prescott, SandyBridge and Haswell and NumPy's loops capped at its
  baseline or X86_V3. Each leg shows that the path it asked for took effect
  before its tests count (task 273). Its forced-OpenBLAS legs set
  `OPENBLAS_L2_SIZE=2048`: OpenBLAS 0.3.34 resizes a forced core's blocking
  on Zen 4 and 5 hosts with AVX-512, which overruns the Haswell kernel's
  stack buffer on Windows (OpenMathLib/OpenBLAS#6013 and #6021); a run
  without a forced core is unaffected.
  Its Prescott leg expects the name OpenBLAS reports for forced Prescott
  kernels, "Katmai": NumPy's and SciPy's Windows builds compile five cores
  and alias the rest, and OpenBLAS's name lookup reaches the Katmai alias
  first. On Windows the peak-memory budgets are held on `windows-latest`
  alone and reported on `windows-2022`, whose machines read the same build
  up to 8% apart (task 273).

### Changed

- **`huber` and `sgd(loss="huber")` draw their cut in a robust scale by
  default** (task 271). The scale `σ` in `w · min(1, δσ/|e|)` and in
  `clamp(p − y, ±δσ)` is now the exponentially weighted median of `|e|`
  over `Φ⁻¹(3/4) = 0.6745` (`huber_scale="mad"`), read per target from a
  quantile sketch within 0.78% of the exact weighted median. It was the
  exponentially weighted root mean square, which one wild row sets and
  which, without decay, a fit's first residuals (the target's level while
  the intercept is near 0) widen for good. On `y = 1000 + 2x + 0.5ε`,
  20,000 rows, no decay, +50 on a share of the rows with `x > 0` (median of
  five seeds), `huber`'s slope at 1% / 5% / 12.5% outliers is 2.008 / 2.040
  / 2.128, where it was 2.057 / 2.676 / 4.885; least squares gives 2.415 /
  4.050 / 6.975 and statsmodels' converged `RLM` with the MAD scale 2.008 /
  2.039 / 2.124. On MODELS.md's example the slope is 2.02 (it was 2.18;
  truth 2). On the release probe 376 of 384 `huber` predictions move, by
  at most 0.21, with a median relative move of 0.9%, and 378 of 384
  `sgd(loss="huber")` predictions, by at most 0.055, with a median
  relative move of 5.1%. **`huber_scale="rms"`
  restores 1.2.0's numbers to the bit**, and a state saved by 1.0 to 1.2
  loads as `"rms"`, the scale it was fitted under, and continues to the
  bit. `quantile` is unchanged. After a level shift the robust scale holds
  the fit back until its median has crossed, which costs `sgd` more than
  `huber`: with a fit half-life of 500 rows and a +20 shift, `sgd`'s mean
  absolute error over rows 200-400 after the shift is 17.3 under the new
  default against 2.45 under `"rms"` (`huber`: 19.5 against 15.1);
  `huber_scale_half_life` at a fifth of `half_life` brings `sgd` to 3.0.
  This changes a default in a minor release,
  by the maintainer's decision, against the policy's row for a changed
  default; the way back is the one setting above.
- `huber`'s docs state how a delayed label is weighed: under `embargo` or
  a look-ahead formula target a held row is weighed when it is learned, by
  the fit of every row before it, which is the undelayed run's fit (`coef`
  under an embargo of E rows is the undelayed `coef` from E rows earlier,
  to the bit); `ewridge`'s and `rls`' docs point to `huber` for a Huber
  loss (task 272).
- **The command line refuses a URL in any path** (input, output,
  `load_state`, `save_state`, `closed_groups`), from a flag or the TOML,
  naming it before anything is read, under `--dry-run` too (task 270). An
  `https://` input used to go to Polars' cloud reader, which retries an
  unreachable host ten times over up to three minutes. A `file://` URL is
  read as the local path it names, and a URL in the TOML is no longer
  joined under the config's directory.
- **`ewridge`'s `gram_block_rows` budget counts a block per target under
  `target_gaps="own_rows"`** (task 270): each pattern of missing rows holds
  its own block, so ten targets missing on different rows held 2.5 GiB
  where the refusal named 256 MiB. The refusal now says so and suggests
  `target_gaps="pairwise"`; a spec with many always-present targets is
  refused at a smaller block.
- `ew_cov` with both `partial_corr` and `mahal` reads `mahal` from the
  precision `partial_corr` already computed; on that path `mahal` moves
  within rounding, about 1e-15 relative (1.03e-15 on the test stream,
  which holds it to 4e-15) (task 270).
- **`sgd(schedule="inv_scaling")`'s rate is `learning_rate / (1 +
  λ·weight_sum)^power`**, with λ the row's decay (task 270). It read the
  weight before the row's decay, so a zero-weight row in between moved the
  next row's rate, by up to 1.6e-2 relative at a half-life of 1. Numbers
  move under any finite half-life and not without decay: on a 20,000-row
  stream the largest move in a prediction or coefficient, relative to 1 +
  its value, is 9.5e-2 at half-life 1 (median about 1e-3), 3.1e-3 at 50 and
  3.3e-4 at 500; the rate itself moves by 22%, 0.69% and 0.069% at the
  weight ceiling. No release-probe spec uses `inv_scaling`.
- `corrchange`'s `monitor` statistic is computed from one pass of running
  sums per pair instead of two passes per prefix; it moves in its last
  bits, by about 1e-13 relative over 60,000-row streams (5.3e-14 at a
  level of 1e4), and on the 400-row release probe by 1 to 4 ulp (at most
  6.3e-16 relative) on 6 of the 8 values it emits, with no flag or change
  date different (task 270).
- Dependencies: rustls 0.23.46, rustls-webpki 0.103.16 and tokio 1.54.1,
  which fix published advisories; the dev group's pillow 13.0.0 and the
  docs group's certifi 2026.10.23 (task 270).
- `kmeans`' refusal of `n_clusters` past 65,536 states the default
  seeding's cost, and the docs give its one-off cost per group (93 ms at
  400 features and 8 centres; 0.2 ms under `init="k-means++"`) (task 270).
- The state schema is 63 (the robust scale's sketch and settings, and a
  window's running products of its decay factors); 1.x states load and
  continue to the bit, except a windowed one's rows after a snapshot taken
  since its load, which continue within 1e-12.
- A window's saved state keeps the running products of its decay factors:
  12 bytes more per snapshot per decay (23 for `lasso`), and 48 bytes more
  per snapshot in memory; no measurable change in speed (task 275).
- `lasso`'s coordinate descent subtracts each row's terms in two runs
  either side of the coordinate, with no bound checked per term: 28% fewer
  instructions on a 20-feature `lasso` (Linux x86-64), the same bits. A
  bank's per-row copy of the predictions into its output columns no longer
  reloads the column on every slot: 2.3% fewer instructions on a
  230-statistic `ew_cov` bank (task 275).
- Tests: six Python tests whose results moved when the math library or a
  linear-algebra kernel changed in its last bits now hold derived bounds
  (the quantile fit's nudge ratio by the band system's condition number,
  the API snapshot's resolved defaults at 12 significant figures, `deco`'s
  density by the correlation matrix's condition number, the `bocpd` clock
  oracle summed with one `logsumexp`, `half_life` against `lam`, and the
  standardized Kalman's `coef·x = pred`). Found by running the whole suite
  with the extension's `exp`, `exp2`, `ln` and `powf`, and NumPy's, `math`'s
  and the linear algebra's results, moved by 1 or 2 ulp (task 273).

### Fixed

- **`sgd(loss="huber")` on a target with a level and no decay was the
  squared loss** (task 271): its scale held the first residuals, the
  level's size, for good, so no later row was clipped. At 5% outliers its
  slope was 3.751, the squared loss's to the bit; under the new default it
  is 2.016. The docs no longer say the cut "narrows as [the first
  residuals] leave the mean", which held only under decay.
- **`kalman` no longer predicts 0.0 before a target's prior is sized**
  (task 270). Standardized without `obs_var`, a row past `min_weight`
  before the first three observed rows had sized the prior was predicted
  as `0.0` whatever the target's level (with a weight column or a lowered
  `min_weight`), and that 0 seeded the residual variance with the level
  squared: `sigma` was 106 at row 2 for a noise of 0.5, and 1.84 at row
  1,000 where it is now 0.97. Those rows are withheld as
  `below_warm_rows`, and the residual variance starts at the first
  prediction of a sized filter.
- **`po.gram.coef_stats` and `po.gram.solve_subsets` no longer count a
  column of no variance as a degree of freedom** (task 270): `sigma2` was
  inflated by `(n−p)/(n−p−1)` (1.8% at 60 rows, 0.2% at 500) where
  `standardize` dropped a constant; `sigma2`, `se` and `t` are now
  statsmodels' OLS on the columns kept, to 1e-12.
- **The robust standard errors (`se_coef_hc0`, `se_coef_hac`) no longer
  read 0.0 after a row whose score overflows** (task 270): such a row now
  only ages the sums, and a variance that is not a number is NaN, not 0;
  `kalman`'s intercept variance keeps a NaN the same way.
- **The core clock refuses a NaN or infinite clock value, or a step past
  the largest double**, instead of taking it as a step of `gap_cap` on
  that row and the next and leaving every later exact stamp NaN (task
  270). The bank already refused these; `with_windows` now refuses an
  overflowing step by row, naming both values.
- **`sgd` under `standardize` with a constraint or `schedule="adagrad"`,
  and `pa` with a constraint and `standardize`, no longer turn NaN for good
  after a zero-weight row about 1,074 half-lives on** (task 270): the
  feature scaler kept a history of subnormal weight, the next row's scale
  came out near 1e-161 and every coefficient became NaN. A scaler history
  whose share of the weight after the row is below the smallest normal
  double is now forgotten, as one the decay takes to exactly 0 already was;
  no output moves anywhere else (112,320 probe cases over gaps of 0 to
  1,075 half-lives).
- **`break_wald` waits for twice as many learned rows as the coefficients
  it compares** (task 270), from a stream's first row and again after a
  gap that leaves the old rows below the rounding of the new ones (a
  history of W unit rows and a gap of 26.5 + log₂ W half-lives or more),
  where the distance was otherwise read off a variance of the new rows'
  rank alone. The variance of the difference between the
  fast and slow fits has rank at most `n − k` over `n` rows, so it is
  singular below `2k` rows; rounding used to decide whether that singular
  variance factorized, and what the distance was when it did, so
  multiplying every weight by one constant, even by 2, could turn
  `break_wald` on or off on a row. On the release probe's 400-row stream
  this turns 1-3 early rows per spec from a number into null and changes no
  other value; `break_wald` is now invariant to the weights' scale, as
  every other output of these models is.
- Documentation: `bocpd`'s full covariance costs `O(runs·d³)` a row, not
  `O(runs·d²)`; `predict` reads no `coef_every`/`max_rows_between_coefs`;
  a wide eager `predict` peaks at the width of its input (164 MiB at 1,000
  features × 20,000 rows, 23 MiB in 2,000-row chunks); the README's install
  size is 1.2.0's (24-30 MB download, 70-98 MB installed); the release policy records `huber_scale`'s default
  moving in a minor release, with the reason and the way back; the MAD
  scale costs 14-17 ns per target per row in `huber` and 20-32 in
  `sgd(loss="huber")`; IPC writes a struct at about the cost of flat
  columns; the suite's test counts are 2026-10-24's (task 270).
- Documentation: the README's throughput table is measured again (five
  runs of `scripts/benchmark.py`, each row the best of the five, at a load
  average of 2.8-3.6); it had been a run three releases old, and 19 of its
  35 rows had moved by more than its stated 11% noise (`corrchange`'s
  monitor 11× faster, its window kind 2.6×, `ftrl` +30%, `ew_cov` +26%,
  `bocpd`'s diagonal −21%, `micro` −16%, `ewridge` at 5 features −14%);
  `quantile`, `holt`, `marginal` and `audit` join it. Release step 2 now
  refreshes the table, and a test holds its date to no older than the
  previous release (task 270).
- Documentation: `docs/PERFORMANCE.md`'s headline gives the bank's share
  of a row by width in place of "a bank's row is the model's own
  arithmetic": one stream on one thread, the model's step against the
  bank, `seqtest` 13 → 70 ns; at 5 features `sgd` 41 → 130, `ewridge` 118
  → 235, `ew_cov` 57 → 153; at 50 features `sgd` 1.9×, `ewridge` 1.2×,
  `ew_cov` (1,328 fields) 3.1×; at 500 features `sgd` 1.7×, `ewridge`
  1.0-1.2×, `ew_cov` 4.8×. The bank adds about 60 to 170 ns a row, plus
  about 2 ns a feature read and 2 to 4 ns a field written (task 270).
- The "min_weight has not been met" notice no longer advises lowering
  `min_weight` below a figure such as 5.7e-17 (task 273). Where a stream's
  weight is still its first row's alone, the projected settled weight
  `(W − w₁(1 − s))/s` is 0, and the computed value was its rounding,
  positive or negative by the platform's math library (+5.7e-17 with
  `exp2` one ulp high, at or below 0 on macOS). A difference within
  `8·2⁻⁵³` times the rows read times `W + w₁` now reads exactly 0, so that
  notice gives no figure on any platform and `summary()`'s
  `weight_sum_settled` reads 0 there; a real ceiling as small as 7.7e-9
  keeps its figure, and on the release probe's workload no
  `weight_sum_settled` value or notice moved.
- `po.eval.merge_sums`' docstring quoted one random stream's accuracy
  ("within 3e-15"): over eight streams 2 chunks merge within 6e-15 and
  1,000 within 9e-15 of the exact sums, and the test holds them to a bound
  derived from the parts' rounding and any summation order (task 273).
- A windowed target or feature whose variance inside the window is within
  the subtraction's rounding now reads 0 also when the variance from before
  the window has decayed below the smallest normal double (task 273): the
  guard's `64 ε` of the terms underflowed to 0 there, so a held target over
  a long window read a variance of 5e-324.
- A window's snapshot is decayed forward by the product of the per-row
  decay factors the model's moments took, rather than by one factor of the
  summed clock (task 275). The two differed by the per-row factor's
  rounding times the rows in the window, so on a platform whose `exp2`
  rounds a half-life's per-row factor differently, a target held at one
  value over a window of a few hundred rows could read a variance near
  1e-15 instead of 0. Windowed outputs move in their last bits, by at most
  3.3e-13 relative on the release probe. A windowed state saved by 1.2.0
  loads and its first read is the one it saved; once a snapshot taken
  after the load is the window's edge it continues within 1e-12, not to
  the bit, since such a state never recorded the per-row factors.
- The memory budget for `marginal`'s Huber bins (`bin_huber_delta`) holds
  every target's scale sketch to what it counts (task 272). The sketch could
  allocate 11,168 bytes per target where the budget counted 8,488, its
  buckets growing by doubling past the 1,025 its band spans; and while half
  or more of a target's weight was on deviations of exactly 0 it kept no
  band at all, so deviations of `1e-300` and `1e300` spanned 127,563
  buckets, about a megabyte. A bounded sketch now allocates exactly the
  buckets it spans, and a median of 0 keeps the band reaching up from the
  smallest value, the bucket the median steps to when it leaves 0; the
  budget counts the 8,384 bytes a sketch takes at its most. The same bound
  holds for `huber`'s and `sgd`'s MAD scales in every state. No output
  changes.

- **A panic in the MAD scale's quantile sketch, introduced during this
  release's development and absent from 1.2.0**, in `po.spec.huber` under
  the default `huber_scale="mad"` and in `po.spec.marginal(bin_stat="huber")`
  (task 272 (p)): while half or more of the residual weight sat at exactly
  0, a residual arriving more than the sketch's band below every other one,
  as residuals near 1e-12 beside 1e100 weights do, raised `PanicException:
  range start index ... out of range`. The sketch now folds every bucket
  above the band into the band's end bucket: the weight is kept, the sketch
  stays within its 1,025 buckets, and no output on any other stream moves.
- Documentation: GRAM.md's comparison of Huber's Proposal 2 with the MAD
  scale measured Proposal 2 at statsmodels' default cut, `d = 2.5`, not
  the `c = 1.345` the text defines. At `HuberScale(d=1.345)` (statsmodels
  0.15.0, seeds 0-9) Proposal 2 holds nearly as well as the MAD to a fifth
  of the rows wild (2.129 against 2.124 at 12.4%, 2.392 against 2.317 at
  20%) and breaks between a fifth and a quarter (11.825 against 2.889 at
  25%) (task 273).
- Tests: the test-only oracle's rounding-error bounds hold as computed:
  `γ` is rounded up, each bound is lifted by its own evaluation's rounding,
  and `|A⁻¹|` is bounded through the inverse's residual rather than read as
  the computed inverse. A CPU without FMA made an LU error equal its old
  bound, which floating point then put 14 units of roundoff below the error
  (task 273).

### Performance

- **1.3.0 against 1.2.0, measured before the release**: on an Apple M4 Pro
  at a load of 1.8 to 4.5, 1.2.0's wheel from PyPI, 1.2.0 built here from
  its tag and this build, five interleaved rounds of `scripts/benchmark.py`
  at 200,000 rows: `corrchange`'s monitor 10.3 times as fast and its window
  kind 2.4 times, `deco` 24-34% faster, `audit` 25%, `kmeans` 18-19%,
  `seqtest` 11%, `pa` and the raw `sgd` 6-8%, and `huber` 6% slower, the
  robust scale's cost (`huber_scale="rms"` is 1.2.0's row). The cheapest
  rows read 2 to 5% slower: `marginal` 5%, `holt` and `ew_cov` with lags 4%,
  `ew_class` diagonal and `micro` 2%. For all but `ew_cov`'s lags, a
  bisection over the commits between the two puts it at the clock's refusal
  of a step that is not a number, about 3 ns a row. That cost is in wall
  time alone: a float clock's row counts 103 instructions on Linux x86-64
  against 1.2.0's 112. `scripts/wide_benchmark.py`'s `ewridge` at 200 to
  1,000 features is within 1.5% of 1.2.0 built here on every schedule, and
  `lasso` runs 14-21% faster there (task 275).

- **A bank of many small groups runs faster** (task 270): each spec keeps
  its run buffers between chunks, each stream builds its diagnostics'
  configuration once, a spec with no window skips the window pre-pass, and
  a run's buffers move between threads as one pointer. With one row per
  group per chunk on one thread, `seqtest` 1,219 → 779 ns a row at 1,000
  groups and 1,990 → 1,308 at 10,000; `ewridge` with 5 features 1,863 →
  1,298 and 2,556 → 1,800. Ten rows per group run 1.03-1.24× faster; a
  single stream is unchanged.
- **Paths whose cost grew with the rows they held now cost the same each
  row** (task 270), every value the same to the bit unless stated: a
  windowed `ewm_mean`, `ewm_var` or `ewm_std` during a stream's first
  window and after each restart, and on a sparse input, whose reads
  scanned the null rows between its values (one value in 5,000: 1,264 →
  184 ns a row) (at a window of 100,000 rows the mean
  39,317 → 172 ns a row, the variance 73,954 → 182); a target that is a
  window expression, or any `embargo` (10,000 rows held: 21,526 → 523 ns a
  row, the same rows learned in the same order and the same saved bytes);
  `emit_autocorr` with a long `resid_autocorr_lag` (lag 100,000: about
  9,000-12,000 → under 340 ns a row); `corrchange`'s `monitor`, which
  closed a span in time quadratic in the span (4 features, `window_rows =
  4000`: 17,206 → 269 ns a row; 10 features 179,744 → 944); and
  `corrchange`'s `window` kind, which copied its two windows every row and
  the pooled rows every permutation draw (286 → about 6 allocations a row,
  5,842 → 2,307 ns a row at `window_rows = 100`).
- **The model's step no longer allocates its prediction vector** in
  `seqtest`, `pa`, `sgd`, `ewridge`, `ew_cov` and `kmeans` (task 270): the bank
  keeps one output buffer per model instance. Through the wheel, ns a row:
  `seqtest` 79 → 70, `pa` (20 features) 131 → 121, `ewridge` 5 features
  209 → 199, 20 features 343 → 334, `kmeans` (20 features, 8 clusters)
  323 → 309, `sgd` (20 features) 135 → 126 (standardized 190 → 179,
  Huber loss 153 → 144). Outputs unchanged to the bit.
- **The command line has its own allocator**, jemalloc off Windows and
  mimalloc on it, as py-polars uses; the wheel is unchanged (task 270). On
  the system allocator the binary ran 1.23-1.25× slower a row on one
  `ewridge` stream, 1.89-1.93× slower with 1,000 one-row groups a chunk and
  1.77-1.80× slower at 10,000 groups. Its jemalloc runs on one arena
  (`narenas:1`), so the peak memory stays bounded: on jemalloc's default
  of four arenas a core each arena kept the pages freed in it for its own
  threads, and the peak grew with the rows, 0.94 / 1.36 / 1.48 GB at 3M /
  12M / 24M rows of the memory benchmark's file, where it is now 0.75 /
  0.85 / 0.86 GB (0.42 / 0.43 / 0.44 GB with
  `POLARS_ROW_GROUP_PREFETCH_SIZE=1`), at the same wall and CPU time;
  `_RJEM_MALLOC_CONF` still overrides it. Windows' mimalloc was not
  measured (task 270 (o)).
- A bank decides whether to keep a finished run's output buffers for the
  next chunk from its flat buffers' sizes alone, without walking every
  row's coefficient lists: 99.5 µs a 100,000-row run before, 2 ns now
  (task 270).
- `audit`'s distinct-value counters are a hash table, so a column of mostly
  new values costs the same at any `distinct_cap`: 75 / 214 / 3,183 ns a
  row became 59 / 52 / 58 at caps 256 / 4,096 / 65,536; a saved state's
  bytes are unchanged; the counters take about 14 KiB a column at the
  defaults, against 10 (task 270).
- The robust Huber scale's sketch keeps its buckets within 16 octaves of
  the median: at most 9.3 KB per target (13.9 before the bound; one
  residual of `1e200` stretched an unbounded sketch to about 680 KB), its
  outputs unchanged on every probe and test stream (task 271). Every
  exponentially weighted quantile sketch (the Huber scale,
  `resid_quantiles`, `ew_cov`'s `mahal_quantiles`) updates a level's
  weight without a data-dependent branch: 10-12 → 7-9.5 ns a row for the
  sketch, answers bit-identical. The MAD scale costs about 14-32 ns per
  target per row over `"rms"` (17 for `huber` and 32 for
  `sgd(loss="huber")` at 20 features and 10 targets, where `sgd`'s row
  is cheap enough for that to be 52% of it).
- `kmeans` searches for a learned row's nearest centre once instead of
  twice (163 → 124 ns a row at 8 centres and 4 features, 629 → 409 at 20
  and 20); `marginal` skips the pair work on a row of weight 0 (5,167 →
  991 ns per such row at 10,000 features); `refresh_time` with `group` no
  longer builds a string per row (10.8 → 12.3 M rows/s); `po.eval`'s scores
  carry only the columns they read (peak 2.0 → 0.7 GiB on a 290 MiB frame
  with 43 columns and 8 slots); fewer allocations per row in `ew_cov` with
  lags (5 → 0 through the bank's kept step), `deco` (19 → 14), `hmm` (29 → 26), `lasso` (6 → 2) and the
  feature-health and specification diagnostics (1.86 → 0) (task 270).
- `po.gbdt.lightgbm_dataset`'s streamed route hands LightGBM each batch's
  labels and weights with its rows rather than holding the whole dataset's
  until the end: 16M weighted rows peaked 125 MB lower; the trees are
  bit-identical (task 270).
- **A clock's row costs less** (task 270): the check that refuses a clock
  value that is not a number took the clock's difference a second time on
  every row; it now takes it once, and a temporal clock's step forward
  within its session takes a short cut held to the general step to the
  bit. On Linux x86-64, counted under Valgrind, a float clock's row is 103
  instructions (112 before the check, 120 with its first form) and a
  temporal clock's 218 (394 before); on aarch64 96 and 167 (117 and 275
  before). In wall time on an M-series Mac a float clock's row is 3.3 ns
  where it was 5.0, and a temporal clock's 5.3 ns where it was 10.5.

## [1.2.0] — 2026-10-17

### Fixed

- **`closed_groups="….csv"` now works from `df.online.fit_predict`,
  `lf.online.fit_predict` and `ModelBank.fit_predict_batches` / `fit`**
  (task 269). Every closed row carries a list column (`coef` at least,
  twelve for a Gram kind), which Polars' CSV writer refuses, so the call
  failed after the whole fit, and the plan form then saved no
  `save_state`. The CSV now goes through the command line's own writer,
  each list as JSON text that `str.json_decode` reads back, so the file is
  the command line's byte for byte. A CSV that cannot carry a column is
  refused before the fit, and a sidecar that cannot be written still
  leaves the state saved before its error is raised.
- **A column named twice in `features=` or in a job's `features` is
  refused by name** (task 269) by `po.gram.solve_subsets`, `solve`,
  `lasso_path`, `lars_path`, `lars_paths`, `coef_stats`, `vif` and the
  native fits. It used to be solved as two regressors, a singular system,
  with one coefficient written over the other: `solve_subsets` reported 0
  for the column beside the fit's own `r2`, and `lars_path` reported 0
  for a column it listed as active. `po.gram.subset` still takes a column
  twice.
- **The command line no longer refuses a config holding a TOML float `inf`
  or `nan` beside a deprecated model key** ("invalid type: null"; task
  269). Deprecated names are now forwarded over the parsed TOML document
  rather than through JSON, which has no infinity, and an error after a
  forwarded key names and quotes the file's own line.
- **`threads=` takes numpy's integers** (anything `operator.index` reads)
  on `ModelBank`, the frame and query surfaces, `po.gram` and
  `po.thread_pool_size` (not yet `po.gbdt`), and a hand-built job's `target` and `features` take
  them too, as its `subset` and `draw` already did (task 269). A bool is
  refused, and a bool given as a column or target position, once read as
  0 or 1, is now refused as well. The command line refuses a `threads`
  that is not a positive integer (`-1`, `2.5`, `"auto"`, `true`) with the
  same message `0` and the Python surfaces give.
- **A blocked `pairwise` fit now forms each target's feature means over
  a block from that target's own rows** (task 269). Under
  `target_gaps="pairwise"` and `gram_block_rows`, it took the block's
  whole sum and subtracted the rows the target missed. That difference
  carried the whole block's rounding, magnified by the block's weight over
  the target's: for a target present on one light row of a 256-row block
  it was 300 to 500,000 last-place units off, where a row at a time is
  exact. Blocked `pairwise` fits move by at most 3e-16 of a target's
  largest cross-moment, and feature means by at most 1.8e-14 relative.
  `own_rows`, the default, is unchanged to the bit, as is every fit a row
  at a time. With ten targets that each have gaps, a blocked fit takes 1.0%
  to 4.6% longer at 100 to 400 features (one thread, measured
  interleaved); without gaps it takes no longer.
- **A blocked Gram, and the targets folded into it, no longer turn NaN for
  good** when the accumulated weight's square overflows (weights past
  about 1.3e154) or leaves the normal range (below about 1.5e-154), or
  when the weight is subnormal (task 269). Each now merges as a row at a
  time does; every other merge keeps its bits.
- **`po.gram.solve_subsets` (and `jobs=`) and `po.gram.coef_stats` give
  the other columns their standard errors when one fitted column has no
  variance** (task 269), such as a constant that `standardize` drops, or a
  constant in a Gram pooled within parts and fitted through the origin.
  Before, every `se` and `t` was NaN. The constant's own stay NaN; the
  others are least squares' on the columns kept.
- **A query with a formula target, or a `with_windows` state, writes the
  same state bytes however its input was chunked** (task 269). The rows
  held for an open window were written as one record batch per chunk fed.
  No file written before changes how it loads.
- **The tooling around the tests and the budgets** (task 269):
  `scripts/budgets.py raises` holds each budget to the last number the
  file's history gave it, so a budget deleted and written back higher
  needs a `[[raised]]` entry like any other raise; a `[[signoff]]` must
  name a budget that exists, and on a push only a sign-off for a release
  after the last one silences the drift warning; the peak-memory
  measurements are written out only by the CI leg that holds their
  budget; `scripts/slot.sh take` refuses a branch another worktree has
  checked out; `scripts/clear_caches.sh --worktrees` run from inside a
  slot keeps the other slots' builds and the main checkout's; `scripts/gate.sh` builds with
  `--locked`, as CI does, and in the extended tier treats an empty
  `serial` selection as a failure; and every `subprocess` call in the
  tests and scripts that decodes a child's output names UTF-8 rather than
  the locale's encoding (cp1252 on Windows).

### Changed

- **The peak-memory (RSS) budgets tighten whenever a run reads more than
  7.5% under them**, a quarter of their 30% tolerance, instead of only
  past the whole 30% (task 269). Before, a budget could sit 43% above the
  run that set it and a later run could pass 86% above that run; now the
  most is 41%. All nine RSS budgets were tightened: eight to the 1.1.0
  release run's measurements at the oldest supported Python (Linux
  streaming 218.9 to 154.2 MiB, Windows ten-target `ewridge` 110.9 to
  80.1 MiB, the wide `ew_cov` on macOS and Windows about 7%), and the
  wide `ew_cov` on Linux to 416.8 MiB, which the first 1.2.0 release run
  read 7.5% under its budget of 450.7; none was raised.
- **The `gram_threads` docstring says it caps the update of the `k x k`
  matrix alone** (task 269): a solve's factorization can use every thread
  of the pool, so only `threads=` bounds how many threads multiply. The
  corrections to 1.1.0's, 1.0.1's and 1.0.0's notes are made in place
  (the Windows commit's remedy, a four-thread instruction case, the
  Polars floor of 1.0.0, a pair of transposed figures).

### Performance

- **1.2.0 against 1.1.0, measured before the release**: every one of
  `scripts/compare_release.py`'s 297 output fields over 31 specs is
  identical to 1.1.0's, bit for bit (the number moves under *Fixed* are in
  a blocked `pairwise` fit with gaps and in `po.gram`, which it does not
  run; each entry states its own size). On an Apple M4 Pro at a load of
  3.5 to 7.7, both builds compiled here, six interleaved rounds of
  `scripts/benchmark.py` at 200,000 rows: 37 of its 38 configurations
  have a median within 2% of 1.1.0's, and `scripts/wide_benchmark.py`'s
  twelve, `ewridge` and `lasso` at 200 to 1,000 features, have medians
  from 0.98 to 1.01. `rcov`'s kernel estimator over groups of 1,000 rows
  read 0.91 of 1.1.0 in every round, though none of its code changed. It
  comes from where the compiler lays out code in the binary: with the
  symbols kept, everything else unchanged, this build runs it at 0.99 of a
  build without the numeric fixes above, inside the runs' scatter.

## [1.1.0] — 2026-10-15

### Added

- **`po.gram.solve_subsets(..., jobs=...)`: each fit on columns of its
  own, in one call.** A job is a mapping of `"subset"`, `"target"`,
  `"draw"` (a column of `target_signs`), `"features"` and, for a path,
  `"penalty_weights"`; `jobs` is a list of them or a polars DataFrame of
  them, and the call returns one fit per job, in order. A screen per
  subset, per target and per draw, as complementary-pairs stability
  selection runs, no longer needs a `merge`, a `subset` and a `lars_path`
  per job from Python: each subset is still merged once, on every column
  its jobs read, each draw re-pools its cross-moments once, and every job
  runs on the pool. A job's path is `lars_path`'s on its columns of its
  merged subset, to the bit, and a ridge job the call's on its own columns
  and target. Measured on an M4 Pro at a load near 7: 100 subsets of 19 of
  38 Grams, 11 draws and 10 targets, 11,000 paths each to 10 active
  columns on its own 400 of 1,500 columns, took 11 ms a subset, against
  115 ms for the loop (409 ms on 1.0.1, see *Performance*); at 150 of 500
  columns, 3.3 ms against 24 ms (65 ms on 1.0.1). `docs/GRAM.md`, *A
  screen of its own for each path*, has a recipe.

### Fixed

- **On x86, wide fits could be wrong in 1.0.0 and 1.0.1, at widths set
  by the CPU's L3 cache: upgrade if you fit hundreds of features or more
  on x86** (tasks 266 to 268). Apple Silicon is not affected. faer's x86
  product kernel, `private-gemm-x86` 0.1.20 (since yanked by its author),
  sizes its blocks from the L3 cache, and once a chunk was clamped to fit
  that cache it could stop being a multiple of the chunk inside it, so the
  blocks no longer covered the product; three tests of a triangular
  product compared against the wrong size as well. Version 0.1.22, which
  this release builds with and requires, fixes both. Measured with a
  ridge solve against an LU solve of the same system, the kernel's L3
  forced on one Intel Linux machine: with an L3 of 8 MiB, a common
  desktop size, solves at 3,700 and 3,900 columns were wrong by 3e-4 to
  1e-3 relative, where rounding allows 4e-5; at 4 MiB, at 1,800 to 1,920
  and 2,600 to 2,650 columns; at 2 MiB, at 700 and 1,350 to 1,540 columns,
  by up to 1e-2. The failures come in bands of width, not past one size,
  and larger caches move them wider: at the machine's own 33 MiB nothing
  failed at the widths tried, up to 4,097. Every one of those 29 cases
  passes with 0.1.22. On GitHub's Windows runners with an AMD EPYC 7763,
  whose CPUID reports no L3 (the kernel then takes 4 x L2 = 2 MiB), the
  1,500-column solve failed on four runners of four with 0.1.20 and passes
  on all of them with 0.1.22, as on every other runner tried, Windows and
  Linux, AMD and Intel. The products in question are those of the wide
  models (`ewridge`, `ew_cov` and the rest of the Gram's users, blocked or
  not) and of the `po.gram` functions: the factorizations, solves and
  Gram products past 16 x 16 x 16. Where the block sizes changed, an x86
  result's last bits can move against 1.0.1.

### Known issues

- **On Windows on x86, each thread that runs one of faer's larger matrix
  products commits memory of about twice the CPU's L3 cache**, as in 1.0.0
  and 1.0.1 (tasks 258, 260, 263 and 264). faer's x86 product kernel keeps
  a packing buffer of that size on every thread that has multiplied, made
  at the thread's first product and held until it exits, and Windows
  commits all of it: a wide Gram (`ewridge`, `ew_cov` and the `po.gram`
  functions) fitted on many threads can exhaust the commit limit. Measured
  on a GitHub Windows runner (Intel Xeon, 260 MiB of L3), one 1,000-feature
  `ewridge` fit with `gram_block_rows=256` on four `gram_threads` peaked at
  2,382 MiB of commit (1.0.1's wheel, the same kernel). Fewer threads
  bound it: `threads=` sizes the pool, and every thread of the pool can
  multiply, because a solve's factorization spreads over the whole pool
  and in a bank of many groups every thread runs a group. So the commit is
  at most about twice the L3 times `threads`. `gram_threads=` caps the
  update of the `k x k` matrix alone, and bounds nothing here. The kernel
  stays on every x86 build because it is the faster product:
  a build without it held that fit to 374 MiB, but ran wide `ewridge`
  12-26% slower there on the default solve schedule and 29-63% slower with
  `gram_block_rows`. On Linux and macOS the buffer is address space
  reserved, of which only the pages a product touches are backed; a host
  that refuses to overcommit memory (`vm.overcommit_memory=2`) counts the
  whole reservation.

### Changed

- **A blocked `ewridge`'s target cross-moments move in their last bits**
  (task 257), and so do its coefficients, predictions and the Gram
  export's `cross_moments`, `cross_centered` and (under `pairwise`)
  `means_by_target`, by the merge above: each target's sums are now taken
  over the block at once, in another order. Measured over three specs of
  40 features and four targets, 3,000 rows, against the build before: the
  outputs within `2e-12` relative, `cross_moments` within `2e-16` of their
  largest entry and `cross_centered` within `6e-15`. The weights, the
  target means and variances and every model without `gram_block_rows` are
  the same bits. Each entry is held to an exactly rounded oracle of its
  definition, within the rounding bound of a sum in any order, at every
  block size and thread count, with gaps, zero weights and decays; and a
  target's sums are the same bits at every `gram_threads`, on every
  platform, and whichever other targets share the block. Under `own_rows`
  a target's feature means are its Gram's to the bit again. The state
  file's schema is 62: a blocked fit saved mid-block holds each target's
  values on the held rows. A file from 1.0 loads and goes on; a blocked one
  saved mid-block takes the rows after it into the merge, its targets
  having taken the held rows already.
- **A ridge fit of `po.gram.solve_subsets` may move in its last bits, on
  one target or several, and is its own whichever targets share the call**
  (task 256 (b)). faer's blocked solve of every target at once let one
  target's last bits depend on the others beside it. The targets are now
  solved together by forward and back substitution in which each target's
  arithmetic is the textbook recurrence's, one rounded product and one
  rounded difference per term in index order, whatever targets run beside
  it: the factor is read once for a tile of targets, and the targets share
  its loads without sharing a sum. A fit's `coef`, `se`, `t`, `resid_var`,
  `sigma2` and `r2` are the same bits alone or beside any set of targets,
  in any order, at any thread count. Measured against 1.0.1 on three Grams
  of 12 to 300 columns, ridge 0 and 0.5, with and without `standardize`:
  `coef` within `6e-15` of its largest entry and `se` within `6e-14`. Each
  target's `b' A b` and the intercept's `m' A⁻¹ m` are summed beside the
  other targets' the same way, the bits of the sums they replace. **Faster
  on several targets:** measured on an Apple M4 Pro at a load of 6 to 20
  (other work on the machine), one subset of a Gram fitted on 2,000 rows,
  ridge 1, best of 1 to 15 calls in 2 or 3 interleaved rounds of each
  build, 50 targets took 1.00 ms against 1.88 on 1.0.1 at 200 columns, 6.8
  against 13.6 ms at 500, 109 against 195 ms at 1,500, 309 against 513 ms
  at 3,000, 1.21 against 1.77 s at 5,000 and 7.6 against 9.6 s at 10,000;
  10 targets 0.62 against 0.78 ms at 200 and 7.0 against 7.4 s at 10,000;
  one target the same as 1.0.1 (0.52 against 0.53 ms at 200, 6.85 against
  6.91 s at 10,000), its time the factorizations'. The same at `threads=1`
  and on the default pool. From 3,000 columns those 2,000 rows leave no
  degrees of freedom, so no standard errors; with them, in one process
  (`examples/ridge_solve_bench.rs`, which CI's Benchmark runs), 50 targets
  took 0.46 to 0.67 of 1.0.1's time at 500 to 5,000 columns, 10 targets
  0.77 to 0.92, one target 0.99 to 1.00.

### Performance

- **1.1.0 against 1.0.1, measured before the release**: every one of
  `scripts/compare_release.py`'s 297 output fields over 31 specs is
  identical to 1.0.1's, bit for bit (the changes under *Changed* are in
  models and calls it does not run, and it runs on Apple Silicon, which
  the x86 kernel's fix does not reach). On an Apple M4 Pro at a load of
  2.1 to 4.8, both builds compiled here, five interleaved rounds of
  `scripts/benchmark.py` at 200,000 rows: 18 of its 38 configurations are
  faster beyond the runs' scatter, by 1-8% (`sgd` 7.8%, `pa` 6.8%,
  `ew_cov` with lags 5.9%, `ewridge` at 20 features about 3%), and
  `corrchange`'s monitor at four features is 3.1% slower (464 against 479
  thousand rows a second); the rest are within the scatter. Three rounds
  of `scripts/wide_benchmark.py`, the wide specs `benchmark.py` does not
  reach: `ewridge` at 200 features 1-3% faster, at 500 features 1-3%
  slower, at 1,000 features the same (blocked 38.7-39.1 against
  38.5-39.4 thousand rows a second), `lasso` at 200 and 500 features
  0.99.

- **A blocked `ewridge`'s targets wait for the block's merge with the Gram**
  (task 257). Under `gram_block_rows`, each target's cross-moments with the
  features were updated row by row beside the blocked product, about
  0.35 ns per feature and row each at 200 features; they are now merged
  once per block, from the same pass over the block's rows as the Gram's
  product. Measured on an Apple M4 Pro at a load of 3 to 4, one fit of
  200 features, a 256-row block, never solved, ns per feature and row on
  one core, three rounds of each build in turn: 1, 2, 5 and 10 targets took
  7.1-7.4, 7.4-7.7, 8.5-8.7 and 10.2-10.6 before and 6.9-7.1, 7.1-7.3,
  7.4-7.6 and 8.1-8.3 after; with each target null on one row in ten,
  10.0-10.3 and 8.5-8.8 at 10 targets. A target now costs about 0.13 ns
  (0.18 with gaps), beside the product's 5. Nothing changes without
  `gram_block_rows`.
- **A bank of several specs transposes each spec's chunk on its own
  thread** (task 257). Each spec's features are copied from columns into
  rows once a chunk, and that copy ran on every thread of the pool for
  every spec at once, waiting on memory: CPU time that bought no wall time.
  A lone spec keeps the parallel copy. Together with the entry above,
  measured on the same machine over five specs of 200 float32 features
  each, 20,000 rows, ten targets null on one row in ten, never solved:
  13.4-13.7 ns of CPU per feature and row against 17.0-20.4 before, at
  2.9-3.2 ns of wall time against 3.3-3.8; one target 11.1-12.5 against
  13.9-17.5. The outputs are the same bits.
- **A stream copies its features into rows a run at a time, in cache**
  (task 257 (b)). The bank used to transpose each chunk's feature columns
  into one block of rows before any stream read them: 160 MB at 20,000
  rows of 1,000 features, written to memory and read back. Each stream now
  copies only the rows of the run it is about to learn (about a thousand
  rows at 200 features) into a buffer of its own, reused run after run,
  and reads them back from cache; a float64 column with no nulls is read
  in place. Measured on an Apple M4 Pro at a load of 2 to 4, three rounds
  of each build in turn, five specs of 200 float32 features over 20,000
  rows, ten targets, never solved: 11.1-11.6 ns of CPU per feature and row
  on one thread against 11.7-12.4, and 12.8-13.4 on the default pool
  against 13.9-14.5, at 2.9-3.2 ns of wall time against 3.1-3.5. One spec
  of 1,000 features: the same wall time, 1 to 7 ns of CPU less on the
  default pool. Peak memory of an eager fit falls by the block: those five
  specs 395 -> 231 MiB, one `sgd` of 1,000 float64 features over 20,000
  rows 165 -> 14 MiB, ten-target `ewridge` over 200,000 rows 145-152 ->
  122 MiB. `scripts/benchmark.py`'s rows are the same within their scatter
  (median ratio 1.002), and every output and saved state is the same bits.
- **A float32 or integer feature column is no longer cast to float64 when
  a chunk is read** (task 257 (c)). A column only features read stays in
  the type the frame holds it in (float32, or an integer of 8 to 64 bits),
  and a float64 column's nulls stay in its validity rather than in a filled
  copy. The copy of a run's rows widens each value as it writes it, with
  the conversion polars' own cast makes, so every value a model reads is
  the same bits. That cast was the last copy of a whole chunk's features:
  160 MB a chunk at 20,000 rows of 1,000 float32 features. Measured on an
  Apple M4 Pro, three runs of each build in turn: the peak memory of five
  specs of 200 float32 features over 20,000 rows, `fit`, falls from 230 MiB
  to 67, and one `sgd` of 1,000 float32 features from 179 MiB to 14, as on
  float64. At a load of 4 to 6 the same five specs take 2.08-2.14 ns of
  wall time per feature and row on the default pool against 2.44-2.54 at
  one target, and 2.55-2.66 against 2.89-3.01 at ten, with 0.1-0.4 ns less
  CPU; one spec of 1,000 features is the same within its scatter. The copy
  of a run now reads up to 448 features eight columns at a time through a
  small buffer: 0.31-0.67 ns a value at 4 to 200 features, where the copy
  row by row took 0.32-0.95. Past that width it copies plain float64 or
  float32 columns row by row, as before. A column another role also reads
  as a number, and a boolean, decimal or 128-bit integer column, is still
  cast once. `scripts/benchmark.py`'s rows are the same within their
  scatter (median ratio 1.001).
- **A Gram's columns are looked up by name in a table**, where each name
  was a scan of the column list: `po.gram.subset` of 400 of 1,500 columns
  took 3.2 ms and takes 0.56 ms, and every `features=` resolves likewise.


## [1.0.1] — 2026-10-13

### Known issues

- **On Windows on x86, each thread that runs one of faer's larger matrix
  products commits memory of about twice the CPU's L3 cache.** faer's x86
  product keeps a packing buffer of that size per thread, made the first
  time a thread multiplies; on GitHub's Windows runners it is 960 MiB a
  thread. Linux backs only the pages touched, and Apple Silicon takes
  another path, so it shows only on Windows: a wide Gram (`ewridge`,
  `ew_cov` and the `po.gram` functions) fitted on many threads can exhaust
  the commit limit. Present since 1.0.0. Until a release fixes it, a
  smaller pool (`threads=`) bounds it: a solve's factorization spreads over
  every thread of the pool, whatever `gram_threads=` says, so that option,
  which caps the update of the `k x k` matrix alone, does not bound it.

### Performance

- **1.0.1 against 1.0.0, measured before the release**: every one of
  `scripts/compare_release.py`'s 297 output fields over 31 specs is
  identical to 1.0.0's, bit for bit. `scripts/benchmark.py` at 200,000
  rows, seven rounds of each build in turn on an Apple M4 Pro at load 3.4
  to 5.1, both builds compiled here from their tags: `ew_cov` at twenty
  features is 6.8% faster and `corrchange`'s monitor 3.3%, and every other
  configuration is within the runs' scatter. Against 1.0.0's wheel from
  PyPI, built on another machine, a few configurations read 2-7% slower,
  which the two builds of one place did not: the difference is the build,
  not the code.
- **A wide eager `fit_predict` holds its output once, not twice, for a
  spec with no group or a single group.** Each run of rows is copied into
  the output as it ends, on another thread while the next run is learned,
  rather than every run waiting for the end of the call. A 230-statistic
  `ew_cov` over 200,000 rows returns 393 MiB and now peaks at 443-446 MiB
  above where it started, against 803-831 before, and runs 8% faster. A
  spec of many groups is unchanged. The output is the same to the bit.
  At every `threads` count from 1 to 8 it is as fast as before or faster:
  that `ew_cov` 16-29% faster, narrow models over a million rows 2-13%
  faster, with their peak down 29-63% (`seqtest` 277-297
  MiB to 98-118). A test holds every field to the bit at each count.
- **The peak-memory (RSS) budgets are recorded, at a 30% tolerance.** Two
  runs of one build on one CI runner read 14-18% apart, so 15% failed one
  of them either way. They are held on CI at each OS's oldest supported
  Python, where they are measured.
- **The streaming memory budget measures this library, not Polars'
  read-ahead.** It streams a million rows with Polars reading one row group
  ahead on four threads. With Polars' default read-ahead, a plain scan to a
  sink with no model peaked at 227 MiB for a million rows and 473 for three
  million, and at 95 and 110 with the read-ahead at one row group.

### Fixed

- **`po.gbdt.lightgbm_dataset(engine="python", in_memory=False)` refuses
  a source that gives more rows on its second pass than it was counted
  at.** LightGBM reads only the counted rows of the dataset's row
  sequence, so the extra rows went unseen. The native engine already
  refused them, and so did every other pass.
- **`ModelBank(spec)` and `ModelBank("name")` say to pass a list.** A dict
  or a string was read as a list of its keys or its characters, and refused
  as `invalid type: string "name", expected struct Spec`. It is still a
  `ValueError`.
- **A closed-groups file written as CSV no longer fails partway through a
  run.** For a model that keeps a Gram (`ewridge`, `lasso`, `ew_cov`), its
  `columns` field is a list of names, which CSV refused only after rows
  had been learned. It is written as JSON text now, as a numeric list such
  as `coef` already was, and `str.json_decode(pl.List(pl.String))` reads
  it back.

### Removed

- **`Windows.held()` and `Windows.queued()` on the native extension's window class.**
  Nothing in the package called them; they were not part of the public API.

### Tests and documents

- **The FFI leak checks each run in a fresh interpreter.** One failed on
  both Ubuntu legs of a CI run and passed on their re-run. A Linux
  measurement found no leak: the same loop in a plain script held within
  0.5 MB over 3,120 iterations, from a baseline near 73 MB, where the check
  inside the suite read a process near 615 MB. Each check now runs as its
  own pytest process, and fails if that process runs anything but the one
  test.
- **A test for every path no test reached.** Coverage of the Rust crates
  under their own tests and the whole Python suite together, and of the
  package with child interpreters traced, found 1,583 Rust lines and 157
  package statements that nothing ran. Tests now reach every one that can
  be reached, each asserting what the path produces; what is left is
  listed with the reason it cannot run.
- **Exact where nothing excuses a tolerance.** In the clock, `bocpd`'s run
  structure and the clustering models' compensated sums, 33 comparisons
  that tolerated a difference are now exact, each shown to fail on a value
  one bit off; every tolerance kept says why (libm in the path, an oracle
  computed another way, or a summation order the code does not promise).
- **Mutation testing of the new output assembly.** A scoped
  `cargo mutants` run over it left 43 mutants; tests holding each field to
  the same spec run as its parts now catch 34, and the other 9 change no
  output (each with its reason). The output is held to the bit at 1, 2, 4
  and 8 threads, and an instruction-count budget covers it at one thread
  (a four-thread count did not repeat within its tolerance, and was
  dropped).

## [1.0.0] — 2026-10-12

**Upgrading from 0.13.0.** Refit every saved bank: a bank file from 0.13.0
or any earlier release is refused by its schema version, naming the way
out. Many public names now follow Polars', and an old name is refused,
naming the new one. A target computed from its own row's columns, which
`po.target(relative_to=)` built, is now a column made with `with_columns`
before the bank, or upstream of the command line. Some models' numbers move,
and some defaults: `sgd` and `pa` standardize their features, and a model
window drops a row exactly `window_size` old. Each is under *Changed*.

### Added

- **`threads=`: the bank's thread count through the API** (task 253). On
  `ModelBank(...)`, `ModelBank.load(...)`/`load_bytes`, `po.fit_predict`
  and `po.predict`, the frame namespaces' `fit_predict` and `predict`, and
  `po.gram.lars_path`, `lars_paths` and `solve_subsets`; on the command
  line as `--threads N` or a `threads` key in the TOML, the flag overriding
  the key. An integer is a pool of that many threads, shared by every bank
  given the same count and released with the last of them; `None` is the
  process pool, which `POLARS_ONLINE_MAX_THREADS` still sizes and which a
  count never reads. Not state: a save carries no count, `load` takes its
  own, and pickle and `deepcopy` keep it beside the state. The output and
  the saved state are the same to the bit at every count. A count below 1,
  a float or a bool is refused naming `threads` (`TypeError` or
  `ValueError`); 0 is not "one per core" here. `po.thread_pool_size(threads=)`
  reports the size a count names without building anything, and
  `ModelBank.threads` the count a bank was given. `with_windows` runs no
  bank and takes none.
- **`gram_threads="auto"`** on `ewridge` and `ew_cov` (task 253), as
  `shards` takes it, and now the default (*Changed*). A count is now capped
  at the pool's size. The output is the same to the bit at every setting.
- **Added with the API's standardization** (task 244):
  `po.ColumnNotFoundError` (both Polars' `ColumnNotFoundError` and
  `ValueError`) for a missing spec column; `po.PolarsOnlineWarning`, the
  base of every warning (`UnstableWarning` also derives from Polars' own);
  `Literal` annotations on every word-valued parameter; `rmse` in every
  `po.eval` metric frame; `chunk_size` on the eager frame forms;
  `load_state=` takes a `ModelBank`; `save`/`load` take file objects and
  `load` takes bytes; `ewm_mean`/`ewm_sum`/`ewm_rate` take `reverse=True`;
  `pl.col(c).alias(n)` accepted as a target; `po.gram.Gram`, a TypedDict;
  `solve_subsets` takes a ridge grid; `unnest(separator=)`, as Polars 2.0's;
  a leading `spec` column on `marginal()` and `audit()`; `half_life`/`lam`
  on `coef()`, `marginal()` and `closed_groups()`; `instance` and
  `component` in `output_index`; `cluster`/`state`/`class` in the coef
  tables; `solve_failures` in `summary()`; `polars-online --version` names
  the state schema and the Rust Polars; the runner guide shows a TOML
  file's `[[specs]]` building the same bank in Python. The exit status and
  the files written are the command line's contract; its printed text may
  be reworded in a minor release. The test suite treats a deprecation
  warning as an error, so no recipe or test uses an old name.
- **The README's versioning is written for 1.x** (task 238 follow-up,
  review 7). 1.0 starts the API promise: pin `~=1.0` or `>=1,<2`, and the
  minor too to hold the numbers; 0.x code needs the *Changed (breaking)*
  entries, and a 0.x state is refused by its schema. GRAM says a target
  with gaps is fitted pairwise within parts (the co-moments over every
  row of a part, the target's cross-moments over its own rows), so it is
  not the fixed-effects regression on that target's rows alone (2.2e-02
  on slopes of 0.2-0.5; a target on every row is that regression to
  1.7e-14), held by a test.
- **Every capability in the docs is shown by a recipe whose printed output
  a test holds** (task 238): the README (57 recipes), a recipe per model in
  `docs/MODELS.md`, and the WINDOWS, RUNNER, STATE-WORKFLOW, GBDT and
  PERFORMANCE pages. Every guide has a README section that leads to it, and
  no reader-facing doc, docstring or error message cites the plan, a task, a
  review or a record. The model entries moved to `docs/MODELS.md`, the
  window operators to `docs/WINDOWS.md`, and `po.gbdt`'s guide is
  `docs/GBDT.md`.
- **`po.gbdt` (unstable): XGBoost and LightGBM inputs built from Arrow, as
  fast as they go** (task 242). `xgboost_matrix` returns an
  `xgboost.QuantileDMatrix` and `lightgbm_dataset` a constructed
  `lightgbm.Dataset`, built in Rust from Arrow IPC or Parquet files or a
  `LazyFrame`, streamed, through the libraries' C APIs loaded from the
  user's installed packages (nothing linked); `xgb.train` and `lgb.train`
  run unchanged, evaluation sets and callbacks included. XGBoost's cuts
  come from a 200,000-row sample (`sample_rows=None` for its own sketch).
  At 1e7 rows x 50 features XGBoost reaches its first round in 4.0 s
  against 7.0 s from numpy, and LightGBM builds in 1.45 s at 2.9 GB
  against 1.54 s at 8.9 GB from a collected frame.
  A `ref=`/`reference=` built by `po.gbdt` records each feature's
  `pl.Enum` categories, and a later frame whose Enum lists other
  categories, or (LightGBM, as XGBoost already) other features in order,
  is refused by name; a reference built elsewhere is checked by its
  feature names alone. `in_memory=1` is refused, as the docstring says.
  The native engine's XGBoost floor is 2.1, the first whose C API takes
  Arrow columns (`XGProxyDMatrixSetDataColumnar`); 2.0.x fell back to the
  Python engine with a message that did not name the floor.
- **`po.gram.merge(..., within=True)` and `solve_subsets(..., within=True)`:
  pooling within parts** (task 240). Each part's moments are pooled about
  its own means, without the between-part term: the Gram of the rows
  demeaned part by part, the within (fixed-effects) estimator. Fits give
  their slopes and a `nan` intercept, and `coef_stats` takes one degree of
  freedom per part. Held to the demeaned rows (1e-12) and to statsmodels
  with a dummy per part.
- **`solve_subsets(..., within=True, target_signs=signs)`** (task 241): a
  sign per Gram on its targets, one column per draw, so a selection's null
  runs by sign flips go through the native path; each draw gives the same
  bits as negating each block's target rows. 100 null runs of 100 halves at
  200 columns take 0.08 s.
- **Online diagnostics: calibration, breaks, robust standard errors,
  specification, tails, influence and feature health** (task 221), each a
  spec switch with its own memory (`*_half_life`; `inf` is the run-once
  form, read at Kish's size): `emit_calibration` (Mincer-Zarnowitz slope,
  intercept and Wald; default memory four times the model's half-life),
  `emit_breaks` (the recursive residual over its own spread, its CUSUM and
  CUSUM of squares, and the Wald distance between fits at one and four
  memories), `emit_robust_se` (`se_coef_hc0`, `se_coef_hac` beside
  `se_coef`; `ewridge` and `rls`), `emit_specification` (Ljung-Box, past a
  look-ahead horizon by Bartlett's covariance; Breusch-Pagan; RESET),
  `emit_tails` (skew, kurtosis, Jarque-Bera; the README sets
  `drift_threshold` by the kurtosis), `emit_influence` (an online DFFITS),
  `emit_feature_health` (each feature's spread ratio and mean shift), and
  `po.eval.diebold_mariano` and `po.eval.clark_west` with Newey-West
  variance. Each measured on planted streams and held to statsmodels or
  scipy.
- **`ModelBank.check()`** (task 223): findings about what each stream was
  fed, with severity, value, threshold and a link to its section of
  `docs/DATA-QUALITY.md`; free, the same from a state file and at any
  chunking; no error or warning on 750 clean runs, every planted problem
  found.
- **`po.spec.audit`** (task 223): what a stream's columns hold, in one pass
  and constant memory per column -- nulls, NaN, ±inf and out-of-bound
  values apart, moments, distinct values and the most repeated one, runs,
  lag-1 autocorrelation and Dickey-Fuller, median/MAD/robust z, optional
  pair correlations, and the clock's duplicate stamps, gaps and spacing;
  read with `ModelBank.audit()`, and by `check()` for nine more codes.
- **`gram_threads` on `ewridge` and `ew_cov`** (task 225): the Gram's
  update on up to n threads of the bank's pool, the same bits at every
  count; 6-7 times the merge on 8 threads at 2,000-10,000 features.
- **`po.gram.lars_path`, `po.gram.lars_paths` and `po.gram.solve_subsets`**
  (tasks 226 and 227): the exact lasso path by least angle regression,
  stopped early, many Grams at once; and many subsets of Grams merged and
  fitted in one call (3.0 s against 76 s at 2,000 columns). `lasso_path`
  runs in Rust, about 70 times faster.
- **`ModelBank.gram(dtype="float32", layout="packed")`** (task 229): half
  or a quarter of the bytes; every `po.gram` function reads every form.
- **`po.ewm_var` and `po.ewm_std`, streaming on the clock** (task 212),
  beside `po.ewm_mean`, taking Polars' parameters with `bias` (default
  False). On a clock that steps by 1 they equal Polars' `ewm_var` and
  `ewm_std(adjust=False)` to 4e-14; a window that holds one value gives
  null, a constant input exactly 0 at any level. They make a feature in
  units of its own spread a column in a stream: the README's *Features in
  units of their spread* shows the recipe, and `docs/VALIDATION.md` §6
  measures it (every model there loses less on z-scored features;
  `ewridge`'s R² on the five-row target -0.052 to -0.026).
- **`po.eval` reads a target through the spec that wrote it** (task 188;
  review round 4, YB1). `metrics`, `window_metrics` and `sums` take
  `spec=`, and `compare_specs` takes `specs=`. With it, a target renamed by
  `po.target(name=)` reads its column. Without `spec=` nothing changes,
  except that a slot naming no column now asks for it. (Task 188 also scored
  a relative target on the bank's own difference or ratio; task 201 removed
  relative targets.)
- **The release checks more before it publishes** (task 191; review round
  4, CI5, CI7, CI12, CI13). It installs the sdist into a fresh environment
  and runs it, as it does each wheel, and builds with `--locked`. It refuses
  a README pin on another minor, and publishing with entries left under
  `[Unreleased]`. A tag's annotation keeps the CHANGELOG section's `###`
  headings.
- **`bocpd`'s hazard can be a duration** (task 179): on a temporal clock,
  `hazard="1h"` is the expected time between changepoints. The step `d`
  into each row carries the chance `1 − exp(−d/τ)` of a break, computed as
  `-expm1(-d/τ)` and applied before the row is read, on the decayed clock
  (a gap past `gap_cap` counts as the cap). Under it `p_change` is the
  chance that the row began a run, a row whose step is 0 reports 0, and a
  row of weight 0 applies its step's chance and learns nothing: two steps
  with nothing learned between them give the posterior one step of their
  sum. A duration is refused without a temporal clock, beside plain-number
  clock parameters, at 0 or below, and beside `hazard_col`. A number keeps
  its per-row meaning on any spec, bit for bit. `hazard_col`'s value at a
  row is the chance of a break after that row, as its docs now say.
- **Window operators, as Polars expressions** (tasks 78, 143 and 144):
  `po.ewm_mean`, `po.rewm_mean`, `po.ewm_sum`, `po.rewm_sum`, `po.ewm_rate`,
  `po.rewm_rate` and `po.increment` each return a `pl.Expr`. The
  time-weighted mean is Polars' `ewm_mean_by`, each value held from the
  operator's last valued row, as Polars skips a null; the first of a
  stretch is held from before it. The sum is `ewm_sum_by` on distinct
  stamps (at a repeated stamp every row carries the stamp's total), and the rate is
  the sum over the decayed time the window covers. The `rewm_` forms are
  their mirrors, looking ahead. A formula around them composes with
  `pl.col`, literals, arithmetic, negation, comparisons,
  `log`/`exp`/`abs`/`sqrt`/`pow`/`clip`/`fill_null`/`is_null`/`is_not_null`,
  `when/then/otherwise`, `cast` and `alias`. A node that is not
  element-wise (`shift`, `cum_sum`, a rolling function, `over`, an
  aggregation) is refused by name. A `cast` is strict unless written
  `strict=False`, as in Polars, and a wrapping cast has no form. A formula
  is kept as a compact tree of this library's own, rebuilt through Polars'
  public builders on both sides, so a spec and a saved state carry it. The
  tree writes a null literal as `["lit"]`, a form TOML can carry, and still
  reads `["lit", null]`.
- **Which rows a window holds follows Polars' `rolling_*_by`** (task 144,
  review rounds R1 and R3): `closed` (`"right"` by default), `min_samples`,
  and every row at one stamp sharing one window. So a backward output under
  `"right"` or `"both"` waits for the next distinct stamp, and a forward
  window counts a row exactly `window_size` later. Looking ahead, `"left"`
  and `"both"` hold every row at the row's own stamp, the row included: the
  mirror of a backward `"right"` window. A window's edge between two rows is
  decided from the two rows' clocks, compared as integers in nanoseconds on
  a temporal clock and with the window's own nanoseconds. So a row exactly
  one window from another lands where Polars puts it, where a difference of
  two rounded times could put it on either side: 0.4 − 0.1 is not 0.3 in a
  double. `partial="keep"` ends a cut window at the last row the group saw.
- **`po.stream.with_windows` runs them over a stream in one pass** (tasks 78
  and 143): `po.stream.with_windows(lf, *exprs, **named, clock=..., ...)`
  reads like `with_columns`, and `lf.online.with_windows` is the same for a
  chain. Each distinct operator is computed once, operators with one
  direction, half-life, window and `closed` share a queue, and the formula
  is evaluated by Polars on each chunk it emits. A row costs the same
  however long the window, the memory is about one window of rows, and the
  rows a look-ahead holds are the input's own chunks, not copies. The clock
  policy is a spec's, in the same words: a gap past `gap_cap` or a session
  change ends every window open across it, under the operator's `partial`,
  and a reset discards them. With `group`, a session is each group's. The
  call's own keywords are not output names. Each operator is held to a
  brute-force loop from its definition, to the time-reversal identity, and
  to Polars' own functions. `tests/data.py` downloads and caches one
  symbol-day of Binance USD-M futures quotes and trades for the tests.
- **A window run saves and resumes** (review rounds R3 to R9): `save_state=`
  and `load_state=` on `with_windows`. A run on the next file goes on where
  the last one ended, and first returns the rows the state held. Under a
  slice (`head(n)`) the state records how many rows of the input were
  consumed so far, and a run resumed with `load_state` on the same input,
  unsliced, skips them. So any chain of sliced runs gives one run's output,
  whatever `chunk_rows`. The state knows its input by the input's first
  row's clock and session, and by the last rows it read. Another input, the
  same input sliced by hand, or one shorter than the rows consumed is
  refused by name (`another input`), with and without a clock column. Any
  resumed run, sliced or not, refuses an input whose first row is at the
  last stamp the state read, unless that row starts a new session: a file
  boundary inside a tied stamp cannot be told from an input that repeats
  rows the state read, which would come out twice (task 158). A group or
  session column of another dtype than the state was saved with is refused
  too, since a key is its value's text (task 159). A refusal
  while a query runs surfaces as `polars.exceptions.ComputeError` under
  py-polars 1.x and as `ValueError` under 2.0.
  Windows state version 7; a state file that cannot be read is reported as
  damaged. Nine review rounds of the window operators, each finding pinned
  by a test, are in docs/PLAN.md §14.
- **Window expressions as model targets** (task 104): a spec's `targets`
  may hold a window expression looking ahead -- `po.rewm_mean("mid",
  half_life="10s", window_size="1m") - pl.col("mid")`, or a VWAP as a
  ratio of two `po.rewm_sum`s -- named by its `.alias()`. The bank keeps a
  window core per group under the spec's own clock policy. It resolves the
  target when the row's window closes, or when a gap past `gap_cap`, a
  session change or a reset cuts it, under the operator's `partial`. It
  learns the row from it once its `embargo` has passed too, and every row is
  scored where it sits, as under any embargo. `fit_predict` refuses an embargo
  shorter than the longest forward `window_size` (none included), since a
  state that learned a row before its window closed would score the rows
  that window covers in sample. `fit` and the command line's `--no-output`
  keep no prediction, so they take any embargo, and learn each row once its
  window closes; that is a property of each call, not a flag on the bank.
  The spec, the saved state and the CLI's TOML carry the formula as the
  compact tree (`{ name = "fwd", formula = [...] }`), and a loaded bank
  resumes with the windows still open. `resid_<name>` is null on the scored
  row, where the target is not yet known. Held to the column form -- the
  same expression through `with_windows(..., like=spec)` fed back as a
  plain target under the same embargo -- prediction for prediction through
  every clock event. `po.FormulaTarget` is the table's type. `ModelBank.fit`
  over such a spec is never exempt from `OrderNotGuaranteedWarning`, since
  the target reads the rows ahead. And:
  - a boolean column reaches a formula as a boolean, where a feature also
    reads it as a number;
  - `partial="drop"` on one target leaves the row's other targets;
  - `rows_learned` counts a row when its target is released with a value;
  - `drop_groups` drops a group's window core with it;
  - a refusal leaves every spec's core as it was;
  - a window expression in a hand-written spec dict is taken on every
    surface.
- **`emit_clocks`: the clock a row was scored at, and the clock of the last
  row learned** (task 152). Two fields on every scored row, `scored_clock`
  and `learned_clock`, show an `embargo` row by row: the difference is at
  least the delay everywhere. Both are in the clock column's own type: a
  `Datetime` in its unit and zone, exact to the nanosecond, and the row's
  index in its group with no clock. `learned_clock` is null before the first row
  learned and after a reset; a zero-weight row is never the learned row.
  Kept in the state and in `last_row()`.
- `po.ops`, the operators' module, in the API reference.
- **A release is tested on the newest NumPy and on its next release
  candidate** (task 142). NumPy is the optional extra (`polars-online[numpy]`),
  and every other run used the locked version. The newest NumPy now blocks
  a publish, as the newest Polars in range does, and NumPy's next release
  candidate is an early warning, at the release and in the weekly canary.
- **Every wheel is installed and run before a release publishes** (task
  160). Three of the six wheels (Intel macOS, aarch64 Linux and musl) were
  built and uploaded without ever being imported. Each build now installs
  its wheel into a fresh environment, with its dependencies from PyPI, and
  runs a fit, a saved and resumed state, and the streaming plan on it; the
  musl wheel runs in Alpine.

- **A raw spec dict takes a duration under a clock parameter, and a formula
  keeps a date's dtype** (task 160). A `pl.duration(...)` expression or a
  `timedelta` under `half_life`, `gap_cap` or another clock parameter of a
  raw dict is converted as the builders convert it; an expression anywhere
  but `targets` is refused by name. In a window formula a `Date` or `Time`
  literal keeps its dtype, and a formula may cast to `Date` or `Time`. A
  builder's `targets` is typed as a `Sequence`, so mypy accepts a list of
  `po.target` tables, of window expressions, or a mix.

- **`PolarsOnlineDeprecationWarning` and its forwarding table** (task 198;
  review round 4, D2). From 1.0 a renamed parameter keeps working under its
  old name, with this warning, until the next major version refuses it. The
  table is empty: every rename before 1.0 stays refused by name. A spec dict,
  a builder's keyword and a TOML file (on the command line's stderr) are
  forwarded alike, at any depth. The table carries spec parameters only:
  another function's keyword, a function, a word a parameter takes, a flag,
  an environment variable, an output field or a frame column renamed after
  1.0 ships its own forwarding (a check at the function's top, a stub, an
  alias or a second column) with the same warning, in the release that
  renames it (review round 5, D4).
- **`UnstableWarning`, opt-in with `POLARS_ONLINE_WARN_UNSTABLE=1`** (task
  198; review round 4, D5, N23), as Polars' `POLARS_WARN_UNSTABLE`. It and a
  docstring label mark what 1.0 does not promise: the `with_windows` state
  file, a formula target's written form, `fit_predict_arrow`,
  `predict_arrow` and `ArrowStruct`, and the modules `po.sim` and `po.corr`.
- **`summary()` gains `weight_sum_settled`, the weight the stream settles
  at** (task 198; review round 4, D8): `(W − w₁(1 − s))/s`, with `w₁` a
  row's weight and `s` the settled fraction, null on the first row, with no
  decay and under a window. A `ReadinessWarning` says when a `min_weight`
  has gone unmet for a half-life since the stream settled, with the weight
  the row rate so far implies, and the noise gate's notice names the
  half-life that would open it (task 208). The readiness floors are final
  for 1.0.
- **`max_error_inflation` and `emit_error_inflation` work on `rls` and
  `kalman`, and `max_error_inflation` on `lasso`** (task 116, A;
  WARMUP-AND-CONVERGENCE §5.8). Each is off unless set, and `ewridge`'s
  numbers do not move. `rls` reads the gate as `ewridge` does, in sum form:
  `sqrt(1 + k / n_kish)`, Kish's sample size `s₁² / s₂` from a second weight
  sum `s₂ = Σ λ^(2i) w_i²` the state now keeps; its row field is
  `sqrt(1 + ‖R⁻ᵀz‖² s₂ / s₁)` from the factor it keeps. `kalman` reads each
  row's own `sqrt(1 + z'P⁻z / R)`, exact, `P⁻` the prior covariance after
  the row's transition and process noise and `R` its `obs_var` or the
  residual variance it holds; the gate withholds row by row, and the summary
  reports the mean field. `lasso` reads `sqrt(1 + df / n_kish)` per path
  point, `df` the active coefficients plus the intercept, which bounds the
  elastic net's degrees of freedom from above; `emit_error_inflation` stays
  refused on it, since it keeps no factor. The scalar EW models keep their
  refusal, which now says why: `1 / n_kish` is a mean's variance, not a
  standard deviation's or a correlation's, and their `min_weight` of `k + 1`
  holds the gate's point. Held to filterpy's prior predictive variance,
  padasip's inverse Gram, and a calibration on data the filter is exact for:
  the mean of `e² / (R (1 + h))` is 1 within 0.089.
- **`huber` and `quantile` report `support_coef` beside `coef`** (task 116,
  B): the share of each coefficient the data determined rather than the
  ridge, as the loss weighs the rows, `1 − λ (A⁻¹)_jj` on the band system
  each solve inverts, the formula `ewridge` reads (WARMUP-AND-CONVERGENCE
  §2.2). A new column in the output struct, on `coef`'s rows, with
  `min_support_coef` and its feature in `summary()`, and the once-only
  warning under its rule (*Fixed*). Persisted per target (schema 45). Held
  to numpy's and faer's inverse: a duplicated pair reads 0.5 under both
  losses.
- **`emit_se_coef` writes `se_coef`, each coefficient's standard error in
  `coef`'s units** (task 116, F), the intercept's included, on `coef`'s rows
  and laid out like it: `diag(T Cov Tᵀ)`, `T` the map `coef` is read out by.
  `ewridge`: `Cov = σ̂² M`, `M = Σ̂⁻¹ / n_kish`, which leaves out the
  ridge's sandwich and so errs large; `rls`: `σ̂² (s₂ / s₁) A⁻¹`, `O(k³)` on
  the `coef` schedule; `kalman`: `P`, its posterior, exact. `σ̂` is the
  row's EW out-of-sample residual spread (`sigma`), `sqrt(1 + h)` larger
  than the noise in warm-up, so the error errs large there; null until a
  spread exists. A report, not a gate. Refused by name on `lasso`
  (post-selection), `huber` and `quantile` (an M-estimator's covariance is a
  sandwich) and the gradient models (no second moment). Unnests as
  `se_coef_<target>_<term>`. Held to statsmodels' WLS `cov_params()`,
  filterpy's `P` and padasip.
- **The command line closes a run with a readiness line per spec** (task
  116, H; WARMUP-AND-CONVERGENCE §7.9). After `wrote N rows`, one line per
  spec whose groups ended withheld, counted by `withheld_reason`, or whose
  `min_support_coef` is below 0.5:
  `spec "m": 2 groups whose last row was withheld (below_min_weight 2); 1 group with min_support_coef < 0.5`.
  Nothing for a spec with neither.
- **The command line takes `--skip-learned`**, and the TOML key
  `skip_learned` (task 196; review round 4, N26, AP22), as Python's
  `ModelBank.skip_learned` does: a resumed run drops the rows each group's
  saved clock has passed. It needs `load_state` and a spec that reads a
  clock.
- **New parameters, each with an entry under *Changed (breaking)*:**
  `closed` on the windowed models (task 196), `pa`'s `standardize` and
  `bocpd`'s `warm_rows` (task 195), and `sgd`'s `strict_binary`, which
  refuses a chunk holding a logistic label outside 0 and 1 (task 195, S4).
- **`gram.INTERCEPT`, `eval.SUM_FIELDS`, `eval.RESERVED`, `stream.ROLE` and
  `corr.Z_CLIP` are exported and in the reference** (task 197; review round
  4, N24), where the docs linked them.

### Changed (breaking)

Code that ran on 0.13.0 must change for these: a name, a refusal, a
reinterpreted parameter, an output's dtype or a file to refit.

- **py-polars 1.37.0 or newer is required, up from 1.34.0**
  (`polars>=1.37.0,<3`; task 247). Every surface that reads a query in
  chunks reads with `LazyFrame.collect_batches`, which before 1.37.0 runs
  the query ahead of a slower reader without limit. Draining a 1 GB parquet
  scan held 1,586 MB on 1.34.0 and 1,591 MB on 1.36.1, against 329 MB on
  1.37.0; `lf.online.fit_predict` over it peaked at 1,768 MB on 1.34.0 and
  920 MB on 1.37.1, and `po.gbdt`'s parquet build at 2.3 GB. On those
  versions a source larger than memory was held whole. The release's floor
  leg found it, and the whole suite passes on 1.37.0. `ModelBank` on frames
  still works from 1.28.1, outside the declared range.

- **The API standardized before 1.0** (task 244). A renamed *parameter*
  is read under its old name with a `PolarsOnlineDeprecationWarning` until
  2.0; a renamed *output field or frame column* cannot be, so these are
  breaking now, and the field grammar is declared stable from 1.0.
  - **Output fields** follow one grammar: kind, qualifier, columns,
    `__`grid point, `@`instance. `lag_corr_<a>_<b>_l<l>` ->
    `lag_corr_l<l>_<a>_<b>`; `filtered_<j>`/`predicted_<j>` ->
    `filtered_state<j>`/`predicted_state<j>`; a lasso's `__l<penalty>` ->
    `__p<penalty>`, and a one-point path writes none; `reset_<t>` ->
    `ramsey_reset_<t>`.
  - **Dtypes:** every count is `Int64` in every frame and field.
    `ew_class`'s `class`, `selected_<t>`, `describe().role` and
    `closed_groups().rcov_kind` are `Enum`s.
  - **Frame columns:** `groups()` `rows_processed` -> `rows_fed` (now the
    rows fed, skipped ones included, as `summary()` counts them) and
    `last_clock` -> `clock_last`; `summary()` `last_clock` -> `clock_last`,
    `min_support_coef(_feature)` -> `support_coef_min(_feature)`;
    `describe()` `null_count` (every unusable value) -> `unusable_count`;
    `audit()` `rows` -> `rows_fed`, `null` -> `null_count`, `nan` ->
    `nan_count`, `pos_inf`/`neg_inf`/`beyond_bound` -> `*_count`,
    `distinct` -> `n_unique`, and its clock table's `max_step` -> `step_max`;
    `marginal()` `t_serial` -> `t_stat_serial`; `closed_groups()`
    `pair_t_serial` -> `pair_t_stat_serial`; the `gram()` key and closed
    column `cross_centred` -> `cross_centered` (`po.gram.from_row` still
    reads the old key). `solve_failures()` returns a frame;
    `output_index`'s `half_life` is `inf`, not null, for an infinite one;
    `coef_index` returns an empty frame for a spec with no named
    coefficient, as `coef_fields` does.
  - **`po.eval`:** under `binary` (now read from a logistic spec by
    default) the columns are `brier_skill` and `accuracy`, not `r2` and
    `hit_rate`; a key below `min_samples` keeps its row with null metrics.
    The LARS paths' `stop` value is `"max_iter"`, not `"max_steps"`.
  - **`ew_cov`'s variances are unbiased by default, as Polars'**:
    `ew_cov(bias=False)` is Polars' name and default; `var`, `std`, `cov`
    and `pc<j>_var` carry `W²/(W² - Σw²)`, Kish's `n/(n-1)` (1.018 in
    steady state at a half-life of 20 rows, 1.207 at 2), matching Polars'
    `ewm_var(bias=False)` to 2.8e-15; `mahal` is divided by its square
    root; no correlation moves; with one row of weight the variance is
    null. `bias=True` keeps the old numbers. `gram()` and closed groups
    report the moments as kept, with `n_kish` beside them.
  - **The command line is `polars-online`**, not `online` (release assets
    `polars-online-<target>[.exe]`, messages prefixed `polars-online:`).
    `ONLINE_TIMING` is `POLARS_ONLINE_TIMING`, on at `1` only; the old name
    turns nothing on and prints a notice. A relative path in a TOML config
    is read from the config file's directory; a flag's path still from the
    working directory. The progress line is drawn only on a terminal.
  - **Spec keywords renamed (forwarded):** `kmeans(k=)` -> `n_clusters`,
    `hmm(k=)` -> `n_components`; `seed_rule` -> `init` (`"kmeanspp"` ->
    `"k-means++"`); `hmm`/`ew_class` `covariance` -> `covariance_type`
    (`"shared"` -> `"tied"`, `"diagonal"` -> `"diag"`; `ew_class` defaults to
    `"full"`); `bocpd` `emission` -> `covariance_type` (`"gaussian"` ->
    `"full"`; `"robust"` -> giving `robust_beta`, defaults unchanged);
    `bocpd`'s `prior_mean`/`prior_kappa`/`prior_nu`/`prior_scale` ->
    `mean_prior`/`mean_precision_prior`/`degrees_of_freedom_prior`/
    `covariance_prior` (scikit-learn's names); `window_every` ->
    `snapshot_every`; `max_rows_between_pca` -> `max_rows_between_pcas`;
    `lasso(lasso_path=)` -> `penalties=`; `corrchange` `span_rows` ->
    `window_rows`, `reset_on_flag` -> `change_action` (`"flag"`/`"reset"`).
    Spec dicts, `ModelBank.specs`, saved states and the resolved defaults
    write only the new names. `holt`/`seqtest` lose `features=`;
    `kmeans`/`micro` show `standardize=True`, and `scale_floor` beside
    `standardize=False` is refused. Every refusal of a word outside a list
    reads `spec "m": <param> must be one of ...; got "x"`.
  - **Methods and helpers renamed (forwarded):** `fit`/`fit_predict_batches`
    take their options by keyword; `save(path=)` -> `file`, `load(path=)`
    -> `source`; `to_json`/`save_json` -> `write_json(file=None)`;
    `rows_fed()`/`output_fields()` -> properties; `po.rewm_*` ->
    `po.ewm_*(..., reverse=True)`; `po.eval` `spec_name` -> `spec` (a str
    or the spec dict), `spec_names` -> `specs`; `po.gram.subset(cols=)` ->
    `columns`; `lars_path`/`lars_paths(max_steps=)` -> `max_iter`;
    `po.corr.matrix` -> `po.gram.correlation`. `po.gram.lasso_path`'s
    `tol` is the `lasso` model's 1e-10 (was 1e-7). Saved banks are refused
    (schema 61).
- **Every integer output is `Int64`** (task 238): `kmeans`' `cluster`,
  `micro`'s `n_clusters` and `n_micro`, and `hmm`'s `state` were `Int32`,
  while `micro`'s `cluster` and every id, count and run length was `Int64`,
  so code reading `cluster` from one model broke on the other's.
  `output_index` no longer names an `i32`.
- **The window formula refusal lists everything a formula reads**, in
  Python and on the command line (negation, `&`/`|`, `pow`, `is_null` and
  `is_not_null` were missing), from one list both check.

- **A Poisson `sgd` fit refuses a chunk with a negative target, naming the
  row** (task 209), as scikit-learn's `PoissonRegressor` does and as
  `strict_binary` refuses a logistic label. A count is never negative, and
  the gradient `p - y` drove the prediction to the link's floor, e^-30.
- **Every saved bank must be refit.** A bank file now carries schema 60,
  and one saved by 0.13.0 (schema 20) or any earlier release is refused by
  its version, naming the way out: refit from the input. Ten changes
  moved the layout: the stream's diagnostics (task 146), the names the
  specs a file stores carry (task 144), the window core a formula target
  keeps (task 104, then review rounds R4 and R6), the PCA, pruning and
  window cadences of `ew_cov`, `micro` and the windowed models (tasks 161,
  163 and 162), `lasso`'s threshold per target (task 174), the stamps a
  model window keys its snapshots by (task 175), `quantile`'s band factor
  (task 170), the elapsed clock an embargo's held rows are measured on
  (task 176), and where each stream's `coef` cadence stands (task 178). An
  `ew_cov` state with `mahal_quantiles` is refused too. Schema 37 keeps
  the stamp of the last solve, PCA refresh and checkpoint (task 180).
  Schema 38 drops a Gram index nothing read from the systems `ewridge` keeps
  for its readiness statistics, and lets a closed `rcov` row's
  `psd_repaired` be null (task 186). Schema 44 follows tasks 194 to 202:
  the stream's state as one value, the clock range as clock values and the
  key columns' types (194); the residual scales, `pa`'s scaler, `bocpd`'s
  warm-up and the per-target thresholds (195); the window's edge and the
  renamed counts (196); an integer clock held as one (200, windows state
  8); relative targets removed (201); and the target's own spread (202).
  Schema 45 adds `rls`'s second weight sum `s₂` and the data shares
  `huber` and `quantile` keep per target (task 116). Schema 46 keeps the
  readiness notices' wait (task 208). Schema 47 adds the standardized
  models' warm-up count and holds `sgd`'s and `pa`'s coefficients in the
  caller's units after it (task 206). Schema 48 is a `with_windows`
  state's version 9 (task 212: the variance operators). Schema 49 adds
  `kalman`'s anchor and the clock since each covariance last observed its
  target (task 211). Schema 50 sizes `kalman`'s prior from a median of its
  first innovations (task 214). Schema 51 drops `ftrl`'s penalty scale and
  keeps `kmeans`'s and `micro`'s centres as compensated pairs (task 215).
  Schema 52 counts each target's rows of positive weight, and all rows, in
  a windowed model's state and in each of its snapshots, and each slot's
  residuals in the residual spread's window (task 217). Schema 54 adds
  `audit` (task 223), schema 56 the diagnostics' accumulators (task 221),
  schema 57 the error inflation a held row was scored with (review
  round 6), schema 59 the audit's smallest change and schema 60 the
  diagnostics' sums under a horizon and the twin fits' cross sums (tasks
  233 and 232). It refuses 59 and older, and so do the models' own states.
- **A state is loaded whole or refused, never mended** (task 198; review
  round 4, D1, CC8). A state missing a field written since an older layout,
  or holding a vector of the wrong length, such as a mean's low part, is
  refused where it was filled in. Every schema this build loads is frozen
  in the tests: each loads, goes on to the bit and saves its bytes again.

- **A fit nobody solved predicts nothing** (task 186; review round 4, CC1).
  `ewridge`, `lasso`, `huber` and `quantile` predicted a target no solve had
  fit, such as a late target or one whose first solve failed, from zero
  coefficients. That gave exactly 0.0 until the next scheduled solve: on 56
  of 80 rows under `solve_every=1000`, and 4 under the default cadence. Such
  a target now predicts null, and its `coef` entries are null, until a solve
  sees its rows. A fitted target whose weight a gap takes to exactly 0 keeps
  its fit, and so does `ewridge`'s `coef_prior` under a ridge.

- **A spec value that sizes memory before the first row has a ceiling**
  (task 193 and its merge; review round 4, CD10, CF2, CE9). A lag,
  `resid_autocorr_lag` and `n_perm` are refused past 2^20, `kmeans`' `k`
  past 2^16 and its warm-up buffer past 256 MiB, and `hmm`'s `k` past 1024.
  An `rcov` ring whose lagged products would pass 256 MiB is refused too.
  Each refusal names the value. `marginal(lags=[2**62])` panicked in the
  builder, `lags=[10**11]` reserved 2.4 TB, and `hmm(k=2**62)` grew memory
  without bound. One legitimate spec is hit: an automatic `rcov` ring at 100
  features over a block of 10^6 returns, about 1.1 GB.

- **A parameter its mode does not read is refused** (task 193; review round
  4, CF6, CE4, PC6, PC8). `kmeans` refuses `dead_frac > 0` beside
  `split_merge = 0`, and a `warm_rows` below `k`, as `hmm` does; its default
  `warm_rows` is now the larger of 500 and `k`. `hmm` refuses `transition`
  beside `tvtp_coef`, and its warm-up parameters beside given states.
  `rcov` refuses `jitter`, `max_bandwidth` and `theta` under a kind that
  does not read them, and the builder's `jitter` and `theta` default to
  `None`. `kalman` takes `coef_half_life` or `q`, exactly one: `q` alone was
  refused for a missing half-life, and the two together ignored the
  half-life.

- **The builders refuse the empties a raw dict is refused for** (task 193;
  review round 4, YA5, PC10). `feature_sets={}`, `blocks={}`,
  `mahal_quantiles=[]` and an empty target name are refused. A grid listing
  0 and -0 is refused as a duplicate, and a field named for -0 is named for
  0: `__r0`, where it was `__r-0`. A tuple comes back from the bank as the
  list it went in as.

- **A number-clock step past the largest double is refused by row** (task
  193; review round 4, PB7), before anything is touched. A clock from
  −1e308 to 1e308 was taken, and `coef_every=100` then wrote `coef` on
  every row.

- **`corrchange(kind="sequential")` refuses an `alpha` too small for its
  pairs** (task 186; review round 4, CD12). A share of `alpha` per pair
  below 2^-52 is refused by name, where the test ran unable to flag.

- **`refresh_time` refuses a value column that is not numeric** (task 187;
  review round 4, PC7), as the bank refuses such a feature. A `String`
  column gave an empty grid, day counts or epoch numbers. A value that is
  NaN, infinite or past 1e100 is now a tick that observed nothing, so a
  point that waits on such a series completes later.

- **`po.gram.solve` and `po.gram.lasso_path` refuse numbers outside their
  ranges** (task 188; review round 4, YB12). A ridge, penalty or weight must
  be finite and at least 0, `l1_ratio` within [0, 1], `max_iter` at least 1,
  and `tol` finite and above 0. `max_iter=0` returned all-zero slopes.

- **Four frames keep the columns and dtypes their inputs give** (tasks 187
  and 188; review round 4, SF3, YB2, YB14). `coef()` on a bank with no
  coefficient spec has every other `coef()`'s columns, so the frames
  concatenate. `po.stream.embargo(weight=)` takes an integer or Boolean
  weight, which died in the merge with a `SchemaError`, and the weight
  comes back `Float64`. `sim.regimes`' `activity` is `Float64` with or
  without activity. `rolling_metrics`' `window_start` keeps an integer
  clock's dtype, and refuses a `window_size` that is not whole on such a
  clock.

- **`ew_cov`'s principal components refresh on the clock** (task 161), as a
  regression's solve does. `pca_every` counts the clock's units, as
  `solve_every` does: a number of the clock column's units, a duration on a
  temporal clock (`"5m"`), or `0` for every row. `max_rows_between_pca`
  caps the rows between refreshes, as `max_rows_between_solves` does, and
  whichever comes first refreshes. With neither, the components refresh
  every row, as before. A spec without a clock column is unchanged, its
  clock being the row's number; a clocked spec that gave `pca_every=N` now
  refreshes every `N` clock units, not every `N` rows. A spec with
  `max_rows_between_pca` alone exports to JSON again; `to_json()` refused
  it while it was unreleased.

- **`micro` prunes on the clock** (task 163), as a regression's solve is
  scheduled. `prune_every` counts the clock's units: a number of the clock
  column's units, a duration on a temporal clock (`"10m"`), or `0` for
  every row. `max_rows_between_prunes` caps the learned rows between
  checkpoints, and whichever comes first checkpoints. With neither, every
  100 learned rows, as before. A row of weight zero advances the clock, so a
  quiet spell prunes faded summaries on time, where the learned rows alone
  waited for the next learned row. DenStream checks every `Tp` clock units,
  which `prune_every` can now follow. A spec that gave `prune_every=N`
  counts clock units now: without a clock column the first checkpoint comes
  a row later (the first row is at clock 0) and rows of weight zero count;
  `max_rows_between_prunes=N` is the old cadence exactly.

- **A window's snapshots are spaced on the clock** (task 162), in
  `ewridge`, `lasso`, `ew_cov`, `ew_class` and `marginal`, as a regression's
  solve is scheduled. `window_every` counts the clock's units: a number of
  the clock column's units, a duration on a temporal clock (`"1m"`), or `0`
  for every row. `max_rows_between_snapshots` caps the rows between
  snapshots, and whichever comes first takes one; with neither, every row,
  as before. Under a clock spacing the effective window is within
  `window_every` of `window_size`, in clock units, and a burst of rows inside
  one spacing takes no snapshot of its own, so the ring holds about
  `window_size / window_every` snapshots whatever the row rate. A thinning
  `window_budget` doubles whichever cadence is in force, and the refusal
  past a budget names it. A spec without a clock column is unchanged; a
  clocked spec that gave `window_every=N` now means `N` clock units, and
  `max_rows_between_snapshots=N` is the old cadence exactly.

- **`drift_threshold` is a clock parameter, required with a clock** (task
  168; review 2026-10-05, CC1). The drift detector sums each row's excess,
  in `sigma`, times the row's clock step, so its threshold is `sigma` times
  clock time: a number of the clock column's units, or a duration on a
  temporal clock (`"20m"` is one `sigma` of excess held for twenty
  minutes). The default of 20 stays only without a clock column, where a
  row is one unit and it is the classic Page-Hinkley test. A spec with a
  clock and `emit_drift` and no threshold is refused, naming the fix, as one
  without `gap_cap` is. A temporal clock refuses a plain number, and a
  numeric clock a duration. Before, every temporal clock was read in
  seconds whatever the column's own unit, so the default was 20
  `sigma`-seconds there. At a row a minute it flagged 244 of 3,000 rows of
  noise, where `"20m"` flags none. To keep a spec's old numbers, give
  `drift_threshold=20.0` on a numeric clock and `"20s"` on a temporal one.

- **A window looking ahead under `group` needs a clock column** (task 173;
  review 2026-10-05, PC1). `with_windows` refuses `po.rewm_mean`,
  `po.rewm_sum` or `po.rewm_rate` under `group` with no clock column,
  through `po.stream.with_windows`, `lf.online`, `df.online` and `like=`,
  and a spec refuses a window target under `group` with no `clock`, both
  with one message naming the fix. On a row count, a group that fell
  silent never closed its windows, and `with_windows` held every later row
  of every group to the end of the input: 500,001 rows and 117 MiB more
  over 2M rows, where about one window was promised. Without a clock there
  is no `gap_cap`, which bounds a silent group on a clock.

- **`coef_every` counts the clock, as `solve_every` does** (task 178). A
  number of the clock column's units, or a duration on a temporal clock
  (`"5m"`), writes `coef` once the clock has moved that far since the
  group's last `coef` row, measured on the exact decayed clock (task 175);
  `max_rows_between_coefs` writes it after that many accepted rows, counted
  as `coef_every` counted them; whichever comes first. **`coef_every=0`,
  which meant the default, now means every row**; leave it unset (`None`)
  for the default, each group's last row in each chunk. A spec with a clock
  that gave `coef_every=N` now means N clock units: write
  `max_rows_between_coefs=N` for N rows. Without a clock column the clock
  is the row's number, so `coef_every=N` still writes every N-th row,
  skipped rows counted. Under a cadence the `coef` rows no longer include
  each chunk's last row, so they no longer depend on the chunking; a clock
  reset starts the cadence over. Both are refused on a model without
  coefficients, `coef_every=0` included. No model's numbers move: only
  which rows carry `coef` and `support_coef`.

- **The public names follow Polars, and say what they do** (task 144; the
  user, 2026-10-02: "Add all", and no backward compatibility for outputs).
  No aliases: an old parameter is refused naming the new one, from a spec
  builder, a spec dict, `with_windows` and a TOML file; an old output name
  is simply gone. The two tables after this list name each one.
  - **One clock rule in place of two**: `on_clock_reset` and
    `min_backwards_jump` are `restart_after_step_back`. Unset, every step
    back is refused (what `"error"` did); given a clock amount, a step back
    larger than it starts the model over and one no larger is still refused
    as a late row (what `"reset_state"` with `min_backwards_jump` did; the
    comparison is inclusive). In the specs and in `with_windows`.
  - The spec builders refuse two outputs that would render to one field
    name, naming the inputs that collided.

The parameters task 144 renamed:

| was | is |
|---|---|
| `halflife`, alone and in every name that spells it | `half_life`: `long_half_life`, `coef_half_life`, `revert_half_life`, `select_half_life`, `level_half_life`, `trend_half_life` |
| `label_delay` | `embargo` |
| `max_dclock` | `gap_cap` |
| a model's `window`, and `po.eval.rolling_metrics`'s | `window_size` |
| `min_periods` | `min_weight` |
| `emit_resid_z` | `emit_zscore` |
| `sgd`'s `scale_features` | `standardize` |
| `add_intercept` | `fit_intercept` |
| `lasso`'s `max_cd_iters` and `cd_tol` | `max_iter` and `tol` |
| `corrchange`'s `reset` | `reset_on_flag` |
| `ridge_decay`, a bool | `ridge_scale = "mean"`, the default, or `"sum"`, checked by value in the builder |
| `on_clock_reset` and `min_backwards_jump` | `restart_after_step_back`, the one clock rule above |

The output names task 144 renamed:

| was | is |
|---|---|
| `n_eff`, in every field, in `closed_groups`, `coef()`, `last_row()`, `marginal()` and `gram()`, and in `po.eval`'s sums | `weight_sum`, as `summary()` already called it: `weight_sum@h10`, `pair_weight_sum` |
| `withheld_reason`'s value for the weight gate | `below_min_weight` |
| `coef()`'s `lambda` column | `penalty` |
| `resid_z_<t>` | `zscore_<t>`: a target named `z_y` collided with `resid_z_y` |
| `lam_selected_<t>` | `penalty_selected_<t>` |
| `pcorr_<a>_<b>` | `partial_corr_<a>_<b>` |
| `absresid_q<p>` | `abs_resid_q<p>` |
| the PCA loadings, `pc<j>_<feature>` | `pc<j>_loading_<feature>`: a feature named `var`, `share` or `score` collided with the component's fields |
| `bocpd`'s `logscore` | `loglik` |
| `hmm`'s `p_<k>` and `p1_<k>` | `filtered_<k>` and `predicted_<k>` |
| `micro`'s field `micro` | `micro_id` |

- **`po.corr.signal_share` takes a Kish size** (task 148): its second
  argument is `n_kish_blocks`, was `n_eff_blocks`. The sampling variance of
  a correlation's Fisher-z is `1 / (n - 3)` at Kish's `n`; `weight_sum` is
  a weight, about half of it, and doubled the noise floor.

- **Builders and helpers refuse what they ignored or crashed on** (task
  160). The eight builders of models with no target (`ew_cov`, `kmeans`,
  `micro`, `deco`, `bocpd`, `corrchange`, `hmm` and `rcov`) refuse
  `features=[]` by name, where it raised `IndexError`, and the five that took
  `targets=` refuse it as `ew_cov` does. `po.spec.sgd` refuses `huber_delta`,
  `quantile` or `eps` beside a loss that does not read it, and `power` beside
  a schedule other than `"inv_scaling"`. `po.spec.lasso` refuses
  `max_iter=0`. A window operator refuses `partial` without a `window_size`,
  and a formula refuses a cast to a parametrized dtype, such as `Datetime`
  or `List`, by name. A `save_state` that is a directory is refused when the
  plan is built, not after the stream.

- **A broken bank refuses every export** (task 160). `save(path)` raises
  `ValueError`, as `save_bytes` does, and leaves the file untouched, where it
  raised `OSError`; `to_json()` and `save_json()` refuse it too.

- **`po.spec.marginal` writes its optional keys only when they are given**
  (task 160): `lags`, `cross_lags`, `serial_rule`, `bins`, `bin_rule`,
  `bin_warm_rows`, `bin_edges`, `bin_budget`, `shards`, `window_lags` and
  `feature_moments`, as the saved spec does, so `ModelBank([s]).specs[0] ==
  s`. Code that indexed one of them in a marginal spec reads it with `.get`.

- **More of what crashed, hung or did nothing is refused by name** (task
  160). `corrchange` refuses a `boundary_gamma` above 0.49, whose critical
  value took minutes to hours to solve, and a non-default `alpha_adjust`
  under `kind="window"`, where it did nothing. `rcov` refuses a
  `bandwidth`, `max_bandwidth` or `preavg_rows` above `block_rows`, a
  `theta` whose window is longer than `block_rows`, and a ring, window,
  jitter or stride above 2^20, which panicked or aborted the process; and
  `preavg_rows=2` without `psd`, where the estimate is the zero matrix. A
  spec refuses `gap_cap` without a clock (so does `with_windows`),
  `half_life=[]`, and a `coef_min`/`coef_max` list of the wrong length even
  when every bound is infinite. A dict or TOML spec now meets the builders'
  refusals of a formula target named after its own column and of an `sgd`
  parameter its loss or schedule does not read. In Rust, every model's
  `new` refuses a half-life of 0 or below, or NaN, and a `lam` outside
  (0, 1].

- **`refresh_time` returns its columns in their own dtypes** (task 160).
  `time_refresh` is the completing tick's clock in the clock column's own
  dtype, exactly, where it was a `Float64` of the clock's physical integer.
  The group column is the input's own value at each completing tick, so
  `Datetime`, `Time`, `Struct`, `Boolean` and zoned keys round-trip.

- **Names that follow Polars and the library's own words** (tasks 194, 196
  and 197; review round 4, N1-N10, N14-N16). Each old name is refused,
  naming the new one:

  | was | is |
  |---|---|
  | a spec's `type = "ew_ridge"` | `type = "ewridge"`, the builder's spelling; core messages say `huber:` or `quantile:` where they said `robust:` |
  | `chunk_rows`, the TOML key and `--chunk-rows` | `chunk_size` and `--chunk-size`, Polars' name |
  | the command line's `--resume` | `--load-state` |
  | `kmeans`' `update_every` and `split_merge_every`; `corrchange`'s `permute_every` | `update_every_rows`, `split_merge_every_rows`; `permute_every_rows` |
  | `rls`'s `ridge` | `delta`, the prior's strength |
  | `ew_cov`'s statistic `lagcorr`, its fields `lagcorr_<a>_<b>_l<ℓ>`, and `marginal()`'s and `closed_groups()`' `lagcorr_*` | `lag_corr`, `lag_corr_<a>_<b>_l<ℓ>`, `lag_corr_xx` and the rest |
  | `kmeans`' field `dist2` | `dist_second`, the distance to the second-nearest centre |
  | `marginal()`'s `t` and `closed_groups()`' `pair_t` | `t_stat` and `pair_t_stat` |
  | `closed_groups()`' rcov block `bandwidth_used`, `omega2`, `iv_sparse`, `iq`, `psd_repaired` | `rcov_bandwidth_used`, `rcov_omega2`, `rcov_iv_sparse`, `rcov_iq`, `rcov_psd_repaired` |
  | `ModelBank.rows_seen()` | `rows_fed()` |
  | `po.eval`'s `min_obs` and `by=` | `min_samples` and `group=`, which takes one key as a bare string |
  | `po.eval.rolling_metrics(window_size=)` | `window_metrics(every=)`: its windows do not overlap, as Polars' `group_by_dynamic(every=)` |
  | `po.corr.shift` | `po.corr.absorption_shift`; Polars' `shift` is a lag |
  | `po.gram.lasso_path(lambdas=)` | `penalties=`, the name `coef()` reports them under |

- **`holt`'s `level_half_life` is removed** (task 196; review round 4,
  N16): the level takes the spec's `half_life`, and the old key is refused
  naming it.
- **Relative targets are removed** (task 201). `po.target(relative_to=,
  relative=)` and a target table's `relative_to` and `relative` keys are
  refused by name. Compute such a target as a column with `with_columns`,
  upstream for the command line, preferring a log ratio or a difference for
  a return: a plain ratio sits about 1, where every row is a hit. Every
  target's `hit_rate` is taken about 0. `po.target` keeps `name=`, and
  formula targets stay.
- **The windowed models take Polars' `closed`, default `"right"`** (task
  196; review round 4, N17): a row exactly `window_size` old leaves the
  window, as in Polars' `rolling_*_by`, where it stayed. On a 1 ms clock
  with a 1 s window `weight_sum` is 1000 where it was 1001. `closed="both"`
  keeps the old edge, to the bit; `"left"` and `"none"` are refused.
- **`sgd` and `pa` standardize their features by default** (task 195; review
  round 4, U2): `standardize=True`, and `pa` gains the parameter. A fit is
  then free of the features' units. Standardizing costs about 23% of `sgd`'s
  throughput and 21% of `pa`'s.
- **`huber_delta` defaults to 1.345, the 95%-efficiency constant** (task 195;
  review round 4, U1, U3), on `huber` (from 1.5) and on `sgd`'s `huber` loss
  (from 1.0), whose cut is now in units of each target's residual standard
  deviation, as `huber`'s is.
- **An insensitivity band is in units of the target's own spread** (tasks
  195 and 202; review round 4, U1). `pa`'s `eps`, and `sgd`'s under
  `loss="epsilon_insensitive"`, are multiples of the EW standard deviation
  of the target around its EW mean, where they were in the target's units.
  A fit from zero coefficients on a target far from zero no longer stalls:
  at a level of 1,000 without decay, `pa`'s R² goes from −52.6 to +0.965.
  The default `eps` is 0.01, where it was 0.1 (task 203): the band does not
  shrink as the fit improves, so it must sit below a good fit's errors. On
  a target predicted to within 1% of its spread, 0.1 stopped the fit about
  0.08 off the truth. On a target predicted less well (R² 0.98), `pa` at
  `c = 1` fits about a third worse out of sample at 0.01 than at 0.1 (2.2
  against 1.6 times the noise), and `c=0.1` damps best there (1.3): the
  tube is its only damping against noise. Above about R² 0.99 the wider
  tube is the smaller loss (at R² 0.9975: 2.0, 1.2 and 1.7).
- **`bocpd` sets a left-out prior from its first rows** (task 195; review
  round 4, U4, U5). Without `prior_mean` or `prior_scale`, the first
  `warm_rows` learned rows (default the feature count plus 2) set it from
  their mean and covariance, and report null. The identity prior made a
  stream at 1e-4 a changepoint on every row and one at 1e4 never break.
  `prior_nu` defaults to `d + 2` under `gaussian` and 3 under `diag` and
  `robust`. In the REGIMES benchmark false alarms fall from 0.68 to 0.53
  per 1000 rows and the median delay rises from 98 to 121.
- **Clock columns in the frames keep the clock's own type** (tasks 194 and
  200; review round 4, N18). `summary()`, `groups()` and `closed_groups()`
  give the clock range in the clock column's dtype, exactly, where a
  temporal clock read as Float64 seconds (64 ns off at 2024). An integer
  clock is held as an integer: its fields, `scored_clock` and
  `learned_clock` included, are its own integer dtype. Over specs whose
  clocks differ in type, these frames ask for `spec=`.
- **`closed_groups()`' `rows_fed` and `rows_learned` are UInt64**, as in
  `summary()` (task 194; review round 4, N19).
- **`group=` on the readers takes a list of keys, `None` for the null
  group, and integer keys list in numeric order** (task 194; review round
  4, N21): `coef`, `gram`, `last_row`, `summary`, `describe` and `marginal`.
  Another element type raises `TypeError`.
- **A key or clock column that changes type is refused by name** (tasks 194
  and 200; review round 4, N22). A group or session column keeps the type
  of its first chunk, in the state too: an Int64 key then a Float64 one
  started every group over. A clock that turns from integer to float, or to
  an integer of another width, is refused, and so is a `UInt64` clock value
  past the largest `Int64`.
- **More values are refused by name** (tasks 195, 196 and 197): a row cap
  of 0 (`max_rows_between_solves` and the rest) and `pca = 0` (U7);
  `refresh_time`'s clock column of any type but a number or a time, where
  text of digits was read as a clock; and `corrchange(kind="sequential")`
  with a share of `alpha` per pair below 5e-11, where the floor was 2^-52:
  past it the critical value's error grows from 2% to 29% at 1e-11.

### Changed

Numbers move in most of these, each saying by how much; under this
project's versioning that is carried by the minor version before 1.0.

- **`gram_threads` defaults to `"auto"`, which splits only into idle
  threads** (task 253; it was 1). On each chunk the bank counts the
  (spec, group) tasks it runs side by side, `T`, on its pool of `P`; when
  `2T <= P` each task's `k x k` update may use `2P // T` threads, at most
  `P`, and otherwise one. A function of the two counts, not of timing.
  Outputs and saved states are unchanged to the bit, at every count and
  chunking, and a state saved under the old default loads and continues
  bit-identically (the setting is not state; a spec that left the key out
  still writes none, and now runs `"auto"`). Measured on 14 cores against
  1, two runs of 11 interleaved rounds: one group at 2,000 features
  2.93x/3.00x (3.56x/3.51x with `gram_block_rows=256`), four groups at 800
  1.66x/1.58x, seven 1.09x/1.00x, and 14, 28 (`ewridge` and `ew_cov`), 56
  and 64 groups 0.98x to 1.05x, where they run the one-thread path. Taking
  the whole pool, as `"auto"` did before, was 0.89x/0.91x at 28 groups.

- **The online diagnostics' nulls, after review round 7** (A-1, A-2, A-3,
  E-1). Beside a `kalman` with a `coef_half_life` list, the windowed
  `cusum`'s null share reads the intercept's memory, which is what absorbs
  a level (`[inf, 30, 30]`: spread 1.40 -> about 1.0); on a quantile fit
  under a horizon, `break_wald` scales by the residuals' long-run variance,
  not the indicators' (10.8% of quiet streams past 7.815 -> near the
  least-squares fits' 5.9%; schema 61, a second long-run accumulator);
  beside a `window_size`, `cusum` divides by the window's own share, a
  closed form exact at both ends (0.703 with no decay at the default
  memory, where it read 1/2: spread 1.08-1.20 -> 1.00-1.02 on windows of
  100 and 400 rows, with no decay and at half-lives 50 and 200; `cusum_sq`
  unmoved). The diagnostics page states the two readiness clocks: the
  notices and `settled_frac` read the model's own half-life, a diagnostic
  waits for the longest memory the model keeps.
- **`po.gbdt.xgboost_matrix` (unstable): with `weight=`, the cuts follow
  the weights, as XGBoost's own sketch does** (review 7, C-4). The sample
  is drawn in proportion to the weights (priority sampling, Duffield, Lund
  and Thorup 2007) and the cuts are its weighted quantiles; a row of
  weight 0 places no cut. On 50,000 normal rows with weight 50 above 1.0
  at `max_bin=16`, 14 of 15 cuts fall above 1.0, as 14 of XGBoost's own do
  (2 before), and the heaviest bin holds 6.3% of the weight (35% before).
  Unweighted cuts, and cuts under equal weights, are unchanged to the bit.
  `lightgbm_dataset`'s bins still ignore weights, as LightGBM's own do.
- **`bocpd`'s capped runs fold into the oldest, scaled back to the cap**
  (task 243): the runs that reach `max_run - 1` rows fold into the oldest
  of them, whose weight and scatter shrink to `max_run - 1` rows' worth
  while its mean stays, instead of into the youngest. Under heavy pruning
  the dominant run keeps its memory: at `max_run=20`, `prune_below=1e-2`, a
  quiet stream's summed `p_change` falls from 34.8 to 7.1, and REGIMES §5's
  false alarms from 1.00 to 0.53 per 1,000 quiet rows. Nothing moves at the
  defaults below 10,000 rows. Among the runs at the cap, the one carrying
  the most mass is kept (the oldest among equals): after a break the
  post-break run keeps its statistics there, where folding into the oldest
  put its mass on the row-0 run's and raised a second alarm about
  `max_run` rows after the break when nothing was pruned (`prune_below=0`,
  `max_run=200`: 17 of 20 streams, none now). The capped run is an
  exponentially weighted run scaled by `(max_run - 1)/max_run` each row,
  half-life about `0.69 (max_run - 1)` rows.
- **Diagnostics wait for the longest finite memory a model keeps** (task
  234): `holt`'s trend, `kalman`'s longest, `ewridge`'s
  `long_half_life`, `lasso`'s `select_half_life`. Settled on its level
  alone, `holt`'s CUSUM spread 2.7 and passed 3 on 28% of the rows after
  its diagnostics started; now 0.74 and on none.
- **The diagnostics, after review round 6** (tasks 232 and 233): each
  folds a row only once its instance is settled on the fit's memory; its
  default memory follows the fit's (a window's Kish half-life, `kalman`'s
  `coef_half_life`); a new `horizon_rows` puts the calibration,
  Breusch-Pagan and RESET under a Newey-West variance and the CUSUMs and
  `break_wald` under the long-run variance; `cusum_sq` reads the measured
  fourth moment and the windowed CUSUMs their derived null, so a threshold
  means what the docs say; `break_wald` uses the exact variance of the two
  fits' difference; quantile fits report `calibration_coverage` and a
  CUSUM over indicators. `check()`: `frozen` is a zero-change atom test,
  `collinear` reads the centred index for the centring models, `leakage`
  is a warning, `rls`'s level limit follows the prior it still holds;
  `audit` refuses `group_close`, and its clock reports uncapped steps in
  Durations; `gram(dtype="float32")` refuses an entry out of range.
- **A `kalman` state with a damaged noise basis is refused with a message
  that says what the basis must hold** (task 218): at most 3 rows, as many
  weights as squared innovations, each finite and above 0; a bad clock
  since an observation has a message of its own.
- **Under a half-life `ftrl`'s penalties no longer carry the per-target
  scale `W/W*`** (task 215; task 115(d)'s scale). A zero-weight row is now
  clock alone for `ftrl` as for every model (hard rule 9 holds for every
  model, with no exemption left), predict-only rows still read the fit as
  the last row that taught it left it, and the row that ends a gap meets
  the aged sums against the full penalties, as FTRL's objective (a fixed
  regularizer against losses aged on the clock) implies: a constant 5 at a
  half-life of 100 is 0.75 of itself after a gap of one half-life.
  `GOLDEN_FTRL` moves by up to 0.18% where a target is null.
- **A reverting `kalman` slot's process noise over a gap is bounded**,
  `q·((1 - 2^(-D/r))/θ)²` with `θ = ln2/r` (task 214): it grows as `q·D²`
  for a short gap and saturates for a long one, so the slot's uncertainty
  no longer grows without limit across a run of null targets. A random-walk
  slot is unchanged to the bit.
- **A standardized `sgd`, `pa` or `kalman` fit no longer moves because its
  scaler did** (task 206; review round 5, G1). Under a finite half-life the
  EW moments wander, and a fit held in standardized coordinates and read
  through them moved its predictions with them: at R² 0.99998 and a
  half-life of 50, `sgd` paid 237 noise variances of excess error where an
  unstandardized fit paid 0.010; at a half-life of 10 it reported a slope
  of 3.03 for a truth of 2. Each now runs as before for its first 22 rows
  (Kish's count of the rows its scaler has learned; the bits are 0.13's),
  then `sgd` and `pa` hold their coefficients in the caller's units, each
  step still formed in standardized coordinates, and `kalman` re-maps its
  coefficients and covariance exactly through every move of the scaler. A
  re-map a row of data cannot make -- a scale moving by more than 1,024
  times in one row, or a mean by more than 1,024 of its scale -- is refused
  whole, as a non-finite update is. G1's worst cell goes from 1.8e5 times
  the unstandardized error to 1.05; short histories are unchanged; a ×10
  change of a feature's scale goes from 7.5e4 noise variances to 0.12.
  Holding the fit in the caller's units from the first row was built
  first and wrecked short histories (R² at rows 25-50 of 200-row groups
  from 0.43 to -207), which is why the warm-up exists. On
  `docs/VALIDATION.md`'s real BTCUSDT data `kalman` reads lower (R²
  -0.109 to -0.123 and -0.005 to -0.059): reading the fit through the
  moving moments had shrunk its coefficients in volatility bursts, which
  on those near-noise targets happened to help. Schema 47.
- The windowed models' `closed` is a two-word type: `left` and `none` are
  refused with their reason as the spec is read, where `validate` refused
  them a step later (review 5, E2). `resolved_defaults` renders the clock
  policy as `restart_after_step_back`, the name a spec takes, where it
  rendered the two names task 144 merged (C5).
- **The noise gate's "cannot be met" notice is off on `kalman`** (task 116;
  WARMUP-AND-CONVERGENCE §7.11). Its gate reads each row's own
  `sqrt(1 + z'P⁻z / R)`, and `P` settles on `coef_half_life`'s clock, not
  the spec's decay, so neither the notice's trigger nor its projection to
  steady state, which is Kish's, applies to it. At `coef_half_life=50` and
  ten features the rows spread from 1.02 to 1.11 around an average of 1.07;
  at a limit of 1.06, below that average, 40% of rows still pass. One
  withheld row says nothing of the rows after it, and a notice cannot be
  retracted. The other models' notice is unchanged.

- **`kalman(share_p=True)` reads the shared noise once a row** (task 186;
  review round 4, CC3), so no target's noise holds another target's
  residual from the same row. On the review's stream, 394 or 395 of 396
  rows move per target, by a median of 0.004 to 0.066 and at most 3.06.
  Mean squared error falls for both targets: 0.5723 to 0.5494 for one, and
  3.4501 to 3.2305 for the other. The targets still update the shared
  covariance in turn, so their order still matters, as in filterpy's
  sequential form, which now holds it as a second opinion.

- **A windowed `ew_class` reports the class means inside the window**
  (task 186; review round 4, CE1), in `coef` and `ModelBank.coef()`, as its
  docstring said, and null for a class with no row in it. They were the
  whole history's: on the review's stream, a class mean at row 599 moves
  from 2.460 to 4.828.

- **A break inside an `rcov` block restarts its subsampled grids** (task
  186; review round 4, CE8). They feed `omega2`, `iv_sparse`, `iq` and the
  automatic bandwidth, and carried returns across the break. On the
  review's blocks, `iv_sparse` moves from [3.403, 5.561] to [3.085, 5.295]
  and `iq` from [12.10, 27.32] to [9.62, 25.21]; `omega2` at stride 1 does
  not move.

- **`settled_frac` after a drift reset under an embargo counts the held
  rows' clock** (task 187; review round 4, PB1). `drift_action="reset"`
  zeroed it while rows were still held. The first row after a reset read
  0.034 where the definition gives 0.875, and `min_settled_frac=0.95`
  opened 87 rows after the reset instead of 28.

- **`po.corr.shrink(target="identity")` takes Ledoit and Wolf's (2004)
  intensity** (task 188; review round 4, YB4), as
  `sklearn.covariance.ledoit_wolf` does. A constant-correlation term made it
  shrink 16-36% too little: an intensity of 0.0754 is now 0.1051, and a
  shrunk matrix moves by up to 0.063.

- **`po.corr.nearest` stops on Higham's infinity-norm test** (task 188;
  review round 4, YB7): 19 iterations on his 4×4, as published, where it
  took 20. The result moves by 2.4e-9, inside `tol`.

- **`po.eval.sums(weight=)` drops a row whose weight the bank would not
  learn from** (task 188; review round 4, YB21): null, NaN, infinite, past
  the input bound or negative. A null weight was counted in `n`.

- **`po.stream.embargo` warns when its input is a Python scan** (task 188;
  review round 4, YB3). It reads its input twice, once for each copy. A
  source that can be read once, such as `pl.scan_arrow_c_stream` over a
  DuckDB relation, gave the second copy nothing, and half the stream went
  missing in silence. `ConsumedSourceWarning` now says so as the plan is
  built. This package's own plan forms, `with_windows` and
  `lf.online.fit_predict`, are Python scans too and are warned about,
  though over a file they read twice correctly.

- **Every refusal names its spec, key and value** (task 193; review round
  4, PC11, YA4, YA6, PA6, CD16). A refusal of a value gives the value, and
  every one starts `spec "m":`. A type error names the model's key, as
  `[0].model.eps`, where it said `[0].model`. A count past its Rust width
  is refused by name, where `seed=2**64` was blamed on `model`. Loading a
  file of other specs names the spec and the first key that differs. The
  bin budget is given in MiB, `hmm`'s covariance refusal names `hmm`, not
  `ew_class`, and no message cites a document a wheel lacks.

- **The `numpy` extra needs numpy 1.26 or later** (task 191; review round
  4, CI10), the first release with wheels for Python 3.12.

- **`rcov`'s pre-averaged estimate is rescaled as its paper does** (task
  171; review 2026-10-05, CE1). Under `kind="preavg"` and `psd=False` the
  bias-corrected estimate is now divided by `1 − ψ₁/(2ψ₂kₙ²)`, from
  footnote 1 of Christensen, Kinnebrock and Podolskij (2010): `Σxxᵀ` holds
  the integrated covariance beside the noise, so subtracting the bias term
  took a share of the covariance with it. The estimate averaged 0.50 of
  the block's integrated covariance at a window `kₙ` of 3, 0.84 at 6, 0.94
  at 10 and 0.985 at 20; it now averages the covariance itself. Every such
  `rcov` entry moves up by `(kₙ² + 2)/(kₙ² − 4)` at an even window: 19/16
  at 6, 17/16 at 10, 67/66 at 20. `rcorr`, `rcov_n`, `omega2`, `iv_sparse`,
  `iq` and `psd_repaired` do not move, and `psd=True`, `"kernel"` and
  `"plain"` are bit-identical. The paper's authors apply the rescaling in
  their own simulations and data work and leave it out of the text "to
  simplify notation".

- **A reset keeps a window looking ahead whole when its far edge is the
  last row before the reset** (task 173; review 2026-10-05, PC2), as a gap
  or a session change has since task 160. A reset is a step back past
  `restart_after_step_back`, or a session change under
  `session_gap="reset"`. `with_windows` now gives that row its value where
  it gave null, under every `partial`; every other window a reset meets is
  still discarded, null and never dropped. A bank's predictions do not
  move: a reset clears the rows waiting for their labels before any window
  resolves, so the row is not learned, as before.

- **`lasso` counts each target's selection errors from its own `min_weight`**
  (task 174; review 2026-10-05, CA3). `penalty_selected_<t>` adds a row's
  error once the target's own weight has reached its own `min_weight`, so
  one target's choice no longer depends on another target's threshold:
  `min_weight=[0, 60]` and `[60, 60]` chose differently for the second
  target on 84 of the 240 rows from row 60, and now agree. A target that is
  null on some rows counts from where its own weight, not the shared one,
  reaches the threshold, under a scalar `min_weight` too (11 to 26 rows of
  240 moved in the measured cases). Only `penalty_selected` moves; a
  target present on every row under a scalar or an equal list is unchanged.

- **A model window's edge is exact** (task 175; review 2026-10-05, CB1), in
  `ewridge`, `lasso`, `ew_cov`, `ew_class` and `marginal`. The edge and the
  `window_every` spacing are decided on each row's decayed clock held
  exactly -- integer nanoseconds on a temporal clock, the raw value beside
  the time the caps removed on a number clock, the row count without a
  clock column -- where they were decided on a clock the model summed from
  rounded steps. A thousand steps of 1 ms summed to 1.0000000000000004 s,
  so a row exactly `window_size` old was dropped on some rows and kept on
  others: rows 1001 to 1007 of a 1 ms stream under `window_size="1s"` now
  hold 1001 rows where they held 1000. On a number clock the edge is one
  subtraction of the raw values, as the window operators and Polars'
  `rolling_*_by` decide it: on a clock of tenths under `window_size=0.3`, a
  row 0.30000000000000004 back is outside, and 186 of 400 rows hold 3 rows
  where they held 4. `window_every`'s snapshots fall on the exact spacing,
  and `sigma` and `zscore` under a window move with the model's window.
  Decay is unchanged bit for bit; only rows at an edge or a spacing move.
  The window operators decide their edges with the same comparison.

- **`kalman`'s warm-up no longer depends on the target's units** (task
  172; review 2026-10-05, CC4). Before a target has a residual variance,
  its observation noise, and the `σ²` its process noise is derived from,
  are the row's own innovation squared, computed before the update, where
  they were the literal 1.0 in the target's units. `p0` is now a ratio: the
  prior variance is `p0` times that first noise estimate, set on the
  target's first row with a noise, so `p0 = 1` is a prior as uncertain as
  one observation. Before that row a target's `P` is unsized and takes no
  process noise; a row whose innovation is exactly 0, a null target or a
  row of weight 0 sizes nothing and corrects nothing. Scaling every target
  by `c` now scales every prediction by `c`, bit for bit at powers of two,
  at any `p0`: targets at 1e-6 and at 1e6 differed by 0.8 to 1.2 target
  standard deviations in the warm-up and now agree to 1.4e-15. With
  `obs_var` given the prior is `p0·obs_var·I`, so with no process noise the
  filter is the ridge regression with penalty `1/p0`; to keep an absolute
  prior `v`, give `p0 = v/obs_var`. Under `share_p`, before any target has
  a residual variance, the shared noise is the mean of the squared
  innovations of the targets the row observes. Warm-up predictions move,
  and so do the comparisons that read them, such as `seqtest`.

- **`huber` and `quantile` warm up without the target's units** (task 177).
  Before a target has a residual scale (its residual variance above 0),
  `huber` down-weights no row, and `quantile` draws no band but takes
  least-squares rows, as in its warm-up. Both read the scale `s` as a
  literal 1 in the target's units there: `huber`'s cut `huber_delta·s`
  down-weighted every early row of a target in millions as an outlier and
  none of one in millionths (predictions up to 35 times apart), and
  `quantile`'s band put a target in millionths up to 2e5 of its own units
  off. Scaling every target by `c` now scales every prediction by `c`, bit
  for bit at powers of two. Outputs move only from a row judged without a
  scale whose residual exceeded the old cut; no pinned value moved.

- **`session_shrink` is the long run's share of the data, as documented**
  (task 145). At a session boundary `ewridge` mixed its fit with the slow
  twin by accumulated weight, and the twin's weight is many times the fast
  sums'. So every value above 0 reverted almost fully: `0.25` took a slope
  96% of the way back and multiplied `weight_sum` by 20, which also delayed
  the next solve (32 rows against 3). The moments now mix `1 - f` of
  today's data and `f` of the long run's, at today's weight, Kish size and
  prior scale. Numbers move for any `session_shrink` above 0: below 1 the
  fit, and at 1 `weight_sum`, the warm-up gates, the solve schedule and a
  `ridge_scale` fit.

- **The stream's diagnostics run on the clock** (task 146), so their
  numbers do not change with the rows' density, and their fields move:
  - `resid_quantiles` and `ew_cov`'s `mahal_quantiles` are the
    exponentially weighted quantiles at the model's half-life, each value at
    its row's weight, from one decaying DDSketch per slot within 0.78%
    (`tanh(1/128)`) of the exact one. The P² estimator never forgot: at a
    half-life of 10 rows, 3,000 rows after the noise fell tenfold, its 0.9
    quantile read 1.55 where the recent one is 0.166. A quantile now
    reports from the first residual, where P² needed five.
  - `emit_drift` integrates the excess over the clock against a mean that
    decays at the model's half-life, so `drift_threshold` is in `sigma`
    times clock units: the same 30-unit burst was flagged at four rows a
    unit and not at one. With rows one unit apart only the mean's decay is
    new.
  - `emit_autocorr` pairs no residual across a gap capped by `gap_cap` or
    a session change, and a residual with no partner adds nothing to the
    cross moment.
  - A row with no residual, or of weight 0, now ages all three.
  - The docs say what still counts rows: `sgd`'s coefficients under a
    constant rate, `deco`'s linear dynamics, `hmm`'s transitions and the
    conformal step.

- **A row weight scales evidence, not counts** (task 147): where a weight
  entered a count it now enters against the stream's mean weight, so a
  constant multiple of every weight changes nothing, and a stream of one
  constant weight is unchanged.
  - The conformal step reads `w / w̄` over the scored rows: at weight 100
    the same rows swung the band (width sd 3.10 against 0.21).
  - `bocpd` teaches at `w / w̄`, in its run's statistics and in the
    likelihood it passes, and reports a row as one of the mean weight; it
    entered the statistics at its raw weight and the recursion not at all.
  - `quantile`'s warm-up and band floor count rows: rows at weight 100 left
    the warm-up on their first row and the fit reached 1e51.
  - `micro`'s `beta_mu`, `ξ` and the row a summary admits take the mean
    weight; the docs' `ξ` is corrected to what the code computes.
  - `rls`, `kalman`, `sgd`, `ftrl`, `pa` and `hmm` keep a weight on the sum
    scale, as their docs say, and `tests/test_weight_scale.py` names each.

- **`kalman`'s `coef_half_life` is a clock half-life at any row spacing**
  (task 150). The process noise a row adds is `sigma^2 (ln 2 * d / h)^2`
  for a row `d` clock units after the last: `q_i * d^2`. It was `q_i * d`,
  a random walk whose gain grew with the root of the spacing, so a
  coefficient adapted in `h * sqrt(d)` clock units rather than `h`. With
  `coef_half_life=50` on one stream it took 74, 38 and 13 clock units at
  rows 1, 0.25 and 0.04 apart, where `ewridge` took 44 to 59; on the new
  test's stream the Kalman now takes 165 to 176 at every spacing. The
  numbers are unchanged at unit spacing. At every other spacing they move,
  and so do those of a `seqtest` that compares a Kalman among them. An
  explicit `q` is added as `q_i * d^2` too: the noise a row one clock unit
  after the last adds.

- **The docs say what the code does** (tasks 148 and 151). `ridge_scale`
  penalizes the intercept, as RLS does, and reads `coef_prior`'s intercept
  slot; `po.gram.solve` does not reproduce a `ridge_scale` or `coef_prior`
  fit. `weight_sum` is the weight behind the state and settles at
  `1 / (1 - λ^d)` for rows `d` clock units apart, not `1 / (1 - λ)`; the
  Kish size likewise. `emit_selected` and `emit_averaged` rank each slot at
  its own half-life. `embargo` without a clock counts every row of the
  group, a skipped one included. The README's account of the doubled
  stream is the current one: every residual diagnostic parts from it.
  `hmm`'s `transition` is the prior's mean, not seeded counts. `ftrl`'s
  `l1`, `l2` and `beta` are a prior of fixed mass against evidence that
  grows with weight and density, and `lasso` is the penalty on the mean
  scale. `micro`'s thresholds stay in points: a summary meant to hold a
  share `s` of a stream of `v` rows per clock unit needs
  `beta_mu ≈ 1.44·s·v·h`. No numbers move.

- **`po.stream.embargo` keeps an integer clock's dtype** (task 160). A whole
  delay, `5.0` as well as `5`, is added in the clock's dtype, where merging
  the two copies died in a `SchemaError`. A fractional delay on an integer
  clock is refused, and so is a frame that already holds the
  `role + "_weight"` column.

- **`po.eval` reads as missing what the bank reads as missing** (task 160).
  `metrics`, `rolling_metrics` and `sums` drop rows whose target or
  prediction is NaN, infinite or beyond 1e100, where `r2`, `ic` and `mse`
  came out NaN. `metrics` reports `r2`, `ic` and `hit_rate` as null where they
  are undefined, as `from_sums` does, not as an infinity or a NaN.
  `rolling_metrics` refuses a window that is not finite.

- **Durations compare by their length** (task 160). `"5s"` is `"5000ms"`:
  a `with_windows` state resumes under another spelling of the same
  length, and a bank loads under a half-life grid spelled another way,
  labelling its outputs the caller's way and keeping `ew_cov`'s PCA sign
  continuity.

- **Under an embargo, drift is flagged on the row that released the label
  that tripped it** (task 160). The flag was never written under an
  embargo, and `drift_action="reset"` restarted the model without one. The
  restart now takes effect before the releasing row is scored.

- **A window counts every row with a weight for its held-value rule**
  (task 160). In `marginal`, `ew_cov`, `ewridge`, `lasso` and `ew_class`, a
  row lighter than `1e-12` of the history did not count, so a window made
  of such rows read each value as held at the last heavy row's. It now
  reads their moments, and a light row with another value ends a held run.

- **`ewridge` reports no fit where its first solve fails** (task 160): a
  NaN `coef` and a null prediction, where it reported zeros. A combination
  skipped for having no weight at `ridge=0` reports no readiness shares,
  where it reported the previous solve's.

- **The command line checks its paths before it reads any input** (task
  160). A `save_state`, output or `closed_groups` path that is a directory
  is refused before the run. `--dry-run` opens the input, and names one
  that is not there, where it said "config OK". `--no-output --predict` is
  refused, naming `--predict`.

- **A `quantile` row outside the band moves the fit at most to its own
  target** (task 160). Its nudge was bounded by the row's leverage under the
  band Gram's diagonal alone, and on correlated features the next solve
  moved the fit at that row by 6 to 127 times its residual. The bound is
  now the full leverage against the band's own system. Predictions move
  where the bound binds, and the speed it costs is under *Performance*.

- **A zoned `Datetime` group or session column works** (task 160), in the
  bank, `with_windows`, `refresh_time` and `skip_learned`, where it was
  refused with polars' inner message. Its key is its instant, written as
  the UTC wall time, so the same instants shown in two zones are one group,
  and two instants a zone shows at one wall time are two. `with_windows`
  and `refresh_time` return the column's own values.

- **`hit_rate` leaves out a prediction of exactly zero** (task 195; review
  round 4, S6), in the bank's `emit_metrics` and in `po.eval`, as it leaves
  out a target of zero. The bank's sign test called a prediction of 0 "up",
  a hit on every rising row, where `po.eval` scored it a miss.
- **A Poisson `sgd` fit's `hit_rate` is null** (task 195; review round 4,
  S5): a rate and a count have no sign to hit, and it read 1.0 whatever the
  fit.
- **`sgd(loss="logistic")` clamps a label into [0, 1]**, as `ftrl` does
  (task 195; review round 4, S4); `strict_binary=True` refuses the chunk
  instead.
- **A late target is solved on the row its own weight reaches its
  `min_weight`** (task 195; review round 4, S9b), in `ewridge`, `lasso`,
  `huber` and `quantile`, where it waited for the next scheduled solve: on
  the review's stream it predicts from row 24, where 56 rows were null.

### Performance

- **1.0 against 0.13.0, measured before the release**:
  `scripts/benchmark.py` at 200,000 rows, seven rounds of each build in
  turn, run twice on an Apple M4 Pro, at load 4.0 to 9.0 and then 3.0 to
  4.7. The two runs agree within 2% on every configuration, with 1-13%
  scatter within a build. Most configurations run at 0.94 to 1.20 of
  0.13.0's rows a second. Thirteen are as fast or faster: `bocpd` with a
  full covariance at 1.20, `ftrl` at 1.10, `ew_cov` with
  lags or `mahal` at 1.08, `kmeans` at twenty features at 1.07, `hmm` at
  twenty features at 1.05, and `ewridge` at fifty features, `ew_class`
  with a full or shared covariance, `corrchange`, `deco` in blocks and
  `hmm` at four features at 0.99 to 1.04. These run slower, most of them
  light configurations where a row costs a few hundred nanoseconds and
  the bank's per-row work weighs most:

  | configuration | of 0.13.0 |
  |---|---:|
  | `bocpd`, four features, `diag` | 0.74-0.76 |
  | `micro`, four features | 0.80 |
  | `ewridge`, five features | 0.81 |
  | `seqtest` | 0.83-0.84 |
  | `sgd`, without standardizing | 0.85 |
  | `kalman`, twenty features, with or without `revert_half_life` | 0.87-0.89 |
  | `pa`, without standardizing | 0.88-0.89 |
  | `ewridge`, ten targets or five half-lives | 0.90-0.91 |
  | `deco` (within its 10-13% scatter) | 0.91-0.92 |

  The first measurement, before the fixes below, read 0.72 to 0.98 for
  most configurations, `sgd` and `pa` 0.63 and 0.64, and `bocpd` 0.25.
  Each step down was bisected to the commit that made it, and each is the
  cost of a correction listed in this release's entries:
  - Every row's place on the clock held exactly (task 175, beside tasks
    153, 176 and 200): `seqtest` went from 0.97 to 0.84 of 0.13.0 at that
    commit, `sgd` from 0.98 to 0.87, and `ewridge` at five features from
    0.95 to 0.89. Every stream stamps every row, with a window or without
    one, and the stamps are saved with its state, so skipping them where
    nothing reads them would change the state file.
  - `sgd` checks that every gradient of a row is finite before any
    coefficient moves (task 186): `sgd` went from 0.83 to 0.76.
  - `kmeans`' and `micro`'s centres are compensated pairs (task 215): 0.89
    to 0.81 and 0.81 to 0.76.
  - `bocpd` sets its prior from its first rows, with
    `degrees_of_freedom_prior` 3 under `diag` (task 195). Before the fixes
    below, 921,000 rows a second became 234,000 at four features, at
    `max_run` 20 or 200; given 0.13.0's prior (`mean_prior` of zeros,
    `covariance_prior=[1.0]`, `degrees_of_freedom_prior=6`), 1.0 ran at
    879,000. The cost follows the prior, not the code.
  - The rest comes a few percent at a time from the corrections between
    0.13.0 and 1.0, the value checks of task 183 among them.
- **Eleven fixes took back most of that, every output and state
  byte-identical** (task 250), each found by sampling a symbol build and
  each held to `scripts/release_probe.py`'s 297 columns and 31 states:
  - The clock: the check of a row's step shares one refusal path, and a
    float clock's forward step takes a short cut. `seqtest` runs about
    1.18x faster.
  - `bocpd` allocates nothing per run, and caches its gamma terms keyed on
    the degrees of freedom and the density's dimension: 2.8 to 2.9x
    faster: 0.75 of 0.13.0 under the default prior, and 1.8 times
    0.13.0's speed when given 0.13.0's prior.
  - `kmeans` and `micro`: inlining, a clean flag per batch, `micro` reuses
    the squared distances its search found, and a step factor is cached on
    the step's bits. `kmeans` at four features runs 16% faster.
  - `sgd`, `pa`, `ew_cov` and `kalman`: the gradient check folded into the
    step, `pa` reads its features in place, the rank-one loop moved out of
    line, and the noise median kept on the stack. Each runs about 15%
    faster a row.
  - The bank: the run budget counts the plans, the embargo's stamp sits
    beside the plans (128 bytes a plan, from 160), and the summary's
    Welford step runs across columns at eight features or more.
  Two release builds of the same source differ by up to 19% in wall time
  from inlining alone, so a whole-build ratio within about 15% can be the
  layout rather than the change. The instruction-count budgets in
  `budgets.toml` measure the change itself.
- **A sharded `marginal` (`shards=`) steps each target through a held
  batch in turn, and cuts its feature ranges on even features** (task
  127): on one thread up to 24% faster at thirty targets, and 9-16% where
  the equal shares fell on odd features; the numbers are unchanged to the
  bit. E74's pair-major run was measured and not built (PERFORMANCE §39).
- **A standardizing `kalman` holds its state at an anchor and re-maps it
  only when the moments drift past a factor of 2 in scale or one scale in
  mean** (task 211): the same filter to rounding, 277 to 221 ns a row at
  ten features.
- **`kalman` with `standardize` costs about +85% per row** (task 206: the
  exact re-map through each move of the scaler; 142 to 270 ns at k = 10).
  A re-map only past a 10% move was measured at 141 against the exact
  195 ns in a replica, within 0.046 of a noise std of it, and not shipped.
  `sgd` and `pa` are 2-3% faster after the warm-up.
- **The core checks every value a model is handed** (task 183; *Fixed*
  says what it buys). The inlined check costs `sgd` and `pa` 6-16% of a
  core `step`, 1.4 to 4.5 ns a row, and `kalman` 3-6%. Through the bank,
  which checks first, `sgd` and `pa` run 3.4% and 4.7% slower at two
  features, and other models 0.8-2.3%, inside the measurement's spread.
  The cost was accepted.
- **`micro`'s linkage holds at most 128 MiB** (task 193 and its merge;
  review round 4, CF2). Up to 4,096 potential summaries it keeps their
  squared distances in a matrix, as before. Past that it takes each pair's
  distance when it needs it, in memory linear in their count, where the
  matrix was `max_clusters²` doubles: 8 TB at a cap of 10^6. That path,
  measured at 200 summaries, is 14-18% slower a checkpoint. Labels, counts
  and the threshold are bit-identical on either path.
- **`RefreshTime.feed`, `Windows.feed` and `Windows.finish` release the
  GIL** (task 187; review round 4, SF12), as the bank's calls do, so other
  Python threads run while they work.
- The nanosecond clock's conversion to seconds takes a 64-bit road when the
  difference fits one, as it does for any two stamps less than 292 years
  apart (task 143). It runs the same two Euclidean operations, so every
  clock in the library gets the same bits. The 128-bit division it skips
  took a tenth of each row of the window core.
- **`quantile` on its default solve schedule runs at about half its former
  speed** (task 160): 2.6M rows a second against 5.0M at 10 features, the
  cost of bounding each nudge by the full leverage (above). Solving every
  row it is 8% slower. A way back that keeps the results to the bit, by
  skipping the leverage where the bound provably cannot bind, is a
  follow-up.
- **`quantile` takes back about a sixth of that, every output bit
  unchanged** (task 170). A nudge reads its leverage from a kept column
  without allocating, the band system reads the Gram in place, and each
  Cholesky factorization allocates one matrix where it allocated two.
  Through the bank at ten features: 2.83M rows a second against 2.44M on
  the default schedule, and 1.17M against 1.01M solving every row, measured
  side by side. The core's `SpdFactor` can now be moved in place to the
  factor of `c·A + v vᵀ` or of `E·A·E` in O(k²), held to faer's fresh
  factorization. At `ridge=0` a `quantile` fit uses it: a row inside the
  band moves the kept band factor instead of refactorizing it at the next
  nudge, about 4% more through the bank (2.97M rows a second against
  2.85M), and the factor is saved with the state, so a resumed fit goes on
  to the bit (about 670 bytes more per target at ten features). A moved
  factor holds its matrix to rounding, not to the bit, so at `ridge=0`
  predictions move at that level where a nudge's bound binds: 1,357 of
  3,000 on a stream with features correlated at 0.999, by at most
  3.4e-13 relative. Under a ridge the factor is rebuilt as before, since
  forgetting shifts a ridge off a rank-one step, and `huber` keeps no band
  system.
- **`huber` and `quantile` pay for `support_coef` by default** (task 116,
  B): at ten features a `huber` row goes from 205 to 215 ns, about 5%, and a
  `quantile` row from 340 to 354, about 4%, the last solve's factor held
  until the shares are read, as `ewridge` holds its systems; an eager copy
  cost about 10%. The rest of task 116 costs nothing unless set: the
  `kalman` gate or row field 115-120 ns a row; `rls`'s second weight sum
  within the noise, its gate 1 ns and its row field 70 ns; `lasso`'s gate
  15-20 ns; `se_coef` at the default `coef` cadence 10 ns on `ewridge` and
  within the noise on `rls` and `kalman`, and with `coef` on every row
  `ewridge` 445 to 950 ns, `rls` 440 to 1,315 and `kalman` 330 to 465.

### Fixed

- **The API review's fixes** (task 239): `coef_stats` and `solve_subsets`
  read least squares through the origin when a fit has no intercept (the
  old centred reading put a standard error at 3.7 times statsmodels') and
  give an intercept its standard error; `bocpd`'s `max_run` caps a run's
  length (with pruning the oldest run grew past it); `withheld_reason`
  gains `below_warm_rows` for the rows `kmeans`, `hmm` and `bocpd` hold to
  seed themselves, and `ew_cov`'s floor reads `below_min_weight`;
  `se_coef_hac`'s default lags follow the shared rule, `2·ceil(embargo)`;
  `po.sim`'s smooth design blends the ramp its `mix` reports;
  `output_index` gains `lag`, `state` and `class`; `pa` refuses `c` under
  `mode="pa"` and `marginal` empty `lags`, by name.
- **`po.gram.lars_path`: columns that tie are settled at one knot** (task
  236). A column whose correlation tied the active set's never entered, so
  an identity Gram with `d = [1, 1]` ended at `[1, 0]` rather than least
  squares `[1, 1]`. Steps within 64 ulps of `λ_max` of the smallest are now
  one knot, entering or leaving as the lasso admits; `max_steps` counts
  knots, and `max_active` can be passed at a tie. A non-finite `d_j` never
  enters (an infinite one hung the walk). Paths without ties are unchanged
  to the bit.
- **`audit`'s MAD past `distinct_cap` reaches the median absolute
  deviation however far the column's range** (task 234): its bisection
  stopped after 200 halvings, so one value at 1e70 beside a spread of 1
  read a MAD of 6.2e9 (true 0.7), and the robust z meant for that outlier
  read nothing. `po.gram.lasso_path` gives +0.0 where numpy and `lasso` do,
  not -0.0.
- **A wide fit is the same to the bit whatever the thread count** (task
  231): from 512 columns, `ewridge`'s one-column solves and the
  eigendecomposition behind `ew_cov`'s `pca` and `rcov`'s repair followed
  the pool's size in their last bits. One-column solves moved once, by at
  most 1.5e-11 relative at 4,000 columns.
- **Review round 6** (PLAN §20): the windowed `ljung_box` is Box-Pierce at
  Kish's size (Ljung-Box's factor passed 15.8% of clean streams at
  half-life 10); the tails keep central moments (a far first residual read
  a kurtosis of -129,065); under `embargo` a row is folded with the error
  inflation it was scored with; `po.gram.vif` reports `inf` only for a
  constant column, and an exact duplicate no longer reads 0.25;
  `solve_subsets` reports NaN, not 0, standard errors where the system is
  not positive definite; `po.corr.epps_invert` reads packed and float32
  Grams; `lars_paths` takes Grams whose targets differ; a column LARS sets
  aside makes no knot; `audit`'s Dickey-Fuller is ±inf for an exact trend;
  a bank saved under one `gram_threads` resumes under another;
  `online.unnest` expands the robust standard errors; `check()` reports
  steps back for an audit-only bank and the statistic that fired for
  `heavy_tails`.
- **A windowed `lasso` or `ewridge` at a level of ±1,000 or more no longer
  reads the means' rounding as spread once a dominant row leaves the
  window** (task 217, D4). A `lasso` diverged to `inf` and an `ewridge`
  held slopes of -6.6e41 where the rows inside give 4.5e-49. The
  truncation's bound now includes the rounding the subtraction of two
  means at a level leaves, `64ε·(g·C + ratio·C_u + ratio·g·|d|·(|m| +
  |m_u|))`, and such a window reads what it reads at level 0, to the bit;
  ordinary windowed streams do not move.
- **A windowed `marginal` pair no longer reads a truncation's rounding as
  moments** (task 217). With a dominant row leaving the window, `var_y`
  read 6.1e82 where the rows inside give 4e-4, and `beta` and `corr` had
  the wrong sign, at level 0 as well. `cut` takes the same bound.
- **A windowed `ewridge` or `lasso` fit has no slope on a target with no
  spread in the window** (task 217): the slope came from the target's mean
  rounded at its level, 6.6e-4 at a level of 1e8.
- **A window holding none of a target's rows reads nothing for it, as for
  a target never seen** (task 217; `ewridge`, `lasso`, `marginal`,
  `ew_cov`, `ew_class`, and the residual spread behind `sigma`, `zscore`
  and the conformal band). A target absent for some 1,060 half-lives, or a
  window of zero-weight rows, read a weight of 5e-324 that per-row decay
  never takes to 0, predicted from it on its return, and read a `sigma` of
  0 or 1; two such returns parted under `pairwise`. The window now counts
  each target's rows of positive weight, and all rows (schema 52).
- **A target absent for some 1,060 half-lives while the others go on no
  longer keeps its own history at a subnormal weight** (task 217;
  `ewridge` and `lasso`, `target_gaps="own_rows"`): the solve on its return
  read the old fit back from it, and two returns parted by up to 7.6e-2.
  Such a history is forgotten, as a decay that underflows forgets it.
- **An `ewridge` fit whose every feature the standardizer dropped reads as
  the intercept alone** (task 218), with `emit_error_inflation` or
  `emit_se_coef` on: the documented `error_inflation`, `sqrt(1 +
  1/n_kish)`, and the intercept's standard error, where it read `inf` and
  null live and other values after a save and load until the next solve.
  A resumed stream now matches the uninterrupted one.
- **`kmeans` and `micro` centres are compensated pairs** (task 215): a
  value held at 1e12 is reached to the bit, where the centres stalled
  short of it and mis-assigned rows 40-60 half-lives later.
- **`bocpd`'s predictive mean mixes deviations from the most probable
  run's mean** (task 215): at a feature level of 1e12 it carried the
  posterior's rounding times the level (2.3e-13 of it); now within one
  rounding step of the level.
- **A `lasso` whose history the decay leaves in the subnormal range
  forgets it** before the row it meets (task 215), where it read its old
  fit back from it and parted by up to 4.66 from the stream whose decay
  underflowed.
- **`hmm` reads its transition matrix from counts aged by the row's
  clock**, so a zero-weight row is clock alone (hard rule 9); after a gap
  that takes the counts to nothing it reads the prior's mean (task 214).
- **`ewridge`, `lasso`, `huber` and `quantile` never solve on a zero-weight
  row, nor count one toward `max_rows_between_solves`** (task 214): a solve
  due on such a row falls on the next row with weight, so the solves fall
  where they would without it.
- **A standardizing `kalman` sizes its prior from the median of its first
  three squared innovations** (scaled to keep `p0` meaning `p0` times the
  noise variance) **and re-sizes a covariance that loses its positive
  diagonal** (task 214): a target at the input bound among the first rows
  no longer leaves the filter stuck.
- **`kalman` keeps hard rule 9: a zero-weight or null-target row inside a
  clock gap no longer moves later predictions** (task 211). The process
  noise `Q·d²` was charged per row, and a square is not additive over a
  split gap, so such a row shrank the gap's noise (0.197 at a
  `coef_half_life` of 20). Each covariance now keeps the clock since it
  last took a row that observed its target, and is charged for all of it
  then. `po.stream.embargo`'s doubled stream equals the native embargo
  where no gap is capped, and a target seen on one row in 25 forgets on
  its clock like a dense one (100 clock units, where it took 462).
- **A standardizing `kalman` sizes each coefficient's prior once its
  feature's scale is usable**, from the mean squared innovation over at
  least three rows, and follows the moments from its first row with no
  warm-up (task 211). The prior was sized on row 0 against empty moments
  from one squared innovation: short histories' R² goes from 0.72 to
  0.95, and a feature level of 1e8 no longer moves a prediction.
- **Under `share_p`, `kalman`'s `se_coef` and `pred_var` read each
  target's own noise** (task 211); they read the mean, 7.3 times too wide
  for a quiet target.
- **Under `share_p`, `kalman`'s shared noise is the mean residual variance
  over the targets that have one** (task 208, A1). A target with none yet
  counted as a noise of 0: a copy null for its first 300 rows moved its
  twin's predictions by up to 0.45. The order of the targets moves no bit,
  and a copy present on the same rows moves nothing.
- **The "has not been met" notices' wait is kept in the state file**
  (task 208; schema 46). A stream saved and loaded inside the wait gives
  its notice on the row an unbroken stream would; one saved and loaded
  each day with a long half-life never gave it.
- **A frame with a `UInt128` column is refused by name**, whether a spec
  reads it or not, on every surface a frame crosses to the extension on,
  with a cast to make (task 208). It panicked inside Polars' conversion.
- **A `min_weight` notice gives no projected ceiling it cannot stand
  behind** (task 208). A target no row has given a value with a positive
  weight is named as such; a projection that is not a positive weight --
  a target present on one row in 50, a heavy row where the notice comes --
  is said to be unprojectable, where they read "tops out near -0.0256 ...
  Lower min_weight below -0.0256" or a ninety-digit negative. A figure
  prints with four significant figures outside 0.001 to 1,000,000.
- **The `min_weight` and noise-gate `ReadinessWarning`s no longer tell a
  stream whose rows then come faster that its floor "cannot be met ... for
  good"** (task 208; review round 5, F3). The README's own kernel example
  met its floor on 381 of 400 rows after the notice. Each now fires only
  after its gate has withheld every row for a further half-life of the
  learned clock past 95% settled; a row that meets it restarts the wait.
  The notice says what the row rate so far implies, not what can never
  happen.
- **`summary()`'s `settled_frac` counts the clock covered by the rows held
  under `embargo`, as each row's field does** (task 208, C4), so it reads
  what the next row will: 0.97632 became 0.98325 at `half_life=10`,
  `embargo=5`. `weight_sum_settled` is unchanged.
- **An `Int128` clock is refused by name** by the bank, the command line,
  `with_windows`, `refresh_time` and `po.stream.embargo` (task 208, B3). It
  was read as a double and lost its steps of 1 past 2^53. `po.increment`
  takes an `Int128` input's step in integers, exact past 2^63.
- **Review round 5 (2026-10-08): the changes since round 4, read by seven
  reviewers** (PLAN §19). The fixes with a plain shape:
  - Under `embargo` the two "cannot be met" readiness notices read how far
    the rows the model has learned from have settled, as `weight_sum_settled`
    does. Paired with the row's held-inclusive `settled_frac`, the ceiling
    read negative before anything was learned, and a floor the stream then
    met was called unreachable (C1).
  - A state file whose `emit_metrics` accumulators are a value short is
    refused at load, naming them, where it loaded and the next scored row
    panicked (C2). A `decay_time` of the wrong length is refused rather
    than replaced with zeros, which restarted `settled_frac` and
    `weight_sum_settled` from the load (C7).
  - `--skip-learned` keeps a row whose clock is NaN for the bank to refuse
    by row, as `ModelBank.skip_learned` does, instead of dropping it in
    silence (C3).
  - `po.stream.embargo` on an integer clock within `delay` of its dtype's
    top raises `InvalidOperationError` when the plan runs, where Polars'
    wrapping add put the learn copy before the stream, a silent look-ahead
    (B2).
  - `po.eval.metrics`, `window_metrics` and `sums` null a Poisson `sgd`
    fit's `hit_rate` when `spec=` names the loss, as the bank does (E1).
  - `sgd`, `holt` and `ew_cov` refuse a NaN or negative `min_weight`, and
    `ewridge`, `lasso`, `huber` and `quantile` a NaN `solve_every`, through
    the Rust API and a state file, as the spec layer does (C6).
  - A `lam` decay factor at a step of one clock unit is `lam` by
    construction, no longer the platform's `pow(lam, 1)` (A4).
  - `ModelBank.to_json` raises `UnstableWarning` for a formula target's
    written form under `POLARS_ONLINE_WARN_UNSTABLE=1`, as `save` does (D7).
- **The once-only warning that a coefficient is more ridge than data waits
  until the stream is 95% settled** (task 116, G), the rule the other two
  readiness notices keep. It fired on the first row the gates let through,
  where the first rows' noisy feature variances under a mean-form ridge read
  a share below 0.5 that later settled above it: `ewridge` on two correlated
  features with `half_life=200`, `standardize=True` and `ridge=0.7` settles
  at `support_coef` 0.51-0.53, and 8 seeds of 12 warned at row 4. A stream
  without decay never settles, so it never warns; `support_coef` and
  `summary()`'s `min_support_coef` carry what it reads.
- **`kalman`'s `share_p` takes each row once, so neither the order nor the
  number of targets moves a prediction** (task 204). The shared `P` took
  each row once per target, as if the targets shared their coefficients.
  A target beside an exact copy of itself moved by up to 0.92 on a spread
  of 1.1, and swapping two targets moved predictions by up to 0.36. Every
  target observed on a row now takes its gain from `P` as the row finds
  it, and `P` takes the row once. The mean noise is summed in ascending
  order. A target beside a copy now predicts as it would alone, to the
  bit, and every order gives the same bits. On `docs/VALIDATION.md`'s two
  targets, `share_p`'s R² moves from −0.054 to −0.079 and from +0.003 to
  −0.006; a `P` per target gives −0.109 and −0.005. `share_p` is off by
  default, and the state's layout is unchanged.
- **Every model refuses a value that is not a usable number, as the bank
  always has** (task 183; Rust API only). A feature, target or weight that
  is NaN, infinite or past the input bound of 1e100 reached the core
  through the Rust API. Most models learned it into their state, and the
  rest reported it or counted it. A target that is not usable is now
  absent. A feature or weight that is not usable makes the row a row of
  weight 0 that keeps and counts nothing, with NaN predictions for a
  feature. One contract test holds all 20 models to it, and
  `online_core::usable` is public. Through the bank nothing changes.
- **A damaged state is refused by name at load** (task 185; review round 4,
  CF4, CA2, CA3, CA4, CB2, CB8, CE3, PD1, PB4, PA8, PA2, CF11). Every
  model's restore checks the configuration a state carries, as its `new`
  checks a fresh one, and every shape the next row reads: window snapshots,
  kept leverage systems, a factor's order, a quantile sketch, a lag ring,
  the `with_windows` core, held embargo rows and a bank file's per-spec
  entries. Each panicked, at load or at a later row, or ran on NaN. A file
  from a newer version is told to upgrade, not to refit, and the core's
  refusal names the schemas it loads.
- **`sgd` skips a row whose gradient is not finite** (task 186; review
  round 4, CC2), or under AdaGrad a row whose squared gradient is not, so
  `clip_gradient=inf` no longer kills the stream: every prediction after
  such a row was null.
- **Four crashes, fixed** (tasks 187 and 188; review round 4, PA1, SF2,
  YB13, PD3). A Boolean group column that a formula target also reads no
  longer panics. Neither does `predict` on a fresh bank or on unseen
  groups, nor `fit_predict` on an empty grouped frame, in a bank with a
  `seqtest` comparison. `po.corr.loss` of a singular forecast is NaN, as
  documented, where it raised `LinAlgError`. A hand-built `@po:` column is
  refused as a reserved name, where it raised `IndexError`.
- **`online --dry-run` refuses what the run refuses at its first step**
  (task 187; review round 4, SF2): a missing or mismatched `--resume`
  state, and an input or `keep_columns` without a column a spec reads. It
  said "config OK".
- **`describe()` reports a formula target's values** (task 187; review
  round 4, PA9): the statistics of the values the target was learned from,
  where every row read null.
- **Saving over a state file keeps its permissions on Unix** (task 187;
  review round 4, PA4): a `chmod 600` file stays 600.
- **The JSON export refuses a broken bank in Rust, as in Python** (task
  187; review round 4, PA12), and the Python wrapper no longer encodes the
  whole state twice to check.
- **A spec with `half_life=inf` is order-free, as one with `lam=1.0` is**
  (task 187; review round 4, TB2), written as a float or as `"inf"`, so
  `fit` no longer warns `OrderNotGuaranteedWarning` for it.
- **The chunked runs name the method the caller used** (task 187; review
  round 4, SF11) when they refuse a chunk that is not a frame.
- **Four edge cases read their numbers right** (task 186; review round 4,
  CA1, CB7, CE10, CE9). An `ewridge` window emptied by a gap reports a null
  `support_coef` beside its null fit. A windowed variance whose co-moment
  is not a number stays NaN, where it read 0. `bocpd` keeps a row whose
  predictive fails out of `weight_sum` and the weight mean, and still
  counts it as a failure. `rcov`'s `psd_repaired` is null where the repair
  could not run, where it read `false`.
- **`po.gram.merge` keeps `n_kish` across a part that learned nothing**
  (task 188; review round 4, YB6), where it reported `None`.
- **`po.stream.embargo` refuses a clock that is neither numeric nor
  temporal** (task 188; review round 4, YB15), by name and with a
  `TypeError`, where polars failed when the plan ran.
- **`min_samples` takes any integer up to 4,294,967,295** (task 188; review
  round 4, PD4), numpy's included, and both sides name that ceiling.
- **`po.increment` of a `Time` column gives the step in seconds since
  midnight** (task 188; review round 4, PD5), negative across midnight, as
  its docstring says, where it was refused.
- **A window operator refuses a word or duration where it is written**
  (task 188; review round 4, PD9): `"nan"`, `"-inf"`, `"reset"`, an infinite
  `window_size` word, and a duration at or below 0. Each failed later,
  under a serde path.
- **The Rust API is held to the spec's rules** (task 193; review round 4,
  CA6, CC7). A NaN or infinite value that made a model learn nothing is
  refused by name, as a spec refuses it.
- **Four more ways a NaN feature reached a state, closed** (task 182;
  Rust API only, the bank passes no NaN feature). `hmm` refuses a row that
  is not finite before its states are seeded, where it buffered the row and
  seeding replayed it into a state, after which every row failed; `deco`
  learns such a row as a row of no weight, where its standardiser learned
  the NaN and `loglik` was NaN ever after; `huber` and `quantile` do not
  learn one (it still counts in `n_eff` and the target's present weight),
  where it froze the fit and failed every later solve; and a NaN variance
  in `EwCov` and `EwDiag` reads NaN where it read 0, so `ew_cov` emits NaN
  `var` and `std` for such a column. Finite inputs move nothing.
- **`solve_every`, `pca_every` and `prune_every` are decided on the exact
  clock** (task 180), as `window_every` and `coef_every` are. Each summed
  its clock in doubles, so on a temporal clock `"2s"` on rows 1 ms apart
  fired each solve, PCA refresh or `micro` checkpoint a row late, and the
  lateness built up (2,001 rows apart). They are now decided in integer
  nanoseconds from the last event on a temporal clock, and by one
  subtraction of the raw values on a number clock. Where an event moves,
  the fit, the components or the clusters change on another row, with
  every field read from them between. Default cadences, whole-second and
  row-count cadences, and a model stepped through the core without stamps
  do not move.
- **A quadratic form of a NaN is NaN, not 0** (task 181). The clamp that
  lifts a rounding below zero to 0 used `max(0.0)`, which also turned a NaN
  into 0: a vector holding a NaN read as sitting on every mean. It keeps
  NaN now; every finite form keeps its exact bits. Through the Rust API,
  where a NaN feature can arrive (the bank passes none): `hmm` refuses such
  a row in every shape (it reports nulls, counts a failure if it carries
  weight, ages the clock) where `full` and `shared` learned it and any
  shape at weight 0 wrote NaN into every co-moment; `ew_class` withholds a
  row whose class score is NaN instead of classing it; a `quantile` nudge
  whose leverage is NaN takes no step instead of an unbounded one;
  `ewridge`'s row error inflation and `deco`'s `loglik` are NaN where they
  read a NaN form.
- **`bocpd`'s `gaussian` emission refuses a row whose feature is NaN**
  (task 179), as its other emission does. It learned such a row, read
  through a quadratic form that clamps with `max(0.0)` -- which turns NaN
  into 0, as on every run's mean -- and every learned row after it was a
  solve failure. Reachable only through the Rust API: the bank passes no
  NaN feature.
- **An `embargo` is decided on the elapsed clock held exactly** (task 176;
  found by task 175's worker). The countdown subtracted each row's elapsed
  time from the embargo in doubles, so it drifted both ways: on rows 1 ms
  apart under `embargo="2s"` every row was learned 2.001 s after it arrived,
  a row late; on a clock of tenths under `embargo=2.0` the row at 0.3 was
  learned at 2.3, though `2.3 − 0.3` is 1.9999999999999998, a row early; and
  under `"100d2ns"` a row was learned a nanosecond early. The release now
  compares two rows' places on the elapsed clock: integer nanoseconds on a
  temporal clock, one subtraction of raw values on a number clock, the
  row's place without a clock column, inclusive at exactly the embargo as
  before. What counts as elapsed (a gap in full, task 153), a break
  releasing nothing early, a reset clearing what is held and a formula
  target's release are unchanged. Where the countdown drifted, every output
  from the release on moves by one row's learning; a clock with no column,
  or with whole-number steps, releases where it did.
- **`rcov`'s pre-averaged estimate forms every one of CKP's terms** (task
  159). It never formed the first, `Ȳ₀`, over a stretch's first `k_n − 1`
  returns, so each stretch summed one term fewer than its scale counted:
  0.2% to 14% of a block's diagonal in the review's streams. The scale is
  now `n` over the terms summed, the paper's "true number of summands",
  which across a break inside a block is fewer than `n − k_n + 2`, and
  `rcov_n` counts them. The numbers of `kind="preavg"` move.
- **`ewridge(ridge_scale="sum")` ages its prior across a zero-weight row at
  the head of a stream** (task 159), as `rls`, the same estimator, does;
  after two such rows a clock unit apart the two were 6.8e-2 apart.
- **A step back of exactly `restart_after_step_back` on a `Datetime` clock
  is a late row at every scale** (task 159). A negative delta under a
  second read a last bit large (`−1 ms` as `−1 + 0.999`), so at 1 to 3 ms
  the inclusive edge restarted the model instead.
- **`bocpd`'s `prior_nu` floor message says what the floor buys** (task
  159): a finite mean under `diag` and `robust` (`> 1`), a finite variance
  under `gaussian` (`> d + 1`).
- **`rls` keeps its precision on rows near `1e-155`** (task 158). Its
  rotation took the plain root of `r² + z²` whenever that root was positive
  and finite, including where the squares are subnormal and have lost most
  of their bits: rows scaled by `2^-530`, inside the input bound, gave a
  slope 1.4e-5 off the same rows at scale 1. It now takes `hypot` wherever
  the sum of the squares is below the smallest normal double.
- **An `hmm` given a `transition` matrix with `transition_prior=0` uses it**
  (task 158). A row of the matrix with no counts fell back to uniform,
  where the row is the given matrix at every positive prior; under
  `learn=False` the filter never used the matrix it was given. Such a row
  is now the prior's mean: the given matrix, or uniform when none is given.
- **A list `min_weight` in `sgd`, `pa`, `ftrl` and `rls` checks each
  target's threshold against that target's own weight** (task 158). Each
  model held a target on its own weight, but against the smallest
  threshold of the list, and the bank checked the list against the shared
  weight. So under `min_weight=[5, 30]`, a target present on one row in ten
  first predicted at row 30, where the other regression models wait for its
  thirtieth row. A single threshold, or a list of equal ones, gives the
  numbers it gave.
- **`rcov`'s pre-averaged estimate under `psd=False` subtracts the bias at
  the window actually run** (task 158). Its bias term read the configured
  `theta`, where Christensen, Kinnebrock and Podolskij define θ by the
  window, `k_n / sqrt(n)`. The two part when `preavg_rows` is given, or a
  block's length differs from `block_rows`, and too little noise was
  subtracted: 195 on pure noise whose true value is 0 under
  `preavg_rows=20` and the default `theta`. The numbers of `kind="preavg"`
  with `psd=False` move wherever they parted.

- **`embargo` no longer learns a label before its delay has passed**
  (task 153). A break -- a gap past `gap_cap`, or a session change --
  released every held row at once. So a forward label was learned before it
  was known wherever the break was shorter than the delay. With a 10-unit
  delay and a 5-unit cap, an 8-unit gap learned two labels early, and a
  session change with no gap at all learned nine. The delay now counts the
  time that passed on the clock column, skipped rows included, and
  `session_gap` where a session change restarts the clock. A break's events,
  the lag rings' clear and `session_shrink`'s blend, wait with the row after
  it and run when that row is learned. So the models run one delay behind,
  events included. Where a break is longer than the delay, as overnight,
  the release is where it was.
- `--predict` drops the configuration's `closed_groups` with its
  `save_state`, where the scoring run refused it (review round R2).
- `po.eval.rolling_metrics` names a missing clock column before it reads
  the window (review round R2).
- **`online --dry-run` refuses what the run's first chunk would** (task 154).
  A window target whose embargo is shorter than its window passed the dry
  run, and the run then refused it. A run with `--no-output`, and a scoring
  run, still take any embargo. A dry run whose one product is the closed
  groups now names them, where it said the run's product was nothing.
- **Refusals name what the reader can act on** (task 154). `rcov`'s
  refusals name `preavg_rows`, where they named a `window` parameter it has
  not got. A step back tells the command line to filter its input, since it
  has no `skip_learned`. And `group_close="monotone"` no longer says a
  `Categorical` sorts by first seen: Polars sorts it as text from 1.34.0,
  the floor.
- `ModelBank.to_json`'s docstring said the export refuses a NaN or an
  infinity. It writes each as a string, `"nan"`, `"inf"` or `"-inf"`, and
  the docstring now says so (task 154).
- **An infinity word in a clock parameter is kept as the number** (task
  160). `half_life="inf"`, `"+INF"` or `"infinity"` is held as
  `float("inf")`, so a bank's `specs` equal the dicts it was built from.
- `ModelBank.fit` and `fit_predict_batches` over this package's own plan form
  on an empty input no longer warn `ConsumedSourceWarning`, and
  `po.gram.merge([g])` returns `g` with its group, instance and lags, which
  it dropped (task 160).
- Every example in the API reference builds the spec, bank and output it
  reads, or reads only the README's example data (task 160).
  `ModelBank.skip_learned`'s read a bank it never built.
- **A damaged state file is refused, not a panic** (task 160). A bank file
  whose group states do not match its specs, or whose queued closed rows
  do not fit their spec, is refused at load, where it panicked at load or
  at the first drain; so is an `ewridge` state whose coefficients,
  readiness statistics or session twin do not match its configuration, and
  a `refresh_time` state whose grids do not fit its series. Row counts in
  the bank, its streams and its summaries saturate rather than wrap, and a
  summary whose counts wrap is refused.
- **`ftrl` keeps learning after a row too light or too heavy to square**
  (task 160). With `beta = l2 = 0` and no decay, a row whose squared
  gradient underflows made a coefficient infinite and every later row
  null. A row skipped because its squared gradient would overflow moved the
  next prediction 2.2% before; it now teaches nothing, as a null target.
- **Windowed Kish sizes are null where the window keeps no digit of them**
  (task 160). `marginal`'s and `ew_cov`'s windowed `n_kish`, the
  regressions' readiness Kish size and the Gram's `target_n_kish` were
  rounding, or `inf`, for a window of rows far lighter than the history:
  10.75 against 11 at `1e-6`.
- **A target held at one value over a window has no spread there** (task
  160): `ewridge`'s and `lasso`'s exported `target_vars` read `2.5e-9` at a
  level of `1e8`.
- **`marginal`'s bins hold a target spread up to the input bound** (task
  160): the scale folds at `1e-50`, where the bin variances overflowed above
  a spread of about `1e79` and the split gain read 0.
- **A forward window whose far edge is the last row before a break is
  whole** (task 160). Under `closed="right"` or `"both"`, a window ending
  exactly at the last row before a capped gap, a session change or a
  silent-group cut was null, or dropped under `partial="drop"`.
- An `Int128` group key past the `Int64` range is its own group, where it
  was a null key, and `group_close="monotone"` orders such keys as numbers.
  `drop_groups` drops a group's PCA sign continuity with it, so a state no
  longer grows by an entry for every key ever seen. A
  `POLARS_ONLINE_MAX_THREADS` the system cannot start threads for raises
  the documented `ComputeError`, where it panicked. A `quantile` row of
  weight 0 outside the band no longer moves `sigma`'s last bit. `bocpd`
  names `prune_below` in its refusal, where it said `truncate` (task 160).
- **A source build names the Rust it needs: 1.95** (task 160). The
  declared `rust-version` said 1.85 while the locked dependencies needed
  1.95, so installing the sdist with Rust 1.85 to 1.94 failed inside a
  dependency's build. Cargo now refuses an older Rust up front, by name.
- `coef_every`'s doc says what it counts: each group's accepted rows, rows
  of weight zero and rows with a null target included, where it said
  learned rows (task 164).
- **`kalman` withholds a prediction that overflows** (task 158). A feature
  at the input bound, standardized against a scale earlier rows set, made
  `z · β` infinite, and `step` and `predict` reported `inf` for that
  target; both now report it as missing (a null in the output). Through
  the bank the output already read as null.
- **A subnormal half-life is refused by the window operators** (task 159,
  W5). One below `f64::MIN_POSITIVE` passed the check, and its mass
  underflowed: a mean of 0 and an infinite rate.

- **An integer clock is exact at any size** (task 200). Its steps, stamps,
  window edges, cadences and embargo releases are taken in integers, where
  an `Int64` of epoch nanoseconds was rounded to 256. `po.increment` of an
  integer column takes its step in integers too, so a running count past
  2^53 keeps its steps of 1.
- **Without a cadence, `coef` is written on each group's last accepted row
  of a chunk** (task 194; review round 4, S2). A chunk ending in a skipped
  row carried none.
- **`corrchange`'s permutation critical value is redrawn every
  `permute_every_rows` reports**, as documented, not one more (task 196;
  review round 4, S1). In the REGIMES benchmark the window test's false
  alarms go from 51.33 to 53.67 per 1000 rows and its median delay from 29
  to 25 rows.
- **The noise gate's notice no longer says "cannot be met" of a gate that
  opens later** (task 198). It read the ratio at 95% settled, which is still
  falling there; it now projects it to steady state.

### Tests and documents

- **Performance and memory budgets that only tighten** (task 251).
  `budgets.toml` holds a fixed number for each hot path and output field,
  and every push is measured against it, never against the previous run.
  The 14 instruction-count cases are `crates/online-bench`, counted by
  Callgrind through Gungraun (iai-callgrind's new name), a test-only
  dependency of a crate that ships in nothing. They run twice a push in
  the new `Budgets` workflow on Linux, with the compiler, the image and
  the harness pinned. The memory budgets are the bytes per row of 890
  output fields, every model with every output on, and the peak resident
  memory of three standard runs per platform, held by pytest on every CI
  leg. Over a budget by more than its tolerance fails (2% for counts and
  bytes, 15% for resident memory). So does under it, naming the number to
  tighten to. A raise needs a `[[raised]]` entry with its reason and a
  quoted *Performance* entry here, held over the file's git history. Each
  run reports every case's drift since the last release, and a release
  fails on one past 10% without a `[[signoff]]`.
- **The mutation runs test each mutant under cargo-nextest, inside a
  memory scope.** A process per test, as the gate's tests run: the full
  Rust suite took 84 s under nextest against `cargo test`'s 182 s. The
  extended tier stays in every mutant's run (`--run-ignored=all`). Each
  shard's run is in one systemd scope whose cgroup the kernel holds to 12
  GiB, so a runaway mutant's test is killed alone and counted caught. The
  per-binary `prlimit --data` cap it replaces counted virtual memory, which
  glibc keeps after a free, and failed the baseline of 4 of 32 shards with
  far less resident. A job proves the scope on every run: a runaway is
  killed near the cap, and its parent goes on.
- **Every long example names its steps** (task 249): 326 one-line step
  comments across the README and the guides, as WRITING's rule asks ("a
  long example names its steps in brief comments"); the code and its
  printed output are unchanged. `tests/test_example_steps.py` holds it: a
  block of ten statements or more has no run of more than eight without a
  comment.
- **`scripts/compare_release.py` names a dtype change** where every
  value is equal: an integer widened, or an Enum's categories renamed, as
  `dtype Int32 -> Int64, every value equal` or `Enum categories: gone
  [...], new [...]`. It said "bits differ (e.g. -0.0 against 0.0)" for
  these, which sent a reader after a sign of zero that was not there. A
  row that differs is reported with the dtype change beside it.
  `scripts/release_probe.py`'s table of old names gains the ones
  `scripts/benchmark.py` uses, so the benchmark runs under 0.13.0's wheel
  too: `ew_class`'s `covariance_type` and its words, `bocpd`'s
  `covariance_type` as the `emission` word, `rls`'s `delta`, the
  row-count names of task 196, and `pa`'s `standardize=False`.
- **Faster dev iterations** (task 245). The gate runs tests in parallel:
  cargo-nextest where installed (with `cargo test --doc` beside it) and
  pytest-xdist, in as many processes as the machine has cores free (2 to
  8; `--serial`, `GATE_WORKERS`); the tests that read their process's
  memory run afterwards, alone, and still fail on a planted leak. The full
  gate fell from 501 s to about 230 s, the essentials from 139-191 s to
  48 s, with the same tests and results both ways. A `dev-wheel` Cargo
  profile rebuilds the extension in 4.5-37 s after an edit, against
  release's 189 s, with bit-identical outputs; the gate, CI and the
  shipped wheel keep `release`. `scripts/quick.sh` runs the tests for a
  branch's changed paths; `scripts/clear_caches.sh` clears the build
  caches (dry-run first), and TESTING has "Iterating" and "Caches and when
  to clear them". `.cache` downloads and the release comparison's venvs
  are written atomically, so parallel workers never read a half-written
  file; the gbdt leak check measures after the allocator's ramp. sccache
  was measured and left opt-in: from a new worktree path a third of the
  crates miss its cache (their build scripts' absolute paths enter its
  hash), so a fresh worktree built slower with it (92 s against 66 s for
  the tests, 519 s against 241 s for the wheel).
- Every public docstring was read against the code (task 239): about 120
  wrong facts corrected, about 150 refusals the code made now stated, no
  docstring cites the plan, a task or a record
  (`tests/test_no_plan_references.py`), and figures no source backed are
  re-measured with how they were made.
- `docs/GBDT-ADAPTER.md`: how to train LightGBM and XGBoost from Arrow
  files as fast as they go, five options measured (task 235);
  `scripts/gbdt_adapter.py` and `scripts/gbdt_driver` are the measurement
  tools, not part of the package.
- **`docs/GRAM.md`, wide fits from Grams** (task 237): the offline Gram
  tools with the theory and twelve runnable recipes, from one pass's
  weekly Grams to stability selection, per-week coefficients, compact
  exports and `gram_threads`.
- The memory checks (`tests/test_ffi_memory.py`) fail only when both the
  median gap between blocks and Theil and Sen's slope over every pair pass
  the limit: under load the machine compresses or swaps memory and RSS
  swings both ways, which read as growth to the median gap alone (two flakes,
  6.33 KB/iter where the net was about 1).
- Every mutant that survived the push to `5cb8f7e` is caught by a test or
  listed with its reason (task 234): 342 survivors, 297 caught by tests,
  39 equivalent and 2 tolerated, 4 in `lars.rs` left for a decision; the cut Cholesky factor and the threaded
  Gram merge are held to faer's on every architecture, not only aarch64.
  The diagnostics page gives each model's CUSUM spread and the share of
  breaks it finds.
- The changed-lines mutation job takes a shard for every thirty mutants,
  not forty, and stops each at 160 minutes, not 100: the slowest shard of
  the last push took 83 of its 100, and the suite every mutant runs grew
  from 146 s to 209 s.
- **`docs/DIAGNOSTICS.md` and `docs/DATA-QUALITY.md`** (tasks 222 and 224):
  is the model working, and can the data be learned from -- each ordered
  by question or by `check()` code, with the theory in prose and its
  sources, and runnable recipes (18 and 12) whose printed output a test
  holds. Task 228's result is in the second: a pairwise-complete Gram was
  measured and not built (biased by 0.6-19 standard errors where listwise
  deletion is not).
- **`rls` winds up on a feature held at one value other than 0** (task
  217, documented): its slope wanders to ±3e13 from about 43-54
  half-lives held, after the feature has moved or from its first row, and
  predictions are off by up to 4.4e12 once it moves. `ewridge`, whose
  ridge does not fade, does not; use it for a long stream whose features
  can go quiet. A research round measured two level-free fixes for `rls`;
  neither passed every regime without a cost, so none was built.
- Every mutant that survived the push to `103d721` is caught by a test or
  listed with its reason (tasks 218 and 220): 105 on its changed lines (86
  caught by ten new tests, 19 equivalent) and the 698 its Mutants job left
  untested (71 survived the first pass: 55 caught by 21 new tests, 14
  equivalent, 2 tolerated as rounding). One equivalent entry was wrong, and
  a test now catches it.
- The changed-lines mutation job lists its mutants first and takes a shard
  for every forty (at least one, at most 256), so it tests every mutant a
  push lists, however large (task 219). Each push to `main` runs that pass
  in a concurrency group of its own, so no later push cancels it or
  replaces it while it waits; a pull request's newer commit still cancels
  its older.
- `ftrl` does not centre a feature: its error at a feature level `L` is
  about 0.018·L, so standardize or z-score the feature upstream (task 215;
  `ftrl` takes no `standardize`, since its `l1` zeroes a coefficient in the
  feature's own units). `pa` on a level that moves: a PA-I cap of 1 moves
  the intercept about one unit a row; raise `c` (10 settled the test
  stream from row 467 instead of 2,795) or difference the target (task
  216).
- **Two tiers of tests** (task 210). The essentials -- a fast test of
  every hard rule, the goldens, the contract, the refusals, each module's
  unit tests -- gate the commits of a task in progress, in 74 s of pytest
  and 29 s of `cargo test` where the whole suite took 196 s and 146 s; the
  full suite runs before every push, in CI on all three OSes and before a
  release (`scripts/gate.sh --extended`, `cargo test -- --include-ignored`;
  a plain `pytest` runs everything). Every test module declares its tier,
  every `extended` mark gives its reason, and `tests/test_tiers.py` holds
  the rules. Every model is tested on all-positive, all-negative and mixed
  streams, and every regression on a level crossing zero against its
  oracle (task 209). `kalman`'s `share_p` is documented: the default stays
  one `P` per target, the exact filter per target, and sharing helps
  related targets with noisy noise estimates (task 214; `docs/VALIDATION.md`
  §4: shared -0.0110 against -0.0168 on the one-row target, -0.0880
  against -0.0746 on the five-row one).
- `scripts/validate.py` learns each target only once it is known -- the
  next row's return after one row, the five-row sum after five -- where it
  learned both at their own rows, and compares the models at matched
  memory (half-life 500; `ewridge` solved every row beside its default
  cadence); `docs/VALIDATION.md` is regenerated (task 212). Its §4 now
  reads differently: one shared `P` beats one per target on both targets.
  A pre-registered sweep kept the insensitivity band at 0.01 of the
  target's spread for `pa` and `sgd` (task 207): the residual-scaled
  alternative left a target at a level of 1,000 at R² 0.76. The docs give
  the band's width in noise units (`eps/sqrt(1 - R²)`), the measured
  trade-off at R² 0.978 and 0.99998, and say that PA-I's `c` is in the
  target's units while PA-II's is not; at its documented rates `sgd`'s
  sign-valued step cannot carry a fit to a target at 1,000 within 30,000
  rows. A relative `c` was measured with a clipped spread and a unit-free
  seed for the first rows and passed no pre-registered regime set (task
  213): `c` stays in the target's units.
- Review round 5 (PLAN §19): the frozen state fixtures hold every layout
  the pre-1.0 bumps moved -- `sgd`'s per-loss state, `ewridge`'s kept
  systems, `bocpd`'s warm-up rows, and the integer clock's forms in a bank
  file, a `with_windows` state and a `refresh_time` state -- each held
  non-empty by a test that decodes the file, and the cross-OS hand-off
  carries the same specs (B1, D3). The fixture harnesses run the from-1.0
  loop: each kept schema's set is included from `state_fixtures/v<N>/`,
  loaded through its loader, continued to the bit and held to convert into
  the current schema's fixture bytes, with the coverage check proven on a
  temporary directory while the list is empty (D2); the state promise
  states that third check so, in README, RELEASE-READINESS and CLAUDE.md
  (D1). The API snapshot pins the words a parameter accepts, each probed
  to be taken, and labels unstable names (`sim`, `corr`, `ArrowStruct`,
  `fit_predict_arrow`, `predict_arrow`) (E2, E7). The rule-2 property
  perturbs a target the bank reads (E3). The floor-leg workflow test is
  marked `pins`, since the canary unpins the line it reads (F4b). `pa`'s
  documented eps/c trade-off is stated at the R² it was measured, 0.98,
  with the regime above about R² 0.99 where the wider tube loses less than
  `c=0.1` (F2, E5, A3, G4); the eps review's worst case for `eps=0.1` is a
  range, 0.6× to 280× over 20 seeds with a median of 35×, not 54× (G2); the
  `sgd` docstring and the README no longer claim the same slopes at
  half-life 10 and 10,000, false under `standardize=True` (G1's sentence).
  `bocpd`'s docstring says the warm-up prior is set once from `warm_rows`
  rows and kept, and that a break accepted a row or two late shows as a
  fall in `run_mode` (A5, A6). Stale counts, names and readings corrected
  in the records (F1, F5, F6, F8, D5, D6, E6, B4, A2).
- The 23 survivors and one timeout of the mutation run over tasks 168-182
  are each killed by a test or listed with the reason no test can (task 184).
- Library oracles for `kalman` (filterpy), `sgd` (scikit-learn), `hmm`
  (hmmlearn) and `rls` (padasip), generators that reach the input bound, a
  second golden bank, and a cross-OS hand-off over every kind (task 189).
- The API snapshot pins every resolved default, helper signature, frame
  column, TOML key, CLI flag, environment variable and allowed word (task
  190).
- OUTPUTS.md gives each field's dtype and where else it is null, and
  RUNNER.md the exit statuses (task 191).
- Every dated record under `docs/` points to one table of renamed names,
  PERFORMANCE.md's "Names that changed" (task 192).
- One state of every model kind, a bank file, a `with_windows` state and a
  `refresh_time` state are frozen at the schema shipped, each held to
  loading, going on to the bit and saving its bytes again, and the variant
  names are frozen (task 198). `tests/test_released_state.py` holds a 1.x
  release's files to loading from 1.0.
- The README and RELEASE-READINESS state what is stable, unstable and not
  stable at 1.0, and which release each kind of change needs from 1.0
  (task 199; review round 4, D3).
- Every release, and the canary each month, run the suite on the Polars
  floor (1.34.0 then; 1.37.0 since this release, under *Changed
  (breaking)*); a test that needs a newer Polars skips there by version,
  naming it (task 199, D4). The changed-lines mutation job runs in
  shards with one report (D10), a shard for every forty mutants since
  task 219, a weekly job checks the declared rust-version, 1.95 (D11), and
  SECURITY.md says which release receives fixes (D12).
- Records nothing outside `docs/` cites moved to `docs/records/`, and every
  document has an index row, held by a test (task 199, D13).
- Documented: under an embargo `predict` releases nothing (task 194, S3);
  every unit-bearing default states its unit (task 195, U6);
  `ModelBank.load` checks no checksum (task 198, D14); `bocpd`'s `robust`
  emission is not free of the data's units, so centre and scale its
  features (task 202).
- The development environment is built and tested against polars 2.0.0,
  where it was 1.44.2; the declared range `>=1.34.0,<3` (raised to
  `>=1.37.0,<3` by this release, under *Changed (breaking)*) and the Rust
  crate pin were unchanged by this move (D9).

## [0.13.0] — 2026-09-30

### Added

- **`corrchange(kind="sequential")`: Wied and Galeano's (2013) detector.**
  A cycle is `span_rows` rows of history, taken as stable, then up to
  `monitor_rows` rows (default `span_rows`), each tested against the
  history as it arrives: `|V_k| / w(k/m)`, the monitored rows' correlation
  against the history's in units of its long-run standard deviation, over
  the boundary `w(b) = (1 + b)(b/(1 + b))^boundary_gamma`. A flag, or the
  period's last row, ends the cycle. The critical value is the paper's: at
  `boundary_gamma=0` from the series of `sup|W|`, and above 0 solved as a
  diffusion with an absorbing boundary, within 0.03 of the paper's Table 1.
  On the paper's GARCH design the size is within two standard errors of its
  Table 2 in every cell (`docs/REGIMES.md` §9). `scalar=True` monitors the
  equicorrelation's mean instead of every pair.

- **`marginal(serial_rule="bartlett")`.** Newey and West's weights on the
  kept lags, `1 − l/(L + 1)` with `L` the longest, in the serial
  correction behind `n_serial` and `t_serial`. The long lags, whose
  estimates are noisiest, count less, and the bracket stays positive on a
  pair where `"truncated"`'s goes below zero (two series whose
  autocorrelations have opposite signs, at lag 1: 0.36 where
  `"truncated"` has none).

- **`marginal(window_lags=True)`: lags under a window.** `window` with
  `lags` was refused, since the snapshots held no lag moments. They can
  now, at a price the parameter documents: a snapshot gains `L·T + (L +
  2C)·p·T` doubles beside its `(3p + 5)·T`, twice the size at one lag and
  six times at five with the default cross lags. Without the parameter the
  pair is still refused, and the message gives the spec's own numbers.
  Under the window, `lagcorr`, `n_serial` and `t_serial` describe the rows
  inside it.

- **The Gram's target moments under a `window`.** `bank.gram()` and a
  closed row reported `target_means`, `target_vars` and `target_n_kish` as
  null for a windowed `ewridge` or `lasso`, since the window's snapshots
  held no target moments. They carry them now, at three doubles a target
  a snapshot (under 6 % of it from five features up), and the Gram reports
  the window's.

- **`since_change`, a new `corrchange` output.** On a flag it dates the
  change: the rows from the first changed one through the flag's. It uses
  the paper's Eq. 8 under `"sequential"`, the CUSUM's maximum under
  `"monitor"`, and the second window under `"window"`. It is null
  otherwise.
- **Vowpal Wabbit is a second opinion for `ftrl` in the tests.** Its
  `--ftrl` holds `pred` and `coef` on every row, to its single precision,
  under both losses, with the intercept, row weights and null targets.
  river's comparison checks the logistic state recursion alone. The package
  still depends on polars alone.
- **`holt(trend=False)`, the level alone.** The trend is held at zero and
  the forecast is flat: simple exponential smoothing, whose level is the
  target's exponentially weighted mean. It is held to pandas' `ewm(times=)`
  and statsmodels' `DescrStatsW`. Since an infinite `trend_halflife` became
  the whole history's drift, this is the way to ask for no trend.
  `trend_halflife` is refused beside it.
- **`marginal(feature_moments="shared")`: many targets for less.** Each
  feature keeps one mean and variance over every learned row, and each pair
  only its covariance. Where every target is on every row the pairs are the
  default's, to the bit. At 20,000 pairs it runs 2.7 times as fast at ten
  targets and 3.2 times at thirty, and a ten-target state is under half the
  size. Where a target is absent on some rows it is a different estimator,
  as the docstring says. It takes `lags`, 4.6 times as fast with them at
  ten targets, and no `window`.

### Changed

- **`ftrl`'s fit holds still on rows that teach it nothing.** Under a
  halflife its penalties were constants against sums that decay, so a row
  with no target, or at weight 0, shrank every coefficient toward zero: to
  0.748 of itself over one halflife at `halflife = 100`. The penalties now
  take a per-target scale that ages with the sums on such rows and comes
  back as the target's rows return, so the fit does not move without data,
  as `ewridge`'s does not. The steady state is unchanged, and without a
  halflife the fit is river's and Vowpal Wabbit's as before. Every `ftrl`
  under a halflife reports new numbers, since it no longer shrinks between
  rows either.
- **A chunk past a refusing `window_budget` is refused whole.** The bank
  replays each chunk's clock schedule on its window rings before it learns
  a row, so it refuses such a chunk untouched and goes on as it was. It
  used to find the overrun with the chunk half learned, and then refused
  every later call. Under `drift_action="reset"`, whose resets depend on
  the residuals, the old path remains.
- **`deco` goes on learning past a column with no spread.** A column that
  has been constant from its first row has no standardised value. It is
  now left out of its block's sums, so its block reads the correlation
  among its other columns, and every other value learns as before. Each
  correlation value keeps its own weight, and `loglik` is null on such a
  row. Before, a row with any such column taught no value anything. A
  saved state loads with its one weight on every value.
- **`corrchange`'s long-run variance uses its paper's kernel.** Lag `l` is
  weighted `1 − l/γ`, as Wied, Krämer and Dehling (2012, Appendix A.1)
  write it, where it was Newey–West's `1 − l/(γ+1)`. The `"monitor"`
  statistic and the `scalar` one move with it. Their size moved by at most
  0.001 a cell and their power by 0.009 at the paper's settings
  (`docs/REGIMES.md` §2–3). At `bandwidth=1` only lag 0 is left.

- **`corrchange` refuses a parameter that belongs to another kind.**
  `crit`, `reset`, `seed`, `norm` and the permutation settings under
  `"monitor"`, and `bandwidth` under `"window"`, were accepted and
  ignored. A `crit` given to the monitor changed nothing, and `reset`
  under it did not raise, though its documentation said it would. Each is
  now refused, naming the kinds it applies to.

- **The window budget's refusal says how to raise it.** When a spec sets
  no `window_budget`, the refusal names the default of 256 MiB it ran
  under, and it spells each way out: a larger `{"refuse": MiB}`, or
  `{"refuse": inf}` for no bound, a larger `window_every`, or `{"thin":
  MiB}`.

- **The default solve cadence goes by weight.** `ewridge`, `lasso`, `huber`
  and `quantile` with no `solve_every` solved every `halflife / 50` of
  clock, so a halflife much longer than the stream solved once, at
  `min_periods`, and never again: under `halflife=1e6` the fit after 4,000
  rows was the first rows' slope, 0.99, where the stream said 2.95. They
  now solve once the weight learned since the last solve reaches `ln 2 /
  50` of the weight the fit holds. On evenly spaced rows in steady state
  that is the same `halflife / 50` of clock, one row later at most, and
  from the start of a stream it solves more often, while the fit holds
  little weight. Numbers change for these four models whenever
  `solve_every` is left out under a finite halflife; an explicit
  `solve_every` keeps its clock, and `lam` or `halflife=inf` still solve
  every row. On the validation data every cadence from 9 to 11 rows scores
  within 1.2% of the others, the new default among them
  (`docs/VALIDATION.md` §1).

- **`pa`, `sgd`, `ftrl` and `rls` check a target's `min_periods` against
  that target's own weight.** They checked the weight of every row, so rows
  with a null target counted toward a fit they never moved. After ten such
  rows, `min_periods=10` was met with every coefficient at zero: `ftrl`
  predicted 0.5, and `pa` and `sgd` predicted 0. Each target now waits for
  the weight of the rows that carried it, as the other regression models
  already did. For `rls`, which learns a row only when every target is
  present, that is the weight of the rows it learned from. The `n_eff`
  field does not change: it is still the weight of every row. Outputs
  change only on rows where a target's own weight is below `min_periods`,
  which now report null.

- **State schema 20.** The four solving models keep the weight learned
  since their last solve, `pa`, `sgd`, `ftrl` and `rls` each target's own
  weight, and `corrchange` its sequential monitoring period. A schema-19 file loads: the first count starts at 0, and each
  target's weight at the shared one its gate read. A 0.12.0 build refuses a
  schema-20 file by its version.

### Performance

- **`ewridge` takes its readiness shares only when they are read.** Since
  0.9.0 every solve formed the diagonal of `A⁻¹`, `O(k³)`, for each
  coefficient's data share and the noise gate's `edf`. A solve now keeps its
  factor, and the shares are computed when a `coef` row, `summary()` or a
  save reads them, or at the end of the chunk. The gate reads them only
  where `edf`'s bound, one per coefficient, cannot decide. Every output and
  every saved state is the same to the bit. `ewridge` learns 16% more rows a
  second at k=20, 41% at k=50, and 42% where every row solves; with
  `coef_every=1`, which reads the shares on every row, it is 1 to 3% slower.

- **`ewridge`'s solve copies and divides less.** Without `standardize` it
  reads its system straight from the feature sums, where it copied them
  and divided every entry by a scale of 1; with it, it reads only their
  diagonal for the scales. The Cholesky factor solves in place. Every output
  is the same to the bit. Where every row solves, `ewridge` is 8% faster at
  k=5, 14% at k=20 and 19% at k=50; at the default cadence, 3 to 10%.

- **`ftrl` takes a target's penalties once a row, not once a coefficient.**
  Its penalties now take the target's scale (Changed, above), and computed
  per coefficient that cost `ftrl` 18% of its rows a second. Taken once per
  target, the same operations in the same order, the outputs are the same to
  the bit. Against 0.12.0 `ftrl` is still 11% slower under a halflife and 6%
  without one: the scale's division and the per-target weights, once a row.

- **The CLI writes NDJSON from one thread.** It serialized a slice per
  thread, which on macOS's system allocator once ran from 4.8 s to 54 s on
  3M rows. Measured again, one thread is the faster: 4.7 s against 5.0 s on
  the example bank over 3M rows, with half the system time, and within
  0.05 s of the same run to parquet.

### Fixed

- **`ModelBank.fit` warns about row order for `huber` and `lasso`.** A fit
  whose query does not fix its row order raises `OrderNotGuaranteedWarning`,
  except where the state it leaves cannot depend on the order. `huber` and
  `lasso` at `lam=1.0` were exempt, and neither qualifies. `huber` weighs
  each row by the fit before it, so once it down-weights a row its sums
  depend on the order: with one row in ten lifted by 5, a shuffle moved its
  coefficients by 1.05e-02. The exemption had been measured on rows it never
  down-weighted. `lasso`'s path points commute, but it selects its penalty
  by out-of-sample error, and shuffles moved the penalty it selected from
  0.01 to 0.1 and to 0.001. Both now warn, as the other models already did;
  `ewridge` and `rls` stay exempt.

- **`to_json` exports a bank with a column that never held a value.** The
  data summary starts each column's `min` and `max` at infinity, and a
  column no row gave a value keeps them: a target null on every row, or
  every row skipped for its weight. The export refused such a bank as a
  dropped value. It now writes them as `"inf"` and `"-inf"`, as it writes
  every other non-finite float; the binary state is unchanged.

- **A `scalar` `corrchange` monitor saved in the middle of a span loads.**
  Its ring holds one value a row, the equicorrelation `u`, and the load
  held those rows to the feature count and refused the state as the wrong
  shape.

- **`po.stream.refresh_time` under `.head(n)` saves the state behind the
  rows returned.** With `save_state`, it read the whole input and saved
  that, so the points after the slice were computed and thrown away, and a
  run resumed on the rest of the input refused every row. It now stops at
  the tick that completed the n-th point and saves the state after that
  tick, the same whatever the chunk size, as a bank saves the state after
  the rows a `head(n)` pulled: a run resumed on the input after that tick
  goes on with point n + 1. With `pairs=True`, a tick that completes several
  pairs' points at once is taken whole.

- **`deco` with `dynamics="linear"` holds `rho` still on a zero-weight
  row.** The linear recursion took the row's `u` at full strength in its
  `alpha * u` term whatever the row's weight, so a row meant to advance the
  clock and learn nothing -- an `embargo` predict copy, a row masked by a
  weight column -- moved the level: one such row moved it from 0.0525 to
  0.0392 in the test. Numbers change only for `"linear"` streams with
  zero-weight rows. A positive weight still reaches `rho` only through
  `rho_bar`, as the paper's recursion has no row weights; the docstring
  says so.

## [0.12.0] — 2026-09-28

### Changed (breaking)

- **A clock that steps back is refused by default.** `on_clock_reset`
  keeps `"error"`, now the default, and `"reset_state"`. `"max"` and
  `"zero"` are gone. `"max"`, the old default, took the cap as the step, so
  under an infinite cap a step back handed the models an infinite step, and
  `holt` predicted null from there on. A spec that names either is refused.

- **What a late row is, the caller says.** `min_backwards_jump` no longer
  defaults to `max_dclock`. It is required with `"reset_state"`, where a step
  back no larger than it is refused as a late row and a larger one starts
  the model over, and refused with `"error"`, which refuses every step back.
  A step back equal to the minimum is now late: a `Date` clock's one-day
  step passed a one-day minimum before.

- **`max_dclock` is finite and above 0, and `session_gap` is finite or
  `"reset"`.** A cap of 0 froze the clock, and every gap then read as a
  break, so `label_delay` released each held label on the next row: for no
  decay, set `halflife="inf"`. An infinite cap took the break away; an
  infinite session gap is `"reset"`.

- **`predict` scores a row before the last learned clock against the state
  as it stands**, as a step of 0, under either policy. `"error"` refused
  such a row, `"reset_state"` scored it as a fresh stream, and `"max"`
  scored it a whole cap on.

- **`po.prep` is `po.stream`, and `refresh_time` speaks the specs'
  vocabulary.** The module is renamed with no alias, and
  `refresh_time`'s `time=` and `by=` are `clock=` and `group=`.
  `po.stream.embargo` and `po.stream.refresh_time` give back the kind of
  frame they are given: a `DataFrame` for a `DataFrame`, where both
  returned a `LazyFrame` before. Chain them with
  `lf.pipe(po.stream.embargo, ...)`.

- **A bank file from before this release is refused, by its version.**
  Every one names an `on_clock_reset` that no longer exists (`"max"` was
  written whether or not a spec had a clock). The bank now loads schema 19
  only; refit from the input. A model's own state from schema 14 on still
  loads.

### Added

- **`po.stream.refresh_time` resumes.** `save_state=` writes the sampler's
  state once the input's last row is fed, every group's grid part-way
  through an interval included, and `load_state=` goes on from it, so a
  stream fed in two runs gives the grid one run gives. A temporal state
  resumes on any unit.

- **`ModelBank.skip_learned(frame)`, for resuming on input that overlaps a
  saved state.** It keeps each row after its group's last clock, in every
  spec with a clock, and every row of a group the bank has not seen, so a
  rerun learns each row once. It takes a `LazyFrame` too, and compares a
  temporal clock in exact nanoseconds. docs/STATE-WORKFLOW.md has the
  recipe.

### Fixed

- **A zero-weight row forgets what its decay forgets.** Where the decay
  factor underflows to 0, from 1075 halflives on, the update is 0/0, and
  the guard used to keep the history whole, so the next row saw the old
  weight. The row is now the decay alone, as one halflife short of it all
  but is. Five models kept the history (`ewridge`, `lasso`, `kalman`,
  `ew_cov`, `deco`) and `sgd` a scale; a contract test now holds every
  model to it.

- **An error names the row of the input, not of the chunk.** Fed in
  chunks, by `fit`, `fit_predict_batches`, a plan or the CLI, a refused row
  was counted from the start of its chunk. `refresh_time` counted the same
  way.

- **`refresh_time` compares a temporal clock exactly.** It read the clock
  as a double, which on a `Datetime` in nanoseconds resolves 256 ns, so a
  step back smaller than that passed as a tie. It reads the column's
  integer now, in nanoseconds, so an instant nanoseconds cannot hold
  (before 1677 or after 2262) is refused by row, as a bank refuses it; the
  output column is unchanged.

- **The plan check warns about a sort by several keys without
  `maintain_order=True`.** It took any sort as settling the order, but such
  a sort leaves rows with equal keys in no particular order: 2,109 of
  10,000 rows moved when measured. A sort by one key was stable, and still
  passes.

- **`po.prep.embargo` says it needs the clock order across all rows.** It
  merges by the clock alone, so a frame sorted only within its groups came
  back with a group out of order. The docs said "as a stream must be",
  which is each group's order; they name the remedy now.

- **`coverage_<slot>` under `label_delay` counts the intervals the frame
  showed.** A held row was scored, when its label arrived, against the
  conformal radius reached by then rather than the radius its interval was
  shown with, so the coverage described intervals nobody was shown (the
  residuals had the same fault, fixed in 0.8.0 as C21). The row is now
  scored against the radius it was shown, and the radius steps on that
  error -- adaptive conformal inference with delayed feedback -- so `lo_`
  and `hi_` move too under a delay. A row shown no interval, because the
  radius had not started yet, counts for neither. Without a delay nothing
  changes. Found by a test of the coverage against the frame's own
  columns.

- **`po.eval.seqtest` with `by` works on Polars 1.34.0, the declared
  floor.** Its log wealth was a running sum over the groups of a bet sized
  by running sums over the groups, a window inside a window, which Polars
  1.34.0 refuses and 1.44.2 accepts. It is computed in two passes now, with the same
  numbers. Found by running the suite at the floor.

### Changed

- **The Linux CLI binaries run on glibc 2.17 and later.** 0.11.1's were
  built on the release runner and needed its glibc, 2.39, so they did not
  start on Ubuntu 22.04, Debian 12 or RHEL 9. They are built in the
  manylinux2014 image the wheels come from now, and the release refuses a
  binary that needs more than 2.17.

- **State schema 19.** Under `label_delay` with a conformal interval, each
  held row keeps the radius it was shown (schema 18), and the clock settings
  changed (19, above).

### Performance

- **`ew_cov(lags=...)` and a multi-target `ewridge` run at 0.10.0's speed
  again.** 0.11.0's compensated means put a two-part deviation inside the
  lagged co-moments' innermost loop, `k * k` times a lag where `k` would
  do, and the own means that 0.11.0's review gave each target checked their
  low parts' length once a feature. Each is now taken once. At k = 20,
  `ew_cov` with lags 1 to 5 runs 1.14M rows a second, against 0.58M in
  0.11.1 and 1.06M in 0.10.0, and `ewridge` with 10 targets 1.49M, against
  1.02M and 1.59M. Outputs are unchanged to the bit.

### Documentation

- **`marginal`'s lead/follow reading is limited to series of the same
  moment.** The docs read a feature whose `lagcorr_xy[0]` exceeds its
  `corr` as one that follows its target, a late-sampled column. Against a
  forward-looking target, one built from the rows after its own, every
  timely feature built from the same news shows that, since the target one
  row back starts with the return the feature already holds. The README,
  both docstrings and MARGINAL-LAGS-AND-BINS say so now, and a test holds
  the terms to their closed form on such a target. The terms themselves
  are unchanged. Reported by a caller whose late-sampling screen flagged
  527 correctly sampled pairs on the old reading (E75).

## [0.11.1] — 2026-09-27

0.11.0 was tagged (`v0.11.0`) and never published: its release run failed
in CI, and a release tag cannot be moved. 0.11.1 is everything below. From
this release on, the release workflow creates the tag itself, after every
job has passed and the upload is done (docs/RELEASE-READINESS.md, "Cutting a
release").

### Added

- **Python 3.14 is declared, and CI tests every supported version on
  purpose.** The package metadata lists 3.12, 3.13 and 3.14; one `abi3`
  wheel per platform already installed on all three. CI runs the suite on
  each version on Linux, and on the oldest and newest on macOS and Windows,
  choosing the interpreter where it used to take whatever the runner had.
- **The README says how a bank detects rows out of order**, in "Row order
  and the two guarantees": the query check and what it can miss, the clock
  checked group by group, what a backwards step of each size means, why a
  refused chunk leaves the bank as it was, what scoring and the summary do,
  and the group-key check under `group_close="monotone"`.
- **`scripts/compare_release.py` compares every output with a release's,
  bit for bit**: the new first step of a release
  (docs/RELEASE-READINESS.md), and a report on every CI push.
- **pyarrow is tested as a reader of the Arrow output, and as a source.**
  On pyarrow 25.0.1, `pa.array`, `pa.chunked_array`, `pa.record_batch` and
  `pa.table` each take a struct from `fit_predict_arrow`, with
  `fit_predict`'s values. A `RecordBatchReader` streams into a bank with
  the whole frame's numbers. pyarrow is a test-only dependency: the
  package still depends on polars alone.

- **`kmeans` and `micro` take `scale_floor`**, default 0.1: the metric's
  variance is floored at that fraction of the feature's long-run variance.
  `1 / var` alone grew as `2^Q` over `Q` halflives of a flag that stops
  firing or a sensor at rest, a million at twenty, and the row on which the
  feature moved again was infinitely far from every centre: `kmeans`'s
  split–merge move ended at ARI 0.22 after such a spell and now recovers to
  0.73; floored, the weight grows as `2^(Q/8) / scale_floor`, 58 at
  twenty. `0` is the metric as it was, and what a state saved before the
  floor loads with. On a static, drifting, rescaled or growing-noise stream
  the floor binds on no row. The long-run variance is tracked at eight times
  the halflife, a feature at a time, each row's weight and deviation clipped
  against it, its start the medians of the feature's first five rows, and
  started over by a first move from no spread, so a row at the input bound
  moves it by a factor of 26 at most, which a few of its halflives undo.
- **`marginal` takes `cross_lags`**: the lags at which `lagcorr_xy` and
  `lagcorr_yx` are kept, each one of `lags`, `[]` for none. The default
  keeps them at every lag, as before. `n_serial` reads the
  autocorrelations alone, which every lag keeps, and they are the same to
  the bit. At `lags=[1, 2, 5, 10, 20, 50]` and nine targets, one cross
  lag takes 30% off the lags' cost and 22% off the row
  (docs/PERFORMANCE.md §23).
- **A target can be taken against another column of its own row.**
  `po.target("price_5m", relative_to="mid")` in a spec's `targets` has the
  model learn and predict `price_5m - mid`; `relative="ratio"` and
  `"log_ratio"` take `y / r` and `ln(y / r)`. The reference is read at the
  target's own row, so nothing looks ahead, and a value either side cannot
  use makes the target null on that row. In the CLI's TOML the target is a
  table. A spec with plain targets writes the bytes it always did.
- **`marginal(bin_budget=...)` sets the bins' memory limit.** The warm-up
  hold and the histogram are each refused past 256 MiB per group when the
  bank is built; `bin_budget` moves that limit, in MiB, and
  `float("inf")` removes it. The CLI's TOML takes it as `bin_budget = 512`
  or `"inf"`.
- **A wide `marginal` can use every thread: `marginal(shards=...)`.** The
  bank runs groups and specs in parallel, so one wide spec on one group
  was one thread's work. `shards` splits its pairs into ranges of features,
  each run on a thread of its own a batch of rows at a time; `"auto"` sizes
  the split to the width and the pool. Every number is the same to the bit
  whatever the count, and a saved bank resumes under any count. At 10,000
  features through the bank: 1.2 times as fast at one target, 2.0 at
  nine, 4.9 with lags and bins on 14 threads (docs/PERFORMANCE.md §25).
  Off by default: with more groups than threads the pool is already full.
- **scikit-learn is a second opinion in the tests, and the Rust models are
  property-tested.** `huber` is held to `LinearRegression` and
  `HuberRegressor`, `marginal`'s split to a decision stump, and `sgd` to
  `SGDRegressor`, live. proptest generates streams for all 21 models and
  holds each to the model contract. Every test library, Python or Rust,
  must name an open-source licence, which a test checks. The package still
  depends on polars alone.

### Fixed

- **A `po.target` table on the LazyFrame paths.** `lf.online.fit_predict`
  and `lf.online.predict` raised `unhashable type: 'dict'` on any spec with
  a table target, relative or merely renamed, while the plan was built; the
  projection now keeps the target's column and its reference (a fix by name
  alone would have projected the reference away and refused the run).
- **A row at the input bound against a vanishing spread.** A row of weight
  `1e-100` followed by one of `1e100` leaves `kmeans` and `micro` a
  standardized spread of `4e-200`; the next row at the bound is `5e199`
  scaled units away, whose square overflows. `micro` reported the distance
  as infinite and, reading the overflowed square as "no change to the
  radius", *absorbed* the row; `kmeans` reported infinity and never set a
  runner-up. Both report the distance now (computed without squaring where
  the square overflows; ordinary rows keep their bits), an overflowed
  square is infinitely far, and ties among such squares go to the nearest.
  Found by the model contract's property test once its values reached the
  bound, which they now do.
- **A quantile `robust` fit no longer overshoots a row far outside its
  band.** A row outside the band is one term of the score and none of the
  Hessian, by design, so its nudge moved the fit at the row by the row's
  leverage -- unbounded when the Gram holds no curvature that far out. A
  target of `1e100` and features of `1e100` at weights from `1e-100`
  (inside the input bound) took a slope to `1e248` and the prediction at
  `1e100` to `-inf`. The nudge is now bounded so the fit at the row moves
  by at most the row's residual: the row reaches its band's edge. The bound
  binds on early rows too, where a row's leverage exceeds its share of the
  weight: the quantile golden's three values moved by about 1e-3 of
  themselves, and the QuantReg comparisons hold. Found by the same
  property test.
- **`ew_ridge` and `lasso` keep each target's own feature mean as a pair
  of its own.** The cross accumulator kept it as an offset from the
  all-row mean and reconstructed it as their sum: a target absent on a row
  whose features stood at the input bound left both at `1e99`, the sum
  resolved nothing below `1e83`, and the next present row's deviation of
  `0.4` read as `1e83` -- the prediction at `1e100` was `-inf`, windowed
  or not. State schema 17; a file from 14, 15 or 16 loads, its offsets
  turned into means once. Under `own_rows` the own mean is the Gram's to
  the bit, so a stream whose targets are present on every row is
  unchanged; a target with gaps moves in its last bits. Found by the same
  property test.
- **`deco`'s `loglik` is NaN where the density is past the double's
  range**, as `hmm`'s is, where it reported `-inf`: a row far past the
  standardiser's spread, under a `ρ` the density clamps to the edge of
  singular, overflowed the quadratic form. Found by the same property test.
- **A window no longer fits the rounding of rows it cannot resolve.** Rows
  at `1e100` before a window and a row of weight `1e100` inside it leave
  the live co-moments near `1e100`, with the window's own spread below
  their last digit; the subtraction that removes the older rows returned
  noise near `1e84`, and a windowed `lasso` fitted it to a prediction of
  `-inf`. A windowed variance no larger than the rounding of the terms it
  is formed from is read as no spread, as a held feature's is, so the
  feature gets no slope. Every window oracle and the held-feature tests are
  unchanged. Found by the same property test.
- **`marginal`'s bins from a 0.10.0 state.** A histogram loaded from a
  state written before the means' low parts panicked on its first row whose
  features were all null with a target present (the Rust API; the bank
  skips such a row). The low parts are sized on the first row that could
  write a cell, on both the plain and the sharded path.
- **Relative targets, four refusals and a message.** `sgd` under
  `loss="logistic"` or `"poisson"` took a relative target, as `ftrl` does
  not; a table target merely *named* like a feature was refused as a leak
  (the column is the leak, not the name); a `bocpd`/`hmm` slot compared
  names, so a table named like the hazard put another column in the slot
  (the slot must be the column itself, and its message prints the names
  again); a temporal clock read as a table target's column or reference
  was refused later, with another message; and an empty `name`, `column`
  or `relative_to` was refused nowhere.
- **`hit_rate` under a ratio target.** A ratio is positive by construction,
  so sign agreement about zero read 1.0 whatever the fit; the hit test is
  about 1 -- did the ratio go up or down. A difference and a log ratio are
  about zero, as before.
- **The runs follow the window on a restore.** A state written before the
  runs' flag (0.10.0) resumed without a window kept tracking runs for the
  rest of its life; a windowed spec given a state whose runs are off would
  have read every held feature as moving, with no error. Every model's
  restore now sets the runs off without a window and refuses the other;
  `ewridge`'s also refuses a windowed spec whose state has no ring, as its
  siblings did; and `hmm`, a windowless owner, keeps no runs.
- **Corrupt states refused before the first row**: a bins histogram whose
  offsets disagree with its edges or whose low parts are the wrong length;
  a lag ring longer than the deepest lag, which read every lag a row too
  recent, or with fewer target rows than feature rows, which panicked.
- **`marginal(shards=...)`, three things it did not say.** With a window at
  the default `window_every` a sharded model flushed -- one fork-join --
  per row, the regime measured as slower than unsplit, and `"auto"` did not
  know: it sizes a windowed model's flush by the snapshot cadence now, and
  a snapshot every row is never split. The bins' warm-up hold was reserved
  whole on the first held row, 64 KB a group at the default and 640 MB
  across ten thousand short groups: it grows by doubling, capped at the
  hold, so the budget still counts the most it can reach. `ModelBank.load`
  compared `shards` with the saved specs and refused a bank resumed under
  another count, which the docstring promised: the count is the caller's
  now, the saved one when `load` is given no specs. And a Rust caller
  reading `state()` or `pair()` with rows held is refused in every build,
  not only a debug one.
- **The Python builders check a table target's shape and a list's
  floor.** `targets=[3]`, `[None]`, `[{"col": "p"}]` and a table entry of
  the wrong type are named by parameter, where they reached Rust and were
  named by JSON path; `lags=[-1]` and `cross_lags=[-1]` likewise, where
  serde named `model`; and `bin_edges={"x": [inf]}` is refused by name.
- **`max_error_inflation` is refused by name on a model without a ridge
  system** (every model but `ewridge`), where it was range-checked and
  dropped.
- **A feature that stops moving leaves no rounding artefact in the
  clusters' metric.** The feature moments' means are pairs, as every other
  running mean now is, and `kmeans` and `micro` give a stream at a level of
  1e8 the labels they give it at 0.
- **`corrchange`'s constancy test works at any level of the columns.** The
  long-run standard deviation its statistic divides by was formed from raw
  moments, so `E[x²] − E[x]²` lost the columns' level digits: off by 1.2e-5
  of itself at a level of 1e5 and NaN at 1e8, where the test then flagged
  nothing. It is now computed on the span centred at its means and scaled
  by its standard deviations, which is the same number in exact arithmetic
  (the paper's own `ξ_t` form) and is the same double at levels up to 1e12.
  Two verdicts change with it. A pair within rounding of `|ρ| = 1` -- a
  derived or duplicated column -- has no long-run standard deviation and no
  verdict, where the old form's rounding stood in for one; and a span with
  one far outlier gets the verdict the formula gives, a flag, where the old
  form cancelled to nothing (docs/REGIMES.md §4 on the delta method's heavy
  tails).
- **A feature or target that stops moving keeps the fit it had.** A
  running mean given one value row after row stopped a few rounding steps
  short of it, once its step rounded to nothing. Every variance and
  covariance centred on that mean then settled on the gap instead of
  decaying, and a model that divides one by the other read two rounding
  artefacts. A lasso's slope on a feature held at a level of 1e8 went from
  0.5 to -4.7e3 within 40 halflives. `sgd` and `kalman` predicted
  differently at each level the feature sat at, by up to 0.12. `marginal`
  gave a held target a correlation of -0.06 with a moving feature, where it
  goes to zero, and its bins' split gain read 0.45. Every running mean is
  now carried as two doubles, so no part of a step is rounded off, and the
  fits stay within a rounding step of the level of the fit at 0.5: they keep
  their slopes for 150 halflives at levels from 0 to 1e12, and with no decay. This covers `ew_cov`,
  `ew_ridge`, `lasso`, `robust`, `kalman`, `sgd`, `marginal` (pairs, lags
  and bins), `deco`, `bocpd`, `ew_class`, `hmm` and `emit_autocorr`.
- **A windowed fit no longer standardizes a feature by rounding.** A
  feature that holds one value over every row inside a `window` has no
  spread there, and the fit now says so exactly: its variance, covariances
  and covariance with the target are zero, so a lasso or a standardized
  ridge drops it and a plain ridge gives it a slope of zero. The window's
  subtraction left a remainder instead, which grows with the feature's
  level and with the rows since the window's edge, and which a lasso at a
  penalty of zero divided by itself: predictions of 1e55 where the fit was
  -1.0. A feature counts as held when every row inside the window carried
  its value, under any window and halflife. `ew_cov`'s and `ew_class`'s own
  windows read it the same way: a held feature has a variance of 0 and a
  null correlation there, where it read noise.
- **A windowed `marginal` pair reports a feature or target held over the
  window as having no spread.** Its variance and covariance are zero, its
  mean is the value, its correlation is null, and a held target has a
  slope of zero. The window's subtraction left a remainder, which `beta`
  divided by itself.
- **A lasso's `lam_selected` under a `window` is chosen on the errors
  inside it.** Four things had it read others: the window's snapshot took
  the selection after the row's own error, with its weight aged twice; the
  choice read the window as it stood a row earlier; a row that did not
  score the target left the choice from an older window standing; and the
  window aged by the model's halflife where `select_halflife` differs. A
  window with no scored row of the target leaves the last choice standing.
- **The PyPI page's links to other files work.** PyPI shows the README,
  where a relative link such as `docs/PLAN.md` resolved against pypi.org
  and was a 404; only in-page links worked. The release workflow now
  rewrites the README's 76 relative links to GitHub at the release's tag
  before it builds the packages (`scripts/pypi_readme.py`), and the README
  in the repository keeps its relative links. PyPI never changes a
  published description, so 0.10.0's page stays as it was; the fix shows
  from the next release.
- **`label_delay` keeps chunk invariance (hard rule 3).** The clock a
  stream's held rows cover was a running sum within a chunk and a fresh sum
  at each chunk's start, which round differently, so `settled_frac` could
  differ in its last bit between one chunk and several, and a
  `min_settled_frac` at that value could withhold a prediction in one
  chunking and not the other. The stream now keeps that clock per model in
  its state. A single-chunk run's numbers do not change; a run fed in
  chunks now gives them too.
- **A duration past 292 years is refused, naming the parameter.** A
  `pl.duration` that long wrapped silently, so 585 years was kept as
  `384ns`, and a `timedelta` that long raised an `OverflowError` that
  named nothing. Both now read `spec "m": halflife is longer than 292
  years, the most a clock can hold`, as the text form always did.
- **A space after a duration's sign is refused.** `"+ 5m"` was accepted,
  and named a grid's field `@h+ 5m`; polars refuses it too.
- **`lasso`'s docstring says what its coordinate descent reads.** `c_i` is
  each feature's covariance with the target over the feature's standard
  deviation, not a correlation; an elastic net's predictions do not scale
  with the target, since its ridge part is added to a correlation; and the
  selection's errors count the rows the target's `min_periods` still
  withholds from the output.
- **`sgd`'s docstring gave the intercept an `l2` it does not get.** The
  update equations and the `l2` entry now say the ridge is on the slopes
  only, as the code has always done.
- **`marginal`'s docs no longer promise the bins' warm-up replay to the
  bit in every case.** A row of weight zero inside the warm-up is held as
  its decay alone, folded into the next held row, and a product of decays
  rounds differently from the decays one at a time: the histogram then
  agrees with one built from the edges given up front to about 1e-15 of
  the data's scale, as the design doc has always said.
- **The Arrow output's docs no longer say duckdb reads it directly.**
  `fit_predict_arrow`'s docstring and the type stub named duckdb among
  its direct readers. duckdb 1.5.5 refuses the struct and takes it
  through `pl.Series(s)`, as the README already said.

### Changed

- **The ridge and lasso window snapshots are `k + T` doubles smaller.** A
  snapshot cloned the means' low parts, which the window's subtraction
  never reads; a ring near its budget thins or refuses a little later than
  it did in 0.10.0 + task 130.
- **Finding a spec's columns no longer takes time quadratic in their
  number.** Seven lookups by column name scanned every column. At 10,000
  features they took about 480 ms of an 800 ms call, and now take none
  that shows (docs/PERFORMANCE.md §24).
- **`marginal`'s bins take a third more memory, and the 256 MiB check
  counts what they take.** Each bin carries its mean as a pair now, four
  values where 0.10.0 kept three (see "A feature or target that stops
  moving keeps the fit it had", above), and the check had gone on counting
  three. It counts from the types now: every value a bin keeps, and a held
  warm-up target at its real sixteen bytes rather than eight. So a spec
  0.10.0 built can now be refused: one whose histogram needs more than
  256 MiB at four values a bin, or whose warm-up hold does once its targets
  are counted in full. Each limit is per group, and per halflife of a grid,
  as it always was; the docs now say so, and that the last warm-up row
  holds both at once.
- **A window's snapshots are counted in full against `window_budget`.**
  `ewridge`'s, `lasso`'s and `marginal`'s hold a little more than 0.10.0's,
  the means' low parts and each target's count of learned rows, and the
  budget now counts it, so a ring near its limit thins or refuses a little
  sooner than it would have.
- **`marginal`'s row takes 7–50% less time, every number the same to the
  bit.** Its pair, lag and bin updates are each one loop over slices now,
  shared with the sharded path. With six lags at nine targets a row takes
  48% less time, and with the moments alone 38% less (docs/PERFORMANCE.md
  §25).
- **`marginal`'s bins cost 2.6 times less at nine targets.** Each
  feature's bin is found once per row rather than once per target, and
  every number is the same to the bit. One target's bins cost a fifth
  less too (docs/PERFORMANCE.md §22).
- **Outputs differ from 0.10.0 in their last bits** wherever a running
  mean is taken, since the means are carried as pairs now (see "A feature
  or target that stops moving keeps the fit it had", above). On the release
  comparison that is 21 of its 30 specs and 130 of its 291 fields, by a
  median of 3e-16 of the value and at most 1.4e-14 (`bocpd`'s `p_change`;
  measured again on 2026-09-26, after tasks 102 and 103). The diagonal accumulator behind `sgd`, `kalman`,
  `deco` and `corrchange` takes 6.0, 10.7 and 24 ns a row at 4, 16 and 64
  slots where it took 5.2, 6.3 and 15.5, so `sgd` steps about 16% slower at
  16 features; `ew_ridge` and `ew_cov` are unchanged.
- **State schema 17.** The cross accumulator's own means (17, above). A
  stream with a `label_delay` keeps, per model, the
  clock its held rows cover (15). A model with a window keeps, per slot of
  its accumulators, the value the slot has held since it last changed and
  the learned row that started that run, for the window to read; a model
  without one keeps a flag saying it keeps none; every
  running mean keeps what its double leaves out; and `kmeans` and `micro`
  keep each feature's long-run reference for the metric's floor; and
  `marginal` keeps its `cross_lags` (16). States saved by 0.10.0
  (schema 14) still load. The clock is rebuilt as 0.10.0 did at a chunk
  boundary, the runs start at the next row, and a mean starts as the
  double it was saved as, and a `marginal` keeps its cross terms at every
  lag.
- **`holt` reports null coefficients for a target not yet observed.** It
  reported `[0, 0]`, a level no row had given, where every other model's
  `coef` is null before it has anything to report.

## [0.10.0] — 2026-09-24

### Added

- **A temporal clock, with its parameters as durations (`docs/PLAN.md` task
  88).** A `Datetime`, `Date` or `Duration` column can now be the clock, and
  every parameter measured in clock units -- `halflife`, `max_dclock`,
  `min_backwards_jump`, `session_gap`, `label_delay`, and a model's
  `window`, `solve_every` and own halflives -- is then a duration, written
  as `pl.duration(minutes=10)`, a `timedelta`, or polars' duration text
  `"10m"`. The spec keeps the text, which the command line's TOML takes
  too. The clock is read in its own integer nanoseconds, and the gap
  between rows is taken in integers before it becomes seconds, so the same
  instants stored in milliseconds, microseconds or nanoseconds give the
  same numbers to the bit, a nanosecond timestamp keeps its nanoseconds
  whatever the stream's age, and a zone-aware column is read as its UTC
  instants. A clock must lie between the years 1677 and 2262.
  A numeric clock is unchanged. `po.prep.embargo`'s `delay` and
  `po.eval.rolling_metrics`' `window` take durations the same way.

### Changed

- **A clock parameter of the wrong kind is refused, naming the column, the
  parameter and the fix.** A plain number on a temporal clock, which would
  silently take the column's storage unit; a duration on a numeric clock; a
  spec that mixes the two; a rate per clock unit (`lam`, `kalman`'s `q`) on
  a temporal clock; a `Time` column as the clock; and a duration finer than
  the clock can act on, such as `max_dclock="12h"` on a `Date` clock.
  Before, every temporal clock was refused.
- **A string given for a clock parameter is read as a duration**, so a bad
  one is a `ValueError` naming what is wrong with it (`halflife "10" is not
  a duration: 10 has no unit`) where it was a `TypeError` about its type.
- **A state file whose specs carry a duration is envelope version 3.** Every
  other file is still version 2, byte for byte, so a build from before
  durations reads it, and refuses a file with a duration by its version.
- **State schema 14.** The clock state keeps its previous row's value in the
  form the source had it, a number or a temporal clock's nanoseconds, so the
  gap between two instants is taken in integers. States saved by 0.9.x do
  not load (`MIN_SCHEMA_VERSION` is 14, the pre-1.0 policy).

### Fixed

- **The guides are rewritten to `docs/WRITING.md`, and the false claims the
  rewrite found are corrected against the code (`docs/PLAN.md` task 90).**
  Twelve documents: the eight guides under `docs/`, the documents index,
  `CONTRIBUTING.md`, `SECURITY.md` and `llms.txt`. Each keeps every number,
  name, link, ID and section number another file cites. The corrections
  that change what a reader would do:
  - `docs/REGIMES.md`'s `corrchange` tables predated the S3 fix of
    2026-09-19, and are measured again: a shared-scale t5's size of .075 is
    now .045, and both readings of the paper's distribution land within
    0.011 of its table. Its `hmm` recovery ran on streams that never
    switched regime, and now says so.
  - `docs/RUNNER.md` credited the command line with 0.95 and 1.41 GB of
    memory, which were the removed `po.run`'s.
  - `docs/STATE-WORKFLOW.md` said `load_state` accepts a `ModelBank`, which
    raises a `TypeError`.
  - `docs/OUTPUTS.md` named a halflife grid's suffix `__hl` where it is
    `@h`.
  - `docs/PERFORMANCE.md`'s headline read 2.8× where its own table gives
    2.4×, and it showed `hmm` with `covariance="diag"`, which is refused.
  - The README's chunk-size figure is now the one `docs/PERFORMANCE.md` §20
    measured, and it says `coef` lands on each group's last row in a chunk.
- **`kalman` is held to river's `BayesianLinearRegression` again.** The test
  was deleted in `509c6cf` with nothing in its message, while
  `docs/TESTING.md`'s T-R2 went on citing it. It passes unchanged, and is
  restored.
- **`scripts/regime_experiments.py all` runs to the end again.** Its `epps`
  step had stopped the run since a rename on 2026-09-07, and its printed
  conclusions now match the measured document.
- **The command line's `--no-output` help** says a run needs `save_state`
  or `closed_groups`, which is what the runner checks.

## [0.9.1] — 2026-09-21

### Fixed

- **The `support_coef` warning no longer fires on a fit the model is itself
  withholding.** It is raised only on a row whose prediction the gates let
  through, so the first solve of a spec — one row against `k` slopes,
  under-determined by construction, and a warning that cannot be retracted —
  no longer produces a message the next row contradicts. Found on the
  shipped 0.9.0 wheel: an ordinary two-feature fit warned at `n_eff = 1.00`
  ("0.27 data and 0.73 ridge") and read `support_coef = 1.00` from row 2 to
  the end of the stream. A genuinely undetermined design still warns, once,
  as before — including a spec frozen on a one-row solve, where the warning
  is correct.
- **The two clock assertions in `tests/test_frame.py` accept either
  exception.** py-polars 1.x wraps an exception raised inside a Python IO
  source as its own `ComputeError`; 2.0.0rc2 — published hours after 0.9.0
  was cut — lets it through unwrapped as the bank's `ValueError`. Both are
  right for their version, so the tests assert the one the installed polars
  has rather than pinning 1.x, which is what the advisory next-major leg
  reported red on. No library code changed by this one.

## [0.9.0] — 2026-09-21

### Added

- **Not using a model before it is ready, stated as intent
  (`docs/WARMUP-AND-CONVERGENCE.md`).** Two gates, neither a number that
  needs a formula in the user's head. `min_settled_frac` withholds
  predictions until the decay window has filled this far toward steady
  state, `settled_frac = 1 − 2^(−T/h)` with `T` the decay time the models
  have seen -- `0.5` is one halflife whatever the row rate. Off by default:
  under a stationary process a mean-form fit is unbiased from its first
  row, so what it guards is a history that does not represent the process,
  which only the user can judge. `max_error_inflation` withholds while the
  estimation error is expected to inflate a prediction's error over the
  noise floor by more than this ratio, `sqrt(1 + edf / n_kish)` -- the
  effective degrees of freedom the solve used over Kish's effective sample
  size. Default `sqrt(2)`: the estimation variance no larger than the noise.
  It tracks the model, so adding a feature moves the gate, and reads Kish's
  `n` rather than the weight, so uneven weights withhold for longer. Every
  row now says how settled its stream is (`settled_frac`) and why its
  predictions are null (`withheld_reason`, a categorical), and on `coef`'s
  rows `ewridge` reports each coefficient's data share (`support_coef`,
  `1 − ridge·(S⁻¹)_jj`: a duplicated pair reads 0.5 each). Opt-in,
  `emit_error_inflation` gives the same ratio for each row's own features,
  the row's leverage against the factor its fit came from, so a row leaning
  on a direction the data never showed reads large. `summary()` carries the
  same statistics per group, and a `ReadinessWarning` is raised once per
  (spec, group) for a coefficient more ridge than data, and for a noise gate
  the stream has settled below, with the way out. Measured against the
  identities the theory predicts: the gate's ratio tracks the observed
  out-of-sample error within 2% down to `n_kish ≈ 2.3` coefficients; the
  per-row form is conservative (10% high at 12, 54% high at 2.3 -- what an
  exact `G₂` would buy).

### Changed

- **`ewridge`'s `min_periods` defaults to 0**: the noise gate is its
  readiness gate, and the `k + 1` rows its first solve needs are the
  model's own floor, not a setting. An explicit `min_periods` still floors.
  On unit weights `sqrt(2)` opens where `k + 1` did, so the first prediction
  is on the same row; under a heavy row weight it opens later, as it should.
  Every other model keeps `min_periods` and its default. States saved by
  0.8.x do not load (schema 13).
- **One clock check instead of two, and it reads `max_dclock`.** A backwards
  clock jump smaller than `min_backwards_jump` is refused as out-of-order
  rows whatever `on_clock_reset` says, and the bank is untouched. It
  defaults to `max_dclock`: adjacent rows are never further apart than that
  and a session is longer, so a jump back by less is a late row, not a
  boundary. A jump of at least that much takes the policy. `0` switches it
  off, and it is off by default under an infinite `max_dclock`, which gives
  it nothing to compare against. Measured on one-second ticks under a
  five-minute cap: a row thirty seconds late and one four minutes late are
  refused, where the old typical-step rule accepted both silently, and a
  real day boundary is accepted. States saved by 0.8.x do not load (schema
  12: the clock state no longer carries the removed rules' fields; the
  pre-1.0 policy).

### Removed

- **`min_session_clock`** and the frequency rule behind it, which measured
  the span between two backwards jumps against a "session length" that no
  parameter carries. Its default of `max_dclock` caught nothing (0 of 2,965
  jumps in a reordered file), and 0.8.1's default of the halflife refused
  every intraday stream whose sessions were shorter than the model's
  memory, which is the normal case. Session length was never the question;
  only the size of the jump is.
- **`backwards_jitter_ratio`** and the typical-step estimate behind it,
  which compared a jump to an exponentially weighted mean of recent forward
  steps. That mean never exceeds `max_dclock`, so the rule could not refuse
  anything the cap does not, while costing two persisted fields, a decay
  constant and a warmup during which it was silent. Set `min_backwards_jump`
  instead, in clock units.

## [0.8.1] — tagged 2026-09-20, never published

### Changed

- **`min_session_clock` now defaults to the larger of `max_dclock` and the
  halflife** (a `lam` read as the halflife it is), where it defaulted to
  `max_dclock` alone. That bound -- a session shorter than one adjacency gap
  is not a session -- put a row-scale number on a session-scale quantity.
  Measured against a query engine's block reordering of an ordered file
  (DuckDB with `preserve_insertion_order = false`), `max_dclock = 10` caught
  none of 2,965 backwards jumps whose spans ran from 71 to 2.5e6 clock
  units, and the bank fitted the shuffled rows silently; the halflife
  refused at the seventh. Never weaker than before, since `max_dclock` still
  sets the floor, and now on for a `max_dclock = inf` spec with a finite
  halflife, where it was off. A stream whose genuine sessions are shorter
  than its halflife should declare them with a `session` column, which
  clears the inferred session and is the intended way to state a boundary.

## [0.8.0] — 2026-09-19

### Added

- **Obviously out-of-order rows are refused by default, whatever
  `on_clock_reset` says.** Two checks on the clock, each disabled with `0`,
  each named in the error together with the key that turns it off:
  `backwards_jitter_ratio` (default `1.0`) refuses a backwards step no larger
  than that many typical forward steps -- a transposed pair, a row one tick
  late, two sources never merged -- on its first occurrence;
  `min_session_clock` (default
  `max_dclock`, off when that is `inf`) refuses a second backwards jump
  closer than that to the previous one, a "session" too short to be one. A
  single backwards jump that then holds is still a session boundary and
  takes the policy as before. The refusal is chunk-level through the
  existing pre-scan, so the bank is untouched. Both need `clock`, and both
  guard learning only: `predict` scores a row before the last learned clock
  as the policy says, as it always did. This is a behaviour change for
  streams that fed such rows under `"max"`, `"zero"` or `"reset_state"`,
  which absorbed them silently; `"error"` streams see no change.

The review of 2026-09-18 (`docs/REVIEW-2026-09-18.md`: every file in the
repository read, each finding reproduced by the test that now pins it) found
three defects that returned silently wrong numbers, and a set of crashes and
lost guards. All are fixed here. Numbers move for `corrchange`, for
`huber`/`quantile` on data with a level, for `ew_ridge` through the origin
with a prior, and for `kmeans` where a far row sat in the seed buffer; the
state schema is 11.

### Fixed

- **`corrchange`'s monitor statistic depended on the columns' units.** The
  delta-method gradient of `ρ` with respect to the two variances carried
  `σ²` where it needed `σ³`, so the long-run variance of the estimate scaled
  with the data: on `(x, 100·y)` the size test read 0 flags in 200 where
  5 % is nominal. The gradient is now `[−½σ_xy/(σ_x³σ_y), −½σ_xy/(σ_xσ_y³),
  1/(σ_xσ_y)]`, and the statistic is free of the units to 1e-9 (S3).
- **`huber` and `quantile` lost the fit on data with a level.** The robust
  models kept raw cross-moments `E[z·y]` beside a centred covariance, and at
  a level of 1e8 the slopes were rounding noise (a quantile slope of 1.000
  for 2.032). The cross-moments are now centred, `E[(z − m)(y − ȳ)]` in
  Welford form, with the intercept solved out; a level of 1e8 costs the fit
  nothing (S2).
- **`ew_ridge` dropped `coef_prior` in the standardized solve through the
  origin.** With `standardize = true` and `add_intercept = false` the prior
  never entered the right-hand side: `ridge = 1e12` landed on zero, not on
  the prior, and a moderate ridge fit the same with a prior as without. It
  enters as `ridge · c0 · s`, the prior in standardized units (B2).
- **A NaN passed the core validators of `sgd`, `kalman` and `rls`.** Each
  bound tested `v <= 0` or `v < 0`, which a NaN passes: a NaN
  `clip_gradient` panicked in `f64::clamp` on the first learned row, a NaN
  `obs_var` made `kalman` predict its prior for the life of the stream with
  no error, and a NaN entry in `rls`'s `coef_prior` never left the QR state.
  Every such field is refused by name, from TOML and from hand-written JSON
  as the Python builders already did (B4).
- **`kmeans`: a far row in the seed buffer set a cluster radius to ∞.**
  Seeding marks rows far by the buffer's own cut, and the first split-merge
  check folded that pool at a cut of ∞ before any cluster had a trusted
  radius, after which the typical radius was ∞ or NaN. The pool is skipped
  while there is no cut to place it at (B5).
- **A windowed `ew_cov` without lags did not round-trip in the compact
  encoding.** Two `skip_serializing_if` fields, one of them not last, in a
  positional encoding: the window decoded into the lag slot. `ew_cov` writes
  its lag slot as `nil` (B6). A window snapshot's Kish sum is now its last
  field for the same rule; no model wrote a snapshot without one, so nothing
  observed changes there (B7).
- **Every model's `restore`, and the stream's, check the state's shape
  against its cfg.** `rls`, `robust`, `pa`, `ftrl`, `lasso`'s selection
  fields, `deco`, `ew_cov`, `rcov`, `corrchange`, `marginal`, `ew_class`,
  `hmm`, `holt`, `bocpd`, `seqtest`, `kmeans` and `micro` loaded a state
  whose vectors had the wrong length and panicked on the first row; the
  stream copied its per-instance diagnostics with only their outer length
  checked, replayed waiting rows of any width, and never compared a
  restored model's width with the spec's. All of it is refused as an
  invalid state, as `ew_ridge`, `sgd` and `kalman` already did, and the
  bit-flip fuzz now runs `fit_predict` on every state that loads (B3).
- **The sign hit rate ages on an excluded zero target.** Under a non-binary
  loss a finite `y == 0` is not scored (it has no sign), but it now ages the
  hit weight like any other row; before, the hit rate ran on a different
  clock from `ic` and `r2` on a stream with exact zeros (S1).
- **`ew_cov`'s lag-correlation reads `None` until a pair of the configured
  lag exists.** For `lag >= 2` the accessor returned `Some(0.0)` for the
  first `lag - 1` rows, before any lagged pair had been seen (D1).
- **A windowed `marginal` or `ew_class` reports `n_eff` as exactly 0 once a
  gap empties the window**, matching its pairs, rather than the rounding
  crumb the truncating subtraction left (S4).
- **An integer column used as both `session` and `group` is ordered
  numerically under `group_close = "monotone"`.** It was read as text, so a
  numerically sorted key spanning single and double digits was refused at
  "10 after 9".
- **`hmm`: a row whose state densities are all non-finite ages the whole
  filter, not just `n_eff`.** Such a row now decays the states and the
  transition counts like a zero-weight row instead of advancing `n_eff`
  alone; it is rare (every state's density must underflow at once).

### Performance

- **The windowed models build their per-row snapshot only when the window
  keeps it.** `ew_ridge`, `ew_cov`, `lasso`, `marginal` and `ew_class` built
  an O(k²) snapshot on every row and dropped all but one in `window_every`;
  it is now formed inside the store, so the discarded ones are never built.
  No output changes (P1).
- **`hmm`'s `full` covariance shape caches its per-state factorization**, as
  `ew_class` does, so a `learn = false` scorer factorizes each state once
  rather than on every row. Output is bit-identical.
- **`refresh_time` clones a group key only when the group is first seen**, not
  on every row.

### Changed

- **State schema 11.** `huber`/`quantile` keep centred cross-moments,
  `ew_cov` writes its lag slot unconditionally, and a window snapshot's
  Kish sum is its last field. States saved by 0.7.x do not load
  (`MIN_SCHEMA_VERSION` is 11, the pre-1.0 policy).

## [0.7.4] — 2026-09-18

A patch: `fit(lf)` reads its plan once and skips the deprecated JSON scan
where nothing in the plan can match, 2.3× faster on a small fit. No model
returns a different number than it did in 0.7.3, the state schema is
unchanged, and no spec changed. The review before this tag found no live
defect; one latent blind path — an empty `explain` text would have been read
as "no markers" and skipped a real hazard, though polars never produces one
— is closed and pinned, and the join-variant test now covers `right`, `full`,
`join_asof` and `join_where` as the code comment already claimed.

### Changed

- **A plan is read once per run, and the deprecated JSON scan is skipped when
  nothing in it can be a hazard.** `fit(lf)` inspected the plan twice: once
  through `explain` for `ConsumedSourceWarning`, and once through
  `serialize(format="json")` for `OrderNotGuaranteedWarning`. The `explain`
  text is now read once and handed to both, and it is read *first* as a
  filter — a plan whose text holds no `JOIN`, `AGGREGATE` or `UNIQUE` cannot
  contain a node the order walk reports, so the JSON is never touched.

  Measured on a 100-row fit with a typical `halflife` spec: **0.382 ms →
  0.164 ms**, the fixed plan overhead falling from +0.280 ms to +0.060 ms
  (4.7×). Sharing the text is what makes this a saving rather than a wash —
  reading `explain` twice would cost more than the JSON scan it avoids on a
  small plan. On a 200-column schema the JSON path alone was 2.13 ms, of which
  the walk, not the serialization, was 1.28 ms.

  The stronger reason is durability, not speed: `serialize(format="json")` is
  deprecated in polars, and most plans now never reach it.

  The filter can only make the check cheaper, never blinder. A plan that
  cannot be explained falls through to the JSON path rather than being
  skipped, and `_HAZARD_TAGS` — the one table both the filter and the walk
  read — is guarded by a test that fails if the walk ever reports a tag the
  table omits, since such a drift would silently stop warning. Checked across
  every join variant (inner, left, right, full, semi, anti, cross),
  `group_by` and `unique`: each carries its marker, and a plan with no hazard
  carries none.

  **No regression preceded this.** Bisected across `v0.7.0`–`v0.7.3` by
  running each tag's Python package against one compiled extension:
  `fit(LazyFrame)` was 0.371, 0.381, 0.376, 0.382 ms — ~3% drift, inside
  noise — and `fit(DataFrame)` flat at 0.102 ms throughout. `ModelBank.fit`
  did not exist before 0.7.0, so there is nothing older to compare. What is
  real is structural and has been true since `fit` was introduced: the plan
  inspection is a fixed ~0.27 ms against ~0.10 ms of fitting, so on small
  inputs `fit(lf)` has always cost several times `fit(df)`. That is the cost
  this change removes.

## [0.7.3] — 2026-09-18

A patch: the spent-stream guard reads polars 2.0's plan spelling as well as
1.x's. No model returns a different number than it did in 0.7.2, the state
schema is unchanged, and no spec changed — one string match widens.

### Fixed

- **`ConsumedSourceWarning`'s discriminator missed this package's own plan
  form on polars 2.0.** py-polars 2.0 renders an IO source that was given an
  `explain_name` as `PYTHON[<name>] SCAN` rather than `PYTHON SCAN`, and
  `_explain_kwargs` names our plan form `polars-online` — passed only where
  the signature accepts it, so 2.0 gets the name and 1.x does not. The check
  tested for the literal `"PYTHON SCAN"`, so on 2.0 it stopped recognising our
  own plan, and `release.yml`'s advisory next-major leg failed the assertion
  that documents why the marker alone is not the discriminator (1 failed,
  2,660 passed, on 2.0.0-rc.1).

  **What was never affected**, measured on 2.0.0-rc.1 rather than assumed:
  `pl.scan_arrow_c_stream` is *not* named by polars — neither over a polars
  frame nor over a DuckDB relation — so both still read `PYTHON SCAN`, and a
  second `fit` over a spent DuckDB stream still raised
  `ConsumedSourceWarning` there. The defect never reached the hazard the
  guard exists for; it was confined to our own plan form, which is reusable
  in any case. No user could meet it either: 2.0.0 final is not on PyPI
  (latest 1.44.2, with 2.0.0rc1 the only 2.x), and installers decline
  prereleases, so `polars>=1.34.0,<3` resolves to 1.44.2. The advisory leg
  did exactly what it exists for — catching a break before the major ships.

  The pattern now matches `PYTHON(\[...\])?\s+SCAN`, so 1.x and 2.0 behave the
  same rather than quietly differently; on 1.44.2 the answers are unchanged
  where they were already right. Both spellings are pinned as literal text, so
  they hold with no 2.0 install, alongside three shapes that must *not* match
  — an in-memory frame, a parquet scan, and a bare `SCAN []` — because the
  risk in widening this pattern is that it starts matching anything with
  `SCAN` in it, and would then warn on exactly the reusable sources the guard
  exists to leave alone.

## [0.7.2] — 2026-09-18

A patch: `ModelBank.fit` stops warning about a row order that cannot change
what it leaves behind. No model returns a different number than it did in
0.7.1, the state schema is unchanged, and no spec changed. What moves is where
a warning fires, and it fires in strictly fewer places — every case that
warned before still warns, except a fit whose result provably does not depend
on the order.

### Changed

- **`ModelBank.fit` no longer warns about row order where order cannot change
  what it produces.** A fit whose every spec is an accumulator with no decay —
  `ewridge`, `rls`, `huber` or `lasso` at `lam=1.0` — reaches the same
  coefficients whatever order the rows arrived in, because its sums commute.
  Measured over 200 rows, ordered against shuffled: 3.3e-16, 8.9e-16, 6.7e-16
  and 7.8e-16 respectively — **to rounding, never to the bit**, since the sums
  commute mathematically but not in floating point. `fit` returns nothing and
  keeps only the state, so for those specs the order genuinely does not matter
  and `OrderNotGuaranteedWarning` was a false positive.

  The exception is deliberately narrow, and the measurements are why. It does
  **not** extend to `fit_predict_batches` or `lf.online.fit_predict` over the
  very same specs: `pred` is out-of-sample by construction, so row *i* is
  predicted from the rows before it and reordering moves every prediction —
  1.33 on the same rows whose coefficients agreed to 3.3e-16. Nor does it
  extend to any model whose update does not commute, which is every other one:
  with no decay at all, `sgd` moves 5.9e-03, `pa` 5.1e-02, `ftrl` 3.3e-02 and
  `quantile` 2.8e-03. "No halflife" is not on its own a reason to expect order
  not to matter.

  Each disqualifying option was measured rather than assumed: `window` 8.3e-03
  (and it lives *inside* the nested `model` dict for `ewridge` and `lasso`, so
  a top-level check would miss it), `gram_block_rows` 6.3e-04 — which row sits
  in the pending block when a solve fires depends on arrival order —
  `label_delay` 4.3e-04, and `drift_action="reset"` **8.9e-01**, the largest of
  all. That last one read as harmless at 3.3e-16 until the fixture actually
  made drift fire: a green result from a code path that never executed is not
  evidence, and it is now denied on measurement. Options whose path could not
  be made to fire at all — `session`/`session_gap`, `ridge_decay`,
  `long_halflife`/`session_shrink` — are denied as unproven rather than
  promoted. A spec key the check does not recognise counts as unsafe, so an
  option added later cannot quietly become exempt.

## [0.7.1] — 2026-09-18

A patch: a plan that read a spent Arrow stream is reported now, rather than
passing for a fit that learned nothing. No model returns a different number
than it did in 0.7.0, the state schema is unchanged, and no spec changed —
`ConsumedSourceWarning` is a diagnostic over what a bank already did, and the
only public name this release adds.

### Fixed

- **A plan that read a spent Arrow stream no longer passes for a fit.** An
  Arrow C stream is consumed once — the PyCapsule specification says a capsule
  "can only be consumed once" — so a `LazyFrame` from
  `pl.scan_arrow_c_stream(...)` over a DuckDB relation or a pyarrow reader
  gives its rows the first time it is collected and nothing every time after,
  silently, with no error from polars (measured: 1,000 rows, then 0). A bank
  fed that plan twice learned from every row and then from none, and the
  second run left a state that looked finished and was empty.
  `ModelBank.fit`, `ModelBank.fit_predict_batches` and `lf.online.fit_predict`
  now raise `ConsumedSourceWarning` when a plan whose source is a Python scan
  delivers no rows at all, naming the call, the fix (rebuild the plan per run)
  and how to silence it. The *pair* is the discriminator, not either half:
  re-collecting a `scan_parquet` is legitimate, and this package's own plan
  form carries the same `PYTHON SCAN` marker while being perfectly reusable,
  so neither can trip it — nor can an empty in-memory frame, which stays an
  ordinary run. It warns rather than raises because an empty query is not a
  mistake, and a `head(0)` pushed into the scan is excluded for the same
  reason. The check reads `explain`, which does not execute the plan and so
  cannot consume the stream it exists to protect.

### Changed

- **`duckdb` joins the `dev` dependency group**, which turned three claims in
  `docs/ARROW-SOURCES.md` from recalled into measured — and two of them were
  wrong. Its DuckDB recipe did not run: `record_batch` is not a method on
  duckdb 1.5.5, and a relation resolves an unknown attribute as a column name,
  so it raised `AttributeError: This relation does not contain a column by the
  name of 'record_batch'`; the surviving spellings (`fetch_record_batch`,
  `to_arrow_reader`, `rel.pl()`) all need pyarrow, so the recipe is now
  `pl.scan_arrow_c_stream(rel)`, which needs neither and streams. Its §3 cited
  DuckDB issue #17084 for a relation's `__arrow_c_stream__` working only once
  and built a decision on that precedent; on 1.5.5 both calls succeed, so the
  precedent is gone — what survives is that the captured *stream* is
  single-use while the relation is not. And its comparison table claimed
  memory "O(state), independent of stream length" end to end, which the
  measurements do not earn (106 MB at 1M rows and 271 MB at 4M with no bank
  attached at all). The package's own dependencies are unchanged: it still
  requires only `polars`, and every duckdb test is `importorskip`ed, so none
  of this is needed to run the suite.

## [0.7.0] — 2026-09-17

### Documentation

- **Every document reads as the code stands.** A sweep for what tasks 83, 85
  and 86 and the 0.7.0 release had left behind: `docs/PLAN.md` §6 described
  the expression plugin as shipping and is now the record of its removal and
  the reason; `docs/STATE-WORKFLOW.md` presented `po.run` as a live surface in
  its four-step guide, its table and its rules, and now carries a dated note
  with the CLI and `ModelBank.fit(lf)` in its place; `docs/PERFORMANCE.md`'s
  runner and memory sections say what their `po.run` and expression rows now
  are; `runner.rs`'s doc comments no longer describe a Python caller; the
  testing ledger's runner row cites the tests that exist rather than the
  deleted `test_runner.py`; the extending guide no longer cites a deleted
  helper; the version-pin examples say `~=0.7.0`; and CLAUDE.md's layout
  calls the bank what it is. The ledgers -- this file, the enhancement and
  improvement lists, the phrasing and review logs -- keep their history.
- **The runner guide's shell examples are executed, not asserted.** Its
  Python block became a command-line invocation when the runner left Python
  (task 83), which left the guide with no runnable examples at all while the
  README's every Python block runs. The harness now collects ```sh blocks
  from that guide -- and only that guide, since the README's are `pip
  install`, `uv sync` and the development commands -- and runs each line
  against the built `online` binary, in a directory holding the files the
  fixture writes, with the binary on `PATH` so a block runs exactly as
  printed.
  The first run found three false claims in the guide, all now fixed rather
  than worked around: the `--resume` examples read a state saved from a
  *different* spec than the config declared, so they could not have loaded;
  the sidecar example borrowed a config with no `group_close`, which the
  binary correctly refuses; and the state-vocabulary paragraph, the
  parallelism note, the chunk-size note and the version floor all still cited
  the removed Python entry point. The seven-line montage of invocations is
  now separate blocks, each self-contained, because an example that cannot
  run on its own cannot be checked.


- **How to tune memory with Polars' own settings**, in the README. The
  read-ahead is what grows, not the bank: the streaming engine prefetches row
  groups ahead of whatever consumes them, sized from the thread count, and a
  local disk needs none of it while a bank is the bottleneck. Three variables
  move it, and the section says which, what they measure, and what not to
  carry across. Measured here on 8M rows by 12 columns in 80 row groups, with
  the allocator's page retention off: a sink goes 1.63 to 1.12 GB and a
  batched bank 1.41 to 1.07 GB. The prefetch is read *per scan*, not once at
  import, so it can be set from Python at any point before the scan that
  should use it -- shell, before-import and after-import all give the same
  number. The reduction depends on how much data a row group holds, so the
  larger figure in `docs/PERFORMANCE.md` (1.86 to 0.51 GB, on 262,000-row
  groups) does not carry to a file with smaller ones.

### Added

- **A plan whose row order is unspecified is warned about before a bank reads
  it.** An online model learns in row order, so the order a plan delivers is
  part of the model -- and `fit(lf)`, `fit_predict_batches(lf)` and
  `lf.online.fit_predict` run the plan through polars' streaming engine, where
  a `join`, `group_by` or `unique` without an order guarantee delivers a
  different stream than `lf.collect()` gives. Measured on 200,000 rows:
  `collect()` kept the input order, `collect_batches()` did not, and
  `maintain_order="left"` made the two agree. So a plan checked by collecting
  it learned something else when fed, silently. Each entry point now inspects
  the plan when it is handed over and raises `OrderNotGuaranteedWarning`,
  naming the node and the fix: `maintain_order="left"` on a join,
  `maintain_order=True` on a `group_by`, a sort after a `unique` (whose flag
  the streaming engine does not honour), or a sort before the bank. A sort
  above the node settles it and is not flagged. Best-effort by design: the
  inspection reads `LazyFrame.serialize(format="json")`, which polars has
  deprecated, and falls silent rather than fail on a plan it cannot read --
  one already holding a bank, for instance. The warning is a `UserWarning`,
  shown by default; the note on each method is the guarantee under it.
- **`ArrowStruct` is exported and documented.** `fit_predict_arrow` returned a
  type a caller could neither import for an annotation nor find in the
  reference; it is `polars_online.ArrowStruct` now.
- **The model bank is Arrow inside, with Polars as an adapter.** The bank no
  longer reads a `DataFrame` or builds a named Series: it reads an
  `ArrowChunk` and returns one Arrow struct array per spec. `fit_predict` and
  `predict` are that pair between the adapter that makes a chunk from a frame
  and the step that names each struct after its spec, so they behave exactly
  as before — same values, same errors, same order.
  The split is by what a decision is *about*. Everything Polars-shaped —
  finding a column by name, refusing a dtype that is not numeric, the cast to
  `Float64`, the cast of a key or a label to text, whether a group key is an
  integer, and the refusals of a temporal clock and of a group column
  `group_close = "monotone"` cannot order — moved into one adapter module.
  Everything model-shaped stayed in the bank, because it holds whoever
  supplies the data: a finite clock, a non-negative weight, a `strict_binary`
  target that is 0 or 1, a hazard above 1, a label among the declared classes.
  Four helpers went away entirely, and `compare_targets` — the one place the
  bank read its *own* output back through Polars — now finds the field in the
  struct's own schema.
  What this opens: a caller holding Arrow arrays can feed the bank with no
  Polars in the process, through `Bank::fit_predict_arrow`. Proven rather than
  claimed, with a chunk built by hand from arrays nothing in Polars ever
  touched.

- **`ModelBank.fit_predict_arrow`**, the output over the Arrow PyCapsule
  interface. `PySeries` reaches py-polars' private `_export`/`_import`, which
  is why this package carries a Polars floor and why that interface promises
  no stability; `__arrow_c_array__` is public and standardised, and any Arrow
  consumer reads it — `pl.Series(obj)`, pyarrow, duckdb. The values are
  `fit_predict`'s exactly: equal dtype, equal length, identical field names,
  identical per-field null counts, and the nested `coef` list equal with nulls
  compared rather than skipped. Exporting hands the buffers to the consumer,
  so a struct exports once and says so if asked twice. The input side still
  arrives as a frame.

- **`ModelBank.fit_predict_batches` takes a `LazyFrame`**, and does the
  chunking itself: the plan is read `chunk_rows` rows at a time (100,000 by
  default) and fed chunk by chunk, so memory is the state plus a chunk however
  long the plan's input. A `DataFrame` is one chunk. An iterator of frames is
  fed as it comes, unchanged. This is the shape the file-to-file runner had,
  without the file.
- **`ModelBank.fit`**, the run whose product is its state: the same pass with
  the output dropped as it comes, so no chunk's result is held and no frame is
  assembled from them. The state it leaves is byte-identical to the one
  `fit_predict_batches` leaves over the same rows. It saves the result, not the
  work -- every row is still predicted before it is learned from, which is what
  makes the fit out-of-sample, and the output columns are still built before
  they are dropped.

### Changed

- **The advisory next-major CI leg is green again.** One test line passed
  `memory_map=False` to `pl.read_ipc`, a keyword polars 2.0 removed, which
  made the `release.yml` "next major" leg red on 2.0.0rc1 from 0.6.0 onward --
  two failures in 2,639, both the same parametrised test. On polars 1.x that
  keyword already defaults to `False`, so it was passing the default
  explicitly and dropping it changes nothing on either version. Verified by
  building the wheel and running the test against a real 2.0.0rc1, not by
  reading the signature. Test-side only: the package never passes
  `memory_map`, so no published wheel was ever affected, and the leg is
  `continue-on-error` by design so it never withheld a release.

- **`predict` refuses a group key that `group_close = "monotone"` cannot
  order, as `fit_predict` does.** The check moved into the chunk adapter with
  the Arrow work, so both calls meet it; before, `predict` alone ran on a float
  key the spec could never have ordered. Pinned by a test either way.
- **A hand-built `ArrowChunk` is validated.** `ArrowChunk::new` refuses a
  column given but not listed in `names` -- it was silently invisible to the
  check that lets a scoring call leave a target out, so the target scored as
  missing -- and a name given twice in the same form, where the first silently
  won. A column supplied in the wrong form, a clock as `Int64Array`, is named
  as such rather than reported "not found" beside a list that includes it.
  `new` takes any name that converts to a `PlSmallStr`, `&str` included, and
  the crate re-exports `PlSmallStr`.
- **`fit_predict_batches` slices a `DataFrame` when `chunk_rows` is given**,
  where it validated the argument and then fed the frame whole.
- **A broken bank refuses a chunk before the chunk is cast.** The Arrow
  adapter had moved the cast ahead of the check, so a bank that could not go
  on reported a column error instead, and did a frame's worth of casting to
  find it. The "not found" message says "the input has columns", one wording
  for a frame and for a hand-built chunk alike.
- **The API snapshot records `ModelBank` signatures, not names alone.** A
  parameter added or a default moved on a method is now a diff in
  `tests/api_surface.txt`, as it already was for the spec builders and the
  frame namespaces; `fit_predict_batches` gaining `chunk_rows` had left no
  trace.

### Removed

- **The expression plugin is gone: `pl.col("y").online.<model>(...)`,
  `po.online`, and `InMemoryExpressionWarning` with it.** It read every row by
  construction -- polars hands a stateful user expression its whole column in
  either engine -- so it was the one surface that could not stream, and it
  warned on every use to say so. Everything it did, the bank and the plan do
  without the warning: features that were expressions become columns computed
  before the call, and `.over(group)` becomes the spec's `group`.
  **What it cost the wheel, measured rather than assumed.** The plugin needed
  `pyo3-polars`'s `derive` feature, which turns on `polars-plan/python`, which
  only `polars-lazy/python` propagates to `polars-mem-engine` -- so `derive`
  was why `lazy` was there too. With the plugin gone, `cargo check -p online-py`
  builds clean without either, and the extension no longer asks pyo3-polars for
  the query engine. `online-polars` still does, for the runner the command line
  uses.
  About thirty tests went with it. They asserted expression-equals-bank, but
  both entry points call the same `Bank::fit_predict` in Rust, so what they
  checked was the plugin's column packing, not any model's arithmetic; each
  model keeps its own bank tests and, for five of them, its numpy oracle.

- **`polars_online.run` is gone; the `online` command line keeps the runner.**
  The runner's Python entry point built its chunk iterator with py-polars and
  handed the frames to Rust, so it duplicated what `ModelBank` and
  `lf.online.fit_predict` already do from Python while dragging the whole
  Polars lazy engine, the parquet, CSV and IPC readers and the sinks into the
  extension module, where no Python path reached them. File-to-file work is
  the `online` binary's (`docs/RUNNER.md`), and in-process work is the bank's.
  Its per-chunk `progress` callback has no command-line equivalent and is
  removed rather than moved; a caller driving `fit_predict_batches` counts
  chunks itself.

## [0.6.0] — 2026-09-16

A minor: many models return a different number than they did in 0.5.1, and
the state schema is 10, so a file written by an earlier release is refused
rather than loaded. Behind it are two full reviews of the library
(`docs/REVIEW-2026-09-12.md`, `docs/REVIEW-2026-09-15.md`), a new reading
for a target that is null on some rows (`target_gaps`), and the whole
Python API reference and the README written again from scratch. Every
entry that moves a number says so.

### Changed

- **A target null on some rows is fitted on its own rows** (`ewridge`,
  `lasso`; docs/PLAN.md task 81). They read the Gram over every row against
  the target's cross-moments over its own, so each slope moved with the
  target's level: `(m_j − m)·ȳ_j / Var(x)` on top of the fit, `m_j` a
  feature's mean over the target's rows. A new parameter, `target_gaps`,
  picks between two readings. `"own_rows"`, the default, is exactly the fit
  of the rows with the target's nulls dropped. `"pairwise"` keeps one Gram
  over every row and centres each target's cross-moments at its own means,
  as pandas' pairwise-complete covariance does. **Numbers change** for every
  `ewridge` and `lasso` stream with a null target. Under `"own_rows"` a null
  target no longer moves its own fit at all, and `n_eff` keeps counting the
  row.
- **`lasso` keeps its cross-moments centred**, as `ewridge` has since the
  code review's N1, so a level on the features and the target costs its path
  nothing. Without gaps its fit is the same to rounding; the arithmetic is
  different, so the last bits are not guaranteed. `ewridge` without gaps is
  unchanged to the bit.
- **`bank.gram()` returns one entry per Gram**, each naming its `targets`,
  and every entry carries `means_by_target`, each target's column means over
  its own rows. Under `"own_rows"` targets missing on different rows have
  Grams of their own, and a closed group writes one row per Gram, with that
  Gram's targets' coefficients and weight. `po.gram.solve`, `lasso_path` and
  `coef_stats` read `means_by_target`, so they reproduce either reading.
- **`po.gram.solve(standardize=True)` and `po.gram.lasso_path` without an
  intercept read raw moments**, as the models have since the code review's
  C8. They scaled centred co-moments against raw cross-moments there, which
  is least squares only when every feature has mean zero.
- **State schema 8; files of schema 6 and 7 no longer load.** While the
  project is pre-1.0, a state saved before this release has to be refit.
- **A spec that sets a knob its switch leaves off is refused**, where it
  was ignored without a word. That covers `drift_action = "reset"`,
  `drift_delta` or `drift_threshold` without `emit_drift`; `average_eta`
  without `emit_averaged`; `resid_autocorr_lag` without `emit_autocorr`;
  `long_halflife` without `session_shrink`; `session_gap` without
  `session`; `on_clock_reset` without `clock`; and `coef_every` on a model
  that reports no coefficients. `session_shrink` is refused beside
  `session_gap = "reset"` or `group_close = "session"`, where the blend
  never ran. `holt` takes `halflife` or `level_halflife`, not both.
- **Feature sets and spec names are checked by name.** A set named twice, a
  column twice in one set, an empty set and `feature_sets = []` are refused,
  and so is a spec named `""`, `"spec"` or `"group"`, which the bank's
  tables use for their own columns. `marginal` refuses `window` with
  `lags`, as `ew_cov` does.
- **Every way into the library checks a spec the same way.**
  `output_fields`, `output_index`, `coef_fields`, the expression plugin and
  a run config's `validate` now fill a spec's defaults and build its models,
  as the bank does. A dict for a model with no target that leaves `targets`
  out is accepted by all of them, and a spec the bank refuses is refused by
  all of them.
- **`add_intercept` no longer moves the warm-up** of `ew_cov`, `kmeans`,
  `micro`, `ew_class` and `holt`, which have no intercept. Their default
  `min_periods` is `k + 1` either way, the value the builders' default
  gave, so only a spec with `add_intercept = False` reports its first row
  one row later.
- **A window costs less per row.** `predict` and `n_eff` read the window's
  weights alone, where they copied the whole O(k²) accumulator on every row.
  At k = 200 a windowed `ewridge` row went from 34.6 to 17.2 µs, and an
  `ew_cov` that emits no statistics from 28.1 to 9.8 µs. Every number is
  the same to the bit.
- **`Kalman::pred_var` takes the row it answers for** (Rust API). It read
  the last row's regressor, which a save drops, so a loaded filter answered
  with the observation noise alone.
- **A window's snapshots have a memory budget** (`ewridge`, `lasso`,
  `ew_cov`, `ew_class`, `marginal`; the code review's P4). The ring grew with
  the window and nothing checked it: at k = 1000 over a 3,600-row window it
  was about 29 GB per instance. A new parameter, `window_budget`, bounds each
  ring in MiB. `{"thin": mib}` drops every other snapshot and doubles the
  spacing, as often as it takes, which can only shorten the window.
  `{"refuse": mib}` stops the ring at the budget and refuses the chunk,
  naming the ring's size and `window_every`. **A window with no budget
  refuses past 256 MiB**, so a run whose ring grew past that, which ran
  before, now stops with that error; `{"refuse": float("inf")}` is no bound.
  The budget is checked as the rows are learned, so a bank refused for it
  has learned part of the chunk: it refuses every later `fit_predict`,
  `predict` and `save`, and is rebuilt from its last save.
- **`fit_predict_batches` drains the closed groups as it goes**, given a
  `closed_groups` path, and writes what it drained when the chunks stop --
  at their end, at a `break`, or at an error (the code review's P5). A
  `ModelBank`'s queue is bounded only by draining it.
- **`holt`'s level and trend are weighted means** (the code review's S29
  and S30). Each is `(λ·W·old + w·new)/(λ·W + w)`, as every accumulator here
  is, so a row at weight `w` counts `w` times and an infinite halflife fits
  the whole history. **Numbers change for every `holt` stream**, weighted or
  not: the gains start at 1 and fall to the textbook's fixed ones as the
  weight saturates, so the first rows follow the series sooner; from there
  the fit is statsmodels' `Holt`. The textbook recursion read a weight only
  as learn-or-not, and at an infinite halflife froze the level at the first
  row. **`trend_halflife=inf` is now the whole history's drift, not a trend
  pinned at zero**, and a plain level with no trend no longer has a
  spelling. A row at the previous row's clock is a second observation the
  level takes in. `lam=1` builds, as `halflife=inf`; it was refused in
  `level_halflife`'s name.
- **`ftrl` keeps its proximal term as a decayed sum** (the code review's
  C24). Under a halflife, decaying `n` inside the square root shrank every
  coefficient toward zero on every row: a constant target of 5 settled at
  2.25 at `halflife=100`. It settles at 4.65 now, `5/(1 + (1 − λ)(β/α +
  l2))`: the penalties stay constants on the sums' scale, a mean-scale ridge
  of `(1 − λ)(β/α + l2)`, and a gap still scales the coefficients by
  `λ^t(d + c)/(λ^t·d + c)`, which the docs state. **Numbers change for every
  `ftrl` stream with a halflife**; without one it is river's to the bit, as
  before.
- **State files are schema 9.** `holt` and `ftrl` carry new state, and a
  schema-8 file is refused by its version (pre-1.0, no loader).
- **Each target's `min_periods` is checked against its own weight** (the
  code review's S2) -- the rows it was present on, inside the window under
  one -- for `ewridge`, `lasso`, `kalman`, `huber`, `quantile` and `holt`.
  It was checked against the shared `n_eff`, the same for every target, so
  a target present on one row in ten passed warm-up on the other targets'
  count. **A sparse target's first prediction comes later**; the emitted
  `n_eff` is the shared weight, as before.
- **`emit_averaged` compares the slots' errors as ratios** (the code
  review's S23): each slot weighs `exp(−eta·(σ²/σ²_best − 1))`, so `eta`
  means the same in any target's units. The weights were
  `exp(−eta·(σ² − σ²_best))`, in the target's units squared, so `eta = 1`
  was an equal-weight mean for a return and the argmin for a price.
  **`pred_<t>__averaged` changes** for every spec that emits it.
- **`inf` is taken where it means something** (the code review's S27):
  `huber_delta` is least squares for `huber` and the squared loss for
  `sgd`, `long_halflife` makes the long run the whole history,
  `select_halflife` selects on the plain mean, `level_halflife` forgets
  nothing (as `halflife` does, the same knob), `pa`'s `c` caps nothing (mode
  `"pa"`), and `average_eta` is `emit_selected`'s argmin, a tie shared. The
  builders refused each, and JSON could not carry it. **Where it means
  nothing it is refused by name**, in `validate` as in the builders:
  `drift_delta` and `drift_threshold` (a detector that never fires),
  `quantile_eps` and `pa`'s `eps` (a model that never learns), `ftrl`'s
  `alpha`, `beta`, `l1` and `l2`, and `sgd`'s `learning_rate`. A TOML `inf`
  got past `validate` for each. `ridge` and `kalman`'s `q` leave the
  builders' table of what may be infinite; Rust refused both already.
- **`marginal` takes `min_periods = inf`**, a gate that never opens, as
  every other model does and the builders document. It alone refused it.
- **Under a `window`, `sigma` and `resid_z` are the window's** (`ewridge`,
  `lasso`; the code review's S1). The fit is read from the rows inside the
  window, and the spread beside it was the stream's EW mean over the whole
  history, so a burst of errors the window had dropped still widened
  `sigma` for as long as the halflife remembered it. The stream now cuts
  its spread with a ring of snapshots of its own, on the model's rows and
  its `window` and `window_every`, so the two describe the same rows. What
  reads the spread moves with it: drift's scale, the conformal band, and
  the ranking `emit_selected` and `emit_averaged` take. **Numbers change**
  for every windowed spec that emits or reads `sigma`. The ring is a pair
  of floats a slot a snapshot, kept only when something reads the spread,
  and `window_budget` bounds it as it bounds the fit's.
- **`bank.gram()` and a closed row carry `cross_centred`**, each target's
  cross-moments centred at its column means and its own mean -- what the
  model solves from (the code review's N4). `po.gram.solve`, `lasso_path`
  and `coef_stats` read them, and `merge` pools them, where each formed
  `cross_moments − m·ȳ`: two numbers the size of `L²` at a level `L`,
  which kept `L²·ε` of the answer. At a level of `1e6` the offline solve's
  predictions were off by `2.5e-4`, and at `1e8` by 3. A closed-groups
  sidecar gains the column.
- **`quantile` fits by a Newton step on the smoothed check loss** (the code
  review's N9). It reweighted each row by the IRLS weight of its prior
  residual, `2·side·σ/max(|r|, eps·σ)`, and froze that weight into the
  accumulators. Splitting the cross-moment by `y = p + r` shows what that
  left: beside the pinball subgradient, a spring of strength `E[w]` pulling
  the fit back toward the fit each row was scored against, so it settled a
  fraction `1/(1 + ln(1/eps))` of the way to the quantile regression -- 0.164
  from `statsmodels`' `QuantReg` at the median of a skewed noise after 20 000
  rows and 0.477 at the 0.9 quantile, against `QuantReg`'s own standard
  errors of 0.007 and 0.021, and a `quantile = 0.9` fit whose coverage read
  0.777. A row inside a band of half-width `h = quantile_eps·σ` is now a
  least-squares row with target `y + 2h(τ − ½)`, and a row outside it adds
  `2h·ψ_τ(r)·z` to the cross-moment and nothing to the Gram: the curvature is
  the same density on both sides, so the rows' own history cancels. Measured
  after it: 0.005 and 0.006 from `QuantReg`, inside its standard errors, and
  a coverage of 0.858 over that stream and 0.893 over its second half.
  **Numbers change for every `quantile` stream**, and the fit follows a level
  shift the frozen weights lagged -- 600 rows after a jump of 3 it covers
  2.87 of it, where it covered 1.2. `huber` is untouched, its weights having
  always been bounded by 1.
- **`quantile_eps` is that band's half-width, and defaults to 0.2**, where it
  was the floor under `|r|` in the IRLS weight and defaulted to `1e-3`. It
  still says "closer than this counts as zero"; what it bounds is the
  curvature the step leans on rather than a weight. At 0.2 the band holds
  about a fifth of a stream at the median and a fifteenth at the 0.9
  quantile: a much narrower one converges more slowly (0.19 from `QuantReg`
  after 8 000 rows at `0.1`, against 0.10 at 0.2) and a much wider one
  smooths the quantile toward the mean (0.08 the other way after 100 000
  rows at 0.4).
- **`quantile`'s warm-up counts the rows present, and its band has a
  floor** (the second review of 2026-09-15, F3). The warm-up read the
  band's weight, which a halflife caps at the band's share of the
  effective sample -- a fifteenth of it at `τ = 0.9` -- so under a short
  halflife the fit kept falling back into warm-up, least-squares rows
  aimed at the mean: coverage 0.825 at `halflife = 30` and 0.864 at 40
  where 0.9 was asked. It counts the rows the target was present on now,
  and the band is never narrower than `(k/n)^{2/5}` of `σ` for the
  target's effective sample `n`, the smoothed-quantile bandwidth rate,
  which a long stream leaves behind and which keeps the step fed where the
  band's share of the sample would be a few rows. Measured after: 0.895
  and 0.896. The warm-up had also rebuilt a fit a row at the input bound
  left behind, whose moments the band's nudges cannot move until its
  weight has decayed to nothing -- where each is a step that outgrows the
  band, and the fit oscillates until that weight underflows (the
  bounded-extremes contract, 1500 halflives on). A band holding under one
  row per coefficient takes least-squares rows until it holds rows again;
  the floor keeps a settled band well clear of that. **Numbers change**
  for `quantile` streams under a finite halflife.
- **The per-target `min_periods` gate reads the rows present for `huber`
  and `quantile`** (F1). It read the target's accumulated weight, which
  for `quantile` was the band's from N9 on -- so under a halflife a
  `min_periods` above the band's saturated weight closed the gate again
  after the first predictions: at `τ = 0.9`, `halflife = 100` and
  `min_periods = 20`, predictions per thousand rows of 178, 233, 2, 88,
  155 and 203 -- and for `huber` the reweighted sum. Both now read the
  rows the target was present on, at their raw weights, decayed: hard rule
  8's number, and the one the docs named. A `huber` stream whose warm-up
  met outliers reports its first prediction a row or two earlier.
- **State files are schema 10.** `robust` carries the per-target
  observation weights, and a schema-9 file is refused by its version
  (pre-1.0, no loader).

### Fixed

- **A failed `run` publishes the closed groups it drained** (F2). The
  sidecar was drained from the bank after every chunk and published only
  by a run that completed, so a run that failed midway had drained the
  earlier chunks' closed rows into a temporary it then deleted, and a
  caller-owned bank no longer held them. It is published with what it
  drained however the run stops, as `fit_predict_batches` already wrote
  on a `break` or an error; the output keeps its whole-run rule.

- **`sgd(loss="huber")` panicked on a NaN `huber_delta`**, from TOML or the
  Rust API: neither the spec nor the model checked it, and `f64::clamp`
  panics on a NaN bound. Both refuse it now.

- **`max_dclock` caps the time a run of skipped rows hands the next row**
  (the code review's S3). Each skipped row's delta was capped on its own and
  the folded total was not: ten skipped rows 100 apart under a cap of 60
  handed the next row 660. **Numbers change** wherever a run of skipped rows
  spans more than `max_dclock`.
  A run of skipped rows past the cap also breaks adjacency, so `ew_cov`'s
  and `marginal`'s lagged co-moments are cleared at the next row, as on a
  capped row.

- **`ftrl(strict_binary=True)` refuses a chunk whose target is not 0 or 1**,
  naming the row and the value, before any stream is touched (the code
  review's S31). It skipped the row silently and still counted it toward
  `n_eff`, where the docs said it was an error.

- **`po.run` and the CLI held every closed group until the run ended** (the
  code review's P5). The `closed_groups` sidecar was drained once, after the
  last chunk, so its rows waited in the bank for the length of the run. The
  bank is drained after every chunk now, and each drain goes to the
  sidecar's writer as it comes; the file is published with the output, as
  before.

- **A window with `window_every` above 1 kept rows older than itself**
  (`ewridge`, `lasso`, `ew_cov`, `ew_class`, `marginal`). After a clock gap
  longer than the window, or wherever `window_every` rows span more clock
  than it, the newest snapshot was older than the window and stayed the
  boundary, so rows the window excludes stayed in the fit. At a cadence of
  5, the row after a long gap reported a window weight of 2 where it alone
  was inside. Such a row is now snapshotted whatever the cadence. **Numbers
  change** for those streams; `window_every = 1` is unchanged.
- **`σ²` ages on every row** (`ewridge`, `robust`, `kalman`). A row with a
  target and no prediction, such as one where `min_periods` is unmet after a
  clock gap, rightly added nothing to the residual variance. It did not age
  its weight either, and neither did a zero-weight row in `robust`. So `σ²`
  forgot less across such rows than the clock says. In `kalman` it sets the
  observation and process noise, and in `robust` the width of every cut, so
  their predictions after such rows change.
- **A `session_shrink` blend under `ridge_decay` keeps the decaying
  prior** (`ewridge`). The blend put the ridge prior back at full strength
  at every session boundary. The prior now mixes as the moments do, `1 − f`
  of the model's and `f` of the long-run twin's, so `session_shrink = 1`
  lands exactly on the twin's fit.
- **`lasso`'s selection survives a zero-weight row on the first
  prediction.** It formed `0/0` there, and the NaN held the selection at the
  heaviest penalty for the life of the state.
- **`solve_failures` counts what it says.** `lasso` counts a coordinate
  descent that runs out of `max_cd_iters` before `cd_tol`, one per target
  and path point; it was always 0. `robust` counts a standardized solve that
  fails at every jitter, as it counted a plain one.
- **A state that contradicts its own config is refused.** That is an `sgd`
  state whose `scaler` disagrees with `scale_features`, or whose AdaGrad
  sums disagree with its schedule. The first loaded and read raw inputs with
  coefficients learned on standardized ones; the second panicked on its
  first step.
- **`ftrl` refuses NaN** in `alpha`, `beta`, `l1` and `l2`. The core took it
  and returned NaN from the first row; the spec layer refused only `alpha`.
- **The bare `EwCov` state calls itself `ew_cov_accumulator`** in errors. It
  said `ew_cov`, which is the bank model's name.
- **`bocpd` and `hmm` read the column they name.** A spec that named
  `hazard_col` or `exog_tvtp` and left `targets` out had `targets` filled
  from the first feature, which was then read as the hazard or the
  exogenous series. **Numbers change** for such specs. The expression form
  now packs that column too; before, it failed when the plan ran.
- **A feature expression named after the clock, session or weight column
  is refused** in the expression form. The plugin reads its inputs by name,
  so such a feature silently replaced that column.
- **`ModelBank.specs` is the spec the bank runs**, filled in, before a save
  as after one. A hand-written dict read differently on the two sides of a
  round trip.
- **`output_index` and `coef_fields` carry a single combo's ridge**, and a
  single named feature set's name; both were null.
- **`po.spec.coef_index` refuses every kind with no coefficients** with a
  `ValueError` naming it. Four of them raised whatever polars raises on an
  empty series.
- **Building `lf.online.predict(bank)` leaves the bank's closed groups
  alone.** Reading the output's schema drained them, so the rows a caller
  had fitted and not yet read were gone before the plan ran.
- **`predict` scores a new session the way `fit_predict` does** under
  `group_close = "session"`: as the first row of a fresh stream, null with
  `n_eff` 0. It scored the new session with the closed one's fit.
- **`hmm`'s warm-up ages as `n_eff` does.** The states were seeded with the
  buffered rows' raw weights, so they started heavier than the rows' age
  warranted, and their ridge weaker. **Numbers change** for an `hmm` with a
  finite halflife.
- **A drift reset builds one model instance**, the one it resets, instead
  of every instance of the halflife grid.

### Documentation

- **The Python API reference is written from scratch** to `docs/WRITING.md`:
  the spec builders, `ModelBank`, the frame and expression namespaces, the
  runner, and the `gram`, `eval`, `corr`, `prep` and `sim` modules. A
  builder now states its fit as math, its parameters with their units, the
  fields of the struct it writes with a link to `docs/OUTPUTS.md`, a
  runnable example, and what it refuses. A function or model that generates
  a downstream table summarises that table and the contents of its structs,
  and links to them. Every `.. code-block:: python` in a public docstring
  runs in the test suite: 59 of them.
- **A mechanism is documented where its API is.** The review records under
  `docs/` are unchanged, and the explanations they carried now also live in
  the docstrings and Rust module comments that own the behaviour, so the
  current state of the library is readable from the API alone.
- **The README is written again from an outline**, so that like sits with
  like, each idea is stated once, and no term is used before it is defined:
  the introduction, the model table, install, how a bank sees a stream,
  running a bank, saving and serving, preparing a stream, reading the fit,
  diagnostics, the model sections, performance, and the comparisons. Prose
  is kept for what code with comments cannot carry — the update rules, the
  sweeps as tables, and the reason behind a rule — and all 59 of its Python
  blocks run in the test suite. The steps for raising the Polars ceiling
  moved to `docs/RELEASE-READINESS.md`, beside the measurements they rest
  on.

## [0.5.1] — 2026-09-11

The first release of the 0.5 series: 0.5.0 was tagged but never published
(below). A minor because the supported range of Polars is wider, which is
a minor release by this package's own rule. No model returns a different
number than it did in 0.4.1.

### Changed

- **The Polars ceiling is raised: `polars>=1.34.0,<3`.** py-polars 2.0 is
  now inside the declared range. The whole suite passes on `2.0.0rc1` with
  the same numbers as on 1.44, all three interfaces work, and
  `LazyFrame.collect_batches` — the floor — is unchanged; the measurements
  are in `docs/RELEASE-READINESS.md`. Nothing about an install changes
  today, since no installer resolves to a release candidate: a user gets
  the newest 1.x until 2.0.0 ships, and then gets it without waiting on a
  release of ours. One Polars 2.0 behaviour moves in our favour — a query
  that fails *after* the bank stops the source instead of draining it, so
  `save_state` is not written on a long stream, narrowing the gap
  `docs/STATE-WORKFLOW.md` calls R6. The Rust side is unchanged: py-polars'
  major and the `polars` crate's version are independent, and the wheel
  carries its own statically linked copy.
- **A release is now blocked on the newest Polars its own range admits.**
  `release.yml`'s check runs in two legs: *the newest in-range* (stable
  only, honouring the ceiling) gates the publish, and *the next major*
  (unpinned, prereleases allowed) is advisory. So a wheel is never
  published green only on the version it was built against, and a beta of
  a major we have not adopted cannot withhold a release.

### Added

- **A plan with nothing to write runs once where a query uses it twice**
  (task 77). The IO source declares `is_pure` to `register_io_source`
  exactly when a run has no `save_state` and no `closed_groups` sidecar, so
  Polars shares one execution for a self-join or `pl.concat([plan, plan])`
  instead of running the source twice, concurrently. A run that writes
  keeps both, because dropping a duplicate node drops its effects too.

### Documentation

- **Time order is no longer stated as a requirement of the library.** The
  README's opening sentence said the library was for "rows that arrive in
  time order"; it is for data that never fits in memory at once, and time
  order matters only for a local, rolling fit through a decay. The same
  claim is corrected in three other places, and `llms.txt` now gives the
  current Polars range.
- **The README is rewritten for one stated reader** (task 68): someone who
  knows statistics, a little Polars and time-ordered data, and nothing about
  the internals of Polars or this project. The introduction now defines the
  four words the rest of the document uses (spec, model bank, stream and
  chunk, state) before using them; the stream section is split into one
  subsection per concept, with the clock, decay, groups, weights and warm-up
  each defined where it lives; and prose that described what a parameter
  does, what comes out of a structure or how to call something is a code
  example with comments, all of which run in the test suite. `po.run` and
  the `online` command line moved to their own guide, `docs/RUNNER.md`. The
  rules the rewrite followed are `docs/WRITING.md`, each drawn from a
  reported problem in `docs/PHRASING.md`, the log the rewrite was worked
  from.

## [0.5.0] — tagged 2026-09-10, never published

The `v0.5.0` tag exists and points at the same package as 0.5.1, but its
release run withheld the publish, and nothing reached PyPI or GitHub
Releases. The cause was in the new release-time Polars check, not the
package: the blocking leg passed `--prerelease=disallow`, which applies to
every dependency, and the docs group's `furo` needs a beta of
`sphinx-basic-ng`, so the environment could not be resolved and the check
failed before running a test. The check failing closed is what it was
built to do. 0.5.1 is the same change with that flag removed.

## [0.4.1] — 2026-09-09

A patch: `hit_rate` on a `sgd(loss="logistic")` or `ftrl` fit reads a
different number now, because the old one was wrong (1.0 on every such fit,
whatever it had learned), not because a definition changed underneath a
working number. `pred`, `coef` and every other output are untouched; nothing
here needed a `SCHEMA_VERSION` bump or a spec change. A regression fit's
`hit_rate` is unaffected to the bit.

### Fixed

- **`hit_rate` read 1.0 for every `sgd(loss="logistic")` or `ftrl` fit**
  (task 76). It was scored as `pred.signum() == y.signum()` with `y == 0`
  rows dropped; a logistic fit's `pred` is a probability and its `y` a 0/1
  label, both positive by construction, so the sign test always agreed and
  `hit_rate` reported a perfect score whatever the fit did — 1.0 on a fit
  trained on pure-noise features, measured. It is now accuracy at a 0.5
  threshold on a fit whose declared loss is `logistic`, gated on that
  declared loss rather than on a row's values, with every row scoring
  (`y == 0` is one of the two classes, not the excluded case it is for a
  signed target). A regression fit's `hit_rate` is unchanged to the bit.
  `r2` and `ic` were already correct there under different names — the
  Brier skill score and the point-biserial correlation — and are now
  documented as such (`emit_metrics`'s docstring, the README's output
  table). `po.eval.metrics`/`rolling_metrics`/`sums` take the same reading
  behind a new `binary` keyword, and `metrics`/`rolling_metrics` add a
  `log_loss` column when it is set; there is no streaming log loss, since
  an EW accumulator would put a `ln` result into a model's persisted state
  (`docs/PLAN.md` §11a's B4 rule). No `SCHEMA_VERSION` bump: `SlotMetrics`'s
  fields are unchanged in shape, only in what the caller's own choice of
  loss makes them count.

## [0.4.0] — 2026-09-08

**`sgd` changes its numbers in this release, and nothing else does.** Both
entries under Changed are `sgd`: a defect in `scale_features` that made it
diverge wherever there are few rows per feature, and a rewrite of the inner
loop that costs 2 ns per feature per row instead of 13–14 and sums the dot
product in a different order. A fit that uses `scale_features=True` moves
materially; every other `sgd` fit moves at rounding level; no other model's
numbers, no output field and no state file changes. The chunk gather that
carries half of that speed-up is in the bank's shared path, so every model
now reads its rows from rewritten code — the values are identical, and
`ewridge` at `k = 10,000` runs at the same 52 rows/second it did before,
because its own Gram update dominates the gather.

**A minor, not a patch, and nothing here breaks.** The API, the spec keys
and `SCHEMA_VERSION` 6 are as they were, and a 0.3.x state file loads into
this build and continues to the bit. The version moved anyway because the
rule this project states — while pre-1.0, the minor carries the changes a
user has to read about before upgrading — is about what a reader must know,
and "the same `sgd` fit now returns different numbers" is that, whether or
not a signature moved. So, upgrading from 0.3.x: an `sgd` prediction you
stored will not reproduce, and a `scale_features=True` fit is a different
fit, which is the point of the fix. Everything else, including every state
file you hold, is untouched.

### Changed

- **`sgd` with `scale_features=True` standardises each row against moments
  that include the row** (task 74) — sklearn's `partial_fit` then
  `transform` — instead of the moments from before it. **Every
  `scale_features=True` prediction changes**; nothing else does (`n_eff`,
  chunk invariance, the state after a row, the cost, and every unscaled fit
  are as they were; no `SCHEMA_VERSION` bump). Why: standardising against
  moments from before the row is stricter than the leakage rule requires —
  the rule is about the target, and the row's features are known at
  prediction time — and it is unstable wherever there are few rows per
  feature. A variance estimate a few rows old can be tiny by chance, the
  standardised value huge, and one step with `lr · |z|² > 2` throws a
  coefficient where the next hundred rows do not bring it back: R² −6.9
  over rows 25–50 of 200-row groups (`k = 20`, `learning_rate=0.01`),
  where `SGDRegressor` at the same rate scored 0.45. With the row inside
  the moments a standardised value is bounded by `sqrt(n_eff)`, the first
  row of a stream standardises to zero (only the intercept learns from
  it), and the same table now reads 0.43 / 0.70 / 0.91 over rows 25–50 /
  50–100 / 100–200 against sklearn's 0.45 / 0.71 / 0.91 — what is left is
  that sklearn's loop standardises the row it *predicts* against the
  moments before it, where here the one standardised row serves the
  prediction and the step, and it closes as the fit converges. The wide fit was
  the same defect on every row — 2,000 rows over 10,000 features is 0.2
  rows per feature — and its predictions now correlate 0.999997 with
  `SGDRegressor`'s at `k = 10,000` (was 0.52) and 0.99999995 at `k = 1,000`
  (was 0.978), R² 0.8512 against `scale_features=False`'s 0.8516. The row
  is admitted at *unit* weight for the standardisation (its actual weight
  still governs what it teaches), which is what keeps `predict(x)` equal to
  the `pred` the next `step` reports, weight or no weight, and makes the
  moments read on a unit-weight stream bit-identical to the ones the update
  leaves behind. `kalman`'s `standardize` still reads the moments from
  before the row and is unchanged: its gain is normalised by `z'Pz + σ²`,
  so a large standardised value moves the state by a bounded amount.

- **`sgd` costs 2 ns per feature per row instead of 13–14** (task 75,
  `docs/PERFORMANCE.md` §20). At `k = 10,000` a bank runs it at 45,000
  rows/second against 13,000 before, and against 20,007 for `SGDRegressor`
  in batches of 1,000 — with every prediction made from the state as it
  stands; at `k = 1,000`, 486,000 against 217,000, and at `k = 20`, 10.8M
  against 8.9M. None of it was the arithmetic. The bank walked a row across
  one `Vec` per feature at a stride of the chunk height — a page per feature
  per row — and now gathers each chunk into a row-major buffer once (a
  tiled transpose, 3 µs per row at that width); the step indexed three
  nested `Vec`s per feature with a bounds check on each and chose its
  schedule per feature, and now runs zipped slices with the rate raised
  once per row; the accept check no longer short-circuits, so it
  vectorises; and the dot product is summed in eight interleaved partial
  sums instead of one chain of dependent additions. **That last one changes
  bits**: the order is fixed by the code and the same on every platform,
  but it is not 0.3.1's, so `sgd` predictions differ from the last
  release's at rounding level — 1e-16 relative on squared loss, 1.5e-11 at
  worst across 48 configurations of a 400-row stress stream (Huber at a
  constant rate, whose clipped gradient lets a perturbation persist). The
  pipeline goldens (1e-12) pass unchanged; everything else in the change
  is bit-identical, checked by a signature of every `step` and `predict`
  over those 48 configurations. What is left at `k = 10,000` is measured
  in §20: the data summary is now the largest item (7 of 20 µs per row),
  and the frame hand-off costs about 8 ms per call at 10,000 columns
  (pyo3-polars' per-Series export), so feed wide frames in tall chunks.

### Documentation

- **"How does this compare to scikit-learn?" now has an answer** (task 72).
  A new README section, "Against scikit-learn", and `docs/PERFORMANCE.md` §19
  behind it, measured with `scripts/sklearn_comparison.py` — scikit-learn is
  not a dependency; the script says so and exits if it is missing. Accuracy
  is not the difference: at `k = 20` every contender reaches the stream's
  noise ceiling, and under drifting coefficients the row-by-row contenders
  are within 0.001 of each other (`ewridge` 0.9907, `sgd` 0.9906,
  `SGDRegressor` 0.9899 row by row and 0.9820 in batches of 1,000 — the
  batched gap is staleness, not algorithm). The throughput difference is a
  difference in semantics: sklearn's mini-batch form predicts from a state
  up to 999 rows stale, and asked for a prediction from the state as it
  stands it runs at 3,400 rows/second against `sgd`'s 6M. Where the Gram
  wins is short histories: 500 groups of 200 rows, `ewridge` is at 0.97 by
  rows 25–50 of a group where the best `SGDRegressor` setting is at 0.73.
  Where sklearn wins is stated with the same numbers: at `k = 10,000`,
  `ewridge` carries 860 MB of state and 53 rows/second against 0.31 MB and
  18,980 for `SGDRegressor` in batches of 1,000 — and that cost is the
  matrix, not the output: `coef_every=1` costs 2.5%, the solve 0.2%, and
  the rank-1 update moves 800 MB per row at 85 GB/s, near memory
  bandwidth; `gram_block_rows=1024` buys 7.2×. `sgd` is the `O(k)` answer,
  and at the same semantics (row by row) it is faster than `SGDRegressor`
  (3× when the section was written, 15× after task 75 above) and agrees
  with it (correlation 0.9954). The wide table's
  learning rate is now `0.2 / k` for both libraries — an LMS step is
  stable only while `eta · |z|² < 2` and `|z|² ≈ k` — and it carries an R²
  column; its first version had timed two diverged fits at 0.01.

### Fixed

- **`sgd` with `scale_features=True` was wrong wherever there are few rows
  per feature** — the start of every group, measured at R² −6.9 over rows
  25–50 of 200-row groups where `ewridge` scores 0.97, and every row of a
  wide fit: at `k = 10,000` its predictions correlated 0.52 with
  `SGDRegressor`'s at the same step where `scale_features=False` correlates
  0.9954 (task 72's comparison found both, in 0.3.1). Fixed by the change
  of scaler order above (task 74); the tests that would have caught it are
  in `tests/test_sgd.py::TestFeatureScaling` — the short-history table,
  a numpy replica agreeing to 1e-12, and the wide case.

## [0.3.1] — 2026-09-08

### Added

- **`ewridge(gram_block_rows=)`** (task 71, `docs/ENHANCEMENTS.md` E51).
  Hold that many rows back and bring the `k×k` co-moment matrix up to date
  once per block with one matrix product, instead of one rank-one update
  per row. Off by default. Measured single-threaded on the whole step with a
  256-row block: 5.1× at 256 features, 6.6× at 1,000, 5.9× at 2,000; a
  solve every 512 rows dilutes that to about 4×, since the solve costs the
  same either way (`docs/PERFORMANCE.md` §18).

  What does not change: `n_eff`, the timing of every prediction, chunk
  invariance (the block is merged when it fills, before a solve and before a
  session blend — never because a chunk ended) and `gram()`, which reports
  the held rows without merging them. What does: the merged matrix is a
  floating-point sum in a different order, so a blocked fit agrees with the
  per-row fit to rounding rather than to the bit, and its last bits can
  differ between CPUs. Refused with `window`, with a solve every row
  (`solve_every <= 0` or `max_rows_between_solves <= 1` — the default
  `solve_every` is `halflife / 50`, so `lam` and `halflife=inf` need it
  set) and where the held rows would exceed 256 MiB. A state file written by
  0.3.0 loads with the block off; one written by this build with an `ewridge`
  spec, blocked or not, does not load on 0.3.0, since a saved spec carries
  the new key and a spec denies unknown fields.

  The held rows travel in the state file, so a bank saved mid-block resumes
  on the same block boundary. `SCHEMA_VERSION` stays 6: the field is
  additive, so an `ewridge` state written before it loads with blocking
  off, continues to the bit, and re-saves carrying the new key; a bank with
  no `ewridge` re-saves byte for byte.

### Documentation

- **The clock does not have to be a time, and now the headline says so.** It
  is any monotone numeric column, so sorting a frame by one of its features
  and clocking on that feature makes `halflife` a bandwidth in the feature's
  units: each row is fit on the rows before it under weight
  `0.5 ** (Δx / halflife)`, which is a local linear regression with a
  one-sided exponential kernel. Named in the README's opening list, with an
  example and the caveats under "A clock that is not time", and in
  `polars_online.spec` and `llms.txt`. Measured, not asserted: the fit agrees
  to 1e-12 with a kernel-weighted least squares recomputed from scratch at
  every row (`tests/test_ewridge.py`), and on `sin(x)` at a bandwidth of 0.25
  it sits 0.08 from the truth where the best straight line sits 0.39. The
  kernel is one-sided because a row is scored before it is learned from, so a
  curve is followed with the lag that implies.

## [0.3.0] — 2026-09-08

### Added

- **A state file describes itself, and can be read without this library**
  (task 69, `docs/ENHANCEMENTS.md` E68). `ModelBank.load("bank.state")`
  needs no `specs=`: `bank.specs` returns every spec as the dict its builder
  made, and with `groups()`, `output_fields()`, `rows_seen()` and the four
  diagnostic tables that is enough to walk a bank nothing has described.
  New `to_json()` and `save_json()` write the whole state as JSON — an
  export, not a second state format, since `load` reads msgpack and only
  msgpack.

  The export is faithful including the values JSON has no literal for.
  `serde_json` writes `NaN` and `±inf` as `null` and says nothing about it,
  and `halflife=inf` (no decay) puts an infinity in every stream's `decay`,
  so a naive export was silently wrong on ordinary banks. Non-finite floats
  are written as `"nan"`, `"inf"` and `"-inf"` — the spelling a spec's
  `halflife` already used — and every export is read back and checked against
  the state before it is returned. **State files are unchanged**: the tagging
  keys on `is_human_readable()`, which msgpack reports as `false`, and
  `crates/online-core/tests/state_encoding.rs` pins an annotated field to the
  same bytes as a bare `f64`.

### Changed

- **`ModelBank.coef()` takes every spec by default, and leads with a `spec`
  column.** It was the only one of the four read accessors that required a
  spec name, while `last_row()`, `summary()` and `describe()` all defaulted
  to the whole bank — so a bank's coefficients could not be read in one call,
  and frames from different banks did not stack. `bank.coef()` now sweeps,
  skipping a spec that has no coefficients (an `ew_cov` emits statistics, a
  `seqtest` emits evidence); naming one of those still raises, because then
  the question was about that spec. **Breaking**: every `coef()` frame gains
  `spec` as its first column, so code selecting by position needs a look.
- **`ModelBank.specs` is read-only and returns a copy** (task 69). It was an
  attribute set in `__init__`, which kept it out of `tests/api_surface.txt`
  (the snapshot walks the class) and let an in-place edit desynchronise the
  Python view from the Rust bank — `bank.specs[0]["features"] = [...]` left
  `coef()` labelling coefficients from a spec the bank was not running.
  Assigning to it now raises `AttributeError`, and mutating what it returns
  changes nothing. Reading it is unaffected.

- **`marginal(bins=)`: the nonlinear view** (task 66,
  `docs/ENHANCEMENTS.md` E67). Every statistic `marginal` reported was
  linear, and a feature can be strongly related to a target with `corr` at
  zero — a threshold, a V, a saturation. `bins=16` adds the target's weight,
  mean and variance inside each of the feature's bins (its response curve)
  and the best single cut of it: `split_gain`, the fraction of the target's
  variance that cut removes, `split_at`, and `split_gain_t`. That is a
  regression stump's gain for every pair in the pass that gives it `corr`,
  at `O(bins)` of state per pair. Edges come from `bin_edges` outright or are
  learned from the first `bin_warm_rows` rows under `bin_rule`; the warm-up
  rows are held and replayed, not spent, so the histogram is what it would
  have been had the edges been known first. Each bin's moments are kept in
  Welford form, so a target far from zero keeps its variance; the quantile
  rule is weighted and keeps a point mass — an indicator's zero — in a bin
  of its own; `bin_edges` is refused beside the learned kind's knobs
  (task 67 review).
- **`marginal(lags=)` and `n_serial`** (task 65, `docs/ENHANCEMENTS.md` E66).
  `t` is built on `n_kish`, which says nothing about serial dependence: on a
  smooth stream `t` reports evidence that is not there. `lags=` accumulates
  the pair's moments at those lags — bit-identical to `ew_cov(lags=)` — and
  `serial_rule` turns them into Bartlett's correction, reported as
  `n_serial` and `t_serial` beside `t`, with `phi_x`, `phi_y` and the four
  `lagcorr_*` columns — `ew_cov`'s `lagcorr` numbers exactly, unclamped. On
  two independent AR(1) series at `phi = 0.9` and `0.8`: `t = 2.39`,
  `t_serial = 1.03`. A row where the target is missing holds the lag moments
  as it holds the pair's (task 67 review).
- **Closed-group rows carry the lag and bin blocks** (task 67). A closing
  `marginal` with `lags=` or `bins=` adds `pair_lagcorr_*`, `pair_n_serial`,
  `pair_t_serial`, `pair_phi_*`, `pair_bin_*` and `pair_split_*` beside the
  other `pair_*` columns — lists of lists where `marginal()` has a list per
  pair — under `marginal()`'s null rule (NaN is null, `±inf` stays). A bank
  none of whose closing marginals asked has no such columns.
- **`docs/OUTPUTS.md`: what every model writes** (task 64,
  `docs/ENHANCEMENTS.md` E65). Every spec adds one struct column and nothing
  said what was in it per model — `hmm`'s `state` and `p1_<j>`, `bocpd`'s
  `run_mode`, `micro`'s `outlier`, `seqtest`'s two e-processes. The field
  lists are generated from `po.spec.output_fields` and the meanings written
  once per field stem, so the document cannot drift from the code;
  `tests/test_outputs_doc.py` regenerates and compares, and the generator
  writes `**undocumented**` for a stem it has no meaning for so a new field
  cannot ship silently. Each model's README section links to its section.
- **`llms.txt`** (task 59), the [llmstxt.org](https://llmstxt.org) map for
  coding agents: the two streaming surfaces in full, the rules an assistant
  otherwise guesses wrong (predict-before-update, chunk invariance, decay on
  the clock, `n_eff` as a weight, the in-memory expression form), and links
  into the README, the API reference and the guides. It is served from the
  repo root and, through `html_extra_path`, from the API reference, so one
  file cannot drift from itself; `tests/test_llms_txt.py` holds its model
  names, links and README anchors to the registry and the tree.
- **The README links into the API reference** (task 60). Each model in the
  table of models links to its builder's entry, with a `math` link beside it
  to the section below; the first mention of `ModelBank`, `po.run`,
  `lf.online.fit_predict`, `po.eval`, `po.gram`, `po.corr`, `po.prep`,
  `po.sim`, the spec helpers and `InMemoryExpressionWarning` links to its
  entry. `tests/test_api_links.py` resolves every such link against the
  Python objects the anchors are built from, so a rename breaks the suite
  rather than the page.

### Added

- **`window` on `ew_cov`: an exponentially weighted accumulator with a hard
  cutoff** (task 63b, `docs/PLAN.md` §13). `window=w`, in clock units, makes
  a row older than `w` contribute *nothing*, where the exponential weight
  alone leaves 12.5% of it at three halflives. Inside the window the weights
  are still exponential — it is not a rolling flat mean. It is exact rather
  than approximate: an EW sum contains its own past, so everything at or
  before a time `u` is `lam^(t-u)` times the accumulator as it stood then,
  and subtracting that leaves precisely the rest. The model keeps a ring of
  snapshots to do it, which is the one place here where memory grows with a
  window rather than with the state; `window_every` trades boundary
  tightness for memory, and only ever shortens the effective window. `n_eff`
  becomes the weight inside the window, a clock gap longer than `window`
  reports nulls rather than stale numbers, and the sixteen models whose
  state is not a sum of per-row contributions refuse the keyword by name.
  Verified against a direct windowed sum in Rust and against polars'
  `rolling().agg()` in Python.
- **`window` on `ewridge`: a regression with a hard cutoff** (task 63c). The
  Gram, the per-target cross-moments and the residual variance are truncated
  by the same identity and the fit is solved from the result, so no row older
  than the window is in the coefficients at all — the thing an exponential
  decay cannot promise and polars has no primitive for. `n_eff`, `sigma` and
  `resid_z` come from the window too. On a stream whose slope flips from +3
  to −2, a halflife of 60 still reports −0.53 a hundred rows later where
  `window=40` reports −1.999996. Refused with `ridge_decay` and
  `session_shrink`, where the identity does not hold. Verified against a
  direct weighted-least-squares solve over the in-window rows, in Rust and
  again in Python against numpy.
- **`window` on `lasso`** (task 63c). The same cutoff on the path, and the
  selection error is truncated with it, so the chosen `lambda` fits the rows
  the coefficients see. A window can therefore change the *support*, not just
  the magnitudes: on a stream where `x0` drives the first regime and `x1` the
  second, a 40-unit window takes `x0` to exactly zero where the decayed path
  still carries 0.67 of it.
- **`window` on `marginal`** (task 63c). The pairwise screen gains the same
  cutoff, applied at the readout: every moment a pair is built from is
  truncated, so `corr`, `beta` and `t` describe the window. Two regimes of
  opposite sign cancel over a long history, so an unwindowed screen can
  report no relationship where there is a strong one — 0.0006 against −0.99
  on the same stream.
- **`window` on `ew_class`** (task 63c), which completes the set. Each class's
  moments get the cutoff, so a classifier can follow class means that move:
  on a stream where they swap halfway, the windowed labels are right 100% of
  the time over the last hundred rows and the unwindowed ones are at chance.
  It is also the one that costs — `covariance="full"` pays a factorization
  per class per row, because a truncated covariance moves every row where a
  decayed one does not. `docs/PERFORMANCE.md` §16 measures every model's
  window, and measures that the windowless path is unchanged.

### Changed

- **State files written before 2026-09-07 no longer load, deliberately.**
  `MIN_SCHEMA_VERSION` is 6. The naming pass renamed spec keys with no
  aliases, and a spec denies unknown fields, so an older file names fields no
  builder has; rejecting it on the version is a better error than failing on
  a field name nobody chose. The fixtures that proved older files load are
  gone, as are the schema-1 and schema-2 conversion paths in `rls`, `sgd` and
  `kalman`. A state saved by 0.2.0 has to be refit. This is a one-off taken
  while the library is days old — the rule that a loader is kept for the
  previous version applies from here.
- **Six more names** (the 2026-09-07 pass): `rcov`'s `n_max` → `block_rows`
  and `h_max` → `max_bandwidth`, `kmeans`'s `sm_every` →
  `split_merge_every`, the ridge family's `coef0` → `coef_prior`, `rcov`'s
  `preavg_ticks` → `preavg_rows`, and `refresh_time`'s `n_ticks_<s>` column →
  `n_obs_<s>`. "Ticks" assumed market data in a library whose clock is
  anything monotone; `coef0` read as "the coefficient of feature 0" when it
  is the prior mean. Names that appear as symbols in a quoted formula
  (`theta`, `jitter`, `q`, `p0`, `c`, `beta_mu`) were left alone.
- **`SCHEMA_VERSION` is 6.** The `ew_cov` spec gained two keys, and spec
  fields serialize their nulls, so every spec's bytes moved — the same
  reason 4 and 5 bumped. A schema-5 file loads, continues to the bit and
  re-saves as 6.
- **Three spec keywords renamed** (task 63a), because a fourth meaning of
  `window` is about to be added and the existing two did not describe what
  they do. `rcov`'s `window` is the pre-averaging length, counted in rows,
  and is now `preavg_rows`. `corrchange`'s `horizon` and `window` were one concept —
  rows per comparison block — under two names, one required by each kind,
  and are now a single `span_rows` required by both (at least 8 for
  `"monitor"`, at least 3 for `"window"`). `bocpd`'s `truncate` is a
  probability floor for pruning the run-length vector and is now
  `prune_below`. No alias is accepted, and a bank file written by 0.2.0 does
  not load: a saved state carries its specs verbatim, so the old bytes name
  fields no builder has.

### Fixed

- **The FFI leak test no longer mistakes an allocator step for a leak**
  (task 61). `assert_plateaus` compared the first and last of its
  post-warm-up marks, so one late 3.4 MB jump read as a 14.5 KB/iter slope
  and failed a `main` run whose previous run on the same tree was green. The
  statistic is now the median block-to-block gap over five blocks, which
  reads that trace as 0.0 KB/iter and a sustained leak unchanged. Test-only.

### Fixed

- **A spec float lost its last bit crossing into Rust.** `serde_json`'s
  default parser is fast rather than correctly rounded, so
  `-0.41215148805088475` arrived as `-0.4121514880508847`. Nothing for a
  halflife; everything for a `marginal` bin edge, since edges read back from
  an earlier run are data values and a row sitting exactly on one is the
  common case rather than a coincidence. The crate's `float_roundtrip`
  feature is now enabled in all three crates.
- **`window_every` without `window` was accepted by three models**
  (`marginal`, `lasso`, `ew_class`), where `ew_cov` and `ewridge` had always
  refused it. In the compact msgpack encoding it also decoded silently as
  `window = 1`.
- **Two optional fields in a row broke the compact msgpack encoding.** That
  encoding writes a struct as a bare array, so a skipped field slides
  everything after it; `marginal`'s lag moments had been added in front of
  its window. State files use the named encoding and were unaffected, which
  is exactly why `crates/online-core/tests/state_encoding.rs` now sweeps
  every combination in both.

## [0.2.0] — 2026-09-06

A large release. 0.1.x had eleven regression and filtering models; 0.2.0
adds **ten more** — `kmeans` and `micro` (clustering), `ew_class`
(classification), `seqtest` (a sequential test of a sign), `marginal`
(per-pair moments), `deco` (equicorrelation), `rcov` (a block's realised
covariance), `hmm` (a hidden Markov filter), `corrchange` (is the
correlation structure constant?) and `bocpd` (how long has this regime
lasted?) — and the options and plumbing they wanted: conformal intervals,
the Mahalanobis distance, EW-PCA and lagged co-moments on `ew_cov`,
constrained coefficients on `sgd` and `pa`, coefficient reversion on
`kalman`, a label delay, refresh-time sampling, a closed-group queue that
bounds a bank over an unbounded key space, and the `po.corr` and `po.sim`
modules beside them. Throughput went up 2.0–2.8× and thread scaling from
3.2× to 6.2× on ten cores.

Every model keeps the contract — predict before update, O(state) memory,
chunk invariance, `n_eff` before the row — and **every number the 0.1
models produce is unchanged**, checked by a bit-identical dump over twelve
configurations. State files are written in schema 5, and files from 0.1.x
load and continue. One breaking change: a residual diagnostic set on a
model that has no predictions is refused by name, where `ew_cov` used to
accept it silently.

### Added

- **`docs/REGIMES.md`: what the detectors actually find**, generated by
  `scripts/regime_experiments.py` (committed, not regenerated by the gate —
  the size study alone is 2000 replications). Five experiments: `hmm`'s
  recovery of a covariance regime and why the default seeding cannot find
  one; `corrchange`'s size and power against Wied, Krämer & Dehling's tables
  at 2000 and 1000 replications, across three data-generating processes,
  which found that **their tables cannot be matched without knowing which
  bivariate `t₅` they used** — the two readings straddle their figure, and
  the gate's test now pins the nominal level on Gaussian pairs instead; the
  denominator behind both, `D̂`, against the closed-form value it estimates,
  which is exact on Gaussian pairs and 15 % low with a 40 % scatter on a
  `t₅` and so accounts for the whole difference;
  `corrchange(window)` against `bocpd` on the same break, which is 29 rows
  and 51 flags per 1000 quiet rows against 98 rows and 0.68; and the Epps
  curve, where refresh time recovers 0.54 of a true 0.8 and the lag
  inversion 0.76.

- **The five new models are in the benchmark and in
  `docs/PERFORMANCE.md` §15**, so the README's throughput table covers them
  and a regression shows up where every other model's would. Two findings
  worth the knob names: **`bocpd`'s `truncate` is not a tuning knob, it is
  what makes the model finite** — the run vector grows by one entry a row,
  so `truncate = 0` is `O(rows²)` (measured halving at 5k/10k/20k rows) and
  the default `1e-6` is flat in the length of the stream; and **`rcov` pays
  at the close, not per row** — `plain` is free, the BNHLS `kernel` is
  `O(n·H)` with the automatic bandwidth, so 0.21 ms, 2.7 ms and 44 ms per
  close at 1,000, 5,000 and 20,000 rows a block, and `preavg` is the one to
  reach for on very large blocks.

- **`bocpd`: how long has this regime lasted?** (`docs/ENHANCEMENTS.md`
  E61, task 55). Adams & MacKay's run-length posterior, their Algorithm 1 in
  log space, with a normal-inverse-gamma (`emission="diag"`) or
  normal-inverse-Wishart (`"gaussian"`) conjugate emission per run, a
  hazard that can be read per row from a column, and `truncate` / `max_run`
  to keep the run vector finite. Outputs `p_change`, `run_mode`, `run_mean`,
  `pred_<f>`, `logscore` and `n_eff`.

  **`run_mode` is the answer and `p_change` is the alarm.** `run_mode` is
  the pre-row run length, so `t − run_mode` is the row the current run began
  on; a variance step, a mean shift and a correlation break are all dated to
  the right row within a few rows of it. `p_change` is `P(r ≤ 1)` — `P(r =
  0)` is exactly the hazard on every row whatever the data, which is why it
  is not what is reported — and it is a per-row likelihood ratio, so it
  spikes on a variance step (0.83) and barely lifts on a mean shift.

  `emission="robust"` tempers each row by `(π(x)/π(mode))**robust_beta` in
  what the run learns *and* in the message it passes, so one 20-σ row is a
  non-event where the plain model calls it a changepoint and swallows it.
  The knob trades against detection: above about 0.2 nothing is ever found,
  and the default is 0.1. This is a β-power weighting and **not** the
  diffusion-score-matching posterior of Altamirano, Briol & Knoblauch
  (2023), which stays a follow-up.

- **`corrchange`: has the correlation structure changed?**
  (`docs/ENHANCEMENTS.md` E59, task 54). Two tests. `kind="monitor"` is Wied,
  Krämer & Dehling's **closed-sample** constancy test run over consecutive
  spans of `horizon` rows, with the paper's `D̂`, Kolmogorov critical values
  computed from the series (1.3581 at 5%), and Bonferroni over the pairs —
  and its size and power held to the paper's own Tables 1 and 2 rather than
  to numbers this implementation produced. `scalar=True` runs the same CUSUM
  on `deco`'s equicorrelation, which is one statistic however many columns.
  `kind="window"` measures how big a change is between two adjacent windows,
  against a fixed threshold or a **permutation** quantile — not a sign-flip
  null, which leaves every correlation exactly where it was.

  The sequential form with a boundary function is Wied & Galeano (2013),
  unread here; it stays an ENHANCEMENTS §10 follow-up.

- **`hmm`: a Gaussian hidden Markov model, filtered online**
  (`docs/ENHANCEMENTS.md` E60, task 53). `ew_class` without the labels:
  Hamilton's filter one row at a time, each state's accumulator taking the
  row at its responsibility, and the transition matrix learned from the
  **filtered joint of consecutive states** with a Dirichlet prior. Outputs
  `p_<k>`, `p1_<k>`, `state`, `loglik` and `n_eff`, all read before the row,
  with the state means as `coef`. `exog_tvtp` drives the matrix from a
  column instead.

  Measured: on two-dimensional blobs 1.5 apart, a memoryless nearest-centre
  rule *given the true centres* is 85% right and the filter is 99%.

- **`po.sim.regimes`: a seeded simulator for correlation regimes**
  (`docs/ENHANCEMENTS.md` E64, task 52). A stream of `m` series whose
  correlation changes by regime, with the parts that make a detector's job
  real: asynchronous observation, microstructure noise on the level, AR(1)
  returns, a volatility that moves with the state, a cycle_profile pattern and a
  volume clock. Returns `rows` (levels, clock, session, volume),
  `truth_rows` and `truth_blocks`, all byte-identical for a given seed.

- **`po.corr`: correlation matrices, read and repaired**
  (`docs/ENHANCEMENTS.md` E62, task 51). `po.gram`'s complement, in the same
  style: numpy only, pure functions, one longhand check each. Higham's
  nearest correlation matrix (his Algorithm 3.3 with Dykstra's correction,
  and his own published examples as the test — the matrices, the distances,
  the null vector, the rank and the iteration count), Ledoit–Wolf shrinkage
  with the optimal intensity from the rows, Fisher's transform, the
  equicorrelation trio that pins `deco` offline, the absorption ratio,
  spectral and block summaries, the Marchenko–Pastur edges and density,
  `signal_share`, three forecast losses, the Epps inversion over `ew_cov`'s
  lagged co-moments, and `fisher_se` with its AR(1) inflation.

- **`rcov`: a block's realised covariance, robust to microstructure noise**
  (`docs/ENHANCEMENTS.md` E57, task 50). Three estimators over a group's
  returns: `plain` (`Σ x x'`, which equals `n` times an `ew_cov(lam=1)`'s
  uncentred second moment at close, to the bit), `kernel` (the multivariate
  realised kernel of Barndorff-Nielsen, Hansen, Lunde & Shephard, Parzen
  weights and jittered end points) and `preavg` (Christensen, Kinnebrock &
  Podolskij's modulated realised covariance). No decay and nothing per row
  but `n_eff`: the value is the block, emitted in the `group_close` row with
  `rcov`, `rcorr`, `rcov_n`, `bandwidth_used`, `omega2`, `iv_sparse`, `iq`
  and `psd_repaired`.

  Nothing reads a future row: the jittered *end* point is formed at close
  from observations already in state, and a product enters `Γ̂_h` only once
  both legs are final. `weight` is 0 or 1 only, and the block is the same
  from one chunk or a thousand.

- **`po.prep.refresh_time`: asynchronous series on a common grid**
  (`docs/ENHANCEMENTS.md` E58, task 49). Barndorff-Nielsen, Hansen, Lunde &
  Shephard's refresh-time rule — a grid point wherever every series has
  ticked at least once since the last one, each carrying its last observed
  value — as a Rust operator (`crates/online-polars/src/refresh.rs`) wrapped
  as a lazy source, so a tick stream too long to hold still is. Long input,
  one row per grid point out: `time_refresh`, `<s>_value`, `n_obs_<s>` and
  `retained_fraction`, plus `by` and `keep` columns. `pairs=True` runs an
  independent two-series grid per pair, which keeps far more of the data
  when one series is slow. Nothing is interpolated, and the grid is the same
  from one chunk or a thousand.

- **Lagged co-moments on `ew_cov`** (`docs/ENHANCEMENTS.md` E56, tasks 47
  and 48). `ew_cov(lags=[1, 2, 5])` accumulates `E_w[d_t d'_{t−ℓ}]` beside
  the contemporaneous co-moments, with both deviations against the mean
  before the row — the same `a` and `b`, so lag 0 would be `comoments`
  exactly. Read them from `gram()` as `lags` and `lag_comoments`, from a
  closed group's row, or as `lagcorr_<a>_<b>_l<ℓ>` output fields by adding
  `"lagcorr"` to `stats` (both orientations: a lagged matrix is not
  symmetric).

  With it, the rule for anything a model keeps *by row*: the ring is emptied
  on a session change and on a clock gap beyond `max_dclock`, through a new
  `OnlineModel::clear_lags` the stream calls at those two events and no
  others. `po.gram.merge` reports no lags for a pooled Gram — the pairings
  across a part boundary are what no part holds — and `subset` slices them.

- **`deco`: one correlation for the whole matrix** (`docs/ENHANCEMENTS.md`
  E55, task 46). Engle & Kelly's dynamic equicorrelation as an
  `OnlineModel`, `O(m)` a row where a full correlation matrix is `O(m²)`.
  The row is standardised against an `EwDiag`'s pre-row moments and their
  Lemma 2.3 closed form gives the row's estimate; the level follows either
  `EwCov`'s mean recursion (`dynamics="ew"`) or the paper's eq. 21 with
  correlation targeting (`"linear"`, with `alpha` and `beta`). `blocks`
  estimates one number per named block and one per pair of blocks instead,
  through a `K x K` Woodbury factorization rather than an `n x n` one.
  Outputs `u`, `rho`, `loglik` and `n_eff`, all read before the row.

  Two facts the docstring and the README state plainly, because both are
  easy to assume otherwise: `u` is a **downward biased** estimate of the
  equicorrelation (the paper's own remark — `E[u]` is about 0.20 for a true
  0.30 at six columns), and `rho` is **not** an `ew_cov`'s `corr` over the
  same columns, the mean of a ratio not being the ratio of means.

- **Closed-group emission: a group's accumulators, emitted when it is
  finished** (`docs/ENHANCEMENTS.md` E54, task 45). A new common parameter
  `group_close = "monotone" | "session"`. Under `"monotone"` a key smaller
  than the largest one fed so far is finished; under `"session"` a group's
  span ends where its `session` value changes. Either way the bank emits one
  row per (group, decay instance) — the accumulators exactly as they stood,
  `n_eff`, `n_kish`, the span's row counts and clock range, the Gram as list
  columns, `coef`, an `ew_cov(pca=r)`'s eigendecomposition, a `marginal`'s
  pairs — and **drops the stream**. That is what keeps a bank over an
  unbounded key space bounded.

  Read the rows with `ModelBank.closed_groups(spec=None, *, drop=True)`, or
  write them as a sidecar file with `po.run(closed_groups=path)`,
  `lf.online.fit_predict(closed_groups=path)` or `online --closed-groups
  path`; the sidecar is written once, at the end, through a temporary
  renamed into place. `po.gram.from_row(row)` reads a row back as a
  `gram()` mapping, so `po.gram.solve(po.gram.from_row(row))` is the exact
  solve on a closed group. The closed row **is** the `gram()` a driver would
  have read at that point, bit for bit — one builder makes both.

  With `output=None` that is the whole shape of an accumulate-only pass:
  read a stream that does not fit in memory, write one row per block.
  `"monotone"` refuses a chunk whose keys are out of order (naming the row),
  a null key, and a key column it cannot order; `group_close` is refused
  without `group`, with `label_delay`, and — for `"session"` — without
  `session` or with `session_gap`.

- **`label_delay`: a target that is only known later is learned later**
  (`docs/ENHANCEMENTS.md` E47, task 40). A common parameter in clock units:
  each row is scored where it sits and learned from only once the model's
  clock has moved `label_delay` further on. The prediction, `sigma`,
  `resid_z`, the metrics, drift, the conformal interval, `n_eff` and
  `min_periods` then all see only labels that had really arrived. Without
  it, a forward-looking target hands the model that much of the future
  before it predicts the rows in between: a test shows an autocorrelated
  *noise* column scoring +5% out-of-sample R^2 on a 20-row forward sum, and
  below zero once the delay is on. The buffer is per group, lives in the
  state and is saved with it; a reset drops it and a session change releases
  it. Rows still waiting when a stream ends are never learned from.
- **`polars_online.prep.embargo`**: the same thing written out as data --
  every row twice, a zero-weight prediction at `t` and a lesson at
  `t + delay`, merged back into clock order with `merge_sorted`. The native
  path is tested against it field by field and agrees to the bit, except for
  `resid_quantiles`, `emit_autocorr` and `emit_drift`, which take no row
  weight and so are fed twice by the doubled stream and once by
  `label_delay`.
- **`marginal`: every (feature, target) pair's EW moments, kept in the
  state** (`po.spec.marginal`, `docs/ENHANCEMENTS.md` E44, task 37). Per
  pair the mean, variance and covariance in `ew_cov`'s arithmetic — a
  pair's `corr` is bit-identical to an `ew_cov` over the two columns — plus
  Σw and Σw² per target, O(p·T) per row where one `ew_cov` over all the
  columns is O((p+T)²). Nothing is emitted per row but `n_eff`;
  `ModelBank.marginal(spec, group=None)` reads the pairs as a long frame
  (`group, instance, feature, target, n_eff, n_kish, mean_x, var_x,
  mean_y, var_y, cov, corr, beta, t`), null below the target's
  `min_periods` (default 3) or where undefined. A null target ages its own
  pairs and learns nothing; weights, clocks, sessions, groups, the halflife
  grid, save/load, `describe`/`summary`, the lazy plan, the runner, the CLI
  and the expression form (`n_eff` alone) as for every model. Golden
  values on every OS include the pairs at the end of the stream.
- **The Gram export is a complete sufficient statistic** (`ModelBank.gram`,
  `docs/ENHANCEMENTS.md` E45, task 38). `gram()` gains `n_kish`,
  `target_means`, `target_vars` and `target_n_kish`: the accumulators now
  track `Σw²` beside `Σw`, and `ewridge` and `lasso` keep each target's own
  mean and variance beside its cross-moments. Without `Var[y]` no residual
  variance, R², information criterion or standard error could be computed
  from a saved Gram; with it they can, and `n_kish = n_eff² / Σw²` is the
  sample size they divide by (`n_eff` counts weight, not rows). A target's
  variance is bit-identical to an `ew_cov` over that column. A state saved by
  0.1.x has none of them and reports `None` for all four for the
  rest of its life — the sums cannot be replayed, and a partial one paired
  with a whole-stream `n_eff` would be wrong by the length of the history.
  Such a state still loads, still continues to the bit, and still re-saves
  byte for byte.
- **`polars_online.gram`: the accumulators, read back** (`po.gram`,
  `docs/ENHANCEMENTS.md` E46, task 39). Eight numpy-only functions over what
  `ModelBank.gram()` returns: `merge` (Chan-Golub-LeVeque pooling of disjoint
  row sets), `subset`, `correlation`, `solve` (the model's own ridge, in
  original units, a grid of ridges from one eigendecomposition),
  `lasso_path` (the `lasso` model's coordinate descent offline, plus
  per-feature `penalty_weights`), `coef_stats` (residual variance, R^2,
  standard errors and t at the Kish sample size), `vif` and `condition`
  (Belsley's indexes and variance-decomposition proportions). `gram()` also
  names its axes now -- `columns`, with `"intercept"` first where the spec
  has one, and `targets` -- so a column can be taken by name. `solve` and
  `lasso_path` are held against the models themselves rather than against a
  second copy of the formula; they agree to a few ulps, not bit for bit,
  since the models factorize with `faer` and numpy with LAPACK.
- **`po.eval.sums` / `merge_sums` / `from_sums`: metrics for output that is
  never materialised** (`docs/ENHANCEMENTS.md` E49, task 42). `sums` reduces
  a chunk of output to ten doubles per (slot, target, key), `merge_sums` adds
  the sums of disjoint row sets, and `from_sums` gives back the same `n`,
  `r2`, `ic`, `hit_rate` and `mse` that `metrics` computes from the rows,
  plus `rmse`. A run comparing fifty slots over a billion rows keeps ten
  doubles per key instead of writing the rows out to evaluate them later.
  The sums are centred rather than raw -- weighted means and centred second
  moments, merged with a parallel-axis term -- so a target sitting on a
  large offset does not destroy the variance the way `sum(y**2) -
  sum(y)**2 / n` does. `weight=` names a column to weight rows by.
- **`ew_cov` with `stats=[]`** accumulates only (`docs/ENHANCEMENTS.md`
  E43, task 36): the spec learns the same moments, emits nothing but
  `n_eff`, and its value is its state — `ModelBank.gram()`,
  `ModelBank.describe()`, `ModelBank.summary()`. `pca` and
  `mahal_quantiles` still add their outputs without a statistic in the
  list; `stats=None` still means `["mean", "std", "corr"]`. Before this the
  constructor refused an empty list, so accumulating a Gram over a wide
  set of columns meant emitting every mean on every row.
- **A run whose product is its state writes no output** (`po.run` without
  `output`, `online --no-output`, `docs/ENHANCEMENTS.md` E50, task 43). An
  accumulator-only spec emits `n_eff` a row and nothing else; over a billion
  rows that is 8 GB of file written so it can be deleted. `output` is now
  optional and `save_state` is required in its place -- a run that writes
  nothing and saves nothing has done nothing. The state a quiet run leaves
  is byte-identical to the one a writing run leaves. `no_output=True`
  clears an `output` a config carries.
- **`targets` is optional in TOML for a model that has none**
  (`docs/ENHANCEMENTS.md` E53, task 43). `ew_cov`, `kmeans` and `micro`
  learn from no target, and `po.spec.*` fills `targets` with `features[0]`
  silently; a TOML author had to write that line by hand or be refused with
  "targets must be non-empty" -- for a model that has none. It is filled at
  parse time now, along with `drift_action = "flag"`, so a spec written in
  TOML and the same spec written in Python save byte-identical state files.
  A spec that names no features either says *that*.
- **The Gram update is 14-63% faster, bit for bit** (`docs/ENHANCEMENTS.md`
  E48, task 41). `EwCov::update` is the hottest loop in the library, run
  once a row by every Gram model. It now computes the deviations `x - m`
  once into a row scratch instead of recomputing `x[j] - m[j]` inside every
  row of the matrix, and runs its inner loop over slice iterators so the
  bounds checks that were keeping it scalar are gone. Same operations in the
  same order: every golden value is unchanged and no state file moves. -14%
  at k = 4, -63% at 16, -45% at 64, -22% to -27% from 200 up.
  E48's actual proposal -- compute one triangle and mirror it -- was
  measured and **rejected**: it is not bit-identical (the products commute
  but the association does not, so the two triangles differ in the last bit)
  and it is 49% to 107% *slower*, because the mirror store walks a new cache
  line per element. `docs/PERFORMANCE.md` section 14 has the numbers.

- **`kmeans`: exponentially weighted k-means** (`po.spec.kmeans`,
  `docs/PLAN.md` §11a, task 23). The first model with no target: every
  column of interest goes in `features`, and each row's outputs are read
  from the centres *before* the row is learned — `cluster` (the nearest
  centre's index, `i32`), `dist` and `dist2` (the distance to it and to the
  runner-up), `n_eff`, and the centres as `coef` (`k` slots `cluster{j}`,
  one coordinate per feature; `po.spec.coef_index` lays them out and
  `unnest` names them `coef_cluster0_x1`). Distances are in units of each
  feature's EW standard deviation unless `standardize=False`. Seeding waits
  for `warm_rows` learned rows, then `lloyd` (default; ten restarts of ten
  iterations over the buffer), `kmeanspp`, `farthest` or `first`; centres
  update every `update_every` rows. Halflife, clock, weight, group and
  `min_periods` mean what they mean everywhere else, and a null or
  non-finite feature row is skipped with its clock tick folded into the next
  row's, as for every other model.
- **A split–merge move for `kmeans`**, on by default (`split_merge=0.5`,
  `split_merge_every=100`, `dead_frac=0.05`; `split_merge=0` gives plain k-means).
  Rows farther from every centre than `1 + 4·sqrt(2/p)` typical radii are
  summarised per cluster instead of learned, so an outlier neither drags a
  centre nor widens its radius. At each check the two closest clusters merge
  when their centres are within `split_merge` summed radii, and a cluster
  lighter than `dead_frac·n_eff/k` is declared dead; the freed centre goes
  to the far rows' mean, once those are at least three rows and five per
  cent of the window's weight. Measured on a blob born after seeding: tail
  ARI 1.000 after `log2(1/dead_frac)` halflives (4.3 by default, 2 at
  `dead_frac=0.25`), where plain k-means never recovers (0.71–0.73). Five
  per cent uniform outliers: ARI 0.984, no spurious move. What it repairs
  and what it costs is in the README's `kmeans` section.
- Rust: `online_core::{KMeans, KMeansCfg, SeedRule, ClusterSummary,
  FeatureMoments, SplitMix64, dist2}`; `ModelState::KMeans`. The state
  schema stays at 2 — a new variant, not a new layout — so a 0.2 bank that
  holds a `kmeans` model fails to load on 0.1 at deserialization rather than
  by version.
- `tests/reference_cluster.py`: a numpy oracle for the whole recursion
  (seeding, standardization, the batch update, far rows, split–merge), held
  bit-exact by `tests/test_kmeans.py`.
- **`micro`: density-based clustering** (`po.spec.micro`, `docs/PLAN.md`
  §11a, task 24): DenStream-style micro-clusters — a decayed weight, a
  centre and a Welford radius each — with a single-linkage step over the
  established ones, so clusters can have any shape and any number, rows
  that belong to none are flagged, and a cluster can be born or die
  mid-stream. Per row, all read before the row is learned: `cluster` (the
  label of the nearest established summary, `i64`, null while there is
  none), `dist`, `micro` (the id of the summary the row goes to, or opens),
  `outlier` (`bool`: no established summary takes it), `n_clusters` and
  `n_micro` (`i32`), `n_eff`, and `coef` = one `[id, label, n, radius,
  centre…]` row per established summary, ragged (`coef_fields` is empty and
  `coef_index` refuses it, like `ew_cov`). Parameters: `eps` (required, the
  per-standardized-coordinate radius bound), `beta_mu` (3), `max_clusters`
  (200), `prune_every` (100), `macro_link` (derived from the spacing the
  summaries show unless set; `2` links only summaries that touch),
  `standardize` (true). A row is admitted where a unit row would be and
  absorbed with its full weight; the radius is capped at `eps` after every
  absorption. Measured at 20k rows against the truth: moons, rings and five
  Gaussians in twenty dimensions ARI 1.000, 5% uniform noise flagged 94% /
  real rows 0.3%, a cluster born mid-stream labelled within 200 rows. How
  to choose `eps`, and the two ways to get it wrong, are in the README's
  `micro` section.
- Rust: `online_core::{Micro, MicroCfg, MicroCluster, merged_radius2}`;
  `ModelState::Micro` (schema still 2, as for `kmeans`). Two `Source`
  kinds (`Id`, `Flag`) carry `i64` and boolean fields out of the `pred`
  buffer, so a struct can now hold `Int64` and `Boolean` columns.
- **Adaptive conformal intervals** on every regression model
  (`docs/ENHANCEMENTS.md` E36, `docs/PLAN.md` task 25): `conformal=0.9`
  adds `lo_<slot>`, `hi_<slot>` and `coverage_<slot>` — the interval
  `pred ± q` at that coverage and the exponentially weighted coverage it
  has delivered so far. `q` tracks the quantile of `|resid|` directly,
  `q ← max(0, q + conformal_rate · sigma · w · (1{|resid| > q} − α))`, so
  the long-run coverage is the level asked for with no assumption on the
  residuals and an error that shrinks like `1/T` (the bound is stated in
  `online_core::conformal` and asserted as a hard inequality on 200k-row
  streams in `tests/test_conformal.py`). The step is in units of the slot's
  `sigma` (`conformal_rate`, default 0.05), the radius starts at the
  Gaussian one `sigma · Φ⁻¹(1 − α/2)` and is null until then, and it is
  read before the row like every other output. Measured on 200k rows:
  coverage within 0.01 of 0.9 on Gaussian, Student-t(2.5),
  lognormal-scale-mixture and regime-shifting residuals, where
  `pred ± 1.645·sigma` covers 0.94–0.95 on the last three. Weighted,
  chunk-invariant, in the state file, honoured by `predict` (the radius is
  read and not moved) and restarted by a drift `reset`. Refused by name for
  `ew_cov`, `kmeans` and `micro`.
- Rust: `online_core::{Conformal, norm_ppf}` (Acklam's inverse normal, so a
  Python mirror is bit-exact); `Source::Conformal`. The state schema stays
  at 2: the per-slot trackers are a `#[serde(default)]` field, so a 0.1
  state file loads and a bank with `conformal` set starts its intervals
  from the first chunk it sees.
- **Mahalanobis distance and EW-PCA on `ew_cov`** (`docs/ENHANCEMENTS.md`
  E37 and E38, `docs/PLAN.md` task 26). `"mahal"` in `stats` adds `mahal`,
  the row's distance from the running mean in the metric of the running
  covariance, `sqrt(δᵀ (C + s·prior·I)⁻¹ δ)` with the same fading prior as
  `partial_corr` (so it needs `precision_prior`); with one column it is
  `|z|`, and on Gaussian columns `mahal²` is χ²_k — measured on 200k rows at
  k = 8, the mean of `mahal²` is within 2% of 8 and the 99% χ² threshold
  flags 1.0% ± 0.2% of rows. `mahal_quantiles=[0.5, 0.99]` adds
  `mahal_q0.5`, `mahal_q0.99`: P² quantiles of the scores so far (unweighted,
  like `resid_quantiles`), read before the row. `pca=r` adds the top `r`
  eigenpairs of the covariance as `pc<j>_var`, `pc<j>_share` (of the trace),
  `pc<j>_<feature>` (the loadings) and `pc<j>_score` (the row's projection,
  `Σ v_j,i (x_i − m_i)`); `pca_every=n` refreshes the O(k³)
  eigendecomposition every `n` learned rows once `min_periods` is reached
  and scores the rows in between on the last loadings. Loadings are signed
  for continuity — each refresh keeps `v_new · v_old ≥ 0` with the previous
  one, and the first makes the largest-magnitude entry positive — because
  the usual largest-entry rule flips a component the moment two loadings
  trade the lead (it did, on a 5-column Gaussian). All three are read
  before the update, are chunk-invariant, live in the state file (schema
  still 2, `#[serde(default)]`), and are frozen by `predict`. Throughput
  at 200k rows, Mrows/s, k = 4 / 16: the base `mean, std, corr` 7.4 / 0.86;
  `+ mahal` 3.9 / 1.1; `pca=2, pca_every=1` 0.98 / 0.10 and `pca_every=100`
  6.5 / 1.9 — the decomposition is the cost, so amortize it. `output_index`
  kinds `mahal`, `mahal_q` (with its `quantile`), `pc_var`, `pc_share`,
  `pc_score` (over every feature) and `pc_loading` (its own column).
- Rust: `online_core::{Pca, EwCovStat::Mahal}`, `EwCovCfg::{mahal_quantiles,
  pca, pca_every}`; `EwCovModel::{mahal, pca}` and
  `EwCovModel::labels(names, stats, mahal_quantiles, pca)`.
- `tests/test_ew_cov_scores.py`: a Welford replay oracle for the whole
  stream (clock gaps, `max_dclock`, zero and null weights, skipped rows), a
  numpy `eigh` oracle with the continuity rule at every refresh, χ²
  calibration and a 3-factor recovery at 200k rows, a covariance switch,
  and the chunk / save-load / predict / expression / runner / TOML contract.
- **`ew_class`: Gaussian classification on per-class `ew_cov` moments**
  (`po.spec.ew_class`, `docs/ENHANCEMENTS.md` E39, `docs/PLAN.md` task 27).
  A label column (`label=`) and its declared `classes=` in place of a
  numeric target; one exponentially weighted Gaussian per class, scored by
  Bayes' rule *before* the row is learned. `covariance="full"` (default)
  gives each class its own covariance (QDA), `"shared"` pools them by the
  class weights (LDA, one factorization per row), `"diagonal"` keeps the
  variances (naive Bayes). `precision_prior` (required) is the per-class
  ridge that makes a class scoreable from its first row and fades like
  `partial_corr`'s. Outputs: `class` (String, null before `min_periods` or
  before any class has been seen), `p_<class>` for every declared class
  (exactly 0 for a class no row has carried), `n_eff` (every accepted row,
  labelled or not), and `coef` = the class means (`coef_<class>_<feature>`
  after `unnest`; `coef_index`'s `target` is the class). A null label
  scores the row and learns nothing from it — the late-label case; a value
  the spec does not list raises, naming the row, the value and the classes;
  integer, boolean and categorical label columns are read through their
  text (`classes=["0", "1"]`). Residual diagnostics are refused by name, as
  for the clusterers. Measured at 200k rows, six features, three classes
  with their own covariances: within 0.001 of the Bayes rate the generating
  parameters allow, posteriors calibrated to 0.01; 0.9 / 1.8 / 5 Mrows/s
  full / shared / diagonal (k = 6), 1.6 / 3.2 / 6.9 at k = 2.
- Rust: `online_core::{EwClass, EwClassCfg, Covariance}`,
  `online_core::quad_forms_logdet` (every quadratic form and the
  log-determinant off one Cholesky), `ModelState::EwClass` (schema still 2);
  `online_polars::ModelKind::EwClass { classes, covariance, precision_prior }`
  and the TOML `type = "ew_class"` with `targets = ["<label column>"]`.
- `tests/test_ew_class.py`: a replay oracle of the per-class recursion in
  the core's operation order (weights, means, `n_eff` bit-exact; posteriors
  to 1e-9 through numpy's `slogdet` / `solve`) over null labels, null
  features, zero and null weights and a capped irregular clock; the Bayes
  rate and calibration at 200k rows; LDA ≡ QDA on a shared covariance, QDA
  alone seeing a spread, a full covariance alone seeing a correlation, a
  class swap relearned; and the chunk / save-load / predict / groups / grid
  / coef / expression / lazy / runner / CLI / refusal contract.
- **`coef` lists are finite-or-null inside the list too.** A slot with no
  value (an `ew_class` class no row has carried) is null in the list and in
  `ModelBank.coef`, where a NaN would have broken the frame's
  finite-or-null rule.
- **Constrained coefficients on `sgd` and `pa`** (`coef_min`, `coef_max`,
  `coef_sum`; `docs/ENHANCEMENTS.md` E40, task 28). A bound is a number for
  every slope or one per feature, `inf` for no bound on that side; the sum
  fixes the slopes' total. After every update the slopes move to the
  nearest feasible point — the Euclidean projection onto the box, the
  simplex (`coef_min=0.0, coef_sum=1.0`: long-only, fully invested
  weights), or any box with a sum — and the intercept is never constrained.
  `O(k)` per row for a box alone, `O(k log k)` with a sum; no new state, and
  a saved state loads unchanged. The fit starts from the projected zero
  (uniform weights on a simplex); a zero-weight or null-target row leaves it
  alone; under `scale_features=True` the bound is on the coefficient in the
  caller's units and `coef` reports what the projection returned. `pa`
  projects after each update, so its step no longer meets the margin
  exactly and a truth outside the set is never reached — keep `c` small. A
  sum the bounds cannot reach, a floor above a cap, or an infinite bound on
  the wrong side is refused by name, from Python, the plan and the CLI.
  Rust: `online_core::Constraint` (`lo`, `hi`, `sum`, and `project`) on
  `SgdCfg::constraint` / `PaCfg::constraint`; TOML `coef_min = [0.0, 0.0]`,
  `coef_max = "inf"`, `coef_sum = 1.0` under `[specs.model]`. Verified by
  a Python replay of both models and the projection, held bit-exact to the
  bank's `pred`, `n_eff` and `coef` over six constraint sets, three
  schedules and three PA modes, with nulls, zero and NaN weights and an
  irregular clock; and on 200k rows, Dirichlet weights recovered on the
  simplex to 0.005 and feasible to 1e-12 at every row
  (`tests/test_constraints.py`).
- **Coefficient reversion on `kalman`** (`revert_halflife`;
  `docs/ENHANCEMENTS.md` E41, task 29). The coefficient prior was a random
  walk: a learned slope stayed until new rows moved it, and its variance
  grew without bound over a run of null targets. With `revert_halflife` a
  slot decays toward zero with the clock — `β ← Φβ`, `P ← ΦPΦ + Q·Δclock`,
  `Φ = diag(2^(−Δclock/r_i))` — before each update, so a regressor active
  only in bursts is forgotten between them and the prior variance settles
  at `q_i·Δclock/(1−φ_i²)`: a mean-reverting (AR(1)) coefficient. A scalar
  applies to every slot including the intercept; a list is one halflife per
  slot, intercept first, and `inf` (the default) keeps a slot a random
  walk. Zero is zero in the standardized coordinates — "no effect" for a
  slope, "the target averages zero" for the intercept — and the default is
  bit-identical to before. A zero-weight or null-target row still advances
  the transition (it is clock), and `ModelBank.predict` propagates by the
  same `Φ` over the distance from the last learned row, capped by
  `max_dclock`, so it returns exactly what the next `fit_predict` row would.
  NaN, zero and negative halflives are refused by name, from Python, the
  plan and the CLI; `rls`/`ewridge` do not take the argument. Rust:
  `KalmanCfg::revert_halflife` (`serde(default)`: a saved state loads
  unchanged); TOML `revert_halflife = ["inf", 40.0, 40.0]`. Verified by a
  Python replay held to 1e-9 against the bank over thirteen configurations
  (per-slot and shared `P`, explicit `q`/`obs_var`/`p0`, nulls, skipped
  features, zero weights, no intercept, `standardize=False`, a capped gap);
  the shrink over an irregular clock with skipped rows exact to 1e-11; and
  on 300k rows with the exact-Bayes process noise, a reverting filter
  tracks a sparse mean-reverting slope at 0.51× the random walk's
  tracking error (0.86× dense), a slot left at `inf` is within 0.2% of the
  unmodified filter, and a random-walk truth is still best tracked by the
  random walk (`tests/test_kalman_revert.py`).
- **`seqtest`: a sequential test of a sign, by betting** (`po.spec.seqtest`,
  `docs/ENHANCEMENTS.md` E42, task 30). Not a regression: per target it keeps
  two e-processes — the wealth of a gambler betting the next sign is
  positive, and of one betting it is negative, each staking the
  Krichevsky–Trofimov fraction `max(0, (n⁺−n⁻)/(n+1))` of the counts so far
  — so `log_e_pos ≥ ln(1/α)` rejects "no more likely positive than negative,
  given the past" at level α by Ville's inequality, read at any row, as
  often as you like, with no distribution assumed. Fields, all as they
  stood *before* the row: `log_e_pos_<t>`, `log_e_neg_<t>`, `n_pos_<t>`,
  `n_neg_<t>` (`i64`) and `n_eff`; a zero, null or NaN is a tie that bets
  and counts nothing. **With `a` and `b` it compares two specs of the
  bank**: the sign tested is `|resid_b| − |resid_a|` (positive when `a` came
  closer), the fields are `log_e_a_<t>`, `log_e_b_<t>`, `wins_a_<t>`,
  `wins_b_<t>`, `a_suffix`/`b_suffix` pick a grid instance, and a row where
  either side is null is no trial. The bank runs in two phases — every
  other spec, then the comparisons, reading the residuals from the structs
  just assembled — and returns the columns in spec order; a refused chunk
  updates neither phase; `ModelBank.predict` compares the scored sides.
  `weight`, `halflife`/`lam`, `features` and every residual diagnostic are
  refused by name (a trial is a row; an e-process does not forget);
  `session` and `on_clock_reset="reset_state"` restart it. `coef_fields` is
  empty and `coef_index`/`ModelBank.coef` refuse it. The expression form
  runs column mode on its column and refuses `a`/`b` with the way to write
  the comparison. **`po.eval.seqtest(df, targets=, a=, b=, by=)`** is the
  same computation in polars expressions over a frame you already have,
  bit-identical to the bank in both modes. Rust: `online_core::{SeqTest,
  SeqTestCfg}`, `ModelKind::SeqTest { a, b, a_suffix, b_suffix }`,
  `ModelState::SeqTest`; TOML `type = "seqtest"`, `a = "ridge"`, `b =
  "kalman"`. Verified against a scalar replay (`tests/reference.py::
  seqtest_ref`) to 1e-12, the closed-form KT wealth `2ⁿ B(n⁺+½, n⁻+½)/π`
  through `math.lgamma`, the twin on a million rows, and the guarantee
  itself: over twenty thousand fair-coin streams, and twenty thousand
  dependent ones still under the null, the crossing rate of `1/α` stays
  under α at 0.05 and 0.01 (`tests/test_seqtest.py`).
- **`--dry-run` and `RunConfig::validate` build the bank**, so a duplicate
  spec name or a comparison naming a spec the bank has not got is reported
  before the input is opened, not on the first chunk.
- **`ModelBank.last_row(spec=None, group=None)`: the output row of the last
  row each group learned from, as a frame**, from a live bank or one loaded
  from a state file (`docs/PLAN.md` §11a, task 34). One row per `(spec,
  group)` — `spec`, `group`, then the spec's output fields unnested, so it
  is the `fit_predict` row field for field: `pred`, `sigma`, the metrics,
  the interval, `n_eff`, `coef` when that row carried it (a chunk's last
  learned row does; `coef()` has them otherwise). Specs with different
  fields stack with nulls (`diagonal_relaxed`), and `pl.concat` over the
  `last_row()` of many saved banks is a table of fits to compare — the
  reason it exists: a run's diagnostics no longer have to be kept from the
  last row of its output file. The state file carries the row (`Bank::
  last_row`, `LastRow` in Rust), an additive field with no format bump, so
  a file from 0.1.x loads and reports a null row, as does a group that has
  not learned from a row yet; a chunk that ends in skipped rows keeps the
  row before them, and `predict` never moves it.
- **`ModelBank.summary(spec=None, group=None)` and
  `ModelBank.describe(spec=None, group=None)`: what each group was fed, as
  frames**, from a live bank or a loaded state file (`docs/PLAN.md` §11a,
  task 35). `summary` is one row per `(spec, group)`: `rows_fed`,
  `rows_processed`, `rows_skipped`, `rows_learned`, `rows_zero_weight`,
  `weight_sum`, `clock_min`, `clock_max`, `last_clock`, `session_changes`,
  `clock_backwards`, `resets`. `describe` is one row per input column per
  `(spec, group)` — `column`, `role` (`feature`, `target`, `weight`),
  `count`, `null_count`, `mean`, `std`, `min`, `max` — over every row fed,
  counting a null, NaN, infinity or magnitude beyond `1e100` as the models
  do, as missing; a label column carries counts only and an unsupervised
  model lists its features. Undecayed and accumulated in row order
  (Welford), so chunking cannot move a bit; `predict` does not move them.
  The state file carries the summary (`Bank::summary`, `Bank::describe`,
  `DataSummary` in Rust), an additive field with no format bump: a file
  from before it loads and reports nulls for everything but
  `rows_processed` and `last_clock`, which the stream always kept, and
  never grows a summary — a count that began partway would read as the
  whole history. A loaded summary is checked against its spec and its own
  arithmetic (`Stream::restore`), so a file whose summary is not its spec's
  is refused, not reported. Costs about a nanosecond per input column per
  row.
- **The state-file tests made rigorous** alongside it: every facet of a
  bank (coefficients, last row, summary, column statistics) equal across
  save and load; a loaded bank re-saves byte for byte; every truncation and
  a sweep of bit flips of a real file are refused or loaded, never a panic;
  and a frozen 0.2.0 fixture (`crates/online-polars/tests/state_schema3.rs`)
  that the next layout change has to keep loading, with its summary and last
  row intact and its stream continued to the bit.

### Changed

- **`SCHEMA_VERSION` is 5** (`po.schema_version()`), up from 2 in 0.1.x, in
  three steps that are each additive with defaults — so **a 0.1.x bank file
  loads, continues the stream to the bit, and re-saves in the new layout**,
  and `MIN_SCHEMA_VERSION` is still 1. Schema 3 is `kalman`'s and `sgd`'s
  standardiser (below); schema 4 is the spec gaining `label_delay`, and a
  stream carrying the rows it has accepted but not yet learned from; schema
  5 is the spec gaining `group_close`, and a stream that closes on session
  keeping the session value of the span it is in. Nothing in a model's own
  state changed in 4 or 5 — the bumps are because every spec's bytes moved,
  which is what hard rule 5 asks to be told about. Each layout is frozen as
  a fixture with a real file's bytes (`state_v1.rs`, `state_schema2.rs`
  through `state_schema5.rs`); the schema-5 one carries a bank with two
  closed groups nobody has read, an `ew_cov` with a partly filled lag ring,
  and one of each new model mid-stream. The bank `format_version` is
  unchanged at 2. A 0.1.x build cannot read a file saved by 0.2.0, as
  before for any newer schema.

- **Residual diagnostics are refused for a model that predicts no target.**
  `ew_cov` already refused `emit_selected` and `emit_averaged`; it,
  `kmeans`, `micro` and `ew_class` now refuse `emit_sigma`, `emit_resid_z`,
  `emit_metrics`, `resid_quantiles`, `emit_autocorr`, `emit_drift` and
  `conformal` too, by name
  (`"emit_sigma does not apply to ew_cov (it has no predictions, so no
  residuals)"`), where `ew_cov` used to accept the flag and silently emit
  nothing for it. A spec that set one of them on `ew_cov` must drop it.
- `Decay::factor` computes `exp2(-d/h)` rather than `0.5.powf(d/h)`. A
  release build already did (LLVM rewrites the one into the other), so no
  released number moves; a debug build now agrees with it bit for bit, and
  so can a reference in another language.
- **The new families surveyed and tuned, bit for bit** (`docs/PERFORMANCE.md`
  §13, task 31). Every output is unchanged — two dumps over every family,
  with groups, weights, `predict` and chunk ends, compare bit-identical
  against the previous build — and, per 400k rows at k = 20 on one thread:
  the default `ew_cov` (mean, std, corr: 230 statistics) 758 → 222 ms, the
  full-covariance `ew_class` 1510 → 808, `ew_cov` PCA at `pca_every=100`
  332 → 162, `micro` 37 → 24, the simplex-constrained `sgd` and `pa`
  179 → 157 and 169 → 149, `kalman` 238 → 225. What changed: `ew_cov`
  takes `k` square roots per row for its correlations rather than
  `k(k−1)`; `ew_class` keeps one Cholesky factor per class and
  refactorizes only the class a row learns; `kalman` and the constraint
  projection reuse per-row scratch instead of allocating; a model with no
  predictions no longer carries residual tracking; and the bank feeds each
  (spec, group) through its stream in runs sized to the cache with a
  slot stride that is an odd number of cache lines, so a wide model's
  speed no longer depends on the caller's chunk size — a 65 536-row chunk
  used to run the default `ew_cov` 2–3× slower than one of 65 552. Output
  validity is assembled as a packed bitmap, and a single-group chunk is
  copied rather than scattered. Thread scaling at 14 threads over 64
  groups: `ew_cov` 276 → 152 ms, `ew_class` 324 → 191 per 800k rows.
- **`kalman` and `sgd` standardize with a diagonal accumulator**
  (`EwDiag`; task 33). Both kept a full exponentially weighted covariance
  of the features and read only its diagonal — O(k²) of co-moment updates
  a row for k variances. The replacement is that diagonal, operation for
  operation, so every output is unchanged to the bit (a dump over twelve
  configurations, with groups, weights, two targets, a save/load mid-stream
  and `predict`, compares bit-identical against the previous build), and,
  per 400k rows on one thread: `kalman` at k = 20 223 → 179 ms, at k = 50
  1065 → 908; `sgd` with `scale_features` at k = 20 109 → 62, at k = 50
  279 → 121. **State schema 3.** The two models' serialized layout changed,
  so `SCHEMA_VERSION` is 3 (`po.schema_version()`); bank files written by
  0.1.x (schema 2) still load — a schema-2 `kalman` or `sgd` state is
  converted by taking the diagonal it was already using, and continues the
  stream identically — and a real 0.1 bank file is frozen as a fixture
  (`crates/online-polars/tests/state_schema2.rs`) alongside the schema-1
  one. The bank `format_version` is unchanged at 2. A 0.1.x build cannot
  read a file saved by 0.2.0, as before for any newer schema.
- **README *Performance*: the other families** — a second table
  (`scripts/benchmark.py --markdown`) for conformal intervals, `sgd`, `pa`
  and the simplex, `kalman` with reversion, `ew_cov` with and without the
  Mahalanobis distance, the three `ew_class` covariances, `kmeans`,
  `micro` and `seqtest`; the regression table regenerated on the same
  build.
- **A documentation pass over everything** (task 58). Each of the twenty
  model sections in the README now links to its builder in the API
  reference, whose docstring lists every keyword with its default, and to
  its Rust module, whose comment states the recursion; the README gains a
  table of contents and groups what a stream needs before a bank sees it
  (*Preparing a stream*) and what a bank can tell you afterwards (*Reading
  the fit*). Every builder docstring was checked against the code that
  reads it, so a default in the reference is the default the model uses.
  `docs/README.md` is new and says which document to read for what; every
  document under `docs/` opens with a dated line saying what became of it,
  and `docs/STATE-WORKFLOW.md` opens with the four-step workflow itself
  rather than the research behind it. Nothing that runs changed.

### Fixed

- **A review of the E54–E64 batch, and every item it found**
  (`docs/REVIEW-E54-E64.md`, task 57). Twenty-nine items: twenty-eight
  fixed, each with the test that catches it, and one made and then reverted
  (below) when CI showed what it cost. Four were wrong answers rather than
  missing guards, and three moved a golden.

  - **`ew_cov` lags: a bank saved before the ring was full never filled it
    again.** The ring's depth was read back from `VecDeque::capacity()`,
    which a clone or a msgpack round-trip shrinks to the length it holds, so
    a save taken on a short stream — or just after a session change or a
    capped gap, both of which empty the ring — left the deeper lags decaying
    to nothing for the rest of the run.
  - **`rcov`: a clock break inside a block corrupted the estimate.** A gap
    over `max_dclock`, or a session change, dropped the end-jitter ring
    instead of closing the stretch: the `m` returns waiting in it were lost
    and the first return after the break was emitted `m + 1` times, with the
    reported `n` unchanged so nothing downstream could tell. A break now
    splits the block into **stretches**, each closed the way the last one
    is, and the lagged sums add over them — so no product pairs two returns
    across the break, and an unbroken block is unchanged to the bit.
  - **`hmm`: `min_periods` withheld the row from the filter, not just from
    the output.** A warm-up row was never learned from, and under decay an
    `n_eff` that plateaued below the threshold meant a filter that never
    learned at all — every row null, for ever. It now gates the report
    alone, as it does in every other model here.
  - **`bocpd`: a hazard column value of 1 or less silently dropped the
    row.** It reported nulls and left the posterior where it stood while
    `n_eff` counted it. Such a value is now refused naming the row (null
    still falls back to the spec's own `hazard`), and a row whose predictive
    cannot be evaluated is counted in `solve_failures`.
  - **`predict` reads the targets slot** where a model takes a parameter
    from it: `bocpd`'s `hazard_col` and `hmm`'s `exog_tvtp` used to answer
    from the configured default, so `ModelBank.predict` disagreed with
    `fit_predict` on every row where the column differed. Both now give the
    step's answer, and the `predict == step` contract tests it with a column
    that varies.
  - **`hmm` and `corrchange` did not decay `n_eff` on a zero-weight row**,
    so `min_periods` quietly meant a different number of rows for them after
    any weightless stretch (hard rule 8). The model contract now checks the
    recursion against zero-weight rows for every model, without naming a
    decay: a zero-weight row must advance the clock and nothing else.
  - **`label_delay` replayed across a capped gap after the ring was
    cleared.** Only the matured rows were released on a gap over
    `max_dclock`, so the ones still waiting were learned *after* the models
    dropped their row-lagged state — pairing rows across the very break the
    clear was for. All pending rows are now released on a capped gap, as
    they already were on a session change.
  - **`prep.refresh_time` returned the `by` column as text** whatever it
    came in as, so the result did not join back to its own input (and did
    not match the schema the lazy plan declared).
  - **A high-water mark is now refused in both directions.** A mark written
    under an integer key column and read under a text one let through
    exactly the groups it exists to refuse (`"9" > "10"` bytewise); the flag
    is saved with the bank.
  - Validation gaps closed, each with the case that reaches it: `bocpd`'s
    `prior_scale` (positive, symmetric, positive definite) and `prior_mean`
    (finite); `hmm`'s given `covs` (symmetric and positive definite — a
    state with no density takes no responsibility for any row);
    `corrchange`'s `crit`; `rcov`'s `window`, `block_rows` and `max_bandwidth` against
    `bandwidth`; `deco`'s block list written by hand (a duplicate name, an
    empty list); `corr.nearest` on non-finite input and `max_iter = 0`;
    `corr.shrink` on fewer than two rows; and `sim.regimes` normalises a
    transition row that `np.allclose` accepts but `rng.choice` does not.
  - One item was fixed and then **reverted**: renormalising `bocpd`'s
    run-length log-joint is exact on paper, but `z` comes out of `ln` and
    libm's last bit differs between platforms, so feeding it into the state
    made a bank saved on macOS continue differently on Linux. A libm result
    may be compared or reported; putting one into the state costs
    cross-platform reproducibility, which is worth more than the tidiness.
  - Documented rather than changed, with a test pinning each: ties in
    `refresh_time` are broken by row order; a zero-weight `corrchange` row
    is reported as if it would be learned; `eig_vecs` are signed for
    continuity with the previous *closed group*; a `Categorical` group key
    is ordered bytewise, so a frame sorted by its physical order is refused.

## [0.1.1] — 2026-09-04

A faster chunk plan and a guide to the chunk size. Every number a model
produces is unchanged.

### Changed

- **The chunk plan: every phase parallel, and no stride** (`docs/PERFORMANCE.md`
  §12, P9–P11). A spec's columns are gathered once, group after group, so
  each stream reads a contiguous run; output fields are built one job each
  into `Vec<f64>` + validity; columns are read in parallel, multi-chunk
  columns copied per arrow chunk, and an integer group key is bucketed by
  its value rather than cast to text (the keys and output are identical, and
  a test says so). Measured on a 400k-row chunk over 64 groups at 14
  threads: 37 → 17 ms; the README's 12M-row grouped workload 3.25 → 2.48 s.
  Below 4096 rows a chunk's columns and fields are done on the calling
  thread, so the expression plugin's small `.over()` groups do not fan out
  for nothing. Every output is bit-identical; the golden, chunk-invariance
  and oracle suites are unchanged.
- **README: a *Chunk size* guide** under Parallelism — what `chunk_rows`
  does and does not change, and a sweep from 20k to 2M rows on interleaved
  and group-sorted data. The section's memory numbers are now peak
  footprint (`/usr/bin/time -l`), where they had been RSS with the
  memory-mapped input counted in. The README's measured numbers are
  regenerated on this build.
- **`benchmark.yml` keeps both tables in its artifact** (throughput and
  thread scaling), so two runs can be compared without the browser.

## [0.1.0] — 2026-09-03

First release.

### Models

Ten online regression models plus streaming moments, all on exponentially
weighted **mean-form** accumulators with centered (Welford) co-moments:
`ewridge`, `rls`, `lasso`, `kalman`, `huber`, `quantile`, `sgd`, `pa`, `ftrl`,
`holt`, and `ew_cov`.

### Interfaces

Three, with identical numerics: a Polars **expression plugin**
(`pl.col("y").online.ewridge(...)`; in-memory only, and it warns so since —
see *Changed* below), a chunk-fed **`ModelBank`** with O(state)
memory that reports what it holds (`groups()`, `rows_seen()`) and can forget
stale groups (`drop_groups()`), and a standalone **CLI** (parquet in, parquet
out, TOML config). The Python surface is typed: PEP 692 keywords on the
builders and the namespace, and `po.online(expr)` for type checkers, which
cannot see a registered namespace.

### Parallelism

One task per (spec × group) per chunk on the bank's own thread pool, sized
by **`POLARS_ONLINE_MAX_THREADS`** (unset: one per core; read when the pool
is built, at the first bank call; `po.thread_pool_size()` reports it).
Polars' readers and writers stay on polars' pool, `POLARS_MAX_THREADS`,
which also sizes what its reader holds in flight — so a run can keep polars
small for memory and give the bank every core, and the README's
*Parallelism* section measures why. A value that is not a count is refused
by name. Thread count changes speed and nothing else; a test runs the same
stream at 1 and 8 threads and requires identical output.

### Guarantees

- Predictions are out-of-sample by construction.
- Chunk invariance: 1 chunk or 1000 produces identical output, as does saving
  state mid-stream and resuming. (`coef` is a reporting cadence and excepted.)
- `n_eff` means the same thing in every model, which is what makes
  `min_periods` portable across a bank.

### Diagnostics

`emit_sigma`, `emit_resid_z`, `emit_drift` (Page-Hinkley), `emit_metrics`
(ic / r² / hit rate), `emit_autocorr`, `resid_quantiles` (P²),
`emit_selected` and `emit_averaged` for online model selection and averaging.

### Verified against [river](https://riverml.xyz)

FTRL's z/n recursion to 1e-12; Kalman ≡ `BayesianLinearRegression` to 3.6e-15;
`EwCov` ≡ river's Welford statistics exactly. Two documented places where the
libraries legitimately differ are pinned by tests rather than left as
surprises.

### Known limitations

- `polars>=1.34.0,<2`. The floor is measured (`LazyFrame.collect_batches`,
  which the streaming paths read with, arrived in py-polars 1.34.0; see
  *Changed* below); the ceiling is a bet that 1.x keeps the interface, hedged
  by a version-negotiated plugin ABI that refuses to load rather than
  misbehave and by a weekly canary against the latest polars. The Rust
  `polars` inside the wheel is pinned exactly, but it never meets the user's
  copy — data crosses on the Arrow C Data Interface. The README's
  *Versioning and the Polars pin* has the matrix.
- Requires Python 3.12+ (`abi3-py312`).
- Wheels for macOS (arm64, x86_64), Windows x64 and Linux (x64 glibc and
  musl, aarch64 glibc); anything else builds from the sdist with a Rust
  toolchain.

### Before the release

*The entries below were written as the code evolved, before anything was
published; they describe changes relative to earlier development snapshots,
not to a released version, and stay because they record why things are the
way they are.*

#### Added

- **`online.unnest(specs)`: a bank's output as flat columns, the
  coefficients named.** `lf.online.unnest(specs)`, `df.online.unnest(specs)`
  and `po.unnest(frame, specs)` take each spec's struct column apart in
  place — scalar fields under their own names, each `coef` list as one
  column per coefficient named on the field grammar (`coef_y_intercept`,
  `coef_y_x1__r0.5@h500` beside `pred_y__r0.5@h500`). `specs` may be the
  spec dicts, a `ModelBank`, or the path of a saved state; a parquet the
  CLI wrote reads back flat through `pl.scan_parquet(..).online.unnest(..)`.
  The names come from the new **`polars_online.spec.coef_fields(spec)`**:
  every coefficient with the `coef` field it sits in, its `position` there,
  its column `name`, and `target`, `halflife`/`lam`, `ridge`,
  `feature_set`, `lambda`, `term` — rendered by the same Rust code as the
  field names (`online_polars::coef_fields`, `CoefField`). `coef_index` is
  unchanged and is now derived from it.
- **A weekly native leak check in CI** (`.github/workflows/leakcheck.yml`,
  PLAN task 18): `scripts/leakcheck.sh` under `leaks` on macOS and valgrind
  on Linux, Mondays and on demand; nothing gates on it, a red scheduled run
  is the report. Wiring it showed the script's earlier "0 leaks" was a blind
  check — pymalloc's arenas are invisible to both tools — so it now runs
  under `PYTHONMALLOC=malloc`, counts differentially (1 iteration against
  1000) and has a control mode that leaks one object per iteration and must
  be caught; the job runs the control too. What it cannot see, and says so:
  memory from polars' allocator, i.e. the Rust side, which
  `tests/test_ffi_memory.py` covers by RSS.
- **An API reference, built from the docstrings with Sphinx** (`docs/reference/`;
  `uv run --group docs sphinx-build -W docs/reference docs/_build/html`), in
  the gate and CI with warnings as errors, and published to GitHub Pages
  from `main`: <https://hgilde.github.io/polars-online/>. The `docs`
  dependency group (`sphinx`, `furo`) is separate from `dev`. The first
  build found four docstrings that were not valid reStructuredText; fixed.
- **`ModelBank.coef(spec, group=None)`: the coefficients behind a fit, as a
  frame**, from a live bank or one loaded from a state file — one row per
  `(group, instance, position)` with `coef_index`'s `target`, grid values
  and `term`, so `bank.coef("ols").pivot("term", index=["group",
  "instance"], values="coef")` is the betas per group. The values are what
  the output's `coef` field reported on the last row each group learned
  from (the fit after that row, which the next `pred` is computed from);
  null before the group's first solve — the solve schedule's decision, not
  `min_periods`', which gates `pred` alone, so the frame carries `n_eff`
  for how much weight is behind each fit; an empty frame for a group the
  bank has never seen; `ValueError` for `ew_cov`. Rust: `Bank::coef` and
  the `Coef` row. Before this, the betas of a saved state were reachable
  only through `predict` on a one-row frame or a hand solve over `gram()`.
  The README's new *Saving, loading and reading a model* section shows
  save/load and the coefficients each in both the bank's form and the
  query's (`coef_index` + `list.to_struct` for the per-row path in polars).
- **`save_state=` on the plan: `lf.online.fit_predict(specs, load_state=,
  save_state=)`**, and on `df.online.fit_predict` and `po.fit_predict`. The
  state the execution ends in — after the last row the source fed the bank:
  the stream's end, or the `n` rows of a `head(n)` — is written to the path
  when the run ends, atomically (`ModelBank.save`), as the same bytes a bank
  fed those rows saves and `po.run(save_state=)` writes. `load_state` and
  `save_state` may be the same path, for a resume in place. The plan stays
  pure, and that is what makes the write safe: polars runs a plan's source
  once per use in a query — twice, on two threads, under a self-join,
  `pl.concat` or `pl.collect_all` of two sinks — and every run writes the
  same bytes. Nothing is written by a run the caller abandons or one the
  bank ended with an error; a node after the bank failing does not stop the
  bank, so the state is written then (`po.run` saves only after its output
  is committed). `docs/STATE-WORKFLOW.md` has the measurements and the
  rules.

- **`lf.online.fit_predict(specs)` — the bank as a polars source.** A
  `LazyFrame` in, a `LazyFrame` out: executing it (`collect`,
  `collect_batches`, `sink_parquet`, …) streams the plan's rows through a
  fresh `ModelBank` in `chunk_rows` chunks, so a query with the bank in it is
  O(chunk) in memory — where the expression plugin in the same query is
  O(data) in either engine, because polars calls a user expression once with
  its whole column. Bit-identical to `po.run`'s output; 12M rows in 2.8 s at
  0.78 GB live (the plugin: 14.4 s, 7.3 GB). Filters, selections and `head`
  after the bank are pushed into the source and honoured there — a filter
  after never changes what the bank learns from — and a selection reaches
  the input scan. The plan is pure: every run starts from the specs' state
  or `load_state` (and `save_state=`, below, writes where it ends). Also
  `lf.online.predict(bank)` to
  score against a bank or a state file, the eager twins
  `df.online.fit_predict(specs)` / `df.online.predict(bank)`, and
  `po.fit_predict(frame, …)` / `po.predict(frame, bank)` for type checkers.
  Rides on polars' IO-plugin interface (`register_io_source`), documented
  but marked unstable by polars (`docs/RELEASE-READINESS.md`).
- **`ModelBank.predict(df)`** scores a frame against the bank exactly as it
  stands and updates nothing: no clock advance, no decay, `n_eff` frozen. Row
  `i` carries what `fit_predict` would have reported had it been the next row
  of the stream — the same fields, the same values — every row scored from the
  same state, with the clock distance measured from the last row the bank
  learned from. The target column is optional (then `resid` is null), `weight`
  is not read, unknown groups score null, and the stream's session and clock
  policies still hold. Concurrent `predict` calls are fine; a `fit_predict`
  racing one is refused. Also from the runner, as `po.run(predict=True)`, the
  TOML key `predict = true`, and the CLI flag `--predict` — each needs a
  loaded state and refuses `save_state` (the keyword and the flag drop a
  config's own `save_state`, so one TOML serves both the learning and the
  scoring run). Roughly twice the throughput of `fit_predict`.
- **`OnlineModel::predict`** (Rust, `online-core`): the step without the
  step, implemented by every model and held to `predict == step` row by row
  in `tests/model_contract.rs`.
- **The runner reads and writes parquet, ipc, csv and ndjson**, told from
  the extension (`.parquet`/`.pq`, `.ipc`/`.arrow`/`.feather`, `.csv`,
  `.ndjson`/`.jsonl`) or named with `input_format=` / `output_format=` (TOML
  keys of the same names, CLI `--input-format` / `--output-format`). CSV
  cannot hold the bank's struct columns, so there each spec is flattened to
  `<spec>.<field>` columns and `coef` is a JSON list;
  `pl.col("ridge.coef").str.json_decode(pl.List(pl.Float64))` reads it back
  bit-exact.
- **`po.run(input=...)` takes any source py-polars can stream**: a path
  (globs and cloud URLs as `pl.scan_*` takes them), a `LazyFrame` — any query,
  including one with a Python UDF — a `DataFrame`, or any iterable of
  `DataFrame`s in stream order. The reading is py-polars' own
  (`collect_batches`), so a CSV streams through the wheel's SIMD parser;
  frames handed in are taken as they come, `chunk_rows` chunking what polars
  reads. Also `keep_columns=` to select input columns before the bank (and
  before the scan reads them), and `progress(rows, chunks)`, called after
  each chunk; an exception raised in it or in the input iterator surfaces as
  itself and no output is published.
- **Rust:** `online_polars::run(bank, Input, Output, RunOptions, progress)`
  is the pipeline the CLI and `po.run` share — `Input::Lazy(LazyFrame)` or
  `Input::Batches { frames, schema }` in, `Output::File { path, format }` or
  `Output::Batches(callback)` out — with `run_config_on(cfg, Input, ..)` for
  a `RunConfig` over an input the caller already has, `Format`
  (`from_path`, `scan`, `name`, `ALL`) and `DEFAULT_CHUNK_ROWS`.
  The three stages now overlap and a plan is read in one streaming pass
  instead of a `slice().collect()` per chunk, so on parquet the same run is
  **1.6–2.7× faster than before** (`po.run`, 3M rows: 1.93 → 0.72 s with
  groups interleaved, 3.30 → 2.12 s group-sorted; the CLI 2.14 → 0.83 s and
  3.84 → 2.55 s). The extension grows 4% for the three extra formats
  (gzipped 18.8 → 19.8 MB; wheel 19.8 → 20.8 MB; CLI 51 → 53 MB); no new
  dependency outside polars. Measured in `docs/PERFORMANCE.md` §10.

#### Changed

- **The bank's thread pool is its own, sized by `POLARS_ONLINE_MAX_THREADS`**
  (`crates/online-polars/src/pool.rs`), where it used to be rayon's global
  pool and `RAYON_NUM_THREADS` — a name that said nothing about which pool
  it was next to `POLARS_MAX_THREADS`. `RAYON_NUM_THREADS` now reaches
  nothing here: the per-core default is spelled out rather than left to
  rayon, and `tests/test_portability.py` checks that neither it nor
  `POLARS_MAX_THREADS` sizes the bank's pool. The runner's parquet page
  encoding and NDJSON serialization moved the other way, onto polars' pool
  (`polars_core::runtime::THREAD_POOL`, already in the tree), so that
  `POLARS_MAX_THREADS` is polars' readers *and* writers in every form.
  `po.thread_pool_size()` is new, the mirror of `pl.thread_pool_size()`.
- **Every public entry point documents its failure modes, and they follow
  one contract** (`polars_online.__doc__` states it): a file problem is
  the `OSError` subclass for what went wrong, naming the path; a parameter,
  spec or column problem is `ValueError` naming the spec, parameter or
  column; the wrong kind of object is `TypeError`; a name or position that
  is not there is `KeyError`/`IndexError`; a bank used from two threads is
  `RuntimeError`; inside a plan, a run-time error is polars'
  `ComputeError` carrying that message. Rust: `# Errors` on `Bank::new`,
  `fit_predict`, `predict`, `run_config`, `run_config_on` and `run`;
  `cargo doc` builds without a warning. What did not fit the contract
  was changed to:
  - **An unknown key in a spec, a `po.run` config or a CLI TOML is
    refused**, naming the keys there are (the CLI with the line), where a
    misspelt `halflfe` used to fall silently back to the default.
    `Spec`, `ModelKind` and `RunConfig` are `deny_unknown_fields`.
  - `ModelBank.load` raises `FileNotFoundError` (or the `OSError` subclass
    the platform gives, e.g. `PermissionError` for a directory on Windows)
    for a path it cannot read and `ValueError` for a file that is not a bank, a
    newer build's file (now told from garbage by its envelope), or a spec
    mismatch — it raised `OSError` for all of them.
  - `po.run` raises the `OSError` subclass for an unreadable `load_state`,
    an unwritable output or `save_state`, each naming the path, and checks
    `save_state`'s directory *before* the run, so a typo there no longer
    costs the run and the state. `config=` that is not a dict, a path or
    `None` is `TypeError`; `chunk_rows < 1` is `ValueError` on every surface.
  - A spec position out of range (`ModelBank.gram(3)`, `groups(3)`, …) is
    `IndexError`, not `ValueError`; every bank method, not just
    `fit_predict`, gives the "bank is busy" `RuntimeError` when another
    thread holds it.
  - A chunk the bank refuses leaves no empty new groups behind: `groups()`
    lists only what the bank has learned from.
  - `eval.unpack` on a struct with no `pred_*` fields is `TypeError` naming
    the fields it found; `rolling_metrics(window=0)` is `ValueError`; a
    non-numeric `clock` there is `TypeError`.
  - Rust: `online_polars::Gram` is exported like the other bank types.
- **`load_state=` on the plan, and `predict(path)`, read the file when the
  plan is built**, not each time it runs: the plan carries the state, as
  `df.lazy()` carries a frame, so a plan collected twice gives the same
  frame whatever happened to the file in between, and `load_state=p,
  save_state=p` used twice in one query cannot race one run's load against
  the other's write. Build the plan again to pick up a newer file.
  `predict(bank_object)` still scores the bank as it stands when the plan
  runs.
- **`head(n)` on the plan feeds the bank exactly `n` rows.** The source
  applied polars' pushed slice to its output and fed the bank the whole
  chunk the `n`th row fell in; it now trims the input chunk, so the state
  after a `head(n)` is the state after `n` rows. The numbers are unchanged
  except `coef`, reported on each chunk's last row, which now lands on the
  `n`th row.

- **The expression form warns on every use** (`docs/PLAN.md` §6). Each
  `pl.col("y").online.<model>(...)` call now issues
  `polars_online.InMemoryExpressionWarning`, new and exported: polars hands
  a stateful user expression its whole column in either engine, so that
  form is O(data) — 7.3 GB at 12M rows against 1.35 GB for
  `lf.online.fit_predict([spec])` in the same query — and a reader who took
  it for the streaming form learned otherwise from a memory profile. The
  warning says why, names the plan to write instead, and gives the one-line
  filter for a frame in memory on purpose; it is a `UserWarning` because a
  `DeprecationWarning` is hidden outside `__main__`, which is exactly the
  pipeline module where it matters. The README shows the two forms side
  by side in a closing note. Nothing else moves: the expression still runs,
  `po.online` is still exported, and the numbers are the same bits.
- **`polars>=1.34.0,<2`** (was `>=1.28.1`). `po.run` over a path or a plan
  has read with `LazyFrame.collect_batches` since the runner became
  format-agnostic, and so does `lf.online.fit_predict`; py-polars added it
  in 1.34.0, and on 1.28.1–1.33 those calls failed with an `AttributeError`
  the latest-only canary could not see. The whole suite passes on 1.34.0,
  1.38.1 and 1.44.1 with identical numbers; `ModelBank` and the expression
  plugin alone still work from 1.28.1.
- **`lasso`'s `lam_selected_<target>`** is reported as it stood *before* the
  row — the λ the row was scored with — rather than after the row's error
  joined the selection. A one-row shift in that column, which makes it
  identical between `fit_predict` and `predict`.
- **`Step.coef`** (Rust) is gone: it was never `Some`; coefficients are read
  through `coefficients()`.
- The busy-bank message now says the bank "is in use on another thread" and
  that concurrent `predict` calls are fine.
- **Rust runner API:** `run_lazy` is `run` and takes an `Input`; the progress
  closure returns `PolarsResult<()>` (an `Err` stops the run) instead of
  `()`; the native module's `run_config` is `run_config_frames` (the Python
  side reads, Rust fits and writes). `po.run`'s signature and TOML keys are
  unchanged, with the new keywords optional.
- **`rls` is 20% faster at k=20 and 57% faster at k=50** (model arithmetic;
  1.63M → 1.93M rows/s through the bank at k=20). The per-row
  back-substitution summed each row in the one order that serialized it on
  the coefficient just solved; it now sums from the far end. A summation
  order is a rounding-level change: the golden signatures moved by at most
  1.2e-15, inside their 1e-12 tolerance.

#### Documented

- **The docs say what the bank is for a table in any row order.** It was
  introduced as "built for ordered event data"; row order reaches a bank's
  fit only through decay, and with decay off (`halflife=inf`, `lam=1.0`)
  `ewridge` is least squares over every row seen — `numpy.linalg.lstsq` to
  2e-13 in any order, 1.4 GB against `lstsq`'s 3.97 GB at 6M rows × 20
  features from parquet. The README leads with both shapes and has an "Any
  row order" section; `tests/test_row_order.py` pins the claim. Documented
  with it, and pinned, a trap that is not fixed: a huge *finite* halflife
  inherits the `halflife/50` solve cadence and so solves once and stays
  there — `inf` is the no-decay setting and `solve_every` the throttle.
- **How to score without learning**: give the rows weight `0`, which freezes
  the coefficients bit for bit, rather than nulling the target, which does
  not. The README also states the cost — a zero-weight row still advances the
  clock, so `n_eff` decays while scoring and `min_periods` can blank the
  output.
- **An upstream `filter` costs a bounded window, not the data**
  (`docs/PERFORMANCE.md` §11). The plan form reads its input with polars'
  `collect_batches`; the streaming engine bounds what is in flight in
  morsels per thread, and for a predicate pushed into a parquet scan the
  morsels are whole row groups of what the filter keeps: the reader applies
  the predicate itself and the scan then restores the column order through
  a 7-slot-per-thread pipeline — 2.5 GB at 12M rows and 3.1 GB at 36M on
  14 threads when the filter keeps every row, 1.2 GB when it keeps half,
  0.7 GB on 2 threads, 0.38 GB with the predicate column first in the
  projection (the stage becomes a no-op; an accident, not a recipe) —
  against 0.65 GB with nothing upstream and 0.78 GB for the same filter
  written *after* the bank, where it is pushed into the source and applied
  per chunk. Filter after unless the model should skip those rows, and then
  give them weight 0 through `when/then/otherwise` (1.3 GB; 1.1 with
  `pl.Config.set_streaming_chunk_size(25_000)`, which shrinks any
  `with_columns` window, and the filter's only once such a node above the
  scan makes the reader split its row groups) rather than filtering.
  `.over()` and `sort` upstream are O(data), and `sink_batches` with the
  default engine collects its input on polars 1.x (`engine="streaming"`
  streams; 2.0 makes it the default). The
  engine's own map of which is which:
  `lf.show_graph(engine="streaming", plan_stage="physical")`. A filter run
  inside the source would be 0.81 GB with identical output; not added — a
  second `filter` differing only in memory, for a cost that is
  polars' column-reorder stage to remove.
- **Which surface is O(data)**, measured (`docs/PERFORMANCE.md` §11): the
  expression plugin — 2.0 GB at 3M rows, 7.3 GB at 12M, in either engine,
  because polars' streaming engine collects the input of any user
  expression before calling it. The bank, `po.run` and the CLI are flat at
  ~0.75 GB from 3M to 12M rows, nearly all of it polars' parquet
  read-ahead; `POLARS_ROW_GROUP_PREFETCH_SIZE=1` takes the CLI to 0.15 GB at
  the same speed.
- **The README is written to be read.** It opens with a summary of what the
  package does — the model table, the ways to run a bank, the stream
  semantics, the two guarantees, state as a file, diagnostics — and then
  keeps like with like: the loop, the query, the job and the expression form
  under one heading, the stream parameters under another, the coefficient
  and field-name material together. The facts are the previous README's;
  measurements stay where they decide something and otherwise point at
  `docs/`. Every python block still runs under the README harness in
  `tests/test_production_hardening.py`.

#### Fixed

- **Two writers of one state file in one process no longer share a
  temporary.** `atomic.rs` named its temporary sibling by pid alone, so two
  threads saving the same path at the same moment — `ModelBank.save` from
  two threads, or now a plan with `save_state=` used twice in one query —
  created and wrote *the same* temporary and the rename published a
  mixture. The name now carries a process-wide sequence number, so the
  destination is always the old file or one writer's whole file; the
  runner's output file is written the same way and is covered too. Held by
  a two-thread, fifty-round test that fails under the old name.

- **The runner (`po.run` and the CLI) no longer panics on a parquet with
  more than one row group.** A chunk that spanned a row-group boundary
  arrived as a multi-chunk frame, the bank's outputs are single-chunk, and
  the batched parquet writer handed arrow a record batch of mismatched arrays
  (`RecordBatch requires all its arrays to have an equal number of rows`).
  Polars writes 262,144-row groups by default, so any file longer than
  `chunk_rows` was affected; the tests had written every input in one row
  group. The chunks are now aligned before writing.

- **`coef` is null, never an empty list, before a model's first solve.** Rows
  between `coef_every` snapshots were already null; warmup rows were an empty
  list, which made `coef.list.get(position)` — the documented way to read one
  coefficient — raise "index out of bounds" instead of returning null.
- **`holt` accepts `level_halflife` on its own.** For Holt the level halflife
  is the spec's halflife — one knob under two names — but a spec that gave
  only `level_halflife` was refused with "one of halflife/lam is required",
  including the example in this project's own README.
- **Every polars dtype can now cross into the model bank.** A `Decimal` or
  `Int128` column *anywhere* in the frame — even one no spec named — aborted
  the process with `activate 'dtype-decimal' feature`, and `Int8`/`UInt8`/
  `Array` columns failed with a polars error naming neither the column nor the
  fix. The missing dtype features are enabled, so unused columns are carried
  through whatever they are, and narrow numeric columns (`UInt8`, `Decimal`,
  …) are usable as features, cast to `f64` and bit-identical to the `Float64`
  columns they came from. The extension grows 7% (gzipped 17.6 → 18.9 MB); no
  new dependency.

- **State and output files are written atomically** — a temporary sibling,
  then a rename into place. `ModelBank.save` used to truncate the destination
  and write into it, so an interrupted save (a kill, a full disk, a quota)
  left a truncated file *and* destroyed the last good state; a `--resume` loop
  then started the stream over. The CLI's output parquet is published the same
  way, so a run that fails halfway leaves the previous output intact instead
  of a headless parquet under its name. Saving now costs a filesystem sync
  (~4 ms on macOS, where `sync_all` is `F_FULLFSYNC`); save less often if that
  matters more than surviving a crash.
