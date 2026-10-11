# Models

Each model's entry says when to reach for it, its update rule, its
parameters and what it is checked against. The README's
[Models](../README.md#models) has the table of all twenty-two, how each
learns from a row, and the notation the update rules use.

## Linear models

Ten models predict a numeric target. Nine regress it on features, and
`holt`, a baseline, uses none. Several of them take the same parameters:

| parameters | taken by | explained under |
|---|---|---|
| `solve_every`, `max_rows_between_solves` | `ewridge`, `lasso`, `huber`, `quantile` | [`ewridge`](#ewridge--ew-ridge-on-sufficient-statistics) |
| `target_gaps` | `ewridge`, `lasso` | [`ewridge`](#ewridge--ew-ridge-on-sufficient-statistics) |
| `window_size` | `ewridge`, `lasso` | [A hard window](../README.md#a-hard-window) |
| `standardize` | `ewridge`, `huber` and `quantile`; and `kalman`, `sgd` and `pa`, where it is the default | `ewridge` and `kalman` |
| `coef_min`, `coef_max`, `coef_sum` | `sgd`, `pa` | [`sgd`](#sgd--stochastic-gradient-descent) |
| `weight` | every model, though `rls`, `kalman`, `sgd`, `pa` and `ftrl` read it differently from a weighted mean | [Weights](../README.md#weights) |

What `half_life` decays differs by model:

| model | `half_life` decays |
|---|---|
| `ewridge` | the running sums the fit is solved from |
| `lasso` | the same sums, and the error that picks the penalty unless `select_half_life` is given |
| `rls` | `A` and `b`, as decayed sums rather than means |
| `kalman` | the observation-noise estimate and the features' standardization; `coef_half_life` sets how fast a coefficient may drift |
| `huber`, `quantile` | their running sums, kept per target |
| `sgd` | `weight_sum`, the running moments `standardize` scales by, and adagrad's sums, but not the coefficients |
| `pa` | `weight_sum`, each target's own weight and the running moments `standardize` scales by, but not the coefficients |
| `ftrl` | its per-coordinate sums, and its penalties with them |
| `holt` | the level; the trend has `trend_half_life` |

### `ewridge` — EW ridge on sufficient statistics

*API:* [`po.spec.ewridge`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.ewridge) — *Rust:* [`ewridge.rs`](../crates/online-core/src/ewridge.rs) — *Outputs:* [fields](OUTPUTS.md#ewridge)

Start with `ewridge`, ridge regression solved from running sums, and change
to another linear model when your data needs what it adds, such as
coefficients that drift (`kalman`).

`S_j` and `r_j` are target `j`'s running means of `z zᵀ` and `z·y_j`, kept
over the rows where `y_j` is present, and `W_j` is the weight behind them.
The coefficients are solved from them on a schedule:

```
W'   = λW + w                       weight_sum, over every row
W_j' = λW_j + w                     on the rows where y_j is present, λW_j on the others
S_j' = (λW_j·S_j + w·z zᵀ) / W_j'   r_j' = (λW_j·r_j + w·z·y_j) / W_j'
solve:  (S_j + ridge·D) β_j = r_j + ridge·D·β₀     D = I with a 0 at the intercept, β₀ = coef_prior (0 by default)
```

```python
rr = po.spec.ewridge(
    "rr", targets=["y"], features=["x0", "x1", "x2"], clock="t", gap_cap=300.0, half_life=600.0,
    ridge=[1e-6, 0.1],             # a grid of two ridge values, solved from one set of sums
    feature_sets={"mkt": ["x0"], "all": ["x0", "x1", "x2"]},   # named feature subsets, likewise
    solve_every=12.0,              # solve every 12 clock units ...
    max_rows_between_solves=100,   # ... and at least every 100 rows, whatever the clock does
    standardize=True,              # solve on the correlation matrix, then unscale the coefficients
    ridge_scale="mean",            # the default; "sum" makes the ridge a warm start that fades
    coef_prior=None,               # shrink toward a stated belief instead of toward zero
    target_gaps="own_rows",        # the default: a target null on some rows is fitted on its own rows
)
```

`ewridge`'s own parameters fall into six groups:

| group | parameters | refused or required |
|---|---|---|
| the grid | `ridge` as a list, `feature_sets` | |
| when it solves | `solve_every`, `max_rows_between_solves` | |
| how the ridge applies | `standardize`, `ridge_scale`, `coef_prior` | `ridge_scale="sum"` is refused beside `standardize`, a ridge or feature-set grid, and a window |
| a target null on some rows | `target_gaps` | |
| what the sums remember | `window_size` ([A hard window](../README.md#a-hard-window)), `session_shrink` | `session_shrink` needs `session` and `long_half_life`, and is refused beside `session_gap="reset"` and `window_size` |
| a wide fit | `gram_block_rows`, `gram_threads` | `gram_block_rows` is refused with `window_size=`, and where a solve happens every row |

**A grid of `ridge` values or of `feature_sets`, as in `rr`, adds a solve
per value and no update of the sums.** Only the named sets are fitted, so
to keep a fit on every feature, name a set that holds them all, such as
`"all"`.

**To keep the coefficients current at every row, give
`max_rows_between_solves=1`.** Otherwise `ewridge` predicts with the
coefficients of its last solve. Under a finite half-life it solves once the
weight learned since then reaches `ln 2 / 50` of the weight the fit holds:
about 1.4 %, or every `half_life/50` of clock in steady state. To solve on
the clock instead, give `solve_every` in clock units. `half_life=inf` and
`lam=` solve on every row, so when the fit need not be current at every
row, give them a `solve_every`. On 6M rows × 20 features from a parquet stream,
`solve_every=1000` takes 1.4 s instead of 10.7 s, with the coefficients at
most 1000 rows out of date.

**`ewridge` solves by Cholesky factorization, and retries a near-singular
system with a small jitter on the diagonal.** `bank.solve_failures()`
counts the solves that needed it. A count above zero points at constant or
collinear features, or too few rows for the features.

**To start from a belief that fades as data arrives, such as yesterday's
fit, give it as `coef_prior` with `ridge_scale="sum"`.** The ridge then
sits on the decaying sum scale, as `rls`'s does, and pulls every
coefficient toward `coef_prior`, the intercept included, until the evidence
takes over. `S` is a mean, so under the default `"mean"` the ridge is a
penalty per observation that never fades: use it to hold the slopes near
`coef_prior`, or near zero, for good.

**To make one `ridge` value mean the same whatever the features' units,
give `standardize=True`:** the solve then runs on the correlation matrix,
and drops a feature whose variance is zero instead of blowing up. Under the
default `standardize=False`, `ridge` is in the features' squared units.

**When a target is null on some rows, `target_gaps` chooses which rows its
`S_j` covers:**

| `target_gaps` | `S_j` covers | the targets of one spec keep |
|---|---|---|
| `"own_rows"`, the default | the rows the target is present on, so its fit is the fit of the frame with those nulls dropped | one `k × k` matrix per pattern of missing rows |
| `"pairwise"` | every row, the way pandas' `DataFrame.cov` takes a pairwise-complete covariance, with `r_j` centred at the target's own rows' means | one matrix |

The two modes agree when the gaps are independent of the features. When
they depend on the features, as with a target present only on trade rows
between market-data rows, keep the default. There `"pairwise"` would scale
each slope by the feature's variance on the target's rows over its variance
on every row.

**To start each session from a blend with the long run, give
`session_shrink=f` and `long_half_life=`, beside a `session` column and a
`session_gap` that is a number:**

```python
blend = po.spec.ewridge(
    "blend", targets=["y"], features=["x0", "x1"], clock="t", gap_cap=300.0, half_life=100.0,
    session="session",       # the column whose change starts a session ...
    session_gap=60.0,        # ... and the clock step applied there; "reset" is refused beside a blend
    long_half_life=5000.0,   # a twin of the sums, forgetting at this longer half-life
    session_shrink=0.3,      # at each session change, the moments become 0.7 of today's and 0.3 of the twin's
)
```

The blended moments keep today's weight, Kish's effective sample size and
prior scale, so `weight_sum`, the warm-up gates and the solve schedule do
not move.

**For a fit over hundreds of features, give `gram_block_rows=256`.**
Updating `S` takes O(k²) time per row for `k` features, and at a thousand
features that update takes most of the run. The option holds 256 rows back
and adds them to `S` with one matrix product
([docs/PERFORMANCE.md](PERFORMANCE.md) §18). A solve brings `S` up to
date before it runs, so solve rarely: under `half_life=inf`, give a
`solve_every`, as `wide` does:

```python
wide = po.spec.ewridge(
    "wide", targets=["y"],
    features=["x0", "x1", "x2"],   # three here, so the block runs; the speed-ups are for 256 to 2,000
    half_life=float("inf"), solve_every=1e9, max_rows_between_solves=1000,
    gram_block_rows=256,           # add 256 held rows to S in one matrix product
)
```

| features | rows per second, against one row at a time, on one thread |
|---:|---:|
| 256 | 5.1× |
| 1,000 | 6.6× |
| 2,000 | 5.9× |

With a solve every 512 rows, each speed-up falls to about 4×, because the
solve takes the same time either way. Blocking leaves `weight_sum`, the
timing of every prediction and chunk invariance unchanged, and the
coefficients agree with the row-by-row fit to rounding.

**To run that product on several cores, give `gram_threads=8` as well.**
The threads come from the pool the bank runs its groups on, and the output
is the same to the bit at every thread count. At 2,000 features a bank
learned 3.9 times as fast on eight threads
([docs/PERFORMANCE.md](PERFORMANCE.md) §37). A bank whose groups
already keep every core busy gains nothing from it.

**With decay off and `ridge=0`, `ewridge` fits ordinary least squares over
every row seen, in any row order.** Its coefficients match
`numpy.linalg.lstsq` to 2e-13, fed forwards or backwards. On 6M rows × 20
features from a parquet stream, memory peaks at 1.4 GB, against 3.97 GB
for `lstsq` on the same rows.

### `rls` — recursive least squares

*API:* [`po.spec.rls`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.rls) — *Rust:* [`rls.rs`](../crates/online-core/src/rls.rs) — *Outputs:* [fields](OUTPUTS.md#rls)

Use `rls` when the coefficients must be current at every row: it updates
them on every row in O(k²) time, without a solve. Its fit equals
`ewridge(ridge_scale="sum")` solved on every row, to better than 1e-9.

`A` and `b_j` are the decayed sums of `zzᵀ` and `y_j z`, started from the
prior strength `delta`, the classic RLS name for `P₀ = I/δ`:

```
A ← λA + w zzᵀ       b_j ← λb_j + w y_j z        β_j = A⁻¹ b_j        decayed sums, not means
A₀ = delta·I         b₀ = delta·coef_prior
```

Because `A` is a sum, `delta` is in the features' squared units summed over
rows, a row's weight is on the sum scale, and a heavier stream outweighs
the starting prior sooner ([Weights](../README.md#weights)). It is `ewridge`'s `ridge`
under `ridge_scale="sum"`, not its default `ridge`, which never fades, and
it penalizes every coefficient, the intercept included. The targets share one Cholesky factor of `A`, so a row
with any target null is scored but teaches none of them. So when the
targets go missing on different rows, give each its own `rls` spec.
Otherwise one spec holds them all.

```python
rls = po.spec.rls(
    "rls", targets=["y"], features=["x0", "x1"], clock="t", gap_cap=300.0, half_life=600.0,
    delta=1e-3,                    # A starts at delta * I (default 1)
)
```

**`rls` keeps the Cholesky factor of `A`, updated by Givens rotations** (the
*square-root form*). The textbook recursion on the inverse `P` loses
symmetry to rounding by a factor of `1/λ` per row, and one extreme row can
cancel `P` and freeze a coefficient for good.

**A feature that holds one value for long winds `rls` up, so use `ewridge`
for a long-running stream whose features can go quiet.** The prior `delta`
fades with the sums, so nothing holds a coefficient in a direction the data
stop exciting. A feature held at a value `c` other than 0 shows the fit
only `intercept + c·slope`. The information that tells the slope from the
intercept then decays by `λ` a row, with nothing to renew it. Once it is
under a rounding step, rounding sets the slope, and the slope wanders. A
feature constant from its first row winds up the same way. Held at exactly
0, it excites nothing in that direction, and the slope stays where it was.
`ewridge`'s default `ridge` is a penalty on the means that never fades, and
it centres each feature, so it does not wind up. Measured with
`half_life=20` (a row a clock unit), `delta=1` and noise 0.17, on a feature
held for 150 half-lives after 300 rows of moving, then moving again for 10:

| the run | the slope while held (0.5 in the data) | it first passes 1 | `pred − y` once it moves: rms / worst |
|---|---|---|---|
| `rls`, held at 0.87 | −3.1e13 to 3.0e13 | 54 half-lives in | 3.1e11 / 4.4e12 |
| `rls`, held at 1,000 | −8.3e9 to 8.1e9 | 43 | 8.3e7 / 1.2e9 |
| `rls`, held at 1e8 | −4.6e5 to 4.5e5 | 62 | 4.6e3 / 6.5e4 |
| `rls`, constant at 0.37 from the first row | −3.0e13 to 3.1e13 | 50 | 7.0e11 / 9.9e12 |
| `rls`, held at 0 | 0.41 to 0.53 | never | 0.20 / 0.74 |
| `ewridge` (`ridge=1e-6`, solved every row), any of these | 0 to 0.51 | never | 0.19 to 0.20 / 0.37 to 0.74 |

The fit held for some 30 half-lives in every case measured. Constant from
the first row at 1,000 and 1e8, the slope first passed 1 at 59 and 76
half-lives.

### `lasso` — lasso path, its penalty chosen as it runs

*API:* [`po.spec.lasso`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.lasso) — *Rust:* [`lasso.rs`](../crates/online-core/src/lasso.rs) — *Outputs:* [fields](OUTPUTS.md#lasso)

Use `lasso` when only some features should carry weight: its L1 penalty
sets the others to exactly zero. It fits a path of penalties from
`ewridge`'s centred running sums.

It runs coordinate descent on the standardized sums, warm-started along the
path and from one solve to the next. `C` is the features' correlation
matrix, and `c_i` is feature `i`'s covariance with the target over the
feature's standard deviation. For each penalty `l` in `lasso_path`, the
sweep repeats until no coefficient moves by more than `tol`:

```
ρ_i = c_i − Σ_{j≠i} C_ij β_j
β_i = soft(ρ_i, l·l1_ratio) / (C_ii + l·(1 − l1_ratio))        soft(v, t) = sign(v)·max(|v| − t, 0)
```

After the sweeps the slopes are unscaled, and the intercept is `ȳ − m·β`,
with `m` the features' means. The target is centred but not scaled, so the
L1 threshold `l·l1_ratio` is in the target's units. So scale `lasso_path`
with the target: for a pure lasso, a target ten times larger needs
penalties ten times larger to zero the same coefficients. A descent that
runs out of its `max_iter` sweeps is counted in `bank.solve_failures()`,
so if that count is above zero, give a larger `max_iter`.

```python
las = po.spec.lasso(
    "las", targets=["y"], features=["x0", "x1", "x2"], clock="t", gap_cap=300.0, half_life=600.0,
    lasso_path=[0.1, 0.01, 0.001],   # the penalties, decreasing
    select_half_life=None,           # the half_life of the error that ranks them (default: the model's)
    l1_ratio=1.0,                    # below 1: an elastic net
    max_iter=100, tol=1e-10,         # the defaults
    target_gaps="own_rows",          # which rows a target null on some is fitted from, as for ewridge
)
```

**At each row, `lasso` reports in `penalty_selected_<target>` the penalty
whose exponentially weighted out-of-sample squared error is lowest so far,**
as it stood before the row. Predictions for every penalty are computed
anyway, so the selection adds no work of its own. To rank the penalties on
the plain mean of that error, give `select_half_life=inf`.

**Under a `window_size`, a feature that stays constant inside the window
gets a coefficient of exactly zero,** and the error that picks the penalty
covers only the window ([A hard window](../README.md#a-hard-window)).

### `kalman` — random-walk-β dynamic linear model

*API:* [`po.spec.kalman`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.kalman) — *Rust:* [`kalman.rs`](../crates/online-core/src/kalman.rs) — *Outputs:* [fields](OUTPUTS.md#kalman)

Use `kalman` when a relationship moves faster than a half-life can follow,
or when a coefficient should be forgotten once the rows stop supporting it.

Per target `j`, `β_j` are the coefficients and `P_j` their covariance.
`r_i` is coefficient `i`'s `revert_half_life`, and `φ_i = 2^(−Δclock/r_i)`
its entry of `Φ`. `Q` is the process noise, which `coef_half_life` sets, and
`R_j` the observation noise, the target's exponentially weighted residual
variance unless `obs_var` is given:

```
β_j ← Φβ_j    P_j ← ΦP_jΦ              Φ = diag(2^(−Δclock/r_i)), every row
P_j ← P_j + Q·D_j²,  D_j ← 0           on a row that observes y_j; D_j the clock since the last
                                       (bounded on a reverting coefficient, below)
s   = zᵀP_j z + R_j/w                   K   = P_j z / s
β_j ← β_j + K(y_j − zᵀβ_j)              P_j ← P_j − K zᵀP_j
```

A row's weight scales its observation's precision: the rule divides `R_j`
by `w`, so with a fixed `obs_var=` the observation's variance is
`obs_var / w`.

**Until a target has a residual variance, each row takes its noise from
its own innovation:** `R_j = (y_j − zᵀβ_j)²`, computed before the update,
so the prediction stays out of sample. The process noise reads the same
number. The residual variance starts at the target's first prediction. A
row whose innovation is exactly 0 says nothing about the size of the noise,
and changes nothing.

**The prior variance comes from the squared innovations:** `p0` times a
noise estimate, so the default `p0=1` gives a prior as uncertain as one
observation. Unstandardized, the first row with a noise sets `P_j =
p0·R_j·I` before its correction. Standardized, each coefficient waits for
its feature's scale: its prior is set on the first row that observes the
target once the feature's variance is above 0. The estimate there is the
mean squared innovation over the target's rows so far, once three rows give
it. One squared innovation falls below 1% of its mean one time in twelve,
and with no process noise a prior that small pins the fit for good. Since
the noise and the prior both come from the data, the warm-up does not
depend on the target's units. With a fixed `obs_var=`, the noise is known
from the start.

This code uses `df` from [Example data](../README.md#example-data):

```python
revert = po.spec.kalman(
    "k", targets=["y"], features=["signal_a", "signal_b"], clock="t", gap_cap=10.0,
    half_life=200.0,                   # how fast the noise estimate and the standardization forget
    coef_half_life=100.0,              # how fast a coefficient may drift; or q, never both
    revert_half_life=[float("inf"), 50.0, 50.0],   # inf leaves the intercept alone; the slopes revert at 50
    standardize=True,                  # the default
    p0=1.0,                            # the default: the prior variance is p0 times the first noise estimate
    share_p=False,                     # the default: one P per target; True keeps one for all, driven by their mean σ²
)
fit = po.ModelBank([revert]).fit_predict(df)
```

**By default each coefficient drifts as a random walk: between observations
its estimate holds and its variance grows.** With `h_i` its
`coef_half_life`, a row that observes the target `D` clock units after its
last observation adds `σ²(ln2 · D / h_i)²` to that variance on standardized
features. That matches EW-RLS's steady state at any spacing. The noise is
charged once for the whole of `D`, since `D²` does not split into the
squares of its parts. Charged per row, a row of weight 0 inside a gap moved
every later prediction, and a target seen on one row in ten forgot 3.6
times slower on the clock. For a rate per coefficient, give
`coef_half_life` a list with the intercept's entry leading, and an entry
of `inf` pins its coefficient. To set the process noise outright, give
`q=`, whose `q_i` is added as `q_i · D²`.

**`coef_half_life` is each standardized direction's memory, with the other
directions held still.** Under correlated features, a direction of the
design's correlation with eigenvalue `λ` is learned with a memory of about
`h/√λ` (Ljung and Gunnarsson, 1990). At a correlation of 0.9 and `h = 50`
that measured 37.5 and 160.5 clock units, against 36.3 and 158.1. EW-RLS
forgets every direction at `h`. The default `R`, the EW residual variance,
grows at a step in the truth, so a step is learned about 1.45 times slower
than with the true noise. Under `standardize` the drift is per standard
deviation: in the caller's units a slope's walk has variance `q_i·D²/s_i²`
per observation, `s_i` the feature's spread as it stands.

**For a regressor that is active only now and then, give its coefficient a
finite `revert_half_life`.** The coefficient is then pulled toward zero on
every row, while the rows that support it pull it back (an AR(1) prior). So
the effect is forgotten between its bursts, and cannot persist through a
run of null targets. For a regressor that is always active, keep the default
`inf`: the pull would settle a persistent effect below its true size, the
more so the shorter the reversion half-life. A reverting coefficient's
process noise over a gap `D` is `q_i·g_i(D)²`, with `g_i(D) =
(1 − 2^(−D/r_i))/θ_i` and `θ_i = ln 2/r_i`. That is `q_i·D²` for a short
gap, and never more than `q_i/θ_i²`, so the coefficient's uncertainty stays
bounded across a run of null targets, where a random walk's grows without
bound.

**Under `standardize`, the default, the reversion pulls toward zero in the
standardized coordinates:** "no effect" for a slope, and "the target
averages zero" for the intercept. So unless the target averages zero, keep
the intercept's entry at `inf`, as in `[float("inf"), 50.0, 50.0]`.

**The filter follows the standardization from its first row.** The
coefficients and their covariance are held in the coordinates of an anchor:
the means and scales at the last re-map. The process noise and the priors,
defined in the means and scales as they stand, are carried into the
anchor's coordinates exactly. When a scale moves past twice or half the
anchor's, or a mean by more than the anchor's scale, the state is mapped to
the new coordinates, `b ← A b` and `P ← A P Aᵀ`. So no prediction moves
because the scaler did. Read through the scaler as it stood, as release
0.13.0 read it, its own wander under a half-life of 50 cost 223 noise
variances of out-of-sample error at R² 0.99998: one feature, 30,000 rows,
the median of five seeds. This filter's error stands 0.014 above the noise
there, as an unstandardized one's does. A map of more than 1024 times a
scale is not followed. The old reading, a coefficient per current standard
deviation, is this filter on a column z-scored in the stream with
`standardize=False`, as in
[Features in units of their spread](WINDOWS.md#features-in-units-of-their-spread).

**`predict` applies the reversion,** by the same `Φ` over the distance
from the last learned row, capped by `gap_cap`. A slope therefore keeps at
least `2^(−gap_cap/r_i)` of its value. To let a far prediction approach the
intercept, give a `gap_cap` that spans several reversion half-lives. Under
`standardize` the intercept is the target's level at the features' means.

With `standardize=False`, `q=0` and a fixed `obs_var=`, the filter is
exactly Bayesian linear regression, the ridge with penalty `1/p0`: river's
`BayesianLinearRegression` agrees to 1e-13. Standardized, each prior waits
for its feature's scale, so it is not that ridge.

**`se_coef` and `error_inflation` are exact under the model,** which says
the coefficients walk at the noise `coef_half_life` implies. On a truth that
does not move, the standard errors come out about √2 too large. With `R`
the residual variance, which already carries the estimation error,
`error_inflation` counts that error twice: the mean squared error is 0.96 of
`R·error_inflation²` at `coef_half_life=50`. Under `share_p`, each target's
`se_coef` reads the shared `P` times its own noise over the mean.

**Keep one `P` per target, the default, unless the targets share their
noise.** With a constant noise, `P/σ²` and the gain never read `σ²`, so a
shared `P` and one per target part only where the targets' noises move
apart. Give `share_p=True` for related targets, whose noises move together,
or where each target's own noise estimate is noisy, from a short or sparse
history, which the mean over the targets smooths. It also costs one `k×k`
update a row instead of one per target. On section 4 of
[docs/VALIDATION.md](VALIDATION.md), the next minute's return scored
R² -0.0110 shared against -0.0168 per target, and the next five minutes'
-0.0880 against -0.0746.

### `huber` / `quantile` — robust regression

*API:* [`po.spec.huber`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.huber) and [`po.spec.quantile`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.quantile) — *Rust:* [`robust.rs`](../crates/online-core/src/robust.rs) — *Outputs:* [huber](OUTPUTS.md#huber), [quantile](OUTPUTS.md#quantile)

Use these when one wild target must not move the fit: `huber` down-weights
the rows that miss by far, and `quantile` fits a conditional quantile, such
as the median, which outliers cannot pull. Both read each row's residual
against the fit *before* the row is learned, so both stay out-of-sample.
Both take `ewridge`'s `ridge=` as a single number, and refuse a list.

`r` is the row's residual, `σ` the plain exponentially weighted standard
deviation of the residuals, `δ` is `huber_delta` and `τ` the `quantile`
level. The *Gram* is `ewridge`'s `S`, and the cross-moment its `r_j`:

```
huber:     the ridge update at weight  w · min(1, δσ / |r|)
quantile:  one Newton step on the check loss, smoothed by a uniform kernel of half-width h = quantile_eps · σ
           |r| < h:  a least-squares row with target  y + 2h(τ − ½)
           |r| ≥ h:  adds  w · 2h · ψ_τ(r) · z  to the cross-moment and nothing to the Gram,  ψ_τ(r) = τ − 1{r < 0},
                     bounded so the fit at the row moves by at most the row's residual
           h is never narrower than (k/n)^{2/5} · σ, for the target's effective sample n
```

**Until a target has a `σ` above 0, `huber` down-weights no row.** That is
before its first residual, or while every residual so far is exactly zero:
there is nothing to judge an outlier against. `quantile` has no band to draw
there, and takes least-squares rows.

```python
hub = po.spec.huber("hub", targets=["y"], features=["x0", "x1"], clock="t", gap_cap=300.0, half_life=600.0,
                    huber_delta=1.5)      # a residual beyond huber_delta * sigma is down-weighted
med = po.spec.quantile("med", targets=["y"], features=["x0", "x1"], clock="t", gap_cap=300.0, half_life=600.0,
                       quantile=0.5,      # the level, strictly between 0 and 1: 0.5 is a median regression
                       quantile_eps=0.2)  # the band's half-width, in units of sigma (default 0.2)
```

**The band's rows, those with `|r| < h`, give the Newton step its
curvature.** So a much smaller `quantile_eps` narrows the band and slows
the convergence, and a much larger one smooths the quantile toward the
mean. The floor on `h` keeps a short half-life from leaving too few rows
inside the band.

`quantile` takes least-squares rows in four cases:

| case | `quantile` takes least-squares rows |
|---|---|
| a target's first rows | until the target has three rows per coefficient, its present rows counted one each, decayed, whatever their weights |
| after a gap or a reset | by the same rule, which rebuilds the fit |
| a band holding less than one row per coefficient | until the band holds rows again, which rebuilds a fit that a row near the input bound, `1e100`, has moved |
| a target with no `σ` above 0 | until a residual other than 0 gives it one, since the band is a width in units of `σ` |

**Both models are held to numpy references of their own recursions within
1e-14.** Against batch fits of the same objectives, scikit-learn's
`HuberRegressor` and statsmodels' `QuantReg`, they land close but not
equal, for three reasons:

| reason | applies to |
|---|---|
| each row's weight was set by the fit before it | both |
| `σ` is the plain spread rather than a robust one | `huber` |
| the check loss is smoothed over the band | `quantile` |

### `sgd` — stochastic gradient descent

*API:* [`po.spec.sgd`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.sgd) — *Rust:* [`sgd.rs`](../crates/online-core/src/sgd.rs) — *Outputs:* [fields](OUTPUTS.md#sgd)

Use `sgd` when each row must take only O(k) time: it takes one gradient
step per row and never solves. It is the only model here that takes count
targets (`loss="poisson"`). With `Gᵢ` adagrad's running sum of squared
gradients, a row takes this step:

```
eta = zᵀβ        p = link(eta)        d = dL/d eta
g₀ = d·w         gᵢ = d·zᵢ·w + l2·βᵢ     each clipped to ±clip_gradient; the intercept takes no l2
βᵢ -= lrᵢ·gᵢ     lrᵢ = learning_rate                           schedule="constant"
                     = learning_rate/(1 + weight_sum)^power    schedule="inv_scaling"
                     = learning_rate/(√Gᵢ + 1e-8)              schedule="adagrad"
```

**To set how many rows the coefficients remember, choose `learning_rate`.**
Every row's step moves the coefficients and the half-life does not decay
them, so under a constant rate they remember about
`1 / (learning_rate · E[z²])` rows, whatever the clock between rows. Under
`schedule="adagrad"`, `Gᵢ` decays on the clock, so the adapted rate opens
up again after a long gap.

The loss sets the link and the gradient:

| loss | link | `dL/d eta` | its constant |
|---|---|---|---|
| `squared` | identity | `p − y` | |
| `huber` | identity | `clamp(p − y, ±delta·s)` | `delta` is `huber_delta=`, 1.345 by default, in units of `s` as `huber`'s is |
| `quantile` | identity | `1{y < p} − τ` | `τ` is `quantile=`, the level, between 0 and 1 |
| `epsilon_insensitive` | identity | 0 within `eps·s_y` of `y`, else `sign(p − y)` | the tube's half-width is `eps=`, 0.01 by default, in units of `s_y` |
| `poisson` | log, for count targets | `p − y` | a negative target refuses the chunk, naming the row |
| `logistic` | sigmoid, for 0/1 targets | `p − clamp(y, 0, 1)` | `strict_binary=True` refuses a chunk with a label not 0 or 1 |

**`s` is the exponentially weighted standard deviation of the target's
out-of-sample residuals**, as `huber`'s `σ` is, and **`s_y` is the
target's own**, the spread of `y` around its exponentially weighted mean.
So a cut and a tube mean the same thing at any scale of the target. Until
the target has an `s`, the Huber row is a squared-loss row; until it has
an `s_y`, two weighted rows of different values, the tube has no width. In
the target's own units a tube of 0.1 never let a target in thousandths be
learned. `sgd` standardizes its features by default, as `kalman` does: one
learning rate has to suit every feature, and raw, features in hundreds took
the default fit's R² from 0.96 to below −70,000.

**Once the scaler has warmed up, the coefficients are held in the caller's
units.** Until it has learned 22 rows by Kish's count, the fit runs in its
standardized coordinates and is read out through its moments. From then on
each step is taken on the standardized row and mapped back by that row's
means and scales. So the scaler's moving moves no prediction. Read through
the moments instead, a half-life of 10 put the slope at 3.03 for a truth of
2 at R² 0.978; held, it is 2.02.

**The cut reads the residuals and the tube does not, because the fit starts
from zero.** Its first residuals are the target's whole level. Beyond the
cut the gradient is clipped, not zero, so a cut those residuals widen still
lets every row teach. Inside the tube the gradient is zero. A tube drawn in
the residuals' spread on a target at 1,000 with a spread of 2 was about 100
wide and held every row, and without decay it never narrowed.

**The tube's default, 0.01, keeps errors under 1% of the target's own
spread from moving the fit.** In units of the noise the tube is
`eps/√(1 − R²)` wide, `R²` the fit's: 0.067 noise standard deviations at
R² 0.978, and 2.2 at 0.99998. Inside it the gradient is zero, so a tube
many noise standard deviations wide holds the fit wherever it first lands.
Measured with one feature and no decay, as the out-of-sample error above
the noise in noise variances:

| R² | `inv_scaling` at 0.5, `eps=0.01` | `eps=0.1` | `constant` at 0.01, `eps=0.01` | `eps=0.1` |
|---|---|---|---|---|
| 0.978 | 0.015 | 0.010 | 0.042 | 0.027 |
| 0.99998 | 0.20 | 24 (0.7 to 214) | 0.81 | 17 (3.4 to 30) |

The sign-valued step moves the fit's level by the rate on each row of
weight 1, so at the default rate of 0.01 a target at a level of 1,000 is
100,000 rows away.

**Under `loss="poisson"`, keep `clip_gradient`, `1e3` by default,** because
through the log link one large count would make the next gradient
exponentially bigger. At ordinary scales it does not bind for the other
losses. A Poisson fit takes counts, so a negative target refuses the chunk
and names the row, as scikit-learn's `PoissonRegressor` refuses one. Taken
as it stood, `p − y` drove the prediction down to the link's floor,
`e^−30`, and held it there.
This code uses `df` from [Example data](../README.md#example-data):

```python
portfolio = po.spec.sgd(
    "portfolio", targets=["y"], features=["signal_a", "signal_b", "x0"], half_life=200.0,
    loss="squared",              # any loss in the table above
    learning_rate=0.01,
    schedule="constant",         # or "inv_scaling", or "adagrad"
    clip_gradient=1e3,           # the default
    coef_min=0.0,                # every slope at least 0 ...
    coef_sum=1.0,                # ... and the slopes summing to 1
    coef_every=1,
)
fit = po.ModelBank([portfolio]).fit_predict(df)
last = fit["portfolio"].struct.field("coef").drop_nulls()[-1]
assert min(last[1:]) >= 0.0 and abs(sum(last[1:]) - 1.0) < 1e-12   # the last row's slopes: at least 0, summing to 1
```

**To bound the slopes, give `coef_min` and `coef_max` (one number, or one
per feature, with `-inf` or `inf` for an open side) and `coef_sum` for
their total:**

| setting | what it expresses |
|---|---|
| `coef_min=0, coef_sum=1` | a long-only, fully invested portfolio |
| `coef_min=0` alone | a sign the model must respect |
| `coef_min` equal to `coef_max` | a slope pinned at a known value |

After every step the slopes move to the nearest point that satisfies every
bound (the Euclidean projection). The fit starts from zero, projected, so a
fit on the simplex starts at uniform weights. The intercept is never
constrained, and `coef` reports what the projection returned, in the
caller's units even under `standardize=True`.

### `pa` — passive-aggressive regression

*API:* [`po.spec.pa`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.pa) — *Rust:* [`pa.rs`](../crates/online-core/src/pa.rs) — *Outputs:* [fields](OUTPUTS.md#pa)

Use `pa` to step on every row without tuning a learning rate. Each row asks
the fit to come within `eps·σ` of its target, `σ` the target's own
exponentially weighted standard deviation, as `sgd`'s tube is drawn in. The
update is the smallest change that does so:

```
loss = max(0, |y − p| − eps·σ)      s = ‖z‖²
pa    τ = loss / s          pa1  τ = min(c, loss/s)      pa2  τ = loss / (s + 1/(2c))
β    += min(w, 1) · τ · sign(y − p) · z
```

In the target's units, a tube of 0.1 held a target in hundredths on every
row, and the model never learned it. In the residuals' spread, a fit from
zero on a target at 1,000 drew a tube about 100 wide from its first
residuals, and without decay stopped learning: R² −52. `σ` reads `y` alone,
so until the target has two weighted rows of different values the tube has
no width. `eps` is 0.01 and `c` is 1 by default.

**`c` is in the target's units.** `pa1`'s cap of 1 caps every step of a
target in thousands and none of a target in thousandths, so the figures
below are for a target of spread about 2. `pa2`'s `c` sits beside `s` and
is free of the target's units: under `"pa"` and `"pa2"` a target scaled by
`k` fits as the unscaled one, scaled by `k`.

**On a level that moves, raise `c` or difference the target.** The fit
starts from zero coefficients, and a `pa1` cap of 1 moves the intercept
about one unit a row. On a target falling from 1,000 to −1,000 at 0.5 a
row, at a half-life of 200 and noise 0.3, the intercept met the level near
row 670. With the level as a feature too, the capped steps, mapped into the
caller's units through the scaler, swung the fit far past the level
instead. A drift the intercept carries alone keeps a lag of about 2 at any
`c`, half of it the tube's, since `σ` holds the drift. Differencing the
target and the moving features, `y − y.shift(1)`, takes the level out of
the fit, and the level is the last row's `y` plus the predicted change.
Measured as `pred − y`:

| the run | row 800 | row 1,600 | every row within 3 from row |
|---|---|---|---|
| the defaults, the level a feature | 1,790 | −3.2 | 2,795 |
| `c=10`, the level a feature | 0.67 | 1.1 | 467 |
| `c=1000`, the level a feature | −0.52 | −0.60 | 11 |
| the defaults, the level the intercept's alone | 1.3 | 2.1 | never: 2.7 rms over rows 800 to 1,600 |
| differenced, the level read back | −0.73 | −0.05 | 12 |

**In units of the noise the tube is `eps/√(1 − R²)` wide**, `R²` the
fit's. A tube well inside the noise damps nothing, so only a cap that binds
damps. A wider tube damps too, while the cap does not bind: the fit moves
only on the rows in the noise's tails. A tube many noise standard
deviations wide holds the fit wherever it first lands. Measured with `pa1`,
one feature, a target of spread about 2 and no decay, as the out-of-sample
error above the noise in noise variances:

| R² | the tube at 0.01 | the defaults | `c=0.1` | `eps=0.1` | `eps=0.5` |
|---|---|---|---|---|---|
| 0.978 | 0.067 noise stds | 1.15 | 0.33 | 0.57 | 0.06 |
| 0.99998 | 2.2 noise stds | 0.14 | 0.14 | 65 (25 to 347) | 1,496 |

The default tube had the smallest worst regret over 108 streams and all
three modes. `pa` standardizes its features by default, with `sgd`'s
scaler, so `s` and `c` are not in the features' units either. Past the
scaler's warm-up its coefficients are held in the
caller's units, as `sgd`'s are, and an uncapped step still puts the row on
its tube's edge. A row weight below 1 scales the step, and a weight above 1 counts as 1.
Where outliers are possible, keep a `mode` that caps or damps the step:

| `mode` | the step |
|---|---|
| `"pa"` | moves the fit as far as one bad row demands |
| `"pa1"`, the default | is capped at `c` |
| `"pa2"` | is damped by `c` instead of capped |

```python
pa = po.spec.pa(
    "pa", targets=["y"], features=["x0", "x1"], half_life=200.0,
    mode="pa1",                  # the default
    c=0.1,
    eps=0.05,                    # within 0.05 of the target's std a row is "close enough", and nothing moves
)
```

**`pa`'s coefficients keep no running sums, so its `half_life` does not
decay the fit.** It decays `weight_sum` and each target's own weight, which
`min_weight` reads, and the scaler and the target's spread that `eps` is
in.

**When `coef_min`, `coef_max` or `coef_sum` bound the slopes, keep `c`
small.** The projection that follows each update keeps the step from
meeting the row's margin exactly, and a truth outside the allowed set is
never reached, so the fit keeps stepping against the bounds. A small `c`
keeps those steps small.

### `ftrl` — online logistic regression

*API:* [`po.spec.ftrl`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.ftrl) — *Rust:* [`ftrl.rs`](../crates/online-core/src/ftrl.rs) — *Outputs:* [fields](OUTPUTS.md#ftrl)

Use `ftrl` for a 0/1 target, or for a sparse fit that steps on every row
instead of solving. It runs FTRL-proximal (McMahan et al. 2013), a gradient
method whose L1 penalty zeroes a coefficient with too little evidence, and
whose per-coordinate rates adapt to each feature's history. Choose the use
with `loss`:

| `loss` | `pred` | the model |
|---|---|---|
| `"logistic"`, the default | a probability, with `resid = y - p` | logistic regression |
| `"squared"` | the linear prediction | a sparse linear regression with no solves and an L1 penalty |

Under the logistic loss a target outside `[0, 1]` is clamped into it. To
refuse such a target instead, give `strict_binary=True`, which refuses a
chunk with any target but 0 or 1 and names the row.

`b` is the coefficients, `zz`, `n` and `d` are the per-coordinate sums, and
`α` and `β` are its `alpha` and `beta`:

```
b_i  = 0 if |zz_i| ≤ l1 else −(zz_i − sign(zz_i)·l1) / (β/α + d_i + l2)
p    = sigmoid(zᵀb)
zz_i ← λzz_i      n_i ← λn_i      d_i ← λd_i      λ over the clock since the row that last taught the target
g_i  = (p − y)·z_i·w      s_i = (√(n_i + g_i²) − √n_i)/α
zz_i += g_i − s_i·b_i      n_i += g_i²      d_i += s_i
```

This code uses `df` from [Example data](../README.md#example-data):

```python
df_up = df.with_columns(up=(pl.col("ret") > 0).cast(pl.Float64))      # df with a 0/1 target: did ret rise?
rise = po.spec.ftrl(
    "rise", targets=["up"], features=["x0", "x1"], half_life=500.0,
    loss="logistic",             # the default
    alpha=0.1, beta=1.0,         # the learning-rate scale and its smoothing (the defaults)
    l1=0.01, l2=0.0,             # l1 zeroes a coefficient whose evidence is below it (defaults: l1=0, l2=1)
    strict_binary=True,          # this target is 0 or 1, so refuse anything else
)
fit = po.ModelBank([rise]).fit_predict(df_up)
p_up = fit["rise"].struct.field("pred_up")   # per row: the probability that ret rises, before the row
```

**`ftrl`'s penalties keep a fixed mass, while the evidence grows with the
weights and the rows' density.** So a heavier or denser stream overcomes
them sooner, and a shorter half-life shrinks the fit more. At the defaults,
with unit weights and rows one clock unit apart, they act in steady state
as a ridge of `(1 − λ)(β/α + l2)` on the mean scale. A constant target of 5
then settles at 4.65 at `half_life=100`, and at 4.96 at 1000. For a
penalty that weight and density leave alone, use
[`lasso`](#lasso--lasso-path-its-penalty-chosen-as-it-runs), whose penalty
is on the mean scale.

**`ftrl` does not centre a feature, so give it features near 0.** A
coefficient's error reaches the prediction multiplied by the feature's
value, so a feature at a level `L` carries it at `L` times its size. On the
stream of `crates/online-core/tests/held_values.rs`, with one of three
features at `L`, the squared loss's error is about `0.018·L`: 18.3 at 1,000
and at -1,000, and 1.8e6 at 1e8, where it is 0.75 at 0.5. Standardize or
z-score such a feature first, as
[Features in units of their spread](WINDOWS.md#features-in-units-of-their-spread)
shows. `ftrl` has no `standardize` because `l1` zeroes a coefficient in the
feature's own units: a scale inside the model would move which coefficients
`l1` zeroes as the feature's spread moved.

**A row whose target is null or whose weight is 0 leaves that target's fit
where it was,** because the sums wait for the next row that teaches it.
That row is scored with the fit as it stood, then ages the sums by the
whole clock since. The gap ages the evidence and not the penalties, so the
fit gives way to them: a constant 5 settled at `half_life=100` stands at
0.75 of itself after the row that ends a gap of one half-life.

With decay off, `d_i` is `√n_i/α`, which is river's FTRL: the
two agree to 1e-12, row for row, and Vowpal Wabbit's `--ftrl` agrees to its
single precision.

### `holt` — Holt's linear trend

*API:* [`po.spec.holt`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.holt) — *Rust:* [`holt.rs`](../crates/online-core/src/holt.rs) — *Outputs:* [fields](OUTPUTS.md#holt)

`holt` is the only linear model that takes no features: it extrapolates the
target's own level and trend. To measure what a model's features add, run
`holt` in the same bank, then compare the two models' `sigma` ([How wide
is its error?](DIAGNOSTICS.md#how-wide-is-its-error)), or let a
[`seqtest`](#seqtest--a-sequential-test-of-a-sign-by-betting) with `a` and
`b` say which predicts closer.
This code uses `df` from [Example data](../README.md#example-data):

```python
with_x = po.spec.ewridge("with_x", targets=["y"], features=["x0", "x1"], clock="t", gap_cap=600.0,
                         half_life=200.0, emit_sigma=True)     # sigma_y is written only when switched on
naive = po.spec.holt("naive", targets=["y"], clock="t", gap_cap=600.0,
                     half_life=200.0, emit_sigma=True)         # the same target, from its own past alone
gain = po.spec.seqtest("gain", targets=["y"], a="with_x", b="naive")   # evidence that the features predict closer
compared = po.ModelBank([with_x, naive, gain]).fit_predict(df)
```

Level and trend, `l` and `b`, are weighted means of what each row observes
and what the model forecast, with `W` and `V` the weight each has gathered.
`s` is the clock since the target was last observed, this row's step
included. So a row with a null target or weight 0 leaves the level and
trend where they were, and its step goes into the next observation's `s`:

```
λ_l = 0.5^(s/half_life)                        λ_b = 0.5^(s/trend_half_life)
pred = l + b·s
l'   = (λ_l·W·pred + w·y)/(λ_l·W + w)          W' = λ_l·W + w
b'   = (λ_b·V·b + w·(l' − l)/s)/(λ_b·V + w)    V' = λ_b·V + w
```

```python
baseline = po.spec.holt(
    "baseline", targets=["y"], clock="t", gap_cap=600.0,
    half_life=200.0,              # how fast the level forgets, in clock units
    trend_half_life=2000.0,       # how fast the trend forgets (default four times the level's)
    trend=True,                  # the default
)                                # coef is [level, trend] per target
```

**`holt` measures the trend per clock unit, and per second on a temporal
clock,** so on an irregular clock it extrapolates the right distance.

**The level's half-life is the spec's `half_life`, as every model's
decay is.** `level_half_life`, a second name it once had, is refused and
the message names `half_life`. To make the trend the whole history's
drift, give `trend_half_life=inf`.

**For a flat forecast, give `trend=False`:** simple exponential smoothing,
whose level is the target's exponentially weighted mean. It holds the
trend at zero, and refuses `trend_half_life`.

**`holt` has no seasonal term, so for a seasonal baseline, group the spec
on the phase.** Make the phase a column with Polars' `with_columns` in the
query before the bank, and name it as the spec's `group`. The bank then
fits one level and trend per hour.
This code uses `lf` from [Example data](../README.md#example-data):

```python
seasonal = po.spec.holt("seasonal", targets=["y"], clock="ts", gap_cap="1h",
                        half_life="2h", group="hour")         # one level and trend per hour of the day
by_hour = lf.with_columns(hour=pl.col("ts").dt.hour()).online.fit_predict([seasonal]).collect()   # the phase, as a column
```

## Moments and correlation

These models track how the columns you name vary and move together: their
means, variances and correlations, for a few columns or for thousands. Each
row's values are read from the state before that row, so they can serve as
features for that row without leaking it.

### `ew_cov` — exponentially weighted moments

*API:* [`po.spec.ew_cov`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.ew_cov) — *Rust:* [`ewcov.rs`](../crates/online-core/src/ewcov.rs) — *Outputs:* [fields](OUTPUTS.md#ew_cov)

Reach for `ew_cov` for the running means, variances and correlations of the
columns you name. It takes one O(k²) step per row, where Polars expressions
alone take O(k²) *passes over the data* for every pairwise exponentially
weighted correlation.

The model keeps `m` and `C`, the weighted means of `x` and of
`(x − m)(x − m)ᵀ`. It updates them in Welford's centred form, because the raw
form, `E[x xᵀ] − m mᵀ`, loses a variance when the columns sit far from zero:

```
W' = λW + w      a = λW / W'      b = w / W'      δ = x − m          (a + b = 1)
m' = m + b·δ     C'ᵢⱼ = a·Cᵢⱼ + a·b·δᵢδⱼ
varᵢ = Cᵢᵢ       covᵢⱼ = Cᵢⱼ      corrᵢⱼ = Cᵢⱼ / √(CᵢᵢCⱼⱼ)
```

**To compare a variance or a covariance with Polars', give Polars `ddof=0`
or `bias=True`.** The model's are the population form, `C` itself, with no
`ddof` correction, where Polars' `var` and `ewm_var` default to the
unbiased form.

Give `stats` the statistics each row writes, `["mean", "std", "corr"]` when
left out:

| in `stats` | what it reports | needs |
|---|---|---|
| `mean`, `var`, `std`, `cov`, `corr` | the moments, per column or per pair | |
| `partial_corr` | the correlation of two columns with all the others held fixed, read off `(C + s·prior·I)⁻¹` in O(k³) time, spent only when asked | `precision_prior` |
| `mahal` | the Mahalanobis distance | `precision_prior` |
| `lag_corr` | the correlation of one column with another `ℓ` learned rows back | `lags=` |

For a wide set of columns, give `stats=[]`, so that each row writes no
statistics, and read the moments from the state after the run with
`bank.gram("mv")`.
This code uses `df` from [Example data](../README.md#example-data):

```python
mv = po.spec.ew_cov(
    "mv", features=["x0", "x1", "x2"], clock="t", gap_cap=300.0, half_life=500.0,
    stats=["mean", "std", "corr", "partial_corr", "mahal"],   # what each row writes
    precision_prior=1e-6,        # needed by partial_corr and mahal; fades as data arrives
    mahal_quantiles=[0.99],      # adds mahal_q0.99; needs "mahal" in stats
    pca=1, pca_every=20,         # one principal component, refreshed every 20 units of t (a row each here)
)
scores = po.ModelBank([mv]).fit_predict(df).unnest("mv")   # mean_x0, ..., corr_x0_x1, ..., mahal, pc0_*
odd = scores.filter(pl.col("mahal") > pl.col("mahal_q0.99"))   # the joint outliers: every column in range, the combination not
first = scores.select("pc0_share", "pc0_loading_x0", "pc0_loading_x1", "pc0_loading_x2", "pc0_score")
```

**`mahal` measures the row's distance from the mean in standard deviations,
allowing for the correlations.** It is `√(δᵀ (C + s·prior·I)⁻¹ δ)`, with
`δ = x − m` and `s` a scale on the prior that decays with the co-moments.
With one column it is `|z|`. On Gaussian columns `mahal²` is only roughly χ²
with `k` degrees of freedom, since the moments are estimated and the prior
adds a little.

**To flag joint outliers without assuming a distribution, give
`mahal_quantiles=[0.99]` and filter the bank's output on
`mahal > mahal_q0.99`,** as `odd` does, to keep about one row in a hundred.
`mahal_q0.99` is the exponentially weighted 0.99 quantile of the past scores,
at the `half_life` and within 0.78%.

**Each eigendecomposition takes O(k³) time, so give `pca_every` to refresh
the principal components on a schedule.** It counts the clock's units, as a
regression's `solve_every` does: `pca_every=20` refreshes them every 20 units
of `t`, a row each here, and a temporal clock takes a duration such as
`"5m"`. `max_rows_between_pca` caps the rows between refreshes, and
whichever comes first refreshes. The default is every row, and the rows
between refreshes are scored on the last loadings. With `pca=1`, each row writes `pc0_var`,
`pc0_share` (of the trace), `pc0_loading_<feature>` and `pc0_score`. Each
refresh keeps the previous sign, so a loading never flips.

**`window_size` cuts the history off at a fixed age** ([A hard
window](../README.md#a-hard-window)). For a frame that fits in memory, Polars'
`df.rolling("t", period="3h").agg(...)` with an exponential weight gives the
same numbers to 1e-14, but recomputes every window, in `O(n·W)` time, where
the spec takes `O(n)`. So use the spec for a stream, a state to save, or
speed: measured, it ran 24 times as fast at a 74-row window and 1,100 times
at a 4,680-row one.

**Give `lags` to see how each column moves with the others `ℓ` learned rows
back in its group.** With `W` and `m` the weight and mean before the row, and
both deviations taken against that mean:

```
C_ℓ' = a·C_ℓ + a·b·(x_t − m)(x_{t−ℓ} − m)'      the co-moments' a and b, so lag 0 would be C exactly
```

Give `lags` as a strictly increasing list of whole numbers `>= 1`. Add
`"lag_corr"` to `stats` to write a correlation for each lag and each *ordered*
pair, since a lagged matrix is not symmetric: a leading b is not b leading a.

**A session change, or a clock gap beyond `gap_cap`, empties the ring of past
rows that the lags read**, because after either, "the row `ℓ` back" no longer
means a row `ℓ` ago. The means, the co-moments and `weight_sum` stay. A
zero-weight row ages the matrices without entering the ring.
This code uses `df` from [Example data](../README.md#example-data):

```python
lagged = po.spec.ew_cov(
    "lagged", features=["x0", "x1"], half_life=500.0,
    lags=[1, 5],                 # in learned rows within the group
    stats=["corr", "lag_corr"],   # lag_corr_<a>_<b>_l<l> for each lag and ordered pair
)
bank = po.ModelBank([lagged])
lead = bank.fit_predict(df).unnest("lagged")
moments = bank.gram("lagged")[0]   # the same numbers from the state: "lags", and "lag_comoments" as an (L, k, k) array
```

### `marginal` — every pair's moments, kept in the state

*API:* [`po.spec.marginal`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.marginal) — *Rust:* [`marginal.rs`](../crates/online-core/src/marginal.rs) — *Outputs:* [fields](OUTPUTS.md#marginal)

Reach for `marginal` to screen thousands of features against a few targets.
It keeps the exponentially weighted moments of each (feature, target) pair on
its own, with `ew_cov`'s arithmetic, so a pair's `corr` equals a two-column
`ew_cov`'s to the bit. For `p` features and `T` targets that takes O(p·T)
time per row, where one `ew_cov` over all the columns would take
O((p + T)²). Each row writes only `weight_sum`, and `bank.marginal()` reads
the pairs from the state as a table after the run.

Each pair (feature `j`, target `t`) keeps its own `m_x`, `S_xx` and `S_xy`,
over its target's rows. Per target `t`, on a row where `y_t` is present, with
`W_t` the weight behind that target before the row:

```
W'_t = λW_t + w        a = λW_t / W'_t        b = w / W'_t        Q'_t = λ²Q_t + w²
S'_yy = a·S_yy + a·b·(y_t − m_y)²             S'_xx = a·S_xx + a·b·(x_j − m_x)²
S'_xy = a·S_xy + a·b·(x_j − m_x)(y_t − m_y)    m' = m + b·(value − m)
```

This code uses `df` from [Example data](../README.md#example-data):

```python
pairs = po.spec.marginal("pairs", targets=["y", "ret"],
                         features=["x0", "x1", "x2", "signal_a", "signal_b"],
                         clock="t", gap_cap=300.0, half_life=500.0, group="stock_id")
bank = po.ModelBank([pairs])
bank.fit_predict(df)                             # the output record holds weight_sum alone
table = bank.marginal("pairs")                   # one row per (group, instance, feature, target)
one_stock = bank.marginal("pairs", group="b0")   # 10 rows here: five features by two targets
# the columns of table:
# weight_sum                   the target's W_t, the weight behind its pairs
# n_kish                       W_t² / Q_t: the count of equally weighted rows that carry the same information
# mean_x, var_x, mean_y, var_y, cov    the pair's moments, population form
# corr                         cov / sqrt(var_x * var_y)
# beta                         cov / var_x: the slope of the target on that feature alone
# t                            corr * sqrt((n_kish - 2) / (1 - corr²)): the t-statistic at the Kish sample size
```

For unit weights a step `Δclock` apart, `n_kish` tends to
`(1 + λ)/(1 − λ)`, where `λ` is that step's decay.

**A null target ages its own pairs (`W_t ← λW_t`, `Q_t ← λ²Q_t`) and teaches
them nothing.** A null feature skips the row, as in every model.

**`corr`, `beta` and `t` are null until the target's `W_t` reaches
`min_weight`**, 3 by default here, because two rows give a correlation of ±1
whatever the data. A statistic is null wherever it is undefined: all three
for a constant feature, and `t` at `n_kish ≤ 2`.

**Use `t` to rank pairs, and never as a p-value**, because the rows are
neither independent nor Gaussian. `t` is built on `n_kish`, the right count
for unequal weights, which says nothing about rows that resemble their
neighbours. So on a smooth stream it claims evidence that is not there:
over 1,000 pairs of *independent* AR(1) series with `φ = 0.9` and `0.8`,
3,000 rows each, `|t|` passed 1.96 on 45% of them.

Five settings extend the pairs, the first two with views that are off by
default:

| setting | what it adds |
|---|---|
| `lags`, with `serial_rule` | the pair's moments at each lag: a count that allows for rows that resemble their neighbours, and lead/lag correlations |
| `bins` | the target's moments inside each bin of a feature: the relations a correlation cannot see |
| `window_size` | the pairs over a recent stretch |
| `feature_moments="shared"` | speed: one feature moment for every target, and the same table where every target is on every row |
| `shards` | speed: one wide spec on every thread, and the same numbers at any count |

When a group closes, `bank.closed_groups()` carries the two views as
`pair_*` columns: `pair_split_gain` as a list over the pairs, and
`pair_lag_corr_xx` and `pair_bin_n` as lists of lists.

**To correct the count for rows that resemble their neighbours, give `lags`
and a `serial_rule`.** Each pair then keeps its moments at each lag, the
statistic `ew_cov(lags=)` computes, to the bit, under
[`ew_cov`'s](#ew_cov--exponentially-weighted-moments) rules for the ring.
`serial_rule` turns them into a count that allows for the resemblance, after
Bartlett (1935): `n_serial = n_kish / (1 + 2 Σ_l ρ_x(l)·ρ_y(l))`. Under
`"geometric"`, `|t_serial|` passed 1.96 on 5.4% of those pairs of AR(1)
series, the 5% the threshold stands for.

| `serial_rule` | the sum over the kept lags | suits |
|---|---|---|
| not given | not taken: `n_serial` and `t_serial` are null | |
| `"truncated"` | as it stands; null where the bracket reaches zero or below, as for two series whose autocorrelations have opposite signs | dense lags |
| `"bartlett"` | lag `l` weighted `1 − l/(L + 1)`, `L` the longest kept lag, as Newey and West weight it, so the noisiest lags count least | dense lags, where the long ones are noisy |
| `"geometric"` | `ρ(l) = φˡ` fitted to each series, and the tail summed in closed form, `2φ_xφ_y/(1 − φ_xφ_y)`; `phi_x` and `phi_y` are reported | exponentially weighted series, whose lags need not be dense |

This code uses `df` from [Example data](../README.md#example-data):

```python
serial = po.spec.marginal(
    "serial", targets=["y"], features=["x0", "x1"], half_life=500.0,
    lags=[1, 2, 3, 5, 8],        # the pair's moments at each lag, in learned rows within the group
    cross_lags=[1],              # the lead/lag terms at these lags only; the default is every lag, [] none
    serial_rule="geometric",     # how the lagged correlations become a count correction
)
bank = po.ModelBank([serial])
bank.fit_predict(df)
table = bank.marginal("serial")
# the columns lags add:
# lag_corr_xx, lag_corr_yy      each series' own autocorrelation, one entry per lag
# lag_corr_xy, lag_corr_yx      the feature now against the target l rows back, and the reverse, one entry per cross lag
# n_serial                    n_kish over the serial_rule's bracket
# t_serial                    the same statistic as t, against that count
# phi_x, phi_y                the fitted per-row decays, under serial_rule="geometric"
```

**Read lead and lag from the cross terms only when the target is measured at
the same time as the feature.** Then a feature whose `lag_corr_yx[0]` beats
its `corr` leads its target, and one whose `lag_corr_xy[0]` does follows it.
Against a target that looks ahead, read `corr` alone. There every timely
feature built from the same news shows `lag_corr_xy` above `corr`, because the
target `l` rows back is built partly from the feature's newest `l` rows.

**Give `bins` to see the relations a correlation cannot.** A feature can be
strongly related to a target with `corr` at zero: a threshold, a V, a
saturation. With `bins`, each pair keeps the target's moments inside each bin
of the feature, and all three become visible. The edges are fixed once and
never move, set by one of three rules:

| the edges | how they are set |
|---|---|
| `bin_rule="quantile"` | learned from the first `bin_warm_rows` rows (default 1,000), by weighted quantile |
| `bin_rule="fixed"` | equal widths |
| `bin_edges=` | given outright, a list per feature or a dict by name: exact and comparable across runs, and refused beside `bins`, `bin_rule` and `bin_warm_rows` |

The warm-up rows are replayed once the edges are set, so none is lost.
This code uses `df` from [Example data](../README.md#example-data):

```python
binned = po.spec.marginal(
    "binned", targets=["y"], features=["x0", "x1"], half_life=500.0,
    bins=16,                     # bin each feature and keep the target's moments inside each bin
    bin_rule="quantile",         # equal weight per bin, the edges learned by weighted quantile
    bin_warm_rows=200,           # the rows the quantile edges are learned from
)
bank = po.ModelBank([binned])
bank.fit_predict(df)
table = bank.marginal("binned")
# the columns bins add:
# bin_edges                   the feature's edges, fixed once and never moved
# bin_n, bin_mean_y, bin_var_y   the target's weight, mean and variance in each bin: the response curve
# split_gain                  the fraction of the target's variance removed by the best single cut
# split_at                    where that cut falls, in the feature's units
# split_gain_t                the t a corr would need to match that gain, at n_serial where there is one
```

**Read `split_gain` against `corr²`, and `split_gain_t` as a ranking.**
`split_gain` is a regression stump's R², so its excess over `corr²` is the
nonlinear surplus. As a p-value, `split_gain_t` would overstate the evidence,
because the cut was chosen as the best of the candidates.

**A feature keeps only the bins it can support.** Under
`bin_rule="quantile"`, a value that carries more than a bin's share, such as
an indicator's zero, fills a bin of its own, and the rest share what is left.
So the 5% of rows that carry the signal are not lost among the zeros. A
binary feature has two bins, and a constant one has a single bin and no
split.

**Binning keeps `O(bins)` of state per pair, and takes one search of the
edges per feature per row,** so ten thousand columns bin in the pass that
gives their `corr`. A spec whose held rows or histogram would pass 256 MiB,
per group and per half-life, is refused when it is built. To allow more,
give a larger `bin_budget`, or `float("inf")` for no limit.

**Give a screen a `window_size` when a relation may flip sign**, because two
regimes of opposite sign average to nothing over a long history. The window
truncates every pair moment, so `corr`, `beta` and `t` describe the rows
inside it ([A hard window](../README.md#a-hard-window)). `weight_sum`, in the record and
in the table, is the weight inside the window. A window refuses `bins`, so
bin the features in a second spec without one.

**To use `lags` under a window, give `window_lags=True`; otherwise the spec is
refused.** Each snapshot then holds the lag moments beside the pair moments:
twice its size at one lag, and six times at five with the default cross lags.
A windowed lag moment is an estimate, while the windowed pair moments are
exact.

**With several targets present on every row, give
`feature_moments="shared"` for speed.** Each feature then keeps one mean and
variance, over every learned row, in place of one per pair over its target's
rows. At 20,000 pairs that runs 2.7 times as fast at ten targets, with a
state under half the size and the same table to the bit ([PERFORMANCE
§27](PERFORMANCE.md#27-marginals-shared-feature-moments-e72-task-125-2026-09-29)).
Where a target is absent on some rows, `var_x` and the centre of `cov` then
come from every row. That is a different estimator, sound only where the
absence says nothing about the feature, so otherwise keep the default. It
takes `lags`, and refuses a `window_size`.

**Give `shards="auto"` to spread one wide spec over the threads.** The bank
runs groups and specs in parallel ([Parallelism](../README.md#parallelism)), so a wide
`marginal` on one group is otherwise one thread's work. `shards=10` splits
its pairs into ten ranges of features, each run on a thread of its own, a
batch of rows at a time. `"auto"` sizes the split to the width and the pool,
and leaves whole a narrow spec, or a windowed one that snapshots every
row, the default, whose batch is a single row. The numbers are the same
to the bit at any count, so a saved bank resumes under any count. On 14
threads
([PERFORMANCE §25](PERFORMANCE.md#25-a-wide-marginal-split-across-the-pool-e73-task-126-2026-09-25)):

| the spec | `"auto"` ran the bank |
|---|---|
| 10,000 features, nine targets, lags and bins | 4.9 times as fast |
| the moments of one target alone | 1.2 times as fast, since the bank's own work on each row does not split |

### `deco` — one correlation for the whole matrix

*API:* [`po.spec.deco`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.deco) — *Rust:* [`deco.rs`](../crates/online-core/src/deco.rs) — *Outputs:* [fields](OUTPUTS.md#deco)

Reach for `deco` (Engle & Kelly 2012) for one number that moves with the
correlation of many series. A stream cannot keep every entry of their
correlation matrix moving without O(m²) work a row. So `deco` replaces the
`m(m−1)/2` free entries of the matrix of `m` series with their average, the
*equicorrelation*, and estimates that in O(m) a row, for the whole matrix or
per block.

The row is standardised against the means and variances as they stood before
it, `r_i = (x_i − m_i)/√v_i`. With `S₁ = Σ r_i` and `S₂ = Σ r_i²` over the
`n` features that have a standardised value, the row's estimate `u` is Engle
and Kelly's Lemma 2.3. The reported level, `rho`, then follows `u` under one
of two dynamics, with `α` and `β` the `"linear"` dynamic's two weights:

```
u = (S₁² − S₂) / ((n − 1)·S₂)        = (mean of r_i·r_j over i ≠ j) / (mean of r_i²)

"ew":      W' = λW + w,  b = w/W'      ρ' = ρ + b·(u − ρ)                       ρ = u at W = 0
"linear":  ρ' = (1 − α − β)·ρ̄' + α·u + β·ρ      (ρ̄ the "ew" level)          ρ' = ρ at w = 0
```

| `dynamics` | `rho` is | rules |
|---|---|---|
| `"ew"` | the exponentially weighted mean of `u`, decayed on the model's clock: exactly what an `ew_cov(stats=["mean"])` over the rows that have a `u` would report | |
| `"linear"` | the paper's eq. 21, its intercept written as `(1 - alpha - beta) * rho_bar`, since a stream has no sample to fit a free one on. It steps once per row, as a DCC (dynamic conditional correlation) model does, so a capped gap moves it as far as one row's `α·u` | needs `alpha=` and `beta=`, each `>= 0`, with `alpha + beta < 1`. A row's weight reaches `rho` only through `rho_bar`, since the paper's recursion has no row weights |

**Give `blocks` for one number per block and one per pair of blocks**, the
useful middle between one correlation and all of them. Put every feature in
exactly one block, and at least two in each, or the spec is refused.
This code uses `df` from [Example data](../README.md#example-data):

```python
eq = po.spec.deco(
    "eq", features=["x0", "x1", "x2"], clock="t", gap_cap=300.0, half_life=500.0,
    dynamics="ew",               # the EW mean of u; "linear" needs alpha= and beta=
)
blocked = po.spec.deco(
    "blocks", features=["x0", "x1", "x2", "signal_a"], half_life=500.0,
    blocks={"fast": ["x0", "x1"], "slow": ["x2", "signal_a"]},   # every feature in exactly one block
)
out = df.online.fit_predict([eq, blocked])
# u             this row's own estimate, read before the row is learned from
# rho           the level as it stood before the row
# loglik        the row's Gaussian log-density in standardised coordinates under that level
# with blocks:  u_fast, u_slow, u_fast_slow and their rho_* twins, and one loglik over all of them
```

**Use `u` and `rho` as a signal that moves with the market's correlation, and
for the correlation itself read an `ew_cov`'s `corr`.** Both read low,
because `u` is a ratio of two averages, and the mean of a ratio is not the
ratio of the means. As the paper says, `E[u]` is about 0.20 for a true 0.30
at six columns. `rho`, an average of `u`, carries the same bias:
[docs/REGIMES.md §7](REGIMES.md#7-deco-an-equicorrelation-that-moves)
measures it settling near two-thirds of the truth.

**A column constant from its first row is left out of its block,** since it
has no standardised value. The block reads the correlation among its other
columns, and a block left with fewer than two has no `u`. `loglik` needs
every column, so leave a constant column out of `features`, or it nulls
`loglik` on every row.

### `rcov` — a block's realised covariance, robust to noise

*API:* [`po.spec.rcov`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.rcov) — *Rust:* [`rcov.rs`](../crates/online-core/src/rcov.rs) — *Outputs:* [fields](OUTPUTS.md#rcov)

Reach for `rcov` for each block's realised covariance from its tick returns,
robust to measurement noise and to series not observed together. It requires
`group`, naming the block column, and `group_close`, and writes each block's
estimate when the block closes, as that block's row of
`bank.closed_groups()`. Each row writes only `weight_sum`. A block's close
ends its memory, so `rcov` refuses a `half_life` or `lam`. A `weight` must be
0 or 1, since a sum over returns has no fractional row, and any other value
is an error naming the row.

The plain realised covariance, the sum of the returns' outer products, goes
wrong over real ticks in two ways. Each price is the efficient price plus a
measurement error, whose variance accumulates with every tick, and series not
observed together have their correlation pulled toward zero. Published
estimators remove both with sums over lags, which a stream can accumulate.
In the kernel, `Gamma_h` sums the products of returns `h` rows apart, up to a
bandwidth `H`. Give `kind` to pick an estimator.
[docs/REGIMES.md](REGIMES.md) §8 measures the three against each
block's truth:

| `kind` | the estimator | use it on |
|---|---|---|
| `"plain"` | `sum(x x')`, equal to `n` times an `ew_cov(lam=1)`'s uncentred second moment at close, to the bit: the cross-check, and the reference the other two are measured against | clean returns, where it has the smallest error |
| `"kernel"` | the multivariate realised kernel (Barndorff-Nielsen, Hansen, Lunde & Shephard 2011), `sum_h k(h/(H+1)) Gamma_h` with Parzen weights and jittered end points | returns with measurement noise, which barely moves its error |
| `"preavg"` | the modulated realised covariance (Christensen, Kinnebrock & Podolskij 2010): returns pre-averaged over `k_n = ceil(theta * block_rows^0.6)` rows | series not observed on every row, where it is the least biased |

**Compute the returns from prices with Polars' `diff()` in the query before
the bank, over each block with `.over("block")`,** so that no return spans two
blocks. The bank skips each block's first row, whose return is null. For a
series not observed on every row, carry its last price forward with
`forward_fill().over("block")` before the diff, and use `kind="preavg"`. Or
build a common grid with `po.stream.refresh_time` before the diff ([Series
that tick at their own times](../README.md#series-that-tick-at-their-own-times)), and
keep the kernel.

```python
rk = po.spec.rcov(
    "rk", features=["x0", "x1"],       # x0 and x1 stand in for two prices, differenced in `returns`
    group="block", group_close="monotone",
    kind="kernel",               # the realised kernel; or "plain" or "preavg"
    psd=True,                    # the default: clip any negative eigenvalue
    block_rows=2000,             # a sizing hint for the ring
    bandwidth=None,              # a fixed H; left out, set from the data
)
returns = by_block.with_columns(              # by_block: df with a block id rising every 100 rows
    pl.col("x0", "x1").diff().over("block")   # each row's change since the row before it, within its block
)
bank = po.ModelBank([rk])
bank.fit_predict(returns)
finished = bank.closed_groups()
# rcov, rcorr                  vech of the upper triangle: its entries as one flat list
# rcov_n, rcov_kind, rcov_bandwidth_used
# rcov_omega2, rcov_iv_sparse  the noise variance and sparse integrated variance behind the bandwidth
# rcov_iq                      a realised-quarticity proxy, labelled one
# rcov_psd_repaired            whether the estimate had to be made positive semi-definite; null where
#                              the repair could not run, on an estimate with an entry that is not finite
# a block too short to estimate from gives nulls, not an error

# A series not observed on every row: carry its last price forward within the block, then difference,
# and pre-average.
held = by_block.with_columns(pl.col("x0", "x1").forward_fill().diff().over("block"))
pre = po.spec.rcov("pre", features=["x0", "x1"], group="block", group_close="monotone",
                   kind="preavg", block_rows=100)
po.ModelBank([pre]).fit_predict(held)
```

**The estimate reads no future row.** The kernel's jittered end points
average observations already in the state at close, and a product enters
`Γ̂_h` only once both its legs are final.

**`psd=True`, the default, clips any negative eigenvalue and reports
`rcov_psd_repaired`.** For a correlation read on its own under `"preavg"`, give
`psd=False`. It selects the balanced form, a shorter window of
`k_n = floor(theta * sqrt(block_rows))` rows with the bias term subtracted,
which was the more accurate on
[docs/REGIMES.md §8](REGIMES.md#8-rcov-three-estimators-against-each-blocks-truth)'s
streams. The bias term is `b` times `sum(x x')`, where
`b = psi1 / (2 psi2 k_n^2)` tends to `6 / k_n^2` as the window grows.
That sum holds the block's covariance as well as the noise, so the term
removes `b` of the covariance too. The balanced form divides its estimate by
`1 - b` to restore it, as Christensen, Kinnebrock and Podolskij's footnote 1
does. It is not always positive semi-definite, so keep the default where
the matrix must be.

**Fix the bandwidth `H` with `bandwidth=`; otherwise give `block_rows`, the
block's expected length in rows, and `H` is set from the data** as
`H = ceil(c* xi^(4/5) n^(3/5))` with `c* = 3.5134`. `block_rows` is a sizing
hint for the ring: a longer block runs, clipped, and reports
`rcov_bandwidth_used`. `"preavg"` needs `block_rows` unless `preavg_rows=` is
given.

**A clock gap past `gap_cap`, or a session change, splits a block into
stretches**, and no product pairs two returns across the break.

### `audit` — what a stream's columns hold

*API:* [`po.spec.audit`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.audit) — *Rust:* [`audit.rs`](../crates/online-core/src/audit.rs) — *Outputs:* [fields](OUTPUTS.md#audit)

Reach for `audit` before choosing a model, to find out what is wrong with the
data. It learns nothing. Per column it counts the nulls, the NaNs, the
infinities and the values past `1e100` apart, and measures what makes a
column hard to learn from: a value repeated far more often than any other,
a run of one value, persistence like a random walk's, and heavy tails. Per
stream it measures the clock's duplicate stamps, gaps and spread of steps.
Each row writes only `weight_sum`, and `bank.audit()` reads the counts from
the state as a table after the run. `bank.check()` reads them too, and says
what they mean.

An audit reads every row, where a model skips a row with a feature that is
not a usable number. Its memory does not grow with the rows: about 10 KiB a
column at the defaults. Its counts are the same in one chunk or a thousand.
Each statistic, its definition and its cost are in the builder's docstring.

| measured per column | read as |
|---|---|
| `null`, `nan`, `pos_inf`, `neg_inf`, `beyond_bound` | what a model reads as missing, each kind counted apart |
| `mean`, `std`, `skew`, `kurtosis`, `min`, `max` | the moments of the usable values, in one pass |
| `distinct`, `top_value`, `top_count` | the distinct values, exact up to `distinct_cap` (default 256), and the most repeated one |
| `longest_run`, `equal_prev`, `equal_by_chance` | the longest run of one value, and the rows equal to the row before beside what independent rows would give |
| `min_change`, `min_change_count` | the smallest nonzero change between consecutive rows, and how many changes sit at it: whether repeats are a step of the column's grid or a stale feed's |
| `autocorr`, `unit_root_t` | the lag-1 autocorrelation and the Dickey-Fuller statistic, near 0 for a random walk |
| `median`, `mad`, `robust_z` | the median and median absolute deviation, from exact counts or a t-digest, and the largest robust z |

With `pairs=True` it keeps each pair of columns' correlation, which finds a
duplicated column. Groups merge into one audit with `pooled=True`: the
counts exactly, the moments to rounding, and the repeats past
`distinct_cap` and the median within the bounds the docstring gives.

This code uses `df` from [Example data](../README.md#example-data):

```python
audit = po.spec.audit("audit", columns=["x0", "x1", "x2", "y"], clock="t", gap_cap=300.0,
                      pairs=True, group="stock_id")
bank = po.ModelBank([audit])
bank.fit_predict(df)                              # the output record holds weight_sum alone
columns = bank.audit()                            # one row per (group, column)
pairs = bank.audit(table="pairs")                 # one row per (group, pair of columns)
clock = bank.audit(table="clock")                 # one row per group: duplicate stamps, gaps, the steps' spread
everything = bank.audit(pooled=True)              # the four groups as one stream
problems = bank.check()                           # sentinel, frozen, few_values, random_walk, heavy_tails, ...
```

## Clustering and classification

These models put a label on each row: a cluster found in the features alone,
or a class learned from a label column. Every label, distance and probability
is read before the row is learned from, so it is out-of-sample like every
prediction here.

### `kmeans` — exponentially weighted k-means

*API:* [`po.spec.kmeans`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.kmeans) — *Rust:* [`cluster/kmeans.rs`](../crates/online-core/src/cluster/kmeans.rs) — *Outputs:* [fields](OUTPUTS.md#kmeans)

Reach for `kmeans` when you know the number of clusters and they are round.
For clusters of any other shape, or of a number you do not know, use
[`micro`](#micro--density-based-clustering-any-shape). It labels each row
with the nearest of `k` centres, each the exponentially weighted mean of the
rows assigned to it (`ew_cov`'s mean recursion, per cluster). Three rules
keep the centres on the data: seeding, a split–merge move that finds a
cluster born after seeding, and a dead rule that re-places a centre whose
blob vanished.

With `c_j` centre `j` and `n_j` its decayed weight:

```
j*   = argmin_j ‖x − c_j‖²          distances in units of each feature's EW sd (standardize=False: raw units)
n'_j = λn_j + w                      c'_j = c_j + (w/n'_j)(x − c_j)     for j = j*
```

This code uses `df` from [Example data](../README.md#example-data):

```python
km = po.spec.kmeans(
    "km", features=["x0", "x1", "x2"], clock="t", half_life=2000.0, gap_cap=300.0,
    k=3,
    warm_rows=100,               # rows to seed from (default 500)
    seed_rule="lloyd",           # the default: the best of ten k-means++ starts
    split_merge=0.5,             # two centres nearer than this times the sum of their radii are one blob
    split_merge_every_rows=200,       # rows between split–merge checks
    dead_frac=0.05,              # under this share of an equal share, a centre is re-placed
    scale_floor=0.1,             # the metric's variance floor, as a fraction of each feature's long-run variance
    update_every_rows=1,              # the default, sequential: learned rows between applying each centre's batch; more is mini-batch
)
out = po.ModelBank([km]).fit_predict(df).unnest("km")
# cluster   the nearest centre's label, before the row is learned from
# dist      the distance to it;  dist_second  the distance to the second-nearest
# weight_sum, and coef: the centres, k rows of len(features)
index = po.spec.coef_index(km)   # how coef is labelled: target = "cluster0", "cluster1", ..., term = the feature
```

**Seeding waits for `warm_rows` rows, places the centres, and replays the
rows.** One k-means++ start lands in the wrong partition a third of the time
on five blobs in four dimensions. So the default, `seed_rule="lloyd"`, takes
ten starts, refines each by ten Lloyd iterations, and keeps the one of least
inertia. `seed=` fixes the random draws.

**While `split_merge` is above 0, a far row is scored but not learned from.**
A row is far when it sits outside its cluster, about four standard deviations
of `dist²` above the typical radius. Far rows wait for the split–merge move,
so they neither drag a centre nor widen its radius.

**The split–merge move finds a cluster born after seeding.** Every
`split_merge_every_rows` rows, if the two closest centres are nearer than
`split_merge` times the sum of their radii, the model treats them as two
centres on one blob. Once at least three rows far from every centre carry 5%
of the weight learned since the last check, the move frees the emptier of
the two and places it on those rows. It needs `k` of at least 3, so at
`k=2` only the dead rule acts. It cannot see one centre that owns two blobs,
whose rows all sit within its own radius, so keep the default
`seed_rule="lloyd"`, which prevents that. `split_merge=0` turns off the move
and the dead rule, which leaves plain sequential k-means.

**Increase `dead_frac` when regimes change faster than a dead centre is
re-placed.** A centre whose blob vanished fades, and is re-placed once it holds
less than `dead_frac` of an equal share: `log2(1/dead_frac)` half-lives after
the blob vanished, 4.3 at 0.05 and 2 at 0.25. In exchange, a cluster lighter
than `dead_frac/k` of the stream loses its centre whenever any row is far.

**`scale_floor` keeps a feature that goes quiet from taking over the
distance.** It floors each feature's variance at a fraction of its long-run
variance, a tenth by default. After twenty half-lives of quiet, the
feature's floored weight in the distance has grown about 57 times, where
unfloored it would have grown a million times.

### `micro` — density-based clustering, any shape

*API:* [`po.spec.micro`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.micro) — *Rust:* [`cluster/micro.rs`](../crates/online-core/src/cluster/micro.rs) — *Outputs:* [fields](OUTPUTS.md#micro)

Reach for `micro` when the clusters may have any shape. It finds them without
being told how many there are, flags the rows that belong to none, and follows
clusters that appear and vanish. It builds on DenStream's micro-clusters,
with a linking step over them.

Each micro-cluster is a *summary*: a decayed weight `n`, a centre `c`, and a
radius `r`, the exponentially weighted root-mean-square distance of its rows
from the centre. A summary is *established* once its weight reaches `beta_mu`
rows of the stream's mean row weight `w̄`. With `p` the number of features,
and distances in units of each feature's exponentially weighted standard
deviation, a row joins a summary only if the summary's radius stays within
`eps`:

```
a = n_j/(n_j + w̄)      b = w̄/(n_j + w̄)
n_j  ← λ n_j                                               every summary
j*   = the nearest established summary, if it keeps  a r²_j + a b ‖x − c_j‖² ≤ eps² p,
       else the nearest other summary, if it does,
       else a new one at x
n_j* ← n_j* + w     c_j* ← c_j* + (w/n_j*)(x − c_j*)     r²_j* ← min(·, eps² p)
```

The metric's variance is floored as
[`kmeans`](#kmeans--exponentially-weighted-k-means)'s is (`scale_floor`), so
the row on which a quiet feature moves again is not an outlier by that alone.

**Every `prune_every` units of the clock, the light summaries are dropped
and the established ones linked.** A temporal clock takes a duration, and
`max_rows_between_prunes` caps the learned rows between checkpoints, as a
regression's `solve_every` and `max_rows_between_solves` schedule a solve;
with neither, every 100 learned rows. Centres within `L` of each other share a label.
Give `macro_link=2` to link only summaries that touch. Left out, `L` is read
from the spacing the summaries show. A cluster's label is the smallest id
among its linked summaries, so it lasts as long as that summary does. A
summary whose rows stop lingers `half_life · log2(n / (beta_mu · w̄))`, with
`n` the weight it had. `max_clusters`, 200 by default, caps the live
summaries: past it the lightest is evicted, one not yet established first.
This code uses `df` from [Example data](../README.md#example-data):

```python
mc = po.spec.micro(
    "mc", features=["x0", "x1"], clock="t", half_life=2000.0, gap_cap=300.0, min_weight=50.0,
    eps=0.1,                     # the spread the model reads as *one* cluster, per standardized coordinate
    beta_mu=5.0,                 # a summary holding this many rows of the mean weight is established (default 3)
    prune_every=100,             # units of t between pruning and linking
    macro_link=None,             # the linking distance L, read from the spacing unless given
    max_clusters=200,            # the default cap on live summaries
)
out = po.ModelBank([mc]).fit_predict(df).unnest("mc")
out.select("cluster", "outlier", "n_clusters", "n_micro").tail(3)
# cluster              the label of the nearest established summary, on an outlier row too; null while there is none
# dist                 the distance to that summary's centre
# micro_id             the id of the summary this row goes to; ids only go up and are never reused
# outlier              no established summary takes the row
# n_clusters, n_micro  how many of each the state holds
# coef                 the established summaries, one [id, label, n, radius, c_1 .. c_p] row each
```

**Set `eps` to the spread of one cluster, per standardized coordinate:**
about 0.07 for two-dimensional shapes, 0.3 for well-separated Gaussians in 20
dimensions. Either mistake shows in the outputs:

| what you see | `eps` is | what to do |
|---|---|---|
| nearly every row is an `outlier`, and `cluster` stays null | too small: no summary reaches `beta_mu` before it is pruned | increase `eps` |
| `n_micro` is about the number of clusters | too coarse: each cluster is one summary, so the derived `L` reads the spacing *between* clusters and bridges them into one | lower `eps`, or set `macro_link=2` |

**Set `beta_mu` from the arrival rate, because it counts rows**, as
DenStream's `MinPts` does. At half-life `h` and `v` rows per clock unit, the
stream's steady-state weight is about `1.44·v·h` rows. So for a summary meant
to hold a share `s` of the stream, give `beta_mu ≈ 1.44·s·v·h`.

Measured at 20k rows and a half-life of 3,000, moons, rings and five Gaussians
in twenty dimensions all score an adjusted Rand index (ARI) of 1.000 against
the truth. Their `eps` were 0.07, 0.1 and 0.3, and `kmeans` cannot follow the
first two. At `eps=0.07`, uniform noise is flagged `outlier` 94% of the time,
and real rows 0.3%. A cluster born mid-stream had a label 31 rows in.

### `ew_class` — Gaussian classification on `ew_cov` moments

*API:* [`po.spec.ew_class`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.ew_class) — *Rust:* [`ewclass.rs`](../crates/online-core/src/ewclass.rs) — *Outputs:* [fields](OUTPUTS.md#ew_class)

Reach for `ew_class` to classify rows by a label column: name the column in
`label` and list its values in `classes`. It keeps one `ew_cov` state per
class, a weight `n_c`, a mean `μ_c` and a centred covariance `C_c`, and
scores a row by Bayes' rule over Gaussian classes. `C_c` is in the
population form, with no `ddof` correction, as `ew_cov`'s is. On three
Gaussian classes with their own covariances, its accuracy is within 0.001
of the Bayes rate, and its probabilities are calibrated to about 0.01.

`π_c` is a class's share of the weight, and `r_c` the ridge `precision_prior`,
which makes a class scoreable from its first row and fades as `ew_cov`'s does:

```
π_c = n_c / Σ n         r_c = precision_prior · s_c        (s_c: the prior's fade, as in ew_cov)
M_c = C_c + r_c I  (full)      M = Σ π_c M_c  (shared)      diag(C_c) + r_c  (diagonal)
ℓ_c = ln π_c − ½ ln det M_c − ½ (x − μ_c)ᵀ M_c⁻¹ (x − μ_c)
p_c = exp(ℓ_c − max ℓ) / Σ exp(ℓ − max ℓ)                  class = argmax ℓ
n_c ← λ n_c + w·[y = c]        μ_c, C_c ← weighted Welford on the row's own class
```

Give `covariance` to set what each class keeps:

| `covariance` | what each class keeps | the right choice when |
|---|---|---|
| `"full"`, the default (QDA) | its own covariance, and its Cholesky factor, which only the row's own class refactors (under a `window_size`, every class on every row) | in general |
| `"shared"` (LDA) | one covariance pooled by class weight, factorized once per row | the classes differ in location but not in spread; it then labels more than 98% of rows as `"full"` does in the tests, with fewer parameters to learn |
| `"diagonal"` (Gaussian naive Bayes) | variances only | speed matters most, since it runs about five times as fast as the other two ([Throughput](../README.md#throughput)); it cannot see a correlation, so two classes with the same marginals and opposite correlations are one class to it |

**Give `window_size` when the class means move,** because over a long history
two regimes average together. Each class's moments then cover only the
window ([A hard window](../README.md#a-hard-window)).
This code uses `df` from [Example data](../README.md#example-data):

```python
labelled = df.with_columns(      # the label: "up" where y is positive, else "down"
    pl.when(pl.col("y") > 0).then(pl.lit("up")).otherwise(pl.lit("down")).alias("dir")
)
cl = po.spec.ew_class(
    "cl", features=["x0", "x1", "x2"], clock="t", half_life=200.0, gap_cap=300.0, min_weight=20.0,
    label="dir",                 # the label column
    classes=["down", "up"],      # declared up front
    covariance="shared",         # one covariance pooled by class weight
    precision_prior=0.1,         # required: the ridge that makes a class scoreable from its first row
)
out = po.ModelBank([cl]).fit_predict(labelled).unnest("cl")
out.select("dir", "class", "p_up", "weight_sum").tail(3)
# class        the most probable class, as a string
# p_<class>    one per declared class; exactly 0 for a class no row has carried yet
# coef         the class means, in the order of classes (coef_up_x0 after df.online.unnest([cl]))
```

**List `classes` as text, because the label is read as text.** For an
integer or boolean label column, give `["0", "1"]` or `["true", "false"]`. A
label not in `classes` is an error naming the row. A null label scores the
row and learns nothing from it, so `weight_sum` counts every row the model
accepts, labelled or not, while `π_c` counts the labelled rows' weights. When
labels arrive late, give `embargo` the delay, and each label is learned once
that much clock has passed ([Labels that arrive
late](../README.md#labels-that-arrive-late)).

## Sequential tests and regimes

These models weigh the evidence that a sign holds or that the correlations
have changed, date the current regime, and say which regime the stream is in
now.

### `seqtest` — a sequential test of a sign, by betting

*API:* [`po.spec.seqtest`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.seqtest) — *Rust:* [`seqtest.rs`](../crates/online-core/src/seqtest.rs) — *Outputs:* [fields](OUTPUTS.md#seqtest)

Reach for `seqtest` to read evidence as it accumulates, and act the first time
it is enough. It tests whether a column tends to be positive, or whether one
spec predicts closer than another. A p-value cannot be read that way,
because checking it repeatedly inflates its error rate. An *e-process*,
which this model keeps, can be read at any row, as often as you like.

Each row counts as one trial, so `seqtest` refuses a `weight` or a
`half_life`. To skip a row, null its target. For a frame already in memory,
[`po.eval.seqtest`](https://hgilde.github.io/polars-online/eval.html#polars_online.eval.seqtest)
runs the same computation.

Per target it keeps the wealth of two gamblers, one betting that the next sign
is positive and one that it is negative. Each stakes the Krichevsky–Trofimov
fraction set by the counts so far, and stakes nothing unless the counts favour
its side:

```
s = sign(y)                    n⁺, n⁻: the signs counted before this row,  n = n⁺ + n⁻
λ⁺ = max(0, (n⁺ − n⁻) / (n + 1))          λ⁻ = max(0, (n⁻ − n⁺) / (n + 1))
ln E⁺ ← ln E⁺ + ln(1 + λ⁺ s)              ln E⁻ ← ln E⁻ + ln(1 − λ⁻ s)
```

**To test a column's sign, name it in `targets`.** A zero, null or NaN counts
as a tie: it bets nothing and counts nothing.
This code uses `df` from [Example data](../README.md#example-data):

```python
sign = po.spec.seqtest("sign", targets=["y"], group="stock_id")     # does y tend to be positive?
out = po.ModelBank([sign]).fit_predict(df).unnest("sign")
# log_e_pos_y, log_e_neg_y     the two gamblers' log wealth, as they stood before the row
# n_pos_y, n_neg_y             the signs counted so far
```

**To test which of two specs predicts closer, put both in the bank and name
them in `a` and `b`.** The sign tested is then `|resid_b| - |resid_a|`,
positive when `a` came closer, on the out-of-sample residuals the two specs'
output records report. A row where either side is null (warm-up, a skipped
row) sits out the test. When a side is a grid, pick its instance with
`a_suffix` or `b_suffix`, such as `"@h500"` or `"__r0.5@h500"` ([Output field
names](../README.md#output-field-names)). Without the suffix, the bank refuses the spec
and lists the grid's residual fields.
This code uses `df` from [Example data](../README.md#example-data):

```python
common = dict(targets=["y"], features=["x0", "x1"], clock="t", gap_cap=300.0, group="stock_id")
ridge = po.spec.ewridge("ridge", half_life=500.0, **common)
kalman = po.spec.kalman("kalman", half_life=500.0, coef_half_life=100.0, **common)
closer = po.spec.seqtest("closer", targets=["y"], a="kalman", b="ridge", group="stock_id")   # does kalman predict closer?
out = po.ModelBank([ridge, kalman, closer]).fit_predict(df)
# log_e_a_y, log_e_b_y, wins_a_y, wins_b_y   the two sides' log wealth and counts, as for a sign
verdict = out.group_by("stock_id").agg(pl.col("closer").struct.field("log_e_a_y").max())
# log_e_a_y >= ln(20): on that stock, kalman beat ridge at the 5% level, read at any row
```

**Reject on the first row where `log_e_pos ≥ ln 20`: that is the 5% level,
however often you look.** The null is that, given everything so far, the
next sign is no more likely positive than negative. Under it, `E⁺` is a
nonnegative supermartingale, and Ville's inequality gives
`P(E⁺ ever reaches 1/α) ≤ α`, however the rows depend on each other. No
distribution is assumed, and the size of the values is invisible: 60% small
gains and 40% huge losses count as "positive". For the two-sided question,
average the two sides' wealth: that is an e-value.
This code uses `df` from [Example data](../README.md#example-data):

```python
import math

signs = po.ModelBank([po.spec.seqtest("signs", targets=["y"], group="stock_id")]).fit_predict(df).unnest("signs")
rejected = (signs.filter(pl.col("log_e_pos_y") >= math.log(20))         # the rows at the 5% level, read at any row ...
            .group_by("stock_id", maintain_order=True).first())         # ... and each stock's first
e_value = (pl.col("log_e_pos_y").exp() + pl.col("log_e_neg_y").exp()) / 2   # two-sided: the mean of the wealth, not of its log
two_sided = signs.select("stock_id", e_value.alias("e_value"))
```

**Two events restart the test.** A session change restarts it under
`session_gap="reset"` or `group_close="session"`, and so does a step back
larger than `restart_after_step_back`. A session change with a numeric
`session_gap` does not.

Where the `max(0, …)` in a stake never binds, the wealth has the closed form
`2ⁿ B(n⁺+½, n⁻+½) / π`, the Beta(½, ½) mixture, and a test holds the bank to
it.

### `corrchange` — has the correlation structure changed?

*API:* [`po.spec.corrchange`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.corrchange) — *Rust:* [`corrchange.rs`](../crates/online-core/src/corrchange.rs) — *Outputs:* [fields](OUTPUTS.md#corrchange)

Reach for `corrchange` to test whether the correlations among the columns you
name have changed. Give `kind` to pick one of its three tests:

| `kind` | the question | reports | against | a change is found |
|---|---|---|---|---|
| `"monitor"` | was the correlation constant over a span? | on a span's last row | the span's own correlation | at the span's end, up to `span_rows` rows late |
| `"sequential"` | has it left the level a stable history set? | on each row tested against the history, from the second on | a history of `span_rows` rows | as soon as it crosses a boundary |
| `"window"` | how big is the change? | on every row once both windows are full, `2·span_rows` rows in | the window before | as soon as the two windows differ |

Every kind writes the same five fields, `stat`, `crit`, `flag`,
`since_flag` and `since_change`. A parameter that belongs to another kind
is refused, naming the kinds it applies to.

Under `"monitor"` and `"sequential"`, the statistic is the largest over the
pairs, and the level `alpha` (0.05 by default) is spread over the pairs as
`alpha / npairs` unless `alpha_adjust="none"`. For one test of the average
correlation's level, however many columns, give either kind `scalar=True`,
which runs it on the equicorrelation of the standardised row (`deco`'s `u`).

**`kind="monitor"` tests each span of `span_rows` rows for a constant
correlation,** with the closed-sample constancy test of Wied, Krämer and
Dehling (2012). With `ρ̂_j` the correlation of the span's first `j` rows, and
`D̂` the delta-method long-run standard deviation of `ρ̂`, the last row of a
span reports, per pair:

```
Q = max_{2≤j≤T} (j/√T)·|ρ̂_j − ρ̂_T| / D̂
```

Under the null, `Q` converges to `sup|B|` for a Brownian bridge `B`, so the
critical value is the Kolmogorov quantile, the published 1.3581 at 5%. `D̂`
uses the paper's Bartlett kernel, lag `l` at `1 − l/γ` with `γ = ⌊ln T⌋`.
This code uses `df` from [Example data](../README.md#example-data):

```python
monitor = po.spec.corrchange(
    "break", features=["x0", "x1"],
    kind="monitor",              # the constancy test, span by span
    span_rows=500,               # nothing is reported until a span closes: a delay of at most this many rows
    alpha=0.05,                  # the level, the default
    scalar=False,                # True: one test on the equicorrelation instead of one per pair
)
out = df.online.fit_predict([monitor]).unnest("break")
# stat          the test statistic, null except where one is due: here, on a span's last row
# crit          the critical value it is compared against
# flag          true on the row where stat crossed crit
# since_flag    learned rows since the last flag
# since_change  on a flag, the rows from the first changed one through the flag's
```

**`kind="sequential"` tests each new row against a stable history,** with
the monitoring procedure of Wied and Galeano (2013). A cycle is `span_rows`
rows of history, taken as stable, then up to `monitor_rows` rows, each tested
against the history as it arrives. A flag, or the cycle's last monitored row,
ends the cycle, and the row after it starts a new history. From the history
it reads each pair's correlation `ρ̂_h` and its long-run standard deviation
`D̂`, with the monitor's estimator. The `k`-th monitored row then reports

```
V_k  = (k/√m)·(ρ̂_k − ρ̂_h) / D̂           m = span_rows, ρ̂_k over the k monitored rows
stat = max over pairs of |V_k| / w(k/m)
w(b) = (1 + b)·(b/(1 + b))^γ               γ = boundary_gamma, 0 ≤ γ ≤ 0.49
```

and flags where `stat` passes `crit`.

The critical value is the paper's Eq. 7, `crit = (T/(1+T))^(1/2−γ)·q`, with
`T = monitor_rows/span_rows` and `q` a quantile of `sup_{0<s≤1} |W(s)|/s^γ`
for a Brownian motion `W`. At `γ = 0` and `T = 1`, `crit` is 1.5849 at 5%.
Where the paper simulates `q`, above `γ = 0`, this library solves it as a
diffusion with an absorbing boundary
([`boundary.rs`](../crates/online-core/src/boundary.rs)), within 0.03 of the
paper's Table 1. The paper allows `γ` up to 1/2, but the solve's work grows
as `1/(1/2 − γ)`: seconds at 0.49 and hours just under 1/2. So
`boundary_gamma` above 0.49 is refused.

**Increase `boundary_gamma`, the `γ` of `w(b)`, to catch an early change
sooner.** Above 0 the boundary starts lower, and so it flags more stable
streams:

| `boundary_gamma` | a change soon after the history | stable streams flagged, at a nominal 0.05 |
|---|---|---|
| 0, the default | caught latest | 0.04 to 0.09, the size nearest nominal |
| 0.25 | caught sooner | 0.04 to 0.09 |
| 0.45 | caught soonest | 0.12 to 0.18 |

```python
watch = po.spec.corrchange(
    "watch", features=["x0", "x1"],
    kind="sequential",
    span_rows=500,               # 500 rows of history, assumed stable (kind="monitor" over them checks it)
    monitor_rows=1000,           # then each of up to 1000 rows tested as it arrives: the paper's T = 2
    boundary_gamma=0.25,         # between the size of 0 and the early catch of 0.45
)
```

**`kind="window"` measures how big the change is,** as
`||vech(R_pre - R_post)||` over two adjacent windows of `span_rows` rows
each. `R_pre` and `R_post` are the windows' correlation matrices, and `vech`
stacks their upper triangles into one list. It compares that with a
permutation quantile: `n_perm` shuffles of the pooled rows between the
windows, redrawn every `permute_every_rows` rows. The shuffles move blocks of
`perm_block` rows, so rows that resemble their neighbours do not make the
null too liberal. To skip the permutations, give `crit=` as a fixed
threshold. The same `seed` gives the same critical values.

**Count a run of the window kind's flags as one change.** Two windows that
slide by one row are almost the same windows, so a statistic above the
quantile stays above it for a run of rows, and no rate of `alpha` per row
applies.

```python
size = po.spec.corrchange(
    "size", features=["x0", "x1"],
    kind="window",               # the size of the change between two adjacent windows
    span_rows=100,
    n_perm=200, permute_every_rows=500,   # 200 shuffles, redrawn every 500 rows
    perm_block=10,               # shuffled in blocks of 10 rows
    seed=0,                      # the default
    reset_on_flag=False,         # the default; True empties both windows at a flag
)
```

The monitor's size and power are close to Wied, Krämer and Dehling's tables,
on the `t_5` design of their own paper
([docs/REGIMES.md §2–3](REGIMES.md#2-is-the-monitor-the-size-its-paper-says)).
The sequential detector's size is within two standard errors of Wied and
Galeano's Table 2 in every cell ([docs/REGIMES.md
§9](REGIMES.md#9-the-sequential-detector-against-its-paper)).

### `bocpd` — how long has this regime lasted?

*API:* [`po.spec.bocpd`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.bocpd) — *Rust:* [`bocpd.rs`](../crates/online-core/src/bocpd.rs) — *Outputs:* [fields](OUTPUTS.md#bocpd)

Reach for `bocpd` (Adams & MacKay 2007) to know how long the current regime
has lasted. It keeps a probability distribution over the **run length**, the
rows since the last break, so it can say "we are forty rows into a regime",
not only "something broke".

`bocpd` refuses a `half_life` or `lam`, because the run-length posterior does
the forgetting. Set how fast with `hazard`, the expected run length. With
`H = 1/hazard` the chance of a break after each row, and `π_r` the posterior
predictive of run length `r` for this row, their Algorithm 1 is:

```
growth:      P(r_t = r+1, x_{1:t}) = P(r_{t−1} = r, x_{1:t−1})·π_r·(1 − H)
changepoint: P(r_t = 0,   x_{1:t}) = Σ_r P(r_{t−1} = r, x_{1:t−1})·π_r·H
```

**Give `hazard` as a duration to make breaks follow time, not rows.** A
number says breaks follow the row rate, so on irregular rows a weekend counts
as one row. On a temporal clock, `hazard="1h"` is `τ`, the expected time
between changepoints. The step of `d` into a row then carries the chance
`h = 1 − exp(−d/τ)` that a break fell inside it. That chance belongs before
the row, so `h` of every run's mass moves to the empty run before the row is
read. The recursion above then runs with `H = 0`, since no time has passed
since the row.

| `hazard` | what it is | the chance of a break |
|---|---|---|
| a number, 250 by default | the expected rows between changepoints, on any clock | `1/hazard`, after every row |
| a duration, such as `"1h"` | `τ`, the expected time between changepoints, on a temporal clock | `1 − exp(−d/τ)` for the step of `d` into each row, before the row is read |

The duration form follows the clock's rules:

| case | what happens |
|---|---|
| a gap past `gap_cap` | `d` is the step decay reads, so the gap counts as the cap |
| a step of 0, a stamp shared with the row before | no chance of a break: the row cannot begin a run, and reports `p_change` 0 |
| a row of weight 0 | its step's chance applies and it teaches nothing, so two steps with nothing learned between them read as one step of their sum |
| `prune_below=0` | each step of 0 leaves one run of no mass among the runs kept, until `max_run` folds it |
| no temporal clock, a plain-number clock parameter beside it, 0 or below, or `hazard_col` beside it | refused, naming the parameter |

This code uses `df` from [Example data](../README.md#example-data):

```python
runs = po.spec.bocpd(
    "regime", features=["ret"], group="stock_id",
    hazard=250.0,                # the expected run length: H = 1/hazard is the chance of a break after each row
    prior_nu=2.0,                # the prior on the variance, as 2a ...
    prior_scale=[2e-4],          # ... and 2b; left out, both come from the first rows
    emission="diag",             # the default: a normal-inverse-gamma per feature
    prune_below=1e-6,            # drop the runs holding less than this share of the mass
    max_run=None,                # 10,000 by default
    hazard_col=None,             # a column of per-row hazards instead
)
out = df.online.fit_predict([runs]).unnest("regime")
# p_change      P(r_t <= 1) given this row: the alarm
# run_mode      the most likely run length, before the row: the run began run_mode rows before this one
# run_mean      the posterior mean run length, before the row
# pred_<f>      the pre-row predictive mean of each feature, mixed over runs
# loglik        the row's log predictive density under that mixture
started = out.with_columns(      # the row each group's current run began on, counted in the group's own rows
    run_start=pl.int_range(pl.len()).over("stock_id") - pl.col("run_mode")
)
```

**Leave `prior_mean` and `prior_scale` out, and the first rows set them.**
The first `warm_rows` learned rows, `d + 2` by default, report null; their
mean and covariance (the diagonal under `"diag"` and `"robust"`) become the
prior, and they are then read as rows of a model that had it from the start.
A prior in the data's units decides what counts as surprising: the identity
it once defaulted to made every row of data in 1e-4 a break, and no row of
data at 1e4 able to begin a run. `prior_nu` defaults to the smallest value
that gives the prior's variance a mean, `d + 2` under `"gaussian"` and 3
otherwise, and at it the prior's mean variance is that covariance.
`prior_nu` and `prior_scale` are 2a and 2b in the gamma parametrisation. So
the example's values are Adams and MacKay's own finance example (a = 1, b =
1e-4, hazard = 250): a precision of 1e4 on average.

**Read the regime's age from `run_mode`, and use `p_change` only as an
alarm.** `p_change` is a per-row likelihood ratio, so it is spiky, and its
height depends on the size of the break against the prior scale. On three
kinds of break:

| the break | `p_change` | the run length |
|---|---|---|
| a ten-fold variance step | 0.83 on the row itself | finds it one to three rows later, and dates it to the right row |
| a four-sigma mean shift, with a diffuse prior | barely lifts | finds it one to three rows later, and dates it to the right row |
| a change in correlation alone | never moves at all | reaches it only under `emission="gaussian"`, a median of 121 rows later ([docs/REGIMES.md §5](REGIMES.md#5-two-changepoint-detectors-on-the-same-break)) |

**`p_change` reports `P(r ≤ 1)`, because `P(r = 0)` equals `H` on every
row:** the two branches share one predictive, so the normalised mass at
`r = 0` is *exactly* `H`, whatever the data. Under a duration `hazard` that
does not hold: the chance of a break comes before the row, nothing sits at
`r = 0` after it, and `p_change` is the chance that the row began a run. Row
one of a group reports nothing under the default `min_weight` of 1, since
`P(r ≤ 1)` is 1 there however the row looks.

Give `emission` to choose each run's model of the rows:

| `emission` | each run's model | note |
|---|---|---|
| `"diag"`, as in the example | a normal-inverse-gamma per feature | |
| `"gaussian"` | a normal-inverse-Wishart over all of them, O(runs d²) a row | the one that can see a break in the *correlation* alone |
| `"robust"` | each row's contribution weighted by `(pi(x)/pi(mode))**robust_beta` | one 20-sigma row moves nothing; without it, that row is a changepoint at `p_change` 0.91. Not free of the data's units |

**Keep `robust_beta` below about 0.2.** Above it, the weighting forgives the
rows of a new regime one at a time, so no change is ever detected again. The
default of 0.1 ignores a single 20-sigma row and still finds a four-sigma
shift within three rows, dated to the right one.

**Centre and scale the features for `emission="robust"`.** A change of
units multiplies every run's predictive density by one constant. Under
`"diag"` and `"gaussian"` the posterior's normalisation takes it out.
`"robust"` tempers each run's message by a power that differs from run to
run, so the constant does not cancel, and the runs' odds move with the
units. At a level of 1e4 with a spread of 1e-2, `p_change` moved by up to
0.86 against the same stream at level 0 and spread 1.

Two settings bound the runs kept, and a third sets the hazard row by row:

| setting | what it does |
|---|---|
| `prune_below` | drops the runs holding less than this share of the mass |
| `max_run` | 10,000 by default: folds every longer run into the last kept one. In a stream that seldom breaks, this is what bounds the runs kept |
| `hazard_col` | reads a per-row hazard from a column instead: its value at a row sets the chance of a break after that row. A null falls back to `hazard`, a value of 1 or less is an error naming the row, and a duration `hazard` beside it is refused |

### `hmm` — which regime are we in

*API:* [`po.spec.hmm`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.hmm) — *Rust:* [`hmm.rs`](../crates/online-core/src/hmm.rs) — *Outputs:* [fields](OUTPUTS.md#hmm)

Reach for `hmm` to ask "which regime are we in", not only "which regime does
this row look like".
[`ew_class`](#ew_class--gaussian-classification-on-ew_cov-moments) classifies
a row against *labelled* Gaussians. An `hmm` does the same arithmetic without
labels: the regime is a *hidden state*, and a transition matrix carries
information from one row to the next.

Here a hidden state means one of the `k` regimes, not the bank's saved state.
The model runs Hamilton's filter one row at a time, from the filtered `p` the
previous row left, and every output is read before the row is learned from:

```
p1_l   = Σ_k p_k·Π_kl                      the predicted state
f_l    = N(x | μ_l, Σ_l + r_l·I)           the state's density, r_l its fading prior ridge
loglik = ln Σ_l p1_l·f_l                   the row's surprise
p_l   ← p1_l·f_l / Σ                       the filtered state
```

This code uses `df` from [Example data](../README.md#example-data):

```python
hidden = po.spec.hmm(
    "regime", features=["x0", "x1"], half_life=500.0,
    k=2,                         # the number of hidden states
    precision_prior=1e-2,        # required: a state's centred co-moments start at zero, and a zero matrix has no density
    covariance="full",           # the default; "shared" or "diagonal", as for ew_class
    warm_rows=400,               # seed the states from this many learned rows; every output is null until then
    means=None, covs=None,       # or give the states outright, and learn=False to freeze them
    transition_prior=1.0,        # the default Dirichlet pseudo-count per cell of the transition matrix
)
out = df.online.fit_predict([hidden]).unnest("regime")
# filtered_0, filtered_1      the filtered state, before the row
# predicted_0, predicted_1    the predicted state
# state                       the most probable predicted state
# loglik                      the row's surprise
```

**Seeding decides which regimes the filter can find.** `warm_rows` seeds the
states from that many learned rows with `kmeans`' rule (`seed_rule="lloyd"`,
`seed=0`). Give a `warm_rows` large enough for those rows to span more than
one regime; otherwise the seeds split one regime in two.

**When the regimes differ only in covariance, give the states with `means`
and `covs`.** The default seeding is k-means, and zero-mean states differ in
nothing k-means can see, so it splits their rows by direction. On streams
that stay in one of two zero-mean states, the filter then puts about half the
rows after seeding in the true state, 0.51, which is chance. Given `covs`, it
puts 98% there ([docs/REGIMES.md
§1](REGIMES.md#1-does-hmm-recover-the-stream-that-made-it)). Without
the covariances, give the model a feature whose mean shifts with the regime,
made with Polars expressions in the query before the bank.

**A row splits across the states at weight `w·p_l`,** which sums to `w`, so
`weight_sum` follows the same recursion as in every model.

**A single extreme row can capture a state,** and move its mean far from the
data. A state with zero responsibility keeps its moments as they are, so a
state that stops winning never forgets, and the mixture is left one state
short. To guard against it, clip the features with Polars' `clip()` in the
query before the bank, give the states with `means` and `covs` and freeze
them with `learn=False`, or give a larger `precision_prior`. Keep that ridge
well below the data's variance: a ridge near it halves every correlation.

**With a sticky transition matrix, the filter beats a nearest-centre rule
that knows the true centres.** On two-dimensional blobs 1.5 apart, that rule
is 85% right, and the filter 99%. The matrix is learned from the **filtered
joint of consecutive states**, `ξ_kl = p_k(t−1)·Π_kl·f_l / Σ`, and a
Dirichlet pseudo-count keeps the matrix row of a state never visited a
distribution.

**The transition matrix moves one step per row, whatever the clock between
rows, so the gap over a weekend counts as a single step.** The transition
counts decay on the clock, and each row adds its weight, so the chance of
staying rises with the rows' density.

Three settings shape the matrix:

| setting | what it does |
|---|---|
| `transition_prior` | the Dirichlet pseudo-count per cell of the transition matrix, 1.0 by default |
| `transition=` | spreads that pseudo-count over a matrix of your own |
| `exog_tvtp=` | drives the transitions from a column, through fixed `tvtp_coef=`: time-varying transition probabilities (TVTP) |
