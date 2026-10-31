# Models

Each model's entry says when to reach for it, shows it working in a recipe
on data whose answer is known, states its update rule and shows its
parameters. Most also say what the model is checked against. The README's
[Models](../README.md#models) has the table of all twenty-two, how each
learns from a row, and which to choose for which question.

**The update rules share one notation.** A symbol in this table means the
same thing in every entry that uses it, and each entry defines its other
symbols beside its rule:

| symbol | what it is | used by |
|---|---|---|
| `z` | `[1, x]` when there is an intercept, and `x` when there is not | the linear models |
| `w`, `W` | the row's weight, and the weight accumulated before the row | every model |
| `λ` | the row's decay, `0.5 ** (Δclock / half_life)` (`lam=` on a spec gives it directly), with `Δclock` the clock's step from the row before, capped at `gap_cap` ([Time and decay](../README.md#time-and-decay)) | every model |
| `S_j`, `η_j` | target `j`'s running means of `z zᵀ` and of `z·y_j`, which the least-squares fits solve from | `ewridge`, `huber`, `quantile` |
| `e` | the row's residual: `y` less the prediction made before the row | `huber`, `quantile` |
| `ω` | a ridge added to a covariance: `precision_prior`, fading as `ew_cov`'s prior does, so that a class or a state can be scored from its first row | `ew_class`, `hmm` |

Without a clock column, each row is one unit of clock. A model keeps its
running sums in *accumulators* of three kinds:

| accumulator | where | why |
|---|---|---|
| an exponentially weighted **mean** | most models | it stays bounded over a stream of any length |
| a decayed **sum** | `rls`'s matrix, and the gradient sums of `ftrl` and of `sgd`'s adagrad | |
| a **centred** second moment, by a weighted Welford update | wherever a model keeps second moments as means | a variance is right even when a feature sits far from zero |

## Linear models

Ten models predict a numeric target. Nine regress it on features, and
`holt`, a baseline, uses none. Several of them take the same parameters:

| parameters | taken by | explained under |
|---|---|---|
| `ridge` | `ewridge`, where a list fits a grid; `huber` and `quantile`, as one number. `rls` refuses it and takes `delta` | [`ewridge`](#ewridge--ew-ridge-on-sufficient-statistics) and [`rls`](#rls--recursive-least-squares) |
| `solve_every`, `max_rows_between_solves` | `ewridge`, `lasso`, `huber`, `quantile` | [`ewridge`](#ewridge--ew-ridge-on-sufficient-statistics) |
| `target_gaps` | `ewridge`, `lasso` | [`ewridge`](#ewridge--ew-ridge-on-sufficient-statistics) |
| `window_size` | `ewridge`, `lasso` | [A hard window](../README.md#a-hard-window) |
| `standardize` | `ewridge`, `huber` and `quantile`, off by default; `kalman`, `sgd` and `pa`, on by default. `lasso` always standardizes, and `ftrl` never does | `ewridge`, `kalman` and `ftrl` |
| `coef_min`, `coef_max`, `coef_sum` | `sgd`, `pa` | [`sgd`](#sgd--stochastic-gradient-descent) |
| `weight` | every model, though `rls`, `kalman`, `sgd`, `pa` and `ftrl` read it differently from a weighted mean | [Weights](../README.md#weights) |

What `half_life` decays differs by model:

| model | `half_life` decays |
|---|---|
| `ewridge` | the running sums the fit is solved from |
| `lasso` | the same sums, and the error that picks the penalty unless `select_half_life` is given |
| `rls` | `A` and `b`, as decayed sums rather than means |
| `kalman` | the observation-noise estimate and the features' standardization; `coef_half_life` sets how fast a coefficient may drift |
| `huber`, `quantile` | their running sums, kept per target, and the residuals' spread `σ`, which `huber_scale_half_life` decays instead when given |
| `sgd` | `weight_sum`, the running moments `standardize` scales by, the spreads its cut and tube are drawn in (the Huber cut's at `huber_scale_half_life` when given), and adagrad's sums, but not the coefficients |
| `pa` | `weight_sum`, each target's own weight, the running moments `standardize` scales by and the spread its tube is drawn in, but not the coefficients |
| `ftrl` | its per-coordinate sums, but not its penalties, so a shorter half-life shrinks the fit more |
| `holt` | the level; the trend has `trend_half_life` |

### `ewridge` — EW ridge on sufficient statistics

*API:* [`po.spec.ewridge`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.ewridge) — *Rust:* [`ewridge.rs`](../crates/online-core/src/ewridge.rs) — *Outputs:* [fields](OUTPUTS.md#ewridge)

Start with `ewridge`, ridge regression solved from running sums, and change
to another linear model when your data need what it adds, such as
coefficients that drift (`kalman`). For a Huber loss use
[`huber`](#huber--quantile--robust-regression), which is these sums with
each row down-weighted by its residual.

**A grid of half-lives and ridge values is fitted in one pass.** Each
half-life keeps a set of running sums, and each ridge value is one more
solve from that set. Here a slope drifts from 1 to 3 over 20,000 rows, and
four fits read it at the last row:

```python
import numpy as np
import polars as pl
import polars_online as po

# A slope on x that drifts from 1 to 3 over 20,000 rows, one row a clock unit.
rng = np.random.default_rng(1)
n = 20_000
x = rng.standard_normal(n)
slope = 1.0 + 2.0 * np.arange(n) / n
y = slope * x + 0.5 * rng.standard_normal(n)
pl.DataFrame({"t": np.arange(n, dtype=float), "x": x, "y": y}).write_parquet("drift.parquet")

# One spec holds the whole grid, fitted in one pass.
drift = po.spec.ewridge(
    "drift", targets=["y"], features=["x"], clock="t", gap_cap=10.0,
    half_life=[200.0, 20_000.0],   # a grid of two half-lives, one set of running sums each ...
    ridge=[1e-6, 0.1],             # ... by two ridges, solved from each set: four fits
)
bank = po.ModelBank([drift])
bank.fit(pl.scan_parquet("drift.parquet"))   # learn every row, keep no output
# The truth at the last row, beside each fit's slope.
print(f"the true slope at the last row: {slope[-1]:.2f}")
for r in bank.coef().filter(pl.col("term") == "x").iter_rows(named=True):
    print(f"half-life {r['instance']}, ridge {r['ridge']:g}: slope {r['coef']:.2f}")
```

```text
the true slope at the last row: 3.00
half-life @h200, ridge 1e-06: slope 2.98
half-life @h200, ridge 0.1: slope 2.71
half-life @h20000, ridge 1e-06: slope 2.11
half-life @h20000, ridge 0.1: slope 1.92
```

The half-life of 200 rows follows the drift. The half-life of 20,000 still
weighs the rows from the middle of the stream, where the slope was near 2.
A ridge of 0.1 divides a slope by about `1 + 0.1/var(x)`, here 1.1, because
the ridge is added to the feature's variance in the solve.

`S_j` and `η_j` are target `j`'s running means of `z zᵀ` and `z·y_j`, kept
over the rows where `y_j` is present, and `W_j` is the weight behind them.
The coefficients are solved from them on a schedule:

```
W'   = λW + w                       weight_sum, over every row
W_j' = λW_j + w                     on the rows where y_j is present, λW_j on the others
S_j' = (λW_j·S_j + w·z zᵀ) / W_j'   η_j' = (λW_j·η_j + w·z·y_j) / W_j'
solve, ridge_scale="mean":  (S_j + ridge·D) β_j = η_j + ridge·D·β₀      D = I with a 0 at the intercept
solve, ridge_scale="sum":   (W_j·S_j + p·ridge·I) β_j = W_j·η_j + p·ridge·β₀
                            p the product of every decay so far, β₀ = coef_prior (0 by default)
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
| what the sums remember | `window_size` ([A hard window](../README.md#a-hard-window)), `session_shrink` | `session_shrink` needs `session` and `long_half_life`, and is refused beside `session_gap="reset"`, `group_close="session"` and `window_size` |
| a wide fit | `gram_block_rows`, `gram_threads` | `gram_block_rows` is refused beside `window_size`, and wherever no schedule spaces the solves: `solve_every=0`, `max_rows_between_solves=1`, or `half_life=inf` and `lam=` without a `solve_every` |

**A grid of `ridge` values or of `feature_sets`, as in `rr`, adds a solve
per value and no update of the sums.** Only the named sets are fitted, so
to keep a fit on every feature, name a set that holds them all, such as
`"all"`.

**To keep the coefficients current at every row, give
`max_rows_between_solves=1`.** Otherwise `ewridge` predicts with the
coefficients of its last solve. Under a finite half-life it solves once the
weight learned since then reaches `ln 2 / 50` of the weight the fit holds:
about 1.4 %, or every `half_life/50` of clock in steady state. To solve on
the clock instead, give `solve_every` in clock units. Until `solve_every`
has passed since the stream's start, or `max_rows_between_solves` rows, the
model solves by weight as it does by default, so a fit of its first few rows
is not held for a whole period: with `solve_every=500` and `min_weight=50`,
the error of the predictions it served in the first 600 rows fell from
5.6e-4 to 3.1e-6 on five seeds of a five-feature stream at a half-life of
1,000. `solve_every=float("inf")`, or any value from `1e18` on, past the end
of any stream, never comes due and has no first period: the model collects
its sums and solves only to give its first rows a fit and where a row
cap says. `half_life=inf` and
`lam=` solve on every row, so when the fit need not be current at every
row, give them a `solve_every`. On 6M rows × 20 features from a parquet
stream, `solve_every=1000` took 1.4 s instead of 10.7 s, with the
coefficients at most 1000 rows out of date.

**`ewridge` solves by Cholesky factorization, and retries a near-singular
system with a small jitter on the diagonal.** `bank.solve_failures()`
counts the solves that needed it. A count above zero points at constant or
collinear features, or too few rows for the features.

**A system can factorize and still be too nearly singular for its solve to
keep a digit, and then `ewridge` withholds the fit.** Rounding can leave a
matrix past `1/u` in condition positive definite, `u = 2^-53`, and its
coefficients then come out wrong by more than their own size. Two features
equal to within `1e-11` of their spread gave slopes of -364.1 and 363.1 for a
truth of 1 and -2. So each factorization estimates the condition of the
system with each column scaled by the root of its diagonal. That is what a
Cholesky solve is accurate to, so the features' units do not enter. A
factorization that needed jitter is judged as the system it solves, `A + δI`:
the first rung's `δ`, `1e-12` of the mean diagonal, leaves `‖H⁻¹‖₁` near
`1e12` on a design with a column of no variance of its own, past the limit
from about 30 coefficients. An upper bound from the factor settles most systems at
the cost of one solve with it, and the rest take Hager and Higham's
estimate, about four. Where it puts the solve's
forward-error bound at 1 or more, no digit of the coefficients is promised:

| | |
|---|---|
| the estimate | `‖H⁻¹‖₁`, `H = D⁻¹ (A + δI) D⁻¹`, `D = diag(√(a_ii + δ))`, `δ` the jitter (0 where none was needed), by Hager's method as LAPACK's `dlacn2` runs it |
| the solve's backward error in `H` | at most `η = k γ_{3k+1}` an entry, `γ_m = m u / (1 − m u)` |
| the forward-error bound | `2κη / (1 − κη)`, with `κη = k γ_{3k+1} ‖H⁻¹‖₁` |
| withheld where | `3 k γ_{3k+1} ‖H⁻¹‖₁ ≥ 1`: at `k = 3` from `1.0e14`, at `k = 10` from `9.7e12`, at `k = 50` from `4.0e11` |

A withheld fit has null `coef`, `support_coef` and predictions,
`withheld_reason` reads `ill_conditioned`, and the solve is counted in
`bank.solve_failures()`. The next solve inside the limit reports again.
NIST's statistical reference datasets show where the limit falls. Their
hardest designs that double precision can solve by the normal equations
have correlation matrices conditioned at `2.7e6`, far inside it. `Filip`, a
degree-10 polynomial at `1.5e16`, is past it, and is withheld. A ridge
keeps a system inside the limit. So does `rls`, which rotates the design
rather than squaring it and loses half as many digits: it keeps 7 of
`Filip`'s.

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
| `"pairwise"` | every row, the way pandas' `DataFrame.cov` takes a pairwise-complete covariance, with `η_j` centred at the target's own rows' means | one matrix |

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
not move. The blend changes the moments and nothing else: the coefficients
stay as last solved until the next solve the schedule brings due, which
reads the blended moments. So a small `f` moves the output by a small amount,
and `0` is the same as no `session_shrink`. With `max_rows_between_solves=1`
the solve on the session's first row reads the blend, and every row after
it is predicted from it.

**For a fit over hundreds of features, give `gram_block_rows=256`.**
Updating `S` takes O(k²) time per row for `k` features, and at a thousand
features that update takes most of the run. The option holds 256 rows back
and adds them to `S` with one matrix product
([docs/PERFORMANCE.md](PERFORMANCE.md) §18). A solve brings `S` up to
date before it runs, so solve rarely: under `half_life=inf`, give
`solve_every=float("inf")`, which never comes due, and a row cap, as `wide`
does:

```python
wide = po.spec.ewridge(
    "wide", targets=["y"],
    features=["x0", "x1", "x2"],   # three here, so the block runs; the speed-ups are for 256 to 2,000
    half_life=float("inf"), solve_every=float("inf"), max_rows_between_solves=1000,
    gram_block_rows=256,           # add 256 held rows to S in one matrix product
)
```

| features | rows per second with no solve, against one row at a time, on one thread |
|---:|---:|
| 256 | 5.1× |
| 1,000 | 6.6× |
| 2,000 | 5.9× |

With a solve every 512 rows, each speed-up falls to about 4×, because the
solve takes the same time either way. Blocking leaves `weight_sum`, the
timing of every prediction and chunk invariance unchanged, and the
coefficients agree with the row-by-row fit to rounding.

**That product runs on several cores wherever the bank has idle threads.**
`gram_threads` defaults to `"auto"`, which splits a group's update only
when the groups the bank runs at once leave at least half its pool idle.
The threads come from the pool the bank runs its groups on, and the output
is the same to the bit at every thread count. At 2,000 features a bank
learned 3.9 times as fast on eight threads
([docs/PERFORMANCE.md](PERFORMANCE.md) §37). A bank whose groups
already keep every core busy gains nothing from a split, so `"auto"` does
not split there; `gram_threads=8` asks for eight whatever the groups.

**With decay off and `ridge=0`, `ewridge` fits ordinary least squares over
every row seen, in any row order.** Its coefficients match
`numpy.linalg.lstsq` to 2e-13, fed forwards or backwards. On 6M rows × 20
features from a parquet stream, memory peaked at 1.4 GB, against 3.97 GB
for `lstsq` on the same rows.

### `rls` — recursive least squares

*API:* [`po.spec.rls`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.rls) — *Rust:* [`rls.rs`](../crates/online-core/src/rls.rs) — *Outputs:* [fields](OUTPUTS.md#rls)

Use `rls` when the coefficients must be current at every row: it updates
them on every row in O(k²) time, without a solve. For a Huber loss use
[`huber`](#huber--quantile--robust-regression) with `ridge_scale="sum"`,
which is this fit with each row down-weighted by its residual, solved on
a schedule.

**`rls` is `ewridge(ridge_scale="sum")` solved on every row, to better
than 1e-9.** Its `delta` is that `ridge`. Here the two run side by side on
a stream whose slope drifts:

```python
import numpy as np
import polars as pl
import polars_online as po

# y = 0.5 + slope·x0 − x1 plus noise, with x0's slope drifting from 1 to 2.
rng = np.random.default_rng(2)
n = 5_000
x0, x1 = rng.standard_normal((2, n))
y = 0.5 + (1.0 + np.arange(n) / n) * x0 - x1 + 0.3 * rng.standard_normal(n)
pl.DataFrame({"t": np.arange(n, dtype=float), "x0": x0, "x1": x1, "y": y}).write_parquet("rls.parquet")

# rls beside the ewridge it equals, on the same rows.
common = dict(targets=["y"], features=["x0", "x1"], clock="t", gap_cap=10.0, half_life=300.0)
specs = [
    po.spec.rls("rls", delta=2.0, **common),                      # A starts at 2·I, a prior that fades
    po.spec.ewridge("sum", ridge=2.0, ridge_scale="sum",          # the same prior on ewridge's sum scale ...
                    max_rows_between_solves=1, **common),         # ... solved on every row
]
out = pl.scan_parquet("rls.parquet").online.fit_predict(specs).collect()
# The largest gap between their predictions, then rls's fit.
gap = (out["rls"].struct.field("pred_y") - out["sum"].struct.field("pred_y")).abs().max()
print(f"the two predictions differ by less than 1e-9 on every row: {gap < 1e-9}")
print("rls's coefficients at the last row:", [round(c, 2) for c in out["rls"].struct.field("coef").drop_nulls()[-1]])
```

```text
the two predictions differ by less than 1e-9 on every row: True
rls's coefficients at the last row: [0.52, 1.92, -0.99]
```

`A` and `b_j` are the decayed sums of `zzᵀ` and `y_j z`, started from the
prior strength `delta`, the classic RLS name for `P₀ = I/δ`:

```
A ← λA + w zzᵀ       b_j ← λb_j + w y_j z        β_j = A⁻¹ b_j        decayed sums, not means
A₀ = delta·I         b₀ = delta·coef_prior
```

Because `A` is a sum, `delta` is in the features' squared units summed over
rows, a row's weight is on the sum scale, and a heavier stream outweighs
the starting prior sooner ([Weights](../README.md#weights)). It is
`ewridge`'s `ridge` under `ridge_scale="sum"`, not its default `ridge`,
which never fades, and it penalizes every coefficient, the intercept
included. `ridge` is refused here, with a message naming `delta`.

**The targets share one Cholesky factor of `A`, so a row with any target
null is scored but teaches none of them.** When the targets go missing on
different rows, give each its own `rls` spec. Otherwise one spec holds them
all. For the same reason, `min_weight` counts only the rows with every
target present, where `weight_sum` counts every row.

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
`ewridge`'s centred running sums, and chooses among them as the stream
runs.

**The penalty that has predicted best so far is reported on every row.**
Here 3 of 20 features carry the target, and the penalty in force at the
last row keeps exactly those three:

```python
import numpy as np
import polars as pl
import polars_online as po

# 20 features, of which only x0, x1 and x2 carry weight in the target.
rng = np.random.default_rng(3)
n, k = 5_000, 20
x = rng.standard_normal((n, k))
y = 2.0 * x[:, 0] - 1.5 * x[:, 1] + 1.0 * x[:, 2] + rng.standard_normal(n)
names = [f"x{i}" for i in range(k)]
pl.DataFrame(x, schema=names).with_columns(y=y).write_parquet("wide.parquet")

# A path of penalties, the best so far chosen as the stream runs.
las = po.spec.lasso(
    "las", targets=["y"], features=names,
    half_life=float("inf"),                  # every row counts the same
    penalties=[0.3, 0.1, 0.03, 0.01],        # the penalties, decreasing
)
bank = po.ModelBank([las])
out = pl.concat(bank.fit_predict_batches(pl.scan_parquet("wide.parquet")))   # the output, chunk by chunk
# The penalty chosen at the last row, and the features it keeps.
chosen = out["las"].struct.field("penalty_selected_y")[-1]
betas = bank.coef().filter(pl.col("penalty") == chosen, pl.col("term") != "intercept")
kept = betas.filter(pl.col("coef") != 0.0)
print(f"penalty in force at the last row: {chosen}")
print(f"coefficients left nonzero: {kept.height} of {betas.height}:",
      {r["term"]: round(r["coef"], 2) for r in kept.iter_rows(named=True)})
```

```text
penalty in force at the last row: 0.03
coefficients left nonzero: 3 of 20: {'x0': 1.97, 'x1': -1.44, 'x2': 0.99}
```

It runs coordinate descent on the standardized sums, warm-started along the
path and from one solve to the next. `C` is the features' correlation
matrix, and `c_i` is feature `i`'s covariance with the target over the
feature's standard deviation. For each penalty `l` in `penalties`, the
sweep repeats until a sweep moves no coefficient by `tol·s_y` or more, with
`s_y` the target's spread:

```
ρ_i = c_i − Σ_{j≠i} C_ij β_j
β_i = soft(ρ_i, l·l1_ratio) / (C_ii + l·(1 − l1_ratio))        soft(v, t) = sign(v)·max(|v| − t, 0)
```

After the sweeps the slopes are unscaled, and the intercept is `ȳ − m·β`,
with `m` the features' means. The target is centred but not scaled, so the
L1 threshold `l·l1_ratio` is in the target's units. So scale `penalties`
with the target: for a pure lasso, a target ten times larger needs
penalties ten times larger to zero the same coefficients.

**The stopping rule has no units.** The coefficients `β_i` the sweep moves
are in the standardized coordinates, so they are in the target's units, and
`tol` is compared with a move as a share of the target's spread `s_y`: its
standard deviation over the rows the fit reads, or its root mean square
without an intercept. So a target scaled by a power of two, with
`penalties` scaled by the same factor, gives the same fit scaled, to the
bit. A descent that runs out of its `max_iter` sweeps is counted in
`bank.solve_failures()`, so if that count is above zero, give a larger
`max_iter`. Columns that nearly repeat each other slow the descent most: it
shrinks the error along a pair of correlation `ρ` by `ρ²` a sweep, about
`1 − 4/κ` with `κ = (1 + ρ)/(1 − ρ)` their correlation matrix's condition
number, so each e-fold of the error takes about `κ/4` sweeps.

```python
las = po.spec.lasso(
    "las", targets=["y"], features=["x0", "x1", "x2"], clock="t", gap_cap=300.0, half_life=600.0,
    penalties=[0.1, 0.01, 0.001],    # the penalties, decreasing
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
the plain mean of that error, give `select_half_life=inf`. A target's errors
count once the fit holds `k + 1` of its rows of positive weight (`k` without
an intercept), a row per coefficient, and its first solve falls on that row
where the schedule has not solved it sooner. Neither reads `min_weight`,
which withholds the output alone: the selection scores the predictions the
gate withholds, so the fit and the penalty chosen are the same at any
`min_weight`. When the gate held the selection back too, the penalty chosen
moved with it: six penalties on a stream with t(4) noise chose otherwise
after 3,000 rows on 6 of 50 seeds at the default `min_weight` and 13 at 50.

**Under a `window_size`, a feature that stays constant inside the window
gets a coefficient of exactly zero,** and the error that picks the penalty
covers only the window ([A hard window](../README.md#a-hard-window)).

### `kalman` — random-walk-β dynamic linear model

*API:* [`po.spec.kalman`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.kalman) — *Rust:* [`kalman.rs`](../crates/online-core/src/kalman.rs) — *Outputs:* [fields](OUTPUTS.md#kalman)

Use `kalman` when a relationship moves faster than a half-life can follow,
or when a coefficient should be forgotten once the rows stop supporting it.

**A coefficient that may drift follows a jump that running sums only
average in.** Here a slope jumps from 1 to 3 at row 2,000. `kalman` lets
half a coefficient's memory go in 20 clock units, and `ewridge`'s sums
forget at the spec's half-life of 200:

```python
import numpy as np
import polars as pl
import polars_online as po

# A slope on x that jumps from 1 to 3 at row 2,000, one row a clock unit.
rng = np.random.default_rng(4)
n = 3_000
x = rng.standard_normal(n)
slope = np.where(np.arange(n) < 2_000, 1.0, 3.0)
y = slope * x + 0.5 * rng.standard_normal(n)
pl.DataFrame({"t": np.arange(n, dtype=float), "x": x, "y": y}).write_parquet("jump.parquet")

# kalman, whose slope may drift, beside ewridge's running sums.
common = dict(targets=["y"], features=["x"], clock="t", gap_cap=10.0, half_life=200.0, coef_every=0)
specs = [
    po.spec.kalman("kalman", coef_half_life=20.0, **common),            # a coefficient may drift
    po.spec.ewridge("ewridge", max_rows_between_solves=1, **common),    # solved on every row
]
out = pl.scan_parquet("jump.parquet").online.fit_predict(specs).collect()
# Each model's slope just before the jump and after it.
for row in (1_999, 2_010, 2_050, 2_200):
    k = out["kalman"].struct.field("coef")[row][1]     # [intercept, slope] after the row
    e = out["ewridge"].struct.field("coef")[row][1]
    print(f"row {row}: truth {slope[row]:.0f}, kalman {k:.2f}, ewridge {e:.2f}")
```

```text
row 1999: truth 1, kalman 1.00, ewridge 1.01
row 2010: truth 3, kalman 1.56, ewridge 1.09
row 2050: truth 3, kalman 2.76, ewridge 1.45
row 2200: truth 3, kalman 3.04, ewridge 2.05
```

Per target `j`, `β_j` are the coefficients and `R_j·P_j` their covariance:
`P_j` is held in units of the observation noise `R_j`, the target's
exponentially weighted residual variance. `κ_i` is coefficient `i`'s
`revert_half_life`, and `φ_i = 2^(−Δclock/κ_i)` its entry of `Φ`. `Q` is the
process noise, which `coef_half_life` sets, in the same units:

```
β_j ← Φβ_j    P_j ← ΦP_jΦ              Φ = diag(2^(−Δclock/κ_i)), every row
P_j ← P_j + Q·D_j²,  D_j ← 0           on a row that observes y_j; D_j the clock since the last
                                       (bounded on a reverting coefficient, below)
s   = zᵀP_j z + 1/w                     K   = P_j z / s
β_j ← β_j + K(y_j − zᵀβ_j)              P_j ← P_j − K zᵀP_j
```

A row's weight scales its observation's precision: the observation's
variance is `R_j / w`.

**The noise is a scale, not a weight: the gain never reads the noise
estimate.** The model's noise is one constant per target, and its prior
(`p0` times the noise) and its drift (`(ln 2·D/h)²` times the noise) are both
multiples of it, so it cancels from the gain. The estimate only scales what
is read out: `pred_var`, `se_coef` and the predictions' error, `R_j·zᵀP_j z`.
It learns from the filter's own prediction on every row the filter predicts,
whether or not `min_weight` lets that prediction out, so the readouts, and a
`q=` given outright, which is read over the estimate, are the same at any
`min_weight`: the gate withholds the output alone.
An estimate inside the gain weighs each row by the inverse of the noise as
it stood before the row. A burst of large residuals then inflates it, and
the rows after the burst count for less than the burst did. With no drift,
a fifth of the first 500 rows 50 too high held the slopes at [−25.5, 33.6]
where least squares on the same rows gives [4.1, −1.4], and after a change
of regime the fit stayed 45.7 of a gap of 46 away. Weighing rows by a running
noise estimate is also volatility weighting: on the intraday returns of
[docs/VALIDATION.md](VALIDATION.md), section 5, it had scored R² −0.059
against −0.070 now on the next minute's return, and −0.075 against −0.101 on
five minutes'. A fixed `obs_var=` leaves the gain as it is and scales the
readouts by its own number; only the intercept's prior differs (below). A
state saved by a release before this form goes on as it was fitted, with
the estimate in the gain.

**The intercept's prior is diffuse, so a level costs nothing.** The first
row that observes the target fills the intercept with its whole innovation,
`β_0 = y − zᵀβ`, at a variance of `zᵀP z + 1/w`: the limit of a prior that
knows nothing of the level (Durbin and Koopman, *Time Series Analysis by
State Space Methods*, §5.2). Every prediction on a target shifted by a level
`L` then moves by `L`, to the rounding of a number of its size, from the
first. When the noise estimate set the prior and the gain, the first
innovations, the level among them, inflated it: at `L = 1e4` the predictions
stood 0.03 of the target's spread off the unshifted fit for thousands of
rows. Each slope's prior is `p0` in units of the noise, so the default
`p0=1` gives a prior as uncertain as one observation: unstandardized from
the start, standardized on the first row that observes the target once the
feature's variance is above 0. A target with nothing sized yet is not
predicted, and the row is withheld (`below_warm_rows`); with an intercept
no row waits for it, the first sizing the intercept. Since
neither the gain nor the prior reads the noise, the warm-up does not depend
on the target's units. With a fixed `obs_var=`, `P` is absolute and starts
at `p0·obs_var` on every coefficient, the intercept included.

```python
revert = po.spec.kalman(
    "k", targets=["y"], features=["signal_a", "signal_b"], clock="t", gap_cap=10.0,
    half_life=200.0,                   # how fast the noise estimate and the standardization forget
    coef_half_life=100.0,              # how fast a coefficient may drift; or q, never both
    revert_half_life=[float("inf"), 50.0, 50.0],   # inf leaves the intercept alone; the slopes revert at 50
    standardize=True,                  # the default
    p0=1.0,                            # the default: each slope's prior variance, in units of the noise
    share_p=False,                     # the default: one P per target; True keeps one for all
)
```

**By default each coefficient drifts as a random walk: between observations
its estimate holds and its variance grows.** With `h_i` its
`coef_half_life`, a row that observes the target `D` clock units after its
last observation adds `(ln2 · D / h_i)²` to that variance, in units of the
noise, on standardized features. That matches EW-RLS's steady state at any
spacing. The noise is
charged once for the whole of `D`, since `D²` does not split into the
squares of its parts. Charged per row, a row of weight 0 inside a gap moved
every later prediction, and a target seen on one row in ten forgot 3.6
times slower on the clock. For a rate per coefficient, give
`coef_half_life` a list with the intercept's entry leading, and an entry
of `inf` pins its coefficient. To set the process noise outright, give
`q=`, whose `q_i` is added as `q_i · D²` in the coefficient's own units:
measured against the noise estimate, the one place the estimate reaches
the fit.

**`coef_half_life` is each standardized direction's memory, with the other
directions held still.** Under correlated features, a direction of the
design's correlation with eigenvalue `λ` is learned with a memory of about
`h/√λ` (Ljung and Gunnarsson, 1990). At a correlation of 0.9 and `h = 50`
that measured 37.5 and 160.5 clock units, against 36.3 and 158.1. EW-RLS
forgets every direction at `h`. The gain reads no noise estimate, so a
step in the truth is learned at that memory, however much the step inflates
the residual variance. Under `standardize` the drift is per standard
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
(1 − 2^(−D/κ_i))/θ_i` and `θ_i = ln 2/κ_i`. That is `q_i·D²` for a short
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
anchor's coordinates exactly. The state is mapped to new coordinates,
`b ← A b` and `P ← A P Aᵀ`, when a scale moves past twice or half the
anchor's, or a mean by more than the anchor's scale. So no prediction moves
because the scaler did. A filter read through the scaler as it stands would
move every prediction with the scaler's own wander under a half-life. At a
half-life of 50 that wander added 223 noise variances to the out-of-sample
error at R² 0.99998 (one feature, 30,000 rows, the median of five seeds). This
filter's error stands 0.014 above the noise there, as an unstandardized
one's does. A map of more than 1024 times a scale is not
followed. For a coefficient per current standard deviation, z-score the
column in the stream and give it with `standardize=False`, as in
[Features in units of their spread](WINDOWS.md#features-in-units-of-their-spread).

**`predict` applies the reversion,** by the same `Φ` over the distance
from the last learned row, capped by `gap_cap`. A slope therefore keeps at
least `2^(−gap_cap/κ_i)` of its value. To let a far prediction approach the
intercept, give a `gap_cap` that spans several reversion half-lives. Under
`standardize` the intercept is the target's level at the features' means.

**With `standardize=False`, `q=0` and a fixed `obs_var=`, the filter is
exactly Bayesian linear regression,** the ridge with penalty `1/p0`:
river's `BayesianLinearRegression` agrees within 1e-12. Standardized, each
prior waits for its feature's scale, so it is not that ridge.

**`se_coef` and `error_inflation` are exact under the model,** which says
the coefficients walk at the noise `coef_half_life` implies. On a truth that
does not move, the standard errors come out about √2 too large. With `R`
the residual variance, which already carries the estimation error,
`error_inflation` counts that error twice: the mean squared error is 0.96 of
`R·error_inflation²` at `coef_half_life=50`. Under `share_p`, each target's
`se_coef` reads the shared `P` times its own noise.

**`share_p=True` saves work and changes no prediction where every target
is on every row.** `P`'s recursion reads neither the targets nor the noise,
so targets observed on the same rows would each carry the same `P`, and one
`k×k` update a row serves them all: on section 4 of
[docs/VALIDATION.md](VALIDATION.md) the two specs score the same, to the
bit. Where a target is null on rows the others are present on, the shared
`P` takes those rows too, and that target's memory follows the other
targets' rows. With an explicit `q`, the shared `P` measures it against the
targets' mean noise.

### `huber` / `quantile` — robust regression

*API:* [`po.spec.huber`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.huber) and [`po.spec.quantile`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.quantile) — *Rust:* [`robust.rs`](../crates/online-core/src/robust.rs) — *Outputs:* [huber](OUTPUTS.md#huber), [quantile](OUTPUTS.md#quantile)

Use these when one wild target must not move the fit: `huber` down-weights
the rows that miss by far, and `quantile` fits a conditional quantile, such
as the median, which outliers cannot pull. Both read each row's residual
against the fit *before* the row is learned, so both stay out-of-sample.
Both take `ewridge`'s `ridge=` as a single number, and refuse a list.

**A few wild targets pull a least-squares slope, and neither robust model
lets them.** Here about 2% of the rows are 50 too high, all where `x` is
above 0, so they tilt the line:

```python
import numpy as np
import polars as pl
import polars_online as po

# y = 1 + 2x plus noise, and about 2% of the rows wild: 50 too high, on rows where x is above 0.
rng = np.random.default_rng(5)
n = 10_000
x = rng.standard_normal(n)
y = 1.0 + 2.0 * x + 0.5 * rng.standard_normal(n)
wild = (rng.random(n) < 0.04) & (x > 0)
y[wild] += 50.0
pl.DataFrame({"x": x, "y": y}).write_parquet("wild.parquet")

# Least squares beside the two robust models, every row weighed the same.
common = dict(targets=["y"], features=["x"], half_life=float("inf"))
specs = [
    po.spec.ewridge("ewridge", **common),
    po.spec.huber("huber", **common),                         # huber_delta=1.345 and huber_scale="mad", the defaults
    po.spec.huber("huber_rms", huber_scale="rms", **common),  # the cut drawn in the residuals' root mean square
    po.spec.quantile("median", quantile=0.5, **common),
]
bank = po.ModelBank(specs)
bank.fit(pl.scan_parquet("wild.parquet"))
# Each model's slope, beside the truth.
print(f"wild rows: {wild.sum()} of {n}; the true slope: 2")
for r in bank.coef().filter(pl.col("term") == "x").iter_rows(named=True):
    print(f"{r['spec']}: slope {r['coef']:.2f}")
```

```text
wild rows: 218 of 10000; the true slope: 2
ewridge: slope 2.89
huber: slope 2.02
huber_rms: slope 2.18
median: slope 2.03
```

`huber` still learns from a wild row, at a weight that shrinks with its
miss, so it lands near the truth and not exactly on it. The median takes
only the sign of a row's miss beyond the band, so a wild row counts no more
than an ordinary one. `huber_rms` draws its cut in a scale that the wild
rows themselves widen, so it lets more of their pull through (the scales
are below).

**Huber's loss is quadratic for a small miss and linear for a large one.**
With `u = e/σ` the residual `e` in units of a scale `σ`, and `δ` the cut
`huber_delta`:

```
ρ(u) = u²/2           |u| ≤ δ
     = δ|u| − δ²/2    |u| > δ
ψ(u) = ρ'(u) = u inside the cut, ±δ beyond it
```

`ψ` is a row's pull on the fit, and it stops growing at the cut. Least
squares has `ρ(u) = u²/2`, so one row that misses by 50 pulls as hard as
10,000 rows that miss by 0.5. Writing `ψ(u) = w(u)·u` with
`w(u) = min(1, δ/|u|)` turns the fit into weighted least squares, each row
at the weight `w` of its own residual. A batch fit iterates: fit, compute
the residuals, reweight, fit again. `huber` weighs each row once, by its
residual against the fit before the row, and keeps only the weighted sums.
The exception is each target's first 500 learned rows, which it refits to
convergence once (`huber_refit_rows=`, below).

`e` is the row's residual, and `σ` the target's residual scale (below).
`δ` is `huber_delta` and `τ` the `quantile` level. The *Gram* is
`ewridge`'s `S`, and the cross-moment its `η_j`:

```
huber:     the ridge update at weight  w · min(1, δσ / |e|)
quantile:  one Newton step on the check loss, smoothed by a uniform kernel of half-width h = quantile_eps · σ
           |e| < h:  a least-squares row with target  y + 2h(τ − ½)
           |e| ≥ h:  adds  w · 2h · ψ_τ(e) · z  to the cross-moment and nothing to the Gram,  ψ_τ(e) = τ − 1{e < 0},
                     bounded so the fit at the row moves by at most the row's residual
           h is never narrower than (k/n)^{2/5} · σ, for the target's effective sample n
```

The weights are per target, so `huber` and `quantile` keep one set of
sums per target, where `ewridge` shares one set across targets.

**A label that arrives late is weighed when it is learned.** Under an
`embargo`, or with a target that is a window expression looking ahead, a
row is scored where it arrives and learned once its label is known, the
held rows in the order they arrived. Its weight is read then, from its
residual against the fit at that moment: the fit of every row before it
and of none after it, which is the fit it would be weighed by without the
delay. So the delay moves the predictions, which come from an older fit,
and leaves the fit, the weights and the scale as they are without it.
`coef` under an embargo of `E` rows is the undelayed run's `coef` of `E`
rows earlier, to the bit. The weight is out of sample either way, since
the row's own label is in neither fit. `huber_scale_<t>` is read where the
row is scored, as `pred_<t>` is.

**`huber`'s `feature_sets=` fits subsets of the features from the same
sums, as `ewridge`'s do, and weighs each row by the largest set's
residual.** A set's fields are `pred_<t>__<set>` and `resid_<t>__<set>`.
The sets share the target's sums, and a sum holds each row at one weight,
so a row's Huber weight is one number per target. It is read from the
residual of the largest set, the one with the most features, the first of
them where two tie, and so is the scale `σ`. Weighting each set by its own
residual would need a set of sums per set. So the largest set is the Huber
fit of its features. It is the same to the bit as a `huber` on those
features alone when it is every feature, or when both are given the same
`min_weight`. The default `min_weight` counts a coefficient for every
feature of the spec, so a largest set short of them starts predicting later
than the `huber` on its features alone. Its first Huber weights and its
scale then differ, and the two fits part until the decay forgets those
rows: on a stream with a noise of 0.4 and outliers of 40, at
`half_life=300`, by up to 0.17 in the first 30 rows and 1e-4 after 1,200.
A smaller set is the weighted least-squares fit of its features at
the weights the largest set's residuals chose, which is not a Huber fit of
its own. `quantile` takes no `feature_sets`: its rows outside the band step
the fit through one system per target.

**`huber`'s `ridge_scale="sum"` is the Huber form of `rls`.** It puts RLS's
prior on the decaying sum scale in place of the ridge on the means, as
`ewridge`'s does: `(W S + prior_scale · ridge · I) b = W r`, with `W` the
weight the target's sums hold after the Huber weights, and the intercept
penalized too. The prior fades as the rows arrive. At `huber_delta=inf`,
solved on every row, it gives `rls`'s predictions with `delta=ridge`
within 1e-9, as `ewridge(ridge_scale="sum")` does. It is refused beside
`standardize` and beside two or more `feature_sets`.

**`emit_huber_scale=True` writes `huber_scale_<t>`, the scale each row's
cut is drawn in.** It is read after the row's decay and before the row's
own residual enters the scale, so a wild value cannot widen its own scale:
the field is causal by construction. Use it as a row weight downstream, or
as the scale of a robust loss on the residuals. It is one field per target,
the largest set's under `feature_sets`. It is null until the target has a
`σ` above 0, and on a row the model does not read because a feature is
null or not usable (NaN, ±inf or beyond 1e100), as `lasso`'s
`penalty_selected_<t>` is.

**The scale `σ` is the part of Huber's estimator that makes it robust.**
The cut is `δσ`, so a scale that the outliers inflate moves the cut out of
their way. `huber_scale=` chooses between two scales:

| `huber_scale` | `σ` | what one wild row does to it |
|---|---|---|
| `"mad"`, the default | the exponentially weighted median of `\|e\|`, over `Φ⁻¹(3/4) = 0.6745` so that it is the standard deviation when the errors are normal | moves it by the row's weight, whatever the size of the miss, so the scale holds until half the weight is outliers |
| `"rms"` | the exponentially weighted root mean square of `e`, about 0 | sets it: every row counts at full weight, a wild row included |

The median is read from a quantile sketch kept per target, within 0.78% of
the exact weighted median. Each scale learns a residual at the row's own
weight, not at its Huber weight, and ages on every row.

The sketch keeps each residual in units of the target's first one, so
rescaling the target rescales the fit to rounding. Its buckets are fixed
in the residual's own units, and only a power of two maps them onto
themselves. Kept in those units, a target in grams instead of kilograms
regrouped residuals across bucket edges and moved the median by up to a
bucket, 1.6%. Over 50 streams of 1,500 rows with heavy-tailed noise and a
half-life of 300 rows, predictions moved by up to 0.4% of the target's
spread at scales of 1e±3 and 1e±6. In the first residual's units they move
by at most 3e-12 of the spread, under the 2e-11 the fit's own rounding
allows. The read costs a division more a row, about 2 ns per target. A
state saved before this change keeps its fixed buckets and continues as it
was.

The sketch's buckets span at most 16 octaves around the median, a factor
of 65,536. A residual beyond them is folded into the outermost bucket on
its side, its weight kept, so a wild residual of `1e200` costs no more
memory than any other.
While half the weight or more is on residuals of exactly 0, the median is
0, and the 16 octaves reach up from the smallest residual instead: the
median steps there when it leaves 0.
Nothing is folded while the residuals span less than that, and the median
stays the one an unbounded sketch reads unless it later moves past a
bucket a residual was folded into, at least a factor of 256 from where the
median stood then. A fit's first residuals, at the target's level while
the intercept is still near 0, do that when they are more than 65,536
times the noise's: the median then reads too high until the later
residuals outweigh the first ones. The sketch costs at most about 9 KB of
state per target. Against `"rms"`, with 20 features and 10 targets over
100,000 to 200,000 rows on an Apple M4 Pro under other load, it adds 14 to
17 ns per target per row to `huber` and 20 to 32 to `sgd(loss="huber")`,
whose step is cheaper.

**The root mean square is not robust, and the median is.** Measured on
`y = 1000 + 2x + 0.5ε` over 20,000 rows with no decay, and a wild row 50
too high on a share of the rows with `x` above 0, as the median slope over
five seeds, with the other settings at their defaults:

| wild rows | least squares | `"mad"` | `"rms"` | statsmodels `RLM`, MAD scale, iterated to convergence |
|---|---|---|---|---|
| none | 1.999 | 1.999 | 2.000 | 2.000 |
| 1% | 2.415 | 2.008 | 2.057 | 2.008 |
| 5% | 4.050 | 2.040 | 2.675 | 2.039 |
| 12.5% | 6.975 | 2.124 | 4.882 | 2.124 |

The root mean square fails twice. Its first residuals are the target's whole
level, about 1,000 while the intercept is still near 0, and without decay
they stay in it for good. And the wild rows inflate the scale that is meant
to catch them. The median leaves the first residuals behind once the fit has
the level. The small bias left at 1% and 5% is Huber's own: a wild row still
teaches, at about `1.345 · 0.5 / 50 ≈ 0.013` of an ordinary row's weight,
and hundreds of them add up.

**A median over five streams hides the streams the outliers capture, so
`huber` refits each target's first rows to convergence.** One pass weighs
each row by the fit before it. Over the first rows that fit is close to
least squares, so the wild rows pull it, and `σ` learns the pulled fit's
residuals, several times the noise's scale. The cut `δσ` is then wide, the
wild rows keep entering at nearly their full weight, and the fit stays where
they pulled it. Without decay it stays for good, since the first rows keep
their share of the sums and `σ` its memory of them. So on the row a
target's count of learned rows reaches `huber_refit_rows` (by default 500,
or 10 a feature from 51 features to 100, and off above 100),
`huber` refits those rows by iteratively reweighted least squares, as a
batch fit does:

```
v_i = w_i · λ_{i+1} ⋯ λ_n                 each buffered row's weight, decayed to the refit's row n
                                          (v_i' the same by the scale's clock, huber_scale_half_life)
r_i = y_i − z_i·β                         against the last pass's fit (the first pass: the online fit)
σ   = weighted median of |r_i| at v_i', over 0.6745       (the weighted root mean square under "rms")
u_i = min(1, δσ / |r_i|)
β   = the ridge solve of the rows at v_i · u_i            repeated until no row's fit z_i·β moves by over 1e-10 of σ, or 200 passes
```

It then rebuilds the target's sums from the rows at their converged
weights, and `σ` from the converged fit's residuals, and goes on online.
Rebuilding `σ` is what keeps the refit: a refit that kept the pulled fit's
`σ` was pulled back within a few hundred rows. The refit's row is predicted
before any of it, so the prediction stays out-of-sample. It moves only its
own target's fit, and the solve schedule stays as it was. `weight_sum`,
and each target's weight that `min_weight` reads, count the raw weights
and do not move.

The refit takes a target from the first poor fit to the batch fit.
Measured on the same stream as the table above, over 40 seeds, as the
streams whose final slope is more than 0.05 from statsmodels' `RLM`, then
more than 0.25, and the worst distance:

| wild rows | 20,000 rows, `huber_refit_rows=0` | 20,000 rows, 500 (the default) | 2,000 rows, 0 | 2,000 rows, 200 | 2,000 rows, 500 | 2,000 rows, 1,000 |
|---|---|---|---|---|---|---|
| 5% | 0 and 0 of 40; worst 0.014 | 0 and 0; 0.001 | 4 and 0; 0.17 | 0 and 0; 0.004 | 0 and 0; 0.003 | 0 and 0; 0.003 |
| 10% | 6 and 0; 0.19 | 0 and 0; 0.004 | 12 and 6; 0.49 | 0 and 0; 0.008 | 0 and 0; 0.006 | 0 and 0; 0.005 |
| 12.5% | 13 and 2; 0.43 | 0 and 0; 0.007 | 15 and 13; 0.97 | 0 and 0; 0.013 | 0 and 0; 0.009 | 0 and 0; 0.007 |
| 15% | 15 and 12; 1.08 | 0 and 0; 0.012 | 20 and 18; 2.33 | 1 and 0; 0.054 | 0 and 0; 0.016 | 0 and 0; 0.019 |
| 20% | 25 and 23; 4.30 | 4 and 0; 0.14 | 29 and 24; 6.68 | 18 and 5; 5.76 | 10 and 1; 0.32 | 3 and 0; 0.15 |

The median stream was within 0.002 of `RLM` at 12.5% either way, which is
why five seeds did not show the captured ones. With a half-life of 5,000
rows over 20,000 rows, the refit took the captured streams at 15% and 20%
from 15 and 26 of 40 to 0 and 2. At 20% wild rows, 40% of those with `x`
above 0, a first window of 200 rows can itself be captured, and one of 500
rows now and then; 1,000 rows holds there. Over 20,000 rows 200 rows left
the same 4 streams at 20% as 500 do, the worst by 0.18; over 2,000 rows the
table shows what the longer default buys. `huber_refit_rows=0` turns the refit off.
Under `huber_scale="rms"` it gives `huber` the numbers of the releases before it,
to the bit; under the MAD, the default, they move by up to a bucket of the MAD's
sketch, which now keeps each residual in units of the first one it takes. A state
saved before the refit existed goes on to the bit. `sgd`'s
steps have changed since (its clip, AdaGrad's epsilon and its intercept read the
target's own spread and level), so for it the refit off is the one-pass fit of
today's steps.

**The refit holds each target's rows until it runs, then frees them.** A
learned row takes `(features + 4) · 8` bytes. The rows that only aged the
target -- its value null, or its weight 0 -- take 32 bytes a run, one run
for such rows in a row at one clock step. So one feature and 500 rows take
about 20 KB a target. A group shorter than `huber_refit_rows` never refits
and holds its rows to its end: measured with 5,000 groups of 150 rows, two
targets and 20 features and no bound, the bank's peak resident memory was
87 KB a group more than without the refit, and its saved state 63 KB more.
Past four runs a learned row of
`huber_refit_rows` the target gives its refit up, frees its rows and keeps
its one-pass fit: under a regular clock that never happens, since a gap of
any length is one run, and under an irregular one it happens to a target
present on fewer than about one row in five. The refit runs once, in 0.06
to 7.2 ms for 1 to 50 features and 200 to 1,000 rows (an Apple M4 Pro under
other load, best of 9), and a row after it costs what it did, within 2%.
The default count grows with the features, 10 rows a feature past 50, so
that the refit has rows enough for its coefficients, and stops at 100
features, where a target's rows would pass 0.8 MB. A count given is taken at
any width: at 500 features and more, 200 rows are fewer than the
coefficients' count, the fit is the ridge's, and the iteration runs to its
cap of 200 passes, about 9 s once at 1,000 features (2 s at 500), which
`solve_failures` counts. The cap was 50 passes, about 2 s at 1,000
features; 200 is where the slow refits of heavy one-sided outliers end:
over 3,000 streams of 50 to 500 rows with up to 45% of the rows on one side
moved by a level, 50 passes ran out on 149 and 200 on 9, all nine past 39%.

**`refit_budget` bounds the buffers of all a spec's groups together: 256
MiB by default, or `inf` for no bound.** Each new group reserves, when it
first appears, the most its targets' buffers can hold: `(features + 20) · 8`
bytes a target for each row of `huber_refit_rows`, the row and its four
runs. The group that would cross the bound, and every new group after it,
gives its refit up and keeps its one-pass fit; a notice says so once, and
`summary()`'s `refit_given_up` counts the targets. A reservation lasts the
bank's life, even after the refit frees the buffer, and is saved with the
bank, so which groups refit depends neither on the chunking nor on the order
the groups run in, nor on a save and load between them. The price is that
once the bound is reached no later group refits, though the first groups'
buffers have long been freed. At 20 features and 500 rows a target reserves
160 KB, and 256 MiB refits the first 1,677 targets. Measured on the 150-row
groups above, fed from parquet, as the peak resident memory after the fit and
the saved state (at 20,000 groups the median of three runs, which spread by
up to 130 MiB; the reservation is the most a buffer can hold, and these held
less):

| groups | `huber_refit_rows=0` | the default, 256 MiB | `refit_budget=inf` |
|---|---|---|---|
| 5,000 | 660 MiB peak; state 126 MiB | 749 MiB; 178 MiB; 8,324 of 10,000 targets given up | 1,084 MiB; 435 MiB |
| 20,000 | 1,625 MiB; 502 MiB | 1,641 MiB; 556 MiB; 38,324 of 40,000 given up | 3,161 MiB; 1,741 MiB |

**Two settings have no refit: it is off by default there, and refused when
asked for.** Under `ridge_scale="sum"` the prior has aged with every row
before the first, which a refit of the buffered rows cannot hold. Under
`huber_delta=inf` the fit is least squares, with no wild rows to refit for.
Reweighting passes over a trailing window of rows would leave the first
rows' weights and `σ` as they are, so they would free no captured fit.

**A robust scale slows the fit after a change of level.** In a series, a
run of large residuals is either a burst of outliers or a new level, and no
rule on one row tells them apart. The median treats every row after a level
shift as an outlier until the median itself has moved, which holds the fit
back. `huber_scale_half_life=` gives the scale a memory of its own, the
spec's `half_life` by default: shorter than the fit's, the scale follows a
shift sooner. The price is that a longer burst of outliers widens the cut.
Measured on `y = 2x + 0.5ε`, with `x` and `ε` standard normal (seed 1,
14,000 rows), a fit half-life of 500 rows, a shift of +20 at row 10,000 and
no outliers, as the mean absolute error after the shift:

| scale | rows 0-200 after the shift | rows 200-400 |
|---|---|---|
| least squares (`huber_delta=inf`) | 17.6 | 13.3 |
| `"rms"` | 19.0 | 15.1 |
| `"mad"`, the scale's half-life the fit's 500 rows | 19.9 | 19.5 |
| `"mad"`, `huber_scale_half_life=100` | 19.2 | 14.6 |

The fit's own half-life sets how long the fit takes to settle at the new
level, about 2,000 rows in every row of the table. The break tests of
[DIAGNOSTICS.md](DIAGNOSTICS.md#has-it-broken) flag such a shift as it
happens.

**Huber's weights read the residual alone, so they do not protect against a
wild feature.** A row whose features are far from the rest pulls the fit
toward itself, so its residual is small and it keeps its full weight. Wild
rows concentrated in one region of `x` act the same way, because there they
are the majority: at 30% of the rows (60% of those with `x` above 0) the
median breaks too, and every estimator that reads only residuals does.

**Until a target has a `σ` above 0, `huber` down-weights no row.** That is
before its first residual, or while every residual so far is exactly zero
(under `"mad"`, while half the weight is on such residuals): there is
nothing to judge an outlier against. `quantile` has no band to draw there,
and takes least-squares rows. `quantile`'s `σ` is always the MAD, on the
model's clock: it sets the width of the band, not a weight. In the root
mean square, wild rows widened the band drawn to bound their pull: at 5%
of the rows lifted by a hundred noise deviations on one side of `x`, every
one of 50 streams of 20,000 rows ended more than 0.05 from statsmodels'
`QuantReg` on the same rows (the worst 0.098 at 5%, 0.98 at 20%); in the
MAD none did (the worst 0.006 at 5%, 0.026 at 20%).

**`min_weight` withholds the output alone.** On a row it withholds, both
models still learn from the prediction the fit made: the scale takes its
residual and `huber` its weight, so the fit is the same at any
`min_weight`. A target's first prediction, reported or learned from, waits
instead for its own first row per coefficient, the models' floor, since a
fit of fewer rows than unknowns is the ridge's alone.

```python
hub = po.spec.huber("hub", targets=["y"], features=["x0", "x1"], clock="t", gap_cap=300.0, half_life=600.0,
                    huber_delta=1.5,              # a residual beyond huber_delta * sigma is down-weighted (default 1.345)
                    huber_scale_half_life=120.0,  # sigma's own memory, shorter than the fit's (default half_life)
                    huber_refit_rows=1000)        # each target's first rows refit to convergence (default 500; 0 is off)
med = po.spec.quantile("med", targets=["y"], features=["x0", "x1"], clock="t", gap_cap=300.0, half_life=600.0,
                       quantile=0.5,             # the level, strictly between 0 and 1: 0.5 is a median regression
                       quantile_eps=0.2,         # the band's half-width, in units of sigma (default 0.2)
                       quantile_refit_rows=1000) # each target's first rows refit to convergence (default 500; 0 is off)
```

**`huber_delta`'s default, 1.345, is the cut at which Huber's estimator is
95% as efficient as least squares when the errors are normal.**
statsmodels' `HuberT` takes 1.345 and scikit-learn's `HuberRegressor`
1.35. `huber_delta=inf` cuts nothing, which is least squares. To come
close to the numbers of a version before the refit existed (1.3.0 and
earlier), give `huber_refit_rows=0`: under the MAD they still move by up to
a bucket of its sketch, which now keeps each residual in units of the first
one it takes. Before `huber_scale` existed (1.2.0 and earlier), give
`huber_scale="rms"` as well, which keeps no sketch. A state saved by an
earlier version goes on to the bit.

**The band's rows, those with `|e| < h`, give the Newton step its
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

**`quantile` refits each target's first 500 learned rows to convergence
once, as `huber` does (`quantile_refit_rows=`).** One pass keeps each row in
its sums as it took it, and without decay keeps it there at full weight: its
warm-up's rows as least-squares rows, and the rows inside a band drawn around
a fit that wild rows pulled. A batch fit reads those wild rows as outliers,
whose pull the band's width bounds; the one pass lets them go only as the rows
after them outweigh them. On the row a target's count of learned rows is
reached, its rows are refit to the minimum of the smoothed check loss at the
band `h = σ·max(quantile_eps, (k/n)^(2/5))` its own residuals draw, `σ`
their weighted median `|e|` over `Φ⁻¹(3/4)`. Each pass is a Newton step of
that loss: the rows inside the band are least-squares rows at target
`y + 2h(τ − ½)`, each row outside adds `2h·ψ(e)·z` with no cap, and a step
that would raise the loss is halved until it does not. A band of fewer rows
than coefficients has no curvature to step on; such a pass moves each
prediction toward its row by at most the band's width instead. The sums are
then rebuilt as the stream takes each row scored by the converged fit, or,
where the converged band holds fewer rows than coefficients, as the
majorizer's step takes it, every row a least-squares row at its weight, so
the sums keep a weight and solve to that fit; `σ` is rebuilt from its
residuals, and the stream goes on online. Measured over 50 streams of
8,000 rows of `y = 1 + 2x₀ − x₁ + 0.5·Laplace`, no decay, as the worst
stream's largest coefficient gap to statsmodels' `QuantReg` on the same rows:

| the first 500 rows | `quantile_refit_rows=0` | 500 (the default) |
|---|---|---|
| clean | 0.0057 | 0.0052 |
| a fifth wild, +50, where `x₀ > 0` | 0.038 | 0.0085 |
| the first three 1e4 off | 0.22 | 0.0053 |
| t(2) noise | 0.0078 | 0.0058 |
| Cauchy noise | 0.0175 | 0.0058 |
| a shift of 20 | 0.093 | 0.061 |

A shift of every one of the first rows is no outlier among them, so the
refit keeps them as the shifted fit's rows, and the one pass lets them go
as before, faster than `1/n`. A state loaded from another regime has
refit its first rows there, and the same holds of its rows. The refit's
cost, its buffer, the count's default and `refit_budget` are `huber`'s
(below); it takes 0.09 to 2.3 ms once, for 1 to 50 features and 500 to
1,000 rows. A count given is taken, but one below three rows a
coefficient, the rows the warm-up takes as least squares, is refused.

**Both models' one pass is held to numpy references of its own recursion
within 1e-14, and `huber`'s refit to statsmodels' `RLM` on the same rows
within 1e-5.** Against batch fits of the same objectives over a whole
stream, scikit-learn's `HuberRegressor` and statsmodels' `QuantReg`, they
land close but not equal, for three reasons:

| reason | applies to |
|---|---|
| each row's weight was set by the fit before it | both, for each row after its target's first `huber_refit_rows` or `quantile_refit_rows` |
| `σ` is an exponentially weighted median, read from a sketch, where a batch fit takes the median of the final residuals | both |
| the check loss is smoothed over the band | `quantile` |

### `sgd` — stochastic gradient descent

*API:* [`po.spec.sgd`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.sgd) — *Rust:* [`sgd.rs`](../crates/online-core/src/sgd.rs) — *Outputs:* [fields](OUTPUTS.md#sgd)

Use `sgd` when each row must take only O(k) time: it takes one gradient
step per row and never solves. It is the only model here that takes count
targets (`loss="poisson"`), and its coefficients can be bounded or summed
to a constant, as portfolio weights are.

**A count target takes the Poisson loss, whose log link makes `pred` a
rate.** Here the counts' log-rate is `0.5 + 0.3·x0 − 0.2·x1`, and adagrad
adapts each coefficient's rate to its own gradients:

```python
import numpy as np
import polars as pl
import polars_online as po

# A count target: y is Poisson with log-rate 0.5 + 0.3·x0 − 0.2·x1.
rng = np.random.default_rng(6)
n = 50_000
x0, x1 = rng.standard_normal((2, n))
y = rng.poisson(np.exp(0.5 + 0.3 * x0 - 0.2 * x1)).astype(float)
pl.DataFrame({"x0": x0, "x1": x1, "y": y}).write_parquet("counts.parquet")

# The Poisson loss, fitted one gradient step per row.
counts = po.spec.sgd(
    "counts", targets=["y"], features=["x0", "x1"], half_life=float("inf"),
    loss="poisson",                          # a negative target refuses the chunk, naming the row
    schedule="adagrad", learning_rate=0.05,  # a rate per coefficient, adapted to its gradients
)
bank = po.ModelBank([counts])
bank.fit(pl.scan_parquet("counts.parquet"))
print("truth:  {'intercept': 0.5, 'x0': 0.3, 'x1': -0.2}")
print("fitted:", {r["term"]: round(r["coef"], 2) for r in bank.coef().iter_rows(named=True)})
```

```text
truth:  {'intercept': 0.5, 'x0': 0.3, 'x1': -0.2}
fitted: {'intercept': 0.51, 'x0': 0.31, 'x1': -0.21}
```

With `Gᵢ` adagrad's running sum of squared gradients, a row takes this
step, with `z = [1, x]` and `β` in the caller's units under
`standardize=False`:

```
eta = zᵀβ        p = link(eta)        d = dL/d eta
g₀ = d·w         gᵢ = d·zᵢ·w + l2·βᵢ     each clipped to ±clip_gradient·u; the intercept takes no l2
βᵢ -= lrᵢ·gᵢ     lrᵢ = learning_rate                           schedule="constant"
                     = learning_rate/(1 + λ·weight_sum)^power  schedule="inv_scaling"
                     = learning_rate/(√Gᵢ + 1e-8·u)            schedule="adagrad"
```

`u` is the unit the gradient is in: `s_y`, the target's own spread (below),
under `squared` and `huber`, whose gradient is in the target's units, with
no clip before the target has one; 1 under the other losses, whose gradient
is a sign, a count's residual or a probability's. So the clip and the
epsilon mean the same at every scale of the target: in the target's units,
the default clip bound every gradient of a target in the millions, and its
fit was 0.59 to 6.5 of the target's spread off the unscaled one, scaled.

**The intercept takes the running mean's step while it is the larger.** The
coefficients start at 0, so the first residuals are the target's level, and
left to the steps it moved every slope by about the rate times the level:
2.49 off least squares after 8,000 rows at a level of 1,000 on the worst of
50 streams, against 0.0096 at a level of 0. Under an identity link (every
loss but `poisson` and `logistic`) with an intercept, with `W` the
target's learned weight (undecayed, the row's in it) and `a = w/W`, the
intercept's step is the running mean's, `−a·d` for `squared` and `huber`
while `a` is above the rate or the rows number at most 22 by Kish's count,
`a·(y − p)` for the sign-valued losses while they do, and the target's first
row puts its whole residual there and moves nothing else. Now a level costs
rounding (0.0077 at either level), and a target with no level loses
nothing measurable.

**A rate past its stability bound is withheld.** A step multiplies its own
row's residual by `1 − q`, `q = learning_rate·(w·c·|z|² + l2)` with `c` the
loss's curvature at the prediction (1 for `squared` and inside the Huber
cut, 0 beyond it, `p` for `poisson`, `p·(1 − p)` for `logistic`), so the fit
diverges in mean square once `E[q²] > 2·E[q]`. Each target keeps both sums
over its recent steps (each decayed by `2^(−1/100)` a row), and while the
squares' passes twice the gains' its predictions are withheld as
`unstable_rate`, with a notice; the fit learns every row as before. On five
standardized normal features the bound on a constant rate is 0.261; at 0.5
the fit had diverged until `clip_gradient` held it, its error 50,000 times
the target's variance, with nothing said. `inv_scaling` from a rate past
the bound is withheld on its first rows only. `adagrad` and the sign-valued
losses take steps that cannot grow, and are never withheld.

`λ` is the row's decay, so `λ·weight_sum` is the weight before the row
decayed to it, and `Gᵢ` is decayed by it too. A row of weight 0 in between
therefore leaves the next row's rate where it would be without that row.

Under `standardize`, the default, `z` is the row standardized,
`zᵢ = (xᵢ − mᵢ)/sᵢ`, and the ridge reads `l2·sᵢ·βᵢ`. Once the scaler has
warmed up, the step taken in `z` is mapped into the caller's units:

```
βᵢ -= lrᵢ·gᵢ / sᵢ        β₀ -= lr₀·g₀ − Σᵢ mᵢ·lrᵢ·gᵢ / sᵢ
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
| `huber` | identity | `clamp(p − y, ±delta·s)` | `delta` is `huber_delta=`, 1.345 by default, in units of `s` as `huber`'s is; `huber_scale=` and `huber_scale_half_life=` choose `s` as for `huber`, and `huber_refit_rows=` sets the first rows' refit (below) |
| `quantile` | identity | `1{y < p} − τ` | `τ` is `quantile=`, the level, between 0 and 1 |
| `epsilon_insensitive` | identity | 0 within `eps·s_y` of `y`, else `sign(p − y)` | the tube's half-width is `eps=`, 0.01 by default, in units of `s_y` |
| `poisson` | log, for count targets | `p − y` | a negative target refuses the chunk, naming the row |
| `logistic` | sigmoid, for 0/1 targets | `p − clamp(y, 0, 1)` | `strict_binary=True` refuses a chunk with a label not 0 or 1 |

**`s` is the scale of the target's out-of-sample residuals**, `huber`'s
`σ`: the exponentially weighted median of `|y − p|` over 0.6745 by default
(`huber_scale="mad"`), or their root mean square (`"rms"`). **`s_y` is the
target's own spread**, around its exponentially weighted mean. So a cut and
a tube mean the same thing at any scale of the target. Until the target has
an `s`, the Huber row is a squared-loss row. Until it has an `s_y`, which
takes two weighted rows of different values, the tube has no width. In the
target's own units a tube of 0.1 never let a target in thousandths be
learned. `sgd` standardizes its features by default, as `kalman` does: one
learning rate has to suit every feature, and raw, features in hundreds took
the default fit's R² from 0.96 to below −70,000.

**Once the scaler has warmed up, the coefficients are held in the caller's
units.** Until it has learned 22 rows by Kish's count, the fit runs in its
standardized coordinates and is read out through its moments. From then on
each step is taken on the standardized row and mapped back by that row's
means and scales, so the scaler's moving moves no prediction. Read through
the moments instead, a half-life of 10 put the slope at 3.03 for a truth of
2 at R² 0.978. Held in the caller's units, it is 2.02.

**The cut reads the residuals and the tube does not, because the fit starts
from zero.** A residual's spread can carry the target's level. Beyond the
cut the gradient is clipped, not zero, so a cut those residuals widen still
lets every row teach, and the median scale leaves them behind once the fit
has the level. The intercept takes the level from the first row (above), so
the root mean square no longer keeps it, but the outliers still inflate it.
On `y = 1000 + 2x + 0.5ε` with a row 50 too high on 5% of the rows, all
with `x` above 0, and no decay, the median slope over five seeds is 3.751
under the squared loss, 2.557 under `"rms"` and 2.013 under the default
`"mad"`, with the refit below or without it. Over 40 seeds of 20,000 rows at
5% wild rows every stream stayed more than 0.05 from `RLM` under `"rms"`,
and 36 of them more than 0.25 (a median gap of 0.60), with the refit or
without it; before the intercept took the level, all 40 were without it
(1.85). Inside the tube the gradient is zero. A tube drawn in
the residuals' spread on a target at 1,000 with a spread of 2 was about 100
wide and held every row, and without decay it never narrowed.

**`sgd`'s Huber loss refits each target's first 500 learned rows to
convergence, as `huber` does.** Its first steps start from zero slopes, so
the outliers capture it as they capture `huber`'s first solves: over 2,000
rows at 20% wild rows, 28 of 40 streams ended more than 0.05 in slope from
statsmodels' `RLM` in one pass, the worst by 3.6 (at 10%, 4 of them, the
worst by 0.069; all 40 more than 0.25 off before the intercept took the
level from the first row). On the row a target's count of
learned rows reaches `huber_refit_rows`, the converged Huber fit of those
rows replaces its coefficients, and its scale `s` is rebuilt from that
fit's residuals. The fit is `huber`'s iteration (in its section), started
from the rows' weighted least squares, as statsmodels starts, with a ridge
of 1e-6 on the mean-form system. It leaves out `l2`, a penalty on the
steps, which the steps after it apply. The coefficients are set in the
coordinates the fit is held in: the caller's units, or, while the scaler
still warms up, the coordinates of its moments as they stand, so that they
read out as the fit. A constraint projects the fit as it projects a step.
The learning rate's state stays. Adagrad's sums of squared gradients keep
the step sizes the rows taught; emptied, they would make the next steps
throw the fit off. `inv_scaling` reads the weight, which the refit does not
move. Measured on
the stream above, over 40 seeds, as the streams more than 0.05 and more
than 0.25 from `RLM`, and the worst distance:

| wild rows | 20,000 rows, `huber_refit_rows=0` | 20,000 rows, 500 (the default) | 2,000 rows, 0 | 2,000 rows, 200 | 2,000 rows, 500 | 2,000 rows, 1,000 |
|---|---|---|---|---|---|---|
| 5% | 9 and 0 of 40; worst 0.097 | 9 and 0; 0.091 | 30 and 0; 0.16 | 5 and 0; 0.065 | 5 and 0; 0.065 | 5 and 0; 0.065 |
| 10% | 14 and 0; 0.15 | 9 and 0; 0.11 | 40 and 40; 1.16 | 2 and 0; 0.075 | 2 and 0; 0.075 | 2 and 0; 0.076 |
| 12.5% | 22 and 0; 0.19 | 11 and 0; 0.11 | 40 and 40; 2.50 | 7 and 0; 0.098 | 7 and 0; 0.097 | 7 and 0; 0.098 |
| 15% | 36 and 4; 0.35 | 12 and 0; 0.13 | 40 and 40; 4.20 | 10 and 0; 0.076 | 9 and 0; 0.078 | 10 and 0; 0.079 |
| 20% | 40 and 40; 2.89 | 24 and 0; 0.16 | 40 and 40; 8.55 | 16 and 4; 2.98 | 8 and 0; 0.14 | 8 and 0; 0.13 |

The streams still more than 0.05 away after the refit are the constant
learning rate's own wander, a median of about 0.03 at every share of wild
rows. Under `huber_delta=inf`, the squared loss, the refit is off, and
refused when asked for.

**Under the default `"mad"`, `sgd`'s Huber loss follows a change of level
far more slowly than under `"rms"`.** Its gradient is clipped at
`delta·s`, so while `s` is the noise's scale the intercept moves at most
`learning_rate·delta·s` a row, and a shift many times the noise waits
until the median itself has crossed to the new level. For `huber` the
median cost 19.5 against 15.1 under `"rms"` in the table above; for `sgd`
it costs seven times as much. On
`huber`'s stream above (a fit half-life of 500 rows, a shift of +20 at row
10,000, no outliers), the mean absolute error after the shift, and the
rows until 200 in a row average under 1.0:

| `sgd` | rows 0-200 after the shift | rows 200-400 | rows to settle |
|---|---|---|---|
| `loss="squared"` | 8.67 | 1.17 | 418 |
| `loss="huber"`, `"rms"` | 13.55 | 2.45 | 493 |
| `loss="huber"`, `"mad"`, the scale's half-life the fit's 500 rows | 19.26 | 17.33 | 885 |
| `loss="huber"`, `"mad"`, `huber_scale_half_life=100` | 15.81 | 2.99 | 510 |
| `loss="huber"`, `"mad"`, `huber_scale_half_life=20` | 10.34 | 1.41 | 440 |

Seeds 2 and 3 agree to within 0.06. **On a stream whose level can change,
give `huber_scale_half_life` about a fifth of `half_life`, or shorter.** A
shorter memory widens the cut sooner in a long burst of outliers too.

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
weight 1, so at the default rate of 0.01 a target that moves by 1,000 is
100,000 rows from its new level. Its first level, the first learned row
sets.

**Under `loss="poisson"`, keep `clip_gradient`, `1e3` by default,** because
through the log link one large count would make the next gradient
exponentially bigger. At ordinary scales it does not bind for the other
losses. A Poisson fit takes counts, so a negative target refuses the chunk
and names the row, as scikit-learn's `PoissonRegressor` refuses one. Taken
as it stood, `p − y` drove the prediction down to the link's floor,
`e^−30`, and held it there.

**To bound the slopes, give `coef_min` and `coef_max` (one number, or one
per feature, with `-inf` or `inf` for an open side) and `coef_sum` for
their total:**

| setting | what it expresses |
|---|---|
| `coef_min=0, coef_sum=1` | a long-only, fully invested portfolio |
| `coef_min=0` alone | a sign the model must respect |
| `coef_min` equal to `coef_max` | a slope pinned at a known value |

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

After every step the slopes move to the nearest point that satisfies every
bound (the Euclidean projection). The fit starts from zero, projected, so a
fit on the simplex starts at uniform weights. The intercept is never
constrained, and `coef` reports what the projection returned, in the
caller's units even under `standardize=True`.

### `pa` — passive-aggressive regression

*API:* [`po.spec.pa`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.pa) — *Rust:* [`pa.rs`](../crates/online-core/src/pa.rs) — *Outputs:* [fields](OUTPUTS.md#pa)

Use `pa` to step on every row without tuning a learning rate. Each row asks
the fit to come within `eps·s_y` of its target, `s_y` the target's own
exponentially weighted standard deviation, as `sgd`'s tube is drawn in. The
update is the smallest change that does so.

**`pa` takes `sgd`'s bounds on the slopes.** Here the truth is
`1 + 2·x0 − x1`, and `coef_max=1.5` holds `x0`'s slope below its true
value, so the fit settles on the nearest line the bound allows:

```python
import numpy as np
import polars as pl
import polars_online as po

# y = 1 + 2·x0 − x1 plus noise.
rng = np.random.default_rng(7)
n = 20_000
x0, x1 = rng.standard_normal((2, n))
y = 1.0 + 2.0 * x0 - 1.0 * x1 + 0.3 * rng.standard_normal(n)
pl.DataFrame({"x0": x0, "x1": x1, "y": y}).write_parquet("pa.parquet")

# The same pa twice, the second with its slopes capped.
common = dict(targets=["y"], features=["x0", "x1"], half_life=float("inf"), c=0.1)
specs = [
    po.spec.pa("free", **common),
    po.spec.pa("capped", coef_max=1.5, **common),   # no slope above 1.5, so x0's cannot reach 2
]
bank = po.ModelBank(specs)
bank.fit(pl.scan_parquet("pa.parquet"))
# What each fit settled on.
for name in ("free", "capped"):
    fitted = {r["term"]: round(r["coef"], 2) for r in bank.coef().filter(pl.col("spec") == name).iter_rows(named=True)}
    print(f"{name}: {fitted}")
```

```text
free: {'intercept': 0.99, 'x0': 2.02, 'x1': -0.98}
capped: {'intercept': 0.89, 'x0': 1.48, 'x1': -1.09}
```

With `p` the prediction, and the intercept's 1 inside `z`, a row takes the
step its `mode` sets:

```
loss = max(0, |y − p| − eps·s_y)      s = ‖z‖²
pa    τ = loss / s          pa1  τ = min(c, loss/s)      pa2  τ = loss / (s + 1/(2c))
β    += min(w, 1) · τ · sign(y − p) · z
```

That is the step under `standardize=False`, with `z = [1, x]`. Under
`standardize`, the default, `z` is the row standardized with `sgd`'s scaler,
and past the scaler's warm-up the step is mapped into the caller's units as
`sgd`'s is.

In the target's units, a tube of 0.1 held a target in hundredths on every
row, and the model never learned it. In the residuals' spread, a fit from
zero on a target at 1,000 read a spread near 1,000 from its first
residuals, so `eps = 0.1` drew a tube about 100 wide, and without decay the
fit stopped learning: R² −52. `s_y` reads `y` alone,
so until the target has two weighted rows of different values the tube has
no width. `eps` is 0.01 and `c` is 1 by default.

**`c` is in the target's units.** `pa1`'s cap of 1 caps every step of a
target in thousands and none of a target in thousandths, so the figures
below are for a target of spread about 2. `pa2`'s `c` sits beside `s` and
is free of the target's units: under `"pa"` and `"pa2"` a target scaled by
`k` fits as the unscaled one, scaled by `k`.

**On a level that moves, give a larger `c` or difference the target.** The fit
starts from zero coefficients, and a `pa1` cap of 1 moves the intercept
about one unit a row. On a target falling from 1,000 to −1,000 at 0.5 a
row, at a half-life of 200 and noise 0.3, the intercept met the level near
row 670. With the level as a feature too, the capped steps, mapped into the
caller's units through the scaler, swung the fit far past the level
instead. A drift the intercept carries alone keeps a lag of about 2 at any
`c`, half of it the tube's, since `s_y` holds the drift. Differencing the
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
scaler's warm-up its coefficients are held in the caller's units, as
`sgd`'s are, and an uncapped step still puts the row on its tube's edge. A
row weight below 1 scales the step, and a weight above 1 counts as 1.

**Where outliers are possible, keep a `mode` that caps or damps the
step:**

| `mode` | the step |
|---|---|
| `"pa"` | moves the fit as far as one bad row demands |
| `"pa1"`, the default | is capped at `c` |
| `"pa2"` | is damped by `c` instead of capped |

```python
pa = po.spec.pa(
    "pa", targets=["y"], features=["x0", "x1"], half_life=200.0,
    mode="pa1",                  # the default; "pa" takes no c and refuses one
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

**`l1` keeps the features with evidence and zeroes the rest.** Here a 0/1
target depends on 5 of 50 features. The fit keeps those five, and its
probabilities class the rows of the second half, each before it is learned:

```python
import numpy as np
import polars as pl
import polars_online as po

# A 0/1 target from 50 features, of which only x0 to x4 move its probability.
rng = np.random.default_rng(8)
n, k = 20_000, 50
x = rng.standard_normal((n, k))
logit = x[:, :5] @ np.array([1.5, -1.0, 1.0, 0.8, -0.8])
up = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(float)
names = [f"x{i}" for i in range(k)]
pl.DataFrame(x, schema=names).with_columns(up=up).write_parquet("clicks.parquet")

# One ftrl over all 50 features, each row scored before it is learned.
sparse = po.spec.ftrl(
    "sparse", targets=["up"], features=names, half_life=float("inf"),
    l1=300.0,                      # zero a coefficient whose summed evidence stays under 300
    strict_binary=True,            # refuse a target that is not 0 or 1
)
bank = po.ModelBank([sparse])
out = pl.concat(bank.fit_predict_batches(pl.scan_parquet("clicks.parquet")))
# The features the L1 penalty kept.
betas = bank.coef().filter(pl.col("term") != "intercept")
kept = betas.filter(pl.col("coef") != 0.0)["term"].to_list()
# How often the second half's rows are classed right, out of sample.
p_up = out["sparse"].struct.field("pred_up")      # each row's probability, from the fit before the row
hits = ((p_up > 0.5).cast(pl.Float64) == out["up"])[n // 2:].mean()
print(f"coefficients left nonzero: {len(kept)} of {betas.height}: {kept}")
print(f"rows of the second half classed right, out of sample: {hits:.2f}")
```

```text
coefficients left nonzero: 5 of 50: ['x0', 'x1', 'x2', 'x3', 'x4']
rows of the second half classed right, out of sample: 0.81
```

Under the logistic loss a target outside `[0, 1]` is clamped into it. To
refuse such a target instead, give `strict_binary=True`, which refuses a
chunk with any target but 0 or 1 and names the row.

`b` is the coefficients, `zz`, `n` and `d` are the per-coordinate sums, and
`α` and `β` are its `alpha` and `beta`:

```
b_i  = 0 if |zz_i| ≤ l1 else −(zz_i − sign(zz_i)·l1) / (β/α + d_i + l2)
p    = sigmoid(zᵀb)                  zᵀb itself under loss="squared"
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
as a ridge of `(1 − λ)(β/α + l2)` on the mean scale. Under
`loss="squared"`, a constant target of 5 then settles at 4.65 at `half_life=100`, and at 4.96 at 1000. For a
penalty that weight and density leave alone, use
[`lasso`](#lasso--lasso-path-its-penalty-chosen-as-it-runs), whose penalty
is on the mean scale. The ridge weighs most on a 0/1 target, whose
log-odds the evidence pins only weakly: a probability's curvature is at
most a quarter. On two features at `half_life=200`, with one row a clock
unit, the defaults left the settled logistic fit up to 0.41 from the
decayed maximum likelihood in some coefficient, the worst of 50 seeds, and
`beta=0.1, l2=0` left it 0.17 from it.

**Under a half-life, `ftrl` follows a break at the pace of its steps, not
of its half-life.** Each coordinate steps by its gradient over the root of
its recent squared gradients, so a settled coefficient moves at most about
`2α·√(1 − λ)` a row, with `λ` the decay over one row's clock, however
large the error. A break that moves a coefficient by `D` takes about
`D / (2α·√(1 − λ))` rows to follow. The large gradients just after a
break also raise the sums that divide every step, for about a half-life.
Measured on two features at `half_life=200`, one row a clock unit, after a
change of regime that moved the intercept by 46 (the worst of 50 seeds,
the largest coefficient gap to the decayed least squares):

| `alpha` | settled, before the break | 4 half-lives after | 12 half-lives after |
|---|---|---|---|
| 0.1, the default | 0.17 | 34.4 | 19.9 |
| 0.3 | 0.24 | 18.4 | 0.90 |
| 1.0 | 0.43 | 2.6 | 0.70 |

At the default the intercept closed at 0.011 a row, against
`2α·√(1 − λ)` = 0.012. A larger `alpha` follows a break sooner and settles
noisier: over the 24 half-lives after the noise quadrupled, the default
came at most 0.23 from the decayed least squares, and `alpha=1` 1.17.

**`ftrl` does not centre a feature, so give it features near 0.** A
coefficient's error reaches the prediction multiplied by the feature's
value, so a feature at a level `L` carries it at `L` times its size. On a
stream of three features, one of them held at `L`, the squared loss's error
is about `0.018·L` once the level is large: 18.3 at 1,000 and at -1,000,
and 1.8e6 at 1e8, against 0.75 for the feature held at 0.5. Standardize or
z-score such a feature in the query before the bank, as
[Features in units of their spread](WINDOWS.md#features-in-units-of-their-spread)
shows. `ftrl` has no `standardize` because `l1` zeroes a coefficient in the
feature's own units: a scale inside the model would move which coefficients
`l1` zeroes as the feature's spread moved.

**A row whose target is null or whose weight is 0 leaves that target's fit
where it was,** because the sums wait for the next row that teaches it.
That row is scored with the fit as it stood, then ages the sums by the
whole clock since. The gap ages the evidence and not the penalties, so the
fit gives way to them. A constant 5 settled at `half_life=100` stands at
0.75 of itself after the row that ends a gap of one half-life, and at 0.15
after a gap of five.

**With decay off, `d_i` is `√n_i/α`, which is river's FTRL:** the two
agree to 1e-12, row for row, and Vowpal Wabbit's `--ftrl` agrees to its
single precision.

### `holt` — Holt's linear trend

*API:* [`po.spec.holt`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.holt) — *Rust:* [`holt.rs`](../crates/online-core/src/holt.rs) — *Outputs:* [fields](OUTPUTS.md#holt)

`holt` is the only linear model that takes no features: it extrapolates the
target's own level and trend. To measure what a model's features add, run
`holt` in the same bank. Then compare the two models' `sigma` ([How wide
is its error?](DIAGNOSTICS.md#how-wide-is-its-error)), or let a
[`seqtest`](#seqtest--a-sequential-test-of-a-sign-by-betting) with `a` and
`b` say which predicts closer.

**A feature earns its place when the model that reads it beats `holt`.**
Here the target's level wanders and trends, and `2·x` rides on it. A
regression on `x`, whose intercept follows the level, beats `holt` on the
spread of its error, and `seqtest` says when the evidence for that is
enough:

```python
import math

import numpy as np
import polars as pl
import polars_online as po

# A level that wanders and trends, plus 2·x and noise, one row a clock unit.
rng = np.random.default_rng(9)
n = 2_000
t = np.arange(n, dtype=float)
x = rng.standard_normal(n)
level = 0.02 * t + np.cumsum(0.1 * rng.standard_normal(n))
y = level + 2.0 * x + 0.5 * rng.standard_normal(n)
pl.DataFrame({"t": t, "x": x, "y": y}).write_parquet("trend.parquet")

# A regression on x, holt alone, and a test of which predicts closer.
common = dict(targets=["y"], clock="t", gap_cap=10.0, half_life=20.0, emit_sigma=True)   # sigma_y: the error's spread
specs = [
    po.spec.ewridge("with_x", features=["x"], **common),    # the feature, and an intercept that follows the level
    po.spec.holt("naive", **common),                         # the target's own level and trend alone
    po.spec.seqtest("gain", targets=["y"], a="with_x", b="naive"),   # evidence that with_x predicts closer
]
out = pl.scan_parquet("trend.parquet").online.fit_predict(specs).collect()
# Each model's error spread, then when the evidence for with_x is enough.
for name in ("with_x", "naive"):
    print(f"{name}: sigma_y at the last row {out[name].struct.field('sigma_y')[-1]:.2f}")
passed = out["gain"].struct.field("log_e_a_y") >= math.log(20)   # the 5% level, read at any row
print(f"with_x predicts closer at the 5% level from row {passed.arg_true()[0]}")
```

```text
with_x: sigma_y at the last row 1.14
naive: sigma_y at the last row 2.21
with_x predicts closer at the 5% level from row 15
```

Each row moves the level `l` and the trend `b` by a gain times the forecast
error `e = y − pred`. `s` is the clock since the target was last observed,
this row's step included. So a row with a null target or weight 0 leaves
the level and trend where they were, and its step goes into the next
observation's `s`. A row takes the larger of two gains, on the level and on
the slope:

```
λ_l = 0.5^(s/half_life)                        λ_b = 0.5^(s/trend_half_life)
pred = l + b·s                                 e = y − pred
textbook:  g_t = w/(λ_l·W + w)                 W' = λ_l·W + w
           h_t = w/(λ_b·V + w) · g/s           V' = λ_b·V + w     (s > 0)
line fit:  S0' = S0 + w     ā' = S0·(ā + s)/S0'     Q' = Q + S0·w·(ā + s)²/S0'
           g_f = w·(1/S0' + ā'²/Q')            h_f = w·ā'/Q'
l' = pred + g·e     g = max(g_f, g_t)
b' = b + h·e        h = max(h_f, h_t)
```

**The textbook gains are weighted means'.** `W` and `V` are the weight the
level and the trend have gathered, so a row at weight `w` counts `w` times.
Where they win, the level is the weighted mean of the forecast and the row,
and the trend the weighted mean of `b` and the slope the level moved at.
Once the weight saturates they are the textbook's fixed `1 − λ`.

**The line fit's gains are the weighted least-squares line's,** fitted
through every observation so far on the clock and read at the last. `S0` is
its weight, `ā` the observations' weighted mean age and `Q` the spread of
their ages about it, none decayed. With every observation at one clock
(`Q = 0`), no slope is determined: `g_f = w/S0'` and `h_f = 0`.

**Without decay, `holt` is the weighted line fit,** whatever the weights
and the clock, since an infinite half-life's fixed gain `1 − λ` is 0:
statsmodels' `WLS` on `(1, t)` agrees to 1e-15. Under a half-life the line
fit's gains fall as `1/n` and `1/n²` while the textbook's settle. After a
warm-up of a few half-lives every row takes the textbook's, and from there
the fit is statsmodels' `Holt`: to 1e-12 from row 745 at half-lives of 6
and 25. Gains of `1/n` on both level and trend, without the line fit, pinned
the trend to the first observation. The level then carried that
observation's error times `ln n`: 11 off a flat series after 40,000 rows
without decay, and 6.3 after 10,000 rows at `half_life=10000`, the worst of
50 seeds at a noise of 0.5. The line fit's worst was 0.010 and 0.026, the
weighted least squares' own.

A target's first observation sets `l = y`, predicts nothing, and starts the
trend at 0. A row at the last row's clock is a second observation the level
takes in: the textbook trend holds, since a move over no clock has no
slope, and the line fit's slope moves by its own gain.

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

**The level forgets at the spec's `half_life`, as every model's decay
does.** `level_half_life` is refused, with a message naming `half_life`. To
make the trend the whole history's drift, give `trend_half_life=inf`: the
slope then takes the line fit's gain, and is the weighted least-squares
slope exactly when the level does not decay either (`half_life=inf`).
Beside a finite level half-life the slope learns from errors against a level
that forgets, and lands near the line fit's slope: 0.2% to 1% of it off at a
level half-life of 5 over 3,000 rows of a line in noise.

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

**It follows a correlation that moves, at the pace its half-life sets.**
This code makes two columns whose correlation jumps from 0.2 to 0.8 at row
2,000, and reads `corr_a_b` before the jump and after it:

```python
import numpy as np
import polars as pl
import polars_online as po

# Two columns whose correlation jumps partway through.
rng = np.random.default_rng(1)
n = 4000
rho = np.where(np.arange(n) < 2000, 0.2, 0.8)                 # the truth: 0.2, then 0.8 from row 2000
a, e = rng.standard_normal((2, n))
pl.DataFrame({"a": a, "b": rho * a + np.sqrt(1 - rho**2) * e}).write_parquet("pair.parquet")

# Fit, and read corr_a_b around the jump.
mv = po.spec.ew_cov("mv", features=["a", "b"], half_life=200.0, stats=["corr"])   # no clock: a half-life of 200 rows
out = pl.scan_parquet("pair.parquet").online.fit_predict([mv]).collect()
corr = out["mv"].struct.field("corr_a_b")
for row in (1999, 2200, 2400, 3999):                         # before the jump, one and two half-lives after, the end
    print(f"row {row}: corr {corr[row]:.2f}")
```

```text
row 1999: corr 0.20
row 2200: corr 0.49
row 2400: corr 0.68
row 3999: corr 0.80
```

One half-life after the jump, `corr` has gone half of the way to the new
level, and after two it has gone about three quarters. A shorter half-life
follows faster, and gives a noisier estimate.

The model keeps `m` and `C`, the weighted means of `x` and of
`(x − m)(x − m)ᵀ`. It updates them in Welford's centred form, because the raw
form, `E[x xᵀ] − m mᵀ`, loses a variance when the columns sit far from zero:

```
W' = λW + w      Q' = λ²Q + w²      a = λW / W'      b = w / W'      δ = x − m      (a + b = 1)
m' = m + b·δ     C'ᵢⱼ = a·Cᵢⱼ + a·b·δᵢδⱼ
c = W² / (W² − Q)       (1 under bias=True)
varᵢ = c·Cᵢᵢ     covᵢⱼ = c·Cᵢⱼ     corrᵢⱼ = Cᵢⱼ / √(CᵢᵢCⱼⱼ)
```

**A variance or a covariance is Polars' `ewm_var` under its default,
`bias=False`.** `C` is a weighted mean, which reads low when a few rows
carry most of the weight. `c` corrects it as Polars' `ewm_var` does: `c` is
`n / (n − 1)` at Kish's effective number of rows, `n = W² / Q`, which is
Bessel's correction. At a half-life of 20 rows `c` is 1.018 once the
weights have settled, and at 2 rows 1.207. Give `bias=True` for `C` itself,
Polars' `bias=True` and `ddof=0`. No correlation moves, since `c` cancels
from each. While the moments hold one row of weight, a variance is null,
since one value has no spread to correct; Polars' is null there too.

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
`bank.gram("mv")`. This code uses `df` from [Example
data](../README.md#example-data):

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
allowing for the correlations.** It is `√(δᵀ (c·(C + s·prior·I))⁻¹ δ)`, with
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
regression's `solve_every` does: `pca_every=20` refreshes them every 20
units of `t`, a row each here, and a temporal clock takes a duration such as
`"5m"`. `max_rows_between_pcas` caps the rows between refreshes, and
whichever comes first refreshes. The default is every row, and the rows
between refreshes are scored on the last loadings. With `pca=1`, each row
writes `pc0_var`, `pc0_share` (of the trace), `pc0_loading_<feature>` and
`pc0_score`. Each refresh keeps the previous sign, so a loading never flips.

**`window_size` cuts the history off at a fixed age** ([A hard
window](../README.md#a-hard-window)). For a frame that fits in memory, Polars'
`df.rolling("t", period="3h").agg(...)` with an exponential weight, and `c`
from the weights inside the window, gives the same numbers within 1e-12,
but recomputes every window, in `O(n·W)` time,
where the spec takes `O(n)`. So use the spec for a stream, a state to save, or
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
    stats=["corr", "lag_corr"],   # lag_corr_l<l>_<a>_<b> for each lag and ordered pair
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

**A screen finds the few features that matter among many.** This code makes
500 features, of which five move the target, screens every pair in one pass,
and prints the six with the largest `|t_stat|`:

```python
import numpy as np
import polars as pl
import polars_online as po

# Many features, of which only a few move the target.
rng = np.random.default_rng(2)
n, p = 2000, 500
x = rng.standard_normal((n, p))
y = x[:, [7, 42, 99, 250, 401]] @ np.array([0.5, -0.4, 0.3, 0.3, -0.2]) + rng.standard_normal(n)   # five matter
features = [f"f{j}" for j in range(p)]
pl.DataFrame(x, schema=features).with_columns(y=y).write_parquet("screen.parquet")

# Screen every pair in one pass, then rank by |t_stat|.
screen = po.spec.marginal("screen", targets=["y"], features=features, half_life=float("inf"))   # every row weighs the same
bank = po.ModelBank([screen])
bank.fit(pl.scan_parquet("screen.parquet"))                  # learn, and keep only the state
table = bank.marginal("screen").sort(pl.col("t_stat").abs(), descending=True)
for r in table.head(6).iter_rows(named=True):
    print(f"{r['feature']}: corr {r['corr']:+.2f}, t_stat {r['t_stat']:+.1f}")
```

```text
f7: corr +0.40, t_stat +19.2
f42: corr -0.34, t_stat -16.0
f250: corr +0.23, t_stat +10.7
f99: corr +0.23, t_stat +10.6
f401: corr -0.19, t_stat -8.5
f275: corr -0.07, t_stat -3.1
```

The five planted features lead, in the order of their coefficients, and the
best of the 495 others trails at a `t_stat` of −3.1. With 495 chances, a
value that size is what noise alone gives, which is why `t_stat` ranks pairs
and is no p-value (below).

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
# t_stat                       corr * sqrt((n_kish - 2) / (1 - corr²)): the t-statistic at the Kish sample size
```

For unit weights a step `Δclock` apart, `n_kish` tends to
`(1 + λ)/(1 − λ)`, where `λ` is that step's decay.

**A null target ages its own pairs (`W_t ← λW_t`, `Q_t ← λ²Q_t`) and teaches
them nothing.** A null feature skips the row, as in every model.

**`corr`, `beta` and `t_stat` are null until the target's `W_t` reaches
`min_weight`**, 3 by default here, because two rows give a correlation of ±1
whatever the data. A statistic is null wherever it is undefined: all three
for a constant feature, and `t_stat` at `n_kish ≤ 2`.

**Use `t_stat` to rank pairs, and never as a p-value**, because the rows are
neither independent nor Gaussian. `t_stat` is built on `n_kish`, the right count
for unequal weights, which says nothing about rows that resemble their
neighbours. So on a smooth stream it claims evidence that is not there.
Over 1,000 pairs of *independent* AR(1) series with `φ = 0.9` and `0.8`,
3,000 rows each, `|t_stat|` passed 1.96 on 45% of them.

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
`"geometric"`, `|t_stat_serial|` passed 1.96 on 5.4% of those pairs of AR(1)
series, the 5% the threshold stands for.

| `serial_rule` | the sum over the kept lags | suits |
|---|---|---|
| not given | not taken: `n_serial` and `t_stat_serial` are null | |
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
# t_stat_serial               the same statistic as t_stat, against that count
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

**Give `bin_stat="huber"` when one spike should not draw a shape.** A bin's
mean moves by a row's size times its share of the bin, so one wild row in a
thin bin bends the response curve there, and a screen that reads binned
means of a residual, or of its square, sees a shape that is not in the data.
Under `"huber"` each bin's location is Huber's M-estimate of location,
learned in one pass, as [`huber`](#huber--quantile--robust-regression)
learns a regression past its first rows, and its first rows are refit to
convergence once (below). With `c` the cut, `bin_huber_delta` (default 1.345)
times the target's robust scale `σ_t`, both read before the row, and `e` the
row's distance from the bin's location `m`:

```
ω = w·min(1, c/|e|)      Ω' = λΩ + ω      m' = m + (ω/Ω')·e
```

The location is the mean over the weights `ω`, so its fixed point is the
Huber M-location of the bin's rows, and one row moves it by at most
`c·w/Ω' = c·w/(λΩ + ω)`, which under decay is up to `1/λ` times `c·w/Ω`.
A row far from the bin still counts, at weight `c/|e|` of a row.

| quantity | under `"huber"` |
|---|---|
| `σ_t` | one per target: the exponentially weighted median of `\|y − m_t\|` over Φ⁻¹(3/4) = 0.6745, `m_t` the target's own location by the same recursion. A cut is there for a row far from the target's spread, which is the same row whichever feature bins it, so a scale per feature would cost a quantile sketch per pair and buy nothing |
| a bin's first row | judged against `m_t`, the only location there is: winsorized at `m_t ± c` |
| before `σ_t` exists | the target's first two rows: nothing is clipped, and the numbers are the mean's, as `huber` weighs no row down before its scale exists |
| `bin_mean_y` | the location `m` |
| `bin_n` | the rows' weight, as under the mean: a clipped row still counts as a row |
| `bin_var_y` | the weighted mean square of `e` clipped at `±c`; the variance where nothing is clipped |
| `split_gain` | read from those, so a spike bounds its pull on the curve and on the spread alike |
| `bin_huber_delta=float("inf")` | nothing clipped: the mean, to the bit |

This code puts one row of 1,000 into a bin of twenty rows near 0:

```python
import numpy as np
import polars as pl
import polars_online as po

# Twenty rows near 0 in the middle bin, among rows at -5 and +5, then a spike there.
rng = np.random.default_rng(3)
i = np.arange(420)
x = np.where(i % 21 == 0, 0.0, np.where(i % 2 == 0, -0.9, 0.9))
y = np.where(x == 0.0, 0.0, 5.0 * np.sign(x)) + 0.01 * rng.standard_normal(420)
pl.DataFrame({"x": np.append(x, 0.0), "y": np.append(y, 1000.0)}).write_parquet("thin.parquet")

# The same bins under each statistic; print the middle bin.
for stat in ["mean", "huber"]:
    spec = po.spec.marginal("m", targets=["y"], features=["x"], half_life=float("inf"),
                            bin_edges=[[-0.5, 0.5]], bin_stat=stat)
    bank = po.ModelBank([spec])
    bank.fit(pl.scan_parquet("thin.parquet"))
    r = bank.marginal("m").row(0, named=True)
    print(f"{stat}: middle bin {r['bin_mean_y'][1]:.2f} over {r['bin_n'][1]:.0f} rows")
```

```text
mean: middle bin 47.62 over 21 rows
huber: middle bin 0.51 over 21 rows
```

The mean takes the spike's twenty-first share, 47.6. The location moves by
the cut over twenty, 0.51, and the count still takes the row.

**The first rows are refit once, `bin_refit_rows`.** One pass weighs each
row against the location and the scale before it, and over the first rows
those are still settling. A share of spikes pulls the first locations, the
target's scale learns the pulled deviations, the cut is drawn wide, and the
spikes go on entering at nearly full weight; without decay the bins stay
there, captured, as one pass of `huber` can be. So once `bin_refit_rows`
learned rows are in, each target's location and MAD scale are refit jointly,
as statsmodels' `RLM` fits a constant with `scale_est="mad"`, the scale is
rebuilt from the converged deviations, and each bin's location is solved
exactly at the cut it gives. The stream then goes on in one pass. No weight
moves: `bin_n`, `weight_sum` and every pair moment are the one pass's.

| quantity | the refit |
|---|---|
| default count | `max(500, 30 · bins)`, `bins` the most any feature has; 0 under `"mean"` or an infinite cut |
| learned edges | the warm-up's held rows are the refit's: within `bin_warm_rows` it runs as the edges are fixed and holds no row of its own |
| given edges, or a count past the warm-up | the rows are held until the count: `bin_refit_rows · (8 · features + 16 · targets + 64)` bytes a group |
| `refit_budget` | the holds of all a spec's groups, 256 MiB by default; past it a new group keeps the one pass, with a notice and `summary()`'s `refit_given_up` |
| `bin_refit_rows=0` | the one pass alone |

Against statsmodels' M-location of each bin over the rows so far (16 bins,
spikes of about 20 of the target's scales, no decay), the worst bin of the
worst of 50 streams, in the target's scales:

| spikes | after 2,000 rows: one pass / refit | after 20,000 rows: one pass / refit |
|---|---|---|
| 5% | 0.36 / 0.005 | 0.03 / 0.001 |
| 12% | 1.29 / 0.015 | 0.23 / 0.004 |
| 20% | 3.5 / 0.14 | 1.0 / 0.057 |

A bin needs about 30 rows at the refit: one whose rows are then mostly
spikes has its location among them, rightly for those rows, and the rows
after bring it back only slowly. At 20% spikes, 12.5 rows a bin left 6 of
50 streams half a scale off after 20,000 rows, and 31 left none. These
figures are at the default cut, `bin_huber_delta=1.345`. A wider cut
converges more slowly: at 2.5 with 12% spikes, the worst of 50 streams was
0.30 scales off after 1,000 rows with the refit, against 1.25 without it.
The refit runs once, in about 1 ms at 10 features, 13 ms at 1,000 and 110 ms at
10,000 over 500 rows; a row after it costs what it did.

**The one pass lands on the converged batch estimate.** Against
statsmodels' M-location of each bin (`estimate_location` with
`HuberT(1.345)` at the target's MAD), on 20,000 rows in 16 bins with no
decay, with spikes of 50 on a share of the rows, the largest gap over ten
streams, in units of the target's scale:

| spikes | `"huber"` against the batch estimate | the mean against it |
|---|---|---|
| none | 0.0006 | 0 |
| 1% | 0.0010 | 0.45 |
| 5% | 0.0016 | 1.51 |
| 12% | 0.030 | 2.95 |

The gap is the pass's: the first rows are weighed against a location and a
scale still settling. On 2,000 rows, 125 to a bin, it was 0.013 at 1% and
0.37 at 5%, the largest from a spike on a stream's second row. A spike among a target's first two rows, before it has a scale,
counts in full, and leaves the bins it lands in as a mean's start does.
Later, a bin's first row is winsorized at `m_t ± c`, so it starts at most
`c` off. The location steps over `Ω`, the weight its rows count for,
rather than `W`, the weight they carry. From a spike `E₀` off at full
weight `Ω₀`, while each row is clipped the error decays as
`E(n) = E₀·exp(−c·n/(Ω₀E₀))`, so it is within one scale `σ` after about
`(Ω₀E₀/c)·ln(E₀/σ)` rows. Simulated with no decay, unit weights and `σ`
fixed, a spike of 10, 30 or 100 scales was within one scale after 27, 101
or 381 rows. Over `W` a row moves the location back at most `c·w/W`, so
`n` rows recover at most `c·ln n`, 13 scales in 20,000 rows: in the same
simulation the spike of 30 scales was still 17 scales off after 20,000
rows, and the one of 100 still 87. A bin keeps a fifth more state, 40 bytes a cell in place of
32, and each target up to 9 KB for its scale. Measured at 16 bins, a row
took 0.75 to 0.9 times the mean's throughput at 100 and 10,000 features
and one and ten targets, and the saved state was a fifth to a third larger.

**Give a screen a `window_size` when a relation may flip sign**, because two
regimes of opposite sign average to nothing over a long history. The window
truncates every pair moment, so `corr`, `beta` and `t_stat` describe the
rows inside it ([A hard window](../README.md#a-hard-window)). `weight_sum`,
in the record and in the table, is the weight inside the window. A window
refuses `bins`, so bin the features in a second spec without one.

**To use `lags` under a window, give `window_lags=True`; otherwise the spec is
refused.** Each snapshot then holds the lag moments beside the pair moments,
and beside each lag the lagged rows' own sums: 2.3 times its size at one lag,
and 7.7 times at five with the default cross lags. A windowed lag correlation
is that of the rows inside the window, each product taken about the window's
own means. The moments are kept about each series' first value, which
truncates exactly, and read about the window's means when asked; a
correlation centred at the running mean instead carries the history's mean
into the window, and on a window whose rows had moved to another level read
22. Only the window's first `l` rows differ from a correlation computed on
the window's rows alone: they take the rows before the window as partners.
On 50 AR(1) series, 500 rows a window, the two parted by 0.0068 at most.

**With several targets present on every row, give `feature_moments="shared"`
for speed.** Each feature then keeps one mean and variance, over every
learned row, in place of one per pair over its target's rows. At 20,000
pairs that runs 2.7 times as fast at ten targets, with a state under half
the size and the same table to the bit ([PERFORMANCE
§27](PERFORMANCE.md#27-marginals-shared-feature-moments-2026-09-29)).
Where a target is absent on some rows, `var_x` and the centre of `cov` then
come from every row. That is a different estimator, sound only where the
absence says nothing about the feature, so otherwise keep the default. It
takes `lags`, and refuses a `window_size`.

**Give `shards="auto"` to spread one wide spec over the threads.** The bank
runs groups and specs in parallel ([Parallelism](../README.md#parallelism)),
so a wide `marginal` on one group is otherwise one thread's work.
`shards=10` splits its pairs into ten ranges of features, each run on a
thread of its own, a batch of rows at a time. `"auto"` sizes the split to
the width and the pool, and leaves whole a narrow spec, or a windowed one
that snapshots every row, the default, whose batch is a single row. The
numbers are the same to the bit at any count, so a saved bank resumes under
any count. On 14 threads ([PERFORMANCE
§25](PERFORMANCE.md#25-a-wide-marginal-split-across-the-pool-2026-09-25)):

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

**`rho` moves with the average correlation, and reads below it.** This code
makes ten series whose every pair correlates at 0.2 and then, from row
2,000, at 0.6. It runs `deco` beside an `ew_cov` over the same columns, and
prints `rho` beside the mean of the 45 pairwise correlations:

```python
import numpy as np
import polars as pl
import polars_online as po

# Ten series whose common correlation rises partway through.
rng = np.random.default_rng(3)
n, m = 4000, 10
rho = np.where(np.arange(n) < 2000, 0.2, 0.6)                 # the truth: every pair at 0.2, then 0.6 from row 2000
common, own = rng.standard_normal(n), rng.standard_normal((m, n))
names = [f"s{i}" for i in range(m)]
pl.DataFrame(dict(zip(names, np.sqrt(rho) * common + np.sqrt(1 - rho) * own))).write_parquet("ten.parquet")

# deco beside an ew_cov over the same columns.
eq = po.spec.deco("eq", features=names, half_life=100.0)
pairs = po.spec.ew_cov("pairs", features=names, half_life=100.0, stats=["corr"])
out = pl.scan_parquet("ten.parquet").online.fit_predict([eq, pairs]).collect()
# rho beside the mean pairwise correlation, before and after the rise.
level = out["eq"].struct.field("rho")
mean_corr = out["pairs"].struct.unnest().select(pl.mean_horizontal(pl.selectors.starts_with("corr_"))).to_series()
for row in (1999, 3999):
    print(f"row {row}: truth {rho[row]:.1f}, deco's rho {level[row]:.2f}, the mean of ew_cov's corr {mean_corr[row]:.2f}")
```

```text
row 1999: truth 0.2, deco's rho 0.14, the mean of ew_cov's corr 0.19
row 3999: truth 0.6, deco's rho 0.40, the mean of ew_cov's corr 0.60
```

`deco` takes O(m) a row where the `ew_cov` takes O(m²), and its `rho`
settles near two-thirds of the truth: each row's estimate is a ratio of
two averages, which reads low.

The row is standardised against the means and variances as they stood before
it, `ζ_i = (x_i − m_i)/√v_i`. With `S₁ = Σ ζ_i` and `S₂ = Σ ζ_i²` over the
`n` features that have a standardised value, the row's estimate `u` is Engle
and Kelly's Lemma 2.3. The reported level, `rho`, then follows `u` under one
of two dynamics, with `α` and `β` the `"linear"` dynamic's two weights:

```
u = (S₁² − S₂) / ((n − 1)·S₂)        = (mean of ζ_i·ζ_j over i ≠ j) / (mean of ζ_i²)

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

**The kernel recovers a correlation that noise hides from the plain sum.**
This code makes 20 blocks of two prices whose returns correlate at 0.6, each
tick measured with noise, and prints each estimator's mean correlation over
the blocks:

```python
import numpy as np
import polars as pl
import polars_online as po

# Blocks of two noisy prices whose underlying returns correlate.
rng = np.random.default_rng(4)
blocks, rows = 20, 1000
z = rng.standard_normal((2, blocks * rows))
latent = np.cumsum([z[0], 0.6 * z[0] + 0.8 * z[1]], axis=1)  # two efficient prices whose returns correlate at 0.6
observed = latent + rng.normal(0.0, 0.7, latent.shape)        # each tick measured with noise
pl.DataFrame({
    "block": np.repeat(np.arange(blocks), rows),              # a block id: a trading day, say
    "p1": observed[0], "p2": observed[1],
}).write_parquet("blocks.parquet")

# Each price's returns within its block, fed to both estimators.
returns = pl.scan_parquet("blocks.parquet").with_columns(pl.col("p1", "p2").diff().over("block"))   # within each block
specs = [
    po.spec.rcov(kind, features=["p1", "p2"], group="block", group_close="monotone", kind=kind, block_rows=rows)
    for kind in ("plain", "kernel")
]
bank = po.ModelBank(specs)
bank.fit(returns)
# Each estimator's mean correlation over the blocks that closed.
for kind in ("plain", "kernel"):
    closed = bank.closed_groups(kind)                        # one row per block that closed
    corr = closed["rcorr"].list.get(1)                         # vech of [[1, r], [r, 1]] is [1, r, 1]
    print(f"{kind}: mean correlation {corr.mean():.2f} over {closed.height} blocks (truth 0.60)")
```

```text
plain: mean correlation 0.29 over 19 blocks (truth 0.60)
kernel: mean correlation 0.59 over 19 blocks (truth 0.60)
```

The last block never closes, because no later block arrives to close it
under `group_close="monotone"`, so 19 of the 20 report.

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
the bank, over each block with `.over("block")`,** so that no return spans
two blocks. The bank skips each block's first row, whose return is null. For
a series not observed on every row, carry its last price forward with
`forward_fill().over("block")` before the diff, and use `kind="preavg"`. Or
build a common grid with `po.stream.refresh_time` before the diff ([Series
that tick at their own
times](../README.md#series-that-tick-at-their-own-times)), and keep the
kernel. Each parameter is shown in this code, which uses `df` from [Example
data](../README.md#example-data), cut into blocks of 100 rows:

```python
by_block = df.with_columns(block=pl.int_range(pl.len()) // 100)   # a block id that rises every 100 rows
rk = po.spec.rcov(
    "rk", features=["x0", "x1"],       # x0 and x1 stand in for two prices, differenced in `returns`
    group="block", group_close="monotone",
    kind="kernel",               # the realised kernel; or "plain" or "preavg"
    psd=True,                    # the default: clip any negative eigenvalue
    block_rows=100,              # a sizing hint for the ring: the block's expected length
    bandwidth=None,              # a fixed H; left out, set from the data
)
returns = by_block.with_columns(
    pl.col("x0", "x1").diff().over("block")   # each row's change since the row before it, within its block
)
# Fit, and read the estimate of each block that closed.
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
`rcov_psd_repaired`.** For a correlation read on its own under `"preavg"`,
give `psd=False`. It selects the balanced form, a shorter window of
`k_n = floor(theta * sqrt(block_rows))` rows with the bias term subtracted,
which was the more accurate on [docs/REGIMES.md
§8](REGIMES.md#8-rcov-three-estimators-against-each-blocks-truth)'s streams.
The bias term is `b` times `sum(x x')`, where `b = psi1 / (2 psi2 k_n^2)`
tends to `6 / k_n^2` as the window grows. That sum holds the block's
covariance as well as the noise, so the term removes `b` of the covariance
too. The balanced form divides its estimate by `1 - b` to restore it, as
Christensen, Kinnebrock and Podolskij's footnote 1 does. It is not always
positive semi-definite, so keep the default where the matrix must be.

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

Reach for `audit` before choosing a model, to find out what is wrong with
the data. It learns nothing. Per column it counts the nulls, the NaNs, the
infinities and the values past `1e100`, each kind apart. It also measures
what makes a column hard to learn from: a value repeated far more often than
any other, a run of one value, persistence like a random walk's, and heavy
tails. Per stream it measures the clock's duplicate stamps, gaps and spread
of steps. Each row writes only `weight_sum`, and `bank.audit()` reads the
counts from the state as a table after the run. `bank.check()` reads them
too, and says what they mean.

**An audit finds a frozen feed and a sentinel value.** This code makes a
price that stops moving at row 1,200 and a size column that holds `-999` on
one row in 40. It prints what `bank.audit()` measured, and what
`bank.check()` reads from it:

```python
import numpy as np
import polars as pl
import polars_online as po

# A feed with two faults planted: a frozen price and a sentinel size.
rng = np.random.default_rng(5)
n = 2000
price = 100 + np.cumsum(rng.normal(0.0, 0.1, n))
price[1200:] = price[1199]                                    # the feed froze at row 1200
size = rng.exponential(1.0, n)
size[::40] = -999.0                                           # a sentinel standing for "missing", on one row in 40
pl.DataFrame({"price": price, "size": size, "flow": rng.standard_normal(n)}).write_parquet("feed.parquet")

# Audit the three columns, then print the counts and what they mean.
bank = po.ModelBank([po.spec.audit("audit", columns=["price", "size", "flow"])])
bank.fit(pl.scan_parquet("feed.parquet"))
for r in bank.audit().iter_rows(named=True):                  # what each column holds
    print(f"{r['column']:>5}: longest run of one value {r['longest_run']}, smallest value {r['min']:g}")
for r in bank.check().iter_rows(named=True):                  # what that means, the worst first
    print(f"{r['severity']}: {r['code']} on {r['column']}")
```

```text
price: longest run of one value 801, smallest value 97.3958
 size: longest run of one value 1, smallest value -999
 flow: longest run of one value 1, smallest value -3.22357
warning: frozen on price
warning: random_walk on price
warning: sentinel on size
info: heavy_tails on size
```

Each finding's `message` says what to do, and links its section of
[DATA-QUALITY.md](DATA-QUALITY.md), which shows what the problem does to
each model and how to fix it with Polars.

An audit reads every row, where a model skips a row with a feature that is
not a usable number. Its memory does not grow with the rows: about 14 KiB a
column at the defaults. Its counts are the same in one chunk or a thousand.
Each statistic, its definition and its cost are in the builder's docstring.

| measured per column | read as |
|---|---|
| `null_count`, `nan_count`, `pos_inf_count`, `neg_inf_count`, `beyond_bound_count` | what a model reads as missing, each kind counted apart, of the `rows_fed` |
| `mean`, `std`, `skew`, `kurtosis`, `min`, `max` | the moments of the usable values, in one pass |
| `n_unique`, `top_value`, `top_count` | the distinct values, exact up to `distinct_cap` (default 256), and the most repeated one |
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

**The split–merge move gives a cluster born after seeding a centre of its
own.** This code makes three round clusters, and a fourth that appears at
row 4,000. It fits `n_clusters=4` and counts each label's rows before the fourth
appears and at the end:

```python
import numpy as np
import polars as pl
import polars_online as po

# Three round clusters, and a fourth that appears halfway through.
rng = np.random.default_rng(6)
n = 8000
centres = np.array([[0.0, 0.0], [6.0, 0.0], [0.0, 6.0], [6.0, 6.0]])
blob = rng.integers(0, 3, n)                                  # three clusters ...
late = np.arange(n) >= 4000
blob[late] = rng.integers(0, 4, late.sum())                   # ... and a fourth born at row 4000
xy = centres[blob] + rng.standard_normal((n, 2))
pl.DataFrame({"x": xy[:, 0], "y": xy[:, 1]}).write_parquet("blobs.parquet")

# Label every row, then count each label's rows before the birth and at the end.
km = po.spec.kmeans("km", features=["x", "y"], half_life=1000.0, n_clusters=4, warm_rows=500)
label = pl.scan_parquet("blobs.parquet").online.fit_predict([km]).collect()["km"].struct.field("cluster")
for lo, hi in ((3000, 4000), (7000, 8000)):
    counts = label[lo:hi].value_counts().sort("cluster")
    print(f"rows {lo}-{hi - 1}: " + ", ".join(f"cluster {c}: {k}" for c, k in counts.iter_rows()))
```

```text
rows 3000-3999: cluster 0: 322, cluster 1: 164, cluster 2: 338, cluster 3: 176
rows 7000-7999: cluster 0: 237, cluster 1: 266, cluster 2: 255, cluster 3: 242
```

Before the fourth cluster appears, clusters 1 and 3 share one of the three,
about 170 rows each. Once it appears, the split–merge move frees cluster 3
and places it on the new cluster, and each of the four holds about a quarter
of the rows.

With `c_j` centre `j` and `n_j` its decayed weight:

```
j*   = argmin_j ‖x − c_j‖²          distances in units of each feature's EW sd (standardize=False: raw units)
n'_j = λn_j + w                      c'_j = c_j + (w/n'_j)(x − c_j)     for j = j*
```

This code uses `df` from [Example data](../README.md#example-data):

```python
km = po.spec.kmeans(
    "km", features=["x0", "x1", "x2"], clock="t", half_life=2000.0, gap_cap=300.0,
    n_clusters=3,
    warm_rows=100,               # rows to seed from (default 500)
    init="lloyd",                # the default: the best of ten k-means++ starts
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
index = po.spec.coef_index(km)   # how coef is labelled: cluster = 0, 1, ..., term = the feature
```

**Seeding waits for `warm_rows` rows, places the centres, and replays the
rows.** `init="k-means++"` is the greedy variant scikit-learn uses: each centre
after the first is the best of `2 + ⌊ln k⌋` rows drawn by their squared
distance from the centres so far, the one that leaves the least inertia. On five
blobs in four dimensions one start, run to its fixed point, still lands in a
partition 1.87 times as costly as the right one in 9 of 100 seeds (28 of 100
for plain k-means++). So the default, `init="lloyd"`, takes ten starts, runs
Lloyd's passes from each until no row changes centre, and keeps the one of
least inertia. `seed=` fixes the random draws. A start still moving rows
after 300 passes stops there and is counted in `solve_failures`; the most a
start needed on 500 rows was 54 passes, on eight overlapping blobs. On
Fränti's benchmark sets, in 50 orders each, no order of S1 or S2 misses a
cluster, where ten passes from ten plain k-means++ starts missed one in 10
and 13 of them. The A-sets' 20 to 50 clusters leave 10 to 25 of the 500
buffered rows a cluster, and their median order ends 1.01 to 1.10 times the
optimal inertia, where it ended 1.26 to 1.35 times it.

The seeding is paid once by each group's stream. With the default 500 rows
and 8 centres it took 56 ms at 400 features on separated blobs, where a start
reaches its fixed point in a few passes, and 91 ms on overlapping ones; 1.9
ms at 20 features (one thread, the load average about 110, 2026-10-25).
At 400 features `init="k-means++"`'s greedy draw, four candidates a centre
at 8 centres, places the centres in 2.8 ms, where plain k-means++, one
candidate, places them in 0.8 ms; its seeding row, the placement and the
replay of the 500 rows, takes 3.9 ms (medians over 200 and 40 runs in one
native process, the load average under 15, 2026-10-31). A bank of many groups at many
features pays it per group: 10,000 groups at 400 features spend up to a
quarter of an hour seeding.

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
`n_clusters=2` only the dead rule acts. It cannot see one centre that owns two blobs,
whose rows all sit within its own radius, so keep the default
`init="lloyd"`, which prevents that. `split_merge=0` turns off the move
and the dead rule, which leaves plain sequential k-means.

**Increase `dead_frac` when regimes change faster than a dead centre is
re-placed.** A centre whose blob vanished fades, and is re-placed once it holds
less than `dead_frac` of an equal share: `log2(1/dead_frac)` half-lives after
the blob vanished, 4.3 at 0.05 and 2 at 0.25. In exchange, a cluster lighter
than `dead_frac/k` of the stream loses its centre whenever any row is far.
It stays below 0.5. A re-placed centre starts with half the weight of the
cluster it was split from, so at 0.5 it can start at the floor itself. The
next check then re-places it again: over 50 streams, 3 cascaded at 0.5 and
29 at 0.6, none at 0.45.

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

**It finds two rings that no centre-based rule can split, and flags
scattered rows as noise.** This code makes an inner and an outer ring, with
one row in twenty scattered over the square around them, and reads the last
4,000 rows:

```python
import numpy as np
import polars as pl
import polars_online as po

# Two rings, an inner and an outer, as points in the plane.
rng = np.random.default_rng(7)
n = 10000
ring = rng.integers(0, 2, n)                                  # 0: the inner ring, radius 1; 1: the outer, radius 3
angle = rng.uniform(0.0, 2 * np.pi, n)
radius = np.where(ring == 0, 1.0, 3.0) + rng.normal(0.0, 0.1, n)
xy = np.column_stack([radius * np.cos(angle), radius * np.sin(angle)])
# Scatter some rows as noise, and keep the truth to score against.
noise = rng.random(n) < 0.05                                  # one row in twenty is scattered at random
xy[noise] = rng.uniform(-4.0, 4.0, (noise.sum(), 2))
pl.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "ring": ring, "noise": noise}).write_parquet("rings.parquet")

# Fit, and score only the last rows.
mc = po.spec.micro("mc", features=["x", "y"], half_life=3000.0, eps=0.07)
out = pl.scan_parquet("rings.parquet").online.fit_predict([mc]).collect().unnest("mc")
late = out.tail(4000)                                         # once the summaries have formed
# How many clusters, and how each ring and the noise are labelled.
print(f"clusters: {late['n_clusters'][-1]}")
for name, ring_id in (("inner ring", 0), ("outer ring", 1)):
    rows = late.filter(~pl.col("noise") & (pl.col("ring") == ring_id))
    labels = rows.filter(~pl.col("outlier"))["cluster"].n_unique()
    print(f"{name}: {labels} label, {rows['outlier'].mean():.1%} flagged as noise")
print(f"scattered rows: {late.filter(pl.col('noise'))['outlier'].mean():.1%} flagged as noise")
```

```text
clusters: 2
inner ring: 1 label, 0.4% flagged as noise
outer ring: 1 label, 1.5% flagged as noise
scattered rows: 61.1% flagged as noise
```

A scattered row that lands on a ring, or close enough to join one of its
summaries, takes the ring's label, so not every scattered row is flagged.

Each micro-cluster is a *summary*: a decayed weight `n`, a centre `c`, and a
radius `ν`, the exponentially weighted root-mean-square distance of its rows
from the centre. A summary is *established* once its weight reaches `beta_mu`
rows of the stream's mean row weight `w̄`. With `p` the number of features,
and distances in units of each feature's exponentially weighted standard
deviation, a row joins a summary only if the summary's radius stays within
`eps`:

```
a = n_j/(n_j + w̄)      b = w̄/(n_j + w̄)
n_j  ← λ n_j                                               every summary
j*   = the nearest established summary, if it keeps  a ν²_j + a b ‖x − c_j‖² ≤ eps² p,
       else the nearest other summary, if it does,
       else a new one at x
n_j* ← n_j* + w     c_j* ← c_j* + (w/n_j*)(x − c_j*)     ν²_j* ← min(·, eps² p)
```

The metric's variance is floored as
[`kmeans`](#kmeans--exponentially-weighted-k-means)'s is (`scale_floor`), so
the row on which a quiet feature moves again is not an outlier by that alone.

**Every `prune_every` units of the clock, the light summaries are dropped
and the established ones linked.** A temporal clock takes a duration, and
`max_rows_between_prunes` caps the learned rows between checkpoints, as a
regression's `solve_every` and `max_rows_between_solves` schedule a solve.
With neither, the checkpoint comes every 100 learned rows. Centres within
`L` of each other share a label.
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
first two. In that measurement, at `eps=0.07`, uniform noise over the whole
plane was flagged `outlier` 94% of the time, and real rows 0.3%. The
recipe's scattered rows read 61% because many land near a ring and take its
label. A cluster born mid-stream had a label 31 rows in.

### `ew_class` — Gaussian classification on `ew_cov` moments

*API:* [`po.spec.ew_class`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.ew_class) — *Rust:* [`ewclass.rs`](../crates/online-core/src/ewclass.rs) — *Outputs:* [fields](OUTPUTS.md#ew_class)

Reach for `ew_class` to classify rows by a label column: name the column in
`label` and list its values in `classes`. It keeps one `ew_cov` state per
class, a weight `n_c`, a mean `μ_c` and a centred covariance `C_c`, and
scores a row by Bayes' rule over Gaussian classes. `C_c` is in the
population form, with no `ddof` correction, as `ew_cov`'s is under
`bias=True`.

**Out of sample, it classes about as well as the Bayes rule that knows the
true distributions.** This code makes three Gaussian classes with their own
spreads, and scores the last 5,000 rows, each classed before its own label
is learned:

```python
import numpy as np
import polars as pl
import polars_online as po

# Three Gaussian classes, each with its own spread.
rng = np.random.default_rng(8)
n = 6000
means = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 2.0]])
spread = np.array([[1.0, 0.6], [0.6, 1.0], [0.8, 0.8]])        # each class's own standard deviations
label = rng.integers(0, 3, n)                                 # three classes, equally likely
xy = means[label] + rng.standard_normal((n, 2)) * spread[label]
pl.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "kind": np.array(["a", "b", "c"])[label]}).write_parquet("classes.parquet")

# Class the last rows, each before its own label is learned.
cl = po.spec.ew_class("cl", features=["x", "y"], label="kind", classes=["a", "b", "c"],
                      precision_prior=0.1, half_life=float("inf"))
scored = pl.scan_parquet("classes.parquet").online.fit_predict([cl]).collect().unnest("cl").tail(5000)
print(f"classed right, out of sample: {(scored['class'] == scored['kind']).mean():.3f}")
# The Bayes rule that knows the true distributions, on the same rows.
z = (xy[:, None, :] - means[None]) / spread[None]             # the Bayes rule, from the true means and spreads
best = np.argmax(-0.5 * (z**2).sum(axis=2) - np.log(spread).sum(axis=1), axis=1)
print(f"the Bayes rule, on the same rows: {(best == label)[-5000:].mean():.3f}")
```

```text
classed right, out of sample: 0.857
the Bayes rule, on the same rows: 0.860
```

The test suite holds it to the same on 200,000 rows of three Gaussian
classes in six dimensions, each with its own covariance. There its accuracy
is within 0.01 of the Bayes rate, and each band of its probabilities is
within 0.03 of the share of rows it classes right.

`π_c` is a class's share of the weight, and `ω_c` the ridge `precision_prior`,
which makes a class scoreable from its first row and fades as `ew_cov`'s does:

```
π_c = n_c / Σ n         ω_c = precision_prior · s_c        (s_c: the prior's fade, as in ew_cov)
M_c = C_c + ω_c I  (full)      M = Σ π_c M_c  (shared)      diag(C_c) + ω_c  (diagonal)
ℓ_c = ln π_c − ½ ln det M_c − ½ (x − μ_c)ᵀ M_c⁻¹ (x − μ_c)
p_c = exp(ℓ_c − max ℓ) / Σ exp(ℓ − max ℓ)                  class = argmax ℓ
n_c ← λ n_c + w·[y = c]        μ_c, C_c ← weighted Welford on the row's own class
```

Give `covariance_type`, scikit-learn's words, to set what each class keeps:

| `covariance_type` | what each class keeps | the right choice when |
|---|---|---|
| `"full"`, the default (QDA) | its own covariance, and its Cholesky factor, which only the row's own class refactors (under a `window_size`, every class on every row) | in general |
| `"tied"` (LDA) | one covariance pooled by class weight, factorized once per row | the classes differ in location but not in spread; it then labels more than 98% of rows as `"full"` does in the tests, with fewer parameters to learn |
| `"diag"` (Gaussian naive Bayes) | variances only | speed matters most, since it runs about five times as fast as the other two ([Throughput](../README.md#throughput)); it cannot see a correlation, so two classes with the same marginals and opposite correlations are one class to it |

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
    covariance_type="tied",      # one covariance pooled by class weight
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

**Its evidence passes the 5% level once a sign holds, and stays below it
when none does.** This code makes a column positive 54% of the time and one
positive half the time, and prints the first row at which each test's
evidence reaches `ln 20`:

```python
import math

import numpy as np
import polars as pl
import polars_online as po

# One column with a slight positive edge, and one with none.
rng = np.random.default_rng(9)
n = 3000
pl.DataFrame({
    "edge": rng.normal(0.1, 1.0, n),                           # positive 54% of the time
    "none": rng.normal(0.0, 1.0, n),                           # positive half the time
}).write_parquet("signs.parquet")

# Test both signs, and find the first row each passes the 5% level.
sign = po.spec.seqtest("sign", targets=["edge", "none"])
out = pl.scan_parquet("signs.parquet").online.fit_predict([sign]).collect().unnest("sign")
for col in ("edge", "none"):
    passed = out.with_row_index("row").filter(pl.col(f"log_e_pos_{col}") >= math.log(20))
    when = f"at row {passed['row'][0]}" if passed.height else "never"
    print(f"{col}: the evidence first passes the 5% level {when}")
```

```text
edge: the evidence first passes the 5% level at row 324
none: the evidence first passes the 5% level never
```

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
`a_suffix` or `b_suffix`, such as `"@h500"` or `"__r0.5@h500"` ([Output
field names](../README.md#output-field-names)). Without the suffix, the bank
refuses the spec and lists the grid's residual fields. This code uses `df`
from [Example data](../README.md#example-data):

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
| `"monitor"` | was the correlation constant over a span? | on a span's last row | the span's own correlation | at the span's end, up to `window_rows` rows late |
| `"sequential"` | has it left the level a stable history set? | on each row tested against the history, from the second on | a history of `window_rows` rows | as soon as it crosses a boundary |
| `"window"` | how big is the change? | on every row once both windows are full, `2·window_rows` rows in | the window before | as soon as the two windows differ |

Every kind writes the same five fields, `stat`, `crit`, `flag`,
`since_flag` and `since_change`. A parameter that belongs to another kind
is refused, naming the kinds it applies to.

**The sequential kind flags a correlation break soon after it happens.**
This code makes two columns that are uncorrelated until row 2,200 and
correlate at 0.7 after it, and prints each flag:

```python
import numpy as np
import polars as pl
import polars_online as po

# Two columns that start to correlate partway through.
rng = np.random.default_rng(10)
n = 4000
rho = np.where(np.arange(n) < 2200, 0.0, 0.7)                 # the truth: uncorrelated, then 0.7 from row 2200
a, e = rng.standard_normal((2, n))
pl.DataFrame({"a": a, "b": rho * a + np.sqrt(1 - rho**2) * e}).write_parquet("break.parquet")

# The sequential test, then each row it flags.
watch = po.spec.corrchange(
    "watch", features=["a", "b"],
    kind="sequential",           # each row tested against a stable history as it arrives
    window_rows=500,             # 500 rows of history ...
    monitor_rows=1000,           # ... then up to 1000 rows monitored: a cycle of 1500 rows, the second from row 1500
)
out = pl.scan_parquet("break.parquet").online.fit_predict([watch]).collect().unnest("watch")
for r in out.with_row_index("row").filter(pl.col("flag")).iter_rows(named=True):
    print(f"flag at row {r['row']}: stat {r['stat']:.2f} past crit {r['crit']:.2f}")
```

```text
flag at row 2285: stat 1.84 past crit 1.83
```

The break falls in the second cycle's monitored rows, which start at row
2,000, and the flag comes 85 rows after it. No row of the first cycle, or of
the second cycle before the break, is flagged.

Under `"monitor"` and `"sequential"`, the statistic is the largest over the
pairs, and the level `alpha` (0.05 by default) is spread over the pairs as
`alpha / npairs` unless `alpha_adjust="none"`. For one test of the average
correlation's level, however many columns, give either kind `scalar=True`,
which runs it on the equicorrelation of the standardised row (`deco`'s `u`).

**`kind="monitor"` tests each span of `window_rows` rows for a constant
correlation,** with the closed-sample constancy test of Wied, Krämer and
Dehling (2012). With `m = window_rows`, `ρ̂_j` the correlation of the
span's first `j` rows, and `D̂` the delta-method long-run standard deviation
of `ρ̂`, the last row of a span reports, per pair:

```
Q = max_{2≤j≤m} (j/√m)·|ρ̂_j − ρ̂_m| / D̂
```

Under the null, `Q` converges to `sup|B|` for a Brownian bridge `B`, so the
critical value is the Kolmogorov quantile, the published 1.3581 at 5%. `D̂`
uses the paper's Bartlett kernel, lag `l` at `1 − l/L` with `L = ⌊ln m⌋`.
This code uses `df` from [Example data](../README.md#example-data):

```python
monitor = po.spec.corrchange(
    "break", features=["x0", "x1"],
    kind="monitor",              # the constancy test, span by span
    window_rows=500,             # nothing is reported until a span closes: a delay of at most this many rows
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
the monitoring procedure of Wied and Galeano (2013). A cycle is `window_rows`
rows of history, taken as stable, then up to `monitor_rows` rows, each tested
against the history as it arrives. A flag, or the cycle's last monitored row,
ends the cycle, and the row after it starts a new history. From the history
it reads each pair's correlation `ρ̂_h` and its long-run standard deviation
`D̂`, with the monitor's estimator. The `k`-th monitored row then reports

```
V_k  = (k/√m)·(ρ̂_k − ρ̂_h) / D̂           m = window_rows, ρ̂_k over the k monitored rows
stat = max over pairs of |V_k| / w(k/m)
w(b) = (1 + b)·(b/(1 + b))^γ               γ = boundary_gamma, 0 ≤ γ ≤ 0.49
```

and flags where `stat` passes `crit`.

The critical value is the paper's Eq. 7, `crit = (T/(1+T))^(1/2−γ)·q`, with
`T = monitor_rows/window_rows` and `q` a quantile of `sup_{0<s≤1} |W(s)|/s^γ`
for a Brownian motion `W`. At `γ = 0` and `T = 1`, `crit` is 1.5849 at 5%.
Where the paper simulates `q`, above `γ = 0`, this library solves it as a
diffusion with an absorbing boundary
([`boundary.rs`](../crates/online-core/src/boundary.rs)), within 0.03 of the
paper's Table 1. The paper allows `γ` up to 1/2, but the solve's work grows
as `1/(1/2 − γ)`: seconds at 0.49 and hours just under 1/2. So
`boundary_gamma` above 0.49 is refused.

**Increase `boundary_gamma`, the `γ` of `w(b)`, to catch an early change
sooner.** Above 0 the boundary starts lower, and near the top of the range
it flags more stable streams:

| `boundary_gamma` | a change soon after the history | stable streams flagged, at a nominal 0.05 |
|---|---|---|
| 0, the default | caught latest | 0.04 to 0.09 |
| 0.25 | caught sooner | 0.04 to 0.09 |
| 0.45 | caught soonest | 0.12 to 0.18 |

```python
watch = po.spec.corrchange(
    "watch", features=["x0", "x1"],
    kind="sequential",
    window_rows=500,             # 500 rows of history, assumed stable (kind="monitor" over them checks it)
    monitor_rows=1000,           # then each of up to 1000 rows tested as it arrives: the paper's T = 2
    boundary_gamma=0.25,         # between the size of 0 and the early catch of 0.45
)
```

**`kind="window"` measures how big the change is,** as
`||vech(R_pre - R_post)||` over two adjacent windows of `window_rows` rows
each. `R_pre` and `R_post` are the windows' correlation matrices, and `vech`
stacks their upper triangles into one list. It compares that with a
permutation quantile: `n_perm` shuffles of the pooled rows between the
windows, redrawn every `permute_every_rows` rows. The shuffles move blocks of
`perm_block` rows, so rows that resemble their neighbours do not make the
null too liberal. To skip the permutations, give `crit=` as a fixed
threshold. The same `seed` gives the same critical values.

**The window test's size is about `alpha`.** A report flags when fewer
than `floor(alpha * (n_perm + 1))` shuffles reach its statistic. That counts
the observed split as one more shuffle, so on rows with no change its rank
among `n_perm + 1` shuffles of its own windows is uniform, and that test's
size, the count over `n_perm + 1`, is at most `alpha`: 4.98% at `alpha` 0.05
and the default 200 shuffles. The critical value is drawn from the windows
after a report, though, and held for `permute_every_rows` reports, so each
later report is compared with shuffles of windows a few rows older than its
own, and the size is near that figure rather than held to it. Over 200
streams of iid rows, at `alpha` 0.05 and 200 shuffles, 5.1% of
non-overlapping spans flagged. Fewer than `ceil(1/alpha) - 1` shuffles could never
flag, so such an `n_perm` is refused: 99 at least at `alpha` 0.01.

**Count a run of the window kind's flags as one change.** Two windows that
slide by one row are almost the same windows, so a statistic above the
quantile stays above it for a run of rows, and no rate of `alpha` per row
applies.

```python
size = po.spec.corrchange(
    "size", features=["x0", "x1"],
    kind="window",               # the size of the change between two adjacent windows
    window_rows=100,
    n_perm=200, permute_every_rows=500,   # 200 shuffles, redrawn every 500 rows
    perm_block=10,               # shuffled in blocks of 10 rows
    seed=0,                      # the default
    change_action="flag",        # the default; "reset" empties both windows at a flag
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

**`run_mode` dates a break, while `p_change` may hardly move.** This code
makes a column whose spread quadruples at row 1,000, and reads the row each
row's current run began on, `row - run_mode`:

```python
import numpy as np
import polars as pl
import polars_online as po

# A column whose spread quadruples partway through.
rng = np.random.default_rng(11)
n = 2000
scale = np.where(np.arange(n) < 1000, 1.0, 4.0)               # the truth: the spread quadruples at row 1000
pl.DataFrame({"x": rng.standard_normal(n) * scale}).write_parquet("spread.parquet")

# Fit, and date each row's current run from run_mode.
runs = po.spec.bocpd("runs", features=["x"], hazard=500.0)    # a break expected every 500 rows
out = pl.scan_parquet("spread.parquet").online.fit_predict([runs]).collect().unnest("runs")
out = out.with_row_index("row").with_columns(began=pl.col("row") - pl.col("run_mode"))   # the row the current run began on
# The dating of the break, and how little p_change moved.
found = out.filter((pl.col("row") >= 1000) & (pl.col("began") >= 1000)).row(0, named=True)
print(f"first row whose run began after the break: {found['row']}, dated to row {found['began']}")
for row in (1050, 1999):
    print(f"row {row}: the regime began at row {out['began'][row]}")
print(f"largest p_change on rows 1000 to 1049: {out['p_change'][1000:1050].max():.2f}")
```

```text
first row whose run began after the break: 1009, dated to row 1001
row 1050: the regime began at row 1001
row 1999: the regime began at row 1001
largest p_change on rows 1000 to 1049: 0.02
```

Nine rows after the break the posterior has accepted it, and dates it within
a row of the truth. No single row of the wider regime is surprising enough
to lift `p_change`, which is why the answer is read from `run_mode` (below).

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
    degrees_of_freedom_prior=2.0,  # the prior on the variance, as 2a ...
    covariance_prior=[2e-4],     # ... and 2b; left out, both come from the first rows
    covariance_type="diag",      # the default: a normal-inverse-gamma per feature
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

**Leave `mean_prior` and `covariance_prior` out, and the first rows set
them.** The four priors take the names scikit-learn's
`BayesianGaussianMixture` gives them. The first `warm_rows` learned rows,
`p + 2` by default with `p` the number of features, report null. Their
mean and covariance (the diagonal under `covariance_type="diag"`) become the
prior, and the rows are then read as rows of a model that had it from the
start. `degrees_of_freedom_prior` defaults to the smallest value that gives
the prior's variance a mean, `p + 2` under `"full"` and 3 under `"diag"`, and
at it the prior's mean variance is that covariance.

**A prior you give must be in the data's units, because it decides what
counts as surprising.** Too wide, and no row is surprising. A column of
spread 1e-4 whose spread grew tenfold at row 500 was never found to break
under a prior of unit precision (`mean_prior=[0.0]`,
`covariance_prior=[2.0]`, `degrees_of_freedom_prior=2.0`). The prior set from
its first rows dated the break to row 500. Too narrow or off-centre, and a
run that begins on a row has no density for it. `degrees_of_freedom_prior`
and `covariance_prior` are 2a and 2b in the gamma parametrisation. So the
values in the code above, `degrees_of_freedom_prior=2.0` and
`covariance_prior=[2e-4]` with `hazard=250.0`, are Adams and MacKay's own
finance example (a = 1, b = 1e-4, hazard = 250), a precision of 1e4 on
average.

**Read the regime's age from `run_mode`, and use `p_change` only as an
alarm.** `p_change` is a per-row likelihood ratio, so it is spiky, and its
height depends on the size of the break against the prior scale. A break
one row makes plain lifts it on that row. A break that takes several rows
to show, as the recipe's fourfold spread does, lifts it hardly at all,
since by the time the evidence is in, the new run is longer than the one
row `p_change` reads: the recipe's peak is 0.02. On three kinds of break,
measured on streams of their own:

| the break | `p_change` | the run length |
|---|---|---|
| the spread steps tenfold, under a prior of the rows' scale (`degrees_of_freedom_prior=2`, `covariance_prior=2e-4`, Adams and MacKay's) and `hazard=250` | 0.83 on the row of the break | finds it one to three rows later, and dates it to the right row |
| a four-sigma mean shift, with a diffuse prior | barely lifts | finds it one to three rows later, and dates it to the right row |
| a change in correlation alone | never moves at all | reaches it only under `covariance_type="full"`, a median of 121 rows later ([docs/REGIMES.md §5](REGIMES.md#5-two-changepoint-detectors-on-the-same-break)) |

**`p_change` reports `P(r ≤ 1)`, because `P(r = 0)` equals `H` on every
row:** the two branches share one predictive, so the normalised mass at
`r = 0` is *exactly* `H`, whatever the data. Under a duration `hazard` that
does not hold. The chance of a break comes before the row and nothing sits
at `r = 0` after it, so `p_change` is the chance that the row began a run. Row
one of a group reports nothing under the default `min_weight` of 1, since
`P(r ≤ 1)` is 1 there however the row looks.

Give `covariance_type`, scikit-learn's words, to choose each run's model
of the rows, and `robust_beta` to temper it:

| setting | each run's model | note |
|---|---|---|
| `covariance_type="diag"`, as in the example | a normal-inverse-gamma per feature | |
| `covariance_type="full"` | a normal-inverse-Wishart over all of them, O(runs p³) a row, since each run's p × p scale is factored every row | the one that can see a break in the *correlation* alone |
| `robust_beta=0.1`, under `"diag"` | each row's contribution weighted by `(pi(x)/pi(mode))**robust_beta` | one 20-sigma row moves nothing; untempered, that row is a changepoint at `p_change` 0.91. Not free of the data's units |

**Keep `robust_beta` below about 0.2.** Above it, the weighting forgives the
rows of a new regime one at a time, so no change is ever detected again.
At 0.1 it ignores a single 20-sigma row and still finds a four-sigma shift
within three rows, dated to the right one. Left out, nothing is tempered.

**Centre and scale the features for a tempered run.** A change of units
multiplies every run's predictive density by one constant. Untempered, under
`"diag"` and `"full"`, the posterior's normalisation takes it out.
`robust_beta` tempers each run's message by a power that differs from run to
run, so the constant does not cancel, and the runs' odds move with the
units. At a level of 1e4 with a spread of 1e-2, `p_change` moved by up to
0.86 against the same stream at level 0 and spread 1.

Two settings bound the runs kept, and a third sets the hazard row by row:

| setting | what it does |
|---|---|
| `prune_below` | drops the runs holding less than this share of the mass |
| `max_run` | 10,000 by default: folds the runs that reach `max_run - 1` rows into the one carrying the most mass (the oldest among equals), whose weight and scatter are scaled back to `max_run - 1` rows. The capped run is then exponentially weighted, scaled by `(max_run - 1) / max_run` each row: its mean steps by `1 / max_run`, and its half-life is `ln 2 / ln(max_run / (max_run - 1))`, about `0.69 (max_run - 1)` rows. In a stream that seldom breaks, this is what bounds the runs kept |
| `hazard_col` | reads a per-row hazard from a column instead: its value at a row sets the chance of a break after that row. A null falls back to `hazard`, a value of 1 or less is an error naming the row, and a duration `hazard` beside it is refused |

### `hmm` — which regime are we in

*API:* [`po.spec.hmm`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.hmm) — *Rust:* [`hmm.rs`](../crates/online-core/src/hmm.rs) — *Outputs:* [fields](OUTPUTS.md#hmm)

Reach for `hmm` to ask "which regime are we in", not only "which regime does
this row look like".
[`ew_class`](#ew_class--gaussian-classification-on-ew_cov-moments)
classifies a row against *labelled* Gaussians. An `hmm` does the same
arithmetic without labels: the regime is a *hidden state*, and a transition
matrix carries information from one row to the next.

**Because regimes last, the filter beats a rule that knows the true
centres.** This code makes two overlapping blobs, 1.5 apart in each
coordinate, and a sticky chain that switches between them about once in 200
rows. It compares the filter's state with the nearest true centre, row by
row:

```python
import numpy as np
import polars as pl
import polars_online as po

# A sticky two-state chain, and two overlapping blobs it switches between.
rng = np.random.default_rng(12)
n = 6000
state = np.zeros(n, dtype=int)
for i in range(1, n):                                         # a sticky chain: each row has a 0.5% chance of a switch
    state[i] = 1 - state[i - 1] if rng.random() < 0.005 else state[i - 1]
centres = np.array([[-0.75, -0.75], [0.75, 0.75]])            # two blobs of unit spread
xy = centres[state] + rng.standard_normal((n, 2))
pl.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "truth": state}).write_parquet("states.parquet")

# Filter the states, then score the last rows against the nearest true centre.
hidden = po.spec.hmm("hidden", features=["x", "y"], n_components=2, precision_prior=1e-2,
                     warm_rows=500, half_life=float("inf"))
rows = pl.scan_parquet("states.parquet").online.fit_predict([hidden]).collect().unnest("hidden").tail(5000)
right = (rows["state"] == rows["truth"]).mean()               # the states are numbered as seeded, so read either way
nearest = ((rows["x"] + rows["y"] > 0).cast(pl.Int64) == rows["truth"]).mean()   # the nearest true centre
print(f"rows in their true state: the filter {max(right, 1 - right):.2f}, the nearest true centre {nearest:.2f}")
```

```text
rows in their true state: the filter 0.92, the nearest true centre 0.85
```

The nearest centre reads each row alone. The filter carries what the rows
before told it through the transition matrix it learns, so a row that lands
nearer the other blob does not flip its state.

Here a hidden state means one of the `k` regimes, not the bank's saved state.
The model runs Hamilton's filter one row at a time, from the filtered `p` the
previous row left, and every output is read before the row is learned from:

```
p1_l   = Σ_k p_k·Π_kl                      the predicted state
f_l    = N(x | μ_l, Σ_l + ω_l·I)           the state's density, ω_l its fading prior ridge
loglik = ln Σ_l p1_l·f_l                   the row's surprise
p_l   ← p1_l·f_l / Σ                       the filtered state
```

This code uses `df` from [Example data](../README.md#example-data):

```python
hidden = po.spec.hmm(
    "regime", features=["x0", "x1"], half_life=500.0,
    n_components=2,              # the number of hidden states
    precision_prior=1e-2,        # required: a state's centred co-moments start at zero, and a zero matrix has no density
    covariance_type="full",      # the default; "tied" or "diag", as for ew_class
    warm_rows=400,               # seed the states from this many learned rows; every output is null until then
    means=None, covs=None,       # or give the states outright, and learn=False to freeze them
    transition_prior=1.0,        # the default Dirichlet pseudo-count per cell of the transition matrix
)
out = df.online.fit_predict([hidden]).unnest("regime")
# filtered_state0, filtered_state1    the filtered state, before the row
# predicted_state0, predicted_state1  the predicted state
# state                               the most probable predicted state
# loglik                              the row's surprise
```

**Seeding decides which regimes the filter can find.** `warm_rows` seeds the
states from that many learned rows with `kmeans`' rule (`init="lloyd"`,
`seed=0`), by default 1,000 rows. A group with fewer rows than `warm_rows`
never seeds, and reports null on every row. The seeds sit where the buffered
rows are, so the buffer must hold every regime, several times over. A regime absent
from it gets no state, and one with a few percent of its rows costs less
inertia to share than to own, so the seeds split a larger regime in two
instead. Learning does not move a misplaced state out. On three regimes
three noise standard deviations apart, each lasting 50 rows on average, the
worst of 50 seeds of 5,000 rows put 0.95 of the rows in their true state
from 1,000 buffered rows, 0.81 from 500 and 0.46 from 50. The rows the
buffer holds report nothing, and it holds `warm_rows` rows of every feature
until the seeding frees it.

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

**The transition matrix is learned from the filtered joint of consecutive
states,** `ξ_kl = p_k(t−1)·Π_kl·f_l / Σ`, and a Dirichlet pseudo-count keeps
the matrix row of a state never visited a distribution.

**When one row says little about its regime, give the matrix a sticky
prior:** `transition=` a matrix whose diagonal is near 1, and
`transition_prior=` the number of rows that prior counts as. The README's
[Detecting changes and regimes](../README.md#detecting-changes-and-regimes)
runs two series whose returns widen 2.7 times in every other block of 500
rows. Under the default prior the filter put 58% of the rows after seeding
in their true regime. Under `transition=[0.998, 0.002, 0.002, 0.998]` and
`transition_prior=500.0` it put 93% there.

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
