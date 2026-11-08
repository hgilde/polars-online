<!-- kind: reference -->
# Models

The reference entry of each of the twenty-two models a spec can name, by
family, for a reader who knows which model they want to look up. A
model's parameters, their defaults and its refusals are in its docstring,
and the fields it writes in [OUTPUTS.md](OUTPUTS.md). The README's
[Models](../README.md#models) has the table of all twenty-two, how each
learns from a row, and which to choose for which question.

**The update rules share one notation.** A symbol in this table means the
same thing in every entry its *used by* column names, and each entry defines its other
symbols beside its rule:

| symbol | what it is | used by |
|---|---|---|
| `z` | `[1, x]` when there is an intercept, and `x` when there is not | the linear models |
| `w`, `W` | the row's weight, and the weight accumulated before the row | every model |
| `λ` | the row's decay, `0.5 ** (Δclock / half_life)`, or `lam ** Δclock` when the spec gives `lam=`, with `Δclock` the clock's step since the last row the model did not skip, capped at `gap_cap` (or `session_gap` at a session change) ([Time and decay](explanation/time-and-decay.md)) | every model |
| `S_j`, `η_j` | target `j`'s running means of `z zᵀ` and of `z·y_j`, which the least-squares fits solve from | `ewridge`, `huber`, `quantile` |
| `e` | the row's residual: `y` less the prediction made before the row | `huber`, `quantile` |
| `ω` | a ridge added to a covariance: `precision_prior`, fading as `ew_cov`'s prior does, so that a class or a state can be scored from its first row | `ew_class`, `hmm` |

Without a clock column, each row is one unit of clock. A model keeps its
running sums as means, decayed sums or centered second moments, and [The
running sums that
decay](explanation/time-and-decay.md#the-running-sums-that-decay) says
which models keep each, and why.

## Linear models

Ten models predict a numeric target. Nine regress it on features, and
`holt`, a baseline, uses none. Several of them take the same parameters:

| parameters | taken by | explained under |
|---|---|---|
| `ridge` | `ewridge`, where a list fits a grid; `huber` and `quantile`, as one number. `rls` refuses it and takes `delta` | [`ewridge`](#ewridge--exponentially-weighted-ridge-regression) and [`rls`](#rls--recursive-least-squares) |
| `solve_every`, `max_rows_between_solves` | `ewridge`, `lasso`, `huber`, `quantile` | [`ewridge`](#ewridge--exponentially-weighted-ridge-regression) |
| `target_gaps` | `ewridge`, `lasso` | [`ewridge`'s docstring](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.ewridge) |
| `window_size` | `ewridge`, `lasso` | [A hard window](../README.md#a-hard-window) |
| `standardize` | `ewridge`, `huber` and `quantile`, off by default; `kalman`, `sgd` and `pa`, on by default. `lasso` always standardizes, and `ftrl` never does | [`ewridge`](#ewridge--exponentially-weighted-ridge-regression), [`kalman`](#kalman--random-walk-β-dynamic-linear-model) and [`ftrl`](#ftrl--online-logistic-regression) |
| `coef_min`, `coef_max`, `coef_sum` | `sgd`, `pa` | [`sgd`](#sgd--stochastic-gradient-descent) |
| `weight` | every model, though `rls`, `kalman`, `sgd`, `pa` and `ftrl` read it differently from a weighted mean | [Weights](../README.md#weights) |

What `half_life` decays differs by model:

| model | `half_life` decays |
|---|---|
| `ewridge` | the running sums the fit is solved from |
| `lasso` | the same sums, and the error that picks the penalty unless `select_half_life` is given |
| `rls` | `A` and `b`, which it keeps as decayed sums |
| `kalman` | the observation-noise estimate and the features' standardization; `coef_half_life` sets how fast a coefficient may drift |
| `huber`, `quantile` | their running sums, kept per target, and the residuals' spread `σ` (for `huber`, at `huber_scale_half_life` when given) |
| `sgd` | `weight_sum`, the running moments `standardize` scales by, the spreads its cut and tube are drawn in (the Huber cut's at `huber_scale_half_life` when given), and adagrad's sums, but not the coefficients |
| `pa` | `weight_sum`, each target's own weight, the running moments `standardize` scales by and the spread its tube is drawn in, but not the coefficients |
| `ftrl` | its per-coordinate sums, but not its penalties, so a shorter half-life shrinks the fit more |
| `holt` | the level, and the trend at `trend_half_life`, four times `half_life` by default |

### `ewridge` — exponentially weighted ridge regression

*API:* [`po.spec.ewridge`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.ewridge) — *Rust:* [`ewridge.rs`](../crates/online-core/src/ewridge.rs) — *Outputs:* [fields](OUTPUTS.md#ewridge)

Exponentially weighted ridge regression on running sums.

Start with `ewridge` for a numeric target, where its
[Gram](GLOSSARY.md#fitting-a-model), the sums it keeps of its columns'
means and co-moments, fits a grid of ridge values and feature sets for one
extra solve each. Change to another linear model when your data need what
it adds:
[`kalman`](#kalman--random-walk-β-dynamic-linear-model) for coefficients
that drift, [`huber`](#huber--robust-regression) for outliers or
[`quantile`](#quantile--robust-regression) for a quantile, and
[`lasso`](#lasso--lasso-path-its-penalty-chosen-as-it-runs) for a sparse
fit. Use [`sgd`](#sgd--stochastic-gradient-descent) or
[`pa`](#pa--passive-aggressive-regression) where learning a row must take O(k) time for
`k` features: this model takes O(k²) time.

Its parameters, their defaults and what each refuses are in its
[docstring](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.ewridge),
and the fields it writes in [OUTPUTS.md](OUTPUTS.md#ewridge). The
parameters every spec takes are in [What a spec
names](../README.md#what-a-spec-names).

#### The update

Under the default `target_gaps="own_rows"`, `S_j` and `η_j` are taken over
the rows where `y_j` is present, and `W_j` is the
weight behind them. Under `"pairwise"`, one `S`, weighted by `W`, is taken
over every row, while `η_j`, `W_j` and the feature means the intercept is
centered on stay over `y_j`'s rows, and the slopes solve the centered
system the docstring's Notes give under *Target gaps*. `W` is the
`weight_sum` a row reports. The equations are the default's, and the
coefficients are solved on a schedule:

```
W'   = λW + w                       weight_sum, over every row
W_j' = λW_j + w                     on the rows where y_j is present, λW_j on the others
S_j' = (λW_j·S_j + w·z zᵀ) / W_j'   η_j' = (λW_j·η_j + w·z·y_j) / W_j'
solve, ridge_scale="mean":  (S_j + ridge·D) β_j = η_j + ridge·D·β₀      D = I with a 0 at the intercept
solve, ridge_scale="sum":   (W_j·S_j + p·ridge·I) β_j = W_j·η_j + p·ridge·β₀
                            p the product of every decay so far, β₀ = coef_prior (0 by default)
```

#### Example

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

#### Choosing its settings

The docstring defines every setting the advice names.

```python
rr = po.spec.ewridge(
    "rr", targets=["y"], features=["x0", "x1", "x2"], clock="t", gap_cap=300.0, half_life=600.0,
    ridge=[1e-6, 0.1],             # a grid of two ridge values, solved from one set of sums
    feature_sets={"mkt": ["x0"], "all": ["x0", "x1", "x2"]},   # named feature subsets, likewise
    solve_every=12.0,              # solve every 12 clock units ...
    max_rows_between_solves=100,   # ... and at least every 100 rows, whatever the clock does
    standardize=True,              # solve on the correlation matrix, then unscale the coefficients
    ridge_scale="mean",            # the default ("sum", a warm start that fades, is refused beside standardize, a ridge or feature-set grid, or window_size)
    coef_prior=None,               # the default: shrink toward zero, not toward a stated belief
    target_gaps="own_rows",        # the default: a target null on some rows is fitted on its own rows
)
```

**A grid of `ridge` values or of `feature_sets` adds a solve per grid point
(each ridge value with each feature set: four in `rr`), and the model still
learns each row once.** Only the named sets are fitted, so
to keep a fit on every feature, name a set that holds them all, such as
`"all"`.

**To keep the coefficients current at every row, give
`max_rows_between_solves=1`.** Otherwise `ewridge` predicts with the
coefficients of its last solve. Under a finite half-life it solves once the
weight learned since then reaches `ln 2 / 50` of the weight the fit holds:
about 1.4%, or every `half_life/50` of clock in steady state.

**While a group's weight is more than about 1.4% below its floor, the
model runs no solve that nothing would read.** The floor is the least of a
row per unknown (`k + 1` for `k` features, `k` without an intercept) and
`min_weight`. Below it every prediction is withheld, so under the schedule
by weight, and not under a hard window, a solve that comes due there is
not run, unless `max_rows_between_solves` also brings it due. A row cap's
solves always run. `bank.solve_failures()`, which counts the solves that
needed a jitter (a small value added to the diagonal so that a
near-singular system factorizes) or were withheld as ill-conditioned,
counts only the solves run. So a group whose
weight never reaches the floor, such as one of two rows over two features,
reports no `coef` on any row, in `bank.coef()` or as a closed group. To
keep every solve, give `min_weight=0`, which makes the floor 0, or
`max_rows_between_solves=1`, whose solves always run. The docstring's
Notes, under *The solve schedule*, give the exact margin and its two rare
cases.

**To solve on the clock instead, give `solve_every` in clock units.** Until
`solve_every` has passed since the group's first row or its last reset, or
`max_rows_between_solves` rows with weight have been learned, the model
solves by weight in a first period. So a fit of its first few rows is not
held for a whole period. With `solve_every=500` and `min_weight=50`, the mean squared error
of the predictions over the first 600 rows was 3.1e-6, where holding the
first fit for the whole period gave 5.6e-4, on five seeds of a
five-feature stream at a half-life of 1,000.

**To solve only for each target's first fit, and then only where a row
cap says, give `solve_every=float("inf")`.** Neither it nor any value from
`1e18` on comes due, since `1e18` lies past the end of any stream. Neither
has a first period. The model keeps its sums, and solves only to give each
target its first fit and where a row cap says.

**Under `half_life=float("inf")` or `lam=`, give a `solve_every` when the
fit need not be current at every row,** since both solve on every row by
default. On 6M rows × 20 features from a parquet stream, `solve_every=1000`
took 1.4 s instead of 10.7 s, with the coefficients at most 1000 rows out
of date.

**To start from a belief that fades as data arrives, such as yesterday's
fit, give it as `coef_prior` with `ridge_scale="sum"`.** The ridge then
sits on the decaying sum scale, as `rls`'s `delta` does, and pulls every
coefficient toward `coef_prior`, the intercept included, until the evidence
takes over. `S` is a mean, so under the default `"mean"` the ridge is a
penalty per observation that never fades: use it to hold the slopes near
`coef_prior`, or near zero, for good.

**To start each session from a blend with the long run, give
`session_shrink=f` and `long_half_life=`, beside a `session` column and a
`session_gap` that is a number:**

```python
blend = po.spec.ewridge(
    "blend", targets=["y"], features=["x0", "x1"], clock="t", gap_cap=300.0, half_life=100.0,
    session="session",       # the column whose change starts a session ...
    session_gap=60.0,        # ... and the clock step applied there; "reset" is refused beside a blend
    long_half_life=5000.0,   # a twin of the sums, forgetting at this longer half-life
    session_shrink=0.3,      # at each session change, the moments become 0.7 of this session's and 0.3 of the twin's
)
```

The blended moments keep this session's weight, Kish's effective sample
size and prior scale (`p` in the update), so `weight_sum`, the warm-up
gates and the solve schedule do not move. The blend changes the moments and
nothing else. The coefficients stay as last solved until the next solve the
schedule brings due, which reads the blended moments. So a small `f` moves the output by a small amount,
and `0` is the same as no `session_shrink`. With `max_rows_between_solves=1`
the solve on the session's first row reads the blend, and the next row is
predicted from it.

**For a fit over hundreds of features, give `gram_block_rows=256`.**
Learning a row takes O(k²) time in `S` for `k` features, and at a thousand
features that takes most of the run. The option holds 256 rows back
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
when the work the bank runs at once, one piece per spec and group, leaves
at least half the bank's threads idle. The threads are the bank's own, and
the output is the same to the bit at every thread count. At 2,000 features a bank
learned 3.9 times as fast on eight threads
([docs/PERFORMANCE.md](PERFORMANCE.md) §37). A bank whose groups
already keep every core busy gains nothing from a split, so `"auto"` does
not split there. `gram_threads=8` asks for eight whatever the groups.

#### See also

When `ewridge`'s fit is not the one you need, or to use its sums outside a
bank:

- [`rls`](#rls--recursive-least-squares), the same fit by recursive least
  squares, which loses half as many digits on a nearly collinear design.
- [`huber`](#huber--robust-regression), which keeps sums of the
  same form, each row down-weighted by its residual.
- [GRAM.md](GRAM.md), to read the sums back, pool them and solve them
  offline.

#### Notes

**With decay off and `ridge=0`, `ewridge` fits ordinary least squares over
every row seen, in any row order.** Its coefficients match
`numpy.linalg.lstsq` to 2e-13, fed forwards or backwards. On 6M rows × 20
features from a parquet stream, memory peaked at 1.4 GB, against 3.97 GB
for `lstsq` on the same rows.

**NIST's statistical reference datasets show where the solve stops
promising a digit.** Their hardest designs that double precision can solve
by the normal equations have correlation matrices conditioned at `2.7e6`,
far inside the limit at which `ewridge` withholds a fit. `Filip`, a
degree-10 polynomial at `1.5e16`, is past it, and is withheld. `rls`,
which rotates the design rather than squaring it, keeps 7 of `Filip`'s
digits. The docstring's Notes give the limit, and how each solve is
checked against it.

### `rls` — recursive least squares

*API:* [`po.spec.rls`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.rls) — *Rust:* [`rls.rs`](../crates/online-core/src/rls.rs) — *Outputs:* [fields](OUTPUTS.md#rls)

Recursive least squares, which solves the ridge fit exactly on every row.

Use `rls` when the coefficients must be current at every row: it learns
each row in O(k²) time for `k` features, by rotating the row into a kept
factor, where `ewridge` solves a `k × k` system on a schedule. Its prior
fades as rows arrive. Use
[`ewridge`](#ewridge--exponentially-weighted-ridge-regression) for a long
stream whose features can hold one value for a long time, which winds this
fit up, and [`huber`](#huber--robust-regression) with
`ridge_scale="sum"` for this fit under a Huber loss, solved on a schedule.

Its parameters, their defaults and what each refuses are in its
[docstring](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.rls),
and the fields it writes in [OUTPUTS.md](OUTPUTS.md#rls). The parameters
every spec takes are in [What a spec names](../README.md#what-a-spec-names).

#### The update

`A` and `b_j` are decayed sums of `z zᵀ` and `y_j·z`, not means. They start
from `A₀ = delta·I`, the classic RLS prior `P₀ = I/delta`.
Because `A` is a sum, a row's weight is on the sum scale, and a heavier
stream outweighs the starting prior sooner
([Weights](../README.md#weights)). The model keeps the Cholesky factor of
`A` and folds each row into it by Givens rotations (the square-root form):

```
A'   = λA + w·z zᵀ        b_j' = λb_j + w·y_j·z        β_j = A⁻¹ b_j
A₀   = delta·I            b_j₀ = delta·β₀_j             β₀_j = coef_prior (0 by default)
```

#### Example

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
print("planted at the last row: [0.50, 2.00, -1.00]")
print("rls's coefficients at the last row:", [round(c, 2) for c in out["rls"].struct.field("coef").drop_nulls()[-1]])
```

```text
the two predictions differ by less than 1e-9 on every row: True
planted at the last row: [0.50, 2.00, -1.00]
rls's coefficients at the last row: [0.52, 1.92, -0.99]
```

#### Choosing its settings

**The targets share one Cholesky factor of `A`, so a row with any target
null is predicted for every target and learned for none.** When the
targets go missing on different rows, give each its own `rls` spec.
Otherwise one spec holds them all.

#### See also

When `rls`'s fit is not the one you need:

- [`ewridge`](#ewridge--exponentially-weighted-ridge-regression), the same
  fit under `ridge_scale="sum"`, solved on a schedule, whose default
  `ridge` does not wind up. The docstring's Notes, under *Wind-up*, measure
  how a feature that holds one value winds `rls` up.
- [`huber`](#huber--robust-regression), with
  `ridge_scale="sum"`, this fit with each row down-weighted by its
  residual.
- [`kalman`](#kalman--random-walk-β-dynamic-linear-model), for
  coefficients that drift as a random walk.

#### Notes

**`rls` matches `ewridge(ridge_scale="sum")` solved on every row, to better
than 1e-9,** as the Example shows. At unit weights on a row clock it is
padasip's `FilterRLS`, which keeps the textbook inverse `P`, and agrees
with it to 1.6e-14 relative. On NIST's `Filip`, a degree-10 polynomial
whose correlation matrix is conditioned at `1.5e16`, `rls` keeps 7 digits,
where `ewridge` withholds the fit. It folds each row into a triangular
factor by rotations, so it never forms `zᵀz` and keeps the digits that
squaring the design loses.

### `lasso` — lasso path, its penalty chosen as it runs

*API:* [`po.spec.lasso`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.lasso) — *Rust:* [`lasso.rs`](../crates/online-core/src/lasso.rs) — *Outputs:* [fields](OUTPUTS.md#lasso)

A lasso or elastic-net path on running sums, with the penalty chosen as
the stream runs.

Use `lasso` for a sparse fit, where some features should get a coefficient
of exactly zero. Every penalty of the path is predicted, and
`penalty_selected_<t>` names the one with the lowest exponentially weighted
out-of-sample squared error so far, at no extra solve. Use
[`ewridge`](#ewridge--exponentially-weighted-ridge-regression) for a dense
fit, and [`ftrl`](#ftrl--online-logistic-regression) for a sparse logistic
fit of a 0/1 target.

Its parameters, their defaults and what each refuses are in its
[docstring](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.lasso),
and the fields it writes in [OUTPUTS.md](OUTPUTS.md#lasso). The parameters
every spec takes are in [What a spec names](../README.md#what-a-spec-names).

#### The update

The running sums are `ewridge`'s, centered. On the solve schedule, for each
penalty `l` in `penalties`, the model runs coordinate descent on the
standardized sums. The descent starts the first penalty from the last
solve's fit, and each penalty after it in `penalties` from the one before it. `C` is the
features' correlation matrix, `s_i` feature `i`'s standard deviation,
`c_i = cov(x_i, y)/s_i` its covariance with the target over its standard
deviation, and `b_i = β_i·s_i` its coefficient in standardized units. `s_y`
is the target's spread: its standard deviation over the rows the fit
reads, or its root mean square without an intercept. A sweep sets, for
every `i`:

```
ρ_i = c_i − Σ_{j≠i} C_ij b_j
b_i = soft(ρ_i, l·l1_ratio) / (C_ii + l·(1 − l1_ratio))      soft(v, t) = sign(v)·max(|v| − t, 0)
stop after a sweep that moves no b_i by tol·max(s_y, maxᵢ |c_i|) or more, or after max_iter sweeps
β_i = b_i / s_i         β_0 = ȳ − m·β         m the features' means, ȳ the target's
```

Without an intercept, the sums are raw moments, each feature is scaled by
its root mean square, and no intercept is recovered.

#### Example

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
slopes = bank.coef().filter(pl.col("penalty") == chosen, pl.col("term") != "intercept")
kept = slopes.filter(pl.col("coef") != 0.0)
print(f"penalty in force at the last row: {chosen}")
print("planted: x0 2.00, x1 -1.50, x2 1.00, the other 17 0")
print(f"coefficients left nonzero: {kept.height} of {slopes.height}:",
      {r["term"]: round(r["coef"], 2) for r in kept.iter_rows(named=True)})
```

```text
penalty in force at the last row: 0.03
planted: x0 2.00, x1 -1.50, x2 1.00, the other 17 0
coefficients left nonzero: 3 of 20: {'x0': 1.97, 'x1': -1.44, 'x2': 0.99}
```

#### Choosing its settings

**Scale `penalties` with the target.** The L1 threshold `l·l1_ratio` is in
the target's units, so for a pure lasso, a target ten times larger needs
penalties ten times larger to zero the same coefficients.

**To rank the penalties on the weighted mean of every out-of-sample
squared error so far, give `select_half_life=float("inf")`.** Under a hard
window the mean covers the hard window's rows. By default the error forgets at
the model's half-life.

**If `bank.solve_failures()` counts a descent that ran out of its
`max_iter` sweeps, give a larger `max_iter`.** Columns that nearly repeat
each other slow the descent most. The docstring's Notes, under *The
descent*, measure how many sweeps they take.

#### How it works

**The selection scores every prediction, the ones `min_weight` withholds
included.** If the gate held the selection back as well, the penalty
chosen would move with `min_weight`. With six penalties on a stream with
t(4) noise, such a selection differs after 3,000 rows from the choice at
`min_weight=0` on 6 of 50 seeds at the default `min_weight`, and on 13 at
a `min_weight` of 50.

#### See also

When `lasso`'s fit is not the one you need, or to run its descent outside a
bank:

- [`ewridge`](#ewridge--exponentially-weighted-ridge-regression), the
  running sums this model solves, under a ridge.
- [GRAM.md](GRAM.md), whose `po.gram.lasso_path` runs the same descent
  offline on a Gram.
- [`ftrl`](#ftrl--online-logistic-regression), a sparse fit of a 0/1
  target.
- [A hard window](../README.md#a-hard-window), which cuts the sums and the
  selection's errors at the same row, so a feature constant inside the
  hard window gets a coefficient of exactly zero.

#### Notes

**With no decay, a point of `lasso`'s path is statsmodels' elastic net on
the same rows.** At a penalty of 0.1, with a third feature that is noise,
the coefficients agree within 1e-7 and both zero the noise feature. At a
penalty of 0 the predictions are least squares from numpy, within 1e-6.
Its solves are also held to the lasso's optimality conditions
(Karush-Kuhn-Tucker), and `po.gram.lasso_path` to scikit-learn's
`enet_path`.

### `kalman` — random-walk-β dynamic linear model

*API:* [`po.spec.kalman`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.kalman) — *Rust:* [`kalman.rs`](../crates/online-core/src/kalman.rs) — *Outputs:* [fields](OUTPUTS.md#kalman)

A regression whose coefficients drift, tracked by a Kalman filter.

Use `kalman` when a coefficient moves at a drift rate of its own, faster
than one half-life for the whole fit can follow, or when a coefficient no
row supports should be forgotten (`revert_half_life`). Use
[`ewridge`](#ewridge--exponentially-weighted-ridge-regression) for
coefficients that hold still over a half-life, and
[`rls`](#rls--recursive-least-squares) for an exact fit whose prior fades.

Its parameters, their defaults and what each refuses are in its
[docstring](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.kalman),
and the fields it writes in [OUTPUTS.md](OUTPUTS.md#kalman). The parameters
every spec takes are in [What a spec names](../README.md#what-a-spec-names).
Here `half_life` decays the noise estimate and the features'
standardization, and `coef_half_life` sets how fast a coefficient may
drift.

#### The update

Per target `j`, `β_j` are the coefficients and `R_j·P_j` their covariance.
`P_j` is held in units of the observation noise `R_j`, the target's
exponentially weighted residual variance, or `obs_var` where given. `z̃`
is `z` with each feature standardized against the running means `m` and
standard deviations `s` of the rows before it, `[1, (x − m)/s]` (`x/r`
without an intercept, with `r` each feature's root mean square), and `z`
itself under `standardize=False`. `d` is the row's clock step, capped
at `gap_cap`, and `κ_i` is coefficient `i`'s `revert_half_life`. `Q` is the
process noise in the same units as `P_j`: `(ln 2/h_i)²` from a
`coef_half_life` of `h_i`, or `q_i/R_j` from `q`:

```
β_j ← Φβ_j     P_j ← ΦP_jΦ        Φ = diag(2^(−d/κ_i))        the reversion, every row
D_j ← D_j + d                                                the clock since y_j was last observed
P_j ← P_j + Q·D_j²,  D_j ← 0                                 the drift, on a row that observes y_j
v   = z̃ᵀP_j z̃ + 1/w                 the innovation's variance
K   = P_j z̃ / v                     the gain
β_j ← β_j + K(y_j − z̃ᵀβ_j)          P_j ← P_j − K z̃ᵀP_j
```

A row's weight scales its observation's precision: the observation's
variance is `R_j / w`. A reverting coefficient's drift over a gap `D` is
`Q_i·g_i(D)²` in place of `Q_i·D²`, with `g_i(D) = (1 − 2^(−D/κ_i))/θ_i` and
`θ_i = ln 2/κ_i`. It is `Q_i·D²` for a short gap, and never more than
`Q_i/θ_i²`, so a reverting coefficient's uncertainty stays bounded across a
run of null targets. The docstring's update gives the form under
`obs_var`, which runs on the covariance itself.

#### Example

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

Ten rows after the jump `kalman` has covered over a quarter of it, and 200 rows
after, all of it. `ewridge` still averages the slope over its half-life of
200 rows.

#### Choosing its settings

**For a drift rate per coefficient, give `coef_half_life` a list with the
intercept's entry leading.** An entry of `inf` gives its coefficient no
drift. To set the process noise outright, give `q=` in place of
`coef_half_life`. Each `q_i` is added as `q_i·D²` in the coefficient's squared
own units, and is measured against the noise estimate, the one place the
estimate reaches the fit.

**For a feature that is active only now and then, give its coefficient a
finite `revert_half_life`.** The coefficient is then pulled toward zero on
every row, while the rows that support it pull it back (an AR(1) prior).
So the effect is forgotten between its bursts, and cannot persist through
a run of null targets. For a feature that is always active, keep the
default `inf`. The pull would hold a persistent effect below its true
size, the more so the shorter the reversion half-life.

**Under `standardize`, the default, the reversion pulls toward zero in the
standardized coordinates:** "no effect" for a slope, and "the target
averages zero" for the intercept. So unless the target averages zero, keep
the intercept's entry at `inf`, as in `[float("inf"), 50.0, 50.0]`.

**For a coefficient per current standard deviation, make the z-score with
`po.ewm_mean` and `po.ewm_std` in the query's `.online.with_windows`
([`po.stream.with_windows`](https://hgilde.github.io/polars-online/stream.html#polars_online.stream.with_windows)),
chained in the query before the bank, and give that
column with `standardize=False`,** as in [Features in units of their
spread](WINDOWS.md#features-in-units-of-their-spread).
Under `standardize`, the coefficients stay in the columns' own units
however the standardization moves.

**To let a far prediction approach the intercept, give a `gap_cap` that
spans several of the slopes' reversion half-lives, and keep the
intercept's `revert_half_life` at `inf`.** `predict` applies the reversion by
the same `Φ` over the clock since the last learned row, capped by
`gap_cap`, so a slope keeps at least `2^(−gap_cap/κ_i)` of its value.
Under `standardize` the intercept is the target's level at the features'
means.

#### How it works

**Under `coef_half_life`, the gain never reads the noise estimate.** A
filter whose gain read it would weigh each row by the inverse of the noise
as it stood before the row, so a burst of large residuals would set the
weights. The docstring's Notes, under *The noise*, give such a run with no drift: a
burst in a fifth of the first 500 rows held the slopes at `[-25.5, 33.6]`,
where least squares gives `[4.1, -1.4]`. Weighing rows by a running noise
estimate is also volatility weighting. On the intraday returns of
[VALIDATION.md](VALIDATION.md), section 5, the filter with the estimate in
its gain reached R² −0.059 on the next minute's return, against −0.070 for
this one, and −0.075 against −0.101 on five minutes'. So on returns whose
volatility clusters, keeping the estimate out of the gain gives up some
R², in exchange for a fit that weighs a burst of large residuals as it weighs
any other rows.

**Without `obs_var`, the intercept's prior is diffuse, so a level in the
target moves every prediction by that level from the first row.** The
first row that observes the target puts its whole innovation into the
intercept. That is the limit of an intercept prior whose variance grows
without bound (Durbin and Koopman, *Time Series Analysis by State Space
Methods*, §5.2). A filter whose prior was sized by the noise estimate took
the level in through its first innovations. At a level of `1e4`, with a
`coef_half_life` of 200 and a half-life of 300, its predictions stood 0.019 to 0.032
of the target's spread off the unshifted fit over rows 3,000 to 6,000.
Under `coef_half_life` and without `obs_var`, neither the gain nor the
prior reads the noise, so the warm-up does not depend on the target's
units.

#### See also

When `kalman`'s fit is not the one you need, or to use it past the data:

- [`ewridge`](#ewridge--exponentially-weighted-ridge-regression), for
  coefficients that hold still over a half-life.
- [`rls`](#rls--recursive-least-squares), the fit this filter gives with
  no drift, a fixed noise, `standardize=False`, no `revert_half_life` and
  `half_life=inf`, at `delta = 1/p0`.
- [`bank.predict()`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.ModelBank.predict),
  which moves the coefficients by `Φ` over the clock since the last
  learned row.

#### Notes

**With `standardize=False`, `q` all zeros, no `revert_half_life` and a
fixed `obs_var=`, the filter is exactly Bayesian linear regression,** the ridge with penalty
`1/p0`: river's `BayesianLinearRegression`, without an intercept, agrees
within 1e-12. Standardized, each prior waits for its feature's scale, so it
is not that ridge. Held to filterpy's `KalmanFilter`, configured to
match, an explicit `q`, a fixed `obs_var` with `p0`, and a
`coef_half_life` per coefficient agree to 1.1e-15 relative, and no
intercept under `standardize` to 1.0e-14. `share_p=True`
changes no prediction where every target is on every row, apart from under
`q`: on section 4 of
[VALIDATION.md](VALIDATION.md) it and a `P` per target score the same, to
the bit.

### `huber` — robust regression

*API:* [`po.spec.huber`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.huber) — *Rust:* [`robust.rs`](../crates/online-core/src/robust.rs) — *Outputs:* [fields](OUTPUTS.md#huber)

Huber regression on running sums, which down-weights the rows a prediction
misses by far.

Use `huber` in place of
[`ewridge`](#ewridge--exponentially-weighted-ridge-regression) where a
share of the rows are outliers that would pull a least-squares fit. Use
[`quantile`](#quantile--robust-regression) to fit a quantile of the
target, such as the median, and [`sgd`](#sgd--stochastic-gradient-descent)
with `loss="huber"` where learning a row must take O(k) time for `k`
features: this model takes O(k²) time.

Its parameters, their defaults and what each refuses are in its
[docstring](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.huber),
and the fields it writes in [OUTPUTS.md](OUTPUTS.md#huber). It takes
`ewridge`'s `ridge=` as a single number, and refuses a list. The
parameters every spec takes are in [What a spec
names](../README.md#what-a-spec-names).

#### The update

`e` is the row's residual against the fit before the row is learned, so
the weight the row is learned at stays out-of-sample. `σ` is the target's
residual scale as the row arrives: by default the MAD (median absolute
deviation), the exponentially weighted median of `|e|` over
`Φ⁻¹(3/4) = 0.6745`. `δ` is `huber_delta`. The row enters [`ewridge`'s](#ewridge--exponentially-weighted-ridge-regression)
update equations for `S_j`, `η_j` and `W_j` at its weight cut
by its residual, and the coefficients `β_j` are solved on `ewridge`'s
schedule:

```
w_r = w · min(1, δσ / |e|)           the row's Huber weight; w itself until σ is above 0
S_j, η_j, W_j:  ewridge's update equations, at weight w_r in place of w
solve, ridge_scale="mean":  (S_j + ridge·D) β_j = η_j          D = I with a 0 at the intercept
solve, ridge_scale="sum":   (W_j·S_j + p·ridge·I) β_j = W_j·η_j
                            W_j after the Huber weights, and the intercept penalized too
```

`z`, `w`, `S_j` and `η_j` are as in the [notation table](#models). `W_j` is
the weight behind `S_j` and `η_j`, as in `ewridge`'s update, and `p` the
product of every decay the stream has applied.

The weights are per target, so `huber` keeps one set of sums per target,
where `ewridge` shares one set across targets. `weight_sum`, and each
target's weight that `min_weight` reads, count the raw weights `w`. When a
target's count of learned rows reaches `huber_refit_rows`, its first rows
are refit once, to convergence, and the stream goes on online from that
fit. The docstring's Notes give the refit, under *The refit of the first
rows*, and the scale `σ`, under *The scale*.

#### Example

**A few wild targets pull a least-squares slope. The default Huber fit and
the median hold near the truth, and a Huber fit drawn in a root mean square
does not.** Here about 2% of the rows are 50 too high, all where `x` is
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
only the sign of a row's miss beyond its band (see
[`quantile`](#quantile--robust-regression)), so a wild row counts no more
than an ordinary one. `huber_rms` draws its cut in a scale that the wild
rows themselves widen, so it lets more of their pull through: a root mean
square is not a robust scale.

#### Choosing its settings

The docstring defines every setting the advice names.

**To come close to the numbers of a version before the refit existed
(1.3.0 and earlier), give `huber_refit_rows=0`.** The MAD's sketch is then
on the fixed buckets those versions used. For a version before
`huber_scale` existed (1.2.0 and earlier), give `huber_scale="rms"` as
well, which keeps no sketch. Under `huber_scale="rms"`, with the default
`min_weight` on unit weights, `huber_refit_rows=0` gave 1.3.0's numbers to
the bit on the stream it was tested on. `sgd`'s steps have changed since (its clip, AdaGrad's
epsilon and its intercept read the target's own spread and level), so for
`sgd(loss="huber")` the refit off is the one-pass fit of its current steps.
A state saved by an earlier version continues to the bit.

**`refit_budget` bounds the refit buffers of all a spec's groups together:
320 MiB by default, or `inf` for no bound.** A learned row takes
`(features + 10) · 8` bytes under the MAD: its features, target, weight
and two decay factors, the residual the scale reads, and the residual's
node in the tree it is read from. It takes `(features + 4) · 8` under
`"rms"`. So one feature and 500 rows take about 44 KB a target under the
MAD, and 20 KB under `"rms"`. A group
shorter than `huber_refit_rows` never refits, and holds its rows to its
end.

Each group's reservation is saved with the bank. So which groups refit
depends neither on the chunking, nor on the order the groups run in, nor
on a save and load between them. In exchange, once the bound is reached no
later group refits, though the first groups' buffers have long been freed.
Measured on the same 150-row groups, fed from parquet, as the peak
resident memory after the fit and the saved state (the median of three
runs, which spread by up to 120 MiB):

| groups | `huber_refit_rows=0`: peak; state | the default, 320 MiB: peak; state; targets given up | `refit_budget=inf`: peak; state |
|---|---|---|---|
| 5,000 | 572 MiB; 128 MiB | 691 MiB; 177 MiB; 8,178 of 10,000 | 1,199 MiB; 391 MiB |
| 20,000 | 1,195 MiB; 512 MiB | 1,390 MiB; 563 MiB; 38,178 of 40,000 | 3,915 MiB; 1,565 MiB |

A reservation is the most a buffer can hold, and these groups' buffers
held less.

#### How it works

[The robust scale and the refit of the first
rows](explanation/robust-scale.md) explains Huber's loss and why its
scale must be robust. It measures the streams one pass leaves captured,
the refit over 40 seeds, and what a robust scale gives up when the level
shifts.

**A refit runs once, and a row after it takes the time it did, within
2%.** The default count grows with the features, 10 rows a feature past
50, so that the refit has rows enough for its coefficients. It stops at
100 features, where a target's rows would pass 0.8 MB. A count given is
taken at any width. A count given below the number of coefficients, such
as 200 rows at 1,000 features, leaves the fit the ridge's, and the
iteration runs to its cap of 200 passes: about 9 s once, which
`solve_failures` counts. A cap of 50 passes would take about 2 s there,
but 200 is where the slow refits of heavy one-sided outliers end. Over
3,000 streams of 50 to 500 rows, with up to 45% of the rows on one side
moved by a level, 50 passes ran out on 127 streams for `huber` and 200 on
8, the slowest refit taking 428 passes. For `sgd(loss="huber")` they ran
out on 149 and 9, all nine past 39%.

#### See also

When `huber`'s fit is not the one you need, or to see its loss from
another side:

- [`ewridge`](#ewridge--exponentially-weighted-ridge-regression), the
  least-squares fit on sums of the same form.
- [`quantile`](#quantile--robust-regression), a fit of a quantile of the
  target, on the same sums, scale and refit.
- [`sgd`](#sgd--stochastic-gradient-descent) with `loss="huber"`, the same
  loss with a step per row in O(k) time.
- [`rls`](#rls--recursive-least-squares), whose fit `huber` gives with
  `ridge_scale="sum"` and `huber_delta=inf`.
- [GRAM.md](GRAM.md#huber-regression-on-a-gram), a Huber fit from one
  shared Gram, a query over the rows per pass.

#### Notes

**`huber`'s one pass is held to a numpy reference of its own recursion
within 1e-12 of `1 + |pred|` (measured at most 1.9e-15), and its refit to statsmodels' `RLM` on the same rows within
1e-5.** At `huber_delta=inf` with `ridge_scale="sum"`, solved on every row,
it gives `rls`'s predictions with `delta=ridge` within 1e-9, as
`ewridge(ridge_scale="sum")` does. Against a batch fit of the same
objective over a whole stream, scikit-learn's `HuberRegressor`, it lands
close but not equal, for two reasons:

| reason | applies to |
|---|---|
| each row's weight was set by the fit before it | each row past the refit, and every row where the refit is off or never runs |
| `σ` is an exponentially weighted median, read exactly from the held residuals while the refit waits, and from a sketch after it or where the refit is off, where a batch fit takes the median of the final residuals | every row (under `"rms"`, `σ` is a root mean square) |

### `quantile` — robust regression

*API:* [`po.spec.quantile`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.quantile) — *Rust:* [`robust.rs`](../crates/online-core/src/robust.rs) — *Outputs:* [fields](OUTPUTS.md#quantile)

Quantile regression on running sums, the fit that a share `quantile` of the
targets fall below.

Use `quantile` for a conditional median, which outliers cannot pull, or
for a tail, such as the 0.9 quantile of a return, where a mean says
nothing. Use [`huber`](#huber--robust-regression) for a mean that outliers
should not pull, and [`sgd`](#sgd--stochastic-gradient-descent) with
`loss="quantile"` where learning a row must take O(k) time for `k`
features: this model takes O(k²) time.

Its parameters, their defaults and what each refuses are in its
[docstring](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.quantile),
and the fields it writes in [OUTPUTS.md](OUTPUTS.md#quantile). It takes
`ewridge`'s `ridge=` as a single number, and refuses a list. It takes no
`feature_sets`, since its rows outside the band step the fit through one
system per target. The parameters every spec takes are in [What a spec
names](../README.md#what-a-spec-names).

#### The update

`quantile` takes one Newton step per row on the check loss, smoothed by a
uniform kernel of half-width `h`. The step is linearized at the fit the row
was predicted with, so it stays out-of-sample. `e` is the row's residual,
`τ` is `quantile`, and `σ` is the target's residual scale, the MAD: the
exponentially weighted median of `|e|` over `Φ⁻¹(3/4) = 0.6745`. `k` is the
number of coefficients, and `n` the target's present rows, counted one
each and decayed, whatever their weights. `A` is the band's system the
solve factorizes, and `u` the row's deviation from the band's means.
`z`, `w`, `S_j` and `η_j` are as in the [notation table](#models):

```
ψ_τ(e) = τ − 1{e < 0}
h      = σ · max(quantile_eps, (k/n)^(2/5))      the band's half-width
|e| <  h:  a least-squares row (ewridge's update equations for S_j and η_j), with target  y + 2h(τ − ½)
|e| ≥  h:  nothing to S_j;  w · 2h · ψ_τ(e) · z  into η_j, over the band's weight,
           capped at  |e| / (1 + uᵀA⁻¹u)
```

The cap reads the row's full leverage against `A`, so a step moves the
fit at the row by at most the row's residual. Under three rows per
coefficient, and until a target has a `σ` above 0, a row is a
least-squares row. Like `huber`, `quantile` keeps one set of sums per
target, and refits each target's first rows once, to convergence, when its
count of learned rows reaches `quantile_refit_rows`. The docstring's
Notes give the warm-up under *Warm-up and rebuilding*, the band's floor
under *The band*, and the refit under *The refit of the first rows*.

#### Example

**`quantile=0.9` fits the line that 90% of the targets fall below.** Here the noise
is normal with a spread of 1, so the 0.9 quantile of `y` is the line
`1 + 1.28 + 2x`:

```python
import numpy as np
import polars as pl
import polars_online as po

# A line with normal noise of spread 1: its 0.9 quantile is 2.28 + 2x.
rng = np.random.default_rng(4)
x = rng.standard_normal(20_000)
y = 1.0 + 2.0 * x + rng.standard_normal(20_000)
pl.DataFrame({"x": x, "y": y}).write_parquet("tail.parquet")

q90 = po.spec.quantile(
    "q90", targets=["y"], features=["x"], half_life=float("inf"),
    quantile=0.9,                                  # the level to fit
)
bank = po.ModelBank([q90])
bank.fit(pl.scan_parquet("tail.parquet"))          # learn every row, keep no output
# The line found, beside the true one.
print("the true line: intercept 2.28, slope 2.00")
c = {r["term"]: r["coef"] for r in bank.coef().iter_rows(named=True)}
print(f"found: intercept {c['intercept']:.2f}, slope {c['x']:.2f}")
```

```text
the true line: intercept 2.28, slope 2.00
found: intercept 2.31, slope 1.99
```

The median takes only the sign of a row's miss beyond its band, so a wild
row counts no more than an ordinary one. [`huber`'s
entry](#huber--robust-regression) runs `quantile(quantile=0.5)` beside
`huber` and least squares: the median's slope is 2.03 against a true 2,
where least squares gives 2.89.

#### How it works

[The refit of the first
rows](explanation/robust-scale.md#the-refit-of-the-first-rows) measures
`quantile`'s refit against statsmodels' `QuantReg` over 50 streams. The
refit's buffer, its count's default and `refit_budget` are
[`huber`'s](#huber--robust-regression).

#### See also

When `quantile`'s fit is not the one you need:

- [`huber`](#huber--robust-regression), the robust fit of the mean, whose
  scale sketch this model reads.
- [`ewridge`](#ewridge--exponentially-weighted-ridge-regression), the
  least-squares fit on sums of the same form.
- [`sgd`](#sgd--stochastic-gradient-descent) with `loss="quantile"`, a
  step per row in O(k) time.

#### Notes

**`quantile`'s one pass is held to a numpy reference of its own recursion
within 1e-9 of `1 + |pred|`.** Against a batch fit of the same objective over a whole
stream, statsmodels' `QuantReg`, it lands close but not equal, for three
reasons:

| reason | applies to |
|---|---|
| each row was taken by the fit before it | each row past the refit, and every row where the refit is off or never runs |
| `σ` is an exponentially weighted median, read exactly from the held residuals while the refit waits, and from a sketch after it or where the refit is off, where a batch fit takes the median of the final residuals | every row |
| the check loss is smoothed over the band | every row |

The docstring's Notes, under *Why a Newton step*, measure the step against
`QuantReg` at the median of a skewed noise, and under *The scale*, the MAD
against a root mean square.

### `sgd` — stochastic gradient descent

*API:* [`po.spec.sgd`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.sgd) — *Rust:* [`sgd.rs`](../crates/online-core/src/sgd.rs) — *Outputs:* [fields](OUTPUTS.md#sgd)

Stochastic gradient descent under a choice of losses, with one gradient
step a row.

Use `sgd` when learning a row must take only O(k) time for `k` features:
it takes one gradient step a row and never solves. It is the only model
that takes count targets (`loss="poisson"`), and its coefficients can be
bounded or summed to a constant, as portfolio weights are. Use
[`ewridge`](#ewridge--exponentially-weighted-ridge-regression) for an exact
fit in O(k²) time a row, [`huber`](#huber--robust-regression) or
[`quantile`](#quantile--robust-regression) for a robust or quantile fit
solved on running sums, and [`pa`](#pa--passive-aggressive-regression) for
steps with no learning rate to tune.

Its parameters, their defaults and what each refuses are in its
[docstring](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.sgd),
and the fields it writes in [OUTPUTS.md](OUTPUTS.md#sgd). The parameters
every spec takes are in [What a spec names](../README.md#what-a-spec-names).

#### The update

Here `p` is the prediction, `h` the inverse of the loss's link, `d` the
loss's gradient at `p`, `lrᵢ`
coefficient `i`'s rate on the row, and `Gᵢ` AdaGrad's running sum of its
squared gradients. With `standardize=False`, `z = [1, x]` and `β` is in the
caller's units, so a row takes this step:

```
eta = zᵀβ        p = h(eta)        d = dL/d eta
g₀ = d·w         gᵢ = d·zᵢ·w + l2·βᵢ     each clipped to ±clip_gradient·u; the intercept takes no l2
βᵢ -= lrᵢ·gᵢ     lrᵢ = learning_rate                           schedule="constant"
                     = learning_rate/(1 + λ·W)^power           schedule="inv_scaling"
                     = learning_rate/(√Gᵢ + 1e-8·u)            schedule="adagrad"
```

`λ·W` is the weight before the row, decayed to it, and `Gᵢ` decays by `λ`
too. So a row of weight 0 in between leaves the next row's rate where it
would be without that row.

`u` is the unit the gradient is in. Under `squared` and `huber`, whose
gradient is in the target's units, it is `s_y`, the target's own spread,
and there is no clip before the target has one. Under the other losses,
whose gradient is a sign, a count's residual or a probability's, it is 1.
So the clip and AdaGrad's epsilon mean the same at every scale of the
target.

The loss sets the link, whose inverse `h` gives the prediction, and the
gradient:

| loss | link | `p = h(eta)` | `dL/d eta` |
|---|---|---|---|
| `squared` | identity | `eta` | `p − y` |
| `huber` | identity | `eta` | `clamp(p − y, ±delta·s)` |
| `quantile` | identity | `eta` | `1{y < p} − τ` |
| `epsilon_insensitive` | identity | `eta` | 0 within `eps·s_y` of `y`, else `sign(p − y)` |
| `poisson` | log, for count targets | `exp(clamp(eta, ±30))` | `p − y` |
| `logistic` | logit, for 0/1 targets | `sigmoid(eta)` | `p − clamp(y, 0, 1)` |

`delta` is `huber_delta`, `τ` is `quantile` and `eps` is the tube's
half-width. `s` is the scale of the target's out-of-sample residuals, as
for `huber`: the exponentially weighted median of `|y − p|` over 0.6745
under the default `huber_scale="mad"`, or their root mean square under
`"rms"`. `s_y` is the target's own exponentially weighted standard
deviation.

Under `standardize`, the default, the step is taken on the standardized
row `z̃`, with `z̃ᵢ = (xᵢ − mᵢ)/sᵢ` and `mᵢ` and `sᵢ` column `i`'s running
mean and standard deviation. Without an intercept, `z̃ᵢ = xᵢ/sᵢ`, with `sᵢ`
its root mean square (not centered). The ridge reads `l2·sᵢ·βᵢ`. So
`gᵢ` is `d·z̃ᵢ·w + l2·sᵢ·βᵢ`. Once the scaler has warmed up, the step taken
on `z̃` is mapped into the caller's units:

```
βᵢ -= lrᵢ·gᵢ / sᵢ        β₀ -= lr₀·g₀ − Σᵢ mᵢ·lrᵢ·gᵢ / sᵢ
```

Under an identity link with an intercept, the intercept takes the running
mean's step in place of its gradient's. Here `W_y` is the target's learned
weight (undecayed, this row's in it), `a = w/W_y`, and `n` is Kish's
effective sample size of the target's rows:

```
squared, huber:                  β₀ += −a·d        while a is above the row's rate, or n ≤ 22
                                                   (under schedule="adagrad", only while n ≤ 22)
quantile, epsilon_insensitive:   β₀ += a·(y − p)   while n ≤ 22
```

The docstring's Notes, under *The target's level*, give the reason and
its measurements.

#### Example

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
    loss="poisson",                          # a chunk with a negative target is refused, naming the row
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

Each coefficient the fit found is within about 0.01 of the log-rate's.

#### Choosing its settings

The docstring defines every setting the advice names.

**To set how many rows the coefficients remember, choose `learning_rate`.**
Every row's step moves the coefficients and the half-life does not decay
them, so under a constant rate they remember about
`1 / (learning_rate · E[z²])` rows, whatever the clock between rows. Under
`schedule="adagrad"`, `Gᵢ` decays on the clock, so the adapted rate opens
up again after a long gap.

**On a stream whose level can change, give `huber_scale_half_life` about a
fifth of `half_life`, or shorter.** Under the default `"mad"`, `sgd`'s
Huber loss follows a change of level far more slowly than under `"rms"`.
Its gradient is clipped at `delta·s`, so while `s` is the noise's scale the
intercept moves at most `learning_rate·delta·s` a row. A shift many times
the noise waits until the median itself has crossed to the new level. For
`huber`, the mean absolute error under the MAD scale (`"mad"`) 200-400 rows
after the shift was 19.5, against 15.1 under `"rms"`
([its entry](#huber--robust-regression) has the table). For `sgd` the error
under `"mad"` is seven times the error under `"rms"`. On `huber`'s stream (`y = 2x + 0.5ε`, a fit half-life of 500
rows, a shift of +20 at row 10,000, no outliers), the mean absolute error
after the shift, and the rows after the shift until every 200-row average
stays under 1.0:

| `sgd` | rows 0-200 after the shift | rows 200-400 | rows after the shift until every 200-row average stays under 1.0 |
|---|---|---|---|
| `loss="squared"` | 8.67 | 1.17 | 418 |
| `loss="huber"`, `"rms"` | 13.55 | 2.45 | 493 |
| `loss="huber"`, `"mad"`, the scale's half-life the fit's 500 rows | 19.26 | 17.33 | 885 |
| `loss="huber"`, `"mad"`, `huber_scale_half_life=100` | 15.81 | 2.99 | 510 |
| `loss="huber"`, `"mad"`, `huber_scale_half_life=20` | 10.34 | 1.41 | 440 |

Seeds 2 and 3 agree to within 0.06. A shorter memory widens the cut sooner
in a long burst of outliers too.

**Under `loss="poisson"`, keep `clip_gradient`, `1e3` by default,** because
through the log link one large count would make the next gradient
exponentially bigger. At ordinary scales it does not bind for the other
losses. A Poisson fit takes counts, so a call that reads the target (one that
learns, or `predict` on a chunk that holds it) refuses a chunk with a
negative target and names the row, as scikit-learn's `PoissonRegressor`
refuses one. A negative target taken as it stood would make `p − y` drive
the prediction down to the link's floor, `e^−30`, and hold it there.

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
    loss="squared",              # any loss in the update's table
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

#### How it works

**The stability check reads a sample of rows, so it is built to resist one
wide row and to rest while the rate is calm.** The docstring's Notes,
under *A rate past its stability bound*, give the rule and the check.

**One wide row cannot hold the output.** Each gain after a target's first
enters the sums bounded, at most the larger of 2 and four times the
running mean `m` before it. A gain of 2 leaves its row's residual where it
was and adds nothing to the margin `Σ(q² − 2q)`. So while `m` is below 1/2,
no single row moves the verdict toward unstable. Above it, a row moves the
verdict by at most `16m² − 8m`. Entered whole, one row at 100 times a
feature's spread held a stable rate's output for 1,075 rows (raw features,
a rate of 0.01), and one at 10,000 times for 3,736.

**The check's memory counts rows, a half-life of 100 stepped rows, because
its verdict estimates an expectation from a sample of the rate's rows.** The error of a ratio of two means of `N` gains is about `CV(q)/√N`, where
`CV(q)` is the gains' standard deviation over their mean. That spread is
set by the spread of `|z|²`, which is the rows' and not the rate's or the
decay's.
The fit's own memory, about `k / mean(q)` rows for a coefficient, is a
handful of rows near the bound, too few to estimate from. As the check's
memory it withheld 6-11% of a stable rate's rows (at 0.77 times the bound),
and served a third to a half of a diverging one's (at 1.15 times). The
model's decay ages the target's weight and AdaGrad's sums, not the
coefficients, so it is no memory of the rate either.

**A calm check rests.** Under the squared and Huber losses, it rests once a
target's sums hold two of those half-lives (200 stepped rows, a weight of 108, three quarters
of their limit) and the gains' mean square is at most their mean. Resting,
it takes no gains, so no row forms `|z|²` for it. It probes every 8th
row's gain, and wakes where the probe passes twice the resting mean, or
where a row's residual passes 8 of the target's spreads. On five
standardized features over 10 streams each, a rate increased after 2,000
rows from half the bound to 1.15 times it was first withheld within a row
of where an awake check withheld it (113 to 225 rows after the change).
From half the bound to 1.9 times, it was withheld within 8 rows of it (26
to 47 rows after). From 0.3 to 1.5 times, it was withheld within 27 rows
(65 to 99 rows after, the awake check's range). At a constant rate of 0.01
on 20 features, the check rests on 95% of the
rows. The Poisson and logistic checks take every gain, since their
residual is not in the target's units.

**`sgd` standardizes its features by default, as `kalman` does, because one
learning rate has to suit every feature.** Without it, features 100 times
larger put the default rate past its stability bound, so every prediction
is withheld as `unstable_rate`. Once the scaler has learned 22 rows by
Kish's effective sample size, the coefficients are held in the caller's
units, so the scaler's moving moves no prediction. Read through the
moments instead, a half-life of 10 put the slope at 3.03 for a truth of 2
at R² 0.978. Held in the caller's units, it is 2.02.

**The refit of each target's first rows brings the Huber loss's one pass
near statsmodels' `RLM`.** In one pass, the outliers capture the first
steps. [The refit of the first
rows](explanation/robust-scale.md#the-refit-of-the-first-rows) gives
`sgd(loss="huber")`'s figures over 40 seeds beside `huber`'s.

#### See also

When `sgd`'s steps are not the fit you need:

- [`pa`](#pa--passive-aggressive-regression), steps with no learning rate,
  with the same scaler and bounds.
- [`huber`](#huber--robust-regression) and
  [`quantile`](#quantile--robust-regression), the robust and quantile fits
  solved on running sums.
- [`ftrl`](#ftrl--online-logistic-regression), a logistic or linear fit
  whose L1 penalty zeroes coefficients.

#### Notes

**Under `loss="quantile"` and `schedule="inv_scaling"`, `sgd` converges to
statsmodels' `QuantReg`.** At 100,000 rows, at the median and the 0.9
quantile of a skewed noise, its coefficients came within 0.022 and 0.027
of `QuantReg`'s, where the least-squares fit is 0.3 and 1.3 away.

**Under the squared loss, with `standardize` and no intercept, `sgd`
ends within 0.05 of numpy's least squares** over 20,000 rows, and the
coefficients it reports reproduce each prediction to 1e-13 of its size.

### `pa` — passive-aggressive regression

*API:* [`po.spec.pa`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.pa) — *Rust:* [`pa.rs`](../crates/online-core/src/pa.rs) — *Outputs:* [fields](OUTPUTS.md#pa)

Passive-aggressive regression, which moves the fit by the smallest step
that brings each row within a margin of its target.

Use `pa` to step on every row, in O(k) time a row for `k` features, without
tuning a learning rate. Use [`sgd`](#sgd--stochastic-gradient-descent) for
other losses or a learning rate, and
[`ewridge`](#ewridge--exponentially-weighted-ridge-regression) for an exact
fit on running sums.

Its parameters, their defaults and what each refuses are in its
[docstring](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.pa),
and the fields it writes in [OUTPUTS.md](OUTPUTS.md#pa). The parameters
every spec takes are in [What a spec names](../README.md#what-a-spec-names).

#### The update

Each row asks the fit to come within `eps·s_y` of its target, `s_y` the
target's own exponentially weighted standard deviation, as `sgd`'s tube is
drawn in. The step is the smallest change that does so. With `p` the
prediction, and the intercept's 1 inside `z`, a row takes the step its
`mode` sets:

```
loss = max(0, |y − p| − eps·s_y)      q = ‖z‖²
pa    τ = loss / q          pa1  τ = min(c, loss/q)      pa2  τ = loss / (q + 1/(2c))
β    += min(w, 1) · τ · sign(y − p) · z
```

That is the step under `standardize=False`, with `z = [1, x]`. Under
`standardize`, the default, the step is taken on the row standardized with
`sgd`'s scaler, `z̃`, which takes `z`'s place in `q` and the step. Past the
scaler's warm-up the coefficients are held in the caller's units, and the
step is mapped into them by the row's own means `mᵢ` and standard
deviations `sᵢ`:

```
step  = min(w, 1)·τ·sign(y − p)
βᵢ   += step·z̃ᵢ / sᵢ        β₀ += step − Σᵢ mᵢ·step·z̃ᵢ / sᵢ
```

The docstring's Notes, under *The margin's scale*, *c and the target's
units* and *The margin*, give why the margin is drawn in `s_y`, what `c`
is in, and how wide the margin is in units of the noise.

#### Example

**`pa` takes `sgd`'s bounds on the slopes.** Here the truth is
`1 + 2·x0 − x1`, and `coef_max=1.5` holds `x0`'s slope below its true
value, so the fit stays inside the bound, and its other coefficients move
to make up for it:

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
# What each fit found.
for name in ("free", "capped"):
    fitted = {r["term"]: round(r["coef"], 2) for r in bank.coef().filter(pl.col("spec") == name).iter_rows(named=True)}
    print(f"{name}: {fitted}")
```

```text
free: {'intercept': 0.99, 'x0': 2.02, 'x1': -0.98}
capped: {'intercept': 0.89, 'x0': 1.48, 'x1': -1.09}
```

The free fit finds the truth to within 0.02. The capped fit holds `x0`'s
slope under 1.5, and its intercept and `x1`'s slope move from the truth as
well, to 0.89 and −1.09.

#### Choosing its settings

The docstring defines every setting the advice names.

**On a level that moves, give a larger `c` or difference the target.** The
fit starts from zero coefficients, and a `pa1` cap of 1 moves the intercept
about one unit a row. On a target falling from 1,000 to −1,000 at 0.5 a
row, at a half-life of 200 and noise 0.3, the intercept met the level near
row 670. With the level as a feature too, the default cap's steps, mapped
into the caller's units through the scaler, swung the fit far past the
level instead. With `c=10`, every row stayed within 3 of its target from
row 467.
A drift the intercept carries alone keeps a lag of about 2 at any `c`, half
of it the tube's, since `s_y` holds the drift. Differencing the target and the moving features, `y − y.shift(1)`, takes
the level out of the fit, and the level is the last row's `y` plus the
predicted change. The docstring's Notes, under *On a level that moves, give a larger c or
difference the target*, give each run's `pred − y` at rows 800 and 1,600.

**Where outliers are possible, keep a `mode` that caps or damps the
step:**

| `mode` | the step |
|---|---|
| `"pa"` | moves the fit as far as one bad row demands |
| `"pa1"`, the default | is capped at `c` |
| `"pa2"` | is damped by `c` instead of capped |

**When `coef_min`, `coef_max` or `coef_sum` bound the slopes, keep `c`
small.** The projection that follows each step keeps the step from
meeting the row's margin exactly, and a truth outside the allowed set is
never reached, so the fit keeps stepping against the bounds. A small `c`
keeps those steps small.

#### See also

When `pa`'s steps are not the fit you need:

- [`sgd`](#sgd--stochastic-gradient-descent), gradient steps with a
  learning rate, under several losses, with the same scaler and bounds.
- [`ewridge`](#ewridge--exponentially-weighted-ridge-regression), an exact
  fit on running sums.

#### Notes

**With no margin, no intercept and unit weights, `pa` matches river's
`PARegressor`** in all three modes, on the raw features: the predictions
agree to 2e-15 and the coefficients to 9e-16 over 500 rows. The two are
compared at `eps=0` because river's margin is in the target's units, and
this model's is in units of `s_y`.

### `ftrl` — online logistic regression

*API:* [`po.spec.ftrl`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.ftrl) — *Rust:* [`ftrl.rs`](../crates/online-core/src/ftrl.rs) — *Outputs:* [fields](OUTPUTS.md#ftrl)

Logistic regression for a 0/1 target, or a sparse linear regression, one
gradient step per row.

Use `ftrl` for a 0/1 target, or for a sparse fit that steps on every row
instead of solving. It fits by FTRL-Proximal (follow the regularized
leader, with a proximal step): its L1 penalty zeroes a coefficient with too little
evidence, and its rates adapt to each feature's history (McMahan et al.
2013). Use [`lasso`](#lasso--lasso-path-its-penalty-chosen-as-it-runs) for
an L1 penalty on the mean scale, which more rows or heavier ones do not
outweigh, and [`sgd`](#sgd--stochastic-gradient-descent) for other losses.

Its parameters, their defaults and what each refuses are in its
[docstring](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.ftrl),
and the fields it writes in [OUTPUTS.md](OUTPUTS.md#ftrl). The parameters
every spec takes are in [What a spec names](../README.md#what-a-spec-names).

#### The update

`β` is the coefficients, and `zz`, `n` and `d` are the sums kept per
coefficient `i`. `λ_t` is the decay over the clock since the last row the
target was learned from, which is the page's `λ` on a stream where the
target is learned from every row (every row has it, at a weight above 0). `alpha`, `beta`, `l1`
and `l2` are the parameters of those names:

```
β_i  = 0 if |zz_i| ≤ l1 else −(zz_i − sign(zz_i)·l1) / (beta/alpha + d_i + l2)
p    = sigmoid(zᵀβ)                  zᵀβ itself under loss="squared"
zz_i ← λ_t·zz_i    n_i ← λ_t·n_i    d_i ← λ_t·d_i    on a row the target is learned from
g_i  = (p − y)·z_i·w      s_i = (√(n_i + g_i²) − √n_i)/alpha
zz_i += g_i − s_i·β_i      n_i += g_i²      d_i += s_i
```

Without a half-life, `d_i` is `√n_i/alpha`, so the denominator is
`(beta + √n_i)/alpha + l2`, the closed form of river's `FTRLProximal`.

#### Example

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

# One ftrl over all 50 features, each row predicted before it is learned.
sparse = po.spec.ftrl(
    "sparse", targets=["up"], features=names, half_life=float("inf"),
    l1=300.0,                      # zero a coefficient whose summed evidence stays under 300
    strict_binary=True,            # refuse a target that is not 0 or 1
)
bank = po.ModelBank([sparse])
out = pl.concat(bank.fit_predict_batches(pl.scan_parquet("clicks.parquet")))
# The features the L1 penalty kept.
slopes = bank.coef().filter(pl.col("term") != "intercept")
kept = slopes.filter(pl.col("coef") != 0.0)["term"].to_list()
# How often the second half's rows are classed right, out-of-sample.
p_up = out["sparse"].struct.field("pred_up")      # each row's probability, from the fit before the row
hits = ((p_up > 0.5).cast(pl.Float64) == out["up"])[n // 2:].mean()
print(f"coefficients left nonzero: {len(kept)} of {slopes.height}: {kept}")
print(f"rows of the second half classed right, out-of-sample: {hits:.2f}")
```

```text
coefficients left nonzero: 5 of 50: ['x0', 'x1', 'x2', 'x3', 'x4']
rows of the second half classed right, out-of-sample: 0.81
```

The penalty kept exactly the five features that move the probability, and
zeroed the other 45.

#### Choosing its settings

The docstring defines every setting the advice names.

**`ftrl`'s penalties keep a fixed mass, while the evidence grows with the
weights and the rows' density.** So a heavier or denser stream overcomes
them sooner, and a shorter half-life shrinks the fit more. At the defaults,
with unit weights and rows one clock unit apart, they act in steady state
as a ridge of `(1 − λ)(beta/alpha + l2)` on the mean scale. Under
`loss="squared"`, a constant target of 5 is then fitted at 4.65 at
`half_life=100`, and at 4.96 at 1000. For a penalty that weight and density
leave alone, use
[`lasso`](#lasso--lasso-path-its-penalty-chosen-as-it-runs), whose penalty
is on the mean scale. The ridge weighs most on a 0/1 target, whose
log-odds the evidence pins only weakly: a probability's curvature is at
most a quarter. On two features at `half_life=200`, with one row a clock
unit, the defaults left the logistic fit at its steady state up to 0.41
from the decayed maximum likelihood in some coefficient, the worst of 50
seeds. `beta=0.1, l2=0` left it 0.17 from it.

**Under a half-life, `ftrl` follows a break at the pace of its steps, not
of its half-life.** Each coordinate steps by its gradient over the root of
its recent squared gradients, so a coefficient at its steady state moves
at most about `2·alpha·√(1 − λ)` a row, with `λ` the decay over one row's
clock, however large the residual. A break that moves a coefficient by `D`
takes about `D / (2·alpha·√(1 − λ))` rows to follow. The large gradients just
after a break also increase the sums that divide every step, for about a
half-life. Measured under `loss="squared"` on two features at
`half_life=200`, one row a clock unit, after a change of regime that moved the intercept by 46 (the worst
of 50 seeds, the largest coefficient gap to the decayed least squares):

| `alpha` | at its steady state, before the break | 4 half-lives after | 8 half-lives after | 12 half-lives after |
|---|---|---|---|---|
| 0.1, the default | 0.17 | 34.4 | 28.5 | 19.9 |
| 0.3 | 0.24 | 18.4 | 2.7 | 0.86 |
| 1.0 | 0.43 | 2.7 | 0.61 | 0.85 |

At the default the intercept closed at 0.011 a row, against
`2·alpha·√(1 − λ)` = 0.012. A larger `alpha` follows a break sooner and moves
more from row to row at its steady state: at `alpha=1` the gap stops
shrinking by 8 half-lives, within twice its steady state's. Over the 24 half-lives after the
noise quadrupled, the default came at most 0.23 from the decayed least
squares, and `alpha=1` 1.17.

**`ftrl` does not center a feature, so give it features near 0.**
Standardize or z-score such a feature in the query before the bank, as
[Features in units of their spread](WINDOWS.md#features-in-units-of-their-spread)
shows. The docstring's Notes, under *Features at a level*, measure how far
a feature's level carries a coefficient's error into the prediction, and
say why the model has no `standardize`.

#### See also

When `ftrl`'s fit is not the one you need:

- [`lasso`](#lasso--lasso-path-its-penalty-chosen-as-it-runs), an L1
  penalty on the mean scale, solved on running sums.
- [`sgd`](#sgd--stochastic-gradient-descent), gradient steps under other
  losses.

#### Notes

**With decay off, `ftrl` matches river's `FTRLProximal`:** the two agree to 1e-12, row for
row, and Vowpal Wabbit's `--ftrl` agrees to its single precision.

### `holt` — Holt's linear trend

*API:* [`po.spec.holt`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.holt) — *Rust:* [`holt.rs`](../crates/online-core/src/holt.rs) — *Outputs:* [fields](OUTPUTS.md#holt)

Holt's linear trend, which extrapolates the target's own level and slope.

Use `holt` as the baseline a model with features should beat: it is the
only linear model that takes no features. Run it in the same bank as the
regression, with `emit_sigma=True` on both, then compare the two models'
`sigma` ([How wide is its
error?](DIAGNOSTICS.md#how-wide-is-its-error)), or let a
[`seqtest`](#seqtest--a-sequential-test-of-a-sign-by-betting) with `a` and
`b` say which predicts closer.

Its parameters, their defaults and what each refuses are in its
[docstring](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.holt),
and the fields it writes in [OUTPUTS.md](OUTPUTS.md#holt). The parameters
every spec takes are in [What a spec names](../README.md#what-a-spec-names),
apart from `features`, which `holt` does not take.

#### The update

Each row moves the level `l` and the trend `b` by a gain times the
residual `e = y − pred`. `s` is the clock since the target was last
observed, this row's step included, each step capped at `gap_cap` (or
`session_gap` at a session change). So a row with a null target or weight
0 leaves the level and trend where they were, and its step goes into the
next observation's `s`. `W` and `V` are the weight the level and the trend
have gathered, and `S0`, `ā` and `Q` are the line fit's weight, mean age
and spread of ages. A row takes the larger of two gains, on the level and
on the slope:

```
λ_l = 0.5^(s/half_life)                        λ_b = 0.5^(s/trend_half_life)
pred = l + b·s                                 e = y − pred
textbook:  g_t = w/(λ_l·W + w)                 W' = λ_l·W + w
           h_t = w/(λ_b·V + w) · g/s           V' = λ_b·V + w     (s > 0)
line fit:  S0' = S0 + w     ā' = S0·(ā + s)/S0'     Q' = Q + S0·w·(ā + s)²/S0'
           g_f = w·(1/S0' + ā'²/Q')            h_f = w·ā'/Q'
l' = pred + g·e     g = g_f under half_life=inf, else max(g_f, g_t)
b' = b + h·e        h = h_f under trend_half_life=inf or at s = 0, else max(h_f, h_t)
```

Under `trend=False` the line fit is not used: `g = g_t` at every
half-life, and `b = 0`. A target's first observation sets `l = y`,
predicts nothing, and starts the trend at 0. The docstring's Notes, under
*The textbook gains*, *The line fit* and *Which gain a row takes*, give
what each gain is and when each wins.

#### Example

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

The regression's error spread is about half of `holt`'s, and the test
finds the evidence enough at row 15.

#### Choosing its settings

The docstring defines every setting the advice names.

**The level forgets at the spec's `half_life`, as every model's decay
does.** `level_half_life` is refused, with a message naming `half_life`. To
make the trend the whole history's drift, give `trend_half_life=inf`: the
slope then takes the line fit's gain, and is the weighted least-squares
slope exactly when the level does not decay either (`half_life=inf`).
Beside a finite level half-life the slope learns from residuals against a
level that forgets, and lands near the line fit's slope: 0.2% to 1% of it
off at a level half-life of 5 over 3,000 rows of a line in noise.

**For a flat prediction, give `trend=False`:** simple exponential
smoothing, whose level is the target's exponentially weighted mean. It
holds the trend at zero, and refuses `trend_half_life`.

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

#### How it works

**The line fit keeps the trend from being pinned to the first
observation.** Gains of `1/n` on both level and trend, without the line
fit, pinned the trend to the first observation. The level then carried
that observation's noise times `ln n`: 11 off a flat series after 40,000
rows without decay, and 6.3 after 10,000 rows at `half_life=10000`, the
worst of 50 seeds at a noise of 0.5. The line fit's worst was 0.010 and
0.026, the weighted least squares' own.

#### See also

When `holt` is not the baseline you need, or to test a model against it:

- [`seqtest`](#seqtest--a-sequential-test-of-a-sign-by-betting), to test
  whether a regression predicts closer than this baseline.
- [`ewridge`](#ewridge--exponentially-weighted-ridge-regression), a
  regression on features to set against it.
- [How wide is its error?](DIAGNOSTICS.md#how-wide-is-its-error), the
  `sigma` that compares the two.

#### Notes

**Without decay, `holt` fits the weighted least-squares line,** whatever the weights
and the clock: statsmodels' `WLS` on `(1, t)` agrees to 1e-15. Under a
half-life, every row takes the textbook's gains after a warm-up of a few
half-lives (from row 745 at half-lives of 6 and 25). From row 1,000 in
that run, where the textbook gains have reached statsmodels' fixed
ones to 1e-12, the fit is statsmodels' `Holt` to 1e-12.

## Moments and correlation

These models measure the columns you name without fitting a target: their
means, variances and correlations, for a few columns or for thousands, and,
in `audit`'s case, the counts of what the columns hold. `ew_cov` and `deco`
write their statistics on every row, from the state before that row. The
means, spreads, correlations and `rho` read that state alone, so they can
serve as features for the row without leaking it. `mahal`, the
principal-component scores, `u` and `loglik` measure the row's own values
against that state. `marginal`,
`rcov` and `audit` write only `weight_sum` on a row, and keep their results
in the state: read them as a table after the run, or, for `rcov`, when a
block closes.

### `ew_cov` — exponentially weighted moments

*API:* [`po.spec.ew_cov`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.ew_cov) — *Rust:* [`ewcov.rs`](../crates/online-core/src/ewcov.rs) — *Outputs:* [fields](OUTPUTS.md#ew_cov)

Exponentially weighted moments of the feature columns, and the statistics
read off them.

Use `ew_cov` for the running means, spreads, correlations, principal
components or Mahalanobis distances of a stream's columns, each read before
its row so that it can be a feature for that row. It takes one O(k²) step a
row for `k` columns, where Polars expressions alone take O(k²) *passes over
the data*, one for each pair's exponentially weighted correlation. Use
[`marginal`](#marginal--every-pairs-moments-kept-in-the-state) to screen
thousands of features against a few targets,
[`deco`](#deco--one-correlation-for-the-whole-matrix) for one number that
moves with the whole correlation matrix, and
[`rcov`](#rcov--a-blocks-realized-covariance-robust-to-noise) for a block's
covariance under microstructure noise.

Its parameters, their defaults and what each refuses are in its
[docstring](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.ew_cov),
and the fields it writes in [OUTPUTS.md](OUTPUTS.md#ew_cov). The
parameters every spec takes are in [What a spec
names](../README.md#what-a-spec-names).

#### The update

The model keeps `m`, the weighted mean of the row's features `x`, and `C`,
the weighted mean of `(x − m)(x − m)ᵀ`, beside `W` and `Q`, the sum of the
squared weights. It updates them in Welford's centered form, because the
raw form, `E[x xᵀ] − m mᵀ`, loses a variance when the columns sit far from
zero. `c` is Bessel's correction at Kish's effective number of rows,
`n = W² / Q`, as Polars' `ewm_var` applies it under its default,
`bias=False`. Give `bias=True` for `C` itself, Polars' `bias=True` and
`ddof=0`. No correlation moves, since `c` cancels from each. `P`, read
before the row, serves `partial_corr` and `mahal` alone, with `s` a scale
on `precision_prior` that decays with the co-moments. Each lag `ℓ` of
`lags` keeps `C_ℓ`, with both deviations taken against the mean before the
row:

```
W' = λW + w      Q' = λ²Q + w²      a = λW / W'      b = w / W'      δ = x − m      (a + b = 1)
m' = m + b·δ     C'ᵢⱼ = a·Cᵢⱼ + a·b·δᵢδⱼ
c = W² / (W² − Q)       (1 under bias=True)
varᵢ = c·Cᵢᵢ     covᵢⱼ = c·Cᵢⱼ     corrᵢⱼ = Cᵢⱼ / √(CᵢᵢCⱼⱼ)
P = (c·(C + s·precision_prior·I))⁻¹      s = 1 on the first row, s' = a·s on each row after
partial_corrᵢⱼ = −Pᵢⱼ / √(PᵢᵢPⱼⱼ)      mahal = √(δᵀ P δ)
C_ℓ' = a·C_ℓ + a·b·(x_t − m)(x_{t−ℓ} − m)ᵀ      the co-moments' a and b, so lag 0 would be C exactly
```

The docstring's Notes, under *Bias*, give `c` in steady state at
half-lives of 20 and 2 rows.

#### Example

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

#### Choosing its settings

To flag rows that are odd only in combination, with `mahal` and
`mahal_quantiles`, see [Flag joint outliers](how-to/flag-joint-outliers.md).

To track principal components on a schedule, with `pca`, `pca_every` and
`max_rows_between_pcas`, see [Track principal
components](how-to/track-principal-components.md).

To find which column leads another, with `lags` and `"lag_corr"`, see
[Find which series leads](how-to/find-which-series-leads.md).

#### How it works

**Under a `window_size`, Polars gives the same numbers on a frame in
memory, but recomputes every window** ([A hard
window](../README.md#a-hard-window)). Polars'
`df.rolling("t", period="3h").agg(...)`, with an exponential weight and `c`
from the weights inside the window, agrees with the spec within 1e-12. Its
work grows with `n` rows times the rows of a window, where the spec's grows
with `n`. Measured, the spec ran 24 times as fast at a 74-row
window and 1,100 times at a 4,680-row one. So use the spec for a stream, a
state to save, or speed. The docstring's Notes, under *Against Polars*,
give how to weight the rows.

#### See also

To fit, classify or screen on the same moments:

- [`ewridge`](#ewridge--exponentially-weighted-ridge-regression), a
  regression on the same running co-moments.
- [`ew_class`](#ew_class--gaussian-classification-on-ew_cov-moments) and
  [`hmm`](#hmm--which-regime-are-we-in), which classify rows by Gaussians
  on these moments.
- [`marginal`](#marginal--every-pairs-moments-kept-in-the-state), the same
  arithmetic one (feature, target) pair at a time.
- [`bank.gram()`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.ModelBank.gram),
  to read the moments back from the state.

#### Notes

**Each row's statistics are pandas' at the row before.** Against pandas'
`ewm(halflife=...)` mean, variance, covariance and correlation, under
either `bias`, they agree within 1e-12 on columns near zero. Under a hard
window, the means and covariances agree within 1e-12 with numpy's weighted
moments of the rows inside it, and `pc0_var` agrees with
`numpy.linalg.eigh` of the same matrix within 1e-9 relative.

### `marginal` — every pair's moments, kept in the state

*API:* [`po.spec.marginal`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.marginal) — *Rust:* [`marginal.rs`](../crates/online-core/src/marginal.rs) — *Outputs:* [fields](OUTPUTS.md#marginal)

Each feature's exponentially weighted moments with each target, read back
as a table.

Use `marginal` to screen thousands of features against a few targets in one
pass. Each (feature, target) pair is its own two-column
[`ew_cov`](#ew_cov--exponentially-weighted-moments), so for `p` features
and `T` targets a row takes O(p·T) time, where one `ew_cov` over all the
columns would take O((p + T)²). Use `ew_cov` for the joint covariance of a
few columns, and
[`ewridge`](#ewridge--exponentially-weighted-ridge-regression) or
[`lasso`](#lasso--lasso-path-its-penalty-chosen-as-it-runs) for a joint fit
of the target on every feature.

Its parameters, their defaults and what each refuses are in its
[docstring](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.marginal).
Each row writes only `weight_sum` ([OUTPUTS.md](OUTPUTS.md#marginal)), and
`bank.marginal()` reads the pairs from the state as a table after the run.
[Its
docstring](https://hgilde.github.io/polars-online/polars_online.html#polars_online.ModelBank.marginal)
defines each column. The parameters every spec takes are in [What a spec
names](../README.md#what-a-spec-names).

#### The update

Each pair (feature `j`, target `t`) keeps its own `m_x`, `S_xx` and `S_xy`,
over its target's rows, beside the target's `m_y` and `S_yy`. Per target
`t`, on a row where `y_t` is present, with `W_t` the weight behind that
target before the row and `Q_t` the sum of its squared weights:

```
W'_t = λW_t + w        a = λW_t / W'_t        b = w / W'_t        Q'_t = λ²Q_t + w²
S'_yy = a·S_yy + a·b·(y_t − m_y)²             S'_xx = a·S_xx + a·b·(x_j − m_x)²
S'_xy = a·S_xy + a·b·(x_j − m_x)(y_t − m_y)    m' = m + b·(value − m)      for m_y and each m_x
n_kish = W_t² / Q_t
```

`value` is the row's `y_t` for `m_y` and its `x_j` for `m_x`, and every
deviation is from the means before the row. A null target decays that
target (`W_t` by `λ`, `Q_t` by `λ²`) and learns nothing for it, and a null
feature skips the row.

For unit weights a step `Δclock` apart, `n_kish` tends to
`(1 + λ)/(1 − λ)`, where `λ` is that step's decay. Under
`bin_stat="huber"`, each bin keeps a location `m` and `Ω`, the weight its
rows count for. With `c` the cut, `bin_huber_delta` (default 1.345) times
the target's robust scale `σ_t`, both read before the row, and `e = y − m`
the row's signed deviation from the bin's location:

```
ω = w·min(1, c/|e|)      Ω' = λΩ + ω      m' = m + (ω/Ω')·e
```

The location is the mean over the weights `ω`, so its fixed point is the
Huber M-location of the bin's rows. One row moves it by at most
`c·w/Ω' = c·w/(λΩ + ω)`, which under decay is up to `1/λ` times `c·w/Ω`.
The docstring's update gives the form `feature_moments="shared"` takes.

#### Example

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
value that size is what noise alone gives, which is why `t_stat` ranks
pairs and is no p-value.

#### Choosing its settings

Five settings extend the pairs, each off by default, and the first two add
columns to the table:

| setting | what it adds |
|---|---|
| `lags`, with `serial_rule` | the pair's moments at each lag: a count that allows for rows that resemble their neighbors, and lead/lag correlations |
| `bins` | the target's moments inside each bin of a feature: the relations a correlation cannot see |
| `window_size` | the pairs over a recent stretch |
| `feature_moments="shared"` | speed: one feature moment for every target, and the same table where every target is on every row |
| `shards` | speed: one wide spec on every thread, and the same numbers at any count |

When a group closes, `bank.closed_groups()` writes every pair as `pair_*`
columns: a list over the pairs, such as `pair_split_gain`, or a list of
lists for the columns that hold a list per pair, the `lag_corr_*` and
`bin_*` columns, such as `pair_lag_corr_xx` and `pair_bin_n`.

To rank the pairs by `t_stat`, correct the ranking with `lags` and
`serial_rule`, and read `split_gain` and `bin_stat="huber"`'s bins, see
[Screen features against a target](how-to/screen-features-against-a-target.md).

**Give a screen a `window_size` when a relation may flip sign**, because two
regimes of opposite sign average to nothing over a long history. The hard
window truncates every pair moment, so `corr`, `beta` and `t_stat` describe
the rows inside it ([A hard window](../README.md#a-hard-window)). A hard
window refuses `bins`, so bin the features in a second spec without one. To
use `lags` under a hard window, give `window_lags=True`, whose docstring
entry gives the memory its snapshots take.

**With several targets present on every row, give `feature_moments="shared"`
for speed.** Each feature then keeps one mean and variance, over every
learned row, in place of one per pair over its target's rows. At 20,000
pairs that runs 2.7 times as fast at ten targets, with a state under half
the size and the same table to the bit ([PERFORMANCE
§27](PERFORMANCE.md#27-marginals-shared-feature-moments-2026-09-29)).
Where a target is absent on some rows, `var_x` and the center of `cov` then
come from every row. That is a different estimator, sound only where the
absence says nothing about the feature, so otherwise keep the default. It
takes `lags`, and refuses a `window_size`.

**Give `shards="auto"` to spread one wide spec over the threads.** The bank
runs groups and specs in parallel ([Parallelism](../README.md#parallelism)),
so a wide `marginal` on one group is otherwise one thread's work.
`shards=10` splits its pairs into ten ranges of features, each run on a
thread of its own, a block of rows at a time. `"auto"` sizes the split to
the width and the bank's threads. It leaves whole a narrow spec, and a spec
under a hard window that snapshots every row, the default, whose block is a
single row. The numbers are the same to the bit at any count, so a saved
bank resumes under any count. On 14 threads ([PERFORMANCE
§25](PERFORMANCE.md#25-a-wide-marginal-split-across-the-pool-2026-09-25)):

| the spec | `"auto"` ran the bank |
|---|---|
| 10,000 features, nine targets, lags and bins | 4.9 times as fast |
| the moments of one target alone | 1.2 times as fast, since the bank's own work on each row does not split |

#### How it works

**Under `"huber"`, a bin's first row is judged against `m_t`, the target's
own location, the only location there is,** and winsorized at `m_t ± c`. A
row far from the bin still counts, at weight `c/|e|` of a row. A spike among
a target's first two rows, before it has a scale, counts in full, and leaves
the bins it lands in as a mean's start does.

**The first rows are refit once, `bin_refit_rows`, because one pass can be
captured.** Against statsmodels' M-location of each bin over the rows so far
(16 bins, spikes of about 20 of the target's scales, no decay), the worst
bin of the worst of 50 streams, in the target's scales:

| spikes | after 2,000 rows: one pass / refit | after 20,000 rows: one pass / refit |
|---|---|---|
| 5% | 0.36 / 0.005 | 0.03 / 0.001 |
| 12% | 1.29 / 0.015 | 0.23 / 0.004 |
| 20% | 3.5 / 0.14 | 1.0 / 0.057 |

**The one pass alone lands near the converged estimate of all the rows at
once, and the refit brings it closer.** The reference is statsmodels'
M-location of each bin (`estimate_location` with `HuberT(1.345)` at the
target's MAD), on 20,000 rows in 16 bins with no decay, with spikes of 50
on a share of the rows. Against it, the largest gap over ten streams of the
one pass alone (`bin_refit_rows=0`), in units of the target's scale:

| spikes | `"huber"`, one pass alone, against the converged estimate | the mean against it |
|---|---|---|
| none | 0.0006 | 0 |
| 1% | 0.0010 | 0.45 |
| 5% | 0.0016 | 1.51 |
| 12% | 0.030 | 2.95 |

With the refit of the first 500 rows, the worst of 50 streams, a harder test than the table's ten, was
7.7e-16, 3.0e-4, 1.2e-3 and 3.1e-3 of the scale at the same four shares.
The one pass's gap comes from its first rows, which are weighed against a
location and a scale that are still moving. On 2,000 rows, 125 to a bin, it was 0.013 at
1% and 0.37 at 5%, the largest from a spike on a stream's second row. The
location steps over `Ω`, the weight its rows count for, rather than `W`, the
weight they carry. From a spike `E₀` off at full weight `Ω₀`, while each row
is clipped the error decays as `E(n) = E₀·exp(−c·n/(Ω₀E₀))`. So it is within
one scale `σ` after about `(Ω₀E₀/c)·ln(E₀/σ)` rows. Simulated with no
decay, unit weights and `σ` fixed, a spike of 10, 30 or 100 scales was
within one scale after 27, 101 or 381 rows. Over `W` a row moves the
location back at most `c·w/W`, so `n` rows recover at most `c·ln n`, 13
scales in 20,000 rows. In the same simulation the spike of 30 scales was
still 17 scales off after 20,000 rows, and the one of 100 still 87.

Measured at 16 bins, `"huber"` ran at 0.75 to 0.9 times the mean's
throughput at 100 and 10,000 features and one and ten targets. The
saved state was a fifth to a third larger.

**Under `serial_rule="geometric"`, each series' autocorrelations are
fitted as `ρ(l) = φˡ`, and the bracket's sum is taken in closed form,
`2φ_xφ_y/(1 − φ_xφ_y)`,** so the kept lags need not be dense. `"truncated"`
suits dense lags, and `"bartlett"`, which weights lag `l` by
`1 − l/(L + 1)` for `L` the longest kept lag, suits dense lags whose long
ones are noisy.

**Under a hard window, the lag moments are kept about each series' first
value, which truncates exactly, and read about the hard window's means when
asked.** A correlation centered at the running mean instead carries the
history's mean into the hard window. On a hard window whose rows had moved
to another level, it read a lag correlation of 22, outside `[−1, 1]`.

#### See also

To read the pairs, or to fit the features jointly:

- [`bank.marginal()`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.ModelBank.marginal),
  the table of pairs and every column it holds.
- [`ew_cov`](#ew_cov--exponentially-weighted-moments), the joint
  covariance whose arithmetic each pair uses.
- [`huber`](#huber--robust-regression), whose weighting `bin_stat="huber"`
  gives each bin.
- [`lasso`](#lasso--lasso-path-its-penalty-chosen-as-it-runs), to choose
  the features of a joint fit.

#### Notes

**Each pair matches a two-column `ew_cov`.** Its means and variances equal those an
`ew_cov(bias=True)` over the two columns reports, to the bit. Its `cov` and
`corr` agree within a few roundings: the pair rounds each row's product as
`(a·b·dx)·dy`, while `ew_cov`, which keeps one triangle of its symmetric
matrix, rounds it as `(a·b·dy)·dx`. A pair's `corr` agrees with pandas'
`ewm(halflife=...).corr` within 1e-12 relative, and under a hard window its
moments agree with numpy's weighted moments of the rows inside it. The
docstring's Notes give the serial count's check on AR(1) series, the Huber
locations' against statsmodels, and the lags' under a hard window against
statsmodels' `acf`.

### `deco` — one correlation for the whole matrix

*API:* [`po.spec.deco`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.deco) — *Rust:* [`deco.rs`](../crates/online-core/src/deco.rs) — *Outputs:* [fields](OUTPUTS.md#deco)

Engle and Kelly's dynamic equicorrelation, one number for the whole
correlation matrix of the features.

Use `deco` (Engle & Kelly 2012) for one number that moves with the
correlation of many series. It replaces the `m(m−1)/2` free entries of the
correlation matrix of `m` series with their average, the
*equicorrelation*, and estimates that in O(m) a row, where keeping every
entry moving takes O(m²). Use
[`ew_cov`](#ew_cov--exponentially-weighted-moments) for each pair's
correlation, and `deco`'s own `blocks` for one number per block of features
and one per pair of blocks.

Its parameters, their defaults and what each refuses are in its
[docstring](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.deco),
and the fields it writes in [OUTPUTS.md](OUTPUTS.md#deco). The parameters
every spec takes are in [What a spec names](../README.md#what-a-spec-names).

#### The update

The row is standardized against the means and variances as they stood
before it, `ζ_i = (x_i − m_i)/√v_i`. With `S₁ = Σ ζ_i` and `S₂ = Σ ζ_i²`
over the `n` features that have a standardized value, the row's estimate
`u` is Engle and Kelly's Lemma 2.3, which lies in `[−1/(n − 1), 1]`.
Between two blocks `A` and `B` the estimate is `u_AB`. The reported level,
`rho`, then follows `u` under one of two dynamics, with `α` and `β` the
`"linear"` dynamic's two weights and `ρ̄` the `"ew"` level run beside it:

```
u    = (S₁² − S₂) / ((n − 1)·S₂)        = (mean of ζ_i·ζ_j over i ≠ j) / (mean of ζ_i²)
u_AB = S₁_A·S₁_B / √(n_A·n_B·S₂_A·S₂_B)

"ew":      W' = λW + w,  b = w/W'      ρ' = ρ + b·(u − ρ)                       ρ = u at W = 0
"linear":  ρ' = (1 − α − β)·ρ̄' + α·u + β·ρ                                      ρ' = u on the first row, ρ' = ρ at w = 0
```

The `"ew"` level is the exponentially weighted mean of `u`, decayed on the
model's clock. The docstring's Notes, under *The clock* and *Departures
from the paper*, give how the `"linear"` level treats a capped gap and a
row's weight, and why its intercept is written as
`(1 - alpha - beta) * rho_bar`.

#### Example

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
reads low, at about two-thirds of the truth here.

#### Choosing its settings

To read `u` and `rho` as a signal, per block with `blocks`, and to leave a
constant column out of `features`, see [Track the average
correlation](how-to/track-the-average-correlation.md).

#### How it works

**`u` is biased downward, as Engle and Kelly note:** `E[u]` is about 0.20
for a true 0.30 at six series. `rho`, an average of `u`, carries the same
bias, and
[docs/REGIMES.md §7](REGIMES.md#7-deco-an-equicorrelation-that-moves)
measures it reading near two-thirds of the truth.

#### See also

For each pair's correlation, or a test of a change in it:

- [`ew_cov`](#ew_cov--exponentially-weighted-moments), every pair's
  correlation, at O(m²) a row.
- [`corrchange`](#corrchange--has-the-correlation-structure-changed), a test
  for a change in the correlation structure.

#### Notes

**Under `"ew"`, `rho` is exactly what an `ew_cov(stats=["mean"])` on the
`u` column, over the rows that have one, would report, to the bit, wherever both report.** The `"linear"` level at
`alpha` and `beta` of 0 is the `"ew"` level, to the bit, and `loglik` is
held to the dense Gaussian density of the correlation matrix the levels
describe. The docstring's Notes, under *The estimate's downward bias*, give
the paper's alternative estimate, which this model does not compute.

### `rcov` — a block's realized covariance, robust to noise

*API:* [`po.spec.rcov`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.rcov) — *Rust:* [`rcov.rs`](../crates/online-core/src/rcov.rs) — *Outputs:* [fields](OUTPUTS.md#rcov)

A block's realized covariance of the features, robust to microstructure
noise.

Use `rcov` for the covariance of returns over each block of a tick stream,
such as a trading day, where noise in each price biases a plain realized
covariance. For series not observed together, use its `"preavg"` kind. Use [`ew_cov`](#ew_cov--exponentially-weighted-moments)
for running moments that decay and are read on every row.

Its parameters, their defaults and what each refuses are in its
[docstring](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.rcov).
It requires `group`, naming the block column, and `group_close`, and
refuses a `half_life` or `lam`, since a block's close ends its memory. Each
row writes only `weight_sum` ([OUTPUTS.md](OUTPUTS.md#rcov)), and each
block's estimate is that block's row of `bank.closed_groups()` when it
closes. The parameters every spec takes are in [What a spec
names](../README.md#what-a-spec-names).

#### The update

Each row's returns `x` enter the block's sums over lags, and at the close
the estimator `kind` names reads them. In the kernel, `Gamma_h` sums the
products of returns `h` rows apart, up to a bandwidth `H`, with `k` the
Parzen weight. `xi² = omega2 / iv_sparse` is the noise variance over the
sparse integrated variance. `"preavg"` pre-averages the returns over `k_n`
rows with `g`, its window scaled by `theta` (default 1), and `rcov_n`
counts the pre-averaged terms. `psi1` and `psi2` are its finite-sample
constants (1 and 1/12 in the limit). Over a block of `n` returns, with
`block_rows` the block's expected length:

```
"plain":    sum_j x_j x_jᵀ
"kernel":   K = sum_{h=−H}^{H} k(h/(H+1))·Gamma_h      Gamma_h = sum_j x_j x_{j−h}ᵀ,   Gamma_{−h} = Gamma_hᵀ
            H = ceil(c*·xi^(4/5)·n^(3/5)),  c* = 3.5134, unless bandwidth= fixes it
"preavg":   Ybar_i = sum_{j=1}^{k_n−1} g(j/k_n)·x_{i+j}      g(x) = min(x, 1 − x)
            MRC = n / rcov_n / (psi2·k_n) · sum_i Ybar_i Ybar_iᵀ
            psd=True:   MRC                                  k_n = ceil(theta * block_rows^0.6)
            psd=False:  (MRC − b·sum_j x_j x_jᵀ) / (1 − b)    k_n = floor(theta * sqrt(block_rows)),  b = psi1 / (2 psi2 k_n^2) → 6 / k_n^2
```

**The estimate reads no future row.** The kernel's jittered end points
average observations already in the state at close, and a product enters
`Gamma_h` only once both its legs are final.

#### Example

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
    corr = closed["rcorr"].list.get(1)                         # the upper triangle, row by row: [1, r, 1]
    print(f"{kind}: mean correlation {corr.mean():.2f} over {closed.height} blocks (truth 0.60)")
```

```text
plain: mean correlation 0.29 over 19 blocks (truth 0.60)
kernel: mean correlation 0.59 over 19 blocks (truth 0.60)
```

The last block never closes, because no later block arrives to close it
under `group_close="monotone"`, so 19 of the 20 report.

#### Choosing its settings

To choose `kind` for the returns you have, and to read each block's estimate, see
[Estimate a block's covariance from
ticks](how-to/estimate-a-blocks-covariance-from-ticks.md).

#### See also

To put the prices on one grid before taking returns, or to read each block's estimate:

- [`ew_cov`](#ew_cov--exponentially-weighted-moments), running moments that
  decay, read before each row.
- [Align series that tick at their own
  times](how-to/align-series-that-tick-at-their-own-times.md), to put the
  prices of several series on one grid with `po.stream.refresh_time` before
  taking returns.
- [`bank.closed_groups()`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.ModelBank.closed_groups),
  which returns each block's estimate when the block closes.

#### Notes

**`"plain"` equals `n` times an `ew_cov(lam=1)`'s uncentered second moment
at close, to the bit.** It is the cross-check, and the reference the other
two are measured against. The jittered kernel agrees with hfhd's `krvm` on
the jittered prices within 1e-13 of the estimate's largest entry, over
three features, cross terms included. The default `"preavg"`
(`psd=True`) agrees with hfhd's `mrc(bias_correction=False)` within 1e-13
of the estimate's largest entry, after two mappings, with `block_rows` set to match. The test
gives hfhd's prices a leading copy of the first, since hfhd's
pre-averaging starts one return later. hfhd then counts `n + 1` returns,
`n` the block's, so the test sets `block_rows` to `n + 1`, and both take
the same `k_n`. The test
then replaces hfhd's scale `(n + 1)/(n − k_n + 3)` with this library's
`n/(n − k_n + 2)`. [docs/REGIMES.md
§8](REGIMES.md#8-rcov-three-estimators-against-each-blocks-truth) measures
each kind against each block's truth.

### `audit` — what a stream's columns hold

*API:* [`po.spec.audit`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.audit) — *Rust:* [`audit.rs`](../crates/online-core/src/audit.rs) — *Outputs:* [fields](OUTPUTS.md#audit)

A count of what a stream's columns hold, read in one pass by a spec that
fits nothing and predicts nothing.

Use `audit` before choosing a model, to find out what is wrong with the
data, such as a sentinel value, a frozen feed or a heavy tail. Run it alone,
or beside the models in the same pass, and
[`bank.check()`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.ModelBank.check)
reads its counts and says what they mean. Use `bank.describe()` for a few
statistics of the columns a model has already been fed.

Its parameters, their defaults and what each refuses are in its
[docstring](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.audit).
Each row writes only `weight_sum` ([OUTPUTS.md](OUTPUTS.md#audit)), and
`bank.audit()` reads the counts from the state as a table after the run.
[Its
docstring](https://hgilde.github.io/polars-online/polars_online.html#polars_online.ModelBank.audit)
defines each column. The parameters every spec takes are in [What a spec
names](../README.md#what-a-spec-names).

#### The update

Per column, each row's value `x` is first classified. A null, a NaN,
`+inf`, `-inf` and a finite value past `1e100` in magnitude each add one to
a count of their own, and anything else is usable. A usable value enters
the moments by Terriberry's update of the central sums `M2`, `M3` and `M4`,
with `n` the usable values before it. Its value's counter adds one, and
past `distinct_cap` distinct values the counters run as a Misra-Gries
summary. Nothing decays, and a row's weight plays no part:

```
n' = n + 1      d = x − mean      e = d / n'      t = d·e·n
mean' = mean + e
M4' = M4 + t·e²·(n'² − 3n' + 3) + 6e²·M2 − 4e·M3
M3' = M3 + t·e·(n' − 2) − 3e·M2
M2' = M2 + t
```

#### Example

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

#### Choosing its settings

**Give `pairs=True` to find a duplicated column.** The audit then keeps each
pair of columns' correlation and the rows on which the two are equal, and
`bank.check()` reads them for its `duplicate` finding.

**Give `pooled=True` to `bank.audit()` to read every group as one stream.**
Groups merge into one audit: the counts exactly, the moments to rounding,
and the repeats past `distinct_cap` and the median within the bounds the
docstring gives. This code uses `df` from [Example data](../README.md#example-data):

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

#### See also

To judge the counts, and to fix what they find:

- [`bank.check()`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.ModelBank.check),
  what the counts say is wrong.
- [DATA-QUALITY.md](DATA-QUALITY.md), each finding, what it does to each
  model and how to fix it with Polars.

#### Notes

**Its statistics are the libraries' own, from the raw values.** The mean
agrees with numpy's within 1e-14 relative, or 1e-12 where it is near 0, and the standard deviation with
`np.std(ddof=1)` within 1e-10. The skew and excess kurtosis agree with
scipy's biased estimators within 1e-8 relative, and so does `unit_root_t`
with statsmodels' `adfuller`, with a constant and no lags. `autocorr` agrees with numpy's
correlation of a column with its own lag, and the median and median
absolute deviation are numpy's while the counts are exact. Past the cap,
they are within about 1% of the rows in rank. Each pair's correlation
agrees with `np.corrcoef` over the rows where both columns are usable,
within 1e-13 relative, in each group and in the pooled merge.

## Clustering and classification

These models put a label on each row: a cluster found in the features alone,
or a class learned from a label column. Every label, distance and probability
is read before the row is learned from, so it is out-of-sample like every
prediction here.

### `kmeans` — exponentially weighted k-means

*API:* [`po.spec.kmeans`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.kmeans) — *Rust:* [`cluster/kmeans.rs`](../crates/online-core/src/cluster/kmeans.rs) — *Outputs:* [fields](OUTPUTS.md#kmeans)

Exponentially weighted k-means over the feature columns.

Use `kmeans` when you know the number of clusters and they are round: it
labels each row with the nearest of `n_clusters` centers, before the row is
learned. Use [`micro`](#micro--density-based-clustering-any-shape) for
clusters of any shape or of a number you do not know, and
[`hmm`](#hmm--which-regime-are-we-in) where a row's cluster depends on the
one before it.

Its parameters, their defaults and what each refuses are in its
[docstring](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.kmeans),
and the fields it writes in [OUTPUTS.md](OUTPUTS.md#kmeans). The
parameters every spec takes are in [What a spec
names](../README.md#what-a-spec-names).

#### The update

Each center is the exponentially weighted mean of the rows assigned to it
(`ew_cov`'s mean recursion, per cluster). With `x` the row's features, `c_j`
center `j` and `n_j` its decayed weight:

```
j*    = argmin_j ‖x − c_j‖²          distances in units of each feature's EW sd (standardize=False: raw units)
n'_j  = λn_j                          every other center
n'_j* = λn_j* + w                     c'_j* = c_j* + (w/n'_j*)(x − c_j*)
```

Three rules keep the centers on the data: seeding, a split-merge move that
finds a cluster born after seeding, and a dead rule that re-places a center
whose cluster vanished. A row far outside its cluster moves neither `n_j`
nor `c_j`, and waits for the split-merge move. The docstring's Notes give
each rule, under *Seeding*, *Far rows, split and merge* and *Dead centers*.

#### Example

**The split-merge move gives a cluster born after seeding a center of its
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
centers = np.array([[0.0, 0.0], [6.0, 0.0], [0.0, 6.0], [6.0, 6.0]])
blob = rng.integers(0, 3, n)                                  # three clusters ...
late = np.arange(n) >= 4000
blob[late] = rng.integers(0, 4, late.sum())                   # ... and a fourth born at row 4000
xy = centers[blob] + rng.standard_normal((n, 2))
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
about 170 rows each. Once it appears, the split-merge move frees cluster 3
and places it on the new cluster, and each of the four holds about a quarter
of the rows.

#### Choosing its settings

To choose `init` and `dead_frac`, see [Keep `kmeans` centers on the
data](how-to/keep-kmeans-centers-on-the-data.md).

#### How it works

**Seeding waits for `warm_rows` rows, places the centers, and replays the
rows.** [Seeding](explanation/warm-up-and-readiness.md#seeding-a-model-that-places-its-starting-values)
explains why the default takes the best of ten starts, measures it on
Fränti's benchmark sets, and gives what it costs each group.

#### See also

When the clusters are not round, not fixed in number, or labeled:

- [`micro`](#micro--density-based-clustering-any-shape), density
  clustering that finds the number of clusters and flags the rows in none.
- [`hmm`](#hmm--which-regime-are-we-in), Gaussian states whose next state
  depends on the last, seeded by `kmeans`' `init`.
- [`ew_class`](#ew_class--gaussian-classification-on-ew_cov-moments),
  Gaussian classes, one for each value of the column `label=` names.

#### Notes

**The default seeding places the centers scikit-learn's `KMeans` places.**
`init="lloyd"` is held to scikit-learn's `KMeans(n_init=10)` centers on the
same weighted rows, and `init="k-means++"` to scikit-learn's
`kmeans_plusplus` in distribution. On hundreds of thousands of rows, the
labels are held to the clusters that generated them and to scikit-learn's
`KMeans` on the same rows.

### `micro` — density-based clustering, any shape

*API:* [`po.spec.micro`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.micro) — *Rust:* [`cluster/micro.rs`](../crates/online-core/src/cluster/micro.rs) — *Outputs:* [fields](OUTPUTS.md#micro)

Density-based clustering over the feature columns, by small decayed
summaries linked into clusters.

Use `micro` when the clusters may have any shape, such as moons or rings,
or their number is not known. It flags the rows that belong to no cluster,
and follows clusters that appear and vanish. Use
[`kmeans`](#kmeans--exponentially-weighted-k-means) for a known number of
round clusters.

Its parameters, their defaults and what each refuses are in its
[docstring](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.micro),
and the fields it writes in [OUTPUTS.md](OUTPUTS.md#micro). [Choose
`micro`'s `eps`](how-to/choose-micro-eps.md) says how to choose `eps`, and
how a wrong value shows in the output. The parameters every spec takes are
in [What a spec names](../README.md#what-a-spec-names).

#### The update

It builds on DenStream's micro-clusters, with a linking step over them.
Each micro-cluster is a *summary*: a decayed weight `n`, a center `c`, and
a radius `ν`, the exponentially weighted root-mean-square distance of its
rows from the center. A summary is *established* once its weight reaches
`beta_mu` rows of the stream's mean row weight `w̄`. Distances are in units
of each feature's exponentially weighted standard deviation, floored by
`scale_floor` as [`kmeans`](#kmeans--exponentially-weighted-k-means)' are.
With `p` features, a row joins a summary only if the summary's radius stays
within `eps·√p`:

```
r²(j, v) = a ν²_j + a b ‖x − c_j‖²      a = n_j/(n_j + v),  b = v/(n_j + v)
n_j  ← λ n_j                                             every summary
j*   = the nearest established summary, if r²(j, w̄) ≤ eps² p,
       else the nearest summary not yet established, if r²(j, w̄) ≤ eps² p,
       else a new summary at x
ν²_j* ← min(r²(j*, w), eps² p)     c_j* ← c_j* + (w/(n_j* + w))(x − c_j*)     n_j* ← n_j* + w
```

Every `prune_every` units of the clock, or every `max_rows_between_prunes`
learned rows, the model drops the light summaries and links the established
ones: centers within `L` of each other share a label. The docstring's Notes
give the rules, under *Pruning*, *The prune schedule* and *The link*.

#### Example

**It finds two rings that no center-based rule can split, and flags
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
# How many clusters, and how each ring and the noise are labeled.
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

#### How it works

**A summary whose rows stop lingers `half_life · log2(n / (beta_mu · w̄))`
of clock, with `n` the weight it had.** A cluster's label is the smallest id
among its linked summaries, so it lasts as long as that summary does.

#### See also

When the number of clusters is known, or a row's cluster depends on the last:

- [`kmeans`](#kmeans--exponentially-weighted-k-means), a fixed number of
  clusters whose centers follow the stream.
- [`hmm`](#hmm--which-regime-are-we-in), Gaussian states whose next state
  depends on the last.

#### Notes

**It recovers shapes that `kmeans` cannot.** Measured at 20,000 rows and a
half-life of 3,000, moons, rings and five Gaussians in twenty dimensions all
score an adjusted Rand index (ARI) of 1.000 against the truth. Their `eps`
were 0.07, 0.1 and 0.3, and `kmeans` cannot follow the first two.

**It flags scattered rows and leaves the clusters' rows alone.** On four
clusters with one row in twenty scattered uniformly over a box three times
their extent (40,000 rows, a half-life of 2,000, `eps=0.07`), 94% of the
scattered rows were flagged `outlier`, and 0.3% of the rest. The recipe's
scattered rows read 61% because many land near a ring and take its label.

**It labels a cluster born mid-stream within tens of rows.** On four
clusters, one of which stops halfway through as a fifth is born (a
half-life of 1,000), the new cluster had a label 31 rows in.

### `ew_class` — Gaussian classification on `ew_cov` moments

*API:* [`po.spec.ew_class`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.ew_class) — *Rust:* [`ewclass.rs`](../crates/online-core/src/ewclass.rs) — *Outputs:* [fields](OUTPUTS.md#ew_class)

A Gaussian classifier on exponentially weighted moments, one set per class.

Use `ew_class` to predict a row's class from a column of classes, named in
`label` with its values listed in `classes`. It fits quadratic discriminant
analysis, linear discriminant analysis or Gaussian naive Bayes. Use
[`hmm`](#hmm--which-regime-are-we-in) where no column gives the class and
the model must infer a hidden regime, and
[`kmeans`](#kmeans--exponentially-weighted-k-means) for clusters with no
classes at all.

Its parameters, their defaults and what each refuses are in its
[docstring](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.ew_class),
and the fields it writes in [OUTPUTS.md](OUTPUTS.md#ew_class). It takes
`label` in place of `targets`. The parameters every spec takes are in
[What a spec names](../README.md#what-a-spec-names).

#### The update

Each class `c` keeps one [`ew_cov`](#ew_cov--exponentially-weighted-moments)
state: a weight `n_c`, a mean `μ_c` and a centered covariance `C_c`. `C_c`
is in the population form, with no `ddof` correction, as `ew_cov`'s is under
`bias=True`. `π_c` is a class's share of the weight, and `ω_c` the ridge
`precision_prior`, scaled by `s_c`, which lets a class be predicted from its
first row and fades as `ew_cov`'s prior does. A row is predicted by Bayes'
rule over Gaussian classes, then learned by its own class:

```
π_c = n_c / Σ n         ω_c = precision_prior · s_c
M_c = C_c + ω_c I  ("full")      M = Σ π_c M_c  ("tied")      diag(C_c) + ω_c I  ("diag")
ℓ_c = ln π_c − ½ ln det M_c − ½ (x − μ_c)ᵀ M_c⁻¹ (x − μ_c)
p_c = exp(ℓ_c − max ℓ) / Σ exp(ℓ − max ℓ)                  class = argmax ℓ
n_c ← λ n_c + w·[y = c]        μ_c, C_c ← weighted Welford on the row's own class
s_c ← s_c · λn_c / (λn_c + w)  on the row's own class, and 1 on its first row
```

#### Example

**It predicts a row's class, out-of-sample, about as well as the Bayes rule
that knows the true distributions.** This code makes three Gaussian classes
with their own spreads, and scores the last 5,000 rows, each predicted
before its own class is learned:

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

# Class the last rows, each before its own class is learned.
cl = po.spec.ew_class("cl", features=["x", "y"], label="kind", classes=["a", "b", "c"],
                      precision_prior=0.1, half_life=float("inf"))
scored = pl.scan_parquet("classes.parquet").online.fit_predict([cl]).collect().unnest("cl").tail(5000)
print(f"classed right, out-of-sample: {(scored['class'] == scored['kind']).mean():.3f}")
# The Bayes rule that knows the true distributions, on the same rows.
z = (xy[:, None, :] - means[None]) / spread[None]             # the Bayes rule, from the true means and spreads
best = np.argmax(-0.5 * (z**2).sum(axis=2) - np.log(spread).sum(axis=1), axis=1)
print(f"the Bayes rule, on the same rows: {(best == label)[-5000:].mean():.3f}")
```

```text
classed right, out-of-sample: 0.857
the Bayes rule, on the same rows: 0.860
```

On the last 5,000 rows the model puts 85.7% of the rows in their true
class, against 86.0% for the Bayes rule built from the true means and
spreads.

#### Choosing its settings

To choose `covariance_type`, and `window_size` and `embargo` when the
classes move or arrive late, see [Choose a covariance
type](how-to/choose-a-covariance-type.md).

#### See also

For the moments behind each class, or for classes no column names:

- [`ew_cov`](#ew_cov--exponentially-weighted-moments), the moments each
  class keeps, for one set of rows.
- [`hmm`](#hmm--which-regime-are-we-in), the same Gaussian densities for a
  regime that no column names.
- [`kmeans`](#kmeans--exponentially-weighted-k-means), for clusters with no
  classes.

#### Notes

**It reaches the Bayes rate and is calibrated.** The test suite runs it on
200,000 rows of three Gaussian classes in six dimensions, with shares 0.5,
0.3 and 0.2 and each its own covariance. On the second half its accuracy is
within 0.01 of the Bayes rate, and each band of its probabilities is within 0.03 of the
share of rows it classes right.

## Sequential tests and regimes

These models predict no target. Each reads its columns, or a target's
sign, and answers, as the rows arrive, a question about the stream: whether
something holds, whether and when it changed, or which regime it is in now.

### `seqtest` — a sequential test of a sign, by betting

*API:* [`po.spec.seqtest`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.seqtest) — *Rust:* [`seqtest.rs`](../crates/online-core/src/seqtest.rs) — *Outputs:* [fields](OUTPUTS.md#seqtest)

A sequential test of a target's sign by betting, whose evidence can be read
at any row.

Use `seqtest` to read evidence as it accumulates and act the first time it
is enough: whether a column tends to be positive, or whether one spec
predicts closer than another. A p-value cannot be read that way, because
checking it repeatedly inflates its error rate, but the *e-process* this
model keeps can be read at any row. Use
[`po.eval.seqtest`](https://hgilde.github.io/polars-online/eval.html#polars_online.eval.seqtest)
for the same test over a frame in memory, and
[`po.eval.compare_specs`](https://hgilde.github.io/polars-online/eval.html#polars_online.eval.compare_specs)
for several specs' scores in one table.

Its parameters, their defaults and what each refuses are in its
[docstring](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.seqtest),
and the fields it writes in [OUTPUTS.md](OUTPUTS.md#seqtest). The
parameters every spec takes are in [What a spec
names](../README.md#what-a-spec-names), apart from `weight`, `half_life`
and `lam`, which it refuses. Each row counts as one trial, so to skip a
row, null its target.

#### The update

Per target it keeps the wealth of two gamblers, one betting that the next
sign is positive and one that it is negative. Each stakes the
Krichevsky–Trofimov fraction set by the counts so far, and stakes nothing
unless the counts favor its side. A zero or null target bets nothing and
counts nothing:

```
s = sign(y)                    n⁺, n⁻: the signs counted before this row,  n = n⁺ + n⁻
b⁺ = max(0, (n⁺ − n⁻) / (n + 1))          b⁻ = max(0, (n⁻ − n⁺) / (n + 1))     the stakes
ln E⁺ ← ln E⁺ + ln(1 + b⁺ s)              ln E⁻ ← ln E⁻ + ln(1 − b⁻ s)
```

#### Example

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

#### Choosing its settings

To read its evidence and reject at a level that holds however often you
look, see [Read a sequential test](how-to/read-a-sequential-test.md). To
test which of two specs predicts closer, put both in the bank and name
them in `a` and `b` ([Compare two
models](how-to/compare-two-models.md#read-the-comparison-on-every-row)).

#### See also

For the same test outside a bank, or a comparison of many specs at once:

- [`po.eval.seqtest`](https://hgilde.github.io/polars-online/eval.html#polars_online.eval.seqtest),
  the same computation in Polars expressions over a frame in memory.
- [`po.eval.compare_specs`](https://hgilde.github.io/polars-online/eval.html#polars_online.eval.compare_specs),
  several specs' scores in one table.

#### Notes

**Where the `max(0, …)` in a stake never binds, the wealth has the closed
form `2ⁿ B(n⁺+½, n⁻+½) / π`**, the Beta(½, ½) mixture, and a test holds the
bank to it. `po.eval.seqtest`, the same computation in Polars expressions,
is held to the bank to the bit.

### `corrchange` — has the correlation structure changed?

*API:* [`po.spec.corrchange`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.corrchange) — *Rust:* [`corrchange.rs`](../crates/online-core/src/corrchange.rs) — *Outputs:* [fields](OUTPUTS.md#corrchange)

Tests of whether the correlations between the feature columns have changed,
by three kinds of test.

Use `corrchange` to test whether the correlations among the columns you
name have changed, with `kind` choosing the test. Use
[`bocpd`](#bocpd--how-long-has-this-regime-lasted) for the age of the
current regime of the features' levels and spreads, and
[`deco`](#deco--one-correlation-for-the-whole-matrix) to follow an
equicorrelation that moves.

Its parameters, their defaults and what each refuses are in its
[docstring](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.corrchange),
and the fields it writes in [OUTPUTS.md](OUTPUTS.md#corrchange). The
parameters every spec takes are in [What a spec
names](../README.md#what-a-spec-names), apart from `half_life` and `lam`,
which it takes only under `scalar=True`.

#### The update

Each kind answers its own question:

| `kind` | the question | reports | against | a change is found |
|---|---|---|---|---|
| `"monitor"` | was the correlation constant over a span? | on a span's last row | the span's own correlation | at the span's end, up to `window_rows` rows late, and reliably only where it falls well inside a span |
| `"sequential"` | has it left the level a stable history set? | on each row tested against the history, from the second on | a history of `window_rows` rows | as soon as it crosses a boundary |
| `"window"` | how big is the change? | on every row once both windows are full, `2·window_rows` rows in | the window before | as soon as the two windows differ |

Under `"monitor"` and `"sequential"`, the statistic is the largest over the
pairs, and the level `alpha` (0.05 by default) is spread over the pairs as
`alpha / npairs` unless `alpha_adjust="none"`.

`kind="monitor"` is the closed-sample constancy test of Wied, Krämer and
Dehling (2012). With `m = window_rows`, `ρ̂_j` the correlation of the span's
first `j` rows, and `D̂` the delta-method long-run standard deviation of
`ρ̂`, the last row of a span reports, per pair:

```
Q = max_{2≤j≤m} (j/√m)·|ρ̂_j − ρ̂_m| / D̂
```

`kind="sequential"` is the monitoring procedure of Wied and Galeano (2013).
A cycle is `window_rows` rows of history, taken as stable, then up to
`monitor_rows` rows, each tested against the history as it arrives. A flag,
or the cycle's last monitored row, ends the cycle, and the row after it
starts a new history. From the history it reads each pair's correlation
`ρ̂_h` and its `D̂`. The `k`-th monitored row then reports

```
V_k  = (k/√m)·(ρ̂_k − ρ̂_h) / D̂           m = window_rows, ρ̂_k over the k monitored rows
stat = max over pairs of |V_k| / w(k/m)
w(b) = (1 + b)·(b/(1 + b))^γ               γ = boundary_gamma, 0 ≤ γ ≤ 0.49
```

and flags where `stat` passes `crit`.

`kind="window"` reports `‖vech(R_pre − R_post)‖₁` (under `norm="linf"`, its
largest absolute entry instead) over two adjacent windows of `window_rows` rows each. `R_pre` and `R_post` are the windows'
correlation matrices, and `vech` stacks their upper triangles into one list.
It flags against a permutation quantile, or against `crit` where one is
given.

The docstring's Notes give each critical value, `D̂` and the dating of a
change, under *The long-run deviation*, *The monitor test*, *The sequential
test* and *The window test*.

#### Example

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

#### Choosing its settings

To catch an early change sooner with `boundary_gamma`, the `γ` of `w(b)`,
and to count a run of the window kind's flags as one change, see [Detect a
break](how-to/detect-a-break.md#watch-the-features-correlations-with-corrchange).

#### How it works

**The sequential kind's critical value is solved, not simulated.** Where
the paper simulates its quantile `q`, above `γ = 0`, this library solves it
as a diffusion with an absorbing boundary
([`boundary.rs`](../crates/online-core/src/boundary.rs)). The paper allows
`γ` up to 1/2, but the solve's work grows as `1/(1/2 − γ)`: seconds at 0.49
and hours just under 1/2. So `boundary_gamma` above 0.49 is refused.

**A window's report takes the same work whatever the window's length.** Each window's
correlations are merged from summaries of blocks of rows: a block's count,
mean and centered co-moments, kept once and combined by the pairwise update
rule. So for `p` features a report reads only the rows at each window's
ends, about `p/2 + 3` of them, plus a handful of merges. A row that leaves a
window is never subtracted, so nothing drifts. Each block keeps its mean as
an offset from one of its own rows, so a level of 1e8 under a spread of 1 costs no accuracy: the
statistic is within a few units of rounding of the exact one.

**Its redraws are held for a window's length.** A redraw shuffles the
`2 * window_rows` pooled rows `n_perm` times, so `permute_every_rows`
defaults to `window_rows`. The redraws then cost `n_perm * p**2` a row
whatever the window. Each value is held for `permute_every_rows` reports, so
a report is compared with draws from windows up to
`permute_every_rows − 1` rows older than its own.

#### See also

For the age of a regime, or a correlation that moves rather than breaks:

- [`bocpd`](#bocpd--how-long-has-this-regime-lasted), a posterior over the
  age of the current regime.
- [`deco`](#deco--one-correlation-for-the-whole-matrix), an
  equicorrelation that moves with the stream.
- [docs/REGIMES.md](REGIMES.md), the size and power of these tests,
  measured against their papers.

#### Notes

**The monitor's size and power are close to Wied, Krämer and Dehling's
tables,** on the `t_5` design of their own paper ([docs/REGIMES.md
§2–3](REGIMES.md#2-is-the-monitor-the-size-its-paper-says)). The sequential
detector's size is within two standard errors of Wied and Galeano's Table 2
in every cell ([docs/REGIMES.md
§9](REGIMES.md#9-the-sequential-detector-against-its-paper)). Its critical
values are within 0.03 of their Table 1. Over 200 streams of independent
rows, the window kind flagged 4.8% to 5.2% of non-overlapping spans at
`alpha` 0.05, as the docstring's Notes give under *The window test*.

### `bocpd` — how long has this regime lasted?

*API:* [`po.spec.bocpd`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.bocpd) — *Rust:* [`bocpd.rs`](../crates/online-core/src/bocpd.rs) — *Outputs:* [fields](OUTPUTS.md#bocpd)

Bayesian online changepoint detection, a posterior over how long the current
regime has lasted.

Use `bocpd` (Adams & MacKay 2007) to know how long the current regime has
lasted. It keeps a probability distribution over the **run length**, the
rows since the last break, so it can say "we are forty rows into a regime",
not only "something broke". Use
[`corrchange`](#corrchange--has-the-correlation-structure-changed) for a
test of a change in correlation, and a regression's `emit_breaks` for a
break in its coefficients.

Its parameters, their defaults and what each refuses are in its
[docstring](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.bocpd),
and the fields it writes in [OUTPUTS.md](OUTPUTS.md#bocpd). The parameters
every spec takes are in [What a spec names](../README.md#what-a-spec-names),
apart from `half_life` and `lam`, which it refuses: the run-length posterior
does the forgetting, and `hazard`, the expected run length, sets how fast.

#### The update

With `H = 1/hazard` the chance of a break after each row, and `π_r` the
posterior predictive of run length `r` for this row, the model runs their
Algorithm 1:

```
growth:      P(r_t = r+1, x_{1:t}) = P(r_{t−1} = r, x_{1:t−1})·π_r·(1 − H)
changepoint: P(r_t = 0,   x_{1:t}) = Σ_r P(r_{t−1} = r, x_{1:t−1})·π_r·H
```

Under a duration `hazard`, such as `"1h"`, the duration is `τ`, the
expected time between changepoints. The step of `d` into a row then carries
the chance `h = 1 − exp(−d/τ)` that a break fell inside it. That
chance belongs before the row, so `h` of every run's mass moves to the
empty run before the row is read. Algorithm 1 then runs with
`H = 0`, since no time has passed since the row.

#### Example

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
to lift `p_change`, which is why the answer is read from `run_mode`.

#### Choosing its settings

To choose `hazard`, the priors and `robust_beta`, see [Choose a
changepoint hazard](how-to/choose-a-changepoint-hazard.md). To read the
regime's age from `run_mode` and use `p_change` only as an alarm, see
[Detect a break](how-to/detect-a-break.md#date-a-regime-with-bocpd).

#### See also

For a test of a change in correlation, or a fixed set of regimes:

- [`corrchange`](#corrchange--has-the-correlation-structure-changed),
  tests for a change in the correlations among the features.
- [`hmm`](#hmm--which-regime-are-we-in), a fixed number of regimes the
  stream moves between.
- [docs/REGIMES.md §5](REGIMES.md#5-two-changepoint-detectors-on-the-same-break),
  `bocpd` beside `corrchange` on the same break.

#### Notes

**The pruned posterior is held to the whole one.** At the defaults,
`prune_below=1e-6` and `max_run=10,000`, the run-length posterior is held
to the one the `bayesian_changepoint_detection` package computes with
nothing pruned. Its `p_change` is held to the package's within a bound set
by the mass pruned.

**It flags a step up in the spread at once, and dates a step down by the
run length.** Adams and MacKay's own finance example (a = 1, b = 1e-4,
hazard = 250) is `degrees_of_freedom_prior=2.0`, `covariance_prior=[2e-4]`
and `hazard=250.0`. Under it, with `max_run=600`, a tenfold step up in the
spread lifts `p_change` above 0.5 within ten rows, and `run_mode` dates it
to its row. The step back down is dated within one row.

### `hmm` — which regime are we in

*API:* [`po.spec.hmm`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.hmm) — *Rust:* [`hmm.rs`](../crates/online-core/src/hmm.rs) — *Outputs:* [fields](OUTPUTS.md#hmm)

A Gaussian hidden Markov model of the features, filtered one row at a time.

Use `hmm` to ask "which regime are we in", not only "which regime does this
row look like": a transition matrix carries what the rows before have shown
into each row's answer.
[`ew_class`](#ew_class--gaussian-classification-on-ew_cov-moments)
classifies a row against *labeled* Gaussians, so use it when a column names
each row's class. An `hmm` does the same arithmetic without labels, and its
regime is a *hidden state*.

Its parameters, their defaults and what each refuses are in its
[docstring](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.hmm),
and the fields it writes in [OUTPUTS.md](OUTPUTS.md#hmm). The parameters
every spec takes are in [What a spec names](../README.md#what-a-spec-names).

#### The update

Here a hidden state means one of the `n_components` regimes, not the bank's
saved state. The model runs Hamilton's filter one row at a time, from the
filtered `p` the previous row left. Every output but `coef` is read before
the row is learned. With `μ_l` and `Σ_l` state `l`'s mean and covariance,
`ω_l` its `precision_prior` ridge as it has decayed, and `Π` the transition
matrix:

```
p1_l   = Σ_k p_k·Π_kl                      the predicted state
f_l    = N(x | μ_l, Σ_l + ω_l·I)           the state's density
loglik = ln Σ_l p1_l·f_l                   the row's log-likelihood
p_l   ← p1_l·f_l / Σ                       the filtered state
```

A row splits across the states at weight `w·p_l`, which sums to `w`, so
`weight_sum` follows the same recursion as in every model. The transition
matrix is learned from the filtered joint of consecutive states. `N_kl` are
its decayed counts. `τ_kl` is the Dirichlet pseudo-count:
`transition_prior`, or `transition_prior · n_components · transition_kl`
where `transition` is given, so that the given matrix is the prior's mean.
It keeps the matrix row of a state never visited a distribution:

```
N_kl ← λ N_kl                                    before the row reads Π
Π_kl = (N_kl + τ_kl) / Σ_l (N_kl + τ_kl)
ξ_kl = p_k(t−1)·Π_kl·f_l / Σ                     N_kl ← N_kl + w·ξ_kl
```

#### Example

**Because regimes last, the filter beats a rule that knows the true
centers.** This code makes two overlapping blobs, 1.5 apart in each
coordinate, and a sticky chain that switches between them about once in 200
rows. It compares the filter's state with the nearest true center, row by
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
centers = np.array([[-0.75, -0.75], [0.75, 0.75]])            # two blobs of unit spread
xy = centers[state] + rng.standard_normal((n, 2))
pl.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "truth": state}).write_parquet("states.parquet")

# Filter the states, then score the last rows against the nearest true center.
hidden = po.spec.hmm("hidden", features=["x", "y"], n_components=2, precision_prior=1e-2,
                     warm_rows=500, half_life=float("inf"))
rows = pl.scan_parquet("states.parquet").online.fit_predict([hidden]).collect().unnest("hidden").tail(5000)
right = (rows["state"] == rows["truth"]).mean()               # the states are numbered as seeded, so read either way
nearest = ((rows["x"] + rows["y"] > 0).cast(pl.Int64) == rows["truth"]).mean()   # the nearest true center
print(f"rows in their true state: the filter {max(right, 1 - right):.2f}, the nearest true center {nearest:.2f}")
```

```text
rows in their true state: the filter 0.92, the nearest true center 0.85
```

The nearest center reads each row alone. The filter carries what the rows
before told it through the transition matrix it learns, so a row that lands
nearer the other blob does not flip its state.

#### Choosing its settings

To give the transition matrix a sticky prior, and keep a single extreme
row from capturing a state, see [Keep `hmm` states on their
regimes](how-to/keep-hmm-states-on-their-regimes.md).

#### See also

For labeled classes, or for the age of a regime rather than its name:

- [`ew_class`](#ew_class--gaussian-classification-on-ew_cov-moments), the
  same Gaussian densities against labeled classes.
- [`kmeans`](#kmeans--exponentially-weighted-k-means), whose `init` seeds
  the states.
- [`bocpd`](#bocpd--how-long-has-this-regime-lasted), how long the current
  regime has lasted.

#### Notes

**Its learning is held to the rule rebuilt with hmmlearn and scipy.** Each
row's filtered posterior comes from hmmlearn's `GaussianHMM`, and each joint
of consecutive states from scipy's Gaussian densities, with none of the
model's arithmetic. Its seeding is held to scikit-learn's, as `kmeans`'
is. [docs/REGIMES.md §1](REGIMES.md#1-does-hmm-recover-the-stream-that-made-it)
measures how often it puts a row in the state that generated it, seeded from
the rows and given the true covariances.
