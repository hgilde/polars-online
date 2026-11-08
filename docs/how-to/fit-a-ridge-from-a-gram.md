<!-- kind: how-to -->
# Fit a ridge from a Gram

This how-to is for anyone with a [Gram](../GLOSSARY.md#fitting-a-model)
per block of rows, such as one per week from one pass, who wants a ridge
regression and its t statistics without reading the rows again. You fit
a grid of ridges in one call, choose one by its error on blocks held out,
and read each coefficient's standard error and t.

## Prerequisites

These steps need the library and a Gram per block:

- polars-online installed with numpy, an optional extra, which
  `pip install "polars-online[numpy]"` adds. The recipe generates its own
  data: 40 weeks of 200 features, of which six matter.
- A Gram per block from one pass, as [Keep a Gram per block in one
  pass](fit-weekly-grams-in-one-pass.md) makes them.
- [A ridge and its t](../GRAM.md#a-ridge-and-its-t) explains the system
  the ridge solves, how its standard errors are read from the Gram, and
  what its t assumes.

## Choose a ridge on blocks held out, and read its t

Use this when the blocks you fit on hold few rows for their columns, so
least squares on them is noisy. Two weeks of rows for 201 columns is such
a case.

1. **Keep a Gram per week in one pass,** with an `ewridge` spec whose
   `group` is the week and whose `half_life` is `float("inf")`.
   `bank.gram(name)` returns one dict per week, oldest first.
2. **Pool the weeks to fit on, and the weeks held out, with
   `po.gram.merge`.** Each call returns the Gram of all their rows.
3. **Fit a grid of ridges in one call, with
   `po.gram.solve(train, ridge=[...], standardize=True)`.** It returns a
   row of coefficients per ridge, the intercept first. With
   `standardize=True` one ridge means the same whatever the columns'
   units.
4. **Score each row of the grid on the held-out Gram.** The score is a
   function of the held-out Gram's arrays, as `mean_squared_error` in the
   recipe.
5. **Keep the ridge with the lowest held-out error.**
6. **Read the standard errors and t with `po.gram.coef_stats(train,
   coef)`,** for the least-squares fit and for the ridge you kept.

This code makes the stream and runs step 1:

```python
from datetime import datetime, timedelta

import numpy as np
import polars as pl

import polars_online as po

# 40 weeks, a row every 20 minutes. 200 features in 40 families of five that share a factor, so
# two features of one family correlate at 0.5. y depends on six of them, and x25's coefficient is
# drawn again each week, 0.15 on average.
rng = np.random.default_rng(7)
n_weeks, per_week, k = 40, 504, 200
n = n_weeks * per_week
x = 0.7 * (rng.standard_normal((n, k)) + np.repeat(rng.standard_normal((n, k // 5)), 5, axis=1))
b25 = 0.15 + 0.3 * rng.standard_normal(n_weeks)
y = x[:, [0, 5, 10, 15, 20]] @ [0.5, 0.4, 0.3, 0.2, 0.1] + b25.repeat(per_week) * x[:, 25]
y += rng.standard_normal(n)
features = [f"x{j}" for j in range(k)]
# Stamp each row, from a Monday, and save the table for `pl.scan_parquet` to read.
start = datetime(2025, 1, 6)  # a Monday
ts = pl.datetime_range(start, start + timedelta(minutes=20 * (n - 1)), "20m", eager=True)
pl.DataFrame({"ts": ts, **dict(zip(features, x.T)), "y": y}).write_parquet("weeks.parquet")

# 1. One pass over the file leaves a Gram per week. The week is a column made in the query.
lf = pl.scan_parquet("weeks.parquet").with_columns(week=pl.col("ts").dt.truncate("1w"))
spec = po.spec.ewridge(
    "blocks", targets=["y"], features=features, group="week",
    half_life=float("inf"), solve_every=float("inf"), target_gaps="pairwise", gram_block_rows=256,
)
bank = po.ModelBank([spec])
bank.fit(lf)  # learns from every row, and keeps only the state
blocks = bank.gram("blocks")  # one dict per week, oldest first
```

The next code fits a grid of ridges on weeks 1 and 2, and scores each on
the other 38 weeks. Then it reads the t of the least-squares fit and of
the best ridge, and checks the standard errors against the two weeks'
rows:

```python
# The score of step 4: a fit's mean squared error on any set of weeks, without reading their rows.
def mean_squared_error(g, coef):
    """A fit's mean squared error over the rows behind g, from g's arrays alone."""
    b0, b = coef[0], coef[1:]
    gap = g["cross_moments"][0][0] - b0 - g["means"][1:] @ b
    spread = g["target_vars"][0] - 2 * b @ g["cross_centered"][0][1:] + b @ g["comoments"][1:, 1:] @ b
    return spread + gap**2


# 2 to 5. Pick the ridge by its error on the weeks held out.
train, held_out = po.gram.merge(blocks[:2]), po.gram.merge(blocks[2:])  # fit on two weeks, judge on 38
ridges = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0]
grid = po.gram.solve(train, ridge=ridges, standardize=True)  # a row per ridge, one eigendecomposition
errors = [mean_squared_error(held_out, coef) for coef in grid]
print("held-out error by ridge: " + ", ".join(f"{r} {e:.4f}" for r, e in zip(ridges, errors)))
best = int(np.argmin(errors))

# 6. coef_stats on the least-squares fit and on the chosen ridge's, for the planted columns.
plain, ridged = po.gram.coef_stats(train, grid[0]), po.gram.coef_stats(train, grid[best])
print(f"{'column':>6} {'coef':>6} {'t':>5} {'ridge ' + str(ridges[best]):>12} {'t':>5}")
for name in ["x0", "x5", "x10", "x15", "x20", "x25"]:
    i = train["columns"].index(name)
    print(f"{name:>6} {grid[0][i]:>6.3f} {plain['t'][i]:>5.1f} {grid[best][i]:>12.3f} {ridged['t'][i]:>5.1f}")

# At ridge 0, coef_stats' standard errors are least squares' own, from the two weeks' rows.
rows = pl.read_parquet("weeks.parquet").head(2 * per_week)
design = np.column_stack([np.ones(rows.height), rows.select(features).to_numpy()])
resid = rows["y"].to_numpy() - design @ grid[0]
dof = rows.height - design.shape[1]
se = np.sqrt(np.diag(np.linalg.inv(design.T @ design)) * (resid @ resid) / dof)
gap = np.abs(plain["se"] / se - 1).max()  # the largest relative gap, the intercept's included
print(f"coef_stats' standard errors equal the rows' within 1e-9 relative, the intercept's included: {gap < 1e-9}")
```

```text
held-out error by ridge: 0.0 1.3521, 0.01 1.3433, 0.03 1.3285, 0.1 1.2959, 0.3 1.2706, 1.0 1.3050
column   coef     t    ridge 0.3     t
    x0  0.487  10.0        0.320   6.4
    x5  0.344   7.0        0.225   4.5
   x10  0.204   4.2        0.139   2.8
   x15  0.200   4.2        0.134   2.7
   x20  0.138   3.0        0.078   1.6
   x25  0.085   1.8        0.057   1.2
coef_stats' standard errors equal the rows' within 1e-9 relative, the intercept's included: True
```

A ridge of 0.3, in correlation units, lowers the held-out error from
1.3521 to 1.2706. It shrinks every coefficient toward 0, that of `x0`
from 0.487 to 0.320, and each t falls with its coefficient, from 10.0 to
6.4 for `x0`. The standard errors barely move: they are least squares',
read at the ridge's slightly larger residual variance.

## Check the result

**At ridge 0, `po.gram.coef_stats` gives the standard errors numpy
computes from the rows,** within 1e-9 relative in the recipe, the
intercept's included, when every row
weighs 1 and `half_life` is `float("inf")`. What else you may see, and
what to do:

| what you see | what it means | what to do |
|---|---|---|
| a coefficient's t is as large as a steady column's, though its coefficient changes from week to week | the t assumes one coefficient for all the pooled rows | fit each week alone, and read the spread of the weekly coefficients ([Fit per-block coefficients](fit-per-block-coefficients.md)) |
| a ridge's t is smaller than least squares' | the ridge shrinks each coefficient, and its standard errors leave out the shrinkage and its bias | read a ridge's t to compare its coefficients with one another, not as a test |
| the errors are correlated or heteroskedastic | least squares' standard errors are then too small | switch on `emit_robust_se` on the stream ([Can its t be trusted?](../DIAGNOSTICS.md#can-its-t-be-trusted)), or fit per-block coefficients |
| the intercept's standard error and t are `nan` | the Gram was pooled within blocks, and each block had its own intercept | read the slopes ([Pool within blocks to leave their levels out](pool-grams.md#pool-within-blocks-to-leave-their-levels-out)) |
| `po.gram.solve` differs from `bank.coef()` | the spec has `ridge_scale="sum"` or a `coef_prior`, or a `solve_every` that leaves `bank.coef()` as of an earlier row | to reproduce `bank.coef()`, pass `po.gram.solve` the spec's `ridge` and `standardize`. `solve` does not reproduce `ridge_scale="sum"` (a ridge on a decaying scale) or a `coef_prior` (a prior other than zero), so read those fits from `bank.coef()`. For the fit as of the last row, solve the Gram |

The [reference for `po.gram.solve`](https://hgilde.github.io/polars-online/gram.html#polars_online.gram.solve)
and for [`po.gram.coef_stats`](https://hgilde.github.io/polars-online/gram.html#polars_online.gram.coef_stats)
give every parameter.
