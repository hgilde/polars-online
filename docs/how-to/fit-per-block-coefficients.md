<!-- kind: how-to -->
# Fit per-block coefficients

This how-to is for anyone with a [Gram](../GLOSSARY.md#fitting-a-model)
per block of rows, such as one per week from one pass, who wants to know
whether a coefficient holds from block to block. You fit each block
alone in one call, and read a t from the spread of the weekly
coefficients beside the pooled fit's t.

## Prerequisites

These steps need the library and a Gram per block:

- polars-online installed with numpy, an optional extra, which
  `pip install "polars-online[numpy]"` adds. The recipe generates its own
  data: 40 weeks of 200 features, of which six matter, and `x25`'s
  coefficient is drawn again each week.
- A Gram per block from one pass, as [Keep a Gram per block in one
  pass](fit-weekly-grams-in-one-pass.md) makes them.
- The columns to fit, already chosen. A week of 504 rows fits 201 columns
  poorly, so choose them before you fit each week, as [Select columns that hold across the
  weeks](select-columns-stably-over-weeks.md) does.
- [Per-block coefficients](../GRAM.md#per-block-coefficients) explains
  Fama and MacBeth's t, which reads the weekly coefficients as a sample,
  and what it assumes.

## Fit each block alone, and read the spread

Use this before you trust a pooled t, for each column whose coefficient
you mean to read.

1. **Keep a Gram per week in one pass,** with an `ewridge` spec whose
   `group` is the week and whose `half_life` is `float("inf")`.
2. **Fit each week alone with `po.gram.solve_subsets(blocks, [[0], [1],
   ...], features=chosen)`.** It returns one list per subset, a fit per
   target, each with its own coefficients, standard errors and t.
3. **Fit the pooled weeks on the same columns,** with `po.gram.merge`,
   `po.gram.solve(..., features=chosen)` and `po.gram.coef_stats`.
4. **For each column, compute Fama and MacBeth's t,** the mean of its
   weekly coefficients over `sd / √B`, with `sd` their standard deviation
   and `B` the number of weeks, and set it beside the pooled t.
5. **Count the weeks whose sign matches the pooled fit's.**

`po.gram.solve_subsets` reads the Grams once and fits every subset in one
call. For 38 Grams and 100 subsets of 19, measured on an M4 Pro under a
load of 12, it took 0.02 s at 200 columns and 3.0 s at 2,000, where a
loop of `po.gram.merge`, `po.gram.solve` and `po.gram.coef_stats` took
0.43 s and 76 s.

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
# Stamp each row, from a Monday, and save the table for the scan to read.
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

The next code fits the six planted columns each week alone and pooled
over all forty, and sets the two t side by side:

```python
# 2 and 3. The six chosen columns, fit each week alone and pooled over all forty.
chosen = ["x0", "x5", "x10", "x15", "x20", "x25"]
each = po.gram.solve_subsets(blocks, [[i] for i in range(n_weeks)], features=chosen)  # each week alone
pooled = po.gram.merge(blocks)
coef = po.gram.solve(pooled, features=chosen)
t = po.gram.coef_stats(pooled, coef, features=chosen)["t"]

# 4 and 5. The pooled fit beside the weekly ones, column by column.
columns = pooled["columns"]
print(f"{'column':>6} {'pooled':>7} {'t':>5} {'weekly mean':>12} {'sd':>6} {'t':>5} {'same sign':>10}")
for name in chosen:
    i = columns.index(name)
    # The column's forty weekly coefficients, read as a sample.
    weekly = np.array([fits[0]["coef"][i] for fits in each])
    sd = weekly.std(ddof=1)
    t_weeks = weekly.mean() / (sd / np.sqrt(n_weeks))  # Fama and MacBeth's t
    sign = np.mean(np.sign(weekly) == np.sign(coef[i]))
    print(f"{name:>6} {coef[i]:>7.3f} {t[i]:>5.1f} {weekly.mean():>12.3f} {sd:>6.3f} {t_weeks:>5.1f} {sign:>10.0%}")
```

```text
column  pooled     t  weekly mean     sd     t  same sign
    x0   0.496  65.8        0.494  0.049  63.7       100%
    x5   0.390  51.7        0.392  0.041  60.5       100%
   x10   0.307  40.9        0.308  0.040  48.9       100%
   x15   0.196  26.3        0.193  0.038  32.2       100%
   x20   0.092  12.4        0.096  0.051  11.9        98%
   x25   0.196  26.2        0.195  0.303   4.1        75%
```

The pooled fit here is on the six chosen columns, so its t are larger
than those of the fit on all 200 in [Keep weekly Grams of hundreds of
features](fit-weekly-grams-in-one-pass.md#keep-weekly-grams-of-hundreds-of-features).
`x0` to `x20` hold: their weekly means match the pooled coefficients, and
the t from their spread, 11.9 to 63.7, stands beside the pooled t, 12.4
to 65.8. `x25`'s pooled t is 26.2, as firm as `x15`'s 26.3. But its
weekly coefficients spread by 0.303, and its t from the weeks is 4.1. In
a quarter of the weeks its sign is the opposite of the pooled fit's.

## Check the result

**A coefficient holds from block to block when its t from the weeks
stands near its pooled t and its sign rarely changes.** What else you may
see, and what to do:

| what you see | what it means | what to do |
|---|---|---|
| the t from the weeks is far below the pooled t | the coefficient changes from week to week, and the pooled t leaves that variation out | read the t from the weeks, not the pooled t |
| the weekly coefficients drift slowly, from one run of weeks to the next | they are not independent of one another, as Fama and MacBeth's t assumes | take Newey and West's long-run variance of their series in place of `sd²` ([Per-block coefficients](../GRAM.md#per-block-coefficients)) |
| the weekly coefficients are wild, and their standard errors large | a week holds too few rows for the columns fitted | fit fewer columns, or make each block longer |

## Sources

Fama and MacBeth's t and Newey and West's long-run variance come from
these papers:

| source | for |
|---|---|
| Fama, E. F. and MacBeth, J. D. (1973). Risk, return, and equilibrium: empirical tests. *Journal of Political Economy* 81(3), 607–636. | a coefficient's t from its estimates period by period |
| Newey, W. K. and West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica* 55(3), 703–708. | the variance of a series of weekly coefficients that drift |
