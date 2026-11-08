<!-- kind: how-to -->
# Choose a model's settings from the stream's own errors

This how-to is for anyone who fits a model bank and wants to choose a
model's settings, such as its half-life, from the stream's own
out-of-sample errors with `po.eval.grid_search`, and to report what that
choice earned. A model's settings say how long it remembers (`half_life`),
how hard it shrinks (`ridge`), how fast its coefficients may move
(`coef_half_life`, `learning_rate`) and how often it expects a regime to
end (`hazard`).

## Prerequisites

These steps need the library, a stream in a file and a spec:

- polars-online and numpy installed. Each recipe on this page generates
  its own data, so none needs the README's example data.
- The stream saved as a file in clock order, to read back with
  `pl.scan_parquet` as a [query](GLOSSARY.md#the-model-bank-and-what-it-reads)
  (a Polars `LazyFrame`).
- A [spec](GLOSSARY.md#the-model-bank-and-what-it-reads) for the model
  whose setting you choose, and the values of the setting to try: a
  [grid](GLOSSARY.md#grids).

## Choose a half-life with `po.eval.grid_search`

Use this for any setting a spec takes as a list: `half_life` on every
model that decays, `ridge` and `feature_sets` on `ewridge`, and
`penalties` on `lasso`. A bank makes every prediction before it learns
from the row, so each row's residual is out-of-sample, and a setting is
judged by its residuals on rows it had not yet seen.

1. **Give the spec a list of the values to try.** The spec fits each
   value as a grid point, from the same rows (the README's [Output field
   names](../README.md#output-field-names) says how each is named).
2. **Pass the query and the spec to `po.eval.grid_search`.** It runs one
   bank over the query, reading the file once. It cuts the rows into
   `n_splits` test spans, 5 by default, and before each span chooses from
   the rows before it. It returns a frame with a row for each
   [grid point](GLOSSARY.md#grids) and target, and a row for the choice of
   each target.
3. **Run the grid point ranked 1 on the rows to come.** Read it with
   `table.filter(kind="candidate")`, where `rank` is 1.
4. **Report the error of the choice, from `table.filter(kind="choice")`,
   not the error of rank 1.** The rows before each test span chose a grid
   point, and the choice was scored on the span, so its error was earned on
   rows the choice had not seen.

This code chooses a half-life for a stream whose coefficient wanders:

```python
import numpy as np
import polars as pl

import polars_online as po

# y = b x0 + 0.5 x1 + noise, where b wanders as a slow random walk.
rng = np.random.default_rng(9)
n = 40_000
x0, x1, e = rng.standard_normal((3, n))
b = 1.0 + np.cumsum(rng.normal(0, 0.01, n))
pl.DataFrame({"t": np.arange(n, dtype=float), "x0": x0, "x1": x1, "y": b * x0 + 0.5 * x1 + e}).write_parquet("wander.parquet")

# 1. One spec, five half-lives: a grid, one fit per half-life, from the same rows.
grid = po.spec.ewridge(
    "grid", targets=["y"], features=["x0", "x1"], clock="t", gap_cap=5.0,
    half_life=[25.0, 100.0, 400.0, 1_600.0, 6_400.0],
)
# 2. Each half-life scored on the rows it had not seen, in one pass; five spans, each chosen from the rows before it.
table = po.eval.grid_search(pl.scan_parquet("wander.parquet"), grid)

# 3. Each half-life's error, its rank, and how often the rows before a span chose it.
for half_life, mse, rank, chosen in table.filter(kind="candidate").select("half_life", "mse", "rank", "folds_chosen").iter_rows():
    print(f"half-life {half_life:>6.0f}: mse {mse:.4f}, rank {rank}, chosen for {chosen} of 5 spans")
# 4. The number to report.
print(f"choosing from earlier rows only: mse {table.filter(kind='choice')['mse'].item():.4f}")
```

```text
half-life     25: mse 1.0556, rank 3, chosen for 0 of 5 spans
half-life    100: mse 1.0278, rank 1, chosen for 5 of 5 spans
half-life    400: mse 1.0425, rank 2, chosen for 0 of 5 spans
half-life   1600: mse 1.1833, rank 4, chosen for 0 of 5 spans
half-life   6400: mse 1.8327, rank 5, chosen for 0 of 5 spans
choosing from earlier rows only: mse 1.0278
```

A half-life of 100 rows predicts best: 25 forgets so fast that its fit is
noisy, and 6,400 averages over a coefficient that has moved. Here every
span chose 100, so the error of the choice and the error of rank 1 agree.

**Choose from earlier rows, and report the choice on later ones,** since
the best error over the rows you report belongs to the grid point whose
noise ran lowest ([Why the best error over a grid is
optimistic](explanation/grids-and-instances.md#why-the-best-error-over-a-grid-is-optimistic)).
`po.eval.grid_search` cuts the rows into spans as scikit-learn's
`TimeSeriesSplit(n_splits)` does, and scores every grid point on the same
rows. Its
[docstring](https://hgilde.github.io/polars-online/eval.html#polars_online.eval.grid_search)
gives the cut, every keyword and every column of the table it returns.
The `kind` column says what each row of the table is:

| `kind` | the row |
|---|---|
| `"candidate"` | a slot (one target at one grid point) to choose among, with its `rank` and `folds_chosen`, the spans the choice ran it on |
| `"selected"`, `"averaged"` | a spec's `emit_selected` or `emit_averaged` prediction, where it writes one: scored on the same rows, never ranked or chosen |
| `"choice"` | what choosing from the rows before each span earned on the span: the number to report |

To follow a best grid point that moves, rank the grid points on every
row with `emit_selected` instead: [Choose among a grid's settings as the
stream runs](how-to/choose-among-a-grid-online.md) shows how, and when
to prefer it to one fixed grid point.

To make the same choice by hand from an output frame you already have, see
[Choose settings by hand](how-to/choose-settings-by-hand.md).

## Choose a setting that takes no list

Use this for a setting a spec does not take as a list, such as `kalman`'s
`coef_half_life` (where a list gives one value per coefficient, not a
grid), `sgd`'s `learning_rate`, `holt`'s `trend_half_life` and `bocpd`'s
`hazard`.

1. **Build a spec builder with `functools.partial`,** giving it the
   keywords every spec shares.
2. **Give `po.eval.grid_search` the builder, and a `param_grid` of the
   values to try.** It builds one spec per grid point and fits them all in
   one bank, which reads the file once. A grid inside one spec is the
   faster of the two forms, by 13% to 28% for five half-lives on one thread
   ([How a grid shares its
   work](explanation/grids-and-instances.md#how-a-grid-shares-its-work)
   gives the measurement and the reason).
3. **Read the table as in [step 3 and step
   4](#choose-a-half-life-with-poevalgrid_search)** of the half-life
   search.

[A Kalman filter's drift speed](#a-kalman-filters-drift-speed) runs these
steps.

## Score a target that looks ahead

Use this when a target looks ahead, such as one summing the next ten
rows. A row's residual is known only once its target has played out, so
without `gap` the choice before a span would read residuals whose
targets were not yet known at the span's start.

1. **Count the most rows any target looks ahead.** For a target defined
   by rows, that is its count of rows. For a target defined by time, it
   is the most rows its window spans.
2. **Pass that count as `gap=` to `po.eval.grid_search`.** It leaves those
   rows out between the rows a choice reads and the span it is scored on.
3. **Read the table as in [step 3 and step
   4](#choose-a-half-life-with-poevalgrid_search)** of the half-life
   search.

## Choose each setting

Each recipe plants a truth, searches its setting with
`po.eval.grid_search`, and prints the error of each value and of the
choice:

| setting | model | searched as | how-to |
|---|---|---|---|
| `half_life` | every model that decays | a list in one spec | [Choose a half-life](#choose-a-half-life-with-poevalgrid_search) |
| `ridge` | `ewridge` | a list in one spec | [Choose a ridge and a half-life](how-to/choose-a-ridge-and-half-life.md#choose-a-ridge) |
| `ridge` and `half_life` | `ewridge` | both lists in one spec | [Choose a ridge and a half-life](how-to/choose-a-ridge-and-half-life.md#choose-the-ridge-and-the-half-life-together) |
| `coef_half_life`, `revert_half_life` | `kalman` | one spec per value | [A Kalman filter's drift speed](#a-kalman-filters-drift-speed) |
| `learning_rate`; `c`; `alpha` | `sgd`; `pa`; `ftrl` | one spec per value | [The learning rate](#the-learning-rate) |
| `half_life` and `trend_half_life` | `holt` | a list in each of one spec per value | [Holt's smoothing](#holts-smoothing) |
| `hazard` and the priors | `bocpd` | one spec per value, scored by `loglik` | [Choose a changepoint hazard](how-to/choose-a-changepoint-hazard.md) |
| `n_clusters`, `n_components`, `huber_delta` | `kmeans`, `hmm`, the robust losses | not by this pattern | [What this cannot choose](#what-this-cannot-choose) |

### A Kalman filter's drift speed

**`kalman`'s `coef_half_life` sets how fast a coefficient may drift:** it
is a coefficient's memory on the clock, the half-life of an exponentially
weighted fit whose steady-state gain the filter's process noise matches. A
list there gives one value per coefficient, so search it with one spec per
value. Give `functools.partial` the keywords
every spec shares and `param_grid` the values to try:

```python
import functools

import numpy as np
import polars as pl

import polars_online as po

# y = b x0 + 0.5 x1 + noise, where b wanders as a random walk.
rng = np.random.default_rng(8)
n = 30_000
x0, x1, e = rng.standard_normal((3, n))
b = 1.0 + np.cumsum(rng.normal(0, 0.02, n))
pl.DataFrame({"t": np.arange(n, dtype=float), "x0": x0, "x1": x1, "y": b * x0 + 0.5 * x1 + e}).write_parquet("drift.parquet")

# One kalman spec per drift speed, in one bank.
kalman = functools.partial(po.spec.kalman, targets=["y"], features=["x0", "x1"], clock="t", gap_cap=5.0, half_life=500.0)
grid = {"coef_half_life": [3.0, 10.0, 30.0, 100.0, 300.0, 1_000.0]}
table = po.eval.grid_search(pl.scan_parquet("drift.parquet"), kalman, grid)

for speed, mse, rank in table.filter(kind="candidate").select("coef_half_life", "mse", "rank").iter_rows():
    print(f"coef_half_life {speed:>6g}: mse {mse:.4f}, rank {rank}")
print(f"chosen from earlier rows: mse {table.filter(kind='choice')['mse'].item():.4f}")
```

```text
coef_half_life      3: mse 1.3468, rank 5
coef_half_life     10: mse 1.1146, rank 3
coef_half_life     30: mse 1.0540, rank 1
coef_half_life    100: mse 1.0553, rank 2
coef_half_life    300: mse 1.1297, rank 4
coef_half_life   1000: mse 1.3533, rank 6
chosen from earlier rows: mse 1.0553
```

Drift speeds of 30 and 100 clock units nearly tie, and the rows before
some spans preferred 100. So the choice earned 1.0553, a little worse than
the 1.0540 that 30 shows over the same rows, and 1.0553 is the number to
report. `revert_half_life`, which pulls each coefficient back toward 0, is
searched the same way: add it to `param_grid`, and every combination of
the two is a spec.

### The learning rate

**`sgd`'s `learning_rate` sets how many rows the coefficients remember:**
about `1 / learning_rate` under `standardize`, the default, since every
row's step moves them and the half-life does not decay them. So the rate
plays the part a half-life plays in a model that keeps sums. It is one
number a spec, searched as `kalman`'s drift speed is. `pa`'s `c`, which
caps each step, and `ftrl`'s `alpha`, its rate's scale, are searched the
same way:

```python
import functools

import numpy as np
import polars as pl

import polars_online as po

# y = b x0 + 0.5 x1 + noise, where b wanders as a slow random walk.
rng = np.random.default_rng(10)
n = 30_000
x0, x1, e = rng.standard_normal((3, n))
b = 1.0 + np.cumsum(rng.normal(0, 0.01, n))
pl.DataFrame({"x0": x0, "x1": x1, "y": b * x0 + 0.5 * x1 + e}).write_parquet("slow.parquet")

# One sgd spec per learning rate.
sgd = functools.partial(po.spec.sgd, targets=["y"], features=["x0", "x1"], half_life=1_000.0)
table = po.eval.grid_search(pl.scan_parquet("slow.parquet"), sgd, {"learning_rate": [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]})

for rate, mse, rank in table.filter(kind="candidate").select("learning_rate", "mse", "rank").iter_rows():
    print(f"learning_rate {rate:<6g} (about {1 / rate:>4.0f} rows): mse {mse:.4f}, rank {rank}")
print(f"chosen from earlier rows: mse {table.filter(kind='choice')['mse'].item():.4f}")
```

```text
learning_rate 0.0003 (about 3333 rows): mse 1.4051, rank 6
learning_rate 0.001  (about 1000 rows): mse 1.0749, rank 4
learning_rate 0.003  (about  333 rows): mse 1.0263, rank 2
learning_rate 0.01   (about  100 rows): mse 1.0262, rank 1
learning_rate 0.03   (about   33 rows): mse 1.0584, rank 3
learning_rate 0.1    (about   10 rows): mse 1.2047, rank 5
chosen from earlier rows: mse 1.0262
```

A memory of about 100 rows predicts best here, the same as a half-life
of 100 in the first example on a coefficient wandering at the same speed.
With a shorter memory each step chases the noise, and with a longer one
the coefficient has moved on before the fit catches up.

### Holt's smoothing

**`holt`'s two smoothing constants are half-lives:** `half_life` for the
level and `trend_half_life` for the trend. A row's textbook gain on the
level is `α = 1 − 2^(−1/half_life)` once the weight has saturated, so a
half-life of 8 rows is an `α` of 0.083. `half_life` takes a list and
`trend_half_life` does not, so this code builds one spec per trend
half-life, each with a grid of level half-lives:

```python
import functools

import numpy as np
import polars as pl

import polars_online as po

# A local linear trend: the slope wanders, and the level follows it with steps of its own.
rng = np.random.default_rng(12)
n = 30_000
slope = np.cumsum(rng.normal(0, 0.002, n))
level = np.cumsum(slope + rng.normal(0, 0.1, n))
pl.DataFrame({"t": np.arange(n, dtype=float), "y": level + rng.standard_normal(n)}).write_parquet("trend.parquet")

# One spec per trend half-life, each a grid of five level half-lives: 25 fits.
holt = functools.partial(po.spec.holt, targets=["y"], clock="t", gap_cap=5.0, half_life=[2.0, 4.0, 8.0, 16.0, 32.0])
table = po.eval.grid_search(pl.scan_parquet("trend.parquet"), holt, {"trend_half_life": [25.0, 100.0, 400.0, 1_600.0, float("inf")]})

# The three best pairs, and what choosing from earlier rows earned.
for trend, level, mse in table.filter(kind="candidate").sort("rank").head(3).select("trend_half_life", "half_life", "mse").iter_rows():
    print(f"level half-life {level:>2.0f} (alpha {1 - 2 ** (-1 / level):.3f}), trend half-life {trend:>4g}: mse {mse:.4f}")
print(f"chosen from earlier rows: mse {table.filter(kind='choice')['mse'].item():.4f}")
```

```text
level half-life  8 (alpha 0.083), trend half-life   25: mse 1.1331
level half-life  4 (alpha 0.159), trend half-life  100: mse 1.1362
level half-life  4 (alpha 0.159), trend half-life   25: mse 1.1401
chosen from earlier rows: mse 1.1342
```

A level half-life of 8 rows with a trend half-life of 25 ranks first.
The trend forgets faster than its default, four times the level's, because
the planted slope wanders. Choosing from earlier rows earned 1.1342
against the 1.1331 the winner shows over the same rows.

## What this cannot choose

`kmeans`' `n_clusters`, `hmm`'s `n_components` and `huber_delta` have no
out-of-sample score this pattern can choose them by: the Notes of
[`po.eval.grid_search`'s
docstring](https://hgilde.github.io/polars-online/eval.html#polars_online.eval.grid_search)
give the reason for each, and what to do instead.

## Check that it worked

The search worked when the table has a `"choice"` row for each target and
the grid point ranked 1 is one you can run. The error of the choice is
then the number to report. What else you may see, and what to do:

| what you see | what it means | what to do |
|---|---|---|
| the call raises `ValueError`, naming the rows it read and the grid point that predicted on the fewest of them | the first choice reads the rows before the first test span, `n // (n_splits + 1)` of them and up to `n_splits` more, less `gap`. A row counts only where every grid point predicts, so a warm-up (`min_weight`, `warm_rows`) longer than that leaves nothing to choose by | give fewer `n_splits`, which lengthen the first span, a smaller `gap`, a shorter warm-up, or more rows |
| the choice's error is above the error of rank 1 | the rows before some spans chose another grid point, as the Kalman recipe's did | report the choice's error: it is the one earned on rows the choice had not seen |
| the error of the choice is the error of rank 1 | every span chose the same grid point, as the half-life recipe's did | report either |
