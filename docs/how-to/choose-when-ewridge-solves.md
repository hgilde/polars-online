<!-- kind: how-to -->
# Choose when `ewridge` solves

This how-to is for anyone who fits `ewridge` and wants to trade how
current its coefficients are against how long a run takes. You set
`solve_every` and `max_rows_between_solves`, and check how stale each
schedule leaves the fit.

## Prerequisites

These steps need the library and an `ewridge` spec:

- polars-online and numpy installed. Each recipe generates its own data.
- An `ewridge` [spec](../GLOSSARY.md#the-model-bank-and-what-it-reads)
  with a [clock](../GLOSSARY.md#time) and a half-life, or with
  `half_life=float("inf")` or `lam`.
- What a [solve](../GLOSSARY.md#fitting-a-model) is: `ewridge` learns each
  row into its [Gram](../GLOSSARY.md#fitting-a-model), the running means
  and co-moments of its columns, and computes its coefficients from the
  Gram on a schedule. **Each prediction uses the coefficients of the last
  solve,** so a coarser schedule is faster and its coefficients older.

`solve_every` and `max_rows_between_solves` are defined in
[`po.spec.ewridge`'s
docstring](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.ewridge),
whose Notes, under *The solve schedule*, give the rules at the schedule's
edges.

## Choose a schedule under a finite half-life

Use this for a fit with a finite half-life, where the default schedule
solves by weight. **The default solves once the weight learned since the
last solve reaches `ln 2 / 50` of the weight the fit holds:** about 1.4%,
or every `half_life/50` of clock once the weight has neared its steady
level, a few half-lives in. It solves more often during the warm-up and
after a gap, where the fit moves most.

| if you want | give |
|---|---|
| the coefficients current at every row | `max_rows_between_solves=1`: every row is then predicted from a fit of every row before it |
| solves on the clock | `solve_every` in clock units, such as `solve_every=100.0` on a clock in seconds |
| a solve only for each target's first fit, and then where a row cap says | `solve_every=float("inf")` with a `max_rows_between_solves` |
| a bound on how old the coefficients can get, under any schedule | `max_rows_between_solves`, the most rows between two solves, whatever the clock does |

**A `solve_every` starts with a first period that solves by weight,** until
`solve_every` has passed since the group's first row or its last
[reset](../GLOSSARY.md#time), or `max_rows_between_solves` rows with weight
have been learned. So a fit of its first few rows is not held for a whole
period. With `solve_every=500` and `min_weight=50`, the mean squared error
of the predictions over the first 600 rows was 3.1e-6, where holding the
first fit for the whole period gave 5.6e-4, on five seeds of a
five-feature stream at a half-life of 1,000.

**`solve_every=float("inf")` never comes due, and neither does any value
from `1e18` on,** since `1e18` lies past the end of any stream. Neither has
a first period. The model keeps its sums, and solves only to give each
target its first fit and where the row cap says.

1. **Choose a schedule from the table.**
2. **Fit it beside `max_rows_between_solves=1` in one bank,** over the same
   rows.
3. **Compare the two specs' `pred_<target>` fields.**
4. **Time each schedule in a bank of its own,** as [Space the solves under
   `half_life=float("inf")` or
   `lam`](#space-the-solves-under-half_lifefloatinf-or-lam) does.

This code fits one stream under four schedules, at a half-life of 200
rows, and holds each to the fit that solves on every row. A slope jumps
from 1 to 3 at row 10,000, so a stale fit shows in the error after the
jump. Each spec writes its coefficients on every row (`coef_every=0`), and
the recipe counts a solve where the slope changes:

```python
import numpy as np
import polars as pl

import polars_online as po

# y = slope * x0 - x1 + 0.5 x2 plus noise, one row a clock unit; the slope on x0 jumps from 1 to 3 at row 10,000.
rng = np.random.default_rng(7)
n = 20_000
x = rng.standard_normal((3, n))
slope = np.where(np.arange(n) < 10_000, 1.0, 3.0)
y = slope * x[0] - x[1] + 0.5 * x[2] + 0.5 * rng.standard_normal(n)
pl.DataFrame({"t": np.arange(n, dtype=float), "x0": x[0], "x1": x[1], "x2": x[2], "y": y}).write_parquet("jump.parquet")

# One spec per schedule, each writing its coefficients on every row.
common = dict(targets=["y"], features=["x0", "x1", "x2"], clock="t", gap_cap=10.0, half_life=200.0, coef_every=0)
schedules = {
    "max_rows_between_solves=1": po.spec.ewridge("every_row", max_rows_between_solves=1, **common),
    "the default, by weight": po.spec.ewridge("by_weight", **common),
    "solve_every=100": po.spec.ewridge("on_clock", solve_every=100.0, **common),
    "solve_every=inf, a row cap of 2,000": po.spec.ewridge("capped", solve_every=float("inf"), max_rows_between_solves=2000, **common),
}
out = pl.scan_parquet("jump.parquet").online.fit_predict(list(schedules.values())).collect()

# For each schedule: the solves it ran, its gap from the fit of every row, and its error after the jump.
current = out["every_row"].struct.field("pred_y")
print("the slope on x0 jumps from 1 to 3 at row 10,000; the noise's variance is 0.25")
for label, spec in schedules.items():
    # A solve shows where the slope it reports changes from the row before.
    fields = out[spec["name"]].struct.unnest()
    slope0 = fields["coef"].list.get(1)
    solves = int((slope0.is_not_null() & (slope0 != slope0.shift(1)).fill_null(True)).sum())
    gap = float(((fields["pred_y"] - current) ** 2).mean())
    after = float(((fields["pred_y"] - out["y"]) ** 2)[10_000:10_200].mean())
    print(f"{label}: {solves} solves; gap from every row's fit {gap:.5f}; error over the 200 rows after the jump {after:.2f}")
```

```text
the slope on x0 jumps from 1 to 3 at row 10,000; the noise's variance is 0.25
max_rows_between_solves=1: 20000 solves; gap from every row's fit 0.00000; error over the 200 rows after the jump 2.52
the default, by weight: 4195 solves; gap from every row's fit 0.00003; error over the 200 rows after the jump 2.54
solve_every=100: 287 solves; gap from every row's fit 0.00185; error over the 200 rows after the jump 3.23
solve_every=inf, a row cap of 2,000: 10 solves; gap from every row's fit 0.53283; error over the 200 rows after the jump 3.88
```

The default ran a fifth of the solves of every row, and its predictions
are within 0.00003 of every row's fit in mean squared difference. A
schedule that solves every 100 rows, half the half-life, lags the jump.
The row cap of 2,000 holds a fit for ten half-lives.

## Space the solves under `half_life=float("inf")` or `lam`

Use this for a fit that weighs every row the same, or that sets its decay
by `lam`. **Under both, the default solves on every row.**

1. **Give a `solve_every` when the fit need not be current at every
   row.** The coefficients are then at most `solve_every` clock units out
   of date.
2. **Time it beside the default, each in a bank of its own,** over the
   same rows.
3. **Compare each fit's coefficients with the other's,** or with the
   planted ones where you know them.

This code times both schedules over 200,000 rows of 20 features on one of
the bank's threads, and compares each fit with the planted slopes:

```python
import time

# 200,000 rows of 20 features, every row weighed the same.
rng = np.random.default_rng(8)
n, k = 200_000, 20
x = rng.standard_normal((n, k))
beta = np.linspace(-1.0, 1.0, k)
frame = pl.DataFrame({f"x{j}": x[:, j] for j in range(k)}).with_columns(y=pl.Series(x @ beta + rng.standard_normal(n)))
frame.write_parquet("flat.parquet")
features = [f"x{j}" for j in range(k)]
# The same fit, solved on every row and every 1,000 rows.
for label, extra in [("every row (the default under half_life=inf)", {}), ("solve_every=1000", {"solve_every": 1000.0})]:
    spec = po.spec.ewridge("flat", targets=["y"], features=features, half_life=float("inf"), **extra)
    bank = po.ModelBank([spec], threads=1)
    # Time the pass, then read the fit's slopes beside the planted ones.
    start = time.perf_counter()
    bank.fit(pl.scan_parquet("flat.parquet"))
    took = time.perf_counter() - start
    coef = bank.coef().filter(pl.col("term") != "intercept")["coef"].to_numpy()
    print(f"time {label}: {took:.2f} s")
    print(f"{label}: largest slope error {np.abs(coef - beta).max():.3f}")
```

```text
time every row (the default under half_life=inf): 0.40 s
every row (the default under half_life=inf): largest slope error 0.006
time solve_every=1000: 0.06 s
solve_every=1000: largest slope error 0.006
```

## Keep the solves a group's warm-up skips

Use this when a group's weight stays below its floor, and you need its
`coef` or its count of failed solves. **Under a finite half-life, the
default schedule skips each solve that nothing would read while a group's
weight is more than about 1.4% below its floor,** where every prediction
is withheld. The
docstring's Notes, under *The solve schedule*, give the floor and the
cases where such a solve still runs.

1. **Read `bank.coef()` for the group.** A group whose weight never
   reaches the floor, such as one of two rows over two features, reports
   no `coef`, and `bank.solve_failures()` counts only the solves run.
2. **If you need its solves, give `min_weight=0`,** which makes the floor
   0 and withholds no prediction for its weight, **or
   `max_rows_between_solves=1`,** whose solves always run.

## Check the result

**Hold a schedule to the fit that solves on every row, as the first recipe
does:** fit both in one bank and compare their `pred_<target>` fields. The
mean squared difference is what the schedule costs in staleness. What it
saves is the time of a run, measured in a bank of each schedule's own, as
[Space the solves under `half_life=float("inf")` or
`lam`](#space-the-solves-under-half_lifefloatinf-or-lam) does.

| what you see | what it means | what to do |
|---|---|---|
| a large gap from every row's fit, or a slow response to a change | the solves are too far apart for the half-life | give a smaller `solve_every`, or add `max_rows_between_solves` |
| a run that spends its time solving, under `half_life=float("inf")` or `lam` | the default solves on every row | give a `solve_every` |
| no `coef` for a group with only a few rows | its weight never reached the floor, so no solve ran | give `min_weight=0` or `max_rows_between_solves=1` |
