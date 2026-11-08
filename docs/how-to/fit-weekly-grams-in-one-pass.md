<!-- kind: how-to -->
# Keep a Gram per block in one pass, and fit from it

This how-to is for anyone who wants to fit many least-squares models, on
any run of weeks or hours and any of the columns, from one pass over a
stream too large to hold.

Three models, `ewridge`, `lasso` and `ew_cov`, keep a
[Gram](../GLOSSARY.md#fitting-a-model) that every least-squares fit on
its columns can read in place of the rows. [How a Gram stands in for the
rows](../GRAM.md#how-a-gram-stands-in-for-the-rows) explains why.

## Prerequisites

These steps need the library, numpy and a column that names each block:

- polars-online installed with numpy, an optional extra, which
  `pip install "polars-online[numpy]"` adds. `bank.gram(spec)` returns
  numpy arrays, and `po.gram` fits from them.
- The README's [Example data](../../README.md#example-data) built. The
  example reads `lf`, a
  [query](../GLOSSARY.md#the-model-bank-and-what-it-reads) over
  `ticks.parquet`: 400 rows a minute apart, with the timestamp `ts`, five
  features and the target `y`, which is `1 + 2·x0 − x1` plus noise.
- A column whose value names each row's block, or a timestamp to make one
  from in the query, as `pl.col("ts").dt.truncate("1h")`.

## Keep a Gram per block, and fit from them

One pass over the rows keeps the Grams, and each fit afterwards reads only
them.

1. **Make the block column in the query,** with Polars' `with_columns`
   before the bank.
2. **Name it as the spec's `group`,** and the bank keeps a Gram for each
   block. Give `half_life=float("inf")`, so the Grams are least squares'
   sums, and `solve_every=float("inf")`, so the pass solves nothing.
3. **Run the pass with `bank.fit`,** which learns from every row and keeps
   no output.
4. **Read the Grams with `bank.gram(name)`,** one dict of numpy arrays per
   block, oldest first.
5. **Pool the blocks you want with `po.gram.merge`,** and fit with
   `po.gram.solve`. `po.gram.coef_stats` gives each coefficient's t.

Here one pass leaves eight hourly Grams of `df`'s five features. Pooled,
they give least squares on all 400 rows, with `df`'s truth and the t of
each coefficient. Any run of hours, on any of the columns, is one more
call. This code uses `lf` from [Example data](../../README.md#example-data):

```python
# 1 to 3. One Gram per hour, from one pass.
features = ["x0", "x1", "x2", "signal_a", "signal_b"]
hours = po.spec.ewridge(
    "hours", targets=["y"], features=features, group="hour",   # a Gram per hour
    half_life=float("inf"), solve_every=float("inf"),          # least squares' sums; the pass solves nothing
)
bank = po.ModelBank([hours])
bank.fit(lf.with_columns(hour=pl.col("ts").dt.truncate("1h")))   # the hour, made in the query
hourly = bank.gram("hours")                 # 4. one dict of numpy arrays per hour, oldest first

# 5. Pool every hour into one fit.
pooled = po.gram.merge(hourly)              # the Gram of all the hours' rows
coef = po.gram.solve(pooled)                # least squares: the intercept, then the features
t = po.gram.coef_stats(pooled, coef)["t"]   # and each coefficient's t
print(f"{len(hourly)} hourly Grams, pooled over {pooled['weight_sum']:.0f} rows")
for name, b, tb in zip(pooled["columns"], coef, t):
    print(f"{name:>9} {b:+.2f}  t {tb:+6.1f}")
morning = po.gram.solve(po.gram.merge(hourly[:3]), features=["x0", "x1"])   # any hours, any columns, from the same pass
print("x0 and x1 before 12:00:", " ".join(f"{b:+.2f}" for b in morning[1:3]))
```

```text
8 hourly Grams, pooled over 400 rows
intercept +0.99  t  +39.1
       x0 +1.99  t  +78.3
       x1 -1.01  t  -39.9
       x2 -0.03  t   -1.0
 signal_a +0.02  t   +0.6
 signal_b +0.03  t   +1.3
x0 and x1 before 12:00: +2.00 -1.05
```

## Keep weekly Grams of hundreds of features

Use this for a wide pass: weeks of a stream too large to hold, with
hundreds or thousands of features. These are the steps of [Keep a Gram
per block, and fit from them](#keep-a-gram-per-block-and-fit-from-them),
with the week as the block and these settings:

| setting | value in the recipe | what it does |
|---|---|---|
| `group` | `"week"`, the column made by `pl.col("ts").dt.truncate("1w")` | a state, and so a Gram, for each week |
| `half_life` | `float("inf")` | every row weighs the same, so a block's Gram is least squares' and blocks pool as they are |
| `solve_every` | `float("inf")`, never, on any clock | under an infinite half-life the model solves on every row unless told otherwise. The fits are made from the Grams, so the pass solves nothing |
| `target_gaps` | `"pairwise"` | one Gram per block whatever rows each target is missing on, so every block has the same targets and they pool |
| `gram_block_rows` | `256` | adds 256 rows to the matrix in one product, 5.1 to 6.6 times as fast as one row at a time from 256 to 2,000 features ([Fit hundreds of features](fit-hundreds-of-features.md#add-rows-to-the-gram-in-blocks)) |
| `gram_threads` | left out, so `"auto"` | runs that product on the threads the bank's other groups leave idle ([Fit hundreds of features](fit-hundreds-of-features.md#run-the-grams-work-on-idle-threads)) |
| `group_close` | not in the recipe; `"monotone"` when the weeks arrive in order | with it and `closed_groups=` on `fit`, the bank writes each week's Gram to a file once the next week starts, and holds one week at a time ([Read Grams from saved files](read-grams-from-saved-files.md)) |

1. **Make the week column in the query,** with
   `pl.col("ts").dt.truncate("1w")` in Polars' `with_columns` before the
   bank.
2. **Give the spec the settings in the table.**
3. **Run the pass with `bank.fit`.**
4. **Read the Grams with `bank.gram(name)`,** one dict per week, oldest
   first.
5. **Pool them with `po.gram.merge`.**

**A block's Gram has `k²` entries, whatever its rows.**
A week of 504 rows at 201 columns has a co-moment matrix of 323,208
bytes, and so has a week of a million rows. At 10,000 columns a block is
800 MB in float64. 9.5 million rows read as 38 weekly blocks of 10,000
columns make 30 GB of Grams. The bank holds the Gram of every group it
has seen, so a pass needs about `8k²` bytes for each block it holds. With
`group_close="monotone"` and `closed_groups=` on `fit`, the bank writes each finished week out and holds one week
at a time. [Store a compact Gram](store-a-compact-gram.md) halves or
quarters what `gram()` hands back.

**Choose the block so that every question a fit will ask is a set of
whole blocks.** Weeks let a fit pool any run of weeks, hold out the last
ten, or set one week against another. Days would allow finer choices,
with seven times as many Grams to hold.

**Give `target_gaps="pairwise"` only when every target is present on the
same rows, or the rows a target misses do not depend on the features'
values.** Under the default `target_gaps="own_rows"`, a block can hold
several Grams: a target that is missing on rows where another is present
gets a Gram of its own, one per pattern of gaps, each naming its
`targets`, and `po.gram.merge` refuses parts whose targets differ. With
the default, pool each set of targets' Grams across the weeks, those with
the same `targets`. MODELS.md's
[`ewridge`](../MODELS.md#ewridge--exponentially-weighted-ridge-regression)
section says what each mode fits.

The recipe makes a stream of 40 weeks with 200 features, of which six
matter, runs the pass, pools the forty weeks, and checks the pooled fit
against least squares on the rows, read into memory for the check. This
code builds its own data:

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

# 1 to 3. One pass over the file leaves a Gram per week. The week is a column made in the query.
lf = pl.scan_parquet("weeks.parquet").with_columns(week=pl.col("ts").dt.truncate("1w"))
spec = po.spec.ewridge(
    "blocks", targets=["y"], features=features, group="week",
    half_life=float("inf"), solve_every=float("inf"), target_gaps="pairwise", gram_block_rows=256,
)
bank = po.ModelBank([spec])
bank.fit(lf)  # learns from every row, and keeps only the state
blocks = bank.gram("blocks")  # 4. one dict per week, oldest first

# What one week's block holds, and its size.
first = blocks[0]
print(f"{len(blocks)} blocks, the first for the week of {first['group'][:10]}")
print(f"weight_sum {first['weight_sum']:.0f} and n_kish {first['n_kish']:.0f}: its rows, weighed alike")
print(f"comoments {first['comoments'].shape}: {first['comoments'].nbytes:,} bytes a block")

# 5. Pool the forty weeks, and fit with standard errors as if on the rows.
pooled = po.gram.merge(blocks)  # the Gram of all 20,160 rows
coef = po.gram.solve(pooled)  # least squares: the intercept, then x0 .. x199
stats = po.gram.coef_stats(pooled, coef)  # its standard errors and t
# Each planted coefficient beside its estimate.
columns = pooled["columns"]
print(f"pooled over {pooled['weight_sum']:,.0f} rows:")
print(f"{'column':>6} {'truth':>6} {'coef':>6} {'t':>6}")
truth = {"x0": 0.5, "x5": 0.4, "x10": 0.3, "x15": 0.2, "x20": 0.1, "x25": b25.mean()}
for name, b in truth.items():
    i = columns.index(name)
    print(f"{name:>6} {b:>6.3f} {coef[i]:>6.3f} {stats['t'][i]:>6.1f}")
# The columns with no effect, and how many look as if they had one.
rest = np.abs([stats["t"][columns.index(f)] for f in features if f not in truth])
print(f"the other {len(rest)} columns: {(rest > 3).sum()} with |t| above 3, the largest {rest.max():.1f}")

# The pooled Gram's least squares, against least squares on the rows read into memory.
rows = pl.read_parquet("weeks.parquet")
design = np.column_stack([np.ones(n), rows.select(features).to_numpy()])
by_rows = np.linalg.lstsq(design, rows["y"].to_numpy(), rcond=None)[0]
print(f"the pooled fit and least squares on the rows agree within 1e-12: {np.abs(coef - by_rows).max() < 1e-12}")
```

```text
40 blocks, the first for the week of 2025-01-06
weight_sum 504 and n_kish 504: its rows, weighed alike
comoments (201, 201): 323,208 bytes a block
pooled over 20,160 rows:
column  truth   coef      t
    x0  0.500  0.489   50.0
    x5  0.400  0.389   39.6
   x10  0.300  0.297   30.5
   x15  0.200  0.196   20.3
   x20  0.100  0.100   10.4
   x25  0.186  0.196   20.3
the other 194 columns: 1 with |t| above 3, the largest 3.1
the pooled fit and least squares on the rows agree within 1e-12: True
```

Each block holds its week's 504 rows, weighed alike, so its `weight_sum`
and `n_kish` are both the row count. Pooled, the forty blocks give the
least-squares fit that numpy's `lstsq` computes from all 20,160 rows, to
within 1e-12 in every coefficient. That fit finds each planted
coefficient within about one standard error. Of the 194 columns with no
effect, one has a t past 3, at 3.1, where chance alone expects about half
a column.

The coefficient of `x25` is drawn again each week around 0.15, and its
forty weekly values average 0.186, the truth its line shows. The pooled
fit sees only that average. Its t of 20.3 reads as firm as the t of
`x15`, whose coefficient never moves. [Fit per-block
coefficients](fit-per-block-coefficients.md) reads the weekly values from
the same forty Grams.

## Check that it worked

**When every row weighs 1, the pooled Gram's `weight_sum` counts the rows
the pass read,** 400 in the first example and 20,160 in the second. Its
fit is least squares on those rows, so it matches a least-squares fit on
the rows read into memory, as the recipe of [Keep weekly Grams of
hundreds of features](#keep-weekly-grams-of-hundreds-of-features) checks.

| what you see | what it means | what to do |
|---|---|---|
| `po.gram.merge` refuses parts whose targets differ | under `target_gaps="own_rows"` a block holds a Gram per pattern of gaps | pool the Grams with the same `targets`, or give `target_gaps="pairwise"` where its condition holds |
| the pass runs out of memory | the bank holds a `k²` Gram for every block it has seen | give `group_close="monotone"` and `closed_groups=`, so it holds one week at a time and writes each finished week to a file ([Close groups and read each one's row](close-groups.md), [Read Grams from saved files](read-grams-from-saved-files.md)) |
| `bank.coef()` differs from `po.gram.solve` on the Gram | `bank.coef()` is as of the last solve, and the pass solves nothing | solve the Gram with `po.gram.solve` |

## Go on from the Grams

[Wide fits from Grams](../GRAM.md) explains how a Gram stands in for the
rows, which rows it holds, and what it cannot answer. It opens with a
table from each task on a set of Grams to the how-to that does it.
