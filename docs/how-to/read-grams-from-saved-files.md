<!-- kind: how-to -->
# Read Grams from saved files

This how-to is for anyone who wants to fit from the
[Grams](../GLOSSARY.md#fitting-a-model) of a pass in a later program,
one that reads no rows and runs no model. It writes each finished week's
Gram to a parquet file and the last week's to a saved state, then reads
both back with `po.gram`, and with numpy alone.

## Prerequisites

These steps need the library, numpy and a column that names each block,
the rows whose Gram is kept apart (here a week):

- polars-online installed with numpy, an optional extra, which
  `pip install "polars-online[numpy]"` adds. The recipe builds its own
  data.
- A timestamp to make each row's week from, in the query. The week must be
  an integer, text or categorical column for the bank to close it, so the recipe
  makes it the Monday's day number.
- An `ewridge` spec that keeps a Gram per week, as [Keep weekly Grams of
  hundreds of
  features](fit-weekly-grams-in-one-pass.md#keep-weekly-grams-of-hundreds-of-features)
  sets one up.

[Wide fits from Grams](../GRAM.md) explains why a fit can read a Gram in
place of its rows.

## Write each finished week's Gram to a file

Use this when a later program should fit from the Grams without reading
the rows. `group_close="monotone"` closes each week once the next begins,
and `closed_groups=` on `fit` takes each closed week out of the bank and
writes them all to the file when the pass ends. Until then the closed
weeks wait in the process's memory. To hold only the weeks of one chunk
at a time, loop over `bank.fit_predict_batches(lf)` and write
`bank.closed_groups()` to a file of its own after each chunk; read the
files back together with `pl.read_parquet` on a glob, such as
`"blocks-*.parquet"`.

1. **Make the week in the query as the Monday's day number,** with
   `pl.col("ts").dt.truncate("1w").dt.date().cast(pl.Int32)`. A
   [group](../GLOSSARY.md#the-model-bank-and-what-it-reads) the bank
   closes must be an integer, text or categorical column.
2. **Give the spec `group="week"` and `group_close="monotone"`.**
3. **Run the pass with `bank.fit(lf, closed_groups="blocks.parquet")`.**
   Each closed week's Gram becomes a row of that file, written when the
   pass ends.
4. **Save the state with `bank.save`.** The last week never closes, since
   no row shows it has finished, so its Gram stays in the state.
   [Close groups and read each one's row](close-groups.md) has the rules
   for closing groups.

This code makes a stream of 40 weeks with 200 features, of which six
matter, and runs the pass. It builds its own data:

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

# 1. The week, made in the query as the Monday's day number.
lf = pl.scan_parquet("weeks.parquet").with_columns(
    week=pl.col("ts").dt.truncate("1w").dt.date().cast(pl.Int32)  # the Monday, as days since 1970
)
# 2. A Gram per week, each closed once the next week begins.
spec = po.spec.ewridge(
    "blocks", targets=["y"], features=features, group="week", group_close="monotone",
    half_life=float("inf"), solve_every=float("inf"), target_gaps="pairwise", gram_block_rows=256,
)
bank = po.ModelBank([spec])
bank.fit(lf, closed_groups="blocks.parquet")  # 3. a row per closed week, written when the pass ends
bank.save("blocks.state")  # 4. the state holds the last week, which no later row closed
```

## Read the Grams back without running a model

Use this in the program that fits: it reads the two files and no rows.

1. **Read the file of closed weeks with `pl.read_parquet`,** and turn each
   row into a Gram with `po.gram.from_row`. It returns the dict
   `bank.gram()` returns, with the same numbers to the bit.
2. **Load the state with `po.ModelBank.load`, and read the open week with
   `.gram(name)`.**
3. **Pool the weeks with `po.gram.merge`, and fit with `po.gram.solve`.**

This code reads the 39 closed weeks and the open one, and fits from all
40. It continues from the code before:

```python
# 1. The closed weeks, a Gram per row of the file.
closed = pl.read_parquet("blocks.parquet")
blocks = [po.gram.from_row(row) for row in closed.iter_rows(named=True)]
# 2. The open week, from the saved state.
blocks += po.ModelBank.load("blocks.state").gram("blocks")
print(f"{closed.height} weeks from blocks.parquet and {len(blocks) - closed.height} from blocks.state")
# 3. Pool the 40 weeks and fit; each planted coefficient beside its estimate.
coef = po.gram.solve(po.gram.merge(blocks))
print("x0, x5 and x10, planted 0.5, 0.4, 0.3, found " + ", ".join(f"{coef[i]:.3f}" for i in (1, 6, 11)))
```

```text
39 weeks from blocks.parquet and 1 from blocks.state
x0, x5 and x10, planted 0.5, 0.4, 0.3, found 0.489, 0.389, 0.297
```

The file holds 39 weeks, and the state the fortieth. Pooled, they give
the coefficients the same stream gives in [Keep weekly Grams of hundreds
of
features](fit-weekly-grams-in-one-pass.md#keep-weekly-grams-of-hundreds-of-features),
where the pass kept every week in memory.

## Read a closed row with numpy alone

Use this in a program that has numpy but not this library. A program in
another language that cannot read parquet's lists can read a whole state
as JSON instead, written by `bank.write_json(path)`, as [Read a state
without this library](read-a-state-without-this-library.md) shows. A closed row holds a
Gram's arrays laid out flat:

| field | what it holds | how it is laid out |
|---|---|---|
| `columns` | the names: `"intercept"` first, then the features | a list |
| `comoments` | the co-moment matrix | its upper triangle, row by row |
| `cross_moments` | each target's cross-moments; at the intercept, the target's mean | one list, the targets' rows one after another |
| `cross_centered` | each target's centered cross-moments | one list, the targets' rows one after another |
| `means_by_target` | the column means each target's cross-moments are centered at | one list, the targets' rows one after another |

With one target, as here, each of the last three lists is that target's.

1. **Lay `comoments` into the upper triangle,** with
   `numpy.triu_indices`, and mirror it into the lower one.
2. **Solve the centered system for the slopes:** the co-moments of the
   features, against `cross_centered` past the intercept.
3. **Take the intercept from the means:** the target's mean less the
   column means times the slopes.

This code solves the first week's row with numpy and compares the result
with `po.gram.solve`. It continues from the code before:

```python
# 1. Unpack the triangle, and mirror it.
row = closed.row(0, named=True)
m = len(row["columns"])  # the intercept and the 200 features
cov = np.zeros((m, m))
cov[np.triu_indices(m)] = row["comoments"]  # the upper triangle, row by row
cov += np.triu(cov, 1).T
# 2 and 3. The slopes from the centered system, and the intercept from the means.
slopes = np.linalg.solve(cov[1:, 1:], np.asarray(row["cross_centered"])[1:])
intercept = row["cross_moments"][0] - np.asarray(row["means_by_target"])[1:] @ slopes
same = np.allclose(np.r_[intercept, slopes], po.gram.solve(blocks[0]), rtol=0, atol=1e-10)
print(f"numpy alone and po.gram agree on the first week within 1e-10: {same}")
```

```text
numpy alone and po.gram agree on the first week within 1e-10: True
```

## Check the result

**The weeks from the file and from the state add up to the weeks the
stream holds,** 39 and 1 of 40 here, and their pooled fit finds the
planted coefficients. numpy's solve of a row agrees with `po.gram.solve`
within 1e-10.

| what you see | what it means | what to do |
|---|---|---|
| a `WrongTypeError` that the spec's `group_close` needs a group column it can order | the week is a datetime, and a closed group's key must be an integer, text or categorical column | make the week the Monday's day number, with `.dt.date().cast(pl.Int32)` |
| one week fewer than the stream holds | the last week never closes, so it is in the state, not the file | read it from the saved state with `po.ModelBank.load(path).gram(name)` |
| numpy's fit differs from `po.gram.solve` | the triangle is unpacked in another order, or not mirrored | fill `numpy.triu_indices(m)`, which runs row by row, and add the strict upper triangle's transpose |
| the solve reads the wrong target's numbers | with several targets each list holds every target's rows, one after another | take the `k` entries of each target in turn |
