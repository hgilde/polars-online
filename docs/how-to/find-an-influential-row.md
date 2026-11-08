<!-- kind: how-to -->
# Find an influential row

This how-to is for anyone whose fit on a stream may have been moved by a
few bad rows, such as misprints, and who wants to find them. It turns on
each row's influence, finds the rows past a cutoff, and decides what to do
with them.

## Prerequisites

These steps need the library and a fit that reads a row's leverage:

- polars-online installed, and numpy to make the example's data.
- A [spec](../GLOSSARY.md#the-model-bank-and-what-it-reads) of `ewridge`,
  `rls` or `kalman`, the three models that take `emit_influence`.

## Rank the rows by influence

One bad row can move a fit more than a thousand good ones: a row far out
in the features and far off the line moves the coefficients most.
`influence_<t>` is the online form of Belsley, Kuh and Welsch's DFFITS,
the move a row made against the fit that predicted it. [Which row moved
it?](../DIAGNOSTICS.md#which-row-moved-it) defines it and derives its
cutoff.

1. **Turn on `emit_influence=True` in the spec.** Each row gains
   `influence_<t>` for each target `<t>`.
2. **Run the query, and read the rows from where the diagnostic
   starts.** `influence_<t>` is null until the stream has come 95% of the way to
   its steady state, about 4.3 of the fit's half-lives in: from row 867 for the recipe's
   half-life of 200, and the recipe reads from row 1,000. The wait keeps
   out the first rows, which move a young fit by much more.
3. **Flag each row whose `|influence_<t>|` passes
   `4 · sqrt((k + 1) / weight_sum)`** for `k` features, where
   [`weight_sum`](../GLOSSARY.md#warm-up-and-readiness) is the weight the
   model has accumulated, which each row's output carries. A fit with more
   features or a shorter memory reads larger on every row, so the cutoff
   scales with the fit: 0.41 for the recipe's fit, at a half-life of 200
   with two features.
4. **Rank the flagged rows by `|influence_<t>|`, and look at each.**

This code plants three bad prints in 10,000 rows, each with `x0` six
spreads out and `y` six noise sds off the line. It prints the five rows
that moved the fit most, beside the clean rows for scale:

```python
import numpy as np
import polars as pl

import polars_online as po

# 10,000 rows of y = 1.0 x0 + 0.5 x1 + noise, and three bad prints.
rng = np.random.default_rng(7)
n = 10_000
x0, x1, e = rng.standard_normal((3, n))
y = 1.0 * x0 + 0.5 * x1 + e
# The bad prints: a far-out x0 whose y sits far off the line.
bad = [2_500, 6_000, 8_200]
x0[bad] = 6.0  # six spreads out ...
y[bad] = 1.0 * x0[bad] + 0.5 * x1[bad] - 6.0  # ... and six noise sds off the line
pl.DataFrame({"x0": x0, "x1": x1, "y": y}).write_parquet("prints.parquet")
prints = pl.scan_parquet("prints.parquet").with_row_index("row")

# 1. A ridge that reports each row's influence, with its coefficients on every row.
model = po.spec.ewridge(
    "m", targets=["y"], features=["x0", "x1"], half_life=200.0,
    emit_influence=True,  # influence_y: how far the row moved the fit
    coef_every=0, max_rows_between_solves=1,  # solve and write the coefficients on every row
)
# 2. Run the query, from row 1,000: past where influence_y starts.
out = prints.online.fit_predict([model]).online.unnest([model]).collect().slice(1_000)
moved = out.with_columns(step=pl.col("coef_y_x0").diff().abs())  # how far x0's coefficient moved

# 3 and 4. The rows that moved the fit most, beside the clean rows for scale.
print("the five rows that moved the fit most:")
top = moved.sort(pl.col("influence_y").abs(), descending=True, nulls_last=True).head(5)
for row, influence, step in top.select("row", "influence_y", "step").iter_rows():
    print(f"  row {row:>5}: influence {influence:+.2f}, x0's coefficient moved by {step:.3f}")
clean = moved.filter(~pl.col("row").is_in(bad))["influence_y"].abs()
print(f"every other row: |influence| at most {clean.max():.2f}, 99.9th percentile {clean.quantile(0.999):.2f}")
```

```text
the five rows that moved the fit most:
  row  8200: influence -1.63, x0's coefficient moved by 0.122
  row  6000: influence -1.55, x0's coefficient moved by 0.110
  row  2500: influence -1.49, x0's coefficient moved by 0.114
  row  2360: influence -0.55, x0's coefficient moved by 0.035
  row  3593: influence -0.43, x0's coefficient moved by 0.035
every other row: |influence| at most 0.55, 99.9th percentile 0.30
```

The three planted rows rank first, each at an influence near −1.5, and
each moved `x0`'s coefficient by 0.11 to 0.12 in one row. The next row
reads −0.55. Among the 8,997 clean rows from row 1,000 on, the largest
reads 0.55 and the 99.9th percentile 0.30. Step 3's cutoff of 0.41 flags
every planted row and two clean rows, rows 2,360 and 3,593.

## Decide what a flagged row is

Find the rows, then decide what each one is:

| the row is | what to do |
|---|---|
| a print error | drop it with a filter in the query before the bank |
| a real event the model should keep | keep it |
| real but rare, and you want no one row to move the fit far | fit `huber`, whose loss turns linear past `huber_delta` |

## Check that it worked

Run the query again after the fix, and rank the rows again. A clean
stream's largest `|influence_<t>|` sits near the cutoff of step 3. Over
many clean streams at half-lives of 20 to 1,000 rows and 2 to 10
features, four times `sqrt((k + 1) / weight_sum)` flagged 0.001% to 0.014%
of the rows. One stream can pass more: the recipe's flagged 2 of 8,997, or
0.022%.

| what you see | what it means | what to do |
|---|---|---|
| many flagged rows at a fixed cutoff, such as 0.5 | the fit's memory is short, or it has many features, so every row reads larger: 0.5 flagged 7.3% of the clean rows of a fit at a half-life of 20 with three features | use the cutoff of step 3, which scales with the fit |
