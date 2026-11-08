<!-- kind: how-to -->
# Store a compact Gram

This how-to is for anyone who keeps many wide
[Grams](../GLOSSARY.md#fitting-a-model), such as one a week at thousands
of features, and wants them in fewer bytes. You will check that your fit
can bear float32's rounding, then ask `bank.gram()` for float32, the upper
triangle, or both.

## Prerequisites

These steps need the library, numpy and a pass that leaves Grams:

- polars-online installed with numpy, an optional extra, which
  `pip install "polars-online[numpy]"` adds. The recipe builds its own
  data.
- A bank whose `ewridge`, `lasso` or `ew_cov` spec keeps a Gram per block
  (the rows of one value of the spec's `group` column, here a week),
  as [Keep weekly Grams of hundreds of
  features](fit-weekly-grams-in-one-pass.md#keep-weekly-grams-of-hundreds-of-features)
  sets one up.

`comoments` (and, for an `ew_cov` with `lags`, `lag_comoments`) are the
arrays of a Gram that grow as `k²`: `comoments` is 800 MB a block in
float64 at 10,000 columns. [A compact Gram](../GRAM.md#a-compact-gram)
explains how far each form can move a fit, and why.

## Choose a form

Each form shapes only what `gram()` returns: the state itself stays
float64 and whole, and `po.gram` reads every form in float64. float32 and
the upper triangle each take about half the bytes of `comoments`, and
both together a quarter. `lag_comoments` is never packed, since a lagged
matrix is not symmetric, so with both options together it takes half its bytes, from float32 alone.

| `bank.gram(...)` | what it loses | what it keeps |
|---|---|---|
| `dtype="float32"` | each entry is rounded to nearest, within a relative `2⁻²⁴`, about 6e-8, of its float64 value. An entry past float32's largest number, about 3.4e38, or nonzero below its smallest normal number, about 1.2e-38, is refused with a `ValueError` naming its column. A column spread 1e20 wide, or 1e-20, gets there, since its variance squares the spread | every other array, in float64: the means, from which `merge` forms its spread term, and each target's moments |
| `layout="packed"` | nothing: the model keeps the matrix symmetric, so the lower triangle is the upper one mirrored, bit for bit | every entry, bit for bit |

1. **Pool the blocks and fit in float64,** with `po.gram.merge` and
   `po.gram.solve`, as the reference.
2. **Compute the condition number `κ` of the features' correlation
   matrix,** with
   `np.linalg.cond(po.gram.correlation(g)[1:, 1:])`. Row and column 0 are
   the intercept's, which are `nan`.
3. **Give `dtype="float32"` when `κ` times 6e-8 is below the precision the
   fit needs.** Rounding moves a fit by about that much at the worst, and
   usually by far less.
4. **Give `layout="packed"` in either dtype.** It halves the bytes and
   loses nothing.
5. **Read the compact Grams with `po.gram` as any other Grams.**

This code builds 40 weeks of 200 features, solves the forty weeks from
each form, and then asks for a float32 Gram of a column spread 1e20 wide:

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

# One pass over the file leaves a Gram per week. The week is a column made in the query.
lf = pl.scan_parquet("weeks.parquet").with_columns(week=pl.col("ts").dt.truncate("1w"))
spec = po.spec.ewridge(
    "blocks", targets=["y"], features=features, group="week",
    half_life=float("inf"), solve_every=float("inf"), target_gaps="pairwise", gram_block_rows=256,
)
bank = po.ModelBank([spec])
bank.fit(lf)  # learns from every row, and keeps only the state
blocks = bank.gram("blocks")  # one dict per week, oldest first

# 1 and 2. The float64 fit as the reference, and the condition number that bounds the loss.
pooled = po.gram.merge(blocks)
reference = po.gram.solve(pooled)
r = po.gram.correlation(pooled)  # row and column 0, the constant intercept's, are nan
kappa = np.linalg.cond(r[1:, 1:])  # of the features' correlation matrix
print(f"condition number {kappa:.1f}; float32's rounding times it, {kappa * 2**-24:.1e}")
# 3 to 5. Each dtype and layout: its bytes a week, and how far the fit moves.
for dtype in ("float64", "float32"):
    for layout in ("full", "packed"):
        parts = bank.gram("blocks", dtype=dtype, layout=layout)
        coef = po.gram.solve(po.gram.merge(parts))  # merged and solved in float64
        moved = np.abs(coef - reference).max() / np.abs(reference).max()
        size = parts[0]["comoments"].nbytes
        print(f"{dtype} {layout:>6}: {size:>7,} bytes a week; the fit moved by {moved:.0e} of its largest coefficient")

# A column spread 1e20 wide has a variance that float32 cannot hold.
pl.DataFrame({"a": 1e20 * rng.standard_normal(1_000), "y": rng.standard_normal(1_000)}).write_parquet("huge.parquet")
huge = po.spec.ewridge(
    "huge", targets=["y"], features=["a"], half_life=float("inf"), solve_every=float("inf")
)
bank = po.ModelBank([huge])
bank.fit(pl.scan_parquet("huge.parquet"))
# Asking for it in float32 is refused, with the column named.
try:
    bank.gram("huge", dtype="float32")
except ValueError as e:
    print(e)
```

```text
condition number 7.8; float32's rounding times it, 4.7e-07
float64   full: 323,208 bytes a week; the fit moved by 0e+00 of its largest coefficient
float64 packed: 162,408 bytes a week; the fit moved by 0e+00 of its largest coefficient
float32   full: 161,604 bytes a week; the fit moved by 7e-09 of its largest coefficient
float32 packed:  81,204 bytes a week; the fit moved by 7e-09 of its largest coefficient
ModelBank.gram(dtype="float32"): the co-moment of "a" with itself is 1.03e40, larger in size than float32's largest number, 3.40e38, where it reads inf. Change the scale of "a" before the bank learns the rows, or leave dtype at its default, "float64".
```

The packed float64 form solves to the same bits as the full one. In
float32 the fit moves by 7e-9 of its largest coefficient, far inside the
4.7e-7 its condition number of 7.8 allows. Both together are a quarter of
the bytes, 81,204 a week. The column spread 1e20 wide has a variance of
1.03e40, past float32's range, and `gram()` refuses to round it to
`inf`.

## Check the result

**Solve the pooled compact Grams and compare the fit with the float64
reference,** as the recipe does: the largest change, over the largest
coefficient, should be within `κ` times 6e-8.

| what you see | what it means | what to do |
|---|---|---|
| a `ValueError` from `gram(dtype="float32")` that names a column | that column's co-moment is past float32's range, or nonzero below its smallest normal number | change the column's scale before the bank learns the rows, or leave `dtype` at `"float64"` |
| the float32 fit moves by more than the fit can bear | `κ` is large, so rounding is magnified | keep float64, and use `layout="packed"` alone; [Check that a fit is determined](check-a-fit-is-determined.md) finds the columns behind a large `κ` |
