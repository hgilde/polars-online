<!-- kind: how-to -->
# Flag joint outliers

This how-to is for anyone who watches several columns of a stream and
wants to flag the rows that are far from the rest once the columns'
correlations are allowed for. Such a row can have every column in range,
with only the combination odd. It uses `ew_cov`'s Mahalanobis distance and
a threshold from the distances seen before each row.

## Prerequisites

These steps need the library and a few columns that move together:

- polars-online installed, and numpy to make the example's data.
- A stream with two or more numeric columns, read as the
  [features](../GLOSSARY.md#the-model-bank-and-what-it-reads) of an
  [`ew_cov`](../MODELS.md#ew_cov--exponentially-weighted-moments) spec.
  The distance reads an inverse of their covariance on every row, which
  takes O(k³) time for `k` columns.
- A [half-life](../GLOSSARY.md#time): how far back the mean and the
  correlations reach. [Time and decay](../explanation/time-and-decay.md)
  says how a model forgets along its clock.

## Flag the rows far from the mean

`mahal` is the row's distance from the mean in standard deviations, with
the correlations allowed for: with one column and a negligible
`precision_prior`, it is `|z|`. Use a quantile of the distances seen
before each row as the threshold: it assumes no distribution.

1. **Add `"mahal"` to the spec's `stats`.** Each row gains `mahal`, read
   before the row is learned, so it is
   [out-of-sample](../explanation/out-of-sample.md).
2. **Give `precision_prior=1e-6`.** `mahal` needs it: a ridge on the
   co-moments, in the features' squared units, that makes their inverse
   solvable on the first rows. It fades as rows arrive, so a small value
   serves.
3. **Give `mahal_quantiles=[0.99]`.** Each row gains `mahal_q0.99`, the
   exponentially weighted 99th percentile of the distances before it, at
   the model's half-life. The row's own distance joins after it is read.
4. **Add the spec to the query with `online.fit_predict([mv])`, unnest
   its output column with Polars' `unnest("mv")`, and run it with
   `collect()`.** Each row then has `mahal` and `mahal_q0.99` as
   columns.
5. **Keep the rows where `mahal > mahal_q0.99`.** About one row in a
   hundred passes: the rows far from the mean once the correlations are
   allowed for.

This code makes two columns that correlate at 0.9 and a third on its own.
At four rows it sets `x0` to 1.5 and `x1` to −1.5: each is in range, but
two columns that move together seldom move apart that far. It flags the
rows and prints the planted four:

```python
import numpy as np
import polars as pl
import polars_online as po

# Two columns that correlate at 0.9, a third on its own, and four rows where only the combination is odd.
rng = np.random.default_rng(7)
n = 5_000
a, b, c = rng.standard_normal((3, n))
x0, x1 = a, 0.9 * a + np.sqrt(1 - 0.9**2) * b
planted = [1_500, 2_500, 3_500, 4_500]
x0[planted], x1[planted] = 1.5, -1.5
pl.DataFrame({"x0": x0, "x1": x1, "x2": c}).write_parquet("joint.parquet")

# Fit the distances, and keep the rows past the threshold.
mv = po.spec.ew_cov(
    "mv", features=["x0", "x1", "x2"], half_life=1_000.0,
    stats=["mean", "std", "mahal"],      # 1. the distance, and each column's own spread to compare
    precision_prior=1e-6,                # 2. needed by mahal; fades as rows arrive
    mahal_quantiles=[0.99],              # 3. adds mahal_q0.99
)
out = pl.scan_parquet("joint.parquet").with_row_index("row").online.fit_predict([mv]).unnest("mv").collect()   # 4. run, and unnest the fields
flagged = out.filter(pl.col("mahal") > pl.col("mahal_q0.99"))   # 5. the rows far from the mean
print(f"flagged {flagged.height} of {out.height} rows")
# Each planted row: its columns in their own standard deviations, and its distance against the threshold.
for r in out.filter(pl.col("row").is_in(planted)).iter_rows(named=True):
    z0, z1 = ((r[f"x{i}"] - r[f"mean_x{i}"]) / r[f"std_x{i}"] for i in (0, 1))
    print(f"row {r['row']}: x0 {z0:+.1f} sd, x1 {z1:+.1f} sd; mahal {r['mahal']:.1f} > {r['mahal_q0.99']:.1f}: "
          f"{r['row'] in flagged['row']}")
```

```text
flagged 53 of 5000 rows
row 1500: x0 +1.5 sd, x1 -1.5 sd; mahal 6.8 > 3.4: True
row 2500: x0 +1.5 sd, x1 -1.5 sd; mahal 7.1 > 3.3: True
row 3500: x0 +1.5 sd, x1 -1.5 sd; mahal 7.0 > 3.4: True
row 4500: x0 +1.5 sd, x1 -1.5 sd; mahal 6.7 > 3.4: True
```

Each planted column sits 1.5 of its standard deviations from its mean,
which a check of one column at a time passes. Its distance is about twice
the threshold, because `x0` and `x1` moved apart where they almost always
move together.

## Check that it worked

**Count the rows flagged: near one in a hundred at `mahal_quantiles=[0.99]`.**
The threshold is a quantile of the stream's own distances, so the share
flagged holds whatever the columns' distribution. To tell a flagged row
from the rest, print its columns in their own standard deviations beside
its `mahal`, as the recipe does.

| what you see | what it means | what to do |
|---|---|---|
| `mahal` and `mahal_q0.99` null on a group's first rows | the model is in its [warm-up](../GLOSSARY.md#warm-up-and-readiness): its statistics are null until `weight_sum` reaches `min_weight`, the number of features plus one by default | read the rows after it; a filter on `>` drops the nulls |
| a `ValueError` naming `mahal_quantiles` and `window_size` | the quantile runs over the whole history, which a [hard window](../GLOSSARY.md#fitting-a-model) does not cut | drop `window_size`, or set the threshold yourself |
| a `ValueError` naming `precision_prior` | `mahal` reads the inverse of the co-moments, which the prior makes solvable | give `precision_prior`, finite and above 0 |

The fields and every parameter are defined in `ew_cov`'s
[docstring](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.ew_cov).
