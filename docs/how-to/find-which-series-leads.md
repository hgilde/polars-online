<!-- kind: how-to -->
# Find which series leads

This how-to is for anyone who wants to know whether one column of a
stream moves before another, and by how many rows. It uses `ew_cov`'s
lagged correlations, which pair each column with every column a set
number of rows back.

## Prerequisites

These steps need the library and the columns to compare:

- polars-online installed, and numpy to make the example's data.
- A stream with two or more numeric columns, read as the
  [features](../GLOSSARY.md#the-model-bank-and-what-it-reads) of an
  [`ew_cov`](../MODELS.md#ew_cov--exponentially-weighted-moments) spec,
  each [group](../GLOSSARY.md#the-model-bank-and-what-it-reads) in row
  order. For series that tick at their own times, put them on one grid
  first, as [Align series that tick at their own
  times](align-series-that-tick-at-their-own-times.md) shows.
- A [half-life](../GLOSSARY.md#time), or `half_life=float("inf")` for
  every row to weigh the same. [Time and
  decay](../explanation/time-and-decay.md) says how a model forgets along
  its clock.
- The lags to try, as numbers of rows.

## Read the lagged correlations

A lagged correlation matrix is not symmetric: `a` leading `b` is not `b`
leading `a`. So the model writes one correlation for each lag and each
*ordered* pair. `lag_corr_l<l>_<a>_<b>` is the correlation of `a` on a row
with `b` `l` rows before it, and a large value there means `b` leads `a`
by `l` rows. The row's deviation and the lagged row's are both taken
from the mean before the current row, as
[`ew_cov`'s update](../MODELS.md#ew_cov--exponentially-weighted-moments)
defines them.

1. **Give `lags`, the lags to try, as a list that increases.** A lag counts
   the rows the model has learned in the group, not
   [clock units](../GLOSSARY.md#time), and each is at least 1.
2. **Add `"lag_corr"` to the spec's `stats`.** Each row gains
   `lag_corr_l<l>_<a>_<b>` for each lag and each ordered pair, the pairs of
   a column with itself included: `k²` fields a lag for `k` columns.
3. **Add the spec to the query with `online.fit_predict([mv])`, run it
   with `collect()`, and read each pair's correlations across the lags.**
   The lag where `lag_corr_l<l>_<a>_<b>` peaks is how far `b` leads `a`.
   Read `lag_corr_l<l>_<b>_<a>` too: it peaks where `a` leads `b`.

This code plants a series `follow` that repeats `lead` three rows later,
with noise, so the two correlate at 0.8 three rows apart and not at all on
the same row. It tries lags of one to five rows, and prints both
directions at each lag, read before the last row:

```python
import numpy as np
import polars as pl
import polars_online as po

# follow repeats lead three rows later, with noise: the truth is 0.8 at a lag of 3.
rng = np.random.default_rng(9)
n = 5_000
lead = rng.standard_normal(n + 3)
follow = 0.8 * lead[:-3] + 0.6 * rng.standard_normal(n)
pl.DataFrame({"lead": lead[3:], "follow": follow}).write_parquet("leadlag.parquet")

# The correlations at each lag, read before the last row.
mv = po.spec.ew_cov(
    "mv", features=["lead", "follow"], half_life=float("inf"),
    lags=[1, 2, 3, 4, 5],              # 1. the lags to try, in rows
    stats=["corr", "lag_corr"],        # 2. the correlation on the same row, and at each lag
)
last = pl.scan_parquet("leadlag.parquet").online.fit_predict([mv]).unnest("mv").collect().row(-1, named=True)
# 3. Both directions at each lag.
print(f"on the same row: {last['corr_lead_follow']:+.2f}")
for lag in [1, 2, 3, 4, 5]:
    print(f"lag {lag}: follow with lead {lag} rows back {last[f'lag_corr_l{lag}_follow_lead']:+.2f}, "
          f"lead with follow {lag} rows back {last[f'lag_corr_l{lag}_lead_follow']:+.2f}")
```

```text
on the same row: +0.00
lag 1: follow with lead 1 rows back +0.01, lead with follow 1 rows back -0.02
lag 2: follow with lead 2 rows back -0.00, lead with follow 2 rows back +0.01
lag 3: follow with lead 3 rows back +0.80, lead with follow 3 rows back +0.02
lag 4: follow with lead 4 rows back +0.00, lead with follow 4 rows back +0.00
lag 5: follow with lead 5 rows back -0.00, lead with follow 5 rows back -0.00
```

`lead` leads `follow` by three rows, at the planted 0.8, and nothing leads
`lead`. The correlation on the same row, the one a plain `corr` reads,
finds no relation at all.

## Check that it worked

**A lead shows as a peak at one lag in one direction.** In the recipe,
`lag_corr_l3_follow_lead` stands out from every other lag and from the
other direction. Each column's correlation with its own past is in the
`lag_corr_l<l>_<a>_<a>` fields beside them.

| what you see | what it means | what to do |
|---|---|---|
| `lag_corr_*` null on a group's first rows | the model is in its [warm-up](../GLOSSARY.md#warm-up-and-readiness), until `weight_sum` reaches `min_weight`, the number of features plus one by default | read the rows after it |
| `lag_corr_l<l>_*` reads 0.0, not null, on early rows | the group has learned no more than `l` rows, so the lag has no product yet | read the rows after the longest lag |
| `lag_corr_*` after a gap or a session change | a [clock break](../GLOSSARY.md#time), a step that `gap_cap` (the largest clock step the model decays by) shortened or a session change, emptied the model's record of past rows, since "`l` rows back" is not `l` rows ago there. The rows after it add no lagged product until `l` rows have been learned again | expected: read the lags once the rows after the break have filled them |
| a `ValueError` naming `lags` and `window_size` | lagged moments run over a history a [hard window](../GLOSSARY.md#fitting-a-model) does not cut | drop `window_size` |

The fields, under `lags`, and every parameter are defined in `ew_cov`'s
[docstring](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.ew_cov).
