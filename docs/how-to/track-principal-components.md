<!-- kind: how-to -->
# Track principal components

This how-to is for anyone who wants the main direction several columns of
a stream move in together, as it stands at each row. It reads the
direction's loadings, its share of the variance and each row's coordinate
along it. It uses
`ew_cov`'s principal components, refreshed on a schedule.

## Prerequisites

These steps need the library and a few columns that share a common move:

- polars-online installed, and numpy to make the example's data.
- A stream with two or more numeric columns, read as the
  [features](../GLOSSARY.md#the-model-bank-and-what-it-reads) of an
  [`ew_cov`](../MODELS.md#ew_cov--exponentially-weighted-moments) spec.
- A [half-life](../GLOSSARY.md#time): how far back the covariance the
  components are read from reaches. [Time and
  decay](../explanation/time-and-decay.md) says how a model forgets along
  its clock.
- Optionally a [clock](../GLOSSARY.md#time) column, as the recipe's `t`,
  which needs a `gap_cap`: the largest clock step the model decays by.

## Read the components as the stream runs

The components are the eigenvectors of the covariance the model keeps.
Each eigendecomposition takes O(k³) time for `k` columns, so the model
refreshes them on a schedule, and the rows between refreshes read their
scores from the last loadings.

1. **Give `pca=1` for the top component,** or a larger count, at most the
   number of features. Each row gains, per component `j`, `pc<j>_var` (its
   eigenvalue), `pc<j>_share` (its share of the total variance),
   `pc<j>_loading_<feature>` (its unit loading on each column) and
   `pc<j>_score` (the row's coordinate along it).
2. **Give `pca_every` to refresh the components every so many
   [clock units](../GLOSSARY.md#time).** Without a clock, a clock unit is a
   row. On a clock of times it takes a duration, such as `"5m"`, and `0`
   refreshes on every row. With neither `pca_every` nor
   `max_rows_between_pcas`, the components refresh on every row.
3. **Give `max_rows_between_pcas` to cap the rows between refreshes,**
   where a burst of rows can arrive within one `pca_every`. Whichever of the
   two comes first refreshes.
4. **Add the spec to the query with `online.fit_predict([mv])`.** The
   rows gain the spec's output column, `mv`, which holds its fields.
5. **Unnest `mv` with Polars' `unnest("mv")`.** Its fields become columns.
6. **Run the query with `collect()`.** Each row then has the `pc0_*`
   fields as columns. The first refresh makes each component's largest loading in magnitude positive, and each
   later refresh keeps a component within 90° of its previous loadings.
   So a component and its score never flip sign from one refresh to the next.

This code plants one factor that moves four columns with known loadings,
and adds noise of its own to each column, on a clock `t` that advances
about one unit a row. It tracks the top component, refreshed every 50
units of `t`, and prints the planted loadings and share beside
the ones read before the last row:

```python
import numpy as np
import polars as pl
import polars_online as po

# Four columns moved by one factor with known loadings, each with noise of its own.
rng = np.random.default_rng(8)
n = 6_000
names = ["s0", "s1", "s2", "s3"]
loading = np.array([0.6, 0.5, 0.5, 0.37]) / np.linalg.norm([0.6, 0.5, 0.5, 0.37])   # the truth, of unit length
factor = 3.0 * rng.standard_normal(n)                       # the factor's variance, 9, against each column's noise of 1
x = factor[:, None] * loading + rng.standard_normal((n, 4))
t = np.cumsum(rng.exponential(1.0, n))                     # a clock: about one unit a row, at uneven steps
pl.DataFrame(dict(zip(names, x.T))).with_columns(factor=factor, t=t).write_parquet("factor.parquet")

# Track the top component, and read it before the last row.
mv = po.spec.ew_cov(
    "mv", features=names, clock="t", gap_cap=10.0, half_life=2_000.0,
    stats=[],                    # no moments on each row: only the component
    pca=1,                       # 1. the top component
    pca_every=50.0,              # 2. refreshed every 50 units of t
    max_rows_between_pcas=100,   # 3. and at least every 100 rows, should rows bunch on the clock
)
out = pl.scan_parquet("factor.parquet").online.fit_predict([mv]).unnest("mv").collect()   # 4 to 6. add the spec, unnest pc0_*, run
last = out.row(-1, named=True)
print("planted loadings " + " ".join(f"{v:.2f}" for v in loading) + f", share {10 / 13:.2f}")
print("found loadings   " + " ".join(f"{last[f'pc0_loading_{s}']:.2f}" for s in names) + f", share {last['pc0_share']:.2f}")
print(f"the scores correlate with the factor at {out.select(pl.corr('pc0_score', 'factor')).item():.2f}")
```

```text
planted loadings 0.60 0.50 0.50 0.37, share 0.77
found loadings   0.61 0.49 0.50 0.37, share 0.77
the scores correlate with the factor at 0.95
```

The planted share is the factor's variance and one unit of noise along it,
10, over the total, 13. The scores miss the factor by the noise each
column adds along the loadings.

## Check that it worked

**On data with a planted factor, compare the loadings, the share and the
scores with the truth, as the recipe does.** On your own data, compare
the loadings across refreshes: a component read from the covariance of
the last few half-lives changes as that covariance does, and no faster.
Their sign checks nothing, since the refresh rule keeps it. The scores between two
refreshes use the same loadings, so a row's score does not depend on how
the rows were chunked.

| what you see | what it means | what to do |
|---|---|---|
| the `pc0_*` fields null on a group's first rows | the model is in its [warm-up](../GLOSSARY.md#warm-up-and-readiness): its statistics are null until `weight_sum` reaches `min_weight`, the number of features plus one by default | read the rows after it |
| the run slows as the columns grow | each refresh takes O(k³) time | increase `pca_every`, and `max_rows_between_pcas` too if you gave it: whichever comes first refreshes |
| a `ValueError` naming `pca_every` or `max_rows_between_pcas` | either was given without `pca` | give `pca` |

The fields, in the table under `pca`, and every parameter are defined in
`ew_cov`'s
[docstring](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.ew_cov).
