<!-- kind: how-to -->
# Choose `micro`'s `eps`

This how-to is for anyone who clusters a stream with `micro` and must
choose `eps`, its one required setting. `eps` bounds the spread of the
small summaries the clusters are built from, and a wrong value shows in
the output in one of two ways.

## Prerequisites

These steps need the library, a stream and a first guess:

- polars-online installed, and numpy to make the example's data.
- A stream of numeric
  [feature](../GLOSSARY.md#the-model-bank-and-what-it-reads) columns,
  saved as a file in clock order, to read back with `pl.scan_parquet`.
- A [half-life](../GLOSSARY.md#time): how long a summary whose rows stop
  lingers. [Time and decay](../explanation/time-and-decay.md) says how a
  model forgets along its clock.
- [`micro` in
  MODELS.md](../MODELS.md#micro--density-based-clustering-any-shape),
  which defines the summaries and how they are linked into clusters.

## Choose `eps`

`micro` keeps summaries of the rows, each a decayed weight, a center and a
radius. A summary takes a row only while its radius stays within
`eps·√p`, for `p` features, so one value of `eps` means the same at any
`p`. Under the default `standardize=True`, `eps` is in units of each
feature's exponentially weighted standard deviation. A summary is
established once its weight reaches `beta_mu` rows of the stream's mean
weight (default 3), and only established summaries form clusters.

1. **Start from about 0.07 for shapes in two dimensions, and 0.3 for
   well-separated Gaussian clusters in twenty.** Measured at 20,000 rows,
   moons at an `eps` of 0.07, rings at 0.1 and five Gaussians in twenty
   dimensions at 0.3 were each recovered with an adjusted Rand index of
   1.000.
2. **Build the spec with `po.spec.micro`,** giving `features`,
   `half_life` and `eps`.
3. **Run it on a stretch of the stream with
   `pl.scan_parquet(...).online.fit_predict([mc])`.** The rows gain the
   spec's output column, `mc`, whose fields Polars' `unnest("mc")` spreads
   into columns.
4. **Read the last rows' `n_clusters`, `n_micro` and `outlier`.**
   `n_clusters` is the live clusters, `n_micro` the live summaries, and
   `outlier` whether no established summary took the row.
5. **If nearly every row is an `outlier`, increase `eps`.** No summary
   reaches `beta_mu` before a prune drops it.
6. **If `n_micro` is about the number of clusters you expect and
   `n_clusters` is lower, lower `eps`, or set `macro_link=2`.** Without
   `macro_link`, the link distance is derived from the summaries'
   spacing, so a few coarse summaries bridge each other. `macro_link=2` links only summaries that
   touch.

This code makes three round clusters, 5 apart, of spread 0.3, and fits
`micro` at an `eps` too small, one that fits, and one too coarse, which it
fits again with `macro_link=2`. It prints what each finds on the last 1,000 rows:

```python
import numpy as np
import polars as pl
import polars_online as po

# Three round clusters of spread 0.3, 5 apart.
rng = np.random.default_rng(6)
n = 6_000
centers = np.array([[0.0, 0.0], [5.0, 0.0], [0.0, 5.0]])
xy = centers[rng.integers(0, 3, n)] + 0.3 * rng.standard_normal((n, 2))
pl.DataFrame({"a": xy[:, 0], "b": xy[:, 1]}).write_parquet("blobs3.parquet")

# 2 to 4. Build and fit each eps, and read the last rows' clusters, summaries and outliers.
for eps, macro_link in ((0.002, None), (0.07, None), (0.3, None), (0.3, 2.0)):
    mc = po.spec.micro("mc", features=["a", "b"], half_life=1_000.0, eps=eps, macro_link=macro_link)
    late = pl.scan_parquet("blobs3.parquet").online.fit_predict([mc]).collect().unnest("mc").tail(1_000)
    print(f"eps {eps}, macro_link {macro_link}: clusters {late['n_clusters'][-1]}, "
          f"summaries {late['n_micro'][-1]}, outliers {late['outlier'].mean():.0%}")
```

```text
eps 0.002, macro_link None: clusters 0, summaries 105, outliers 100%
eps 0.07, macro_link None: clusters 3, summaries 23, outliers 1%
eps 0.3, macro_link None: clusters 1, summaries 3, outliers 0%
eps 0.3, macro_link 2.0: clusters 3, summaries 3, outliers 0%
```

At 0.002 the summaries are too small to be established, so every row is
an outlier. At 0.3 each cluster is one summary, and the derived link joins
all three into one cluster. `macro_link=2` separates them again.

## Check that it worked

On a stretch where you know the clusters, count them in `n_clusters`, and
check that few rows of a cluster are flagged `outlier`. On four clusters
with one row in twenty scattered over a box three times their extent, 94%
of the scattered rows were flagged and 0.3% of the rest.

| what you see | what it means | what to do |
|---|---|---|
| nearly every row is an `outlier`, and `cluster` stays null | `eps` is too small: no summary reaches `beta_mu` before it is pruned | increase `eps` |
| `n_micro` is about the number of clusters, and fewer clusters than you planted | `eps` is too coarse: the derived link, set from the summaries' spacing, bridges them | lower `eps`, or set `macro_link=2` |
| a `TypeError` naming `eps` | `eps` is required, and was left out | give it, starting from step 1's values |
| a `ValueError` naming `eps` | `eps` is 0, below 0 or not finite | give a finite value above 0 |

The fields are defined in [OUTPUTS.md](../OUTPUTS.md#micro), and every
parameter, `beta_mu` and `macro_link` among them, in `micro`'s
[docstring](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.micro).
