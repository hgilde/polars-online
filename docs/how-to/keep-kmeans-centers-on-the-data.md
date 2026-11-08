<!-- kind: how-to -->
# Keep `kmeans` centers on the data

This how-to is for anyone who fits `kmeans` to a stream whose clusters
come and go, and wants every center to sit on a cluster that still has
rows. It sets `init`, which places the centers at the start, and
`dead_frac`, which moves a center whose cluster has vanished.

## Prerequisites

These steps need the library, a stream and a few choices:

- polars-online installed, and numpy to make the example's data.
- A stream of numeric
  [feature](../GLOSSARY.md#the-model-bank-and-what-it-reads) columns,
  saved as a file in clock order, to read back with `pl.scan_parquet`.
- The number of clusters, `n_clusters`, and a
  [half-life](../GLOSSARY.md#time): how far back each center's mean
  reaches. [Time and decay](../explanation/time-and-decay.md) says how a
  model forgets along its clock.

## Re-place the center of a cluster that vanished

Use these steps when a cluster can stop and another start while the stream
runs. A center keeps the mean of the rows it was given, so once its
cluster stops, it holds less weight with each half-life. `dead_frac` sets
the weight at which the model gives up on it and moves it to the rows that
sit far from every center.

1. **Leave `init` at its default, `"lloyd"`.** It runs Lloyd's passes from
   ten k-means++ starts and keeps the best. One start can leave one center
   over two clusters, and nothing after seeding can split them, since
   their rows all sit within that center's radius. [Seeding: a model that
   places its starting
   values](../explanation/warm-up-and-readiness.md#seeding-a-model-that-places-its-starting-values)
   explains why, and measures it.
2. **Set `dead_frac` from how fast your clusters change.** A center is
   re-placed once it holds less than `dead_frac` of an equal share of the
   weight, `log2(1/dead_frac)` half-lives after its cluster vanished: 4.3
   half-lives at the default, 0.05, and 2 at 0.25. Increase it when
   regimes change faster than that. It must stay below 0.5.
3. **Build the spec with `po.spec.kmeans`,** giving `n_clusters`,
   `half_life` and `dead_frac`. Give `coef_every=0` too, to write every
   center on every row, as it stands after learning that row.
4. **Run it with `pl.scan_parquet(...).online.fit_predict([km])`, and
   unnest the output with `.online.unnest([km])`.** Each row then has
   `cluster`, the index of the nearest center, read before the row is
   learned, and every center as `coef_cluster<j>_<feature>`.

This code makes three round clusters. At row 4,000 the one at (0, 6)
stops and a new one starts at (6, 6). It fits three centers at a
half-life of 500 rows, and prints the row at which the center that held
the vanished cluster reaches the new one, under three values of
`dead_frac`:

```python
import numpy as np
import polars as pl
import polars_online as po

# Three clusters; at row 4,000 the one at (0, 6) stops and one at (6, 6) starts.
rng = np.random.default_rng(6)
n = 8_000
centers = np.array([[0.0, 0.0], [6.0, 0.0], [0.0, 6.0], [6.0, 6.0]])
blob = rng.integers(0, 3, n)
blob[4_000:] = rng.choice([0, 1, 3], n - 4_000)               # cluster 2 vanishes, cluster 3 is born
xy = centers[blob] + rng.standard_normal((n, 2))
pl.DataFrame({"x": xy[:, 0], "y": xy[:, 1]}).write_parquet("swap.parquet")


# How far center j sits from (x, y), on every row of the output.
def distance(j, x, y):
    return ((pl.col(f"coef_cluster{j}_x") - x) ** 2 + (pl.col(f"coef_cluster{j}_y") - y) ** 2).sqrt()


for dead_frac in (0.0, 0.05, 0.25):                           # 0 turns the rule off; 0.05 is the default
    # 2 and 3. The rule, and every center on every row.
    km = po.spec.kmeans("km", features=["x", "y"], n_clusters=3, half_life=500.0,
                        dead_frac=dead_frac, coef_every=0)
    # 4. Run, and unnest the centers into columns.
    out = pl.scan_parquet("swap.parquet").online.fit_predict([km]).online.unnest([km]).collect()
    # The center nearest (0, 6) on the row before the cluster vanished, and the first row it sits within 0.5 of (6, 6).
    j = min(range(3), key=lambda j: out.slice(3_999, 1).select(distance(j, 0.0, 6.0)).item())
    after = out.with_row_index("row").filter(pl.col("row") >= 4_000)
    moved = after.filter(distance(j, 6.0, 6.0) < 0.5)["row"][0]
    print(f"dead_frac {dead_frac}: center {j} reaches the new cluster at row {moved}")
```

```text
dead_frac 0.0: center 1 reaches the new cluster at row 6091
dead_frac 0.05: center 1 reaches the new cluster at row 6091
dead_frac 0.25: center 1 reaches the new cluster at row 5199
```

At 0.25 the rule re-places the center in one jump, at row 5,199. That is
2.4 half-lives after its cluster vanished, a little later than the
formula's 2, which counts from a center that learns no rows: this one had
drifted toward the new rows, learning some of them. With the rule off, or at the default, the center
drifts the whole way instead, and reaches the new cluster at row 6,091,
before the default's 4.3 half-lives have passed. The Notes of `kmeans`'s
[docstring](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.kmeans)
say why, under *Far rows, split and merge*.

## Check that it worked

Give `coef_every=0` and plot or print each center's coordinates on the
rows after a cluster vanishes, as the recipe does. Every center should
come to sit on a cluster that still has rows.

| what you see | what it means | what to do |
|---|---|---|
| a center stays where a cluster used to be, many half-lives on | the weight it holds has not yet fallen below `dead_frac` of an equal share | increase `dead_frac` |
| a small cluster loses its center whenever a far row arrives | a cluster lighter than `dead_frac / n_clusters` of the stream counts as dead | lower `dead_frac` |
| one center over two clusters from the start | a single seeding start left it there, and the split-merge move cannot see it | leave `init` at `"lloyd"` |
| a `ValueError` naming `dead_frac` | it is below 0 or not finite, 0.5 or more, where the re-placed center starts at the floor itself, or above 0 beside `split_merge=0`, which runs no check for the rule to run at | keep it below 0.5, and leave `split_merge` above 0 |

The fields are defined in [OUTPUTS.md](../OUTPUTS.md#kmeans), and every
parameter in `kmeans`'s
[docstring](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.kmeans),
whose Notes give the measurements behind `dead_frac`'s limit under *Dead
centers*. [`kmeans` in
MODELS.md](../MODELS.md#kmeans--exponentially-weighted-k-means) shows the
split-merge move, which gives a center to a cluster born after seeding,
when no center has died.
