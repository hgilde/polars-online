<!-- kind: how-to -->
# Choose a ridge and a half-life

This how-to is for anyone who fits `ewridge` and wants to choose its
`ridge`, alone or together with its `half_life`, from the stream's own
out-of-sample errors with `po.eval.grid_search`.

## Prerequisites

These steps need the library, a stream in a file and a spec:

- polars-online and numpy installed. Each recipe generates its own data.
- The stream saved as a file in clock order, to read back with
  `pl.scan_parquet` as a [query](../GLOSSARY.md#the-model-bank-and-what-it-reads)
  (a Polars `LazyFrame`).
- An `ewridge` [spec](../GLOSSARY.md#the-model-bank-and-what-it-reads),
  and the values to try: a [grid](../GLOSSARY.md#grids).
  [Choose a model's settings from the stream's own
  errors](../CALIBRATION.md#choose-a-half-life-with-poevalgrid_search)
  gives the steps of the search and why the choice is scored on rows it
  had not seen.

## Choose a ridge

**`ewridge`'s `ridge` is in the features' squared units, so a ridge
chosen on one set of features carries to features of another scale only
under `standardize=True`.** The update equation is in [its entry in
MODELS.md](../MODELS.md#ewridge--exponentially-weighted-ridge-regression).

1. **Give the spec a list of ridges, spaced by factors of about 3.** The
   spec solves every ridge from one [Gram](../GLOSSARY.md#fitting-a-model).
2. **If you will carry the ridge to features of another scale, set
   `standardize=True`.** Otherwise a ridge chosen on one set of features does
   not shrink as hard on features ten times as large.
3. **Pass the query and the spec to `po.eval.grid_search`, one spec a
   call.** It returns a row for each ridge, with a `ridge` column, and a
   row for the choice. It ranks and chooses across every spec it is given,
   so specs passed together compete as one grid.
4. **Run the ridge ranked 1 on the rows to come.**
5. **Report the error of the choice,** from `table.filter(kind="choice")`.

This code searches the ridge for forty features with small effects, as
they are and ten times as large:

```python
import numpy as np
import polars as pl

import polars_online as po

# 40 features, each with a small effect, and the same features ten times as large.
rng = np.random.default_rng(5)
n, k = 20_000, 40
x = rng.standard_normal((n, k))
y = x @ rng.normal(0, 0.1, k) + rng.standard_normal(n)
unit, big = [f"x{j}" for j in range(k)], [f"big{j}" for j in range(k)]
pl.DataFrame({**dict(zip(unit, x.T)), **dict(zip(big, 10 * x.T)), "y": y}).write_parquet("wide.parquet")

# 1 and 2. One spec per scaling, each a grid of ridges solved from one Gram.
common = dict(targets=["y"], half_life=200.0, ridge=[0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0])
specs = [
    po.spec.ewridge("unit", features=unit, **common),
    po.spec.ewridge("tenfold", features=big, **common),
    po.spec.ewridge("standardized", features=big, standardize=True, **common),
]
# 3 to 5. Each spec in a call of its own: its best ridge, the error of the choice, and its least ridge's error.
for spec in specs:
    table = po.eval.grid_search(pl.scan_parquet("wide.parquet"), spec)
    best = table.filter(rank=1).row(0, named=True)
    choice = table.filter(kind="choice")["mse"].item()
    least = table.filter(ridge=0.01)["mse"].item()
    print(f"{spec['name']:>12}: rank 1 ridge {best['ridge']:g}, mse {best['mse']:.4f}; choice {choice:.4f}; at ridge 0.01, {least:.4f}")
```

```text
        unit: rank 1 ridge 0.3, mse 1.0604; choice 1.0604; at ridge 0.01, 1.0710
     tenfold: rank 1 ridge 30, mse 1.0604; choice 1.0604; at ridge 0.01, 1.0725
standardized: rank 1 ridge 0.3, mse 1.0602; choice 1.0602; at ridge 0.01, 1.0710
```

The tenfold features want a ridge a hundred times as large for the same
fit, and standardizing brings it back to 0.3. A half-life of 200 rows
weighs about 290 rows in all, too few to estimate forty small effects
well, so shrinking every one toward 0 lowers the error.

To choose a ridge on weeks held out from the fit, from Grams that one pass
leaves, see [A ridge and its t](../GRAM.md#a-ridge-and-its-t). That fits on
some weeks and scores on others, and needs no stream order.

## Choose the ridge and the half-life together

**A shorter memory has fewer rows behind each fit, so it wants more
shrinkage: choose the two together.**

1. **Give one spec both lists, `half_life` and `ridge`.** It keeps a Gram
   per half-life, and solves each ridge from each Gram.
2. **Pass the query and the spec to `po.eval.grid_search`.** It returns a
   row for each pair, with a `half_life` and a `ridge` column.
3. **Run the pair ranked 1 on the rows to come.**
4. **Report the error of the choice.**

This code plants forty coefficients that wander about 0:

```python
import numpy as np
import polars as pl

import polars_online as po

# 40 coefficients, each wandering about 0 with a spread of 0.1 (an AR(1) at 0.999 a row).
rng = np.random.default_rng(6)
n, k = 40_000, 40
x = rng.standard_normal((n, k))
# Each coefficient's path, a row at a time.
beta = np.empty((n, k))
beta[0] = rng.normal(0, 0.1, k)
step = rng.normal(0, 0.1 * np.sqrt(1 - 0.999**2), (n, k))
for i in range(1, n):
    beta[i] = 0.999 * beta[i - 1] + step[i]
# The target, and the table saved for the scan to read.
names = [f"x{j}" for j in range(k)]
y = (x * beta).sum(axis=1) + rng.standard_normal(n)
pl.DataFrame({"t": np.arange(n, dtype=float), **dict(zip(names, x.T)), "y": y}).write_parquet("walk.parquet")

# 1. Three half-lives by seven ridges: 21 fits from three Grams.
spec = po.spec.ewridge(
    "grid", targets=["y"], features=names, clock="t", gap_cap=5.0,
    half_life=[50.0, 200.0, 800.0], ridge=[0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
)
# 2. Every pair, scored in one pass.
table = po.eval.grid_search(pl.scan_parquet("walk.parquet"), spec)

# 3 and 4. The best ridge at each half-life, then the pair chosen and what choosing it earned.
settings = table.filter(kind="candidate")
for half_life, ridge, mse in settings.sort("mse").group_by("half_life", maintain_order=True).first().sort("half_life").select("half_life", "ridge", "mse").iter_rows():
    print(f"half-life {half_life:>3.0f}: best ridge {ridge:g}, mse {mse:.4f}")
best = settings.filter(rank=1).row(0, named=True)
print(f"rank 1: half-life {best['half_life']:.0f}, ridge {best['ridge']:g}; chosen from earlier rows: mse {table.filter(kind='choice')['mse'].item():.4f}")
```

```text
half-life  50: best ridge 1, mse 1.2224
half-life 200: best ridge 0.3, mse 1.1662
half-life 800: best ridge 0.1, mse 1.2533
rank 1: half-life 200, ridge 0.3; chosen from earlier rows: mse 1.1662
```

The best ridge rises as the half-life falls, from 0.1 at a half-life of
800 rows to 1 at 50.

## Check that it worked

The search worked when the ridge ranked 1 lies inside the grid you gave,
and the error of the choice is close to the error of rank 1:

| what you see | what it means | what to do |
|---|---|---|
| the ridge ranked 1 is the smallest or the largest in the list | the best ridge may lie outside the grid | extend the list past that end, and search again |
| the best ridge moves with the half-life, as in the second recipe | the two settings interact | search them together, in one spec |
| the best ridge changes when the features' scale changes, as `tenfold`'s did | the ridge is in the units of the features' variance | set `standardize=True`, or choose the ridge again on the new features |
