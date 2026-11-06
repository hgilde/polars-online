# Calibrating a model

A model's settings say how long it remembers (`half_life`), how hard it
shrinks (`ridge`), how fast its coefficients may move (`coef_half_life`,
`learning_rate`) and how often it expects a regime to end (`hazard`). Each
is chosen here from the stream's own errors, in one pass over the rows.
A bank makes every prediction before it learns from the row, so each row's
error is out of sample, and a setting is judged by the errors it made on
rows it had not yet seen. This code chooses a half-life for a stream whose
coefficient wanders:

```python
import numpy as np
import polars as pl

import polars_online as po

# y = b x0 + 0.5 x1 + noise, where b wanders as a slow random walk.
rng = np.random.default_rng(9)
n = 40_000
x0, x1, e = rng.standard_normal((3, n))
b = 1.0 + np.cumsum(rng.normal(0, 0.01, n))
pl.DataFrame({"t": np.arange(n, dtype=float), "x0": x0, "x1": x1, "y": b * x0 + 0.5 * x1 + e}).write_parquet("wander.parquet")

# One spec, five half-lives, each scored on the rows it had not seen, in one pass.
grid = po.spec.ewridge(
    "grid", targets=["y"], features=["x0", "x1"], clock="t", gap_cap=5.0,
    half_life=[25.0, 100.0, 400.0, 1_600.0, 6_400.0],  # a grid: one fit per half-life, from the same rows
)
table = po.eval.grid_search(pl.scan_parquet("wander.parquet"), grid)  # five spans, each chosen from the rows before it

# Each half-life's error, its rank, and how often the rows before a span chose it.
for half_life, mse, rank, chosen in table.drop_nulls("rank").select("half_life", "mse", "rank", "folds_chosen").iter_rows():
    print(f"half-life {half_life:>6.0f}: mse {mse:.4f}, rank {rank}, chosen for {chosen} of 5 spans")
print(f"choosing from earlier rows only: mse {table.filter(pl.col('spec').is_null())['mse'].item():.4f}")
```

```text
half-life     25: mse 1.0556, rank 3, chosen for 0 of 5 spans
half-life    100: mse 1.0278, rank 1, chosen for 5 of 5 spans
half-life    400: mse 1.0425, rank 2, chosen for 0 of 5 spans
half-life   1600: mse 1.1833, rank 4, chosen for 0 of 5 spans
half-life   6400: mse 1.8327, rank 5, chosen for 0 of 5 spans
choosing from earlier rows only: mse 1.0278
```

A half-life of 100 rows predicts best: 25 forgets so fast that its fit is
noisy, and 6,400 averages over a coefficient that has moved. The
half-life to run on the rows to come is the one ranked 1. The last line is the
number to report for it. The rows before each of five spans chose a
half-life, and the choice was scored on the span, so that error was earned
on rows the choice had not seen. Here every span chose 100, so the two
numbers agree.

| setting | model | searched as | recipe |
|---|---|---|---|
| `half_life` | every model that decays | a list in one spec | the example above |
| `ridge` | `ewridge` | a list in one spec | [The ridge](#the-ridge) |
| `ridge` and `half_life` | `ewridge` | both lists in one spec | [The ridge and the half-life together](#the-ridge-and-the-half-life-together) |
| `coef_half_life`, `revert_half_life` | `kalman` | one spec per value | [A Kalman filter's drift speed](#a-kalman-filters-drift-speed) |
| `learning_rate`; `c`; `alpha` | `sgd`; `pa`; `ftrl` | one spec per value | [The learning rate](#the-learning-rate) |
| `half_life` and `trend_half_life` | `holt` | a list in each of one spec per value | [Holt's smoothing](#holts-smoothing) |
| `hazard` and the priors | `bocpd` | one spec per value, scored by `loglik` | [The changepoint hazard](#the-changepoint-hazard) |
| `n_clusters`, `n_components`, `huber_delta` | `kmeans`, `hmm`, the robust losses | not by this pattern | [What this cannot choose](#what-this-cannot-choose) |

## One pattern for every setting

**Run every setting in one pass, as a grid where the setting takes one and
as one spec per value where it does not.** A spec takes a list for
`half_life`, for `ridge` and `feature_sets` on `ewridge`, and for
`penalties` on `lasso`, and fits one model per value from the same rows
(the README's [Output field names](../README.md#output-field-names) says
how each is named). A list given to `kalman`'s `coef_half_life` is one
value per coefficient, not a grid. For `kalman`'s drift speeds,
`sgd`'s `learning_rate`, `holt`'s `trend_half_life` and `bocpd`'s
`hazard`, build one spec per value and give the bank all of them. The
bank reads the file once either way. A grid inside one spec is the faster
of the two: five half-lives in one spec took 10% to 25% less time than five
specs of one half-life each, over 400,000 rows of five features, because
each spec adds its own column and bookkeeping to every row (measured on `ewridge`, `kalman`, `sgd` and
`holt`).

**Score every setting on the same rows.** A model withholds its
predictions while it warms up, and settings warm up at different speeds,
so a setting scored over its own rows is scored over fewer or more rows
than another. `po.eval.grid_search` counts a row only where every setting
for its target has a prediction and the target is present.

**Choose from earlier rows, and report the choice on later ones.** Each
setting's error over a stretch of rows is its true error plus noise. The
smallest of several such errors belongs to the setting whose noise happened
to run lowest, so the minimum over the rows you report understates what
choosing it will earn on the next rows. The fewer the rows and the more
settings, the larger the gap. This code makes twenty settings that differ
only in which noise column they read, so none is better than another, and
chooses one on the first half of the rows:

```python
import numpy as np
import polars as pl

import polars_online as po

# y = x0 + noise; z0 ... z19 are noise that y never reads.
rng = np.random.default_rng(3)
n = 20_000
x0, e = rng.standard_normal((2, n))
z = rng.standard_normal((20, n))
pl.DataFrame({"x0": x0, **{f"z{j}": z[j] for j in range(20)}, "y": x0 + e}).write_parquet("noise.parquet")

# Twenty feature sets, each x0 and one z: alike but for the noise each z adds.
sets = {f"z{j}": ["x0", f"z{j}"] for j in range(20)}
spec = po.spec.ewridge("sets", targets=["y"], features=["x0", *(f"z{j}" for j in range(20))], half_life=50.0, feature_sets=sets)
out = pl.scan_parquet("noise.parquet").online.fit_predict([spec]).collect()

# The set with the lowest error on the first half, against the average of the twenty, on each half.
first, second = po.eval.metrics(out.head(n // 2), spec), po.eval.metrics(out.tail(n // 2), spec)
pick = first.sort("mse").row(0, named=True)
later = second.filter(slot=pick["slot"])["mse"].item()
print(f"{pick['slot']} on the rows it was chosen on: mse {pick['mse']:.4f}, the average {first['mse'].mean():.4f}")
print(f"{pick['slot']} on the rows after: mse {later:.4f}, the average {second['mse'].mean():.4f}, rank {(second['mse'] < later).sum() + 1} of 20")
```

```text
pred_y__z18 on the rows it was chosen on: mse 0.9933, the average 0.9953
pred_y__z18 on the rows after: mse 1.0298, the average 1.0302, rank 5 of 20
```

On the rows it was chosen on, `z18` beat the average of the twenty by
0.0021. On the rows after, it beat it by 0.0004, which is noise. The first
number is the optimism, and on real data it is mixed in with any real
difference between the settings. The README's factor-grid example reads
its R² both ways for the same reason ([Parallelism](../README.md#parallelism)).

### Choosing with `po.eval.grid_search`

`po.eval.grid_search` runs one bank of every setting over a query, scores
each one, and scores the choice. It cuts the rows as scikit-learn's
`TimeSeriesSplit(n_splits)` cuts a series: `n` rows into `n_splits + 1`
spans of `n // (n_splits + 1)` rows, the first a little longer. Before
each of the last `n_splits` spans, it chooses the setting with the lowest
error over every row before the span, and scores that choice on the span.
It keeps only two sums per setting and span, so its memory is the bank's
state and one chunk of rows, however long the stream.

| keyword | what it does |
|---|---|
| `specs` | a spec, a list of specs, or a builder called once per point of `param_grid`, as `specs(name, **point)` |
| `param_grid` | a dict of each keyword's values, as scikit-learn's `ParameterGrid` takes one: every combination is a point |
| `n_splits` | the test spans, 5 by default; `1` chooses on the first half and scores the second |
| `gap` | rows left out between the rows a choice reads and its span: give it the rows a target looks ahead, so the choice reads only outcomes known when the span starts |
| `scoring` | `"mse"`, lowest best, or `"loglik"`, highest best, for `bocpd`, `hmm` and `deco` |
| `chunk_size`, `threads` | as [`ModelBank`](../README.md#in-a-loop-modelbank) takes them |

It returns a frame with one row per setting and target:

| column | what it holds |
|---|---|
| `spec`, the `param_grid` keys, and `half_life`, `ridge`, `feature_set`, `penalty` where a spec's own grid varies them | the setting |
| `slot`, `target` | the field scored and the target it predicts |
| `n` | the rows scored: those of the test spans on which every setting has a prediction |
| `mse` or `loglik` | the setting's score over those rows |
| `rank` | 1 for the best: the setting to run on the rows to come. Its score was read on the rows that ranked it, so it is the optimistic number |
| `folds_chosen` | how many spans the choice ran it on |

One more row per target, whose `spec` is null and `slot` is
`pred_<target>__chosen`, holds the choice: its score over the same rows is
the number to report. A spec's `emit_selected` and `emit_averaged`
predictions, where it writes them, are scored on the same rows with a null
`rank`. They choose differently: `emit_selected` ranks the settings on
every row by an exponentially weighted error at each setting's own
half-life, so it follows a best setting that moves, at the price of the
noise in a short memory. [Choosing among a grid's
settings](DIAGNOSTICS.md#choosing-among-a-grids-settings) measures it.

**To score a target that looks ahead, give `gap` the rows it looks
ahead.** A row's residual is known only once its target has played out,
which for a target summing the next ten rows is ten rows later. Without
`gap`, the choice before a span would read the residuals of its last ten
rows, whose outcomes were not yet known at the span's start.

### Choosing by hand

`po.eval.metrics` and `po.eval.compare_specs` score an output frame, and
on two halves of it they make the same choice with `n_splits=1`. This code
runs both on the half-life grid of the first example, which writes
`wander.parquet`:

```python
import numpy as np
import polars as pl

import polars_online as po

# The first example's stream.
rng = np.random.default_rng(9)
n = 40_000
x0, x1, e = rng.standard_normal((3, n))
b = 1.0 + np.cumsum(rng.normal(0, 0.01, n))
pl.DataFrame({"t": np.arange(n, dtype=float), "x0": x0, "x1": x1, "y": b * x0 + 0.5 * x1 + e}).write_parquet("wander.parquet")
grid = po.spec.ewridge(
    "grid", targets=["y"], features=["x0", "x1"], clock="t", gap_cap=5.0,
    half_life=[25.0, 100.0, 400.0, 1_600.0, 6_400.0],
)
lf = pl.scan_parquet("wander.parquet")

# By hand: score every half-life on each half, choose on the first, and report on the second.
out = lf.online.fit_predict([grid]).collect()
first, second = po.eval.metrics(out.head(n // 2), grid), po.eval.metrics(out.tail(n // 2), grid)
pick = first.sort("mse")["slot"][0]
print(f"by hand: {pick}, mse {second.filter(slot=pick)['mse'].item():.4f} on the second half")

# The same with grid_search: one span to test, so the choice is read from the first half.
table = po.eval.grid_search(lf, grid, n_splits=1)
chosen = table.filter(pl.col("folds_chosen") == 1)["slot"].item()
print(f"grid_search: {chosen}, mse {table.filter(pl.col('spec').is_null())['mse'].item():.4f} on the second half")
```

```text
by hand: pred_y@h100, mse 1.0287 on the second half
grid_search: pred_y@h100, mse 1.0287 on the second half
```

The scores agree here because every half-life predicts from the same row.
Where one setting warms up later than another, `po.eval.metrics` scores
each over its own rows, and `grid_search` over the rows they share.

**The call saves the loop, and the memory the output takes.** By hand, one
split takes four statements after the specs are built: run and collect,
score the two halves, pick, and read the pick's later error. The recipes
below take four to eight that way, and one more to build a spec per value
where the setting does not grid. Five spans, each chosen from the rows
before it on the rows every setting scored, took eleven statements for
the half-life grid above, and gave the call's numbers to the bit. Each
form by hand also holds the whole output, one row per row and setting once
it is unpacked, where the call keeps two sums per setting and span.

## Recipes

Each recipe plants a truth, searches its setting with
`po.eval.grid_search`, and prints the error of each value and of the
choice. Every value is in one bank, so each reads its file once.

### The ridge

**`ewridge`'s `ridge` is added to the features' weighted covariance, so it
is in the units of the features' variance.** The solve is
`(S + ridge·D) β = η`, `S` the running mean of `z zᵀ` and `D` the identity
with a 0 at the intercept, which is the features' covariance plus `ridge`
on its diagonal. A ridge of 0.3 on features of variance 1 shrinks as hard
as a ridge of 30 on the same features times ten. With `standardize=True`
the system is solved in correlation form, where a ridge is in units of
each feature's own variance, whatever its scale. This code searches the
ridge for forty features with small effects, as they are and ten times as
large:

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

# One spec per scaling, each a grid of ridges solved from one set of sums.
common = dict(targets=["y"], half_life=200.0, ridge=[0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0])
specs = [
    po.spec.ewridge("unit", features=unit, **common),
    po.spec.ewridge("tenfold", features=big, **common),
    po.spec.ewridge("standardized", features=big, standardize=True, **common),
]
table = po.eval.grid_search(pl.scan_parquet("wide.parquet"), specs)

# Each spec's best ridge, beside its least.
for name in ("unit", "tenfold", "standardized"):
    mine = table.filter(spec=name).sort("mse")
    least = mine.filter(ridge=0.01)["mse"].item()
    print(f"{name:>12}: best ridge {mine['ridge'][0]:g}, mse {mine['mse'][0]:.4f}; at ridge 0.01, {least:.4f}")
```

```text
        unit: best ridge 0.3, mse 1.0604; at ridge 0.01, 1.0710
     tenfold: best ridge 30, mse 1.0604; at ridge 0.01, 1.0725
standardized: best ridge 0.3, mse 1.0602; at ridge 0.01, 1.0710
```

The tenfold features want a ridge a hundred times as large for the same
fit, and standardizing brings it back to 0.3. So a ridge chosen on one
set of features carries to another only under `standardize`. The truth
here is forty effects of spread 0.1 in noise of 1. A half-life of 200 rows
weighs about 290 rows in all, too few to estimate forty small effects
well, so shrinking every one toward 0 lowers the error.

To choose a ridge on weeks held out from the fit, from Grams that one pass
leaves, see [A ridge and its t](GRAM.md#a-ridge-and-its-t). That fits on
some weeks and scores on others, and needs no stream order.

### The ridge and the half-life together

**A shorter memory has fewer rows behind each fit, so it wants more
shrinkage, and the two are chosen together.** Give both lists to one spec:
it keeps a set of running sums per half-life, and solves each ridge from
each set. This code plants forty coefficients that wander about 0:

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

# Three half-lives by seven ridges: 21 fits from three sets of sums.
spec = po.spec.ewridge(
    "grid", targets=["y"], features=names, clock="t", gap_cap=5.0,
    half_life=[50.0, 200.0, 800.0], ridge=[0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
)
table = po.eval.grid_search(pl.scan_parquet("walk.parquet"), spec)

# The best ridge at each half-life, then the pair chosen and what choosing it earned.
settings = table.drop_nulls("rank")
for half_life, ridge, mse in settings.sort("mse").group_by("half_life", maintain_order=True).first().sort("half_life").select("half_life", "ridge", "mse").iter_rows():
    print(f"half-life {half_life:>3.0f}: best ridge {ridge:g}, mse {mse:.4f}")
best = settings.filter(rank=1).row(0, named=True)
print(f"rank 1: half-life {best['half_life']:.0f}, ridge {best['ridge']:g}; chosen from earlier rows: mse {table.filter(pl.col('spec').is_null())['mse'].item():.4f}")
```

```text
half-life  50: best ridge 1, mse 1.2224
half-life 200: best ridge 0.3, mse 1.1662
half-life 800: best ridge 0.1, mse 1.2533
rank 1: half-life 200, ridge 0.3; chosen from earlier rows: mse 1.1662
```

The best ridge rises as the half-life falls. A ridge chosen once at the
default half-life and kept while the half-life is searched would leave the
short half-lives too little shrinkage and the long ones too much, and
would rank them wrongly.

### A Kalman filter's drift speed

**`kalman`'s `coef_half_life` sets how fast a coefficient may drift:** it
is a coefficient's memory on the clock, the half-life of an exponentially
weighted fit whose steady-state gain the filter's process noise matches. A
list there gives one value per coefficient, so search it with one spec per
value. Give `functools.partial` the keywords
every spec shares and `param_grid` the values to try:

```python
import functools

import numpy as np
import polars as pl

import polars_online as po

# y = b x0 + 0.5 x1 + noise, where b wanders as a random walk.
rng = np.random.default_rng(8)
n = 30_000
x0, x1, e = rng.standard_normal((3, n))
b = 1.0 + np.cumsum(rng.normal(0, 0.02, n))
pl.DataFrame({"t": np.arange(n, dtype=float), "x0": x0, "x1": x1, "y": b * x0 + 0.5 * x1 + e}).write_parquet("drift.parquet")

# One kalman spec per drift speed, in one bank.
kalman = functools.partial(po.spec.kalman, targets=["y"], features=["x0", "x1"], clock="t", gap_cap=5.0, half_life=500.0)
grid = {"coef_half_life": [3.0, 10.0, 30.0, 100.0, 300.0, 1_000.0]}
table = po.eval.grid_search(pl.scan_parquet("drift.parquet"), kalman, grid)

for speed, mse, rank in table.drop_nulls("rank").select("coef_half_life", "mse", "rank").iter_rows():
    print(f"coef_half_life {speed:>6g}: mse {mse:.4f}, rank {rank}")
print(f"chosen from earlier rows: mse {table.filter(pl.col('spec').is_null())['mse'].item():.4f}")
```

```text
coef_half_life      3: mse 1.3468, rank 5
coef_half_life     10: mse 1.1146, rank 3
coef_half_life     30: mse 1.0540, rank 1
coef_half_life    100: mse 1.0553, rank 2
coef_half_life    300: mse 1.1297, rank 4
coef_half_life   1000: mse 1.3533, rank 6
chosen from earlier rows: mse 1.0553
```

Drift speeds of 30 and 100 clock units nearly tie, and the rows before
some spans preferred 100. So the choice earned 1.0553, a little worse than
the 1.0540 that 30 shows over the same rows, and 1.0553 is the honest
number. `revert_half_life`, which pulls each coefficient back toward 0, is
searched the same way: add it to `param_grid`, and every combination of
the two is a spec.

### The learning rate

**`sgd`'s `learning_rate` sets how many rows the coefficients remember:**
about `1 / learning_rate` under `standardize`, the default, since every
row's step moves them and the half-life does not decay them. So the rate
plays the part a half-life plays in a model that keeps sums. It is one
number a spec, searched as `kalman`'s drift speed is. `pa`'s `c`, which
caps each step, and `ftrl`'s `alpha`, its rate's scale, are searched the
same way:

```python
import functools

import numpy as np
import polars as pl

import polars_online as po

# y = b x0 + 0.5 x1 + noise, where b wanders as a slow random walk.
rng = np.random.default_rng(10)
n = 30_000
x0, x1, e = rng.standard_normal((3, n))
b = 1.0 + np.cumsum(rng.normal(0, 0.01, n))
pl.DataFrame({"x0": x0, "x1": x1, "y": b * x0 + 0.5 * x1 + e}).write_parquet("slow.parquet")

# One sgd spec per learning rate.
sgd = functools.partial(po.spec.sgd, targets=["y"], features=["x0", "x1"], half_life=1_000.0)
table = po.eval.grid_search(pl.scan_parquet("slow.parquet"), sgd, {"learning_rate": [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]})

for rate, mse, rank in table.drop_nulls("rank").select("learning_rate", "mse", "rank").iter_rows():
    print(f"learning_rate {rate:<6g} (about {1 / rate:>4.0f} rows): mse {mse:.4f}, rank {rank}")
print(f"chosen from earlier rows: mse {table.filter(pl.col('spec').is_null())['mse'].item():.4f}")
```

```text
learning_rate 0.0003 (about 3333 rows): mse 1.4051, rank 6
learning_rate 0.001  (about 1000 rows): mse 1.0749, rank 4
learning_rate 0.003  (about  333 rows): mse 1.0263, rank 2
learning_rate 0.01   (about  100 rows): mse 1.0262, rank 1
learning_rate 0.03   (about   33 rows): mse 1.0584, rank 3
learning_rate 0.1    (about   10 rows): mse 1.2047, rank 5
chosen from earlier rows: mse 1.0262
```

A memory of about 100 rows predicts best here, the same as a half-life
of 100 in the first example on a coefficient wandering at the same speed.
With a shorter memory each step chases the noise, and with a longer one
the coefficient has moved on before the fit catches up.

### Holt's smoothing

**`holt`'s two smoothing constants are half-lives:** `half_life` for the
level and `trend_half_life` for the trend. A row's textbook gain on the
level is `α = 1 − 2^(−1/half_life)` once the weight has saturated, so a
half-life of 8 rows is an `α` of 0.083. `half_life` takes a list and
`trend_half_life` does not, so this code builds one spec per trend
half-life, each with a grid of level half-lives:

```python
import functools

import numpy as np
import polars as pl

import polars_online as po

# A local linear trend: the slope wanders, and the level follows it with steps of its own.
rng = np.random.default_rng(12)
n = 30_000
slope = np.cumsum(rng.normal(0, 0.002, n))
level = np.cumsum(slope + rng.normal(0, 0.1, n))
pl.DataFrame({"t": np.arange(n, dtype=float), "y": level + rng.standard_normal(n)}).write_parquet("trend.parquet")

# One spec per trend half-life, each a grid of five level half-lives: 25 fits.
holt = functools.partial(po.spec.holt, targets=["y"], clock="t", gap_cap=5.0, half_life=[2.0, 4.0, 8.0, 16.0, 32.0])
table = po.eval.grid_search(pl.scan_parquet("trend.parquet"), holt, {"trend_half_life": [25.0, 100.0, 400.0, 1_600.0, float("inf")]})

# The three best pairs, and what choosing from earlier rows earned.
for trend, level, mse in table.drop_nulls("rank").sort("rank").head(3).select("trend_half_life", "half_life", "mse").iter_rows():
    print(f"level half-life {level:>2.0f} (alpha {1 - 2 ** (-1 / level):.3f}), trend half-life {trend:>4g}: mse {mse:.4f}")
print(f"chosen from earlier rows: mse {table.filter(pl.col('spec').is_null())['mse'].item():.4f}")
```

```text
level half-life  8 (alpha 0.083), trend half-life   25: mse 1.1331
level half-life  4 (alpha 0.159), trend half-life  100: mse 1.1362
level half-life  4 (alpha 0.159), trend half-life   25: mse 1.1401
chosen from earlier rows: mse 1.1342
```

A level half-life of 8 rows with a trend half-life of 25 ranks first.
The trend forgets faster than its default, four times the level's, because
the planted slope wanders. Choosing from earlier rows earned 1.1342
against the 1.1331 the winner shows over the same rows.

### The changepoint hazard

**A changepoint model predicts a density, so it is scored by the density
it gave each row before learning from it.** `bocpd` writes `loglik`, the
log of that predictive density, on every row. The sum over a stream is the
log of the stream's marginal likelihood under the model, which is the
criterion a Bayesian model's settings are compared by, and each term was
read before its row was learned from (Dawid's prequential score). A
squared error on `pred_<f>` would score only the mean it forecasts, and a
model that dated every break right would gain nothing for it. So give
`grid_search` `scoring="loglik"`. This code plants a mean that jumps at
random, every 200 rows on average:

```python
import functools

import numpy as np
import polars as pl

import polars_online as po

# A mean that jumps at random, every 200 rows on average, to a level of spread 2, in noise of 1.
rng = np.random.default_rng(14)
n = 20_000
regime = np.cumsum(rng.random(n) < 1 / 200)
x = rng.normal(0, 2.0, regime.max() + 1)[regime] + rng.standard_normal(n)
pl.DataFrame({"x": x}).write_parquet("jumps.parquet")

# One bocpd spec per hazard, its priors matching the planted spreads.
bocpd = functools.partial(
    po.spec.bocpd, features=["x"], mean_prior=[0.0], covariance_prior=[1.0],
    mean_precision_prior=0.25, degrees_of_freedom_prior=2.0,   # a level's spread of 2 is the noise's over 0.25, squared
)
table = po.eval.grid_search(pl.scan_parquet("jumps.parquet"), bocpd, {"hazard": [20.0, 60.0, 200.0, 600.0, 2_000.0]}, scoring="loglik")

for hazard, loglik, rank in table.drop_nulls("rank").select("hazard", "loglik", "rank").iter_rows():
    print(f"hazard {hazard:>5g}: loglik {loglik:.4f} a row, rank {rank}")
print(f"chosen from earlier rows: loglik {table.filter(pl.col('spec').is_null())['loglik'].item():.4f} a row")
```

```text
hazard    20: loglik -1.5035 a row, rank 5
hazard    60: loglik -1.4779 a row, rank 3
hazard   200: loglik -1.4729 a row, rank 1
hazard   600: loglik -1.4751 a row, rank 2
hazard  2000: loglik -1.4796 a row, rank 4
chosen from earlier rows: loglik -1.4731 a row
```

The hazard of 200, the planted mean run, ranks first. The priors are
chosen the same way, as more keys of `param_grid`: searched over a
`covariance_prior` of 0.01, 0.1, 1 and 10 at a hazard of 200, the score
was highest at 1, the planted noise's variance, at −1.4729 a row against
−1.4921 at 0.01.

## What this cannot choose

Three settings have no out-of-sample score this pattern can choose them
by: one model writes none, one writes a score that was measured not to
find the setting, and one would be pushed by the squared error toward the
loss it exists to replace.

| setting | why the pattern does not choose it | what to do |
|---|---|---|
| `kmeans`' `n_clusters` | its only score is `dist`, the distance to the nearest centre, and that falls as clusters are added even on rows not yet learned from: on three planted clusters its mean square on the second half went 1.18, 0.45, 0.38, 0.32 and 0.26 at two to six clusters. It writes no likelihood | choose it by what the clusters are for, or from a Gram or a sample with a criterion of your own |
| `hmm`'s `n_components` | `hmm` writes `loglik`, but on a stream planted with three Gaussian states of spread 1 at means −2, 0 and 2, four states scored −1.60 a row on the second half and three −1.67, so the score is not shown to find the count | read `n_components` from what the states mean |
| `huber_delta` | it is the cut, in units of the residuals' spread, past which a row's residual counts linearly. The default 1.345 keeps 95% of least squares' efficiency when the noise is normal, and searching it by squared error would push it toward least squares | keep 1.345, or choose by a robust loss of your own over `po.eval.unpack`'s residuals |
