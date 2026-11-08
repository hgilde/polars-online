<!-- kind: explanation -->
# How `po.gbdt` builds training data

This page explains, for anyone who trains XGBoost or LightGBM with
`po.gbdt`, how it bins rows that do not fit in memory and hands them to
each library, and why the kind of file it reads decides its memory. [Train XGBoost or LightGBM on
files larger than memory](../GBDT.md) gives the steps.

**Both libraries train in two stages, and only the first reads every
row.** *Binning* places each feature's bins and puts every value into one,
which reads every row. *Boosting* grows trees from the per-bin sums and
never reads a raw value again. `po.gbdt` does the binning from data that
streams: it reads the rows a chunk at a time, in Rust. It hands XGBoost
each chunk's Arrow buffers as they are, with no further copy, with each null marked by
its validity bit. It hands LightGBM the rows as float32.
Each library is loaded from the package you installed, so polars-online
still depends on Polars alone.

## Where the bins come from

**XGBoost's cuts come from a sample of 200,000 rows, which removes
XGBoost's pass that sketches every row.** That pass is about half the time
before the first round. `sample_rows=None` keeps it, and with it XGBoost's
own trees. Without weights the sample is uniform. On 10 million unweighted
rows of 50 features, at 50 rounds, measured on 2026-10-10, XGBoost's sketch scored a validation
RMSE of 1.2379, and cuts from five seeds of the sample 1.2347 to 1.2411.
LightGBM's own sampling spans 1.2298 to 1.2443 over five seeds.
`po.gbdt.lightgbm_dataset` returns a constructed `lgb.Dataset`, binned on
LightGBM's own sample, and holds the data in memory only when it fits.

**With `weight=`, XGBoost's cuts follow the weights, as its own sketch
does, and LightGBM's bins do not, as its own do not.** XGBoost's sketch
weights every row, so `xgboost_matrix` draws its sample in proportion to
the weights (priority sampling) and places the cuts at the sample's
weighted quantiles. A row of weight 0 places no cut. LightGBM's sample is
uniform, and `LGBM_DatasetCreateFromSampledColumn`, the call that bins it,
takes no weights, so `lightgbm_dataset` bins as LightGBM does. Measured on
2026-10-11, on 50,000 normal values with weight 50 above 1.0 and
`max_bin=16`:

| bins | bounds | above 1.0 | heaviest bin's share of the weight |
|---|---:|---:|---:|
| XGBoost's own weighted sketch | 15 | 14 | 6.47% |
| `xgboost_matrix` | 15 | 14 | 6.26% |
| LightGBM's own, and `lightgbm_dataset`'s | 14 | 2 | 35% |

The ideal share is 1/16, 6.25%. LightGBM's bounds are those its trees
split at over 50 rounds, the same with the weights as without. The
docstring of `xgboost_matrix` defines the sample and states its sampling
error.

**The first round begins sooner than XGBoost's own route from Polars, and
`lightgbm_dataset`'s in-memory route begins as soon as LightGBM's own, at
a third of the memory.** On the 10 million rows of 50 features the sample's cuts were scored on, it began after 4.0 seconds for XGBoost,
against 8.7 for XGBoost's own iterator over Polars' chunks. LightGBM's
own route, from a frame held in memory, began after 1.54 seconds at
8.9 GB. `lightgbm_dataset`'s in-memory route began after 1.45 seconds at
2.9 GB, and its streamed route after 1.68 seconds. Each time is the
median of three interleaved runs on 8 threads, from Arrow IPC files, on an
Apple M4 Pro at a load average of 4.5 to 10, on 2026-10-10.

## How a build runs

**Both libraries read the same way, and differ in how they bin and hand
the rows over:**

| step | XGBoost | LightGBM |
|---|---|---|
| read | Arrow IPC files in Rust, a chunk (an Arrow record batch) at a time (polars-arrow's reader); Parquet and a `LazyFrame` through Polars' `collect_batches`, which gives the result a chunk at a time, each frame's pieces joined into one on the reader thread; two chunks ready ahead | the same |
| bins | a sample of 200,000 rows by the SplitMix64 hash of their index, the same rows however the source is cut into chunks: the rows of smallest hash, or with weights the rows of largest `w / u`, `u` uniform from the hash (priority sampling); edges at the sample's weighted quantiles, or a bin per value for a feature of at most `max_bin` values; built into a dense `ref` | LightGBM's own, unweighted: `LGBM_SampleIndices`, those rows gathered in the first pass, `LGBM_DatasetCreateFromSampledColumn` |
| hand-over | each chunk's columns as array interfaces on their Arrow buffers, the validity bitmap as the mask, into `XGQuantileDMatrixCreateFromCallback`; the chunk's margin as its `base_margin` | float32 rows built in parallel, pushed with their labels, weights and initial scores a chunk at a time, `LGBM_DatasetPushRowsWithMetadata`; or in memory, every chunk as float32 Arrow, `LGBM_DatasetCreateFromArrowStream`, and every row's initial scores set after it |
| categoricals | a `pl.Enum`'s codes, feature type `c`, its levels planted in the `ref` | a `pl.Enum`'s codes in `categorical_feature`, and any integer or Boolean column named |
| refusals | a null or non-finite label, weight or margin, a negative weight, a margin row whose number of values differs from the first row's or the `ref`'s, a `ref` with a margin and none here or none and one here, a `pl.Categorical`, a column of another type, a source whose passes give different rows, a `ref` built here on other features or on a `pl.Enum` of other categories, under `engine="native"`, a library version outside the range the native engine is built for (XGBoost from 2.1, the first whose C API takes Arrow columns, to before 4.0), where `engine="auto"` falls back to the Python engine with a `UserWarning` | the same, with `reference` for `ref`, with LightGBM from 4.0 to before 5.0, and more than 2³¹ − 1 rows; a `categorical=` that names a float column or a column not among the features, or that is one name and not a list; and `in_memory=True` under the native engine with a LightGBM before 4.7.0, from which `engine="auto"` does not fall back |

### For XGBoost, a feature of few values gets a bin per value

**A feature of at most `max_bin` distinct values gets a bin per value, with
the values as its edges.** A `ref` column of `d <= max_bin` distinct values
`u_0 < .. < u_{d-1}` comes back as the cuts
`[-inf, u_1, .., u_{d-1}, u_{d-1} + |u_{d-1}| + 1e-5]` on XGBoost 3.4.1.
`tests/test_gbdt.py` holds every feature's cuts to the sample's definition.
This recipe plants a rating of four values beside a continuous feature, and
reads the cuts back from the matrix:

```python
import numpy as np
import polars as pl
import polars_online as po

# A rating of four values beside a continuous feature.
rng = np.random.default_rng(6)
n = 10_000
pl.DataFrame({
    "rating": rng.choice([1.0, 2.0, 3.5, 5.0], n),     # four distinct values
    "size": rng.standard_normal(n),                    # continuous
    "y": rng.standard_normal(n),
}).write_parquet("ratings.parquet")

# Build the matrix, and read each feature's cuts back from it.
train = po.gbdt.xgboost_matrix("ratings.parquet", label="y", features=["rating", "size"])
indptr, cuts = train.get_quantile_cut()                # every feature's cuts, end to end
print("rating's cuts:", [round(float(c), 5) for c in cuts[indptr[0]:indptr[1]]])
print("size's cut values:", indptr[2] - indptr[1], "at max_bin=256")
```

```text
rating's cuts: [-inf, 2.0, 3.5, 5.0, 10.00001]
size's cut values: 257 at max_bin=256
```

### The Python engine grows the native engine's trees

**`engine="python"` builds through the libraries' Python APIs, and trains
the same trees.** It feeds numpy chunks to an `xgboost.DataIter`, with the
`ref` built from the same sample by the same Rust code, and a
`lightgbm.Sequence` that answers LightGBM's sample rows from a first pass.
`engine="auto"`, the default, falls back to it, with a warning, when the
native engine cannot load a library. This recipe builds one file both ways
and trains each for 20 rounds:

```python
import numpy as np
import polars as pl
import polars_online as po
import xgboost as xgb

# One file, whose target is x0 times x1 plus noise.
rng = np.random.default_rng(6)
n = 10_000
x0, x1 = rng.standard_normal((2, n))
pl.DataFrame({"x0": x0, "x1": x1, "y": x0 * x1 + rng.standard_normal(n)}).write_parquet("pairs.parquet")

# The file built by each engine, and the same trees trained on each.
native = po.gbdt.xgboost_matrix("pairs.parquet", label="y", features=["x0", "x1"], engine="native")
python = po.gbdt.xgboost_matrix("pairs.parquet", label="y", features=["x0", "x1"], engine="python")
params = {"tree_method": "hist", "max_depth": 3}
a = xgb.train(params, native, 20).predict(native)
b = xgb.train(params, python, 20).predict(python)
print("largest difference between the engines' predictions:", float(np.abs(a - b).max()))
```

```text
largest difference between the engines' predictions: 0.0
```

### Why a categorical feature is a `pl.Enum`

**A `pl.Enum`'s levels fix what each code means before the data, and a
`pl.Categorical`'s codes depend on the process that made them.** A
`pl.Enum`'s codes go in as a categorical feature, so a model reads them
the same way at prediction. XGBoost bins a category code missing from the
`ref` without an error, so a `pl.Categorical`'s codes that meant something
else in another process would go unnoticed, and `po.gbdt` refuses one. A
`ref=` or `reference=` built here records each feature's levels, and a
later frame whose `pl.Enum` lists them in another order is refused by
name, since its codes would be read as the training levels.

### Why LightGBM takes its binning parameters when the dataset is built

**The dataset `lightgbm_dataset` returns is already *constructed*,
LightGBM's word for a dataset whose bins are set and whose raw data is
freed.** `lgb.train` passes its parameters through
`LGBM_DatasetUpdateParamChecking`, which refuses a changed `max_bin`,
`data_random_seed`, `min_data_in_bin` and the rest, as it refuses for
LightGBM's own constructed dataset. The dataset records its categorical
features under `categorical_column`, the key LightGBM's own `Dataset`
records them under, so a validation set you build yourself with
`reference=train` gets no warning.

### LightGBM's route in memory

**`in_memory="auto"` takes LightGBM's Arrow route when the features as
float32 take at most a quarter of the memory available, under the native
engine with LightGBM 4.7.0 or later, the first that takes an Arrow
stream.** On 1e7 × 50,
measured on 2026-10-10, that route was 14% sooner than the two passes, at a peak 1.4 times the float32
data (2.9 GB for 2.0 GB). Both routes hand LightGBM the same float32
values, so the choice, which depends on the machine, never changes the
trees.

## Why a Huber loss's slope is set from the residuals

**A Huber loss is quadratic for a residual within its slope `δ` and
linear beyond it, so a wild label pulls a tree no harder than a residual
of `δ` does.** LightGBM's is Huber's piecewise loss. XGBoost's is the
pseudo-Huber loss `δ²(√(1 + (r/δ)²) − 1)`, a smooth version of it that
bends from the one regime to the other around `δ` rather than at it. `δ`
is in the label's units. On a target whose residuals are about 0.001,
every row falls inside the cut and the loss is squared error by another
name. On one whose residuals are about 100, every row falls beyond it.
At 1.345 standard deviations Huber's loss keeps 95% of least squares'
efficiency on normal errors, and the MAD is not moved by the wild labels
the cut is meant to catch, where a root mean square is. Hence a slope
of `1.345 · MAD / 0.6745`, from a first fit's residuals. `po.gbdt` builds
only the training input, so the objective and its slope are the train
call's.

## When a margin helps, and when a feature does

**A margin, the raw score each row's prediction starts from, wins where
the level can leave the range the training rows reached, and a feature
can win where it cannot.** A tree predicts within the range of the targets
it was trained on, and a margin is added after the trees, so it carries a
level the trees never saw. Measured on 2026-11-05
with LightGBM 4.7.0's trees, early stopping on a validation slice, each
data set split by time with no shuffling. The planted rows are 100 a
clock step over 2,000 steps, with noise of spread 1.0 and a level that
wanders and that no feature carries. Their online model has a half-life
of 5 steps (`ewridge` for the Gaussian target, `ftrl` for the Bernoulli
one). Each seed trains on the first part of its clock and tests on the
last.
BTCUSDT is Binance's one-minute trade counts over 60 days from 2024-01-02,
a Poisson objective, scored as RMSE on the log scale. Flights is every
2013 departure from New York (nycflights13), arrival delay in minutes, with
the origin airport's recent delays as the online prediction:

| data | score | trees alone | the prediction as a feature | the prediction as the margin |
|---|---|---:|---:|---:|
| planted, Gaussian, test level inside the training range (seeds 0, 1) | RMSE | 1.777, 1.742 | 1.026, 1.024 | 1.021, 1.022 |
| planted, Gaussian, test level outside it (seed 2) | RMSE | 5.825 | 3.131 | **1.023** |
| planted, Bernoulli, inside (seeds 0, 1) | log loss | 0.516, 0.386 | 0.480, 0.353 | 0.479, 0.354 |
| planted, Bernoulli, outside (seed 2) | log loss | 0.389 | 0.204 | **0.152** |
| BTCUSDT trade counts | log-RMSE | 0.4682 | 0.4652 | **0.4603** |
| flights' arrival delay | RMSE, minutes | 37.39 | **35.13** | 36.00 |

The planted Gaussian rows' noise floor is 1.0. Where the level stays
inside the training range, the margin and the feature score within 0.005
of each other. On flights the feature won, and passing the prediction both
ways scored 35.22, within 0.09 of it, which is why [the margin's
steps](../GBDT.md#train-trees-around-an-online-models-prediction) pass
both.

**The online prediction leaks nothing into the margin.** The online model
predicts each row before it learns from that row's target, so the margin
of a training row never contains the target the trees are fitted to.

## Polars' read-ahead

**Polars, when it runs a query, reads ahead of a slower reader by an
amount set by its thread count and by `POLARS_ROW_GROUP_PREFETCH_SIZE`,
how many blocks of a parquet file it reads ahead.** From py-polars 1.37.0,
the oldest this library supports, `collect_batches` fed directly by a
file read as a query waits for its reader, as [Why the range is 1.37.0
to 3](polars-interfaces.md#why-the-range-is-1370-to-3) measures. A step such as `with_columns` or a `select` that
computes reads ahead by about ten frames a thread.

`xgboost_matrix` over a 1.0 GB source, on py-polars 2.0.0 on 2026-10-10,
measured as the growth of peak memory:

| source | reader | peak growth |
|---|---|---:|
| Arrow IPC | the native reader | 0.46 GB |
| Parquet | `collect_batches`, `POLARS_ROW_GROUP_PREFETCH_SIZE=1` and 4 Polars threads | 0.74 GB |

**So the native engine reads Arrow IPC files with the least memory,
because it reads them itself, with two chunks ready ahead.** For a
Parquet or query source, how far Polars reads ahead in the file and its
thread count bound what it holds, as [Bound a run's memory](../how-to/bound-memory.md) says.

**A frame from `collect_batches` arrives in several pieces, and joining
them before the library is faster than handing the pieces over one by
one.** Each piece was a chunk of its own for XGBoost: 378 on 1e7 × 50,
where the IPC files give 132. XGBoost's construction took 4.4 seconds
against 3.9 with the pieces joined on the reader thread, the copy
overlapping XGBoost's work.
