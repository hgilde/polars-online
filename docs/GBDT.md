# Training XGBoost and LightGBM: `po.gbdt`

**`po.gbdt` builds XGBoost's and LightGBM's own training objects from Arrow
IPC files, Parquet files or a query, so that `xgb.train` and `lgb.train` run
on data that does not fit in memory.** Both libraries train in two stages.
*Binning* places each feature's bins and puts every value into one, which
reads every row. *Boosting* grows trees from the per-bin sums and never
reads a raw value again. `po.gbdt` does the binning from data that streams:
it reads the rows in batches, in Rust, and hands each library a batch's
Arrow buffers with no copy, a null by its validity bit. Each library is
loaded from the package you installed, so polars-online still depends on
Polars alone. `po.gbdt` is unstable: it is built on the libraries' C APIs,
so its keywords may change in any release, and a new library version may
need a new release here.

Each recipe on this page writes its own files and reads them back. This
one writes a training file and a validation file whose target depends on
`x0` and `x1` but not `x2`, with noise of spread 0.3, and trains each
library on them for 50 rounds. The validation rows are binned on the
training rows' bins, through `ref=` for XGBoost and `reference=` for
LightGBM:

```python
import lightgbm as lgb
import numpy as np
import polars as pl
import polars_online as po
import xgboost as xgb

# The training and validation files, with the target planted in x0 and x1.
rng = np.random.default_rng(5)
for name, n in (("train", 20_000), ("valid", 5_000)):
    x0, x1, x2 = rng.standard_normal((3, n))
    pl.DataFrame({
        "x0": x0, "x1": x1, "x2": x2,                                      # x2 plays no part
        "y": np.sin(2 * x0) + (x1 > 0.5) + 0.3 * rng.standard_normal(n),   # noise of spread 0.3
    }).write_parquet(f"{name}.parquet")
features = ["x0", "x1", "x2"]

# an xgb.QuantileDMatrix; a list of files or a glob such as "train/part-*.arrow" works too
train = po.gbdt.xgboost_matrix("train.parquet", label="y", features=features)
valid = po.gbdt.xgboost_matrix("valid.parquet", label="y", features=features, ref=train)
history = {}
xgb.train({"tree_method": "hist", "max_depth": 4, "eta": 0.3}, train, 50,
          evals=[(valid, "valid")], evals_result=history, verbose_eval=False)
print(f"XGBoost:  validation RMSE {history['valid']['rmse'][-1]:.3f} after 50 rounds")

# a constructed lgb.Dataset, binned on LightGBM's own sample
train = po.gbdt.lightgbm_dataset("train.parquet", label="y", features=features)
valid = po.gbdt.lightgbm_dataset("valid.parquet", label="y", features=features, reference=train)
history = {}
lgb.train({"num_leaves": 15, "learning_rate": 0.3, "verbose": -1}, train, 50,
          valid_sets=[valid], valid_names=["valid"], callbacks=[lgb.record_evaluation(history)])
print(f"LightGBM: validation RMSE {history['valid']['l2'][-1] ** 0.5:.3f} after 50 rounds")
```

```text
XGBoost:  validation RMSE 0.312 after 50 rounds
LightGBM: validation RMSE 0.313 after 50 rounds
```

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
as soon as LightGBM's at a third of the memory.** On those 10 million rows, it began after 4.0 seconds for XGBoost,
against 8.7 for XGBoost's own iterator over Polars batches. For LightGBM it
began after 1.68 seconds streamed and 1.45 in memory, against 1.54 for its
own route from a collected frame at three times the memory. Each time is the
median of three interleaved runs on 8 threads, from Arrow IPC files, on an
Apple M4 Pro at a load average of 4.5 to 10, on 2026-10-10. The docstrings of
`xgboost_matrix` and `lightgbm_dataset` have every parameter.

**A Huber loss is the train call's objective, and its slope is a scale to
set from the residuals, not a fixed number.** `po.gbdt` builds the training
input and leaves the objective to the caller's `xgb.train` or `lgb.train`.
XGBoost's `{"objective": "reg:pseudohubererror", "huber_slope": δ}` and
LightGBM's `{"objective": "huber", "alpha": δ}` each take a Huber loss:
quadratic for a residual within `δ` and linear beyond it, so a wild label
pulls a tree no harder than a residual of `δ` does. LightGBM's is Huber's
piecewise loss; XGBoost's is the pseudo-Huber loss
`δ²(√(1 + (r/δ)²) − 1)`, a smooth version of it that bends from the one
regime to the other around `δ` rather than at it. `δ` is in the label's
units, and the defaults, 1.0 and 0.9, are fixed numbers. On a target whose
residuals are about 0.001, every row falls inside the cut and the loss is
squared error by another name; on one whose residuals are about 100, every
row falls beyond it. So set `δ = 1.345 · MAD / 0.6745`, where the MAD is the
median absolute deviation of the residuals of a first fit, such as a
squared-error model trained for a few rounds. At 1.345 standard deviations Huber's loss keeps
95% of least squares' efficiency on normal errors, and the MAD is not moved
by the wild labels the cut is meant to catch, where a root mean square is.
On a stream, `po.ewm_quantile` gives the same scale as it moves
([Quantiles and a robust scale](WINDOWS.md#quantiles-and-a-robust-scale)).

## What each step does

Each step of a build, for each library:

| step | XGBoost | LightGBM |
|---|---|---|
| read | Arrow IPC files in Rust, a record batch at a time (polars-arrow's reader); Parquet and a `LazyFrame` through Polars' `collect_batches`, each frame's chunks joined into one on the reader thread; two batches ready ahead | the same |
| bins | a sample of 200,000 rows by the SplitMix64 hash of their index, the same rows however the source is batched: the rows of smallest hash, or with weights the rows of largest `w / u`, `u` uniform from the hash (priority sampling); edges at the sample's weighted quantiles, or a bin per value for a feature of at most `max_bin` values; built into a dense `ref` | LightGBM's own, unweighted: `LGBM_SampleIndices`, those rows gathered in the first pass, `LGBM_DatasetCreateFromSampledColumn` |
| hand-over | each batch's columns as array interfaces on their Arrow buffers, the validity bitmap as the mask, into `XGQuantileDMatrixCreateFromCallback`; the batch's margin as its `base_margin` | float32 rows built in parallel, pushed with their labels, weights and initial scores a batch at a time, `LGBM_DatasetPushRowsWithMetadata`; or in memory, every batch as float32 Arrow, `LGBM_DatasetCreateFromArrowStream`, and every row's initial scores set after it |
| categoricals | a `pl.Enum`'s codes, feature type `c`, its levels planted in the `ref` | a `pl.Enum`'s codes in `categorical_feature`, and any integer column named |
| refusals | a null or non-finite label, weight or margin, a negative weight, a margin row whose number of values differs from the first row's or the `ref`'s, a `ref` with a margin and none here or none and one here, a `pl.Categorical`, a column of another type, a source whose passes give different rows, a `ref` built here on other features or on a `pl.Enum` of other categories, a library version outside the range the native engine is built for (XGBoost from 2.1, the first whose C API takes Arrow columns, to before 4.0) | the same, with LightGBM from 4.0 to before 5.0, and more than 2³¹ − 1 rows |

### XGBoost places its cuts where they are planted

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
the same trees.** It feeds numpy batches to an `xgboost.DataIter`, with the
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

### A categorical feature is a `pl.Enum`

**Give a categorical feature as a `pl.Enum`, whose levels are fixed before
the data, and a `pl.Categorical` is refused.** A `pl.Enum`'s codes go in as
a categorical feature, and its levels fix what each code means, so a model
reads them the same way at prediction. A `pl.Categorical`'s codes depend on
the process that made them. XGBoost bins a category code missing from the
`ref` without an error, so codes that meant something else in another
process would go unnoticed. Cast such a column to a `pl.Enum` with
`pl.col(c).cast(pl.Enum(levels))` before the build. A `ref=` or
`reference=` built here records each feature's levels, and a later frame
whose `pl.Enum` lists them in another order is refused by name, since its
codes would be read as the training levels. This recipe gives each
of four venues its own shift of the target, and predicts with the same
`pl.Enum` through `po.gbdt` itself, on the training bins:

```python
import numpy as np
import polars as pl
import polars_online as po
import xgboost as xgb

# Four venues, each shifting the target by its own amount.
rng = np.random.default_rng(7)
n = 20_000
venues = ["NYSE", "ARCA", "BATS", "IEX"]
effect = {"NYSE": 0.0, "ARCA": 1.0, "BATS": -1.0, "IEX": 2.0}          # each venue shifts the target
venue = rng.choice(venues, n)
frame = pl.DataFrame({
    "venue": pl.Series(venue, dtype=pl.Enum(venues)),                  # its levels, known before the data
    "x0": rng.standard_normal(n),
}).with_columns(y=pl.col("x0") + pl.Series([effect[v] for v in venue]) + 0.1 * rng.standard_normal(n))
frame.write_parquet("venues.parquet")

# Train on the venue as a category, then predict each venue at x0 = 0.
train = po.gbdt.xgboost_matrix("venues.parquet", label="y", features=["venue", "x0"])
print("feature types:", train.feature_types)                          # "c": a categorical feature
booster = xgb.train({"tree_method": "hist", "max_depth": 4}, train, 50)
test = pl.DataFrame({"venue": pl.Series(venues, dtype=pl.Enum(venues)), "x0": 0.0, "y": 0.0})
pred = booster.predict(po.gbdt.xgboost_matrix(test, label="y", features=["venue", "x0"], ref=train))
print("prediction at x0 = 0:", ", ".join(f"{v} {p:.1f}" for v, p in zip(venues, pred)))

# The same column as a pl.Categorical is refused.
frame.with_columns(pl.col("venue").cast(pl.String).cast(pl.Categorical)).write_parquet("cat.parquet")
try:
    po.gbdt.xgboost_matrix("cat.parquet", label="y", features=["venue", "x0"])
except ValueError:
    print("a pl.Categorical is refused: ValueError")
```

```text
feature types: ['c', 'float']
prediction at x0 = 0: NYSE -0.1, ARCA 1.1, BATS -1.1, IEX 2.1
a pl.Categorical is refused: ValueError
```

### A LightGBM dataset takes its binning parameters when it is built

**Pass `max_bin`, and every other parameter that changes the bins, to
`lightgbm_dataset`, because training cannot change them.** The dataset
it returns is already *constructed*, LightGBM's word for a dataset whose
bins are set and whose raw data is freed. `lgb.train` passes its
parameters through `LGBM_DatasetUpdateParamChecking`, which refuses a
changed `max_bin`, `data_random_seed`, `min_data_in_bin` and the rest, as
it refuses for LightGBM's own constructed dataset.
`max_bin` has a keyword of its own, and the others go in `params=`. The
dataset records its categorical features under `categorical_column`, the
key LightGBM's own `Dataset` records them under, so a validation set you
build yourself with `reference=train` raises no warning. In this recipe
the first dataset is built with 63 bins and trains, and the second, built
with the default, refuses 63 at training. LightGBM also writes the refusal
to standard error before it raises:

```python
import lightgbm as lgb
import numpy as np
import polars as pl
import polars_online as po

# One feature and its target, written to a file.
rng = np.random.default_rng(8)
n = 10_000
x0 = rng.standard_normal(n)
pl.DataFrame({"x0": x0, "y": x0 + rng.standard_normal(n)}).write_parquet("rows.parquet")

# Built with 63 bins, the dataset trains.
train = po.gbdt.lightgbm_dataset("rows.parquet", label="y", features=["x0"], max_bin=63)
booster = lgb.train({"verbose": -1}, train, 10)               # binned with 63 bins, as built
print("trees:", booster.num_trees())

# Built with the default, it refuses 63 bins at training.
again = po.gbdt.lightgbm_dataset("rows.parquet", label="y", features=["x0"])
try:
    lgb.train({"max_bin": 63, "verbose": -1}, again, 10)       # a binning parameter, too late
except lgb.basic.LightGBMError as e:
    print("LightGBMError:", str(e).splitlines()[0])
```

```text
trees: 10
LightGBMError: Cannot change max_bin after constructed Dataset handle.
```

### LightGBM's route in memory

**`in_memory="auto"` takes LightGBM's Arrow route when the features as
float32 take at most a quarter of the memory available.** On 1e7 × 50,
measured on 2026-10-10, that route was 14% sooner than the two passes, at a peak 1.4 times the float32
data (2.9 GB for 2.0 GB). Both routes hand LightGBM the same float32
values, so the choice, which depends on the machine, never changes the
trees.

## Training trees around an online model's prediction

**Give the trees an online model's prediction as the *margin*, the raw
score each row's prediction starts from, and they fit what the online
model leaves of the target.** The margin is `base_margin=` in
`po.gbdt.xgboost_matrix` and `init_score=` in `po.gbdt.lightgbm_dataset`,
each the name its library uses. A tree predicts within the range of the
targets it was trained on, so a level that has since moved past them is out
of its reach, but a margin is added after the trees and carries the level
with it. A model trained around a margin must be served with that margin
added back, so this task has a training half and a serving half.

You need:

- a Parquet file, or a query, of rows in clock order, with the features
  and the target;
- a spec for an online model of the target, whose prediction follows its
  level (see the README's [A first fit](../README.md#a-first-fit));
- XGBoost or LightGBM installed.

The steps, each numbered in the recipe below:

1. Run the online model over the rows with `lf.online.fit_predict`, and
   write each row's prediction beside it. Drop the rows it has no
   prediction for yet, since every row needs a margin.
2. Copy the prediction to a column of its own, to pass it as a feature
   too. A margin and a feature are each better on some data, and passing
   both was within 0.09 of the better of the two on the real data below.
3. Build the training input with the prediction as the margin, and train.
4. Serve: give XGBoost the rows' margin in its `DMatrix`, or add it to
   LightGBM's `predict(X, raw_score=True)`.
5. Check the test error against the same rows served without the margin.

This recipe plants a level that drifts above every value the training rows
reach, trains on the steps before 1,200 and tests on those from 1,600:

```python
import lightgbm as lgb
import numpy as np
import polars as pl
import polars_online as po
import xgboost as xgb

# Rows in clock order, 50 a step over 2,000 steps; a level no feature carries wanders and drifts up.
rng = np.random.default_rng(9)
steps, per = 2_000, 50
level = np.repeat(np.cumsum(rng.normal(0.0, 0.1, steps)) + 0.004 * np.arange(steps), per)
x0, x1 = rng.standard_normal((2, steps * per))
pl.DataFrame({
    "t": np.repeat(np.arange(steps), per), "x0": x0, "x1": x1,
    "y": level + np.sin(2 * x0) + (x1 > 0.5) + 0.3 * rng.standard_normal(steps * per),   # noise of spread 0.3
}).write_parquet("rows.parquet")

# 1. Each row's online prediction, made before the row is learned, written beside it.
online = po.spec.ewridge("online", targets=["y"], features=["x0", "x1"],
                         clock="t", half_life=5, gap_cap=10)          # follows the level within a few steps
(
    pl.scan_parquet("rows.parquet")
    .online.fit_predict([online]).online.unnest([online])             # adds pred_y, resid_y, ...
    .filter(pl.col("pred_y").is_not_null())                           # the first rows have no prediction yet
    # 2. The prediction again, as a feature.
    .select("t", "x0", "x1", "y", "pred_y", online_pred=pl.col("pred_y"))
    .sink_parquet("scored.parquet")
)
features = ["x0", "x1", "online_pred"]
train = pl.scan_parquet("scored.parquet").filter(pl.col("t") < 1_200)
test = pl.scan_parquet("scored.parquet").filter(pl.col("t") >= 1_600).collect()
low, high = train.select(low=pl.col("pred_y").min(), high=pl.col("pred_y").max()).collect().row(0)
print(f"online prediction: {low:.1f} to {high:.1f} on the training rows, "
      f"{test['pred_y'].min():.1f} to {test['pred_y'].max():.1f} on the test rows")
X, margin = test.select(features).to_numpy(), test["pred_y"].to_numpy()

def rmse(pred):
    return float(np.sqrt(np.mean((pred - test["y"].to_numpy()) ** 2)))

# 3. XGBoost, trained around the margin.
matrix = po.gbdt.xgboost_matrix(train, label="y", features=features, base_margin="pred_y")
booster = xgb.train({"tree_method": "hist", "max_depth": 4, "eta": 0.1}, matrix, 200)
# 4. Served with each row's margin in the DMatrix; 5. and, for the check, without it.
served = booster.predict(xgb.DMatrix(X, base_margin=margin, feature_names=features))
forgot = booster.predict(xgb.DMatrix(X, feature_names=features))
print(f"XGBoost:  test RMSE {rmse(served):.2f} served with the margin, {rmse(forgot):.2f} without it")

# 3. LightGBM, trained around the margin.
dataset = po.gbdt.lightgbm_dataset(train, label="y", features=features, init_score="pred_y")
booster = lgb.train({"num_leaves": 15, "learning_rate": 0.1, "verbose": -1}, dataset, 200)
# 4. Served as the trees' raw score plus each row's margin; 5. and without it.
served = booster.predict(X, raw_score=True) + margin
forgot = booster.predict(X, raw_score=True)
print(f"LightGBM: test RMSE {rmse(served):.2f} served with the margin, {rmse(forgot):.2f} without it")
```

```text
online prediction: -1.5 to 8.0 on the training rows, 7.3 to 13.6 on the test rows
XGBoost:  test RMSE 0.38 served with the margin, 7.02 without it
LightGBM: test RMSE 0.37 served with the margin, 10.27 without it
```

**The test rows' level reaches 13.6, above every training row's, and the
trees served with the margin come within 0.08 of the noise's spread of
0.3.** The same trees served without it miss by the level they were never
shown.

**A model trained with a margin must be served with it, and neither
library warns when it is not.** Its trees learned only what the margin
leaves, so a prediction without the margin lacks the level the margin
carried: in the recipe the error grows from 0.38 to 7.02. XGBoost's
`predict` adds the margin a `DMatrix` holds, so build every matrix you
predict on with `base_margin=`. A matrix built with `ref=` from
`po.gbdt.xgboost_matrix` does the same, and refuses a missing margin when
its `ref` had one. LightGBM's `predict` takes no margin, so add it to the
raw score yourself, and apply the objective's link after, such as the
logistic function for a binary objective. A validation set for early
stopping needs its own rows' margin too: `po.gbdt` refuses one built with
`ref=` or `reference=` without a margin when the training set had one.

**The online prediction leaks nothing into the margin.** The online model
predicts each row before it learns from that row's target, so the margin
of a training row never contains the target the trees are fitted to.

### When a margin helps, and when a feature does

**A margin wins where the level can leave the range the training rows
reached, and a feature can win where it cannot.** Measured on 2026-11-05
with LightGBM 4.7.0's trees, early stopping on a validation slice, each
data set split by time with no shuffling. The planted rows are 100 a clock step over 2,000 steps, with a
level that wanders and that no feature carries, and an online model of
half-life 5 steps (`ewridge` for the Gaussian target, `ftrl` for the
Bernoulli one), and noise of spread 1.0; each seed trains on the first
part of its clock and tests on the last.
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
ways scored 35.22, within 0.09 of it, which is why the recipe passes both.

## Polars' read-ahead

**`collect_batches` reads far ahead of its consumer, and on a source of 1 GB
it held the whole source before the first batch arrived.** Polars'
streaming engine prefetches row groups and files sized from its thread
count, so a consumer slower than the reader, as a library is, sees it hold
them. Measured as the growth of peak memory:

| source | consumer | peak growth |
|---|---|---:|
| 1.0 GB of Parquet, 50,000-row groups | nothing, `collect_batches` alone | 1.0 GB |
| the same | the same, `POLARS_ROW_GROUP_PREFETCH_SIZE=2` | 0.39 GB |
| 4.1 GB raw (1e7 × 50) of Parquet | nothing, a 30 ms pause a batch | 2.1 GB |
| 1.0 GB of Arrow IPC | `xgboost_matrix`, the native reader | 0.46 GB |
| the same, as `pl.scan_ipc` | `xgboost_matrix`, `collect_batches` | 2.1 GB |
| 1.0 GB of Parquet | `xgboost_matrix`, prefetch 1 and 4 Polars threads | 0.74 GB |

**So give the native engine Arrow IPC files where you can, because it reads
them itself, with two batches ready ahead.** For a Parquet or query source,
set Polars' prefetch and thread count as [Bound a run's
memory](how-to/bound-memory.md) says. py-polars 1.44.2 held as much as 2.0.0.

**A frame from `collect_batches` arrives in several chunks, and joining
them before the library is faster than handing the chunks over one by
one.** Each chunk was a batch of its own for XGBoost: 378 on 1e7 × 50,
where the IPC files give 132. XGBoost's construction took 4.4 seconds
against 3.9 with the chunks joined on the reader thread, the copy
overlapping XGBoost's work.
