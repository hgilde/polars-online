<!-- kind: how-to -->
# Train XGBoost or LightGBM on files larger than memory

This how-to is for anyone who trains XGBoost or LightGBM on data that does
not fit in memory. `po.gbdt` builds each library's own training object
from Arrow IPC files, Parquet files or a query, for `xgb.train` and
`lgb.train` to run on. `po.gbdt` is unstable: it is built
on the libraries' C APIs, so its keywords may change in any release, and a
new library version may need a new release here.

## Prerequisites

These steps need the library, one of the two tree libraries, and the rows
in files:

- polars-online installed, and XGBoost or LightGBM. `po.gbdt` loads each
  from the package you installed. Outside XGBoost 2.1 to before 4.0 and
  LightGBM 4.0 to before 5.0, the default falls back to the slower Python
  engine.
- The training rows as Arrow IPC files, Parquet files or a query, with a
  numeric label, numeric features, and any categorical feature as a
  `pl.Enum`. Each recipe on this page writes its own files.
- To train around an online model's prediction: the rows in clock order,
  and a spec for an online model of the target whose prediction follows
  its level (see the README's [A first fit](../README.md#a-first-fit)).

[How `po.gbdt` builds training
data](explanation/how-po-gbdt-builds-training-data.md) explains how it
bins the rows, how its bins compare with each library's own, and why the
kind of file decides its memory.

## Train on files larger than memory

Use this for any training set too large to hold in memory as a frame.

1. **Save the training rows as Arrow IPC files where you can.** The
   native engine (`engine="native"`, tried first by the default `"auto"`:
   it reads the rows in Rust and hands them to each library's C API)
   reads them itself, with two chunks ready ahead, and
   holds the least memory. Parquet files and a query are read through
   Polars.
2. **Build the training object.** `po.gbdt.xgboost_matrix(source,
   label=..., features=...)` returns an `xgb.QuantileDMatrix`, and
   `po.gbdt.lightgbm_dataset(source, label=..., features=...)` a
   constructed `lgb.Dataset` (its bins set and its raw rows freed). The source is a frame, a file, a list of files, a
   glob such as `"train/part-*.arrow"`, or a query.
3. **Build the validation object on the training rows' bins,** with
   `ref=train` for XGBoost and `reference=train` for LightGBM.
4. **Train with `xgb.train` or `lgb.train`,** as on the library's own
   object.

This code writes a training file and a validation file whose target
depends on `x0` and `x1` but not `x2`, with noise of spread 0.3, and
trains each library on them for 50 rounds:

```python
import lightgbm as lgb
import numpy as np
import polars as pl
import polars_online as po
import xgboost as xgb

# 1. The training and validation files (Parquet here, read through Polars), with the target planted in x0 and x1.
rng = np.random.default_rng(5)
for name, n in (("train", 20_000), ("valid", 5_000)):
    x0, x1, x2 = rng.standard_normal((3, n))
    pl.DataFrame({
        "x0": x0, "x1": x1, "x2": x2,                                      # x2 plays no part
        "y": np.sin(2 * x0) + (x1 > 0.5) + 0.3 * rng.standard_normal(n),   # noise of spread 0.3
    }).write_parquet(f"{name}.parquet")
features = ["x0", "x1", "x2"]

# 2 and 3. An xgb.QuantileDMatrix, and the validation rows on its bins; 4. train.
train = po.gbdt.xgboost_matrix("train.parquet", label="y", features=features)
valid = po.gbdt.xgboost_matrix("valid.parquet", label="y", features=features, ref=train)
history = {}
xgb.train({"tree_method": "hist", "max_depth": 4, "eta": 0.3}, train, 50,
          evals=[(valid, "valid")], evals_result=history, verbose_eval=False)
print(f"XGBoost:  validation RMSE {history['valid']['rmse'][-1]:.3f} after 50 rounds")

# 2 and 3. A constructed lgb.Dataset, binned on LightGBM's own sample; 4. train.
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

Both libraries come within 0.013 of the noise's spread of 0.3. The
docstrings of `xgboost_matrix` and `lightgbm_dataset` have every
parameter.

## Set LightGBM's binning parameters when you build

**Pass `max_bin`, and every other parameter that changes the bins, to
`lightgbm_dataset`, because training cannot change them.** `lgb.train`
refuses a changed `max_bin`, `data_random_seed`, `min_data_in_bin` and
the rest on the dataset `lightgbm_dataset` returns, as it does on
LightGBM's own constructed dataset.

1. **Give `max_bin=` its own keyword in `po.gbdt.lightgbm_dataset`.**
2. **Give the other binning parameters in `params=`.**
3. **Leave them out of the parameters you pass to `lgb.train`.**

In this code the first dataset is built with 63 bins and trains, and the
second, built with the default, refuses 63 at training. LightGBM also
writes the refusal to standard error before it raises:

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

## Train with a categorical feature

**Give a categorical feature as a `pl.Enum`, whose levels are fixed before
the data.** `po.gbdt` refuses a `pl.Categorical`, whose codes depend on
the process that made them ([Why a categorical feature is a
`pl.Enum`](explanation/how-po-gbdt-builds-training-data.md#why-a-categorical-feature-is-a-plenum)).

1. **Cast the column to a `pl.Enum` before the build,** with
   `pl.col(c).cast(pl.Enum(levels))`, and name it among the features.
2. **Build and train as in [Train on files larger than
   memory](#train-on-files-larger-than-memory).** XGBoost marks the
   feature as type `c`.
3. **Predict on rows whose `pl.Enum` lists the same levels in the same
   order.** For XGBoost, build the matrix to predict on with
   `po.gbdt.xgboost_matrix(..., ref=train)`, which refuses a frame that
   lists the levels in another order. For LightGBM, pass the codes,
   `pl.col(c).to_physical()`, to the booster's `predict`.

This code gives each of four venues its own shift of the target, and
predicts with the same `pl.Enum` through `po.gbdt` itself, on the training
bins:

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
    "venue": pl.Series(venue, dtype=pl.Enum(venues)),                  # 1. its levels, known before the data
    "x0": rng.standard_normal(n),
}).with_columns(y=pl.col("x0") + pl.Series([effect[v] for v in venue]) + 0.1 * rng.standard_normal(n))
frame.write_parquet("venues.parquet")

# 2. Train on the venue as a category; 3. predict each venue at x0 = 0, on the training bins.
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

## Train with a Huber loss

**Give a Huber loss as the train call's objective, and set its slope `δ`
from the residuals of a first fit, not as a fixed number.** `δ` is in the
label's units, so the defaults, 1.0 for XGBoost and 0.9 for LightGBM, fit
only a label of the right scale ([Why a Huber loss's slope is set from
the residuals](explanation/how-po-gbdt-builds-training-data.md#why-a-huber-losss-slope-is-set-from-the-residuals)).

1. **Train a first fit with squared error,** such as a model trained for a
   few rounds.
2. **Take its residuals,** the labels less its predictions.
3. **Set `δ = 1.345 · MAD / 0.6745`,** where the MAD is the median absolute
   deviation of those residuals. On a stream, `po.ewm_quantile` gives the
   same scale as it moves ([Quantiles and a robust
   scale](WINDOWS.md#quantiles-and-a-robust-scale)).
4. **Train with the Huber objective and `δ`:** XGBoost's
   `{"objective": "reg:pseudohubererror", "huber_slope": δ}`, or
   LightGBM's `{"objective": "huber", "alpha": δ}`.

This code plants labels in the hundreds, 2% of them 5,000 too high, and
scores each objective against the planted truth on clean test rows:

```python
import numpy as np
import polars as pl
import polars_online as po
import xgboost as xgb

# Training rows whose labels are in hundreds, 2% of them wild; clean test rows.
rng = np.random.default_rng(11)
for name, n, wild in (("train", 20_000, 0.02), ("test", 5_000, 0.0)):
    x0, x1 = rng.standard_normal((2, n))
    truth = 100 * (np.sin(2 * x0) + (x1 > 0.5))
    y = truth + 30 * rng.standard_normal(n) + np.where(rng.random(n) < wild, 5_000.0, 0.0)
    pl.DataFrame({"x0": x0, "x1": x1, "y": y, "truth": truth}).write_parquet(f"{name}.parquet")
# The training matrix, and the clean test rows' features and truth.
features = ["x0", "x1"]
train = po.gbdt.xgboost_matrix("train.parquet", label="y", features=features)
test = pl.read_parquet("test.parquet")
X, truth = test.select(features).to_numpy(), test["truth"].to_numpy()


# Train a model with params, and score it against the planted truth.
def error(params, rounds=100):
    """The RMSE against the planted truth of a model trained with params."""
    booster = xgb.train({"tree_method": "hist", "max_depth": 4, "eta": 0.3, **params}, train, rounds)
    pred = booster.predict(xgb.DMatrix(X, feature_names=features))
    return float(np.sqrt(np.mean((pred - truth) ** 2)))


# 1 and 2. A first fit with squared error, for a few rounds, and its residuals.
first = xgb.train({"tree_method": "hist", "max_depth": 4, "eta": 0.3}, train, 10)
resid = train.get_label() - first.predict(train)
# 3. The slope from the residuals' MAD.
delta = 1.345 * np.median(np.abs(resid - np.median(resid))) / 0.6745
print(f"delta from the MAD: {delta:.1f}")
# 4. Each objective's error against the planted truth on the clean test rows.
print(f"squared error:            RMSE {error({}):.1f}")
print(f"pseudo-Huber, slope 1.0:  RMSE {error({'objective': 'reg:pseudohubererror'}):.1f}")
print(f"pseudo-Huber, slope {delta:.0f}:   RMSE {error({'objective': 'reg:pseudohubererror', 'huber_slope': delta}):.1f}")
```

```text
delta from the MAD: 64.1
squared error:            RMSE 178.6
pseudo-Huber, slope 1.0:  RMSE 35.2
pseudo-Huber, slope 64:   RMSE 9.2
```

Squared error follows the wild labels, and the default slope of 1.0 is far
below residuals of 30. The slope from the MAD comes within 9.2 of the
truth.

## Train trees around an online model's prediction

**Give the trees an online model's prediction as the *margin*, the raw
score each row's prediction starts from, and they fit what the online
model leaves of the target.** The margin is `base_margin=` in
`po.gbdt.xgboost_matrix` and `init_score=` in `po.gbdt.lightgbm_dataset`,
each the name its library uses. A tree predicts within the range of the
targets it was trained on, so a level that has since moved past them is out
of its reach, but a margin is added after the trees and carries the level
with it. A model trained around a margin must be served with that margin
added back, so this task has a training half and a serving half.

1. **Run the online model over the rows with `lf.online.fit_predict`.**
2. **Unpack its fields beside each row with `.online.unnest`.**
3. **Keep only the rows that have a prediction,** since every row needs a
   margin.
4. **Copy the prediction to a column of its own, to pass it as a feature
   too.** A margin and a feature are each better on some data ([When a
   margin helps, and when a feature
   does](explanation/how-po-gbdt-builds-training-data.md#when-a-margin-helps-and-when-a-feature-does)).
5. **Write the result to a file with `sink_parquet`,** which writes it
   without holding it all in memory.
6. **Build the training input with the prediction as the margin,** as
   `base_margin="pred_y"` or `init_score="pred_y"`.
7. **If you stop early on a validation set, build it with its own rows'
   margin.** `po.gbdt` refuses one built with `ref=` or `reference=`
   without a margin when the training set had one.
8. **Train with `xgb.train` or `lgb.train`.**
9. **Serve with the margin, since neither library warns when it is
   missing.** XGBoost's `predict` adds the margin a `DMatrix` holds, so
   build every matrix you predict on with `base_margin=`. LightGBM's
   `predict` takes no margin, so add it to `predict(X, raw_score=True)`,
   and apply the objective's link after, such as the logistic function for
   a binary objective.
10. **Check the test error against the same rows served without the
    margin.**

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

# 1 and 2. Each row's online prediction, made before the row is learned, unpacked beside it.
online = po.spec.ewridge("online", targets=["y"], features=["x0", "x1"],
                         clock="t", half_life=5, gap_cap=10)          # follows the level within a few steps
(
    pl.scan_parquet("rows.parquet")
    .online.fit_predict([online]).online.unnest([online])             # adds pred_y, resid_y, ...
    # 3. The rows with a prediction: the first rows have none yet.
    .filter(pl.col("pred_y").is_not_null())
    # 4. The prediction again, as a feature.
    .select("t", "x0", "x1", "y", "pred_y", online_pred=pl.col("pred_y"))
    # 5. Written to a file.
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

# 6 and 8. XGBoost: the margin in the training input, then train.
matrix = po.gbdt.xgboost_matrix(train, label="y", features=features, base_margin="pred_y")
booster = xgb.train({"tree_method": "hist", "max_depth": 4, "eta": 0.1}, matrix, 200)
# 9. Served with each row's margin in the DMatrix; 10. and, for the check, without it.
served = booster.predict(xgb.DMatrix(X, base_margin=margin, feature_names=features))
forgot = booster.predict(xgb.DMatrix(X, feature_names=features))
print(f"XGBoost:  test RMSE {rmse(served):.2f} served with the margin, {rmse(forgot):.2f} without it")

# 6 and 8. LightGBM: the margin in the training input, then train.
dataset = po.gbdt.lightgbm_dataset(train, label="y", features=features, init_score="pred_y")
booster = lgb.train({"num_leaves": 15, "learning_rate": 0.1, "verbose": -1}, dataset, 200)
# 9. Served as the trees' raw score plus each row's margin; 10. and without it.
served = booster.predict(X, raw_score=True) + margin
forgot = booster.predict(X, raw_score=True)
print(f"LightGBM: test RMSE {rmse(served):.2f} served with the margin, {rmse(forgot):.2f} without it")
```

```text
online prediction: -1.5 to 8.0 on the training rows, 7.3 to 13.6 on the test rows
XGBoost:  test RMSE 0.38 served with the margin, 7.02 without it
LightGBM: test RMSE 0.37 served with the margin, 10.27 without it
```

**The online prediction reaches 13.6 on the test rows, above every
training row's, and the
trees served with the margin come within 0.08 of the noise's spread of
0.3.** The same trees served without it miss by the level they were never
shown.

Its trees learned only what the margin leaves, so a prediction without
the margin lacks the level the margin carried: in the recipe the error
grows from 0.38 to 7.02.

## Check that it worked

The build worked when training runs and the validation error is near
the noise in the target, as in the first recipe. What else you may see,
and what to do:

| what you see | what it means | what to do |
|---|---|---|
| `ValueError` naming a `pl.Categorical` | its codes depend on the process that made them | cast the column to a `pl.Enum` ([Train with a categorical feature](#train-with-a-categorical-feature)) |
| `LightGBMError: Cannot change max_bin after constructed Dataset handle.` | a binning parameter reached `lgb.train`, after the bins were set | pass it to `lightgbm_dataset` ([Set LightGBM's binning parameters when you build](#set-lightgbms-binning-parameters-when-you-build)) |
| a `UserWarning` that the native engine cannot load a library, under `engine="auto"` | the build fell back to the Python engine, which grows the same trees more slowly | read the cause the warning names in parentheses, or pass `engine="python"` to choose that engine and silence the warning |
| XGBoost's trees differ from those its own sketch of every row gives | the cuts come from a sample of 200,000 rows | pass `sample_rows=None` to keep XGBoost's own sketch |
| memory grows far past a chunk on a Parquet file or a query | Polars reads ahead of a slower reader, as where a step that runs on every thread, such as `with_columns`, sits between the scan and `po.gbdt` ([Polars' read-ahead](explanation/how-po-gbdt-builds-training-data.md#polars-read-ahead)) | save the rows as Arrow IPC files, or lower `POLARS_ROW_GROUP_PREFETCH_SIZE` and `POLARS_MAX_THREADS` ([Tuning memory with Polars' own settings](../README.md#tuning-memory-with-polars-own-settings)) |
| a model trained around a margin predicts far from the targets | it was served without its margin | serve it with the margin, as in step 9 of [Train trees around an online model's prediction](#train-trees-around-an-online-models-prediction) |
