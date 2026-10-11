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

**XGBoost's cuts come from a uniform sample of 200,000 rows, which removes
XGBoost's pass that sketches every row.** That pass is about half the time
before the first round. `sample_rows=None` keeps it, and with it XGBoost's
own trees. Otherwise the trees differ from XGBoost's own by no more than a
sample's seed moves them. On 10 million rows of 50 features, at 50 rounds,
XGBoost's sketch scored a validation RMSE of 1.2379, and cuts from five
seeds of a uniform sample 1.2347 to 1.2411. LightGBM's own sampling spans
1.2298 to 1.2443 over five seeds. `po.gbdt.lightgbm_dataset` returns a
constructed `lgb.Dataset`, binned on LightGBM's own sample, and holds the
data in memory only when it fits.

**The first round begins sooner than XGBoost's own route from Polars, and
as soon as LightGBM's at a third of the memory.** On those 10 million rows, it began after 4.0 seconds for XGBoost,
against 8.7 for XGBoost's own iterator over Polars batches. For LightGBM it
began after 1.68 seconds streamed and 1.45 in memory, against 1.54 for its
own route from a collected frame at three times the memory. Each time is the
median of three interleaved runs on 8 threads, from Arrow IPC files, on an
Apple M4 Pro at a load average of 4.5 to 10. The docstrings of
`xgboost_matrix` and `lightgbm_dataset` have every parameter.

## What each step does

Each step of a build, for each library:

| step | XGBoost | LightGBM |
|---|---|---|
| read | Arrow IPC files in Rust, a record batch at a time (polars-arrow's reader); Parquet and a `LazyFrame` through Polars' `collect_batches`, each frame's chunks joined into one on the reader thread; two batches ready ahead | the same |
| bins | a bottom-k sample: the 200,000 rows of smallest SplitMix64 hash of their index, the same rows however the source is batched; edges at the sample's quantiles, or a bin per value for a feature of at most `max_bin` values; built into a dense `ref` | LightGBM's own: `LGBM_SampleIndices`, those rows gathered in the first pass, `LGBM_DatasetCreateFromSampledColumn` |
| hand-over | each batch's columns as array interfaces on their Arrow buffers, the validity bitmap as the mask, into `XGQuantileDMatrixCreateFromCallback` | float32 rows built in parallel, `LGBM_DatasetPushRows`; or in memory, every batch as float32 Arrow, `LGBM_DatasetCreateFromArrowStream` |
| categoricals | a `pl.Enum`'s codes, feature type `c`, its levels planted in the `ref` | a `pl.Enum`'s codes in `categorical_feature`, and any integer column named |
| refusals | a null or non-finite label or weight, a negative weight, a `pl.Categorical`, a column of another type, a source whose passes give different rows, a `ref` built here on other features or on a `pl.Enum` of other categories, a library version outside the range the native engine is built for (XGBoost from 2.1, the first whose C API takes Arrow columns, to before 4.0) | the same, with LightGBM from 4.0 to before 5.0, and more than 2³¹ − 1 rows |

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

rng = np.random.default_rng(6)
n = 10_000
pl.DataFrame({
    "rating": rng.choice([1.0, 2.0, 3.5, 5.0], n),     # four distinct values
    "size": rng.standard_normal(n),                    # continuous
    "y": rng.standard_normal(n),
}).write_parquet("ratings.parquet")

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

rng = np.random.default_rng(6)
n = 10_000
x0, x1 = rng.standard_normal((2, n))
pl.DataFrame({"x0": x0, "x1": x1, "y": x0 * x1 + rng.standard_normal(n)}).write_parquet("pairs.parquet")

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

train = po.gbdt.xgboost_matrix("venues.parquet", label="y", features=["venue", "x0"])
print("feature types:", train.feature_types)                          # "c": a categorical feature
booster = xgb.train({"tree_method": "hist", "max_depth": 4}, train, 50)
test = pl.DataFrame({"venue": pl.Series(venues, dtype=pl.Enum(venues)), "x0": 0.0, "y": 0.0})
pred = booster.predict(po.gbdt.xgboost_matrix(test, label="y", features=["venue", "x0"], ref=train))
print("prediction at x0 = 0:", ", ".join(f"{v} {p:.1f}" for v, p in zip(venues, pred)))

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

rng = np.random.default_rng(8)
n = 10_000
x0 = rng.standard_normal(n)
pl.DataFrame({"x0": x0, "y": x0 + rng.standard_normal(n)}).write_parquet("rows.parquet")

train = po.gbdt.lightgbm_dataset("rows.parquet", label="y", features=["x0"], max_bin=63)
booster = lgb.train({"verbose": -1}, train, 10)               # binned with 63 bins, as built
print("trees:", booster.num_trees())

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
float32 take at most a quarter of the memory available.** On 1e7 × 50 that
route was 14% sooner than the two passes, at a peak 1.4 times the float32
data (2.9 GB for 2.0 GB). Both routes hand LightGBM the same float32
values, so the choice, which depends on the machine, never changes the
trees.

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
set Polars' prefetch and thread count as the README's [Tuning memory with
Polars' own settings](../README.md#tuning-memory-with-polars-own-settings)
says. py-polars 1.44.2 held as much as 2.0.0.

**A frame from `collect_batches` arrives in several chunks, and joining
them before the library is faster than handing the chunks over one by
one.** Each chunk was a batch of its own for XGBoost: 378 on 1e7 × 50,
where the IPC files give 132. XGBoost's construction took 4.4 seconds
against 3.9 with the chunks joined on the reader thread, the copy
overlapping XGBoost's work.
