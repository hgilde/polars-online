# Training XGBoost and LightGBM: `po.gbdt`

**`po.gbdt` builds XGBoost's and LightGBM's own training objects from Arrow
IPC files, Parquet files or a query, so that `xgb.train` and `lgb.train` run
on data that does not fit in memory.** Both libraries bin each feature
before they boost, and binning reads every row. `po.gbdt` reads the rows in
batches, in Rust, and hands each library a batch's Arrow buffers with no
copy, a null by its validity bit. Each library is loaded from the package
you installed, so polars-online still depends on Polars alone.
This code uses the files `ticks.parquet` and `today.parquet` from
[Example data](../README.md#example-data):

```python
import xgboost as xgb

features = ["x0", "x1", "x2"]
# an xgb.QuantileDMatrix; a list of files or a glob such as "train/part-*.arrow" works too
train = po.gbdt.xgboost_matrix("ticks.parquet", label="y", features=features)
# the rows that follow, binned on the training cuts
valid = po.gbdt.xgboost_matrix("today.parquet", label="y", features=features, ref=train)
booster = xgb.train({"tree_method": "hist", "max_bin": 256}, train, 50,
                    evals=[(valid, "valid")], early_stopping_rounds=5, verbose_eval=False)
```

**XGBoost's cuts come from a uniform sample of 200,000 rows, which removes
XGBoost's pass that sketches every row.** `sample_rows=None` keeps that
pass, and with it XGBoost's own trees. `po.gbdt.lightgbm_dataset` returns a
constructed `lgb.Dataset`, binned on LightGBM's own sample, and holds the
data in memory only when it fits. On 10 million rows of 50 features, the
first round began after 4.0 seconds for XGBoost, against 8.7 for its own
iterator over Polars batches. For LightGBM it began after 1.7 seconds
streamed and 1.5 in memory, against 1.5 for its own route from a collected
frame at three times the memory. Each time is the median of three
interleaved runs on 8 threads, from Arrow IPC files, and the docstrings
have every parameter.

## What the component does

| step | XGBoost | LightGBM |
|---|---|---|
| read | Arrow IPC files in Rust, a record batch at a time (polars-arrow's reader); Parquet and a `LazyFrame` through Polars' `collect_batches`, each frame's chunks joined into one on the reader thread; two batches ready ahead | the same |
| bins | a bottom-k sample: the 200,000 rows of smallest SplitMix64 hash of their index, the same rows however the source is batched; edges at the sample's quantiles, or a bin per value for a feature of at most `max_bin` values; built into a dense `ref` | LightGBM's own: `LGBM_SampleIndices`, those rows gathered in the first pass, `LGBM_DatasetCreateFromSampledColumn` |
| hand-over | each batch's columns as array interfaces on their Arrow buffers, the validity bitmap as the mask, into `XGQuantileDMatrixCreateFromCallback` | float32 rows built in parallel, `LGBM_DatasetPushRows`; or in memory, every batch as float32 Arrow, `LGBM_DatasetCreateFromArrowStream` |
| categoricals | a `pl.Enum`'s codes, feature type `c`, its levels planted in the `ref` | a `pl.Enum`'s codes in `categorical_feature`, and any integer column named |
| refusals | a null or non-finite label or weight, a negative weight, a `pl.Categorical`, a column of another type, a source whose passes give different rows, a library version outside the C API's range (XGBoost 2.0 to 4.0, LightGBM 4.0 to 5.0) | the same, and more than 2³¹ − 1 rows |

**XGBoost places its cuts exactly where they are planted.** A `ref` column
of `d <= max_bin` distinct values `u_0 < .. < u_{d-1}` comes back as the
cuts `[-inf, u_1, .., u_{d-1}, u_{d-1} + |u_{d-1}| + 1e-5]` on 3.4.1, so
the planted values are the edges; `tests/test_gbdt.py` holds every feature's
cuts to the sample's definition. A category code missing from the `ref` is binned without an error,
which is why a categorical feature must be a `pl.Enum`, whose levels are
known before the data.

**The Python engine grows the native engine's trees.** `engine="python"`
feeds numpy batches to an `xgboost.DataIter`, with the `ref` built from the
same sample by the same Rust code, and a `lightgbm.Sequence` that answers
LightGBM's sample rows from a first pass. `engine="auto"` falls back to it,
with a warning, when the native engine cannot load a library.

**A LightGBM dataset built here is constructed, as any whose raw data is
freed: training may not change a dataset parameter.** `lgb.train` passes its
parameters through `LGBM_DatasetUpdateParamChecking`, which refuses a
changed `max_bin`, `data_random_seed`, `min_data_in_bin` and the rest
(`c_api.cpp:248`), as it refuses for LightGBM's own constructed dataset. So
those parameters go to `lightgbm_dataset(params=...)`. The dataset records
its categorical features under `categorical_column`, the key LightGBM's own
`Dataset` records them under, so a user's own validation set built with
`reference=train` raises no warning.

**`in_memory="auto"` takes LightGBM's Arrow route when the features as
float32 take at most a quarter of the memory available.** On 1e7 × 50 that
route was 14% sooner than the two passes, at a peak 1.4 times the float32
data (2.9 GB for 2.0 GB). Both routes hand LightGBM the same float32 values,
so the choice, which depends on the machine, never changes the trees.

## Polars' read-ahead

**`collect_batches` reads far ahead of its consumer, and on a source of 1 GB
it held the whole source before the first batch arrived.** Polars' streaming
engine prefetches row groups and files sized from its thread count, so a
consumer that is slower than the reader, as a library is, sees it hold
them:

| source | consumer | peak growth |
|---|---|---:|
| 1.0 GB of Parquet, 50,000-row groups | nothing, `collect_batches` alone | 1.0 GB |
| the same | the same, `POLARS_ROW_GROUP_PREFETCH_SIZE=2` | 0.39 GB |
| 4.1 GB raw (1e7 × 50) of Parquet | nothing, a 30 ms pause a batch | 2.1 GB |
| 1.0 GB of Arrow IPC | `xgboost_matrix`, the native reader | 0.46 GB |
| the same, as `pl.scan_ipc` | `xgboost_matrix`, `collect_batches` | 2.1 GB |
| 1.0 GB of Parquet | `xgboost_matrix`, prefetch 1 and 4 Polars threads | 0.74 GB |

So the native engine reads Arrow IPC files itself, and the docstrings point
a Parquet or query source at the README's "Tuning memory with Polars' own
settings". py-polars 1.44.2 held as much as 2.0.0.

**A frame from `collect_batches` arrives in several chunks, and joining
them before the library is faster than handing the chunks over one by
one.** Each chunk was a batch of its own for XGBoost, 378 on 1e7 × 50 where
the IPC files give 132, and XGBoost's construction took 4.4 seconds against
3.9 with the chunks joined on the reader thread, the copy overlapping
XGBoost's work.
