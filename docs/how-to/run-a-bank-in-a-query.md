<!-- kind: how-to -->
# Run a bank in a query

This how-to is for anyone who runs a model bank inside a Polars query and
wants to choose the chunk size, write the output to a file, filter it, or
catch a misnamed column early.

A [model bank](../GLOSSARY.md#the-model-bank-and-what-it-reads) runs in one
of three ways, and each gives the same numbers, whatever the chunking of
the rows. From a bank object, the output can leave as Arrow, for a consumer
that is not Polars:

| way | the call | use it for | where |
|---|---|---|---|
| inside a Polars query | `lf.online.fit_predict(specs)` | ordinary Polars steps chained after the bank | this page |
| in your own Python loop | `bank.fit_predict(chunk)` | reading the bank between chunks | [Run a bank in a Python loop](run-a-bank-in-a-python-loop.md) |
| from a standalone command line, with no live Python at all | `polars-online` | a scheduled job, or a deployment with no Python | [Run a bank from the command line](run-a-bank-from-the-command-line.md) |
| output as Arrow, from a bank object | `fit_predict_arrow`, `predict_arrow` | a consumer that is not Polars | [Hand the output to Arrow](hand-the-output-to-arrow.md) |

[`lf.online.fit_predict(specs)`](https://hgilde.github.io/polars-online/namespaces.html#polars_online._frame.LazyFrameOnlineNamespace.fit_predict)
adds a model bank to a Polars
[query](../GLOSSARY.md#the-model-bank-and-what-it-reads), a `LazyFrame`,
which runs only when you ask for its result. `df.online.fit_predict(specs)`
does the same for a `DataFrame` in memory. When the query runs, it feeds
its rows to a bank that starts with nothing learned, `chunk_size` rows at a
time (100,000 by default). [Fit a model bank over parquet files in a
query](../tutorials/stream-from-parquet.md) walks through a first such
query from the start.

## Prerequisites

These steps need the library and a stream on disk:

- polars-online installed, and the README's
  [Example data](../../README.md#example-data) built. The examples read the
  file `ticks.parquet` and the same rows split across `ticks/*.parquet`,
  400 rows on the clock `t`. In them four stocks, `b0` to `b3` in the
  column `stock_id`, take turns row by row. `df`'s `y` is `1 + 2·x0 − x1`
  plus noise, so the fit of each stock finds those coefficients, and 0 for
  `x2`. They also read `lf`, a query over `ticks.parquet`, and `later`, a
  query over the 400 rows that follow it.
- One or more [specs](../GLOSSARY.md#the-model-bank-and-what-it-reads).

## Run the query to a file

Build the query from the files to the bank and on to the output file, then
run it once.

1. **Read the files with `pl.scan_parquet`,** which builds a query and
   reads nothing yet.
2. **Add the bank with `.online.fit_predict(specs, chunk_size=...)`.**
   The chunk size trades memory for overhead, and never changes the
   numbers ([Chunk invariance and
   streaming](../explanation/chunk-invariance-and-streaming.md)). Memory is
   the bank's state plus one chunk, however long the files ([Cut what
   Polars reads ahead](bound-memory.md#cut-what-polars-reads-ahead) covers
   a step before the bank that makes Polars read ahead).
3. **Chain any Polars steps after the bank,** such as a `filter` on the
   output, a `select` of the columns to keep, or a `head(5)`, which feeds
   the bank the first 5 rows and no more.
4. **Run the query.** Run it with `collect()` for one frame,
   `sink_parquet()` to write a file without holding the result in memory,
   or `collect_batches()` for one chunk at a time.

Here the query runs the same spec over `ticks/*.parquet` in one chunk, and
over `ticks.parquet` one row at a time, and the two agree to the bit. This
code uses the files `ticks.parquet` and `ticks/*.parquet` from [Example
data](../../README.md#example-data):

```python
spec = po.spec.ewridge(                                   # a ridge regression of y for each stock, at two ridge values
    "ridge",
    targets=["y"], features=["x0", "x1", "x2"],
    clock="t", half_life=600.0, gap_cap=300.0,
    group="stock_id", ridge=[1e-6, 0.1], standardize=True,
)
# Run the query to a file, then read what it wrote.
(
    pl.scan_parquet("ticks/*.parquet")                    # 1. a query over the files; nothing is read yet
    .online.fit_predict([spec], chunk_size=100_000)       # 2. a bank with nothing learned yet; every run starts from the same place
    .filter(pl.col("ridge").struct.field("weight_sum") > 50)   # 3. after the bank: filters what comes out, never what the bank learns from
    .select("t", "stock_id", "ridge")                     # Polars reads only these columns, and those the specs read, from the files
    .sink_parquet("fitted.parquet")                       # 4. runs the query; memory is state + one chunk, however long the files
)
fitted = pl.read_parquet("fitted.parquet")
print(f"{fitted.height} rows past the filter; the record's fields: {fitted['ridge'].struct.fields}")

# The same spec, one row per chunk.
one_row_chunks = pl.scan_parquet("ticks.parquet").online.fit_predict([spec], chunk_size=1).collect()
same = one_row_chunks.filter(pl.col("ridge").struct.field("weight_sum") > 50)["ridge"].struct.field("pred_y__r0.1")
print(f"largest difference, chunks of 100,000 against chunks of 1: {(fitted['ridge'].struct.field('pred_y__r0.1') - same).abs().max()}")

# One stock's fit on its last row.
last = fitted.filter(pl.col("stock_id") == "b3").row(-1, named=True)
intercept, b0, b1, b2 = last["ridge"]["coef"][:4]        # the fit at the first ridge value; the second's follow
print(f"b3's coefficients on its last row: intercept {intercept:.2f}, x0 {b0:.2f}, x1 {b1:.2f}, x2 {b2:.2f}")
```

```text
172 rows past the filter; the record's fields: ['pred_y__r0.000001', 'resid_y__r0.000001', 'pred_y__r0.1', 'resid_y__r0.1', 'weight_sum', 'settled_frac', 'withheld_reason', 'coef', 'support_coef']
largest difference, chunks of 100,000 against chunks of 1: 0.0
b3's coefficients on its last row: intercept 1.04, x0 1.99, x1 -0.97, x2 -0.13
```

`"ridge"` is one column whose value in each row is a record with named
fields ([Find an output field](find-an-output-field.md) lists them).
`weight_sum` is the weight a fit rests on, so the filter keeps each stock's
rows once its fit has seen about 50 rows' worth.

## Check the result

**Compare the run with one of another chunk size, and the fit with what
the data plants.** In the output of [Run the query to a
file](#run-the-query-to-a-file), chunks of 100,000 and chunks of 1 differ
by 0.0. Stock `b3`'s coefficients on its last row are near the intercept 1,
the 2 on `x0` and the −1 on `x1` that `df` plants, and near 0 on `x2`.

## Unnest a wide spec before writing it

**Unnest a wide spec before `sink_parquet`.** Polars' parquet writer
encodes a record's fields several times slower than the same values as
plain columns, and a wide spec writes many fields: an `ew_cov` over 20
series writes 233.

1. **Chain `.online.unnest([spec])` after the bank,** which turns each
   field into a column of its own.
2. **Write the result,** as
   `.online.unnest([spec]).sink_parquet("fitted.parquet")`.

The command line's `--output-layout flat` does the same for its files
([RUNNER.md](../RUNNER.md)).

## Filter after the bank

**Filter after the bank, not before it.** A `filter` after the bank keeps
the query running a chunk at a time. One before the bank makes Polars hold
several blocks of each parquet file per thread. A run over 12M rows
peaked at 2.53 GB with the filter before the bank, against 0.78 GB with
the same filter after it ([Filter after the
bank](../PERFORMANCE.md#filter-after-the-bank) gives the figures, and [How
it was measured](../PERFORMANCE.md#how-it-was-measured-and-why-not-rss) the
method).

To keep a model from learning some rows while still predicting them, give
them weight `0` in place of a filter ([Hold a row
back](hold-a-row-back.md#keep-a-model-from-learning-rows-in-a-query)).

## Catch a misnamed column before the query runs

**`lf.online.fit_predict` checks each spec against the query's columns when
it is called, so a missing or non-numeric column raises `ValueError` before
any row is read.** The bank refuses what only the values reveal, such as a
null clock or a step back, while the query runs. That error arrives as
`polars.exceptions.ComputeError` under py-polars 1.x, and as the bank's own
`ValueError` under 2.0, since how a query wraps an error is Polars' to
change. Either way its message names the row and the way out.

1. **Call `.online.fit_predict` inside a `try`,** before you run the
   query.
2. **Catch `ValueError`,** and read which column the message names.

This code uses `ticks.parquet` from [Example
data](../../README.md#example-data):

```python
misspelled = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x9"], half_life=100.0)
try:
    pl.scan_parquet("ticks.parquet").online.fit_predict([misspelled])   # refused here: the query never runs
except ValueError as e:
    print(e)
```

```text
spec "ridge": feature column "x9" is not in the input, whose columns are ["t", "ts", "x0", "x1", "x2", "signal_a", "signal_b", "y", "ret", "stock_id", "group", "session", "venue"]. Name a column the input has, or add it to the input.
```

## Call the functions a type checker sees

**With a type checker, call `po.fit_predict`,
[`po.predict`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.predict)
and [`po.unnest`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.unnest)
in place of the `.online` methods.** The package adds those methods to
Polars' frames when it is imported, so a type checker cannot see them.

1. **Replace each `.online` method with the function of the same name.**
2. **Pass the frame as the function's first argument.**

This code uses `lf` and `later` from [Example
data](../../README.md#example-data):

```python
spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], clock="t", half_life=600.0, gap_cap=300.0)
fitted = po.fit_predict(lf, [spec], save_state="ridge.state")   # lf.online.fit_predict([spec], save_state="ridge.state")
columns = po.unnest(fitted, [spec]).collect()                   # fitted.online.unnest([spec]); running it saves the state
predicted = po.predict(later, "ridge.state").collect()             # later.online.predict("ridge.state")
```
