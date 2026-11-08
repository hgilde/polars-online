<!-- kind: tutorial -->
# Fit a model bank over parquet files in a query

This tutorial is for anyone new to polars-online who has parquet files too
large to hold in memory and wants to fit a model over them. You fit a
regression over two million rows in eight parquet files, in a Polars
[query](../GLOSSARY.md#the-model-bank-and-what-it-reads) (a chain of steps
that runs only when you ask for its result). The query writes its
predictions to a new parquet file, and in that file you find the
coefficients you planted in the rows.

To run the code, install the library and numpy with
`pip install "polars-online[numpy]"`. Run each step's code in the same
Python process, in a directory where it may write files.

A [spec](../GLOSSARY.md#the-model-bank-and-what-it-reads) describes one
model: which model it is and which columns it reads. A [model
bank](../GLOSSARY.md#the-model-bank-and-what-it-reads) fits a model for
each spec, all in one pass over the rows. Its
[stream](../GLOSSARY.md#the-model-bank-and-what-it-reads) is the rows it
reads, in the order it reads them. In a query, it takes them in
[chunks](../GLOSSARY.md#the-model-bank-and-what-it-reads), 100,000 rows at
a time by default.

## 1. Write the stream to parquet files

Make up a stream with a known answer, and write it to eight parquet files,
one file at a time. Its target `y` is `1 + 2·x0 − x1` plus noise with
standard deviation 0.5, so the coefficients the fit should find are an
intercept of 1, 2 on `x0` and −1 on `x1`:

```python
from pathlib import Path

import numpy as np
import polars as pl

import polars_online as po

# Eight files of 250,000 rows each, made and written one at a time.
Path("stream").mkdir(exist_ok=True)
rng = np.random.default_rng(0)
for part in range(8):
    x0, x1, noise = rng.standard_normal((3, 250_000))
    pl.DataFrame({
        "x0": x0,
        "x1": x1,
        "y": 1.0 + 2.0 * x0 - 1.0 * x1 + 0.5 * noise,   # the planted answer: 1, 2 and -1, noise of standard deviation 0.5
    }).write_parquet(f"stream/part-{part}.parquet")
print(sorted(p.name for p in Path("stream").glob("*.parquet")))
```

```text
['part-0.parquet', 'part-1.parquet', 'part-2.parquet', 'part-3.parquet', 'part-4.parquet', 'part-5.parquet', 'part-6.parquet', 'part-7.parquet']
```

## 2. Build the query

Build a query that reads the files and runs a bank over them.
`po.spec.ewridge` builds the spec of a ridge regression whose rows can be
weighted by their age, and `half_life=float("inf")` weighs every row the
same, however old. `pl.scan_parquet` reads the files as a query,
`.online.fit_predict` adds the bank to it, and `.online.unnest` turns the
bank's output into plain columns. Building the query reads no rows, but
its schema already names the columns it will write:

```python
spec = po.spec.ewridge(
    "ridge",                                  # the spec's name
    targets=["y"], features=["x0", "x1"],
    half_life=float("inf"),                   # every row counts the same
)
query = (
    pl.scan_parquet("stream/*.parquet")       # a query over the eight files: nothing is read yet
    .online.fit_predict([spec])               # a bank that starts with nothing learned, fed 100,000 rows at a time
    .online.unnest([spec])                    # the column "ridge" the bank adds, as plain columns: one per field, one per coefficient
)
print(query.collect_schema().names())
```

```text
['x0', 'x1', 'y', 'pred_y', 'resid_y', 'weight_sum', 'settled_frac', 'withheld_reason', 'coef_y_intercept', 'coef_y_x0', 'coef_y_x1', 'support_coef_y_intercept', 'support_coef_y_x0', 'support_coef_y_x1']
```

The query keeps the stream's three columns. The bank adds one [output
column](../GLOSSARY.md#what-a-bank-writes), `ridge`, which holds a
[field](../GLOSSARY.md#what-a-bank-writes) for each named value the spec
writes, and `.online.unnest` turns each field into a column of its own.
The fields `coef` and `support_coef` hold lists, so each becomes a column
for each coefficient. `pred_y` is each row's prediction of `y`, made from
the rows before it, and `resid_y` is `y` less `pred_y`. `weight_sum` is
the weight of the rows the prediction was made from: here, with every row
weighted 1, their count. The `coef_y_*` columns are the regression's
coefficients, written on the last row of each chunk and null on the
others. [OUTPUTS.md](../OUTPUTS.md#ewridge) lists every field this spec
writes.

## 3. Run the query to a file

Run the query with `sink_parquet`, which writes its result to a file
without holding it all in memory. The bank takes the stream one chunk at a
time, predicts each row, learns from it, and passes the chunk on to be
written. Between chunks, the bank keeps only its model's
[Gram](../GLOSSARY.md#fitting-a-model) (the weight of the rows, and the
columns' means and co-moments), a few numbers however many rows have
passed:

```python
query.sink_parquet("fitted.parquet")           # runs the query: reads, fits and writes a chunk at a time
print(pl.scan_parquet("fitted.parquet").select(pl.len()).collect().item(), "rows written")
```

```text
2000000 rows written
```

## 4. Read what it wrote

Read the output file back as a query. Look at its first rows, then check
the coefficients on its last row and the spread of its residuals against
the planted answer:

```python
fitted = pl.scan_parquet("fitted.parquet")     # a query over the output file

# The first five rows: each row's prediction from the rows before it, or null where it is withheld.
first = fitted.head(5).select("y", "pred_y", "weight_sum", "withheld_reason").collect()
for y, pred, weight, reason in first.iter_rows():
    pred_text = "null" if pred is None else f"{pred:.2f}"
    print(f"y {y:5.2f}   pred_y {pred_text:>5}   weight_sum {weight:.0f}   withheld_reason {reason or 'null'}")

# The coefficients on the last row, beside the planted ones.
last = fitted.tail(1).collect().row(0, named=True)
print(f"intercept {last['coef_y_intercept']:.3f} (planted 1), x0 {last['coef_y_x0']:.3f} (planted 2), "
      f"x1 {last['coef_y_x1']:.3f} (planted -1)")

# The residuals' root mean square, beside the planted noise's standard deviation.
spread = fitted.select(pl.col("resid_y").pow(2).mean().sqrt()).collect().item()
print(f"residuals' root mean square {spread:.3f} (planted noise 0.5)")
```

```text
y  0.22   pred_y  null   weight_sum 0   withheld_reason above_max_error_inflation
y  1.26   pred_y  null   weight_sum 1   withheld_reason above_max_error_inflation
y  1.65   pred_y  null   weight_sum 2   withheld_reason above_max_error_inflation
y  2.75   pred_y  1.87   weight_sum 3   withheld_reason null
y  0.71   pred_y  0.97   weight_sum 4   withheld_reason null
intercept 1.000 (planted 1), x0 2.000 (planted 2), x1 -1.000 (planted -1)
residuals' root mean square 0.500 (planted noise 0.5)
```

The first three rows have no prediction, since the model has learned too
few rows to predict from. `withheld_reason` says why: here the
[gate](../GLOSSARY.md#warm-up-and-readiness) `max_error_inflation` (a
parameter that withholds a prediction until a condition holds) is not yet
met. From the fourth row on, each row has a prediction, made before the
model learned the row, so `resid_y` is an
[out-of-sample](../GLOSSARY.md#fitting-a-model) residual. The coefficients
on the last row match the planted 1, 2 and −1 to three decimals, and the
residuals' spread matches the planted noise.

## Where to go next

To take the query to your own data, start from the task that matches it:

| to | read |
|---|---|
| fit a separate model for each value of a column, such as one per stock | [Fit one model per group](../how-to/fit-one-model-per-group.md) |
| fit a model that follows the recent rows, along a clock | [Follow a local fit through a day](local-fit-with-a-clock.md) |
| save what the bank learned and predict new rows from it | [Fit, save and serve a model bank](fit-save-and-serve.md) |
| change the chunk size, filter the output or write a wide spec's fields | [As a query](../../README.md#as-a-query-lfonlinefit_predict) |
| keep a large run's memory within a limit | [Bound a run's memory](../how-to/bound-memory.md) |
| know why the chunking never changes the numbers | [Chunk invariance and streaming](../explanation/chunk-invariance-and-streaming.md) |
| know why each row is predicted before it is learned | [Why every prediction is out-of-sample](../explanation/out-of-sample.md) |
| know why the first rows are withheld | [Warm-up and readiness](../explanation/warm-up-and-readiness.md) |
