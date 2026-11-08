<!-- kind: how-to -->
# Set the chunk size

This how-to is for anyone running a model bank over a query or a file who
wants to change how many rows the bank takes at a time. Larger chunks make
a wide run faster, and smaller ones make a run hold less memory.

## Prerequisites

These steps need the library and a run that reads a query:

- polars-online installed, and the README's
  [Example data](../../README.md#example-data) built. The examples read
  the file `ticks.parquet` and `df`, the same 400 rows of four stocks in
  the column `stock_id`.
- A bank run over a query or a frame.

## Set how many rows the bank takes at a time

A [model bank](../GLOSSARY.md#the-model-bank-and-what-it-reads) takes its
rows a [chunk](../GLOSSARY.md#the-model-bank-and-what-it-reads) at a time,
and you give the chunk's size, in rows, to the call that runs the bank.
**The chunk size changes the speed and the memory, never a prediction.**
It changes only which rows carry `coef`, and `support_coef` and the
standard errors beside it (`se_coef`, `se_coef_hc0`, `se_coef_hac`): by
default, each group's last row not skipped in each chunk. [Why the
chunking cannot change the
numbers](../explanation/chunk-invariance-and-streaming.md#why-the-chunking-cannot-change-the-numbers)
explains why.

1. **Pass the keyword `chunk_size`, 100,000 by default,** to the call that
   runs the bank. `lf.online.fit_predict`, `lf.online.predict`,
   `po.fit_predict`, `po.predict`, `ModelBank.fit_predict_batches`,
   `ModelBank.fit`, `po.eval.grid_search`, `with_windows` and
   `refresh_time` take it.
2. **To cut a frame into chunks, pass it to `ModelBank.fit` or
   `ModelBank.fit_predict_batches` with `chunk_size`.** Without
   `chunk_size`, those two take a frame whole, and a frame passed to
   `ModelBank.fit_predict` is one chunk, whatever its size.
3. **To report the coefficients on rows of your choosing whatever the
   chunk size, set `coef_every` or `max_rows_between_coefs` on the spec.**
   [OUTPUTS.md](../OUTPUTS.md#fields-most-models-write) says which rows
   each fills.

## Check that the predictions did not move

**Run the same spec at two chunk sizes, and compare the predictions.**
This code reads the same file 10 rows at a time and all at once, with four
groups. So the small chunks write the coefficients on 4 rows of each of 40
chunks, less the first chunk's four, where each group has had two or three
rows, too few for a fit.

This code uses the file `ticks.parquet` and `df` from [Example
data](../../README.md#example-data):

```python
# The same spec over chunks of 10 rows, the whole file, and a frame.
spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], clock="t", gap_cap=300.0,
                       half_life=50.0, group="stock_id")
query = pl.scan_parquet("ticks.parquet")
small = query.online.fit_predict([spec], chunk_size=10).collect()       # read 10 rows at a time
large = query.online.fit_predict([spec], chunk_size=100_000).collect()  # the whole file at once
whole = po.ModelBank([spec]).fit_predict(df)                            # a frame: one chunk, whatever its size

# Compare the predictions, and count the rows that carry coef.
def field(out, name):
    return out["ridge"].struct.field(name)

print("largest difference in pred_y:", (field(small, "pred_y") - field(large, "pred_y")).abs().max())
print("the same as the frame's:", field(large, "pred_y").equals(field(whole, "pred_y")))
print("rows carrying coef:", field(small, "coef").is_not_null().sum(), "at 10 rows a chunk,",
      field(large, "coef").is_not_null().sum(), "at 100,000")
```

```text
largest difference in pred_y: 0.0
the same as the frame's: True
rows carrying coef: 156 at 10 rows a chunk, 4 at 100,000
```

## Use larger chunks for a wide frame

**For a wide frame, use larger chunks, because each chunk carries a fixed
overhead.** At 10,000 columns, handing a chunk across from Polars alone
takes about 8 ms a call: 4 µs of every row at 2,000 rows a call, and 0.4 µs
at 20,000 (measured on 2026-09-08). That time is pyo3-polars extracting
the frame's columns, about 0.8 µs a column.

## Use smaller chunks for less memory

**For less memory, use smaller chunks, because a few are in flight at
once:** three on the command line, and a few more in Python. [Where the
time and memory go](../explanation/where-the-time-and-memory-go.md#where-the-memory-goes)
names the other parts of a run's memory, and [Bound a run's
memory](bound-memory.md) cuts each.
