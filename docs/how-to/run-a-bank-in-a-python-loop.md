<!-- kind: how-to -->
# Run a bank in a Python loop

This how-to is for anyone who feeds a model bank from a Python loop of
their own, and wants to read the bank between chunks or drop the groups
that have gone quiet.

A [model bank](../GLOSSARY.md#the-model-bank-and-what-it-reads) fits a set
of models, one for each
[spec](../GLOSSARY.md#the-model-bank-and-what-it-reads) and
[group](../GLOSSARY.md#the-model-bank-and-what-it-reads), over the same
rows. In a loop you hold the bank as a
[`ModelBank`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.ModelBank)
object, pass it each [chunk](../GLOSSARY.md#the-model-bank-and-what-it-reads),
and ask it what it holds between chunks ([Inspect a
bank](inspect-a-bank.md)). [Run a bank in a query](run-a-bank-in-a-query.md)
runs the same bank inside a Polars query instead.

## Prerequisites

These steps need the library and a stream:

- polars-online installed, and the README's
  [Example data](../../README.md#example-data) built. The examples read
  `df`, a frame of 400 rows on the clock `t`, in which four stocks, `b0` to
  `b3` in the column `stock_id`, take turns row by row. Its target `y` is
  `1 + 2·x0 − x1` plus noise. They also read the file `ticks.parquet`, the
  same rows on disk.
- One or more specs.

## Feed the bank in a loop

Build the bank once, then hand it the stream a chunk at a time.

**Feed a bank from one thread at a time, though `predict`, which learns
nothing, may run from any number of threads at once.** A bank reads one
ordered stream, so a call that finds it busy on another thread raises
`RuntimeError`: wait for that call to return, or give each thread its own
bank.

1. **Build a bank from the specs with `po.ModelBank(specs)`.** It starts
   with nothing learned.
2. **Feed it the stream a chunk at a time with
   `bank.fit_predict_batches(query, chunk_size=...)`,** which reads a query
   or a `DataFrame` in chunks of the size given, and yields each chunk's
   output. With chunks of your own, call `bank.fit_predict(chunk)` on each.
3. **Between chunks, read what the bank holds,** as `bank.coef()` or
   `bank.rows_fed`.
4. **Save the bank when the stream ends,** with `bank.save(path)`.

Here the bank reads 100 rows at a time, and stock `b0`'s coefficient on
`x0` stays near the 2 that `df` plants. This code uses `ticks.parquet`
from [Example data](../../README.md#example-data):

```python
spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"],
                       clock="t", half_life=600.0, gap_cap=300.0, group="stock_id")
bank = po.ModelBank([spec])                                # 1. nothing learned yet
for out in bank.fit_predict_batches(pl.scan_parquet("ticks.parquet"), chunk_size=100):   # 2.
    # out: the chunk's columns, plus one column per spec. 3. Between chunks, the bank answers questions:
    x0 = bank.coef().filter(pl.col("group") == "b0", pl.col("term") == "x0")["coef"][0]
    print(f"after {bank.rows_fed} rows: b0's coefficient on x0 is {x0:.3f}")
bank.save("bank.state")                                    # 4. written whole or not at all: a temporary file, then a rename

# With chunks of your own: for chunk in pl.scan_parquet("ticks.parquet").collect_batches(): bank.fit_predict(chunk)
# A run whose product is the state: po.ModelBank([spec]).fit(query) learns from every row and keeps no output.
```

```text
after 100 rows: b0's coefficient on x0 is 2.105
after 200 rows: b0's coefficient on x0 is 2.055
after 300 rows: b0's coefficient on x0 is 2.087
after 400 rows: b0's coefficient on x0 is 2.023
```

## Check the result

**Read the coefficient after each chunk against the value the data
plants.** In the output of [Feed the bank in a
loop](#feed-the-bank-in-a-loop), stock `b0`'s coefficient on `x0` stays
near 2 from the first chunk to the last, ending at 2.023.

## Drop the groups that have gone quiet

**A bank keeps every group until you drop it, so in a long-running bank,
drop the quiet ones with `bank.drop_groups`.** A dropped group starts
over, with nothing learned, if it appears again.

1. **List the groups with `bank.groups()`,** which gives each group's
   `clock_last` in the clock column's own dtype. That is a `Datetime` in
   its unit and zone, an integer clock in its own width, or a float clock
   as a `Float64`.
2. **Compare `clock_last` with the current time of the same kind,** and
   keep the groups quiet for longer than you choose.
3. **Drop them with `bank.drop_groups(...)`,** which returns how many it
   dropped.

In `df` the four stocks take turns, so each last ticked at a different
`t`. This code uses `df` and `ticks.parquet` from [Example
data](../../README.md#example-data):

```python
spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"],
                       clock="t", half_life=600.0, gap_cap=300.0, group="stock_id")
bank = po.ModelBank([spec])
bank.fit(pl.scan_parquet("ticks.parquet"))
for group, rows, clock_last in bank.groups().select("group", "rows_fed", "clock_last").iter_rows():   # 1.
    print(f"{group}: {rows} rows, the last at t = {clock_last:.0f}")
now = df["t"].max()                                        # the current time on the clock t; on a temporal clock, a datetime
stale = bank.groups().filter(pl.col("clock_last") < now - 2)   # 2. quiet for more than 2 units of t
print(f"dropped {bank.drop_groups(stale['group'])}; left: {bank.groups()['group'].to_list()}")   # 3.
```

```text
b0: 100 rows, the last at t = 396
b1: 100 rows, the last at t = 397
b2: 100 rows, the last at t = 398
b3: 100 rows, the last at t = 399
dropped 1; left: ['b1', 'b2', 'b3']
```

`b0`, last seen at `t = 396`, was quiet for more than 2 units of `t` at
`t = 399`, so it is the one dropped.
