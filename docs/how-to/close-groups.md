<!-- kind: how-to -->
# Close groups and read each one's row

This how-to is for anyone whose stream keeps bringing new group values,
such as a day id, a session id or a block number, and who wants each
group's running sums written out once its rows have ended, with the
bank's memory kept bounded.

A [model bank](../GLOSSARY.md#the-model-bank-and-what-it-reads) fits a
model for each [group](../GLOSSARY.md#the-model-bank-and-what-it-reads),
and without `group_close` it keeps every group's state until
`bank.drop_groups` removes it. With `group_close`, a group whose rows have
ended becomes a [closed group](../GLOSSARY.md#saving-and-serving): the
bank writes out its running sums, a row per half-life and per
[Gram](../GLOSSARY.md#fitting-a-model), and frees its state.

## Prerequisites

These steps need the library and a column whose values name the groups:

- polars-online and numpy installed, and
  [Example data](../EXAMPLE-DATA.md) built. The examples read
  `df`, a frame of 400 rows with the features `x0` and `x1`.
- A group column whose groups end in one of two ways: its key only ever
  rises, as a block number or a date cast to an integer does, or a
  `session` column changes where a group's span ends.
- No `embargo`, `label_clock` or [formula target](../GLOSSARY.md#fitting-a-model) on the spec. `group_close`
  is refused beside each, since each holds a group's rows past its end.

## Close each group as its rows end

`group_close` takes one of two modes, which decide when a group is
closed:

| `group_close` | a group is closed when | refused |
|---|---|---|
| `"monotone"` | its key is below the largest seen, since the key only ever rises | a null key, or a chunk whose keys are not in rising order, naming the row; a float or temporal key column, which Polars' `cast` turns into an integer or text in the query before the bank |
| `"session"` | its `session` value changes | a spec without a `session` column, or with `session_gap` |

1. **Name the group column in `group=`, and set `group_close`** to
   `"monotone"` for keys that only rise, or to `"session"` for spans a
   `session` column marks.
2. **Feed the bank as usual,** with `fit_predict`, `fit` or a query.
3. **Take the closed rows out with `bank.closed_groups()`,** oldest first.
   Called with `drop=False`, it reads them and leaves them queued.

In either mode the last group never closes, since no row proves it ended,
and `bank.gram()` still reads it.

**Under `"monotone"`, the bank orders an integer key as a number and a
String or `Categorical` key as text, so `"9"` comes after `"10"`.**
Polars' own `sort` agrees, so sort the input by the key before the bank.

**Each closed row holds, bit for bit, what `bank.gram()` would have
returned for that group just before it closed**, its matrix packed as the
upper triangle. It adds the span's own `rows_fed`, `rows_learned`,
`clock_min` and `clock_max`. A model with coefficients adds its `coef`, an
`ew_cov` with `pca` its eigendecomposition, and a `marginal` its pairs.

Here a block id rises every 100 rows, so blocks 0 to 2 close, and block 3,
the last, stays open. `po.gram.from_row` turns a closed row back into the
dict `gram()` returns, so everything in `po.gram` works on it. This code
uses `df` from [Example data](../EXAMPLE-DATA.md):

```python
# 1. A Gram per block, each closed once a higher block arrives.
blocks = po.spec.ew_cov("cov", features=["x0", "x1"], lam=1.0,   # lam=1.0: no decay
                        group="block", group_close="monotone")   # a block is closed once a higher one arrives
by_block = df.with_columns(block=pl.int_range(pl.len()) // 100)   # a block id that rises every 100 rows

# 2 and 3. Feed the rows, and take the closed rows out.
bank = po.ModelBank([blocks])
bank.fit_predict(by_block)
closed = bank.closed_groups()          # blocks 0, 1 and 2, oldest first: one half-life and one Gram, so one row each
print(closed.select("group", "rows_fed", "rows_learned").rows())

first = po.gram.from_row(closed.head(1))   # block 0's row, as the dict gram() returns
print(f"{po.gram.correlation(first)[0, 1]:.3f}")   # everything in po.gram works on it: x0 and x1's correlation in block 0
```

```text
[('0', 100, 100), ('1', 100, 100), ('2', 100, 100)]
0.021
```

**Closed rows wait in the bank until `bank.closed_groups()` takes them
out.** Rows not yet read are saved with the state, so a loop that saves
between chunks loses none. Read them as the stream runs to keep the state
bounded.

## Write the closed rows to a file

With `closed_groups=`, the bank's queue is emptied after every chunk, so
neither the bank nor its saved state holds the closed rows. From Python,
the rows read out are kept by the process and written to the file once,
when the chunks stop. So memory grows with the number of groups that
close, never with the rows. `fit` and `fit_predict_batches` write the file
even when the run stops early or fails, but `lf.online.fit_predict` writes
it only after the last chunk, so a run that fails writes no file. The
command line writes the rows as it goes.

1. **Pass a path as `closed_groups=`** to `lf.online.fit_predict`,
   `fit_predict_batches` or `fit`, or `--closed-groups` to the command line
   ([Closed groups to a sidecar
   file](../RUNNER.md#closed-groups-to-a-sidecar-file)).
2. **Run the query or the bank.** With `fit`, which keeps no per-row
   output, the file is the run's whole product: a stream too large for
   memory goes in, and one row per block comes out.

Here the blocked stream is saved to disk, learned from as it is read back,
and each closed block's row goes to `blocks.parquet`. This code uses `df`
from [Example data](../EXAMPLE-DATA.md):

```python
blocks = po.spec.ew_cov("cov", features=["x0", "x1"], lam=1.0, group="block", group_close="monotone")
df.with_columns(block=pl.int_range(pl.len()) // 100).write_parquet("by_block.parquet")   # the stream, on disk
po.ModelBank([blocks]).fit(pl.scan_parquet("by_block.parquet"), closed_groups="blocks.parquet")   # 1 and 2: read back as a query
print(pl.read_parquet("blocks.parquet").select("group", "rows_fed").rows())
```

```text
[('0', 100), ('1', 100), ('2', 100)]
```

## Check that it worked

**Set a closed row beside the Gram of a bank that has seen only that
group's rows, and the two agree to the bit.** In this code, block 0's bank
has not seen block 1, so its block 0 is still open. This code uses `blocks`,
`by_block` and `closed` from [Close each group as its rows
end](#close-each-group-as-its-rows-end), which read `df` from [Example
data](../EXAMPLE-DATA.md):

```python
# Block 0 in a bank that has seen only its rows, so it is still open: its Gram, packed as a closed row's is.
open_bank = po.ModelBank([blocks])
open_bank.fit_predict(by_block.head(100))
still_open = open_bank.gram("cov", layout="packed")[0]
row = closed.row(0, named=True)
print(all(row[k] == still_open[k].tolist() for k in ["means", "comoments"]), row["weight_sum"] == still_open["weight_sum"])
```

```text
True True
```

[Repair and score a correlation
matrix](repair-and-score-a-correlation-matrix.md) reads what a closed
row's correlations say.
