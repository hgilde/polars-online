<!-- kind: how-to -->
# Read the coefficients

This how-to is for anyone who wants a regression's coefficients from a
model bank: at the end of a run, row by row as the fit moved, from a saved
output, or as a numpy matrix.

A [model bank](../GLOSSARY.md#the-model-bank-and-what-it-reads) gives the
coefficients in two places, and both are the fit *after* the model learns the row:

| from | the fit after | read with |
|---|---|---|
| the bank, live or loaded from a [state file](../GLOSSARY.md#saving-and-serving) with no data at hand | the last row each [group](../GLOSSARY.md#the-model-bank-and-what-it-reads) learned from | `bank.coef()` |
| the output, as columns that show the fit as it moved | its own row | the field `coef`, spread into columns by `.online.unnest` |

A row's own `pred` came from the fit *before* that row
([Predictions made before learning](../explanation/out-of-sample.md)).

## Prerequisites

These steps need the library and a model with coefficients:

- polars-online and numpy installed, and the README's
  [Example data](../../README.md#example-data) built. The examples read
  `lf`, a [query](../GLOSSARY.md#the-model-bank-and-what-it-reads) over
  `ticks.parquet`, and `df`, the same 400 rows as a frame, in which four
  stocks, `b0` to `b3` in the column `stock_id`, take turns. Their target
  `y` is `1 + 2·x0 − x1` plus noise.
- A spec of a model with coefficients, such as `ewridge`.
  [OUTPUTS.md](../OUTPUTS.md) lists each model's `coef` field.

## Read the coefficients four ways

Each way suits a different point in the pipeline: a bank at hand, a run's
output, a saved output, or numpy code that wants a matrix.

1. **From a bank, with `bank.coef()`,** one row per coefficient. Pivot it
   on `term` for a row per group and a column per term.
2. **From the output, with `.online.unnest(specs)` chained after
   `fit_predict`,** which spreads the `coef` list into a column per term,
   as `coef_y_x0`. Give the spec `coef_every=0` to write `coef` on every
   row but a skipped one.
3. **From a saved output, with `.online.unnest` on a query that reads the
   file with `pl.scan_parquet`,** given the
   specs, a bank, or the path of the state the run saved.
4. **As a matrix, by fixing the list's width on the rows that carry it,**
   with `.list.to_array(k)`, `k` being the height of
   `po.spec.coef_index(spec)`.

Here the bank, the output and a saved output each give the same fit, near
`df`'s truth. This code uses `lf` from [Example
data](../../README.md#example-data):

```python
ols = po.spec.ewridge("ols", targets=["y"], features=["x0", "x1"], clock="t",
                      half_life=600.0, gap_cap=300.0, group="stock_id",
                      coef_every=0)         # write coef on every row but a skipped one

# 1. From a bank -- live, or loaded from a state file with no data at hand.
bank = po.ModelBank([ols])
bank.fit(lf)
betas = bank.coef()                  # one row per coefficient: spec, group, instance, weight_sum, ..., term, coef
wide = betas.pivot("term", index=["group", "instance"], values="coef")   # a row per (group, instance), a column per term
print(wide.select("group", pl.col("intercept", "x0", "x1").round(2)).rows())

# 2. From the output, as columns: the fit as it moved, one row per row.
path = (
    lf.online.fit_predict([ols])
    .online.unnest([ols])            # pred_y, resid_y, weight_sum, coef_y_intercept, coef_y_x0, coef_y_x1, ...
    .select("t", "stock_id", "^coef_.*$")
    .collect()
)
for t, x0 in path.filter(pl.col("t").is_in([0.0, 4.0, 8.0, 40.0, 200.0, 396.0])).select("t", "coef_y_x0").iter_rows():
    print(f"b0 at t {t:>3.0f}: coef on x0 {'null' if x0 is None else f'{x0:.2f}'}")

# 3. From a saved output: unnest takes the specs, a bank, or the path of the state the run saved.
lf.online.fit_predict([ols], save_state="ols.state").sink_parquet("fitted.parquet")
saved = pl.scan_parquet("fitted.parquet").online.unnest("ols.state").collect()
print(saved.select(path.columns).equals(path))

# 4. As a matrix: the rows that carry coef, fixed to the layout's width.
k = po.spec.coef_index(ols).height   # 3: intercept, x0, x1
matrix = (
    pl.read_parquet("fitted.parquet")
    .select(pl.col("ols").struct.field("coef"))
    .filter(pl.col("coef").is_not_null())
    .select(pl.col("coef").list.to_array(k))
    .to_series()
    .to_numpy()
)
print(matrix.shape)
```

```text
[('b0', 1.0, 2.02, -1.01), ('b1', 1.01, 2.03, -1.06), ('b2', 0.93, 1.94, -0.99), ('b3', 1.03, 1.99, -0.97)]
b0 at t   0: coef on x0 null
b0 at t   4: coef on x0 null
b0 at t   8: coef on x0 2.72
b0 at t  40: coef on x0 2.24
b0 at t 200: coef on x0 2.06
b0 at t 396: coef on x0 2.02
True
(392, 3)
```

Stock `b0`'s rows are every fourth, from `t` 0. Its path starts null: a
fit needs a row per coefficient before any row is predicted from it, three
here, so the model solves nothing on its first two rows, where nothing
would read the solve. It starts on its third row, is near 2 by its
eleventh, and ends on the 2.02 `bank.coef()` reports. Each stock's first
two rows are the 8 of 400 that carry no `coef`.

## Choose which rows carry `coef`

**`coef_every=0` writes `coef` on every row but a skipped one**, as a list
of one float per term. Without a cadence the bank writes it on each
group's last accepted row in a chunk, and on none for a group whose rows
there were all skipped.

1. **Give `coef_every` a number of clock units,** or a duration such as
   `"5m"` on a temporal clock. It writes `coef` once the clock has moved
   that far since the group's last `coef` row, as `solve_every` schedules a
   solve.
2. **Or give `max_rows_between_coefs` a number of rows,** to write it
   after that many rows. With both, it is written by whichever comes
   first.

A cadence reads only the clock and the rows, so the rows that carry `coef`
do not depend on the chunking ([Chunk invariance and
streaming](../explanation/chunk-invariance-and-streaming.md)).

**`coef` is a list because a list stores nothing for a null row,** and
without a cadence `coef` is null on most rows. That is why step 4
keeps only the rows that carry it before it fixes the width.

**With `separator="."`, `.online.unnest` names each column after its spec
too,** as `ols.coef_y_x0`, as Polars' own `unnest(separator=)` does, so
two specs that fit one target unnest side by side.

## Read the blocks of a grid

**Several targets, or a [grid](../GLOSSARY.md#grids), put several blocks
in one `coef` list:** one per target and point of a grid of `ridge`
values, `feature_sets` or a `penalties`. Each half-life of a `half_life`
grid has a `coef` field of its own.

1. **Spread the blocks into columns with `.online.unnest([spec])`.** Each
   block's columns are named the way its `pred` field is ([Find an output
   field by what it holds](find-an-output-field.md)).
2. **Or read them from `bank.coef()`,** whose columns `target`, `ridge`,
   `feature_set` and `penalty` tell the blocks apart. Pivot it on `term`
   for a row per block.

Here `grid` has two half-lives and two ridge values, and the ridge of 0.5
shrinks each slope by more than a third. This code uses `df` from [Example
data](../../README.md#example-data):

```python
# A grid of two half-lives by two ridge values, fitted on df.
grid = po.spec.ewridge("m", targets=["y"], features=["x0", "x1"], clock="t",
                       gap_cap=300.0, half_life=[100.0, 500.0], ridge=[1e-6, 0.5])
grid_bank = po.ModelBank([grid])
out = grid_bank.fit_predict(df)                             # df's columns, plus "m"
# 1. The blocks spread into columns.
cols = out.online.unnest([grid])                            # coef@h100 and coef@h500, spread into columns
print(cols.select("pred_y__r0.5@h500", "coef_y_x0__r0.5@h500").columns)   # each block's columns named the way its pred field is
# 2. The same blocks from the bank, a row each.
betas = grid_bank.coef()                                    # target, ridge, feature_set and penalty tell the blocks apart
blocks_wide = betas.pivot("term", index=["instance", "ridge"], values="coef")   # a row per block
for instance, ridge, b0, b1, b2 in blocks_wide.sort("instance", "ridge").iter_rows():
    print(f"{instance} ridge {ridge}: intercept {b0:.2f}, x0 {b1:.2f}, x1 {b2:.2f}")
```

```text
['pred_y__r0.5@h500', 'coef_y_x0__r0.5@h500']
@h100 ridge 1e-06: intercept 0.99, x0 1.97, x1 -1.00
@h100 ridge 0.5: intercept 0.97, x0 1.25, x1 -0.60
@h500 ridge 1e-06: intercept 0.99, x0 1.99, x1 -1.00
@h500 ridge 0.5: intercept 0.97, x0 1.30, x1 -0.63
```

## Check that it worked

**The last row of each group's path in the output equals what
`bank.coef()` reports for that group,** as `b0`'s 2.02 does in the example. The
check in step 3 prints `True` when a saved output unnests to the same
columns as the live one.
