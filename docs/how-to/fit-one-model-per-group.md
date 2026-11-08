<!-- kind: how-to -->
# Fit one model per group

This how-to is for anyone who wants a separate model for each value of a
column, such as one for each stock, fitted in one pass over the stream.

A [model bank](../GLOSSARY.md#the-model-bank-and-what-it-reads) fits one
model for each [spec](../GLOSSARY.md#the-model-bank-and-what-it-reads)
(the description of a model) and each
[group](../GLOSSARY.md#the-model-bank-and-what-it-reads): the rows that
share one value of the column `group=` names. Each group's model sees only
its own rows, on its own [clock](../GLOSSARY.md#time), so a group's numbers
are those of the same spec fitted to that group's rows alone.

## Prerequisites

These steps need the library and a column whose values name the groups:

- polars-online installed, and the README's
  [Example data](../../README.md#example-data) built. The examples read
  `df`, a frame of 400 rows in which four stocks, `b0` to `b3` in the
  column `stock_id`, take turns row by row. They also read `lf`, a
  [query](../GLOSSARY.md#the-model-bank-and-what-it-reads) over the same
  rows saved as `ticks.parquet`.
- A column of the stream, or one you make in the query, whose value says
  which group a row belongs to. The column may be of an integer, float,
  text, boolean or temporal type.

## Fit a model for each value of a column

Name the column in the spec, and the bank fits every group's model in the
same pass over the rows.

1. **Name the column in the spec's `group=`,** as
   `group="stock_id"`. The bank fits a separate model for each distinct
   value.
2. **Fit the bank as you would without groups,** with `fit_predict`, `fit`
   or a query's `online.fit_predict`. Each row's output comes from the
   model of that row's group.
3. **List the groups with `bank.groups()`,** which gives each group's key,
   the rows it was fed and its last clock value.

This code fits one model per stock. It uses `df` from [Example
data](../../README.md#example-data):

```python
# 1. One model per stock.
per_stock = po.spec.ewridge(
    "per_stock", targets=["y"], features=["x0", "x1"], clock="t", half_life=600.0, gap_cap=300.0,
    group="stock_id",           # one model per distinct value of this column
)
# 2. Fit them all in one pass.
bank = po.ModelBank([per_stock])
together = bank.fit_predict(df)
# 3. The groups the bank has seen.
for row in bank.groups().iter_rows(named=True):
    print(f"group {row['group']}: {row['rows_fed']} rows, the last at t = {row['clock_last']}")
```

```text
group b0: 100 rows, the last at t = 396.0
group b1: 100 rows, the last at t = 397.0
group b2: 100 rows, the last at t = 398.0
group b3: 100 rows, the last at t = 399.0
```

## Group on a column made in the query

**When the groups are not a column of the stream, make the column with
Polars' `with_columns` in the query before the bank, and name it as the
spec's `group`.** A seasonal baseline is one such case. `holt`, a model of
a target's level and trend, has no seasonal term, so group the spec on the
phase, such as the hour of the day. The bank then fits one level and trend
per hour.

1. **Make the phase a column with `with_columns`,** as
   `hour=pl.col("ts").dt.hour()`.
2. **Name that column as the spec's `group`,** as `group="hour"`.
3. **Run the query,** here with `collect`, which runs it and returns its
   result.

This code uses `lf` from [Example data](../../README.md#example-data):

```python
seasonal = po.spec.holt("seasonal", targets=["y"], clock="ts", gap_cap="1h",
                        half_life="2h", group="hour")         # 2. one level and trend per hour of the day
by_hour = lf.with_columns(hour=pl.col("ts").dt.hour()).online.fit_predict([seasonal]).collect()   # 1 and 3: the phase, as a column
# One model per hour: each hour's rows, and the prediction its own model gave its last row.
per_hour = by_hour.group_by("hour", maintain_order=True).agg(
    rows=pl.len(), last_pred=pl.col("seasonal").struct.field("pred_y").last())
for row in per_hour.iter_rows(named=True):
    print(f"hour {row['hour']}: {row['rows']} rows, last prediction {row['last_pred']:.3f}")
```

```text
hour 9: 30 rows, last prediction 0.696
hour 10: 60 rows, last prediction 1.576
hour 11: 60 rows, last prediction 0.783
hour 12: 60 rows, last prediction 0.948
hour 13: 60 rows, last prediction 0.480
hour 14: 60 rows, last prediction 0.724
hour 15: 60 rows, last prediction 0.466
hour 16: 10 rows, last prediction 3.179
```

## Check that it worked

**Compare one group's predictions in the bank with the same spec fitted to
that group's rows alone.** Drop `group` from the spec, feed it the group's
rows, and the predictions agree. This code compares stock `b0`'s. It uses
`df` from [Example data](../../README.md#example-data), and `per_stock`
and `together` from [Fit a model for each value of a
column](#fit-a-model-for-each-value-of-a-column):

```python
# Stock b0's rows alone give the same predictions.
alone = {**per_stock, "group": None}           # a spec is a dict: the same spec without a group
b0 = po.ModelBank([alone]).fit_predict(df.filter(pl.col("stock_id") == "b0"))
in_bank = together.filter(pl.col("stock_id") == "b0")["per_stock"].struct.field("pred_y")
print(f"b0 in the bank against b0 alone: {(in_bank - b0['per_stock'].struct.field('pred_y')).abs().max()} largest difference")
```

```text
b0 in the bank against b0 alone: 0.0 largest difference
```

## Keep the group column's type

**Give a group or session column the same kind of type in every chunk and
file,** casting it with Polars' `cast` in the query where one file or chunk
differs. The column keeps the type its first chunk had. A key is
its value's text, so an `Int64` `1` and a `Float64` `1.0` would be two
groups. The bank refuses a column of another kind by name, and keeps the
kind in its state: integer, float, text, boolean, or a temporal type with
its unit and zone. An `Int32` after an `Int64`, or a `Categorical` after a
`String`, is the same kind and goes on. `bank.groups()` and the other
readers return the keys as text, an integer column's in numeric order, the
null group first.

## Bound the memory of many groups

**Without `group_close`, the bank keeps every group it has seen.** To free
the groups that have gone quiet, find them with `bank.groups()` and drop
them with `bank.drop_groups(...)` ([In a
loop](../../README.md#in-a-loop-modelbank)).

**With `group_close`, the bank writes out a [closed
group](../GLOSSARY.md#saving-and-serving)'s running sums and frees its
memory,** which keeps its memory bounded when new group
values never stop appearing. `group_close="monotone"` closes a group once
a higher key arrives, so it needs the keys in order, as a block number or a
date is. `group_close="session"` closes a group once its `session` value
changes. [One row per finished
group](../../README.md#one-row-per-finished-group) shows what is written.
