<!-- kind: how-to -->
# Hold a row back

This how-to is for anyone who wants some rows predicted but not learned,
such as a row whose target is wrong, or rows from a venue the model should
not learn from.

## Prerequisites

These steps need the library and a stream:

- polars-online installed, and the README's
  [Example data](../../README.md#example-data) built. The examples read
  `df`, a frame of 400 rows on the clock `t`, and the file `ticks.parquet`,
  the same rows on disk. Its column `venue` takes turns between `"X"` and
  `"Y"` row by row.
- For a weight of 0, a spec that names a weight column in `weight=`
  ([Weight rows](weight-rows.md)).

## Choose a way to hold a row back

**A null target keeps the model from learning that target, and leaves the
row predicted and the clock moving.** A null in any feature or in the
weight makes a [skipped row](../GLOSSARY.md#the-model-bank-and-what-it-reads)
instead, which the model neither predicts nor learns, and at which it takes
no clock step. In a feature, target or weight, NaN, ±inf and any magnitude
above `1e100` count as
[null](../GLOSSARY.md#the-model-bank-and-what-it-reads), so a sentinel such
as `1e300` never reaches a model. A sentinel inside that range, such as
`-999`, does: make it null with Polars' `when` in the query before the
bank ([Check a stream's data quality](check-data-quality.md) finds one). A clock value cannot be null, and is refused by row
instead ([What the bank refuses](../MODELS.md#what-the-bank-refuses)). A
null target is one of three ways to keep a model from learning a row,
beside a weight of 0 and `predict`:

| | the row is predicted | the clock advances | the fit moves | `weight_sum` | use it to |
|---|---|---|---|---|---|
| **a null target** | yes | yes | not that target's, except under `target_gaps="pairwise"` | counts the row. In a regression model the target's own weight only decays, so a long stretch of nulls can take that target below its `min_weight` | leave a target out |
| **weight `0`** | yes | yes | no | keeps decaying, so a long stretch can fall below `min_weight` | keep a row's place in the stream |
| **`predict`** | yes | no | no | frozen | serve ([Serve from a saved state](serve-from-a-saved-state.md)) |

Choose the way by what the row should still do: move the clock, count in
`weight_sum`, or neither.

1. **To leave a row's target out, make it null** with Polars'
   `with_columns` in the query before the bank.
2. **To keep the row's place in the stream but learn nothing from it, give
   it weight `0`:** make the weight column with `with_columns`, and name it
   in the spec's `weight`.
3. **To [serve](../GLOSSARY.md#saving-and-serving) rows from a fitted
   bank without moving its clock or its fit, pass them to `bank.predict`**
   in place of `fit_predict`.

This code holds row 100 of `df` back from a model in each of the three
ways, beside a run that learns it. For each it prints whether row 100 was
predicted, the clock the model stands at after it, whether the fit moved, and
`weight_sum` on the row after it. It uses `df` from [Example
data](../../README.md#example-data):

```python
# One spec, and four copies of the stream that differ only at row 100.
held = po.spec.ewridge("held", targets=["y"], features=["x0", "x1"], clock="t", half_life=100.0, gap_cap=300.0,
                       weight="w", max_rows_between_solves=1)   # a solve on every row, so each fit read is current
rows = df.with_columns(w=pl.lit(1.0))
row_100 = pl.int_range(pl.len()) == 100
ways = {
    "learned": rows,
    "a null target": rows.with_columns(y=pl.when(row_100).then(None).otherwise(pl.col("y"))),   # 1.
    "weight 0": rows.with_columns(w=pl.when(row_100).then(0.0).otherwise(pl.col("w"))),          # 2.
    "predict": rows,                                                                             # 3.
}
# A bank that has seen rows 0 to 99, to tell whether the fit moved.
before = po.ModelBank([held])
before.fit_predict(rows.head(100))                            # rows 0 to 99
print(f"{'':<14} {'predicted':>9} {'clock':>6} {'fit moved':>9} {'weight_sum':>10}")
# Each way: rows 0 to 99, then row 100, then row 101.
for way, frame in ways.items():
    bank = po.ModelBank([held])
    bank.fit_predict(frame.head(100))
    if way == "predict":
        predicted = bank.predict(frame.slice(100, 1))         # predicts row 100 and learns nothing
    else:
        predicted = bank.fit_predict(frame.slice(100, 1))
    # What row 100 did: the clock, the fit, and the next row's weight_sum.
    clock = bank.groups()["clock_last"][0]
    moved = bank.coef()["coef"].to_list() != before.coef()["coef"].to_list()
    after = bank.fit_predict(frame.slice(101, 1))["held"].struct.field("weight_sum")[0]
    is_predicted = predicted["held"].struct.field("pred_y")[0] is not None
    print(f"{way:<14} {'yes' if is_predicted else 'no':>9} {clock:>6g} {'yes' if moved else 'no':>9} {after:>10.2f}")
```

```text
               predicted  clock fit moved weight_sum
learned              yes    100       yes      72.89
a null target        yes    100        no      72.89
weight 0             yes    100        no      71.89
predict              yes     99        no      72.39
```

## Check the result

**Compare each way's row with the run that learns it.** In the output of
[Choose a way to hold a row back](#choose-a-way-to-hold-a-row-back), every
way predicts row 100, and none moves the fit. A null target and weight `0`
advance the clock to 100, where `predict` leaves it at 99. A null target
leaves `weight_sum` as learning the row does, weight `0` leaves it a row
lighter, and `predict` leaves it between the two.

## Keep a model from learning rows in a query

**To keep a model from learning rows, give them weight `0` before the
bank, in place of a filter:** make the column with `with_columns`, and name
it in the spec's `weight`. The rows still come out predicted, and the clock
advances through them, so no gap opens where they were.

1. **Make a weight column of 1 on the rows to learn from, and 0 on the
   rest,** with `pl.when(...).then(1.0).otherwise(0.0)` in `with_columns`.
2. **Name it in the spec's `weight`,** as `weight="w"`.
3. **Run the query,** and count the rows predicted and the weight learned.

Here the model learns only venue X's 200 rows. Under
`half_life=float("inf")` no row's weight decays, so `weight_sum` counts the
rows learned before each row. This code uses `ticks.parquet` from [Example
data](../../README.md#example-data):

```python
learn_from_x = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"],
                               clock="t", half_life=float("inf"), gap_cap=300.0, weight="w")   # 2.
fitted = (
    pl.scan_parquet("ticks.parquet")
    .with_columns(pl.when(pl.col("venue") == "X").then(1.0).otherwise(0.0).alias("w"))  # 1. learn from venue X only
    .online.fit_predict([learn_from_x])
    .collect()                                                                           # 3.
)
# How many rows were predicted, and how much weight was learned.
predicted = fitted["ridge"].struct.field("pred_y").is_not_null().sum()
print(f"rows predicted: {predicted} of {fitted.height}; weight_sum on the last row: {fitted['ridge'].struct.field('weight_sum')[-1]:.0f}")
```

```text
rows predicted: 395 of 400; weight_sum on the last row: 200
```

## Know what a null skips

**A null in any feature, or in the weight, skips the row**: its outputs are
null, the model learns nothing from it, and the model takes no clock step
at it. The step at the next row it learns is measured from the last row it
did not skip.

**A null in one target skips learning that target alone.** The row still
writes that target's `pred`, and leaves its `resid` null. In `rls`, whose
targets share one factor, the model learns the row for none of them. In `ewridge` and
`lasso`, whose targets share running sums, the row still reaches the other
targets' sums. Under the default `target_gaps="own_rows"`, the missing
target keeps its own copy of the sums from that row on, and over the rows
it lacks the copy only ages, so its fit holds still. Under `"pairwise"` one
set of sums learns every row, so a null target's
coefficients drift with the feature noise.
