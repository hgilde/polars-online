<!-- kind: how-to -->
# Weight rows

This how-to is for anyone who wants some rows to count more than others in
a fit, and wants to know what the weights do to `weight_sum` and to
`min_weight`.

A [weight](../GLOSSARY.md#the-model-bank-and-what-it-reads) says how much a
row counts in a fit. In a model that keeps running sums, a row of weight 2
moves the fit as two rows of weight 1 do.

**A weight of 0 is legal, and a negative weight is refused.** A row of
weight 0 is predicted and moves the clock, but the model learns nothing
from it
([Hold a row back](hold-a-row-back.md)).

## Prerequisites

These steps need the library and a column of weights:

- polars-online installed, and the README's
  [Example data](../../README.md#example-data) built. The example reads
  `df`, a frame of 400 rows whose target `y` is `1 + 2·x0 − x1` plus noise.
- A numeric column of row weights, each 0 or above, in the stream or made
  in the query before the bank with Polars' `with_columns`.

## Weight the rows

Name the weights in the spec, and set its readiness gate in the same
units.

1. **Name the column of row weights in the spec's `weight`,** as
   `weight="w"`.
2. **If the model keeps means, scale `min_weight` by the same factor as
   the weights.** Scaling every weight by the same factor changes nothing
   but `weight_sum`, apart from how soon an `ewridge`'s `sigma` and
   `zscore` start: their floor of a row per unknown is a weight too. So a
   `min_weight` set for weights of 1 withholds too little under weights of
   10. Otherwise, read how the model takes a
   weight in [Check how the model reads a
   weight](#check-how-the-model-reads-a-weight).
3. **Fit the stream,** and read `weight_sum`, the weight learned before
   each row.

This code gives row 100 a weight of 0 and every other row a weight of 1,
then scales every weight by 10. With no decay, `weight_sum` counts the
weight learned before each row: row 100 is predicted, but the row after it
finds the weight unchanged. It reads `df` from [Example
data](../../README.md#example-data):

```python
weighted = po.spec.ewridge(
    "weighted", targets=["y"], features=["x0", "x1"], half_life=float("inf"),
    weight="w",                # 1. a column of row weights
)
# Row 100 weighs nothing; then every weight times 10.
ones = df.with_columns(w=pl.when(pl.int_range(pl.len()) == 100).then(0.0).otherwise(1.0))
tens = ones.with_columns(pl.col("w") * 10)
fields = [po.ModelBank([weighted]).fit_predict(frame)["weighted"].struct for frame in (ones, tens)]   # 3. fit each
# The rows around row 100, and what the scaling moved.
for row in [99, 100, 101, 102]:
    print(f"row {row}: pred_y {fields[0].field('pred_y')[row]:.4f}, weight_sum {fields[0].field('weight_sum')[row]:g}")
print(f"weights times 10: pred_y moves by at most {(fields[0].field('pred_y') - fields[1].field('pred_y')).abs().max():g}, "
      f"weight_sum is {fields[1].field('weight_sum')[-1] / fields[0].field('weight_sum')[-1]:g} times as large")
```

```text
row 99: pred_y -2.2004, weight_sum 99
row 100: pred_y 0.6661, weight_sum 100
row 101: pred_y 2.5369, weight_sum 100
row 102: pred_y 2.4487, weight_sum 101
weights times 10: pred_y moves by at most 0, weight_sum is 10 times as large
```

## Check the result

**Read `weight_sum` on the rows around a row you weighted, and the
predictions under the scaled weights.** In the output of [Weight the
rows](#weight-the-rows), row 101 finds `weight_sum` at 100, unchanged from
row 100's, since row 100 weighed nothing. Scaling every weight by 10 moves
no prediction, and makes `weight_sum` 10 times as large.

## Check how the model reads a weight

**Nine models read a weight differently.** Every other model reads a
weight as [Weight the rows](#weight-the-rows) step 2 says:

| model | a row's weight is |
|---|---|
| `rls` | on the sum scale, against a ridge of fixed size: a heavier stream outweighs it sooner |
| `kalman` | a scale on the observation's precision: its update equation divides the observation noise `R_j` by `w`, so with a fixed `obs_var=` the observation's variance is `obs_var / w` |
| `sgd` | a step size: the gradient carries it |
| `ftrl` | an importance weight, as Vowpal Wabbit's, against penalties of fixed size |
| `pa` | a step size below 1; a weight above 1 counts as 1 |
| `hmm` | on the sum scale, against the transitions' prior count |
| `rcov` | 0 or 1, since a realized covariance is a sum over returns; any other weight is refused |
| `seqtest` | refused, since every learned row is one trial: to skip a row, null its target |
| `audit` | refused: it counts values, whatever their row weighs |
