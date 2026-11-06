<!-- kind: explanation -->
# Why every prediction is out-of-sample

This page explains, for anyone who reads a model bank's predictions or
diagnostics, why each one was made before the model learned its row, and
what that guarantees about them.

**Every prediction polars-online writes as it learns, and every diagnostic
it reads from the residuals, is out-of-sample.** A model predicts each row
from what it has learned from the rows before it, and only then learns
from the row. So no row's target reaches its own prediction. A
[model bank](../GLOSSARY.md#the-model-bank-and-what-it-reads) fits a set of
models over the same rows, one for each
[spec](../GLOSSARY.md#the-model-bank-and-what-it-reads) (the description
of a model) and each [group](../GLOSSARY.md#the-model-bank-and-what-it-reads)
(the rows that share one value of the column `group=` names). A
[diagnostic](../GLOSSARY.md#fitting-a-model) is a statistic a spec writes
beside its predictions, such as a calibration slope or a test for a break.

## The order at each row

A bank reads its rows one at a time, and does three things with each:

1. **Predict.** Each model computes the row's prediction,
   `pred_<target>`, from what it has learned so far, and nothing of this
   row's target.
2. **Compare.** The bank computes the residual, `resid_<target>`, the
   target less the prediction of step 1, and the diagnostics that read the
   residuals. Most diagnostics are read from the rows learned before the
   row. A few read the row's own residual: a studentized residual scales it
   by a spread from the rows learned before it, and the drift test that
   `emit_drift` switches on counts it. Each of
   them reads the residual of step 1, made before the model learned the
   row.
3. **Learn.** The model takes the row into its
   [state](../GLOSSARY.md#saving-and-serving), so the row counts toward
   the predictions of the rows after it. Under an `embargo`, a delay on the
   [clock](../GLOSSARY.md#time), the model learns the row later, once the
   delay has passed.

The bank may learn a row before it writes the row's comparison, and under
an `embargo` it learns the row long after. Either way, the comparison reads
only the prediction of step 1, never the state after learning. Every way of
running a bank keeps this order: a bank object, a query (a chain of Polars
steps that runs only when you ask for its result) and the command line.

A few fields are not predictions, and are read after the row: `coef`,
`support_coef` and the standard errors describe the fit after the row. A
diagnostic that reads no prediction, such as the feature health that
`emit_feature_health` switches on, describes the features of the rows
learned before the row.

So a prediction made in one pass over a stream is the prediction a bank
that had learned only the rows before it would make. This code fits `df`
from the README's [Example data](../../README.md#example-data) in one run,
then predicts row 300 with a bank that has learned only rows 0 to 299:

```python
spec = po.spec.ewridge("m", targets=["y"], features=["x0", "x1"], clock="t", half_life=100.0, gap_cap=300.0)
one_run = po.ModelBank([spec]).fit_predict(df)["m"].struct.field("pred_y")   # every row, in one pass

first_300 = po.ModelBank([spec])
first_300.fit_predict(df.head(300))                                           # learns rows 0 to 299
alone = first_300.predict(df.slice(300, 1))["m"].struct.field("pred_y")[0]    # predicts row 300, learning nothing
print(f"row 300: {one_run[300]:.6f} in one run, {alone:.6f} from rows 0 to 299 alone")
```

```text
row 300: 2.142699 in one run, 2.142699 from rows 0 to 299 alone
```

## What predicting before learning guarantees

**A residual measures an error on a row the model had not seen.** A fit's
own residuals, read after it has learned every row, are too small: the fit
has bent toward each row, the more so the more parameters it has. An
out-of-sample residual has no such bias. So the mean of the squared
residuals over a stream estimates the error of the fits along the stream
on rows they had not seen, with no rows held out and without fitting the
model again. Dawid (1984) set out this way of judging a model: each
prediction is judged on a target value that came after it. The mean
includes the early rows, where the fit had learned little, so it is an
average over the stream and not the error of the final fit alone.

**Every diagnostic that reads the residuals judges the predictions the
model actually made, in the order it made them.** A calibration slope, a
test for a break and the error metrics need no held-out set either.
[Out-of-sample, by construction](../DIAGNOSTICS.md#out-of-sample-by-construction)
gives the consequence for the classical tests. `error_inflation` is the
factor by which estimation error widens a row's prediction error beyond
the noise. For least squares with no decay and no ridge, each residual
divided by its `error_inflation` is Brown, Durbin and Evans' recursive
residual, so their tests apply without fitting the model again.

## What predicting before learning cannot guarantee

The guarantee is about the order in which the bank reads a row as it
learns. It cannot see what the columns hold before they reach the bank,
or which rows you ask it to serve.

**A feature computed from rows after its own carries what happened after
the row into its prediction.** A column made with a
[window operator](../GLOSSARY.md#fitting-a-model) that looks ahead
(`reverse=True`), or a value filled backward from a later row, holds
information the model could not have had at the row. The bank reads it as
any other feature, and the residuals then look better than the model
would do on new rows. Make every feature from the row itself and the rows
before it. A target column that looks ahead needs an `embargo`, as [A
target that looks ahead](#a-target-that-looks-ahead) explains.

**`predict` on rows the bank has already learned is in-sample.** `predict`,
which serves rows from the state without learning them, reads the state
as it stands. On rows that state has learned, such as `bank.predict(df)`
after `bank.fit_predict(df)`, each prediction has seen its own target.
Serve only rows the bank has not learned.

**The caller gives the stream its order.** A model learns the rows in the
order the bank reads them, so a stream out of clock order gives different
predictions and can let the model learn a later row before it predicts an
earlier one. The bank refuses a
[chunk](../GLOSSARY.md#the-model-bank-and-what-it-reads) in which a
group's clock [steps back](../GLOSSARY.md#time) within a
[session](../GLOSSARY.md#time), unless `restart_after_step_back` is set
below the step, where the model [resets](../GLOSSARY.md#time) instead. It
also warns of a query step whose order Polars does not promise ([Put rows
in clock order](../how-to/put-rows-in-clock-order.md)).

## A target that looks ahead

A target can be computed from the rows after its row, such as the mean
return over the next ten minutes. Its value is not known when the row
arrives, so the bank cannot learn the row there. It predicts the row where
it sits, as it predicts every row, and holds the row back from learning
until the target is known.

**The bank holds a row with a formula target until its window has closed
and the spec's `embargo`, a delay on the clock from the row, has passed.**
A [formula target](../GLOSSARY.md#fitting-a-model) is a target given in
the spec's `targets` as an expression (a column computation, such as
`pl.col("x") * 2`) that holds a window operator looking ahead
(`reverse=True`). Once the row is released, the bank computes the target
and learns the row.

**`fit_predict` requires an `embargo` of at least the longest
`window_size` (an operator's span on the clock) among the look-ahead
window operators of the spec's formula targets, and refuses a shorter one
before it reads a row.** A row's target is known only one window after
the row, so an `embargo` shorter than the window would state a delay at
which the target is not yet known. A run that keeps no prediction
(`ModelBank.fit`, or the command line's `--no-output`) takes any
`embargo`, and learns each row once its window
has closed and any `embargo` has passed.

**`resid_<target>` is null on every row for a formula target.** A row's
target is not known when the row is predicted, so there is no residual to
write there. The diagnostics take each row in when it is learned, from the
prediction it was given, so they stay out-of-sample.

**The same target can instead be a column, made in the query before the
bank with `po.stream.with_windows`, and it then needs an `embargo` of at
least its window's length.** A column holds the target's value at the
row, so the bank writes a residual there. Without an `embargo`, the model
learns each row's value at the row, before the rows that value was
computed from have arrived. Its predictions of the rows inside that window
then draw on target values from after their own rows. [Relative and
look-ahead targets](../../README.md#relative-and-look-ahead-targets) shows
both forms side by side.
