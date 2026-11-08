<!-- kind: tutorial -->
# Fit, save and serve a model bank

This tutorial is for anyone new to polars-online who wants to fit a model
once, and then use it from another program to predict new rows. You fit a
regression over a file of past rows and save what it learned to a file.
Then, in a new Python process, you load that file, predict rows whose
target is not yet known, and compare the predictions with the answer you
planted.

To run the code, install the library and numpy with
`pip install "polars-online[numpy]"`, and run it in a directory where it
may write files. Run steps 1 and 2 in one Python process, and steps 3 and
4 in a second one, which step 3 starts.

A [spec](../GLOSSARY.md#the-model-bank-and-what-it-reads) describes one
model: which model it is and which columns it reads. A [model
bank](../GLOSSARY.md#the-model-bank-and-what-it-reads) fits a model for
each spec in one pass over the rows. Its
[state](../GLOSSARY.md#saving-and-serving) is everything it has learned,
which it can save to a file and load back. To
[serve](../GLOSSARY.md#saving-and-serving) is to predict rows from a state
without learning from them.

## 1. Make up past rows and new rows

Make up 10,000 past rows whose target `y` is `0.5 + 1.5·x0 − 2·x1` plus
noise with standard deviation 0.1. Make up five new rows with the features
only, since their target is not known yet. Save each to a parquet file:

```python
import numpy as np
import polars as pl

import polars_online as po

rng = np.random.default_rng(3)
# The past rows: the planted answer is an intercept of 0.5, 1.5 on x0 and -2 on x1.
x0, x1, noise = rng.standard_normal((3, 10_000))
past = pl.DataFrame({"x0": x0, "x1": x1, "y": 0.5 + 1.5 * x0 - 2.0 * x1 + 0.1 * noise})
past.write_parquet("past.parquet")
# Five new rows, with no target.
x0, x1 = rng.standard_normal((2, 5))
pl.DataFrame({"x0": x0, "x1": x1}).write_parquet("new_rows.parquet")
print(f"{past.height} past rows; new rows' columns: {pl.read_parquet('new_rows.parquet').columns}")
```

```text
10000 past rows; new rows' columns: ['x0', 'x1']
```

## 2. Fit the past rows and save the state

Fit a ridge regression over the past rows in a
[query](../GLOSSARY.md#the-model-bank-and-what-it-reads), a chain of
Polars steps that runs only when you ask for its result. With every row
weighted the same, the saved state sums up all of the past rows. Besides
saving the state, the query writes each past row's prediction to
`past_fitted.parquet`, which the steps after it do not need:

```python
spec = po.spec.ewridge(
    "ridge",                                     # the spec's name
    targets=["y"], features=["x0", "x1"],
    half_life=float("inf"),                      # every row counts the same
)
(
    pl.scan_parquet("past.parquet")              # a query over the past rows: nothing is read yet
    .online.fit_predict([spec], save_state="ridge.state")   # the bank, which saves its state after the last row
    .sink_parquet("past_fitted.parquet")         # runs the query, writing the predictions to a file
)
print(f"the state is saved: {po.ModelBank.load('ridge.state')}")
```

```text
the state is saved: ModelBank(['ridge'], groups=1, rows_fed=10000)
```

The saved bank holds one model, since the spec names no
[group](../GLOSSARY.md#the-model-bank-and-what-it-reads) column
(`group=`), and has learned 10,000 rows. The state file holds the spec and
what the model learned from those rows, and nothing of the rows
themselves.

## 3. Load the state in a new Python process

Close Python and start it again in the same directory. Nothing from the
first process is left but the files. Load the state, and read the
coefficients it holds:

```python
import polars as pl

import polars_online as po

bank = po.ModelBank.load("ridge.state")         # the bank, with everything it learned
for term, coef in bank.coef().select("term", "coef").iter_rows():
    print(f"{term:>9} {coef:6.3f}")
```

```text
intercept  0.501
       x0  1.499
       x1 -2.000
```

The coefficients match the planted 0.5, 1.5 and −2 to the second decimal
place, read from the file alone.

## 4. Predict the new rows

Serve the new rows from the state with the query's `.online.predict`. The
bank adds one [output column](../GLOSSARY.md#what-a-bank-writes) named
after the spec, `ridge`, whose value in each row is a record of named
[fields](../GLOSSARY.md#what-a-bank-writes). `pred_y` is the prediction of
`y`, and `weight_sum` is the weight of the rows the prediction was made
from. Compare each prediction with the planted `0.5 + 1.5·x0 − 2·x1`:

```python
served = (
    pl.scan_parquet("new_rows.parquet")          # the new rows, with no target
    .online.predict("ridge.state")               # serves the new rows from the state
    .online.unnest("ridge.state")                # the column "ridge" as one column per field: the state names the spec
    .with_columns(planted=0.5 + 1.5 * pl.col("x0") - 2.0 * pl.col("x1"))   # the answer, to compare
    .collect()
)
for x0, x1, pred, planted, weight in served.select("x0", "x1", "pred_y", "planted", "weight_sum").iter_rows():
    print(f"x0 {x0:5.2f}   x1 {x1:5.2f}   pred_y {pred:6.3f}   planted {planted:6.3f}   weight_sum {weight:.0f}")
```

```text
x0  2.33   x1  1.69   pred_y  0.605   planted  0.606   weight_sum 10000
x0  0.92   x1  1.59   pred_y -1.303   planted -1.303   weight_sum 10000
x0 -0.13   x1  0.05   pred_y  0.204   planted  0.202   weight_sum 10000
x0 -0.91   x1  0.43   pred_y -1.722   planted -1.724   weight_sum 10000
x0  2.84   x1  2.85   pred_y -0.946   planted -0.943   weight_sum 10000
```

Each prediction is within a few thousandths of the planted value.
`weight_sum` is 10,000 on every row: each row is predicted from the same
state, the state saved after the 10,000 past rows. Serving learns nothing,
so that state never changes.

## Where to go next

To use a saved state in your own work, start from the task that matches
it:

| to | read |
|---|---|
| serve from the command line, with no Python, or serve a frame you hold in memory | [Serve from a saved state](../how-to/serve-from-a-saved-state.md) |
| go on learning from new rows once their targets are known | [Resume from a saved state](../how-to/resume-from-a-saved-state.md) |
| save and load the state in every way a bank runs | [Save and load](../../README.md#save-and-load) |
| know what a state holds, and why a resumed run matches one that never stopped | [A bank's state and resuming it](../explanation/state-and-resuming.md) |
| know why saving a fit that follows the recent rows is of limited use | [Local fits](../explanation/time-and-decay.md#local-fits) |
| fit a model over parquet files too large to hold | [Fit a model bank over parquet files in a query](stream-from-parquet.md) |
