<!-- kind: how-to -->
# Choose among a grid's settings as the stream runs

This how-to is for anyone who fits a spec over a grid of settings, such as
several half-lives, and wants one prediction from it on every row. It lets
the spec choose the best [grid point](../GLOSSARY.md#grids) so far, or
average every grid point's prediction by its error, as the stream runs.

## Prerequisites

These steps need the library and a grid:

- polars-online installed, and numpy to make the example's data.
- A [spec](../GLOSSARY.md#the-model-bank-and-what-it-reads) with more than
  one [slot](../GLOSSARY.md#grids) per target: for `ewridge`, several
  values in `half_life` or `ridge`, or several named `feature_sets`. A
  spec with one slot per target refuses `emit_selected` and
  `emit_averaged`, since one slot
  leaves nothing to choose or average. The refusal names the grids its
  model takes.

## Choose or average the grid's predictions

Both switches rank the slots by their exponentially weighted
[out-of-sample](../explanation/out-of-sample.md) squared error.
`emit_selected` commits to the slot whose error is lowest so far.
`emit_averaged` hedges: it weights every slot's prediction by the ratio
of its error to the best slot's, and `average_eta` (default 1) sets how
sharply. [Choosing among a grid's
settings](../DIAGNOSTICS.md#choosing-among-a-grids-settings) gives the
weights' formula.

1. **Turn on `emit_selected=True`, `emit_averaged=True`, or both, in the
   spec.** `emit_selected` adds `selected_<t>`, the slot chosen on the
   row, and `pred_<t>__selected`, its prediction. `emit_averaged` adds
   `pred_<t>__averaged`.
2. **Run the query.** Each choice and each average is made from the rows
   before its row, so it can serve the row.
3. **Score every slot with `po.eval.metrics`.** The chosen and averaged
   predictions are slots of their own, scored beside the grid's.
4. **Count how often each slot was chosen,** from `selected_<t>`.

This code makes a stream whose coefficient on `x0` wanders as a slow
random walk, fits it at five half-lives, and scores the choice and two
averages beside each half-life:

```python
import numpy as np
import polars as pl

import polars_online as po

# y = b x0 + 0.5 x1 + noise, where b wanders as a slow random walk.
rng = np.random.default_rng(9)
n = 40_000
x0, x1, e = rng.standard_normal((3, n))
b = 1.0 + np.cumsum(rng.normal(0, 0.01, n))
pl.DataFrame({"t": np.arange(n, dtype=float), "x0": x0, "x1": x1, "y": b * x0 + 0.5 * x1 + e}).write_parquet("wander.parquet")
wander = pl.scan_parquet("wander.parquet")

# 1. Five half-lives, chosen among and averaged as the stream goes; and a sharper average.
common = dict(
    targets=["y"], features=["x0", "x1"], clock="t", gap_cap=5.0,
    half_life=[25.0, 100.0, 400.0, 1_600.0, 6_400.0],  # one fit per half-life, from the same rows
)
halves = po.spec.ewridge(
    "halves", **common,
    emit_selected=True,  # selected_y, pred_y__selected: the best half-life so far, row by row
    emit_averaged=True,  # pred_y__averaged: every half-life's prediction, weighted by its error
)
sharp = po.spec.ewridge("sharp", **common, emit_averaged=True, average_eta=10.0)  # leans harder on the best
# 2. Run the query.
out = wander.online.fit_predict([halves, sharp]).collect()

# 3. Every slot's out-of-sample error over the whole stream, best first.
scores = pl.concat([
    po.eval.metrics(out, "halves"),
    po.eval.metrics(out, "sharp").filter(slot="pred_y__averaged").with_columns(slot=pl.lit("sharp: pred_y__averaged")),
])
for slot, mse, r2 in scores.sort("mse").select("slot", "mse", "r2").iter_rows():
    print(f"{slot:24} mse {mse:.4f}, R² {r2:.4f}")
# 4. The share of rows on which each half-life was the one chosen.
held = out.select(pl.col("halves").struct.field("selected_y")).to_series().value_counts(normalize=True)
for slot, share in held.sort("proportion", descending=True).head(3).iter_rows():
    print(f"selected {slot}: {share:.0%} of the rows")
```

```text
pred_y@h100              mse 1.0249, R² 0.9064
sharp: pred_y__averaged  mse 1.0360, R² 0.9055
pred_y@h400              mse 1.0405, R² 0.9050
pred_y__selected         mse 1.0439, R² 0.9047
pred_y@h25               mse 1.0523, R² 0.9039
pred_y__averaged         mse 1.0570, R² 0.9035
pred_y@h1600             mse 1.1654, R² 0.8936
pred_y@h6400             mse 1.7095, R² 0.8439
selected @h25: 41% of the rows
selected @h400: 30% of the rows
selected @h100: 27% of the rows
```

Over the whole stream, a half-life of 100 rows predicts best, with 400 and
25 close behind and 6,400 far worse: too long a memory averages over a
coefficient that has moved. The choice made as the stream ran held 25 on
41% of the rows, 400 on 30% and 100 on 27%, and its error sits between the
best half-life's and the third's. Each slot is ranked at its own
half-life, so the shortest is ranked on the fewest rows and wins its share
partly by luck. The average at the default `average_eta` of 1 still gives
the long half-lives much of their weight, and trails the choice. At an
`average_eta` of 10 it leans on the best slots, and beats the choice.

## Decide between a choice as the stream runs and one fixed setting

Choose by whether the best setting itself moves:

| what you need | what to use |
|---|---|
| a best setting that moves over the stream | `emit_selected`, or `emit_averaged` to hedge between slots that are close |
| one setting to run on the rows to come | `po.eval.grid_search`, which chooses over a long stretch and scores the choice on rows it had not seen ([Choose a model's settings from the stream's own errors](../CALIBRATION.md)) |
| a choice made by hand from an output frame you already have | [Choose settings by hand](choose-settings-by-hand.md) |

The best error over the rows that chose it is optimistic, so report a
choice's error on rows after the ones it was chosen on ([Why the best
error over a grid is
optimistic](../explanation/grids-and-instances.md#why-the-best-error-over-a-grid-is-optimistic)).

## Check that it worked

Read the chosen and averaged slots' errors against the grid's own, as
step 3 does:

| what you see | what it means | what to do |
|---|---|---|
| `pred_<t>__selected` or `pred_<t>__averaged` near the best slot's error | the choice follows the best slot | serve it |
| the choice worse than several fixed slots, and `selected_<t>` held mostly by the shortest half-life | the short half-life is ranked on few rows and wins by luck | use `emit_averaged` with a larger `average_eta`, drop the shortest half-life from the grid, or fix one setting with `po.eval.grid_search` |
| the average worse than the choice | `average_eta` is too small, so slots far from the best keep much of their weight | increase `average_eta`. At `inf` the average is the choice |
| `pred_<t>__averaged` null on a row | no slot had a prediction and an error on that row, as in the warm-up | read the rows after the warm-up |
