<!-- kind: how-to -->
# Choose settings by hand

This how-to is for anyone who has the output of a [model
bank](../GLOSSARY.md#the-model-bank-and-what-it-reads) for a grid and
wants to choose one of its grid points with `po.eval.metrics`, as
`po.eval.grid_search` does with `n_splits=1`.

## Prerequisites

These steps need the library and a spec with a grid:

- polars-online and numpy installed. The recipe generates its own data.
- A [spec](../GLOSSARY.md#the-model-bank-and-what-it-reads) with a
  [grid](../GLOSSARY.md#grids), such as several values of `half_life`.
  [Choose a model's settings from the stream's own
  errors](../CALIBRATION.md) gives the same choice in one call, and why
  the choice is scored on rows it had not seen.

## Choose on one half, and report on the other

Use this when you already hold the output frame, or want to score it in a
way `po.eval.grid_search` does not. `po.eval.metrics` scores an output
frame. The choice by hand holds the whole output in memory, where
`po.eval.grid_search` keeps a few sums per slot and span.

1. **Run the query and keep its result in memory,** with
   `out = lf.online.fit_predict([spec]).collect()`. The output holds one
   prediction per row and [slot](../GLOSSARY.md#grids).
2. **Split the output into halves with
   `out.head(out.height - out.height // 2)` and
   `out.tail(out.height // 2)`,** the rows `po.eval.grid_search` chooses
   on and tests on with `n_splits=1`.
3. **Score each half with `po.eval.metrics(half, spec)`.** It returns a
   row per slot, with its `mse`.
4. **Pick the slot with the lowest error on the first half.**
5. **Report the pick's error on the second half,** the rows the pick had
   not seen.

This code runs the five steps on a half-life grid, and then
`po.eval.grid_search` with one test span:

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
grid = po.spec.ewridge(
    "grid", targets=["y"], features=["x0", "x1"], clock="t", gap_cap=5.0,
    half_life=[25.0, 100.0, 400.0, 1_600.0, 6_400.0],
)
lf = pl.scan_parquet("wander.parquet")

# 1. Run the query and keep its result.
out = lf.online.fit_predict([grid]).collect()
# 2 and 3. Split the output into halves, and score each.
first = po.eval.metrics(out.head(out.height - out.height // 2), grid)
second = po.eval.metrics(out.tail(out.height // 2), grid)
# 4. Pick on the first half.
pick = first.sort("mse")["slot"][0]
# 5. Report the pick's error on the second half.
print(f"by hand: {pick}, mse {second.filter(slot=pick)['mse'].item():.4f} on the second half")

# The same with grid_search: one span to test, so the choice is read from the first half.
table = po.eval.grid_search(lf, grid, n_splits=1)
chosen = table.filter(pl.col("folds_chosen") == 1)["slot"].item()
print(f"grid_search: {chosen}, mse {table.filter(kind='choice')['mse'].item():.4f} on the second half")
```

```text
by hand: pred_y@h100, mse 1.0287 on the second half
grid_search: pred_y@h100, mse 1.0287 on the second half
```

## Check that it worked

The choice by hand worked when it picks the slot `po.eval.grid_search`
picks with `n_splits=1`, at the same error, as in the recipe:

| what you see | what it means | what to do |
|---|---|---|
| the same slot, at a different error | one slot warms up later than another: `po.eval.metrics` scores each over its own rows, and `po.eval.grid_search` over the rows they share | before `po.eval.metrics`, keep only the rows where every slot's prediction is not null, or use `po.eval.grid_search` |
| a different slot | the slots' errors nearly tie, or they were scored on different rows | read both slots' errors, and report the error of the one you run on the half it was not chosen on |
