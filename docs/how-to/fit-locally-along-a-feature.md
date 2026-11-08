<!-- kind: how-to -->
# Fit locally along a feature

This how-to is for anyone who wants a fit that describes the rows near
each value of a feature, such as a curve in `x` that one straight line
through every row cannot follow.

A model that keeps running sums, such as `ewridge`, fits locally along a
feature when the rows are sorted by that feature and the spec names it as
its [clock](../GLOSSARY.md#time): the [half-life](../GLOSSARY.md#time) is
then a bandwidth in the feature's units. [Time and
decay](../explanation/time-and-decay.md#a-feature-as-the-clock) explains the decay along a clock,
which here becomes a kernel in the feature.

## Prerequisites

These steps need the library, numpy, and a stream with the feature to fit
along:

- polars-online installed, with numpy, which the example uses to make its
  data and to fit the straight line it compares against.
- A numeric column to fit along. The rows must fit in memory to be sorted
  by it: a sort holds every row while it sorts.

## Fit a model locally along a feature

Put the rows in the feature's order in the query, and give the spec that
feature as its clock.

1. **Sort the rows by the feature with Polars' `sort`, in the query before
   the bank, with `maintain_order=True`,** as
   `pl.scan_parquet("curve.parquet").sort("x", maintain_order=True)`. Rows
   with equal values keep their order only under `maintain_order=True`, and
   their order changes the predictions, since the second is predicted after
   the first is learned. A feature has ties more often than a time does.
2. **Name the feature as the spec's `clock`,** as `clock="x"`.
3. **Set `half_life` to the bandwidth, in the feature's own units,** as
   `half_life=0.25`, and `gap_cap` to the widest gap a step should decay
   by, as `gap_cap=1.0`. Each row is then fitted on the rows before it,
   each weighted by `0.5 ** (Δx / half_life)`, with `Δx` its distance in
   the feature, and a step wider than `gap_cap` decays as if it were
   `gap_cap`.
4. **Set `max_rows_between_solves=1` and `coef_every=0`,** so the model
   solves at every row and writes the line through each row's neighborhood
   on every row.
5. **Run the query,** and read each row's `pred_y`: the local fit's value
   at that row's `x`.

**The kernel is one-sided, so the fit lags.** Narrow the bandwidth to lag
less, and widen it to smooth the noise. This code makes 1,200 points of
`sin(x)` plus noise of standard deviation 0.1, for x between −3 and 3, and
fits a straight line in `x` locally at two bandwidths. It prints each fit's
root-mean-square distance from the curve, beside the best straight line's:

```python
import numpy as np
import polars as pl
import polars_online as po

# 1,200 noisy points on sin(x), saved.
rng = np.random.default_rng(1)
x = rng.uniform(-3.0, 3.0, 1200)
pl.DataFrame({"x": x, "y": np.sin(x) + 0.1 * rng.standard_normal(1200)}).write_parquet("curve.parquet")

# A straight line fitted locally at a given bandwidth, over the points sorted by x.
def local(bandwidth):
    kernel = po.spec.ewridge(
        "kernel", targets=["y"], features=["x"],
        clock="x",                  # 2. the clock is a feature: the decay is a kernel in it
        half_life=bandwidth,        # 3. the bandwidth, in x's own units
        gap_cap=1.0,                # a wider gap decays as if it were this wide
        max_rows_between_solves=1,  # 4. a solve at every row
        coef_every=0,               # write the coefficients on every row: the line through the row's neighborhood
        min_weight=20.0,
    )
    query = pl.scan_parquet("curve.parquet").sort("x", maintain_order=True)   # 1. sorted by the feature, in the query before the bank
    return query.online.fit_predict([kernel]).online.unnest([kernel]).collect()   # 5. pred_y: the curve at each row's x

# Two bandwidths, and the best straight line through every point.
narrow, wide = local(0.25), local(1.0)
fit = narrow["pred_y"].is_not_null()                    # the rows past the warm-up
curve = narrow.filter(fit)
truth = np.sin(curve["x"].to_numpy())
line = np.polyval(np.polyfit(narrow["x"], narrow["y"], 1), curve["x"].to_numpy())

# Each fit's distance from the true curve.
def rms(pred):
    return float(np.sqrt(np.mean((np.asarray(pred) - truth) ** 2)))

print(f"bandwidth 0.25: {rms(curve['pred_y']):.2f}")
print(f"bandwidth 1.0:  {rms(wide.filter(fit)['pred_y']):.2f}")
print(f"straight line:  {rms(line):.2f}")
```

```text
bandwidth 0.25: 0.08
bandwidth 1.0:  0.29
straight line:  0.39
```

## Check the result

**Compare the local fit's distance from the curve with the best straight
line's.** In the output of [Fit a model locally along a
feature](#fit-a-model-locally-along-a-feature), the narrow kernel, at
0.08, follows the curve the straight line, at 0.39, cannot. The wide one,
at 0.29, is most of the way back to the line.

## Fit other columns along the feature

**The clock column can stay out of `features`.** With other columns as
features, the fit is a regression whose coefficients move along the clock.

1. **List the other columns in `features`, and keep the feature as the
   `clock`,** as `features=["z"], clock="x"`, with the rows still sorted by
   `x` as in step 1 of [Fit a model locally along a
   feature](#fit-a-model-locally-along-a-feature).
