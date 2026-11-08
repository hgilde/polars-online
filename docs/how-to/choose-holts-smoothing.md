<!-- kind: how-to -->
# Choose Holt's smoothing

This how-to is for anyone who fits `holt`, the baseline that extrapolates
a target's own level and trend, and needs to set how fast each forgets. It
sets the level's memory with `half_life`, the trend's with
`trend_half_life`, and turns the trend off with `trend=False` for a flat
prediction.

## Prerequisites

These steps need the library and a stream with a clock:

- polars-online installed, with numpy for the recipe, which builds its own
  data and saves it to a parquet file.
- A [clock](../GLOSSARY.md#time) column, and the `gap_cap` a clock
  requires. The recipe's clock `t` steps by 1 a row.
- The parameters of
  [`po.spec.holt`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.holt),
  which its docstring defines. Its entry in
  [MODELS.md](../MODELS.md#holt--holts-linear-trend) gives its update
  equation.

## Set how fast the level and the trend forget

`holt` keeps a level and a trend for each target, and predicts the level
plus the trend times the clock since the target was last seen. Each
forgets at a [half-life](../GLOSSARY.md#time) of its own.

**The level forgets at the spec's `half_life`, as every model's decay
does.** `level_half_life` is refused, with a message naming `half_life`. To
make the trend the whole history's drift, give `trend_half_life=inf`: the
slope then takes the line fit's gain, and is the weighted least-squares
slope exactly when the level does not decay either (`half_life=inf`).
Beside a finite level half-life the slope learns from residuals against a
level that forgets, and lands near the line fit's slope: 0.2% to 1% of it
off at a level half-life of 5 over 3,000 rows of a line in noise.

1. **Give the level's memory as `half_life`,** in clock units. A short one
   follows a level that wanders, and a long one averages out more noise.
2. **Give the trend's memory as `trend_half_life`.** By default it is four
   times the level's half-life, so the trend forgets four times more
   slowly than the level.
3. **For the whole history's drift, give `trend_half_life=float("inf")`.**
   With `half_life=float("inf")` as well, the trend is the weighted
   least-squares slope of the target on the clock.

## Make the prediction flat

**For a flat prediction, give `trend=False`:** simple exponential
smoothing, whose level is the target's exponentially weighted mean. It
holds the trend at zero, and refuses `trend_half_life`.

1. **Give the spec `trend=False` and a `half_life`.** The level is then
   the weighted mean of the targets seen, each weighed by its weight times
   `0.5 ** (age / half_life)`.
2. **Leave `trend_half_life` out.** The spec refuses it beside
   `trend=False`.

The recipe plants a line of slope 0.05 in noise, over 3,000 rows. It fits
the line fit (both half-lives `inf`), a trend that is the whole history's
drift beside a level half-life of 5, and a flat prediction at a half-life
of 20. It prints each beside an independent computation of the same
quantity: numpy's least-squares line, and Polars' own exponentially
weighted mean. This code builds its own data:

```python
import numpy as np
import polars as pl
import polars_online as po

# A line of slope 0.05 in noise, one row a clock unit.
rng = np.random.default_rng(2)
n = 3_000
t = np.arange(n, dtype=float)
y = 10.0 + 0.05 * t + rng.standard_normal(n)
pl.DataFrame({"t": t, "y": y}).write_parquet("line.parquet")

# The line fit, the whole history's drift beside a short level half-life, and a flat prediction.
common = dict(targets=["y"], clock="t", gap_cap=10.0)
inf = float("inf")
specs = [
    po.spec.holt("line_fit", half_life=inf, trend_half_life=inf, **common),
    po.spec.holt("drift", half_life=5.0, trend_half_life=inf, **common),
    po.spec.holt("flat", half_life=20.0, trend=False, **common),
]
bank = po.ModelBank(specs)
bank.fit(pl.scan_parquet("line.parquet"))
fitted = {(s, term): c for s, term, c in bank.coef().select("spec", "term", "coef").iter_rows()}

# Each fit beside an independent computation of the same quantity.
slope, _ = np.polyfit(t, y, 1)
ewm = pl.Series(y).ewm_mean(half_life=20.0, adjust=True)[-1]
print(f"line_fit trend {fitted['line_fit', 'trend']:.6f}, numpy's least-squares slope {slope:.6f} (planted 0.05)")
print(f"drift trend {fitted['drift', 'trend']:.6f}: {abs(fitted['drift', 'trend'] / slope - 1):.1%} off the line fit's")
print(f"flat level {fitted['flat', 'level']:.4f}, Polars' ewm_mean {ewm:.4f}, trend {fitted['flat', 'trend']}")
```

```text
line_fit trend 0.050031, numpy's least-squares slope 0.050031 (planted 0.05)
drift trend 0.050395: 0.7% off the line fit's
flat level 158.6659, Polars' ewm_mean 158.6659, trend 0.0
```

With both half-lives `inf`, the trend is numpy's least-squares slope. A
level half-life of 5 moves it 0.7% off that slope. Under `trend=False`, the
level is Polars' `ewm_mean` with `adjust=True`, and the trend is 0.

## Choose the half-lives from the stream's own errors

**Run one spec per trend half-life, each with a list of level
half-lives.** `half_life` takes a list, a [grid](../GLOSSARY.md#grids)
fitted in one pass, and `trend_half_life` takes one value. So
`po.eval.grid_search` with `param_grid={"trend_half_life": [...]}` scores
every pair on the rows each had not yet seen. [Holt's
smoothing](../CALIBRATION.md#holts-smoothing) runs it on a trend that
wanders, and gives each level half-life as the textbook's smoothing
constant `α`.

## Check the result

**Read the level and the trend with `bank.coef()`.** Its `term` column
names them `level` and `trend`, as the recipe reads them. On a target
whose drift you know, the trend under `trend_half_life=inf` should stand
near it.

**Compare `holt` with the regression it is a baseline for.** Run both in
one bank with `emit_sigma=True`, and compare their `sigma` fields
(`sigma_y` for a target `y`), the spread of each model's error.
[`holt`'s entry](../MODELS.md#holt--holts-linear-trend) in MODELS.md runs
that comparison, with a test of which predicts closer.
