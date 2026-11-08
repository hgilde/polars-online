<!-- kind: how-to -->
# Check the scale of a model's predictions

This how-to is for anyone who fits a linear model on a stream and wants to
know whether its predictions are the right size. It turns on the model's
calibration, reads it, and rescales or refits a model whose predictions are
too small or too large.

## Prerequisites

These steps need the library and a linear model:

- polars-online installed, and numpy to make the example's data.
- A [spec](../GLOSSARY.md#the-model-bank-and-what-it-reads) of one of the
  ten linear models, such as `ewridge`, `rls` or `kalman`, which are the
  models that take `emit_calibration`. [What each costs, and which models
  take it](../DIAGNOSTICS.md#what-each-costs-and-which-models-take-it)
  lists them.
- For a [target](../GLOSSARY.md#the-model-bank-and-what-it-reads) that
  looks ahead, the spec's `horizon_rows=` (with a clock) or `embargo=`
  (without one), so the calibration's test allows for residuals that
  overlap. [Can its t be trusted?](../DIAGNOSTICS.md#can-its-t-be-trusted)
  says how the horizon is read.

## Read the calibration

A prediction can follow the target and still be the wrong size, as when a
ridge shrinks every coefficient. The calibration regresses the target on
the prediction as the stream runs, so a slope of 1 and an intercept of 0
mean the predictions are the right size. [Is it scaled
right?](../DIAGNOSTICS.md#is-it-scaled-right) defines its fields and the
threshold of its test.

1. **Turn on `emit_calibration=True` in the spec.** Each row of the
   model's output gains `calibration_slope_<t>`,
   `calibration_intercept_<t>` and `calibration_wald_<t>` for each target
   `<t>`. Each is read before the row, so it is
   [out-of-sample](../explanation/out-of-sample.md).
2. **Run the query, and read the rows from where the calibration
   starts.** A diagnostic starts once the stream has come 95% of the way to its
   steady state, about 4.3 of the fit's half-lives in (not of the
   calibration's own memory). Its fields are null until then, and start at row 8,647 for a half-life of 2,000 rows. The recipe
   reads from row 9,000, so every error it prints is over the same rows.
3. **Read the slope's median, and the share of rows where
   `calibration_wald_<t>` passes 5.99.** A slope above 1 says the
   predictions are too small, and a slope below 1 says they are too large.
   5.99 is the test's 5% value, which a calibrated fit passes on few rows.
4. **Rescale each prediction by the calibration read on its row:**
   `calibration_intercept_<t> + calibration_slope_<t> · pred_<t>`. It is
   still out-of-sample, so it can serve the row.

This code fits one stream three ways: with a ridge as large as the
features' variance, with a half-life too short for 21 coefficients (twenty
features and the intercept), and with neither fault. It prints each fit's
slope, the share of rows its test flags, and its error before and after
rescaling:

```python
import numpy as np
import polars as pl

import polars_online as po

# Twenty features, of which two matter: y = 1.0 x0 + 0.5 x1 + noise.
rng = np.random.default_rng(1)
n, k = 20_000, 20
x = rng.standard_normal((n, k))
y = 1.0 * x[:, 0] + 0.5 * x[:, 1] + rng.standard_normal(n)
features = [f"x{j}" for j in range(k)]
pl.DataFrame({"t": np.arange(n, dtype=float), "y": y} | {f: x[:, j] for j, f in enumerate(features)}).write_parquet("wide.parquet")
wide = pl.scan_parquet("wide.parquet")

# 1. The three fits, each with its calibration: too much ridge, too short a memory, and neither fault.
common = dict(targets=["y"], features=features, clock="t", gap_cap=5.0, emit_calibration=True)
specs = [
    po.spec.ewridge("shrunk", half_life=2_000.0, ridge=1.0, **common),  # a ridge the size of each feature's variance
    po.spec.ewridge("overfit", half_life=30.0, **common),  # 21 coefficients on about 87 rows' worth
    po.spec.ewridge("steady", half_life=2_000.0, **common),
]
# 2. Run the query, from row 9,000: every calibration has started there.
out = wide.online.fit_predict(specs).slice(9_000).collect()

# 3 and 4. Each fit's calibration, and its error before and after rescaling.
print(f"{'spec':8} {'slope':>6} {'wald > 5.99':>12} {'mse':>6} {'mse rescaled':>13}")
for spec in specs:
    rows = out.select("y", pl.col(spec["name"]).struct.unnest())
    # Rescale each prediction by the calibration read before its row: still out-of-sample.
    rescaled = pl.col("calibration_intercept_y") + pl.col("calibration_slope_y") * pl.col("pred_y")
    slope, flagged, mse, mse_rescaled = rows.select(
        slope=pl.col("calibration_slope_y").median(),
        flagged=(pl.col("calibration_wald_y") > 5.99).mean(),
        mse=(pl.col("y") - pl.col("pred_y")).pow(2).mean(),
        mse_rescaled=(pl.col("y") - rescaled).pow(2).mean(),
    ).row(0)
    print(f"{spec['name']:8} {slope:>6.2f} {flagged:>12.0%} {mse:>6.3f} {mse_rescaled:>13.3f}")
```

```text
spec      slope  wald > 5.99    mse  mse rescaled
shrunk     2.00         100%  1.326         1.020
overfit    0.82          99%  1.277         1.239
steady     1.00           0%  1.017         1.017
```

The shrunk fit's median slope is 2.00: its ridge halved every
coefficient, so every prediction is half the size it should be. The test
flags every row, and rescaling takes its error from 1.326 to 1.020, the
steady fit's level. The overfit fit's slope is 0.82, flagged on 99% of
rows: its predictions are about 22% too large. 21 coefficients
estimated on about 87 rows' worth of data add their estimation noise to
every prediction. Rescaling recovers only a little, from 1.277 to 1.239,
since most of that noise varies from row to row, and no single multiplier
removes it. The steady fit's slope is 1.00, flagged on no row, and
rescaling changes nothing.

For a `quantile` fit, or `sgd` under `loss="quantile"`, the calibration
is the share of rows below the prediction, `calibration_coverage_<t>`,
which should read the quantile `q`. [Is it scaled
right?](../DIAGNOSTICS.md#is-it-scaled-right) gives its test.

## Fix a fit that is the wrong size

Choose the fix by what the calibration shows over many half-lives:

| what you see | what it means | what to do |
|---|---|---|
| a slope above 1 | the predictions are too small: a ridge shrank them, or a learning rate is too small | rescale by the calibration, as step 4 does, or lower the `ridge`, or increase a gradient fit's step size, such as `sgd`'s `learning_rate` or `ftrl`'s `alpha` |
| a slope below 1 | the predictions are too large: a fit with too many coefficients for its memory follows its own noise | lengthen the half-life, or fit fewer features. Rescaling recovers little |
| a slope of 1, and a high error | the model is calibrated but missing information, which no rescaling repairs | look for what it misses ([Find what a model misses](find-what-the-model-misses.md)) |

## Check that it worked

For a refitted spec, run steps 1 to 3 again. A fit of the right size
reads a slope near 1, and its test passes 5.99 on few rows: at the default
memory, 0% to 0.3% of the rows of calibrated fits. For a rescaled
prediction, compare its mean squared error with the plain prediction's,
over the same rows, as step 4's code does. Rescaling a fit of the right
size leaves its error where it was, as the steady fit's 1.017 against
1.017. The calibration's memory is four times the fit's by default, and
`calibration_half_life` sets it ([One memory per diagnostic: run once or
windowed](../DIAGNOSTICS.md#one-memory-per-diagnostic-run-once-or-windowed)).
If the slope still sits away from 1 for many of those half-lives, the
fault remains: choose another row of the table in [Fix a fit that is the
wrong size](#fix-a-fit-that-is-the-wrong-size).
