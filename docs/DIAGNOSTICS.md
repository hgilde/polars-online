# Is the model working?

A model fitted on a stream always returns a number. Whether that number
can be trusted is a set of separate questions. Is the prediction scaled
right? Has the relationship broken? Can a coefficient's t be believed? Is
the fit missing a lag or a curvature? Another model may predict better,
and the model may simply not be ready yet. Each question has its own
diagnostic. A diagnostic is a switch in the spec that adds fields to the
model's output on every row, or a call in `po.eval` that scores an output
after the run.

[Detecting data issues](DATA-ISSUES.md) asks whether the data can be
learned from at all. This page asks whether the model learned from it well.

Here is a stream whose relationship changes halfway: the slope on `x1`
flips from 0.5 to −0.5 at row 3,000. A ridge regression with a half-life of
200 rows is fitted with one diagnostic for each of three questions:

```python
import numpy as np
import polars as pl

import polars_online as po

# y = 1.0 x0 + b x1 + noise, where b flips from 0.5 to -0.5 at row 3,000 of 6,000.
rng = np.random.default_rng(0)
n, at = 6_000, 3_000
x0, x1 = rng.standard_normal((2, n))
y = 1.0 * x0 + np.where(np.arange(n) < at, 0.5, -0.5) * x1 + rng.standard_normal(n)
pl.DataFrame({"t": np.arange(n, dtype=float), "x0": x0, "x1": x1, "y": y}).write_parquet("flip.parquet")
lf = pl.scan_parquet("flip.parquet")

model = po.spec.ewridge(
    "ridge", targets=["y"], features=["x0", "x1"], clock="t", gap_cap=5.0, half_life=200.0,
    emit_metrics=True,  # how good is it: ic_y, r2_y, hit_rate_y
    emit_calibration=True,  # is it scaled right: calibration_slope_y, _intercept_y, _wald_y
    emit_breaks=True,  # has it broken: studentized_y, cusum_y, cusum_sq_y, break_wald_y
)
out = lf.online.fit_predict([model]).online.unnest([model]).collect()

# Each diagnostic's share of rows past its threshold before and after the flip, and its first row after.
before, after = pl.col("t").is_between(1_000, at - 1), pl.col("t") >= at
flags = {
    "calibration_wald > 5.99": pl.col("calibration_wald_y") > 5.99,
    "|cusum| > 3": pl.col("cusum_y").abs() > 3,
    "|cusum_sq| > 3": pl.col("cusum_sq_y").abs() > 3,
    "break_wald > 21.1": pl.col("break_wald_y") > 21.1,
}
print(f"{'flag':24} {'rows 1,000-2,999':>17} {'rows 3,000-5,999':>17} {'first row after':>16}")
for name, flag in flags.items():
    share_before = out.filter(before).select(flag.mean()).item()
    share_after = out.filter(after).select(flag.mean()).item()
    first = out.filter(after & flag)["t"].min()
    print(f"{name:24} {share_before:>17.1%} {share_after:>17.1%} {'-' if first is None else f'{first:.0f}':>16}")
r2 = out.filter(pl.col("t").is_in([2_999.0, 3_100.0, 3_300.0, 5_999.0]))["r2_y"]
print("r2_y at rows 2,999, 3,100, 3,300 and 5,999: " + ", ".join(f"{v:.2f}" for v in r2))
```

```text
flag                      rows 1,000-2,999  rows 3,000-5,999  first row after
calibration_wald > 5.99               0.0%              0.0%                -
|cusum| > 3                           0.0%              0.0%                -
|cusum_sq| > 3                        0.0%              0.0%                -
break_wald > 21.1                     0.0%             77.0%             3051
r2_y at rows 2,999, 3,100, 3,300 and 5,999: 0.51, 0.43, 0.44, 0.55
```

Only `break_wald` sees the flip, 51 rows after it, and it stays past its
threshold on 77% of the rows after. It is the Wald distance between two
fits of the same regression, one at the diagnostic's memory and one at four
times it, and it grows when a coefficient moves. The two CUSUMs watch the
residuals' mean and spread. A slope that flips on a feature centred at zero
leaves the residuals' mean at zero, so the CUSUM cannot see it, and the
spread grows too little here for the CUSUM of squares. The calibration
stays quiet because the fit forgets: within a few half-lives it has learned
the new slope, and its predictions are scaled right again. The running R²
falls from 0.51 before the flip to 0.43 a hundred rows after, and is back
to 0.55 by the end. Each section of this page takes one question, says
which diagnostic answers it and what that diagnostic cannot see, and gives
a recipe that runs.

These are the words of the library that the page uses:

| word | what it means here |
|---|---|
| spec | one model's description: the model, the columns it reads, and how it treats time. `po.spec.ewridge(...)` makes one, and [the README's model table](../README.md#models) lists every kind |
| bank | `po.ModelBank`, a set of specs fitted together over the same rows |
| slot | one prediction of one target, at one point of a grid such as a list of ridge values. A diagnostic writes one field per slot, suffixed with the target, as `cusum_y` |
| out of sample | a row's prediction is made before the model learns from the row, so its residual `resid = y − pred` is one the model has not seen |
| `half_life` | each row's weight halves every `half_life` of the clock column, or of the rows when there is no clock. `inf` forgets nothing |
| memory | a diagnostic's own half-life, its `*_half_life` keyword. `inf` runs it once over the whole stream |
| Kish size | `n_kish = (Σw)² / Σw²`: the number of equally weighted rows that a set of decayed weights is worth, `2.885 × half_life` rows in steady state |
| `weight_sum` | the weight a model has accumulated before the row: a weight, which counts rows only when nothing decays |

Each question, the diagnostic that answers it, and its fields:

| question | switch or call | fields, per slot `<t>` | section |
|---|---|---|---|
| is it scaled right? | `emit_calibration` | `calibration_slope_<t>`, `calibration_intercept_<t>`, `calibration_wald_<t>` | [Is it scaled right?](#is-it-scaled-right) |
| has it broken? | `emit_breaks`, `emit_drift` | `studentized_<t>`, `cusum_<t>`, `cusum_sq_<t>`, `break_wald_<t>`, `drift_<t>` | [Has it broken?](#has-it-broken) |
| can its t be trusted? | `emit_se_coef`, `emit_robust_se` | `se_coef`, `se_coef_hc0`, `se_coef_hac`, beside `coef` | [Can its t be trusted?](#can-its-t-be-trusted) |
| what is it missing? | `emit_specification`, `emit_autocorr` | `ljung_box_<t>`, `breusch_pagan_<t>`, `reset_<t>`, `autocorr_<t>` | [What is it missing?](#what-is-it-missing) |
| are its tails heavy? | `emit_tails` | `skew_<t>`, `kurtosis_<t>`, `jarque_bera_<t>` | [Are its tails heavy?](#are-its-tails-heavy) |
| which row moved it? | `emit_influence` | `influence_<t>` | [Which row moved it?](#which-row-moved-it) |
| are its features healthy? | `emit_feature_health` | `spread_ratio_<feature>`, `mean_shift_<feature>` | [Are its features healthy?](#are-its-features-healthy) |
| is another model better? | `emit_metrics`, `emit_selected`, `emit_averaged`; `po.eval` | `ic_<t>`, `r2_<t>`, `hit_rate_<t>`, `selected_<t>`, `pred_<t>__selected`, `pred_<t>__averaged` | [Is another model better?](#is-another-model-better) |
| is it ready? | `min_weight`, `min_settled_frac`, `max_error_inflation`, `emit_error_inflation` | `weight_sum`, `settled_frac`, `withheld_reason`, `error_inflation_<t>` | [Is it ready?](#is-it-ready) |
| how wide is its error? | `emit_sigma`, `emit_zscore`, `resid_quantiles`, `conformal` | `sigma_<t>`, `zscore_<t>`, `abs_resid_q<p>_<t>`, `lo_<t>`, `hi_<t>`, `coverage_<t>` | [How wide is its error?](#how-wide-is-its-error) |

## How the diagnostics work

Four ideas hold for every diagnostic on this page. Each reads a residual
the model has not learned, and keeps a memory of its own. Its statistic is
read on every row, which changes what a threshold means. And each has a
cost a row and a set of models that take it.

### Out of sample, by construction

**Every residual a diagnostic reads was made before the model learned from
its row.** A bank predicts a row, records the residual, and only then
learns the row. So a diagnostic built from those residuals is a
*prequential* one, in Dawid's (1984) term: it judges the forecasts the
model actually made, in the order it made them, on outcomes it had not yet
seen. Nothing has to be held out, and nothing is refitted.

**Run once, that residual is Brown, Durbin and Evans' (1975) recursive
residual, which is why the classical tests apply without a refit.** For
least squares on the rows before `t`, with coefficients `b_{t−1}` and Gram
`X'X` over those rows,

```text
v_t = (y_t − x_t' b_{t−1}) / sqrt(1 + x_t' (X'X)⁻¹ x_t)
```

Under a constant relationship with independent Gaussian noise of variance
`σ²`, the `v_t` are independent `N(0, σ²)`. A residual from the final fit
is not: the fit has seen it, so the residuals are correlated and shrunk
toward zero. The denominator is the model's `error_inflation`, so
`resid / error_inflation` is `v_t`. `ewridge`, `rls` and `kalman` keep the
factor, and on the other models 1 stands in. The CUSUM, the CUSUM of squares,
Ljung-Box, Breusch-Pagan, RESET and Jarque-Bera all read these residuals,
and run once with no ridge each matches statsmodels' test on the same rows
(`tests/test_second_opinion.py`).

### One memory per diagnostic: run once or windowed

**Each diagnostic keeps one accumulator per slot, with a half-life of its
own.** Its keyword is the switch's name with `_half_life`, as
`calibration_half_life`. Left out, it is the model's half-life, except for
the calibration, whose memory is four times the model's. `inf` runs the
diagnostic once over the whole stream.

| form | memory | what it is | use it beside |
|---|---|---|---|
| run once | `inf` | the classical test, with its critical values, at `n` rows | a fit that does not forget (`half_life=inf`) |
| windowed | finite | the same statistic over exponentially weighted moments, read at the Kish size `n_kish` in place of `n` | a fit that forgets |

**A run-once test assumes a relationship that does not change and a fit
that does not forget.** Under those assumptions its statistic has a known
distribution, and a threshold has a known false-alarm rate. Once a fit
forgets, the relationship it estimates is allowed to move, and the question
becomes local: is the fit wrong *now*? A windowed diagnostic answers that.
It computes the same statistic from moments that decay as the fit's do,
and reads it at the Kish size of its weights. For exponential weights at a
half-life of `h` rows, `n_kish = (1 + λ) / (1 − λ)` with `λ = 2^(−1/h)`,
about `2.885 h`. A half-life of 200 rows is worth about 577 independent
rows. Windowed, the specification tests and Jarque-Bera passed their 5%
values on 2% to 7% of the rows of clean streams, as they did run once.

**A fit that forgets absorbs a miscalibration at its own pace, so a
diagnostic at the fit's own memory sees only part of it.** If a fit's
predictions are 30% too large, the fit drifts toward the truth as it
learns. Over one half-life it has moved halfway, so the residuals a
diagnostic at the same memory reads have had half the error taken out of
them. That diagnostic is conservative. Measured on a fit whose calibration
slope was 0.7, the calibration test passed its 5% value on 5% of rows at
the model's memory, on 42% at four times it, and on 89% run once. On
calibrated fits it passed on 0.3% to 0.6% of rows at the model's memory
and 0% to 0.3% at four times. That is why the calibration's memory
defaults to four times the model's, and every other diagnostic's to the
model's own.

**A shorter memory finds a change sooner and misses more small ones.** A
windowed statistic responds to a change within about one memory, so its
delay scales with its half-life. A shorter memory also has a smaller Kish
size, which makes the statistic of the same effect smaller, so a small
effect stays below the threshold. A diagnostic's memory is the place to trade delay for
power. On 200 streams of 3,000 rows whose intercept moved by half a noise
sd at row 1,500, the CUSUM at a memory of 200 rows found every break within
100 rows. Run once over the whole stream, it found each 244 rows after the
break on the median.

**A run-once diagnostic keeps the first predictions for good, so give the
model a `min_weight` of a few rows per coefficient.** A fit with three rows
for three coefficients predicts wildly, and run once those first residuals
weigh in the statistic forever. On 40 clean streams of 6,000 rows, a
run-once `reset` beside a run-once `ewridge` at its default `min_weight`
of 0 passed its 5% value on 26% of rows past row 2,000. At `min_weight=10`
it passed on 2.6%. The calibration test passed on 19% of rows at 0 and 9.4%
at 10. A windowed diagnostic forgets the warm-up and needs no such care.

### Reading a statistic that is read on every row

**Over many streams, a windowed statistic sits past its 5% value on about
5% of rows, but on any one stream that share can be anything from 0% to
20%.** Each row's statistic shares almost all its data with the row
before, so the flags come in long runs. Over 40 clean streams, Ljung-Box at
a memory of 1,000 rows sat past its 5% value on 4.2% of rows. On single
streams the share ran from 0% to 20%. Read a statistic's median over a
stretch, or how long it stays past a threshold, before reading one row.

**Every recipe on this page is a complete program.** It generates its data
with the answer planted, saves it to parquet and reads it back with
`pl.scan_parquet`, as a stream too large to hold would be read. Its output
is shown below it. `tests/test_diagnostics_page.py` runs each recipe and
holds its output to what is shown, so none goes stale. A number past a
million in an output is a model that has blown up. Its digits depend on
the platform's rounding, so the test holds such a number only to being past
a million. Warnings that a recipe raises, such as a `ReadinessWarning`, go
to the standard error stream and are not shown.

### Every switch at once

Every switch adds its fields to the spec's output struct, and one spec can
carry them all. A switch's tuning keywords sit beside it, each at its
default if it has one:

```python
import numpy as np
import polars as pl

import polars_online as po

rng = np.random.default_rng(16)
n = 2_000
x0, x1 = rng.standard_normal((2, n))
pl.DataFrame(
    {"t": np.arange(n, dtype=float), "x0": x0, "x1": x1, "y": x0 + 0.5 * x1 + rng.standard_normal(n)}
).write_parquet("every.parquet")
lf = pl.scan_parquet("every.parquet")

every = po.spec.ewridge(
    "every", targets=["y"], features=["x0", "x1"], clock="t", gap_cap=5.0, half_life=200.0,
    emit_sigma=True,              # sigma_y: EW standard deviation of the out-of-sample residuals
    emit_zscore=True,             # zscore_y: resid / sigma, how surprising the row was
    resid_quantiles=[0.5, 0.9],   # abs_resid_q0.5_y, abs_resid_q0.9_y: EW quantiles of |resid|
    conformal=0.9,                # lo_y, hi_y, coverage_y: an interval at this coverage ...
    conformal_rate=0.05,          #   ... whose radius moves this fast
    emit_metrics=True,            # ic_y, r2_y, hit_rate_y: running accuracy
    emit_autocorr=True,           # autocorr_y: each residual against the one ...
    resid_autocorr_lag=1,         #   ... this many scored rows back
    emit_drift=True,              # drift_y: Page-Hinkley on |resid| ...
    drift_delta=0.5,              #   ... with this tolerance, in units of sigma ...
    drift_threshold=20.0,         #   ... and this threshold, in sigma times clock units
    emit_calibration=True,        # calibration_slope_y, _intercept_y, _wald_y
    calibration_half_life=800.0,  #   the default: four times the model's half-life
    emit_breaks=True,             # studentized_y, cusum_y, cusum_sq_y, break_wald_y
    breaks_half_life=200.0,       #   the default: the model's half-life, as for each below
    emit_specification=True,      # ljung_box_y, breusch_pagan_y, reset_y
    ljung_box_lags=10,            #   the default
    emit_tails=True,              # skew_y, kurtosis_y, jarque_bera_y
    emit_influence=True,          # influence_y: ewridge, rls and kalman
    emit_feature_health=True,     # spread_ratio_x0, mean_shift_x0, and x1's
    emit_error_inflation=True,    # error_inflation_y: ewridge, rls and kalman
    emit_se_coef=True,            # se_coef, beside coef: ewridge, rls and kalman
    emit_robust_se=True,          # se_coef_hc0, se_coef_hac, beside coef: ewridge and rls ...
    robust_se_lags=10,            #   ... with Newey and West's lags to this many rows
    emit_clocks=True,             # scored_clock, learned_clock: every model takes it
)
out = lf.online.fit_predict([every]).online.unnest([every]).collect()
fields = [c for c in out.columns if c not in ("t", "x0", "x1", "y")]
for i in range(0, len(fields), 6):
    print(" ".join(fields[i : i + 6]))
```

```text
pred_y resid_y error_inflation_y sigma_y zscore_y lo_y
hi_y coverage_y ic_y r2_y hit_rate_y abs_resid_q0.5_y
abs_resid_q0.9_y autocorr_y drift_y calibration_slope_y calibration_intercept_y calibration_wald_y
studentized_y cusum_y cusum_sq_y break_wald_y ljung_box_y breusch_pagan_y
reset_y skew_y kurtosis_y jarque_bera_y influence_y spread_ratio_x0
spread_ratio_x1 mean_shift_x0 mean_shift_x1 weight_sum settled_frac withheld_reason
coef_y_intercept coef_y_x0 coef_y_x1 support_coef_y_intercept support_coef_y_x0 support_coef_y_x1
se_coef_y_intercept se_coef_y_x0 se_coef_y_x1 se_coef_hc0 se_coef_hac scored_clock
learned_clock
```

### What each costs, and which models take it

**Most diagnostics add a few nanoseconds a row, and the ones that solve
or keep a matrix add microseconds.** Each was timed alone, added to an
`ewridge` with one slot, over 200,000 rows with 5 features and 100,000 rows
with 20, in one chunk, the best of five runs. The machine was shared, at a
load average near 3.3, and the noise of a measurement is about ±10 ns. The
fit alone took 202 ns a row at 5 features and 416 ns at 20.

| switch | added per row, 5 features | 20 features | grows with | models |
|---|---:|---:|---|---|
| `emit_sigma`, `emit_zscore`, `emit_autocorr`, `conformal`, `emit_drift` | under 10 ns | under 10 ns | nothing | the ten linear models |
| `emit_metrics` | 8 ns | 15 ns | nothing | the ten linear models |
| `resid_quantiles` (two levels) | 14 ns | 24 ns | nothing | the ten linear models |
| `emit_se_coef` | 10 ns | 19 ns | `k²`, on `coef`'s rows only | `ewridge`, `rls`, `kalman` |
| `emit_calibration` | 20 ns | 26 ns | nothing | the ten linear models |
| `emit_feature_health` | 36 ns | 84 ns | `k` | the nine linear models with features |
| `emit_error_inflation` | 125 ns | 334 ns | `k²`: a triangular solve | `ewridge`, `rls`, `kalman` |
| `emit_influence` | 145 ns | 349 ns | `k²`, the same solve | `ewridge`, `rls`, `kalman` |
| `emit_tails` | 157 ns | 355 ns | `k²`, the same solve | the ten linear models |
| `emit_robust_se` (10 lags) | 350 ns | 3,266 ns | `L × k²` for `L` lags | `ewridge`, `rls` |
| `emit_specification` | 719 ns | 1,922 ns | `k²`, and the lags | the ten linear models |
| `emit_breaks` | 903 ns | 4,567 ns | `k²`: two fits and their distance | the ten linear models |

The ten linear models are `ewridge`, `rls`, `lasso`, `kalman`, `huber`,
`quantile`, `sgd`, `pa`, `ftrl` and `holt`. Any other model refuses a
residual switch by name, and a model outside a switch's list refuses that
switch and names the models that take it.

## Is it scaled right?

**A prediction can be correlated with the outcome and still be the wrong
size, and the calibration measures by how much.** Mincer and Zarnowitz
(1969) regress the outcome on the prediction, `y = a + b · pred`. A
prediction that is scaled right has `b = 1` and `a = 0`. A slope above 1
says the predictions are too small: a ridge shrank them, or a learning rate
was too slow. A slope below 1 says they are too large: a fit with too many
coefficients for its memory follows its own noise.

| | `emit_calibration` |
|---|---|
| fields | `calibration_slope_<t>`, `calibration_intercept_<t>`, `calibration_wald_<t>`, read before the row |
| update | exponentially weighted means of `pred`, `y`, `pred²` and `pred · y` at `calibration_half_life`; `b = cov(pred, y) / var(pred)`, `a = ȳ − b · p̄` |
| statistic | `wald = (n_kish − 2) · ((ȳ − p̄)² + (b − 1)² · var(pred)) / s²`, with `s²` the regression's residual mean square: `chi2(2)` when `a = 0` and `b = 1` |
| memory | four times the model's half-life; `inf` runs it once, where `wald / 2` is the least-squares F statistic, `F(2, n − 2)` |
| threshold | `wald > 5.99`, `chi2(2)`'s 5% value. Run once on calibrated fits it passed on 5.0% to 7.0% of 400 streams; at the default memory, on 0% to 0.3% of rows |
| cost, models | 20 ns a row; the ten linear models |

**Rescale a prediction by its calibration: multiply it by the slope and
add the intercept.** The fields on a row are read before the row, so
`calibration_intercept + calibration_slope · pred` is still out of sample,
and it can serve the row.

**The Wald statistic assumes independent residuals, so it overstates on a
target that looks ahead.** Overlapping labels correlate the residuals, as
[Can its t be trusted?](#can-its-t-be-trusted) explains, and the
calibration has no Newey-West form. On the two-stage example's look-ahead
target, a slope of 0.99 came with a `wald` of 72.7
([A worked example](#a-worked-example-the-two-stage-reversion-beta)). Read
the slope there, and not the Wald.

**It cannot see a fit that is wrong but calibrated.** A model missing a
feature still predicts `E[y | its features]`, whose calibration slope is 1.
A low R² with a slope of 1 is a model with too little information, which
no rescaling repairs.

The recipe fits one stream three ways: with a ridge as large as the
features' variance, with a half-life too short for twenty coefficients,
and with neither fault.

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
lf = pl.scan_parquet("wide.parquet")

common = dict(targets=["y"], features=features, clock="t", gap_cap=5.0, emit_calibration=True)
specs = [
    po.spec.ewridge("shrunk", half_life=2_000.0, ridge=1.0, **common),  # a ridge the size of each feature's variance
    po.spec.ewridge("overfit", half_life=30.0, **common),  # 21 coefficients on about 87 rows' worth
    po.spec.ewridge("steady", half_life=2_000.0, **common),
]
out = lf.online.fit_predict(specs).slice(5_000).collect()

print(f"{'spec':8} {'slope':>6} {'wald > 5.99':>12} {'mse':>6} {'mse rescaled':>13}")
for spec in specs:
    rows = out.select("y", pl.col(spec["name"]).struct.unnest())
    # Rescale each prediction by the calibration read before its row: still out of sample.
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
shrunk     1.99         100%  1.329         1.021
overfit    0.82          98%  1.278         1.241
steady     0.99           0%  1.018         1.018
```

The shrunk fit's median slope is 1.99: its ridge halved every
coefficient, so every prediction is half the size it should be. The Wald
test flags every row, and rescaling by the slope takes its error from
1.329 to 1.021, the steady fit's level. The overfit fit's slope is 0.82,
flagged on 98% of rows: its predictions are about 20% too large, because
twenty coefficients estimated on about 87 rows' worth of data add their
estimation noise to every prediction. Rescaling recovers only a little,
from 1.278 to 1.241, since most of that noise varies from row to row, and
no single multiplier removes it. Its fix is a longer half-life or fewer
features. The steady fit's slope is 0.99, flagged on no row, and
rescaling changes nothing.

**What to do.** When the slope sits away from 1 for many half-lives,
rescale the predictions by the calibration as the recipe does, or remove
the cause: a smaller ridge, a faster learning rate, or a longer half-life.
When the slope is 1 and the error still high, the model is missing
information, and [What is it missing?](#what-is-it-missing) is the next
question.

## Has it broken?

**A relationship can break in three ways, and each diagnostic sees some of
them.** The intercept can shift, which gives the residuals a mean. The
noise can grow, which changes the residuals' spread. Or a slope can move,
which on a feature centred at zero leaves the residuals' mean at zero and
widens their spread only until the fit catches up.

| field | what it is | sees | with no break |
|---|---|---|---|
| `studentized_<t>` | the row's recursive residual over the spread of those before it, `z = (resid / error_inflation) / s`, read once that spread has 10 rows of Kish size | each row's surprise | about `N(0, 1)` |
| `cusum_<t>` | the weighted sum of the studentized residuals before the row, standardized: `Σωz / sqrt(Σω²)` | a mean in the residuals: an intercept shift | `N(0, 1)` |
| `cusum_sq_<t>` | their weighted mean square less 1, standardized: `(Σωz² / Σω − 1) · Σω / sqrt(2 Σω²)` | a change in the residuals' spread | `N(0, 1)` for Gaussian residuals |
| `break_wald_<t>` | the Wald distance between two least-squares fits of the target on the slot's features, one at the memory and one at four times it; null run once | a coefficient that moved, the slope's or the intercept's | `chi2(k)`, with `k` the coefficients |
| `drift_<t>` | true on the row a Page-Hinkley detector (Page 1954; Hinkley 1971) on `\|resid\| / sigma` climbs `drift_threshold` above its lowest point | a rise in the residuals' level, beyond `drift_delta` sigmas | false |

`ω` is each row's weight times its decay at `breaks_half_life`, the
model's half-life by default.

**Run once, the CUSUMs are Brown, Durbin and Evans' tests.** With unit
weights `Σω²` is the row count `r`, so `cusum · sqrt(r)` is their CUSUM
path, read against the 5% boundary `±0.948 (sqrt(T) + 2r / sqrt(T))` over
a run of `T` rows. `T` is the length of the run being tested, so a monitor
fixes it in advance as the horizon it will watch. On 200 streams of 3,000
rows with an intercept break of half a noise sd at row 1,500, the CUSUM
crossed on every one, 244 rows after the break on the median. It crossed
on 3.5% of the streams with no break, where statsmodels' own crossed on
5.0%, and found the breaks 277 rows after. The CUSUM of squares crossed on
5.0% of streams with no break and on every stream whose noise doubled.

**With a memory, the CUSUM is a moving sum**, as Chu, Hornik and Kuan's
(1995) MOSUM is, with an exponential window in place of a rectangular one.
Beside a fit at a half-life of 200, `|cusum| > 3` found every intercept
break within 100 rows, and `|cusum_sq| > 3` every variance break within 20.
`break_wald > 21.1`, `chi2(3)`'s 0.01% value, found every slope break, 94
rows after it on the median. On 40 streams of 25,000 rows with no
break, at the same half-life, `|cusum| > 3` held on 0.001% of rows,
`|cusum_sq| > 3` on 0.01% and `break_wald > 21.1` on 0.003%.

**The CUSUM cannot see a slope that moves on a feature centred at zero.**
The residuals of the old fit on the new relationship are `(b_old − b_new)
· x + noise`, whose mean is zero when `x`'s mean is. Run once over 200
streams, the CUSUM crossed on 2.5% of streams whose slope moved, the same
as with no break. `break_wald` exists for this case.

**`drift` sees a rise in the residuals' level, and little else.** It is
scaled by `sigma`, which adapts to a larger spread within a half-life. Beside
a fit at a half-life of 200 it found 1.5% of the variance breaks above and
none of the others. Its false-alarm rate depends on the residuals' tails,
which [Are its tails heavy?](#are-its-tails-heavy) measures.

The first recipe runs the CUSUM once over a stream whose intercept rises
at row 2,000, and restarts the fit from where the CUSUM crossed:

```python
import numpy as np
import polars as pl

import polars_online as po

# y = 0.2 + 1.0 x0 + 0.5 x1 + noise until row 2,000 of 4,000, where the intercept rises to 0.7.
rng = np.random.default_rng(2)
n, at = 4_000, 2_000
x0, x1 = rng.standard_normal((2, n))
t = np.arange(n, dtype=float)
y = 0.2 + 0.5 * (t >= at) + 1.0 * x0 + 0.5 * x1 + rng.standard_normal(n)
pl.DataFrame({"t": t, "x0": x0, "x1": x1, "y": y}).write_parquet("shift.parquet")
lf = pl.scan_parquet("shift.parquet")

# Run once: half_life=inf weighs every row alike, and the CUSUM runs over the whole stream.
model = po.spec.ewridge(
    "ols", targets=["y"], features=["x0", "x1"], half_life=float("inf"), min_weight=30.0,
    emit_breaks=True,
)
out = lf.online.fit_predict([model]).online.unnest([model]).collect()

# Brown, Durbin and Evans' path, the running sum of the studentized residuals, against the 5%
# boundary for a run of T of them.
path = out.select(
    "t",
    path=pl.col("studentized_y").fill_null(0.0).cum_sum(),
    r=pl.col("studentized_y").is_not_null().cum_sum(),
)
T = path["r"].max()
path = path.with_columns(bound=0.948 * (T**0.5 + 2 * pl.col("r") / T**0.5))
crossed = path.filter(pl.col("path").abs() > pl.col("bound"))["t"].min()
print(f"the path crossed its boundary at row {crossed:.0f}")
for row, value, bound in path.filter(pl.col("t").is_in([1_000.0, 2_000.0, 2_200.0, crossed])).select("t", "path", "bound").iter_rows():
    print(f"  row {row:>5.0f}: path {value:>6.1f}, boundary {bound:.1f}")


def coef(query):
    """The coefficients of a fit over every row of the query."""
    bank = po.ModelBank([po.spec.ewridge("ols", targets=["y"], features=["x0", "x1"], half_life=float("inf"))])
    bank.fit(query)
    return ", ".join(f"{term} {c:.2f}" for term, c in bank.coef().select("term", "coef").iter_rows())


print(f"every row:              {coef(lf)}")
print(f"from the crossing on:   {coef(lf.filter(pl.col('t') >= crossed))}")
```

```text
the path crossed its boundary at row 2310
  row  1000: path   30.3, boundary 88.6
  row  2000: path   -9.1, boundary 118.7
  row  2200: path   69.8, boundary 124.8
  row  2310: path  128.4, boundary 128.1
every row:              intercept 0.44, x0 1.02, x1 0.50
from the crossing on:   intercept 0.69, x0 0.98, x1 0.48
```

The path reads 30 at row 1,000 and −9 at row 2,000, the wandering of a
sum of independent residuals. Then it climbs, to 70 by row 2,200, because
each new row's residual carries part of the half-unit shift the fit has
not yet absorbed. It crosses the boundary at row 2,310, 310 rows after the break. A crossing is
a detection, so it comes after the break, by a delay that shrinks as the
break grows. The fit over every row mixes the two regimes and puts the
intercept at 0.44, between 0.2 and 0.7. Restarted from the crossing, the
fit reads 0.69, the new regime's intercept. Every row from the crossing on
belongs to the new regime, so restarting there loses rows but mixes
nothing. The model's `min_weight=30` keeps its first, wildest predictions
out of the run-once CUSUM ([One memory per
diagnostic](#one-memory-per-diagnostic-run-once-or-windowed)).

The second recipe runs the windowed forms beside a fit that forgets, on
three streams that break at row 3,000, each in one way:

```python
import numpy as np
import polars as pl

import polars_online as po

# Three streams that break at row 3,000 of 6,000, each in one way.
rng = np.random.default_rng(3)
n, at = 6_000, 3_000
x0, x1, e = rng.standard_normal((3, n))
after = np.arange(n) >= at
streams = {
    "mean": 1.0 * x0 + 0.5 * x1 + e + 0.5 * after,  # the intercept rises by half a noise sd
    "variance": 1.0 * x0 + 0.5 * x1 + e * np.where(after, 2.0, 1.0),  # the noise doubles
    "slope": 1.0 * x0 + np.where(after, -0.5, 0.5) * x1 + e,  # x1's slope flips sign
}
pl.DataFrame({"x0": x0, "x1": x1} | {f"y_{k}": v for k, v in streams.items()}).write_parquet("breaks.parquet")
lf = pl.scan_parquet("breaks.parquet").with_row_index("row")

specs = [
    po.spec.ewridge(
        kind, targets=[f"y_{kind}"], features=["x0", "x1"], half_life=200.0,
        emit_breaks=True,  # memory: the model's 200 rows
        emit_drift=True,  # at drift_delta=0.5 and drift_threshold=20
    )
    for kind in streams
]
out = lf.online.fit_predict(specs).collect()

flags = {
    "|cusum| > 3": lambda y: pl.col(f"cusum_{y}").abs() > 3,
    "|cusum_sq| > 3": lambda y: pl.col(f"cusum_sq_{y}").abs() > 3,
    "break_wald > 21.1": lambda y: pl.col(f"break_wald_{y}") > 21.1,
    "drift": lambda y: pl.col(f"drift_{y}"),
}
print(f"{'break':9}" + "".join(f"{name:>19}" for name in flags))
flagged_before = 0
for kind in streams:
    rows = out.select("row", pl.col(kind).struct.unnest())
    cells = []
    for flag in flags.values():
        first = rows.filter((pl.col("row") >= at) & flag(f"y_{kind}"))["row"].min()
        cells.append("-" if first is None else str(first))
        flagged_before += rows.filter(pl.col("row").is_between(1_000, at - 1) & flag(f"y_{kind}")).height
    print(f"{kind:9}" + "".join(f"{c:>19}" for c in cells))
print(f"flags on rows 1,000-2,999, before any break: {flagged_before}")
```

```text
break            |cusum| > 3     |cusum_sq| > 3  break_wald > 21.1              drift
mean                    3106                  -               3137                  -
variance                   -               3025                  -                  -
slope                      -               3115               3072                  -
flags on rows 1,000-2,999, before any break: 0
```

Each cell is the first row the flag fired after the break. Read together,
they tell the three breaks apart:

| break | `cusum` | `cusum_sq` | `break_wald` |
|---|---|---|---|
| the intercept shifted | fires | quiet | fires, since the intercept is a coefficient |
| the noise grew | quiet | fires first, within 25 rows | quiet |
| a slope moved | quiet | fires, while the fit catches up | fires |

`drift` fired on none of them. No flag fired before row 3,000.

**What to do.** When `cusum` fires alone or with `break_wald`, the level
moved: restart the fit from the crossing, or let a fit that forgets catch
up. When `cusum_sq` fires alone, the noise changed and the coefficients
did not: the fit is still right, but `sigma` and every interval built on
it will be wrong until they adapt. When `break_wald` fires with `cusum_sq`
and without `cusum`, a slope moved. `drift_action="reset"` restarts a
model at a `drift` flag, which suits a level shift only.

## Can its t be trusted?

**A coefficient's standard error assumes residuals that are independent
and equally spread, and a target that looks ahead breaks the first
assumption.** A target that sums the next `h` rows' changes shares `h − 1`
of them with its neighbour. Its residuals are then correlated over `h − 1`
rows by construction, a moving average of order `h − 1`. Against a feature
that moves slowly, the coefficient's variance grows by about `1 + 2 Σ ρ_x(l)
ρ_e(l)` over the lags, close to `h`, so a t that ignores the overlap is
about `sqrt(h)` times too large. Newey and West (1987) estimate the
coefficients' covariance with the lagged products of the scores added
back, at Bartlett weights that keep the estimate positive.

| field | standard error | robust to |
|---|---|---|
| `se_coef` (`emit_se_coef`) | `sigma · sqrt(diag(M))`, `M` the inverse Gram over `n_kish`; `kalman`'s posterior | nothing: it assumes independent residuals of one spread |
| `se_coef_hc0` (`emit_robust_se`) | White's (1980) sandwich `B⁻¹ (Σ w² e² z z') B⁻¹`, `B` the weighted Gram of `(1, x)` | a spread that moves with the features |
| `se_coef_hac` (`emit_robust_se`) | the same sandwich with Newey and West's lag products to `robust_se_lags`, weighted `1 − l / (L + 1)` | that, and residuals correlated up to `L` rows apart |

All three sit on `coef`'s rows, laid out like `coef`, the intercept first.
`e` is the row's out-of-sample residual, so the sandwich carries the
estimation error that a sandwich over in-sample residuals leaves out. Run
once on 1,500 rows, it read 0.9% to 1.3% above statsmodels'
`cov_type="HC0"` and `"HAC"`. The lags default to twice the target's
horizon in rows, `2 × embargo` on a spec with no clock column, and 0 with
one. Give `robust_se_lags` when the clock is a time. The memory is
`robust_se_half_life`, the model's by default. It costs 350 ns a
row at 5 features and 10 lags, and only `ewridge` and `rls` take it, the
least-squares fits.

The recipe fits 100 streams in one pass, one group each. Each target sums
the next 10 shocks, which the feature knows nothing about, so the true
slope is 0 and a test at 5% should reject on about 5% of the streams:

```python
import numpy as np
import polars as pl

import polars_online as po

# 100 independent streams of 3,000 rows. x is persistent (AR(1) at 0.95); the target sums the next
# h = 10 shocks, which x knows nothing about, so its true slope is 0.
rng = np.random.default_rng(4)
streams, n, h = 100, 3_000, 10
frames = []
for s in range(streams):
    x = np.zeros(n)
    u = rng.standard_normal(n)
    for i in range(1, n):
        x[i] = 0.95 * x[i - 1] + u[i]
    e = rng.standard_normal(n + h)
    fwd = np.convolve(e, np.ones(h), "valid")[1 : n + 1]  # e[i+1] + ... + e[i+h]
    frames.append(pl.DataFrame({"stream": s, "x": x, "fwd": fwd}))
pl.concat(frames).write_parquet("overlap.parquet")
lf = pl.scan_parquet("overlap.parquet")

model = po.spec.ewridge(
    "m", targets=["fwd"], features=["x"], half_life=float("inf"), group="stream",
    embargo=h,  # a row's label is known h rows later, and is learned only then
    emit_se_coef=True,  # se_coef: assumes uncorrelated residuals
    emit_robust_se=True,  # se_coef_hc0, and se_coef_hac to 2 * embargo = 20 lags
)
out = lf.online.fit_predict([model]).collect()
last = out.group_by("stream", maintain_order=True).last().select(pl.col("m").struct.unnest())
slope = last.select(
    coef=pl.col("coef").list.get(1),  # [intercept, x]
    se=pl.col("se_coef").list.get(1),
    hc0=pl.col("se_coef_hc0").list.get(1),
    hac=pl.col("se_coef_hac").list.get(1),
)
print(f"the slope's spread over the {streams} streams: {slope['coef'].std():.4f}")
for se in ["se", "hc0", "hac"]:
    rejected = (slope["coef"] / slope[se]).abs().gt(1.96).mean()
    print(f"{se:4} median {slope[se].median():.4f}, |t| > 1.96 on {rejected:.0%} of the streams")
```

```text
the slope's spread over the 100 streams: 0.0546
se   median 0.0188, |t| > 1.96 on 44% of the streams
hc0  median 0.0188, |t| > 1.96 on 42% of the streams
hac  median 0.0504, |t| > 1.96 on 8% of the streams
```

The slope estimates spread with a standard deviation of 0.0546 over the
100 streams. `se_coef` puts it at 0.0188, 2.9 times too small, close to
`sqrt(10) = 3.2`, so a t that reads it finds a slope on 44% of the streams
where there is none. `se_coef_hc0` is no better, because the residuals'
spread does not move with `x`: their fault is correlation. `se_coef_hac`
reads 0.0504, near the true spread, and rejects on 8%.

Measured more widely, on a target summing the next `h` rows' shocks
against a persistent feature, the coefficient's true spread was 2.1 to 2.3
times `se_coef` at `h = 5`, and 3.1 to 4.2 times at `h = 20`. `se_coef_hac`
read 83% to 97% of the true spread at `L = h`, and 90% to 104% at `L = 2h`,
the default.

**`po.gram.coef_stats` and `marginal`'s `t` read Kish's size too, and are
silent about the overlap the same way.** `po.gram.coef_stats` computes each
coefficient's standard error and t from a Gram, such as one loaded from a
state, at `n = target_n_kish` ([The running sums behind a
fit](../README.md#the-running-sums-behind-a-fit)). Read its t as a scale for
comparing coefficients. `marginal`'s `t_serial` corrects each pair's count
for rows that resemble their neighbours, after Bartlett (1935): on two
independent AR(1) series, `t = 2.39` became `t_serial = 1.03`
([`marginal`](../README.md#marginal--every-pairs-moments-kept-in-the-state)).

**What to do.** For a target that looks ahead, read `se_coef_hac`, with
`robust_se_lags` at least the horizon in rows. For a spread that moves with
the features, read `se_coef_hc0`. `se_coef` is right only for a one-step
target with residuals of one spread, which `breusch_pagan` and
`ljung_box` can confirm.

## What is it missing?

**A residual stream with nothing left to learn looks like noise: no lag,
no spread that follows a feature, no curvature.** Three tests and one
correlation look for what is left.

| field | statistic | with nothing missing | sees |
|---|---|---|---|
| `ljung_box_<t>` | Ljung and Box's (1978) `Q = n(n + 2) Σ_{l=1..L} ρ̂_l² / (n − l)` over `ljung_box_lags` lags (default 10), at Kish's `n` | `chi2(L)`: 18.3 at 5% for 10 lags | a lag the fit is missing, or a half-life too long |
| `breusch_pagan_<t>` | Koenker's (1981) form of Breusch and Pagan's (1979) test: `n R²` of `resid²` regressed on the slot's features | `chi2(k)` for `k` features | a spread that moves with a feature, linearly |
| `reset_<t>` | Ramsey's (1969) RESET as a Lagrange multiplier: `n R²` of `resid` on `pred²` and `pred³` beside `pred` | `chi2(2)`: 5.99 at 5% | a curvature the fit is missing |
| `autocorr_<t>` (`emit_autocorr`) | the EW correlation of each residual with the one `resid_autocorr_lag` scored rows back (default 1) | near 0 | one lag, read as a correlation |

`emit_specification` writes the first three at `specification_half_life`,
the model's half-life by default. Run once, they are statsmodels'
`acorr_ljungbox`, `het_breuschpagan` and `compare_lm_test` on the
out-of-sample residuals. On 200 streams of 3,000 rows, each passed its 5%
value on 2.0% to 7.0% of the rows of streams missing nothing, run once or
windowed. Each passed on 98% to 100% of the rows of streams missing what it
looks for: an AR(1) at 0.3, a spread linear in a feature, a square of one.
`emit_autocorr` reads at the model's half-life, and a gap capped by
`gap_cap` or a session change starts a new run of lags.

**On a target that looks ahead `h` rows, `ljung_box` tests lags `h` to
`h + L − 1`.** Its residuals are correlated within the horizon by
construction, so lags 1 to `h − 1` would find the overlap and nothing else.
The plain `Q` passed its 5% value on 61% of the rows of five-row look-ahead
streams with nothing missing, and the shifted one on 6.0% to 6.7%. The
horizon is `embargo` on a spec with no clock column.

**Each test sees only its own shape.** Breusch-Pagan regresses `resid²` on
the features themselves, so a spread symmetric in a feature, as `|x0|`, is
invisible to it. On the recipe's stream with its noise times `0.5 +
|x0|` in place of `exp(0.5 x0)`, it read a median of 3.0, under `chi2(2)`'s
5% value of 5.99. `reset`, whose `pred²` carries `x0²`, read 7.7, and
caught the spread in its place. Ljung-Box sees correlation at the lags it
tests and no further. RESET sees curvature along the prediction, so a
curvature in a feature the prediction barely uses can hide from it.

The recipe plants one fault in each of three streams, fixes two of them
with Polars before the bank, and reads each statistic's median over rows
2,000 to 5,999:

```python
import numpy as np
import polars as pl

import polars_online as po

# y = 1.0 x0 + 0.5 x1 + noise on every stream, but for one fault each.
rng = np.random.default_rng(5)
n = 6_000
x0, x1, e = rng.standard_normal((3, n))
ar = np.zeros(n)  # noise that carries 0.4 of the row before
for i in range(1, n):
    ar[i] = 0.4 * ar[i - 1] + e[i]
base = 1.0 * x0 + 0.5 * x1
pl.DataFrame(
    {
        "x0": x0,
        "x1": x1,
        "y_clean": base + e,
        "y_lag": base + ar,  # a lag the fit is missing
        "y_spread": base + e * np.exp(0.5 * x0),  # a spread that grows with x0
        "y_curve": base + 0.3 * x0**2 + e,  # a curvature in x0
    }
).write_parquet("missing.parquet")
lf = pl.scan_parquet("missing.parquet").with_columns(
    # The fixes, made with Polars before the bank: last row's error for the lag, x0's square for the curvature.
    y_lag_prev=pl.col("y_lag").shift(1) - pl.col("x0").shift(1) - 0.5 * pl.col("x1").shift(1),
    x0_sq=pl.col("x0") ** 2,
)


def spec(name, target, features):
    return po.spec.ewridge(
        name, targets=[target], features=features, half_life=1_000.0,
        emit_specification=True,  # ljung_box (10 lags), breusch_pagan, reset, at a memory of 1,000 rows
    )


specs = [
    spec("clean", "y_clean", ["x0", "x1"]),
    spec("lag", "y_lag", ["x0", "x1"]),
    spec("spread", "y_spread", ["x0", "x1"]),
    spec("curve", "y_curve", ["x0", "x1"]),
    spec("lag_fixed", "y_lag", ["x0", "x1", "y_lag_prev"]),
    spec("curve_fixed", "y_curve", ["x0", "x1", "x0_sq"]),
]
out = lf.online.fit_predict(specs).slice(2_000).collect()

# Each statistic's median. With nothing missing: chi2(10) has a median of 9.3, chi2(2) or chi2(3) of 1.4 or 2.4.
print(f"{'spec':12} {'ljung_box':>10} {'breusch_pagan':>14} {'reset':>6}")
for s in specs:
    name, y = s["name"], s["targets"][0]
    medians = out.select(pl.col(name).struct.unnest()).select(
        pl.col(f"ljung_box_{y}").median(),
        pl.col(f"breusch_pagan_{y}").median(),
        pl.col(f"reset_{y}").median(),
    )
    print(f"{name:12} " + " ".join(f"{v:>{w}.1f}" for v, w in zip(medians.row(0), (10, 14, 6))))
```

```text
spec          ljung_box  breusch_pagan  reset
clean              14.0            1.0    1.8
lag               432.9            2.3    1.9
spread             15.5          419.2    4.8
curve              12.9            1.6  214.5
lag_fixed          13.8            3.2    0.7
curve_fixed        13.8            1.8    1.1
```

The clean stream reads 14.0, 1.0 and 1.8, each below its 5% value (18.3,
5.99 and 5.99). Each faulty stream lights one test, by a factor of 30 or
more over the clean stream: `ljung_box` reads 432.9 for the missing lag,
`breusch_pagan` 419.2 for the spread, and `reset` 214.5 for the
curvature.
With last row's error as a feature, the lagged stream's `ljung_box` falls
back to 13.8, and with `x0²` as a feature the curved stream's `reset`
falls to 1.1. The spread has no fix in the features.
Its coefficients are still right, but their `se_coef` is not, so read
`se_coef_hc0` ([Can its t be trusted?](#can-its-t-be-trusted)), or weight
each row by the inverse of its expected variance.

**What to do.** For `ljung_box`, add the lag the residuals carry, as a
feature made with Polars' `shift` in the query before the bank, or
shorten the half-life if the fit is lagging a moving relationship. For
`reset`, add the curvature: a square, a product, or a transform of the
feature that carries it. For `breusch_pagan`, read the robust standard
errors.

## Are its tails heavy?

**Heavy tails change what every threshold on the residuals means, so
measure them before setting one.** A detector tuned on Gaussian residuals
fires more often on residuals with heavy tails, because large residuals
are more common there, and nothing has broken.

| | `emit_tails` |
|---|---|
| fields | `skew_<t>`, `kurtosis_<t>` (excess: 0 for a Gaussian), `jarque_bera_<t>` |
| what it reads | the recursive residuals `resid / error_inflation`, or `resid` on a model without it |
| statistic | Jarque and Bera's (1980) `n / 6 · (skew² + kurtosis² / 4)` at Kish's `n`: `chi2(2)` for Gaussian residuals |
| memory | `tails_half_life`, the model's by default; run once they are `scipy.stats`' `skew` and `kurtosis` and statsmodels' `jarque_bera` |
| threshold | `jarque_bera > 5.99`: 4.7% to 5.0% of the rows of Gaussian residuals on 200 streams, and every row of Student's t with 5 degrees of freedom |
| cost, models | 157 ns a row at 5 features; the ten linear models |

**`drift_threshold` is set against the residuals' kurtosis.** On 40
streams of 25,000 rows at a half-life of 200, read as the mean of
`kurtosis` over each stream, the default threshold of 20 flagged:

| noise | mean `kurtosis` | flags per 100,000 rows at 20 | at 30 | at 50 | at 80 |
|---|---:|---:|---:|---:|---:|
| Gaussian, Student's t with 10, 6 or 5 degrees of freedom | under 4 | 0 | 0 | 0 | 0 |
| Student's t, 4 degrees of freedom | 8.6 | 0.4 | 0 | 0 | 0 |
| Student's t, 3 degrees of freedom | 36 | 6.3 | 2.1 | 0.4 | 0 |

**The CUSUM of squares assumes Gaussian residuals, and it fires often on
heavy tails.** It standardizes a squared studentized residual by 2, its
variance under a Gaussian. Under an excess kurtosis `κ` that variance is
`2 + κ`. On 40 streams of 25,000 rows at a half-life of 200, `|cusum_sq| > 3`
held on 0.01% of Gaussian rows, 2.8% of rows of t with 5 degrees of
freedom, 6.8% with 4 and 19% with 3. Widening its threshold to `3 · sqrt(1 +
κ / 2)` brought those to 0.5%, 1.1% and 2.8%, still far above the Gaussian
rate. `|cusum| > 3` and `break_wald > 21.1` held on 0.14% of rows or
fewer, even with 3 degrees of freedom.

**A windowed kurtosis is itself noisy under heavy tails.** The kurtosis of
Student's t with 3 degrees of freedom is infinite, so a windowed estimate
never settles. Over the rows of those 40 streams its median was 13 and its
mean 32. Read the mean over a long stable stretch, as the measurements in
the table did.

The recipe measures the tails over a stretch known to be stable, sets
`drift_threshold` from them, and counts each detector's flags over 200,000
stable rows and after a break that triples the noise:

```python
import numpy as np
import polars as pl

import polars_online as po

# 200,000 rows of a stable relation whose noise is Student's t with 3 degrees of freedom, then
# 20,000 rows where the noise is three times as large.
rng = np.random.default_rng(6)
n, at = 220_000, 200_000
x0, x1 = rng.standard_normal((2, n))
noise = rng.standard_t(3, n) * np.where(np.arange(n) >= at, 3.0, 1.0)
pl.DataFrame({"x0": x0, "x1": x1, "y": 1.0 * x0 + 0.5 * x1 + noise}).write_parquet("tails.parquet")
lf = pl.scan_parquet("tails.parquet").with_row_index("row")

# 1. Read the tails over a stretch known to be stable: the mean of kurtosis over rows 1,000-20,000.
stable = po.spec.ewridge("m", targets=["y"], features=["x0", "x1"], half_life=200.0, emit_tails=True)
tails = lf.head(20_000).online.fit_predict([stable]).online.unnest([stable]).collect()
kurtosis = tails.filter(pl.col("row") >= 1_000)["kurtosis_y"].mean()
print(f"excess kurtosis of the residuals: {kurtosis:.1f}")

# 2. The thresholds that kurtosis calls for: drift's from the table above, and the CUSUM of squares'
#    3 widened by the spread a squared residual has under that kurtosis.
threshold = 20.0 if kurtosis < 4 else 30.0 if kurtosis < 10 else 50.0 if kurtosis < 30 else 80.0
widened = 3.0 * (1.0 + kurtosis / 2.0) ** 0.5
print(f"drift_threshold {threshold:.0f}; |cusum_sq| past {widened:.1f}")


# 3. Count each detector's flags over the 200,000 stable rows, and find the break.
def report(name, out, flag):
    false = out.filter(pl.col("row") < at).select(flag.sum()).item()
    found = out.filter((pl.col("row") >= at) & flag)["row"].min()
    print(f"{name:18} {false:>6} flags before the break, first after it {'never' if found is None else f'{found - at} rows in'}")


for value in [20.0, threshold]:
    spec = po.spec.ewridge(
        "m", targets=["y"], features=["x0", "x1"], half_life=200.0,
        emit_drift=True, drift_threshold=value, emit_breaks=True,
    )
    out = lf.online.fit_predict([spec]).online.unnest([spec]).collect()
    report(f"drift at {value:.0f}", out, pl.col("drift_y"))
report("|cusum_sq| > 3", out, pl.col("cusum_sq_y").abs() > 3)
report(f"|cusum_sq| > {widened:.1f}", out, pl.col("cusum_sq_y").abs() > widened)
```

```text
excess kurtosis of the residuals: 16.6
drift_threshold 50; |cusum_sq| past 9.1
drift at 20            11 flags before the break, first after it 15011 rows in
drift at 50             1 flags before the break, first after it never
|cusum_sq| > 3      33513 flags before the break, first after it 18 rows in
|cusum_sq| > 9.1     6740 flags before the break, first after it 80 rows in
```

The stretch reads an excess kurtosis of 16.6, which calls for a
`drift_threshold` of 50. At the default 20, `drift` flagged 11 of the
200,000 stable rows, and at 50 it flagged one. Neither found the break: when
the noise triples, `sigma` triples with it, and `drift` scores each
residual against `sigma`. Its first flag after the break at 20 is a false alarm,
15,011 rows in. The CUSUM of squares found the break 18 rows in, but it
also flagged 33,513 of the stable rows, 17% of them. Widened to 9.1 it
flagged 6,740, still 3.4%, and found the break 80 rows in.

**What to do.** Measure `kurtosis` over a stable stretch before trusting
any threshold on the residuals. Set `drift_threshold` from it by the table.
On residuals with an excess kurtosis past 4, read a change of spread from
a run of `cusum_sq` above its threshold for many rows, or from `sigma`
itself, and not from a single crossing. A robust model, `huber` or
`quantile`, limits how far one large residual moves the fit.

## Which row moved it?

**One bad row can move a fit more than a thousand good ones, and
`influence` names it.** A row's influence on a least-squares fit is its
residual times its leverage: a row far out in the features and far off the
line moves the coefficients most. Belsley, Kuh and Welsch's (1980) DFFITS
measures that move in the fit's own metric. `influence` is its online
form, against the fit before the row:

```text
influence = (v / s) · sqrt(h),     h = error_inflation² − 1
```

Here `h` is the row's leverage against the fit before it, `v` its
recursive residual, and `s` the spread of those before it at
`influence_half_life`. Its square is the move of the coefficients,
`Δb' A Δb / s²`, times `1 + h`. Run once with no ridge, it is statsmodels'
`OLSInfluence.dffits` at the last row of the rows so far, to 7e-11. It is
not the in-sample DFFITS of an earlier row, which reads the rows after it
too. It costs 145 ns a row at 5 features, and only `ewridge`, `rls` and
`kalman` take it, the models that read a row's leverage.

The recipe plants three bad prints in 10,000 rows, each with `x0` six
spreads out and `y` six noise sds off the line:

```python
import numpy as np
import polars as pl

import polars_online as po

# 10,000 rows of y = 1.0 x0 + 0.5 x1 + noise, and three bad prints.
rng = np.random.default_rng(7)
n = 10_000
x0, x1, e = rng.standard_normal((3, n))
y = 1.0 * x0 + 0.5 * x1 + e
bad = [2_500, 6_000, 8_200]
x0[bad] = 6.0  # six spreads out ...
y[bad] = 1.0 * x0[bad] + 0.5 * x1[bad] - 6.0  # ... and six noise sds off the line
pl.DataFrame({"x0": x0, "x1": x1, "y": y}).write_parquet("prints.parquet")
lf = pl.scan_parquet("prints.parquet").with_row_index("row")

model = po.spec.ewridge(
    "m", targets=["y"], features=["x0", "x1"], half_life=200.0,
    emit_influence=True,  # influence_y: how far the row moved the fit
    coef_every=0, max_rows_between_solves=1,  # solve and write the coefficients on every row
)
out = lf.online.fit_predict([model]).online.unnest([model]).collect().slice(1_000)  # past the warm-up
moved = out.with_columns(step=pl.col("coef_y_x0").diff().abs())  # how far x0's coefficient moved

print("the five rows that moved the fit most:")
top = moved.sort(pl.col("influence_y").abs(), descending=True, nulls_last=True).head(5)
for row, influence, step in top.select("row", "influence_y", "step").iter_rows():
    print(f"  row {row:>5}: influence {influence:+.2f}, x0's coefficient moved by {step:.3f}")
clean = moved.filter(~pl.col("row").is_in(bad))["influence_y"].abs()
print(f"every other row: |influence| at most {clean.max():.2f}, 99.9th percentile {clean.quantile(0.999):.2f}")
```

```text
the five rows that moved the fit most:
  row  8200: influence -1.63, x0's coefficient moved by 0.122
  row  6000: influence -1.55, x0's coefficient moved by 0.110
  row  2500: influence -1.49, x0's coefficient moved by 0.114
  row  2360: influence -0.55, x0's coefficient moved by 0.035
  row  3593: influence -0.43, x0's coefficient moved by 0.035
every other row: |influence| at most 0.55, 99.9th percentile 0.30
```

The three planted rows rank first, each at an influence near −1.5, and
each moved `x0`'s coefficient by 0.11 to 0.12 in one row. The next row
reads −0.55. Among the 8,997 clean rows past the warm-up, the largest
reads 0.55 and the 99.9th percentile 0.30. On a fit at a half-life of 200,
`|influence| > 0.5` held every planted row and one clean row in 9,000.
The first rows of a stream move the fit by much more, because a fit on a
handful of rows has little to resist with, so read `influence` once the
fit has settled.

**What to do.** Find the rows, then decide what they are: a print error
to drop with a filter in the query before the bank, or a real event the
model should keep. A row that is real but rare is the case for `huber`,
whose loss turns linear past `huber_delta`, so one row cannot move the fit
far.

## Are its features healthy?

**A feature can go wrong without any residual showing it: a feed that
stops and is carried forward is still a number on every row.** A fit that
keeps sums, such as `rls`, then sees a column that no longer varies. Under
forgetting, the information about that coefficient decays and nothing
replaces it, so the coefficient winds up (Åström and Wittenmark 1995).
[Detecting data issues](DATA-ISSUES.md#constant-and-frozen-features)
measures the windup. `emit_feature_health` watches the features
themselves, before any coefficient moves.

| | `emit_feature_health` |
|---|---|
| fields | `spread_ratio_<feature>`, `mean_shift_<feature>`, once per instance and feature, read before the row |
| update | each feature's EW mean and standard deviation at `feature_health_half_life` and at four times it |
| statistic | `spread_ratio` is the fast standard deviation over the slow one; `mean_shift` the two means' difference in units of the slow standard deviation. A steady feature reads about 1 and 0 |
| memory | the model's half-life by default; null run once, with no longer run to compare with |
| threshold | `spread_ratio` outside 0.5 to 2: no row of a clean feature at a half-life of 200, independent or AR(1) at 0.95. `\|mean_shift\| > 0.3`: every move of one spread, within 190 rows; 6% of rows of an AR(1) feature at 0.95, and 0.03% above 0.5 |
| cost, models | 36 ns a row at 5 features; the nine linear models with features |

The recipe holds `x1` at its last value for 3,000 rows, as a feed that
stopped would, and fits `rls`:

```python
import numpy as np
import polars as pl

import polars_online as po

# y = x0 + 0.5 x1 + noise. x1's feed stops at row 300 and comes back at row 3,300; the join that
# built the table carried its last value forward.
rng = np.random.default_rng(3)
n = 3_500
t = np.arange(n, dtype=float)
x0, x1 = rng.standard_normal((2, n))
y = x0 + 0.5 * x1 + 0.2 * rng.standard_normal(n)
held = (t >= 300) & (t < 3_300)
pl.DataFrame({"t": t, "x0": x0, "x1": np.where(held, x1[299], x1), "y": y}).write_parquet("feed.parquet")
lf = pl.scan_parquet("feed.parquet")

model = po.spec.rls(
    "m", targets=["y"], features=["x0", "x1"], clock="t", gap_cap=5.0, half_life=20.0,
    coef_every=0,  # the coefficients on every row
    emit_feature_health=True,  # spread_ratio_x0, spread_ratio_x1, mean_shift_x0, mean_shift_x1
)
out = lf.online.fit_predict([model]).online.unnest([model]).collect()

quiet = out.filter((pl.col("t") >= 300) & (pl.col("spread_ratio_x1") < 0.5))["t"].min()
wound = out.filter((pl.col("t") >= 300) & (pl.col("coef_y_x1").abs() > 1.0))["t"].min()
print(f"x1 held from row 300; its spread_ratio fell below 0.5 at row {quiet:.0f}")
print(f"rls's coefficient on x1 (0.5 in the data) passed 1 at row {wound:.0f}")
for row, ratio, coef in out.filter(pl.col("t").is_in([299.0, quiet, 1_000.0, wound, 3_299.0])).select("t", "spread_ratio_x1", "coef_y_x1").iter_rows():
    print(f"  row {row:>5.0f}: spread_ratio_x1 {ratio:.3f}, coefficient on x1 {coef:.3g}")
```

```text
x1 held from row 300; its spread_ratio fell below 0.5 at row 358
rls's coefficient on x1 (0.5 in the data) passed 1 at row 1352
  row   299: spread_ratio_x1 0.891, coefficient on x1 0.492
  row   358: spread_ratio_x1 0.499, coefficient on x1 0.332
  row  1000: spread_ratio_x1 0.000, coefficient on x1 0.333
  row  1352: spread_ratio_x1 0.000, coefficient on x1 1.23
  row  3299: spread_ratio_x1 0.000, coefficient on x1 -4.61e+13
```

The feed stops at row 300. `spread_ratio_x1` falls below 0.5 at row 358,
2.9 half-lives in, while the coefficient has barely moved. `rls`'s
coefficient on `x1` passes 1 at row 1,352, 52.6 half-lives in, and by the
time the feed returns it has blown up. The feature-health flag came a
thousand rows before the fit showed anything.

**What to do.** Fix a stale feed in the query before the bank: null each
value older than its own sampling interval, so the model skips the rows it
did not observe ([Detecting data issues](DATA-ISSUES.md#constant-and-frozen-features)
has the recipe). Or use a model whose ridge does not fade, such as
`ewridge`, which does not wind up. A `mean_shift` that stays large says the
feature moved to a new level: a coefficient fitted at the old level may
not hold there.

## Is another model better?

**A model is better than another only on residuals neither has learned,
and every comparison here reads those.** The comparisons come in three
kinds: a choice among a grid's settings as the stream runs, a test that
one model predicts better than another, and metrics over an output frame
after the run.

| kind | switch or call | what it gives |
|---|---|---|
| running accuracy | `emit_metrics` | `ic_<t>`, the EW correlation of prediction with target; `r2_<t>`, the out-of-sample R² against the running mean; `hit_rate_<t>`, the share of rows whose sign the prediction got right |
| a choice as the stream runs | `emit_selected`, `emit_averaged` | the grid slot with the lowest EW error so far, and every slot's prediction weighted by its error |
| a test of two models | `po.eval.diebold_mariano`, `po.eval.clark_west`, `po.eval.seqtest`, the `seqtest` model | a t statistic, or evidence that can be read at any row |
| metrics after the run | `po.eval.metrics`, `window_metrics`, `compare_specs`; `po.eval.sums` for an output too large to hold | R², IC, hit rate and MSE, per group or per window |

**`hit_rate` asks whether prediction and outcome fall on the same side of
zero.** A row where either is exactly zero is on neither side, and both
`emit_metrics` and `po.eval.metrics` leave it out. So on a target that is
always positive, such as a plain ratio, every row counts as a hit. Write a
return about zero with Polars expressions ([Relative and look-ahead
targets](../README.md#relative-and-look-ahead-targets)). An `sgd` fit with
`loss="poisson"` has no side to be on, a positive rate against a count,
and its `hit_rate` is null.

On an `sgd` or `ftrl` fit with `loss="logistic"`, `pred` is a probability
and `y` a 0/1 label, so the three metrics mean something else under the
same names:

| field | on a logistic `sgd` or `ftrl` fit |
|---|---|
| `hit_rate` | the accuracy at a 0.5 threshold |
| `r2` | the Brier skill score against the running base rate |
| `ic` | the point-biserial correlation between the probability and the label |
| a log loss | not streamed: `po.eval.metrics(..., binary=True)` computes one over a frame in memory |

### Choosing among a grid's settings

Two switches choose among a grid's slots as the stream runs, by each
slot's exponentially weighted out-of-sample error so far. `emit_selected`
commits to the best slot, and a spec with one slot per target refuses it.
`emit_averaged` hedges across all of them.

| switch | fields, one per target | what it gives |
|---|---|---|
| `emit_selected` | `selected_<t>`, `pred_<t>__selected` | the ridge value, feature set or half-life with the lowest EW out-of-sample error so far, each slot at its own half-life |
| `emit_averaged` | `pred_<t>__averaged` | every slot's prediction, weighted by `exp(-eta * (mse / best_mse - 1))`, with `mse` the slot's EW squared error and `best_mse` the best slot's; `average_eta` is `eta` (default 1), and `inf` gives `emit_selected`'s choice |

**Across a half-life grid, each slot's error decays at its own half-life,
so a short half-life is ranked on fewer, more recent rows.** Within a
ridge or feature-set grid, the slots share one half-life and are ranked
like for like. `lasso`'s `select_half_life` ranks its path on one
half-life.

The recipe chooses a half-life for a stream whose coefficient on `x0`
wanders as a slow random walk:

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
lf = pl.scan_parquet("wander.parquet")

grid = po.spec.ewridge(
    "grid", targets=["y"], features=["x0", "x1"], clock="t", gap_cap=5.0,
    half_life=[25.0, 100.0, 400.0, 1_600.0, 6_400.0],  # one fit per half-life, from the same rows
    emit_selected=True,  # selected_y, pred_y__selected: the best half-life so far, row by row
)
out = lf.online.fit_predict([grid]).collect()

# Each half-life's out-of-sample error over the whole stream, and the stream-time choice's.
for slot, mse, r2 in po.eval.metrics(out, "grid").sort("mse").select("slot", "mse", "r2").iter_rows():
    print(f"{slot:18} mse {mse:.4f}, R² {r2:.4f}")
held = out.select(pl.col("grid").struct.field("selected_y")).to_series().value_counts(normalize=True)
for slot, share in held.sort("proportion", descending=True).head(3).iter_rows():
    print(f"selected {slot}: {share:.0%} of the rows")
```

```text
pred_y@h100        mse 1.0249, R² 0.9064
pred_y@h400        mse 1.0405, R² 0.9050
pred_y__selected   mse 1.0439, R² 0.9047
pred_y@h25         mse 1.0523, R² 0.9039
pred_y@h1600       mse 1.1654, R² 0.8936
pred_y@h6400       mse 1.7095, R² 0.8439
selected @h25: 41% of the rows
selected @h400: 30% of the rows
selected @h100: 27% of the rows
```

Over the whole stream, a half-life of 100 rows predicts best, with 400 and
25 close behind and 6,400 far worse: too long a memory averages over a
coefficient that has moved. That is the half-life to choose. Chosen as the
stream ran, the selection held 25 on 41% of the rows, 400 on 30% and 100
on 27%, and its error sits between the best and the third. Each slot is
ranked at its own half-life, so the shortest is ranked on the fewest rows
and wins its share by luck. Choose a half-life over a long stretch with
`po.eval.metrics`, and use `emit_selected` where the best setting itself
moves.

### Comparing two models

**Diebold and Mariano's (1995) test asks whether two models' squared
errors differ on average.** Its statistic is the mean of the loss
differential `d = resid_b² − resid_a²` over Newey and West's standard
error, about `N(0, 1)` when the two predict equally well.
`po.eval.diebold_mariano(out, a=, b=)` computes it, with `lags` at least
`h − 1` for a target that looks `h` rows ahead, and 0 for one-step
predictions.

**Between two nested models, the test leans toward the smaller one, and
Clark and West's (2007) correction removes the lean.** When the larger
model's extra coefficients are truly zero, it still estimates them, and
their noise adds `(pred_small − pred_big)²` to its squared error on
average. So under the null the differential favours the smaller model,
and the test rarely finds the larger one better even when it is (Clark and
McCracken 2001). Clark and West add the term back:

```text
f  = resid_small² − resid_big² + (resid_big − resid_small)²
cw = f̄ · n / sqrt(S)            one-sided: cw > 1.645 rejects at 5%
```

`po.eval.clark_west(out, big=, small=)` computes it. With `half_life`,
each call gives its statistic on every row instead, exponentially
weighted. `po.eval.seqtest` and the `seqtest` model give evidence that one
model predicts closer, by betting on the sign of `|resid_b| − |resid_a|`.
It can be read at any row, as often as you like: reject the first time
`log_e_a ≥ ln 20` for the 5% level
([`seqtest`](../README.md#seqtest--a-sequential-test-of-a-sign-by-betting)).

The recipe compares two nested models on 200 streams, one group each: on
half of them the larger model's extra coefficient is 0.05, on the other
half 0:

```python
import numpy as np
import polars as pl

import polars_online as po

# 200 streams of 2,000 rows: y = x0 + c x1 + noise, with c = 0.05 on half of them and 0 on the rest.
# "small" fits x0 alone; "big" fits x0 and x1, so it nests "small".
rng = np.random.default_rng(10)
streams, n = 200, 2_000
c = np.repeat([0.05, 0.0], streams // 2)
x0, x1, e = rng.standard_normal((3, streams, n))
pl.DataFrame(
    {
        "stream": np.repeat(np.arange(streams), n),
        "c": np.repeat(c, n),
        "x0": x0.ravel(),
        "x1": x1.ravel(),
        "y": (x0 + c[:, None] * x1 + e).ravel(),
    }
).write_parquet("nested.parquet")
lf = pl.scan_parquet("nested.parquet")

common = dict(targets=["y"], half_life=float("inf"), group="stream", min_weight=20.0)
small = po.spec.ewridge("small", features=["x0"], **common)
big = po.spec.ewridge("big", features=["x0", "x1"], **common)
out = lf.online.fit_predict([small, big]).collect()

# One test per stream: does big predict better than small?
dm = po.eval.diebold_mariano(out, a="big", b="small", group="stream")
cw = po.eval.clark_west(out, big="big", small="small", group="stream")
tests = (
    out.group_by("stream").agg(pl.col("c").first())
    .join(dm.select("stream", "dm", dm_p="p_value"), on="stream")
    .join(cw.select("stream", cw_p="p_value"), on="stream")
)
found = tests.group_by("c").agg(
    diebold_mariano=((pl.col("dm") > 0) & (pl.col("dm_p") < 0.10)).mean(),  # one-sided at 5%
    clark_west=(pl.col("cw_p") < 0.05).mean(),
)
for c_, dm_share, cw_share in found.sort("c", descending=True).iter_rows():
    print(f"extra coefficient {c_:.2f}: big found better on {dm_share:.0%} of streams by "
          f"Diebold-Mariano, {cw_share:.0%} by Clark-West")
```

```text
extra coefficient 0.05: big found better on 1% of streams by Diebold-Mariano, 44% by Clark-West
extra coefficient 0.00: big found better on 0% of streams by Diebold-Mariano, 2% by Clark-West
```

Where the extra coefficient is 0.05, Diebold and Mariano's test finds the
larger model better on 1% of the streams, below its own 5% size: the lean
toward the smaller model swamps a small real gain. Clark and West's finds
it on 44%. Where the extra coefficient is 0, Clark and West's rejects on
2%, within its 5% size. On 200 streams in the measurements behind
`po.eval`, the same comparison gave 1.5% to 4% and 56%, and 1.5% to 2%
with nothing to find.

### Evaluating an output frame

After the run, pass the output frame to `po.eval`: its calls score a spec,
compare specs, or unpack the frame to long form. `po.eval.unpack(out,
"ridge")` gives one row per row and slot, with its target, `pred` and `y`,
for a `group_by` of your own.

```python
import numpy as np
import polars as pl

import polars_online as po

# Three stocks, a row a second each, y = b x0 + 0.5 x1 + noise, where each stock's b wanders.
rng = np.random.default_rng(13)
n = 20_000
x0, x1, e = rng.standard_normal((3, 3, n))
b = 1.0 + np.cumsum(rng.normal(0, 0.01, (3, n)), axis=1)
pl.DataFrame(
    {
        "t": np.tile(np.arange(n, dtype=float), 3),
        "stock": np.repeat(["A", "B", "C"], n),
        "x0": x0.ravel(),
        "x1": x1.ravel(),
        "y": (b * x0 + 0.5 * x1 + e).ravel(),
    }
).sort("t", "stock").write_parquet("stocks.parquet")
lf = pl.scan_parquet("stocks.parquet")

common = dict(targets=["y"], features=["x0", "x1"], clock="t", gap_cap=5.0, group="stock")
ridge = po.spec.ewridge("ridge", half_life=500.0, **common)
kalman = po.spec.kalman("kalman", half_life=500.0, coef_half_life=100.0, **common)  # coefficients that drift
out = lf.online.fit_predict([ridge, kalman]).collect()

# One table, many specs: which had the lower error?
for spec, n_, r2, ic, hit, mse in po.eval.compare_specs(out, ["ridge", "kalman"]).select("spec", "n", "r2", "ic", "hit_rate", "mse").iter_rows():
    print(f"{spec:7} {n_} rows: R² {r2:.4f}, IC {ic:.4f}, hit rate {hit:.4f}, mse {mse:.4f}")
# The same metrics per group, and per tumbling window of the clock.
for stock, r2 in po.eval.metrics(out, "ridge", group="stock").select("stock", "r2").iter_rows():
    print(f"ridge on {stock}: R² {r2:.4f}")
for start, r2 in po.eval.window_metrics(out, "ridge", clock="t", every=5_000.0).select("window_start", "r2").iter_rows():
    print(f"ridge from t = {start:>6.0f}: R² {r2:.4f}")
# Is kalman closer? Diebold and Mariano's t, and the evidence by betting at the last row.
dm, p = po.eval.diebold_mariano(out, a="kalman", b="ridge").select("dm", "p_value").row(0)
log_e = po.eval.seqtest(out, a="kalman", b="ridge").select(pl.col("seqtest").struct.field("log_e_a_y")).max().item()
print(f"kalman against ridge: dm {dm:.2f}; the betting evidence peaked at log_e_a {log_e:.2f} (ln 20 = 3.00)")
```

```text
ridge   59988 rows: R² 0.7899, IC 0.8889, hit rate 0.8255, mse 1.0341
kalman  59988 rows: R² 0.7922, IC 0.8901, hit rate 0.8258, mse 1.0225
ridge on A: R² 0.8089
ridge on B: R² 0.8525
ridge on C: R² 0.5585
ridge from t =      0: R² 0.6802
ridge from t =   5000: R² 0.7853
ridge from t =  10000: R² 0.8200
ridge from t =  15000: R² 0.8240
kalman against ridge: dm 8.41; the betting evidence peaked at log_e_a 9.56 (ln 20 = 3.00)
```

`kalman`, whose coefficients drift, predicts a little better than `ridge`
here, with an error of 1.0225 against 1.0341. The difference is small, but
it holds on enough rows that Diebold and Mariano's t reads 8.4. The
betting evidence peaks at 9.56, past `ln 20`, the 5% level however often
it is read. R² differs between stocks because each stock's coefficient
wandered its own way, and an R² is larger where the signal is larger
against the same noise. The first window's R² is the lowest because it
holds the fit's warm-up.

**Pass the spec as `spec=` when a target is renamed or looks ahead.** The
output frame does not record how a target was formed, so without the spec
each slot is scored against the column named after it. A target renamed
with `name=` names no column, and the call asks for `spec=`. A target
expression looking ahead is scored against a column of its own name, which
`po.stream.with_windows` makes. `metrics`, `window_metrics`, `sums` and
`unpack` take `spec=`, and `compare_specs` takes the list as `specs=`.

### Evaluating a stream too large to hold

`po.eval.metrics` needs the whole output in one frame. When the output is
too large to hold, say fifty slots over a billion rows, reduce each chunk
to ten numbers per key with `po.eval.sums`. Add the parts with
`po.eval.merge_sums`, which is exact whatever the split, and turn the total
into the metrics with `po.eval.from_sums`.

**The sums are centred, so a target far from zero keeps its variance.** Raw
`Σy` and `Σy²` would lose all of it for a target around 1e8 with unit
spread, which is what the recipe uses:

```python
import numpy as np
import polars as pl

import polars_online as po

# A target near 1e8 with unit spread, three stocks, 30,000 rows.
rng = np.random.default_rng(15)
n = 30_000
x0, x1, e = rng.standard_normal((3, n))
pl.DataFrame(
    {"stock": rng.choice(["A", "B", "C"], n), "x0": x0, "x1": x1, "y": 1e8 + 1.0 * x0 + 0.5 * x1 + e}
).write_parquet("far.parquet")
lf = pl.scan_parquet("far.parquet")

ridge = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], half_life=500.0, group="stock")

# Reduce each chunk of the output to a few sums per stock, and add the parts as they come.
running = None
for out in po.ModelBank([ridge]).fit_predict_batches(lf, chunk_size=1_000):
    part = po.eval.sums(out, "ridge", group="stock")  # weight= names a column to weight the rows by
    running = part if running is None else po.eval.merge_sums(running, part)
streamed = po.eval.from_sums(running).sort("stock")

# The same metrics from the whole output held in memory, for comparison.
held = po.eval.metrics(lf.online.fit_predict([ridge]).collect(), "ridge", group="stock").sort("stock")
for (stock, n_, r2, mse), (r2_held, mse_held) in zip(
    streamed.select("stock", "n", "r2", "mse").iter_rows(), held.select("r2", "mse").iter_rows()
):
    print(f"{stock}: {n_} rows, R² {r2:.6f} streamed and {r2_held:.6f} held, mse {mse:.6f} and {mse_held:.6f}")
```

```text
A: 10160 rows, R² 0.545017 streamed and 0.545017 held, mse 1.017160 and 1.017160
B: 9801 rows, R² 0.551971 streamed and 0.551971 held, mse 1.006657 and 1.006657
C: 10027 rows, R² 0.551899 streamed and 0.551899 held, mse 1.017914 and 1.017914
```

The streamed metrics agree with the ones computed from the whole output to
every digit printed, with only one chunk of the output in memory at a
time.

## Is it ready?

**A model should not report a number before it has seen enough data to
give one, and three gates say what "enough" means.** The gates hold back
predictions only: the model learns from every row either way.
[Warm-up](../README.md#warm-up) in the README states each gate and its
default, and [docs/WARMUP-AND-CONVERGENCE.md](WARMUP-AND-CONVERGENCE.md)
the reasoning behind them.

| gate | withholds a prediction while | field it reads | `withheld_reason` |
|---|---|---|---|
| `min_settled_frac` | the decay window is less full than this fraction of its steady state | `settled_frac = 1 − 2^(−T / half_life)`, with `T` the decay time seen | `below_min_settled_frac` |
| `min_weight` | the weight behind the state is below it | `weight_sum` | `below_min_weight` |
| `max_error_inflation` | estimation error would inflate the prediction's error over the noise floor by more than this factor | `error_inflation = sqrt(1 + edf / n_kish)` | `above_max_error_inflation` |

The reasons are listed in their order of precedence, which is the order
`withheld_reason` reports them in. `max_error_inflation` is on by default
on `ewridge` alone, at `sqrt(2)`. `emit_error_inflation` writes each row's
own factor, `error_inflation_<t>`, against the row's own features: large
for a row leaning on a direction the data never showed.

The recipe fits ten features under each gate, and reads when each first
predicts and how good its early predictions are:

```python
import numpy as np
import polars as pl

import polars_online as po

# Ten features, a row a second, y on all of them.
rng = np.random.default_rng(14)
n, k = 3_000, 10
x = rng.standard_normal((n, k))
y = x @ np.linspace(1.0, 0.1, k) + rng.standard_normal(n)
features = [f"x{j}" for j in range(k)]
pl.DataFrame({"t": np.arange(n, dtype=float), "y": y} | {f: x[:, j] for j, f in enumerate(features)}).write_parquet("ready.parquet")
lf = pl.scan_parquet("ready.parquet")

common = dict(targets=["y"], features=features, clock="t", gap_cap=5.0, half_life=300.0)
specs = [
    po.spec.ewridge("default", **common),  # max_error_inflation=sqrt(2), the one gate on by default
    po.spec.ewridge("strict", max_error_inflation=1.05, **common),  # 5% over the noise floor
    po.spec.ewridge("settled", min_settled_frac=0.5, **common),  # one half-life of history
    po.spec.ewridge("weighted", min_weight=100.0, **common),  # 100 of weight_sum
]
out = lf.online.fit_predict(specs).collect()

for spec in specs:
    rows = out.select("t", pl.col(spec["name"]).struct.unnest())
    first = rows.filter(pl.col("pred_y").is_not_null())["t"].min()
    reasons = rows["withheld_reason"].drop_nulls().value_counts().sort("withheld_reason")
    held = ", ".join(f"{reason} on {count} rows" for reason, count in reasons.iter_rows())
    mse = rows.filter(pl.col("t") < 600).select(pl.col("resid_y").pow(2).mean()).item()
    print(f"{spec['name']:9} predicts from row {first:>3.0f} ({held}); its mse before row 600: {mse:.2f}")
```

```text
default   predicts from row  12 (above_max_error_inflation on 12 rows); its mse before row 600: 1.12
strict    predicts from row 108 (above_max_error_inflation on 108 rows); its mse before row 600: 1.01
settled   predicts from row 301 (below_min_settled_frac on 301 rows); its mse before row 600: 0.99
weighted  predicts from row 114 (below_min_weight on 114 rows); its mse before row 600: 1.02
```

The default gate holds back the first 12 rows, where 11 coefficients would
otherwise be fitted on fewer rows than they number. Its predictions before
row 600 have an error of 1.12 against a noise floor of 1, because the
earliest ones still carry much estimation error. The strict gate, at 5%
over the floor, waits for 108 rows and its early error is 1.01. The other
two gates wait for one half-life of history or for 100 of weight, and
reach the same early error. Which gate to use depends on what a
withheld prediction costs. The error inflation gate reads the fit's own
size, so it moves with the number of features. `min_settled_frac` and
`min_weight` ask for a fixed amount of history.

## How wide is its error?

**A prediction is only as useful as the error it comes with, and four
diagnostics measure that error as the stream runs.** Each is read before
its row, so it can be served with the row's prediction.

| field | what it is | assumes |
|---|---|---|
| `sigma_<t>` (`emit_sigma`) | the EW standard deviation of the out-of-sample residuals, at the model's half-life | nothing, as a spread; `pred ± z · sigma` assumes Gaussian residuals |
| `zscore_<t>` (`emit_zscore`) | `resid / sigma`: how surprising the row was, in units of recent error | as `sigma` |
| `abs_resid_q<p>_<t>` (`resid_quantiles`) | the EW quantile of `\|resid\|` at level `p`, from one decaying DDSketch per slot, within 0.78% of the exact weighted quantile | no distribution |
| `lo_<t>`, `hi_<t>`, `coverage_<t>` (`conformal`) | an interval tracked to cover the share asked for, and the coverage it has delivered | no distribution |

`sigma`'s weight ages on every row the model sees, a row with no
prediction or with weight 0 included, so it forgets across a gap as the
clock says. Under a `window_size` it is the window's, as the fit is. Each
of these costs under 25 ns a row, and the ten linear models take them.

### Conformal intervals

**Use `conformal` for an interval when the residuals are not Gaussian.**
It tracks a quantile of `|resid|` directly, so its long-run coverage is the
number asked for, whatever the residuals do. Its radius `q` grows by
`conformal_rate · sigma · coverage` on a miss and shrinks by
`conformal_rate · sigma · (1 − coverage)` on a hit. That is the
quantile-tracking step of Angelopoulos, Candès and Tibshirani (2023),
scaled by `sigma` so that the rate means the same in any units. A miss is rarer than a hit by
the ratio the coverage sets, so the two steps balance exactly at the
coverage asked for. This is conformal prediction (Vovk, Gammerman and
Shafer 2005) without a held-out set, because every residual is already out
of sample.

**Under a `weight` column, each step is scaled by the row's weight over
the mean weight**, so a heavy row's miss counts for more, and the share of
rows covered can sit on either side of the number asked for. The step is
taken once per scored row, so the radius moves faster in clock time where
rows are denser. `conformal_rate` is 0.05 by default.

The recipe builds three 90% intervals on two streams, one with Gaussian
noise and one with Student's t at 3 degrees of freedom, scaled to the same
standard deviation:

```python
import numpy as np
import polars as pl

import polars_online as po

# The same relation twice, once with Gaussian noise and once with Student's t at 3 degrees of
# freedom scaled to the same standard deviation.
rng = np.random.default_rng(11)
n = 50_000
x0, x1 = rng.standard_normal((2, n))
noise = {"gauss": rng.standard_normal(n), "t3": rng.standard_t(3, n) / np.sqrt(3.0)}
pl.DataFrame({"x0": x0, "x1": x1} | {f"y_{k}": 1.0 * x0 + 0.5 * x1 + v for k, v in noise.items()}).write_parquet("width.parquet")
lf = pl.scan_parquet("width.parquet")

specs = [
    po.spec.ewridge(
        k, targets=[f"y_{k}"], features=["x0", "x1"], half_life=500.0,
        emit_sigma=True,  # sigma: for the Gaussian interval pred ± 1.645 sigma
        resid_quantiles=[0.9],  # abs_resid_q0.9: the EW 90% quantile of |resid|
        conformal=0.9,  # lo, hi: an interval tracked to cover 90% of the rows
    )
    for k in noise
]
out = lf.online.fit_predict(specs).slice(5_000).collect()

print(f"{'noise':6} {'interval':>20} {'covered':>8} {'half-width':>11}")
for k in noise:
    rows = out.select(pl.col(k).struct.unnest(), y=pl.col(f"y_{k}"))
    miss = (pl.col("y") - pl.col(f"pred_y_{k}")).abs()
    intervals = {
        "pred ± 1.645 sigma": 1.645 * pl.col(f"sigma_y_{k}"),
        "pred ± q0.9": pl.col(f"abs_resid_q0.9_y_{k}"),
        "conformal": (pl.col(f"hi_y_{k}") - pl.col(f"lo_y_{k}")) / 2,
    }
    for name, half in intervals.items():
        covered, width = rows.select(covered=(miss <= half).mean(), width=half.mean()).row(0)
        print(f"{k:6} {name:>20} {covered:>8.3f} {width:>11.3f}")
```

```text
noise              interval  covered  half-width
gauss    pred ± 1.645 sigma    0.899       1.642
gauss           pred ± q0.9    0.900       1.647
gauss             conformal    0.900       1.650
t3       pred ± 1.645 sigma    0.928       1.585
t3              pred ± q0.9    0.900       1.359
t3                conformal    0.900       1.364
```

On Gaussian noise the three intervals are the same: each covers 90% of the
rows at the same width. On Student's t, the Gaussian interval covers 92.8%,
more than asked, because `sigma` is inflated by the rare huge residuals,
and it is wider than it needs to be for most rows. The quantile and the
conformal interval cover 90.0% at a half-width of 1.36, against the
Gaussian interval's 1.59. On three streams of 200,000 rows with Student's
t at 2.5 degrees of freedom, a scale mixture and a regime shift, the
Gaussian interval covered 94% to 95%, and `conformal` stayed within 1
point of 90%.

**What to do.** Serve `lo` and `hi` from `conformal` when the coverage
must hold, and `sigma` when a scale is enough. A `zscore` beyond 3 is a
surprise on Gaussian residuals and an ordinary row on heavy-tailed ones,
so read it against [the tails](#are-its-tails-heavy).

## A worked example: the two-stage reversion beta

**A two-stage fit is checked stage by stage, with every diagnostic on this
page that its stages ask for.** The example estimates how fast a price
reverts to its fair value. Stage 1 fits the fair value: each series less
its own 10-minute EWMA, with the mid's regressed on the factors'. Its
out-of-sample residual, `resid_A`, is the mid less its fair value, known
at the row. Stage 2 regresses the mid's move over the next 150 seconds on
`resid_A`, and its coefficient is the reversion beta: the share of the gap
to fair value that closes.

The stream is generated with the answer planted. The mid is an efficient
price, a random walk the factors follow, plus a transient `u` that halves
every 30 seconds and that no factor sees. Over a 150-second window
weighted at a 30-second half-life, `u`'s expected share that reverts is
0.490, so the beta on `u` itself is −0.490.

```python
from datetime import datetime, timedelta

import numpy as np
import polars as pl

import polars_online as po

# A quote a second for 17 hours. mid = p + u: p is a random walk (the efficient price) and u a
# transient that halves every 30 s. Three factors s1-s3 follow p, each with noise of its own; none
# of them sees u.
rng = np.random.default_rng(12)
n = 60_000
p = 100 + np.cumsum(rng.normal(0, 0.01, n))
phi = 0.5 ** (1 / 30)
u = np.zeros(n)
shocks = rng.normal(0, 0.05 * np.sqrt(1 - phi**2), n)
for i in range(1, n):
    u[i] = phi * u[i - 1] + shocks[i]
s = np.array([1.0, 0.8, 1.2])[:, None] * p + np.cumsum(rng.normal(0, 0.002, (3, n)), axis=1)
ts = [datetime(2026, 1, 5) + timedelta(seconds=i) for i in range(n)]
pl.DataFrame({"ts": ts, "mid": p + u, "s1": s[0], "s2": s[1], "s3": s[2]}).write_parquet("quotes.parquet")

# Stage 1, fair value: each series less its own 10-minute EWMA, and A (the mid's) regressed on the
# factors'. Its out-of-sample residual resid_A is the mid less its fair value, known at the row.
clock = dict(clock="ts", gap_cap="5m")
back = dict(half_life="10m", window_size="60m")
signals = pl.scan_parquet("quotes.parquet").online.with_windows(
    A=pl.col("mid") - po.ewm_mean("mid", **back),
    x1=pl.col("s1") - po.ewm_mean("s1", **back),
    x2=pl.col("s2") - po.ewm_mean("s2", **back),
    x3=pl.col("s3") - po.ewm_mean("s3", **back),
    **clock,
)
fair = po.spec.ewridge("fair", targets=["A"], features=["x1", "x2", "x3"], half_life="2h", **clock)

# Stage 2, reversion: the next 150 s's 30-second EWMA of the mid, less the mid, regressed on
# resid_A. Each row is learned once its window has closed (embargo), and its residuals overlap.
fwd = (po.rewm_mean("mid", half_life="30s", window_size="150s") - pl.col("mid")).alias("B")
stage2 = dict(
    targets=[fwd], half_life=float("inf"), embargo="150s", min_weight=1_000.0,
    emit_se_coef=True, emit_robust_se=True, robust_se_lags=300,  # Newey-West to twice the window
    emit_calibration=True,  # B on pred: a slope of 1 says the beta needs no rescaling
    **clock,
)
revert = po.spec.ewridge("revert", features=["resid_A"], **stage2)
joint = po.spec.ewridge("joint", features=["A", "x1", "x2", "x3"], **stage2)  # one stage in place of two
both = po.spec.ewridge("both", features=["resid_A", "x1", "x2", "x3"], **stage2)  # do the factors add?

out = (
    signals.online.fit_predict([fair])
    .online.unnest([fair])  # pred_A, resid_A
    .online.fit_predict([revert, joint, both])
    .collect()
)

for spec in [revert, joint, both]:
    name = spec["name"]
    last = out.select(pl.col(name).struct.unnest()).filter(pl.col("se_coef").is_not_null()).tail(1)
    coef, se, hac = (last[f].item().to_list() for f in ["coef", "se_coef", "se_coef_hac"])
    slope, wald = last.select("calibration_slope_B", "calibration_wald_B").row(0)
    print(f"{name}: calibration slope {slope:.2f}, wald {wald:.1f}")
    for term, b, s_, h in zip(spec["features"], coef[1:], se[1:], hac[1:]):
        print(f"  {term:8} coef {b:+.3f}, t {b / s_:+7.1f} by se_coef and {b / h:+6.1f} by se_coef_hac")

# The planted beta: the forward EWMA of u over its 150 s window, as a share of u, less 1.
j = np.arange(1, 151)
w = 0.5 ** (j / 30)
print(f"planted beta on u itself: {(w * phi**j).sum() / w.sum() - 1:.3f}")
```

```text
revert: calibration slope 0.99, wald 72.7
  resid_A  coef -0.450, t  -107.7 by se_coef and  -15.8 by se_coef_hac
joint: calibration slope 0.80, wald 616.4
  A        coef -0.444, t  -105.4 by se_coef and  -15.5 by se_coef_hac
  x1       coef +0.124, t   +27.7 by se_coef and   +3.0 by se_coef_hac
  x2       coef -0.011, t    -2.0 by se_coef and   -0.2 by se_coef_hac
  x3       coef +0.258, t   +57.3 by se_coef and   +6.1 by se_coef_hac
both: calibration slope 0.84, wald 348.4
  resid_A  coef -0.446, t  -105.2 by se_coef and  -15.6 by se_coef_hac
  x1       coef +0.029, t    +6.8 by se_coef and   +0.7 by se_coef_hac
  x2       coef -0.083, t   -15.8 by se_coef and   -1.7 by se_coef_hac
  x3       coef +0.021, t    +5.1 by se_coef and   +0.5 by se_coef_hac
planted beta on u itself: -0.490
```

Each check reads one part of the answer:

| check | what it read | what it says |
|---|---|---|
| the beta | `revert`'s coefficient on `resid_A`, −0.450 | close to the −0.490 planted on `u`. `resid_A` is `u` plus the fair value's own error, which attenuates the beta toward zero, as noise in a regressor does |
| calibration | `revert`'s slope 0.99 | the predictions `−0.450 · resid_A` are scaled right, so the beta needs no rescaling |
| the Wald beside it | 72.7 | inflated by the overlap: 150 rows share each window, and the calibration has no Newey-West form. Read the slope |
| Newey-West | t of −107.7 by `se_coef`, −15.8 by `se_coef_hac` | the beta is real either way, but the plain t is 6.8 times too large: each residual shares its 150-second window with 149 others |
| `joint` | one stage, `A` and the factors together: `A`'s coefficient −0.444, calibration slope 0.80 | `A`'s coefficient matches `resid_A`'s, since the factors absorb the fair value. Its predictions are 20% too large, so the one-stage fit needs rescaling where the two-stage does not |
| `both` | `resid_A` beside the factors: the factors' t are +6.8, −15.8 and +5.1 by `se_coef`, and +0.7, −1.7 and +0.5 by `se_coef_hac` | once `resid_A` is in, the factors add nothing that Newey-West believes. By `se_coef` alone, all three would look significant |

**What the example shows.** The two-stage beta is right, and its
calibration says so. Every t on a look-ahead target needs `se_coef_hac`:
`se_coef` would have kept three factors that carry nothing.

## Data whose truth is known

To test a model that claims to find a changing correlation structure, fit
it on the `rows` of
[`po.sim.regimes`](https://hgilde.github.io/polars-online/sim.html#polars_online.sim.regimes)
and compare what it finds with `truth_rows` and `truth_blocks`, the truth
returned beside them. The measurements in [docs/REGIMES.md](REGIMES.md)
show what `hmm`, `corrchange` and `bocpd` find on such streams, and what
they miss.

```python
import polars_online as po

sim = po.sim.regimes(
    4, states=[0.2, 0.7],                     # four series; two regimes, at these equicorrelations
    transition=[[0.98, 0.02], [0.02, 0.98]],  # how the regimes switch
    n_blocks=8, rows_per_block=500,
    durations=None,                           # or a block count per state: each state lasts exactly as long as it says
    design="step", smooth_rows=0,             # or "smooth": interpolate the matrix over smooth_rows rows at a boundary
    phi=0.3, noise=0.01,                      # returns correlated with their own past; observation noise
    async_rates=[1.0, 1.0, 0.4, 0.4],         # two series observed less often; a series with no observation is null
    seed=0,                                   # the same seed twice is byte-identical, under one numpy version
)
rows, truth_rows, truth_blocks = sim["rows"], sim["truth_rows"], sim["truth_blocks"]
for name, frame in [("rows", rows), ("truth_rows", truth_rows), ("truth_blocks", truth_blocks)]:
    print(f"{name}: {frame.height} rows; {', '.join(frame.columns)}")
```

```text
rows: 4000 rows; entity, t, clock, session, x_1, x_2, x_3, x_4, activity
truth_rows: 4000 rows; t, block, state, scale_mult, mix
truth_blocks: 8 rows; block, state, n_rows, corr
```

| frame | what it holds |
|---|---|
| `rows` | what a consumer sees: an entity, the row's index `t`, a clock and a session, levels `x_1` .. `x_m`, and `activity`, null unless `activity=(mean, shape)` is given. For returns, `unpivot` the levels into the long input of [`refresh_time`](../README.md#series-that-tick-at-their-own-times), and apply `.diff()` to the grid it builds |
| `truth_rows` | per row `t`: the block, state, volatility multiplier and interpolation fraction |
| `truth_blocks` | each block's true correlation matrix, as the upper triangle in a list |

## Sources

Each claim above that is not measured here rests on one of these.

| source | for |
|---|---|
| Angelopoulos, A. N., Candès, E. J. and Tibshirani, R. J. (2023). Conformal PID control for time series prediction. *Advances in Neural Information Processing Systems* 36. | the quantile-tracking step of `conformal` |
| Åström, K. J. and Wittenmark, B. (1995). *Adaptive Control*, 2nd ed. Addison-Wesley. | estimator windup under exponential forgetting |
| Bartlett, M. S. (1935). Some aspects of the time-correlation problem in regard to tests of significance. *Journal of the Royal Statistical Society* 98(3), 536–543. | the count correction behind `marginal`'s `t_serial` |
| Belsley, D. A., Kuh, E. and Welsch, R. E. (1980). *Regression Diagnostics*. Wiley. | DFFITS, the basis of `influence` |
| Breusch, T. S. and Pagan, A. R. (1979). A simple test for heteroscedasticity and random coefficient variation. *Econometrica* 47(5), 1287–1294. | `breusch_pagan` |
| Brown, R. L., Durbin, J. and Evans, J. M. (1975). Techniques for testing the constancy of regression relationships over time. *Journal of the Royal Statistical Society, Series B* 37(2), 149–192. | recursive residuals, the CUSUM and the CUSUM of squares, and their boundaries |
| Chu, C.-S. J., Hornik, K. and Kuan, C.-M. (1995). MOSUM tests for parameter constancy. *Biometrika* 82(3), 603–617. | the moving-sum form of the CUSUM |
| Clark, T. E. and McCracken, M. W. (2001). Tests of equal forecast accuracy and encompassing for nested models. *Journal of Econometrics* 105(1), 85–110. | why equal-accuracy tests lean toward the smaller of two nested models |
| Clark, T. E. and West, K. D. (2007). Approximately normal tests for equal predictive accuracy in nested models. *Journal of Econometrics* 138(1), 291–311. | `po.eval.clark_west` |
| Dawid, A. P. (1984). Statistical theory: the prequential approach. *Journal of the Royal Statistical Society, Series A* 147(2), 278–292. | prequential evaluation |
| Diebold, F. X. and Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business and Economic Statistics* 13(3), 253–263. | `po.eval.diebold_mariano` |
| Hinkley, D. V. (1971). Inference about the change-point from cumulative sum tests. *Biometrika* 58(3), 509–523. | the Page-Hinkley detector behind `drift` |
| Jarque, C. M. and Bera, A. K. (1980). Efficient tests for normality, homoscedasticity and serial independence of regression residuals. *Economics Letters* 6(3), 255–259. | `jarque_bera` |
| Kish, L. (1965). *Survey Sampling*. Wiley. | the effective sample size of unequal weights |
| Koenker, R. (1981). A note on studentizing a test for heteroscedasticity. *Journal of Econometrics* 17(1), 107–112. | the `n R²` form of `breusch_pagan` |
| Ljung, G. M. and Box, G. E. P. (1978). On a measure of lack of fit in time series models. *Biometrika* 65(2), 297–303. | `ljung_box` |
| Mincer, J. A. and Zarnowitz, V. (1969). The evaluation of economic forecasts. In *Economic Forecasts and Expectations*, ed. J. A. Mincer, 3–46. NBER. | the calibration regression |
| Newey, W. K. and West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica* 55(3), 703–708. | `se_coef_hac`, and the variance in `diebold_mariano` and `clark_west` |
| Page, E. S. (1954). Continuous inspection schemes. *Biometrika* 41(1/2), 100–115. | the CUSUM detector behind `drift` |
| Ramsey, J. B. (1969). Tests for specification errors in classical linear least-squares regression analysis. *Journal of the Royal Statistical Society, Series B* 31(2), 350–371. | `reset` |
| Vovk, V., Gammerman, A. and Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer. | conformal prediction |
| White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity. *Econometrica* 48(4), 817–838. | `se_coef_hc0` |
