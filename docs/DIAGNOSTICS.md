# Is the model working?

This reference page lists, for anyone who reads a model's output and
wants to know whether its predictions can be trusted, each question, the
diagnostic that answers it and its fields. Each diagnostic is a switch in
the spec that adds fields to the model's output on every row, or a call in
`po.eval` that scores an output after the run.

[Data quality](DATA-QUALITY.md) asks whether the data can be
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

# A ridge with one diagnostic for each of the three questions.
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
# R² just before the flip, and as the fit relearns after it.
r2 = out.filter(pl.col("t").is_in([2_999.0, 3_100.0, 3_300.0, 5_999.0]))["r2_y"]
print("r2_y at rows 2,999, 3,100, 3,300 and 5,999: " + ", ".join(f"{v:.2f}" for v in r2))
```

```text
flag                      rows 1,000-2,999  rows 3,000-5,999  first row after
calibration_wald > 5.99               0.0%              0.0%                -
|cusum| > 3                           0.0%              1.8%             3052
|cusum_sq| > 3                        0.1%              2.8%             3051
break_wald > 21.1                     0.0%             76.8%             3059
r2_y at rows 2,999, 3,100, 3,300 and 5,999: 0.51, 0.43, 0.44, 0.55
```

`break_wald` sees the flip 59 rows after it, and it stays past its
threshold on 77% of the rows after. It is the Wald distance between two
fits of the same regression, one at the diagnostic's memory and one at four
times it, and it grows when a coefficient moves. The two CUSUMs watch the
residuals' mean and spread. After the flip they pass 3 on 1.8% and 2.8% of
the rows, near the 0.3% they pass on a stream with no break. A
slope that flips on a feature centered at zero leaves the residuals' mean at
zero, so the CUSUM barely sees it, and the spread grows too little here for
the CUSUM of squares. The calibration
stays quiet because the fit forgets: within a few half-lives it has learned
the new slope, and its predictions are scaled right again. The running R²
falls from 0.51 before the flip to 0.43 a hundred rows after, and is back
to 0.55 by the end. Each question below has a section of its own: which
diagnostic answers it, what that diagnostic cannot see, and a link to the
how-to page whose recipe runs it.

**Every recipe on this page, and on the how-to pages it links, is a
complete program.** Each recipe generates its data
with the answer planted, saves it to parquet and reads it back with
`pl.scan_parquet`, as a stream too large to hold would be read. Its output
is shown below it. `tests/test_diagnostics_page.py` runs this page's
recipes, and `tests/test_kind_pages.py` runs the how-to pages'. Each holds
its output to what is shown, so none goes stale. A number past a
million in an output is a model that has blown up. Its digits depend on
the platform's rounding, so the test holds such a number only to being past
a million. Warnings that a recipe raises, such as a `ReadinessWarning`, go
to the standard error stream and are not shown.

These are the words of the library that the page uses:

| word | what it means here |
|---|---|
| spec | one model's description: the model, the columns it reads, and how it treats time. `po.spec.ewridge(...)` makes one, and [the README's model table](MODELS.md#the-models-at-a-glance) lists every kind |
| bank | `po.ModelBank`, a set of specs fitted together over the same rows |
| slot | one prediction of one target, at one point of a grid such as a list of ridge values. A diagnostic writes one field per slot, suffixed with the target, as `cusum_y` |
| out-of-sample | a row's prediction is made before the model learns from the row, so its residual `resid = y − pred` is one the model has not seen |
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
| what is it missing? | `emit_specification`, `emit_autocorr` | `ljung_box_<t>`, `breusch_pagan_<t>`, `ramsey_reset_<t>`, `autocorr_<t>` | [What is it missing?](#what-is-it-missing) |
| are its tails heavy? | `emit_tails` | `skew_<t>`, `kurtosis_<t>`, `jarque_bera_<t>` | [Are its tails heavy?](#are-its-tails-heavy) |
| which row moved it? | `emit_influence` | `influence_<t>` | [Which row moved it?](#which-row-moved-it) |
| are its features healthy? | `emit_feature_health` | `spread_ratio_<feature>`, `mean_shift_<feature>` | [Are its features healthy?](#are-its-features-healthy) |
| is another model better? | `emit_metrics`, `emit_selected`, `emit_averaged`; `po.eval` | `ic_<t>`, `r2_<t>`, `hit_rate_<t>`, `selected_<t>`, `pred_<t>__selected`, `pred_<t>__averaged` | [Is another model better?](#is-another-model-better) |
| is it ready? | `min_weight`, `min_settled_frac`, `max_error_inflation`, `emit_error_inflation` | `weight_sum`, `settled_frac`, `withheld_reason`, `error_inflation_<t>` | [Is it ready?](#is-it-ready) |
| how wide is its error? | `emit_sigma`, `emit_zscore`, `resid_quantiles`, `conformal` | `sigma_<t>`, `zscore_<t>`, `abs_resid_q<p>_<t>`, `lo_<t>`, `hi_<t>`, `coverage_<t>` | [How wide is its error?](#how-wide-is-its-error) |

[Check a two-stage fit](how-to/check-a-two-stage-fit.md)
checks a two-stage fit stage by stage with these diagnostics, and [Test a
detector on known truth](how-to/test-a-detector-on-known-truth.md) scores a
regime or break detector against streams whose truth is known.

## How the diagnostics work

[How the diagnostics work](explanation/how-diagnostics-work.md) explains
why each diagnostic keeps a memory of its own, when it starts to read, how
to read a statistic that is read on every row, and how a fit built in
stages is checked stage by stage.

### Out-of-sample, by construction

**Every residual a diagnostic reads was made before the model learned from
its row.** A bank predicts a row, records the residual, and only then
learns the row. So a diagnostic built from those residuals is
out-of-sample, as Dawid's (1984) evaluation is. It judges the predictions the
model actually made, in the order it made them, on targets it had not yet
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
(`tests/test_second_opinion_diagnostics.py`).

### One memory per diagnostic: run once or windowed

**Each diagnostic keeps one accumulator per slot, with a half-life of its
own.** Its keyword is the switch's name with `_half_life`, as
`calibration_half_life`. Left out, it is the fit's own memory, except for
the calibration, whose memory is four times it. `inf` runs the diagnostic
once over the whole stream.

**The fit's own memory is its half-life, but for two fits that forget on
another clock.** Under `window_size`, it is the half-life whose
exponential weights have the window's Kish size. A window of `W` rows is
worth `W` rows, and weights `λ^i` are worth `(1 + λ)/(1 − λ)`. So `λ = (W −
1)/(W + 1)`, and the half-life is `ln 2 / ln((W + 1)/(W − 1))` rows, about
`W / 2.885`. On a clock column the window is in clock units, and the
half-life is `W ln 2 / 2` of them. A window beside a half-life takes the
Kish size of the decayed weights cut at the window. `kalman` forgets its
coefficients at `coef_half_life`, so its diagnostics read at that, the
shortest finite one where it gives one per coefficient; the CUSUM's null
share alone reads the intercept's entry, since the intercept is what
absorbs a level. Under `window_size` that share reads the window itself
([Has it broken?](#has-it-broken)).

| form | memory | what it is | use it beside |
|---|---|---|---|
| run once | `inf` | the classical test, with its critical values, at `n` rows | a fit that does not forget (`half_life=inf`) |
| windowed | finite | the same statistic over exponentially weighted moments, read at the Kish size `n_kish` in place of `n` | a fit that forgets |

Why a run-once test needs a fit that does not forget, why the
calibration's memory is four times the fit's, and what a shorter memory
trades are in [Run once or
windowed](explanation/how-diagnostics-work.md#run-once-or-windowed).

**A diagnostic folds a row only once the model is ready.** A row whose
prediction a gate withheld has no residual, so `min_weight`,
`max_error_inflation` and `min_settled_frac` keep it out. Beside a fit that
forgets, a diagnostic also waits until the stream has come 95% of the way to its steady state on
the longest finite memory the model keeps, 4.3 of that memory's half-lives.
Under `window_size` with no decay that memory is the window's Kish
half-life, so a diagnostic starts about `1.5 W` rows in. A fit with several
memories reaches its steady state only once the longest has, so six models count a
memory beside their half-life:

| model | the memory it counts beside `half_life` |
|---|---|
| `holt` | its trend's, `trend_half_life`, four times its level's by default |
| `kalman` | `coef_half_life` |
| `ewridge` | `long_half_life`, when given |
| `lasso` | `select_half_life`, when given |
| `huber`, `sgd` | the Huber scale's `huber_scale_half_life`, when given |

`sgd` and `pa` under `standardize` fold from the row their scaler
switches the fit to the caller's units, 22 rows of Kish's size in. The
fields of the twin fits and of feature health read the target and the
features alone, and fold every row.

**The readiness warnings read the model's own half-life, not the longest
memory.** `settled_frac`, `min_settled_frac`, `summary()` and the readiness
warnings that a gate is not met read the decay `half_life` sets, so they can judge
a stream long before its diagnostics start. Under `window_size` with no
decay there is no steady state to approach, so `settled_frac` is null and no readiness warning
judges the stream. Its diagnostics still start 4.3 of the hard window's Kish half-lives
in.

**Run once, beside a fit that forgets nothing, give the model a
`min_weight` of a few rows per coefficient,** since only the gates hold
back the warm-up there. Why a diagnostic waits for the longest memory,
and what a fit far from its truth or with no `min_weight` does to the
statistics, are in [When a diagnostic starts to
read](explanation/how-diagnostics-work.md#when-a-diagnostic-starts-to-read).
How to read a statistic that is read on every row is in [Reading a
statistic that is read on every
row](explanation/how-diagnostics-work.md#reading-a-statistic-that-is-read-on-every-row).

### Every switch at once

Every switch adds its fields to the spec's output struct, and one spec can
carry them all. A switch's tuning keywords sit beside it, each at its
default if it has one:

```python
import numpy as np
import polars as pl

import polars_online as po

# Two thousand rows of y = x0 + 0.5 x1 + noise.
rng = np.random.default_rng(16)
n = 2_000
x0, x1 = rng.standard_normal((2, n))
pl.DataFrame(
    {"t": np.arange(n, dtype=float), "x0": x0, "x1": x1, "y": x0 + 0.5 * x1 + rng.standard_normal(n)}
).write_parquet("every.parquet")
lf = pl.scan_parquet("every.parquet")

# One spec with every switch on.
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
    calibration_half_life=800.0,  #   the default: four times the fit's memory, here its half-life
    emit_breaks=True,             # studentized_y, cusum_y, cusum_sq_y, break_wald_y
    breaks_half_life=200.0,       #   the default: the fit's memory, as for each below
    emit_specification=True,      # ljung_box_y, breusch_pagan_y, ramsey_reset_y
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
# Fit, and list every field the switches added, six to a line.
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
ramsey_reset_y skew_y kurtosis_y jarque_bera_y influence_y spread_ratio_x0
spread_ratio_x1 mean_shift_x0 mean_shift_x1 weight_sum settled_frac withheld_reason
coef_y_intercept coef_y_x0 coef_y_x1 support_coef_y_intercept support_coef_y_x0 support_coef_y_x1
se_coef_y_intercept se_coef_y_x0 se_coef_y_x1 se_coef_hc0_y_intercept se_coef_hc0_y_x0 se_coef_hc0_y_x1
se_coef_hac_y_intercept se_coef_hac_y_x0 se_coef_hac_y_x1 scored_clock learned_clock
```

### What each costs, and which models take it

**Most diagnostics add a few nanoseconds a row, and the ones that solve
or keep a matrix add microseconds.** Each was timed alone, added to an
`ewridge` with one slot, over 200,000 rows with 5 features and 100,000 rows
with 20, in one chunk, the best of five runs, on 2026-10-09. The machine
was shared, at a load average near 3.3, and the noise of a measurement is
about ±10 ns. The fit alone and the two switches whose paths changed since,
`emit_feature_health` and `emit_specification`, were timed again on
2026-10-24 for 1.3.0 the same way, at a load of 1.9 to 3.4: the fit alone
took 191 ns a row at 5 features and 326 ns at 20.

| switch | added per row, 5 features | 20 features | grows with | models |
|---|---:|---:|---|---|
| `emit_sigma`, `emit_zscore`, `emit_autocorr`, `conformal`, `emit_drift` | under 10 ns | under 10 ns | nothing | the ten linear models |
| `emit_metrics` | 8 ns | 15 ns | nothing | the ten linear models |
| `resid_quantiles` (two levels) | 14 ns | 24 ns | nothing | the ten linear models |
| `emit_se_coef` | 10 ns | 19 ns | `k²`, on `coef`'s rows only | `ewridge`, `rls`, `kalman` |
| `emit_calibration` | 20 ns | 26 ns | nothing | the ten linear models |
| `emit_feature_health` | 31 ns | 75 ns | `k` | the nine linear models with features |
| `emit_error_inflation` | 125 ns | 334 ns | `k²`: a triangular solve | `ewridge`, `rls`, `kalman` |
| `emit_influence` | 145 ns | 349 ns | `k²`, the same solve | `ewridge`, `rls`, `kalman` |
| `emit_tails` | 157 ns | 355 ns | `k²`, the same solve | the ten linear models |
| `emit_robust_se` (10 lags) | 350 ns | 3,266 ns | `L × k²` for `L` lags | `ewridge`, `rls` |
| `emit_specification` | 737 ns | 1,898 ns | `k²`, and the lags | the ten linear models |
| `emit_breaks` | 903 ns | 4,567 ns | `k²`: two fits and their distance | the ten linear models |

The ten linear models are `ewridge`, `rls`, `lasso`, `kalman`, `huber`,
`quantile`, `sgd`, `pa`, `ftrl` and `holt`. Any other model refuses a
residual switch by name, and a model outside a switch's list refuses that
switch and names the models that take it.

## Is it scaled right?

**A prediction can be correlated with the target and still be the wrong
size, and the calibration measures by how much.** Mincer and Zarnowitz
(1969) regress the target on the prediction, `y = a + b · pred`. A
prediction that is scaled right has `b = 1` and `a = 0`. A slope above 1
says the predictions are too small: a ridge shrank them, or a learning rate
was too slow. A slope below 1 says they are too large: a fit with too many
coefficients for its memory follows its own noise.

| | `emit_calibration` |
|---|---|
| fields | `calibration_slope_<t>`, `calibration_intercept_<t>`, `calibration_wald_<t>`, read before the row |
| update | exponentially weighted means of `pred`, `y`, `pred²` and `pred · y` at `calibration_half_life`; `b = cov(pred, y) / var(pred)`, `a = ȳ − b · p̄` |
| statistic | `wald = (n_kish − 2) · ((ȳ − p̄)² + (b − 1)² · var(pred)) / s²`, with `s²` the regression's residual mean square: `chi2(2)` when `a = 0` and `b = 1` |
| memory | four times the fit's memory; `inf` runs it once, where `wald / 2` is the least-squares F statistic, `F(2, n − 2)` |
| threshold | `wald > 5.99`, `chi2(2)`'s 5% value. Run once on calibrated fits it passed on 5.0% to 7.0% of 400 streams; at the default memory, on 0% to 0.3% of rows |
| cost, models | 20 ns a row; the ten linear models |

[Check the scale of a model's predictions](how-to/check-the-scale.md) reads
the calibration, rescales a prediction by it, and fixes a fit of the wrong
size.

**On a target that looks ahead, the Wald statistic takes Newey and West's
variance.** Targets that overlap correlate the residuals, as [Can its t be
trusted?](#can-its-t-be-trusted) explains. Under a horizon of `h` rows the
test of `a = 0, b = 1` is Wald's with the regression's sandwich at `2h`
lags, built from the residual under the null, `y − pred`. On the two-stage
example's look-ahead target, a slope of 0.99 came with a `wald` of 1.1
under `horizon_rows=150`, consistent with `a = 0, b = 1` ([Check each
stage with the diagnostics it asks
for](how-to/check-a-two-stage-fit.md#check-each-stage-with-the-diagnostics-it-asks-for)).

**A quantile fit's calibration is its coverage.** A prediction of the
`q`-th quantile is calibrated when the target falls below it on a share
`q` of rows. Mincer and Zarnowitz's regression asks whether it is the
mean, and on a stable fit at `q = 0.9` it passed its 5% value on every
row. So on `quantile`, and on `sgd` under `loss="quantile"`,
`emit_calibration` writes two other fields:

| field | what it is |
|---|---|
| `calibration_coverage_<t>` | the EW share of rows below the prediction |
| `calibration_wald_<t>` | the binomial test of that share at Kish's size, `(Σω d)² / (q (1 − q) Σω²)` with `d = 1{y < pred} − q`: `chi2(1)` at a calibrated quantile, with Newey and West's lag products under a horizon |

On 30 clean streams the coverage's median read 0.499 to 0.500 at
`q = 0.5`, and 0.895 to 0.900 at `q = 0.9`. At the default memory the test
passed 3.84 on 0% to 1.7% of rows. It is conservative, as the slope's test
is beside a fit that absorbs what it reads. `cusum` sums
`d / sqrt(q (1 − q))` there in place of the studentized residual. The
studentized residual flagged every row at `q = 0.9`, because a quantile's
residuals do not have a zero mean.

**It cannot see a fit that is wrong but calibrated.** A model missing a
feature still predicts `E[y | its features]`, whose calibration slope is 1.
A low R² with a slope of 1 is a model with too little information, which
no rescaling repairs.

## Has it broken?

**A relationship can break in three ways, and each diagnostic sees some of
them.** The intercept can shift, which gives the residuals a mean. The
noise can grow, which changes the residuals' spread. Or a slope can move,
which on a feature centered at zero leaves the residuals' mean at zero and
widens their spread only until the fit catches up.

| field | what it is | sees | with no break |
|---|---|---|---|
| `studentized_<t>` | the row's recursive residual over the spread of those before it, `z = (resid / error_inflation) / s`, read once that spread has 10 rows of Kish size | each row's surprise | about `N(0, 1)` |
| `cusum_<t>` | the weighted sum of the studentized residuals before the row, standardized: `Σωz / sqrt(r Σω²)` | a mean in the residuals: an intercept shift | `N(0, 1)` |
| `cusum_sq_<t>` | the weighted sum of `z² − 1`, standardized by its measured spread: `Σω(z² − 1) / sqrt(r₂ (m₄ − 1) Σω²)`, `m₄` the mean of `z⁴` at four times the memory | a change in the residuals' spread | `N(0, 1)`, whatever the tails |
| `break_wald_<t>` | the Wald distance between two least-squares fits of the target on the slot's features, one at the memory and one at four times it, over the exact variance of their difference; null run once | a coefficient that moved, the slope's or the intercept's | `chi2(k)`, with `k` the coefficients |
| `drift_<t>` | true on the row a Page-Hinkley detector (Page 1954; Hinkley 1971) on `\|resid\| / sigma` climbs `drift_threshold` above its lowest point | a rise in the residuals' level, beyond `drift_delta` sigmas | false |

`ω` is each row's weight times its decay at `breaks_half_life`, the
fit's memory by default. `r` and `r₂` are 1 run once, and windowed they
are the nulls' shares that *A windowed CUSUM is over its own null* derives.

**Run once, the CUSUM is Brown, Durbin and Evans' test.** With unit
weights `Σω²` is the row count `r`, so `cusum · sqrt(r)` is their CUSUM
path, read against the 5% boundary `±0.948 (sqrt(T) + 2r / sqrt(T))` over
a run of `T` rows. `T` is the length of the run being tested, so a monitor
fixes it in advance as the horizon it will watch. On 1,000 streams of
3,000 rows with an intercept break of half a noise sd at row 1,500, the
CUSUM crossed on every one, 236 rows after the break on the median, and
statsmodels' own crossed 241 rows after. On the same streams with no
break, both crossed on 6.0% of them, within about one standard error, 0.75
points, of the boundary's 5%. `scripts/diagnostics_rates.py run_once`
makes these streams and counts the crossings.

**The CUSUM of squares is not Brown, Durbin and Evans' CUSUM of squares.**
Theirs, `s_r`, is the share of a whole run's squares that fell in its
first `r` rows, read against a boundary. This one is the sum of `z² − 1`
over its own standard deviation, which reads the spread the residuals have
drifted to from the one their studentizing scale held. It divides by the
measured `E[z⁴] − 1`, the variance of `z²`, and not by a Gaussian's 2, so
it holds its rate on heavy tails. Each cell below is the share past 1.96,
of rows windowed, and of 40 streams' last rows run once:

| noise | over 2, windowed | over `E[z⁴] − 1`, windowed | over 2, run once | over `E[z⁴] − 1`, run once |
|---|---:|---:|---:|---:|
| Student's t, 3 to 5 degrees of freedom | 11% to 33% | 4.7% to 5.5% | 28% to 53% | 2.5% to 7.5% |
| Gaussian | | 6.3% | | 2.5% |

Run once its mean sits a little above 0, +0.1 to +0.4 at the
last of 6,000 rows, since the first rows' `z` are read against a young
scale and spread wider than 1. The fourth moment is read at four times the
memory, so that a change of spread does not hide itself in it at once.

**With a memory, the CUSUM is a moving sum**, as Chu, Hornik and Kuan's
(1995) MOSUM is, with an exponential window in place of a rectangular one.

**A windowed CUSUM is over its own null, since the fit absorbs part of
every level it sums.** A fit that forgets is an EW mean of the shocks
before each row, so its residual is the shock less that mean. A sum of
those residuals at a memory `h` therefore cancels much of itself. In
continuous time the sum's variance is `r = h_fit / (h_fit + h)` of the
plain sum's, whatever the rows' rate: one half at equal memories, a
standard deviation of 0.71. `h_fit` is the memory at which the fit absorbs
a level, the fit's own, but for a `kalman` whose `coef_half_life` gives
each coefficient its own: a level is absorbed by the intercept's
coefficient alone, so `h_fit` is the intercept's entry, and `inf` there
pins it, absorbs nothing, and leaves the plain sum. Read at the shortest
entry instead, `[inf, 30, 30]` spread 1.40 on clean streams and `[30, 10,
10]` 1.29, where `[30, 30, 30]` read 1.03; at the intercept's each reads
about 1.03. The CUSUM of squares is studentized by a scale
at its own memory, which absorbs the squares' level the same way, so its
share is one half. Each is divided by its share, as far as its sums have
filled (*A young sum's share is nearer 1*). On 30 clean streams beside `ewridge`, `rls`, `lasso` and
`kalman` at a half-life of 200, at memories of 50 to 800, from each
stream's readiness on, the spreads read:

| | not divided | divided by its share |
|---|---:|---:|
| `cusum` | 0.48 to 0.94 | 1.03 to 1.06 |
| `cusum_sq` | 0.72 to 0.79 | 0.98 to 1.26 |

The widest reads of the CUSUM of squares come from a few of its first
values, whose fourth moment is read off a handful of rows.
`scripts/diagnostics_rates.py memories` makes these streams.

**A young sum's share is nearer 1.** The share that *A windowed CUSUM is
over its own null* derives is that of sums which have filled their memory.
The diagnostics start once the stream is 95% of the way to its steady
state, so the fit's residuals are steady from the first row. But a sum of a few
of them holds little of what the fit's mean has spread a level over, and
its variance is nearly the plain sum's. Over sums that have run for a time
`T`, `x = T ln 2 / h`, the CUSUM's share beside an exponential fit is

```text
r(x) = h_fit / (h_fit + h) + (2h / (h_fit + h)) x e^(−(1 + min(1, β)) x) F(|1 − β| x) / (1 − e^(−2x))
```

with `β = h / h_fit` and `F(a) = (1 − e^(−a)) / a`. It is 1 at the first
row and the filled share in the limit. Beside a window it is the same
integral over the window's weights. The CUSUM of squares' scale starts a
few rows before its sums, once it has 10 rows of Kish's size, and its
share runs from 1 to one half as

```text
r₂ = 1/2 + (q − D) / (1 + q) + D² ln(N / D) / (1 − q²)
```

with `q = e^(−x)`, `D` the scale's lead on the sums, and `N = D + 1 − q`.
The sums' own weights tell how far they have filled. Their `Σω` decays at
the memory and the fourth moment's at four times it, so the ratio of the
two runs from 1 to 1/4 as `(1 + p)(1 + p²) / 4`, `p = e^(−x/4)`, whatever
the rows' rate and the weights' scale. Divided by the filled share from
the first row, the two CUSUMs were too wide while young. Beside `ewridge`
at a half-life of 200, at memories of 50 to 800, the share of rows past
1.96 read, on 3 sets of 100 clean streams over the filled share
and of 300 over the share as filled:

| | first 150 rows after readiness | rows 300 to 600 | rows 4,700 on |
|---|---:|---:|---:|
| `cusum`, over the filled share | 6.5% to 34% | 4.8% to 23% | 5.3% |
| `cusum`, over the share as filled | 6.1% to 6.4% | 5.4% to 5.9% | 5.0% to 5.4% |
| `cusum_sq`, over the filled share | 11% to 16% | 5.0% to 15% | 4.9% to 5.2% |
| `cusum_sq`, over the share as filled | 6.2% to 6.7% | 5.0% to 5.2% | 4.8% to 5.0% |

A longer memory was worse, since its sums take longer to fill. The rest
of the excess over 5% in the first rows is the young scale's: the first
`z` are read against a spread of few rows, which leaves them a little
wider than 1. The CUSUM of squares beside `sgd`, divided by one half from
its first row, read 15% there.

`quantile` with an intercept absorbs its coverage the same way, and its
indicators' CUSUM is divided by the same share. On the 30 clean streams
of the spreads' table at `q = 0.5`, that CUSUM's spread read 1.03, 1.05 and 1.09 at
memories of 50, 200 and 800; at `q = 0.9` it reads up to 1.25 at 800 once
filled. `huber`, `holt` and the gradient fits absorb at a pace of their
own, so they keep the plain sum.

**A window absorbs a level as a mean over its rows, so its share is the
window's own.** Under `window_size` the fit's prediction is a mean of the
rows in the window, weighted by the decay inside it. A sum of its residuals
cancels less than one beside an exponential of the same Kish size. With the
fit's weights `g(y)` on the rows before it, the share is

```text
r = 1 − 2 ∫ g(y) e^(−κ y) dy + ∫∫ g(y) g(y') e^(−κ |y − y'|) dy dy'
```

with `κ = ln 2 / h`, `h` the CUSUM's memory. A window of `W` with no decay
weighs its rows alike. With `u = W ln 2 / h` that gives `r = 1 + 2 e^(−u) / u
− 2 (1 − e^(−u)) / u²`, 0.703 at the default memory, where read as an
exponential it is one half. Beside a decay the weights are the decay's, cut
at `W`. The share then runs from the window's form, for a window much
shorter than the half-life, to `h_fit / (h_fit + h)`, for one much longer.
The table is 20 sets of 20 clean streams of 6,000 rows beside `ewridge`,
over rows 3,000 on:

| window, half-life | spread of `cusum`, over the window's share | read as an exponential of its Kish size |
|---|---:|---:|
| 100 rows, none | 1.01 | 1.20 |
| 400 rows, none | 1.00 | 1.19 |
| 100 rows, 200 | 1.02 | 1.17 |
| 400 rows, 200 | 1.01 | 1.08 |
| 100 rows, 50 | 1.02 | 1.10 |

**Beside a fit that keeps the plain sum, `|cusum| > 3` is conservative and
finds less.** Its spread reads below 1 there. The table is measured at a
half-life of 200 on 30 clean streams of 6,000 rows, from each stream's
readiness on, and on 60 streams of 3,000 rows whose intercept moved half a
noise sd at row 1,500. For `holt` the streams are 12,000 and 9,000 rows,
with the break at 6,000, past its trend's readiness.
`scripts/diagnostics_rates.py memories` makes them:

| model | spread of `cusum`, memory 50 / 200 / 800 | breaks `\|cusum\| > 3` found | `break_wald` found |
|---|---|---|---|
| `ewridge`, `rls`, `lasso`, `kalman` | 1.03-1.04 / 1.05-1.06 / 1.05-1.06 | 100% / 100% / 100% | 93% / 100% / 100% |
| `quantile` | 1.03 / 1.05 / 1.09 | 100% / 100% / 100% | 93% / 100% / 100% |
| `huber` | 0.95 / 0.81 / 0.62 | 100% / 100% / 100% | 93% / 100% / 100% |
| `holt` | 0.92 / 0.74 / 0.52 | 80% / 58% / 5% | 20% / 100% / 100% |
| `ftrl` | 0.77 / 0.53 / 0.39 | 93% / 92% / 48% | 93% / 100% / 100% |
| `sgd` | 0.78 / 0.55 / 0.37 | 88% / 50% / 0% | 93% / 100% / 100% |
| `pa` | 0.51 / 0.48 / 0.47 | 0% / 0% / 0% | 93% / 100% / 100% |

A gradient fit absorbs a level at its learning rate, so the residuals the
CUSUM sums carry little of the move. Beside `pa` it found no break at any
memory, and beside `sgd` none to 88% of them by memory. `break_wald` reads the
target and the features alone, so it finds a break beside every model
with features alike. Beside `holt` its pair of fits has only the level to compare. On
the clean streams `|cusum| > 3` held on at most 0.6% of rows beside any
model, and `break_wald` on at most 0.01%. `calibration_wald` past
`chi2(2)`'s 0.01% value found none of the breaks beside `sgd` and `ftrl`.
Beside `pa` it held on every clean row, whose predictions here were not
calibrated.

Beside a fit at a half-life of 200, each flag found its break at the
rate and delay this paragraph's table gives.
The breaks are on 200 streams of 3,000 rows, at row 1,500, made by
`scripts/diagnostics_rates.py breaks` in the three ways of [Detect a
break](how-to/detect-a-break.md#test-a-fit-that-forgets-for-a-break)'s
first recipe. The rows
flagged with no break are on 200 streams of 25,000 rows, the Gaussian
streams of [Are its tails heavy?](#are-its-tails-heavy), made by
`scripts/diagnostics_rates.py windowed`:

| flag | the breaks it found | median delay | rows flagged with no break |
|---|---|---:|---:|
| `\|cusum\| > 3` | every intercept break | 66 rows | 0.28% |
| `\|cusum_sq\| > 3` | every variance break | 20 rows | 0.28% |
| `break_wald > 21.1`, `chi2(3)`'s 0.01% value | every slope break | 55 rows | 0.011% |

A normal variable passes 3 on 0.27% of draws. Not divided by their
null's share, the two CUSUMs held on 0.002% and 0.011% of the rows with
no break, and found the breaks 96 and 20 rows after them.

**`break_wald` reads the exact variance of the two fits' difference.**
Each fit is `β = G⁻¹ Σ ω z y` over `z = (1, x)`. With noise of variance
`σ²`, the difference has the variance

```text
σ² (A_f H_ff A_f + A_s H_ss A_s − A_f H_fs A_s − A_s H_fs A_f)
```

where `A = G⁻¹`, and the `H` are the sums of `ω_f² z z'`, `ω_s² z z'`
and `ω_f ω_s z z'`, which the diagnostic keeps. That variance is a sum
of one outer product per row, of `(A_f ω_f − A_s ω_s) z`, and those
vectors are orthogonal to the design: summed against `z'` they give
`A_f G_f − A_s G_s`, the identity less itself. So over `n` rows its rank
is at most `n − k` for `k` coefficients, and `break_wald` waits for `2k`
rows, whatever their weights. Read sooner, the rounding would decide the
distance, and multiplying every weight by one constant could turn it on
or off. A gap long enough to leave the old rows below the rounding of the
new ones starts the wait over: that is where the history's share of the
weight after the row, squared, is at most `2^-53`, so for a history of `W`
rows at weight 1 a gap of `26.5 + log₂ W` half-lives before a row of
weight 1. Without starting the wait over, the rows just after such a gap read their
distance off a variance whose rank was theirs alone: multiplying every
weight by one constant moved it by 8.6e-4 of itself after a gap's decay of
`2^-60`, by more than itself at `2^-80`, and turned it on or off after
one that took the history to 0. The slow fit's Gram alone
would serve both fits only while the design holds still. When a feature's
spread fell tenfold with nothing broken, that shortcut passed 21.1 after
the change on 85% of 100 streams, and the exact variance on none. On
streams with no break, `break_wald` passed `chi2(3)`'s 5% value on 4.5% to
5.0% of rows. Beside AR(1) features at 0.99 and 0.999 it passed on 3.4%
and 3.1%, with means of 2.9 and 2.7 for `chi2(3)`'s 3. With 20 features at
a half-life of 20 its mean was 20.7, for `chi2(21)`'s 21.

**The CUSUM cannot see a slope that moves on a feature centered at zero.**
The residuals of the old fit on the new relationship are `(b_new − b_old)
· x + noise`, whose mean is zero when `x`'s mean is. Run once over the
same 1,000 streams as the intercept breaks of *Run once, the CUSUM is
Brown, Durbin and Evans' test*, with `x0`'s slope rising
by half at row 1,500, the CUSUM crossed on 6.5% of them. It crossed on
6.0% of the streams with no break, within one standard error of 6.5%.
`break_wald` exists for this case, on a fit that forgets: run once, it is
null.

**`drift` sees a rise in the residuals' level, and little else.** It is
scaled by `sigma`, which adapts to a larger spread within a half-life. Beside
a fit at a half-life of 200 it found 1.5% of the variance breaks on those
200 streams and none of the others. Its false-alarm rate depends on the residuals' tails,
which [Are its tails heavy?](#are-its-tails-heavy) measures. It reads the
residuals of the predictions a readiness gate withholds too, as `sigma`
does, so a gate never hides a break: under `drift_action="reset"` the rows
a `min_weight` withholds after a reset are the rows that show the next
break.

[Detect a break](how-to/detect-a-break.md) runs these tests on a stream
that breaks in each of the three ways, dates a break with the run-once
CUSUM, and says what to do at each. Read together, the flags tell the
breaks apart: `cusum` fires on an intercept shift and `cusum_sq` first on
a change of noise. `break_wald` fires on a moved coefficient, the
intercept's included, so it fires on an intercept shift too. `cusum_sq`
fires on a slope break as well, while the fit catches up. Apart from those
two cases, each flag is quiet, or late, on the other kinds. For the noise
and slope breaks, over the 200 streams of each break that
`scripts/diagnostics_rates.py breaks` makes, the share of streams each
flag fired on, and its median delay, are:

| break | flag | streams it fired on | median delay |
|---|---|---:|---:|
| the noise doubled | `\|cusum_sq\| > 3` | every one | within 20 rows |
| the noise doubled | `\|cusum\| > 3` | half | 174 rows |
| the noise doubled | `break_wald > 21.1` | 17% | 205 rows |
| a slope moved | `break_wald > 21.1` | every one | within 55 rows |
| a slope moved | `\|cusum\| > 3` | a quarter | 204 rows |
| a slope moved | `\|cusum_sq\| > 3` | nine in ten | 74 rows |

## Can its t be trusted?

**A coefficient's standard error assumes residuals that are independent
and equally spread, and a target that looks ahead breaks the first
assumption.** A target that sums the next `h` rows' changes shares `h − 1`
of them with its neighbor. Its residuals are then correlated over `h − 1`
rows by construction, a moving average of order `h − 1`. Against a feature
that moves slowly, the coefficient's variance grows by about `1 + 2 Σ ρ_x(l)
ρ_e(l)` over the lags, close to `h`. So a t that ignores the overlap is
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
estimation error that a sandwich over in-sample residuals leaves out, and
it tends to read above statsmodels' `cov_type="HC0"` and `"HAC"`, though
not always. Run once on 1,500 rows, over 50 seeds, the ratio to
statsmodels' ran from 0.990 to 1.034 for HC0 and from 0.985 to 1.059 for
HAC, and was below 1 for some coefficient on 5 and 8 of the seeds. The lags default to twice the target's
horizon in rows. The memory is `robust_se_half_life`, the fit's by default.
It costs 350 ns a row at 5 features and 10 lags, and only `ewridge` and
`rls` take it, the least-squares fits.

**The horizon is one number in rows, and every diagnostic that needs it
reads it.** On a spec with no clock column it is `embargo`, rounded up,
since a row is a unit there. On a clock column `embargo` is in clock units,
and the rows it spans are not known before the stream, so the spec gives
`horizon_rows`, the rows the target looks ahead. It is a count of rows,
named as `warm_rows` and `window_rows` are, because the residuals overlap by
rows whatever the clock. A spec with a clock column, an `embargo` and no
`horizon_rows` has a horizon of 0, and each diagnostic that reads one says
so once, in a `ReadinessWarning`. Under a horizon of `h` rows:

| diagnostic | under a horizon |
|---|---|
| `se_coef_hac` | its lags default to `2h` |
| `calibration_wald` | Wald's test of `a = 0, b = 1` with Newey and West's variance at `2h` lags, from the residual under the null, `y − pred` |
| `cusum`, `cusum_sq` | each sum over its long-run variance: Bartlett's weights on its terms' products over `2h` lags |
| `break_wald` | the noise variance times the studentized residuals' long-run over short-run variance |
| `ljung_box` | tests lags `h` to `h + L − 1` against Bartlett's covariance past an `MA(h − 1)` |
| `breusch_pagan`, `ramsey_reset` | Wald's test of the same coefficients with Newey and West's variance at `2h` lags, from the residual under each null |

**Lags read from `embargo` stop at 64.** Each lag adds a sum to every slot
of each diagnostic that reads the horizon:

| diagnostic | numbers a slot keeps per lag |
|---|---:|
| `se_coef_hac`'s sandwich | `(k + 1)²` |
| `breusch_pagan` | `(k + 1)²` |
| `ramsey_reset` | 16 |
| `calibration_wald` | 4 |
| `cusum`, `cusum_sq` | 2 each |

So an embargo of 5,000 rows would ask 40 MB a slot. 64 lags cover a
horizon of 32 rows at twice it, and past that a `ReadinessWarning` says so.
A `horizon_rows` given by the spec takes twice its value in lags, whatever
they cost. `robust_se_lags` sets `se_coef_hac`'s lags alone.

The table is measured on 30 streams of 4,000 rows, each a five-row
look-ahead target against two AR(1) features at 0.5 to 0.95, with nothing
missing. Each cell is the share of rows 1,500 to 3,999 on which the plain
form passed its 5% value, and after the arrow the horizon's form:

| at a half-life of 200 | `breusch_pagan` | `ramsey_reset` | `calibration_wald` | `cusum` | `cusum_sq` | `break_wald` |
|---|---:|---:|---:|---:|---:|---:|
| AR 0.5 | 20.6% → 4.0% | 10.1% → 3.1% | 7.9% → 0.0% | 23.6% → 7.6% | 12.7% → 6.2% | 47.6% → 2.5% |
| AR 0.8 | 28.0% → 4.3% | 22.4% → 3.9% | 11.7% → 0.1% | 23.1% → 7.4% | 14.0% → 8.2% | 56.4% → 4.4% |
| AR 0.95 | 37.9% → 4.7% | 39.2% → 5.2% | 18.6% → 0.0% | 24.9% → 8.1% | 14.7% → 7.1% | 62.5% → 4.7% |

Run once, the plain forms passed on 5% to 68%, and the horizon's on 2.1%
to 13.2%. Newey and West's estimate errs small on a few thousand rows of a
persistent feature, and a run-once statistic keeps every row of it. At a
ten-row horizon the windowed forms passed on 0.0% to 8.4%. `break_wald`'s
factor is exact for a persistent feature and generous for one that is not,
whose slopes' variance the overlap barely moves: against independent
features it passed on 1.1% to 2.1%. The CUSUMs read 6% to 9%: Bartlett's
weights over `2h` lags take in a little less than the overlap's whole
long-run variance. The windowed calibration reads low because the fit
absorbs part of a miscalibration ([Is it scaled
right?](#is-it-scaled-right)), with or without a horizon.

On a target summing the next `h` rows' shocks against a persistent
feature, the coefficient's true spread was 2.1 to 2.3 times `se_coef` at
`h = 5`. It was 3.1 to 4.2 times at `h = 20`. `se_coef_hac` read 83% to
97% of the true spread at `L = h`, and 90% to 104% at `L = 2h`, the
default.

[Check a coefficient's standard error](how-to/check-standard-errors.md)
compares the three standard errors on 100 streams whose true slope is 0,
and chooses the one to read.

**`po.gram.coef_stats` and `marginal`'s `t_stat` read Kish's size too, and are
silent about the overlap the same way.** `po.gram.coef_stats` computes each
coefficient's standard error and t from a Gram, such as one loaded from a
state, at `n = target_n_kish` ([A ridge and its
t](GRAM.md#a-ridge-and-its-t)). Read its t as a scale for
comparing coefficients. `marginal`'s `t_stat_serial` (with `lags=` and
`serial_rule=`) corrects each pair's count
for rows that resemble their neighbors, after Bartlett (1935): over
1,000 pairs of independent AR(1) series, `|t|` passed 1.96 on 45% of them
and `|t_stat_serial|` on 5.4%
([`marginal`](MODELS.md#marginal--every-pairs-moments-kept-in-the-state)).

## What is it missing?

**A residual stream with nothing left to learn looks like noise: no lag,
no spread that follows a feature, no curvature.** Three tests and one
correlation look for what is left.

| field | statistic | with nothing missing | sees |
|---|---|---|---|
| `ljung_box_<t>` | Box and Pierce's (1970) `Q = n Σ_{l=1..L} ρ̂_l²` over `ljung_box_lags` lags (default 10), at Kish's `n`; run once, Ljung and Box's (1978) `n(n + 2) Σ ρ̂_l² / (n − l)` | `chi2(L)`: 18.3 at 5% for 10 lags | a lag the fit is missing, or a half-life too long |
| `breusch_pagan_<t>` | Koenker's (1981) form of Breusch and Pagan's (1979) test: `n R²` of `resid²` regressed on the slot's features | `chi2(k)` for `k` features | a spread that moves with a feature, linearly |
| `ramsey_reset_<t>` | Ramsey's (1969) RESET as a Lagrange multiplier: `n R²` of `resid` on `pred²` and `pred³` beside `pred` | `chi2(2)`: 5.99 at 5% | a curvature the fit is missing |
| `autocorr_<t>` (`emit_autocorr`) | the EW correlation of each residual with the one `resid_autocorr_lag` predicted rows back (default 1) | near 0 | one lag, read as a correlation |

`emit_specification` writes the first three at `specification_half_life`,
the model's half-life by default. Run once, they are statsmodels'
`acorr_ljungbox`, `het_breuschpagan` and `compare_lm_test` on the
out-of-sample residuals. On 200 streams of 3,000 rows, each passed its 5%
value on 2.0% to 7.0% of the rows of streams missing nothing, run once or
windowed. Each passed on 98% to 100% of the rows of streams missing what it
looks for: an AR(1) at 0.3, a spread linear in a feature, a square of one.
`emit_autocorr` reads at the model's half-life, and a gap capped by
`gap_cap` or a session change starts a new run of lags.

**Under a memory, `ljung_box` drops Ljung and Box's small-sample
factor.** Their `(n + 2)/(n − l)` corrects for the `l` rows of a sample
that have no partner `l` rows back. Under exponential weights every row
but the oldest has one, and the factor only inflates `Q` at Kish's `n`.
On iid residuals with 10 lags, Ljung and Box's form passed its 5% value
on 15.8% of streams at a half-life of 10 rows, and on 9.7% at 20. Box and
Pierce's passed on 4.8% and 5.7%. From a half-life of 50 to 200 the
two read 4.6% to 6.7%. Run once, Ljung and Box's held at 5.3% to 7.7%
from 30 to 1,000 rows, where Box and Pierce's fell to 2.7% at 30, so the
run-once form keeps it.

**On a target that looks ahead `h` rows, `ljung_box` tests lags `h` to
`h + L − 1`.** Its residuals are correlated within the horizon by
construction, so lags 1 to `h − 1` would find the overlap and nothing else.
The plain `Q` passed its 5% value on 61% of the rows of five-row look-ahead
streams with nothing missing, and the shifted one on 6.0% to 6.7%. The
horizon is `embargo` on a spec with no clock column and `horizon_rows` on
one with ([Can its t be trusted?](#can-its-t-be-trusted)). There
`breusch_pagan` and `ramsey_reset` become Wald's tests of the same coefficients
with Newey and West's variance. Beside an AR(1) feature at 0.95, on
streams missing nothing, their `n R²` forms passed their 5% values on 38%
and 39% of the rows, and the Wald forms on 4.7% and 5.2%.

**Each test sees only its own shape.** Breusch-Pagan regresses `resid²` on
the features themselves, so a spread symmetric in a feature, as `|x0|`, is
invisible to it. On the stream of [Find what a model
misses](how-to/find-what-the-model-misses.md)'s recipe, with its noise times `0.5 +
|x0|` in place of `exp(0.5 x0)`, it read a median of 3.0, under `chi2(2)`'s
5% value of 5.99. `ramsey_reset`, whose `pred²` carries `x0²`, read 7.7, and
caught the spread in its place. Ljung-Box sees correlation at the lags it
tests and no further. RESET sees curvature along the prediction, so a
curvature in a feature the prediction barely uses can hide from it.

[Find what a model misses](how-to/find-what-the-model-misses.md) plants a
lag, a spread and a curvature, reads which test fires on each, and adds the
feature that fixes it.

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

**`drift_threshold` is set against the residuals' kurtosis.** On 200
streams of 25,000 rows at a half-life of 200, `y = 0.5 + x0 − x1` plus
noise, `drift` flagged at these rates. `kurtosis` is read as its mean over each
stream, averaged over the streams.
`scripts/diagnostics_rates.py windowed` makes the streams and every figure
in this section's tables:

| noise | mean `kurtosis` | flags per 100,000 rows at 20 | at 30 | at 50 | at 80 |
|---|---:|---:|---:|---:|---:|
| Gaussian, Student's t with 10 or 6 degrees of freedom | under 3 | 0 | 0 | 0 | 0 |
| Student's t, 5 degrees of freedom | 3.9 | 0.1 | 0 | 0 | 0 |
| Student's t, 4 degrees of freedom | 8.8 | 0.8 | 0.1 | 0 | 0 |
| Student's t, 3 degrees of freedom | 30 | 5.6 | 1.6 | 0.3 | 0.1 |

**The CUSUM of squares reads the tails it is given.** It standardizes
the sum of `z² − 1` by the measured variance of `z²`, `E[z⁴] − 1`, and
not by a Gaussian's 2. Under an excess kurtosis `κ` that variance is
`2 + κ`. On the same 200 streams, `|cusum_sq| > 3` held on these
shares of rows. The bank writes the first column. The second is the same
statistic rebuilt from `studentized` with 2 in its place, a rebuild that
gives the bank's own `cusum_sq` back within 1e-11:

| noise | over `E[z⁴] − 1` | over 2 |
|---|---:|---:|
| Gaussian | 0.28% | 0.35% |
| Student's t, 5 degrees of freedom | 0.54% | 8.4% |
| Student's t, 4 degrees of freedom | 0.73% | 15% |
| Student's t, 3 degrees of freedom | 0.99% | 31% |

On the same streams `|cusum| > 3` held on 0.28% to 0.70% of rows, and
`break_wald > 21.1` on 0.11% or fewer, even with 3 degrees of freedom.
What it gives up is power: under heavy tails a squared residual varies so
much that a change of spread takes longer to stand out from it.

**A windowed kurtosis is itself noisy under heavy tails.** The kurtosis of
Student's t with 3 degrees of freedom is infinite, so a windowed estimate
never steadies. Over the rows of those 200 streams its median was 11.5,
and its mean 30, the table's figure. Read the mean over a long stable
stretch, as the measurements in the table did.

[Measure heavy tails](how-to/measure-heavy-tails.md) measures the tails
over a stable stretch, sets `drift_threshold` from them by the table, and
counts each detector's flags before and after a break that triples the
noise.

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
`kalman` take it, the models that read a row's leverage. Under `embargo` a
row is folded at its release with the residual and the error inflation of
its prediction, so its influence is the move it made against the fit that
predicted it.

[Find an influential row](how-to/find-an-influential-row.md) plants three
bad prints, ranks the rows by influence, and decides what to do with the
rows past the cutoff.

**A cutoff scales with the fit: `4 · sqrt((k + 1) / weight_sum)` for `k`
features.** A clean row's influence is about its leverage's square root,
and a leverage is about `(k + 1) / weight_sum`. So a fit with more
features or a shorter memory reads larger on every row. On clean rows at
half-lives of 20 to 1,000 rows and 2 to 10 features, the 99.9th
percentile of `|influence|` read 2.5 to 3.2 times `sqrt((k + 1) /
weight_sum)`. Four times it held 0.001% to 0.014% of the clean rows, and
every row planted six spreads out and six off the line, one in 200. For a
fit at a half-life of 200 with two features, that cutoff is 0.41. A fixed
0.5 held 7.3% of the clean rows of a fit at a half-life of 20 with three
features.

## Are its features healthy?

**A feature can go wrong without any residual showing it: a feed that
stops and is carried forward is still a number on every row.** A fit that
keeps sums, such as `rls`, then sees a column that no longer varies. Under
forgetting, the information about that coefficient decays and nothing
replaces it, so the coefficient winds up (Åström and Wittenmark 1995).
[Data quality](DATA-QUALITY.md#constant-and-frozen-features)
measures the windup. `emit_feature_health` watches the features
themselves, and flags a frozen feed long before a coefficient runs away.

| | `emit_feature_health` |
|---|---|
| fields | `spread_ratio_<feature>`, `mean_shift_<feature>`, once per instance (the part of the model that fits one half-life) and feature, read before the row |
| update | each feature's EW mean and standard deviation at `feature_health_half_life` and at four times it |
| statistic | `spread_ratio` is the fast standard deviation over the slow one; `mean_shift` the two means' difference in units of the slow standard deviation. A steady feature reads about 1 and 0 |
| memory | the fit's memory by default; refused where that is `inf` (run once, with no window), since both memories would be the whole stream |
| threshold | `spread_ratio` outside 0.5 to 2: no row of a clean feature at a half-life of 200, independent or AR(1) at 0.95. `\|mean_shift\| > 0.3`: every move of one spread, within 190 rows; 6% of rows of an AR(1) feature at 0.95, and 0.03% above 0.5 |
| cost, models | 31 ns a row at 5 features; the nine linear models with features |

Under `embargo` the features fold when their row is released, with the
row's target, so the fields lag the feed by the embargo. On a feed frozen
at row 1,500 at a half-life of 20, the ratio fell below 0.5 at row 1,547.
Under an embargo of 100 rows it fell below at row 1,646.

[Check the health of a model's features](how-to/check-feature-health.md)
freezes a feed, flags it about a thousand rows before `rls`'s coefficient
runs away, and fixes a stale or moved feature.

## Is another model better?

**A model is better than another only on residuals neither has learned,
and every comparison here reads those.** The comparisons come in three
kinds: a choice among a grid's settings as the stream runs, a test that
one model predicts better than another, and metrics over an output frame
after the run.

| kind | switch or call | what it gives | how-to |
|---|---|---|---|
| running accuracy | `emit_metrics` | `ic_<t>`, the EW correlation of prediction with target; `r2_<t>`, the out-of-sample R² against the running mean; `hit_rate_<t>`, the share of rows whose sign the prediction got right | |
| a choice as the stream runs | `emit_selected`, `emit_averaged` | the grid slot with the lowest EW error so far, and every slot's prediction weighted by its error | [Choose among a grid's settings as the stream runs](how-to/choose-among-a-grid-online.md) |
| a test of two models | `po.eval.diebold_mariano`, `po.eval.clark_west`, `po.eval.seqtest`, the `seqtest` model | a t statistic, or evidence that can be read at any row | [Compare two models](how-to/compare-two-models.md) |
| metrics after the run | `po.eval.metrics`, `window_metrics`, `compare_specs`; `po.eval.sums` for an output too large to hold | R², IC, hit rate and MSE, per group or per window | [Evaluate an output frame](how-to/evaluate-an-output-frame.md) |

**`hit_rate` asks whether prediction and outcome fall on the same side of
zero.** A row where either is exactly zero is on neither side, and both
`emit_metrics` and `po.eval.metrics` leave it out. So on a target that is
always positive, such as a plain ratio, every row counts as a hit. Write a
return about zero with Polars expressions ([Write a ratio target to sit
about zero](how-to/make-a-target-that-looks-ahead.md#write-a-ratio-target-to-sit-about-zero)). An `sgd` fit with
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
commits to the best slot. `emit_averaged` hedges across all of them. A
spec with one slot per target refuses both, since one slot leaves nothing
to choose or average.

| switch | fields, one per target | what it gives |
|---|---|---|
| `emit_selected` | `selected_<t>`, `pred_<t>__selected` | the ridge value, feature set or half-life with the lowest EW out-of-sample error so far, each slot at its own half-life |
| `emit_averaged` | `pred_<t>__averaged` | every slot's prediction, weighted by `exp(-eta * (mse / best_mse - 1))`, with `mse` the slot's EW squared error and `best_mse` the best slot's; `average_eta` is `eta` (default 1), and `inf` gives `emit_selected`'s choice |

**Across a half-life grid, each slot's error decays at its own half-life,
so a short half-life is ranked on fewer, more recent rows.** Within a
ridge or feature-set grid, the slots share one half-life and are ranked
like for like. `lasso`'s `select_half_life` ranks its path on one
half-life.

[Choose among a grid's settings as the stream
runs](how-to/choose-among-a-grid-online.md) turns both switches on, reads
them against the grid's own slots, and says when to fix one setting with
`po.eval.grid_search` instead.

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
average. So under the null the differential favors the smaller model,
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
([`seqtest`](MODELS.md#seqtest--a-sequential-test-of-a-sign-by-betting)).

[Compare two models](how-to/compare-two-models.md) runs both tests on 200
pairs of nested models, and reads a `seqtest` of two models as the stream
runs.

### Evaluating an output frame

`po.eval`'s calls score a finished run's output frame: `metrics` per spec
and group, `window_metrics` per window of the clock, `compare_specs` over
several specs, and `unpack` to long form. For an output too large to hold,
`sums` reduces each chunk to ten numbers per slot and group. The sums are centered,
so a target far from zero keeps its variance. `merge_sums` adds the parts,
the same to rounding whatever the split, and `from_sums` turns the total
into the metrics. [Evaluate an output
frame](how-to/evaluate-an-output-frame.md) runs each, and says when to
pass the spec dict in place of its name.

## Is it ready?

**A model should not report a number before it has seen enough data to
give one, and three gates say what "enough" means.** A fourth, the
seeding warm-up, holds back the models that seed themselves from their
first rows. The gates hold back predictions only: the model learns from
every row either way. [Warm-up gates](MODELS.md#warm-up-gates), in
MODELS.md, states each gate, its default and the field it reads, and [Warm-up and
readiness](explanation/warm-up-and-readiness.md) explains what each
measures. A withheld
prediction's `withheld_reason` names the gate that held it:

| gate | `withheld_reason` |
|---|---|
| `min_settled_frac` | `below_min_settled_frac` |
| `min_weight` | `below_min_weight` |
| `max_error_inflation` | `above_max_error_inflation` |
| `warm_rows` (`kmeans`, `hmm`, `bocpd`): the rows a model learns from before it has seeded its centers, states or prior; `kalman`'s rows before a target's prior is sized | `below_warm_rows` |
| none, the solve itself (`ewridge`): a fit whose normal equations are past the condition at which a solve keeps a digit, see [`ewridge`](MODELS.md#ewridge--exponentially-weighted-ridge-regression) | `ill_conditioned` |
| none, the learning rate itself (`sgd`): a target whose recent steps' gains have a mean square above twice their mean, so each step grows its own row's residual and the fit diverges | `unstable_rate` |

The reasons are listed in their order of precedence, which is the order
`withheld_reason` reports them in. `max_error_inflation` is on by default
on `ewridge` alone, at `sqrt(2)`. `emit_error_inflation` writes each row's
own factor, `error_inflation_<t>`, against the row's own features: large
for a row leaning on a direction the data never showed.

**A `min_weight` above the weight a stream can reach is never met, and a
readiness warning says so.** Under a half-life, a row's weight decays, so a
target's weight stops growing at a ceiling, where each new row only
replaces what decay removes. For rows of weight `w` spaced `d` clock units
apart, the ceiling is `w / (1 − 2^(−d / half_life))`: for rows of weight 1,
one clock unit apart, about 7.7 at a half-life of 5. Rows that come
faster or weigh more lift the ceiling, so the warning projects it and does
not promise it. The ceiling holds for a target every row carries. A target
the rows stop carrying has none: its weight halves every half-life, toward
0. The warning tells the two apart by `weight_sum`, the weight of every row
the model learned, whether the row carried the target or not: a target's
own weight below it means the model has not learned the target from every
row. One weight at one time cannot say whether such a target has stopped or
comes on a share of the rows, so the warning says only that its weight falls
on each row without it. Once the model is
within 5% of its steady state and the gate has withheld every row for a
further half-life, the bank issues a `ReadinessWarning` that names the
gate, and the target where the model predicts one. Its forms:

| what the warning finds | what it gives | what to do |
|---|---|---|
| a target every row carries, with a ceiling it can project from the rows so far | the ceiling, rounded down, and the `half_life` (or `lam`) above which the ceiling passes the floor the gate reads (`min_weight`, or 2 for `ew_cov`) | lower `min_weight` below the ceiling, or increase `half_life` (or `lam`) above the value given |
| a target the model has not learned from every row, where rows that each carried it would reach the floor | no figure: the target's weight falls on each row without it, toward 0 if the rows have stopped carrying it | check that rows reach the model with the target's value (`rls` learns a row only where every target has a value) |
| a target the model has not learned from every row, where rows that each carried it would not reach the floor either | the ceiling of rows that each carry the target, rounded down, and the `half_life` (or `lam`) above which it passes the floor | lower `min_weight` below that ceiling, or increase `half_life` (or `lam`) above the value given, and check that rows carry the target's value |
| no row the model learned has the target's value and a positive weight (for a model that predicts no target, a positive weight alone) | no figure: there is no weight to project from | check that rows reach the model with the target's value and a positive weight |
| the warning comes on a row of weight 0, which adds nothing to the weight | no figure: no stream of rows like it reads a positive weight | check that rows reach the model with a positive weight, lower `min_weight`, or increase `half_life` (or `lam`) |
| rows too uneven to project, such as one row far heavier than the rest | no figure | lower `min_weight`, or increase `half_life` (or `lam`) |
| a hard window, which limits the weight in addition to the decay | no figure | lower `min_weight`, or increase `half_life` (or `lam`) or `window_size` |

`ew_cov` needs a weight of at least 2 for a covariance, so it lifts a
lower `min_weight` to 2, and its warning says so and does not offer
lowering `min_weight`.

**The noise gate warns the same way while the model has not fit at
all.** Until `weight_sum` reaches a row per coefficient, or `min_weight`
where it is set, the model has no fit and `error_inflation` is infinite,
which no `max_error_inflation` opens. It comes at the same point as the
`min_weight` warning. It gives where `weight_sum` tops out at the rows so
far, rounded down. It also gives the `half_life` (or `lam`) above which the
model fits and Kish's effective sample size carries its coefficients,
rounded up. On a row of
weight 0, or on rows too uneven to project, it gives no figure. A target's
nulls do not move `weight_sum`, so this warning has no form for a target
the rows stopped carrying.

[Decide when a model is ready](how-to/decide-when-a-model-is-ready.md)
reads `withheld_reason`, chooses among the gates on one stream, and
applies a warning's remedy.

## How wide is its error?

**Four diagnostics measure a prediction's error as the stream runs.**
Each is read before its row, so it can be served with the row's
prediction. [Put an interval on a
prediction](how-to/put-an-interval-on-a-prediction.md) builds an interval
from each and reads its coverage.

| field | what it is | assumes |
|---|---|---|
| `sigma_<t>` (`emit_sigma`) | the EW standard deviation of the out-of-sample residuals, at the model's half-life | nothing, as a spread; `pred ± z · sigma` assumes Gaussian residuals |
| `zscore_<t>` (`emit_zscore`) | `resid / sigma`: how surprising the row was, in units of recent error | as `sigma` |
| `abs_resid_q<p>_<t>` (`resid_quantiles`) | the EW quantile of `\|resid\|` at level `p`, from one decaying DDSketch per slot, within 0.78% of the exact weighted quantile | no distribution |
| `lo_<t>`, `hi_<t>`, `coverage_<t>` (`conformal`) | an interval tracked to cover the share asked for, and the coverage it has delivered | no distribution |

`sigma`'s weight ages on every row the model sees, a row with no
prediction or with weight 0 included, so it forgets across a gap as the
clock says. Under a `window_size` it is the window's, as the fit is. It
folds the residuals of the predictions a readiness gate withholds too. A
gate withholds output only, so `sigma` -- and `zscore`, the drift
detector's scale, the ranking of `emit_selected` and `emit_averaged`, and
the conformal radius's start -- is the same whatever `min_weight`,
`min_settled_frac` or `max_error_inflation` says, and it can be a number
on a row whose prediction is withheld. On `ewridge` it reads none below
the model's own floor of a row per unknown, where the system has fewer
rows than unknowns. Each of these costs under 25 ns a row, and the ten
linear models take them.

### Conformal intervals

**`conformal` tracks a quantile of `|resid|` directly, so its long-run
coverage is the number asked for, whatever the residuals do.** Its radius `q` grows by
`conformal_rate · sigma · coverage` on a miss and shrinks by
`conformal_rate · sigma · (1 − coverage)` on a hit. That is the
quantile-tracking step of Angelopoulos, Candès and Tibshirani (2023),
scaled by `sigma` so that the rate means the same in any units. A miss is rarer than a hit by
the ratio the coverage sets, so the two steps balance exactly at the
coverage asked for. This is conformal prediction (Vovk, Gammerman and
Shafer 2005) without a held-out set, because every residual is already out
of sample.

**Under a `weight` column, each step is scaled by the row's weight over
the mean weight**, so a heavy row's miss counts for more. The share of
rows covered can then sit on either side of the number asked for. The step is
taken once per scored row, so the radius moves faster in clock time where
rows are denser. `conformal_rate` is 0.05 by default.

**Under a long delay, set `conformal_rate="auto"`.** Under
`embargo` or `label_clock` a row's label arrives after the row was shown
its interval, and the row is scored against the radius it was shown. So
the radius learns of a miss only when the label arrives, and every row
served in the meantime was shown the same radius. With `n` rows waiting
for their labels on average (the rows that arrive in a unit of clock,
times the mean delay), the radius is a feedback loop whose news is `n` steps old.

Each step moves the share of misses by about `conformal_rate · g`. Here `g`
is the density of `|Z|` at the quantile asked for, with `Z` standard
normal: 0.64 for a 50% interval, 0.21 for 90%, 0.12 for 95% and 0.029 for
99%. A loop whose news is `n` steps old holds steady when that move times
`n` stays below about one half. Past about 1.6 it swings: the radius
overshoots, the misses stop, it undershoots, and the interval is wider on
average. So the step should be `0.05 / (1 + 0.1 · g · n)`, which is the
default when nothing waits. For a 90% interval that is `0.05 / (1 + n /
48)`.

`"auto"` takes that step with `n` counted as the stream runs: at each
label's arrival, the rows still waiting in its group, averaged over the
model's half-life. With nothing waiting it is 0.05 to the bit. The mean
is slow on purpose. A step that read the count at each instant would take
its smallest steps when the waiting rows pile up, which is when the
misses do. Flights pile up in the afternoon, when the delays are worst, and
such a step covered 92.1% of them where 90% was asked. Under a delay that
does not move, the formula with a fixed `n` gives the same step.

On New York's 2013 flights (327,346 departures, the residual of a tree
model's arrival delay, scored on October to December), `"auto"` held 90%
intervals within 10% of the width with no delay, as the formula did:

| labels | rows waiting | default 0.05: covered, half-width | the formula's rate | the formula: covered, half-width | `"auto"`: covered, half-width |
|---|---:|---:|---:|---:|---:|
| at once | 0 | 90.0%, 44.5 minutes | 0.05 | 90.0%, 44.5 minutes | 90.0%, 44.5 minutes |
| an `embargo` of 1 hour | 54 | 90.0%, 47.4 | 0.024 | 89.9%, 46.1 | 89.9%, 46.0 |
| at landing (`label_clock`) | 152 | 90.0%, 63.6 | 0.012 | 89.9%, 48.5 | 89.9%, 48.6 |
| an `embargo` of 8 hours | 338 | 90.3%, 189 | 0.0063 | 89.9%, 46.6 | 89.9%, 46.5 |

At 50% the default's swing pins the radius at zero for long stretches, so
under the 8-hour delay it covered 70% of rows at a half-width of 127
minutes. The formula's 0.0022 covered 49.95% at 15.4.

A smaller step also recovers more slowly from a radius that is wrong for
another reason. That can be a start from the first residuals of a model
that has learned little, or a jump in the noise. No step corrects the
radius faster than the labels arrive. Where the noise moves within the
delay, choose the rate on earlier months, as
[CALIBRATION.md](CALIBRATION.md) chooses any setting.

[Put an interval on a
prediction](how-to/put-an-interval-on-a-prediction.md) compares the
Gaussian, quantile and conformal intervals on Gaussian and heavy-tailed
noise, and sets `conformal_rate="auto"` under a delay.

## Sources

Each claim above that is not measured here rests on one of these.

| source | for |
|---|---|
| Angelopoulos, A. N., Candès, E. J. and Tibshirani, R. J. (2023). Conformal PID control for time series prediction. *Advances in Neural Information Processing Systems* 36. | the quantile-tracking step of `conformal` |
| Åström, K. J. and Wittenmark, B. (1995). *Adaptive Control*, 2nd ed. Addison-Wesley. | estimator windup under exponential forgetting |
| Bartlett, M. S. (1935). Some aspects of the time-correlation problem in regard to tests of significance. *Journal of the Royal Statistical Society* 98(3), 536–543. | the count correction behind `marginal`'s `t_stat_serial` |
| Belsley, D. A., Kuh, E. and Welsch, R. E. (1980). *Regression Diagnostics*. Wiley. | DFFITS, the basis of `influence` |
| Breusch, T. S. and Pagan, A. R. (1979). A simple test for heteroscedasticity and random coefficient variation. *Econometrica* 47(5), 1287–1294. | `breusch_pagan` |
| Brown, R. L., Durbin, J. and Evans, J. M. (1975). Techniques for testing the constancy of regression relationships over time. *Journal of the Royal Statistical Society, Series B* 37(2), 149–192. | recursive residuals, the CUSUM and the CUSUM of squares, and their boundaries |
| Chu, C.-S. J., Hornik, K. and Kuan, C.-M. (1995). MOSUM tests for parameter constancy. *Biometrika* 82(3), 603–617. | the moving-sum form of the CUSUM |
| Clark, T. E. and McCracken, M. W. (2001). Tests of equal forecast accuracy and encompassing for nested models. *Journal of Econometrics* 105(1), 85–110. | why equal-accuracy tests lean toward the smaller of two nested models |
| Clark, T. E. and West, K. D. (2007). Approximately normal tests for equal predictive accuracy in nested models. *Journal of Econometrics* 138(1), 291–311. | `po.eval.clark_west` |
| Dawid, A. P. (1984). Statistical theory: the prequential approach. *Journal of the Royal Statistical Society, Series A* 147(2), 278–292. | judging predictions made before each row is learned |
| Diebold, F. X. and Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business and Economic Statistics* 13(3), 253–263. | `po.eval.diebold_mariano` |
| Hinkley, D. V. (1971). Inference about the change-point from cumulative sum tests. *Biometrika* 58(3), 509–523. | the Page-Hinkley detector behind `drift` |
| Jarque, C. M. and Bera, A. K. (1980). Efficient tests for normality, homoscedasticity and serial independence of regression residuals. *Economics Letters* 6(3), 255–259. | `jarque_bera` |
| Kish, L. (1965). *Survey Sampling*. Wiley. | the effective sample size of unequal weights |
| Koenker, R. (1981). A note on studentizing a test for heteroscedasticity. *Journal of Econometrics* 17(1), 107–112. | the `n R²` form of `breusch_pagan` |
| Ljung, G. M. and Box, G. E. P. (1978). On a measure of lack of fit in time series models. *Biometrika* 65(2), 297–303. | `ljung_box` |
| Mincer, J. A. and Zarnowitz, V. (1969). The evaluation of economic forecasts. In *Economic Forecasts and Expectations*, ed. J. A. Mincer, 3–46. NBER. | the calibration regression |
| Newey, W. K. and West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica* 55(3), 703–708. | `se_coef_hac`, and the variance in `diebold_mariano` and `clark_west` |
| Page, E. S. (1954). Continuous inspection schemes. *Biometrika* 41(1/2), 100–115. | the CUSUM detector behind `drift` |
| Ramsey, J. B. (1969). Tests for specification errors in classical linear least-squares regression analysis. *Journal of the Royal Statistical Society, Series B* 31(2), 350–371. | `ramsey_reset` |
| Vovk, V., Gammerman, A. and Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer. | conformal prediction |
| White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity. *Econometrica* 48(4), 817–838. | `se_coef_hc0` |
