<!-- kind: how-to -->
# Check a two-stage fit

This how-to is for anyone whose answer is a second-stage coefficient, such
as how fast a price reverts to its fair value, fitted on a first stage's
out-of-sample residual. You will turn on the diagnostics each stage asks
for, fit two comparison specs beside it, and read the coefficient beside
one you planted.

## Prerequisites

The recipe builds its own data, so these steps need only:

- polars-online installed, and numpy, which the recipe uses to make up its
  rows.
- For your own data, a two-stage pipeline: a first-stage
  [spec](../GLOSSARY.md#the-model-bank-and-what-it-reads) whose residual
  `resid_<target>` is a feature of a second, as [Fit on factor
  residuals](fit-on-factor-residuals.md) builds one.
- The second stage's target window in rows, when its target looks ahead.

[Checking a fit stage by
stage](../explanation/how-diagnostics-work.md#checking-a-fit-stage-by-stage)
explains why each stage asks for these diagnostics: the residual's own
error attenuates the coefficient, and a target that looks ahead makes its
residuals overlap.

## Check each stage with the diagnostics it asks for

Use these steps when a second stage's coefficient is the answer you want
and you need to know whether to trust it.

1. **Give the second stage `emit_calibration=True`.** Its slope
   `calibration_slope_<target>` says whether the predictions, and so the
   coefficient that made them, are scaled right: a slope of 1 needs no
   rescaling. Under `half_life=float("inf")` the calibration runs once and
   keeps the rows from [readiness](../GLOSSARY.md#warm-up-and-readiness)
   on, the early fits' included; give
   `calibration_half_life` (such as `"4h"`) to read it past them.
2. **Give it `emit_se_coef=True` and `emit_robust_se=True`, with
   `horizon_rows` the target's window in rows.** `se_coef_hac` is
   [Newey and West's](check-standard-errors.md) standard error, which
   allows for the overlap of targets that look ahead. Under `horizon_rows`
   the calibration's Wald statistic (its test of slope 1 and intercept 0)
   takes Newey and West's variance too.
3. **Fit two more second-stage specs beside the two-stage one:** `joint`,
   on the first stage's target and its features, in place of both stages;
   and `both`, on the residual beside those features. The first says
   whether the stages change the answer, and the second whether the
   features add anything once the residual is in.
4. **Read each fit's last row that holds its standard errors,** and each
   coefficient's t by `se_coef_hac` as well as by `se_coef`.
5. **Compare the coefficient with the answer you planted.**

This code estimates how fast a price reverts to its fair value. Stage 1
fits the fair value. Each series becomes its distance from its own
10-minute exponentially weighted mean, and the mid's distance, `A`, is
regressed on the reference series' distances `x1` to `x3`. Its out-of-sample
residual, `resid_A`, is the mid less its fair value, known at the row.
Stage 2 regresses the distance from the mid to its exponentially weighted
mean over the next 150 seconds, at a 30-second half-life, on `resid_A`.
Its coefficient is minus the share of the gap to fair value that closes. The
mid is an efficient price, a random walk the reference series follow, plus a
transient `u` that halves every 30 seconds and that no reference series sees. Over a
150-second window weighted at a 30-second half-life, `u`'s expected share
that reverts is 0.490, so the planted coefficient on `u` itself is
−0.490. The code builds its own rows:

```python
from datetime import datetime, timedelta

import numpy as np
import polars as pl

import polars_online as po

# A quote a second for 17 hours. mid = p + u: p is a random walk (the efficient price) and u a
# transient that halves every 30 s. Three reference series s1-s3 follow p, each with noise of its own; none
# of them sees u.
rng = np.random.default_rng(12)
n = 60_000
p = 100 + np.cumsum(rng.normal(0, 0.01, n))
# The transient u, which halves every 30 s.
phi = 0.5 ** (1 / 30)
u = np.zeros(n)
shocks = rng.normal(0, 0.05 * np.sqrt(1 - phi**2), n)
for i in range(1, n):
    u[i] = phi * u[i - 1] + shocks[i]
# Three reference series that follow p with noise of their own, and a quote a second.
s = np.array([1.0, 0.8, 1.2])[:, None] * p + np.cumsum(rng.normal(0, 0.002, (3, n)), axis=1)
ts = [datetime(2026, 1, 5) + timedelta(seconds=i) for i in range(n)]
pl.DataFrame({"ts": ts, "mid": p + u, "s1": s[0], "s2": s[1], "s3": s[2]}).write_parquet("quotes.parquet")

# Stage 1, fair value: each series less its own 10-minute EWMA, and A (the mid's) regressed on the
# reference series'. Its out-of-sample residual resid_A is the mid less its fair value, known at the row.
clock = dict(clock="ts", gap_cap="5m")
back = dict(half_life="10m", window_size="60m")
windows = pl.scan_parquet("quotes.parquet").online.with_windows(
    A=pl.col("mid") - po.ewm_mean("mid", **back),
    x1=pl.col("s1") - po.ewm_mean("s1", **back),
    x2=pl.col("s2") - po.ewm_mean("s2", **back),
    x3=pl.col("s3") - po.ewm_mean("s3", **back),
    **clock,
)
fair = po.spec.ewridge("fair", targets=["A"], features=["x1", "x2", "x3"], half_life="2h", **clock)

# Stage 2, reversion: the next 150 s's 30-second EWMA of the mid, less the mid, regressed on
# resid_A. Each row is learned once its window has closed (embargo), and its residuals overlap.
fwd = (po.ewm_mean("mid", half_life="30s", window_size="150s", reverse=True) - pl.col("mid")).alias("B")
settings = dict(
    targets=[fwd], half_life=float("inf"), embargo="150s", min_weight=1_000.0,
    emit_se_coef=True, emit_robust_se=True,                # 2.
    horizon_rows=150,  # the window in rows: Newey-West to twice it, for se_coef_hac and the calibration
    emit_calibration=True,  # 1. B on pred: a slope of 1 says the coefficient needs no rescaling
    **clock,
)
revert = po.spec.ewridge("revert", features=["resid_A"], **settings)
joint = po.spec.ewridge("joint", features=["A", "x1", "x2", "x3"], **settings)  # 3. one stage in place of two
both = po.spec.ewridge("both", features=["resid_A", "x1", "x2", "x3"], **settings)  # 3. do the references add?

# Run both stages in one pass over the file.
out = (
    windows.online.fit_predict([fair])
    .online.unnest([fair])  # pred_A, resid_A
    .online.fit_predict([revert, joint, both])
    .collect()
)

# 4. Each stage-2 fit's calibration, and each coefficient's t by both standard errors.
for spec in [revert, joint, both]:
    name = spec["name"]
    last = out.select(pl.col(name).struct.unnest()).filter(pl.col("se_coef").is_not_null()).tail(1)
    coef, se, hac = (last[f].item().to_list() for f in ["coef", "se_coef", "se_coef_hac"])
    slope, wald = last.select("calibration_slope_B", "calibration_wald_B").row(0)
    print(f"{name}: calibration slope {slope:.2f}, wald {wald:.1f}")
    for term, b, s_, h in zip(spec["features"], coef[1:], se[1:], hac[1:]):
        print(f"  {term:8} coef {b:+.3f}, t {b / s_:+7.1f} by se_coef and {b / h:+6.1f} by se_coef_hac")

# 5. The planted coefficient: the forward EWMA of u over its 150 s window, as a share of u, less 1.
j = np.arange(1, 151)
w = 0.5 ** (j / 30)
print(f"planted coefficient on u itself: {(w * phi**j).sum() / w.sum() - 1:.3f}")
```

```text
revert: calibration slope 0.99, wald 1.1
  resid_A  coef -0.450, t  -107.7 by se_coef and  -15.8 by se_coef_hac
joint: calibration slope 0.80, wald 8.3
  A        coef -0.444, t  -105.4 by se_coef and  -15.5 by se_coef_hac
  x1       coef +0.124, t   +27.7 by se_coef and   +3.0 by se_coef_hac
  x2       coef -0.011, t    -2.0 by se_coef and   -0.2 by se_coef_hac
  x3       coef +0.258, t   +57.3 by se_coef and   +6.1 by se_coef_hac
both: calibration slope 0.84, wald 5.4
  resid_A  coef -0.446, t  -104.3 by se_coef and  -15.6 by se_coef_hac
  x1       coef +0.029, t    +6.8 by se_coef and   +0.7 by se_coef_hac
  x2       coef -0.083, t   -15.8 by se_coef and   -1.7 by se_coef_hac
  x3       coef +0.021, t    +5.1 by se_coef and   +0.5 by se_coef_hac
planted coefficient on u itself: -0.490
```

Each check reads one part of the answer:

| check | what it read | what it says |
|---|---|---|
| the coefficient | `revert`'s coefficient on `resid_A`, −0.450 | close to the −0.490 planted on `u`, and attenuated toward zero by the fair value's own error in `resid_A` |
| calibration | `revert`'s slope 0.99 | the predictions `−0.450 · resid_A` are scaled right, so the coefficient needs no rescaling |
| the Wald beside it | `revert`'s Wald, 1.1 | under `horizon_rows=150` it takes Newey and West's variance at 300 lags, so the overlap of the 150 rows that share each window does not inflate it |
| Newey-West | t of −107.7 by `se_coef`, −15.8 by `se_coef_hac` | the coefficient is real either way, but the plain t is 6.8 times too large: each residual shares its 150-second window with 149 others |
| `joint` | one stage, `A` and `x1` to `x3` together: `A`'s coefficient −0.444, calibration slope 0.80 | `A`'s coefficient matches `resid_A`'s, since `x1` to `x3` absorb the fair value. `joint` fits five coefficients, and its first hours of predictions are scaled worse than `revert`'s. The run-once calibration still holds their rows, so the 0.80 is theirs as much as the fit's: from eight hours after its first prediction on, the same slope reads 0.99 |
| `both` | `resid_A` beside `x1` to `x3`: their t are +6.8, −15.8 and +5.1 by `se_coef`, and +0.7, −1.7 and +0.5 by `se_coef_hac` | once `resid_A` is in, `x1` to `x3` add nothing that Newey and West's standard error believes. By `se_coef` alone, all three would look significant |

**The early rows were measured, on 2026-11-08.** Each row's target was
computed from the mid, and the target regressed on each stage-two
prediction by least squares over each two-hour stretch of the stream.
Over the first six hours, `joint`'s slope read 0.54, 0.57 and 0.75, where
`revert`'s read 0.65, 0.92 and 1.10. From the sixth hour on, the two read
within 0.14 of each other. Over every scored row, `joint`'s slope is
0.80, the calibration's figure, and from eight hours after its first
prediction on it is 0.99.
A two-hour stretch is a noisy read: `revert`'s ranged from 0.65 to 1.41,
so read the calibration over many hours.

[Checking a fit stage by
stage](../explanation/how-diagnostics-work.md#checking-a-fit-stage-by-stage)
gives the reasons behind each reading: why a residual feature attenuates,
and why a run-once calibration keeps the early rows.

## Check that it worked

**The two-stage coefficient is within 1.5 of its `se_coef_hac` of the
planted value, and its calibration says it needs no rescaling.** Every t
on a target that looks ahead needs `se_coef_hac`: `se_coef` would have
kept three features that carry nothing.

| what you see | what it means | what to do |
|---|---|---|
| a calibration slope far from 1 under `half_life=float("inf")` | the run-once calibration still holds the early fits' rows | give `calibration_half_life`, such as `"4h"`; if the slope stays far from 1 past the warm-up, rescale ([Check the scale of a model's predictions](check-the-scale.md)) |
| a t by `se_coef` far larger than by `se_coef_hac` | the target looks ahead, so its residuals overlap | read `se_coef_hac`, with `horizon_rows` the target's window in rows ([Check a coefficient's standard error](check-standard-errors.md)) |
| `se_coef` and `se_coef_hac` null on the early rows | the model is not yet [ready](../GLOSSARY.md#warm-up-and-readiness), here under its `min_weight`. Beside a fit that forgets, they also wait until the stream has come 95% of the way to its steady state | read the last row that holds them, as the recipe does |
