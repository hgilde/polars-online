<!-- kind: how-to -->
# Choose a learning rate

This how-to is for anyone who fits one of the models that take a step on
every row, `sgd`, `ftrl` or `pa`, and needs to choose how large each step
is. A larger step follows a change sooner, and a smaller one averages more
rows. It sets `sgd`'s `learning_rate`, `ftrl`'s `alpha`, and the `mode`
that caps `pa`'s step.

## Prerequisites

These steps need the library:

- polars-online installed, with numpy for the recipes. Each recipe builds
  its own data and saves it to a parquet file.
- The parameters of
  [`po.spec.sgd`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.sgd),
  [`po.spec.ftrl`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.ftrl)
  and [`po.spec.pa`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.pa),
  which their docstrings define. Each model's entry in MODELS.md gives its
  update equation: [`sgd`](../MODELS.md#sgd--stochastic-gradient-descent),
  [`ftrl`](../MODELS.md#ftrl--online-logistic-regression) and
  [`pa`](../MODELS.md#pa--passive-aggressive-regression).

## Choose `sgd`'s learning rate

**To set how many rows the coefficients remember, choose `learning_rate`.**
Every row's step moves the coefficients and the half-life does not decay
them. So under a constant rate they remember about
`1 / (learning_rate · E[z²])` rows, whatever the clock between rows, where
`z` is the row's features as the step sees them.

1. **Decide how many rows the coefficients should remember.** Under
   `standardize`, the default, each feature is scaled to a standard
   deviation of 1, so `E[z²]` is about 1 and the memory is about
   `1 / learning_rate` rows. A memory of 100 rows is a `learning_rate` of
   0.01, the default.
2. **Give the spec that `learning_rate`.** Under the default
   `schedule="constant"` it stays the same on every row.
3. **Give the spec a `half_life` as well.** It decays
   [`weight_sum`](../GLOSSARY.md#warm-up-and-readiness), the feature
   scaler and, under `schedule="adagrad"`, the running sums of squared
   gradients that divide each step, not the coefficients. So under
   `"adagrad"` the step opens up again after a long gap.

The recipe plants a slope that jumps from 1 to 3, and counts the rows each
rate takes to close 63% of the jump, `1 − 1/e` of it, beside
`1 / learning_rate`. This code builds its own data:

```python
import math

import numpy as np
import polars as pl
import polars_online as po

# y = slope·x plus noise, where the slope jumps from 1 to 3 at row 3,000.
rng = np.random.default_rng(3)
n = 6_000
x, e = rng.standard_normal((2, n))
slope = np.where(np.arange(n) < 3_000, 1.0, 3.0)
pl.DataFrame({"x": x, "y": slope * x + 0.3 * e}).write_parquet("jump.parquet")

# One sgd spec per learning rate, each writing its coefficients on every row.
rates = [0.003, 0.01, 0.03]
specs = [po.spec.sgd(f"lr_{r:g}", targets=["y"], features=["x"], half_life=float("inf"),
                     learning_rate=r, coef_every=0) for r in rates]
out = pl.scan_parquet("jump.parquet").online.fit_predict(specs).collect()

# The rows after the jump until the slope has covered 1 - 1/e of it.
for r in rates:
    path = out[f"lr_{r:g}"].struct.field("coef").list.get(1).to_numpy()[3_000:]
    rows = int(np.argmax(path >= 1.0 + 2.0 * (1 - math.exp(-1))))
    print(f"learning_rate {r:<5g}: {rows:>3} rows to cover 63% of the jump (1 / learning_rate = {1 / r:.0f})")
```

```text
learning_rate 0.003: 324 rows to cover 63% of the jump (1 / learning_rate = 333)
learning_rate 0.01 :  82 rows to cover 63% of the jump (1 / learning_rate = 100)
learning_rate 0.03 :  28 rows to cover 63% of the jump (1 / learning_rate = 33)
```

Each rate follows the jump within about `1 / learning_rate` rows. A rate
that remembers fewer rows follows a change sooner, and moves more with
each row's noise. [The learning rate](../CALIBRATION.md#the-learning-rate)
chooses the rate from the stream's own out-of-sample errors, with
`po.eval.grid_search`.

## Choose `ftrl`'s `alpha`

`alpha` is `ftrl`'s learning-rate scale, in a coefficient's units: the
target (log-odds, under the logistic loss) per unit of the feature. Its
default is 0.1.

**Under a half-life, `ftrl` follows a break at the pace of its steps, not
of its half-life.** Each coefficient steps by its gradient over the root of
its recent squared gradients. Once those sums hold a break's large
gradients, a coefficient closes on the break at about `2·alpha·√(1 − f)` a
row, however large the error. Here `f` is the decay factor over one row's
clock, `0.5 ** (1 / half_life)` at one row a clock unit. Just after the
break, before the sums have grown, a coefficient moves faster, by up to
about `alpha` in one row. The
[`ftrl` docstring](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.ftrl)'s
Notes, under *Breaks*, measure how far each `alpha` stays from the
decayed least-squares fit after a break.

1. **Estimate the largest break to follow, `D`, in the coefficient's
   units.**
2. **Choose the rows you can wait to follow it.**
3. **Give `alpha` the value `D / (2 · rows · √(1 − f))` as a first
   estimate.** At `half_life=200` and one row a clock unit, `√(1 − f)` is
   0.059, so at the default `alpha` the estimate for a break of 20 is
   1,700 rows.
4. **Count the rows on a planted break, as the recipe does, and increase
   `alpha` if they are too many.** The estimate is not a bound. Over
   breaks of 2 and 20 in the intercept, at half-lives of 200 and 1,000
   and `alpha` from 0.1 to 1, the rows to cover 90% of a break came to
   0.37 to 1.54 times the estimate.

A larger `alpha` follows a break sooner, and moves the coefficients more
from row to row at their steady state.

The recipe plants a level that jumps by 20, and prints each `alpha`'s rows
to cover 90% of the jump beside the estimate for the whole jump,
`D / (2·alpha·√(1 − f))`. Beside them, it prints the intercept's standard
deviation over the 2,000 rows before the jump, where the truth holds still
at 0. This code builds its own data:

```python
# y = level + x plus noise, where the level jumps from 0 to 20 at row 4,000.
rng = np.random.default_rng(4)
n = 8_000
x, e = rng.standard_normal((2, n))
level = np.where(np.arange(n) < 4_000, 0.0, 20.0)
pl.DataFrame({"t": np.arange(n, dtype=float), "x": x, "y": level + x + 0.3 * e}).write_parquet("level.parquet")

# One ftrl spec per alpha, under the squared loss and a half-life of 200 rows.
alphas = [0.1, 0.3, 1.0]
specs = [po.spec.ftrl(f"alpha_{a:g}", targets=["y"], features=["x"], loss="squared", clock="t", gap_cap=10.0,
                      half_life=200.0, alpha=a, coef_every=0) for a in alphas]
out = pl.scan_parquet("level.parquet").online.fit_predict(specs).collect()

# Rows to cover 90% of the jump, beside the pace's estimate, and the spread at steady state.
root = math.sqrt(1 - 0.5 ** (1 / 200))                 # √(1 − f), f the decay factor over one row
for a in alphas:
    intercept = out[f"alpha_{a:g}"].struct.field("coef").list.get(0).to_numpy()
    rows = int(np.argmax(intercept[4_000:] >= 18.0))
    print(f"alpha {a:<3g}: {rows:>4} rows to cover 90% of the jump (D / (2·alpha·√(1 − f)) = {20 / (2 * a * root):.0f}), "
          f"steady-state spread {intercept[2_000:4_000].std():.3f}")
```

```text
alpha 0.1: 1895 rows to cover 90% of the jump (D / (2·alpha·√(1 − f)) = 1700), steady-state spread 0.037
alpha 0.3:  716 rows to cover 90% of the jump (D / (2·alpha·√(1 − f)) = 567), steady-state spread 0.060
alpha 1  :  165 rows to cover 90% of the jump (D / (2·alpha·√(1 − f)) = 170), steady-state spread 0.109
```

Each `alpha` covers 90% of the jump in about the rows the estimate gives
for the whole of it: 1,895 against 1,700 at the default. The intercept's
spread at its steady state grows with `alpha`. [Choose `ftrl`'s
penalties](choose-ftrl-penalties.md) sets the penalties that shrink the
same fit.

## Choose `pa`'s `mode`

`pa` takes, on each row, the smallest step that brings the fit within a
margin of the row's target. `mode` sets how far one row may move it, and
`c` sets the cap under `"pa1"` and the damping under `"pa2"`.

**Where outliers are possible, keep a `mode` that caps or damps the
step:**

| `mode` | the step |
|---|---|
| `"pa"` | moves the fit as far as one bad row demands |
| `"pa1"`, the default | is capped at `c` |
| `"pa2"` | is damped by `c` instead of capped |

1. **Keep `mode="pa1"`, the default, where a row may be wild.** Its cap
   `c` is on the step size `τ`, in the target's units over the units of
   `q`, the row's squared length. `q` has no units under `standardize`,
   the default, so there `c` is in the target's units alone, and the
   default `c=1` caps a step at one unit of the target per unit of `q`.
2. **Or give `mode="pa2"` with a small `c`.** Its damping `1 / (2c)` is
   added to `q`, so it is free of the target's units, and a smaller `c`
   damps more. A damped step is `τ = loss / (q + 1/(2c))`, with `loss` how
   far the prediction falls outside the row's margin. So it still grows
   with the row's residual, where `"pa1"`'s cap does not.
3. **Give `mode="pa"` only where no row can be wild.** It reads no `c`,
   and refuses one.

The recipe plants one row whose target is 500 too high, and prints each
mode's intercept on the row before it and on the row itself, beside the
planted 1. It then prints each mode's mean squared error over the next 100
rows, beside the noise's variance, 0.09. This code builds its own data:

```python
# y = 1 + 2·x0 − x1 plus noise of variance 0.09, with one row 500 too high.
rng = np.random.default_rng(7)
n = 5_000
x0, x1, e = rng.standard_normal((3, n))
y = 1.0 + 2.0 * x0 - 1.0 * x1 + 0.3 * e
y[2_500] += 500.0
pl.DataFrame({"x0": x0, "x1": x1, "y": y}).write_parquet("spike.parquet")

# The same pa under each mode.
common = dict(targets=["y"], features=["x0", "x1"], half_life=float("inf"), coef_every=0)
specs = [
    po.spec.pa("pa", mode="pa", **common),             # no cap
    po.spec.pa("pa1", **common),                       # the default: capped at c=1
    po.spec.pa("pa2", mode="pa2", **common),           # damped by 1/(2c) at c=1
    po.spec.pa("pa2_small", mode="pa2", c=0.01, **common),
]
out = pl.scan_parquet("spike.parquet").online.fit_predict(specs).collect()

# The intercept around the bad row, and the error on the rows after it.
for s in ("pa", "pa1", "pa2", "pa2_small"):
    intercept = out[s].struct.field("coef").list.get(0)
    after = (out[s].struct.field("resid_y")[2_501:2_601] ** 2).mean()
    print(f"{s:>9}: intercept {intercept[2_499]:.2f} before the bad row, {intercept[2_500]:>6.2f} after it "
          f"(planted 1); mse over the next 100 rows {after:>7.2f} (noise 0.09)")
```

```text
       pa: intercept 1.09 before the bad row, 484.70 after it (planted 1); mse over the next 100 rows 6927.40 (noise 0.09)
      pa1: intercept 1.09 before the bad row,   2.09 after it (planted 1); mse over the next 100 rows    0.17 (noise 0.09)
      pa2: intercept 1.09 before the bad row, 326.66 after it (planted 1); mse over the next 100 rows 3129.07 (noise 0.09)
pa2_small: intercept 1.03 before the bad row,  10.80 after it (planted 1); mse over the next 100 rows   27.34 (noise 0.09)
```

One bad row moves the uncapped fit by nearly the whole of its error, and
the fit takes many rows to come back. `"pa1"`'s cap holds the move to
about one unit of the target. `"pa2"` at the default `c` damps the step
only a little. At `c=0.01` it damps it to about 10, and the small steps
that follow bring the fit back slowly.

## Check the result

**Read each coefficient's path around a change you know of.** With
`coef_every=0`, the output's `coef` field holds the coefficients after
every row, as the recipes read them. A step too small leaves the path
behind a change for many rows, and a step too large makes it wander at
its steady state or jump at one bad row.

**Compare the settings' out-of-sample errors on the same rows.** Run one
spec per value in one bank, and score them with `po.eval.grid_search`, as
[The learning rate](../CALIBRATION.md#the-learning-rate) does for `sgd`'s
`learning_rate`, and the same way for `pa`'s `c` and `ftrl`'s `alpha`.
