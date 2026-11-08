<!-- kind: how-to -->
# Find what a model misses

This how-to is for anyone who wants to know what a linear model on a
stream leaves out: a lag, a spread that follows a feature, or a
curvature. It runs the specification
tests on the model's residuals, reads which one fires, and adds the
feature that fixes it.

## Prerequisites

These steps need the library and a linear model:

- polars-online installed, and numpy to make the example's data.
- A [spec](../GLOSSARY.md#the-model-bank-and-what-it-reads) of one of the
  ten linear models, such as `ewridge`, which take `emit_specification`
  and `emit_autocorr`. [What each costs, and which models take
  it](../DIAGNOSTICS.md#what-each-costs-and-which-models-take-it) lists
  them.
- For a [target](../GLOSSARY.md#the-model-bank-and-what-it-reads) that
  looks ahead: `horizon_rows=` (with a clock) or `embargo=` (without one),
  so that the tests allow for the overlap: `ljung_box` tests the lags
  past the horizon, and `breusch_pagan` and `ramsey_reset` take Newey and
  West's variance ([Can its t be
  trusted?](../DIAGNOSTICS.md#can-its-t-be-trusted)).

## Run the specification tests

A model's [residuals](../GLOSSARY.md#fitting-a-model) with nothing left to
learn look like noise: no lag, no spread that follows a feature, no
curvature. Each specification test looks for one of those shapes in the
[out-of-sample](../explanation/out-of-sample.md) residuals, and sees only
its own. [What is it missing?](../DIAGNOSTICS.md#what-is-it-missing)
defines each test and its distribution with nothing missing.

1. **Turn on `emit_specification=True` in the spec.** Each row gains
   `ljung_box_<t>`, `breusch_pagan_<t>` and `ramsey_reset_<t>` for each
   target `<t>`, at `specification_half_life`, the model's half-life by
   default.
2. **Run the query, and read from row 6,000 on.** The tests fold rows
   once the stream has come 95% of the way to its steady state, about 4.3
   of the fit's half-lives in, and their fields start at row 4,334 at a
   half-life of 1,000 rows.
3. **Read each statistic's median, and compare it with its 5% value:**
   18.3 for `ljung_box` at its default of 10 lags, `chi2(k)`'s for
   `breusch_pagan` with `k` features (5.99 for two), and 5.99 for
   `ramsey_reset`.
4. **Add the feature the test that fires asks for,** with Polars in the
   query before the bank, and run the tests again.

This code plants one fault in each of three streams, fixes two of them
with Polars before the bank, and reads each statistic's median over rows
6,000 to 9,999:

```python
import numpy as np
import polars as pl

import polars_online as po

# y = 1.0 x0 + 0.5 x1 + noise on every stream, but for one fault each.
rng = np.random.default_rng(5)
n = 10_000
x0, x1, e = rng.standard_normal((3, n))
ar = np.zeros(n)  # noise that carries 0.4 of the row before
for i in range(1, n):
    ar[i] = 0.4 * ar[i - 1] + e[i]
base = 1.0 * x0 + 0.5 * x1
# A clean stream and one stream per fault.
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
missing = pl.scan_parquet("missing.parquet").with_columns(
    # The fixes step 4 adds, made with Polars before the bank: the row before's target and features for the lag,
    # x0's square for the curvature.
    y_lag_prev=pl.col("y_lag").shift(1),
    x0_prev=pl.col("x0").shift(1),
    x1_prev=pl.col("x1").shift(1),
    x0_sq=pl.col("x0") ** 2,
)


# 1. A ridge with the three specification tests, on each stream and each fix.
def spec(name, target, features):
    return po.spec.ewridge(
        name, targets=[target], features=features, half_life=1_000.0,
        emit_specification=True,  # ljung_box (10 lags), breusch_pagan, ramsey_reset, at a memory of 1,000 rows
    )


specs = [
    spec("clean", "y_clean", ["x0", "x1"]),
    spec("lag", "y_lag", ["x0", "x1"]),
    spec("spread", "y_spread", ["x0", "x1"]),
    spec("curve", "y_curve", ["x0", "x1"]),
    spec("lag_fixed", "y_lag", ["x0", "x1", "y_lag_prev", "x0_prev", "x1_prev"]),
    spec("curve_fixed", "y_curve", ["x0", "x1", "x0_sq"]),
]
# 2. Run the query, past the fit's first 6,000 rows.
out = missing.online.fit_predict(specs).slice(6_000).collect()

# 3. Each statistic's median. With nothing missing: chi2(10) has a median of 9.3, and chi2(2), chi2(3)
# or chi2(5) of 1.4, 2.4 or 4.4.
print(f"{'spec':12} {'ljung_box':>10} {'breusch_pagan':>14} {'ramsey_reset':>12}")
for s in specs:
    name, y = s["name"], s["targets"][0]
    medians = out.select(pl.col(name).struct.unnest()).select(
        pl.col(f"ljung_box_{y}").median(),
        pl.col(f"breusch_pagan_{y}").median(),
        pl.col(f"ramsey_reset_{y}").median(),
    )
    print(f"{name:12} " + " ".join(f"{v:>{w}.1f}" for v, w in zip(medians.row(0), (10, 14, 12))))
```

```text
spec          ljung_box  breusch_pagan ramsey_reset
clean               8.0            1.0          0.7
lag               447.7            2.5          0.5
spread              4.9          337.6          1.8
curve               8.5            0.8        261.5
lag_fixed           8.0            3.0          1.0
curve_fixed         8.0            1.7          1.1
```

The clean stream reads 8.0, 1.0 and 0.7, each below its 5% value (18.3,
5.99 and 5.99). Each faulty stream lights one test, by a factor of 50 or
more over the clean stream. `ljung_box` reads 447.7 for the missing lag,
`breusch_pagan` 337.6 for the spread, and `ramsey_reset` 261.5 for the
curvature. With the row before's target and features added as features,
made with Polars' `shift(1)`, the lagged stream's `ljung_box` falls back
to 8.0. Its `breusch_pagan` of 3.0, now on five features, is below
`chi2(5)`'s 5% value of 11.07. With `x0²` as a feature, the curved
stream's `ramsey_reset` falls to 1.1. The spread has no fix in the
features.

## Fix what a test finds

Choose the fix by the test that fires:

| the test that fires | what the model misses | what to do |
|---|---|---|
| `ljung_box` | a lag in the residuals, or a half-life too long for a moving relationship | add the lag as a feature, made with Polars' `shift` in the query before the bank, or shorten the half-life. To read one lag as a correlation, turn on `emit_autocorr=True`: `autocorr_<t>` correlates each residual with the one `resid_autocorr_lag` predicted rows back (1 by default), and reads near 0 with nothing missing |
| `ramsey_reset` | a curvature | add a square, a product, or a transform of the feature that carries it, made with Polars' `with_columns` in the query before the bank |
| `breusch_pagan` | a spread that moves with a feature | keep the fit, whose coefficients are still right. On `ewridge` or `rls`, turn on `emit_robust_se=True` and read `se_coef_hc0` in place of `se_coef` ([Check a coefficient's standard error](check-standard-errors.md)). For a more efficient fit, on any model, weight each row by the inverse of its expected variance, in a column the spec names as `weight=` |

## Check that it worked

Run the tests again on the spec with its fix, as the recipe's `lag_fixed`
and `curve_fixed` do. A fixed model reads each median below its 5% value,
near the clean stream's. On streams missing nothing, each test passes its
5% value on 2.0% to 7.0% of rows, so a few rows past it are expected.
Where a test stays high:

| what you see | what it means | what to do |
|---|---|---|
| `ljung_box` high on a target that looks ahead | the residuals overlap by construction | give `horizon_rows=` or `embargo=`, so it tests lags past the horizon |
| `breusch_pagan` quiet, and a spread you suspect | the spread is symmetric in a feature, as `\|x0\|`, which it cannot see | read `ramsey_reset`, whose `pred²` carries `x0²` |
| `ramsey_reset` quiet, and a curvature you suspect | the curvature is in a feature the prediction barely uses | add the feature's square as a feature, as `curve_fixed` does |
