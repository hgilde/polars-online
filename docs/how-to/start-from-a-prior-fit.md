<!-- kind: how-to -->
# Start from a prior fit

This how-to is for anyone who fits `ewridge` and has a belief about its
coefficients before the first row, such as yesterday's fit. You give the
belief as `coef_prior`, and choose with `ridge_scale` whether its pull
fades as the rows arrive or lasts for good.

## Prerequisites

These steps need the library, an `ewridge` spec and the coefficients to
start from:

- polars-online and numpy installed. The recipe generates its own data.
- An `ewridge` [spec](../GLOSSARY.md#the-model-bank-and-what-it-reads).
  For a prior that fades, a spec of one ridge value and one feature set:
  `ridge_scale="sum"` is refused beside a grid of `ridge` values or
  feature sets, beside `standardize` and beside `window_size`.
- The prior: one vector per target, in the features' original units, the
  intercept first and then a coefficient per feature (`len(features) + 1`
  long when there is an intercept). For a target `y`,
  `[yesterday_bank.coef().filter(pl.col("target") == "y")["coef"].to_list()]`
  gives them in that order.

`ridge`, `ridge_scale` and `coef_prior` are defined in [`po.spec.ewridge`'s
docstring](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.ewridge),
and [`ewridge`'s update](../MODELS.md#the-update) gives the two systems a
[solve](../GLOSSARY.md#fitting-a-model) reads.

## Start from a belief that fades

Use this to begin at yesterday's fit and let today's rows take over.

1. **Give the prior as `coef_prior`,** as `coef_prior=[[0.5, 2.0]]` for
   one target with an intercept and one feature.
2. **Give `ridge_scale="sum"`.** The ridge is then multiplied by every
   decay the stream applies, and set beside the rows' decayed sums, as
   [`rls`](../MODELS.md#rls--recursive-least-squares) sets its prior. So
   its pull fades as rows arrive. It pulls every coefficient toward
   `coef_prior`, the intercept included, until the evidence takes over.
3. **Give `ridge` the prior's strength.** On features of unit scale, at
   the first row it pulls about as hard as `ridge` rows that agree with the
   prior would. Its pull is multiplied by every decay after that, so it
   halves each half-life while the rows' weight grows.

## Hold the fit near a belief for good

Use this to keep the slopes near `coef_prior`, or near zero without one,
however many rows arrive.

1. **Give the prior as `coef_prior`, and leave `ridge_scale` at its
   default, `"mean"`.** The [Gram](../GLOSSARY.md#fitting-a-model)'s sums are means, so under `"mean"` the
   ridge is a penalty per observation that never fades. It never penalizes
   the intercept, so the intercept's entry of `coef_prior` is not read.
2. **Give `ridge` the strength of the pull,** in the features' squared
   units.

This code fits today's rows, whose slope is 2.2, three ways: with no
prior, with yesterday's slope of 2.0 as a prior that fades, and with the
same prior held for good. Each solves on every row, and `min_weight=0`
lets each predict from its first row:

```python
import numpy as np

# Today's truth: y = 0.5 + 2.2 x plus noise. Yesterday's fit was 0.5 + 2.0 x.
rng = np.random.default_rng(11)
n = 2_000
x = rng.standard_normal(n)
y = 0.5 + 2.2 * x + 1.0 * rng.standard_normal(n)
pl.DataFrame({"t": np.arange(n, dtype=float), "x": x, "y": y}).write_parquet("today_prior.parquet")
yesterday = [[0.5, 2.0]]   # one vector per target: the intercept, then a coefficient per feature

# No prior, a prior that fades, and a prior held for good, each worth about 20 rows at the start.
common = dict(targets=["y"], features=["x"], clock="t", gap_cap=10.0, half_life=500.0,
              coef_every=0, max_rows_between_solves=1, min_weight=0)
specs = [
    po.spec.ewridge("no_prior", **common),
    po.spec.ewridge("fading", coef_prior=yesterday, ridge_scale="sum", ridge=20.0, **common),
    po.spec.ewridge("lasting", coef_prior=yesterday, ridge=20.0, **common),
]
out = pl.scan_parquet("today_prior.parquet").online.fit_predict(specs).collect()

# Each fit's slope as the rows arrive, and its error over the first 20 rows.
print("today's slope 2.2, yesterday's 2.0")
for spec in specs:
    fields = out[spec["name"]].struct.unnest()
    s = fields["coef"].list.get(1)
    first = float(((fields["pred_y"] - out["y"]) ** 2).head(20).mean())
    print(f"{spec['name']}: slope after 3 rows {s[2]:.2f}, 50 rows {s[49]:.2f}, 2,000 rows {s[-1]:.2f}; error over the first 20 rows {first:.2f}")
```

```text
today's slope 2.2, yesterday's 2.0
no_prior: slope after 3 rows 2.47, 50 rows 2.20, 2,000 rows 2.19; error over the first 20 rows 1.32
fading: slope after 3 rows 2.02, 50 rows 2.13, 2,000 rows 2.19; error over the first 20 rows 1.03
lasting: slope after 3 rows 2.01, 50 rows 2.01, 2,000 rows 2.01; error over the first 20 rows 1.09
```

With no prior, the first rows' slope is noise: 2.47 after three rows. The
prior that fades starts at yesterday's 2.0 and reaches today's slope as the
rows outweigh it. The prior held for good keeps the slope at 2.01, where a
ridge of 20 per row outweighs a slope that moved by 0.2.

## Check the result

**Read the slope at a few rows, as the recipe prints, with `coef_every=0`,
which writes the coefficients on every row.** A prior that fades starts at
`coef_prior` and ends at the fit with no prior. A prior held for good stays
near `coef_prior`.

| what you see | what it means | what to do |
|---|---|---|
| the fit never leaves the prior | `ridge_scale` is `"mean"`, whose penalty never fades | give `ridge_scale="sum"` |
| the prior has no effect on the first rows | `ridge` is small beside the rows' weight | give a larger `ridge`, the prior's weight in rows |
| the intercept ignores its prior | under `"mean"` the intercept is not penalized | give `ridge_scale="sum"`, the one solve that penalizes the intercept |
| `ValueError` naming `ridge_scale` | `"sum"` beside `standardize`, a grid of `ridge` values or feature sets, or `window_size` | give one ridge value and one feature set, without `standardize` or `window_size` |
| `ValueError` naming `coef_prior` | the wrong number of vectors, a vector of the wrong length, or a value that is not finite | give one finite vector per target, `len(features) + 1` long with an intercept |
