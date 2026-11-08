<!-- kind: how-to -->
# Bound the coefficients

This how-to is for anyone who needs a regression's slopes held to bounds
or to a fixed total, as portfolio weights are. It bounds the slopes of
`sgd` or `pa`, the two models that take bounds, with `coef_min`,
`coef_max` and `coef_sum`.

## Prerequisites

These steps need the library:

- polars-online installed, and the README's
  [Example data](../../README.md#example-data) built. The first recipe
  reads `df`, 400 rows whose `signal_a` and `signal_b` are columns of
  noise, and whose `y` is `1 + 2·x0 − x1` plus noise. The second builds
  its own data, with numpy.
- A model that takes bounds: `sgd`, stochastic gradient descent, or `pa`,
  passive-aggressive regression. No other model takes `coef_min`,
  `coef_max` or `coef_sum`. Their docstrings,
  [`po.spec.sgd`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.sgd)
  and [`po.spec.pa`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.pa),
  define each parameter.

## Bound the slopes of `sgd`

After each step, `sgd` replaces the slopes by the nearest point of the
set the bounds allow, so the [coefficients](../GLOSSARY.md#fitting-a-model)
it reports meet the bounds after every learned row. The intercept is
always free. The bounds hold under any of `sgd`'s losses and any of its
three schedules, which
[`sgd`'s entry](../MODELS.md#sgd--stochastic-gradient-descent) in
MODELS.md lists.

**To bound the slopes, give `coef_min` and `coef_max` (one number, or one
per feature, with `-inf` or `inf` for an open side) and `coef_sum` for
their total:**

| setting | what it expresses |
|---|---|
| `coef_min=0, coef_sum=1` | a long-only, fully invested portfolio |
| `coef_min=0` alone | a sign the model must respect |
| `coef_min` equal to `coef_max` | a slope pinned at a known value |

1. **Give `coef_min` and `coef_max`,** one number for every slope or a
   list with one entry per feature.
2. **Give `coef_sum` for the slopes' total,** where they must add up to a
   constant.
3. **Fit with `bank.fit_predict(...)`, or with a query's
   `online.fit_predict(...)`.** Each row's `coef` field holds the bounded
   coefficients, the intercept first.

This code uses `df` from [Example data](../../README.md#example-data):

```python
portfolio = po.spec.sgd(
    "portfolio", targets=["y"], features=["signal_a", "signal_b", "x0"], half_life=200.0,
    loss="squared",              # any loss sgd takes
    learning_rate=0.01,
    schedule="constant",         # or "inv_scaling", or "adagrad"
    clip_gradient=1e3,           # the default
    coef_min=0.0,                # every slope at least 0 ...
    coef_sum=1.0,                # ... and the slopes summing to 1
    coef_every=0,
)
fit = po.ModelBank([portfolio]).fit_predict(df)
last = fit["portfolio"].struct.field("coef").drop_nulls()[-1]
assert min(last[1:]) >= 0.0 and abs(sum(last[1:]) - 1.0) < 1e-12   # the last row's slopes: at least 0, summing to 1
```

## Bound the slopes of `pa`, with a small `c`

`pa` takes the same three bounds, exactly as `sgd` does. It moves the fit
by the smallest step that brings each row within a margin of its target,
capped by `c` under its default `mode="pa1"`.

**When `coef_min`, `coef_max` or `coef_sum` bound the slopes, keep `c`
small.** The projection that follows each step keeps the step from
meeting the row's margin exactly, and a truth outside the allowed set is
never reached, so the fit keeps stepping against the bounds. A small `c`
keeps those steps small.

1. **Give `pa` the bounds as you would give them to `sgd`.**
2. **Give it a small `c`,** such as 0.01, in place of the default 1.

The recipe plants two features whose true weights, 0.9 and 0.6, sum to
1.5. Under `coef_sum=1` that truth is outside the allowed set, so the fit
never reaches it. The closest weights the set allows, by least squares,
are 0.65 and 0.35: each true weight less half the excess of 0.5. The
recipe prints them beside each fit's last weights, and how far `a`'s
weight wanders over the last 5,000 rows, its standard deviation there.
This code builds its own data:

```python
import numpy as np
import polars as pl
import polars_online as po

# y = 0.9·a + 0.6·b plus noise: weights that sum to 1.5, outside a sum of 1.
rng = np.random.default_rng(12)
n = 20_000
a, b, e = rng.standard_normal((3, n))
pl.DataFrame({"a": a, "b": b, "y": 0.9 * a + 0.6 * b + 0.3 * e}).write_parquet("weights.parquet")

# pa with the slopes at least 0 and summing to 1, at the default c and at a small one, and sgd beside them.
bounds = dict(targets=["y"], features=["a", "b"], half_life=float("inf"), coef_min=0.0, coef_sum=1.0, coef_every=0)
specs = [po.spec.pa("pa_c1", **bounds), po.spec.pa("pa_c001", c=0.01, **bounds), po.spec.sgd("sgd", **bounds)]
out = pl.scan_parquet("weights.parquet").online.fit_predict(specs).collect()

# The last weights, and how far a's weight wanders over the last 5,000 rows.
print("closest allowed: a 0.650, b 0.350")
for name in ("pa_c1", "pa_c001", "sgd"):
    coef = out[name].struct.field("coef")
    _, w_a, w_b = coef[-1]
    wander = coef.list.get(1)[-5_000:].std()
    print(f"{name:>7}: a {w_a:.3f}, b {w_b:.3f}, sum {w_a + w_b:.3f}; a's weight wanders by {wander:.3f}")
```

```text
closest allowed: a 0.650, b 0.350
  pa_c1: a 0.811, b 0.189, sum 1.000; a's weight wanders by 0.151
pa_c001: a 0.722, b 0.278, sum 1.000; a's weight wanders by 0.038
    sgd: a 0.683, b 0.317, sum 1.000; a's weight wanders by 0.028
```

Every fit keeps its weights summing to 1. At the default `c`, `pa`'s
weights wander by 0.15 as it steps against the bounds. At `c=0.01` they
wander a quarter as far.

## Check the result

**Read the coefficients after each row, and check the bounds hold.**
With `coef_every=0`, each row's `coef` field holds the coefficients after
it, the intercept first. Every slope stays between its `coef_min` and
`coef_max`, and under `coef_sum` the slopes add up to it, as the first
recipe's `assert` checks on its last row.
