<!-- kind: how-to -->
# Check that a fit is determined

This how-to is for anyone who fits least squares from a
[Gram](../GLOSSARY.md#fitting-a-model) and wants to know whether nearly
dependent columns leave its coefficients undetermined. You read Belsley's
condition indexes and the variance inflation factors from the Gram, find
the columns involved, and refit without one of them.

## Prerequisites

These steps need the library and a Gram:

- polars-online installed with numpy, an optional extra, which
  `pip install "polars-online[numpy]"` adds. The recipe generates its own
  data.
- A Gram, as `bank.gram(name)` returns it for an `ewridge`, `lasso` or
  `ew_cov` spec, or one pooled from blocks ([Pool
  Grams](pool-grams.md)).
- [Is the fit determined?](../GRAM.md#is-the-fit-determined) defines the
  variance inflation factor and Belsley's condition indexes, and says how
  to read each.

## Find the columns that leave a fit undetermined

Use this before you read a fit's coefficients, whenever its columns may
be nearly dependent.

1. **Read Belsley's diagnostics with `po.gram.condition(g)`.** It returns
   the `condition_indexes`, and the `proportions`: each column's share of
   its coefficient's variance in each component, with the intercept among
   the `columns`.
2. **Look for an index above 30 whose component holds more than half the
   variance of two or more columns.** Those columns are the dependency.
3. **Read each column's variance inflation factor with
   `po.gram.vif(g)`.** A VIF of 10 makes a coefficient's standard error
   3.2 times as wide as it would be with its column uncorrelated with the
   rest. Read a VIF past 1e10 as an exact dependency.
   `po.gram.correlation(g)` gives the correlation matrix behind it.
4. **Refit on a subset that leaves one of the dependent columns out,**
   with `po.gram.subset(g, ["intercept", ...])` and `po.gram.solve`. Keep
   `"intercept"` among the columns, since a subset without it is fitted
   through the origin.
5. **Compare the t with and without that column,** with
   `po.gram.coef_stats`.

The recipe has five independent features and a sixth, `x5`, that is
`x1 + x2` plus a little noise of its own:

```python
import numpy as np
import polars as pl

import polars_online as po

# Five features, and a sixth that is nearly the sum of two of them: x5 = x1 + x2 + a little noise.
rng = np.random.default_rng(11)
n = 5_000
x = rng.standard_normal((n, 5))
x5 = x[:, 1] + x[:, 2] + 0.05 * rng.standard_normal(n)
y = x @ [0.5, 0.4, 0.3, 0.2, 0.1] + rng.standard_normal(n)
features = ["x0", "x1", "x2", "x3", "x4", "x5"]
pl.DataFrame({**dict(zip(features, [*x.T, x5])), "y": y}).write_parquet("near.parquet")

# One pass leaves one Gram of the six features.
spec = po.spec.ewridge(
    "fit", targets=["y"], features=features, half_life=float("inf"), solve_every=float("inf")
)
bank = po.ModelBank([spec])
bank.fit(pl.scan_parquet("near.parquet"))
g = bank.gram("fit")[0]

# 1 to 3. Belsley's indexes, each column's share of the last component, and the VIFs.
cond = po.gram.condition(g)  # Belsley's diagnostics, with the intercept among the columns
print("condition indexes: " + ", ".join(f"{v:.1f}" for v in cond["condition_indexes"]))
last = cond["proportions"][-1]  # each column's share of its variance in the last component
print("shares of the last: " + ", ".join(f"{c} {p:.2f}" for c, p in zip(cond["columns"], last)))
print("VIF: " + ", ".join(f"{c} {v:.0f}" for c, v in zip(features, po.gram.vif(g))))
# 4 and 5. The t with x5 in the fit, and without it.
for cols in (features, features[:5]):
    part = po.gram.subset(g, ["intercept", *cols])  # without "intercept", a fit through the origin
    t = po.gram.coef_stats(part, po.gram.solve(part))["t"]
    print(f"t with {len(cols)} features: " + ", ".join(f"{c} {v:.1f}" for c, v in zip(cols, t[1:])))
```

```text
condition indexes: 1.0, 1.4, 1.4, 1.4, 1.4, 1.5, 57.1
shares of the last: intercept 0.00, x0 0.00, x1 1.00, x2 1.00, x3 0.00, x4 0.00, x5 1.00
VIF: x0 1, x1 411, x2 399, x3 1, x4 1, x5 816
t with 6 features: x0 35.7, x1 0.3, x2 0.0, x3 14.8, x4 8.2, x5 1.1
t with 5 features: x0 35.7, x1 29.1, x2 22.2, x3 14.8, x4 8.2
```

One condition index, 57.1, stands far above the others, and its component
holds all of the variance of `x1`, `x2` and `x5`: the dependency, and its
three columns. Their VIFs are 399 to 816. With `x5` in the fit, the t of
`x1` and `x2` falls from 29.1 and 22.2 to 0.3 and 0.0, while `x0`, `x3`
and `x4` keep theirs.

## Check the result

**A fit is determined when no condition index stands above 30 with two or
more columns' shares above 0.5 in its component.** Run step 1 again on the
columns you keep. What else you may see, and what to do:

| what you see | what it means | what to do |
|---|---|---|
| one index above 30, with two or more columns' shares above 0.5 | those columns are nearly dependent, and only their combined effect is fixed | drop one of them, combine them into one column with Polars' `with_columns` in the query before the bank ([Data quality](../DATA-QUALITY.md#collinear)), or fit with a ridge, `po.gram.solve(g, ridge=..., standardize=True)` ([Fit a ridge from a Gram](fit-a-ridge-from-a-gram.md)) |
| a VIF past 1e10 | an exact dependency, which rounding left finite ([`vif`'s reference](https://hgilde.github.io/polars-online/gram.html#polars_online.gram.vif)) | drop a column of the dependency |
| the intercept's share above 0.5 in a component with a large index | a column's level is far above its spread, so it depends on the intercept | read the columns that share the component with it, as in the first row |
