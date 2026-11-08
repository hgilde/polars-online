<!-- kind: how-to -->
# Fit targets that go missing on different rows

This how-to is for anyone who fits several targets on the same features,
where each target is null on rows of its own. The steps keep each
target's fit learning from every row on which that target is present. That takes one
`rls` spec per target, or one `ewridge` spec, whose `target_gaps` decides
which rows each target's fit reads.

## Prerequisites

These steps need the library:

- polars-online installed, with numpy for the recipe, which builds its own
  data and saves it to a parquet file.
- Targets that are [null](../GLOSSARY.md#the-model-bank-and-what-it-reads)
  on some rows. A null target leaves the row predicted and the clock
  moving, but the model learns nothing about that target from the row.
- The parameters of
  [`po.spec.rls`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.rls)
  and
  [`po.spec.ewridge`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.ewridge),
  which their docstrings define.

## Give each target its own `rls` spec

`rls`, recursive least squares, solves the ridge fit exactly on every row
by folding the row into a kept factor of its running sums.

**The targets share one factor of the running sums, so a row with any
target null is predicted for every target and learned for none.** When the
targets go missing on different rows, give each its own `rls` spec.
Otherwise one spec holds them all.

1. **Check which rows each target is present on.** Compare the targets'
   missing values with Polars, NaN counted as null, as the bank counts it:
   `pl.scan_parquet(...).select((pl.col("a").fill_nan(None).is_null() != pl.col("b").fill_nan(None).is_null()).sum()).collect()`
   counts the rows where `a` and `b` differ. Where it is 0, keep them in
   one `rls` spec.
2. **Otherwise, build one `rls` spec per target,** each naming one target
   in `targets=` and the same features.
3. **Give every spec to one bank.** The bank reads the stream once for all
   of them.

## Fit them all in one `ewridge` spec

`ewridge`, exponentially weighted ridge regression, solves its fit from
its running sums on a schedule. A null target is still predicted, and
[`weight_sum`](../GLOSSARY.md#warm-up-and-readiness), the weight the model
has accumulated, counts every row. `target_gaps` decides which rows a
target's fit is read from where the target is null on some. `lasso` takes
`target_gaps` too, with the same two values.

| `target_gaps` | the feature covariance is taken over | what that gives |
|---|---|---|
| `"own_rows"`, the default | the target's own rows, so its fit is the fit of those rows alone. Targets present on the same rows share one matrix, and one that goes missing where the others are present takes a copy and keeps its own from then on | several targets keep one `k × k` matrix per pattern of missing rows, and a single target keeps one |
| `"pairwise"` | every row, the way pandas' `DataFrame.cov` takes a pairwise-complete covariance: one matrix whatever the gaps | exact when the gaps have nothing to do with the features. Where they do, each slope is scaled by the ratio of the feature's variance on the target's rows to its variance on all of them |

1. **Name every target in one `ewridge` spec.**
2. **Keep the default `target_gaps="own_rows"` where the gaps depend on
   the features,** as with a target present only on trade rows between
   market-data rows. Each target's fit is then the fit of its own rows.
3. **Give `target_gaps="pairwise"` where the gaps have nothing to do with
   the features,** to keep one matrix however many patterns of missing
   rows the targets make.

The recipe plants two targets on one feature `x`. Target `b` goes missing
from row 3,000, and from there `a`'s slope on `x` steps from 2 to 3. It
prints `a`'s slope at the last row from one `rls` spec over both targets,
from an `rls` spec of `a`'s own, and from one `ewridge` spec under each
`target_gaps`, beside the planted 3. This code builds its own data:

```python
import numpy as np
import polars as pl
import polars_online as po

# Two targets on x: a's slope steps from 2 to 3 at row 3,000, where b goes missing.
rng = np.random.default_rng(3)
n = 6_000
x, e1, e2 = rng.standard_normal((3, n))
second_half = np.arange(n) >= 3_000
a = 1.0 + np.where(second_half, 3.0, 2.0) * x + 0.3 * e1
b = np.where(second_half, np.nan, -1.0 + 0.5 * x + 0.3 * e2)        # NaN counts as null in a target
pl.DataFrame({"t": np.arange(n, dtype=float), "x": x, "a": a, "b": b}).write_parquet("two_targets.parquet")

# One rls over both targets, one per target, and one ewridge under each target_gaps.
common = dict(features=["x"], clock="t", gap_cap=10.0, half_life=500.0)
specs = [
    po.spec.rls("rls_both", targets=["a", "b"], **common),
    po.spec.rls("rls_a", targets=["a"], **common),
    po.spec.rls("rls_b", targets=["b"], **common),
    po.spec.ewridge("ridge_own_rows", targets=["a", "b"], **common),                          # the default
    po.spec.ewridge("ridge_pairwise", targets=["a", "b"], target_gaps="pairwise", **common),
]
bank = po.ModelBank(specs)
bank.fit(pl.scan_parquet("two_targets.parquet"))

# a's slope on x at the last row, beside the planted 3.
slopes = bank.coef().filter(target="a", term="x")
for spec, slope in slopes.select("spec", "coef").iter_rows():
    print(f"{spec:>14}: a's slope {slope:.3f} (planted 3)")
```

```text
      rls_both: a's slope 2.000 (planted 3)
         rls_a: a's slope 2.978 (planted 3)
ridge_own_rows: a's slope 2.979 (planted 3)
ridge_pairwise: a's slope 2.979 (planted 3)
```

The `rls` spec over both targets learned nothing from row 3,000 on, since
every row there has `b` null, so it still reports the first half's slope.
Its own spec, and either `ewridge` spec, follow `a` to its new slope. They
stand a little under 3 because, at a half-life of 500 rows, about 2% of
the weight still sits on the rows before the step. `a` has no gaps, so
`"pairwise"` and `"own_rows"` read its fit from the same rows, and give it
the same slope.

## Check the result

**Read each target's coefficients with `bank.coef()`,** filtered by its
`target` column, as the recipe does. On a target whose relation you know,
each fit should stand near it.

**Read each target's coefficients over the stream where another target
goes missing.** Give the spec `coef_every=0`, and its output's `coef`
field holds the coefficients after every row. A target's fit that stops
moving on the rows where another target is null is learning none of
them: give that target a spec of its own.
