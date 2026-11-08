<!-- kind: how-to -->
# Choose columns on the lasso path

This how-to is for anyone who fits `lasso`, the sparse regression that
gives some features a coefficient of exactly zero and picks its penalty
as the stream runs. It sets the `penalties` to try, how the penalty in
force is ranked (`select_half_life`), and how many sweeps each descent
may take (`max_iter`).

## Prerequisites

These steps need the library:

- polars-online installed, with numpy for the recipes. Each recipe builds
  its own data and saves it to a parquet file.
- The parameters of
  [`po.spec.lasso`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.lasso),
  which its docstring defines. Its entry in
  [MODELS.md](../MODELS.md#lasso--lasso-path-its-penalty-chosen-as-it-runs)
  gives its update equation.

## Scale the penalties with the target

`lasso` keeps the [running sums](../GLOSSARY.md#fitting-a-model) of
`ewridge` and solves them, on a schedule, by coordinate descent at each
penalty of `penalties`. Each penalty is a [grid
point](../GLOSSARY.md#grids) with a prediction of its own, and
`penalty_selected_<t>` names the one with the lowest exponentially
weighted [out-of-sample](../GLOSSARY.md#fitting-a-model) squared error so
far, as it stood before the row.

**Scale `penalties` with the target.** Each penalty `l` of `penalties`,
times `l1_ratio`, the share of it that is L1 (1 by default), is an L1
threshold in the target's units. So for a pure lasso, a target ten times
larger needs penalties ten times larger to zero the same coefficients.

1. **Choose the penalties in the target's units,** strictly decreasing,
   each finite and at least 0. Each solve runs one descent for each
   penalty, so keep the path to the few penalties worth comparing.
2. **Multiply them by the same factor as the target.** A target in cents
   rather than dollars needs penalties a hundred times larger.
3. **Read `penalty_selected_<t>` on each row** for the penalty in force,
   and `bank.coef()` for the coefficients at each penalty.

The recipe plants 3 of 10 features in a target `y`, and fits `y` and
`10·y` with the same penalties and with penalties ten times larger. It
prints the features each fit keeps at each penalty. This code builds its
own data:

```python
import numpy as np
import polars as pl
import polars_online as po

# 10 features, of which only x0, x1 and x2 carry weight in y; y10 is ten times y.
rng = np.random.default_rng(3)
n, k = 5_000, 10
x = rng.standard_normal((n, k))
y = 2.0 * x[:, 0] - 1.5 * x[:, 1] + 1.0 * x[:, 2] + rng.standard_normal(n)
names = [f"x{i}" for i in range(k)]
pl.DataFrame(x, schema=names).with_columns(y=y, y10=10 * y).write_parquet("sparse.parquet")

# The same penalties on y and on y10, and the penalties ten times larger on y10.
penalties = [0.3, 0.1, 0.03]
common = dict(features=names, half_life=float("inf"))
specs = [
    po.spec.lasso("fit_y", targets=["y"], penalties=penalties, **common),
    po.spec.lasso("same_on_y10", targets=["y10"], penalties=penalties, **common),
    po.spec.lasso("scaled_on_y10", targets=["y10"], penalties=[10 * p for p in penalties], **common),
]
bank = po.ModelBank(specs)
bank.fit(pl.scan_parquet("sparse.parquet"))

# The features each fit keeps at each penalty.
kept = (bank.coef().filter(pl.col("term") != "intercept", pl.col("coef") != 0.0)
        .group_by("spec", "penalty", maintain_order=True).agg(pl.col("term")))
print("planted: x0, x1, x2")
for spec, penalty, terms in kept.iter_rows():
    print(f"{spec:>13} at {penalty:<4g}: keeps {', '.join(terms)}")
```

```text
planted: x0, x1, x2
        fit_y at 0.3 : keeps x0, x1, x2
        fit_y at 0.1 : keeps x0, x1, x2
        fit_y at 0.03: keeps x0, x1, x2
  same_on_y10 at 0.3 : keeps x0, x1, x2
  same_on_y10 at 0.1 : keeps x0, x1, x2, x4, x6, x7, x9
  same_on_y10 at 0.03: keeps x0, x1, x2, x3, x4, x5, x6, x7, x9
scaled_on_y10 at 3   : keeps x0, x1, x2
scaled_on_y10 at 1   : keeps x0, x1, x2
scaled_on_y10 at 0.3 : keeps x0, x1, x2
```

On `10·y`, the penalties that kept exactly the planted three on `y` let
noise features in at 0.1 and 0.03. Penalties ten times larger keep the
same three at every point of the path.

## Rank the penalties on every error so far

**To rank the penalties on the weighted mean of every out-of-sample
squared error so far, give `select_half_life=float("inf")`.** Under a hard
window the mean covers the hard window's rows. By default the error forgets at
the model's half-life.

1. **Keep the default where the best penalty may change,** so the choice
   forgets at the pace the fit does.
2. **Give `select_half_life=float("inf")` where it should not,** so every
   error the stream has made counts toward the choice. This line reuses
   the first recipe's `penalties` and `common`:

```python
steady = po.spec.lasso("steady", targets=["y"], penalties=penalties, select_half_life=float("inf"), **common)
```

## Give a larger `max_iter` when a descent runs out

Within a solve, the descent stops after `max_iter` sweeps, 100 by default,
or after a sweep that moved no coefficient by more than `tol` (1e-10 by
default) relative to the target's spread. A descent that runs out of sweeps first leaves that penalty's fit
short of the lasso's solution.

**If `bank.solve_failures()` counts a descent that ran out of its
`max_iter` sweeps, give a larger `max_iter`.** Columns that nearly repeat
each other slow the descent most. The docstring's Notes, under *The
descent*, measure how many sweeps they take.

1. **Read `bank.solve_failures()` after a fit.** For `lasso` it counts each
   descent that ran out of `max_iter` sweeps, one per target and penalty.
2. **If it is above 0, give the spec a larger `max_iter`,** and fit again
   until the count is 0.

The recipe makes `x1` nearly repeat `x0`, and counts the descents that ran
out of sweeps at three values of `max_iter`. This code continues from the
first recipe, whose `x`, `rng`, `n`, `names`, `penalties` and `common` it
reuses:

```python
# The same target, with x1 now x0 plus a little noise, so the two nearly repeat each other.
x[:, 1] = x[:, 0] + 0.3 * rng.standard_normal(n)
y = 2.0 * x[:, 0] - 1.5 * x[:, 2] + 1.0 * x[:, 3] + rng.standard_normal(n)
pl.DataFrame(x, schema=names).with_columns(y=y).write_parquet("repeats.parquet")

# The same lasso at three caps on its sweeps.
specs = [po.spec.lasso(f"max_iter_{m}", targets=["y"], penalties=penalties, max_iter=m, **common)
         for m in (100, 1_000, 10_000)]
bank = po.ModelBank(specs)
bank.fit(pl.scan_parquet("repeats.parquet"))
for spec, failures in bank.solve_failures().select("spec", "solve_failures").iter_rows():
    print(f"{spec:>14}: {failures} descents ran out of sweeps")
```

```text
  max_iter_100: 7455 descents ran out of sweeps
 max_iter_1000: 2 descents ran out of sweeps
max_iter_10000: 0 descents ran out of sweeps
```

At the default of 100 sweeps, thousands of descents stop short. At
10,000, every descent reaches its tolerance.

## Check the result

**Read `bank.solve_failures()`:** a count of 0 for the spec says every
descent reached its tolerance.

**Read the coefficients at the penalty in force.** Fit with
`bank.fit_predict`, and read the last row's `penalty_selected_<t>` from
the spec's output column. Filter `bank.coef()` by that `penalty`, and
check that the features it keeps are the ones you expect to matter.
[MODELS.md's
example](../MODELS.md#lasso--lasso-path-its-penalty-chosen-as-it-runs)
reads them that way.

To run the same path offline on a Gram, use `po.gram.lasso_path`, as
[Choosing columns on the lasso
path](../GRAM.md#choosing-columns-on-the-lasso-path) in GRAM.md shows.
