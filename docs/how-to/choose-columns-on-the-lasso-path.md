<!-- kind: how-to -->
# Choose columns on the lasso path

This how-to is for anyone who chooses columns with the lasso, the sparse
regression that gives some features a coefficient of exactly zero, either
as the stream runs or afterwards from the Grams one pass leaves. On the
stream, the `lasso` model picks its penalty as it runs: you set the
`penalties` to try, how the penalty in force is ranked
(`select_half_life`), and how many sweeps each descent may take
(`max_iter`). From a [Gram](../GLOSSARY.md#fitting-a-model), you walk the
path knot by knot with `po.gram.lars_path`, and choose a penalty by its
error on weeks held out with `po.gram.lasso_path`. A knot is a penalty at
which the set of columns with a coefficient other than 0 changes.

## Prerequisites

These steps need the library:

- polars-online installed with numpy, an optional extra, which
  `pip install "polars-online[numpy]"` adds. Each recipe builds its own
  data and saves it to a parquet file.
- For the model, the parameters of
  [`po.spec.lasso`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.lasso),
  which its docstring defines. Its entry in
  [MODELS.md](../MODELS.md#lasso--lasso-path-its-penalty-chosen-as-it-runs)
  gives its update equation.
- For a Gram, a Gram per block from one pass, as [Keep a Gram per block
  in one pass](fit-weekly-grams-in-one-pass.md) makes them. [Choosing
  columns on the lasso
  path](../GRAM.md#choosing-columns-on-the-lasso-path) explains the
  problem both paths solve, their knots, and how each stops.

## Scale the penalties with the target

`lasso` keeps the same [Gram](../GLOSSARY.md#fitting-a-model) as
`ewridge` and solves it, on a schedule, by coordinate descent at each
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
default) relative to the target's spread. Where the largest covariance
of a standardized feature with the target is larger than that spread,
`tol` is relative to it instead. A descent that runs out of sweeps first leaves that penalty's fit
short of the lasso's solution.

**If `bank.solve_failures()` counts a descent that ran out of its
`max_iter` sweeps, give a larger `max_iter`.** Columns that nearly repeat
each other slow the descent most. The Notes of `po.spec.lasso`'s
docstring, under *The descent*, measure how many sweeps they take.

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

## Choose columns on a Gram's lasso path

Use this to choose columns after the pass, from the Grams it left, on any
set of weeks. A knot is a penalty at which the set of active columns,
those whose coefficient is not 0, changes. `po.gram` follows the lasso
path two ways, and each suits a different question:

| | `po.gram.lars_path`, and `po.gram.lars_paths` for many Grams | `po.gram.lasso_path` |
|---|---|---|
| computes | each knot exactly, from the largest penalty down, by least angle regression | the solution at each penalty given, each started from the one before, by coordinate descent |
| stops | after `max_iter` knots past its first, or at the first knot with at least `max_active` columns active | at the last penalty |
| time | 4 ms for the whole path, 201 knots, at 200 columns. [`po.gram.lars_path`'s docstring](https://hgilde.github.io/polars-online/gram.html#polars_online.gram.lars_path) gives 0.2 ms for 40 knots at 200 columns, and 3 ms at 2,000 | 3 to 8 ms for 25 penalties at 200 columns |
| takes | `penalty_weights=`, a scale above 0 on each column's penalty | `penalty_weights=`, where a weight of 0 keeps a column in the fit at every penalty, and an elastic net (`l1_ratio` below 1) |
| use it for | the first columns to enter: a selection, and [stability selection](select-columns-stably-over-weeks.md) | a penalty chosen by held-out error, an elastic net, a column that must stay in |

The times were measured on 14 cores at a load average near 12, the number
of processes running or waiting to run, averaged over a minute.

1. **Pool the weeks to fit on, and the weeks held out, with
   `po.gram.merge`.**
2. **Walk the path from its start with `po.gram.lars_path(train,
   max_active=8)`.** It returns the `penalties` at each knot, the `active`
   columns there, and the `coef` at each knot, the intercept first.
3. **Read the order in which the columns enter.** The first to enter are
   the ones the lasso chooses.
4. **Run `po.gram.lasso_path(train, grid)` on a grid of penalties,** from
   the first knot's down a thousandfold.
5. **Score each row of the grid on the held-out Gram.**
6. **Keep the penalty with the lowest held-out error.**

The recipe walks the path on weeks 1 to 4 to its eighth active column,
and checks coordinate descent at its knots. Then it chooses a penalty
from a grid of 25 by the error on the other 36 weeks. This code builds
its own data, 40 weeks of 200 features in families of five, of which six
matter:

```python
from datetime import datetime, timedelta

import numpy as np
import polars as pl

import polars_online as po

# 40 weeks, a row every 20 minutes. 200 features in 40 families of five that share a factor, so
# two features of one family correlate at 0.5. y depends on six of them, and x25's coefficient is
# drawn again each week, 0.15 on average.
rng = np.random.default_rng(7)
n_weeks, per_week, k = 40, 504, 200
n = n_weeks * per_week
x = 0.7 * (rng.standard_normal((n, k)) + np.repeat(rng.standard_normal((n, k // 5)), 5, axis=1))
b25 = 0.15 + 0.3 * rng.standard_normal(n_weeks)
y = x[:, [0, 5, 10, 15, 20]] @ [0.5, 0.4, 0.3, 0.2, 0.1] + b25.repeat(per_week) * x[:, 25]
y += rng.standard_normal(n)
features = [f"x{j}" for j in range(k)]
# Stamp each row, from a Monday, and save the table for `pl.scan_parquet` to read.
start = datetime(2025, 1, 6)  # a Monday
ts = pl.datetime_range(start, start + timedelta(minutes=20 * (n - 1)), "20m", eager=True)
pl.DataFrame({"ts": ts, **dict(zip(features, x.T)), "y": y}).write_parquet("weeks.parquet")

# One pass over the file leaves a Gram per week. The week is a column made in the query.
lf = pl.scan_parquet("weeks.parquet").with_columns(week=pl.col("ts").dt.truncate("1w"))
spec = po.spec.ewridge(
    "blocks", targets=["y"], features=features, group="week",
    half_life=float("inf"), solve_every=float("inf"), target_gaps="pairwise", gram_block_rows=256,
)
bank = po.ModelBank([spec])
bank.fit(lf)  # learns from every row, and keeps only the state
blocks = bank.gram("blocks")  # one dict per week, oldest first


# The score of step 5: a fit's mean squared error on any set of weeks, without reading their rows.
def mean_squared_error(g, coef):
    """A fit's mean squared error over the rows behind g, from g's arrays alone."""
    b0, b = coef[0], coef[1:]
    gap = g["cross_moments"][0][0] - b0 - g["means"][1:] @ b
    spread = g["target_vars"][0] - 2 * b @ g["cross_centered"][0][1:] + b @ g["comoments"][1:, 1:] @ b
    return spread + gap**2


# 1. Fit on four weeks, judge on the other 36.
train, held_out = po.gram.merge(blocks[:4]), po.gram.merge(blocks[4:])
print(f"x25's coefficient: {b25[:4].mean():.2f} on average over these four weeks, {b25.mean():.3f} over all forty")

# 2 and 3. LARS: the path from its start, knot by knot, to the first knot with at least eight columns active.
path = po.gram.lars_path(train, max_active=8)
active = []
for penalty, now in zip(path["penalties"], path["active"]):
    change = [f"+{c}" for c in now if c not in active] + [f"-{c}" for c in active if c not in now]
    print(f"knot at penalty {penalty:.4f}: {' '.join(change)}")
    active = now
cd = po.gram.lasso_path(train, path["penalties"], tol=1e-12)  # coordinate descent at the knots
print(f"coordinate descent at the knots, against LARS: within 1e-9: {np.abs(cd - path['coef']).max() < 1e-9}")

# 4 to 6. Coordinate descent on 25 penalties, from the first knot's down a thousandfold, scored on the 36 weeks.
grid = np.geomspace(path["penalties"][0], path["penalties"][0] / 1_000, 25)
fits = po.gram.lasso_path(train, grid)
errors = [mean_squared_error(held_out, coef) for coef in fits]
best = int(np.argmin(errors))
print(f"the best penalty, {grid[best]:.4f}: {np.count_nonzero(fits[best][1:])} columns, held-out error {errors[best]:.4f}")
print(f"least squares on the same four weeks: held-out error {mean_squared_error(held_out, po.gram.solve(train)):.4f}")
```

```text
x25's coefficient: 0.32 on average over these four weeks, 0.186 over all forty
knot at penalty 0.4918: +x0
knot at penalty 0.3408: +x5
knot at penalty 0.3172: +x25
knot at penalty 0.2741: +x10
knot at penalty 0.1933: +x15
knot at penalty 0.0784: +x20
knot at penalty 0.0752: +x117
knot at penalty 0.0720: +x4
coordinate descent at the knots, against LARS: within 1e-9: True
the best penalty, 0.0369: 18 columns, held-out error 1.1328
least squares on the same four weeks: held-out error 1.2265
```

The six planted columns enter first. `x25` is third, since its
coefficient averages 0.32 over these four weeks, where it averages 0.186
over all forty. Then come `x117`, a column with no effect, and `x4`,
which shares `x0`'s family factor. Coordinate descent at each knot gives
LARS's coefficients within 1e-9. On the grid, a penalty of 0.0369 keeps
18 columns and has a held-out error of 1.1328, where least squares on the
same four weeks has 1.2265.

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

**On a Gram, check coordinate descent against `po.gram.lars_path` at the
knots,** as the recipe of [Choose columns on a Gram's lasso
path](#choose-columns-on-a-grams-lasso-path) does: at a knot, least angle
regression gives the lasso's coefficients exactly, and coordinate descent
reaches them to its tolerance. If a `ReadinessWarning` names a penalty
whose descent ran out of sweeps, that penalty's row is where the descent
stopped: give `po.gram.lasso_path` a larger `max_iter`, or take the knots
from `po.gram.lars_path`.
