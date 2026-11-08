<!-- kind: how-to -->
# Choose a changepoint hazard

This how-to is for anyone who fits `bocpd` and wants to choose its
`hazard`, the expected run between breaks, and its priors, from the
stream's own out-of-sample log-likelihood with `po.eval.grid_search`. It
also tempers the model against single outlying rows with `robust_beta`.

## Prerequisites

These steps need the library, a stream in a file and a spec:

- polars-online and numpy installed. The recipe generates its own data.
- The stream saved as a file in clock order, to read back with
  `pl.scan_parquet` as a [query](../GLOSSARY.md#the-model-bank-and-what-it-reads)
  (a Polars `LazyFrame`).
- A `bocpd` [spec](../GLOSSARY.md#the-model-bank-and-what-it-reads)
  ([its entry in MODELS.md](../MODELS.md#bocpd--how-long-has-this-regime-lasted)),
  and the hazards to try. [Choose a model's settings from the stream's own
  errors](../CALIBRATION.md) gives the steps of the search and why the
  choice is scored on rows it had not seen.

## Choose the hazard

**Score `bocpd` by `loglik`, the log of the density it gave each row
before learning from it.** Summed over the stream, it is the stream's log
marginal likelihood, the criterion that compares a Bayesian model's
settings. `bocpd` predicts no target, so it has no squared error to score.
The Notes of [`po.eval.grid_search`'s
docstring](https://hgilde.github.io/polars-online/eval.html#polars_online.eval.grid_search),
under *What is scored*, say so.

1. **Decide whether breaks follow rows or time.** A number `hazard` says
   breaks follow the row rate, so on irregular rows a weekend counts as one
   row. On a temporal clock, give `hazard` as a duration, such as
   `hazard="1h"`, the expected time between breaks. The docstring
   gives the rules at a gap and at a step of 0, under its `hazard` entry
   and under *A duration hazard* in its Notes.
2. **Build a `bocpd` spec builder with `functools.partial`,** giving it
   the keywords every spec shares. `hazard` takes one value a spec.
3. **Pass the query, the builder and a `param_grid` of hazards to
   `po.eval.grid_search`, with `scoring="loglik"`.** It returns a row per
   hazard with its `loglik` a row, highest best, and a row for the choice.
4. **Run the hazard ranked 1 on the rows to come.**
5. **Report the `loglik` of the choice,** from `table.filter(kind="choice")`.

This code plants a mean that jumps at random, every 200 rows on average:

```python
import functools

import numpy as np
import polars as pl

import polars_online as po

# A mean that jumps at random, every 200 rows on average, to a level of spread 2, in noise of 1.
rng = np.random.default_rng(14)
n = 20_000
regime = np.cumsum(rng.random(n) < 1 / 200)
x = rng.normal(0, 2.0, regime.max() + 1)[regime] + rng.standard_normal(n)
pl.DataFrame({"x": x}).write_parquet("jumps.parquet")

# 2. One bocpd spec per hazard, its priors matching the planted spreads.
bocpd = functools.partial(
    po.spec.bocpd, features=["x"], mean_prior=[0.0], covariance_prior=[1.0],
    mean_precision_prior=0.25, degrees_of_freedom_prior=2.0,   # a level's variance of 4: the noise's 1 over 0.25
)
# 3. Every hazard, scored by its log-likelihood in one pass.
table = po.eval.grid_search(pl.scan_parquet("jumps.parquet"), bocpd, {"hazard": [20.0, 60.0, 200.0, 600.0, 2_000.0]}, scoring="loglik")

# 4 and 5. Each hazard's score and rank, and what choosing from earlier rows earned.
for hazard, loglik, rank in table.filter(kind="candidate").select("hazard", "loglik", "rank").iter_rows():
    print(f"hazard {hazard:>5g}: loglik {loglik:.4f} a row, rank {rank}")
print(f"chosen from earlier rows: loglik {table.filter(kind='choice')['loglik'].item():.4f} a row")
```

```text
hazard    20: loglik -1.5035 a row, rank 5
hazard    60: loglik -1.4779 a row, rank 3
hazard   200: loglik -1.4729 a row, rank 1
hazard   600: loglik -1.4751 a row, rank 2
hazard  2000: loglik -1.4796 a row, rank 4
chosen from earlier rows: loglik -1.4731 a row
```

The hazard of 200, the planted mean run, ranks first.

## Leave the priors to the first rows

A prior decides what counts as surprising, so it is in the data's units.
Too wide, and no row is surprising, so the model never reports a break.

1. **Leave `mean_prior` and `covariance_prior` out.** The first
   `warm_rows` rows then set them in the data's units, and report null
   while they do.
2. **If you give a prior, give it in the data's units,** such as the
   variance of a feature's rows within one regime for `covariance_prior`.

This code makes a column of spread 1e-3 whose mean shifts by four of its
standard deviations at row 150. It dates the regime at row 299 with the
prior left out and with four priors given:

```python
import numpy as np
import polars as pl
import polars_online as po

# A column of spread 1e-3 whose mean shifts by 4e-3 at row 150.
rng = np.random.default_rng(16)
x = 1e-3 * rng.standard_normal(300)
x[150:] += 4e-3
pl.DataFrame({"x": x}).write_parquet("small.parquet")

# 1 and 2. The prior left to the first rows, then given at four scales.
for prior in (None, [1e-6], [1e-3], [1.0], [100.0]):
    cp = po.spec.bocpd("cp", features=["x"], hazard=250.0, degrees_of_freedom_prior=2.0, covariance_prior=prior)
    out = pl.scan_parquet("small.parquet").online.fit_predict([cp]).collect().unnest("cp")
    print(f"covariance_prior {prior}: at row 299 the regime began at row {299 - out['run_mode'][299]}")
```

```text
covariance_prior None: at row 299 the regime began at row 150
covariance_prior [1e-06]: at row 299 the regime began at row 150
covariance_prior [0.001]: at row 299 the regime began at row 0
covariance_prior [1.0]: at row 299 the regime began at row 0
covariance_prior [100.0]: at row 299 the regime began at row 0
```

The prior set from the first rows, and the one given in the data's units,
date the shift to its row. A prior a thousand times the data's variance
or wider never sees it.

## Choose the priors with the hazard

Use this when you give a prior, and want the stream's own log-likelihood
to choose it with the hazard.

1. **Leave the prior to search out of the builder.**
2. **Add it to `param_grid` as another key, each value a list with one
   number a feature,** as `"covariance_prior": [[0.01], [0.1], [1.0]]`.
   Every combination of the keys is a spec of its own, all in one model
   bank.
3. **Run `po.eval.grid_search` with `scoring="loglik"`.**
4. **Read the table** as in [Choose the hazard](#choose-the-hazard).

This code searches four values of `covariance_prior` at a hazard of 200,
on the stream of the first recipe, `jumps.parquet`:

```python
# 1. The builder, with covariance_prior left out.
bocpd = functools.partial(
    po.spec.bocpd, features=["x"], mean_prior=[0.0],
    mean_precision_prior=0.25, degrees_of_freedom_prior=2.0,
)
# 2. Each value a list of one number, for the one feature.
grid = {"hazard": [200.0], "covariance_prior": [[0.01], [0.1], [1.0], [10.0]]}
# 3. Every pair, scored by its log-likelihood in one pass.
table = po.eval.grid_search(pl.scan_parquet("jumps.parquet"), bocpd, grid, scoring="loglik")
# 4. Each prior's score and rank.
for prior, loglik, rank in table.filter(kind="candidate").select("covariance_prior", "loglik", "rank").iter_rows():
    print(f"covariance_prior {prior}: loglik {loglik:.4f} a row, rank {rank}")
```

```text
covariance_prior [0.01]: loglik -1.4921 a row, rank 4
covariance_prior [0.1]: loglik -1.4817 a row, rank 2
covariance_prior [1.0]: loglik -1.4729 a row, rank 1
covariance_prior [10.0]: loglik -1.4830 a row, rank 3
```

The score is highest at 1, the planted noise's variance.

## Temper single outlying rows

Untempered, a single row far from the rest reads as a break, and the run
it starts carries that row in its mean. `robust_beta` weights each row by
how typical it is, so an outlying row moves nothing. A new regime is then
a run of rows each forgiven one at a time, so the larger the power, the
later a real shift is found.

1. **Keep `covariance_type` at `"diag"`, the default.** `robust_beta` is
   refused beside `"full"`.
2. **Center and scale each feature in the query before the bank,** since
   the tempered predictive density (the density the model gives each row
   before learning it) is not free of the data's units. Make its
   z-score `(pl.col("x") - po.ewm_mean("x", half_life=h)) /
   po.ewm_std("x", half_life=h)` with `lf.online.with_windows`, which
   runs the window operators (Polars' `with_columns` cannot).
3. **Give `robust_beta=0.1`, and keep it below about 0.2.** Above that, a
   real shift is found late or not at all. Over 50 streams with a
   four-sigma shift, 0.1 found every shift, at worst 6 rows late and
   dated within 5 rows of it, and ignored a single 20-sigma row. The
   docstring's Notes, under *Tempering*, give the measurements at 0.15 to
   0.5.

This code makes a column already centered and scaled, with one 20-sigma
row at row 200 and a four-sigma shift at row 400. It reads `p_change` on
the outlying row, and the start of the regime on the rows before and after
the shift, under three powers:

```python
import numpy as np
import polars as pl
import polars_online as po

# Level 0 and spread 1, one 20-sigma row at row 200, and a shift of 4 at row 400.
rng = np.random.default_rng(15)
x = rng.standard_normal(600)
x[200] = 20.0
x[400:] += 4.0
pl.DataFrame({"x": x}).write_parquet("outlier.parquet")

# 1 and 3. The diagonal predictive, untempered and at two powers.
for beta in (None, 0.1, 0.3):
    cp = po.spec.bocpd("cp", features=["x"], hazard=250.0, covariance_type="diag", robust_beta=beta)
    out = pl.scan_parquet("outlier.parquet").online.fit_predict([cp]).collect().unnest("cp")
    began = [row - out["run_mode"][row] for row in (399, 599)]   # the row each current regime began on
    print(f"robust_beta {beta}: p_change {out['p_change'][200]:.2f} on the outlier; "
          f"regime began at row {began[0]} before the shift, at row {began[1]} after it")
```

```text
robust_beta None: p_change 0.70 on the outlier; regime began at row 201 before the shift, at row 400 after it
robust_beta 0.1: p_change 0.00 on the outlier; regime began at row 0 before the shift, at row 400 after it
robust_beta 0.3: p_change 0.01 on the outlier; regime began at row 257 before the shift, at row 518 after it
```

Untempered, the outlier starts a regime of its own. At 0.1 it moves
nothing, and the shift is dated to its row. At 0.3, by row 599 the shift
is still not dated to its row: the regime began at row 518. Before the
shift, it reports a regime begun at row 257, where nothing changed.

## Check that it worked

The search worked when the hazard ranked 1 lies inside the list you gave,
and the `loglik` of the choice is close to the `loglik` of rank 1:

| what you see | what it means | what to do |
|---|---|---|
| the call raises `ValueError`, saying the spec makes no prediction of a target to score | `scoring` was left at `"mse"`, and `bocpd` predicts no target | pass `scoring="loglik"` |
| the hazard ranked 1 is the smallest or the largest in the list | the best hazard may lie outside the list | extend the list past that end, and search again |
| the `loglik` of the choice is below the `loglik` of rank 1 | the rows before some spans chose another hazard | report the choice's `loglik`: it is the one earned on rows the choice had not seen |
| no break reported on a shift you can see | a prior given too wide makes no row surprising | leave `mean_prior` and `covariance_prior` out, or give them in the data's units |
| a single outlying row starts a regime of its own | the predictive is untempered | give `robust_beta=0.1` under `covariance_type="diag"`, on centered and scaled features |
