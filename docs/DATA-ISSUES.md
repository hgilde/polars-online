# Detecting data issues

A model fitted on a stream fits whatever it is given. It cannot tell you
that a column is `-999` where it was not measured, or that a feed stopped
and its last value was carried forward. Nor can it tell you that a feature
is a price level where a change was meant. `bank.check()` reads what a bank
kept during a pass and lists what is wrong with the data, and an `audit`
spec counts what no model keeps. For each problem, the fix is made with
Polars, in the query before the bank.

The [diagnostics](DIAGNOSTICS.md) ask
whether a model is working. This page asks whether the data can be learned
from at all.

Here is a stream with three faults planted, fitted by a ridge regression
with an audit beside it. `x1` is `-999` where it was not measured, `x2` is
the same value on every row, and `x3` is null on 30% of the rows:

```python
import numpy as np
import polars as pl

import polars_online as po

# Quotes, a row a second, with the three faults planted.
rng = np.random.default_rng(1)
n = 20_000
x0, x1, x3 = rng.standard_normal((3, n))
y = 1.0 * x0 + 0.5 * x1 + 0.2 * x3 + rng.standard_normal(n)
pl.DataFrame(
    {
        "t": np.arange(n, dtype=float),
        "x0": x0,
        "x1": np.where(rng.random(n) < 0.05, -999.0, x1),  # -999 where it was not measured
        "x2": np.full(n, 1.0),  # one value on every row
        "x3": np.where(rng.random(n) < 0.3, np.nan, x3),  # null on 30% of the rows
        "y": y,
    }
).fill_nan(None).write_parquet("quotes.parquet")
lf = pl.scan_parquet("quotes.parquet")

features = ["x0", "x1", "x2", "x3"]
audit = po.spec.audit("audit", columns=[*features, "y"], clock="t", gap_cap=5.0)
model = po.spec.ewridge(
    "ridge", targets=["y"], features=features, clock="t", gap_cap=5.0, half_life=600.0
)
bank = po.ModelBank([audit, model])
bank.fit(lf)  # learn from every row; keep only the state

findings = bank.check()  # one row per problem, the worst first
shown = findings.select("severity", "code", "spec", "column", pl.col("value").round_sig_figs(3))
for row in shown.iter_rows():
    print(*row)
print(findings.row(0, named=True)["message"])  # what to do, and the section of this page on it
```

```text
warning constant audit x2 0.0
warning sentinel audit x1 0.0464
warning constant ridge x2 0.0
warning missing ridge x3 0.3
info heavy_tails audit x1 951.0
info missing audit x3 0.3
column 'x2' took one value (1) on every usable row of the stream: as a feature it is collinear with the intercept, and as a target there is nothing to learn. See https://github.com/hgilde/polars-online/blob/main/docs/DATA-ISSUES.md#constant
```

Each finding names the spec that found it. The audit sees every fault:
`x2` is constant, `-999` is 4.6% of `x1`'s values, and `x3` is missing on
30% of the rows. The ridge regression sees the two faults it can measure,
the constant feature and the missing one. It cannot see the sentinel,
because to a model `-999` is a number like any other. The sentinel also
gives `x1` heavy tails: one value 951 robust standard deviations from the
median. The audit reports `x3`'s gaps as `info`, and the model as a
`warning`. An audit knows no roles, and a gap in a target can be a design,
where a gap in a feature skips the row. A finding's `value` is the
number measured, here the share missing, the share of the commonest value,
or a standard deviation of 0. Every message ends with a link to the
section of this page that its `code` names.

These are the words of the library that the page uses:

| word | what it means here |
|---|---|
| spec | one model's description: the model, the columns it reads, and how it treats time. `po.spec.ewridge(...)` makes one; [the README's model table](../README.md#models) lists every kind |
| bank | `po.ModelBank`, a set of specs fitted together over the same rows; `bank.fit(lf)` learns from a query's rows, and `bank.fit_predict_batches(lf)` also returns each row's prediction, made before the row is learned |
| pass | one run of a bank over the stream |
| audit | `po.spec.audit`, a spec that learns nothing and counts what its columns hold |
| usable value | a number a model can learn from: not null, NaN or infinite, and no larger than `1e100` in magnitude |
| group | a spec's `group` column splits the stream, and each value of it gets a fit of its own |
| `half_life`, `gap_cap` | each row's weight halves every `half_life` of the clock column; `gap_cap` is the longest step between two rows that a model ages by |
| Gram | the matrix of running sums that `ewridge`, `lasso` and `ew_cov` keep, from which a fit is solved |
| Kish rows | `weight_sum² / Σw²`: the number of equally weighted rows that a decayed set of rows is worth |
| robust z | how far a value sits from the median, in units of the median absolute deviation times 1.4826 |

| problem | the codes `check()` gives it |
|---|---|
| [Missing values](#missing-values) | [`missing`](#missing), [`few_learned`](#few_learned), [`nothing_learned`](#nothing_learned) |
| [Sentinel values](#sentinel-values) | [`sentinel`](#sentinel) |
| [Constant and frozen features](#constant-and-frozen-features) | [`constant`](#constant), [`frozen`](#frozen), [`low_support`](#low_support), [`solve_failures`](#solve_failures) |
| [A level far above its spread, and scales apart](#a-level-far-above-its-spread-and-scales-apart) | [`level_over_spread`](#level_over_spread), [`scales_apart`](#scales_apart), [`ridge_shrinks`](#ridge_shrinks) |
| [Duplicated and collinear features](#duplicated-and-collinear-features) | [`duplicate`](#duplicate), [`collinear`](#collinear) |
| [Categories stored as numbers](#categories-stored-as-numbers) | [`few_values`](#few_values) |
| [Random walks](#random-walks) | [`random_walk`](#random_walk) |
| [Heavy tails](#heavy-tails) | [`heavy_tails`](#heavy_tails) |
| [Leakage](#leakage) | [`leakage`](#leakage) |
| [The clock](#the-clock) | [`step_back`](#step_back), [`resets`](#resets), [`duplicate_stamps`](#duplicate_stamps), [`gaps`](#gaps), [`irregular_clock`](#irregular_clock) |
| [Too little data for the fit](#too-little-data-for-the-fit) | [`few_rows`](#few_rows), [`group_sizes`](#group_sizes), [`never_settled`](#never_settled), [`below_min_weight`](#below_min_weight), [`withheld`](#withheld) |
| [How the checks work](#how-the-checks-work) | [`not_checked`](#not_checked) |

## How the checks work

**`check()` adds no work to the run, because it reads only what a bank
keeps anyway.** That is `summary()`, `describe()`, `solve_failures()`,
`last_row()`, the Gram (the matrix of running sums) of an `ewridge`,
`lasso` or `ew_cov`, and a `marginal`'s pairs. So a bank loaded from a state file gives the same
findings as the bank that saved it, and the chunking of the stream moves
none of them. Call it after a first pass and before trusting a fit. A
group closed under `group_close` has left the bank with its sums, so
`check()` sees only the groups still open, and an audit refuses
`group_close`.

**An audit counts what a model does not keep.** A model skips a row whose
feature is missing, so it never sees the row. An audit reads every row of
its columns and counts nulls, NaNs and infinities apart. It also counts
repeated values, runs of one value, persistence like a random walk's,
heavy tails, duplicated columns and the clock's stamps. Its memory does not
grow with the rows: about 10 KiB a column. Run it alone before choosing a
model, or beside the models in the same pass, as the [three-fault example](#detecting-data-issues) does.
Without one in the bank, `check()` says nothing about sentinels, frozen
feeds, few values, random walks, tails, duplicates or the clock's spacing.

**Each finding has a severity.** An `error` means the model cannot learn,
such as a target that never varies. A `warning` means the fit is
unreliable. An `info` is worth knowing and often fine, such as a target
present on a third of the rows by design.

**Every threshold was measured, both for how often it finds a planted
problem and for how often it fires on clean data.** On five clean shapes
over ten seeds and fifteen specs, no model-side check gave an error or a
warning in 750 runs. On eleven clean shapes over ten seeds, an audit of
every column gave none either. Each planted problem was found on every
seed, for every spec it harms, and for no other. The docstring of
[`ModelBank.check`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.ModelBank.check)
lists the clean shapes and each threshold with the numbers it was set
from, and each problem's section gives the numbers for its own codes.
`tests/test_check.py` runs the sweep.

<a id="not_checked"></a>

**`not_checked` (info) says that numpy is not installed.** The checks that
read a Gram, `collinear` and `leakage`, need numpy, an optional extra.
Install it with `pip install polars-online[numpy]`.

**Every recipe on this page is a complete program.** It generates its data
with the problem planted, saves it to parquet and reads it back with
`pl.scan_parquet`, as a stream too large to hold would be read. Its output
is shown below it. `tests/test_data_issues.py` runs each recipe and holds
its output to what is shown, so none goes stale. A number past a million
in an output is a model that has blown up, and its digits depend on the
platform's rounding. So the test holds such a number only to being past a
million.
Warnings that a recipe raises, such as a `ReadinessWarning`, go to the
standard error stream and are not shown.

## Missing values

<a id="missing"></a>
<a id="few_learned"></a>
<a id="nothing_learned"></a>

**A model skips a row on which any feature or the weight is missing.** A
null, a NaN, an infinity or a magnitude past `1e100` all count as missing.
The row's outputs are null, its clock still advances, and nothing is
learned from it. In `ewridge`, `lasso` and `ew_cov`, which keep one matrix
of running sums over every column, the whole row is lost to every column,
not only to the one with the gap. With gaps in 30% of one feature's rows,
30% of the rows are lost to all of them. With 1,000 columns in one model,
a row is lost wherever any one of them is missing.

| code | severity | fires when |
|---|---|---|
| `missing` | warning; error at 100% | a feature or the weight is missing on 10% of the rows fed or more |
| `missing` | info | a target is missing on half the rows or more: a sparse label is a design, and a join gone wrong otherwise |
| `missing`, from an audit | info; error at 100% | a column is missing on 10% of its rows or more. An audit knows no roles, so it cannot tell a feature from a target |
| `few_learned` | warning | fewer than half the rows fed were processed, the rest skipped for a missing feature or weight; the column missing most often is named |
| `nothing_learned` | error | a stream was fed rows and learned from none |

**Leaving the gap alone is right whenever the gap does not depend on the
target.** Statisticians sort gaps by what they depend on (Rubin 1976).
Gaps *missing completely at random* depend on nothing. Gaps *missing at
random* depend only on values that are observed. Gaps *missing not at
random* depend on the missing value itself. Skipping every row with a gap,
which is called *listwise deletion*, is safe for a regression in a case
those labels do not name. Its slopes are unbiased whenever the chance that
a row has a gap does not depend on the target, given the features. That holds even when the gap depends on the missing feature's
own value (Little 1992; Allison 2001). Its slopes are only noisier, since the
fit sees fewer rows. When the gap does depend on the target, the slopes are biased,
and no fill made from the features repairs it. An imputation that
conditions on the target does, and the library has none (below). Measured in task 228, at 30% missing and features
correlated at 0.5, listwise deletion was biased by 4.5 standard errors of
the full data's slope when the gap depended on `y`. It stayed within 0.2
when the gap depended on another feature or on the feature itself.

**Do not fill a gap with a running mean.** A filled value carries no
variation, so on the filled rows the column is flat. Its slope shrinks
toward zero, and a feature correlated with it absorbs the rest of its
effect, which is the omitted-variable bias of the column the fill erased.
Measured in task 228, a running-mean fill biased the slopes by 0.4 to 6
standard errors, where listwise deletion was unbiased on the same streams.
A forward fill does the same, and adds the error of a stale value. Fill a
value forward only when it truly holds until it changes, such as the last
quote of an instrument. In that case the forward-filled value is the
observation itself.

**A Gram over each pair's own rows would lose fewer rows, and it was
measured and not built, because it is biased where listwise deletion is
not.** It would
keep, for each pair of columns, the sums over the rows where both are
present, so a gap in one column would cost only that column's pairs. Task
228 measured it against listwise deletion. It was biased by 0.6 to 19
standard errors whenever the gap depended on another feature, on the
target or on the feature itself. At 30% missing and features correlated
at 0.9, that was −4.6 when the gap depended on another feature, −5.8 when
on the feature's own value, and −14.2 when on `y`. Listwise deletion was
unbiased in the first two of those cases. The pairwise matrix is also not
a covariance matrix of any one set of rows, so it need not be positive
semi-definite. It was indefinite in 62% to 100% of wide, correlated cases
(20 to 200 columns, correlation 0.9, 30% missing in every column). Every
guard that kept it usable acted as a ridge and removed no bias. The
readiness readings could not see it: `error_inflation` read 1.03 to 1.05
against 13.5 realized. It took 3.6 to 6.1 times as long a row, and 1.5 to
3.5 times the state. Its gain was real only when gaps were completely at
random in a model so wide that listwise deletion kept no rows at all. An
imputation by online expectation-maximization, the one method that
measured unbiased when gaps depended on observed values, is not built
(PLAN task 230).

The recipe plants gaps in `x1` on more than half the rows, in two ways:
at random, and more often when `y` is high. It fits plain least squares
four ways: with the gaps left alone, filled by the running mean, filled
forward, and with `x1` dropped.

```python
import numpy as np
import polars as pl

import polars_online as po

# y = 1.0 x0 + 0.5 x1 + noise, with x0 and x1 correlated at 0.6. Two copies of x1 with gaps on
# half the rows or more: one whose gaps fall at random, one whose gaps come when y is high.
rng = np.random.default_rng(7)
n = 20_000
x0 = rng.standard_normal(n)
x1 = 0.6 * x0 + 0.8 * rng.standard_normal(n)
y = 1.0 * x0 + 0.5 * x1 + rng.standard_normal(n)
gap_at_random = rng.random(n) < 0.6
gap_when_y_high = rng.random(n) < 1 / (1 + np.exp(-2 * y))
pl.DataFrame(
    {
        "x0": x0,
        "y": y,
        "x1_random": np.where(gap_at_random, np.nan, x1),
        "x1_by_y": np.where(gap_when_y_high, np.nan, x1),
    }
).fill_nan(None).write_parquet("gaps.parquet")
lf = pl.scan_parquet("gaps.parquet")


def slopes(query, features):
    """Fit plain least squares (no decay) over the query; return the bank and the coefficients."""
    spec = po.spec.ewridge("ols", targets=["y"], features=features, half_life=float("inf"))
    bank = po.ModelBank([spec])
    bank.fit(query)
    return bank, dict(bank.coef().select("term", "coef").iter_rows())


# What check() says about the copy whose gaps fall at random.
bank, _ = slopes(lf, ["x0", "x1_random"])
shown = bank.check().select("severity", "code", "spec", "column", pl.col("value").round_sig_figs(3))
for row in shown.iter_rows():
    print(*row)

# The slopes of x0 and x1 (1.00 and 0.50 in the data) under four ways of handling the gaps.
print(f"{'gaps':10} {'left alone':>12} {'mean fill':>12} {'forward fill':>12} {'drop x1':>12}")
for x1 in ["x1_random", "x1_by_y"]:
    running_mean = pl.col(x1).cum_sum().forward_fill() / pl.col(x1).cum_count()
    ways = [
        lf,  # each row with a gap skipped
        lf.with_columns(pl.col(x1).fill_null(running_mean)),  # the mean of the values so far
        lf.with_columns(pl.col(x1).fill_null(strategy="forward")),  # the last value seen
    ]
    cells = []
    for query in ways:
        b = slopes(query, ["x0", x1])[1]
        cells.append(f"{b['x0']:.2f} {b[x1]:.2f}")
    cells.append(f"{slopes(lf, ['x0'])[1]['x0']:.2f} -")
    print(f"{x1:10} " + " ".join(f"{c:>12}" for c in cells))
```

```text
warning few_learned ols x1_random 0.398
warning missing ols x1_random 0.602
gaps         left alone    mean fill forward fill      drop x1
x1_random     0.99 0.51    1.21 0.37    1.26 0.15       1.30 -
x1_by_y       0.73 0.37    1.28 0.13    1.29 0.07       1.30 -
```

`check()` finds `x1_random` missing on 60% of the rows, and the model
processing only 40% of them. Left alone, the gaps that fall at random cost
nothing but rows: the slopes are 0.99 and 0.51 for 1.00 and 0.50. Every
fill is worse. The running mean shrinks `x1`'s slope to 0.37 and moves its
effect onto `x0`, which reads 1.21. A forward fill shrinks it to 0.15, and
dropping `x1` leaves `x0` at 1.30, its own effect plus 0.6 of `x1`'s. When
the gaps come with high `y`, the rows kept are not like the rows lost.
Then every way is biased, and leaving the gaps alone is still the least
wrong.

**What to do.** Leave a feature's gaps as nulls, and let the model skip
those rows, unless the gaps depend on the target. When a column skips most
rows and adds little, drop it from the spec, after checking that it is not
correlated with the features that stay. When a target is missing by design
(a label that exists on some rows), `missing` is `info` only. Use
`target_gaps` on `ewridge` and `lasso` to choose how a target's own gaps
are handled.

## Sentinel values

<a id="sentinel"></a>

**Every model learns from a finite sentinel, such as `-999` or `0` for "not
measured", as if it were a measurement.** The models already treat NaN, an infinity and a
magnitude past `1e100` as missing, so a sentinel of `1e308` is skipped. A
sentinel inside that range is learned from as a real value. It is an
outlier planted at one point, and least squares follows outliers.

| code | severity | fires when |
|---|---|---|
| `sentinel` | warning | a column holds `+inf`, `-inf` or a value past `1e100`. Each such row is skipped as a null is: a division by zero upstream, or a sentinel such as `1e308` |
| `sentinel` | info | a column holds NaN, not null. The models read it as missing, but Polars' `fill_null`, `drop_nulls` and `null_count` do not |
| `sentinel` | warning | one value is 1% of a column's usable rows or more, and at least 5 times the largest count any other value could have. A column with 10 values or fewer, or a frozen one, is left out |

The thresholds were set by measurement. `-999` planted on 5% of a
continuous column's rows read 3.9% to 5.8%, at 9.8 to 12 times the next
count's upper bound. A sentinel on 3% of the rows was found on ten seeds
of ten, and one on 2% on one seed, or on ten at `distinct_cap=1024`.
Poisson counts about 50, whose commonest value is 6% of the rows, read
1.23 times the next value at most.

**Why one value on 5% of the rows erases a slope.** A slope is the
covariance of the feature with the target over the feature's variance.
`-999` on 5% of the rows adds about `0.05 × 999² ≈ 50,000` to the
variance of a column whose own variance is 1. It adds almost nothing to
the covariance, because the target is unrelated to which rows hold the
sentinel. So the slope falls by a factor of about 50,000.

```python
import numpy as np
import polars as pl

import polars_online as po

# A feed that writes -999 where x1 was not measured, on 5% of the rows.
rng = np.random.default_rng(11)
n = 20_000
x0 = rng.standard_normal(n)
x1 = rng.standard_normal(n)
y = 1.0 * x0 + 0.5 * x1 + rng.standard_normal(n)
x1_fed = np.where(rng.random(n) < 0.05, -999.0, x1)
pl.DataFrame({"x0": x0, "x1": x1_fed, "y": y}).write_parquet("sentinel.parquet")
lf = pl.scan_parquet("sentinel.parquet")


def run(query):
    """Fit an audit and a ridge regression; return the bank, the final coefficients and R²."""
    audit = po.spec.audit("audit", columns=["x0", "x1", "y"])
    model = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], half_life=2_000.0)
    bank = po.ModelBank([audit, model])
    out = pl.concat(bank.fit_predict_batches(query, chunk_size=5_000)).unnest("ridge").tail(10_000)
    r2 = out.select(1 - (pl.col("y") - pl.col("pred_y")).pow(2).mean() / pl.col("y").var()).item()
    return bank, dict(bank.coef().select("term", "coef").iter_rows()), r2


bank, coef, r2 = run(lf)
shown = bank.check().select("severity", "code", "spec", "column", pl.col("value").round_sig_figs(3))
for row in shown.iter_rows():
    print(*row)
print(f"left alone:     x0 {coef['x0']:.2f}, x1 {coef['x1']:.3f}, R² {r2:.3f}")

# The fix: make the sentinel null, so the model skips those rows.
fixed = lf.with_columns(
    pl.when(pl.col("x1") == -999).then(None).otherwise(pl.col("x1")).alias("x1")
)
bank, coef, r2 = run(fixed)
print(f"-999 made null: x0 {coef['x0']:.2f}, x1 {coef['x1']:.3f}, R² {r2:.3f}")
print(f"findings after the fix: {bank.check().height}")
```

```text
warning sentinel audit x1 0.0461
info heavy_tails audit x1 937.0
left alone:     x0 1.01, x1 0.000, R² 0.442
-999 made null: x0 1.00, x1 0.480, R² 0.549
findings after the fix: 0
```

The audit finds `-999` on 4.6% of `x1`'s usable rows, and the tails it
makes: one value 937 robust standard deviations from the median. The
ridge regression has nothing to say, since to it the column is a column.
Left alone, `x1`'s slope is 0.000 where the data has 0.5, and the
out-of-sample R² over the last 10,000 rows is 0.442. With `-999` made
null, the rows that held it are skipped, the slope is 0.480, and R² is
0.549 of the 0.556 the noise allows. The 5% of rows lost are lost at
random, so the slopes stay unbiased ([Missing values](#missing-values)).

**What to do.** Make a sentinel null in the query before the bank, with
`pl.when(pl.col("x1") == -999).then(None).otherwise(pl.col("x1"))`. Turn
NaN into null with `pl.col("x1").fill_nan(None)`, so that Polars' own null
handling sees it. When a repeated value is real, such as a price that
rests at zero, keep it, and consider an indicator column for it.

## Constant and frozen features

<a id="constant"></a>
<a id="frozen"></a>
<a id="low_support"></a>
<a id="solve_failures"></a>

**A feature that holds one value is indistinguishable from the intercept.**
A column equal to `c` on every row is `c` times the intercept's column of
ones, so the data cannot say how much of the fit is the slope and how much
the intercept. A model with a ridge that does not fade, such as `ewridge`,
sets that slope from the ridge and goes on predicting well. A model whose
prior fades with its sums, such as `rls`, loses the information that
separates the two. A feed that stops and is forward-filled makes a feature
constant over a stretch, which is the same problem in time.

| code | severity | fires when |
|---|---|---|
| `constant` | error | a target took one value on every row: there is nothing to learn |
| `constant` | warning | a feature, or an audited column, took one value on every row |
| `frozen` | warning | from an audit: the share of rows equal to the row before exceeds by 0.05 what independent rows with the column's value frequencies would give, or a run of one value is 10 rows or more and 4 times the longest run that independent rows would show |
| `low_support` | warning | more than half of the smallest coefficient comes from the ridge rather than the data; not repeated for a column already found constant, collinear or shrunk by the ridge |
| `solve_failures` | warning | a solve needed jitter, or failed and kept the previous fit: constant or collinear features, or too few rows for their number. For `lasso` (`info`) its coordinate descent ran out of sweeps more than 10 times; clean data runs out up to 7 times on its first rows |

`frozen`'s thresholds were set by measurement. A feed stopped for the last 10% of the
rows was found, as were a forward fill every 2, 3, 5 or 10 rows and 10 rows
stuck in the middle. Five rows stuck were not. Clean columns repeated the
row before 0.6 points past chance at most (Poisson counts), and ran 4 rows
at most.

**Why `rls` winds up on a frozen feature.** `rls` keeps `A`, the decayed
sum of each row's `z zᵀ` (`z` is the intercept's 1 and the features), and
its prior `delta` decays with it. While a feature holds the value `c`,
each new row adds information only along `(1, c)`, the direction of
`intercept + c·slope`. The information that separates the slope from the
intercept decays by the forgetting factor every row, and nothing renews it.
Once it falls below a rounding step, rounding sets the slope, and the slope
wanders. Control engineers call this *estimator windup* under exponential
forgetting (Åström and Wittenmark 1995). Predictions stay right while the
feature is held, because only `intercept + c·slope` is used. They fail the
moment the feature moves again. A forgetting rule that keeps a floor under
`A` (Kulhavý and Zarrop 1993) was measured for `rls` in task 216. In its
uncentred coordinates, the floor acted as a ridge whose pull grows with
the square of the level, and it was not built. The
[`rls` section](../README.md#rls--recursive-least-squares) of the README
gives the measurements at other levels.

The recipe stops `x1`'s feed at row 300 and brings it back at row 3,300.
Its last value is carried forward in between, and the join that built the
table kept the time each value was observed, in `x1_seen`. At a half-life
of 20 rows, that is 150 half-lives held.

```python
import numpy as np
import polars as pl

import polars_online as po

# x1's feed stops at row 300 and comes back at row 3,300; the join that built the table carried
# its last value forward, and kept the time it was observed in x1_seen.
rng = np.random.default_rng(3)
n = 3_500
t = np.arange(n, dtype=float)
x0 = rng.standard_normal(n)
x1 = rng.standard_normal(n)
y = 1.0 * x0 + 0.5 * x1 + 0.2 * rng.standard_normal(n)
stopped = (t >= 300) & (t < 3_300)
pl.DataFrame(
    {
        "t": t,
        "x0": x0,
        "x1": np.where(stopped, x1[299], x1),
        "x1_seen": np.where(stopped, 299.0, t),
        "y": y,
    }
).write_parquet("frozen.parquet")
lf = pl.scan_parquet("frozen.parquet")


def run(query, model):
    """Fit an audit and the model; print the findings once. Return x1's largest |slope| while it
    was held, and the rms error of the predictions after it moves again."""
    bank = po.ModelBank(
        [po.spec.audit("audit", columns=["x0", "x1"], clock="t", gap_cap=5.0), model]
    )
    out = pl.concat(bank.fit_predict_batches(query, chunk_size=1_000)).unnest(model["name"])
    held = out.filter((pl.col("t") >= 300) & (pl.col("t") < 3_300))
    slope = held.select(pl.col("coef").list.get(2).abs().max()).item()  # coef: [intercept, x0, x1]
    moved = out.filter(pl.col("t") >= 3_300)
    rmse = moved.select((pl.col("pred_y") - pl.col("y")).pow(2).mean().sqrt()).item()
    return bank, slope, rmse


common = dict(
    targets=["y"], features=["x0", "x1"], clock="t", gap_cap=5.0, half_life=20.0, coef_every=1
)
rls = po.spec.rls("rls", **common)
bank, slope, rmse = run(lf, rls)
shown = bank.check().select("severity", "code", "spec", "column", pl.col("value").round_sig_figs(3))
for row in shown.iter_rows():
    print(*row)
print(f"rls, left alone:      largest |slope| while held {slope:.2g}, rms error after {rmse:.2g}")

# The fix: null each value older than one row, so the model skips the rows it did not observe.
stale = pl.col("t") - pl.col("x1_seen") > 1
fixed = lf.with_columns(pl.when(stale).then(None).otherwise(pl.col("x1")).alias("x1"))
bank, slope, rmse = run(fixed, rls)
print(f"rls, stale rows null: largest |slope| while held {slope:.3g}, rms error after {rmse:.3g}")

# Or a model whose ridge does not fade.
bank, slope, rmse = run(lf, po.spec.ewridge("ewridge", **common))
print(f"ewridge, left alone:  largest |slope| while held {slope:.3g}, rms error after {rmse:.3g}")
```

```text
warning frozen audit x1 0.857
info heavy_tails audit x1 16.4
rls, left alone:      largest |slope| while held 5.4e+13, rms error after 3.7e+12
rls, stale rows null: largest |slope| while held 0.481, rms error after 0.234
ewridge, left alone:  largest |slope| while held 0.481, rms error after 0.247
```

The audit finds `x1` repeating the row before on 86% of the rows. Most of
its values are one value, so its robust spread is zero and its largest
robust z is undefined (`None`). `heavy_tails` fires on the column's
kurtosis, 16.4, and reports it as its value. Left alone, `rls`'s slope on `x1`, 0.5 in the
data, wandered past 1e13 while the feed was stopped. When `x1` moved
again, its predictions were off by trillions. With the stale rows made
null, `rls` skips them, the clock still advances, and `gap_cap` caps the
step after the stretch at 5 rows of decay. The slope stays at 0.481 and
the error after the feed returns is 0.234, near the noise of 0.2. An
`ewridge` on the frozen data, left alone, is as good: its ridge does not
fade, and it centres each feature. It raises a `ReadinessWarning`, on
the standard error stream, that `x1`'s coefficient is half ridge, which is
the `low_support` reading.

**What to do.** Make a stale value null in the query before the bank. When
the table records the time each value was observed, compare it with the
row's clock, as the recipe does. Better still, join each feed on the time
its value was observed, so that a stopped feed leaves nulls in place of
copies. Drop a feature that is constant everywhere. For a long-running
stream whose features can go quiet, use `ewridge`, which does not wind up.

## A level far above its spread, and scales apart

<a id="level_over_spread"></a>
<a id="scales_apart"></a>
<a id="ridge_shrinks"></a>

**The models that do not centre a feature lose accuracy when its level is
far from zero, and a ridge shrinks the coefficient of a feature in tiny
units.** A
price near 5,000 that moves by about 1 a row has a level 5,000 times its
spread. A return written as a fraction has a spread near 0.001, so its
variance is a millionth. Four models do not subtract a feature's mean
before they fit it: `rls` and `ftrl` never do, and `kalman`, `sgd` and
`pa` do not with `standardize=False`. Each loses accuracy once the level
passes a limit measured for it.

| code | severity | fires when |
|---|---|---|
| `level_over_spread` | warning | a feature's `abs(mean) / std` passes the limit of a model that does not centre it (the table below) |
| `level_over_spread` | info | the same ratio passes `1e6` for a model that centres its features, which loses nothing; arithmetic on the raw values elsewhere may |
| `scales_apart` | warning | the widest feature's spread over the narrowest's passes the limit of a model that uses one step size for every column (the table below) |
| `ridge_shrinks` | warning | `ewridge`, `huber` or `quantile` without `standardize`: the smallest ridge is a quarter of a feature's variance or more, so it takes a fifth of that coefficient or more |

Each limit is the largest value at which the model's out-of-sample R² fell
by no more than 0.05, at half-lives of 20, 200 and infinite. `rls`'s level
limit is not a constant, for a reason the next paragraphs give: it is
`0.9·sqrt(W/delta_left)`, `W` the weight its fit holds and `delta_left`
the part of its prior that has not decayed. A level or a
spread ratio counts only when it is five standard errors from chance:
from what a centred column, or two of equal spread, would show in that
many rows. That guard keeps a small group quiet. At a quarter of the variance, the
ridge took 0.02 to 0.03 off the R², and at the whole variance 0.14.

| model | level limit | spread-ratio limit |
|---|---|---|
| `ftrl` | 0.5 | 1.5 |
| `kalman`, `standardize=False` | 2 | 2 |
| `sgd`, `standardize=False` | 3 | 3 |
| `pa`, `standardize=False` | 3 | 2 |
| `rls` | `0.9·sqrt(W/delta_left)` | 30 |

The models that centre their features (`ewridge`, `lasso`, `huber`,
`quantile`, and `kalman`, `sgd` and `pa` at their default
`standardize=True`) lost nothing at a level of `1e14` times the spread.
Nor did they at spreads `1e8` apart, except where a narrow feature met a
ridge that was not standardized, which `ridge_shrinks` finds.

**Why a level loses precision.** A double keeps 53 bits, about 16 decimal
digits. A value at level `L` with spread `s` spends `log10(L/s)` of them
on the level, so it keeps about `16 − log10(L/s)` digits of its variation
(Goldberg 1991). A sum-form variance, `Σx²/n − x̄²`, subtracts two numbers
that agree in their first `2·log10(L/s)` digits, and loses those digits
(Chan, Golub and LeVeque 1983). The models that centre keep their running
means compensated, as a pair of doubles, and accumulate co-moments about
them, so they lose nothing at a level. Subtracting any origin
near the level does the same for a model that does not centre: precision
depends on `|x − origin| / s`, whichever origin is chosen.

**Why a level breaks `rls`'s prior.** In uncentred coordinates, a feature
at level `L` and the intercept's column of ones point almost the same way.
The direction that separates the slope from the intercept carries
information of only about `n·s²/L²` after `n` rows. `rls` starts from a
prior `delta·I` in those same coordinates, `delta = 1` by default. Without
a decay the prior never fades, and once `L/s` passes about `sqrt(n/delta)`
it outweighs the data in that direction and pins the slope near zero.
Measured at 20,000 rows and no decay, a level of 100 spreads kept an R² of
0.42 against 0.51, and 1,000 spreads kept 0.015. With `delta=1e-6`, both
read 0.51. Under a decay the prior fades with the sums, so the same levels
left `rls`'s R² where `ewridge`'s was at half-lives of 20 and 200. A short
stream at a long half-life is still in the prior's shadow: over 2,000 rows
at a half-life of 200, a thousandth of `delta` is left beside a weight of
289, and a level of 10,000 spreads cost 0.16 of R² over the last tenth.

So the check reads `r = (L/s)·sqrt(delta_left/W)` at the end of the
stream. `delta_left` is `delta·(1 − settled_frac)`, all of `delta` without
decay, and `W` the weight the fit holds, `settled_frac` times
`weight_sum_settled`, or `weight_sum` without decay. It is the weight and
not the Kish size: weights 100 times larger moved the harm to levels 10
times higher. The R² lost over the last tenth of a stream was one curve in
`r`, at half-lives of 200, 2,000 and infinite and over 2,000 and 20,000
rows: 0.007 to 0.018 at 0.55 to 0.71, 0.013 to 0.037 at 0.8, 0.10 to 0.11
at 1.8 to 2.2, and all of the feature's share past 5. It stays at or below
0.05 up to 0.9, the limit.

**Why a step size needs a scale.** `sgd`, `pa` and `ftrl` move the
coefficients a step along the gradient on each row. Least mean squares
converges only when the step is below `2/λmax`, where `λmax` is the
largest eigenvalue of `E[z zᵀ]`, and its speed is set by the eigenvalue
ratio `λmax/λmin` (Haykin 2002). A feature at level 5,000 makes `λmax` about
25 million, so a step tuned for unit features diverges. Features whose
standard deviations are 1,000 apart make that ratio about a million, so the narrow feature's
coefficient barely moves. `kalman` and `rls` are second order and need no
step, but a ridge in the features' squared units penalizes a narrow
feature's coefficient more for the same effect.

The recipe fits a price near 5,000 and a return written as a fraction with
three models: `rls` with no decay, `sgd` with `standardize=False`, and
`ewridge` at its defaults.

```python
import numpy as np
import polars as pl

import polars_online as po

# A price near 5,000 that moves by about 1 a row, and a return written as a fraction (spread 0.001).
rng = np.random.default_rng(5)
n = 20_000
price = 5_000 + rng.standard_normal(n)
ret = 0.001 * rng.standard_normal(n)
y = 0.5 * (price - 5_000) + 300 * ret + 0.5 * rng.standard_normal(n)
pl.DataFrame({"price": price, "ret": ret, "y": y}).write_parquet("level.parquet")
lf = pl.scan_parquet("level.parquet")

specs = [
    po.spec.rls("rls", targets=["y"], features=["price", "ret"], half_life=float("inf")),
    po.spec.sgd(
        "sgd", targets=["y"], features=["price", "ret"], half_life=500.0, standardize=False
    ),
    po.spec.ewridge("ewridge", targets=["y"], features=["price", "ret"], half_life=500.0),
]


def run(query):
    """Fit every spec; print check()'s findings and each spec's R² over the last 10,000 rows."""
    bank = po.ModelBank(specs)
    out = pl.concat(bank.fit_predict_batches(query, chunk_size=5_000)).tail(10_000)
    shown = bank.check().select(
        "severity", "code", "spec", "column", pl.col("value").round_sig_figs(3)
    )
    for row in shown.iter_rows():
        print(*row)
    for s in specs:
        pred = pl.col(s["name"]).struct.field("pred_y")
        r2 = out.select(1 - (pl.col("y") - pred).pow(2).mean() / pl.col("y").var()).item()
        print(f"{s['name']:8} R² {r2:.3g}")


run(lf)
print("-- the price less its first value, the return in thousandths --")
run(lf.with_columns(pl.col("price") - pl.col("price").first(), pl.col("ret") * 1_000))
```

```text
warning level_over_spread rls price 4960.0
warning scales_apart rls ret 1020.0
warning level_over_spread sgd price 4960.0
warning scales_apart sgd ret 1020.0
warning ridge_shrinks ewridge ret 1.02
rls      R² 0.00487
sgd      R² -2.06e+09
ewridge  R² 0.541
-- the price less its first value, the return in thousandths --
rls      R² 0.579
sgd      R² 0.57
ewridge  R² 0.578
```

`check()` finds the price 4,960 sample spreads from zero and the two
sample spreads 1,020 times apart (5,000 and 1,000 in the generator), for
both uncentred models. For `ewridge`, which centres,
it finds only that its default ridge, `1e-6`, is as large as the return's
variance. Left alone, `rls`'s R² over the last 10,000 rows is 0.005, and
`sgd` diverges. `rls`'s prior pins both slopes near zero: the price's,
whose direction apart from the intercept carries little information at a
level, and the return's, whose information `n·s²` is 0.02 against
`delta = 1`. A ridge with that prior solved in numpy reads the same
0.00487, so it is the prior and not lost precision. Measuring the price
from its first value alone reads 0.435, and the return in thousandths
alone 0.149: it takes both. `ewridge` reads
0.541, where the noise allows about 0.576: its ridge takes about half of the
return's coefficient. With the price measured from its first value and the
return in thousandths, `check()` finds nothing, and all three models read
0.57 to 0.58.

**What to do.** Subtract an origin from a level in the query before the
bank, such as `pl.col("price") - pl.col("price").first()` or a known
reference price. When the change is what matters, difference it with
`diff()` instead ([Random walks](#random-walks)). Put features in units of their
spread, by hand as the recipe does or with the windowed standardization in
the README's
[Features in units of their spread](../README.md#features-in-units-of-their-spread).
Or use `standardize=True`, the default for `kalman`, `sgd` and `pa`, and an
option on `ewridge`, `huber` and `quantile`.

## Duplicated and collinear features

<a id="duplicate"></a>
<a id="collinear"></a>

**When one feature is nearly a linear combination of the others, the
predictions stay good and the coefficients mean nothing.** The data
determine only the combined effect. How it is split among the
near-duplicates is set by the ridge and by noise, and it moves whenever the
noise does.

| code | severity | fires when |
|---|---|---|
| `duplicate` | warning | from an audit with `pairs=True`: two columns correlate at 0.999 or more |
| `collinear` | warning | the largest variance inflation factor, or the condition index of the design the model solves, over an `ewridge`'s or `lasso`'s features passes 30; `info` for an `ew_cov`. It is checked once the Gram holds ten Kish rows a coefficient |

Clean designs reached a variance inflation factor of 5.6 and a condition
index of 7.6, with ten features correlated at 0.8. A near-duplicate
`x0 + 0.1·N(0, 1)`, a correlation of 0.995, read 92 to 113. At `0.3`, a
correlation of 0.957 to 0.963, it read 10.9 to 13.7 and was not flagged. For `duplicate`, a copy, a copy times 100
plus 3, and a copy plus noise of 0.03 of its spread (a correlation of
0.9995) were found. A copy plus 0.05 was not. Clean columns correlated at
0.94 at most.

**Why.** A coefficient's variance is the residual variance over `n`
times the feature's variance, multiplied by its *variance inflation factor*, `1/(1 − R²ⱼ)`.
Here `R²ⱼ` is the R² of that feature regressed on the others. At a
correlation of 0.999 with another feature, the factor is about 500, so the
coefficient's standard error is 22 times what it would be alone.
Belsley's condition indexes find a dependency among three or more columns
that no single pair shows (Belsley, Kuh and Welsch 1980). Each is the
square root of the ratio of the scaled design's largest eigenvalue to one
of the others. `check()` reads the design each model solves. A model with
an intercept centres its features, so its design is their correlation
matrix, and two independent features that share a level are not
collinear: two at 5,000 and 3,000 read 1.0 where the uncentred index read
5,061. A model fitted through the origin (`fit_intercept=False`) solves
the raw design, where those two are nearly proportional, so it reads
Belsley's uncentred index, which `po.gram.condition` computes, since
centring would hide a dependency on the constant. `po.gram.vif` computes
the factors from a saved Gram.

The recipe records one quantity twice, in dollars as `x0` and in cents as
`x2`, with a little noise of its own in `x2`.

```python
import numpy as np
import polars as pl

import polars_online as po

# x0 in dollars and x2 the same quantity in cents, recorded with a little noise of its own.
rng = np.random.default_rng(9)
n = 20_000
x0 = rng.standard_normal(n)
x1 = rng.standard_normal(n)
x2 = 100 * x0 + 0.5 * rng.standard_normal(n)
y = 1.0 * x0 + 0.5 * x1 + rng.standard_normal(n)
pl.DataFrame({"x0": x0, "x1": x1, "x2": x2, "y": y}).write_parquet("collinear.parquet")
lf = pl.scan_parquet("collinear.parquet")


def run(features):
    """Fit an audit with pairs and a ridge regression; print the findings, R², and x0's
    coefficient over the last 10,000 rows, alone and with x2's share of the same effect."""
    audit = po.spec.audit("audit", columns=features, pairs=True)
    model = po.spec.ewridge(
        "ridge", targets=["y"], features=features, half_life=1_000.0, coef_every=1
    )
    bank = po.ModelBank([audit, model])
    out = pl.concat(bank.fit_predict_batches(lf, chunk_size=5_000)).unnest("ridge").tail(10_000)
    shown = bank.check().select(
        "severity", "code", "spec", "column", pl.col("value").round_sig_figs(3)
    )
    for row in shown.iter_rows():
        print(*row)
    r2 = out.select(1 - (pl.col("y") - pl.col("pred_y")).pow(2).mean() / pl.col("y").var()).item()
    slope = out.select(pl.col("coef").list.get(1)).to_series()  # coef: [intercept, x0, x1, (x2)]
    whole = slope + 100 * out["coef"].list.get(3) if "x2" in features else slope
    print(
        f"R² {r2:.3f}; x0's coefficient from {slope.min():.2f} to {slope.max():.2f}; "
        f"x0's whole effect from {whole.min():.2f} to {whole.max():.2f}"
    )


run(["x0", "x1", "x2"])
print("-- x2 dropped --")
run(["x0", "x1"])
```

```text
warning duplicate audit x2 1.0
warning collinear ridge x0 36500.0
R² 0.556; x0's coefficient from -5.81 to 9.49; x0's whole effect from 0.96 to 1.04
-- x2 dropped --
R² 0.556; x0's coefficient from 0.96 to 1.04; x0's whole effect from 0.96 to 1.04
```

The audit finds `x2` a copy of `x0` (correlation 1.0 to three figures),
and the model finds a variance inflation factor of 36,500. The R² is 0.556
either way, as good as the noise allows. With both columns, `x0`'s own
coefficient wanders from −5.81 to 9.49 over the last 10,000 rows, while
the whole effect of `x0`, its coefficient plus 100 times `x2`'s, stays
between 0.96 and 1.04. With `x2` dropped, `x0`'s coefficient is that
effect.

**What to do.** Drop one of a pair of duplicates from the spec. For a
group of collinear features, keep the one you can explain, combine them
into one (an average, or a difference that means something), or increase the
ridge. When only the predictions matter, collinearity does no harm.

## Categories stored as numbers

<a id="few_values"></a>

**A linear model reads a number as a quantity, so a code stored as a
number becomes a straight line through the codes.** A venue numbered 0 to
4, a day of the week or a flag is a category. Its effect on the target has
no reason to rise by the same amount from code 1 to code 2 as from code 3
to code 4.

| code | severity | fires when |
|---|---|---|
| `few_values` | info | from an audit: a column takes 10 distinct values or fewer over 100 usable rows or more |

It is `info`, because a flag that is truly 0 or 1 is fine as a
number: one indicator is its own one-hot encoding. The distinct count is
exact up to the audit's `distinct_cap`, 256 by default.

```python
import numpy as np
import polars as pl

import polars_online as po

# A venue code 0-4 stored as a number; each venue adds its own amount to y, in no order.
rng = np.random.default_rng(12)
n = 20_000
venue = rng.integers(0, 5, n)
x = rng.standard_normal(n)
y = 0.5 * x + np.array([0.0, 1.0, -1.0, 0.5, 2.0])[venue] + rng.standard_normal(n)
pl.DataFrame({"venue": venue.astype(float), "x": x, "y": y}).write_parquet("venues.parquet")
lf = pl.scan_parquet("venues.parquet")


def run(query, features):
    """Fit an audit of the columns as they arrive, and a ridge regression on the features."""
    audit = po.spec.audit("audit", columns=["x", "venue"])
    model = po.spec.ewridge("ridge", targets=["y"], features=features, half_life=2_000.0)
    bank = po.ModelBank([audit, model])
    out = pl.concat(bank.fit_predict_batches(query, chunk_size=5_000)).unnest("ridge").tail(10_000)
    shown = bank.check().select(
        "severity", "code", "spec", "column", pl.col("value").round_sig_figs(3)
    )
    for row in shown.iter_rows():
        print(*row)
    r2 = out.select(1 - (pl.col("y") - pl.col("pred_y")).pow(2).mean() / pl.col("y").var()).item()
    print(f"R² {r2:.3f}")


run(lf, ["x", "venue"])
print("-- one indicator per venue but the first --")
indicators = [(pl.col("venue") == v).cast(pl.Float64).alias(f"venue_{v}") for v in range(1, 5)]
run(lf.with_columns(indicators), ["x", "venue_1", "venue_2", "venue_3", "venue_4"])
```

```text
info few_values audit venue 5.0
R² 0.223
-- one indicator per venue but the first --
info few_values audit venue 5.0
R² 0.562
```

The audit finds 5 values in `venue`, and goes on finding them after the
fix, since it reads the column as it arrives. Read as a quantity, `venue`
gives an R² of 0.223. With one indicator for each venue but the first,
whose effect the intercept takes, the R² is 0.562, as good as the noise
allows.

**What to do.** One-hot encode a category in the query before the bank:
one column per value but one, each `(pl.col("venue") == v).cast(pl.Float64)`.
Leave out one value, or the indicators sum to the intercept's column, which
is a [constant feature](#constant-and-frozen-features) made of several.
With many values, consider a separate group per value (the spec's `group`)
instead.

## Random walks

<a id="random_walk"></a>

**A regression of one random walk on another finds a relation that is not
there.** A random walk is a level whose changes are independent, such as a
price. Two random walks that have nothing to do with each other wander
apart and together by chance over any finite stretch. Least squares reads
the shared drift as a slope, with a t-statistic far past any significance
line (Granger and Newbold 1974).

| code | severity | fires when |
|---|---|---|
| `random_walk` | warning | from an audit: the Dickey-Fuller statistic of the column's regression on its own previous row is above −3.5, over 100 pairs of rows or more (the 1% critical value is −3.43) |

Random walks of 300 and 2,000 rows read no lower than −3.2, and the clean
shapes of 2,000 rows read no higher than −6.5 (columns that persist at
0.95 a row).
Groups of 50 rows, which the 100-row guard leaves out, read −4.7. A column
that persists at 0.99 a row over 2,000 rows reads as a random walk on
seven seeds of ten, and one at 0.98 on none.

**Why.** The t-statistic assumes residuals that are independent from row
to row. Regressing one random walk on another leaves residuals that are a
random walk themselves, so the effective number of independent rows is a
handful, not `n`. The statistic grows with `sqrt(n)` instead of settling,
so a longer stream makes the false relation look more certain (Phillips
1986). Differencing both series turns each into its independent changes,
and the ordinary statistics hold again. The Dickey-Fuller statistic
(Dickey and Fuller 1979) is the t-statistic of `ρ − 1` in a column's
regression on its own previous value. A random walk has `ρ = 1` and reads
near zero, and independent rows read about `−sqrt(n)`. Measured over 200
pairs of independent random walks of 5,000 rows, 98.5% had a slope with
`|t| > 1.96`, the median `|t|` was 33, and the median R² was 0.18. The
same pairs differenced had `|t| > 1.96` in 2.5%.

```python
import numpy as np
import polars as pl

import polars_online as po

# Two random walks that have nothing to do with each other.
rng = np.random.default_rng(2)
n = 5_000
pl.DataFrame(
    {"x": rng.standard_normal(n).cumsum(), "y": rng.standard_normal(n).cumsum()}
).write_parquet("walks.parquet")
lf = pl.scan_parquet("walks.parquet")


def run(query):
    """Fit an audit and plain least squares; print the findings, the slope and its t."""
    audit = po.spec.audit("audit", columns=["x", "y"])
    ols = po.spec.ewridge("ols", targets=["y"], features=["x"], half_life=float("inf"), ridge=1e-9)
    bank = po.ModelBank([audit, ols])
    bank.fit(query)
    shown = bank.check().select(
        "severity", "code", "spec", "column", pl.col("value").round_sig_figs(3)
    )
    for row in shown.iter_rows():
        print(*row)
    g = bank.gram("ols")[0]  # the model's running sums
    coef = po.gram.solve(g, ridge=1e-9)  # [intercept, slope]
    stats = po.gram.coef_stats(g, coef)  # the slope's t, and the R², from the same sums
    print(f"slope {coef[1]:.3f}, t {stats['t'][1]:.1f}, R² {stats['r2']:.3f}")


run(lf)
print("-- both differenced --")
run(lf.select(pl.col("x").diff(), pl.col("y").diff()))
```

```text
warning random_walk audit x -1.06
warning random_walk audit y -1.84
slope -0.228, t -10.9, R² 0.023
-- both differenced --
slope -0.014, t -1.0, R² 0.000
```

The audit finds both columns indistinguishable from random walks
(Dickey-Fuller −1.06 and −1.84). On the levels, the slope is −0.228 with a
t of −10.9, a relation that does not exist. On the changes, it is −0.014
with a t of −1.0, which is noise, as it should be. The first row of each
differenced column is null, so the model skips it.

**What to do.** Difference a level with `diff()` in the query before the
bank, or take its deviation from a moving mean, such as
`pl.col("x") - po.ewm_mean("x", ...)`. Model the change of the target on
the changes of the features. A level that truly belongs in the model, such
as a spread between two prices that reverts, is not a random walk, and its
Dickey-Fuller statistic says so.

## Heavy tails

<a id="heavy_tails"></a>

**A squared loss follows the largest rows, so a few extreme residuals move
a fit, its spread and anything built on the spread.** That includes the
coefficients, `sigma` (the residuals' recent standard deviation), the
Gaussian interval `pred ± z·sigma`, and `drift`, which tests residuals in
units of `sigma`.

| code | severity | fires when |
|---|---|---|
| `heavy_tails` | info | from an audit: an excess kurtosis of 5 or more, or a value 10 robust standard deviations or more from the median (the median absolute deviation, scaled by 1.4826); its value is the robust z where that fired, the kurtosis otherwise |

Clean columns reached an excess kurtosis of 3.8 (in 50-row groups) and a
robust z of 5.6. Over 2,000 rows on ten seeds, Student's t with 3 degrees
of freedom, whose kurtosis is infinite, read a largest robust z of 10 to
28 and an excess kurtosis of 7 to 60. One value at 30 standard deviations
read a robust z of 29.

**Why.** The influence of one row on a least-squares fit grows without
bound with its residual: a residual twice as large moves the fit twice as
far (Huber 1964). Huber's loss is squared near zero and linear past
`huber_delta`, so a residual's influence is capped, and quantile
regression caps it at a constant. `sigma` is an exponentially weighted
root mean square, so one residual of `k` times the usual size inflates it
for several half-lives, the more the larger `k` is. When the residuals have infinite variance, as
Student's t with 2 degrees of freedom does, `sigma` never settles. A
Gaussian interval then over-covers in calm stretches, since `sigma` is
inflated by the last large row. `drift` fires on stable data whose tails
are heavier than its threshold allows: Student's t with 3 degrees of
freedom flagged 3 to 10 rows in every 200,000 at `drift_threshold=20`. A
conformal interval tracks a quantile of `|resid|` directly, so its
long-run coverage is the one asked for whatever the tails (Vovk, Gammerman and
Shafer 2005).

```python
import numpy as np
import polars as pl

import polars_online as po

# y = 0.5 x + noise from Student's t with 2 degrees of freedom: a stable relation, rare huge rows.
rng = np.random.default_rng(4)
n = 50_000
x = rng.standard_normal(n)
y = 0.5 * x + 0.3 * rng.standard_t(2, n)
pl.DataFrame({"t": np.arange(n, dtype=float), "x": x, "y": y}).write_parquet("tails.parquet")
lf = pl.scan_parquet("tails.parquet")

common = dict(targets=["y"], features=["x"], clock="t", gap_cap=5.0, half_life=200.0, coef_every=1)
specs = [
    po.spec.audit("audit", columns=["x", "y"]),
    po.spec.ewridge(
        "ewridge", **common, emit_sigma=True, emit_drift=True, drift_threshold=20.0, conformal=0.9
    ),
    po.spec.huber("huber", **common),
]
bank = po.ModelBank(specs)
out = pl.concat(bank.fit_predict_batches(lf, chunk_size=10_000)).tail(40_000)
shown = bank.check().select("severity", "code", "spec", "column", pl.col("value").round_sig_figs(3))
for row in shown.iter_rows():
    print(*row)

ew = out.unnest("ewridge")
gaussian = ew.select(
    ((pl.col("y") - pl.col("pred_y")).abs() <= 1.645 * pl.col("sigma_y")).mean()
).item()
conformal = ew.select(pl.col("y").is_between(pl.col("lo_y"), pl.col("hi_y")).mean()).item()
flagged = ew.select(pl.col("drift_y").sum()).item()
print(f"90% interval from sigma covers {gaussian:.3f}; conformal covers {conformal:.3f}")
print(f"sigma from {ew['sigma_y'].min():.3g} to {ew['sigma_y'].max():.3g}; drift flagged {flagged}")
for name in ["ewridge", "huber"]:
    slope = out.select(pl.col(name).struct.field("coef").list.get(1))  # coef: [intercept, x]
    within = slope.select((pl.all() - 0.5).abs().quantile(0.99)).item()
    print(f"{name:8} slope: 99% of rows within {within:.3f} of 0.5")
```

```text
info heavy_tails audit y 175.0
90% interval from sigma covers 0.952; conformal covers 0.899
sigma from 0.52 to 6.99; drift flagged 12
ewridge  slope: 99% of rows within 0.224 of 0.5
huber    slope: 99% of rows within 0.057 of 0.5
```

The audit finds `y`'s largest value 175 robust standard deviations from
its median. Over the last 40,000 rows, the 90% Gaussian interval built
from `sigma` covered 95.2% of them, too wide on average, because
`sigma` ranged from 0.52 to 6.99 as the large rows came and went. The
conformal interval covered 89.9%. `drift` flagged 12 rows of a relation
that never changed. `huber`'s slope stayed within 0.057 of 0.5 on 99% of
the rows, and `ewridge`'s within 0.224.

**What to do.** Fit `huber` or `quantile` when the coefficients matter,
and use `conformal` for an interval. Increase `drift_threshold` until a
stretch known to be stable stays unflagged. Clipping a feature at a known
bound, with `pl.col("x").clip(lo, hi)` in the query before the bank, keeps
one bad value from moving the fit. Clip a target only when the extremes
are errors, since clipping real outcomes biases the fit toward the middle.

## Leakage

**A feature that contains the target makes every out-of-sample number a
fiction.** It can be the target itself, joined under another name. It can
be a window computed over rows that the target looks ahead to. Or it can
be a label learned before it would have been known (Kaufman, Rosset and
Perlich 2012). `check()` finds the first two when the feature is nearly
the target. It cannot find the third, which is a matter of when a label is
learned, and the spec's `embargo` is its fix.

| code | severity | fires when |
|---|---|---|
| `leakage` | warning | a feature correlates with a target at 0.9999 or more, from an `ewridge`'s, `lasso`'s or `ew_cov`'s Gram, or a `marginal`'s pairs |

Clean fits reached a correlation of 0.991 at an R² of 0.99. A feature
that is the target plus a thousandth of its spread read 0.999999. So the
line catches a feature that is the target, and not one that merely
predicts it well. It is a warning, not an error, because a correlation
cannot tell a copy of the target from two other things. A relation
measured with little noise passes it: any fit with an R² of 0.9998 or
more, such as `y = 2x + 0.01 N(0, 1)` at 0.99999. So does a random walk
against its own previous row once the stream is long. With no decay it
passed on one seed of ten at 10,000 rows, five at 30,000 and nine at
100,000.

**Why a look-ahead target needs its embargo.** A target that looks `k`
rows ahead, such as the mean return of the next ten rows, is known only
`k` rows after its own row. A model that learns it at its own row learns
returns from the rows it has yet to predict. Its coefficients then carry
part of the very outcomes they are scored on, and a feature built from
recent returns will seem to predict them. An `embargo` of `k` learns each
row only once its label would have arrived (López de Prado 2018, ch. 7).
A target written as a [window expression](../README.md#windows-as-a-models-inputs-and-target)
that looks ahead carries its own delay, and `fit_predict` refuses an
embargo shorter than its window. A target made as a column, as here, does
not, so the embargo is the caller's to set
([Labels that arrive late](../README.md#labels-that-arrive-late)).

```python
import numpy as np
import polars as pl

import polars_online as po

# Returns with no structure: nothing in the past predicts the next ten rows.
rng = np.random.default_rng(8)
n = 20_000
pl.DataFrame({"t": np.arange(n, dtype=float), "r": rng.standard_normal(n)}).write_parquet(
    "returns.parquet"
)
lf = pl.scan_parquet("returns.parquet").with_columns(
    fwd=pl.col("r").rolling_mean(10).shift(-10),  # the target: the mean return of the next ten rows
    past_bad=pl.col("r")
    .rolling_mean(10)
    .shift(-10),  # meant as the last ten rows' mean, shifted the wrong way
    past=pl.col("r").rolling_mean(10),  # the last ten rows' mean, this row included
)


def run(feature, **embargo):
    """Fit a ridge regression of fwd on one feature; print the findings and the R²."""
    spec = po.spec.ewridge(
        "m", targets=["fwd"], features=[feature], clock="t", gap_cap=5.0, half_life=50.0, **embargo
    )
    bank = po.ModelBank([spec])
    out = pl.concat(bank.fit_predict_batches(lf, chunk_size=5_000)).unnest("m")
    out = out.drop_nulls(["fwd", "pred_fwd"])
    r2 = out.select(
        1 - (pl.col("fwd") - pl.col("pred_fwd")).pow(2).mean() / pl.col("fwd").var()
    ).item()
    shown = bank.check().select(
        "severity", "code", "spec", "column", pl.col("value").round_sig_figs(3)
    )
    for row in shown.iter_rows():
        print(*row)
    print(f"{feature}, {'embargo 10' if embargo else 'no embargo'}: R² {r2:.3f}")


run("past_bad")
run("past")
run("past", embargo=10.0)  # learn each row ten rows later, when its target is known
```

```text
warning leakage m past_bad 1.0
past_bad, no embargo: R² 1.000
past, no embargo: R² 0.061
past, embargo 10: R² -0.114
```

`check()` finds `past_bad` correlating with the target at 1.0, and the
R² of 1.000 confirms it. With the feature fixed to look back, and
no embargo, a stream with nothing to predict shows an R² of 0.061, which
is leakage that no check can see. With `embargo=10.0`, the R² is −0.114:
a model chasing noise at a half-life of 50 rows predicts worse than the
stream's mean would, which is the honest answer for data with nothing in
it.

**What to do.** Build each feature from rows at or before its own row, and
check every `shift(-k)` in the query. Give a spec whose target looks ahead
an `embargo` as long as the look-ahead, or write the target as a window
expression, which carries its own. An R² that is too good on data you
expected to be hard is itself a finding.

## The clock

<a id="step_back"></a>
<a id="resets"></a>
<a id="duplicate_stamps"></a>
<a id="gaps"></a>
<a id="irregular_clock"></a>

**A model with a clock weighs each row by the time that has passed, so a
clock that runs backwards, repeats or stalls changes what the model
learns.** Each row's weight halves every `half_life` of clock. Two rows
with the same stamp weigh the same. A clock that steps back has no
elapsed time to decay by, and a long gap would age the fit by all of it.
The spec's clock policy decides each case, and `check()` reports what the
policy met
([Sessions, gaps and steps back](../README.md#sessions-gaps-and-steps-back)).

| code | severity | fires when |
|---|---|---|
| `step_back` | warning | the clock stepped back within a group, each a restart or a late row under `restart_after_step_back` |
| `resets` | info | a stream started over, at a `session_gap="reset"` or a step back: each start forgets what came before |
| `duplicate_stamps` | info | from an audit: a row shares the clock of the row before |
| `gaps` | info | from an audit: a step between rows reaches `gap_cap` |
| `irregular_clock` | info | from an audit: the coefficient of variation of the regular steps is 0.5 or more |

For `irregular_clock`, steps drawn from an exponential, which is what
arrivals at random times give, read 0.96 to 1.04. A clock whose steps
vary uniformly between 0.5 and 1.5 read 0.29.

**What each clock event does.** With `restart_after_step_back` unset, the
default, the bank refuses a chunk with any step back, before it learns any
row of it. A step back larger than the setting restarts the model, which
forgets everything, and one no larger is refused as a late row. Equal
stamps are zero apart: no decay passes between them, and a window holds
them together. That is right for two trades in one millisecond, and wrong
for a row the feed sent twice, which then counts double. A step that
reaches `gap_cap` is capped, so a quiet hour ages the fit by no more than
`gap_cap`. It is also a break, where a model that keeps past rows by
position, as a lag does, empties them. On an irregular clock, a half-life
in clock units weighs rows by time, as it should. A lag or a window counted
in rows means a different span of time on each row.

The recipe makes trades at random times, stamped in whole milliseconds. The
feed sent 1% of them twice, and delivered 0.5% a few rows late.

```python
import numpy as np
import polars as pl

import polars_online as po

# Trades at random times, in whole milliseconds, so some share a stamp. The feed sent 1% of them
# twice, and delivered 0.5% of them a few rows after their place.
rng = np.random.default_rng(6)
n = 10_000
t = np.cumsum(rng.exponential(20.0, n)).round()
x = rng.standard_normal(n)
trades = pl.DataFrame({"t": t, "x": x, "y": 0.5 * x + rng.standard_normal(n)})
place = np.arange(n, dtype=float)
late = rng.random(n) < 0.005
place[late] += rng.integers(2, 6, late.sum())
resent = trades.with_columns(place=place + 0.5).filter(pl.Series(rng.random(n) < 0.01))
pl.concat([trades.with_columns(place=place), resent]).sort("place").drop("place").write_parquet(
    "trades.parquet"
)
lf = pl.scan_parquet("trades.parquet")

audit = po.spec.audit("audit", columns=["x", "y"], clock="t", gap_cap=200.0)
model = po.spec.ewridge(
    "ridge", targets=["y"], features=["x"], clock="t", gap_cap=200.0, half_life=5_000.0
)

# Left alone: every step back is refused.
try:
    po.ModelBank([audit, model]).fit(lf)
except ValueError as err:
    print(str(err).split(";")[0])

# Told that any step back starts the stream over, the model restarts at each late row.
# (The audit is left out: it reads the same clock, and would refuse the stream too.)
restarting = po.spec.ewridge(
    "ridge",
    targets=["y"],
    features=["x"],
    clock="t",
    gap_cap=200.0,
    half_life=5_000.0,
    restart_after_step_back=0.0,
)
bank = po.ModelBank([restarting])
bank.fit(lf)
shown = bank.check().select("severity", "code", "spec", "column", pl.col("value").round_sig_figs(3))
for row in shown.iter_rows():
    print(*row)

# The fix: drop the rows sent twice, then put the rest in clock order.
fixed = lf.unique(maintain_order=True).sort("t", maintain_order=True)
bank = po.ModelBank([audit, model])
bank.fit(fixed)
shown = bank.check().select("severity", "code", "spec", "column", pl.col("value").round_sig_figs(3))
for row in shown.iter_rows():
    print(*row)
fed, learned = bank.summary("ridge").select("rows_fed", "rows_learned").row(0)
print(f"rows fed {fed}, learned {learned}")
```

```text
spec "audit": clock column "t" goes backwards by 1 at row 185 (restart_after_step_back is unset, so every step back is refused)
warning step_back ridge t 65.0
info resets ridge t 65.0
info duplicate_stamps audit t 0.0248
info irregular_clock audit t 0.969
rows fed 10000, learned 10000
```

Left alone, the bank refuses the stream at its first late row, and names
the row. Told that any step back starts the stream over, the model
restarts at each late row: `check()` finds the steps back and the
restarts, each of which threw away everything learned. With the copies
dropped and the rows sorted by their clock, every row is learned once. The
audit then finds 2.5% of the rows sharing a millisecond with the row
before, which is two trades at once, and steps whose coefficient of
variation is 0.97, which is arrivals at random. Both are `info`, and
both are what the data are.

**What to do.** Sort each group by its clock in the query before the bank,
with `sort("t", maintain_order=True)`, and drop rows sent twice with
`unique(maintain_order=True)`, or with `unique(subset=[...])` on the
columns that identify a row. Set `restart_after_step_back` only when a step
back truly starts the stream over, such as a replayed day, and give the
stream a `session` column when its clock restarts at each session. Set
`gap_cap` to the longest quiet stretch that should age the fit in full.

## Too little data for the fit

<a id="few_rows"></a>
<a id="group_sizes"></a>
<a id="never_settled"></a>
<a id="below_min_weight"></a>
<a id="withheld"></a>

**A group needs at least as many rows as its fit has coefficients, and a
decayed fit needs a half-life long enough to gather the weight
`min_weight` asks for.** A bank fits each group on its own rows, so a
group that trades twice cannot determine three coefficients. A half-life
too short for the stream's row rate caps the weight a model can ever
gather. A half-life far longer than the stream leaves the fit resting on a
window that has not filled.

| code | severity | fires when |
|---|---|---|
| `few_rows` | warning | a regression's group learned from fewer rows than its fit has coefficients for each target |
| `group_sizes` | info | the largest group learned from 100 times the rows of the smallest or more |
| `never_settled` | warning | a stream has seen less than one half-life of clock (`settled_frac` below 0.5) |
| `below_min_weight` | warning | the weight a stream settles at, `weight_sum_settled`, is below the spec's `min_weight`, so its predictions stay null |
| `withheld` | warning | a stream's last row had its predictions withheld, with the reason |

**Why.** With `k` coefficients and fewer than `k` rows, the normal
equations have no unique solution, and the ridge alone sets the fit. Under
a decay, a stream whose rows come one clock unit apart can gather at most
`1/(1 − 2^(−1/h))` of weight at half-life `h`. That is 3.4 at `h = 2`, so
a `min_weight` of 10 is never met. After a time `T` on the clock, a stream
has gathered `1 − 2^(−T/h)` of that ceiling, which is `settled_frac`. The
README's [Warm-up](../README.md#warm-up) section has the gates.

```python
import numpy as np
import polars as pl

import polars_online as po

# Forty stocks with the same relation y = 0.5 x0 + 0.3 x1 + noise: four trade often, the rest twice.
rng = np.random.default_rng(10)
rows = [5_000] * 4 + [2] * 36
stock = np.repeat([f"S{i:02d}" for i in range(40)], rows)
n = len(stock)
x0, x1 = rng.standard_normal(n), rng.standard_normal(n)
y = 0.5 * x0 + 0.3 * x1 + rng.standard_normal(n)
order = rng.permutation(n)
pl.DataFrame(
    {
        "t": np.arange(n, dtype=float),
        "stock": stock[order],
        "x0": x0[order],
        "x1": x1[order],
        "y": y[order],
    }
).write_parquet("stocks.parquet")
lf = pl.scan_parquet("stocks.parquet")
small = [f"S{i:02d}" for i in range(4, 40)]


def run(query, group):
    """Fit a regression per group; count the findings by code, and the small stocks' predictions."""
    spec = po.spec.ewridge(
        "ridge",
        targets=["y"],
        features=["x0", "x1"],
        group=group,
        half_life=float("inf"),
        min_weight=10.0,
    )
    bank = po.ModelBank([spec])
    out = pl.concat(bank.fit_predict_batches(query, chunk_size=5_000)).unnest("ridge")
    for row in bank.check().group_by("severity", "code", maintain_order=True).len().iter_rows():
        print(*row)
    predicted = out.filter(pl.col("stock").is_in(small))["pred_y"].is_not_null().mean()
    print(f"the small stocks' rows predicted: {predicted:.0%}")


run(lf, "stock")
print("-- the small stocks pooled into one group --")
pooled = lf.with_columns(
    pool=pl.when(pl.col("stock").is_in(small)).then(pl.lit("rest")).otherwise("stock")
)
run(pooled, "pool")

print("-- half-lives that do not suit the stream --")
specs = [
    po.spec.ewridge(
        "short",
        targets=["y"],
        features=["x0", "x1"],
        clock="t",
        gap_cap=5.0,
        half_life=2.0,
        min_weight=10.0,
    ),
    po.spec.ewridge(
        "long", targets=["y"], features=["x0", "x1"], clock="t", gap_cap=5.0, half_life=100_000.0
    ),
]
bank = po.ModelBank(specs)
bank.fit(lf)
shown = bank.check().select(
    "severity", "code", "spec", pl.col("value").round_sig_figs(3), "threshold"
)
for row in shown.iter_rows():
    print(*row)
```

```text
warning few_rows 36
warning low_support 36
warning withheld 36
info group_sizes 1
the small stocks' rows predicted: 0%
-- the small stocks pooled into one group --
the small stocks' rows predicted: 86%
-- half-lives that do not suit the stream --
warning below_min_weight short 3.41 10.0
warning withheld short None None
warning never_settled long 0.13 0.5
```

The first lines count the findings by code. Grouped by stock, each of the
36 small stocks learned from 2 rows for 3 coefficients. `check()` finds
that on all 36, with the coefficients set by the ridge rather than the
data, and the predictions withheld. One `group_sizes` finding notes the
sizes 2,500 times apart. Not one of the small stocks' rows is predicted.
Pooled into one group, the small stocks share one fit, `check()` finds
nothing, and 86% of their rows are predicted, the rest before the pool met
`min_weight`. A
half-life of 2 rows settles at a weight of 3.41, below the `min_weight` of
10 it was given, so its predictions stay null. A half-life of 100,000 rows
over a stream of 20,072 has settled to 0.13 of its ceiling, so its fit
still rests on a window that has not filled.

**What to do.** Pool groups too small to fit, with `pl.when` in the query
before the bank as the recipe does, or fit them with fewer features. Pick
a half-life that gathers `min_weight` at the stream's row rate, or lower
`min_weight`. When a stream is shorter than its half-life, feed more
history, or shorten the half-life.

## Sources

Each claim above that is not measured here rests on one of these.

| source | for |
|---|---|
| Allison, P. D. (2001). *Missing Data*. Sage. | listwise deletion is unbiased for regression slopes when the gap does not depend on the target |
| Åström, K. J. and Wittenmark, B. (1995). *Adaptive Control*, 2nd ed. Addison-Wesley. | estimator windup under exponential forgetting |
| Belsley, D. A., Kuh, E. and Welsch, R. E. (1980). *Regression Diagnostics*. Wiley. | condition indexes and variance-decomposition proportions |
| Chan, T. F., Golub, G. H. and LeVeque, R. J. (1983). Algorithms for computing the sample variance: analysis and recommendations. *The American Statistician* 37(3), 242–247. | the precision a sum-form variance loses at a level |
| Dickey, D. A. and Fuller, W. A. (1979). Distribution of the estimators for autoregressive time series with a unit root. *Journal of the American Statistical Association* 74(366), 427–431. | the unit-root statistic `random_walk` reads |
| Goldberg, D. (1991). What every computer scientist should know about floating-point arithmetic. *ACM Computing Surveys* 23(1), 5–48. | the digits a double keeps |
| Granger, C. W. J. and Newbold, P. (1974). Spurious regressions in econometrics. *Journal of Econometrics* 2(2), 111–120. | a regression of one random walk on another |
| Haykin, S. (2002). *Adaptive Filter Theory*, 4th ed. Prentice Hall. | the step-size bound and eigenvalue spread of least mean squares |
| Huber, P. J. (1964). Robust estimation of a location parameter. *Annals of Mathematical Statistics* 35(1), 73–101. | the influence of a large residual, and Huber's loss |
| Kaufman, S., Rosset, S. and Perlich, C. (2012). Leakage in data mining: formulation, detection, and avoidance. *ACM Transactions on Knowledge Discovery from Data* 6(4). | leakage |
| Kulhavý, R. and Zarrop, M. B. (1993). On a general concept of forgetting. *International Journal of Control* 58(4), 905–924. | stabilized forgetting |
| Little, R. J. A. (1992). Regression with missing X's: a review. *Journal of the American Statistical Association* 87(420), 1227–1237. | what complete-case analysis does to a regression |
| López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. Chapter 7. | purging and the embargo |
| Phillips, P. C. B. (1986). Understanding spurious regressions in econometrics. *Journal of Econometrics* 33(3), 311–340. | why the t-statistic grows with the rows |
| Rubin, D. B. (1976). Inference and missing data. *Biometrika* 63(3), 581–592. | missing completely at random, at random, and not at random |
| Vovk, V., Gammerman, A. and Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer. | conformal prediction |
| PLAN task 228 (2026-10-08) | listwise deletion against a pairwise Gram, a running-mean fill and online EM, on gaps of every kind |
