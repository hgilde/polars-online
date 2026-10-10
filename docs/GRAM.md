# Wide fits from Grams

A least-squares fit reads its rows only through a few running sums: the
weight of the rows, the columns' means and co-moments, and each target's
mean, variance and cross-moments with the columns. Together they are the
fit's *Gram*. `ewridge`, `lasso` and `ew_cov` keep one in their state, and
`bank.gram()` returns it as numpy arrays. A Gram's size depends on its
number of columns alone, so one pass over a stream too large to hold can
leave a Gram for each week. Every fit after the pass reads those Grams and
never the rows. It can pool any set of weeks, fit a ridge with its t, walk
a lasso path, or count how often each column is chosen on halves of the
weeks. The module `po.gram` does each of these.

The README's [The running sums behind a fit](../README.md#the-running-sums-behind-a-fit)
introduces a Gram's arrays, and the reference of
[`ModelBank.gram`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.ModelBank.gram)
states each of them and the identities between them.

Here is a stream of 40 weeks with 200 features, of which six matter. One
pass leaves a Gram per week, and the fit pools all forty. The spec's
settings make each week's Gram plain least squares' sums, and
[Weekly blocks from one pass](#weekly-blocks-from-one-pass) explains each:

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
start = datetime(2025, 1, 6)  # a Monday
ts = pl.datetime_range(start, start + timedelta(minutes=20 * (n - 1)), "20m", eager=True)
pl.DataFrame({"ts": ts, **dict(zip(features, x.T)), "y": y}).write_parquet("weeks.parquet")

# One pass over the file leaves a Gram per week. The week is a column made in the query.
lf = pl.scan_parquet("weeks.parquet").with_columns(week=pl.col("ts").dt.truncate("1w"))
spec = po.spec.ewridge(
    "blocks", targets=["y"], features=features, group="week",
    half_life=float("inf"), solve_every=1e18, target_gaps="pairwise", gram_block_rows=256,
)
bank = po.ModelBank([spec])
bank.fit(lf)  # learns from every row, and keeps only the state
blocks = bank.gram("blocks")  # one dict per week, oldest first

pooled = po.gram.merge(blocks)  # the Gram of all 20,160 rows
coef = po.gram.solve(pooled)  # least squares: the intercept, then x0 .. x199
stats = po.gram.coef_stats(pooled, coef)  # its standard errors and t

columns = pooled["columns"]
print(f"{len(blocks)} weekly Grams of {len(columns)} columns, pooled over {pooled['weight_sum']:,.0f} rows")
print(f"{'column':>6} {'truth':>6} {'coef':>6} {'t':>6}")
truth = {"x0": 0.5, "x5": 0.4, "x10": 0.3, "x15": 0.2, "x20": 0.1, "x25": b25.mean()}
for name, b in truth.items():
    i = columns.index(name)
    print(f"{name:>6} {b:>6.3f} {coef[i]:>6.3f} {stats['t'][i]:>6.1f}")
rest = np.abs([stats["t"][columns.index(f)] for f in features if f not in truth])
print(f"the other {len(rest)} columns: {(rest > 3).sum()} with |t| above 3, the largest {rest.max():.1f}")
```

```text
40 weekly Grams of 201 columns, pooled over 20,160 rows
column  truth   coef      t
    x0  0.500  0.489   50.0
    x5  0.400  0.389   39.6
   x10  0.300  0.297   30.5
   x15  0.200  0.196   20.3
   x20  0.100  0.100   10.4
   x25  0.186  0.196   20.3
the other 194 columns: 1 with |t| above 3, the largest 3.1
```

The pooled Gram holds all 20,160 rows. Its least-squares fit finds each
planted coefficient within about one standard error. Of the 194 columns
with no effect, one has a t past 3, at 3.1, where chance alone expects
about half a column. The coefficient of `x25` is drawn again each week
around 0.15, and its forty weekly values average 0.186, the truth its
line shows. The pooled fit sees only that average. Its t of 20.3 reads as
firm as the t of `x15`, whose coefficient never moves.
[Per-block coefficients](#per-block-coefficients) reads the weekly values
from the same forty Grams.

**Every recipe on this page is a complete program.** It generates its
data with the answer planted, saves it to parquet and reads it back with
`pl.scan_parquet`, as a stream too large to hold would be read. Most
recipes start from the opening example's stream and its pass, and repeat
the code that makes them, so each runs alone. The output is shown under
each recipe. `tests/test_gram_page.py` runs every recipe and holds its
output to what is shown, so none goes stale. A line that reports a wall
time is this machine's, so the test checks its words and not its
numbers. Warnings that a recipe raises go to the standard error stream
and are not shown.

These are the words of the library that the page uses:

| word | what it means here |
|---|---|
| spec, bank | a spec is one model's description: the model, the columns it reads, and how it treats time. `po.ModelBank` fits a set of specs together over the same rows |
| `ewridge`, `lasso`, `ew_cov` | the three models whose state holds a Gram: a ridge regression, a lasso, and the moments of columns with no target. [The README's model table](../README.md#models) lists every model |
| Gram | a fit's running sums, as `bank.gram(spec)` returns them: one dict per group, whose keys the next table lists |
| column | one axis of a Gram: `"intercept"` first when the spec fits one, then the spec's features |
| block | the rows of one value of a spec's `group` column, and the Gram the bank keeps for them. Every recipe here makes a week a block |
| pool | merge the Grams of disjoint rows into the Gram of all of them, with `po.gram.merge` |
| clock, `gap_cap` | the column a spec's `half_life`, `gap_cap` and `solve_every` are measured on, as `clock="ts"`, or the row count for a spec without one. `gap_cap` is the longest step between two rows that the bank ages a Gram by, and a spec with a clock needs one |
| `half_life` | each row's weight halves every `half_life` of the clock. `float("inf")` weighs every row alike, so a Gram holds plain least squares' sums |
| `weight_sum`, `n_kish` | the weight behind a Gram, and `weight_sum² / Σw²`, the number of equally weighted rows that weight is worth. Both are the row count when every row weighs 1 |
| chunk | the rows a bank reads from a query at one time |
| thread pool | the threads a bank runs its work on: `POLARS_ONLINE_MAX_THREADS` of them, one per core by default |
| load average | the number of processes running or waiting to run, averaged over a minute. The recipes ran on 14 cores shared with other users' work |
| active set, knot | on a lasso path, the columns whose coefficient is not 0, and a penalty at which that set changes |
| half | 20 of the 40 weeks, drawn at random, and their pooled Gram |

The keys of a Gram that the recipes read, in the names the formulas on
this page give them (the README's
[The running sums behind a fit](../README.md#the-running-sums-behind-a-fit)
lists every key):

| key | what it holds |
|---|---|
| `group` | the block's key, as text: here the week's Monday |
| `columns`, `targets` | the names along each axis |
| `weight_sum`, `n_kish` | the weight behind the Gram, `W`, and its Kish size |
| `means` | the columns' weighted means, `m` |
| `comoments` | the columns' centred co-moments, `C`, a `k × k` matrix |
| `cross_moments` | per target, the uncentred moments `E[z y]`, with `z` the columns of a row of data: one row of the array per target. At the intercept, a constant 1, the moment is the target's mean `ȳ` |
| `cross_centred` | per target, the same moments centred at the columns' means and the target's own mean: `c`, the vector a fit is solved from |
| `means_by_target` | per target, the columns' means over the rows the target was present on |
| `target_weights`, `target_means`, `target_vars`, `target_n_kish` | per target, its weight, its mean, its variance `V` and its Kish size |

What you can do with a set of Grams, and the calls that do it:

| to | call | section |
|---|---|---|
| collect a Gram per week in one pass | `po.spec.ewridge(..., group="week")`, `bank.gram()` | [Weekly blocks from one pass](#weekly-blocks-from-one-pass) |
| fit on some of the columns | `po.gram.subset`, or `features=` on any fit | [Every least-squares fit is a function of the Gram](#every-least-squares-fit-is-a-function-of-the-gram) |
| pool any weeks, and score a fit on others | `po.gram.merge` | [Pooling blocks](#pooling-blocks) |
| pool weeks and leave their levels out | `po.gram.merge(..., within=True)` | [Pooling within blocks](#pooling-within-blocks) |
| fit a ridge and read its t | `po.gram.solve`, `po.gram.coef_stats` | [A ridge and its t](#a-ridge-and-its-t) |
| know whether a fit is determined | `po.gram.condition`, `po.gram.vif`, `po.gram.correlation` | [Is the fit determined?](#is-the-fit-determined) |
| choose columns | `po.gram.lars_path`, `po.gram.lasso_path` | [Choosing columns on the lasso path](#choosing-columns-on-the-lasso-path) |
| choose columns that hold across the weeks | `po.gram.lars_paths`, `po.gram.solve_subsets(..., path=...)` | [Stability selection over halves of the weeks](#stability-selection-over-halves-of-the-weeks) |
| know how often a selection chooses a column with no effect of one sign | `po.gram.solve_subsets(..., within=True, target_signs=...)` | [A null for the selection, by sign flips](#a-null-for-the-selection-by-sign-flips) |
| see whether a coefficient holds from week to week | `po.gram.solve_subsets` | [Per-block coefficients](#per-block-coefficients) |
| read Grams back from files | `po.gram.from_row`, `po.ModelBank.load`, or numpy alone | [Reading Grams from saved files](#reading-grams-from-saved-files) |
| halve or quarter a Gram's bytes | `bank.gram(dtype="float32", layout="packed")` | [A compact Gram](#a-compact-gram) |
| collect one wide Gram on several cores | `gram_threads` | [One wide Gram on several threads](#one-wide-gram-on-several-threads) |

## How a Gram stands in for the rows

A fit can read a Gram in place of its rows for two reasons. Every
least-squares fit on the Gram's columns is a function of the Gram, and
the Gram's moments are centred, so they keep their digits however far a
column sits from zero. Some questions need each row, and no Gram answers
them.

### Every least-squares fit is a function of the Gram

**Every fit to a weighted sum of squared residuals reads its rows only
through the Gram.** Take weights `w_i` that sum to `W`, the column means
`m` and the centred co-moments `C = Σ w (x − m)(x − m)' / W`. For a target
`y`, take its mean `ȳ`, its variance `V` and its centred cross-moments
`c = Σ w (x − m)(y − ȳ) / W`. For any intercept `b_0` and slopes `b`, the
mean squared residual is

```text
Σ w_i (y_i − b_0 − x_i'b)² / W  =  V − 2 b'c + b'C b + (ȳ − b_0 − m'b)²
```

Nothing on the right reads a row. Least squares minimizes it, a ridge
adds `λ b'b` to it, and a lasso adds `λ Σ |b_j|`. Best subset, stepwise
selection and an information criterion over sets of columns compare its
minimum from one set to another. Each of them is therefore computed from
the Gram. Under the Gaussian linear model this is Fisher's (1922)
sufficiency: the Gram is a sufficient statistic for the coefficients and
the noise variance.

**To fit on some of the columns, take their sub-block of the Gram.**
`po.gram.subset(g, ["intercept", "x0", "x5"])` selects those columns'
rows and columns of `C` and their entries of each target's moments. A fit
on the subset is the fit a pass over those columns alone would give on
the same rows. Keep `"intercept"` among the columns, since a subset
without it is fitted through the origin.
[Is the fit determined?](#is-the-fit-determined) fits a subset. Every fit
in `po.gram` also takes `features=`, which fits the same columns with the
intercept kept, and returns a coefficient for every column of the full
Gram, 0 for each column left out.

**The same identity scores a fit on rows it was not fitted on.** Put a
held-out Gram's moments into the right side, and it gives the fit's mean
squared error over the held-out rows, without reading them. The target's
moments and the columns' must cover the same rows, as they do when the
target is present on every row of the Gram.
[Pooling blocks](#pooling-blocks) scores a fit on the last ten weeks this
way.

**The moments are centred, so they keep their digits when a column's
level, its typical size, is far from zero.** A raw second moment of a
column at a level `L` holds `L²` plus its variance. A variance read back
from it as `E[x²] − m²` loses the digits that `L²` takes up. A double
carries about sixteen, and for a unit-variance column at 1e8 the
difference keeps none of them. Each row's update adds the row's deviation from the running
means to the co-moments, as Welford's (1962) recursion does, with weights
as West (1979) gives them. Each target's cross-moments are kept
centred at its own means, as `cross_centred`, for the same reason (Chan,
Golub and LeVeque 1983).

**`weight_sum` counts weight, and `n_kish` counts rows.** With
`half_life=float("inf")` and every row's weight 1, both are the row
count. Under a decay, `n_kish = W² / Σ w²` is the number of equally
weighted rows the moments are worth (Kish 1965), and a standard error
divides by it.

### What a Gram cannot answer

A question that needs each row's residual, or a column the Gram does not
hold, needs the rows, or a model that ran over them:

| question | why a Gram cannot answer it | where it is answered |
|---|---|---|
| a fit whose loss is not a sum of squares: robust, quantile or logistic | its minimum depends on each residual, so no sum of moments holds it | the robust and quantile regressions `huber` and `quantile`, and the online logistic regression `ftrl`, run over the stream |
| the residuals' tails, autocorrelation and spread, and which row moved the fit | each reads a row's own residual | the diagnostics a spec switches on, which judge each row's residual as the bank runs ([Is the model working?](DIAGNOSTICS.md)) |
| standard errors robust to heteroskedasticity or autocorrelation | their middle term sums each row's `x x' e²`, or products of residuals rows apart | the spec's `emit_robust_se` on the stream ([Can its t be trusted?](DIAGNOSTICS.md#can-its-t-be-trusted)), or the spread of [per-block coefficients](#per-block-coefficients) |
| a column the Gram does not hold: a square, a product or a lag | its moments with the other columns are sums the pass never took | make the column with Polars' `with_columns` in the query before the bank, and run the pass again |
| a lagged co-moment across two pooled parts | a part holds only the pairs of its own rows | `ew_cov(lags=...)` over the whole stream. `po.gram.merge` returns `lags` as `None` |

## Weekly blocks from one pass

A block is the rows of one value of a spec's `group` column, and the bank
keeps a Gram for each. So a column that names the week gives a Gram per
week from one pass. Make the column with Polars' `dt.truncate("1w")`,
chained in the query before the bank, and name it as the spec's `group`.
The spec's other settings make each Gram plain least squares' sums,
collected as fast as the bank can:

| setting | value in the recipes | what it does |
|---|---|---|
| `group` | `"week"`, the column made by `pl.col("ts").dt.truncate("1w")` | a state, and so a Gram, for each week |
| `half_life` | `float("inf")` | every row weighs the same, so a block's Gram is least squares' and blocks pool as they are |
| `solve_every` | `1e18` rows, past the end of any stream (a duration, such as `"1000w"`, on a spec with a time clock) | under an infinite half-life the model solves on every row unless told otherwise. The fits are made from the Grams, so the pass solves nothing |
| `target_gaps` | `"pairwise"` | one Gram per block whatever rows each target is missing on, so every block has the same targets and they pool |
| `gram_block_rows` | `256` | adds 256 rows to the matrix in one product, 5.1 to 6.6 times as fast as one row at a time from 256 to 2,000 features ([`ewridge`](../README.md#ewridge--ew-ridge-on-sufficient-statistics)) |
| `gram_threads` | left out, so 1 | runs that product on several threads ([One wide Gram on several threads](#one-wide-gram-on-several-threads)) |
| `group_close` | left out | frees each week's Gram once the week has finished, and with `closed_groups=` on `fit` writes it to a file ([Reading Grams from saved files](#reading-grams-from-saved-files)) |

**A block's Gram has `k²` entries, whatever its rows.** A week of 504
rows at 201 columns has a co-moment matrix of 323,208 bytes, and so has a
week of a million rows. At 10,000 columns a block is 800 MB in float64.
The workload these tools were built for, 9.5 million rows read as 38
weekly blocks of 10,000 columns, is 30 GB of Grams ([PLAN](PLAN.md)
tasks 225 and 229). The bank holds the Gram of every group it has seen,
so a pass needs about `8k²` bytes for each block it holds. With
`group_close`, the bank writes each finished week out and holds one week
at a time. [A compact Gram](#a-compact-gram) halves or quarters what
`gram()` hands back.

**Choose the block so that every question a fit will ask is a set of
whole blocks.** Weeks let a fit pool any run of weeks, hold out the last
ten, or set one week against another. Days would allow finer choices,
with seven times as many Grams to hold.

**Under the default `target_gaps="own_rows"`, a block can hold several
Grams.** A target that is missing on rows where another is present gets a
Gram of its own, one per pattern of gaps, each naming its `targets`, and
`po.gram.merge` refuses parts whose targets differ. When every target is
present on the same rows, or the rows a target misses do not depend on
the features' values, give `target_gaps="pairwise"`, which keeps one Gram
per block. Otherwise keep
the default, and pool each set of targets' Grams across the weeks, those
with the same `targets`. The README's
[`ewridge`](../README.md#ewridge--ew-ridge-on-sufficient-statistics)
section says what each mode fits.

The recipe makes the stream, runs the pass, and checks the pooled Gram's
fit against least squares on the rows, read into memory for the check:

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
start = datetime(2025, 1, 6)  # a Monday
ts = pl.datetime_range(start, start + timedelta(minutes=20 * (n - 1)), "20m", eager=True)
pl.DataFrame({"ts": ts, **dict(zip(features, x.T)), "y": y}).write_parquet("weeks.parquet")

# One pass over the file leaves a Gram per week. The week is a column made in the query.
lf = pl.scan_parquet("weeks.parquet").with_columns(week=pl.col("ts").dt.truncate("1w"))
spec = po.spec.ewridge(
    "blocks", targets=["y"], features=features, group="week",
    half_life=float("inf"), solve_every=1e18, target_gaps="pairwise", gram_block_rows=256,
)
bank = po.ModelBank([spec])
bank.fit(lf)  # learns from every row, and keeps only the state
blocks = bank.gram("blocks")  # one dict per week, oldest first

first = blocks[0]
print(f"{len(blocks)} blocks, the first for the week of {first['group'][:10]}")
print(f"weight_sum {first['weight_sum']:.0f} and n_kish {first['n_kish']:.0f}: its rows, weighed alike")
print(f"comoments {first['comoments'].shape}: {first['comoments'].nbytes:,} bytes a block")

# The pooled Gram's least squares, against least squares on the rows read into memory.
rows = pl.read_parquet("weeks.parquet")
design = np.column_stack([np.ones(n), rows.select(features).to_numpy()])
by_rows = np.linalg.lstsq(design, rows["y"].to_numpy(), rcond=None)[0]
by_gram = po.gram.solve(po.gram.merge(blocks))
print(f"the two fits agree within 1e-12: {np.abs(by_gram - by_rows).max() < 1e-12}")
```

```text
40 blocks, the first for the week of 2025-01-06
weight_sum 504 and n_kish 504: its rows, weighed alike
comoments (201, 201): 323,208 bytes a block
the two fits agree within 1e-12: True
```

Each block holds its week's 504 rows, weighed alike, so its `weight_sum`
and `n_kish` are both the row count. Its co-moment matrix is 201 by 201,
323,208 bytes. Pooled, the forty blocks give the least-squares fit that
numpy's `lstsq` computes from all 20,160 rows, to within 1e-12 in every
coefficient.

## Pooling blocks

`po.gram.merge` pools the Grams of disjoint rows into the Gram of all of
them, so one pass serves every choice of weeks a fit makes. Blocks run
under `half_life=float("inf")` pool as they are. Under a decay, each
block is first aged to the last row of the stream it joins
([Aging a block before it is pooled](#aging-a-block-before-it-is-pooled)).

**A pooled Gram is formed from the parts' moments and the gaps between
their means, and never from a difference of two large sums.** This is
Chan, Golub and LeVeque's (1983) pairwise update. For parts `a` and `b`,
with weights `W_a` and `W_b`, means `m_a` and `m_b`, co-moments `C_a` and
`C_b`, and the gap `Δ = m_b − m_a` between their means, the union has

```text
W = W_a + W_b
m = m_a + (W_b / W) · Δ
C = (W_a · C_a + W_b · C_b) / W + (W_a · W_b / W²) · Δ Δ'
```

Each target's moments pool the same way over the rows it was present on,
by its `target_weights`. Every term is a weighted average of the parts'
moments, or a product of the gap. None is a difference of two large
cumulative sums, which is where pooling raw sums loses its digits. The
forty weeks of [Weekly blocks from one pass](#weekly-blocks-from-one-pass)
pooled to least squares on their rows within 1e-12.

**The parts' rows must be disjoint.** A row in two parts counts twice,
and `po.gram.merge` cannot tell. `po.gram.solve_subsets` refuses a subset
that names one Gram twice.

**A pooled Gram scores a fit on rows it was not fitted on.** The identity
in [Every least-squares fit is a function of the Gram](#every-least-squares-fit-is-a-function-of-the-gram)
turns a held-out Gram into the fit's mean squared error over the held-out
rows. The recipe fits weeks 1 to 30, scores the fit on weeks 31 to 40,
and checks the score against those weeks' rows:

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
start = datetime(2025, 1, 6)  # a Monday
ts = pl.datetime_range(start, start + timedelta(minutes=20 * (n - 1)), "20m", eager=True)
pl.DataFrame({"ts": ts, **dict(zip(features, x.T)), "y": y}).write_parquet("weeks.parquet")

# One pass over the file leaves a Gram per week. The week is a column made in the query.
lf = pl.scan_parquet("weeks.parquet").with_columns(week=pl.col("ts").dt.truncate("1w"))
spec = po.spec.ewridge(
    "blocks", targets=["y"], features=features, group="week",
    half_life=float("inf"), solve_every=1e18, target_gaps="pairwise", gram_block_rows=256,
)
bank = po.ModelBank([spec])
bank.fit(lf)  # learns from every row, and keeps only the state
blocks = bank.gram("blocks")  # one dict per week, oldest first


def mean_squared_error(g, coef):
    """A fit's mean squared error over the rows behind g, from g's arrays alone."""
    b0, b = coef[0], coef[1:]
    gap = g["cross_moments"][0][0] - b0 - g["means"][1:] @ b  # the target's mean less the fit's
    spread = g["target_vars"][0] - 2 * b @ g["cross_centred"][0][1:] + b @ g["comoments"][1:, 1:] @ b
    return spread + gap**2


train = po.gram.merge(blocks[:30])  # weeks 1 to 30
held_out = po.gram.merge(blocks[30:])  # weeks 31 to 40
coef = po.gram.solve(train)
print(f"weeks 1-30: {train['weight_sum']:,.0f} rows, error {mean_squared_error(train, coef):.4f} in sample")
print(f"weeks 31-40: {held_out['weight_sum']:,.0f} rows, error {mean_squared_error(held_out, coef):.4f} held out")

# The held-out error from the rows themselves, read into memory to check.
rows = pl.read_parquet("weeks.parquet").filter(pl.col("ts") >= start + timedelta(weeks=30))
pred = coef[0] + rows.select(features).to_numpy() @ coef[1:]
print(f"from the rows of weeks 31-40: {np.mean((rows['y'].to_numpy() - pred) ** 2):.4f}")
```

```text
weeks 1-30: 15,120 rows, error 1.0912 in sample
weeks 31-40: 5,040 rows, error 1.1337 held out
from the rows of weeks 31-40: 1.1337
```

The held-out Gram gives the error the held-out rows give, 1.1337, to
every digit printed. It is above the in-sample error, 1.0912, as a fit's
error on rows it never saw is. In `mean_squared_error`, `gap` is the
target's mean less the fit's, and `spread` the error about it. The spread
reads `cross_centred`, the form the fit itself is solved from, so the
score keeps its digits at any level.

### Aging a block before it is pooled

**Under a finite half-life, age each block to the last row of the pooled
stream before pooling it.** A block's weights are relative to its own
last row. A week that ended a month before the stream did is therefore
weighted as if it ended with the stream: too heavily, by exactly the
decay between its last row and the stream's. Multiply its `weight_sum`
and `target_weights` by `0.5 ** (dt / half_life)`, with `dt` the clock
from its last row to the stream's last row, each step capped at `gap_cap`
as the bank caps it. Leave the means and co-moments as they are, since
they are weighted means. Leave `n_kish` too, since it does not change
when every weight is scaled by one factor. `bank.groups()` gives each
group's `last_clock`, in the order `gram()` gives the groups. Blocks run
under `half_life=float("inf")` pool as they are.

The recipe runs two specs over the same stream in one pass: a Gram per
week at a half-life of four weeks, and one Gram of the whole stream at
the same half-life. Both decay on the clock `ts`, a timestamp, so their
`half_life`, `gap_cap` and `solve_every` are durations. Steps of 20
minutes are all below the `gap_cap` of an hour, so `dt` is the plain
difference of the clocks:

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
start = datetime(2025, 1, 6)  # a Monday
ts = pl.datetime_range(start, start + timedelta(minutes=20 * (n - 1)), "20m", eager=True)
pl.DataFrame({"ts": ts, **dict(zip(features, x.T)), "y": y}).write_parquet("weeks.parquet")

# One pass, two specs: a decayed Gram per week, and one decayed Gram of the whole stream.
lf = pl.scan_parquet("weeks.parquet").with_columns(week=pl.col("ts").dt.truncate("1w"))
decayed = dict(
    targets=["y"], features=features, clock="ts", gap_cap="1h", half_life="4w", solve_every="1000w"
)
bank = po.ModelBank(
    [po.spec.ewridge("by_week", group="week", **decayed), po.spec.ewridge("whole", **decayed)]
)
bank.fit(lf)
blocks, whole = bank.gram("by_week"), bank.gram("whole")[0]

# Age each week to the stream's last row: its weights are relative to its own last row.
last = bank.groups("by_week")["last_clock"]  # each week's last row, in the order gram() gives them
aged = []
for block, end in zip(blocks, last):
    decay = 0.5 ** ((last.max() - end) / timedelta(weeks=4))  # 0.5 ** (dt / half_life)
    aged.append(
        dict(block, weight_sum=block["weight_sum"] * decay, target_weights=block["target_weights"] * decay)
    )

for how, parts in [("as they are", blocks), ("aged", aged)]:
    gap = np.abs(po.gram.solve(po.gram.merge(parts)) - po.gram.solve(whole)).max()
    shown = "below 1e-12" if gap < 1e-12 else f"{gap:.3f}"
    print(f"40 weeks pooled {how}: the largest gap from the whole stream's coefficients is {shown}")
```

```text
40 weeks pooled as they are: the largest gap from the whole stream's coefficients is 0.085
40 weeks pooled aged: the largest gap from the whole stream's coefficients is below 1e-12
```

Pooled as they are, the decayed weeks weigh the first week as much as the
last, and a coefficient lands 0.085 from the whole stream's. Aged before
they are pooled, they give the whole stream's own coefficients.

### Pooling within blocks

**`po.gram.merge(blocks, within=True)` pools each block's moments about
its own means, and leaves out the spread between the blocks' means.** The
union above adds `(W_a · W_b / W²) · Δ Δ'`, the co-moment of the gaps
between the parts' means. The within form drops that term, so the pooled
co-moments are the blocks' own, averaged by weight:

```text
W = Σ W_i        C = Σ W_i · C_i / W
T = Σ T_i        c = Σ T_i · c_i / T        V = Σ T_i · V_i / T
```

Each target's weights `T_i`, centred cross-moments `c_i` and variances
`V_i` pool over the rows it was present on. A week with no rows of a
target adds to `C` and not to that target's moments. Where the blocks'
means are equal, the two forms give the same numbers.

**The pooled Gram is that of the rows less their week's means, and a fit
on it has an intercept for each week.** A regression on the rows less
their weekly means gives the slopes that the raw rows give with a constant
column for each week (Frisch and Waugh 1933, Lovell 1963). That is the
within, or fixed-effects, estimator (Wooldridge 2010, chapter 10). It asks
only how the columns and the target move together inside each week, as
each week's own fit does, and the weekly levels drop out of it.

**So the fit has no intercept of its own, and loses a degree of freedom
for each week.** The pooled Gram's `means` are 0, beside the intercept's
1, and it counts its weeks in `within_parts` and, per target, in
`target_within_parts`. `po.gram.solve` and every other fit return `nan`
for the intercept, since each week had one and the pooled Gram holds none
of them. `po.gram.coef_stats` divides by `n − p − B`, with `p` the
features and `B` the weeks the target has rows in.
`po.gram.solve_subsets(..., within=True)` pools each subset the same way.
A union of several parts refuses a Gram pooled within, since its means
are 0 and the union would add the gap from 0 to the others' means.

**Pooled as one union, a chance correlation of the weekly means reads as
a correlation of the rows.** Take a feature whose level wanders from week
to week, such as an exponentially weighted mean at a half-life near a
week, and a target whose weekly level wanders apart from it. Their forty
weekly means correlate by chance, at about ±1/√40 = ±0.16 for independent
series. The union's co-moments hold that correlation, and its standard
errors divide by the 20,160 rows, as if each row had confirmed it. It
rests on forty weekly means.

**Pool within weeks to ask how a feature moves the target inside a week,
and keep the union to ask how their levels move together.** The second
question has forty observations, one a week, and
[Per-block coefficients](#per-block-coefficients) or the weekly means
answer it with forty.

The recipe makes `slow`, whose weekly level wanders, and `quick`, which
moves `y` inside each week. `y`'s weekly level wanders too, apart from
both:

```python
from datetime import datetime, timedelta

import numpy as np
import polars as pl

import polars_online as po

# 40 weeks, a row every 20 minutes. slow is an exponentially weighted mean of noise at a half-life
# of a week, so its weekly level wanders. y's weekly level wanders apart from it, drawn for each
# week. Only quick moves y inside a week, by 0.05.
rng = np.random.default_rng(7)
n_weeks, per_week = 40, 504
n = n_weeks * per_week
slow = pl.Series(rng.standard_normal(n)).ewm_mean(half_life=per_week).to_numpy()
slow = slow / slow.std()
quick = rng.standard_normal(n)
level = 0.3 * rng.standard_normal(n_weeks)  # y's level in each week
y = 0.05 * quick + level.repeat(per_week) + rng.standard_normal(n)
start = datetime(2025, 1, 6)  # a Monday
ts = pl.datetime_range(start, start + timedelta(minutes=20 * (n - 1)), "20m", eager=True)
pl.DataFrame({"ts": ts, "slow": slow, "quick": quick, "y": y}).write_parquet("levels.parquet")

# One pass over the file leaves a Gram per week.
lf = pl.scan_parquet("levels.parquet").with_columns(week=pl.col("ts").dt.truncate("1w"))
spec = po.spec.ewridge(
    "blocks", targets=["y"], features=["slow", "quick"], group="week",
    half_life=float("inf"), solve_every=1e18,
)
bank = po.ModelBank([spec])
bank.fit(lf)
blocks = bank.gram("blocks")

weekly = np.array([[g["means"][1], g["target_means"][0]] for g in blocks])  # each week's mean of slow and of y
print(f"the weekly means of slow and y correlate at {np.corrcoef(weekly.T)[0, 1]:.2f} over {n_weeks} weeks")
print(f"{'pooled':>12} {'slow':>7} {'t':>5} {'quick':>7} {'t':>5}")
for how, pooled in [("as a union", po.gram.merge(blocks)), ("within weeks", po.gram.merge(blocks, within=True))]:
    coef = po.gram.solve(pooled)  # within weeks, coef[0], the intercept, is nan
    t = po.gram.coef_stats(pooled, coef)["t"]
    print(f"{how:>12} {coef[1]:>7.3f} {t[1]:>5.1f} {coef[2]:>7.3f} {t[2]:>5.1f}")

each = po.gram.solve_subsets(blocks, [[i] for i in range(n_weeks)])  # each week alone
weekly_slow = np.array([fits[0]["coef"][1] for fits in each])
t_weeks = weekly_slow.mean() / (weekly_slow.std(ddof=1) / np.sqrt(n_weeks))  # Fama and MacBeth's t
print(f"each week alone: slow's mean coefficient {weekly_slow.mean():.3f}, its t from the weeks {t_weeks:.1f}")
```

```text
the weekly means of slow and y correlate at -0.10 over 40 weeks
      pooled    slow     t   quick     t
  as a union  -0.025  -3.4   0.047   6.4
within weeks  -0.004  -0.3   0.046   6.5
each week alone: slow's mean coefficient 0.006, its t from the weeks 0.2
```

The weekly means of `slow` and `y` correlate at −0.10, inside the ±0.16
that chance gives forty weeks. Pooled as a union, that correlation gives
`slow` a coefficient of −0.025 and a t of −3.4. Pooled within weeks,
`slow`'s t is −0.3. Each week alone agrees: the forty weekly coefficients
average 0.006, with a t of 0.2 from their spread. `quick` moves `y` inside
each week, and both forms find it, with a t of 6.4 and of 6.5.

## A ridge and its t

`po.gram.solve` fits a ridge on a Gram, for a grid of penalties in one
call, and `po.gram.coef_stats` gives any fit its residual variance,
standard errors and t. The ridge solves the same system the `ewridge`
model solves. The t is least squares', with least squares' assumptions.

**The ridge solves the centred system, so the intercept is not
penalized.** The slopes solve `(C + λI) b = c`, and the intercept is
`b_0 = ȳ − m'b` (Hoerl and Kennard 1970). With `standardize=True` the
system is solved in correlation form, so one `λ` means the same whatever
the columns' units. A grid of penalties rides one eigendecomposition:
with `C = Q diag(e) Q'`, each penalty's slopes are
`Q diag(1 / (e + λ)) Q' c`. To reproduce the coefficients a spec reports
in `bank.coef()`, pass its `ridge` and `standardize`. `solve` does not
reproduce a spec's `ridge_scale="sum"` or `coef_prior`, a ridge on a
decaying scale and a prior other than zero, as the README's
[`po.gram` table](../README.md#the-running-sums-behind-a-fit) says.

**`coef_stats` computes least squares' standard errors from the Gram:**

```text
resid_var = V − 2 b'c + b'C b
sigma2    = resid_var · n / (n − p)      n = target_n_kish; p, the columns with the intercept
se_j      = sqrt((C⁻¹)_jj · sigma2 / n)
t_j       = b_j / se_j
```

With `half_life=float("inf")` and every row's weight 1, `n` is the row
count, and these are the classical standard errors: the recipe gets the
ones numpy computes from the rows, to within 1e-9.

**The t carries least squares' assumptions, whatever fit it is given:**

| it assumes | so |
|---|---|
| independent errors of one variance | correlated or heteroskedastic errors leave the standard errors too small. Standard errors robust to both need each row's residual: `emit_robust_se` on the stream ([Can its t be trusted?](DIAGNOSTICS.md#can-its-t-be-trusted)), or the spread of [per-block coefficients](#per-block-coefficients) |
| one coefficient for all the pooled rows | a coefficient that changes from week to week reads as firm a t as a constant one: the 20.3 of `x25` beside the 20.3 of `x15` in the opening example |
| least squares' covariance, `sigma2 · C⁻¹ / n` | a ridge's standard errors are that covariance at the ridge's residual variance. They leave out the shrinkage and its bias, so a ridge's t compares its coefficients with one another, and is no test |
| `n` independent rows | under a decay the rows weigh unequally and are not independent, and `n_kish` stands in for `n`. Read the t as a comparison there too |
| an intercept that centring has absorbed | the intercept's standard error and t are `nan` |

With two weeks of rows for 201 columns, a ridge helps. The recipe fits a
grid of ridges on weeks 1 and 2, and scores each on the other 38 weeks
with their pooled Gram. Then it reads the t of the least-squares fit and
of the best ridge:

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
start = datetime(2025, 1, 6)  # a Monday
ts = pl.datetime_range(start, start + timedelta(minutes=20 * (n - 1)), "20m", eager=True)
pl.DataFrame({"ts": ts, **dict(zip(features, x.T)), "y": y}).write_parquet("weeks.parquet")

# One pass over the file leaves a Gram per week. The week is a column made in the query.
lf = pl.scan_parquet("weeks.parquet").with_columns(week=pl.col("ts").dt.truncate("1w"))
spec = po.spec.ewridge(
    "blocks", targets=["y"], features=features, group="week",
    half_life=float("inf"), solve_every=1e18, target_gaps="pairwise", gram_block_rows=256,
)
bank = po.ModelBank([spec])
bank.fit(lf)  # learns from every row, and keeps only the state
blocks = bank.gram("blocks")  # one dict per week, oldest first


def mean_squared_error(g, coef):
    """A fit's mean squared error over the rows behind g, from g's arrays alone."""
    b0, b = coef[0], coef[1:]
    gap = g["cross_moments"][0][0] - b0 - g["means"][1:] @ b
    spread = g["target_vars"][0] - 2 * b @ g["cross_centred"][0][1:] + b @ g["comoments"][1:, 1:] @ b
    return spread + gap**2


train, held_out = po.gram.merge(blocks[:2]), po.gram.merge(blocks[2:])  # fit on two weeks, judge on 38
ridges = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0]
grid = po.gram.solve(train, ridge=ridges, standardize=True)  # a row per ridge, one eigendecomposition
errors = [mean_squared_error(held_out, coef) for coef in grid]
print("held-out error by ridge: " + ", ".join(f"{r} {e:.4f}" for r, e in zip(ridges, errors)))
best = int(np.argmin(errors))

# coef_stats on the least-squares fit and on the chosen ridge's, for the planted columns.
plain, ridged = po.gram.coef_stats(train, grid[0]), po.gram.coef_stats(train, grid[best])
print(f"{'column':>6} {'coef':>6} {'t':>5} {'ridge ' + str(ridges[best]):>12} {'t':>5}")
for name in ["x0", "x5", "x10", "x15", "x20", "x25"]:
    i = train["columns"].index(name)
    print(f"{name:>6} {grid[0][i]:>6.3f} {plain['t'][i]:>5.1f} {grid[best][i]:>12.3f} {ridged['t'][i]:>5.1f}")

# At ridge 0, coef_stats' standard errors are least squares' own, from the two weeks' rows.
rows = pl.read_parquet("weeks.parquet").head(2 * per_week)
design = np.column_stack([np.ones(rows.height), rows.select(features).to_numpy()])
resid = rows["y"].to_numpy() - design @ grid[0]
dof = rows.height - design.shape[1]
se = np.sqrt(np.diag(np.linalg.inv(design.T @ design)) * (resid @ resid) / dof)
print(f"coef_stats' standard errors equal the rows' within 1e-9: {np.allclose(plain['se'][1:], se[1:], rtol=1e-9)}")
```

```text
held-out error by ridge: 0.0 1.3521, 0.01 1.3433, 0.03 1.3285, 0.1 1.2959, 0.3 1.2706, 1.0 1.3050
column   coef     t    ridge 0.3     t
    x0  0.487  10.0        0.320   6.4
    x5  0.344   7.0        0.225   4.5
   x10  0.204   4.2        0.139   2.8
   x15  0.200   4.2        0.134   2.7
   x20  0.138   3.0        0.078   1.6
   x25  0.085   1.8        0.057   1.2
coef_stats' standard errors equal the rows' within 1e-9: True
```

A ridge of 0.3, in correlation units, lowers the held-out error from
1.3521 to 1.2706. It shrinks every coefficient toward 0, that of `x0`
from 0.487 to 0.320, and each t falls with its coefficient, from 10.0 to
6.4 for `x0`.
The standard errors barely move: they are least squares', read at the
ridge's slightly larger residual variance. At ridge 0 they are the ones
numpy computes from the two weeks' rows.

## Is the fit determined?

When columns are nearly dependent, the rows fix only their combined
effect, and the split among them moves with the noise. The fit then
predicts well, and its coefficients mean nothing. `po.gram.vif` and
`po.gram.condition` read this from a Gram, and `po.gram.correlation`
gives the matrix behind them.

**A variance inflation factor says how far a column's variance is
inflated by the other columns.** `VIF_j = 1 / (1 − R²_j)`, where `R²_j`
is the R² of column `j` regressed on the rest (Marquardt 1970). It is the
diagonal of the inverse correlation matrix. A coefficient's standard
error is `√VIF_j` times what it would be if its column were uncorrelated
with the rest, so a VIF of 10 makes it 3.2 times as wide. Read a VIF past
1e10 as an exact dependency: rounding can leave it finite, as
[`vif`'s reference](https://hgilde.github.io/polars-online/gram.html#polars_online.gram.vif)
explains.

**Belsley's condition indexes say which columns are involved.**
`po.gram.condition` scales the design to unit column length, without
centring it, and takes its singular values `s`. Each index is
`s_max / s_j`. The variance-decomposition proportions give each column's
share of its coefficient's variance in each component. Belsley, Kuh and
Welsch (1980) read an index above 30, with two or more columns' shares
above 0.5 in its component, as a dependency among those columns. The
design is left uncentred so that a dependency on the intercept shows,
such as a column whose level is far above its spread.

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

spec = po.spec.ewridge(
    "fit", targets=["y"], features=features, half_life=float("inf"), solve_every=1e18
)
bank = po.ModelBank([spec])
bank.fit(pl.scan_parquet("near.parquet"))
g = bank.gram("fit")[0]

cond = po.gram.condition(g)  # Belsley's diagnostics, with the intercept among the columns
print("condition indexes: " + ", ".join(f"{v:.1f}" for v in cond["condition_indexes"]))
last = cond["proportions"][-1]  # each column's share of its variance in the last component
print("shares of the last: " + ", ".join(f"{c} {p:.2f}" for c, p in zip(cond["columns"], last)))
print("VIF: " + ", ".join(f"{c} {v:.0f}" for c, v in zip(features, po.gram.vif(g))))
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
and `x4` keep theirs. To read the coefficients, drop one of the three,
combine them into one column in the query before the bank, or increase
the ridge ([Data quality](DATA-QUALITY.md#collinear)).

## Choosing columns on the lasso path

The lasso adds `λ Σ_j |b_j|` to the squared loss (Tibshirani 1996). At a
large enough `λ` every slope is 0, and as `λ` falls, columns enter the
fit. `po.gram` follows that path two ways. `lars_path` walks it knot by
knot from its start, by least angle regression. `lasso_path` solves it at
the penalties given, by coordinate descent.

**Both solve the lasso on the Gram's correlation form.** With `R` the
features' correlation matrix and `d` their standardized cross-moments
with the target, centred at its own means, the lasso minimizes

```text
f(b) = ½ b'R b − d'b + λ Σ_j w_j |b_j|
```

with `w_j` a scale on column `j`'s penalty, 1 unless `penalty_weights=`
gives it. The coefficients come back in the columns' own units, with the
intercept recovered. The solution is piecewise linear in `λ`. A knot is a
`λ` at which the active set changes, as columns enter it, or leave it
when their coefficients reach 0 (Efron, Hastie, Johnstone and Tibshirani
2004).

**Use least angle regression for the start of a path, and coordinate
descent for a grid of penalties:**

| | `lars_path`, and `lars_paths` for many Grams | `lasso_path` |
|---|---|---|
| computes | each knot exactly, from the largest penalty down | the solution at each penalty given, each started from the one before |
| stops | after `max_steps` knots, or at the first knot with at least `max_active` columns active | at the last penalty |
| time, measured at a load average near 12 | 4 ms for the whole path, 201 knots, at 200 columns. Its reference gives 0.2 ms for 40 knots at 200 columns, and 3 ms at 2,000 | 3 to 8 ms for 25 penalties at 200 columns |
| takes | a weight `w_j` above 0 for each column | a weight of 0, which keeps a column in the fit at every penalty, and an elastic net (`l1_ratio` below 1), whose path is not piecewise linear (Zou and Hastie 2005) |
| use it for | the first columns to enter: a selection, and [stability selection](#stability-selection-over-halves-of-the-weeks) | a penalty chosen by held-out error, an elastic net, a column that must stay in |

**At a knot, LARS gives the lasso's coefficients exactly, and coordinate
descent reaches them to its tolerance.** `lasso_path`'s `tol` bounds the
largest move of a sweep. On a poorly conditioned Gram that says little
about the distance from the solution: at a condition number of 1e8, a
descent at `tol=1e-14` stopped with its intercept 0.48 from the knot's
(`lars_path`'s
reference). The descent follows Friedman, Hastie and Tibshirani (2010).

The recipe walks the path on weeks 1 to 4 to its eighth active column,
and checks coordinate descent at its knots. Then it chooses a penalty
from a grid of 25 by the error on the other 36 weeks:

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
start = datetime(2025, 1, 6)  # a Monday
ts = pl.datetime_range(start, start + timedelta(minutes=20 * (n - 1)), "20m", eager=True)
pl.DataFrame({"ts": ts, **dict(zip(features, x.T)), "y": y}).write_parquet("weeks.parquet")

# One pass over the file leaves a Gram per week. The week is a column made in the query.
lf = pl.scan_parquet("weeks.parquet").with_columns(week=pl.col("ts").dt.truncate("1w"))
spec = po.spec.ewridge(
    "blocks", targets=["y"], features=features, group="week",
    half_life=float("inf"), solve_every=1e18, target_gaps="pairwise", gram_block_rows=256,
)
bank = po.ModelBank([spec])
bank.fit(lf)  # learns from every row, and keeps only the state
blocks = bank.gram("blocks")  # one dict per week, oldest first


def mean_squared_error(g, coef):
    """A fit's mean squared error over the rows behind g, from g's arrays alone."""
    b0, b = coef[0], coef[1:]
    gap = g["cross_moments"][0][0] - b0 - g["means"][1:] @ b
    spread = g["target_vars"][0] - 2 * b @ g["cross_centred"][0][1:] + b @ g["comoments"][1:, 1:] @ b
    return spread + gap**2


train, held_out = po.gram.merge(blocks[:4]), po.gram.merge(blocks[4:])  # fit on four weeks, judge on 36

# LARS: the path from its start, knot by knot, to the first knot with at least eight columns active.
path = po.gram.lars_path(train, max_active=8)
active = []
for penalty, now in zip(path["penalties"], path["active"]):
    change = [f"+{c}" for c in now if c not in active] + [f"-{c}" for c in active if c not in now]
    print(f"knot at penalty {penalty:.4f}: {' '.join(change)}")
    active = now
cd = po.gram.lasso_path(train, path["penalties"], tol=1e-12)  # coordinate descent at the knots
print(f"coordinate descent at the knots, against LARS: within 1e-9: {np.abs(cd - path['coef']).max() < 1e-9}")

# Coordinate descent on 25 penalties, from the first knot's down a thousandfold, scored on the 36 weeks.
grid = np.geomspace(path["penalties"][0], path["penalties"][0] / 1_000, 25)
fits = po.gram.lasso_path(train, grid)
errors = [mean_squared_error(held_out, coef) for coef in fits]
best = int(np.argmin(errors))
print(f"the best penalty, {grid[best]:.4f}: {np.count_nonzero(fits[best][1:])} columns, held-out error {errors[best]:.4f}")
print(f"least squares on the same four weeks: held-out error {mean_squared_error(held_out, po.gram.solve(train)):.4f}")
```

```text
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
over all forty. Then come `x117`, a column with no effect, and
`x4`, which shares `x0`'s family factor. Coordinate descent at each knot
gives LARS's coefficients within 1e-9. On the grid, a penalty of 0.0369
keeps 18 columns and has a held-out error of 1.1328, where least squares
on the same four weeks has 1.2265.

## Stability selection over halves of the weeks

A lasso path on one sample chooses a set of columns, and another sample
can choose another set. Stability selection runs the choice on many
halves of the data and keeps the columns chosen on most of them
(Meinshausen and Bühlmann 2010). With a Gram per week, a half of the
weeks is one merge. So a hundred halves are a hundred merges and paths,
and no pass over the rows.

**Each half chooses every column that is active at a knot of its path, up
to the path's stop.** That is Meinshausen and Bühlmann's choice over a
range of penalties: the union of the active sets, which a path stopped at
`max_active` gives. For column `j`, `π_j` is the share of the halves that
chose it, and the stable set is every column with `π_j` at or above a
threshold `π_thr`.

**The threshold bounds the expected number of false selections.** Let
`q` be the average number of columns a half chooses, and `p` the number
of columns. The count `F` of columns with no effect in the stable set
then has an expected value of at most

```text
E[F] ≤ q² / ((2 π_thr − 1) p),      for π_thr above 1/2
```

Meinshausen and Bühlmann prove it when the columns with no effect are
chosen exchangeably, and the choice does no worse than guessing at
random. Shah and Samworth (2013) proved a bound of the same form with no
exchangeability, when each half is drawn beside its complement. Theirs
counts the columns whose chance of being chosen on a half is at most
`q / p`. The recipe draws its halves that way. At `q = 8`, `p = 200` and
`π_thr = 0.75`, the bound is 0.64.

**Draw the halves from whole weeks.** The theorem's halves are halves of
independent observations, and the rows of a stream are rarely
independent of their neighbours. A half of the weeks keeps each week's
rows on one side, so the weeks become the units being sampled, and the
assumption becomes that the weeks are alike. A run of weeks in another
regime from the rest breaks it.

**Stability selection bounds false selections, and says nothing about
whether a chosen coefficient holds.** `x25` has a coefficient that is a
different number each week around a nonzero average, and every half
chooses it. [Per-block coefficients](#per-block-coefficients) is the
reading for that.

**`po.gram.solve_subsets(..., path=...)` gives the same paths in one
call.** It reads the Grams once into memory the call holds, then merges
each half and walks its path there, on the bank's thread pool, without a
Python copy of a matrix per half. Its ridge form, on 38 Grams and 100 subsets at 2,000
columns, took 3.0 s where a loop of `merge`, `solve` and `coef_stats`
took 76 s ([PLAN](PLAN.md) task 227).

```python
import time
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
start = datetime(2025, 1, 6)  # a Monday
ts = pl.datetime_range(start, start + timedelta(minutes=20 * (n - 1)), "20m", eager=True)
pl.DataFrame({"ts": ts, **dict(zip(features, x.T)), "y": y}).write_parquet("weeks.parquet")

# One pass over the file leaves a Gram per week. The week is a column made in the query.
lf = pl.scan_parquet("weeks.parquet").with_columns(week=pl.col("ts").dt.truncate("1w"))
spec = po.spec.ewridge(
    "blocks", targets=["y"], features=features, group="week",
    half_life=float("inf"), solve_every=1e18, target_gaps="pairwise", gram_block_rows=256,
)
bank = po.ModelBank([spec])
bank.fit(lf)  # learns from every row, and keeps only the state
blocks = bank.gram("blocks")  # one dict per week, oldest first

halves = []  # 50 random splits of the 40 weeks: 100 halves of 20, each beside its complement
splitter = np.random.default_rng(1)
for _ in range(50):
    order = splitter.permutation(n_weeks)
    halves += [sorted(order[:20]), sorted(order[20:])]

t0 = time.perf_counter()
merged = [po.gram.merge([blocks[i] for i in half]) for half in halves]
paths = po.gram.lars_paths(merged, max_active=8)  # a list per half, a path per target
seconds = time.perf_counter() - t0

chosen, sizes = np.zeros(k), []  # how many halves chose each column, and how many each half chose
for (path,) in paths:
    ever = set().union(*path["active"])  # active at any knot, up to the stop
    sizes.append(len(ever))
    chosen[[features.index(c) for c in ever]] += 1
freq = dict(zip(features, chosen / len(halves)))

planted = ["x0", "x5", "x10", "x15", "x20", "x25"]
print("planted: " + ", ".join(f"{c} {freq[c]:.2f}" for c in planted))
rest = sorted((f for f in features if f not in planted), key=lambda f: -freq[f])
print("the most chosen of the rest: " + ", ".join(f"{c} {freq[c]:.2f}" for c in rest[:4]))
q, threshold = np.mean(sizes), 0.75
print(f"chosen on at least {threshold:.0%} of the halves: {' '.join(f for f in features if freq[f] >= threshold)}")
print(f"{q:.1f} columns a half, so at most {q**2 / ((2 * threshold - 1) * k):.2f} false selections expected")

t0 = time.perf_counter()
same = po.gram.solve_subsets(blocks, halves, path={"max_active": 8})  # the same paths, in one call
agree = all(a[0]["active"] == b[0]["active"] for a, b in zip(same, paths))
print(f"solve_subsets chooses the same: {agree}")
print(f"time: {seconds:.2f} s by merge and lars_paths, {time.perf_counter() - t0:.2f} s by solve_subsets")
```

```text
planted: x0 1.00, x5 1.00, x10 1.00, x15 1.00, x20 1.00, x25 1.00
the most chosen of the rest: x4 0.45, x11 0.25, x86 0.20, x14 0.17
chosen on at least 75% of the halves: x0 x5 x10 x15 x20 x25
8.0 columns a half, so at most 0.64 false selections expected
solve_subsets chooses the same: True
time: 0.19 s by merge and lars_paths, 0.01 s by solve_subsets
```

Every planted column is chosen on every half, and the stable set at 75%
is exactly those six. The most chosen of the rest is `x4`, on 45% of the
halves, a member of `x0`'s family that the lasso sometimes takes in its
place. Each half chose 8.0 columns, so over draws of the halves the
stable set holds at most 0.64 columns with no effect on average, and
this draw's holds none. `solve_subsets`
chooses the same columns on every half. It took 0.01 s, where the merges
and `lars_paths` took 0.19 s, at a load average near 14 on 14 cores.

### A null for the selection, by sign flips

**Negate each week's target at random and run the selection again, to see
what it chooses where no effect keeps its sign across the weeks.**
Stability selection's bound assumes that the columns with no effect are
chosen exchangeably, and no better than at random. A path that always
stops at eight columns, among columns that share families, need not meet
that. A null run measures the selection instead. Draw a sign for each
week, negate that week's target, and rerun every half. A column whose
effect keeps its sign across the weeks loses that in a null run. The run
keeps everything else: each week's columns, their correlations, and the
size of each week's effects. Over many draws, the largest selection
frequency in each null run gives a threshold. Only 5% of the null runs
choose any column on more halves than its 95th percentile. That is the
maximum statistic of a permutation test by sign flips (Winkler et al.
2014), with a week as the unit that is flipped.

**Pooled within weeks, a negated target changes only the cross-moments,
so a null run needs no new merge of the co-moments.** Negating a week's
target negates its centred cross-moments `c_i`. Its co-moments, its target
weight and its variance stay as they were. So a draw of signs `s_i` gives
`c = Σ T_i · s_i · c_i / T`, and nothing else in the pooled Gram moves.
`po.gram.solve_subsets(..., within=True, target_signs=signs)` takes the
signs as an array of weeks by draws, and returns one run of every subset
per draw. Pooled as a union, a negated week's target mean would also move
the gaps between the weeks' means, so `target_signs` is refused without
`within=True`. A hundred null runs of the hundred halves below, at 200
columns, took 0.08 s with the real run. A Python loop of `merge` and
`lars_paths` took about 0.08 s a draw, at a load average near 4 on 14
cores.

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
start = datetime(2025, 1, 6)  # a Monday
ts = pl.datetime_range(start, start + timedelta(minutes=20 * (n - 1)), "20m", eager=True)
pl.DataFrame({"ts": ts, **dict(zip(features, x.T)), "y": y}).write_parquet("weeks.parquet")

# One pass over the file leaves a Gram per week. The week is a column made in the query.
lf = pl.scan_parquet("weeks.parquet").with_columns(week=pl.col("ts").dt.truncate("1w"))
spec = po.spec.ewridge(
    "blocks", targets=["y"], features=features, group="week",
    half_life=float("inf"), solve_every=1e18, target_gaps="pairwise", gram_block_rows=256,
)
bank = po.ModelBank([spec])
bank.fit(lf)  # learns from every row, and keeps only the state
blocks = bank.gram("blocks")  # one dict per week, oldest first

halves = []  # 50 random splits of the 40 weeks: 100 halves of 20, each beside its complement
splitter = np.random.default_rng(1)
for _ in range(50):
    order = splitter.permutation(n_weeks)
    halves += [sorted(order[:20]), sorted(order[20:])]


def frequencies(run):
    """The share of the halves that chose each column: active at any knot of its path."""
    chosen = np.zeros(k)
    for (path,) in run:
        chosen[[features.index(c) for c in set().union(*path["active"])]] += 1
    return chosen / len(run)


signs = np.random.default_rng(2).choice([-1.0, 1.0], size=(n_weeks, 100))  # a sign per week, 100 draws
real = po.gram.solve_subsets(blocks, halves, path={"max_active": 8}, within=True)
null = po.gram.solve_subsets(blocks, halves, path={"max_active": 8}, within=True, target_signs=signs)

freq = dict(zip(features, frequencies(real)))
null_max = np.array([frequencies(run).max() for run in null])  # each null run's most chosen column
threshold = np.quantile(null_max, 0.95)
planted = ["x0", "x5", "x10", "x15", "x20", "x25"]
print("planted: " + ", ".join(f"{c} {freq[c]:.2f}" for c in planted))
rest = sorted((f for f in features if f not in planted), key=lambda f: -freq[f])
print("the most chosen of the rest: " + ", ".join(f"{c} {freq[c]:.2f}" for c in rest[:4]))
print(f"the most chosen column of a null run: {np.median(null_max):.2f} of the halves at the median draw, {threshold:.2f} at the 95th percentile")
print(f"chosen on more halves than {threshold:.2f}: {' '.join(f for f in features if freq[f] > threshold)}")
```

```text
planted: x0 1.00, x5 1.00, x10 1.00, x15 1.00, x20 1.00, x25 1.00
the most chosen of the rest: x4 0.49, x11 0.25, x86 0.20, x14 0.16
the most chosen column of a null run: 0.81 of the halves at the median draw, 0.95 at the 95th percentile
chosen on more halves than 0.95: x0 x5 x10 x15 x20 x25
```

With each week's sign drawn at random, the most chosen column of a null
run is still chosen on 0.81 of the halves at the median draw. Each half
chooses eight columns whatever the signs, and a column with a large
effect in some weeks keeps being chosen when its sign changes from week
to week. The threshold of 0.75 that
[Stability selection over halves of the weeks](#stability-selection-over-halves-of-the-weeks)
used passes such a column in 73 of the 100 null runs. The null runs'
95th percentile, 0.95, is the threshold this selection earns on these
weeks. The six planted columns pass it, and the most chosen of the rest,
`x4`, is at 0.49.

## Per-block coefficients

Fitted alone, each week's Gram gives that week's coefficients. A pooled t
assumes one coefficient for all the pooled rows, and the spread of the
weekly coefficients shows whether there is one.

**Fama and MacBeth's (1973) t reads the weekly coefficients as a
sample.** With `b_w` the coefficient of week `w` out of `B`, it is the
mean of the `b_w` over their standard error, `sd(b_w) / √B`. That
standard error grows with the coefficient's own variation from week to
week, which a pooled standard error leaves out, and with any correlation
of the errors within a week. It assumes the weekly coefficients are
independent of one another. When they drift slowly, take Newey and
West's (1987) long-run variance of their series in place of `sd(b_w)²`.

**Fit each week on the columns already chosen.** A week of 504 rows fits
201 columns poorly, so pass the chosen columns as `features=`.
`po.gram.solve_subsets(blocks, [[0], [1], ...])` fits each block alone in
one call, each with its own standard errors and t.

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
start = datetime(2025, 1, 6)  # a Monday
ts = pl.datetime_range(start, start + timedelta(minutes=20 * (n - 1)), "20m", eager=True)
pl.DataFrame({"ts": ts, **dict(zip(features, x.T)), "y": y}).write_parquet("weeks.parquet")

# One pass over the file leaves a Gram per week. The week is a column made in the query.
lf = pl.scan_parquet("weeks.parquet").with_columns(week=pl.col("ts").dt.truncate("1w"))
spec = po.spec.ewridge(
    "blocks", targets=["y"], features=features, group="week",
    half_life=float("inf"), solve_every=1e18, target_gaps="pairwise", gram_block_rows=256,
)
bank = po.ModelBank([spec])
bank.fit(lf)  # learns from every row, and keeps only the state
blocks = bank.gram("blocks")  # one dict per week, oldest first

chosen = ["x0", "x5", "x10", "x15", "x20", "x25"]
each = po.gram.solve_subsets(blocks, [[i] for i in range(n_weeks)], features=chosen)  # each week alone
pooled = po.gram.merge(blocks)
coef = po.gram.solve(pooled, features=chosen)
t = po.gram.coef_stats(pooled, coef, features=chosen)["t"]

columns = pooled["columns"]
print(f"{'column':>6} {'pooled':>7} {'t':>5} {'weekly mean':>12} {'sd':>6} {'t':>5} {'same sign':>10}")
for name in chosen:
    i = columns.index(name)
    weekly = np.array([fits[0]["coef"][i] for fits in each])
    sd = weekly.std(ddof=1)
    t_weeks = weekly.mean() / (sd / np.sqrt(n_weeks))  # Fama and MacBeth's t
    sign = np.mean(np.sign(weekly) == np.sign(coef[i]))
    print(f"{name:>6} {coef[i]:>7.3f} {t[i]:>5.1f} {weekly.mean():>12.3f} {sd:>6.3f} {t_weeks:>5.1f} {sign:>10.0%}")
```

```text
column  pooled     t  weekly mean     sd     t  same sign
    x0   0.496  65.8        0.494  0.049  63.7       100%
    x5   0.390  51.7        0.392  0.041  60.5       100%
   x10   0.307  40.9        0.308  0.040  48.9       100%
   x15   0.196  26.3        0.193  0.038  32.2       100%
   x20   0.092  12.4        0.096  0.051  11.9        98%
   x25   0.196  26.2        0.195  0.303   4.1        75%
```

The pooled fit here is on the six chosen columns, so its t are larger
than the opening example's, which fitted all 200. `x0` to `x20` hold:
their weekly means match the pooled coefficients, and
the t from their spread, 11.9 to 63.7, stands beside the pooled t, 12.4
to 65.8. `x25`'s pooled t is 26.2, as firm as `x15`'s 26.3. But its
weekly coefficients spread by 0.303, and its t from the weeks is 4.1. In
a quarter of the weeks its sign is the opposite of the pooled fit's.

## Reading Grams from saved files

A pass can leave its Grams in two files: the bank's state, which
`po.ModelBank.load` reads back, and the parquet file that `closed_groups=`
writes, a row per finished block. Reading either needs no rows and runs
no model. `po.gram` works on what they hold, and so does numpy alone.

**`group_close="monotone"` closes each week once the next begins, and
frees it.** With `closed_groups=` on `fit`, the bank writes each closed
week's Gram to that file as a row. The bank then holds one week at a
time. Its key must be
an integer or a text column, so make the week the Monday's day number,
with `dt.date().cast(pl.Int32)` in the query. The last week never closes,
since no row shows it has finished, so it stays in the state, where
`bank.gram()` reads it.
[One row per finished group](../README.md#one-row-per-finished-group) in
the README has the rules.

**A closed row holds the numbers `bank.gram()` would have returned for
that week, bit for bit, laid out flat.** Its `comoments` is the upper
triangle, row by row, and each per-target array is one list, the
targets' rows one after another. With one target, as here, the first
`k` entries are that target's. `po.gram.from_row` turns the row back
into the dict `gram()` returns. numpy reads it as it stands: lay
`comoments` into the upper triangle with `numpy.triu_indices`, mirror it,
and solve.

**A state can be exported as JSON for a program in another language,**
with `bank.save_json(path)` ([Reading a state without this library](../README.md#reading-a-state-without-this-library)).
The forty open weeks' state was 19 MB as `bank.save` wrote it, and 106 MB
as JSON.

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
start = datetime(2025, 1, 6)  # a Monday
ts = pl.datetime_range(start, start + timedelta(minutes=20 * (n - 1)), "20m", eager=True)
pl.DataFrame({"ts": ts, **dict(zip(features, x.T)), "y": y}).write_parquet("weeks.parquet")

# One pass that writes each finished week's Gram to a file, and saves the last, open week.
lf = pl.scan_parquet("weeks.parquet").with_columns(
    week=pl.col("ts").dt.truncate("1w").dt.date().cast(pl.Int32)  # the Monday, as days since 1970
)
spec = po.spec.ewridge(
    "blocks", targets=["y"], features=features, group="week", group_close="monotone",
    half_life=float("inf"), solve_every=1e18, target_gaps="pairwise", gram_block_rows=256,
)
bank = po.ModelBank([spec])
bank.fit(lf, closed_groups="blocks.parquet")  # a row per week, written once the next week begins
bank.save("blocks.state")  # the state holds the last week, which no later row closed

# Later, in a program that runs no model: the Grams from the two files.
closed = pl.read_parquet("blocks.parquet")
blocks = [po.gram.from_row(row) for row in closed.iter_rows(named=True)]
blocks += po.ModelBank.load("blocks.state").gram("blocks")
print(f"{closed.height} weeks from blocks.parquet and {len(blocks) - closed.height} from blocks.state")
coef = po.gram.solve(po.gram.merge(blocks))
print("x0, x5 and x10 from the 40 weeks: " + ", ".join(f"{coef[i]:.3f}" for i in (1, 6, 11)))

# numpy alone, on the first week's row: unpack the triangle, and solve the centred system.
row = closed.row(0, named=True)
m = len(row["columns"])  # the intercept and the 200 features
cov = np.zeros((m, m))
cov[np.triu_indices(m)] = row["comoments"]  # the upper triangle, row by row
cov += np.triu(cov, 1).T
slopes = np.linalg.solve(cov[1:, 1:], np.asarray(row["cross_centred"])[1:])
intercept = row["cross_moments"][0] - np.asarray(row["means_by_target"])[1:] @ slopes
same = np.allclose(np.r_[intercept, slopes], po.gram.solve(blocks[0]), rtol=0, atol=1e-10)
print(f"numpy alone and po.gram agree on the first week within 1e-10: {same}")
```

```text
39 weeks from blocks.parquet and 1 from blocks.state
x0, x5 and x10 from the 40 weeks: 0.489, 0.389, 0.297
numpy alone and po.gram agree on the first week within 1e-10: True
```

The file holds 39 weeks, and the state the fortieth. Pooled, they give
the opening example's coefficients. numpy's own solve of the first
week's row agrees with `po.gram.solve` within 1e-10. In the row,
`cross_moments` at the intercept is the target's mean, and
`means_by_target` the column means its cross-moments are centred at.

## A compact Gram

`comoments` is the one array of a Gram that grows as `k²`: 800 MB a block
in float64 at 10,000 columns. `bank.gram()` can return it in float32, or
as its upper triangle, each about half the bytes, and both together a
quarter. `po.gram` reads every form in float64. The state itself stays
float64 and whole, and the forms shape only what `gram()` returns.

| `bank.gram(...)` | what it loses | what it keeps |
|---|---|---|
| `dtype="float32"` | each entry is rounded to nearest, within a relative `2⁻²⁴`, about 6e-8, of its float64 value. An entry past float32's largest number, about 3.4e38, or nonzero below its smallest normal number, about 1.2e-38, is refused with a `ValueError` naming its column. A column spread 1e20 wide, or 1e-20, gets there, since its variance squares the spread | every other array, in float64: the means, from which `merge` forms its spread term, and each target's moments |
| `layout="packed"` | the lower triangle, which reads back as the upper one mirrored. A Gram updated a row at a time adds the same two products to `C[i][j]` and `C[j][i]` in either order, so its two triangles can differ in the last bit | every other entry, bit for bit. With `gram_block_rows`, no entry of the recipe's Grams differed, so packing lost nothing |

**A float32 Gram is enough when `κ` times 6e-8 is below the precision
the fit needs.** A system whose entries carry a relative error `δ` gives
a solution whose relative error is of the order of `κδ` (Higham 2002).
Here `κ` is the condition number of the matrix solved, its largest
eigenvalue over its smallest. For a Gram that matrix is the features'
correlation matrix, which a standardized solve works on:
`np.linalg.cond(po.gram.correlation(g)[1:, 1:])`. Belsley's indexes of
[Is the fit determined?](#is-the-fit-determined) are of the uncentred
design. The square of the largest is the condition number of that
design's second moments, so they flag the same trouble on another scale. Rounding moves a solve by about `κδ`
at most, and usually by far less, since the errors of many entries
offset one another.

**A packed Gram loses the last bits of a lower triangle that a row-by-row
update wrote.** Measured on the recipe's stream without
`gram_block_rows`, 73% of the off-diagonal entries differed between the
two triangles, and a solve from the packed form moved by 2e-15 of the
largest coefficient.

The recipe solves the forty weeks from each form, and then asks for a
float32 Gram of a column spread 1e20 wide:

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
start = datetime(2025, 1, 6)  # a Monday
ts = pl.datetime_range(start, start + timedelta(minutes=20 * (n - 1)), "20m", eager=True)
pl.DataFrame({"ts": ts, **dict(zip(features, x.T)), "y": y}).write_parquet("weeks.parquet")

# One pass over the file leaves a Gram per week. The week is a column made in the query.
lf = pl.scan_parquet("weeks.parquet").with_columns(week=pl.col("ts").dt.truncate("1w"))
spec = po.spec.ewridge(
    "blocks", targets=["y"], features=features, group="week",
    half_life=float("inf"), solve_every=1e18, target_gaps="pairwise", gram_block_rows=256,
)
bank = po.ModelBank([spec])
bank.fit(lf)  # learns from every row, and keeps only the state
blocks = bank.gram("blocks")  # one dict per week, oldest first

pooled = po.gram.merge(blocks)
reference = po.gram.solve(pooled)
r = po.gram.correlation(pooled)  # row and column 0, the constant intercept's, are nan
kappa = np.linalg.cond(r[1:, 1:])  # of the features' correlation matrix
print(f"condition number {kappa:.1f}; float32's rounding times it, {kappa * 2**-24:.1e}")
for dtype in ("float64", "float32"):
    for layout in ("full", "packed"):
        parts = bank.gram("blocks", dtype=dtype, layout=layout)
        coef = po.gram.solve(po.gram.merge(parts))  # merged and solved in float64
        moved = np.abs(coef - reference).max() / np.abs(reference).max()
        size = parts[0]["comoments"].nbytes
        print(f"{dtype} {layout:>6}: {size:>7,} bytes a week; the fit moved by {moved:.0e} of its largest coefficient")

# A column spread 1e20 wide has a variance that float32 cannot hold.
pl.DataFrame({"a": 1e20 * rng.standard_normal(1_000), "y": rng.standard_normal(1_000)}).write_parquet("huge.parquet")
huge = po.spec.ewridge("huge", targets=["y"], features=["a"], half_life=float("inf"), solve_every=1e18)
bank = po.ModelBank([huge])
bank.fit(pl.scan_parquet("huge.parquet"))
try:
    bank.gram("huge", dtype="float32")
except ValueError as e:
    print(e)
```

```text
condition number 7.8; float32's rounding times it, 4.7e-07
float64   full: 323,208 bytes a week; the fit moved by 0e+00 of its largest coefficient
float64 packed: 162,408 bytes a week; the fit moved by 0e+00 of its largest coefficient
float32   full: 161,604 bytes a week; the fit moved by 7e-09 of its largest coefficient
float32 packed:  81,204 bytes a week; the fit moved by 7e-09 of its largest coefficient
gram(dtype="float32"): the co-moment of 'a' with itself is 1.029e40, past float32's largest number, 3.4e38, where it reads inf: rescale 'a' upstream, or read the Gram in float64
```

The packed float64 form solves to the same bits as the full one. In
float32 the fit moves by 7e-9 of its largest coefficient, far inside the
4.7e-7 its condition number of 7.8 allows. Both together are a quarter of the
bytes, 81,204 a week. The column spread 1e20 wide has a variance of
1.029e40, past float32's range, and `gram()` refuses to round it to
`inf`.

## One wide Gram on several threads

Updating a Gram takes `O(k²)` time a row, and at a few thousand columns
that update is the whole pass. `gram_threads=n`, on `ewridge` and
`ew_cov`, runs it on up to `n` threads, and the Gram comes out the same
to the bit at every `n`.

**The bits do not move, because each entry is summed by one thread in
one order.** The update is cut into pieces fixed by `k` alone: diagonal
blocks of 256 columns, and tiles of 128 to 255 columns a side. Each piece
is one sequential call on one thread, whatever the number of threads
([PERFORMANCE §37](PERFORMANCE.md#37-one-wide-gram-on-several-threads-task-225-2026-10-08)).
A fit's factorization and solves are cut by shape the same way, and its
eigendecomposition runs at a fixed parallel degree of eight. So a
coefficient's bits do not follow the size of the pool either
([§38](PERFORMANCE.md#38-a-solves-bits-whatever-the-pool-task-231-2026-10-09)).

**`gram_threads` takes its threads from the pool the bank runs its
groups on.** `POLARS_ONLINE_MAX_THREADS` sizes that pool, one thread per
core by default. A bank whose groups already keep every core busy gains
nothing from the option, and loses nothing. Fourteen groups of 1,000
columns ran as fast with eight threads each as with one. Run instead on
a separate pool of eight threads for the option, they took 28% longer
(§37). A bank of weekly blocks fed a stream in time order often reads a
chunk that holds one week's rows, and so has one group to run. There
`gram_threads` gives that group the rest of the pool.

**It helps past 256 columns with `gram_block_rows`, and past 725
without.** Up to 256 columns, a block's product is one piece. Without a
block, a row's update is bound by memory, and below 725 columns it stays
on one thread. On eight threads at 2,000 columns, a block's product ran
6.1 times as fast, and a whole bank 3.9 times, since the rest of a row's
work, `O(k)`, stays on one thread (§37). Without a block, the gain was 2
to 3 times.

The recipe learns 5,120 rows of 2,000 columns on one thread and on
eight:

```python
import time

import numpy as np
import polars as pl

import polars_online as po

# 5,120 rows of 2,000 features in float32; y is the sum of the first ten, plus noise.
rng = np.random.default_rng(3)
n, k = 5_120, 2_000
x = rng.standard_normal((n, k), dtype=np.float32)
y = x[:, :10].sum(axis=1) + rng.standard_normal(n, dtype=np.float32)
features = [f"x{j}" for j in range(k)]
pl.DataFrame({**dict(zip(features, x.T)), "y": y}).write_parquet("wide.parquet")

grams, seconds = {}, {}
for threads in (1, 8):
    spec = po.spec.ewridge(
        "wide", targets=["y"], features=features, half_life=float("inf"), solve_every=1e18,
        gram_block_rows=256, gram_threads=threads,
    )
    bank = po.ModelBank([spec])
    t0 = time.perf_counter()
    bank.fit(pl.scan_parquet("wide.parquet"))
    seconds[threads] = time.perf_counter() - t0
    grams[threads] = bank.gram("wide")[0]["comoments"]

print(f"the Gram on 1 thread and on 8, bit for bit: {grams[1].tobytes() == grams[8].tobytes()}")
print(f"time to learn: {seconds[1]:.2f} s on 1 thread, {seconds[8]:.2f} s on 8, {seconds[1] / seconds[8]:.1f} times as fast")
```

```text
the Gram on 1 thread and on 8, bit for bit: True
time to learn: 0.73 s on 1 thread, 0.33 s on 8, 2.2 times as fast
```

The Gram is the same to the bit. Eight threads learned 2.2 times as
fast, on a machine whose other work kept its load average near 14 on 14
cores. The time includes reading the file and each row's `O(k)` work,
which stay on one thread.

## Sources

Each claim above that is not measured here rests on one of these.

| source | for |
|---|---|
| Belsley, D. A., Kuh, E. and Welsch, R. E. (1980). *Regression Diagnostics*. Wiley. | condition indexes and variance-decomposition proportions |
| Chan, T. F., Golub, G. H. and LeVeque, R. J. (1983). Algorithms for computing the sample variance: analysis and recommendations. *The American Statistician* 37(3), 242–247. | the pairwise update that pools two sets of moments, and the precision a raw sum loses at a level |
| Efron, B., Hastie, T., Johnstone, I. and Tibshirani, R. (2004). Least angle regression. *The Annals of Statistics* 32(2), 407–499. | least angle regression and its lasso modification, and the lasso path's knots |
| Fama, E. F. and MacBeth, J. D. (1973). Risk, return, and equilibrium: empirical tests. *Journal of Political Economy* 81(3), 607–636. | a coefficient's t from its estimates period by period |
| Fisher, R. A. (1922). On the mathematical foundations of theoretical statistics. *Philosophical Transactions of the Royal Society of London, Series A* 222, 309–368. | sufficiency |
| Friedman, J., Hastie, T. and Tibshirani, R. (2010). Regularization paths for generalized linear models via coordinate descent. *Journal of Statistical Software* 33(1), 1–22. | coordinate descent for the lasso and the elastic net, warm-started along a path |
| Frisch, R. and Waugh, F. V. (1933). Partial time regressions as compared with individual trends. *Econometrica* 1(4), 387–401. | a regression's slopes with a constant per block, from the rows less their block's means |
| Higham, N. J. (2002). *Accuracy and Stability of Numerical Algorithms*, 2nd ed. SIAM. | the error a perturbation of a linear system's entries carries into its solution |
| Hoerl, A. E. and Kennard, R. W. (1970). Ridge regression: biased estimation for nonorthogonal problems. *Technometrics* 12(1), 55–67. | the ridge |
| IEEE (2019). *IEEE Standard for Floating-Point Arithmetic*, IEEE Std 754-2019. | float32's rounding and range |
| Kish, L. (1965). *Survey Sampling*. Wiley. | the effective sample size of unequal weights |
| Lovell, M. C. (1963). Seasonal adjustment of economic time series and multiple regression analysis. *Journal of the American Statistical Association* 58(304), 993–1010. | the general form of that theorem: the slopes with any set of columns partialled out first |
| Marquardt, D. W. (1970). Generalized inverses, ridge regression, biased linear estimation, and nonlinear estimation. *Technometrics* 12(3), 591–612. | the variance inflation factor |
| Meinshausen, N. and Bühlmann, P. (2010). Stability selection. *Journal of the Royal Statistical Society, Series B* 72(4), 417–473. | stability selection and its bound on false selections |
| Newey, W. K. and West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica* 55(3), 703–708. | the variance of a series of weekly coefficients that drift |
| Shah, R. D. and Samworth, R. J. (2013). Variable selection with error control: another look at stability selection. *Journal of the Royal Statistical Society, Series B* 75(1), 55–80. | halves drawn beside their complements, and the bound without exchangeability |
| Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society, Series B* 58(1), 267–288. | the lasso |
| Welford, B. P. (1962). Note on a method for calculating corrected sums of squares and products. *Technometrics* 4(3), 419–420. | the centred update of each row |
| West, D. H. D. (1979). Updating mean and variance estimates: an improved method. *Communications of the ACM* 22(9), 532–535. | the same update with weights |
| Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data*, 2nd ed. MIT Press. | the within, or fixed-effects, estimator (chapter 10) |
| Winkler, A. M., Ridgway, G. R., Webster, M. A., Smith, S. M. and Nichols, T. E. (2014). Permutation inference for the general linear model. *NeuroImage* 92, 381–397. | sign flipping as a permutation test, flipping whole blocks, and the maximum statistic over many tests |
| Zou, H. and Hastie, T. (2005). Regularization and variable selection via the elastic net. *Journal of the Royal Statistical Society, Series B* 67(2), 301–320. | the elastic net |
| [PERFORMANCE §37 and §38](PERFORMANCE.md#37-one-wide-gram-on-several-threads-task-225-2026-10-08), [PLAN](PLAN.md) tasks 225 to 229 and 231 | `gram_threads`, a solve's bits in any pool, `lars_path`, `solve_subsets` and the compact forms, as measured when they were built |
