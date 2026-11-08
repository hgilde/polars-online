<!-- kind: how-to -->
# Screen features against a target

This how-to is for anyone with many features and a few targets
who wants to know, in one pass over a stream, which features move with
which target. It ranks every (feature, target) pair with `marginal`,
corrects the ranking for rows that resemble their neighbors, and finds
the relations a correlation cannot see.

## Prerequisites

These steps need the library and the columns to screen:

- polars-online installed, and numpy to make the example's data.
- A stream whose columns hold the
  [features](../GLOSSARY.md#the-model-bank-and-what-it-reads) and one or
  more [targets](../GLOSSARY.md#the-model-bank-and-what-it-reads). A
  column may not be both. Thousands of features are fine: each pair is
  its own two-column covariance, so a row takes O(p·T) time for `p`
  features and `T` targets.
- A [half-life](../GLOSSARY.md#time), or `half_life=float("inf")` for
  every row to weigh the same. [Time and
  decay](../explanation/time-and-decay.md) says how a model forgets along
  its clock.

## Rank the pairs

Use `t_stat`, the correlation's t statistic, to rank the pairs, and never
as a p-value, for two reasons. The rows are neither independent nor
Gaussian. And the top of a ranking is the best of many chances, which
noise alone pushes past a threshold set for one test.

1. **Build a `marginal` spec with the targets and every feature to
   screen.**
2. **Make a [model bank](../GLOSSARY.md#the-model-bank-and-what-it-reads)
   of the spec with `po.ModelBank([screen])`.**
3. **Fit it with `bank.fit`.** Each row writes only `weight_sum`, so
   `fit` learns the stream and keeps only the state, where the pairs live.
4. **Read the pairs with `bank.marginal(name)`, and sort them by
   `|t_stat|`.** The table has one row per (feature, target) pair, with
   its moments, `corr`, `beta`, `t_stat` and `n_kish`, the effective number
   of rows behind them. [Its
   docstring](https://hgilde.github.io/polars-online/polars_online.html#polars_online.ModelBank.marginal)
   defines each column.

This code makes 200 features, of which three move the target, screens
every pair, and prints the five with the largest `|t_stat|`:

```python
import numpy as np
import polars as pl
import polars_online as po

# 200 features, of which three move the target, and one (f150) moves it in a V.
rng = np.random.default_rng(2)
n, p = 4_000, 200
x = rng.standard_normal((n, p))
y = x[:, [7, 42, 99]] @ np.array([0.5, -0.4, 0.3]) + 0.6 * np.abs(x[:, 150]) + rng.standard_normal(n)
features = [f"f{j}" for j in range(p)]
pl.DataFrame(x, schema=features).with_columns(y=y).write_parquet("screen.parquet")

# Screen every pair, and rank them by |t_stat|.
screen = po.spec.marginal("screen", targets=["y"], features=features, half_life=float("inf"))   # 1. every pair
bank = po.ModelBank([screen])                                                # 2. the bank
bank.fit(pl.scan_parquet("screen.parquet"))                                  # 3. learn, and keep only the state
table = bank.marginal("screen").sort(pl.col("t_stat").abs(), descending=True)   # 4. rank the pairs
for r in table.head(5).iter_rows(named=True):
    print(f"{r['feature']}: corr {r['corr']:+.2f}, t_stat {r['t_stat']:+.1f}")
```

```text
f7: corr +0.40, t_stat +27.8
f42: corr -0.30, t_stat -20.2
f99: corr +0.24, t_stat +15.4
f198: corr -0.05, t_stat -3.1
f81: corr -0.05, t_stat -2.9
```

The three planted features lead, in the order of their coefficients. The
best of the other 197 trails at a `t_stat` of −3.1, a value noise alone
gives with that many chances. The V-shaped `f150` is not among the five, since its correlation
with the target is near zero: [Find the relations a correlation cannot
see](#find-the-relations-a-correlation-cannot-see) finds it.

## Correct the count for rows that resemble their neighbors

Use this when both a feature and the target are smooth, each row close
to the one before it. `t_stat` is built on `n_kish`, the right count of rows
for unequal weights, which says nothing about rows that resemble their
neighbors. So on a smooth stream it claims evidence that is not there.
Over 1,000 pairs of *independent* AR(1) series with `φ = 0.9` and `0.8`,
3,000 rows each, `|t_stat|` passed 1.96 on 45% of them. Corrected under
`"geometric"` at the lags 1, 2, 3, 5, 8 and 13, `|t_stat_serial|` passed
1.96 on 5.4% of them, the 5% the threshold stands for.

1. **Give `lags`, a list of lags in rows that increases.** Each pair then
   keeps its moments at each lag.
2. **Give `serial_rule="geometric"`.** The table gains `n_serial`, a count
   that allows for the resemblance, after Bartlett (1935):
   `n_serial = n_kish / (1 + 2 Σ_l ρ_x(l)·ρ_y(l))`, and `t_stat_serial`,
   the statistic against it. `"geometric"` fits each series'
   autocorrelations as `ρ(l) = φˡ`, so the lags need not be dense, and
   reports the fitted `phi_x` and `phi_y`. The
   [docstring](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.marginal)
   gives the other two rules, `"truncated"` and `"bartlett"`, for dense
   lags.
3. **Rank by `|t_stat_serial|` in place of `|t_stat|`.**

This code makes 500 AR(1) features with `φ = 0.8` and an AR(1) target with
`φ = 0.9`, all independent, so no feature moves the target. It counts the
pairs where each statistic passes 1.96:

```python
import numpy as np
import polars as pl
import polars_online as po

# 500 smooth features and a smooth target, all independent: no pair is related.
rng = np.random.default_rng(5)
n, p = 3_000, 500


# An AR(1) series: each row phi times the row before, plus noise.
def ar1(phi, shape):
    e = rng.standard_normal(shape)
    out = np.empty(shape)
    out[0] = e[0]
    for i in range(1, shape[0]):
        out[i] = phi * out[i - 1] + e[i]
    return out


# 500 features with phi 0.8 and a target with phi 0.9, saved.
columns = [f"f{j}" for j in range(p)]
pl.DataFrame(ar1(0.8, (n, p)), schema=columns).with_columns(y=ar1(0.9, (n,))).write_parquet("smooth.parquet")

# Screen every pair with the serial count.
spec = po.spec.marginal(
    "smooth", targets=["y"], features=columns, half_life=float("inf"),
    lags=[1, 2, 3, 5, 10],               # 1. the moments at five lags
    serial_rule="geometric",             # 2. n_serial and t_stat_serial
)
bank = po.ModelBank([spec])
bank.fit(pl.scan_parquet("smooth.parquet"))
table = bank.marginal("smooth")
# 3. The share of these unrelated pairs each statistic calls significant at 1.96.
print(f"|t_stat| past 1.96: {(table['t_stat'].abs() > 1.96).mean():.0%}")
print(f"|t_stat_serial| past 1.96: {(table['t_stat_serial'].abs() > 1.96).mean():.0%}")
print(f"n_kish {table['n_kish'][0]:.0f}, n_serial about {table['n_serial'].median():.0f}")
```

```text
|t_stat| past 1.96: 44%
|t_stat_serial| past 1.96: 5%
n_kish 3000, n_serial about 482
```

`t_stat` calls 44% of these unrelated pairs significant. `t_stat_serial`
calls about the 5% a threshold of 1.96 stands for, because it counts the
3,000 smooth rows as about 480 independent ones.

## Find the relations a correlation cannot see

Use this when a feature may act on the target through a threshold, a V or
a saturation, with `corr` at zero.

1. **Give `bins`, such as `bins=8`.** Each pair gains the target's weight,
   mean and variance inside each of the feature's bins, its response
   curve, in `bin_edges`, `bin_n`, `bin_mean_y` and `bin_var_y`. The edges
   are learned from the first `bin_warm_rows` learned rows (1,000 by
   default) and never move. Or give `bin_edges` in place of `bins` to
   set the edges yourself.
2. **Read `split_gain` against `corr²`.** `split_gain` is a regression
   stump's R²: the share of the target's variance the best single cut of
   the curve removes. Its excess over `corr²` is the nonlinear surplus.
   `split_at` is where the cut falls.
3. **Rank by that surplus, and read `split_gain_t` as a ranking too.** As
   a p-value it would overstate the evidence, because the cut was chosen
   as the best of `bins − 1`.

Binning keeps `O(bins)` of state per pair, and takes one search of the edges
per feature per row, so ten thousand columns bin in the pass that gives
their `corr`. This code bins the 200 features of `screen.parquet`, from
[Rank the pairs](#rank-the-pairs), and prints the three with the largest
surplus:

```python
binned = po.spec.marginal("binned", targets=["y"], features=features, half_life=float("inf"),
                          bins=8)                                  # 1. a response curve per pair
bank = po.ModelBank([binned])
bank.fit(pl.scan_parquet("screen.parquet"))
table = bank.marginal("binned").with_columns(surplus=pl.col("split_gain") - pl.col("corr") ** 2)   # 2. each pair's surplus over corr²
for r in table.sort("surplus", descending=True).head(3).iter_rows(named=True):              # 3. rank by the surplus
    print(f"{r['feature']}: corr {r['corr']:+.2f}, corr² {r['corr'] ** 2:.3f}, split_gain {r['split_gain']:.3f}")
```

```text
f150: corr +0.02, corr² 0.000, split_gain 0.021
f181: corr +0.00, corr² 0.000, split_gain 0.002
f115: corr +0.01, corr² 0.000, split_gain 0.002
```

The V-shaped `f150` stands out by ten times the next, with a correlation
of 0.02.

## Keep one spike from drawing a shape

Use this when the target has rare wild values and you read the bins'
means. A bin's mean moves by a row's size times its share of the bin. So
one wild row in a thin bin bends the response curve there, and a screen
that reads binned means of a residual, or of its square, sees a shape that
is not in the data.

1. **Give `bin_stat="huber"`.** Each bin's location is then Huber's
   M-estimate of location, learned in one pass, as
   [`huber`](../MODELS.md#huber--robust-regression) weighs a regression's
   rows. A row's weight is cut back by how far it lies from
   the bin's location before it, at `bin_huber_delta` (default 1.345)
   times the target's robust scale.
2. **Leave `bin_refit_rows` at its default.** Once that many rows are
   learned, the first rows are refit to convergence once, because, where
   spikes are a share of the first rows, one pass can be captured by
   them. [The streams one pass leaves
   captured](../explanation/robust-scale.md#the-streams-one-pass-leaves-captured)
   explains why.
3. **Read `bin_mean_y` as before:** under `"huber"` it is the location,
   and `bin_n` still counts the rows' weight.

This code puts one row of 1,000 into a bin of twenty rows near 0:

```python
import numpy as np
import polars as pl
import polars_online as po

# Twenty rows near 0 in the middle bin, among rows whose target is -5 or +5, then a spike there.
rng = np.random.default_rng(3)
i = np.arange(420)
x = np.where(i % 21 == 0, 0.0, np.where(i % 2 == 0, -0.9, 0.9))
y = np.where(x == 0.0, 0.0, 5.0 * np.sign(x)) + 0.01 * rng.standard_normal(420)
pl.DataFrame({"x": np.append(x, 0.0), "y": np.append(y, 1000.0)}).write_parquet("thin.parquet")

# The same bins under each statistic; print the middle bin.
for stat in ["mean", "huber"]:
    spec = po.spec.marginal("m", targets=["y"], features=["x"], half_life=float("inf"),
                            bin_edges=[[-0.5, 0.5]], bin_stat=stat)
    bank = po.ModelBank([spec])
    bank.fit(pl.scan_parquet("thin.parquet"))
    r = bank.marginal("m").row(0, named=True)
    print(f"{stat}: middle bin {r['bin_mean_y'][1]:.2f} over {r['bin_n'][1]:.0f} rows")
```

```text
mean: middle bin 47.62 over 21 rows
huber: middle bin 0.51 over 21 rows
```

The mean takes the spike's twenty-first share, 47.6. The location moves by
the cut over twenty, 0.51, and the count still takes the row. These 421
rows are fewer than the refit's default, `max(500, 30·bins)`, which is
500 here. So the refit has not run, and 0.51 is what the single pass
learned.

## Check that it worked

**Plant a feature you know moves the target, and check that it leads the
ranking.** The recipes plant theirs. On your own stream, add a column you
build from the target with noise, and read where it ranks.

| what you see | what it means | what to do |
|---|---|---|
| `corr`, `beta` and `t_stat` null for every pair | the target has not reached `min_weight`, 3 by default: two rows give a correlation of ±1 whatever the data | feed more rows |
| many unrelated features with `t_stat` beyond ±2 | the rows resemble their neighbors, so `n_kish` overstates the evidence | give `lags` and `serial_rule`, and rank by `t_stat_serial` |
| `n_serial` and `t_stat_serial` null for a pair | under `"truncated"` or `"bartlett"`, the two series' autocorrelations have opposite signs; under `"geometric"`, fewer than two kept lags are positive on a side | read `t_stat` for that pair, or try another rule |
| the bin columns empty and `split_gain` null | the edges are not fixed yet: the first `bin_warm_rows` rows are held to learn them | feed more rows, or give `bin_edges` |
| a `ValueError` naming `bins` and `window_size` | each of a [hard window](../GLOSSARY.md#fitting-a-model)'s snapshots would hold the whole histogram | bin the features in a second spec without `window_size` |

Every parameter is in `marginal`'s
[docstring](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.marginal),
and the model's reference entry is
[`marginal`](../MODELS.md#marginal--every-pairs-moments-kept-in-the-state).
