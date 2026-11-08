<!-- kind: how-to -->
# Align series that tick at their own times

This how-to is for anyone with series observed at instants of their own,
such as the trades of several stocks, who wants to fit a correlation, a
covariance or a regression across them. It puts the series on one grid
with `po.stream.refresh_time`, then fits the models on the grid's
changes. [The Epps effect, and the two ways
out](../REGIMES.md#6-the-epps-effect-and-the-two-ways-out) says why a fine
common grid fails and what the refresh grid recovers.

## Prerequisites

These steps need the library and the ticks:

- polars-online installed, and numpy to make the example's data.
- The ticks in long form, one row per tick: a column naming the series,
  a [clock](../GLOSSARY.md#time) and the observed value. A wide frame
  becomes long with Polars' `unpivot`, its `index` the clock column, as
  `unpivot(index="t", variable_name="symbol", value_name="px")` does for a
  clock `t`.
- The ticks in clock order, within each group if you give `group`.
  [Put rows in clock order](put-rows-in-clock-order.md) shows how.

## Put the series on one grid

The grid has a point at each *refresh time*: the first instant by which
**every** series has ticked at least once since the previous point.
Nothing is interpolated.

1. **Call `po.stream.refresh_time` with the ticks, naming the `series`
   column, every series in `names`, the `clock` and the `value`.** `names`
   sets the order of the output columns, and a tick of a series not in it
   is refused by name. Give `group`, such as a trading day, for a grid of
   each group's own.
2. **Read the grid: one row per refresh time.** `time_refresh` is the clock
   of the tick that completed the point, and `<s>_value` each series' last
   observed value there. `n_obs_<s>` counts that series' ticks since the
   previous point, of which the grid kept the last.
3. **Read `retained_fraction` before trusting a correlation,** the share
   of the ticks since the previous refresh time that the grid kept.

This code gives three series eight ticks on a clock `t`, and prints the
grid they make:

```python
import polars as pl
import polars_online as po

ticks = pl.DataFrame({
    "symbol": ["AAA", "BBB", "AAA", "CCC", "BBB", "AAA", "AAA", "CCC"],  # which series ticked
    "t": [0.4, 0.9, 1.3, 1.6, 2.2, 2.5, 2.8, 3.1],                      # when, in clock order
    "px": [100.0, 20.0, 100.2, 50.0, 20.1, 100.1, 100.4, 49.9],         # what it observed
})
grid = po.stream.refresh_time(ticks, series="symbol", names=["AAA", "BBB", "CCC"],   # 1.
                              clock="t", value="px")
print(grid.columns)                                                                  # 2.
for row in grid.rows():                                                              # 2. and 3.: one row per point
    print(row)
```

```text
['time_refresh', 'AAA_value', 'BBB_value', 'CCC_value', 'n_obs_AAA', 'n_obs_BBB', 'n_obs_CCC', 'retained_fraction']
(1.6, 100.2, 20.0, 50.0, 2, 1, 1, 0.75)
(3.1, 100.4, 20.1, 49.9, 2, 1, 1, 0.75)
```

The first refresh time is 1.6, when CCC ticks for the first time. AAA's
value there is 100.2, from its tick at 1.3, and its tick at 0.4 is
dropped. After 1.6, BBB ticks at 2.2 and AAA twice, so CCC's tick at 3.1
completes the second point. The output looks synchronous and is not: each
value is up to one of its own series' intervals between ticks old.

## Fit models on the grid

A model reads the grid as it reads any stream, with the grid's clock as
its own.

1. **Compute each series' change between points with Polars' `diff()`,**
   in a `select` on the grid.
2. **Give each spec the grid's clock, `time_refresh`,** with a `gap_cap`,
   the largest clock step the model decays by. The recipe's refresh times
   fall about two clock units apart, and it gives 10.0.
3. **Chain `online.fit_predict(specs)` after the grid.** Given a query,
   `refresh_time` returns one, so the grid and the models run in one pass.
   The first point has no change, so the models skip it.

This code plants a correlation: three prices share one move and add one of
their own, as large, so their changes correlate at 0.5. Each is observed
at times of its own, about once per clock unit. `ew_cov` reads the
correlations on the grid, and `ewridge` regresses one series' change on
the other two's, whose true coefficients are 1/3 each. It then reads the
same ticks on a grid every 0.1 clock units, each series' last price
carried forward:

```python
import numpy as np

# Three prices that share one move, each observed at times of its own.
rng = np.random.default_rng(1)
steps = 200_000                                          # a hidden price path, 0.01 clock units a step
shared = np.cumsum(rng.normal(0.0, 0.1, steps))          # the move the three series share
# One frame per series, then one stream in clock order.
names = ["AAA", "BBB", "CCC"]
frames = []
for s in names:
    price = shared + np.cumsum(rng.normal(0.0, 0.1, steps))    # plus a move of its own, as large
    t = np.cumsum(rng.exponential(1.0, 2_100))
    t = t[t < 2_000]                                     # about one tick per clock unit, at times of its own
    frames.append(pl.DataFrame({"symbol": s, "t": t, "px": price[(t * 100).astype(int)]}))
pl.concat(frames).sort("t").write_parquet("own_times.parquet")   # one stream, in clock order

# The refresh grid and the models on it, in one query.
grid = po.stream.refresh_time(pl.scan_parquet("own_times.parquet"), series="symbol", names=names,
                              clock="t", value="px")    # a query, since it was given one
returns = grid.select("time_refresh", *(pl.col(f"{s}_value").diff().alias(s) for s in names))  # 1. each series' change
specs = [
    po.spec.ew_cov("comove", features=names, stats=["corr"],
                   clock="time_refresh", gap_cap=10.0, half_life=float("inf")),          # 2. the grid's clock
    po.spec.ewridge("fit", targets=["AAA"], features=["BBB", "CCC"],
                    clock="time_refresh", gap_cap=10.0, half_life=float("inf")),
]
out = returns.online.fit_predict(specs).collect()         # 3. the grid and the models in one pass
# The correlations and the coefficients on the last point.
last = out.row(-1, named=True)
print(f"{out.height} refresh times from {pl.read_parquet('own_times.parquet').height} ticks")
print("on the refresh grid: corr " + ", ".join(f"{last['comove'][f'corr_{a}_{b}']:.2f}" for a, b in [("AAA", "BBB"), ("AAA", "CCC"), ("BBB", "CCC")])
      + f"; AAA on BBB {last['fit']['coef'][1]:.2f}, on CCC {last['fit']['coef'][2]:.2f}")

# The same ticks on a grid every 0.1 clock units, for comparison.
ticks_by_series = pl.read_parquet("own_times.parquet").partition_by("symbol", as_dict=True)
fine = pl.DataFrame({"t": np.arange(0.0, 2_000.0, 0.1)})
for s in names:                                          # each series' last price at each point of the fine grid
    fine = fine.join_asof(ticks_by_series[(s,)].select("t", pl.col("px").alias(s)), on="t")
moves = fine.select(pl.col(names).diff()).drop_nulls()
print(f"on a grid every 0.1: corr {moves.corr()['AAA'][1]:.2f}, {moves.corr()['AAA'][2]:.2f}, {moves.corr()['BBB'][2]:.2f}")
```

```text
1064 refresh times from 5835 ticks
on the refresh grid: corr 0.34, 0.32, 0.36; AAA on BBB 0.27, on CCC 0.23
on a grid every 0.1: corr 0.01, 0.02, 0.02
```

The fine grid finds almost no correlation: the Epps effect. The refresh
grid finds about two thirds of the planted 0.5. What it misses comes from
each value's age at a refresh time, which is up to one of its own series'
intervals between ticks.

## Check that it worked

**Read `retained_fraction` and each `n_obs_<s>`.** The grid runs at the
pace of the slowest series, so a fast series loses most of its ticks. The
series holding the grid up, whose tick completes each point, keeps an
`n_obs` near 1.

| what you see | what it means | what to do |
|---|---|---|
| `retained_fraction` low, and one series' `n_obs` large | one series is much slower than the others, and the grid drops most of the fast ones' ticks | pass `pairs=True` for an independent two-series grid per pair, which keeps far more ticks. The grid is then one long frame with the columns `group` (under `group`), `pair` (the two series' names joined by a vertical bar, the one listed first in `names` as `a`), `time_refresh`, `a_value`, `b_value`, `n_obs_a`, `n_obs_b` and `retained_fraction`: filter it on `pair` and take the changes of `a_value` and `b_value` before fitting |
| an error naming a row whose clock steps back | the ticks are not in clock order within a group | sort the ticks by the clock within each `group` before the call |
| an error naming a series not in `names` | a tick names a series `names` does not list, such as a misspelling | add it to `names`, or filter it out before the call |

Every parameter is in `refresh_time`'s
[docstring](https://hgilde.github.io/polars-online/stream.html#polars_online.stream.refresh_time).
For a block's covariance on the grid's changes, robust to noise, see
[Estimate a block's covariance from
ticks](estimate-a-blocks-covariance-from-ticks.md).
