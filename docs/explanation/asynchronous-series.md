<!-- kind: explanation -->
# Series that tick at their own times

This page is for anyone who fits a correlation, a covariance or a
regression across series observed at instants of their own, such as the
trades of several stocks. It explains why such series bias a correlation
toward zero, and the two ways out the library offers.

**A correlation measured on returns over short intervals falls toward
zero when the series tick at their own times.** Epps (1979) found it in
stock prices, and it carries his name: the Epps effect. Two series can
share every move and still read almost no correlation, because the moves
they share are recorded at different instants.

## Why the correlation falls

**A series that has not ticked records no move, so the moves two series
share land on different rows.** Previous-tick sampling carries each
series' last observed value forward to every row. On a row where one
series ticked and the other did not, the first records its move and the
second records none. The second records the same move later, on the row
where it next ticks. So a move both series share appears in the first
series' return on one row and in the second's on a later row.

**The returns keep their variances and lose their covariance.** Each
series' moves all appear, each in some return, so the sum of its squared
returns is unchanged in expectation: the moves a return gathers are
independent, so their cross terms average to zero. The product of the two series' returns on one row
counts a shared move only where both recorded it on that row. The rest of
the shared moves sit at a lag, in the covariance of one series' return
with the other's on a later row. A correlation at lag 0 divides the
shrunken covariance by the whole variances, so it falls toward zero.

**A longer interval loses less, and gives fewer returns.** Over an
interval of many rows, most shared moves are recorded by both series
within it, and only those near its edges fall into the next interval. So
the correlation rises toward the truth as the interval grows. But an
interval `k` rows long gives a `k`-th of the returns, so the estimate
grows noisier as it grows less biased.

## The two ways out

**A refresh grid samples the series only once each has ticked.**
`po.stream.refresh_time` puts a refresh time at the first instant by which
every series has ticked since the previous refresh time, the refresh-time rule of
Barndorff-Nielsen, Hansen, Lunde and Shephard (2011). No interval between
two refresh times is one in which a series recorded nothing, so most shared moves
fall in the same interval for both. It does not remove the bias. Each
series' value at a refresh time is its last tick, which can be up to one of its
own intervals between ticks old. So a move shortly before a refresh time can be
recorded by one series in this interval and by the other in the next. The
refresh grid runs at the pace of the slowest series and drops the faster series'
other ticks. [Align series that tick at their own
times](../how-to/align-series-that-tick-at-their-own-times.md) builds the
refresh grid and fits models on it.

**The lag inversion reads every return, and sums the covariance the lags
hold back.** The covariance the effect moves away from lag 0 is in the
lagged cross-covariances, and `ew_cov(lags=...)` keeps those co-moments
as it runs. `po.corr.epps_invert` sums them over lags up to `L − 1`, for a scale of
`L` rows, with
the triangular weights of Tóth and Kertész (2009). Those weights make the
sum the covariance of returns over `L` rows, taken from every window of
`L` rows rather than from a grid, with the variances summed the same way.
So it reads the correlation at a scale of `L` rows from every one-row
return. `L` makes the same trade-off as the sampling interval: too small,
and a shared move recorded more than `L − 1` rows apart is left out; too
large, and the sum takes in the noise of many lags. Its formula is in the
[docstring](https://hgilde.github.io/polars-online/corr.html#polars_online.corr.epps_invert).

## The effect and both ways out on one stream

This code plants a correlation of 0.8 between two series, each observed
on about a quarter of the rows at random, with `po.sim.regimes`. It reads
the correlation by previous-tick sampling at three intervals, on the
refresh grid, and by the lag inversion at `L = 64`:

```python
import polars as pl
import polars_online as po

# Two series whose returns correlate at 0.8, each observed on about a quarter of the rows.
sim = po.sim.regimes(2, states=[0.8], transition=[[1.0]], n_blocks=1, rows_per_block=40_000,
                     async_rates=[0.25, 0.25], seed=7)
sim["rows"].select("t", "x_1", "x_2").write_parquet("own_ticks.parquet")
own_ticks = pl.read_parquet("own_ticks.parquet")
# Previous-tick sampling: each series' last value carried forward, returns every `step` rows.
last = own_ticks.select(pl.col("x_1", "x_2").forward_fill()).drop_nulls()
for step in (1, 10, 100):
    moves = last.gather_every(step).select(pl.all().diff()).drop_nulls()
    print(f"previous tick, every {step:>3} rows: corr {moves.corr()['x_1'][1]:.3f} from {moves.height} returns")
# The refresh grid: a refresh time once both series have ticked.
long = own_ticks.unpivot(index="t", variable_name="series", value_name="px").drop_nulls().sort("t")
grid = po.stream.refresh_time(long, series="series", names=["x_1", "x_2"], clock="t", value="px")
moves = grid.select(pl.col("x_1_value", "x_2_value").diff()).drop_nulls()
print(f"refresh grid:              corr {moves.corr()['x_1_value'][1]:.3f} from {moves.height} returns")
# The lag inversion: every one-row return, with the lagged co-moments summed over L rows.
ones = last.select(pl.all().diff()).drop_nulls()
L = 64
lagged = po.spec.ew_cov("c", features=["x_1", "x_2"], half_life=float("inf"), stats=[], lags=list(range(1, L)))
bank = po.ModelBank([lagged])
bank.fit_predict(ones)
print(f"lag inversion, L = {L}:      corr {po.corr.epps_invert(bank.gram('c')[0], L=L)[0, 1]:.3f} from {ones.height} returns")
```

```text
previous tick, every   1 rows: corr 0.099 from 39989 returns
previous tick, every  10 rows: corr 0.514 from 3998 returns
previous tick, every 100 rows: corr 0.719 from 399 returns
refresh grid:              corr 0.531 from 6294 returns
lag inversion, L = 64:      corr 0.727 from 39989 returns
```

At one row, previous-tick sampling keeps an eighth of the planted 0.8.
Over 100 rows it recovers 0.72, from a hundredth of the returns. The
refresh grid recovers two thirds of it, and the lag inversion 0.73 while
it reads every return. [REGIMES.md](../REGIMES.md#6-the-epps-effect-and-the-two-ways-out)
measures both ways out on a longer stream, over a sweep of `L` and five
seeds.

## Sources

| source | for |
|---|---|
| Barndorff-Nielsen, O. E., Hansen, P. R., Lunde, A. and Shephard, N. (2011). Multivariate realised kernels: consistent positive semi-definite estimators of the covariation of equity prices with noise and non-synchronous trading. *Journal of Econometrics* 162(2), 149–169. | the refresh-time rule |
| Epps, T. W. (1979). Comovements in stock prices in the very short run. *Journal of the American Statistical Association* 74(366), 291–298. | the fall of a correlation over short intervals |
| Tóth, B. and Kertész, J. (2009). The Epps effect revisited. *Quantitative Finance* 9(7), 793–802. | the lagged co-moments that undo it, and their triangular weights |
