<!-- kind: how-to -->
# Make features that look back over the stream

This how-to is for anyone who wants a model to read a feature computed
from the rows before each row, such as the last minute's time-weighted
mid. It makes the feature a column in the
[query](../GLOSSARY.md#the-model-bank-and-what-it-reads) (a chain of
Polars steps that runs only when you ask for its result) before the
[model bank](../GLOSSARY.md#the-model-bank-and-what-it-reads), with the
window operators of `po.ops` and `.online.with_windows`.

## Prerequisites

These steps need the library, a stream with a clock, and rows in clock
order:

- polars-online installed, and
  [Example data](../EXAMPLE-DATA.md) built. The examples read
  `trades`, a stream of quotes with trades between them, saved as
  `trades.parquet`. Its `mid` is on every row, and `side`, `quantity` and
  `price` are null on a quote.
- A [clock](../GLOSSARY.md#time) column, here the timestamp column `ts`.
  The operators measure their windows on it. A clock requires a
  [`gap_cap`](../GLOSSARY.md#time), the largest clock step a window
  measures: a longer step, such as a weekend, ends every window open
  across it.
- The rows in clock order across all
  [groups](../GLOSSARY.md#the-model-bank-and-what-it-reads), as one
  stream. [Put rows in clock order](put-rows-in-clock-order.md) shows how.

## Add the features as columns before the bank

A [window operator](../GLOSSARY.md#fitting-a-model) of
[`po.ops`](https://hgilde.github.io/polars-online/ops.html), such as
`po.ewm_mean`, returns a Polars expression (a column computation, such as
`pl.col("x") + 1`). Its value on each row is computed over the rows near
it on the clock. By default the window looks back from the row, so it
makes a feature. Each operator keeps running sums, so it does the same
work on a row however long its window, where Polars' `rolling` gathers the
window's rows again for every row. [What a window operator
costs](../PERFORMANCE.md#what-a-window-operator-costs) has the timings.
[How a window operator weighs the rows near each row](../WINDOWS.md)
lists every operator, and how each one weighs the rows of its window.

1. **Write each feature as a formula around a window operator.** Give it
   a `half_life`, how fast the weights fall away from the row, and, to cut
   the window off, a `window_size`, how far back it reaches. Both are in the clock's
   units: numbers on a numeric clock, and durations such as `"10s"` on a
   temporal one. Around the operators, write element-wise Polars:
   `pl.col("mid") - po.ewm_mean("mid", half_life="5s", window_size="2m")`
   is the mid against its recent mean. [What a formula may
   hold](../WINDOWS.md#why-a-formula-around-the-operators-is-element-wise)
   lists what a formula may hold.
2. **Call `.online.with_windows` on the query, with each formula as a
   keyword argument.** It adds one column per formula after the input's
   columns, named by its keyword or its `.alias()`. The call is
   [`po.stream.with_windows(frame, name=formula, ...)`](https://hgilde.github.io/polars-online/stream.html#polars_online.stream.with_windows),
   written in a chain as `lf.online.with_windows(...)` on a query or
   `df.online.with_windows(...)` on a `DataFrame`.
3. **Give the same call the clock keywords, once for every formula in
   it:** `clock`, its `gap_cap`, and `group` for one set of windows per
   group.
4. **Make the target column with Polars' `with_columns`, chained after
   `with_windows`,** if the input does not hold it.
5. **Chain `.online.fit_predict` after it, with a spec that names the new
   columns in its `features`.** Give the
   [spec](../GLOSSARY.md#the-model-bank-and-what-it-reads) the same clock
   keywords, so the windows and the model share one clock.
6. **Run the query.** Each row of the output has the input's columns, the
   features, and the spec's
   [output column](../GLOSSARY.md#what-a-bank-writes).

This code makes two trends of the mid, over 5 and 30 seconds, and fits
how far from the mid each trade printed on them. It uses `trades.parquet`
from [Example data](../EXAMPLE-DATA.md):

```python
# Features that look back, on one clock shared with the model.
clock = dict(clock="ts", gap_cap="5m", group="symbol")             # one clock for the windows and the model
trends = pl.scan_parquet("trades.parquet").online.with_windows(    # 1 to 3: the features, windows that look back
    trend_5s=pl.col("mid") - po.ewm_mean("mid", half_life="5s", window_size="2m"),
    trend_30s=pl.col("mid") - po.ewm_mean("mid", half_life="30s", window_size="2m"),
    **clock,
)
# 5. A model that reads them: the target, how far from the mid each trade printed, null on a quote.
by_trend = po.spec.ewridge("by_trend", targets=["edge"], features=["trend_5s", "trend_30s"], half_life="30m", **clock)
fitted = (trends.with_columns(edge=(pl.col("price") - pl.col("mid")).abs())  # 4. the target column
                .online.fit_predict([by_trend])
                .collect())                                         # 6. run the query
print(fitted.columns)
print(f"rows {fitted.height}, of which predicted {fitted['by_trend'].struct.field('pred_edge').count()}")
```

```text
['ts', 'symbol', 'mid', 'side', 'quantity', 'price', 'trend_5s', 'trend_30s', 'edge', 'by_trend']
rows 3000, of which predicted 2984
```

## Check the result

**Compare an operator with the Polars function it is named after, on one
group's rows.** Each operator computes what that function computes. On one
symbol's rows, where no gap cuts a window, the running mean is Polars'
`ewm_mean_by`. An evenly weighted sum over the last minute
(`half_life=float("inf")`) is Polars' `rolling_sum_by` over the same
minute, on a stream where no two rows share a clock value, as here. This code
uses `trades.parquet` from [Example data](../EXAMPLE-DATA.md):

```python
aaa = pl.read_parquet("trades.parquet").filter(pl.col("symbol") == "AAA")   # one symbol's quotes and trades
aaa.write_parquet("aaa.parquet")
# The running mean and the last minute's volume, by the library.
ours = pl.scan_parquet("aaa.parquet").online.with_windows(
    mean=po.ewm_mean("mid", half_life="5s"),                                 # no window_size: every row of the stretch
    volume=po.ewm_sum("quantity", half_life=float("inf"), window_size="1m"), # evenly weighted: the last minute's volume
    clock="ts", gap_cap="5m",
).collect()
# The same two by Polars' own functions.
polars = aaa.with_columns(
    mean=pl.col("mid").ewm_mean_by("ts", half_life="5s"),
    volume=pl.col("quantity").fill_null(0.0).rolling_sum_by("ts", "1m"),    # a quote adds no volume
)
for name in ["mean", "volume"]:
    print(f"{name}: largest difference from Polars {(ours[name] - polars[name]).abs().max():.12f}")
```

```text
mean: largest difference from Polars 0.000000000000
volume: largest difference from Polars 0.000000000000
```

## Make a weighted mean, such as a VWAP

**For a weighted mean such as a VWAP (volume-weighted average price),
divide one decayed sum by another:** decayed notional (price times
quantity) by decayed volume, to which the quotes add nothing. For one
side's VWAP, put the same `when/then` inside both sums. The recipe makes
a trend and two VWAPs, each over the last minute. It computes one row's
VWAP again by hand, from the trades in the minute up to it, each weighed
by `0.5 ** (age / 10 s)`. Its last step writes the column to a file with
`sink_parquet`, which runs the query and writes the result to a file
without holding it all in memory. A window that looks ahead, under
`reverse=True`, makes the same column over the rows after each row, which
is a target ([Make a target that looks
ahead](make-a-target-that-looks-ahead.md)). This code uses `trades` and
`trades.parquet` from [Example data](../EXAMPLE-DATA.md):

```python
from datetime import timedelta

# A trend and two VWAPs, each looking back over the last minute.
notional = pl.col("price") * pl.col("quantity")
buys = pl.when(pl.col("side") == "buy")
last_minute = dict(half_life="10s", window_size="1m")
windowed = po.stream.with_windows(
    trades,
    mid_trend=po.ewm_mean("mid", half_life="5s", window_size="1m") - pl.col("mid"),  # the last minute's time-weighted mid, less the mid now
    vwap=po.ewm_sum(notional, **last_minute) / po.ewm_sum("quantity", **last_minute),      # the last minute's trades, weighted most on the latest
    buy_vwap=po.ewm_sum(buys.then(notional), **last_minute) / po.ewm_sum(buys.then("quantity"), **last_minute),   # the same, buys only
    clock="ts", gap_cap="5m", group="symbol",
)
print(windowed.columns)

# One row's vwap again by hand, from the trades in its last minute.
row = windowed.row(1000, named=True)
behind = trades.filter(                                        # the trades in (ts - 1m, ts] of the row's symbol
    pl.col("symbol") == row["symbol"], pl.col("price").is_not_null(),
    pl.col("ts") > row["ts"] - timedelta(minutes=1), pl.col("ts") <= row["ts"],
)
weight = 0.5 ** ((row["ts"] - behind["ts"]).dt.total_microseconds() / 1e6 / 10)
by_hand = (weight * behind["price"] * behind["quantity"]).sum() / (weight * behind["quantity"]).sum()
print(f"row 1000: {behind.height} trades in the last minute; vwap {row['vwap']:.6f}, by hand {by_hand:.6f}")

# The same column over the file, written to parquet.
(pl.scan_parquet("trades.parquet")
    .online.with_windows(vwap=po.ewm_sum(notional, **last_minute) / po.ewm_sum("quantity", **last_minute),
                         clock="ts", gap_cap="5m", group="symbol")
    .sink_parquet("with_windows.parquet"))     # streams, holding about one window of rows
```

```text
['ts', 'symbol', 'mid', 'side', 'quantity', 'price', 'mid_trend', 'vwap', 'buy_vwap']
row 1000: 21 trades in the last minute; vwap 99.529989, by hand 99.529989
```

## Feed the rows in clock order across groups

**Feed `with_windows` the rows in clock order across all groups, as one
stream.** Besides each group's clock, it reads the clock across groups. On
that clock, a [step back](../GLOSSARY.md#time) is refused, naming the row,
unless it is larger than `restart_after_step_back`, which
[resets](../GLOSSARY.md#time) every group's windows. A gap past `gap_cap` on it ends every group's windows. So sort an
input ordered by group with `sort("ts", maintain_order=True)` before the
call. A query whose row order Polars does not guarantee draws an
`OrderNotGuaranteedWarning`. The refusal arrives as the library's
`ValueError` under py-polars 2.0, and as `polars.exceptions.ComputeError`
carrying the same message under 1.x. This code uses `trades` from
[Example data](../EXAMPLE-DATA.md):

```python
import re

# Sorted by symbol, the clock across symbols steps back, and is refused.
by_symbol = trades.sort("symbol")                          # every AAA row, then every BBB row
try:
    po.stream.with_windows(by_symbol, mid_mean=po.ewm_mean("mid", half_life="5s"),
                           clock="ts", gap_cap="5m", group="symbol")
except (ValueError, pl.exceptions.ComputeError) as e:
    print(re.search(r'clock column .* at row \d+', str(e)).group())
# Back in clock order, the same call runs.
in_order = by_symbol.sort("ts", maintain_order=True)       # back in clock order across the symbols
print(po.stream.with_windows(in_order, mid_mean=po.ewm_mean("mid", half_life="5s"),
                             clock="ts", gap_cap="5m", group="symbol").height)
```

```text
clock column "ts" steps back by 24m45s118ms961us at row 1502
3000
```

## Window an operator's output

Use this when a feature is a window over another window, such as the
recent trend of a rate. An operator's input can be an element-wise formula,
`po.increment` included, but not another operator, which the outer operator
refuses when it is called.

1. **Make the inner operator a column in one `.online.with_windows`
   call.**
2. **Window that column in a second `.online.with_windows` call,** chained
   after the first, with the same clock keywords.

Here a rate of traded volume comes from a column of cumulative volume, as
a feed prints it, and a trend of that rate from the second call. Each
symbol trades about 1.56 units a second over the whole stream, and the
rate, read per second, sits near that. This code uses `trades` and
`trades.parquet` from [Example data](../EXAMPLE-DATA.md):

```python
clock = dict(clock="ts", gap_cap="5m", group="symbol")
volume = (
    pl.scan_parquet("trades.parquet")
    .with_columns(cum_volume=pl.col("quantity").cum_sum().over("symbol"))   # a running volume: null on the quotes
    .online.with_windows(                                                    # 1. an operator over an increment
        volume_rate=po.ewm_rate(po.increment("cum_volume"), half_life="30s", window_size="5m"),   # units traded per second
        **clock,
    )
    .online.with_windows(                                                    # 2. a mean of that rate
        rate_trend=po.ewm_mean("volume_rate", half_life="2m", window_size="10m"),
        **clock,
    )
    .collect()
)
# The stream's own rate, beside each symbol's median rate and trend.
seconds = (trades["ts"].max() - trades["ts"].min()).total_seconds()
print(f"units traded a second by each symbol, over the stream: {trades['quantity'].sum() / seconds / 2:.2f}")
for symbol, rate, trend in volume.group_by("symbol").agg(pl.col("volume_rate", "rate_trend").median()).sort("symbol").iter_rows():
    print(f"{symbol}: volume_rate's median {rate:.2f}, rate_trend's median {trend:.2f}")
```

```text
units traded a second by each symbol, over the stream: 1.56
AAA: volume_rate's median 1.61, rate_trend's median 1.51
BBB: volume_rate's median 1.59, rate_trend's median 1.52
```

## Rows the output holds back or leaves out

`with_windows` returns each row once every window over it has its value
([Which rows a window holds](../WINDOWS.md#which-rows-a-window-holds));
under `group`, a window that looks ahead needs `clock`.

**Two things keep a row out of the output: `partial="drop"` (each
operator's `partial` sets what a window cut short by a gap gives), and
`save_state=`, which saves for the next run the rows still waiting for a
window when the input ends.** Without `save_state=`, those rows come out
with their open windows null. [The edges of a window](../WINDOWS.md#the-edges-of-a-window-gaps-sessions-and-resets)
says what `partial` does to a window that a gap or a session change cuts
short, and [Resume a window run](resume-a-window-run.md) shows how to go on
from the saved state.
