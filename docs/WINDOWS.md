# Windowed means, looking back or ahead

A window operator of [`po.ops`](https://hgilde.github.io/polars-online/ops.html)
returns a Polars expression whose value on each row is computed over the
rows near it on the clock: those before it, or those after it. A window
that looks back makes a feature, such as the last minute's time-weighted
mid. A window that looks ahead makes a target, such as the next minute's
VWAP (volume-weighted average price). Each operator does the same work on
a row however long its window, since it keeps running sums, where
Polars' `rolling` gathers the window's rows again for every row
([Window operators](../README.md#window-operators) has the timings). The
operators follow one set of rules in each of three uses:

| use | the call | subsection |
|---|---|---|
| any number of formulas as new columns, before the bank, as `with_columns` adds columns to a frame | [`po.stream.with_windows`](https://hgilde.github.io/polars-online/stream.html#polars_online.stream.with_windows) | [Windows as columns](#windows-as-columns) |
| a look-ahead as the target a model learns, its windows computed inside the bank | a spec's `targets` | [Windows as a model's inputs and target](#windows-as-a-models-inputs-and-target) |
| a `with_windows` run fed in parts, each resuming where the one before it stopped | `save_state=`, then `load_state=` | [Saving and resuming a window run](#saving-and-resuming-a-window-run) |

**Every example on this page is a recipe: a complete program, with what it
prints shown under it.** Most read `trades`, the quotes and trades that
[Example data](../README.md#example-data) builds and saves as
`trades.parquet`, and the line above each says which. Its `mid` is on every
row, and `side`, `quantity` and `price` are null on a quote.
`tests/test_windows_page.py` runs each recipe and holds what it prints to
what is shown.

Eleven operators make the expressions. `half_life` and `window_size` are in
the clock's units: numbers on a numeric clock, and durations (`"10s"`, a
`timedelta`, `pl.duration`) on a temporal one:

| operator | the window of row *t* | what it computes |
|---|---|---|
| `po.ewm_mean(x, half_life=, window_size=)` | the rows at or before *t*, less than `window_size` older; with no `window_size`, every row since the last break | the time-weighted mean, as Polars' `ewm_mean_by` computes it: each value is held over the interval ending at its row and weighed by that interval's decayed time, so a burst of rows does not outweigh a quiet period. A row whose `x` is null is skipped, as `ewm_mean_by` skips a null; so is one whose `x` is NaN, infinite or beyond 1e100. Such a row gets the window's value as it stands, where `ewm_mean_by` gives it null |
| `po.ewm_mean(x, half_life=, window_size=, reverse=True)` | the rows after *t*, at most `window_size` after it; `window_size` is required | the mirror of `ewm_mean`: each value is held until the next row |
| `po.ewm_sum(x, ...)`, `po.ewm_sum(x, ..., reverse=True)` | as for the mean in the same direction | `Σ 0.5 ** (age / half_life) · x`, each row counted once at its own time, as Polars' `ewm_sum_by` computes it on distinct stamps; at a repeated stamp every row carries the stamp's total, where `ewm_sum_by`'s is a running sum. A row whose `x` is null, NaN, infinite or beyond 1e100 adds nothing, and gets the window's sum as it stands, where `ewm_sum_by` gives it null |
| `po.ewm_rate(x, ...)`, `po.ewm_rate(x, ..., reverse=True)` | as for the mean in the same direction | the sum divided by the decayed time the window covers: a quantity per unit of clock. A row whose `x` is null adds nothing, and gets the window's rate as it stands |
| `po.ewm_quantile(x, quantile=, half_life=, window_size=)`, and with `reverse=True` | as for the mean in the same direction | the quantile at `quantile` of the window's values, each row counted once at its own time with the weight `po.ewm_sum` gives it: the smallest value whose share of the weight, with every value below it, reaches `quantile`. It is read from buckets each spanning a ratio of at most `e^(1/64)`, so it is within 0.78% of the exact weighted quantile, relatively ([Quantiles and a robust scale](#quantiles-and-a-robust-scale)). A row whose `x` is null adds nothing, and gets the window's quantile as it stands |
| `po.ewm_var(x, ...)`, `po.ewm_std(x, ...)` | as for `po.ewm_mean`; they look back only | the variance about `po.ewm_mean`, each value weighed as the mean weighs it, and its square root. With `bias=False`, the default, the variance is corrected for the weights' number and inequality as Polars' `ewm_var` corrects it, so a window that one row's weight fills is null. On a clock that steps by 1 a row, with `half_life` in rows, they are Polars' `ewm_var` and `ewm_std` with `adjust=False` (Polars before 1.44 gave the first row 0.0 under `bias=False` where it now gives null, as these do). A window that holds one value gives 0 exactly |
| `po.increment(x)` | back to the last row with a value, within the group and session | `x_t − x_{t−1}`: null on a session's first row and after a restart, and in seconds when `x` is a temporal column |

**Each operator computes what the Polars function it is named after
computes.** On one symbol's rows, where no gap cuts a window, the running
mean is Polars' `ewm_mean_by`. An evenly weighted sum over the last minute
(`half_life=float("inf")`) is Polars' `rolling_sum_by` over the same
minute, on a stream whose stamps are distinct, as these are.
This code uses `trades.parquet` from [Example data](../README.md#example-data):

```python
aaa = pl.read_parquet("trades.parquet").filter(pl.col("symbol") == "AAA")   # one symbol's quotes and trades
aaa.write_parquet("aaa.parquet")
# The running mean and the last minute's volume, by the library.
ours = pl.scan_parquet("aaa.parquet").online.with_windows(
    mean=po.ewm_mean("mid", half_life="5s"),                                 # every row since the stream began
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

Every operator except `po.increment` takes these keywords. `po.ewm_var`
and `po.ewm_std` also take Polars' `bias`, and `po.ewm_quantile` its
`quantile`, above 0 and below 1. Here `min_samples=3` nulls the
first two rows of each symbol, whose windows hold fewer than three values.
This code uses `trades.parquet` from [Example data](../README.md#example-data):

```python
mid_mean = po.ewm_mean(
    "mid",
    half_life="5s",     # how fast the weights fall: largest at the row, halving every 5 s away from it; float("inf") weighs the window evenly
    window_size="1m",   # how far the window reaches from the row; an evenly weighted window needs it, and so does one looking ahead
    closed="right",     # which ends of the window are in: "right", the default
    min_samples=3,      # the fewest rows with a value the window must hold; fewer give null
    partial="keep",     # what a window gives when a gap or a session change cuts it short
)
# The window over the file, each symbol on its own.
windowed_mid = pl.scan_parquet("trades.parquet").online.with_windows(
    mid_mean=mid_mean, clock="ts", gap_cap="5m", group="symbol",
).collect()
print(f"{windowed_mid.height} rows, {windowed_mid['mid_mean'].null_count()} of them null")
```

```text
3000 rows, 4 of them null
```

**A window holds the rows Polars' `rolling_*_by` would give it.** Windows
are taken over timestamps, so every row that shares a timestamp (its
*stamp*) gets the same window, the rows after it at that stamp included.
With `w` the `window_size`, `closed` sets which ends are in, and so when a
window that looks back has its value:

| `closed` | looking back, row *t*'s window | its value is known | looking ahead |
|---|---|---|---|
| `"right"`, the default | `(t − w, t]`: the rows at *t*'s own stamp are in | at the next distinct stamp | `(t, t + w]`: a row exactly `w` after *t* is in |
| `"left"` | `[t − w, t)` | as the stamp arrives | `[t, t + w)`: every row at *t*'s stamp, *t* itself included |
| `"both"` | both ends | at the next distinct stamp | both ends |
| `"none"` | neither end | as the stamp arrives | neither end |

Counting the rows in each window shows the rule. An evenly weighted sum of
a column of ones is the number of rows in the window. Five rows on a
numeric clock, two of them at the stamp 1.0, with a window of 2:

```python
stamps = pl.DataFrame({"t": [0.0, 1.0, 1.0, 2.0, 3.5], "one": 1.0})
# Count the rows in each window, back and ahead, under both closed rules.
counts = po.stream.with_windows(
    stamps,
    **{f"back_{c}": po.ewm_sum("one", half_life=float("inf"), window_size=2.0, closed=c) for c in ["right", "left"]},
    **{f"ahead_{c}": po.ewm_sum("one", half_life=float("inf"), window_size=2.0, closed=c, reverse=True) for c in ["right", "left"]},
    clock="t", gap_cap=10.0,
)
# The counts as a table, a line per row.
show = lambda v: "null" if v is None else f"{v:.0f}"
print("   t  back_right  back_left  ahead_right  ahead_left")
for r in counts.iter_rows(named=True):
    print(f"{r['t']:4.1f}  {show(r['back_right']):>10}  {show(r['back_left']):>9}"
          f"  {show(r['ahead_right']):>11}  {show(r['ahead_left']):>10}")
```

```text
   t  back_right  back_left  ahead_right  ahead_left
 0.0           1       null            3           3
 1.0           3          1            1           3
 1.0           3          1            1           3
 2.0           3          3         null        null
 3.5           2          1         null        null
```

Both rows at 1.0 get one window. Looking back under `"right"`, it is
`(−1, 1]`: the row at 0.0 and both rows at 1.0. Under `"left"` it is
`[−1, 1)`, the row at 0.0 alone, and the row at 0.0 has an empty window,
so null. Looking ahead from 0.0 under `"right"`, `(0, 2]` holds both rows
at 1.0 and the row at 2.0. The last two rows' windows ahead are still open
when the input ends, so they are null.

**Around the operators, write element-wise Polars,** since the formula is
evaluated one chunk at a time:

| | in a formula |
|---|---|
| allowed, around at least one operator | columns, literals, arithmetic, negation, comparisons, `&` and `\|`, `log`, `exp`, `abs`, `sqrt`, `pow`, `clip`, `fill_null`, `is_null`, `is_not_null`, `when/then/otherwise`, `cast` and `alias` |
| refused by name when the query is built, since each would depend on where a chunk ended | a `shift`, a cumulative or rolling function, `over` or an aggregation |
| refused by name when the query is built: write it with Polars' own `with_columns` | a formula with no operator |

This code uses `trades` from [Example data](../README.md#example-data):

```python
clock = dict(clock="ts", gap_cap="5m", group="symbol")
lagged = po.ewm_mean("mid", half_life="5s").shift(1)       # reads the row before: a chunk's first row has none
spread = pl.col("price") - pl.col("mid")                    # reads the row alone, and no operator
for formula in [lagged, spread]:
    try:
        po.stream.with_windows(trades, out=formula, **clock)
    except ValueError as e:
        print(str(e).split(". ")[0])
```

```text
a window formula is element-wise: columns, literals, arithmetic, negation, comparisons, & and |, log, exp, abs, sqrt, pow, clip, fill_null, is_null, is_not_null, when/then/otherwise, cast and alias over the operators; the function 'Shift' is not read
with_windows: 'out' holds no operator; a formula of the row alone is Polars' with_columns
```

**An operator's input can be a formula of the same kind, `po.increment`
included, but not another operator.** So to window an operator's output,
make it a column in one `with_windows` call, and window that column in a
second. Here a rate of traded volume comes from a column of cumulative
volume, as a feed prints it. Each symbol trades about 1.56 units a second
over the whole stream, and the rate, read per second, sits near that.
This code uses `trades` and `trades.parquet` from [Example data](../README.md#example-data):

```python
clock = dict(clock="ts", gap_cap="5m", group="symbol")
volume = (
    pl.scan_parquet("trades.parquet")
    .with_columns(cum_volume=pl.col("quantity").cum_sum().over("symbol"))   # a running volume: null on the quotes
    .online.with_windows(                                                    # the first call: an operator over an increment
        volume_rate=po.ewm_rate(po.increment("cum_volume"), half_life="30s", window_size="5m"),   # units traded per second
        **clock,
    )
    .online.with_windows(                                                    # the second call: that rate, windowed again
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

**The operators read a spec's clock keywords:** `clock`, `gap_cap`,
`restart_after_step_back`, `session`, `session_gap` and `group`. Pass them
to `with_windows` once, for every formula in the call. `gap_cap` is
required with a clock. A window target reads them from its spec. With
`group`, each group keeps its own sessions and windows. A session needs no
`session_gap` here, as a spec's does: every window ends at a session
change, so no window measures the clock's step across one. A number or a
duration given for it changes nothing, and raises a
`PolarsOnlineDeprecationWarning`. Only `session_gap="reset"` acts, on the
windows looking ahead that a session change leaves open (below). A
`session_gap` without `session`, and a `restart_after_step_back` without
`clock`, would do nothing, and are refused.

**A gap longer than `gap_cap`, or a session change, ends every window open
across it, and the operator's `partial` sets what the cut window gives.**
A window that looks back is cut short in the same way where it reaches back
past the stream's first row, or past the first row after a break or a
restart. A restart (a step back larger than `restart_after_step_back`, or a
session change under `session_gap="reset"`) discards the open windows
instead, so their rows come out null, never dropped:

| `partial` | a cut window gives | for a window target, the row is | the default for |
|---|---|---|---|
| `"keep"` | the value over what it saw, the window ending at the last row seen | learned from what the window saw | looking back |
| `"null"` | null | not learned | looking ahead |
| `"drop"` | nothing: the row leaves the output | not learned, as under `"null"`, while its other targets still are | |

`partial` needs a `window_size`, since without one no window is cut short.
On `trades`, a window of a minute looking back is cut short on each
symbol's first minute of rows, 102 rows in all.
This code uses `trades` from [Example data](../README.md#example-data):

```python
for partial in ["keep", "null", "drop"]:
    out = po.stream.with_windows(
        trades,
        mid_mean=po.ewm_mean("mid", half_life="5s", window_size="1m", partial=partial),
        clock="ts", gap_cap="5m", group="symbol",
    )
    print(f'partial="{partial}": {out.height} rows, {out["mid_mean"].null_count()} null')
```

```text
partial="keep": 3000 rows, 0 null
partial="null": 3000 rows, 102 null
partial="drop": 2898 rows, 0 null
```

**A window that looks ahead under `closed="right"` or `"both"` is whole
when its far edge is the last row before a break or a restart.** That row
is in the window and no later row can join it, so the window gives its
value under any `partial`. For a window target, a restart still starts the
model over and drops every row waiting to be learned, a whole one with the
rest.

## Windows as columns

Call `po.stream.with_windows(frame, name=formula, ...)` before the bank. It
adds one column per formula after the input's columns, named by its keyword
or its `.alias()`. In a chain, write it as `lf.online.with_windows(...)` on
a query or `df.online.with_windows(...)` on a `DataFrame`.

**For a weighted mean such as a VWAP, divide one decayed sum by another:**
decayed notional (price times quantity) by decayed volume, to which the
quotes add nothing. For one side's VWAP, put the same `when/then` inside
both sums. The recipe computes one row's VWAP again by hand, from the
trades in the minute after it, each weighed by `0.5 ** (age / 10 s)`.
This code uses `trades` and `trades.parquet` from [Example data](../README.md#example-data):

```python
from datetime import timedelta

# A trend that looks back, and two VWAPs that look ahead.
notional = pl.col("price") * pl.col("quantity")
buys = pl.when(pl.col("side") == "buy")
next_minute = dict(half_life="10s", window_size="1m", reverse=True)
windowed = po.stream.with_windows(
    trades,
    mid_trend=po.ewm_mean("mid", half_life="5s", window_size="1m") - pl.col("mid"),  # the last minute's time-weighted mid, less the mid now
    fwd_vwap=po.ewm_sum(notional, **next_minute) / po.ewm_sum("quantity", **next_minute),      # the next minute's trades, weighted most on the next one
    fwd_buy_vwap=po.ewm_sum(buys.then(notional), **next_minute) / po.ewm_sum(buys.then("quantity"), **next_minute),   # the same, buys only
    clock="ts", gap_cap="5m", group="symbol",
)
print(windowed.columns)

# One row's fwd_vwap again by hand, from the trades in its next minute.
row = windowed.row(1000, named=True)
ahead = trades.filter(                                         # the trades in (ts, ts + 1m] of the row's symbol
    pl.col("symbol") == row["symbol"], pl.col("price").is_not_null(),
    pl.col("ts") > row["ts"], pl.col("ts") <= row["ts"] + timedelta(minutes=1),
)
weight = 0.5 ** ((ahead["ts"] - row["ts"]).dt.total_microseconds() / 1e6 / 10)
by_hand = (weight * ahead["price"] * ahead["quantity"]).sum() / (weight * ahead["quantity"]).sum()
print(f"row 1000: {ahead.height} trades in the next minute; fwd_vwap {row['fwd_vwap']:.6f}, by hand {by_hand:.6f}")

# The same column over the file, streamed to parquet.
(pl.scan_parquet("trades.parquet")
    .online.with_windows(fwd_vwap=po.ewm_sum(notional, **next_minute) / po.ewm_sum("quantity", **next_minute),
                         clock="ts", gap_cap="5m", group="symbol")
    .sink_parquet("with_windows.parquet"))     # streams, holding about one window of rows
```

```text
['ts', 'symbol', 'mid', 'side', 'quantity', 'price', 'mid_trend', 'fwd_vwap', 'fwd_buy_vwap']
row 1000: 22 trades in the next minute; fwd_vwap 99.516156, by hand 99.516156
```

**Feed `with_windows` the rows in clock order across all groups, as one
stream.** Besides each group's clock, it reads the clock across groups. On
that clock, a step back is refused, naming the row, unless it is larger
than `restart_after_step_back`, which restarts every group. A gap past
`gap_cap` on it ends every group's windows. So sort an input ordered by
group with `sort("ts", maintain_order=True)` before the call. A query whose
row order Polars does not guarantee draws an `OrderNotGuaranteedWarning`.
The refusal arrives as the library's `ValueError` under py-polars 2.0, and
as `polars.exceptions.ComputeError` carrying the same message under 1.x.
This code uses `trades` from [Example data](../README.md#example-data):

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
clock column "ts" goes backwards by 24m45s118ms961us at row 1502
3000
```

**`with_windows` returns the rows in input order, each once every window
over it has its value,** so the output trails the input by the longest
window. A group that falls silent holds back the rows after it for at most
`gap_cap` of the stream's time. Only a clock column has a `gap_cap`, so a
window that looks ahead under `group` needs `clock`, and `with_windows`
raises `ValueError` without it. Each group's clock would count only its own
rows, and a silent group's windows would hold back every later row until
the input ended.

**Two things keep a row out of the output: `partial="drop"`, and
`save_state=`, which saves for the next run the rows still waiting for a
window when the input ends.** Without `save_state=`, those rows come out
with their open windows null.

## Windows as a model's inputs and target

To give a model a feature that looks back, make it a column with
`with_windows` in the query, chain `.online.fit_predict` after it, and name
the column in the spec's `features`. To make a window that looks ahead the
target, put its expression in the spec's `targets`. The bank then computes
the window itself, one set per group on the spec's clock, so the groups may
interleave in time.

**Give a target expression an operator that looks ahead (`po.ewm_mean`,
`po.ewm_sum`, `po.ewm_rate` or `po.ewm_quantile` with `reverse=True`) and a name, with `.alias()`.** The alias
is used to name the output fields: `.alias("fwd_edge")` gives
`pred_fwd_edge`. Without an operator that looks ahead, the target is known
at its own row, and the spec builder raises `ValueError`. Make such a
target a column in the query before the bank instead, and name the column
in `targets`.
Make it with `with_windows` if it reads a window that looks back, or with
Polars' `with_columns` if it reads only its own row. The builder refuses
`group_close` beside a window target, and `group` without `clock`, as
`with_windows` does.

**Each row is scored where it sits, and learned from once its window has
closed and its `embargo` has passed.** A state that learned a row before
its window closed would score the rows that window covers in sample. So
`fit_predict` requires an `embargo` of at least the target's
`window_size`; otherwise it raises `ValueError` before reading a row. For a
shorter embargo, or none, use `bank.fit` or the command line's
`--no-output`, which keep no predictions.

**`resid_<name>` is null on every row (`resid_fwd_edge` in the recipe
below), because a
row's target is not known when the row is scored.** The diagnostics take
each row in when it is learned, as under any `embargo` ([Labels that arrive
late](../README.md#labels-that-arrive-late)).

**To see the targets the model learned, write the target as a column with
`with_windows(fwd_edge, like=edge)`, on a frame that holds the spec's
features, as `trends` does in the recipe.** `like=edge` takes the clock
keywords from `edge`, and refuses any given beside it. It nulls the target
on every row the spec would not learn from.

**Named in `targets` under the same `embargo`, that column gives the same
predictions row for row, except when the embargo equals the window under
`closed="right"`.** Then the column form learns each row at the row exactly
one window after it, and the expression in `targets` at the next distinct
stamp after that. The recipe gives the embargo two minutes, past the
window's one, and the two forms agree on every row.
This code uses `trades.parquet` from [Example data](../README.md#example-data):

```python
# Features that look back, on one clock shared with the model.
clock = dict(clock="ts", gap_cap="5m", group="symbol")             # one clock for the windows and the model
trends = pl.scan_parquet("trades.parquet").online.with_windows(    # the features: windows that look back
    trend_5s=pl.col("mid") - po.ewm_mean("mid", half_life="5s", window_size="2m"),
    trend_30s=pl.col("mid") - po.ewm_mean("mid", half_life="30s", window_size="2m"),
    **clock,
)
# A target that looks ahead, learned two minutes late.
notional = pl.col("price") * pl.col("quantity")
next_minute = dict(half_life="10s", window_size="1m", reverse=True)
fwd_edge = (po.ewm_sum(notional, **next_minute) / po.ewm_sum("quantity", **next_minute) - pl.col("mid")).alias("fwd_edge")   # the target: the next minute's VWAP, less the mid
edge = po.spec.ewridge("edge", targets=[fwd_edge], features=["trend_5s", "trend_30s"],
                       half_life="30m", embargo="2m", **clock)   # each row learned two minutes later, once its minute is known
fitted = trends.online.fit_predict([edge]).collect()      # trades, the two trends, and the column edge

# The same target as a column, and the same model on it.
as_column = trends.online.with_windows(fwd_edge, like=edge).collect()   # the target the model learned, as a column
by_column = po.spec.ewridge("by_column", targets=["fwd_edge"], features=["trend_5s", "trend_30s"],
                            half_life="30m", embargo="2m", **clock)     # the same model, its target named as that column
column_form = as_column.online.fit_predict([by_column])

# The two forms compared row for row.
expression, column = fitted["edge"].struct, column_form["by_column"].struct
print(f"rows {fitted.height}, of which predicted {fitted.height - expression.field('pred_fwd_edge').null_count()}")
print(f"largest difference between the two forms: {(expression.field('pred_fwd_edge') - column.field('pred_fwd_edge')).abs().max()}")
print(f"resid_fwd_edge null on {expression.field('resid_fwd_edge').null_count()} rows")

# An embargo shorter than the target's window is refused.
short = po.spec.ewridge("short", targets=[fwd_edge], features=["trend_5s"], half_life="30m", embargo="30s", **clock)
try:
    trends.online.fit_predict([short])                    # refused when the query is built
except ValueError as e:
    print(str(e).split(" (got")[0])
```

```text
rows 3000, of which predicted 2761
largest difference between the two forms: 0.0
resid_fwd_edge null on 3000 rows
spec "short": fit_predict needs an embargo of at least 1m, the longest window_size among its formula targets' forward operators
```

## Features in units of their spread

**When a target responds to a feature relative to the feature's current
volatility, fit the model on the feature's z-score.** A coefficient on a
raw feature is a response per unit of the feature, the same in a calm hour
and a volatile one. A coefficient on its z-score is a response per standard
deviation, at the feature's spread as it stands on that row. Make each
z-score a column with `po.ewm_mean` and `po.ewm_std` in `with_windows`,
chained in the query before the bank, and name the columns in the spec's
`features`.

`ticks.parquet` holds the README's `df`, whose `y` is `1 + 2·x0 − x1` plus noise, so per standard deviation of
`x0` the response is 2 times that spread. The recipe prints the spread
`po.ewm_std` gives against Polars' own `ewm_std`, and the fitted
coefficients against the truth.
This code uses `ticks.parquet` from [Example data](../README.md#example-data):

```python
# Each feature in units of its current spread, and a fit on those.
row_clock = dict(clock="t", gap_cap=10.0)                # one clock for the z-scores and the model; t steps by 1 a row
spread = dict(half_life=100.0)                           # how fast the mean and the spread forget, in the clock's units
zscored = pl.scan_parquet("ticks.parquet").online.with_windows(
    z_x0=(pl.col("x0") - po.ewm_mean("x0", **spread)) / po.ewm_std("x0", **spread),   # x0 in units of its current spread
    z_x1=(pl.col("x1") - po.ewm_mean("x1", **spread)) / po.ewm_std("x1", **spread),
    sd_x0=po.ewm_std("x0", **spread), sd_x1=po.ewm_std("x1", **spread),
    **row_clock,
)
per_sd = po.spec.ewridge("per_sd", targets=["y"], features=["z_x0", "z_x1"], half_life=100.0, **row_clock)
fitted = zscored.online.fit_predict([per_sd]).collect()   # the rows, the z-scores, the spreads and the column per_sd

# The spread against Polars', and the fit against the truth.
polars_sd = pl.read_parquet("ticks.parquet")["x0"].ewm_std(half_life=100.0, adjust=False)
print(f"po.ewm_std against Polars' ewm_std: largest difference {(fitted['sd_x0'] - polars_sd).abs().max():.12f}")
last = fitted.row(-1, named=True)
intercept, b0, b1 = last["per_sd"]["coef"]
print(f"per standard deviation: z_x0 {b0:.2f} (truth 2 x {last['sd_x0']:.2f} = {2 * last['sd_x0']:.2f}), "
      f"z_x1 {b1:.2f} (truth -1 x {last['sd_x1']:.2f} = {-last['sd_x1']:.2f})")
```

```text
po.ewm_std against Polars' ewm_std: largest difference 0.000000000000
per standard deviation: z_x0 1.81 (truth 2 x 0.95 = 1.90), z_x1 -0.92 (truth -1 x 0.98 = -0.98)
```

The fitted responses sit within about a tenth of the truth, each read at
the feature's spread on the last row.

**The operators run over a stream in one pass, where Polars' own forms
need the column in memory.** On any clock but one that steps by 1 a row,
where they are Polars' forms with `adjust=False` (the table of operators
above), they weigh each row by the time it covers.

**A z-score adds one column per feature, and it works with every model.**
Each z-score is one more column, made by two operators on one queue in the
`with_windows` pass. A model reads it as any other column, so `ewridge`,
`rls`, `kalman` and the rest take it alike. A model's coefficients are per
unit of its features, so z-scored features give coefficients per standard
deviation.

**On minute returns, every model lost less with z-scored features.**
Section 6 of [docs/VALIDATION.md](VALIDATION.md) fits three models, four ways, to
BTCUSDT's next one and next five minutes of log returns (`ewridge` at two
solve cadences, `rls` and `kalman`), each at a
half-life of 500 minutes and with the features raw, then z-scored. No model
predicts these returns better than their mean, so every R² is below 0, but
each is closer to 0 on the z-scores:

| model | next minute, raw | next minute, z-scores | next five minutes, raw | next five minutes, z-scores |
|---|---|---|---|---|
| `ewridge`, at its default solve cadence | -0.0148 | -0.0096 | -0.0521 | -0.0262 |
| `ewridge`, solved every row | -0.0647 | -0.0412 | -0.1113 | -0.0474 |
| `rls` | -0.0644 | -0.0412 | -0.1108 | -0.0475 |
| `kalman` | -0.0699 | -0.0455 | -0.1005 | -0.0427 |

## Quantiles and a robust scale

**`po.ewm_quantile` is the quantile of the window's values, each weighed as
`po.ewm_sum` weighs it.** Row *j* weighs `0.5 ** (age / half_life)`, its age
taken from the row the window is read at. The quantile at `q` is the
smallest value whose share of the weight, counting it and every value below
it, reaches `q`. That is numpy's weighted quantile,
`np.quantile(x, q, weights=w, method="inverted_cdf")`. Under
`half_life=float("inf")` every row of the window weighs the same, and it is
a rolling quantile.

**It reads each value as a bucket, so it is within 0.78% of the exact
quantile, relatively.** A value `m · 2^e`, with `m` in `[1, 2)`, lands in
bucket `⌊64 (e + m − 1)⌋`; a negative value lands below zero by its
magnitude's bucket. A bucket spans a ratio of at most `e^(1/64)`, and the
value reported for it is within `tanh(1/128)`, about 0.0078, of every value
in it, at any scale and either sign. A magnitude below the smallest normal
double, about 2.2e-308, reads as 0. These are DDSketch's buckets (Masson,
Rim and Lee, 2019), the ones `po.spec.huber`'s MAD scale reads too.

**A row leaves a hard window without a subtraction.** The windowed
operators keep a window's rows in two stacks: new rows join the newer one,
and when the older one is empty the newer one turns over into it. A row
leaving must take its weight out of its bucket. A subtraction would leave a
rounding behind in a bucket that should be empty, and the roundings would
build up over a long stream. So at the turnover each row keeps what its
bucket, its bucket's octave and the total weighed before it, the newest
row first. A row leaving puts those three numbers back, and the sketch is
then exactly the one built from the rows that are left. A sketch kept per
row, the textbook form of a two-stack queue, would cost the whole sketch a
row: about 11 KB for a window of 1,000 rows, where this costs 24 bytes.

| the window | what it keeps | its sketches, on standard normal values | time a row, against the same window's `po.ewm_mean` |
|---|---|---|---|
| no `window_size` | one sketch of every row since the stretch began, decayed, folding the decay in every 64 half-lives and then dropping an end octave that holds under 1e-12 of the weight | 1,985 buckets, about 16 KB, at a half-life of 500 rows | 1.9 times |
| a `window_size` | the window's rows, as every windowed operator keeps them, three numbers a row, and two sketches | 1,345 buckets for a window of 1,000 rows | 1.7 times |
| `reverse=True` | as under a `window_size` | as under a `window_size` | 1.9 times `po.ewm_sum`'s |

The times are the best of five runs over 2,000,000 rows of one group,
interleaved, through `po.stream.with_windows`, measured on 2026-10-24 on a
shared laptop at a load average of 8 to 12. A read
walks the buckets an octave at a time, then the one octave's 64 buckets, so
a quantile costs about as much as its window's values span octaves.

**The median of `|x|` over `Φ⁻¹(3/4)` is the median absolute deviation of
`x` about 0, a scale that a few wild values do not move.** `Φ⁻¹(3/4) =
0.6744897501960817` is the MAD of a standard normal, so for normal values
the result estimates their standard deviation. It holds while up to half the
weight is wild, where one value of 1e6 sets a standard deviation. The recipe
adds a wild print of 1.00 to every 40th trade, whose prints sit about 0.02
from the mid, and compares the two scales. It checks the last row against
numpy's weighted median by hand.
This code uses `trades` from [Example data](../README.md#example-data):

```python
# The trades' prints about the mid, a few of them wild: every 40th print 1.00 off.
prints = trades.filter(pl.col("price").is_not_null()).with_row_index("i").with_columns(
    dev=pl.col("price") - pl.col("mid") + pl.when(pl.col("i") % 40 == 0).then(1.0).otherwise(0.0),
)
spread = dict(half_life="2m")
scales = po.stream.with_windows(
    prints,
    mad=po.ewm_quantile(pl.col("dev").abs(), quantile=0.5, **spread) / 0.6744897501960817,   # the EW-MAD about 0
    std=po.ewm_std("dev", **spread),                                                         # moved by every wild print
    clock="ts", gap_cap="5m", group="symbol",
)
# The last row's EW-MAD again by hand: numpy's weighted median of |dev| over its symbol's prints.
last = scales.row(-1, named=True)
mine = scales.filter(pl.col("symbol") == last["symbol"])
minutes = (last["ts"] - mine["ts"]).dt.total_microseconds().to_numpy() / 6e7
by_hand = np.quantile(mine["dev"].abs(), 0.5, weights=0.5 ** (minutes / 2), method="inverted_cdf") / 0.6744897501960817
print(f"last row: mad {last['mad']:.5f}, by hand {by_hand:.5f}, apart by {abs(last['mad'] / by_hand - 1):.4f}")
print(f"medians over the prints: mad {scales['mad'].median():.4f}, std {scales['std'].median():.4f}; the prints' spread is 0.02")
```

```text
last row: mad 0.02126, by hand 0.02132, apart by 0.0027
medians over the prints: mad 0.0204, std 0.1815; the prints' spread is 0.02
```

The two values of the last row are 0.27% apart, inside the buckets' 0.78%.
One print in 40 at 50 times the spread multiplies the standard deviation by
nine, and moves the MAD by 2%.

**Centre a value before its MAD.** The MAD about 0 is a scale only for a
value centred on 0, such as a model's residual or a return. A column with a
level needs its median taken off first, and an operator's input cannot hold
another operator. So make the median a column in one `with_windows` call,
and take the MAD of the deviation from it in a second.

### A scale of a target, used as a weight

**A weight read from a target must leave out the labels not yet known on
the row it weighs.** A spec's `weight` multiplies the row's place in the
fit. A weight from the row's own label, such as `1 / scale²` or Huber's
`min(1, c · scale / |y − m|)` about a level `m`, would let a wild label set its own weight, and
the label of a row is not known at that row: in a backtest it would weigh
the row with the future. Two cases cover it:

| the label of row *t* is known | the scale on row *t* | how |
|---|---|---|
| at the next row | over the rows before *t* | `closed="left"`: the rows at *t*'s own stamp are outside its window, so its label is too |
| `d` of the clock later, as under a spec's `embargo` | over the rows whose labels have arrived by *t*, those at or before `t − d` | the stream twice, from `po.stream.embargo(lf, clock=, delay=d)`: a predict row at *t* and a learn row at `t + d`; the operator reads the label only on the learn rows, `pl.when(pl.col("_online_role") == "learn").then(...)`, and the predict rows, filtered back, carry the scale |

A shift of the label column by `k` rows in Polars' `with_columns`, before
the call (`pl.col("y").shift(k).over("symbol")`), lags it by rows rather
than by clock: right only on a stream whose labels arrive a
fixed number of rows late. The doubled stream lags it by the clock, as an
`embargo` does, and in the same order: a label that arrives at the row's
own stamp counts there.

**For the scale of a model's residual, use the scale `po.spec.huber` can
emit.** It is the model's own robust scale of its out-of-sample residuals,
read before the row, and it takes each residual in when the row is
learned, so the spec's `embargo` holds it back as it holds back the fit.

Here `ret`'s label arrives 5 rows late. The recipe computes its EW-MAD on
each row from the labels that have arrived, and checks row 250 by hand
against numpy's weighted median of the labels of rows 0 to 245 alone.
This code uses `df` and `ticks.parquet` from [Example data](../README.md#example-data):

```python
# Each label of ret arrives 5 rows late: the stream twice, a predict row at t and a learn row at t + 5.
doubled = po.stream.embargo(pl.scan_parquet("ticks.parquet"), clock="t", delay=5.0)
known = pl.when(pl.col("_online_role") == "learn").then(pl.col("ret").abs())   # a label counts once it has arrived
scaled = (
    doubled.online.with_windows(
        ret_scale=po.ewm_quantile(known, quantile=0.5, half_life=20.0) / 0.6744897501960817,
        clock="t", gap_cap=10.0,
    )
    .filter(pl.col("_online_role") == "predict")                               # back to one row per row
    .collect()
)
# Row 250's scale again by hand, from the labels of rows 0 to 245 alone.
arrived = df["ret"].abs().to_numpy()[:246]
weight = 0.5 ** ((250 - (np.arange(246) + 5)) / 20.0)                          # each weighed from its arrival
by_hand = np.quantile(arrived, 0.5, weights=weight, method="inverted_cdf") / 0.6744897501960817
got = scaled["ret_scale"][250]
print(f"row 250: ret_scale {got:.4f}, by hand from rows 0 to 245 {by_hand:.4f}, apart by {abs(got / by_hand - 1):.4f}")
print(f"null on the first {scaled['ret_scale'].null_count()} rows, before any label has arrived")
print("in the calm and the volatile stretches:", ", ".join(f"{scaled['ret_scale'][i]:.2f}" for i in (90, 190, 290, 390)))
```

```text
row 250: ret_scale 1.1409, by hand from rows 0 to 245 1.1362, apart by 0.0041
null on the first 5 rows, before any label has arrived
in the calm and the volatile stretches: 0.91, 2.30, 0.77, 1.66
```

`ret`'s noise has a spread of 0.5 on rows 0-99 and 200-299 and 2.0 on the
others, so its spread is about 0.94 in the calm stretches and 2.15 in the
volatile ones, and the scale follows it.

## Saving and resuming a window run

To feed `with_windows` a stream in two runs, give the first `save_state=`
and the second `load_state=`. The second run returns the rows the first
left waiting before its own, so two runs give what one run gives. A state
loads only into a call with the same formulas, clock keywords and kind of
clock; otherwise `load_state=` raises `ValueError`. A window target is
saved with its bank's state instead ([Saving, loading and
serving](../README.md#saving-loading-and-serving)). The state file's format
is not promised: a state one release saved need not load in the next.

**Under a slice such as `head(500)`, the run reads the input only until the
windows of the last row asked for have closed, and the state records how
many input rows it consumed.** A run resumed on the same input, unsliced,
skips those rows and goes on. The recipe cuts `trades` in two files, and
also stops a run after 500 rows: each pair of runs gives what one run
gives. An input that starts before the last row the state read is refused
as a step back.
This code uses `trades` and `trades.parquet` from [Example data](../README.md#example-data):

```python
# The forward VWAP in one run over the whole stream.
next_minute = dict(half_life="10s", window_size="1m", reverse=True)
vwap = dict(fwd_vwap=po.ewm_sum(pl.col("price") * pl.col("quantity"), **next_minute) / po.ewm_sum("quantity", **next_minute))
clock = dict(clock="ts", gap_cap="5m", group="symbol")
one_run = po.stream.with_windows(trades, **vwap, **clock)            # one run

# Two files, the second run resumed from the state the first saved.
trades.head(2000).write_parquet("day1.parquet")                       # the same stream as two files
trades.slice(2000).write_parquet("day2.parquet")
first = pl.scan_parquet("day1.parquet").online.with_windows(**vwap, **clock, save_state="w.state").collect()   # its last rows wait
second = pl.scan_parquet("day2.parquet").online.with_windows(**vwap, **clock, load_state="w.state").collect()  # they come out first
print(f"two files: {first.height} rows, then {second.height}; one run's output: {pl.concat([first, second]).equals(one_run)}")

# A run stopped at head(500), then resumed on the same file.
head = pl.scan_parquet("trades.parquet").online.with_windows(**vwap, **clock, save_state="h.state").head(500).collect()
rest = pl.scan_parquet("trades.parquet").online.with_windows(**vwap, **clock, load_state="h.state").collect()  # the same file: it skips the rows read
print(f"head(500), then the rest: {head.height} rows, then {rest.height}; one run's output: {pl.concat([head, rest]).equals(one_run)}")

# An input that starts before the state's last row is refused.
trades.slice(300).write_parquet("overlap.parquet")                   # starts before the last row the state read
try:
    pl.scan_parquet("overlap.parquet").online.with_windows(**vwap, **clock, load_state="h.state").collect()
except (ValueError, pl.exceptions.ComputeError) as e:                # ComputeError under py-polars 1.x
    message = str(e)[str(e).index("with_windows"):]
    print(message.split(". ")[0])
```

```text
two files: 1884 rows, then 1116; one run's output: True
head(500), then the rest: 500 rows, then 2500; one run's output: True
with_windows: the state was saved under a slice of another input: this input starts before the last row the state read, a step back the clock policy refuses
```

The first run returns 1,884 of its 2,000 rows. The other 116 wait in the
state for the next minute of trades, and the second run returns them first.

**A state saved under a slice records its input's first row (clock and
session) and the last rows it read, so it refuses the wrong input, never
silently skipping or doubling rows.** If the clock starts over at the same
stamp each day, add a session column, such as the date, and name it in
`session=`. Otherwise the next day's file starts as
the saved input did, and passes for it. If the stream's rows can repeat, as
a daily grid's can, resume only on the same input: another input that
matches the state's rows where it was cut is taken for the same input.

**Any resumed run refuses an input whose first row is at the last stamp
the state read, unless that row starts a new session.** A file boundary
inside a stamp that several rows share looks the same as an input that
repeats rows the state read, which would come out twice. So cut a stream
into files between stamps. Without a clock column there is no stamp to
compare, and a repeated row is not caught.

**A run resumed from a state saved under a slice decides from its input's
first row whether to skip rows, go on or refuse.** A refusal names
`another input`. It arrives as `polars.exceptions.ComputeError` under
py-polars 1.x and as `ValueError` under 2.0:

| the input a resumed run is given | the run |
|---|---|
| the input the state was saved from, unsliced | skips the rows the state consumed, then goes on |
| the next file, whose first row is a step forward on the clock or a new start (a step back past `restart_after_step_back`, or a new session) | skips nothing, and returns the rows the state held before its own |
| an input whose first row is at the last stamp the state read | refused, as after any state: a file boundary inside a stamp that several rows share looks the same as the saved input sliced inside it |
| an input that steps back where the clock policy refuses it, such as the same input sliced by hand, or an overlapping file | refused |
| without a clock column, an input that does not begin with a new session | refused: a row-count clock steps forward at every row |
| an input that starts as the saved one did but differs where the state was cut, or ends before the rows consumed | refused |
