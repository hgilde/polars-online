# Windowed means, looking back or ahead

Each operator of [`po.ops`](https://hgilde.github.io/polars-online/ops.html)
returns a Polars expression. The examples run on `trades`, the quotes and
trades built in [Example data](../README.md#example-data). The operators follow one set
of rules in each of three uses:

| use | the call | subsection |
|---|---|---|
| any number of formulas as new columns, before the bank, as `with_columns` adds columns to a frame | [`po.stream.with_windows`](https://hgilde.github.io/polars-online/stream.html#polars_online.stream.with_windows) | [Windows as columns](#windows-as-columns) |
| a look-ahead as the target a model learns, its windows computed inside the bank | a spec's `targets` | [Windows as a model's inputs and target](#windows-as-a-models-inputs-and-target) |
| a `with_windows` run fed in parts, each resuming where the one before it stopped | `save_state=`, then `load_state=` | [Saving and resuming a window run](#saving-and-resuming-a-window-run) |

Nine operators make the expressions. Each takes the same work on a row
however long its window, while Polars' `rolling` gathers a window's rows
again for every row ([Window operators](../README.md#window-operators)):

| operator | the window of row *t* | what it computes |
|---|---|---|
| `po.ewm_mean(x, half_life=, window_size=)` | the rows at or before *t*, less than `window_size` older; with no `window_size`, every row since the last break | the time-weighted mean, as Polars' `ewm_mean_by` computes it: each value is held over the interval ending at its row and weighed by that interval's decayed time, so a burst of rows does not outweigh a quiet period. A row whose `x` is null is skipped, as `ewm_mean_by` skips a null; so is one whose `x` is NaN, infinite or beyond 1e100. Such a row gets the window's value as it stands, where `ewm_mean_by` gives it null |
| `po.rewm_mean(x, half_life=, window_size=)` | the rows after *t*, at most `window_size` after it; `window_size` is required | the mirror of `ewm_mean`: each value is held until the next row |
| `po.ewm_sum(x, ...)`, `po.rewm_sum(x, ...)` | as for the mean in the same direction | `Σ 0.5 ** (age / half_life) · x`, each row counted once at its own time, as Polars' `ewm_sum_by` computes it on distinct stamps; at a repeated stamp every row carries the stamp's total, where `ewm_sum_by`'s is a running sum. A row whose `x` is null, NaN, infinite or beyond 1e100 adds nothing, and gets the window's sum as it stands, where `ewm_sum_by` gives it null |
| `po.ewm_rate(x, ...)`, `po.rewm_rate(x, ...)` | as for the mean in the same direction | the sum divided by the decayed time the window covers: a quantity per unit of clock. A row whose `x` is null adds nothing, and gets the window's rate as it stands |
| `po.ewm_var(x, ...)`, `po.ewm_std(x, ...)` | as for `po.ewm_mean`; they look back only | the variance about `po.ewm_mean`, each value weighed as the mean weighs it, and its square root. With `bias=False`, the default, the variance is corrected for the weights' number and inequality as Polars' `ewm_var` corrects it, so a window that one row's weight fills is null. On a clock that steps by 1 a row, with `half_life` in rows, they are Polars' `ewm_var` and `ewm_std` with `adjust=False` (Polars before 1.44 gave the first row 0.0 under `bias=False` where it now gives null, as these do). A window that holds one value gives 0 exactly |
| `po.increment(x)` | back to the last row with a value, within the group and session | `x_t − x_{t−1}`: null on a session's first row and after a restart, and in seconds when `x` is a temporal column |

Every operator except `po.increment` takes these keywords, `half_life` and
`window_size` in the clock's units. `po.ewm_var` and `po.ewm_std` also take
Polars' `bias`.
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
windowed_mid = pl.scan_parquet("trades.parquet").online.with_windows(mid_mean=mid_mean, clock="ts", gap_cap="5m", group="symbol").collect()
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

**Around the operators, write element-wise Polars,** since the formula is
evaluated one chunk at a time:

| | in a formula |
|---|---|
| allowed, around at least one operator | columns, literals, arithmetic, negation, comparisons, `log`, `exp`, `abs`, `sqrt`, `pow`, `clip`, `fill_null`, `is_null`, `is_not_null`, `when/then/otherwise`, `cast` and `alias` |
| refused by name when the query is built, since each would depend on where a chunk ended | a `shift`, a cumulative or rolling function, `over` or an aggregation |
| refused by name when the query is built: write it with Polars' own `with_columns` | a formula with no operator |

**An operator's input can be a formula of the same kind, `po.increment`
included, but not another operator.** So to window an operator's output,
make it a column in one `with_windows` call, and window that column in a
second. Here a rate of traded volume comes from a column of cumulative
volume.
This code uses `trades.parquet` from [Example data](../README.md#example-data):

```python
clock = dict(clock="ts", gap_cap="5m", group="symbol")
volume = (
    pl.scan_parquet("trades.parquet")
    .with_columns(cum_volume=pl.col("quantity").cum_sum().over("symbol"))   # a running volume, as a feed prints it: null on the quotes
    .online.with_windows(                                                    # the first call: an operator over an increment
        volume_rate=po.ewm_rate(po.increment("cum_volume"), half_life="30s", window_size="5m"),   # the volume traded per second
        **clock,
    )
    .online.with_windows(                                                    # the second call: that rate, windowed again
        rate_trend=po.ewm_mean("volume_rate", half_life="2m", window_size="10m"),
        **clock,
    )
    .collect()
)
```

**The operators read a spec's clock keywords:** `clock`, `gap_cap`,
`restart_after_step_back`, `session`, `session_gap` and `group`. Pass them
to `with_windows` once, for every formula in the call. A window target
reads them from its spec. With `group`, each group keeps its own sessions.

**A gap longer than `gap_cap`, or a session change, ends every window open
across it, and the operator's `partial` sets what the cut window gives.**
A window that looks back is cut short in the same way where it reaches back
past the stream's first row, or past the first row after a break or a
restart. So under `partial="null"` the rows less than `window_size` after
the stream's first row give null, and under `"drop"` they leave the
output. A restart (a step back larger than `restart_after_step_back`, or a
session change under `session_gap="reset"`) discards the open windows
instead, so their rows come out null, never dropped:

| `partial` | a cut window gives | for a window target, the row is | the default for |
|---|---|---|---|
| `"keep"` | the value over what it saw, the window ending at the last row seen | learned from what the window saw | looking back |
| `"null"` | null | not learned | looking ahead |
| `"drop"` | nothing: the row leaves the output | not learned, as under `"null"`, while its other targets still are | |

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

For a weighted mean such as a VWAP (volume-weighted average price), divide
one decayed sum by another: decayed notional (price times quantity) by
decayed volume, to which the quotes add nothing. For one side's VWAP, put
the same `when/then` inside both sums.
This code uses `trades` and `trades.parquet` from [Example data](../README.md#example-data):

```python
notional = pl.col("price") * pl.col("quantity")
buys = pl.when(pl.col("side") == "buy")
next_minute = dict(half_life="10s", window_size="1m")
windowed = po.stream.with_windows(
    trades,
    mid_trend=po.ewm_mean("mid", half_life="5s", window_size="1m") - pl.col("mid"),  # the last minute's time-weighted mid, less the mid now
    fwd_vwap=po.rewm_sum(notional, **next_minute) / po.rewm_sum("quantity", **next_minute),      # the next minute's trades, weighted most on the next one
    fwd_buy_vwap=po.rewm_sum(buys.then(notional), **next_minute) / po.rewm_sum(buys.then("quantity"), **next_minute),   # the same, buys only
    clock="ts", gap_cap="5m", group="symbol",
)
# every input column, then mid_trend, fwd_vwap, fwd_buy_vwap

(pl.scan_parquet("trades.parquet")
    .online.with_windows(fwd_vwap=po.rewm_sum(notional, **next_minute) / po.rewm_sum("quantity", **next_minute),
                         clock="ts", gap_cap="5m", group="symbol")
    .sink_parquet("with_windows.parquet"))     # streams, holding about one window of rows
```

**Feed `with_windows` the rows in clock order across all groups, as one
stream.** Besides each group's clock, it reads the clock across groups. On
that clock, a step back is refused, naming the row, unless it is larger
than `restart_after_step_back`, which restarts every group. A gap past
`gap_cap` on it ends every group's windows. So sort an input ordered by
group with `sort("ts", maintain_order=True)` before the call. A query whose
row order Polars does not guarantee draws an `OrderNotGuaranteedWarning`.

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

**Give a target expression an operator that looks ahead (`po.rewm_mean`,
`po.rewm_sum` or `po.rewm_rate`) and a name, with `.alias()`.** The alias
is used to name the output fields: `.alias("fwd_edge")` gives
`pred_fwd_edge`. Without an operator that looks ahead, the target is known
at its own row, and the spec builder raises `ValueError`. Make such a
target a column in the query before the bank instead, and name the column
in `targets`.
Make it with `with_windows` if it reads a window that looks back, or with
Polars' `with_columns` if it reads only its own row. The builder refuses
`group_close` beside a window target, and `group` without `clock`, as
`with_windows` does.
This code uses `trades.parquet` from [Example data](../README.md#example-data):

```python
clock = dict(clock="ts", gap_cap="5m", group="symbol")             # one clock for the windows and the model
trends = pl.scan_parquet("trades.parquet").online.with_windows(    # the features: windows that look back
    trend_5s=pl.col("mid") - po.ewm_mean("mid", half_life="5s", window_size="2m"),
    trend_30s=pl.col("mid") - po.ewm_mean("mid", half_life="30s", window_size="2m"),
    **clock,
)
notional = pl.col("price") * pl.col("quantity")
next_minute = dict(half_life="10s", window_size="1m")
fwd_edge = (po.rewm_sum(notional, **next_minute) / po.rewm_sum("quantity", **next_minute) - pl.col("mid")).alias("fwd_edge")   # the target: the next minute's VWAP, less the mid
edge = po.spec.ewridge("edge", targets=[fwd_edge], features=["trend_5s", "trend_30s"],
                       half_life="30m", embargo="1m", **clock)   # each row learned once the next minute is known
fitted = trends.online.fit_predict([edge]).collect()      # trades, the two trends, and the column edge
as_column = trends.online.with_windows(fwd_edge, like=edge).collect()   # the target the model learned, as a column
by_column = po.spec.ewridge("by_column", targets=["fwd_edge"], features=["trend_5s", "trend_30s"],
                            half_life="30m", embargo="1m", **clock)     # the same model, its target named as that column
column_form = as_column.online.fit_predict([by_column])                 # the column form: the same predictions as fitted's
```

**Each row is scored where it sits, and learned from once its window has
closed and its `embargo` has passed.** A state that learned a row before
its window closed would score the rows that window covers in sample. So
`fit_predict` requires an `embargo` of at least the target's
`window_size`; otherwise it raises `ValueError` before reading a row. For a
shorter embargo, or none, use `bank.fit` or the command line's
`--no-output`, which keep no predictions.

**`resid_<name>` is null on every row (`resid_fwd_edge` here), because a
row's target is not known when the row is scored.** The diagnostics take
each row in when it is learned, as under any `embargo` ([Labels that arrive
late](../README.md#labels-that-arrive-late)).

**To see the targets the model learned, write the target as a column with
`with_windows(fwd_edge, like=edge)`, on a frame that holds the spec's
features, as `trends` does.** `like=edge` takes the clock keywords from
`edge`, and refuses any given beside it. It nulls the target on every row
the spec would not learn from.

**Named in `targets` under the same `embargo`, that column gives the same
predictions row for row, except when the embargo equals the window under
`closed="right"`, as in the example.** Then the column form learns each row
at the row exactly one window after it, and the expression in `targets` at
the next distinct stamp after that.

## Features in units of their spread

**When a target responds to a feature relative to the feature's current
volatility, fit the model on the feature's z-score.** A coefficient on a
raw feature is a response per unit of the feature, the same in a calm hour
and a volatile one. A coefficient on its z-score is a response per standard
deviation, at the feature's spread as it stands on that row. Make each
z-score a column with `po.ewm_mean` and `po.ewm_std` in `with_windows`,
chained in the query before the bank, and name the columns in the spec's
`features`.
This code uses `ticks.parquet` from [Example data](../README.md#example-data):

```python
row_clock = dict(clock="t", gap_cap=10.0)                # one clock for the z-scores and the model
spread = dict(half_life=100.0)                           # how fast the mean and the spread forget, in the clock's units
zscored = pl.scan_parquet("ticks.parquet").online.with_windows(
    z_x0=(pl.col("x0") - po.ewm_mean("x0", **spread)) / po.ewm_std("x0", **spread),   # x0 in units of its current spread
    z_x1=(pl.col("x1") - po.ewm_mean("x1", **spread)) / po.ewm_std("x1", **spread),
    **row_clock,
)
per_sd = po.spec.ewridge("per_sd", targets=["y"], features=["z_x0", "z_x1"], half_life=100.0, **row_clock)
fitted = zscored.online.fit_predict([per_sd]).collect()   # the rows, z_x0, z_x1 and the column per_sd
```

**On a frame in memory, Polars' own `ewm_mean` and `ewm_std` make the same
column.** On a clock that steps by 1 a row, `po.ewm_std("x0",
half_life=100.0)` is `pl.col("x0").ewm_std(half_life=100.0, adjust=False)`
to rounding, and so is the mean. The window operators read the clock
rather than the row count, and run over a stream in one pass.

**The recipe costs a column per feature, and it works with every model.**
Each z-score is one more column, made by two operators on one queue in the
`with_windows` pass. A model reads it as any other column, so `ewridge`,
`rls`, `kalman` and the rest take it alike.

**On minute returns, every model lost less with z-scored features.**
Section 6 of [docs/VALIDATION.md](VALIDATION.md) fits four models to
BTCUSDT's next one and next five minutes of log returns, each at a
half-life of 500 minutes and with the features raw, then z-scored. No model
predicts these returns better than their mean, so every R² is below 0, but
each is closer to 0 on the z-scores:

| model | next minute, raw | next minute, z-scores | next five minutes, raw | next five minutes, z-scores |
|---|---|---|---|---|
| `ewridge`, at its default solve cadence | -0.0148 | -0.0096 | -0.0521 | -0.0262 |
| `ewridge`, solved every row | -0.0647 | -0.0412 | -0.1113 | -0.0474 |
| `rls` | -0.0644 | -0.0412 | -0.1108 | -0.0475 |
| `kalman` | -0.0592 | -0.0412 | -0.0746 | -0.0585 |

**`kalman` fitted this model on its own in 0.13.0.** It standardized each
feature by the feature's moving moments, and kept its coefficients per
standard deviation. It now keeps them per unit, as every other model does.
For the model per standard deviation, give `kalman` the z-scored columns.

## Saving and resuming a window run

To feed `with_windows` a stream in two runs, give the first `save_state=`
and the second `load_state=`. The second run returns the rows the first
left waiting before its own, so two runs give what one run gives. An input
that starts before the last row the first run read is refused as a step
back. A state loads only into a call with the same formulas, clock keywords
and kind of clock; otherwise `load_state=` raises `ValueError`. A window
target is saved with its bank's state instead ([Saving, loading and
serving](../README.md#saving-loading-and-serving)).

Under a slice such as `head(500)`, the run reads the input only until the
windows of the last row asked for have closed, and the state records how
many input rows it consumed.
This code uses `trades` and `trades.parquet` from [Example data](../README.md#example-data):

```python
next_minute = dict(half_life="10s", window_size="1m")
vwap = dict(fwd_vwap=po.rewm_sum(pl.col("price") * pl.col("quantity"), **next_minute) / po.rewm_sum("quantity", **next_minute))
clock = dict(clock="ts", gap_cap="5m", group="symbol")
one_run = po.stream.with_windows(trades, **vwap, **clock)            # one run

day1, day2 = trades.head(2000), trades.slice(2000)                   # the same stream as two files
first = po.stream.with_windows(day1, **vwap, **clock, save_state="w.state")    # its last rows wait
second = po.stream.with_windows(day2, **vwap, **clock, load_state="w.state")   # they come out first
assert pl.concat([first, second]).equals(one_run)

head = pl.scan_parquet("trades.parquet").online.with_windows(**vwap, **clock, save_state="h.state").head(500).collect()
rest = pl.scan_parquet("trades.parquet").online.with_windows(**vwap, **clock, load_state="h.state").collect()
assert pl.concat([head, rest]).equals(one_run)                       # the same input, unsliced: it skips them
```

**A state saved under a slice records its input's first row (clock and
session) and the last rows it read, so it refuses the wrong input, never
silently skipping or doubling rows.** If the clock starts over at the same
stamp each day, add a session column, such as the date, and name it in
`session=` with a `session_gap`. Otherwise the next day's file starts as
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
