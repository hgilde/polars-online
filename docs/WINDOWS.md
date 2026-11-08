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

| use | the call | where |
|---|---|---|
| any number of formulas as new columns, before the bank, as `with_columns` adds columns to a frame | [`po.stream.with_windows`](https://hgilde.github.io/polars-online/stream.html#polars_online.stream.with_windows) | [Make features that look back over the stream](how-to/make-window-features.md) |
| a look-ahead as the target a model learns, its windows computed inside the bank | a spec's `targets` | [Make a target that looks ahead](how-to/make-a-target-that-looks-ahead.md) |
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

[Check the result](how-to/make-window-features.md#check-the-result) shows how
to hold an operator to the Polars function it is named after, such as
`ewm_mean_by` and `rolling_sum_by`, on one symbol's rows.

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
with_windows: a formula over window operators reads each row alone, but its expression (a column computation, such as pl.col("x") + 1) holds the method shift, which it cannot read
with_windows: 'out' holds no window operator, so it reads only its own row
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

[Make features that look back over the stream](how-to/make-window-features.md)
shows how to add windows as columns before the bank with
`po.stream.with_windows`, a VWAP among them, and feed the rows in clock
order across groups.

[Make a target that looks ahead](how-to/make-a-target-that-looks-ahead.md)
shows how to give a model a window target in its `targets`, under an
`embargo` of at least the window, and check the targets it learned.

[Scale a feature by its spread](how-to/scale-a-feature-by-its-spread.md)
shows how to fit a model on a feature's z-score against its own recent
spread, made with `po.ewm_mean` and `po.ewm_std`.

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

[Weight rows by a robust scale of the target](how-to/weight-rows-by-a-robust-scale.md)
shows how to make a MAD scale with `po.ewm_quantile`, and weight each row
by it without reading targets not yet known.

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
with_windows: the state was saved from a slice of its input, and this is another input: it starts before the last row the state read
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
