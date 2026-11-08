<!-- kind: explanation -->
# How a window operator weighs the rows near each row

This page explains, for anyone who makes features or targets with the
window operators of [`po.ops`](https://hgilde.github.io/polars-online/ops.html),
which rows each operator's window holds and how it weighs them on a clock
whose steps are uneven. It also explains why the formula around the
operators reads each row alone, what a window gives at its edges, and how
a quantile and a saved run are kept.

**A window operator returns a Polars expression whose value on each row is
computed over the rows near it on the [clock](GLOSSARY.md#time): those
before it, or those after it.** A window that looks back makes a feature,
such as the last minute's time-weighted mid. A window that looks ahead,
under `reverse=True`, makes a target, such as the next minute's VWAP
(volume-weighted average price). Each operator updates running sums, or
for a quantile a sketch, as each row arrives, where Polars' `rolling`
gathers the window's rows again for every row ([What a window operator
costs](PERFORMANCE.md#what-a-window-operator-costs) has the timings).

[Make features that look back over the
stream](how-to/make-window-features.md) adds windows as columns before the
bank with `po.stream.with_windows`, and [Make a target that looks
ahead](how-to/make-a-target-that-looks-ahead.md) gives a model one as its
target. [Scale a feature by its spread](how-to/scale-a-feature-by-its-spread.md)
fits a model on a feature's z-score, made with `po.ewm_mean` and
`po.ewm_std`. Each operator's parameters, their defaults and what each refuses
are in its docstring under
[`po.ops`](https://hgilde.github.io/polars-online/ops.html).

## A row's weight on an uneven clock

**A row's weight falls with its distance on the clock from the row the
window is read at, and not with its count of rows.** A row `half_life`
away weighs half as much, and `half_life=float("inf")` weighs every row of
the window the same. `half_life` and `window_size` are in the clock's
units: numbers on a numeric clock, and durations such as `"10s"` on a
temporal one. Each operator then reads the weighted rows its own way:

| operator | how it weighs the window's rows | what it is held to |
|---|---|---|
| `po.ewm_mean` | each value is held over the interval ending at its row and weighed by that interval's decayed time, so a burst of rows does not outweigh a quiet period, and a row at a repeated clock value moves no mean. Looking ahead, each value is held until the next row | `ewm_mean_by` |
| `po.ewm_sum` | `Σ 0.5 ** (age / half_life) · x`, each row counted once at its own time. Every row at a repeated clock value carries that value's total | `ewm_sum_by` on distinct clock values, whose sum at a repeated one runs row by row |
| `po.ewm_rate` | the sum divided by the decayed time the window covers: a quantity per unit of clock | the sum and the decayed time, each as above |
| `po.ewm_quantile` | each row weighed as `po.ewm_sum` weighs it ([Quantiles and a robust scale](#quantiles-and-a-robust-scale)) | numpy's weighted quantile (`inverted_cdf`), within 0.78% |
| `po.ewm_var`, `po.ewm_std` | about `po.ewm_mean`, each value weighed as the mean weighs it, and looking back only. With `bias=False`, the default, the variance is corrected for the weights' number and inequality, so a window that one row's weight fills is null. A window that holds one value gives 0 exactly | `ewm_var` and `ewm_std` with `adjust=False`, on a clock that steps by 1 a row with `half_life` in rows. Polars before 1.44 gave the first row 0.0 under `bias=False`, where it now gives null, as these do |

`po.increment`, which is not a window operator, gives an operator an
input: `x_t − x_{t−1}`, back to the last row with a value in the group and
session. It is null on a session's first row and after a reset, and in
seconds when `x` is a temporal column.

[Check the result](how-to/make-window-features.md#check-the-result)
compares `po.ewm_mean` and `po.ewm_sum` with Polars' `ewm_mean_by` and
`rolling_sum_by`, on one symbol's rows.

**A row whose value is missing adds nothing, and gets the window's value
as it stands.** A null is missing, and so is a NaN, an infinite value or
one beyond 1e100 in magnitude. Polars' `ewm_mean_by` and `ewm_sum_by`
give such a row null.

**A time-weighted mean weighs each value by how long it held.** Here `x`
is 1 for ten clock units, and then 0 on ten rows in the next unit. A mean
that counts each row once reads the burst of zeros as most of the window.
`po.ewm_mean` reads the time each value held, and the first value of a
[stretch](#which-rows-a-window-holds) is held from before the stretch
begins. This code builds its own rows:

```python
# x is 1 from t = 0 to t = 10, then 0 on ten rows in the next clock unit.
burst = [10.1 + 0.1 * i for i in range(10)]
rows = pl.DataFrame({"t": [0.0, 10.0, *burst], "x": [0.0, 1.0, *[0.0] * 10], "one": 1.0})
# The last ten clock units, every row weighed evenly, by its count and by its time.
even = dict(half_life=float("inf"), window_size=10.0)
out = po.stream.with_windows(
    rows,
    rows_in=po.ewm_sum("one", **even),                                  # how many rows the window holds
    by_count=po.ewm_sum("x", **even) / po.ewm_sum("one", **even),       # each row counted once
    by_time=po.ewm_mean("x", **even),                                   # each value weighed by the time it held
    clock="t", gap_cap=20.0,
)
last = out.row(-1, named=True)
print(f"at t = {last['t']:.1f}: {last['rows_in']:.0f} rows in the window, x = 1 on one of them, for 9 of its 10 clock units")
print(f"mean by count {last['by_count']:.3f}, time-weighted mean {last['by_time']:.3f}")
```

```text
at t = 11.0: 11 rows in the window, x = 1 on one of them, for 9 of its 10 clock units
mean by count 0.091, time-weighted mean 0.900
```

The window at `t = 11` is `(1, 11]`. The row at 10 holds its value of 1
over `(0, 10]`, of which 9 units are inside the window, and the zeros hold
the last unit. So the time-weighted mean is 0.9, where a mean by count is
1 in 11.

## Which rows a window holds

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

**A row is in a window by its stamp, but a mean weighs it by the part of
its held interval inside the window.** So under `"left"` or `"both"`, a row
exactly one window old counts toward `min_samples`, the fewest rows with a
value a window must hold, and weighs nothing in a mean, since its interval
ends at its row.

**Without a `window_size`, a window holds every row since its stretch
began.** A *stretch* is the rows between two clock breaks: it starts at the
stream's first row, or at the first row after a gap longer than `gap_cap`,
a session change or a reset. Those rows are weighed by their decay alone.
An evenly weighted mean, variance or standard deviation
(`half_life=float("inf")`) needs a `window_size`, and `with_windows`
refuses one without it.

## Why a formula around the operators is element-wise

**`with_windows` computes a formula one chunk of rows at a time, so the
Polars around the operators must read each row alone.** A step that reads
another row, such as a `shift`, would give a chunk's first row a different
value from the one it gets in the middle of a chunk. So the call refuses
it by name, and chunking the stream cannot change its output:

| | in a formula |
|---|---|
| allowed, around at least one operator | columns, literals, arithmetic, negation, comparisons, `&` and `\|`, `log`, `exp`, `abs`, `sqrt`, `pow`, `clip`, `fill_null`, `is_null`, `is_not_null`, `when/then/otherwise`, `cast` and `alias` |
| refused by name when the query is built, since each would depend on where a chunk ended | a `shift`, a cumulative or rolling function, `over` or an aggregation |
| refused by name when the query is built: write it with Polars' own `with_columns` | a formula with no operator |

This code tries both refusals on `trades`, a stream of quotes with trades
between them. Its `mid` is on every row, and `side`, `quantity` and
`price` are null on a quote. This code uses `trades` from [Example
data](../README.md#example-data):

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
included, but not another operator.** To window an operator's output, make
it a column in one `with_windows` call, and window that column in a second
([Window an operator's
output](how-to/make-window-features.md#window-an-operators-output)).

## The edges of a window: gaps, sessions and resets

**The operators read a spec's clock keywords, given once to `with_windows`
for every formula in the call.** These are `clock`, `gap_cap`,
`restart_after_step_back`, `session`, `session_gap` and `group`, and
`gap_cap` is required with a clock. A window target reads them from its
spec. With `group`, each group keeps its own sessions and windows.

**Unlike a spec, `with_windows` needs no `session_gap` for a session, since
every window ends at a session change.** So no window measures the clock's
step across one. A number or a duration given for `session_gap` changes nothing,
and raises a `PolarsOnlineDeprecationWarning`. Only `session_gap="reset"`
acts, on the windows looking ahead that a session change leaves open.

**A gap longer than `gap_cap`, or a session change, ends every window open
across it, and the operator's `partial` sets what the cut window gives.**
A window that looks back is cut short in the same way where it reaches back
past the stream's first row, or past the first row after a gap or a reset.
A [reset](GLOSSARY.md#time) (a step back larger than
`restart_after_step_back`, or a session change under
`session_gap="reset"`) discards the open windows instead, so their rows
come out null, never dropped:

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

**A window that looks ahead under `closed="right"` or `"both"` gives its
value under any `partial` when its far edge falls on the last row before a
gap or a reset.** That row is in the window and no later row can join it, so the
window is whole. For a window target, a reset still starts the model
over and drops every row waiting to be learned, a whole one with the rest.

## Quantiles and a robust scale

**`po.ewm_quantile` is the quantile of the window's values, each weighed as
`po.ewm_sum` weighs it.** Row *j* weighs `0.5 ** (age / half_life)`, its age
taken from the row the window is read at. The quantile at `quantile=q` is
the smallest value whose share of the weight, counting it and every value
below it, reaches `q`. That is numpy's weighted quantile,
`np.quantile(x, q, weights=w, method="inverted_cdf")`. Under
`half_life=float("inf")` every row of the window weighs the same, and it is
the plain quantile of the window's values.

**It reads each value as a bucket, so it is within 0.78% of the exact
quantile, relatively.** A value `m · 2^e`, with `m` in `[1, 2)`, lands in
bucket `⌊64 (e + m − 1)⌋`; a negative value lands below zero by its
magnitude's bucket. So each octave (the range from one power of two to the
next) holds 64 buckets. A bucket spans a ratio of at most `e^(1/64)`, and the
value reported for it is within `tanh(1/128)`, about 0.0078, of every value
in it, at any scale and either sign. A magnitude below the smallest normal
double, about 2.2e-308, reads as 0. These are DDSketch's buckets (Masson,
Rim and Lee, 2019), the ones `po.spec.huber`'s MAD scale reads too.

**A row leaves a hard window without a subtraction.** The windowed
operators keep a window's rows in two stacks: new rows join the newer one,
and when the older one is empty the newer one turns over into it. A row
leaving must take its weight out of its bucket. A subtraction would leave a
rounding behind in a bucket that should be empty, and the roundings would
build up over a long stream. So at the turnover each row records three
weights, taken newest row first: what its bucket, its bucket's octave and
the whole sketch held over the rows newer than it. A row leaving puts those three weights back, and the sketch is
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

**A window run's state holds the rows still waiting for their windows to
close, so a run resumed from it returns those rows before its own.** Two
runs then give what one run gives. [Resume a window
run](how-to/resume-a-window-run.md) shows how to save and load the state,
and which inputs a resumed run takes or refuses.
