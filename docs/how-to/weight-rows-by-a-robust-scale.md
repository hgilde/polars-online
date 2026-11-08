<!-- kind: how-to -->
# Weight rows by a robust scale of the target

This how-to is for anyone who wants each row to count in a fit by a weight
from a scale of its target that a few wild values do not move, such as
`1 / scale²`.
It makes the scale with `po.ewm_quantile` from the targets already known
on each row, so a row's own target never sets its weight.

## Prerequisites

These steps need the library and a stream with a clock:

- polars-online installed, and the README's
  [Example data](../../README.md#example-data) built. The examples read
  `trades`, a stream of quotes with trades between them, and `df`, saved
  as `ticks.parquet`. `df`'s `ret` is `±0.8·signal_a` plus noise of
  spread 0.5 on rows 0-99 and 200-299 and 2.0 on the others.
- A [clock](../GLOSSARY.md#time) column, and the
  [`gap_cap`](../GLOSSARY.md#time) a clock requires: the largest clock
  step a window or a model measures. A longer step ends every window open
  across it.
- When each row's target becomes known: at the next row, or a fixed delay
  on the clock later, as under a spec's [`embargo`](../GLOSSARY.md#time)
  (the delay on the clock between predicting a row and learning from it).

## Make a robust scale with `po.ewm_quantile`

**Divide the median of `|x|` by `Φ⁻¹(3/4)` for a scale that a few wild
values do not move: the median absolute deviation (MAD) of `x` about 0.**
`Φ⁻¹(3/4) = 0.6744897501960817` is the MAD of a standard normal, so for
normal values the result estimates their standard deviation. It holds while
up to half the weight is wild, where a single value of 1e6 is enough to set
a standard deviation. `po.ewm_quantile` is the quantile of a window's values, each
weighed by its age on the clock, as [docs/WINDOWS.md](../WINDOWS.md#quantiles-and-a-robust-scale)
describes.

1. **Center the value first.** The MAD about 0 is a scale only for a
   value centered on 0, such as a model's
   [residual](../GLOSSARY.md#fitting-a-model) or a return. A column with a
   level needs its median taken off before its MAD, and an operator's input cannot
   hold another operator. So make the median a column in one
   `with_windows` call, and take the MAD of the deviation from it in a
   second.
2. **Make the scale a column with `po.ewm_quantile` of the absolute
   value, at `quantile=0.5`, divided by `0.6744897501960817`.** Call it
   in `with_windows` before the
   [model bank](../GLOSSARY.md#the-model-bank-and-what-it-reads), with
   the stream's clock keywords.

The recipe takes the exponentially weighted MAD (EW-MAD) of each trade's
print less the mid. That deviation is already centered on 0, so step 1
has nothing to do. The recipe adds a wild print of 1.00 to every 40th
trade, whose prints sit about 0.02 from the mid, and compares the two
scales. It checks the last
row against numpy's weighted median by hand. This code uses `trades` from
[Example data](../../README.md#example-data):

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

The two values of the last row are 0.27% apart, inside the 0.78% within
which `po.ewm_quantile` reads a quantile. One print in 40 at 50 times the
spread multiplies the standard deviation by nine, and moves the MAD by 2%.

## Leave out the targets not yet known

**A weight read from a target must leave out the targets not yet known on
the row it weighs.** A spec's `weight` multiplies the row's place in the
fit. A weight from the row's own target, such as `1 / scale²` or Huber's
`min(1, c · scale / |y − m|)` about a level `m`, would let a wild target
value set its own weight. And the target of a row is not known at that
row: in a backtest it would weigh the row with the future. Choose the
window by when each row's target becomes known:

| the target of row *t* is known | the scale on row *t* | how |
|---|---|---|
| at the next row | over the rows before *t* | `closed="left"`: the rows at *t*'s own stamp are outside its window, so its target is too |
| `d` of the clock later, as under a spec's `embargo` | over the rows whose targets have arrived by *t*, those at or before `t − d` | the stream twice, from `po.stream.embargo(lf, clock=, delay=d)`: a predict row at *t* and a learn row at `t + d`; the operator reads the target only on the learn rows, `pl.when(pl.col("_online_role") == "learn").then(...)`, and the predict rows, filtered back, carry the scale |

A shift of the target column by `k` rows in Polars' `with_columns`, before
the call (`pl.col("y").shift(k).over("symbol")`), lags it by rows rather
than by clock: right only on a stream whose targets arrive a fixed number
of rows late. The doubled stream lags it by the clock, as an `embargo`
does, and in the same order: a target that arrives at the row's own stamp
counts there.

For a target known `d` of the clock later:

1. **Double the stream with `po.stream.embargo(query, clock=, delay=d)`.**
   Each row comes out twice: a predict row at its own clock value, and a
   learn row `d` later, told apart by the column `_online_role`.
2. **Read the target only on the learn rows,** with
   `pl.when(pl.col("_online_role") == "learn").then(...)` inside the
   operator.
3. **Filter back to the predict rows** with
   `.filter(pl.col("_online_role") == "predict")`. Each carries the scale
   from the targets that had arrived by its clock value.

Here each row's `ret` arrives 5 rows late. The recipe computes its EW-MAD
on each row from the targets that have arrived, and checks row 250 by
hand against numpy's weighted median of the targets of rows 0 to 245
alone. This code uses `df` and `ticks.parquet` from [Example
data](../../README.md#example-data):

```python
# Each target of ret arrives 5 rows late: the stream twice, a predict row at t and a learn row at t + 5.
doubled = po.stream.embargo(pl.scan_parquet("ticks.parquet"), clock="t", delay=5.0)
known = pl.when(pl.col("_online_role") == "learn").then(pl.col("ret").abs())   # a target counts once it has arrived
scaled = (
    doubled.online.with_windows(
        ret_scale=po.ewm_quantile(known, quantile=0.5, half_life=20.0) / 0.6744897501960817,
        clock="t", gap_cap=10.0,
    )
    .filter(pl.col("_online_role") == "predict")                               # back to one row per row
    .collect()
)
# Row 250's scale again by hand, from the targets of rows 0 to 245 alone.
arrived = df["ret"].abs().to_numpy()[:246]
weight = 0.5 ** ((250 - (np.arange(246) + 5)) / 20.0)                          # each weighed from its arrival
by_hand = np.quantile(arrived, 0.5, weights=weight, method="inverted_cdf") / 0.6744897501960817
got = scaled["ret_scale"][250]
print(f"row 250: ret_scale {got:.4f}, by hand from rows 0 to 245 {by_hand:.4f}, apart by {abs(got / by_hand - 1):.4f}")
print(f"null on the first {scaled['ret_scale'].null_count()} rows, before any target has arrived")
print("in the calm and the volatile stretches:", ", ".join(f"{scaled['ret_scale'][i]:.2f}" for i in (90, 190, 290, 390)))
```

```text
row 250: ret_scale 1.1409, by hand from rows 0 to 245 1.1362, apart by 0.0041
null on the first 5 rows, before any target has arrived
in the calm and the volatile stretches: 0.91, 2.30, 0.77, 1.66
```

`ret` is `±0.8·signal_a` plus noise of spread 0.5 on rows 0-99 and
200-299 and 2.0 on the others, so its spread is about 0.94 in the calm
stretches and 2.15 in the volatile ones, and the scale follows it.

## Use the scale as the row's weight

Give the model the scale as a weight column, and the delay the scale
assumed.

1. **Make the weight a column from the scale with Polars' `with_columns`,**
   such as `1 / scale²`.
2. **Name the column in the spec's `weight`.** A row whose weight is null
   is a [skipped row](../GLOSSARY.md#the-model-bank-and-what-it-reads):
   the model neither predicts it nor learns from it. So the first rows,
   before any target has arrived, are skipped.
3. **Give the spec the `embargo` the scale assumed,** so the model learns
   each row when its target arrives.

This code fits `ret` on `signal_a` with each row weighed by
`1 / ret_scale²`, and counts the rows skipped for a null weight. It uses
`scaled` from the code in [Leave out the targets not yet
known](#leave-out-the-targets-not-yet-known):

```python
weighted = scaled.with_columns(w=1.0 / pl.col("ret_scale") ** 2)            # 1. the weight, null where the scale is
by_scale = po.spec.ewridge("by_scale", targets=["ret"], features=["signal_a"], weight="w",
                           clock="t", gap_cap=10.0, half_life=100.0, embargo=5.0)   # 2 and 3
bank = po.ModelBank([by_scale])
bank.fit_predict(weighted)
print(f"rows skipped for a null weight: {bank.summary()['rows_skipped'][0]}")
```

```text
rows skipped for a null weight: 5
```

## Use a model's own scale of its residuals

**For the scale of a model's residual, use the scale `po.spec.huber` can
emit, with `emit_huber_scale=True`, in the field `huber_scale_<target>`.**
It is the model's own robust scale of its residuals, read before the row.
It takes each residual in when the row is learned, so the spec's
`embargo` holds it back as it holds back the fit. Each residual is
[out-of-sample](../GLOSSARY.md#fitting-a-model), except that a target's
first 500 learned rows enter it, once those rows are refit to convergence,
as residuals against that fit (`huber_refit_rows`; 10 rows a feature past
50 features, and no refit above 100). A row's own residual never enters
the scale it is emitted with. [The robust scale and the refit of the first
rows](../explanation/robust-scale.md) explains why the scale is robust
and why the first rows are refit.

This code fits `huber` on `y`, whose noise has a spread of 0.5, and weighs
each row of a second model by `1 / huber_scale_y²`. It uses
`ticks.parquet` from [Example data](../../README.md#example-data):

```python
# A huber fit that emits its robust scale of the residuals, read before each row.
robust = po.spec.huber("robust", targets=["y"], features=["x0", "x1"], clock="t", gap_cap=10.0,
                       half_life=100.0, emit_huber_scale=True)
residual_scale = pl.col("robust").struct.field("huber_scale_y")
rows = (pl.scan_parquet("ticks.parquet").online.fit_predict([robust])
          .with_columns(w=1.0 / residual_scale ** 2)               # the weight, null where the scale is
          .collect())
# A second model, each row weighed by the scale.
by_residual = po.spec.ewridge("by_residual", targets=["y"], features=["x0", "x1"], weight="w",
                              clock="t", gap_cap=10.0, half_life=100.0)
bank = po.ModelBank([by_residual])
bank.fit_predict(rows)
print(f"huber_scale_y on the last row: {rows['robust'].struct.field('huber_scale_y')[-1]:.3f}; the noise's spread is 0.5")
print(f"rows skipped for a null weight: {bank.summary()['rows_skipped'][0]}")
```

```text
huber_scale_y on the last row: 0.506; the noise's spread is 0.5
rows skipped for a null weight: 4
```

## Check the result

**Compare the scale on one row with numpy's weighted median of the
targets known by that row:** `np.quantile(x, 0.5, weights=w,
method="inverted_cdf")`, with each value weighed by
`0.5 ** (age / half_life)` from its arrival. The recipes of [Make a robust
scale with `po.ewm_quantile`](#make-a-robust-scale-with-poewm_quantile)
and [Leave out the targets not yet
known](#leave-out-the-targets-not-yet-known) do so. The two agree within the
0.78% within which `po.ewm_quantile` reads a quantile. Then check that the
scale is null until the first target arrives, and that `bank.summary()`
counts those rows in `rows_skipped`.
