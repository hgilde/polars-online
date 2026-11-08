<!-- kind: how-to -->
# Forget rows past a window

This how-to is for anyone whose model should count only the rows of a
recent span, such as a relation that may flip sign, and give every older
row no weight at all.

A [half-life](../GLOSSARY.md#time) never forgets entirely: three
half-lives back, a row still carries 12.5% of the weight. To make every
row `w` [clock units](../GLOSSARY.md#time) old or older count for exactly
nothing, give the [spec](../GLOSSARY.md#the-model-bank-and-what-it-reads)
`window_size=w`, a [hard window](../GLOSSARY.md#fitting-a-model). Five
models take it: `ewridge`, `lasso`, `ew_cov`, `ew_class` and `marginal`.
[A hard window](../explanation/time-and-decay.md#a-hard-window) explains
how the model cuts its running sums exactly, and shows a hard window forget
a change that a half-life still holds.

## Prerequisites

These steps need the library and a spec of one of the five models:

- polars-online installed, and the README's
  [Example data](../../README.md#example-data) built. The examples read
  `ticks.parquet`, 400 rows a unit of the clock `t` apart, in which the
  coefficient of `ret` on `signal_a` flips from 0.8 to −0.8 at row 200.
- A spec of `ewridge`, `lasso`, `ew_cov`, `ew_class` or `marginal`. With a
  [clock](../GLOSSARY.md#time), `window_size` is in its units: a number, or
  a duration such as `"1h"` on a clock of times. Without one, it counts
  rows.

## Give the spec a hard window

**Give a hard window where what the model fits can change, and rows from
before the change should count for nothing.** Two cases call for one:

- **Give a `marginal` screen a `window_size` when a relation may flip
  sign**, because two spans of opposite sign average to nothing over a
  long history. The hard window truncates every pair moment, so `corr`,
  `beta` and `t_stat` describe the rows inside it.
- **Give an `ew_class` `window_size` when the class means move,** because
  over a long history the spans before and after a move average
  together. Each class's moments
  then cover only the rows inside the hard window ([Give `window_size`
  when the classes
  move](choose-a-covariance-type.md#give-window_size-when-the-classes-move)
  shows it on `ew_class`).

1. **Give `window_size` the span to keep, in clock units,** as
   `window_size=100.0` for the last 100 units of `t`. The model then keeps
   a ring of snapshots of its [running sums](../GLOSSARY.md#fitting-a-model),
   one every row by default, to cut them at the hard window's edge.
2. **Choose `closed` for the row exactly `window_size` old,** as Polars'
   `rolling_*_by(closed=)` does. Under `"right"`, the default, that row
   has left, as it has in the [window
   operators](../../README.md#making-features-and-targets-from-windows)
   and in `rolling_sum_by`.
   Under `"both"` it stays. Either way, a row's weight is:

   ```
   weight(age) = 0.5 ** (age / half_life)   if age < window_size   (closed="right", the default)
                                            or age <= window_size  (closed="both")
               = 0                          otherwise
   ```

   A model's window always holds the row it ends
   at, because the model reads its fit after it has learned that row. So
   `"left"` and `"none"` are refused, and so is `"both"` without a
   `window_size`.
3. **For equal weights over the window, give
   `half_life=float("inf")` with the `window_size`.** Inside the window
   the weights still halve every `half_life`, so with a finite one the
   newest row counts most.
4. **To take fewer snapshots, space them on the clock with
   `snapshot_every`,** as `solve_every` spaces a regression's solves. It
   takes a number of clock units, or a duration such as `"1m"` on a clock
   of times. `max_rows_between_snapshots` caps the rows between them, and
   whichever comes first takes a snapshot. Under `snapshot_every` the hard
   window in effect spans `[window_size − snapshot_every, window_size]`,
   whatever the row rate, with `snapshot_every` the spacing in force. A
   `window_budget` that thins the ring doubles that spacing, as [Cap the
   snapshots' memory](#cap-the-snapshots-memory) explains.

This code screens `signal_a` and `signal_b` against `ret` over every row,
and over a hard window of the last 100. It uses `ticks.parquet` from
[Example data](../../README.md#example-data), where the planted
coefficient on `signal_a` is −0.8 over the last 200 rows:

```python
# The same screen over every row, and over a hard window of the last 100 units of t.
common = dict(targets=["ret"], features=["signal_a", "signal_b"], clock="t", gap_cap=300.0,
              half_life=float("inf"))                      # equal weights
whole = po.spec.marginal("whole", **common)
recent = po.spec.marginal("recent", window_size=100.0, **common)
bank = po.ModelBank([whole, recent])
bank.fit(pl.scan_parquet("ticks.parquet"))                  # learn, and keep only the state
# Each screen's beta and t_stat on signal_a, beside the planted -0.8.
for name in ("whole", "recent"):
    row = bank.marginal(name).filter(pl.col("feature") == "signal_a").row(0, named=True)
    print(f"{name:<6} beta {row['beta']:+.2f}, t_stat {row['t_stat']:+.1f}, weight_sum {row['weight_sum']:.0f} (planted beta -0.80)")
```

```text
whole  beta +0.10, t_stat +1.2, weight_sum 400 (planted beta -0.80)
recent beta -0.78, t_stat -3.7, weight_sum 100 (planted beta -0.80)
```

Over every row, the spans before and after the break average to a coefficient near 0 that a
screen would pass over. The hard window reads the new one.

## Know what each model keeps inside the window

What the window truncates depends on the model, and some refuse settings
beside `window_size`:

| model | inside the window | refused beside `window_size` |
|---|---|---|
| `ewridge` | the sums the fit is solved from, and `sigma` and `zscore` with them. The coefficients are the window's as of the last solve | `ridge_scale="sum"`, `session_shrink`, `gram_block_rows` |
| `lasso` | the same, and the error that selects the penalty. A feature that stays constant inside the window has no evidence there, and goes to exactly zero | |
| `ew_cov` | every moment, and `mahal`, `partial_corr` and the principal components read from them | `lags`, `mahal_quantiles` |
| `ew_class` | each class's moments, so the classifier follows class means that move | |
| `marginal` | every pair's weight, means and second moments, so `corr`, `beta` and `t_stat` describe the window. Its lag correlations, under `window_lags=True`, are the window's rows' own, about the window's means, the first `l` rows paired with the rows before the window | `bins` or `bin_edges`, `feature_moments="shared"`, and `lags` unless `window_lags=True` |

**To bin a `marginal`'s features beside a hard window, bin them in a second
spec without one,** since a hard window refuses `bins` and `bin_edges`.

**To keep a `marginal`'s `lags` under a hard window, give
`window_lags=True`,** whose docstring entry gives the memory its snapshots
take. `ew_cov` refuses `lags` beside `window_size` outright.

## Cap the snapshots' memory

**`window_budget` caps the snapshots' memory, per ring (one per spec and
group), in MiB.** The ring's memory grows with the window: for an `ew_cov`
over 20 columns, a 1,000-row window holds about 3 MB per group, divided by
the rows between snapshots.

1. **Give `window_budget` a policy and a cap,** as
   `window_budget={"thin": 64}`:

   | `window_budget` | when a chunk would take a ring past the cap |
   |---|---|
   | not given | the chunk is refused past 256 MiB |
   | `{"refuse": 64}` | the chunk is refused before any of it is learned, and the bank goes on as it was. The error names the ring's size, its snapshot cadence and the ways to lift the cap |
   | `{"thin": 64}` | the ring drops every other snapshot and doubles its spacing, on the clock or in rows, which, like `snapshot_every`, only ever shortens the window |
   | `{"refuse": float("inf")}` | nothing: no cap |

2. **After a refusal, increase the cap or give `{"thin": ...}`, and feed
   the chunk again.** A refusal comes before the bank learns any of the chunk,
   so the bank goes on as it was. Two overruns are found only as the rows
   go in. One is under `drift_action="reset"`. The other is a snapshot that
   grows within the chunk, as an `ewridge`'s or a `lasso`'s does when its
   targets first go missing on different rows. After either, the bank
   refuses every call that follows, so rebuild it from its last save with
   `po.ModelBank.load`.

## Check the result

**Fit the same spec without `window_size` on the rows inside the window
alone, and compare the two fits on the last row:** they agree when the
model takes a snapshot every row, the default, and no `window_budget`
thins them. Choose the rows inside the window by the clock, with
`pl.col("t") > last_t - window_size` under the default `closed="right"`.
This code compares `ewridge`'s coefficients under a flat hard window of
100 units of `t` with the same spec fitted to the rows of the last 100
units. It uses `ticks.parquet` from [Example
data](../../README.md#example-data):

```python
# A flat hard window of the last 100 units of t, and the same spec on those rows alone.
common = dict(targets=["ret"], features=["signal_a"], clock="t", gap_cap=300.0,
              half_life=float("inf"), coef_every=0)          # the coefficients on every row
windowed = po.spec.ewridge("windowed", window_size=100.0, **common)
alone = po.spec.ewridge("alone", **common)
in_window = pl.scan_parquet("ticks.parquet").online.fit_predict([windowed]).collect()
last_t = pl.scan_parquet("ticks.parquet").select(pl.col("t").max()).collect().item()
last_rows = pl.scan_parquet("ticks.parquet").filter(pl.col("t") > last_t - 100.0).online.fit_predict([alone]).collect()
# The last row's [intercept, signal_a] each way.
a = in_window["windowed"].struct.field("coef")[-1]
b = last_rows["alone"].struct.field("coef")[-1]
print(f"hard window: {a[1]:+.4f}; the last 100 units' rows alone: {b[1]:+.4f}")
print(f"the same within 1e-12: {(a - b).abs().max() < 1e-12}")
```

```text
hard window: -0.7815; the last 100 units' rows alone: -0.7815
the same within 1e-12: True
```

The two agree to rounding: the cut is a subtraction, whose precision falls
as the share of the weight it discards grows. Before you read the other
numbers of a model under a hard window, read the README's [five things to
know](../../README.md#a-hard-window) about them, such as `weight_sum`
being the weight inside the window.
