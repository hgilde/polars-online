<!-- kind: how-to -->
# Learn targets known late

This how-to is for anyone whose target is known only some time after its
row, such as the next five minutes' return. It shows how to predict each
row where it sits and learn from it only once its target is known.

Learning a row before its target is known lets the future into every
out-of-sample number after it ([A target that looks
ahead](../explanation/out-of-sample.md#a-target-that-looks-ahead) explains
why). Three ways hold each row back until its target would have arrived:

| the delay | use | where |
|---|---|---|
| the same for every row | `embargo`, the delay on the clock | the spec |
| a time of each row's own | `label_clock`, a column of the times | the spec |
| one delay, to be visible in the frame or for another engine to fit | `po.stream.embargo` | the query, before the bank |

## Prerequisites

These steps need the library and a stream with a clock:

- polars-online installed, and
  [Example data](../EXAMPLE-DATA.md) built. The examples read
  `df`, a frame of 400 rows a minute apart on the clock `ts`, and `lf`, a
  [query](../GLOSSARY.md#the-model-bank-and-what-it-reads) over the same
  rows saved as `ticks.parquet`, whose clock `t` counts them.
- A [spec](../GLOSSARY.md#the-model-bank-and-what-it-reads). With a
  [clock](../GLOSSARY.md#time) column, the delay is in the clock's units.
  Without one, it is in rows of the
  [group](../GLOSSARY.md#the-model-bank-and-what-it-reads).
- The delay after which each row's target is known, or a column of the
  time each one is known.

## Give every row the same delay

**Give the spec an [`embargo`](../GLOSSARY.md#time) when every target takes
the same time to arrive,** as the next five minutes' return does.
Everything computed from the targets then sees only those that had
arrived: the prediction, `sigma`, `zscore`, the metrics, break detection,
the conformal interval, `weight_sum` and the `min_weight` gate. A
classifier takes it the same way: [Give `embargo` when the classes arrive
late](choose-a-covariance-type.md#give-embargo-when-the-classes-arrive-late)
shows it on `ew_class`.

1. **Give the spec `embargo="5m"`,** or any finite delay above 0, in the
   clock's units. The bank predicts each row where it sits and learns from
   it five minutes later on the clock.
2. **To see the delay row by row, give the spec `emit_clocks=True`.** It
   writes `scored_clock`, the row's own clock, and `learned_clock`, the
   clock of the newest row the model had learned from when it predicted
   the row, both in the clock column's own type. With no clock they are
   the row's index in its group.
3. **Fit the bank** with `fit_predict` or a query's `online.fit_predict`.

This code fits `df`'s minute-apart rows under `embargo="5m"`, and prints
the clocks of five rows, counting from 0. Each row is predicted from the
rows at least five minutes older, the one exactly five minutes older
included, so the first five rows have learned nothing. It uses `df` from
[Example data](../EXAMPLE-DATA.md):

```python
fwd = po.spec.ewridge("fwd", targets=["y"], features=["x0", "x1"],
                      clock="ts", gap_cap="1h", half_life="30m",
                      embargo="5m",      # 1. a target takes five minutes to arrive: learned five minutes later
                      emit_clocks=True)  # 2. scored_clock and learned_clock on every row
rows = po.ModelBank([fwd]).fit_predict(df).unnest("fwd")   # 3. fit
for row in [0, 4, 5, 6, 300]:
    r = rows.row(row, named=True)
    learned = f"the rows up to {r['learned_clock']:%H:%M}" if r["learned_clock"] else "no row yet"
    print(f"row {row}: predicted at {r['scored_clock']:%H:%M}, having learned {learned}, weight_sum {r['weight_sum']:.2f}")
```

```text
row 0: predicted at 09:30, having learned no row yet, weight_sum 0.00
row 4: predicted at 09:34, having learned no row yet, weight_sum 0.00
row 5: predicted at 09:35, having learned the rows up to 09:30, weight_sum 1.00
row 6: predicted at 09:36, having learned the rows up to 09:31, weight_sum 1.98
row 300: predicted at 14:30, having learned the rows up to 14:25, weight_sum 43.74
```

**How `embargo` counts the delay, and what it does at the stream's
edges:**

| case | what `embargo` does |
|---|---|
| what the delay counts | the time that passed on the clock column, skipped rows included, and `session_gap` in place of the clock's jump at a session change. It is not the capped step the model decays by, so the release depends on the rows alone and survives any chunking |
| the step a row is predicted over | the clock from the newest row learned to the row, the held rows' steps included, capped at `gap_cap` as one gap. A `holt` prediction reaches that far ahead, and `kalman`'s `error_inflation` adds that much drift. `predict` against a bank holding rows reads the same clock. Either way the prediction is the one a bank fed only the learned rows gives |
| exactly at the delay | the row is learned. The time that passed is held exactly: in integer nanoseconds on a temporal clock, and by one subtraction of the two rows' values on a number clock, in integers on an integer clock. Rows 1 ms apart under `embargo="2s"` are each learned 2,000 rows on |
| with no clock column | one unit is one row of the group, a skipped row included: `embargo=20` is twenty rows |
| at a clock break | nothing is released early. A clock break's own effect, such as a lag ring's clearing or `session_shrink`'s blend, waits with the row after it, and runs when that row is learned. A reset drops the rows still waiting |
| at the end of the stream | the rows still waiting are not learned, unless the state is saved and another run resumes from it |
| in the state | the waiting rows are saved with it: one row's values for each row inside the delay, in each group |
| row by row | `emit_clocks=True`, on any model, writes `scored_clock` and `learned_clock` |

## Give each row its own target time

**When targets arrive at times of their own, name a column of those
times in the spec's `label_clock=`,** of the same type as the clock. A flight's
delay is known when it lands, so a spec of flights takes
`label_clock="landed"`. `embargo="5m"` is the
case where every such time is the clock plus five minutes. The two give
the same numbers wherever the clock adds up exactly, as a temporal clock
always does.

1. **Make the column of the times each target is known,** with Polars'
   `with_columns`, before the bank or when you write the file.
2. **Name it in the spec's `label_clock=`.** The bank predicts each row
   where it sits, holds it, and learns it at its `label_clock` time, at the
   first row whose clock has reached it. Its target is then compared with
   the prediction, the interval and the spread the row was given, so every
   residual diagnostic describes what was shown.
3. **Fit the bank.**

This code makes up flights, writes them to a file and fits them from it.
It builds its own rows:

```python
from datetime import datetime, timedelta

import numpy as np
import polars as pl

import polars_online as po

# 1. Make up flights, one every ten minutes, whose delay is known only when each one lands.
rng = np.random.default_rng(3)
n = 20_000
distance = rng.uniform(200.0, 2_000.0, n)                            # miles
start = datetime(2024, 1, 1)
pl.DataFrame({
    "departed": pl.datetime_range(start, start + timedelta(minutes=10 * (n - 1)), "10m", eager=True),
    "distance": distance,
    "delay": 0.01 * distance + 10.0 * rng.standard_t(3, n),         # minutes late
}).with_columns(
    landed=pl.col("departed") + pl.duration(minutes=pl.col("distance") / 8.0 + 20.0),  # when the delay is known
).write_parquet("flights.parquet")

# 2. Fit them from the file, each delay learned at its landing.
late = po.spec.ewridge(
    "late", targets=["delay"], features=["distance"],
    clock="departed", gap_cap="1h", half_life="2d",
    label_clock="landed",   # learned at landing; each interval compared with the one given
    conformal=0.9,          # lo_delay and hi_delay: an interval meant to hold 90% of delays
)
rows = pl.scan_parquet("flights.parquet").online.fit_predict([late]).unnest("late").collect()   # 3. fit
# Count the intervals that held the delay each flight was given.
given = rows.filter(pl.col("lo_delay").is_not_null())
covered = given.select(((pl.col("delay") - pl.col("pred_delay")).abs() <= pl.col("hi_delay") - pl.col("pred_delay")).mean()).item()
print(f"{given.height} intervals given, {covered:.3f} of them held the delay that landed,",
      f"half-width {given.select((pl.col('hi_delay') - pl.col('pred_delay')).mean()).item():.1f} minutes")
```

```text
19975 intervals given, 0.900 of them held the delay that landed, half-width 23.7 minutes
```

**How `label_clock` orders the rows it holds, and what it refuses:**

| case | what `label_clock` does |
|---|---|
| the order of learning | by `label_clock` time, so a short flight landing first is learned before an earlier long one; rows with one such time in the order they arrived. Each is learned at its time: its step is the one from the time learned before it |
| `resid` on a row | the target less the prediction the row was given |
| a weight of 0, or a null `label_clock` time | the row is predicted and never held. A null is a target that never arrives: `bank.summary()` counts it in `labels_missing` |
| a `label_clock` time before its row's clock | refused, naming the row: the column does not hold the times the targets are known |
| how many rows wait | `max_held_rows` bounds them per group. Past it the chunk is refused, or with `held_overflow="drop_oldest"` the row that arrived first is dropped, counted in `labels_dropped` |
| at a reset, and at the end of the stream | a reset drops the rows still waiting, counted in `labels_dropped`. Rows waiting at the end are not learned, unless the state is saved and another run resumes from it |
| what it refuses | `embargo` beside it, a [formula target](../GLOSSARY.md#fitting-a-model), `group_close`, a `session` column and `drift_action="reset"` (resetting the model when it flags [drift](detect-a-break.md#act-on-a-break)); and `holt`, `kalman`, `hmm` and `micro`, whose prediction reaches from the newest row learned, which `label_clock` moves to a `label_clock` time |
| cost | each held row goes into the waiting rows in `label_clock` order: 0.006 microseconds a row with about 100 waiting, 0.38 with about 800 (Apple M4 Pro) |

**With hundreds of targets in flight, set `conformal_rate="auto"`.** Each
row is compared with the interval it was shown, so misses arrive in bursts
and the default step of 0.05 overshoots. `"auto"` shrinks the step as more
rows wait for their targets, counting the rows waiting as the stream runs,
and is 0.05 when nothing waits. On New York's 2013 flights under an 8-hour `embargo`,
with 338 rows waiting, 0.05 held 90% of delays at a half-width of 203
minutes, and `"auto"` held 90% at 48. To set the rate by hand for a 90%
interval with `n` rows waiting, use `0.05 / (1 + n / 48)`.
[DIAGNOSTICS.md](../DIAGNOSTICS.md#conformal-intervals) has the
measurement.

## Write the delay out as rows

**Use
[`po.stream.embargo`](https://hgilde.github.io/polars-online/stream.html#polars_online.stream.embargo)
when the delay has to be visible in the frame, or when another engine fits
the model.** It returns every row twice: a copy to predict at `t` with
weight 0, and a copy to learn from at `t + delay`.

1. **Sort the input by the clock across all rows, as
   `lf.sort("t", maintain_order=True)`, before the call.** Order within
   each group is not enough, since the call merges the two copies by the
   clock alone.
2. **Call `po.stream.embargo` with the clock and the delay.** It adds two
   columns: `_online_role`, which says which copy a row is, and
   `_online_role_weight`, 0 on the copies to predict and 1 on the rest.
3. **Pass `_online_role_weight` to the spec's `weight`.**
4. **Keep the copies whose `_online_role` is `"predict"`** after the bank.

**The built-in delay agrees with the doubled stream to the bit on `pred`,
`resid` and `weight_sum` when each row's `t + delay` is another row's
clock.** An evenly spaced clock with a delay of whole steps is such a case.
On an irregular clock they differ, since the doubled stream decays a target
from `t + delay` and the built-in delay learns it at the row that releases
it. A model whose prediction reads the clock since its last learned row
differs on any clock: `holt`, `hmm`, `micro`, `kalman`'s
`error_inflation`, and `sgd` and `pa` while their scaler warms up. The
doubled stream's last learned row sits `delay` after its row, so such a
prediction reads `delay` less clock.

**Every residual diagnostic differs in any case**, from `sigma` and
`zscore` to the conformal band and drift. `embargo` takes in the residual
of the prediction the row was given. The doubled stream's learning copy
forms its residual at `t + delay`, from a model that has since learned
every row before it.

This code doubles `lf`, whose clock `t` moves a row at a time, and holds
the predicted copies to the built-in delay of the same five units. It uses
`lf` from [Example data](../EXAMPLE-DATA.md):

```python
# 1. lf is already in clock order across all rows. 2. Every row twice, in clock order.
doubled = po.stream.embargo(lf, clock="t", delay=5.0)
predicted = doubled.online.fit_predict(
    [po.spec.ewridge("m", targets=["y"], features=["x0"], clock="t", gap_cap=10.0,
                     half_life=50.0, weight="_online_role_weight")]   # 3. 0 on the copies to predict, 1 on the rest
).filter(pl.col("_online_role") == "predict").collect()               # 4. keep the predicted copies

# The built-in delay on the same rows: t is a row apart, so each t + 5 is another row's clock.
built_in = lf.online.fit_predict(
    [po.spec.ewridge("m", targets=["y"], features=["x0"], clock="t", gap_cap=10.0, half_life=50.0, embargo=5.0)]
).collect()
for field in ["pred_y", "weight_sum"]:
    a, b = predicted["m"].struct.field(field), built_in["m"].struct.field(field)
    print(f"{field}: {a.count()} rows, {(a == b).sum()} equal to the bit")
```

```text
pred_y: 393 rows, 393 equal to the bit
weight_sum: 400 rows, 400 equal to the bit
```

## Check that it worked

**Under `embargo`, check that `scored_clock` less `learned_clock` is at
least the delay on every row.** `learned_clock` is null until the model
has learned a row. This code fits `fwd`, the spec of [Give every row the
same delay](#give-every-row-the-same-delay), on `df` again, and its delay
is five minutes. It uses `df` from [Example
data](../EXAMPLE-DATA.md):

```python
rows = po.ModelBank([fwd]).fit_predict(df).unnest("fwd")
gap = rows.select((pl.col("scored_clock") - pl.col("learned_clock")).min()).item()
print(f"the shortest time from the newest row learned to the row predicted: {gap}")
```

```text
the shortest time from the newest row learned to the row predicted: 0:05:00
```

**Under `label_clock`, read `bank.summary()`'s `labels_missing` and
`labels_dropped`.** A query keeps no bank, so fit the spec with one first:
`bank = po.ModelBank([late])`, then
`bank.fit(pl.scan_parquet("flights.parquet"))`. The two columns count the rows whose target never arrived and the
rows dropped while they waited, which the model never learned.
