<!-- kind: tutorial -->
# Follow a local fit through a day

This tutorial is for anyone new to polars-online who wants a regression
that follows the most recent rows of a
[stream](../GLOSSARY.md#the-model-bank-and-what-it-reads) (the rows a
[model bank](../GLOSSARY.md#the-model-bank-and-what-it-reads) reads, in
the order it reads them). You fit a regression that forgets old rows along
the timestamps, and watch it recover a coefficient you planted, which
drifts through a trading day and changes overnight.

To run the code, install the library and numpy with
`pip install "polars-online[numpy]"`. Run each step's code in the same
Python process, in a directory where it may write files.

A [spec](../GLOSSARY.md#the-model-bank-and-what-it-reads) describes one
model: which model it is, which columns it reads, and how it treats time.
A [model bank](../GLOSSARY.md#the-model-bank-and-what-it-reads) fits a
model for each spec in one pass over the rows. A
[query](../GLOSSARY.md#the-model-bank-and-what-it-reads) is a chain of
Polars steps that runs only when you ask for its result.

## 1. Make up two trading days

Make up two days of one stock's returns, `ret`, a row a second from 9:30
to 16:00, with a feature `flow`. The coefficient of `ret` on `flow` is the
answer to recover, and the column `true_coef` keeps it on every row. On
the first day it rises steadily from 0.2 to 0.8. Overnight it changes, and
the second day it stays at −0.4. Save the rows to a parquet file:

```python
from datetime import datetime, time

import numpy as np
import polars as pl

import polars_online as po

# Two trading days, 9:30 to 16:00, a row a second.
rng = np.random.default_rng(1)
days = []
for day, (opening, closing) in [(2, (0.2, 0.8)), (3, (-0.4, -0.4))]:   # the coefficient at each day's open and close
    ts = pl.datetime_range(datetime(2024, 1, day, 9, 30), datetime(2024, 1, day, 16), "1s",
                           closed="left", eager=True)
    flow = rng.standard_normal(len(ts))
    true_coef = np.linspace(opening, closing, len(ts))
    days.append(pl.DataFrame({
        "ts": ts,
        "day": ts.dt.date(),                       # the trading day: it changes at the second day's open
        "flow": flow,
        "true_coef": true_coef,                    # the planted coefficient, to compare with the fit
        "ret": true_coef * flow + rng.standard_normal(len(ts)),
    }))
pl.concat(days).write_parquet("two_days.parquet")

# How many rows each day has, and its first and last timestamps.
spans = pl.scan_parquet("two_days.parquet").group_by("day").agg(
    rows=pl.len(), first=pl.col("ts").min(), last=pl.col("ts").max()
).sort("day").collect()
for day, rows, first, last in spans.iter_rows():
    print(f"{day}: {rows} rows, {first:%H:%M:%S} to {last:%H:%M:%S}")
```

```text
2024-01-02: 23400 rows, 09:30:00 to 15:59:59
2024-01-03: 23400 rows, 09:30:00 to 15:59:59
```

## 2. Fit a regression that follows the recent rows

`po.spec.ewridge` builds the spec of a ridge regression whose rows are
weighted by their age. Give it a [clock](../GLOSSARY.md#time) (the
timestamp column `ts`) and `half_life="10m"`, so each row's weight halves
for every ten minutes of `ts` after it. The fit is then local: it is made
mostly from the last few tens of minutes, and it moves from row to row.

`gap_cap` is required with a clock. With `gap_cap="5m"`, a gap in `ts`
longer than five minutes, such as a night, decays the rows the model has
learned as a five-minute step would.

`coef_every=0` writes, on every row, the coefficients the model holds
after that row, from its latest solve.

Run the spec over both days in a query. The bank adds one [output
column](../GLOSSARY.md#what-a-bank-writes), named after the spec, whose
value in each row is a record of named
[fields](../GLOSSARY.md#what-a-bank-writes), and `.online.unnest` turns
each field into a column. Read the spec's coefficient on `flow` at half
past each hour of the first day, from 10:30:

```python
local = po.spec.ewridge(
    "local", targets=["ret"], features=["flow"],
    clock="ts",                                    # the clock
    half_life="10m",                               # halves every 10m
    gap_cap="5m",                                  # caps a gap at 5m
    coef_every=0,                                  # every row
)
fitted = (
    pl.scan_parquet("two_days.parquet")            # a query over the file
    .online.fit_predict([local])                   # adds the bank's output column, "local"
    .online.unnest([local])                        # one column per field: the coefficient on flow is coef_ret_flow
    .collect()                                     # runs the query
)

# The first day, at 10:30 and each hour after it.
first_day = fitted.filter(pl.col("day") == pl.date(2024, 1, 2), pl.col("ts").dt.time().is_in(
    [time(hour, 30) for hour in range(10, 16)]))
for ts, true, fit, weight in first_day.select("ts", "true_coef", "coef_ret_flow", "weight_sum").iter_rows():
    print(f"{ts:%H:%M}   true {true:5.2f}   fitted {fit:5.2f}   weight_sum {weight:4.0f}")
```

```text
10:30   true  0.29   fitted  0.29   weight_sum  853
11:30   true  0.38   fitted  0.40   weight_sum  866
12:30   true  0.48   fitted  0.47   weight_sum  866
13:30   true  0.57   fitted  0.53   weight_sum  866
14:30   true  0.66   fitted  0.64   weight_sum  866
15:30   true  0.75   fitted  0.73   weight_sum  866
```

The fitted coefficient climbs with the true one through the day.
`weight_sum`, the weight of the rows behind each row's prediction, levels
off at about 866 rows' worth, since the older rows count for less and
less.

## 3. Read the fit at the next day's open

Read the same output at the second day's open, in its first half hour:

```python
# The second day's first half hour.
open_times = [time(9, 30), time(9, 31), time(9, 35), time(9, 40), time(9, 50), time(10, 0)]
second_open = fitted.filter(pl.col("day") == pl.date(2024, 1, 3), pl.col("ts").dt.time().is_in(open_times))
for ts, true, fit, weight in second_open.select("ts", "true_coef", "coef_ret_flow", "weight_sum").iter_rows():
    print(f"{ts:%H:%M}   true {true:5.2f}   fitted {fit:5.2f}   weight_sum {weight:4.0f}")
```

```text
09:30   true -0.40   fitted  0.80   weight_sum  866
09:31   true -0.40   fitted  0.70   weight_sum  630
09:35   true -0.40   fitted  0.36   weight_sum  687
09:40   true -0.40   fitted  0.15   weight_sum  740
09:50   true -0.40   fitted -0.13   weight_sum  803
10:00   true -0.40   fitted -0.29   weight_sum  834
```

The fit opens the second day at 0.80, near where the first day's fit
ended, and is still at −0.29 half an hour in. Under `gap_cap`, the night
decayed the first day's rows by only five minutes, so they still carry
weight into the new day and hold the fit above −0.4 through its first half
hour. `weight_sum` falls from 866 to 630 by 9:31, as the night's five
minutes and the first minute decay it.

## 4. Start the fit over at each day's open

Tell the spec where each day starts. Give it a
[session](../GLOSSARY.md#time) (`session="day"`, a column whose value
changes at a boundary) and `session_gap="reset"`. The model then starts
over with nothing learned wherever `day` changes. Fit it, and read the
second day's open beside the fit from step 2:

```python
by_day = po.spec.ewridge(
    "by_day", targets=["ret"], features=["flow"],
    clock="ts", half_life="10m", gap_cap="5m", coef_every=0,
    session="day",                                 # a new value of day is a new session
    session_gap="reset",                           # the model starts over at each new session
)
by_day_fitted = pl.scan_parquet("two_days.parquet").online.fit_predict([by_day]).online.unnest([by_day]).collect()   # as in step 2

# The second day's first half hour, from both fits.
by_day_open = by_day_fitted.filter(pl.col("ts").is_in(second_open["ts"].implode()))
for ts, true, carried, reset, weight in zip(second_open["ts"], second_open["true_coef"], second_open["coef_ret_flow"],
                                            by_day_open["coef_ret_flow"], by_day_open["weight_sum"]):
    reset_text = "null" if reset is None else f"{reset:5.2f}"   # null on the first row after the reset
    print(f"{ts:%H:%M}   true {true:5.2f}   step 2 {carried:5.2f}   reset {reset_text:>5}   reset weight_sum {weight:4.0f}")
```

```text
09:30   true -0.40   step 2  0.80   reset  null   reset weight_sum    0
09:31   true -0.40   step 2  0.70   reset -0.45   reset weight_sum   58
09:35   true -0.40   step 2  0.36   reset -0.41   reset weight_sum  254
09:40   true -0.40   step 2  0.15   reset -0.32   reset weight_sum  433
09:50   true -0.40   step 2 -0.13   reset -0.32   reset weight_sum  650
10:00   true -0.40   step 2 -0.29   reset -0.40   reset weight_sum  758
```

The reset fit forgets the first day at 9:30, where `weight_sum` falls to
0. Its coefficient is null on that row, since the model has learned only
that one row, and at each time in the table from 9:31 on, it is within 0.1
of −0.4. The fit from step 2 is still above 0 ten minutes into the day.
The reset fit's estimates are noisier in the first minutes, since they
rest on fewer rows. Its `weight_sum` is 58 at 9:31, against 630 for the
step 2 fit in step 3's table.

## Where to go next

To take the local fit to your own stream, start from the task that matches
it:

| to | read |
|---|---|
| treat a session boundary as a short step in place of a reset | [Handle sessions, gaps and steps back](../how-to/handle-sessions-gaps-and-steps-back.md) |
| give each day a model of its own, written out when the day ends | [Close groups and read each one's row](../how-to/close-groups.md) |
| choose the half-life | [Choose a ridge and a half-life](../how-to/choose-a-ridge-and-half-life.md) |
| keep each stock's rows in clock order, and read the warning when a query may not | [Put rows in clock order](../how-to/put-rows-in-clock-order.md) |
| fit a separate model for each stock | [Fit one model per group](../how-to/fit-one-model-per-group.md) |
| know how a model forgets along its clock, and what `gap_cap` and a session do to the decay | [Time and decay](../explanation/time-and-decay.md) |
| know why a model withholds its predictions on its first rows after a reset | [Warm-up and readiness](../explanation/warm-up-and-readiness.md) |
| fit a model over parquet files too large to hold | [Fit a model bank over parquet files in a query](stream-from-parquet.md) |
