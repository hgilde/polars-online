<!-- kind: how-to -->
# Handle sessions, gaps and steps back

This how-to is for anyone who fits a model with a clock on a stream from a
market, or on any stream whose clock has session boundaries, quiet spells
or replays, and who wants to choose what each of them does to the fit.

## Prerequisites

These steps need the library and a stream with a clock:

- polars-online installed, and
  [Example data](../EXAMPLE-DATA.md) built. The examples read
  `df`, a frame of 400 rows a minute apart, on the timestamp column `ts`.
  Its column `session` is `"m"` on the first 200 rows (a morning session)
  and `"a"` on the last 200 (an afternoon one).
- A spec with a clock: a `clock=` column, a `half_life=`, and the
  `gap_cap=` a clock requires. On a temporal clock such as `ts`, each is a
  duration, as `half_life="30m"`. A numeric clock takes the same parameters
  as numbers of its units, such as `half_life=600.0, gap_cap=300.0` on `t`,
  and `lam`, the weight kept per unit, may stand in for `half_life`.

## Set the clock's policy

A stream from a market has session boundaries, gaps and steps back in its
[clock](../GLOSSARY.md#time). The
[spec](../GLOSSARY.md#the-model-bank-and-what-it-reads) has a parameter for
each, and together they are the clock's *policy*:

| the stream has | the parameter | what it does |
|---|---|---|
| a **session** boundary, such as the start of a trading day, where the clock's jump is not elapsed time | `session`, a column whose value changes at a session boundary, and `session_gap`, required with it unless `group_close="session"` closes the group at the change instead ([Close groups](close-groups.md)) | applies a step you choose, `session_gap`, in place of the jump, capped at `gap_cap` like any step. `session_gap="reset"` starts the model over instead |
| a **gap** in the clock, such as an hour with no rows | `gap_cap`, required with a clock, finite and above 0 | caps each step between two rows a model learns from, so a quiet hour ages the fit no more than `gap_cap` would. After a run of skipped rows, the next row's step is capped too, however long the run |
| a clock that **steps back**, such as a replayed day | `restart_after_step_back` | starts the model over at a step back larger than the setting, and refuses one no larger as a late row. Unset, the default, every step back is refused ([Choose what a step back means](#choose-what-a-step-back-means)) |

**A stream has a [clock break](../GLOSSARY.md#time) wherever `gap_cap`
shortened a step, and wherever the session changes.** A model that keeps
past rows by position, as a lag does, empties them at a clock break, where
the previous row no longer stands for the moment just before. [Gaps,
sessions and steps
back](../explanation/time-and-decay.md#gaps-sessions-and-steps-back)
explains why the clock needs a policy at all.

Choose a step for each kind of jump in the clock, and read the result in
`weight_sum`, the weight behind each row's prediction.

1. **Set `gap_cap` to the longest step a quiet spell should age the fit
   by,** as `gap_cap="5m"`. A gap longer than that ages the fit as
   `gap_cap` does.
2. **If the stream has session boundaries, name the column that marks them
   in `session=`, and the step to take there in `session_gap=`,** as
   `session="session", session_gap="1m"`. Give `session_gap="reset"` to
   start the model over at each boundary instead.
3. **If the clock can step back, choose `restart_after_step_back`** as
   [Choose what a step back means](#choose-what-a-step-back-means) says.
   Otherwise, leave it unset, and the bank refuses every step back.
4. **Fit the stream, and read `weight_sum` on the rows around a
   boundary.**

To show each policy, this code moves the afternoon session of `df` two days
on, which puts a weekend between rows 199 and 200 (counting from 0), and
fits one regression under four policies:

| spec | its policy | the weekend counts as |
|---|---|---|
| `capped` | `gap_cap="5m"` | five minutes: the fit ages a little |
| `uncapped` | `gap_cap="3d"` | two days: about a hundred half-lives empty the fit |
| `one_minute` | `session="session"`, `session_gap="1m"` | one minute, the step chosen for the session boundary |
| `reset` | `session="session"`, `session_gap="reset"` | a reset: the model starts over with nothing learned |

This code uses `df` from [Example data](../EXAMPLE-DATA.md):

```python
# df with a weekend between its two sessions, saved.
afternoon = pl.col("session") == "a"
df.with_columns(ts=pl.when(afternoon).then(pl.col("ts") + pl.duration(days=2)).otherwise(pl.col("ts"))) \
  .write_parquet("weekend.parquet")

# Four ways to treat the weekend.
common = dict(targets=["y"], features=["x0", "x1"], clock="ts", half_life="30m")   # ts is a Datetime: durations
policies = [
    po.spec.ewridge("capped", gap_cap="5m", **common),                             # the weekend counts as five minutes
    po.spec.ewridge("uncapped", gap_cap="3d", **common),                           # the weekend counts in full
    po.spec.ewridge("one_minute", gap_cap="5m", session="session", session_gap="1m", **common),   # one minute at the boundary
    po.spec.ewridge("reset", gap_cap="5m", session="session", session_gap="reset", **common),  # a reset there
]
# Each policy's weight_sum around the boundary at row 200.
fitted = pl.scan_parquet("weekend.parquet").online.fit_predict(policies).collect()
print("row" + "".join(f"{p['name']:>12}" for p in policies))
for row in [199, 200, 201, 202]:
    print(f"{row:<3}" + "".join(f"{fitted[p['name']].struct.field('weight_sum')[row]:>12.2f}" for p in policies))
```

```text
row      capped    uncapped  one_minute       reset
199       43.34       43.34       43.34       43.34
200       43.35       43.35       43.35        0.00
201       39.62        1.00       43.36        1.00
202       39.72        1.98       43.37        1.98
```

## Check the result

**Each policy shows in `weight_sum`, read on the rows just after the
boundary.** `weight_sum` is read before a row's own step, so a reset shows
on row 200 itself, and a step's decay on row 201. In the output of [Set
the clock's policy](#set-the-clocks-policy):

- `capped` loses a little weight at row 201: the weekend aged the fit by
  five minutes.
- `uncapped` falls to 1.00 at row 201: two days emptied the fit, and only
  row 200 counts.
- `one_minute` barely moves: the boundary aged the fit by one minute.
- `reset` is 0.00 at row 200: the model started over there.

## Choose what a step back means

**`restart_after_step_back` and the size of a step back decide what it
means.** It is unset by default, since only the caller knows how late a
row can be, and `0` resets the model at every step back:

| a step back | is read as | and the bank |
|---|---|---|
| any size, with `restart_after_step_back` unset, the default | rows not in clock order | refuses the chunk |
| no larger than `restart_after_step_back` | a late row, such as a transposed pair or a row a minute late | refuses the chunk |
| larger than `restart_after_step_back` | a new start, such as a replayed day or a feed that started again | resets the model |
| any size, on a row whose `session` value changes | a new session, where the step is no measure of time | applies `session_gap`, or resets the model under `session_gap="reset"` |

1. **Set `restart_after_step_back` above the largest step back a late row
   can make, and below the step a new start makes,** as
   `restart_after_step_back=1.0` on a clock whose replays step back by more
   than 1.
2. **Fit the stream, and count the resets with `bank.summary()`**, whose
   `clock_backwards` counts the steps back and `resets` the resets.

[Put rows in clock order](put-rows-in-clock-order.md#fix-a-chunk-the-bank-refused)
runs a replayed day under `restart_after_step_back=1.0` and prints those
two counts. It also shows the error a refused chunk raises, and how to
choose a fix by what caused the step back.
