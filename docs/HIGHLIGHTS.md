<!-- kind: how-to -->
# What you can do with polars-online

Each task runs on data it makes up with a known answer, and prints what
the library found, to read against that answer.

| task | what it shows |
|---|---|
| [Automate a round of model selection](#automate-a-round-of-model-selection) | exponentially weighted ridge regressions over many combinations of factors, targets and [half-lives](GLOSSARY.md#time) in time and in traded volume, in one pass in bounded memory, so the files may be larger than memory, that writes each fit's [out-of-sample](GLOSSARY.md#fitting-a-model) residuals and coefficients, the combination chosen as best on each trade, and markers where the factors' regime changed, to read how stable each model is |
| [Build a forward-looking signal and resume it](#build-a-forward-looking-signal-and-resume-it) | a target built as the data streams, a bounded reverse EWMA of the mid, decaying forward over the next five minutes, minus the mid, a feature of the mid minus its EWMA over the last five, and a first-stage ridge's out-of-sample residual as a feature of the second stage, learned under an [embargo](GLOSSARY.md#time) so no prediction has learned from a target still being formed, fitted robustly through jumps, and saved partway and resumed in a fresh process exactly where it left off |
| [Prediction intervals on late-arriving residuals](#prediction-intervals-on-late-arriving-residuals) | a model whose target is known about 30 seconds after it was predicted, with a 90% prediction interval around each forecast that still covers nine answers in ten, and flags where a coefficient of the regression and the correlation between the features significantly change, each checked against changes planted at known rows |
| [Measure correlations on tick data](#measure-correlations-on-tick-data) | correlations from series that trade at their own times, measured on the times by which every series has traded again, where sampling them on a fixed grid all but loses them |
| [Capabilities](#capabilities) | each capability the tasks use, with the step that shows it and a guide that covers it |

Every task reads its rows from parquet files, most often as a
[query](GLOSSARY.md#the-model-bank-and-what-it-reads) (a chain of Polars
steps that runs only when you ask for its result). It runs the rows
through a [model bank](GLOSSARY.md#the-model-bank-and-what-it-reads): a
set of models, one for each
[spec](GLOSSARY.md#the-model-bank-and-what-it-reads) (the description of
one model), fitted together in one pass over the rows. Every model
predicts each row before it learns from it, so every prediction and
residual on this page, and every test read from them, is
[out-of-sample](GLOSSARY.md#fitting-a-model).

## Prerequisites

Each task builds its own data, so the tasks need only:

- polars-online installed with numpy, as
  `pip install "polars-online[numpy]"`.

## Automate a round of model selection

Fit exponentially weighted ridge regressions over many combinations of
factors, targets and half-lives, measured both in time and in traded
volume, beside a detector of regime changes in the factors. All of it
runs in one pass over the files, in bounded memory so the files may be
larger than memory. The pass writes one output: each fit's out-of-sample
residuals and coefficients, the combination chosen as best
on each trade (always from the trades before it), and markers where the
regime changed. Read from it how stable each model is, how often each
combination was chosen, and where the regimes changed:

- **Many combinations of factors and targets, fitted at once**: five
  named sets drawn from ten factors, for two targets, at two half-lives on
  each of two clocks. That is 20 [grid points](GLOSSARY.md#grids)
  (combinations of settings), each fitted to both targets, from one read
  of the rows.
- **Local fits on half-lives**, where each row's weight halves every
  [half-life](GLOSSARY.md#time) on a [clock](GLOSSARY.md#time): two
  half-lives measured in time (30 minutes and 2 hours) and two measured in
  traded volume (200,000 and 800,000 shares).
- **Out-of-sample selection**: the grid point chosen for each target on
  each trade is the one whose out-of-sample error, exponentially weighted
  at its own half-life, was lowest on the trades before it.
- **Changes of coefficients, selection and regime, detected in the same
  pass**: a test for a break in the coefficients, the grid point chosen on
  each trade, and the age of the factors' regime.
- All in one pass over sorted data too large for RAM, split by date
  across files with session breaks between them. Every
  [slot](GLOSSARY.md#grids) (each prediction a spec writes) is then scored
  by **one call over a query of the written output**, which adds up a few
  sums per slot a chunk at a time, so the output is never held in memory
  whole.

The recipe plants each answer it looks for. The return `ret_a` loads 0.5
on factor `f0` throughout, and on `f1` from 0 to 0.6 at the open of
2024-01-04. The return `ret_b` loads 0.4 on `f3` and −0.3 on `f4`
throughout. The other six factors play no part, and every factor triples
in size at noon on 2024-01-03.

1. **Make up four days of trades, one parquet file per day,** each day a
   [session](GLOSSARY.md#time) (the column `day`), with ten factors, two
   returns, and the cumulative volume of the day beside the timestamps.
2. **Give the grid to two specs, one for each clock.** A dict of named
   column lists in `feature_sets`, beside a list of half-lives in
   `half_life` and two targets in `targets`, makes a
   [grid](GLOSSARY.md#grids): every combination is fitted from the same
   rows. `emit_selected=True` adds the grid point chosen for each target
   on each row. On the time clock, `session="day"` with
   `session_gap="1m"` ages each fit by one minute at a new day, in place
   of the night. The volume spec's clock is the day's cumulative volume,
   so its half-life and its `gap_cap` (the longest step a fit ages by) are
   in shares, and `session_gap` is the step it takes at a new day, where
   the volume starts again from zero.
3. **Add the watchers:** a ridge with `emit_breaks=True`, whose
   `break_wald_<target>` tests whether a target's coefficients have moved,
   and a `bocpd` spec, which tracks how long the factors' current regime
   has lasted.
4. **Run one query over every file,** `pl.scan_parquet("days/*.parquet")`
   into `.online.fit_predict`, and write the bank's output, with the
   readings to count, to a file with `sink_parquet`, which runs the query
   and writes the result without holding it all in memory.
5. **Count the choices, the break flags and the new regimes by day,** with
   queries over that file.
6. **Score every slot of both specs** with `po.eval.compare_specs` over a
   query of that file. It runs the query once and reads its result a chunk
   at a time, adding up a few sums per slot, so it holds one chunk of the
   file and the sums. Polars holds what it reads of the file ahead of them,
   which [Cut what Polars reads ahead](how-to/bound-memory.md#cut-what-polars-reads-ahead)
   bounds.

```python
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import polars_online as po

# 1. Four days of trades, one file per day, with the answers planted.
rng = np.random.default_rng(1)
Path("days").mkdir(exist_ok=True)
n = 5_000                                                          # trades a day
for k in range(4):
    # A day of trades at uneven times, 9:30 to 16:00, and the shares each one traded.
    day = date(2024, 1, 2 + k)
    ts = [datetime(2024, 1, 2 + k, 9, 30) + timedelta(seconds=float(s))
          for s in np.cumsum(rng.exponential(23_400 / n, n))]
    volume = np.round(rng.lognormal(6.0, 0.8, n))                  # shares per trade
    # Ten factors, three times as large from noon on 2024-01-03, and two returns on them.
    calm = (k == 0) | ((k == 1) & (np.array(ts) < datetime(2024, 1, 3, 12)))   # True on day one, and on day two before noon
    f = rng.standard_normal((10, n)) * np.where(calm, 1.0, 3.0)    # f[i] is factor f{i}'s value on each trade
    ret_a = 0.5 * f[0] + (0.6 if k >= 2 else 0.0) * f[1] + rng.standard_normal(n)   # f1 joins from 2024-01-04
    ret_b = 0.4 * f[3] - 0.3 * f[4] + rng.standard_normal(n)       # the same loadings every day
    # One row per trade; cum_volume starts again from zero each day, and is the volume spec's clock.
    pl.DataFrame({"ts": ts, "day": day, "volume": volume, "cum_volume": np.cumsum(volume), "ret_a": ret_a, "ret_b": ret_b}
                 | {f"f{i}": f[i] for i in range(10)}).write_parquet(f"days/{day}.parquet")

# 2. The grid: two targets, five named sets of the ten factors, two half-lives in time and two in traded volume.
factors = [f"f{i}" for i in range(10)]
# Each named set is one candidate feature list; its name tags the slots it writes, e.g. pred_ret_a__f0+f1@h30m.
sets = {"f0": ["f0"], "f0+f1": ["f0", "f1"], "f3+f4": ["f3", "f4"], "f0-f4": factors[:5], "f0-f9": factors}
# Shared by both specs: every set is fitted to both targets, and emit_selected adds each row's chosen grid point.
grid = dict(targets=["ret_a", "ret_b"], features=factors, feature_sets=sets, emit_selected=True)
# The time clock: rows age by the time between trades, a step capped at five minutes, and a new day ages them one minute.
on_time = dict(clock="ts", gap_cap="5m", session="day", session_gap="1m")   # a new day ages a fit by one minute
by_time = po.spec.ewridge("by_time", half_life=["30m", "2h"], **on_time, **grid)   # 5 sets x 2 half-lives = 10 points
# The volume clock: rows age by the shares traded between them, so the half-lives and caps are in shares.
by_volume = po.spec.ewridge("by_volume", clock="cum_volume", half_life=[200_000.0, 800_000.0],   # in shares
                            gap_cap=20_000.0, session="day", session_gap=1_000.0, **grid)

# 3. The watchers: a test for a break in the coefficients, and the age of the factors' regime.
# A plain ridge on the four planted factors whose break_wald_<target> field tests for moved coefficients.
breaks = po.spec.ewridge("breaks", targets=["ret_a", "ret_b"], features=["f0", "f1", "f3", "f4"],
                         half_life="30m", emit_breaks=True, **on_time)
regime = po.spec.bocpd("regime", features=factors[:5], hazard=20_000.0)   # a prior of one change in 20,000 rows

# 4. One streaming query over every file: the bank's output, with the readings to count, written to a file.
field = lambda spec, name: pl.col(spec).struct.field(name)   # each spec writes one column of fields, named after it
(pl.scan_parquet("days/*.parquet")                                # all four days, read in order
   .online.fit_predict([by_time, by_volume, breaks, regime])     # every spec fitted in one pass over the rows
   # Keep the two grid specs whole for step 6, and reduce the watchers to the readings counted in step 5.
   .select("ts", "day", "ret_a", "ret_b", "by_time", "by_volume",
           # the grid point the time spec chose on each trade, for each target
           pick_a=field("by_time", "selected_ret_a"), pick_b=field("by_time", "selected_ret_b"),
           # chi2(5)'s 0.01% value: four factors and the intercept
           break_a=field("breaks", "break_wald_ret_a") > 25.7, break_b=field("breaks", "break_wald_ret_b") > 25.7,
           # a regime younger than ten rows, once the detector has read a hundred
           new_regime=(field("regime", "run_mode") < 10) & (field("regime", "weight_sum") > 100))
   .sink_parquet("fitted.parquet"))

# 5. Each day's most chosen sets, and the break test's flags, read from the file as queries.
fitted = pl.scan_parquet("fitted.parquet")
for target in ("a", "b"):
    # Count each day's choices, sort the most chosen first within each day, and keep that one.
    top = (fitted.group_by("day", f"pick_{target}").len().sort("day", "len", descending=[False, True])
           .group_by("day", maintain_order=True).first().collect(engine="streaming"))
    # The break test's flags for the target, by day.
    flags = fitted.group_by("day").agg(pl.col(f"break_{target}").sum(), pl.col("ts").filter(f"break_{target}").min()).sort("day")
    flags = flags.collect(engine="streaming")
    # One line per day: the set chosen most, on how many trades, and the first break flag if any.
    for (day, pick, held), (_, n_flagged, first) in zip(top.iter_rows(), flags.iter_rows()):
        flag = f"flagged from {first:%H:%M:%S}" if n_flagged else "no break"
        print(f"ret_{target} {day}: {pick} on {held:,} trades; {flag}")
# The first young-regime row of each day: where the detector saw a new regime begin.
starts = fitted.filter("new_regime").group_by("day").agg(pl.col("ts").min()).sort("day").collect(engine="streaming")
print("a new regime from " + ", ".join(f"{t:%Y-%m-%d %H:%M:%S}" for t in starts["ts"]))

# 6. Every slot's score in both specs, read from the file's query in one pass, a chunk at a time.
# One row per spec, slot and target with its mse and other scores; sorted so each group's best slot comes first.
scores = po.eval.compare_specs(fitted, ["by_time", "by_volume"]).sort("spec", "target", "mse")
# Each spec's best grid point for each target, beside the selection made as it ran.
for (spec, target), rows in scores.group_by("spec", "target", maintain_order=True):
    # The lowest mse over the grid (known only after the fact), and the slot that followed each row's choice.
    best, chose = rows.row(0, named=True), rows.filter(slot=f"pred_{target}__selected").row(0, named=True)
    print(f"{spec}, {target}: best {best['slot']} at mse {best['mse']:.3f}, the selection made as it ran {chose['mse']:.3f}")
```

```text
ret_a 2024-01-02: f0@h2h on 2,002 trades; no break
ret_a 2024-01-03: f0@h30m on 2,104 trades; no break
ret_a 2024-01-04: f0+f1@h30m on 3,914 trades; flagged from 09:31:57
ret_a 2024-01-05: f0+f1@h30m on 4,276 trades; flagged from 09:30:02
ret_b 2024-01-02: f3+f4@h2h on 3,419 trades; no break
ret_b 2024-01-03: f3+f4@h2h on 2,051 trades; no break
ret_b 2024-01-04: f3+f4@h2h on 2,754 trades; no break
ret_b 2024-01-05: f3+f4@h2h on 3,111 trades; no break
a new regime from 2024-01-03 12:00:07
by_time, ret_a: best pred_ret_a__f0+f1@h30m at mse 1.043, the selection made as it ran 1.087
by_time, ret_b: best pred_ret_b__f3+f4@h2h at mse 1.007, the selection made as it ran 1.008
by_volume, ret_a: best pred_ret_a__f0+f1@h200000 at mse 1.040, the selection made as it ran 1.082
by_volume, ret_b: best pred_ret_b__f3+f4@h800000 at mse 1.007, the selection made as it ran 1.008
```

**The pass found each planted change.** For `ret_a`, the time clock's
selection chose `f0` alone most often on the first two days, and `f0` and
`f1` from 2024-01-04, the day the coefficient on `f1` rose to 0.6. For
`ret_b`, it chose `f3` and `f4` on every day. The break test flagged
nothing on the first two days. It flagged `ret_a`'s coefficients from
09:31:57 on 2024-01-04, under two minutes after the open, and never
flagged `ret_b`'s, which do not move. The regime detector found a new
regime 7 seconds after noon on 2024-01-03, where the factors tripled, and
on no other day.

**Report the error of the selection made as the stream ran, though it is
worse than the best grid point's.** The best grid point is known only once
the stream has ended: picking it after the fact uses every row to choose
([Why the best error over a grid is
optimistic](explanation/grids-and-instances.md#why-the-best-error-over-a-grid-is-optimistic)).
`pred_ret_a__selected` was chosen on each trade from the trades before
it, so its error is the one a live run would have had. Each of the two
grid specs writes 22 slots: one prediction per grid point and target, and
the selection for each target.

**Every part of this round exists elsewhere on its own. What one pass
gives here is the combination.** scikit-learn refits each feature subset
on data held in memory (`SequentialFeatureSelector`), statsmodels tests
for a break on an array in memory (`breaks_cusumolsresid`, and
`RecursiveLS`'s CUSUM), and river chooses among online models by their
error (`GreedyRegressor`). Here the 20 grid points for two targets, the
selection, the break test and the regime detector all run in one query
over the files. Each is out-of-sample, on clocks of time and of volume
with sessions between the days, and memory stays bounded however large the
files.

## Build a forward-looking signal and resume it

Build a target and features as the data streams, and fit a forecast on
them. The target is a bounded reverse EWMA of the mid, whose weights
decay forward over the next five minutes, minus the current mid; a feature
is the mid minus its bounded EWMA over the last five. A first-stage ridge fits a factor on its inputs, and its
out-of-sample residual becomes a feature of a second stage, which
forecasts the target. The second stage learns under an embargo: a
prediction at time T has learned only from rows whose five minutes had
closed by T, never from a target still being formed, such as the one at
T − 1. A robust regression handles the price's jumps. The fit is saved
partway through the data and resumed in a fresh Python process, where
prediction and training go on exactly as they left off:

- **A target from a bounded EWMA that looks forward, and a feature from
  one that looks back.** An EWMA is an exponentially weighted moving
  average. A [bounded EWMA](GLOSSARY.md#fitting-a-model) is an EWMA whose
  history has a maximum age: its weights decay along the clock as any
  EWMA's do, but rows farther than `window_size` from the row drop out
  instead of fading forever. It runs forward (a target) or backward (a
  feature), streaming on uneven timestamps. Polars' `ewm_mean_by` has no
  window and `rolling_mean_by` has no decay. [How a window operator weighs
  the rows near each row](WINDOWS.md) gives the full definition.
- **An [embargo](GLOSSARY.md#time)**, the delay on the clock between
  predicting a row and learning from it: the model learns each row only
  once its five minutes are known, so it never learns from the future.
- **A factor from a two-stage regression**: a first-stage ridge fits the
  factor on its inputs. Its out-of-sample residual (the factor's own move,
  which its inputs do not explain) is computed in the same query as the
  windows and written beside them, as a feature of the second stage.
- **A robust regression**, `huber`, beside a ridge, on a price that jumps.
- **An exact resume**: a fresh Python process loads the state the first
  run saved, drops the rows that state has read, and goes on as one
  unbroken run would.

The recipe plants a price whose moves follow each trade's signed size,
`flow`, and a factor's own move, `e`: right after each trade the mid price
(`mid`) moves 0.5 per unit of flow and 0.3 per unit of `e`. So each row's
forward mean holds that move, and the coefficients on `flow` and on the
factor's residual are 0.5 and 0.3. The factor, `d`, is the sum of its three
inputs plus `e`. The mid's other steps are a random walk, so how far it
sits from its bounded EWMA over the last five minutes (the feature
`stretch`) predicts nothing, and its coefficient is 0. One trade in a
thousand is followed by a jump of 50.

1. **Make up three days of trades, one parquet file per day.**
2. **Add the feature and the target as columns with
   `.online.with_windows`.** `po.ewm_mean("mid", half_life="1m",
   window_size="5m")` is the bounded EWMA of the last five minutes, and
   `reverse=True` makes it look at the next five.
3. **Fit the first-stage ridge of the factor on its inputs,** with
   `.online.fit_predict` chained after `.online.with_windows` in the query
   of step 2.
4. **Take its residual as a column,** `resid_d`, with
   `pl.col("factor").struct.field("resid_d")`, which reads the field
   `resid_d` of the spec `factor`'s output column, and drop that output
   column.
5. **Write the columns to a file with `sink_parquet`,** which runs the
   query and writes the result without holding it all in memory.
6. **Give the bank two specs,** `robust` (a `huber` model) and `ridge` (an
   `ewridge` model), on `flow`, `stretch` and `resid_d`, each with
   `embargo="5m"`. `coef_every=0` writes each fit's coefficients on every
   row.
7. **Fit the rows up to noon on the second day.**
8. **Save the state** (everything the bank has learned, and the rows it
   holds for their embargo) with `bank.save`.
9. **In a fresh Python process, load the state, and drop the rows it has
   read** (learned, or held for their embargo) with `bank.skip_learned`.
10. **Resume from the state on the rows that remain,** with the query's
    `online.fit_predict(load_state=..., save_state=...)`, which saves the
    state again when the run ends, and write the predictions with
    `sink_parquet`.
11. **Run one unbroken query over every row with the same specs, and
    compare** its predictions and saved state with the resumed run's, and
    each spec's coefficients with the planted ones.

```python
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import polars_online as po

# 1. Three days of trades, one file per day, with the answer planted.
rng = np.random.default_rng(2)
Path("trades").mkdir(exist_ok=True)
n = 4_000                                                          # trades a day
for k in range(3):
    # A day of trades at uneven times, each with its signed size and the factor's inputs and own move.
    day = date(2024, 1, 2 + k)
    ts = [datetime(2024, 1, 2 + k, 9, 30) + timedelta(seconds=float(s))
          for s in np.cumsum(rng.exponential(23_400 / n, n))]
    flow = rng.standard_normal(n)                                  # each trade's signed size
    x = rng.standard_normal((3, n))                                # the factor's three inputs
    e = rng.standard_normal(n)                                     # the factor's own move
    # The mid's move after each trade.
    jumps = np.where(rng.random(n) < 0.001, rng.choice([-50.0, 50.0], n), 0.0)
    step = 0.5 * flow + 0.3 * e + 0.1 * rng.standard_normal(n) + jumps
    mid = 100.0 + np.concatenate([[0.0], np.cumsum(step)[:-1]])    # each trade sees the mid before its move
    # The factor d is the sum of its inputs plus its own move e; e itself is not written, only d and the inputs.
    pl.DataFrame({"ts": ts, "day": day, "flow": flow, "mid": mid, "d": x.sum(axis=0) + e}
                 | {f"x{j}": x[j] for j in range(3)}).write_parquet(f"trades/{day}.parquet")

# 2. The feature and the target, bounded EWMAs over five minutes, back and ahead.
recent = dict(half_life="1m", window_size="5m")                    # weights halve each minute; rows past 5 minutes drop out
# with_windows adds each expression as a column, computed as the rows stream past.
windows = pl.scan_parquet("trades/*.parquet").online.with_windows(
    stretch=pl.col("mid") - po.ewm_mean("mid", **recent),                 # the mid against its last five minutes
    fwd=po.ewm_mean("mid", **recent, reverse=True) - pl.col("mid"),       # the next five minutes against the mid
    clock="ts", gap_cap="5m",                                             # the windows' clock, and its longest step
)
# 3. The first stage: a ridge of the factor on its inputs, in the same query.
factor = po.spec.ewridge("factor", targets=["d"], features=["x0", "x1", "x2"], clock="ts", gap_cap="5m", half_life="1h")
staged = windows.online.fit_predict([factor])                    # still a query: nothing has run yet
# 4 and 5. Its out-of-sample residual as a column, and the columns written to a file, a chunk at a time.
# resid_d is d minus the ridge's prediction made before learning that row, so it is out-of-sample.
staged.with_columns(pl.col("factor").struct.field("resid_d")).drop("factor").sink_parquet("features.parquet")

# 6. A robust regression and a ridge, each learning a row once its five minutes are known.
common = dict(targets=["fwd"], features=["flow", "stretch", "resid_d"], clock="ts", half_life="2h", gap_cap="5m",
              embargo="5m",       # a row is learned once its five minutes are known
              coef_every=0)       # each fit's coefficients on every row
bank = po.ModelBank([po.spec.huber("robust", **common), po.spec.ewridge("ridge", **common)])   # same settings, two losses
# 7 and 8. Fit up to noon on the second day, and save the state.
bank.fit(pl.scan_parquet("features.parquet").filter(pl.col("ts") < datetime(2024, 1, 3, 12)))   # learns, writes nothing
bank.save("signal.state")                                        # the fits, and the rows still waiting out their embargo
```

Steps 9 to 11 run in a fresh Python process, as they would on another
machine that has the state file and the input:

```python
from pathlib import Path

import polars as pl
import polars_online as po

# 9. Load the state, and keep only the rows it has not read.
bank = po.ModelBank.load("signal.state")
kept = bank.skip_learned(pl.scan_parquet("features.parquet"))   # still a query
# 10. The rest of the rows through the state, which is saved again at the end.
kept.online.fit_predict(load_state="signal.state", save_state="signal.state").sink_parquet("served.parquet")
# Row counts of the two files: the resumed run's rows, and all rows; the difference is what the state had read.
resumed, read = (pl.scan_parquet(f).select(pl.len()).collect().item() for f in ("served.parquet", "features.parquet"))
print(f"resumed on {resumed:,} rows, after {read - resumed:,} the state had read")

# 11. One unbroken run over every row, with the same specs, as a query of its own.
(pl.scan_parquet("features.parquet").online.fit_predict(bank.specs, save_state="whole.state")
   .sink_parquet("whole.parquet"))
# Each spec's prediction from both runs, lined up row for row: the unbroken run's last rows are the resumed run's.
preds = dict(robust=pl.col("robust").struct.field("pred_fwd"), ridge=pl.col("ridge").struct.field("pred_fwd"))
served = pl.scan_parquet("served.parquet").select(**preds)
whole = pl.scan_parquet("whole.parquet").tail(resumed).select(**{f"{k}_whole": v for k, v in preds.items()})
differ = pl.concat([served, whole], how="horizontal").select(   # rows where the two runs differ, counted as a stream
    sum((~pl.col(k).eq_missing(pl.col(f"{k}_whole"))).sum() for k in preds)).collect(engine="streaming").item()
print(f"the same predictions to the bit: {differ == 0}")
print(f"and the same state, byte for byte: {Path('signal.state').read_bytes() == Path('whole.state').read_bytes()}")
# Each fit's coefficients averaged over the last two days, against the planted 0.5, 0.3 and 0.
last_two_days = pl.scan_parquet("whole.parquet").tail(8_000)      # 4,000 trades a day
for s in ("robust", "ridge"):
    coef = pl.col(s).struct.field("coef")                          # [intercept, flow, stretch, resid_d] on every row
    on_flow, on_stretch, on_resid = (coef.list.get(i) for i in (1, 2, 3))
    # The mean of each coefficient over those rows, and how far the one on flow typically sits from 0.5.
    flow, rms, resid, stretch = last_two_days.select(
        flow=on_flow.mean(), rms=((on_flow - 0.5) ** 2).mean().sqrt(), resid=on_resid.mean(), stretch=on_stretch.mean(),
    ).collect(engine="streaming").row(0)
    print(f"{s}: on flow {flow:.3f}, {rms:.3f} from 0.5 (root mean square); on resid_d {resid:.3f}; on stretch {stretch:.3f}")
```

```text
resumed on 6,494 rows, after 5,506 the state had read
the same predictions to the bit: True
and the same state, byte for byte: True
robust: on flow 0.506, 0.023 from 0.5 (root mean square); on resid_d 0.305; on stretch -0.016
ridge: on flow 0.521, 0.079 from 0.5 (root mean square); on resid_d 0.325; on stretch -0.029
```

**The resumed run writes what a run that never stopped writes.** The
fresh process read the state saved at noon, dropped the 5,506 rows the
state had already read, and wrote the predictions and the state an
unbroken run writes, to the bit. A state file loads on macOS, Windows and
Linux alike. Resumed on another operating system, its predictions match
within `1e-12 × (1 + |value|)`, since the platforms' math libraries can
round the last bit differently ([Resume from a saved
state](how-to/resume-from-a-saved-state.md#check-that-it-worked)).

**The robust fit stays near the planted coefficients through the jumps,
and the ridge strays further.** Averaged over the last two days, `huber`
reads 0.506 on `flow` and 0.305 on the factor's residual, against the
planted 0.5 and 0.3, and its coefficient on `flow` is 0.023 from 0.5 on a
typical row, against the ridge's 0.079. A jump of 50 lands in the
five-minute window of some fifty rows' targets. The ridge weighs each of
those rows by its squared error, so every jump pulls its coefficients
away, while `huber` down-weights the rows it misses by far.

**Each part has a counterpart elsewhere. What one stream gives here is the
combination.** Polars and pandas each bound an EWMA by a count of rows
(Polars' `rolling_mean` with `weights=`, pandas' `rolling(n,
win_type="exponential")`), and neither bounds one by a span of time. Asked
for one, pandas refuses `win_type` on a window of time. In Polars, such an
EWMA is written by hand, as a weighted aggregation inside `rolling`. The
river library evaluates a model on targets that arrive late
(`progressive_val_score`'s `delay=`), but its exponentially weighted
statistics have no window of time. Here the bounded EWMAs, forward and
back, the embargo, the factor's out-of-sample residual, the robust fit and
the resume work together on one stream, from files to files.

## Prediction intervals on late-arriving residuals

For a model whose target is known about 30 seconds after it was
predicted, build a 90% prediction interval around each forecast, one that
still covers nine answers in ten while they arrive late. Enrich it by
flagging the rows where a coefficient of the regression significantly
changes, and where the correlation between the features does. All of it
runs in one pass over the data, and each check is tested here against
changes planted at known rows:

- **Intervals that hold their coverage when targets arrive late:** a
  conformal interval (`conformal=0.9`) on each prediction, learned as each
  target becomes known (`label_clock`), with a step that adapts to how
  many targets are waiting (`conformal_rate="auto"`).
- **Per-row diagnostics:** a test, on every row, of whether the
  regression's coefficients have moved (`emit_breaks`).
- **A detector of the features' correlation:** `corrchange`, which flags
  a change in how the features move together.
- **Detectors tested against known truth:** `po.sim.regimes` simulates
  features whose correlation changes by regime, and reports where each
  regime begins.

The recipe plants a correlation between the two features that takes turns
between 0 and 0.7 every 5,000 rows, and a slope on `x_2` that flips from
0.5 to −0.5 at row 10,000. The rows are trades about 0.1 seconds apart,
each row's target becomes known 25 to 35 seconds after the trade, and its
noise has heavy tails.

1. **Simulate the features' levels with `po.sim.regimes`,** which also
   returns the block of rows (one regime) each row is in.
2. **Take the levels' changes with `diff()`,** since a correlation
   detector reads changes, not levels.
3. **Add each trade's time, the target, and the time each target becomes
   known, as columns.**
4. **Write the stream to a parquet file.**
5. **Give the model `label_clock="known"`, `conformal=0.9`,
   `conformal_rate="auto"` and `emit_breaks=True`,** and a second spec
   with the default step, 0.05, for comparison.
6. **Add a `corrchange` spec on the features.**
7. **Run the query over the file, and write the readings to count to a
   file with `sink_parquet`,** which runs the query and writes the result
   without holding it all in memory.
8. **Read each interval's share of targets covered, and its mean
   half-width,** past the first five half-lives, with a query over that
   file.
9. **Read the break test's flags** before the slope flipped and after it.
10. **Find the first correlation flag in each block of rows.** Every block
    after the first begins with a planted change, so a flag in the first
    block is a false alarm.

```python
from datetime import datetime

import numpy as np
import polars as pl
import polars_online as po

# 1. The features' levels, and the regime of each row.
sim = po.sim.regimes(2, states=[0.0, 0.7], transition=[[0.0, 1.0], [1.0, 0.0]],
                     n_blocks=4, rows_per_block=5_000, seed=4)   # the correlation takes turns every 5,000 rows
rng = np.random.default_rng(4)
n = sim["rows"].height                                             # 20,000 rows: four blocks of 5,000
slope = pl.when(pl.col("t") < 10_000).then(0.5).otherwise(-0.5)   # x_2's slope flips at row 10,000
# Attach each row's block, then build the stream's columns one step at a time.
(sim["rows"].join(sim["truth_rows"].select("t", "block"), on="t")
   .with_columns(pl.col("x_1", "x_2").diff().fill_null(0.0))        # 2. the levels' changes
   # 3. A trade every 0.1 s on average, the target with heavy-tailed noise, and when each target becomes known.
   .with_columns(ts=pl.lit(datetime(2024, 1, 2, 9, 30)) + pl.duration(microseconds=pl.lit(np.cumsum(rng.exponential(1e5, n)).astype(np.int64))))
   .with_columns(y=pl.col("x_1") + slope * pl.col("x_2") + pl.lit(rng.standard_t(3, n) / np.sqrt(3.0)),
                 known=pl.col("ts") + pl.duration(milliseconds=pl.lit(rng.uniform(25_000, 35_000, n).astype(np.int64))))
   .write_parquet("live.parquet"))                                  # 4.

# 5. The model, its intervals and its break test; and the same model at the default step.
live = dict(targets=["y"], features=["x_1", "x_2"], clock="ts", gap_cap="1s", half_life="200s",
            label_clock="known",                  # each row is learned when its target is known, 25 to 35 s later
            conformal=0.9)                        # an interval meant to cover 90% of the targets
model = po.spec.ewridge("model", **live, conformal_rate="auto",   # a step that shrinks as more targets wait
                        emit_breaks=True)                         # a per-row test of whether the coefficients moved
fixed = po.spec.ewridge("fixed", **live)          # the default step, 0.05
# 6. A watcher of the features' correlation.
watch = po.spec.corrchange("watch", features=["x_1", "x_2"], kind="sequential",
                           window_rows=300,       # the history the rows after it are monitored against
                           monitor_rows=2_000)    # the rows monitored after each history
# 7. The query, run a chunk at a time, with the readings to count written to a file.
field = lambda spec, name: pl.col(spec).struct.field(name)   # each spec writes one column of fields, named after it
(pl.scan_parquet("live.parquet").online.fit_predict([model, fixed, watch])
   # Per spec: did the interval [lo_y, hi_y] cover the target, and its half-width.
   .select("t", "block",
           **{f"covered_{s}": pl.col("y").is_between(field(s, "lo_y"), field(s, "hi_y")) for s in ("model", "fixed")},
           **{f"width_{s}": (field(s, "hi_y") - field(s, "lo_y")) / 2 for s in ("model", "fixed")},
           broke=field("model", "break_wald_y") > 21.1,   # chi2(3)'s 0.01% value: two features and the intercept
           flag=field("watch", "flag"))                    # True on a row where corrchange flags a change
   .sink_parquet("watched.parquet"))
watched = pl.scan_parquet("watched.parquet")

# 8. Each interval's coverage and mean half-width, past the first five half-lives.
for s in ("model", "fixed"):
    # The share of rows whose target fell inside the interval, and the interval's mean half-width.
    covered, width = watched.filter(pl.col("t") >= 10_000).select(f"covered_{s}", f"width_{s}").mean().collect(engine="streaming").row(0)
    print(f"{s}: covered {covered:.3f} of the targets, half-width {width:.2f}")
# 9. The break test's flags before the slope flipped, and its first after.
# Of the flagged rows: how many fell before the flip (false alarms), and the first one after it.
before, first = watched.filter("broke").select(
    before=(pl.col("t") < 10_000).sum(), first=pl.col("t").filter(pl.col("t") >= 10_000).min()).collect(engine="streaming").row(0)
print(f"break test: {before} flags before row 10000, the first after it at row {first:.0f}")
# 10. Each block's first row and first correlation flag: the first block holds no change.
# Per block: its first row (where the planted change is), its first flag, and its number of flags.
blocks = watched.group_by("block").agg(start=pl.col("t").min(), flagged=pl.col("t").filter("flag").min(), flags=pl.col("flag").sum())
for block, start, flagged, flags in blocks.sort("block").collect(engine="streaming").iter_rows():
    if block == 0:
        print(f"flags before any change: {flags}")
    else:
        print(f"correlation changed at row {start:.0f}: " + (f"flagged at row {flagged:.0f}" if flagged is not None else "missed"))
```

```text
model: covered 0.904 of the targets, half-width 1.56
fixed: covered 0.895 of the targets, half-width 3.39
break test: 0 flags before row 10000, the first after it at row 10474
flags before any change: 0
correlation changed at row 5000: flagged at row 5110
correlation changed at row 10000: flagged at row 10098
correlation changed at row 15000: flagged at row 15056
```

**Both intervals cover 90% of the targets, and the adaptive one is
narrower, 1.56 against 3.39.** About 300 trades, the last 30 seconds of
them, wait for their targets at any time. At the default step, the
interval learns of each miss 30 seconds late, overshoots and undershoots,
and is wider on average.
`"auto"` takes smaller steps the more rows wait. [Keep the coverage when
targets arrive
late](how-to/put-an-interval-on-a-prediction.md#keep-the-coverage-when-targets-arrive-late)
measures the step on more streams.

**The break test flagged nothing before the slope flipped, and the
correlation detector found each planted change, with no false alarm.**
The break test needs the targets of the rows after the flip, which arrive
about 30 seconds late, so it flagged the flip 474 rows after it. [Test a
detector on known truth](how-to/test-a-detector-on-known-truth.md) runs a
detector over many simulated streams to measure its false alarms.

**Each part has a counterpart elsewhere. What one pass gives here is the
combination.** MAPIE's adaptive conformal inference updates its interval
with a step you give (`gamma`). Salesforce's `online_conformal` (FACI)
learns its step from the misses, with a tracker for each horizon it
predicts, and river's `RegressionJackknife` reads quantiles of past
residuals. Of the three, checked on 2026-11-08, none sets its step by how
many targets are waiting. Here that interval, the per-row break test and
the correlation detector run in the same pass as the model.

## Measure correlations on tick data

This task works on trades, where each instrument ticks at times of its
own:

- **Correlations from asynchronous ticks:** `po.stream.refresh_time` puts
  series that tick at their own times on *refresh times*, the times by
  which every series has ticked again, and a model tracks their moves'
  correlation there.

The recipe plants three prices that share one move, each adding a move of
its own as large, so the moves of any two correlate at 0.5.

1. **Make up the three series' trades, each at times of its own, in one
   parquet file in clock order.**
2. **Put the series on refresh times with `po.stream.refresh_time`.**
3. **Take each series' change from one refresh time to the next.**
4. **Track the changes' correlations with an `ew_cov` spec.**
5. **Read each series' last price every 0.1 clock units, for
   comparison,** and correlate the changes there.

```python
import numpy as np
import polars as pl
import polars_online as po

# 1. Three prices that share one move, each traded at times of its own, about once a clock unit.
rng = np.random.default_rng(1)
shared = np.cumsum(rng.normal(0.0, 0.1, 200_000))           # a hidden path, 0.01 clock units a step
names = ["AAA", "BBB", "CCC"]
frames = []                                                      # one frame of trades per series
for s in names:
    # The series' price, plus a move of its own as large, and the times it trades.
    price = shared + np.cumsum(rng.normal(0.0, 0.1, 200_000))
    t = np.cumsum(rng.exponential(1.0, 2_100))                   # trade times, about one per clock unit
    t = t[t < 2_000]
    # Each trade's price is the series' price at that time on the 0.01-unit grid.
    frames.append(pl.DataFrame({"symbol": s, "t": t, "px": price[(t * 100).astype(int)]}))
pl.concat(frames).sort("t").write_parquet("ticks.parquet")      # one stream: every series' trades in time order

# 2 to 4. The series on refresh times, their changes, and the changes' correlations.
# One row per refresh time, with each series' last price by then in <symbol>_value.
refresh = po.stream.refresh_time(pl.scan_parquet("ticks.parquet"), series="symbol", names=names, clock="t", value="px")
# Each series' change since the previous refresh time.
moves = refresh.select("time_refresh", *(pl.col(f"{s}_value").diff().alias(s) for s in names))
# The changes' correlations over the whole stream: an infinite half-life weighs every row alike.
comove = po.spec.ew_cov("comove", features=names, stats=["corr"], clock="time_refresh", gap_cap=10.0, half_life=float("inf"))
last = moves.online.fit_predict([comove]).select(pl.col("comove").last()).collect().item()   # the last row's struct
pairs = [("AAA", "BBB"), ("AAA", "CCC"), ("BBB", "CCC")]
print("planted 0.5; on the refresh times: " + ", ".join(f"{last[f'corr_{a}_{b}']:.2f}" for a, b in pairs))
# 5. Each series' last price every 0.1 clock units, and the correlations of its changes there.
every = pl.LazyFrame({"t": np.arange(0.0, 2_000.0, 0.1)})        # a regular grid of times
for s in names:                                                  # builds the query: one as-of join per series
    every = every.join_asof(pl.scan_parquet("ticks.parquet").filter(symbol=s).select("t", pl.col("px").alias(s)), on="t")
# The changes between grid times, and their correlations; most grid steps see no trade, so most changes are 0.
corr = (every.select(pl.col(names).diff()).drop_nulls()
        .select(pl.corr(a, b).alias(f"{a}_{b}") for a, b in pairs).collect(engine="streaming").row(0))
print("read every 0.1: " + ", ".join(f"{c:.2f}" for c in corr))

```

```text
planted 0.5; on the refresh times: 0.34, 0.32, 0.36
read every 0.1: 0.01, 0.02, 0.02
```

**On refresh times the correlation is two thirds of the planted 0.5, and
read every 0.1 clock units it all but vanishes.** Read so often, most
moves are zero because a series has not ticked, so the moves of two series
almost never fall together. [Series that tick at their own
times](explanation/asynchronous-series.md) explains the fall, and [the two
ways out](explanation/asynchronous-series.md#the-two-ways-out) of it.

**Each part has a counterpart elsewhere. What one pass gives here is the
combination.** Neither Polars nor pandas samples on refresh times, and
Python's `hfhd` package and R's `highfrequency` package do, on series held
in memory. Here the refresh times and the correlations on them run as
queries over parquet files.

## Capabilities

Look a capability up to find the step that shows it and a guide that
covers it. For every model, function and parameter of
the library, one line each, see [the feature index](FEATURES.md).

| capability | what it gives you | shown in | guide |
|---|---|---|---|
| a target from a bounded EWMA that looks forward | the exponentially weighted mean of the rows after each row, up to a maximum distance on the clock, less the row's value | [build a signal](#build-a-forward-looking-signal-and-resume-it), step 2 | [Make a target that looks ahead](how-to/make-a-target-that-looks-ahead.md) |
| a feature from a bounded EWMA that looks back | how far a value sits from its recent weighted mean, with rows older than the window dropped | [build a signal](#build-a-forward-looking-signal-and-resume-it), step 2 | [Make features that look back over the stream](how-to/make-window-features.md) |
| `embargo` | each row predicted at its own clock value and learned only once its target is known, so no model learns from the future | [build a signal](#build-a-forward-looking-signal-and-resume-it), step 6 | [Learn targets known late](how-to/learn-targets-known-late.md) |
| a two-stage regression | each first-stage fit's out-of-sample residual as a feature of the second stage | [build a signal](#build-a-forward-looking-signal-and-resume-it), steps 3 to 6 | [Fit on factor residuals](how-to/fit-on-factor-residuals.md) |
| a half-life in time or in traded volume | a local fit that forgets on any sorted clock, with sessions and gaps measured in its units | [model selection](#automate-a-round-of-model-selection), step 2 | [Handle sessions, gaps and steps back](how-to/handle-sessions-gaps-and-steps-back.md) |
| many combinations of factors and targets, with multiple specs | every feature set at every half-life, for every target, in several specs, from one read of the rows | [model selection](#automate-a-round-of-model-selection), step 2 | [Grids, instances and slots](explanation/grids-and-instances.md) |
| one pass that detects changes of coefficients, selection and regimes | a break test, the grid point chosen on each row and a regime's age, beside the fits they watch | [model selection](#automate-a-round-of-model-selection), steps 2 to 5 | [Detect a break](how-to/detect-a-break.md) |
| model selection made out-of-sample | the grid point whose exponentially weighted out-of-sample error is lowest so far, chosen on each row from the rows before it | [model selection](#automate-a-round-of-model-selection), step 2 | [Choose among a grid's settings as the stream runs](how-to/choose-among-a-grid-online.md) |
| scores of an output too large to hold | one call over a query of the output, which adds up sums per slot a chunk at a time | [model selection](#automate-a-round-of-model-selection), step 6 | [Score an output too large to hold](how-to/evaluate-an-output-frame.md#score-an-output-too-large-to-hold) |
| robust regression | a fit that down-weights the rows it misses by far, such as jumps | [build a signal](#build-a-forward-looking-signal-and-resume-it), step 6 | [The robust scale](explanation/robust-scale.md) |
| an exact resume | a saved state goes on in a fresh process as one unbroken run would, dropping the rows it has read | [build a signal](#build-a-forward-looking-signal-and-resume-it), steps 8 to 11 | [Resume from a saved state](how-to/resume-from-a-saved-state.md) |
| intervals when targets arrive late | a conformal interval whose step adapts to the targets still waiting | [intervals](#prediction-intervals-on-late-arriving-residuals), steps 5 and 8 | [Put an interval on a prediction](how-to/put-an-interval-on-a-prediction.md) |
| per-row break tests | a test on every row of whether a regression has broken | [intervals](#prediction-intervals-on-late-arriving-residuals), steps 5 and 9 | [Detect a break](how-to/detect-a-break.md) |
| a detector of the features' correlation | a flag where the features' correlations change | [intervals](#prediction-intervals-on-late-arriving-residuals), steps 6 and 10 | [Detect a break](how-to/detect-a-break.md#watch-the-features-correlations-with-corrchange) |
| a detector tested against known truth | a simulated stream with planted regime changes, and each change matched to the detector's flag | [intervals](#prediction-intervals-on-late-arriving-residuals), steps 1 and 10 | [Test a detector on known truth](how-to/test-a-detector-on-known-truth.md) |
| correlations from asynchronous ticks | series that tick at their own times, aligned on refresh times | [tick data](#measure-correlations-on-tick-data), steps 2 to 4 | [Align series that tick at their own times](how-to/align-series-that-tick-at-their-own-times.md) |
