<!-- kind: how-to -->
# What you can do with polars-online

This page is for anyone with one of five tasks, from a round of model
selection to correlations and curves on tick data, who wants to see
polars-online do it end to end. Each task runs on data it makes up with a
known answer, and prints what the library found, to read against that
answer.

| task | what it shows |
|---|---|
| [Automate a round of model selection](#automate-a-round-of-model-selection) | every combination of factors built from first-stage residuals, fitted at once at [half-lives](GLOSSARY.md#time) in time and in traded volume, with the best chosen [out-of-sample](GLOSSARY.md#fitting-a-model) on each trade, and breaks and a regime change found in the same pass |
| [Build a forward-looking signal and deploy it](#build-a-forward-looking-signal-and-deploy-it) | a target over the next five minutes and a feature over the last five, learned under an [embargo](GLOSSARY.md#time), fitted robustly through jumps, and resumed in a fresh process with the same predictions to the bit |
| [Train gradient-boosted trees on files larger than memory](#train-gradient-boosted-trees-on-files-larger-than-memory) | XGBoost and LightGBM trained from parquet files around an online model's prediction, which follows a level that drifts beyond every training row's |
| [Monitor a live model](#monitor-a-live-model) | 90% intervals that keep their coverage on targets that arrive late, and a test for a broken slope and a detector of the features' correlation, both checked against planted changes |
| [Measure correlations and fit curves on tick data](#measure-correlations-and-fit-curves-on-tick-data) | correlations from series that trade at their own times, and a curve fitted locally along trade size |
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
- For [Train gradient-boosted trees on files larger than
  memory](#train-gradient-boosted-trees-on-files-larger-than-memory),
  XGBoost and LightGBM, as `pip install xgboost lightgbm`. `po.gbdt`, which
  builds their training objects, is unstable: its keywords may change in
  any release. It runs fastest with XGBoost 2.1 to before 4.0 and LightGBM
  4.0 to before 5.0, and falls back to a slower engine outside them.

## Automate a round of model selection

A round of model selection builds factors, fits every combination of them
under a few settings, picks the best, and asks whether what it picked
holds. This task does the whole round in one pass:

- **Local fits on half-lives**, where each row's weight halves every
  [half-life](GLOSSARY.md#time) on a [clock](GLOSSARY.md#time): two
  half-lives measured in time (30 minutes and 2 hours) and two measured in
  traded volume (200,000 and 800,000 shares).
- **Factors from a two-stage regression**: a first-stage ridge per factor,
  whose out-of-sample residual (the factor's own move) becomes a feature
  of the second stage.
- **Every combination of factors and specs in one pass**: the seven
  non-empty sets of three factors, at two half-lives on each of the two
  clocks, 28 [grid points](GLOSSARY.md#grids) (combinations of settings)
  from one read of the rows.
- **Changes of coefficients, selection and regime, detected in the same
  pass**: a test for a break in the coefficients, the grid point chosen on
  each trade, and the age of the factors' regime.
- **Out-of-sample selection**: the grid point chosen on each trade is the
  one whose out-of-sample error, exponentially weighted at its own
  half-life, was lowest on the trades before it.
- All in one pass over sorted data too large for RAM, split by date
  across files with session breaks between them, with every
  [slot](GLOSSARY.md#grids) (each prediction a spec writes) scored from
  **sums merged chunk by chunk**, so no output is held.

The recipe plants each answer it looks for. Three factors each add a move
of their own to the sum of their inputs. The return loads 0.5 on factor
0's move throughout, and on factor 1's from 0 to 0.6 at the open of
2024-01-04. Factor 2 plays no part. The factors' moves triple in size at
noon on 2024-01-03.

1. **Make up four days of trades, one parquet file per day,** each day a
   [session](GLOSSARY.md#time) (the column `day`), with the cumulative
   volume of the day beside the timestamps.
2. **Fit a first-stage ridge per factor.** `session="day"` with
   `session_gap="1m"` ages each fit by one minute at a new day, in place of
   the night.
3. **Take each ridge's residual as a column.**
   `pl.col("fac0").struct.field("resid_d0")` reads the field `resid_d0` of
   the spec `fac0`'s output column.
4. **Give the grid to two specs, one for each clock.** A dict of named
   column lists in `feature_sets`, beside a list of half-lives in
   `half_life`, makes a [grid](GLOSSARY.md#grids): every combination is
   fitted from the same rows. `emit_selected=True` adds the grid point
   chosen on each row. The volume spec's clock is the day's cumulative
   volume, so its half-life and its `gap_cap` (the longest step a fit
   ages by) are in shares, and `session_gap` is the step it takes at a new
   day, where the volume starts again from zero.
5. **Add the watchers:** a ridge on every factor with `emit_breaks=True`,
   whose `break_wald_ret` tests whether the coefficients have moved, and a
   `bocpd` spec, which tracks how long the factors' current regime has
   lasted.
6. **Run both stages over the files in a loop, a chunk at a time.** Each
   chunk goes through the first stage, gains the residual columns and goes
   through the second. `po.eval.sums` reduces each chunk's output to a
   few sums per slot, and `po.eval.merge_sums` adds them up, so the
   loop holds one chunk at a time.
7. **Print what the pass found:** the grid point chosen most each day, the
   break test's flags each day, where the regime started over, and each
   spec's best grid point beside its selection made as it ran.

```python
from datetime import date, datetime, timedelta
from itertools import combinations
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
    volume = np.round(rng.lognormal(6.0, 0.8, n))
    # The factors' inputs and moves, and the return on the moves.
    x = rng.standard_normal((3, 3, n))                             # each factor's three inputs
    calm = (k == 0) | ((k == 1) & (np.array(ts) < datetime(2024, 1, 3, 12)))
    e = rng.standard_normal((3, n)) * np.where(calm, 1.0, 3.0)     # each factor's own move
    ret = 0.5 * e[0] + (0.6 if k >= 2 else 0.0) * e[1] + rng.standard_normal(n)
    pl.DataFrame({"ts": ts, "day": day, "volume": volume, "cum_volume": np.cumsum(volume), "ret": ret}
                 | {f"d{f}": x[f].sum(axis=0) + e[f] for f in range(3)}
                 | {f"x{f}_{j}": x[f, j] for f in range(3) for j in range(3)}).write_parquet(f"days/{day}.parquet")

# 2. The first stage: a ridge per factor, on the clock ts, with a session a day.
on_time = dict(clock="ts", gap_cap="5m", session="day", session_gap="1m")   # a new day ages a fit by one minute
factors = [po.spec.ewridge(f"fac{f}", targets=[f"d{f}"], features=[f"x{f}_{j}" for j in range(3)],
                           half_life="1h", **on_time) for f in range(3)]
# 3. Each ridge's out-of-sample residual, the factor's own move, as the columns resid_d0 to resid_d2.
residuals = [pl.col(f"fac{f}").struct.field(f"resid_d{f}") for f in range(3)]

# 4. The grid points: every set of the residuals, at two half-lives in time and two in traded volume.
moves = ["resid_d0", "resid_d1", "resid_d2"]
every_set = {"+".join(c[-2:] for c in s): list(s) for k in (1, 2, 3) for s in combinations(moves, k)}
grid = dict(targets=["ret"], features=moves, feature_sets=every_set, emit_selected=True)
by_time = po.spec.ewridge("by_time", half_life=["30m", "2h"], **on_time, **grid)
by_volume = po.spec.ewridge("by_volume", clock="cum_volume", half_life=[200_000.0, 800_000.0],   # in shares
                            gap_cap=20_000.0, session="day", session_gap=1_000.0, **grid)

# 5. The watchers: a test for a break in the coefficients, and the age of the factors' regime.
breaks = po.spec.ewridge("breaks", targets=["ret"], features=moves, half_life="30m", emit_breaks=True, **on_time)
regime = po.spec.bocpd("regime", features=moves, hazard=20_000.0)   # a prior of one change in 20,000 rows

# 6. One pass over the files, a chunk at a time, keeping only sums and counts.
first, second = po.ModelBank(factors), po.ModelBank([by_time, by_volume, breaks, regime])
scores, chosen, flags, changes, last_run = {}, [], [], [], 0
for chunk in pl.scan_parquet("days/*.parquet").collect_batches(chunk_size=2_000):
    out = second.fit_predict(first.fit_predict(chunk).with_columns(residuals))
    for spec in ("by_time", "by_volume"):                          # each slot's errors, as sums that add up
        part = po.eval.sums(out, spec)
        scores[spec] = po.eval.merge_sums(scores[spec], part) if spec in scores else part
    # The grid point chosen on each trade, the break test's flags, and where the regime starts over.
    picked = pl.col("by_time").struct.field("selected_ret")
    chosen.append(out.group_by("day", picked).len())
    flagged = pl.col("breaks").struct.field("break_wald_ret") > 23.5   # chi2(4)'s 0.01% value: three factors and the intercept
    flags.append(out.group_by("day").agg(flagged.sum(), pl.col("ts").filter(flagged).min()))
    run = out["regime"].struct.field("run_mode")                   # how long the current regime has lasted
    changes += out.filter(run < run.shift(1, fill_value=last_run))["ts"].to_list()   # where it starts over
    last_run = run[-1]

# 7. What the pass found, day by day, and every slot's score.
picks = pl.concat(chosen).group_by("day", "selected_ret").agg(pl.sum("len")).sort("day", "len", descending=[False, True])
found = pl.concat(flags).group_by("day").agg(pl.sum("break_wald_ret"), pl.min("ts"))
for (day, slot, held), (n_flagged, first_flag) in zip(picks.group_by("day", maintain_order=True).first().iter_rows(),
                                                     found.sort("day").select("break_wald_ret", "ts").iter_rows()):
    flag = f"flagged {n_flagged:,} trades from {first_flag:%H:%M:%S}" if n_flagged else "flagged none"
    print(f"{day}: selected {slot} on {held:,} trades; the break test {flag}")
print("the factors' regime started over at " + ", ".join(f"{t:%Y-%m-%d %H:%M:%S}" for t in changes))
# Each spec's best grid point over the whole pass, beside the selection made as it ran.
for spec in ("by_time", "by_volume"):
    ranked = po.eval.from_sums(scores[spec]).sort("mse")
    best, chose = ranked.row(0, named=True), ranked.filter(slot="pred_ret__selected").row(0, named=True)
    print(f"{spec}: {ranked.height} slots over {best['n']:,} trades; best {best['slot']} at mse {best['mse']:.3f}, "
          f"the selection made as it ran {chose['mse']:.3f}")
```

```text
2024-01-02: selected d0@h30m on 2,938 trades; the break test flagged none
2024-01-03: selected d0@h2h on 2,546 trades; the break test flagged none
2024-01-04: selected d0+d1@h30m on 3,413 trades; the break test flagged 4,960 trades from 09:32:42
2024-01-05: selected d0+d1@h30m on 3,863 trades; the break test flagged 791 trades from 09:30:02
the factors' regime started over at 2024-01-03 12:00:23
by_time: 15 slots over 19,990 trades; best pred_ret__d0+d1@h30m at mse 1.074, the selection made as it ran 1.111
by_volume: 15 slots over 19,990 trades; best pred_ret__d0+d1@h200000 at mse 1.070, the selection made as it ran 1.106
```

**The pass found each planted change.** On the first two days the time
clock's selection chose factor 0 alone most often, and from 2024-01-04
factors 0 and 1, the day factor 1's coefficient rose to 0.6. Factor 2 was
never the most chosen. The break test flagged nothing on the first two
days. It flagged the coefficient's move from 09:32:42 on 2024-01-04, under
three minutes after the open, on 4,960 of that day's trades and 791 of the
next day's. The regime detector started a new regime 23 seconds after noon
on 2024-01-03, where the factors' moves tripled, and at no other trade.

**Report the error of the selection made as the stream ran, though it is
worse than the best grid point's.** The best grid point is known only once
the stream has ended: picking it after the fact uses every row to choose
([Why the best error over a grid is
optimistic](explanation/grids-and-instances.md#why-the-best-error-over-a-grid-is-optimistic)).
`pred_ret__selected` was chosen on each trade from the trades before it,
so its error is the one a live run would have had. Each of the two grid
specs writes 15 slots: one prediction per grid point, and the selection.

**Every part of this round exists elsewhere on its own. What one pass
gives here is the combination.** scikit-learn refits each feature subset on
data held in memory (`SequentialFeatureSelector`), statsmodels tests for a
break on an array in memory (`breaks_cusumolsresid`, and `RecursiveLS`'s
CUSUM), and river chooses among online models by their error
(`GreedyRegressor`). Here the factors, the 28 grid points, the selection,
the break test and the regime detector all run in the same read of the
files. Each is out-of-sample, on clocks of time and of volume with
sessions between the days, and the loop holds one chunk at a time.

## Build a forward-looking signal and deploy it

This task builds a signal that predicts the next five minutes of a price
from the last five, fits it robustly, saves it partway, and resumes it
where it stopped in a fresh Python process, as another machine would:

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
- **A robust regression**, `huber`, beside a ridge, on a price that jumps.
- **An exact resume**: a fresh Python process loads the state the first
  run saved, drops the rows that state has read, and goes on as one
  unbroken run would.

The recipe plants a price whose moves follow each trade's signed size,
`flow`: the mid price (`mid`) moves 0.5 per unit of flow right after the
trade. So each row's forward mean holds that move, and the coefficient on
`flow` is 0.5. The mid's other steps are a random walk, so how far it sits
from its bounded EWMA over the last five minutes (the feature `stretch`)
predicts nothing, and its coefficient is 0. One trade in a thousand is
followed by a jump of 50.

1. **Make up three days of trades, one parquet file per day.**
2. **Add the feature and the target as columns with
   `.online.with_windows`.** `po.ewm_mean("mid", half_life="1m",
   window_size="5m")` is the bounded EWMA of the last five minutes, and
   `reverse=True` makes it look at the next five.
3. **Write the columns to a file with `sink_parquet`,** which runs the
   query and writes the result without holding it all in memory.
4. **Give the bank two specs,** `robust` (a `huber` model) and `ridge` (an
   `ewridge` model), each with `embargo="5m"`. `coef_every=0` writes each
   fit's coefficients on every row.
5. **Fit the rows up to noon on the second day.**
6. **Save the state** (everything the bank has learned, and the rows it
   holds for their embargo) with `bank.save`.
7. **In a fresh Python process, load the state, and drop the rows it has
   read** (learned, or held for their embargo) with `bank.skip_learned`.
8. **Resume from the state on the rows that remain,** with the query's
   `online.fit_predict(load_state=..., save_state=...)`, which saves the
   state again when the run ends, and write the predictions with
   `sink_parquet`.
9. **Compare the resumed run with one unbroken run,** and each spec's
   coefficient on `flow` with the planted 0.5.

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
    # A day of trades at uneven times, and the mid's move after each one.
    day = date(2024, 1, 2 + k)
    ts = [datetime(2024, 1, 2 + k, 9, 30) + timedelta(seconds=float(s))
          for s in np.cumsum(rng.exponential(23_400 / n, n))]
    flow = rng.standard_normal(n)                                  # each trade's signed size
    jumps = np.where(rng.random(n) < 0.001, rng.choice([-50.0, 50.0], n), 0.0)
    step = 0.5 * flow + 0.1 * rng.standard_normal(n) + jumps       # the mid's move after each trade
    mid = 100.0 + np.concatenate([[0.0], np.cumsum(step)[:-1]])    # each trade sees the mid before its move
    pl.DataFrame({"ts": ts, "day": day, "flow": flow, "mid": mid}).write_parquet(f"trades/{day}.parquet")

# 2. The feature and the target, bounded EWMAs over five minutes, back and ahead.
recent = dict(half_life="1m", window_size="5m")
windows = pl.scan_parquet("trades/*.parquet").online.with_windows(
    stretch=pl.col("mid") - po.ewm_mean("mid", **recent),                 # the mid against its last five minutes
    fwd=po.ewm_mean("mid", **recent, reverse=True) - pl.col("mid"),       # the next five minutes against the mid
    clock="ts", gap_cap="5m",
)
# 3. Written to a file, one chunk at a time.
windows.sink_parquet("features.parquet")

# 4. A robust regression and a ridge, each learning a row once its five minutes are known.
common = dict(targets=["fwd"], features=["flow", "stretch"], clock="ts", half_life="2h", gap_cap="5m",
              embargo="5m",       # a row is learned once its five minutes are known
              coef_every=0)       # each fit's coefficients on every row
bank = po.ModelBank([po.spec.huber("robust", **common), po.spec.ewridge("ridge", **common)])
# 5 and 6. Fit up to noon on the second day, and save the state.
bank.fit(pl.scan_parquet("features.parquet").filter(pl.col("ts") < datetime(2024, 1, 3, 12)))
bank.save("signal.state")
```

Steps 7 to 9 run in a fresh Python process, as they would on another
machine that has the state file and the input:

```python
import polars as pl
import polars_online as po

# 7. Load the state, and keep only the rows it has not read.
bank = po.ModelBank.load("signal.state")
kept = bank.skip_learned(pl.scan_parquet("features.parquet"))   # still a query
# 8. The rest of the rows through the state, which is saved again at the end.
kept.online.fit_predict(load_state="signal.state", save_state="signal.state").sink_parquet("served.parquet")
served = pl.read_parquet("served.parquet")
print(f"resumed on {served.height:,} rows, after {pl.read_parquet('features.parquet').height - served.height:,} the state had read")

# 9. One unbroken run over every row, with the same specs, against the resumed one.
whole = po.ModelBank(bank.specs)
fitted = whole.fit_predict(pl.read_parquet("features.parquet"))
same = all(fitted[s].tail(served.height).struct.field("pred_fwd").equals(served[s].struct.field("pred_fwd"))
           for s in ("robust", "ridge"))
print(f"the same predictions to the bit: {same}")
print(f"and the same state, byte for byte: {po.ModelBank.load('signal.state').save_bytes() == whole.save_bytes()}")
# Each fit's coefficient on flow over the last two days, against the planted 0.5.
for s in ("robust", "ridge"):
    coef = fitted[s].struct.field("coef")                          # [intercept, flow, stretch] on every row
    on_flow = coef.list.get(1).tail(8_000)
    print(f"{s}: coefficient on flow {on_flow[-1]:.3f} at the end, {((on_flow - 0.5) ** 2).mean() ** 0.5:.3f} from 0.5 "
          f"on average (root mean square); on stretch {coef.list.get(2)[-1]:.3f}")
```

```text
resumed on 6,458 rows, after 5,542 the state had read
the same predictions to the bit: True
and the same state, byte for byte: True
robust: coefficient on flow 0.483 at the end, 0.019 from 0.5 on average (root mean square); on stretch 0.012
ridge: coefficient on flow 0.537 at the end, 0.109 from 0.5 on average (root mean square); on stretch 0.012
```

**The resumed run writes what a run that never stopped writes.** The
fresh process read the state saved at noon, dropped the 5,542 rows the
state had already read, and wrote the predictions and the state an
unbroken run writes, to the bit. A state file loads on macOS, Windows and
Linux alike. Resumed on another operating system, its predictions match
within `1e-12 × (1 + |value|)`, since the platforms' math libraries can
round the last bit differently ([Resume from a saved
state](how-to/resume-from-a-saved-state.md#check-that-it-worked)).

**The robust fit stays near 0.5 through the jumps, and the ridge does
not.** A jump of 50 lands in the five-minute window of some fifty rows'
targets. The ridge weighs each of those rows by its squared error, so
every jump pulls its coefficient away, while `huber` down-weights the rows
it misses by far.

**Each part has a counterpart elsewhere. What one stream gives here is the
combination.** Polars and pandas each bound an EWMA by a count of rows
(Polars' `rolling_mean` with `weights=`, pandas' `rolling(n,
win_type="exponential")`), and neither bounds one by a span of time. Asked
for one, pandas refuses `win_type` on a window of time. In Polars, such an
EWMA is written by hand, as a weighted aggregation inside `rolling`. The
river library evaluates a model on targets that arrive late
(`progressive_val_score`'s `delay=`), but its exponentially weighted
statistics have no window of time. Here the bounded EWMAs, forward and
back, the embargo, the robust fit and the resume work together on one
stream, from files to files.

## Train gradient-boosted trees on files larger than memory

This task trains XGBoost and LightGBM on parquet files the process does
not hold, around an online model's prediction:

- **A tree fit in passes over the files:** `po.gbdt` builds each library's
  training object from files or a query, read a chunk at a time. For
  XGBoost, one pass places each feature's bins from a sample of the rows,
  and the passes after it put every row in its bin, so it keeps each row's
  bins and never the rows. LightGBM bins on its own sample, and holds the
  rows only when they fit in memory ([How `po.gbdt` builds training
  data](explanation/how-po-gbdt-builds-training-data.md)).
- **An online model's prediction as the trees' starting point**, the
  *margin*: XGBoost's `base_margin` and LightGBM's `init_score`. The
  online ridge follows a level that drifts, and the trees learn the
  nonlinear rest.

The recipe plants a target with a level that drifts upward, out of reach
of every training row by the test rows, plus `sin(2·x0)` and a step in
`x1`, plus noise of spread 0.3. No feature carries the level.

1. **Make up the rows in clock order, across five parquet files.**
2. **Run an online ridge on a clock over the files,** with its
   out-of-sample prediction beside each row as a column of its own,
   `pred_y` (`.online.unnest`).
3. **Keep the rows that have a prediction,** since every row needs a
   margin.
4. **Write them to a file with `sink_parquet`,** which runs the query and
   writes the result without holding it all in memory.
5. **Build XGBoost's training matrix with the prediction as its margin,**
   with `po.gbdt.xgboost_matrix(..., base_margin="pred_y")`.
6. **Train XGBoost with `xgb.train`.**
7. **Build LightGBM's dataset the same way,** with `init_score="pred_y"`.
8. **Train LightGBM with `lgb.train`.**
9. **Serve the test rows with each row's margin added back.**
10. **Compare the error with the online ridge's alone, and with the trees
    served without the margin.**

```python
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
import polars_online as po
import xgboost as xgb

# 1. Rows in clock order over five files, 20 a step over 1,000 steps, with the answer planted.
rng = np.random.default_rng(9)
steps, per = 1_000, 20
level = np.repeat(np.cumsum(rng.normal(0.0, 0.1, steps)) + 0.008 * np.arange(steps), per)   # drifts up
x0, x1 = rng.standard_normal((2, steps * per))
rows = pl.DataFrame({"t": np.repeat(np.arange(steps), per), "x0": x0, "x1": x1,
                     "y": level + np.sin(2 * x0) + (x1 > 0.5) + 0.3 * rng.standard_normal(steps * per)})
Path("rows").mkdir(exist_ok=True)
for k, part in enumerate(rows.iter_slices(4_000)):
    part.write_parquet(f"rows/part-{k}.parquet")

# 2 to 4. The online ridge's prediction beside each row that has one, written to a file.
online = po.spec.ewridge("online", targets=["y"], features=["x0", "x1"], clock="t", half_life=5, gap_cap=10)
(pl.scan_parquet("rows/*.parquet")
   .online.fit_predict([online]).online.unnest([online])        # 2.
   .filter(pl.col("pred_y").is_not_null())                      # 3. every row needs a margin
   .select("t", "x0", "x1", "y", "pred_y")
   .sink_parquet("predicted.parquet"))                          # 4.
# The rows before step 600 train the trees, and those from step 800 test them.
train = pl.scan_parquet("predicted.parquet").filter(pl.col("t") < 600)
test = pl.scan_parquet("predicted.parquet").filter(pl.col("t") >= 800).collect()
X, margin, y = test.select("x0", "x1").to_numpy(), test["pred_y"].to_numpy(), test["y"].to_numpy()

def rmse(pred):
    return float(np.sqrt(np.mean((pred - y) ** 2)))

# 5 and 6. XGBoost: bins from a sample in one pass, every row binned in the passes after it, then train.
matrix = po.gbdt.xgboost_matrix(train, label="y", features=["x0", "x1"], base_margin="pred_y")
booster = xgb.train({"tree_method": "hist", "max_depth": 4, "eta": 0.1}, matrix, 100)
# 7 and 8. LightGBM, the same way.
dataset = po.gbdt.lightgbm_dataset(train, label="y", features=["x0", "x1"], init_score="pred_y")
lgb_trees = lgb.train({"num_leaves": 15, "learning_rate": 0.1, "verbose": -1}, dataset, 100)

# 9. Served with each row's margin; and, for the comparison, XGBoost's trees without it.
served = booster.predict(xgb.DMatrix(X, base_margin=margin, feature_names=["x0", "x1"]))
forgot = booster.predict(xgb.DMatrix(X, feature_names=["x0", "x1"]))
# 10. The test error of each.
print(f"the online ridge alone: test RMSE {rmse(margin):.2f}")
print(f"XGBoost around it:      test RMSE {rmse(served):.2f}, and {rmse(forgot):.2f} served without the margin")
print(f"LightGBM around it:     test RMSE {rmse(lgb_trees.predict(X, raw_score=True) + margin):.2f}")
```

```text
the online ridge alone: test RMSE 0.81
XGBoost around it:      test RMSE 0.40, and 5.03 served without the margin
LightGBM around it:     test RMSE 0.40
```

**Together, the online ridge and the trees reach a test error of 0.40,
against the noise's spread of 0.3.** The ridge alone follows the level but
misses the curve in `x0` and the step in `x1`, at 0.81. The trees learned
only what the margin leaves of the target, so served without it they carry
none of the level, and miss by 5.03. Served with the margin, each library
reaches 0.40. [Train XGBoost or LightGBM on files larger than
memory](GBDT.md) gives every step of the build, and what to do when a
check fails.

**Both libraries train from data larger than memory on their own, given
an iterator you write, or for LightGBM a CSV or LibSVM text file.**
XGBoost reads a `DataIter` subclass whose `next` and `reset` you write,
and LightGBM an `lgb.Sequence` that answers any row or range of rows by
its index. Both take a margin. `po.gbdt` writes that part for you, from
parquet files, Arrow IPC files or a query, and the margin comes from an
online model whose every prediction is out-of-sample.

## Monitor a live model

This task watches a model whose targets arrive about 300 rows after its
predictions, and tests the watchers against changes planted at known rows:

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
0.5 to −0.5 at row 10,000. Each row's target becomes known 250 to 350
clock units after the row, and its noise has heavy tails.

1. **Simulate the features' levels with `po.sim.regimes`,** which also
   returns the regime each row is in.
2. **Take the levels' changes with `diff()`,** since a correlation
   detector reads changes, not levels.
3. **Add the target, and the time each target becomes known, as
   columns.**
4. **Write the stream to a parquet file.**
5. **Give the model `label_clock="known"`, `conformal=0.9`,
   `conformal_rate="auto"` and `emit_breaks=True`,** and a second spec
   with the default step, 0.05, for comparison.
6. **Add a `corrchange` spec on the features.**
7. **Run the query.**
8. **Read each interval's share of targets covered, and its mean
   half-width,** past the first five half-lives.
9. **Read the break test's flags** before the slope flipped and after it.
10. **Match each planted change of correlation to the first flag after
    it.**

```python
import numpy as np
import polars as pl
import polars_online as po

# 1. The features' levels, and the regime of each row.
sim = po.sim.regimes(2, states=[0.0, 0.7], transition=[[0.0, 1.0], [1.0, 0.0]],
                     n_blocks=4, rows_per_block=5_000, seed=4)   # the correlation takes turns every 5,000 rows
rng = np.random.default_rng(4)
n = sim["rows"].height
slope = pl.when(pl.col("t") < 10_000).then(0.5).otherwise(-0.5)   # x_2's slope flips at row 10,000
(sim["rows"].join(sim["truth_rows"].select("t", "state"), on="t")
   .with_columns(pl.col("x_1", "x_2").diff().fill_null(0.0))        # 2. the levels' changes
   # 3. The target, with heavy-tailed noise, and when each target becomes known.
   .with_columns(y=pl.col("x_1") + slope * pl.col("x_2") + pl.lit(rng.standard_t(3, n) / np.sqrt(3.0)),
                 known=pl.col("t") + pl.lit(rng.uniform(250.0, 350.0, n)))
   .write_parquet("live.parquet"))                                  # 4.

# 5. The model, its intervals and its break test; and the same model at the default step.
live = dict(targets=["y"], features=["x_1", "x_2"], clock="t", gap_cap=10.0, half_life=2_000.0,
            label_clock="known",                  # each row is learned when its target is known
            conformal=0.9)                        # an interval meant to cover 90% of the targets
model = po.spec.ewridge("model", **live, conformal_rate="auto", emit_breaks=True)
fixed = po.spec.ewridge("fixed", **live)          # the default step, 0.05
# 6. A watcher of the features' correlation.
watch = po.spec.corrchange("watch", features=["x_1", "x_2"], kind="sequential",
                           window_rows=300,       # the history the rows after it are monitored against
                           monitor_rows=2_000)    # the rows monitored after each history
# 7. The query.
out = pl.scan_parquet("live.parquet").online.fit_predict([model, fixed, watch]).collect()

# 8. Each interval's coverage and mean half-width, past the first five half-lives.
for name in ("model", "fixed"):
    lo, hi = (pl.col(name).struct.field(f) for f in ("lo_y", "hi_y"))
    covered, width = out.slice(10_000).select(pl.col("y").is_between(lo, hi).mean(), ((hi - lo) / 2).mean()).row(0)
    print(f"{name}: covered {covered:.3f} of the targets, half-width {width:.2f}")
# 9. The break test, against chi2(3)'s 0.01% value (two features and the intercept), before and after the flip.
flagged = out.filter(pl.col("model").struct.field("break_wald_y") > 21.1)["t"]
print(f"break test: {(flagged < 10_000).sum()} flags before row 10000, the first after it at row {flagged.filter(flagged >= 10_000).min():.0f}")
# 10. Each planted change of correlation beside the first flag after it.
flag = pl.col("watch").struct.field("flag")
starts = out.filter(flag & ~flag.shift(1, fill_value=False))["t"].to_list()
planted = out.filter(pl.col("state") != pl.col("state").shift(1))["t"].to_list()
for change in planted:
    hit = [f for f in starts if change <= f < change + 5_000]
    print(f"correlation changed at row {change:.0f}: " + (f"flagged at row {hit[0]:.0f}" if hit else "missed"))
print(f"flags no change explains: {sum(not any(c <= f < c + 5_000 for c in planted) for f in starts)}")
```

```text
model: covered 0.902 of the targets, half-width 1.59
fixed: covered 0.900 of the targets, half-width 2.51
break test: 0 flags before row 10000, the first after it at row 10436
correlation changed at row 5000: flagged at row 5110
correlation changed at row 10000: flagged at row 10098
correlation changed at row 15000: flagged at row 15056
flags no change explains: 0
```

**Both intervals cover 90% of the targets, and the adaptive one is
narrower, 1.59 against 2.51.** About 300 rows wait for their targets at any time. At the
default step, the interval's radius learns of each miss 300 rows late,
overshoots and undershoots, and is wider on average. `"auto"` takes
smaller steps the more rows wait. [Keep the coverage when targets arrive
late](how-to/put-an-interval-on-a-prediction.md#keep-the-coverage-when-targets-arrive-late)
measures the step on more streams.

**The break test flagged nothing before the slope flipped, and the
correlation detector found each planted change, with no false alarm.**
The break test needs the targets of the rows after the flip, which arrive
about 300 rows late, so it flagged the flip 436 rows after it. [Test a
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

## Measure correlations and fit curves on tick data

This task works on trades, where each instrument ticks at times of its
own:

- **Correlations from asynchronous ticks:** `po.stream.refresh_time` puts
  series that tick at their own times on *refresh times*, the times by
  which every series has ticked again, and a model tracks their moves'
  correlation there.
- **A local regression along any feature:** a fit whose clock is trade
  size, so its half-life is a bandwidth in shares. Sorted by size, with
  size as the clock, each trade is predicted from the trades learned
  before it: those at its own size and at nearby smaller sizes, weighted
  by how near they are. Sorting the trades by size holds them all while it
  sorts, so this part is for trades that fit in memory.

The recipe plants three prices that share one move, each adding a move of
its own as large, so the moves of any two correlate at 0.5. Each trade
prints away from its price by `0.05·√size` plus noise.

1. **Make up the three series' trades, each at times of its own, in one
   parquet file in clock order.**
2. **Put the series on refresh times with `po.stream.refresh_time`.**
3. **Take each series' change from one refresh time to the next.**
4. **Track the changes' correlations with an `ew_cov` spec.**
5. **Read each series' last price every 0.1 clock units, for
   comparison,** and correlate the changes there.
6. **Keep the trades of 40 shares or fewer, and sort them by size.** A
   bandwidth of two shares needs a few trades at every size, which the
   rarer, larger trades do not give.
7. **Fit `impact` on `size` with `size` as the clock,** with
   `half_life=2.0` shares and a solve on every row.
8. **Compare the local fit with the planted curve and with one straight
   line** at a few sizes.

```python
import numpy as np
import polars as pl
import polars_online as po

# 1. Three prices that share one move, each traded at times of its own, about once a clock unit.
rng = np.random.default_rng(1)
shared = np.cumsum(rng.normal(0.0, 0.1, 200_000))           # a hidden path, 0.01 clock units a step
names = ["AAA", "BBB", "CCC"]
frames = []
for s in names:
    # The series' price, plus a move of its own as large, and the times it trades.
    price = shared + np.cumsum(rng.normal(0.0, 0.1, 200_000))
    t = np.cumsum(rng.exponential(1.0, 2_100))
    t = t[t < 2_000]
    # Each trade's size, and how far it printed from the price.
    size = np.round(rng.lognormal(2.0, 1.0, t.size)) + 1        # shares, 1 to a few hundred
    impact = 0.05 * np.sqrt(size) + 0.02 * rng.standard_normal(t.size)
    frames.append(pl.DataFrame({"symbol": s, "t": t, "px": price[(t * 100).astype(int)], "size": size, "impact": impact}))
pl.concat(frames).sort("t").write_parquet("ticks.parquet")

# 2 to 4. The series on refresh times, their changes, and the changes' correlations.
refresh = po.stream.refresh_time(pl.scan_parquet("ticks.parquet"), series="symbol", names=names, clock="t", value="px")
moves = refresh.select("time_refresh", *(pl.col(f"{s}_value").diff().alias(s) for s in names))
comove = po.spec.ew_cov("comove", features=names, stats=["corr"], clock="time_refresh", gap_cap=10.0, half_life=float("inf"))
last = moves.online.fit_predict([comove]).collect()["comove"][-1]
pairs = [("AAA", "BBB"), ("AAA", "CCC"), ("BBB", "CCC")]
print("planted 0.5; on the refresh times: " + ", ".join(f"{last[f'corr_{a}_{b}']:.2f}" for a, b in pairs))
# 5. Each series' last price every 0.1 clock units, and the correlations of its changes there.
every = pl.DataFrame({"t": np.arange(0.0, 2_000.0, 0.1)})
for s, ticks in pl.read_parquet("ticks.parquet").partition_by("symbol", as_dict=True).items():
    every = every.join_asof(ticks.select("t", pl.col("px").alias(s[0])), on="t")
corr = every.select(pl.col(names).diff()).drop_nulls().corr()
print("read every 0.1: " + ", ".join(f"{corr[b][names.index(a)]:.2f}" for a, b in pairs))

# 6 and 7. A local fit along trade size: the trades up to 40 shares, sorted by size, and size as the clock.
along = po.spec.ewridge("along", targets=["impact"], features=["size"], clock="size", half_life=2.0, gap_cap=10.0,
                        max_rows_between_solves=1)
by_size = pl.scan_parquet("ticks.parquet").filter(pl.col("size") <= 40).sort("size", maintain_order=True)
curve = by_size.online.fit_predict([along]).online.unnest([along]).collect()
# 8. The local fit, the planted curve and one straight line through every trade, at a few sizes.
line = np.polyfit(curve["size"], curve["impact"], 1)
for at in (2, 8, 32):
    near = curve.filter(pl.col("size") == at)
    print(f"size {at:>2}: planted {0.05 * at ** 0.5:.3f}, local fit {near['pred_impact'].mean():.3f}, straight line {np.polyval(line, at):.3f}")
```

```text
planted 0.5; on the refresh times: 0.34, 0.35, 0.34
read every 0.1: 0.03, 0.02, 0.02
size  2: planted 0.071, local fit 0.070, straight line 0.092
size  8: planted 0.141, local fit 0.143, straight line 0.134
size 32: planted 0.283, local fit 0.285, straight line 0.301
```

**On refresh times the correlation is two thirds of the planted 0.5, and
read every 0.1 clock units it all but vanishes.** Read so often, most
moves are zero because a series has not ticked, so the moves of two series
almost never fall together. [Series that tick at their own
times](explanation/asynchronous-series.md) explains the fall, and [the two
ways out](explanation/asynchronous-series.md#the-two-ways-out) of it.


**The local fit follows the square-root curve that one straight line
cannot.** It is the same `ewridge` the other tasks fit, with a feature as
its clock ([Fit locally along a
feature](how-to/fit-locally-along-a-feature.md)).

**Each part has a counterpart elsewhere. What one pass gives here is the
combination.** Neither Polars nor pandas samples on refresh times, and
Python's `hfhd` package and R's `highfrequency` package do, on series held
in memory. The statsmodels library fits a local line along a feature
(`lowess`) on data held in memory too. Here the refresh times, the
correlations on them and the local fit run as queries over parquet files.

## Capabilities

Look a capability up to find the step that shows it and a guide that
covers it. For every model, function and parameter of
the library, one line each, see [the feature index](FEATURES.md).

| capability | what it gives you | shown in | guide |
|---|---|---|---|
| a target from a bounded EWMA that looks forward | the exponentially weighted mean of the rows after each row, up to a maximum distance on the clock, less the row's value | [deploy a signal](#build-a-forward-looking-signal-and-deploy-it), step 2 | [Make a target that looks ahead](how-to/make-a-target-that-looks-ahead.md) |
| a feature from a bounded EWMA that looks back | how far a value sits from its recent weighted mean, with rows older than the window dropped | [deploy a signal](#build-a-forward-looking-signal-and-deploy-it), step 2 | [Make features that look back over the stream](how-to/make-window-features.md) |
| `embargo` | each row predicted at its own clock value and learned only once its target is known, so no model learns from the future | [deploy a signal](#build-a-forward-looking-signal-and-deploy-it), step 4 | [Learn targets known late](how-to/learn-targets-known-late.md) |
| a half-life in time or in traded volume | a local fit that forgets on any sorted clock, with sessions and gaps measured in its units | [model selection](#automate-a-round-of-model-selection), steps 2 and 4 | [Handle sessions, gaps and steps back](how-to/handle-sessions-gaps-and-steps-back.md) |
| a two-stage regression | each first-stage fit's out-of-sample residual as a feature of the second stage, in one pass | [model selection](#automate-a-round-of-model-selection), steps 2, 3 and 6 | [Fit on factor residuals](how-to/fit-on-factor-residuals.md) |
| many combinations of columns, with multiple specs | every feature set at every half-life, in several specs, from one read of the rows | [model selection](#automate-a-round-of-model-selection), step 4 | [Grids, instances and slots](explanation/grids-and-instances.md) |
| one pass that detects changes of coefficients, selection and regimes | a break test, the grid point chosen on each row and a regime's age, beside the fits they watch | [model selection](#automate-a-round-of-model-selection), steps 4 to 7 | [Detect a break](how-to/detect-a-break.md) |
| model selection made out-of-sample | the grid point whose exponentially weighted out-of-sample error is lowest so far, chosen on each row from the rows before it | [model selection](#automate-a-round-of-model-selection), step 4 | [Choose among a grid's settings as the stream runs](how-to/choose-among-a-grid-online.md) |
| scores of an output too large to hold | sums per slot that add up across chunks, files and machines | [model selection](#automate-a-round-of-model-selection), step 6 | [Score an output too large to hold](how-to/evaluate-an-output-frame.md#score-an-output-too-large-to-hold) |
| robust regression | a fit that down-weights the rows it misses by far, such as jumps | [deploy a signal](#build-a-forward-looking-signal-and-deploy-it), step 4 | [The robust scale](explanation/robust-scale.md) |
| an exact resume | a saved state goes on in a fresh process as one unbroken run would, dropping the rows it has read | [deploy a signal](#build-a-forward-looking-signal-and-deploy-it), steps 6 to 9 | [Resume from a saved state](how-to/resume-from-a-saved-state.md) |
| a tree fit in passes over files larger than memory | XGBoost's and LightGBM's training objects, binned from files or a query a chunk at a time | [trees](#train-gradient-boosted-trees-on-files-larger-than-memory), steps 5 to 8 | [Train XGBoost or LightGBM on files larger than memory](GBDT.md) |
| an online model as the trees' margin | trees that learn what an out-of-sample online prediction leaves, and follow its level | [trees](#train-gradient-boosted-trees-on-files-larger-than-memory), steps 2 and 9 | [Train trees around an online model's prediction](GBDT.md#train-trees-around-an-online-models-prediction) |
| intervals when targets arrive late | a conformal interval whose step adapts to the targets still waiting | [monitor a model](#monitor-a-live-model), steps 5 and 8 | [Put an interval on a prediction](how-to/put-an-interval-on-a-prediction.md) |
| per-row break tests | a test on every row of whether a regression has broken | [monitor a model](#monitor-a-live-model), steps 5 and 9 | [Detect a break](how-to/detect-a-break.md) |
| a detector of the features' correlation | a flag where the features' correlations change | [monitor a model](#monitor-a-live-model), steps 6 and 10 | [Detect a break](how-to/detect-a-break.md#watch-the-features-correlations-with-corrchange) |
| a detector tested against known truth | a simulated stream with planted regime changes, and each change matched to the detector's flag | [monitor a model](#monitor-a-live-model), steps 1 and 10 | [Test a detector on known truth](how-to/test-a-detector-on-known-truth.md) |
| correlations from asynchronous ticks | series that tick at their own times, aligned on refresh times | [tick data](#measure-correlations-and-fit-curves-on-tick-data), steps 2 to 4 | [Align series that tick at their own times](how-to/align-series-that-tick-at-their-own-times.md) |
| a local regression along any feature | a fit whose clock is a feature, so its half-life is a bandwidth in that feature | [tick data](#measure-correlations-and-fit-curves-on-tick-data), step 7 | [Fit locally along a feature](how-to/fit-locally-along-a-feature.md) |
