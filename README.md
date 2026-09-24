# polars-online

> **GitHub project:** [github.com/hgilde/polars-online](https://github.com/hgilde/polars-online)

Online model fitting for [Polars](https://pola.rs): linear models,
streaming moments, clustering and regime detection, for data too large to
hold in memory at once. When the rows have a time order, a fit can also be
local, following the recent rows without refitting a window at every step.
Rust core, Python API, and a standalone command line.

| section | what it covers |
|---|---|
| [Introduction](#introduction) | [the idea](#the-idea) · [four words](#four-words) · [install](#install) · [a first fit](#a-first-fit) · [what you can rely on](#what-you-can-rely-on) |
| [How a bank sees a stream](#how-a-bank-sees-a-stream) | [what a spec names](#what-a-spec-names) · [time and decay](#time-and-decay) · [a local fit along any feature](#a-local-fit-along-any-feature) · [convergence without a decay](#convergence-without-a-decay) · [groups](#groups) · [weights](#weights) · [warm-up](#warm-up) · [labels that arrive late](#labels-that-arrive-late) · [nulls](#nulls-and-three-ways-to-hold-a-row-back) · [row order](#row-order-and-the-two-guarantees) |
| [Running a bank](#running-a-bank) | [as a query](#as-a-query-lfonlinefit_predict) · [in a loop](#in-a-loop-modelbank) · [output as Arrow](#output-as-arrow) · [outside Python](#outside-a-live-python-process) · [series that tick at their own times](#series-that-tick-at-their-own-times) |
| [Saving, loading and serving](#saving-loading-and-serving) | [save and load](#save-and-load) · [serving without learning](#serving-without-learning) · [what a state file holds](#what-a-state-file-holds) · [a state as JSON](#reading-a-state-without-this-library) |
| [Reading the fit](#reading-the-fit) | [coefficients](#coefficients) · [output field names](#output-field-names) · [the running sums](#the-running-sums-behind-a-fit) · [one row per finished group](#one-row-per-finished-group) · [correlation matrices](#reading-a-correlation-matrix) |
| [Diagnostics, selection and evaluation](#diagnostics-selection-and-evaluation) | [per-row diagnostics](#per-row-diagnostics) · [conformal intervals](#conformal-intervals) · [evaluating an output](#evaluating-an-output-frame) · [evaluating a stream too large to hold](#evaluating-a-stream-too-large-to-hold) · [simulated data](#data-whose-truth-is-known) |
| [Models](#models) | [linear models](#linear-models) · [moments and correlation](#moments-and-correlation) · [clustering and classification](#clustering-and-classification) · [sequential tests and regimes](#sequential-tests-and-regimes) |
| [Performance](#performance) | [throughput](#throughput) · [memory](#memory-which-calls-stream) · [tuning memory](#tuning-memory-with-polars-own-settings) · [chunk size](#chunk-size) · [parallelism](#parallelism) · [against scikit-learn](#against-scikit-learn) |
| [Scope and integrations](#scope-and-integrations) | [what this is not](#what-this-is-not) · [Pathway](#pathway) · [DuckDB and ADBC](#databases-duckdb-and-adbc) |
| [Versions, testing and development](#versions-testing-and-development) | [versioning and the Polars pin](#versioning-and-the-polars-pin) · [testing](#testing) · [development](#development) · [license](#license) |

## Introduction

### The idea

You describe one or more models: say, a ridge regression of a stock's
return on two signals, with a separate regression for every stock.
polars-online fits all of them in a single pass over your rows. Each row is
*predicted* first, from what the models have learned so far, and *learned
from* second, so no row's own outcome is ever in the number predicted for
it. The models keep what they have learned and never the rows, so the
stream can be far larger than memory.

Row order matters when a model forgets. With a decay, older rows count
less, so the rows must arrive in time order. Without one, the models that
solve or accumulate give the same answer in any order ([Convergence without
a decay](#convergence-without-a-decay) says which).

### Four words

This README uses four words of its own:

| word | meaning |
|---|---|
| **spec** | the description of one model: which model, which columns it reads, and how it treats time. `po.spec.ewridge(...)` builds one |
| **model bank**, or *the bank* | a set of specs fitted together over the same rows, and the Python object that holds them, [`ModelBank`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.ModelBank). *The bank* always means a model bank |
| **stream**, **chunk** | the rows in the order the bank reads them, and the pieces they arrive in. The bank takes one chunk at a time, and its results never depend on where one chunk ended and the next began |
| **state** | everything a bank has learned. Its size depends on the models, not on how many rows have passed, so a stream can be any length |

### Install

```sh
pip install polars-online      # or: uv add polars-online
```

| need | detail |
|---|---|
| Python | 3.12 or newer. One wheel per platform covers every CPython from 3.12 on, and CI runs the suite on each of 3.12, 3.13 and 3.14 |
| Polars | `polars>=1.34.0,<3`. The range is measured, not guaranteed: a weekly job and every release run the whole suite on the newest Polars ([Versioning and the Polars pin](#versioning-and-the-polars-pin)) |
| wheels | macOS (arm64, x86_64), Windows x64, and Linux (x64 glibc and musl, aarch64 glibc), on PyPI and on each GitHub release beside the command-line binaries |
| size | about 19 MB to download and 59 MB installed. The wheel carries its own copy of Polars' Rust half, so nothing beyond `polars` is needed at run time |
| optional | `numpy`, which only `ModelBank.gram()` and the `po.gram`, `po.corr` and `po.sim` helpers need |

From a checkout:

```sh
uv sync
uv run maturin develop --release -m crates/online-py/Cargo.toml
```

### A first fit

**One fit over every row.** Turn forgetting off, and a model that solves
converges to the batch fit over every row it has seen, in any row order.
The state is then a complete summary of those rows, which makes it worth
saving and serving. Fit it over a folder of parquet files too large for
memory, save it at the last row, then score new rows against it without
learning from them.

```python
import polars as pl
import polars_online as po

ols = po.spec.ewridge(
    "ridge",                                          # the spec's name; its output column is named after it
    targets=["ret"], features=["signal_a", "signal_b"],
    halflife=float("inf"),                            # no forgetting: every row counts the same, so this is
                                                      # ridge regression over the whole stream, in bounded memory
    group="stock_id",                                 # one separate regression per stock
)

(
    pl.scan_parquet("ticks/*.parquet")                # a Polars query over the files; nothing is read yet
    .online.fit_predict([ols], save_state="bank.state")    # the bank, inside the query
    .filter(pl.col("ridge").struct.field("n_eff") > 20)    # ordinary Polars on what comes out
    .sink_parquet("fitted.parquet")                   # runs the query, writing the result a chunk at a time
)

scored = pl.scan_parquet("today.parquet").online.predict("bank.state").collect()   # score; learn nothing
flat = scored.online.unnest([ols])   # pred_ret, resid_ret, n_eff, coef_ret_intercept, coef_ret_signal_a, ...
# Each spec adds one column, named after it, whose value in each row is a record of named fields:
# the prediction pred_<target>, the residual resid_<target>, the effective number of observations
# n_eff, the coefficients coef, and the diagnostics you switch on. unnest spreads them into columns.
```

**One fit that follows the recent rows.** Give the same spec a clock and a
`halflife`, and each row's weight halves every ten minutes of `ts`, a
timestamp column. The fit is now *local*: it describes the recent past, and
it moves from row to row. So the thing to read is the path the coefficients
took, which `coef_every=1` writes on every row, rather than the state they
ended on.

```python
local = po.spec.ewridge(
    "local", targets=["ret"], features=["signal_a", "signal_b"],
    clock="ts",                                       # a Datetime column, so the clock's parameters are durations
    halflife=pl.duration(minutes=10),                 # a row's weight halves every ten minutes of ts
    max_dclock=pl.duration(minutes=5),                # and a gap longer than five minutes decays as though it were five
    group="stock_id",
    coef_every=1,                                     # write the coefficients on every row, not once a chunk
)

betas = (
    pl.scan_parquet("ticks/*.parquet")
    .online.fit_predict([local])
    .online.unnest([local])                           # coef_ret_intercept, coef_ret_signal_a, coef_ret_signal_b
    .select("ts", "stock_id", "^coef_.*$")            # support_coef_ret_* sits beside them: each one's data share
    .collect()
)
# One row per input row: each stock's exposure to each signal, as it stood before that row.
# That is a time series -- plot it, difference it, or compare two stocks' exposures over a day.
path = betas.filter(pl.col("stock_id") == "b0").select("ts", "coef_ret_signal_a")
# null until the fit exists, then one value per row: how b0's return loaded on signal_a, over time.
```

The decay is the whole difference between the two. Without one, every row
counts forever and the saved state is the model. With one, the state holds
only the last few tens of minutes, so the path of the coefficients is the
output, and serving from the final state predicts with the most recent fit
alone.

### What you can rely on

| | |
|---|---|
| **honest predictions** | every row is predicted before its own outcome is learned ([Row order and the two guarantees](#row-order-and-the-two-guarantees)) |
| **any chunking** | one chunk or a thousand gives the same output, to the last bit |
| **any thread count** | each (spec, group) pair is fitted on its own thread, and the count changes only the speed: with 64 groups, 14 threads process 8× the rows per second of one ([Parallelism](#parallelism)) |
| **bounded memory** | memory is proportional to the models' state, not to the number of rows that have passed ([Memory](#memory-which-calls-stream)) |
| **named mistakes** | every keyword is checked against its type, and a missing column is reported with the spec that wanted it and the role it had there |
| **a tested claim** | about 650 Rust tests and 2,200 Python cases, against numpy and an independent implementation and on adversarial streams, run on macOS, Windows and Linux at every push ([Testing](#testing)) |

A bank runs three ways, with the same numbers from each: inside a Polars
query as above, in your own Python loop, or from a standalone command line
with no live Python at all ([Running a bank](#running-a-bank)). What it has
learned can be saved to a file and loaded back, to keep learning or to
score new rows without learning ([Saving, loading and
serving](#saving-loading-and-serving)). Residual spread, break detection, a
choice among several settings and running accuracy are computed from what
the models have already learned, so none of them sees the row it describes
([Diagnostics, selection and
evaluation](#diagnostics-selection-and-evaluation)).

## How a bank sees a stream

A model bank reads a stream of rows one chunk at a time. The parameters in
this section belong to every model. They say which columns to read, how
time and forgetting work, which rows belong to which model, how much each
row counts, and when a model has seen enough to report. The
[`polars_online.spec`](https://hgilde.github.io/polars-online/spec.html)
reference gives each one's units and default.

### What a spec names

```python
spec = po.spec.ewridge(
    "ridge",
    targets=["y"],                 # at least one; the targets of one spec share the features' running sums
    features=["x0", "x1", "x2"],   # numeric columns of any width, Decimal and Boolean included; read as 64-bit floats
    add_intercept=True,            # the default: the fit has a level of its own
    clock="t", halflife=600.0, max_dclock=300.0,
)
# A text column in either list is refused rather than silently read as nulls.
# Columns the spec does not name pass through to the output untouched.
```

Two more parameters shape the output rather than the input: `coef_every`,
under [Coefficients](#coefficients), and `label_delay`, under [Labels that
arrive late](#labels-that-arrive-late).

### Time and decay

A model forgets. Each row's weight halves every `halflife` along a
*clock*, a column that says how far apart two rows are. At each row, the
weight of everything learned so far is multiplied by
`λ = 0.5 ** (Δclock / halflife)`, where `Δclock` is the clock's step from
the previous row.

**The clock's type decides how every clock parameter is written.** A
timestamp carries its own unit, so its parameters are durations. A number
carries none, so its parameters are numbers of whatever it counts.

| the clock | a clock parameter is | for example |
|---|---|---|
| a `Datetime`, `Date` or `Duration` column | a duration | `halflife=pl.duration(minutes=10)` |
| a numeric column, such as seconds or cumulative traded volume | a number of the column's own units | `halflife=600.0` |
| none | a number of rows | `halflife=100`: a row 100 rows back counts half as much as the latest |

A duration is written three ways: `pl.duration(minutes=10)`,
`timedelta(minutes=10)`, or polars' duration text `"10m"`. The clock
parameters are `halflife`, `max_dclock`, `min_backwards_jump`,
`session_gap` and `label_delay` below, and a model's `window`, `solve_every`
and its own halflives. The
[`polars_online.spec`](https://hgilde.github.io/polars-online/spec.html)
reference lists every one.

A stream from a market brings three more things, and the spec has a
parameter for each:

| the stream has | the spec does |
|---|---|
| a **session** boundary, such as the start of a trading day, where the clock's jump is not elapsed time | applies a step you choose, `session_gap`, in place of the jump |
| a **gap** in the clock, such as an hour with no rows | caps each step at `max_dclock`, so a quiet hour does not age the fit like a busy one |
| a clock that **resets** or runs backwards between sessions | follows a policy, `on_clock_reset` |

```python
timed = po.spec.ewridge(
    "timed", targets=["y"], features=["x0", "x1"],
    clock="ts",                           # a Datetime column, so every clock parameter is a duration
    halflife=pl.duration(minutes=10),     # a row's weight halves every ten minutes of ts
    max_dclock=pl.duration(minutes=5),    # the most the clock may step between two rows a model learns from;
                                          # required with a clock
    on_clock_reset="max",                 # a backwards clock: "max" (the step is max_dclock), "zero",
                                          # "reset_state", or "error"
    # A backwards jump smaller than min_backwards_jump (default max_dclock) is refused whatever
    # the policy, and the bank is untouched: adjacent rows are never further apart than the cap
    # and a session is longer, so that is a late row, not a boundary. 0 switches the check off.
    session="session",                    # a column whose value changes at a session boundary ...
    session_gap=pl.duration(minutes=1),   # ... and the clock step to apply there, at most max_dclock;
                                          # "reset" starts the model over
)
# halflife=inf turns forgetting off. A list of halflives fits one model per value.
# max_dclock=0 turns forgetting off; max_dclock=inf removes the cap. The cap also bounds the step
# a run of skipped rows hands the row after them, however long the run.
# ewridge only: session_shrink= and long_halflife= pull the fit partway back, at a session
# boundary, toward a twin that forgets more slowly.

counted = po.spec.ewridge(
    "counted", targets=["y"], features=["x0", "x1"],
    clock="t",                 # a numeric column, so every clock parameter is a number of its units
    halflife=600.0,            # a row's weight halves every 600 units of t (or lam=, the weight kept per unit)
    max_dclock=300.0,
    session="session", session_gap=60.0,
)
```

**A clock parameter of the other kind is refused.** A plain number on a
`Datetime` clock would silently take the column's storage unit, so
`halflife=600` on a microsecond column would mean 600 microseconds. A
duration on a numeric clock has nothing to measure it against. The bank
refuses either before it reads a row, naming the column, the parameter and
the fix. It also refuses a spec that mixes the two kinds, and a duration
finer than the clock can act on, such as `max_dclock="12h"` on a `Date`
clock, which moves in days. `0` and `inf` mean the same in every unit, so
they may stay numbers.

A temporal clock is read in its own integer nanoseconds, so the same
instants stored in milliseconds, microseconds or nanoseconds give the same
numbers, and a time zone changes nothing. A model reads only the gap
between consecutive rows, and the bank takes that gap in integer
nanoseconds before it becomes seconds, so a nanosecond timestamp keeps its
nanoseconds whatever the stream's age. A clock must lie between the years
1677 and 2262, the range nanoseconds in a 64-bit integer cover. Where a
clock quantity reaches an output, it is in seconds: `holt`'s trend is per
second, and `summary()` gives the clock's range as seconds since 1970. A
huge finite halflife is
not `inf`: `halflife=1e12` still forgets, and the schedules keyed to the
halflife scale with it. Say `inf` for no forgetting.

### A local fit along any feature

The clock need not be a time. Sort the frame by one of its own feature
columns and name that column as the clock. `halflife` is then a bandwidth
in that feature's units, and each row is fitted on the rows before it,
weighted by `0.5 ** (Δx / halflife)`: an exponential kernel. The result is a
local regression, computed in one pass from running sums that do not grow,
where the usual method refits a window of rows at every point.

```python
curve = df.sort("x0")
local = po.spec.ewridge(
    "local",
    targets=["y"],
    features=["x0"],
    clock="x0",                 # the clock is a feature: the decay is a kernel in it
    halflife=0.5,               # the bandwidth, in x0's own units
    max_dclock=1.0,             # a wider gap decays as if it were this wide
    max_rows_between_solves=1,  # refit at every row
    min_periods=10.0,
)
fitted = po.ModelBank([local]).fit_predict(curve).unnest("local")
# pred_y  is the fitted curve, read at each row's own x0
# coef    is the line through that row's neighbourhood
```

These are the numbers a batch fit gives: a kernel-weighted least squares
recomputed from scratch at every row agrees with them to 1e-12.

The kernel is one-sided, since a row is fitted on the rows before it and
never after, so the fit follows a curve with a lag. The bandwidth trades
that lag against noise. On `sin(x)` at a bandwidth of 0.25 the fit sits
0.08 from the truth, where the best straight line sits 0.39; at a bandwidth
of 1.0 it sits 0.29, most of the way back to the line. `features` need not
include the clock column: with other features, the same fit is a
regression whose coefficients move along the clock. Every model takes a
clock, not only `ewridge`.

### Convergence without a decay

With decay off, `halflife=inf` or `lam=1.0`, a model that *solves* or
*accumulates* converges to the batch fit it defines over every row it has
seen, and row order does not reach that fit at all: forwards, backwards or
shuffled gives the same coefficients. A model that *steps*, *filters* or
*tests* depends on the order whether or not decay is on. [The model
table](#models) says which model learns which way, and each model's own
section says what it converges to.

Either way, memory is proportional to the model's state, the running sums
it keeps, and not to the number of rows that have passed. So the frame
never has to fit in memory, and a stream of any length fits in the same
state. `solve_every` makes a solving model compute its coefficients less
often than every row, when an exact fit at every row is not needed; they
are then at most that far out of date.

### Groups

`group` names a column, and the bank keeps one separate model per distinct
value of it, all fitted in the same pass over the stream. Every other
parameter, the clock included, applies within the group.

```python
per_stock = po.spec.ewridge(
    "per_stock", targets=["y"], features=["x0", "x1"], clock="t", halflife=600.0, max_dclock=300.0,
    group="stock_id",           # one model per distinct value of this column
    group_close="monotone",    # or "session": when a group is finished, write its running sums out as
)                              # one row and free its memory -- read them with bank.closed_groups()
```

`group_close` is what keeps a bank's memory bounded when new group values
never stop appearing ([One row per finished
group](#one-row-per-finished-group)). Otherwise a group lives as long as
the bank, and a long-running bank can drop the ones that have gone quiet
([In a loop](#in-a-loop-modelbank)).

### Weights

```python
weighted = po.spec.ewridge(
    "weighted", targets=["y"], features=["x0", "x1"], clock="t", halflife=600.0, max_dclock=300.0,
    weight="w",                # a column of row weights: a row of weight w counts as w observations would
)                              # weight 0 is legal: the row is scored, the clock advances, nothing is learned
```

`n_eff`, under [Warm-up](#warm-up), counts weight rather than rows.
[Nulls, and three ways to hold a row back](#nulls-and-three-ways-to-hold-a-row-back)
says when weight `0` is the right one of the three.

### Warm-up

A model should not report a number it is not yet informed enough to give.
Two settings say what "informed enough" means, each as the intent rather
than as a number that needs a formula in your head
([docs/WARMUP-AND-CONVERGENCE.md](docs/WARMUP-AND-CONVERGENCE.md)):

| setting | default | withholds a prediction while | read from |
|---|---|---|---|
| `max_error_inflation` | `sqrt(2)` | estimation error would inflate the prediction's error over the noise floor by more than this factor | `error_inflation = sqrt(1 + edf / n_kish)`: the effective degrees of freedom the fit used, over Kish's effective sample size behind it |
| `min_settled_frac` | `0`, off | the decay window is less full than this fraction of its steady state | `settled_frac = 1 − 2^(−T / halflife)`, with `T` the decay time the model has seen: `0.5` at one halflife, `0.75` at two, whatever the row rate |

```python
warm = po.spec.ewridge(
    "warm", targets=["y"], features=["x0", "x1"], clock="t", halflife=100.0, max_dclock=300.0,
    max_error_inflation=1.1,    # withhold while estimation error would add more than 10% over the noise floor
    min_settled_frac=0.5,       # and until the decay window is half full: one halflife of history
    emit_error_inflation=True,  # also each row's own ratio, against the row's own features
)                               # the model learns from every row either way
rows = po.ModelBank([warm]).fit_predict(df).unnest("warm")
rows.select("pred_y", "settled_frac", "withheld_reason", "error_inflation_y")
# settled_frac        how full the decay window is
# withheld_reason     why pred_y is null, or null when it is not: below_min_settled_frac, below_min_periods
#                     or above_max_error_inflation, in that order of precedence
# error_inflation_y   the row's leverage against the fit: high on a row leaning on a direction the data
#                     never showed, at one triangular solve a row
# support_coef        beside coef: each coefficient's data share, 1 - ridge * (S^-1)_jj; a duplicated pair
#                     of features reads 0.5 each
```

`max_error_inflation` tracks the model: add a feature and the gate moves.
It reads Kish's count, so one row carrying a hundred times the weight of
the others counts as barely one. Only `ewridge` computes it; the other
models keep `min_periods`, below. `min_settled_frac` is off by default
because a mean-form fit is unbiased from its first row when the process is
stationary. Set it when the halflife was chosen to average over regimes or
seasons that a shorter history would not represent. `summary()` carries
the same readings per group, and a `ReadinessWarning` names, once, a
coefficient more ridge than data or a noise gate the stream has settled
below.

`min_periods` is the older, absolute floor, in `n_eff` units: every output
is null until `n_eff` reaches it. A list gives one threshold per target.
Every model but `ewridge` still defaults it to one observation per
unknown.

`n_eff` is the *effective number of observations*: the total weight behind
the state that produced this row's prediction, after forgetting and before
the row's own update. It is `0` on a stream's first row. It runs one behind
the row count while nothing is forgotten, and settles at `1 / (1 − λ)` once
forgetting balances arrival. It means the same thing in every model, so one
`min_periods` means the same thing across a bank. A model that keeps a
weight per target checks each target against that target's own weight, the
weight of the rows it was present on, so an often-null target reports later
than the others. The `n_eff` field is the shared weight either way, and
[`polars_online.spec`](https://hgilde.github.io/polars-online/spec.html)
says which models keep one.

### Labels that arrive late

A target that is a forward quantity, such as the next five minutes'
return, is not known at the row it sits on. A stream that learns it there
hands the model that much of the future before it predicts the rows in
between. Every out-of-sample number after that is contaminated, and with a
feature that is correlated with its own past, even a pure-noise column
starts to look predictive. `label_delay` holds each label back until it
would really have arrived:

```python
spec = po.spec.ewridge("fwd", targets=["ret_5m"], features=["x0", "x1"],
                       clock="ts", max_dclock="1h", halflife="30m",
                       label_delay="5m")      # the return takes five minutes to be known
# Each row is scored where it sits and learned from five minutes later. Everything
# downstream of the label moves with it -- the prediction, sigma, resid_z, the metrics,
# break detection, the conformal interval, n_eff and min_periods all see only labels
# that had really arrived.
#   - the clock is the model's own (capped by max_dclock, skipped rows' time included),
#     which is what makes the release depend on the clock alone and survive any chunking
#   - with no clock column, one unit is one accepted row: label_delay=20 is twenty rows
#   - a reset drops the rows still waiting; a session change releases them in order
#   - rows still waiting when the stream ends are never learned from
#   - the waiting rows live in the state and are saved with it: one row's values per
#     row inside the delay, per group
```

[`po.prep.embargo`](https://hgilde.github.io/polars-online/prep.html#polars_online.prep.embargo)
writes the same delay out as data, for when it has to be visible in the
frame, or for an engine other than this one:

```python
doubled = po.prep.embargo(lf, clock="t", delay=300.0)   # every row twice: a zero-weight copy to score at t,
                                                          # and a copy to learn from at t + delay, in clock order
```

The built-in delay agrees with the doubled stream field by field, to the
bit, except for `resid_quantiles`, `emit_autocorr` and `emit_drift`. Those
three take no row weight, so a zero-weight copy feeds them as much as its
learning copy does: the doubled stream lands every residual in them twice,
where `label_delay` lands it once.

### Nulls, and three ways to hold a row back

A null in any feature, or in the weight, skips the row: its outputs are
null, no update happens, and the clock still advances. A null in one
target still writes that target's `pred`, leaves its `resid` null, and
skips only that target's update. NaN, ±inf and any magnitude above `1e100`
count as null, so sentinel values never reach a model.

Three ways keep a row from teaching a model, and they differ:

| | the row is scored | the clock advances | the fit moves | `n_eff` | use it to |
|---|---|---|---|---|---|
| **weight `0`** | yes | yes | no | keeps decaying, so a long stretch can fall below `min_periods` | keep a row's place in the stream |
| **a null target** | yes | yes | as the model decides; its section says | counts the row, so it does not decay toward `min_periods` | leave a label out |
| **`predict`** | yes | no | no | frozen | serve |

In `ewridge` and `lasso` under the default `target_gaps="own_rows"`, a
null target's fit does not move: its sums cover only the rows it is
present on, and a row without it only ages them. Under
`target_gaps="pairwise"` the feature sums take the row and the target's
sums do not, so the coefficients wander with the feature noise. `predict`
is also the fast path: `ewridge` scores at 1.8–2.9× its learning
throughput.

### Row order and the two guarantees

- **Predictions are out-of-sample.** Every row is predicted from the state
  as it stood before the row's own target was learned, so nothing can leak
  a row's outcome into its own prediction.
- **Chunk invariance.** One chunk or a thousand, with or without a save and
  resume in the middle, gives bit-identical output. The one exception is
  `coef`, which is a reporting cadence: it is written every `coef_every`
  rows and on each group's last row in every chunk, so smaller chunks
  report it more often.

Both rest on something the caller supplies: a fixed row order. A model
learns in row order, so a query whose row order Polars does not guarantee
is a different model each time it runs. A `LazyFrame` handed to a bank runs
through Polars' streaming engine, where a step without an order guarantee
can deliver a different order than `lf.collect()` gives. That was measured:
`collect()` kept the input order and `collect_batches()` did not. Give each
such step its guarantee:

| a query step that may reorder rows | give it |
|---|---|
| `join` | `maintain_order="left"` |
| `group_by` | `maintain_order=True` |
| `unique` | a sort after it |
| anything else | a sort before the bank |

A query with such a step raises `OrderNotGuaranteedWarning` when it is
handed to a bank, naming the step.

#### How a bank detects rows out of order

A bank checks the order it is given in three places. The query check
warns. The other two refuse a chunk before any of its rows is learned.

| check | what it reads | when it finds disorder |
|---|---|---|
| the query | the plan handed to the bank: a `join`, `group_by` or `unique` whose order Polars does not guarantee, unless a `sort` sits above it | raises `OrderNotGuaranteedWarning`, naming the step and its fix; the run goes on |
| the clock | each group's clock, row by row, for every spec with a `clock` | refuses the chunk on a backwards step the settings below do not allow |
| the group keys | with `group_close="monotone"`, the keys in the column's own order | refuses the chunk on a key below one already closed, since a closed group cannot reopen |

**The clock is checked group by group.** Each spec keeps one clock per
group, so groups may interleave freely. Only the rows of one group need to
be in clock order. Equal clock values are a gap of zero. A row with a null
feature is skipped, but its clock is still checked. On a temporal clock the
comparison is exact, in integer nanoseconds.

**The size of a backwards step decides what it means:**

| a step back of | is read as | and the bank |
|---|---|---|
| less than `min_backwards_jump` | rows out of order: adjacent rows are never further apart than `max_dclock`, and a session is longer | refuses the chunk, whatever `on_clock_reset` says |
| at least `min_backwards_jump` | a boundary, such as a clock that restarts | follows `on_clock_reset`: `"max"`, the default, takes a gap of `max_dclock`, `"zero"` a gap of 0, `"reset_state"` restarts the model, and `"error"` refuses the chunk |
| any size, on a row whose `session` value changes | a new session, not a step | applies `session_gap`, or restarts the model under `session_gap="reset"` |

`min_backwards_jump` defaults to `max_dclock`. Under an infinite
`max_dclock` it defaults to 0, which switches the check off, since no step
is then too large for two adjacent rows. Set it to 0 to switch it off
yourself.

**A refused chunk leaves the bank as it was.** The bank runs the clock of
every group of every spec over the chunk before it learns any row. So one
late row in one group changes nothing, and the corrected chunk can be fed
again. The error names the spec, the clock column, the row and the size of
the step. It also says which setting refused the step, and how to change it:

```text
spec "m": clock column "t" goes backwards by 30 at row 6, less than min_backwards_jump = 60
(which defaults to max_dclock: adjacent rows are never further apart, and a session is longer)
-- out-of-order rows, not a session boundary; the bank was not updated. Sort each group by the
clock, or add a `session` column if these are real boundaries. ...
```

**Scoring does not refuse a late row.** `predict` learns nothing, so a
frame scored against a bank may start a little before the last clock the
bank learned. Only `on_clock_reset="error"` refuses a backwards step there.

**The summary counts what a policy absorbed.** `bank.summary()` reports
`clock_backwards`, the rows whose clock fell below the previous row's
within a session, and `resets`, the rows where a stream restarted. A
nonzero `clock_backwards` under `"max"` or `"zero"` is disorder the
settings let through.

**The query check is best-effort.** It reads the plan's `explain` text,
and walks the plan only when that names a join, an aggregation or a
`unique`. A step under a `sort` counts as ordered. A join with
`maintain_order="left"` is followed on its left side only. A plan Polars
cannot serialize passes without a warning, and the check never fails a run.
`ModelBank.fit` does not warn when every spec is an accumulator with no
decay, since those sums reach the same state in any order. For a plan
whose order you know, silence the warning with
`warnings.simplefilter("ignore", po.OrderNotGuaranteedWarning)`.

## Running a bank

A bank runs three ways, and each gives the same numbers: inside a Polars
query, in your own Python loop, or from a standalone command line with no
live Python at all. The query comes first, because it is Polars' own idiom.

### As a query: `lf.online.fit_predict`

A Polars `LazyFrame` is a query that runs only when you ask for its result.
[`lf.online.fit_predict(specs)`](https://hgilde.github.io/polars-online/namespaces.html#polars_online._frame.LazyFrameOnlineNamespace.fit_predict)
puts a model bank inside one. When the query
runs, its rows go through a bank that starts with nothing learned,
`chunk_rows` rows at a time, and every step after the bank is ordinary
Polars. `collect()` gives the result as one frame, `sink_parquet()` writes
it to a file without holding it all, and `collect_batches()` gives it a
chunk at a time.

```python
(
    pl.scan_parquet("ticks/*.parquet")
    .online.fit_predict([spec], chunk_rows=100_000)       # a bank with nothing learned yet; every run starts from the same place
    .filter(pl.col("ridge").struct.field("n_eff") > 100)  # after the bank: filters what comes out, never what the bank learns from
    .select("ts", "stock_id", "ridge")                     # Polars reads only these columns (and the specs') from the files
    .sink_parquet("fitted.parquet")                       # runs the query; memory is state + one chunk, however long the files
)
# "ridge" is one column whose value per row is a record of named fields:
#   {pred_y__r0.000001, resid_y__r0.000001, pred_y__r0.1, resid_y__r0.1, n_eff, coef}

lf.online.fit_predict([spec]).head(5).collect()                           # learns from the first 5 rows and no more
lf.online.predict(bank).collect()                                         # score against an existing bank; learn nothing
lf.online.fit_predict(load_state="bank.state", save_state="bank.state")   # continue from a saved state; save again at the last row
```

**`save_state` writes when the run reaches the last row**, whole or not at
all, with the same bytes a `ModelBank` would write. A run abandoned early,
or ended by an error inside the bank, leaves the file untouched. An error
in a later step of the query does not stop the bank, so the state is
written although the query failed
([docs/STATE-WORKFLOW.md](docs/STATE-WORKFLOW.md) has the measurements).

**Filter after the bank, not before it, unless the model must skip those
rows.** A filter after the bank never changes what the bank learns from,
and the query still runs a chunk at a time. A filter before the bank makes
Polars hold several blocks of each parquet file per thread: 2.5 GB at 12M
rows, against 0.78 GB for the same filter after
([docs/PERFORMANCE.md](docs/PERFORMANCE.md) §11). When the model must not
learn from some rows, give them weight `0` instead of filtering them out.
They still flow through and come out scored, and no gap opens in the clock:

```python
(
    lf.with_columns(pl.when(pl.col("venue") == "X").then(1.0).otherwise(0.0).alias("w"))
    .online.fit_predict([po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"],
                                         clock="t", halflife=600.0, max_dclock=300.0,
                                         weight="w")])
    .sink_parquet("fitted.parquet")
)
```

`df.online.fit_predict(specs)` does the same for a `DataFrame` already in
memory. `po.fit_predict(frame, ...)`,
[`po.predict(frame, bank)`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.predict)
and [`po.unnest(frame, specs)`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.unnest)
are the same calls as plain functions, for a type checker, which cannot see
a registered namespace.

### In a loop: `ModelBank`

```python
spec = po.spec.ewridge(
    "ridge",
    targets=["y"], features=["x0", "x1", "x2"],
    clock="t", halflife=600.0, max_dclock=300.0,
    group="stock_id", ridge=[1e-6, 0.1], standardize=True,
)
bank = po.ModelBank([spec])                # nothing learned yet

for chunk in lf.collect_batches():        # the files, one chunk at a time; the whole stream is never in memory
    out = bank.fit_predict(chunk)         # the chunk's columns, plus one column per spec
    ...

bank.save("bank.state")                    # written whole or not at all: a temporary file, then a rename

repr(bank)        # ModelBank(['ridge'], groups=4, rows_seen=400)
bank.specs        # the spec dicts back, as their builders made them -- a copy, read-only
bank.groups()     # one row per (spec, group):
                  #   ┌───────┬───────┬────────────────┬────────────┐
                  #   │ spec  ┆ group ┆ rows_processed ┆ last_clock │
                  #   │ ridge ┆ b0    ┆ 100            ┆ 396.0      │
                  #   │ ridge ┆ b1    ┆ 100            ┆ 397.0      │
                  #   └───────┴───────┴────────────────┴────────────┘

# Groups live until dropped -- a long-running bank forgets the ones that have gone quiet:
stale = bank.groups().filter(pl.col("last_clock") < now - 30 * 86400)
bank.drop_groups(stale["group"])           # they start over if they reappear
```

A bank is one ordered stream, so it is not for two threads at once. A call
that finds it busy on another thread raises `RuntimeError` rather than
interleave the two. `predict` learns nothing, and may run from any number
of threads.

### Output as Arrow

For a consumer that is not Polars, the same output comes out through the
Arrow PyCapsule interface:

```python
structs = po.ModelBank([spec]).fit_predict_arrow(df)    # one per spec, as Arrow
out = df.with_columns([pl.Series(s) for s in structs])  # or any reader of __arrow_c_array__
```

Each struct is one spec's output, exposing `__arrow_c_array__`, and its
values are `fit_predict`'s field for field and null for null; only the way
out differs. A Polars `Series` crosses on py-polars' private methods, which
is why this package measures a Polars range rather than promising one. The
capsule interface is an Arrow specification instead, so a consumer that
reads `__arrow_c_array__` takes the result directly. A consumer that wants
the *stream* interface takes it through a `Series` first. DuckDB is one: on
duckdb 1.5.5 it refuses an `ArrowStruct` and accepts `pl.Series(s)`,
because `__arrow_c_stream__` is the dunder it looks for, and a spec's
output arrives table-shaped, one column per field. Exporting hands the
buffers to the consumer, so each struct is read once, and says so if asked
twice. `predict_arrow` is the same for `predict`.

### Outside a live Python process

A scheduled job, or a deployment with no Python at all, runs the same bank
from a file to a file: [docs/RUNNER.md](docs/RUNNER.md) has the standalone
`online` command line. Same specs, same state file, same numbers.

### Series that tick at their own times

Two series observed at different instants cannot be correlated directly. A
fine common grid pushes the correlation toward zero, the Epps effect, and
filling values forward invents observations that were never made.
[`po.prep.refresh_time`](https://hgilde.github.io/polars-online/prep.html#polars_online.prep.refresh_time)
puts them on the grid Barndorff-Nielsen, Hansen, Lunde and Shephard
defined: a point wherever **every** series has ticked at least once since
the last point, each carrying its last observed value.

```python
from polars_online import prep

grid = prep.refresh_time(ticks,                  # long input: one row per tick, the series named in a column
                         series="symbol", names=["AAA", "BBB", "CCC"],
                         time="t", value="px").collect()
# one row per grid point:
#   time_refresh        the grid point
#   AAA_value, ...      each series' last observed value at that point
#   n_obs_AAA, ...      ticks of that series since the previous point: the staleness of its value
#   retained_fraction   how much of the data survived -- read this before trusting a correlation
```

The grid runs at the pace of the slowest series, so a fast one loses most
of its ticks, and `retained_fraction` says how many. `pairs=True` runs an
independent two-series grid for each pair instead, which keeps far more
when one series is slow. Rows must be in time order: a backwards time is an
error naming the row, and nothing is interpolated. The output looks
synchronous and is not. Each value is up to one of its own inter-tick
intervals old, and the series with the largest `n_obs` is the one holding
the grid up.

## Saving, loading and serving

What a bank has learned, the running sums of every (spec, group), can be
saved to one file, written whole or not at all, and loaded back. A bank
object saves with `bank.save(path)` and loads with `po.ModelBank.load`, a
query takes `save_state=` and `load_state=`, and the command line takes
`--save-state` and `--resume`. The file is the same whichever wrote it, to
the byte. [docs/STATE-WORKFLOW.md](docs/STATE-WORKFLOW.md) walks the whole
workflow, fit, save, serve and learn on, with what each step guarantees.

### Save and load

```python
# From a bank object
bank.fit_predict(df)
bank.save("bank.state")                               # written whole or not at all: a temporary file, then a rename
bank = po.ModelBank.load("bank.state", specs=[spec])  # specs= checks the file holds this model, not another
bank.fit_predict(today)                               # keep learning: the state moves
scored = bank.predict(today)                          # serve: score the rows, learn nothing

# From a query
lf.online.fit_predict([spec], save_state="bank.state").sink_parquet("fitted.parquet")             # fit, then save at the last row
lf.online.fit_predict(load_state="bank.state", save_state="bank.state").sink_parquet("more.parquet")   # continue, then save again
served = lf.online.predict("bank.state").collect()                                                # serve from the file

# The same file, in memory rather than on disk -- for a checkpoint that lives somewhere else:
blob = bank.save_bytes()
bank = po.ModelBank.load_bytes(blob, specs=[spec])
```

Loading names the problem it meets: `FileNotFoundError` when there is no
file yet, and `ValueError` for a file that is not a bank, was written by a
newer version, or holds a different model.

### Serving without learning

```python
scored = bank.predict(today)
# Row i of `scored` carries what fit_predict would have reported had it been the next row
# of the stream -- pred, n_eff, sigma, resid_z, selection, metrics, field for field -- and
# every row is scored from the same state: nothing moves.
#   - the target column may be absent; resid is then null
#   - the weight column is not read
#   - a group the bank has never seen scores null
#   - the stream's session and clock rules still hold
```

### What a state file holds

A saved bank can be read by something that knows nothing about it. It needs
no specs, no configuration and no data; the file carries what it needs:

```python
bank = po.ModelBank.load("bank.state")   # no specs=: the file is enough

bank.specs                # every spec back, as the dict its builder made -- a copy, read-only:
                          # the bank runs the state it was built from, so a list edited on the
                          # Python side would only ever mislabel what coef() reports
bank.groups()             # spec, group, rows_processed, last_clock
bank.output_fields()      # {'ridge': ['pred_y', 'resid_y', 'n_eff', 'coef'], ...}
bank.rows_seen()          # rows fed, over every chunk and group
bank.solve_failures()     # per spec, per group: solves that needed jitter or kept the previous fit

# Four more tables: how the fit is doing, and what it was trained on.
# Each returns every spec by default with `spec` as the first column, and the columns are the
# same for every spec, so banks from different runs stack with a plain concat.
bank.last_row()           # the output row of the last row each group learned from
bank.coef()               # one row per coefficient, with the term it belongs to         (Coefficients, below)
bank.summary()            # per group: rows fed, learned, skipped, and the clock's range
bank.describe()           # per input column per group: count, nulls, mean, std, min, max
```

The last row and the two tables about what the bank was fed, in more
detail:

```python
bank = po.ModelBank.load("bank.state", specs=[spec])
last = bank.last_row("ridge")    # one row per group: spec, group, pred_y__r0.000001, ..., n_eff, coef
                                 # -- the fit_predict row field for field: pred, sigma, the metrics and the
                                 # interval when the spec asks for them, n_eff, and coef when that row carried it

# Fit many models, save each, and comparing them is one concat over the files:
from pathlib import Path
table = pl.concat(
    [po.ModelBank.load(f).last_row() for f in sorted(Path(".").glob("*.state"))],
    how="diagonal_relaxed",          # specs with different fields stack with nulls
)

fed = bank.summary("ridge")    # one row per group:
                               #   rows_fed          routed to the group
                               #   rows_processed    the model accepted
                               #   rows_skipped      it did not (a feature or the weight was missing)
                               #   rows_learned      moved the fit (a weight above zero and a target present)
                               #   rows_zero_weight  advanced the clock and nothing else
                               #   weight_sum, clock_min, clock_max, last_clock
                               #   session_changes, clock_backwards, resets   what the clock rules met
cols = bank.describe("ridge")  # one row per input column per group: column, role, count, null_count, mean, std, min, max
                               # -- counting as the models count: a null, a NaN, an infinity or a magnitude
                               # beyond 1e100 is a null_count, not a value
```

Neither `summary()` nor `describe()` forgets: they are plain counts over the
whole stream, taken in row order, so they are the same whatever the
chunking, and `predict` does not move them. A group that has not learned
from a row yet gives a last row of nulls.

### Reading a state without this library

```python
text = bank.to_json()          # everything save() writes, as JSON: look at a state, compare two, hand one to a non-Python program
bank.save_json("bank.json")    # the same, to a file
```

The JSON is an export, not a second format: `load` reads the binary form
only. It is faithful, including the values JSON cannot write: `NaN` and
`±inf` are written as the strings `"nan"`, `"inf"` and `"-inf"`, the same
spelling a spec's `halflife` uses. That matters because `halflife=inf`
means no forgetting, so an ordinary state carries an infinity, and a plain
JSON encoder writes one as `null` without saying so. Every export is read
back and checked against the state before you get it, so a state that
could not be carried is an error, not a file that is quietly wrong.

## Reading the fit

What a bank can tell you about its fit with no data at hand, and how to
reach any field of its output without building the field's name.

### Coefficients

Two ways, and they agree row for row:

```python
ols = po.spec.ewridge("ols", targets=["y"], features=["x0", "x1"], clock="t",
                      halflife=600.0, max_dclock=300.0, group="stock_id",
                      coef_every=1)         # write coef on every row (default 0: each group's last row in a chunk)

# 1. From a bank -- live, or loaded from a state file with no data at hand.
bank = po.ModelBank([ols])
bank.fit_predict(df)
betas = bank.coef()                  # one row per coefficient: spec, group, instance, n_eff, ..., term, coef
                                     # -- the fit as of the last row each group learned from
wide = betas.pivot("term", index=["group", "instance"], values="coef")

# 2. From the output, as columns: the fit as it moved, one row per row.
path = (
    lf.online.fit_predict([ols])
    .online.unnest([ols])            # pred_y, resid_y, n_eff, coef_y_intercept, coef_y_x0, coef_y_x1, ...
    .select("t", "stock_id", "^coef_.*$")
    .collect()
)
```

The output's `coef` is written *after* each row's update, while the row's
own `pred` comes from the fit *before* it. With `coef_every=1` it is a list
of `k` floats on every row. Under a grid, of several `ridge` values,
`feature_sets`, a `lasso_path` or several targets, the list holds one block
per target and grid point. `unnest` names each block's columns the way the
`pred` fields are named, so `coef_y_x0__r0.5@h500` sits beside
`pred_y__r0.5@h500`, and `bank.coef()` carries the same columns to tell the
blocks apart; add them to the pivot's `index`. `unnest` reads a saved
output the same way, as in
`pl.scan_parquet("fitted.parquet").online.unnest([ols])`, and takes the
specs, a bank, or the path of a saved state.

### Output field names

You address the output by field name, so the names are a contract. The
grammar:

```
pred_{target}{combo}{instance}     combo    = ""            single ridge, no feature sets
resid_{target}{combo}{instance}             | __r{ridge}     ridge grid
sigma_{target}{combo}{instance}             | __{set}        feature sets, single ridge
absresid_q{level}_{target}...               | __{set}_r{ridge}
n_eff{instance}                    instance = ""            single halflife
coef{instance}                              | @h{halflife}   halflife grid
```

Numbers render as plain decimals in `[1e-6, 1e7)` and in compact
scientific notation outside it. Every name, default and signature is
pinned against a checked-in snapshot, so a change is a reviewable diff and a
version bump, never a silent rename of your columns. You never have to
build these strings:

```python
grid = po.spec.ewridge("m", targets=["y"], features=["x0", "x1"], clock="t",
                       max_dclock=300.0, halflife=[100.0, 500.0], ridge=[1e-6, 0.5])

idx = po.spec.output_index(grid)     # every field, with the values its name encodes, and its dtype
name = idx.filter((pl.col("kind") == "pred") & (pl.col("target") == "y")
                  & (pl.col("ridge") == 0.5) & (pl.col("halflife") == 500.0))["field"].item()
out["m"].struct.field(name)                                    # "pred_y__r0.5@h500"

row = po.spec.coef_fields(grid).filter(                        # one row per coefficient: its list, position, and unnest column
    (pl.col("term") == "x1") & (pl.col("ridge") == 0.5) & (pl.col("halflife") == 500.0)
).row(0, named=True)
out["m"].struct.field(row["field"]).list.get(row["position"])  # field "coef@h500", position 5
```

Both tables come from the same Rust code that renders the names, so they
cannot drift from the strings. The index also carries each field's `dtype`,
the column type the bank declares to Polars before it reads the first row.
One sharp edge: if you parse field names downstream, avoid `__` and `@` in
target names and feature-set labels, because a target named `y__r0.5`
renders like a ridge grid on `y`. [docs/OUTPUTS.md](docs/OUTPUTS.md) lists
every field of every model.

### The running sums behind a fit

`bank.gram(spec)` returns the matrices the model itself solves against, its
*Gram* in least-squares terms, per group and per halflife. They are a
complete summary of the rows the model has seen, a sufficient statistic in
the statistical sense, so a saved state answers questions the run never
asked:

```python
ols = po.spec.ewridge("ols", targets=["y"], features=["x0", "x1", "x2"],
                      halflife=500.0, ridge=1e-9, standardize=False)
fitted = po.ModelBank([ols])
fitted.fit_predict(df)

g = fitted.gram("ols")[0]      # one dict per (group, halflife, Gram); needs numpy, an optional extra
g["targets"]                              # the targets fitted from this Gram
g["means"], g["comoments"]                # the feature means, and the centred k x k co-moment matrix
g["cross_moments"], g["target_weights"]   # per target: the uncentred E[z*y], and the weight behind it
g["means_by_target"]                      # per target: the column means over the rows it was present on
g["cross_centred"]                        # per target: E[(z - m)(y - ybar)] at those means, what the fit is solved from
g["target_means"], g["target_vars"]       # per target: the target's own mean and centred variance
g["n_eff"], g["n_kish"], g["target_n_kish"]   # the accumulated weight, and Kish's effective sample size (features, and per target)

# The algebra the model runs, done by hand: the centred system, in which a level costs nothing.
slopes = np.linalg.solve(g["comoments"][1:, 1:], g["cross_centred"][0][1:])   # column 0 is the intercept
intercept = g["cross_moments"][0][0] - g["means_by_target"][0][1:] @ slopes
resid_var = g["target_vars"][0] - slopes @ g["comoments"][1:, 1:] @ slopes
r2 = 1 - resid_var / g["target_vars"][0]

# po.gram is the toolkit for these, so the same algebra is one call:
r2 = po.gram.coef_stats(g, po.gram.solve(g, ridge=1e-9))["r2"]          # residual variance, R², standard errors and t
ridges = po.gram.solve(g, ridge=[0.0, 0.01, 0.1, 1.0], standardize=True)  # the model's own ridge, in original units; a list of ridges is one eigendecomposition
worst = po.gram.condition(g)["kappa"]                                     # Belsley's condition indexes and variance-decomposition proportions
po.gram.correlation(g)                                                    # the correlation matrix
po.gram.vif(g)                                                            # variance inflation factors
po.gram.subset(g, ["x0", "x1"])                                           # the Gram of some of the columns: a sub-block, not a recomputation
po.gram.merge([g, g])                                                     # pools the Grams of disjoint row sets into the Gram of their union, exactly
po.gram.lasso_path(g, [0.1, 0.01])                                        # the lasso model's coordinate descent, offline
```

Three things to know before reading the numbers:

| | why |
|---|---|
| `n_eff` counts weight, not rows | `n_kish = n_eff² / Σw²` is the number of equally weighted rows the moments are worth, which is what a standard error divides by. It does not fall when a stream goes quiet; `n_eff` does |
| a spec can have several Grams | under the default `target_gaps="own_rows"`, a target that goes missing on different rows from the others is fitted from a Gram of its own. `gram()` returns one dict per Gram, each naming its `targets` |
| `coef()` and `gram()` can disagree | `bank.coef()` is as of the model's last *solve*, which its `solve_every` schedule decides, while `gram()` is as of the last row |

The [`ModelBank.gram`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.ModelBank.gram)
reference states every array and the identities that relate them.

### One row per finished group

A bank keeps one state per group key for the life of the bank. On a stream
whose key space keeps growing, such as a day id, a session id or a block
number, that is unbounded memory for state nobody will read again.
`group_close` says when a group is finished, and the bank then writes the
group's running sums out as one row and frees its memory.

```python
blocks = po.spec.ew_cov("cov", features=["x0", "x1"], lam=1.0,
                        group="block", group_close="monotone")   # "monotone": the key never goes backwards, so
by_block = df.with_columns(block=pl.int_range(pl.len()) // 100)   # a key below the largest seen is finished

bank = po.ModelBank([blocks])
bank.fit_predict(by_block)

closed = bank.closed_groups()          # one row per finished block: the gram() a driver would have read at that
                                       # moment, bit for bit, plus the span's own rows_fed, rows_learned,
                                       # clock_min, clock_max; coef for a model that has one; the
                                       # eigendecomposition for an ew_cov with pca; a marginal's pairs
first = po.gram.from_row(closed.head(1))
corr = po.gram.correlation(first)      # everything in po.gram works on it
```

`"monotone"` refuses a chunk whose keys are out of order, naming the row.
An integer key column is ordered as numbers, and anything else as text, so
`"9"` comes after `"10"`: sort by the same rule the bank reads, or the
chunk is refused. A `Categorical` column sorts by the order its categories
were first seen, so cast it to `pl.String` first. `"session"` closes a
group where its `session` value changes. Either way the last group never
closes, since nothing proves it is finished, and it stays readable through
`gram()`.

A run also writes the same rows to a sidecar file. With no other output at
all, that is the whole of an accumulate-only pass: read a stream that does
not fit in memory, and write one row per block.

```python
for _ in by_block.lazy().online.fit_predict([blocks], closed_groups="blocks.parquet").collect_batches():
    pass   # the per-row output is not wanted; only the sidecar file is
```

`by_block` and `blocks` are the ones built in the block above. The command
line writes the sidecar too ([docs/RUNNER.md](docs/RUNNER.md)). What has
closed and not yet been read is saved with the state, so a driver that
saves between chunks loses no rows.

### Reading a correlation matrix

`po.gram` solves and diagnoses a design matrix. `po.corr` is its
complement: the arithmetic that comes *after* a correlation matrix, in the
same style, numpy only, pure functions, each held against the paper it
comes from.

```python
r = po.corr.matrix(closed.head(1))         # an array, a gram() dict, or a closed row -> a correlation matrix
fixed, dist, iters = po.corr.nearest(r)    # Higham (2002): the nearest correlation matrix, by alternating projections
shrunk, alpha = po.corr.shrink(r, alpha=0.2)   # Ledoit-Wolf shrinkage toward a constant-correlation (or identity) target
z = po.corr.to_z(r); back = po.corr.from_z(z)  # Fisher's transform, clipped so a degenerate +-1 is finite
rho = po.corr.equicorr(r)                  # deco's one number for the whole matrix, offline
vals, vecs = po.corr.spectral(r, 1)        # the top eigenpairs ...
po.corr.from_spectral(vals, vecs)          # ... and the completion back to a correlation matrix
po.corr.absorption(r, 1)                   # the absorption ratio: how much of the variance the top k eigenpairs carry
lo, hi = po.corr.mp_edge(n=2000, m=50)     # the Marchenko-Pastur edges: eigenvalues inside them are what pure noise gives
po.corr.fisher_se(n=2000, rho=0.3)         # the standard error of a correlation, with the AR(1) inflation and its caveats
```

The rest of the module:

| function | what it gives |
|---|---|
| `block_means`, `from_blocks` | the mean correlation within and between labelled blocks, and the block matrix back from those means |
| `mp_density` | the Marchenko–Pastur density |
| `signal_share` | how much of a correlation's movement between blocks is not sampling noise |
| `loss` | how wrong a forecast correlation matrix was: `qlike`, `z_mse` or Engle–Colacito `minvar` |
| `epps_invert` | the correlation at a coarser scale, from `ew_cov`'s lagged co-moments |
| `shift` | the standardised absorption shift, `(fast − slow) / scale` |
| `equicorr_loglik` | the Gaussian log-density of a standardised row under an equicorrelation matrix |

The [API reference](https://hgilde.github.io/polars-online/corr.html) has
each one's arguments.

## Diagnostics, selection and evaluation

Outputs you switch on in a spec, and tools that read the output
afterwards. Every per-row output is computed from what the models have
already learned and read *before* the row, so none of them sees the row it
describes: they are as out-of-sample as the predictions. They all live in
memory that does not grow with the stream.

### Per-row diagnostics

```python
diag = po.spec.ewridge(
    "diag", targets=["y"], features=["x0", "x1"], clock="t", max_dclock=300.0, halflife=500.0,
    ridge=[1e-6, 0.1],           # a grid, so there is something to select among
    emit_sigma=True,             # sigma_<slot>:     EW standard deviation of that slot's out-of-sample residuals
    emit_resid_z=True,           # resid_z_<slot>:   resid / sigma -- how surprising the row was, in units of recent error
    emit_selected=True,          # selected_<t>, pred_<t>__selected: the ridge value, feature set or halflife with the
                                 #                   lowest EW out-of-sample error so far
    emit_averaged=True,          # pred_<t>__averaged: softmax(-eta * EW error / the best's) blend over the same
                                 #                   choices -- hedges where emit_selected commits
    emit_drift=True,             # drift_<slot>:     Page-Hinkley break detection on the residuals;
                                 #                   drift_action="reset" also starts the model over
    emit_metrics=True,           # ic_, r2_, hit_rate_<slot>: what po.eval computes, kept beside the model
    resid_quantiles=[0.5, 0.9],  # absresid_q<p>_<slot>: running quantiles of |resid| (the P² algorithm) --
                                 #                   an interval that assumes no distribution
    emit_autocorr=True,          # autocorr_<slot>:  EW correlation of each residual with the previous one;
                                 #                   non-zero means the model is missing something
    conformal=0.9,               # lo_, hi_, coverage_<slot>: an interval at that coverage, assuming no
                                 #                   distribution, and the coverage it has actually delivered
)
band = po.ModelBank([diag]).fit_predict(df).unnest("diag")
```

`emit_metrics` reads differently on an `sgd` or `ftrl` fit with
`loss="logistic"`, because `pred` is then a probability and `y` a 0/1
label rather than a signed target. Under their usual names, `hit_rate` is
the accuracy at a 0.5 threshold, `r2` the Brier skill score against the
running base rate, and `ic` the point-biserial correlation between the
probability and the label. There is no streaming log loss;
`po.eval.metrics(..., binary=True)` adds it over a collected frame.

### Conformal intervals

`conformal` is the interval to use when the residuals are not Gaussian. It
tracks the `coverage` quantile of `|resid|` directly, widening its radius
on a miss and narrowing it on a hit, so its long-run coverage is the number
you asked for, whatever the residuals do. `sigma` gives a Gaussian interval
instead, and on fat-tailed or heteroskedastic residuals that one
over-covers by several points where this one lands on target.

```python
ci = po.spec.ewridge("ci", targets=["y"], features=["x0", "x1"], clock="t",
                     max_dclock=300.0, halflife=500.0, conformal=0.9)
band = po.ModelBank([ci]).fit_predict(df).unnest("ci")
held = band.select(((pl.col("lo_y") <= df["y"]) & (df["y"] <= pl.col("hi_y"))).mean())   # the realized coverage
```

### Evaluating an output frame

After the fact, `po.eval` reads the output frame:

```python
po.eval.metrics(out, "ridge", by=["stock_id"])                       # R², IC, hit rate, MSE
po.eval.rolling_metrics(out, "ridge", clock="t", window=3600.0)     # the same, per clock window
po.eval.compare_specs(out, ["ridge", "kalman"])                     # one table, many specs: which had the lower error
po.eval.seqtest(out, a="kalman", b="ridge", by=["stock_id"])        # is kalman closer? evidence per row -- the same test
                                                                    # the seqtest model runs inside a bank
```

### Evaluating a stream too large to hold

The four calls above need the whole frame. When the output is never held
in one place, say fifty slots over a billion rows, reduce each chunk
instead and keep ten numbers per key:

```python
ridge = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"],
                        clock="t", max_dclock=300.0, halflife=500.0, group="stock_id")
scoring = po.ModelBank([ridge])

running = None
for chunk in df.iter_slices(100):
    part = po.eval.sums(scoring.fit_predict(chunk), "ridge", by=["stock_id"])   # ten numbers per key
    running = part if running is None else po.eval.merge_sums(running, part)  # exact, whatever the split

po.eval.from_sums(running, min_obs=10)   # R², IC, hit rate, MSE and RMSE -- the same numbers metrics() gives
```

The sums are **centred**, weighted means and centred second moments merged
with a parallel-axis term, rather than raw `Σy` and `Σy²`. A target sitting
around 1e8 with unit spread destroys the raw form's variance entirely, and
this form does not notice. `weight=` names a column to weight the rows by.

### Data whose truth is known

The regime detectors, `deco`, `hmm`, `corrchange` and `bocpd`, make claims
about streams whose correlation structure changes, and a claim like that is
measured against data whose truth is known.
[`po.sim.regimes`](https://hgilde.github.io/polars-online/sim.html#polars_online.sim.regimes)
generates such a stream from one seed and hands back the truth beside it:

```python
out = po.sim.regimes(4, states=[0.2, 0.7],                    # four series; two regimes, at these equicorrelations
                     transition=[[0.98, 0.02], [0.02, 0.98]],  # how the regimes switch
                     n_blocks=8, rows_per_block=500,
                     phi=0.3, noise=0.01,                      # returns correlated with their own past; observation noise
                     async_rates=[1.0, 1.0, 0.4, 0.4],         # two series tick less often: a bar with no tick is null
                     seed=0)                                   # two calls with the same seed are byte-identical
rows, truth = out["rows"], out["truth_blocks"]
# rows:          what a consumer sees -- levels x_1 .. x_m (so refresh_time then .diff() apply),
#                a clock, a session and an optional volume
# truth_rows:    per bar, the block, state, volatility multiplier and interpolation fraction
# truth_blocks:  each block's true correlation matrix
```

`durations` makes each state last exactly as long as it says, and
`design="smooth"` interpolates the matrix across a boundary instead of
stepping. What `hmm`, `corrchange` and `bocpd` find on such streams, and
what they miss, is measured in [docs/REGIMES.md](docs/REGIMES.md).

## Models

Twenty model families share one set of stream semantics: a spec's clock,
decay, grouping and warm-up mean the same thing whichever model it names
([How a bank sees a stream](#how-a-bank-sees-a-stream)). Each model's
section below gives what it is for, its update rule, the parameters that
are its own, and what it writes. Each builder's docstring in the API
reference lists every keyword with its default, and
[docs/OUTPUTS.md](docs/OUTPUTS.md) lists every field of every model.

*Learns by* says how a model takes in a row, which decides what it does
when forgetting is off:

| learns by | the model | with decay off |
|---|---|---|
| **solve** | keeps running sums and computes its coefficients from them | converges to the batch answer, in any row order |
| **accumulate** | keeps running sums and reports them | converges to the batch answer, in any row order |
| **step** | moves its coefficients a little on each row | depends on the row order |
| **filter** | carries a belief forward from row to row | depends on the row order |
| **test** | counts evidence as it goes | depends on the row order |

Each row links to the model's builder in the API reference, which has
every parameter, and to its section below, which states its update rule:

| model | learns by | what it is |
|---|---|---|
| **[Linear models](#linear-models)** | | |
| [`ewridge`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.ewridge) · [math](#ewridge--ew-ridge-on-sufficient-statistics) | solve | exponentially weighted ridge regression, the workhorse; several ridge values, feature sets and halflives are fitted from the same running sums at almost no extra work |
| [`rls`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.rls) · [math](#rls--recursive-least-squares) | solve | recursive least squares, in the numerically safe square-root form |
| [`lasso`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.lasso) · [math](#lasso--lasso-path-with-free-λ-selection) | solve | lasso and elastic-net path, with the penalty chosen as the stream runs |
| [`kalman`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.kalman) · [math](#kalman--random-walk-β-dynamic-linear-model) | filter | a Kalman filter whose coefficients drift as random walks |
| [`huber`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.huber) · [`quantile`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.quantile) · [math](#huber--quantile--robust-regression) | solve | robust and quantile regression |
| [`sgd`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.sgd) · [math](#sgd--stochastic-gradient-descent) | step | stochastic gradient descent with squared, Huber, quantile, ε-insensitive, Poisson and logistic losses |
| [`pa`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.pa) · [math](#pa--passive-aggressive-regression) | step | passive-aggressive regression, with no learning rate to tune |
| [`ftrl`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.ftrl) · [math](#ftrl--online-logistic-regression) | step | FTRL-proximal logistic regression, with an L1 penalty that zeroes coefficients |
| [`holt`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.holt) · [math](#holt--holts-linear-trend) | step | Holt's linear trend, the baseline that uses no features |
| **[Moments and correlation](#moments-and-correlation)** | | |
| [`ew_cov`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.ew_cov) · [math](#ew_cov--exponentially-weighted-moments) | accumulate | running mean, variance, covariance, correlation, partial correlation, Mahalanobis distance and principal components |
| [`marginal`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.marginal) · [math](#marginal--every-pairs-moments-kept-in-the-state) | accumulate | every (feature, target) pair's running mean, variance, covariance, correlation, slope and t, for a wide set of columns, kept in the state and read back as a table; optionally at a set of lags, and with binned target moments for the relations a correlation cannot see |
| [`deco`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.deco) · [math](#deco--one-correlation-for-the-whole-matrix) | accumulate | one correlation for the whole matrix: Engle & Kelly's equicorrelation, or one per block and per pair of blocks |
| [`rcov`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.rcov) · [math](#rcov--a-blocks-realised-covariance-robust-to-noise) | accumulate | a block's realised covariance, robust to microstructure noise: the Barndorff-Nielsen–Hansen–Lunde–Shephard kernel or Christensen–Kinnebrock–Podolskij pre-averaging, reported when a group closes |
| **[Clustering and classification](#clustering-and-classification)** | | |
| [`kmeans`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.kmeans) · [math](#kmeans--exponentially-weighted-k-means) | step | exponentially weighted k-means: cluster labels assigned before the row is learned from, with a split–merge move that finds a cluster born after seeding |
| [`micro`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.micro) · [math](#micro--density-based-clustering-any-shape) | step | density-based clustering: DenStream micro-clusters linked into clusters of any shape and number, flagging the rows that belong to none |
| [`ew_class`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.ew_class) · [math](#ew_class--gaussian-classification-on-ew_cov-moments) | accumulate | Gaussian classification, QDA, LDA or naive Bayes, with one set of running moments per class: a label column in, class probabilities out |
| **[Sequential tests and regimes](#sequential-tests-and-regimes)** | | |
| [`seqtest`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.seqtest) · [math](#seqtest--a-sequential-test-of-a-sign-by-betting) | test | a sequential test of a sign by betting, giving evidence you can read at any row: on its own, of a column's sign; with `a` and `b`, of whether one spec of the bank predicts closer than another |
| [`corrchange`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.corrchange) · [math](#corrchange--has-the-correlation-structure-changed) | test | has the correlation structure changed: the Wied–Krämer–Dehling constancy test span by span, or the size of a change between two windows against a permutation null |
| [`hmm`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.hmm) · [math](#hmm--which-regime-are-we-in) | filter | a Gaussian hidden Markov model, filtered as the stream runs: `ew_class` without the labels, with a transition matrix that can be learned |
| [`bocpd`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.bocpd) · [math](#bocpd--how-long-has-this-regime-lasted) | filter | how long this regime has lasted: Adams & MacKay's run-length posterior, so the answer is the regime's age and not a flag |

The running sums a model keeps are its *accumulators*. All of them are
exponentially weighted **means**, not sums, so they stay bounded over a
stream of any length. Second moments are kept **centred**, by a weighted
Welford update, so a variance is right even when a feature sits far from
zero. In the update rules, `z` is `[1, x]` when there is an intercept, `w`
is the row's weight, `λ` the row's decay, and `W` the total weight so far.

### Linear models

Regressions of numeric targets on features, and one baseline that uses no
features.

#### `ewridge` — EW ridge on sufficient statistics

*API:* [`po.spec.ewridge`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.ewridge) — *Rust:* [`ewridge.rs`](crates/online-core/src/ewridge.rs) — *Outputs:* [fields](docs/OUTPUTS.md#ewridge)

Ridge regression on running sums. Each target's sums are updated on the
rows where it is present, and the coefficients are solved from them on a
schedule.

```
W'   = λW + w                       n_eff, over every row
W_j' = λW_j + w                     on the rows where y_j is present, λW_j on the others
S_j' = (λW_j·S_j + w·z zᵀ) / W_j'   r_j' = (λW_j·r_j + w·z·y_j) / W_j'
solve:  (S_j + ridge·D) β_j = r_j   D = I minus the intercept slot
```

```python
rr = po.spec.ewridge(
    "rr", targets=["y"], features=["x0", "x1", "x2"], clock="t", max_dclock=300.0, halflife=600.0,
    ridge=[1e-6, 0.1],             # one value, or a list: every value is solved from the same sums, so a grid is nearly free
    feature_sets={"mkt": ["x0"], "all": ["x0", "x1", "x2"]},   # named subsets, likewise solved from one set of sums
    solve_every=12.0,              # solve every 12 clock units (default halflife/50; every row when halflife=inf or lam= is given)
    max_rows_between_solves=100,   # ... and at least every 100 rows, whatever the clock does
    standardize=True,              # solve on the correlation matrix and undo afterwards; a feature with almost no
                                   # variance is dropped rather than allowed to blow the solve up
    ridge_decay=False,             # True: the ridge is a fading warm start ("start at yesterday's fit"), not a
                                   # permanent per-observation penalty -- because S is a mean, a plain ridge is permanent
    coef_prior=None,               # shrink toward a stated belief instead of toward zero
    target_gaps="own_rows",        # a target null on some rows is fitted on its own rows; "pairwise": one S over every row
)
```

Updating `S` takes O(k²) time per row for `k` features, and the solve is a
Cholesky factorization. With decay off and `ridge=0` this is ordinary least
squares over every row seen, in any row order. The coefficients match
`numpy.linalg.lstsq` to 2e-13 forwards, backwards or shuffled, and 6M rows
× 20 features from a parquet stream peak at 1.4 GB, against 3.97 GB for
`lstsq` on the same rows. One trap in that setting: the solve schedule
defaults to `halflife/50`, so `halflife=1e12` solves once, at
`min_periods`, and never again. Say `inf`, or set `solve_every`.
`solve_every=1000` on that stream takes 1.4 s instead of 11 s, with the
coefficients at most 1000 rows out of date.

`target_gaps` says which rows a target's `S_j` covers when the target is
null on some of them. Under `"own_rows"`, the default, it covers exactly the
rows the target is present on, so the target's fit is the fit of the frame
with those nulls dropped. Targets present on the same rows share one `S`. A
target that goes missing on a row where the others are present takes a
copy and keeps its own from then on, so a bank of targets holds one `k × k`
matrix per pattern of missing rows. `"pairwise"` keeps one `S` over every
row, the way pandas' `DataFrame.cov` takes a pairwise-complete covariance,
and centres each target's `r_j` at its own rows' means. The two agree when
the gaps have nothing to do with the features. Where they do, as with a
target present only on trade rows between market-data rows, `"pairwise"`
scales each slope by the feature's variance on the target's rows over its
variance on every row. Both are held to independent libraries in
`tests/test_second_opinion.py`.

At a thousand features, the O(k²) update of `S` takes most of the time:

```python
wide = po.spec.ewridge(
    "wide", targets=["y"], features=["x0", "x1", "x2"], halflife=float("inf"), solve_every=1e9,
    max_rows_between_solves=1000,
    gram_block_rows=256,           # hold 256 rows back and add them to S with one matrix product instead of
)                                  # 256 single-row updates; refused with window=, and where a solve happens every row
```

| features | rows per second, against one row at a time, on one thread |
|---:|---:|
| 256 | 5.1× |
| 1,000 | 6.6× |
| 2,000 | 5.9× |

A solve every 512 rows brings each of those down to about 4×, because the
solve takes the same time either way. `n_eff`, the timing of every
prediction and chunk invariance do not change. The coefficients agree with
the row-by-row fit to rounding, since the blocked sum is the same sum in a
different order. The held rows travel in the state file, so a save
mid-block resumes on the same block
([docs/PERFORMANCE.md](docs/PERFORMANCE.md) §18).

#### `rls` — recursive least squares

*API:* [`po.spec.rls`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.rls) — *Rust:* [`rls.rs`](crates/online-core/src/rls.rs) — *Outputs:* [fields](docs/OUTPUTS.md#rls)

The coefficients move on every row. There is no solve schedule, so they
are never out of date.

```
A ← λA + w zzᵀ       b_j ← λb_j + w y_j z        β_j = A⁻¹ b_j
A₀ = ridge·I         b₀ = ridge·coef_prior
```

```python
rls = po.spec.rls(
    "rls", targets=["y"], features=["x0", "x1"], clock="t", max_dclock=300.0, halflife=600.0,
    ridge=1e-3,                    # A starts at ridge * I -- unlike ewridge, this penalizes the intercept too
)                                  # a row with any null target is scored but not learned from, for every target,
                                   # because the factor is shared between them
```

The model stores the Cholesky factor of `A`, updated row by row with Givens
rotations: the *square-root form* of the recursion. It takes O(k²) time per
row, the same as the textbook recursion on the inverse `P`, and avoids both
of that form's failures. `P` loses symmetry to rounding by a factor of
`1/λ` per row, and one extreme row can cancel it and freeze a coefficient
for good. The result equals `ewridge(ridge_decay=True)` solved on every
row, to better than 1e-9.

#### `lasso` — lasso path with free λ selection

*API:* [`po.spec.lasso`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.lasso) — *Rust:* [`lasso.rs`](crates/online-core/src/lasso.rs) — *Outputs:* [fields](docs/OUTPUTS.md#lasso)

Coordinate descent on the standardized running sums, started from the
previous solution both along the path of penalties and from one solve to
the next:

```
ρ_i = c_i − Σ_{j≠i} C_ij β_j
β_i = soft(ρ_i, λ·l1_ratio) / (C_ii + λ(1 − l1_ratio))
```

```python
las = po.spec.lasso(
    "las", targets=["y"], features=["x0", "x1", "x2"], clock="t", max_dclock=300.0, halflife=600.0,
    lasso_path=[0.1, 0.01, 0.001],   # the penalties; predictions for every one are computed anyway, so
                                     # lam_selected_<target> -- the one with the lowest EW out-of-sample squared
                                     # error so far, as it stood before the row -- adds no work of its own
    l1_ratio=1.0,                    # below 1: an elastic net
    target_gaps="own_rows",          # which rows a target null on some is fitted from, as for ewridge
)
```

It reads the running sums `ewridge` keeps, centred the same way, so a
feature or a target far from zero loses the path no precision, and
`target_gaps` means what it means there. The path is checked against the
KKT conditions of its objective, and a penalized path point against
statsmodels' elastic net on the target's own rows.

#### `kalman` — random-walk-β dynamic linear model

*API:* [`po.spec.kalman`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.kalman) — *Rust:* [`kalman.rs`](crates/online-core/src/kalman.rs) — *Outputs:* [fields](docs/OUTPUTS.md#kalman)

A regression whose coefficients are allowed to drift, tracked by a Kalman
filter.

```
β_j ← Φβ_j    P_j ← ΦP_jΦ + Q·Δclock    Φ = diag(2^(−Δclock/r_i))
s   = zᵀP_j z + R_j/w                   k   = P_j z / s
β_j ← β_j + k(y_j − zᵀβ_j)              P_j ← P_j − k zᵀP_j
```

```python
revert = po.spec.kalman(
    "k", targets=["y"], features=["signal_a", "signal_b"], clock="t", max_dclock=10.0,
    halflife=200.0,                   # the observation-noise estimate (the EW residual variance) forgets at this rate
    coef_halflife=100.0,              # how fast a coefficient may drift, on standardized features: q_i = σ²(ln2 / h_i)²,
                                      # matching EW-RLS's steady state; one number, or one per slot; inf pins a coefficient
    revert_halflife=[float("inf"), 50.0, 50.0],   # by default a coefficient is a random walk and keeps its last value;
                                      # with this, a slope halves toward zero every 50 clock units while nothing is observed
                                      # -- a mean-reverting (AR(1)) prior; inf in the first slot leaves the intercept alone
    standardize=True,                 # the default; with standardize=False, q=0 and a fixed obs_var= this is exactly
                                      # Bayesian linear regression (river's BayesianLinearRegression to 3.6e-15)
)
out = po.ModelBank([revert]).fit_predict(df)
```

Reverting helps a regressor that is only occasionally active: it is
forgotten between its bursts instead of held at its last value, and a stale
effect cannot persist through a run of null targets. The reversion acts in
the standardized coordinates, so zero means "no effect" for a slope and
"the target averages zero" for the intercept. A reverting slot's long-run
prior variance is `q_i·Δclock/(1−φ_i²)`, where a random walk's grows without
bound. `predict` moves the coefficients by the same `Φ` over the distance
from the last learned row, capped by `max_dclock`, so a prediction far past
the data is the intercept alone.

#### `huber` / `quantile` — robust regression

*API:* [`po.spec.huber`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.huber) and [`po.spec.quantile`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.quantile) — *Rust:* [`robust.rs`](crates/online-core/src/robust.rs) — *Outputs:* [huber](docs/OUTPUTS.md#huber), [quantile](docs/OUTPUTS.md#quantile)

Two regressions that do not let one wild target move the fit. Both read
each row's residual `r` against the fit *before* the row is learned, so
both stay out-of-sample. Both keep their running sums per target, because
their weights are per target.

```
huber:     the ridge update at weight  w · min(1, δσ / |r|)
quantile:  one Newton step on the check loss, smoothed by a uniform kernel of half-width h = quantile_eps · σ
           |r| ≤ h:  a least-squares row with target  y + 2h(τ − ½)
           |r| > h:  adds  2h · ψ_τ(r) · z  to the cross-moment and nothing to the Gram,  ψ_τ(r) = τ − 1{r < 0}
           h is never narrower than (k/n)^{2/5} · σ, for the target's effective sample n
```

```python
hub = po.spec.huber("hub", targets=["y"], features=["x0", "x1"], clock="t", max_dclock=300.0, halflife=600.0,
                    huber_delta=1.5)      # a residual beyond huber_delta * sigma is down-weighted
med = po.spec.quantile("med", targets=["y"], features=["x0", "x1"], clock="t", max_dclock=300.0, halflife=600.0,
                       quantile=0.5,      # the level: 0.5 is a median regression
                       quantile_eps=0.2)  # the band the Newton step leans on, in units of sigma (default 0.2)
```

The band's floor keeps a short halflife from leaving the Newton step
nothing to lean on. A band holding fewer than one row per coefficient takes
least-squares rows until it holds rows again, which rebuilds a fit that a
row at the input bound has moved. Both models are held to numpy references
to about 1e-13.

#### `sgd` — stochastic gradient descent

*API:* [`po.spec.sgd`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.sgd) — *Rust:* [`sgd.rs`](crates/online-core/src/sgd.rs) — *Outputs:* [fields](docs/OUTPUTS.md#sgd)

One gradient step per row and no solves, in O(k) time per row: the cheap
baseline, and the only model here that takes count targets
(`loss="poisson"`).

```
eta = zᵀβ        p = link(eta)        gᵢ = (dL/d eta)·zᵢ·w + l2·βᵢ        βᵢ -= lrᵢ·gᵢ
```

| loss | link | `dL/d eta` |
|---|---|---|
| `squared` | identity | `p − y` |
| `huber` | identity | `clamp(p − y, ±delta)` |
| `quantile` | identity | `1{y < p} − τ` |
| `epsilon_insensitive` | identity | 0 inside the tube, else `sign(p − y)` |
| `poisson` | log | `p − y` |
| `logistic` | sigmoid | `p − y` |

```python
weights = po.spec.sgd(
    "w", targets=["y"], features=["signal_a", "signal_b", "x0"], halflife=200.0,
    loss="squared",              # or huber, quantile, epsilon_insensitive, poisson (count targets), logistic (0/1 targets)
    learning_rate=0.01,
    schedule="constant",         # or inv_scaling (lr / (1 + n_eff)^power), or adagrad, whose running sum of squared
                                 # gradients decays on the clock so an adapted rate opens up again after a long gap
    clip_gradient=1e3,           # the default; with a log link one large count would make the next gradient
                                 # exponentially bigger. Never binds for the identity-link losses
    coef_min=0.0,                # bound each slope from below (one number, or one per feature; inf for none) ...
    coef_sum=1.0,                # ... and fix their total: after every step the slopes are moved to the nearest point
                                 # that satisfies all bounds (the Euclidean projection). The intercept is never constrained.
                                 # coef_min=0, coef_sum=1 is a long-only, fully invested portfolio;
                                 # coef_min=0 alone is a sign the model must respect;
                                 # coef_min equal to coef_max pins a slope at a known value
    coef_every=1,
)
fit = po.ModelBank([weights]).fit_predict(df)
last = fit["w"].struct.field("coef").drop_nulls()[-1]
assert min(last[1:]) >= 0.0 and abs(sum(last[1:]) - 1.0) < 1e-12   # the fit starts from the projected zero: uniform weights
```

`coef` reports what the projection returned, in the caller's units even
under `scale_features=True`.

#### `pa` — passive-aggressive regression

*API:* [`po.spec.pa`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.pa) — *Rust:* [`pa.rs`](crates/online-core/src/pa.rs) — *Outputs:* [fields](docs/OUTPUTS.md#pa)

Each row asks the fit to come within `eps` of its target, and the update is
the smallest change that does so, so there is no learning rate to tune.

```
loss = max(0, |y − p| − eps)      s = ‖z‖²
pa    τ = loss / s          pa1  τ = min(c, loss/s)      pa2  τ = loss / (s + 1/(2c))
β    += τ · sign(y − p) · z
```

```python
pa = po.spec.pa(
    "pa", targets=["y"], features=["x0", "x1"], halflife=200.0,
    mode="pa1",                  # the default: the step is capped at c. Plain "pa" moves the fit as far as one bad
    c=0.1,                       # row demands; "pa2" damps the step by c instead of capping it
    eps=0.05,                    # the row is "close enough" inside this margin, and nothing moves
)                                # a row weight below 1 scales the step; above 1 it counts as 1
```

PA keeps no running sums, so its coefficients have no halflife, and the
clock drives only `n_eff`. It takes the same `coef_min`, `coef_max` and
`coef_sum` as `sgd`, with the projection applied after each update. The
step then no longer meets the row's margin exactly, and a truth outside the
allowed set is never reached, so keep `c` small: each row moves the fit
only as far as `c` allows, and the projection takes the rest back.

#### `ftrl` — online logistic regression

*API:* [`po.spec.ftrl`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.ftrl) — *Rust:* [`ftrl.rs`](crates/online-core/src/ftrl.rs) — *Outputs:* [fields](docs/OUTPUTS.md#ftrl)

FTRL-proximal (McMahan et al. 2013) for binary targets, with its running
sums decayed on the same clock as everything else:

```
β_i = 0 if |z_i| ≤ l1 else −(z_i − sgn(z_i)l1) / ((β + √n_i)/α + l2)
p   = sigmoid(zᵀβ)     g_i = (p − y)·z_i·w
z_i += g_i − ((√(n_i + g_i²) − √n_i)/α)·β_i      n_i += g_i²
```

```python
click = po.spec.ftrl(
    "click", targets=["y"], features=["x0", "x1"], halflife=500.0,
    loss="logistic",             # the default: pred is a probability and resid = y - p. "squared": the linear
                                 # prediction -- a sparse linear regression with no solves and an L1 penalty
    alpha=0.1, beta=1.0,         # the learning-rate scale and its smoothing
    l1=0.01, l2=0.0,             # l1 zeroes a coefficient whose evidence is below it
)
```

The recursion agrees with river's FTRL to 1e-12, row for row.

#### `holt` — Holt's linear trend

*API:* [`po.spec.holt`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.holt) — *Rust:* [`holt.rs`](crates/online-core/src/holt.rs) — *Outputs:* [fields](docs/OUTPUTS.md#holt)

The one model that takes no features: it extrapolates the target's own
level and trend.

```
pred = l + b·Δt
l'   = (λ_l·W·pred + w·y)/(λ_l·W + w)      b' = (λ_b·V·b + w·(l' − l)/Δt)/(λ_b·V + w)
```

```python
baseline = po.spec.holt(
    "baseline", targets=["y"], clock="t", max_dclock=600.0,
    level_halflife=200.0,        # how fast the level forgets, in clock units
    trend_halflife=2000.0,       # how fast the trend forgets; inf is the whole history's drift
)                                # coef is [level, trend] per target
```

Level and trend are weighted means of what each row observes and what the
model forecast, with `W` and `V` the weight each has gathered; a row at
weight `w` counts `w` times. The trend is per clock unit, so on an
irregular clock it extrapolates the right distance. There is no seasonal
term, because a seasonal index is a `group` on the phase, which the bank
already does. Run it in the same bank as a real model to see how much the
regression actually adds: compare `sigma`, or let `emit_selected` choose.

### Moments and correlation

Running moments and correlations of the columns you name, from one pair at
a time to one number for the whole matrix.

#### `ew_cov` — exponentially weighted moments

*API:* [`po.spec.ew_cov`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.ew_cov) — *Rust:* [`ewcov.rs`](crates/online-core/src/ewcov.rs) — *Outputs:* [fields](docs/OUTPUTS.md#ew_cov)

Running moments of the columns you name, on the same clock as every model
here. It takes one O(k²) update per row, where computing every pairwise
exponentially weighted correlation with Polars expressions alone takes
O(k²) *passes over the data*. Values are read from the state before each
row, so an `ew_cov` output can be a feature for that same row without
leaking it.

```
W'   = λW + w        m'ᵢ = (λW·mᵢ + w·xᵢ) / W'      S'ᵢⱼ = (λW·Sᵢⱼ + w·xᵢxⱼ) / W'
varᵢ = Sᵢᵢ − mᵢ²     covᵢⱼ = Sᵢⱼ − mᵢmⱼ             corrᵢⱼ = covᵢⱼ / √(varᵢ·varⱼ)
```

```python
mv = po.spec.ew_cov(
    "mv", features=["x0", "x1", "x2"], clock="t", max_dclock=300.0, halflife=500.0,
    stats=["mean", "std", "corr", "partial_corr", "mahal"],   # any of mean, var, std, cov, corr, partial_corr, mahal;
                                 # default mean + std + corr. [] is legal: learn the moments, write nothing but n_eff,
                                 # and read them back with bank.gram("mv") -- the form for a wide set of columns
    precision_prior=1e-6,        # needed by partial_corr (the correlation of two columns with all the others held fixed,
                                 # read off (C + s*prior*I)^-1, O(k³), paid only when asked) and by mahal; fades as data arrives
    mahal_quantiles=[0.99],      # mahal_q0.99: a running quantile of the Mahalanobis scores, so mahal > mahal_q0.99 is
                                 # "one row in a hundred" without assuming a distribution
    pca=1, pca_every=20,         # pc0_var, pc0_share (of the trace), pc0_<feature> (the loading), pc0_score (this row's);
                                 # the eigendecomposition is O(k³), so refresh it every 20 rows and score the rows between on
                                 # the last loadings; each refresh keeps the previous sign, so a loading never flips
)
scores = po.ModelBank([mv]).fit_predict(df).unnest("mv")
odd = scores.filter(pl.col("mahal") > pl.col("mahal_q0.99"))   # the joint outliers: every column in range, the combination not
first = scores.select("pc0_share", "pc0_x0", "pc0_x1", "pc0_x2", "pc0_score")
```

`mahal` is `√(δᵀ (C + s·prior·I)⁻¹ δ)`, with `δ = x − m`: how far the row
is from what the columns have been doing *together*, in standard
deviations. On Gaussian columns `mahal²` is χ² with `k` degrees of freedom,
and with one column it is `|z|`.

**`window`: a hard cutoff, not a softer decay.** A halflife never forgets
entirely: three halflives back still carries 12.5% of the weight.
`window=w` makes a row older than `w` clock units contribute exactly
nothing:

```
weight(age) = 0.5 ** (age / halflife)   if age <= window
            = 0                          otherwise
```

```python
cut = po.spec.ew_cov(
    "cut", features=["x0", "x1"], clock="t", max_dclock=300.0, halflife=500.0,
    window=1500.0,               # a row older than this many clock units contributes exactly nothing
    window_every=10,             # snapshot every 10 rows: the boundary is the oldest snapshot still inside the window,
)                                # so a coarse cadence discards a little more than asked, never less
```

Inside the window the weights are still exponential, so this is not a flat
rolling mean, and the newest row still dominates. It is exact, because an
exponentially weighted sum contains its own past: everything at or before a
time `u` is `λ^(t−u)` times the running sum as it stood then, so
subtracting that leaves precisely the rest. The model keeps a ring of past
snapshots to do it, the one place here where memory grows with a *window*
rather than with the state: about 3 MB per group for a 1,000-row window
over 20 columns, divided by `window_every`. `window_budget` caps that
memory per ring, in MiB:

| `window_budget` | when a chunk would take a ring past the cap |
|---|---|
| `{"refuse": 64}` | the chunk is refused, and the error names the ring's size and `window_every`. The bank then refuses every later call rather than go on from a chunk it has half learned, so rebuild it from its last save |
| `{"thin": 64}` | the ring drops every other snapshot and doubles its spacing, which, like `window_every`, only ever shortens the window |
| not given | the chunk is refused past 256 MiB |

Four things to know before reading the numbers:

| | |
|---|---|
| the clock is the **decayed** one | the step after `max_dclock` and any `session_gap` |
| the edge is a **discontinuity** | a row ageing out drops its whole weight at once, so the series has small steps that a plain exponential average does not |
| it is a **subtraction** | precision falls with the fraction discarded: negligible at `window = 3h`, worse as the window shortens toward the halflife |
| `n_eff` is the weight inside the window | so `min_periods` gates on something that stops growing, and a clock gap longer than `window` empties it and reports nulls rather than stale numbers |

If the data fits in memory and only the moments are wanted, Polars already
does this: `df.rolling("t", period="3h").agg(...)` with an exponential
weight gives the same number to 1e-14. The reasons to reach for the spec
are a stream, a saved state, or the time: the rolling window recomputes
each window, in `O(n·W)`, where this is `O(n)`, 15.8 s against 14 ms at
200k rows and a 4,680-row window.

**`lags`: how a column moves with another `ℓ` rows ago.** The same
co-moments, kept one step further out. With `W` and `m` the weight and mean
before the row, and both deviations taken against that mean,

```
C_ℓ' = a·C_ℓ + a·b·(x_t − m)(x_{t−ℓ} − m)'
```

with the same `a` and `b` the co-moments use, so lag 0 would be `comoments`
exactly.

```python
lagged = po.spec.ew_cov(
    "lagged", features=["x0", "x1"], halflife=500.0,
    lags=[1, 5],                 # in learned rows within the group, not clock units; strictly increasing, >= 1
    stats=["corr", "lagcorr"],   # lagcorr_<a>_<b>_l<l> per lag and *ordered* pair -- both orders, because a
)                                # lagged matrix is not symmetric: a leading b is not b leading a
lead = po.ModelBank([lagged]).fit_predict(df).unnest("lagged")
# the same numbers from the state: bank.gram("lagged")[0]["lags"], ["lag_comoments"] (an (L, k, k) array)
```

The ring of past rows is emptied on a session change and on a clock gap
beyond `max_dclock`, the two events after which "the row `ℓ` back" no
longer means a row `ℓ` ago. A zero-weight row ages the matrices without
entering the ring. Nothing else moves: clearing the ring is not a reset.

#### `marginal` — every pair's moments, kept in the state

*API:* [`po.spec.marginal`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.marginal) — *Rust:* [`marginal.rs`](crates/online-core/src/marginal.rs) — *Outputs:* [fields](docs/OUTPUTS.md#marginal)

A `marginal` is neither a regression nor a joint fit. It keeps the
exponentially weighted moments of each (feature, target) pair on its own,
as if every pair were a two-column `ew_cov`. For `p` features and `T`
targets that takes O(p·T) time per row, where one `ew_cov` over all the
columns would take O((p + T)²). Per target `t`, on a row where `y_t` is
present, with `W_t` the weight behind that target before the row:

```
W'_t = λW_t + w        a = λW_t / W'_t        b = w / W'_t        Q'_t = λ²Q_t + w²
S'_yy = a·S_yy + a·b·(y_t − m_y)²             S'_xx = a·S_xx + a·b·(x_j − m_x)²
S'_xy = a·S_xy + a·b·(x_j − m_x)(y_t − m_y)    m' = m + b·(value − m)
```

```python
pairs = po.spec.marginal("pairs", targets=["y", "ret"],
                         features=["x0", "x1", "x2", "signal_a", "signal_b"],
                         clock="t", max_dclock=300.0, halflife=500.0, group="stock_id")
bank = po.ModelBank([pairs])
bank.fit_predict(df)                            # the output record holds n_eff alone
table = bank.marginal("pairs")                  # one row per (group, instance, feature, target):
one_stock = bank.marginal("pairs", group="b0")   #   10 rows here: five features by two targets
# n_eff                        the target's W_t, the weight behind its pairs
# n_kish                       W_t² / Q_t: the count of equally weighted rows that carry the same information
#                              ((1 + λ)/(1 − λ) in the limit for unit weights)
# mean_x, var_x, mean_y, var_y, cov    the pair's moments, population form
# corr                         cov / sqrt(var_x * var_y)
# beta                         cov / var_x: the slope of the target on that feature alone
# t                            corr * sqrt((n_kish - 2) / (1 - corr²)): the t-statistic at the Kish sample size --
#                              a scale for comparing pairs, not a p-value; the rows are neither independent nor Gaussian
```

That is `ew_cov`'s arithmetic, so a pair's correlation is the one an
`ew_cov` over the two columns would report, to the bit. A null target ages
its own pairs, `W_t ← λW_t`, and teaches them nothing; a null feature skips
the row, as everywhere. Nothing is written per row but `n_eff`: the pairs
are the state, read back as a table. `corr`, `beta` and `t` are null until
the target's `W_t` reaches `min_periods`, whose default of 3 exists because
two rows give ±1 whatever the data. They are also null wherever they are
undefined: a constant feature, or `n_kish ≤ 2` for `t`. A bank loaded from a
file reports the pairs the bank that saved it would, and one chunk or a
thousand gives the same table to the bit.

Two views sit on top of that, both off unless asked for.

**`lags`: is `t` telling the truth?** `t` is built on `n_kish`, which is
the right count for unequal weights and says nothing about rows that
resemble their neighbours. On a smooth stream, consecutive rows are nearly
the same observation, so `t` claims evidence that is not there. Lags fix
that. The pair's moments are kept at each lag too, the same statistic
`ew_cov(lags=)` computes, to the bit, and `serial_rule` turns them into
Bartlett's correction. Two *independent* AR(1) series with `φ = 0.9` and
`0.8` come out at `t = 2.39` and `t_serial = 1.03`: the first looks like a
finding, and the second is the truth.

**`bins`: what a correlation cannot see.** Everything above is linear, and
a feature can be strongly related to a target with `corr` at zero: a
threshold, a V, a saturation. Bin the feature and keep the target's moments
inside each bin, and all three become visible.

```python
honest = po.spec.marginal(
    "pairs", targets=["y"], features=["x0", "x1"], halflife=500.0,
    lags=[1, 2, 3, 5, 8],        # the pair's moments at each lag, in learned rows within the group
    serial_rule="geometric",     # how the lagged correlations become a count correction
    bins=16,                     # bin each feature and keep the target's moments inside each bin
    bin_rule="quantile",         # edges learned from the first bin_warm_rows rows (default 1,000), by weighted quantile or
    bin_warm_rows=200,           # equal width; or give bin_edges= outright (a list per feature, or a dict by name), which is
)                                # exact and comparable across runs, and refuses bins, bin_rule and bin_warm_rows beside it
# added columns of bank.marginal("pairs"):
#   lagcorr_xx, lagcorr_yy      each series' own autocorrelation, one entry per lag
#   lagcorr_xy, lagcorr_yx      the feature now against the target l rows back, and the reverse -- a feature whose
#                               lagcorr_yx[0] beats its corr *leads* its target; one whose lagcorr_xy[0] does *follows* it
#   n_serial                    n_kish divided by 1 + 2 * sum(rho_x(l) * rho_y(l)) (Bartlett 1935)
#   t_serial                    the same statistic as t, against that count
#   phi_x, phi_y                the fitted per-row decays, under serial_rule="geometric"
#   bin_edges                   the feature's edges, fixed once and never moved
#   bin_n, bin_mean_y, bin_var_y   the target's weight, mean and variance in each bin: the response curve
#   split_gain                  the fraction of the target's variance removed by the best single cut -- a regression
#                               stump's R², so it compares directly with corr² and the difference is the nonlinear surplus
#   split_at                    where that cut falls, in the feature's units
#   split_gain_t                the t a corr would need to match that gain: a ranking, not a p-value -- the cut was
#                               chosen by maximising over the candidates, and the statistic does not know that
```

The lagged lists are `ew_cov`'s `lagcorr` numbers exactly: the lagged
covariance over the two standard deviations, not clamped to `[−1, 1]`,
since a lagged correlation is not bounded by one in a finite sample. A row
where the target is missing ages its weight and holds the lag moments, as
it holds the pair's.

Binning keeps `O(bins)` of state per pair and does one binary search per
pair per row, which is why it can run across ten thousand columns in the
pass that gives them `corr`. A value that carries more than a bin's share,
such as an indicator's zero, fills a bin of its own, and the rest share
what is left, so the 5% of rows that carry the signal are not lost among
the zeros. The warm-up rows are held and replayed, not spent: the histogram
is what it would have been had the edges been known before the first row. A
feature keeps only the bins it can support, so a binary feature has two,
and a constant one has a single bin and no split. Each bin's moments are
kept the way every accumulator here is kept, so a target at `1e7` keeps its
variance. Both views ride into `bank.closed_groups()` as `pair_*` columns:
`pair_split_gain` as a list over the pairs, and `pair_lagcorr_xx` and
`pair_bin_n` as lists of lists.

#### `deco` — one correlation for the whole matrix

*API:* [`po.spec.deco`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.deco) — *Rust:* [`deco.rs`](crates/online-core/src/deco.rs) — *Outputs:* [fields](docs/OUTPUTS.md#deco)

A correlation matrix of `m` series has `m(m−1)/2` free entries. A stream
cannot keep them all moving without O(m²) work a row, and most of them are
estimated from too little data to be worth moving. `deco` (Engle & Kelly
2012) replaces them with their average and estimates that, in O(m) a row.
The row is standardised against the means and variances as they stood
before it, `r_i = (x_i − m_i)/√v_i`. With `S₁ = Σ r_i` and `S₂ = Σ r_i²`
over `n` features, the row's estimate is their Lemma 2.3, and the level
follows one of two dynamics, on the model's own clock:

```
u = (S₁² − S₂) / ((n − 1)·S₂)        = mean of r_i·r_j over i ≠ j, over mean r_i²

"ew":      W' = λW + w,  b = w/W'      ρ' = ρ + b·(u − ρ)
"linear":  ρ' = (1 − α − β)·ρ̄' + α·u + β·ρ      (ρ̄ the "ew" level)
```

```python
eq = po.spec.deco(
    "eq", features=["x0", "x1", "x2"], clock="t", max_dclock=300.0, halflife=500.0,
    dynamics="ew",               # the exponentially weighted mean of u: rho is exactly what an ew_cov(stats=["mean"]) over
)                                # the u sequence would report. "linear": the paper's eq. 21 with correlation targeting,
                                 # which needs alpha= and beta= with alpha + beta < 1
blocked = po.spec.deco(
    "blocks", features=["x0", "x1", "x2", "signal_a"], halflife=500.0,
    blocks={"fast": ["x0", "x1"], "slow": ["x2", "signal_a"]},   # one number per block and one per pair of blocks --
)                                # the useful middle between one correlation and all of them. Every feature in exactly
                                 # one block, and a block needs at least two
out = df.online.fit_predict([eq, blocked])
# u             this row's own estimate, read before the row is learned from
# rho           the level as it stood before the row
# loglik        the row's Gaussian log-density in standardised coordinates under that level
# with blocks:  u_fast, u_slow, u_fast_slow and their rho_* twins, and one loglik over all of them
```

Two things to know. `u` is a **downward-biased** estimate of the
equicorrelation, as the paper says: it is a ratio of two averages, so
`E[u]` is about 0.20 for a true 0.30 at six columns. Use it as a signal
that moves with the market's correlation, not as the correlation itself.
And `rho` is not an `ew_cov`'s `corr` over the columns: the mean of a ratio
is not the ratio of the means, and the gap is large.

#### `rcov` — a block's realised covariance, robust to noise

*API:* [`po.spec.rcov`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.rcov) — *Rust:* [`rcov.rs`](crates/online-core/src/rcov.rs) — *Outputs:* [fields](docs/OUTPUTS.md#rcov)

A realised covariance over ticks is the sum of the outer products of the
returns. Over real tick data it is wrong in two ways. Each price is the
efficient price plus a measurement error, and the error's variance
accumulates with every tick. And when the series are not observed together,
the correlation is pulled toward zero. Published estimators remove both
with sums over lags, which is exactly what a stream can accumulate. `rcov`
has no decay and no per-row output but `n_eff`: its value is the block,
written when the group closes, so it needs `group` and `group_close`, and
the estimate rides in that row.

```python
r = po.spec.rcov(
    "rk", features=["x0", "x1"],       # rows are *returns*: difference upstream
    group="block", group_close="monotone",
    kind="kernel",               # "plain": sum(x x'), equal to n * an ew_cov(lam=1)'s uncentred second moment at close, to the
                                 #          bit -- the cross-check, and the reference the other two are measured against
                                 # "kernel": the multivariate realised kernel (Barndorff-Nielsen, Hansen, Lunde & Shephard
                                 #          2011), sum_h k(h/(H+1)) Gamma_h with Parzen weights and jittered end points
                                 # "preavg": the modulated realised covariance (Christensen, Kinnebrock & Podolskij 2010):
                                 #          returns pre-averaged over k_n = floor(theta sqrt(n)), less the residual bias
    block_rows=2000,             # a sizing hint for the ring, needed when bandwidth= is left out: a longer block runs,
                                 # clipped, and reports bandwidth_used
    bandwidth=None,              # a fixed H; left out, H = ceil(c* xi^(4/5) n^(3/5)) with c* = 3.5134
)
bank = po.ModelBank([r])
bank.fit_predict(by_block.select("x0", "x1", "block"))
blocks = bank.closed_groups()
# rcov, rcorr                  vech of the upper triangle
# rcov_n, rcov_kind, bandwidth_used
# omega2, iv_sparse            the noise variance and sparse integrated variance behind the bandwidth
# iq                           a realised-quarticity proxy, labelled one
# psd_repaired                 whether the estimate had to be made positive semi-definite
```

Parzen is the only kernel: the Bartlett kernel is not consistent for this
estimator, and Parzen's 0.97 efficiency beats the quadratic spectral's
0.93. A block too short to estimate from gives nulls, not an error. Nothing
reads a future row: the jittered *end* point is formed at close from
observations already in the state, and a product enters `Γ̂_h` only once
both its legs are final. `weight` is read as 0 or 1 only, since a sum over
returns has no fractional row.

### Clustering and classification

Labels for rows: clusters found with no target, or classes learned from a
label column.

#### `kmeans` — exponentially weighted k-means

*API:* [`po.spec.kmeans`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.kmeans) — *Rust:* [`cluster/kmeans.rs`](crates/online-core/src/cluster/kmeans.rs) — *Outputs:* [fields](docs/OUTPUTS.md#kmeans)

A model with no target: it labels each row with the nearest of `k` centres,
read before the row is learned from, so the label is out-of-sample like
every prediction here. Each centre is the exponentially weighted mean of
the rows assigned to it, which is `ew_cov`'s mean recursion, per cluster.

```
j*   = argmin_j ‖x − c_j‖²          distances in units of each feature's EW sd
n'_j = λn_j + w                      c'_j = c_j + (w/n'_j)(x − c_j)     for j = j*
```

```python
km = po.spec.kmeans(
    "km", features=["x0", "x1", "x2"], clock="t", halflife=2000.0, max_dclock=300.0,
    k=3,
    warm_rows=100,               # seeding waits for this many rows (default 500), places the centres, replays the rows
    seed_rule="lloyd",           # the best of ten k-means++ starts, by inertia: one start lands in the wrong partition
                                 # a third of the time on five blobs in four dimensions
    split_merge=0.5,             # every split_merge_every rows, if the two closest centres are nearer than this times the
    split_merge_every=200,       # sum of their radii -- two centres in one blob -- one is freed and placed on the rows
                                 # far from every centre. 0: plain sequential k-means
    dead_frac=0.05,              # a centre whose blob vanished fades; under this share of an equal share it is re-placed.
)                                # that takes log2(1/dead_frac) halflives: 4.3 at 0.05, 2 at 0.25
out = po.ModelBank([km]).fit_predict(df).unnest("km")
# cluster   the nearest centre's label, before the row is learned from
# dist      the distance to it;  dist2  the distance to the second-nearest
# n_eff, coef = the centres, k rows of len(features)
po.spec.coef_index(km)        # target = "cluster0".., term = the feature
```

A row far outside its cluster, about four standard deviations of `dist²`
above the typical radius, is scored but not learned from: it is set aside
for the split–merge move. Raise `dead_frac` when regimes change faster than
the fade allows; a cluster lighter than `dead_frac/k` of the stream then
loses its centre whenever any row is far. The move cannot see one centre
owning two blobs, whose rows are all within its own radius, and seeding
with `lloyd` is what prevents that.

#### `micro` — density-based clustering, any shape

*API:* [`po.spec.micro`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.micro) — *Rust:* [`cluster/micro.rs`](crates/online-core/src/cluster/micro.rs) — *Outputs:* [fields](docs/OUTPUTS.md#micro)

`kmeans` needs `k` and finds round clusters. `micro` finds clusters of any
shape without being told their number, flags the rows that belong to none,
and follows clusters that appear and vanish. It is DenStream's
micro-clusters with a linking step over them. Each micro-cluster is a
*summary*: a decayed weight `n`, a centre `c`, and a radius `r`, the
exponentially weighted root-mean-square distance of its rows from the
centre. Each row goes to the nearest summary that can take it without its
radius passing `eps`, in units of each feature's exponentially weighted
standard deviation. If none can, the row opens a new summary.

```
n_j  ← λ n_j                                               every summary
j*   = nearest summary that keeps  a r²_j + a b ‖x − c_j‖² ≤ eps² p,
       a = n_j/(n_j + 1),  b = 1/(n_j + 1);  else a new one at x
n_j* ← n_j* + w     c_j* ← c_j* + (w/n_j*)(x − c_j*)     r²_j* ← min(·, eps² p)
```

```python
mc = po.spec.micro(
    "mc", features=["x0", "x1"], clock="t", halflife=2000.0, max_dclock=300.0, min_periods=50.0,
    eps=0.1,                     # the spread the model reads as *one* cluster, per standardized coordinate:
                                 # about 0.07 for two-dimensional shapes, 0.3 for well-separated Gaussians in 20 dimensions
    beta_mu=5.0,                 # a summary with at least this much weight is established
    prune_every=100,             # every this many rows: drop the light summaries, link the established ones -- centres
    macro_link=None,             # within L of each other share a label. L is read from the spacing the summaries show
)                                # unless macro_link sets it (2 = link only summaries that touch)
out = po.ModelBank([mc]).fit_predict(df).unnest("mc")
out.select("cluster", "outlier", "n_clusters", "n_micro").tail(3)
# cluster              the label of the nearest established summary; null while there is none
# dist                 the distance to that summary's centre
# micro                the id of the summary this row goes to; ids only go up and are never reused
# outlier              no established summary takes the row
# n_clusters, n_micro  how many of each the state holds
# coef                 the established summaries, one [id, label, n, radius, c_1 .. c_p] row each
```

Every output is read before the row is learned from. A label is the
smallest id in its chain, so it outlives everything but that summary. Both
ways to get `eps` wrong show in the outputs:

| what you see | `eps` is | what to do |
|---|---|---|
| nearly every row is an `outlier`, and `cluster` stays null | too small: no summary reaches `beta_mu` before it is pruned | raise `eps` |
| `n_micro` is about the number of clusters | too coarse: each cluster is one summary, so the derived `L` reads the spacing *between* clusters and bridges them into one | lower `eps`, or set `macro_link=2` |

Measured at 20k rows with the `eps` above: moons, rings and five Gaussians
in twenty dimensions all score ARI 1.000 against the truth, where `kmeans`
cannot follow the first two. Noise drawn uniformly over the box is flagged
`outlier` 94% of the time, and real rows 0.3% of the time. A cluster born
mid-stream has a label within 200 rows, and one whose rows stop lingers
`halflife · log2(n / beta_mu)`, with `n` the weight it had.

#### `ew_class` — Gaussian classification on `ew_cov` moments

*API:* [`po.spec.ew_class`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.ew_class) — *Rust:* [`ewclass.rs`](crates/online-core/src/ewclass.rs) — *Outputs:* [fields](docs/OUTPUTS.md#ew_class)

A label column in place of a numeric target. The model keeps one `ew_cov`
state per class, a weight `n_c`, a mean `μ_c` and a centred covariance
`C_c`, and scores a row by Bayes' rule over Gaussian classes.

```
π_c = n_c / Σ n         r_c = precision_prior · s_c        (s_c: the prior's fade)
M_c = C_c + r_c I  (full)      M = Σ π_c M_c  (shared)      diag(C_c) + r_c  (diagonal)
ℓ_c = ln π_c − ½ ln det M_c − ½ (x − μ_c)ᵀ M_c⁻¹ (x − μ_c)
p_c = exp(ℓ_c − max ℓ) / Σ exp(ℓ − max ℓ)                  class = argmax ℓ
n_c ← λ n_c + w·[y = c]        μ_c, C_c ← weighted Welford on the row's own class
```

```python
labelled = df.with_columns(
    pl.when(pl.col("y") > 0).then(pl.lit("up")).otherwise(pl.lit("down")).alias("dir")
)
cl = po.spec.ew_class(
    "cl", features=["x0", "x1", "x2"], clock="t", halflife=200.0, max_dclock=300.0, min_periods=20.0,
    label="dir",                 # the label column; a null label scores the row and learns nothing from it
    classes=["down", "up"],      # declared up front: a label not in the list is an error naming the row.
                                 # integer and boolean columns work through their text: ["0", "1"], ["true", "false"]
    covariance="shared",         # "full": each class its own covariance (QDA); "shared": pooled by class weight (LDA);
                                 # "diagonal": variances only (Gaussian naive Bayes)
    precision_prior=0.1,         # the ridge that makes a class scoreable from its first row; fades as ew_cov's does
)
out = po.ModelBank([cl]).fit_predict(labelled).unnest("cl")
out.select("dir", "class", "p_up", "n_eff").tail(3)
# class        the most probable class, as a string
# p_<class>    one per declared class; exactly 0 for a class no row has carried yet
# coef         the class means, in the order of classes (coef_up_x0 after unnest)
```

Every output is read before the row is learned from, so a row's
probabilities never saw its own label. That is also how to score a stream
whose labels arrive late: null the label and keep the features.

| `covariance` | what each class keeps | the right choice when | measured, 400k rows, 6 features, 3 classes |
|---|---|---|---|
| `"full"` | its own covariance: one `k×k` Cholesky per class per row | in general | 0.9M rows/s |
| `"shared"` | one covariance pooled by class weight, factorized once per row | the classes differ in location but not in spread; it then matches `"full"` to a fraction of a percent on the test data, with fewer parameters to learn | 1.8M rows/s |
| `"diagonal"` | variances only | speed matters most; it cannot see a correlation, so two classes with the same marginals and opposite correlations are one class to it | 5M rows/s |

On three Gaussian classes with their own covariances, the accuracy sits
within 0.001 of the best any classifier could do given the generating
parameters, and the probabilities are calibrated to about 0.01.

### Sequential tests and regimes

Evidence that something holds or has changed, which you can read at any
row, and which regime the stream is in.

#### `seqtest` — a sequential test of a sign, by betting

*API:* [`po.spec.seqtest`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.seqtest) — *Rust:* [`seqtest.rs`](crates/online-core/src/seqtest.rs) — *Outputs:* [fields](docs/OUTPUTS.md#seqtest)

Not a regression. A `seqtest` asks whether a column tends to be positive,
or, with `a` and `b`, whether one spec of the bank predicts closer than
another. Either way the answer is evidence you can read at any row, as
often as you like, and act on the first time it is enough. A p-value cannot
be used that way, because checking it repeatedly inflates its error rate;
an *e-process* can, which is the reason to reach for this model. Per target
it keeps the wealth of two gamblers, one betting that the next sign is
positive and one that it is negative. Each stakes the Krichevsky–Trofimov
fraction set by the counts so far, and never bets against its own lead:

```
s = sign(y)                    n⁺, n⁻: the signs counted before this row,  n = n⁺ + n⁻
λ⁺ = max(0, (n⁺ − n⁻) / (n + 1))          λ⁻ = max(0, (n⁻ − n⁺) / (n + 1))
ln E⁺ ← ln E⁺ + ln(1 + λ⁺ s)              ln E⁻ ← ln E⁻ + ln(1 − λ⁻ s)
```

```python
common = dict(targets=["y"], features=["x0", "x1"], clock="t", max_dclock=300.0, group="stock_id")
ridge = po.spec.ewridge("ridge", halflife=500.0, **common)
kalman = po.spec.kalman("kalman", halflife=500.0, coef_halflife=100.0, **common)

sign = po.spec.seqtest("sign", targets=["y"], group="stock_id")     # does y tend to be positive?
# log_e_pos_y, log_e_neg_y     the two gamblers' log wealth, as they stood before the row
# n_pos_y, n_neg_y             the signs counted so far; a zero, null or NaN is a tie: bets nothing, counts nothing

closer = po.spec.seqtest("closer", targets=["y"], a="kalman", b="ridge", group="stock_id")   # does kalman predict closer?
# log_e_a_y, log_e_b_y, wins_a_y, wins_b_y   the sign tested is |resid_b| - |resid_a|, positive when a came closer,
#                                             on the out-of-sample residuals the two specs' output records report;
#                                             a row where either side is null (warm-up, a skipped row) is no trial
out = po.ModelBank([ridge, kalman, closer]).fit_predict(df)
verdict = out.group_by("stock_id").agg(pl.col("closer").struct.field("log_e_a_y").max())
# log_e_a_y >= ln(20): on that stock, kalman beat ridge at the 5% level, read at any row
```

Under the null, that given everything so far the next sign is no more
likely positive than negative, `E⁺` is a nonnegative supermartingale, and
Ville's inequality gives `P(E⁺ ever reaches 1/α) ≤ α`. So
`log_e_pos ≥ ln 20` rejects at the 5% level however many times you looked,
and however the rows depend on each other. No distribution is assumed, and
the size of the values is invisible: 60% small gains and 40% huge losses is
"positive". Where the clip never binds, the wealth has the closed form
`2ⁿ B(n⁺+½, n⁻+½) / π`, the Beta(½, ½) mixture, and the bank is held to it.
The two sides' average is an e-value for the two-sided question.

A trial is a row, so there is no `weight` and no `halflife`, and a spec that
gives them is refused. `session`, or `on_clock_reset="reset_state"`,
restarts the test. `a_suffix` and `b_suffix` pick a grid instance, such as
`"@h500"` or `"__r0.5@h500"`. A comparison inside a bank is
chunk-invariant, saved with the state, and works a chunk at a time like
everything else.
[`po.eval.seqtest`](https://hgilde.github.io/polars-online/eval.html#polars_online.eval.seqtest)
is the same computation over a frame you already have.

#### `corrchange` — has the correlation structure changed?

*API:* [`po.spec.corrchange`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.corrchange) — *Rust:* [`corrchange.rs`](crates/online-core/src/corrchange.rs) — *Outputs:* [fields](docs/OUTPUTS.md#corrchange)

Two tests, for two questions: has the correlation been constant, and how
big is the change.

`kind="monitor"` is the **closed-sample** constancy test of Wied, Krämer
and Dehling (2012), run over consecutive spans of `span_rows` rows. At the
last row of a span, per pair:

```
Q = max_{2≤j≤T} (j/√T)·|ρ̂_j − ρ̂_T| / D̂
```

with `ρ̂_j` the correlation of the span's first `j` rows, and `D̂` the
delta-method long-run standard deviation of `ρ̂`. Under the null, `Q`
converges to `sup|B|` for a Brownian bridge `B`, so the critical value is
the Kolmogorov quantile. It is computed from the series, not pinned, and it
reproduces the published 1.3581 at 5%. The test's size and power are held
to the paper's own tables: `.035` at ρ = 0 and `T = 500`, and `.587` power
on a `0.5 → 0.7` break.

```python
c = po.spec.corrchange(
    "break", features=["x0", "x1"],
    kind="monitor",              # the constancy test above, or "window": how *big* the change is, below
    span_rows=500,               # nothing is reported until a span closes: a delay of at most this many rows
    scalar=False,                # True: run the test on the equicorrelation of the standardised row (deco's u) --
)                                # one statistic however many columns, and a test of its level rather than of a pair
out = df.online.fit_predict([c]).unnest("break")   # stat, crit, flag, since_flag
```

```python
w = po.spec.corrchange(
    "size", features=["x0", "x1"],
    kind="window",               # ||vech(R_pre - R_post)|| over two adjacent windows of span_rows rows each ...
    span_rows=100,
    n_perm=200, permute_every=500,   # ... against a permutation quantile: n_perm shuffles of the pooled rows between the
    perm_block=10,               # windows, in blocks of perm_block so rows that resemble their neighbours do not make the
)                                # null too liberal; or give crit= as a number and skip the permutations entirely
```

The window kind's null is a permutation, not a sign flip, because negating
a whole row leaves every correlation exactly where it was. Its flag rate
per row is not `alpha`: two windows that slide by one row are almost the
same windows, so a statistic above the quantile stays above it for a run
of rows.

#### `hmm` — which regime are we in

*API:* [`po.spec.hmm`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.hmm) — *Rust:* [`hmm.rs`](crates/online-core/src/hmm.rs) — *Outputs:* [fields](docs/OUTPUTS.md#hmm)

`ew_class` classifies a row against *labelled* Gaussians. An `hmm` does the
same arithmetic with no labels: the state is hidden, and a transition
matrix carries information from one row to the next. That is the
difference between "which regime does this row look like" and "which
regime are we in", and the second is usually the question. It runs
Hamilton's filter one row at a time, from the filtered `p` the previous row
left:

```
p1_l   = Σ_k p_k·Π_kl                      the predicted state
f_l    = N(x | μ_l, Σ_l + r_l·I)           the state's density
loglik = ln Σ_l p1_l·f_l                   the row's surprise
p_l   ← p1_l·f_l / Σ                       the filtered state
```

```python
h = po.spec.hmm(
    "regime", features=["x0", "x1"], halflife=500.0,
    k=2,                         # the number of hidden states
    precision_prior=1e-2,        # required: a state's centred co-moments start at zero, and a zero matrix has no density
    warm_rows=400,               # seeds the states from this many learned rows with kmeans' rule; every output is null
                                 # until then. Should span more than one regime, or the seeds are two halves of one
    means=None, covs=None,       # or give the states outright, and learn=False to freeze them
)                                # exog_tvtp= drives the transition matrix from a column, through fixed tvtp_coef=
out = df.online.fit_predict([h]).unnest("regime")
# p_0, p_1      the filtered state, before the row;  p1_0, p1_1   the predicted state
# state         the most probable one;  loglik   the row's surprise
```

Everything reported is read before the row is learned from. Each state's
running sums then take the row at weight `w·p_l`. The responsibilities sum
to `w`, so `n_eff` is the shared recursion untouched: a row splits across
the states rather than counting more than once. The transition matrix is
learned from the **filtered joint of consecutive states**,
`ξ_kl = p_k(t−1)·Π_kl·f_l / Σ`, with a Dirichlet pseudo-count keeping a
never-visited row a distribution. The transitions are what separate this
from a clustering: on two-dimensional blobs 1.5 apart, a memoryless
nearest-centre rule *given the true centres* is 85% right, and the filter is
99% right.

One limitation: a single extreme row can be captured by one state, and in
mean form a state with zero responsibility keeps its moments. So a state
that stops winning never forgets, and the mixture is left one state short.
A larger `precision_prior`, given states, or cleaning upstream are the
mitigations.

#### `bocpd` — how long has this regime lasted?

*API:* [`po.spec.bocpd`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.bocpd) — *Rust:* [`bocpd.rs`](crates/online-core/src/bocpd.rs) — *Outputs:* [fields](docs/OUTPUTS.md#bocpd)

Every other detector here answers "has something changed?" with a
statistic. `bocpd` (Adams & MacKay 2007) keeps a probability distribution
over the **run length**, how many rows since the last break, so the answer
carries the age of the regime with it. "We are forty rows into a regime" is
different information from "something broke". Their Algorithm 1, with
`H = 1/hazard` and `π_r` run `r`'s posterior predictive for this row:

```
growth:      P(r_t = r+1, x_1:t) = P(r_t-1 = r, x_1:t-1)·π_r·(1 − H)
changepoint: P(r_t = 0,   x_1:t) = Σ_r P(r_t-1 = r, x_1:t-1)·π_r·H
```

Each run keeps its own conjugate sufficient statistics, so slot `r` holds
exactly the `r` rows that hypothesis says came before this one in the run.
Slot 0 holds none, so its predictive is the prior's, which is what makes
"a new run starts here" a hypothesis the data can vote on.

```python
b = po.spec.bocpd(
    "regime", features=["ret"], group="stock_id",
    hazard=250.0,                # the expected run length: H = 1/hazard is the per-row chance of a break
    prior_nu=2.0,                # the prior on the variance, as 2a and 2b in the gamma parametrisation -- how Adams and
    prior_scale=[2e-4],          # MacKay give their own finance example (a = 1, b = 1e-4, hazard = 250). prior_scale is the
                                 # one parameter you must set from your data: too large and no row is ever surprising
    emission="diag",             # a normal-inverse-gamma per feature; "gaussian": a normal-inverse-Wishart over all of them,
                                 # O(runs d²) a row and the one that can see a break in the *correlation* alone;
                                 # "robust": each row's contribution weighted by (pi(x)/pi(mode))**robust_beta, so one 20-sigma
                                 # row moves nothing (without it, that row is a changepoint at p_change 0.91)
    prune_below=1e-6,            # drop the runs holding less than this share of the mass: what makes the model finite
    max_run=None,                # fold every longer run into the last kept one: caps how much history any run holds
    hazard_col=None,             # read the hazard per row from a column, declared in the target slot the way a weight is
)
out = df.online.fit_predict([b]).unnest("regime")
run_started_at = pl.int_range(pl.len()) - pl.col("run_mode")
# p_change      P(r_t <= 1) given this row: the alarm
# run_mode      the most likely run length, before the row -- so t - run_mode is the row the run began on
# run_mean      the posterior mean run length, before the row
# pred_<f>      the pre-row predictive mean of each feature, mixed over runs
# logscore      the row's log predictive density under that mixture
```

**`run_mode` is the answer, and `p_change` is the alarm.** They are not the
same quality of signal. `p_change` is a per-row likelihood ratio, so it is
spiky, and its height depends on the size of the break against the prior
scale. A ten-fold variance step takes it to 0.83 on the row itself, a
four-sigma mean shift with a diffuse prior barely lifts it, and a change in
correlation alone never moves it at all. The run length finds all three,
one to three rows later, and dates each to the right row. `p_change` is
`P(r ≤ 1)` rather than `P(r = 0)` because the changepoint branch and the
growth branch share the same predictive, which makes the normalised mass at
`r = 0` *exactly* `H` on every row, whatever the data. Row one of a group
reports nothing, since `P(r ≤ 1)` is 1 there however the row looks.

`robust_beta` is a trade-off: a whole new regime is a run of individually
forgiven rows, so above about 0.2 nothing is ever detected again. The
default of 0.1 ignores the outlier and still dates a four-sigma shift to
the right row.

## Performance

### Throughput

Apple M-series, single process, best of 3, 200k rows per run
(`uv run python scripts/benchmark.py --markdown`):

| configuration | notes | rows/sec |
|---|---|---|
| `ewridge` k=5 | 1 target, 1 halflife | 10,306,024 |
| `ewridge` k=20 | 1 target, 1 halflife | 4,122,355 |
| `ewridge` k=50 | 1 target, 1 halflife | 1,040,742 |
| `ewridge` k=20 | 10 targets | 2,340,621 |
| `ewridge` k=20 | 5 halflives | 2,355,548 |
| `rls` | k=20, 1 target | 1,843,129 |
| `kalman` | k=20, 1 target | 2,122,805 |
| `lasso` | k=20, 1 target (3-point path) | 2,060,329 |
| `huber` | k=20, 1 target | 4,175,881 |
| `ftrl` | k=20, 1 target | 6,006,156 |

Targets share one set of feature sums, so 10 targets take far less than 10×
the time of one. Each halflife in a grid is its own set of sums, but they
run in parallel, so a 5-halflife grid takes about 2× the time of one rather
than 5×. `rls` is 1.3–2.1× slower for its square-root form, which is what
keeps one extreme row from destroying it by cancellation.

The other families, and the options that add work, on the same machine and
rows:

| configuration | notes | rows/sec |
|---|---|---|
| `ewridge` + `conformal` | k=20, 90% interval | 4,097,559 |
| `sgd` | k=20, squared loss | 8,961,276 |
| `sgd` | k=20, `coef_min=0`, `coef_sum=1` | 2,481,671 |
| `pa` | k=20 | 11,150,059 |
| `kalman` | k=20, `revert_halflife` | 1,844,755 |
| `ew_cov` | k=20: mean, std, corr (230 statistics) | 2,117,751 |
| `ew_cov` | k=20: mean, mahal, `mahal_q0.99` | 750,903 |
| `ew_class` | k=20, 3 classes, full covariance | 495,968 |
| `ew_class` | k=20, 3 classes, shared covariance | 577,082 |
| `ew_class` | k=20, 3 classes, diagonal | 2,824,922 |
| `kmeans` | 4 features, K=8 | 6,118,711 |
| `kmeans` | k=20, K=8 | 3,103,963 |
| `micro` | 4 features, `eps=1` | 15,060,666 |
| `seqtest` | sign of one column | 22,436,196 |

A conformal interval adds nothing measurable, because it reads the residual
the model already has. A simplex constraint sorts `2k` breakpoints per row,
which makes `sgd` about 4× slower. The Mahalanobis distance and the
full-covariance `ew_class` each factor a `k × k` matrix by Cholesky: once
per row for `mahal`, and once per *learned* row for `ew_class`. `kmeans` and
`micro` compute a distance to each centre, and `seqtest` a handful of
operations.

The correlation families, on the same machine and rows:

| configuration | notes | rows/sec |
|---|---|---|
| `deco` | k=20, one equicorrelation | 2,666,809 |
| `deco` | k=20 in 4 blocks | 1,892,836 |
| `rcov` | 4 features, kernel, blocks of 1000 | 3,864,849 |
| `ew_cov` | k=20: mean, cov, lags 1–5 | 1,148,104 |
| `hmm` | 4 features, K=2 | 1,343,086 |
| `hmm` | k=20, K=2 | 353,258 |
| `bocpd` | 4 features, diagonal | 1,009,075 |
| `bocpd` | 4 features, full covariance | 585,180 |
| `corrchange` | 4 features, monitor, `span_rows=500` | 388,557 |
| `corrchange` | 4 features, window 100, permute every 500 | 187,407 |

`deco` is one number for the whole matrix, computed in `O(m)` a row, which
is why it runs at `ew_cov`'s speed and not at a covariance matrix's. `rcov`
accumulates per row and computes its kernel only when the block closes.
`hmm` factorizes a `k × k` covariance per state per row, which is
`ew_class`'s work with the classes hidden. `bocpd` does `O(runs · d²)` work
a row, and `prune_below` is what keeps it finite: at `prune_below=0` the
run vector grows by one entry every row and the model is `O(rows²)`, while
at the default it is flat in the length of the stream. The setting is a
direct dial on throughput, on i.i.d. Gaussian rows:

| `prune_below` | rows/sec |
|---|---:|
| `1e-8` | 204,000 |
| `1e-6`, the default | 324,000 |
| `1e-4` | 687,000 |

`bocpd` is faster on data that breaks, because a changepoint collapses the
distribution onto a short run. `corrchange`'s window kind is the slowest
model here, deliberately: its permutation null redraws `n_perm` statistics
every `permute_every` rows, and giving `crit` as a number skips that
entirely. Where the time goes, and what to reach for, is in
[docs/PERFORMANCE.md](docs/PERFORMANCE.md).

### Memory: which calls stream

Every way of running a bank works a chunk at a time. Measured as the most
memory the process ever held, on one file of `ewridge` with 20 features,
parquet in and parquet out:

| what you write | 3M rows | 12M rows | |
|---|---:|---:|---|
| `lf.online.fit_predict([spec])` | 0.90 GB | 1.35 GB | the bank inside a query |
| `for chunk in lf.collect_batches(): bank.fit_predict(chunk)` | 0.80 GB | 1.24 GB | your own loop |

[docs/RUNNER.md](docs/RUNNER.md) has the same measurement for the command
line, and it is flat too.

Memory is three things: the state, the chunks in flight, and whatever
Polars' reader has read ahead. What growth the two rows show is not the
bank's: it is the memory allocator keeping pages it has freed, and nearly
all the rest is Polars reading ahead in the parquet file. The read-ahead is sized from Polars' thread count, so
`POLARS_MAX_THREADS` shrinks it ([Parallelism](#parallelism)), and the
settings below tune it directly. Every step after the bank in a query,
such as a filter, a join, a group-by or writing the result, runs a chunk at
a time, as Polars itself does. Polars' rules hold for the steps *around*
the bank too: a rolling window over groups, with `.over("group")` or
`group_by=`, makes Polars hold every row, 6.5 GB and 1.7 GB on the same
rows against 0.25–0.28 GB without groups. A bank's `group=` keeps one set of
running sums per group instead, and grows with the number of groups, not
the number of rows.

### Tuning memory with Polars' own settings

A bank's own memory is its state plus the chunks in flight, and neither
grows with the stream. What does grow is Polars' read-ahead: the streaming
engine prefetches row groups ahead of whatever consumes them, sized from
the thread count, and while a bank is the slowest step, a local disk needs
none of it. Three environment variables move it. All are Polars' own, all
are read at run time rather than at import, and all can be set from
Python:

```python
import os

os.environ["POLARS_ROW_GROUP_PREFETCH_SIZE"] = "1"   # row groups read ahead: the lever that matters
os.environ["POLARS_MAX_THREADS"] = "4"               # scales the same term, since the prefetch is sized from it
os.environ["POLARS_ROW_GROUP_PREFETCH_KBYTES_BUDGET"] = "65536"   # a byte cap on the same read-ahead
```

Measured on 8M rows by 12 columns in 80 row groups, as peak resident
memory, with the allocator's page retention off, so that the figure is live
data rather than the high-water mark of everything ever allocated:

| what runs | default | prefetch 1 |
|---|---:|---:|
| `lf.online.fit_predict(...).sink_parquet(...)` | 1.63 GB | 1.12 GB |
| `bank.fit_predict_batches(lf)` | 1.41 GB | 1.07 GB |

The prefetch is read per scan, not once at import, so unlike
`POLARS_MAX_THREADS` it can be set at any point before the scan that should
use it, and a scan that has already run does not lock it in. Setting it in
the shell, before the import or after the import all give the same number.

Two things to know before comparing with other measurements. The gain
depends on how much data one row group holds:
[docs/PERFORMANCE.md](docs/PERFORMANCE.md) reports a larger reduction on a
file whose row groups hold 262,000 rows, where pinning the prefetch takes a
run from 1.86 GB to 0.51 GB. So measure on your own files rather than
carrying either ratio across. And the byte budget counts *compressed*
bytes, which cost almost nothing for a memory-mapped local file, so it
rarely binds.

### Chunk size

`chunk_rows` is how many rows the bank takes at a time: a keyword on
`lf.online.fit_predict`, `lf.online.predict` and
`ModelBank.fit_predict_batches`, 100,000 by default. With
`ModelBank.fit_predict(df)`, the chunk is whatever frame you pass. It never
changes the numbers: one chunk or a thousand gives the same output, and
only where `coef` lands moves, since each stream reports its coefficients
on its last row of every chunk. It does change the speed and the memory.
Each chunk carries a fixed overhead, of handing the frame across from
Polars, gathering the columns and assembling the output, so tall chunks
spread it thinner. On wide frames that hand-off is about 8 ms per call at
10,000 columns: 4 µs of every row at 2,000 rows per call, and 0.4 µs at
20,000 ([docs/PERFORMANCE.md](docs/PERFORMANCE.md) §20). Three chunks are
in flight at once, so `chunk_rows` also sizes the middle one.

### Parallelism

The unit of work is a *stream*: one spec on one group, or one stream per
spec with no `group`. On every chunk, each stream in the bank becomes one
task on the bank's own thread pool, which is separate from Polars' own. It
is one flat pool across all specs and all groups, longest stream first, so
a few big groups do not leave cores idle at the end. Within a stream the
rows go one at a time, because each row's update depends on the last,
which is what makes the numbers independent of how the work is split. It
also means a bank with one spec and one group is one thread's work per
chunk, while Polars' own reading and writing still run in parallel around
it. So a bank fills the pool with groups, with specs, or with both.

A search over factor sets is a list of specs, one per set. Each spec is its
own set of running sums, with its own standardization and its own grid
inside, and each is one task. The list runs as one query in one pass, with
the thread counts set before anything is built:

```python
import os
os.environ["POLARS_ONLINE_MAX_THREADS"] = "8"   # the bank's pool: read at the first bank call
os.environ["POLARS_MAX_THREADS"] = "8"          # polars' readers and writers: read at import

import polars as pl
import polars_online as po
from itertools import product

factors = {"mkt": ["x0"], "mkt-sz": ["x0", "x1"], "mkt-sz-val": ["x0", "x1", "x2"]}

def spec(name, features, standardize):
    return po.spec.ewridge(f"{name}-std{standardize:d}",
                           targets=["y"], features=features, clock="t", max_dclock=300.0,
                           group="stock_id", session="session", session_gap=60.0,
                           halflife=[100.0, 1000.0], ridge=[1e-3, 0.1],   # gridded inside the spec
                           standardize=standardize)

specs = [spec(n, f, s) for (n, f), s in product(factors.items(), [False, True])]

(pl.scan_parquet("ticks.parquet")
   .online.fit_predict(specs, chunk_rows=200_000, save_state="grid.state")
   .sink_parquet("grid.parquet"))

scores = po.eval.compare_specs(pl.read_parquet("grid.parquet"),
                               [s["name"] for s in specs]).sort("r2", descending=True)
```

Every chunk puts 6 × 64 stream tasks on the pool. On 2.56M rows over 64
groups, that query takes 12.3 s at one thread and 2.2 s at fourteen. The
output is one column per spec, which is what `compare_specs` reads, and one
state file holds them all. Where the parallelism comes from:

| source | measured |
|---|---|
| groups | k=20 over 64 groups: 1.02M, 1.91M, 3.52M, 6.44M and 8.20M rows/s at 1, 2, 4, 8 and 14 threads, 8.0× on a 14-core machine |
| specs | eight single-group specs in one bank run in 130 ms, against 515 ms one at a time |
| halflives | each halflife in a grid is its own set of running sums, and the instances of a stream run alongside each other. Ridge and feature-set grids share one set of sums and are expanded at solve time, so they need no thread |
| Python | Python's global lock is released while a chunk is in the bank, so a Python reader thread can run ahead of `ModelBank.fit_predict` |

The thread count is `POLARS_ONLINE_MAX_THREADS` for the bank's pool and
`POLARS_MAX_THREADS` for Polars' readers and writers; unset, each is one
thread per core. The bank builds its pool at the first bank call, and
Polars builds its own at import, so each must be set before that point:
as above, or in the shell (`POLARS_ONLINE_MAX_THREADS=8 python fit.py`),
which is the form that always works. Set later, the variable is ignored,
and [`po.thread_pool_size()`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.thread_pool_size)
says what took effect (`pl.thread_pool_size()` for Polars'). A value that
is not a count is refused by name at the first bank call. Threads change
the speed and nothing else: the same stream at 1 and 8 threads, in separate
processes, gives identical output. Everything runs in one process; there is
no distributed execution, by design ([What this is
not](#what-this-is-not)).

The two counts do different things. Polars' count also sizes how much of a
parquet file its reader holds in flight, so more threads means a bigger
pile of decoded rows, while the bank's count changes only the speed. A run
that has to fit in a smaller box keeps Polars small and gives the bank
every core:

```python
import os
os.environ["POLARS_MAX_THREADS"] = "4"           # the reader's read-ahead is sized from this
os.environ["POLARS_ONLINE_MAX_THREADS"] = "14"   # the bank still has every core

import polars as pl
import polars_online as po

(pl.scan_parquet("ticks.parquet")
   .online.fit_predict([spec], chunk_rows=200_000)
   .sink_parquet("fit.parquet"))
```

On 12M rows over 64 groups with one spec:

| Polars threads | bank threads | time | peak memory |
|---:|---:|---:|---:|
| 14 | 14 | 2.6 s | 1.1 GB |
| 4 | 14 | 2.6 s | 0.8 GB |

Keeping Polars at four threads uses a third less memory at the same speed.
The pools never wait on each other, because a bank task never calls back into Polars'
pool, so giving both more threads than there are cores slows neither.

### Against scikit-learn

The model most people compare this to is
[`SGDRegressor.partial_fit`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html),
and a fair comparison starts by naming what each side is. `SGDRegressor` is
a first-order stochastic optimiser, whose answer depends on the learning
rate, the schedule, the feature scaling and the row order. The primary
regressions here are a different class of algorithm: `ewridge`, `rls`,
`lasso`, `huber` and `quantile` accumulate sufficient statistics and solve,
so there is no learning rate, and with decay off `ewridge` is ordinary
least squares to 2e-13 of `numpy.linalg.lstsq`, in any row order. The
counterpart to `SGDRegressor` here is
[`sgd`](#sgd--stochastic-gradient-descent), the cheap `O(k)` baseline.

Measured on one generated stream of 100,000 rows with `k = 20`, each
contender at its best over a sweep of its own settings
(`scripts/sklearn_comparison.py`, scikit-learn 1.9.0;
[docs/PERFORMANCE.md](docs/PERFORMANCE.md) §19 has the full tables and the
sweeps). The noise ceiling is the R² of the generating signal itself:

| contender | R² stationary | R² drifting | rows/sec | what a prediction saw |
|---|---:|---:|---:|---|
| noise ceiling | 0.9831 | 0.9923 | | |
| `SGDRegressor`, row by row | 0.9829 | 0.9899 | 3,400 | every row before it |
| `SGDRegressor`, batches of 1,000 | 0.9830 | 0.9820 | 2,200,000 | every row before its *batch* |
| `po.spec.sgd` | 0.9826 | 0.9906 | 6,000,000 | every row before it |
| `po.spec.ewridge` | **0.9831** | **0.9907** | 575,000 | every row before it |

**Accuracy is not the difference.** Everything reaches the ceiling on the
stationary stream, and the row-by-row contenders are within 0.001 of each
other on the drifting one. `SGDRegressor` and `po.spec.sgd` at the same
constant step give the same number to four places, because they are the
same recursion. The one gap in the table, the batched 0.9820, is
staleness: a prediction made up to 999 rows before its update.

**The speed difference is a difference in semantics.** sklearn's fast form
updates once per 1,000-row batch, while every prediction here is made from
the state as it stands. Asked for that same guarantee, `partial_fit` per
row, `SGDRegressor` runs at 3,400 rows a second, and the time goes to
Python's per-row overhead rather than to the algorithm.

**A halflife is not the advantage either.** A constant learning rate
forgets too, at about `1/eta` rows, and on evenly spaced rows that is a
halflife. The clock matters when the rows are unevenly spaced, and this
stream's rows are evenly spaced.

**Where the Gram wins: the first hundred rows of every group.** An exact
solve is right as soon as its `X'X` is full rank, about `k` rows in, where
a first-order method needs about `1/eta` rows per direction. On 500 groups
of 200 rows, each group with its own coefficients and `k = 20`, R² by
position in the group, with sklearn as one estimator and one scaler per
group in a dict, row by row, and the bank as `group="g"`:

| contender | rows 25–50 | rows 50–100 | rows 100–200 | rows/sec |
|---|---:|---:|---:|---:|
| noise ceiling | 0.9896 | 0.9903 | 0.9900 | |
| `SGDRegressor` per group, at its best | 0.7277 | 0.9242 | 0.9789 | 3,342 |
| `po.spec.sgd`, `scale_features=True`, the same step | 0.7182 | 0.9213 | 0.9788 | 18,519,660 |
| `po.spec.ewridge` | **0.9693** | **0.9860** | **0.9882** | 5,130,803 |

The `sgd` row uses sklearn's step. The 0.01 of R² between the two `sgd` rows
is where the *prediction* is standardised. sklearn's loop standardises the
row it predicts against the moments before it, and the row it learns from
against the moments including it. Here one standardised row serves both,
so a prediction is on the same footing as every row the coefficients were
learned from, and the gap closes as the fit converges. The other
differences:

| | sklearn | here |
|---|---|---|
| a grid of six penalties | 3.8× the work: six estimators | 1.6× the work: one accumulator, six solves |
| several targets | one estimator each | one shared `X'X` |
| one fit per key | a dict of estimators | `group=`, one state per key |
| standardisation | a `StandardScaler` fitted on the whole frame can leak | streaming, so it cannot leak |
| chunking | | chunk invariance is a test |
| saved model | a pickle | a versioned file that loads on every OS |

**Where sklearn wins: a wide row against `ewridge`, by batching.**
`ewridge` keeps a `(k+1)²` matrix, so at `k = 10,000` it carries 860 MB of
state and runs at 53 rows a second. `SGDRegressor` in batches of 1,000
carries 0.31 MB and runs at 20,007 rows a second. The time goes to the matrix
itself: a rank-1 update moves all 800 MB of it every row, at 85 GB/s, near
this machine's memory bandwidth. `gram_block_rows=1024` touches the matrix
once per 1,024 rows instead, for 7.2× the throughput, 379 rows a second,
and no less memory. `sgd` is the `O(k)` answer here: 0.89 MB and 33,271
rows a second at the same width. That is faster than sklearn's batch, with
every prediction made from the state as it stands. `SGDRegressor` asked for
the same, row by row, runs at 2,206 rows a second at that width, and the
two agree, with a correlation of 0.999997 under `scale_features=True`,
which is sklearn's own recipe of a scaler in front of the step. Both run at
`learning_rate = 0.2 / k`, because an LMS step is stable only while
`eta · |z|² < 2`, and a standardised row has `|z|² ≈ k`. The ecosystem is
sklearn's: pipelines, `GridSearchCV`, calibration, and far more use. What
this library has is the stream.

## Scope and integrations

### What this is not

A model layer, not a stream-processing framework. It expects a frame that
is already aligned, and, when a spec names a `clock`, each group's rows in
clock order. It keeps a fixed amount of memory per stream. It deliberately
does **not** provide:

| not provided | use instead |
|---|---|
| connectors or ingestion | whatever Polars can read |
| event-time windowing, asof or interval joins | Polars expressions upstream, or a streaming framework such as [Pathway](https://pathway.com) |
| watermarks or a late-arrival policy | nothing: `clock`, `max_dclock`, `on_clock_reset` and `session` describe time *within* a stream, not pipeline lateness. Under a `clock`, a row that arrives out of order is a data error, and `on_clock_reset="error"` says so |
| distributed execution | one process, with a thread pool across (spec × group) |

### Pathway

Those boundaries make the two compose.
[examples/pathway_integration.py](examples/pathway_integration.py) runs a
`ModelBank` as a stateful operator inside a Pathway pipeline: Pathway does
the ingestion, the event-time alignment and the windowing, and the bank
does the model. Chunk invariance means the engine's batching cannot change
the numbers, and `save_bytes` and `load_bytes` let a pipeline checkpoint
carry the model state. Pathway is not a dependency; the example imports it
lazily.

### Databases: DuckDB and ADBC

Databases compose the same way, through the Arrow PyCapsule interface.
[examples/duckdb_cursors.py](examples/duckdb_cursors.py) and
[examples/adbc_cursors.py](examples/adbc_cursors.py) sort a query by the
clock and stream it into a bank, with no pyarrow. Each gives every lazy
plan its own cursor, because a connection holds one open result, so a
second plan built on the same connection breaks the first. DuckDB reports
that: the broken plan yields no rows and raises `ConsumedSourceWarning`.
ADBC does not: the broken plan returns wrong rows with no warning at all.
Both examples show the rule and the trap, and fail if either stops holding.
DuckDB and ADBC are dev dependencies of this project, not dependencies of
the package.

## Versions, testing and development

### Versioning and the Polars pin

#### What is pinned

| py-polars | rust polars | pyo3-polars | pyo3 | Python |
|---|---|---|---|---|
| **>= 1.34.0, < 3** (built and tested against 1.44.1) | 0.55.2 | 0.28 | 0.29 | ≥ 3.12 (`abi3-py312`) |

The Rust `polars` is pinned exactly and built into the wheel. The runtime
requirement is a range, because the two copies never meet. The floor is
`LazyFrame.collect_batches`, which `lf.online.fit_predict` and the
file-to-file runner read with, and which py-polars added in 1.34.0. The
whole suite passes on 1.34.0, 1.38.1, 1.44.1 and the 2.0 release candidate
with identical numbers, and `ModelBank` alone works from 1.28.1. A test
asserts the pins, and the matrix is in
[docs/RELEASE-READINESS.md](docs/RELEASE-READINESS.md).

#### Which interfaces carry a promise

This library crosses into Polars three ways, and only one of them carries a
guarantee:

| interface | used by | its promise |
|---|---|---|
| pyo3-polars' extension types | `ModelBank` | none beyond the latest definitions working with the latest Polars: provided "for convenience" |
| the IO plugin | `lf.online.fit_predict` | none: documented, but `@unstable` in py-polars |
| the Arrow PyCapsule interface | [`fit_predict_arrow`](#output-as-arrow) | an Arrow specification, which py-polars, pyarrow and duckdb all consume, so a break there would be Arrow's rather than Polars' |

The first two both stream, so a break on a new Polars is expected
maintenance, not a surprise. A mismatch is an error, not a crash:
`ModelBank` moves data across the boundary through the Arrow C Data
Interface, and a Polars without the two private methods it reads fails with
a clean `AttributeError` before any data moves. The third narrows the
exposure rather than removing it: only the output side uses it today, and
the frame still goes in as a Polars frame.

#### How the pin moves

A weekly job ([`polars-canary.yml`](.github/workflows/polars-canary.yml))
drops the range from `pyproject.toml`, installs the newest py-polars,
builds the wheel as CI does, and runs the whole suite. Only Polars moves in
that run, so a red canary means Polars broke this library and nothing else
did. The response is decided in advance: **cap** the range at the last
release that passed, in a patch release, so no resolver hands anyone the
broken pair; then **fix**, and widen again. Every release runs the same
check at the moment it matters, in two legs (`release.yml`):

| leg | resolves to | blocks the publish |
|---|---|---|
| the newest in-range | the newest stable inside `<3` | **yes** |
| the next major | unpinned, prereleases allowed | no |

The first is a promise. `<3` admits every 1.x and 2.x, so a resolver can
hand someone a Polars newer than the one the wheel was built against the
day after it ships, and a pass on the pinned version is not what the range
says. The second is early warning. The steps for raising the ceiling to a
new major are in [docs/RELEASE-READINESS.md](docs/RELEASE-READINESS.md).

#### This package's own versioning

Semantic versioning. While pre-1.0, the **minor** version carries breaking
changes and any change to the numbers a model returns, so pin the minor
version, `~=0.9.0` for the 0.9 series, if you need stability. Widening the Polars range is a minor release, and
narrowing it is a breaking one. Output field names are part of the API
([Output field names](#output-field-names)). See
[CHANGELOG.md](CHANGELOG.md).

### Testing

The guarantees above are only worth what checks them, so the suite is built
around oracles and invariants rather than expected values typed in by
hand: about 650 Rust tests and 2,200 pytest cases, all green on three
operating systems. [docs/TESTING.md](docs/TESTING.md) is the ledger of what
each part proves.

**Against references.** Each model is held to something it cannot share a
bug with:

| model | held against | agreement |
|---|---|---|
| `ewridge`, `rls` | numpy references | 1e-9 |
| `rls` | `ewridge(ridge_decay=True)` solved on every row | below 1e-9 |
| `kalman` | numpy | about 1e-15 |
| `huber`, `quantile` | numpy | about 1e-13 |
| `ftrl` | numpy, and [river](https://riverml.xyz)'s FTRL row for row | about 1e-16, and 1e-12 |
| `lasso` | the KKT conditions of its objective, rather than a ported solver | the conditions hold |
| EW moments | river | in closed form and in the limit |
| quantile and Huber | river | statistically |

**Invariants, for every model**, checked at the bank, and at the command
line too where they apply:

| invariant | checked as |
|---|---|
| chunk invariance | one chunk, seven, a hundred, one row at a time, and with a save and load in the middle |
| thread invariance | 1 thread against 8 |
| group independence | a group's numbers do not depend on what else is in the bank |
| the paths agree | runner ≡ bank for every input source and format; the Arrow output ≡ the Polars one, field for field and null for null |
| `predict` ≡ `fit_predict` | of the next row, field for field, with every diagnostic on |
| stream semantics | the null policy, warm-up, and the clock |
| `n_eff` | the same recursion in every model (`crates/online-core/tests/model_contract.rs`) |

Hypothesis generates adversarial streams, with mixed nulls, duplicate and
long-gap clocks, values at ±1e8, zero weights and tiny groups. It asserts
the strongest invariant: **changing a row's own target never changes that
row's own prediction.** An IC of about 0 on pure-noise targets says the
same thing from the other side.

**Fixed numbers.** There is one golden stream per model in the Rust core.
The whole pipeline, from extraction through fan-out and diagnostics to
struct assembly, is pinned to fixed output and compared on every operating
system, so a divergence in Polars' vectorized paths on another CPU would
show.

**Hardening.** What the suite does to a bank on purpose:

| attack | what must hold |
|---|---|
| everything at once | a 30k-row stream with every output switched on, compared by digest across chunkings, a mid-stream save and load, and thread counts |
| weight scale | all weights ×1e±6 changes nothing but `n_eff` |
| parameter edges | `halflife` from `1e-3` to `inf` |
| a corrupt state file | any byte flipped fails cleanly, and never panics |
| concurrent misuse | two threads calling `fit_predict` at once get a clean error |
| copying a bank | `pickle` and `copy.deepcopy` resume bit-exactly |
| across the FFI | memory safety where two copies of Polars share one process |
| sustained load | a 10M-row soak, opt-in with `pytest -m soak` |

**Contracts that are files.** The public API, every name, default and
signature and every output field name, is a checked-in snapshot
(`tests/api_surface.txt`), so a change is a reviewable diff. Every python
block in this README runs, and so does every example in the API reference.
Everything under `examples/` runs unmodified: the TOML through the real
command line, the Pathway operator end to end, and the cursor examples
against real DuckDB and SQLite databases. `docs/VALIDATION.md`, where the
defaults were chosen, is regenerated and compared, so the numbers behind
them cannot silently stop being true. A data file, a large file or
generated output that gets tracked fails a test.

**Where it runs.** `./scripts/gate.sh` runs before every commit:
`cargo fmt`, `clippy -D warnings`, `cargo test`, `ruff`, `mypy`, the build,
`pytest` and `sphinx -W`. CI runs the same on Ubuntu, Windows and macOS for
every push and pull request. The release build writes a state file on macOS
and continues the stream from it on Windows and Linux. The weekly canary
runs the suite against the newest py-polars. Tests generate or download
their own data, so there are no data files in the repository; downloads
are cached under `.cache/` and skipped when offline.

### Development

```sh
uv sync                                                # Python env (CPython 3.12)
./scripts/gate.sh                                      # everything CI checks
uv run cargo test --workspace --exclude online-py      # Rust tests
uv run maturin develop --release -m crates/online-py/Cargo.toml
uv run pytest                                          # Python tests
uv run --group docs sphinx-build -W docs/reference docs/_build/html   # API reference
uv run python scripts/validate.py > docs/VALIDATION.md # re-run the [validate] experiments
uv run python scripts/regime_experiments.py all        # the docs/REGIMES.md experiments
uv run python scripts/benchmark.py                     # throughput
```

Prerequisites: [uv](https://docs.astral.sh/uv/) and a stable Rust toolchain
([rustup](https://rustup.rs)). `source scripts/env.sh`, or
`. .\scripts\env.ps1` in PowerShell, puts both on the `PATH` for a shell,
and `.vscode/settings.json` does it for VS Code's terminal. `cargo` runs
through `uv run` because `online-py` builds against pyo3's `abi3-py312` and
needs a 3.12+ interpreter at build time.

| to find | read |
|---|---|
| every document, and which to read for what | [docs/README.md](docs/README.md) |
| a map for coding agents, at the repo root and on the docs site | [llms.txt](llms.txt) ([llmstxt.org](https://llmstxt.org)) |
| the API reference, built from the docstrings and published from every green push to `main` | <https://hgilde.github.io/polars-online/> |
| the design and the task list | [docs/PLAN.md](docs/PLAN.md) |
| running a bank as a job | [docs/RUNNER.md](docs/RUNNER.md) |
| saving, serving and resuming | [docs/STATE-WORKFLOW.md](docs/STATE-WORKFLOW.md) |
| how the defaults were chosen | [docs/VALIDATION.md](docs/VALIDATION.md) |
| what the regime detectors find | [docs/REGIMES.md](docs/REGIMES.md) |
| where the time and memory go | [docs/PERFORMANCE.md](docs/PERFORMANCE.md) |
| adding a model | [docs/EXTENDING.md](docs/EXTENDING.md) |
| how the docs are written | [docs/WRITING.md](docs/WRITING.md) |

### License

Apache-2.0. See [CONTRIBUTING.md](CONTRIBUTING.md) to make changes, and
[SECURITY.md](SECURITY.md) to report a vulnerability.
