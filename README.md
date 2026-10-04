# polars-online

> **GitHub project:** [github.com/hgilde/polars-online](https://github.com/hgilde/polars-online)

Online model fitting for [Polars](https://pola.rs): linear models,
streaming moments, clustering and regime detection, for data too large to
hold in memory at once. When the rows have a time order, a fit can also be
local, following the recent rows without refitting a window at every step.
Features and targets can be windowed means over the stream, looking back or
ahead. Rust core, Python API, and a standalone command line.

| section | what it covers |
|---|---|
| [Introduction](#introduction) | [the idea](#the-idea) · [terminology](#terminology) · [what you can rely on](#what-you-can-rely-on) · [install](#install) · [a first fit](#a-first-fit) |
| [How a bank sees a stream](#how-a-bank-sees-a-stream) | [what a spec names](#what-a-spec-names) · [time and decay](#time-and-decay) · [convergence without a decay](#convergence-without-a-decay) · [a hard window](#a-hard-window) · [a local fit along any feature](#a-local-fit-along-any-feature) · [row order](#row-order-and-the-two-guarantees) · [groups](#groups) · [weights](#weights) · [warm-up](#warm-up) · [labels that arrive late](#labels-that-arrive-late) · [nulls](#nulls-and-three-ways-to-hold-a-row-back) |
| [Preparing a stream](#preparing-a-stream) | [windowed means](#windowed-means-looking-back-or-ahead) · [windows as columns](#windows-as-columns) · [windows as a model's inputs and target](#windows-as-a-models-inputs-and-target) · [saving and resuming a window run](#saving-and-resuming-a-window-run) · [series that tick at their own times](#series-that-tick-at-their-own-times) |
| [Running a bank](#running-a-bank) | [as a query](#as-a-query-lfonlinefit_predict) · [in a loop](#in-a-loop-modelbank) · [outside Python](#outside-a-live-python-process) · [output as Arrow](#output-as-arrow) |
| [Saving, loading and serving](#saving-loading-and-serving) | [save and load](#save-and-load) · [serving without learning](#serving-without-learning) · [a state without this library](#reading-a-state-without-this-library) |
| [Reading the fit](#reading-the-fit) | [what a bank holds](#what-a-bank-holds) · [output field names](#output-field-names) · [coefficients](#coefficients) · [the running sums](#the-running-sums-behind-a-fit) · [one row per finished group](#one-row-per-finished-group) · [correlation matrices](#reading-a-correlation-matrix) |
| [Diagnostics, selection and evaluation](#diagnostics-selection-and-evaluation) | [per-row diagnostics](#per-row-diagnostics) · [conformal intervals](#conformal-intervals) · [choosing among a grid's settings](#choosing-among-a-grids-settings) · [evaluating an output](#evaluating-an-output-frame) · [evaluating a stream too large to hold](#evaluating-a-stream-too-large-to-hold) · [simulated data](#data-whose-truth-is-known) |
| [Models](#models) | [linear models](#linear-models) · [moments and correlation](#moments-and-correlation) · [clustering and classification](#clustering-and-classification) · [sequential tests and regimes](#sequential-tests-and-regimes) |
| [Performance](#performance) | [throughput](#throughput) · [parallelism](#parallelism) · [chunk size](#chunk-size) · [memory](#memory) · [tuning memory](#tuning-memory-with-polars-own-settings) · [window operators](#window-operators) · [against scikit-learn](#against-scikit-learn) |
| [Scope and integrations](#scope-and-integrations) | [what this is not](#what-this-is-not) · [DuckDB and ADBC](#databases-duckdb-and-adbc) · [Pathway](#pathway) |
| [Versions, testing and development](#versions-testing-and-development) | [versioning and the Polars pin](#versioning-and-the-polars-pin) · [testing](#testing) · [development](#development) · [license](#license) |

## Introduction

polars-online fits models in a single pass over data too large to hold in
memory. *The idea* says how, *Terminology* defines the words this README
uses, and *What you can rely on* lists what holds in every run. *Install*
and *A first fit* then get it running.

### The idea

You describe one or more models: say, a ridge regression of a stock's
return on two signals, with a separate regression for every stock.
polars-online fits all of them in a single pass over your rows, as one
*model bank*: a set of models fitted together over the same rows. The
paragraphs below say what that pass guarantees, when row order matters, and
how a bank is run, saved and checked.

**Every prediction is out-of-sample.** Each row is *predicted* first, from
what the models have learned so far, and *learned from* second, so no row's
own outcome is ever in the number predicted for it.

**The stream can be far larger than memory.** The models keep what they
have learned, and hold a row only while a delayed label or a window must
wait for it.

**Row order matters when a model forgets.** With a *decay*, older rows count
less, so the rows must arrive in time order. Without a decay, the models
that solve or accumulate give the same answer in any order ([Convergence
without a decay](#convergence-without-a-decay) says which).

**Features and targets can be windows over the stream.** Examples are the
last minute's time-weighted mean or the next minute's VWAP, written as
Polars expressions and computed in the same pass. A target that looks ahead
is learned only once its window has closed, so no prediction sees the
future it is scored against. It needs time order either way, with a decay
or without ([Preparing a stream](#preparing-a-stream)).

**A model bank runs three ways, with the same numbers from each.** It runs
inside a Polars query, in your own Python loop, or from a standalone command
line with no live Python at all ([Running a bank](#running-a-bank)).

**What a bank has learned can be saved.** It is saved to a file and loaded
back, to keep learning or to score new rows without learning ([Saving,
loading and serving](#saving-loading-and-serving)).

**The diagnostics are out-of-sample too.** Residual spread, break detection,
a choice among several settings and running accuracy are computed from what
the models have already learned. So none of them lets a row's own outcome
into what it is measured against ([Diagnostics, selection and
evaluation](#diagnostics-selection-and-evaluation)).

### Terminology

This README uses these terms of its own:

| term | meaning |
|---|---|
| **spec** | the description of one model: which model, which columns it reads, and how it treats time. `po.spec.ewridge(...)` builds one |
| **model bank**, or *the bank* | a set of specs fitted together over the same rows, and the Python object that holds them, [`ModelBank`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.ModelBank). *The bank* always means a model bank |
| **stream** | the rows in the order the bank reads them |
| **chunk** | one of the pieces the stream arrives in. The bank takes one chunk at a time |
| **state** | everything a bank has learned |
| **clock**, **decay** | the column that says how far apart two rows are, and the forgetting measured along it: each row's weight halves every `half_life` of the clock ([Time and decay](#time-and-decay)) |
| **`weight_sum`** | the *weight behind the state* that produced a row's prediction, which `min_weight` reads ([Warm-up](#warm-up)) |

### What you can rely on

These hold for every spec, however a bank is run. Most rows link to the
section that shows how.

| | |
|---|---|
| **honest predictions** | every row is predicted before its own outcome is learned ([Row order and the two guarantees](#row-order-and-the-two-guarantees)), and a target that looks ahead is learned only once its window has closed ([Windows as a model's inputs and target](#windows-as-a-models-inputs-and-target)) |
| **bounded memory** | memory is proportional to the models' state and the rows a delay or a window holds, not to the number of rows that have passed ([Memory](#memory)) |
| **any chunking** | one chunk or a thousand, with or without a save and resume in the middle, gives the same numbers, to the last bit. Only which rows carry the coefficients, `coef`, can change ([Row order and the two guarantees](#row-order-and-the-two-guarantees)) |
| **any thread count** | the thread count changes only the speed. Each spec and group is one task on the bank's thread pool: with 64 groups, 14 threads process 7.3× the rows per second of one ([Parallelism](#parallelism)) |
| **named mistakes** | every keyword is checked against its type, and a missing column is reported with the spec that wanted it and the role it had there |
| **tested** | 1,215 Rust tests and 3,840 Python cases (counted on 2026-10-03), held to independent libraries such as scikit-learn, statsmodels and river and to adversarial streams, run on macOS, Windows and Linux at every push ([Testing](#testing)) |

### Install

One command installs the wheel. The table lists what it needs and what it
carries, and the last block builds it from a checkout instead.

```sh
pip install polars-online      # or: uv add polars-online
```

| need | detail |
|---|---|
| Python | 3.12 or newer: one wheel per platform covers every CPython from 3.12 on ([Testing](#testing) lists the versions CI runs) |
| Polars | `polars>=1.34.0,<3`: the range the test suite has measured, which is not a guarantee. A weekly job and every release run the test suite on the newest Polars ([Versioning and the Polars pin](#versioning-and-the-polars-pin)) |
| depends on | nothing beyond `polars` at run time, since the wheel carries its own copy of Polars' Rust code. `numpy` is optional: only `ModelBank.gram()` and the `po.gram`, `po.corr` and `po.sim` helpers need it |
| wheels | macOS (arm64, x86_64), Windows x64, and Linux (x64 glibc and musl, aarch64 glibc), on PyPI and on each GitHub release beside the command-line binaries |
| size | 8 to 10 MB to download and 26 to 37 MB installed, by platform, for 0.12.0 |

From a checkout:

```sh
uv sync
uv run maturin develop --release -m crates/online-py/Cargo.toml
```

### A first fit

Two fits of one spec follow, and the decay is the whole difference between
them. Without a decay, every row counts forever, and the saved state is the
model. With a decay, the fit follows the recent rows, and the path of its
coefficients is the output.

Each spec adds one column to the output, named after the spec, whose value
in each row is a record of named fields. They are the prediction
`pred_<target>`, the residual `resid_<target>`, the weight behind the state
`weight_sum`, the coefficients `coef`, why a prediction is null
(`withheld_reason`, under [Warm-up](#warm-up)), and the diagnostics you
switch on. `unnest` spreads them into columns.

**One fit over every row.** Turn forgetting off, and a model that solves,
such as this ridge regression, converges to the batch fit over every row it
has seen, in any row order and in bounded memory. The state is then a
complete summary of those rows, which makes it worth saving and serving.
The example fits the spec over a folder of parquet files too large for memory,
whose rows carry a `stock_id`, a timestamp `ts`, a return `ret` and two
signals, `signal_a` and `signal_b`. It saves the state at the last row, then
scores new rows against that state without learning from them.

```python
import polars as pl
import polars_online as po

whole = po.spec.ewridge(
    "ridge",                                          # the spec's name; its output column is named after it
    targets=["ret"], features=["signal_a", "signal_b"],
    half_life=float("inf"),                           # no forgetting: every row counts the same
    group="stock_id",                                 # one separate regression per stock
)

(
    pl.scan_parquet("ticks/*.parquet")                # a Polars query over the files; nothing is read yet
    .online.fit_predict([whole], save_state="bank.state")    # the bank, inside the query
    .filter(pl.col("ridge").struct.field("weight_sum") > 20)    # ordinary Polars on what comes out
    .sink_parquet("fitted.parquet")                   # runs the query, writing the result a chunk at a time
)

scored = pl.scan_parquet("today.parquet").online.predict("bank.state").collect()   # score; learn nothing
flat = scored.online.unnest([whole])   # pred_ret, resid_ret, weight_sum, ..., coef_ret_intercept, coef_ret_signal_a, ...
```

**One fit that follows the recent rows.** Give the same spec a clock, the
timestamp column `ts`, and a finite `half_life`, and each row's weight
halves every ten minutes of `ts`. The fit is now *local*: it describes the
recent past, and it moves from row to row. Its state is weighted toward the
last few tens of minutes, so serving from the final state predicts with the
most recent fit alone. The thing to read is the path the coefficients took,
which `coef_every=1` writes on every row, rather than the state they ended
on. That path has one row per input row: each stock's exposure to each
signal, as it stood before that row. It is a time series, to plot,
difference, or compare between two stocks over a day.

```python
local = po.spec.ewridge(
    "local", targets=["ret"], features=["signal_a", "signal_b"],
    clock="ts",                                       # a Datetime column, so the clock's parameters are durations
    half_life=pl.duration(minutes=10),                # a row's weight halves every ten minutes of ts
    gap_cap=pl.duration(minutes=5),                   # and a gap longer than five minutes decays as though it were five
    group="stock_id",
    coef_every=1,                                     # write the coefficients on every row, not once a chunk
)

betas = (
    pl.scan_parquet("ticks/*.parquet")
    .online.fit_predict([local])
    .online.unnest([local])                           # coef_ret_intercept, coef_ret_signal_a, coef_ret_signal_b
    .select("ts", "stock_id", "^coef_.*$")            # leaves out support_coef_ret_*, each one's data share
    .collect()
)
path = betas.filter(pl.col("stock_id") == "b0").select("ts", "coef_ret_signal_a")
# null until the fit exists, then one value per row: how b0's return loaded on signal_a, over time
```

## How a bank sees a stream

A model bank reads a stream of rows one chunk at a time, and the parameters
in this section say how each model reads it. Most models take every one of
them, and each exception is named where its parameter is introduced. They
answer these questions:

| the question | the parameters | the subsection |
|---|---|---|
| which columns a model reads, and what a null in them does | `targets`, `features`, `fit_intercept` | [What a spec names](#what-a-spec-names) |
| how far apart two rows are, and how fast the past fades | `clock`, `half_life`, `gap_cap`, `session`, `session_gap`, `restart_after_step_back` | [Time and decay](#time-and-decay) |
| what a fit becomes when nothing fades | `half_life=inf`, `lam=1.0` | [Convergence without a decay](#convergence-without-a-decay) |
| when an old row stops counting at all | `window_size` | [A hard window](#a-hard-window) |
| how to fit along a feature instead of time | `clock` | [A local fit along any feature](#a-local-fit-along-any-feature) |
| which row order a bank needs, and how it checks it | | [Row order and the two guarantees](#row-order-and-the-two-guarantees) |
| which rows belong to which model | `group`, `group_close` | [Groups](#groups) |
| how much each row counts | `weight` | [Weights](#weights) |
| when a model has seen enough to report | `min_weight`, `min_settled_frac`, `max_error_inflation` | [Warm-up](#warm-up) |
| when a label may be learned | `embargo` | [Labels that arrive late](#labels-that-arrive-late) |
| how to keep a row from teaching a model | a weight of 0, a null target, `predict` | [Nulls, and three ways to hold a row back](#nulls-and-three-ways-to-hold-a-row-back) |

With a `group`, every parameter applies within each group. The
[`polars_online.spec`](https://hgilde.github.io/polars-online/spec.html)
reference gives the shared parameters' units and defaults, and each
builder's own page gives the rest.

The examples from here on read `df`, a frame of 400 rows, and `lf`, the
same rows as a query (`df.lazy()`). `df` holds a numeric clock `t`, a
timestamp `ts` a minute apart, the numeric columns `x0`, `x1`, `x2`,
`signal_a`, `signal_b`, `y` and `ret`, and the text columns `stock_id`
(four stocks), `group`, `session` and `venue`.

### What a spec names

A spec names its model, the columns it reads and how it treats time; its
first argument is the spec's own name, which its output column takes. What
those columns may hold, and the forms a target can take, follow the
example. Two more parameters change when and where a model reports:
`coef_every`, under [Coefficients](#coefficients), and `embargo`, under
[Labels that arrive late](#labels-that-arrive-late).

```python
ridge_spec = po.spec.ewridge(
    "ridge",
    targets=["y"],                 # at least one; the targets of one spec share the features' running sums
    features=["x0", "x1", "x2"],   # numeric columns of any width, Decimal and Boolean included; read as 64-bit floats
    fit_intercept=True,            # the default: the fit has a level of its own
    clock="t", half_life=600.0, gap_cap=300.0,   # how it treats time: Time and decay, below
)
```

**A text column in either list is refused** rather than silently read as
nulls. Columns the spec does not name pass through to the output untouched.

**A null skips an update, never the clock.** NaN, ±inf and any magnitude
above `1e100` count as null, so sentinel values never reach a model. A null
in any feature, or in the weight, skips the row: its outputs are null, no
update happens, and the clock still advances. A null in one target still
writes that target's `pred` and leaves its `resid` null. It skips only that
target's update; in `rls`, whose targets share one factor, the row updates
none of them.

#### Relative and look-ahead targets

A target can also be taken against another column of its own row. Or it
can be a window expression that looks ahead, such as the next minute's VWAP
less the mid ([Windows as a model's inputs and
target](#windows-as-a-models-inputs-and-target)). Four kinds of model refuse
both forms: a model with no target, `ew_class`, `seqtest`, and an `ftrl` or
`sgd` fitting a probability or a count.

**`po.target` takes a target against another column of its row**: a price
against the mid, a VWAP against the last trade. The model then learns and
predicts on that scale.

```python
po.spec.ewridge(
    "fwd",
    targets=["ret_5m", po.target("price_5m", relative_to="mid")],   # learns and predicts price_5m - mid
    features=["x0", "x1"], half_life=600.0,
)
```

In the CLI's TOML the same target is a table:
`targets = ["ret_5m", { column = "price_5m", relative_to = "mid" }]`. With
`y` the target column and `r` the `relative_to` column, `relative` sets the
scale:

| `relative` | the target | null where |
|---|---|---|
| `"difference"`, the default | `y − r` | `y` or `r` is null, or above `1e100` in magnitude |
| `"ratio"` | `y / r` | the same, or `y` or `r` is not positive |
| `"log_ratio"` | `ln(y / r)` | the same as `"ratio"` |

`r` is read at the target's own row, so the target adds no look-ahead. The
output fields carry the target's `name`, which is its column's by default.
`pred`, `resid`, `sigma` and the metrics are on the relative scale, so a
prediction of the level is `pred + r`, or `pred * r` for a ratio.

### Time and decay

A model forgets, and that forgetting is its *decay*: each row's weight
halves every `half_life` along a *clock*, a column that says how far apart
two rows are. *Clock types and units* below says how the clock's type sets
the units of every clock parameter, and *Sessions, gaps and steps back*
says what three more parameters do to a market stream.

At each row, the weight of everything learned so far is multiplied by
`λ = 0.5 ** (Δclock / half_life)`, where `Δclock` is the clock's step from
the previous row. A list of half-lives fits one model per value, and
`half_life=inf` turns forgetting off ([Convergence without a
decay](#convergence-without-a-decay)).

#### Clock types and units

**The clock's type decides how every clock parameter is written.** A
timestamp carries its own unit, so its parameters are durations. A number
carries none, so its parameters are numbers of whatever it counts.

| the clock | a clock parameter is | for example |
|---|---|---|
| a `Datetime`, `Date` or `Duration` column: a *temporal* clock | a duration | `half_life=pl.duration(minutes=10)` |
| a numeric column, such as seconds or cumulative traded volume | a number of the column's own units | `half_life=600.0` |
| none | a number of rows | `half_life=100`: a row 100 rows back counts half as much as the latest |

A duration is written three ways: `pl.duration(minutes=10)`,
`timedelta(minutes=10)`, or polars' duration text `"10m"`.

**Every clock parameter acts on the clock, so a model's numbers do not
change with how densely the rows arrive.** The clock parameters are
`half_life`, `gap_cap`, `restart_after_step_back`, `session_gap` and
`embargo` below, and a model's `window_size`, `solve_every` and its own
half-lives. The
[`polars_online.spec`](https://hgilde.github.io/polars-online/spec.html)
reference lists every one. A few things still count rows, and each model's
page names them.

**A clock parameter of the other kind is refused**, except `0` and `inf`,
which mean the same in every unit: where a parameter takes them, they may
stay numbers. The bank refuses before it reads a row, naming the column,
the parameter and the fix:

| refused | on | because |
|---|---|---|
| a plain number | a `Datetime` clock | it would silently take the column's storage unit: `half_life=600` on a microsecond column would mean 600 microseconds |
| a duration | a numeric clock | it has nothing to measure it against |
| a spec that mixes the two kinds | any clock | |
| a duration finer than the clock can act on, such as `gap_cap="12h"` | a `Date` clock, which moves in days | |
| `lam`, a decay per clock unit | a temporal clock | it has no duration form, so a temporal clock takes `half_life` instead |

**A temporal clock is read in its own integer nanoseconds.** So it must lie
between the years 1677 and 2262, the range nanoseconds in a 64-bit integer
cover. The same instants stored in milliseconds, microseconds or
nanoseconds give the same numbers, and a time zone changes nothing. A model
reads only the gap between consecutive rows. The bank takes that gap in
integer nanoseconds before it becomes seconds, so a nanosecond timestamp
keeps its nanoseconds whatever the stream's age.

Where such a clock's quantity reaches an output, it is in seconds, except
under `emit_clocks`, which writes clocks in the clock column's own type
([Labels that arrive late](#labels-that-arrive-late)). `holt`'s trend is
per second, and `summary()` gives the clock's range as seconds since 1970.

#### Sessions, gaps and steps back

A stream from a market brings three more things, and the spec has a
parameter for each. Together, `gap_cap`, `session` with `session_gap`, and
`restart_after_step_back` are the clock's *policy*: what a gap, a session
and a step back each mean.

| the stream has | the parameter | what it does |
|---|---|---|
| a **session** boundary, such as the start of a trading day, where the clock's jump is not elapsed time | `session`, a column whose value changes at a session boundary, and `session_gap`, required with it | applies a step you choose, `session_gap`, in place of the jump, capped at `gap_cap` like any step. `session_gap="reset"` starts the model over instead |
| a **gap** in the clock, such as an hour with no rows | `gap_cap`, required with a clock, finite and above 0 | caps each step between two rows a model learns from, so a quiet hour ages the fit no more than `gap_cap` would. After a run of skipped rows, the next row's step is capped too, however long the run |
| a clock that **steps back**, such as a replayed day | `restart_after_step_back` | refuses it, or starts the model over. A step back larger than the setting starts the model over, and one no larger is a late row, refused. Unset, the default, every step back is refused ([what each step back means](#how-a-bank-detects-rows-out-of-order)) |

**A stream has a *break* wherever `gap_cap` shortened a step, and wherever
the session changes.** A model that keeps past rows by position, as a lag
does, empties them at a break, since the row before a break is no longer
the row just before.

```python
timed = po.spec.ewridge(
    "timed", targets=["y"], features=["x0", "x1"],
    clock="ts",                           # a Datetime column, so every clock parameter is a duration
    half_life=pl.duration(minutes=10),    # a row's weight halves every ten minutes of ts
    gap_cap=pl.duration(minutes=5),       # the most one step between learned rows can count
    restart_after_step_back=pl.duration(minutes=1),  # a larger step back starts the model over
    session="session",                    # a column whose value changes at a session boundary ...
    session_gap=pl.duration(minutes=1),   # ... and the clock step to apply there
)

counted = po.spec.ewridge(
    "counted", targets=["y"], features=["x0", "x1"],
    clock="t",                 # a numeric column, so every clock parameter is a number of its units
    half_life=600.0,           # a row's weight halves every 600 units of t (or lam=, the weight kept per unit)
    gap_cap=300.0,
    session="session", session_gap=60.0,
)
```

### Convergence without a decay

With decay off, `half_life=inf` or `lam=1.0`, a model that *solves* or
*accumulates* converges to the batch fit it defines over every row it has
seen. A model that *reweights*, *steps*, *filters* or *tests* depends on the
order whether or not decay is on. [The model table](#models) says which
model learns which way, and each model's own section says what it converges
to.

**Row order does not change the fit of a model that solves or
accumulates.** One that solves keeps running sums and computes its
coefficients from them; one that accumulates keeps running sums and reports
them. With no clock column, rows fed forwards, backwards or shuffled give
the same coefficients, to rounding. With a clock column, a row out of clock
order is refused, as [How a bank detects rows out of
order](#how-a-bank-detects-rows-out-of-order) says.

**A huge finite half-life is not `inf`.** `half_life=1e12` still forgets, a
little. Its `settled_frac` stays near zero until the stream has run for
about 10¹² clock units, so a `min_settled_frac` gate ([Warm-up](#warm-up))
stays closed. A model that solves on a schedule solves by weight under it,
where `inf` solves on every row. Say `inf` for no forgetting.

### A hard window

A half-life never forgets entirely: three half-lives back still carries
12.5% of the weight. `window_size=w` makes a row older than `w` clock units
contribute exactly nothing. Five models take it, `ewridge`, `lasso`,
`ew_cov`, `ew_class` and `marginal`; below come what it does to their
numbers and what its snapshots cost in memory.

```
weight(age) = 0.5 ** (age / half_life)   if age <= window_size
            = 0                          otherwise
```

**Inside the window the weights are still exponential.** So this is not a
flat rolling mean, and the newest row still dominates.

**The cut is exact.** An exponentially weighted sum contains its own past:
everything at or before a time `u` is `λ^(t−u)` times the running sum as it
stood then, `λ` being the weight kept per clock unit. Subtracting that
leaves precisely the rest. The model keeps a ring of past snapshots to do
it, and `window_every` sets their spacing. The boundary is the oldest
snapshot inside the window, so a coarse cadence discards a little more than
asked, never less.

```python
cut = po.spec.ew_cov(
    "cut", features=["x0", "x1"], clock="t", gap_cap=300.0, half_life=500.0,
    window_size=1500.0,            # a row older than this many clock units contributes exactly nothing
    window_every=10,               # a snapshot every 10 rows
    window_budget={"refuse": 64},  # at most 64 MiB of snapshots per ring, as the table below says
)
```

Five things to know before reading the numbers:

| | |
|---|---|
| the clock is the **decayed** one | a row's age adds up each step after `gap_cap` and any `session_gap` |
| the edge is a **discontinuity** | a row ageing out drops its whole weight at once, so the series has small steps that a plain exponential average does not |
| it is a **subtraction** | precision falls with the fraction discarded: negligible at a window of three half-lives, worse as the window shortens toward the half-life |
| `weight_sum` is the weight inside the window | so `min_weight` gates on a weight that settles once the window is full |
| a long step **empties** the window | a step longer than `window_size`, after `gap_cap`, empties the window, and the model reports nulls from the row after it, never stale numbers; a cap shorter than the window never empties it |

What the window truncates depends on the model:

| model | inside the window | refused beside `window_size` |
|---|---|---|
| `ewridge` | the sums the fit is solved from, and `sigma` and `zscore` with them; the coefficients are the window's as of the last solve | `ridge_scale="sum"`, `session_shrink`, `gram_block_rows` |
| `lasso` | the same, and the error that selects the penalty; a feature with no spread inside the window has no evidence there, and goes to exactly zero | |
| `ew_cov` | every moment, and `mahal`, `partial_corr` and the principal components read from them | `lags`, `mahal_quantiles` |
| `ew_class` | each class's moments, so the classifier follows class means that move | |
| `marginal` | every pair's weight, means and second moments, so `corr`, `beta` and `t` describe the window; its lag moments, under `window_lags=True`, are an estimate rather than exact | `bins`, `feature_moments="shared"`, and `lags` unless `window_lags=True` |

**`window_budget` caps the snapshots' memory, per ring, in MiB.** The ring
makes the model's memory grow with the window as well as with the state:
for an `ew_cov` over 20 columns, a 1,000-row window holds about 3 MB per
group, divided by `window_every`. A refusal comes before any of the chunk
is learned, except for two overruns found only as the rows go in. One is
under `drift_action="reset"`; the other is a snapshot that grows within the
chunk, as an `ewridge`'s or a `lasso`'s does when its targets first go
missing on different rows. The bank then refuses every later call, so
rebuild it from its last save.

| `window_budget` | when a chunk would take a ring past the cap |
|---|---|
| not given | the chunk is refused past 256 MiB |
| `{"refuse": 64}` | the chunk is refused before any of it is learned, and the bank goes on as it was. The error names the ring's size, `window_every` and the ways to raise the cap |
| `{"thin": 64}` | the ring drops every other snapshot and doubles its spacing, which, like `window_every`, only ever shortens the window |
| `{"refuse": float("inf")}` | nothing: no cap |

### A local fit along any feature

The clock need not be a time, and any model can take a feature as its
clock. Sort the frame by one of its own feature columns and name that
column as the clock, and `half_life` becomes a bandwidth in that feature's
units.

Each row is then fitted on the rows before it, each weighted by
`0.5 ** (Δx / half_life)`, with `Δx` its distance from the row in that
feature: an exponential kernel. The result is a local regression, computed
in one pass from running sums that do not grow, while the usual method
refits a window of rows at every point. These are the numbers a batch fit
gives: a kernel-weighted least squares recomputed from scratch at every row
agrees with them to 1e-12.

```python
curve = df.sort("x0")
kernel = po.spec.ewridge(
    "kernel",
    targets=["y"],
    features=["x0"],
    clock="x0",                 # the clock is a feature: the decay is a kernel in it
    half_life=0.5,              # the bandwidth, in x0's own units
    gap_cap=1.0,                # a wider gap decays as if it were this wide
    max_rows_between_solves=1,  # refit at every row
    coef_every=1,               # write the coefficients on every row
    min_weight=10.0,
)
fitted = po.ModelBank([kernel]).fit_predict(curve).unnest("kernel")
# pred_y  is the fitted curve, read at each row's own x0
# coef    is the line through that row's neighbourhood
```

**The kernel is one-sided, so the fit lags.** A row is fitted on the rows
before it and never after, so the fit follows a curve with a lag. The
bandwidth trades that lag against noise. On `sin(x)` at a bandwidth of
0.25, the fit sits 0.08 from the truth, where the best straight line sits
0.39. At a bandwidth of 1.0 it sits 0.29, most of the way back to the line.

**`features` need not include the clock column.** With other features, the
same fit is a regression whose coefficients move along the clock.

### Row order and the two guarantees

Two guarantees hold for every run, and both rest on something the caller
supplies: a fixed row order. Below come the guarantees, the query steps
that can break that order, and [How a bank detects rows out of
order](#how-a-bank-detects-rows-out-of-order), the checks a bank runs on
what it is given.

- **Predictions are out-of-sample.** Every row is predicted from the state
  as it stood before the row's own target was learned, so nothing can leak
  a row's outcome into its own prediction.
- **Chunk invariance.** One chunk or a thousand, with or without a save and
  resume in the middle, gives bit-identical numbers, with one exception:
  which rows carry `coef`, and `support_coef` beside it. That is a reporting
  cadence, written every `coef_every` rows and on each group's last row in
  every chunk, so smaller chunks report it more often.

**A model learns in row order.** So a query whose row order Polars does not
guarantee is a different model each time it runs. A query handed to a bank
runs a chunk at a time. Run that way, a step without an order guarantee can
deliver rows in another order than `lf.collect()` gives. Measured,
`collect()` kept the input order where reading the same query a chunk at a
time did not. Give each such step its guarantee:

| a query step that may reorder rows | give it |
|---|---|
| `join` | `maintain_order="left"` |
| `group_by` | `maintain_order=True` |
| `unique` | a sort after it |
| anything else | a sort before the bank |

**A query with such a step raises `OrderNotGuaranteedWarning`** when it is
handed to a bank, naming the step. For a query whose order you know,
silence the warning with
`warnings.simplefilter("ignore", po.OrderNotGuaranteedWarning)`.

**The query check is best-effort.** It reads the text of the query's
`explain()`, and walks the query's steps only when that names a join, an
aggregation, a `unique` or a `sort`. A step followed by a `sort` counts as
ordered, unless that sort is by several keys without `maintain_order=True`.
Such a sort leaves rows with equal keys in no particular order: 2,109 of
10,000 rows moved when measured. A join with `maintain_order="left"` is
followed on its left side only. A query Polars cannot serialize passes
without a warning, and the check never fails a run.

**`ModelBank.fit` does not warn when its state cannot depend on the row
order.** That is when every spec is an `ewridge` or `rls` with decay off,
and with none of the settings that read the order. Those include a window,
a session, a weight, an embargo, a drift reset, `coef_every` and any
emitted diagnostic. Their sums reach the same state in any order, to
rounding, and `fit` keeps only the state. A spec with a window expression
as its target always warns, since the target reads the rows ahead.

#### How a bank detects rows out of order

A bank checks the order it is given in three places. The query check warns,
as above. The other two refuse a chunk before any of its rows is learned.

| check | what it reads | when it finds disorder |
|---|---|---|
| the query | the query handed to the bank: a `join`, `group_by` or `unique` whose order Polars does not guarantee, unless a `sort` follows it, and a sort by several keys without `maintain_order=True` | raises `OrderNotGuaranteedWarning`, naming the step and its fix; the run goes on |
| the clock | each group's clock, row by row, for every spec with a `clock` | refuses the chunk on a backwards step the settings below do not allow |
| the group keys | with `group_close="monotone"`, the keys in the column's own order, an integer column as numbers and any other as text | refuses the chunk on a key below the one before it, since a closed group cannot reopen |

**The clock is checked group by group.** Each spec keeps one clock per
group, so groups may interleave freely. Only the rows of one group need to
be in clock order. Equal clock values are a gap of zero. A row with a null
feature is skipped, but its clock is still checked. On a temporal clock the
comparison is exact, in integer nanoseconds. A row that `predict` scores is
never refused ([Serving without learning](#serving-without-learning)).

**`restart_after_step_back` and the size of a step back decide what it
means.** It has no default, since only the caller knows how late a row can
be. A step back equal to it counts as late, and `0` restarts the model at
every step back:

| a step back | is read as | and the bank |
|---|---|---|
| any size, with `restart_after_step_back` unset, the default | rows out of order | refuses the chunk |
| no larger than `restart_after_step_back` | a late row, such as a transposed pair or a row a minute late | refuses the chunk |
| larger than `restart_after_step_back` | a new start, such as a replayed day or a restarted feed | restarts the model |
| any size, on a row whose `session` value changes | a new session, where the step is no measure of time | applies `session_gap`, or restarts the model under `session_gap="reset"` |

**The summary counts what the clock rules met.** `bank.summary()` reports
`clock_backwards`, the rows whose clock fell below the previous row's
within a session, and `resets`, the rows where a stream restarted. With
`restart_after_step_back` given, each step back larger than it counts once
in each.

**A refused chunk leaves the bank as it was.** The bank runs the clock of
every group of every spec over the chunk before it learns any row. So one
late row in one group changes nothing, and the corrected chunk can be fed
again. Input that overlaps a saved state is a step back in every group;
[Save and load](#save-and-load) says how `skip_learned` resumes on it.

**The error locates the step.** It names the spec, the clock column, the
size of the step and the row. The row is
counted from the start of the input however it arrives: one frame,
batches, a query or the CLI's file. On a temporal clock the step is a
duration. The error also names the way out:

```text
spec "m": clock column "t" goes backwards by 30 at row 6 (restart_after_step_back is unset, so
every step back is refused); the bank was not updated. Sort each group by the clock; to resume a
saved state on input that overlaps it, drop the rows it has learned, with
ModelBank.skip_learned(frame) in Python or by filtering the command line's input to the rows after
them; or, if a step back this large starts the stream over, set restart_after_step_back to the
smallest one that does.
```

### Groups

`group` names a column, and the bank keeps one separate model per distinct
value of it, all fitted in the same pass over the stream. Every other
parameter, the clock included, applies within the group. `group_close`
says when a group is finished, which keeps a bank's memory bounded when new
group values never stop appearing.

```python
per_stock = po.spec.ewridge(
    "per_stock", targets=["y"], features=["x0", "x1"], clock="t", half_life=600.0, gap_cap=300.0,
    group="stock_id",           # one model per distinct value of this column
    group_close="monotone",     # or "session": when a group is finished, write it out and free its memory
)
```

**Without `group_close`, a group lives as long as the bank.** A
long-running bank can drop the ones that have gone quiet ([In a
loop](#in-a-loop-modelbank)).

**With it, a finished group is written out and freed.** The bank writes the
group's running sums out and frees its memory. It writes a row per
half-life and per *Gram*, the matrices a fit is solved against ([The
running sums behind a fit](#the-running-sums-behind-a-fit)), and
`bank.closed_groups()` reads them ([One row per finished
group](#one-row-per-finished-group)). `group_close` is refused beside an
`embargo` or a window target, which hold rows past a group's end. Its two
values decide when a group is finished:

| `group_close` | a group is finished when |
|---|---|
| `"monotone"` | a higher key arrives: the key never goes backwards, so a key below the largest seen is finished |
| `"session"` | its `session` value changes |

### Weights

`weight` names a column of row weights. In a model that keeps means a
weight is relative, and seven models read it otherwise, as the table below
shows.

```python
weighted = po.spec.ewridge(
    "weighted", targets=["y"], features=["x0", "x1"], clock="t", half_life=600.0, gap_cap=300.0,
    weight="w",                # a column of row weights: here a row of weight 2 counts as two rows of weight 1
)
```

**A weight is relative.** In a model that keeps means, scaling every weight
by the same factor changes nothing but `weight_sum`, so a `min_weight`
scales with it.

**A weight of 0 is legal, and a negative weight is refused.** A row of
weight 0 is scored, the clock advances, and nothing is learned. [Nulls, and
three ways to hold a row back](#nulls-and-three-ways-to-hold-a-row-back)
compares weight `0` with the other two ways to keep a row from teaching a
model.

**Seven models read a weight differently**, each as its own page says, and
`tests/test_weight_scale.py` names each:

| model | a row's weight is |
|---|---|
| `rls` | on the sum scale, against a ridge of fixed size: a heavier stream outweighs it sooner |
| `kalman` | a scale on the observation's precision: the update divides the observation noise `R_j` by `w`, so with a fixed `obs_var=` the observation's variance is `obs_var / w` |
| `sgd` | a step size: the gradient carries it |
| `ftrl` | an importance weight, as Vowpal Wabbit's, against penalties of fixed size |
| `pa` | a step size below 1; a weight above 1 counts as 1 |
| `hmm` | on the sum scale, against the transitions' prior count |
| `seqtest` | 0 or 1, since a trial is counted or not |

### Warm-up

A model should not report a number it is not yet informed enough to give.
Every model with a decay shares two gates that say what "informed enough"
means, and `ewridge` adds a third
([docs/WARMUP-AND-CONVERGENCE.md](docs/WARMUP-AND-CONVERGENCE.md)). The
gates hold back output only: the model learns from every row either way.

| setting | default | withholds a prediction while | read from |
|---|---|---|---|
| `min_weight` | by model, below | `weight_sum` is below it | the weight behind the state |
| `min_settled_frac` | `0`, off | the decay window is less full than this fraction of its steady state | `settled_frac = 1 − 2^(−T / half_life)`, with `T` the decay time the model has seen: `0.5` at one half-life, `0.75` at two, whatever the row rate |
| `max_error_inflation`, `ewridge` only | `sqrt(2)` | estimation error would inflate the prediction's error over the noise floor by more than this factor | `error_inflation = sqrt(1 + edf / n_kish)`: the effective degrees of freedom the fit used, over Kish's effective sample size behind it |

**`weight_sum` is the *weight behind the state* that produced a row's
prediction.** It is the weight as the previous row left it, before this
row's decay and its update, and it is `0` on a stream's first row. It runs
one behind the row count while nothing is forgotten, and settles at
`1 / (1 − λ^d)` for rows of weight 1, `d` clock units apart,
`λ = 2^(−1/half_life)`. It is a weight and not a sample size: at a
half-life of 600 with rows 0.1 apart it settles near 8,657. Kish's
`n_kish` in `gram()` is the sample size. It means the same thing in every
model, so one `min_weight` means the same thing across a bank.

**`min_weight` is an absolute floor.** Every output is null until
`weight_sum` reaches it. A list gives one threshold per target. A
regression model checks each target against that target's own weight, the
weight of the rows it was present on, so an often-null target reports later
than the others. For `rls`, which learns a row only when every target is
present, that is the weight of the rows it learned from. The `weight_sum`
field is the shared weight either way. The default of `min_weight` depends
on the model:

| default | models |
|---|---|
| one per unknown: the features, and the intercept when there is one | `lasso`, `kalman`, `huber`, `quantile`, `rls`, `sgd`, `pa`, `ftrl` |
| the feature count plus one (1 for `holt`) | `ew_cov`, `ew_class`, `kmeans`, `micro`, `holt` |
| 3 | `marginal`, `deco` |
| 1 | `bocpd` |
| 0, each having a gate of its own | `ewridge`, `seqtest`, `rcov`, `hmm`, `corrchange` |

**`min_settled_frac` is off by default**, because a fit kept as weighted
means is unbiased from its first row when the process is stationary. Set it
when the half-life was chosen to average over regimes or seasons that a
shorter history would not represent.

**`max_error_inflation` tracks the model**: add a feature and the gate
moves. It reads Kish's count, so one row carrying a hundred times the
weight of the others counts as barely one.

**Each row says how ready its model was.** `summary()` carries the same
readings per group, and a `ReadinessWarning` names, once, a coefficient
more ridge than data or a noise gate the settled stream can no longer meet.
The fields `ewridge` writes:

| field | what it holds |
|---|---|
| `settled_frac` | how full the decay window is |
| `withheld_reason` | why `pred_y` is null, or null when it is not: `below_min_settled_frac`, `below_min_weight` or `above_max_error_inflation`, in that order of precedence |
| `error_inflation_y` | the row's leverage against the fit: high on a row leaning on a direction the data never showed, at one triangular solve a row |
| `support_coef` | beside `coef`: each coefficient's data share, `1 - ridge * (S^-1)_jj`; a duplicated pair of features reads 0.5 each |

```python
warm = po.spec.ewridge(
    "warm", targets=["y"], features=["x0", "x1"], clock="t", half_life=100.0, gap_cap=300.0,
    max_error_inflation=1.1,    # withhold while estimation error would add more than 10% over the noise floor
    min_settled_frac=0.5,       # and until the decay window is half full: one half_life of history
    emit_error_inflation=True,  # also each row's own ratio, against the row's own features
)
rows = po.ModelBank([warm]).fit_predict(df).unnest("warm")
rows.select("pred_y", "settled_frac", "withheld_reason", "error_inflation_y")
```

### Labels that arrive late

`embargo` holds each label back until it would really have arrived, so a
target that is a forward quantity is learned only once it is known. *The
built-in delay* shows the parameter, and *The delay as data* shows
`po.stream.embargo`, which writes the same delay out as rows, and where the
two differ.

A target that is a forward quantity, such as the next five minutes'
return, is not known at the row it sits on. A model that learns it there
sees that much of the future before it predicts the rows in between. Every
out-of-sample number after that is contaminated, and with a feature that is
correlated with its own past, even a pure-noise column starts to look
predictive.

#### The built-in delay

Give a spec `embargo`, finite and above 0, and each row is scored where it
sits and learned from that much later. Everything downstream of the label
moves with it: the prediction, `sigma`, `zscore`, the metrics, break
detection, the conformal interval, `weight_sum` and `min_weight` all see
only labels that had really arrived. `group_close` is refused beside it.

```python
fwd = po.spec.ewridge("fwd", targets=["ret_5m"], features=["x0", "x1"],
                      clock="ts", gap_cap="1h", half_life="30m",
                      embargo="5m")      # the return takes five minutes to be known: learned five minutes later
```

| case | what `embargo` does |
|---|---|
| what the delay counts | the time that passed on the clock column, skipped rows included, and `session_gap` where a session restarts the clock. It is not the capped step the model decays by, so the release depends on the rows alone and survives any chunking |
| with no clock column | one unit is one row of the group, a skipped row included: `embargo=20` is twenty rows |
| at a break | nothing is released early. A break's own effect, such as a lag ring's clearing or `session_shrink`'s blend, waits with the row after it, and runs when that row is learned. A reset drops the rows still waiting |
| at the end of the stream | the rows still waiting are not learned, unless the state is saved and a later run resumes it |
| in the state | the waiting rows are saved with it: one row's values for each row inside the delay, in each group |
| row by row | `emit_clocks=True`, on any model, writes `scored_clock`, the row's own clock, and `learned_clock`, the clock of the newest row the model had learned from when it was scored, both in the clock column's own type. With no clock they are the row's index in its group |

#### The delay as data

[`po.stream.embargo`](https://hgilde.github.io/polars-online/stream.html#polars_online.stream.embargo)
writes the same delay out as data, for when it has to be visible in the
frame, or for an engine other than this one. Every row comes back twice: a
copy to score at `t` with weight 0, and a copy to learn from at
`t + delay`.

**Its input must be in clock order across all its rows**, since order
within each group alone is not enough: sort by the clock first. It adds two
columns: `_online_role`,
which says which copy a row is, and `_online_role_weight`, which the spec
must read as its weight:

```python
doubled = po.stream.embargo(lf, clock="t", delay=5.0)     # every row twice, in clock order
scored = doubled.online.fit_predict(
    [po.spec.ewridge("m", targets=["y"], features=["x0"], clock="t", gap_cap=10.0,
                     half_life=50.0, weight="_online_role_weight")]   # 0 on the copies to score, 1 on the rest
).filter(pl.col("_online_role") == "predict").collect()               # keep the scored copies
```

**The built-in delay agrees with the doubled stream to the bit on `pred`,
`resid` and `weight_sum` when each row's `t + delay` is another row's
clock.** An evenly spaced clock with a delay of whole steps is such a case.
On an irregular clock they differ, since the doubled stream decays a label
from `t + delay` and the built-in delay learns it at the row that releases
it.

**Every residual diagnostic differs in any case**: `sigma`, `zscore`, the
metrics, the conformal band, the quantiles, the autocorrelation and drift.
`embargo` takes in the residual of the prediction the row was scored with.
The doubled stream's learning copy forms its residual at `t + delay`, from
a model that has since learned every row before it.

### Nulls, and three ways to hold a row back

Three ways keep a row from teaching a model: a weight of 0, a null target,
and `predict`. The table compares what each does to the row's score, the
clock, the fit and `weight_sum`; what a null does in each column is under
[What a spec names](#what-a-spec-names).

**In `ewridge` and `lasso` a null target does not take the row from the
other targets' running sums.** Under the default `target_gaps="own_rows"`,
a target missing where the others are present takes a copy of the sums and
keeps its own from then on. Over the rows the target lacks, its copy only
ages, so its fit holds still. Under `"pairwise"` one set of sums learns
every row, so a null target's coefficients drift with the feature noise.

| | the row is scored | the clock advances | the fit moves | `weight_sum` | use it to |
|---|---|---|---|---|---|
| **weight `0`** | yes | yes | no | keeps decaying, so a long stretch can fall below `min_weight` | keep a row's place in the stream |
| **a null target** | yes | yes | not that target's, except under `target_gaps="pairwise"` | counts the row; the target's own weight, which its `min_weight` reads, only decays | leave a label out |
| **`predict`** | yes | no | no | frozen | serve ([Serving without learning](#serving-without-learning)) |

## Preparing a stream

Two tools turn a stream's raw rows into what a model reads. Each runs in one
pass over the stream, returns a `DataFrame` for a `DataFrame` and a query for
a query, and can save its state to go on where a run stopped:

| tool | what it makes | where |
|---|---|---|
| the window operators of [`po.ops`](https://hgilde.github.io/polars-online/ops.html), run by [`po.stream.with_windows`](https://hgilde.github.io/polars-online/stream.html#polars_online.stream.with_windows) or inside a spec | exponentially weighted means, sums and rates over a window of the clock, looking back or ahead, as columns or as a model's target | [Windowed means, looking back or ahead](#windowed-means-looking-back-or-ahead) |
| [`po.stream.refresh_time`](https://hgilde.github.io/polars-online/stream.html#polars_online.stream.refresh_time) | one grid for series that tick at their own times | [Series that tick at their own times](#series-that-tick-at-their-own-times) |

### Windowed means, looking back or ahead

The window operators of [`po.ops`](https://hgilde.github.io/polars-online/ops.html)
compute a trailing time-weighted mean with a hard cutoff, its mirror image
looking ahead, and sums and rates of the same kind, in one pass. Each returns
a Polars expression, used in one of two forms, and a run of the first can be
saved and resumed:

| form | what it gives | where |
|---|---|---|
| [`po.stream.with_windows`](https://hgilde.github.io/polars-online/stream.html#polars_online.stream.with_windows) | any number of formulas as new columns, as `with_columns` runs expressions over a frame | [Windows as columns](#windows-as-columns) |
| a spec's `targets` | a look-ahead as the target a model learns, its windows computed inside the bank | [Windows as a model's inputs and target](#windows-as-a-models-inputs-and-target) |
| `save_state=`, `load_state=` | a `with_windows` run that resumes where it stopped | [Saving and resuming a window run](#saving-and-resuming-a-window-run) |

The rules from here to the first subsection bind both forms.

**A row takes the same work however long its window.** Polars has no form of
either mean that does: its `rolling` window gathers its rows again for every
row. These operators keep each window's weighted sums in a queue instead.
One operator asked for twice is computed once, and operators with the same
direction, half-life, window and `closed` share one queue. What a window
takes in time and memory is under [Window operators](#window-operators), in
Performance.

| operator | the window of row *t* | what it computes |
|---|---|---|
| `po.ewm_mean(x, half_life=, window_size=)` | the rows at or before *t*, less than `window_size` older; with no `window_size`, every row since the last break | Polars' `ewm_mean_by`: the time-weighted mean. Each value is held over the interval ending at its row and weighed by the decayed time, so a burst of rows does not outweigh a quiet period |
| `po.rewm_mean(x, half_life=, window_size=)` | the rows after *t*, at most `window_size` later; `window_size` is required | its mirror: each value held until the next row |
| `po.ewm_sum(x, ...)`, `po.rewm_sum(x, ...)` | the same | Polars' `ewm_sum_by`: `Σ λ^age x`, each row counted once at its own time |
| `po.ewm_rate(x, ...)`, `po.rewm_rate(x, ...)` | the same | the sum over the decayed time the window covers: a quantity per unit of clock |
| `po.increment(x)` | back to the last row with a value, within the group and session | `x_t − x_{t−1}`; null on a session's first row and after a restart; seconds on a temporal column |

Every operator but `po.increment` takes the same keywords, `half_life` and
`window_size` in the clock's units:

| keyword | what it sets |
|---|---|
| `half_life` | how fast the weights fall: largest nearest the row, they halve every `half_life` away from it. `half_life=float("inf")` weighs the window evenly, and an evenly weighted mean needs a `window_size` |
| `window_size` | how far the window reaches from the row; required looking ahead |
| `closed` | which ends of the window are in, `"right"` by default (below) |
| `min_samples` | the fewest rows with a value a window must hold; one holding fewer is null |
| `partial` | what a window cut short by a gap or a session change gives (below) |

**Which rows a window holds follows Polars' `rolling_*_by`.** A window is a
set of timestamps, so every row that shares one timestamp, its *stamp*, gets
the same window, later rows at that stamp included. Whether a row falls
inside another row's window is decided from the two rows' own clocks, exact
in nanoseconds on a temporal clock, as Polars decides it. `closed` says which
ends are in, and so when a trailing window's value is known:

| `closed` | looking back, row *t*'s window | its value is known | looking ahead |
|---|---|---|---|
| `"right"`, the default | `(t − w, t]`: the rows at *t*'s own stamp are in | at the next distinct stamp | `(t, t + w]`: a row exactly `w` later is in |
| `"left"` | `[t − w, t)` | as the stamp arrives | `[t, t + w)`: every row at *t*'s stamp, *t* itself included |
| `"both"` | both ends | at the next distinct stamp | both ends |
| `"none"` | neither end | as the stamp arrives | neither end |

**A formula is element-wise Polars around the operators,** evaluated on each
chunk that goes out:

| | in a formula |
|---|---|
| allowed, around at least one operator | columns, literals, arithmetic, negation, comparisons, `log`, `exp`, `abs`, `sqrt`, `pow`, `clip`, `fill_null`, `is_null`, `is_not_null`, `when/then/otherwise`, `cast` and `alias` |
| refused by name while the query is built, since each would depend on where a chunk ended | a `shift`, a cumulative or rolling function, `over` or an aggregation |
| refused by name while the query is built | a formula with no operator, which is Polars' own `with_columns` |

An operator's input is the same kind of formula, `po.increment` included, so
a rate of traded volume, over a column of cumulative volume, is
`po.ewm_rate(po.increment("cum_volume"), half_life="30s", window_size="5m")`.
An operator's input is not another operator: a formula over an operator's
output is a second call.

**A row with no value for an operator adds nothing to a sum or a rate,** and
a mean skips it, as Polars' `ewm_mean_by` skips a null.

**The window operators read the clock as a spec does, with the same
keywords:** `clock`, `gap_cap`, `restart_after_step_back`, `session`,
`session_gap` and `group`. A window target takes them from its spec, and
`with_windows` takes them itself. With `group`, each group has its own
sessions.

**A gap longer than `gap_cap` or a session change ends every window open
across it.** A reset, where the clock policy starts over, discards every
open window instead: null, never dropped. For a window that a gap or a
session change ends, the operator's `partial` says what it gives:

| `partial` | a cut window gives | the default for |
|---|---|---|
| `"keep"` | the value over what it saw, the window ending at the last row seen | looking back |
| `"null"` | null | looking ahead |
| `"drop"` | the row leaves the output | |

The examples below read `trades`: quotes, each with a `mid`, and trades
between them with a `side`, a `quantity` and a `price`, null on the quotes,
for two symbols on the clock `ts`.

#### Windows as columns

`po.stream.with_windows` adds a column per formula, named by its keyword, and
the input's columns come through as they were. `lf.online.with_windows(...)`
and `df.online.with_windows(...)` are the same call, for a chain. It needs
its rows in clock order across groups, and holds each row back until every
window over it has its value.

A weighted mean is a ratio of two sums. A VWAP, a volume-weighted average
price, is decayed notional (price times quantity) over decayed volume, and a
VWAP of one side puts the same `when/then` inside both sums:

```python
notional = pl.col("price") * pl.col("quantity")
buys = pl.when(pl.col("side") == "buy")
w = dict(half_life="10s", window_size="1m")
out = po.stream.with_windows(
    trades,
    mid_trend=po.ewm_mean("mid", half_life="5s", window_size="1m") - pl.col("mid"),  # the quotes less than a minute old, less the mid
    fwd_vwap=po.rewm_sum(notional, **w) / po.rewm_sum("quantity", **w),      # the next minute's trades, weighted most on the next one
    fwd_buy_vwap=po.rewm_sum(buys.then(notional), **w) / po.rewm_sum(buys.then("quantity"), **w),   # the same, buys only
    clock="ts", gap_cap="5m", group="symbol",
)
# every input column, then mid_trend, fwd_vwap, fwd_buy_vwap

(trades.lazy()
    .online.with_windows(fwd_vwap=po.rewm_sum(notional, **w) / po.rewm_sum("quantity", **w),
                         clock="ts", gap_cap="5m", group="symbol")
    .sink_parquet("with_windows.parquet"))     # streams, holding about one window of rows
```

A quote row has no `price` or `quantity`, so it adds nothing to either sum,
and leaves a VWAP where the trades put it.

**The rows must be in clock order across groups, as one stream.** A spec
checks each group's clock on its own, so a spec's groups may interleave in
time. `with_windows` also reads one clock across the whole stream. A step
back on it is refused, or past `restart_after_step_back` restarts every
group, and a gap past `gap_cap` on it ends every group's windows. A query
whose row order Polars does not guarantee raises `OrderNotGuaranteedWarning`,
as a bank's does.

**Rows leave in input order, each once every window over it has its value,**
so the output trails the input by the longest window. A group that falls
silent holds back the rows after it for at most `gap_cap` of the stream's
time.

**Two things take a row out of the output.** `partial="drop"` does, and so
does `save_state=` for a row still waiting when the input ends: the next run
returns it first
([Saving and resuming a window run](#saving-and-resuming-a-window-run)).
Without a saved state, a row still inside its window at the end is null.

**The call's own keywords cannot name an output.** The clock keywords,
`clock`, `gap_cap`, `restart_after_step_back`, `session`, `session_gap` and
`group`, are not output names: an expression passed as `session=` is
refused.

#### Windows as a model's inputs and target

A trailing window made by `with_windows` is an input like any column, so a
model in the same query can learn from it. A look-ahead can instead be the
target: its expression goes straight into the spec's `targets`, named by its
alias, and the bank computes its windows itself. Each row is scored where it
sits and learned from once its window has closed and its `embargo` has
passed, and `with_windows`, given the spec as `like=`, writes the same target
as a column.

**A target expression needs an alias and an operator that looks ahead,** and
`group_close` is refused beside a window target. The bank keeps one set of
windows per group on the spec's clock, so the groups may interleave in time,
as a spec's always may.

```python
notional = pl.col("price") * pl.col("quantity")
w = dict(half_life="10s", window_size="1m")
clock = dict(clock="ts", gap_cap="5m", group="symbol")    # one clock for the windows and the model
fwd_edge = (po.rewm_sum(notional, **w) / po.rewm_sum("quantity", **w) - pl.col("mid")).alias("fwd_edge")   # the next minute's VWAP, less the mid
edge = po.spec.ewridge("edge", targets=[fwd_edge], features=["trend_5s", "trend_30s"],   # features made below
                       half_life="30m", embargo="1m", **clock)   # the target reads the next minute
fitted = (
    trades.lazy()
    .online.with_windows(
        trend_5s=pl.col("mid") - po.ewm_mean("mid", half_life="5s", window_size="2m"),   # the inputs read only the past
        trend_30s=pl.col("mid") - po.ewm_mean("mid", half_life="30s", window_size="2m"),
        **clock,
    )
    .online.fit_predict([edge])
    .collect()
)
```

**Each row is scored where it sits, and learned from once its window has
closed and its `embargo` has passed, whichever is later.** A state that
learned a row before its window closed would score the rows that window
covers in sample. So `fit_predict` requires an `embargo` of at least the
look-ahead's `window_size`, on the same clock. `fit` and the command line's
`--no-output` keep no prediction, so they take any embargo, none included,
and learn each row once its window closes. The embargo counts elapsed time,
and a break releases nothing early, so even under a window longer than
`gap_cap` no row is learned before its time.

**A gap past `gap_cap` or a session change cuts an open window.** The
operator's `partial` then says whether the row is learned from what the
window saw (`"keep"`), or not at all (`"null"`, the default looking ahead).
For a target, `"drop"` means `"null"`, and the row's other targets are still
learned.

**The target is not known on the row it is scored on, so `resid_<name>` is
null there:** `resid_fwd_edge` in the example. The diagnostics take the row
in when it is learned, as under any embargo
([Labels that arrive late](#labels-that-arrive-late)).

**The same target as a column shows what the model learned.**
`po.stream.with_windows(trades, fwd_edge, like=edge)` writes the window under
the spec's clock, and nulls it on every row the spec would not learn from.
`like=` takes the clock keywords from the spec, and refuses any given beside
it. Fed back as a plain target under the same embargo, that column gives the
same predictions, row for row, except when the embargo equals the window,
under `closed="right"`, as in the example above. A row exactly one window
later is then learned at its embargo in the column form, and at the next
distinct stamp in the native form, the expression in `targets`.

**The spec, the saved state and the command line's TOML carry the
expression,** and a bank saved mid-window resumes with its windows open.

#### Saving and resuming a window run

`save_state=` writes the windows' sums and the rows still waiting for their
windows to close, and a run given `load_state=` returns those rows first. So
a stream fed in two runs gives what one run gives. A state resumes only the
call that saved it, on the same kind of clock, and a bank keeps a window
target's windows in its own state instead
([Saving, loading and serving](#saving-loading-and-serving)).

That holds for a stream split into two files, and for one read under a
slice. Under a slice such as `head(500)`, the run reads only until the
windows of the last row asked for have closed, and the state records how many
rows of the input it consumed:

```python
w = dict(half_life="10s", window_size="1m")
vwap = dict(fwd_vwap=po.rewm_sum(pl.col("price") * pl.col("quantity"), **w) / po.rewm_sum("quantity", **w))
clock = dict(clock="ts", gap_cap="5m", group="symbol")
one_run = po.stream.with_windows(trades, **vwap, **clock)            # one run

day1, day2 = trades.head(2000), trades.slice(2000)                   # the same stream as two files
first = po.stream.with_windows(day1, **vwap, **clock, save_state="w.state")    # its last rows wait
second = po.stream.with_windows(day2, **vwap, **clock, load_state="w.state")   # they come out first
assert pl.concat([first, second]).equals(one_run)

head = trades.lazy().online.with_windows(**vwap, **clock, save_state="h.state").head(500).collect()
rest = trades.lazy().online.with_windows(**vwap, **clock, load_state="h.state").collect()
assert pl.concat([head, rest]).equals(one_run)                       # the same input, unsliced: it skips them
```

**The state knows its input.** It keeps the clock and session of the input's
first row, and the last rows it read, so a run resumed on another input
cannot skip or double rows silently. Two limits follow. A clock that starts
over at the same stamp each day needs a session column, or the next day's
file starts as the saved input did. And an input whose rows match the
state's where it was cut is taken as the same input, so a stream whose rows
can repeat resumes on the same input only.

A refusal names `another input`, and while a query runs it surfaces as
`polars.exceptions.ComputeError`, with the message inside. A resumed run
treats each input this way:

| the input a resumed run is given | the run |
|---|---|
| the input the state was saved from, unsliced | skips the rows the state consumed, then goes on |
| the next file, whose first row is a step forward on the clock | skips nothing, and first returns the rows the state held |
| the next file, at a new start: its first row is a step back past `restart_after_step_back`, or a new session | skips nothing, and first returns the rows the state held |
| an input whose first row is at the last stamp the state read | refused: a file boundary inside a stamp several rows share cannot be told from the same input sliced inside it |
| an input that steps back where the clock policy refuses it: the same input sliced by hand, or an overlapping file | refused |
| without a clock column, an input that does not begin with a new session | refused: a row-count clock steps forward at every row |
| an input that starts as the saved one did but differs where the state was cut, or ends before the rows consumed | refused |

### Series that tick at their own times

Two series observed at different instants cannot be correlated directly, so
[`po.stream.refresh_time`](https://hgilde.github.io/polars-online/stream.html#polars_online.stream.refresh_time)
puts them on the grid Barndorff-Nielsen, Hansen, Lunde and Shephard defined.
Each grid point holds every series' last observed value, its tick count since
the previous point, and the share of ticks the grid kept. A model reads the
grid as it reads any frame.

A fine common grid pushes the correlation toward zero, the Epps effect, and
filling values forward invents observations that were never made. A
*refresh time* is the first instant by which **every** series has ticked at
least once since the previous refresh time. The grid has a point at each
refresh time.

The input is in long form: one row per tick, saying which series ticked,
when, and what it observed. Three series on a clock `t`:

```python
ticks = pl.DataFrame({
    "symbol": ["AAA", "BBB", "AAA", "CCC", "BBB", "AAA", "AAA", "CCC"],  # which series ticked
    "t": [0.4, 0.9, 1.3, 1.6, 2.2, 2.5, 2.8, 3.1],                      # when, in clock order
    "px": [100.0, 20.0, 100.2, 50.0, 20.1, 100.1, 100.4, 49.9],         # what it observed
})
grid = po.stream.refresh_time(ticks, series="symbol", names=["AAA", "BBB", "CCC"],
                              clock="t", value="px")
```

| time_refresh | AAA_value | BBB_value | CCC_value | n_obs_AAA | n_obs_BBB | n_obs_CCC | retained_fraction |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.6 | 100.2 | 20.0 | 50.0 | 2 | 1 | 1 | 0.75 |
| 3.1 | 100.4 | 20.1 | 49.9 | 2 | 1 | 1 | 0.75 |

Each column of the grid holds:

| column | what it holds |
|---|---|
| `time_refresh` | the refresh time: the clock of the tick that completed the point |
| `AAA_value`, ... | each series' last observed value at that time. The output looks synchronous and is not: each value is up to one of its own inter-tick intervals old |
| `n_obs_AAA`, ... | that series' ticks since the previous point, of which the grid kept the last |
| `retained_fraction` | the share of those ticks the grid kept. Read it before trusting a correlation |

The first refresh time is 1.6, when CCC ticks for the first time. AAA's
value there is 100.2, from its tick at 1.3, and its tick at 0.4 is dropped.
Since 1.6, BBB has ticked at 2.2 and AAA twice, so CCC's tick at 3.1
completes the second point.

**The grid runs at the pace of the slowest series,** so a fast one loses
most of its ticks, and `retained_fraction` says what share it kept. The
series holding the grid up is the one whose tick completes each point, so
its `n_obs` stays near 1. A large `n_obs` counts ticks the grid dropped,
since it keeps only the last of them.

**`pairs=True` runs an independent two-series grid for each pair instead,**
which keeps far more when one series is slow.

**Rows must be in clock order within each `group`:** a step back is an error
naming the row, and nothing is interpolated.

**With `save_state=` and `load_state=`, the grid resumes where a run
stopped,** part-way through an interval or not, as a bank does.

**Fit a model on each series' change between grid points, on the grid's own
clock, `time_refresh`.** Given a query, `refresh_time` returns one, and
[`lf.online.fit_predict(specs)`](#as-a-query-lfonlinefit_predict) runs the
models in the same pass, with the same numbers. Here `ticks` is a longer
stream of the same shape, a hundred ticks of each series. `ew_cov` tracks the
three correlations, and `ewridge` regresses one series' change on the other
two's:

```python
names = ["AAA", "BBB", "CCC"]
grid = po.stream.refresh_time(ticks, series="symbol", names=names, clock="t", value="px")
returns = grid.select(
    "time_refresh",
    *(pl.col(f"{s}_value").diff().alias(s) for s in names),  # each series' change since the last point
)
specs = [
    po.spec.ew_cov("comove", features=names, stats=["corr"],
                   clock="time_refresh", gap_cap=10.0, half_life=20.0),
    po.spec.ewridge("beta", targets=["AAA"], features=["BBB", "CCC"],
                    clock="time_refresh", gap_cap=10.0, half_life=20.0),
]
out = po.ModelBank(specs).fit_predict(returns)
# comove: corr_AAA_BBB, corr_AAA_CCC, corr_BBB_CCC, each from the points before the row
# beta:   pred_AAA, resid_AAA and coef: AAA's change predicted from BBB's and CCC's
# The first grid point has no change, so both models skip it.
```

## Running a bank

A bank runs three ways, and each gives the same numbers. Whichever way it
runs, its output can also leave as Arrow, for a consumer that is not Polars:

| way | the call | use it for | where |
|---|---|---|---|
| inside a Polars query | `lf.online.fit_predict(specs)` | ordinary Polars steps after the bank | [As a query](#as-a-query-lfonlinefit_predict) |
| in your own Python loop | `bank.fit_predict(chunk)` | reading the bank between chunks | [In a loop](#in-a-loop-modelbank) |
| from a standalone command line, with no live Python at all | `online` | a scheduled job, or a deployment with no Python | [Outside a live Python process](#outside-a-live-python-process) |
| output as Arrow | `fit_predict_arrow`, `predict_arrow` | a consumer that is not Polars | [Output as Arrow](#output-as-arrow) |

### As a query: `lf.online.fit_predict`

A Polars `LazyFrame` is a query that runs only when you ask for its result.
[`lf.online.fit_predict(specs)`](https://hgilde.github.io/polars-online/namespaces.html#polars_online._frame.LazyFrameOnlineNamespace.fit_predict)
puts a model bank inside one, and `df.online.fit_predict(specs)` does the
same for a `DataFrame` already in memory. When the query runs, its rows go
through a bank that starts with nothing learned, `chunk_rows` rows at a time,
and every step after the bank is ordinary Polars.

`collect()` gives the result as one frame, `sink_parquet()` writes it to a
file without holding it all, and `collect_batches()` gives it a chunk at a
time:

```python
spec = po.spec.ewridge(                                   # a ridge regression of y for each stock, at two ridge values
    "ridge",
    targets=["y"], features=["x0", "x1", "x2"],
    clock="t", half_life=600.0, gap_cap=300.0,
    group="stock_id", ridge=[1e-6, 0.1], standardize=True,
)
files = pl.scan_parquet("ticks/*.parquet")                # a query over the files; nothing is read yet
(
    files
    .online.fit_predict([spec], chunk_rows=100_000)       # a bank with nothing learned yet; every run starts from the same place
    .filter(pl.col("ridge").struct.field("weight_sum") > 100)  # after the bank: filters what comes out, never what the bank learns from
    .select("ts", "stock_id", "ridge")                     # Polars reads only these columns, and those the specs read, from the files
    .sink_parquet("fitted.parquet")                       # runs the query; memory is state + one chunk, however long the files
)
# "ridge" is one column whose value per row is a record of named fields:
#   {pred_y__r0.000001, resid_y__r0.000001, pred_y__r0.1, resid_y__r0.1,
#    weight_sum, settled_frac, withheld_reason, coef, support_coef}

lf.online.fit_predict([spec]).head(5).collect()           # learns from the first 5 rows and no more
```

**For a type checker, the same calls are plain functions.** The package adds
the `.online` methods to Polars' frames when it is imported, so a type
checker cannot see them. `po.fit_predict(frame, ...)`,
[`po.predict(frame, bank)`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.predict)
and [`po.unnest(frame, specs)`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.unnest)
are the same calls as plain functions, which it can see.

**Filter after the bank, not before it, unless the model must skip those
rows.** A filter after the bank never changes what the bank learns from, and
the query still runs a chunk at a time. A filter before the bank makes
Polars hold several blocks of each parquet file per thread: 2.5 GB at 12M
rows, against 0.78 GB for the same filter after
([docs/PERFORMANCE.md](docs/PERFORMANCE.md) §11).

**When the model must not learn from some rows, give them weight `0` instead
of filtering them out.** They still flow through and come out scored, and no
gap opens in the clock:

```python
(
    lf.with_columns(pl.when(pl.col("venue") == "X").then(1.0).otherwise(0.0).alias("w"))  # learn from venue X only
    .online.fit_predict([po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"],
                                         clock="t", half_life=600.0, gap_cap=300.0,
                                         weight="w")])
    .sink_parquet("fitted.parquet")
)
```

**A mistake in the query's construction, such as a missing column, is a
`ValueError` when the query is built.** What the bank refuses while the query
runs, such as a step back, arrives as `polars.exceptions.ComputeError`, with
the bank's message inside.

### In a loop: `ModelBank`

A [`ModelBank`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.ModelBank)
is the bank as a Python object: you feed it chunks with
`bank.fit_predict(chunk)`, and between chunks it can tell you what it holds
([What a bank holds](#what-a-bank-holds)). `fit_predict_batches` does the
chunking for you, `fit` keeps the state and no output, and a long-running
bank can drop the groups that have gone quiet.

```python
bank = po.ModelBank([spec])                # the spec above; nothing learned yet

for chunk in lf.collect_batches():        # the query, one chunk at a time; the whole stream is never in memory
    out = bank.fit_predict(chunk)         # the chunk's columns, plus one column per spec
    ...

bank.save("bank.state")                    # written whole or not at all: a temporary file, then a rename

# The chunking done for you: a query or a DataFrame in chunk_rows rows, an iterator of frames as it comes.
for out in po.ModelBank([spec]).fit_predict_batches(lf, chunk_rows=100_000):
    ...

# A run whose product is the state: learn from every row, keep no output.
fitted = po.ModelBank([spec])
fitted.fit(lf)
```

**One thread at a time feeds a bank, and `predict`, which learns nothing, may
run from any number of threads.** A bank reads one ordered stream, so a call
that finds it busy on another thread raises `RuntimeError` rather than
interleave the two.

**Groups live until dropped,** so a long-running bank forgets the ones that
have gone quiet, and they start over if they reappear. `last_clock` is in the
clock's own units, and in seconds since 1970 on a temporal clock; `now` below
is the current time in the same units:

```python
bank = po.ModelBank([spec])
bank.fit(lf)
stale = bank.groups().filter(pl.col("last_clock") < now - 30 * 86400)
bank.drop_groups(stale["group"])           # they start over if they reappear
```

### Outside a live Python process

A scheduled job, or a deployment with no Python at all, runs the same bank
from a file to a file: [docs/RUNNER.md](docs/RUNNER.md) has the standalone
`online` command line. It takes the same specs and the same state file, and
gives the same numbers.

### Output as Arrow

For a consumer that is not Polars, `fit_predict_arrow` and `predict_arrow`
give the output of `fit_predict` and `predict` through the Arrow PyCapsule
interface. The capsule interface is an Arrow specification, so a consumer
that reads `__arrow_c_array__` takes the result directly. Each result can be
read once: exporting hands the buffers to the consumer, and a second read
raises `ValueError`.

```python
bank = po.ModelBank([spec])
structs = bank.fit_predict_arrow(df)                    # one per spec, as Arrow; each can be read once
out = df.with_columns([pl.Series(s) for s in structs])  # or any reader of __arrow_c_array__
scored = bank.predict_arrow(today)                      # the same for predict
```

**Each struct, an `ArrowStruct`, is one spec's output, exposing
`__arrow_c_array__`.** Its values are `fit_predict`'s field for field and null
for null; only the way out differs. The input still goes in as a Polars
frame.

A consumer that wants the *stream* interface takes it through a `Series`
first:

| consumer | measured on | takes |
|---|---|---|
| pyarrow | pyarrow 25.0.1 | a struct as it is: `pa.array(s)` and `pa.table(s)` each take one, values and types unchanged |
| DuckDB | duckdb 1.5.5 | `pl.Series(s)`, and refuses an `ArrowStruct`, because `__arrow_c_stream__` is the method it looks for; a spec's output arrives table-shaped, one column per field |

**A Polars `Series` crosses on py-polars' private methods,** which is why this
package measures a Polars range rather than promising one
([Which interfaces carry a promise](#which-interfaces-carry-a-promise)). The
capsule interface does not rest on them.

## Saving, loading and serving

A bank's state is everything it has learned, for every spec and group,
including the rows still waiting out an `embargo` and each window target's
open windows. It can be saved to one file and loaded back to keep learning
([Save and load](#save-and-load)), or to score new rows without learning
([Serving without learning](#serving-without-learning)). It can also be
exported as JSON
([Reading a state without this library](#reading-a-state-without-this-library)).
No state saved by a release so far, 0.13.0 included, loads in this build:
refit from the input.

A run of `with_windows` keeps a state file of its own, under rules of its own
([Saving and resuming a window run](#saving-and-resuming-a-window-run)).
[docs/STATE-WORKFLOW.md](docs/STATE-WORKFLOW.md) walks a bank's whole
workflow, fit, save, serve and learn on, with what each step guarantees.

### Save and load

Each way of running a bank saves and loads the same file: written whole or
not at all, and the same whichever way wrote it, to the byte. Each writes it
at its own moment:

| run by | saves with | loads with | the file is written |
|---|---|---|---|
| a bank object | `bank.save(path)` | `po.ModelBank.load` | when `bank.save` is called |
| a query | `save_state=` | `load_state=` | when the run reaches its last row, with the same bytes a `ModelBank` would write |
| the command line | `--save-state` | `--resume` | only after its output is committed |

```python
# df is the stream so far; today, and the query later, hold the rows that follow it.
# From a bank object
bank.fit_predict(df)
bank.save("bank.state")                               # written whole or not at all: a temporary file, then a rename
bank = po.ModelBank.load("bank.state", specs=[spec])  # specs= checks the file holds this model, not another
scored = bank.predict(today)                          # serve: score the rows, learn nothing
bank.fit_predict(today)                               # keep learning, once the targets have arrived: the state moves

# From a query
lf.online.fit_predict([spec], save_state="bank.state").sink_parquet("fitted.parquet")             # fit, then save at the last row
served = later.online.predict("bank.state").collect()                                             # serve from the file
later.online.fit_predict(load_state="bank.state", save_state="bank.state").sink_parquet("more.parquet")  # the next rows: continue, save again

# A rerun overlaps the state; skip_learned keeps only the rows not yet learned (below):
bank.fit_predict(bank.skip_learned(today))

# The same file, in memory rather than on disk -- for a checkpoint that lives somewhere else:
blob = bank.save_bytes()
bank = po.ModelBank.load_bytes(blob, specs=[spec])
```

**A run abandoned early, or ended by an error inside the bank, leaves the
file untouched.** On Polars 1.x an error in a later step of the query does
not stop the bank, so the state is written although the query failed. On
py-polars 2.0.0rc1, the error stops the bank instead when it arrives before
the bank reaches the end of its input, and the state is not written.
[docs/STATE-WORKFLOW.md](docs/STATE-WORKFLOW.md) has the measurements.

**Loading names the problem it meets:**

| the file | raises |
|---|---|
| does not exist yet | `FileNotFoundError` |
| is not a bank, or holds different specs from the `specs=` given | `ValueError` |
| holds a state that contradicts its own spec | `ValueError` |
| was written under a newer state schema or file format than this build reads | `ValueError` |
| was written under a state schema below 25, which is every release so far, 0.13.0 included | `ValueError`, naming the range: refit from the input. `po.schema_version()` gives this build's schema |

**`skip_learned` resumes on input that overlaps the state.** Input that
starts before the save makes every group's clock step back, which is refused
unless `restart_after_step_back` reads it as a new start. `skip_learned`
needs a clock to resume from, so it refuses a bank whose specs all count
rows. `bank.skip_learned(frame)` takes a `DataFrame`, or a `LazyFrame` it
keeps lazy. It keeps the rows after each group's last clock, a row at that
clock counting as learned, and every row of a group the bank has not seen,
so each row is learned once.

### Serving without learning

`predict` scores rows against a state and learns nothing, so every row is
scored from the same state: `bank.predict(frame)` on a bank object, and
`lf.online.predict(...)` in a query. It skips each row's update, so it runs
faster than learning: `ewridge` scores 1.8 times as fast as it learns at 5
features, and 2.9 times at 20 ([docs/PERFORMANCE.md](docs/PERFORMANCE.md) §9).

```python
scored = bank.predict(today)                             # every row scored from the same state: nothing moves
served = later.online.predict(bank).collect()            # a bank, as it stands when the query runs
from_file = later.online.predict("bank.state").collect() # a path, read when the query is built
```

**Row *i* of `scored` carries what `fit_predict` would have reported had it
been the next row of its group's stream:** `pred`, `weight_sum`, `sigma`,
`zscore`, the selection and the metrics, field for field
([Per-row diagnostics](#per-row-diagnostics)). Two fields differ: `drift`
never fires, and `coef` is filled on each group's last accepted row only,
since the same coefficients score every row. How `predict` reads a row:

| the row | `predict` |
|---|---|
| its target column | may be absent; `resid` is then null |
| its weight column | is not read |
| of a group the bank has never seen | scores null |
| before the last clock its group learned | is scored against the state as it stands, with a clock step of 0: never refused, never a restart |
| where `session_gap="reset"` or `group_close="session"` would restart its group | scores null, with `weight_sum` 0 |
| after rows an `embargo` still holds | sees the state as it stands: a held row whose delay has passed by the scored row's clock is not released |

### Reading a state without this library

`bank.to_json()` gives everything `save()` writes, as JSON, to look at a
state, compare two, or hand one to a non-Python program. `load` reads the
binary form only, so the JSON goes one way. Every export is read back and
checked against the state before you get it, so a state that could not be
carried raises an error rather than leaving a file that is quietly wrong.

```python
text = bank.to_json()          # everything save() writes, as JSON
bank.save_json("bank.json")    # the same, to a file
```

**`NaN` and `±inf` are written as strings.** An ordinary state carries an
infinity, since `half_life=inf` means no forgetting, and a plain JSON encoder
writes one as `null` without saying so. So the export writes them as the
strings `"nan"`, `"inf"` and `"-inf"`, the same spelling a spec's
`half_life` uses, and stays faithful even to the values JSON cannot write.

## Reading the fit

A fit can be read from two places: the bank, live or loaded from its file,
with no data at hand, and the output, row by row. The bank's readings
describe each group after the last row it learned from, except a closed
group's row, read at the moment the group closed. Each subsection reads one
thing:

| to read | from | subsection |
|---|---|---|
| what the bank holds, and what it was fed | the bank | [What a bank holds](#what-a-bank-holds) |
| any field of the output, by name | the output | [Output field names](#output-field-names) |
| the coefficients | either | [Coefficients](#coefficients) |
| the running sums a fit is solved from, and `po.gram`'s algebra on them | the bank | [The running sums behind a fit](#the-running-sums-behind-a-fit) |
| a finished group's running sums | the bank | [One row per finished group](#one-row-per-finished-group) |
| the arithmetic after a correlation matrix, with `po.corr` | an array, a `gram()` dict or a closed row | [Reading a correlation matrix](#reading-a-correlation-matrix) |

### What a bank holds

A bank, live or loaded from its file alone, answers two kinds of question
with no data at hand. Six calls describe what it holds, and four tables say
how the fit is doing and what it was trained on. Loading needs no specs, no
configuration and no data.

```python
bank = po.ModelBank.load("bank.state")   # no specs=: the file is enough

repr(bank)                # ModelBank(['ridge'], groups=4, rows_seen=400)
bank.specs                # every spec back, as the dict its builder made: a copy, read-only
bank.groups()             # one row per (spec, group):
                          #   ┌───────┬───────┬────────────────┬────────────┐
                          #   │ spec  ┆ group ┆ rows_processed ┆ last_clock │
                          #   │ ridge ┆ b0    ┆ 100            ┆ 396.0      │
                          #   │ ridge ┆ b1    ┆ 100            ┆ 397.0      │
                          #   └───────┴───────┴────────────────┴────────────┘
bank.output_fields()      # {'ridge': ['pred_y__r0.000001', ..., 'weight_sum', 'settled_frac', 'withheld_reason', 'coef', ...]}
bank.rows_seen()          # rows fed, over every chunk and group
bank.solve_failures()     # per spec, per group: solves that needed jitter or kept the previous fit
```

**`bank.specs` is a copy, and read-only.** The bank runs the state it was
built from, so a list edited on the Python side would only ever mislabel
what `coef()` reports.

**The four tables return every spec by default, with `spec` as the first
column.** `summary()` and `describe()` have the same columns for every
spec, so banks from different runs stack with a plain `concat`.
`last_row()` stacks with `"diagonal_relaxed"`, since specs with different
fields leave nulls.

```python
bank = po.ModelBank.load("bank.state")
last = bank.last_row("ridge")     # one row per group, the output row of the last row it learned from:
                                  # spec, group, pred_y__r0.000001, ..., weight_sum, coef
betas = bank.coef()               # one row per coefficient, with the term it belongs to (Coefficients, below)
fed = bank.summary("ridge")       # one row per group: what it was fed, and its warm-up readings (below)
cols = bank.describe("ridge")     # one row per input column per group: column, role, count,
                                  # null_count, mean, std, min, max

# Fit many models, save each, and compare them with one concat over the files:
from pathlib import Path
table = pl.concat(
    [po.ModelBank.load(f).last_row() for f in sorted(Path(".").glob("*.state"))],
    how="diagonal_relaxed",       # specs with different fields stack with nulls
)
```

`last_row()` is the `fit_predict` row field for field: `pred`, `sigma`, the
metrics and the interval when the spec asks for them ([Per-row
diagnostics](#per-row-diagnostics)), `weight_sum`, and `coef` when that row
carried it. A group that has not learned from a row yet gives nulls.

**`summary()`'s counts and `describe()` never forget.** They are plain
counts over the whole stream, taken in row order, so they are the same
whatever the chunking, and `predict` does not move them. `describe()`
counts as the models count: a null, a NaN, an infinity or a magnitude
beyond 1e100 is a `null_count`, not a value. `summary()`'s warm-up readings
are read from the state as it stands after the last row. Its columns:

| columns | what they hold |
|---|---|
| `rows_fed` | rows routed to the group |
| `rows_processed` | rows the model accepted |
| `rows_skipped` | rows it did not accept: a feature or the weight was missing |
| `rows_learned` | rows that moved the fit: a weight above zero and a target present. Under an embargo a row is counted as it arrives; a window target's row, when it is released |
| `rows_zero_weight` | rows that advanced the clock and nothing else |
| `weight_sum`, `clock_min`, `clock_max`, `last_clock` | the weight behind the state, and the clock's range and last value |
| `session_changes`, `clock_backwards`, `resets` | what the clock rules met |
| `settled_frac`, `error_inflation`, `min_support_coef` and the feature it belongs to, `n_coef` | the warm-up readings after the last row |

### Output field names

You address the output by field name, so the names are a contract. Two
tables look any name up, so you never have to build one:
`po.spec.output_index` for every field, and `po.spec.coef_fields` for every
coefficient. A grammar, after them, is for reading names by eye, and
[docs/OUTPUTS.md](docs/OUTPUTS.md) lists every field of every model.

Every name, default and signature is pinned against a checked-in snapshot,
so a change is a reviewable diff and a version bump. No release renames
your columns silently.

A spec given several values of `ridge` or `half_life`, several
`feature_sets`, or a `lasso_path` is a *grid*. Each combination of its
values is a point of the grid, and each field's name says which point it
holds. Here `grid` has two half-lives and two ridge values:

```python
grid = po.spec.ewridge("m", targets=["y"], features=["x0", "x1"], clock="t",
                       gap_cap=300.0, half_life=[100.0, 500.0], ridge=[1e-6, 0.5])
out = df.online.fit_predict([grid])  # df's columns, plus the spec's column "m"

idx = po.spec.output_index(grid)     # every field, with the values its name encodes, and its dtype
name = idx.filter((pl.col("kind") == "pred") & (pl.col("target") == "y")
                  & (pl.col("ridge") == 0.5) & (pl.col("half_life") == 500.0))["field"].item()
out["m"].struct.field(name)                                    # "pred_y__r0.5@h500"

row = po.spec.coef_fields(grid).filter(                        # one row per coefficient: its list, position, and unnest column
    (pl.col("term") == "x1") & (pl.col("ridge") == 0.5) & (pl.col("half_life") == 500.0)
).row(0, named=True)
out["m"].struct.field(row["field"]).list.get(row["position"])  # field "coef@h500", position 5
```

Both tables come from the same Rust code that renders the names, so they
cannot drift from the strings. The index also carries each field's
`dtype`, the column type the bank declares to Polars before it reads the
first row.

**The names follow a grammar, for reading them by eye.** A number in a name
renders as a plain decimal in `[1e-6, 1e7)` and in compact scientific
notation outside it.

**If you parse field names downstream, keep `__` and `@` out of target
names and feature-set labels.** A target named `y__r0.5` renders like a
ridge grid on `y`.

```
pred_{target}{combo}{instance}
resid_{target}{combo}{instance}
sigma_{target}{combo}{instance}
abs_resid_q{level}_{target}...
weight_sum{instance}
settled_frac{instance}
withheld_reason{instance}
coef{instance}
support_coef{instance}
scored_clock, learned_clock        under emit_clocks, once per spec
```

| `{combo}` | when |
|---|---|
| `""` | a single ridge, no feature sets |
| `__r{ridge}` | a ridge grid |
| `__{set}` | feature sets, a single ridge |
| `__{set}_r{ridge}` | feature sets and a ridge grid |
| `__l{lambda}` | a lasso path |

| `{instance}` | when |
|---|---|
| `""` | a single `half_life` |
| `@h{half_life}` | a `half_life` grid; a duration as its text, `@h10m` |

### Coefficients

Coefficients come from the bank, as a table, or from the output, as columns
that show the fit as it moved. At any row the two give the same numbers.
Both are the fit *after* that row's update, while the row's own `pred`
comes from the fit *before* it.

```python
ols = po.spec.ewridge("ols", targets=["y"], features=["x0", "x1"], clock="t",
                      half_life=600.0, gap_cap=300.0, group="stock_id",
                      coef_every=1)         # write coef on every row

# 1. From a bank -- live, or loaded from a state file with no data at hand.
bank = po.ModelBank([ols])
bank.fit_predict(df)
betas = bank.coef()                  # one row per coefficient: spec, group, instance, weight_sum, ..., term, coef
wide = betas.pivot("term", index=["group", "instance"], values="coef")   # a row per (group, instance), a column per term

# 2. From the output, as columns: the fit as it moved, one row per row.
path = (
    lf.online.fit_predict([ols])
    .online.unnest([ols])            # pred_y, resid_y, weight_sum, coef_y_intercept, coef_y_x0, coef_y_x1, ...
    .select("t", "stock_id", "^coef_.*$")
    .collect()
)
```

**`coef_every` sets which rows carry `coef`.** With `coef_every=1` it is a
list of one float per term on every row. The default, `0`, writes it on
each group's last row in a chunk.

**`unnest` spreads that list into columns, and reads a saved output the
same way.** It takes the specs, a bank, or the path of a saved state, as in
`pl.scan_parquet("fitted.parquet").online.unnest([ols])`.

**Several targets, or a grid, put several blocks in one `coef` list.** The
list holds one block per target and point of a grid of `ridge` values,
`feature_sets` or a `lasso_path`. Each half-life of a `half_life` grid has
a `coef` field of its own, as `coef@h500`. `unnest` names each block's
columns the way the `pred` fields are named, so `coef_y_x0__r0.5@h500` sits
beside `pred_y__r0.5@h500`. `bank.coef()` tells the blocks apart by its
`target`, `ridge`, `feature_set` and `penalty` columns; add them to the
pivot's `index`.

### The running sums behind a fit

`ewridge`, `lasso` and `ew_cov` keep a matrix of running sums and solve
against it, their *Gram* in least-squares terms; the other models keep
none. `bank.gram(spec)` returns it, per group and per half-life, and
`po.gram` does the algebra on it offline; both need numpy, an optional
extra. A Gram is a complete summary of the rows the model has seen, a
sufficient statistic in the statistical sense, so a saved state answers
questions the run never asked.

Four things to know before reading a Gram:

| | why |
|---|---|
| `weight_sum` counts weight, not rows | `n_kish = weight_sum² / Σw²` is the number of equally weighted rows the moments are worth, which is what a standard error divides by. It does not fall when a stream goes quiet; `weight_sum` does |
| a spec can have several Grams | under the default `target_gaps="own_rows"`, a target that goes missing on different rows from the others is fitted from a Gram of its own. `gram()` returns one dict per Gram, each naming its `targets` |
| `coef()` and `gram()` can disagree | `bank.coef()` is as of the model's last *solve*, which its `solve_every` schedule decides, while `gram()` is as of the last row |
| under a `window_size`, a Gram is the window's | every array, the target moments included, covers the rows inside the window, so `po.gram.solve` on it fits the window |

```python
ols = po.spec.ewridge("ols", targets=["y"], features=["x0", "x1", "x2"],
                      half_life=500.0, ridge=1e-9, standardize=False)
fitted = po.ModelBank([ols])
fitted.fit_predict(df)

g = fitted.gram("ols")[0]      # one dict per (group, half_life, Gram); this spec has one
g["columns"]                              # the k terms of a row z: "intercept" first, then the features
g["targets"]                              # the targets fitted from this Gram
g["means"], g["comoments"]                # the feature means, and the centred k x k co-moment matrix
g["cross_moments"], g["target_weights"]   # per target: the uncentred E[z*y], and the weight behind it
g["means_by_target"]                      # per target: the column means m over the rows it was present on
g["cross_centred"]                        # per target: E[(z - m)(y - ybar)], ybar its own mean: what the fit is solved from
g["target_means"], g["target_vars"]       # per target: the target's own mean and centred variance
g["weight_sum"], g["n_kish"], g["target_n_kish"]   # the accumulated weight, and Kish's effective sample size (features, and per target)

# The algebra the model runs, done by hand: the centred system, in which a feature far from zero
# loses no precision.
slopes = np.linalg.solve(g["comoments"][1:, 1:], g["cross_centred"][0][1:])   # column 0 is the intercept
intercept = g["cross_moments"][0][0] - g["means_by_target"][0][1:] @ slopes
resid_var = g["target_vars"][0] - slopes @ g["comoments"][1:, 1:] @ slopes
r2 = 1 - resid_var / g["target_vars"][0]
```

The [`ModelBank.gram`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.ModelBank.gram)
reference states every array and the identities that relate them.

**`po.gram` is the toolkit for these, so the same algebra is one call.** It
works on any Gram, here one read from a saved state with no data at hand.
Two limits:

| | why |
|---|---|
| `po.gram.solve` solves a plain ridge | it does not reproduce a `ridge_scale="sum"` or `coef_prior` fit |
| `po.gram.merge` pools parts that share a weighting | two halves of a decayed stream in time order need the earlier one's weights decayed first |

```python
g = po.ModelBank.load("bank.state").gram("ridge")[0]   # a saved state's Gram, its first group's

# residual variance, R², standard errors and t:
r2 = po.gram.coef_stats(g, po.gram.solve(g, ridge=1e-9))["r2"]
# the model's own ridge, in original units; a list of ridges is one eigendecomposition:
ridges = po.gram.solve(g, ridge=[0.0, 0.01, 0.1, 1.0], standardize=True)
worst = po.gram.condition(g)["kappa"]   # Belsley's condition indexes and variance-decomposition proportions
po.gram.correlation(g)                  # the correlation matrix
po.gram.vif(g)                          # variance inflation factors
po.gram.subset(g, ["x0", "x1"])         # the Gram of some of the columns: a sub-block, not a recomputation
po.gram.merge([g, g])                   # pools the Grams of disjoint row sets into the Gram of their union
po.gram.lasso_path(g, [0.1, 0.01])      # the lasso model's coordinate descent, offline
```

### One row per finished group

`group_close` says when a group is finished, in one of two modes, and is
refused beside an `embargo` or a window target, which hold rows past a
group's end. The bank then writes the group's running sums out, a row per
half-life and per Gram, and frees the group's state. The rows wait in the
bank until `closed_groups()` reads them, or a run writes them to a file.

**It keeps a bank's memory bounded when new group values never stop
appearing.** A bank keeps one state per group key for the life of the
bank. On a stream whose keys keep arriving, such as a day id, a session id
or a block number, that is unbounded memory for state nobody will read
again.

```python
blocks = po.spec.ew_cov("cov", features=["x0", "x1"], lam=1.0,   # lam=1.0: no decay
                        group="block", group_close="monotone")   # the modes are below
by_block = df.with_columns(block=pl.int_range(pl.len()) // 100)   # a block id that rises every 100 rows

bank = po.ModelBank([blocks])
bank.fit_predict(by_block)

closed = bank.closed_groups()          # the finished blocks, oldest first
first = po.gram.from_row(closed.head(1))
corr = po.gram.correlation(first)      # everything in po.gram works on it
```

**A closed row is the `gram()` a loop would have read at that moment, bit
for bit.** It adds the span's own `rows_fed`, `rows_learned`, `clock_min`
and `clock_max`. It also carries `coef` for a model that has one, the
eigendecomposition for an `ew_cov` with `pca`, and a `marginal`'s pairs.
The spec above has one half-life and one Gram, so it gives one row per
block.

**Closed rows wait in the bank until they are read.** Reading takes them
out of the bank, and `drop=False` reads and keeps them. What has closed and
not yet been read is saved with the state, so a loop that saves between
chunks loses no rows.

**Two modes say when a group is finished:**

| `group_close` | a group is finished when | refused |
|---|---|---|
| `"monotone"` | its key is below the largest seen, since the key never goes backwards | a chunk whose keys are out of order, naming the row |
| `"session"` | its `session` value changes | |

Under `"monotone"`, an integer key column is ordered as numbers, and any
other column as text, so `"9"` comes after `"10"`. Sort by the same rule
the bank reads, or the chunk is refused. A `Categorical` column is compared
as text, as Polars sorts it. **In either mode the last group never
closes**, since nothing proves it is finished, and it stays readable
through `gram()`.

**A run can write the closed rows to a file beside its output.**
`closed_groups=` does it on `lf.online.fit_predict`, `fit_predict_batches`
or `fit`, and the command line writes the file too
([docs/RUNNER.md](docs/RUNNER.md)). `fit` keeps no per-row output, so with
`fit` the file is the run's whole product. A stream that does not fit in
memory goes in, and one row per block comes out. Here the spec and frame
built above learn, and each closed block's row goes to the file:

```python
po.ModelBank([blocks]).fit(by_block.lazy(), closed_groups="blocks.parquet")
```

### Reading a correlation matrix

`po.gram` solves and diagnoses a design matrix, and `po.corr` does the
arithmetic that comes *after* a correlation matrix. It repairs a matrix,
reads its spectrum against noise, sums it up in one number, measures
sampling error, works on blocks, changes scale and scores a forecast. It
keeps `po.gram`'s style: numpy only, pure functions, each held against the
paper it comes from.

```python
r = po.corr.matrix(closed.head(1))         # an array, a gram() dict, or a closed row -> a correlation matrix
fixed, dist, iters = po.corr.nearest(r)    # Higham (2002): the nearest correlation matrix, by alternating projections
shrunk, alpha = po.corr.shrink(r, alpha=0.2)   # Ledoit-Wolf shrinkage toward a constant-correlation (or identity) target
z = po.corr.to_z(r); back = po.corr.from_z(z)  # Fisher's transform, clipped so a degenerate +-1 is finite
rho = po.corr.equicorr(r)                  # deco's one number for the whole matrix, offline
vals, vecs = po.corr.spectral(r, 1)        # the top eigenpairs ...
po.corr.from_spectral(vals, vecs)          # ... and the completion back to a correlation matrix
po.corr.absorption(r, 1)                   # the absorption ratio: how much of the variance the top k eigenpairs carry
lo, hi = po.corr.mp_edge(n=2000, m=50)     # the Marchenko–Pastur edges: eigenvalues inside them are what pure noise gives
po.corr.fisher_se(n=2000, rho=0.3)         # the standard error of a correlation, with the AR(1) inflation and its caveats
```

The rest of the module, by kind; the [API
reference](https://hgilde.github.io/polars-online/corr.html) has each one's
arguments:

| kind | function | what it gives |
|---|---|---|
| spectrum | `mp_density` | the Marchenko–Pastur density |
| spectrum | `shift` | the standardised absorption shift, `(fast − slow) / scale` |
| one number | `equicorr_row` | the equicorrelation estimate of one standardised row, the `u` [`deco`](#deco--one-correlation-for-the-whole-matrix) computes per row |
| one number | `equicorr_loglik` | the Gaussian log-density of a standardised row under an equicorrelation matrix |
| sampling error | `signal_share` | how much of a correlation's movement between blocks is not sampling noise, at Kish's sample size |
| blocks | `block_means`, `from_blocks` | the mean correlation within and between labelled blocks, and the block matrix back from those means |
| scale | `epps_invert` | the correlation at a coarser scale, from `ew_cov`'s lagged co-moments |
| forecast | `loss` | how wrong a forecast correlation matrix was: `qlike`, `z_mse` or Engle–Colacito `minvar` |

## Diagnostics, selection and evaluation

Four kinds of tool live here. Diagnostics and a selection are switched on
in a spec, evaluation reads the output afterwards, and simulated data comes
with its truth. The diagnostics are read from what the models have already
learned. So none of them lets a row's own outcome into what
it is measured against, and all of them live in memory that does not grow
with the stream.

| kind | what it gives | subsection |
|---|---|---|
| diagnostics, switched on in a spec | residual spread, break detection and running accuracy, row by row | [Per-row diagnostics](#per-row-diagnostics) |
| | an interval that assumes no distribution | [Conformal intervals](#conformal-intervals) |
| selection, switched on in a spec | a choice among a grid's settings, as the stream runs | [Choosing among a grid's settings](#choosing-among-a-grids-settings) |
| evaluation, after the run | metrics and comparisons over a collected frame | [Evaluating an output frame](#evaluating-an-output-frame) |
| | the same metrics, from sums merged chunk by chunk | [Evaluating a stream too large to hold](#evaluating-a-stream-too-large-to-hold) |
| simulation | a stream whose regimes are known, and the truth beside it | [Data whose truth is known](#data-whose-truth-is-known) |

### Per-row diagnostics

The switches below read the residuals, so they belong to the ten [linear
models](#linear-models), which predict a target, and any other model
refuses them by name. Each adds fields to a spec's output, one per *slot*:
one prediction of one target, at one point of a grid. `emit_clocks=True`
is the one switch every model takes, since it reads no residual.

| kind | switches | fields |
|---|---|---|
| spread and surprise | `emit_sigma`, `emit_zscore`, `resid_quantiles` | `sigma_`, `zscore_`, `abs_resid_q<p>_` |
| breaks | `emit_drift` | `drift_` |
| running accuracy | `emit_metrics` | `ic_`, `r2_`, `hit_rate_` |
| residual autocorrelation | `emit_autocorr` | `autocorr_` |
| an interval | `conformal` | `lo_`, `hi_`, `coverage_`: [Conformal intervals](#conformal-intervals) |
| a choice among the slots | `emit_selected`, `emit_averaged` | one per target: [Choosing among a grid's settings](#choosing-among-a-grids-settings) |
| clocks, on every model | `emit_clocks` | `scored_clock`, the clock a row was scored at, and `learned_clock`, the clock of the last row learned, once per spec ([Labels that arrive late](#labels-that-arrive-late)) |

In the comments, *EW* means exponentially weighted. The keywords that tune
a switch sit under it, at their defaults.

```python
diag = po.spec.ewridge(
    "diag", targets=["y"], features=["x0", "x1"], clock="t", gap_cap=300.0, half_life=500.0,
    ridge=[1e-6, 0.1],           # a grid of two ridge values: two slots for the one target
    emit_sigma=True,             # sigma_<slot>:     EW standard deviation of that slot's out-of-sample residuals
    emit_zscore=True,            # zscore_<slot>:    resid / sigma -- how surprising the row was, in units of recent error
    emit_drift=True,             # drift_<slot>:     Page-Hinkley break detection on |resid| ...
    drift_delta=0.5,             #                   ... with this tolerance, in units of the slot's sigma ...
    drift_threshold=20.0,        #                   ... and this threshold, in sigma times clock units;
    drift_action="flag",         #                   "reset" also starts the model over at a break
    emit_metrics=True,           # ic_, r2_, hit_rate_<slot>: the IC (the correlation of prediction with target),
                                 #                   R² and hit rate that po.eval computes (below), EW, kept beside the model
    resid_quantiles=[0.5, 0.9],  # abs_resid_q<p>_<slot>: EW quantiles of |resid| at the half_life, within 0.78% of
                                 #                   the exact weighted quantile -- an interval that assumes no distribution
    emit_autocorr=True,          # autocorr_<slot>:  EW correlation of each residual with the one this many
    resid_autocorr_lag=1,        #                   back, never across a break; nonzero means something is missing
    conformal=0.9,               # lo_, hi_, coverage_<slot>: an interval at that coverage, assuming no
                                 #                   distribution, and the coverage it has delivered
    conformal_rate=0.05,         #                   how fast its radius moves (Conformal intervals)
)
band = po.ModelBank([diag]).fit_predict(df).unnest("diag")
```

**`sigma`, the interval and its coverage, the quantiles, the
autocorrelation and the metrics all stand as they were before the row.**
`resid`, `zscore` and `drift` measure the row against them.

**Accuracy reads differently on a ratio or a probability.** A target taken
as a ratio (`po.target(..., relative="ratio")`) is positive by
construction. So its `hit_rate` asks whether prediction and outcome fall
on the same side of 1: whether the ratio went up or down. A difference and
a log ratio are about zero, as a plain target is.

`emit_metrics` reads differently on an `sgd` or `ftrl` fit with
`loss="logistic"`, because `pred` is then a probability and `y` a 0/1
label rather than a signed target. There is no streaming log loss; a
collected frame gets one from `po.eval` ([Evaluating an output
frame](#evaluating-an-output-frame)). Under their usual names, the three
read:

| field | on a logistic `sgd` or `ftrl` fit |
|---|---|
| `hit_rate` | the accuracy at a 0.5 threshold |
| `r2` | the Brier skill score against the running base rate |
| `ic` | the point-biserial correlation between the probability and the label |

### Conformal intervals

**Use `conformal` for an interval when the residuals are not Gaussian.**
It tracks the `conformal` quantile of `|resid|` directly, so its long-run
coverage is the number you asked for, whatever the residuals do. The
Gaussian interval, `pred ± z·sigma`, over-covers by several points on
fat-tailed or heteroskedastic residuals, where this one lands on target.

**The radius moves by `conformal_rate`, 0.05 by default.** It widens on a
miss, by rate·sigma·coverage, and narrows on a hit, by rate·sigma·(1 −
coverage). Each step is scaled by the row's weight over the mean weight.

```python
ci = po.spec.ewridge("ci", targets=["y"], features=["x0", "x1"], clock="t",
                     gap_cap=300.0, half_life=500.0, conformal=0.9)
band = po.ModelBank([ci]).fit_predict(df).unnest("ci")   # lo_y, hi_y: the interval; coverage_y: the coverage delivered
held = band.select(pl.col("y").is_between(pl.col("lo_y"), pl.col("hi_y")).mean())   # the realized coverage
```

### Choosing among a grid's settings

A grid gives one prediction per target and point, a *slot*, and two
switches choose among the slots as the stream runs, by each one's
exponentially weighted out-of-sample error so far. `emit_selected` commits
to the best slot, and `emit_averaged` hedges across all of them. Comparing
whole specs after the run is `po.eval`'s job ([Evaluating an output
frame](#evaluating-an-output-frame)).

```python
pick = po.spec.ewridge(
    "pick", targets=["y"], features=["x0", "x1"], clock="t", gap_cap=300.0, half_life=500.0,
    ridge=[1e-6, 0.1],           # a grid, so there is something to select among
    emit_selected=True,          # selected_y, pred_y__selected
    emit_averaged=True,          # pred_y__averaged
    average_eta=1.0,             # the default
)
chosen = po.ModelBank([pick]).fit_predict(df).unnest("pick")
chosen.select("selected_y", "pred_y__selected", "pred_y__averaged")
```

| switch | fields, one per target | what it gives |
|---|---|---|
| `emit_selected` | `selected_<target>`, `pred_<target>__selected` | the ridge value, feature set or half_life with the lowest EW out-of-sample error so far, each slot at its own half_life |
| `emit_averaged` | `pred_<target>__averaged` | every slot's prediction, weighted by `exp(-eta * (mse / best_mse - 1))`, with `mse` the slot's EW squared error and `best_mse` the best slot's; `average_eta` is `eta`, and `inf` gives `emit_selected`'s choice |

### Evaluating an output frame

`po.eval` reads an output frame after the run, in three ways: it scores a
spec, compares specs, and unpacks the frame to long form.
`po.eval.metrics` tests signs about zero, so hand it a target taken as a
ratio as the ratio less 1. On a 0/1 label,
`po.eval.metrics(..., binary=True)` adds the log loss that no streaming
metric gives.

The block builds `out`, the output of two specs over `df`, both grouped by
`stock_id`, and reads it each way. `seqtest` runs the test the
[`seqtest`](#seqtest--a-sequential-test-of-a-sign-by-betting) model runs
inside a bank:

```python
ridge = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], clock="t",
                        gap_cap=300.0, half_life=500.0, group="stock_id")
kalman = po.spec.kalman("kalman", targets=["y"], features=["x0", "x1"], clock="t",
                        gap_cap=300.0, half_life=500.0, group="stock_id",
                        coef_half_life=100.0)   # how fast a coefficient may drift, as a half-life on the clock
out = df.online.fit_predict([ridge, kalman])    # df's columns, plus one column per spec

po.eval.metrics(out, "ridge", by=["stock_id"])                       # R², IC, hit rate, MSE
po.eval.rolling_metrics(out, "ridge", clock="t", window_size=3600.0)  # the same, per clock window
po.eval.compare_specs(out, ["ridge", "kalman"])                     # one table, many specs: which had the lower error
po.eval.seqtest(out, a="kalman", b="ridge", by=["stock_id"])        # is kalman closer? evidence per row
po.eval.unpack(out, "ridge")                                        # long form: one row per (row, slot), with slot,
                                                                    # target, pred and y, for your own group_by
```

### Evaluating a stream too large to hold

The five `po.eval` calls above need the whole frame. When the output is
never held in one place, say fifty slots over a billion rows,
`po.eval.sums` reduces each chunk to ten numbers per key instead.
`po.eval.merge_sums` adds two such sets exactly, whatever the split, and
`po.eval.from_sums` turns them into the metrics.

**The sums are centred.** They hold weighted means and centred second
moments, merged with a parallel-axis term, rather than raw `Σy` and `Σy²`.
A target sitting around 1e8 with unit spread destroys the raw form's
variance entirely, and the centred form keeps it.

```python
ridge = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"],
                        clock="t", gap_cap=300.0, half_life=500.0, group="stock_id")

running = None
for out in po.ModelBank([ridge]).fit_predict_batches(lf, chunk_rows=100):   # the output, 100 rows at a time
    part = po.eval.sums(out, "ridge", by=["stock_id"])   # ten numbers per key; weight= names a column to weight the rows by
    running = part if running is None else po.eval.merge_sums(running, part)   # exact, whatever the split

po.eval.from_sums(running, min_obs=10)   # R², IC, hit rate and MSE, as metrics() gives them, and the RMSE
```

### Data whose truth is known

A model that claims to find a changing correlation structure is measured
against data whose truth is known.
[`po.sim.regimes`](https://hgilde.github.io/polars-online/sim.html#polars_online.sim.regimes)
generates such a stream from one seed, byte-identical when the seed
repeats, and hands back the truth beside it. What `hmm`, `corrchange` and
`bocpd` find on such streams, and what they miss, is measured in
[docs/REGIMES.md](docs/REGIMES.md).

```python
sim = po.sim.regimes(
    4, states=[0.2, 0.7],                     # four series; two regimes, at these equicorrelations
    transition=[[0.98, 0.02], [0.02, 0.98]],  # how the regimes switch
    n_blocks=8, rows_per_block=500,
    durations=None,                           # or a block count per state: each state lasts exactly as long as it says
    design="step", smooth_rows=0,             # "smooth" with smooth_rows above 0 interpolates the matrix across a
                                              # boundary instead of stepping
    phi=0.3, noise=0.01,                      # returns correlated with their own past; observation noise
    async_rates=[1.0, 1.0, 0.4, 0.4],         # two series observed less often: a row where one has no observation
                                              # is null there
    seed=0,                                   # the same seed twice is byte-identical
)
rows, truth_rows, truth_blocks = sim["rows"], sim["truth_rows"], sim["truth_blocks"]
```

| frame | what it holds |
|---|---|
| `rows` | what a consumer sees: an entity, the row's index `t`, a clock and a session, levels `x_1` .. `x_m` (so [`refresh_time`](#series-that-tick-at-their-own-times) then `.diff()` apply), and `activity`, null unless `activity=(mean, shape)` is given |
| `truth_rows` | per row `t`: the block, state, volatility multiplier and interpolation fraction |
| `truth_blocks` | each block's true correlation matrix, as the upper triangle in a list |

## Models

There are twenty-one models, in four families. [Linear models](#linear-models)
regress a target on features, and [moments and
correlation](#moments-and-correlation) track the running moments and
correlations of the columns you name. [Clustering and
classification](#clustering-and-classification) put labels on rows, and
[sequential tests and regimes](#sequential-tests-and-regimes) give evidence
that something holds or has changed, and which regime the stream is in.

Each row of the table links to the model's builder in the API reference,
which lists every keyword with its default. It also links to the model's
section below, which gives what the model is for, its update rule, the
parameters that are its own, and what it writes.
[docs/OUTPUTS.md](docs/OUTPUTS.md) lists every field of every model, and the
*learns by* column is explained after the table.

| model | learns by | what it is |
|---|---|---|
| **[Linear models](#linear-models)** | | |
| [`ewridge`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.ewridge) · [math](#ewridge--ew-ridge-on-sufficient-statistics) | solve | exponentially weighted ridge regression, the workhorse; several ridge values and feature sets are solved from the same running sums at almost no extra work |
| [`rls`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.rls) · [math](#rls--recursive-least-squares) | solve | recursive least squares, in the numerically safe square-root form |
| [`lasso`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.lasso) · [math](#lasso--lasso-path-its-penalty-chosen-as-it-runs) | solve | lasso and elastic-net path, with the penalty chosen as the stream runs |
| [`kalman`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.kalman) · [math](#kalman--random-walk-β-dynamic-linear-model) | filter | a Kalman filter whose coefficients drift as random walks |
| [`huber`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.huber) · [`quantile`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.quantile) · [math](#huber--quantile--robust-regression) | reweight | robust and quantile regression |
| [`sgd`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.sgd) · [math](#sgd--stochastic-gradient-descent) | step | stochastic gradient descent with squared, Huber, quantile, ε-insensitive, Poisson and logistic losses |
| [`pa`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.pa) · [math](#pa--passive-aggressive-regression) | step | passive-aggressive regression, with no learning rate to tune |
| [`ftrl`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.ftrl) · [math](#ftrl--online-logistic-regression) | step | FTRL-proximal regression, logistic by default, with an L1 penalty that zeroes coefficients |
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
| [`corrchange`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.corrchange) · [math](#corrchange--has-the-correlation-structure-changed) | test | has the correlation structure changed: the Wied–Krämer–Dehling constancy test span by span, the Wied–Galeano detector row by row against a stable history, or the size of a change between two windows against a permutation null |
| [`bocpd`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.bocpd) · [math](#bocpd--how-long-has-this-regime-lasted) | filter | how long this regime has lasted: Adams & MacKay's run-length posterior, so the answer is the regime's age and not a flag |
| [`hmm`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.hmm) · [math](#hmm--which-regime-are-we-in) | filter | a Gaussian hidden Markov model, filtered as the stream runs: `ew_class` without the labels, with a transition matrix that can be learned |

**Every model reads the stream the same way.** A spec's clock, grouping and
warm-up mean the same thing whichever model it names. Its half-life is
always a half-life on the clock, though what decays differs by model ([How a
bank sees a stream](#how-a-bank-sees-a-stream)).

*Learns by* says how a model takes in a row, which decides what it does
when forgetting is off. Three exceptions read the rows in sequence even
among the models that solve or accumulate, and the last column names them:

| learns by | what the model does with a row | with decay off | except |
|---|---|---|---|
| **solve** | keeps running sums and computes its coefficients from them | converges to the batch answer, in any row order | `lasso`'s `penalty_selected`, ranked by out-of-sample error, though its path converges in any order |
| **accumulate** | keeps running sums and reports them | converges to the batch answer, in any row order | a lag, in `ew_cov` or `marginal`, which counts learned rows; `rcov`'s block and `deco`'s per-row estimate |
| **reweight** | solves from running sums, but lets the fit before each row decide how that row enters them | depends on the row order | |
| **step** | moves its coefficients a little on each row | depends on the row order | |
| **filter** | carries a belief forward from row to row | depends on the row order | |
| **test** | counts evidence as it goes | depends on the row order | |

**In the update rules,** `z` is `[1, x]` when there is an intercept, `w` is
the row's weight, `λ` the row's decay, and `W` the total weight so far. The
running sums a model keeps are its *accumulators*:

| accumulator | where | why |
|---|---|---|
| an exponentially weighted **mean** | most models | it stays bounded over a stream of any length |
| a decayed **sum** | `rls`'s matrix, and the gradient sums of `ftrl` and of `sgd`'s adagrad | |
| a **centred** second moment, by a weighted Welford update | wherever a model keeps second moments as means | a variance is right even when a feature sits far from zero |

### Linear models

Ten models predict a numeric target. Nine regress it on features, and
`holt`, a baseline, uses none. They differ in how a row moves the fit:

| a row | models |
|---|---|
| updates running sums, and the coefficients are computed from them | [`ewridge`](#ewridge--ew-ridge-on-sufficient-statistics), [`rls`](#rls--recursive-least-squares), [`lasso`](#lasso--lasso-path-its-penalty-chosen-as-it-runs) |
| updates a filtered belief about coefficients that drift | [`kalman`](#kalman--random-walk-β-dynamic-linear-model) |
| enters the running sums as its residual against the fit before it decides | [`huber` and `quantile`](#huber--quantile--robust-regression) |
| moves the coefficients one step | [`sgd`](#sgd--stochastic-gradient-descent), [`pa`](#pa--passive-aggressive-regression), [`ftrl`](#ftrl--online-logistic-regression) |
| updates the target's own level and trend | [`holt`](#holt--holts-linear-trend) |

Some parameters are shared by several of these models, and each is
explained once:

| parameters | taken by | explained under |
|---|---|---|
| `solve_every`, `max_rows_between_solves` | `ewridge`, `lasso`, `huber`, `quantile` | [`ewridge`](#ewridge--ew-ridge-on-sufficient-statistics) |
| `target_gaps` | `ewridge`, `lasso` | [`ewridge`](#ewridge--ew-ridge-on-sufficient-statistics) |
| `window_size` | `ewridge`, `lasso` | [A hard window](#a-hard-window) |
| `standardize` | `ewridge`, `huber`, `quantile`, `sgd`, and `kalman`, where it is the default | `ewridge` and `kalman` |
| `coef_min`, `coef_max`, `coef_sum` | `sgd`, `pa` | [`sgd`](#sgd--stochastic-gradient-descent) |

What `half_life` decays differs by model:

| model | `half_life` decays |
|---|---|
| `ewridge` | the running sums the fit is solved from |
| `lasso` | the same sums, and the error that picks the penalty unless `select_half_life` is given |
| `rls` | `A` and `b`, as decayed sums rather than means |
| `kalman` | the observation-noise estimate and the features' standardization; `coef_half_life` sets how fast a coefficient may drift |
| `huber`, `quantile` | their running sums, kept per target |
| `sgd` | `weight_sum`, the running moments `standardize` scales by, and adagrad's sums, but not the coefficients |
| `pa` | `weight_sum` and each target's own weight, but not the coefficients |
| `ftrl` | its per-coordinate sums, and its penalties with them |
| `holt` | the level, as `level_half_life`; the trend has `trend_half_life` |

Several of them also read a row's weight differently from a weighted mean,
and [Weights](#weights) compares them.

#### `ewridge` — EW ridge on sufficient statistics

*API:* [`po.spec.ewridge`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.ewridge) — *Rust:* [`ewridge.rs`](crates/online-core/src/ewridge.rs) — *Outputs:* [fields](docs/OUTPUTS.md#ewridge)

The workhorse, and the model to reach for first: ridge regression on
running sums. With decay off and `ridge=0` it is ordinary least squares
over every row seen, in any row order. Several ridge values and feature
sets are fitted at once, from the same sums.

Each target's sums are updated on the rows where it is present, and the
coefficients are solved from them on a schedule:

```
W'   = λW + w                       weight_sum, over every row
W_j' = λW_j + w                     on the rows where y_j is present, λW_j on the others
S_j' = (λW_j·S_j + w·z zᵀ) / W_j'   r_j' = (λW_j·r_j + w·z·y_j) / W_j'
solve:  (S_j + ridge·D) β_j = r_j + ridge·D·β₀     D = I with a 0 at the intercept, β₀ = coef_prior (0 by default)
```

```python
rr = po.spec.ewridge(
    "rr", targets=["y"], features=["x0", "x1", "x2"], clock="t", gap_cap=300.0, half_life=600.0,
    ridge=[1e-6, 0.1],             # a grid of two ridge values, solved from one set of sums
    feature_sets={"mkt": ["x0"], "all": ["x0", "x1", "x2"]},   # named feature subsets, likewise
    solve_every=12.0,              # solve every 12 clock units ...
    max_rows_between_solves=100,   # ... and at least every 100 rows, whatever the clock does
    standardize=True,              # solve on the correlation matrix, and undo it afterwards
    ridge_scale="mean",            # the default; "sum" makes the ridge a warm start that fades
    coef_prior=None,               # shrink toward a stated belief instead of toward zero
    target_gaps="own_rows",        # the default: a target null on some rows is fitted on its own rows
)
```

Its own parameters fall into six groups, each developed below. Three of the
groups refuse some combinations:

| group | parameters | refused or required |
|---|---|---|
| the grid | `ridge` as a list, `feature_sets` | |
| when it solves | `solve_every`, `max_rows_between_solves` | |
| how the ridge applies | `standardize`, `ridge_scale`, `coef_prior` | `ridge_scale="sum"` is refused beside `standardize`, a ridge or feature-set grid, and a window |
| a target null on some rows | `target_gaps` | |
| what the sums remember | `window_size`, `session_shrink` | `session_shrink` needs `session` and `long_half_life`, and is refused beside `session_gap="reset"` and `window_size` |
| a wide fit | `gram_block_rows` | refused with `window_size=`, and where a solve happens every row |

**A grid adds solves, not updates.** A list of `ridge` values, or a dict of
named `feature_sets`, is a *grid*. Every value is solved from the same
sums, so a grid adds one solve per value at each scheduled solve, and no
update. Only the sets named are fitted, so name the full set to fit it too.

**Without `solve_every`, it solves by weight.** `half_life=inf` solves every
row, and so does `lam=`. A finite half-life solves once the weight learned
since the last solve reaches `ln 2 / 50` of the weight the fit holds, about
1.4 %. So a very long half-life keeps solving as the stream grows. In steady
state the schedule comes to a solve every `half_life/50` of clock. On 6M
rows × 20 features from a parquet stream, `solve_every=1000` takes 1.5 s
instead of 13 s, with the coefficients at most 1000 rows out of date.

**Each solve is a Cholesky factorization.** Updating `S` takes O(k²) time
per row for `k` features. A near-singular system is retried with a small
jitter on the diagonal, and `bank.solve_failures()` counts each retry.

**`ridge_scale` says what the ridge is measured against.** `S` is a mean,
so under `"mean"`, the default, a plain ridge is a penalty per observation
that never fades. `"sum"` makes it a warm start that fades as evidence
arrives, as `rls`'s does, for example to start at yesterday's fit. It then
penalizes the intercept too, and reads `coef_prior`'s intercept entry.

**`standardize=True` solves on the correlation matrix and undoes it
afterwards.** A feature with zero variance is then dropped from the solve
rather than allowed to blow it up.

**`target_gaps` says which rows a target's `S_j` covers when the target is
null on some of them.** Under `"own_rows"`, the default, it covers exactly
the rows the target is present on, so the target's fit is the fit of the
frame with those nulls dropped. Targets present on the same rows share one
`S`. A target that goes missing on a row where the others are present takes
a copy of the shared `S` and keeps its own from then on. So the targets of
one spec hold one `k × k` matrix per pattern of missing rows.

`"pairwise"` keeps one `S` over every row, the way pandas' `DataFrame.cov`
takes a pairwise-complete covariance, and centres each target's `r_j` at its
own rows' means. The two agree when the gaps have nothing to do with the
features. Where they do, as with a target present only on trade rows
between market-data rows, `"pairwise"` scales each slope by the feature's
variance on the target's rows over its variance on every row.

**Two settings change what the sums remember.** `window_size=` cuts the
history off at a fixed age ([A hard window](#a-hard-window)).

**`session_shrink=f` starts each new session from a blend rather than from
a few rows.** With `long_half_life=`, it keeps a twin of the sums that
forgets more slowly. At each session change the moments then take `1 − f` of
today's data and `f` of the long run's, at today's weight, Kish's effective
sample size and prior scale.

**`gram_block_rows` speeds a wide fit.** At a thousand features, the O(k²)
update of `S` takes most of the time. `gram_block_rows=256` holds 256 rows
back and adds them to `S` with one matrix product, instead of 256
single-row updates ([docs/PERFORMANCE.md](docs/PERFORMANCE.md) §18):

```python
wide = po.spec.ewridge(
    "wide", targets=["y"],
    features=["x0", "x1", "x2"],   # three here, so the block runs; the table below is at 256 to 2,000
    half_life=float("inf"), solve_every=1e9, max_rows_between_solves=1000,
    gram_block_rows=256,           # add 256 held rows to S in one matrix product
)
```

| features | rows per second, against one row at a time, on one thread |
|---:|---:|
| 256 | 5.1× |
| 1,000 | 6.6× |
| 2,000 | 5.9× |

A solve every 512 rows brings each of those down to about 4×, because the
solve takes the same time either way. Blocking leaves `weight_sum`, the
timing of every prediction and chunk invariance unchanged. The held rows
travel in the state file, so a save mid-block resumes on the same block.
The coefficients agree with the row-by-row fit to rounding, since the
blocked sum is the same sum in a different order.

**With decay off and `ridge=0`, the coefficients match `numpy.linalg.lstsq`
to 2e-13,** fed forwards or backwards. On 6M rows × 20 features from a
parquet stream, memory peaks at 1.4 GB, against 3.97 GB for `lstsq` on the
same rows. Both `target_gaps` modes are held to independent libraries in
`tests/test_second_opinion.py`.

#### `rls` — recursive least squares

*API:* [`po.spec.rls`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.rls) — *Rust:* [`rls.rs`](crates/online-core/src/rls.rs) — *Outputs:* [fields](docs/OUTPUTS.md#rls)

The coefficients move on every row. There is no solve schedule, so they
are never out of date. The fit equals `ewridge(ridge_scale="sum")` solved
on every row, to better than 1e-9.

`A` and `b_j` are the decayed sums of `zzᵀ` and `y_j z`, started from the
ridge:

```
A ← λA + w zzᵀ       b_j ← λb_j + w y_j z        β_j = A⁻¹ b_j        decayed sums, not means
A₀ = ridge·I         b₀ = ridge·coef_prior
```

A row's weight is on the sum scale, so a heavier stream outweighs the
starting ridge sooner ([Weights](#weights)). Unlike `ewridge`'s default,
the starting ridge penalizes the intercept too. A row with any null target
is scored, but no target learns from it, because the targets share one
Cholesky factor of `A`.

```python
rls = po.spec.rls(
    "rls", targets=["y"], features=["x0", "x1"], clock="t", gap_cap=300.0, half_life=600.0,
    ridge=1e-3,                    # A starts at ridge * I (default 1)
)
```

**The model stores the Cholesky factor of `A`,** updated row by row with
Givens rotations: the *square-root form* of the recursion. It takes O(k²)
time per row, the same as the textbook recursion on the inverse `P`, and
avoids both of that form's failures. `P` loses symmetry to rounding by a
factor of `1/λ` per row. And one extreme row can cancel `P` and freeze a
coefficient for good.

#### `lasso` — lasso path, its penalty chosen as it runs

*API:* [`po.spec.lasso`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.lasso) — *Rust:* [`lasso.rs`](crates/online-core/src/lasso.rs) — *Outputs:* [fields](docs/OUTPUTS.md#lasso)

A sparse regression: the lasso, or an elastic net, over a path of
penalties, with the penalty chosen as the stream runs by each point's
out-of-sample error. It reads the running sums `ewridge` keeps, centred the
same way, so the path loses no precision when a feature or a target sits
far from zero. `target_gaps` means what it means for `ewridge`.

It runs coordinate descent on the standardized running sums, started from
the previous solution both along the path of penalties and from one solve
to the next. Below, `C` is the features' correlation matrix and `c_i` is
feature `i`'s covariance with the target over the feature's standard
deviation. For each penalty `l` in `lasso_path`, the sweep repeats until no
coefficient moves by more than `tol`:

```
ρ_i = c_i − Σ_{j≠i} C_ij β_j
β_i = soft(ρ_i, l·l1_ratio) / (C_ii + l·(1 − l1_ratio))        soft(v, t) = sign(v)·max(|v| − t, 0)
```

The slopes are then unscaled, and the intercept is `ȳ − m·β`, with `m` the
features' means. The target is centred but not scaled, so the L1 threshold
`l·l1_ratio` is in the target's units. A descent that runs out of sweeps,
`max_iter` of them, is counted in `bank.solve_failures()`.

```python
las = po.spec.lasso(
    "las", targets=["y"], features=["x0", "x1", "x2"], clock="t", gap_cap=300.0, half_life=600.0,
    lasso_path=[0.1, 0.01, 0.001],   # the penalties, decreasing
    select_half_life=None,           # the half_life of the error that ranks them (default: the model's)
    l1_ratio=1.0,                    # below 1: an elastic net
    max_iter=100, tol=1e-10,         # the defaults
    target_gaps="own_rows",          # which rows a target null on some is fitted from, as for ewridge
)
```

**`penalty_selected_<target>` is the penalty with the lowest exponentially
weighted out-of-sample squared error so far,** as it stood before the row.
Predictions for every penalty are computed anyway, so the selection adds no
work of its own. `select_half_life` is the half-life of that error, by
default the model's; `inf` ranks on the plain mean.

**With decay off, every point of the path converges to its batch fit in any
row order.** `penalty_selected` does not, because it ranks the points by
out-of-sample error, which depends on the order.

**Under a `window_size`, the sums are the window's,** so a feature with no
spread inside the window has no evidence there and goes to exactly zero.
The error that picks the penalty is the window's too ([A hard
window](#a-hard-window)).

The path is checked against the KKT conditions of its objective, and a
penalized path point against statsmodels' elastic net on the target's own
rows.

#### `kalman` — random-walk-β dynamic linear model

*API:* [`po.spec.kalman`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.kalman) — *Rust:* [`kalman.rs`](crates/online-core/src/kalman.rs) — *Outputs:* [fields](docs/OUTPUTS.md#kalman)

A regression whose coefficients are allowed to drift, tracked by a Kalman
filter. Each coefficient is a random walk by default, or, under
`revert_half_life`, is pulled toward zero on every row while the rows that
support it pull it back.

Per target `j`, `β_j` are the coefficients and `P_j` their covariance.
`r_i` is coefficient `i`'s `revert_half_life`, and `φ_i = 2^(−Δclock/r_i)`
its entry of `Φ`. `Q` is the process noise, which `coef_half_life` sets, and
`R_j` the observation noise, the target's exponentially weighted residual
variance unless `obs_var` is given:

```
β_j ← Φβ_j    P_j ← ΦP_jΦ + Q·Δclock²   Φ = diag(2^(−Δclock/r_i))
s   = zᵀP_j z + R_j/w                   K   = P_j z / s
β_j ← β_j + K(y_j − zᵀβ_j)              P_j ← P_j − K zᵀP_j
```

A row's weight scales its observation's precision: the rule divides `R_j`
by `w`, so with a fixed `obs_var=` the observation's variance is
`obs_var / w`.

```python
revert = po.spec.kalman(
    "k", targets=["y"], features=["signal_a", "signal_b"], clock="t", gap_cap=10.0,
    half_life=200.0,                   # how fast the noise estimate and the standardization forget
    coef_half_life=100.0,              # required: how fast a coefficient may drift
    revert_half_life=[float("inf"), 50.0, 50.0],   # inf leaves the intercept alone; the slopes revert at 50
    standardize=True,                  # the default
    p0=1.0,                            # the default: P starts at p0 * I
    share_p=False,                     # the default: one P per target; True keeps one for all, driven by their mean σ²
)
out = po.ModelBank([revert]).fit_predict(df)
```

**By default a coefficient is a random walk** and keeps its last value.
`coef_half_life` says how fast it may drift, on standardized features: a
row Δ clock units after the last adds `σ²(ln2 · Δ / h_i)²`, matching
EW-RLS's steady state at any spacing. It takes one number, or one per
coefficient, the intercept first, and `inf` pins a coefficient. `q=` gives
`q_i` outright instead, added as `q_i · Δ²` too.

**`revert_half_life` makes a coefficient mean-reverting.** Each slope is
pulled toward zero on every row, and the rows that support it pull it back:
a mean-reverting (AR(1)) prior. In the example the slopes revert with a
50-unit half-life, and `inf` for the intercept leaves it alone. A reverting
coefficient's long-run prior variance, at rows `Δclock` apart, is
`q_i·Δclock²/(1−φ_i²)`, where a random walk's grows without bound.

**Reverting helps a regressor that is only occasionally active.** It is
forgotten between its bursts instead of held at its last value, and a stale
effect cannot persist through a run of null targets. The pull acts on every
row, so a persistent effect settles below its true size, the more so the
shorter the reversion half-life.

**Under `standardize`, the default, the reversion acts in the standardized
coordinates,** so zero means "no effect" for a slope and "the target
averages zero" for the intercept.

**`predict` moves the coefficients too,** by the same `Φ` over the distance
from the last learned row, capped by `gap_cap`. So a slope keeps at least
`2^(−gap_cap/r_i)` of its value. Only where `gap_cap` spans several
reversion half-lives does a far prediction approach the intercept, which
under `standardize` is the target's level at the features' means.

With `standardize=False`, `q=0` and a fixed `obs_var=`, the filter is
exactly Bayesian linear regression: river's `BayesianLinearRegression`
agrees to 1e-13.

#### `huber` / `quantile` — robust regression

*API:* [`po.spec.huber`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.huber) and [`po.spec.quantile`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.quantile) — *Rust:* [`robust.rs`](crates/online-core/src/robust.rs) — *Outputs:* [huber](docs/OUTPUTS.md#huber), [quantile](docs/OUTPUTS.md#quantile)

Two regressions that do not let one wild target move the fit. `huber`
down-weights a residual beyond `huber_delta` times the residuals' standard
deviation, and `quantile` fits a conditional quantile, a median regression
at `quantile=0.5`. Both read each row's residual against the fit *before*
the row is learned, so both stay out-of-sample.

Both keep their running sums per target, because their weights are per
target. Both take `ewridge`'s `ridge=` (one value), `standardize=`,
`solve_every=` and `max_rows_between_solves=`, meaning the same.

Below, `r` is the row's residual, `σ` the plain exponentially weighted
standard deviation of the residuals, taken as 1 until one exists, `δ` is
`huber_delta` and `τ` the `quantile` level. The *Gram* is `ewridge`'s `S`,
and the cross-moment its `r_j`:

```
huber:     the ridge update at weight  w · min(1, δσ / |r|)
quantile:  one Newton step on the check loss, smoothed by a uniform kernel of half-width h = quantile_eps · σ
           |r| < h:  a least-squares row with target  y + 2h(τ − ½)
           |r| ≥ h:  adds  w · 2h · ψ_τ(r) · z  to the cross-moment and nothing to the Gram,  ψ_τ(r) = τ − 1{r < 0},
                     bounded so the fit at the row moves by at most the row's residual
           h is never narrower than (k/n)^{2/5} · σ, for the target's effective sample n
```

**Against batch fits of the same objectives, scikit-learn's `HuberRegressor`
and statsmodels' `QuantReg`, they land close but not equal,** for three
reasons:

| reason | applies to |
|---|---|
| each row's weight was set by the fit before it | both |
| `σ` is the plain spread rather than a robust one | `huber` |
| the check loss is smoothed over the band | `quantile` |

```python
hub = po.spec.huber("hub", targets=["y"], features=["x0", "x1"], clock="t", gap_cap=300.0, half_life=600.0,
                    huber_delta=1.5)      # a residual beyond huber_delta * sigma is down-weighted
med = po.spec.quantile("med", targets=["y"], features=["x0", "x1"], clock="t", gap_cap=300.0, half_life=600.0,
                       quantile=0.5,      # the level, strictly between 0 and 1: 0.5 is a median regression
                       quantile_eps=0.2)  # the band's half-width, in units of sigma (default 0.2)
```

**The band's rows, those with `|r| < h`, give the Newton step its
curvature.** The floor on `h`, the rule's last line, keeps a short
half-life from leaving too few of them inside the band.

`quantile` takes least-squares rows in three cases:

| case | `quantile` takes least-squares rows |
|---|---|
| a target's first rows | until the target has three rows per coefficient, its present rows counted one each, decayed, whatever their weights |
| after a gap or a reset | by the same rule, which rebuilds the fit |
| a band holding less than one row per coefficient | until the band holds rows again, which rebuilds a fit that a row near the input bound, `1e100`, has moved |

Both models are held to numpy references of their own recursions within
1e-14.

#### `sgd` — stochastic gradient descent

*API:* [`po.spec.sgd`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.sgd) — *Rust:* [`sgd.rs`](crates/online-core/src/sgd.rs) — *Outputs:* [fields](docs/OUTPUTS.md#sgd)

One gradient step per row and no solves, in O(k) time per row: the cheap
baseline, and the only model here that takes count targets
(`loss="poisson"`). It takes six losses, three learning-rate schedules, and
bounds on the slopes, such as a long-only, fully invested portfolio.

In the rule, `Gᵢ` is adagrad's running sum of squared gradients:

```
eta = zᵀβ        p = link(eta)        d = dL/d eta
g₀ = d·w         gᵢ = d·zᵢ·w + l2·βᵢ     each clipped to ±clip_gradient; the intercept takes no l2
βᵢ -= lrᵢ·gᵢ     lrᵢ = learning_rate                           schedule="constant"
                     = learning_rate/(1 + weight_sum)^power    schedule="inv_scaling"
                     = learning_rate/(√Gᵢ + 1e-8)              schedule="adagrad"
```

**The half-life does not decay the coefficients.** Every row's step moves
them, so under a constant rate their memory is in rows, about
`1 / (learning_rate · E[z²])` of them, whatever the clock between rows. The
half-life reaches `weight_sum`, the running moments `standardize` scales by,
and adagrad's sums. Adagrad's `Gᵢ` decays on the clock, so an adapted rate
opens up again after a long gap. A constant rate fits the same slopes at a
half-life of 10 or 10,000.

The loss sets the link and the gradient. `poisson` takes count targets and
`logistic` 0/1 targets. `huber_delta=` and `eps=` set the `huber` and
`epsilon_insensitive` constants, in the target's units, and `quantile=` is
the level, between 0 and 1. In the table, `delta` is `huber_delta` and `τ`
the level:

| loss | link | `dL/d eta` |
|---|---|---|
| `squared` | identity | `p − y` |
| `huber` | identity | `clamp(p − y, ±delta)` |
| `quantile` | identity | `1{y < p} − τ` |
| `epsilon_insensitive` | identity | 0 inside the tube, else `sign(p − y)` |
| `poisson` | log | `p − y` |
| `logistic` | sigmoid | `p − y` |

**`clip_gradient` guards the log link.** Each gradient is clipped to
`±clip_gradient`, `1e3` by default, since with a log link one large count
would make the next gradient exponentially bigger. It does not bind at
ordinary scales for the other losses.

```python
portfolio = po.spec.sgd(
    "portfolio", targets=["y"], features=["signal_a", "signal_b", "x0"], half_life=200.0,
    loss="squared",              # any loss in the table above
    learning_rate=0.01,
    schedule="constant",         # or "inv_scaling", or "adagrad"
    clip_gradient=1e3,           # the default
    coef_min=0.0,                # every slope at least 0 ...
    coef_sum=1.0,                # ... and the slopes summing to 1
    coef_every=1,
)
fit = po.ModelBank([portfolio]).fit_predict(df)
last = fit["portfolio"].struct.field("coef").drop_nulls()[-1]
assert min(last[1:]) >= 0.0 and abs(sum(last[1:]) - 1.0) < 1e-12   # the fit starts from the projected zero: uniform weights
```

**`coef_min`, `coef_max` and `coef_sum` constrain the slopes.** `coef_min`
bounds each slope from below, as one number or one per feature, with `-inf`
for none. `coef_max` bounds it from above, and `coef_sum` fixes their total.
After every step the slopes are moved to the nearest point that satisfies
all bounds (the Euclidean projection). The intercept is never constrained,
and `coef` reports what the projection returned, in the caller's units even
under `standardize=True`:

| setting | what it expresses |
|---|---|
| `coef_min=0, coef_sum=1` | a long-only, fully invested portfolio |
| `coef_min=0` alone | a sign the model must respect |
| `coef_min` equal to `coef_max` | a slope pinned at a known value |

#### `pa` — passive-aggressive regression

*API:* [`po.spec.pa`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.pa) — *Rust:* [`pa.rs`](crates/online-core/src/pa.rs) — *Outputs:* [fields](docs/OUTPUTS.md#pa)

Each row asks the fit to come within `eps` of its target, and the update is
the smallest change that does so. So there is no learning rate to tune.
`mode` picks one of three steps.

```
loss = max(0, |y − p| − eps)      s = ‖z‖²
pa    τ = loss / s          pa1  τ = min(c, loss/s)      pa2  τ = loss / (s + 1/(2c))
β    += min(w, 1) · τ · sign(y − p) · z
```

A row weight below 1 scales the step, and a weight above 1 counts as 1: the
rule's `min(w, 1)`. The three modes:

| `mode` | the step |
|---|---|
| `"pa"` | moves the fit as far as one bad row demands |
| `"pa1"`, the default | is capped at `c` |
| `"pa2"` | is damped by `c` instead of capped |

```python
pa = po.spec.pa(
    "pa", targets=["y"], features=["x0", "x1"], half_life=200.0,
    mode="pa1",                  # the default
    c=0.1,
    eps=0.05,                    # inside this margin a row is "close enough", and nothing moves
)
```

**PA keeps no running sums, so its coefficients have no half-life.** The
clock drives only the weights: `weight_sum`, and each target's own, which
its `min_weight` reads.

**Under bounds, keep `c` small.** `pa` takes the same `coef_min`,
`coef_max` and `coef_sum` as [`sgd`](#sgd--stochastic-gradient-descent),
with the projection applied after each update. The step then no longer
meets the row's margin exactly, and a truth outside the allowed set is never
reached. Each row moves the fit only as far as `c` allows, and the
projection takes the rest back.

#### `ftrl` — online logistic regression

*API:* [`po.spec.ftrl`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.ftrl) — *Rust:* [`ftrl.rs`](crates/online-core/src/ftrl.rs) — *Outputs:* [fields](docs/OUTPUTS.md#ftrl)

Reach for it on a 0/1 target, or for a sparse fit with no solves.
FTRL-proximal (McMahan et al. 2013) is a gradient method whose L1 penalty
zeroes a coefficient with too little evidence, so the fit is sparse, and
whose per-coordinate rates adapt to each feature's history. Its two losses
are those two uses:

| `loss` | `pred` | the model |
|---|---|---|
| `"logistic"`, the default | a probability, with `resid = y - p` | logistic regression |
| `"squared"` | the linear prediction | a sparse linear regression with no solves and an L1 penalty |

By default a target outside `[0, 1]` is clamped into it.
`strict_binary=True` refuses the chunk instead, naming the row.

**Its penalties are a prior of fixed mass,** against evidence that grows
with the weights and the rows' density. So a heavier or denser stream
overcomes them sooner. For a penalty on the mean scale, use
[`lasso`](#lasso--lasso-path-its-penalty-chosen-as-it-runs).

Its sums decay on the model's clock. With `b` the coefficients, `zz`, `n`
and `d` the per-coordinate sums, and `α` and `β` the `alpha` and `beta`
below:

```
zz_i ← λzz_i      n_i ← λn_i      d_i ← λd_i
b_i  = 0 if |zz_i| ≤ l1·m else −(zz_i − sign(zz_i)·l1·m) / (β/α·m + d_i + l2·m)
p    = sigmoid(zᵀb)      g_i = (p − y)·z_i·w      s_i = (√(n_i + g_i²) − √n_i)/α
zz_i += g_i − s_i·b_i      n_i += g_i²      d_i += s_i
m    = W / W*, per target: W its weight, W* its weight on a clock that runs only on the rows that teach it
```

```python
click = po.spec.ftrl(
    "click", targets=["y"], features=["x0", "x1"], half_life=500.0,
    loss="logistic",             # the default
    alpha=0.1, beta=1.0,         # the learning-rate scale and its smoothing (the defaults)
    l1=0.01, l2=0.0,             # l1 zeroes a coefficient whose evidence is below it (defaults: l1=0, l2=1)
    strict_binary=False,         # the default: clamp a target outside [0, 1] into it
)
```

**A row that teaches a target nothing leaves its fit where it was.** Such a
row, absent or at weight 0, ages the sums and the penalties alike, so the
fit does not move, as `ewridge`'s does not. The rows that teach it bring `m`
back toward 1, restoring the penalties. At the defaults, with unit weights
and rows one clock unit apart, the penalties act in steady state as a ridge
of `(1 − λ)(β/α + l2)` on the mean scale. A constant target of 5 then
settles at 4.65 at `half_life=100`, and at 4.96 at 1000.

Without a half-life `m` is 1 and `d_i` is `√n_i/α`, which is river's FTRL:
the two agree to 1e-12, row for row. Vowpal Wabbit's `--ftrl` gives the same
prediction and coefficients on every row without a half-life too, to its
single precision, under both losses. No third-party library forgets as a
half-life does, so the decayed form is held to a reference written from the
recursion above.

#### `holt` — Holt's linear trend

*API:* [`po.spec.holt`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.holt) — *Rust:* [`holt.rs`](crates/online-core/src/holt.rs) — *Outputs:* [fields](docs/OUTPUTS.md#holt)

The baseline to beat: the only linear model that takes no features. It
extrapolates the target's own level and trend, so a real model in the same
bank shows what its features add. Compare the two models' `sigma` ([Per-row
diagnostics](#per-row-diagnostics)), or let a
[`seqtest`](#seqtest--a-sequential-test-of-a-sign-by-betting) with `a` and
`b` say which predicts closer.

Level and trend, `l` and `b`, are weighted means of what each row observes
and what the model forecast, with `W` and `V` the weight each has gathered.
A row at weight `w` counts `w` times. With `s` the clock since the target
was last observed, this row's step included, and
`λ_l = 0.5^(s/level_half_life)`, `λ_b = 0.5^(s/trend_half_life)`:

```
pred = l + b·s
l'   = (λ_l·W·pred + w·y)/(λ_l·W + w)          W' = λ_l·W + w
b'   = (λ_b·V·b + w·(l' − l)/s)/(λ_b·V + w)    V' = λ_b·V + w
```

```python
baseline = po.spec.holt(
    "baseline", targets=["y"], clock="t", gap_cap=600.0,
    level_half_life=200.0,        # how fast the level forgets, in clock units
    trend_half_life=2000.0,       # how fast the trend forgets (default four times the level's)
    trend=True,                  # the default
)                                # coef is [level, trend] per target
```

**`level_half_life` is the spec's `half_life` by another name,** so give one
of the two. `trend_half_life` defaults to four times the level's, and `inf`
makes the trend the whole history's drift.

**The forecast has a trend by default.** `trend=False` holds the trend at
zero, and refuses `trend_half_life`. The forecast is then flat: simple
exponential smoothing, whose level is the target's exponentially weighted
mean. There is no seasonal term, because a seasonal index is a
`group` on the phase, which the bank already does.

**The trend is per clock unit, and per second on a temporal clock,** so on
an irregular clock it extrapolates the right distance.

**A row with a null target or weight 0 leaves the level and trend where they
were,** and its clock step is added to the next observation's `s`.

### Moments and correlation

Moments and correlations of the columns you name: the full matrix
(`ew_cov`), each feature–target pair on its own (`marginal`), or one
correlation for the whole matrix (`deco`). `rcov` estimates a block's
covariance from tick returns, written when the block closes. Every per-row
value is read from the state as it stood before the row, so an output can
be a feature for that same row without leaking it.

#### `ew_cov` — exponentially weighted moments

*API:* [`po.spec.ew_cov`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.ew_cov) — *Rust:* [`ewcov.rs`](crates/online-core/src/ewcov.rs) — *Outputs:* [fields](docs/OUTPUTS.md#ew_cov)

Running moments of the columns you name, on the same clock as every model
here: mean, variance, covariance and correlation, and from them partial
correlation, a Mahalanobis distance and principal components. Reach for it
over Polars expressions: it takes one O(k²) update per row, where computing
every pairwise exponentially weighted correlation with Polars expressions
alone takes O(k²) *passes over the data*. A `window_size` cuts its history
off at a fixed age, and `lags` add how a column moves with another some
rows back.

`m` and `C` are the weighted means of `x` and of `(x − m)(x − m)ᵀ`, updated
in Welford's centred form. The raw form, `E[x xᵀ] − m mᵀ`, loses a
variance when the columns sit far from zero.

```
W' = λW + w      a = λW / W'      b = w / W'      δ = x − m          (a + b = 1)
m' = m + b·δ     C'ᵢⱼ = a·Cᵢⱼ + a·b·δᵢδⱼ
varᵢ = Cᵢᵢ       covᵢⱼ = Cᵢⱼ      corrᵢⱼ = Cᵢⱼ / √(CᵢᵢCⱼⱼ)
```

`stats` names what each row writes, mean + std + corr by default:

| in `stats` | what it reports | needs |
|---|---|---|
| `mean`, `var`, `std`, `cov`, `corr` | the moments above, per column or per pair | |
| `partial_corr` | the correlation of two columns with all the others held fixed, read off `(C + s·prior·I)⁻¹` in O(k³) time, spent only when asked | `precision_prior` |
| `mahal` | the Mahalanobis distance, below | `precision_prior` |
| `lagcorr` | the lagged correlation, below | `lags=` |

`stats=[]` is legal: the model learns the moments and writes nothing but
`weight_sum`, and `bank.gram("mv")` reads them back. That is the form for a
wide set of columns.

```python
mv = po.spec.ew_cov(
    "mv", features=["x0", "x1", "x2"], clock="t", gap_cap=300.0, half_life=500.0,
    stats=["mean", "std", "corr", "partial_corr", "mahal"],   # the table above
    precision_prior=1e-6,        # needed by partial_corr and mahal; fades as data arrives
    mahal_quantiles=[0.99],      # adds mahal_q0.99; needs "mahal" in stats
    pca=1, pca_every=20,         # one principal component, its loadings refreshed every 20 rows
)
scores = po.ModelBank([mv]).fit_predict(df).unnest("mv")   # mean_x0, ..., corr_x0_x1, ..., mahal, pc0_*
odd = scores.filter(pl.col("mahal") > pl.col("mahal_q0.99"))   # the joint outliers: every column in range, the combination not
first = scores.select("pc0_share", "pc0_loading_x0", "pc0_loading_x1", "pc0_loading_x2", "pc0_score")
```

**`mahal` is how far the row sits from what the columns do together.** It
is `√(δᵀ (C + s·prior·I)⁻¹ δ)`, with `δ = x − m` and `s` a scale on the
prior that decays with the co-moments, so the prior fades as data arrives.
It says how far the row is from what the columns have been doing
*together*, in standard deviations. On Gaussian columns `mahal²` is about
χ² with `k` degrees of freedom. It is not exact, because the moments are
estimated and the prior adds a little. With one column it is `|z|`.

**`mahal_quantiles` gives a threshold without a distribution.**
`mahal_q0.99` is the exponentially weighted quantile of the Mahalanobis
scores, at the `half_life`, within 0.78%. So `mahal > mahal_q0.99` is "one
row in a hundred" without assuming a distribution.

**`pca` refreshes its loadings on a schedule.** It writes `pc0_var`,
`pc0_share` (of the trace), `pc0_loading_<feature>` and `pc0_score` (this
row's). The eigendecomposition is O(k³), so `pca_every=20` refreshes it
every 20 rows and scores the rows between on the last loadings. Each
refresh keeps the previous sign, so a loading never flips.

**`window_size` cuts `ew_cov`'s history off at a fixed age** ([A hard
window](#a-hard-window)). For moments over a frame that fits in memory,
Polars already does this: `df.rolling("t", period="3h").agg(...)` with an
exponential weight gives the same number to 1e-14. Reach for the spec for a
stream, a saved state, or speed. The rolling window recomputes each window,
in `O(n·W)`, where this is `O(n)`: measured, the spec ran 24 times as fast
at a 74-row window and 1,100 times at a 4,680-row one.

**`lags`: how a column moves with another `ℓ` rows ago.** Lags count
learned rows within the group, not clock units, and the list is strictly
increasing, each lag `>= 1`. They pair each row with the learned rows
before it, kept in a ring. The ring is emptied on a session change and on a
clock gap beyond `gap_cap`, the two events after which "the row `ℓ` back"
no longer means a row `ℓ` ago. Emptying it moves nothing else: the means,
the co-moments and `weight_sum` stay as they were. A zero-weight row ages
the matrices without entering the ring.

The lagged co-moments are the same co-moments, taken between each row and
the learned row `ℓ` before it. With `W` and `m` the weight and mean before
the row, and both deviations taken against that mean,

```
C_ℓ' = a·C_ℓ + a·b·(x_t − m)(x_{t−ℓ} − m)'
```

with the same `a` and `b` the co-moments use, so lag 0 would be the
co-moments `C` exactly. `lagcorr` is written per lag and *ordered* pair,
both orders, because a lagged matrix is not symmetric: a leading b is not b
leading a.

```python
lagged = po.spec.ew_cov(
    "lagged", features=["x0", "x1"], half_life=500.0,
    lags=[1, 5],                 # in learned rows within the group
    stats=["corr", "lagcorr"],   # lagcorr_<a>_<b>_l<l> for each lag and ordered pair
)
lead = po.ModelBank([lagged]).fit_predict(df).unnest("lagged")
# the same numbers from the state: bank.gram("lagged")[0]["lags"], ["lag_comoments"] (an (L, k, k) array)
```

#### `marginal` — every pair's moments, kept in the state

*API:* [`po.spec.marginal`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.marginal) — *Rust:* [`marginal.rs`](crates/online-core/src/marginal.rs) — *Outputs:* [fields](docs/OUTPUTS.md#marginal)

A screen of thousands of features against a few targets, which keeps the
exponentially weighted moments of each (feature, target) pair on its own,
as if every pair were a two-column `ew_cov`. Nothing is written per row but
`weight_sum`: the pairs are the state, read back as a table with
`bank.marginal()`. Two optional views and three settings extend it, in the
table after the example.

A `marginal` is neither a regression nor a joint fit. For `p` features and
`T` targets it takes O(p·T) time per row, where one `ew_cov` over all the
columns would take O((p + T)²). Each pair (feature `j`, target `t`) keeps
its own `m_x`, `S_xx` and `S_xy`, over its target's rows. Per target `t`, on
a row where `y_t` is present, with `W_t` the weight behind that target
before the row:

```
W'_t = λW_t + w        a = λW_t / W'_t        b = w / W'_t        Q'_t = λ²Q_t + w²
S'_yy = a·S_yy + a·b·(y_t − m_y)²             S'_xx = a·S_xx + a·b·(x_j − m_x)²
S'_xy = a·S_xy + a·b·(x_j − m_x)(y_t − m_y)    m' = m + b·(value − m)
```

That is `ew_cov`'s arithmetic, so a pair's correlation is the one an
`ew_cov` over the two columns would report, to the bit.

```python
pairs = po.spec.marginal("pairs", targets=["y", "ret"],
                         features=["x0", "x1", "x2", "signal_a", "signal_b"],
                         clock="t", gap_cap=300.0, half_life=500.0, group="stock_id")
bank = po.ModelBank([pairs])
bank.fit_predict(df)                             # the output record holds weight_sum alone
table = bank.marginal("pairs")                   # one row per (group, instance, feature, target)
one_stock = bank.marginal("pairs", group="b0")   # 10 rows here: five features by two targets
# the columns of table:
# weight_sum                   the target's W_t, the weight behind its pairs
# n_kish                       W_t² / Q_t: the count of equally weighted rows that carry the same information
# mean_x, var_x, mean_y, var_y, cov    the pair's moments, population form
# corr                         cov / sqrt(var_x * var_y)
# beta                         cov / var_x: the slope of the target on that feature alone
# t                            corr * sqrt((n_kish - 2) / (1 - corr²)): the t-statistic at the Kish sample size
```

A bank loaded from a file reports the pairs the bank that saved it would,
and one chunk or a thousand gives the same table to the bit. For unit
weights `d` clock units apart, `n_kish` is `(1 + λ^d)/(1 − λ^d)` in the
limit.

**`t` is a scale for comparing pairs, not a p-value:** the rows are neither
independent nor Gaussian.

**Nulls teach the pairs nothing, and an undefined statistic is null.** A
null target ages its own pairs, `W_t ← λW_t` and `Q_t ← λ²Q_t`, and teaches
them nothing; a null feature skips the row, as everywhere. `corr`, `beta`
and `t` are null until the target's `W_t` reaches `min_weight`, whose
default of 3 exists because two rows give ±1 whatever the data. They are
also null wherever they are undefined: a constant feature, or `n_kish ≤ 2`
for `t`.

Five settings extend the pairs, each in a part of its own below. The two
views, `lags` and `bins`, are off unless asked for. Both also ride into
`bank.closed_groups()` as `pair_*` columns: `pair_split_gain` as a list over
the pairs, and `pair_lagcorr_xx` and `pair_bin_n` as lists of lists. The two
speed settings make a wide `marginal` faster without changing what it
reports where every target is on every row.

| setting | what it adds |
|---|---|
| `lags`, with `serial_rule` | the pair's moments at each lag: a count that allows for rows that resemble their neighbours, and lead/lag correlations |
| `bins` | the target's moments inside each bin of a feature: the relations a correlation cannot see |
| `window_size` | the pairs over a recent stretch |
| `feature_moments="shared"` | speed: one feature moment for every target |
| `shards` | speed: one wide spec on every thread |

**`lags`: is `t` telling the truth?** `t` is built on `n_kish`, which is
the right count for unequal weights and says nothing about rows that
resemble their neighbours. On a smooth stream, consecutive rows are nearly
the same observation, so `t` claims evidence that is not there. Lags fix
that. The pair's moments are kept at each lag too, in learned rows within
the group, the same statistic `ew_cov(lags=)` computes, to the bit. The lag
ring empties at a session change and at a gap past `gap_cap`, as
`ew_cov`'s does.

`serial_rule` turns them into a count that allows for the resemblance,
after Bartlett (1935): `n_serial = n_kish / (1 + 2 Σ_l ρ_x(l)·ρ_y(l))`.
Two *independent* AR(1) series with `φ = 0.9` and `0.8` come out at
`t = 2.39`, and at `t_serial = 1.03` under `"geometric"`. The first looks
like a finding, and the second is the truth.

| `serial_rule` | the sum over the kept lags | suits |
|---|---|---|
| not given | not taken: `n_serial` and `t_serial` are null | |
| `"truncated"` | as it stands; null where the bracket reaches zero or below, as for two series whose autocorrelations have opposite signs | dense lags |
| `"bartlett"` | lag `l` weighted `1 − l/(L + 1)`, `L` the longest kept lag, as Newey and West weight it, so the noisiest lags count least | dense lags, where the long ones are noisy |
| `"geometric"` | `ρ(l) = φˡ` fitted to each series, and the tail summed in closed form, `2φ_xφ_y/(1 − φ_xφ_y)`; `phi_x` and `phi_y` are reported | exponentially weighted series, whose lags need not be dense |

```python
serial = po.spec.marginal(
    "serial", targets=["y"], features=["x0", "x1"], half_life=500.0,
    lags=[1, 2, 3, 5, 8],        # the pair's moments at each lag, in learned rows within the group
    cross_lags=[1],              # the lead/lag terms at these lags only; the default is every lag, [] none
    serial_rule="geometric",     # how the lagged correlations become a count correction
)
bank = po.ModelBank([serial])
bank.fit_predict(df)
table = bank.marginal("serial")
# the columns lags add:
# lagcorr_xx, lagcorr_yy      each series' own autocorrelation, one entry per lag
# lagcorr_xy, lagcorr_yx      the feature now against the target l rows back, and the reverse, one entry per cross lag
# n_serial                    n_kish over the serial_rule's bracket, in the table above
# t_serial                    the same statistic as t, against that count
# phi_x, phi_y                the fitted per-row decays, under serial_rule="geometric"
```

**Read a lead or a follow only against a target measured at the same time
as the feature.** For two such series, a feature whose `lagcorr_yx[0]`
beats its `corr` leads its target, and one whose `lagcorr_xy[0]` does
follows it. Against a forward-looking target the reading fails: the target
`l` rows back is built partly from the feature's newest `l` rows, so every
timely feature built from the same news beats `corr` there.

The lagged lists are `ew_cov`'s `lagcorr` numbers exactly: the lagged
covariance over the two standard deviations. They are not clamped to
`[−1, 1]`, since a lagged correlation is not bounded by one in a finite
sample. A row where the target is missing ages its weight and holds the lag
moments, as it holds the pair's.

**`bins`: what a correlation cannot see.** Everything above is linear, and
a feature can be strongly related to a target with `corr` at zero: a
threshold, a V, a saturation. Bin the feature and keep the target's moments
inside each bin, and all three become visible. The edges come from one of
three rules:

| the edges | how they are set |
|---|---|
| `bin_rule="quantile"` | learned from the first `bin_warm_rows` rows (default 1,000), by weighted quantile |
| `bin_rule="fixed"` | equal widths |
| `bin_edges=` | given outright, a list per feature or a dict by name: exact and comparable across runs, and refused beside `bins`, `bin_rule` and `bin_warm_rows` |

```python
binned = po.spec.marginal(
    "binned", targets=["y"], features=["x0", "x1"], half_life=500.0,
    bins=16,                     # bin each feature and keep the target's moments inside each bin
    bin_rule="quantile",         # the table above
    bin_warm_rows=200,           # the rows the quantile edges are learned from
)
bank = po.ModelBank([binned])
bank.fit_predict(df)
table = bank.marginal("binned")
# the columns bins add:
# bin_edges                   the feature's edges, fixed once and never moved
# bin_n, bin_mean_y, bin_var_y   the target's weight, mean and variance in each bin: the response curve
# split_gain                  the fraction of the target's variance removed by the best single cut
# split_at                    where that cut falls, in the feature's units
# split_gain_t                the t a corr would need to match that gain, at n_serial where there is one
```

**Read `split_gain` against `corr²`, and `split_gain_t` as a ranking.**
`split_gain` is a regression stump's R², so it compares directly with
`corr²`, and the difference is the nonlinear surplus. `split_gain_t` is a
ranking, not a p-value: the cut was chosen by maximising over the
candidates, and the statistic does not know that.

**Binning runs across ten thousand columns in the pass that gives them
`corr`.** It keeps `O(bins)` of state per pair, and takes one search of the
edges per feature per row and a constant per pair. The warm-up rows are
held until the edges are set, then replayed, so the histogram is what it
would have been had the edges been known before the first row. A spec
whose held rows or histogram would pass 256 MiB, per group and per
half-life, is refused when it is built. `bin_budget` moves that limit, and
`float("inf")` removes it.

**A feature keeps only the bins it can support.** Under
`bin_rule="quantile"`, a value that carries more than a bin's share, such as
an indicator's zero, fills a bin of its own, and the rest share what is
left. So the 5% of rows that carry the signal are not lost among the zeros.
A binary feature has two bins, and a constant one has a single bin and no
split. Each bin's moments are kept centred, as every mean here is, so a
target at `1e7` keeps its variance.

**`window_size`: the pairs over a recent stretch.** `window_size` truncates
every pair moment, so `corr`, `beta` and `t` describe the rows inside it and
nothing else ([A hard window](#a-hard-window)). That matters most for a
screen: two regimes of opposite sign average to nothing over a long
history. `weight_sum`, in the record and in the table, is the weight inside
the window. `bins` are refused beside a window.

**`lags` under a window need `window_lags=True`.** Each snapshot then holds
the lag moments too, twice its size at one lag and six times at five with
the default cross lags. A windowed lag moment is an estimate, where the
windowed pair moments are exact.

**`feature_moments="shared"`: one feature moment for every target.** By
default each pair keeps the feature's mean and variance over its own
target's rows. Shared, each feature keeps one over every learned row, and
each pair only its covariance. It takes `lags`, and no `window_size`. Where
every target is on every row the table is the same, to the bit. Where a
target is absent on some rows, `var_x` is the feature's over every row and
`cov` is centred on that mean. That is a different estimator, sound where
the absence says nothing about the feature. At 20,000 pairs it runs 2.7
times as fast at ten targets and 3.2 times at thirty, and a ten-target state
is under half the size. With lags it runs 4.6 times as fast at ten targets
and no cross lags.

**`shards`: one wide spec on every thread.** The bank runs groups and
specs in parallel ([Parallelism](#parallelism)), so a wide `marginal` on
one group is one thread's work. `shards=10` splits its pairs into ten
ranges of features, each run on a thread of its own, a batch of rows at a
time. `shards="auto"` sizes the split to the width and the pool, and leaves
a narrow spec whole; unset, the pairs run on the group's thread. A windowed
`marginal` batches at most `window_every` rows, so `shards="auto"` does not
split it at the default of one. The numbers are the same to the bit at any
count, so a saved bank resumes under any count. On 14 threads
([PERFORMANCE §25](docs/PERFORMANCE.md#25-a-wide-marginal-split-across-the-pool-e73-task-126-2026-09-25)):

| the spec | `"auto"` ran the bank |
|---|---|
| 10,000 features, nine targets, lags and bins | 4.9 times as fast |
| the moments of one target alone | 1.2 times as fast, since the bank's own work on each row does not split |

#### `deco` — one correlation for the whole matrix

*API:* [`po.spec.deco`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.deco) — *Rust:* [`deco.rs`](crates/online-core/src/deco.rs) — *Outputs:* [fields](docs/OUTPUTS.md#deco)

`deco` (Engle & Kelly 2012) replaces the `m(m−1)/2` free entries of a
correlation matrix of `m` series with their average, the
*equicorrelation*. It estimates that in O(m) a row, for the whole matrix or
per block. Its estimate is biased low (below), so use it as a signal that
moves with the market's correlation, not as the correlation itself.

A stream cannot keep every entry moving without O(m²) work a row, and most
are estimated from too little data to be worth moving. The row is
standardised against the means and variances as they stood before it,
`r_i = (x_i − m_i)/√v_i`. With `S₁ = Σ r_i` and `S₂ = Σ r_i²`
over the `n` features that have a standardised value, the row's estimate
`u` is Engle and Kelly's Lemma 2.3, the first line below. The reported
level, `rho`, then follows `u` under one of two dynamics, with `α` and `β`
the `"linear"` dynamic's two weights:

```
u = (S₁² − S₂) / ((n − 1)·S₂)        = (mean of r_i·r_j over i ≠ j) / (mean of r_i²)

"ew":      W' = λW + w,  b = w/W'      ρ' = ρ + b·(u − ρ)                       ρ = u at W = 0
"linear":  ρ' = (1 − α − β)·ρ̄' + α·u + β·ρ      (ρ̄ the "ew" level)          ρ' = ρ at w = 0
```

| `dynamics` | `rho` is | rules |
|---|---|---|
| `"ew"` | the exponentially weighted mean of `u`, decayed on the model's clock: exactly what an `ew_cov(stats=["mean"])` over the rows that have a `u` would report | |
| `"linear"` | the paper's eq. 21, its intercept written as `(1 - alpha - beta) * rho_bar`, since a stream has no sample to fit a free one on. It steps once per row, as a DCC (dynamic conditional correlation) model does, so a capped gap moves it as far as one row's `α·u` | needs `alpha=` and `beta=`, each `>= 0`, with `alpha + beta < 1`. A row's weight reaches `rho` only through `rho_bar`, since the paper's recursion has no row weights |

**`blocks` keep one number per block and one per pair of blocks**, the
useful middle between one correlation and all of them. Every feature sits
in exactly one block, and a block needs at least two.

```python
eq = po.spec.deco(
    "eq", features=["x0", "x1", "x2"], clock="t", gap_cap=300.0, half_life=500.0,
    dynamics="ew",               # or "linear", with alpha= and beta=: the table above
)
blocked = po.spec.deco(
    "blocks", features=["x0", "x1", "x2", "signal_a"], half_life=500.0,
    blocks={"fast": ["x0", "x1"], "slow": ["x2", "signal_a"]},   # every feature in exactly one block
)
out = df.online.fit_predict([eq, blocked])
# u             this row's own estimate, read before the row is learned from
# rho           the level as it stood before the row
# loglik        the row's Gaussian log-density in standardised coordinates under that level
# with blocks:  u_fast, u_slow, u_fast_slow and their rho_* twins, and one loglik over all of them
```

**`u` is a downward-biased estimate of the equicorrelation**, as the paper
says: it is a ratio of two averages, so `E[u]` is about 0.20 for a true
0.30 at six columns.

**`rho` is not an `ew_cov`'s `corr` over the columns.** The mean of a ratio
is not the ratio of the means, and the gap is large.

**A column with no spread is left out.** A column constant from its first
row has no standardised value, so its block's sums leave it out, and its
block reads the correlation among its other columns. A block left with
fewer than two has no `u`. Each correlation value keeps its own weight, so
a value with no `u` on a row learns nothing and does not decay. `loglik` is
null on any row where a column has no standardised value, even when every
block still has a `u`. So a column constant from its first row nulls it on
every row.

#### `rcov` — a block's realised covariance, robust to noise

*API:* [`po.spec.rcov`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.rcov) — *Rust:* [`rcov.rs`](crates/online-core/src/rcov.rs) — *Outputs:* [fields](docs/OUTPUTS.md#rcov)

`rcov` estimates each block's realised covariance from its tick returns,
robust to measurement noise and to series not observed together. Its rows
are *returns*, so difference upstream, and it needs `group` and
`group_close`: a block is one group, and its estimate is written when the
group closes, as that group's row of `bank.closed_groups()`. It takes no
`half_life` or `lam`, since the block is its boundary, and writes no
per-row output but `weight_sum`.

A realised covariance over ticks is the sum of the outer products of the
returns. Over real tick data it is wrong in two ways. Each price is the
efficient price plus a measurement error, and the error's variance
accumulates with every tick. And when the series are not observed together,
the correlation is pulled toward zero. Published estimators remove both
with sums over lags, which is exactly what a stream can accumulate. `kind`
picks one, and [docs/REGIMES.md](docs/REGIMES.md) §8 measures the three
against each block's truth:

| `kind` | the estimator | use it on |
|---|---|---|
| `"plain"` | `sum(x x')`, equal to `n` times an `ew_cov(lam=1)`'s uncentred second moment at close, to the bit: the cross-check, and the reference the other two are measured against | clean returns, where it has the smallest error |
| `"kernel"` | the multivariate realised kernel (Barndorff-Nielsen, Hansen, Lunde & Shephard 2011), `sum_h k(h/(H+1)) Gamma_h` with Parzen weights and jittered end points | returns with measurement noise, which moves its error only from 0.042, on clean returns, to 0.045 |
| `"preavg"` | the modulated realised covariance (Christensen, Kinnebrock & Podolskij 2010): returns pre-averaged over `k_n = ceil(theta * block_rows^0.6)` rows | series not observed on every row, where it is the least biased; `psd=False` was the more accurate form there |

In the kernel, `Gamma_h` sums the products of returns `h` rows apart, `H`
is the bandwidth, and each end point is jittered: averaged over a few
observations at that end. Nothing reads a future row: the jittered *end*
point is formed at close from observations already in the state, and a
product enters `Γ̂_h` only once both its legs are final.

**Parzen is the only kernel:** the Bartlett kernel is not consistent for
this estimator, and Parzen's 0.97 efficiency beats the quadratic
spectral's 0.93.

**`psd=True`, the default, clips any negative eigenvalue and reports
`psd_repaired`.** The table's `k_n` is that form's. For `"preavg"`,
`psd=False` is the balanced form, `k_n = floor(theta * sqrt(block_rows))`
less the bias term: the optimal rate, but not always positive
semi-definite.

**The bandwidth `H` is fixed by `bandwidth=`, or set from the data.** Left
out, `H = ceil(c* xi^(4/5) n^(3/5))` with `c* = 3.5134`, and `block_rows`
is needed: a sizing hint for the ring, which `"preavg"` needs too unless
`preavg_rows=` is given. A longer block runs, clipped, and reports
`bandwidth_used`.

**A gap or a session change splits a block, and a weight is 0 or 1.** A
clock gap past `gap_cap`, or a session change, splits the block into
stretches, and no product pairs two returns across the break. `weight` is
read as 0 or 1 only, since a sum over returns has no fractional row.

```python
rk = po.spec.rcov(
    "rk", features=["x0", "x1"],       # rows are *returns*: difference upstream
    group="block", group_close="monotone",
    kind="kernel",               # "plain", "kernel" or "preavg": the table above
    psd=True,                    # the default: clip any negative eigenvalue
    block_rows=2000,             # a sizing hint for the ring
    bandwidth=None,              # a fixed H; left out, set from the data
)
bank = po.ModelBank([rk])
bank.fit_predict(by_block.select("x0", "x1", "block"))   # by_block: df with a block column, one value per 100 rows
finished = bank.closed_groups()
# rcov, rcorr                  vech of the upper triangle: its entries as one flat list
# rcov_n, rcov_kind, bandwidth_used
# omega2, iv_sparse            the noise variance and sparse integrated variance behind the bandwidth
# iq                           a realised-quarticity proxy, labelled one
# psd_repaired                 whether the estimate had to be made positive semi-definite
# a block too short to estimate from gives nulls, not an error
```

### Clustering and classification

Labels for rows: clusters found with no target, or classes learned from a
label column. `kmeans` finds a set number `k` of round clusters and `micro`
any number of any shape, flagging the rows that belong to none; `ew_class`
learns the classes. Every label, distance and probability is read before
the row is learned from, so it is out-of-sample like every prediction here.

#### `kmeans` — exponentially weighted k-means

*API:* [`po.spec.kmeans`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.kmeans) — *Rust:* [`cluster/kmeans.rs`](crates/online-core/src/cluster/kmeans.rs) — *Outputs:* [fields](docs/OUTPUTS.md#kmeans)

`kmeans` labels each row with the nearest of `k` centres, each the
exponentially weighted mean of the rows assigned to it, which is
`ew_cov`'s mean recursion, per cluster. Reach for it when the number of
clusters is known and they are round; `micro` takes any shape. Three rules
keep the centres on the data, each below: seeding, a split–merge move that
finds a cluster born after seeding, and a dead rule that re-places a centre
whose blob vanished.

With `c_j` centre `j` and `n_j` its decayed weight:

```
j*   = argmin_j ‖x − c_j‖²          distances in units of each feature's EW sd (standardize=False: raw units)
n'_j = λn_j + w                      c'_j = c_j + (w/n'_j)(x − c_j)     for j = j*
```

```python
km = po.spec.kmeans(
    "km", features=["x0", "x1", "x2"], clock="t", half_life=2000.0, gap_cap=300.0,
    k=3,
    warm_rows=100,               # rows to seed from (default 500)
    seed_rule="lloyd",           # the default: the best of ten k-means++ starts
    split_merge=0.5,             # two centres nearer than this times the sum of their radii are one blob
    split_merge_every=200,       # rows between split–merge checks
    dead_frac=0.05,              # under this share of an equal share, a centre is re-placed
    scale_floor=0.1,             # the metric's variance floor, as a fraction of each feature's long-run variance
    update_every=1,              # the default: learned rows between applying each centre's batch
)
out = po.ModelBank([km]).fit_predict(df).unnest("km")
# cluster   the nearest centre's label, before the row is learned from
# dist      the distance to it;  dist2  the distance to the second-nearest
# weight_sum, and coef: the centres, k rows of len(features)
index = po.spec.coef_index(km)   # how coef is labelled: target = "cluster0", "cluster1", ..., term = the feature
```

**Seeding waits for `warm_rows` rows, places the centres, and replays the
rows.** The default `seed_rule="lloyd"` takes ten k-means++ starts, refines
each by ten Lloyd iterations, and keeps the one of least inertia. One start
lands in the wrong partition a third of the time on five blobs in four
dimensions. `seed=` keys the random draws.

**The split–merge move finds a cluster born after seeding.** Every
`split_merge_every` rows, if the two closest centres are nearer than
`split_merge` times the sum of their radii, they are two centres in one
blob. The move then frees the emptier of the two and places it on the rows
far from every centre, once at least three such rows carry 5% of the weight
since the last check. So it needs `k` of at least 3; at `k=2` only the dead
rule acts. `split_merge=0` is plain sequential k-means, with no dead rule
either.

**While `split_merge` is above 0, a far row is scored but not learned
from.** A row is far when it sits outside its cluster, about four standard
deviations of `dist²` above the typical radius, and such rows wait for the
split–merge move. It cannot see one centre owning two blobs, whose rows are
all within its own radius, and seeding with `lloyd` is what prevents that.

**Raise `dead_frac` when regimes change faster than a dead centre is
re-placed.** A centre whose blob vanished fades, and under `dead_frac` of an
equal share it is re-placed. That takes `log2(1/dead_frac)` half-lives: 4.3
at 0.05, 2 at 0.25. A cluster lighter than `dead_frac/k` of the stream loses
its centre whenever any row is far.

**A quiet feature does not take over the metric.** Each feature's variance
is floored at a tenth of its long-run variance (`scale_floor`), which is
tracked at eight times the half-life. After twenty half-lives of quiet, a
feature's weight in the distance, the inverse of its floored variance, is
about 57 times what it was, where the unfloored inverse would be a million
times. So the row on which it moves again is not infinitely far from every
centre.

**`update_every` sets the batch.** It counts the learned rows between
applying each centre's batch: 1, the default, is sequential k-means, and
more a mini-batch one.

#### `micro` — density-based clustering, any shape

*API:* [`po.spec.micro`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.micro) — *Rust:* [`cluster/micro.rs`](crates/online-core/src/cluster/micro.rs) — *Outputs:* [fields](docs/OUTPUTS.md#micro)

`kmeans` needs `k` and finds round clusters. `micro` finds clusters of any
shape without being told their number, flags the rows that belong to none,
and follows clusters that appear and vanish. It is DenStream's
micro-clusters with a linking step over them, tuned by two settings, `eps`
and `beta_mu`.

Each micro-cluster is a *summary*: a decayed weight `n`, a centre `c`, and
a radius `r`, the exponentially weighted root-mean-square distance of its
rows from the centre. A summary is *established* once its weight reaches
`beta_mu` rows of the stream's mean row weight `w̄`.

A row goes to the nearest established summary if that one can take it
without its radius passing `eps`, in units of each feature's exponentially
weighted standard deviation. Otherwise it goes to the nearest summary not
yet established, on the same test, and failing both it opens a new
summary. A weighted row is admitted as a row of the mean weight, and
absorbed at its own. The metric's variance is floored as
[`kmeans`](#kmeans--exponentially-weighted-k-means)'s is (`scale_floor`), so
the next row that moves a quiet feature is not an outlier by that alone.
With `p` the number of features:

```
a = n_j/(n_j + w̄)      b = w̄/(n_j + w̄)
n_j  ← λ n_j                                               every summary
j*   = the nearest established summary, if it keeps  a r²_j + a b ‖x − c_j‖² ≤ eps² p,
       else the nearest other summary, if it does,
       else a new one at x
n_j* ← n_j* + w     c_j* ← c_j* + (w/n_j*)(x − c_j*)     r²_j* ← min(·, eps² p)
```

**Every `prune_every` rows, the light summaries are dropped and the
established ones linked.** Centres within `L` of each other share a label.
`L` is read from the spacing the summaries show, unless `macro_link` sets
it: 2 links only summaries that touch. A cluster's label is the smallest id
among its linked summaries, so it lasts as long as that summary does. One
whose rows stop lingers `half_life · log2(n / (beta_mu · w̄))`, with `n` the
weight it had.

**`max_clusters` caps the live summaries**, 200 by default: past it the
lightest is evicted, an unestablished one first.

```python
mc = po.spec.micro(
    "mc", features=["x0", "x1"], clock="t", half_life=2000.0, gap_cap=300.0, min_weight=50.0,
    eps=0.1,                     # the spread the model reads as *one* cluster, per standardized coordinate
    beta_mu=5.0,                 # a summary holding this many rows of the mean weight is established (default 3)
    prune_every=100,             # rows between pruning and linking
    macro_link=None,             # the linking distance L, read from the spacing unless given
    max_clusters=200,            # the default cap on live summaries
)
out = po.ModelBank([mc]).fit_predict(df).unnest("mc")
out.select("cluster", "outlier", "n_clusters", "n_micro").tail(3)
# cluster              the label of the nearest established summary, on an outlier row too; null while there is none
# dist                 the distance to that summary's centre
# micro_id             the id of the summary this row goes to; ids only go up and are never reused
# outlier              no established summary takes the row
# n_clusters, n_micro  how many of each the state holds
# coef                 the established summaries, one [id, label, n, radius, c_1 .. c_p] row each
```

**`eps` is the spread the model reads as one cluster,** per standardized
coordinate: about 0.07 for two-dimensional shapes, 0.3 for well-separated
Gaussians in 20 dimensions. Both ways to get it wrong show in the outputs:

| what you see | `eps` is | what to do |
|---|---|---|
| nearly every row is an `outlier`, and `cluster` stays null | too small: no summary reaches `beta_mu` before it is pruned | raise `eps` |
| `n_micro` is about the number of clusters | too coarse: each cluster is one summary, so the derived `L` reads the spacing *between* clusters and bridges them into one | lower `eps`, or set `macro_link=2` |

**`beta_mu` is a density of points, set against the arrival rate**, as
DenStream's `MinPts` is. At half-life `h` and `v` rows per clock unit, the
stream's steady-state weight is about `1.44·v·h` rows, so a summary meant to
hold a share `s` of it needs `beta_mu ≈ 1.44·s·v·h`. A denser stream fills
its summaries sooner.

Measured at 20k rows and a half-life of 3,000, moons, rings and five
Gaussians in twenty dimensions all score an adjusted Rand index (ARI) of
1.000 against the truth. `eps` was 0.07, 0.1 and 0.3 for the three, and
`kmeans` cannot follow the first two. At `eps=0.07`, noise drawn uniformly
over the box is flagged `outlier` 94% of the time, and real rows 0.3% of
the time. A cluster born mid-stream had a label 31 rows in, and a test
holds that under 200.

#### `ew_class` — Gaussian classification on `ew_cov` moments

*API:* [`po.spec.ew_class`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.ew_class) — *Rust:* [`ewclass.rs`](crates/online-core/src/ewclass.rs) — *Outputs:* [fields](docs/OUTPUTS.md#ew_class)

A label column in place of a numeric target. The model keeps one `ew_cov`
state per class, a weight `n_c`, a mean `μ_c` and a centred covariance
`C_c`, and scores a row by Bayes' rule over Gaussian classes. Three
covariance forms set what each class keeps, in the table below: its own
covariance (QDA), one pooled (LDA), or variances only (naive Bayes).

`π_c` is a class's share of the weight, and `r_c` the ridge
`precision_prior`, which makes a class scoreable from its first row and
fades as `ew_cov`'s does:

```
π_c = n_c / Σ n         r_c = precision_prior · s_c        (s_c: the prior's fade, as in ew_cov)
M_c = C_c + r_c I  (full)      M = Σ π_c M_c  (shared)      diag(C_c) + r_c  (diagonal)
ℓ_c = ln π_c − ½ ln det M_c − ½ (x − μ_c)ᵀ M_c⁻¹ (x − μ_c)
p_c = exp(ℓ_c − max ℓ) / Σ exp(ℓ − max ℓ)                  class = argmax ℓ
n_c ← λ n_c + w·[y = c]        μ_c, C_c ← weighted Welford on the row's own class
```

| `covariance` | what each class keeps | the right choice when | rows/s, 20 features, 3 classes ([Throughput](#throughput)) |
|---|---|---|---|
| `"full"`, the default (QDA) | its own covariance, and its Cholesky factor, which only the row's own class refactors (under a `window_size`, every class on every row) | in general | 0.47M |
| `"shared"` (LDA) | one covariance pooled by class weight, factorized once per row | the classes differ in location but not in spread; it then labels more than 98% of rows as `"full"` does in the tests, with fewer parameters to learn | 0.53M |
| `"diagonal"` (Gaussian naive Bayes) | variances only | speed matters most; it cannot see a correlation, so two classes with the same marginals and opposite correlations are one class to it | 2.4M |

**`window_size` lets the classifier follow class means that move.** It
makes each class's moments the window's ([A hard
window](#a-hard-window)).

```python
labelled = df.with_columns(      # the label: "up" where y is positive, else "down"
    pl.when(pl.col("y") > 0).then(pl.lit("up")).otherwise(pl.lit("down")).alias("dir")
)
cl = po.spec.ew_class(
    "cl", features=["x0", "x1", "x2"], clock="t", half_life=200.0, gap_cap=300.0, min_weight=20.0,
    label="dir",                 # the label column
    classes=["down", "up"],      # declared up front
    covariance="shared",         # "full" (the default), "shared" or "diagonal": the table above
    precision_prior=0.1,         # required: the ridge that makes a class scoreable from its first row
)
out = po.ModelBank([cl]).fit_predict(labelled).unnest("cl")
out.select("dir", "class", "p_up", "weight_sum").tail(3)
# class        the most probable class, as a string
# p_<class>    one per declared class; exactly 0 for a class no row has carried yet
# coef         the class means, in the order of classes (coef_up_x0 after df.online.unnest([cl]))
```

**A row's probabilities never saw its own label**, since every output is
read before the row is learned from. A label not in `classes` is an error
naming the row. Integer and boolean columns work through their text:
`["0", "1"]`, `["true", "false"]`. A null label scores the row and learns
nothing from it. `weight_sum` counts every row the model accepts, labelled
or not, while `π_c` counts the labelled rows' weights. A stream whose
labels arrive late takes `embargo`, which learns each label once it would
have arrived ([Labels that arrive late](#labels-that-arrive-late)).

On three Gaussian classes with their own covariances, the accuracy sits
within 0.001 of the best any classifier could do given the generating
parameters, and the probabilities are calibrated to about 0.01.

### Sequential tests and regimes

Evidence that something holds or has changed, and which regime the stream
is in. `seqtest` tests whether a column tends to be positive, or which of
two specs predicts closer, and `corrchange` whether the correlation
structure has changed. `bocpd` says how long the current regime has
lasted, and `hmm` which of `k` hidden states the stream is in.

#### `seqtest` — a sequential test of a sign, by betting

*API:* [`po.spec.seqtest`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.seqtest) — *Rust:* [`seqtest.rs`](crates/online-core/src/seqtest.rs) — *Outputs:* [fields](docs/OUTPUTS.md#seqtest)

A `seqtest` asks whether a column tends to be positive, or, with `a` and
`b`, whether one spec of the bank predicts closer than another. Either way
the answer is evidence you can read at any row, as often as you like, and
act on the first time it is enough. A p-value cannot be used that way,
because checking it repeatedly inflates its error rate; an *e-process* can,
which is the reason to reach for this model.

[`po.eval.seqtest`](https://hgilde.github.io/polars-online/eval.html#polars_online.eval.seqtest)
is the same computation over a frame you already have. A `seqtest` is not a
regression. A trial is a row, so there is no `weight` and no `half_life`,
and a spec that gives them is refused.

Per target it keeps the wealth of two gamblers, one betting that the next
sign is positive and one that it is negative. Each stakes the
Krichevsky–Trofimov fraction set by the counts so far, and stakes nothing
unless the counts favour its side:

```
s = sign(y)                    n⁺, n⁻: the signs counted before this row,  n = n⁺ + n⁻
λ⁺ = max(0, (n⁺ − n⁻) / (n + 1))          λ⁻ = max(0, (n⁻ − n⁺) / (n + 1))
ln E⁺ ← ln E⁺ + ln(1 + λ⁺ s)              ln E⁻ ← ln E⁻ + ln(1 − λ⁻ s)
```

**A column's sign.** A zero, null or NaN is a tie, which bets nothing and
counts nothing.

```python
sign = po.spec.seqtest("sign", targets=["y"], group="stock_id")     # does y tend to be positive?
out = po.ModelBank([sign]).fit_predict(df).unnest("sign")
# log_e_pos_y, log_e_neg_y     the two gamblers' log wealth, as they stood before the row
# n_pos_y, n_neg_y             the signs counted so far
```

**Which of two specs predicts closer.** With `a` and `b`, the sign tested
is `|resid_b| - |resid_a|`, positive when `a` came closer, on the
out-of-sample residuals the two specs' output records report. A row where
either side is null (warm-up, a skipped row) is no trial. `a_suffix` and
`b_suffix` pick a grid instance, such as `"@h500"` or `"__r0.5@h500"` ([Output
field names](#output-field-names)). A comparison inside a bank is
chunk-invariant, saved with the state, and works a chunk at a time like
everything else.

```python
common = dict(targets=["y"], features=["x0", "x1"], clock="t", gap_cap=300.0, group="stock_id")
ridge = po.spec.ewridge("ridge", half_life=500.0, **common)
kalman = po.spec.kalman("kalman", half_life=500.0, coef_half_life=100.0, **common)
closer = po.spec.seqtest("closer", targets=["y"], a="kalman", b="ridge", group="stock_id")   # does kalman predict closer?
out = po.ModelBank([ridge, kalman, closer]).fit_predict(df)
# log_e_a_y, log_e_b_y, wins_a_y, wins_b_y   the two sides' log wealth and counts, as for a sign
verdict = out.group_by("stock_id").agg(pl.col("closer").struct.field("log_e_a_y").max())
# log_e_a_y >= ln(20): on that stock, kalman beat ridge at the 5% level, read at any row
```

**The evidence holds however often you look.** The null is that, given
everything so far, the next sign is no more likely positive than negative.
Under it, `E⁺` is a nonnegative supermartingale, and Ville's inequality
gives `P(E⁺ ever reaches 1/α) ≤ α`. So `log_e_pos ≥ ln 20` rejects at the
5% level however many times you looked, and however the rows depend on
each other. No distribution is assumed, and the size of the values is
invisible: 60% small gains and 40% huge losses is "positive". The two
sides' average is an e-value for the two-sided question.

**Two events restart the test.** A session change restarts it under
`session_gap="reset"` or `group_close="session"`, and so does a step back
larger than `restart_after_step_back`. A session change with a numeric
`session_gap` does not.

Where the clip never binds, the wealth has the closed form
`2ⁿ B(n⁺+½, n⁻+½) / π`, the Beta(½, ½) mixture, and the bank is held to it.

#### `corrchange` — has the correlation structure changed?

*API:* [`po.spec.corrchange`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.corrchange) — *Rust:* [`corrchange.rs`](crates/online-core/src/corrchange.rs) — *Outputs:* [fields](docs/OUTPUTS.md#corrchange)

Three tests, for three questions: was the correlation constant over a
span, has it left the level a stable history set, and how big is the
change. `kind` picks the question, and each kind has a part of its own
below.

| `kind` | the question | reports | against | a change is found |
|---|---|---|---|---|
| `"monitor"` | was the correlation constant over a span? | on a span's last row | the span's own correlation | at the span's end, up to `span_rows` rows late |
| `"sequential"` | has it left the level a stable history set? | on every monitored row but the first of each cycle (defined below) | a history of `span_rows` rows | as soon as it crosses a boundary |
| `"window"` | how big is the change? | on every row once both windows are full, `2·span_rows` rows in | the window before | as soon as the two windows differ |

Every kind writes `stat`, `crit`, `flag`, `since_flag` and `since_change`.
On a flag, `since_change` dates the change: the rows from the first changed
one through the flag's. A parameter that belongs to another kind is
refused, naming the kinds it applies to. Under `"monitor"` and
`"sequential"`, the statistic is the largest over the pairs, and the level
`alpha` (0.05 by default) is spread over the pairs as `alpha / npairs`
unless `alpha_adjust="none"`. Under both, `scalar=True` runs the test on
the equicorrelation of the standardised row (`deco`'s `u`) instead: one
statistic however many columns, and a test of its level rather than of a
pair.

**`kind="monitor"`: the closed-sample constancy test** of Wied, Krämer and
Dehling (2012), run over consecutive spans of `span_rows` rows. With `ρ̂_j`
the correlation of the span's first `j` rows, and `D̂` the delta-method
long-run standard deviation of `ρ̂`, the last row of a span reports, per
pair:

```
Q = max_{2≤j≤T} (j/√T)·|ρ̂_j − ρ̂_T| / D̂
```

On a flag, `since_change` counts the rows after the CUSUM's maximum. Under
the null, `Q` converges to `sup|B|` for a Brownian bridge `B`, so the
critical value is the Kolmogorov quantile. It is computed from the
Kolmogorov series rather than stored, and it reproduces the published
1.3581 at 5%. `D̂`'s Bartlett kernel is the paper's, lag `l` at `1 − l/γ`
with `γ = ⌊ln T⌋`.

```python
monitor = po.spec.corrchange(
    "break", features=["x0", "x1"],
    kind="monitor",              # the constancy test above
    span_rows=500,               # nothing is reported until a span closes: a delay of at most this many rows
    alpha=0.05,                  # the level, the default
    scalar=False,                # True: one test on the equicorrelation instead of one per pair
)
out = df.online.fit_predict([monitor]).unnest("break")   # stat, crit, flag, since_flag, since_change
```

A test in the suite holds the size near the nominal 0.05 on Gaussian
pairs. On the paper's own `t_5` design, against the paper's table
([docs/REGIMES.md §2–3](docs/REGIMES.md#2-is-the-monitor-the-size-its-paper-says)):

| on the paper's `t_5` design | here | the paper's table |
|---|---|---|
| the size, at ρ = 0 and `T = 500` | 0.031 | .035 |
| the power on a `0.5 → 0.7` break | 0.552, or 0.582 size-adjusted | .587 |

**`kind="sequential"`: the monitoring procedure** of Wied and Galeano
(2013). A cycle is `span_rows` rows of history, taken as stable, then up to
`monitor_rows` rows, each tested against the history as it arrives. A flag,
or the cycle's last monitored row, ends the cycle, and the next row starts
a new history. From the history it reads each pair's correlation `ρ̂_h` and
its long-run standard deviation `D̂`, with the estimator above. The `k`-th
monitored row then reports

```
V_k  = (k/√m)·(ρ̂_k − ρ̂_h) / D̂           m = span_rows, ρ̂_k over the k monitored rows
stat = max over pairs of |V_k| / w(k/m)
w(b) = (1 + b)·(b/(1 + b))^γ               γ = boundary_gamma, 0 ≤ γ < 1/2
```

and flags where `stat` passes `crit`. On a flag, `since_change` counts the
rows from the change through the flag. The paper's Eq. 8 dates the change:
the argmax of the same CUSUM over the monitored rows before the flag.

The critical value is the paper's Eq. 7: with `T = monitor_rows/span_rows`,
`crit = (T/(1+T))^(1/2−γ)·q`, where `q` is a quantile of
`sup_{0<s≤1} |W(s)|/s^γ` for a Brownian motion `W`. At `γ = 0` it has a
series, and `crit` is 1.5849 at 5 % and `T = 1`. Above 0 the paper
simulates it; here it is solved as a diffusion with an absorbing boundary
([`boundary.rs`](crates/online-core/src/boundary.rs)), within 0.03 of the
paper's Table 1. The detector's size is within two standard errors of
their Table 2 in every cell ([docs/REGIMES.md
§9](docs/REGIMES.md#9-the-sequential-detector-against-its-paper)).

**`boundary_gamma`, the `γ` above, trades early detection for late.** Above
0 the boundary starts lower, so a change soon after the history is caught
sooner, and a stable stream is flagged more often. The default, 0, keeps
the size nearest nominal; 0.45 catches an early change soonest, and 0.25
sits between. For a nominal 0.05, the share of stable streams flagged runs
0.04 to 0.09 at `γ` of 0 and 0.25, and 0.12 to 0.18 at 0.45.

```python
watch = po.spec.corrchange(
    "watch", features=["x0", "x1"],
    kind="sequential",
    span_rows=500,               # 500 rows of history, assumed stable (kind="monitor" over them checks it)
    monitor_rows=1000,           # then each of up to 1000 rows tested as it arrives: the paper's T = 2
    boundary_gamma=0.25,         # between the size of 0 and the early catch of 0.45
)
```

**`kind="window"`: how big the change is.** It measures
`||vech(R_pre - R_post)||` over two adjacent windows of `span_rows` rows
each, `R_pre` and `R_post` being their correlation matrices and `vech`
their upper triangles stacked into one list. It compares that with a
permutation quantile: `n_perm` shuffles of the pooled rows between the
windows, redrawn every `permute_every` rows. The shuffles move blocks of
`perm_block` rows, so rows that resemble their neighbours do not make the
null too liberal. Or give `crit=` as a number and skip the permutations
entirely. The same `seed` gives the same critical values.

The window kind's null is a permutation, not a sign flip, because negating
a whole row leaves every correlation exactly where it was. Its flag rate
per row is not `alpha`. Two windows that slide by one row are almost the
same windows, so a statistic above the quantile stays above it for a run of
rows.

```python
size = po.spec.corrchange(
    "size", features=["x0", "x1"],
    kind="window",               # the size of the change between two adjacent windows
    span_rows=100,
    n_perm=200, permute_every=500,   # 200 shuffles, redrawn every 500 rows
    perm_block=10,               # shuffled in blocks of 10 rows
    seed=0,                      # the default
    reset_on_flag=False,         # the default; True empties both windows at a flag
)
```

#### `bocpd` — how long has this regime lasted?

*API:* [`po.spec.bocpd`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.bocpd) — *Rust:* [`bocpd.rs`](crates/online-core/src/bocpd.rs) — *Outputs:* [fields](docs/OUTPUTS.md#bocpd)

Every other detector here answers "has something changed?" with a
statistic. `bocpd` (Adams & MacKay 2007) keeps a probability distribution
over the **run length**, how many rows since the last break. So its answer
carries the age of the regime with it. "We are forty rows into a regime" is
different information from "something broke".

There is no `half_life` or `lam`, and a spec that gives one is refused: the
run-length posterior is what forgets, and `hazard`, the expected run
length, says how fast. Their Algorithm 1, with `H = 1/hazard` the per-row
chance of a break and `π_r` the posterior predictive of run length `r` for
this row:

```
growth:      P(r_t = r+1, x_{1:t}) = P(r_{t−1} = r, x_{1:t−1})·π_r·(1 − H)
changepoint: P(r_t = 0,   x_{1:t}) = Σ_r P(r_{t−1} = r, x_{1:t−1})·π_r·H
```

Each run length `r` has a *slot* with its own conjugate sufficient
statistics, so slot `r` holds exactly the `r` rows that hypothesis says came
before this one in the run. Slot 0 holds none, so its predictive is the
prior's, which is what makes "a new run starts here" a hypothesis the data
can vote on.

```python
runs = po.spec.bocpd(
    "regime", features=["ret"], group="stock_id",
    hazard=250.0,                # the expected run length: H = 1/hazard is the per-row chance of a break
    prior_nu=2.0,                # the prior on the variance, as 2a ...
    prior_scale=[2e-4],          # ... and 2b: set it from your data (below)
    emission="diag",             # a normal-inverse-gamma per feature: the table below
    prune_below=1e-6,            # the settings table below
    max_run=None,                # 10,000 by default
    hazard_col=None,             # a column of per-row hazards instead
)
out = df.online.fit_predict([runs]).unnest("regime")
# p_change      P(r_t <= 1) given this row: the alarm
# run_mode      the most likely run length, before the row: the run began run_mode rows before this one
# run_mean      the posterior mean run length, before the row
# pred_<f>      the pre-row predictive mean of each feature, mixed over runs
# loglik        the row's log predictive density under that mixture
started = out.with_columns(      # the row each group's current run began on, counted in the group's own rows
    run_start=pl.int_range(pl.len()).over("stock_id") - pl.col("run_mode")
)
```

**Set `prior_scale` from your data.** It is the one parameter you must: too
large and no row is ever surprising. `prior_nu` and `prior_scale` give the
prior on the variance as 2a and 2b in the gamma parametrisation. That is how
Adams and MacKay give their own finance example (a = 1, b = 1e-4,
hazard = 250).

**`run_mode` is the answer, and `p_change` is the alarm.** They are not the
same quality of signal. `p_change` is a per-row likelihood ratio, so it is
spiky, and its height depends on the size of the break against the prior
scale. On three kinds of break:

| the break | `p_change` | the run length |
|---|---|---|
| a ten-fold variance step | 0.83 on the row itself | finds it one to three rows later, and dates it to the right row |
| a four-sigma mean shift, with a diffuse prior | barely lifts | finds it one to three rows later, and dates it to the right row |
| a change in correlation alone | never moves at all | reaches it only under `emission="gaussian"`, a median of 98 rows later ([docs/REGIMES.md §5](docs/REGIMES.md#5-two-changepoint-detectors-on-the-same-break)) |

**`p_change` is `P(r ≤ 1)` rather than `P(r = 0)`.** The changepoint
branch and the growth branch share the same predictive, which makes the
normalised mass at `r = 0` *exactly* `H` on every row, whatever the data.
Row one of a group reports nothing under the default `min_weight` of 1,
since `P(r ≤ 1)` is 1 there however the row looks.

`emission` sets each run's model of the rows:

| `emission` | each run's model | note |
|---|---|---|
| `"diag"`, as in the example | a normal-inverse-gamma per feature | |
| `"gaussian"` | a normal-inverse-Wishart over all of them, O(runs d²) a row | the one that can see a break in the *correlation* alone |
| `"robust"` | each row's contribution weighted by `(pi(x)/pi(mode))**robust_beta` | one 20-sigma row moves nothing; without it, that row is a changepoint at `p_change` 0.91 |

**Keep `robust_beta` below about 0.2.** It is a trade-off: a whole new
regime is a run of individually forgiven rows, so above about 0.2 nothing
is ever detected again. The default of 0.1 ignores a single 20-sigma row
and still finds a four-sigma shift within five rows, dated within a row of
the right one.

Three settings bound the runs kept and set the hazard row by row:

| setting | what it does |
|---|---|
| `prune_below` | drops the runs holding less than this share of the mass |
| `max_run` | 10,000 by default: folds every longer run into the last kept one. In a stream that seldom breaks, this is what bounds the runs kept |
| `hazard_col` | reads the hazard per row from a column instead: a null falls back to `hazard`, and a value of 1 or less is an error naming the row |

#### `hmm` — which regime are we in

*API:* [`po.spec.hmm`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.hmm) — *Rust:* [`hmm.rs`](crates/online-core/src/hmm.rs) — *Outputs:* [fields](docs/OUTPUTS.md#hmm)

`ew_class` classifies a row against *labelled* Gaussians. An `hmm` does the
same arithmetic with no labels: the regime is a *hidden state*, and a
transition matrix carries information from one row to the next. That is the
difference between "which regime does this row look like" and "which
regime are we in", and the second is usually the question.

A hidden state here is one of `k` regimes, not the bank's saved state. The
model runs Hamilton's filter one row at a time, from the filtered `p` the
previous row left:

```
p1_l   = Σ_k p_k·Π_kl                      the predicted state
f_l    = N(x | μ_l, Σ_l + r_l·I)           the state's density, r_l its fading prior ridge
loglik = ln Σ_l p1_l·f_l                   the row's surprise
p_l   ← p1_l·f_l / Σ                       the filtered state
```

```python
hidden = po.spec.hmm(
    "regime", features=["x0", "x1"], half_life=500.0,
    k=2,                         # the number of hidden states
    precision_prior=1e-2,        # required: a state's centred co-moments start at zero, and a zero matrix has no density
    covariance="full",           # the default; "shared" or "diagonal", as for ew_class
    warm_rows=400,               # seed the states from this many learned rows; every output is null until then
    means=None, covs=None,       # or give the states outright, and learn=False to freeze them
    transition_prior=1.0,        # the default Dirichlet pseudo-count per cell of the transition matrix
)
out = df.online.fit_predict([hidden]).unnest("regime")
# filtered_0, filtered_1      the filtered state, before the row
# predicted_0, predicted_1    the predicted state
# state                       the most probable predicted state
# loglik                      the row's surprise
```

**Seeding decides which regimes the filter can find.** `warm_rows` seeds the
states from that many learned rows with `kmeans`' rule
(`seed_rule="lloyd"`, `seed=0`). Those rows should span more than one
regime, or the seeds are two halves of one.

**A regime that lives only in the covariance needs covariances to start
from.** The default seeding is k-means, and zero-mean states differ in
nothing k-means can see, so it splits the rows by direction. On streams
that stay in one of two zero-mean states, the filter then puts about half
the rows after seeding in the true state, 0.51, which is chance. Given
`covs`, it puts 98% there ([docs/REGIMES.md
§1](docs/REGIMES.md#1-does-hmm-recover-the-stream-that-made-it)). Pass
`means` and `covs`, or a feature in which the regime is a shift in
location.

**A row splits across the states.** Everything reported is read before the
row is learned from. Each state's running sums then take the row at weight
`w·p_l`. The responsibilities sum to `w`, so `weight_sum` follows the same
recursion as in every model: a row splits across the states rather than
counting more than once.

**A single extreme row can be captured by one state**, and a state's
moments are weighted means, which a state with zero responsibility keeps as
they are. So a state that stops winning never forgets, and the mixture is
left one state short. Three things mitigate it: a larger
`precision_prior`, states given through `means` and `covs`, or cleaning
upstream. A ridge near the data's own variance, though, halves every
correlation.

**The transitions are what separate this from a clustering.** The
transition matrix is learned from the **filtered joint of consecutive
states**, `ξ_kl = p_k(t−1)·Π_kl·f_l / Σ`. A Dirichlet pseudo-count keeps the
matrix row of a state never visited a distribution. Three settings shape
it:

| setting | what it does |
|---|---|
| `transition_prior` | the Dirichlet pseudo-count per cell of the transition matrix, 1.0 by default |
| `transition=` | spreads that pseudo-count over a matrix of your own |
| `exog_tvtp=` | drives the transitions from a column, through fixed `tvtp_coef=`: time-varying transition probabilities (TVTP) |

**A transition is one row, so a weekend is one step.** The transition
counts decay on the clock, and each row adds its weight, so the chance of
staying rises with the rows' density.

On two-dimensional blobs 1.5 apart, a memoryless nearest-centre rule *given
the true centres* is 85% right, and the filter, with a sticky transition
matrix, is 99% right.

## Performance

This section measures how fast a bank runs and how much memory it holds,
and gives the settings that move both. Each subsection states the
conditions of its own figures, and
[docs/PERFORMANCE.md](docs/PERFORMANCE.md) has every run, where the time
goes and what to reach for:

| subsection | what it gives |
|---|---|
| [Throughput](#throughput) | rows a second for every model, and what sets each speed |
| [Parallelism](#parallelism) | how a bank spreads its work over threads, and the two thread settings |
| [Chunk size](#chunk-size) | what `chunk_rows` changes: the speed and the memory, never the numbers |
| [Memory](#memory) | what a run holds, and which parts of it grow |
| [Tuning memory with Polars' own settings](#tuning-memory-with-polars-own-settings) | the Polars settings that shrink its read-ahead |
| [Window operators](#window-operators) | the time and memory of a window, and of a window as a target |
| [Against scikit-learn](#against-scikit-learn) | the same streams through `SGDRegressor` and through a bank |

### Throughput

The tables give rows a second for each model at its usual settings, one
table per model family, with what sets each speed beside it. The paragraphs
under the tables
take what a cell cannot hold: how targets and half-life grids scale,
`bocpd`'s runs, and `corrchange`'s permutation null. Every table shares
these conditions:

| | |
|---|---|
| machine | an Apple M4 Pro, one process |
| runs | best of 3, 200k rows per run, measured on 2026-09-29 with `uv run python scripts/benchmark.py --markdown`; [PERFORMANCE §28](docs/PERFORMANCE.md#28-the-readmes-numbers-re-measured-2026-09-29) has the run |
| `ewridge` and `rls` | re-measured the same day, once `ewridge`'s solve had been made cheaper ([§30](docs/PERFORMANCE.md#30-where-every-row-solves-2026-09-29)) |
| noise | a run moves by up to about 11% from the last on this machine, so read a gap smaller than that as noise |
| half-life | 1,000, in every row that gives none of its own |
| `k`, `K` | the number of features; the number of clusters or hidden states |

The [linear models](#linear-models):

| model | settings | rows/sec | what sets its speed |
|---|---|---:|---|
| `ewridge` | k=5, 1 target, 1 half-life | 6,199,532 | |
| `ewridge` | k=20, 1 target, 1 half-life | 3,068,424 | |
| `ewridge` | k=50, 1 target, 1 half-life | 946,975 | |
| `ewridge` | k=20, 10 targets | 1,653,010 | one set of feature sums for every target (below) |
| `ewridge` | k=20, 5 half-lives, 500 to 2,500 | 1,714,699 | one set of sums per half-life, run in parallel (below) |
| `ewridge` + `conformal` | k=20, 90% interval | 3,070,074 | nothing measurable: the interval reads the residual the model already has |
| `rls` | k=20, 1 target | 1,662,687 | its square-root form, at 54% of `ewridge`'s speed: the form is what keeps one extreme row from destroying it by cancellation |
| `kalman` | k=20, 1 target | 1,876,826 | |
| `kalman` | k=20, `revert_half_life` | 1,677,506 | |
| `lasso` | k=20, 1 target (3-point path) | 1,629,739 | |
| `huber` | k=20, 1 target | 3,267,509 | |
| `sgd` | k=20, squared loss | 8,423,373 | |
| `sgd` | k=20, `coef_min=0`, `coef_sum=1` | 2,288,779 | a simplex constraint sorts `2k` breakpoints per row, which makes `sgd` 3.7 times slower |
| `pa` | k=20 | 8,284,518 | |
| `ftrl` | k=20, 1 target | 3,873,282 | |

**Targets share one set of feature sums**, so 10 targets take 1.86 times as
long as one, rather than 10 times.

**Each half-life in a grid is its own set of sums, but they run in
parallel.** So the 5-half-life grid takes 1.43 times as long as its
shortest half-life alone, rather than 5 times; that half-life, 500, runs
alone at 2,443,773 rows a second. A shorter half-life solves more often, so
the half-life of 500 runs slower than the 1,000 every other row here uses.

The models of [moments and correlation](#moments-and-correlation):

| model | settings | rows/sec | what sets its speed |
|---|---|---:|---|
| `ew_cov` | k=20: mean, std, corr (230 statistics) | 1,655,993 | |
| `ew_cov` | k=20: mean, mahal, `mahal_q0.99` | 692,690 | the Mahalanobis distance factors a `k × k` matrix by Cholesky once per row |
| `ew_cov` | k=20: mean, cov, lags 1–5 | 1,123,565 | |
| `deco` | k=20, one equicorrelation | 2,093,753 | one number for the whole matrix of its `m` series, computed in `O(m)` a row with no `m × m` matrix to factor |
| `deco` | k=20 in 4 blocks | 1,563,248 | |
| `rcov` | 4 features, kernel, blocks of 1000 | 3,385,427 | it accumulates per row and computes its kernel only when the block closes |

The models of [clustering and classification](#clustering-and-classification):

| model | settings | rows/sec | what sets its speed |
|---|---|---:|---|
| `ew_class` | k=20, 3 classes, full covariance | 466,466 | a `k × k` matrix factored by Cholesky once per *learned* row |
| `ew_class` | k=20, 3 classes, shared covariance | 531,384 | |
| `ew_class` | k=20, 3 classes, diagonal | 2,448,700 | |
| `kmeans` | 4 features, K=8 | 4,590,376 | a distance to each centre |
| `kmeans` | k=20, K=8 | 2,383,197 | a distance to each centre |
| `micro` | 4 features, `eps=1` | 8,799,815 | a distance to each centre |

The [sequential tests and regimes](#sequential-tests-and-regimes):

| model | settings | rows/sec | what sets its speed |
|---|---|---:|---|
| `seqtest` | sign of one column | 14,919,853 | a handful of operations |
| `corrchange` | 4 features, monitor, `span_rows=500` | 449,526 | |
| `corrchange` | 4 features, window 100, permute every 500 | 173,711 | its permutation null (below) |
| `bocpd` | 4 features, diagonal, `max_run=200` | 884,387 | the number of runs it keeps (below) |
| `bocpd` | 4 features, full covariance, `max_run=200` | 568,724 | the number of runs it keeps (below) |
| `hmm` | 4 features, K=2 | 1,182,841 | a `k × k` covariance factorized per state per row, which is `ew_class`'s work with the classes hidden |
| `hmm` | k=20, K=2 | 325,937 | |

**`bocpd` does `O(runs · d²)` work a row, for `d` features, so its speed is
the number of runs it keeps.** With no bound, the run vector grows by one
entry every row and a stream costs `O(rows²)`. A changepoint collapses the
runs to a few dozen, and the distribution onto a short run, so `bocpd` is
faster on data that breaks. A stream that does not break spreads them over
thousands of run lengths, and there `max_run`, 10,000 by default, is the
bound. `prune_below` drops every run holding less than that share of the
mass, which makes it a direct dial on throughput. On 20,000 i.i.d. Gaussian
rows with no `max_run`
([PERFORMANCE §15](docs/PERFORMANCE.md#15-the-correlation-families-bocpds-prune_below-keeps-it-finite-and-rcovs-estimator-sets-its-cost-2026-09-06)):

| `prune_below` | rows/sec |
|---|---:|
| `1e-8` | 204,000 |
| `1e-6`, the default | 324,000 |
| `1e-4` | 687,000 |

**`corrchange`'s window kind is the slowest model here, deliberately.** Its
permutation null redraws `n_perm` statistics every `permute_every` rows, and
giving `crit` as a number skips that entirely.

### Parallelism

**A bank runs one task per spec and group on its own thread pool, and each
task's rows in order, so threads change the speed and nothing else.** The
same stream at 1 and 8 threads, in separate processes, gives identical
output, and everything runs in one process: there is no distributed
execution, by design ([What this is not](#what-this-is-not)). Below are
where the parallelism comes from, a search over factor sets that uses it,
and the two environment variables that set the thread counts.

A task is one spec on one group, or one spec with no `group`. On every
chunk, each task goes onto the bank's own thread pool, which is separate
from Polars' own. It is one flat pool across all specs and all groups,
longest task first, so a few big groups do not leave cores idle at the end.
Within a task the rows go one at a time, because each row's update depends
on the last. That order is what makes the numbers independent of how the
work is split. It also means a bank with one spec and one group is one
thread's work per chunk, while Polars' own reading and writing still run in
parallel around it. So a bank fills the pool with groups, with specs, or
with both:

| source | how it runs in parallel | measured |
|---|---|---|
| groups | one task per group | k=20 over 64 groups: 7.3× from 1 to 14 threads on a 14-core machine (below) |
| specs | one task per spec | eight single-group specs, k=20 over 300k rows, run in 155 ms in one bank, against 641 ms one at a time |
| half-lives | each half-life in a grid is its own set of running sums, and a task's half-lives run in parallel on the bank's pool. Ridge and feature-set grids share one set of sums and are expanded at solve time, so they need no thread | |
| a wide `marginal` | the one exception to a task's single thread: `shards` splits its pairs across the pool within a group ([`marginal`](#marginal--every-pairs-moments-kept-in-the-state)) | |
| Python | Python's global lock is released while a chunk is in the bank, so a Python reader thread can run ahead of `ModelBank.fit_predict` | |

The groups row at each thread count, k=20 over 64 groups
([PERFORMANCE §31](docs/PERFORMANCE.md#31-0130-against-0120-2026-09-29)):

| threads | 1 | 2 | 4 | 8 | 14 |
|---|---:|---:|---:|---:|---:|
| rows/s | 0.98M | 1.89M | 3.44M | 5.98M | 7.08M |

**A search over factor sets is a list of specs, one per set.** Each spec is
its own set of running sums, with its own standardization and its own grid
inside, and runs as its own task on each group. The list runs as one query
in one pass, with the thread counts set before anything is built:

```python
import os
os.environ["POLARS_ONLINE_MAX_THREADS"] = "8"   # the bank's pool: read at the first bank call
os.environ["POLARS_MAX_THREADS"] = "8"          # polars' readers and writers: read at import

import polars as pl
import polars_online as po
from itertools import product

factors = {"mkt": ["x0"], "mkt-sz": ["x0", "x1"], "mkt-sz-val": ["x0", "x1", "x2"]}

def factor_spec(name, features, standardize):
    return po.spec.ewridge(f"{name}-std{standardize:d}",                  # "mkt-std0", "mkt-std1", ...
                           targets=["y"], features=features, clock="t", gap_cap=300.0,
                           group="stock_id", session="session", session_gap=60.0,
                           half_life=[100.0, 1000.0], ridge=[1e-3, 0.1],   # gridded inside the spec
                           standardize=standardize)

specs = [factor_spec(n, f, s) for (n, f), s in product(factors.items(), [False, True])]

(pl.scan_parquet("ticks.parquet")
   .online.fit_predict(specs, chunk_rows=200_000, save_state="grid.state")
   .sink_parquet("grid.parquet"))

scores = po.eval.compare_specs(pl.read_parquet("grid.parquet"),
                               [s["name"] for s in specs]).sort("r2", descending=True)
```

Every chunk puts 6 × 64 tasks on the pool: six specs over 64 groups. On
2.56M rows over 64 groups, that query takes 29.0 s at one thread and 4.3 s
at fourteen
([PERFORMANCE §31](docs/PERFORMANCE.md#31-0130-against-0120-2026-09-29)).
The output is one column per spec, which is what `compare_specs` reads, and
one state file holds them all.

**Two environment variables set the thread counts, one for each pool.** The
bank builds its pool at the first bank call, and Polars builds its own at
import, so each must be set before that point:

| variable | sizes | unset | read at | what took effect |
|---|---|---|---|---|
| `POLARS_ONLINE_MAX_THREADS` | the bank's pool | one thread per core | the first bank call | [`po.thread_pool_size()`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.thread_pool_size) |
| `POLARS_MAX_THREADS` | Polars' readers and writers, and their read-ahead ([Tuning memory with Polars' own settings](#tuning-memory-with-polars-own-settings)) | one thread per core | import | `pl.thread_pool_size()` |

Set each as in the example above, or in the shell, the form that always
works: `POLARS_ONLINE_MAX_THREADS=8 python fit.py`. Set later, the variable
is ignored, and the last column says what took effect. A value that is not
a count is refused by name at the first bank call. The pools never wait on
each other, because a bank task never calls back into Polars' pool, so
giving both more threads than there are cores slows neither.

### Chunk size

**`chunk_rows`, 100,000 by default, is how many rows the bank takes at a
time, and it changes the speed and the memory, never the numbers.** One
chunk or a thousand gives the same output; only where `coef` lands moves,
since a spec reports each group's coefficients on that group's last row of
every chunk. `chunk_rows` is a keyword on `lf.online.fit_predict`,
`lf.online.predict`, `ModelBank.fit_predict_batches`, `ModelBank.fit`,
`with_windows` and `refresh_time`; with `ModelBank.fit_predict(df)`, the
chunk is whatever frame you pass.

**Taller chunks spread a fixed overhead thinner.** Each chunk carries the
work of handing the frame across from Polars, gathering the columns and
assembling the output. On wide frames that hand-off is about 8 ms per call
at 10,000 columns: 4 µs of every row at 2,000 rows per call, and 0.4 µs at
20,000 ([docs/PERFORMANCE.md](docs/PERFORMANCE.md) §20).

**Three chunks are in flight at once**, so `chunk_rows` also sets how much
memory the chunks in flight hold, one of the four parts of memory under
[Memory](#memory).

### Memory

**Memory is proportional to the models' state and the rows a delay or a
window holds, not to the number of rows that have passed.** Every way of
running a bank works a chunk at a time. Below are the peak memory of a
query and of a loop, the four parts memory is made of, and the steps around
the bank, which follow Polars' own rules.

Measured as the most memory the process ever held, on one file of `ewridge`
with 20 features, parquet in and parquet out
([PERFORMANCE §11](docs/PERFORMANCE.md#11-memory-which-surface-is-odata-2026-09-02)):

| what you write | 3M rows | 12M rows | what it is |
|---|---:|---:|---|
| `lf.online.fit_predict([spec])` | 0.90 GB | 1.35 GB | the bank inside a query |
| `for chunk in lf.collect_batches(): bank.fit_predict(chunk)` | 0.80 GB | 1.24 GB | your own loop |

The growth from 3M to 12M rows is not the bank's. It is the memory
allocator keeping pages it has freed, and nearly all the rest is Polars
reading ahead in the parquet file. [docs/RUNNER.md](docs/RUNNER.md) has the
same measurement for the command line, and it is flat too.

**Memory is four things:**

| part | grows with | what bounds it |
|---|---|---|
| the state | the models and their settings, and the number of groups: a bank's `group=` keeps one set of running sums per group, and grows with the number of groups, not the number of rows | it does not grow with the stream, but a window's snapshots and `marginal`'s bins grow with their settings, and each is capped per group, at 256 MiB by default |
| the chunks in flight | `chunk_rows` | three chunks at once ([Chunk size](#chunk-size)) |
| the rows a delay or a window holds | the delay or the window, times the rows' rate | never the stream's length |
| whatever Polars' reader has read ahead | Polars' thread count | `POLARS_MAX_THREADS` shrinks it, and Polars' prefetch settings tune it directly ([Tuning memory with Polars' own settings](#tuning-memory-with-polars-own-settings)) |

**Steps around the bank follow Polars' own rules.** Every step after the
bank in a query, such as a filter, a join, a group-by or writing the
result, runs a chunk at a time, as Polars itself does. A rolling window
over groups, with `.over("group")` or `group_by=`, makes Polars hold every
row, where a bank's `group=` keeps one set of running sums per group. On
the same rows:

| a rolling window | peak memory |
|---|---:|
| with `.over("group")` | 6.5 GB |
| with `group_by=` | 1.7 GB |
| without groups | 0.25–0.28 GB |

### Tuning memory with Polars' own settings

**Polars' read-ahead is the part of memory you can tune.** The streaming
engine prefetches blocks of the parquet file ahead of whatever consumes
them, sized from the thread count, and while a bank is the slowest step, a
local disk needs none of it. Three environment variables move it, all
Polars' own and all settable from Python, and each has a paragraph below:

```python
import os

os.environ["POLARS_ROW_GROUP_PREFETCH_SIZE"] = "1"   # blocks (row groups) read ahead: read at each scan
os.environ["POLARS_MAX_THREADS"] = "4"               # the read-ahead is sized from this too: read at import
os.environ["POLARS_ROW_GROUP_PREFETCH_KBYTES_BUDGET"] = "65536"   # a byte cap on the same read-ahead: read at each scan
```

**`POLARS_ROW_GROUP_PREFETCH_SIZE` is the lever that matters.** It is read
per scan, not once at import. So unlike `POLARS_MAX_THREADS` it can be set
at any point before the scan that should use it, and a scan that has
already run does not lock it in. Setting it in the shell, before the import
or after the import all give the same number. Measured on 8M rows by 12
columns in 80 blocks, as peak resident memory. The allocator's page
retention was off, so that the figure is live data rather than the
high-water mark of everything ever allocated:

| what runs | default prefetch | `POLARS_ROW_GROUP_PREFETCH_SIZE=1` |
|---|---:|---:|
| `lf.online.fit_predict(...).sink_parquet(...)` | 1.63 GB | 1.12 GB |
| `bank.fit_predict_batches(lf)` | 1.41 GB | 1.07 GB |

The gain depends on how much data one block holds.
[docs/PERFORMANCE.md](docs/PERFORMANCE.md) reports a larger reduction on a
file whose blocks hold 262,000 rows, where pinning the prefetch takes a
query from 1.86 GB to 0.51 GB. So measure on your own files rather than
carrying either ratio across.

**`POLARS_MAX_THREADS` sizes the read-ahead too, while the bank's own count
changes only the speed.** Polars' count also sizes how much of a parquet
file its reader holds in flight, so more threads means a bigger pile of
decoded rows. A run that has to fit in a smaller box keeps Polars small and
gives the bank every core:

```python
import os
os.environ["POLARS_MAX_THREADS"] = "4"           # the reader's read-ahead is sized from this
os.environ["POLARS_ONLINE_MAX_THREADS"] = "14"   # the bank still has every core

import polars as pl
import polars_online as po

(pl.scan_parquet("ticks.parquet")
   .online.fit_predict([spec], chunk_rows=200_000)   # spec: any spec, such as As a query's "ridge"
   .sink_parquet("fit.parquet"))
```

On 12M rows over 64 groups with one spec of four features and two
half-lives:

| Polars threads | bank threads | time | peak memory |
|---:|---:|---:|---:|
| 14 | 14 | 1.6 s | 0.86 GB |
| 4 | 14 | 1.9 s | 0.58 GB |

Keeping Polars at four threads holds 33% less memory, for 15% more time.

**`POLARS_ROW_GROUP_PREFETCH_KBYTES_BUDGET` rarely binds.** The byte budget
counts *compressed* bytes, which cost almost nothing for a memory-mapped
local file.

### Window operators

**A window operator's row takes the same work however long the window, and
its memory is about one window of rows.** The exception is an input that is
null on most rows, the first paragraph after the table. The table sets the
operators against Polars' `rolling` recipe and against a window target's
two forms, and the paragraphs after it give what a shared queue saves and
what a window target costs.

Measured on 16M rows arriving about two a second on the clock, parquet in
and parquet out, one group
([PERFORMANCE §33](docs/PERFORMANCE.md#33-window-operators-task-143-2026-10-03),
and [§34](docs/PERFORMANCE.md#34-a-window-expression-as-a-target-task-104-2026-10-03)
for the window targets):

| what runs | time | peak memory |
|---|---:|---:|
| reading and writing the file, and nothing else | 0.40 s | 1.03 GB |
| a forward VWAP through `with_windows` | 2.43 s | 0.99 GB |
| the same VWAP through Polars' `rolling` recipe | 19.52 s | 5.24 GB |
| a model whose target is a window expression, which the bank resolves itself: the native form | 9.37 s | 1.09 GB |
| the window written by `with_windows(..., like=spec)` and fed back as a plain target: the column form | 5.87 s | 1.23 GB |

A window's memory stays within 0.04 GB of reading and writing the file
alone, and a 4-minute window takes what a 1-minute one does. The general
engine takes 1.5 times the time of a core built for one window, at the same
memory.

**An input that is null on most rows is scanned.** An operator whose input
is null on most rows scans the rows between its values on each read: 0.24 s
against 0.04 s on 300k rows, at one value in 5,000
([§35](docs/PERFORMANCE.md#35-two-costs-the-review-of-2026-10-03-named-and-did-not-change)).

**Operators with the same direction, half-life, window and `closed` share
one queue**: sixteen sharing one queue run in 57% of the time of sixteen
with queues of their own.

**The column form of a window target runs faster, and the native form holds
less memory.** Polars runs the column form's window and bank as two stages
of one query, at once, where the native form runs both in one source, one
after the other. The native form also evaluates one Polars query per group
per chunk, on the calling thread, before the groups' rows run on the pool.
With many groups, that is a fixed amount of work per chunk before any row's
own (§35).

### Against scikit-learn

Every contender reaches the noise ceiling on a stationary stream, so
accuracy is not the difference, and the bank is faster at the same
guarantee. Its exact solve wins the first hundred rows of every group,
while sklearn wins on a very wide row by batching and has the ecosystem:
pipelines, `GridSearchCV`, calibration, and far more use. What this library
has is the stream: one pass over rows that need not fit in memory, each
predicted before it is learned.

The model most people compare this to is
[`SGDRegressor.partial_fit`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html).
`SGDRegressor` is a first-order stochastic optimiser, whose answer depends
on the learning rate, the schedule, the feature scaling and the row order.
The primary regressions here are a different class of algorithm:
`ewridge`, `rls`, `lasso`, `huber` and `quantile` accumulate sufficient
statistics and solve, so there is no learning rate. With decay off and
`ridge=0`, `ewridge` is ordinary least squares to 2e-13 of
`numpy.linalg.lstsq`, in any row order. The counterpart to `SGDRegressor`
here is [`sgd`](#sgd--stochastic-gradient-descent), the cheap `O(k)`
baseline.

**Beyond speed and accuracy, the two differ in what one fit can hold:**

| | sklearn | here |
|---|---|---|
| a grid of six penalties | 3.8× the work: six estimators | 1.6× the work: one accumulator, six solves |
| several targets | one estimator each | one shared `X'X` |
| one fit per key | a dict of estimators | `group=`, one state per key |
| standardisation | a `StandardScaler` fitted on the whole frame can leak | streaming, so it cannot leak |
| chunking | | chunk invariance is a test |
| saved model | a pickle | a versioned file that loads on every OS |

Measured on 2026-09-08 on one generated stream of 100,000 rows with
`k = 20`, each contender at its best over a sweep of its own settings. The
script is `scripts/sklearn_comparison.py`, on scikit-learn 1.9.0, and
[docs/PERFORMANCE.md](docs/PERFORMANCE.md) §19 has the full tables and the
sweeps. `po.spec.ewridge` solves on every row here, unlike in
[Throughput](#throughput), and its rates in this section were re-measured
on 2026-09-29, once its solve had been made cheaper
([§30](docs/PERFORMANCE.md#30-where-every-row-solves-2026-09-29)). The
noise ceiling is the R² of the generating signal itself:

| contender | R² stationary | R² drifting | rows/sec | what a prediction saw |
|---|---:|---:|---:|---|
| noise ceiling | 0.9831 | 0.9923 | | |
| `SGDRegressor`, row by row | 0.9829 | 0.9899 | 3,400 | every row before it |
| `SGDRegressor`, batches of 1,000 | 0.9830 | 0.9820 | 2,200,000 | every row before its *batch* |
| `po.spec.sgd` | 0.9826 | 0.9906 | 6,000,000 | every row before it |
| `po.spec.ewridge` | **0.9831** | **0.9907** | 480,000 | every row before it |

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

**A half-life is not the advantage either.** A constant learning rate
`eta` forgets too, at about `1/eta` rows, and on evenly spaced rows that is
a half-life. The clock matters when the rows are unevenly spaced, and this
stream's rows are evenly spaced.

**Where the exact solve wins: the first hundred rows of every group.** An
exact solve is right as soon as its Gram, `X'X`
([The running sums behind a fit](#the-running-sums-behind-a-fit)), is full
rank, about `k` rows in, where a first-order method needs about `1/eta`
rows per direction. The stream below is 500 groups of 200 rows, each group
with its own coefficients and `k = 20`. sklearn runs one estimator and one
scaler per group, in a dict, row by row, and the bank runs `group="g"`. R²
by position in the group:

| contender | R², rows 25–50 | R², rows 50–100 | R², rows 100–200 | rows/sec |
|---|---:|---:|---:|---:|
| noise ceiling | 0.9896 | 0.9903 | 0.9900 | |
| `SGDRegressor` per group, at its best | 0.7277 | 0.9242 | 0.9789 | 3,342 |
| `po.spec.sgd`, `standardize=True`, the same step | 0.7182 | 0.9213 | 0.9788 | 18,519,660 |
| `po.spec.ewridge` | **0.9693** | **0.9860** | **0.9882** | 3,765,320 |

The `sgd` row uses sklearn's step. The 0.01 of R² between the two `sgd`
rows comes from where the *prediction* is standardised. sklearn's loop
standardises the row it predicts against the moments before it, and the
row it learns from against the moments including it. Here one standardised
row serves both, so a prediction is on the same footing as every row the
coefficients were learned from, and the gap closes as the fit converges.

**Where sklearn wins: a wide row against `ewridge`, by batching.**
`ewridge` keeps a `(k+1)²` matrix. At `k = 10,000`:

| contender | state | rows/sec | what a prediction saw |
|---|---:|---:|---|
| `ewridge` | 860 MB | 53 | |
| `ewridge`, `gram_block_rows=1024` | no less | 379 | |
| `SGDRegressor`, batches of 1,000 | 0.31 MB | 20,007 | every row before its *batch* |
| `SGDRegressor`, row by row | | 2,206 | the state as it stands |
| `sgd` | 0.89 MB | 33,271 | the state as it stands |

The time goes to the matrix itself: a rank-1 update moves all 800 MB of it
every row, at 85 GB/s, near this machine's memory bandwidth.
`gram_block_rows=1024` touches the matrix once per 1,024 rows instead, for
7.2× the throughput and no less memory. `sgd` is the `O(k)` answer here,
faster than sklearn's batch, with every prediction made from the state as
it stands. `sgd` and `SGDRegressor` row by row agree, with a correlation of
0.999997 under `standardize=True`, which is sklearn's own recipe of a
scaler in front of the step. Both run at `learning_rate = 0.2 / k`, because
an LMS step is stable only while `eta · |z|² < 2`, and a standardised row
has `|z|² ≈ k`.

## Scope and integrations

polars-online fits models to a frame that is already prepared, and leaves
ingestion, alignment and most windowing to other tools. [What this is
not](#what-this-is-not) lists what it leaves out and what to use instead,
and [Databases](#databases-duckdb-and-adbc) and [Pathway](#pathway) show a
bank fed by database cursors and run inside a stream processor.
[Chunk invariance](#row-order-and-the-two-guarantees) means another
engine's batching cannot change the numbers.

### What this is not

**polars-online is a model layer over a frame that is already aligned: when
a spec names a `clock`, it expects each group's rows in clock order.**
Under a `clock`, a row that arrives out of order is refused, and a label
that arrives late is [`embargo`](#labels-that-arrive-late)'s to hold back.
`clock`, `gap_cap`, `restart_after_step_back` and `session` describe time
*within* a stream, not pipeline lateness.

It does two things to a stream itself, both under [Preparing a
stream](#preparing-a-stream). It puts series that tick at their own times
on one grid, and it computes exponentially weighted means, sums and rates
over a window of the clock. Beyond those, it deliberately does **not**
provide:

| not provided | use instead |
|---|---|
| connectors or ingestion | whatever Polars can read |
| windows beyond those, such as a tumbling or sliding aggregation of any function, and asof or interval joins | Polars expressions upstream, or a streaming framework such as [Pathway](https://pathway.com) |
| watermarks or a late-arrival policy | nothing: see the clock rule above |
| distributed execution | one process, with a thread pool across (spec × group) ([Parallelism](#parallelism)) |

### Databases: DuckDB and ADBC

A database query can feed a bank through the Arrow PyCapsule interface,
with no pyarrow and on py-polars 1.43.0 or later: Polars'
`pl.scan_arrow_c_stream` reads the query's result. DuckDB and ADBC are dev
dependencies of this project, not dependencies of the package.

[examples/duckdb_cursors.py](examples/duckdb_cursors.py) and
[examples/adbc_cursors.py](examples/adbc_cursors.py) sort a query by the
clock and stream it into a bank. **Each gives every lazy query its own
cursor, because a cursor holds one open result.** Sharing one springs a
trap:

| database | what springs the trap | what comes back | what says so |
|---|---|---|---|
| DuckDB | a second query built on the same connection | the first query yields no rows | `ConsumedSourceWarning` |
| ADBC | the cursor executed again before its first query is read | both queries come back wrong: on SQLite, 198,976 and 201,024 of 200,000 rows, out of order | nothing |

Both examples show the rule and the trap, and fail if either stops holding.

### Pathway

A `ModelBank` can run as a stateful operator inside a Pathway pipeline:
Pathway does the ingestion, the event-time alignment and the windowing, and
the bank does the model.

[examples/pathway_integration.py](examples/pathway_integration.py) wraps a
`ModelBank` as a stateful operator, the shape a Pathway pipeline calls. The
example runs the operator over plain batches, checkpointing its state with
`save_bytes` and `load_bytes`. It also sketches the pipeline around it,
which needs Pathway and an input connector to run: Pathway is not a
dependency, and the example imports it lazily.

## Versions, testing and development

This section says which versions of this package and of Polars go
together, what checks each guarantee, and how to build from a checkout.
[Versioning and the Polars pin](#versioning-and-the-polars-pin) is for
anyone installing the package, [Testing](#testing) for anyone deciding
whether to trust it, and [Development](#development) and
[License](#license) for contributors.

### Versioning and the Polars pin

**Pin this package's minor version if you need stability; its Polars range,
`polars>=1.34.0,<3`, is measured rather than guaranteed.** Below are this
package's version rules, the pins and the lowest Polars each feature needs,
how the range moves when a Polars breaks this library, and which interfaces
into Polars carry a promise.

#### This package's own versioning

**Pin the minor version, `~=0.13.0` for the 0.13 series, if you need
stability.** This package follows semantic versioning
([CHANGELOG.md](CHANGELOG.md)), and while it is pre-1.0, the **minor**
version carries breaking changes and any change to the numbers a model
returns. Output field names are part of the API
([Output field names](#output-field-names)), and the Polars range moves by
these rules:

| a change to the Polars range | release |
|---|---|
| widening it | minor |
| narrowing it | breaking |
| capping it below a Polars that broke this library, as [How the pin moves](#how-the-pin-moves) describes | patch |

#### What is pinned

The Rust `polars` is pinned exactly and built into the wheel, while the
py-polars you install is a range, whose floor depends on what you use. The
two copies share one process, and data crosses between them only through
the interfaces listed under
[Which interfaces carry a promise](#which-interfaces-carry-a-promise), so
the runtime requirement can be a range:

| py-polars | rust polars | pyo3-polars | pyo3 | Python |
|---|---|---|---|---|
| **>= 1.34.0, < 3** (built and tested against 1.44.2) | 0.55.2 | 0.28 | 0.29 | ≥ 3.12 (`abi3-py312`) |

| what you use | the lowest py-polars | why |
|---|---|---|
| `ModelBank` alone | 1.28.1 | |
| `lf.online.fit_predict`, `ModelBank.fit_predict_batches`, `ModelBank.fit`, `with_windows` and `refresh_time` | 1.34.0 | they read with `LazyFrame.collect_batches`, which py-polars added in 1.34.0 |
| the examples that stream a DuckDB, ADBC or pyarrow source into a bank | 1.43.0 | for `pl.scan_arrow_c_stream` |

The suite passes on 1.44.2, the pin, at every change, and has passed on
1.34.0, 1.38.1, 1.44.1 and 2.0.0-rc.1 with identical numbers. Its last
whole run on 1.34.0 came before the window operators, and the formula
reader they use was checked there on its own. On 1.34.0, two duration edge
cases differ in Polars' own parser. A test asserts the Python pin and the
range, and the matrix, with the date of each run, is in
[docs/RELEASE-READINESS.md](docs/RELEASE-READINESS.md).

#### How the pin moves

When a new Polars breaks this library, the response is decided in advance.
First **cap** the range at the last release that passed, in a patch release,
so no resolver hands anyone the broken pair. Then **fix**, and widen again.
Two checks look for such a break, a weekly canary and the legs of every
release, and both watch NumPy too, the one optional dependency
(`polars-online[numpy]`, `numpy>=1.24`).

**The canary is a weekly job**
([`polars-canary.yml`](.github/workflows/polars-canary.yml)). It drops the
range from `pyproject.toml`, installs the newest py-polars, prereleases
included, and builds the wheel as CI does. It then runs the suite, all but
the opt-in soak tests and the checks of this repository's own pins. Only
Polars moves in that run, so a red canary means Polars broke this library
and nothing else did. The canary also installs NumPy's next release
candidate each week.

**Every release runs the same check before it publishes**, in legs
(`release.yml`):

| leg | resolves to | blocks the publish |
|---|---|---|
| the newest in-range | the newest stable inside `<3` | **yes** |
| the next major | unpinned, prereleases allowed | no |
| the newest NumPy | the newest NumPy | yes |
| NumPy's next release candidate | its next release candidate | no |

The first leg backs the range. `<3` admits every 1.x and 2.x, so a resolver
can hand someone a Polars newer than the one the wheel was built against,
the day after it ships. A pass on the pinned version alone would not show
that the range holds. The second is early warning, and the steps for
raising the ceiling to a new major are in
[docs/RELEASE-READINESS.md](docs/RELEASE-READINESS.md). Only NumPy moves in
the NumPy legs and the canary's NumPy run, so a red one names NumPy.

#### Which interfaces carry a promise

This library crosses into Polars four ways, and only one of them carries a
guarantee. The three without one include the two that stream, `ModelBank`
and the IO plugin, so a break on a new Polars is expected maintenance:

| interface | used by | its promise |
|---|---|---|
| pyo3-polars' extension types | `ModelBank` | none beyond the latest definitions working with the latest Polars: provided "for convenience" |
| the IO plugin | `lf.online.fit_predict`, `lf.online.predict`, and `with_windows` and `refresh_time` on a query | none: documented, but `@unstable` in py-polars |
| an expression's serialized form, `expr.meta.serialize(format="json")` | the window formulas, read into this library's own tree | none: Polars calls it unstable across versions; its shapes were measured the same on 1.34.0 and 1.44.2 |
| the Arrow PyCapsule interface | [`fit_predict_arrow`](#output-as-arrow) | an Arrow specification, which py-polars and pyarrow consume, so a break there would be Arrow's rather than Polars' |

The Arrow PyCapsule interface narrows the exposure rather than removing it:
only the output side uses it today, and the frame still goes in as a Polars
frame.

**A mismatch raises an error; it does not crash the process.** `ModelBank`
moves data across the boundary through the Arrow C Data Interface, and a
Polars without the two private methods it reads fails with a clean
`AttributeError` before any data moves.

### Testing

A guarantee in [What you can rely on](#what-you-can-rely-on) is only worth
what checks it, so the suite holds each to an oracle or an invariant,
rather than to expected values typed in by hand. It has 1,215
Rust tests and 3,840 pytest cases, counted on 2026-10-03, run on three
operating systems at every push. Each kind of check follows, led by its
name in bold, then where each runs; [docs/TESTING.md](docs/TESTING.md) is
the ledger of what each part proves.

**Against references.** Each model is held to something it cannot share a
bug with: a reference written from its documented recursion, and, wherever
another library computes the same quantity, that library.

| model | held against | agreement |
|---|---|---|
| `ewridge`, `rls` | references written from the recursions | 1e-9 |
| `ewridge` | scikit-learn's `Ridge` | 1e-8, relative |
| `rls` | `ewridge(ridge_scale="sum")` solved on every row | below 1e-9 |
| `kalman` | a reference; filterpy's Kalman filter; [river](https://riverml.xyz)'s `BayesianLinearRegression`, the case with no drift | about 1e-15; 1e-9; 3.6e-15 |
| `lasso` | the KKT conditions of its objective; a reference descent, for every prediction | the conditions hold; about 1e-14 |
| `huber`, `quantile` | references; scikit-learn's `HuberRegressor` and statsmodels' `QuantReg`, batch fits of the same objectives | about 1e-13; statistically |
| `sgd` | scikit-learn's `SGDRegressor`, one per group | R² within 0.03 |
| `ftrl` | a reference; river's FTRL, row for row; Vowpal Wabbit's `--ftrl`, on every row | about 1e-16; 1e-12; its single precision |
| `pa` | river's `PARegressor` | 1e-12 |
| `holt` | statsmodels' `Holt`, once the weight saturates; pandas' `ewm(times=)`, without a trend | 1e-12; 1e-8 |
| `ew_cov` and the EW moments | pandas' `ewm`; river | 1e-12; 1e-9 |
| `ew_class`, `mahal` | scipy's multivariate normal and Mahalanobis distance | 1e-9 |
| `marginal` | scipy's `binned_statistic` and a scikit-learn decision stump, for the bins; statsmodels' autocorrelations, for the serial count | 1e-9; 0.02 |
| `bocpd` | the `bayesian_changepoint_detection` package | 1e-9, and `run_mode` exactly |
| targets with gaps | statsmodels' weighted least squares, ridge and elastic net | 1e-7 |
| the window operators | a loop written from each definition, and the same window over the reversed stream; Polars' `ewm_mean_by` and `ewm_sum_by`; Polars' `rolling_sum_by`, for which rows a window holds | to rounding; 1e-9; exactly |
| a window target | the same window written as a column by `with_windows(like=spec)` and fed back as a plain target under the same embargo | prediction for prediction |
| the core's hand-written solves, in the Rust tests | `faer`'s | |

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
| `weight_sum` | the same recursion in every model (`crates/online-core/tests/model_contract.rs`) |
| a window run resumed | a chain of runs saved under a slice and resumed gives one run's output, at chunks of 1, 7 and 100,000 rows, and another input is refused |

**Generated streams.** In the Rust tests, `proptest` drives generated
streams through every model against the same contract. Hypothesis
generates adversarial streams, with mixed nulls, duplicate and long-gap
clocks, values at ±1e8, zero weights and tiny groups. It asserts the
strongest invariant: **changing a row's own target never changes that
row's own prediction.** An information coefficient (IC) of about 0 on
pure-noise targets says the same thing from the other side.

**Fixed numbers.** There is one golden stream per model in the Rust core.
The whole pipeline, from reading the columns through the models and the
diagnostics to assembling each spec's output, is pinned to fixed output
and compared on every operating system. So a divergence in Polars'
vectorized paths on another CPU would show.

**Hardening.** What the suite does to a bank on purpose:

| attack | what must hold |
|---|---|
| everything at once | a 30k-row stream with every output switched on, compared by digest across chunkings, a mid-stream save and load, and thread counts |
| weight scale | all weights ×1e±6, with `min_weight` scaled alike, change nothing but `weight_sum` |
| parameter edges | `half_life` from `1e-3` to `inf` |
| a corrupt state file | any byte flipped fails cleanly, and never panics |
| concurrent misuse | two threads calling `fit_predict` at once get a clean error |
| copying a bank | `pickle` and `copy.deepcopy` resume bit-exactly |
| across the FFI | memory safety where two copies of Polars share one process |
| sustained load | a 10M-row soak, opt-in with `pytest -m soak` |

**Contracts that are files.** What is held to a file, and how:

| contract | how it is held |
|---|---|
| the public API: every name, default and signature, and every output field name | a checked-in snapshot (`tests/api_surface.txt`), so a change is a reviewable diff |
| every python block in this README, and every example in the API reference | each one runs |
| some docstring text | pinned where a test reads it |
| everything under `examples/` | runs unmodified: the TOML through the real command line, the Pathway example's operator over plain batches, and the cursor examples against real DuckDB and SQLite databases |
| the state files 0.10.0 and 0.11.1 wrote | read and refused by their schema, as the CHANGELOG says they are |
| `docs/VALIDATION.md`, where the defaults were chosen | regenerated and compared, so the numbers behind them cannot silently stop being true |
| no data files in the repository | tests generate or download their own data, and a data file, a large file or generated output that gets tracked fails a test |

**Where it runs.**

| when | what runs |
|---|---|
| before every commit | `./scripts/gate.sh`: `cargo fmt`, `clippy -D warnings`, `cargo test`, `uv lock --check`, `ruff`, `mypy`, the build, `pytest` and `sphinx -W` |
| every push and pull request | the tests on Ubuntu, Windows and macOS, on Python 3.12 and 3.14, and on 3.13 on Linux; the format, lint, type and documentation checks, on Linux; mutation testing of the lines the change touched, which fails on a mutant no test catches; and every output compared with the newest release's, bit for bit, as a report |
| every release | a state file written on macOS and continued on Windows and Linux; the suite on the newest Polars the range admits, and on the newest NumPy |
| weekly | the suite on the newest py-polars, and on NumPy's next release candidate; a leak check across the boundary with Polars; mutation testing of all of `online-core`, as a report |

### Development

Building and testing from a checkout needs [uv](https://docs.astral.sh/uv/)
and a stable Rust toolchain ([rustup](https://rustup.rs)), and
[CONTRIBUTING.md](CONTRIBUTING.md) says how to make changes.
`source scripts/env.sh`, or `. .\scripts\env.ps1` in PowerShell, puts both
on the `PATH` for a shell, and `.vscode/settings.json` does it for VS
Code's terminal. The commands below run everything CI checks, and the table
after them says which document to read for what.

`cargo` runs through `uv run` because `online-py` builds against pyo3's
`abi3-py312` and needs a 3.12+ interpreter at build time. Downloaded test
data is cached under `.cache/`, and the tests that need it are skipped when
offline.

```sh
uv sync                                                # Python env (CPython 3.12 or newer)
./scripts/gate.sh                                      # everything CI checks
uv run cargo test --workspace --exclude online-py      # Rust tests
uv run maturin develop --release -m crates/online-py/Cargo.toml
uv run pytest                                          # Python tests
uv run --group docs sphinx-build -W docs/reference docs/_build/html   # API reference
uv run python scripts/validate.py > docs/VALIDATION.md # re-run the experiments that chose the defaults
uv run python scripts/regime_experiments.py all        # the docs/REGIMES.md experiments
uv run python scripts/benchmark.py                     # throughput
uv run python scripts/windows_bench.py                 # the window operators against Polars' rolling recipe
uv run python scripts/sklearn_comparison.py            # the comparison with scikit-learn
uv run python scripts/compare_release.py               # every output against the newest release's, bit for bit
uv run pytest -m soak                                  # the opt-in 10M-row soak
./scripts/mutants.sh --in-diff <(git diff main...)     # mutation testing of the code a branch touches
```

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

Apache-2.0. See [SECURITY.md](SECURITY.md) to report a vulnerability.
