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
| [Introduction](#introduction) | [a first fit](#a-first-fit) · [the idea](#the-idea) · [terminology](#terminology) · [what you can rely on](#what-you-can-rely-on) · [install](#install) · [example data](#example-data) |
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

Here is a first model fit and an introduction to the library.

### A first fit

This first fit regresses each stock's return on two signals over a made-up
trading day with known true betas. To run it, install the library and numpy
with `pip install polars-online numpy` ([Install](#install) has the
requirements).

A *spec* describes one model, and `fit_predict` fits a list of specs in one
pass over the rows, as a *model bank*.

```python
from datetime import datetime

import numpy as np                              # only to make up the rows
import polars as pl
import polars_online as po

# 1. Make up the input: a trading day, 9:30 to 16:00, a row a second for stocks A, B and C in turn.
rng = np.random.default_rng(0)
n = 23_400                                      # six and a half hours of seconds
signal_a = rng.standard_normal(n)               # new every second
signal_b = np.cumsum(rng.normal(0.0, 0.05, n))  # wanders slowly
prices = pl.DataFrame({
    "ts": pl.datetime_range(datetime(2024, 1, 2, 9, 30), datetime(2024, 1, 2, 16), "1s",
                            closed="left", eager=True),
    "stock_id": ["A", "B", "C"] * (n // 3),
    "signal_a": signal_a,
    "signal_b": signal_b,
    # beta 0.2 rising to 0.8 on signal_a, and -0.2 on signal_b
    "ret": np.linspace(0.2, 0.8, n) * signal_a - 0.2 * signal_b + rng.standard_normal(n),
})

# 2. One fit over every row: fit the rows before 15:30, save the state, read it back, serve the rest.
whole = po.spec.ewridge(
    "whole",                                    # the spec's name: it names the output column
    targets=["ret"], features=["signal_a", "signal_b"],
    half_life=float("inf"),                     # no forgetting: every row counts the same
    group="stock_id",                           # one regression per stock
)
flat = (
    prices.head(21_600).lazy()                  # a query over the rows before 15:30
    .online.fit_predict([whole], save_state="bank.state")   # adds a column "whole", saves the state
    .online.unnest([whole])                     # as columns: pred_<target>, resid_<target>, ...
    .collect()                                  # runs the query
)
saved = (
    po.ModelBank.load("bank.state").coef()      # the fit in the file, a row per coefficient
    .pivot("term", index="group", values="coef")   # a row per stock, a column per term
)
served = prices.tail(1_800).lazy().online.predict("bank.state").collect()   # scores the last half hour, learns nothing

# 3. A fit that follows the recent rows, and its betas on every row.
local = po.spec.ewridge(
    "local", targets=["ret"], features=["signal_a", "signal_b"], group="stock_id",
    clock="ts", half_life="10m",                # a row's weight halves every ten minutes of ts
    gap_cap="5m",                               # and a gap longer than five minutes decays as five
    coef_every=1,                               # the coefficients on every row
)
betas = prices.lazy().online.fit_predict([local]).online.unnest([local]).collect()   # a row of betas per input row

# 4. A target that looks ahead: each row's mean return over the next ten minutes.
fwd_ret = po.rewm_mean("ret", half_life=float("inf"), window_size="10m").alias("fwd_ret")
ahead = po.spec.ewridge(
    "ahead", targets=[fwd_ret], features=["signal_a", "signal_b"], group="stock_id",
    clock="ts", half_life="10m", gap_cap="5m",
    embargo="10m",                              # learn each row once its ten minutes have passed
)
forecast = prices.lazy().online.fit_predict([ahead]).online.unnest([ahead]).collect()   # each row's forecast of fwd_ret
```

The input, `prices`, begins with these rows:

| ts | stock_id | signal_a | signal_b | ret |
|---|---|---:|---:|---:|
| 09:30:00 | A | 0.13 | 0.01 | -0.10 |
| 09:30:01 | B | -0.13 | 0.03 | 0.60 |
| 09:30:02 | C | 0.64 | 0.02 | 0.39 |
| 09:30:03 | A | 0.10 | -0.03 | 0.21 |
| 09:30:04 | B | -0.54 | -0.08 | -0.08 |
| 09:30:05 | C | 0.36 | -0.01 | 0.24 |

**Step 2 weights every row equally, because `half_life=float("inf")` forgets
nothing.** So `whole`, a ridge regression solved from running sums, converges
to the batch fit over every row it has seen. Its state is a complete summary
of those rows, worth saving and serving.

`fit_predict` adds one column per spec, holding a record of fields in each
row, and `.online.unnest`, chained after it, spreads them into columns. For
stock A, `flat` begins:

| ts | ret | pred_ret | resid_ret | weight_sum | withheld_reason |
|---|---:|---:|---:|---:|---|
| 09:30:00 | -0.10 | null | null | 0 | above_max_error_inflation |
| 09:30:03 | 0.21 | null | null | 1 | above_max_error_inflation |
| 09:30:06 | 0.52 | null | null | 2 | above_max_error_inflation |
| 09:30:09 | 1.22 | -0.47 | 1.69 | 3 | null |
| 09:30:12 | -1.18 | 1.64 | -2.81 | 4 | null |

`pred_ret` is each row's prediction from the stock's rows before it, and
`resid_ret` is `ret` less `pred_ret`. `weight_sum` is the weight behind that
prediction: here the count of the stock's rows learned so far. A stock's
first three rows have too little behind them, so `pred_ret` is null and
`withheld_reason` says why ([Warm-up](#warm-up)).

The state saved at 15:30 holds each stock's fit, near the average of the true
betas until then:

| group | intercept | signal_a | signal_b |
|---|---:|---:|---:|
| A | -0.01 | 0.48 | -0.19 |
| B | -0.01 | 0.45 | -0.19 |
| C | 0.01 | 0.48 | -0.20 |

`served` scores the rows from 15:30 with that state, and learns nothing from
them. To keep learning from new rows instead, call
`.online.fit_predict(load_state="bank.state")` on a query over them
([Saving, loading and serving](#saving-loading-and-serving)).

**Step 3 follows the recent rows, with a clock and the `gap_cap` a clock
requires.** Give the same spec a clock (timestamp column `ts`) and
`half_life="10m"` and each row's weight halves every ten minutes of `ts`. The
fit is now local: it describes the recent past, and it moves from row to row,
weighted toward the last few tens of minutes. Because this is a local
regression, remember that saving the final state may be of limited value.
But using `coef_every=1` produces a time series of the betas at every row, as
they stood after learning that row. This can be used to track the betas over
time.

Each hour from 10:30, stock A's beta on `signal_a` climbs with the true one:

| ts | coef_ret_signal_a | coef_ret_signal_b |
|---|---:|---:|
| 10:30:00 | 0.30 | -0.16 |
| 11:30:00 | 0.35 | -0.16 |
| 12:30:00 | 0.47 | -0.21 |
| 13:30:00 | 0.57 | -0.18 |
| 14:30:00 | 0.66 | -0.22 |
| 15:30:00 | 0.72 | -0.18 |

**Step 4 forecasts a target that looks ahead.** `fwd_ret`, a Polars
expression written with `po.rewm_mean`, is the time-weighted mean of the
stock's returns over the next ten minutes.

`fit_predict` requires an `embargo` of at least the target's `window_size`;
otherwise it refuses the spec, so for a shorter embargo use `ModelBank.fit`,
which keeps no predictions. With `embargo="10m"`, each row is scored where it
sits, before its target is known, and learned from ten minutes later. So
`ahead` learns nothing from 9:30 to 9:40, and `resid_fwd_ret` is null on
every row:

| ts | pred_fwd_ret | resid_fwd_ret | weight_sum |
|---|---:|---:|---:|
| 09:30:00 | null | null | 0 |
| 09:35:00 | null | null | 0 |
| 09:40:00 | null | null | 0 |
| 09:45:00 | -0.17 | null | 85 |
| 09:50:00 | -0.26 | null | 145 |

### The idea

Here are a few points to remember about fitting models with polars-online.

**Every prediction and diagnostic is out-of-sample.** The bank predicts each
row and computes its diagnostics from what the models learned before it, and
only then learns from the row ([Diagnostics, selection and
evaluation](#diagnostics-selection-and-evaluation)).

**The stream can be far larger than memory.** A bank keeps only what its
models have learned, and the rows an `embargo` or a window still holds. To fit
parquet files larger than memory, start the query with
`pl.scan_parquet("ticks/*.parquet")` before `.online.fit_predict`, and end it
with `.sink_parquet("fitted.parquet")`, which writes the result to a file, in
place of `.collect()`.

**Row order matters when a model forgets.** With a *decay*, older rows count
less, so sort the rows by time with Polars' `sort`, in the query before the
bank, unless they arrive sorted. Without a decay, the models that solve or
accumulate give the same answer in any order ([Convergence without a
decay](#convergence-without-a-decay) names them).

**Features and targets can be windows over the stream, written as Polars
expressions.** A feature can look back, such as the last minute's
time-weighted mean of the mid. Write it with `po.ewm_mean`, and add it as a
column in the query before the bank with `.online.with_windows`, which takes
the same `clock` and `gap_cap` as a spec. A target can look ahead, such as the
next minute's VWAP less the mid. Write it with `po.rewm_sum`, and pass it in
the spec's `targets`. Windows always need the rows in time order ([Preparing
a stream](#preparing-a-stream)).

**A model bank runs three ways, with the same numbers from each.** It runs
inside a Polars query with `.online.fit_predict`, in your own Python loop
calling `ModelBank.fit_predict` on each chunk, or from the standalone
`online` command line, without Python ([Running a bank](#running-a-bank)).

### Terminology

polars-online uses these words in a sense of its own:

| term | meaning |
|---|---|
| **spec** | the description of one model: which model, which columns it reads, and how it treats time. `po.spec.ewridge(...)` builds one |
| **model bank**, or *the bank* | a set of specs fitted together over the same rows, and the Python object that holds them, [`ModelBank`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.ModelBank). *The bank* always means a model bank |
| **stream** | the rows in the order the bank reads them |
| **chunk** | one of the pieces the bank takes the stream in, one at a time: `chunk_rows` rows in a query, 100,000 by default, or each frame passed to `ModelBank.fit_predict` |
| **state** | everything a bank has learned |
| **clock**, **decay** | the column, named by `clock=`, that says how far apart two rows are, and the forgetting measured along it: each row's weight halves every `half_life` of the clock ([Time and decay](#time-and-decay)) |
| **`weight_sum`** | the *weight behind the state* that produced a row's prediction, which `min_weight` reads ([Warm-up](#warm-up)) |

### What you can rely on

These hold for every spec, however a bank is run:

| | |
|---|---|
| **honest predictions** | every row is predicted before its own outcome is learned ([Row order and the two guarantees](#row-order-and-the-two-guarantees)), and a target that looks ahead is learned only once its window has closed ([Windows as a model's inputs and target](#windows-as-a-models-inputs-and-target)) |
| **bounded memory** | memory grows with the models' state and the rows a delay or a window holds, and never with the number of rows that have passed ([Memory](#memory)) |
| **any chunking** | one chunk or a thousand, with or without a save and resume in the middle, gives the same numbers, to the last bit. Only which rows carry the coefficients, `coef`, can change ([Row order and the two guarantees](#row-order-and-the-two-guarantees)) |
| **any thread count** | the thread count, set with `POLARS_ONLINE_MAX_THREADS`, changes only the speed. Each spec and group is one task on the bank's thread pool: with 64 groups, 14 threads process 7.3× the rows per second of one ([Parallelism](#parallelism)) |
| **named mistakes** | every keyword is checked against its type when the spec is built, and a missing column is reported, with the spec that wanted it and the role it had there, before the bank learns any row |
| **tested** | 1,215 Rust tests and 3,840 Python cases (counted on 2026-10-03), held to independent libraries such as scikit-learn, statsmodels and river and to adversarial streams, run on macOS, Windows and Linux at every push ([Testing](#testing)) |

### Install

Install the wheel from PyPI with pip or uv:

```sh
pip install polars-online      # or: uv add polars-online
```

| need | detail |
|---|---|
| Python | 3.12 or newer: one wheel per platform covers every CPython from 3.12 on ([Testing](#testing) lists the versions CI runs) |
| Polars | `polars>=1.34.0,<3`: the range the test suite has measured, which is not a guarantee. A weekly job and every release run the test suite on the newest Polars ([Versioning and the Polars pin](#versioning-and-the-polars-pin)) |
| depends on | nothing beyond `polars` at run time, since the wheel carries its own copy of Polars' Rust code |
| numpy | needed only by `ModelBank.gram()` and the `po.gram`, `po.corr` and `po.sim` helpers: to use them, install it with the package, as `pip install "polars-online[numpy]"`. Without it, those calls raise `ModuleNotFoundError` |
| wheels | macOS (arm64, x86_64), Windows x64, and Linux (x64 glibc and musl, aarch64 glibc), on PyPI and on each GitHub release beside the command-line binaries. On any other platform, the install builds the package from its source distribution, which needs a stable Rust toolchain ([rustup](https://rustup.rs)) |
| size | 8 to 10 MB to download and 26 to 37 MB installed, by platform, for 0.12.0 |

To build from a checkout, install uv and a stable Rust toolchain
([Development](#development)), then run from its root:

```sh
uv sync
uv run maturin develop --release -m crates/online-py/Cargo.toml
```

### Example data

Most examples after the first fit read one of two made-up frames: `df`,
400 rows of numbers and labels a minute apart, and `trades`, a stream of
quotes with trades between them. The examples that serve or resume a bank
also read `today`, the rows that follow `df`'s. Before running an example,
build all three, and the files some examples scan, with this code:

```python
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import polars_online as po

rng = np.random.default_rng(0)
df = pl.DataFrame({
    "t": np.arange(400.0),                                       # a numeric clock: the row's number
    "ts": pl.datetime_range(datetime(2024, 1, 2, 9, 30), datetime(2024, 1, 2, 16, 9), "1m", eager=True),   # a minute apart
    **{name: rng.standard_normal(400) for name in ["x0", "x1", "x2", "signal_a", "signal_b", "y", "ret"]},
    "stock_id": [f"b{i % 4}" for i in range(400)],               # four stocks, taking turns row by row
    "group": [f"g{i % 3}" for i in range(400)],
    "session": ["m"] * 200 + ["a"] * 200,                        # a morning session, then an afternoon one
    "venue": ["X", "Y"] * 200,
})
lf = df.lazy()                                                   # the same rows, as a query
today = df.with_columns(pl.col("t") + 400.0)                     # the 400 rows that follow df's, on the clock t
later = today.lazy()                                             # today, as a query
df.write_parquet("ticks.parquet")                                # df as one file, for the examples that scan one
Path("ticks").mkdir(exist_ok=True)                               # and as two, for the examples that scan ticks/*.parquet
df.head(200).write_parquet("ticks/part-0.parquet")
df.tail(200).write_parquet("ticks/part-1.parquet")

rng = np.random.default_rng(7)
is_trade = rng.random(3000) < 0.3                                # three rows in ten are trades, the rest quotes
micros = np.cumsum(rng.exponential(5e5, 3000)).astype(int)       # about two rows a second
mid = 100 + np.cumsum(rng.normal(0.0, 0.01, 3000))               # the quotes' mid, on every row
trades = pl.DataFrame({
    "ts": [datetime(2024, 1, 2, 9, 30) + timedelta(microseconds=int(u)) for u in micros],
    "symbol": rng.choice(["AAA", "BBB"], 3000),
    "mid": mid,
    "side": rng.choice(["buy", "sell"], 3000),
    "quantity": rng.integers(1, 10, 3000).astype(float),
    "price": mid + rng.normal(0.0, 0.02, 3000),
}).with_columns(
    pl.when(pl.Series(is_trade)).then(pl.col("side", "quantity", "price"))   # null on a quote
)
```

The numbers in `df` are independent random draws, so a model fitted on
them finds no relation. Run the examples for the calls and the shape of
what they return.

## How a bank sees a stream

The parameters every spec shares say how its model reads the stream, one
chunk at a time. Most models take all of them, and the
[`polars_online.spec`](https://hgilde.github.io/polars-online/spec.html)
reference gives their units and defaults.

| the question | the parameters | the subsection |
|---|---|---|
| which columns a model reads, and what its output column is called | the spec's name, `targets`, `features`, `fit_intercept` | [What a spec names](#what-a-spec-names) |
| how far apart two rows are, and how fast the past fades | `clock`, `half_life`, `gap_cap`, `session`, `session_gap`, `restart_after_step_back` | [Time and decay](#time-and-decay) |
| what a fit becomes when nothing fades | `half_life=inf`, `lam=1.0` | [Convergence without a decay](#convergence-without-a-decay) |
| when an old row stops counting at all | `window_size` | [A hard window](#a-hard-window) |
| how to fit along a feature instead of time | `clock` | [A local fit along any feature](#a-local-fit-along-any-feature) |
| which row order a bank needs, and how it checks it | | [Row order and the two guarantees](#row-order-and-the-two-guarantees) |
| which rows belong to which model | `group`, `group_close` | [Groups](#groups) |
| how much each row counts | `weight` | [Weights](#weights) |
| when a model has seen enough to report | `min_weight`, `min_settled_frac`, `max_error_inflation` | [Warm-up](#warm-up) |
| when a label may be learned | `embargo` | [Labels that arrive late](#labels-that-arrive-late) |
| what a null does, and how to keep a row from teaching a model | a null, a weight of 0, `predict` | [Nulls, and three ways to hold a row back](#nulls-and-three-ways-to-hold-a-row-back) |

With a `group`, every parameter applies within each group, the clock
included.

### What a spec names

A spec describes one model: which model it is, the columns it reads, and
how it treats time. The name of the spec (the first parameter) is used to
name the output column, for example `po.spec.ewridge("ridge", ...)` adds a
column named `ridge`. The input's columns pass through untouched, and the
bank refuses a spec named like one of them, since its output column would
replace it.
This code uses `df` from [Example data](#example-data):

```python
ridge_spec = po.spec.ewridge(
    "ridge",
    targets=["y"],                 # at least one; the targets of one spec share the features' running sums
    features=["x0", "x1", "x2"],   # numeric columns of any width, Decimal and Boolean included; read as 64-bit floats
    fit_intercept=True,            # the default: the fit has a level of its own
    clock="t", half_life=600.0, gap_cap=300.0,   # how it treats time: Time and decay, below
)
po.ModelBank([ridge_spec]).fit_predict(df).columns   # df's 13 columns, then "ridge"
```

**The bank refuses a text column in `targets` or `features`, so convert it
with Polars' `with_columns` in the query before the bank.** Use
`cast(pl.Float64)` for numbers stored as text, as the error suggests, and a
comparison such as `pl.col("venue") == "X"` for a category.

#### Relative and look-ahead targets

A target is often one value less another, such as the next five minutes'
mean trade price less the mid at the row. Build it in one of two ways, by
when its value is known. Only a model that regresses its targets takes a
target expression or
[`po.target`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.target).
Four kinds of model refuse both by name: a model that reads only its
features, `ew_class`, `seqtest`, and an `ftrl` or `sgd` fitting a
probability or a count.

**To use a target computed from the rows after its row, write it as a
Polars expression and put it in the spec's `targets`.** Its `.alias` names
the output fields, as `pred_fwd_move`. The bank learns each row once the
expression's window has closed, so give the spec an `embargo` of at least
the window's `window_size`. [Windows as a model's inputs and
target](#windows-as-a-models-inputs-and-target) gives the rule in full,
with what to do for a shorter one. This example runs on `trades`, the
quotes and trades built in [Example data](#example-data):

```python
flows = trades.lazy().with_columns(                                        # a query, and a feature in it: a trade's signed size
    flow=pl.when(pl.col("side") == "buy").then(pl.col("quantity")).otherwise(-pl.col("quantity"))
)
fwd_move = (
    po.rewm_mean("price", half_life=float("inf"), window_size="5m")       # the mean trade price over the next five minutes
    - pl.col("mid")                                                        # less the mid at the row
).alias("fwd_move")                                                        # the alias names the target
ahead_5m = po.spec.ewridge("ahead_5m", targets=[fwd_move], features=["flow"],
                           clock="ts", gap_cap="5m", half_life="30m", embargo="5m", group="symbol")
moves = flows.online.fit_predict([ahead_5m]).online.unnest([ahead_5m]).collect()   # does buying predict the price rising?
```

`pred`, `resid`, `sigma` and the metrics are on the target's scale, so
`pred + mid` predicts the mean trade price itself. `hit_rate` asks whether
prediction and outcome fall on the same side of zero ([Per-row
diagnostics](#per-row-diagnostics)), so write a ratio target to sit about
zero, as the quotient less 1 or as its `.log()`:

```python
next_5m = po.rewm_mean("price", half_life=float("inf"), window_size="5m")   # the mean trade price over the next five minutes
fwd_return = (next_5m / pl.col("mid") - 1).alias("fwd_return")            # the quotient less 1, about zero
fwd_log = (next_5m / pl.col("mid")).log().alias("fwd_log")                # the quotient's log, about zero too
ratios = po.spec.ewridge("ratios", targets=[fwd_return, fwd_log], features=["flow"],
                         clock="ts", gap_cap="5m", half_life="30m", embargo="5m", group="symbol")
by_ratio = flows.online.fit_predict([ratios]).online.unnest([ratios]).collect()   # pred_fwd_return, pred_fwd_log, ...
level = by_ratio["mid"] * (1 + by_ratio["pred_fwd_return"])                   # the mean trade price predicted, back from the ratio
```

**A target expression requires an operator that looks ahead
(`po.rewm_mean`, `po.rewm_sum` or `po.rewm_rate`); otherwise, make the
target a column with Polars' `with_columns`, chained in the query before
the bank, and name it in `targets`.** The bank's refusal names
the call that makes the column. Here the target, how far from the mid each
trade printed, is known at the trade's own row, so the spec needs no
embargo.
This code uses `trades` from [Example data](#example-data):

```python
edges = trades.lazy().with_columns(edge=(pl.col("price") - pl.col("mid")).abs())   # the target as a column, null on the quotes
by_size = po.spec.ewridge("by_size", targets=["edge"], features=["quantity"],
                          clock="ts", gap_cap="5m", half_life="30m", group="symbol")
printed = edges.online.fit_predict([by_size]).online.unnest([by_size]).collect()   # does a larger trade print farther from the mid?
```

The quotes have no `quantity`, so the bank skips them. If a column holds
a value from after its row, such as the price five minutes on, give the
spec an `embargo` of that horizon ([Labels that arrive
late](#labels-that-arrive-late)). To make a column from a window over the
stream, looking back or ahead, use `po.stream.with_windows` in the query
before the bank ([Windows as columns](#windows-as-columns)).

**To take one column against another of its row without `with_columns`,
put `po.target` in `targets`.** Here three targets take each trade's
`price` against the `mid` of its row, each on a scale of its own, and
regress it on the `flow` of the first example:

```python
vs_mid = po.spec.ewridge(
    "vs_mid", features=["flow"], clock="ts", gap_cap="5m", half_life="30m", group="symbol",
    targets=[                                                                      # name= tells apart one column taken two ways
        po.target("price", relative_to="mid", name="diff"),                       # price - mid: the default, relative="difference"
        po.target("price", relative_to="mid", relative="ratio", name="ratio"),    # price / mid, which sits about 1: its hit_rate asks which side of 1
        po.target("price", relative_to="mid", relative="log_ratio", name="log"),  # ln(price / mid), which sits about 0
    ],
)
# A null price or mid, or under a ratio one at or below 0, makes the row's target null: it is scored, not learned from.
against_mid = flows.online.fit_predict([vs_mid]).online.unnest([vs_mid]).collect()   # pred_diff, pred_ratio, pred_log: each on its own scale
level = against_mid["pred_diff"] + against_mid["mid"]                                 # the trade price predicted: pred + mid, or pred * mid for the ratio
# In the command line's TOML, each target is a table, as
# targets = [{ column = "price", relative_to = "mid", relative = "ratio", name = "ratio" }]
```

### Time and decay

A model forgets old rows by its *decay*: each row's weight halves every
`half_life` along a *clock*, a column that says how far apart two rows
are. At each row, the model multiplies the weight of everything it has
learned by `λ = 0.5 ** (Δclock / half_life)`, where `Δclock` is the
clock's step from the previous row. To fit one model per half-life, give
`half_life` a list.

#### Clock types and units

**Write every clock parameter in the clock column's own kind of value,
since a timestamp carries its unit and a number does not.**

| the clock | a clock parameter is | for example |
|---|---|---|
| a `Datetime`, `Date` or `Duration` column: a *temporal* clock | a duration | `half_life=pl.duration(minutes=10)` |
| a numeric column, such as seconds or cumulative traded volume | a number of the column's own units | `half_life=600.0` |
| none | a number of rows | `half_life=100`: a row 100 rows back counts half as much as the latest |

A duration is written three ways: `pl.duration(minutes=10)`,
`timedelta(minutes=10)`, or Polars' duration text `"10m"`.

**Every clock parameter acts on the clock, so a model's numbers do not
change with how densely the rows arrive.** The clock parameters are
`half_life`, `gap_cap`, `restart_after_step_back`, `session_gap` and
`embargo`, and a model's `window_size`, `solve_every` and its own
half-lives. A few things still count rows, and each model's page names
them.

**The bank refuses a clock parameter of the wrong kind** before it reads a
row, and its error names the column, the parameter and the fix. Only `0`
and `inf`, which mean the same in every unit, may stay numbers where a
parameter takes them:

| refused | on | because |
|---|---|---|
| a plain number | a `Datetime` clock | it would silently take the column's storage unit: `half_life=600` on a microsecond column would mean 600 microseconds |
| a duration | a numeric clock | it has nothing to measure it against |
| a spec that mixes the two kinds | any clock | |
| a duration finer than the clock can act on, such as `gap_cap="12h"` | a `Date` clock, which moves in days | |
| `lam`, a decay per clock unit | a temporal clock | it has no duration form, so a temporal clock takes `half_life` instead |

**A temporal clock is read in its own integer nanoseconds**, so it must lie
between the years 1677 and 2262, the range nanoseconds in a 64-bit integer
cover. The same instants stored in milliseconds, microseconds or
nanoseconds give the same numbers, and a time zone changes nothing. A model
reads only the gap between consecutive rows, taken in integer nanoseconds
before it becomes seconds, so a nanosecond timestamp keeps its nanoseconds
whatever the stream's age.

**Quantities measured on a temporal clock reach the output in seconds.**
`holt`'s trend is per second, and `summary()` gives the clock's range as
seconds since 1970. Only `emit_clocks` writes clocks in the clock column's
own type ([Labels that arrive late](#labels-that-arrive-late)).

#### Sessions, gaps and steps back

A stream from a market has session boundaries, gaps and steps back in its
clock. The spec has a parameter for each, and together they are the
clock's *policy*:

| the stream has | the parameter | what it does |
|---|---|---|
| a **session** boundary, such as the start of a trading day, where the clock's jump is not elapsed time | `session`, a column whose value changes at a session boundary, and `session_gap`, required with it | applies a step you choose, `session_gap`, in place of the jump, capped at `gap_cap` like any step. `session_gap="reset"` starts the model over instead |
| a **gap** in the clock, such as an hour with no rows | `gap_cap`, required with a clock, finite and above 0 | caps each step between two rows a model learns from, so a quiet hour ages the fit no more than `gap_cap` would. After a run of skipped rows, the next row's step is capped too, however long the run |
| a clock that **steps back**, such as a replayed day | `restart_after_step_back` | starts the model over at a step back larger than the setting, and refuses one no larger as a late row. Unset, the default, every step back is refused ([what each step back means](#how-a-bank-detects-rows-out-of-order)) |

**A stream has a *break* wherever `gap_cap` shortened a step, and wherever
the session changes.** A model that keeps past rows by position, as a lag
does, empties them at a break, where the previous row no longer stands for
the moment just before.

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

With decay off (`half_life=inf` or `lam=1.0`), a model that *solves* or
*accumulates* converges to the batch fit it defines over every row it has
seen. A model that *reweights*, *steps*, *filters* or *tests* depends on
the row order whether or not decay is on. Each model's kind is in [the
model table](#models), with the few settings that read the rows in
sequence even so.

**Row order does not change the fit of a model that solves or accumulates,
apart from those settings,** since both keep running sums. With no clock
column, rows fed forwards, backwards or shuffled give the same
coefficients, to rounding. With a clock column, the bank refuses a row out
of clock order ([How a bank detects rows out of
order](#how-a-bank-detects-rows-out-of-order)).

**For no forgetting, give `half_life=float("inf")`: a huge finite
half-life still forgets, a little.** Under `half_life=1e12`, `settled_frac`
stays near zero until the stream has run for about 10¹² clock units, so a
`min_settled_frac` gate ([Warm-up](#warm-up)) stays closed.

### A hard window

A half-life never forgets entirely: three half-lives back still carries
12.5% of the weight. To make every row older than `w` clock units count for
exactly nothing, give `window_size=w`. Five models take it: `ewridge`,
`lasso`, `ew_cov`, `ew_class` and `marginal`.

```
weight(age) = 0.5 ** (age / half_life)   if age <= window_size
            = 0                          otherwise
```

**Inside the window the weights still halve every `half_life`, so the
newest row counts most.** For a flat rolling fit with equal weights, give
`half_life=float("inf")` with the `window_size`.

**The cut is exact.** At time `t`, the rows learned at or before a time `u`
contribute `0.5 ** ((t − u) / half_life)` times the running sum as it stood
at `u`. Subtracting that leaves exactly the rest. The model keeps a ring
of snapshots for this, one every `window_every` rows. The edge is the
oldest snapshot inside the window, so a coarse spacing discards a little
more than asked, never less.

```python
cut = po.spec.ew_cov(
    "cut", features=["x0", "x1"], clock="t", gap_cap=300.0, half_life=500.0,
    window_size=1500.0,            # a row older than this many clock units contributes exactly nothing
    window_every=10,               # a snapshot every 10 rows
    window_budget={"refuse": 64},  # at most 64 MiB of snapshots per ring
)
```

Five things to know before reading a windowed model's numbers:

| | |
|---|---|
| the clock is the **decayed** one | a row's age adds up each step after `gap_cap` and any `session_gap` |
| the edge is a **discontinuity** | a row ageing out drops its whole weight at once, so the series has small steps that a plain exponential average does not |
| it is a **subtraction** | precision falls with the fraction discarded: negligible at a window of three half-lives, worse as the window shortens toward the half-life |
| `weight_sum` is the weight inside the window | so `min_weight` gates on a weight that settles once the window is full |
| a long step **empties** the window | a step longer than `window_size`, after `gap_cap`, empties the window, and the model reports nulls from the row after it, never stale numbers. A cap shorter than the window never empties it |

What the window truncates depends on the model:

| model | inside the window | refused beside `window_size` |
|---|---|---|
| `ewridge` | the sums the fit is solved from, and `sigma` and `zscore` with them. The coefficients are the window's as of the last solve | `ridge_scale="sum"`, `session_shrink`, `gram_block_rows` |
| `lasso` | the same, and the error that selects the penalty. A feature that stays constant inside the window has no evidence there, and goes to exactly zero | |
| `ew_cov` | every moment, and `mahal`, `partial_corr` and the principal components read from them | `lags`, `mahal_quantiles` |
| `ew_class` | each class's moments, so the classifier follows class means that move | |
| `marginal` | every pair's weight, means and second moments, so `corr`, `beta` and `t` describe the window. Its lag moments, under `window_lags=True`, are an estimate rather than exact | `bins`, `feature_moments="shared"`, and `lags` unless `window_lags=True` |

**`window_budget` caps the snapshots' memory, per ring, in MiB.** The
ring's memory grows with the window: for an `ew_cov` over 20 columns, a
1,000-row window holds about 3 MB per group, divided by `window_every`. A
refusal comes before the bank learns any of the chunk, except for two
overruns found only as the rows go in. One is under
`drift_action="reset"`. The other is a snapshot that grows within the
chunk, as an `ewridge`'s or a `lasso`'s does when its targets first go
missing on different rows. After either, the bank refuses every call that
follows, so rebuild it from its last save with `po.ModelBank.load`.

| `window_budget` | when a chunk would take a ring past the cap |
|---|---|
| not given | the chunk is refused past 256 MiB |
| `{"refuse": 64}` | the chunk is refused before any of it is learned, and the bank goes on as it was. The error names the ring's size, `window_every` and the ways to lift the cap |
| `{"thin": 64}` | the ring drops every other snapshot and doubles its spacing, which, like `window_every`, only ever shortens the window |
| `{"refuse": float("inf")}` | nothing: no cap |

### A local fit along any feature

To fit any model locally along a feature, sort the frame by that feature
with Polars' `sort` before the bank, and name the feature as the spec's
`clock`. `half_life` is then a bandwidth in that feature's units. Each row
is fitted on the rows before it in that order, each weighted by
`0.5 ** (Δx / half_life)`, with `Δx` its distance from the row in that
feature: an exponential kernel.
This code uses `df` from [Example data](#example-data):

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

The fit is a local regression from running sums that do not grow, and
agrees to 1e-12 with a kernel-weighted least squares recomputed from
scratch at every row.

**The kernel is one-sided, so the fit lags.** Narrow the bandwidth to lag
less, and widen it to smooth the noise. On `sin(x)` at a bandwidth of 0.25,
the fit sits 0.08 from the truth, where the best straight line sits 0.39.
At a bandwidth of 1.0 it sits 0.29, most of the way back to the line.

**The clock column can stay out of `features`.** With other columns as
features, the fit is a regression whose coefficients move along the clock.

### Row order and the two guarantees

Two guarantees hold for every run, and both rest on a fixed row order,
which the caller supplies.

- **Every prediction and diagnostic is out-of-sample.** Each row is scored
  from the state as it stood before the row's own target was learned, so
  no row's outcome reaches its own prediction or the diagnostics measured
  on it.
- **Chunk invariance.** One chunk or a thousand, with or without a save and
  resume in the middle, gives bit-identical numbers. Only which rows carry
  `coef`, and `support_coef` beside it, can differ. The bank writes them
  every `coef_every` rows ([Coefficients](#coefficients)) and on each
  group's last row in every chunk, so smaller chunks report them more
  often.

**A model learns in row order, so a query whose row order Polars does not
guarantee can give a different model each time it runs.** Run a chunk at a
time, as a bank runs it, a `join`, a `group_by` or a `unique` can deliver
rows in another order than `lf.collect()` gives. Give each such step its
guarantee in the query:

| a query step that may reorder rows | give it |
|---|---|
| `join` | `maintain_order="left"` |
| `group_by` | `maintain_order=True` |
| `unique` | a sort after it |
| anything else | a sort before the bank |

#### How a bank detects rows out of order

A bank checks the order it is given in three places, warning about the
query and refusing a chunk that runs backwards before learning any of it.

| check | what it reads | when it finds disorder |
|---|---|---|
| the query | the query handed to the bank: a `join`, `group_by` or `unique` whose order Polars does not guarantee, unless a `sort` follows it, and a sort by several keys without `maintain_order=True` | raises `OrderNotGuaranteedWarning`, naming the step and its fix. The run goes on |
| the clock | each group's clock, row by row, for every spec with a `clock` | refuses the chunk on a backwards step the settings below do not allow |
| the group keys | with `group_close="monotone"`, the keys in the column's own order, an integer column as numbers and any other as text | refuses the chunk on a key below the one before it, since a closed group cannot reopen |

**The query check is best-effort.** It reads the query's `explain()` text,
and walks the query's steps only when that names a join, an aggregation, a
`unique` or a `sort`. A step followed by a `sort` counts as
ordered, unless that sort is by several keys without `maintain_order=True`.
Such a sort leaves rows with equal keys in no particular order: 2,109 of
10,000 rows moved when measured. A join with `maintain_order="left"` is
followed on its left side only. A query Polars cannot serialize passes
without a warning, and the check never fails a run. For a query whose order
you know, silence the warning with
`warnings.simplefilter("ignore", po.OrderNotGuaranteedWarning)`.

**`ModelBank.fit` does not warn when its state cannot depend on the row
order.** That is when every spec is an `ewridge` or `rls` with decay off
and none of the settings that read the order. Those include a window, a
session, a weight, an embargo, a drift reset, `coef_every`, a window
expression as a target, and any diagnostic but `emit_drift`, `emit_clocks`
and `emit_error_inflation`. Such sums reach the same state in any order, to
rounding, and `fit` keeps only the state.

**The clock is checked group by group,** so groups may interleave freely:
only the rows of one group need to be in clock order. Equal clock values
are zero apart. A row with a null feature is skipped, but its clock is
still checked. A row that `predict` scores is never refused ([Serving without
learning](#serving-without-learning)).

**`restart_after_step_back` and the size of a step back decide what it
means.** It has no default, since only the caller knows how late a row can
be, and `0` restarts the model at every step back:

| a step back | is read as | and the bank |
|---|---|---|
| any size, with `restart_after_step_back` unset, the default | rows out of order | refuses the chunk |
| no larger than `restart_after_step_back` | a late row, such as a transposed pair or a row a minute late | refuses the chunk |
| larger than `restart_after_step_back` | a new start, such as a replayed day or a restarted feed | restarts the model |
| any size, on a row whose `session` value changes | a new session, where the step is no measure of time | applies `session_gap`, or restarts the model under `session_gap="reset"` |

**The summary counts what the clock rules met.** `bank.summary()` reports
`clock_backwards`, the rows whose clock fell below the previous row's
within a session, and `resets`, the rows where a stream restarted.

**A refused chunk leaves the bank as it was.** The bank runs the clock of
every group of every spec over the chunk before it learns any row. So one
late row in one group changes nothing: correct the chunk and feed it again.
To resume a saved state on input that overlaps it, pass the input through
`bank.skip_learned(...)`, which drops the rows the state has learned ([Save
and load](#save-and-load)).

**The error locates the step and names the way out.** The row it names is
counted from the start of the input however it arrives (one frame,
batches, a query or the CLI's file). On a temporal clock the step is a
duration.

```text
spec "m": clock column "t" goes backwards by 30 at row 6 (restart_after_step_back is unset, so
every step back is refused); the bank was not updated. Sort each group by the clock; to resume a
saved state on input that overlaps it, drop the rows it has learned, with
ModelBank.skip_learned(frame) in Python or by filtering the command line's input to the rows after
them; or, if a step back this large starts the stream over, set restart_after_step_back to the
smallest one that does.
```

### Groups

To fit one model per value of a column, name the column with `group`: the
bank fits a separate model for each distinct value, in the same pass.
`group_close` says when a group is finished, which keeps a bank's memory
bounded when new group values never stop appearing.

```python
per_stock = po.spec.ewridge(
    "per_stock", targets=["y"], features=["x0", "x1"], clock="t", half_life=600.0, gap_cap=300.0,
    group="stock_id",           # one model per distinct value of this column
    group_close="monotone",     # or "session": when a group is finished, write it out and free its memory
)
```

**Without `group_close`, the bank keeps every group it has seen.** To free
the groups that have gone quiet, find them with `bank.groups()` and drop
them with `bank.drop_groups(...)` ([In a loop](#in-a-loop-modelbank)).

**With `group_close`, the bank writes out a finished group's running sums
and frees its memory.** `"monotone"` finishes a group once a higher key
arrives, and `"session"` once its `session` value changes. [One row per
finished group](#one-row-per-finished-group) shows what is written.

### Weights

To make some rows count more than others, name a column of row weights
with `weight`.

```python
weighted = po.spec.ewridge(
    "weighted", targets=["y"], features=["x0", "x1"], clock="t", half_life=600.0, gap_cap=300.0,
    weight="w",                # a column of row weights: here a row of weight 2 counts as two rows of weight 1
)
```

**A weight of 0 is legal, and a negative weight is refused.** A row of
weight 0 is scored and moves the clock, but teaches the model nothing
([Nulls, and three ways to hold a row back](#nulls-and-three-ways-to-hold-a-row-back)).

**In a model that keeps means, scaling every weight by the same factor
changes nothing but `weight_sum`, so scale `min_weight` by the same
factor.**

**Eight models read a weight differently**, and
`tests/test_weight_scale.py` holds every other model to that rule:

| model | a row's weight is |
|---|---|
| `rls` | on the sum scale, against a ridge of fixed size: a heavier stream outweighs it sooner |
| `kalman` | a scale on the observation's precision: the update divides the observation noise `R_j` by `w`, so with a fixed `obs_var=` the observation's variance is `obs_var / w` |
| `sgd` | a step size: the gradient carries it |
| `ftrl` | an importance weight, as Vowpal Wabbit's, against penalties of fixed size |
| `pa` | a step size below 1; a weight above 1 counts as 1 |
| `hmm` | on the sum scale, against the transitions' prior count |
| `rcov` | 0 or 1, since a realised covariance is a sum over returns; any other weight is refused |
| `seqtest` | refused, since every learned row is one trial: to skip a row, null its target |

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

**`weight_sum` means the same in every model, so one `min_weight` means
the same for every spec in a bank.** It is the *weight behind the state*
that produced a row's prediction: the weight as the previous row left it,
before this row's decay and its update, and `0` on a stream's first row. It
runs one behind the row count while nothing is forgotten. For rows of
weight 1 a step `Δclock` apart, it settles at `1 / (1 − λ)`, with `λ` from
[Time and decay](#time-and-decay). It is a weight and not a sample size: at
a half-life of 600 with rows 0.1 apart it settles near 8,657. For the
sample size, read Kish's `n_kish` from `gram()`.

**Every output is null until the model's weight reaches `min_weight`.** A
regression model checks each target against that target's own weight, the
weight of the rows it was present on, so an often-null target reports
later than the others. For `rls`, which learns a row only when every target
is present, that is the weight of the rows it learned from. The
`weight_sum` field is the shared weight either way. To set one threshold
per target, give `min_weight` a list. The default depends on the model:

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

**`max_error_inflation` follows the size of the fit:** add a feature, and
the gate holds predictions back longer. It reads Kish's count, so one row
carrying a hundred times the weight of the others counts as barely one.

**Each row says how ready its model was.** `summary()` carries the same
readings per group, and a `ReadinessWarning` names, once, a coefficient
more ridge than data or a noise gate the settled stream can no longer meet.
This code uses `df` from [Example data](#example-data):

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

The readiness fields `ewridge` writes:

| field | what it holds |
|---|---|
| `settled_frac` | how full the decay window is |
| `withheld_reason` | why `pred_y` is null, or null when it is not: `below_min_settled_frac`, `below_min_weight` or `above_max_error_inflation`, in that order of precedence |
| `error_inflation_y` | the row's leverage against the fit: high on a row leaning on a direction the data never showed, at one triangular solve a row |
| `support_coef` | beside `coef`: each coefficient's data share, `1 - ridge * (S^-1)_jj`. A duplicated pair of features reads 0.5 each |

### Labels that arrive late

Some labels arrive after their row: the next five minutes' return is known
five minutes after its row. A model that learns it at its own row sees
that much of the future before it predicts the rows in between. Every
out-of-sample number after that is contaminated, and with a feature
correlated with its own past, even a pure-noise column starts to look
predictive. To learn each label only once it would have arrived, give the
spec an `embargo`. To write the same delay out as rows,
use `po.stream.embargo` in the query before the bank.

#### The built-in delay

Give a spec `embargo="5m"` (any finite delay above 0), and the bank scores
each row where it sits and learns from it five minutes later on the clock.
Everything computed from the labels sees only those that had arrived: the
prediction, `sigma`, `zscore`, the metrics, break detection, the conformal
interval, `weight_sum` and the `min_weight` gate.

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
| at the end of the stream | the rows still waiting are not learned, unless the state is saved and another run resumes from it |
| in the state | the waiting rows are saved with it: one row's values for each row inside the delay, in each group |
| row by row | `emit_clocks=True`, on any model, writes `scored_clock`, the row's own clock, and `learned_clock`, the clock of the newest row the model had learned from when it was scored, both in the clock column's own type. With no clock they are the row's index in its group |

#### The delay as data

Use
[`po.stream.embargo`](https://hgilde.github.io/polars-online/stream.html#polars_online.stream.embargo)
when the delay has to be visible in the frame, or when another engine fits
the model. It returns every row twice: a copy to score at `t` with weight
0, and a copy to learn from at `t + delay`.

**Sort its input by the clock across all rows, as
`lf.sort("t", maintain_order=True)`, before the call.** Order within each
group is not enough, since the call merges the two copies by the clock
alone. It adds two columns: `_online_role`, which says which copy a row is,
and `_online_role_weight`, which you pass to the spec's `weight`.
This code uses `lf` from [Example data](#example-data):

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

**Every residual diagnostic differs in any case**, from `sigma` and
`zscore` to the conformal band and drift. `embargo` takes in the residual
of the prediction the row was scored with. The doubled stream's learning
copy forms its residual at `t + delay`, from a model that has since learned
every row before it.

### Nulls, and three ways to hold a row back

A null skips an update but never stops the clock. NaN, ±inf and any
magnitude above `1e100` count as null, so sentinel values never reach a
model. A null target is one of three ways to keep a row from teaching a
model, beside a weight of 0 and `predict`:

| | the row is scored | the clock advances | the fit moves | `weight_sum` | use it to |
|---|---|---|---|---|---|
| **a null target** | yes | yes | not that target's, except under `target_gaps="pairwise"` | counts the row. In a regression model the target's own weight only decays, so a long stretch of nulls can take that target below its `min_weight` | leave a label out |
| **weight `0`** | yes | yes | no | keeps decaying, so a long stretch can fall below `min_weight` | keep a row's place in the stream |
| **`predict`** | yes | no | no | frozen | serve ([Serving without learning](#serving-without-learning)) |

**A null in any feature, or in the weight, skips the row**: its outputs are
null, and no update happens.

**A null in one target skips only that target's update.** The row still
writes that target's `pred`, and leaves its `resid` null. In `rls`, whose
targets share one factor, the row updates none of them. In `ewridge` and
`lasso`, whose targets share running sums, the row still reaches the other
targets' sums. Under the default `target_gaps="own_rows"`, the missing
target keeps its own copy of the sums from that row on, and over the rows
it lacks the copy only ages, so its fit holds still. Under `"pairwise"` one
set of sums learns every row, so a null target's
coefficients drift with the feature noise.

## Preparing a stream

Two tools turn a stream's raw rows into the columns a model reads. Each
reads the rows in one pass, returns a `DataFrame` for a `DataFrame` and a
query for a query, and can save its state with `save_state=` for the next
run:

| tool | what it makes | where it runs | subsection |
|---|---|---|---|
| the window operators of [`po.ops`](https://hgilde.github.io/polars-online/ops.html) | exponentially weighted means, sums and rates over a window of the clock, looking back or ahead | in [`po.stream.with_windows`](https://hgilde.github.io/polars-online/stream.html#polars_online.stream.with_windows), before the bank, as columns; or in a spec's `targets`, as a target the bank computes | [Windowed means, looking back or ahead](#windowed-means-looking-back-or-ahead) |
| [`po.stream.refresh_time`](https://hgilde.github.io/polars-online/stream.html#polars_online.stream.refresh_time) | one grid for series that tick at their own times | before the bank | [Series that tick at their own times](#series-that-tick-at-their-own-times) |

### Windowed means, looking back or ahead

Each operator of [`po.ops`](https://hgilde.github.io/polars-online/ops.html)
returns a Polars expression. The examples run on `trades`, the quotes and
trades built in [Example data](#example-data). The operators follow one set
of rules in each of three uses:

| use | the call | subsection |
|---|---|---|
| any number of formulas as new columns, before the bank, as `with_columns` adds columns to a frame | [`po.stream.with_windows`](https://hgilde.github.io/polars-online/stream.html#polars_online.stream.with_windows) | [Windows as columns](#windows-as-columns) |
| a look-ahead as the target a model learns, its windows computed inside the bank | a spec's `targets` | [Windows as a model's inputs and target](#windows-as-a-models-inputs-and-target) |
| a `with_windows` run fed in parts, each resuming where the one before it stopped | `save_state=`, then `load_state=` | [Saving and resuming a window run](#saving-and-resuming-a-window-run) |

Seven operators make the expressions. Each takes the same work on a row
however long its window, while Polars' `rolling` gathers a window's rows
again for every row ([Window operators](#window-operators)):

| operator | the window of row *t* | what it computes |
|---|---|---|
| `po.ewm_mean(x, half_life=, window_size=)` | the rows at or before *t*, less than `window_size` older; with no `window_size`, every row since the last break | the time-weighted mean, as Polars' `ewm_mean_by` computes it: each value is held over the interval ending at its row and weighed by that interval's decayed time, so a burst of rows does not outweigh a quiet period. A row whose `x` is null is skipped, as `ewm_mean_by` skips a null |
| `po.rewm_mean(x, half_life=, window_size=)` | the rows after *t*, at most `window_size` after it; `window_size` is required | the mirror of `ewm_mean`: each value is held until the next row |
| `po.ewm_sum(x, ...)`, `po.rewm_sum(x, ...)` | as for the mean in the same direction | `Σ 0.5 ** (age / half_life) · x`, each row counted once at its own time, as Polars' `ewm_sum_by` computes it. A row whose `x` is null adds nothing |
| `po.ewm_rate(x, ...)`, `po.rewm_rate(x, ...)` | as for the mean in the same direction | the sum divided by the decayed time the window covers: a quantity per unit of clock. A row whose `x` is null adds nothing |
| `po.increment(x)` | back to the last row with a value, within the group and session | `x_t − x_{t−1}`: null on a session's first row and after a restart, and in seconds when `x` is a temporal column |

Every operator except `po.increment` takes these keywords, `half_life` and
`window_size` in the clock's units.
This code uses `trades` from [Example data](#example-data):

```python
mid_mean = po.ewm_mean(
    "mid",
    half_life="5s",     # how fast the weights fall: largest at the row, halving every 5 s away from it; float("inf") weighs the window evenly
    window_size="1m",   # how far the window reaches from the row; an evenly weighted window needs it, and so does one looking ahead
    closed="right",     # which ends of the window are in: "right", the default
    min_samples=3,      # the fewest rows with a value the window must hold; fewer give null
    partial="keep",     # what a window gives when a gap or a session change cuts it short
)
windowed_mid = trades.lazy().online.with_windows(mid_mean=mid_mean, clock="ts", gap_cap="5m", group="symbol").collect()
```

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

**Around the operators, write element-wise Polars,** since the formula is
evaluated one chunk at a time:

| | in a formula |
|---|---|
| allowed, around at least one operator | columns, literals, arithmetic, negation, comparisons, `log`, `exp`, `abs`, `sqrt`, `pow`, `clip`, `fill_null`, `is_null`, `is_not_null`, `when/then/otherwise`, `cast` and `alias` |
| refused by name when the query is built, since each would depend on where a chunk ended | a `shift`, a cumulative or rolling function, `over` or an aggregation |
| refused by name when the query is built: write it with Polars' own `with_columns` | a formula with no operator |

**An operator's input can be a formula of the same kind, `po.increment`
included, but not another operator.** So to window an operator's output,
make it a column in one `with_windows` call, and window that column in a
second. Here a rate of traded volume comes from a column of cumulative
volume.
This code uses `trades` from [Example data](#example-data):

```python
clock = dict(clock="ts", gap_cap="5m", group="symbol")
volume = (
    trades.lazy()
    .with_columns(cum_volume=pl.col("quantity").cum_sum().over("symbol"))   # a running volume, as a feed prints it: null on the quotes
    .online.with_windows(                                                    # the first call: an operator over an increment
        volume_rate=po.ewm_rate(po.increment("cum_volume"), half_life="30s", window_size="5m"),   # the volume traded per second
        **clock,
    )
    .online.with_windows(                                                    # the second call: that rate, windowed again
        rate_trend=po.ewm_mean("volume_rate", half_life="2m", window_size="10m"),
        **clock,
    )
    .collect()
)
```

**The operators read a spec's clock keywords:** `clock`, `gap_cap`,
`restart_after_step_back`, `session`, `session_gap` and `group`. Pass them
to `with_windows` once, for every formula in the call. A window target
reads them from its spec. With `group`, each group keeps its own sessions.

**A gap longer than `gap_cap`, or a session change, ends every window open
across it, and the operator's `partial` sets what the cut window gives.**
A restart (a step back larger than `restart_after_step_back`, or a session
change under `session_gap="reset"`) discards the open windows instead, so
their rows come out null, never dropped:

| `partial` | a cut window gives | for a window target, the row is | the default for |
|---|---|---|---|
| `"keep"` | the value over what it saw, the window ending at the last row seen | learned from what the window saw | looking back |
| `"null"` | null | not learned | looking ahead |
| `"drop"` | nothing: the row leaves the output | not learned, as under `"null"`, while its other targets still are | |

#### Windows as columns

Call `po.stream.with_windows(frame, name=formula, ...)` before the bank. It
adds one column per formula after the input's columns, named by its keyword
or its `.alias()`. In a chain, write it as `lf.online.with_windows(...)` on
a query or `df.online.with_windows(...)` on a `DataFrame`.

For a weighted mean such as a VWAP (volume-weighted average price), divide
one decayed sum by another: decayed notional (price times quantity) by
decayed volume, to which the quotes add nothing. For one side's VWAP, put
the same `when/then` inside both sums.
This code uses `trades` from [Example data](#example-data):

```python
notional = pl.col("price") * pl.col("quantity")
buys = pl.when(pl.col("side") == "buy")
next_minute = dict(half_life="10s", window_size="1m")
windowed = po.stream.with_windows(
    trades,
    mid_trend=po.ewm_mean("mid", half_life="5s", window_size="1m") - pl.col("mid"),  # the last minute's time-weighted mid, less the mid now
    fwd_vwap=po.rewm_sum(notional, **next_minute) / po.rewm_sum("quantity", **next_minute),      # the next minute's trades, weighted most on the next one
    fwd_buy_vwap=po.rewm_sum(buys.then(notional), **next_minute) / po.rewm_sum(buys.then("quantity"), **next_minute),   # the same, buys only
    clock="ts", gap_cap="5m", group="symbol",
)
# every input column, then mid_trend, fwd_vwap, fwd_buy_vwap

(trades.lazy()
    .online.with_windows(fwd_vwap=po.rewm_sum(notional, **next_minute) / po.rewm_sum("quantity", **next_minute),
                         clock="ts", gap_cap="5m", group="symbol")
    .sink_parquet("with_windows.parquet"))     # streams, holding about one window of rows
```

**Feed `with_windows` the rows in clock order across all groups, as one
stream.** Besides each group's clock, it reads the clock across groups. On
that clock, a step back is refused, naming the row, unless it is larger
than `restart_after_step_back`, which restarts every group. A gap past
`gap_cap` on it ends every group's windows. So sort an input ordered by
group with `sort("ts", maintain_order=True)` before the call. A query whose
row order Polars does not guarantee draws an `OrderNotGuaranteedWarning`.

**`with_windows` returns the rows in input order, each once every window
over it has its value,** so the output trails the input by the longest
window. A group that falls silent holds back the rows after it for at most
`gap_cap` of the stream's time.

**Two things keep a row out of the output: `partial="drop"`, and
`save_state=`, which saves for the next run the rows still waiting for a
window when the input ends.** Without `save_state=`, those rows come out
with their open windows null.

#### Windows as a model's inputs and target

To give a model a feature that looks back, make it a column with
`with_windows` in the query, chain `.online.fit_predict` after it, and name
the column in the spec's `features`. To make a window that looks ahead the
target, put its expression in the spec's `targets`. The bank then computes
the window itself, one set per group on the spec's clock, so the groups may
interleave in time.

**Give a target expression an operator that looks ahead (`po.rewm_mean`,
`po.rewm_sum` or `po.rewm_rate`) and a name, with `.alias()`.** The alias
is used to name the output fields: `.alias("fwd_edge")` gives
`pred_fwd_edge`. Without an operator that looks ahead, the target is known
at its own row, and the spec builder raises `ValueError`. Make such a
target a column in the query before the bank instead, and name the column
in `targets`.
Make it with `with_windows` if it reads a window that looks back, or with
Polars' `with_columns` if it reads only its own row. The builder refuses
`group_close` beside a window target.
This code uses `trades` from [Example data](#example-data):

```python
clock = dict(clock="ts", gap_cap="5m", group="symbol")    # one clock for the windows and the model
trends = trades.lazy().online.with_windows(               # the features: windows that look back
    trend_5s=pl.col("mid") - po.ewm_mean("mid", half_life="5s", window_size="2m"),
    trend_30s=pl.col("mid") - po.ewm_mean("mid", half_life="30s", window_size="2m"),
    **clock,
)
notional = pl.col("price") * pl.col("quantity")
next_minute = dict(half_life="10s", window_size="1m")
fwd_edge = (po.rewm_sum(notional, **next_minute) / po.rewm_sum("quantity", **next_minute) - pl.col("mid")).alias("fwd_edge")   # the target: the next minute's VWAP, less the mid
edge = po.spec.ewridge("edge", targets=[fwd_edge], features=["trend_5s", "trend_30s"],
                       half_life="30m", embargo="1m", **clock)   # each row learned once the next minute is known
fitted = trends.online.fit_predict([edge]).collect()      # trades, the two trends, and the column edge
as_column = trends.online.with_windows(fwd_edge, like=edge).collect()   # the target the model learned, as a column
by_column = po.spec.ewridge("by_column", targets=["fwd_edge"], features=["trend_5s", "trend_30s"],
                            half_life="30m", embargo="1m", **clock)     # the same model, its target named as that column
column_form = as_column.online.fit_predict([by_column])                 # the column form: the same predictions as fitted's
```

**Each row is scored where it sits, and learned from once its window has
closed and its `embargo` has passed.** A state that learned a row before
its window closed would score the rows that window covers in sample. So
`fit_predict` requires an `embargo` of at least the target's
`window_size`; otherwise it raises `ValueError` before reading a row. For a
shorter embargo, or none, use `bank.fit` or the command line's
`--no-output`, which keep no predictions.

**`resid_<name>` is null on every row (`resid_fwd_edge` here), because a
row's target is not known when the row is scored.** The diagnostics take
each row in when it is learned, as under any `embargo` ([Labels that arrive
late](#labels-that-arrive-late)).

**To see the targets the model learned, write the target as a column with
`with_windows(fwd_edge, like=edge)`, on a frame that holds the spec's
features, as `trends` does.** `like=edge` takes the clock keywords from
`edge`, and refuses any given beside it. It nulls the target on every row
the spec would not learn from.

**Named in `targets` under the same `embargo`, that column gives the same
predictions row for row, except when the embargo equals the window under
`closed="right"`, as in the example.** Then the column form learns each row
at the row exactly one window after it, and the expression in `targets` at
the next distinct stamp after that.

#### Saving and resuming a window run

To feed `with_windows` a stream in two runs, give the first `save_state=`
and the second `load_state=`. The second run returns the rows the first
left waiting before its own, so two runs give what one run gives. An input
that starts before the last row the first run read is refused as a step
back. A state loads only into a call with the same formulas, clock keywords
and kind of clock; otherwise `load_state=` raises `ValueError`. A window
target is saved with its bank's state instead ([Saving, loading and
serving](#saving-loading-and-serving)).

Under a slice such as `head(500)`, the run reads the input only until the
windows of the last row asked for have closed, and the state records how
many input rows it consumed.
This code uses `trades` from [Example data](#example-data):

```python
next_minute = dict(half_life="10s", window_size="1m")
vwap = dict(fwd_vwap=po.rewm_sum(pl.col("price") * pl.col("quantity"), **next_minute) / po.rewm_sum("quantity", **next_minute))
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

**A state saved under a slice records its input's first row (clock and
session) and the last rows it read, so it refuses the wrong input, never
silently skipping or doubling rows.** If the clock starts over at the same
stamp each day, add a session column, such as the date, and name it in
`session=` with a `session_gap`. Otherwise the next day's file starts as
the saved input did, and passes for it. If the stream's rows can repeat, as
a daily grid's can, resume only on the same input: another input that
matches the state's rows where it was cut is taken for the same input.

**A run resumed from a state saved under a slice decides from its input's
first row whether to skip rows, go on or refuse.** A refusal arrives as
`polars.exceptions.ComputeError`, with a message naming `another input`:

| the input a resumed run is given | the run |
|---|---|
| the input the state was saved from, unsliced | skips the rows the state consumed, then goes on |
| the next file, whose first row is a step forward on the clock or a new start (a step back past `restart_after_step_back`, or a new session) | skips nothing, and returns the rows the state held before its own |
| an input whose first row is at the last stamp the state read | refused: a file boundary inside a stamp that several rows share looks the same as the saved input sliced inside it |
| an input that steps back where the clock policy refuses it, such as the same input sliced by hand, or an overlapping file | refused |
| without a clock column, an input that does not begin with a new session | refused: a row-count clock steps forward at every row |
| an input that starts as the saved one did but differs where the state was cut, or ends before the rows consumed | refused |

### Series that tick at their own times

Series observed at different instants cannot be compared row by row: a fine
common grid pushes their correlation toward zero (the Epps effect), and
filling values forward invents observations. So put them on one grid with
[`po.stream.refresh_time`](https://hgilde.github.io/polars-online/stream.html#polars_online.stream.refresh_time)
before the bank. Its grid, which Barndorff-Nielsen, Hansen, Lunde and
Shephard defined, has a point at each *refresh time*: the first instant by
which **every** series has ticked at least once since the previous point.
Nothing is interpolated.

Give it the ticks in long form, one row per tick. Three series on a clock
`t`:

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

The first refresh time is 1.6, when CCC ticks for the first time. AAA's
value there is 100.2, from its tick at 1.3, and its tick at 0.4 is dropped.
After 1.6, BBB ticks at 2.2 and AAA twice, so CCC's tick at 3.1 completes
the second point.

Each column of the grid holds:

| column | what it holds |
|---|---|
| `time_refresh` | the refresh time: the clock of the tick that completed the point |
| `AAA_value`, ... | each series' last observed value at that time. The output looks synchronous and is not: each value is up to one of its own inter-tick intervals old |
| `n_obs_AAA`, ... | that series' ticks since the previous point, of which the grid kept the last |
| `retained_fraction` | the share of those ticks the grid kept. Read it before trusting a correlation |

**The grid runs at the pace of the slowest series, so a fast series loses
most of its ticks.** The series holding the grid up, whose tick completes
each point, keeps an `n_obs` near 1. When one series is much slower than
the others, pass `pairs=True` for an independent two-series grid per pair,
which keeps far more ticks.

**Sort the ticks by the clock within each `group` before the call:** a
step back stops the run with an error naming the row.

**To build the grid in two runs, give the first `save_state=` and the
second `load_state=`, with the same `names` and `pairs`.** The second run
goes on from the first one's last tick, even part-way through an interval.
On the eight `ticks` of the first grid:

```python
names = ["AAA", "BBB", "CCC"]
first = po.stream.refresh_time(ticks.head(5), series="symbol", names=names, clock="t", value="px",
                               save_state="refresh.state")   # the ticks to 2.2: part-way through the second interval
rest = po.stream.refresh_time(ticks.tail(3), series="symbol", names=names, clock="t", value="px",
                              load_state="refresh.state")    # the other three, from where the first run stopped
assert pl.concat([first, rest]).equals(                      # the two grids together are the grid of one run
    po.stream.refresh_time(ticks, series="symbol", names=names, clock="t", value="px")
)
```

**To fit models on the grid, compute each series' change between points
with Polars' `diff()`, and give the specs the grid's clock, `time_refresh`.**
Given a query, `refresh_time` returns one, so
[`lf.online.fit_predict(specs)`](#as-a-query-lfonlinefit_predict) chained
after it runs the grid and the models in one pass. On a hundred ticks of
each series, `ew_cov` tracks the three correlations, and `ewridge`
regresses one series' change on the other two's:

```python
rng = np.random.default_rng(1)
ticks = pl.concat(                                 # a hundred ticks of each series, at times of its own
    pl.DataFrame({"symbol": [s] * 100, "t": np.cumsum(rng.exponential(1.0, 100)), "px": rng.standard_normal(100)})
    for s in ["AAA", "BBB", "CCC"]
).sort("t")                                        # one stream, in clock order
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

Run a bank in one of three ways, each giving the same numbers. From a bank
object, the output can leave as Arrow, for a consumer that is not Polars:

| way | the call | use it for | where |
|---|---|---|---|
| inside a Polars query | `lf.online.fit_predict(specs)` | ordinary Polars steps chained after the bank | [As a query](#as-a-query-lfonlinefit_predict) |
| in your own Python loop | `bank.fit_predict(chunk)` | reading the bank between chunks | [In a loop](#in-a-loop-modelbank) |
| from a standalone command line, with no live Python at all | `online` | a scheduled job, or a deployment with no Python | [Outside a live Python process](#outside-a-live-python-process) |
| output as Arrow, from a bank object | `fit_predict_arrow`, `predict_arrow` | a consumer that is not Polars | [Output as Arrow](#output-as-arrow) |

### As a query: `lf.online.fit_predict`

[`lf.online.fit_predict(specs)`](https://hgilde.github.io/polars-online/namespaces.html#polars_online._frame.LazyFrameOnlineNamespace.fit_predict)
adds a model bank to a Polars query, a `LazyFrame`, which runs only when
you ask for its result. `df.online.fit_predict(specs)` does the same for a
`DataFrame` in memory. When the query runs, it feeds its rows to a bank
that starts with nothing learned, `chunk_rows` rows at a time (100,000 by
default). Run it with `collect()` for one frame, `sink_parquet()` to write
a file without holding the result in memory, or `collect_batches()` for one
chunk at a time.
This code uses `lf` and the files `ticks/*.parquet` from [Example data](#example-data):

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

**Filter after the bank, not before it.** A `filter` after the bank keeps
the query running a chunk at a time. One before the bank makes Polars hold
several blocks of each parquet file per thread: 2.5 GB at 12M rows,
against 0.78 GB for the same filter after
([docs/PERFORMANCE.md](docs/PERFORMANCE.md) §11).

**To keep rows from teaching the model, give them weight `0` before the
bank, in place of a filter:** make the column with `with_columns`, and name
it in the spec's `weight`. The rows still come out scored, and the clock
advances through them, so no gap opens where they were.
This code uses `lf` from [Example data](#example-data):

```python
(
    lf.with_columns(pl.when(pl.col("venue") == "X").then(1.0).otherwise(0.0).alias("w"))  # learn from venue X only
    .online.fit_predict([po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"],
                                         clock="t", half_life=600.0, gap_cap=300.0,
                                         weight="w")])
    .sink_parquet("fitted.parquet")
)
```

**`lf.online.fit_predict` checks each spec against the query's columns when
it is called, so a missing or non-numeric column raises `ValueError` before
any row is read.** The bank refuses what only the values reveal, such as a
null clock or a step back, while the query runs. The error arrives as
`polars.exceptions.ComputeError`, its message naming the row and the way
out.

**With a type checker, call `po.fit_predict`,
[`po.predict`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.predict)
and [`po.unnest`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.unnest)
in place of the `.online` methods.** The package adds those methods to
Polars' frames when it is imported, so a type checker cannot see them. Each
function takes the frame first.
This code uses `lf` and `later` from [Example data](#example-data):

```python
fitted = po.fit_predict(lf, [spec], save_state="ridge.state")   # lf.online.fit_predict([spec], save_state="ridge.state")
columns = po.unnest(fitted, [spec]).collect()                   # fitted.online.unnest([spec]); running it saves the state
scored = po.predict(later, "ridge.state").collect()             # later.online.predict("ridge.state")
```

### In a loop: `ModelBank`

To feed a bank from your own Python loop, build a
[`ModelBank`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.ModelBank)
from the specs, pass it each chunk with `bank.fit_predict(chunk)`, and ask
it what it holds between chunks ([What a bank holds](#what-a-bank-holds)).
This code uses `lf` from [Example data](#example-data):

```python
bank = po.ModelBank([spec])                # the spec above, with nothing learned yet

for chunk in lf.collect_batches():        # the query, one chunk at a time: the whole stream is never in memory
    out = bank.fit_predict(chunk)         # the chunk's columns, plus one column per spec
    ...

bank.save("bank.state")                    # written whole or not at all: a temporary file, then a rename

# Or let the bank chunk: a query or a DataFrame in chunk_rows rows, an iterator of frames as it comes.
for out in po.ModelBank([spec]).fit_predict_batches(lf, chunk_rows=100_000):
    ...

# A run whose product is the state: learn from every row, keep no output.
state_only = po.ModelBank([spec])
state_only.fit(lf)
```

**Feed a bank from one thread at a time, though `predict`, which learns
nothing, may run from any number of threads at once.** A bank reads one
ordered stream, so a call that finds it busy on another thread raises
`RuntimeError`: wait for that call to return, or give each thread its own
bank.

**A bank keeps every group until you drop it, so in a long-running bank,
drop the quiet ones with `bank.drop_groups`.** `bank.groups()` gives each
group's `last_clock`, in the clock's own units, or in seconds since 1970 on
a temporal clock, so compare it with the current time in the same units.
This code uses `df` and `lf` from [Example data](#example-data):

```python
bank = po.ModelBank([spec])
bank.fit(lf)
now = df["t"].max()                        # the current time on the clock t; on a temporal clock, time.time()
stale = bank.groups().filter(pl.col("last_clock") < now - 30 * 86400)
bank.drop_groups(stale["group"])           # they start over if they reappear
```

### Outside a live Python process

For a scheduled job, or a deployment with no Python, run the same bank from
a file to a file with the standalone `online` command line, a binary on
each GitHub release. It reads the input, the output and the specs from a
TOML file. Each `[[specs]]` table takes the keywords every spec shares, and
its `[specs.model]` table names the model and takes the model's own:

```toml
input = "ticks.parquet"          # read a chunk at a time; the extension names the format
output = "fitted.parquet"        # the input's columns, plus one column per spec
save_state = "bank.state"        # the state after the last row; load_state resumes from one

[[specs]]                        # one table per spec
name = "ridge"
targets = ["y"]
features = ["x0", "x1", "x2"]
clock = "t"
half_life = 600.0
gap_cap = 300.0
group = "stock_id"
[specs.model]                    # the model, and its own keywords
type = "ew_ridge"
ridge = [1e-6, 0.1]
```

```sh
online --config bank.toml        # parquet in, parquet out: the numbers the bank gives in Python
```

[docs/RUNNER.md](docs/RUNNER.md) has every key and flag.

### Output as Arrow

To hand a bank's output to a consumer that is not Polars, call
`bank.fit_predict_arrow(df)` or `bank.predict_arrow(df)` in place of
`fit_predict` or `predict`. Each returns the output through the Arrow
PyCapsule interface, an Arrow specification, so any reader of
`__arrow_c_array__` takes it directly. `fit_predict` itself hands its
output over as a Polars `Series`, on py-polars' private methods, so this
package measures a Polars range and promises none ([Which interfaces carry
a promise](#which-interfaces-carry-a-promise)).
This code uses `df` and `today` from [Example data](#example-data):

```python
bank = po.ModelBank([spec])
structs = bank.fit_predict_arrow(df)                    # one per spec, as Arrow
out = df.with_columns([pl.Series(s) for s in structs])  # or any reader of __arrow_c_array__
scored_structs = bank.predict_arrow(today)              # the same for predict, on today: the rows after df's
```

**Each `ArrowStruct` holds one spec's output, with `fit_predict`'s values,
field for field and null for null.** The input still goes in as a Polars
`DataFrame`.

**Read each struct once.** Exporting hands its buffers to the consumer, so
a second read raises `ValueError`.

Here is how two consumers take a struct, as measured:

| consumer | measured on | takes |
|---|---|---|
| pyarrow | pyarrow 25.0.1 | a struct as it is: `pa.array(s)` and `pa.table(s)` each take one, values and types unchanged |
| DuckDB | duckdb 1.5.5 | `pl.Series(s)`. It refuses an `ArrowStruct`, because `__arrow_c_stream__` is the method it looks for. A spec's output arrives table-shaped, one column per field |

## Saving, loading and serving

Save a bank's state (everything it has learned, for every spec and group)
to one file, and load it back to keep learning or to score new rows
without learning. Or export it as JSON. A state saved by any release so
far, 0.13.0 included, does not load in this build: refit the bank from its
input. `with_windows` keeps a state of its own ([Saving and resuming a
window run](#saving-and-resuming-a-window-run)).

### Save and load

Each way of running a bank saves and loads the same file, written whole or
not at all, and the same to the byte whichever way wrote it. Each writes
the file at its own moment:

| run by | saves with | loads with | the file is written |
|---|---|---|---|
| a bank object | `bank.save(path)` | `po.ModelBank.load` | when `bank.save` is called |
| a query | `save_state=` | `load_state=` | when the run reaches its last row, with the same bytes a `ModelBank` would write |
| the command line | `--save-state` | `--resume` | only after its output is committed |

This code uses `df`, `lf`, `today` and `later` from [Example data](#example-data):

```python
# df is the stream so far, and today (later, as a query) holds the rows that follow it.
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

# today again, which the bank has learned: skip_learned keeps only the rows it has not (below)
bank.fit_predict(bank.skip_learned(today))

# The same file as bytes, for a checkpoint that lives somewhere other than a file:
blob = bank.save_bytes()
bank = po.ModelBank.load_bytes(blob, specs=[spec])
```

**To resume on input that overlaps the state, pass it through
`bank.skip_learned(frame)` before the bank.** Without it, the overlap steps
every group's clock back, which the bank refuses unless
`restart_after_step_back` reads it as a new start. `skip_learned` keeps the
rows after each group's last clock (a row at that clock counts as learned)
and every row of a group the bank has not seen. It keeps a `LazyFrame`
lazy. For a bank whose specs all count rows, it raises `ValueError`; drop
the learned rows with `frame.slice(bank.rows_seen())` instead, when the
input starts where the saved one did.

**A query abandoned before its last row, or ended by an error inside the
bank, leaves the `save_state` file as it was.** An error in a step after
the bank, such as a full disk under `sink_parquet`, does not stop the bank
on Polars 1.x. The state is then written although the output is missing
([docs/STATE-WORKFLOW.md](docs/STATE-WORKFLOW.md) has the measurements).
To write the state only with the output, use the command line, which saves
it after committing the output. In a query, give each batch of data its own
dated `save_state` file, so a rerun loads the previous batch's state.
This code uses `lf` and `later` from [Example data](#example-data):

```python
(lf.online.fit_predict([spec], save_state="bank-2024-01-02.state")   # the first batch, and a state file of its own
   .sink_parquet("fitted-2024-01-02.parquet"))
(later.online.fit_predict(load_state="bank-2024-01-02.state",        # the next batch loads the one before ...
                          save_state="bank-2024-01-03.state")        # ... and writes a file of its own,
   .sink_parquet("fitted-2024-01-03.parquet"))                       # so a rerun of it loads bank-2024-01-02 again
```

**`po.ModelBank.load` raises an error that names what is wrong with the
file:**

| the file | raises | what to do |
|---|---|---|
| does not exist yet | `FileNotFoundError` | on a first run, build the bank from its specs with `po.ModelBank(specs)` |
| is not a bank, or holds different specs from the `specs=` given | `ValueError` | check the path, and pass the specs the state was saved with |
| holds a state that contradicts its own spec | `ValueError` | refit the bank from its input |
| was written under a newer state schema or file format than this build reads | `ValueError` | load it with the version of polars-online that wrote it |
| was written under a state schema below 25, which is every release so far, 0.13.0 included | `ValueError`, naming the range | refit the bank from its input. `po.schema_version()` gives this build's schema |

### Serving without learning

To score rows without learning from them, call `bank.predict(frame)` on a
bank object, or `lf.online.predict(...)` in a query. Every row is scored
from the same state, which `predict` never moves. Since it skips each
row's update, `ewridge` scores 1.8 times as fast as it learns at 5
features, and 2.9 times at 20 ([docs/PERFORMANCE.md](docs/PERFORMANCE.md)
§9).
This code uses `today` and `later` from [Example data](#example-data):

```python
scored = bank.predict(today)                             # every row scored from the same state: nothing moves
served = later.online.predict(bank).collect()            # a bank, as it stands when the query runs
from_file = later.online.predict("bank.state").collect() # a path, read when the query is built
```

**Row *i* of `scored` carries what `fit_predict` would have reported had it
been the next row of its group's stream:** `pred`, `weight_sum`, `sigma`,
`zscore`, the selection and the metrics, field for field ([Per-row
diagnostics](#per-row-diagnostics)). Two fields differ: `drift` never
fires, and `coef` is filled on each group's last accepted row only, since
the same coefficients score every row. How `predict` reads a row:

| the row | `predict` |
|---|---|
| its target column | may be absent; `resid` is then null |
| its weight column | is not read |
| of a group the bank has never seen | scores null |
| before the last clock its group learned | is scored against the state as it stands, with a clock step of 0: never refused, never a restart |
| where `session_gap="reset"` or `group_close="session"` would restart its group | scores null, with `weight_sum` 0 |
| after rows an `embargo` still holds | sees the state as it stands: a held row whose delay has passed by the scored row's clock is not released |

### Reading a state without this library

To read a state from a program without this library, export it with
`bank.to_json()`, or write it to a file with `bank.save_json(path)`. Keep
the binary file for resuming, since `load` reads only that form. Every
export is read back and checked against the state before you get it, and
one that does not match raises `ValueError`.

```python
text = bank.to_json()          # everything save() writes, as JSON
bank.save_json("bank.json")    # the same, to a file
```

**A program reading the JSON must turn the strings `"nan"`, `"inf"` and
`"-inf"` back into numbers.** JSON has no literal for `NaN` or `±inf`, and
an ordinary state can hold them, as a spec with `half_life=inf` (no
forgetting) does. The strings are the spelling a spec's `half_life` takes.

## Reading the fit

Read a fit from the bank, with its methods, between chunks or after the
run, or from the output, row by row. The bank reports each group as it
stood after the last row it learned from.

| to read | call | subsection |
|---|---|---|
| what a bank holds, and what it was fed | `repr(bank)`, `bank.groups()`, `bank.summary()`, `bank.describe()`, `bank.last_row()` | [What a bank holds](#what-a-bank-holds) |
| any field of the output, by name | `po.spec.output_index`, `po.spec.coef_fields` | [Output field names](#output-field-names) |
| the coefficients, at the end or row by row | `bank.coef()`, or `coef_every=1` and `.online.unnest` | [Coefficients](#coefficients) |
| the running sums a fit is solved from, and the algebra on them | `bank.gram(spec)`, `po.gram` | [The running sums behind a fit](#the-running-sums-behind-a-fit) |
| a finished group's running sums | `bank.closed_groups()` | [One row per finished group](#one-row-per-finished-group) |
| what a correlation matrix says, from an array, a `gram()` dict or a closed row | `po.corr` | [Reading a correlation matrix](#reading-a-correlation-matrix) |

### What a bank holds

A bank describes itself without its data, live or loaded from its file.
Six calls say what it holds, and four tables say how each fit is doing and
what it was fed.

```python
bank = po.ModelBank.load("bank.state")   # no specs=: the file is enough

repr(bank)                # ModelBank(['ridge'], groups=4, rows_seen=400)
bank.specs                # every spec back, as the dict its builder made
bank.groups()             # one row per (spec, group):
                          #   ┌───────┬───────┬────────────────┬────────────┐
                          #   │ spec  ┆ group ┆ rows_processed ┆ last_clock │
                          #   │ ridge ┆ b0    ┆ 100            ┆ 396.0      │
                          #   │ ridge ┆ b1    ┆ 100            ┆ 397.0      │
                          #   │ ridge ┆ b2    ┆ 100            ┆ 398.0      │
                          #   │ ridge ┆ b3    ┆ 100            ┆ 399.0      │
                          #   └───────┴───────┴────────────────┴────────────┘
bank.output_fields()      # {'ridge': ['pred_y__r0.000001', ..., 'weight_sum', 'settled_frac', 'withheld_reason', 'coef', ...]}
bank.rows_seen()          # rows fed, over every chunk and group
bank.solve_failures()     # per spec, per group: solves that needed jitter or kept the previous fit
```

**`bank.specs` returns a read-only copy.** The bank runs the state it was
built from, so an editable list could drift from it and mislabel what
`coef()` reports. To change a spec, build and feed a new bank.
This code uses `df` from [Example data](#example-data):

```python
bank = po.ModelBank.load("bank.state")
last = bank.last_row("ridge")     # one row per group: what fit_predict wrote on the last row it learned from
betas = bank.coef()               # one row per coefficient, with the term it belongs to
fed = bank.summary("ridge")       # one row per group: what it was fed, and its warm-up readings
cols = bank.describe("ridge")     # one row per input column per group: column, role, count, null_count, mean, std, min, max

# Fit many models, save each in a folder of its own, and compare them in one table:
from pathlib import Path
Path("fits").mkdir(exist_ok=True)
for name, half_life in [("fast", 100.0), ("slow", 1000.0)]:
    one = po.ModelBank([po.spec.ewridge(name, targets=["y"], features=["x0", "x1"], clock="t",
                                        gap_cap=300.0, half_life=half_life)])
    one.fit_predict(df)
    one.save(f"fits/{name}.state")
table = pl.concat(
    [po.ModelBank.load(f).last_row() for f in sorted(Path("fits").glob("*.state"))],
    how="diagonal_relaxed",
)
```

**All four cover every spec by default, with `spec` as the first column.**
`coef()`, `summary()` and `describe()` have the same columns for every
spec, so a plain `concat` stacks the tables of banks from different runs.
The fields of `last_row()` differ between specs, so stack it with
`"diagonal_relaxed"`, which fills a missing field with nulls.

**`last_row()` matches `fit_predict` field for field:** `pred`, `sigma`,
the metrics and the interval when the spec asks for them ([Per-row
diagnostics](#per-row-diagnostics)), `weight_sum`, and `coef` when that row
carried it. Until a group learns from a row, its record is all nulls.

**`summary()`'s counts and `describe()` cover every row fed to the group,
without decay.** They say what the model was trained on, and `predict`
moves neither. `describe()` counts a value as the models do: a null, a NaN,
an infinity or a magnitude beyond 1e100 is a `null_count`. The columns of
`summary()`:

| columns | what they hold |
|---|---|
| `rows_fed` | rows routed to the group |
| `rows_processed` | rows the model accepted |
| `rows_skipped` | rows the model skipped, because a feature or the weight was missing |
| `rows_learned` | rows that moved the fit: a weight above zero and a target present. Under an embargo a row is counted as it arrives; a window target's row, when it is released |
| `rows_zero_weight` | rows that advanced the clock and nothing else |
| `weight_sum`, `clock_min`, `clock_max`, `last_clock` | the weight behind the state, and the clock's range and last value |
| `session_changes`, `clock_backwards`, `resets` | what the clock rules met |
| `settled_frac`, `error_inflation`, `min_support_coef` and the feature it belongs to, `n_coef` | the warm-up readings after the last row |

### Output field names

Your code reads the output by field name, so a test holds every name,
default and signature to a checked-in snapshot, and no release renames your
columns silently ([Versioning and the Polars pin](#versioning-and-the-polars-pin)).
To find a name without building it, filter one of two tables:
`po.spec.output_index` for every field, or `po.spec.coef_fields` for every
coefficient. The code that renders the names builds both from the spec, so
they always agree with the output. [docs/OUTPUTS.md](docs/OUTPUTS.md) lists
every field of every model.

Give a spec several values of `ridge` or `half_life`, several
`feature_sets` or a `lasso_path`, and it fits a *grid*: one fit per
combination of values, each a *point* of the grid. A field's name says
which point it holds. Here `grid` has two half-lives and two ridge values.
This code uses `df` from [Example data](#example-data):

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

**If your own code parses the field names after the bank, keep `__` and
`@` out of target names and feature-set labels.** A target named `y__r0.5`
renders like a ridge grid on `y`. Filtering `po.spec.output_index` needs no
such care.

**Each name follows a grammar, so you can read it by eye.** A number in a
name renders as a plain decimal in `[1e-6, 1e7)`, and in compact scientific
notation outside it.

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

Read the coefficients from the bank with `bank.coef()`, or from the
output, as columns that show the fit as it moved. Both are the fit *after*
a row's update: `bank.coef()` after the last row each group learned from,
and the output's `coef` after its own row. A row's own `pred` came from the
fit *before* it.
This code uses `df` and `lf` from [Example data](#example-data):

```python
ols = po.spec.ewridge("ols", targets=["y"], features=["x0", "x1"], clock="t",
                      half_life=600.0, gap_cap=300.0, group="stock_id",
                      coef_every=1)         # write coef on every row but a skipped one

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

# 3. From a saved output: unnest takes the specs, a bank, or the path of the state the run saved.
lf.online.fit_predict([ols], save_state="ols.state").sink_parquet("fitted.parquet")
saved = pl.scan_parquet("fitted.parquet").online.unnest("ols.state").collect()
```

**`coef_every=1` writes `coef` on every row but a skipped one**, as a list
of one float per term. The default, `0`, writes it on each group's
last row in a chunk.

**To spread that list into columns, chain `.online.unnest(specs)` after
`fit_predict`.** It takes the specs, a bank, or the path of a saved state,
and works on a saved output.

**Several targets, or a grid, put several blocks in one `coef` list:** one
per target and point of a grid of `ridge` values, `feature_sets` or a
`lasso_path`. Each half-life of a `half_life` grid has a `coef` field of its
own. Here `grid`, from [Output field names](#output-field-names), has two
half-lives and two ridge values.
This code uses `df` from [Example data](#example-data):

```python
grid_bank = po.ModelBank([grid])
cols = grid_bank.fit_predict(df).online.unnest([grid])     # coef@h100 and coef@h500, spread into columns
cols.select("pred_y__r0.5@h500", "coef_y_x0__r0.5@h500")    # each block's columns named the way its pred field is
betas = grid_bank.coef()                                    # target, ridge, feature_set and penalty tell the blocks apart
blocks_wide = betas.pivot("term", index=["group", "instance", "ridge"], values="coef")   # a row per block ...
one_block = betas.filter(pl.col("ridge") == 0.5).pivot("term", index=["group", "instance"], values="coef")   # ... or one block
```

### The running sums behind a fit

Three models, `ewridge`, `lasso` and `ew_cov`, keep a matrix of running
sums, their *Gram* in least-squares terms. It summarizes every row the
model has seen, as a sufficient statistic, so a saved state can answer
questions the run never asked. `bank.gram(spec)` returns it per group and
per half-life, or an empty list for any other model. `gram()` and
`po.gram` need numpy, an optional extra.

Four things to know before reading a Gram:

| | why |
|---|---|
| `weight_sum` counts weight, not rows | `n_kish = weight_sum² / Σw²` is the number of equally weighted rows the moments are worth, which is what a standard error divides by. It does not fall when a stream goes quiet; `weight_sum` does |
| a spec can have several Grams | under the default `target_gaps="own_rows"`, a target that goes missing on different rows from the others is fitted from a Gram of its own. `gram()` returns one dict per Gram, each naming its `targets` |
| `coef()` and `gram()` can disagree | `bank.coef()` is as of the model's last *solve*, which its `solve_every` schedule decides, while `gram()` is as of the last row |
| under a `window_size`, a Gram covers only the window | every array does, the target moments included, so `po.gram.solve` on it fits the window |

This code uses `df` from [Example data](#example-data):

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

# The model's algebra by hand, on the centred system, which keeps a feature far from zero precise:
slopes = np.linalg.solve(g["comoments"][1:, 1:], g["cross_centred"][0][1:])   # column 0 is the intercept
intercept = g["cross_moments"][0][0] - g["means_by_target"][0][1:] @ slopes
resid_var = g["target_vars"][0] - slopes @ g["comoments"][1:, 1:] @ slopes
r2 = 1 - resid_var / g["target_vars"][0]
```

The [`ModelBank.gram`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.ModelBank.gram)
reference states every array and the identities that relate them.

**`po.gram` does the same algebra in one call**, on any Gram, such as one
read from a saved state. Two calls need care:

| limit | what to do |
|---|---|
| `po.gram.solve` solves a plain ridge | read a `ridge_scale="sum"` or `coef_prior` fit from `bank.coef()`, since `solve` does not reproduce it |
| `po.gram.merge` pools parts that share a weighting, such as one Gram per group or per shard of a pass | before merging two halves of a decayed stream, multiply the earlier half's `weight_sum` and `target_weights` by `0.5 ** (dt / half_life)`, with `dt` the clock from its last row to the later half's last row. Halves run with `half_life=float("inf")` merge as they are |

```python
g = po.ModelBank.load("bank.state").gram("ridge")[0]   # the first group's Gram, from a saved state

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

**Before merging two halves of a decayed stream, age the earlier half to
the later half's end.** Here each half of `df` is fitted on its own.
This code uses `df` from [Example data](#example-data):

```python
halves = po.spec.ewridge("halves", targets=["y"], features=["x0", "x1"], clock="t", gap_cap=300.0,
                         half_life=100.0)


def gram_of(rows):                                        # one bank per part, as a shard or a day is fitted
    part = po.ModelBank([halves])
    part.fit_predict(rows)
    return part.gram("halves")[0]


early, late = gram_of(df.head(200)), gram_of(df.tail(200))
decay = 0.5 ** ((df["t"][399] - df["t"][199]) / 100.0)   # dt: from the earlier half's last row to the later half's last row
early = dict(early, weight_sum=early["weight_sum"] * decay, target_weights=early["target_weights"] * decay)
pooled = po.gram.merge([early, late])
assert np.allclose(pooled["comoments"], gram_of(df)["comoments"])   # the Gram of one run over all 400 rows
```

### One row per finished group

When new group values never stop appearing, such as a day id, a session
id or a block number, set `group_close` to keep the bank's memory bounded.
The bank then writes each finished group's running sums out, a row per
half-life and per Gram, and frees its state. Without it, every group's
state stays until `bank.drop_groups` removes it. `group_close` is refused
beside an `embargo` or a window target, since both hold a group's rows past
its end.
This code uses `df` from [Example data](#example-data):

```python
blocks = po.spec.ew_cov("cov", features=["x0", "x1"], lam=1.0,   # lam=1.0: no decay
                        group="block", group_close="monotone")   # a block is finished once a higher one arrives
by_block = df.with_columns(block=pl.int_range(pl.len()) // 100)   # a block id that rises every 100 rows

bank = po.ModelBank([blocks])
bank.fit_predict(by_block)

closed = bank.closed_groups()          # blocks 0, 1 and 2, oldest first: one half-life and one Gram, so one row each
first = po.gram.from_row(closed.head(1))   # block 0's row, as the dict gram() returns
corr = po.gram.correlation(first)      # everything in po.gram works on it
```

**Each closed row holds, bit for bit, what `bank.gram()` would have
returned for that group just before it closed**, its matrix packed as the
upper triangle. It adds the span's own `rows_fed`, `rows_learned`,
`clock_min` and `clock_max`. A model with coefficients adds its `coef`, an
`ew_cov` with `pca` its eigendecomposition, and a `marginal` its pairs.

**Closed rows wait in the bank until `bank.closed_groups()` takes them
out.** Called with `drop=False`, it reads them and leaves them queued. Rows
not yet read are saved with the state, so a loop that saves between chunks
loses none.

**Two modes decide when a group is finished.** In either, the last group
never closes, since no row proves it finished, and `bank.gram()` still
reads it.

| `group_close` | a group is finished when | refused |
|---|---|---|
| `"monotone"` | its key is below the largest seen, since the key never goes backwards | a null key, or a chunk whose keys are out of order, naming the row; a float or temporal key column, which Polars' `cast` turns into an integer in the query before the bank |
| `"session"` | its `session` value changes | a spec without a `session` column, or with `session_gap` |

**Under `"monotone"`, the bank orders an integer key as a number and a
String or `Categorical` key as text, so `"9"` comes after `"10"`.** Polars'
own `sort` agrees, so sort the input by the key before the bank.

**To write the closed rows to a file as the run goes, pass a path as
`closed_groups=`** to `lf.online.fit_predict`, `fit_predict_batches` or
`fit`, or `--closed-groups` to the command line
([docs/RUNNER.md](docs/RUNNER.md)). With `fit`, which keeps no per-row
output, the file is the run's whole product: a stream too large for memory
goes in, and one row per block comes out. Here `blocks` learns from
`by_block`, and each closed block's row goes to `blocks.parquet`:

```python
po.ModelBank([blocks]).fit(by_block.lazy(), closed_groups="blocks.parquet")
```

### Reading a correlation matrix

`po.corr` repairs, shrinks, summarizes and scores a correlation matrix,
given as an array, a `gram()` dict or a closed row. Each function needs
only numpy, and follows the paper it names.

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
po.corr.fisher_se(n=2000, rho=0.3)         # the standard error of a correlation; phi_a= and phi_b= add the AR(1) inflation, with its caveats
```

Nine more, by kind, each with its arguments in the [API
reference](https://hgilde.github.io/polars-online/corr.html):

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

Judge a model as the stream runs, by switching diagnostics on in its spec,
or after the run, by passing its output to `po.eval`. A spec's diagnostics
and selection measure each row against what the models learned before it,
never against the row's own outcome. Their memory does not grow with the
stream.

| kind | switch or call | what it gives | subsection |
|---|---|---|---|
| diagnostics | `emit_sigma`, `emit_drift`, `emit_metrics` and others, in a spec | residual spread, break detection and running accuracy, row by row | [Per-row diagnostics](#per-row-diagnostics) |
| | `conformal`, in a spec | an interval that assumes no distribution | [Conformal intervals](#conformal-intervals) |
| selection | `emit_selected`, `emit_averaged`, in a spec | a choice among a grid's settings, as the stream runs | [Choosing among a grid's settings](#choosing-among-a-grids-settings) |
| evaluation | `po.eval`, after the run | metrics and comparisons over a frame in memory | [Evaluating an output frame](#evaluating-an-output-frame) |
| | `po.eval.sums`, per chunk | the same metrics, without holding the output | [Evaluating a stream too large to hold](#evaluating-a-stream-too-large-to-hold) |
| simulation | `po.sim.regimes` | a stream whose regimes are known, and the truth beside it | [Data whose truth is known](#data-whose-truth-is-known) |

### Per-row diagnostics

Switch a diagnostic on with its keyword in the spec, and the output gains
its fields, one per *slot*: one prediction of one target, at one point of a
grid. Every switch but `emit_clocks` reads the residuals, so only the ten
[linear models](#linear-models), which predict a target, take them. Any
other model refuses them by name.

| kind | switches | fields |
|---|---|---|
| spread and surprise | `emit_sigma`, `emit_zscore`, `resid_quantiles` | `sigma_`, `zscore_`, `abs_resid_q<p>_` |
| breaks | `emit_drift` | `drift_` |
| running accuracy | `emit_metrics` | `ic_`, `r2_`, `hit_rate_` |
| residual autocorrelation | `emit_autocorr` | `autocorr_` |
| an interval | `conformal` | `lo_`, `hi_`, `coverage_`: [Conformal intervals](#conformal-intervals) |
| a choice among the slots | `emit_selected`, `emit_averaged` | one per target: [Choosing among a grid's settings](#choosing-among-a-grids-settings) |
| clocks, on every model | `emit_clocks` | `scored_clock`, the clock a row was scored at, and `learned_clock`, the clock of the last row learned, once per spec ([Labels that arrive late](#labels-that-arrive-late)) |

This code uses `df` from [Example data](#example-data):

```python
diag = po.spec.ewridge(
    "diag", targets=["y"], features=["x0", "x1"], clock="t", gap_cap=300.0, half_life=500.0,
    ridge=[1e-6, 0.1],           # a grid of two ridge values: two slots for the one target
    # EW: exponentially weighted. A switch's tuning keywords sit under it, at their defaults.
    emit_sigma=True,             # sigma_<slot>:     EW standard deviation of that slot's out-of-sample residuals
    emit_zscore=True,            # zscore_<slot>:    resid / sigma: how surprising the row was, in units of recent error
    emit_drift=True,             # drift_<slot>:     Page-Hinkley break detection on |resid| ...
    drift_delta=0.5,             #                   ... with this tolerance, in units of the slot's sigma ...
    drift_threshold=20.0,        #                   ... and this threshold, in sigma times clock units
    drift_action="flag",         #                   "reset" also starts the model over at a break
    emit_metrics=True,           # ic_, r2_, hit_rate_<slot>: EW IC (the correlation of prediction with target), R², hit rate
    resid_quantiles=[0.5, 0.9],  # abs_resid_q<p>_<slot>: EW quantiles of |resid|, within 0.78% of the exact ones
    emit_autocorr=True,          # autocorr_<slot>:  EW correlation of each residual with the one this many back:
    resid_autocorr_lag=1,        #                   away from zero, the model is missing something
    conformal=0.9,               # lo_, hi_, coverage_<slot>: an interval at this coverage, and the coverage delivered
    conformal_rate=0.05,         #                   how fast its radius moves (Conformal intervals)
    emit_clocks=True,            # scored_clock, learned_clock: on every model, since it reads no residual
)
band = po.ModelBank([diag]).fit_predict(df).unnest("diag")
```

**On each row, `sigma`, the interval and its coverage, the quantiles, the
autocorrelation and the metrics are read before the row updates them.**
`resid`, `zscore` and `drift` measure the row against them.

**`hit_rate`, in `emit_metrics` and in `po.eval.metrics`, asks whether
prediction and outcome fall on the same side of zero.** So on a target
that is always positive, such as a plain ratio, every row counts as a hit.
[Relative and look-ahead targets](#relative-and-look-ahead-targets) says
how to write a ratio target about zero with Polars expressions.

On an `sgd` or `ftrl` fit with `loss="logistic"`, `pred` is a probability
and `y` a 0/1 label, so the three metrics mean something else under the
same names:

| field | on a logistic `sgd` or `ftrl` fit |
|---|---|
| `hit_rate` | the accuracy at a 0.5 threshold |
| `r2` | the Brier skill score against the running base rate |
| `ic` | the point-biserial correlation between the probability and the label |
| a log loss | not streamed: `po.eval` computes one over a frame in memory ([Evaluating an output frame](#evaluating-an-output-frame)) |

### Conformal intervals

**Use `conformal` for an interval when the residuals are not Gaussian.**
It tracks the `conformal` quantile of `|resid|` directly, so its long-run
coverage is the number you asked for, whatever the residuals do. On
Gaussian residuals the interval `pred ± z·sigma`, built from `emit_sigma`'s
field with Polars expressions after the bank, covers equally well. On
fat-tailed or heteroskedastic residuals, that Gaussian interval over-covers
by several points.

**The radius moves by `conformal_rate`, 0.05 by default.** It widens on a
miss, by rate·sigma·coverage, and narrows on a hit, by rate·sigma·(1 −
coverage). Each step is scaled by the row's weight over the mean weight.
This code uses `df` from [Example data](#example-data):

```python
ci = po.spec.ewridge("ci", targets=["y"], features=["x0", "x1"], clock="t",
                     gap_cap=300.0, half_life=500.0, conformal=0.9)
band = po.ModelBank([ci]).fit_predict(df).unnest("ci")   # lo_y, hi_y: the interval; coverage_y: the coverage delivered
held = band.select(pl.col("y").is_between(pl.col("lo_y"), pl.col("hi_y")).mean())   # the realized coverage
```

### Choosing among a grid's settings

Two switches choose among a grid's slots (one prediction per target and
point) as the stream runs, by each slot's exponentially weighted
out-of-sample error so far. `emit_selected` commits to the best slot, and
a spec with one slot per target refuses it. `emit_averaged` hedges across
all of them. To compare whole specs after the run, pass the output to
`po.eval.compare_specs` ([Evaluating an output
frame](#evaluating-an-output-frame)).
This code uses `df` from [Example data](#example-data):

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

After the run, pass the output frame to `po.eval`: its calls score a spec,
compare specs, or unpack the frame to long form. `po.eval.seqtest`
runs the [`seqtest`](#seqtest--a-sequential-test-of-a-sign-by-betting)
model's test on the frame. On a 0/1 label,
`po.eval.metrics(..., binary=True)` reads `pred` as a probability and adds
the log loss.
This code uses `df` from [Example data](#example-data):

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

`po.eval.metrics` needs the whole output in one frame. When the output is
too large to hold, say fifty slots over a billion rows, reduce each chunk
to ten numbers per key with `po.eval.sums`. Add the parts with
`po.eval.merge_sums`, which is exact whatever the split, and turn the total
into the metrics with `po.eval.from_sums`.

**The sums are centred, so a target far from zero keeps its variance.** Raw
`Σy` and `Σy²` would lose all of it for a target around 1e8 with unit
spread.
This code uses `lf` from [Example data](#example-data):

```python
ridge = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"],
                        clock="t", gap_cap=300.0, half_life=500.0, group="stock_id")

running = None
for out in po.ModelBank([ridge]).fit_predict_batches(lf, chunk_rows=100):   # the output, 100 rows at a time
    part = po.eval.sums(out, "ridge", by=["stock_id"])   # weight= names a column to weight the rows by
    running = part if running is None else po.eval.merge_sums(running, part)

po.eval.from_sums(running, min_obs=10)   # R², IC, hit rate and MSE, as metrics() gives them, and the RMSE
```

### Data whose truth is known

To test a model that claims to find a changing correlation structure, fit
it on the `rows` of
[`po.sim.regimes`](https://hgilde.github.io/polars-online/sim.html#polars_online.sim.regimes)
and compare what it finds with `truth_rows` and `truth_blocks`, the truth
returned beside them. The measurements in [docs/REGIMES.md](docs/REGIMES.md)
show what `hmm`, `corrchange` and `bocpd` find on such streams, and what
they miss.

```python
sim = po.sim.regimes(
    4, states=[0.2, 0.7],                     # four series; two regimes, at these equicorrelations
    transition=[[0.98, 0.02], [0.02, 0.98]],  # how the regimes switch
    n_blocks=8, rows_per_block=500,
    durations=None,                           # or a block count per state: each state lasts exactly as long as it says
    design="step", smooth_rows=0,             # or "smooth": interpolate the matrix over smooth_rows rows at a boundary
    phi=0.3, noise=0.01,                      # returns correlated with their own past; observation noise
    async_rates=[1.0, 1.0, 0.4, 0.4],         # two series observed less often; a series with no observation is null
    seed=0,                                   # the same seed twice is byte-identical
)
rows, truth_rows, truth_blocks = sim["rows"], sim["truth_rows"], sim["truth_blocks"]
```

| frame | what it holds |
|---|---|
| `rows` | what a consumer sees: an entity, the row's index `t`, a clock and a session, levels `x_1` .. `x_m`, and `activity`, null unless `activity=(mean, shape)` is given. For returns, `unpivot` the levels into the long input of [`refresh_time`](#series-that-tick-at-their-own-times), and apply `.diff()` to the grid it builds |
| `truth_rows` | per row `t`: the block, state, volatility multiplier and interpolation fraction |
| `truth_blocks` | each block's true correlation matrix, as the upper triangle in a list |

## Models

polars-online has twenty-one models in four families. A spec's clock,
grouping and warm-up mean the same whichever model it names, and its
half-life is always a half-life on the clock ([How a bank sees a
stream](#how-a-bank-sees-a-stream)). To look up a model's keywords and
their defaults, open its builder's reference page, such as the one for
`po.spec.ewridge`. [docs/OUTPUTS.md](docs/OUTPUTS.md) lists every field a
model writes, and [docs/TESTING.md](docs/TESTING.md) what each is checked
against.

| model | learns by | what it is |
|---|---|---|
| **[Linear models](#linear-models)** | | predict a numeric target, from features or from its own past |
| [`ewridge`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.ewridge) · [math](#ewridge--ew-ridge-on-sufficient-statistics) | solve | exponentially weighted ridge regression, the workhorse; several ridge values and feature sets are solved from the same running sums at almost no extra work |
| [`rls`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.rls) · [math](#rls--recursive-least-squares) | solve | recursive least squares, in the numerically safe square-root form |
| [`lasso`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.lasso) · [math](#lasso--lasso-path-its-penalty-chosen-as-it-runs) | solve | lasso and elastic-net path, with the penalty chosen as the stream runs |
| [`kalman`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.kalman) · [math](#kalman--random-walk-β-dynamic-linear-model) | filter | a Kalman filter whose coefficients drift as random walks |
| [`huber`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.huber) · [`quantile`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.quantile) · [math](#huber--quantile--robust-regression) | reweight | robust and quantile regression |
| [`sgd`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.sgd) · [math](#sgd--stochastic-gradient-descent) | step | stochastic gradient descent with squared, Huber, quantile, ε-insensitive, Poisson and logistic losses |
| [`pa`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.pa) · [math](#pa--passive-aggressive-regression) | step | passive-aggressive regression, with no learning rate to tune |
| [`ftrl`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.ftrl) · [math](#ftrl--online-logistic-regression) | step | FTRL-proximal regression, logistic by default, with an L1 penalty that zeroes coefficients |
| [`holt`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.holt) · [math](#holt--holts-linear-trend) | step | Holt's linear trend, the baseline that uses no features |
| **[Moments and correlation](#moments-and-correlation)** | | track the running moments and correlations of the columns you name |
| [`ew_cov`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.ew_cov) · [math](#ew_cov--exponentially-weighted-moments) | accumulate | running mean, variance, covariance, correlation, partial correlation, Mahalanobis distance and principal components |
| [`marginal`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.marginal) · [math](#marginal--every-pairs-moments-kept-in-the-state) | accumulate | every (feature, target) pair's running mean, variance, covariance, correlation, slope and t, for a wide set of columns, kept in the state and read back as a table; optionally at a set of lags, and with binned target moments for the relations a correlation cannot see |
| [`deco`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.deco) · [math](#deco--one-correlation-for-the-whole-matrix) | accumulate | one correlation for the whole matrix: Engle & Kelly's equicorrelation, or one per block and per pair of blocks |
| [`rcov`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.rcov) · [math](#rcov--a-blocks-realised-covariance-robust-to-noise) | accumulate | a block's realised covariance, robust to microstructure noise: the Barndorff-Nielsen–Hansen–Lunde–Shephard kernel or Christensen–Kinnebrock–Podolskij pre-averaging, reported when a group closes |
| **[Clustering and classification](#clustering-and-classification)** | | put labels on rows |
| [`kmeans`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.kmeans) · [math](#kmeans--exponentially-weighted-k-means) | step | exponentially weighted k-means: cluster labels assigned before the row is learned from, with a split–merge move that finds a cluster born after seeding |
| [`micro`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.micro) · [math](#micro--density-based-clustering-any-shape) | step | density-based clustering: DenStream micro-clusters linked into clusters of any shape and number, flagging the rows that belong to none |
| [`ew_class`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.ew_class) · [math](#ew_class--gaussian-classification-on-ew_cov-moments) | accumulate | Gaussian classification, QDA, LDA or naive Bayes, with one set of running moments per class: a label column in, class probabilities out |
| **[Sequential tests and regimes](#sequential-tests-and-regimes)** | | give evidence that something holds or has changed, and which regime the stream is in |
| [`seqtest`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.seqtest) · [math](#seqtest--a-sequential-test-of-a-sign-by-betting) | test | a sequential test of a sign by betting, giving evidence you can read at any row: on its own, of a column's sign; with `a` and `b`, of whether one spec of the bank predicts closer than another |
| [`corrchange`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.corrchange) · [math](#corrchange--has-the-correlation-structure-changed) | test | has the correlation structure changed: the Wied–Krämer–Dehling constancy test span by span, the Wied–Galeano detector row by row against a stable history, or the size of a change between two windows against a permutation null |
| [`bocpd`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.bocpd) · [math](#bocpd--how-long-has-this-regime-lasted) | filter | how long this regime has lasted: Adams & MacKay's run-length posterior, so the answer is the regime's age and not a flag |
| [`hmm`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.hmm) · [math](#hmm--which-regime-are-we-in) | filter | a Gaussian hidden Markov model, filtered as the stream runs: `ew_class` without the labels, with a transition matrix that can be learned |

How a model learns from a row decides whether, with decay off, its answer
depends on the row order:

| learns by | what the model does with a row | with decay off | order-dependent even so |
|---|---|---|---|
| **solve** | keeps running sums and computes its coefficients from them | converges to the batch answer, in any row order | `lasso`'s `penalty_selected`, ranked by out-of-sample error, though its path converges in any order |
| **accumulate** | keeps running sums and reports them | converges to the batch answer, in any row order | a lag, in `ew_cov` or `marginal`, which counts learned rows; `rcov`'s block and `deco`'s per-row estimate |
| **reweight** | solves from running sums, but lets the fit before each row decide how that row enters them | depends on the row order | |
| **step** | moves its coefficients a little on each row | depends on the row order | |
| **filter** | carries a belief forward from row to row | depends on the row order | |
| **test** | counts evidence as it goes | depends on the row order | |

**In the update rules,** `z` is `[1, x]` when there is an intercept, `w` is
the row's weight, and `W` the weight accumulated before the row. `λ` is the
row's decay, `0.5 ** (Δclock / half_life)`. A model keeps its running sums
in *accumulators* of three kinds:

| accumulator | where | why |
|---|---|---|
| an exponentially weighted **mean** | most models | it stays bounded over a stream of any length |
| a decayed **sum** | `rls`'s matrix, and the gradient sums of `ftrl` and of `sgd`'s adagrad | |
| a **centred** second moment, by a weighted Welford update | wherever a model keeps second moments as means | a variance is right even when a feature sits far from zero |

### Linear models

Ten models predict a numeric target. Nine regress it on features, and
`holt`, a baseline, uses none. Several of them take the same parameters:

| parameters | taken by | explained under |
|---|---|---|
| `solve_every`, `max_rows_between_solves` | `ewridge`, `lasso`, `huber`, `quantile` | [`ewridge`](#ewridge--ew-ridge-on-sufficient-statistics) |
| `target_gaps` | `ewridge`, `lasso` | [`ewridge`](#ewridge--ew-ridge-on-sufficient-statistics) |
| `window_size` | `ewridge`, `lasso` | [A hard window](#a-hard-window) |
| `standardize` | `ewridge`, `huber`, `quantile`, `sgd`, and `kalman`, where it is the default | `ewridge` and `kalman` |
| `coef_min`, `coef_max`, `coef_sum` | `sgd`, `pa` | [`sgd`](#sgd--stochastic-gradient-descent) |
| `weight` | every model, though `rls`, `kalman`, `sgd`, `pa` and `ftrl` read it differently from a weighted mean | [Weights](#weights) |

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

#### `ewridge` — EW ridge on sufficient statistics

*API:* [`po.spec.ewridge`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.ewridge) — *Rust:* [`ewridge.rs`](crates/online-core/src/ewridge.rs) — *Outputs:* [fields](docs/OUTPUTS.md#ewridge)

Start with `ewridge`, ridge regression solved from running sums, and change
to another linear model when your data needs what it adds, such as
coefficients that drift (`kalman`).

`S_j` and `r_j` are target `j`'s running means of `z zᵀ` and `z·y_j`, kept
over the rows where `y_j` is present, and `W_j` is the weight behind them.
The coefficients are solved from them on a schedule:

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
    standardize=True,              # solve on the correlation matrix, then unscale the coefficients
    ridge_scale="mean",            # the default; "sum" makes the ridge a warm start that fades
    coef_prior=None,               # shrink toward a stated belief instead of toward zero
    target_gaps="own_rows",        # the default: a target null on some rows is fitted on its own rows
)
```

`ewridge`'s own parameters fall into six groups:

| group | parameters | refused or required |
|---|---|---|
| the grid | `ridge` as a list, `feature_sets` | |
| when it solves | `solve_every`, `max_rows_between_solves` | |
| how the ridge applies | `standardize`, `ridge_scale`, `coef_prior` | `ridge_scale="sum"` is refused beside `standardize`, a ridge or feature-set grid, and a window |
| a target null on some rows | `target_gaps` | |
| what the sums remember | `window_size` ([A hard window](#a-hard-window)), `session_shrink` | `session_shrink` needs `session` and `long_half_life`, and is refused beside `session_gap="reset"` and `window_size` |
| a wide fit | `gram_block_rows` | refused with `window_size=`, and where a solve happens every row |

**A grid of `ridge` values or of `feature_sets`, as in `rr`, adds a solve
per value and no update of the sums.** Only the named sets are fitted, so
to keep a fit on every feature, name a set that holds them all, such as
`"all"`.

**To keep the coefficients current at every row, give
`max_rows_between_solves=1`.** Otherwise `ewridge` predicts with the
coefficients of its last solve. Under a finite half-life it solves once the
weight learned since then reaches `ln 2 / 50` of the weight the fit holds:
about 1.4 %, or every `half_life/50` of clock in steady state. To solve on
the clock instead, give `solve_every` in clock units. `half_life=inf` and
`lam=` solve on every row, so when the fit need not be current at every
row, give them a `solve_every`. On 6M rows × 20 features from a parquet stream,
`solve_every=1000` takes 1.5 s instead of 13 s, with the coefficients at
most 1000 rows out of date.

**`ewridge` solves by Cholesky factorization, and retries a near-singular
system with a small jitter on the diagonal.** `bank.solve_failures()`
counts the solves that needed it. A count above zero points at constant or
collinear features, or too few rows for the features.

**To start from a belief that fades as data arrives, such as yesterday's
fit, give it as `coef_prior` with `ridge_scale="sum"`.** The ridge then
sits on the decaying sum scale, as `rls`'s does, and pulls every
coefficient toward `coef_prior`, the intercept included, until the evidence
takes over. `S` is a mean, so under the default `"mean"` the ridge is a
penalty per observation that never fades: use it to hold the slopes near
`coef_prior`, or near zero, for good.

**To make one `ridge` value mean the same whatever the features' units,
give `standardize=True`:** the solve then runs on the correlation matrix,
and drops a feature whose variance is zero instead of blowing up. Under the
default `standardize=False`, `ridge` is in the features' squared units.

**When a target is null on some rows, `target_gaps` chooses which rows its
`S_j` covers:**

| `target_gaps` | `S_j` covers | the targets of one spec keep |
|---|---|---|
| `"own_rows"`, the default | the rows the target is present on, so its fit is the fit of the frame with those nulls dropped | one `k × k` matrix per pattern of missing rows |
| `"pairwise"` | every row, the way pandas' `DataFrame.cov` takes a pairwise-complete covariance, with `r_j` centred at the target's own rows' means | one matrix |

The two modes agree when the gaps are independent of the features. When
they depend on the features, as with a target present only on trade rows
between market-data rows, keep the default. There `"pairwise"` would scale
each slope by the feature's variance on the target's rows over its variance
on every row.

**To start each session from a blend with the long run, give
`session_shrink=f` and `long_half_life=`, beside a `session` column and a
`session_gap` that is a number:**

```python
blend = po.spec.ewridge(
    "blend", targets=["y"], features=["x0", "x1"], clock="t", gap_cap=300.0, half_life=100.0,
    session="session",       # the column whose change starts a session ...
    session_gap=60.0,        # ... and the clock step applied there; "reset" is refused beside a blend
    long_half_life=5000.0,   # a twin of the sums, forgetting at this longer half-life
    session_shrink=0.3,      # at each session change, the moments become 0.7 of today's and 0.3 of the twin's
)
```

The blended moments keep today's weight, Kish's effective sample size and
prior scale, so `weight_sum`, the warm-up gates and the solve schedule do
not move.

**For a fit over hundreds of features, give `gram_block_rows=256`.**
Updating `S` takes O(k²) time per row for `k` features, and at a thousand
features that update takes most of the run. The option holds 256 rows back
and adds them to `S` with one matrix product
([docs/PERFORMANCE.md](docs/PERFORMANCE.md) §18). A solve brings `S` up to
date before it runs, so solve rarely: under `half_life=inf`, give a
`solve_every`, as `wide` does:

```python
wide = po.spec.ewridge(
    "wide", targets=["y"],
    features=["x0", "x1", "x2"],   # three here, so the block runs; the speed-ups are for 256 to 2,000
    half_life=float("inf"), solve_every=1e9, max_rows_between_solves=1000,
    gram_block_rows=256,           # add 256 held rows to S in one matrix product
)
```

| features | rows per second, against one row at a time, on one thread |
|---:|---:|
| 256 | 5.1× |
| 1,000 | 6.6× |
| 2,000 | 5.9× |

With a solve every 512 rows, each speed-up falls to about 4×, because the
solve takes the same time either way. Blocking leaves `weight_sum`, the
timing of every prediction and chunk invariance unchanged, and the
coefficients agree with the row-by-row fit to rounding.

**With decay off and `ridge=0`, `ewridge` fits ordinary least squares over
every row seen, in any row order.** Its coefficients match
`numpy.linalg.lstsq` to 2e-13, fed forwards or backwards. On 6M rows × 20
features from a parquet stream, memory peaks at 1.4 GB, against 3.97 GB
for `lstsq` on the same rows.

#### `rls` — recursive least squares

*API:* [`po.spec.rls`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.rls) — *Rust:* [`rls.rs`](crates/online-core/src/rls.rs) — *Outputs:* [fields](docs/OUTPUTS.md#rls)

Use `rls` when the coefficients must be current at every row: it updates
them on every row in O(k²) time, without a solve. Its fit equals
`ewridge(ridge_scale="sum")` solved on every row, to better than 1e-9.

`A` and `b_j` are the decayed sums of `zzᵀ` and `y_j z`, started from the
ridge:

```
A ← λA + w zzᵀ       b_j ← λb_j + w y_j z        β_j = A⁻¹ b_j        decayed sums, not means
A₀ = ridge·I         b₀ = ridge·coef_prior
```

Because `A` is a sum, a row's weight is on the sum scale, and a heavier
stream outweighs the starting ridge sooner ([Weights](#weights)). Unlike
`ewridge`'s default, the starting ridge penalizes every coefficient, the
intercept included. The targets share one Cholesky factor of `A`, so a row
with any target null is scored but teaches none of them. So when the
targets go missing on different rows, give each its own `rls` spec.
Otherwise one spec holds them all.

```python
rls = po.spec.rls(
    "rls", targets=["y"], features=["x0", "x1"], clock="t", gap_cap=300.0, half_life=600.0,
    ridge=1e-3,                    # A starts at ridge * I (default 1)
)
```

**`rls` keeps the Cholesky factor of `A`, updated by Givens rotations** (the
*square-root form*). The textbook recursion on the inverse `P` loses
symmetry to rounding by a factor of `1/λ` per row, and one extreme row can
cancel `P` and freeze a coefficient for good.

#### `lasso` — lasso path, its penalty chosen as it runs

*API:* [`po.spec.lasso`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.lasso) — *Rust:* [`lasso.rs`](crates/online-core/src/lasso.rs) — *Outputs:* [fields](docs/OUTPUTS.md#lasso)

Use `lasso` when only some features should carry weight: its L1 penalty
sets the others to exactly zero. It fits a path of penalties from
`ewridge`'s centred running sums.

It runs coordinate descent on the standardized sums, warm-started along the
path and from one solve to the next. `C` is the features' correlation
matrix, and `c_i` is feature `i`'s covariance with the target over the
feature's standard deviation. For each penalty `l` in `lasso_path`, the
sweep repeats until no coefficient moves by more than `tol`:

```
ρ_i = c_i − Σ_{j≠i} C_ij β_j
β_i = soft(ρ_i, l·l1_ratio) / (C_ii + l·(1 − l1_ratio))        soft(v, t) = sign(v)·max(|v| − t, 0)
```

After the sweeps the slopes are unscaled, and the intercept is `ȳ − m·β`,
with `m` the features' means. The target is centred but not scaled, so the
L1 threshold `l·l1_ratio` is in the target's units. So scale `lasso_path`
with the target: for a pure lasso, a target ten times larger needs
penalties ten times larger to zero the same coefficients. A descent that
runs out of its `max_iter` sweeps is counted in `bank.solve_failures()`,
so if that count is above zero, give a larger `max_iter`.

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

**At each row, `lasso` reports in `penalty_selected_<target>` the penalty
whose exponentially weighted out-of-sample squared error is lowest so far,**
as it stood before the row. Predictions for every penalty are computed
anyway, so the selection adds no work of its own. To rank the penalties on
the plain mean of that error, give `select_half_life=inf`.

**Under a `window_size`, a feature that stays constant inside the window
gets a coefficient of exactly zero,** and the error that picks the penalty
covers only the window ([A hard window](#a-hard-window)).

#### `kalman` — random-walk-β dynamic linear model

*API:* [`po.spec.kalman`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.kalman) — *Rust:* [`kalman.rs`](crates/online-core/src/kalman.rs) — *Outputs:* [fields](docs/OUTPUTS.md#kalman)

Use `kalman` when a relationship moves faster than a half-life can follow,
or when a coefficient should be forgotten once the rows stop supporting it.

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
This code uses `df` from [Example data](#example-data):

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
fit = po.ModelBank([revert]).fit_predict(df)
```

**By default each coefficient drifts as a random walk: between rows its
estimate holds and its variance grows.** With `h_i` its `coef_half_life`, a
row Δ clock units after the one before adds `σ²(ln2 · Δ / h_i)²` to that
variance on standardized features, matching EW-RLS's steady state at any
spacing. For a rate per coefficient, give `coef_half_life` a list with the
intercept's entry leading, and an entry of `inf` pins its coefficient. To
set the process noise outright, give `q=`, whose `q_i` is added as
`q_i · Δ²`.

**For a regressor that is active only now and then, give its coefficient a
finite `revert_half_life`.** The coefficient is then pulled toward zero on
every row, while the rows that support it pull it back (an AR(1) prior). So
the effect is forgotten between its bursts, and cannot persist through a
run of null targets. For a regressor that is always active, keep the default
`inf`: the pull would settle a persistent effect below its true size, the
more so the shorter the reversion half-life. A reverting coefficient's
long-run prior variance, at rows `Δclock` apart, is
`q_i·Δclock²/(1−φ_i²)`, where a random walk's grows without bound.

**Under `standardize`, the default, the reversion pulls toward zero in the
standardized coordinates:** "no effect" for a slope, and "the target
averages zero" for the intercept. So unless the target averages zero, keep
the intercept's entry at `inf`, as in `[float("inf"), 50.0, 50.0]`.

**`predict` applies the reversion,** by the same `Φ` over the distance
from the last learned row, capped by `gap_cap`. A slope therefore keeps at
least `2^(−gap_cap/r_i)` of its value. To let a far prediction approach the
intercept, give a `gap_cap` that spans several reversion half-lives. Under
`standardize` the intercept is the target's level at the features' means.

With `standardize=False`, `q=0` and a fixed `obs_var=`, the filter is
exactly Bayesian linear regression: river's `BayesianLinearRegression`
agrees to 1e-13.

#### `huber` / `quantile` — robust regression

*API:* [`po.spec.huber`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.huber) and [`po.spec.quantile`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.quantile) — *Rust:* [`robust.rs`](crates/online-core/src/robust.rs) — *Outputs:* [huber](docs/OUTPUTS.md#huber), [quantile](docs/OUTPUTS.md#quantile)

Use these when one wild target must not move the fit: `huber` down-weights
the rows that miss by far, and `quantile` fits a conditional quantile, such
as the median, which outliers cannot pull. Both read each row's residual
against the fit *before* the row is learned, so both stay out-of-sample.
Both take `ewridge`'s `ridge=` as a single number, and refuse a list.

`r` is the row's residual, `σ` the plain exponentially weighted standard
deviation of the residuals (1 until one exists), `δ` is `huber_delta` and
`τ` the `quantile` level. The *Gram* is `ewridge`'s `S`, and the
cross-moment its `r_j`:

```
huber:     the ridge update at weight  w · min(1, δσ / |r|)
quantile:  one Newton step on the check loss, smoothed by a uniform kernel of half-width h = quantile_eps · σ
           |r| < h:  a least-squares row with target  y + 2h(τ − ½)
           |r| ≥ h:  adds  w · 2h · ψ_τ(r) · z  to the cross-moment and nothing to the Gram,  ψ_τ(r) = τ − 1{r < 0},
                     bounded so the fit at the row moves by at most the row's residual
           h is never narrower than (k/n)^{2/5} · σ, for the target's effective sample n
```

```python
hub = po.spec.huber("hub", targets=["y"], features=["x0", "x1"], clock="t", gap_cap=300.0, half_life=600.0,
                    huber_delta=1.5)      # a residual beyond huber_delta * sigma is down-weighted
med = po.spec.quantile("med", targets=["y"], features=["x0", "x1"], clock="t", gap_cap=300.0, half_life=600.0,
                       quantile=0.5,      # the level, strictly between 0 and 1: 0.5 is a median regression
                       quantile_eps=0.2)  # the band's half-width, in units of sigma (default 0.2)
```

**The band's rows, those with `|r| < h`, give the Newton step its
curvature.** So a much smaller `quantile_eps` narrows the band and slows
the convergence, and a much larger one smooths the quantile toward the
mean. The floor on `h` keeps a short half-life from leaving too few rows
inside the band.

`quantile` takes least-squares rows in three cases:

| case | `quantile` takes least-squares rows |
|---|---|
| a target's first rows | until the target has three rows per coefficient, its present rows counted one each, decayed, whatever their weights |
| after a gap or a reset | by the same rule, which rebuilds the fit |
| a band holding less than one row per coefficient | until the band holds rows again, which rebuilds a fit that a row near the input bound, `1e100`, has moved |

**Both models are held to numpy references of their own recursions within
1e-14.** Against batch fits of the same objectives, scikit-learn's
`HuberRegressor` and statsmodels' `QuantReg`, they land close but not
equal, for three reasons:

| reason | applies to |
|---|---|
| each row's weight was set by the fit before it | both |
| `σ` is the plain spread rather than a robust one | `huber` |
| the check loss is smoothed over the band | `quantile` |

#### `sgd` — stochastic gradient descent

*API:* [`po.spec.sgd`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.sgd) — *Rust:* [`sgd.rs`](crates/online-core/src/sgd.rs) — *Outputs:* [fields](docs/OUTPUTS.md#sgd)

Use `sgd` when each row must take only O(k) time: it takes one gradient
step per row and never solves. It is the only model here that takes count
targets (`loss="poisson"`). With `Gᵢ` adagrad's running sum of squared
gradients, a row takes this step:

```
eta = zᵀβ        p = link(eta)        d = dL/d eta
g₀ = d·w         gᵢ = d·zᵢ·w + l2·βᵢ     each clipped to ±clip_gradient; the intercept takes no l2
βᵢ -= lrᵢ·gᵢ     lrᵢ = learning_rate                           schedule="constant"
                     = learning_rate/(1 + weight_sum)^power    schedule="inv_scaling"
                     = learning_rate/(√Gᵢ + 1e-8)              schedule="adagrad"
```

**To set how many rows the coefficients remember, choose `learning_rate`.**
Every row's step moves the coefficients and the half-life does not decay
them, so under a constant rate they remember about
`1 / (learning_rate · E[z²])` rows, whatever the clock between rows. A
constant rate fits the same slopes at a half-life of 10 or 10,000. Under
`schedule="adagrad"`, `Gᵢ` decays on the clock, so the adapted rate opens
up again after a long gap.

The loss sets the link and the gradient:

| loss | link | `dL/d eta` | its constant |
|---|---|---|---|
| `squared` | identity | `p − y` | |
| `huber` | identity | `clamp(p − y, ±delta)` | `delta` is `huber_delta=`, in the target's units |
| `quantile` | identity | `1{y < p} − τ` | `τ` is `quantile=`, the level, between 0 and 1 |
| `epsilon_insensitive` | identity | 0 inside the tube, else `sign(p − y)` | the tube's half-width is `eps=`, in the target's units |
| `poisson` | log, for count targets | `p − y` | |
| `logistic` | sigmoid, for 0/1 targets | `p − y` | |

**Under `loss="poisson"`, keep `clip_gradient`, `1e3` by default,** because
through the log link one large count would make the next gradient
exponentially bigger. At ordinary scales it does not bind for the other
losses.
This code uses `df` from [Example data](#example-data):

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
assert min(last[1:]) >= 0.0 and abs(sum(last[1:]) - 1.0) < 1e-12   # the last row's slopes: at least 0, summing to 1
```

**To bound the slopes, give `coef_min` and `coef_max` (one number, or one
per feature, with `-inf` or `inf` for an open side) and `coef_sum` for
their total:**

| setting | what it expresses |
|---|---|
| `coef_min=0, coef_sum=1` | a long-only, fully invested portfolio |
| `coef_min=0` alone | a sign the model must respect |
| `coef_min` equal to `coef_max` | a slope pinned at a known value |

After every step the slopes move to the nearest point that satisfies every
bound (the Euclidean projection). The fit starts from zero, projected, so a
fit on the simplex starts at uniform weights. The intercept is never
constrained, and `coef` reports what the projection returned, in the
caller's units even under `standardize=True`.

#### `pa` — passive-aggressive regression

*API:* [`po.spec.pa`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.pa) — *Rust:* [`pa.rs`](crates/online-core/src/pa.rs) — *Outputs:* [fields](docs/OUTPUTS.md#pa)

Use `pa` to step on every row without tuning a learning rate. Each row asks
the fit to come within `eps` of its target, and the update is the smallest
change that does so:

```
loss = max(0, |y − p| − eps)      s = ‖z‖²
pa    τ = loss / s          pa1  τ = min(c, loss/s)      pa2  τ = loss / (s + 1/(2c))
β    += min(w, 1) · τ · sign(y − p) · z
```

A row weight below 1 scales the step, and a weight above 1 counts as 1.
Where outliers are possible, keep a `mode` that caps or damps the step:

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

**`pa` keeps no running sums, so its `half_life` changes when `min_weight`
lets a target report, and nothing in the fit:** it decays only `weight_sum`
and each target's own weight.

**When `coef_min`, `coef_max` or `coef_sum` bound the slopes, keep `c`
small.** The projection that follows each update keeps the step from
meeting the row's margin exactly, and a truth outside the allowed set is
never reached, so the fit keeps stepping against the bounds. A small `c`
keeps those steps small.

#### `ftrl` — online logistic regression

*API:* [`po.spec.ftrl`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.ftrl) — *Rust:* [`ftrl.rs`](crates/online-core/src/ftrl.rs) — *Outputs:* [fields](docs/OUTPUTS.md#ftrl)

Use `ftrl` for a 0/1 target, or for a sparse fit that steps on every row
instead of solving. It runs FTRL-proximal (McMahan et al. 2013), a gradient
method whose L1 penalty zeroes a coefficient with too little evidence, and
whose per-coordinate rates adapt to each feature's history. Choose the use
with `loss`:

| `loss` | `pred` | the model |
|---|---|---|
| `"logistic"`, the default | a probability, with `resid = y - p` | logistic regression |
| `"squared"` | the linear prediction | a sparse linear regression with no solves and an L1 penalty |

Under the logistic loss a target outside `[0, 1]` is clamped into it. To
refuse such a target instead, give `strict_binary=True`, which refuses a
chunk with any target but 0 or 1 and names the row.

`b` is the coefficients, `zz`, `n` and `d` are the per-coordinate sums, and
`α` and `β` are its `alpha` and `beta`:

```
zz_i ← λzz_i      n_i ← λn_i      d_i ← λd_i
b_i  = 0 if |zz_i| ≤ l1·m else −(zz_i − sign(zz_i)·l1·m) / (β/α·m + d_i + l2·m)
p    = sigmoid(zᵀb)      g_i = (p − y)·z_i·w      s_i = (√(n_i + g_i²) − √n_i)/α
zz_i += g_i − s_i·b_i      n_i += g_i²      d_i += s_i
m    = W / W*, per target: W its weight, W* its weight on a clock that runs only on the rows that teach it
```

This code uses `df` from [Example data](#example-data):

```python
df_up = df.with_columns(up=(pl.col("ret") > 0).cast(pl.Float64))      # df with a 0/1 target: did ret rise?
rise = po.spec.ftrl(
    "rise", targets=["up"], features=["x0", "x1"], half_life=500.0,
    loss="logistic",             # the default
    alpha=0.1, beta=1.0,         # the learning-rate scale and its smoothing (the defaults)
    l1=0.01, l2=0.0,             # l1 zeroes a coefficient whose evidence is below it (defaults: l1=0, l2=1)
    strict_binary=True,          # this target is 0 or 1, so refuse anything else
)
fit = po.ModelBank([rise]).fit_predict(df_up)
p_up = fit["rise"].struct.field("pred_up")   # per row: the probability that ret rises, before the row
```

**`ftrl`'s penalties keep a fixed mass, while the evidence grows with the
weights and the rows' density.** So a heavier or denser stream overcomes
them sooner, and a shorter half-life shrinks the fit more. At the defaults,
with unit weights and rows one clock unit apart, they act in steady state
as a ridge of `(1 − λ)(β/α + l2)` on the mean scale. A constant target of 5
then settles at 4.65 at `half_life=100`, and at 4.96 at 1000. For a
penalty that weight and density leave alone, use
[`lasso`](#lasso--lasso-path-its-penalty-chosen-as-it-runs), whose penalty
is on the mean scale.

**A row whose target is null or whose weight is 0 leaves that target's fit
where it was,** because it ages the sums and the penalties alike. The rows
that teach the target bring `m` back toward 1, which restores the
penalties.

With decay off, `m` is 1 and `d_i` is `√n_i/α`, which is river's FTRL: the
two agree to 1e-12, row for row, and Vowpal Wabbit's `--ftrl` agrees to its
single precision.

#### `holt` — Holt's linear trend

*API:* [`po.spec.holt`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.holt) — *Rust:* [`holt.rs`](crates/online-core/src/holt.rs) — *Outputs:* [fields](docs/OUTPUTS.md#holt)

`holt` is the only linear model that takes no features: it extrapolates the
target's own level and trend. To measure what a model's features add, run
`holt` in the same bank, then compare the two models' `sigma` ([Per-row
diagnostics](#per-row-diagnostics)), or let a
[`seqtest`](#seqtest--a-sequential-test-of-a-sign-by-betting) with `a` and
`b` say which predicts closer.
This code uses `df` from [Example data](#example-data):

```python
with_x = po.spec.ewridge("with_x", targets=["y"], features=["x0", "x1"], clock="t", gap_cap=600.0,
                         half_life=200.0, emit_sigma=True)     # sigma_y is written only when switched on
naive = po.spec.holt("naive", targets=["y"], clock="t", gap_cap=600.0,
                     level_half_life=200.0, emit_sigma=True)   # the same target, from its own past alone
gain = po.spec.seqtest("gain", targets=["y"], a="with_x", b="naive")   # evidence that the features predict closer
compared = po.ModelBank([with_x, naive, gain]).fit_predict(df)
```

Level and trend, `l` and `b`, are weighted means of what each row observes
and what the model forecast, with `W` and `V` the weight each has gathered.
`s` is the clock since the target was last observed, this row's step
included. So a row with a null target or weight 0 leaves the level and
trend where they were, and its step goes into the next observation's `s`:

```
λ_l = 0.5^(s/level_half_life)                  λ_b = 0.5^(s/trend_half_life)
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

**`holt` measures the trend per clock unit, and per second on a temporal
clock,** so on an irregular clock it extrapolates the right distance.

**Give the level's half-life as either `level_half_life` or the spec's
`half_life`:** they name one setting, and a spec with both is refused. To
make the trend the whole history's drift, give `trend_half_life=inf`.

**For a flat forecast, give `trend=False`:** simple exponential smoothing,
whose level is the target's exponentially weighted mean. It holds the
trend at zero, and refuses `trend_half_life`.

**`holt` has no seasonal term, so for a seasonal baseline, group the spec
on the phase.** Make the phase a column with Polars' `with_columns` in the
query before the bank, and name it as the spec's `group`. The bank then
fits one level and trend per hour.
This code uses `lf` from [Example data](#example-data):

```python
seasonal = po.spec.holt("seasonal", targets=["y"], clock="ts", gap_cap="1h",
                        level_half_life="2h", group="hour")   # one level and trend per hour of the day
by_hour = lf.with_columns(hour=pl.col("ts").dt.hour()).online.fit_predict([seasonal]).collect()   # the phase, as a column
```

### Moments and correlation

These models track how the columns you name vary and move together: their
means, variances and correlations, for a few columns or for thousands. Each
row's values are read from the state before that row, so they can serve as
features for that row without leaking it.

#### `ew_cov` — exponentially weighted moments

*API:* [`po.spec.ew_cov`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.ew_cov) — *Rust:* [`ewcov.rs`](crates/online-core/src/ewcov.rs) — *Outputs:* [fields](docs/OUTPUTS.md#ew_cov)

Reach for `ew_cov` for the running means, variances and correlations of the
columns you name. It takes one O(k²) step per row, where Polars expressions
alone take O(k²) *passes over the data* for every pairwise exponentially
weighted correlation.

The model keeps `m` and `C`, the weighted means of `x` and of
`(x − m)(x − m)ᵀ`. It updates them in Welford's centred form, because the raw
form, `E[x xᵀ] − m mᵀ`, loses a variance when the columns sit far from zero:

```
W' = λW + w      a = λW / W'      b = w / W'      δ = x − m          (a + b = 1)
m' = m + b·δ     C'ᵢⱼ = a·Cᵢⱼ + a·b·δᵢδⱼ
varᵢ = Cᵢᵢ       covᵢⱼ = Cᵢⱼ      corrᵢⱼ = Cᵢⱼ / √(CᵢᵢCⱼⱼ)
```

Give `stats` the statistics each row writes, `["mean", "std", "corr"]` when
left out:

| in `stats` | what it reports | needs |
|---|---|---|
| `mean`, `var`, `std`, `cov`, `corr` | the moments, per column or per pair | |
| `partial_corr` | the correlation of two columns with all the others held fixed, read off `(C + s·prior·I)⁻¹` in O(k³) time, spent only when asked | `precision_prior` |
| `mahal` | the Mahalanobis distance | `precision_prior` |
| `lagcorr` | the correlation of one column with another `ℓ` rows back | `lags=` |

For a wide set of columns, give `stats=[]`, so that each row writes no
statistics, and read the moments from the state after the run with
`bank.gram("mv")`.
This code uses `df` from [Example data](#example-data):

```python
mv = po.spec.ew_cov(
    "mv", features=["x0", "x1", "x2"], clock="t", gap_cap=300.0, half_life=500.0,
    stats=["mean", "std", "corr", "partial_corr", "mahal"],   # what each row writes
    precision_prior=1e-6,        # needed by partial_corr and mahal; fades as data arrives
    mahal_quantiles=[0.99],      # adds mahal_q0.99; needs "mahal" in stats
    pca=1, pca_every=20,         # one principal component, its loadings refreshed every 20 rows
)
scores = po.ModelBank([mv]).fit_predict(df).unnest("mv")   # mean_x0, ..., corr_x0_x1, ..., mahal, pc0_*
odd = scores.filter(pl.col("mahal") > pl.col("mahal_q0.99"))   # the joint outliers: every column in range, the combination not
first = scores.select("pc0_share", "pc0_loading_x0", "pc0_loading_x1", "pc0_loading_x2", "pc0_score")
```

**`mahal` measures the row's distance from the mean in standard deviations,
allowing for the correlations.** It is `√(δᵀ (C + s·prior·I)⁻¹ δ)`, with
`δ = x − m` and `s` a scale on the prior that decays with the co-moments.
With one column it is `|z|`. On Gaussian columns `mahal²` is only roughly χ²
with `k` degrees of freedom, since the moments are estimated and the prior
adds a little.

**To flag joint outliers without assuming a distribution, give
`mahal_quantiles=[0.99]` and filter the bank's output on
`mahal > mahal_q0.99`,** as `odd` does, to keep about one row in a hundred.
`mahal_q0.99` is the exponentially weighted 0.99 quantile of the past scores,
at the `half_life` and within 0.78%.

**Each eigendecomposition takes O(k³) time, so give `pca_every` to refresh
the principal components on a schedule.** `pca_every=20` refreshes them every
20 learned rows, where the default is every row, and scores the rows in
between on the last loadings. With `pca=1`, each row writes `pc0_var`,
`pc0_share` (of the trace), `pc0_loading_<feature>` and `pc0_score`. Each
refresh keeps the previous sign, so a loading never flips.

**`window_size` cuts the history off at a fixed age** ([A hard
window](#a-hard-window)). For a frame that fits in memory, Polars'
`df.rolling("t", period="3h").agg(...)` with an exponential weight gives the
same numbers to 1e-14, but recomputes every window, in `O(n·W)` time, where
the spec takes `O(n)`. So use the spec for a stream, a state to save, or
speed: measured, it ran 24 times as fast at a 74-row window and 1,100 times
at a 4,680-row one.

**Give `lags` to see how each column moves with the others `ℓ` learned rows
back in its group.** With `W` and `m` the weight and mean before the row, and
both deviations taken against that mean:

```
C_ℓ' = a·C_ℓ + a·b·(x_t − m)(x_{t−ℓ} − m)'      the co-moments' a and b, so lag 0 would be C exactly
```

Give `lags` as a strictly increasing list of whole numbers `>= 1`. Add
`"lagcorr"` to `stats` to write a correlation for each lag and each *ordered*
pair, since a lagged matrix is not symmetric: a leading b is not b leading a.

**A session change, or a clock gap beyond `gap_cap`, empties the ring of past
rows that the lags read**, because after either, "the row `ℓ` back" no longer
means a row `ℓ` ago. The means, the co-moments and `weight_sum` stay. A
zero-weight row ages the matrices without entering the ring.
This code uses `df` from [Example data](#example-data):

```python
lagged = po.spec.ew_cov(
    "lagged", features=["x0", "x1"], half_life=500.0,
    lags=[1, 5],                 # in learned rows within the group
    stats=["corr", "lagcorr"],   # lagcorr_<a>_<b>_l<l> for each lag and ordered pair
)
bank = po.ModelBank([lagged])
lead = bank.fit_predict(df).unnest("lagged")
moments = bank.gram("lagged")[0]   # the same numbers from the state: "lags", and "lag_comoments" as an (L, k, k) array
```

#### `marginal` — every pair's moments, kept in the state

*API:* [`po.spec.marginal`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.marginal) — *Rust:* [`marginal.rs`](crates/online-core/src/marginal.rs) — *Outputs:* [fields](docs/OUTPUTS.md#marginal)

Reach for `marginal` to screen thousands of features against a few targets.
It keeps the exponentially weighted moments of each (feature, target) pair on
its own, with `ew_cov`'s arithmetic, so a pair's `corr` equals a two-column
`ew_cov`'s to the bit. For `p` features and `T` targets that takes O(p·T)
time per row, where one `ew_cov` over all the columns would take
O((p + T)²). Each row writes only `weight_sum`, and `bank.marginal()` reads
the pairs from the state as a table after the run.

Each pair (feature `j`, target `t`) keeps its own `m_x`, `S_xx` and `S_xy`,
over its target's rows. Per target `t`, on a row where `y_t` is present, with
`W_t` the weight behind that target before the row:

```
W'_t = λW_t + w        a = λW_t / W'_t        b = w / W'_t        Q'_t = λ²Q_t + w²
S'_yy = a·S_yy + a·b·(y_t − m_y)²             S'_xx = a·S_xx + a·b·(x_j − m_x)²
S'_xy = a·S_xy + a·b·(x_j − m_x)(y_t − m_y)    m' = m + b·(value − m)
```

This code uses `df` from [Example data](#example-data):

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

For unit weights a step `Δclock` apart, `n_kish` tends to
`(1 + λ)/(1 − λ)`, where `λ` is that step's decay.

**A null target ages its own pairs (`W_t ← λW_t`, `Q_t ← λ²Q_t`) and teaches
them nothing.** A null feature skips the row, as in every model.

**`corr`, `beta` and `t` are null until the target's `W_t` reaches
`min_weight`**, 3 by default here, because two rows give a correlation of ±1
whatever the data. A statistic is null wherever it is undefined: all three
for a constant feature, and `t` at `n_kish ≤ 2`.

**Use `t` to rank pairs, and never as a p-value**, because the rows are
neither independent nor Gaussian. `t` is built on `n_kish`, the right count
for unequal weights, which says nothing about rows that resemble their
neighbours. So on a smooth stream it claims evidence that is not there:
two *independent* AR(1) series with `φ = 0.9` and `0.8` come out at
`t = 2.39`.

Five settings extend the pairs, the first two with views that are off by
default ([docs/MARGINAL-LAGS-AND-BINS.md](docs/MARGINAL-LAGS-AND-BINS.md)
derives both):

| setting | what it adds |
|---|---|
| `lags`, with `serial_rule` | the pair's moments at each lag: a count that allows for rows that resemble their neighbours, and lead/lag correlations |
| `bins` | the target's moments inside each bin of a feature: the relations a correlation cannot see |
| `window_size` | the pairs over a recent stretch |
| `feature_moments="shared"` | speed: one feature moment for every target, and the same table where every target is on every row |
| `shards` | speed: one wide spec on every thread, and the same numbers at any count |

When a group closes, `bank.closed_groups()` carries the two views as
`pair_*` columns: `pair_split_gain` as a list over the pairs, and
`pair_lagcorr_xx` and `pair_bin_n` as lists of lists.

**To correct the count for rows that resemble their neighbours, give `lags`
and a `serial_rule`.** Each pair then keeps its moments at each lag, the
statistic `ew_cov(lags=)` computes, to the bit, under
[`ew_cov`'s](#ew_cov--exponentially-weighted-moments) rules for the ring.
`serial_rule` turns them into a count that allows for the resemblance, after
Bartlett (1935): `n_serial = n_kish / (1 + 2 Σ_l ρ_x(l)·ρ_y(l))`. Under
`"geometric"`, the two AR(1) series that gave `t = 2.39` come out at
`t_serial = 1.03`: no evidence, which is the truth.

| `serial_rule` | the sum over the kept lags | suits |
|---|---|---|
| not given | not taken: `n_serial` and `t_serial` are null | |
| `"truncated"` | as it stands; null where the bracket reaches zero or below, as for two series whose autocorrelations have opposite signs | dense lags |
| `"bartlett"` | lag `l` weighted `1 − l/(L + 1)`, `L` the longest kept lag, as Newey and West weight it, so the noisiest lags count least | dense lags, where the long ones are noisy |
| `"geometric"` | `ρ(l) = φˡ` fitted to each series, and the tail summed in closed form, `2φ_xφ_y/(1 − φ_xφ_y)`; `phi_x` and `phi_y` are reported | exponentially weighted series, whose lags need not be dense |

This code uses `df` from [Example data](#example-data):

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
# n_serial                    n_kish over the serial_rule's bracket
# t_serial                    the same statistic as t, against that count
# phi_x, phi_y                the fitted per-row decays, under serial_rule="geometric"
```

**Read lead and lag from the cross terms only when the target is measured at
the same time as the feature.** Then a feature whose `lagcorr_yx[0]` beats
its `corr` leads its target, and one whose `lagcorr_xy[0]` does follows it.
Against a target that looks ahead, read `corr` alone. There every timely
feature built from the same news shows `lagcorr_xy` above `corr`, because the
target `l` rows back is built partly from the feature's newest `l` rows.

**Give `bins` to see the relations a correlation cannot.** A feature can be
strongly related to a target with `corr` at zero: a threshold, a V, a
saturation. With `bins`, each pair keeps the target's moments inside each bin
of the feature, and all three become visible. The edges are fixed once and
never move, set by one of three rules:

| the edges | how they are set |
|---|---|
| `bin_rule="quantile"` | learned from the first `bin_warm_rows` rows (default 1,000), by weighted quantile |
| `bin_rule="fixed"` | equal widths |
| `bin_edges=` | given outright, a list per feature or a dict by name: exact and comparable across runs, and refused beside `bins`, `bin_rule` and `bin_warm_rows` |

The warm-up rows are replayed once the edges are set, so none is lost.
This code uses `df` from [Example data](#example-data):

```python
binned = po.spec.marginal(
    "binned", targets=["y"], features=["x0", "x1"], half_life=500.0,
    bins=16,                     # bin each feature and keep the target's moments inside each bin
    bin_rule="quantile",         # equal weight per bin, the edges learned by weighted quantile
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
`split_gain` is a regression stump's R², so its excess over `corr²` is the
nonlinear surplus. As a p-value, `split_gain_t` would overstate the evidence,
because the cut was chosen as the best of the candidates.

**A feature keeps only the bins it can support.** Under
`bin_rule="quantile"`, a value that carries more than a bin's share, such as
an indicator's zero, fills a bin of its own, and the rest share what is left.
So the 5% of rows that carry the signal are not lost among the zeros. A
binary feature has two bins, and a constant one has a single bin and no
split.

**Binning keeps `O(bins)` of state per pair, and takes one search of the
edges per feature per row,** so ten thousand columns bin in the pass that
gives their `corr`. A spec whose held rows or histogram would pass 256 MiB,
per group and per half-life, is refused when it is built. To allow more,
give a larger `bin_budget`, or `float("inf")` for no limit.

**Give a screen a `window_size` when a relation may flip sign**, because two
regimes of opposite sign average to nothing over a long history. The window
truncates every pair moment, so `corr`, `beta` and `t` describe the rows
inside it ([A hard window](#a-hard-window)). `weight_sum`, in the record and
in the table, is the weight inside the window. A window refuses `bins`, so
bin the features in a second spec without one.

**To use `lags` under a window, give `window_lags=True`; otherwise the spec is
refused.** Each snapshot then holds the lag moments beside the pair moments:
twice its size at one lag, and six times at five with the default cross lags.
A windowed lag moment is an estimate, while the windowed pair moments are
exact.

**With several targets present on every row, give
`feature_moments="shared"` for speed.** Each feature then keeps one mean and
variance, over every learned row, in place of one per pair over its target's
rows. At 20,000 pairs that runs 2.7 times as fast at ten targets, with a
state under half the size and the same table to the bit ([PERFORMANCE
§27](docs/PERFORMANCE.md#27-marginals-shared-feature-moments-e72-task-125-2026-09-29)).
Where a target is absent on some rows, `var_x` and the centre of `cov` then
come from every row. That is a different estimator, sound only where the
absence says nothing about the feature, so otherwise keep the default. It
takes `lags`, and refuses a `window_size`.

**Give `shards="auto"` to spread one wide spec over the threads.** The bank
runs groups and specs in parallel ([Parallelism](#parallelism)), so a wide
`marginal` on one group is otherwise one thread's work. `shards=10` splits
its pairs into ten ranges of features, each run on a thread of its own, a
batch of rows at a time. `"auto"` sizes the split to the width and the pool,
and leaves whole a narrow spec, or a windowed one at the default
`window_every` of one, whose batch is a single row. The numbers are the same
to the bit at any count, so a saved bank resumes under any count. On 14
threads
([PERFORMANCE §25](docs/PERFORMANCE.md#25-a-wide-marginal-split-across-the-pool-e73-task-126-2026-09-25)):

| the spec | `"auto"` ran the bank |
|---|---|
| 10,000 features, nine targets, lags and bins | 4.9 times as fast |
| the moments of one target alone | 1.2 times as fast, since the bank's own work on each row does not split |

#### `deco` — one correlation for the whole matrix

*API:* [`po.spec.deco`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.deco) — *Rust:* [`deco.rs`](crates/online-core/src/deco.rs) — *Outputs:* [fields](docs/OUTPUTS.md#deco)

Reach for `deco` (Engle & Kelly 2012) for one number that moves with the
correlation of many series. A stream cannot keep every entry of their
correlation matrix moving without O(m²) work a row. So `deco` replaces the
`m(m−1)/2` free entries of the matrix of `m` series with their average, the
*equicorrelation*, and estimates that in O(m) a row, for the whole matrix or
per block.

The row is standardised against the means and variances as they stood before
it, `r_i = (x_i − m_i)/√v_i`. With `S₁ = Σ r_i` and `S₂ = Σ r_i²` over the
`n` features that have a standardised value, the row's estimate `u` is Engle
and Kelly's Lemma 2.3. The reported level, `rho`, then follows `u` under one
of two dynamics, with `α` and `β` the `"linear"` dynamic's two weights:

```
u = (S₁² − S₂) / ((n − 1)·S₂)        = (mean of r_i·r_j over i ≠ j) / (mean of r_i²)

"ew":      W' = λW + w,  b = w/W'      ρ' = ρ + b·(u − ρ)                       ρ = u at W = 0
"linear":  ρ' = (1 − α − β)·ρ̄' + α·u + β·ρ      (ρ̄ the "ew" level)          ρ' = ρ at w = 0
```

| `dynamics` | `rho` is | rules |
|---|---|---|
| `"ew"` | the exponentially weighted mean of `u`, decayed on the model's clock: exactly what an `ew_cov(stats=["mean"])` over the rows that have a `u` would report | |
| `"linear"` | the paper's eq. 21, its intercept written as `(1 - alpha - beta) * rho_bar`, since a stream has no sample to fit a free one on. It steps once per row, as a DCC (dynamic conditional correlation) model does, so a capped gap moves it as far as one row's `α·u` | needs `alpha=` and `beta=`, each `>= 0`, with `alpha + beta < 1`. A row's weight reaches `rho` only through `rho_bar`, since the paper's recursion has no row weights |

**Give `blocks` for one number per block and one per pair of blocks**, the
useful middle between one correlation and all of them. Put every feature in
exactly one block, and at least two in each, or the spec is refused.
This code uses `df` from [Example data](#example-data):

```python
eq = po.spec.deco(
    "eq", features=["x0", "x1", "x2"], clock="t", gap_cap=300.0, half_life=500.0,
    dynamics="ew",               # the EW mean of u; "linear" needs alpha= and beta=
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

**Use `u` and `rho` as a signal that moves with the market's correlation, and
for the correlation itself read an `ew_cov`'s `corr`.** Both read low,
because `u` is a ratio of two averages, and the mean of a ratio is not the
ratio of the means. As the paper says, `E[u]` is about 0.20 for a true 0.30
at six columns. `rho`, an average of `u`, carries the same bias:
[docs/REGIMES.md §7](docs/REGIMES.md#7-deco-an-equicorrelation-that-moves)
measures it settling near two-thirds of the truth.

**A column constant from its first row is left out of its block,** since it
has no standardised value. The block reads the correlation among its other
columns, and a block left with fewer than two has no `u`. `loglik` needs
every column, so leave a constant column out of `features`, or it nulls
`loglik` on every row.

#### `rcov` — a block's realised covariance, robust to noise

*API:* [`po.spec.rcov`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.rcov) — *Rust:* [`rcov.rs`](crates/online-core/src/rcov.rs) — *Outputs:* [fields](docs/OUTPUTS.md#rcov)

Reach for `rcov` for each block's realised covariance from its tick returns,
robust to measurement noise and to series not observed together. It requires
`group`, naming the block column, and `group_close`, and writes each block's
estimate when the block closes, as that block's row of
`bank.closed_groups()`. Each row writes only `weight_sum`. A block's close
ends its memory, so `rcov` refuses a `half_life` or `lam`. A `weight` must be
0 or 1, since a sum over returns has no fractional row, and any other value
is an error naming the row.

The plain realised covariance, the sum of the returns' outer products, goes
wrong over real ticks in two ways. Each price is the efficient price plus a
measurement error, whose variance accumulates with every tick, and series not
observed together have their correlation pulled toward zero. Published
estimators remove both with sums over lags, which a stream can accumulate.
In the kernel, `Gamma_h` sums the products of returns `h` rows apart, up to a
bandwidth `H`. Give `kind` to pick an estimator.
[docs/REGIMES.md](docs/REGIMES.md) §8 measures the three against each
block's truth:

| `kind` | the estimator | use it on |
|---|---|---|
| `"plain"` | `sum(x x')`, equal to `n` times an `ew_cov(lam=1)`'s uncentred second moment at close, to the bit: the cross-check, and the reference the other two are measured against | clean returns, where it has the smallest error |
| `"kernel"` | the multivariate realised kernel (Barndorff-Nielsen, Hansen, Lunde & Shephard 2011), `sum_h k(h/(H+1)) Gamma_h` with Parzen weights and jittered end points | returns with measurement noise, which barely moves its error |
| `"preavg"` | the modulated realised covariance (Christensen, Kinnebrock & Podolskij 2010): returns pre-averaged over `k_n = ceil(theta * block_rows^0.6)` rows | series not observed on every row, where it is the least biased |

**Compute the returns from prices with Polars' `diff()` in the query before
the bank, over each block with `.over("block")`,** so that no return spans two
blocks. The bank skips each block's first row, whose return is null. For a
series not observed on every row, carry its last price forward with
`forward_fill().over("block")` before the diff, and use `kind="preavg"`. Or
build a common grid with `po.stream.refresh_time` before the diff ([Series
that tick at their own times](#series-that-tick-at-their-own-times)), and
keep the kernel.

```python
rk = po.spec.rcov(
    "rk", features=["x0", "x1"],       # x0 and x1 stand in for two prices, differenced in `returns`
    group="block", group_close="monotone",
    kind="kernel",               # the realised kernel; or "plain" or "preavg"
    psd=True,                    # the default: clip any negative eigenvalue
    block_rows=2000,             # a sizing hint for the ring
    bandwidth=None,              # a fixed H; left out, set from the data
)
returns = by_block.with_columns(              # by_block: df with a block id rising every 100 rows
    pl.col("x0", "x1").diff().over("block")   # each row's change since the row before it, within its block
)
bank = po.ModelBank([rk])
bank.fit_predict(returns)
finished = bank.closed_groups()
# rcov, rcorr                  vech of the upper triangle: its entries as one flat list
# rcov_n, rcov_kind, bandwidth_used
# omega2, iv_sparse            the noise variance and sparse integrated variance behind the bandwidth
# iq                           a realised-quarticity proxy, labelled one
# psd_repaired                 whether the estimate had to be made positive semi-definite
# a block too short to estimate from gives nulls, not an error

# A series not observed on every row: carry its last price forward within the block, then difference,
# and pre-average.
held = by_block.with_columns(pl.col("x0", "x1").forward_fill().diff().over("block"))
pre = po.spec.rcov("pre", features=["x0", "x1"], group="block", group_close="monotone",
                   kind="preavg", block_rows=100)
po.ModelBank([pre]).fit_predict(held)
```

**The estimate reads no future row.** The kernel's jittered end points
average observations already in the state at close, and a product enters
`Γ̂_h` only once both its legs are final.

**`psd=True`, the default, clips any negative eigenvalue and reports
`psd_repaired`.** For a correlation read on its own under `"preavg"`, give
`psd=False`. It selects the balanced form, a shorter window of
`k_n = floor(theta * sqrt(block_rows))` rows with the bias term subtracted,
which was the more accurate on
[docs/REGIMES.md §8](docs/REGIMES.md#8-rcov-three-estimators-against-each-blocks-truth)'s
streams. It is not always positive semi-definite, so keep the default where
the matrix must be.

**Fix the bandwidth `H` with `bandwidth=`; otherwise give `block_rows`, the
block's expected length in rows, and `H` is set from the data** as
`H = ceil(c* xi^(4/5) n^(3/5))` with `c* = 3.5134`. `block_rows` is a sizing
hint for the ring: a longer block runs, clipped, and reports
`bandwidth_used`. `"preavg"` needs `block_rows` unless `preavg_rows=` is
given.

**A clock gap past `gap_cap`, or a session change, splits a block into
stretches**, and no product pairs two returns across the break.

### Clustering and classification

These models put a label on each row: a cluster found in the features alone,
or a class learned from a label column. Every label, distance and probability
is read before the row is learned from, so it is out-of-sample like every
prediction here.

#### `kmeans` — exponentially weighted k-means

*API:* [`po.spec.kmeans`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.kmeans) — *Rust:* [`cluster/kmeans.rs`](crates/online-core/src/cluster/kmeans.rs) — *Outputs:* [fields](docs/OUTPUTS.md#kmeans)

Reach for `kmeans` when you know the number of clusters and they are round.
For clusters of any other shape, or of a number you do not know, use
[`micro`](#micro--density-based-clustering-any-shape). It labels each row
with the nearest of `k` centres, each the exponentially weighted mean of the
rows assigned to it (`ew_cov`'s mean recursion, per cluster). Three rules
keep the centres on the data: seeding, a split–merge move that finds a
cluster born after seeding, and a dead rule that re-places a centre whose
blob vanished.

With `c_j` centre `j` and `n_j` its decayed weight:

```
j*   = argmin_j ‖x − c_j‖²          distances in units of each feature's EW sd (standardize=False: raw units)
n'_j = λn_j + w                      c'_j = c_j + (w/n'_j)(x − c_j)     for j = j*
```

This code uses `df` from [Example data](#example-data):

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
    update_every=1,              # the default, sequential: learned rows between applying each centre's batch; more is mini-batch
)
out = po.ModelBank([km]).fit_predict(df).unnest("km")
# cluster   the nearest centre's label, before the row is learned from
# dist      the distance to it;  dist2  the distance to the second-nearest
# weight_sum, and coef: the centres, k rows of len(features)
index = po.spec.coef_index(km)   # how coef is labelled: target = "cluster0", "cluster1", ..., term = the feature
```

**Seeding waits for `warm_rows` rows, places the centres, and replays the
rows.** One k-means++ start lands in the wrong partition a third of the time
on five blobs in four dimensions. So the default, `seed_rule="lloyd"`, takes
ten starts, refines each by ten Lloyd iterations, and keeps the one of least
inertia. `seed=` fixes the random draws.

**While `split_merge` is above 0, a far row is scored but not learned from.**
A row is far when it sits outside its cluster, about four standard deviations
of `dist²` above the typical radius. Far rows wait for the split–merge move,
so they neither drag a centre nor widen its radius.

**The split–merge move finds a cluster born after seeding.** Every
`split_merge_every` rows, if the two closest centres are nearer than
`split_merge` times the sum of their radii, the model treats them as two
centres on one blob. Once at least three rows far from every centre carry 5%
of the weight learned since the last check, the move frees the emptier of
the two and places it on those rows. It needs `k` of at least 3, so at
`k=2` only the dead rule acts. It cannot see one centre that owns two blobs,
whose rows all sit within its own radius, so keep the default
`seed_rule="lloyd"`, which prevents that. `split_merge=0` turns off the move
and the dead rule, which leaves plain sequential k-means.

**Increase `dead_frac` when regimes change faster than a dead centre is
re-placed.** A centre whose blob vanished fades, and is re-placed once it holds
less than `dead_frac` of an equal share: `log2(1/dead_frac)` half-lives after
the blob vanished, 4.3 at 0.05 and 2 at 0.25. In exchange, a cluster lighter
than `dead_frac/k` of the stream loses its centre whenever any row is far.

**`scale_floor` keeps a feature that goes quiet from taking over the
distance.** It floors each feature's variance at a fraction of its long-run
variance, a tenth by default. After twenty half-lives of quiet, the
feature's floored weight in the distance has grown about 57 times, where
unfloored it would have grown a million times.

#### `micro` — density-based clustering, any shape

*API:* [`po.spec.micro`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.micro) — *Rust:* [`cluster/micro.rs`](crates/online-core/src/cluster/micro.rs) — *Outputs:* [fields](docs/OUTPUTS.md#micro)

Reach for `micro` when the clusters may have any shape. It finds them without
being told how many there are, flags the rows that belong to none, and follows
clusters that appear and vanish. It builds on DenStream's micro-clusters,
with a linking step over them.

Each micro-cluster is a *summary*: a decayed weight `n`, a centre `c`, and a
radius `r`, the exponentially weighted root-mean-square distance of its rows
from the centre. A summary is *established* once its weight reaches `beta_mu`
rows of the stream's mean row weight `w̄`. With `p` the number of features,
and distances in units of each feature's exponentially weighted standard
deviation, a row joins a summary only if the summary's radius stays within
`eps`:

```
a = n_j/(n_j + w̄)      b = w̄/(n_j + w̄)
n_j  ← λ n_j                                               every summary
j*   = the nearest established summary, if it keeps  a r²_j + a b ‖x − c_j‖² ≤ eps² p,
       else the nearest other summary, if it does,
       else a new one at x
n_j* ← n_j* + w     c_j* ← c_j* + (w/n_j*)(x − c_j*)     r²_j* ← min(·, eps² p)
```

The metric's variance is floored as
[`kmeans`](#kmeans--exponentially-weighted-k-means)'s is (`scale_floor`), so
the row on which a quiet feature moves again is not an outlier by that alone.

**Every `prune_every` rows, the light summaries are dropped and the
established ones linked.** Centres within `L` of each other share a label.
Give `macro_link=2` to link only summaries that touch. Left out, `L` is read
from the spacing the summaries show. A cluster's label is the smallest id
among its linked summaries, so it lasts as long as that summary does. A
summary whose rows stop lingers `half_life · log2(n / (beta_mu · w̄))`, with
`n` the weight it had. `max_clusters`, 200 by default, caps the live
summaries: past it the lightest is evicted, one not yet established first.
This code uses `df` from [Example data](#example-data):

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

**Set `eps` to the spread of one cluster, per standardized coordinate:**
about 0.07 for two-dimensional shapes, 0.3 for well-separated Gaussians in 20
dimensions. Either mistake shows in the outputs:

| what you see | `eps` is | what to do |
|---|---|---|
| nearly every row is an `outlier`, and `cluster` stays null | too small: no summary reaches `beta_mu` before it is pruned | increase `eps` |
| `n_micro` is about the number of clusters | too coarse: each cluster is one summary, so the derived `L` reads the spacing *between* clusters and bridges them into one | lower `eps`, or set `macro_link=2` |

**Set `beta_mu` from the arrival rate, because it counts rows**, as
DenStream's `MinPts` does. At half-life `h` and `v` rows per clock unit, the
stream's steady-state weight is about `1.44·v·h` rows. So for a summary meant
to hold a share `s` of the stream, give `beta_mu ≈ 1.44·s·v·h`.

Measured at 20k rows and a half-life of 3,000, moons, rings and five Gaussians
in twenty dimensions all score an adjusted Rand index (ARI) of 1.000 against
the truth. Their `eps` were 0.07, 0.1 and 0.3, and `kmeans` cannot follow the
first two. At `eps=0.07`, uniform noise is flagged `outlier` 94% of the time,
and real rows 0.3%. A cluster born mid-stream had a label 31 rows in.

#### `ew_class` — Gaussian classification on `ew_cov` moments

*API:* [`po.spec.ew_class`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.ew_class) — *Rust:* [`ewclass.rs`](crates/online-core/src/ewclass.rs) — *Outputs:* [fields](docs/OUTPUTS.md#ew_class)

Reach for `ew_class` to classify rows by a label column: name the column in
`label` and list its values in `classes`. It keeps one `ew_cov` state per
class, a weight `n_c`, a mean `μ_c` and a centred covariance `C_c`, and
scores a row by Bayes' rule over Gaussian classes. On three Gaussian classes
with their own covariances, its accuracy is within 0.001 of the Bayes rate,
and its probabilities are calibrated to about 0.01.

`π_c` is a class's share of the weight, and `r_c` the ridge `precision_prior`,
which makes a class scoreable from its first row and fades as `ew_cov`'s does:

```
π_c = n_c / Σ n         r_c = precision_prior · s_c        (s_c: the prior's fade, as in ew_cov)
M_c = C_c + r_c I  (full)      M = Σ π_c M_c  (shared)      diag(C_c) + r_c  (diagonal)
ℓ_c = ln π_c − ½ ln det M_c − ½ (x − μ_c)ᵀ M_c⁻¹ (x − μ_c)
p_c = exp(ℓ_c − max ℓ) / Σ exp(ℓ − max ℓ)                  class = argmax ℓ
n_c ← λ n_c + w·[y = c]        μ_c, C_c ← weighted Welford on the row's own class
```

Give `covariance` to set what each class keeps:

| `covariance` | what each class keeps | the right choice when |
|---|---|---|
| `"full"`, the default (QDA) | its own covariance, and its Cholesky factor, which only the row's own class refactors (under a `window_size`, every class on every row) | in general |
| `"shared"` (LDA) | one covariance pooled by class weight, factorized once per row | the classes differ in location but not in spread; it then labels more than 98% of rows as `"full"` does in the tests, with fewer parameters to learn |
| `"diagonal"` (Gaussian naive Bayes) | variances only | speed matters most, since it runs about five times as fast as the other two ([Throughput](#throughput)); it cannot see a correlation, so two classes with the same marginals and opposite correlations are one class to it |

**Give `window_size` when the class means move,** because over a long history
two regimes average together. Each class's moments then cover only the
window ([A hard window](#a-hard-window)).
This code uses `df` from [Example data](#example-data):

```python
labelled = df.with_columns(      # the label: "up" where y is positive, else "down"
    pl.when(pl.col("y") > 0).then(pl.lit("up")).otherwise(pl.lit("down")).alias("dir")
)
cl = po.spec.ew_class(
    "cl", features=["x0", "x1", "x2"], clock="t", half_life=200.0, gap_cap=300.0, min_weight=20.0,
    label="dir",                 # the label column
    classes=["down", "up"],      # declared up front
    covariance="shared",         # one covariance pooled by class weight
    precision_prior=0.1,         # required: the ridge that makes a class scoreable from its first row
)
out = po.ModelBank([cl]).fit_predict(labelled).unnest("cl")
out.select("dir", "class", "p_up", "weight_sum").tail(3)
# class        the most probable class, as a string
# p_<class>    one per declared class; exactly 0 for a class no row has carried yet
# coef         the class means, in the order of classes (coef_up_x0 after df.online.unnest([cl]))
```

**List `classes` as text, because the label is read as text.** For an
integer or boolean label column, give `["0", "1"]` or `["true", "false"]`. A
label not in `classes` is an error naming the row. A null label scores the
row and learns nothing from it, so `weight_sum` counts every row the model
accepts, labelled or not, while `π_c` counts the labelled rows' weights. When
labels arrive late, give `embargo` the delay, and each label is learned once
that much clock has passed ([Labels that arrive
late](#labels-that-arrive-late)).

### Sequential tests and regimes

These models weigh the evidence that a sign holds or that the correlations
have changed, date the current regime, and say which regime the stream is in
now.

#### `seqtest` — a sequential test of a sign, by betting

*API:* [`po.spec.seqtest`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.seqtest) — *Rust:* [`seqtest.rs`](crates/online-core/src/seqtest.rs) — *Outputs:* [fields](docs/OUTPUTS.md#seqtest)

Reach for `seqtest` to read evidence as it accumulates, and act the first time
it is enough. It tests whether a column tends to be positive, or whether one
spec predicts closer than another. A p-value cannot be read that way,
because checking it repeatedly inflates its error rate. An *e-process*,
which this model keeps, can be read at any row, as often as you like.

Each row counts as one trial, so `seqtest` refuses a `weight` or a
`half_life`. To skip a row, null its target. For a frame already in memory,
[`po.eval.seqtest`](https://hgilde.github.io/polars-online/eval.html#polars_online.eval.seqtest)
runs the same computation.

Per target it keeps the wealth of two gamblers, one betting that the next sign
is positive and one that it is negative. Each stakes the Krichevsky–Trofimov
fraction set by the counts so far, and stakes nothing unless the counts favour
its side:

```
s = sign(y)                    n⁺, n⁻: the signs counted before this row,  n = n⁺ + n⁻
λ⁺ = max(0, (n⁺ − n⁻) / (n + 1))          λ⁻ = max(0, (n⁻ − n⁺) / (n + 1))
ln E⁺ ← ln E⁺ + ln(1 + λ⁺ s)              ln E⁻ ← ln E⁻ + ln(1 − λ⁻ s)
```

**To test a column's sign, name it in `targets`.** A zero, null or NaN counts
as a tie: it bets nothing and counts nothing.
This code uses `df` from [Example data](#example-data):

```python
sign = po.spec.seqtest("sign", targets=["y"], group="stock_id")     # does y tend to be positive?
out = po.ModelBank([sign]).fit_predict(df).unnest("sign")
# log_e_pos_y, log_e_neg_y     the two gamblers' log wealth, as they stood before the row
# n_pos_y, n_neg_y             the signs counted so far
```

**To test which of two specs predicts closer, put both in the bank and name
them in `a` and `b`.** The sign tested is then `|resid_b| - |resid_a|`,
positive when `a` came closer, on the out-of-sample residuals the two specs'
output records report. A row where either side is null (warm-up, a skipped
row) sits out the test. When a side is a grid, pick its instance with
`a_suffix` or `b_suffix`, such as `"@h500"` or `"__r0.5@h500"` ([Output field
names](#output-field-names)). Without the suffix, the bank refuses the spec
and lists the grid's residual fields.
This code uses `df` from [Example data](#example-data):

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

**Reject on the first row where `log_e_pos ≥ ln 20`: that is the 5% level,
however often you look.** The null is that, given everything so far, the
next sign is no more likely positive than negative. Under it, `E⁺` is a
nonnegative supermartingale, and Ville's inequality gives
`P(E⁺ ever reaches 1/α) ≤ α`, however the rows depend on each other. No
distribution is assumed, and the size of the values is invisible: 60% small
gains and 40% huge losses count as "positive". For the two-sided question,
average the two sides' wealth: that is an e-value.
This code uses `df` from [Example data](#example-data):

```python
import math

signs = po.ModelBank([po.spec.seqtest("signs", targets=["y"], group="stock_id")]).fit_predict(df).unnest("signs")
rejected = (signs.filter(pl.col("log_e_pos_y") >= math.log(20))         # the rows at the 5% level, read at any row ...
            .group_by("stock_id", maintain_order=True).first())         # ... and each stock's first
e_value = (pl.col("log_e_pos_y").exp() + pl.col("log_e_neg_y").exp()) / 2   # two-sided: the mean of the wealth, not of its log
two_sided = signs.select("stock_id", e_value.alias("e_value"))
```

**Two events restart the test.** A session change restarts it under
`session_gap="reset"` or `group_close="session"`, and so does a step back
larger than `restart_after_step_back`. A session change with a numeric
`session_gap` does not.

Where the `max(0, …)` in a stake never binds, the wealth has the closed form
`2ⁿ B(n⁺+½, n⁻+½) / π`, the Beta(½, ½) mixture, and a test holds the bank to
it.

#### `corrchange` — has the correlation structure changed?

*API:* [`po.spec.corrchange`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.corrchange) — *Rust:* [`corrchange.rs`](crates/online-core/src/corrchange.rs) — *Outputs:* [fields](docs/OUTPUTS.md#corrchange)

Reach for `corrchange` to test whether the correlations among the columns you
name have changed. Give `kind` to pick one of its three tests:

| `kind` | the question | reports | against | a change is found |
|---|---|---|---|---|
| `"monitor"` | was the correlation constant over a span? | on a span's last row | the span's own correlation | at the span's end, up to `span_rows` rows late |
| `"sequential"` | has it left the level a stable history set? | on each row tested against the history, from the second on | a history of `span_rows` rows | as soon as it crosses a boundary |
| `"window"` | how big is the change? | on every row once both windows are full, `2·span_rows` rows in | the window before | as soon as the two windows differ |

Every kind writes the same five fields, `stat`, `crit`, `flag`,
`since_flag` and `since_change`. A parameter that belongs to another kind
is refused, naming the kinds it applies to.

Under `"monitor"` and `"sequential"`, the statistic is the largest over the
pairs, and the level `alpha` (0.05 by default) is spread over the pairs as
`alpha / npairs` unless `alpha_adjust="none"`. For one test of the average
correlation's level, however many columns, give either kind `scalar=True`,
which runs it on the equicorrelation of the standardised row (`deco`'s `u`).

**`kind="monitor"` tests each span of `span_rows` rows for a constant
correlation,** with the closed-sample constancy test of Wied, Krämer and
Dehling (2012). With `ρ̂_j` the correlation of the span's first `j` rows, and
`D̂` the delta-method long-run standard deviation of `ρ̂`, the last row of a
span reports, per pair:

```
Q = max_{2≤j≤T} (j/√T)·|ρ̂_j − ρ̂_T| / D̂
```

Under the null, `Q` converges to `sup|B|` for a Brownian bridge `B`, so the
critical value is the Kolmogorov quantile, the published 1.3581 at 5%. `D̂`
uses the paper's Bartlett kernel, lag `l` at `1 − l/γ` with `γ = ⌊ln T⌋`.
This code uses `df` from [Example data](#example-data):

```python
monitor = po.spec.corrchange(
    "break", features=["x0", "x1"],
    kind="monitor",              # the constancy test, span by span
    span_rows=500,               # nothing is reported until a span closes: a delay of at most this many rows
    alpha=0.05,                  # the level, the default
    scalar=False,                # True: one test on the equicorrelation instead of one per pair
)
out = df.online.fit_predict([monitor]).unnest("break")
# stat          the test statistic, null except where one is due: here, on a span's last row
# crit          the critical value it is compared against
# flag          true on the row where stat crossed crit
# since_flag    learned rows since the last flag
# since_change  on a flag, the rows from the first changed one through the flag's
```

**`kind="sequential"` tests each new row against a stable history,** with
the monitoring procedure of Wied and Galeano (2013). A cycle is `span_rows`
rows of history, taken as stable, then up to `monitor_rows` rows, each tested
against the history as it arrives. A flag, or the cycle's last monitored row,
ends the cycle, and the row after it starts a new history. From the history
it reads each pair's correlation `ρ̂_h` and its long-run standard deviation
`D̂`, with the monitor's estimator. The `k`-th monitored row then reports

```
V_k  = (k/√m)·(ρ̂_k − ρ̂_h) / D̂           m = span_rows, ρ̂_k over the k monitored rows
stat = max over pairs of |V_k| / w(k/m)
w(b) = (1 + b)·(b/(1 + b))^γ               γ = boundary_gamma, 0 ≤ γ < 1/2
```

and flags where `stat` passes `crit`.

The critical value is the paper's Eq. 7, `crit = (T/(1+T))^(1/2−γ)·q`, with
`T = monitor_rows/span_rows` and `q` a quantile of `sup_{0<s≤1} |W(s)|/s^γ`
for a Brownian motion `W`. At `γ = 0` and `T = 1`, `crit` is 1.5849 at 5%.
Where the paper simulates `q`, above `γ = 0`, this library solves it as a
diffusion with an absorbing boundary
([`boundary.rs`](crates/online-core/src/boundary.rs)), within 0.03 of the
paper's Table 1.

**Increase `boundary_gamma`, the `γ` of `w(b)`, to catch an early change
sooner.** Above 0 the boundary starts lower, and so it flags more stable
streams:

| `boundary_gamma` | a change soon after the history | stable streams flagged, at a nominal 0.05 |
|---|---|---|
| 0, the default | caught latest | 0.04 to 0.09, the size nearest nominal |
| 0.25 | caught sooner | 0.04 to 0.09 |
| 0.45 | caught soonest | 0.12 to 0.18 |

```python
watch = po.spec.corrchange(
    "watch", features=["x0", "x1"],
    kind="sequential",
    span_rows=500,               # 500 rows of history, assumed stable (kind="monitor" over them checks it)
    monitor_rows=1000,           # then each of up to 1000 rows tested as it arrives: the paper's T = 2
    boundary_gamma=0.25,         # between the size of 0 and the early catch of 0.45
)
```

**`kind="window"` measures how big the change is,** as
`||vech(R_pre - R_post)||` over two adjacent windows of `span_rows` rows
each. `R_pre` and `R_post` are the windows' correlation matrices, and `vech`
stacks their upper triangles into one list. It compares that with a
permutation quantile: `n_perm` shuffles of the pooled rows between the
windows, redrawn every `permute_every` rows. The shuffles move blocks of
`perm_block` rows, so rows that resemble their neighbours do not make the
null too liberal. To skip the permutations, give `crit=` as a fixed
threshold. The same `seed` gives the same critical values.

**Count a run of the window kind's flags as one change.** Two windows that
slide by one row are almost the same windows, so a statistic above the
quantile stays above it for a run of rows, and no rate of `alpha` per row
applies.

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

The monitor's size and power are close to Wied, Krämer and Dehling's tables,
on the `t_5` design of their own paper
([docs/REGIMES.md §2–3](docs/REGIMES.md#2-is-the-monitor-the-size-its-paper-says)).
The sequential detector's size is within two standard errors of Wied and
Galeano's Table 2 in every cell ([docs/REGIMES.md
§9](docs/REGIMES.md#9-the-sequential-detector-against-its-paper)).

#### `bocpd` — how long has this regime lasted?

*API:* [`po.spec.bocpd`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.bocpd) — *Rust:* [`bocpd.rs`](crates/online-core/src/bocpd.rs) — *Outputs:* [fields](docs/OUTPUTS.md#bocpd)

Reach for `bocpd` (Adams & MacKay 2007) to know how long the current regime
has lasted. It keeps a probability distribution over the **run length**, the
rows since the last break, so it can say "we are forty rows into a regime",
not only "something broke".

`bocpd` refuses a `half_life` or `lam`, because the run-length posterior does
the forgetting. Set how fast with `hazard`, the expected run length. With
`H = 1/hazard` the per-row chance of a break, and `π_r` the posterior
predictive of run length `r` for this row, their Algorithm 1 is:

```
growth:      P(r_t = r+1, x_{1:t}) = P(r_{t−1} = r, x_{1:t−1})·π_r·(1 − H)
changepoint: P(r_t = 0,   x_{1:t}) = Σ_r P(r_{t−1} = r, x_{1:t−1})·π_r·H
```

This code uses `df` from [Example data](#example-data):

```python
runs = po.spec.bocpd(
    "regime", features=["ret"], group="stock_id",
    hazard=250.0,                # the expected run length: H = 1/hazard is the per-row chance of a break
    prior_nu=2.0,                # the prior on the variance, as 2a ...
    prior_scale=[2e-4],          # ... and 2b: the one prior to set from your data
    emission="diag",             # the default: a normal-inverse-gamma per feature
    prune_below=1e-6,            # drop the runs holding less than this share of the mass
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

**Set `prior_scale` to `prior_nu` times the variance you expect of the
feature,** measured with Polars' `var()` on data from before the stream. Too
large a scale makes no row surprising, so no break is found. `prior_nu` and
`prior_scale` are 2a and 2b in the gamma parametrisation. So the example's
values are Adams and MacKay's own finance example (a = 1, b = 1e-4,
hazard = 250): a variance of 1e-4.

**Read the regime's age from `run_mode`, and use `p_change` only as an
alarm.** `p_change` is a per-row likelihood ratio, so it is spiky, and its
height depends on the size of the break against the prior scale. On three
kinds of break:

| the break | `p_change` | the run length |
|---|---|---|
| a ten-fold variance step | 0.83 on the row itself | finds it one to three rows later, and dates it to the right row |
| a four-sigma mean shift, with a diffuse prior | barely lifts | finds it one to three rows later, and dates it to the right row |
| a change in correlation alone | never moves at all | reaches it only under `emission="gaussian"`, a median of 98 rows later ([docs/REGIMES.md §5](docs/REGIMES.md#5-two-changepoint-detectors-on-the-same-break)) |

**`p_change` reports `P(r ≤ 1)`, because `P(r = 0)` equals `H` on every
row:** the two branches share one predictive, so the normalised mass at
`r = 0` is *exactly* `H`, whatever the data. Row one of a group reports
nothing under the default `min_weight` of 1, since `P(r ≤ 1)` is 1 there
however the row looks.

Give `emission` to choose each run's model of the rows:

| `emission` | each run's model | note |
|---|---|---|
| `"diag"`, as in the example | a normal-inverse-gamma per feature | |
| `"gaussian"` | a normal-inverse-Wishart over all of them, O(runs d²) a row | the one that can see a break in the *correlation* alone |
| `"robust"` | each row's contribution weighted by `(pi(x)/pi(mode))**robust_beta` | one 20-sigma row moves nothing; without it, that row is a changepoint at `p_change` 0.91 |

**Keep `robust_beta` below about 0.2.** Above it, the weighting forgives the
rows of a new regime one at a time, so no change is ever detected again. The
default of 0.1 ignores a single 20-sigma row and still finds a four-sigma
shift within three rows, dated to the right one.

Two settings bound the runs kept, and a third sets the hazard row by row:

| setting | what it does |
|---|---|
| `prune_below` | drops the runs holding less than this share of the mass |
| `max_run` | 10,000 by default: folds every longer run into the last kept one. In a stream that seldom breaks, this is what bounds the runs kept |
| `hazard_col` | reads the hazard per row from a column instead: a null falls back to `hazard`, and a value of 1 or less is an error naming the row |

#### `hmm` — which regime are we in

*API:* [`po.spec.hmm`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.hmm) — *Rust:* [`hmm.rs`](crates/online-core/src/hmm.rs) — *Outputs:* [fields](docs/OUTPUTS.md#hmm)

Reach for `hmm` to ask "which regime are we in", not only "which regime does
this row look like".
[`ew_class`](#ew_class--gaussian-classification-on-ew_cov-moments) classifies
a row against *labelled* Gaussians. An `hmm` does the same arithmetic without
labels: the regime is a *hidden state*, and a transition matrix carries
information from one row to the next.

Here a hidden state means one of the `k` regimes, not the bank's saved state.
The model runs Hamilton's filter one row at a time, from the filtered `p` the
previous row left, and every output is read before the row is learned from:

```
p1_l   = Σ_k p_k·Π_kl                      the predicted state
f_l    = N(x | μ_l, Σ_l + r_l·I)           the state's density, r_l its fading prior ridge
loglik = ln Σ_l p1_l·f_l                   the row's surprise
p_l   ← p1_l·f_l / Σ                       the filtered state
```

This code uses `df` from [Example data](#example-data):

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
states from that many learned rows with `kmeans`' rule (`seed_rule="lloyd"`,
`seed=0`). Give a `warm_rows` large enough for those rows to span more than
one regime; otherwise the seeds split one regime in two.

**When the regimes differ only in covariance, give the states with `means`
and `covs`.** The default seeding is k-means, and zero-mean states differ in
nothing k-means can see, so it splits their rows by direction. On streams
that stay in one of two zero-mean states, the filter then puts about half the
rows after seeding in the true state, 0.51, which is chance. Given `covs`, it
puts 98% there ([docs/REGIMES.md
§1](docs/REGIMES.md#1-does-hmm-recover-the-stream-that-made-it)). Without
the covariances, give the model a feature whose mean shifts with the regime,
made with Polars expressions in the query before the bank.

**A row splits across the states at weight `w·p_l`,** which sums to `w`, so
`weight_sum` follows the same recursion as in every model.

**A single extreme row can capture a state,** and move its mean far from the
data. A state with zero responsibility keeps its moments as they are, so a
state that stops winning never forgets, and the mixture is left one state
short. To guard against it, clip the features with Polars' `clip()` in the
query before the bank, give the states with `means` and `covs` and freeze
them with `learn=False`, or give a larger `precision_prior`. Keep that ridge
well below the data's variance: a ridge near it halves every correlation.

**With a sticky transition matrix, the filter beats a nearest-centre rule
that knows the true centres.** On two-dimensional blobs 1.5 apart, that rule
is 85% right, and the filter 99%. The matrix is learned from the **filtered
joint of consecutive states**, `ξ_kl = p_k(t−1)·Π_kl·f_l / Σ`, and a
Dirichlet pseudo-count keeps the matrix row of a state never visited a
distribution.

**The transition matrix moves one step per row, whatever the clock between
rows, so the gap over a weekend counts as a single step.** The transition
counts decay on the clock, and each row adds its weight, so the chance of
staying rises with the rows' density.

Three settings shape the matrix:

| setting | what it does |
|---|---|
| `transition_prior` | the Dirichlet pseudo-count per cell of the transition matrix, 1.0 by default |
| `transition=` | spreads that pseudo-count over a matrix of your own |
| `exog_tvtp=` | drives the transitions from a column, through fixed `tvtp_coef=`: time-varying transition probabilities (TVTP) |

## Performance

A bank spends most of its time on the models' own arithmetic. Its memory
holds their state, a few chunks in flight, and what Polars has read ahead
of the bank. The thread counts and `chunk_rows` change only speed and
memory, never the numbers a bank returns. Every figure here was measured on
one Apple M4 Pro with 14 cores (10 performance, 4 efficiency). The runs
behind them are in [docs/PERFORMANCE.md](docs/PERFORMANCE.md), with a
table of what to change when a number is not what you expected.

### Throughput

**A model's speed is set by the work it does on each row**, such as a
solve, a matrix to factor or a distance to each centre. Here are rows a
second for every model at its usual settings, all under these conditions:

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

**List several targets on the same features in one spec's `targets`, where
they share one set of feature sums:** 10 targets take 1.86 times as long as
one.

**A grid of half-lives keeps one set of sums per half-life, and runs them
in parallel.** So the grid of five takes 1.43 times as long as its shortest
half-life, 500, which runs alone at 2,443,773 rows a second. That is slower
than at the half-life of 1,000 that every other row here uses, because a
shorter half-life solves more often.

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

**`corrchange`'s window kind is the slowest model here, because its
permutation null redraws `n_perm` statistics every `permute_every` rows.**
To skip the redraws, give `crit` a number, which fixes the threshold.

**`bocpd` slows with the number of run lengths it keeps, so a larger
`prune_below` makes it faster.** It does `O(runs · d²)` work a row for `d`
features. `prune_below` drops every run holding less than that share of the
probability mass, and `max_run`, 10,000 by default, caps how many runs it
keeps. With neither bound the runs grow by one every row, and a stream
takes `O(rows²)`. A changepoint collapses them to a few dozen, so `bocpd`
runs faster on data that breaks. A stream that does not break spreads them
over thousands of run lengths, and there `max_run` is the bound. On 20,000
i.i.d. Gaussian rows with no `max_run`
([PERFORMANCE §15](docs/PERFORMANCE.md#15-the-correlation-families-bocpds-prune_below-keeps-it-finite-and-rcovs-estimator-sets-its-cost-2026-09-06)):

| `prune_below` | rows/sec |
|---|---:|
| `1e-8` | 204,000 |
| `1e-6`, the default | 324,000 |
| `1e-4` | 687,000 |

### Parallelism

**To use more cores, give the bank more tasks: set `group=` on each spec,
pass more specs, or both.** On every chunk the bank splits its work into
tasks, one per spec and group, or one per spec without a `group`. It runs
them on its own thread pool, separate from Polars', largest first, so a
few big groups do not leave cores idle at the end. Within a task the rows
go one at a time, because each row's update depends on the last. So a bank
with one spec, one group and one half-life is one thread's work per chunk,
while Polars reads and writes in parallel around it. All of it runs in one
process, by design ([What this is not](#what-this-is-not)).

| what to add | how it runs in parallel | measured |
|---|---|---|
| groups, with a spec's `group=` | one task per group | k=20 over 64 groups: 7.3× from 1 to 14 threads on a 14-core machine (below) |
| specs, in the list the bank takes | one task per spec | eight single-group specs, k=20 over 300k rows, run in 155 ms in one bank, against 641 ms one at a time |
| half-lives, as a list given to `half_life=` | each half-life in a grid is its own set of running sums, and a task's half-lives run in parallel on the bank's pool. Ridge and feature-set grids share one set of sums and are expanded at solve time, so they need no thread | |
| `shards`, on a wide `marginal` | the one task that runs on several threads: `shards` splits its pairs across the pool within a group ([`marginal`](#marginal--every-pairs-moments-kept-in-the-state)) | |
| a Python thread that reads ahead | Python's global lock is released while a chunk is in the bank, so a Python reader thread can run ahead of `ModelBank.fit_predict` | |

Rows a second for k=20 over 64 groups, at each thread count
([PERFORMANCE §31](docs/PERFORMANCE.md#31-0130-against-0120-2026-09-29)):

| threads | 1 | 2 | 4 | 8 | 14 |
|---|---:|---:|---:|---:|---:|
| rows/s | 0.98M | 1.89M | 3.44M | 5.98M | 7.08M |

**Set the two thread counts with two environment variables, one for each
pool:**

| variable | sizes | unset or `0` | read at | what took effect |
|---|---|---|---|---|
| `POLARS_ONLINE_MAX_THREADS` | the bank's pool | one thread per core | the first chunk any bank fits or scores, through any call, or the first `po.thread_pool_size()` | [`po.thread_pool_size()`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.thread_pool_size) |
| `POLARS_MAX_THREADS` | Polars' readers and writers, and their read-ahead ([Tuning memory with Polars' own settings](#tuning-memory-with-polars-own-settings)) | one thread per core | `import polars`, which `import polars_online` runs | `pl.thread_pool_size()` |

Set both in the shell when you start Python, as
`POLARS_ONLINE_MAX_THREADS=8 POLARS_MAX_THREADS=8 python fit.py`, which
works whatever the script imports. Or set them in Python before the point
the table names, as `os.environ["POLARS_ONLINE_MAX_THREADS"] = "8"`. A variable set after
that point is ignored, so check the sizes in effect with
`po.thread_pool_size()` and `pl.thread_pool_size()`. A
`POLARS_ONLINE_MAX_THREADS` that is not a whole number makes that first
call raise `ValueError`, naming the variable. Giving both pools every core
slows neither, because a bank task never calls back into Polars' pool.

**To search over factor sets, build one spec per set and pass the whole
list to one `lf.online.fit_predict` call.** Each spec runs as its own task
on each group, and the query reads the file once.
This code uses the file `ticks.parquet` from [Example data](#example-data):

```python
import os
os.environ["POLARS_ONLINE_MAX_THREADS"] = "8"   # the bank's pool: read at the first fit_predict
os.environ["POLARS_MAX_THREADS"] = "8"          # Polars' readers and writers: read at import polars

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

On 2.56M rows over 64 groups, the six specs make 6 × 64 tasks every chunk.
The query takes 29.0 s at one thread and 4.3 s at fourteen
([PERFORMANCE §31](docs/PERFORMANCE.md#31-0130-against-0120-2026-09-29)).
Its output has one column per spec, which is what `compare_specs` reads,
and one state file, `grid.state`, holds them all.

### Chunk size

**Set how many rows the bank takes at a time with the keyword `chunk_rows`,
100,000 by default.** `lf.online.fit_predict`, `lf.online.predict`,
`ModelBank.fit_predict_batches`, `ModelBank.fit`, `with_windows` and
`refresh_time` take it. A frame passed to `ModelBank.fit_predict` is one
chunk, whatever its size.
This code uses `df` and `lf` from [Example data](#example-data):

```python
fitted = lf.online.fit_predict([spec], chunk_rows=50_000).collect()               # a query, read 50,000 rows at a time
whole = po.ModelBank([spec]).fit_predict(df)                                       # a frame: one chunk, whatever its size
parts = pl.concat(po.ModelBank([spec]).fit_predict_batches(df, chunk_rows=100))   # the same frame, 100 rows at a time
```

The chunk size changes only which rows carry `coef`: each group's last row
of every chunk.

**For a wide frame, use larger chunks, because each chunk carries a fixed
overhead.** At 10,000 columns, handing a chunk across from Polars alone
takes about 8 ms a call: 4 µs of every row at 2,000 rows a call, and 0.4 µs
at 20,000
([PERFORMANCE §20](docs/PERFORMANCE.md#20-sgds-per-feature-cost-task-75-2026-09-08)).

**For less memory, use smaller chunks**, because a few are in flight at
once: three on the command line, and a few more in Python
([Memory](#memory)).

### Memory

**Memory grows with the models' state and the rows a delay or a window
holds, and never with the number of rows that have passed.** Every way of
running a bank works a chunk at a time. Measured as the most memory the
process ever held, on one file of `ewridge` with 20 features, parquet in
and parquet out
([PERFORMANCE §11](docs/PERFORMANCE.md#11-memory-which-surface-is-odata-2026-09-02)):

| what you write | 3M rows | 12M rows | what it is |
|---|---:|---:|---|
| `lf.online.fit_predict([spec])` | 0.90 GB | 1.35 GB | the bank inside a query |
| `for chunk in lf.collect_batches(): bank.fit_predict(chunk)` | 0.80 GB | 1.24 GB | your own loop |

The bank does not grow from 3M to 12M rows: the growth is the memory
allocator keeping pages it has freed. Of the rest, nearly all is Polars
reading ahead in the parquet file. The command line, whose allocator
returns freed pages at once, holds 0.75 GB at 12M rows against 0.73 GB at
3M ([docs/RUNNER.md](docs/RUNNER.md)).

**Memory has four parts, each with its own bound:**

| part | grows with | what bounds it |
|---|---|---|
| the state | the models and their settings, and the number of groups: a bank's `group=` keeps one set of running sums per group, and grows with the number of groups, not the number of rows | it does not grow with the stream, but a window's snapshots and `marginal`'s bins grow with their settings, and each is capped per group, at 256 MiB by default |
| the chunks in flight | `chunk_rows` | a few chunks at once, three on the command line ([Chunk size](#chunk-size)) |
| the rows a delay or a window holds | the delay or the window, times the rows' rate | never the stream's length |
| whatever Polars' reader has read ahead | Polars' thread count | `POLARS_MAX_THREADS` shrinks it, and Polars' prefetch settings tune it directly ([Tuning memory with Polars' own settings](#tuning-memory-with-polars-own-settings)) |

**For a statistic per group over a long stream, use a spec's `group=`,
which keeps one set of running sums per group.** Polars' `rolling` over
groups, with `.over("group")` or `group_by=`, holds every row of its input.
Peak memory on the same 12M rows:

| a rolling window | peak memory |
|---|---:|
| with `.over("group")` | 6.5 GB |
| with `group_by=` | 1.7 GB |
| without groups | 0.25–0.28 GB |

### Tuning memory with Polars' own settings

Polars' streaming engine reads blocks of a parquet file ahead of whatever
consumes them, sized from its thread count. While a bank is the slowest
step, a local disk needs none of that read-ahead, so it is the part of
memory you can cut. Three of Polars' environment variables set it, in the
shell or with `os.environ` in Python:

```python
import os

os.environ["POLARS_ROW_GROUP_PREFETCH_SIZE"] = "1"   # blocks (row groups) read ahead: read when a query starts to run
os.environ["POLARS_ROW_GROUP_PREFETCH_KBYTES_BUDGET"] = "65536"   # a byte cap on the same read-ahead: read when a query starts to run
os.environ["POLARS_MAX_THREADS"] = "4"               # the read-ahead is sized from this too: read at import polars
```

**`POLARS_ROW_GROUP_PREFETCH_SIZE=1` saves the most.** Polars reads it each
time a query that scans the file starts to run, so set it at any point
before the call that runs the query, such as `sink_parquet` or
`bank.fit_predict_batches(lf)`.
Measured as peak resident memory on 8M rows of 12 columns in 80 blocks,
with the allocator's page retention off so that the figure is live data:

| what runs | default prefetch | `POLARS_ROW_GROUP_PREFETCH_SIZE=1` |
|---|---:|---:|
| `lf.online.fit_predict(...).sink_parquet(...)` | 1.63 GB | 1.12 GB |
| `bank.fit_predict_batches(lf)` | 1.41 GB | 1.07 GB |

**The saving depends on how much data a block holds, so compare a run at
the default with one at `1` on your own files.** On a file whose blocks
hold 262,000 rows, the setting takes a query from 1.86 GB to 0.51 GB
([PERFORMANCE §11](docs/PERFORMANCE.md#11-memory-which-surface-is-odata-2026-09-02)).

**The byte cap, `POLARS_ROW_GROUP_PREFETCH_KBYTES_BUDGET`, rarely binds**,
because it counts *compressed* bytes, which cost almost nothing for a
memory-mapped local file.

**To fit a run in less memory, give Polars fewer threads and the bank every
core.** Polars sizes its read-ahead from `POLARS_MAX_THREADS`, while the
bank's `POLARS_ONLINE_MAX_THREADS` changes only the bank's speed.
This code uses the file `ticks.parquet` from [Example data](#example-data):

```python
import os
os.environ["POLARS_MAX_THREADS"] = "4"           # before import polars: sizes the reader's read-ahead
os.environ["POLARS_ONLINE_MAX_THREADS"] = "14"   # before the first fit_predict: the bank keeps every core

import polars as pl
import polars_online as po

spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1", "x2"], clock="t",
                       half_life=600.0, gap_cap=300.0, group="stock_id")
(pl.scan_parquet("ticks.parquet")                     # written in Example data
   .online.fit_predict([spec], chunk_rows=200_000)
   .sink_parquet("fit.parquet"))
```

On 12M rows over 64 groups, with one spec of four features and two
half-lives
([PERFORMANCE §31](docs/PERFORMANCE.md#31-0130-against-0120-2026-09-29)):

| Polars threads | bank threads | time | peak memory |
|---:|---:|---:|---:|
| 14 | 14 | 1.6 s | 0.86 GB |
| 4 | 14 | 1.9 s | 0.58 GB |

Keeping Polars at four threads holds 33% less memory, for 15% more time.

### Window operators

**A window operator, such as `po.ewm_mean`, does the same work on a row
however long its window, and holds about one window of rows.** An operator
whose input is null on most rows runs slower, because each read scans the
rows between its values. On 300k rows, one value in 5,000 takes 0.24 s,
against 0.04 s with a value on every row
([PERFORMANCE §35](docs/PERFORMANCE.md#35-two-costs-the-review-of-2026-10-03-named-and-did-not-change)).

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
alone, and a 4-minute window takes what a 1-minute one does.

**To have operators share one queue, give them the same direction, back or
ahead, and the same `half_life`, `window_size` and `closed`.** Sixteen
operators on one queue run in 57% of the time of sixteen with queues of
their own.

**Write a look-ahead target as a column for speed, or give its expression
in `targets` for less memory.** For the column form, call
`with_windows(..., like=spec)` in the query before the bank, and name the
column it writes in the spec's `targets`, under the same `embargo`.
[Windows as a model's inputs and
target](#windows-as-a-models-inputs-and-target) runs both forms. Polars
then runs the window and the bank at once, as two stages of one query. In
the native form the bank resolves the windows itself, in the same source as
the model and one after the other. With many groups it adds a Polars query
per group per chunk on the calling thread, before the groups' rows run on
the pool (§35).

### Against scikit-learn

Most people compare this library with scikit-learn's
[`SGDRegressor.partial_fit`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html),
a first-order stochastic optimiser whose answer depends on the learning
rate, the schedule, the feature scaling and the row order. The primary
regressions here, `ewridge`, `rls`, `lasso`, `huber` and `quantile`,
accumulate sufficient statistics and solve, so they have no learning rate.
With decay off (`half_life=float("inf")`) and `ridge=0`, `ewridge` is
ordinary least squares to 2e-13 of `numpy.linalg.lstsq`, in any row order.
To compare like with like, run `SGDRegressor` against
[`sgd`](#sgd--stochastic-gradient-descent), the cheap `O(k)` baseline: at
the same constant step the two give the same R² to four places, because
they run the same recursion.

Every contender reaches the noise ceiling on a stationary stream.
Scoring each row before learning it, the bank is faster, while sklearn has
the ecosystem: pipelines, `GridSearchCV`, calibration, and far more use.

**Beyond speed and accuracy, the two differ in what one fit can hold:**

| | sklearn | here |
|---|---|---|
| a grid of six penalties | 3.8× the work: six estimators | 1.6× the work: one accumulator, six solves |
| several targets | one estimator each | one shared `X'X` |
| one fit per key | a dict of estimators | `group=`, one state per key |
| standardisation | a `StandardScaler` fitted on the whole frame can leak | streaming, so it cannot leak |
| chunking | | chunk invariance is a test |
| saved model | a pickle | a versioned file that loads on every OS |

The comparison ran under these conditions:

| | |
|---|---|
| stream | one generated stream of 100,000 rows with `k = 20`, measured on 2026-09-08 |
| contenders | each at its best over a sweep of its own settings, by `scripts/sklearn_comparison.py` on scikit-learn 1.9.0; [PERFORMANCE §19](docs/PERFORMANCE.md#19-against-sklearnlinear_modelsgdregressor-task-72-2026-09-08) has the full tables and the sweeps |
| `po.spec.ewridge` | solves on every row here, unlike in [Throughput](#throughput); its rates were re-measured on 2026-09-29, once its solve had been made cheaper ([§30](docs/PERFORMANCE.md#30-where-every-row-solves-2026-09-29)) |
| noise ceiling | the R² of the generating signal itself |

| contender | R² stationary | R² drifting | rows/sec | what a prediction saw |
|---|---:|---:|---:|---|
| noise ceiling | 0.9831 | 0.9923 | | |
| `SGDRegressor`, row by row | 0.9829 | 0.9899 | 3,400 | every row before it |
| `SGDRegressor`, batches of 1,000 | 0.9830 | 0.9820 | 2,200,000 | every row before its *batch* |
| `po.spec.sgd` | 0.9826 | 0.9906 | 6,000,000 | every row before it |
| `po.spec.ewridge` | **0.9831** | **0.9907** | 480,000 | every row before it |

**sklearn's fast form updates once per 1,000-row batch, so a prediction
can come up to 999 rows before its update.** That staleness is the one gap
in accuracy: on the drifting stream the row-by-row contenders are within
0.001 of each other, and the batched `SGDRegressor` scores 0.9820. Called
on every row, which gives each prediction the state as it stands,
`partial_fit` runs at 3,400 rows a second: Python's overhead on each row.

**A half-life gives no edge on this stream, because its rows are evenly
spaced.** A constant learning rate `eta` remembers about `1/eta` rows,
which on evenly spaced rows acts as a half-life. A half-life on a clock
matters when the rows are unevenly spaced.

**Where the exact solve wins: the first hundred rows of every group.** An
exact solve is right as soon as its Gram, `X'X`
([The running sums behind a fit](#the-running-sums-behind-a-fit)), is full
rank, about `k` rows in, where a first-order method needs about `1/eta`
rows per direction. The stream has 500 groups of 200 rows, each with its
own coefficients and `k = 20`. Each sklearn group gets its own estimator
and scaler, in a dict, row by row, and the bank runs `group="g"`. R² by
position in the group:

| contender | R², rows 25–50 | R², rows 50–100 | R², rows 100–200 | rows/sec |
|---|---:|---:|---:|---:|
| noise ceiling | 0.9896 | 0.9903 | 0.9900 | |
| `SGDRegressor` per group, at its best | 0.7277 | 0.9242 | 0.9789 | 3,342 |
| `po.spec.sgd`, `standardize=True`, the same step | 0.7182 | 0.9213 | 0.9788 | 18,519,660 |
| `po.spec.ewridge` | **0.9693** | **0.9860** | **0.9882** | 3,765,320 |

The two `sgd` rows take sklearn's step, and differ by 0.01 of R² only in
how the *prediction* is standardised. In sklearn's loop the row it predicts
is scaled by the moments before it, and in `sgd` by the moments including
it.

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
every row, at 85 GB/s, near this machine's memory bandwidth. With
`gram_block_rows=1024`, `ewridge` touches the matrix once per 1,024 rows
instead, for 7.2× the throughput and no less memory.

**At this width, use `sgd`, at a learning rate that falls as `1/k`.** It is
`O(k)`, faster than sklearn's batch, and makes every prediction from the
state as it stands. A least-mean-squares step is stable only while
`eta · |z|² < 2`, and a standardised row has `|z|² ≈ k`, so both libraries
ran at `learning_rate = 0.2 / k`. There `sgd` and `SGDRegressor` row by row
agree, with a correlation of 0.999997 under `standardize=True`, which is
sklearn's own recipe of a scaler in front of the step.

## Scope and integrations

polars-online fits models to rows that are already read and aligned, by
Polars in the query before the bank or by another tool. A bank can still
take its rows from a database through Polars, or run as an operator inside
a stream processor, and [chunk invariance](#row-order-and-the-two-guarantees)
means another engine's batching cannot change the numbers.

### What this is not

**When a spec names a `clock`, sort the rows by it before the bank, because
polars-online has no policy for rows that arrive late.** A row that
arrives out of clock order is refused, unless its step back is larger than
`restart_after_step_back`, which starts the model over. A label that
arrives late can be held back with [`embargo`](#labels-that-arrive-late).
`clock`, `gap_cap`, `restart_after_step_back` and `session` describe time
*within* a stream, and none of them waits for a late row.

polars-online does two things to a stream itself
([Preparing a stream](#preparing-a-stream)). `po.stream.refresh_time` puts
series that tick at their own times on one grid, and
`po.stream.with_windows` computes exponentially weighted means, sums and
rates over a window of the clock. It deliberately does **not** provide the
rest:

| not provided | use instead |
|---|---|
| connectors or ingestion | a Polars reader in the query before the bank, such as `pl.scan_parquet`, or `pl.scan_arrow_c_stream` for a database ([Databases](#databases-duckdb-and-adbc)) |
| windows beyond those, such as a tumbling or sliding aggregation of any function, and asof or interval joins | Polars' `group_by_dynamic`, `rolling`, `join_asof` or `join_where` in the query before the bank, or a streaming framework such as [Pathway](https://pathway.com) |
| watermarks or a late-arrival policy | rows put in clock order before the bank, by Polars' `sort` or by a stream processor |
| distributed execution | one process, with a thread pool across (spec × group) ([Parallelism](#parallelism)) |

### Databases: DuckDB and ADBC

To feed a bank from a database, sort the query by the clock with
`ORDER BY` in its SQL. Read its result with Polars'
`pl.scan_arrow_c_stream` (py-polars 1.43.0 or later), and chain
`.online.fit_predict(specs)` on that query.
This code uses the file `ticks.parquet` from [Example data](#example-data):

```python
import duckdb

con = duckdb.connect()                                    # an in-memory database
query = "SELECT * FROM 'ticks.parquet' ORDER BY t"        # clock order, sorted by the database; the file is Example data's
fitted = (
    pl.scan_arrow_c_stream(con.cursor().sql(query))       # a DuckDB relation, on a cursor of its own
    .online.fit_predict([spec])
    .collect()
)
# With ADBC, on a cursor of its own: cur.execute(query), then pl.scan_arrow_c_stream(cur.fetch_arrow())
```

The rows cross through the Arrow PyCapsule interface, so no pyarrow is
needed.
[examples/duckdb_cursors.py](examples/duckdb_cursors.py) and
[examples/adbc_cursors.py](examples/adbc_cursors.py) run these steps
against real databases. Install DuckDB or an ADBC driver yourself: the
package depends on neither.

**Give every lazy query its own cursor, from `con.cursor()`, because a
cursor holds one open result.** Both examples show the rule and the trap
of breaking it, and exit non-zero if either stops holding:

| database | what springs the trap | what comes back | what says so |
|---|---|---|---|
| DuckDB | a second query built on the same connection | the first query yields no rows | `ConsumedSourceWarning` |
| ADBC | the cursor executed again before its first query is read | the newer query one batch short, in clock order, and the older one with that batch added out of order: on SQLite, 198,976 and 201,024 of 200,000 rows | nothing, for the newer query; for the older one, a spec with a `clock` refuses the step back |

### Pathway

To run a bank inside a Pathway pipeline, wrap a `ModelBank` in a stateful
operator. The operator calls `fit_predict` on each batch Pathway hands it,
and Pathway checkpoints it with `save_bytes` and `load_bytes`.
This code uses `df` from [Example data](#example-data):

```python
class BankOperator:                                  # the operator's state is the bank
    def __init__(self, specs):
        self.specs, self.bank = specs, po.ModelBank(specs)

    def __call__(self, batch: pl.DataFrame) -> pl.DataFrame:
        return self.bank.fit_predict(batch)          # each ordered batch Pathway hands it

    def snapshot(self) -> bytes:
        return self.bank.save_bytes()                # what Pathway's persistence stores

    def restore(self, blob: bytes) -> None:
        self.bank = po.ModelBank.load_bytes(blob, specs=self.specs)


op = BankOperator([spec])
first = op(df.head(200))                             # plain batches stand in for Pathway's
op.restore(op.snapshot())                            # a checkpoint and a restart between two batches ...
rest = op(df.tail(200))                              # ... and the bank goes on where it was
```

Pathway does the ingestion, the event-time alignment and the windowing
before the operator.
[examples/pathway_integration.py](examples/pathway_integration.py) runs
this operator over plain batches, so it runs without Pathway, which is not
a dependency. Its sketch of the pipeline around the operator needs
Pathway, installed under its own licence, and an input connector.

## Versions, testing and development

Here is what to pin when you install, what the test suite holds each
guarantee to, and how to build and test from a checkout.

### Versioning and the Polars pin

Two versions matter when you install this package. Its own version says
when to expect a breaking change. The range of Polars it accepts,
`polars>=1.34.0,<3`, is measured by this project, and Polars does not
promise it.

#### This package's own versioning

**For stability, pin the minor version in your requirements, as
`polars-online~=0.13.0`, which takes the 0.13 series' patch releases and
no 0.14.** Without that pin, an upgrade can bring a new minor version.
While this package is pre-1.0, the **minor** version carries breaking
changes and any change to the numbers a model returns. It follows semantic
versioning ([CHANGELOG.md](CHANGELOG.md)), and output field names are part
of its API ([Output field names](#output-field-names)). A change to the
Polars range is released as:

| a change to the Polars range | release |
|---|---|
| widening it | minor |
| narrowing it | breaking |
| capping it below a Polars that broke this library, as [How the pin moves](#how-the-pin-moves) describes | patch |

#### What is pinned

The wheel carries its own Rust `polars`, pinned exactly, and shares one
process with the py-polars you install. Data crosses between the two only
through the interfaces listed under
[Which interfaces carry a promise](#which-interfaces-carry-a-promise), so
py-polars can be any version in the range. The lowest py-polars depends on
what you use:

| py-polars | rust polars | pyo3-polars | pyo3 | Python |
|---|---|---|---|---|
| **>= 1.34.0, < 3** (built and tested against 1.44.2) | 0.55.2 | 0.28 | 0.29 | ≥ 3.12 (`abi3-py312`) |

| what you use | the lowest py-polars | why |
|---|---|---|
| `ModelBank` alone | 1.28.1 | |
| `lf.online.fit_predict`, `ModelBank.fit_predict_batches`, `ModelBank.fit`, `with_windows` and `refresh_time` | 1.34.0 | they read with `LazyFrame.collect_batches`, which py-polars added in 1.34.0 |
| the examples that stream a DuckDB, ADBC or pyarrow source into a bank | 1.43.0 | for `pl.scan_arrow_c_stream` |

The suite passes on 1.44.2, the pin, at every change. It has passed on
1.34.0, 1.38.1, 1.44.1 and 2.0.0-rc.1 with identical numbers, but its last
whole run on 1.34.0 came before the window operators, whose formula reader
was checked there on its own. `tests/test_scaffold.py` asserts the pin and
the range, and [docs/RELEASE-READINESS.md](docs/RELEASE-READINESS.md) has
each run with its date and what differed.

#### How the pin moves

**When a new Polars breaks this library, a patch release caps the range at
the last Polars that passed, so that no resolver hands anyone the broken
pair.** A fix then widens the range again. Two checks look for such a
break, in Polars and in NumPy, the one optional dependency
(`polars-online[numpy]`, `numpy>=1.24`): a weekly canary, and the legs
every release runs before it publishes.

**Every Monday the canary
([`polars-canary.yml`](.github/workflows/polars-canary.yml)) drops the
range from `pyproject.toml` and installs the newest py-polars, prereleases
included.** It builds the wheel as CI does, then runs the suite, all but
the opt-in soak tests and the checks of this repository's own pins. Only
Polars moves in that run, so a red canary means Polars broke this library.
A second job runs the suite on NumPy's next release candidate each week.

**Every release runs the same check before it publishes**, in legs
(`release.yml`):

| leg | resolves to | blocks the publish |
|---|---|---|
| the newest in-range | the newest stable inside `<3` | **yes** |
| the next major | unpinned, prereleases allowed | no |
| the newest NumPy | the newest NumPy | yes |
| NumPy's next release candidate | its next release candidate | no |

**The newest in-range leg blocks the publish, because `<3` admits every
1.x and 2.x.** A resolver can hand someone a Polars newer than the one the
wheel was built against, the day after that Polars ships. A pass on the
pinned version alone would not show that the range holds. The next-major
leg is early warning: [docs/RELEASE-READINESS.md](docs/RELEASE-READINESS.md)
has the steps for moving the ceiling up to a new major. Only NumPy moves in
the NumPy runs, the release's and the canary's, so a red one names NumPy.

#### Which interfaces carry a promise

This library crosses into Polars four ways, and only the Arrow PyCapsule
interface carries a promise. It narrows the exposure without removing it,
because only the output side uses it, and the frame still goes in as a
Polars frame. The three without a promise include the two that stream,
`ModelBank` and the IO plugin. So treat a break on a new Polars as expected
maintenance, and check those two paths before the others:

| interface | used by | its promise |
|---|---|---|
| pyo3-polars' extension types | `ModelBank` | none beyond the latest definitions working with the latest Polars: provided "for convenience" |
| the IO plugin | `lf.online.fit_predict`, `lf.online.predict`, and `with_windows` and `refresh_time` on a query | none: documented, but `@unstable` in py-polars |
| an expression's serialized form, `expr.meta.serialize(format="json")` | the window formulas, read into this library's own tree | none: Polars calls it unstable across versions; its shapes were measured the same on 1.34.0 and 1.44.2 |
| the Arrow PyCapsule interface | [`fit_predict_arrow`](#output-as-arrow) | an Arrow specification, which py-polars and pyarrow consume, so a break there would be Arrow's rather than Polars' |

**A mismatch raises an `AttributeError` before any data moves, and does
not crash the process.** `ModelBank` moves data across the boundary
through the Arrow C Data Interface, with two private methods of py-polars,
`_export` and `_import`, so a Polars without them fails at once.

### Testing

**Each guarantee in [What you can rely on](#what-you-can-rely-on) is held to
an oracle or an invariant.** An oracle is a reference the model cannot
share a bug with, and an invariant is a property every run must keep. The
suite has 1,215 Rust tests and 3,840 pytest cases, counted on 2026-10-03,
and [docs/TESTING.md](docs/TESTING.md) is the ledger of what each part
proves.

**Against references.** Each model is held to a reference written from its
documented recursion, and to another library wherever one computes the
same quantity:

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

**Invariants, for every model**, checked at the bank and, where they apply,
at the command line:

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
pure-noise targets checks the same thing from the other side.

**Fixed numbers.** Each model has one golden stream in the Rust core. The
whole pipeline, from reading the columns through the models and the
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

**Where it runs:**

| when | what runs |
|---|---|
| before every commit | `./scripts/gate.sh`: `cargo fmt`, `clippy -D warnings`, `cargo test`, `uv lock --check`, `ruff`, `mypy`, the build, `pytest` and `sphinx -W` |
| every push and pull request | the tests on Ubuntu, Windows and macOS, on Python 3.12 and 3.14, and on 3.13 on Linux; the format, lint, type and documentation checks, on Linux; mutation testing of the lines the change touched, which fails on a mutant no test catches; and every output compared with the newest release's, bit for bit, as a report |
| every release | a state file written on macOS and continued on Windows and Linux; the suite on the newest Polars the range admits, and on the newest NumPy |
| weekly | the suite on the newest py-polars, and on NumPy's next release candidate; a leak check across the boundary with Polars; mutation testing of all of `online-core`, as a report |

### Development

To build and test from a checkout, install [uv](https://docs.astral.sh/uv/)
and a stable Rust toolchain with [rustup](https://rustup.rs). In each new
shell, run `source scripts/env.sh`, or `. .\scripts\env.ps1` in PowerShell,
to put both on the `PATH`. VS Code's terminal gets them from
`.vscode/settings.json`. [CONTRIBUTING.md](CONTRIBUTING.md) says how to
make changes.

Run `cargo` through `uv run`, as the commands below do, because
`online-py` builds against pyo3's `abi3-py312` and needs a 3.12+
interpreter at build time. The tests download or generate their own data
and cache downloads under `.cache/`. Offline, the tests that need a
download are skipped.

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
