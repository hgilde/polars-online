# polars-online

> **GitHub project:** [github.com/hgilde/polars-online](https://github.com/hgilde/polars-online)

> **Version 1.0 changes the API relative to 0.x.** The API is more standard
> and consistent now, and should remain stable from here on.

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
| [Preparing a stream](#preparing-a-stream) | [windowed means](#making-features-and-targets-from-windows) · [windows as columns](docs/WINDOWS.md#windows-as-columns) · [windows as a model's inputs and target](docs/WINDOWS.md#windows-as-a-models-inputs-and-target) · [features in units of their spread](docs/WINDOWS.md#features-in-units-of-their-spread) · [saving and resuming a window run](docs/WINDOWS.md#saving-and-resuming-a-window-run) · [series that tick at their own times](#series-that-tick-at-their-own-times) |
| [Running a bank](#running-a-bank) | [as a query](#as-a-query-lfonlinefit_predict) · [in a loop](#in-a-loop-modelbank) · [as a job, without Python](#running-a-bank-as-a-job-without-python) · [output as Arrow](#output-as-arrow) |
| [Saving, loading and serving](#saving-loading-and-serving) | [save and load](#save-and-load) · [serving without learning](#serving-without-learning) · [a state without this library](#reading-a-state-without-this-library) |
| [Reading the fit](#reading-the-fit) | [what a bank holds](#what-a-bank-holds) · [output field names](#output-field-names) · [coefficients](#coefficients) · [the running sums](#the-running-sums-behind-a-fit) · [one row per finished group](#one-row-per-finished-group) · [correlation matrices](#reading-a-correlation-matrix) |
| [Checking the data and the model](#checking-the-data-and-the-model) | [your data's quality](#checking-your-datas-quality) · [that a model works](#checking-that-a-model-works), one question at a time: [docs/DIAGNOSTICS.md](docs/DIAGNOSTICS.md) |
| [Models](#models) | [linear models](docs/MODELS.md#linear-models) · [moments and correlation](docs/MODELS.md#moments-and-correlation) · [clustering and classification](docs/MODELS.md#clustering-and-classification) · [sequential tests and regimes](docs/MODELS.md#sequential-tests-and-regimes) |
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
prices.write_parquet("prices.parquet")          # on disk: every fit below reads it back as a query

# 2. One fit over every row: fit the rows before 15:30, save the state, read it back, serve the rest.
whole = po.spec.ewridge(
    "whole",                                    # the spec's name: it names the output column
    targets=["ret"], features=["signal_a", "signal_b"],
    half_life=float("inf"),                     # no forgetting: every row counts the same
    group="stock_id",                           # one regression per stock
)
flat = (
    pl.scan_parquet("prices.parquet").head(21_600)   # a query over the file's rows before 15:30
    .online.fit_predict([whole], save_state="bank.state")   # adds a column "whole", saves the state
    .online.unnest([whole])                     # as columns: pred_<target>, resid_<target>, ...
    .collect()                                  # runs the query
)
saved = (
    po.ModelBank.load("bank.state").coef()      # the fit in the file, a row per coefficient
    .pivot("term", index="group", values="coef")   # a row per stock, a column per term
)
served = pl.scan_parquet("prices.parquet").tail(1_800).online.predict("bank.state").collect()   # scores the last half hour, learns nothing

# 3. A fit that follows the recent rows, and its betas on every row.
local = po.spec.ewridge(
    "local", targets=["ret"], features=["signal_a", "signal_b"], group="stock_id",
    clock="ts", half_life="10m",                # a row's weight halves every ten minutes of ts
    gap_cap="5m",                               # and a gap longer than five minutes decays as five
    coef_every=0,                               # the coefficients on every row
)
betas = pl.scan_parquet("prices.parquet").online.fit_predict([local]).online.unnest([local]).collect()   # a row of betas per input row

# 4. A target that looks ahead: each row's mean return over the next ten minutes.
fwd_ret = po.rewm_mean("ret", half_life=float("inf"), window_size="10m").alias("fwd_ret")
ahead = po.spec.ewridge(
    "ahead", targets=[fwd_ret], features=["signal_a", "signal_b"], group="stock_id",
    clock="ts", half_life="10m", gap_cap="5m",
    embargo="10m",                              # learn each row once its ten minutes have passed
)
forecast = pl.scan_parquet("prices.parquet").online.fit_predict([ahead]).online.unnest([ahead]).collect()   # each row's forecast of fwd_ret
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
But `coef_every=0` produces a time series of the betas at every row, as they
stood after learning that row. This can be used to track the betas over
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
only then learns from the row ([Checking that a model
works](#checking-that-a-model-works)).

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
time-weighted mean of the mid, added as a column in the query before the
bank. A target can look ahead, such as the next minute's VWAP less the mid,
named in the spec's `targets` and learned once its minute has passed.
Windows always need the rows in time order ([Making features and targets
from windows](#making-features-and-targets-from-windows)).

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
| **chunk** | one of the pieces the bank takes the stream in, one at a time: `chunk_size` rows in a query, 100,000 by default, or each frame passed to `ModelBank.fit_predict` |
| **state** | everything a bank has learned |
| **clock**, **decay** | the column, named by `clock=`, that says how far apart two rows are, and the forgetting measured along it: each row's weight halves every `half_life` of the clock ([Time and decay](#time-and-decay)) |
| **`weight_sum`** | the *weight behind the state* that produced a row's prediction, which `min_weight` reads ([Warm-up](#warm-up)) |

### What you can rely on

These hold for every spec, however a bank is run:

| | |
|---|---|
| **honest predictions** | every row is predicted before its own outcome is learned ([Row order and the two guarantees](#row-order-and-the-two-guarantees)), and a target that looks ahead is learned only once its window has closed ([Windows as a model's inputs and target](docs/WINDOWS.md#windows-as-a-models-inputs-and-target)) |
| **bounded memory** | memory grows with the models' state and the rows a delay or a window holds, and never with the number of rows that have passed ([Memory](#memory)) |
| **any chunking** | one chunk or a thousand, with or without a save and resume in the middle, gives the same numbers, to the last bit. Only which rows carry the coefficients, `coef` and `support_coef`, can change ([Row order and the two guarantees](#row-order-and-the-two-guarantees)) |
| **any thread count** | the thread count, set with `POLARS_ONLINE_MAX_THREADS`, changes only the speed. Each spec and group is one task on the bank's thread pool: with 64 groups, 14 threads process 7.3× the rows per second of one ([Parallelism](#parallelism)) |
| **unusable values** | NaN, ±inf and any magnitude above `1e100` count as null, so a sentinel value never reaches a model. A null skips an update but never stops the clock ([Nulls, and three ways to hold a row back](#nulls-and-three-ways-to-hold-a-row-back)) |
| **zero weights** | a row of weight 0 is scored and moves the clock, but teaches the model nothing ([Weights](#weights)) |
| **one `weight_sum`** | `weight_sum` means the same in every model, so one `min_weight` means the same for every spec in a bank ([Warm-up](#warm-up)) |
| **a clock policy** | a spec with a clock caps each step at `gap_cap`, and applies `session_gap` at a session boundary. A step back the model would learn from is refused, naming the row, unless `restart_after_step_back` makes it a restart ([Sessions, gaps and steps back](#sessions-gaps-and-steps-back)) |
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
| size | 8 to 10 MB to download and 26 to 37 MB installed, by platform, for 0.13.0 |

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
build all three with this code. It saves each to a parquet file, and an
example that runs a query reads the file back with `pl.scan_parquet`, as a
stream too large to hold would be read:

```python
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import polars_online as po

rng = np.random.default_rng(0)
x0, x1, x2, signal_a, signal_b, noise_y, noise_ret = rng.standard_normal((7, 400))
row = np.arange(400)
df = pl.DataFrame({
    "t": np.arange(400.0),                                       # a numeric clock: the row's number
    "ts": pl.datetime_range(datetime(2024, 1, 2, 9, 30), datetime(2024, 1, 2, 16, 9), "1m", eager=True),   # a minute apart
    "x0": x0, "x1": x1, "x2": x2, "signal_a": signal_a, "signal_b": signal_b,
    "y": 1.0 + 2.0 * x0 - 1.0 * x1 + 0.5 * noise_y,              # the truth: intercept 1, x0 2, x1 -1; x2 plays no part
    "ret": np.where(row < 200, 0.8, -0.8) * signal_a             # a break: the beta on signal_a flips at row 200
    + np.where(row // 100 % 2 == 1, 2.0, 0.5) * noise_ret,       # two regimes: calm (0.5), volatile (2.0) on rows 100-199 and 300-399
    "stock_id": [f"b{i % 4}" for i in range(400)],               # four stocks, taking turns row by row
    "group": [f"g{i % 3}" for i in range(400)],
    "session": ["m"] * 200 + ["a"] * 200,                        # a morning session, then an afternoon one
    "venue": ["X", "Y"] * 200,
})
df.write_parquet("ticks.parquet")                                # df on disk, as one file
lf = pl.scan_parquet("ticks.parquet")                            # the same rows, read back from the file as a query
today = df.with_columns(pl.col("t") + 400.0)                     # the 400 rows that follow df's, on the clock t
today.write_parquet("today.parquet")
later = pl.scan_parquet("today.parquet")                         # today, read back as a query
Path("ticks").mkdir(exist_ok=True)                               # and df as two files, for the examples that scan ticks/*.parquet
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
trades.write_parquet("trades.parquet")                           # on disk, for the examples that scan it
```

`df` holds a known answer, so an example's output can be read against it.
`y` is `1 + 2·x0 − x1` plus noise of spread 0.5, and `x2`, `signal_a` and
`signal_b` play no part in it. `ret`'s beta on `signal_a` is 0.8 for the
morning session's 200 rows and −0.8 for the afternoon's: a break. Its noise
is calm, of spread 0.5, on rows 0 to 99 and 200 to 299, and volatile, of
spread 2, on the other two hundred: two regimes. `trades` holds no planted
relation: its mid is a random walk.

## How a bank sees a stream

The parameters every spec shares say how its model reads the stream, one
chunk at a time. Most models take all of them, and the
[`polars_online.spec`](https://hgilde.github.io/polars-online/spec.html)
reference gives their units and defaults. With a `group`, every parameter
applies within each group, the clock included.

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

### What a spec names

A spec describes one model: which model it is, the columns it reads, and
how it treats time. The name of the spec (the first parameter) is used to
name the output column: `po.spec.ewridge("ridge", ...)` adds a column named
`ridge`. Each row of that column is a record of the model's fields for the
row, and the input's columns pass through untouched.
This code uses `df` from [Example data](#example-data):

```python
ridge_spec = po.spec.ewridge(
    "ridge",
    targets=["y"],                 # at least one; the targets of one spec share the features' running sums
    features=["x0", "x1", "x2"],   # numeric columns of any width, Decimal and Boolean included; read as 64-bit floats
    fit_intercept=True,            # the default: the fit has a level of its own
    clock="t", half_life=600.0, gap_cap=300.0,   # how it treats time: Time and decay, below
)
fitted = po.ModelBank([ridge_spec]).fit_predict(df)
print(len(df.columns), "columns in, then", fitted.columns[len(df.columns):])
print("each row of ridge holds", fitted["ridge"].struct.fields)
```

```text
13 columns in, then ['ridge']
each row of ridge holds ['pred_y', 'resid_y', 'weight_sum', 'settled_frac', 'withheld_reason', 'coef', 'support_coef']
```

**The bank refuses a spec named like an input column, since its output
column would replace that column.** It refuses a text column in `targets`
or `features` too. Convert such a column with Polars' `with_columns`, in
the query before the bank: `cast(pl.Float64)` for numbers stored as text,
as the error suggests, and a comparison such as `pl.col("venue") == "X"`
for a category. This code prints both refusals, then fits the venue as a
comparison. Its last line finds the truth planted in `y`: an intercept of
1, 2 on `x0`, −1 on `x1`, and nothing on the venue.
It reads `df` and `lf` from [Example data](#example-data):

```python
for refused in [
    po.spec.ewridge("x0", targets=["y"], features=["x1"], half_life=600.0),          # named like a column
    po.spec.ewridge("by_venue", targets=["y"], features=["venue"], half_life=600.0),  # a text feature
]:
    try:
        po.ModelBank([refused]).fit_predict(df)
    except ValueError as err:
        print(err)

by_venue = po.spec.ewridge("by_venue", targets=["y"], features=["x0", "x1", "is_x"], half_life=float("inf"))
coded = lf.with_columns(is_x=pl.col("venue") == "X").online.fit_predict([by_venue]).collect()
print("intercept, x0, x1, venue X:", [round(c, 2) for c in coded["by_venue"].struct.field("coef")[-1]])
```

```text
duplicate: spec "x0" has the same name as an input column; the output struct would replace it. Rename the spec.
spec "by_venue": feature column "venue" has dtype str; it must be numeric (cast it, e.g. pl.col("venue").cast(pl.Float64))
intercept, x0, x1, venue X: [1.02, 1.99, -1.01, -0.05]
```

#### Relative and look-ahead targets

A target is often one value less another, such as the next five minutes'
mean trade price less the mid at the row. Build it in one of two ways, by
when its value is known. Only a model that regresses its targets takes a
target expression. Four kinds of model refuse one by name: a model that
reads only its features, `ew_class`, `seqtest`, and an `ftrl` or `sgd`
fitting a probability or a count.

**To use a target computed from the rows after its row, write it as a
Polars expression and put it in the spec's `targets`.** Its `.alias` names
the output fields, as `pred_fwd_move`. The bank scores each row where it
sits and learns it once the expression's window has closed, so give the
spec an `embargo` of at least the window's `window_size`. [Windows as a
model's inputs and target](docs/WINDOWS.md#windows-as-a-models-inputs-and-target)
gives the rule in full, with what to do for a shorter one. `pred`,
`resid`, `sigma` and the metrics are on the target's scale, so
`pred + mid` predicts the mean trade price itself.

The expression is the column `po.stream.with_windows` makes from it, fed
back to the bank, so both forms give the same predictions. Only the
residual tells them apart: a column holds the outcome at the row, so the
bank can score it there, while the expression's outcome is not known until
its window closes. This code runs on `trades.parquet`, the quotes and
trades saved in [Example data](#example-data):

```python
flows = pl.scan_parquet("trades.parquet").with_columns(                   # a query, and a feature in it: a trade's signed size
    flow=pl.when(pl.col("side") == "buy").then(pl.col("quantity")).otherwise(-pl.col("quantity"))
)
fwd_move = (
    po.rewm_mean("price", half_life=float("inf"), window_size="5m")       # the mean trade price over the next five minutes
    - pl.col("mid")                                                        # less the mid at the row
).alias("fwd_move")                                                        # the alias names the target
ahead_5m = po.spec.ewridge("ahead_5m", targets=[fwd_move], features=["flow"],
                           clock="ts", gap_cap="5m", half_life="30m", embargo="5m", group="symbol")
moves = flows.online.fit_predict([ahead_5m]).online.unnest([ahead_5m]).collect()   # does buying predict the price rising?

# The same target as a column, made in the query by with_windows on the spec's clock, then named in targets.
as_column = {**ahead_5m, "targets": ["fwd_move"]}
made = po.stream.with_windows(flows, fwd_move, like=ahead_5m).online.fit_predict([as_column])
column = made.online.unnest([as_column]).collect()

same = (moves["pred_fwd_move"] == column["pred_fwd_move"]).sum()
print(f"rows predicted: {moves['pred_fwd_move'].count()} by the expression, {column['pred_fwd_move'].count()} by the column, {same} equal")
# The rows of the last five minutes have no outcome yet in either form.
print(f"residuals: {moves['resid_fwd_move'].count()} from the expression, {column['resid_fwd_move'].count()} from the column")
```

```text
rows predicted: 733 by the expression, 733 by the column, 733 equal
residuals: 0 from the expression, 556 from the column
```

**Write a ratio target to sit about zero, as the quotient less 1 or as its
`.log()`.** `hit_rate` asks whether prediction and outcome fall on the same
side of zero ([Is another model
better?](docs/DIAGNOSTICS.md#is-another-model-better)). A plain ratio sits
about 1, where two positive numbers always agree, so its `hit_rate` reads
1.0 whatever the fit. A difference sits about zero too. A target that looks
ahead takes the same forms: `(next_5m / pl.col("mid")).log()`, with
`next_5m` the `po.rewm_mean` above, and `mid * exp(pred)` turns its
prediction back into a price. This code regresses four forms of a trade's
price against the mid at its row on the same signed size, `flow`.
It reads `trades.parquet` from [Example data](#example-data):

```python
flows = pl.scan_parquet("trades.parquet").with_columns(
    flow=pl.when(pl.col("side") == "buy").then(pl.col("quantity")).otherwise(-pl.col("quantity"))
)
returns = flows.with_columns(
    ratio=pl.col("price") / pl.col("mid"),                              # price / mid, about 1
    fwd_return=pl.col("price") / pl.col("mid") - 1,                     # the quotient less 1, about zero
    log_ret=(pl.col("price") / pl.col("mid")).log(),                    # ln(price / mid), about zero
    diff=pl.col("price") - pl.col("mid"),                               # price - mid, about zero
)
vs_mid = po.spec.ewridge("vs_mid", targets=["ratio", "fwd_return", "log_ret", "diff"], features=["flow"],
                         clock="ts", gap_cap="5m", half_life="30m", group="symbol", emit_metrics=True)
against_mid = returns.online.fit_predict([vs_mid]).online.unnest([vs_mid]).collect()
for target in ["ratio", "fwd_return", "log_ret", "diff"]:
    print(f"hit_rate of {target}: {against_mid[f'hit_rate_{target}'].drop_nulls()[-1]:.2f}")
```

```text
hit_rate of ratio: 1.00
hit_rate of fwd_return: 0.51
hit_rate of log_ret: 0.51
hit_rate of diff: 0.51
```

**A target expression requires an operator that looks ahead
(`po.rewm_mean`, `po.rewm_sum` or `po.rewm_rate`); otherwise, make the
target a column with Polars' `with_columns`, chained in the query before
the bank, and name it in `targets`.** The bank's refusal names the call
that makes the column. Here the target, how far from the mid each trade
printed, is known at the trade's own row, so the spec needs no embargo.
The quotes have no `quantity`, so the bank skips them, as it skips any row
with a null feature.
This code uses `trades.parquet` from [Example data](#example-data):

```python
edges = pl.scan_parquet("trades.parquet").with_columns(edge=(pl.col("price") - pl.col("mid")).abs())   # the target as a column, null on the quotes
by_size = po.spec.ewridge("by_size", targets=["edge"], features=["quantity"],
                          clock="ts", gap_cap="5m", half_life="30m", group="symbol")
printed = edges.online.fit_predict([by_size]).online.unnest([by_size]).collect()   # does a larger trade print farther from the mid?
is_trade = printed["quantity"].is_not_null()
print(f"trades predicted: {printed.filter(is_trade)['pred_edge'].count()} of {is_trade.sum()}")
print(f"quotes predicted: {printed.filter(~is_trade)['pred_edge'].count()} of {(~is_trade).sum()}")
```

```text
trades predicted: 893 of 899
quotes predicted: 0 of 2101
```

If a column holds a value from after its row, such as the price five
minutes on, give the spec an `embargo` of that horizon ([Labels that
arrive late](#labels-that-arrive-late)). To make a column from a window
over the stream, looking back or ahead, use `po.stream.with_windows` in the
query before the bank ([Windows as
columns](docs/WINDOWS.md#windows-as-columns)).

The command line has no `with_columns`, so make such a column in the file
it reads. `po.target`'s `relative_to=` and `relative=` are refused, and the
error shows the `with_columns` form.
[`po.target`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.target)
only names a target's output fields apart from its column.

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
`half_life`, `gap_cap`, `restart_after_step_back`, `session_gap`, `embargo`,
`drift_threshold` and `coef_every`, and a model's `window_size`, its cadences
(`window_every`, `solve_every`, `pca_every`, `prune_every`) and its own
half-lives. A few things still count rows, and each model's entry in
[docs/MODELS.md](docs/MODELS.md) names them.

**Quantities measured on a temporal clock reach the output in seconds.**
`holt`'s trend, for one, is per second. The clocks themselves keep the
clock column's own type: `emit_clocks` ([Labels that arrive
late](#labels-that-arrive-late)), and the clock columns of `summary()`,
`groups()` and `closed_groups()`. This code fits `holt`, a level and its
trend, to `t`, a column that rises by one each minute, once on the numeric
clock `t` and once on the `Datetime` clock `ts`. The two clocks mark the
same rows, so the weights agree, and the trend is the same rate in two
units.
It reads `df` from [Example data](#example-data):

```python
import textwrap

on_t = po.spec.holt("on_t", targets=["t"], clock="t", half_life=100.0, gap_cap=300.0)       # numbers: t counts minutes
on_ts = po.spec.holt("on_ts", targets=["t"], clock="ts", half_life="100m", gap_cap="300m")  # durations: ts is a Datetime
last = po.ModelBank([on_t, on_ts]).fit_predict(df).tail(1)
for name, unit in [("on_t", "per unit of t, a minute"), ("on_ts", "per second")]:
    fields = last[name].struct
    print(f"{name}: weight_sum {fields.field('weight_sum')[0]:.2f}, trend {fields.field('coef')[0][1]:.6f} {unit}")

try:
    po.ModelBank([po.spec.holt("wrong", targets=["t"], clock="ts", half_life=100.0, gap_cap=300.0)]).fit_predict(df)
except ValueError as err:   # a plain number on a Datetime clock
    print(textwrap.fill(str(err), 100))
```

```text
on_t: weight_sum 135.66, trend 1.000406 per unit of t, a minute
on_ts: weight_sum 135.66, trend 0.016673 per second
spec "wrong": clock column "ts" has dtype datetime[μs], a temporal clock, but half_life is a plain
number, which it cannot read: a number has no unit, and the column's own (e.g. epoch microseconds)
would silently become it. Give half_life and the other clock parameters as durations, e.g.
pl.duration(minutes=10), timedelta(minutes=10) or "10m"; or cast the clock to the unit you mean,
e.g. pl.col("ts").dt.epoch("s").cast(pl.Float64), and use that column.
```

**The bank refuses a clock parameter of the wrong kind** before it reads a
row, as the last lines show, and its error names the column, the parameter
and the fix. Only `0` and `inf`, which mean the same in every unit, may
stay numbers where a parameter takes them:

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

**An integer clock is read as its integers**: any column from `Int8` to
`Int64` or `UInt8` to `UInt64`. The step between two rows is taken in
integers before it becomes a double. Every decision on the clock is taken on
the integers too: a step back, a gap past `gap_cap`, a cadence, a window's
edge and an embargo's release. So an `Int64` column of epoch nanoseconds
keeps its nanoseconds, where a double near 1.8e18 resolves 256 of them. Its
clock parameters stay numbers, and a fractional one such as `gap_cap=0.5` is
compared as the number it is.

**A clock column the bank cannot read exactly is refused, by row or by
name.** A `UInt64` value past the largest `Int64` is refused by row. An
`Int128` or `UInt128` column is refused by name: cast it to `Int64` if its
values fit, or after subtracting an origin. A `UInt128` column is refused
wherever it is in the frame, since no frame carrying one can reach the
bank. A later chunk whose clock is a float, or an integer of another width,
is refused by name.

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

**Each policy shows in `weight_sum`, the weight behind each row's
prediction.** To show each, this code moves the afternoon session of `df`
two days on, which puts a weekend between rows 199 and 200 (counting from
0), and fits one regression under four policies:

| spec | its policy | the weekend counts as |
|---|---|---|
| `capped` | `gap_cap="5m"` | five minutes: the fit ages a little |
| `uncapped` | `gap_cap="3d"` | two days: about a hundred half-lives empty the fit |
| `one_minute` | `session="session"`, `session_gap="1m"` | one minute, the step chosen for the session boundary |
| `reset` | `session="session"`, `session_gap="reset"` | a fresh start: the model forgets everything |

`weight_sum` is read before a row's own step, so a reset shows on row 200
itself, and a step's decay on row 201. A numeric clock takes the same
parameters as numbers of its units, such as `half_life=600.0,
gap_cap=300.0` on `t`, and `lam`, the weight kept per unit, may stand in
for `half_life`.
This code uses `df` from [Example data](#example-data):

```python
afternoon = pl.col("session") == "a"
df.with_columns(ts=pl.when(afternoon).then(pl.col("ts") + pl.duration(days=2)).otherwise(pl.col("ts"))) \
  .write_parquet("weekend.parquet")

common = dict(targets=["y"], features=["x0", "x1"], clock="ts", half_life="30m")   # ts is a Datetime: durations
policies = [
    po.spec.ewridge("capped", gap_cap="5m", **common),                             # the weekend counts as five minutes
    po.spec.ewridge("uncapped", gap_cap="3d", **common),                           # the weekend counts in full
    po.spec.ewridge("one_minute", gap_cap="5m", session="session", session_gap="1m", **common),   # one minute at the boundary
    po.spec.ewridge("reset", gap_cap="5m", session="session", session_gap="reset", **common),  # a fresh start there
]
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

This code fits `ewridge` with no decay to `df`'s rows in their order,
reversed and shuffled, and solves the same least squares with numpy. The
last line reads `settled_frac` on the last row under `half_life=1e12`.
It reads `df` from [Example data](#example-data):

```python
batch = po.spec.ewridge("batch", targets=["y"], features=["x0", "x1", "x2"],
                        half_life=float("inf"),   # no clock, no decay
                        ridge=0.0)                 # plain least squares, in place of the default ridge of 1e-6
for order, rows in [("in order", df), ("reversed", df.reverse()), ("shuffled", df.sample(fraction=1.0, shuffle=True, seed=1))]:
    coef = po.ModelBank([batch]).fit_predict(rows)["batch"].struct.field("coef")[-1]   # the fit after the last row
    print(f"{order:<9}", " ".join(f"{c:9.6f}" for c in coef))
design = np.column_stack([np.ones(df.height), df["x0"], df["x1"], df["x2"]])
print(f"{'numpy':<9}", " ".join(f"{c:9.6f}" for c in np.linalg.lstsq(design, df["y"], rcond=None)[0]))

huge = po.spec.ewridge("huge", targets=["y"], features=["x0"], clock="t", half_life=1e12, gap_cap=300.0)
print(f"settled_frac after 399 units of t under half_life=1e12: {po.ModelBank([huge]).fit_predict(df)['huge'].struct.field('settled_frac')[-1]:.1e}")
```

```text
in order   0.992088  1.994121 -1.007945 -0.026134
reversed   0.992088  1.994121 -1.007945 -0.026134
shuffled   0.992088  1.994121 -1.007945 -0.026134
numpy      0.992088  1.994121 -1.007945 -0.026134
settled_frac after 399 units of t under half_life=1e12: 2.8e-10
```

### A hard window

A half-life never forgets entirely: three half-lives back still carries
12.5% of the weight. To make every row `w` clock units old or older count
for exactly nothing, give `window_size=w`. Five models take it: `ewridge`,
`lasso`, `ew_cov`, `ew_class` and `marginal`.

```
weight(age) = 0.5 ** (age / half_life)   if age < window_size   (closed="right", the default)
                                         or age <= window_size  (closed="both")
            = 0                          otherwise
```

**`closed` says which edge holds the row exactly `window_size` old, as
Polars' `rolling_*_by(closed=)` does.** Under `"right"`, the default, that
row has left, as it has in the [window operators](#making-features-and-targets-from-windows) and in
`rolling_sum_by`. Under `"both"` it stays. A model's window always holds the
row it ends at, because the model reads its fit after it has learned that
row. So `"left"` and `"none"` are refused, and so is `"both"` without a
`window_size`.

**Inside the window the weights still halve every `half_life`, so the
newest row counts most.** For a flat rolling fit with equal weights, give
`half_life=float("inf")` with the `window_size`.

**A window forgets a break that a half-life still holds.** In `df`, the
beta of `ret` on `signal_a` flips from 0.8 to −0.8 at row 200 (counting
from 0). This code follows it with a half-life of 100 rows, and with a flat
window of 100 rows. A hundred rows after the break, the window holds only
rows after it and reads the new beta, while the half-life still gives the
old rows half their weight.
It reads `df` from [Example data](#example-data):

```python
common = dict(targets=["ret"], features=["signal_a"], clock="t", gap_cap=300.0, coef_every=0)   # the beta on every row
decayed = po.spec.ewridge("decayed", half_life=100.0, **common)                              # a half-life of 100 rows
windowed = po.spec.ewridge("windowed", half_life=float("inf"), window_size=100.0, **common)  # the last 100 rows, equally weighted
fitted = po.ModelBank([decayed, windowed]).fit_predict(df)
print("row  decayed  windowed")
for row in [199, 250, 300, 399]:
    beta = [fitted[name].struct.field("coef")[row][1] for name in ("decayed", "windowed")]   # [intercept, signal_a]
    print(f"{row:<4} {beta[0]:7.2f}  {beta[1]:8.2f}")
```

```text
row  decayed  windowed
199     0.77      0.83
250     0.29      0.01
300     0.00     -0.78
399    -0.39     -0.78
```

**The cut is exact.** At time `t`, the rows learned at or before a time `u`
contribute `0.5 ** ((t − u) / half_life)` times the running sum as it stood
at `u`. Subtracting that leaves exactly the rest. The model keeps a ring
of snapshots for this, one every row by default. The edge is the oldest
snapshot inside the window, so a coarse spacing discards a little more
than asked, never less.

**`window_every` spaces the snapshots on the clock, as `solve_every`
spaces a regression's solves.** It takes a number of clock units, or a
duration such as `"1m"` on a temporal clock. `max_rows_between_snapshots`
caps the rows between them, and whichever comes first takes a snapshot.
Under `window_every` the effective window is in
`[window_size − window_every, window_size]`, whatever the row rate, with
`window_every` the spacing in force: a thinning `window_budget` doubles it.

**`weight_sum` is the weight inside the window.** This code fits `ew_cov`
over a window of 150 units of `t` with a half-life of 50, and a snapshot
every 10 units. The weight a row a unit apart adds settles at
`1 / (1 − 0.5 ** (1 / 50))`, about 72.6, without a window. A window keeps
the share of it younger than its effective edge, `1 − 0.5 ** (edge / 50)`,
with the edge between 140 and 150 units back.
It reads `df` from [Example data](#example-data):

```python
cut = po.spec.ew_cov(
    "cut", features=["x0", "x1"], clock="t", gap_cap=300.0, half_life=50.0,
    window_size=150.0,             # a row this many clock units old or older contributes exactly nothing
    window_every=10,               # a snapshot every 10 units of t
    window_budget={"refuse": 64},  # at most 64 MiB of snapshots per ring
)
whole = po.spec.ew_cov("whole", features=["x0", "x1"], clock="t", gap_cap=300.0, half_life=50.0)
last = po.ModelBank([cut, whole]).fit_predict(df).tail(1)
ceiling = 1 / (1 - 0.5 ** (1 / 50))
print(f"without a window: weight_sum {last['whole'].struct.field('weight_sum')[0]:.2f}, the ceiling {ceiling:.2f}")
print(f"inside the window: weight_sum {last['cut'].struct.field('weight_sum')[0]:.2f}, "
      f"between {ceiling * (1 - 0.5 ** (140 / 50)):.2f} and {ceiling * (1 - 0.5 ** (150 / 50)):.2f}")
```

```text
without a window: weight_sum 72.35, the ceiling 72.64
inside the window: weight_sum 63.43, between 62.21 and 63.56
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
| `marginal` | every pair's weight, means and second moments, so `corr`, `beta` and `t_stat` describe the window. Its lag moments, under `window_lags=True`, are an estimate rather than exact | `bins`, `feature_moments="shared"`, and `lags` unless `window_lags=True` |

**`window_budget` caps the snapshots' memory, per ring, in MiB.** The
ring's memory grows with the window: for an `ew_cov` over 20 columns, a
1,000-row window holds about 3 MB per group, divided by the rows between
snapshots.

| `window_budget` | when a chunk would take a ring past the cap |
|---|---|
| not given | the chunk is refused past 256 MiB |
| `{"refuse": 64}` | the chunk is refused before any of it is learned, and the bank goes on as it was. The error names the ring's size, its snapshot cadence and the ways to lift the cap |
| `{"thin": 64}` | the ring drops every other snapshot and doubles its spacing, on the clock or in rows, which, like `window_every`, only ever shortens the window |
| `{"refuse": float("inf")}` | nothing: no cap |

**A refusal comes before the bank learns any of the chunk, except for two
overruns found only as the rows go in.** One is under
`drift_action="reset"`. The other is a snapshot that grows within the
chunk, as an `ewridge`'s or a `lasso`'s does when its targets first go
missing on different rows. After either, the bank refuses every call that
follows, so rebuild it from its last save with `po.ModelBank.load`.

### A local fit along any feature

To fit any model locally along a feature, sort the rows by that feature
with Polars' `sort` in the query before the bank, and name the feature as
the spec's `clock`. `half_life` is then a bandwidth in that feature's
units. Each row is fitted on the rows before it in that order, each
weighted by `0.5 ** (Δx / half_life)`, with `Δx` its distance from the row
in that feature: an exponential kernel. The fit is a local regression from
running sums that do not grow. It agrees to 1e-12 with a kernel-weighted
least squares recomputed from scratch at every row, which
`tests/test_ewridge.py` holds it to.

**The kernel is one-sided, so the fit lags.** Narrow the bandwidth to lag
less, and widen it to smooth the noise. This code makes 1,200 points of
`sin(x)` plus noise of standard deviation 0.1, for x between −3 and 3, and
fits a straight line in `x` locally at two bandwidths. It prints each fit's
root-mean-square distance from the curve, beside the best straight line's:

```python
import numpy as np
import polars as pl
import polars_online as po

rng = np.random.default_rng(1)
x = rng.uniform(-3.0, 3.0, 1200)
pl.DataFrame({"x": x, "y": np.sin(x) + 0.1 * rng.standard_normal(1200)}).write_parquet("curve.parquet")

def local(bandwidth):
    kernel = po.spec.ewridge(
        "kernel", targets=["y"], features=["x"],
        clock="x",                  # the clock is a feature: the decay is a kernel in it
        half_life=bandwidth,        # the bandwidth, in x's own units
        gap_cap=1.0,                # a wider gap decays as if it were this wide
        max_rows_between_solves=1,  # refit at every row
        coef_every=0,               # write the coefficients on every row: the line through the row's neighbourhood
        min_weight=20.0,
    )
    query = pl.scan_parquet("curve.parquet").sort("x")   # sorted by the feature, in the query before the bank
    return query.online.fit_predict([kernel]).online.unnest([kernel]).collect()   # pred_y: the curve at each row's x

narrow, wide = local(0.25), local(1.0)
fit = narrow["pred_y"].is_not_null()                    # the rows past the warm-up
curve = narrow.filter(fit)
truth = np.sin(curve["x"].to_numpy())
line = np.polyval(np.polyfit(narrow["x"], narrow["y"], 1), curve["x"].to_numpy())

def rms(pred):
    return float(np.sqrt(np.mean((np.asarray(pred) - truth) ** 2)))

print(f"bandwidth 0.25: {rms(curve['pred_y']):.2f}")
print(f"bandwidth 1.0:  {rms(wide.filter(fit)['pred_y']):.2f}")
print(f"straight line:  {rms(line):.2f}")
```

```text
bandwidth 0.25: 0.08
bandwidth 1.0:  0.29
straight line:  0.39
```

The narrow kernel follows the curve the straight line cannot. The wide one
is most of the way back to the line.

**The clock column can stay out of `features`.** With other columns as
features, the fit is a regression whose coefficients move along the clock.

### Row order and the two guarantees

Two guarantees hold for every run, and both rest on a fixed row order,
which the caller supplies.

**Every prediction and diagnostic is out-of-sample.** Each row is scored
from the state as it stood before the row's own target was learned, so no
row's outcome reaches its own prediction or the diagnostics measured on
it.

**The numbers do not depend on the chunking.** One chunk or a thousand,
with or without a save and resume in the middle, gives bit-identical
numbers. Only which rows carry `coef`, and `support_coef` beside it, can
differ, and only by default. The bank then writes them on each group's last
accepted row in every chunk, so smaller chunks report them more often.
Under `coef_every` or `max_rows_between_coefs` they do not move
([Coefficients](#coefficients)).

This code checks both on `df`. It predicts row 300 with a bank that has
learned only rows 0 to 299, and compares that with the same row of one
run over every row. It then runs the stream again one row per chunk.
It reads `df` from [Example data](#example-data):

```python
honest = po.spec.ewridge("honest", targets=["y"], features=["x0", "x1"], clock="t", half_life=100.0, gap_cap=300.0)
one_run = po.ModelBank([honest]).fit_predict(df)["honest"].struct.field("pred_y")

first_300 = po.ModelBank([honest])
first_300.fit_predict(df.head(300))                           # learns rows 0 to 299
alone = first_300.predict(df.slice(300, 1))["honest"].struct.field("pred_y")[0]
print(f"row 300: {one_run[300]:.6f} in one run, {alone:.6f} from rows 0 to 299 alone")

by_row = pl.concat(po.ModelBank([honest]).fit_predict_batches(df, chunk_size=1))   # 400 chunks of one row
print(f"one chunk against 400: {(one_run - by_row['honest'].struct.field('pred_y')).abs().max()} largest difference")
```

```text
row 300: 2.142699 in one run, 2.142699 from rows 0 to 299 alone
one chunk against 400: 0.0 largest difference
```

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

**The query check names the step and its fix.** This code joins each
row's venue to a fee, once as Polars' `join` does by default and once with
`maintain_order="left"`, and prints what the bank warns of each.
It reads `lf` from [Example data](#example-data):

```python
import textwrap
import warnings

pl.DataFrame({"venue": ["X", "Y"], "fee": [0.1, 0.2]}).write_parquet("fees.parquet")
priced = po.spec.ewridge("priced", targets=["y"], features=["x0", "fee"], half_life=100.0)
for order in [None, "left"]:
    joined = lf.join(pl.scan_parquet("fees.parquet"), on="venue", **({"maintain_order": order} if order else {}))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        joined.online.fit_predict([priced]).collect()
    said = [str(w.message) for w in caught if issubclass(w.category, po.OrderNotGuaranteedWarning)]
    print(f"maintain_order={order!r}:", textwrap.fill(said[0], 100) if said else "no warning")
```

```text
maintain_order=None: lf.online.fit_predict: the plan's row order is not guaranteed, and an online model learns in row
order. 1 node(s) leave it unspecified: a join without maintain_order (pass maintain_order="left" to
keep the order of the left input). The streaming engine that runs the plan may deliver rows in an
order lf.collect() would not, and it may differ between runs. Give the plan a fixed order --
maintain_order="left" on a join, maintain_order=True on a group_by, a sort after a unique() or
before the bank -- or, for a plan whose order is fixed by other means,
warnings.simplefilter("ignore", polars_online.OrderNotGuaranteedWarning).
maintain_order='left': no warning
```

**The query check is best-effort.** It reads the query's `explain()` text,
and walks the query's steps only when that names a join, an aggregation, a
`unique` or a `sort`. A step followed by a `sort` counts as ordered, unless
that sort is by several keys without `maintain_order=True`. Such a sort
leaves rows with equal keys in no particular order: 2,109 of 10,000 rows
moved when measured. A join with `maintain_order="left"` is followed on its
left side only. A query Polars cannot serialize passes without a warning,
and the check never fails a run. For a query whose order you know, silence
the warning with
`warnings.simplefilter("ignore", po.OrderNotGuaranteedWarning)`.

**`ModelBank.fit` does not warn when its state cannot depend on the row
order.** That is when every spec is an `ewridge` or `rls` with decay off
and none of the settings that read the order. Those include a window, a
session, a weight, an embargo, a drift reset, a `coef` cadence, a window
expression as a target, and any diagnostic but `emit_drift`, `emit_clocks`
and `emit_error_inflation`. Such sums reach the same state in any order, to
rounding, and `fit` keeps only the state.

**The clock is checked group by group,** so groups may interleave freely:
only the rows of one group need to be in clock order. Equal clock values
are zero apart. A row with a null feature is skipped, but its clock is
still checked. A row that `predict` scores is never refused ([Serving
without learning](#serving-without-learning)).

**`restart_after_step_back` and the size of a step back decide what it
means.** It has no default, since only the caller knows how late a row can
be, and `0` restarts the model at every step back:

| a step back | is read as | and the bank |
|---|---|---|
| any size, with `restart_after_step_back` unset, the default | rows out of order | refuses the chunk |
| no larger than `restart_after_step_back` | a late row, such as a transposed pair or a row a minute late | refuses the chunk |
| larger than `restart_after_step_back` | a new start, such as a replayed day or a restarted feed | restarts the model |
| any size, on a row whose `session` value changes | a new session, where the step is no measure of time | applies `session_gap`, or restarts the model under `session_gap="reset"` |

**A refused chunk leaves the bank as it was.** The bank runs the clock of
every group of every spec over the chunk before it learns any row. So one
late row in one group changes nothing: correct the chunk and feed it again.
To resume a saved state on input that overlaps it, pass the input through
`bank.skip_learned(...)`, which drops the rows the state has learned ([Save
and load](#save-and-load)).

**The error locates the step and names the way out, and the summary counts
what the clock rules met.** The row the error names is counted from the
start of the input however it arrives (one frame, batches, a query or the
command line's file). On a temporal clock the step is a duration.
`bank.summary()` reports `clock_backwards`, the rows whose clock fell below
the previous row's within a session, and `resets`, the rows where a stream
restarted. This code feeds ten rows of `df` with the row at `t = 6` two
rows late, first with `restart_after_step_back` unset, then set below the
step:
It reads `df` from [Example data](#example-data):

```python
import textwrap

late = df.head(10)[[0, 1, 2, 3, 4, 5, 7, 8, 6, 9]]           # t runs 0 ... 5, 7, 8, 6, 9
strict = po.spec.ewridge("strict", targets=["y"], features=["x0"], clock="t", half_life=100.0, gap_cap=300.0)
try:
    po.ModelBank([strict]).fit_predict(late)
except ValueError as err:
    print(textwrap.fill(str(err), 100))

lenient = po.spec.ewridge("lenient", targets=["y"], features=["x0"], clock="t", half_life=100.0, gap_cap=300.0,
                          restart_after_step_back=1.0)       # a step back larger than 1 starts the model over
bank = po.ModelBank([lenient])
bank.fit_predict(late)
counts = bank.summary().row(0, named=True)
print(f"{counts['spec']}: clock_backwards {counts['clock_backwards']}, resets {counts['resets']}")
```

```text
spec "strict": clock column "t" goes backwards by 2 at row 8 (restart_after_step_back is unset, so
every step back is refused); the bank was not updated. Sort each group by the clock; to resume a
saved state on input that overlaps it, drop the rows it has learned, with
ModelBank.skip_learned(frame) in Python or --skip-learned on the command line; or, if a step back
this large starts the stream over, set restart_after_step_back below it (a step back no larger than
the setting is a late row).
lenient: clock_backwards 1, resets 1
```

### Groups

To fit one model per value of a column, name the column with `group`: the
bank fits a separate model for each distinct value, in the same pass. Each
group's model sees only its own rows, on its own clock, so a group's
numbers are those of the same spec fitted to that group's rows alone.
This code uses `df` from [Example data](#example-data):

```python
per_stock = po.spec.ewridge(
    "per_stock", targets=["y"], features=["x0", "x1"], clock="t", half_life=600.0, gap_cap=300.0,
    group="stock_id",           # one model per distinct value of this column
)
bank = po.ModelBank([per_stock])
together = bank.fit_predict(df)
for row in bank.groups().iter_rows(named=True):
    print(f"group {row['group']}: {row['rows_processed']} rows, the last at t = {row['last_clock']}")

alone = {**per_stock, "group": None}           # the same spec without a group, fed stock b0's rows alone
b0 = po.ModelBank([alone]).fit_predict(df.filter(pl.col("stock_id") == "b0"))
in_bank = together.filter(pl.col("stock_id") == "b0")["per_stock"].struct.field("pred_y")
print(f"b0 in the bank against b0 alone: {(in_bank - b0['per_stock'].struct.field('pred_y')).abs().max()} largest difference")
```

```text
group b0: 100 rows, the last at t = 396.0
group b1: 100 rows, the last at t = 397.0
group b2: 100 rows, the last at t = 398.0
group b3: 100 rows, the last at t = 399.0
b0 in the bank against b0 alone: 0.0 largest difference
```

**A group or session column keeps the type its first chunk had.** A key is
its value's text, so an `Int64` `1` and a `Float64` `1.0` would be two
groups. The bank refuses a column of another kind by name, and keeps the
kind in its state: integer, float, text, boolean, or a temporal type with
its unit and zone. An `Int32` after an `Int64`, or a `Categorical` after a
`String`, is the same kind and goes on. `bank.groups()` and the other
readers list an integer column's keys as numbers, the null group first.

**Without `group_close`, the bank keeps every group it has seen.** To free
the groups that have gone quiet, find them with `bank.groups()` and drop
them with `bank.drop_groups(...)` ([In a loop](#in-a-loop-modelbank)).

**With `group_close`, the bank writes out a finished group's running sums
and frees its memory,** which keeps its memory bounded when new group
values never stop appearing. `group_close="monotone"` finishes a group once
a higher key arrives, so it needs the keys in order, as a block number or a
date is. `group_close="session"` finishes a group once its `session` value
changes. [One row per finished group](#one-row-per-finished-group) shows
what is written.

### Weights

To make some rows count more than others, name a column of row weights
with `weight`. A row of weight 2 counts as two rows of weight 1.

**A weight of 0 is legal, and a negative weight is refused.** A row of
weight 0 is scored and moves the clock, but teaches the model nothing
([Nulls, and three ways to hold a row back](#nulls-and-three-ways-to-hold-a-row-back)).

**In a model that keeps means, scaling every weight by the same factor
changes nothing but `weight_sum`, so scale `min_weight` by the same
factor.** This code gives row 100 a weight of 0 and every other row a
weight of 1, then scales every weight by 10. With no decay, `weight_sum`
counts the weight learned before each row: row 100 is scored, but the row
after it finds the weight unchanged.
It reads `df` from [Example data](#example-data):

```python
weighted = po.spec.ewridge(
    "weighted", targets=["y"], features=["x0", "x1"], half_life=float("inf"),
    weight="w",                # a column of row weights
)
ones = df.with_columns(w=pl.when(pl.int_range(pl.len()) == 100).then(0.0).otherwise(1.0))
tens = ones.with_columns(pl.col("w") * 10)
fields = [po.ModelBank([weighted]).fit_predict(frame)["weighted"].struct for frame in (ones, tens)]
for row in [99, 100, 101, 102]:
    print(f"row {row}: pred_y {fields[0].field('pred_y')[row]:.4f}, weight_sum {fields[0].field('weight_sum')[row]:g}")
print(f"weights times 10: pred_y moves by at most {(fields[0].field('pred_y') - fields[1].field('pred_y')).abs().max():g}, "
      f"weight_sum is {fields[1].field('weight_sum')[-1] / fields[0].field('weight_sum')[-1]:g} times as large")
```

```text
row 99: pred_y -2.2004, weight_sum 99
row 100: pred_y 0.6661, weight_sum 100
row 101: pred_y 2.5369, weight_sum 100
row 102: pred_y 2.4487, weight_sum 101
weights times 10: pred_y moves by at most 0, weight_sum is 10 times as large
```

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
means, and the linear fits add a third, on by default on `ewridge` alone.
The gates hold back output only: the model learns from every row either
way.

| setting | default | withholds a prediction while | read from |
|---|---|---|---|
| `min_weight` | by model, below | `weight_sum` is below it | the weight behind the state |
| `min_settled_frac` | `0`, off | the decay window is less full than this fraction of its steady state | `settled_frac = 1 − 2^(−T / half_life)`, with `T` the decay time the model has seen: `0.5` at one half-life, `0.75` at two, whatever the row rate |
| `max_error_inflation` | `sqrt(2)` on `ewridge`; off on `rls`, `kalman` and `lasso` | estimation error would inflate the prediction's error over the noise floor by more than this factor | `error_inflation = sqrt(1 + edf / n_kish)`: the effective degrees of freedom the fit used, over Kish's effective sample size behind it (below) |

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
| 0, each having a gate of its own or nothing to gate | `ewridge`, `seqtest`, `rcov`, `hmm`, `corrchange`, `audit` |

**`min_settled_frac` is off by default**, because a fit kept as weighted
means is unbiased from its first row when the process is stationary. Set it
when the half-life was chosen to average over regimes or seasons that a
shorter history would not represent.

**`max_error_inflation` follows the size of the fit:** add a feature, and
the gate holds predictions back longer. It reads Kish's count, so one row
carrying a hundred times the weight of the others counts as barely one. Each
model that has it reads its own estimation variance. The others refuse the
setting and say why:

| model | `error_inflation` | |
|---|---|---|
| `ewridge` | `sqrt(1 + edf / n_kish)` | on by default at `sqrt(2)` |
| `rls` | `sqrt(1 + k / n_kish)`, `k` the coefficients: its `edf` is at most `k` under the fading prior | off unless set |
| `lasso` | `sqrt(1 + df / n_kish)` per penalty, `df` the active coefficients and the intercept | off unless set |
| `kalman` | `sqrt(1 + z' P z / R)` for each row: the prediction's own variance, its coefficients' covariance carried through the row's clock gap, over the noise | off unless set; read per row, so a row far from the design is held back where others pass |

**Each row says how ready its model was, and why a prediction is
missing.** `withheld_reason` names the gate that held a row back, and the
gates are checked in the order of its values, so the first that applies is
the one named. This code sets all three gates on one spec and counts the
rows each withholds:
It reads `df` from [Example data](#example-data):

```python
warm = po.spec.ewridge(
    "warm", targets=["y"], features=["x0", "x1"], clock="t", half_life=100.0, gap_cap=300.0,
    min_settled_frac=0.02,      # withhold until the decay window is 2% full
    min_weight=8.0,             # and until the weight behind the fit reaches 8
    max_error_inflation=1.05,   # and while estimation error would add more than 5% over the noise floor
)
rows = po.ModelBank([warm]).fit_predict(df).unnest("warm")
counts = rows.group_by("withheld_reason").len().sort("withheld_reason", nulls_last=True)
for reason, n in counts.iter_rows():
    print(f"{reason or 'predicted'}: {n} rows")
first = rows.filter(pl.col("pred_y").is_not_null()).row(0, named=True)
print(f"first prediction at t = {first['t']:g}: weight_sum {first['weight_sum']:.2f}, settled_frac {first['settled_frac']:.3f}")
```

```text
below_min_settled_frac: 4 rows
below_min_weight: 5 rows
above_max_error_inflation: 21 rows
predicted: 370 rows
first prediction at t = 30: weight_sum 27.18, settled_frac 0.182
```

The readiness fields:

| field | on | what it holds |
|---|---|---|
| `settled_frac` | every model that writes a row | how full the decay window is |
| `withheld_reason` | every model that writes a row | why `pred_y` is null, or null when it is not: `below_min_settled_frac`, `below_min_weight`, `above_max_error_inflation` or `below_warm_rows` (the rows `kmeans`, `hmm` and `bocpd` hold to seed themselves from), in that order of precedence |
| `error_inflation_y` | `ewridge`, `rls`, `kalman`, under `emit_error_inflation` | the row's own estimation variance over the noise: high on a row leaning on a direction the data never showed, at one triangular solve a row |
| `support_coef` | `ewridge`, `huber`, `quantile`, beside `coef` | each coefficient's data share, `1 - ridge * (S^-1)_jj` on the system the solve inverts. A duplicated pair of features reads 0.5 each |
| `se_coef` | `ewridge`, `rls`, `kalman`, under `emit_se_coef`, beside `coef` | each coefficient's standard error in its own units: `sigma` times the fit's own covariance, `kalman`'s posterior exactly. A report, not a gate; it errs large in warm-up and under a ridge |

**`summary()` carries the same readings per group, and warns of a gate
that will not open.** It adds `weight_sum_settled`, the weight the stream
settles at. Once the stream is 95% settled, a `ReadinessWarning` names,
once, a coefficient more ridge than data. It also names a noise gate or a
`min_weight` that has withheld every row for a further half-life, and that
the rows, at their rate so far, would not meet. It gives the half-life or
the ceiling that would change that, and promises nothing of rows that come
faster. The command line closes a run with one line per spec whose groups
ended withheld or with a coefficient more ridge than data. [Is it
ready?](docs/DIAGNOSTICS.md#is-it-ready) reads these fields to decide
when a model can be trusted.

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

This code fits `df`'s minute-apart rows under `embargo="5m"`, with
`emit_clocks=True`, and prints the clocks of five rows, counting from 0.
Each row is scored from the rows at least five minutes older, the one
exactly five minutes older included, so the first five rows have learned
nothing.
It reads `df` from [Example data](#example-data):

```python
fwd = po.spec.ewridge("fwd", targets=["y"], features=["x0", "x1"],
                      clock="ts", gap_cap="1h", half_life="30m",
                      embargo="5m",      # a label takes five minutes to arrive: learned five minutes later
                      emit_clocks=True)  # scored_clock and learned_clock on every row
rows = po.ModelBank([fwd]).fit_predict(df).unnest("fwd")
for row in [0, 4, 5, 6, 300]:
    r = rows.row(row, named=True)
    learned = f"the rows up to {r['learned_clock']:%H:%M}" if r["learned_clock"] else "no row yet"
    print(f"row {row}: scored at {r['scored_clock']:%H:%M}, having learned {learned}, weight_sum {r['weight_sum']:.2f}")
```

```text
row 0: scored at 09:30, having learned no row yet, weight_sum 0.00
row 4: scored at 09:34, having learned no row yet, weight_sum 0.00
row 5: scored at 09:35, having learned the rows up to 09:30, weight_sum 1.00
row 6: scored at 09:36, having learned the rows up to 09:31, weight_sum 1.98
row 300: scored at 14:30, having learned the rows up to 14:25, weight_sum 43.74
```

| case | what `embargo` does |
|---|---|
| what the delay counts | the time that passed on the clock column, skipped rows included, and `session_gap` where a session restarts the clock. It is not the capped step the model decays by, so the release depends on the rows alone and survives any chunking |
| exactly at the delay | the row is learned. The time that passed is held exactly: in integer nanoseconds on a temporal clock, and by one subtraction of the two rows' values on a number clock, in integers on an integer clock. Rows 1 ms apart under `embargo="2s"` are each learned 2,000 rows on |
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

This code doubles `lf`, whose clock `t` moves a row at a time, and holds
the scored copies to the built-in delay of the same five units.
It reads `lf` from [Example data](#example-data):

```python
doubled = po.stream.embargo(lf, clock="t", delay=5.0)     # every row twice, in clock order
scored = doubled.online.fit_predict(
    [po.spec.ewridge("m", targets=["y"], features=["x0"], clock="t", gap_cap=10.0,
                     half_life=50.0, weight="_online_role_weight")]   # 0 on the copies to score, 1 on the rest
).filter(pl.col("_online_role") == "predict").collect()               # keep the scored copies

# The built-in delay on the same rows: t is a row apart, so each t + 5 is another row's clock.
built_in = lf.online.fit_predict(
    [po.spec.ewridge("m", targets=["y"], features=["x0"], clock="t", gap_cap=10.0, half_life=50.0, embargo=5.0)]
).collect()
for field in ["pred_y", "weight_sum"]:
    a, b = scored["m"].struct.field(field), built_in["m"].struct.field(field)
    print(f"{field}: {a.count()} rows, {(a == b).sum()} equal to the bit")
```

```text
pred_y: 393 rows, 393 equal to the bit
weight_sum: 400 rows, 400 equal to the bit
```

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

This code holds row 100 of `df` back from a model in each of the three
ways, beside a run that learns it. For each it prints whether row 100 was
scored, the clock the model stands at after it, whether the fit moved, and
`weight_sum` on the row after it, on `df` from [Example data](#example-data):

```python
held = po.spec.ewridge("held", targets=["y"], features=["x0", "x1"], clock="t", half_life=100.0, gap_cap=300.0,
                       weight="w", max_rows_between_solves=1)   # a solve on every row, so each fit read is current
rows = df.with_columns(w=pl.lit(1.0))
row_100 = pl.int_range(pl.len()) == 100
ways = {
    "learned": rows,
    "a null target": rows.with_columns(y=pl.when(row_100).then(None).otherwise(pl.col("y"))),
    "weight 0": rows.with_columns(w=pl.when(row_100).then(0.0).otherwise(pl.col("w"))),
    "predict": rows,
}
before = po.ModelBank([held])
before.fit_predict(rows.head(100))                            # rows 0 to 99
print(f"{'':<14} {'scored':>6} {'clock':>6} {'fit moved':>9} {'weight_sum':>10}")
for way, frame in ways.items():
    bank = po.ModelBank([held])
    bank.fit_predict(frame.head(100))
    if way == "predict":
        scored = bank.predict(frame.slice(100, 1))            # scores row 100 and learns nothing
    else:
        scored = bank.fit_predict(frame.slice(100, 1))
    clock = bank.groups()["last_clock"][0]
    moved = bank.coef()["coef"].to_list() != before.coef()["coef"].to_list()
    after = bank.fit_predict(frame.slice(101, 1))["held"].struct.field("weight_sum")[0]
    is_scored = scored["held"].struct.field("pred_y")[0] is not None
    print(f"{way:<14} {'yes' if is_scored else 'no':>6} {clock:>6g} {'yes' if moved else 'no':>9} {after:>10.2f}")
```

```text
               scored  clock fit moved weight_sum
learned           yes    100       yes      72.89
a null target     yes    100        no      72.89
weight 0          yes    100        no      71.89
predict           yes     99        no      72.39
```

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
| the window operators of [`po.ops`](https://hgilde.github.io/polars-online/ops.html) | exponentially weighted means, sums and rates over a window of the clock, looking back or ahead | in [`po.stream.with_windows`](https://hgilde.github.io/polars-online/stream.html#polars_online.stream.with_windows), before the bank, as columns; or in a spec's `targets`, as a target the bank computes | [Making features and targets from windows](#making-features-and-targets-from-windows) |
| [`po.stream.refresh_time`](https://hgilde.github.io/polars-online/stream.html#polars_online.stream.refresh_time) | one grid for series that tick at their own times | before the bank | [Series that tick at their own times](#series-that-tick-at-their-own-times) |

### Making features and targets from windows

A window operator of [`po.ops`](https://hgilde.github.io/polars-online/ops.html)
returns a Polars expression whose value on each row is computed over the
rows near it on the clock. A window that looks back from the row makes a
feature, and one that looks ahead makes a target. `po.ewm_mean` and its
mirror `po.rewm_mean` are time-weighted means, as Polars' `ewm_mean_by`
computes one, and the sums and rates divide into weighted means such as a
VWAP (volume-weighted average price). Each operator keeps running sums, so
it does the same work on a row however long its window.

**To add windows as columns, call `po.stream.with_windows` before the
bank.** It takes formulas as `with_columns` does, computes them over the
stream in one pass, and returns each row once every window over it has
closed. Here the feature `mid_trend` is the mid against its time-weighted
mean over the last two minutes. The target `fwd_edge` is the next minute's
VWAP of the trades, less the mid now.
This code uses `trades.parquet` from [Example data](#example-data):

```python
notional = pl.col("price") * pl.col("quantity")
next_minute = dict(half_life="10s", window_size="1m")     # weights halve every 10 s from the row; the window ends a minute on
windowed = pl.scan_parquet("trades.parquet").online.with_windows(
    mid_trend=pl.col("mid") - po.ewm_mean("mid", half_life="30s", window_size="2m"),     # a feature: looks back
    fwd_edge=po.rewm_sum(notional, **next_minute) / po.rewm_sum("quantity", **next_minute) - pl.col("mid"),   # a target: looks ahead
    clock="ts", gap_cap="5m", group="symbol",              # the clock, the longest step it ages by, and one set of windows per symbol
).collect()
print("time      symbol  mid      mid_trend  fwd_edge")
for r in windowed.slice(200, 5).iter_rows(named=True):
    print(f"{r['ts']:%H:%M:%S}  {r['symbol']}     {r['mid']:.3f}  {r['mid_trend']:+.4f}    {r['fwd_edge']:+.4f}")
```

```text
time      symbol  mid      mid_trend  fwd_edge
09:31:48  AAA     100.107  +0.0536    -0.1181
09:31:49  BBB     100.105  +0.0504    -0.0299
09:31:50  BBB     100.094  +0.0395    -0.0196
09:31:50  BBB     100.096  +0.0403    -0.0253
09:31:50  BBB     100.083  +0.0275    -0.0127
```

**To have a model learn a target that looks ahead, put its expression in
the spec's `targets`, with an `embargo` of at least its window.** The bank
then computes the window itself, scores each row where it sits, and learns
from the row once its window has closed. A target's value is a minute in
the future when its row arrives, so a model that learned it at once would
score that minute's rows in sample.

[docs/WINDOWS.md](docs/WINDOWS.md) shows each part of this with a recipe.
Its recipes hold the nine operators to Polars' own, and show which ends of
a window are in and what a gap does to a window. Others fit a window
target beside its column form, make features in units of their spread,
and resume a window run.

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
after it runs the grid and the models in one pass. The recipe plants a
correlation: three prices share one move and add one of their own, as
large, so their changes correlate at 0.5. Each is observed at times of its
own, about once per clock unit. `ew_cov` reads the correlations on the
grid, and `ewridge` regresses one series' change on the other two's, whose
true coefficients are 1/3 each. The recipe then reads the same ticks on a
grid every 0.1 clock units, each series' last price carried forward:

```python
rng = np.random.default_rng(1)
steps = 200_000                                          # a hidden price path, 0.01 clock units a step
shared = np.cumsum(rng.normal(0.0, 0.1, steps))          # the move the three series share
names = ["AAA", "BBB", "CCC"]
frames = []
for s in names:
    price = shared + np.cumsum(rng.normal(0.0, 0.1, steps))    # plus a move of its own, as large
    t = np.cumsum(rng.exponential(1.0, 2_100))
    t = t[t < 2_000]                                     # about one tick per clock unit, at times of its own
    frames.append(pl.DataFrame({"symbol": s, "t": t, "px": price[(t * 100).astype(int)]}))
pl.concat(frames).sort("t").write_parquet("own_times.parquet")   # one stream, in clock order

grid = po.stream.refresh_time(pl.scan_parquet("own_times.parquet"), series="symbol", names=names,
                              clock="t", value="px")    # a query, since it was given one
returns = grid.select("time_refresh", *(pl.col(f"{s}_value").diff().alias(s) for s in names))  # each series' change since the last point
specs = [
    po.spec.ew_cov("comove", features=names, stats=["corr"],
                   clock="time_refresh", gap_cap=10.0, half_life=float("inf")),
    po.spec.ewridge("beta", targets=["AAA"], features=["BBB", "CCC"],
                    clock="time_refresh", gap_cap=10.0, half_life=float("inf")),
]
out = returns.online.fit_predict(specs).collect()         # the first point has no change, so both models skip it
last = out.row(-1, named=True)
print(f"{out.height} refresh times from {pl.read_parquet('own_times.parquet').height} ticks")
print("on the refresh grid: corr " + ", ".join(f"{last['comove'][f'corr_{a}_{b}']:.2f}" for a, b in [("AAA", "BBB"), ("AAA", "CCC"), ("BBB", "CCC")])
      + f"; AAA on BBB {last['beta']['coef'][1]:.2f}, on CCC {last['beta']['coef'][2]:.2f}")

ticks_by_series = pl.read_parquet("own_times.parquet").partition_by("symbol", as_dict=True)
fine = pl.DataFrame({"t": np.arange(0.0, 2_000.0, 0.1)})
for s in names:                                          # each series' last price at each point of the fine grid
    fine = fine.join_asof(ticks_by_series[(s,)].select("t", pl.col("px").alias(s)), on="t")
moves = fine.select(pl.col(names).diff()).drop_nulls()
print(f"on a grid every 0.1: corr {moves.corr()['AAA'][1]:.2f}, {moves.corr()['AAA'][2]:.2f}, {moves.corr()['BBB'][2]:.2f}")
```

```text
1064 refresh times from 5835 ticks
on the refresh grid: corr 0.34, 0.32, 0.36; AAA on BBB 0.27, on CCC 0.23
on a grid every 0.1: corr 0.03, 0.02, 0.03
```

The fine grid finds almost no correlation, since in most of its steps one
series moved and the others did not: the Epps effect. The refresh grid
finds about two thirds of the planted 0.5. What it misses comes from each
value's age at a refresh time, which is up to one of its own series'
intervals between ticks.


## Running a bank

Run a bank in one of three ways. Each gives the same numbers, whatever
the chunking of the rows. From a bank object, the output can leave as
Arrow, for a consumer that is not Polars:

| way | the call | use it for | where |
|---|---|---|---|
| inside a Polars query | `lf.online.fit_predict(specs)` | ordinary Polars steps chained after the bank | [As a query](#as-a-query-lfonlinefit_predict) |
| in your own Python loop | `bank.fit_predict(chunk)` | reading the bank between chunks | [In a loop](#in-a-loop-modelbank) |
| from a standalone command line, with no live Python at all | `online` | a scheduled job, or a deployment with no Python | [Running a bank as a job, without Python](#running-a-bank-as-a-job-without-python) |
| output as Arrow, from a bank object | `fit_predict_arrow`, `predict_arrow` | a consumer that is not Polars | [Output as Arrow](#output-as-arrow) |

### As a query: `lf.online.fit_predict`

[`lf.online.fit_predict(specs)`](https://hgilde.github.io/polars-online/namespaces.html#polars_online._frame.LazyFrameOnlineNamespace.fit_predict)
adds a model bank to a Polars query, a `LazyFrame`, which runs only when
you ask for its result. `df.online.fit_predict(specs)` does the same for a
`DataFrame` in memory. When the query runs, it feeds its rows to a bank
that starts with nothing learned, `chunk_size` rows at a time (100,000 by
default). Run it with `collect()` for one frame, `sink_parquet()` to write
a file without holding the result in memory, or `collect_batches()` for one
chunk at a time.

**The chunk size never changes the numbers.** It trades memory for
overhead: memory is the bank's state plus one chunk, however long the
files. The recipe runs the same spec over `ticks/*.parquet` in one chunk,
and over `ticks.parquet` one row at a time, and the two agree to the bit.
`df`'s `y` is `1 + 2·x0 − x1` plus noise, so the fit of each stock finds
those coefficients, and 0 for `x2`.
This code uses the files `ticks.parquet` and `ticks/*.parquet` from [Example data](#example-data):

```python
spec = po.spec.ewridge(                                   # a ridge regression of y for each stock, at two ridge values
    "ridge",
    targets=["y"], features=["x0", "x1", "x2"],
    clock="t", half_life=600.0, gap_cap=300.0,
    group="stock_id", ridge=[1e-6, 0.1], standardize=True,
)
(
    pl.scan_parquet("ticks/*.parquet")                    # a query over the files; nothing is read yet
    .online.fit_predict([spec], chunk_size=100_000)       # a bank with nothing learned yet; every run starts from the same place
    .filter(pl.col("ridge").struct.field("weight_sum") > 50)   # after the bank: filters what comes out, never what the bank learns from
    .select("t", "stock_id", "ridge")                     # Polars reads only these columns, and those the specs read, from the files
    .sink_parquet("fitted.parquet")                       # runs the query; memory is state + one chunk, however long the files
)
fitted = pl.read_parquet("fitted.parquet")
print(f"{fitted.height} rows past the filter; the record's fields: {fitted['ridge'].struct.fields}")

one_row_chunks = pl.scan_parquet("ticks.parquet").online.fit_predict([spec], chunk_size=1).collect()
same = one_row_chunks.filter(pl.col("ridge").struct.field("weight_sum") > 50)["ridge"].struct.field("pred_y__r0.1")
print(f"largest difference, chunks of 100,000 against chunks of 1: {(fitted['ridge'].struct.field('pred_y__r0.1') - same).abs().max()}")

last = fitted.filter(pl.col("stock_id") == "b3").row(-1, named=True)
intercept, b0, b1, b2 = last["ridge"]["coef"][:4]        # the fit at the first ridge value; the second's follow
print(f"b3's coefficients on its last row: intercept {intercept:.2f}, x0 {b0:.2f}, x1 {b1:.2f}, x2 {b2:.2f}")
```

```text
172 rows past the filter; the record's fields: ['pred_y__r0.000001', 'resid_y__r0.000001', 'pred_y__r0.1', 'resid_y__r0.1', 'weight_sum', 'settled_frac', 'withheld_reason', 'coef', 'support_coef']
largest difference, chunks of 100,000 against chunks of 1: 0.0
b3's coefficients on its last row: intercept 1.04, x0 1.99, x1 -0.97, x2 -0.13
```

`"ridge"` is one column whose value in each row is a record with named
fields ([Output field names](#output-field-names) lists them). `weight_sum`
is the weight a fit rests on, so the filter keeps each stock's rows once
its fit has seen about 50 rows' worth. `lf.online.fit_predict([spec]).head(5)`
feeds the bank the first 5 rows and no more.

**Filter after the bank, not before it.** A `filter` after the bank keeps
the query running a chunk at a time. One before the bank makes Polars hold
several blocks of each parquet file per thread: 2.5 GB at 12M rows,
against 0.78 GB for the same filter after
([docs/PERFORMANCE.md](docs/PERFORMANCE.md) §11).

**To keep rows from teaching the model, give them weight `0` before the
bank, in place of a filter:** make the column with `with_columns`, and name
it in the spec's `weight`. The rows still come out scored, and the clock
advances through them, so no gap opens where they were. Here only venue
X's 200 rows teach the model. Under `half_life=float("inf")` no row's
weight decays, so `weight_sum` counts the rows learned before each row.
This code uses `ticks.parquet` from [Example data](#example-data):

```python
learn_from_x = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"],
                               clock="t", half_life=float("inf"), gap_cap=300.0, weight="w")
fitted = (
    pl.scan_parquet("ticks.parquet")
    .with_columns(pl.when(pl.col("venue") == "X").then(1.0).otherwise(0.0).alias("w"))  # learn from venue X only
    .online.fit_predict([learn_from_x])
    .collect()
)
scored = fitted["ridge"].struct.field("pred_y").is_not_null().sum()
print(f"rows scored: {scored} of {fitted.height}; weight_sum on the last row: {fitted['ridge'].struct.field('weight_sum')[-1]:.0f}")
```

```text
rows scored: 397 of 400; weight_sum on the last row: 200
```

**`lf.online.fit_predict` checks each spec against the query's columns when
it is called, so a missing or non-numeric column raises `ValueError` before
any row is read.** The bank refuses what only the values reveal, such as a
null clock or a step back, while the query runs. That error arrives as
`polars.exceptions.ComputeError` under py-polars 1.x, and as the bank's own
`ValueError` under 2.0, since how a query wraps an error is Polars' to
change. Either way its message names the row and the way out.
This code uses `ticks.parquet` from [Example data](#example-data):

```python
misspelt = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x9"], half_life=100.0)
try:
    pl.scan_parquet("ticks.parquet").online.fit_predict([misspelt])   # refused here: the query never runs
except ValueError as e:
    print(e)
```

```text
not found: spec "ridge": feature column "x9" not found; the input has columns ["t", "ts", "x0", "x1", "x2", "signal_a", "signal_b", "y", "ret", "stock_id", "group", "session", "venue"]
```

**With a type checker, call `po.fit_predict`,
[`po.predict`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.predict)
and [`po.unnest`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.unnest)
in place of the `.online` methods.** The package adds those methods to
Polars' frames when it is imported, so a type checker cannot see them. Each
function takes the frame first.
This code uses `lf` and `later` from [Example data](#example-data):

```python
spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], clock="t", half_life=600.0, gap_cap=300.0)
fitted = po.fit_predict(lf, [spec], save_state="ridge.state")   # lf.online.fit_predict([spec], save_state="ridge.state")
columns = po.unnest(fitted, [spec]).collect()                   # fitted.online.unnest([spec]); running it saves the state
scored = po.predict(later, "ridge.state").collect()             # later.online.predict("ridge.state")
```

### In a loop: `ModelBank`

To feed a bank from your own Python loop, build a
[`ModelBank`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.ModelBank)
from the specs, pass it each chunk with `bank.fit_predict(chunk)`, and ask
it what it holds between chunks ([What a bank holds](#what-a-bank-holds)).
`bank.fit_predict_batches(query, chunk_size=...)` runs that loop for you,
reading a query or a `DataFrame` in chunks of the size given, and yields
each chunk's output. Here the bank reads 100 rows at a time, and stock
`b0`'s coefficient on `x0` settles near the 2 that `df` plants.
This code uses `ticks.parquet` from [Example data](#example-data):

```python
spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"],
                       clock="t", half_life=600.0, gap_cap=300.0, group="stock_id")
bank = po.ModelBank([spec])                                # nothing learned yet
for out in bank.fit_predict_batches(pl.scan_parquet("ticks.parquet"), chunk_size=100):
    # out: the chunk's columns, plus one column per spec. Between chunks, the bank answers questions:
    x0 = bank.coef().filter(pl.col("group") == "b0", pl.col("term") == "x0")["coef"][0]
    print(f"after {bank.rows_fed()} rows: b0's coefficient on x0 is {x0:.3f}")
bank.save("bank.state")                                    # written whole or not at all: a temporary file, then a rename

# With chunks of your own: for chunk in pl.scan_parquet("ticks.parquet").collect_batches(): bank.fit_predict(chunk)
# A run whose product is the state: po.ModelBank([spec]).fit(query) learns from every row and keeps no output.
```

```text
after 100 rows: b0's coefficient on x0 is 2.105
after 200 rows: b0's coefficient on x0 is 2.055
after 300 rows: b0's coefficient on x0 is 2.087
after 400 rows: b0's coefficient on x0 is 2.023
```

**Feed a bank from one thread at a time, though `predict`, which learns
nothing, may run from any number of threads at once.** A bank reads one
ordered stream, so a call that finds it busy on another thread raises
`RuntimeError`: wait for that call to return, or give each thread its own
bank.

**A bank keeps every group until you drop it, so in a long-running bank,
drop the quiet ones with `bank.drop_groups`.** `bank.groups()` gives each
group's `last_clock` in the clock column's own dtype: a `Datetime` in its
unit and zone, an integer clock in its own width, a float clock as a
`Float64`. Compare it with the current time of the same kind. A dropped
group starts over, with nothing learned, if it appears again. In `df` the
four stocks take turns, so each last ticked at a different `t`.
This code uses `df` and `ticks.parquet` from [Example data](#example-data):

```python
spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"],
                       clock="t", half_life=600.0, gap_cap=300.0, group="stock_id")
bank = po.ModelBank([spec])
bank.fit(pl.scan_parquet("ticks.parquet"))
for group, rows, last_clock in bank.groups().select("group", "rows_processed", "last_clock").iter_rows():
    print(f"{group}: {rows} rows, the last at t = {last_clock:.0f}")
now = df["t"].max()                                        # the current time on the clock t; on a temporal clock, a datetime
stale = bank.groups().filter(pl.col("last_clock") < now - 2)   # quiet for more than 2 units of t
print(f"dropped {bank.drop_groups(stale['group'])}; left: {bank.groups()['group'].to_list()}")
```

```text
b0: 100 rows, the last at t = 396
b1: 100 rows, the last at t = 397
b2: 100 rows, the last at t = 398
b3: 100 rows, the last at t = 399
dropped 1; left: ['b1', 'b2', 'b3']
```

### Running a bank as a job, without Python

For a scheduled job, or a deployment with no Python, run the same bank from
a file to a file with the standalone `online` command line, a binary on
each GitHub release. It reads the input, the output and the specs from a
TOML file. Each `[[specs]]` table takes the keywords every spec shares, and
its `[specs.model]` table names the model and takes the model's own. This
`bank.toml` is the first recipe's spec of [As a
query](#as-a-query-lfonlinefit_predict):

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
type = "ewridge"
ridge = [1e-6, 0.1]
standardize = true
```

Run it from the directory that holds `bank.toml` and the example data's
`ticks.parquet`. The predictions, residuals and weights it writes are the
bank's in Python, bit for bit:

```sh
online --config bank.toml
```

```text
400 rows in 1 chunks
wrote 400 rows (1 chunks) to fitted.parquet
saved state to bank.state
```

The first line is the progress, which `--quiet` turns off. After `wrote`,
a run prints a line for each spec that ended with groups not ready, and
nothing for the others. Such a line reads
`spec "ridge": 2 groups whose last row was withheld (below_min_weight 2)`.
[docs/RUNNER.md](docs/RUNNER.md) has every key and flag, with a recipe for
each kind of run: a check before the run, resuming, scoring, a run that
writes only its state, and the groups that close.

### Output as Arrow

To hand a bank's output to a consumer that is not Polars, call
`bank.fit_predict_arrow(df)` or `bank.predict_arrow(df)` in place of
`fit_predict` or `predict`. Each returns one `ArrowStruct` per spec, which
holds that spec's output through the Arrow PyCapsule interface, an Arrow
specification, so any reader of `__arrow_c_array__` takes it directly. Its
values are `fit_predict`'s, field for field and null for null. The input
still goes in as a Polars `DataFrame`. `fit_predict` itself hands its
output over as a Polars `Series`, on py-polars' private methods, so this
package measures a Polars range and promises none ([Which interfaces carry
a promise](#which-interfaces-carry-a-promise)).

**Read each struct once.** Exporting hands its buffers to the consumer, so
a second read raises `ValueError`.
This code uses `df` and `today` from [Example data](#example-data):

```python
spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], clock="t", half_life=600.0, gap_cap=300.0)
bank = po.ModelBank([spec])
structs = bank.fit_predict_arrow(df)                    # one per spec, as Arrow
ridge = pl.Series(structs[0])                           # Polars as the reader here: any reader of __arrow_c_array__ takes it
print(f"{len(structs)} {type(structs[0]).__name__}; read as a Series: {ridge.name}, {ridge.len()} rows")
print(f"the same as fit_predict's column: {ridge.equals(po.ModelBank([spec]).fit_predict(df)['ridge'])}")
try:
    pl.Series(structs[0])                               # the second read
except ValueError as e:
    print(e)
scored_structs = bank.predict_arrow(today)              # the same for predict, on today: the rows after df's
```

```text
1 ArrowStruct; read as a Series: ridge, 400 rows
the same as fit_predict's column: True
this ArrowStruct has already been exported: the Arrow PyCapsule interface hands its buffers to the consumer, so it can be read once
```

Here is how two consumers take a struct, as measured:

| consumer | measured on | takes |
|---|---|---|
| pyarrow | pyarrow 25.0.1 | a struct as it is: `pa.array(s)` and `pa.table(s)` each take one, values and types unchanged |
| DuckDB | duckdb 1.5.5 | `pl.Series(s)`. It refuses an `ArrowStruct`, because `__arrow_c_stream__` is the method it looks for. A spec's output arrives table-shaped, one column per field |

## Saving, loading and serving

Save a bank's state (everything it has learned, for every spec and group)
to one file, and load it back to keep learning or to score new rows
without learning. Or export it as JSON. The three uses:

| use | the call | what holds |
|---|---|---|
| resume | `po.ModelBank.load(path)`, or `load_state=` on a query | the next row is learned as one run would learn it |
| serve | `bank.predict(frame)`, or `lf.online.predict(path)` | each row is scored as `fit_predict` would score it as the next row, and the state never moves |
| read elsewhere | `bank.to_json()` | everything `save` writes, as JSON |

The recipe fits `df`, saves the bank, and loads it back. Served, the first
row of `today` gets the prediction one run over `df` and `today` gives it.
Resumed on input that overlaps the state, through `skip_learned`, the bank
learns each row once and gives one run's predictions on every row.
This code uses `df` and `today` from [Example data](#example-data):

```python
spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], clock="t", half_life=600.0, gap_cap=300.0)
one_run = po.ModelBank([spec]).fit_predict(pl.concat([df, today]))   # df and today as one stream, for comparison

bank = po.ModelBank([spec])
bank.fit_predict(df)                                     # the stream so far
bank.save("bank.state")
loaded = po.ModelBank.load("bank.state", specs=[spec])   # specs= checks the file holds this model, not another

served = loaded.predict(today)                           # score today's rows, learning nothing
pred = lambda out: out["ridge"].struct.field("pred_y")
print(f"served, today's first row: {pred(served)[0]:.6f}; one run: {pred(one_run)[400]:.6f}")

rerun = pl.concat([df.tail(50), today])                  # input that overlaps the state by 50 rows
kept = loaded.skip_learned(rerun)                        # the rows the state has not learned
resumed = loaded.fit_predict(kept)
print(f"kept {kept.height} of {rerun.height} rows; largest difference from one run: {(pred(resumed) - pred(one_run.tail(400))).abs().max()}")
```

```text
served, today's first row: 1.603543; one run: 1.603543
kept 400 of 450 rows; largest difference from one run: 0.0
```

A state saved by 0.13.0 or an earlier release does not load in the
releases after it: refit the bank from its input. From 1.0, a state a 1.x
build saved loads in every later 1.x ([This package's own
versioning](#this-packages-own-versioning)). `with_windows` keeps a state
of its own ([Saving and resuming a window
run](docs/WINDOWS.md#saving-and-resuming-a-window-run)).
[docs/STATE-WORKFLOW.md](docs/STATE-WORKFLOW.md) says what each step of
the workflow guarantees and what a saved bank carries, each with a
recipe.

### Save and load

Each way of running a bank saves and loads the same file, written whole or
not at all, and the same to the byte whichever way wrote it. Each writes
the file at its own moment:

| run by | saves with | loads with | the file is written |
|---|---|---|---|
| a bank object | `bank.save(path)` | `po.ModelBank.load` | when `bank.save` is called |
| a query | `save_state=` | `load_state=` | when the run reaches its last row, with the same bytes a `ModelBank` would write |
| the command line | `--save-state` | `--load-state` | only after its output is committed |

This code uses `df` and `ticks.parquet` from [Example data](#example-data):

```python
from pathlib import Path

spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], clock="t", half_life=600.0, gap_cap=300.0)
bank = po.ModelBank([spec])
bank.fit_predict(df)
bank.save("from_bank.state")                             # from a bank object
(pl.scan_parquet("ticks.parquet")                        # the same rows, from a query
   .online.fit_predict([spec], save_state="from_query.state")
   .sink_parquet("fitted.parquet"))
print(f"the same bytes: {Path('from_bank.state').read_bytes() == Path('from_query.state').read_bytes()}")
blob = bank.save_bytes()                                 # the same file as bytes, for a checkpoint that is not a file
print(f"as bytes too: {blob == Path('from_bank.state').read_bytes()}")
```

```text
the same bytes: True
as bytes too: True
```

A query loads and saves with `load_state=` and `save_state=`, and the
two may name one file. This code uses `lf` and `later` from [Example data](#example-data):

```python
spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], clock="t", half_life=600.0, gap_cap=300.0)
lf.online.fit_predict([spec], save_state="bank.state").sink_parquet("fitted.parquet")             # fit, then save at the last row
served = later.online.predict("bank.state").collect()                                             # serve from the file
later.online.fit_predict(load_state="bank.state", save_state="bank.state").sink_parquet("more.parquet")  # the next rows: continue, save again
bank = po.ModelBank.load_bytes(po.ModelBank.load("bank.state").save_bytes(), specs=[spec])         # bytes load as a file does
```

**To resume on input that overlaps the state, pass it through
`bank.skip_learned(frame)` before the bank.** Without it, the overlap steps
every group's clock back, which the bank refuses unless
`restart_after_step_back` reads it as a new start. `skip_learned` keeps the
rows after each group's last clock (a row at that clock counts as learned)
and every row of a group the bank has not seen. It keeps a `LazyFrame`
lazy. For a bank whose specs all count rows, it raises `ValueError`; drop
the learned rows with `frame.slice(bank.rows_fed())` instead, when the
input starts where the saved one did.

**A query abandoned before its last row, or ended by an error inside the
bank, leaves the `save_state` file as it was.** An error in a step after
the bank, such as a full disk under `sink_parquet`, does not stop the bank
on Polars 1.x. The state is then written although the output is missing
([docs/STATE-WORKFLOW.md](docs/STATE-WORKFLOW.md#the-one-gap-a-failure-after-the-bank)
has the measurements). To write the state only with the output, use the
command line, which saves it after committing the output. In a query, give
each batch of data its own dated `save_state` file, so a rerun loads the
previous batch's state.
This code uses `lf` and `later` from [Example data](#example-data):

```python
spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], clock="t", half_life=600.0, gap_cap=300.0)
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
| was saved by 0.13.0 or an earlier release | `ValueError`, naming the range | refit the bank from its input. `po.schema_version()` gives the installed version's schema |

This code uses `df` and `ticks.parquet` from [Example data](#example-data):

```python
spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], clock="t", half_life=600.0, gap_cap=300.0)
bank = po.ModelBank([spec])
bank.fit_predict(df)
bank.save("bank.state")
other = po.spec.ewridge("ridge", targets=["y"], features=["x0"], clock="t", half_life=600.0, gap_cap=300.0)
for path, specs in [("nowhere.state", [spec]), ("bank.state", [other]), ("ticks.parquet", None)]:
    try:
        po.ModelBank.load(path, specs=specs)
    except (FileNotFoundError, ValueError) as e:
        print(f"{type(e).__name__}: {e}")
```

```text
FileNotFoundError: [Errno 2] No such file or directory: 'nowhere.state'
ValueError: saved specs do not match the bank's specs: spec "ridge" (1 of 1) differs at features: ["x0","x1"] in the file, ["x0"] in the bank; refusing to load
ValueError: not a polars-online bank state file (invalid type: integer `80`, expected struct BankHeader)
```

### Serving without learning

To score rows without learning from them, call `bank.predict(frame)` on a
bank object, or `lf.online.predict(...)` in a query. Every row is scored
from the same state, which `predict` never moves. Since it skips each
row's update, `ewridge` scores 1.8 times as fast as it learns at 5
features, and 2.9 times at 20 ([docs/PERFORMANCE.md](docs/PERFORMANCE.md)
§9). A query takes the bank object, read as it stands when the query runs,
or a path, read when the query is built.
This code uses `df`, `today` and `later` from [Example data](#example-data):

```python
spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], clock="t", half_life=600.0, gap_cap=300.0)
bank = po.ModelBank([spec])
bank.fit_predict(df)
bank.save("bank.state")
scored = bank.predict(today)                             # every row scored from the same state: nothing moves
served = later.online.predict(bank).collect()            # a bank, as it stands when the query runs
from_file = later.online.predict("bank.state").collect() # a path, read when the query is built
weights = scored["ridge"].struct.field("weight_sum")
print(f"weight_sum on every row: {weights.min():.4f} to {weights.max():.4f}; the three agree: "
      f"{scored['ridge'].equals(served['ridge']) and served['ridge'].equals(from_file['ridge'])}")
```

```text
weight_sum on every row: 320.4975 to 320.4975; the three agree: True
```

**Row *i* of `scored` carries what `fit_predict` would have reported had it
been the next row of its group's stream:** `pred`, `weight_sum`, `sigma`,
`zscore`, the selection and the metrics, field for field
([docs/DIAGNOSTICS.md](docs/DIAGNOSTICS.md)). Two fields differ: `drift` never
fires, and `coef` is filled on each group's last accepted row only, since
the same coefficients score every row. How `predict` reads a row:

| the row | `predict` |
|---|---|
| its target column | may be absent; `resid` is then null |
| its weight column | is not read |
| of a group the bank has never seen | scores null |
| before the last clock its group learned | is scored against the state as it stands, with a clock step of 0: never refused, never a restart |
| where `session_gap="reset"` or `group_close="session"` would restart its group | scores null, with `weight_sum` 0 |
| after rows an `embargo` still holds | sees the state as it stands, since `predict` releases nothing: a held row whose delay has passed by the scored row's clock stays held until the next `fit_predict` learns it, so `weight_sum` and `learned_clock` are the state's on every row |

### Reading a state without this library

To read a state from a program without this library, export it with
`bank.to_json()`, or write it to a file with `bank.save_json(path)`. Keep
the binary file for resuming, since `load` reads only that form. Every
export is read back and checked against the state before you get it, and
one that does not match raises `ValueError`.

**A program reading the JSON must turn the strings `"nan"`, `"inf"` and
`"-inf"` back into numbers.** JSON has no literal for `NaN` or `±inf`, and
an ordinary state can hold them, as a spec with `half_life=inf` (no
forgetting) does. The strings are the spelling a spec's `half_life` takes.
This code uses `df` from [Example data](#example-data):

```python
import json

bank = po.ModelBank([po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], half_life=float("inf"))])
bank.fit_predict(df)
text = bank.to_json()                     # everything save() writes, as JSON
bank.save_json("bank.json")               # the same, to a file
state = json.loads(text)                  # any JSON reader
half_life = state["specs"][0]["half_life"]
print(f"{state['magic']}: half_life {half_life!r}, as a number {float(half_life)}")
```

```text
polars-online-bank: half_life 'inf', as a number inf
```

## Reading the fit

Read a fit from the bank, with its methods, between chunks or after the
run, or from the output, row by row. The bank reports each group as it
stood after the last row it learned from.

| to read | call | subsection |
|---|---|---|
| what a bank holds, what it was fed, and what is wrong with the data | `repr(bank)`, `bank.groups()`, `bank.summary()`, `bank.describe()`, `bank.last_row()`, `bank.check()`, `bank.audit()` | [What a bank holds](#what-a-bank-holds) |
| any field of the output, by name | `po.spec.output_index`, `po.spec.coef_fields` | [Output field names](#output-field-names) |
| the coefficients, at the end or row by row | `bank.coef()`, or `coef_every=0` and `.online.unnest` | [Coefficients](#coefficients) |
| the running sums a fit is solved from, and the algebra on them | `bank.gram(spec)`, `po.gram` | [The running sums behind a fit](#the-running-sums-behind-a-fit) |
| a finished group's running sums | `bank.closed_groups()` | [One row per finished group](#one-row-per-finished-group) |
| what a correlation matrix says, from an array, a `gram()` dict or a closed row | `po.corr` | [Reading a correlation matrix](#reading-a-correlation-matrix) |

### What a bank holds

A bank describes itself without its data, live or loaded from its file.
Six calls say what it holds, and four tables say how each fit is doing and
what it was fed.

```python
bank = po.ModelBank.load("bank.state")   # no specs=: the file is enough

repr(bank)                # ModelBank(['ridge'], groups=4, rows_fed=400)
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
bank.rows_fed()           # rows fed, over every chunk and group
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
problems = bank.check()           # one row per problem found in the data, the worst first

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
the metrics and the interval when the spec asks for them
([docs/DIAGNOSTICS.md](docs/DIAGNOSTICS.md)), `weight_sum`, and `coef` when that row
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
| `weight_sum`, `clock_min`, `clock_max`, `last_clock` | the weight behind the state, and the clock's range and last value, in the clock column's own dtype |
| `session_changes`, `clock_backwards`, `resets` | what the clock rules met |
| `settled_frac`, `weight_sum_settled`, `error_inflation`, `min_support_coef` and the feature it belongs to, `n_coef` | the warm-up readings after the last row; `weight_sum_settled` is the weight the stream settles at, which says whether a `min_weight` can be met |

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
                      coef_every=0)         # write coef on every row but a skipped one

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

**`coef_every=0` writes `coef` on every row but a skipped one**, as a list
of one float per term. Without a cadence the bank writes it on each
group's last accepted row in a chunk, and on none for a group whose rows
there were all skipped. A number of clock units, or a duration such as
`"5m"` on a temporal clock, writes it once the clock has moved that far
since the group's last `coef` row, as `solve_every` schedules a solve.
`max_rows_between_coefs` writes it after that many rows, and with both,
whichever comes first. A cadence reads only the clock and the rows, so the
rows that carry `coef` do not depend on the chunking.

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
[Wide fits from Grams](docs/GRAM.md) is the guide to fitting from Grams,
with the theory and a recipe that runs for each task: a Gram per week
from one pass, pooling weeks with or without their weekly levels, a ridge
with its t, the lasso path, stability selection and its null by sign
flips, compact exports and `gram_threads`.

**`po.gram` does the same algebra in one call**, on any Gram, such as one
read from a saved state. Two calls need care:

| limit | what to do |
|---|---|
| `po.gram.solve` solves a plain ridge | read a `ridge_scale="sum"` or `coef_prior` fit from `bank.coef()`, since `solve` does not reproduce it |
| `po.gram.merge` pools parts that share a weighting, such as one Gram per group or per shard of a pass | before merging two halves of a decayed stream, multiply the earlier half's `weight_sum` and `target_weights` by `0.5 ** (dt / half_life)`, with `dt` the clock from its last row to the later half's last row. Halves run with `half_life=float("inf")` merge as they are |

```python
g = po.ModelBank.load("bank.state").gram("ridge")[0]   # the first group's Gram, from a saved state
# a quarter of the bytes: comoments in float32 and as its upper triangle; po.gram reads it in float64
small = po.ModelBank.load("bank.state").gram("ridge", dtype="float32", layout="packed")[0]

# residual variance, R², standard errors and t:
r2 = po.gram.coef_stats(g, po.gram.solve(g, ridge=1e-9))["r2"]
# the model's own ridge, in original units; a list of ridges is one eigendecomposition:
ridges = po.gram.solve(g, ridge=[0.0, 0.01, 0.1, 1.0], standardize=True)
worst = po.gram.condition(g)["kappa"]   # Belsley's condition indexes and variance-decomposition proportions
po.gram.correlation(g)                  # the correlation matrix
po.gram.vif(g)                          # variance inflation factors
po.gram.subset(g, ["x0", "x1"])         # the Gram of some of the columns: a sub-block, not a recomputation
po.gram.merge([g, g])                   # pools the Grams of disjoint row sets into the Gram of their union
po.gram.merge([g, g], within=True)      # pools each part about its own means: a fit with an intercept per part
po.gram.lasso_path(g, [0.1, 0.01])      # the lasso model's coordinate descent, offline
po.gram.lars_path(g, max_active=2)      # the lasso path's knots, by least angle regression, stopped early
po.gram.lars_paths([g, g], max_steps=5) # many Grams and targets at once, on the bank's thread pool
po.gram.solve_subsets([g, g], [[0], [0, 1]], ridge=0.1)   # merge each subset, then its ridge, se and t
po.gram.solve_subsets([g, g], [[0, 1]], within=True)      # each subset pooled within its parts
po.gram.solve_subsets([g, g], [[0, 1]], within=True, target_signs=[[1.0], [-1.0]])   # a run with the second Gram's target negated
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
goes in, and one row per block comes out. Here `by_block` is saved to
disk, `blocks` learns from the file as it is read back, and each closed
block's row goes to `blocks.parquet`:

```python
by_block.write_parquet("by_block.parquet")                                             # the stream, on disk
po.ModelBank([blocks]).fit(pl.scan_parquet("by_block.parquet"), closed_groups="blocks.parquet")   # read back as a query
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
| one number | `equicorr_row` | the equicorrelation estimate of one standardised row, the `u` [`deco`](docs/MODELS.md#deco--one-correlation-for-the-whole-matrix) computes per row |
| one number | `equicorr_loglik` | the Gaussian log-density of a standardised row under an equicorrelation matrix |
| sampling error | `signal_share` | how much of a correlation's movement between blocks is not sampling noise, at Kish's sample size |
| blocks | `block_means`, `from_blocks` | the mean correlation within and between labelled blocks, and the block matrix back from those means |
| scale | `epps_invert` | the correlation at a coarser scale, from `ew_cov`'s lagged co-moments |
| forecast | `loss` | how wrong a forecast correlation matrix was: `qlike`, `z_mse` or Engle–Colacito `minvar` |

## Checking the data and the model

### Checking your data's quality

**`check()` reads these tables, and the Gram where a spec keeps one, and
says what is wrong with the data.** Each row is a finding: its `severity`
(`error`, `warning` or `info`), a `code` such as `missing`, `constant`,
`collinear` or `leakage`, the `spec`, `group` and `column`, the measured
`value` beside its `threshold`, and a `message` that says what to do. It
costs the run nothing, so call it after a first pass and before trusting a
fit. Every threshold was measured: on five clean shapes over ten seeds no
check raised an error or a warning, and each planted problem was found on
every seed. The docstring of `ModelBank.check` lists the checks with their
thresholds. A bank keeps no count of sentinel values, frozen feeds,
duplicated columns or the clock's gaps; an [`audit`](docs/MODELS.md#audit--what-a-streams-columns-hold)
spec does, and `check()` reads it. Each message ends with a link to the
section of [docs/DATA-QUALITY.md](docs/DATA-QUALITY.md) on its `code`. That
section shows what the problem does to each model and how to fix it with
Polars before the bank, in a program that runs, with the theory behind it.

**Run it beside the models in the same pass** (`po.ModelBank([audit,
model])`) to check the data a fit was made from, at the cost of one more
spec. `check()` lists an audit's findings with the models' own, and the
docstring of `ModelBank.check` gives each threshold with the rate at which
it found a planted problem and stayed silent on clean data.
[docs/DATA-QUALITY.md](docs/DATA-QUALITY.md) has a recipe for each finding:
the problem planted, the model left alone, the fix in Polars, and the model
after it.

### Checking that a model works

Judge a model as the stream runs, by switching diagnostics on in its spec,
or after the run, by passing its output to `po.eval`. A spec's diagnostics
and selection measure each row against what the models learned before it,
never against the row's own outcome. Their memory does not grow with the
stream. Every switch but `emit_clocks` and `emit_feature_health` reads the
residuals, and feature health reads the features a fit's coefficients lean
on, so only the ten [linear models](docs/MODELS.md#linear-models), which predict a
target, take them, and any other model refuses them by name. A switch adds its fields to the
spec's output, one per *slot*: one prediction of one target, at one point
of a grid.

[docs/DIAGNOSTICS.md](docs/DIAGNOSTICS.md) answers "is this model
working?" one question at a time. For each diagnostic it gives the
definition and its update, the threshold that means something with its
false-alarm rate as measured, its cost a row, the theory it rests on, and a
recipe that runs with its output shown:

| question | switch or call | fields, per slot `<t>` |
|---|---|---|
| [is it scaled right?](docs/DIAGNOSTICS.md#is-it-scaled-right) | `emit_calibration` | `calibration_slope_<t>`, `calibration_intercept_<t>`, `calibration_wald_<t>` |
| [has it broken?](docs/DIAGNOSTICS.md#has-it-broken) | `emit_breaks`, `emit_drift` | `studentized_<t>`, `cusum_<t>`, `cusum_sq_<t>`, `break_wald_<t>`, `drift_<t>` |
| [can its t be trusted?](docs/DIAGNOSTICS.md#can-its-t-be-trusted) | `emit_se_coef`, `emit_robust_se` | `se_coef`, `se_coef_hc0`, `se_coef_hac`, on `coef`'s rows ([Coefficients](#coefficients)) |
| [what is it missing?](docs/DIAGNOSTICS.md#what-is-it-missing) | `emit_specification`, `emit_autocorr` | `ljung_box_<t>`, `breusch_pagan_<t>`, `reset_<t>`, `autocorr_<t>` |
| [are its tails heavy?](docs/DIAGNOSTICS.md#are-its-tails-heavy) | `emit_tails` | `skew_<t>`, `kurtosis_<t>`, `jarque_bera_<t>`; `drift_threshold` is set against the kurtosis |
| [which row moved it?](docs/DIAGNOSTICS.md#which-row-moved-it) | `emit_influence` | `influence_<t>`: `ewridge`, `rls` and `kalman` |
| [are its features healthy?](docs/DIAGNOSTICS.md#are-its-features-healthy) | `emit_feature_health` | `spread_ratio_<feature>`, `mean_shift_<feature>`, once per instance |
| [is another model better?](docs/DIAGNOSTICS.md#is-another-model-better) | `emit_metrics`, `emit_selected`, `emit_averaged`; `po.eval.metrics`, `window_metrics`, `compare_specs`, `diebold_mariano`, `clark_west`, `seqtest`, and `po.eval.sums` for an output too large to hold | `ic_<t>`, `r2_<t>`, `hit_rate_<t>`; `selected_<t>`, `pred_<t>__selected`, `pred_<t>__averaged`, one per target |
| [is it ready?](docs/DIAGNOSTICS.md#is-it-ready) | `min_weight`, `min_settled_frac`, `max_error_inflation`, `emit_error_inflation` ([Warm-up](#warm-up)) | `weight_sum`, `settled_frac`, `withheld_reason`, `error_inflation_<t>` |
| [how wide is its error?](docs/DIAGNOSTICS.md#how-wide-is-its-error) | `emit_sigma`, `emit_zscore`, `resid_quantiles`, `conformal` | `sigma_<t>`, `zscore_<t>`, `abs_resid_q<p>_<t>`, `lo_<t>`, `hi_<t>`, `coverage_<t>` |
| when was a row scored, and from what? | `emit_clocks`, on every model | `scored_clock`, `learned_clock`, once per spec ([Labels that arrive late](#labels-that-arrive-late)) |
| [does a model find a truth it was given?](docs/DIAGNOSTICS.md#data-whose-truth-is-known) | `po.sim.regimes` | a stream whose regimes are known, and the truth beside it |

## Models

polars-online has twenty-two models in four families. A spec's clock,
grouping and warm-up mean the same whichever model it names, and its
half-life is always a half-life on the clock ([How a bank sees a
stream](#how-a-bank-sees-a-stream)). To look up a model's keywords and
their defaults, open its builder's reference page, such as the one for
`po.spec.ewridge`. [docs/OUTPUTS.md](docs/OUTPUTS.md) lists every field a
model writes, and [docs/TESTING.md](docs/TESTING.md) what each is checked
against.

| model | learns by | what it is |
|---|---|---|
| **[Linear models](docs/MODELS.md#linear-models)** | | predict a numeric target, from features or from its own past |
| [`ewridge`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.ewridge) · [math](docs/MODELS.md#ewridge--ew-ridge-on-sufficient-statistics) | solve | exponentially weighted ridge regression, the workhorse; several ridge values and feature sets are solved from the same running sums at almost no extra work |
| [`rls`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.rls) · [math](docs/MODELS.md#rls--recursive-least-squares) | solve | recursive least squares, in the numerically safe square-root form |
| [`lasso`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.lasso) · [math](docs/MODELS.md#lasso--lasso-path-its-penalty-chosen-as-it-runs) | solve | lasso and elastic-net path, with the penalty chosen as the stream runs |
| [`kalman`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.kalman) · [math](docs/MODELS.md#kalman--random-walk-β-dynamic-linear-model) | filter | a Kalman filter whose coefficients drift as random walks |
| [`huber`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.huber) · [`quantile`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.quantile) · [math](docs/MODELS.md#huber--quantile--robust-regression) | reweight | robust and quantile regression |
| [`sgd`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.sgd) · [math](docs/MODELS.md#sgd--stochastic-gradient-descent) | step | stochastic gradient descent with squared, Huber, quantile, ε-insensitive, Poisson and logistic losses |
| [`pa`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.pa) · [math](docs/MODELS.md#pa--passive-aggressive-regression) | step | passive-aggressive regression, with no learning rate to tune |
| [`ftrl`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.ftrl) · [math](docs/MODELS.md#ftrl--online-logistic-regression) | step | FTRL-proximal regression, logistic by default, with an L1 penalty that zeroes coefficients |
| [`holt`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.holt) · [math](docs/MODELS.md#holt--holts-linear-trend) | step | Holt's linear trend, the baseline that uses no features |
| **[Moments and correlation](docs/MODELS.md#moments-and-correlation)** | | track the running moments and correlations of the columns you name |
| [`ew_cov`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.ew_cov) · [math](docs/MODELS.md#ew_cov--exponentially-weighted-moments) | accumulate | running mean, variance, covariance, correlation, partial correlation, Mahalanobis distance and principal components |
| [`marginal`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.marginal) · [math](docs/MODELS.md#marginal--every-pairs-moments-kept-in-the-state) | accumulate | every (feature, target) pair's running mean, variance, covariance, correlation, slope and t, for a wide set of columns, kept in the state and read back as a table; optionally at a set of lags, and with binned target moments for the relations a correlation cannot see |
| [`deco`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.deco) · [math](docs/MODELS.md#deco--one-correlation-for-the-whole-matrix) | accumulate | one correlation for the whole matrix: Engle & Kelly's equicorrelation, or one per block and per pair of blocks |
| [`rcov`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.rcov) · [math](docs/MODELS.md#rcov--a-blocks-realised-covariance-robust-to-noise) | accumulate | a block's realised covariance, robust to microstructure noise: the Barndorff-Nielsen–Hansen–Lunde–Shephard kernel or Christensen–Kinnebrock–Podolskij pre-averaging, reported when a group closes |
| [`audit`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.audit) · [math](docs/MODELS.md#audit--what-a-streams-columns-hold) | accumulate | what each column holds, counted in one pass: nulls, NaNs and infinities apart, repeated values, runs, persistence, tails and duplicated columns, and the clock's duplicate stamps and gaps, read back as a table and by `bank.check()` |
| **[Clustering and classification](docs/MODELS.md#clustering-and-classification)** | | put labels on rows |
| [`kmeans`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.kmeans) · [math](docs/MODELS.md#kmeans--exponentially-weighted-k-means) | step | exponentially weighted k-means: cluster labels assigned before the row is learned from, with a split–merge move that finds a cluster born after seeding |
| [`micro`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.micro) · [math](docs/MODELS.md#micro--density-based-clustering-any-shape) | step | density-based clustering: DenStream micro-clusters linked into clusters of any shape and number, flagging the rows that belong to none |
| [`ew_class`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.ew_class) · [math](docs/MODELS.md#ew_class--gaussian-classification-on-ew_cov-moments) | accumulate | Gaussian classification, QDA, LDA or naive Bayes, with one set of running moments per class: a label column in, class probabilities out |
| **[Sequential tests and regimes](docs/MODELS.md#sequential-tests-and-regimes)** | | give evidence that something holds or has changed, and which regime the stream is in |
| [`seqtest`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.seqtest) · [math](docs/MODELS.md#seqtest--a-sequential-test-of-a-sign-by-betting) | test | a sequential test of a sign by betting, giving evidence you can read at any row: on its own, of a column's sign; with `a` and `b`, of whether one spec of the bank predicts closer than another |
| [`corrchange`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.corrchange) · [math](docs/MODELS.md#corrchange--has-the-correlation-structure-changed) | test | has the correlation structure changed: the Wied–Krämer–Dehling constancy test span by span, the Wied–Galeano detector row by row against a stable history, or the size of a change between two windows against a permutation null |
| [`bocpd`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.bocpd) · [math](docs/MODELS.md#bocpd--how-long-has-this-regime-lasted) | filter | how long this regime has lasted: Adams & MacKay's run-length posterior, so the answer is the regime's age and not a flag |
| [`hmm`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.hmm) · [math](docs/MODELS.md#hmm--which-regime-are-we-in) | filter | a Gaussian hidden Markov model, filtered as the stream runs: `ew_class` without the labels, with a transition matrix that can be learned |

How a model learns from a row decides whether, with decay off, its answer
depends on the row order:

| learns by | what the model does with a row | with decay off | order-dependent even so |
|---|---|---|---|
| **solve** | keeps running sums and computes its coefficients from them | converges to the batch answer, in any row order | `lasso`'s `penalty_selected`, ranked by out-of-sample error, though its path converges in any order |
| **accumulate** | keeps running sums and reports them | converges to the batch answer, in any row order | a lag, in `ew_cov` or `marginal`, which counts learned rows; `rcov`'s block and `deco`'s per-row estimate; `audit`'s runs, smallest change, lag pairs and clock steps, and past `distinct_cap` the digest's `median`, `mad` and `robust_z` and the counters' `top_value`, `top_count`, `second_count` and `equal_by_chance`, each within its bound |
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

### Choosing a model

Each model's entry, with when to reach for it, its update rule, its
parameters and its checks, is in [docs/MODELS.md](docs/MODELS.md), under
the four families of the table above.

### Detecting changes and regimes

The sequential tests and the regime models are under [Sequential tests and
regimes](docs/MODELS.md#sequential-tests-and-regimes), and what each finds
on data whose truth is known is in [docs/REGIMES.md](docs/REGIMES.md).

## Performance

A bank spends most of its time on the models' own arithmetic. Its memory
holds their state, a few chunks in flight, and what Polars has read ahead
of the bank. The thread counts and `chunk_size` change only speed and
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
| runs | best of 3, 200k rows per run, measured on 2026-09-29 at commit `5e96018` with `uv run python scripts/benchmark.py --markdown`, beside one other process on about one core, with no load average recorded; run 2 of [PERFORMANCE §28](docs/PERFORMANCE.md#28-the-readmes-numbers-re-measured-2026-09-29) |
| `ewridge` and `rls` | re-measured the same day at commit `d0f8616`, once `ewridge`'s solve had been made cheaper, beside a load average of 3.2 to 3.5 ([§30](docs/PERFORMANCE.md#30-where-every-row-solves-2026-09-29)) |
| noise | a run moves by up to about 11% from the last on this machine, so read a gap smaller than that as noise |
| half-life | 1,000, in every row that gives none of its own |
| `k`, `K` | the number of features; the number of clusters or hidden states |

The [linear models](docs/MODELS.md#linear-models):

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
| `sgd` | k=20, squared loss, `standardize=False` | 8,423,373 | the raw step; the default, `standardize=True`, scales each feature on the row first |
| `sgd` | k=20, `coef_min=0`, `coef_sum=1`, `standardize=False` | 2,288,779 | a simplex constraint sorts `2k` breakpoints per row, which makes `sgd` 3.7 times slower |
| `pa` | k=20, `standardize=False` | 8,284,518 | the raw step, as for `sgd` |
| `ftrl` | k=20, 1 target | 3,873,282 | |

**List several targets on the same features in one spec's `targets`, where
they share one set of feature sums:** 10 targets take 1.86 times as long as
one.

**A grid of half-lives keeps one set of sums per half-life, and runs them
in parallel.** So the grid of five takes 1.43 times as long as its shortest
half-life, 500, which runs alone at 2,443,773 rows a second. That is slower
than at the half-life of 1,000 that every other row here uses, because a
shorter half-life solves more often.

The models of [moments and correlation](docs/MODELS.md#moments-and-correlation):

| model | settings | rows/sec | what sets its speed |
|---|---|---:|---|
| `ew_cov` | k=20: mean, std, corr (230 statistics) | 1,655,993 | |
| `ew_cov` | k=20: mean, mahal, `mahal_q0.99` | 692,690 | the Mahalanobis distance factors a `k × k` matrix by Cholesky once per row |
| `ew_cov` | k=20: mean, cov, lags 1–5 | 1,123,565 | |
| `deco` | k=20, one equicorrelation | 2,093,753 | one number for the whole matrix of its `m` series, computed in `O(m)` a row with no `m × m` matrix to factor |
| `deco` | k=20 in 4 blocks | 1,563,248 | |
| `rcov` | 4 features, kernel, blocks of 1000 | 3,385,427 | it accumulates per row and computes its kernel only when the block closes |

The models of [clustering and classification](docs/MODELS.md#clustering-and-classification):

| model | settings | rows/sec | what sets its speed |
|---|---|---:|---|
| `ew_class` | k=20, 3 classes, full covariance | 466,466 | a `k × k` matrix factored by Cholesky once per *learned* row |
| `ew_class` | k=20, 3 classes, shared covariance | 531,384 | |
| `ew_class` | k=20, 3 classes, diagonal | 2,448,700 | |
| `kmeans` | 4 features, K=8 | 4,590,376 | a distance to each centre |
| `kmeans` | k=20, K=8 | 2,383,197 | a distance to each centre |
| `micro` | 4 features, `eps=1` | 8,799,815 | a distance to each centre |

The [sequential tests and regimes](docs/MODELS.md#sequential-tests-and-regimes):

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
permutation null redraws `n_perm` statistics every `permute_every_rows` rows.**
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
| `shards`, on a wide `marginal` | the one task that runs on several threads: `shards` splits its pairs across the pool within a group ([`marginal`](docs/MODELS.md#marginal--every-pairs-moments-kept-in-the-state)) | |
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
   .online.fit_predict(specs, chunk_size=200_000, save_state="grid.state")
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

**Set how many rows the bank takes at a time with the keyword `chunk_size`,
100,000 by default.** `lf.online.fit_predict`, `lf.online.predict`,
`ModelBank.fit_predict_batches`, `ModelBank.fit`, `with_windows` and
`refresh_time` take it. A frame passed to `ModelBank.fit_predict` is one
chunk, whatever its size.
This code uses `df` and `lf` from [Example data](#example-data):

```python
fitted = lf.online.fit_predict([spec], chunk_size=50_000).collect()               # a query, read 50,000 rows at a time
whole = po.ModelBank([spec]).fit_predict(df)                                       # a frame: one chunk, whatever its size
parts = pl.concat(po.ModelBank([spec]).fit_predict_batches(df, chunk_size=100))   # the same frame, 100 rows at a time
```

The chunk size changes only which rows carry `coef` and `support_coef`:
each group's last accepted row of every chunk.

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
| the chunks in flight | `chunk_size` | a few chunks at once, three on the command line ([Chunk size](#chunk-size)) |
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
   .online.fit_predict([spec], chunk_size=200_000)
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
target](docs/WINDOWS.md#windows-as-a-models-inputs-and-target) runs both forms. Polars
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
[`sgd`](docs/MODELS.md#sgd--stochastic-gradient-descent), the cheap `O(k)` baseline: at
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

**Do not close an ADBC cursor just after a read of its stream stopped
early.** A read stops early on an error, such as a spec refusing a clock
that steps back, or when the consumer stops. Closing the cursor at once
then crashed the process with a segmentation fault, measured on SQLite,
`adbc_driver_manager` 1.12.0 and polars 1.44.2 at 200,000 rows. A wait of
3 s before the close, or no close before the process exits, avoided it
([docs/ARROW-SOURCES.md](docs/ARROW-SOURCES.md#adbc-measured)).

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

### Training XGBoost and LightGBM on files larger than memory

`po.gbdt` and how to use it are in [Training XGBoost and LightGBM:
`po.gbdt`](docs/GBDT.md).

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
changes and any change to the numbers a model returns. From 1.0, a minor
version breaks nothing stable, so `polars-online~=1.0` keeps your code
working across the 1.x series. Pin the minor as well if your numbers must
not move, because a minor release can correct numbers that were wrong. The
package follows semantic versioning ([CHANGELOG.md](CHANGELOG.md)).

**Stable means part of this package's API, and from 1.0 a breaking change
to a stable part needs a new major version.** Before 1.0 it needs a new
minor. An unstable part can change its form in any minor release:

| | what it covers |
|---|---|
| **stable** | everything in `polars_online.__all__`; the `po.spec` constructors, their keywords and their defaults; the functions of the helper modules `po.eval`, `po.gram`, `po.ops` and `po.stream`; the `.online` namespace; the [output field names](#output-field-names); the columns, and their types, of the frames a bank and `po.spec` return, and of the closed-groups frame; the words each string-valued parameter takes; the TOML config keys; the command line's flags and exit status; and the environment variables the package reads |
| **unstable** | the file format of a window run's state (`with_windows(save_state=...)`); the written form of a formula target, in a TOML config or inside a bank's state; the Arrow export: `fit_predict_arrow`, `predict_arrow` and `ArrowStruct`; and the modules `po.sim`, `po.corr` and `po.gbdt`. Each says so in its docstring, and warns with `UnstableWarning` when used with `POLARS_ONLINE_WARN_UNSTABLE=1` set |
| **not stable** | any name that starts with an underscore, apart from the two columns `po.stream.embargo` adds, `_online_role` and `_online_role_weight`, which are stable; the Rust crates (`online-core`, `online-polars`, `online-cli` and `online-py`), which are not published to crates.io ([R3](docs/RELEASE-READINESS.md#r3--rust-crates-not-published)); and a model's numbers within the 1e-12 tolerance the golden tests pin, which depend on the platform and the Polars version |

**From 1.0, a changed default or meaning is a major release, and a fix to
numbers that were wrong is a minor one.** A number is wrong when it
disagrees with the model's stated definition, the update equations its
docstring gives:

| a change | before 1.0 | from 1.0 |
|---|---|---|
| a fix that moves no number | patch | patch |
| a fix to numbers that were wrong | minor | minor, declared in the CHANGELOG with the difference `scripts/compare_release.py` measures from the last release |
| a new model, parameter, output or function | minor | minor |
| a stable name renamed | minor, and the old name is refused, naming the new one | minor, and the old name keeps working until the next major. Using it warns with a `PolarsOnlineDeprecationWarning` that names the new one, as the next paragraph says |
| a default or a meaning changed, or a stable name removed | minor | major |
| a change to an unstable part | minor | minor, declared in the CHANGELOG |

**From 1.0, a renamed spec parameter is forwarded by a table, and any
other renamed name by a forwarding of its own.** The table reads a spec
builder's old keyword, a spec dict's old key and a TOML file's old key as
the new one, at any depth of a spec. It does not reach another function's
keyword, a function, a word a parameter takes, a command-line flag, an
environment variable, an output field or a frame column. Each of those
ships its own forwarding in the release that renames it: a check at the
function's top, a stub, an alias or a second column. Both warn with
`PolarsOnlineDeprecationWarning`, and both last until the next major.

**From 1.0, a state file a 1.x build wrote loads in every later 1.x.**
Each change to a state's layout ships a loader for the layout before it.
A saved state of each kind, frozen in the test suite, checks it three
ways. The current layout's fixtures load, continue to the bit, and save
back byte for byte. A previous layout's load through the loader, continue
to the bit, and save back as the current layout's fixture bytes of the
same case: the loader turns a state into exactly what this build writes
from the same rows. The two unstable formats, a window run's state and a
formula target inside a bank's, are outside that promise. Before 1.0 there
is no such promise: after upgrading across a release, refit a saved state
from its input ([Saving, loading and serving](#saving-loading-and-serving)).

**From 1.0, raising the Polars floor is a minor release, and dropping a
Polars major version a major one.** A resolver that installs the new
release upgrades Polars with it, inside the same Polars major, so no code
has to change for a raised floor:

| a change to the Polars range | before 1.0 | from 1.0 |
|---|---|---|
| widening it to admit a newer Polars | minor | minor |
| raising the floor | minor | minor |
| capping it below a Polars that broke this library, as [How the pin moves](docs/RELEASE-READINESS.md#how-the-pin-moves) describes | patch | patch |
| dropping a Polars major version | minor | major |

#### What is pinned

The wheel carries its own Rust `polars`, pinned exactly, and shares one
process with the py-polars you install. Data crosses between the two only
through the interfaces listed under
[Which interfaces carry a promise](#which-interfaces-carry-a-promise), so
py-polars can be any version in the range. The lowest py-polars depends on
what you use:

| py-polars | rust polars | pyo3-polars | pyo3 | Python |
|---|---|---|---|---|
| **>= 1.34.0, < 3** (built and tested against 2.0.0) | 0.55.2 | 0.28 | 0.29 | ≥ 3.12 (`abi3-py312`) |

| what you use | the lowest py-polars | why |
|---|---|---|
| `ModelBank` alone | 1.28.1 | |
| `lf.online.fit_predict`, `ModelBank.fit_predict_batches`, `ModelBank.fit`, `with_windows` and `refresh_time` | 1.34.0 | they read with `LazyFrame.collect_batches`, which py-polars added in 1.34.0 |
| the examples that stream a DuckDB, ADBC or pyarrow source into a bank | 1.43.0 | for `pl.scan_arrow_c_stream` |

The suite passes on 2.0.0, the pin, at every change. It runs on the
floor, 1.34.0, before every release and once a month
([How the pin moves](docs/RELEASE-READINESS.md#how-the-pin-moves)), and passed there on 2026-10-07.
On the floor, 23 tests skip, each naming the newer Polars it needs: most
check against a Polars function that has changed since, and the rest
stream a database or a pyarrow reader through `pl.scan_arrow_c_stream`.
The suite has also passed on 1.38.1, 1.44.1, 1.44.2 (the pin until
2026-10-08) and 2.0.0-rc.1 with identical numbers.
`tests/test_scaffold.py` asserts the pin and the range, and
[docs/RELEASE-READINESS.md](docs/RELEASE-READINESS.md) has each run with
its date and what differed.

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

The tables of what each guarantee is held to are in
[docs/TESTING.md](docs/TESTING.md#what-the-readmes-guarantees-are-held-to).

### Development

[CONTRIBUTING.md](CONTRIBUTING.md) has the commands to build and test from
a checkout, and says how to make changes.

| to find | read |
|---|---|
| every document, and which to read for what | [docs/README.md](docs/README.md) |
| a map for coding agents, at the repo root and on the docs site | [llms.txt](llms.txt) ([llmstxt.org](https://llmstxt.org)) |
| the API reference, built from the docstrings and published from every green push to `main` | <https://hgilde.github.io/polars-online/> |

### License

Apache-2.0. See [SECURITY.md](SECURITY.md) to report a vulnerability.
