<!-- kind: landing -->
# polars-online

> **GitHub project:** [github.com/hgilde/polars-online](https://github.com/hgilde/polars-online)
>
> **Every feature, one line each:** [FEATURES.md](docs/FEATURES.md) · **Five tasks, end to end:** [HIGHLIGHTS.md](docs/HIGHLIGHTS.md)

polars-online fits models online, one row at a time, to
[Polars](https://pola.rs) data, for anyone whose data are too large to
hold in memory at once: linear models, streaming moments, clustering and
regime detection. Each row is predicted before the model learns from it.
When the rows have a time order, a fit can also be local, following the
recent rows without fitting the last rows again at every step. Features and targets can be means over a window of the stream,
looking back or ahead. The models are written in Rust, and run from Python
or from a command line of their own.

| section | what it covers |
|---|---|
| [Introduction](#introduction) | [a first fit](#a-first-fit) · [the idea](#the-idea) · [terminology](#terminology) · [what you can rely on](#what-you-can-rely-on) · [install](#install) · [example data](#example-data) |
| [What you can do with it](#what-you-can-do-with-it) | five tasks run end to end in [HIGHLIGHTS.md](docs/HIGHLIGHTS.md), and the capabilities they show |
| [Where to go next](#where-to-go-next) | [the feature index](docs/FEATURES.md), every feature with a line and a link · the pages for each task, by kind |
| [Models](#models) | the four families, each linking its models' entries in [MODELS.md](docs/MODELS.md) |
| [What this is not](#what-this-is-not) | what polars-online leaves to Polars and to other tools |
| [Contributing and license](#contributing-and-license) | building from a checkout, and the license |

## Introduction

Here is a first model fit and an introduction to the library.

### A first fit

In this first fit you regress each stock's return on two features over a
made-up trading day. You know the day's true betas (the coefficients of
the return on each feature), and you read back the betas the model found. It takes
five steps, each with its result. To run it, install the library and numpy
with `pip install "polars-online[numpy]"` ([Install](#install) has the
requirements).

A *spec* describes one model. A *model bank*
([`ModelBank`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.ModelBank))
fits a set of models together in one pass over the rows, one for each spec
and each group (the rows that share one value of the column `group=` names,
here each stock). `.online.fit_predict`, chained on a query (a chain of Polars
steps that runs only when you ask for its result), runs a bank over the
query's rows and adds the specs' output to them.

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
prices.write_parquet("prices.parquet")          # on disk: every fit in this block reads it back as a query

# 2. A fit with every row weighted the same: fit the rows before 15:30, and save the state.
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
# 3. Read the fit back from the saved state, and serve the last half hour from it.
saved = (
    po.ModelBank.load("bank.state").coef()      # the fit in the file, a row per coefficient
    .pivot("term", index="group", values="coef")   # a row per stock, a column per term
)
served = (
    pl.scan_parquet("prices.parquet").tail(1_800)   # the last half hour, from 15:30
    .online.predict("bank.state")               # predicts each row from the state, learns nothing
    .online.unnest([whole])
    .collect()
)

# 4. A fit that follows the recent rows, and its betas on every row.
local = po.spec.ewridge(
    "local", targets=["ret"], features=["signal_a", "signal_b"], group="stock_id",
    clock="ts", half_life="10m",                # a row's weight halves every ten minutes of ts
    gap_cap="5m",                               # and a gap longer than five minutes decays as five
    coef_every=0,                               # the coefficients on every row
)
betas = pl.scan_parquet("prices.parquet").online.fit_predict([local]).online.unnest([local]).collect()   # a row of betas per input row

# 5. A target that looks ahead: each row's mean return over the next ten minutes.
fwd_ret = po.ewm_mean("ret", half_life=float("inf"), window_size="10m", reverse=True).alias("fwd_ret")
ahead = po.spec.ewridge(
    "ahead", targets=[fwd_ret], features=["signal_a", "signal_b"], group="stock_id",
    clock="ts", half_life="10m", gap_cap="5m",
    embargo="10m",                              # learn each row once its ten minutes have passed
)
ahead_out = pl.scan_parquet("prices.parquet").online.fit_predict([ahead]).online.unnest([ahead]).collect()   # each row's prediction of fwd_ret
```

**Step 1 makes up the input,** `prices`, and saves it to a parquet file.
It begins with these rows:

| ts | stock_id | signal_a | signal_b | ret |
|---|---|---:|---:|---:|
| 09:30:00 | A | 0.13 | 0.01 | -0.10 |
| 09:30:01 | B | -0.13 | 0.03 | 0.60 |
| 09:30:02 | C | 0.64 | 0.02 | 0.39 |
| 09:30:03 | A | 0.10 | -0.03 | 0.21 |
| 09:30:04 | B | -0.54 | -0.08 | -0.08 |
| 09:30:05 | C | 0.36 | -0.01 | 0.24 |

**Step 2 fits the rows before 15:30, each with the same weight, and saves
the state, everything the bank has learned.** `whole` is a ridge
regression solved from running sums (the decayed sums a model keeps in
place of its rows), and `half_life=float("inf")` forgets nothing. So at
every row its fit is the batch fit over the rows it has learned, and its
state is a complete summary of those rows, worth saving and serving.
`.online.fit_predict` adds one column per spec, named after the spec,
whose value in each row is a record of named fields (its type is Polars'
`Struct`). `.online.unnest`, chained after it, spreads the fields into
columns. For stock A, `flat` begins:

| ts | ret | pred_ret | resid_ret | weight_sum | withheld_reason |
|---|---:|---:|---:|---:|---|
| 09:30:00 | -0.10 | null | null | 0 | above_max_error_inflation |
| 09:30:03 | 0.21 | null | null | 1 | above_max_error_inflation |
| 09:30:06 | 0.52 | null | null | 2 | above_max_error_inflation |
| 09:30:09 | 1.22 | -0.47 | 1.69 | 3 | null |
| 09:30:12 | -1.18 | 1.64 | -2.81 | 4 | null |

`pred_ret` is each row's prediction, made from the stock's rows before it,
and `resid_ret` is `ret` less `pred_ret`. `weight_sum` is the weight behind
that prediction: here the count of the stock's rows learned so far. A
stock's first three rows have too little behind them, so `pred_ret` is
null and `withheld_reason` says why ([Warm-up gates](docs/MODELS.md#warm-up-gates)).

**Step 3 reads the fit back from the saved state, and serves the last half
hour from it.** The state holds each stock's fit, near the average of the
true betas until 15:30: 0.48 on `signal_a`, whose beta rises from 0.2 to
0.75, and -0.2 on `signal_b`. `saved` holds it:

| group | intercept | signal_a | signal_b |
|---|---:|---:|---:|
| A | -0.01 | 0.48 | -0.19 |
| B | -0.01 | 0.45 | -0.19 |
| C | 0.01 | 0.48 | -0.20 |

`served` predicts the rows from 15:30 with that state, and learns nothing
from them, so its `weight_sum` stays at the 7,200 rows of each stock the
state learned. For stock A it begins:

| ts | pred_ret | weight_sum |
|---|---:|---:|
| 15:30:00 | 0.24 | 7200 |
| 15:30:03 | 0.22 | 7200 |
| 15:30:06 | 0.72 | 7200 |

[Resume from a saved state](docs/how-to/resume-from-a-saved-state.md)
shows how to go on learning from new rows instead.

**Step 4 fits locally, following the recent rows.** Give the same spec a
clock (the column that says how far apart two rows are, here the
timestamps `ts`), `gap_cap="5m"` (required with a clock: a gap longer than
five minutes decays as five), and `half_life="10m"`. Each row's weight then halves every ten
minutes of `ts`. The fit is now local: it describes the recent past, and
it moves from row to row, so its final state is a poor summary of the day
([Local fits](docs/explanation/time-and-decay.md#local-fits)). `coef_every=0` writes, on every row,
the betas the model holds after that row, from its latest solve. Each hour
from 10:30, stock A's beta on `signal_a` climbs with the true one, which is
0.29, 0.38, 0.48, 0.57, 0.66 and 0.75 at those times:

| ts | coef_ret_signal_a | coef_ret_signal_b |
|---|---:|---:|
| 10:30:00 | 0.30 | -0.16 |
| 11:30:00 | 0.35 | -0.16 |
| 12:30:00 | 0.47 | -0.21 |
| 13:30:00 | 0.57 | -0.18 |
| 14:30:00 | 0.66 | -0.22 |
| 15:30:00 | 0.72 | -0.18 |

**Step 5 predicts a target that looks ahead.** `fwd_ret`, a Polars
expression (a column computation, such as `pl.col("x") + 1`) written with
`po.ewm_mean(..., reverse=True)`, is the mean of the stock's returns over
the next ten minutes, weighted by time. Its value at a row is known only
ten minutes later. With `embargo="10m"`, the bank predicts each row where
it sits and learns from it ten minutes later. So `ahead` learns nothing
from 9:30 to 9:40. `resid_fwd_ret` is null on every row, because no row's
target is known when the row is predicted. `fit_predict` refuses an
`embargo` shorter than the target's ten minutes ([Why every prediction is
out-of-sample](docs/explanation/out-of-sample.md)). `weight_sum`
is now a decayed weight, less than the count of rows learned. For stock A,
every five minutes:

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
only then learns from the row ([DIAGNOSTICS.md](docs/DIAGNOSTICS.md)
says what each diagnostic measures).

**The stream can be far larger than memory.** A bank keeps only what its
models have learned, and the rows an `embargo` or a window still holds. To fit
parquet files larger than memory, start the query with
`pl.scan_parquet("ticks/*.parquet")` before `.online.fit_predict`, and end it
with `.sink_parquet("fitted.parquet")`, which writes the result to a file, in
place of `.collect()`. A `sort`, a `collect()` before the bank or a
`rolling` over groups holds its whole input ([What streams, and what does
not](docs/explanation/chunk-invariance-and-streaming.md#what-streams-and-what-does-not)).

**Row order matters when a model forgets, and when a spec has a clock.**
With a *decay*, older rows count less, and a spec with a clock refuses a
row that steps back on it. So sort the rows by time with Polars' `sort`, in
the query before the bank, unless they arrive sorted. A `sort` holds every
row of its input while it sorts, so write a stream larger than memory to
its files in order. Without a decay or a clock, the models that solve or
accumulate give the same answer in any order ([Without a
decay](docs/explanation/time-and-decay.md#without-a-decay) says why, and
[the models at a glance](docs/MODELS.md#the-models-at-a-glance) gives each model's kind).

**Features and targets can be windows over the stream, written as Polars
expressions.** A feature can look back, such as the last minute's
time-weighted mean of the mid, added as a column in the query before the
bank. A target can look ahead, such as the next minute's VWAP less the mid,
named in the spec's `targets` and learned once its minute has passed.
Windows always need the rows in time order ([Make features that look
back over the stream](docs/how-to/make-window-features.md)).

**A model bank runs three ways, with the same numbers from each.** It runs
inside a Polars query with `.online.fit_predict`, in your own Python loop
calling `ModelBank.fit_predict` on each chunk, or from the standalone
`polars-online` command line, without Python. Three how-tos show each:
run a bank [in a query](docs/how-to/run-a-bank-in-a-query.md), [in a Python
loop](docs/how-to/run-a-bank-in-a-python-loop.md) or [from the command
line](docs/how-to/run-a-bank-from-the-command-line.md).

### Terminology

polars-online uses these words in a sense of its own. The
[glossary](docs/GLOSSARY.md) defines every other word the docs use for
the library:

| term | meaning |
|---|---|
| **spec** | the description of one model: which model, which columns it reads, and how it treats time. `po.spec.ewridge(...)` builds one |
| **model bank**, or *the bank* | a set of models, one for each spec and each group, fitted together over the same rows, and the Python object that holds them, [`ModelBank`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.ModelBank). *The bank* always means a model bank |
| **stream** | the rows in the order the bank reads them |
| **chunk** | one of the pieces the bank takes the stream in, one at a time: `chunk_size` rows in a query, 100,000 by default, or each frame passed to `ModelBank.fit_predict` |
| **state** | everything a bank has learned |
| **clock**, **decay** | the column, named by `clock=`, that says how far apart two rows are, and the forgetting measured along it: each row's weight halves every `half_life` of the clock ([Time and decay](docs/explanation/time-and-decay.md)) |
| **`weight_sum`** | the *weight behind the state* that produced a row's prediction, which `min_weight` reads ([`weight_sum`: the weight behind a prediction](docs/explanation/warm-up-and-readiness.md#weight_sum-the-weight-behind-a-prediction)) |

### What you can rely on

These hold for every spec, however a bank is run:

| | |
|---|---|
| **out-of-sample predictions** | every row is predicted before its own target is learned ([Why every prediction is out-of-sample](docs/explanation/out-of-sample.md)), and a target that looks ahead is learned only once its window has closed ([Make a target that looks ahead](docs/how-to/make-a-target-that-looks-ahead.md)) |
| **bounded memory** | memory grows with the models' state and the rows a delay or a window holds, and never with the number of rows that have passed ([Where the time and memory go](docs/explanation/where-the-time-and-memory-go.md)) |
| **any chunking** | one chunk or a thousand, with or without a save and resume in the middle, gives the same numbers, to the last bit. Only which rows carry the coefficients can change: `coef`, and `support_coef` and the standard errors beside it ([Chunk invariance and streaming](docs/explanation/chunk-invariance-and-streaming.md)) |
| **any thread count** | the thread count, set with a bank's `threads=` (or `POLARS_ONLINE_MAX_THREADS` for every bank that does not set it), changes only the speed. The bank's threads share the work by spec and group: with 64 groups, 14 threads process 7.3× the rows per second of one ([Use more cores](docs/how-to/use-more-cores.md)) |
| **unusable values** | in a feature, target or weight, NaN, ±inf and any magnitude above `1e100` count as null, so a sentinel such as `1e300` never reaches a model. A sentinel inside that range, such as `-999`, does, and [`bank.check()`](docs/how-to/check-data-quality.md) flags one that recurs. A null target leaves the row predicted and the clock moving, and a null feature or weight skips the row: no prediction, and nothing learned ([Hold a row back](docs/how-to/hold-a-row-back.md)) |
| **zero weights** | a row of weight 0 is predicted and moves the clock, but the model learns nothing from it ([Weight rows](docs/how-to/weight-rows.md)) |
| **one `weight_sum`** | `weight_sum` means the same in every model, so one `min_weight` means the same for every spec in a bank ([`weight_sum`: the weight behind a prediction](docs/explanation/warm-up-and-readiness.md#weight_sum-the-weight-behind-a-prediction)) |
| **a clock policy** | a spec with a clock caps each step at `gap_cap`, and applies `session_gap` at a session boundary. A step back the model would learn from is refused, naming the row, unless `restart_after_step_back` makes it a reset ([Handle sessions, gaps and steps back](docs/how-to/handle-sessions-gaps-and-steps-back.md)) |
| **named mistakes** | every keyword is checked against its type when the spec is built, and a missing column is reported, with the spec that wanted it and the role it had there, before the bank learns any row |
| **tested** | 2,653 Rust tests and 7,211 Python cases (counted on 2026-10-24), held to independent libraries such as scikit-learn, statsmodels and river and to adversarial streams, run on macOS, Windows and Linux at every push ([How the library is tested](docs/explanation/how-the-library-is-tested.md)) |
| **a stable API** | from 1.0, a minor release adds to the API, fixes numbers that were wrong, and may rename a stable part while the old name keeps working. Only a major release removes a stable part or changes a default or a meaning, apart from a few defaults that minor releases changed, each listed with the setting that restores it ([Versions and the Polars pin](docs/VERSIONS.md#this-packages-own-versioning)) |

### Install

Install the wheel from PyPI with pip or uv:

```sh
pip install polars-online      # or: uv add polars-online
```

| need | detail |
|---|---|
| Python | 3.12 or newer: one wheel per platform covers every CPython from 3.12 on ([How the library is tested](docs/explanation/how-the-library-is-tested.md) lists the versions CI runs) |
| Polars | `polars>=1.37.0,<3`: the range the test suite has measured, which is not a guarantee. A weekly job and every release run the test suite on the newest Polars ([Versions and the Polars pin](docs/VERSIONS.md#the-polars-range)) |
| depends on | nothing beyond `polars` at run time: the wheel carries its own copy of Polars' Rust code, so it needs no other native library |
| numpy | needed only by `ModelBank.gram()`, the `po.gram`, `po.corr`, `po.sim` and `po.gbdt` helpers, and `bank.check()`'s collinearity and leakage checks: to use them, install it with the package, as `pip install "polars-online[numpy]"`. Without it, those calls raise `ModuleNotFoundError`, and `bank.check()` reports one `not_checked` finding for the two checks |
| wheels | macOS (arm64, x86_64), Windows x64, and Linux (x64 glibc and musl, aarch64 glibc), on PyPI and on each GitHub release beside the command-line binaries. On any other platform, the install builds the package from its source distribution, which needs a stable Rust toolchain ([rustup](https://rustup.rs)) |
| size | 25 to 30 MB to download and 71 to 99 MB installed, by platform |

To build from a checkout, install uv and a stable Rust toolchain
([CONTRIBUTING.md](CONTRIBUTING.md)), then run from its root:

```sh
uv sync
uv run maturin develop --release -m crates/online-py/Cargo.toml
```

### Example data

Most examples on the how-to, explanation and reference pages read one of
two made-up frames: `df`, 400 rows of numbers and text keys a minute apart,
and `trades`, a stream of quotes with trades between them. The examples
that serve or resume a bank also read `today`, the 400 rows that follow
`df`'s on the clock `t`. Before running such an example, build all three
with this code. It saves each to a parquet file, and an
example that runs a query reads the file back with `pl.scan_parquet`, as a
stream too large to hold would be read:

```python
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import polars_online as po

# df: 400 rows of numbers and text keys, a minute apart.
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
# Save df, and the rows that follow it, as the files the examples read as a query.
df.write_parquet("ticks.parquet")                                # df on disk, as one file
lf = pl.scan_parquet("ticks.parquet")                            # the same rows, read back from the file as a query
today = df.with_columns(pl.col("t") + 400.0)                     # the 400 rows that follow df's, on the clock t
today.write_parquet("today.parquet")
later = pl.scan_parquet("today.parquet")                         # today, read back as a query
Path("ticks").mkdir(exist_ok=True)                               # and df as two files, for the examples that read ticks/*.parquet
df.head(200).write_parquet("ticks/part-0.parquet")
df.tail(200).write_parquet("ticks/part-1.parquet")

# trades: quotes, with trades between them, about two rows a second.
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
trades.write_parquet("trades.parquet")                           # on disk, for the examples that read it
```

`df` holds a known answer, so an example's output can be read against it.
`y` is `1 + 2·x0 − x1` plus noise of spread 0.5, and `x2`, `signal_a` and
`signal_b` play no part in it. `ret`'s beta on `signal_a` is 0.8 for the
morning session's 200 rows and −0.8 for the afternoon's: a break. Its noise
is calm, of spread 0.5, on rows 0 to 99 and 200 to 299, and volatile, of
spread 2, on the other two hundred: two regimes. `trades` holds no planted
relation: its mid is a random walk.


## What you can do with it

**Five tasks run end to end in [HIGHLIGHTS.md](docs/HIGHLIGHTS.md),** each
on data it makes up with a known answer, and each prints what the library
found, to read against that answer:

| task | what it does |
|---|---|
| automate a round of model selection | fits exponentially weighted ridge regressions over many combinations of factors, targets and half-lives, measured both in time and in traded volume, beside a detector of regime changes in the factors. All of it runs in one pass over the files, in bounded memory so the files may be larger than memory, and writes one output: each fit's out-of-sample residuals and coefficients, the combination chosen as best on each trade from the trades before it, and markers where the regime changed, from which to read how stable each model is |
| build a forward-looking signal and resume it | predicts the next five minutes from the last five and from the out-of-sample residual a first-stage ridge leaves of a factor, learns each row once its five minutes are known, fits through price jumps with a robust regression, and resumes in a fresh Python process from the saved state, to the bit |
| train gradient-boosted trees on files larger than memory | trains XGBoost and LightGBM from parquet files around an online model's prediction, so the trees follow a level that drifts past every training row |
| monitor a live model | keeps 90% intervals on predictions whose targets arrive about 300 rows late, flags a slope that flips, and finds each planted change in the features' correlation |
| measure correlations and fit curves on tick data | measures correlations between series that trade at their own times, and fits a curve locally along trade size |

**[The page's index](docs/HIGHLIGHTS.md#capabilities) lists each
capability the tasks use, with the step that shows it.** The tasks use
[bounded EWMAs](docs/GLOSSARY.md#fitting-a-model) (exponentially weighted
moving averages whose rows drop out past a window) forward and back,
`embargo`, two-stage regressions, and many column combinations and specs
in one pass. They also use change detection, selection made out-of-sample,
scores that merge, robust fits, exact resume, trees around an online
model, and intervals when targets arrive late. On tick data, they align
series that trade at their own times, and fit locally along any clock:
time, traded volume or trade size.

## Where to go next

**Every feature of the library is listed in [the feature index](docs/FEATURES.md),**
one line each with a link to the page that shows it. It lists every model,
every function and method, the parameters every spec shares, the errors
and warnings, the environment variables and the command line.

**To start from what you want to do, find the task in the first column,**
and the pages for it by kind. A tutorial takes you from nothing to one
working result, and a how-to solves one task. A reference describes one
thing completely, and an explanation gives the reasons.

| you want to | tutorial | how-to | reference | explanation |
|---|---|---|---|---|
| run a bank over a stream, in a query, a loop or a job | [Fit a model bank over parquet files in a query](docs/tutorials/stream-from-parquet.md) | [in a query](docs/how-to/run-a-bank-in-a-query.md) · [in a Python loop](docs/how-to/run-a-bank-in-a-python-loop.md) · [from the command line](docs/how-to/run-a-bank-from-the-command-line.md) · [hand the output to Arrow](docs/how-to/hand-the-output-to-arrow.md) · [feed it from a database](docs/how-to/feed-a-bank-from-a-database.md) · [run it in Pathway](docs/how-to/run-a-bank-in-pathway.md) | [RUNNER.md](docs/RUNNER.md), the command line | [Chunk invariance and streaming](docs/explanation/chunk-invariance-and-streaming.md) · [Polars interfaces](docs/explanation/polars-interfaces.md) |
| choose a model and say what it reads | [A first fit](#a-first-fit) | [choose a model](docs/how-to/choose-a-model.md) · [one model per group](docs/how-to/fit-one-model-per-group.md) · [weight rows](docs/how-to/weight-rows.md) · [hold a row back](docs/how-to/hold-a-row-back.md) · [targets that go missing on different rows](docs/how-to/fit-targets-that-go-missing-on-different-rows.md) · [a count target](docs/how-to/fit-a-count-target.md) · [bound the coefficients](docs/how-to/bound-the-coefficients.md) | [MODELS.md](docs/MODELS.md): [the models at a glance](docs/MODELS.md#the-models-at-a-glance), [what every spec takes](docs/MODELS.md#what-every-spec-takes) and each model's entry · [OUTPUTS.md](docs/OUTPUTS.md), what each model writes | [Why every prediction is out-of-sample](docs/explanation/out-of-sample.md) · [polars-online against scikit-learn](docs/explanation/against-scikit-learn.md) |
| follow time: a clock, a half-life, sessions and late targets | [Follow a local fit through a day](docs/tutorials/local-fit-with-a-clock.md) | [handle sessions, gaps and steps back](docs/how-to/handle-sessions-gaps-and-steps-back.md) · [put rows in clock order](docs/how-to/put-rows-in-clock-order.md) · [forget rows past a window](docs/how-to/forget-rows-past-a-window.md) · [fit locally along a feature](docs/how-to/fit-locally-along-a-feature.md) · [learn targets known late](docs/how-to/learn-targets-known-late.md) · [follow a level that moves](docs/how-to/follow-a-level-that-moves.md) | [What every spec takes: the clock](docs/MODELS.md#the-clock-and-its-units) | [Time and decay](docs/explanation/time-and-decay.md) |
| make features and targets from the stream | | [features that look back](docs/how-to/make-window-features.md) · [a target that looks ahead](docs/how-to/make-a-target-that-looks-ahead.md) · [scale a feature by its spread](docs/how-to/scale-a-feature-by-its-spread.md) · [weight rows by a robust scale](docs/how-to/weight-rows-by-a-robust-scale.md) · [align series that tick at their own times](docs/how-to/align-series-that-tick-at-their-own-times.md) · [fit on factor residuals](docs/how-to/fit-on-factor-residuals.md) | | [WINDOWS.md](docs/WINDOWS.md), the window operators |
| save a bank, resume it and serve from it | [Fit, save and serve a model bank](docs/tutorials/fit-save-and-serve.md) | [save a state](docs/how-to/save-a-state.md) · [resume from it](docs/how-to/resume-from-a-saved-state.md) · [resume a window run](docs/how-to/resume-a-window-run.md) · [serve from it](docs/how-to/serve-from-a-saved-state.md) · [read it without this library](docs/how-to/read-a-state-without-this-library.md) | [STATE-WORKFLOW.md](docs/STATE-WORKFLOW.md) | [A bank's state and resuming it](docs/explanation/state-and-resuming.md) |
| read what a bank learned | | [inspect a bank](docs/how-to/inspect-a-bank.md) · [find an output field](docs/how-to/find-an-output-field.md) · [read the coefficients](docs/how-to/read-the-coefficients.md) · [close groups and read their rows](docs/how-to/close-groups.md) · [keep a Gram per block](docs/how-to/fit-weekly-grams-in-one-pass.md) · [pool Grams](docs/how-to/pool-grams.md) · [repair a correlation matrix](docs/how-to/repair-and-score-a-correlation-matrix.md) · fit from a Gram: [a ridge](docs/how-to/fit-a-ridge-from-a-gram.md), [a Huber regression](docs/how-to/fit-huber-on-a-gram.md), [per-block coefficients](docs/how-to/fit-per-block-coefficients.md), [columns that hold across weeks](docs/how-to/select-columns-stably-over-weeks.md) · [check a fit is determined](docs/how-to/check-a-fit-is-determined.md) · [read Grams from saved files](docs/how-to/read-grams-from-saved-files.md) · [store a compact Gram](docs/how-to/store-a-compact-gram.md) | [OUTPUTS.md](docs/OUTPUTS.md) | [GRAM.md](docs/GRAM.md), fits from the Grams one pass leaves · [Grids, instances and slots](docs/explanation/grids-and-instances.md) |
| track moments, correlations, clusters and regimes | | [flag joint outliers](docs/how-to/flag-joint-outliers.md) · [track principal components](docs/how-to/track-principal-components.md) · [find which series leads](docs/how-to/find-which-series-leads.md) · [screen features against a target](docs/how-to/screen-features-against-a-target.md) · [track the average correlation](docs/how-to/track-the-average-correlation.md) · [a block's covariance from ticks](docs/how-to/estimate-a-blocks-covariance-from-ticks.md) · [keep `kmeans` centers on the data](docs/how-to/keep-kmeans-centers-on-the-data.md) · [choose `micro`'s `eps`](docs/how-to/choose-micro-eps.md) · [choose a covariance type](docs/how-to/choose-a-covariance-type.md) · [keep `hmm` states on their regimes](docs/how-to/keep-hmm-states-on-their-regimes.md) · [read a sequential test](docs/how-to/read-a-sequential-test.md) | [MODELS.md](docs/MODELS.md) · [REGIMES.md](docs/REGIMES.md), what the regime models find on data whose truth is known | [Series that tick at their own times](docs/explanation/asynchronous-series.md) |
| check the data and the model | | [check data quality](docs/how-to/check-data-quality.md) · [decide when a model is ready](docs/how-to/decide-when-a-model-is-ready.md) · [check the scale](docs/how-to/check-the-scale.md) · [detect a break](docs/how-to/detect-a-break.md) · [check standard errors](docs/how-to/check-standard-errors.md) · [find what a model misses](docs/how-to/find-what-the-model-misses.md) · [measure heavy tails](docs/how-to/measure-heavy-tails.md) · [find an influential row](docs/how-to/find-an-influential-row.md) · [check feature health](docs/how-to/check-feature-health.md) · [compare two models](docs/how-to/compare-two-models.md) · [evaluate an output frame](docs/how-to/evaluate-an-output-frame.md) · [put an interval on a prediction](docs/how-to/put-an-interval-on-a-prediction.md) · [check a two-stage fit](docs/how-to/check-a-two-stage-fit.md) · [test a detector on known truth](docs/how-to/test-a-detector-on-known-truth.md) | [DATA-QUALITY.md](docs/DATA-QUALITY.md), each finding · [DIAGNOSTICS.md](docs/DIAGNOSTICS.md), each diagnostic | [Warm-up and readiness](docs/explanation/warm-up-and-readiness.md) · [How the diagnostics work](docs/explanation/how-diagnostics-work.md) · [How the data quality checks work](docs/explanation/how-data-quality-checks-work.md) |
| choose a model's settings | | [with `grid_search`](docs/CALIBRATION.md) · [a ridge and a half-life](docs/how-to/choose-a-ridge-and-half-life.md) · [a Kalman filter's drift speed](docs/how-to/choose-a-kalman-drift-speed.md) · [a learning rate](docs/how-to/choose-a-learning-rate.md) · [`ftrl`'s penalties](docs/how-to/choose-ftrl-penalties.md) · [Holt's smoothing](docs/how-to/choose-holts-smoothing.md) · [columns on the lasso path](docs/how-to/choose-columns-on-the-lasso-path.md) · [a changepoint hazard](docs/how-to/choose-a-changepoint-hazard.md) · [among a grid, as the stream runs](docs/how-to/choose-among-a-grid-online.md) · [by hand](docs/how-to/choose-settings-by-hand.md) · [when `ewridge` solves](docs/how-to/choose-when-ewridge-solves.md) · [start from a prior fit](docs/how-to/start-from-a-prior-fit.md) · [blend each session with the long run](docs/how-to/blend-each-session-with-the-long-run.md) | [VALIDATION.md](docs/VALIDATION.md), where the defaults came from | [Grids, instances and slots](docs/explanation/grids-and-instances.md) · [The robust scale](docs/explanation/robust-scale.md) |
| make a run faster or smaller | | [set the chunk size](docs/how-to/set-the-chunk-size.md) · [use more cores](docs/how-to/use-more-cores.md) · [bound a run's memory](docs/how-to/bound-memory.md) · [fit hundreds of features](docs/how-to/fit-hundreds-of-features.md) | [PERFORMANCE.md](docs/PERFORMANCE.md), the measured costs | [Where the time and memory go](docs/explanation/where-the-time-and-memory-go.md) |
| train XGBoost or LightGBM on files larger than memory | | [GBDT.md](docs/GBDT.md) | | [How `po.gbdt` builds training data](docs/explanation/how-po-gbdt-builds-training-data.md) |
| upgrade, or trust the numbers | | [reproduce an earlier version's fit](docs/how-to/reproduce-an-earlier-versions-fit.md) | [VERSIONS.md](docs/VERSIONS.md), what each release may change | [How the library is tested](docs/explanation/how-the-library-is-tested.md) |

The API reference, built from the docstrings, is at
<https://hgilde.github.io/polars-online/>. [docs/README.md](docs/README.md)
lists every document by kind, [CHANGELOG.md](CHANGELOG.md) says what each
release changed, and [llms.txt](llms.txt) is the map a coding agent reads
([llmstxt.org](https://llmstxt.org)). [GLOSSARY.md](docs/GLOSSARY.md)
defines every word the documents use for the library.

## Models

polars-online has twenty-two models in four families, and each family
answers one kind of question about a stream. A spec's clock, grouping and
warm-up mean the same whichever model it names ([What every spec
takes](docs/MODELS.md#what-every-spec-takes)), and [Choose a
model](docs/how-to/choose-a-model.md) says which to start with:

| family | the question it answers | its models |
|---|---|---|
| [Linear models](docs/MODELS.md#linear-models) | what will the target be? | `ewridge`, `rls`, `lasso`, `kalman`, `huber`, `quantile`, `sgd`, `pa`, `ftrl`, `holt` |
| [Moments and correlation](docs/MODELS.md#moments-and-correlation) | how do these columns vary and move together? | `ew_cov`, `marginal`, `deco`, `rcov`, `audit` |
| [Clustering and classification](docs/MODELS.md#clustering-and-classification) | which group does this row belong to? | `kmeans`, `micro`, `ew_class` |
| [Sequential tests and regimes](docs/MODELS.md#sequential-tests-and-regimes) | has something held, or changed, and which regime is the stream in? | `seqtest`, `corrchange`, `bocpd`, `hmm` |

## What this is not

**polars-online fits models to rows that are already read and in clock
order, so reading, joining and ordering the rows is the job of Polars or of
another tool before the bank.** A bank can take its rows from a database
through Polars, or run as an operator inside a stream processor. [Chunk
invariance](docs/explanation/chunk-invariance-and-streaming.md) means the
chunks another engine hands the bank cannot change the numbers. For a model this library does not fit,
such as gradient-boosted trees, `po.gbdt` builds XGBoost's and LightGBM's
training objects from files too large to hold ([GBDT.md](docs/GBDT.md)).

**When a spec names a `clock`, sort the rows by it before the bank, because
polars-online neither buffers nor reorders rows that arrive late.** A row
that steps back on the clock is refused, unless its step back is larger
than `restart_after_step_back`, which resets the model. When a new input
overlaps the rows a saved state has learned, `ModelBank.skip_learned`
drops them before the bank ([Resume from a saved
state](docs/how-to/resume-from-a-saved-state.md)). A target that
arrives late can be held back with [`embargo`](docs/how-to/learn-targets-known-late.md).
`clock`, `gap_cap`, `restart_after_step_back` and `session` describe time
*within* a stream, and none of them waits for a late row.

Three functions of `po.stream` prepare a stream before the bank.
`po.stream.refresh_time` puts series that tick at their own times on one refresh
grid ([Align series that tick at their own
times](docs/how-to/align-series-that-tick-at-their-own-times.md)).
`po.stream.with_windows` computes exponentially weighted means, variances,
sums, rates and quantiles over a window of the clock ([Make features that
look back over the stream](docs/how-to/make-window-features.md)).
`po.stream.embargo` doubles a stream so that each row's target is learned
a set delay after its row ([Learn targets known
late](docs/how-to/learn-targets-known-late.md)). polars-online
deliberately does **not** provide the rest:

| not provided | use instead |
|---|---|
| connectors or ingestion | a Polars reader in the query before the bank, such as `pl.scan_parquet`, or `pl.scan_arrow_c_stream` for a database ([Feed a bank from a database](docs/how-to/feed-a-bank-from-a-database.md)) |
| windows beyond those, such as a tumbling or sliding aggregation of any function, and asof or interval joins | Polars' `group_by_dynamic`, `rolling`, `join_asof` or `join_where` in the query before the bank, or a streaming framework such as [Pathway](https://pathway.com) |
| watermarks or a late-arrival policy | rows put in clock order before the bank, by Polars' `sort` or by a stream processor |
| distributed execution | one process, whose threads share the work by spec and group ([Use more cores](docs/how-to/use-more-cores.md)) |

## Contributing and license

[CONTRIBUTING.md](CONTRIBUTING.md) has the commands to build and test from
a checkout, and how to make a change. The documents for changing the
library say how to add a model ([EXTENDING.md](docs/EXTENDING.md)),
what the tests prove ([TESTING.md](docs/TESTING.md)), how to cut a release
([cut-a-release.md](docs/contributing/cut-a-release.md)) and the rules this
documentation is written to ([WRITING.md](docs/WRITING.md)).

polars-online is licensed under Apache-2.0. See [SECURITY.md](SECURITY.md)
to report a vulnerability.
