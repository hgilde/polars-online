# polars-online

Online model fitting for [Polars](https://pola.rs) — linear models,
streaming moments, clustering and regime detection — for rows that arrive
in time order and never all fit in memory at once. Rust core, Python API,
and a standalone command line ([docs/RUNNER.md](docs/RUNNER.md)).

> **A note on Polars versions.** Two of the three ways this library plugs
> into Polars carry no stability promise from Polars, so `polars>=1.34.0,<2`
> is measured rather than guaranteed. A weekly job runs the whole test suite
> on the newest Polars, and the response to a failure is decided in advance.
> Details in [Versioning and the Polars pin](#versioning-and-the-polars-pin).

**Contents.** [Introduction](#introduction) ·
[The models](#the-models) ·
[Install](#install) ·
[How a bank sees a stream](#how-a-bank-sees-a-stream) ·
[Running a bank](#running-a-bank) ·
[Memory](#memory-which-calls-stream) ·
[Saving, loading and serving](#saving-loading-and-serving) ·
[Preparing a stream](#preparing-a-stream) ·
[Reading the fit](#reading-the-fit) ·
[Diagnostics, selection and evaluation](#diagnostics-selection-and-evaluation) ·
[Models](#models) ·
[Parallelism](#parallelism) ·
[Performance](#performance) ·
[Against scikit-learn](#against-scikit-learn) ·
[What this is not](#what-this-is-not) ·
[Versioning and the Polars pin](#versioning-and-the-polars-pin) ·
[Testing](#testing) ·
[Development](#development)

## Introduction

**The idea.** You describe one or more models — a ridge regression of a
stock's return on two signals, say, with a separate regression for every
stock. polars-online fits all of them in a single pass over your rows. Each
row is *predicted* first, from what the models have learned so far, and
*learned from* second; that order is what makes every prediction honest —
no row's own outcome is in the number predicted for it. The models keep
only what they have learned, never the rows, which is what lets the whole
thing run on far more rows than fit in memory. The order of the rows
matters only when a model forgets: with a decay, older rows count less, so
the rows must come in time order; without a decay, a model that solves or
accumulates gives the same answer in any order.

**Four words this README uses throughout.**

| word | meaning |
|---|---|
| **spec** | the description of one model: which model, which columns it reads, how it should treat time. `po.spec.ewridge(...)` builds one |
| **model bank** — "the bank" | a set of specs fitted together over the same rows, and the Python object that holds them, [`ModelBank`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.ModelBank). Wherever this README says *the bank*, it means a model bank |
| **stream**, **chunk** | the rows, in time order, and the pieces they arrive in. The bank takes one chunk at a time and its results never depend on where one chunk ended and the next began |
| **state** | everything a bank has learned. Its size depends on the models, not on how many rows have gone past — which is why the stream can be any length |

**The basic example.** A ridge regression per stock, fitted over a folder of
parquet files, with the fitted state saved when the last row is reached;
then new rows scored against that state without learning from them:

```python
import polars as pl
import polars_online as po

spec = po.spec.ewridge(
    "ridge",                                         # the spec's name; its output column is named after it
    targets=["ret"], features=["signal_a", "signal_b"],
    clock="ts", halflife=600.0, max_dclock=300.0,   # older rows count less: their weight halves every 600 s
    group="stock_id",                                 # one separate regression per stock
)

(
    pl.scan_parquet("ticks/*.parquet")               # a Polars query over the files; nothing is read yet
    .online.fit_predict([spec], save_state="bank.state")   # the bank, inside the query
    .filter(pl.col("ridge").struct.field("n_eff") > 100)   # ordinary Polars on what comes out
    .sink_parquet("fitted.parquet")                  # runs the query, writing the result to a file a chunk at a time
)

scored = pl.scan_parquet("today.parquet").online.predict("bank.state").collect()   # score; learn nothing
flat = scored.online.unnest([spec])   # pred_ret, resid_ret, n_eff, coef_ret_intercept, coef_ret_signal_a, ...
```

Each spec adds one column to the output, named after the spec. Its value in
each row is a record with named fields: the prediction `pred_<target>`, the
residual `resid_<target>`, the effective number of observations `n_eff`, the
coefficients `coef`, and whatever diagnostics you switch on; `unnest`
spreads those fields into plain columns. Mistakes are named: every keyword
is checked against its type, and a missing column is reported by which spec
wanted it and in what role.

**Three ways to run a model bank, same numbers from each.** Inside a Polars
query, as above — [`lf.online.fit_predict(specs)`](https://hgilde.github.io/polars-online/namespaces.html#polars_online._frame.LazyFrameOnlineNamespace.fit_predict) returns a
`LazyFrame`, and running the query runs the bank. In your own Python loop —
make a [`ModelBank`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.ModelBank) and call `fit_predict` on each chunk yourself. Or
as a Polars expression, for a frame that is already in memory — this third
form loads every row at once, and warns to say so. [Running a
bank](#running-a-bank) shows each.

**Time, decay and convergence.** The rows of a stream are not all equally
relevant, so a model can forget: each row's weight halves every `halflife`
units of a *clock* — a column you name that says how far apart two rows
are, in seconds, in cumulative traded volume, or simply by counting rows.
Streams from markets bring their own shape, and the bank handles it: a
*session* boundary (the row that opens a new trading day), a gap in the
clock (an hour with no rows), and a clock that resets between sessions. Put
the same decay on one of the frame's own feature columns, with the rows
sorted by it, and the fit becomes local in that feature — a curve rather
than a line — in one pass. Turn decay off, and a model that solves for its
coefficients converges to the ordinary batch fit over every row it has
seen, whatever order the rows came in. [How a bank sees a
stream](#how-a-bank-sees-a-stream) explains each of these.

**Two guarantees.** Every row is predicted before its own outcome is
learned. One chunk or a thousand gives the same output, to the last bit.
Both are tests in the suite, not intentions.

**State can be saved and loaded.** Save what a bank has learned to a file;
load it to keep learning, or to score new rows without learning
([Saving, loading and serving](#saving-loading-and-serving)).

**Introspection and diagnostics.** Coefficients as a table or as columns.
Residual standard deviation and z-scores, break detection, choosing or
averaging among several settings as the stream runs, running R²,
correlation and hit rate, residual quantiles and autocorrelation — all
computed from what the models have already learned, so none of them see
the row they describe, and all kept in memory that does not grow with the
stream. What each costs in time is measured in
[docs/PERFORMANCE.md](docs/PERFORMANCE.md).

**Parallel within each chunk.** Each (spec, group) pair is fitted on its
own thread, and the number of threads changes the speed and nothing else.
With 64 groups, 14 threads process 8× the rows per second of one
([Parallelism](#parallelism)).

**Tested to match.** About 650 Rust tests and 2,200 Python test cases —
against numpy, against an independent implementation, on adversarial
streams, with fixed reference numbers on every OS — run on macOS, Windows
and Linux on every push ([Testing](#testing)).

## The models

Twenty model families, one set of stream semantics: a spec's clock, decay,
grouping and warm-up mean the same thing whichever model it names. Each row
links to the model's builder in the API reference and to the section of
this README that states its update rule.

*Learns by* says how a model takes in a row. A model that **solves** keeps
running sums and computes its coefficients from them; one that
**accumulates** keeps running sums and reports them. Both converge to the
ordinary batch answer when decay is off, in any row order. A model that
**steps** moves its coefficients a little on each row, one that **filters**
carries a belief forward from row to row, and one that **tests** counts
evidence as it goes; those three depend on the order the rows came in,
with or without decay ([Without a
decay](#without-a-decay-convergence-in-bounded-memory)).

| model | learns by | what it is |
|---|---|---|
| [`ewridge`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.ewridge) · [math](#ewridge--ew-ridge-on-sufficient-statistics) | solve | exponentially weighted ridge regression — the workhorse; several ridge values, feature sets and halflives can be fitted from the same running sums at almost no extra cost |
| [`rls`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.rls) · [math](#rls--recursive-least-squares) | solve | recursive least squares, in the numerically safe square-root form |
| [`lasso`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.lasso) · [math](#lasso--lasso-path-with-free-λ-selection) | solve | lasso / elastic-net path with the penalty chosen as the stream runs |
| [`kalman`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.kalman) · [math](#kalman--random-walk-β-dynamic-linear-model) | filter | Kalman filter with coefficients that drift as random walks |
| [`huber`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.huber) · [`quantile`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.quantile) · [math](#huber--quantile--robust-regression) | solve | robust and quantile regression |
| [`sgd`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.sgd) · [math](#sgd--stochastic-gradient-descent) | step | stochastic gradient descent with squared, Huber, quantile, ε-insensitive, Poisson and logistic losses |
| [`pa`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.pa) · [math](#pa--passive-aggressive-regression) | step | passive-aggressive regression — no learning rate to tune |
| [`ftrl`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.ftrl) · [math](#ftrl--online-logistic-regression) | step | FTRL-proximal logistic regression, with an L1 penalty that zeroes coefficients |
| [`ew_cov`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.ew_cov) · [math](#ew_cov--exponentially-weighted-moments) | accumulate | running mean, variance, covariance, correlation, partial correlation, Mahalanobis distance and principal components |
| [`holt`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.holt) · [math](#holt--holts-linear-trend) | step | Holt's linear trend — the baseline that uses no features |
| [`kmeans`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.kmeans) · [math](#kmeans--exponentially-weighted-k-means) | step | exponentially weighted k-means — cluster labels assigned before the row is learned from, with a split–merge move that finds a cluster born after seeding |
| [`micro`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.micro) · [math](#micro--density-based-clustering-any-shape) | step | density-based clustering — DenStream micro-clusters linked into clusters of any shape and number; flags the rows that belong to none |
| [`ew_class`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.ew_class) · [math](#ew_class--gaussian-classification-on-ew_cov-moments) | accumulate | Gaussian classification — QDA, LDA or naive Bayes, one set of running moments per class; a label column in, class probabilities out |
| [`seqtest`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.seqtest) · [math](#seqtest--a-sequential-test-of-a-sign-by-betting) | test | a sequential test of a sign by betting — evidence you can read at any row; on its own a column's sign, with `a`/`b` whether one spec of the bank predicts closer than another |
| [`marginal`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.marginal) · [math](#marginal--every-pairs-moments-kept-in-the-state) | accumulate | every (feature, target) pair's running mean, variance, covariance, correlation, slope and t — for a wide set of columns, kept in the state and read back as a table; optionally at a set of lags, and with binned target moments for the relations a correlation cannot see |
| [`deco`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.deco) · [math](#deco--one-correlation-for-the-whole-matrix) | accumulate | one correlation for the whole matrix — Engle & Kelly's equicorrelation, or one per block and per pair of blocks |
| [`rcov`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.rcov) · [math](#rcov--a-blocks-realised-covariance-robust-to-noise) | accumulate | a block's realised covariance, robust to microstructure noise — the Barndorff-Nielsen–Hansen–Lunde–Shephard kernel or Christensen–Kinnebrock–Podolskij pre-averaging, reported when a group closes |
| [`hmm`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.hmm) · [math](#hmm--which-regime-are-we-in) | filter | a Gaussian hidden Markov model, filtered as the stream runs — `ew_class` without the labels, with a transition matrix that can be learned |
| [`corrchange`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.corrchange) · [math](#corrchange--has-the-correlation-structure-changed) | test | has the correlation structure changed — the Wied–Krämer–Dehling constancy test span by span, or the size of a change between two windows against a permutation null |
| [`bocpd`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.bocpd) · [math](#bocpd--how-long-has-this-regime-lasted) | filter | how long has this regime lasted — Adams & MacKay's run-length posterior, so the answer is the age of the regime and not a flag |

## Install

```sh
pip install polars-online      # or: uv add polars-online
```

Wheels for macOS (arm64, x86_64), Windows x64 and Linux (x64 glibc and musl,
aarch64 glibc) are on PyPI; those and the command-line binaries are attached
to each GitHub release. Python 3.12+.
The wheel is ~19 MB to download and ~59 MB installed: it carries its own
copy of the Rust half of Polars, so nothing beyond `polars` itself has to be
present at run time. `numpy` is an optional extra: only `ModelBank.gram()`
and the `po.gram` / `po.corr` helpers need it.

From a checkout:

```sh
uv sync
uv run maturin develop --release -m crates/online-py/Cargo.toml
```

## How a bank sees a stream

A model bank reads a stream of rows in time order, one chunk at a time. The
parameters in this section are shared by every model and say how the rows
are to be read: which columns, how time and forgetting work, which rows
belong to which model, how much each row counts, and when a model has seen
enough to report.

### What a spec names

```python
spec = po.spec.ewridge(
    "ridge",
    targets=["y"],                 # at least one; the targets of one spec share the features' running sums
    features=["x0", "x1", "x2"],   # numeric columns of any width, Decimal and Boolean included; read as 64-bit floats
    add_intercept=True,            # the default
    clock="t", halflife=600.0, max_dclock=300.0,
)
# A text column in either list is refused rather than silently read as nulls.
# Columns the spec does not name pass through to the output untouched.
```

Two more parameters describe the output rather than the input, and are
explained where they matter: `coef_every` under
[Coefficients](#coefficients), and `label_delay` under [Labels that arrive
late](#labels-that-arrive-late).

### Time and decay

A model forgets. Each row's weight in the fit is halved every `halflife`
units of a *clock*: a column of the frame that says how far apart two rows
are, in seconds, in cumulative traded volume, in any number that only goes
up. Row by row, the weight of everything learned so far is multiplied by
`λ = 0.5 ** (Δclock / halflife)`, where `Δclock` is the clock's step from
the previous row. With no clock column, the row count is the clock, and a
halflife of 100 means a row 100 rows back counts half as much as the latest.

A stream from a market brings its own shape. A **session** is a span of
rows whose clock measures time continuously — a trading day, say; at the
boundary between two sessions the previous session's clock no longer
measures time, so the bank applies a step you choose instead of the raw
jump. A **gap** in the clock — an hour with no rows — should not decay the
fit as though an hour of rows had gone by, so the clock's step is capped.
And a clock that **resets** or runs backwards between sessions has a
policy. Every one of those is a parameter of the spec:

```python
timed = po.spec.ewridge(
    "timed", targets=["y"], features=["x0", "x1"],
    clock="t",                 # a numeric column that only goes up; None means the row count
    halflife=600.0,            # a row's weight halves every 600 clock units (or lam=, the weight kept per unit)
    max_dclock=300.0,          # the most the clock may step between two rows; required with a clock
    on_clock_reset="max",      # a backwards clock: "max" (the step is max_dclock), "zero", "reset_state", or "error"
    session="session",         # a column whose value changes at a session boundary ...
    session_gap=60.0,          # ... and the clock step to apply there; "reset" starts the model over, inf never applies it
)
# halflife=inf (or lam=1.0) turns forgetting off. A list of halflives fits one model per value.
# max_dclock=0 turns forgetting off; max_dclock=inf removes the cap.
# ewridge only: session_shrink= and long_halflife= pull the fit partway back, at a session
# boundary, toward a twin that forgets more slowly.
```

Date and time columns are refused as the clock; convert first
(`pl.col("ts").dt.epoch("s")`), so that the units of `halflife`,
`max_dclock` and `session_gap` are the units you chose. A huge finite
halflife is not `inf`: `halflife=1e12` still forgets, and the model
schedules that key off the halflife scale with it. Say `inf` for no
forgetting.

### With a decay: a local fit along any feature

The clock does not have to be a time. Sort the frame by one of its own
feature columns and name that column as the clock. `halflife` is then a
bandwidth in that feature's units: every row is fit on the rows before it,
each weighted by `0.5 ** (Δx / halflife)` — an exponential kernel. The fit
is a local regression, computed in one pass from running sums that do not
grow, where the usual way of getting one refits a window of rows at every
point.

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

They are the numbers a batch fit would give: against a kernel-weighted
least squares recomputed from scratch at every row they agree to 1e-12.

The kernel is one-sided — a row is fit on the rows before it, never after —
so the fit follows a curve with a lag, and the bandwidth trades that lag
against noise. On `sin(x)` with a bandwidth of 0.25 the fit sits 0.08 from
the truth where the best straight line sits 0.39; at a bandwidth of 1.0 the
same stream gives 0.29, most of the way back to the line.

`features` need not include the clock column: with other features the same
fit is a regression whose coefficients move along the clock. The clock
belongs to every model, not only to `ewridge`.

### Without a decay: convergence in bounded memory

With decay off — `halflife=inf`, or `lam=1.0` — a model that *solves* or
*accumulates* ([the model table](#the-models) says which) converges to the
batch fit it defines over every row it has seen, and the order of the rows
does not reach the fit at all: forwards, backwards or shuffled gives the
same coefficients. A model that *steps* or *filters* depends on the order
whether or not decay is on. Each model's own section says what it
converges to.

Either way, memory is proportional to the model's state — the running sums
it keeps — and not to the number of rows that have passed: the frame never
has to fit in memory, and a stream of any length fits in the same state.
`solve_every` makes a solving model compute its coefficients less often
than every row when the exact-at-every-row fit is not needed; they are then
at most that many rows out of date.

### Groups

`group` names a column, and the bank keeps one separate model per distinct
value of it — one per stock, per symbol, per anything — all fitted in the
same pass over the stream. Every other parameter, the clock included,
applies within the group.

```python
per_stock = po.spec.ewridge(
    "per_stock", targets=["y"], features=["x0", "x1"], clock="t", halflife=600.0, max_dclock=300.0,
    group="stock_id",           # one model per distinct value of this column
    group_close="monotone",    # or "session": when a group is finished, write its running sums out as
)                              # one row and free its memory -- read them with bank.closed_groups()
```

`group_close` is what keeps a bank's memory bounded when new group values
never stop appearing ([One row per finished
group](#one-row-per-finished-group)). Groups otherwise live for the life of
the bank; a long-running bank can drop the ones that have gone quiet ([In a
loop](#in-a-loop-modelbank)).

### Weights

```python
weighted = po.spec.ewridge(
    "weighted", targets=["y"], features=["x0", "x1"], clock="t", halflife=600.0, max_dclock=300.0,
    weight="w",                # a column of row weights: a row of weight w counts as w observations would
)                              # weight 0 is legal: the row is scored, the clock advances, nothing is learned
```

`n_eff`, below, counts weight rather than rows. [Three ways to hold a row
back](#three-ways-to-hold-a-row-back) says when weight `0` is the right
one of the three.

### Warm-up

`min_periods` lets a model report only once it has seen enough data to
have converged, so it never reports a number it is not yet informed
enough to give.

```python
warm = po.spec.ewridge(
    "warm", targets=["y"], features=["x0", "x1"], clock="t", halflife=600.0, max_dclock=300.0,
    min_periods=50.0,          # in n_eff units, not rows: every output is null until n_eff reaches it
)                              # a list gives one threshold per target; the model learns from every row either way
```

`n_eff` is the *effective number of observations*: the total weight behind
the state that produced *this row's* prediction, after forgetting, and
measured before the row's own update. So it is `0` on a stream's first row,
runs one behind the row count while nothing is forgotten, settles at
`1 / (1 − λ)` once forgetting balances arrival, and means the same thing in
every model — which is what makes one `min_periods` mean the same thing
across a bank.

### Nulls

A null in any feature, or in the weight, skips the row: outputs are null, no
update happens, the clock still advances. A null in one target still emits
that target's `pred`, leaves its `resid` null, and skips only that target's
update. NaN, ±inf and any magnitude above `1e100` count as null, so sentinels
never reach a model.

### Three ways to hold a row back

They differ, and it matters which you use.

- **Weight `0`** — the row is scored, the clock advances, nothing is learned:
  the coefficients do not move at all. Use it to keep a row's place in the
  stream. Since the clock advances, `n_eff` keeps decaying and can fall
  below `min_periods` if you score for a long stretch this way.
- **A null target** — the feature moments still update while the target's
  cross-moment does not, so the coefficients wander with feature noise. Use
  it only for a label that has not arrived yet.
- **`predict`** — scores every row against the bank exactly as it stands and
  touches nothing: no clock advance, no decay, `n_eff` frozen. Use it to
  serve. It is also the fast path: `ewridge` scores at 1.8–2.9× its learning
  throughput.

### Two guarantees

- **Predictions are out-of-sample.** Every row is predicted from the state
  as it stood before the row's own target was learned. Nothing here can
  leak a row's outcome into its own prediction.
- **Chunk invariance.** One chunk or a thousand, with or without a save and
  resume in the middle, the output is bit-identical. The one exception is
  `coef`, which is a *reporting* cadence: it is written every `coef_every`
  rows and on each chunk's last row, so smaller chunks report it more
  often.

## Running a bank

### As a query: `lf.online.fit_predict`

A Polars `LazyFrame` is a query that runs only when you ask for its result.
`lf.online.fit_predict(specs)` puts a model bank inside such a query. When
the query runs — `collect()` for the result as one frame, `sink_parquet()`
to write it to a file without holding it all, `collect_batches()` to get it
a chunk at a time — the rows go through a bank that starts with nothing
learned, `chunk_rows` rows at a time, and whatever comes after the bank in
the query is ordinary Polars:

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

Two things the comments cannot carry. **`save_state` writes when the run
reaches the last row**, whole or not at all, and the same bytes a
`ModelBank` would write. A run abandoned early, or ended by an error inside
the bank, leaves the file untouched. An error *after* the bank — in a later
step of the query — does not stop the bank, so the state is written although
the query failed ([docs/STATE-WORKFLOW.md](docs/STATE-WORKFLOW.md) has the
measurements). **Filter after the bank, not before, unless the model must
skip those rows.** A filter after the bank never changes what the bank
learns from, and the query still runs a chunk at a time. A filter *before*
the bank makes Polars hold several blocks of each parquet file in memory
per thread — 2.5 GB at 12M rows, against 0.78 GB for the same filter
after ([docs/PERFORMANCE.md](docs/PERFORMANCE.md) §11).

If the model must not learn from some rows, give them weight `0` instead of
filtering them out: they still flow through, still come out scored, and no
gap opens in the clock.

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
[`po.predict(frame, bank)`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.predict) and [`po.unnest(frame, specs)`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.unnest) are the same calls as
plain functions, for a type checker, which cannot see a registered namespace.

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

### The expression form (in memory only)

For a frame that is already in memory, the shortest way to write a model is
as a Polars expression:

```python
out = df.with_columns(
    pl.col("y").online.ewridge(
        features=["x0", "x1", pl.col("y").shift(1).alias("y_lag")],   # features may be expressions
        clock="t", halflife=600.0, max_dclock=300.0,
    ).over("group").alias("fit")            # evaluated per group: a lag never crosses a group boundary
)
```

This form loads every row of its input into memory at once — putting it
inside a lazy query does not change that — and every call warns with
[`polars_online.InMemoryExpressionWarning`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.InMemoryExpressionWarning) to say so. On a frame
that fits, that is fine, and one line says so:

```python
import warnings

warnings.filterwarnings("ignore", category=po.InMemoryExpressionWarning)
```

[`po.online(pl.col("y"))`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.online) is the same thing as a plain function.
[docs/PLAN.md](docs/PLAN.md) §6 has why this form cannot work a chunk at a
time, and the condition under which the warning would go away.

### Outside a live Python process

A scheduled job, or a deployment with no Python at all, runs the same bank
from a file to a file: [docs/RUNNER.md](docs/RUNNER.md) has `po.run` and the
standalone `online` command line.

## Memory: which calls stream

Every way of running a bank works a chunk at a time except the expression
form. Measured as the most memory the process ever held, on one file of
`ewridge` with 20 features, parquet in and parquet out:

| what you write | 3M rows | 12M rows | |
|---|---:|---:|---|
| `lf.online.fit_predict([spec])` | 0.90 GB | 1.35 GB | the bank inside a query |
| `for chunk in lf.collect_batches(): bank.fit_predict(chunk)` | 0.80 GB | 1.24 GB | your own loop |
| `pl.col("y").online.ewridge(...)` in `with_columns` | | 7.3 GB | the expression: every row at once |

[docs/RUNNER.md](docs/RUNNER.md) has the same row for `po.run` and the
command line: flat too, at 0.95 / 0.73 GB and 1.41 / 0.75 GB.

The first two do not grow with the file. What growth they show is the
memory allocator keeping pages it has freed, and nearly all of the rest is
Polars reading ahead in the parquet file (`POLARS_ROW_GROUP_PREFETCH_SIZE=1`
takes it to 0.31–0.46 GB). Everything after the bank in a query — filters,
joins, group-bys, writing the result — runs a chunk at a time as Polars
itself does. That is Polars' rule for the steps *around* the bank too: a
rolling window over groups (`.over("group")` or `group_by=`) makes Polars
hold every row (6.5 GB and 1.7 GB on the same rows, against 0.25–0.28 GB
without groups), whereas a bank's `group=` keeps one set of running sums
per group and grows with the number of groups, not the number of rows.
[docs/PERFORMANCE.md](docs/PERFORMANCE.md) §11 has every measurement.

## Saving, loading and serving

What a bank has learned — the running sums of every (spec, group) — can be
saved to one file, written whole or not at all, and loaded back. The same
two words, `save_state` and `load_state`, work from a bank object and from
a query:

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

The file-to-file runner and its command line ([docs/RUNNER.md](docs/RUNNER.md))
read and write the same file with the same two words, and the bytes are the
same whichever wrote them. Loading names the problem it hits:
`FileNotFoundError` when there is no file yet, `ValueError` for a file that
is not a bank, was written by a newer version, or holds a different model.

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

[docs/STATE-WORKFLOW.md](docs/STATE-WORKFLOW.md) walks the whole workflow —
fit, save, serve, learn on — with what each step guarantees.

### A state file describes itself

A saved bank can be read by something that knows nothing about it. No specs,
no configuration, no data — the file carries what it needs:

```python
bank = po.ModelBank.load("bank.state")   # no specs=: the file is enough

bank.specs                # every spec back, as the dict its builder made -- a copy, read-only:
                          # the bank runs the state it was built from, so a list edited on the
                          # Python side would only ever mislabel what coef() reports
bank.groups()             # spec, group, rows_processed, last_clock
bank.output_fields()      # {'ridge': ['pred_y', 'resid_y', 'n_eff', 'coef'], ...}
bank.rows_seen()          # rows fed, over every chunk and group
bank.solve_failures()     # per spec, per group

# Four more tables: how the fit is doing, and what it was trained on.
# Each returns every spec by default with `spec` as the first column, and the columns are the
# same for every spec, so banks from different runs stack with a plain concat.
bank.last_row()           # the output row of the last row each group learned from      (The last row, below)
bank.coef()               # one row per coefficient, with the term it belongs to         (Coefficients, below)
bank.summary()            # per group: rows fed, learned, skipped, and the clock's range  (What it was fed, below)
bank.describe()           # per input column per group: count, nulls, mean, std, min, max
```

### Reading a state without this library

```python
text = bank.to_json()          # everything save() writes, as JSON: look at a state, compare two, hand one to a non-Python program
bank.save_json("bank.json")    # the same, to a file
```

It is an export, not a second format — `load` reads the binary form only —
and it is faithful, including the values JSON has no way to write. `NaN`
and `±inf` are written as the strings `"nan"`, `"inf"` and `"-inf"`, the same
spelling a spec's `halflife` already uses. That matters more than it
sounds: `halflife=inf` means no forgetting, so an ordinary state carries an
infinity, and a plain JSON encoder writes it as `null` without saying so.
Every export is read back and checked against the state before you get it,
so a state that could not be carried is an error rather than a file that is
quietly wrong.

## Preparing a stream

Two things a stream may need before a bank sees it: a target that is not
known at the row it sits on, and several series that do not tick at the
same moments.

### Labels that arrive late

A target that is a forward quantity — the next five minutes' return, the
next day's fill rate — is not known at the row it sits on. A stream that
learns it there hands the model that much of the future before it predicts
the rows in between. Every "out-of-sample" number after that is
contaminated, and with a feature that is correlated with itself over time
even a pure-noise column starts to look predictive.

`label_delay` is the fix, and it is one parameter:

```python
spec = po.spec.ewridge("fwd", targets=["ret_5m"], features=["x0", "x1"],
                       clock="ts", max_dclock=3600.0, halflife=1800.0,
                       label_delay=300.0)     # the return takes 5 minutes to be known
# Each row is scored where it sits and learned from 300 clock units later. Everything
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

[`po.prep.embargo`](https://hgilde.github.io/polars-online/prep.html#polars_online.prep.embargo) writes the same thing out as data, for when the delay
has to be visible in the frame, or for an engine other than this one:

```python
doubled = po.prep.embargo(lf, clock="t", delay=300.0)   # every row twice: a zero-weight copy to score at t,
                                                          # and a copy to learn from at t + delay, in clock order
```

The built-in path is tested against it field by field and agrees to the
bit, with three exceptions: `resid_quantiles`, `emit_autocorr` and
`emit_drift`. Those three take no row weight, so a zero-weight row feeds
them as much as its learning copy does — in a doubled stream every residual
therefore lands twice, where `label_delay` feeds them once.

### Series that tick at their own times

Two series observed at different instants cannot be correlated directly. A
fine common grid pushes the correlation towards zero (the Epps effect), and
filling forward invents observations that were never made.

[`po.prep.refresh_time`](https://hgilde.github.io/polars-online/prep.html#polars_online.prep.refresh_time) puts them on the grid Barndorff-Nielsen, Hansen, Lunde
and Shephard defined: a point wherever **every** series has ticked at least
once since the last point, each carrying its last observed value.

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
of its ticks; `retained_fraction` says how much. `pairs=True` runs an
independent two-series grid per pair instead, which keeps far more when one
series is slow. Rows must be in time order; a backwards time is an error
naming the row, and nothing is interpolated. The output looks synchronous
and is not: each value is up to one of its own inter-tick intervals old,
and the series with the largest `n_obs` is the one holding the grid up.

## Reading the fit

What a bank can tell you about its fit — and about what it was fed — with
no data at hand, and the grammar of the field names it writes.

### Coefficients

Two ways, and they agree row for row:

```python
ols = po.spec.ewridge("ols", targets=["y"], features=["x0", "x1"], clock="t",
                      halflife=600.0, max_dclock=300.0, group="stock_id",
                      coef_every=1)         # write coef on every row (default 0: on each chunk's last row only)

# 1. From a bank -- live, or loaded from a state file with no data at hand.
bank = po.ModelBank([ols])
bank.fit_predict(df)
betas = bank.coef()                  # one row per coefficient: spec, group, instance, n_eff, ..., term, coef
                                     # -- the fit as of the last row each group learned from
wide = betas.pivot("term", index=["group", "instance"], values="coef")

# 2. From the output, as columns: the fit as it moved, one row per row.
path = (
    lf.online.fit_predict([ols])
    .online.unnest([ols])            # pred_y, resid_y, n_eff, coef_y_intercept, coef_y_x0, coef_y_x1
    .select("t", "stock_id", "^coef_.*$")
    .collect()
)
```

The output's `coef` is written *after* each row's update; the row's own
`pred` comes from the fit *before* it. With `coef_every=1` that is a list
of `k` floats on every row of the output.

Under a grid — several `ridge` values, `feature_sets`, a `lasso_path`,
several targets — the list holds one block per (target × grid point).
`unnest` names each block's columns the way the `pred` fields are named
(`coef_y_x0__r0.5@h500` beside `pred_y__r0.5@h500`), `bank.coef()` carries
the same columns to tell blocks apart (add them to the pivot's `index`), and
`unnest` reads a saved output the same way:
`pl.scan_parquet("fitted.parquet").online.unnest([ols])`. It takes the specs,
a bank, or the path of a saved state.

### The running sums behind a fit

`bank.gram(spec)` hands back the matrices the model itself solves against,
per group and per halflife. They are a *complete* summary of the rows the
model has seen — in the statistical sense, a sufficient statistic — so a
saved state answers questions the run never asked:

```python
ols = po.spec.ewridge("ols", targets=["y"], features=["x0", "x1", "x2"],
                      halflife=500.0, ridge=1e-9, standardize=False)
fitted = po.ModelBank([ols])
fitted.fit_predict(df)

g = fitted.gram("ols")[0]      # one dict per (group, halflife); needs numpy, an optional extra
g["means"], g["comoments"]                # the feature means, and the centred k x k co-moment matrix
g["cross_moments"], g["target_weights"]   # per target: the uncentred E[z*y] the solve consumes, and the weight behind it
g["target_means"], g["target_vars"]       # per target: the target's own mean and centred variance
g["n_eff"], g["n_kish"], g["target_n_kish"]   # the accumulated weight, and Kish's effective sample size (features, and per target)

# The algebra the model runs, done by hand:
raw = g["comoments"] + np.outer(g["means"], g["means"])   # the solve's pairing
beta = np.linalg.solve(raw, g["cross_moments"][0])
slopes = beta[1:]                                          # column 0 is the intercept
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

`n_eff` counts weight, not rows. `n_kish = n_eff² / Σw²` is the number of
equally weighted rows the moments are worth, which is what a standard error
divides by; an exponentially weighted window of unit rows settles at `(1 +
λ)/(1 − λ)` whatever the halflife's units. `n_kish` does not depend on the
scale of the weights: multiply every weight by the same factor and it does
not move, so it does not fall when a stream goes quiet. `n_eff` is the
number that falls, and the one to read for that.

The target moments are the half that makes the rest usable: without
`Var[y]` there is no residual variance, no R², no information criterion and
no standard error to be had from a saved Gram. A state saved by 0.1.x has
no `Σw²` and no target moments, and they cannot be recovered from what it
does have — those four keys are `None` there, for that state's whole
remaining life.

`po.gram` is the same arithmetic the models run, so `solve` on a spec's Gram
is that spec's fit and `lasso_path` is that spec's path — not the same
arithmetic to the last bit, because the models factorize with one library's
Cholesky and numpy with LAPACK's LU, which round differently in the last
place or two. Two things to know before reading a disagreement as a bug:
`bank.coef()` is as of the model's last *solve*, which its `solve_every`
schedule decides, while `gram()` is as of the last row. And `merge` pools
parts that share a weighting — pieces of one pass, groups being combined —
not two halves of a decayed stream in time order, where each part's weights
are relative to its own last row; the docstring gives the rescaling for
that case.

### One row per finished group

A bank keeps one state per group key, for the life of the bank. On a stream
whose key space keeps growing — a day id, a session id, a block number — that
is unbounded memory for state nobody will read again.

`group_close` says when a group is finished. The bank then writes its
running sums out as one row and frees the group's memory.

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
An integer key column is ordered as numbers; anything else is ordered as
text, so `"9"` comes after `"10"`. Sort by the same rule the bank reads, or
the chunk is refused: a `Categorical` column sorts by the order its
categories were first seen by default, so sort it with
`pl.col("k").cast(pl.String)` or cast the column itself. `"session"` closes a
group where its `session` value changes. Either way the last group never
closes — nothing proves it is finished — and stays readable through
`gram()`.

A run writes the same rows to a sidecar file, which with no other output at
all is the whole shape of an accumulate-only pass — read a stream that does
not fit in memory, write one row per block:

```python
for _ in by_block.lazy().online.fit_predict([blocks], closed_groups="blocks.parquet").collect_batches():
    pass   # the per-row output is not wanted; only the sidecar file is
```

(`by_block` and `blocks` are from the block above.) `po.run` and the command
line write it too ([docs/RUNNER.md](docs/RUNNER.md)). What has closed and
not been read is saved with the state, so a driver that saves between chunks
does not lose rows silently.

### Reading a correlation matrix

`po.gram` solves and diagnoses a design matrix. `po.corr` is its
complement: the arithmetic that comes *after* a correlation matrix, in the
same style — numpy only, pure functions, each held against the paper it
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

The rest of the module — `block_means` / `from_blocks` (mean correlation
within and between labelled blocks, and back), `mp_density`,
`signal_share` (how much of a correlation's movement across blocks is not
the sampling floor), `loss` (`qlike`, `z_mse` or Engle–Colacito `minvar`),
`epps_invert` (the correlation at a coarser scale from `ew_cov`'s lagged
co-moments), `shift` and `equicorr_loglik` — is in the [API
reference](https://hgilde.github.io/polars-online/corr.html).

### The last row

The state file also carries the output row of the last row each group
learned from, so a saved model says how it was doing without its output
frame:

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
```

A group that has not learned from a row yet, or a file written by 0.1.x,
gives a row of nulls, and `predict` does not move the row.

### What it was fed

The state file also carries what each group has seen, so a saved model can
say what it was trained on without the data at hand:

```python
bank = po.ModelBank.load("bank.state", specs=[spec])
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

A label column has counts only, and a model with no target lists its
features. Neither table forgets: they are plain counts over the whole
stream, computed in row order, so they are the same whatever the chunking,
to the bit, and `predict` does not move them. A file written before 0.2.0
reports nulls for both — a count that began partway would read as the whole
history — while `rows_processed` and `last_clock`, which the stream always
kept, are filled in.

### Output field names

You index the result column by field names, so the names are a contract.
The grammar:

```
pred_{target}{combo}{instance}     combo    = ""            single ridge, no feature sets
resid_{target}{combo}{instance}             | __r{ridge}     ridge grid
sigma_{target}{combo}{instance}             | __{set}        feature sets, single ridge
absresid_q{level}_{target}...               | __{set}_r{ridge}
n_eff{instance}                    instance = ""            single halflife
coef{instance}                              | @h{halflife}   halflife grid
```

Numbers render as plain decimals in `[1e-6, 1e7)` and as compact scientific
outside it. Every name, default and signature is pinned against a checked-in
snapshot, so a change is a reviewable diff and a version bump, never a silent
rename of your columns.

You never have to build these strings:

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
cannot drift from the strings, and the index carries each field's `dtype` —
the column types the bank declares to Polars before the first row is read.
One sharp edge: avoid `__` and `@` in target names and feature-set labels if
you parse field names downstream, since a target named `y__r0.5` renders
like a ridge grid on `y`.

## Diagnostics, selection and evaluation

Outputs you switch on. All are computed from what the models have already
learned and read *before* each row, so none of them see the row they
describe — they are as out-of-sample as the predictions:

```python
diag = po.spec.ewridge(
    "diag", targets=["y"], features=["x0", "x1"], clock="t", max_dclock=300.0, halflife=500.0,
    ridge=[1e-6, 0.1],           # a grid, so there is something to select among
    emit_sigma=True,             # sigma_<slot>:     EW standard deviation of that slot's out-of-sample residuals
    emit_resid_z=True,           # resid_z_<slot>:   resid / sigma -- how surprising the row was, in units of recent error
    emit_selected=True,          # selected_<t>, pred_<t>__selected: the ridge value, feature set or halflife with the
                                 #                   lowest EW out-of-sample error so far
    emit_averaged=True,          # pred_<t>__averaged: softmax(-eta * EW error) blend over the same choices -- hedges
                                 #                   where emit_selected commits
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

`emit_metrics` on a `sgd` or `ftrl` fit with `loss="logistic"` reads
differently, because `pred` is a probability and `y` a 0/1 label rather than
a signed target: `hit_rate` is accuracy at a 0.5 threshold, not sign
agreement (the sign test always agrees on two positive numbers), `r2` is
the Brier skill score against the running base rate, and `ic` the
point-biserial correlation between the probability and the label, both
under their usual names. There is no streaming log loss (a logarithm cannot
go into the state without the result differing in the last bits between
operating systems — see `docs/PLAN.md` §11a);
`po.eval.metrics(..., binary=True)` adds it over a collected frame instead.

Break detection complements forgetting rather than replacing it: forgetting
is smooth and always on; a detector notices a break and says so, within a
couple of rows of a sign flip.

`conformal` is the interval to use when the residuals are not Gaussian. It
tracks the `coverage` quantile of `|resid|` directly: the radius grows by
`conformal_rate · sigma · coverage` on a miss, and shrinks by
`conformal_rate · sigma · (1 − coverage)` on a hit. So its long-run coverage
is the number you asked for, whatever the residuals do, with an error that
shrinks like `1/T`. `sigma` gives a Gaussian interval instead. On Gaussian
residuals the two agree; on fat-tailed or heteroskedastic ones the Gaussian
interval over-covers by several points where this one lands on target. The
conformal radius starts at `sigma · Φ⁻¹(1 − α/2)` and is null until then.
It adds three numbers to the state per slot.

```python
ci = po.spec.ewridge("ci", targets=["y"], features=["x0", "x1"], clock="t",
                     max_dclock=300.0, halflife=500.0, conformal=0.9)
band = po.ModelBank([ci]).fit_predict(df).unnest("ci")
held = band.select(((pl.col("lo_y") <= df["y"]) & (df["y"] <= pl.col("hi_y"))).mean())   # the realized coverage
```

After the fact, `po.eval` reads the output frame:

```python
po.eval.metrics(out, "ridge", by=["stock_id"])                       # R², IC, hit rate, MSE
po.eval.rolling_metrics(out, "ridge", clock="t", window=3600.0)     # the same, per clock window
po.eval.compare_specs(out, ["ridge", "kalman"])                     # one table, many specs: which had the lower error
po.eval.seqtest(out, a="kalman", b="ridge", by=["stock_id"])        # is kalman closer? evidence per row -- the same test
                                                                    # the seqtest model runs inside a bank
```

Those four all need the whole frame. When the output is never held in one
place — fifty slots over a billion rows — reduce each chunk instead and keep
ten numbers per key:

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

The sums are **centred** (weighted means and centred second moments, merged
with a parallel-axis term) rather than raw `Σy` and `Σy²`: a target sitting
around 1e8 with unit spread destroys the raw form's variance entirely, and
this one does not notice. `weight=` names a column to weight the rows by.

### Data whose truth is known

A regime detector is a claim about a stream, and a claim needs a stream
whose answer is written down. [`po.sim.regimes`](https://hgilde.github.io/polars-online/sim.html#module-polars_online.sim) produces one from a seed,
with the awkward parts included — series that tick at their own times,
prices observed with noise, returns correlated with their own past, a
volatility that moves with the regime, an intraday pattern and a volume
clock — and hands back the truth beside the data.

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

`durations` makes each state last exactly as long as it says;
`design="smooth"` interpolates the matrix across a boundary instead of
stepping.

## Models

One section per model: what it is for, its update rule, and the parameters
that are its own. The parameters every model shares — the clock, decay,
groups, weights and warm-up — are in [How a bank sees a
stream](#how-a-bank-sees-a-stream) and are not repeated here.

The running sums a model keeps are its *accumulators*. All of them are
exponentially weighted **means**, not sums, so they stay bounded over a
stream of any length; second moments are kept **centred** (a weighted
Welford update), so a variance is right even when a feature sits far from
zero. In the update rules, `z` is `[1, x]` when there is an intercept, `w`
the row's weight, `λ` the row's decay, and `W` the total weight so far.

Each model links to its builder in the API reference, whose docstring lists
every keyword with its default, and to its Rust source under
`crates/online-core/src/`, where the module comment states the recursion.

### `ewridge` — EW ridge on sufficient statistics

*API:* [`po.spec.ewridge`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.ewridge) — *Rust:* [`ewridge.rs`](crates/online-core/src/ewridge.rs) — *Outputs:* [fields](docs/OUTPUTS.md#ewridge)

Ridge regression on running sums. The sums are updated on every row; the
coefficients are solved from them on a schedule.

```
W'   = λW + w                       S' = (λW·S + w·z zᵀ) / W'
W_j' = λW_j + w                     r_j' = (λW_j·r_j + w·z·y_j) / W_j'
solve:  (S + ridge·D) β_j = r_j     D = I minus the intercept slot
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
)
```

Each row costs O(k²) for `k` features, to update `S`; the solve is a
Cholesky factorization.

With decay off and `ridge=0` this is ordinary least squares over every row
seen, in any row order: the coefficients match `numpy.linalg.lstsq` to 2e-13
forwards, backwards or shuffled, and 6M rows × 20 features from a parquet
stream peak at 1.4 GB against 3.97 GB for `lstsq` on the same rows. One
trap in that setting: the solve schedule defaults to `halflife/50`, so
`halflife=1e12` solves once, at `min_periods`, and never again — say `inf`,
or set `solve_every`. `solve_every=1000` on that stream takes 1.4 s instead
of 11 s, with coefficients at most 1000 rows out of date.

At a thousand features the O(k²) update of `S` is most of the cost:

```python
wide = po.spec.ewridge(
    "wide", targets=["y"], features=["x0", "x1", "x2"], halflife=float("inf"), solve_every=1e9,
    max_rows_between_solves=1000,
    gram_block_rows=256,           # hold 256 rows back and add them to S with one matrix product instead of
)                                  # 256 single-row updates; refused with window=, and where a solve happens every row
```

Measured on one thread, that is 5.1× the rows per second at 256 features,
6.6× at 1,000 and 5.9× at 2,000; a solve every 512 rows brings each down to
about 4×, because the solve costs the same either way. `n_eff`, the timing
of every prediction and chunk invariance do not change. The coefficients
agree with the row-by-row fit to rounding, not to the bit: the blocked sum
is the same sum in a different order. The held rows travel in the state
file, so a save mid-block resumes on the same block. `docs/PERFORMANCE.md`
§18 has the table.

### `rls` — recursive least squares

*API:* [`po.spec.rls`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.rls) — *Rust:* [`rls.rs`](crates/online-core/src/rls.rs) — *Outputs:* [fields](docs/OUTPUTS.md#rls)

```
A ← λA + w zzᵀ       b_j ← λb_j + w y_j z        β_j = A⁻¹ b_j
A₀ = ridge·I         b₀ = ridge·coef_prior
```

The coefficients move on every row; there is no solve schedule and so
nothing is ever out of date. What the model stores is the Cholesky factor
of `A`, updated row by row with Givens rotations — the *square-root form*
of the recursion. That costs O(k²) per row, the same as the textbook
recursion on the inverse `P`, and avoids both of that form's failures: `P`
loses symmetry to rounding by a factor of `1/λ` per row, and one extreme
row can cancel it and freeze a coefficient for good. The result is the
same as `ewridge(ridge_decay=True)` solved on every row, to better than
1e-9.

```python
rls = po.spec.rls(
    "rls", targets=["y"], features=["x0", "x1"], clock="t", max_dclock=300.0, halflife=600.0,
    ridge=1e-3,                    # A starts at ridge * I -- unlike ewridge, this penalizes the intercept too
)                                  # a row with any null target is scored but not learned from, for every target,
                                   # because the factor is shared between them
```

### `lasso` — lasso path with free λ selection

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
)
```

### `kalman` — random-walk-β dynamic linear model

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

Why revert: a regressor that is only occasionally active is then forgotten
between its bursts rather than kept at its last value, and a stale effect
cannot persist through a run of null targets. The reversion acts in the
standardized coordinates, so "zero" means "no effect" for a slope and "the
target averages zero" for the intercept. The long-run prior variance of a
reverting slot is `q_i·Δclock/(1−φ_i²)` instead of growing without bound.
`predict` moves the coefficients by the same `Φ` over the distance from the
last learned row (capped by `max_dclock`), so a prediction far past the
data is the intercept alone.

### `huber` / `quantile` — robust regression

*API:* [`po.spec.huber`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.huber) and [`po.spec.quantile`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.quantile) — *Rust:* [`robust.rs`](crates/online-core/src/robust.rs) — *Outputs:* [huber](docs/OUTPUTS.md#huber), [quantile](docs/OUTPUTS.md#quantile)

Iteratively reweighted least squares on the ridge update, using each row's
residual *before* the row is learned, so the reweighting stays
out-of-sample. Huber: `w = min(1, δσ/|r|)`. Quantile at level τ: the check
loss's own weight, `2τσ/|r|` above the fit and `2(1−τ)σ/|r|` below it, with
`|r|` floored at `quantile_eps · σ` so a residual near zero cannot blow the
weight up. Weights are per target, so the running sums are per target
here.

```python
hub = po.spec.huber("hub", targets=["y"], features=["x0", "x1"], clock="t", max_dclock=300.0, halflife=600.0,
                    huber_delta=1.5)      # a residual beyond huber_delta * sigma is down-weighted
med = po.spec.quantile("med", targets=["y"], features=["x0", "x1"], clock="t", max_dclock=300.0, halflife=600.0,
                       quantile=0.5,      # the level: 0.5 is a median regression
                       quantile_eps=0.05) # |r| is floored at quantile_eps * sigma, so a residual near zero cannot blow the weight up
```

### `sgd` — stochastic gradient descent

*API:* [`po.spec.sgd`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.sgd) — *Rust:* [`sgd.rs`](crates/online-core/src/sgd.rs) — *Outputs:* [fields](docs/OUTPUTS.md#sgd)

One gradient step per row, no solves:

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

O(k) per row — the cheap baseline, and the only model here that takes
count targets (`loss="poisson"`).

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

### `pa` — passive-aggressive regression

*API:* [`po.spec.pa`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.pa) — *Rust:* [`pa.rs`](crates/online-core/src/pa.rs) — *Outputs:* [fields](docs/OUTPUTS.md#pa)

```
loss = max(0, |y − p| − eps)      s = ‖z‖²
pa    τ = loss / s          pa1  τ = min(c, loss/s)      pa2  τ = loss / (s + 1/(2c))
β    += τ · sign(y − p) · z
```

Each row asks the fit to come within `eps` of its target, and the update is
the smallest change that does so — there is no learning rate to tune. PA
keeps no running sums, so its coefficients have no halflife; the clock only
drives `n_eff`.

```python
pa = po.spec.pa(
    "pa", targets=["y"], features=["x0", "x1"], halflife=200.0,
    mode="pa1",                  # the default: the step is capped at c. Plain "pa" moves the fit as far as one bad
    c=0.1,                       # row demands; "pa2" damps the step by c instead of capping it
    eps=0.05,                    # the row is "close enough" inside this margin, and nothing moves
)                                # a row weight below 1 scales the step; above 1 it counts as 1
```

`pa` takes the same `coef_min`, `coef_max` and `coef_sum` as `sgd`, with
the projection applied after each update. The step then no longer meets
the row's margin exactly, and a truth outside the allowed set is never
reached, so keep `c` small: each row moves the fit only as far as `c`
allows and the projection takes the rest back.

### `ew_cov` — exponentially weighted moments

*API:* [`po.spec.ew_cov`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.ew_cov) — *Rust:* [`ewcov.rs`](crates/online-core/src/ewcov.rs) — *Outputs:* [fields](docs/OUTPUTS.md#ew_cov)

```
W'   = λW + w        m'ᵢ = (λW·mᵢ + w·xᵢ) / W'      S'ᵢⱼ = (λW·Sᵢⱼ + w·xᵢxⱼ) / W'
varᵢ = Sᵢᵢ − mᵢ²     covᵢⱼ = Sᵢⱼ − mᵢmⱼ             corrᵢⱼ = covᵢⱼ / √(varᵢ·varⱼ)
```

Running moments of the columns you name, on the same clock as every model
here — one O(k²) update per row, where computing every pairwise
exponentially weighted correlation with Polars expressions alone takes
O(k²) *passes over the data*. Values are read from the state before each
row, so an `ew_cov` output can be a feature for that same row without
leaking it.

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

`mahal` is `√(δᵀ (C + s·prior·I)⁻¹ δ)` with `δ = x − m` — how far the row
is from what the columns have been doing *together*, in standard
deviations; on Gaussian columns `mahal²` is χ² with `k` degrees of freedom,
and with one column it is `|z|`.

**`window` — a hard cutoff, not a softer decay.** A halflife of `h` never
forgets entirely: three halflives back still carries 12.5% of the weight.
`window=w` makes that exactly zero — a row older than `w` clock units
contributes nothing:

```
weight(age) = 0.5 ** (age / halflife)   if age <= window
            = 0                          otherwise
```

Note the first line: *inside* the window the weights are still exponential,
so this is not a flat rolling mean and the newest row still dominates. It is
exact, not approximate, because an exponentially weighted sum contains its
own past — everything at or before a time `u` is `λ^(t−u)` times the running
sum as it stood then, so subtracting that leaves precisely the rest. The
model keeps a ring of past snapshots to do it, which is the one place here
where memory grows with a *window* rather than with the state: about 3 MB
per group for a 1,000-row window over 20 columns, divided by `window_every`
if you snapshot less often.

```python
cut = po.spec.ew_cov(
    "cut", features=["x0", "x1"], clock="t", max_dclock=300.0, halflife=500.0,
    window=1500.0,               # a row older than this many clock units contributes exactly nothing
    window_every=10,             # snapshot every 10 rows: the boundary is the oldest snapshot still inside the window,
)                                # so a coarse cadence discards a little more than asked, never less
```

Four things worth knowing before you read the numbers. The clock is the
**decayed** one, after `max_dclock` and any `session_gap`. The edge is a
**discontinuity** — a row ageing out drops its whole weight at once, so the
series has small steps a plain exponential average does not. It is a
**subtraction**, so precision falls with the fraction discarded: negligible
at `window = 3h`, worse as the window shortens toward the halflife. And
`n_eff` becomes the weight inside the window, so `min_periods` now gates on
something that stops growing, and a clock gap longer than `window` empties
it and reports nulls rather than stale numbers.

If your data fits in memory and you only want moments, Polars already does
this: `df.rolling("t", period="3h").agg(...)` with an exponential weight is
the same number to 1e-14. The reason to reach for the spec is a stream, a
saved state, or the time: the rolling window recomputes each window at
`O(n·W)` where this is `O(n)`. At 200k rows and a 4,680-row window that is
15.8 s against 14 ms.

**`lags` — how a column moves with another `ℓ` rows ago.** The same
co-moments, kept one step further out; with `W` and `m` the weight and mean
before the row, and both deviations taken against that mean,

```
C_ℓ' = a·C_ℓ + a·b·(x_t − m)(x_{t−ℓ} − m)'
```

with the same `a` and `b` the co-moments use — so lag 0 would be
`comoments` exactly.

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
beyond `max_dclock` — the two events after which "the row `ℓ` back" no
longer means a row `ℓ` ago — and a zero-weight row ages the matrices without
entering it. Nothing else moves: clearing the ring is not a reset.

### `ftrl` — online logistic regression

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

### `holt` — Holt's linear trend

*API:* [`po.spec.holt`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.holt) — *Rust:* [`holt.rs`](crates/online-core/src/holt.rs) — *Outputs:* [fields](docs/OUTPUTS.md#holt)

The one model that takes no features: it extrapolates the target's own
level and trend.

```
pred     = l + b·Δt
l' = α·y + (1−α)·pred        b' = β·(l' − l)/Δt + (1−β)·b
```

```python
baseline = po.spec.holt(
    "baseline", targets=["y"], clock="t", max_dclock=600.0,
    level_halflife=200.0,        # α: how fast the level follows the target, in clock units
    trend_halflife=2000.0,       # β: how fast the trend follows the level's movement; inf pins the trend at zero,
)                                # leaving a plain exponentially weighted level. coef is [level, trend] per target
```

The trend is per clock unit, so an irregular clock extrapolates the right
distance. There is no seasonal term, because a seasonal index is a `group`
on the phase, which the bank already does. Run it in the same bank as the
real model to answer "how much is the regression actually adding?" —
compare `sigma`, or let `emit_selected` choose.

### `kmeans` — exponentially weighted k-means

*API:* [`po.spec.kmeans`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.kmeans) — *Rust:* [`cluster/kmeans.rs`](crates/online-core/src/cluster/kmeans.rs) — *Outputs:* [fields](docs/OUTPUTS.md#kmeans)

The one model with no target: it labels each row with the nearest of `k`
centres, read before the row is learned from, so the label is out-of-sample
like every prediction here.

```
j*   = argmin_j ‖x − c_j‖²          distances in units of each feature's EW sd
n'_j = λn_j + w                      c'_j = c_j + (w/n'_j)(x − c_j)     for j = j*
```

Each centre is the exponentially weighted mean of the rows assigned to it —
`ew_cov`'s mean recursion, per cluster.

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

A row far outside its cluster (about four standard deviations of `dist²`
above the typical radius) is scored but not learned from: it is set aside
for the split–merge move. Raise `dead_frac` when regimes change faster
than the fade allows; the price is that a cluster lighter than
`dead_frac/k` of the stream loses its centre whenever any row is far. What
the move cannot see is one centre owning two blobs, whose rows are all
within its own radius — seeding with `lloyd` is what prevents it.

### `micro` — density-based clustering, any shape

*API:* [`po.spec.micro`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.micro) — *Rust:* [`cluster/micro.rs`](crates/online-core/src/cluster/micro.rs) — *Outputs:* [fields](docs/OUTPUTS.md#micro)

`kmeans` needs `k` and finds round clusters. `micro` finds clusters of any
shape, does not need their number, flags the rows that belong to none, and
follows clusters that appear and vanish. It is DenStream's micro-clusters
with a linking step over them.

A summary is a small cluster: a decayed weight `n`, a centre `c` and a
radius `r`, the exponentially weighted root-mean-square distance of its
rows from the centre. Each row goes to the nearest summary that can take it
without its radius passing `eps`, in units of each feature's exponentially
weighted standard deviation. If none can, the row opens one.

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

All are read before the row is learned from. A label is the smallest id in
its chain, so it outlives everything but that summary.

**Both ways to get `eps` wrong show in the outputs.** If nearly every row
is an `outlier` and `cluster` stays null, `eps` is too small: no summary
reaches `beta_mu` before it is pruned. If `n_micro` is about the number of
clusters, `eps` is too coarse: each cluster is one summary, so the derived
`L` reads the spacing *between* clusters and bridges them into one. Lower
`eps`, or set `macro_link=2`.

Measured at 20k rows with the `eps` above: moons, rings and five Gaussians
in twenty dimensions all score ARI 1.000 against the truth, where `kmeans`
cannot follow the first two. Noise drawn uniformly over the box is flagged
`outlier` 94% of the time, real rows 0.3%. A cluster born mid-stream has a
label within 200 rows; one whose rows stop lingers `halflife · log2(n /
beta_mu)`, with `n` the weight it had.

### `ew_class` — Gaussian classification on `ew_cov` moments

*API:* [`po.spec.ew_class`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.ew_class) — *Rust:* [`ewclass.rs`](crates/online-core/src/ewclass.rs) — *Outputs:* [fields](docs/OUTPUTS.md#ew_class)

A label column in place of a numeric target. The model keeps one `ew_cov`
state per class — a weight `n_c`, a mean `μ_c` and a centred covariance
`C_c` — and scores a row by Bayes' rule over Gaussian classes.

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

All are read before the row is learned from, so a row's probabilities never
saw its own label — which is also how a stream whose labels arrive late is
scored: null the label, keep the features.

**Choosing the shape.** `"full"` is the general case and costs one `k×k`
Cholesky per class per row. `"shared"` factorizes once per row, and is the
right model when the classes differ in location but not in spread — it then
matches `"full"` to a fraction of a percent on the test data, with fewer
parameters to learn. `"diagonal"` is the cheapest and cannot see a
correlation: two classes with the same marginals and opposite correlations
are one class to it. Measured at 400k rows, six features and three classes:
0.9M rows/s full, 1.8M shared, 5M diagonal. On three Gaussian classes with
their own covariances the accuracy sits within 0.001 of the best any
classifier could do given the generating parameters, and the probabilities
are calibrated to about 0.01.

### `seqtest` — a sequential test of a sign, by betting

*API:* [`po.spec.seqtest`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.seqtest) — *Rust:* [`seqtest.rs`](crates/online-core/src/seqtest.rs) — *Outputs:* [fields](docs/OUTPUTS.md#seqtest)

Not a regression. A `seqtest` asks whether a column tends to be positive;
with `a` and `b`, it asks instead whether one spec of the bank predicts
closer than another. Either way the answer is evidence you can read at any
row, as often as you like, and act on the first time it is enough. A
p-value cannot be used that way, because checking it repeatedly inflates
its error rate. An *e-process* can, and that is the whole reason to reach
for it. Per target it keeps the wealth of two gamblers, one betting that
the next sign is positive and one that it is negative. Each stakes the
Krichevsky–Trofimov fraction set by the counts so far, and never bets
against its own lead:

```
s = sign(y)                    n⁺, n⁻: the signs counted before this row,  n = n⁺ + n⁻
λ⁺ = max(0, (n⁺ − n⁻) / (n + 1))          λ⁻ = max(0, (n⁻ − n⁺) / (n + 1))
ln E⁺ ← ln E⁺ + ln(1 + λ⁺ s)              ln E⁻ ← ln E⁻ + ln(1 − λ⁻ s)
```

Under the null — given everything so far, the next sign is no more likely
positive than negative — `E⁺` is a nonnegative supermartingale, and Ville's
inequality gives `P(E⁺ ever reaches 1/α) ≤ α`. So `log_e_pos ≥ ln 20`
rejects at the 5% level however many times you looked, and however the
rows depend on each other. No distribution is assumed and the size of the
values is invisible: 60% small gains and 40% huge losses is "positive". Where
the clip never binds the wealth has the closed form `2ⁿ B(n⁺+½, n⁻+½) / π`,
the Beta(½, ½) mixture, and the bank is held to it; the two sides' average
is an e-value for the two-sided question.

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

A trial is a row, so there is no `weight` and no `halflife` — a spec that
gives them is refused — and `session` or `on_clock_reset="reset_state"`
restarts the test. `a_suffix` and `b_suffix` pick a grid instance
(`"@h500"`, `"__r0.5@h500"`). A comparison inside a bank is chunk-invariant,
saved with the state and works a chunk at a time like everything else;
[`po.eval.seqtest`](https://hgilde.github.io/polars-online/eval.html#polars_online.eval.seqtest) is the same computation over a frame you already
have.

### `marginal` — every pair's moments, kept in the state

*API:* [`po.spec.marginal`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.marginal) — *Rust:* [`marginal.rs`](crates/online-core/src/marginal.rs) — *Outputs:* [fields](docs/OUTPUTS.md#marginal)

A `marginal` is not a regression and not a joint fit. It keeps the
exponentially weighted moments of each (feature, target) pair on its own,
as if every pair were a two-column `ew_cov`. For `p` features and `T`
targets that is O(p·T) per row; one `ew_cov` over all the columns would be
O((p + T)²).

Per target `t`, on a row where `y_t` is present, with `W_t` the weight
behind that target before the row:

```
W'_t = λW_t + w        a = λW_t / W'_t        b = w / W'_t        Q'_t = λ²Q_t + w²
S'_yy = a·S_yy + a·b·(y_t − m_y)²             S'_xx = a·S_xx + a·b·(x_j − m_x)²
S'_xy = a·S_xy + a·b·(x_j − m_x)(y_t − m_y)    m' = m + b·(value − m)
```

That is `ew_cov`'s arithmetic. A pair's correlation is the one an `ew_cov`
over the two columns would report, to the bit. A null target ages its own
pairs (`W_t ← λW_t`) and learns nothing for them. A null feature skips the
row, as everywhere. Nothing is written per row but `n_eff`: the pairs are
the state, read back as a table.

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

`corr`, `beta` and `t` are null until the target's `W_t` reaches
`min_periods` (default 3; two rows give ±1 whatever the data), and where
they are undefined — a constant feature, or `n_kish ≤ 2` for `t`. A bank
loaded from a file reports the pairs the bank that saved it would. One
chunk or a thousand gives the same table to the bit.

Two views sit on top of that, both off unless asked for.

**`lags` — is `t` telling the truth?** `t` is built on `n_kish`, which is
the right count for unequal weights and says nothing about rows that
resemble their neighbours. On a smooth stream consecutive rows are nearly
the same observation, so `t` claims evidence that is not there. Lags fix
that: the pair's moments are kept at each lag too — the same statistic
`ew_cov(lags=)` computes, to the bit — and `serial_rule` turns them into
Bartlett's correction. Two *independent* AR(1) series with `φ = 0.9` and
`0.8` come out at `t = 2.39` and `t_serial = 1.03`. The first is a finding;
the second is the truth.

**`bins` — what a correlation cannot see.** Everything above is linear. A
feature can be strongly related to a target with `corr` at zero: a
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

The lagged lists are `ew_cov`'s `lagcorr` numbers exactly — the lagged
covariance over the two standard deviations, not clamped to `[−1, 1]`,
since a lagged correlation is not bounded by one in a finite sample. A row
where the target is missing ages its weight and holds the lag moments, as
it holds the pair's.

Binning costs `O(bins)` of state per pair and one binary search per pair
per row, which is why it can run across ten thousand columns in the pass
that gives them `corr`. A value that carries more than a bin's share — an
indicator's zero — fills a bin of its own and the rest share what is left,
so the 5% of rows that carry the signal are not lost among the zeros. The
warm-up rows are held and replayed, not spent: the histogram is what it
would have been had the edges been known before the first row. A feature
keeps only the bins it can support, so a binary feature has two and a
constant one has a single bin and no split. Each bin's moments are kept the
way every accumulator here is kept, so a target at `1e7` keeps its
variance. Both views ride into `bank.closed_groups()` as `pair_*` columns:
`pair_split_gain` as a list over the pairs, `pair_lagcorr_xx` and
`pair_bin_n` as lists of lists.

### `corrchange` — has the correlation structure changed?

*API:* [`po.spec.corrchange`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.corrchange) — *Rust:* [`corrchange.rs`](crates/online-core/src/corrchange.rs) — *Outputs:* [fields](docs/OUTPUTS.md#corrchange)

Two tests, because there are two questions.

`kind="monitor"` is the **closed-sample** constancy test of Wied, Krämer and
Dehling (2012), run over consecutive spans of `span_rows` rows. At the last
row of a span, per pair:

```
Q = max_{2≤j≤T} (j/√T)·|ρ̂_j − ρ̂_T| / D̂
```

with `ρ̂_j` the correlation of the span's first `j` rows and `D̂` the
delta-method long-run standard deviation of `ρ̂`. Under the null `Q`
converges to `sup|B|`, a Brownian bridge, so the critical value is the
Kolmogorov quantile — computed from the series, not pinned, and it
reproduces the published 1.3581 at 5%. That published null is the point:
the test's size and power are held to the paper's own tables (`.035` at ρ =
0 and `T = 500`, `.587` power on a `0.5 → 0.7` break), not to numbers this
implementation happened to produce.

```python
c = po.spec.corrchange(
    "break", features=["x0", "x1"],
    kind="monitor",              # the constancy test above, or "window": how *big* the change is, below
    span_rows=500,               # nothing is reported until a span closes: a delay of at most this many rows
    scalar=False,                # True: run the test on the equicorrelation of the standardised row (deco's u) --
)                                # one statistic however many columns, and a test of its level rather than of a pair
out = df.online.fit_predict([c]).unnest("break")   # stat, crit, flag, since_flag
```

The paper's own *sequential* form, with a boundary function, is Wied and
Galeano (2013), which has not been read here.

```python
w = po.spec.corrchange(
    "size", features=["x0", "x1"],
    kind="window",               # ||vech(R_pre - R_post)|| over two adjacent windows of span_rows rows each ...
    span_rows=100,
    n_perm=200, permute_every=500,   # ... against a permutation quantile: n_perm shuffles of the pooled rows between the
    perm_block=10,               # windows, in blocks of perm_block so rows that resemble their neighbours do not make the
)                                # null too liberal; or give crit= as a number and skip the permutations entirely
```

Not a sign-flip null, which a first reading of the literature suggests:
negating a whole row leaves every correlation exactly where it was. The
flag rate per row is not `alpha` for the window kind: two windows that
slide by one row are almost the same windows, so a statistic above the
quantile stays above it for a run of rows.

### `hmm` — which regime are we in

*API:* [`po.spec.hmm`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.hmm) — *Rust:* [`hmm.rs`](crates/online-core/src/hmm.rs) — *Outputs:* [fields](docs/OUTPUTS.md#hmm)

`ew_class` classifies a row against *labelled* Gaussians. An `hmm` does the
same arithmetic with no labels: the state is hidden, and a transition
matrix carries information from one row to the next. That is the
difference between "which regime does this row look like" and "which
regime are we in", and the second is usually the question.

Hamilton's filter, one row at a time, from the filtered `p` the previous
row left:

```
p1_l   = Σ_k p_k·Π_kl                      the predicted state
f_l    = N(x | μ_l, Σ_l + r_l·I)           the state's density
loglik = ln Σ_l p1_l·f_l                   the row's surprise
p_l   ← p1_l·f_l / Σ                       the filtered state
```

Everything reported is read before the row is learned from. Each state's
running sums then take the row at weight `w·p_l`. The responsibilities sum
to `w`, so `n_eff` is the shared recursion untouched — a row splits across
the states rather than counting more than once. The transition matrix is
learned from the **filtered joint of consecutive states**,
`ξ_kl = p_k(t−1)·Π_kl·f_l / Σ`, with a Dirichlet pseudo-count keeping a
never-visited row a distribution.

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

What the transition chain adds, measured: on two-dimensional blobs 1.5
apart, a memoryless nearest-centre rule *given the true centres* is 85%
right and the filter is 99%.

One limitation worth knowing: a single extreme row can be captured by one
state, and in mean form a state with zero responsibility keeps its moments —
so a state that stops winning never forgets, and the mixture is left short
one state. A larger `precision_prior`, given states, or cleaning upstream
are the mitigations.

### `rcov` — a block's realised covariance, robust to noise

*API:* [`po.spec.rcov`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.rcov) — *Rust:* [`rcov.rs`](crates/online-core/src/rcov.rs) — *Outputs:* [fields](docs/OUTPUTS.md#rcov)

A realised covariance over ticks is the sum of outer products of returns.
Over real tick data it is wrong twice: each price is the efficient one plus
a measurement error, and the error's variance accumulates with every tick;
and if the series are not observed together, the correlation is pulled
towards zero. Both are estimated away by published estimators that are sums
over lags — which is exactly what a stream can accumulate.

`rcov` has no decay and no per-row output but `n_eff`. Its value is the
block, written when the group closes, so it needs `group` and `group_close`
and the estimate rides in that row.

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
0.93. A block too short to estimate from gives nulls, not an error.
Nothing reads a future row: the jittered *end* point is formed at close
from observations already in state, and a product enters `Γ̂_h` only once
both legs are final. `weight` is taken as 0 or 1 only — a sum over returns
has no fractional row.

### `deco` — one correlation for the whole matrix

*API:* [`po.spec.deco`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.deco) — *Rust:* [`deco.rs`](crates/online-core/src/deco.rs) — *Outputs:* [fields](docs/OUTPUTS.md#deco)

A correlation matrix of `m` series has `m(m−1)/2` free entries. A stream
cannot keep them all moving without O(m²) work a row, and most of them are
estimated from too little data to be worth moving. `deco` (Engle & Kelly
2012) replaces them with their average and estimates that, in O(m) a row.

The row is standardised against the means and variances as they stood
before it, `r_i = (x_i − m_i)/√v_i`. With `S₁ = Σ r_i` and `S₂ = Σ r_i²`
over `n` features, the row's estimate is their Lemma 2.3:

```
u = (S₁² − S₂) / ((n − 1)·S₂)        = mean of r_i·r_j over i ≠ j, over mean r_i²
```

and the level follows one of two dynamics, on the model's own clock:

```
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

Two things to know. `u` is a **downward biased** estimate of the
equicorrelation — the paper says so, and it is a ratio of two averages, so
`E[u]` is about 0.20 for a true 0.30 at six columns. Use it as a signal that
moves with the market's correlation, not as the correlation. And `rho` is
not the same thing as an `ew_cov`'s `corr` over the columns; the mean of a
ratio is not the ratio of means, and the gap is large.

### `bocpd` — how long has this regime lasted?

*API:* [`po.spec.bocpd`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.bocpd) — *Rust:* [`bocpd.rs`](crates/online-core/src/bocpd.rs) — *Outputs:* [fields](docs/OUTPUTS.md#bocpd)

Every other detector here answers "has something changed?" with a
statistic. `bocpd` (Adams & MacKay 2007) keeps a probability distribution
over the **run length** — how many rows since the last break — so the
answer carries the age of the regime with it. "We are forty rows into a
regime" is different information from "something broke".

Their Algorithm 1, with `H = 1/hazard` and `π_r` run `r`'s posterior
predictive for this row:

```
growth:      P(r_t = r+1, x_1:t) = P(r_t-1 = r, x_1:t-1)·π_r·(1 − H)
changepoint: P(r_t = 0,   x_1:t) = Σ_r P(r_t-1 = r, x_1:t-1)·π_r·H
```

Each run keeps its own conjugate sufficient statistics, so slot `r` holds
exactly the `r` rows that hypothesis says came before this one in the run —
and slot 0 holds none, so its predictive is the prior's. That is what makes
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

**`run_mode` is the answer; `p_change` is the alarm.** The two are not the
same quality of signal. `p_change` is a per-row likelihood ratio, so it is
spiky, and its height depends on the size of the break against the prior
scale. A ten-fold variance step takes it to 0.83 on the row itself. A
four-sigma mean shift with a diffuse prior barely lifts it. A change in
correlation alone never moves it at all. The run length finds all three, one
to three rows later, and dates them to the right row.

It is `P(r ≤ 1)` and not `P(r = 0)` because the changepoint branch and the
growth branch share the same predictive, which makes the normalised mass at
`r = 0` *exactly* `H` on every row whatever the data. Row one of a group
reports nothing at all: `P(r ≤ 1)` is 1 there however the row looks.

`robust_beta` is a trade: a whole new regime is a run of individually
forgiven rows, so above about 0.2 nothing is ever detected again. The
default of 0.1 ignores the outlier and still dates a four-sigma shift to
the right row.

## Parallelism

The unit of work is a *stream*: one spec on one group (with no `group`, one
stream per spec). On every chunk, each stream in the bank becomes one task
on the bank's own thread pool — a pool separate from Polars' own — one flat
pool across all specs and all groups, longest stream first so a few big
groups do not leave cores idle at the end. Within a stream the rows go one
at a time, because each row's update depends on the last. That is what
makes the numbers independent of how the work is split. It also means a
bank with one spec and one group is one thread's work per chunk — Polars'
own reading and writing still run in parallel around it.

So a bank fills the pool with groups, with specs, or with both.

A search over factor sets is a list of specs, one per set. Each spec is its
own set of running sums, with its own standardization and its own grid
inside, and each is one task. A null in a factor a spec does not use costs
that spec nothing. (Subsets of one list that should share running sums are
`feature_sets` on one spec: one solve each, not one task.) The list runs as
one query in one pass, with the thread counts set before anything is
built:

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
groups that query takes 12.3 s at one thread and 2.2 s at fourteen; the
three-factor spec alone goes from 2.5 s to 0.62 s, because with one task
per group the fixed cost of reading and assembling each chunk shows
through. The output is one column per spec, which is what `compare_specs`
reads, and one state file holds them all. The same list runs the same way
through `ModelBank`, and through the file-to-file runner
([docs/RUNNER.md](docs/RUNNER.md)).

Where the parallelism comes from, then:

- **Groups.** k=20 over 64 groups: 1.02M, 1.91M, 3.52M, 6.44M and 8.20M
  rows/s at 1, 2, 4, 8 and 14 threads — **8.0×** on a 14-core machine.
- **Specs.** Eight single-group specs in one bank run in 130 ms against
  515 ms one at a time.
- **Halflives.** Each halflife in a grid is its own set of running sums, and
  the instances of a stream run alongside each other (except with
  `drift_action="reset"`, which couples them). Ridge and feature-set grids
  are not parallel because they need not be: they share one set of running
  sums and are expanded at solve time.
- **Python.** Python's global lock is released while a chunk is in the
  bank, so a Python reader thread can run ahead of `ModelBank.fit_predict`.
- **The file-to-file runner.** Its own three-stage pipeline, and how it
  shares this pool, are in [docs/RUNNER.md](docs/RUNNER.md).
- **The expression form.** Under `.over("group")`, Polars runs the groups
  through its own pool — which is why the expression packs its inputs into
  one record column: the single-input path is parallel, the multi-input
  one is not (12.2M rows/s at 1000 groups).

Thread count is `POLARS_ONLINE_MAX_THREADS` for the bank's pool and
`POLARS_MAX_THREADS` for Polars' readers and writers; unset, each is one
thread per core. The bank builds its pool at the first bank call, and Polars
builds its own at import, so each must be set before that point — as above,
or in the shell (`POLARS_ONLINE_MAX_THREADS=8 python fit.py`), which is the
form that always works. Set later, the variable is ignored, and
[`po.thread_pool_size()`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.thread_pool_size) says what took (`pl.thread_pool_size()` for
Polars'). A value that is not a count is refused by name at the first bank
call. It changes the speed and nothing else: the same stream at 1 and 8
threads, in separate processes, gives identical output. Everything is one
process — there is no distributed execution, by design (see [What this is
not](#what-this-is-not)).

Two knobs because the two counts do different things. Polars' count also
sizes how much of a parquet file its reader holds in flight — it reads
ahead of the consumer, so more threads is a bigger pile of decoded rows
([Memory](#memory-which-calls-stream), above) — while the bank's count buys
speed and nothing else. So a run that has to fit in a smaller box keeps
Polars small and gives the bank every core:

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

On 12M rows over 64 groups, one spec: 14 and 14 takes 2.6 s at a peak of 1.1
GB. 4 and 14 takes the same 2.6 s at 0.8 GB — a third less memory at the same
speed. One shared count of 4 takes 3.9 s at 0.6 GB, and Polars alone at one
thread 7.4 s, because reading and writing are then one thread's work. Six
specs split the same way: 10.3 s at 1.5 GB, 11.8 s at 1.2 GB, 16.6 s at 1.0
GB. (Memory here and below is the peak the process ever held, as
`/usr/bin/time -l` reports it. The resident size reads about 0.7 GB higher,
because the memory-mapped input file counts there.) The pools never wait on
each other, because a bank task never calls back into Polars' pool. So
giving both more threads than there are cores costs no time either: 28 and
28 on 14 cores ran the grid above in 2.18 s against 2.21.

### Chunk size

`chunk_rows` is how many rows the bank takes at a time. It is a keyword on
`lf.online.fit_predict` and `lf.online.predict` (and on the file-to-file
runner, [docs/RUNNER.md](docs/RUNNER.md)); the default is 100,000. With
`ModelBank.fit_predict(df)` the chunk is whatever frame you pass.

It never changes the numbers. One chunk or a thousand gives the same
output; the one thing that moves is where `coef` lands, because each stream
reports its coefficients on its last row of every chunk. `coef_every` gives
it a cadence that does not move.

It does change the speed. The fixed cost of a chunk — handing the frame
across from Polars, gathering the columns, assembling the output — is paid
once per chunk, so tall chunks amortize it; on wide frames (thousands of
columns) that hand-off is about 8 ms per call at 10,000 columns, and chunks
of 20,000 rows run 2.4× faster than chunks of 2,000. The trade is memory:
three chunks are in flight at once, so `chunk_rows` is also the size of the
middle one ([Performance](#performance), below).

## Performance

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

Targets share one set of feature sums, so 10 targets cost far less than
10× one. Each halflife in a grid is its own set of sums, but they run in
parallel, so a 5-halflife grid costs about 2× one rather than 5×. `rls`
pays 1.3–2.1× for the square-root form that keeps it from dying of
cancellation on one extreme row; that is worth it.

The other families, and the options that add a pass, on the same machine
and rows:

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

A conformal interval is free: it reads the residual the model already has.
A simplex constraint sorts `2k` breakpoints per row, so it costs `sgd`
about 4×. `ew_cov` writes 230 numbers a row and still runs at half the
speed of one `ewridge`. The Mahalanobis distance and the full-covariance
`ew_class` each pay for a Cholesky factor of a `k × k` matrix, one per row
for `mahal` and one per *learned* row for `ew_class` — the classes a row
does not touch keep theirs. `kmeans` and `micro` cost a distance to each
centre; `seqtest` a handful of operations.

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

`deco` is one number for the whole matrix and costs `O(m)` a row, which is
why it runs at `ew_cov`'s speed and not at a covariance matrix's. `rcov`
accumulates per row and pays for its kernel only when the block closes.
`hmm` factorizes a `k × k` covariance per state per row, which is
`ew_class`'s cost with the classes hidden. `bocpd` costs
`O(runs · d²)`, and the length of the run vector is the whole story —
see below.

**`prune_below` is not a tuning knob on `bocpd`, it is what makes it finite.**
The run vector grows by one entry every row, so with `prune_below = 0` the
model is `O(rows²)`: measured at 1,897 / 947 / 472 rows/s on 5k / 10k / 20k
rows, halving each time the stream doubles. At the default `1e-6` it is
flat in the length of the stream, and the knob is a direct dial on
throughput — 204k, 324k and 687k rows/s at `1e-8`, `1e-6` and `1e-4` on
i.i.d. Gaussian rows. `max_run` is the belt to that pair of braces and
usually never binds. One consequence worth knowing: **`bocpd` is faster on
data that actually breaks**, because a changepoint collapses the
distribution onto a short run — the 1.0M rows/s in the table is on data
with regimes, against 324k on a stationary stream.

`corrchange`'s window kind is the slowest model here, and deliberately: the
permutation null re-draws `n_perm` statistics every `permute_every` rows.
At the default cadence that is a `O(n_perm · window · k²)` job spread over
500 rows; `crit` given as a number skips it entirely.

Grouped data goes wider, as [Parallelism](#parallelism) shows: 8.2M rows/s
at k=20 over 64 groups.

**Memory** is three things: the state, the chunks in flight, and whatever
Polars' reader has read ahead. Three chunks are in flight at once, so
`chunk_rows` is the knob for the middle one ([Chunk size](#chunk-size),
above). The read-ahead is usually the largest of the three: on a 14-thread
machine the parquet reader front-loads ~0.7 GB of decoded rows whatever the
file's length, and `POLARS_ROW_GROUP_PREFETCH_SIZE=1` takes a file-to-file
run to 0.15 GB at the same speed. It is sized from the thread count, so
`POLARS_MAX_THREADS` shrinks it too. Where the time goes, and what to reach
for, is in [docs/PERFORMANCE.md](docs/PERFORMANCE.md).

## Against scikit-learn

The model most people compare this to is
[`SGDRegressor.partial_fit`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html),
and the honest comparison starts by naming what each side is. `SGDRegressor`
is a first-order stochastic optimiser: its answer depends on the learning
rate, the schedule, the feature scaling and the row order. The primary
regression here is a different algorithm class — `ewridge`, `rls`, `lasso`,
`huber` and `quantile` accumulate sufficient statistics and solve, so there
is no learning rate, and with decay off `ewridge` is ordinary least squares
to 2e-13 of `numpy.linalg.lstsq` in any row order. The counterpart to
`SGDRegressor` here is [`sgd`](#sgd--stochastic-gradient-descent), the cheap
`O(k)` baseline.

Measured on one generated stream, 100,000 rows, `k = 20`, each contender at
its best over a sweep of its own settings (`scripts/sklearn_comparison.py`,
scikit-learn 1.9.0, `docs/PERFORMANCE.md` §19 for the full tables and the
sweeps). The noise ceiling is the R² of the generating signal itself:

| contender | R² stationary | R² drifting | rows/sec | what a prediction saw |
|---|---:|---:|---:|---|
| noise ceiling | 0.9831 | 0.9923 | | |
| `SGDRegressor`, row by row | 0.9829 | 0.9899 | 3,400 | every row before it |
| `SGDRegressor`, batches of 1,000 | 0.9830 | 0.9820 | 2,200,000 | every row before its *batch* |
| `po.spec.sgd` | 0.9826 | 0.9906 | 6,000,000 | every row before it |
| `po.spec.ewridge` | **0.9831** | **0.9907** | 575,000 | every row before it |

Three things to take from it. **Accuracy is not the difference.** Everything
reaches the ceiling on the stationary stream, and the row-by-row contenders
are within 0.001 of each other on the drifting one — `SGDRegressor` and
`po.spec.sgd` at the same constant step give the same number to four
places, because they are the same recursion. The one gap in the table, the
batched 0.9820, is staleness: a prediction made up to 999 rows before its
update. **The speed difference is a difference in semantics**: sklearn's fast
form updates once per 1,000-row batch, while every prediction here is made
from the state as it stands. Asked for that same guarantee — `partial_fit`
per row — `SGDRegressor` runs at 3,400 rows/second, and the cost is Python's
per-row overhead rather than the algorithm. And **a halflife is not the
advantage here**: a constant learning rate forgets too, at about `1/eta`
rows, and on evenly spaced rows that is a halflife. The clock matters when
the rows are not evenly spaced, which this stream is not.

**Where the Gram wins: the first hundred rows of every group.** An exact
solve is right as soon as its `X'X` is full rank, about `k` rows in; a
first-order method needs about `1/eta` rows per direction. 500 groups of 200
rows, each group its own coefficients, `k = 20`, R² by position in the group
— sklearn as one estimator and one scaler per group in a dict, row by row,
the bank as `group="g"`:

| contender | rows 25–50 | rows 50–100 | rows 100–200 | rows/sec |
|---|---:|---:|---:|---:|
| noise ceiling | 0.9896 | 0.9903 | 0.9900 | |
| `SGDRegressor` per group, at its best | 0.7277 | 0.9242 | 0.9789 | 3,342 |
| `po.spec.sgd`, `scale_features=True`, the same step | 0.7182 | 0.9213 | 0.9788 | 18,519,660 |
| `po.spec.ewridge` | **0.9693** | **0.9860** | **0.9882** | 5,130,803 |

The `sgd` row is sklearn's step, and through 0.3.1 it was not this close:
the table found `po.spec.sgd` diverging at the start of every group (R² of
−6.9 at rows 25–50), because `scale_features` standardised a row against
moments from *before* it, and a two-row variance estimate can be tiny by
chance. It now standardises against moments that include the row —
sklearn's scaler order, which is not a leak: the features are known at
prediction time, and the rule is about the target (`docs/PLAN.md` task 74).
The 0.01 of R² left between the two rows is where the *prediction* is
standardised: sklearn's loop standardises the row it predicts against the
moments before it and the row it learns from against the moments including
it, while here one standardised row serves both, so a prediction is on the
same footing as every row the coefficients were learned from; the gap
closes as the fit converges. The condition was few rows per feature, and a
wide fit is that on every row: at `k = 10,000` the scaled fit's predictions
correlated 0.52 with sklearn's before the change and 0.999997 after.

What else is different: a grid of six penalties is 3.8× the work for
sklearn (six estimators) and 1.6× here (one accumulator, six solves);
multiple targets share one `X'X`; `group=` is one state per key rather than a
dict of estimators; standardisation is streaming and cannot leak the way a
`StandardScaler` fitted on the whole frame does; chunk invariance is a test;
and the state is a versioned cross-OS file rather than a pickle.

**Where sklearn wins: a wide row against `ewridge`, by batching.**
`ewridge` keeps a `(k+1)²` matrix, so at `k = 10,000` it carries 860 MB of
state and runs at 53 rows/second, against 0.31 MB and 20,007 for
`SGDRegressor` in batches of 1,000. That cost is the matrix itself — a
rank-1 update moves all 800 MB of it every row, at 85 GB/s, near this
machine's memory bandwidth; emitting coefficients is not it (`coef_every=1`
costs 2.5%), and neither is the solve (0.2%). `gram_block_rows=1024`
touches the matrix once per 1,024 rows instead and buys 7.2× of the
throughput (379 rows/second) and none of the memory. `sgd` is the `O(k)`
answer here — 0.89 MB and 33,271 rows/second at the same width, 45,000 fed
in chunks of 20,000 rows — and it is faster than sklearn's batch with
every prediction made from the state as it stands; `SGDRegressor` asked
for the same, row by row, runs at 2,206 rows/second at that width, and the
two agree (correlation 0.9954 between their predictions unscaled, and
0.999997 with `scale_features=True`, which is sklearn's own recipe of a
scaler in front of the step). Both are run at `learning_rate = 0.2 / k`: an LMS step is
stable only while `eta · |z|² < 2`, and a standardised row has `|z|² ≈ k`.
And the ecosystem is sklearn's: pipelines, `GridSearchCV`, calibration,
and far more use. What this has is the stream.

## What this is not

A model layer, not a stream-processing framework. It expects a frame that is
already aligned — and, when a spec names a `clock`, each group's rows in
clock order — and it keeps a fixed amount of memory per stream. It
deliberately does **not** provide:

- **connectors or ingestion** — feed it whatever Polars can read;
- **event-time windowing, asof or interval joins** — build features with
  Polars expressions upstream, or with a streaming framework such as
  [Pathway](https://pathway.com);
- **watermarks or late-arrival policy** — `clock`, `max_dclock`,
  `on_clock_reset` and `session` describe time *within* a stream, not
  pipeline lateness. Under a `clock`, a row that arrives out of order is a
  data error, and `on_clock_reset="error"` will say so;
- **distributed execution** — one process, a thread pool across (spec × group).

Those boundaries make the two compose:
[examples/pathway_integration.py](examples/pathway_integration.py) runs a
`ModelBank` as a stateful operator inside a Pathway pipeline — Pathway does
ingestion, event-time alignment and windowing; we do the model. Chunk
invariance means the engine's batching cannot change the numbers, and
`save_bytes`/`load_bytes` let a pipeline checkpoint carry the model state.
Pathway is not a dependency; the example imports it lazily.

## Versioning and the Polars pin

### What is pinned

| py-polars | rust polars | pyo3-polars | pyo3 | Python |
|---|---|---|---|---|
| **>= 1.34.0, < 2** (built and tested against 1.44.1) | 0.55.2 | 0.28 | 0.29 | ≥ 3.12 (`abi3-py312`) |

The Rust `polars` is pinned exactly and built into the wheel; the runtime
requirement is a range, because the two copies never meet. The floor is
`LazyFrame.collect_batches`, which `lf.online.fit_predict` and the
file-to-file runner read with and py-polars added in 1.34.0; the whole
suite passes on 1.34.0, 1.38.1 and 1.44.1 with identical numbers.
`ModelBank` and the expression form alone work from 1.28.1 (tested across
17 releases). The pins are asserted by a test; the matrix is in
[docs/RELEASE-READINESS.md](docs/RELEASE-READINESS.md).

### Why a mismatch is an error, not a crash

`ModelBank` and the expression plugin move data across the boundary through
the Arrow C Data Interface, the same cross-language ABI pyarrow and DuckDB
use; nothing here uses the version-sensitive types that cross as serialized
query plans. The only thing `ModelBank` asks of the Python side is
`PySeries._export` / `_import`, and a Polars without them fails with a clean
`AttributeError` before any data moves. The plugin loader goes further and
negotiates its ABI, refusing a major it does not know. The pin exists so you
never see those messages, not because something worse waits behind them.

### Which interfaces carry a promise

Polars supports three, and only one carries a guarantee:

- the **expression plugin** — the supported path, with a negotiated handshake;
- **pyo3-polars' extension types** (`ModelBank`) — provided "for
  convenience", with no guarantee beyond the latest definitions working for
  the latest Polars;
- the **IO plugin** (`lf.online.fit_predict`) — documented, but `@unstable`
  in py-polars.

The two that work a chunk at a time are the two without a promise, so a
break on a new Polars is expected maintenance, not a surprise.

### How the pin moves

A weekly job ([`polars-canary.yml`](.github/workflows/polars-canary.yml))
drops the range from `pyproject.toml`, installs the newest py-polars — a 2.0
included, the week it appears — builds the wheel as CI does and runs the
whole suite. Only polars moves in that run, so a red canary means Polars
broke us and nothing else. The response is decided in advance: **cap** the
range at the last release that passed, in a patch release, so no resolver
hands anyone the broken pair; then **fix**, and widen again. Where to look
first: `ModelBank`, then the IO-plugin tests in `tests/test_frame.py`, then
the plugin. The Rust copy of polars moves by hand, together with
pyo3-polars, polars-arrow, polars-parquet and polars-utils, through CI.

### This package's own versioning

Semantic versioning. While pre-1.0 the **minor** version carries breaking
changes, and any change to the numbers a model returns, so pin `~=0.4.0` if
you need stability. Widening the Polars range is a minor release; narrowing
it is breaking. See [CHANGELOG.md](CHANGELOG.md). Output field names are
part of the API ([above](#output-field-names)).

## Testing

The guarantees above are only worth what checks them, so the suite is built
around oracles and invariants rather than expected values typed in by hand.
About 650 Rust tests and 2,200 pytest cases (from some 1,250 test functions),
all green on three OSes; [docs/TESTING.md](docs/TESTING.md) is the ledger of what
each part proves and what it has found.

**Against references.** `ewridge` and `rls` match numpy references in
`tests/reference.py` to 1e-9, `kalman` to ~1e-15, `huber` and `quantile` to
~1e-13, `ftrl` to ~1e-16; `rls` equals `ewridge(ridge_decay=True)` solved
every row to <1e-9. The lasso is checked against the KKT conditions of its
objective rather than a ported solver, which cannot share a bug with it.
[river](https://riverml.xyz) is an independent implementation of several of
the same algorithms. Its FTRL recursion agrees with ours to 1e-12 row for
row; its EW moments agree in closed form and in the limit; its quantile and
Huber models agree statistically. Two convention differences are pinned as
tests rather than left as surprises.

**Invariants, for every model.** Each of these is checked at the bank, and
where it applies at the expression and command-line levels too:

| | |
|---|---|
| chunk invariance | one chunk, seven, four hundred, one row at a time, and with a save and load in the middle |
| thread invariance | 1 thread against 8 |
| group independence | a group's numbers do not depend on what else is in the bank |
| the paths agree | expression ≡ bank; runner ≡ bank for every input source and format |
| `predict` ≡ `fit_predict` | of the next row, field for field, with every diagnostic on |
| stream semantics | the null policy, warm-up, and the clock |
| `n_eff` | the same recursion in every model (`crates/online-core/tests/model_contract.rs`) |

Hypothesis generates adversarial streams — mixed nulls, duplicate and
long-gap clocks, values at ±1e8, zero weights, tiny groups — and asserts the
strongest one: **changing a row's own target never changes that row's own
prediction.** IC ≈ 0 on pure-noise targets says the same thing from the
other side.

**Fixed numbers.** One golden stream per model in the Rust core, and the
whole pipeline — extraction, fan-out, diagnostics, struct assembly — pinned
to fixed output and compared on every OS, so a divergence in polars'
vectorized paths on another CPU would show.

**Hardening.** What the suite does to a bank on purpose:

| | |
|---|---|
| everything at once | a 30k-row stream with every output switched on, compared by digest across chunkings, a mid-stream save and load, and thread counts |
| weight scale | all weights ×1e±6 changes nothing but `n_eff` |
| parameter edges | `halflife` from `1e-3` to `inf` |
| a corrupt state file | any byte flipped fails cleanly, and never panics |
| concurrent misuse | two threads calling `fit_predict` at once get a clean error |
| copying a bank | `pickle` and `copy.deepcopy` resume bit-exactly |
| across the FFI | memory safety where two copies of Polars share one process |
| sustained load | a 10M-row soak, opt-in with `pytest -m soak` |

**Contracts that are files.** The public API — every name, default and
signature, every output field name — is a checked-in snapshot
(`tests/api_surface.txt`), so a change is a reviewable diff. Every python
block in this README runs. Everything under `examples/` runs unmodified —
the TOML through the real command line, the Pathway operator end to end.
`docs/VALIDATION.md`, where the defaults were chosen,
is regenerated and compared, so the numbers behind them cannot silently stop
being true. A data file, a large file or generated output that gets tracked
fails a test. Bank files from the previous schema version still load.

**Beyond the suite.** `cargo mutants` runs over the core: the last pass left
8.3% of 2,616 mutants surviving, clustered where only the Python suite
reaches (`cargo test` cannot see it), which is what the golden and contract
tests in Rust were added for. Coverage is 96% of the Python package and 75%
of Rust regions, understated for the same reason.

**Where it runs.** `./scripts/gate.sh` before every commit — `cargo fmt`,
`clippy -D warnings`, `cargo test`, `ruff`, `mypy`, the build, `pytest`,
`sphinx -W`. CI runs the same on ubuntu, windows and macos for every push
and pull request. The release build writes a state file on macOS and
continues the stream from it on Windows and Linux. The weekly canary runs
the suite against the newest py-polars.

Tests generate or download their own data; there are no data files in the
repo. Downloads are cached under `.cache/` and skipped when offline.

## Development

```sh
uv sync                                                # Python env (CPython 3.12)
./scripts/gate.sh                                      # everything CI checks
uv run cargo test --workspace                          # Rust tests
uv run maturin develop --release -m crates/online-py/Cargo.toml
uv run pytest                                          # Python tests
uv run --group docs sphinx-build -W docs/reference docs/_build/html   # API reference
uv run python scripts/validate.py > docs/VALIDATION.md # re-run the [validate] experiments
uv run python scripts/regime_experiments.py all        # the docs/REGIMES.md experiments
uv run python scripts/benchmark.py                     # throughput
```

Prerequisites: [uv](https://docs.astral.sh/uv/) and a stable Rust toolchain
([rustup](https://rustup.rs)). `source scripts/env.sh` (`. .\scripts\env.ps1`
in PowerShell) puts both on the `PATH` for a shell; `.vscode/settings.json`
does it for VS Code's terminal. `cargo` runs via `uv run` because `online-py`
builds against pyo3's `abi3-py312` and needs a 3.12+ interpreter at build
time.

- Every document, and which to read for what: [docs/README.md](docs/README.md)
- A map for coding agents, at the repo root and on the docs site:
  [llms.txt](llms.txt) ([llmstxt.org](https://llmstxt.org))
- API reference: <https://hgilde.github.io/polars-online/> — built from the
  docstrings and published from every green push to `main`
- Design and task list: [docs/PLAN.md](docs/PLAN.md)
- Running a bank as a job: [docs/RUNNER.md](docs/RUNNER.md)
- Saving, serving, resuming: [docs/STATE-WORKFLOW.md](docs/STATE-WORKFLOW.md)
- Measured defaults: [docs/VALIDATION.md](docs/VALIDATION.md)
- What the regime detectors find: [docs/REGIMES.md](docs/REGIMES.md)
- Where the time and memory go: [docs/PERFORMANCE.md](docs/PERFORMANCE.md)
- Adding a model: [docs/EXTENDING.md](docs/EXTENDING.md)
- How the docs are written: [docs/WRITING.md](docs/WRITING.md)

## License

Apache-2.0. See [CONTRIBUTING.md](CONTRIBUTING.md) to make changes,
[SECURITY.md](SECURITY.md) to report a vulnerability.
