# polars-online

Online regression for [Polars](https://pola.rs): a bank of models that learns
one chunk at a time and predicts every row *before* it learns from it. Rust
core, Python API, and a standalone CLI.

It is built for data that does not fit in memory. Feed it a stream and memory
stays at *state + one chunk* however long the stream runs, and the numbers are
the same whether the stream arrives as one chunk or a thousand.

> **A note on Polars versions.** Two of the three interfaces this rides on
> carry no stability promise from Polars, so `polars>=1.34.0,<2` is measured
> rather than guaranteed. A weekly canary runs the whole suite on the newest
> py-polars, and the response to a red one is decided in advance. Details in
> [Versioning and the Polars pin](#versioning-and-the-polars-pin).

## What you get

**Fourteen model families, one set of stream semantics.** A spec's clock, decay,
grouping and warm-up mean the same thing whichever model it names.

| model | what it is |
|---|---|
| `ewridge` | exponentially weighted ridge on sufficient statistics — the workhorse; grids over ridge values, feature sets and halflives come almost free |
| `rls` | recursive least squares, in the numerically safe square-root form |
| `lasso` | lasso / elastic-net path with online λ selection |
| `kalman` | Kalman filter with random-walk coefficients |
| `huber`, `quantile` | robust and quantile regression |
| `sgd` | stochastic gradient descent with squared, Huber, quantile, ε-insensitive, Poisson and logistic losses |
| `pa` | passive-aggressive regression — no learning rate |
| `ftrl` | FTRL-proximal logistic regression, L1-sparse |
| `ew_cov` | running mean, variance, covariance, correlation, partial correlation, Mahalanobis distance and PCA |
| `holt` | Holt's linear trend — the no-feature baseline |
| `kmeans` | exponentially weighted k-means — out-of-sample cluster labels, with a split–merge move that finds a cluster born after seeding |
| `micro` | density-based clustering — DenStream micro-clusters linked into clusters of any shape and number; flags the rows that belong to none |
| `ew_class` | Gaussian classification — QDA, LDA or naive Bayes on one `ew_cov` state per class; a label column in, out-of-sample posteriors out |
| `seqtest` | a sequential test of a sign by betting — an e-process you can read at any row; on its own a column's sign, with `a`/`b` whether one spec of the bank predicts closer than another |

**Three ways to run a bank, same numbers from each.** A Python loop over
chunks (`ModelBank`); a Polars query (`lf.online.fit_predict(specs)` is a
`LazyFrame` you `collect`, `sink` or batch like any other); or a file-to-file
job (`po.run(...)` from Python, or the `online` CLI from a TOML with no Python
at all). There is also an expression form for a frame in memory — it cannot
stream, and it says so.

**Time, built in.** A clock column in any unit, decay by halflife in that
unit, a ceiling on gaps, session boundaries, a policy for a clock that runs
backwards. One state per group, row weights, warm-up thresholds. Or no clock
at all: then row order is the clock — and with decay off, the bank is plain
least squares over everything it has seen, in any row order.

**Two guarantees.** Predictions are out-of-sample by construction. Output is
chunk-invariant. Both are tests, not intentions.

**Tested the way those guarantees demand.** Around 470 Rust tests and 1,700
pytest cases: numpy oracles to ~1e-13, cross-checks against river,
hypothesis-generated adversarial streams, chunk and thread invariance, golden
numbers on every OS, the README's own code blocks executed. CI runs all of it
on macOS, Windows and Linux on every push, and a weekly canary runs it on the
newest Polars. Details under [Testing](#testing).

**State is a file.** Save a bank; load it to keep learning, or to serve
predictions without learning. The file is written atomically, is the same
bytes from every entry point, and carries nothing host-specific.

**Introspection and diagnostics.** Coefficients as a table or as columns.
Residual sigma and z-scores, drift detection, online model selection and
averaging across a grid, streaming R²/IC/hit rate, residual quantiles and
autocorrelation — all out-of-sample, all O(state).

**Parallel by group, deterministic by construction.** Every (spec, group)
pair is one task on a thread pool, the halflives of a grid run side by side,
and the runner overlaps reading, fitting and writing. Rows within a stream go
one at a time — that is the recursion — so the thread count changes the speed
and nothing else, and a test holds it to that. Details under
[Parallelism](#parallelism).

## Install

Wheels for macOS (arm64, x86_64), Windows x64 and Linux (x64 glibc and musl,
aarch64 glibc) plus the CLI binaries are attached to each release. Python 3.12+.
The wheel is ~19 MB to download and ~59 MB installed: it statically links the
Rust half of Polars, so nothing beyond `polars` itself has to be present at
run time. `numpy` is an optional extra, needed only by `ModelBank.gram()`.

From source, with [uv](https://docs.astral.sh/uv/) and a stable Rust toolchain:

```sh
uv sync
uv run maturin develop --release -m crates/online-py/Cargo.toml
```

## Quick start

```python
import polars as pl
import polars_online as po

spec = po.spec.ewridge(
    "ridge",
    targets=["ret"], features=["signal_a", "signal_b"],
    clock="ts", halflife=600.0, max_dclock=300.0,   # decay in seconds; gaps capped at 5 min
    group="bond_id",                                 # one model per bond
)

# Fit and predict over a stream; save the fitted state when the run reaches the last row
(
    pl.scan_parquet("ticks/*.parquet")
    .online.fit_predict([spec], save_state="bank.state")
    .filter(pl.col("ridge").struct.field("n_eff") > 100)
    .sink_parquet("fitted.parquet")                  # memory: state + one chunk, end to end
)

# Later: score new rows against the saved state, learning nothing
scored = pl.scan_parquet("today.parquet").online.predict("bank.state").collect()

# The result is one struct column per spec; unnest it into plain columns
flat = scored.online.unnest([spec])   # pred_ret, resid_ret, n_eff, coef_ret_intercept, coef_ret_signal_a, ...
```

Each spec adds one struct column, named after the spec, holding
`pred_<target>`, `resid_<target>`, `n_eff`, `coef` and whatever diagnostics
you switch on. `df.online.fit_predict(specs)` does the same for a frame in
memory, eagerly.

## How a bank sees a stream

These parameters are shared by every model.

### Time and decay

| parameter | meaning |
|---|---|
| `clock` | a monotone **numeric** column — seconds, cumulative volume, anything. `None` means row count. Temporal columns are rejected; cast first (`pl.col("ts").dt.epoch("s")`), so that the units of `halflife`, `max_dclock` and `session_gap` are yours, not the dtype's |
| `halflife` / `lam` | decay in clock units, one or the other. A list of halflives gives one accumulator per value. `halflife=inf` (or `lam=1.0`) turns decay off |
| `max_dclock` | ceiling on the clock step; required with a `clock`. `0` disables decay, `inf` removes the ceiling |
| `on_clock_reset` | what a backwards clock means: `"max"` (default), `"zero"`, `"reset_state"`, or `"error"` to refuse the chunk and leave the bank as it was |
| `session`, `session_gap` | on a session change, apply this clock delta; `"reset"` resets the state, `inf` never applies it |
| `session_shrink`, `long_halflife` | `ewridge` only: at a session change, mix partway back toward a slow-moving twin |

Per-row decay is `λ = 0.5 ** (Δclock / halflife)`.

### Groups, weights and warm-up

| parameter | meaning |
|---|---|
| `targets`, `features` | column names, ≥1 target; targets share one `X'X`. Columns must be numeric (any width, `Decimal` and `Boolean` included; cast to `f64`) — a String column is refused rather than cast to null. Columns the spec does not name pass through untouched |
| `add_intercept` | default `True` |
| `group` | one state per key |
| `weight` | row weight column |
| `min_periods` | in `n_eff` units; outputs are null until it is reached. A list gives one threshold per target. Warm-up gates output, not learning |
| `coef_every` | snapshot the coefficients every N rows (`0` = only on each chunk's last row) |

`n_eff` is the exponentially weighted observation count: the weight behind
the state that produced *this row's* prediction, measured before the row's
own update and decay. So it is `0` on a stream's first row, lags the row
count by one, saturates at `1 / (1 − λ)`, and means the same thing in every
model — which is what makes `min_periods` portable across a bank.

### Nulls

A null in any feature, or in the weight, skips the row: outputs are null, no
update happens, the clock still advances. A null in one target still emits
that target's `pred`, leaves its `resid` null, and skips only that target's
update. NaN, ±inf and any magnitude above `1e100` count as null, so sentinels
never reach a model.

### Three ways to hold a row back

They differ, and it matters which you use.

- **Weight `0`** — the row is scored, the clock advances, nothing is learned:
  the coefficients are frozen bit for bit. Use it to keep a row's place in
  the stream. Since the clock advances, `n_eff` keeps decaying and can fall
  below `min_periods` if you score for a long stretch this way.
- **A null target** — the feature moments still update while the target's
  cross-moment does not, so the coefficients wander with feature noise. Use
  it only for a label that has not arrived yet.
- **`predict`** — scores every row against the bank exactly as it stands and
  touches nothing: no clock advance, no decay, `n_eff` frozen. Use it to
  serve. It is also the fast path: `ewridge` scores at 1.8–2.9× its learning
  throughput.

### Any row order

A bank is a set of sufficient statistics, so row order reaches the fit only
through decay. With decay off, an `ewridge` with `ridge=0` is ordinary least
squares over every row it has seen, in whatever order they came: forwards,
backwards or shuffled, its coefficients match `numpy.linalg.lstsq` to 2e-13.
What it costs is state, not data — 6M rows × 20 features from a parquet
stream peak at 1.4 GB, against 3.97 GB for `lstsq` on the same rows, and the
frame never has to fit. Set `solve_every=1000` to solve less often than every
row (1.4 s instead of 11 s there, coefficients at most 1000 rows stale).

A finite halflife with no clock discounts each row by how far back it sits,
so the fit is a weighted least squares of *that* order. One trap: a huge
finite halflife is not `inf`. Its solve cadence defaults to `halflife/50`, so
`halflife=1e12` solves once, at `min_periods`, and never again. Say `inf`, or
set `solve_every`.

### Two guarantees

- **Predictions are out-of-sample.** Every row is predicted from the state
  before its own target is folded in. Nothing here can leak.
- **Chunk invariance.** One chunk or a thousand, with or without a save and
  resume in the middle, the output is bit-identical. The one exception is
  `coef`, which is a *reporting* cadence: it is snapshotted every
  `coef_every` rows and on each chunk's last row, so smaller chunks report it
  more often.

### Mistakes are named

A builder checks each keyword against its type hints — `halflife="10"` says
`spec "m": halflife must be a number or a list of numbers, got str '10'`. A
missing column says which spec wanted it, in what role, and what the frame
has. A spec named like an input column is refused rather than silently
replacing it.

## Running a bank

### In a loop: `ModelBank`

```python
spec = po.spec.ewridge(
    "ridge",
    targets=["y"], features=["x0", "x1", "x2"],
    clock="t", halflife=600.0, max_dclock=300.0,
    group="bond_id", ridge=[1e-6, 0.1], standardize=True,
)
bank = po.ModelBank([spec])

for chunk in lf.collect_batches():        # never materializes the whole stream
    out = bank.fit_predict(chunk)
    ...

bank.save("bank.state")                    # atomic: temp file, then rename
```

A bank says what it holds: `repr(bank)` is
`ModelBank(['ridge'], groups=412, rows_seen=3000000)`, `bank.specs` gives
back the spec dicts, and `bank.groups()` is a frame of every `(spec, group)`
with its row count and last clock value. Groups live until dropped, so a
long-running bank forgets the quiet ones with:

```python
stale = bank.groups().filter(pl.col("last_clock") < now - 30 * 86400)
bank.drop_groups(stale["group"])           # they start cold if they reappear
```

### As a query: `lf.online.fit_predict`

The loop above as a `LazyFrame`. Executing it — `collect()`,
`collect_batches()`, any `sink_*()` — streams the plan's rows through a
fresh bank in `chunk_rows` chunks, so the query stays at *state + one chunk*
however long the stream, and everything after the bank is ordinary polars:

```python
(
    pl.scan_parquet("ticks/*.parquet")
    .online.fit_predict([spec], chunk_rows=100_000)
    .filter(pl.col("ridge").struct.field("n_eff") > 100)  # after the bank: what comes out
    .select("ts", "bond_id", "ridge")                     # pushed into the scan
    .sink_parquet("fitted.parquet")
)
lf.online.predict(bank).collect()                                         # serve
lf.online.fit_predict(load_state="bank.state", save_state="bank.state")   # resume, and save
```

Things worth knowing about the plan:

- **It is pure.** Every execution starts from the same state (the specs',
  or `load_state`), so collecting twice gives the same frame, and `head(n)`
  learns from the first `n` rows and no more.
- **`save_state` writes when the run reaches the last row**, atomically, and
  the same bytes a `ModelBank` or `po.run` would write. A run abandoned early
  or ended by a bank error leaves the file untouched. A failure *after* the
  bank does not stop the bank, so the state is written although the query
  failed; if the two must be tied together, `po.run` saves only after its
  output is committed. [docs/STATE-WORKFLOW.md](docs/STATE-WORKFLOW.md) has
  the measurements behind each of these.
- **Filters, selections and `head` after the bank are honoured at the
  source**, and a selection reaches the input scan, so only the columns the
  specs and the query need are read.
- **Filter after the bank, not before, unless the model must skip those
  rows.** A filter after never changes what the bank learns from, and it
  streams; a filter *before* holds several row groups per thread in polars'
  parquet reader — 2.5 GB at 12M rows against 0.78 GB for the same filter
  after ([docs/PERFORMANCE.md](docs/PERFORMANCE.md) §11).

If the model must not learn from some rows, give them weight `0` instead of
filtering them out: they still stream, still come out scored, and no gap
opens in the clock.

```python
(
    lf.with_columns(pl.when(pl.col("venue") == "X").then(1.0).otherwise(0.0).alias("w"))
    .online.fit_predict([po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"],
                                         clock="t", halflife=600.0, max_dclock=300.0,
                                         weight="w")])
    .sink_parquet("fitted.parquet")
)
```

`df.online.fit_predict(specs)` is the eager twin. `po.fit_predict(frame, ...)`,
`po.predict(frame, bank)` and `po.unnest(frame, specs)` are the same calls as
plain functions, for a type checker, which cannot see a registered namespace.

### As a job: `po.run` and the `online` CLI

The same bank as a three-stage pipeline — read, fit, write, one chunk in
flight per stage — with the output written through a temporary file and
renamed into place, so a failed run leaves the previous output where it was.

```python
po.run(input="ticks.parquet", output="fitted.parquet",
       specs=[spec], chunk_rows=100_000, save_state="bank.state")   # -> {"rows": ..., "chunks": ...}

po.run("bank.toml", input="today.csv")                              # keywords override the TOML
po.run(input="today.parquet", output="scored.parquet", specs=[spec],
       load_state="bank.state", predict=True)                       # serve: learn nothing
```

`input` is anything py-polars can stream: a path in parquet, ipc, csv or
ndjson (told from the extension, or named with `input_format=`; globs and
cloud URLs as `pl.scan_*` takes them), a `LazyFrame` with whatever scan
options it needs, a `DataFrame`, or any iterable of frames in stream order —
a database cursor, a socket, a generator. `output` is a path in any of the
four formats. `keep_columns=[...]` selects input columns before the bank sees
them, and `progress(rows, chunks)` is called after each chunk; raising in it
stops the run. CSV cannot hold struct columns, so there each spec's struct is
flattened to `<spec>.<field>` columns and the `coef` list becomes a JSON
string that `pl.col("ridge.coef").str.json_decode(pl.List(pl.Float64))` reads
back bit-exact.

The CLI is the same pipeline as one binary and one TOML
([examples/bank.toml](examples/bank.toml)), for deployments with no Python:

```sh
online --config bank.toml
online --config bank.toml --resume bank.state --save-state bank.state
online --config bank.toml --resume bank.state --predict --input today.parquet
online --config bank.toml --input ticks.csv --output scored.ndjson
online --config bank.toml --input feed.dat --input-format ipc
online --config bank.toml --dry-run          # validate and print the output schema
```

`--predict` scores against the resumed state and learns nothing; it drops
the config's `save_state`, so one TOML serves both runs. The CLI reads with
polars' own scanners, which on a stable toolchain lack the SIMD CSV parser
py-polars' wheels have, so for a large CSV `po.run` is the faster of the two.
In TOML, a Windows path needs single quotes or forward slashes
(`input = 'C:\data\in.parquet'`), since a backslash in a double-quoted string
starts an escape sequence.

From Rust, the same pipeline is `online_polars::run_config` for a `RunConfig`,
`run_config_on` for a `LazyFrame` or batches the caller already has, and
`run` with a callback instead of an output file.

### The expression form (in memory only)

For a frame that is already in memory, the shortest way to write a model is
as an expression. Features may be expressions, evaluated per group under
`.over`, so a lag never crosses a group boundary:

```python
out = df.with_columns(
    pl.col("y").online.ewridge(
        features=["x0", "x1", pl.col("y").shift(1).alias("y_lag")],
        clock="t", halflife=600.0, max_dclock=300.0,
    ).over("group").alias("fit")
)
```

The numbers are the bank's — the expression *is* the bank, run over the
column polars hands it. And that is the catch: polars gives a stateful user
expression its whole column at once, in either engine, so wrapping the
expression in a lazy query does not make it stream. On the 12M-row file
below, it peaks at **7.3 GB against 1.35 GB** for the plan. Every call
therefore warns with `polars_online.InMemoryExpressionWarning`; using the
expression on a frame in memory on purpose is fine, and one line says so:

```python
import warnings

warnings.filterwarnings("ignore", category=po.InMemoryExpressionWarning)
```

`po.online(pl.col("y"))` is the same namespace as a plain function.
[docs/PLAN.md](docs/PLAN.md) §6 has the design and the condition under which
the warning would go away.

## Memory: which calls stream

All of them but the expression form. Peak footprint on one file, `ewridge`
with 20 features, parquet in and out:

| what you write | 3M rows | 12M rows | |
|---|---:|---:|---|
| `lf.online.fit_predict([spec])` | 0.90 GB | 1.35 GB | a query over a stream |
| `for chunk in lf.collect_batches(): bank.fit_predict(chunk)` | 0.80 GB | 1.24 GB | your own loop |
| `po.run(...)`, `online --config` | 0.95 / 0.73 GB | 1.41 / 0.75 GB | file in, file out |
| `pl.col("y").online.ewridge(...)` in `with_columns` | | 7.3 GB | the expression: whole column at once |

The first three are flat; what growth they show is the allocator holding
freed pages, and nearly all of the rest is polars' parquet read-ahead
(`POLARS_ROW_GROUP_PREFETCH_SIZE=1` takes it to 0.31–0.46 GB). Everything
after the bank — filters, joins, group-bys, sinks — streams as polars streams
it. Note that this is polars' rule for the operations *around* the bank too:
a rolling window under `.over("group")` or `group_by=` collects (6.5 GB and
1.7 GB on the same rows, against 0.25–0.28 GB ungrouped), whereas a bank's
`group=` is one accumulator per group and stays O(state).
[docs/PERFORMANCE.md](docs/PERFORMANCE.md) §11 has every measurement.

## Saving, loading and serving

A fitted model is the bank's *state* — one accumulator per `(spec, group)` —
and it travels as one file, written whole or not at all. The same words work
from a bank and from a query:

```python
# From a bank
bank.fit_predict(df)
bank.save("bank.state")                               # atomic: temp file, then rename
bank = po.ModelBank.load("bank.state", specs=[spec])  # specs= checks the file is this model's
bank.fit_predict(today)                               # learn on: the state moves
scored = bank.predict(today)                          # serve: score, learn nothing

# From a query
lf.online.fit_predict([spec], save_state="bank.state").sink_parquet("fitted.parquet")
lf.online.fit_predict(load_state="bank.state", save_state="bank.state").sink_parquet("more.parquet")
served = lf.online.predict("bank.state").collect()
```

`po.run(..., save_state=)`, `po.run(..., load_state=, predict=True)` and the
CLI's `--save-state` / `--resume` / `--predict` read and write the same file,
and the bytes are the same whichever wrote them. `save_bytes()` and
`load_bytes()` do the same in memory, for a checkpoint that lives somewhere
else. Loading names the problem it hits: `FileNotFoundError` for no file yet,
`ValueError` for a file that is not a bank, a newer build's, or another
model's.

What `predict` reports: row `i` carries what `fit_predict` would have
reported had it been the next row of the stream — `pred`, `n_eff`, `sigma`,
`resid_z`, selection, metrics, field for field — with every row scored from
the same state. The target column may be absent (then `resid` is null),
`weight` is not read, a group the bank has never seen scores null, and the
stream's session and clock policies still hold.
[docs/STATE-WORKFLOW.md](docs/STATE-WORKFLOW.md) walks the whole workflow:
fit, save, serve, learn on, with what each step guarantees.

## Reading the fit

### Coefficients

Two ways, and they agree row for row:

```python
ols = po.spec.ewridge("ols", targets=["y"], features=["x0", "x1"], clock="t",
                      halflife=600.0, max_dclock=300.0, group="bond_id", coef_every=1)

# 1. From a bank -- live, or loaded from a state file with no data at hand.
#    One row per coefficient, with the term it belongs to.
bank = po.ModelBank([ols])
bank.fit_predict(df)
betas = bank.coef("ols")             # group, instance, n_eff, position, target, ..., term, coef
wide = betas.pivot("term", index=["group", "instance"], values="coef")

# 2. From the output, as columns: the fit as it moved, one row per row.
path = (
    lf.online.fit_predict([ols])
    .online.unnest([ols])            # pred_y, resid_y, n_eff, coef_y_intercept, coef_y_x0, coef_y_x1
    .select("t", "bond_id", "^coef_.*$")
    .collect()
)
```

`bank.coef()` is the fit as of the last row each group learned from, with
`n_eff` for how much weight is behind it. The output's `coef` is the same fit
snapshotted *after* each row's update (the row's `pred` is from the fit
*before* it), every `coef_every` rows and on the last row of every chunk; the
default, `coef_every=0`, is the chunk end only, so the per-row path above
asks for `coef_every=1` and pays a list of `k` floats per row.

Under a grid — several `ridge` values, `feature_sets`, a `lasso_path`,
several targets — the list holds one block per (target × grid point).
`unnest` names each block's columns the way the `pred` fields are named
(`coef_y_x0__r0.5@h500` beside `pred_y__r0.5@h500`), `bank.coef()` carries
the same columns to tell blocks apart (add them to the pivot's `index`), and
`unnest` reads a saved output the same way:
`pl.scan_parquet("fitted.parquet").online.unnest([ols])`. It takes the specs,
a bank, or the path of a saved state. `bank.gram("ols")` gives the EW
accumulators behind the fit (`means`, centered `comoments`, `cross_moments`,
`n_eff`), for anything other than our solve.

### The last row

The state file also carries the output row of the last row each group
learned from, so a saved model says how it was doing without its output
frame:

```python
bank = po.ModelBank.load("bank.state", specs=[spec])
last = bank.last_row("ridge")    # one row per group: spec, group, pred_y__r0.000001, ..., n_eff, coef
```

It is the `fit_predict` row field for field — `pred`, `sigma`, the metrics
and the interval when the spec asks for them, `n_eff`, and `coef` when that
row carried it (a chunk's last row does; `bank.coef()` has the coefficients
either way). Fit many models, save each, and comparing them is one `concat`
over the files:

```python
from pathlib import Path
table = pl.concat(
    [po.ModelBank.load(f).last_row() for f in sorted(Path(".").glob("*.state"))],
    how="diagonal_relaxed",          # specs with different fields stack with nulls
)
```

A group that has not learned from a row yet, or a file written by 0.1.x,
gives a row of nulls, and `predict` does not move the row.

### What it was fed

The state file also carries what each group has seen: how many rows, what
became of them, the clock's range, and a count and moments for every input
column. A saved model can say what it was trained on without the data at
hand:

```python
bank = po.ModelBank.load("bank.state", specs=[spec])
fed = bank.summary("ridge")    # one row per group: rows_fed, rows_processed, rows_skipped, rows_learned, ..., clock_min, clock_max
cols = bank.describe("ridge")  # one row per input column per group: column, role, count, null_count, mean, std, min, max
```

`summary` counts rows. `rows_fed` were routed to the group; `rows_processed`
the model accepted, `rows_skipped` it did not (a feature or the weight was
missing); `rows_learned` moved the fit (a weight above zero and a target
present) and `rows_zero_weight` advanced the clock and nothing else. With
them come `weight_sum`, the clock's first and last values, `last_clock`,
and what the clock schedule met: `session_changes`, `clock_backwards`,
`resets`. `describe` is `DataFrame.describe` for each feature, target and
weight column over the rows fed, counting as the models count: a null, a
NaN, an infinity or a magnitude beyond `1e100` is a `null_count`, not a
value. A label column has counts only, and an unsupervised model lists its
features.

Neither decays: they are plain counts over the whole stream, computed in
row order, so they are the same whatever the chunking, to the bit. `predict`
does not move them. A file written before 0.2.0 reports nulls for both — a
count that began partway would read as the whole history — while
`rows_processed` and `last_clock`, which the stream always kept, are
filled in.

### Output field names

You index the result struct by strings, so the names are a contract. The
grammar:

```
pred_{target}{combo}{instance}     combo    = ""            single ridge, no feature sets
resid_{target}{combo}{instance}             | __r{ridge}     ridge grid
sigma_{target}{combo}{instance}             | __{set}        feature sets, single ridge
absresid_q{level}_{target}...               | __{set}_r{ridge}
n_eff{instance}                    instance = ""            single halflife
coef{instance}                              | @h{halflife}   halflife grid
```

Numbers render as plain decimals in `[1e-6, 1e7)` and as compact scientific
outside it. Every name, default and signature is pinned by
`tests/test_api_surface.py` against a checked-in snapshot, so a change is a
reviewable diff and a version bump, never a silent rename of your columns.

You never have to build these strings. `po.spec.output_index(spec)` lists
every field with the values its name encodes, and `po.spec.coef_fields(spec)`
does the same for the `coef` lists — one row per coefficient with its list,
position, and the column `unnest` gives it — so selecting is a filter, not
string formatting:

```python
grid = po.spec.ewridge("m", targets=["y"], features=["x0", "x1"], clock="t",
                       max_dclock=300.0, halflife=[100.0, 500.0], ridge=[1e-6, 0.5])

idx = po.spec.output_index(grid)
name = idx.filter((pl.col("kind") == "pred") & (pl.col("target") == "y")
                  & (pl.col("ridge") == 0.5) & (pl.col("halflife") == 500.0))["field"].item()
out["m"].struct.field(name)                                    # "pred_y__r0.5@h500"

row = po.spec.coef_fields(grid).filter(
    (pl.col("term") == "x1") & (pl.col("ridge") == 0.5) & (pl.col("halflife") == 500.0)
).row(0, named=True)
out["m"].struct.field(row["field"]).list.get(row["position"])  # field "coef@h500", position 5
```

Both tables come from the same Rust code that renders the names, so they
cannot drift from the strings, and the index carries each field's `dtype` —
the schema the bank declares to polars before the first row is read. One
sharp edge: avoid `__` and `@` in target names and feature-set labels if you
parse field names downstream, since a target named `y__r0.5` renders like a
ridge grid on `y`.

## Diagnostics, selection and evaluation

Opt-in outputs, all derived from state the models already keep and all read
*before* each row, so they are as out-of-sample as the predictions they
describe:

| flag | adds | meaning |
|---|---|---|
| `emit_sigma` | `sigma_<slot>` | EW standard deviation of that slot's out-of-sample residuals |
| `emit_resid_z` | `resid_z_<slot>` | `resid / sigma` — how surprising the row was, in units of the model's own recent error |
| `emit_selected` | `selected_<t>`, `pred_<t>__selected` | online model selection across ridge values, feature sets and halflives, by lowest EW out-of-sample error |
| `emit_averaged` | `pred_<t>__averaged` | `softmax(−eta · EW error)` blend over the same slots — hedges where `emit_selected` commits |
| `emit_drift` | `drift_<slot>` | Page-Hinkley break detection on the residual stream; `drift_action="reset"` also restarts the stream |
| `emit_metrics` | `ic_<slot>`, `r2_<slot>`, `hit_rate_<slot>` | the numbers `po.eval` computes, kept in O(state) beside the model |
| `resid_quantiles` | `absresid_q<p>_<slot>` | P² quantiles of \|resid\| — a distribution-free interval where `sigma` gives a Gaussian one |
| `emit_autocorr` | `autocorr_<slot>` | EW residual autocorrelation; non-zero means the model is mis-specified |
| `conformal=0.9` | `lo_<slot>`, `hi_<slot>`, `coverage_<slot>` | an adaptive conformal interval at that coverage, distribution-free, and the coverage it has actually delivered |

Drift detection complements the halflife rather than replacing it: decay
forgets smoothly and always; a detector notices a break and says so, within
a couple of rows of a sign flip.

`conformal` is the interval to use when the residuals are not Gaussian. It
tracks the `coverage` quantile of `|resid|` directly — the radius grows by
`conformal_rate · sigma · coverage` on a miss and shrinks by
`conformal_rate · sigma · (1 − coverage)` on a hit — so its long-run coverage
is the number you asked for whatever the residuals do, with an error that
shrinks like `1/T`. `sigma` gives a Gaussian interval; on Gaussian residuals the two
agree, and on fat-tailed or heteroskedastic ones the Gaussian interval
over-covers by several points where this one lands on target. It starts at
`sigma · Φ⁻¹(1 − α/2)` and is null until then, is read before the row like
everything else, and costs three numbers per slot.

```python
ci = po.spec.ewridge("ci", targets=["y"], features=["x0", "x1"], clock="t",
                     max_dclock=300.0, halflife=500.0, conformal=0.9)
band = po.ModelBank([ci]).fit_predict(df).unnest("ci")
held = band.select(((pl.col("lo_y") <= df["y"]) & (df["y"] <= pl.col("hi_y"))).mean())
```

After the fact, `po.eval` reads the output frame:

```python
po.eval.metrics(out, "ridge", by=["bond_id"])                       # R², IC, hit rate, MSE
po.eval.rolling_metrics(out, "ridge", clock="t", window=3600.0)     # per clock window
po.eval.compare_specs(out, ["ridge", "kalman"])                     # one table, many specs
po.eval.seqtest(out, a="kalman", b="ridge", by=["bond_id"])        # is kalman closer? evidence per row
```

`compare_specs` says which spec had the lower error over the frame.
`seqtest` says how sure you can be, at every row, and is the same test the
[`seqtest`](#seqtest--a-sequential-test-of-a-sign-by-betting) model runs
inside a bank, streaming.

## Models

All accumulators are exponentially weighted **means**, not sums, so they stay
bounded over arbitrarily long runs; second moments are kept **centered** (a
weighted Welford update), so the variance is right even when features sit on
a large offset. `z` denotes `[1, x]` when an intercept is configured, `w` the
row weight, `λ` the row's decay.

### `ewridge` — EW ridge on sufficient statistics

```
W'   = λW + w                       S' = (λW·S + w·z zᵀ) / W'
W_j' = λW_j + w                     r_j' = (λW_j·r_j + w·z·y_j) / W_j'
solve:  (S + ridge·D) β_j = r_j     D = I minus the intercept slot
```

O(k²) per row; Cholesky solves on a schedule (`solve_every` in clock units,
default `halflife/50`, every row for `halflife=inf` and for `lam`;
`max_rows_between_solves` caps it in rows). Ridge values and named
`feature_sets` are expanded at solve time from the same accumulator, so grids
are nearly free. `coef0` shrinks toward a stated belief rather than toward
zero; because `S` is a mean, a plain `ridge` is a permanent per-observation
penalty, and the fading warm start ("start at yesterday's fit") is
`ridge_decay`. With `standardize`, the solve is done in correlation form and
unscaled afterwards, dropping near-zero-variance features rather than
blowing up.

### `rls` — recursive least squares

```
A ← λA + w zzᵀ       b_j ← λb_j + w y_j z        β_j = A⁻¹ b_j
A₀ = ridge·I         b₀ = ridge·coef0
```

Coefficients move every row with no solve staleness. The state is the
Cholesky factor of `A`, updated by Givens rotations — the square-root form,
O(k²) per row like the textbook `P` recursion but without its two failure
modes (rounding asymmetry growing by `1/λ` per row, and one extreme row
cancelling `P` and freezing a coefficient for good). `ridge` sets
`A₀ = ridge·I` and, unlike `ewridge`, penalizes the intercept too.
Algebraically identical to `ewridge(ridge_decay=True)` solved every row; a
test holds them to <1e-9. A row with any null target is predict-only for all targets, since the
factor is shared.

### `lasso` — lasso path with free λ selection

Coordinate descent on the standardized statistics, warm-started along the
path and across solves:

```
ρ_i = c_i − Σ_{j≠i} C_ij β_j
β_i = soft(ρ_i, λ·l1_ratio) / (C_ii + λ(1 − l1_ratio))
```

`l1_ratio < 1` gives elastic net. Predictions for every path point are
computed anyway, so `lam_selected_<target>` — the argmin of an EW
out-of-sample squared error over the path — costs nothing extra, and is
reported as it stood *before* the row, like every other output.

### `kalman` — random-walk-β dynamic linear model

```
β_j ← Φβ_j    P_j ← ΦP_jΦ + Q·Δclock    Φ = diag(2^(−Δclock/r_i))
s   = zᵀP_j z + R_j/w                   k   = P_j z / s
β_j ← β_j + k(y_j − zᵀβ_j)              P_j ← P_j − k zᵀP_j
```

Process noise comes from a per-factor **coefficient halflife** on
standardized features, `q_i = σ²(ln2 / h_i)²`, matching the steady-state
gain of EW-RLS; `coef_halflife` may be a scalar or one value per slot, and
`inf` pins a coefficient. Observation noise defaults to the EW residual
variance. Standardization is internal and on by default. With
`standardize=False`, `q=0` and a fixed `obs_var`, this is exactly a Bayesian
linear regression (it reproduces river's `BayesianLinearRegression` to
3.6e-15).

**Reverting coefficients.** By default `Φ = I` and a coefficient is a random
walk: once a slope has been learned it stays until new rows move it. With
`revert_halflife`, a slope decays toward zero with the clock instead —
halved every `r_i` clock units while nothing is observed, and pulled back
toward zero by the same factor before each update. That is a mean-reverting
(AR(1)) prior: a regressor that is only occasionally active is forgotten
between its bursts rather than kept at its last value, and a stale effect
cannot persist through a run of null targets. The reversion acts in the
standardized coordinates, so "zero" means "no effect" for a slope and "the
target averages zero" for the intercept; a scalar applies to every slot
including the intercept, and a list with `inf` in the first slot exempts it.
The steady-state prior variance of a reverting slot is `q_i·Δclock/(1−φ_i²)`
instead of growing without bound. `predict` propagates the coefficients by
the same `Φ` over the distance from the last learned row (capped by
`max_dclock`), so a prediction far past the data is the intercept alone.

```python
revert = po.spec.kalman(
    "k",
    targets=["y"],
    features=["signal_a", "signal_b"],
    coef_halflife=100.0,
    revert_halflife=[float("inf"), 50.0, 50.0],  # intercept stays; slopes revert
    halflife=200.0,
    clock="t",
    max_dclock=10.0,
)
out = po.ModelBank([revert]).fit_predict(df)
```

### `huber` / `quantile` — robust regression

IRLS reweighting on the ridge update, using each row's *prior* residual so
the reweighting stays out-of-sample. Huber: `w = min(1, δσ/|r|)`; quantile:
the check-loss weights at level τ. Weights are per target, so `S` is per
target here.

### `sgd` — stochastic gradient descent

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

O(k) per row and no solves — the cheap baseline, and the only model here that
takes count targets (`loss="poisson"`). The learning rate is `constant`,
`inv_scaling` (`lr/(1+n_eff)^power`) or `adagrad`, whose accumulator decays
on the clock so an adapted rate re-opens after a long gap. `clip_gradient`
defaults to `1e3`, because with a log link one large count makes the next
gradient exponentially bigger; it does not bind for identity-link losses.

**Constrained coefficients.** `coef_min` and `coef_max` bound each slope,
and `coef_sum` fixes their total. After every update the slopes are moved to
the nearest point that satisfies all three (the Euclidean projection); the
intercept is never constrained. A bound is a number for every slope or a
list with one entry per feature, and `inf` means no bound on that side.
Portfolio weights that must be long-only and fully invested are
`coef_min=0.0, coef_sum=1.0`; a sign the model must respect is
`coef_min=0.0` alone; a slope pinned at a known value is `coef_min` equal
to `coef_max`. The fit starts from the projected zero (the uniform weights,
on a simplex), and `coef` reports what the projection returned, in the
caller's units even under `scale_features=True`.

```python
weights = po.spec.sgd(
    "w",
    targets=["y"],
    features=["signal_a", "signal_b", "x0"],
    halflife=200.0,
    learning_rate=0.01,
    coef_min=0.0,
    coef_sum=1.0,
    coef_every=1,
)
fit = po.ModelBank([weights]).fit_predict(df)
last = fit["w"].struct.field("coef").drop_nulls()[-1]
assert min(last[1:]) >= 0.0 and abs(sum(last[1:]) - 1.0) < 1e-12
```

### `pa` — passive-aggressive regression

```
loss = max(0, |y − p| − eps)      s = ‖z‖²
pa    τ = loss / s          pa1  τ = min(c, loss/s)      pa2  τ = loss / (s + 1/(2c))
β    += τ · sign(y − p) · z
```

Each row poses a constraint and the update is the smallest change that
satisfies it — no learning rate to tune. Plain `pa` moves the fit as far as
one bad row demands, so `pa1` is the default. PA keeps no accumulators, so
its coefficients have no halflife; the clock only drives `n_eff`. A row
weight below 1 scales `τ`; above 1 it counts as 1.

`pa` takes the same `coef_min`, `coef_max` and `coef_sum` as `sgd`, with
the projection applied after each update. The step then no longer meets
the row's margin exactly, and a truth outside the set is never reached, so
keep `c` small: each row moves the fit only as far as `c` allows and the
projection takes the rest back.

### `ew_cov` — exponentially weighted moments

```
W'   = λW + w        m'ᵢ = (λW·mᵢ + w·xᵢ) / W'      S'ᵢⱼ = (λW·Sᵢⱼ + w·xᵢxⱼ) / W'
varᵢ = Sᵢᵢ − mᵢ²     covᵢⱼ = Sᵢⱼ − mᵢmⱼ             corrᵢⱼ = covᵢⱼ / √(varᵢ·varⱼ)
```

Running mean, variance, std, covariance and correlation of the columns you
name, on the same clock as every model here — one O(k²) update per row,
where a pure-Polars pairwise EW correlation needs O(k²) *passes*. With
`precision_prior` set it also gives `partial_corr`, the correlation between
two columns controlling for all the others, read off `(C + s·prior·I)⁻¹`
(O(k³), paid only when asked for). Values are read from the state before each
row, so an `ew_cov` output can be a feature for that same row without leaking
it.

Three more outputs come off the same state. `"mahal"` in `stats` adds
`mahal`, the Mahalanobis distance of the row from the running mean,
`√(δᵀ (C + s·prior·I)⁻¹ δ)` with `δ = x − m` — how far the row is from what
the columns have been doing *together*, in standard deviations. A row with
every column in range but in a combination never seen before scores high here
and nowhere else; on Gaussian columns `mahal²` is χ² with k degrees of
freedom, and with one column it is `|z|`. It needs `precision_prior`, and the
prior fades like `partial_corr`'s. `mahal_quantiles=[0.99]` adds
`mahal_q0.99`, a P² quantile of the scores so far, so `mahal > mahal_q0.99`
is a distribution-free "one row in a hundred". `pca=r` adds the top `r`
eigenpairs of the covariance: `pc<j>_var`, `pc<j>_share` of the trace, the
loading on each feature `pc<j>_<feature>`, and the row's score on that
component `pc<j>_score`. The eigendecomposition costs O(k³), so
`pca_every=n` refreshes it every `n` rows and scores the rows in between on
the last loadings; each refresh keeps the sign of the previous one, so a
loading never flips between rows.

```python
mv = po.spec.ew_cov("mv", features=["x0", "x1", "x2"], clock="t", max_dclock=300.0,
                    halflife=500.0, stats=["mahal"], precision_prior=1e-6,
                    mahal_quantiles=[0.99], pca=1, pca_every=20)
scores = po.ModelBank([mv]).fit_predict(df).unnest("mv")
odd = scores.filter(pl.col("mahal") > pl.col("mahal_q0.99"))   # the joint outliers
first = scores.select("pc0_share", "pc0_x0", "pc0_x1", "pc0_x2", "pc0_score")
```

### `ftrl` — online logistic regression

FTRL-proximal (McMahan et al. 2013) for binary targets, with the accumulators
decayed on the same clock as everything else:

```
β_i = 0 if |z_i| ≤ l1 else −(z_i − sgn(z_i)l1) / ((β + √n_i)/α + l2)
p   = sigmoid(zᵀβ)     g_i = (p − y)·z_i·w
z_i += g_i − ((√(n_i + g_i²) − √n_i)/α)·β_i      n_i += g_i²
```

With `loss="logistic"` (default) `pred` is a probability and `resid = y − p`;
with `loss="squared"` it is the linear prediction — sparse linear regression
with no solves, and L1 support, which `ewridge` does not have.

### `holt` — Holt's linear trend

The one model that takes no features: it extrapolates the target's own level
and trend.

```
pred     = l + b·Δt
l' = α·y + (1−α)·pred        b' = β·(l' − l)/Δt + (1−β)·b
```

`α` and `β` come from `level_halflife` and `trend_halflife` in clock units;
the trend is per clock unit, so an irregular clock extrapolates the right
distance. `coef` is `[level, trend]` per target; `trend_halflife=inf` pins
the trend at zero, leaving a plain EW level. There is no seasonal term,
because a seasonal index is a `group` on the phase, which the bank already
does. Run it in the same bank as the real model to answer "how much is the
regression actually adding?" — compare `sigma`, or let `emit_selected`
choose.

```python
po.spec.holt("baseline", targets=["y"], clock="t", max_dclock=600.0,
             level_halflife=200.0, trend_halflife=2000.0)
```

### `kmeans` — exponentially weighted k-means

The one model with no target: it labels each row with the nearest of `k`
centres, read before the row is learned, so the label is out-of-sample like
every prediction here.

```
j*   = argmin_j ‖x − c_j‖²          distances in units of each feature's EW sd
n'_j = λn_j + w                      c'_j = c_j + (w/n'_j)(x − c_j)     for j = j*
```

Each centre is the EW mean of the rows assigned to it — `ew_cov`'s mean
recursion, per cluster. The struct holds `cluster`, `dist` (to the centre),
`dist2` (to the runner-up), `n_eff`, and `coef` = the centres, `k` rows of
`len(features)`.

```python
km = po.spec.kmeans("km", features=["x0", "x1", "x2"], k=3, clock="t",
                    halflife=2000.0, max_dclock=300.0, warm_rows=100)
out = po.ModelBank([km]).fit_predict(df).unnest("km")
po.spec.coef_index(km)        # target = "cluster0".., term = the feature
```

Seeding waits for `warm_rows` rows (default 500), then places the centres
with `seed_rule="lloyd"`: the best of ten k-means++ starts by inertia. One
start lands in the wrong partition a third of the time on five blobs in four
dimensions; the restarts tell them apart. The rows are replayed and the
buffer freed, so the model is O(k·p) from then on.

**What split–merge repairs.** A row far outside its cluster (about four
standard deviations of `dist²` above the typical radius) is scored but not
learned: it is summarised. Every `sm_every` rows the two closest centres are
compared, and if they are closer than `split_merge` times the sum of their
radii — two centres in one blob — one is freed and placed on the far rows,
provided enough have gathered to be a cluster's worth. A centre whose blob
vanished decays; once under `dead_frac` of an equal share it is re-placed
the same way. That takes `log2(1/dead_frac)` halflives: 4.3 at the default
0.05, 2 at 0.25. Raise `dead_frac` when regimes change faster than that;
the price is that a cluster lighter than `dead_frac/k` of the stream loses
its centre whenever any row is far. What the move cannot see is one centre
owning two blobs, whose rows are all within its own radius — seeding with
`lloyd` is what prevents it. Set `split_merge=0` for plain sequential
k-means.

### `micro` — density-based clustering, any shape

`kmeans` needs `k` and finds round clusters. `micro` finds clusters of any
shape, does not need their number, flags the rows that belong to none, and
follows clusters that appear and vanish. It is DenStream's micro-clusters
with a linkage step over them.

A summary is a small cluster: a decayed weight `n`, a centre `c` and a
radius `r`, the EW root-mean-square distance of its rows from the centre.
Each row goes to the nearest summary that can take it without its radius
passing `eps`, in units of each feature's EW sd. If none can, the row opens
one. A summary with `n ≥ beta_mu` is established. Every `prune_every` rows
the light summaries are dropped and the established ones are linked:
centres within `L` of each other share a label, and `L` is read from the
spacing the summaries already show unless `macro_link` sets it.

```
n_j  ← λ n_j                                               every summary
j*   = nearest summary that keeps  a r²_j + a b ‖x − c_j‖² ≤ eps² p,
       a = n_j/(n_j + 1),  b = 1/(n_j + 1);  else a new one at x
n_j* ← n_j* + w     c_j* ← c_j* + (w/n_j*)(x − c_j*)     r²_j* ← min(·, eps² p)
```

The struct holds `cluster` (the label of the nearest established summary,
null while there is none), `dist` (to its centre), `micro` (the id of the
summary the row goes to), `outlier` (no established summary takes it),
`n_clusters`, `n_micro`, `n_eff`, and `coef` = the established summaries,
one `[id, label, n, radius, c_1 … c_p]` row each. All are read before the
row is learned. Ids are monotone and never reused; a label is the smallest
id in its chain, so it outlives everything but that summary.

```python
mc = po.spec.micro("mc", features=["x0", "x1"], eps=0.1, clock="t",
                   halflife=2000.0, max_dclock=300.0, min_periods=50.0)
out = po.ModelBank([mc]).fit_predict(df).unnest("mc")
out.select("cluster", "outlier", "n_clusters", "n_micro").tail(3)
```

**Choosing `eps`.** It is the spread the model should read as *one*
cluster, per standardized coordinate: about 0.07 for two-dimensional
shapes, 0.3 for well-separated Gaussians in twenty dimensions. Both ways to
get it wrong show in the outputs. If nearly every row is an `outlier` and
`cluster` stays null, `eps` is too small: no summary reaches `beta_mu`
before it is pruned. If `n_micro` is about the number of clusters, `eps` is
too coarse: each cluster is one summary, so the derived `L` reads the
spacing *between* clusters and bridges them into one. Lower `eps`, or set
`macro_link=2` to link only summaries that touch.

Measured at 20k rows with the `eps` above: moons, rings and five Gaussians
in twenty dimensions all score ARI 1.000 against the truth, where `kmeans`
cannot follow the first two. Noise drawn uniformly over the box is flagged
`outlier` 94% of the time, real rows 0.3%. A cluster born mid-stream has a
label within 200 rows; one whose rows stop lingers `halflife · log2(n /
beta_mu)`, with `n` the weight it had.

### `ew_class` — Gaussian classification on `ew_cov` moments

A label column in place of a numeric target. The model keeps one `ew_cov`
state per class — a weight `n_c`, a mean `μ_c` and a centered covariance
`C_c` — and scores a row by Bayes' rule over Gaussian classes. `covariance`
picks the shape. `"full"` gives each class its own covariance: QDA.
`"shared"` pools them, weighted by the class weights: LDA. `"diagonal"`
keeps only the variances: Gaussian naive Bayes. `precision_prior` is the
ridge that makes a class scoreable from its first row, and it fades the way
`ew_cov`'s does.

```
π_c = n_c / Σ n         r_c = precision_prior · s_c        (s_c: the prior's fade)
M_c = C_c + r_c I  (full)      M = Σ π_c M_c  (shared)      diag(C_c) + r_c  (diagonal)
ℓ_c = ln π_c − ½ ln det M_c − ½ (x − μ_c)ᵀ M_c⁻¹ (x − μ_c)
p_c = exp(ℓ_c − max ℓ) / Σ exp(ℓ − max ℓ)                  class = argmax ℓ
n_c ← λ n_c + w·[y = c]        μ_c, C_c ← weighted Welford on the row's own class
```

The struct holds `class`, the most probable class as a string; one
`p_<class>` per declared class; `n_eff`; and `coef`, the class means in the
order of `classes` (`coef_up_x0` after `unnest`). All are read before the row
is learned, so a row's posterior never saw its own label. A class no row has
carried yet has `p = 0` exactly and null means. A null label scores the row
and learns nothing from it — so a stream whose labels arrive late is scored
by nulling the label and keeping the features. A label the spec does not list
is an error naming the row, the value and the classes. Integer and boolean
columns work as labels through their text: `classes=["0", "1"]`,
`classes=["true", "false"]`.

```python
labelled = df.with_columns(
    pl.when(pl.col("y") > 0).then(pl.lit("up")).otherwise(pl.lit("down")).alias("dir")
)
cl = po.spec.ew_class("cl", features=["x0", "x1", "x2"], label="dir", classes=["down", "up"],
                      covariance="shared", precision_prior=0.1, clock="t",
                      halflife=200.0, max_dclock=300.0, min_periods=20.0)
out = po.ModelBank([cl]).fit_predict(labelled).unnest("cl")
out.select("dir", "class", "p_up", "n_eff").tail(3)
```

**Choosing the shape.** `"full"` is the general case and costs one `k×k`
Cholesky per class per row. `"shared"` factorizes once per row, and is the
right model when the classes differ in location but not in spread — it then
matches `"full"` to a fraction of a percent on the test data, with fewer
parameters to learn. `"diagonal"` is the cheapest and cannot see a
correlation: two classes with the same marginals and opposite correlations
are one class to it. Measured at 400k rows, six features and three classes:
0.9M rows/s full, 1.8M shared, 5M diagonal. On three Gaussian classes with
their own covariances the accuracy sits within 0.001 of the Bayes rate the
generating parameters allow, and the posteriors are calibrated to about 0.01.

### `seqtest` — a sequential test of a sign, by betting

Not a regression. A `seqtest` asks whether a column tends to be positive —
or, with `a` and `b`, whether one spec of the bank predicts closer than
another — and answers with evidence you can read at any row, as often as you
like, and act on the first time it is enough. A p-value cannot be used that
way; an e-process can. Per target it keeps the wealth of two gamblers, one
betting that the next sign is positive and one that it is negative. Each
stakes the Krichevsky–Trofimov fraction set by the counts so far, and never
bets against its own lead:

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
the Beta(½, ½) mixture, and the tests hold the bank to it; the two sides'
average is an e-value for the two-sided question. The struct holds
`log_e_pos_<t>`, `log_e_neg_<t>`, `n_pos_<t>`, `n_neg_<t>` and `n_eff`, all
as they stood **before** the row. A zero, a null or a NaN is a tie: it bets
nothing and counts nothing. A trial is a row, so there is no `weight` and no
`halflife` — a spec that gives them is refused — and `session` or
`on_clock_reset="reset_state"` restarts the test.

With `a` and `b` the test compares two specs of the bank: the sign tested is
`|resid_b| − |resid_a|`, positive when `a` came closer, and the fields are
`log_e_a_<t>`, `log_e_b_<t>`, `wins_a_<t>`, `wins_b_<t>`. The bank runs a
comparison after the specs it names, on the out-of-sample residuals their
structs report; a row where either side is null — warm-up, a skipped row —
is no trial. `a_suffix` and `b_suffix` pick a grid instance (`"@h500"`,
`"__r0.5@h500"`). A comparison inside a bank is chunk-invariant, saved with
the state and streams like everything else; `po.eval.seqtest` is the same
computation over a frame you already have.

```python
common = dict(targets=["y"], features=["x0", "x1"], clock="t", max_dclock=300.0, group="bond_id")
ridge = po.spec.ewridge("ridge", halflife=500.0, **common)
kalman = po.spec.kalman("kalman", halflife=500.0, coef_halflife=100.0, **common)
closer = po.spec.seqtest("closer", targets=["y"], a="kalman", b="ridge", group="bond_id")
out = po.ModelBank([ridge, kalman, closer]).fit_predict(df)
verdict = out.group_by("bond_id").agg(pl.col("closer").struct.field("log_e_a_y").max())
# log_e_a_y >= ln(20): on that bond, kalman beat ridge at the 5% level, read at any row
```

## Parallelism

The unit of work is a *stream*: one spec on one group (with no `group`, one
stream per spec). On every chunk, each stream in the bank becomes one task on
the bank's own thread pool (a [rayon](https://github.com/rayon-rs/rayon)
pool, separate from polars') — one flat pool across all specs and all
groups, longest stream first so a few big groups do not leave cores idle at
the tail. Within a stream the rows go one at a time, because
each row's update depends on the last: that is what makes the numbers
independent of how the work is split, and it also means a bank with one spec
and one group is one thread's work per chunk, with polars' own reading and
writing running in parallel around it.

So a bank fills the pool with groups, with specs, or with both. A search
over factor sets is a list of specs, one per set: each is its own
accumulator — its own standardization, and a null in a factor it does not
use costs it nothing — with its own grid inside, and every one is a task.
(Subsets of one list that should share an accumulator are `feature_sets`
on one spec: one solve each, not one task.) The list runs as one plan in
one pass, with the thread counts set before anything is built:

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
                           group="bond_id", session="session", session_gap=60.0,
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
groups that plan takes 12.3 s at one thread and 2.2 s at fourteen; the
three-factor spec alone goes from 2.5 s to 0.62 s, because with one task
per group the fixed cost of reading and assembling each chunk shows
through. The output is one struct column per spec, which is what
`compare_specs` reads, and one state file holds them all. The same list
runs the same way through `ModelBank`, `po.run` and the CLI.

Where the parallelism comes from, then:

- **Groups.** k=20 over 64 groups: 1.02M, 1.91M, 3.52M, 6.44M and 8.20M
  rows/s at 1, 2, 4, 8 and 14 threads — **8.0×** on a 14-core machine.
- **Specs.** Eight single-group specs in one bank run in 130 ms against
  515 ms one at a time.
- **Halflives.** Each halflife in a grid is its own accumulator, and the
  instances of a stream run alongside each other (except with
  `drift_action="reset"`, which couples them). Ridge and feature-set grids
  are not parallel because they need not be: they share one accumulator and
  are expanded at solve time.
- **The runner.** `po.run` and the CLI are a three-stage pipeline — a reader
  thread, the bank on the calling thread, a writer thread — with one chunk
  in flight per stage; `ONLINE_TIMING=1` prints how long the bank waited on
  each side. Reading and writing are polars' work on polars' pool: parquet
  pages are encoded a column at a time there, NDJSON a slice per thread.
- **Python.** The GIL is released while a chunk is in the bank, so a Python
  reader thread can run ahead of `ModelBank.fit_predict`, and independent
  `po.run` calls in threads of one process share the one pool.
- **The expression form.** Under `.over("group")`, polars runs the groups
  through its own pool — which is why the plugin packs its inputs into one
  struct: the single-input path is parallel, the multi-input one is not
  (12.2M rows/s at 1000 groups).

Thread count is `POLARS_ONLINE_MAX_THREADS` for the bank's pool and
`POLARS_MAX_THREADS` for polars' readers and writers; unset, each is one
thread per core. The bank builds its pool at the first bank call and polars
its own at import, so each must be set before that point, as above, or in
the shell (`POLARS_ONLINE_MAX_THREADS=8 python fit.py`), which is the form
that always works; set later, the variable is ignored, and
`po.thread_pool_size()` says what took (`pl.thread_pool_size()` for
polars'). A value that is not a count is refused by name at the first bank
call. It changes the speed and nothing else: `tests/test_portability.py`
runs the same stream at 1 and 8 threads in separate processes and requires
identical output. Everything is one process — there is no distributed
execution, by design (see [What this is not](#what-this-is-not)).

Two knobs because the two counts do different things. Polars' also sizes
what its parquet reader holds in flight — it prefetches row groups ahead of
the consumer, so more threads is a bigger pile of decoded rows
([Memory](#performance), below) — while the bank's count buys speed and
nothing else. So a run that has to fit in a smaller box keeps polars small
and gives the bank every core:

```python
import os
os.environ["POLARS_MAX_THREADS"] = "4"           # the reader's prefetch is sized from this
os.environ["POLARS_ONLINE_MAX_THREADS"] = "14"   # the bank still has every core

import polars as pl
import polars_online as po

(pl.scan_parquet("ticks.parquet")
   .online.fit_predict([spec], chunk_rows=200_000)
   .sink_parquet("fit.parquet"))
```

On 12M rows over 64 groups, one spec: 14 and 14 takes 2.6 s at a peak of
1.1 GB. 4 and 14 takes the same 2.6 s at 0.8 GB — a third less memory for
free. One shared count of 4 takes 3.9 s at 0.6 GB, and polars alone at one
thread 7.4 s, because reading and writing are then one thread's work. Six
specs split the same way: 10.3 s at 1.5 GB, 11.8 s at 1.2 GB, 16.6 s at
1.0 GB. (Memory here and below is the peak footprint `/usr/bin/time -l`
reports. RSS reads about 0.7 GB higher, because the memory-mapped input
file counts there.) The pools never wait on each other — a bank task never
calls back into polars' pool — so both at the whole machine costs nothing
either: 28 and 28 on 14 cores ran the grid above in 2.18 s against 2.21.

### Chunk size

`chunk_rows` is how many rows the bank takes at a time. It is a keyword on
`lf.online.fit_predict` and `lf.online.predict`, on `po.run`, and on the
CLI (`--chunk-rows`, or `chunk_rows` in the TOML); the default is 100,000.
With `ModelBank.fit_predict(df)` the chunk is whatever frame you pass.

It never changes the numbers. One chunk or a thousand gives the same
output; the one thing that moves is where `coef` lands, because each stream
reports its coefficients on its last row of every chunk. `coef_every` gives
it a cadence that does not move.

What it changes is speed and memory, and two things pull against each
other:

- **A chunk can only run the groups it holds.** If the file is sorted by
  group, a 100k chunk holds one or two groups, so the bank has one or two
  tasks per chunk and most cores sit idle. Bigger chunks fix that.
- **Bigger chunks lose the overlap.** Reading, fitting and writing run side
  by side, a chunk apart. With huge chunks the stages spend more time
  waiting on each other, and the chunks in flight (about three) cost
  memory.

The same 12M rows and 64 groups, one spec, 14 threads:

| `chunk_rows` | groups interleaved | sorted by group | peak memory |
|---|---|---|---|
| 20,000 | 2.7 s | 9.2 s | 1.0 GB |
| 50,000 | 2.5 s | 8.8 s | 0.9 GB |
| 100,000 (default) | 2.4 s | 8.1 s | 1.0 GB |
| 200,000 | 2.6 s | 7.1 s | 1.1 GB |
| 500,000 | 2.8 s | 4.6 s | 1.5 GB |
| 1,000,000 | 3.2 s | 4.2 s | 1.8 GB |
| 2,000,000 | 4.5 s | 6.0 s | 2.4 GB |

(The memory column is the interleaved file's; the sorted file is within
0.1 GB of it, except 2.6 GB at 2M.) So:

- **Groups mixed through the file** — tick data in time order — leave the
  default. Everything from 50k to 500k is within 0.4 s of it.
- **Sorted or clustered by group** — raise it until a chunk spans several
  groups: a few times the rows per group. This file has about 190k rows
  per group, and 1M, five groups a chunk, is twice as fast as the default.
  Past that the overlap goes and memory climbs; the interleaved file shows
  the cost, with 2M nearly twice the default's time.
- **A smaller box** — lower it, but expect little below the default: most
  of the first gigabyte is polars' reader prefetch, not the chunks, and
  `POLARS_MAX_THREADS` or `POLARS_ROW_GROUP_PREFETCH_SIZE` is what shrinks
  that ([Memory](#performance), below).

## Performance

Apple M-series, single process, best of 3, 200k rows per run
(`uv run python scripts/benchmark.py --markdown`):

| configuration | notes | rows/sec |
|---|---|---|
| `ewridge` k=5 | 1 target, 1 halflife | 10,870,697 |
| `ewridge` k=20 | 1 target, 1 halflife | 3,923,931 |
| `ewridge` k=50 | 1 target, 1 halflife | 1,002,231 |
| `ewridge` k=20 | 10 targets | 2,330,458 |
| `ewridge` k=20 | 5 halflives | 2,354,953 |
| `rls` | k=20, 1 target | 1,963,308 |
| `kalman` | k=20, 1 target | 1,851,327 |
| `lasso` | k=20, 1 target (3-point path) | 2,033,627 |
| `huber` | k=20, 1 target | 3,975,119 |
| `ftrl` | k=20, 1 target | 6,995,830 |

Targets share one `S` accumulator, so 10 targets cost far less than 10× one.
Each halflife in a grid is its own accumulator, but they run in parallel, so
a 5-halflife grid costs about 2× one rather than 5×. `rls` pays 1.3–2.1× for
the square-root form that keeps it from dying of cancellation on one extreme
row; that is worth it.

The other families, and the options that add a pass, on the same machine
and rows:

| configuration | notes | rows/sec |
|---|---|---|
| `ewridge` + `conformal` | k=20, 90% interval | 3,917,766 |
| `sgd` | k=20, squared loss | 11,578,457 |
| `sgd` | k=20, `coef_min=0`, `coef_sum=1` | 2,766,116 |
| `pa` | k=20 | 15,681,764 |
| `kalman` | k=20, `revert_halflife` | 1,617,287 |
| `ew_cov` | k=20: mean, std, corr (230 statistics) | 1,921,119 |
| `ew_cov` | k=20: mean, mahal, `mahal_q0.99` | 763,732 |
| `ew_class` | k=20, 3 classes, full covariance | 508,747 |
| `ew_class` | k=20, 3 classes, shared covariance | 586,333 |
| `ew_class` | k=20, 3 classes, diagonal | 2,690,399 |
| `kmeans` | 4 features, K=8 | 6,747,259 |
| `kmeans` | k=20, K=8 | 3,317,180 |
| `micro` | 4 features, `eps=1` | 18,391,650 |
| `seqtest` | sign of one column | 27,480,862 |

A conformal interval is free: it reads the residual the model already has.
A simplex constraint sorts `2k` breakpoints per row, so it costs `sgd`
about 4×. `ew_cov` writes 230 numbers a row and still runs at half the
speed of one `ewridge`. The Mahalanobis distance and the full-covariance
`ew_class` each pay for a Cholesky factor of a `k × k` matrix, one per row
for `mahal` and one per *learned* row for `ew_class` — the classes a row
does not touch keep theirs. `kmeans` and `micro` cost a distance to each
centre; `seqtest` a handful of operations.

Grouped data goes wider, as [Parallelism](#parallelism) shows: 8.2M rows/s
at k=20 over 64 groups.

**Memory** is the state, the chunks in flight (three, so `chunk_rows` is the
knob — [Chunk size](#chunk-size), above) and whatever polars' reader prefetches — on a 14-thread machine the
parquet reader front-loads ~0.7 GB of decoded row groups whatever the file's
length, and `POLARS_ROW_GROUP_PREFETCH_SIZE=1` takes the CLI to 0.15 GB at
the same speed. The prefetch is sized from the thread count, so
`POLARS_MAX_THREADS` shrinks it too. Where the time goes, and what to reach
for, is in [docs/PERFORMANCE.md](docs/PERFORMANCE.md).

## What this is not

A model layer, not a stream-processing framework. It expects a frame that is
already aligned — and, when a spec names a `clock`, each group's rows in
clock order — and it keeps O(state) per stream. It deliberately does **not**
provide:

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

The Rust `polars` is pinned exactly and linked into the wheel; the runtime
requirement is a range, because the two copies never meet. The floor is
`LazyFrame.collect_batches`, which `po.run` and `lf.online.fit_predict` read
with and py-polars added in 1.34.0; the whole suite passes on 1.34.0, 1.38.1
and 1.44.1 with identical numbers. `ModelBank` and the expression form alone
work from 1.28.1 (tested across 17 releases). `tests/test_scaffold.py`
asserts the pins; the matrix is in
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

The two that stream are the two without a promise, so a break on a new
Polars is expected maintenance, not a surprise.

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
changes, so pin `~=0.2.0` if you need stability. Widening the Polars range
is a minor release; narrowing it is breaking. See
[CHANGELOG.md](CHANGELOG.md). Output field names are part of the API
([above](#output-field-names)).

## Testing

The guarantees above are only worth what checks them, so the suite is built
around oracles and invariants rather than expected values typed in by hand.
Around 470 Rust tests and 1,700 pytest cases (from 920-odd functions), all
green on three OSes; [docs/TESTING.md](docs/TESTING.md) is the ledger of what
each part proves and what it has found.

**Against references.** `ewridge` and `rls` match numpy references in
`tests/reference.py` to 1e-9, `kalman` to ~1e-15, `huber` and `quantile` to
~1e-13, `ftrl` to ~1e-16; `rls` equals `ewridge(ridge_decay=True)` solved
every row to <1e-9. The lasso is checked against the KKT conditions of its
objective rather than a ported solver, which cannot share a bug with it.
[river](https://riverml.xyz) is an independent implementation of several of
the same algorithms: its FTRL recursion agrees with ours to 1e-12 row for
row, its EW moments in closed form and in the limit, its quantile and Huber
models statistically — and two convention differences are pinned as tests
rather than left as surprises.

**Invariants, for all ten models.** Chunk invariance at the bank, expression
and CLI levels (one chunk, seven, four hundred, one row at a time, with a
save and load in the middle); thread count 1 against 8; group independence;
expression ≡ bank; `predict` ≡ `fit_predict` of the next row, field for
field with every diagnostic on; runner ≡ bank for every input source and
format; the null policy, warm-up and clock semantics; the same `n_eff`
recursion in every model (`crates/online-core/tests/model_contract.rs`).
Hypothesis generates adversarial streams — mixed nulls, duplicate and
long-gap clocks, values at ±1e8, zero weights, tiny groups — and asserts the
strongest one: **changing a row's own target never changes that row's own
prediction.** IC ≈ 0 on pure-noise targets says the same thing from the
other side.

**Fixed numbers.** One golden stream per model in the Rust core, and the
whole pipeline — extraction, fan-out, diagnostics, struct assembly — pinned
to fixed output and compared on every OS, so a divergence in polars'
vectorized paths on another CPU would show.

**Hardening.** A 30k-row stream with every output switched on, compared
across chunkings, a mid-stream save and load, and thread counts by digest;
weight-scale invariance at ×1e±6 (all weights scaled changes nothing but
`n_eff`); parameter edges from `halflife=1e-3` to `inf`; state files with
any byte flipped fail cleanly and never panic; two threads calling
`fit_predict` at once get a clean error; `pickle` and `copy.deepcopy` resume
bit-exactly; memory safety across the FFI, where two copies of Polars share
one process; and a 10M-row soak, opt-in with `pytest -m soak`.

**Contracts that are files.** The public API — every name, default and
signature, every output field name — is a checked-in snapshot
(`tests/api_surface.txt`), so a change is a reviewable diff. Every python
block in this README runs. Everything under `examples/` runs unmodified —
the TOML through the real CLI, the Pathway operator end to end.
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
uv run python scripts/benchmark.py                     # throughput
```

Prerequisites: [uv](https://docs.astral.sh/uv/) and a stable Rust toolchain
([rustup](https://rustup.rs)). `source scripts/env.sh` (`. .\scripts\env.ps1`
in PowerShell) puts both on the `PATH` for a shell; `.vscode/settings.json`
does it for VS Code's terminal. `cargo` runs via `uv run` because `online-py`
builds against pyo3's `abi3-py312` and needs a 3.12+ interpreter at build
time.

- API reference: <https://hgilde.github.io/polars-online/> — built from the
  docstrings and published from every green push to `main`
- Design and task list: [docs/PLAN.md](docs/PLAN.md)
- Saving, serving, resuming: [docs/STATE-WORKFLOW.md](docs/STATE-WORKFLOW.md)
- Measured defaults: [docs/VALIDATION.md](docs/VALIDATION.md)
- Where the time and memory go: [docs/PERFORMANCE.md](docs/PERFORMANCE.md)
- Adding a model: [docs/EXTENDING.md](docs/EXTENDING.md)

## License

Apache-2.0. See [CONTRIBUTING.md](CONTRIBUTING.md) to make changes,
[SECURITY.md](SECURITY.md) to report a vulnerability.
