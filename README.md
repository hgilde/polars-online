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

**Ten model families, one set of stream semantics.** A spec's clock, decay,
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
| `ew_cov` | running mean, variance, covariance, correlation and partial correlation |
| `holt` | Holt's linear trend — the no-feature baseline |

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

**Tested the way those guarantees demand.** Around 290 Rust tests and 1,100
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

Wheels for macOS (arm64, x86_64), Windows x64 and Linux (x64, aarch64; glibc
and musl) plus the CLI binaries are attached to each release. Python 3.12+.
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

Drift detection complements the halflife rather than replacing it: decay
forgets smoothly and always; a detector notices a break and says so, within
a couple of rows of a sign flip.

After the fact, `po.eval` reads the output frame:

```python
po.eval.metrics(out, "ridge", by=["bond_id"])                       # R², IC, hit rate, MSE
po.eval.rolling_metrics(out, "ridge", clock="t", window=3600.0)     # per clock window
po.eval.compare_specs(out, ["ridge", "kalman"])                     # one table, many specs
```

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
P_j ← P_j + Q·Δclock                s = zᵀP_j z + R_j/w
k   = P_j z / s                     β_j ← β_j + k(y_j − zᵀβ_j)
P_j ← P_j − k zᵀP_j
```

Process noise comes from a per-factor **coefficient halflife** on
standardized features, `q_i = σ²(ln2 / h_i)²`, matching the steady-state
gain of EW-RLS; `coef_halflife` may be a scalar or one value per slot, and
`inf` pins a coefficient. Observation noise defaults to the EW residual
variance. Standardization is internal and on by default. With
`standardize=False`, `q=0` and a fixed `obs_var`, this is exactly a Bayesian
linear regression (it reproduces river's `BayesianLinearRegression` to
3.6e-15).

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

## Parallelism

The unit of work is a *stream*: one spec on one group (with no `group`, one
stream per spec). On every chunk, each stream in the bank becomes one task on
a [rayon](https://github.com/rayon-rs/rayon) pool — one flat pool across all
specs and all groups, longest stream first so a few big groups do not leave
cores idle at the tail. Within a stream the rows go one at a time, because
each row's update depends on the last: that is what makes the numbers
independent of how the work is split, and it also means a bank with one spec
and one group is one thread's work per chunk, with polars' own reading and
writing running in parallel around it.

Where the parallelism comes from, then:

- **Groups.** k=20 over 64 groups: 916k, 1.65M, 2.86M, 4.75M and 6.03M
  rows/s at 1, 2, 4, 8 and 14 threads — **6.6×** on a 14-core machine.
- **Specs.** Eight single-group specs in one bank run in 118 ms against
  685 ms one at a time.
- **Halflives.** Each halflife in a grid is its own accumulator, and the
  instances of a stream run alongside each other (except with
  `drift_action="reset"`, which couples them). Ridge and feature-set grids
  are not parallel because they need not be: they share one accumulator and
  are expanded at solve time.
- **The runner.** `po.run` and the CLI are a three-stage pipeline — a reader
  thread, the bank on the calling thread, a writer thread — with one chunk
  in flight per stage; `ONLINE_TIMING=1` prints how long the bank waited on
  each side. NDJSON output is serialized a slice per thread.
- **Python.** The GIL is released while a chunk is in the bank, so a Python
  reader thread can run ahead of `ModelBank.fit_predict`.
- **The expression form.** Under `.over("group")`, polars runs the groups
  through its own pool — which is why the plugin packs its inputs into one
  struct: the single-input path is parallel, the multi-input one is not
  (12.2M rows/s at 1000 groups).

Thread count is `RAYON_NUM_THREADS` for the bank's pool and
`POLARS_MAX_THREADS` for polars' readers and writers. It changes the speed
and nothing else: `tests/test_portability.py` runs the same stream at 1 and
8 threads in separate processes and requires identical output. Everything is
one process — there is no distributed execution, by design (see [What this
is not](#what-this-is-not)).

## Performance

Apple M-series, single process, best of 3, 200k rows per run
(`uv run python scripts/benchmark.py --markdown`):

| configuration | notes | rows/sec |
|---|---|---|
| `ewridge` k=5 | 1 target, 1 halflife | 8,961,460 |
| `ewridge` k=20 | 1 target, 1 halflife | 3,620,024 |
| `ewridge` k=50 | 1 target, 1 halflife | 960,926 |
| `ewridge` k=20 | 10 targets | 1,906,032 |
| `ewridge` k=20 | 5 halflives | 2,158,579 |
| `rls` | k=20, 1 target | 1,927,999 |
| `kalman` | k=20, 1 target | 1,661,270 |
| `lasso` | k=20, 1 target (3-point path) | 1,878,137 |
| `huber` | k=20, 1 target | 3,680,996 |
| `ftrl` | k=20, 1 target | 6,288,122 |

Targets share one `S` accumulator, so 10 targets cost far less than 10× one.
Each halflife in a grid is its own accumulator, but they run in parallel, so
a 5-halflife grid costs about 2× one rather than 5×. `rls` pays 1.3–2.1× for
the square-root form that keeps it from dying of cancellation on one extreme
row; that is worth it.

Grouped data goes wider, as [Parallelism](#parallelism) shows: 6.0M rows/s
at k=20 over 64 groups.

**Memory** is the state, the chunks in flight (three, so `chunk_rows` is the
knob) and whatever polars' reader prefetches — on a 14-thread machine the
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
- **distributed execution** — one process, `rayon` across (spec × group).

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
changes, so pin `~=0.1.0` if you need stability. Widening the Polars range
is a minor release; narrowing it is breaking. See
[CHANGELOG.md](CHANGELOG.md). Output field names are part of the API
([above](#output-field-names)).

## Testing

The guarantees above are only worth what checks them, so the suite is built
around oracles and invariants rather than expected values typed in by hand.
Around 290 Rust tests and 1,100 pytest cases (from 630-odd functions), all
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
