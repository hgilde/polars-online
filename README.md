# polars-online

Streaming / online regression models for [Polars](https://pola.rs). A Rust core
exposed two ways with identical numerics:

1. **a chunk-fed `ModelBank`** — holds O(state), not O(data) — and, as a plan,
   **`lf.online.fit_predict(specs)`**: a `LazyFrame` that streams through the
   bank when it runs, so a query stays O(chunk) (`df.online.fit_predict(specs)`
   for a frame in memory);
2. **a streaming runner** — `po.run(...)` or the standalone `online` CLI:
   parquet, ipc, csv or ndjson in and out, config from TOML.

There is also an expression form, `pl.col("y").online.ewridge(...)`, for a
frame already in memory. It cannot stream — polars hands it the whole column
— so it warns on every use; [the note at the end](#the-expression-form)
shows it next to the plan and says why.

Built for data that does not fit in memory, in either of two shapes. A plain
table in any row order: with decay off (`halflife=inf`) the bank *is* least
squares over every row it has seen — `numpy.linalg.lstsq`'s coefficients to
2e-13, at O(state) instead of O(data) — and a row halflife makes that a
recency-weighted fit. And ordered event streams (one per group), which is
where the rest of the machinery lives: a clock column, irregular spacing,
session breaks, gaps, nulls, and per-group state.

> **Polars moves, and two of the three interfaces this rides on carry no
> stability promise.** The expression form uses polars' plugin ABI, which is
> negotiated (a mismatch refuses to load); `ModelBank` crosses on
> pyo3-polars' `PyDataFrame`/`PySeries`, which polars provides "for
> convenience" with no guarantee beyond the latest definitions working for
> the latest Polars; and the plan form is an IO plugin, `@unstable` in
> py-polars. So `polars>=1.34.0,<2` is measured — 1.34.0, 1.38.1 and 1.44.1
> pass the whole suite with identical numbers — not promised, and a weekly
> **canary** ([`polars-canary.yml`](.github/workflows/polars-canary.yml))
> installs the newest py-polars and runs the suite on it. A red canary is
> the notification, and the response is decided in advance: cap the range at
> the last release that passed, in a patch release, so no resolver hands you
> the broken pair; then fix, and widen again.
> [Versioning and the Polars pin](#versioning-and-the-polars-pin) has the
> mechanics and where to look first when it breaks.

```python
import polars as pl
import polars_online as po

spec = po.spec.ewridge("ridge", targets=["ret"], features=["signal_a", "signal_b"],
                       clock="ts", halflife=600.0, max_dclock=300.0, group="bond_id")

(pl.scan_parquet("ticks/*.parquet")           # a stream: the bank, as a plan
   .online.fit_predict([spec])
   .filter(pl.col("ridge").struct.field("n_eff") > 100)
   .sink_parquet("fitted.parquet"))           # O(chunk) end to end

df.online.fit_predict([spec])                 # a frame in memory: the same bank, eagerly
```

## Which calls stream

All of them but one, and that one says so. One model, one set of numbers,
and the memory is O(chunk) whichever way you call it — peak footprint on
the same file, `ewridge` with 20 features, parquet in and out:

| what you write | 3M rows | 12M rows | |
|---|---:|---:|---|
| `lf.online.fit_predict([spec])` | 0.90 GB | 1.35 GB | **O(chunk)** — a query over a stream |
| `for chunk in lf.collect_batches(): bank.fit_predict(chunk)` | 0.80 GB | 1.24 GB | O(state + chunk) — your own loop |
| `po.run(input=..., output=...)`, `online --config` | 0.95 / 0.73 GB | 1.41 / 0.75 GB | O(state + chunk) — file in, file out |

All three are flat, and what growth they show is the allocator holding
freed pages rather than live data: told to release them, all three sit at
0.74–0.86 GB at 12M rows, nearly all of it polars' parquet read-ahead
(0.31–0.46 GB with `POLARS_ROW_GROUP_PREFETCH_SIZE=1`). Everything after
the bank — filters, joins, group-bys, sinks — is polars' own and streams as
polars streams it. All of it measured in
[docs/PERFORMANCE.md](docs/PERFORMANCE.md) §11.

The one call that is not in the table is the expression form — the
same model written as `pl.col("y").online.ewridge(...)` inside
`with_columns`. It measures **7.3 GB** on the same 12M rows against the
plan's 1.35 GB, because polars hands a user expression its whole column in
either engine. It is kept for a frame in memory and warns on every call;
[the note at the end](#the-expression-form) puts the two side by side and
explains the number.

**This is polars' rule, not ours**, and it decides what streams *around* the
bank too: polars' own windowed operations split the same way, and the rule
is whether the streaming engine has a node for that call. On the same
12M rows, `pl.col("y").mean().rolling(index_column="t", period="1000i")`
peaks at 0.25 GB and the identical expression under **`.over("group")` at
6.5 GB**; `lf.rolling(index_column="t", period="1000i").agg(...)` at 0.28 GB
and the same call with **`group_by="group"` at 1.7 GB**. Both of the
collecting calls land on the engine's `in-memory-map` node — collect, run in
memory, re-emit — the same class of node a user expression gets
(`columnar-function`: collect, call once, re-emit), because polars'
expression contract has no way to say "call me per morsel, in order, and
let me keep state". Note where that leaves per-group work: in polars, the
*grouped* window is the one that collects; in a bank, `group=` is one
accumulator per group and stays O(state).

## Two guarantees

- **Predictions are out-of-sample by construction.** Every row is predicted from
  the state *before* that row's target is folded in. Nothing here can leak.
- **Chunk invariance.** Feeding a stream as 1 chunk or 1000 chunks produces
  bit-identical output, and so does saving state mid-stream and resuming. The
  one exception is `coef`, which is a *reporting* cadence rather than a
  computed value: it is snapshotted every `coef_every` rows **and** on each
  chunk's last row, so smaller chunks report it more often. Every other field
  is identical.

Both are enforced by tests, not just intended.

## Any row order

A bank is a set of sufficient statistics, so row order reaches the fit only
through decay. With decay off — `halflife=inf`, or `lam=1.0` — an `ewridge`
with `ridge=0` is ordinary least squares over every row it has seen, whatever
order they came in. Forwards, backwards or shuffled, with the groups
interleaved however the file has them, its coefficients match
`numpy.linalg.lstsq` to 2e-13. What it costs is state, not data: 6M rows × 20
features from a parquet stream peak at 1.4 GB against 3.97 GB for
`lstsq` on the same rows, and the frame never has to fit. By default it
re-solves on every row (11 s there); `solve_every=1000` makes it 1.4 s with
the coefficients at most 1000 rows stale (2e-6 off). Everything the
streaming shape gets — `group`, `weight`, `min_periods`, null policy, the
ridge and feature-set grids, `save`/`load` — applies as it stands.

A finite halflife with no `clock` is the same fit with each row discounted by
how far back in the stream it sits: the weighted least squares of *that*
order, so reversing the rows reverses the weights. On rows that carry no
order of their own that is a random weighting, and its fit differs from OLS
by sampling noise only (halflife 1e6 rows on the 6M above: 4e-4 apart, with
OLS itself 5e-4 from the truth). One thing to know: a huge *finite* halflife
is not `inf`. Its solve cadence defaults to `halflife/50`, so `halflife=1e12`
solves once, at `min_periods`, and stays there (0.3 off on the same data).
Say `inf`, or set `solve_every`. `tests/test_row_order.py` pins all of this.

## Install

```sh
uv sync
uv run maturin develop --release -m crates/online-py/Cargo.toml
```

Wheels and CLI binaries for macOS, Windows and Linux are attached to each
release. The wheel is ~19 MB to download and ~59 MB installed: it statically
links the Rust half of Polars, so nothing beyond `polars` itself has to be
present at run time. `numpy` is an optional extra, needed only by
`ModelBank.gram()`.

## The two usage modes

### 1. `ModelBank` (chunk-fed)

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
    ...                                    # (or po.run(input=lf, output=...) -- §2: the same
                                           #  loop, pipelined, written straight to a file)

bank.save("bank.state")                    # atomic: temp file, then rename
```

A bank says what it holds: `repr(bank)` is `ModelBank(['ridge'], groups=412,
rows_seen=3000000)`, `bank.specs` comes back from the state file as the
same dicts the builders made, and `bank.groups()` is a frame of every
`(spec, group)` with its processed-row count and last clock value. State
lives until it is dropped, so a long-running bank forgets the groups that
have gone quiet with:

```python
stale = bank.groups().filter(pl.col("last_clock") < now - 30 * 86400)
bank.drop_groups(stale["group"])           # they start cold if they reappear
```

Loading the state back, serving from it, and reading the fit's coefficients
are in [Saving, loading and reading a model](#saving-loading-and-reading-a-model)
below, each in both the bank's form and the query's.

**As a plan.** `lf.online.fit_predict(specs)` is the loop above as a
`LazyFrame`: executing it — `collect()`, `collect_batches()`, `sink_*()` —
streams the plan's rows through a fresh bank in `chunk_rows` chunks, so a
query with the bank in it is O(chunk) however long the stream, and what
comes after the bank is polars' own:

```python
(
    pl.scan_parquet("ticks/*.parquet")
    .filter(pl.col("venue") == "X")                      # before: what the bank learns from
    .online.fit_predict([spec], chunk_rows=100_000)
    .filter(pl.col("ridge").struct.field("n_eff") > 100) # after: what comes out
    .select("ts", "bond_id", "ridge")                    # pushed into the scan: only these
    .sink_parquet("fitted.parquet")                      #   columns and the specs' are read
)
lf.online.predict(bank).collect()          # serve: score against a bank (or a state file)
lf.online.fit_predict(load_state="bank.state", save_state="bank.state")  # resume, and save
```

One caveat with a number on it: a `filter` *before* the bank holds more
memory than the same filter after it — 2.5 GB at 12M rows against 0.78 GB,
with 0.65 GB for no filter at all. It is not the filtered result being
buffered, and it is not the filter: polars' parquet reader applies the
predicate itself, then restores the column order through a per-thread
pipeline whose slots are whole row groups when a sink sits right above the
scan — about 7 row groups × threads of what the filter *keeps*: 2.5 GB on
14 threads with 125k-row groups of 26 doubles when it keeps every row,
1.2 GB when it keeps half, 3.1 GB at 36M rows, 0.7 GB on 2 threads — while
a plain scan feeding the bank has no parallel stage in between and holds
none of it (`docs/PERFORMANCE.md` §11: the stage, the thread sweep, the
knobs, what is already reported upstream). So prefer the filter *after*
unless the model must skip those rows, and if that is the reason, a zero
weight skips them and still streams:

```python
(
    lf.with_columns(pl.when(pl.col("venue") == "X").then(1.0).otherwise(0.0).alias("w"))
    .online.fit_predict([po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"],
                                         clock="t", halflife=600.0, max_dclock=300.0,
                                         weight="w")])
    .sink_parquet("fitted.parquet")                      # 1.3 GB at 12M rows, not 2.5
)
```

Weight 0 learns nothing, bit for bit; the rows still come out, scored, the
clock advances through them (so `n_eff` decays and `min_periods` can blank
output), and no `max_dclock` gap opens where a filter would leave one.
`when/then/otherwise` streams because it is elementwise, through a window
counted in morsels — `pl.Config.set_streaming_chunk_size(25_000)` takes it
to 1.1 GB, against 0.9 with nothing upstream — unless a branch holds an
`.over()`, which drags the whole expression onto a collecting node
(3.2 GB). A `sort` or an `.over()` upstream collects by definition;
`lf.show_graph(engine="streaming", plan_stage="physical")` shows which
nodes do.

The plan is *pure*: every execution starts from the same state — the specs',
or `load_state`, read when the plan is built — so collecting twice gives the
same frame, and `head(n)` learns from the first `n` rows and no more.
`save_state=` writes the state an execution ends in, after the last row it
fed the bank, when it ends: atomically (the file is the old state or the
new, never half of either), the same bytes a bank fed those rows saves and
the same bytes `po.run(save_state=)` writes. Purity is what makes the write
safe: polars runs a plan's source once per execution and *twice, on two
threads*, when one query uses the plan twice (a self-join, `pl.concat`, a
`pl.collect_all` of two sinks — no common-subplan elimination reaches a
Python source), and every run writes the same bytes. Nothing is written
unless the source reaches the last row: a run abandoned before then, or one
the bank ended with an error, leaves the file as it was; a node *after* the
bank failing does not stop the bank (polars drains a Python source before
it raises), so the state is written although the query failed —
`po.run` saves only after its output is committed, for the case where the
two must be tied together, and a dated `save_state` per batch of data keeps
a rerun from learning it twice (`docs/STATE-WORKFLOW.md`: the measurements
behind each of these, on polars 1.34 to 1.44). Filters, selections and
`head` after the bank are pushed into the source and honoured there (a
filter after never changes what the bank learns from — put it before to do
that), and a selection reaches the input scan. The same numbers as the loop
and as `po.run`, bit for bit, in either engine, held by `tests/test_frame.py`.
`df.online.fit_predict(specs)` is the eager twin; `po.fit_predict(frame, ..)`
and `po.predict(frame, bank)` are the same calls as plain functions, for a
type checker, which cannot see a registered namespace. Rides on polars' IO-plugin interface
(`register_io_source`), which polars documents but marks unstable.

### 2. Streaming runner (Python or CLI)

The same bank as a three-stage pipeline — read, fit, write, one chunk in
flight per stage — so memory is O(state + chunk) however long the stream is.
From Python:

```python
po.run(
    input="ticks.parquet", output="fitted.parquet",
    specs=[spec], chunk_rows=100_000, save_state="bank.state",
)                                    # -> {"rows": ..., "chunks": ...}

po.run("bank.toml", input="today.csv")       # keywords override the config
po.run(input="today.parquet", output="scored.parquet", specs=[spec],
       load_state="bank.state", predict=True)  # serve: score against the state, learn nothing
```

`input` is anything py-polars can stream: a path in **parquet, ipc, csv or
ndjson** (told from the extension, or named with `input_format=`; globs and
cloud URLs as `pl.scan_*` takes them), a `LazyFrame` — any query, with
whatever scan options it needs — a `DataFrame`, or an iterable of frames in
stream order (a database cursor, a socket, a generator). `output` is a path in
any of the four formats. Every source and format gives the numbers
`ModelBank` gives on the same rows; the tests hold each of them to it.

```python
po.run(input=pl.scan_parquet("ticks.parquet").filter(pl.col("bond_id") != "b9"),
       output="fitted.arrow", specs=[spec])          # a query: streamed, never collected
po.run(input=(chunk for chunk in lf.collect_batches()), output="fitted.ndjson",
       specs=[spec], progress=lambda rows, chunks: print(rows))  # any iterable of frames
```

`keep_columns=[...]` selects input columns before the bank sees them (and
before the scan reads them); `progress(rows, chunks)` is called after each
chunk, and raising in it stops the run. The output is written through a
temporary and renamed into place, so a run that fails leaves the previous
file where it was. CSV cannot hold the bank's struct columns, so there each
spec's struct is flattened to `<spec>.<field>` columns and a list field
(`coef`) becomes a JSON string — `pl.col("ridge.coef").str.json_decode(pl.List(pl.Float64))`
reads it back bit-exact.

Or from the CLI, for deployments with no Python — one binary, one TOML:

```sh
online --config examples/bank.toml
online --config examples/bank.toml --resume bank.state --save-state bank.state
online --config examples/bank.toml --resume bank.state --predict --input today.parquet
online --config examples/bank.toml --input ticks.csv --output scored.ndjson
online --config examples/bank.toml --input feed.dat --input-format ipc
online --config examples/bank.toml --dry-run     # validate + print the output schema
```

`--predict` scores against the resumed state and learns nothing; one TOML
serves both runs, since the flag drops the config's `save_state`. The CLI
reads with polars' own scanners, which on a stable toolchain lack the SIMD
CSV parser py-polars' wheels have (`docs/PERFORMANCE.md`); for a large CSV,
`po.run` is the faster of the two.

**Paths on Windows.** A backslash starts an escape sequence in a TOML basic
string, so `input = "C:\data\in.parquet"` is a parse error, not a path. Any of
these works:

```toml
input = 'C:\data\in.parquet'      # literal string (single quotes), no escaping
input = "C:\\data\\in.parquet"    # basic string, backslashes doubled
input = "C:/data/in.parquet"      # forward slashes are fine on Windows
```

The same pipeline is the Rust API: `online_polars::run_config` for a
`RunConfig` (what the CLI and a TOML describe), `run_config_on` with an
`Input::Lazy(LazyFrame)` or `Input::Batches` of frames the caller already
has, and `run` with an `Output::Batches` callback instead of a file.

## Saving, loading and reading a model

A fitted model is the bank's *state* — one accumulator per `(spec, group)` —
and it travels as one file, written whole or not at all. It is saved and
loaded from a bank, or from a query, with the same words:

```python
# From a bank: save after the loop; load it to keep learning, or to serve
bank.fit_predict(df)
bank.save("bank.state")                               # atomic: temp file, then rename
bank = po.ModelBank.load("bank.state", specs=[spec])  # specs= checks the file is this model's
bank.fit_predict(today)                               # learn on: the state moves
scored = bank.predict(today)                          # serve: score, learn nothing

# From a query: the same two keywords; the file is written when the run reaches the last row
lf.online.fit_predict([spec], save_state="bank.state").sink_parquet("fitted.parquet")
lf.online.fit_predict(load_state="bank.state", save_state="bank.state").sink_parquet("more.parquet")
served = lf.online.predict("bank.state").collect()    # serve from the file
```

`po.run(..., save_state=)`, `po.run(..., load_state=, predict=True)` and the
CLI's `--save-state` / `--resume` / `--predict` (section 2) read and write
the same file, and the bytes are the same whichever wrote them: a state a
query saved loads into a bank and the other way round. Loading names the
problem it hits — `FileNotFoundError` for no file yet (start fresh),
`ValueError` for a file that is not a bank, a newer build's, or another
model's. [docs/STATE-WORKFLOW.md](docs/STATE-WORKFLOW.md) walks the whole
workflow — fit, save, serve, learn on — with what each step guarantees.

**The coefficients.** The betas behind a fit can be read two ways, and the
two agree row for row:

```python
ols = po.spec.ewridge("ols", targets=["y"], features=["x0", "x1"], clock="t",
                      halflife=600.0, max_dclock=300.0, group="bond_id", coef_every=1)

# 1. From a bank -- live, or loaded from a state file with no data at hand:
#    one row per coefficient, with the term it belongs to
bank = po.ModelBank([ols])
bank.fit_predict(df)
betas = bank.coef("ols")             # group, instance, n_eff, position, target, ..., term, coef
wide = betas.pivot("term", index=["group", "instance"], values="coef")

# 2. From the output, in polars: `unnest` takes the struct apart into columns,
#    the `coef` list as one named column per coefficient
path = (
    lf.online.fit_predict([ols])
    .online.unnest([ols])            # pred_y, resid_y, n_eff, coef_y_intercept, coef_y_x0, coef_y_x1
    .select("t", "bond_id", "^coef_.*$")
    .collect()                       # the fit as it moved, one row per row
)
```

`bank.coef()` is the fit as of the last row each group learned from — what
`coef` said on that row, and what the group's next `pred` is computed from
— with `n_eff` for how much weight is behind it. The output's `coef` is the
same fit snapshotted *after* each row's update (the row's `pred` is from
the fit *before* it), every `coef_every` rows and on the last row of every
chunk; the default, `coef_every=0`, is the chunk end only, so the per-row
path above asks for `coef_every=1` and pays a list of `k` floats per row.
Under a grid — several `ridge` values, `feature_sets`, a `lasso_path`,
several targets — the list holds one block per (target × grid point).
`unnest` names each block's columns the way the `pred` fields are named
(`coef_y_x0__r0.5@h500` beside `pred_y__r0.5@h500`), and
`po.spec.coef_fields(spec)` is the table behind those names — one row per
coefficient with its `field`, `position`, `name`, `target`, `halflife`,
`ridge`, `feature_set`, `lambda` and `term` — so a grid is a filter on it,
not a string to write. `bank.coef()` carries the same columns to tell the
blocks apart: add the ones the spec varies to the pivot's `index`
(`["group", "instance", "target", "ridge"]` for the `ridge` grid of
section 1). `unnest` reads a saved output the same way
(`pl.scan_parquet("fitted.parquet").online.unnest([ols])`), and takes the
specs, a bank, or the path of a saved state. `bank.gram("ols")` gives the
EW accumulators behind the fit (`means`, centered `comoments`,
`cross_moments`, `n_eff`), for anything other than our solve.

## Diagnostics and selection

Three opt-in outputs, all derived from state the models already keep:

| flag | adds | meaning |
|---|---|---|
| `emit_sigma` | `sigma_<slot>` | EW standard deviation of that slot's out-of-sample residuals |
| `emit_resid_z` | `resid_z_<slot>` | `resid / sigma` — how surprising the row was, in units of the model's own recent error |
| `emit_selected` | `selected_<t>`, `pred_<t>__selected` | online model selection across ridge values, feature sets and halflives, by lowest EW out-of-sample error |
| `emit_averaged` | `pred_<t>__averaged` | `softmax(−eta · EW error)` blend over the same slots — hedges where `emit_selected` commits |
| `emit_drift` | `drift_<slot>` | Page-Hinkley break detection on the residual stream; `drift_action="reset"` also restarts the stream |
| `emit_metrics` | `ic_<slot>`, `r2_<slot>`, `hit_rate_<slot>` | the same numbers `po.eval` computes, but kept in O(state) beside the model |
| `resid_quantiles` | `absresid_q<p>_<slot>` | P² quantiles of \|resid\| — a distribution-free interval where `sigma` gives a Gaussian one |
| `emit_autocorr` | `autocorr_<slot>` | EW residual autocorrelation; non-zero means the model is mis-specified |

All read from the state *before* each row, so they are out-of-sample like the
predictions they describe. Drift detection complements the halflife rather than
replacing it: decay forgets smoothly and always, a detector notices a break and
says so — a sign flip mid-stream is caught within a couple of rows.

## Evaluation

```python
po.eval.metrics(out, "ridge", by=["bond_id"])                       # R², IC, hit rate, MSE
po.eval.rolling_metrics(out, "ridge", clock="t", window=3600.0)     # per clock window
po.eval.compare_specs(out, ["ridge", "kalman"])                     # one table, many specs
```

## What this is not

A model layer, not a stream-processing framework. It expects a frame that is
already aligned — and, when a spec names a `clock`, each group's rows in clock
order; without one the row order is the clock — and it keeps O(state) per
stream. It deliberately does **not** provide:

- **connectors or ingestion** — feed it whatever Polars can read;
- **event-time windowing, tumbling/sliding windows, asof or interval joins** —
  build features with Polars expressions upstream, or with a streaming framework
  such as [Pathway](https://pathway.com), whose Rust engine already does this;
- **watermarks or late-arrival policy** — `clock`, `max_dclock`,
  `on_clock_reset` and `session` describe *within-stream* time, not pipeline
  lateness. Under a `clock`, a row that arrives out of order is a data error
  here, and `on_clock_reset="error"` will say so;
- **distributed execution** — one process, `rayon` across (spec × group).

Those boundaries make the two compose rather than compete:
[`examples/pathway_integration.py`](examples/pathway_integration.py) runs a
`ModelBank` as a stateful operator inside a Pathway pipeline — Pathway does
ingestion, event-time alignment and windowing; we do the model. Chunk
invariance means the engine's batching cannot change the numbers, and
`save_bytes`/`load_bytes` let a pipeline checkpoint carry the model state.
Pathway is BSL-licensed and is *not* a dependency of this project; the example
imports it lazily and runs its plain-batch path without it.

## Common parameters

Every model takes the same stream parameters:

| parameter | meaning |
|---|---|
| `targets`, `features` | column names; ≥1 target, shared `X'X` across targets. Columns must be numeric — any width, `Decimal` and `Boolean` included, cast to `f64` on the way in — and a String column is refused rather than cast to null. Columns the spec does not name are carried through untouched, whatever their dtype |
| `add_intercept` | default `True` |
| `clock` | monotone **numeric** column (seconds, cumulative volume, …). `None` ⇒ row count. A temporal column is rejected — cast it first, e.g. `pl.col("ts").dt.epoch("s")` — because its internal representation would silently set the units of `halflife`, `max_dclock` and `session_gap` |
| `halflife` / `lam` | decay in clock units; mutually exclusive. A list of halflives means one accumulator per value. `halflife=inf` (or `lam=1.0`) is decay off: least squares over every row seen, in any order — and not the same as a huge finite halflife, whose default solve cadence of `halflife/50` never comes due ([Any row order](#any-row-order)) |
| `max_dclock` | ceiling on the clock delta (required with `clock`); `0` disables decay, `inf` removes the ceiling |
| `on_clock_reset` | what a backwards clock means: `"max"` (default), `"zero"`, `"reset_state"`, or `"error"` to refuse the chunk — the bank is left as it was, so the corrected chunk can be fed |
| `session`, `session_gap` | on a session change, apply this delta (`"reset"` resets the state, `inf` never applies it) |
| `session_shrink`, `long_halflife` | `ewridge` only: at a session change, mix partway back toward a slow-moving twin — changes what the model believes, where `session_gap` only changes how confident it is |
| `weight` | row weight column |
| `min_periods` | in `n_eff` units; outputs are null until reached. A list gives one threshold per target — warmup gates output, not learning |
| `coef_every` | 0 = never; coefficients are also emitted on each chunk's last row |
| `group` | one state per key |

Per-row decay is `λ = 0.5 ** (Δclock / halflife)`, and `n_eff` is the
exponentially weighted observation count under the same decay. It is the weight
of the state that produced *this row's* prediction — measured before the row's
own update and before its own decay — so it is `0` on a stream's first row and
lags the row count by one. Every model reports it the same way, which is what
makes `min_periods` mean the same thing across a bank. It saturates at
`1 / (1 − λ)` rather than growing with the stream.

**Null policy.** A null in any feature (or the weight) skips the row entirely:
outputs are null, no update happens, but the clock still advances. A null in
target *j* still emits `pred_j`, leaves `resid_j` null, and skips only that
target's update. NaN, ±inf and any magnitude above `1e100` count as null —
sentinels like `f64::MAX` never reach a model, and every model is tested to
keep a finite state and go on learning through anything below that bound
(`docs/IMPROVEMENTS.md` C2).

**Scoring without learning.** `bank.predict(df)` scores every row against
the bank exactly as it stands and touches nothing: no clock advance, no decay,
`n_eff` frozen, so a fit is served for as long as it stays good. Row `i`
carries what `fit_predict` would have reported had it been the next row of
the stream — the same `pred`, `n_eff`, `sigma`, `resid_z`, selection and
metrics, field for field — and every row is scored from that same state, with
the clock distance measured from the last row the bank learned from (`holt`
extrapolates over it, capped by `max_dclock`). The target column may be
absent, in which case `resid` is null; `weight` is not read; a group the bank
has never seen scores null; `coef` lands on each group's last accepted row.
The stream's session and clock policies still hold: a row that would reset
the stream (`session_gap="reset"`, `on_clock_reset="reset_state"`) scores as
a fresh model — null — and a backwards clock under `"error"` raises the same
error. From the runner it is `po.run(..., load_state=..., predict=True)`, from
the CLI `--predict`. It is also the faster path — nothing is updated, so
`ewridge` scores at 1.8× (`k=5`) to 2.9× (`k=20`) its learning throughput.

Inside a `fit_predict` stream there are two other ways to withhold learning,
and they differ. Give the rows weight `0` and the coefficients are frozen bit
for bit, because the accumulators are exponentially weighted *means* and
decaying them and adding nothing leaves them exactly where they were. A
**null target is not the same thing**: the feature-side moments still update
while the target's cross-moment does not, so the two halves of the fit end up
estimated over different windows and the coefficients wander with feature
noise (measured: 2.00 → 2.39 over 100 scored rows at a halflife of 20). Use a
null target for a label that has not arrived yet, weight `0` to hold a row's
place in the stream, and `predict` to serve.

One consequence of weight `0` to plan for: a zero-weight row still advances
the clock, so `n_eff` keeps decaying while you score. Score for several
halflives and it can fall below `min_periods`, at which point the outputs go
null even though the fit behind them is perfectly good — `min_periods` is
baked into the saved state, so choose it with the scoring tail in mind, or
serve with `predict`, which has no tail.

**Mistakes are named.** A builder checks each keyword against its own type
hints, so `halflife="10"` says `spec "m": halflife must be a number or a list
of numbers, got str '10'`; a missing column says which spec wanted it, in what
role, and what the frame has; a spec named like an input column is refused
rather than silently replacing it (`docs/IMPROVEMENTS.md` U2).

## Models

All accumulators are exponentially weighted **means**, not sums, which keeps them
bounded over arbitrarily long runs, and second moments are kept **centered** (a
weighted Welford update) so the variance is accurate even when features sit on a
large offset. `z` denotes `[1, x]` when an intercept is configured.

### `ewridge` — EW ridge on sufficient statistics (the workhorse)

```
W'   = λW + w                       S' = (λW·S + w·z zᵀ) / W'
W_j' = λW_j + w                     r_j' = (λW_j·r_j + w·z·y_j) / W_j'
solve:  (S + ridge·D) β_j = r_j     D = I minus the intercept slot
```

`coef0` shrinks toward a stated belief rather than toward zero. Note that `S`
is a weighted *mean*, so a plain `ridge` is a fixed per-observation penalty and
its pull is **permanent**; the fading warm start ("start at yesterday's fit")
is `ridge_decay`, where the prior sits on the decaying sum scale.

O(k²) per row; Cholesky solves on a schedule (`solve_every` in clock units,
default `halflife/50`; every row for `halflife=inf` and for `lam` — so set it
with a large finite halflife, or the default never comes due;
`max_rows_between_solves` caps it in rows). Ridge values and named `feature_sets` are expanded at
solve time from the same accumulator, so grids are nearly free. With
`standardize`, the solve is done in correlation form and unscaled afterwards;
near-zero-variance features are dropped rather than blowing up.

### `rls` — recursive least squares

```
A ← λA + w zzᵀ       b_j ← λb_j + w y_j z        β_j = A⁻¹ b_j
A₀ = ridge·I         b₀ = ridge·coef0
```

Coefficients move every row with zero solve staleness. The state is the
Cholesky factor `R` of `A` (`A = RᵀR`) and `u_j = R⁻ᵀb_j`; each row is folded
in by `k` Givens rotations and `β` read off by one back-substitution — the
square-root (QR) form, O(k²) per row like the textbook `P ← P − g zᵀP`
recursion but without its two failure modes: `P`'s rounding asymmetry growing
by `1/λ` every row, and one extreme row cancelling `P` to zero in some
direction and freezing that coefficient for good (`docs/IMPROVEMENTS.md` C5).
`ridge` sets `A₀ = ridge·I` (`P₀ = I/ridge`) and penalizes the intercept. This
is algebraically identical to `ewridge(ridge_decay=True)` solved every row — a
test asserts they agree to <1e-9. *Null-policy deviation:* a row with any null
target is predict-only for all targets, since `R` is shared.

### `lasso` — lasso path with free λ selection

Coordinate descent on the standardized statistics, warm-started along the path
and across solves:

```
ρ_i = c_i − Σ_{j≠i} C_ij β_j
β_i = soft(ρ_i, λ·l1_ratio) / (C_ii + λ(1 − l1_ratio))
```

`l1_ratio < 1` gives elastic net. Because predictions for every path point are
computed anyway, `lam_selected_<target>` — the argmin over the path of an EW
mean squared **out-of-sample** error — costs nothing extra. It is reported as
it stood *before* the row, like every other output: the λ this row was scored
with, not the one its own error then elected.

### `kalman` — random-walk-β dynamic linear model

```
P_j ← P_j + Q·Δclock                s = zᵀP_j z + R_j/w
k   = P_j z / s                     β_j ← β_j + k(y_j − zᵀβ_j)
P_j ← P_j − k zᵀP_j
```

Process noise comes from a **per-factor coefficient halflife** on standardized
features: `q_i = σ²(ln2 / h_i)²`, which matches the steady-state gain of EW-RLS.
`coef_halflife` may be a scalar or one value per slot; `inf` pins a coefficient.
Observation noise defaults to the EW residual variance.

Standardization is internal and on by default (the halflife-derived `q` is only
comparable across features on a common scale). With `standardize=False`, `q=0`
and a fixed `obs_var`, this is exactly a Bayesian linear regression — it
reproduces river's `BayesianLinearRegression` to 3.6e-15.

### `huber` / `quantile` — robust regression

IRLS reweighting on the ridge update, using each row's **prior** residual so the
reweighting stays out-of-sample. Huber: `w = min(1, δσ/|r|)`. Quantile: the
check-loss weights at level τ. Weights are per target, so `S` is per target here.

### `sgd` — stochastic gradient descent, pluggable losses

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
takes **count targets** (`loss="poisson"`). Learning rate is `constant`,
`inv_scaling` (`lr/(1+n_eff)^power`) or `adagrad`; AdaGrad's accumulator decays
on the clock, so an adapted rate re-opens after a long gap.

`clip_gradient` defaults to `1e3` rather than off: with a log link one large
count makes the next gradient exponentially bigger and a constant rate diverges.
It does not bind for identity-link losses.

### `pa` — passive-aggressive regression

```
loss = max(0, |y − p| − eps)      s = ‖z‖²
pa    τ = loss / s          pa1  τ = min(c, loss/s)      pa2  τ = loss / (s + 1/(2c))
β    += τ · sign(y − p) · z
```

Each row poses a constraint and the update is the smallest change that
satisfies it — no learning rate to tune. Note plain `pa` will move the fit as
far as one bad row demands, so `pa1` is the default. PA keeps no accumulators,
so unlike the other models its coefficients have no half-life; the clock only
drives `n_eff`. A row weight below 1 scales `τ`; above 1 it counts as 1 (the
update is a projection, and repeating it changes nothing).

### `ew_cov` — exponentially weighted moments (no regression)

```
W'   = λW + w        m'ᵢ = (λW·mᵢ + w·xᵢ) / W'      S'ᵢⱼ = (λW·Sᵢⱼ + w·xᵢxⱼ) / W'
varᵢ = Sᵢᵢ − mᵢ²     covᵢⱼ = Sᵢⱼ − mᵢmⱼ             corrᵢⱼ = covᵢⱼ / √(varᵢ·varⱼ)
```

Running mean / variance / std / covariance / correlation of the columns you
name, on the same clock as every model here. With `precision_prior` set it also
gives `partial_corr` — the correlation between two columns *controlling for all
the others* — read off `(C + s·prior·I)⁻¹`, solved from the co-moments each
row (O(k³), paid only when asked for; the prior fades as data accumulates like
RLS's `P₀`). One O(k²) update per row otherwise, replacing the O(k²) *passes*
a pure-Polars pairwise EW correlation needs. Values are read from the state
before each row, so an `ew_cov` output can be a feature for that same row
without leaking it.

### `ftrl` — online logistic regression

FTRL-proximal (McMahan et al. 2013) for binary targets, with the accumulators
decayed on the same clock as everything else:

```
β_i = 0 if |z_i| ≤ l1 else −(z_i − sgn(z_i)l1) / ((β + √n_i)/α + l2)
p   = sigmoid(zᵀβ)     g_i = (p − y)·z_i·w
z_i += g_i − ((√(n_i + g_i²) − √n_i)/α)·β_i      n_i += g_i²
```

With `loss="logistic"` (default) `pred` is a probability and `resid = y − p`;
with `loss="squared"` it is the linear prediction, giving sparse linear
regression with no solves — and L1 support, which `ewridge` does not have.

### `holt` — Holt's linear trend (the baseline)

The one model that takes no features at all: it extrapolates the target's own
level and trend.

```
pred     = l + b·Δt
l' = α·y + (1−α)·pred        b' = β·(l' − l)/Δt + (1−β)·b
```

`α` and `β` come from `level_halflife` and `trend_halflife` in clock units. The
trend is per *clock unit*, so an irregular clock extrapolates the right
distance. `coef` is `[level, trend]` per target;
`trend_halflife=float("inf")` pins the trend at zero, leaving a plain EW level.

```python
po.spec.holt("baseline", targets=["y"], clock="t", max_dclock=600.0,
             level_halflife=200.0, trend_halflife=2000.0)
```

There is no seasonal term, because a seasonal index is a `group_by` on the
phase, which the bank already does. Use it to answer "how much is the
regression actually adding?" — run it in the same bank as the real model and
compare `sigma`, or let `emit_selected` choose between them.

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

`rls` is the one that went *down* — it was 3.1M before the square-root
rewrite that keeps it from dying of cancellation on a single extreme row
(`docs/IMPROVEMENTS.md` C5). Measured on the model arithmetic alone, the
QR form costs 1.3–2.1× the covariance form; that is the price of the fix,
and it is worth paying.

Targets share one `S` accumulator, so 10 targets cost far less than 10× one.
Each halflife in a grid is its own accumulator, but they run in parallel, so a
5-halflife grid costs about 2× one rather than 5×.

**Grouped data goes wider.** One state per group is one rayon task, so
throughput rises with the group count rather than falling: **6.0M rows/s** at
k=20 over 64 groups, scaling 5.2× from one thread to eight and 6.6× to
fourteen. A bank of several specs is one flat task pool too — eight
single-group specs over 300k rows take 118 ms, against 685 ms if they ran one
at a time.

**Memory.** The bank and the runner hold the state, the chunks in flight
(three, so `chunk_rows` is the knob) and whatever polars' reader prefetches:
on a 14-thread machine the parquet reader front-loads ~0.7 GB of decoded
row groups whatever the file's length, and `POLARS_ROW_GROUP_PREFETCH_SIZE=1`
takes the CLI to 0.15 GB at the same speed. Measured flat from 3M to 12M
rows in docs/PERFORMANCE.md §11, `lf.online.fit_predict` included (0.78 GB
live at 12M rows, 0.37 GB with the prefetch at 1). *Which calls stream*,
at the top, is the whole comparison in one table.

Where the time goes, and what to reach for, is in
[`docs/PERFORMANCE.md`](docs/PERFORMANCE.md).

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
([rustup](https://rustup.rs)). Both install outside the default `PATH` on some
machines — `source scripts/env.sh` fixes that for a shell (`. .\scripts\env.ps1`
in PowerShell), and `.vscode/settings.json` does it for VS Code's integrated
terminal on all three platforms.

`cargo` runs via `uv run` because `online-py` builds against
pyo3's `abi3-py312`, which needs a ≥3.12 interpreter at build time; `uv run`
exports `VIRTUAL_ENV`, which pyo3's build script picks up. Plain `cargo test`
also works if `PYO3_PYTHON` points at a 3.12+ interpreter.

Tests generate or download their own data — there are no data files in the repo.
Downloads are cached under `.cache/` and skipped when offline.

- API reference: `docs/reference/`, Sphinx over the docstrings, built by the
  gate and CI with warnings as errors and published to GitHub Pages from `main`
- Design and task list: [`docs/PLAN.md`](docs/PLAN.md)
- Measured defaults: [`docs/VALIDATION.md`](docs/VALIDATION.md)
- Adding a model: [`docs/EXTENDING.md`](docs/EXTENDING.md)

## Versioning and the Polars pin

### What is pinned, and why

The Rust `polars` is pinned **exactly** in `Cargo.toml`; the py-polars
requirement in `pyproject.toml` is a range that brackets the release the
wheel is built and tested against, and `tests/test_scaffold.py` asserts both
that and that the dev environment sits on that release.

| py-polars | rust polars | pyo3-polars | pyo3 | Python |
|---|---|---|---|---|
| **>= 1.34.0, < 2** (built and tested against 1.44.1) | 0.55.2 | 0.28 | 0.29 | ≥ 3.12 (`abi3-py312`) |

The *Rust* pin is exact and the wheel links it statically; the *runtime*
requirement is a range, because the two copies of Polars never meet. The floor
is `LazyFrame.collect_batches`, which `po.run` and `lf.online.fit_predict`
read with and py-polars added in 1.34.0; the whole suite passes on 1.34.0,
1.38.1 and 1.44.1 with identical numbers. `ModelBank` and the expression
form alone work from 1.28.1 (tested across 17 releases), and below that the
failure is a clean `AttributeError` on `PySeries._export` rather than
anything subtle. The matrix is in `docs/RELEASE-READINESS.md`.

This is stricter than the mechanism strictly requires, and it is worth being
precise about why, because "pinned" usually implies "fragile" and here it does
not.

`ModelBank` and the expression plugin move data across the boundary
through the **Arrow C Data Interface** — `SeriesExport` is a `#[repr(C)]`
struct of `ArrowSchema` and `ArrowArray` pointers, the same cross-language
ABI pyarrow and DuckDB use. `PyDataFrame` is not special-cased: it extracts
column-by-column as `PySeries`, each through `import_series`. This package
does *not* use `PyExpr` or `PyLazyFrame`, which are the genuinely
version-sensitive types that cross as serialized query plans.

The interface is versioned by name (`polars_ffi::version_0`), and the only
thing `ModelBank` asks of the Python side is `PySeries._export` / `_import`:
a Polars without them fails with a clean `AttributeError` before any data
moves, and one with them ran the whole matrix in `docs/RELEASE-READINESS.md`
— one wheel, 17 releases — with identical numbers. The plugin goes one step
further, because polars' plugin loader **negotiates** the ABI: it calls the
plugin's `_polars_plugin_get_version()` before its first call and refuses a
major it does not know (`ComputeError: this polars engine doesn't support
plugin version: 0-1`), with a dedicated check for layout drift besides. **So
a mismatched Polars is a clear error, not a crash.** The pin exists so you
never see those messages, not because something worse waits behind them.

### What the pin costs you

At install time, nothing: the exact pin is the Rust side's, inside the wheel,
and the runtime requirement is the range `polars>=1.34.0,<2`. What it costs
is that a py-polars newer than the last one the canary passed is untested
until the canary's next run — and, since the two streaming surfaces ride on
interfaces polars does not promise to keep, a break there is expected
maintenance rather than a surprise (`docs/RELEASE-READINESS.md`).

At the time of writing, **1.44.1 is the latest Polars release**, so nothing is
untested today.

### How the pin will move

A scheduled CI job (`.github/workflows/polars-canary.yml`) drops the range
from `pyproject.toml` weekly, installs the newest py-polars — a 2.0 included,
the week it appears — builds the wheel as CI does and runs the whole suite on
it. Only polars moves in that run, so a red canary means Polars broke us and
nothing else. It opens nothing on its own; the failure is the notification,
and the response is decided in advance: **cap** the range at the last release
that passed, in a patch release, so no resolver hands anyone the broken pair;
then **fix**, and widen again. Where to look first is set too: `ModelBank`
(the extension types), then the IO-plugin tests in `tests/test_frame.py`
(a changed pushdown contract shows as a wrong row count, not a crash), and
the plugin last, since its ABI is negotiated and refuses to load rather than
misbehave.

The Rust copy of polars does not move in the canary and is not what it is
for: it never meets the user's, and pyo3-polars, polars-arrow, polars-parquet
and polars-utils pin to the same release, so it moves by hand, all together,
through CI.

### Output field names are part of the API

You index the result struct by strings — `out["m"].struct.field("pred_y")` —
so the names are a contract, not a detail. The grammar:

```
pred_{target}{combo}{instance}     combo    = ""            single ridge, no feature sets
resid_{target}{combo}{instance}             | __r{ridge}     ridge grid
sigma_{target}{combo}{instance}             | __{set}        feature sets, single ridge
absresid_q{level}_{target}...               | __{set}_r{ridge}
n_eff{instance}                    instance = ""            single halflife
coef{instance}                              | @h{halflife}   halflife grid
```

Numbers render as plain decimals in `[1e-6, 1e7)` (`0.000001`, `250.5`) and as
compact scientific outside it (`1e-300`, `2.5e8`). The whole grammar — every
name, every default, every signature — is pinned by
`tests/test_api_surface.py` against a checked-in snapshot, so a change is a
reviewable diff and a version bump, never a silent rename of your columns.

**You never have to construct these strings.** `output_index` gives every
field with the machine values its name encodes, so selection is a filter, not
string formatting:

```python
grid = po.spec.ewridge("m", targets=["y"], features=["x0", "x1"], clock="t",
                       max_dclock=300.0, halflife=[100.0, 500.0], ridge=[1e-6, 0.5])
idx = po.spec.output_index(grid)
name = idx.filter(
    (pl.col("kind") == "pred") & (pl.col("target") == "y")
    & (pl.col("ridge") == 0.5) & (pl.col("halflife") == 500.0)
)["field"].item()          # -> "pred_y__r0.5@h500", resolved for you
po.ModelBank([grid]).fit_predict(df)["m"].struct.field(name)
```

`coef_fields(spec)` does the same for the flat `coef` lists — one row per
coefficient, mapping (target, combo, instance, term) to the list it sits
in, its position there, and the column `online.unnest` gives it:

```python
row = po.spec.coef_fields(grid).filter(
    (pl.col("term") == "x1") & (pl.col("ridge") == 0.5) & (pl.col("halflife") == 500.0)
).row(0, named=True)       # field "coef@h500", position 5, name "coef_y_x1__r0.5@h500"
out["m"].struct.field(row["field"]).list.get(row["position"])
```

(`coef_index(spec)` is the same table for one instance, by position.) All
of them come from the same Rust code that renders the names, so they
cannot drift from the strings. The index also carries each field's `dtype` (`f64`,
`bool` for `drift_*`, `str` for `selected_*`, `list[f64]` for `coef`), which
is the schema the bank declares to polars before the first row is read.

One sharp edge to know: a *target named* `y__r0.5` produces the same field
string as a ridge grid on `y` would. Nothing breaks — the struct is still
well-formed — but if you parse field names downstream, avoid `__` and `@` in
target names and feature-set labels.

### This package's own versioning

Semantic versioning. While pre-1.0 the **minor** version carries breaking
changes, so pin `~=0.1.0` if you need stability. Widening the Polars
constraint would be a minor release; narrowing it is breaking and would be a
major one. See [`CHANGELOG.md`](CHANGELOG.md).

Wheels are published for macOS (arm64 and x86_64), Windows x64, and Linux x64
and aarch64 in both glibc and musl flavours. `abi3-py312` means one wheel per
platform covers 3.12, 3.13, 3.14 and later. An sdist is published too; building
from it needs a Rust toolchain.

## The expression form

The same model can be written as an expression, and it is the shortest way
to write it for a frame that is already in memory:

```python
out = df.with_columns(
    pl.col("y").online.ewridge(
        features=["x0", "x1", pl.col("y").shift(1).alias("y_lag")],
        clock="t", halflife=600.0, max_dclock=300.0,
    ).over("group").alias("fit")
)
out.select(pl.col("fit").struct.field("pred_y"), pl.col("fit").struct.field("n_eff"))
```

Features may be expressions, evaluated per group under `.over`, so the lag
above never crosses a group boundary; `po.online(pl.col("y"))` is the same
namespace as a plain function, so that a type checker can see it. The numbers are the
bank's — the expression *is* the bank, run over the column polars hands it.

**Every call warns**, with `polars_online.InMemoryExpressionWarning`, and
this is why. Polars gives a stateful user expression its whole column at
once — in the in-memory engine by definition, and in the streaming engine
because a plugin is a `columnar-function` node: collect the input, call
once, re-emit. Nothing a plugin does can change that; it is polars'
expression contract, which has no way to say "call me per morsel, in order,
and let me keep state". So wrapping the expression in a lazy query does not
make it stream, and this pair, `ewridge` with 20 features on the same 12M-row
file, is **7.3 GB against 1.35 GB**, not a matter of taste:

```python
lf.with_columns(                                   # O(data): the stream is collected
    pl.col("ret").online.ewridge(                  #   first, then the plugin is called
        features=["signal_a"], clock="ts", halflife=600.0, max_dclock=300.0,
    ).over("bond_id")
).sink_parquet("fitted.parquet")

lf.online.fit_predict([spec]).sink_parquet("fitted.parquet")   # O(chunk): the bank is a
                                                               #   source the engine pulls
```

Both produce the same numbers. The first grows with the file; the second
does not (*Which calls stream*, at the top). The warning is a
`UserWarning`, shown by default wherever the call is made — a
`DeprecationWarning` would be hidden outside `__main__`, which is exactly
the pipeline module where the difference matters. Using the expression on a
frame in memory on purpose is fine; say so once:

```python
import warnings

warnings.filterwarnings("ignore", category=po.InMemoryExpressionWarning)
```

The warning would go, and the expression would stream too, if polars ever
ran a user expression per morsel, in order, with state.
[`docs/PLAN.md`](docs/PLAN.md) §6 has the design and that condition.

## License

Apache-2.0. See [CONTRIBUTING.md](CONTRIBUTING.md) to make changes,
[SECURITY.md](SECURITY.md) to report a vulnerability.
