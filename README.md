# polars-online

Streaming / online regression models for [Polars](https://pola.rs). A Rust core
exposed three ways with identical numerics:

1. **an expression plugin** — `pl.col("y").online.ewridge(...)`, with `.over(group)`,
   for a frame in memory (polars hands it the whole column, so it is O(data));
2. **a chunk-fed `ModelBank`** — holds O(state), not O(data) — and, as a plan,
   **`lf.online.fit_predict(specs)`**: a `LazyFrame` that streams through the
   bank when it runs, so a query stays O(chunk);
3. **a streaming runner** — `po.run(...)` or the standalone `online` CLI:
   parquet, ipc, csv or ndjson in and out, config from TOML.

Built for ordered event data (one stream per group) that does not fit in memory:
irregular clocks, session breaks, gaps, nulls, and per-group state.

```python
import polars as pl
import polars_online as po

df.with_columns(                              # a frame in memory: the expression
    pl.col("ret").online.ewridge(
        features=["signal_a", "signal_b"],
        clock="ts", halflife=600.0, max_dclock=300.0,
    ).over("bond_id")
)

spec = po.spec.ewridge("ridge", targets=["ret"], features=["signal_a", "signal_b"],
                       clock="ts", halflife=600.0, max_dclock=300.0, group="bond_id")
(pl.scan_parquet("ticks/*.parquet")           # a stream: the same bank, as a plan
   .online.fit_predict([spec])
   .filter(pl.col("ridge").struct.field("n_eff") > 100)
   .sink_parquet("fitted.parquet"))           # O(chunk) end to end
```

## Which spelling streams

One model, one set of numbers — but **the syntax decides the memory**, because
it decides where the model sits in polars' plan. Peak footprint on the same
file, `ewridge` with 20 features, parquet in and out:

| what you write | 3M rows | 12M rows | |
|---|---:|---:|---|
| `df.with_columns(pl.col("y").online.ewridge(...))` | 2.0 GB | **7.3 GB** | **O(data)** — a frame in memory |
| `lf.online.fit_predict([spec])` | 0.90 GB | 1.35 GB | **O(chunk)** — a query over a stream |
| `for chunk in lf.collect_batches(): bank.fit_predict(chunk)` | 0.80 GB | 1.24 GB | O(state + chunk) — your own loop |
| `po.run(input=..., output=...)`, `online --config` | 0.95 / 0.73 GB | 1.41 / 0.75 GB | O(state + chunk) — file in, file out |

The bottom three rows are flat, and what growth they show is the allocator
holding freed pages rather than live data: told to release them, all three sit
at 0.74–0.86 GB at 12M rows, nearly all of it polars' parquet read-ahead
(0.31–0.46 GB with `POLARS_ROW_GROUP_PREFETCH_SIZE=1`). The first row is live
data, and it grows with the file.

**Wrapping the expression in a lazy query does not make it stream.** Polars
hands a user expression its whole column — in the in-memory engine by
definition, and in the streaming engine because a plugin is a
`columnar-function` node: collect the input, call once, re-emit. So this pair
is 7.3 GB against 1.35 GB, not a matter of taste:

```python
lf.with_columns(                                   # O(data): the stream is collected
    pl.col("ret").online.ewridge(                  #   first, then the plugin is called
        features=["signal_a"], clock="ts", halflife=600.0, max_dclock=300.0,
    ).over("bond_id")
).sink_parquet("fitted.parquet")

lf.online.fit_predict([spec]).sink_parquet("fitted.parquet")   # O(chunk): the bank is a
                                                               #   source the engine pulls
```

Use the expression when the frame is already in memory — it is the shortest
thing to write and the numbers are identical. Use the plan for a stream.
Everything after the bank — filters, joins, group-bys, sinks — is polars' own
and streams as polars streams it. All of it measured in
[docs/PERFORMANCE.md](docs/PERFORMANCE.md) §11.

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

## The three usage modes

### 1. Expression plugin

Runs one spec over the column it receives, so `.over(group)` gives per-group
streams. Output is a struct column.

```python
out = df.with_columns(
    pl.col("y").online.ewridge(
        features=["x0", "x1", pl.col("y").shift(1).alias("y_lag")],
        clock="t", halflife=600.0, max_dclock=300.0,
        session="session", session_gap=1800.0,
    ).over("group").alias("fit")
)
out.select(pl.col("fit").struct.field("pred_y"), pl.col("fit").struct.field("n_eff"))
```

`pl.col("y").online` is attached when `polars_online` is imported, which no
type checker can see; `po.online(pl.col("y"))` is the same namespace, visibly
typed, and every keyword above is checked and completed there and in the
`po.spec` builders.

Features are column names or named expressions; under `.over` an expression is
evaluated per group, so the lag above never crosses a group boundary. Groups
run in parallel (the inputs travel as one packed struct, which is the polars
path that spreads groups over threads — see docs/PERFORMANCE.md P5). Grids are
allowed but produce wide structs; the bank is the better surface for grids.

**This is the in-memory surface** (*Which spelling streams*, above). Polars
calls the plugin once with the whole column, so its memory is O(data), and
putting it in a `LazyFrame` with `collect(engine="streaming")` or
`sink_parquet` does not change that: the streaming engine has no streaming
node for a user expression — a plugin is a `columnar-function` node, which
collects its input, calls once, and re-emits (the engine's own `rolling`,
`ewm_*` and `cum_*` get dedicated windowed nodes; nothing a user writes
does). For a stream, write the same thing as a plan —
`lf.online.fit_predict([spec])`, next section — which is the bank as a polars
source and stays O(chunk) (docs/PERFORMANCE.md §11, which also explains the
number a Python process reports).

### 2. `ModelBank` (chunk-fed)

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
    ...                                    # (or po.run(input=lf, output=...) -- §3: the same
                                           #  loop, pipelined, written straight to a file)

bank.save("bank.state")                    # atomic: temp file, then rename
bank = po.ModelBank.load("bank.state", specs=[spec])
scored = bank.predict(today)               # serve: score, learn nothing, the bank does not move
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
lf.online.fit_predict(load_state="bank.state")   # resume from a saved bank
```

The plan is *pure*: every execution starts from the same state — the specs',
or `load_state` — so collecting twice gives the same frame, `head(n)` learns
from the first `n` rows and no more, and nothing is saved; the state after
the stream is `po.run(save_state=)`'s or your own bank's. Filters, selections
and `head` after the bank are pushed into the source and honoured there (a
filter after never changes what the bank learns from — put it before to do
that), and a selection reaches the input scan. The same numbers as the loop
and as `po.run`, bit for bit, in either engine, held by `tests/test_frame.py`.
`df.online.fit_predict(specs)` is the eager twin; `po.fit_predict(frame, ..)`
and `po.predict(frame, bank)` are both spelled for a type checker, which
cannot see a registered namespace. Rides on polars' IO-plugin interface
(`register_io_source`), which polars documents but marks unstable.

### 3. Streaming runner (Python or CLI)

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
already aligned and already ordered within each group, and it keeps O(state) per
stream. It deliberately does **not** provide:

- **connectors or ingestion** — feed it whatever Polars can read;
- **event-time windowing, tumbling/sliding windows, asof or interval joins** —
  build features with Polars expressions upstream, or with a streaming framework
  such as [Pathway](https://pathway.com), whose Rust engine already does this;
- **watermarks or late-arrival policy** — `clock`, `max_dclock`,
  `on_clock_reset` and `session` describe *within-stream* time, not pipeline
  lateness. A row that arrives out of order is a data error here, and
  `on_clock_reset="error"` will say so;
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
| `halflife` / `lam` | decay in clock units; mutually exclusive. A list of halflives means one accumulator per value |
| `max_dclock` | ceiling on the clock delta (required with `clock`); `0` disables decay, `inf` removes the ceiling |
| `on_clock_reset` | what a backwards clock means: `"max"` (default), `"zero"`, `"reset_state"`, or `"error"` to refuse the chunk — the bank is left as it was, so the corrected chunk can be fed |
| `session`, `session_gap` | on a session change, apply this delta (`"reset"` resets the state, `inf` never applies it) |
| `session_shrink`, `long_halflife` | `ewridge` only: at a session change, mix partway back toward a slow-moving twin — changes what the model believes, where `session_gap` only changes how confident it is |
| `weight` | row weight column |
| `min_periods` | in `n_eff` units; outputs are null until reached. A list gives one threshold per target — warmup gates output, not learning |
| `coef_every` | 0 = never; coefficients are also emitted on each chunk's last row |
| `group` | bank/CLI only; one state per key (the expression API uses `.over()`) |

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
default `halflife/50`). Ridge values and named `feature_sets` are expanded at
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
at a time. The expression plugin under `.over(group)` parallelizes the same
way: 12.2M rows/s at k=20 over 1000 groups.

**Memory.** The bank and the runner hold the state, the chunks in flight
(three, so `chunk_rows` is the knob) and whatever polars' reader prefetches:
on a 14-thread machine the parquet reader front-loads ~0.7 GB of decoded
row groups whatever the file's length, and `POLARS_ROW_GROUP_PREFETCH_SIZE=1`
takes the CLI to 0.15 GB at the same speed. Measured flat from 3M to 12M
rows in docs/PERFORMANCE.md §11, `lf.online.fit_predict` included (0.78 GB
live at 12M rows, 0.37 GB with the prefetch at 1). The expression plugin is
the O(data) surface — *Which spelling streams*, at the top, is the whole
comparison in one table.

Where the time goes, and what to reach for, is in
[`docs/PERFORMANCE.md`](docs/PERFORMANCE.md).

## Development

```sh
uv sync                                                # Python env (CPython 3.12)
./scripts/gate.sh                                      # everything CI checks
uv run cargo test --workspace                          # Rust tests
uv run maturin develop --release -m crates/online-py/Cargo.toml
uv run pytest                                          # Python tests
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

- Design and task list: [`docs/PLAN.md`](docs/PLAN.md)
- Measured defaults: [`docs/VALIDATION.md`](docs/VALIDATION.md)
- Adding a model: [`docs/EXTENDING.md`](docs/EXTENDING.md)

## Versioning and the Polars pin

### What is pinned, and why

`polars` is pinned **exactly**, in two places that must stay in sync
(`Cargo.toml` and `pyproject.toml`); a test asserts they agree.

| py-polars | rust polars | pyo3-polars | pyo3 | Python |
|---|---|---|---|---|
| **>= 1.34.0, < 2** (built and tested against 1.44.1) | 0.55.2 | 0.28 | 0.29 | ≥ 3.12 (`abi3-py312`) |

The *Rust* pin is exact and the wheel links it statically; the *runtime*
requirement is a range, because the two copies of Polars never meet. The floor
is `LazyFrame.collect_batches`, which `po.run` and `lf.online.fit_predict`
read with and py-polars added in 1.34.0; the whole suite passes on 1.34.0,
1.38.1 and 1.44.1 with identical numbers. `ModelBank` and the expression
plugin alone work from 1.28.1 (tested across 17 releases), and below that the
failure is a clean `AttributeError` on `PySeries._export` rather than anything
subtle. The matrix is in `docs/RELEASE-READINESS.md`.

This is stricter than the mechanism strictly requires, and it is worth being
precise about why, because "pinned" usually implies "fragile" and here it does
not.

Both the expression plugin and `ModelBank` move data across the boundary
through the **Arrow C Data Interface** — `SeriesExport` is a `#[repr(C)]`
struct of `ArrowSchema` and `ArrowArray` pointers, the same cross-language ABI
pyarrow and DuckDB use. `PyDataFrame` is not special-cased: it extracts
column-by-column as `PySeries`, each through `import_series`. This package does
*not* use `PyExpr` or `PyLazyFrame`, which are the genuinely version-sensitive
types that cross as serialized query plans.

That ABI is **negotiated, not assumed**. The plugin exports
`_polars_plugin_get_version()`, and Polars checks it at load time:

```
ComputeError: this polars engine doesn't support plugin version: 0-1
```

**So a mismatched Polars is a clear error, not a crash.** There is even a
dedicated check for layout drift (*"This Polars' version has a different
'binary/string' layout"*). The pin exists so you never see those messages, not
because something worse waits behind them.

### What the pin costs you

`polars-online` cannot currently be installed alongside a different `polars`.
If your environment requires, say, `polars>=1.45`, the resolver will refuse.
There is no workaround other than matching the pin.

At the time of writing, **1.44.1 is the latest Polars release**, so nothing is
excluded today.

### How the pin will move

A scheduled CI job (`.github/workflows/polars-canary.yml`) unpins both
`Cargo.toml` and `pyproject.toml` weekly and runs the whole suite against the
newest Polars. A failure there is the notification. That turns "is it safe to
widen the pin?" into a question answered by evidence rather than by caution: if
the canary passes across a few Polars releases, the constraint widens to a
range in a minor release.

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

`coef_index(spec)` does the same for the flat `coef` list — one row per
position, mapping it to (target, combo, term):

```python
pos = po.spec.coef_index(grid).filter(
    (pl.col("term") == "x1") & (pl.col("ridge") == 0.5)
)["position"].item()
```

Both come from the same Rust code that renders the names, so they cannot
drift from the strings. The index also carries each field's `dtype` (`f64`,
`bool` for `drift_*`, `str` for `selected_*`, `list[f64]` for `coef`), which
is the declaration the expression plugin makes to polars.

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

## License

Apache-2.0. See [CONTRIBUTING.md](CONTRIBUTING.md) to make changes,
[SECURITY.md](SECURITY.md) to report a vulnerability.
