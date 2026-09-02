# polars-online

Streaming / online regression models for [Polars](https://pola.rs). A Rust core
exposed three ways with identical numerics:

1. **an expression plugin** — `pl.col("y").online.ewridge(...)`, with `.over(group)`;
2. **a chunk-fed `ModelBank`** — memory is O(state), not O(data);
3. **a standalone CLI** — parquet in, parquet out, config from TOML.

Built for ordered event data (one stream per group) that does not fit in memory:
irregular clocks, session breaks, gaps, nulls, and per-group state.

```python
import polars as pl
import polars_online as po

df.with_columns(
    pl.col("ret").online.ewridge(
        features=["signal_a", "signal_b"],
        clock="ts", halflife=600.0, max_dclock=300.0,
    ).over("bond_id")
)
```

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
release. The wheel is ~17 MB to download and ~50 MB installed: it statically
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

Features are column names or named expressions; under `.over` an expression is
evaluated per group, so the lag above never crosses a group boundary. Groups
run in parallel (the inputs travel as one packed struct, which is the polars
path that spreads groups over threads — see docs/PERFORMANCE.md P5). Grids are
allowed but produce wide structs; the bank is the better surface for grids.

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
    ...

bank.save("bank.state")                    # versioned msgpack, portable across OSes
bank = po.ModelBank.load("bank.state", specs=[spec])
```

### 3. Streaming runner (Python or CLI)

The same O(state + chunk) parquet→parquet path, from Python:

```python
po.run(
    input="ticks.parquet", output="fitted.parquet",
    specs=[spec], chunk_rows=100_000, save_state="bank.state",
)                                    # -> {"rows": ..., "chunks": ...}

po.run("bank.toml", input="today.parquet")   # keywords override the config
```

or from the CLI:

**Paths on Windows.** A backslash starts an escape sequence in a TOML basic
string, so `input = "C:\data\in.parquet"` is a parse error, not a path. Any of
these works:

```toml
input = 'C:\data\in.parquet'      # literal string (single quotes), no escaping
input = "C:\\data\\in.parquet"    # basic string, backslashes doubled
input = "C:/data/in.parquet"      # forward slashes are fine on Windows
```

```sh
online --config examples/bank.toml
online --config examples/bank.toml --resume bank.state --save-state bank.state
online --config examples/bank.toml --dry-run     # validate + print the output schema
```

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
| `targets`, `features` | column names; ≥1 target, shared `X'X` across targets. Columns must be numeric (Boolean counts): a String column is refused, not cast to null |
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
mean squared **out-of-sample** error — costs nothing extra.

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
po.spec.holt("baseline", targets=["y"], clock="t",
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
| `ewridge` k=5 | 1 target, 1 halflife | 5,823,285 |
| `ewridge` k=20 | 1 target, 1 halflife | 2,775,086 |
| `ewridge` k=50 | 1 target, 1 halflife | 846,000 |
| `ewridge` k=20 | 10 targets | 1,391,097 |
| `ewridge` k=20 | 5 halflives | 1,328,381 |
| `rls` | k=20, 1 target | 3,131,992 |
| `kalman` | k=20, 1 target | 1,345,525 |
| `lasso` | k=20, 1 target (3-point path) | 1,503,706 |
| `huber` | k=20, 1 target | 2,852,545 |
| `ftrl` | k=20, 1 target | 4,814,801 |

Targets share one `S` accumulator, so 10 targets cost far less than 10× one.
Each halflife in a grid is its own accumulator, but they run in parallel, so a
5-halflife grid costs about 2× one rather than 5×.

**Grouped data goes wider.** One state per group is one rayon task, so
throughput rises with the group count rather than falling: **5.1M rows/s** at
k=20 over 64 groups, scaling 6.2× from one thread to ten. A bank of several
specs is one flat task pool too — eight single-group specs over 300k rows take
202 ms, against 1.2 s if they ran one at a time. The expression plugin under
`.over(group)` parallelizes the same way: 6.4M rows/s at k=20 over 1000 groups.

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

## Versioning and the Polars pin

### What is pinned, and why

`polars` is pinned **exactly**, in two places that must stay in sync
(`Cargo.toml` and `pyproject.toml`); a test asserts they agree.

| py-polars | rust polars | pyo3-polars | pyo3 | Python |
|---|---|---|---|---|
| **>= 1.28.1, < 2** (built and tested against 1.44.1) | 0.55.2 | 0.28 | 0.29 | ≥ 3.12 (`abi3-py312`) |

The *Rust* pin is exact and the wheel links it statically; the *runtime*
requirement is a range, because the two copies of Polars never meet. One wheel
was tested against 17 py-polars releases: 1.28.1 through 1.44.1 pass both entry
points with identical numbers, and below that the failure is a clean
`AttributeError` on `PySeries._export` rather than anything subtle. The matrix
is in `docs/RELEASE-READINESS.md`.

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
idx = po.spec.output_index(spec)
name = idx.filter(
    (pl.col("kind") == "pred") & (pl.col("target") == "y")
    & (pl.col("ridge") == 0.5) & (pl.col("halflife") == 500.0)
)["field"].item()          # -> "pred_y__r0.5@h500", resolved for you
out["m"].struct.field(name)
```

`coef_index(spec)` does the same for the flat `coef` list — one row per
position, mapping it to (target, combo, term):

```python
pos = po.spec.coef_index(spec).filter(
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
