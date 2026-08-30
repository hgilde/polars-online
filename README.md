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
release.

## The three usage modes

### 1. Expression plugin

Runs one spec over the column it receives, so `.over(group)` gives per-group
streams. Output is a struct column.

```python
out = df.with_columns(
    pl.col("y").online.ewridge(
        features=["x0", "x1"],
        clock="t", halflife=600.0, max_dclock=300.0,
        session="session", session_gap=1800.0,
    ).over("group").alias("fit")
)
out.select(pl.col("fit").struct.field("pred_y"), pl.col("fit").struct.field("n_eff"))
```

Grids are allowed but produce wide structs; the bank is the better surface for
grids.

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
| `targets`, `features` | column names; ≥1 target, shared `X'X` across targets |
| `add_intercept` | default `True` |
| `clock` | monotone **numeric** column (seconds, cumulative volume, …). `None` ⇒ row count. A temporal column is rejected — cast it first, e.g. `pl.col("ts").dt.epoch("s")` — because its internal representation would silently set the units of `halflife`, `max_dclock` and `session_gap` |
| `halflife` / `lam` | decay in clock units; mutually exclusive. A list of halflives means one accumulator per value |
| `max_dclock` | ceiling on the clock delta (required with `clock`) |
| `on_clock_reset` | what a backwards clock means: `"max"` (default), `"zero"`, `"reset_state"`, or `"error"` to refuse it |
| `session`, `session_gap` | on a session change, apply this delta (or `"reset"`) |
| `session_shrink`, `long_halflife` | `ewridge` only: at a session change, mix partway back toward a slow-moving twin — changes what the model believes, where `session_gap` only changes how confident it is |
| `weight` | row weight column |
| `min_periods` | in `n_eff` units; outputs are null until reached. A list gives one threshold per target — warmup gates output, not learning |
| `coef_every` | 0 = never; coefficients are also emitted on each chunk's last row |
| `group` | bank/CLI only; one state per key (the expression API uses `.over()`) |

Per-row decay is `λ = 0.5 ** (Δclock / halflife)`, and `n_eff` is the
exponentially weighted observation count under the same decay.

**Null policy.** A null in any feature (or the weight) skips the row entirely:
outputs are null, no update happens, but the clock still advances. A null in
target *j* still emits `pred_j`, leaves `resid_j` null, and skips only that
target's update.

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
P ← P/λ            g = P z / (1/w + zᵀ P z)
β_j ← β_j + g(y_j − zᵀβ_j)          P ← P − g zᵀP
```

Coefficients move every row with zero solve staleness (Sherman–Morrison).
`ridge` sets `P₀ = I/ridge` and penalizes the intercept. This is algebraically
identical to `ewridge(ridge_decay=True)` solved every row — a test asserts they
agree to <1e-9. *Null-policy deviation:* a row with any null target is
predict-only for all targets, since `P` is shared.

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
drives `n_eff`.

### `ew_cov` — exponentially weighted moments (no regression)

```
W'   = λW + w        m'ᵢ = (λW·mᵢ + w·xᵢ) / W'      S'ᵢⱼ = (λW·Sᵢⱼ + w·xᵢxⱼ) / W'
varᵢ = Sᵢᵢ − mᵢ²     covᵢⱼ = Sᵢⱼ − mᵢmⱼ             corrᵢⱼ = covᵢⱼ / √(varᵢ·varⱼ)
```

Running mean / variance / std / covariance / correlation of the columns you
name, on the same clock as every model here. With `precision_prior` set it also
tracks the inverse by Sherman–Morrison, giving `partial_corr` — the correlation
between two columns *controlling for all the others* — with no solve per row. One O(k²) update per row, replacing
the O(k²) *passes* a pure-Polars pairwise EW correlation needs. Values are read
from the state before each row, so an `ew_cov` output can be a feature for that
same row without leaking it.

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
| `ewridge` k=5 | 1 target, 1 halflife | 2,267,481 |
| `ewridge` k=20 | 1 target, 1 halflife | 1,586,467 |
| `ewridge` k=50 | 1 target, 1 halflife | 750,719 |
| `ewridge` k=20 | 10 targets | 1,044,468 |
| `ewridge` k=20 | 5 halflives | 549,760 |
| `rls` | k=20, 1 target | 1,517,571 |
| `kalman` | k=20, 1 target | 906,601 |
| `lasso` | k=20, 1 target (3-point path) | 1,091,575 |
| `huber` | k=20, 1 target | 1,562,718 |
| `ftrl` | k=20, 1 target | 1,770,600 |

Targets share one `S` accumulator, so 10 targets cost far less than 10× one.
Each halflife in a grid is its own accumulator, so those do scale roughly
linearly.

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

## Version pins

`polars` is pinned in two places that must stay in sync (`Cargo.toml` and
`pyproject.toml`); a test asserts they agree.

| py-polars | rust polars | pyo3-polars | pyo3 |
|---|---|---|---|
| 1.44.1 | 0.55.2 | 0.28 | 0.29 |

A scheduled CI job builds against the latest Polars so breakage shows up early.

## License

Apache-2.0.
