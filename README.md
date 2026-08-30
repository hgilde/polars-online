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
  bit-identical output, and so does saving state mid-stream and resuming.

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

### 3. CLI

```sh
online --config examples/bank.toml
online --config examples/bank.toml --resume bank.state --save-state bank.state
online --config examples/bank.toml --dry-run     # validate + print the output schema
```

## Evaluation

```python
po.eval.metrics(out, "ridge", by=["bond_id"])                       # R², IC, hit rate, MSE
po.eval.rolling_metrics(out, "ridge", clock="t", window=3600.0)     # per clock window
po.eval.compare_specs(out, ["ridge", "kalman"])                     # one table, many specs
```

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
| `weight` | row weight column |
| `min_periods` | in `n_eff` units; outputs are null until reached |
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
bounded over arbitrarily long runs. `z` denotes `[1, x]` when an intercept is
configured.

### `ewridge` — EW ridge on sufficient statistics (the workhorse)

```
W'   = λW + w                       S' = (λW·S + w·z zᵀ) / W'
W_j' = λW_j + w                     r_j' = (λW_j·r_j + w·z·y_j) / W_j'
solve:  (S + ridge·D) β_j = r_j     D = I minus the intercept slot
```

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

### `huber` / `quantile` — robust regression

IRLS reweighting on the ridge update, using each row's **prior** residual so the
reweighting stays out-of-sample. Huber: `w = min(1, δσ/|r|)`. Quantile: the
check-loss weights at level τ. Weights are per target, so `S` is per target here.

### `ftrl` — online logistic regression

FTRL-proximal (McMahan et al. 2013) for binary targets, with the accumulators
decayed on the same clock as everything else:

```
β_i = 0 if |z_i| ≤ l1 else −(z_i − sgn(z_i)l1) / ((β + √n_i)/α + l2)
p   = sigmoid(zᵀβ)     g_i = (p − y)·z_i·w
z_i += g_i − ((√(n_i + g_i²) − √n_i)/α)·β_i      n_i += g_i²
```

`pred` is a probability and `resid = y − p`.

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
uv run cargo test --workspace                          # Rust tests
uv run maturin develop --release -m crates/online-py/Cargo.toml
uv run pytest                                          # Python tests
uv run python scripts/validate.py > docs/VALIDATION.md # re-run the [validate] experiments
uv run python scripts/benchmark.py                     # throughput
```

Prerequisites: [uv](https://docs.astral.sh/uv/) and a stable Rust toolchain
([rustup](https://rustup.rs)). Both install outside the default `PATH` on some
machines — `source scripts/env.sh` fixes that for a shell, and
`.vscode/settings.json` does it for VS Code's integrated terminal.

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
