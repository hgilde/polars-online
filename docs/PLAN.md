# polars-online — design and plan

Status: design frozen 2026-08-29, no code yet. Items marked **[validate]** are defaults
chosen without data; check them in the evaluation harness (task 12) before relying on them.

## 1. Goal

Online regression models over ordered event data (one stream per group, e.g. per bond),
usable three ways with identical numerics:

1. **Expression plugin** — `pl.col("y").online.<model>(...)`, in-memory / lazy, with `.over(group)`.
2. **Python ModelBank** — chunk-fed, `fit_predict(chunk)` over `LazyFrame.collect_batches()`;
   memory is O(state), not O(data).
3. **Rust CLI** — same bank, parquet in → parquet out, TOML config, for Windows deployment.

Layers 2 and 3 share `online-polars`; all three share `online-core`.

## 2. Core contract (Rust, `online-core`)

```rust
pub trait OnlineModel {
    /// One row. `x` is the feature vector (intercept NOT included; core adds it if configured).
    /// `y[j] = None` => predict-only for target j, update the others.
    /// `d_clock` is the already-capped/gap-adjusted clock delta; `weight` scales the row.
    fn step(&mut self, x: &[f64], y: &[Option<f64>], d_clock: f64, weight: f64) -> Step;
    fn state(&self) -> State;            // versioned, serializable
    fn restore(s: &State) -> Result<Self, StateError>;
    fn n_targets(&self) -> usize;
    fn n_features(&self) -> usize;
}

pub struct Step {
    pub pred: Vec<f64>,                  // per target; NaN when not ready
    pub coef: Option<Vec<Vec<f64>>>,     // per target, only on coef_every rows / last row of chunk
    pub n_eff: f64,
    pub extra: Option<Extra>,            // model-specific (lasso path, lam_selected, ...)
}
```

Invariants: `pred` uses state *before* the update with this row; models are deterministic given
input order; no allocation in the hot path after warmup (preallocate buffers in the struct).

## 3. Common parameters (all entry points, same names)

| param | type | notes |
|---|---|---|
| `targets` | list[str] | ≥1; shared X'X, per-target X'y / coefficients |
| `features` | list[str] | f64 columns |
| `add_intercept` | bool | default true |
| `clock` | str \| None | monotone f64 column (seconds or cumulative volume). None ⇒ row count |
| `halflife` | float \| list[float] | clock units; mutually exclusive with `lam` |
| `lam` | float | per-row decay factor, alternative to `halflife` |
| `max_dclock` | float | ceiling on clock delta (required if `clock` given); `0` disables decay, `inf` removes the ceiling |
| `on_clock_reset` | `"max"` \| `"zero"` \| `"reset_state"` \| `"error"` | negative delta handling; default `"max"`. `"error"` refuses the whole chunk and leaves the bank untouched (IMPROVEMENTS C3) |
| `session` | str \| None | column; on change apply `session_gap` |
| `session_gap` | float \| `"reset"` | clock units to apply at session change |
| `weight` | str \| None | row weight column, default 1 |
| `min_periods` | float | in `n_eff` units; outputs null until reached |
| `coef_every` | int | 0 = never; also emitted on the last row of every chunk |
| `group` | str \| None | ModelBank/CLI only; one state per key. Expression API uses `.over()` |

Per-row decay: `λ_row = 0.5 ** (Δ / halflife)`; `n_eff` = EW count with the same decay.

### Clock semantics
- Δ = clock − prev_clock, clipped to `[0, max_dclock]` (with `on_clock_reset="zero"`) or
  Δ<0 ⇒ `max_dclock` (`"max"`, default) or state reset (`"reset_state"`).
- Session change ⇒ Δ := `session_gap` (or reset), regardless of the clock delta.
- First row of a group ⇒ Δ = 0.
- The clock is per group; the expression API gets the group's rows via `.over()`.

### Null policy
- Null in any feature ⇒ row skipped entirely: outputs null, no update, clock still advances.
- Null in target j ⇒ `pred_j` emitted, `resid_j` null, no update for j; other targets update.
- NaN, ±inf and `|v| > online_core::INPUT_BOUND` (1e100) in a feature, target or weight count
  as null (IMPROVEMENTS C2); models must stay finite and keep learning for anything inside the
  bound (`tests/model_contract.rs`).
- Warmup (`n_eff < min_periods`) ⇒ all outputs null except `n_eff`.

### Output
Struct per model, fields: `pred_<t>`, `resid_<t>` for each target `t`; `coef` (list of lists,
null except on coef rows); `n_eff`; model-specific extras. ModelBank returns one struct column
per spec, named by the user.

## 4. Models

Build order matters: 4.1 is the workhorse and its accumulators are reused by 4.2–4.4.

### 4.1 EW-ridge (sufficient statistics) — primary
State: `S = EW Σ w·x·xᵀ` (k×k, intercept included), `r_j = EW Σ w·x·y_j` per target,
`n_eff`, EW residual variance `σ²_j`, `last_solve_clock`, coefficients per target per grid point.
- Update: O(k²) per row. Solve: Cholesky of `S + λ·D` where D excludes the intercept.
- **Standardization** (`standardize: bool`) at solve time using the means/variances contained in S:
  rescale to correlation form, solve, unscale. Default true for lasso, false for ridge.
- **Grids expanded at solve time, not into separate accumulators**:
  `ridge: list[float]`, `feature_sets: dict[str, list[str]]` (sub-blocks of S), `lasso_path` (4.3).
  `halflife: list` IS a separate accumulator per value — allowed, documented as costing k² per row each.
- **Solve schedule**: `solve_every` in clock units, default `halflife/50`; `max_rows_between_solves`
  cap; forced solve on first-ready and after any capped/session gap. **[validate]** the /50 default
  via task 12 (schedules share the accumulator, so this is a free experiment).
- Between solves, predictions use the last solved coefficients.

### 4.2 RLS (recursive least squares) — variant
Decayed ridge least squares solved exactly every row, O(k²), zero staleness. Square-root (QR)
form: the state is the Cholesky factor of `A = S+λ₀I` and the rotated right-hand side, updated by
Givens rotations, not the covariance `P = A⁻¹` (whose recursion drifts and can freeze —
docs/IMPROVEMENTS.md C5). Cannot share ridge grids (λ₀ is baked into `A₀`). Params: `ridge`
(scalar, as `A₀ = ridge·I`, i.e. `P₀ = I/ridge`), `coef0`. Included mainly as the reference for 4.1
and for very small k.

### 4.3 Lasso path — on top of 4.1
Coordinate descent on standardized `S`, `r_j` over `lasso_path: list[float]` (decreasing),
warm-started along the path and across solves. Returns:
- full path coefficients every `coef_every` rows,
- `lam_selected_j`: argmin over the path of an EW of squared out-of-sample error (halflife
  `select_halflife`, default = model halflife) — free, since preds for all λ are computed anyway,
- `pred_j` / `resid_j` from the selected λ.
Elastic net via `l1_ratio` **[validate]** — cheap to add, may not be needed.

### 4.4 Kalman / random-walk-β (dynamic linear model)
State per target: coefficient mean `β_j`, covariance `P_j` (k×k); shared: `σ²_j` EW residual variance.
- Process noise **derived from a per-factor halflife**: `q_i = σ² · (ln2 / h_i)²` on standardized
  features (steady-state gain matching with EW-RLS). `halflife` may be scalar or per-factor list;
  `halflife=inf` pins a coefficient. Explicit `q: list[float]` overrides.
- Observation noise = `σ²_j` (EW residual variance) unless `obs_var` given.
- Because P is per target (Riccati depends on σ²_j), targets do not share the k×k work here;
  documented, and `share_p: bool` **[validate]** offers the approximation P shared with σ² = mean.

### 4.5 Robust: Huber and quantile regression
IRLS-style reweighting on 4.1's update: each row's weight is scaled by the robust weight of its
*prior* residual (so still out-of-sample). Because the weights are per target, S is per target
here — one accumulator per target, same API. Params: `huber_delta` in units of EW residual std
(default 1.5 **[validate]**), `quantile` (τ) for the quantile variant via the check-loss weights.

### 4.6 Online logistic / FTRL-proximal
For binary targets (direction, "signal accurate now"). FTRL-proximal with `alpha`, `beta`, `l1`,
`l2` (defaults from McMahan et al. **[validate]**), decay applied to the accumulators so it forgets
with the same clock as everything else. Output `pred` is a probability; `resid = y − p`.

### 4.7 Shared primitive
`EwCov`: EW covariance matrix, with an optional regularized precision matrix solved on demand
(not tracked incrementally — IMPROVEMENTS C5) — used by 4.1, 4.4 and exposed on its own as
`online.ew_cov()` (replaces pure-Polars pairwise EW correlations when k>2).

## 5. Model bank (`online-polars`)

- Takes `list[Spec]`; extracts feature/target/clock/session/weight columns from a chunk once,
  computes the clock delta once per row, then runs each spec's state (per group key) — specs are
  independent, so they run in parallel with `rayon` over (spec × group).
- Chunks must be clock-ordered within each group; the bank asserts monotonicity (after reset
  handling) and errors loudly otherwise.
- `state()` / `save(path)` / `load(path)`: msgpack (`rmp-serde`) with header
  `{schema_version, package_version, spec}`; loading checks the spec matches.
- Python: `ModelBank(specs).fit_predict(df) -> df` (appends struct columns); Rust CLI reads the
  same specs from TOML.

## 6. Expression plugin (`online-py`)

`pyo3-polars` expression with `is_elementwise=False`, `returns_scalar=False`, one namespace
`online` with one function per model. Runs a single spec over the full column it receives
(so `.over(group)` gives per-group streams). Output dtype is a struct built from the spec.
Grids are allowed but produce wide structs; the bank is the recommended surface for grids.

## 7. Numerics

- f64 only. `faer` for Cholesky/solves; fall back to a jittered diagonal (`S + εI`, ε from the
  trace) when the factorization fails during warmup — never NaN silently, record `solve_failures`.
- EW accumulators are scaled so that S is a weighted *mean*, not a sum (stable under long runs).
- Standardization uses S's own means/variances; a feature with ~0 variance is dropped from the
  solve for that step (coefficient 0) rather than blowing up.
- Reference implementations in Python (numpy) live in `tests/reference.py` and are used as
  oracles; they are deliberately slow and simple.

## 8. Evaluation harness (`python/polars_online/eval.py`)

Pure Polars over the output structs: per (spec, group, target) rolling out-of-sample R², IC,
hit rate over configurable clock windows; one `group_by` to compare specs. Used for the
**[validate]** items and for the solve-schedule experiment.

## 9. Test plan

**Rule: tests download or generate their own data. No fixtures in the repo.** `tests/data.py`:
- `synthetic(seed, n_groups, n_rows, k, beta_process=...)`: seeded generator with known,
  time-varying β, irregular clock, session breaks, nulls, and a volume clock that resets per
  session — so oracle tests know the truth.
- `public_intraday()`: downloads a small free intraday dataset (choose one with a stable URL;
  crypto minute bars are the pragmatic option), cached under `.cache/`, `pytest.skip` when offline.

Test classes:
1. **Oracle**: on synthetic data, EW-ridge / RLS / Kalman / lasso match `tests/reference.py`
   to 1e-9 (RLS vs EW-ridge with `solve_every` = 1 row must agree to float precision).
2. **Chunk invariance**: stream in 1, 7, 1000 chunks ⇒ bitwise-identical outputs. Same for
   save/load mid-stream.
3. **Out-of-sample by construction**: a target that is pure noise must give IC ≈ 0; leaking the
   current row makes this test fail.
4. **Clock semantics**: gap cap, reset (`"max"`/`"zero"`/`"reset_state"`), session gap, first
   row, per-group independence.
5. **Null policy** and **warmup** exactly as in §3.
6. **Expression ≡ bank**: same spec through `.over()` and through `fit_predict` gives identical
   output.
7. **Cross-platform state**: a state written on CI macOS loads on CI Windows (artifact hand-off).
8. **Benchmark** (not a test): rows/sec for k ∈ {5, 20, 50}, 1 vs 10 targets, 1 vs 5 halflives.

## 10. Packaging / CI

- Cargo workspace at root; `pyproject.toml` at root with `maturin` backend pointing at
  `crates/online-py`. Package `polars-online`, import `polars_online`. (Check PyPI name is free
  before first publish; not needed for private use.)
- GitHub Actions: `ubuntu` for lint + Rust tests (fast), `macos-latest` (arm64) and
  `windows-latest` build wheels + CLI binary, run pytest, and run the cross-platform state test.
  Wheels and binaries uploaded as artifacts / GitHub release.
- Pin Polars; add a scheduled job that tries the latest Polars so breakage is noticed early.

## 11. Task list

Each task ends with green `cargo test` + `pytest`, a commit, and a tick here.

- [x] 1. Scaffold: workspace, four crates, `pyproject.toml`, `uv`, `maturin develop` builds an
      empty plugin, CI runs lint on all three OSes.
- [x] 2. `tests/data.py`: synthetic generator + public download + caching + offline skip.
      `tests/reference.py`: numpy EW-ridge and RLS oracles.
- [x] 3. `online-core`: `OnlineModel` trait, `Step`, clock/decay helper (`halflife`/`lam`,
      `max_dclock`, `on_clock_reset`, session gap), `EwCov` primitive. Unit tests.
- [x] 4. EW-ridge (§4.1) single target, no grids; Cholesky via `faer`; solve schedule.
- [x] 5. Multi-target + `feature_sets` + `ridge` grid + standardization at solve time.
- [x] 6. `online-polars` model bank: column extraction, per-group state, rayon fan-out, chunk
      monotonicity check, save/load (msgpack, versioned).
- [x] 7. `online-py`: `ModelBank` class + `fit_predict`; pytest oracle, chunk-invariance, null,
      warmup, clock tests pass.
- [x] 8. Expression plugin (`online.ewridge`) + expression≡bank test.
- [x] 9. RLS (§4.2) + agreement test with EW-ridge at `solve_every`=1 row.
- [x] 10. Lasso path + online λ selection (§4.3).
- [x] 11. Kalman (§4.4) with halflife-derived q; per-factor halflife; `inf` pinning.
- [x] 12. Evaluation harness (§8); run the solve-schedule experiment and every **[validate]**
      item on public data; record results in `docs/VALIDATION.md` and fix defaults.
- [x] 13. Robust models (§4.5).
- [x] 14. Logistic / FTRL (§4.6).
- [x] 15. `online-cli`: TOML specs, parquet streaming in/out, progress, resume from state.
- [x] 16. CI: wheels + CLI binaries for macOS/Windows, cross-platform state test, Polars-latest
      canary job. Benchmark script + numbers in README.
- [x] 17. README with the three usage modes and the math per model.
- [ ] 18. **Weekly native leak check in CI — after the repo is public.** Add
      `scripts/leakcheck.sh` to a scheduled Linux job (valgrind, with CPython's
      suppression file) and to the weekly macOS run (`leaks`). Deliberately
      deferred, not forgotten: it is worthless on a budget, because valgrind is
      roughly 50x slower than the suite and macOS bills at 10x, and the whole
      2,000-minute month went in a single day while the repo was private
      (`docs/RELEASE-READINESS.md`). Once Actions is unmetered the cost is
      irrelevant and the value is real — `tests/test_ffi_memory.py` can only
      see a leak large enough to move RSS, whereas `leaks`/valgrind find
      unreachable blocks of any size. **Trigger: the repo going public.** Both
      currently report clean, so the job starts from a known-good baseline;
      wire it as reported-not-gating first (like the benchmark job), since
      valgrind on CPython is noisy until the suppressions are tuned.

## 11a. Decisions made while implementing

**Clock dtype decision, 2026-08-30.**

`clock` must be a **numeric** column; temporal dtypes (`Datetime`, `Date`,
`Duration`, `Time`) are rejected with an error naming the column, its dtype and
the fix. Casting a temporal column to f64 exposes its internal representation,
so identical wall-clock data yields deltas differing by 10^3-10^6 depending only
on the column's time unit, and `halflife` / `max_dclock` / `session_gap` inherit
those units. `halflife = 600` against a microsecond-backed `Datetime` therefore
meant 600 microseconds: every row decays to nothing and the output is finite,
non-null, plausible-looking garbage — the worst failure shape available, since
none of the existing guards catch it. Rejecting costs one expression at the call
site (`pl.col("ts").dt.epoch("s")`) and makes the intended scale explicit,
consistent with the null-clock error and hard-rule bias toward loud failure.
Auto-converting to seconds was considered and declined: it would make the
meaning of `halflife` depend on the input dtype, which is the same class of
implicitness that caused the problem.

**Tasks 15-17 (CLI, release CI, README), 2026-08-30.**

- The streaming runner lives in `online-polars` (`RunConfig` / `run_config`), not
  in the CLI crate, so the same code path is testable without spawning a process
  and could back a Python streaming API later.
- Output is written with polars' `BatchedWriter`: one row group per chunk, so
  memory stays O(state + chunk) end to end.
- The cross-OS state test (§9 class 7) is one test file driven by two env vars:
  `ONLINE_WRITE_STATE` writes the hand-off artifact, `ONLINE_FOREIGN_STATE`
  loads one. CI writes on macOS and reads on Windows and Linux; without the env
  vars the test still checks the round trip locally, and `save_bytes` is
  asserted deterministic (which is what makes the hand-off meaningful).
- Benchmarks (Apple M-series, 200k rows, best of 3): ew_ridge 2.27M rows/s at
  k=5, 1.59M at k=20, 0.75M at k=50; 10 targets cost ~1.5x one target (shared
  S), while 5 halflives cost ~2.9x one (separate accumulators, as documented).
  ftrl is the fastest model, kalman the slowest.
- Fixed a real bug the README examples caught: `eval.unpack` collided when a
  target column was literally named `y`; reserved output names are now dropped
  from the passthrough columns.

**Tasks 13-14 (robust + logistic), 2026-08-30.**

- The robust models expose two spec types, `huber` and `quantile`, over one core
  `Robust` model (they differ only in the IRLS weight function).
- `huber_delta` default 1.5 kept: a sweep is only meaningful against a
  contamination model, and the unit test confirms 1.5 recovers the clean slope
  under 3% gross outliers where least squares does not.
- Quantile weights are scaled by the residual std so they are O(1) rather than
  O(1/sigma); `quantile_eps` (default 1e-3, in units of that std) floors |r| so a
  near-zero residual cannot produce an unbounded weight.
- FTRL defaults `alpha=0.1, beta=1.0, l1=0.0, l2=1.0` kept (McMahan et al.).
  Note FTRL's L1 zeroes a coordinate only while `|z_i| <= l1`, and `z_i` grows
  with accumulated gradient, so a moderate `l1` shrinks rather than permanently
  pins - the unit test asserts that behaviour, not exact sparsity.
- **Gotcha worth knowing**: `solve_every` defaults to `halflife/50`, so a very
  large `halflife` (e.g. 1e9, meaning "never forget") means the model solves
  once and never again unless `max_rows_between_solves` is set. Left as-is
  (it is the documented default), but every long-halflife test sets an explicit
  cadence.

**Task 12 (evaluation + [validate] items), 2026-08-30.**

Full numbers in `docs/VALIDATION.md` (regenerate with
`uv run python scripts/validate.py > docs/VALIDATION.md`). Data: 10 days of
BTCUSDT 1-minute bars (14,336 rows) from Binance's public dump, features = past
returns / volume / trade-count z-scores, targets = strictly future returns.

- **`solve_every` = halflife/50 is confirmed as the default.** Sweeping
  halflife/d for d in {1, 5, 10, 50, 200, 1000}: d = 50 gives the lowest MSE
  (1.16172e-06). Solving every row (d = 1) is *worse*, not better -- the extra
  responsiveness is noise. Kept as-is.
- **`standardize` default (false for ridge) confirmed**: on this data plain and
  standardized ridge are within 0.3% MSE of each other, so the default stays
  off for ridge (cheaper) and on for lasso (required by the algorithm).
- **`l1_ratio`: no evidence elastic net is needed.** At matched penalties,
  l1_ratio 1.0 / 0.5 / 0.1 are within ~1% MSE. The parameter is kept (it is
  ~free) but the default stays 1.0 = pure lasso.
- **Kalman `share_p`: not recommended as a default.** With two targets of very
  different noise levels, sharing P helped the short-horizon target slightly
  (-0.049 vs -0.060 R2) but hurt the long-horizon one (-0.085 vs -0.011 R2).
  Default stays `share_p = false`; it remains available.
- Solve-schedule sweeps really are free: 6 schedules over 14k rows in 0.06s,
  because they share one accumulator.
- `tests/data.py` now downloads N days (`public_intraday(dates)`), cached per
  day, so the validation set is 10x bigger than one day.
- Note: negative out-of-sample R2 on this data is expected and not a bug -- a
  1-minute crypto return is close to unpredictable from these features.

**Tasks 4-5 (EW-ridge), 2026-08-30.**

- Grid combos are ordered target-major then (feature_set x ridge); output slots
  `n_targets * n_combos`. Coefficient vectors are always full `k_total` length with
  zeros outside a combo's feature set.
- The forced solve "after any capped/session gap" is implemented as: the accumulated
  clock-since-solve includes the gap, so any gap >= `solve_every` triggers a solve on
  the next row. There is no separate force flag.
- `sigma2_j` uses the first-combo (primary) prediction's residual.
- ridge_decay (the decaying-prior / RLS-equivalent mode) refuses grids and
  standardization at validation time.
- Standardized solves drop a feature when its centered variance is < 1e-10 x its raw
  second moment (cancellation noise scales with the raw moment).
- Solve failure even after jitter keeps the previous coefficients and increments
  `solve_failures` (never NaN silently).

**Task 1 (scaffold), 2026-08-30.**

- Version pins (`Cargo.toml` workspace deps ↔ `pyproject.toml`, kept in sync by hand and
  asserted in `tests/test_scaffold.py`): py-polars **1.44.1** ↔ rust polars **=0.55.2** ↔
  pyo3-polars **0.28** ↔ pyo3 **0.29**. py-polars 1.44.1 is built from rust polars 0.55.1;
  0.55.2 is the same minor and is what pyo3-polars 0.28 resolves its sub-crates to, so the
  facade is pinned there to keep one polars version in the graph.
- Rust edition 2024, `rust-version = 1.85`, `rust-toolchain.toml` pins the stable channel.
- Two non-obvious feature flags on the polars/pyo3-polars side, both needed to compile at all
  (upstream feature-unification gaps in 0.55.2), both commented at the call site:
  - `polars` needs `object`: `pyo3-polars/derive` turns on `polars-plan/python`, which turns on
    `polars-core/object`, and `polars-ops` then fails an exhaustive `DataType` match.
  - `pyo3-polars` needs `lazy` alongside `derive`: only `polars-lazy/python` propagates
    `python` to `polars-mem-engine`, which otherwise fails an exhaustive `DeletionFilesList`
    match.
- `pyo3` is declared **without** `extension-module`; maturin adds it via `features` in
  `pyproject.toml`. With it always on, plain `cargo build`/`cargo test` fails to link.
- `online-py` builds with `abi3-py312`, so the Rust build needs a >= 3.12 interpreter present.
  Everything (including `cargo`) therefore runs under `uv run`, which exports `VIRTUAL_ENV`;
  CI does the same. This is why CI runs `uv sync` before any cargo step.
- `online-core` sets `unsafe_code = "forbid"` at the crate level (hard rule 6).
- `doc/` was renamed to `docs/` to match `CLAUDE.md` and task 12's `docs/VALIDATION.md`.
- CI (`.github/workflows/ci.yml`): `lint` and `test` jobs, each a matrix over
  ubuntu/macos/windows. Lint = `cargo fmt --check`, `cargo clippy -D warnings`,
  `ruff format --check`, `ruff check`. Test = `cargo test --workspace`, `maturin develop`,
  `pytest`. Wheel/binary release jobs are task 16.

## 11b. Follow-on documents

- `docs/ENHANCEMENTS.md` — plan-debt items (`ew_cov` surface, strict clock,
  negative-weight validation, ...) and a feature comparison against river.
- `docs/TESTING.md` — coverage scorecard against §9, found edge-case defects,
  and the oracle/river cross-check backlog.

## 11b. Performance plan

**Done — P1 through P8.** See [`docs/PERFORMANCE.md`](PERFORMANCE.md): the
integration layer cost 3–5× the model arithmetic and capped thread scaling at
3.2× on ten cores. Removing per-row allocation, flattening the rayon fan-out to
(spec × group × instance), extracting columns as `f64`-with-NaN instead of
`Option<f64>`, and pipelining the runner took it to **2.0–2.8× throughput and
6.2× scaling**, with every golden number unchanged. Three of the eight items
were closed by measuring and *rejecting* the change; §5 there records why.

## 11c. Simplification review

[`docs/SIMPLIFICATION.md`](SIMPLIFICATION.md) (S1–S6): a post-performance read
for complexity that can go without costing features, speed or stability. The
one that matters is S1 — the output schema's ordering is written out twice, in
`output_fields()` and again in `assemble()`, which is the duplication that
produced the E23 declared-vs-realized defect and the reason a guard test exists
for it. Proposed, none implemented; two items are recorded as deliberately
deferred and four approaches as rejected.

## 11d. Release readiness and API stability

[`docs/RELEASE-READINESS.md`](RELEASE-READINESS.md): what is left before the
repo goes public (workflow permissions, SHA-pinned actions, whether the Rust
crates are published, branch protection, a history scan), and how to keep the
API promisable. The finding worth acting on is that **the output field names
are the largest and least-guarded part of the API** — users index
`pred_y__r0.000001@h100` by string, and exactly one spec shape is currently
pinned. The proposal is one API snapshot test covering symbols, signatures with
defaults, and `output_fields()` across a matrix of spec shapes, so every API
change becomes a reviewable diff. Proposed, not implemented.

## 11e. Beyond O(state)

[`docs/BEYOND-O-STATE.md`](BEYOND-O-STATE.md): what a relaxed memory bound would
unlock, checked against crates.io so "nobody has built this" is evidence rather than
assumption. Three strong candidates (adaptive conformal prediction, frequent-directions
sketching, rolling-window regression), three weak, and Hoeffding trees left to MOA on
purpose. Survey only — nothing proposed. The one condition attached: a relaxed bound
would have to become a *stated, tested* property, not a habit.

## 11f. Pre-release improvements review

[`docs/IMPROVEMENTS.md`](IMPROVEMENTS.md) (C1–C5, P1–P3, U1–U4, X1–X2, T1–T4):
one pass per axis — correctness, performance, usability, extensibility,
testing — with every finding reproduced before it was written down. Done so
far: the emit flags through the expression plugin (C1), a bounded-input
contract for every model with the test that enforces it (C2/T4), `.over()`
running groups in parallel (P1), features as expressions (U1), a chunk
refused under `on_clock_reset="error"` leaving the bank untouched (C3),
parameters that used to run and produce garbage refused by name (C4), and
the one T4 found: covariance-form `rls` and `ew_cov`'s tracked inverse die of
cancellation on a single extreme row, so `rls` is now in square-root (QR)
form and the precision matrix is solved on demand (C5, schema 2). The rest is
proposed with its measurements next to it.

## 12. Open questions (not blocking)

- ~~Overnight handling beyond `session_gap` (e.g. partial state shrinkage toward a long-run prior).~~
  Answered: `session_shrink` + `long_halflife` mix the accumulators toward a
  slow-moving twin at a session boundary (ENHANCEMENTS E6).
- ~~Whether targets at long horizons need a different `min_periods` than short ones.~~
  Answered: `min_periods` accepts a list, one entry per target (ENHANCEMENTS E7).
- ~~Public intraday dataset choice for tests (stable URL, permissive licence).~~
  Answered and in use: Binance's public daily kline dump
  (`data.binance.vision`, BTCUSDT 1-minute bars) — stable per-day URLs, no
  auth. `tests/data.py` downloads it on demand, caches under the gitignored
  `.cache/`, and skips when offline, so hard rule 1 holds. It backs both the
  reference comparisons and the defaults measured in `docs/VALIDATION.md`
  (14,336 rows).
