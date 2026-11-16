# polars-online — design and plan

Status as of 2026-09-25: design frozen 2026-08-29; released through
**0.10.0**, and 0.11.0 is task 109. Open: tasks 78, 86 (parked), 104, 105,
107 (its window half), 109–120 and 125–127; task 106 was discarded; every
other task is done. Items marked
**[validate]** were defaults chosen without data; task 12 checked them on
public data, and `docs/VALIDATION.md` is the regenerated record.

How to read this file: §1–§10 are the original design, kept as written. §11 is the
task list, ticked as each task landed. §11a holds the decisions taken while building
— the place to look when the code does something §1–§10 did not say. §11b–§11h point
at the follow-on documents, and §12 records the questions the design left open and
what became of them. `docs/README.md` maps every document.

Finished work lives in the plan archive, `docs/records/PLAN-ARCHIVE-1.md` to
`PLAN-ARCHIVE-8.md`, in the order it stood here and word for word: each finished task
keeps one line below (its number and title, linking its entry), each closed
section keeps its heading and a line saying where its text went, so a
citation by task number or section number still resolves. Open tasks stay
here in full.

## 1. Goal

Online regression models over data that does not fit in memory -- ordered event streams
(one per group, e.g. per stock) with a clock, and equally a plain table in any row order,
where decay off (`halflife=inf`) is exact least squares at O(state) (the "Any row order"
decision below, 2026-09-03) -- usable two ways with identical numerics:

1. **Python ModelBank** — chunk-fed, `fit_predict(chunk)` over `LazyFrame.collect_batches()`;
   memory is O(state), not O(data). Also as a plan: `lf.online.fit_predict(specs)` is the bank
   registered as a polars IO-plugin source, a `LazyFrame` that streams when it runs (E33);
   `df.online.fit_predict(specs)` for a frame in memory.
2. **Streaming runner** — same bank as a read → fit → write pipeline, memory O(state + chunk):
   the Rust `online` CLI (parquet / ipc / csv / ndjson, TOML config, no Python) for
   deployment. Its Python entry point `po.run` was removed in task 83: `ModelBank` and the
   plan already do in-process work, and the runner dragged the lazy engine into the
   extension for code no Python path reached.

Both share `online-polars` and `online-core`. A third way, the **expression plugin**
(`pl.col("y").online.<model>(...)`, with `.over(group)`), was built first and removed in
task 85: polars calls a user expression with the whole column in either engine, so it was
the one O(data) surface, and a surface that cannot stream contradicts the point of the
library. Everything it did the bank and the plan do — a feature that was an expression
becomes a column computed before the call, and `.over(group)` becomes the spec's `group`.

## 2. Core contract (Rust, `online-core`)

```rust
pub trait OnlineModel {
    /// One row. `x` is the feature vector (intercept NOT included; core adds it if configured).
    /// `y[j] = None` => predict-only for target j, update the others.
    /// `d_clock` is the already-capped/gap-adjusted clock delta; `weight` scales the row.
    fn step(&mut self, x: &[f64], y: &[Option<f64>], d_clock: f64, weight: f64) -> Step;
    /// The step without the step: what `step` would report for this row, state untouched
    /// (`pred`, `n_eff`, `extra` identical row by row; `tests/model_contract.rs`).
    fn predict(&self, x: &[f64], d_clock: f64) -> Step;
    fn state(&self) -> State;            // versioned, serializable
    fn restore(s: &State) -> Result<Self, StateError>;
    fn n_targets(&self) -> usize;
    fn n_features(&self) -> usize;
}

pub struct Step {
    pub pred: Vec<f64>,                  // per target; NaN when not ready
    pub n_eff: f64,                      // coefficients come from `coefficients()`, read by the
                                         // stream layer on coef_every rows / each group's last
                                         // row within a chunk
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
| `clock` | str \| None | a column that does not run backwards within a group: numeric (seconds, cumulative volume, any units), or `Datetime`, `Date` or `Duration`, which make it a temporal clock whose parameters are durations (task 88). None ⇒ row count |
| `halflife` | float \| list[float] | clock units; mutually exclusive with `lam` |
| `lam` | float | per-row decay factor, alternative to `halflife` |
| `max_dclock` | float | ceiling on clock delta (required if `clock` given); `0` disables decay, `inf` removes the ceiling |
| `on_clock_reset` | `"max"` \| `"zero"` \| `"reset_state"` \| `"error"` | negative delta handling; default `"max"`. `"error"` refuses the whole chunk and leaves the bank untouched (IMPROVEMENTS C3) |
| `min_backwards_jump` | float | a backwards clock jump smaller than this (clock units) is out-of-order rows, not a boundary: adjacent rows are never further apart than `max_dclock` and a session is longer. The chunk is refused whatever `on_clock_reset` says. Default `max_dclock`, and `0` (off) under an infinite `max_dclock`; `0` disables. Needs `clock` |
| `session` | str \| None | column; on change apply `session_gap` |
| `session_gap` | float \| `"reset"` | clock units to apply at session change |
| `weight` | str \| None | row weight column, default 1 |
| `min_periods` | float | in `n_eff` units; outputs null until reached. The default depends on the model (`Spec::default_min_periods`): `k + add_intercept` for `lasso`, `kalman`, `huber`, `quantile`, `rls`, `sgd`, `pa` and `ftrl`; `k + 1` for `ew_cov`, `ew_class`, `kmeans`, `micro` and `holt`; 3 for `marginal` and `deco`; 1 for `bocpd`; and 0 for `seqtest`, `rcov`, `hmm` and `corrchange`, each gated otherwise, and for `ew_ridge`, where the noise gate below is the readiness gate and the rows its first solve needs are its own floor (docs/WARMUP-AND-CONVERGENCE.md, 2026-09-21) |
| `min_settled_frac` | float in `[0, 1)` | withhold predictions until the decay window has filled this far toward steady state, `settled_frac = 1 − 2^(−T/h)`, `T` the decay time seen; `0` (default) off. Off because a mean-form fit is unbiased from row one under stationarity; it guards a history that does not represent the process, which only the user can judge. Needs a decay |
| `max_error_inflation` | float `> 1` | withhold while `error_inflation = sqrt(1 + edf / n_kish)` -- the estimation error's expected inflation of a prediction's error over the noise floor -- exceeds this. Default `sqrt(2)`; `inf` off. Tracks the model; reads Kish's `n`. `ew_ridge` only; the rest keep `min_periods` |
| `emit_error_inflation` | bool | emit `error_inflation_<slot>`, the same ratio for the row's own features (its leverage against the fit's factor). `O(k²)` a row, hence opt-in; `ew_ridge` only |
| `coef_every` | clock units | unset: `coef` on **each group's** last row within every chunk — one row of coefficients per group per chunk, not one per chunk, so the emission schedule of `coef` follows the chunking while every other field is chunk-invariant (hard rule 3 is about the numbers); `0`: every row; a span: on the clock (task 178), with `max_rows_between_coefs` its row cap |
| `group` | str \| None | one state per key |

Per-row decay: `λ_row = 0.5 ** (Δ / halflife)`; `n_eff` = EW count with the same decay.

### Clock semantics
- Δ = clock − prev_clock, capped at `max_dclock`, which is required with a clock and
  finite and above 0: a cap on a step and nothing else (task 120, decided 2026-09-28).
- Δ<0 within a session: `on_clock_reset="error"`, the default, refuses the chunk,
  naming the row, and the bank is untouched. `"reset_state"` starts the model over,
  unless the step back is no larger than `min_backwards_jump` (required there, refused
  under `"error"`, and inclusive), which is a late row and is refused the same way.
  `0` starts over at every step back. `"max"` (the cap as the step) and `"zero"` are
  gone. It guards what the bank *learns*: `predict` scores a row before the last
  learned clock against the state as it stands, a step of 0, under either policy.
- No model ever receives a non-finite step; a property test over every configuration
  the spec allows holds it (`clock.rs`, `finite_steps`).
- Session change ⇒ Δ := `session_gap`, finite and at most the cap (or reset), regardless
  of the clock delta.
- First row of a group ⇒ Δ = 0.
- The clock is per group.

### Null policy
- Null in any feature ⇒ row skipped entirely: outputs null, no update, clock still advances.
- Null in target j ⇒ `pred_j` emitted, `resid_j` null, no update for j; other targets update,
  except in `rls`, whose targets share one factor and so update none. In `ewridge` and `lasso`
  the row stays in the other targets' Gram under either `target_gaps`.
- NaN, ±inf and `|v| > online_core::INPUT_BOUND` (1e100) in a feature, target or weight count
  as null (IMPROVEMENTS C2); models must stay finite and keep learning for anything inside the
  bound (`tests/model_contract.rs`).
- Warmup (`n_eff < min_periods`) ⇒ all outputs null except `n_eff`.

### Output
Struct per model, fields: `pred_<t>`, `resid_<t>` for each target `t`; `coef` (list of lists,
null except on coef rows); `n_eff`; `settled_frac` and `withheld_reason` (a categorical:
`below_min_settled_frac`, `above_max_error_inflation`, `below_min_periods`; null where nothing
was withheld) on every model that writes a row; `support_coef` beside `coef` for `ew_ridge`;
model-specific extras. ModelBank returns one struct column per spec, named by the user.

## 4. Models

Build order matters: 4.1 is the workhorse and its accumulators are reused by 4.2–4.4.

These are the seven original designs. The models added afterwards — `sgd`, `pa`,
`holt`, `kmeans`, `micro`, `ew_class`, `seqtest`, `marginal`, `deco`, `rcov`, `hmm`,
`corrchange` and `bocpd` — are designed in `docs/ENHANCEMENTS.md` (by E number) and
`docs/CLUSTERING.md`, decided in §11a under their task numbers, and stated as
recursions in the README's Models section and in each model's Rust module comment.

### 4.1 EW-ridge (sufficient statistics) — primary
State: `S = EW Σ w·x·xᵀ` (k×k, intercept included), `r_j = EW Σ w·x·y_j` per target,
`n_eff`, EW residual variance `σ²_j`, `last_solve_clock`, coefficients per target per grid point.
- Update: O(k²) per row. Solve: Cholesky of `S + λ·D` where D excludes the intercept.
- **Standardization** (`standardize: bool`) at solve time using the means/variances contained in S:
  rescale to correlation form, solve, unscale. Default true for lasso, false for ridge.
- **Grids expanded at solve time, not into separate accumulators**:
  `ridge: list[float]`, `feature_sets: dict[str, list[str]]` (sub-blocks of S), `lasso_path` (4.3).
  `halflife: list` IS a separate accumulator per value — allowed, documented as costing k² per row each.
- **Solve schedule**: `solve_every` in clock units; by default, under a finite halflife, a solve
  once the weight learned since the last reaches `ln 2 / 50` of the weight the fit holds, which is
  `halflife/50` of clock in steady state (task 115 (b); the default was that clock cadence, which
  never came due under a halflife much longer than the stream); every row under `lam` or an
  infinite halflife; `max_rows_between_solves` cap; forced solve on first-ready and after any
  capped/session gap. **[validate]** the /50 cadence via task 12 (schedules share the accumulator,
  so this is a free experiment).
- Between solves, predictions use the last solved coefficients.

### 4.2 RLS (recursive least squares) — variant
Decayed ridge least squares solved exactly every row, O(k²), zero staleness. Square-root (QR)
form: the state is the Cholesky factor of `A = S+λ₀I` and the rotated right-hand side, updated by
Givens rotations, not the covariance `P = A⁻¹` (whose recursion drifts and can freeze —
docs/IMPROVEMENTS.md C5). Cannot share ridge grids (λ₀ is baked into `A₀`). Params: `ridge`
(scalar, as `A₀ = ridge·I`, i.e. `P₀ = I/ridge`), `coef_prior`. Included mainly as the reference for 4.1
and for very small k.

### 4.3 Lasso path — on top of 4.1
Coordinate descent on standardized `S`, `r_j` over `lasso_path: list[float]` (decreasing),
warm-started along the path and across solves. Returns:
- full path coefficients every `coef_every` rows,
- `lam_selected_j`: argmin over the path of an EW of squared out-of-sample error (halflife
  `select_halflife`, default = model halflife) — free, since preds for all λ are computed anyway.
  Reported as it stood *before* the row's error joined the selection, i.e. the λ the row was
  scored with; that is what makes it identical between `fit_predict` and `predict` (E31),
- `pred_j` / `resid_j` from the selected λ.
Elastic net via `l1_ratio` **[validate]** — cheap to add, may not be needed.

### 4.4 Kalman / random-walk-β (dynamic linear model)
State per target: coefficient mean `β_j`, covariance `P_j` (k×k); shared: `σ²_j` EW residual variance.
- Process noise **derived from a per-factor halflife**: `q_i = σ² · (ln2 / h_i)²` on standardized
  features (steady-state gain matching with EW-RLS), added per row as `q_i · d²` for a row `d` clock
  units after the last, so the match holds at any spacing (task 150). `halflife` may be scalar or
  per-factor list; `halflife=inf` pins a coefficient. Explicit `q: list[float]` overrides.
- Observation noise = `σ²_j` (EW residual variance) unless `obs_var` given.
- Because P is per target (Riccati depends on σ²_j), targets do not share the k×k work here;
  documented, and `share_p: bool` **[validate]** offers the approximation P shared with σ² = mean.

### 4.5 Robust: Huber and quantile regression
Huber reweights on 4.1's update: each row's weight is scaled by the robust weight of its *prior*
residual (so still out-of-sample), bounded by 1. Quantile takes one Newton step on the check loss
smoothed by a uniform kernel of half-width `h = quantile_eps · σ`, linearised at the fit the row was
scored with: inside the band a least-squares row with target `y + 2h(τ − ½)`, outside it
`2h · ψ_τ(r) · z` into the cross-moment and no weight in the Gram (the review's N9 — IRLS's weight is
that step's secant, unbounded near zero, and freezing it held the fit near its own past fits).
Because the weights are per target, S is per target here — one accumulator per target, same API.
Params: `huber_delta` in units of EW residual std (default 1.5 **[validate]**), `quantile` (τ) and
`quantile_eps` (the band, default 0.2) for the quantile variant.

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
  independent, so they run in parallel over (spec × group) on the bank's own pool
  (`POLARS_ONLINE_MAX_THREADS`; polars' readers and writers stay on `POLARS_MAX_THREADS`).
- Under a `clock`, chunks must be clock-ordered within each group; the bank asserts
  monotonicity (after reset handling) and errors loudly otherwise. Without one the row
  order is the clock, and with decay off the order does not reach the fit at all.
- `state()` / `save(path)` / `load(path)`: msgpack (`rmp-serde`) with header
  `{schema_version, package_version, spec}`; loading checks the spec matches.
  Each stream's state also carries the output row of the last row it learned
  from, read back as `ModelBank.last_row()` (task 34), and a summary of what
  it was fed — row counts, weight sum, clock range, schedule events and
  per-column Welford moments — read back as `ModelBank.summary()` and
  `ModelBank.describe()` (task 35).
- Python: `ModelBank(specs).fit_predict(df) -> df` (appends struct columns) and
  `predict(df) -> df`, the same columns scored against the bank as it stands with nothing
  updated — every row from the same state, the clock distance measured from the last learned
  row, session/clock reset policies honored by scoring a fresh model (ENHANCEMENTS E31); Rust
  CLI reads the same specs from TOML.

## 6. Expression plugin (`online-py`) — removed in task 85, 2026-09-17

There was a third surface: a `pyo3-polars` expression with `is_elementwise=False`, one
namespace `online` with one function per model, running a single spec over the full column
it received (so `.over(group)` gave per-group streams), implemented as the bank itself so
that expression ≡ bank by construction. It is gone — `crates/online-py/src/expr.rs`,
`_expr.py`, `tests/test_expr.py`, the namespace, `po.online` and
`InMemoryExpressionWarning` — and this section records why, because the reason is a fact
about polars and not about this library.

**Why it could not stream.** Polars hands a non-elementwise user expression its whole
column: in the in-memory engine by definition, and in the streaming engine because a plugin
lowers to a `columnar-function` node — collect the input, call once, re-emit. There is no
way for a plugin to say "call me per morsel, in order, and let me keep state" (§11a,
2026-09-02). So `lf.with_columns(pl.col("y").online.ewridge(..)).sink_parquet(..)` measured
7.3 GB at 12M rows where `lf.online.fit_predict([spec]).sink_parquet(..)` measures 1.35 GB
(`docs/PERFORMANCE.md` §11): the same numbers, two memory profiles, and users read the
expression as the natural form. From 2026-09-03 (task 19) it warned on every use to say so.
A cargo-feature gate was tried first and reverted the same day — it handed users polars'
bare `AttributeError` and left the plugin's runtime tests skipped in CI — and the warning
was the answer that stood for two weeks. Once the plan form existed and streamed, a surface
whose every use carried a warning against itself cost more than it bought, and it was
removed rather than kept dormant. The build consequence was measured, not assumed:
`pyo3-polars/derive` was the only reason the lazy engine was in the extension's features.

**Nothing the model needed went with it.** The bank fans out over (spec × group) on its own
pool, so `group=` is the parallel path `.over(group)` was; `df.online.fit_predict(specs)` is
the in-memory call; a feature that was an expression becomes a column computed before the
call. What it alone offered was the plugin ABI's MAJOR/MINOR handshake, the one polars
stability guarantee this library rode on. Its place in that story is the Arrow PyCapsule
interface of `ModelBank.fit_predict_arrow` (task 86, CLAUDE.md rule 13), whose contract is
Arrow's rather than polars' to change.

**What would bring an expression form back.** A polars node that lets a user expression run
per morsel, in order, with state — a streaming-engine contract for stateful UDFs. Until then
an expression can only ever be the in-memory form, and the bank already is that.

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
  crypto minute rows are the pragmatic option), cached under `.cache/`, `pytest.skip` when offline.

**Rule: a test may use any library the package does not depend on, when the test needs it**
(the user, 2026-09-24: "packages that we do not want to depend on are fine if needed to test
with other libraries"; "be sure that the future test plan allows non dependent libs for
testing"). The need is an oracle, a second opinion from another implementation, or interop
with that library. The package still depends on polars alone. The library is declared in the
dev group by name, imported plainly rather than `importorskip`ed, and kept out of the package's
reach where its presence changes other libraries, as pyarrow is. `docs/TESTING.md`, "Libraries
the package does not depend on", has the rules, and `tests/test_dependency_policy.py` checks them.

Test classes:
1. **Oracle**: on synthetic data, EW-ridge / RLS / Kalman / lasso match `tests/reference.py`
   to 1e-9 (RLS vs EW-ridge with `solve_every` = 1 row must agree to float precision).
2. **Chunk invariance**: stream in 1, 7, 1000 chunks ⇒ bitwise-identical outputs. Same for
   save/load mid-stream.
3. **Out-of-sample by construction**: a target that is pure noise must give IC ≈ 0; leaking the
   current row makes this test fail.
4. **Clock semantics**: gap cap, the backwards policies (`"max"`, `"zero"`, `"reset_state"`,
   `"error"`) and `min_backwards_jump`, session gap, first row, per-group independence.
5. **Null policy** and **warmup** exactly as in §3.
6. **Query ≡ bank**: the same spec through `lf.online.fit_predict` and through
   `ModelBank.fit_predict` gives identical output (the expression form this compared
   was removed in task 85).
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

*Moved, text unchanged: the finished tasks' full entries and the notes between them (the parked and dropped decisions stay here), from [part 1](records/PLAN-ARCHIVE-1.md#s11) through [part 4](records/PLAN-ARCHIVE-4.md) of the plan archive.*

Each task ends with green `cargo test` + `pytest`, a commit, and a tick here.

- [x] 1. **Scaffold: workspace, four crates, `pyproject.toml`, `uv`, `maturin develop` builds an empty plugin, CI…** -- [part 1](records/PLAN-ARCHIVE-1.md#task-1)
- [x] 2. **`tests/data.py`: synthetic generator + public download + caching + offline skip** -- [part 1](records/PLAN-ARCHIVE-1.md#task-2)
- [x] 3. **`online-core`: `OnlineModel` trait, `Step`, clock/decay helper (`halflife`/`lam`, `max_dclock`…** -- [part 1](records/PLAN-ARCHIVE-1.md#task-3)
- [x] 4. **EW-ridge (§4.1) single target, no grids; Cholesky via `faer`; solve schedule** -- [part 1](records/PLAN-ARCHIVE-1.md#task-4)
- [x] 5. **Multi-target + `feature_sets` + `ridge` grid + standardization at solve time** -- [part 1](records/PLAN-ARCHIVE-1.md#task-5)
- [x] 6. **`online-polars` model bank: column extraction, per-group state, rayon fan-out, chunk monotonicity check…** -- [part 1](records/PLAN-ARCHIVE-1.md#task-6)
- [x] 7. **`online-py`: `ModelBank` class + `fit_predict`; pytest oracle, chunk-invariance, null, warmup, clock tests…** -- [part 1](records/PLAN-ARCHIVE-1.md#task-7)
- [x] 8. **Expression plugin (`online.ewridge`) + expression≡bank test** -- [part 1](records/PLAN-ARCHIVE-1.md#task-8)
- [x] 9. **RLS (§4.2) + agreement test with EW-ridge at `solve_every`=1 row** -- [part 1](records/PLAN-ARCHIVE-1.md#task-9)
- [x] 10. **Lasso path + online λ selection (§4.3)** -- [part 1](records/PLAN-ARCHIVE-1.md#task-10)
- [x] 11. **Kalman (§4.4) with halflife-derived q; per-factor halflife; `inf` pinning** -- [part 1](records/PLAN-ARCHIVE-1.md#task-11)
- [x] 12. **Evaluation harness (§8); run the solve-schedule experiment and every [validate] item on public data…** -- [part 1](records/PLAN-ARCHIVE-1.md#task-12)
- [x] 13. **Robust models (§4.5)** -- [part 1](records/PLAN-ARCHIVE-1.md#task-13)
- [x] 14. **Logistic / FTRL (§4.6)** -- [part 1](records/PLAN-ARCHIVE-1.md#task-14)
- [x] 15. **`online-cli`: TOML specs, streaming in/out (parquet, ipc, csv, ndjson — ENHANCEMENTS E32), progress…** -- [part 1](records/PLAN-ARCHIVE-1.md#task-15)
- [x] 16. **CI: wheels + CLI binaries for macOS/Windows, cross-platform state test, Polars-latest canary job** -- [part 1](records/PLAN-ARCHIVE-1.md#task-16)
- [x] 17. **README with the three usage modes and the math per model** -- [part 1](records/PLAN-ARCHIVE-1.md#task-17)
- [x] 18. **Weekly native leak check in CI — after the repo is public** -- [part 1](records/PLAN-ARCHIVE-1.md#task-18)
- [x] 19. **The expression plugin (task 8) is the in-memory form and says so: every `pl.col(..).online.<model>` call…** -- [part 1](records/PLAN-ARCHIVE-1.md#task-19)
- [x] 20. **State out of a streamed plan — researched and implemented 2026-09-03** -- [part 1](records/PLAN-ARCHIVE-1.md#task-20)
- [x] 21. **Gradient-boosted trees, as online as possible — investigated 2026-09-03** -- [part 1](records/PLAN-ARCHIVE-1.md#task-21)
- [x] 22. **Online clustering, every family that can be made to fit — investigated 2026-09-04, on the branch…** -- [part 1](records/PLAN-ARCHIVE-1.md#task-22)
- [x] 23. **`kmeans` — online k-means in the crates** -- [part 1](records/PLAN-ARCHIVE-1.md#task-23)
- [x] 24. **`micro` — DenStream-style micro-clusters with a linkage macro step** -- [part 1](records/PLAN-ARCHIVE-1.md#task-24)
- [x] 25. **E36: adaptive conformal intervals** -- [part 1](records/PLAN-ARCHIVE-1.md#task-25)
- [x] 26. **E37 + E38 on `ew_cov`** -- [part 1](records/PLAN-ARCHIVE-1.md#task-26)
- [x] 27. **E39: class-conditional `ew_cov`** -- [part 1](records/PLAN-ARCHIVE-1.md#task-27)
- [x] 28. **E40: constrained coefficients** -- [part 1](records/PLAN-ARCHIVE-1.md#task-28)
- [x] 29. **E41: diagonal transition `φ^d`** -- [part 1](records/PLAN-ARCHIVE-1.md#task-29)
- [x] 30. **E42: a sequential e-process test** -- [part 1](records/PLAN-ARCHIVE-1.md#task-30)
- [x] 31. **Performance and parallel-performance deep dive** -- [part 1](records/PLAN-ARCHIVE-1.md#task-31)
- [x] 32. **Prepare 0.2.0** -- [part 1](records/PLAN-ARCHIVE-1.md#task-32)
- [x] 33. **Diagonal standardizer for `kalman` and `sgd`** -- [part 1](records/PLAN-ARCHIVE-1.md#task-33)
- [x] 34. **Last-row diagnostics in the bank file** -- [part 1](records/PLAN-ARCHIVE-1.md#task-34)
- [x] 35. **The training data in the bank file** -- [part 1](records/PLAN-ARCHIVE-1.md#task-35)
- [x] 36. **`ew_cov` with no per-row output** -- [part 1](records/PLAN-ARCHIVE-1.md#task-36)
- [x] 37. **`marginal` model** -- [part 1](records/PLAN-ARCHIVE-1.md#task-37)
- [x] 38. **Complete the Gram export** -- [part 1](records/PLAN-ARCHIVE-1.md#task-38)
- [x] 39. **`po.gram`** -- [part 1](records/PLAN-ARCHIVE-1.md#task-39)
- [x] 40. **`label_delay`** -- [part 1](records/PLAN-ARCHIVE-1.md#task-40)
- [x] 41. **The Gram update, measured** -- [part 1](records/PLAN-ARCHIVE-1.md#task-41)
- [x] 42. **Mergeable evaluation sums** -- [part 1](records/PLAN-ARCHIVE-1.md#task-42)
- [x] 43. **A run with no per-row output** -- [part 1](records/PLAN-ARCHIVE-1.md#task-43)
- [x] 44. **Freeze a schema-4 fixture** -- [part 1](records/PLAN-ARCHIVE-1.md#task-44)
- [x] 45. **Closed-group emission** -- [part 1](records/PLAN-ARCHIVE-1.md#task-45)
- [x] 46. **`deco`** -- [part 1](records/PLAN-ARCHIVE-1.md#task-46)
- [x] 47. **The ring-clearing signal** -- [part 1](records/PLAN-ARCHIVE-1.md#task-47)
- [x] 48. **Lagged co-moments on `ew_cov`** -- [part 1](records/PLAN-ARCHIVE-1.md#task-48)
- [x] 49. **`po.prep.refresh_time`** -- [part 1](records/PLAN-ARCHIVE-1.md#task-49)
- [x] 50. **`rcov`** -- [part 1](records/PLAN-ARCHIVE-1.md#task-50)
- [x] 51. **`po.corr`** -- [part 1](records/PLAN-ARCHIVE-1.md#task-51)
- [x] 52. **`po.sim.regimes`** -- [part 1](records/PLAN-ARCHIVE-1.md#task-52)
- [x] 53. **`hmm`** -- [part 1](records/PLAN-ARCHIVE-1.md#task-53)
- [x] 54. **`corrchange`** -- [part 1](records/PLAN-ARCHIVE-1.md#task-54)
- [x] 55. **`bocpd`** -- [part 1](records/PLAN-ARCHIVE-1.md#task-55)
- [x] 56. **Freeze a schema-5 fixture and close the batch** -- [part 1](records/PLAN-ARCHIVE-1.md#task-56)
- [x] 57. **Fix what the review of 45–56 found** -- [part 1](records/PLAN-ARCHIVE-1.md#task-57)
- [x] 58. **Documentation pass, 2026-09-06** -- [part 1](records/PLAN-ARCHIVE-1.md#task-58)
- [x] 59. **`llms.txt`, 2026-09-06** -- [part 1](records/PLAN-ARCHIVE-1.md#task-59)
- [x] 60. **The README links into the API reference, 2026-09-06** -- [part 1](records/PLAN-ARCHIVE-1.md#task-60)
- [x] 63. **`window`: an EW accumulator with a hard cutoff** -- [part 1](records/PLAN-ARCHIVE-1.md#task-63)
- [x] 64. **Document the output struct of every model** -- [part 1](records/PLAN-ARCHIVE-1.md#task-64)
- [x] 65. **`marginal(lags=)` and `n_serial`, 2026-09-07** -- [part 1](records/PLAN-ARCHIVE-1.md#task-65)
- [x] 66. **`marginal(bins=)` and `split_gain`, 2026-09-07** -- [part 1](records/PLAN-ARCHIVE-1.md#task-66)
- [x] 67. **Review of tasks 65 and 66, 2026-09-07** -- [part 1](records/PLAN-ARCHIVE-1.md#task-67)
- [x] 69. **The bank's read surface, 2026-09-07** -- [part 1](records/PLAN-ARCHIVE-1.md#task-69)
- [x] 70. **The leak test measured the ramp, not the plateau, 2026-09-08** -- [part 1](records/PLAN-ARCHIVE-1.md#task-70)
- [x] 73. **The clock does not have to be a time, and the headline docs did not say so, 2026-09-08** -- [part 1](records/PLAN-ARCHIVE-1.md#task-73)
- [x] 74. **`sgd`'s scaler standardises a row against moments that exclude it, and is wrong wherever there are few…** -- [part 1](records/PLAN-ARCHIVE-1.md#task-74)
- [x] 72. **Against `sklearn.linear_model.SGDRegressor` — analysis done 2026-09-08, measured and written up the same day** -- [part 1](records/PLAN-ARCHIVE-1.md#task-72)
- [x] 75. **`sgd`'s per-feature cost, 2026-09-08: 13–14 ns per feature per row to 2.2 at `k = 10,000`, 45,000…** -- [part 1](records/PLAN-ARCHIVE-1.md#task-75)
- [x] 76. **`hit_rate` reads 1.0 for every binary fit, and classification has no metric of its own — found 2026-09-08…** -- [part 1](records/PLAN-ARCHIVE-1.md#task-76)
- [ ] 86. **The bank on Arrow, with Polars as an adapter -- the bank is done,
      two pieces deliberately are not, 2026-09-17. Parked by the user on
      2026-09-25** with everything that integrates Arrow or a new library
      (the plan of 2026-09-25, its parking note); the import is not blocked
      (2026-09-22, below), it waits by choice. Goal: the model bank takes
      and returns Arrow, so Polars becomes the most convenient way to use the
      library rather than the only one, and the boundary stops riding on
      private py-polars methods. The first holds outright. The second holds on
      the way *out* and not on the way *in*, which is why this stays open
      rather than ticked. The input direction is unbuilt, not blocked
      (corrected 2026-09-22: see "not done" below), and waits by the user's
      choice, parked on 2026-09-25.
      **Not a batch:** four increments, each landed and proven on its own
      against the golden streams rather than as one change nothing can judge.
      Released in 0.7.0.

      *Done:*

      - **The output path.** Every output field is built as an Arrow array and
        `column.rs` imports no polars at all. `coef` is a hand-built
        `ListArray`, written out rather than delegated because a null list and
        a null inside a list are different things. `assemble` ends at
        `StructArray::new`, with one polars call left to give the struct the
        spec's name.
      - **The input path.** The bank reads an `ArrowChunk`, not a `DataFrame`,
        and `fit_predict_arrow` / `predict_arrow` are public and mention no
        polars. The split is by what a decision is *about*: every
        polars-shaped one -- column lookup, the numeric refusal, the `Float64`
        cast, the key and label cast to text, the integer-key choice, the
        temporal-clock refusal, the refusal of a group dtype `"monotone"`
        cannot order -- moved to `crate::arrow`; every model-shaped one stayed
        in the bank, because it holds whoever supplies the data. `column`,
        `f64_series`, `key_column` and `materialized` are gone.

      - **The return type.** `assemble` returns a `StructArray`, so
        `fit_predict_arrow` and `predict_arrow` hand back one struct array per
        spec and the polars pair is those two plus `named_column`.
        `compare_targets` was the last place the bank read its *own* output
        through polars (`col.struct_()?.field_by_name(..)`); it now finds the
        field in the struct's own schema and downcasts the child.
      - **The output reaches Python as Arrow.** `ModelBank.fit_predict_arrow`
        returns `ArrowStruct` objects exposing `__arrow_c_array__`, which is
        public and standardised, against `PySeries`' private
        `_export`/`_import`. Proven equal to the polars path field for field,
        nulls and the nested `coef` list included, and the export-once
        contract holds. `export_struct_to_c` is the whole of the hand-off; a
        `Drop` that calls `release` only when the consumer has not taken it is
        what makes it safe, and polars-arrow already provides that.

      *Not done, and the reason -- both examined rather than skipped:*

      - **The input direction from Python.** A frame still arrives as
        `PyDataFrame`, so the private interface is off half the boundary and
        not all of it. **Still unbuilt, but no longer blocked -- corrected
        2026-09-22.** This bullet used to say the obstacle was ownership: the C
        data interface makes the *consumer* take ownership by moving the struct
        out and nulling the producer's `release` pointer, and polars-arrow keeps
        `ArrowArray`'s fields `pub(super)`, so the original could not be marked
        released. **That was wrong, and it had been recorded as a hard blocker
        for a fortnight.** Nulling `release` needs no field access:
        `ArrowArray::empty()` is `pub` and builds the struct with
        `release: None`, so `std::ptr::replace(ptr, ArrowArray::empty())` moves
        the producer's struct out and marks the original released in one public
        call. polars' own `import_array_pycapsules`
        (`crates/polars-python/src/series/import.rs`) is exactly this idiom, and
        `crates/online-polars/tests/arrow_capsule_import.rs` proves it against
        the pinned `polars-arrow =0.55.2` the wheel ships. The pieces are
        otherwise as before: `import_array_from_c` takes the struct by value and
        wants the dtype separately, from `import_field_from_c`;
        `ArrowArrayStreamReader::try_new` is the stream-shaped alternative; the
        consumer side is `PyCapsuleMethods::pointer_checked(Some(c"arrow_array"))`.
        What remains is to build it. See `docs/ARROW-SOURCES.md` §4, which no
        longer asks for a decision.
      - **The four accessor frame builders.** Judged after reading all four,
        not assumed: the rewrite is not worth it. `summary_frame` (13 columns)
        and `describe_frame` (9) are flat and would convert easily. But
        `closed_frame` is some forty columns across five conditional blocks and
        leans on three separate polars list-builder families --
        `ListPrimitiveChunkedBuilder` for `f64` and `i64`,
        `ListStringChunkedBuilder`, and `AnonymousOwnedListBuilder`, which
        materialises a `Series` per row to build a doubly nested
        `List(List(Float64))`. By hand that is offsets at two levels and three
        validity bitmaps for every such column, and `marginal` has ragged
        inner lists besides. What it would buy does not match it: these calls
        exist to hand back a *table*, which is the one thing polars' builders
        are for, and they run where there is no second copy of polars in the
        process. The boundary that motivated the task is the chunk path, and
        that path is Arrow end to end.

      *What the session of 2026-09-17 established, so it need not be redone:*

      - **The bank's Polars use is thin and mechanical.** Input was one cast to
        `Float64`, a borrowed slice when a column is one
        null-free chunk, and a NaN-filling fallback already written against
        Arrow arrays. Output is `F64Column`: a values buffer plus validity bits
        packed little-endian, which is Arrow's own layout. What the bank asks of
        a data layer is a contiguous float slice, a validity bitmap, a struct of
        named children and a schema. Arrow provides all four.
      - **The interchange was tested, not assumed.** Decimal, Boolean and
        `Int16` inputs are accepted and produce the right struct. A full output
        -- struct with a nested `List(Float64)` and five nulls -- round-trips
        through `__arrow_c_stream__` / `pl.from_arrow` with exact frame
        equality, preserved dtypes and preserved null counts. That was the
        sharpest risk and it passed.
      - **The motivation is the private boundary, not size.** pyo3-polars gets
        `_s` off a Series and calls the private `_export`/`_import`
        (`types.rs:181,250`), which is why the floor is 1.28.1 and why the
        interface carries no promise. `__arrow_c_stream__` is public and
        standardised; the pinned polars 1.44.2 already exposes it on both
        frames and series.
      - **Arrow does not reach the engine, and no longer needs to.** Streaming
        scans, pushdown, multi-file globs, CSV inference and the parallel sinks
        have no Arrow equivalent -- but that is the runner, which after task 83
        is the command line's alone, where static linking costs nothing because
        there is no second copy in the process.
      - **What it would cost:** reimplementing the input cast (arrow-cast, or by
        hand for the numeric types we accept); multi-chunk handling, which
        `ChunkedArray` currently gives free; and returning record batches from
        `summary`, `describe`, `coef`, `marginal` and `closed_groups` for Python
        to wrap. Prototype the cast kernel and the chunk handling first -- that
        is where the work actually is.
      - **What it would open:** any Arrow producer as a source (DuckDB readers,
        pyarrow datasets, Iceberg and Delta, Arrow Flight); Polars optional
        rather than required, which needs the *Python* side restructured too,
        since ten modules import it at module scope including `__init__.py`;
        and one Polars in the process instead of two for the bank path.

- [x] 85. **The expression plugin removed, 2026-09-17** -- [part 1](records/PLAN-ARCHIVE-1.md#task-85)
- [x] 84. **The bank takes a plan, and learns without keeping the output, 2026-09-17** -- [part 1](records/PLAN-ARCHIVE-1.md#task-84)
- [x] 83. **The runner leaves Python; the `online` command line keeps it, 2026-09-17** -- [part 1](records/PLAN-ARCHIVE-1.md#task-83)
- [x] 82. **N9: `quantile` did not settle on the quantile regression -- found 2026-09-15 writing the code review's…** -- [part 1](records/PLAN-ARCHIVE-1.md#task-82)
- [x] 81. **`target_gaps`: which rows a target's fit is read from -- the code review's N3, decided 2026-09-14** -- [part 1](records/PLAN-ARCHIVE-1.md#task-81)
- [x] 80. **The code review of 2026-09-12, worked through, begun 2026-09-13** -- [part 1](records/PLAN-ARCHIVE-1.md#task-80)
- [x] 79. **`label_delay` ignores a clock event on a skipped row — found 2026-09-11, checking task 78's parity; in…** -- [part 1](records/PLAN-ARCHIVE-1.md#task-79)
- [x] 78. **Windowed EWMAs in both directions, as columns -- built 2026-09-30 (`34af36f`)** -- [part 1](records/PLAN-ARCHIVE-1.md#task-78)
- [x] 77. **A plan with nothing to write runs once where a query uses it twice, 2026-09-10** -- [part 1](records/PLAN-ARCHIVE-1.md#task-77)
- [x] 71. **E51, the blocked rank-B Gram update — design settled 2026-09-08, `EwCov` half built and reviewed the same…** -- [part 1](records/PLAN-ARCHIVE-1.md#task-71)
- [x] 68. **README clarity pass, begun 2026-09-07, done 2026-09-09** -- [part 1](records/PLAN-ARCHIVE-1.md#task-68)
- [x] 61. **The leak test's statistic, 2026-09-06** -- [part 1](records/PLAN-ARCHIVE-1.md#task-61)
- [x] 62. **The release gate, 2026-09-06** -- [part 1](records/PLAN-ARCHIVE-1.md#task-62)
- [x] 87. **Not using a model before it is ready, 2026-09-21 (`docs/WARMUP-AND-CONVERGENCE.md`)** -- [part 1](records/PLAN-ARCHIVE-1.md#task-87)
- [x] 88. **Durations for a time clock: `timedelta` half-lives on a `Datetime` clock, requested 2026-09-23** -- [part 1](records/PLAN-ARCHIVE-1.md#task-88)
- [x] 89. **README rewrite: ten sections in a hierarchy, requested 2026-09-23** -- [part 1](records/PLAN-ARCHIVE-1.md#task-89)
- [x] 90. **Every user document rewritten to `docs/WRITING.md`, requested 2026-09-23** -- [part 1](records/PLAN-ARCHIVE-1.md#task-90)
- [x] 91. **Every significant Python, to the newest, requested 2026-09-24** -- [part 1](records/PLAN-ARCHIVE-1.md#task-91)
- [x] 92. **Mutation testing in CI, a release comparison, property tests and oracles, requested 2026-09-24** -- [part 1](records/PLAN-ARCHIVE-1.md#task-92)
- [x] 93. **The PyPI page's links to other files, reported 2026-09-24** -- [part 1](records/PLAN-ARCHIVE-1.md#task-93)
- [x] 94. **A windowed Gram's zero variance comes back as rounding noise** -- [part 1](records/PLAN-ARCHIVE-1.md#task-94)
- [x] 95. **The lasso's `lam_selected` under a `window` departs from its documentation** -- [part 1](records/PLAN-ARCHIVE-1.md#task-95)
- [x] 96. **`lam_selected` under a `min_periods` list** -- [part 1](records/PLAN-ARCHIVE-1.md#task-96)
- [x] 97. **Two conventions to settle** -- [part 1](records/PLAN-ARCHIVE-1.md#task-97)
- [x] 98. **The Rust tests link no Python, 2026-09-24** -- [part 1](records/PLAN-ARCHIVE-1.md#task-98)
- [x] 99. **pyarrow, tested as a reader and as a source, 2026-09-24** -- [part 1](records/PLAN-ARCHIVE-1.md#task-99)
- [x] 100. **Libraries the package does not depend on, allowed in tests, 2026-09-24** -- [part 1](records/PLAN-ARCHIVE-1.md#task-100)
- [x] 101. **A feature that stops moving leaves a rounding artifact outside a window, and the standardizing models read it** -- [part 1](records/PLAN-ARCHIVE-1.md#task-101)
- [x] 102. **A stopped feature in the clusters' metric** -- [part 1](records/PLAN-ARCHIVE-1.md#task-102)
- [x] 103. **`corrchange` takes a span's variance as `E[x²] − E[x]²`** -- [part 1](records/PLAN-ARCHIVE-1.md#task-103)
- [x] 104. **Window expressions as model targets, under an embargo -- built 2026-10-03** -- [part 2](records/PLAN-ARCHIVE-2.md#task-104)
- [x] 105. **`po.prep` renamed `po.stream`, and the rules every function in it follows — split from task 78 on…** -- [part 2](records/PLAN-ARCHIVE-2.md#task-105)
- [x] 106. **Discarded by the user on 2026-09-25: a backward clock is not a session change** -- [part 2](records/PLAN-ARCHIVE-2.md#task-106)
- [x] 107. **Relative targets -- built 2026-09-25 for plain columns; windows take it as arithmetic in task 143** -- [part 2](records/PLAN-ARCHIVE-2.md#task-107)
- [x] 108. **The docs say what the code does** -- [part 2](records/PLAN-ARCHIVE-2.md#task-108)
- [x] 109. **Release 0.11.1** -- [part 2](records/PLAN-ARCHIVE-2.md#task-109)
- [x] 110. **This week's review leftovers, as tests** -- [part 2](records/PLAN-ARCHIVE-2.md#task-110)
- [x] 111. **The model contract, checked where it is only listed** -- [part 2](records/PLAN-ARCHIVE-2.md#task-111)
- [x] 112. **Older review items that need no decision** -- [part 2](records/PLAN-ARCHIVE-2.md#task-112)
- [x] 113. **CI and tooling** -- [part 2](records/PLAN-ARCHIVE-2.md#task-113)
- [x] 114. **Four formulas checked against their papers** -- [part 2](records/PLAN-ARCHIVE-2.md#task-114)
- [x] 115. **Decisions on behaviour that is built** -- [part 2](records/PLAN-ARCHIVE-2.md#task-115)
- [x] 116. **Readiness beyond `ewridge`** -- [part 2](records/PLAN-ARCHIVE-2.md#task-116)
- [ ] 117. **Python versions.** S–L. A 3.15 CI leg when it ships (October;
      recommended); free-threaded builds (non-`abi3` wheels; not yet); a
      floor below 3.12 (no). **Measured 2026-09-29 on 3.15.0rc2** (the user
      asked for a leg now; recommended against until 3.15.0 ships): the
      abi3 wheel installs and a bank fits and predicts, and the suite ran
      2,083 passed, 1,508 failed, all but five outside this package. On
      3.15.0rc2, polars 1.44.2's Series methods that dispatch through an
      expression -- `struct.field`, `fill_nan`, `is_null`, `abs` -- return
      `None`, and a datetime Series cannot be built; our own `bank.coef()`
      and `po.sim` call two of them, so they fail with it. Four test
      libraries have no cp315 wheel -- pyarrow, duckdb, the ADBC driver
      manager, vowpalwabbit -- and their sources need C++ toolchains, so
      `uv sync` would fail on the leg. Every failure traced goes back to
      one of those two (one `seqtest` test was not traced alone). The five
      were one bug of ours on every Python: `to_json` refused a bank with
      a column that never held a value; fixed. Re-run when polars and
      those four ship 3.15 support, then add the leg and the classifier
      together. **Re-measured 2026-10-10 on 3.15.0** (released 2026-10-09):
      polars 2.0.0 carries the fixes (#29489, #29491) and every probe that
      failed on 1.44.2 passes; every polars 1.x still fails on 3.15.0
      (1.34.0 and 1.44.2 return the same `None`s), so the classifier will
      need `polars>=2.0.0` on 3.15 under a `python_version` marker (the
      user's call when it lands; recommended over raising the floor
      everywhere). pyarrow 26.0.0 and pandas 3.0.6 ship cp315 wheels (the
      lock's 25.0.1 cannot build without Arrow C++; a lock refresh fixes
      it); duckdb (1.5.6, 2.0.0.dev), adbc-driver-manager 1.12.0 and
      vowpalwabbit 9.11.9 still have none. Built from source on macOS they
      install and work, and with them the whole suite passes on 3.15.0:
      6,003 passed, 2 skipped (the Windows-spelling guard); without them 29
      fail, each on the missing module. The leg waits on those three
      wheels; then refresh the lock, add the marker, the leg and the
      classifier together. Meanwhile CI's `lint` and `benchmark` name
      `UV_PYTHON: "3.12"`: uv 0.13 downloads 3.15 where it finds no
      interpreter, which the locked pyarrow cannot install on.

- [ ] 118. **Ideas waiting on a need** — kept so they are not lost; none is
      to be built without one, *each the user's call*: `log_loss` in
      `eval.sums`; the data summary stored column-wise, and an opt-out;
      `gram_block_rows` on `lasso`; an O(k) simplex projection; square-root-
      free Givens for `rls`; `sgd`'s iterate average; cohesion-gated
      re-placement in `kmeans`; boosted trees (after the real-data trial of
      BOOSTED-TREES §8); in clustering a second pass, DBSCAN over a retained
      sample, k chosen by EW SSQ, clusterwise regression, the prototyped
      clusterers; BEYOND-O-STATE's B3 (frequent directions), B4 (a
      fixed-lag smoother) and B5 (multi-lag residual checks); state as data
      and `save_state=callable`; E63's `weight_from`; a clock per window
      operator, Polars' `by=`, analysed and costed in task 143 (2026-10-02).
      (`label_delay`'s buffer bound moved to task 104; the docstring pass to
      task 148.)

- [x] 119. **Housekeeping** -- [part 2](records/PLAN-ARCHIVE-2.md#task-119)
- [x] 120. **Rows out of order and clocks that go back: what an audit of 2026-09-25 found. Needs extra review before…** -- [part 2](records/PLAN-ARCHIVE-2.md#task-120)
- [x] 121. **Test libraries under an open-source licence** -- [part 2](records/PLAN-ARCHIVE-2.md#task-121)
- [x] 122. **E71: bin each feature once per row** -- [part 2](records/PLAN-ARCHIVE-2.md#task-122)
- [x] 123. **E70: `marginal(cross_lags=...)`** -- [part 2](records/PLAN-ARCHIVE-2.md#task-123)
- [x] 124. **The stream's fixed cost per feature per row at width** -- [part 2](records/PLAN-ARCHIVE-2.md#task-124)
- [x] 125. **E72: `feature_moments="shared"` — a decision first** -- [part 2](records/PLAN-ARCHIVE-2.md#task-125)
- [x] 126. **E73: a wide `marginal` sharded across the pool (`shards`)** -- [part 2](records/PLAN-ARCHIVE-2.md#task-126)
- [x] 127. **E74: a chunk run pair-major — not as specified** -- [part 2](records/PLAN-ARCHIVE-2.md#task-127)
- [x] 128. **`EwCov` keeps its runs without a window** -- [part 2](records/PLAN-ARCHIVE-2.md#task-128)
- [x] 129. **`marginal`'s bins budget counts three values a cell where there are four** -- [part 2](records/PLAN-ARCHIVE-2.md#task-129)
- [x] 130. **Two window snapshots undercount what they hold** -- [part 2](records/PLAN-ARCHIVE-2.md#task-130)
- [x] 131. **`marginal(bin_budget=...)`: the bins' memory limit, set per spec** -- [part 2](records/PLAN-ARCHIVE-2.md#task-131)
- [x] 132. **Review 2026-09-26 of the unpushed changes** -- [part 2](records/PLAN-ARCHIVE-2.md#task-132)
- [x] 133. **The release workflow tags what it published, after everything ran** -- [part 2](records/PLAN-ARCHIVE-2.md#task-133)
- [x] 134. **E75: `marginal`'s lead/follow reading does not hold against a forward-looking target — correct the docs…** -- [part 2](records/PLAN-ARCHIVE-2.md#task-134)
- [x] 135. **S18: Bartlett weights for `marginal`'s serial rule** -- [part 2](records/PLAN-ARCHIVE-2.md#task-135)
- [x] 136. **S19: the Gram's target moments under a `window`** -- [part 2](records/PLAN-ARCHIVE-2.md#task-136)
- [x] 137. **C18: a lag ring under a `window` in `marginal`** -- [part 2](records/PLAN-ARCHIVE-2.md#task-137)
- [x] 138. **README rewrite against the current API, requested 2026-09-29** -- [part 2](records/PLAN-ARCHIVE-2.md#task-138)
- [x] 139. **`ModelBank.fit`'s order exemption, narrowed to `ewridge` and `rls`** -- [part 2](records/PLAN-ARCHIVE-2.md#task-139)
- [x] 140. **Task 87's inverse diagonal, on every solve** -- [part 2](records/PLAN-ARCHIVE-2.md#task-140)
- [x] 141. **Solve-every-row speed** -- [part 2](records/PLAN-ARCHIVE-2.md#task-141)
- [x] 142. **NumPy's newest and next versions, tested as Polars's are** -- [part 2](records/PLAN-ARCHIVE-2.md#task-142)
- [x] 143. **Window expressions: factors and targets as Polars formulas over exponentially weighted operators -- the…** -- [part 2](records/PLAN-ARCHIVE-2.md#task-143)
- [x] 144. **Window expressions take Polars' names and semantics -- the names built 2026-10-02 (the two tables below…** -- [part 2](records/PLAN-ARCHIVE-2.md#task-144)
- [x] 145. **`session_shrink` mixes data, not weight -- built 2026-10-02** -- [part 2](records/PLAN-ARCHIVE-2.md#task-145)
- [x] 146. **A clock parameter acts on the clock -- built 2026-10-02, but for `kalman`'s `coef_halflife`, which is task 150** -- [part 2](records/PLAN-ARCHIVE-2.md#task-146)
- [x] 150. **`kalman`'s `coef_half_life` on the clock -- built 2026-10-03, the recommended option** -- [part 2](records/PLAN-ARCHIVE-2.md#task-150)
- [x] 147. **A row weight scales evidence, not counts -- built 2026-10-02, but for `ftrl` and the density half of…** -- [part 2](records/PLAN-ARCHIVE-2.md#task-147)
- [x] 151. **`ftrl` and `micro` keep their units: a sum-scale prior and a point density -- decided and the docs written…** -- [part 2](records/PLAN-ARCHIVE-2.md#task-151)
- [x] 152. **The clock a row was scored at, and the clock of the last row learned -- built 2026-10-02** -- [part 2](records/PLAN-ARCHIVE-2.md#task-152)
- [x] 153. **`label_delay` learned labels early at a break -- built 2026-10-02** -- [part 2](records/PLAN-ARCHIVE-2.md#task-153)
- [x] 148. **The docs say what the code does: the 2026-10-01 findings -- built 2026-10-02** -- [part 2](records/PLAN-ARCHIVE-2.md#task-148)
- [x] 149. **A docstring pass under `docs/WRITING.md` -- done (2026-10-03)** -- [part 2](records/PLAN-ARCHIVE-2.md#task-149)
- [x] 154. **The reader-facing documents rewritten after tasks 139 to 153, 104 and the review rounds -- requested…** -- [part 2](records/PLAN-ARCHIVE-2.md#task-154)
- [x] 155. **The weekly mutation pass sized to finish -- requested 2026-10-03** -- [part 3](records/PLAN-ARCHIVE-3.md#task-155)
- [x] 156. **The README rewritten by the ideas of two reviews -- requested 2026-10-04** -- [part 3](records/PLAN-ARCHIVE-3.md#task-156)
- [x] 157. **The README rewritten as a test of every rule -- requested 2026-10-04** -- [part 3](records/PLAN-ARCHIVE-3.md#task-157)
- [x] 158. **The mutation survivors worked through, and the bug behind the scheduled pass's failure fixed -- requested…** -- [part 3](records/PLAN-ARCHIVE-3.md#task-158)
- [x] 159. **The code review of everything since `v0.13.0`, every finding fixed -- requested 2026-10-05** -- [part 3](records/PLAN-ARCHIVE-3.md#task-159)
- [x] 160. **The whole-project review, every finding with a clear fix fixed -- requested 2026-10-05** -- [part 3](records/PLAN-ARCHIVE-3.md#task-160)
- [x] 161. **`ew_cov`'s PCA refreshes on the clock, as the regressions' solves do -- requested 2026-10-06** -- [part 3](records/PLAN-ARCHIVE-3.md#task-161)
- [x] 162. **A model window's snapshots are spaced on the clock, as the regressions' solves are scheduled -- requested…** -- [part 3](records/PLAN-ARCHIVE-3.md#task-162)
- [x] 163. **`micro` prunes on the clock, as the regressions' solves are scheduled -- requested 2026-10-06** -- [part 3](records/PLAN-ARCHIVE-3.md#task-163)
- [x] 164. **The parameters that count rows where a clock would fit, reviewed -- requested 2026-10-06** -- [part 3](records/PLAN-ARCHIVE-3.md#task-164)
- [x] 165. **The user's decisions on §17** -- [part 3](records/PLAN-ARCHIVE-3.md#task-165)
- [x] 166. **The mutation and canary failures examined -- requested 2026-10-06** -- [part 3](records/PLAN-ARCHIVE-3.md#task-166)
- [x] 167. **No `.lazy()` in the examples: each query reads a file -- requested 2026-10-06** -- [part 3](records/PLAN-ARCHIVE-3.md#task-167)
- [x] 168. **`drift_threshold` is a clock parameter, required with a clock -- requested 2026-10-06** -- [part 3](records/PLAN-ARCHIVE-3.md#task-168)
- [x] 171. **`rcov`'s pre-averaged estimate carries CKP's finite-sample rescaling -- requested 2026-10-06** -- [part 3](records/PLAN-ARCHIVE-3.md#task-171)
- [x] 169. **The Rust test oracles in one place, independent of `solve.rs` -- requested 2026-10-06** -- [part 3](records/PLAN-ARCHIVE-3.md#task-169)
- [x] 173. **Forward windows: a silent group needs a clock (PC1), and a reset keeps a complete window whole (PC2) --…** -- [part 3](records/PLAN-ARCHIVE-3.md#task-173)
- [x] 174. **`lasso` counts each target's selection errors from its own `min_weight` -- requested 2026-10-06** -- [part 3](records/PLAN-ARCHIVE-3.md#task-174)
- [x] 175. **A model window's edge on the decayed clock, held exactly -- requested 2026-10-06** -- [part 3](records/PLAN-ARCHIVE-3.md#task-175)
- [x] 172. **`kalman`'s warm-up is unit-free: the noise from the first innovation, the prior `p0` times it -- requested…** -- [part 3](records/PLAN-ARCHIVE-3.md#task-172)
- [x] 170. **Cholesky factors move in place, in `solve.rs`; `quantile` takes back a sixth of TC1b's cost, bit for bit…** -- [part 3](records/PLAN-ARCHIVE-3.md#task-170)
- [x] 176. **`embargo` counts down in doubles, so an embargo of exactly k steps can release a row a row late -- found…** -- [part 3](records/PLAN-ARCHIVE-3.md#task-176)
- [x] 178. **`coef_every` counts the clock, as `solve_every` does -- requested 2026-10-06** -- [part 3](records/PLAN-ARCHIVE-3.md#task-178)
- [x] 180. **`solve_every`, `pca_every` and `prune_every` sum their clock in doubles -- found 2026-10-06 by task 178** -- [part 3](records/PLAN-ARCHIVE-3.md#task-180)
- [x] 177. **`huber` and `quantile` down-weight nothing until a residual scale exists -- requested 2026-10-06** -- [part 3](records/PLAN-ARCHIVE-3.md#task-177)
- [x] 179. **`bocpd`'s hazard takes a duration, applied before the row it leads into -- requested 2026-10-06** -- [part 3](records/PLAN-ARCHIVE-3.md#task-179)
- [x] 181. **A quadratic form of a NaN is NaN, and every caller refuses the row -- requested 2026-10-06** -- [part 3](records/PLAN-ARCHIVE-3.md#task-181)
- [x] 182. **Four older NaN gaps, outside the core's stated contract (every input finite) -- found 2026-10-06 by task 181** -- [part 3](records/PLAN-ARCHIVE-3.md#task-182)
- [x] 183. **`ew_cov`, `sgd`, `ewridge` and `lasso` still learn a row with a non-finite feature -- found 2026-10-06 by…** -- [part 3](records/PLAN-ARCHIVE-3.md#task-183)
- [x] 184. **The survivors of the mutation run over `f49058d..8e28c1f`** -- [part 3](records/PLAN-ARCHIVE-3.md#task-184)
- [x] 185. **A loader refuses a damaged state by name** -- [part 3](records/PLAN-ARCHIVE-3.md#task-185)
- [x] 186. **Round-4 model defects** -- [part 3](records/PLAN-ARCHIVE-3.md#task-186)
- [x] 187. **Round-4 defects in the bank, the stream and the surfaces** -- [part 3](records/PLAN-ARCHIVE-3.md#task-187)
- [x] 188. **Round-4 defects in the helper modules and the window operators** -- [part 3](records/PLAN-ARCHIVE-3.md#task-188)
- [x] 189. **Round-4 test findings** -- [part 3](records/PLAN-ARCHIVE-3.md#task-189)
- [x] 190. **The API snapshot pins what the policy calls stable** -- [part 3](records/PLAN-ARCHIVE-3.md#task-190)
- [x] 191. **Round-4 fixes to the reader-facing documents and the release workflow** -- [part 3](records/PLAN-ARCHIVE-3.md#task-191)
- [x] 192. **The dated records name their dated spellings** -- [part 3](records/PLAN-ARCHIVE-3.md#task-192)
- [x] 193. **Round-4 input validation and the words of a refusal** -- [part 3](records/PLAN-ARCHIVE-3.md#task-193)
- [x] 194. **The bank's and the stream's state and frames** -- [part 3](records/PLAN-ARCHIVE-3.md#task-194)
- [x] 195. **The models' defaults and semantics** -- [part 3](records/PLAN-ARCHIVE-3.md#task-195)
- [x] 196. **Names in the specs, the models and the command line** -- [part 3](records/PLAN-ARCHIVE-3.md#task-196)
- [x] 197. **Names in the helper modules** -- [part 3](records/PLAN-ARCHIVE-3.md#task-197)
- [x] 198. **The mechanisms of the 1.0 promise** -- [part 3](records/PLAN-ARCHIVE-3.md#task-198)
- [x] 199. **The 1.0 policy text and the process** -- [part 3](records/PLAN-ARCHIVE-3.md#task-199)
- [x] 200. **An integer clock column is held as an integer** -- [part 3](records/PLAN-ARCHIVE-3.md#task-200)
- [x] 201. **Relative targets are removed** -- [part 3](records/PLAN-ARCHIVE-3.md#task-201)
- [x] 202. **An insensitivity band in units of the target's own spread** -- [part 3](records/PLAN-ARCHIVE-3.md#task-202)
- [x] 203. **An insensitivity band defaults to 1% of the target's spread** -- [part 3](records/PLAN-ARCHIVE-3.md#task-203)
- [x] 204. **`kalman`'s `share_p` takes each row once** -- [part 3](records/PLAN-ARCHIVE-3.md#task-204)
- [x] 205. **Review round 5: the changes since round 4** -- [part 3](records/PLAN-ARCHIVE-3.md#task-205)
- [x] 206. **A standardized fit is held in the caller's units after a warm-up; the scaler shapes the step** -- [part 3](records/PLAN-ARCHIVE-3.md#task-206)
- [x] 207. **The insensitivity band and `c` in noise units, measured** -- [part 3](records/PLAN-ARCHIVE-3.md#task-207)
- [x] 208. **The smaller round-5 decisions** -- [part 3](records/PLAN-ARCHIVE-3.md#task-208)
- [x] 209. **Every model is tested on positive, negative and mixed values, and on a level that crosses zero** -- [part 3](records/PLAN-ARCHIVE-3.md#task-209)
- [x] 210. **Two tiers of tests: essentials on every commit's gate, the rest before every push, in CI and before a release** -- [part 3](records/PLAN-ARCHIVE-3.md#task-210)
- [x] 211. **Kalman, from its review of 2026-10-08** -- [part 3](records/PLAN-ARCHIVE-3.md#task-211)
- [x] 212. **The validation run, the per-standard-deviation recipe, and `po.ewm_std`** -- [part 3](records/PLAN-ARCHIVE-3.md#task-212)
- [x] 213. **A relative `c` for PA-I, with its two costs fixed -- measured, not built** -- [part 3](records/PLAN-ARCHIVE-3.md#task-213)
- [x] 214. **Hard rule 9 in `ftrl`, `hmm` and `ewridge`'s clock cadence, and kalman's two edge cases** -- [part 3](records/PLAN-ARCHIVE-3.md#task-214)
- [x] 215. **The obvious fixes from tasks 209 and 214** -- [part 3](records/PLAN-ARCHIVE-3.md#task-215)
- [x] 216. **The measured fixes from task 209** -- [part 3](records/PLAN-ARCHIVE-3.md#task-216)
- [x] 217. **The level-aware truncation bound (D4), a sparse target's subnormal history, `rls` windup documented, and…** -- [part 3](records/PLAN-ARCHIVE-3.md#task-217)
- [x] 218. **The 105 mutants that survived the push of `103d721`** -- [part 3](records/PLAN-ARCHIVE-3.md#task-218)
- [x] 219. **The changed-lines mutation pass sized to the push** -- [part 3](records/PLAN-ARCHIVE-3.md#task-219)
- [x] 220. **The 698 mutants the push of `103d721` left untested** -- [part 3](records/PLAN-ARCHIVE-3.md#task-220)
- [x] 221. **Online diagnostics: the questions today's set does not answer** -- [part 3](records/PLAN-ARCHIVE-3.md#task-221)
- [x] 222. **A diagnostics section in the docs** -- [part 3](records/PLAN-ARCHIVE-3.md#task-222)
- [x] 223. **Telling a user their data has a problem, after one pass** -- [part 3](records/PLAN-ARCHIVE-3.md#task-223)
- [x] 224. **A section in the docs on detecting data issues** -- [part 3](records/PLAN-ARCHIVE-3.md#task-224)
- [x] 225. **A multithreaded product for one wide Gram** -- [part 3](records/PLAN-ARCHIVE-3.md#task-225)
- [x] 226. **The lasso path from a Gram, in Rust** -- [part 3](records/PLAN-ARCHIVE-3.md#task-226)
- [x] 227. **Merge and solve many subsets of Grams in one call** -- [part 3](records/PLAN-ARCHIVE-3.md#task-227)
- [x] 228. **Missing values per column in a Gram** -- [part 3](records/PLAN-ARCHIVE-3.md#task-228)
- [x] 229. **A compact Gram** -- [part 3](records/PLAN-ARCHIVE-3.md#task-229)
- [ ] 230. **Online-EM imputation for missing features -- an idea, the
      user's call** (from task 228's research, 2026-10-08). Impute
      `E[x_M | x_O] = μ_M − Λ_MM⁻¹ Λ_MO (x_O − μ_O)` from the Gram's own
      precision matrix `Λ`, refreshed on a schedule, and add `Λ_MM⁻¹` to the
      missing block (Cappé and Moulines 2009): a plain Gram of imputed
      rows, PSD by construction, unbiased under MAR in the limit. A Python
      replica measured risk 1.0-1.2 at p 200 (unguarded pairwise 885) but a
      warm-up bias of 0.5-3.9 sd; `Λ` in the state is `k²` more, and the
      refresh a cadence to choose. Next step if wanted: a measured Rust
      replica (warm-up bias, cadence, cost at k 1,000-2,000) before any
      design.
- [x] 231. **A wide fit's bits whatever the thread count** -- [part 3](records/PLAN-ARCHIVE-3.md#task-231)
- [x] 234. **The mutants that survived the push of `5cb8f7e`** -- [part 3](records/PLAN-ARCHIVE-3.md#task-234)
- [x] 235. **Research: an adapter for faster LightGBM and XGBoost training from Polars/Arrow chunks** -- [part 3](records/PLAN-ARCHIVE-3.md#task-235)
- [x] 236. **LARS on exact ties -- the user's call** -- [part 3](records/PLAN-ARCHIVE-3.md#task-236)
- [x] 237. **`docs/GRAM.md`: wide fits from Grams** -- [part 3](records/PLAN-ARCHIVE-3.md#task-237)
- [x] 238. **A rewrite of the docs** -- [part 3](records/PLAN-ARCHIVE-3.md#task-238)
- [x] 239. **A review of every API doc** -- [part 3](records/PLAN-ARCHIVE-3.md#task-239)
- [x] 240. **Pooling Grams within parts: block fixed effects** -- [part 3](records/PLAN-ARCHIVE-3.md#task-240)
- [x] 241. **A sign per part on the target, under within-pooling** -- [part 3](records/PLAN-ARCHIVE-3.md#task-241)
- [x] 242. **The GBDT input component** -- [part 3](records/PLAN-ARCHIVE-3.md#task-242)
- [x] 243. **`bocpd`'s fold under heavy pruning** -- [part 3](records/PLAN-ARCHIVE-3.md#task-243)
- [x] 244. **The API standardized before 1.0** -- [part 3](records/PLAN-ARCHIVE-3.md#task-244)
- [x] 245. **Faster dev iterations** -- [part 3](records/PLAN-ARCHIVE-3.md#task-245)
- [x] 246. **The mutants run's headroom** -- [part 3](records/PLAN-ARCHIVE-3.md#task-246)
- [x] 247. **Release 1.0** -- [part 3](records/PLAN-ARCHIVE-3.md#task-247)
- [x] 249. **Step comments in the examples (before 1.0)** -- [part 3](records/PLAN-ARCHIVE-3.md#task-249)
- [x] 250. **Performance before 1.0** -- [part 3](records/PLAN-ARCHIVE-3.md#task-250)
- [x] 251. **Performance and memory budgets that only tighten** -- [part 4](records/PLAN-ARCHIVE-4.md#task-251)
- [ ] 252. **No number changes without a test noticing** -- the user,
      2026-10-11: "If we know of areas where numbers can change without a
      test noticing we should add to the plan to improve that", then "Add
      these as tentative items to the plan as there may be a reason that we
      need a tolerance now. Untested paths are always a concern and should
      get a test." Known gaps: most tests compare within a tolerance, so a
      change that moves the last bits passes them; the golden pipeline pins
      a few fields at three rows; the one complete bit-level check
      (`compare_release.py`) runs on release day over 30 specs and 400
      rows; cargo-mutants sees only online-core and `span.rs`. Areas task
      250 named as able to change numbers silently: the exact clock path,
      `bocpd`'s run structure (pruning and the fold), kmeans/micro's
      compensated sums.
      **Firm: every untested path gets a test.** An inventory of the paths
      no test reaches (coverage of the Rust crates and the package on the
      full tier, and the branches the hard-path cases above need: the fold,
      a compensation carrying a non-zero low part, a clock crossing a
      session, a temporal clock at a decade's age, a state saved and loaded
      mid-stream, zero-weight and null first rows), each closed by a test
      that proves the path ran -- a pin on an unexercised path is no pin.
      **Tentative (each starts from why the tolerance is there):**
      (a) a committed bit digest of every output field of every spec on a
      workload that reaches those paths, the digest changing with the
      CHANGELOG entry that declares it -- but a field whose arithmetic
      passes through libm (`exp`/`ln`: decay over irregular steps, a
      half-life read) differs in its last bits between macOS, Linux and
      Windows (a `to_bits` pin failed every Linux leg in task 202), so such
      a field is either left out, held per platform, or held to a tolerance,
      decided field by field; (b) the tolerance-based tests in the three
      areas above converted to bit pins only where no reason for the
      tolerance holds (libm, reassociation across thread counts, an oracle
      deliberately computed another way), each tolerance's reason recorded
      beside it; (c) the mutation blind spot: what extending cargo-mutants
      to online-polars would take, and whether a Python mutation tool earns
      its run time, reported.
      *Firm part done 2026-10-13*: the union of `cargo test --release
      --include-ignored` and the whole pytest suite on an instrumented
      release build (cargo-llvm-cov 0.9.0; the release profile for both, so
      the extension's and the tests' profiles map to the same objects) left
      1,583 Rust lines in 81 files unrun; pytest-cov with child
      interpreters traced left 157 package statements. Four workers (core,
      bank, windows, python) added tests for every reachable one: core
      350 -> 106 non-test lines, bank 358 -> 131, windows 190 -> 26, Python
      157 -> 14, the gbdt/gram bindings 182 -> 68; every line left is listed
      with its reason in the workers' reports (defensive arms behind
      validation, `unreachable!`, debug assertions, library failures after
      success, inlined one-liners shown to run by a probe). Found: the
      LightGBM Python route's unseen extra rows, `ModelBank(spec)`'s
      message, and a closed-groups CSV that failed mid-run for Gram models
      (string lists now written as JSON text). Measurement lesson: llvm-cov
      merging many test binaries reports "functions have mismatched data"
      and loses counts; per-binary profiles unioned are exact. The
      tentative parts (a)-(c) remain open.
      *(c) reported 2026-10-13*: cargo-mutants 27.1.0 lists 17,556 mutants
      in online-core (today's scope), 6,145 in online-polars (stream.rs
      927, bank.rs 856, windows.rs 764, spec.rs 714, windows_frame.rs 373,
      the rest under 230 a file), 2,015 in online-py and 34 in online-cli.
      Two local pilots (nextest, `--cargo-test-arg=--tests`, -j 2): a tenth
      of `summary.rs`, 23 mutants in 380 s, 22 caught and 1 unviable; a
      36th of `bank.rs`, 24 in 335 s, 14 caught, 7 unviable and 3 missed
      (`pair_f64` -> `Default::default()`, `&&` -> `||` in `closed_frame`,
      `+` -> `*` in `assemble`) -- 18% of the viable ones, against the Rust
      tests alone. About 19 s to build and 9 s to test a mutant, so all of
      online-polars is about 48 CPU-hours: some 30 minutes a shard over 96
      shards, inside CI's limits and free. **Recommendation, for the
      user**: add `crates/online-polars/src/**` to the changed-lines pass
      on push (today it takes only `span.rs`) and to the weekly pass, and
      treat each survivor as task 250's were -- a Rust test, or a recorded
      reason (some may be killed by pytest, which cargo-mutants cannot run:
      each such survivor gets a Rust test or a note). online-py (bindings
      checked from Python) stays out. A Python mutation tool over
      `python/polars_online` (4,635 statements) is not piloted; the pilot
      would be mutmut on one module (`_check.py`) to measure its run time
      and survivors before deciding.
      *(b) done 2026-10-13*: every tolerance in the three areas' tests --
      `clock.rs`, the bank's clock readers (`settled_frac`, the temporal and
      integer clocks), `bocpd.rs`, `comp.rs` and `cluster/`, Rust and Python
      -- has its reason written beside it, or was converted. Kept: libm in
      the path (`exp2`/`pow`/`ln` over irregular steps, a half-life read,
      `lgamma`), an oracle computed another way (longhand Algorithm 1,
      two-pass definitions, faer, pandas, Polars' rolling sum, scikit-learn,
      decimal closed forms), reassociation the code does not promise to
      keep (a batch merged as one mean, a merge against rows in turn,
      normalizers, weights scaled by 100), and rows rounded at a level of
      1e8. Converted to bits (33 assertions): every weight that is a count
      or a sum of exact weights, the fold's `n` and folded log joint, a
      decay by the model's own `Decay::factor` (the same libm call on both
      sides is the same bits on every platform), `window_w`, the buffer's
      cut, exact distances and radii, the micro centre and `eps2`, the bocpd
      hazard column against the spec's, `predict` against the step, and
      `settled_frac` at integer `T`; each was shown to fail on a one-ulp
      perturbation of the library that the old tolerance passed, but for
      two lines masked by an earlier line of the same mechanism (micro's
      second centre and radius) and micro's radius bound, which no summary
      on its stream comes within 6% of. Added bit
      pins beside kept exact-arithmetic oracles where the stream is
      libm-free (`comp.rs`'s pairs, a bocpd run's pair), and a temporal
      clock's weights held to the bit across a day, a year and a decade of
      age. No numeric defect found.
      Task 250's phase 2 does not wait on (a): its fixes are held by the
      golden pipeline, the fixtures, `compare_release.py` and
      `release_probe.py`'s columns and states, plus an activation test for
      each path a fix touches.
- [x] 253. **Thread count through the API (before 1.0; the user: "Go" on the design below, 2026-10-12)** -- [part 4](records/PLAN-ARCHIVE-4.md#task-253)
- [x] 259. **Release 1.0.1** -- [part 4](records/PLAN-ARCHIVE-4.md#task-259)
- [x] 258. **On x86, faer's matrix product commits about twice the L3 cache per thread** -- [part 4](records/PLAN-ARCHIVE-4.md#task-258)
- [x] 260. **faer's x86 kernel back on Linux and Intel macOS, off Windows only** -- [part 4](records/PLAN-ARCHIVE-4.md#task-260)
- [x] 263. **Why the bounded product costs twice as much on Windows** -- [part 4](records/PLAN-ARCHIVE-4.md#task-263)
- [x] 264. **The kernel on every x86 build** -- [part 4](records/PLAN-ARCHIVE-4.md#task-264)
- [x] 265. **A wide `ewridge` at 0.56 of 1.0.1's speed, from one settle a run** -- [part 4](records/PLAN-ARCHIVE-4.md#task-265)
- [x] 266. **Windows back on the `gemm` crate: the kernel's products were wrong there on a processor that reports no L3** -- [part 4](records/PLAN-ARCHIVE-4.md#task-266)
- [x] 267. **The kernel on Windows after all, the wrong products a known issue** -- [part 4](records/PLAN-ARCHIVE-4.md#task-267)
- [x] 268. **`private-gemm-x86` 0.1.22: the kernel's wrong products fixed upstream** -- [part 4](records/PLAN-ARCHIVE-4.md#task-268)
- [x] 256. **Per-job feature lists in `po.gram.solve_subsets` and `lars_paths`** -- [part 4](records/PLAN-ARCHIVE-4.md#task-256)
- [x] 257. **The targets inside the blocked Gram product** -- [part 4](records/PLAN-ARCHIVE-4.md#task-257)
- [x] 255. **The memory findings of the 1.0 budgets** -- [part 4](records/PLAN-ARCHIVE-4.md#task-255)
- [x] 254. **The FFI leak check on Linux (after 1.0)** -- [part 4](records/PLAN-ARCHIVE-4.md#task-254)
- [x] 248. **A task worktree rebuilds only what it changes (after 1.0)** -- [part 4](records/PLAN-ARCHIVE-4.md#task-248)
- [ ] 261. **Fit and predict in one bank** -- requested by the user
      2026-10-14 ("Add a plan item to allow fit and predict in one bank").
      Today a bank serves `predict` or learns, never both at once:
      `fit_predict`, `fit` and `fit_predict_batches` take the bank
      exclusively for each native call (one chunk; `try_borrow_mut` in
      `crates/online-py/src/lib.rs`), and a `predict` arriving meanwhile
      from another thread is refused with "the bank is in use on another
      thread", not made to wait. Between chunks the borrow is free, so a
      `predict` succeeds or fails on timing alone. Serving while learning
      takes a second bank, refreshed by save and load. (Task 260 is held
      by the faer kernel branch.)
      **Requirement:** (a) `predict` (and `predict_arrow`) on a bank that
      is learning returns, scored against a consistent state: the state
      as of a chunk boundary, never a chunk half learned. Learning order
      and every number `fit_predict` writes stay as they are, to the bit
      (hard rules 2 and 3), and concurrent `predict` calls still never
      wait on each other. (b) In one call, some specs learn while others
      only score, frozen: a per-spec choice rather than the whole call's
      `learn_only` -- or a documented reason not to, if (a) and a second
      bank cover it.
      **Options for (a), cheapest first:** 1. wait, do not refuse: a
      blocking lock taken per chunk with the GIL released, so a `predict`
      waits for the chunk in flight (at most one chunk's time) and sees
      the state after it; 2. publish a snapshot at each chunk boundary
      that `predict` reads while the next chunk learns (no wait, one
      chunk stale), copy-on-write per group so only the groups a chunk
      touches are copied; 3. both, chosen per bank. Each option's
      memory and per-chunk cost measured before choosing (CLAUDE.md:
      a per-row or layout change carries its cost), on a wide `ewridge`
      and on many small groups.
      **Check:** a thread scoring in a loop while another streams a
      LazyFrame through `fit`: no refusal; every `predict` result equal,
      to the bit, to `predict` on a copy of the bank saved at some chunk
      boundary of that stream; the stream's own output equal to a run
      with no scoring beside it; under (b), a frozen spec's state the
      same bytes before and after the call, its output equal to
      `predict`'s.
      **Potential gain:** one process serves and learns from one bank:
      no second bank in memory, no save/load refresh loop, and no
      refusal to retry on timing.
**Parked by the user on 2026-09-25: integration with new libraries, Arrow,
and licensed libraries in tests.** Nothing here is to be built until the
user lifts it:
- Task 86's Arrow import (`fit_predict_capsule`, `Bank::fit_predict_stream`)
  — unblocked since 2026-09-22, waiting by choice, not on a blocker.
**Dropped by the user on 2026-10-06: both proposed Polars patches.** The
parquet reader's projection-order patch, with its report, and the
polars-arrow ask (whose premise had already proved false, ARROW-SOURCES
§4). The user's words: "We do not want either of the two proposed polars
patches, they can be removed along with everything supporting them." Their
working directory went with them: the polars clone and its two branches,
the drafts, the probe scripts and the generated data. PERFORMANCE.md's
account of the pushed-down filter's memory stays, because it documents
polars' own behaviour, which no patch changed.

## 11a. Decisions made while implementing

*Moved, text unchanged: the text of this section, from [part 4](records/PLAN-ARCHIVE-4.md#s11a) through [part 5](records/PLAN-ARCHIVE-5.md) of the plan archive.*

## 11b. Performance plan

**Done — P1 through P11.** See [`docs/PERFORMANCE.md`](PERFORMANCE.md): the
integration layer cost 3–5× the model arithmetic and capped thread scaling at
3.2× on ten cores. Removing per-row allocation, flattening the rayon fan-out to
(spec × group × instance), extracting columns as `f64`-with-NaN instead of
`Option<f64>`, and pipelining the runner took it to **2.0–2.4× throughput and
6.2× scaling**, with every golden number unchanged. Three of the eight items
were closed by measuring and *rejecting* the change; §5 there records why.
P9–P11 (2026-09-04, §12 there) then made the phases around the per-group
tasks parallel too — a 64-group chunk at 14 threads 37 → 17 ms of wall —
and reversed one of those three rejections on new measurement.

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

[`docs/BEYOND-O-STATE.md`](records/BEYOND-O-STATE.md): what a relaxed memory bound would
unlock, checked against crates.io so "nobody has built this" is evidence rather than
assumption. Three strong candidates (adaptive conformal prediction, frequent-directions
sketching, rolling-window regression), three weak, and Hoeffding trees left to MOA on
purpose. A survey, and since then B1 was built as E36 and B2 as `window`
(task 63); B3 and B4 wait in task 118. The one condition attached: a relaxed bound
would have to become a *stated, tested* property, not a habit.

## 11f. Pre-release improvements review

[`docs/IMPROVEMENTS.md`](IMPROVEMENTS.md) (C1–C6, P1–P4, U1–U8, X1–X2, T1–T5):
one pass per axis — correctness, performance, usability, extensibility,
testing — with every finding reproduced before it was written down. Done so
far: the emit flags through the expression plugin (C1), a bounded-input
contract for every model with the test that enforces it (C2/T4), `.over()`
running groups in parallel (P1), features as expressions (U1), a chunk
refused under `on_clock_reset="error"` leaving the bank untouched (C3),
parameters that used to run and produce garbage refused by name (C4), and
the one T4 found: covariance-form `rls` and `ew_cov`'s tracked inverse die of
cancellation on a single extreme row, so `rls` is now in square-root (QR)
form and the precision matrix is solved on demand (C5, schema 2), and error
messages that name the spec, the parameter or the column and its role — the
builders check their own annotations, the parser names the path, and a
non-numeric column is refused rather than cast to null (U2), and a bank that
can say what it holds — `repr`, `groups()`, `drop_groups()`, `rows_seen()`,
and `specs` that survive `load` (U3), and a typed Python surface — PEP 692
kwargs on the builders and the namespace, `po.online(expr)` for the type
checkers that cannot see a registered namespace, and mypy in the gate (U4),
and the CLI tests running a once-built executable instead of `cargo run`
per call -- 33 s to 3.6 s, and the cost turned out to be macOS validating a
freshly cloned 418 MB binary's signature, not cargo (T1), and doc tests
on the crate roots, the trait and the clock, so `cargo test --doc` now
compiles the examples a Rust reader sees first (T2), and a model registry
(`ModelKind::KINDS`) that every per-model list — builders, namespace,
sweeps, golden bank, API snapshot, README — is tested against, with
`docs/EXTENDING.md` as the ordered list of places a model touches; writing
those checks found `ftrl` missing from the golden pipeline (X2), and state
and output files written through a temporary and renamed into place, so an
interrupted save no longer destroys the state it was updating (C6), and the
dtype features a frame can carry across the boundary, since a `Decimal`
column the spec never named used to abort the process (U5), and a refresh of
the published throughput numbers, which were stale in both directions: every
`ewridge` case 14-62% faster than the README claimed and `rls` 48% slower,
that last one C5's square-root rewrite, measured by A/B against the commit
before it (P4), and the README's python blocks run rather than merely compile,
which found its `holt` example refused by its own validation (T5, U6), and
`coef` reporting "nothing yet" as null rather than as an empty list, which is
what made the documented `coef.list.get(position)` raise (U7), and the
scoring path documented at last -- `weight = 0` freezes the fit bit for bit
where a null target quietly degrades it, at the cost of an `n_eff` that keeps
decaying while you score (U8, ENHANCEMENTS E31). E31 itself followed
(2026-09-02): `ModelBank.predict(df)`, `po.run(predict=True)` and the CLI's
`--predict` score against the bank as it stands and move nothing, built on
an `OnlineModel::predict` that every model implements and derives its own
`step`'s prediction from, so the two cannot drift.
The rest is proposed with its measurements next to it.

## 11g. Gradient-boosted trees

[`docs/BOOSTED-TREES.md`](BOOSTED-TREES.md): how far XGBoost's method can be
pushed toward the contract — the paper and source read with citations, the
streaming-tree literature and implementations compared, a design that keeps
every rule (decayed per-node sums, a bounded histogram pool, growth and
collapse only at checkpoints, a batch warm start), a numpy prototype
(`scripts/ogbt_proto.py`) measured against XGBoost refits
(`scripts/ogbt_experiments.py`), the ideas that failed with their numbers,
and the cost of a Rust build. Investigation only — nothing in the crates;
the build decision is the user's (task 21, §11a).

## 11h. Online clustering

[`docs/CLUSTERING.md`](CLUSTERING.md), investigated on the branch
`online-clustering` and merged 2026-09-04: every
clustering family the field has produced, decided against the contract — what
fails does so for one of three reasons (it needs the rows back, its state is not
bounded by parameters, or it puts randomness on the output path), and what
passes reduces to `EwCov`'s decayed weighted mean with an assignment in front of
it. Nine numpy prototypes (`scripts/clustering_proto.py`) measured
(`scripts/clustering_experiments.py`) on drifting mixtures with outliers and
regime changes: chunk invariance, determinism, zero-weight and null rows all
bit-exact; seeding is the largest source of variance and the right rule depends
on the outliers expected; a split–merge move on a slower clock than the centre
update is what makes fixed-`k` k-means survive drift. On hard geometries the
streaming costs nothing and the family costs everything: `micro` reaches
DBSCAN's ceiling on moons, rings and rows (0.998 / 0.999 / 0.998 against 1.000)
where every k-means and GMM scores 0.000 on the rings, with the threshold rule
measured and derivable at the checkpoint — it is the design worth the build
decision, for the seven reasons in §0 and ENHANCEMENTS §4. §8 settles the spec
and the static output schema, §9 costs a Rust build, §10 lists what failed.
Two designs were built on the user's decision: `kmeans` (task 23) and
`micro` (task 24). ENHANCEMENTS §5.1 is the follow-on inventory of what else fits the
online contract (E36–E42).

## 12. Open questions (not blocking)

- ~~Overnight handling beyond `session_gap` (e.g. partial state shrinkage toward a long-run prior).~~
  Answered: `session_shrink` + `long_halflife` mix the accumulators toward a
  slow-moving twin at a session boundary (ENHANCEMENTS E6).
- ~~Whether targets at long horizons need a different `min_periods` than short ones.~~
  Answered: `min_periods` accepts a list, one entry per target (ENHANCEMENTS E7).
- ~~Public intraday dataset choice for tests (stable URL, permissive licence).~~
  Answered and in use: Binance's public daily kline dump
  (`data.binance.vision`, BTCUSDT 1-minute rows) — stable per-day URLs, no
  auth. `tests/data.py` downloads it on demand, caches under the gitignored
  `.cache/`, and skips when offline, so hard rule 1 holds. It backs both the
  reference comparisons and the defaults measured in `docs/VALIDATION.md`
  (14,336 rows).
- **A zero-weight row on a capped gap does not age the history** (found
  2026-09-08 while reviewing task 71; pre-existing, both Gram paths agree).
  The stream accepts a zero-weight row (`usable(0.0)` holds) and the clock
  consumes the gap, but `EwCov::step_factors` refuses `lam = 0, w = 0`
  because `lam·W + w = 0`, so the accumulator is left as it was: `n_eff`
  reports the old count, and the next weighted row blends with a history the
  clock says is `2^-10000` gone. A gap one unit *under* the cap, or the same
  gap followed by a weighted row, wipes it. One row wide, and only at the cap
  — a fix would set `w_sum = 0` and let the next row's `a = 0` start over,
  which changes shipped output for that row and needs its own test and
  decision. Not blocking task 71.

  *Re-derived 2026-09-24: the trigger is the underflow, not the cap.* The
  refusal happens when the decay factor `2^(-gap/h)` is exactly 0, which is
  from 1075 halflives on, capped or not. An uncapped zero-weight row after
  1075 halflives keeps the history too. One after 1074 halflives (factor
  `2^-1074 > 0`) ages it to nothing, and so does a gap capped at 1000
  halflives. "A gap one unit under the cap wipes it" holds only when that
  gap is under 1075 halflives. So a cap under about 1075 halflives never
  meets this, and `max_dclock = inf` is where it lives. `ewridge` and
  `ew_cov` agree. The test now exists:
  `tests/test_edge_cases.py::TestWeights::test_a_zero_weight_row_keeps_the_history_when_its_decay_underflows`
  pins the current behaviour, and the decision on the fix is still the
  user's.

  *Decided and built 2026-09-28 (task 115 (c), with task 120): the row
  forgets.* It is the decay alone, as the row one halflife short all but is;
  the test above became `test_a_zero_weight_row_forgets_the_history_its_decay_takes`,
  and `model_contract.rs` holds every model to it.
## 13. `window`: an EW accumulator with a hard cutoff (2026-09-06)

*Moved, text unchanged: the text of this section, from [part 5](records/PLAN-ARCHIVE-5.md#s13) of the plan archive.*

## 14. Review round R1 (2026-10-03), after tasks 143, 144, 104 and 150

*Moved, text unchanged: the text of this section, from [part 5](records/PLAN-ARCHIVE-5.md#s14) of the plan archive.*

## Follow-on documents

Each of §11b–§11h below summarises one document under `docs/` and says what
became of it. Four more carry no section of their own:

- `docs/ENHANCEMENTS.md` — every model and feature after the first seven
  (E1–E74): proposed, measured, built or declined.
- `docs/TESTING.md` — coverage scorecard against §9, the defects the suite
  found, and the oracle/river cross-checks.
- `docs/STATE-WORKFLOW.md` — research (2026-09-03) on carrying state out of a
  streamed plan: what polars does with a Python source, measured; the
  candidate forms; the rules `save_state=` on the plan follows and the
  decisions behind them (task 20).
- `docs/ARROW-SOURCES.md` — research and a proposal (2026-09-17, nothing
  built) on feeding a bank from any Arrow producer: which libraries implement
  the PyCapsule interface, the tier that already works through polars with no
  new dependency, and a native import, which needs no second Arrow
  implementation (§4, corrected 2026-09-22; parked by the user on 2026-09-25),
  and a comparison with DuckDB's own statistics and learning extensions,
  which are broad but batch-first and keep no state between queries.

## 15. Review of `v0.13.0..a7a8f3c` (2026-10-05): the 53 commits since the tag

*Moved, text unchanged: the text of this section, from [part 5](records/PLAN-ARCHIVE-5.md#s15) of the plan archive.*

## 16. Whole-project review at `e6b70a2` (2026-10-05): every file but Markdown

*Moved, text unchanged: the text of this section, from [part 5](records/PLAN-ARCHIVE-5.md#s16) of the plan archive.*

## 17. Parameters counted in rows, where a clock would fit (2026-10-06)

*Moved, text unchanged: the text of this section, from [part 5](records/PLAN-ARCHIVE-5.md#s17) of the plan archive.*

## 18. Whole-project review before 1.0 (2026-10-06): everything, with a 1.0 lens

*Moved, text unchanged: the text of this section, from [part 5](records/PLAN-ARCHIVE-5.md#s18) of the plan archive.*

## 19. Review round 5 (2026-10-08): tasks 183-204, code and theory

*Moved, text unchanged: the text of this section, from [part 5](records/PLAN-ARCHIVE-5.md#s19) of the plan archive.*

## 20. Review round 6 (2026-10-09): tasks 221-231, code and concepts

*Moved, text unchanged: the text of this section, from [part 5](records/PLAN-ARCHIVE-5.md#s20) of the plan archive.*

- [x] 232. **Review 6's diagnostic decisions** -- [part 5](records/PLAN-ARCHIVE-5.md#task-232)
- [x] 233. **Review 6's data-check decisions** -- [part 5](records/PLAN-ARCHIVE-5.md#task-233)
## 21. Review round 7 (2026-10-11): everything since round 6, code and concepts

*Moved, text unchanged: the text of this section, from [part 5](records/PLAN-ARCHIVE-5.md#s21) of the plan archive.*

## 22. Review round 8 (2026-10-16): everything since round 7, code and concepts

*Moved, text unchanged: the text of this section, from [part 5](records/PLAN-ARCHIVE-5.md#s22) of the plan archive.*

- [x] 269. **Review round 8's fixes** -- [part 5](records/PLAN-ARCHIVE-5.md#task-269)
## 23. Review round 9 (2026-10-23): the whole repository, code, theory, performance and memory

*Moved, text unchanged: the text of this section, from [part 5](records/PLAN-ARCHIVE-5.md#s23) of the plan archive.*

- [x] 270. **Review round 9's fixes** -- [part 5](records/PLAN-ARCHIVE-5.md#task-270)
## 24. Huber regression, from first principles (2026-10-23)

*Moved, text unchanged: the text of this section, from [part 5](records/PLAN-ARCHIVE-5.md#s24) of the plan archive.*

- [x] 271. **`huber_scale` and the Huber documents** -- [part 5](records/PLAN-ARCHIVE-5.md#task-271)
- [x] 272. **Huber through the bank, `po.eval`, `po.gram`, `marginal` and the window operators** -- [part 5](records/PLAN-ARCHIVE-5.md#task-272)
## 25. Windows' math library and hardware in the tests (2026-10-24)

*Moved, text unchanged: the finished tasks' full entries (the open tasks and the notes still in force stay here), from [part 5](records/PLAN-ARCHIVE-5.md#s25) through [part 6](records/PLAN-ARCHIVE-6.md) of the plan archive.*

The user, 2026-10-24: "Some windows versions produce small test errors on
the python suite but never rust can you think of a reason". The CI logs
since 2026-10-05 show that Rust does fail there too:
`a_windows_cusum_null_share_is_its_absorption` was 1.7× past its `2⁻⁵²/v`
bound on 2026-10-17 (0.9× on Apple's and glibc's `exp`; widened to 4× in
task 269), and `trisolve … up_to_1500` was 100× over on 2026-10-15 (the
private-gemm-x86 0.1.20 kernel, task 268). The Python legs' Windows failures
in that window were cp1252, path separators and `file://`, none numeric.

The causes. (1) On Windows, `exp`, `ln`, `exp2`, `powf` and `exp_m1` from
Rust, CPython and numpy all come from `ucrtbase.dll`, an operating-system
component that changes with the Windows build, while glibc and Apple's
library are fixed per image. (2) The Rust tests were written around this:
pinned streams avoid these functions and each tolerance was measured on the
three platforms. The Python tests compare the wheel with numpy, scipy,
statsmodels, scikit-learn or polars, each with its own CPU-dispatched last
bits (OpenBLAS kernels, numpy's own vector `exp`/`log` on some CPUs, polars'
SIMD sums), so a comparison carries both sides' rounding. (3) A Windows
label is a pool of machines (AVX-512 or not, any L3 size), and CI does not
record which one ran; blocking in the product kernels depends on both. CI's
Hypothesis profile is derandomized, so luck in drawing examples is not a
cause.

Decided (the user, 2026-10-24: "Yes and do it after the current work"):
no blanket epsilon. Tests that compare a build with itself (chunk
invariance, a state continuing to the bit) stay exact; a test pinned to
exact bits of a value that went through one of these functions changes its
stream, not its tolerance; an oracle comparison gets a derived bound, a few
ulps of each such call times the formula's amplification, measured on all
three platforms and written in a comment (task 269's `4·2⁻⁵²/v` is the
model), with ulp comparisons in Python (`np.testing.assert_array_max_ulp`).

- [ ] 273. **Windows hardware coverage and the math-library bounds** --
      after task 272 and its push. (a) CI: `windows-2022` beside
      `windows-latest` in the test matrix (a second `ucrtbase.dll`), and a
      first step on every Windows job printing the CPU model, L3 size,
      AVX2/AVX-512 and `ucrtbase.dll`'s file version, so a failure names its
      machine. (b) A weekly workflow that forces the CPU paths rather than
      sampling the pool: the numeric oracle tests and the wide-product tests
      under Intel SDE posing as an AVX2-only chip, an AVX-512 chip and an
      older one, including a different reported cache size; Python's
      oracles under `OPENBLAS_CORETYPE` and `NPY_DISABLE_CPU_FEATURES`. Check
      SDE's download terms for CI use first; if they forbid it, report and
      keep the rest. (c) The audit: every Rust and Python test whose
      expected or actual value passes through `exp`, `ln`, `exp2`, `powf`,
      `exp_m1`, `ln_1p` or a third-party oracle, sorted into the three kinds
      above; each oracle comparison given its derived, measured, commented
      bound; each bit-pinned one moved to a stream that avoids these
      functions. (d) Measure, do not build: online-core's `exp`/`ln`/`exp2`
      from the pure-Rust `libm` crate (a musl port) instead of the operating
      system's, so the Rust side gives the same bits everywhere. Report its
      speed per call and on the budget cases against the platform's, which
      shipped outputs move and by how much, and the bounds it would let
      shrink; adding it is a new statically linked dependency (rule 12), so
      the measurement goes to the user for a decision.
      *Decided 2026-10-24*: the user accepted Intel SDE's licence (the Intel
      Simplified Software License: free for any use, no modification or
      reverse engineering) for the weekly `cpu-paths.yml`, fetched through
      `petarpetrovt/setup-sde` pinned at `6d2773bc` (MIT; the commit NumPy's
      `linux_simd.yml` uses), with the kit's SHA-256 checked against
      Intel's. (a) and (b) merged locally.
      *Built 2026-10-24* (not yet pushed): (c) Python merged (six derived
      bounds; two defects fixed in (c2): the readiness projection read
      rounding noise as a ceiling, `merge_sums`' one-seed claim); (c) Rust
      merged (5 bounds derived, 8 streams moved off libm; perturbation
      sweeps of every Rust test at ±1..±65,536 ulp, `scratchpad/t273/`);
      two library defects it found (a window's subtraction against the
      per-row decay factor, a subnormal variance past the target guard) in
      (w). (d) measured, for the user: the `libm` crate (0.2.16, already
      linked through faer, so no new linking) gives the same bits
      everywhere; `exp2` as accurate as Apple's (0.502 ulp), the rest
      correctly rounded 90-97% (Apple 99.5-99.98%); `powf` 3.5-4.5×
      slower, `exp`/`exp2` 1.4×; budget cases +2% geometric mean, eight at
      +6-10%; 102 of 1,299 probe fields move, at most 1.7e-11 relative.
      Worker's recommendation: no swap now; if platform-identical decay is
      wanted, `exp2` alone first (the half-life decay factor). The 126
      Python replay tests that hold the wheel to the bit through
      `math.exp2` would need their streams moved under any swap.
      *Decided by the user, 2026-10-24*: "no libm" -- online-core keeps the
      platform's `exp`/`ln`/`exp2`/`powf`; the 126 Python replay tests stay
      as they are.
- [x] 274. **Linear-algebra oracle bounds derived** -- [part 5](records/PLAN-ARCHIVE-5.md#task-274)
- [x] 275. **A window's boundary factor from the product of the per-row factors** -- [part 5](records/PLAN-ARCHIVE-5.md#task-275)
- [ ] 276. **The command line's memory on Windows and on wide machines** --
      from task 270 (o) (2026-10-24): jemalloc's default of four arenas a
      core let the CLI's peak grow with the rows (0.94 / 1.36 / 1.48 GB at
      3M / 12M / 24M), fixed off Windows by `narenas:1` (0.75 / 0.85 /
      0.86 GB, the same wall and CPU time on 14 cores). Two things were not
      measured: Windows, whose mimalloc 3.3.2 (purge delay 1000 ms) grew on
      a macOS stand-in (0.86 → 1.01 GB from 3M to 12M) and was left
      unchanged; and one arena's lock under many more than 14 cores. Measure
      both on CI (a dispatchable job on `windows-latest` and a large Linux
      runner, `scripts/memory_bench.py`'s command-line cases), then bound
      Windows' peak the same way or record why not.

**1.3.0 released (2026-10-25).** Run 40188023866 on `92d5bd41`
(publish on; the user: "After everything is green push the next minor
version", then "Dispatch now" with the weekly CPU-paths workflow's forced
Haswell OpenBLAS leg still failing, a third-party crash on Zen 4 hosts that
no user's install reaches); PyPI approved by the user, `v1.3.0` and the
GitHub release (12 assets) by the workflow; verified from a clean venv
(1.3.0 / 1.3.0 / schema 63, a `huber` fit). Before it: a ten-reviewer
documentation review (2 high, 20 medium, 39 low, all applied or measured),
the `huber` sketch panic found by that work (task 272 (p)), the command
line's bounded memory (task 270 (o)), and the release preparation (steps
1-7: compare, benchmark with declared slowdowns, sign-offs, version).
Recorded after it: the run's budgets (`--release 1.3.0`), `RELEASES` gains
1.3.0 and `UNRELEASED` empties, the README's install size from PyPI's
wheels (24.5-29.8 MB, 70.9-98.4 MB installed).
- [x] 277. **Huber's converged warm-up, on by default, for `huber` and `sgd(loss="huber")`** -- [part 5](records/PLAN-ARCHIVE-5.md#task-277)
- [x] 278. **Convergence tests of every kind, before 1.4.0** -- [part 5](records/PLAN-ARCHIVE-5.md#task-278)
- [ ] 279. **Smaller build caches: no debug info for dependencies** -- the
      user, 2026-10-25 ("Why do agents take so much disk I though we shared
      caches", then "Yes"). Each worktree slot keeps its own `target/` (a
      shared one clobbers the workspace's crates between paths), and they
      grow to 58-171 GB: slot 2's 171 GB was 130 GB of `debug/deps` (234,000
      files: every test binary links Polars with its debug info, and cargo
      keeps each superseded build -- mutants, debug-assertion release builds,
      cross builds, other toolchains) and 24 GB incremental. Measure
      `[profile.dev.package."*"] debug = false` (our crates keep line
      tables): a slot's `target/` size after the essentials gate, the gate's
      cold and warm build times, the link time, and a backtrace through our
      code still naming lines; propose it with those numbers. Meanwhile, a
      slot's `target/` is cleared when its task is given back whenever free
      disk is under 100 GB, and `cargo-sweep` between tasks is to be weighed
      beside it. After the task 277/278 wave, so no slot rebuilds mid-task.
- [x] 280. **An oracle ledger: no drift from the oracles and datasets, no longer gate** -- [part 5](records/PLAN-ARCHIVE-5.md#task-280)
- [ ] 281. **Task 278's follow-ups, after 1.4.0** -- recorded 2026-10-26.
      (1) The stream's own anchored MAD sketch is anchored at its first
      residual, so with the refit off a tiny first residual carries the
      level's rounding into the scale (none of 50 streams hit it; the refit,
      on by default, re-anchors at the residuals' median size); an anchoring
      rule over a few values needs state. (2) `refresh_time(pairs=True)` with
      a `head(n)` ending among one tick's pairs loses the rest of that tick
      on resume (documented, pinned); keeping them is a layout change. (3)
      `kalman`'s covariance form loses digits at a huge prior (4.2e-6 off
      least squares at `p0=1e12`; documented): a square-root or Joseph form.
      (4) The fixture index and shared test-case registry generated rather
      than appended to, so new fixtures stop conflicting between branches.
      (5) `spec.rs` sits at 249.9 KB of the 250 KB cap: split a seam.
      *(4) and (5) built 2026-11-04 (W45, `task-281-followups`); (2)
      became task 321 (built); (1) became task 323 (built); (3) awaits
      the user* (measured and
      proposed, not built:
      `.claude/coord/W45-281-scratch/PROPOSALS.md`). [x] (5): `spec.rs`
      249,960 -> 216,675 bytes, a pure move of two seams, re-exported from
      `spec` so no path, API or serialized form moves: the value types a
      parameter is written in (`spec/values.rs`) and the spec's clock --
      `CLOCK_FIELDS` and its sibling tables, `ClockScale`, `ClockPolicy`,
      `clock_cfg_of`, `Spec::clock_spans`/`clock_scale` (`spec/clock.rs`).
      [x] (4): the state fixtures are discovered, not listed. Each crate's
      `build.rs` (one file in both, a test holds the copies equal) writes
      `$OUT_DIR/state_fixtures_index.rs` from every `<case>.rs` under
      `tests/state_fixtures/` and under each `v<N>/`; the generated
      `index.rs` files and `previous![..]` are gone, so a branch adding a
      fixture adds a file and edits no shared list, and a kept schema is a
      `v<N>/` directory. Counted at run time: the current set compiled in
      equals the `.rs` files on disk, the fixtures' names equal the cases'
      (as sets), each kept set is non-empty and every fixture in it of its
      directory's schema. `PRINT_STATE_FIXTURES=1` rewrites both crates'
      fixtures byte-identically. Not done, and what remains of the
      conflict: the case registries themselves (`tests/cases/mod.rs`'s
      `all()`, the bank harness's `kinds()`) are hand-written code a new
      case appends to; two branches conflict there only where they append
      at the same spot. Discovering them would mean a file per case (78 +
      13), splitting each family's cases and the comments that relate them
      -- more than a tidy refactor, so left for the user. CLAUDE.md rule 5
      named `previous![..]`; corrected to the `build.rs` discovery 2026-11-04.

- [x] 282. **Review round 10's fixes, before 1.4.0** -- [part 5](records/PLAN-ARCHIVE-5.md#task-282)
- [x] 283. **One refit buffer a group, shared by its targets** -- [part 6](records/PLAN-ARCHIVE-6.md#task-283)
- [x] 284. **Pay less for the per-row drift since 1.3.0** -- [part 6](records/PLAN-ARCHIVE-6.md#task-284)
- [x] 285. **No per-row allocation: every model writes into the bank's kept `Step`** -- [part 6](records/PLAN-ARCHIVE-6.md#task-285)
## 26. Performance plan, from the VM analyses (2026-10-31)

*Moved, text unchanged: the finished tasks' full entries (the open tasks and the notes still in force stay here), from [part 6](records/PLAN-ARCHIVE-6.md#s26) through [part 8](records/PLAN-ARCHIVE-8.md) of the plan archive.*

Evidence: the user's three Linux VM reports.
- **The refit bisect** (artifact Sbc47fyMNLqjkNRcAuJXZz).
- **The growth breakdown** of sgd, sgd_huber and holt by function and line (DGCWop9Q8g4J8uhqpRw7Ny).
- **The performance page** (3RonHo8trZ8CQvGWRez3CN): native time against instructions on all 31 model cases, a cost map by area, Callgrind's cache and branch simulation, the dynamic instruction mix, llvm-mca on the hot loop, feature scaling 1-80, and the whole suite against budgets.

Plus tasks 284/285's native measurements and CI's counts.

What the evidence says:
1. **Instructions are not time.** holt grew 17.6% in instructions but 52% in time (35 → 53 ns a row): its new divisions sit on the dependency chain carried from row to row (5 → 10 a row), and its `take_line` branch mispredicts on 535 of 1,000 rows. sgd grew 16.8% in instructions but 11% in time, because its new work overlaps.
2. **The cases are compute- and latency-bound.** The working set fits in L1 (about 3 L1 misses a row, no last-level misses).
3. **The largest remaining costs are older than 1.4.**
   - sgd: the per-feature update loop is 33% (since cut by task 284), `standardize_row` 22%, and the scaler's EwDiag update 10%. The last two are vectorized only two lanes wide (SSE2, baseline x86-64).
   - sgd_huber: the Huber scale (`EwQuantile`) is 11.7%.
   - holt: before task 285, 28% went to allocating a `Step` a row (task 285 removes it).
   - The refits: wls's Gram build dominates `sgd_huber_refit`.
4. **Only sgd, sgd_huber, holt, ewridge_k20 and quantile slowed beyond noise in time.** Elsewhere time tracks instructions.

Ordering below: bit-identical first (no fixture moves, no user decision), then what needs a measurement campaign, then what moves numbers within rounding (the user's call, declared in the CHANGELOG with its size, as a minor version allows).

**Phase A: bit-identical, build now (each held to the digests, fixtures, contract, chunk invariance)**

- [x] 286. **holt's row-to-row chain** -- [part 6](records/PLAN-ARCHIVE-6.md#task-286)
- [x] 287. **The refit's Gram, half of it** -- [part 6](records/PLAN-ARCHIVE-6.md#task-287)
- [x] 288. **sgd after task 284: the spills, and the Huber scale** -- [part 6](records/PLAN-ARCHIVE-6.md#task-288)
**Phase B: breakdowns before building (one VM run)**

  *Built 2026-11-01:* task 287 (W12): the Gram by 4x4 register blocks over its lower triangle; the median by selection with the sort as fallback inside a proven margin; band_system, solve_sum and bocpd's full Σ lower-triangle only. Task 288 (W11): EwQuantile's bucket from the float bits and the representative from a table (no floor, no division); `add`'s guard tested once; step_mapped out of line. Both are bit for bit over 38 to 48 digests.
  Task 286 (W11), measured and not merged:
  - branch-free take_line is −27-29% in the coin-flip regime, but +11% settled and +54% on a regular-clock warm-up (arm64); the VM times it on x86 (V2.3);
  - one exp2 for equal half-lives does not pay (the default is trend = 4 × level);
  - holt's latency (+27% time on −10% instructions vs 1.3.0, VM V1) goes to task 291 with the VM's reciprocal measurement (V5).
- [ ] 289. **A VM performance page for the next tier and the largest cases**, with the same method as 3RonHo8trZ8CQvGWRez3CN: native time, a cost map by area, the instruction mix and the branch simulation.
  - The next tier of drift since 1.3.0: quantile (+6.4% instructions, +7.9% time), ewridge_k20/k5 (+5%, +8% time), bank_ewridge_k5 (+7.5%), bank_seqtest (+5.9%), formula_target_held1k (+5.4%).
  - The largest absolute cases: bank_ew_cov_wide_t1 (166M), corrchange_window_w100 (108M), huber_k20_refit (94M), rcov (91M), quantile_refit (62M), ew_class (50M).
  - Run it at main after tasks 285-288, so it measures what is left. Its findings become the next build wave, ranked as Phase A was.
  - *Checked 2026-11-01 (the user: "verify 289 is superseded"): partly.* The VM's performance map at c9345eb5 and the waves since (tasks 290-298) covered the largest cases and most of the drift. CI's counts at 9a4acde1 against 1.3.0: quantile −3.0% (297 takes about 7% more), ewridge_k20 −1.3%, bank_ew_cov_wide_t1 −22%, rcov −21%, ew_class −11%, bocpd_full −29%. **Still drifting, all through the bank and stream plumbing in online-polars:** bank_ewridge_k5 +6.1%, formula_target_held1k +4.3%, bank_seqtest +4.0% (its groups case +3.5%), ewridge_k5 +2.8%. **Never studied:** corrchange_window_w100 (+0.9%, still the costliest case a row; `read_into` 67%, `permutation_crit` 31%). 289 is narrowed to those five cases, for the VM after V15 and V16; none is past the 10% sign-off, so none blocks 1.4.0.

**Phase C: wider vectors with runtime dispatch (bit-identical by construction, measure first)**

- [x] 290. **SIMD past SSE2 for the per-feature loops** -- [part 6](records/PLAN-ARCHIVE-6.md#task-290)
**Phase D: moves numbers within rounding (the user's decision, before any build)**

- [ ] 291. Measured candidates that change last bits, each with its size, for the user to choose:
  - holt's chain divisions as kept reciprocals: 2-3 divisions off the chain, ≈ 10-15 ns of 53.
  - |z̃|² folded into `standardize_row`'s loop: ≈ 100 Ir a row of sgd at 20 features.
  - The refit's Gram through faer's SIMD matmul.

  Each needs the fixtures regenerated and a CHANGELOG line stating the size of the move. Not built without the user's word.

**Process**
- **Time, not only instructions:** instruction budgets miss latency (holt: +17.6% counted, +52% timed). The release step that benchmarks HEAD against the last release's cached wheel already times the bank. Add a per-model native timing of the budget cases against the last release (the VM page's harness: ABBA rounds, minimum and p25) to the release checklist, and to each performance task's report.
- **The VM is a standing performance lab** (the user, 2026-10-31: "Use the vm not only to analyze inflation between releases but to generally improve performance").
  - The user's Linux x86-64 VM, the hardware CI counts on, has Valgrind/Callgrind and llvm-mca.
  - It reads code only from GitHub. Unpushed commits go to `vm/*` branches, which trigger no workflow.
  - The two sessions talk through one artifact, https://claude.ai/artifact/3RonHo8trZ8CQvGWRez3CN: requests V*, replies R-V*, proposals P*, a one-line log, the page kept small and long reports in separate artifacts.
  - The coordinator reads it every 20 minutes, or sooner when the user sends a comment on it to Claude.
  - The VM profiles the whole suite (V3, which widens task 289), prototypes the AVX2 path (V4, task 290), and proposes optimizations with evidence. Accepted proposals become PLAN tasks that workers build, under the usual bit-identity and fixture rules.
- **Task 283, the shared refit store, resumes after Phase A,** rebased: its single-target sgd_huber time anomaly (+18.8% at +0.96% instructions) gets the same VM timing method before anything merges.
- [x] 291a. **holt: kept reciprocals with the branch-free `take_line` (R+S)** -- [part 6](records/PLAN-ARCHIVE-6.md#task-291a)
- [x] 292. **VM proposals P1, P4, P7, P8 (bit-identical)** -- [part 6](records/PLAN-ARCHIVE-6.md#task-292)
- [x] 293. **VM proposals P3, P5, P6 (bit-identical)** -- [part 6](records/PLAN-ARCHIVE-6.md#task-293)
- [x] 290 follows 291a. **port the VM's AVX2 `standardize_row` patch (artifact H91pgCKWPzWcxge1JBehgM, local commit…** -- [part 6](records/PLAN-ARCHIVE-6.md#task-290-2)
- [x] 294. **P6: every EwCov's co-moment matrix as one symmetric lower triangle** -- [part 6](records/PLAN-ARCHIVE-6.md#task-294)
- [x] 295. **ew_cov's per-row zero-fill, an arm64 regression from task 294** -- [part 6](records/PLAN-ARCHIVE-6.md#task-295)
- [x] 296. **The VM's two AVX2 kernels: the scaler's `EwDiag::update` and the refit's Gram blocks (V8)** -- [part 6](records/PLAN-ARCHIVE-6.md#task-296)
- **Decided 2026-11-01 (the user, "Adopt"; the VM's V11): how "worse on one platform" is judged.** Code layout alone moves untouched cases: a function grown by 4 instructions shifted about 48,700 later ones by 16 bytes and moved `pa` and `huber_k20` by 1-3.6% with byte-identical code (task 296); entries moved within their 64-byte line moved kalman and quantile by 1.5-2.4% (task 295). So a case whose instruction count does not move and whose time moves by less than about 3% is read as layout, and a slowdown counts as a regression only when it persists across two layouts: the build, and the same build with an unrelated, harmless change (`-C llvm-args=-align-all-functions=6` against the default). Pinning the layout was measured and refused: aligning every function narrows some spreads and widens others (the largest still 2.3%), shifts unchanged cases by -4.4% to +5.1%, and costs 2.65% more text; per-function `#[align]` is nightly-only (`fn_align`, E0554 on stable 1.99).
- [x] 297. **Robust's per-row allocations as kept scratch (the VM's P11)** -- [part 6](records/PLAN-ARCHIVE-6.md#task-297)
- [x] 298. **hmm, deco, ew_class and bocpd_full keep their factorizations' working space (the VM's P13)** -- [part 6](records/PLAN-ARCHIVE-6.md#task-298)
- [x] 299. **Rust 1.99.1** -- [part 6](records/PLAN-ARCHIVE-6.md#task-299)
- [x] 300. **Test P2 before 1.4.0** -- [part 6](records/PLAN-ARCHIVE-6.md#task-300)
- [x] 301. **ewridge keeps its factorization's working space (the VM's P14)** -- [part 6](records/PLAN-ARCHIVE-6.md#task-301)
- [x] 302. **Oracle coverage, first batch** -- [part 6](records/PLAN-ARCHIVE-6.md#task-302)
- [ ] 303. **Xcode's tools in quiet periods** -- the user, 2026-11-01 ("Yes"). Each times native code, so each waits until no worker is timing. (a) An Allocations sweep of every budget case: per-row allocations and their callers, ranked, as the next keep-the-scratch tasks. (b) Processor Trace (M4 Pro, Xcode 26.3) on the hard cases: huber_k20_refit after P2, corrchange_window_w100, one bank case. It may need a permission switch only the user can flip. (c) A profile-guided optimization measurement on arm64 and on the VM, using rustup's `llvm-tools`, not Xcode's `llvm-profdata` (its LLVM may not match rustc's). It comes to the user as a proposal with numbers, since it changes how the wheel and CLI are built; nothing in the build changes before that.
  - *Tasks 299-301 done 2026-11-01.* 299: the VM's V16, 34 of 46 counts identical and the rest within two runs' spread, the pin moved alone. 300 (W21, `c9be455f`, `4f1c4ea9`): P2 proved bit-identical. `ewcov/held.rs` holds 64 rows of a pass's steps and applies them tile by tile, each entry taking each row's step in row order; arm64 refit block huber_k20_refit −12 to −16%, quantile_refit −3.5 to −5% in both layouts. A rounding variant (a weighted Gram over each hold, about 25% more on that loop, entries within 4.3e-15) was timed in a micro-benchmark only and not built. 301 (W23, `eccb672d`): `Factors` keeps a boxed `FactorWork`, the closure-taking `ladder` became the iterator `rungs`; arm64 ewridge_k5 −5.6% / −6.8%, ewridge_k20 −3.1% / −2.6%, 49 digests match. W23 counted 69 solves in the ewridge_k5 block where the VM reported 412 per 1,000 rows, a difference unexplained. Watched on x86 (V18): bank_ewridge_k5_groups (about +1% at p25, inseparable from noise on arm64) and ewridge_gram_t10.
  - *303(a) done 2026-11-01 (W24; `.claude/coord/W24-303a-scratch/ALLOCATIONS.md`):* every budget case's allocations a row, callers and allocator share of time. Instruments' Allocations stopped at a password window (`com.apple.dt.instruments.process.analysis`), so a counting allocator with stacks did the sweep; its totals equal the counters in all 46 cases. 19 cases allocate nothing in the block. Allocator share of block time, jemalloc / system: hmm 15.9% / 49.3%, deco 10.8% / 56.6%, corrchange_monitor 11.3% / 25.5%, ewridge_k5 7.8% / 33.2%, bank_seqtest_groups 6.9% / 20.7%. Mimalloc on Windows and Callgrind's glibc were not measured.
  - *263 re-measured 2026-11-02 (W22; `.claude/coord/W22-263-scratch/`, CI runs 41378364798, 41384272408, 41384747574, 41388021276, 41392414142):* still live, and only on Intel. `private-gemm-x86` 0.1.22's `MEM` (`src/lib.rs:1286-1298`) is a thread-local of 2 × L3 made on a thread's first `gemm` and freed at thread exit; nothing outside the kernel bounds it except fewer threads running products. Commit above baseline at a 260 MiB L3, threads 1/2/4: about 594/1,120/2,172 MiB (main and 1.3.0 alike, on the real leaf-4 path under SDE and natively with the kernel told the L3); 3,933 MiB at 480 MiB and 4 threads. **Correction to this task's 2026-10-15 reading:** AMD's L3 of 0 on Windows is not the hypervisor. raw-cpuid 11.6.0 maps associativity field 0x9 to `Unknown`, and the kernel turns that into 0 bytes, so EPYC 7763, 9V74 and 9V45 floor at 2-4 MiB (a 4-8 MiB buffer); 26 of 26 Windows draws were AMD with no issue. Options: a buffer sized to the product (`kernel/bounded-kernel.patch`, 30 lines) gives about 120 MiB at 4 threads, the same speed (median 1.01 over 72 pairs) and the same bits, and 2,034 of 2,034 `online-core` tests pass. `gram_threads=1` bounds nothing; lanes hold k × 2 × L3 but are roughly `threads=k` with more code, so W22 advises against them and against a Windows-only cap. An upstream ask is drafted in W22's report and not filed: filing is the user's. Whichever is picked, the Known issue should say Intel, and the "960 MiB on the Windows runners" comments in `ci.yml` and `product_memory.rs` hold only on an Intel draw.
- [x] 305. **Wide `ewridge` fits against 1.3.0** -- [part 6](records/PLAN-ARCHIVE-6.md#task-305)
- [x] 306. **Wide `ewridge` back to 1.3.0's speed, bit-identically** -- [part 6](records/PLAN-ARCHIVE-6.md#task-306)
- [x] 307. **Mutants survivors of 7b9f9df7 and db5e559b** -- [part 6](records/PLAN-ARCHIVE-6.md#task-307)
- [x] 308. **The condition check on blocked Grams** -- [part 6](records/PLAN-ARCHIVE-6.md#task-308)
- [x] 304. **The rows' remaining allocations, first batch (from 303(a))** -- [part 6](records/PLAN-ARCHIVE-6.md#task-304)
- [x] 309. **The bank's per-row plumbing, bit-identically (the VM's P18 and P19)** -- [part 6](records/PLAN-ARCHIVE-6.md#task-309)
- [x] 310. **Mutants survivors of 946deee7 (task 306's ridge shortcut)** -- [part 6](records/PLAN-ARCHIVE-6.md#task-310)
- [x] 311. **A due solve no row can read is not run (option C; the user's approval of 2026-11-02, knowing it is not…** -- [part 6](records/PLAN-ARCHIVE-6.md#task-311)
- [x] 312. **Mutants survivors of 5f890340 (task 308's merge bound), and the CLI's wall-clock flake** -- [part 6](records/PLAN-ARCHIVE-6.md#task-312)
- [x] 313. **The review of 5f890340..0ab3bac1 (tasks 311 and 312), and the push's mutants survivors** -- [part 6](records/PLAN-ARCHIVE-6.md#task-313)
- [x] 314. **The arm64 slowdowns against 1.3.0 that held release 1.4.0** -- [part 6](records/PLAN-ARCHIVE-6.md#task-314)
- [x] 315. **The mutants survivors of 255ae009 (task 314) that held release 1.4.0** -- [part 6](records/PLAN-ARCHIVE-6.md#task-315)
- [x] 316. **Review round 11's thirteen lows, a clock corner and the audit's restart** -- [part 6](records/PLAN-ARCHIVE-6.md#task-316)
- [x] 317. **A release needs no commit after it** -- [part 6](records/PLAN-ARCHIVE-6.md#task-317)
- [x] 318. **The mutants of 7aa1240e (tasks 316 and 317): two `hmm` survivors and an `sgd` timeout** -- [part 6](records/PLAN-ARCHIVE-6.md#task-318)
- [x] 319. **Oracle coverage, second batch** -- [part 6](records/PLAN-ARCHIVE-6.md#task-319)
- [x] 320. **Oracle coverage, third batch** -- [part 6](records/PLAN-ARCHIVE-6.md#task-320)
- [x] 321. **A resumed `refresh_time(pairs=True)` keeps the rest of a tick** -- [part 6](records/PLAN-ARCHIVE-6.md#task-321)
- [x] 322. **The online-EM rebuild's tolerance, derived from the conditioning rather than measured on one platform** -- [part 6](records/PLAN-ARCHIVE-6.md#task-322)
- [x] 323. **No MAD sketch takes an anchor chosen from the residuals one pass sees** -- [part 6](records/PLAN-ARCHIVE-6.md#task-323)
- [x] 324. **What this library adds to XGBoost and LightGBM, designed and measured before any build** -- [part 6](records/PLAN-ARCHIVE-6.md#task-324)
- [x] 325. **`with_windows(group=...)` walked every group on every row** -- [part 6](records/PLAN-ARCHIVE-6.md#task-325)
- [x] 326. **Test how cost scales, not only what it is** -- [part 6](records/PLAN-ARCHIVE-6.md#task-326)
- [x] 327. **A diagonal covariance keeps O(k) state** -- [part 6](records/PLAN-ARCHIVE-6.md#task-327)
- [x] 328. **`corrchange(kind="window")` in O(1) of the window, by mergeable moments** -- [part 6](records/PLAN-ARCHIVE-6.md#task-328)
- [x] 329. **The mutants survivors of 31ee42bd (task 323)** -- [part 6](records/PLAN-ARCHIVE-6.md#task-329)
- [x] 330. **a9ee94fe's follow-ups: its mutants survivors, a jittering scaling count, and fixture freshness** -- [part 6](records/PLAN-ARCHIVE-6.md#task-330)
- [x] 331. **The default `refit_budget` raised from 256 to 320 MiB** -- [part 6](records/PLAN-ARCHIVE-6.md#task-331)
- [x] 332. **A bank call walks only the groups in its chunk** -- [part 6](records/PLAN-ARCHIVE-6.md#task-332)
- [x] 333. **A readiness notice raised in the call that closes its group or span is said** -- [part 6](records/PLAN-ARCHIVE-6.md#task-333)
- [x] 334. **A readiness notice raised under sessions names its session span** -- [part 6](records/PLAN-ARCHIVE-6.md#task-334)
- [x] 336. **Calibrating a model: a guide, and `po.eval.grid_search`** -- [part 6](records/PLAN-ARCHIVE-6.md#task-336)
- [x] 335. **The release review's findings F1-F10, each verified before it was fixed** -- [part 6](records/PLAN-ARCHIVE-6.md#task-335)
- [x] 337. **`po.eval.grid_search` after the user's review: a column for what each row is, and refusals that name their…** -- [part 6](records/PLAN-ARCHIVE-6.md#task-337)
- [x] 338. **The documentation rewrite: every guide, docstring, printed message and exception text, in stages** -- [part 7](records/PLAN-ARCHIVE-7.md#task-338)
- [x] 339. **The downloaded test data survive an outage at their source, without entering git** -- [part 7](records/PLAN-ARCHIVE-7.md#task-339)
- [x] 340. **Schema 67: a failed call's pending notices are saved, and `marginal`'s pre-65 mark is cleared at the freeze** -- [part 7](records/PLAN-ARCHIVE-7.md#task-340)
- [x] 341. **`label_clock=`: each row's label time as a column, scored where it sits and learned at its label time** -- [part 7](records/PLAN-ARCHIVE-7.md#task-341)
- [x] 342. **The conformal step under a label delay, from first principles** -- [part 7](records/PLAN-ARCHIVE-7.md#task-342)
- [x] 343. **A base margin through `po.gbdt`: `base_margin=` on `xgboost_matrix`, `init_score=` on `lightgbm_dataset`** -- [part 7](records/PLAN-ARCHIVE-7.md#task-343)
- [x] 344. **Factor ridges feeding a windowed forward-target ridge: the benchmark, and two refusals that name their remedy** -- [part 7](records/PLAN-ARCHIVE-7.md#task-344)
- [x] 345. **The two instruction-budget overs since task 335: `sgd` and `corrchange_monitor_w2500`** -- [part 7](records/PLAN-ARCHIVE-7.md#task-345)
- [x] 346. **The instruction bench under a frozen clock, and three CI follow-ups** -- [part 7](records/PLAN-ARCHIVE-7.md#task-346)
- [x] 347. **`sgd`'s resting gate without its per-row copy** -- [part 7](records/PLAN-ARCHIVE-7.md#task-347)
- [x] 348. **Two `fit_predict` calls in series stay in bounded memory** -- [part 7](records/PLAN-ARCHIVE-7.md#task-348)
- [x] 349. **A window's snapshots laid out flat, so a ring holds about what `window_budget` counts** -- [part 7](records/PLAN-ARCHIVE-7.md#task-349)
- [x] 350. **The `min_weight` warning gives no ceiling to a target whose weight is falling** -- [part 7](records/PLAN-ARCHIVE-7.md#task-350)
- [x] 351. **The scaling harness's allocation check counts the measured work alone** -- [part 7](records/PLAN-ARCHIVE-7.md#task-351)
- [x] 352. **Eight defects the message rewrite found** -- [part 7](records/PLAN-ARCHIVE-7.md#task-352)
- [x] 353a. **The two bank cases over their instruction budgets, and a Windows-only noise-gate test** -- [part 7](records/PLAN-ARCHIVE-7.md#task-353a)
- [x] 353b. **The memory tightenings of 89a61454 and its 28 mutants survivors** -- [part 7](records/PLAN-ARCHIVE-7.md#task-353b)
- [x] 354. **Exception classes with the old class and the rule's as parents** -- [part 7](records/PLAN-ARCHIVE-7.md#task-354)
- [x] 355. **The Rust side's refusals of a column of the wrong type raise `po.WrongTypeError`** -- [part 7](records/PLAN-ARCHIVE-7.md#task-355)
- [x] 356. **An unbounded `threads`, and four smaller defects** -- [part 7](records/PLAN-ARCHIVE-7.md#task-356)
- [x] 357. **The scaling tests judge a block's time by the process's CPU time, so a loaded machine does not fail them** -- [part 7](records/PLAN-ARCHIVE-7.md#task-357)
- [x] 358. **The 81 mutants survivors of f90e0a37 and ddd639ef** -- [part 7](records/PLAN-ARCHIVE-7.md#task-358)
- [x] 359. **Behavior defects found by the docs pass, stage 3b** -- [part 7](records/PLAN-ARCHIVE-7.md#task-359)
- [x] 360. **Four docstring facts and three doubtful figures, from task 338's stage 4a** -- [part 7](records/PLAN-ARCHIVE-7.md#task-360)
- [x] 362. **Online-core's unit tests ran a Windows runner out of commit: test pools held faer's x86 packing buffer on…** -- [part 7](records/PLAN-ARCHIVE-7.md#task-362)
- [x] 361. **Four figures the docs stated without a measurement behind them** -- [part 7](records/PLAN-ARCHIVE-7.md#task-361)
- [x] 363. **`ewridge` solves on the default schedule while wider than its rows, beside a row cadence** -- [part 7](records/PLAN-ARCHIVE-7.md#task-363)
- [x] 364. **`ewridge`'s row cap skips the solves nothing reads below the floor, under a weight share** -- [part 7](records/PLAN-ARCHIVE-7.md#task-364)
- [x] 365. **The highlights page, `docs/HIGHLIGHTS.md`, and its README section** -- [part 7](records/PLAN-ARCHIVE-7.md#task-365)
- [x] 365b. **The highlights page shows the Python API alone** -- [part 7](records/PLAN-ARCHIVE-7.md#task-365b)
- [x] 365c. **An index at the top of the highlights page, and the README's line to the feature list** -- [part 7](records/PLAN-ARCHIVE-7.md#task-365c)
- [x] 365d. **The highlights page regrouped: the factor residuals move to the signal task** -- [part 7](records/PLAN-ARCHIVE-7.md#task-365d)
- [x] 365e. **The highlights page: no "deploy", many factors and targets, and streaming queries in place of hand-written…** -- [part 7](records/PLAN-ARCHIVE-7.md#task-365e)
- [x] 367. **Could `ewridge`'s unread-solve floor be `min_weight` alone above `k + 1`? No: the fit is read there** -- [part 8](records/PLAN-ARCHIVE-8.md#task-367)
- [x] 366. **A windowed snapshot's fixed parts packed a block at a time, and the co-moment triangle's side read without…** -- [part 8](records/PLAN-ARCHIVE-8.md#task-366)
- [x] 368. **`conformal_rate="auto"` started too wide under spread label delays: the tracker now keeps a multiplier of…** -- [part 8](records/PLAN-ARCHIVE-8.md#task-368)
- [x] 369. **`want_readers` built only when a cast fails, not on every call (P28)** -- [part 8](records/PLAN-ARCHIVE-8.md#task-369)
- [x] 371. **A windowed model's ring as contiguous runs of doubles, the co-moment triangle packed in the model, and the…** -- [part 8](records/PLAN-ARCHIVE-8.md#task-371)
- [x] 372. **Task 368 put `bank_seqtest` over its instruction budget; and the conformal grid's ledger tightened** -- [part 8](records/PLAN-ARCHIVE-8.md#task-372)
- [x] 373. **The 7 mutants survivors of aa2e0b17 (task 368's conformal multiplier)** -- [part 8](records/PLAN-ARCHIVE-8.md#task-373)
- [x] 374. **P30 on task 371: a windowed model's view of its window kept between rows and written over, and the…** -- [part 8](records/PLAN-ARCHIVE-8.md#task-374)
- [x] 375. **The conformal code out of `run_instance`'s row loop (P31), and the label run only under `label_clock` (P32)** -- [part 8](records/PLAN-ARCHIVE-8.md#task-375)
- [x] 376. **The suite on the floor of the Polars range, py-polars 1.37.0: 8 failures found before release 1.5.0** -- [part 8](records/PLAN-ARCHIVE-8.md#task-376)
- [x] 377. **A blocked Gram's merge lays each piece out on the thread that multiplies it, `ew_cov`'s `mahal` factorizes…** -- [part 8](records/PLAN-ARCHIVE-8.md#task-377)
- [x] 378. **The mutants survivors of 66481237 (tasks 371, 374 and 375: the contiguous ring, kept windows, packed triangle)** -- [part 8](records/PLAN-ARCHIVE-8.md#task-378)
- [ ] 379. **Fewer reads of the files for `po.gbdt.xgboost_matrix`: replay XGBoost's passes from a cache of bins** -- the user, 2026-11-09 ("Yes"), after 1.5.0. *On hold (the user, 2026-11-16): boosting is picked up after the raw performance work is done.* Measured on 3.4.1 with the native engine's `_LAST_STATS`: XGBoost's `XGQuantileDMatrixCreateFromCallback` iterates the callback three times with a `ref` (counting, indexing, the column matrix) and four without; `sample_rows` (the default) replaces XGBoost's sketch pass with our own sample pass, so the files are read 4 times either way, 3 with a given `ref`. The three passes are XGBoost's and fixed; what each pass reads is ours. (1) **Bin cache:** once the cuts are known (after the sample pass), the iterator bins each value on XGBoost's first pass into one byte a value (`max_bin` <= 256; a wider index above it), keeping label, weight and margin exact, and hands the later passes each value's bin as a float inside that bin (the bin's lower cut; the first bin a value below the first cut; missing stays missing), so XGBoost re-bins it to the same index. Files read twice (sample, bin) instead of four times; peak grows by about a byte a value during the build (about XGBoost's own matrix). **Prove before keeping:** XGBoost's bin rule (upper_bound over the cuts, the last bin clamped) read from its source at the pinned versions 2.1-3.x, and trees, predictions and `save_raw` bytes identical to the bit with and without the cache over weights, margins (one and several a row), missing values, `max_bin` 2/16/256/1024, categoricals refused as now, several chunkings; reads, time (Arrow IPC and parquet, 1e7 x 50) and peak memory measured; the explanation guide's pass counts and the highlights page's "four for XGBoost" corrected. (2) **The ridge's pass draws the sample:** a way to take the cut sample (the same SplitMix64 priority sample, so the same cuts) in the pass that writes the predictions -- e.g. a `po.gbdt` sampler a query sinks through, returning the `ref` -- so the files are read once after it. (3) **When the data fits** (the LightGBM rule: features as float32 at most a quarter of available memory): cache float32 instead, one read. **Then the highlights page gets the trees task back** (the user, 2026-11-09: highlight trees only if the library builds them in one pass; taken off in d4ca0719), as a one-pass recipe: data that fits, with the ridge's query fed straight to `po.gbdt` (LightGBM's in-memory route; XGBoost by (3)), or cuts from an earlier matrix's `ref` (XGBoost by (1)). LightGBM keeps its own sample unless the user decides otherwise (an open question: bins from our sample would let LightGBM's two passes become one after the ridge's, but would change every LightGBM model's bins).
- [x] 380. **`po.eval` reads a query, streaming, so no example loops over chunks in Python** -- [part 8](records/PLAN-ARCHIVE-8.md#task-380)
- [x] 381. **The per-row `diebold_mariano` / `clark_west` (`half_life=`) read null where `S` is zero but for rounding…** -- [part 8](records/PLAN-ARCHIVE-8.md#task-381)
- [x] 382. **The `po.eval` review's fixes** -- [part 8](records/PLAN-ARCHIVE-8.md#task-382)
- [x] 383. **The example data moves from the README to its own page** -- [part 8](records/PLAN-ARCHIVE-8.md#task-383)
- [x] 384. **The mutants survivors of 27086fda (task 377's blocked Gram merge and `mahal`'s ridged solve)** -- [part 8](records/PLAN-ARCHIVE-8.md#task-384)
- [x] 385. **`emit_breaks` costs many times the fit it tests** -- [part 8](records/PLAN-ARCHIVE-8.md#task-385)
- [ ] 386. **A study of the performance of every component and feature, with profiling tools and Valgrind** (after 1.5.0; the user, 2026-11-10: "Following 1.5 I want to do a large study of performance of every component and feature both using profiling tools and valgrind. Add this to the plan but do not plan the study until after 1.5"). *Measured 2026-11-10..13* (the performance week, measure only: Mac batches M1-AB7, VM batches V1-V12 and the VM's R-N2 of 2026-11-14; the index `.claude/coord/perf-week/results/INDEX.md`, the published findings page). *Planned 2026-11-14* (the user: "keep working on the performance items"; "use the lowest cost model for each task that will not compromise quality"). Rules for every sub-task: outputs and state bytes unchanged to the bit unless the user decides otherwise (the digests on both platforms, `scripts/compare_release.py`); timed on arm64 (the Mac) and x86-64 (the VM, both layouts), a case slower on either beyond noise is a defect; instruction counts from the VM; a budget that moves down is tightened (`scripts/budgets.py apply`). Each sub-task names its worker model: Sonnet for a bit-identical change with a clear target or a measurement, Haiku only to re-run a fixed harness, Opus only where a numerical argument decides correctness. The sub-tasks, by the size of what they buy:
  - 385 `emit_breaks` (the largest cost: 16-86x the fit), bit-identical first; then a faster form that moves the statistic's last bits, for the user to decide with its measured error.
  - 387 `hmm` k=20's 2% (the per-row triangle unpack).
  - 388 P34 (the VM's `vm/p34-held-record`) and the save and load peaks (a save 2.3-3.7x the file, a load 3.5x).
  - 389 `bocpd`: hold what depends on the run length alone across rows, and a gamma cache with no collisions (AB1 (d) 1.3-1.8x x86, 1.1-1.6x Mac; AB3: 16-25% of its libm calls depend on the run length only).
  - 390 P36: `audit`'s digest sorts only the new values and merges (46% of its mispredicts in the stable sort).
  - 391 The fixed cost of a call across the Python boundary (x86 120-140 µs against ~20 µs of Rust work; found to be the per-call frame query and pyo3-polars' conversion of each output column, not a module import).
  - 392 `emit_specification` and `emit_robust_se` past the readiness gate (+49.9k / +6.9k instructions a row on x86): where the work goes, by line, then a cheaper exact form (VM analysis first).
  - 393 A formula target's window pass, serial in `resolve_targets` (90% of a formula target's 1.59x).
  - 394 `lasso` with several penalties (17x one penalty at k=500): warm starts along the path move the iterates within the stopping rule, so the user's decision, with the measured movement.
  - Not pursued: `ewridge`'s rank-one update in place of a refactor (moves bits; on x86 slower than a refactor at k <= 8, R-N2); the layout differences (noise over 10 pads, V5); the Mac's F_FULLFSYNC on save (a durability guarantee, not a defect).
- [x] 387. **`hmm` at 20 features is about 2% slower than 1.4.0** -- [part 8](records/PLAN-ARCHIVE-8.md#task-387)
- [x] 388. **P34 and the save and load peaks** -- [part 8](records/PLAN-ARCHIVE-8.md#task-388)
- [x] 389. **`bocpd`'s per-run constants and gamma cache** -- [part 8](records/PLAN-ARCHIVE-8.md#task-389)
- [x] 390. **P36: `audit`'s digest fold sorts only the new values** (task 386; worker Sonnet, high). Bit-identical, ties included. *Built* (2026-11-15, W176; the VM's three commits `a9928d80`, `39edb051`, `6b7f3cce` merged, then reshaped): `Digest::fold` sorts only the buffered values and merges them into the sorted centroids (a tie keeps the centroid first, as the stable sort did; unsorted centroids from a merge or a damaged file are stable-sorted first). On top of the VM's: the buffered values are sorted unstably as `f64` in place (two values that tie under `total_cmp` are the same bits and both weigh 1, so no sort can show its stability) and the merge picks its side with a select, not a branch. *Bit-identical*: a digest of the `audit` frames (columns, pairs, clock; pooled and not) and the saved bank by `to_bits` over k = 1, 4, 9 columns (continuous, ~300 distinct with `-0.0`, few values with ties, ~1,200 distinct), 1 and 100 groups, chunks of 20,000, 777 and 1,000 rows and a save and load at row 10,000, equal on main's build and this; the VM's tie test (`the_fold_keeps_the_stable_sort_order_on_ties`) holds the fold to the old stable-sort fold. *Time, arm64* (release, ABBA, 6 rounds, load 3-4; `fit_predict`, 20 continuous columns, 256 counters, ns a row, min): main 1,069 -> the VM's merge 1,035 (-3.2%) -> this 1,010 (-5.5%); counters (CPU Counters, whole process, per row): cycles 5,409 -> 5,038 (-6.9%), instructions 18,558 -> 17,568, mispredicts 61 -> 54 (the VM's merge alone: 70, its merge branch is as unpredictable as the sort's). The survey's 28% for the sort was not there on arm64: the loop is branch-bound (IPC 3.5, 16% of cycles in mispredicts) and what remains is the Misra-Gries decrement pass and the moments, not the fold. *The map sized once*: declined, measured: `reserve(cap)` on a column's first value is no faster (1,013 vs 1,013 ns a row, 5 ABBA rounds): the map grows only while it fills, and past the cap it is the decrement-and-retain pass, not growth, that costs; pre-sizing would also cost every low-cardinality column a table for `cap` values.
- [x] 391. **The fixed cost of a call across the Python boundary** (task 386; worker Sonnet, medium). Find and cache each per-call import or lookup; bit-identical. *Built* (2026-11-14, W153): the `builtins._import` in a profile of the native call is not a module import but pyo3-polars' `Series._import` (Arrow import of each output field, ~5.5 µs a call, in the dependency); no `py.import` runs per call (the `PySeries`/`PyDataFrame` statics are cached by pyo3-polars, ours run only on error paths). What was per-call and need not be, all in `python/polars_online/_bank.py`: the output frame built by `with_columns` of `pl.Series(s)` copies (a one-node lazy query planned, optimized and collected, ~25 µs) now `df.hstack(outs)` (the extension already refuses an input column named as a spec, and spec names are unique); `_check_frame` built a `pl.Schema` and a clock set every call, now scans `df.dtypes` and builds both only for a `UInt128` refusal; `_warn_notices` imported `_frame` every call, now only when a notice is waiting. Mac, one row ewridge k=5: min 50.6 -> 39.1 µs, p25 54.9 -> 42.8; 3 specs: p25 222 -> 150; 1,000 rows: 387 -> 375 (no slower). Digests of fit_predict/predict outputs over two banks and 1/10/1,000 rows equal to main's. Left: the per-column Python round trip in pyo3-polars' `PyDataFrame` extract and `Series._import` (~20 µs), and the pool hand-off (~13 µs), which are not ours to change small.
- [x] 392. **`emit_specification` and `emit_robust_se` past the readiness gate** (W154, slot 2, branch `task-392-spec-robust-se`, at main c7483274; scratch `.claude/coord/W154-392-scratch/`). *Found* by the performance week: past `diag_ready` (4.32 half-lives) `emit_specification` cost 6-7x the base row and `emit_robust_se` +6.9k instructions a row on x86. *Re-measured* at main (release probe, ewridge k=20, half-life 1,000, rows 6,000-20,000, ns a row, base 345-396): specification +2,700-2,850, robust +300-380; k=4 +1,165; under a horizon of 5 rows (`horizon_rows`) specification +26,700 and robust +3,500. *Where it went* (Time Profiler `sample` on a symbol build): the three statistics each formed a matrix, right-hand side, factor and scratch anew on every row (`Specification::{r2,slopes,ljung_box}`: `Mat::from_fn` copies, a `MemBuffer`, the `cov(i, j)` entries pushed one by one into a growing `Vec`); `Sandwich::update` aged every sum in one pass and added the row's term in another and allocated `z`, `u` and the ring's new score each row; under a horizon `Sandwich::covariance` (Breusch-Pagan's and RESET's Wald statistics, every row) picked each of the lag sums into a matrix, factorized the bread with `SpdFactor::of` and solved twice into fresh vectors. What remains at k=20 without a horizon is faer's Cholesky of the 20x20 (Breusch-Pagan's `R²`), which no bit-identical form removes. *Built* (every output and state byte unchanged): the reads write the lower triangle of each system into buffers a thread keeps (`ReadScratch`, `CovScratch`) and solve in a kept `SolveWork` (`solve::{solve_in, kept_work, factorize_kept, solve_kept}`; `solve_in` moved from `breaks.rs`); `Sandwich::update` folds each sum's decay and term in one pass (`S ← λS + term`, the same two roundings), forms `z` and `u` in thread buffers and recycles the ring's oldest buffer; the lag sums are read where they lie and added to the meat once for the pair `(i, j)`, `(j, i)` (`b (g_ij + g_ji)` is the same number either way round); one factor serves both solves. Tests: each form against a copy of the old one, to the bit (`a_row_in_one_pass_is_the_row_aged_then_added_to_the_bit`, `the_covariance_in_kept_buffers_is_the_allocating_one_to_the_bit`, `the_reads_in_kept_buffers_are_the_allocating_reads_to_the_bit`). *Bit-identical:* a digest of every output column by `to_bits` and of the saved bank, with a save and load at row 10,000, over 72 cases (k = 1, 4, 20; horizon 0, 3, 5, 12; specification, robust, both) equal on main's build and this. *Time, arm64* (release, ABBA, 5 rounds of two runs each, load 6-7; ns a row, minimum / p25, main / this): specification k=20 3,056 / 3,075 -> 1,639 / 1,644; robust k=20 677 / 679 -> 529 / 543; both 3,367 / 3,376 -> 1,813 / 1,821; specification k=4 1,514 / 1,530 -> 520 / 522; specification k=20 horizon 5 27,024 / 27,074 -> 12,940 / 12,997; robust k=20 horizon 5 3,845 / 3,872 -> 1,319 / 1,328; both k=4 horizon 5 10,734 / 10,814 -> 3,136 / 3,154; no diagnostics 345 / 356 -> 347 / 356. x86 is the VM's.
- [x] 393. **A formula target's window pass in parallel** -- [part 8](records/PLAN-ARCHIVE-8.md#task-393)
- [x] 394. **`lasso` along several penalties** (task 386; measurement first, Sonnet medium; any change that moves outputs is the user's). *Decided 2026-11-15 (the user: "Decline both"):* the warm start from each penalty's own last fit and the strong rules are declined; the sparse sweep (bit-identical) is the task's result.
  - *Measured and partly built 2026-11-14 (W155, slot 3, branch `task-394-lasso-path`).* The solve already warm-starts along the path (each penalty from the next larger one's fit, the first from the last solve's largest-penalty fit), so the cost is sweeps times the sweep's width: at k=500 and ten penalties a solve runs 620 sweeps, the middle penalties 80-88 each (about 40% of those descents stop at `max_iter` 100) with 300 of 500 coefficients nonzero, and a solve falls due every 16-20 rows under the default share. Built, bit-identical (`cd_sweep_sparse`): from 64 features, while at most three quarters of the coefficients are nonzero, a sweep sums only the nonzero ones, in index order (the left-out terms subtract exactly 0 while every correlation is finite; otherwise the dense sweep runs); Mac ABBA, min of 5 rounds, 1 / 3 / 10 penalties: k=100 1.77 / 1.19 / 1.18x, k=200 3.2 / 1.40 / 1.42x, k=500 5.45 / 1.77 / 1.82x, k=20 unchanged. Measured and not built (they move outputs within the stopping rule, the user's call): each penalty restarted from its own last-solve fit is no faster (denser starts turn the sparse sweep off) and moves predictions by up to 4e-10 relative past warm-up; a working-set rule with the sequential strong rule's KKT check is vacuous at the usual grids (ratio of 2 or more between penalties: bit-equal to the dense descent) and slower when forced.
- [ ] 395. **The rename to `rheon`** (the user, 2026-11-14, planned with the VM session; recorded on the VM exchange as R-N1; not started: "we will work on moving the package name later"). The package becomes `rheon` (held on PyPI by a forwarding 1.5.0 that depends on `polars-online==1.5.0`, published from `github.com/rheon-dev/Rheon-shim`; the GitHub organisation `rheon-dev`); the Polars interface `rheon.polars` (database adapters later as `rheon.duckdb` and so on). One task, one branch, nothing else in flight: crates `online-*` to `rheon-*`, the native library `_rheon`, the CLI binary `rheon`, `python/rheon` with `rheon/polars/`, docs, CI, scripts and CLAUDE.md. Every example (README, guides, docstrings, `examples/`, the tests that read them) writes `from rheon import polars as rp` and calls `rp.`, where it now writes `import polars_online as po` and `po.` (the user, 2026-11-14). Kept: `BANK_MAGIC` and `ModelState`'s variant names, so every 1.x state loads without a schema bump. Environment variables `RHEON_*`, the public `POLARS_ONLINE_*` read with a warning until the next major. Outputs bit-identical to 1.5.0 on both platforms; counts re-baselined only with their cause. At release (2.0.0): the repository moves to `rheon-dev/rheon`, a second trusted publisher on PyPI's `rheon`, then a last `polars-online` forwarding to `rheon`.
- [x] 396. **Every exception type the library raises is tested** -- [part 8](records/PLAN-ARCHIVE-8.md#task-396)
- [x] 397. **`ew_class` factorizes each class from the packed triangle, with no square laid out** -- [part 8](records/PLAN-ARCHIVE-8.md#task-397)
- [x] 398. **The plan split: finished work to an archive under `docs/records/`** (the user, 2026-11-14: "Queue it", on the coordinator's measurement: PLAN.md 1.73 MB and 17,052 lines against `test_repo_hygiene`'s 2 MB `MAX_DOC_BYTES`, about 700 KB grown since 2026-10-09; 375 of 401 tasks done; no reader can hold the file whole). Kept in PLAN.md: the design (§1-10), the open tasks, the decisions still in force, and one line a finished task (number, title, archive link). Moved, text unchanged, to `docs/records/PLAN-ARCHIVE-*.md` (split so each file is under the source cap): each finished task's full entry and the closed review and design sections (§11a onward), each leaving its heading and a link so a citation by number (CLAUDE.md's "PLAN §18 D1", code comments, the CHANGELOG, records) still resolves. Checks: every `PLAN §N` and `PLAN task N` reference in the repository resolves (a test that holds it); `test_no_plan_references`' file lists still hold; the archive is filed in `docs/README.md` among the records; `MAX_DOC_BYTES` lowered to fit the new PLAN with room. Run alone, after the wave in flight merges (every worker branch ticks PLAN). Worker: Sonnet, medium (mechanical; the coordinator checks the references).
  - *Built 2026-11-14 (W152, slot 1, branch `task-398-plan-split`).* `docs/PLAN.md` fell from 1,755,734 bytes (17,055 lines) to about 143 KB (about 1,570 lines):
    the header, §1-§10, §11b-§11h, §12 and the "Follow-on documents" section as they were; the 21 tasks then open in full; one line for each of the 383 finished tasks
    (`- [x] N. **Title** -- [part K](records/PLAN-ARCHIVE-K.md#task-N)`); each closed section's heading with a line saying where its text went; and the notes of §25 and §26,
    the parked and dropped decisions of §11, which are still in force. The rest, word for word and in its original order, is in eight files, `docs/records/PLAN-ARCHIVE-1.md`
    to `-8.md` (117 KB to 236 KB each, under the 240,000-byte cap): every finished task's entry anchored `#task-N` and the bodies of §11 (the notes between its tasks),
    §11a and §13-§24, anchored `#sNN`. A scratch script (`.claude/coord/W152-398-scratch/verify.py`) holds that no line was lost: the old plan's non-blank lines equal
    the new plan's kept lines plus the archives' body lines as multisets (16,288 each), the archives' lines read in the old order, and the only lines outside the count
    are the pointer lines, the added header paragraph, the archives' file headers, anchors and heading copies. Citations needed no edit: a task number resolves through its pointer
    line and a section number through its kept heading. `tests/test_plan_references.py` (essential) holds that every `task N` cited in the repository's text files has a line in the plan,
    every `PLAN §N` a heading, each pointer reaches its anchor, no archived task lacks a pointer, and each archive is under the cap. `docs/README.md` files the archive;
    `MAX_DOC_BYTES` fell from 2,000,000 to 650,000 (CHANGELOG.md, 579 KB, is now the largest prose file; the plan's size plus 300 KB, 450,000, would refuse it). Not moved, by choice: §25's and §26's notes, which sit beside open tasks. The next split moves a task's entry the tick after it is built.
- [x] 399. **Every `OSError` the Rust side builds carries `errno`, `strerror` and `filename`, as Python's `open` gives them** -- [part 8](records/PLAN-ARCHIVE-8.md#task-399)
- [ ] 400. **`lasso`'s descents stop at `max_iter` at default settings past 200 features** (found by task 394's measurement, 2026-11-14, W155). On `wide_benchmark.make` with `half_life=1000` and 10 penalties, 40-70% of the middle penalties' coordinate descents at k=200 and 500 reach `max_iter`=100 before the stopping rule's `tol`=1e-10, so those solves count as `solve_failures` and their coefficients depend on where each descent started (the warm-up's first ~50 rows move 0.2-0.5 between starts). Find whether this is the data (near-collinear features, a long path), the defaults (`max_iter`, `tol` relative to the target's spread), or the descent (cyclic order, no active-set phase); what a user sees (`solve_failures`, a readiness gate); and whether a default should change. Any change that moves outputs is the user's. Worker: Sonnet, medium (measurement and reading first). *Measured 2026-11-15 (W160; `.claude/coord/W160-400-scratch/`): the cap hits are the warm-up, not the defaults or the descent.* Every one falls in the first rows, while the weighted Gram is near-singular (k=200: cond 611 at 230 rows, 18 at 500; k=500: 8,900 at 520 rows, 7 at 3,000); past about 400 rows at k=200 and 1,000 at k=500 none hit the cap, and a steady-state solve takes 17-25 sweeps (4x margin). The converged fit matches scikit-learn's `lasso_path` on the same Gram to 1e-9 of the spread; ours takes 0.5-0.6x its sweeps. Stopping at 100 moves predictions by up to 0.26 of the spread in rows 200-300 only (k=200), 3e-6 by rows 300-400, nothing after; out-of-sample MSE equal either way, and the warm-up fit is poor converged or not. Options measured: `max_iter` 1,000 / 10,000 costs 4x / 26x in the warm-up and still counts failures; a looser `tol` moves steady-state outputs; an active-set phase and other coordinate orders are no faster (the fits are dense). Recommendation: keep the numbers; the open question is the user's: whether a cap hit while the target's Kish n is below a few thousand stops counting as a `solve_failure` (today every wide `lasso` warns from its warm-up), or the warning is documented as expected there. *Decided 2026-11-15 (the user):* "Don't count warmup but be sure that the row count for warmup is not based on test data but on an independent measure": a cap hit stops counting as a `solve_failure` while the fit is in its warm-up, where the warm-up is defined by a property of the solve itself (the Gram's conditioning, or the effective rows against the features by a stated argument), not by a row count tuned on the benchmark; built as task 405.
- [x] 401. **CI red on d6537159: a pool thread's stack overflow in a formula target's window pass, and a new clippy lint** (the coordinator, 2026-11-15; W156, slot 1, branch `fix-ci-stack`, at main 46a8aa78; brief `.claude/coord/W156-ci-stack-and-lint.md`, scratch `.claude/coord/W156-fix-scratch/`). The debug-build `cargo test --workspace --exclude online-py -- --include-ignored` aborted in `tests/formula_targets.rs` on all three OSes ("thread 'polars-online-N' has overflowed its stack"), and the lint job's stable moved to rustc 1.100.0 (the pin is 1.99.1), whose clippy flags `needless_range_loop` where 1.99's did not.
- [x] 402. **The mutants survivors of d6537159 (task 392's code)** (W157, 2026-11-15; CI Mutants run 43674717710: 305 mutants, 289 caught, 14 survived). 13 killed, 1 equivalent, 0 open. `sandwich.rs` `update`'s four `*` survivors (the finiteness check of `λS + t`, lines 213-214): `a_row_is_judged_by_the_sums_as_the_decay_leaves_them`, a second row at `λ = 0.5` whose sums fit only once decayed (1e308 sums, each under `f64::MAX` as `λS + t`, over it as `S + λ + t` or `S/λ + t`). The kept-buffer guards (`with_cov` `>=`, `covariance`'s `+` in the order, `solve_in`'s `>` three ways, `kept_work`'s `Box::leak`, `==` and `>=`, `with_scratch`'s `>=`) were equivalent for outputs, not for memory: tests read the thread's scratch after a read of order 63, 64 and 65 (`the_covariance_scratch_is_kept_below_the_order_and_released_at_it`, `the_read_scratch_is_kept_below_the_order_and_released_at_it`, `the_kept_spaces_are_bounded_by_order_and_count`, which also checks a space is found again by address and that a seventeenth shape clears the sixteen). The equivalent: `ljung_box_in`'s `+` to `-` in Bartlett's `gamma`, already argued in `scripts/mutants_equivalent.toml`, under the function's name before `ljung_box_in` was split off; the entry's `function` is renamed (the count of 380 stands). Tests only; no `src/` change outside `#[cfg(test)]`.
- [x] 403. **The window operators' per-row kernel** (task 386; W158, slot 2, branch `task-403-window-kernel`, at main 46a8aa78; scratch `.claude/coord/W158-403-scratch/`). *Found* by the performance week (N8, P1): `po.ewm_mean` through `with_windows` ran 129 ns a row on one thread against Polars' 12 (7.1x at 1M rows, 11x at 16M), 88% of it in `Windows::push`, `step_clocks` and the clock's `step_any`. *Re-measured* at main with a native harness over `WindowsRun::feed` (4M rows, 1 op, load 6-7): datetime clock 111 ns a row, float clock 99, no clock 93, datetime with `window_size="60s"` 162, four operators 162. *Profile* (Instruments Time Profiler on a symbol and line-table build, datetime clock, one operator): `step_clocks` 27% inclusive (`step_any` 14%, the clones and the assignment of the two `ClockState`s, about 200 bytes each, four copies a row through libc `memmove`: 11% of `push`'s own time sat on those lines), `flush_stamps_before` 18% (`elapsed`, a nanosecond-to-seconds conversion, 6%), `seconds_of_ns` called five times a row (two steps, the stretch's origin, the silent list's head, the stamp test), `expm1` 12% (the mean's held mass, required), `__fixdfti` (a float-to-128-bit conversion in the cap test, hoisted into every row). *Built*, bit-identical: (1) `ClockState::plain_forward` and `take_plain_forward` split `step_forward_unstamped` into deciding and taking, so a row that is a plain step forward on both the stream's clock and the group's (never refused) steps each in place, with no copy and no reassignment; any other row takes the copies as before. The short cut now covers a temporal clock's step and a row-count clock's step past its first row (`step_any` took both before), and `plain_forward_after` takes the step the stream's clock decided for the group's clock where both stood at the same previous clock value to the bit (one conversion a row, not two). (2) `past_cap` (the silent list's head against `gap_cap`) answers a temporal step shorter than the cap's whole seconds from the integers, with no conversion (the conversion never rounds past the next whole second), the 128-bit comparison out of line; `is_after` (the stamp test) compares the integers. (3) `KernelDef::discount(0)` is 1 without a call (every read of a backward window at its newest row; `exp2(-0)` is 1 for the half-lives above 0 every kernel has). (4) A queue's operators are folded at the same two anchors, so `then_by` takes the discount once a row, not once an operator, in `push` and in the turn-over; the mass of a held value's distance and a read's discount are kept for the next operator asked the same (`Last`). (5) `link_back` leaves a group that is already the tail; a chunk with no group column looks the one group up once. *Result* (native harness, 4M rows, load 6-7, ns a row, main then branch): datetime clock 111 to 64, float clock 99 to 61, no clock 93 to 46, datetime with `window_size="60s"` 162 to 116, four operators (mean, std, sum, var) 162 to 104, four operators 10,000 keys 256 to 184. Wheels, ABBA of the release wheels of main and this branch, 5 rounds (10 runs each), `with_windows(...).sink_parquet` over 4M rows, min of 10: datetime `ewm_mean` 0.506 s to 0.273 (1.85x), with `window_size="60s"` 0.726 to 0.495 (1.47x), float clock 0.440 to 0.272 (1.62x), float with the window 0.652 to 0.483 (1.35x), four operators 0.771 to 0.487 (1.58x), four operators with the window 1.306 to 1.093 (1.19x), four operators one key 0.836 to 0.546 (1.53x), 10,000 keys 1.228 to 0.915 (1.34x); load 4.6-7.7; every output column's digest equal between the two wheels. *Bit-identity*: a digest of every output column (`to_bits`) and of the state bytes at the end and at a mid-stream save, across 118 cases (datetime, float, integer and no clock; ungrouped and 7 keys; running, a long and a short `window_size`; `closed` right, left, both, none; `ewm_mean`, `ewm_std`, `ewm_var`, `ewm_sum`, `ewm_rate`, `ewm_quantile` and the forward `rewm_mean`, `rewm_sum`, `rewm_quantile`; a second input with other nulls; sessions with a reset; a restart after a step back; a refused step back) at four chunkings (one chunk, six of 1,000, an uneven six, two of 3,000) with a save and load after the second chunk, equal between main's build and this branch's, and each loaded run re-saving its own bytes. Left, measured: a `window_size` is still the larger half of a windowed row (the two-stack queue's turn-over and its two sets of seven vector pushes; one instruction after the `exp2` of the row's discount holds 18% of a windowed run's samples, the chain `tau` to `fdiv` to `exp2` to the fold being serial a row at a time), the frame's building and assembly (about 14% of an unwindowed row), and the mean's `expm1` (12%).
- [x] 404. **`rcov`'s per-row cost was the lagged-product loop, not an output** (W159, 2026-11-15; slot 1, branch `task-404-rcov`). The performance week's "emit" is `Rcov::emit`, one final effective return into `Γ̂_h` for each of the `ring` lags (336 at the budget case's `block_rows` 2000), `k²` products each; the group-close estimate is once per block and not the row cost.
- [x] 405. **`lasso`: a cap hit in the warm-up is not a `solve_failure`, by an independent measure** (task 400; the user, 2026-11-15). See task 400's decision. Worker: Sonnet, high. *Built* (2026-11-15, W161, branch `task-405-lasso-warmup`; scratch `.claude/coord/W161-405-scratch/`): `gramfit::warming_up` (`gramfit/cd.rs`, the argument in its doc comment) decides, for a descent that ran out of `max_iter` sweeps, whether the Gram's rows are too few for its features. *The measure is the Kish size against the feature count, not the Gram's condition:* a condition number is the same whether too few rows or a dependent pair made the Gram ill-conditioned, so a rule from it would excuse a near-collinear pair that appears late; the row count separates them (more rows cure one and not the other), and it costs O(k) on a cap hit and nothing otherwise (no Hager estimate, no factorization). *The argument:* a sweep is a Gauss-Seidel sweep, contracting the error by `1 - 2 lmin` (the iteration's spectral radius, against `1 - 2 lmin`, measured on sample correlation matrices of 50 and 200 features at 1.2-10 rows a feature: log rate within 25% above it where `lmin <= 0.35`, never below); settling to `tol` takes `ln(1/tol) / (2 lmin)` sweeps (about 0.77 of that on the test Grams, never over); `n` independent rows give at best `lmin = (1 - sqrt(k/dof))^2` (Marchenko-Pastur edge, Bai-Yin), `dof = n - intercept`, plus the penalty's `l2` share (`(edge + l2)/(1 + l2)`); warm-up while that is below `ln(1/tol) / (2 max_iter)`. At the defaults 2.29 rows a feature, at 1,000 sweeps 1.25, tending to 1 as `max_iter` grows; where `ln(1/tol)/(2 max_iter) >= 1` (under about 12 sweeps at the default `tol`) no row count meets it and every cap hit counts. No constant fitted to a data set: W160's 400 rows at k=200 and 1,000 at k=500 are not in the code. *Validated* on data it was not chosen on (`val.py`, the wheel at main against the wheel at the branch, 25-row chunks, `half_life` 1,000, ten penalties, k = 50, 200, 500; rows 2,000 / 3,000 / 3,500): independent features with a sparse truth, AR(1) designs at 0.5 and 0.9, Student-t(3) noise, and BTCUSDT 1-minute return lags from `tests/data.py`. The last excused chunk falls at 1.5 to 2.35 Kish rows a feature in all fifteen cases (the rule's 2.29 plus the chunk's width); every cap hit counted lies at 2.31 or later; excluded counts 521 / 1,216 / 1,792 on independent features go to 0; AR(1) 0.9 keeps 2,052 / 1,786 / 1,404 counted (`lmin` is too low for 100 sweeps at any row count: the user is told), AR(1) 0.5 keeps 12 at k=50 and none beyond, the real lags keep 0 / 24 / 42 (heavy tails). Predictions and every output column are unchanged: a digest of every float column of the output struct and of `coef()` matches main to the bit on three streams (independent k=50, real lags k=200, AR(1) 0.9 k=50), and the per-row selected prediction matches on all fifteen. The converged fit (10,000 sweeps) is no better on the excused rows: out-of-sample squared error of the capped fit against the converged one, k=50: 0.498/0.544 independent, 0.426/0.487 AR(1) 0.5, 0.317/0.459 AR(1) 0.9, 2.20/2.52 heavy-tailed (the capped fit is the lower in each; the real lags 7.1e-7/2.4e-6); k=200 (reference 10,000 sweeps) and k=500 (reference 1,000, the 10,000-sweep run costing 26x in the warm-up) equal to five digits in all ten cases (the selected penalty's descents are the ones that converge). *A cap hit past the warm-up is still counted:* a pair made near-collinear from row 2,000 of 8,000 (k=50; `x1 = x0 + 1e-5 z`) counts 354 hits, all from row 5,600 on, where the pair has taken over the Gram, with none before it (main counts 894: those 354 and the 540 of the warm-up). Rust tests: the edge from the settings alone, the rule's edges, the claims against faer's eigensolver and the descent itself (`gramfit/tests.rs`); a warm-up with real cap hits not counted, and a late near-collinear pair counted, over a stream with none without it (`lasso.rs`). `solve_failures` is in the saved state (a `pub` field of `Lasso`): the layout is unchanged, no schema bump, and a state saved before holds its old count. `solve_failures` falls by the warm-up's cap hits (k=200, `half_life` 1,000: 1,216 to 0 over 3,000 rows); `lasso` with a `max_iter` under about 12 counts as before. The `solve_failures` check's threshold (more than 10, `info`) is unchanged; its reason (clean data ran out up to 7 times on its first rows) is gone, and a lower one is a policy for the user. `tests/test_convergence_equivariance.py`'s lasso count test used the default descent's first-row cap hits and now uses a 2-sweep cap.
  - *Cause: nested work-stealing, not a fixed depth* (task 393 put each group's resolution on the bank's pool with `par_iter_mut`). A group's step (`DataFrame::take`, then the lazy engine) hands work to Polars' own pool by `ThreadPool::install` and waits; a rayon worker that waits for a job of another pool runs other jobs of its own pool on top of its stack. So the thread blocked in one group's step took another group's job from the bank's queue, blocked in it, and took another: a backtrace at the deepest entry shows `resolve_group` -> `group_frame` -> `apply_columns_par` -> `in_worker_cross` -> `wait_until` -> `execute` -> `resolve_group` and again, 9 times in 300 groups on two threads. Measured (the stack used when a step is entered, from the thread's start): debug, two threads, 3 / 50 / 100 / 200 / 400 groups 0.06 / 0.34 / 0.79 / 1.02 / 1.86 MB, then an abort at 800; four threads 0.06 / 0.24 / 0.43 / 1.11 MB at 3 / 50 / 100 / 200, an abort at 400; 2,000 and 20,000 abort at both. Release, 200 / 2,000 groups 0.22-0.29 / 1.6 MB, an abort at 20,000 on two and on four threads (and a release wheel's own `fit_predict` of 10,000 rows in 1,500 groups died of a bus error on the default pool). It is non-deterministic: which job a blocked thread takes depends on timing, so `POLARS_ONLINE_MAX_THREADS=2` and `=4` overflowed on 3 of 4 runs of the existing test (200 groups), and 3 threads (one run) and 14 did not; one thread nests too (a one-thread pool aborted at 2,000 groups). The depth grows with the groups, so a larger pool stack only moves the limit; the fix is not to nest.
  - *Fix, bit-identical* (`resolvers::in_parallel`): the groups are not split into jobs. The caller and up to one job for each other thread run a loop that takes the next group from a shared atomic counter (each core behind a `Mutex` locked once, the results read in the groups' order, so the first refusal is the one a serial pass reports); a job that finds its thread in the loop already (a thread-local flag; one it took while blocked in a step) returns at once, and the loop that thread is in takes the groups left when its step returns; the caller's loop runs whatever its thread was doing, so every group is taken. A thread then runs one group's step at a time: the stack at a step's entry is 61 KB at 3, 200, 2,000 and 20,000 groups on two and four threads (debug). Held by `thousands_of_groups_on_a_small_pool_do_not_nest_to_the_stack_s_end` (2,000 groups, 60,000 rows, a pool of one thread and one of two through `Bank::set_threads`, equal to the bit and in the saved state), which aborts on the old code and passes on the new; CI's run has no `POLARS_ONLINE_MAX_THREADS`, and this test needs none.
  - *Speed and bits* (release wheels, arm64, ABBA, 3 reps a process, load 7-11): 1,000 groups, 1M rows, k=5, `ewm_sum` reverse with embargo: main min 0.181 s (median of the rounds 0.200) -> 0.166 s (0.175); 200,000 rows in chunks of 10,000 0.183 -> 0.158 s; 10 groups 0.116 -> 0.115 s (level). The loop is a counter where `par_iter` split and stole. A digest of every output column and saved state (task 393's, 36 cases a group count, four chunkings, a save and load mid-stream) at 1, 7 and 200 groups, two banks, three bare formulas and a refusal is equal at main and here; 1,500 groups, which main cannot finish, completes here.
  - *The lint*: `kalman.rs`'s `for j in 0..m` over `self.basis[j]` is `iter_mut().enumerate().take(m)`; 1.100's clippy found five more in test code with `--all-targets` (`marglag.rs` twice, `window.rs`, `column.rs`, `windows.rs`), each rewritten as an iterator with the same assertions. Clippy `--all-targets -D warnings` is clean on 1.99.1 and 1.100.0.
  - Left: `docs/how-to/use-more-cores.md` says the bank's work never calls back into Polars' threads, which a formula target's window pass does (each step waits on them); `pool.rs`'s module doc now says so, the how-to is a reader-facing page and is the coordinator's to reword. Any other `par_iter` of the bank that waits on Polars would nest the same way (none found: the others are arithmetic and Arrow reads).
  - *Cause*: the lag loop indexed the ring (`fin[len − h]`, a `VecDeque` wrap computation per lag) and walked rows with a run-time width, so at `k` 4 each lag's 16 products were not unrolled.
  - *Built, bit-identical*: `add_lagged_products` (`rcov.rs`) walks `gamma`'s blocks with `fin.iter().rev()` and dispatches the common widths (1-8, 10, 16, 20) to a const-generic loop; other widths use the plain loop. Each `gamma` entry still takes the same single product per emitted return, so the bits cannot move; `estimate`'s trailing jitter uses the same function. New test `the_lagged_products_are_the_definition_at_every_width`.
  - *Mac (arm64, release, ABBA, 6 rounds, load 9-10)*: budget case k 4, block 2000: min 2,395 -> 624 ns/row (3.8x); k 2: 996 -> 316; k 20: 24,154 -> 16,266 ns/row (1.5x). A digest of the estimates (every float `to_bits`) and the msgpack state over k 1-20, five kinds/settings, jitter 1-3, with breaks and a save/load at mid-stream: equal on main and branch.
  - *Not done*: k 20 stays bound by streaming the 1 MB of `Γ̂` once a row; a deferred batch of emits (several rows' products added per pass, in row order, flushed before a close, a save or a break) would stay bit-identical and cut it further, at the cost of a buffered state.
- [x] 406. **A formula target with computed inputs or arithmetic around its operators: no Polars query per group per chunk** (task 386; W162, slot 1, branch `task-406-formula-computed`, at main a0ef3cdf; brief `.claude/coord/W162-406-formula-computed.md`, scratch `.claude/coord/W162-406-scratch/`). *Re-measured at main* (release wheels, arm64, `fit_predict_batches` over 100,000 rows, ewridge k=5 with one forward operator, ABBA x5, load 11-18 unless noted; `bench2.py`): task 393's short cuts left every formula that is not a bare operator over a column of doubles -- `ewm_mean(y) - y`, `ewm_sum(y * x)`, a cast, two operators combined -- two plans of Polars' lazy engine per group per chunk (the operator inputs, then the formulas over the resolved rows) and, since the stack fix, a `DataFrame::take` per group that hands its columns to Polars' pool and waits. At 1,000 groups in chunks of 2,000 rows (2 rows a group a chunk): arithmetic around one operator 0.653 s (6.5 ms a chunk), computed input 0.456, two operators 0.635, cast 0.846, bare operator 0.299; at 100 groups 0.15-0.16 s; at 1,000 groups in chunks of 50,000, 0.074 s; one group, 0.057-0.068 s whatever the shape.
- [x] 407. **The RSS-plateau tests hold on a busy machine** (the full gate on 65094049 failed `test_sliced_frame_sharing_buffers` at a load of about 14; the same test passed alone and on the re-run). Worker: Sonnet, medium. *Built* (2026-11-15, W163, branch `task-407-rss-tests`): the failure was the warm-up, not the library. The check had taken two flat blocks (120464, 120448 KB) for the end of the allocator's ramp, and the ramp then resumed: marks `[120464, 120448, 124272, 128320, 129920, 132240, 132320]`, a Theil-Sen slope of 19.6 against 4. The whole test ran in 0.93 s, so a ramp stalling for ~240 iterations is a few tens of milliseconds of scheduling. Not reproduced by 24 runs of the same loop under ten busy loops (load 20+) and six at once; every one plateaued, so the stall is rare, and the fix is built on the logged marks (scripted in `TestThePlateauStatistic`). `assert_plateaus` now (1) calls the ramp over only after three consecutive quiet blocks (`quiet_blocks`; was two) with up to 32 warm-up blocks (was 16), and (2) repeats a failed measurement, up to three windows, passing if any plateaus (`windows`). A leak grows in every window and fails all three; a ramp that was still going settles in the next. This catches what the plateau check is for at the same limit (4 KB/iter) and costs a failing run two more windows of ~0.1 s. Held by: `test_a_ramp_that_stalls_before_it_ends_is_not_a_leak` (the logged marks), `test_a_leak_still_fails_every_window` (scripted), and `TestAnOutputKeptLeaksFails` (a real process keeping every output: fails the check, in a quiet run and at a load of 60). Every plateau test goes through `assert_plateaus` (and `test_gbdt.py`'s one), so all inherit it. Not done: measuring the allocator's live bytes instead of RSS (the extension exposes no counter, and a Python-side count would miss the Rust side, which is the point of the tests).
  - *Built, bit-identical*: (1) `WindowsRun::evaluate_inputs` evaluates every operator input over the chunk's whole frame in one plan (inputs are element-wise, so a value does not depend on the rows beside it), and each group is handed its rows' values (`feed_resolving_with`, a take of an `f64` vector, no plan); none for one group (nothing to share), for an input that reads an increment (a group's own state) or where the plan refuses (the groups' own plans then report it, by group). (2) A group's step returns its resolved rows before the formulas (`Rows::Pending`) and `resolvers::settle` stacks every group's rows and evaluates the formulas once (`WindowsRun::resolved_formulas`, the code the step ran, over any number of stacked runs); a bare-operator formula still resolves in the step. A formula that refuses the stacked rows (a strict cast whose message counts the failing values) is evaluated group by group again, so the refusal reported is the first group's own, in the chunk's order, as it was (test `a_cast_that_refuses_on_two_groups_reports_the_first_s_message`; the bank restores its cores as before, `needs_snapshot`). (3) `group_frame` takes a group's rows column by column on the step's own thread instead of `DataFrame::take`, so a step no longer waits on Polars' pool (below).
  - *Why the values are the same*: Polars' element-wise kernels give a row's result from that row's inputs alone; shown, not assumed: a digest (W150's, extended: `digest.py`, 25 formula shapes -- arithmetic, `log`, `exp`, `**`, `sqrt`, `abs`, `clip`, `fill_null`, when/then, string compare, int and f32 inputs, nulls, strict and non-strict casts, two and three operators, an increment -- each plain, under a `window_size` and with a second target; 1, 7 and 200 groups, four chunkings, a save and load mid-stream; and 2,000 groups for eleven of them; every output column by `to_bits` and the saved bank's bytes), a multi-spec bank and a refusal, equal between main's wheel and this branch's: 230 + 33 lines. `digest_refuse.py`: a strict cast that refuses in one or two of 1, 5, 40 and 300 groups, as an operator's input and as the formula's output, three chunkings: 144 lines, messages (with their per-group counts) and the saved state after the refusal and after going on, equal. Rust: `a_computed_formula_over_many_groups_is_the_column_form` (four computed formulas against the column form `with_windows` computes over the whole frame, 1, 5 and 60 groups, three chunkings, the state's bytes the same at each).
  - *Mac (arm64, release wheels, ABBA x5, minimum, p25 in brackets; load as stated)*: 1,000 groups, chunks of 2,000: arithmetic 0.653 (0.664) -> 0.156 (0.158) s, 4.2x; computed input 0.456 (0.474) -> 0.121 (0.124), 3.8x; two operators 0.635 (0.675) -> 0.168 (0.170), 3.8x; cast 0.846 (0.857) -> 0.473 (0.501), 1.8x; bare operator 0.299 -> 0.129, 2.3x (load 11-18, 24-26 for the last). 1,000 groups, chunks of 50,000: arithmetic 0.074 (0.076) -> 0.029 (0.030), 2.6x; computed 0.074 -> 0.028, 2.6x (load 12-14). 100 groups, chunks of 2,000: arithmetic 0.150 (0.161) -> 0.062 (0.062), 2.4x; two operators 0.162 -> 0.067, 2.4x (load 10-13); computed in chunks of 50,000 0.023 -> 0.015, 1.5x. One group: level (arithmetic 0.066 -> 0.061 in chunks of 2,000, computed 0.056 -> 0.056, arithmetic in chunks of 200 0.194 -> 0.151; load 13-24): one group has one plan of each kind either way. The 100-group and 50,000-row rows, first taken at load 24-29 (another worker's tests), were repeated at the loads listed.
  - *The stack*: a group's step takes its rows on its own thread and, for these formulas, no longer waits on Polars' pool, so task 401's nesting cannot arise there; it still can where a step plans (an increment, a plan that refuses), so `in_parallel` stays. Task 401's test now runs a computed formula as well as the arithmetic one, 2,000 groups on one and two threads, debug: passes, and so does the same test at 20,000 groups (100,000 rows, run once and not kept: the state's bytes equal on one and two threads, no abort; 43 s). The stack's depth itself was not sampled.
  - *Not done*: (a) a formula with a cast (and any bank that snapshots its cores, `needs_snapshot`) still clones every group's core each chunk before it feeds one: at 1,000 groups in chunks of 2,000 that is about 0.32 of the cast's 0.47 s, 6 microseconds a group a chunk (the formulas, schemas and column lists each core owns by value; sharing them behind an `Arc` would cut it and each group's memory, which wants its own measurement). (b) An `increment` still plans twice a group a chunk (its input over the group's rows, and the operators' inputs that read it). (c) One group in small chunks still pays one plan for the formulas, about 90-180 microseconds a chunk; a Rust evaluator for the `f64` arithmetic subset (`+ - * /`, negation, comparisons) would be IEEE-exact, but `log`, `exp` and `**` would each need their own proof against Polars' kernels, and it was not attempted. (d) `docs/how-to/use-more-cores.md` still says a formula target's window step hands a group's rows to Polars' threads one group at a time on each of the bank's threads; it now does so once a chunk for the whole chunk's groups, and `pool.rs`'s module doc says so.
- [x] 408. **`corrchange(kind="window")`: the permutation draws read the pooled rows from one flat copy** (task 386 wave 5; 1.5.0 measured 9,157 ns a row at W=500, a redraw every 100 rows, `permutation_crit` 76% of the instructions). Worker: Sonnet, medium. *Built* (2026-11-15, W164, branch `task-408-corrchange`; scratch `.claude/coord/W164-408-scratch/`): each draw ran two passes over `ring[shuffled[i]]`, a `VecDeque` of separately allocated rows, through a closure, with the pair loops' trip counts 3, 2, 1, 0 at four columns. Now `permutation_crit` copies the pooled rows once into a flat `pool` (`Work`, not in the state) and each draw permutes row *places*; `window_corr_flat` runs the same sums in the same order over them, with `co_moments_fixed::<D>` for 1 to 8 columns (pair loops of compile-time length, unrolled, sums in registers) and `co_moments_rows` otherwise. The random stream and its consumption are untouched (same Fisher-Yates, same `next_u64`), each sum adds the same terms in the same order, so every output and state byte is the same. Held by the existing copied-oracle test (`the_window_kind_read_in_place_is_the_copied_one_to_the_bit`) and a digest of the outputs and the saved state of 8 window and monitor runs (W 33 to 500, `permute_every_rows` 1 to 250, 2 to 20 columns, blocks 1 to 7, both norms, save and load mid-stream), equal to main's. Mac (arm64, release, ABBA x6, d=4, W=500, 200 draws every 100 rows): 15,638 (p25 15,686) -> 7,343 (p25 7,376) ns a row, 2.1x; W=200 d=8 block 7: 2.8x. Cost: `Work.pool`, 8 bytes a feature for each pooled row, kept between redraws (about the ring's own size again; the ring of separate rows holds more) and not saved. Not done: fusing the means into the copy, and the per-draw index vector (the remaining cost is the two passes' arithmetic).
- [x] 409. **A small call does not hand off to the bank's thread pool** (task 386; W165, slot 2, branch `task-409-one-row-pool`, at main 44770dbc; scratch `.claude/coord/W165-409-scratch/`). Task 391 left ~13 us of a one-row call in the pool's job hand-off and the wake of a parked worker. *Built*: `Bank::runs_inline` (bank.rs) runs a call on the calling thread, under `pool::Inline` (a thread-local the parallel sections check: the `par_iter`s in `fit_predict_on_pool` and `predict_on_pool` go through `map_maybe_par`-style helpers, `join_maybe_par`, and `Stream`'s instance loops), instead of `Threads::install`, when: the chunk is under `PAR_MIN_ROWS`; the caller is not on a rayon thread; no spec sends work to rayon itself (`Spec::threads_inside`: formula targets, `marginal` shards, `gram_threads` above 1); no spec is wider than 64 features; and either the bank is one ungrouped single-instance spec (one task whatever the chunk) or `rows * sum(instances * outputs * (features+1)^2)` is at most 1,500. The cut is from a sweep on the Mac (1 to 10,000 rows, 1 to 1,000 groups, 1/4/14 threads, `ewridge` k=5 and a 3-spec bank, inline against pool in one build): inline won to 50 rows on 10 groups for the `ewridge` and to 30 rows for the 3-spec bank, lost 1.2x at 100 rows on 4 threads, and won at every size to 3,000 rows with one task (0.84-0.97; 1.03 at 10,000, so the rows cap). Bit-identical: a task's arithmetic is the same on any thread; digest of `fit_predict`/`predict`/`fit` over 3 banks x 1 and 100 groups x 1 to 20,000 rows equal to main's wheel. Tests (threads_tests.rs): a small call's tasks run on the caller and a large one's on the pool; the decision's cases; inline and pool give the same bits over six chunks; a small call fits a 512 KB stack. Task 401's nesting cannot arise: an inline call enters no pool, and a spec that could (formula targets) never runs inline. Mac ABBA, wheels, 6 rounds, 14-thread bank pool: one-row `ewridge` min 36.9 -> 24.2 us; 10 rows 38.1 -> 26.8; 3 specs one row 82.0 -> 48.9 (p25 136.7 -> 55.4), 10 rows 78.5 -> 55.4; 3 groups 10 rows 42.2 -> 28.0; 100,000 rows 34.1 -> 34.4 ms (the same path).
- [x] 410. **A formula target's per-group snapshot copies the chunk's groups' state, not the spec's constants or every group** (task 386; task 406's *Not done* (a); W166, slot 1, branch `task-410-formula-cast-share`, at main 44770dbc; brief `.claude/coord/W166-410-formula-cast.md`, scratch `.claude/coord/W166-410-scratch/`). Worker: Sonnet, high. *Measured at main* (a `#[cfg(test)]` probe with a counting allocator, not kept, over one core of a cast formula after 30 chunks of 2 rows; the counts are the same in a release build): a clone of one group's `WindowsRun` is 81 allocations and 28,262 bytes: the rows it holds, 30 frames of three columns, 31 allocations and 15,840 bytes; the window core's state (queues, open rows, held values) 30 allocations and 10,173 bytes, of which the operator definitions (`kernels`, `ops`, `members`) 3; the per-spec constants -- the call (formulas and clock policy, 11 allocations), the operators' input nodes, the column lists, the input and output schemas -- about 19 allocations and 1,800 bytes. So the premise "each core copies the same constants" was a fifth of the allocations and a fifteenth of the bytes; most of the copy was the held row frames, one allocation and one `Column` array each. *Built* (bit-identical): (1) `TargetWindows::snapshot` keeps only the cores of the chunk's groups and `restore` puts them back: a chunk feeds no other core, so a refusal has nothing else to restore, where the bank cloned every group's core of every spec (at 100,000 groups in chunks of 2,000 rows, fifty times the cores it fed). (2) `WindowsRun`'s constants are one `Arc<Constants>` and the core's operator definitions `Arc`s (serde's `rc` feature: an `Arc` writes as its content, so the file's bytes are as they were, and no layout changed); `TargetWindows` keeps a core that has read no row (`proto`) and `WindowsRun::share_with` points each new core's constants at it where they are equal (a core built on another input schema keeps its own; a loaded core is shared after it loads). (3) The rows a run holds are `VecDeque<Arc<DataFrame>>`: a copy shares the frames, and a frame leaves the queue by `Arc::try_unwrap`, copying only where a snapshot still shares it. A copy of a run is now 28 allocations and 10,773 bytes (the core's own state and the last row read). *Held by*: `a_refused_chunk_mid_stream_leaves_every_group_as_it_was` (forty groups, a refusing chunk after others were fed: the bank's bytes after the refusal are the bytes before it, and going on is going on in a bank that never saw it; fails with `restore` made a no-op), `runs_of_one_call_share_their_constants_and_only_those`, and a digest (W162's `digest.py`, `digest_refuse.py`: 25 formula shapes, plain, windowed and with a second target, 1, 7, 200 groups and 2,000 for eleven, four chunkings, save and load mid-stream; every output column by `to_bits` and the saved bank's bytes; a strict cast refusing in one or two of 1, 5, 40, 300 groups, three chunkings: the messages, the state after the refusal and after going on) equal to main's: 230 + 33 + 144 lines. *Mac (arm64, release wheels, ABBA x5, load 5-7, minimum, p25 in brackets)*, `fit_predict_batches` over 100,000 rows, ewridge k=5 with one forward operator: 1,000 groups, chunks of 2,000 rows: cast 0.497 (0.508) -> 0.279 (0.280) s, 1.78x; arithmetic around an operator 0.155 -> 0.151, 1.03x; two operators 0.164 -> 0.160; bare operator 0.107 -> 0.101; 100 groups: cast 0.079 -> 0.075; 1,000 groups in chunks of 50,000: cast 0.0300 -> 0.0277; one group: cast 0.0585 -> 0.0585 (level; the clone of one group was never the cost). *Memory* (resident set of the bank after feeding, `mem.py`, bytes a group, whole bank, a cast formula): 1,000 groups, 3 rows each, 66,716 -> 51,249 (-15.5 KB); 1,000 groups, 20 rows each, 112,427 -> 87,818 (-24.6 KB); 100,000 groups, 3 rows each, 30,731 -> 24,109 (-6.6 KB); the same bank with a plain target is 28,000 (main) and 26,132 (this branch) a group at 1,000 groups and 12,855 and 12,672 at 100,000 (the plain bank is the same code; the difference is run to run), so the formula's cores cost 38.7 -> 25.1 KB and 17.9 -> 11.4 KB a group. *Not done*: a cast that cannot refuse (a float to a float) could skip the snapshot altogether, which wants a proof of the operand's type per formula; the cast case's remaining 0.13 s over the arithmetic case is the 28 allocations of each core's own queues and held values, which are state.
- [x] 411. **A formula target that reads an increment: no Polars plan per group per chunk** (task 386; W167, slot 3, branch `task-411-formula-increment`, at main 44770dbc; brief `.claude/coord/W167-411-formula-increment.md`, scratch `.claude/coord/W167-411-scratch/`). *Re-measured at main* (release wheels, arm64, `fit_predict_batches` over 100,000 rows, ewridge k=5, one forward operator over an increment, ABBA x5 with 2 reps a process, minimum; loads 6-22, the 12-22 runs repeated): task 406 left a formula whose operator reads an `increment` (the step of a column against its group's previous row) two plans of Polars' lazy engine per group per chunk, the increment's input and then the operators' inputs that read it, because an increment is a group's own state. At 1,000 groups in chunks of 2,000 rows: `ewm_mean(increment(y))` 0.274 s, `ewm_rate(increment(y))` 0.280, `ewm_sum(increment(y * x))` 0.426-0.520, `ewm_mean(increment(y)) - y` 0.337-0.344; 100 groups: 0.066-0.071 / 0.091-0.092 (the first and third); 1,000 groups in chunks of 50,000: 0.036 / 0.045-0.047; one group 0.056-0.060 whatever the shape.
  - *Built, bit-identical*: an increment is `x_i - x_{i-1}` on the doubles, or the integer step, of the group's previous value: element-wise given that value, so only the value is a group's own. `WindowsRun::share` (the resolver calls it for a chunk of two or more groups, in place of `evaluate_inputs`) evaluates (1) the increments' inputs over the chunk's whole frame in one plan and reads them in the forms a step takes them in (`IncrInputs`: integers, doubles, nanoseconds), (2) each run's increments off its own state with `preview_increments` -- the same function `with_increments` runs (`step_increments`, moved out of it unchanged), on a copy of the run's increment state, so the run is not moved -- from its rows of those inputs, the clock and the session read once for the frame, and (3) the operator inputs over the frame with the increments as the hidden columns they stand in as. Each group is handed its rows' part (`Shared::of_group`, `Given`) and its feed computes its own increments from its own state as before (the Rust loop, no plan) from the handed inputs; so the state, the held rows and what a refusal rewinds are the feed's own. None where a plan refuses (the groups' own plans report it, by group, as before), for one group, for a run keyed by a `group` column (a `with_windows` call, not a bank's core) and for a run that skips rows its input read before it was saved. `evaluate_inputs` is private now (the no-increment half of `share`); `feed_resolving_with` takes a `Given`. Held by `a_chunk_s_shared_increments_are_each_run_s_own` (windows_frame.rs: four runs, two chunks, a null, a step back past a restart, a session change, an integer input, two increments, an increment in the formula's body; the same bits and state bytes as runs that plan their own; the shared values are there; a mutated preview that ignores the state fails it), five increment trees in `computed_trees` (formula_targets.rs: the column form `with_windows` computes over the whole frame, 1, 5, 60 groups, three chunkings, the state's bytes the same at each), and a digest.
  - *Digest* (`digest_incr.py`, 16 increment shapes: diff, rate, an increment of a computed input, of a column with nulls, NaNs, an integer, an `f32`, one inside a formula body, one in a backward operator, `partial="drop"`, a cast, two increments, arithmetic around one; each plain, under a `window_size`, with a second target and with a high `min_weight`, and for five of them under a clock that steps back with `restart_after_step_back=0` and a session with `session_gap="reset"`; 1, 7, 200 and 2,000 groups, four chunkings, a save and load mid-stream; every output column by `to_bits` and the saved bank's bytes; a two-spec bank and a refusal): 298 lines equal between main's wheel and this branch's. `digest_refuse_incr.py`: a strict cast that refuses in an increment's input, in an operator's input that reads one and in the formula, on one or two of 1, 5, 40 and 300 groups, three chunkings, the message (with its per-group counts) and the saved state after the refusal and after going on: 240 lines equal. A temporal increment is not reachable from a bank (a temporal column can only be a clock, which a formula target cannot read), so only the `with_windows` tests hold that branch.
  - *Mac (arm64, release wheels, ABBA x5, minimum, p25 in brackets; load 6-22, the three loads over 12 repeated at 8-12 and 16-22: the ratios agreed)*: 1,000 groups, chunks of 2,000: diff 0.274 (0.277) -> 0.143 (0.143) s, 1.9x; rate 0.280 -> 0.144, 1.9x; increment of a product 0.426 (0.429) -> 0.165 (0.166), 2.6x (0.520 -> 0.176 at load 8-22, 2.9x); arithmetic around one 0.344 (0.356) -> 0.194 (0.200), 1.8x; 100 groups: diff 0.066 (0.067) -> 0.057 (0.058), 1.2x; increment of a product 0.091 (0.095) -> 0.059 (0.059), 1.6x; 1,000 groups, chunks of 50,000: diff 0.036 -> 0.033, 1.1x; product 0.045 (0.047) -> 0.035 (0.036), 1.3x; 100 groups, chunks of 50,000, product: level (0.017 -> 0.018). One group (the same code path): level, 1.00-1.02x (diff 0.060 / 0.060 in chunks of 2,000 and 0.056 / 0.056 in chunks of 50,000; product in chunks of 200, 0.119 -> 0.117). What remains at 1,000 groups against a bare operator (0.118 s): the increments' state is walked twice a chunk, once for the preview and once by the group's own feed, and each group takes its rows of the increments' inputs.
  - *Cost*: `Shared` holds the chunk's operator inputs (a double a row an input) and the increments' inputs (an `Option` of 8 or 16 bytes a row, 32 for an integer) for the length of the chunk's step; a copy of a run's increment state a chunk, in the preview. Not in the state.
  - *Not done*: (a) the preview and the feed walk each group's rows twice; handing the feed the preview's increments and the state it ended at would walk them once, but then the snapshot a refusal rewinds to, and the rows a skipped-input run drops, would have to move with it, and the gain is a few microseconds a group a chunk. (b) One group in small chunks still plans twice a chunk. (c) `docs/PERFORMANCE.md` still says a native look-ahead target "adds a Polars query per group per chunk" (task 406's note, not changed here).
- [x] 412. **The window kernel against the processor's pipeline** (the user, 2026-11-15: "Are you considering the processor pipeline and the impact of branching?", then "Yes"). Task 403 left each row's decay as one serial chain (tau, a division, `exp2`, the fold) and the clock's per-row checks (gap cap, session, nulls, reset) as branches in the row loop. Measure on the Mac with Instruments' CPU Counters (IPC, branch mispredicts, stall cycles) and on the VM with Callgrind's branch simulation, then, bit-identically: interleave independent chains (several operators, several groups) so the CPU overlaps one while another waits; take the per-row decisions once per run of plain rows; reuse a decay for a repeated gap (a regular clock repeats it). Worker: Sonnet, high. *Built* (W168, slot 1, branch `task-412-window-pipeline`, at main `c11c1d6c`; scratch `.claude/coord/W168-412-scratch/`). *Measured* with Instruments' CPU Counters (the `polars-cpu-counters` template: cycles, instructions, branch mispredicts, L1D load misses; no stall-cycle event in it) over `WindowsRun::feed`, 4M rows a pass, three passes, a release build with line tables, main's code, datetime clock, per row inside `feed_inner`: one operator 222 cycles, 1,455 instructions, IPC 6.55, 0.35 mispredicts, 0.9 L1D misses; four operators (mean, std, sum, var) 382 cycles, 2,747 instructions, IPC 7.19, 0.39 mispredicts; four operators with `window_size` 60 s 964 cycles, 6,234 instructions, IPC 6.46, 1.2 mispredicts; four operators over 100 keys 517 cycles, 3,288 instructions, IPC 6.35, 0.72 mispredicts, 21 L1D misses. *Verdict*: every case is limited by throughput, not by latency or branches. IPC 6.4 to 7.2 is a loop that is retiring instructions, not waiting on a result, so a dependent chain is not what bounds it, and only fewer instructions make a row faster; mispredicts are 0.2 to 0.35 a row inside `feed_inner` (the 0.35 includes the frame's assembly), which at about 15 cycles each (an assumption: the template has no penalty event) is 1.5 to 3% of a row, 2% under `window_size`, and they fall to 0.005 a row on a clock with a fixed step, so they are the data-dependent paths of libm's `expm1` and `exp2`. The libm calls are the largest single piece: `expm1` 200 instructions and 31 cycles a row (14%) and `exp2` 35, for the mean's held mass and the fold's discount; windowed, `exp2` and `expm1` hold 719 and 515 instructions of the 6,234, from 11 `discount` and 7 `mass` calls a row of which 5.9 and 3.8 repeat the previous call's distance. Dropped, with the reason: (a) overlapping independent chains (not built: no measured wait for it to hide, and it adds instructions to a loop bound by their number); (c) a select for an unpredictable branch, or the per-row decisions taken once per run (at most 3% of a row in mispredicts; `n_wait` and the kernel loops are 15 to 20 instructions each, and a hoisted `n_wait` would be a new field of the saved `Windows`, whose bytes must not move); a 64-bit unsigned road in `seconds_of_ns` (bit-identical, but measured +1.0 to +2.5% on the datetime cases, dropped). *Built* (b), bit-identical: `Queue` keeps a `DecayMemo`, the last `discount` and the last two `mass`es it was asked with the bits of their distances (`discount_in`, `mass_in`, `mass_far_in`; not serialized, never compared), and answers a repeated distance from it, the same function of the same bits. It replaces the `Last` memos of 403 (a row's operators asking the same distance) and serves the clock that repeats a step: the fold's discount (`push`, the turn-over, `then`), the mean's held mass, a read's discount and a window's cut. `KernelDef::mass_finite` is `mass` after its two early answers. *Result* (release builds, ABBA of main's and this branch's `winbench`, 4 rounds of 2 runs each, 4M rows, ns a row, min, load 5-8, main then branch): one operator on an Exp(1) clock 62.2 to 59.9 (-3.7%), on a 1 s step 60.2 to 51.6 (-14.3%); four operators 100.1 to 96.6 (-3.5%) and 98.8 to 88.6 (-10.3%); four operators with a 60 s window 248.8 to 212.2 (-14.7%) and 168.0 to 156.9 (-6.6%); 100 keys 138.0 to 136.7 (-0.9%) and 137.4 to 135.8 (-1.2%); one operator with the window 116.8 to 115.3 (-1.3%); a float clock 60.3 to 59.6 (-1.2%). Counters, main then branch, one operator: Exp(1) 221.6 to 211.8 cycles (1,464 to 1,431 instructions), fixed step 212.7 to 173.4 (1,469 to 1,313; IPC 6.9 to 7.6); four operators windowed 946 to 807 (6,108 to 5,577). *Bit-identity*: the 118-case digest (`digest.rs`, outputs and state bytes at four chunkings with a save and load) equal to main's on four clocks, Exp gaps (the 403 digest, `ce4e8f6bbc3fd8bd`) and, new, steps of 1 s, 0.1 s and half of each, where the memo answers most calls; a unit test holds the memoed `discount` and `mass` to the unmemoed to the bit over repeats, zero, a step back, NaN, infinities and an infinite half-life, and another that the memo is read and kept. Files: `crates/online-polars/src/windows.rs`.
- [x] 413. **`lasso`'s coordinate descent as two chains** (the user, 2026-11-15, as task 412). Task 394 found the dense sweep latency-bound (`rho -= c*b`, ~0.6 ns a term, one dependent chain); coordinates i and i+1 share the prefix j < i and can run it as two independent chains, the suffix staying serial, the same sums in the same order (W155's estimate: up to 1.6-2x on the dense part). Measure with the same counters; bit-identical. Worker: Sonnet, high. *Built* (2026-11-15, W169, branch `task-413-lasso-chains`; scratch `.claude/coord/W169-413-scratch/`; `lasso.rs`, `scripts/mutants_equivalent.toml`). *Counters* (Instruments' CPU Counters, the `polars-cpu-counters` template: cycles, instructions, `BRANCH_MISPRED_NONSPEC`, `L1D_CACHE_MISS_LD`; one sweep in a micro on a seeded correlation matrix of 3k rows, per sweep) *say the dense sweep was latency-bound*: at k = 50 / 200 / 500, 1.85 / 2.22 / 2.43 cycles a term at IPC 3.8 / 2.85 / 2.6, six instructions a term, so the `fsub` chain (about 2.2 cycles) and not the issue width sets the pace; mispredicts 3 / 449 / 1,192 a sweep (loop exits and the threshold, about 3% of the cycles at k=500), L1D misses a sweep 0 / 5.4k / 36k (the matrix streams from L2; not limiting). *The sparse sweep too*: IPC 4.2-4.9, 1.5-2.2 cycles a term, one chain over the gathered nonzeros. *Built, bit-identical:* the terms of `rho_i` before `i` (`c_i0 b_0` ... `c_i,i-1 b_{i-1}`) are known as the sweep goes and none waits for `b_{i-1}` but the last, so they are taken off ahead of time, off the chain that ends in `b_i`: `acc[m]` holds `d_m` with the terms taken off in `j` order for the `j` set so far, and when coordinate `j` has its `b_j` every `acc[m]`, `m > j`, takes off `c_mj b_j` (column `j` of `c`, read as its row `j`), the update running in the loop of coordinate `j+1`'s chain over the terms after it, four terms of the column to four of the chain, so the column goes as vectors. What stays on the path to `b_i` is `acc[i]` and the `k - i - 1` terms after `i`: half the terms in all, the other half beside them. Every `acc[m]` is the same chain of subtractions of the same products in the same order as the one-chain sweep's: the answer to the brief's worry that the `j = i` term sits in the middle of `i+1`'s sum is that it is taken off first at the start of `i+1`'s second run, after the column's earlier terms (they are all in `acc`) and before the terms after it. A coordinate held at scale 0 still has its column taken off (`c * 0`, NaN where `c` is not finite), as before. The sparse sweep does the same over the gathered starting nonzeros, taking off only nonzero coefficients' columns, a share of the column beside each block of eight chain terms. Needs `c` bit-symmetric: `standardized` now writes the lower triangle and mirrors it (the entries were the same numbers already: `cov` reads one triangle either way, `s_i * s_j` commutes), a `debug_assert` in `descend`, a test over both intercept settings, a feature with no spread and a collinear pair; so the serial `cd_sweep` and the old sparse sweep are gone from the library and the serial sweep is the test reference (`cd_sweep_serial`). The soft threshold is `soft_over` (the two equivalent-mutant entries follow it). *Tests* (`lasso.rs`): the dense sweep against the serial sweep to the bit, coefficients and move after every sweep, k = 1-100, penalties from sparse to none and a ridge, features held at scale 0 first, last, in a run and apart, from zero and dense starts; signed zeros in `c`, `d` and the start, a NaN and infinities in `c` and a NaN in `d`; the sparse sweep against the serial sweep on 36 seeded problems (k = 64-130, starts of every sparsity, signed zeros, held coordinates); `descend` over a path on dense problems (k = 2, 3, 20, 50); `is_symmetric`; `standardized`'s symmetry. Breaking the held-column step or the tail of the column's update fails them. *Time, arm64* (release builds, ABBA, 5 rounds, min / p25, load 5-8; the rows of a `fit_predict` of `wide_benchmark.make`, `half_life` 1,000, `POLARS_MAX_THREADS=2`; the wheel at main against this): k=20, 3 penalties 1.12x / 1.09x; k=50, 10 penalties 1.25x / 1.26x; k=100 1.44x / 1.44x; k=200 1.69x / 1.69x; k=500 1.78x / 1.82x (a Rust probe through `Lasso::step`, 1 / 3 / 10 penalties: k=200 1.63x, k=500 1.79x with 10). Per sweep (micro): dense k=50 / 200 / 500, 3,863 / 53,147 / 317,819 cycles against 4,631 / 88,677 / 607,999 (1.2x / 1.7x / 1.9x; 1.55 / 1.33 / 1.27 cycles a term, IPC 4.5 / 4.2 / 4.2, mispredicts 0.1 / 16 / 581); sparse k=100 (90 nonzero) 13.3k against 21.6k, k=200 (122) 33.0k against 61.3k, k=500 (333) 249k against 451k and (200) 130k against 255k cycles (1.6-2.0x; IPC 5.7: now instruction-bound). The floor of the dense chain is the `fsub` latency of the half that stays on it, which the loop is within about 15% of. *Measured and not kept:* a branch-free soft threshold (both quotients computed, selected) saves 1-3%; blocks of 2, 8 and 16 terms for the dense column update (within 4% of 4, 2 costing more instructions) and of 4 and 16 for the sparse (the same as 8); a plain one-term-at-a-time update beside the chain is 10-18% slower at k >= 50 and costs fewer instructions at k = 4-20 (about 4% fewer than the one-chain sweep against 13% more for the blocks at k=20: the blocks stay, for the time). *Bit-identical:* a digest of every float column of a `fit_predict` output frame, of `save_bytes()` and of the continuation of a loaded copy over 200 more rows (`drv2.py`), main's wheel against this one, equal in all 21 cases (k = 20, 50, 63, 64, 100, 200, 500; 1, 3, 10 penalties); and a Rust probe's predictions and coefficients (`Lasso::step`) equal in 39 cases (k = 1-500). No state layout change. *Instructions:* at k=20 the micro's counters put the sweep at 6-13% more instructions than the one-chain sweep (533M against 567-605M over 150k sweeps) for 14-17% fewer cycles; a sampled run total (noisy to 2-3%) read +4% before the symmetry check left `descend`; the VM's x86 `lasso_k20` count decides whether its budget needs a raise (reason: 1.1x at k=20, 1.7-1.8x at k >= 200). *Not done:* two penalties' sweeps interleaved (the path is warm-started, so each penalty starts from the last one's result and they are not independent chains); `gramfit::cd_path` (the offline path) keeps its own serial sum.
- [x] 414. **Every hot loop against the processor's pipeline: a survey** (the user, 2026-11-15: "Have we looked at cpu pipeline for everything?"; only the window kernel and `lasso` had been, tasks 412 and 413). Measure only: for every model's per-row step and the bank's per-row plumbing (the benchmark's cases and the instruction-budget cases), hardware counters on the Mac (instructions per cycle, branch mispredicts, cache misses) and the hot functions, then a verdict per loop (limited by a dependent chain, by branches, by memory, or by instruction count) and a ranked list of loops with headroom and the bit-identical reshaping each suggests. Follow-up tasks are cut from the list. Worker: Sonnet, medium. *Measured 2026-11-15 (W170; `.claude/coord/W170-414-scratch/SURVEY.md`, 39 model cases and 4 bank cases on the Mac):* most loops are instruction-bound (IPC 6-9, mispredicts under 2% of cycles); latency-bound: `rls` (IPC 2.8), `ew_cov` (3.8); branch-bound: `audit` (22% of cycles mispredicting), `seqtest` (12.6%), `kmeans` (5.3%), `bocpd` diag (~5%); memory: `ewridge` k=200; the grouped bank row 18x the model alone at 1,000 groups, 48% of it allocation inside each solve. Follow-ups: tasks 415-421.
- [x] 415. **Solve scratch kept warm across the solving models** (task 414's first): `ewridge`'s solves allocate faer `Mat`/`Vec` scratch each time (48% of the grouped bank row at 1,000 groups); keep it as `lasso` does (task 413), and the same in `huber`, `quantile`, `kalman`, `ew_class`, `hmm`; and look at the `zip` iterator leaf (11-25% of `ewridge`'s cycles) for a lost inlining or vectorization. Bit-identical. Worker: Sonnet, high. *Built* (2026-11-15, W171, branch `task-415-solve-scratch`; scratch `.claude/coord/W171-415-scratch/`; `ewridge.rs`, `solve.rs`, `gaps.rs`, `ewcov/lower.rs`, `ewridge/solve_scratch_tests.rs`, `crates/online-bench/tests/allocations.rs`). *Allocations a solve, counted* (the allocation test's counter, one group, `step_into`, default cadence; before / after): `ewridge` k=5 22 / 0 (1.518 a row, 69 solves in the block), k=20 22 / 0 (the bench cases; on a stream solving at the default share, 3.4-3.6 a row at k=1 to 50 / 0); standardized 3 targets 9.2 a row / 0, through the origin 6.1 / 0, sum scale 6.4 / 0, a grid of 3 ridges over two feature sets 18.5 / 0, the systems kept for the row inflation 7.4 / 0; `huber`, `quantile`, `kalman`, `ew_class`, `hmm`, `deco`: 0 a solve already (tasks 293-304; the bounds stand, `quantile`'s 0.08 a row is its band systems' lifecycle, not scratch); what still allocates: a system past order 64 (`KEPT_ORDER`: its matrix and right-hand sides are made each solve, as before), the blocked Gram's merges (`gram_block_rows`: 7.9 a row at k=5 with `solve_every = 4`, in `flush_riding`), a windowed fit's `window_weights` (2 a row). *Built, bit-identical:* a solve formed about 22 lists at five features -- the combos, readers and feature indices, the system's scales, kept columns, matrix and right-hand sides, the solution's matrix and vector (`SpdFactor::solve`'s `Mat::from_fn`), the fit's lists, the shares' starting values, the certificate's sums, the pending solve's `Arc` with its two lists, and under `emit_error_inflation` the kept system's lists. The lists a solve forms and drops are now one set for each thread (`SOLVE_BUFS`, lent to the solve and left for the next, whichever model runs it: a model's own were 2.5 KB a group at 10,000 groups, +11% of the process, and cold in cache at each group's turn); the fit's storage before the last (`spare_fit`) and the pending solves' `Arc`s with their lists (`Spare`, in the model's `Readiness`, kept across a settle) are the model's, boxed in `Kept`, not state, a copy holds none. Each list is written in full before it is read (`clear` then `resize` where the old one was `vec![0.0; n]`, so the system's upper triangle is still 0), the solution is `SpdFactor::solve_into`, the same two triangular solves on the same numbers in a kept matrix (`QuadWork`, one for each of up to 16 shapes), a system kept for the leverage is written over the last one (`LowerTriangle::set_from_square`), and a list past 4,096 entries (an order-64 system) is dropped after the solve. *Not state:* no layout change, `SCHEMA_VERSION` unchanged. *Bit-identical, shown:* a digest (`to_bits` of every float, `save_bytes()` and a loaded copy's continuation over 200 rows) of `Bank::fit_predict` output over 39 cases -- `ewridge` k=1, 3, 5, 20, 50, 70, 3-4 targets with missing targets, standardized or not, with and without an intercept, sum scale with a prior, ridge grids, feature sets, `pairwise`, window, `emit_error_inflation` with `emit_se_coef` and `emit_influence`, conformal, blocked Gram, a prior; `huber`, `quantile`, `kalman`, `ew_class`, `hmm`, `deco` -- each at 1 group (one chunk, 7 rows a chunk with gaps, 4,096 with a save and load, 64 with gaps and a load), 1,000 groups (4,096 and 500 with a load) and 100 groups (33 with gaps): 274 digests, main's build and this one, equal (`digest415.rs`, in the scratch dir); and `solve_scratch_tests.rs` holds a kept model to one whose scratch is emptied before every row over 5 systems x keep_factor x feature sets x window x k = 3 and 70, predictions, inflations, state `Debug` and bytes. *Counters* (Instruments' CPU Counters, `ewridge_k5`, one group, per row, main / this): 359 / 200 cycles, 2,696 / 1,590 instructions, IPC 7.5 / 8.0, 0.56 / 0.05 mispredicts (2.3% / 0.4% of the cycles), L1D misses 1.7 / 1.7: instruction-bound before and after (IPC 8 of about 10 wide, mispredicts under 1%), the 1,100 instructions a row that went being the allocator's; the process's totals (`/usr/bin/time -l`) say 950 instructions a row fewer at k = 5, 20 and 50 alike (k=50: 3.7% of its instructions). *The `zip` leaf* is no lost inlining or vectorization: `rank_one_lower_rows` is 8-wide NEON (`fmul.2d`/`fadd.2d`, a scalar tail under 8), `rank_one_fixed` is the same at k = 3-12, and `Cross::learn`'s four-way compensated zip loop vectorizes (compiled alone: `fsub.2d`), so the profile's `next<IterMut, Iter>` leaf is the inlined frame's label for the Gram's rank-one and cross-moment loops, which are the instruction count at k >= 20 (22-25% of the cycles at k=50) and have nothing left but fewer operations. *Time, arm64* (release, ABBA, 6 rounds, min / p25, `POLARS_MAX_THREADS=2`, load 4-7; ratio of main to this): `ewridge` alone k=5 87.7 / 88.9 -> 55.9 / 56.2 ns a row 1.57x / 1.58x, k=20 1.22x / 1.22x, k=50 1.07x / 1.06x, k=20 with 10 targets 1.20x / 1.18x, `ewridge_gram_t10` 1.01x / 1.03x; through a bank (`fit_predict`, frames of 4,096, one thread) at 1 group k=5 202.5 -> 170.3 ns 1.19x / 1.19x, at 1,000 groups k=5 1,617 -> 798 ns **2.03x / 2.05x**, k=20 1.03x / 1.03x, k=50 1.02x / 1.01x. A build before the last two edits (thread-shared lists) read k=50 alone at 0.95x in this layout and 1.07x in the other (the bench profile's), and 0.98 / 1.01 at k=20 and 50 grouped: code layout, as the decision of 2026-11-01 has it; both layouts of the kept-per-model build and this layout of the final were measured. *Memory:* the grouped bank's peak resident size at 1,000 and 10,000 groups of k=5, main / this: 35.7-42.4 MB over runs against 36.3, and 220.2-227.9 against 224.9 MB: no change beyond the runs' own spread (the kept-per-model build read +17 MB at 10,000 groups); a model grew by two boxed pointers, 16 bytes (`EwRidge` stays under `large_enum_variant`'s threshold in the first-period tests' enum, as a plain list field would not). *Tried and not kept:* `huber` and `quantile` dropping only their spare factors at a settle and keeping their lists: grouped 1,000 groups 1.01x and 0.99x, their grouped rows (2,430 and 2,653 ns against 286 and 587 ungrouped) are not allocation (0 allocations a row there), so not an allocator share to take; the per-model lists (above). *Allocation tests* (`allocations.rs`): `ewridge_k5` and `ewridge_k20` bounds 1.518 -> 0 (through `step_into` 1.30 -> 0, through `step` 2.30 -> 1), `ewridge_gram_t10` 1.241 -> 1.123, the windowed `ewridge` 9.432 -> 2.084 (release) and 42.432 -> 35.084 (debug), and a new `ewridge_solves_allocate_nothing_in_any_system` over seven systems. *Not done:* `huber`'s and `quantile`'s grouped overhead (task 421's plumbing, or a cache measurement), the blocked Gram's merge allocations, `window_weights`.
- [x] 416. **`rls`'s P update as overlapped chains** (task 414): IPC 2.8; interleave independent rows' dot-product chains, each row's summation order kept. Bit-identical. Worker: Sonnet, high. *Built* (2026-11-15, W172, branch `task-416-rls-chains`; scratch `.claude/coord/W172-416-scratch/`; `rls.rs`, `scripts/mutants_equivalent.toml`). *The premise was off:* `rls` keeps no `P` (it is the Givens/QR form: `R` and `u`, a row folded in by `k` rotations, then one back-substitution), so there were two chains and no dot products to interleave. Counters (Instruments' `polars-cpu-counters`, a Rust probe of `Rls::step_into`, per row, k=20, one target): 2,125 cycles, 6,968 instructions, IPC 3.28, mispredicts about 0: latency-bound, as 414 found. Ablations (a build with the solve skipped, one with the fold skipped; ns a row 518 = 289 + 229): the back-substitution was 44% and the fold 56% (decay, rotations). *Fold:* the working entry `z_{i+1}` that rotation `i` makes is what rotation `i+1` starts from (`z_i`, square, add, `sqrt`, two divisions, multiply, subtract: 8.2 ns measured by a dependent-chain micro, 21 rotations = 172 ns of the 190 the fold now takes), and it went through `zbuf`'s store and load; it is now kept in a register (the first entry of the row is done apart, the rest as before), and the decay pass over `R` and `u` is fused into the rotations (each entry scaled where the rotation reads it, a row that is not rotated scaled alone), where it was a separate pass of short loops (54 ns). *Back-substitution:* coefficient `i`'s sum was a serial chain over the row, the last term waiting on `b_{i+1}` after the loop through memory, the divide, a select for the finite test; the terms are now taken off as the `b_j` arrive (right-looking), `acc_{i-1}` first and kept in a register, the rest of the column in blocks of 2 columns below k=32 and 4 from it (`WIDE`), so each sum's subtractions are the same ones in the same order and what the next coefficient waits on is one subtraction and its division; the finite test is a branch past a `#[cold]` call (`zero_unseen`), which takes it off the chain; targets go down `R` in blocks of 4, 2 and 1 sharing each pass, their divisions side by side. *Bit-identical:* every `R` entry, `u` entry and coefficient is the same chain of the same IEEE operations in the same order (the decay is the same product, taken at the row where it is read: `lam == 1` is a separate instantiation with no product). *Tests* (`rls.rs`): the back-substitution against the serial one to the bit for k = 1-40, 63-65, 1-9 targets, entries with signed zeros, infinities, NaN, overflow and subnormals, pivots that are 0, negative or NaN; the fold against a decay pass plus rotations reading the working row back, to the bit, in `R`, `u`, the working row and the targets' entries, k = 1-40 and 63-65, 1-3 targets, with and without a decay (and a factor of 0 and 1e-170), zeros that skip a rotation, squares that overflow and underflow; breaking the column order, a block's tail, the carried sum or the skipped row's carry each fails them. *Counters after* (per row): k=4 429 -> 247 cycles (IPC 3.50 -> 5.98), k=20 2,125 -> 1,217 (3.28 -> 5.74, instructions 6,968 -> 6,983), k=50 5,734 -> 3,791 (4.46 -> 6.65; mispredicts 2.0 -> 0.01): now the fold sits at the latency floor of its chain, and the back-substitution is within about 30% of its own. *Time, arm64* (release, ABBA, 5 rounds, min / p25, load 5-8): the probe, k=4 / 20 / 50 / 200, one target 1.72x / 1.75x / 1.58x / 1.71x, k=20 with 3 targets 2.16x; through the wheel (`fit_predict`, `half_life` 1,000, `POLARS_MAX_THREADS=2`) k=4 1.22x, k=20 1.54x, k=50 1.53x, k=20 with 3 targets 1.81x, 100 groups 1.27x (min; p25 1.20x). *Digest* (`drv.py`: every float column of a `fit_predict` output frame, `save_bytes()` and a loaded copy's continuation over 300 more rows, main's wheel against this one): equal in all 72 cases (k = 1, 2, 4, 20, 31, 32, 33, 50, 100; 1 and 3 targets; half-lives 50 and none; 1 and 100 groups; zero-weight rows and zero features; one chunk against chunks of 137 rows), and a Rust probe over 384 more (k = 1-65, 1-5 targets, intercept or not, three decays, gaps, null targets, values from 1e-310 to 1e170). No state layout change. *Not done:* overlapping the back-substitution of row `t` with the fold of row `t+1` (the fold's chain is the floor and the processor has the throughput to spare, but the coefficients would be read stale between calls: every accessor and the saved state would have to flush them); the block form's columns still read `R` with a stride (a transposed copy of the factor would make them contiguous at the price of a `k` by `k` copy a row).
- [x] 417. **`ew_cov` and `ew_cov_lags`'s emit** (task 414): IPC 3.8; profile by line, then overlap the independent statistics (the correlations' divisions and square roots). Bit-identical. Worker: Sonnet, high. *Built 2026-11-15 (W173; `.claude/coord/W173-417-scratch/`):* **the premise was half the harness.** Task 414's `survey_loop` sums every emitted statistic through a finite filter, a serial chain of 230 adds (about 900 cycles a row at k=20, 31% of the loop's cycles); with that out the model step is 1,080 cycles at IPC 5.6 (6,067 instructions, mispredicts under 0.1 a row), and the emit is 70% of it. The emit's cost was the divides, not a dependent chain: each of the 190 correlations was `if d > 0 { (c/d).clamp } else { NaN }`, a branch that kept LLVM from vectorizing, so 190 scalar `fdiv` ran one at a time (the cycles landed on the stores after each, per instruction). Now `correlation(c, si, sj)` takes the quotient and the clamp whatever `d` is and selects after (`r = (c/d).clamp(-1, 1); if d > 0 { r } else { NaN }`): the same number for every input (the quotient of a non-positive `d` is computed and dropped; no trap), and LLVM emits `fdiv.2d`, `fcmgt` and `bit` over two pairs at once inside `upper_rows_map`'s four-row runs (36 instructions a four pairs, IPC 7). The same reshape, with a test, in `lag_corr` (`lag_corrs_into`: each lag matrix lies in the slots as it lies in the accumulator, so a row is read and written along its length; `k²` per lag) and `partial_corr` (`partial_corrs_into`: the diagonal's square roots once, not two a pair, and each row of `P` along its length). Bits: `correlation` against the branching form over every triple of 17 awkward values (signed zeros, subnormals, overflow, NaN and its sign, infinities, +-1; the NaN, +1, -1, interior and -0 results all reached), the triangle at k 1-26, 33 and 50 against `tri.get`, `partial_corrs_into` against `partial_corr` and `lag_corrs_into` against the element-at-a-time loop on random matrices full of those values, and a model at k 2-20 over a stream with a constant, a zero, a large and a tiny column: every slot to the bit (`ewcov/emit_tests.rs`; four hand mutants of the new code each fail it). A model-level digest of 1,192 cases (k 1-50, nine stat sets, every stat, lags, window, bias, pca, zero variance, NaN/inf rows, zero-weight rows, a clone continued) is `b91b497f494028a8` on main and on the branch; the wheel's digest over 195 cases (nine stat sets, k 2-50, half-life 50 and inf, 1 and 100 groups, `bias`, 137-row chunkings, `save_bytes`/`load_bytes` and a continuation, every state hash) is identical. Counters on the Mac (cycles / instructions / IPC a row, main to branch): k=4 140 / 1,181 / 8.4 to 149 / 1,216 / 8.1 (+0.5% by the clock, within noise: 6 pairs, a divide and a clamp each now unconditional); k=20 1,080 / 6,067 / 5.6 to 765 / 5,507 / 7.2; k=50 5,815 / 25,652 / 4.4 to 3,568 / 21,250 / 6.0 (mispredicts 3.1 to 0.1 a row); `lag_corr` at k=20, 5 lags 6,978 / 58,356 to 4,211 / 27,668. ABBA, six rounds of the model step on the Mac, min (ns a row, main to branch): k=4 36.3 / 36.5, k=20 273 / 192 (1.42x), k=50 1,455 / 898 (1.62x), every statistic but `partial_corr` at 20 312 / 240 (1.30x), `window_size` 500 at 20 527 / 457 (1.15x), `lag_corr` at 20 1,760 / 1,040 (1.69x), `partial_corr` at 20 2,955 / 2,740 (1.08x: the `k³` precision is the rest); `ew_cov_lags` with `cov` and `mahal` unchanged (the lag update and the solve are their cost; `mahal`'s emit is one square root, nothing to overlap). Through the bank (wheel ABBA): k=20 1.18x, k=50 1.22x, 100 groups 1.06x, `lag_corr` 1.19x. **Left:** the bank's own cost beside the model: at k=20 the wheel spends 463 ns a row where the model step is 192, the rest the bank's plumbing for 230 output columns (task 421). Not a bit change: none found worth proposing; what remains in the correlations' loop is instruction count (about 9 a pair: the loads of four packed rows, two lane inserts, two divides, six compare-selects).
- [x] 418. **Branch-bound loops** (task 414): `seqtest` (12.6% of cycles mispredicting), `kmeans` (5.3%), `corrchange` window, `micro`: an unpredictable branch made a select where the operations stay the same; `audit`'s sort is the VM's task 390 (P36), its map sized once is here. Bit-identical. Worker: Sonnet, medium. *Built* (W174, W178): `seqtest`'s sign and zero-stake `ln_1p` and the `n_pos`/`n_neg` counts made selects (integer-built `±1`/`-0` addends); `kmeans`'s two-nearest scan and `micro`'s nearest scan made selects; each with a to-the-bit test against the branching form. Digest of `fit_predict` outputs, chunked runs, saved state and its continuation over 98 cases (k 1 to 20, 1 and 100 groups, half-life 50 and inf, zero weights, nulls, ties) equal on main's wheel and the branch's. ABBA on the Mac (5 rounds, min, load 7): `seqtest` 1.06x, `kmeans` k=4 1.12x, k=20 1.15x, `micro` k=4 1.12x; W174's counters: `micro` mispredicts 0.35 to 0.01 per row (IPC 7.9 to 8.8), `kmeans` 1.5 to 0.9 (k=4) and 1.56 to 0.90 (k=20); `seqtest` stays near 0.48 per row (the mispredicts left are in `ln_1p`'s own branches, not ours). Not done: the `corrchange` window (not profiled) and `audit`'s map sized once (task 390 takes `audit`).
- [ ] 419. **`bocpd`'s libm share and `max_run`** (task 414): libm is 60% of its cycles (ln 40%, exp 24%); find bit-identical savings beyond task 389; and why `max_run` 20 and 200 cost the same on the benchmark stream (the pruning, or a bound not applied). Worker: Sonnet, medium.
- [x] 420. **`kalman`'s passes** (task 414): instruction-bound at IPC 8.5; fuse the revert pass with predict, and use the covariance's symmetry where the result stays bit-symmetric. Bit-identical. Worker: Sonnet, medium. *Built* (2026-11-15, W177, branch `task-420-kalman`; scratch `.claude/coord/W177-420-scratch/`; `kalman.rs`). Counters (Instruments `polars-cpu-counters`, one output read a row, k=20, one target, 3M rows): `kalman` 16,594 -> 9,808 instructions and 1,881 -> 1,341 cycles a row (IPC 8.8 -> 7.3); `kalman_revert` 19,083 -> 12,354 instructions, 2,168 -> 1,670 cycles. Throughput-bound on instruction count, not latency or branches (mispredicts about 0 a row, 1 for the revert's `exp2`); the bounds-checked `p[row + jj]` indexing in `P z` was an out-of-line `index` call at 19.6% of cycles. Built: `matvec` (`P z` over slices, four rows at a time as four independent add chains, each entry's products still summed from column 0 in order) and `take_the_row` (each row of `P` walked along its own memory, the lower half taking `g_c pz_r`, the product its mirrored write took: contiguous and vectorized, no strided store). Bit-identical on any `P`, symmetric or not (test against the old forms, k 1-40, mixed scales and zeros, symmetric and not). Not built: using `P`'s symmetry in `P z` or the transition's scale (`P` is bit-symmetric by construction, but a hand-edited state need not be, and `P z` is 10% of the cycles); fusing the transition's scale with `P z` (the process noise's writes to the diagonal and the first row and column fall between them); a select form of `take_the_row` over the whole row (measured, worse: 10,019 vs 9,759 instructions). What remains at k=20 (revert): `take_the_row` 24%, the transition's scale 18%, `P z` 10%, `add_process_noise` 9% (its divisions), `map_into` 6%, `lost_definiteness` 5%.
- [ ] 421. **The bank's per-row plumbing** (task 414): ungrouped, `Series::from`'s type check is 29% of a `seqtest` bank row; at 1,000 groups the per-row cost is 6x ungrouped (row plans built, per-group zero-fill). Bit-identical. Worker: Sonnet, high.
- [ ] 422. **Task 415 and 416's mutants survivors** (CI Mutants on `dbb7d455`, 29 survived): `ewridge.rs` 27 -- `solve_in` 1069, 1433; `run_solve` 1489; `solve_scaled_through_origin` 1572; `solve_centred` 1814; the scratch pool (`quad_for` 2123-2134, `Parts::reclaim`/`release` 2193-2232, `Spare::take_factor` 2349) -- and `rls.rs` 2 (`zero_unseen` 213, `solve_block` 238). Each: a test that kills it, or an equivalence note in `scripts/mutants_equivalent.toml` with its reason; the pool's are likely cache-size thresholds where a test asserts the reuse (allocation counts) rather than the numbers. Check each locally only by `--in-diff`, alone, under `.claude/coord/mutants.lock`. Worker: Sonnet, medium.
- [ ] 423. **`corrchange`'s window: branches** (task 418's part not done): profile the window path with counters (mispredicts a row; the VM's x86 Callgrind baseline on `dbb7d455`: 20,008 Ir, 2,776 branches, 65.3 simulated mispredicts a row); make an unpredictable data branch a select where the operations stay the same. Bit-identical. Worker: Sonnet, medium.
- [ ] 424. **Latency re-check of the many-output models** (task 414's harness flaw: it summed every emitted output in one serial chain, so its latency verdicts for wide emitters are suspect; the survey's detail is lost, its summary is task 414). For every model and bank case that emits more than a few columns (`ew_cov` family, `marginal`, `hmm`, `ew_class`, `deco`, `audit`, `drift`, `breaks`, `calibration`, `conformal`, `kmeans` with distances, `bocpd` with run-length outputs, multi-target `ewridge`/`rls`): counters (IPC, mispredicts, L1D misses) with one output read, then with all read through the bank; each verdict (instruction, latency, branches, memory) restated, and a bit-identical fix where a dependency chain or a branch is the bound. Worker: Sonnet, high (survey), then a task per fix.
- [ ] 425. **Allocations sweep** (task 303 (a)): every budget case under Instruments' Allocations (or a counting allocator in a test build): per-row allocations, their callers, ranked; each a keep-the-scratch fix as task 415, bit-identical. Worker: Sonnet, medium.
- [ ] 426. **The held output-moving proposals, measured in full for the user's decision** (the user, 2026-11-16: "be sure they are completely measured with all options explored"; `.claude/coord/BIT-CHANGING-PROPOSALS.md` and task 291). One harness first (426a): two builds, every emitted column's largest and median and 99.9th-percentile relative movement, decisions that flip (gates, readiness, selected slots, test thresholds), over k 4, 20, 200 where the model allows, several targets and columns, 1, 100 and 10,000 groups, several chunkings, 1M+ rows, a save/load mid-stream, and the ABBA time on arm64 and x86 in both layouts. Then each item, with every variant worth trying measured, not only the first: (b) `emit_breaks` through faer's products and held factors; (c) `bocpd`'s split log; (d) the window operators' decay recursion without the per-row `expm1` (each candidate recursion, its drift over a long stream against the current form and against an exact reference); (e) `ew_class` diag's cached log-variances; (f) `holt`'s kept reciprocals; (g) `sgd`'s |z|^2 fold; (h) the robust refit's Gram through faer's matmul; (i) `rls`'s deferred solve (a design change: the flush on every accessor and save, its cost when read every row); (j) a profile-guided build (task 303 (c)): arm64 and x86, rustup's `llvm-tools`, the training workload and its effect on cases outside it. Each item ends with a table for the user: speed per case, movement per column, flipped decisions, fixtures and CHANGELOG it would need. Nothing merges. Workers: Sonnet, high for (a), medium for each item.
- [ ] 427. **Processor Trace on the hard cases** (task 303 (b)): `huber_k20_refit`, `corrchange_window_w100`, one bank case; if it needs a system permission, that is the user's switch, so ask. Worker: Sonnet, medium.
- [ ] 428. **`bocpd` under `gaussian`: the empty run's Cholesky held** (task 419's follow-up, W179): the prior's factor is the same every row; hold it (not serialized; rebuilt after a load) instead of factorizing it a row. Bit-identical. Worker: Sonnet, medium.
- [ ] 429. **The window operators' tax for features turned off** (the user, 2026-11-16: "do we spend time on cases where a feature is turned off but still costs instructions"). `Windows::push` checks every row for sessions, gap caps, resets, restarts, stamps, nulls, a row-count clock and grouping, whether or not the stream uses them; only `window_size` is a compile-time choice (`push<const W: bool>`), and task 412 dropped hoisting the per-row decisions (for want of a saved field, which a value derived on load would not need). Measure first: a stripped prototype of the plain case (one datetime clock, ungrouped, no session, no cap, no `window_size`, no nulls, default `closed`) against today's at one and four operators, instructions and cycles a row, and the same for each feature turned on alone, so each feature's cost when off and when on is known. If the off-cost is a real share, build the row loop specialized by the run's configuration (const-generic flags chosen once per stream or per chunk, the general path kept for everything else), bit-identical, with the digest over every feature combination. Compare against Polars' `ewm_mean_by` per row on the same case. Starts after task 426 (d) leaves `windows.rs`. Worker: Sonnet, high.
