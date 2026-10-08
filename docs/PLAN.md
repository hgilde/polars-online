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
- [x] 15. `online-cli`: TOML specs, streaming in/out (parquet, ipc, csv, ndjson — ENHANCEMENTS
      E32), progress, resume from state.
- [x] 16. CI: wheels + CLI binaries for macOS/Windows, cross-platform state test, Polars-latest
      canary job. Benchmark script + numbers in README.
- [x] 17. README with the three usage modes and the math per model.
- [x] 18. **Weekly native leak check in CI — after the repo is public.** Add
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
      **The trigger fired on 2026-09-02**: the repository is public and
      Actions is unmetered, so this is now doable work rather than a deferral.
      **Done 2026-09-03**, `.github/workflows/leakcheck.yml`: Mondays and on
      demand, ubuntu + macOS, nothing gates on it — a scheduled run that goes
      red is the report (GitHub mails the owner). Wiring it found that the
      "0 leaks" baseline was blindness, not cleanliness: pymalloc hands Python
      objects out of mmap'd arenas that `leaks` does not walk, and the script
      reported 0 leaks against a deliberate 128 MB refcount leak. With
      `PYTHONMALLOC=malloc` it sees Python objects — and the interpreter then
      leaves ~11k blocks (~700 KB) unreachable at exit whatever the workload,
      so the script became differential: 1 iteration against 1000, growth over
      500 blocks or 64 KiB is a leak (two runs of the same workload differ by
      <150). It still cannot see anything allocated through polars' allocator
      (mimalloc on macOS, jemalloc on Linux), which is every Rust-side
      allocation; 160 MB of deliberately leaked `Series` buffers did not move
      the count. That side stays with `test_ffi_memory.py` and RSS. Because a
      blind check reports clean forever, the job also runs a **control**
      (`LEAKCHECK_CONTROL=1`, one Python object leaked per iteration) that
      must fail. Real workload: growth 63 blocks / 3.7 KB, clean; control:
      +1968 blocks / 142 KB, caught. The valgrind path is written to the same
      contract and first runs on the runner — it cannot run on this machine.
- [x] 19. The expression plugin (task 8) is the in-memory form and says so: every
      `pl.col(..).online.<model>` call warns with `InMemoryExpressionWarning` naming the plan
      and the reason (§6); the README shows the two forms side by side in a closing note.
      (First committed as an off-by-default cargo feature that took it out of the wheel;
      reverted the same day — §11a.)
- [x] 20. **State out of a streamed plan — researched and implemented 2026-09-03.**
      The four-step workflow (fit online in bounded memory; export the state, optionally
      to disk; load it and predict without updating; load it and learn on) existed end to
      end on `ModelBank`, `po.run` and the CLI, and on the plan surface for every step
      but the export: `lf.online.fit_predict` is pure by decision (E33, §11a). The
      research — `docs/STATE-WORKFLOW.md`, with the engine facts measured on polars
      1.34.0/1.38.1/1.44.1 by `scripts/io_source_semantics.py` — found one sound
      form, `lf.online.fit_predict(specs, load_state=, save_state=)`: the runner's
      keywords on the plan, the state written atomically when the source has fed the
      bank its last row, idempotent under the two concurrent runs polars gives a plan
      used twice in one query. Implemented as proposed (§11a): the source feeds the bank
      only the rows a `head(n)` asked for, `load_state` and `predict(path)` are read when
      the plan is built, and the collision two concurrent writers had in `atomic.rs` is
      fixed at the root. The memory side (a plan mutating a `ModelBank`) is declined.
- [x] 21. **Gradient-boosted trees, as online as possible — investigated 2026-09-03.**
      Asked how far XGBoost's method can be pushed toward this library's contract, and
      what that does to parallel fitting and memory. `docs/BOOSTED-TREES.md` (§11g) is the
      answer: the XGBoost paper and source (`54155e3`) read with every claim cited by
      `file:line`, the streaming-tree literature and river/MOA/VW/LightGBM code compared,
      a design that keeps the contract (EW-decayed per-node gradient sums, histograms only
      on splittable leaves from a bounded pool, growth and collapse at checkpoints so the
      model is frozen between them — hence chunk-invariant and additive over threads — and
      a batch warm start on the warm-up buffer the bins need anyway), prototyped in numpy
      (`scripts/ogbt_proto.py`) and measured (`scripts/ogbt_experiments.py`) against
      XGBoost refits on synthetic drift. Nothing in the Rust crates; whether to build it is
      the user's call and §9 of the doc costs it. Research sources stay under the
      gitignored `.cache/research/`.
- [x] 22. **Online clustering, every family that can be made to fit — investigated
      2026-09-04, on the branch `online-clustering`.** The user asked for "all the
      clustering types that may be possible Online", so: every family in river 0.26.1,
      MOA, scikit-learn and Spark plus the two survey papers, each decided against the
      contract. `docs/CLUSTERING.md` (§11h) is the answer — the papers read with claims
      cited by line (DenStream's fading function, definitions and pruning thresholds;
      CluStream's micro-cluster; BIRCH's CF triple; DP-means; Cappé–Moulines eq. 15;
      Bottou–Bengio's `1/n_k` as the Newton rate), the four implementations read with
      `file:line` (none of which is both chunk-invariant and bounded, none of which
      reads a real clock, none of which labels a row before learning it), nine designs
      prototyped in numpy (`scripts/clustering_proto.py`) and measured
      (`scripts/clustering_experiments.py`) against Lloyd refits and scikit-learn's
      `MiniBatchKMeans`. Every guarantee holds bit-exactly. The one real defect of
      sequential k-means — two centres collapsing onto one component under drift, on 5
      of 20 streams — is fixed by a split–merge move on a slower clock. On seven
      hard geometries (§7.8) being online costs at most 0.04 ARI against batch
      Lloyd's; the family costs everything — every k-means and GMM scores 0.000
      on concentric rings, where `micro` (DenStream-style micro-clusters with a
      linkage macro step) reaches 0.998 / 0.999 / 0.998 on moons, rings and rows
      against DBSCAN's 1.000, with a measured rule for the threshold that decides
      it. `micro` is the design worth the build decision; §0 of the doc and
      ENHANCEMENTS §4 give the seven reasons and the structural limits. Merged to
      `main` 2026-09-04 as documentation and numpy prototypes. Nothing in the
      Rust crates; whether to build it is the user's call and §9 costs it.
      Research sources stay under the gitignored `.cache/research/`.

- [x] 23. **`kmeans` — online k-means in the crates.** `crates/online-core/src/cluster/`
      (`summary.rs`: the §6.1 mean-form summary and the diagonal feature moments the
      metric reads; `kmeans.rs`: seeding over a bounded warm-up buffer, the assignment,
      the checkpointed centre update, the split–merge move on its own clock), every
      step of `docs/EXTENDING.md`, a from-scratch numpy reference
      (`tests/reference_cluster.py`) held bit-exact, the prototype as a second oracle,
      large streams, and the edge cases §11a lists.
      Built 2026-09-05, on branch `clustering-build`: `ModelState::KMeans`, the spec
      variant with `k`, `warm_rows`, `seed_rule`, `seed`, `update_every`,
      `split_merge`, `split_merge_every`, `dead_frac`, `standardize`; outputs `cluster` (`i32`),
      `dist`, `dist2`, `n_eff`, centres as `coef` (`cluster{j}` slots, one per
      feature; `coef_index` and `unnest` follow); `ModelKind::is_unsupervised`
      refuses every residual diagnostic by name for it and for `ew_cov`. The far-row
      design §11a records replaced the doc's first split–merge move after its
      measurements failed (a seeding artefact had been read as recovery). Tests: 61
      in `tests/test_kmeans.py` (22 bit-exact against the oracle, large streams to
      40k rows, the null-row / zero-weight / constant-feature / min_periods /
      chunk-invariance / save-load edges, the stranded-centre and jumped-blob
      recoveries with their latencies pinned), the core golden and contract suites,
      the Python golden pipeline on three OSes.
- [x] 24. **`micro` — DenStream-style micro-clusters with a linkage macro step.**
      `cluster/micro.rs` on the same summary; ids monotone and never reused; the label
      is the id the row would be absorbed by, computed before the update; `macro_link`
      derived from the observed spacing at each checkpoint (§6.5) with the parameter
      as an override; accuracy measured in-test against a numpy DBSCAN on moons and
      rings.
      Built 2026-09-05, on branch `clustering-build`: `ModelState::Micro`, the spec
      variant with `eps` (required), `beta_mu`, `max_clusters`, `prune_every`,
      `macro_link`, `standardize`; outputs `cluster` (`i64` macro label), `dist`,
      `micro` (`i64` id), `outlier` (`bool`), `n_clusters`, `n_micro` (`i32`),
      `n_eff`, and a ragged `coef` (`Source::Id` and `Source::Flag` are new; the
      id and the label are separate columns, so §6.5's rule 2 holds for the id and
      the label is still a cluster). The admission rule, the derived link, the ξ
      semantics and the two failure regimes are in §11a below. Tests: 63 cases in
      `tests/test_micro.py` (17 bit-exact against the oracle over four geometries,
      seven knob settings, nulls/weights/clock gaps and `predict`; 20k-row shapes
      against an in-test DBSCAN ceiling, 200k rows in 4-D, a cluster born and one
      dying, 5% noise; ids, the cap, promotion and pruning traced row by row, a
      heavy row, the infinite halflife, standardization, chunk invariance,
      save/load, groups, the ragged `coef`, the expression and CLI paths, every
      refusal), 21 core unit tests, the golden and contract suites, the Python
      golden pipeline.
- [x] 25. **E36: adaptive conformal intervals** (`lo`/`hi`/`coverage` per slot) on every
      regression model, O(1) state per slot; oracle + large-data coverage tests.
      Built 2026-09-05, on branch `clustering-build`: `online_core::Conformal`
      (`conformal.rs`, with `norm_ppf` for the warm start), spec fields `conformal`
      (the coverage level) and `conformal_rate` (0.05, in units of the slot's
      `sigma`); `StreamState.conformal` as a `#[serde(default)]` per-slot vector
      (schema stays 2); `Source::Conformal`; fields `lo_<slot>`, `hi_<slot>`,
      `coverage_<slot>` after `resid_z`. The fields are not `pred_lo`/`pred_hi`
      as sketched above: `pred_` marks a prediction for `eval.unpack` and the
      README's field grammar, and a bound is not one. The update rule, the warm
      start and the guarantee are in §11a below. Tests: 70 cases in
      `tests/test_conformal.py` (a longhand replay bit-exact for every regression
      model over a grid, nulls, zero and varying weights, an irregular clock and
      groups; the telescoped `1/T` bound as a hard inequality and coverage
      within 0.01 of target on four 200k-row residual regimes; fields, validation,
      refusals, nulls, zero weights, warmup, out-of-sample-ness, chunk
      invariance, save/load, `predict`, a drift reset, the runner and the
      expression), 9 core unit tests, the golden and API-surface snapshots.
- [x] 26. **E37 + E38 on `ew_cov`:** Mahalanobis distance (`stats: "mahal"`) and EW-PCA at
      checkpoints (`pca`, `pca_every`); oracles via `gram()` and numpy.
      Built 2026-09-05, on branch `clustering-build`: `EwCovStat::Mahal` (one
      slot `mahal`, `solve_spd` on `C + s·prior·I` each row, so it needs
      `precision_prior`), `EwCovCfg::{mahal_quantiles, pca, pca_every}` (all
      `#[serde(default)]`, schema stays 2), `online_core::Pca` (faer
      `self_adjoint_eigen`, continuity-signed), fields `mahal`, `mahal_q<p>`,
      `pc<j>_var`, `pc<j>_share`, `pc<j>_<feature>`, `pc<j>_score`;
      `output_index` kinds `mahal`, `mahal_q`, `pc_var`, `pc_share`,
      `pc_score`, `pc_loading`. Decisions in §11a. Tests: 40 cases in
      `tests/test_ew_cov_scores.py` (a Welford replay of the stream — clock
      gaps, `max_dclock`, zero and null weights, skipped rows — held to the
      solve at 1e-9; numpy `eigh` with the continuity rule at every refresh
      for two cadences; χ² calibration and a 3-factor recovery at 200k rows;
      a covariance switch; chunk invariance, save/load mid-cadence, `predict`,
      the expression, the runner and a TOML config; fields, validation, edge
      cases), 31 core unit tests, the API-surface, error-message and
      output-index suites extended.
- [x] 27. **E39: class-conditional `ew_cov`** (`class` column, per-class moments).
      Built 2026-09-05, on branch `clustering-build`, as a model of its own:
      `po.spec.ew_class(name, features=, label=, classes=, covariance=,
      precision_prior=)` — one `EwCov` per declared class, scored by Bayes'
      rule over Gaussian classes with `covariance` = `full` (QDA, default),
      `shared` (LDA: the class-weighted pool, one factorization) or
      `diagonal` (naive Bayes). Outputs `class` (String), `p_<class>` per
      class, `n_eff`, `coef` = the class means (`coef_<class>_<feature>`).
      `online_core::{EwClass, EwClassCfg, Covariance}`,
      `solve::quad_forms_logdet` (all quadratic forms and the log-determinant
      off one Cholesky), `ModelState::EwClass` (schema stays 2). The label
      column rides as `targets[0]` through the bank and is read as a key
      (`label_column`: cast to String, mapped to the class index, an
      undeclared value is an error naming the row); `Source::Label` and
      `F64Column::finish_label` materialize the class name. Decisions in
      §11a. Tests: 51 cases in `tests/test_ew_class.py` (an in-file replay
      oracle — per-class weighted Welford in the core's operation order —
      holding weights, means and `n_eff` bit-exact and the posteriors to
      1e-9 for every shape over null labels, null features, zero and null
      weights and a capped irregular clock; 200k rows × 6 features × 3
      classes within 0.001 of the Bayes rate and calibrated to 0.01; the
      shapes told apart on data that separates them; a class swap relearned;
      an unseen class, late labels, integer/boolean/categorical labels, an
      undeclared label, the input bound, chunk invariance, save/load,
      `predict`, groups, the grid, `coef` and its index, the expression, the
      lazy path, the runner, the CLI and the refusals), 14 core unit tests
      plus 2 for the solve, the golden, contract, API-surface, error-message
      and registry suites extended.
- [x] 28. **E40: constrained coefficients** on `sgd` / `pa`: `coef_min` /
      `coef_max` (a number for every slope or one per feature, `inf` for no
      bound) and `coef_sum`, one projection (`online_core::Constraint`) for
      the box, the simplex and any box with a sum; intercept free; the
      constraint in the caller's units under `scale_features`. Verified by a
      Python replay of both models and the projection held bit-exact to the
      bank over six constraint sets × three schedules / three PA modes with
      nulls, zero and NaN weights and an irregular clock, an independent
      bisection-and-KKT check of the projection, 200k-row recovery on the
      simplex / a sign / a hyperplane / a box / a million-fold scale gap,
      and the edge cases (pinned slopes, list vs scalar, explicit ±inf,
      several targets, zero-weight and null-target rows, the input bound,
      chunk invariance, save/load, `predict`, groups, the expression, the
      lazy path, the runner, the CLI, every refusal by name); 9 core unit
      tests in `constraint.rs`, 7 in `sgd.rs`, 4 in `pa.rs`, two goldens.
- [x] 29. **E41: diagonal transition `φ^d`** on `kalman` (coefficient dynamics):
      `revert_halflife`, a scalar or one per slot (intercept first, `inf` =
      random walk, the default), `β ← Φβ`, `P ← ΦPΦ + Q·d` before each
      row's prediction, `predict` propagating by the same `Φ`. Verified by
      the Python replay held to 1e-9 against the bank over thirteen
      configurations (per-slot / shared `P`, explicit `q`/`obs_var`/`p0`,
      nulls, skipped features, zero weights, no intercept,
      `standardize=False`, a capped gap), the exact shrink over an irregular
      clock with skipped rows to 1e-11, the covariance bound
      `q/(1−φ²)` after 200k null rows, 300k-row tracking (sparse / dense /
      random-walk truth), and the edge cases (`inf` bit-identical in every
      spelling, scalar == list, zero-weight rows, `predict` over the clock
      distance, chunk invariance, save/load, groups, the expression, the
      lazy path, the CLI, every refusal by name); 7 core unit tests, the
      contract with and without reversion, two goldens.
- [x] 30. **E42: a sequential e-process test** — `seqtest`, a model in two
      modes: the sign of each target column, or, with `a`/`b`, whether one
      spec of the bank predicts closer than another (`|resid_b| − |resid_a|`).
      Two one-sided Kelly bettors per target at the Krichevsky–Trofimov
      stake on the counts before the row; `log_e_pos/neg`, `n_pos/neg`,
      `n_eff` (`log_e_a/b`, `wins_a/b` in compare mode), all before the
      row. The bank runs in two phases (every other spec, then the
      comparisons over the residuals just assembled) and returns columns in
      spec order. Verified by a scalar replay to 1e-12, the closed-form KT
      wealth `2ⁿ B(n⁺+½, n⁻+½)/π` through `math.lgamma`, `po.eval.seqtest`
      bit-identical to the bank on 1M rows (column) and 300k (compare), the
      guarantee as a crossing rate over 20k fair-coin and 20k dependent-null
      streams (2.2% at α = 5%, 0.4% at 1%), power at 60% / 55%, and the
      edge cases (ties, nulls, ±inf, beyond-bound, denormals, integers,
      warm-up with `n_eff` through it, per-target `min_periods`, chunk
      invariance incl. interleaved-group comparisons, save/load, pickle,
      groups and a null key, session and clock restarts, lazy path, runner,
      CLI incl. `--dry-run`, the expression and its `a`/`b` refusal, two
      comparisons in one bank, scoring, a refused chunk, every refusal by
      name); 14 core unit tests, the contract, two goldens, 9 bank tests.
- [x] 31. **Performance and parallel-performance deep dive** over the new models and
      enhancements (`docs/PERFORMANCE.md` §13, `benchmark.py`). Bit-exact
      throughout, two dumps against the Task 30 build say so; per 400k rows
      at k = 20 the default `ew_cov` 758 → 222 ms, full `ew_class` 1510 →
      808, PCA at `pca_every=100` 332 → 162, `micro` 37 → 24, the simplex
      179 → 157; at 14 threads over 64 groups `ew_cov` 276 → 152, `ew_class`
      324 → 191 per 800k rows. The finding: a wide model's speed depended on
      the caller's chunk size through the slot stride (a power-of-two chunk
      2–3× slower), fixed by processing in cache-sized runs with an odd-line
      stride.
- [x] 32. **Prepare 0.2.0**: version bump, CHANGELOG, README and VALIDATION numbers
      regenerated, gate and CI green. Tag and Release dispatch are the user's steps.
- [x] 33. **Diagonal standardizer for `kalman` and `sgd`** (`EwDiag`), the O(k²)
      → O(k) item Task 31 deferred; `SCHEMA_VERSION` 2 → 3 with schema-2 loaders
      for both models and a frozen schema-2 bank fixture
      (`crates/online-polars/tests/state_schema2.rs`). Bit-exact: the Task 31
      dump recipe against a build of the previous commit, plus `EwDiag` vs
      `EwCov` compared as bits. Per 400k rows at k = 20: `kalman` 223 → 179 ms,
      `sgd` with `scale_features` 109 → 62; at k = 50 1065 → 908 and 279 → 121.
- [x] 34. **Last-row diagnostics in the bank file**: the output-struct fields as
      of the last training row, per (spec, group), saved with the state and
      readable from a loaded bank without the output frame
      (`ModelBank.last_row()`, `Bank::last_row`). Additive field, no format
      bump (the `rows_fed` precedent).
- [x] 35. **The training data in the bank file**: per (spec, group), what the
      stream was fed -- row counts (fed, processed, skipped, learned, zero
      weight), the weight sum, the clock range, session changes, backwards
      clocks and resets, and per-column count / nulls / mean / std / min /
      max over every row fed -- saved with the state, read back as
      `ModelBank.summary()` and `ModelBank.describe()`. Undecayed, in row
      order (Welford), so chunking cannot move a bit. Additive field, no
      format bump; a 0.1.x file reports nulls. With it, the state-file tests
      made rigorous: every facet equal across save/load, a re-save byte
      for byte the first, truncated and bit-flipped files refused without a
      panic, and a frozen 0.2.0 fixture (`state_schema3.rs`).

Tasks 36–43 are `docs/ENHANCEMENTS.md` §9 (E43–E53): what the bank needs
to run at the width, target count and block count its design allows.

- [x] 36. **`ew_cov` with no per-row output** (E43): `stats=[]` is legal and
      means "accumulate only"; the spec emits `n_eff` and its value is its
      state (`gram()`, `describe()`, `summary()`).
- [x] 37. **`marginal` model** (E44): per-(feature, target) EW mean, variance,
      covariance, Σw and Σw² — `O(p·T)` per row, `n_eff` the only output —
      read as a long frame by `ModelBank.marginal()`.
- [x] 38. **Complete the Gram export** (E45): `gram()` gains per-target means
      and variances and Kish `n_kish` (features and per target); Σw² tracked
      beside Σw. Additive fields, legacy states report `None`.
- [x] 39. **`po.gram`** (E46): merge / subset / correlation / solve /
      lasso_path / coef_stats / vif / condition over `gram()` dicts, numpy
      only, each held against the model that computes the same thing online.
- [x] 40. **`label_delay`** (E47): a common parameter that holds each learned
      row until the clock reaches `t + delay`; plus `po.prep.embargo` for the
      doubled-stream recipe.
- [x] 41. **The Gram update, measured** (E48): the symmetric-half idea is
      rejected — it is neither bit-identical nor faster — and the deviation
      hoist that *is* both ships in its place.
- [x] 42. **Mergeable evaluation sums** (E49): `po.eval.sums` /
      `merge_sums` / `from_sums`.
- [x] 43. **A run with no per-row output** (E50): `po.run(output=None,
      save_state=)`, `online --no-output`; with it E53, `targets` optional
      in TOML for the unsupervised models.
- [x] 44. **Freeze a schema-4 fixture**: `state_schema4.rs`, as
      `state_v1.rs`, `state_schema2.rs` and `state_schema3.rs` freeze theirs,
      now that tasks 40–43 have stopped moving the layout.

Tasks 45–56 build `docs/ENHANCEMENTS.md` §10 (E54–E64) in the order that
section gives, plus the two tasks the examination found were missing: 47,
the ring-clearing signal three of the models need and no model currently
receives, and 56, the fixture that closes the batch. Each is written to be
built without re-deriving the design — the decisions, the maths, the state
and the oracles are in §11a under *Preparing E54–E64 for implementation* —
and every new model walks `docs/EXTENDING.md`'s eighteen steps. E63 is a
note, not a task.

- [x] 45. **Closed-group emission** (E54): `group_close = "monotone" |
      "session"` as a common parameter; `ModelBank.closed_groups(spec=None,
      *, drop=True)`; the sidecar table through `po.run(closed_groups=)`,
      `lf.online.fit_predict(closed_groups=)` and `online --closed-groups`;
      `po.gram.from_row`. `SCHEMA_VERSION` 4 → 5: a spec field moved every
      bank file's bytes (the `label_delay` precedent); schema-4 files load
      and continue to the bit. Acceptance: a closed row equals `gram()` read
      at the same point, field for field and bit for bit; `eig_*` equals
      `numpy.linalg.eigh` of the row's own `comoments`; the sidecar's bytes
      are invariant to chunking; a smaller later key, a null key, a
      non-orderable key dtype, `group_close` without `group`, `"session"`
      without `session` or with `session_gap`, and `group_close` with
      `label_delay` are each refused by name; the last group never closes
      and stays readable through `gram()`; a mid-stream save/load keeps the
      unclosed groups, the high-water mark and any undrained rows.
- [x] 46. **`deco`** (E55): Engle–Kelly equicorrelation as an `OnlineModel`,
      `O(m)` a row on `EwDiag`-standardised features; `dynamics = "ew" |
      "linear"`, `blocks`; outputs `u`, `rho`, `loglik`, `n_eff`, `coef =
      [rho]`. Acceptance: `u` equals the longhand pair sum; one pair under
      `"ew"` equals `ew_cov`'s `corr` on the same standardised columns to the
      bit; `loglik` equals a dense `numpy` Gaussian density; one block holding
      every feature reproduces the unblocked `u`; a zero-weight first row is
      guarded; the standard sweeps and the clock-rescaling test pass.
- [x] 47. **The ring-clearing signal**: `ClockAdvance::capped`,
      `RowPlan::capped`, `OnlineModel::clear_lags` (a default no-op), called
      by the stream on a session change or a capped gap (a reset already
      rebuilds the model). The rule every row-lagged state follows from here
      on, with its EXTENDING.md step. Enabling task for 48, 50 and 54.
- [x] 48. **Lagged co-moments on `ew_cov`** (E56): `lags=[...]`, `stats=[...,
      "lagcorr"]`, `C_ℓ' = a·C_ℓ + a·b·d_t d'_{t−ℓ}` with both deviations
      against the pre-row mean; `gram()` gains `lags` and `lag_comoments`;
      the E54 row carries both. Acceptance: `ℓ = 0` equals `comoments` to the
      bit; the longhand `numpy` recursion agrees; the ring clears on the
      three events of task 47 and on nothing else; `n_eff` and Kish are
      bit-identical to the spec without `lags`; chunk invariance.
- [x] 49. **`po.prep.refresh_time`** (E58): refresh-time sampling as a Rust
      operator in `online-polars` (`refresh.rs`, no model), exposed through
      `online-py` and wrapped as a lazy IO-plugin source in `po.prep`, the way
      `lf.online.fit_predict` is. Acceptance: a longhand Python loop on
      Poisson streams; an 8/9/10-tick three-series example with `N = 7` and
      retained fraction `21/27`; a synchronous input returned unchanged;
      identical output from 1 and 1000 batches.
- [x] 50. **`rcov`** (E57): the multivariate realised kernel, the
      pre-averaged (modulated) realised covariance and the plain realised
      covariance as a group-scoped, undecayed `OnlineModel` whose value is its
      E54 row. Acceptance: `kind = "plain"` equals `n ×` `ew_cov(lam = 1)`'s
      raw second moment at close, to the bit; the kernel and the pre-averaged
      estimator each equal a longhand `numpy` implementation on the same
      returns; Parzen PSD on adversarial streams; the `ψ`/`Φ` constants at
      their closed forms; a block with fewer than `2·jitter` rows gives nulls;
      zero-weight rows stay out of the ring; chunk invariance.
- [x] 51. **`po.corr`** (E62): the correlation-matrix helpers in `gram.py`'s
      style — Fisher z, Higham's nearest correlation matrix, constant-target
      shrinkage, equicorrelation, absorption, spectral and block forms,
      Marchenko–Pastur edges, signal share, forecast losses, the Epps
      inversion, Fisher standard errors. Acceptance: Higham's published
      examples; PSD and unit diagonal on random inputs; `qlike` zero at
      `fcst = real` and positive elsewhere; `equicorr_row` equals task 46's
      `u`; every function held to a longhand check.
- [x] 52. **`po.sim.regimes`** (E64): the seeded regime simulator, `numpy`
      only, with asynchronous observation, noise, AR(1), a volatility state,
      a cycle_profile factor and a volume process; returns `rows`, `truth_rows`,
      `truth_blocks`. Acceptance: block correlations recover the per-state
      matrices within the Fisher-z floor; the Epps curve of an asynchronous
      run rises with the sampling interval; a seed reproduces bytes.
- [x] 53. **`hmm`** (E60): the Hamilton filter over `K` Gaussian states with
      responsibility-weighted `EwCov`/`EwDiag` updates, an EW transition
      estimate from the filtered joint, `kmeans` seeding, `exog_tvtp`.
      Acceptance: the reduction to `ew_class` (Π uniform, `learn = False`,
      the states built from a fitted `ew_class`'s own `EwCov`s) to the bit; a
      longhand `numpy` filter at fixed parameters; `predict` is the step
      without the step; recovery of Π and the means on task 52's streams as a
      `docs/REGIMES.md` experiment; the sweeps.
- [x] 54. **`corrchange`** (E59): the Wied–Krämer–Dehling (2012)
      closed-sample constancy test run span by span (`horizon` rows, the
      paper's `D̂`, Kolmogorov critical values, Bonferroni over pairs) and
      the two-window `vech` statistic with a permutation critical value.
      Acceptance: the monitor's size and power reproduce WKD's Table 1
      (`.040/.035/.041` at `T = 500`; `.587` power on a `0.5 → 0.7` step)
      to Monte-Carlo error; `D̂` and `Q` against longhand `numpy` to
      `1e-12`; the window statistic to the bit; run length to false alarm
      and detection delay of the window kind on task 52's streams in
      `docs/REGIMES.md`. The Wied–Galeano (2013) sequential detector is a
      §10 follow-up, not part of this task.
- [x] 55. **`bocpd`** (E61): Adams–MacKay run-length recursion with
      normal-inverse-Wishart, per-feature normal-inverse-gamma and the robust
      diffusion-score-matching posterior; tail truncation. Acceptance: a
      longhand `numpy` Algorithm 1 on a univariate stream; a variance step
      detected; truncation changing nothing above `truncate`; the robust
      variant ignoring one 20-σ row where the Gaussian one restarts; the
      sweeps.
- [x] 56. **Freeze a schema-5 fixture and close the batch**:
      `state_schema5.rs` as task 44 froze schema 4, once 45–55 have stopped
      moving the layout, carrying a monotone bank with an undrained closed
      row, an `ew_cov` with `lags`, and one of each new model; ENHANCEMENTS
      §10's rows gain their task numbers; the README's model table, the
      CHANGELOG's `[Unreleased]` and `docs/RELEASE-READINESS.md` are brought
      up to the batch.
- [x] 57. **Fix what the review of 45–56 found** (`docs/REVIEW-E54-E64.md`):
      29 items -- four of them wrong answers rather than missing guards --
      each with the test that catches it, and one (B4) made and then
      reverted when Linux CI showed what it cost. The four: `ew_cov`'s lag ring read
      its depth from `VecDeque::capacity()` and never refilled after a save
      taken before it was full (L1); `rcov`'s `clear_lags` dropped the
      end-jitter ring instead of closing the stretch, losing `m` returns and
      re-emitting the next one `m + 1` times (R2); `hmm`'s `min_periods`
      gated the *update* as well as the report, so under decay a filter
      could never learn at all (H1); `bocpd` silently dropped a row whose
      hazard column was `<= 1` (B1). Acceptance: a regression test per item,
      the `predict == step` contract extended to the value a model reads out
      of the targets slot, and hard rule 8's recursion checked against
      zero-weight rows for every model.
- [x] 58. **Documentation pass, 2026-09-06.** Every document read against the
      code and brought current; the README gains a table of contents, a link
      from each model to its builder in the API reference and to its Rust
      source, and two regroupings (*Preparing a stream*, *Reading the fit*);
      every builder docstring states each keyword's default, checked against
      `stream.rs`; `docs/README.md` maps every document to what it is for;
      each document under `docs/` opens with a dated status line. Section
      numbers and file names are unchanged, because the code cites them
      (§11a records the rule). Acceptance: the README's blocks still run
      (`TestReadmeExamples`), every builder still has its README heading
      (`test_the_readme_documents_every_model`), `sphinx-build -W` is clean.
- [x] 59. **`llms.txt`, 2026-09-06.** The [llmstxt.org](https://llmstxt.org)
      map for coding agents at the repo root, copied into the docs build by
      `html_extra_path` so the same file answers at
      `hgilde.github.io/polars-online/llms.txt`. It carries the two streaming
      surfaces in full and the rules an assistant otherwise guesses wrong,
      chief among them that `po.spec.ewridge` is `type = "ew_ridge"` in a spec
      dict — the one name spelled two ways. Acceptance: `tests/test_llms_txt.py`
      holds every model name to the registry, every repository link to a file
      that exists, every README anchor to a heading and every reference link to
      a built page, and each of those five assertions was checked against a
      deliberately broken copy.
- [x] 60. **The README links into the API reference, 2026-09-06.** The model
      table's name links each model to its builder's entry in the Python API
      reference, and a `math` link beside it still reaches the section below;
      the first prose mention of each public object — `ModelBank`, `po.run`,
      the frame namespace, `po.eval`, `po.gram`, `po.corr`, `po.prep`,
      `po.sim`, the spec helpers, `InMemoryExpressionWarning` — links to its
      entry too. The Python reference is the destination everywhere: it is the
      surface a caller uses, and it is what the docstrings are built from. The
      `*Rust:*` links stay source links, because rustdoc is neither built in CI
      nor published (the crates are not on crates.io, so there is no docs.rs);
      if it is ever published, that line is where it goes. Acceptance:
      `tests/test_api_links.py` resolves every reference link to a page that is
      built and an object that exists, holds the table to the registry, and
      holds every model section to its `*API:*` and `*Rust:*` lines — all four
      checked against a deliberately broken README.
- [x] 63. **`window`: an EW accumulator with a hard cutoff.** A halflife `h`
      with `window = 3h` must guarantee that nothing older than `3h` of clock
      contributes at all — not "contributes 12.5%". Designed in §13 below;
      the colliding names are renamed first (task 63a), the mechanism and
      `ew_cov` land next (63b), the Gram models after (63c).
  - [x] 63a. **Free the word.** `rcov`'s `window` is a pre-averaging length in
        ticks and becomes `preavg_rows`; `corrchange`'s `window` and
        `horizon` are one concept — rows per comparison block — under two
        names, and become `span_rows`; `bocpd`'s `truncate` is a probability
        floor and becomes `prune_below`, since "truncate" now means the
        window. No compatibility shim and no dual spelling: the spec keys are
        renamed, the frozen state fixtures are regenerated, and a state saved
        by 0.2.0 does not load.
  - [x] 63a′. **The naming pass, 2026-09-07.** `preavg_ticks` was market
        jargon in a library whose clock is deliberately generic, and five more
        names did not say what they were: `rcov.n_max` → `block_rows`,
        `rcov.h_max` → `max_bandwidth`, `kmeans.sm_every` →
        `split_merge_every`, the ridge family's `coef0` → `coef_prior` (it is
        the prior mean, not the coefficient of feature 0), and
        `refresh_time`'s `n_obs_<s>` column, which had been `n_ticks_<s>`.
        Names that appear as symbols in a formula the docstring quotes —
        `theta`, `jitter`, `q`, `p0`, `c`, `beta_mu`, `share_p` — were left,
        because renaming them breaks the correspondence a reader checks
        against the cited paper.
  - [x] 63b. **The mechanism, and `ew_cov`.** `Snapshots<S>` in
        `online-core`, the truncated view, `window` and `window_every` on the
        `ew_cov` spec, refused by name everywhere else. Acceptance: the
        oracle in §13.4.
  - [x] 63c. **The Gram models.** `ewridge` done 2026-09-07: the Gram, the
        per-target cross-moments and the residual variance are truncated by
        the same identity, and the solve runs on the result, so the fit
        provably contains no row older than the window. `n_eff`, `sigma` and
        `resid_z` follow it. Refused with `ridge_decay` (the decaying prior's
        scale is a product over the whole stream) and with `session_shrink`
        (a second accumulator under a longer halflife). Acceptance: a direct
        weighted-least-squares solve over the rows inside the window, in Rust
        and again in Python against numpy.
        `lasso` done the same day: it solves from the same Gram, so the view
        threaded through `standardized()` alone. Its selection error is
        truncated with it — choosing `lambda` on the whole history while
        fitting on the window picks a path point for rows the coefficients no
        longer see — which `-D warnings` caught as an unused binding after the
        docstring already claimed the behaviour. A window can therefore change
        the *support*: a feature with no in-window evidence goes to exactly
        zero.
        `marginal` done the same day: it emits nothing per row, so the window
        applies at the readout -- every moment a pair is built from, the
        weight and both means and the three centred second moments, truncated
        one pair at a time so a readout stays O(1). The case for it is
        sharpest here: two regimes of opposite sign average to nothing over a
        long history, so an unwindowed screen reports corr 0.0006 where a
        40-unit window reports -0.99. *Two more of mine the tests caught*:
        `Marginal` was a unit-like spec variant (`Marginal {}`), so the fields
        had to be added to the enum and four `matches!` patterns updated; and
        every part of the window was written except the ring maintenance in
        `step`, which left it inert until the oracle failed.
        `ew_class` done the same day, and it is the one that costs: pure
        decay leaves a covariance unchanged, which is why `covariance="full"`
        caches its factor between rows, and a *truncated* covariance moves
        every row — so the cache dies every row and the shape pays one
        `O(k^3)` factorization per class per row. Stated in the keyword's own
        docstring and measured in `docs/PERFORMANCE.md` §16, which also shows
        the windowless path unchanged (±4%, both signs, against a build from
        before the feature). All five models of §13.2's first row now carry
        `window`.
        *Two things the tests caught*: the compact msgpack encoding writes a
        struct as an **array**, so a `skip_serializing_if` field must be
        **last** or it shifts every field after it — the state round-trip
        failed with "invalid type: boolean, expected f64" until both the
        model's `win` and the config's two keys moved to the end. And the
        first oracle disagreed because the *test* fed a clock that ran
        backwards: `lcg` there is `[-1, 1)`, not `[0, 1)`.
- [x] 64. **Document the output struct of every model** (`docs/ENHANCEMENTS.md`
      E65). Each model's README section and builder docstring gains a table of
      the fields it writes — name, dtype, when null, which switch adds it —
      and a test holds each documented set to `po.spec.output_fields(spec)` for
      a canonical spec, so an undocumented field cannot ship. Today only the
      *grammar* of the names is written down, and the per-model prose is
      inconsistent: `corrchange` lists its outputs, `kmeans` does not.
      Done 2026-09-07 as `docs/OUTPUTS.md`, one section per model, rather
      than a table in twenty-one docstrings: the *field lists* are generated
      from `output_fields` on a canonical spec and the *meanings* are written
      once per field stem in `scripts/outputs_doc.py`, so the two cannot
      disagree. Each model's README section links to its section beside the
      `*API:*` and `*Rust:*` links. `tests/test_outputs_doc.py` regenerates
      and compares, so a new field cannot ship undocumented — the generator
      writes `**undocumented**` for a stem it has no meaning for, and a test
      fails on that string.

- [x] 65. **`marginal(lags=)` and `n_serial`, 2026-09-07**
      (`docs/ENHANCEMENTS.md` E66, `docs/MARGINAL-LAGS-AND-BINS.md`).
      `marginal`'s `t` is built on `n_kish`, which is right for unequal
      weights and silent about serial dependence: on a smooth stream
      consecutive rows are nearly the same observation, so `t` reports
      evidence that is not there, and nothing in the contemporaneous moments
      can see it. `lags=` accumulates the pair's moments at those lags with
      `ew_cov`'s recursion — bit-identical to `ew_cov(lags=)`, which is the
      strongest check available, since both centre at the pre-row mean and
      mix with the same `a`/`b`. From them `serial_rule` forms Bartlett's
      factor and reports `n_serial`, `t_serial`, `phi_x`, `phi_y`, plus the
      four `lagcorr_*` list columns — of which `lagcorr_xy` against
      `lagcorr_yx` says whether a feature leads or follows its target, which
      is worth having on its own (2026-09-27: the lead/follow reading holds
      for two series that describe the same moment, not against a
      forward-looking target, E75). Two independent AR(1) series at
      `phi = 0.9` and `0.8` give `t = 2.39` and `t_serial = 1.03`. Three
      names changed from the sketch; see the enhancement's row for which and
      why. No schema bump: the bank file is a msgpack *map*, so a key with a
      `default` is backward-compatible.

- [x] 66. **`marginal(bins=)` and `split_gain`, 2026-09-07**
      (`docs/ENHANCEMENTS.md` E67, `docs/MARGINAL-LAGS-AND-BINS.md`).
      Everything else `marginal` reports is linear, and a feature can be
      strongly related to a target with `corr` at zero. A histogram of the
      target's moments inside the feature's bins gives the response curve
      (`bin_edges`, `bin_n`, `bin_mean_y`, `bin_var_y`) and the best single
      cut of it (`split_gain`, `split_at`, `split_gain_t`) — a regression
      stump's gain, `O(bins)` of state per pair and a binary search per pair
      per row. Decay stays `O(1)` per row by keeping the sums undecayed
      against one scale factor per group, renormalized at `s < 1e-150` on a
      row the clock alone decides. Learned edges hold their warm-up rows and
      replay them, so the histogram is what it would have been had the edges
      been known first — checked to the bit. Four parameter names and the
      quantile mechanism changed from the sketch; the design doc's API
      section says which and why. Ragged bins, since a binary feature has two
      and a constant one has none. `bins` with `window` is refused.

      Three bugs found by the tests written for it, all pre-existing or
      newly introduced here and none visible in shipped behaviour:
      (a) E66 had put a `skip_serializing_if` field in front of `win`, which
      breaks the *compact* msgpack encoding, where a struct is a bare array
      and a skipped field slides everything after it — `tests/state_encoding.rs`
      now sweeps every combination in both encodings; (b) `window_every`
      without `window` was validated by `ew_cov` and `ewridge` but not by
      `marginal`, `lasso` or `ew_class`, and in the compact encoding it
      decoded *silently* as `window = 1`; (c) `serde_json`'s default float
      parser is fast rather than correctly rounded and moved a spec float by
      one ulp — nothing for a halflife, everything for a bin edge, since
      edges read back from an earlier run are data values and a row sitting
      exactly on one is the common case. The crate's `float_roundtrip`
      feature fixes it; `crates/online-polars/tests/spec_floats.rs` holds it.

      And two costs that a test cannot see, found by measuring the plain
      path against a build from before either task (`docs/PERFORMANCE.md`
      §17). A stream with no lags and no bins had gone from 69.2M to 58.0M
      rows/s: task 65 had put the lag update inside `learn`'s per-target
      loop, and a call there -- even one never taken -- makes the compiler
      assume the callee could reallocate `mx`, `sxx` and `sxy`, so it
      reloads their base pointers every iteration and stops vectorizing;
      and the `O(p)` finiteness scan that guards the lag ring sat outside
      the `if let Some(lag)` that needed it, so every row of every
      `marginal` walked all `p` features to decide whether to push into a
      ring that did not exist. Hoisting the first into its own pass
      (`learn_lags`, bit-identity intact) and moving the second inside its
      guard puts the path back at 72.2M. Both look like ordinary guard
      clauses in a diff, which is the argument for keeping a benchmark
      cheap enough to run: `cargo run --release -p online-core --example
      marg_bench`.

- [x] 67. **Review of tasks 65 and 66, 2026-09-07.** Each changed model
      read in its current form for bugs, logical errors, performance and
      the edge cases its tests missed; every finding applied, since nothing
      had been released or pushed, so the layouts could change with no
      `SCHEMA_VERSION` bump. What changed, and the decision behind each:

      (a) **A capped gap is a `λ` of zero.** `max_dclock` caps a gap at
      `max_dclock`, and at `halflife=10`, `max_dclock=1e5` that is
      `2^-10000 = 0` exactly; the histogram's scale went to zero and every
      later row added `w/0`. `decay` now folds a factor that would take the
      scale under `1e-150` into the weights (multiplying by `s` and `λ`
      separately, since their product is what proved too small) — a zero
      wipes the histogram, which is what the pair moments do on that row.
      (b) **Per-bin Welford.** The sketch's `(Σw, Σwy, Σwy²)` loses the
      variance to cancellation at a modest offset — a target at `1e7` with
      noise of `1e-3` reported none — while every other accumulator in the
      crate is mean-form. Each bin now carries `(w, mean, M2)`; same state,
      one more multiply. The gain is formed as `(d_L/w_L)·(d_L/w_R)/var`,
      because the undecayed weights reach `1e150` before a fold and three of
      them multiplied together overflow.
      (c) **Weighted, point-mass quantile edges.** Plain quantiles of an
      indicator that is zero on 95% of rows all sit on zero and collapse to
      no edge, so the 5% that carry the signal share the zeros' bin — the
      commonest wide-input feature there is, unsplittable. Edges are now
      placed one at a time, each closing a bin of the remaining weight
      divided among the remaining bins, and an edge that would close an
      empty bin moves up to the next value: a point mass fills a bin of its
      own. Weighted by the row weights, not their decay (which says when a
      row arrived, not what the feature looks like). `bin_edges` is refused
      beside `bins`/`bin_rule`/`bin_warm_rows`, and budgeted as a learned
      histogram is.
      (d) **A missing target holds the lag moments.** `marglag.rs` decayed
      them on a row where the target was absent, on top of the ageing `W_t`
      already carries: a hundred null rows at halflife 20 took a `lagcorr`
      of 0.75 to `0.75·2^-5` while `corr` stood still, `n_serial` doubled
      and `t_serial` inflated — invisible at the `halflife=inf` the tests
      ran at. Now it holds, as the pair moments hold (`a = 1, b = 0`).
      (e) **`lagcorr_*` are `ew_cov`'s numbers.** The lists were clamped to
      `[−1, 1]` and the auto terms divided by `var` rather than `sd·sd`;
      neither is what `ew_cov(lags=, stats=["lagcorr"])` reports for the
      same columns. Now `C/(sd_a·sd_b)` in every orientation, unclamped — a
      lagged correlation is not bounded by one in finite samples — and
      `tests/test_marginal_lags.py` holds the two surfaces bit for bit
      through weights, a session change and a capped gap.
      (f) **An empty histogram takes no decay.** `label_delay`'s doubled
      stream begins with a zero-weight prefix; the pair moments carry no
      trace of it, but the histogram picked up a scale, which then moved
      every later weight and read by a rounding — the one last-bit
      difference in the doubled-stream test. An `empty` flag (true at
      construction and after a fold takes every weight to zero) makes
      `decay` a no-op, so `scale == 1` whenever it is set.
      (g) **The closed-group row carries both blocks.** `ClosedRow`'s pair
      block had none of the new columns. It now has `pair_lagcorr_*`,
      `pair_n_serial`, `pair_t_serial`, `pair_phi_*`, `pair_bin_*` and
      `pair_split_*` — `List(List(Float64))` where `marginal()` has a list
      per pair — gated on whether any closing spec asked, under
      `marginal()`'s null rule (NaN only; `±inf` stays, so `pair_t`,
      `pair_t_serial` and `pair_split_gain_t` are `±inf` where the frame
      is), which the block had not been following. `Bank::marginal` builds
      its schema from the spec, so the columns are there before any row and
      for a group never seen. The CSV sidecar refuses every nested column,
      so a closed marginal row stays parquet/ipc/ndjson-only as before.
      (h) Minor: a zero-weight row across a total gap (`λ = 0`, `w = 0`)
      returned from `learn` before touching `wt`/`qt`, so a target's `n_eff`
      outlived a gap the model's did not — it now ages them; `check_edges`
      validates without allocating; docstrings and the design doc rewritten
      to what shipped.

      Tests: `margbins.rs` 6 → 15, `marginal.rs` 17 → 21,
      `state_encoding.rs` covers every optional part; `test_marginal_bins.py`
      13 → 26 functions, `test_marginal_lags.py` 8 → 12,
      `test_closed_groups.py` +3. Every finding above has the test that
      fails without it.

- [x] 69. **The bank's read surface, 2026-09-07** (`docs/ENHANCEMENTS.md`
      E68). A state file was already self-describing and none of it was
      guarded, documented in the README, or reachable from anything but
      Python.

      *Protected.* `ModelBank.specs` was set in `__init__`, so it was an
      instance attribute; `tests/api_surface.txt` is generated by walking
      `dir(po.ModelBank)` and therefore never recorded it. A public name the
      snapshot cannot see is one the README's "a change is a reviewable
      diff" claim does not cover. It is now a read-only property returning a
      copy -- it was writable, and an in-place
      `bank.specs[0]["features"] = [...]` left `coef()` asserting about a
      coefficient count from a spec the bank was not running. A test asserts
      `dir(instance) - dir(class)` is empty, so the next one cannot escape
      the same way.

      *A JSON export*, `to_json()` / `save_json()`. An export, not a second
      state format: `load` reads msgpack and only msgpack, and no second
      format means no second obligation under rule 5.

      **The trap, and why the enforcement is what it is.** `serde_json`
      writes `NaN` and `±inf` as `null` and says nothing about it, where
      msgpack round-trips all three. That is not a corner: `halflife = inf`
      means no decay, is documented, and puts an infinity in every stream's
      `decay`, so the obvious export was silently wrong on ordinary banks.
      Three things follow:

      - A custom `serde_json` `Formatter` **cannot** fix it. `serialize_f64`
        sends a non-finite value to `write_null` before any formatter is
        consulted, so `write_f64` is unreachable for exactly the values it
        would be written for. Tried, and reverted.
      - The fix is per field: `online_core::humanfloat` tags the three as
        `"nan"` / `"inf"` / `"-inf"` -- the spelling `spec::Num` already
        used, so a spec's `halflife` and a state's `decay` read alike --
        keyed on `is_human_readable()`. **msgpack reports `false`, so the
        state file's bytes did not move**: `state_encoding.rs` pins an
        annotated field to the same bytes as a bare `f64` in both encodings,
        because if that ever became untrue every state file would stop
        loading, silently, as a string where a float belongs.
      - Missing a field is caught, not shipped: `save_json_string` re-reads
        its own output into a `BankFile`, re-encodes, and requires the
        msgpack to match byte for byte. That is `O(state)` twice, paid once
        at the end of a run, and it is what found all 18 field annotations,
        across 6 structs -- `Decay`'s two variants, `kalman`'s `halflife` and
        `revert_halflife`, `sgd`'s `clip_gradient`, `holt`'s two halflives,
        `kmeans`'s `far_cut`, and `LastRow`'s ten diagnostic vectors, where
        `NaN` is an ordinary value because an unsupervised model has no
        `pred`. The error quotes the offending lines, since the export runs
        to thousands and a line number alone names nothing.

      `serde_json` moved from a dev-dependency to a workspace dependency;
      nothing new is linked, since `online-py` already depended on it.

      Tests: `tests/test_state_json.py` (25 cases, one per model family with
      a non-finite wherever the spec allows one -- which doubles as the audit
      for the next model), four new cases in `tests/test_bank_ergonomics.py`,
      and `a_human_readable_float_does_not_move_the_msgpack` in
      `crates/online-core/tests/state_encoding.rs`. README: "A state file
      describes itself" and "Reading a state without this library".

- [x] 70. **The leak test measured the ramp, not the plateau, 2026-09-08.**
      `ubuntu-latest` failed the 0.3.0 release candidate on
      `test_output_outliving_its_input`: marks of
      [291868, 291868, 291868, 295840, 301264] KB -- identical to the byte
      three times, then 4.0 MB and 5.3 MB -- a median gap of 33.1 KB/iter
      against a limit of 4.

      Not a leak. Over twelve blocks locally the growth decays to exactly
      zero and stays there (`[1.87, 0.27, 0.13, 0.4, 0.0, 0.27, 0.0 ...]`),
      and `cargo test` passed on the same tree. What failed is the
      measurement: `warmup` was 40 iterations, and the allocator's ramp is
      not over until about 600 -- RSS is 7,344 KB up by iteration 40 and does
      not hold flat to within 8 KB per 120 iterations until 600. Every block
      the test measured was inside the ramp. macOS steps finely enough to
      stay under the threshold; glibc steps in whole arenas, and two
      consecutive steps outvote a median of three gaps, which is the one
      thing task 61's fix was built to survive.

      The repo already knew: `test_plugin_over_groups` passed `warmup=600`
      with a docstring measuring the same ~600-iteration ramp. It was a
      workaround that was right there and silently wrong for the other nine
      callers.

      `assert_plateaus` now **finds** the plateau instead of assuming one --
      it runs blocks until two consecutive ones grow by less than the
      threshold, then measures. That is self-calibrating, so it needs no
      constant tuned per platform, and it cannot be fooled by a leak: a leak
      never plateaus, exhausts `warm_blocks`, and is measured anyway. The
      per-test override is gone.

      Checked both ways rather than just green: three consecutive runs of the
      file pass (6.7s to ~9s), and an injected 8 KB/iter leak is still caught
      (`8.6 KB/iter, gaps [8.7, 8.5, 8.7, 8.5]`).

- [x] 73. **The clock does not have to be a time, and the headline docs did
      not say so, 2026-09-08.** The user's point, and it was right: the clock
      is documented as "a monotone numeric column — seconds, cumulative
      volume, anything", which states the mechanism and leaves the reader to
      derive the consequence. The consequence is worth a headline. Sort a
      frame by one of its features, clock on that feature, and `halflife` is
      a bandwidth in the feature's units — each row fit on the rows before it
      under weight `0.5 ** (Δx / halflife)`, which is local linear regression
      with a one-sided exponential kernel, in one pass, out of `O(k²)` of
      state.

      *Measured before it was written.* An independent oracle — the
      kernel-weighted least squares that definition describes, recomputed
      from scratch at every row, with the mean-form ridge and unpenalised
      intercept the model solves with — agrees with the model to `1.6e-13`
      at three (halflife, ridge) settings, and the null patterns match, so
      `min_periods` gates the same rows. On `sin(x)` at a bandwidth of 0.25
      the fit sits `0.077` from the truth against the best straight line's
      `0.394`; at 1.0 it is `0.290`, most of the way back to the line. Both
      are in `tests/test_ewridge.py`
      (`test_a_feature_as_the_clock_is_a_local_linear_regression`,
      `test_a_bandwidth_in_the_clock_column_follows_a_curve_a_line_cannot`).
      `ew_cov` clocked the same way tracks a correlation that moves with the
      clock column at `corr = 0.91` to the truth — reported here, not
      claimed with a number in the README.

      *What the docs say, and what they admit.* The README's opening list
      gains a paragraph and "How a bank sees a stream" gains "A clock that is
      not time" with a runnable block (so the block test runs it), the
      varying-coefficient reading, one local fit per `group`, and the caveat
      that matters: the kernel is one-sided *because* a row is scored before
      it is learned from, so a curve is followed with a lag, and the
      bandwidth trades that lag against noise. `polars_online.spec`'s
      docstring and `llms.txt` gain a line each.

- [x] 74. **`sgd`'s scaler standardises a row against moments that exclude
      it, and is wrong wherever there are few rows per feature: the start
      of every stream, and every row of a wide fit — found 2026-09-08 by
      task 72's short-history table, and again by its wide table the same
      evening. Fixed 2026-09-08; the decision and the numbers are at the
      end of this entry.** `scale_features` (ENHANCEMENTS E24)
      standardises `x_t` against the running mean and variance from *before*
      row `t`, "so the scaling cannot see the row it is scaling". That is
      stricter than the leakage rule requires and it is unstable: the rule
      forbids the *target*, and the features of the row being predicted are
      known at prediction time; while the moments are two or five rows old,
      a variance estimate can be tiny by chance, the standardised value
      huge, and one LMS step with `eta · |z|² > 2` throws a coefficient far
      enough that the next hundred rows do not bring it back.

      Measured: 500 groups × 200 rows, `k = 20`, each group its own
      coefficients, `learning_rate=0.01`, R² over rows 25–50 of a group is
      **−6.9** (`−99` at 0.03), where `SGDRegressor` with the same step and
      sklearn's scaler order scores 0.45 and `ewridge` 0.97. A pure-numpy
      LMS switched between the two orders reproduces both: moments from
      before the row `−64,309`, moments including the row `0.4328` — and the
      latter is within 0.02 of sklearn's 0.4501 at the same `eta` (the
      rest is where the *prediction* is standardised; see the closing
      paragraph). Over 100,000 rows scored from row 1,000 the two
      orders give the same number (0.9899 both), which is why every earlier
      test and table missed it.

      *The wide table, re-run at a stable learning rate (`0.2 / k`), shows
      the same defect from the other side.* Against `SGDRegressor` row by
      row at the same step, `scale_features=False` predictions correlate
      0.999999 at `k = 20`, 0.999922 at `k = 1,000` and 0.9954 at
      `k = 10,000`; with `scale_features=True` the same three read 0.9999,
      0.978 and **0.521**, and the R² at `k = 10,000` is 0.0209 against
      sklearn's 0.0322. Nothing diverges there — the rate is small enough —
      the standardisation is simply against moments that are never mature,
      because 2,000 rows over 10,000 features is 0.2 rows per feature on
      every row. "Short history" was the special case; **few rows per
      feature** is the condition, and the fix is the same two lines. The
      test list gains the wide case: `scale_features=True` at `k = 1,000`
      within 0.001 R² of `scale_features=False` on the wide stream (today
      0.8221 against 0.8516).

      The fix is sklearn's order — `scaler.partial_fit(x)` then
      `transform(x)` — which bounds a standardised value by about `√n`:
      in `SgdModel::step`, call `sc.update(&raw_z, lam, weight)` *before*
      standardising, not after. `n_eff` is untouched (the scaler does not
      feed it), chunk invariance holds (the order is per row), and the state
      at the end of any row is identical to today's (both orders leave the
      scaler updated with the row), so no `SCHEMA_VERSION` bump — but every
      `scale_features=True` prediction changes, so this is a **behaviour
      change**: golden fixtures regenerate, E24's row and the `sgd`
      docstring say the new order and why, the CHANGELOG entry names it,
      and the README's "Against scikit-learn" and PERFORMANCE §19 drop
      their "until it lands" sentences with the re-measured row. A
      zero-weight row still learns nothing: `update(raw, lam, 0)` applies
      the decay only. Tests: the short-history table as a test (`sgd` at
      `learning_rate=0.01` above 0.4 over rows 25–50 of 200-row groups, and
      the numpy replica agreeing with the model to 1e-12 when the row is
      included); the E24 demonstration (`[0.002, 900]` recovered) kept.

      **Done 2026-09-08, on the user's go — with one change to the recipe
      above.** "Call `sc.update` before standardising" would have broken a
      contract the model already has: `predict(x, d)` is `&self`, carries
      no weight, and must return exactly the `pred` the following `step`
      reports (E31, `model_contract.rs::sgd_predict_is_the_step`, run with
      the scaler on and off over weights 0, 0.5, 1 and 2). Updating the
      scaler with the row's weight before standardising makes the
      standardised row — and so the prediction — depend on the weight,
      which `predict` does not have. The order landed instead is: the row is
      standardised against the moments **with the row admitted at unit
      weight** after the decay — `EwDiag::including(lam)`, a view that reads
      `update`'s recursion slot by slot without touching the accumulator —
      the same `z` serves the prediction and the gradient, and the real
      `update(raw, lam, weight)` stays *after* the gradient step, at the
      row's actual weight. Properties, each with a test: at `w = 1` the
      moments read are to the bit what `update` leaves behind (the
      recursion is written operation for operation, `ewdiag.rs::
      including_reads_what_a_unit_row_would_leave`), so on unit-weight
      streams this *is* sklearn's `partial_fit` then `transform`;
      `predict`/`step` parity is exact by construction (E31 passes
      unchanged); a standardised value is bounded by `sqrt(lam · n_eff)`
      (`z = a·d / sqrt(a·c + a·b·d²) ≤ sqrt(a/b)`, with equality when the
      history has no spread), so the first row of a stream is `z = 0` on
      every feature and only the intercept learns from it
      (`sgd.rs::scaling_admits_the_row_it_scales`, which also checks that
      a row a million deviations out predicts within
      `|b0| + |b1|·sqrt(W)`, where the old order predicted about 1.7e6);
      always well defined, since the total weight is at least the row's own
      1 even on an empty accumulator; and the weight governs *what the row
      teaches*, not where it sits among the rows seen. `kalman` also
      standardises against the moments from before the row and is left
      alone: its gain `P z / (z'P z + σ²)` self-normalises in `|z|` the way
      `pa`'s step does, so a huge `z` moves the state by a bounded amount —
      the defect is specific to a fixed learning rate.

      What did not change: `n_eff`, chunk invariance, the state at the end
      of a row (so no `SCHEMA_VERSION` bump), the cost (`sgd_bench` at
      `k = 10,000`: 20.7 µs a row before, 21.3 after, within the run-to-run
      spread), and a zero-weight row still learns nothing. What did: every
      `scale_features=True` prediction, so `GOLDEN_SGD` regenerated
      (`sgd_squared_golden` and the pipeline goldens run unscaled and stood).

      Measured, `scripts/sklearn_comparison.py`, the day it landed. Short
      history (500 groups × 200 rows, `k = 20`): `sgd` at
      `learning_rate=0.01` reads **0.4328 / 0.7006 / 0.9067** over rows
      25–50 / 50–100 / 100–200 (was −6.9 / −2.4 / 0.1095), against
      `SGDRegressor` with the same constant rate at 0.4501 / 0.7119 /
      0.9111 and `ewridge` at 0.9693 / 0.9860 / 0.9882; at 0.03, **0.7182 /
      0.9213 / 0.9788** against sklearn's best 0.7277 / 0.9242 / 0.9789.
      The numpy replica's 0.4328 is the model's number to four places, and
      switching the replica to predict against the moments from *before*
      the row while learning against the ones including it — sklearn's
      loop, two standardisations a row — gives sklearn's 0.4501 / 0.7119 /
      0.9111 to four places, so that is the whole of the remaining gap.
      Kept as one `z`: a row is then standardised the way every row the
      coefficients were learned from was, the prediction is bounded like
      the step, the scaler is one pass, and the gap is an artefact of an
      unconverged fit (included moments shrink a row's `z` by about `1/n`,
      and a fit still growing into its coefficients scores higher inflated
      by that much) that closes with the history: 0.0044 by rows 100–200
      at 0.01, 0.0001 at 0.03. Wide:
      `scale_features=True` at `k = 1,000` is R² 0.8512 against
      `scale_features=False`'s 0.8516 (within the 0.001 asked for; was
      0.8221), and its predictions correlate **0.99999995** with
      `SGDRegressor`'s row by row (was 0.978); at `k = 10,000` R² 0.0321
      against sklearn's 0.0322 and a correlation of **0.999997** (was
      0.521). Tests added in `tests/test_sgd.py::TestFeatureScaling`: the
      short table (R² > 0.4 over rows 25–50, > 0.85 over 100–200), the
      numpy replica to 1e-12 on every prediction of three groups, and the
      wide case at `k = 200` (scaled within 0.01 R² of unscaled, predictions
      correlating > 0.999); the E24 demonstration (`[0.002, 900]`
      recovered) kept.

- [x] 72. **Against `sklearn.linear_model.SGDRegressor` — analysis done
      2026-09-08, measured and written up the same day.** Written because
      "how does this compare to sklearn" is the first question a reader has
      and the README answered it nowhere.

      **The comparison is not the one the name suggests.** `SGDRegressor`'s
      counterpart here is `po.spec.sgd`, which this repo calls the cheap
      baseline. The primary regression is a different algorithm class:
      `SGDRegressor` is a first-order stochastic optimiser whose answer
      depends on the learning rate, the schedule, feature scaling and row
      order, while `ewridge`/`rls`/`lasso`/`huber`/`quantile` accumulate
      sufficient statistics and solve — with decay off and `ridge = 0`,
      `ewridge` is OLS to `2e-13` of `numpy.linalg.lstsq` in any row order,
      and there is no learning rate because nothing is being descended.

      Where the designs diverge, in the order that matters to a reader:
      forgetting (`partial_fit` weights all history equally and its
      learning-rate decay is about convergence, not recency; every spec here
      takes a `halflife` on a real clock); leakage (sklearn leaves the
      predict-then-fit ordering to the caller, here it is a hard rule with
      tests); grids (`ridge=[...]`, `halflife=[...]` share one accumulator,
      so only the solve repeats, against N estimators over N passes);
      multiple targets sharing one Gram; `group=` against a dict of
      estimators; streaming standardisation against a `StandardScaler` that
      leaks when fitted on a batch; chunk invariance; and a versioned
      cross-OS state file against pickle.

      Where sklearn wins, and the README should say so: `SGDRegressor` is
      `O(k)` in memory where `ewridge` carries a `k×k` Gram — 800 MB at
      k = 10,000, which is the whole reason `marginal` and E51 exist — plus
      the ecosystem (pipelines, `GridSearchCV`, calibration) and far more
      use. `sgd` here is the `O(k)` answer.

      Loss coverage, for the record: `sgd` adds `quantile`, `poisson` and
      `logistic` over `SGDRegressor`'s four, and lacks
      `squared_epsilon_insensitive`; `l1`/elastic net lives in `lasso`,
      solved exactly by coordinate descent rather than by a subgradient.

      **Measured, on one generated stream** (`scripts/sklearn_comparison.py`,
      scikit-learn 1.9.0, which is deliberately *not* a dependency of this
      project — the script says so and exits if it is missing).
      `docs/PERFORMANCE.md` §19 has the four tables; the README gains an
      "Against scikit-learn" section with one of them. Every contender is
      swept over a small grid of its own settings and reported at its best,
      so the comparison is between designs and not between defaults, and
      sklearn is given the setup its own documentation prescribes for
      out-of-core work — an online `StandardScaler`, then `partial_fit`.

      What the numbers said, including where they went against this library:

      - *Accuracy is not the difference.* On a stationary stream at
        `k = 20` everything lands on the same noise ceiling (0.9826 to
        0.9831 out-of-sample R²). The gap opens only under drift, and it is
        about forgetting: `sgd` 0.9906 and `ewridge` at `halflife=500`
        0.9878, against `SGDRegressor`'s 0.9820 batched and 0.9840 row by
        row. A halflife on a clock beats a learning rate tuned in rows —
        which is the claim this repo has been making, now with a number.
      - *The throughput gap is a difference in semantics, and saying so
        matters more than the ratio.* sklearn's fast form is one update per
        1,000-row batch, whose predictions are up to 999 rows stale;
        2.2M rows/s. Ours is one update per row: 5.8M for `sgd`, 570k for
        `ewridge` refitting every row, 5.0M at a 100-row solve cadence. Ask
        sklearn for our guarantee — `partial_fit` per row — and it is 3,300
        rows/s, and that gap is Python's per-row overhead, not the
        algorithm. The README says that in those words rather than quoting
        1,700×.
      - *A grid is nearly free on one side only*: six penalties cost sklearn
        3.66× the work and this 1.40×.
      - *Where sklearn wins, measured rather than conceded in prose.* At
        `k = 10,000`, `ewridge` carries 860 MB of state (`save_bytes`) and
        runs at 51 rows/s against 0.31 MB (a pickle of the fitted estimator
        and its scaler) and 17,221 rows/s. `gram_block_rows=256` buys 5.2×
        of the throughput and none of the memory. `sgd` is the `O(k)`
        answer here — 1.06 MB, 6,206 rows/s — and is *still* slower than
        sklearn at that width, for the same per-row reason. The README says
        that too. (Slower than *batched* sklearn: row by row it is 3×
        faster, and the 0.01 learning rate this row was timed at had both
        libraries diverged — see the evening's addendum below.)

      Two measurement mistakes worth recording, both caught by the numbers
      looking wrong rather than by review. Peak RSS is useless as a memory
      proxy here — the frame and the numpy array dominate it — so the state
      comparison is `save_bytes()` against `pickle.dumps`, which is what a
      deployment actually keeps; and the first wide run had `ewridge`
      solving every row at `k = 1,000` (`solve_every=0.0` with
      `halflife=inf`), which measured a `O(k³)` solve per row at 206 rows/s
      and would have been a strawman of this library, not of sklearn.

      *Corrected the same afternoon, after the user asked whether the
      comparison could improve `ewridge` or whether the feature set buys
      anything.* Re-measured with wider grids, the noise ceiling printed, and
      two new streams; three of the bullets above were wrong in their
      reading and one number was a grid edge:

      - `ewridge`'s halflife sweep stopped at 500 rows, and its best point
        was that edge; at `halflife=100` it scores 0.9907 on the drifting
        stream and ties `sgd` (0.9906), 0.002 from the ceiling of 0.9923.
        **A sweep whose best point is at its edge is a grid, not a result**;
        the script now prints every sweep, best to worst, and the ceiling.
      - "A halflife on a clock beats a learning rate tuned in rows" was the
        wrong reading of the drift table. A constant learning rate *is*
        forgetting (an LMS step `eta` remembers about `1/eta` rows), and on
        evenly spaced rows that is a halflife. `SGDRegressor` row by row at
        `constant, eta0=0.003` scores 0.9899 and `sgd` at
        `learning_rate=0.003` scores 0.9899 — the same recursion, the same
        number. The 0.9840 quoted for sklearn row by row was its
        `invscaling` default, whose decay is about convergence and not
        recency. What separates the drift column is the batched 0.9820,
        and that is staleness. The clock's advantage exists on unevenly
        spaced rows, which this stream cannot show and the write-up
        claimed anyway.
      - Correlated features do not separate the contenders either: a
        one-factor stream at pairwise correlation 0.5 and 0.9 (condition
        number 22 and 187) leaves every row-by-row contender within 0.0007
        of its ceiling over 100,000 rows. First-order convergence slows
        with conditioning, but 100,000 rows is long enough not to notice.
        Heavy-tailed features (`t(3)`, `max |x| = 176`) cost `sgd` its
        best learning rate and nothing at all to `ewridge`; the best
        settings still land within 0.0005 of each other.
      - **Where the Gram measurably wins is a short history.** 500 groups
        of 200 rows, each with its own coefficients: at rows 25–50 of a
        group `ewridge` scores 0.9693 against a ceiling of 0.9896, the best
        `SGDRegressor` setting 0.7277, its default 0.2731; by rows 50–100
        it is 0.9860 against 0.9242. The exact solve is right about `k`
        rows in; a first-order method needs `1/eta` rows per direction. The
        rows/sec column is the group feature: 5.1M for one bank call
        against 3,350 for a Python loop over 500 estimators.
      - **And the same table found `sgd` losing to sklearn's, badly**: R²
        of −6.9 at rows 25–50 at `learning_rate=0.01`, −99 at 0.03. That
        is task 74, a real defect the 100,000-row tables hid by scoring
        from row 1,000.

      What the comparison gives back, then: for `ewridge`, nothing to
      borrow — sklearn's recipe is a scaler and a schedule, `ewridge` has
      the scaler folded into its solve (`standardize`) and no schedule
      because nothing is descended, and its limit is the Gram itself (stored
      full in `EwCov::c`; a triangle would halve 860 MB to 430 and not
      change the wide-row verdict, so it is not done). For `sgd`, the order
      of two lines (task 74), and `average=True` — sklearn's best stationary
      setting, 0.9830 against `sgd`'s 0.9826, and its worst under drift at
      0.817 because the average never forgets — would fit this library as a
      halflife-weighted average of the iterates, if that 0.0005 ever
      matters. `docs/PERFORMANCE.md` §19 is rewritten from the second run;
      the README table carries the corrected numbers and the short-history
      table; the CHANGELOG records the known defect.

      *Two more questions the same evening, both measured at `k = 10,000`.*
      "Is it so much slower because the test emits coefficients every
      row?" No: `coef_every` defaults to 0, so the wide table emitted none,
      and turning it on (10,000 coefficients per row, the output frame
      153 → 302 MB) costs 2.5%; one solve instead of two costs 0.2%;
      `po.spec.sgd` through the same bank with the same 10,005-column
      output runs at 6,217 rows/s against `ewridge`'s 53. The cost is the
      Gram: a rank-1 update moves all 800 MB of it per row, 1.6 GB in
      18.8 ms = 85 GB/s, within 1.5× of a plain in-place numpy add over the
      same 800 MB (123 GB/s on the M4 Pro). Bandwidth, not arithmetic and
      not output. `gram_block_rows` is the lever that exists: 256 → 269
      rows/s, 512 → 306, 1024 → **379** (7.2×, an 82 MB buffer), 2048 →
      376. "Does our `sgd` produce the same thing as `SGDRegressor` at
      6,000 rows/s?" Yes, and 6,000 is the faster side: `SGDRegressor` at
      this library's semantics (row by row) ran at 2,267 rows/s at
      `k = 10,000` and 3,043 at `k = 1,000`, against `sgd`'s 6,944 and
      185,641 (33,271 and 407,925 since task 75), with the correlations
      above; sklearn's 18,980 is its batch of
      1,000, predictions up to 999 rows stale, the narrow table's trade at
      2.7× instead of 1,700×. Checking that exposed that the wide table's
      `learning_rate=0.01` had both libraries diverged (R² −6.8e8 and
      −5.8e26; the timings are unaffected, a diverged LMS costs what a
      converged one does): an LMS step is stable only while
      `eta · |z|² < 2` and a standardised row has `|z|² ≈ k`, so the script
      now runs both at `0.2 / k` and prints R² in the wide table. It also
      put a number on `sgd`'s inner loop — a flat 13–14 ns per feature per
      row from `k = 1,000` to `k = 10,000`, against about 5 ns for sklearn's
      batched Cython — and misread it: "the columnar gather is not it",
      because `to_numpy` of the whole 10,000-column frame ran at 8.4 µs/row
      against `sgd`'s 142. That compared one transpose of the frame with
      a bank that gathered every row *again* from `k` separate columns,
      at a stride; the walk was most of the 142, and one transpose per
      chunk is what the bank does now. Task 75 took the number apart the
      same day: 2.2 ns per feature, 45,000 rows/second at `k = 10,000`,
      2.4× sklearn's batch at the row-by-row semantics. And at
      `k = 1,000` the R² column repeats the short-history lesson: 20 rows
      per feature, `ewridge` at 0.9274 where every first-order contender is
      at 0.85, predicting from coefficients up to 1,000 rows old.

- [x] 75. **`sgd`'s per-feature cost, 2026-09-08: 13–14 ns per feature per
      row to 2.2 at `k = 10,000`, 45,000 rows/second in a bank against
      `SGDRegressor`'s batched 20,007, and every prediction from the state
      as it stands.** Task 72 left the number as "a real target, and an
      independent one"; this is it taken apart, and none of it was the
      arithmetic (`docs/PERFORMANCE.md` §20 has every measurement).

      *The columns.* The bank kept one `Vec<f64>` per feature and a row
      was a walk across all of them at a stride of the chunk's height:
      a cache line and a TLB entry per feature per row, 10,000 pages at
      `k = 10,000`, 78–130 µs per row depending on the stride — the
      "erratic with chunk size" the earlier tables showed. Each chunk is
      now gathered once into a row-major buffer
      (`crates/online-polars/src/rows.rs`, `FeatureRows`): a tiled
      transpose, 128 rows by 64 columns so source and destination lines
      both stay in L1, the layout permutation folded into the gather, in
      parallel above 65,536 cells (`PAR_MIN_CELLS`, alongside the row
      threshold). 3 µs per row at that width. The summary, the label-delay
      buffer and the models all read `features.row(i)`; `Scratch` lost its
      `xs` copy.

      *The step.* Indexed `beta[j][i]`, `zbuf[i]`, `g2[j][i]` through two
      levels of `Vec` per feature, a bounds check on each, the schedule
      chosen by a `match` inside the loop, `(1 + n)^power` raised per
      feature, and three vectors allocated per row on the scaler path.
      Now zipped slices, the rate raised once per row, the intercept's
      constant 1 folded into the dot product as its coefficient (so the
      unscaled path copies nothing), persistent buffers. 2.25 → 0.93 ns per
      feature; `scale_features=True` 14.8 → 2.6. Bit-identical, and proven
      so rather than argued: `crates/online-core/examples/sgd_signature.rs`
      hashes every `step` and `predict` over 48 configurations (intercept ×
      scaler × three schedules × two losses × two penalties, 400 rows with
      zero weights, clock gaps and a missing target) and read the same on
      the old code and the new. `crates/online-core/examples/sgd_bench.rs`
      is the core step's clock.

      *The dot product — the one decision.* A single running sum is a
      chain of dependent additions, three or four cycles each whatever the
      core could do alongside, and 10,000 of them were 7 of the 9.3 µs the
      step still took. It is now `DOT_LANES = 8` interleaved partial sums
      folded pairwise, then the intercept: 0.93 → 0.49 ns per feature.
      The order is fixed by the code, the same on every platform, and not
      0.3.1's, so `sgd` predictions differ from the last release's at
      rounding level: 1e-16 relative on squared loss, 1.5e-11 at worst in
      the signature — Huber at a constant rate, whose clipped gradient
      neither damps a perturbation nor amplifies it, over 400 rows of zero
      weights and gaps. The pipeline goldens (1e-12) pass unchanged, and
      their own docstring says what the tolerance is for: reordered
      arithmetic. `predict` standardises into a thread-local buffer under a
      scaler, since it has `&self` and a fresh vector per row was the cost
      being removed. Taken because the gain is a third of the step and
      neither order is more right than the other; it is its own commit, so
      it can be reverted alone.

      *The accept walk.* `Iterator::all` over the row short-circuits, which
      keeps it scalar; `all_usable` folds over the compare and vectorises.
      1–2 µs per row at `k = 10,000`.

      *Measured, before → after*, one bank, one spec, `learning_rate =
      0.2 / k`, `scale_features=False`: `k = 20` 0.11 → 0.09 µs per row,
      `k = 100` 0.45 → 0.26, `k = 1,000` 4.60 → 2.06, `k = 10,000` 76.5 →
      22.2 (20,000 rows) and 137.9 → 30.4 (2,000 rows); the same stream fed
      as eight chunks at `k = 10,000`, 127.1 → 26.4. `predict` 49.5 → 9.4.

      *What is left at `k = 10,000`, 20 µs per row.* The step 4.9, the
      transpose 3, the accept walk 0.4, and the data summary (task 35)
      about 7 — a Welford update of a 48-byte record per feature per row,
      measured by switching it off. It is now the largest item; making it
      cheaper means laying the records out column-wise so the update
      vectorises, which is a state-layout change (a `SCHEMA_VERSION` bump,
      or a serde mirror of the wire form) for perhaps 4 of the 20 µs. Not
      done. The cheap version — comparing before storing `min` and `max` —
      measured no change and was dropped. The frame hand-off's per-call
      constant is about 8 ms at 10,000 columns, 0.8 µs per column in
      pyo3-polars' `PyDataFrame` extraction (`get_columns`, then one
      `_export` per Series), none of it this library's: it is what
      separates 2,000 rows per call (4 µs per row of it) from 20,000
      (0.4), and the "eight chunks" figure at 2,000 rows from the rest.
      Feed wide frames in tall chunks.

      *Corrected.* PERFORMANCE §19, the task-72 addendum below and the
      README's "Against scikit-learn" all said `sgd`'s inner loop was 13–14
      ns and sklearn's batch faster on a wide row; §19's wide table is
      re-run and the three now say what is measured.

- [x] 76. **`hit_rate` reads 1.0 for every binary fit, and classification has
      no metric of its own — found 2026-09-08, asking whether an SGD
      classifier was worth adding. Fixed the same day; the decision and the
      numbers are at the end of this entry.** A hit is scored as
      `pred.signum() == y.signum()` with `y == 0` rows dropped
      (`SlotMetrics::update` in `crates/online-core/src/stats.rs`, and the
      same expression in `python/polars_online/eval.py`). A logistic fit
      predicts a probability in (0, 1) against a 0/1 target, so every zero
      row is dropped and every surviving row is a hit: the field is **1.0
      whatever the fit does**. Measured, 20,000 rows, `k = 2`, `y` drawn from
      `sigmoid(1.2·x0 − 0.8·x1)`:

    | fit | `hit_rate` | `r2` | `ic` | accuracy at 0.5 |
    |---|---:|---:|---:|---:|
    | `sgd`, `loss="logistic"` | 1.0000 | 0.2719 | 0.5216 | 0.7268 |
    | `ftrl`, `loss="logistic"` | 1.0000 | 0.2784 | 0.5278 | 0.7304 |
    | `sgd` logistic on pure-noise features | **1.0000** | −0.0159 | 0.0189 | — |

      A model that knows nothing reports a perfect hit rate, in the streaming
      metric and in `po.eval.metrics` alike. The other two survive the
      translation and nothing says so: on a 0/1 target `r2` is the Brier
      skill score against the running base rate and `ic` is a point-biserial
      correlation. There is no log loss anywhere, and `ew_class` refuses
      `emit_metrics` outright ("it has no predictions, so no residuals"), so
      the multiclass model has no metric at all.

      The fix, in three parts:

      1. *`hit_rate` on a binary fit is accuracy at a 0.5 threshold.* Gate it
         on the **declared** loss — `sgd`'s and `ftrl`'s `logistic` — not on
         sniffing the target's values, which would make a metric's meaning
         depend on the data a chunk happened to carry. A regression fit keeps
         the sign agreement it has now. Alternative if a threshold is judged
         arbitrary: refuse the field for a logistic loss the way `ew_class`
         refuses the lot, which is honest but leaves classification with two
         metrics rather than three.
      2. *Log loss, and the constraint that shapes it.* `−[y ln p + (1−y)
         ln(1−p)]`, exponentially weighted like the rest. The streaming
         version would put a `ln` result **into the state**, which §11a's B4
         rule forbids: a libm result may be compared or reported, but
         persisting one costs cross-platform reproducibility (that is what
         broke a frozen fixture on `ubuntu-latest` alone). So either it lives
         only in `po.eval`, over the collected frame, or the accumulator is
         excluded from the frozen-state fixtures with the reason written
         down. Decide before building.
      3. *Document what the metrics mean on a 0/1 target* — `r2` as Brier
         skill, `ic` as point-biserial — in `emit_metrics`' docstring, the
         README's output table and `po.eval.metrics`.

      **Not the classifier E28 declined.** No new loss, no new model, no
      margin: this is the plumbing that makes the two logistic fits already
      shipped (`sgd(loss="logistic")`, `ftrl`) reportable. E28 (perceptron
      and PA-classifier hinges) and E39's discriminative multiclass stay
      declined until a caller wants a margin instead of a probability.

      Ships after 0.4.0, which is tagged with the defect in it. Whether
      `hit_rate` changing counts as "a change to the numbers a model
      returns" under 0.4.0's own widened rule is the user's call; the reading
      here is that it is a diagnostic rather than a prediction, and one that
      is currently a constant, so a patch — but it should be the first line
      of the release note either way.

      Tests: the pure-noise fit above as a regression test (a model that
      knows nothing must not report a perfect hit rate); accuracy against a
      numpy replica on a fit that does know something; the regression path's
      `hit_rate` unchanged to the bit; and `po.eval.metrics` agreeing with
      the streaming metric on the same stream, which is the invariant E22
      already claims.

      **Done 2026-09-08, on the user's go.** Built as planned, with the two
      questions the entry above left open resolved:

      - *A threshold, not a refusal.* `SlotMetrics::update` (and
        `_metric_exprs` in `eval.py`) take a `binary` flag: sign agreement
        with `y == 0` excluded when `false`, accuracy at 0.5 with every row
        scoring when `true`. It is set from the model's **declared** loss —
        `sgd`'s or `ftrl`'s `logistic` — computed once per instance in
        `run_instance` (`crates/online-polars/src/stream.rs`), not sniffed
        from a row's value; `po.eval`'s functions take `binary` as an
        explicit keyword, the same fact the caller already knows from
        choosing the loss. No new spec field: a `logistic` fit's `hit_rate`
        simply means something different now, the way task 74 changed what
        `scale_features` computes without adding a knob for it.
      - *Log loss lives only in `po.eval`.* `metrics(..., binary=True)` and
        `rolling_metrics` add a `log_loss` column
        (`-(y·ln p + (1−y)·ln(1−p))`, clipped, mean over the window);
        `sums`/`from_sums` (E49) do not carry it — a straightforward
        addition if a chunked reduction ever needs it, not built because
        nothing has asked. Confirms the B4 call in the entry above: nothing
        here puts a `ln` result into a model's persisted state, so no
        cross-platform risk is taken on.

      `merge_sums` needed no change — it sums whatever `hits`/`signed` hold
      without caring how they were computed, so mixing `binary` settings
      across chunks is a caller error the docstring now names, not a new
      failure mode to guard against. No `SCHEMA_VERSION` bump: `SlotMetrics`'s
      fields are unchanged in shape, only in what a caller's own choice of
      loss makes them count — the same kind of change task 74 made to `sgd`.

      Measured, the pure-noise fixture from the top of this entry (20,000
      rows, `k = 2`, `sgd(loss="logistic")`): `hit_rate` (binary) 0.5042,
      where it read 1.0 before. On the informative fixture: `hit_rate`
      (binary) 0.72677, matching a numpy replica of the same threshold test
      to 1e-12, and `po.eval.metrics(binary=True)` agreeing with the
      streaming `hit_rate_y0` field to 0.72675 (one row's difference in
      20,000 — `emit_metrics` reads before the last row, `po.eval` scores it
      too). `sums`/`from_sums` with `binary=True` reproduce `metrics`'s
      `hit_rate`, `r2` and `ic` (0.72677, 0.27193, 0.52167) exactly.

      Tests: `crates/online-core/src/stats.rs` —
      `a_probability_that_knows_nothing_does_not_score_a_perfect_hit_rate`,
      `binary_hit_rate_is_accuracy_at_one_half` (against a hand-rolled
      replica), `a_zero_label_scores_under_binary_where_it_is_excluded_under_sign`;
      every existing `SlotMetrics` test kept its `binary = false` call
      unchanged, so the regression path is untouched to the bit.
      `tests/test_eval.py::TestBinary` covers the same four cases in Python,
      plus `log_loss` against a numpy replica and the `sums`/`from_sums`
      round trip. Every test file that already used `emit_metrics` or
      `hit_rate` (`test_diagnostics.py`'s E22 test among them) is regression
      and passes unchanged, since none of them named `binary=True`.

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

- [x] 85. **The expression plugin removed, 2026-09-17.** It was O(data) by
      polars' own rules and warned on every use since 2026-09-03 (task 19);
      keeping a surface whose whole behaviour contradicts the library's point
      cost more than it bought. Deleted: `crates/online-py/src/expr.rs`,
      `python/polars_online/_expr.py`, `tests/test_expr.py`, the `online`
      namespace, `po.online` and `InMemoryExpressionWarning`. **The build
      consequence is the real prize and was measured, not assumed**:
      `pyo3-polars/derive` turned on `polars-plan/python`, which only
      `polars-lazy/python` propagates to `polars-mem-engine`, so `derive` was
      the reason `lazy` was in the extension's features at all. Both are gone
      and `cargo check -p online-py` is clean. ~30 expression tests went too:
      both paths call the same `Bank::fit_predict`, so they tested packing, not
      arithmetic (survey in the session log: every model file keeps 3-60 bank
      tests, five keep numpy oracles). Four tests were *restored* after a bulk
      deletion over-reached -- two `test_every_surface_runs_it`, the
      two-specs-stay-distinct hardening case, and two typed-dict runtime cases
      -- because their subject survives the plugin.
- [x] 84. **The bank takes a plan, and learns without keeping the output,
      2026-09-17.** `fit_predict_batches` accepts a `LazyFrame` and builds the
      chunk iterator itself (`chunk_rows`, defaulting to the shared
      `default_chunk_rows`), so the caller no longer writes
      `lf.collect_batches()`; a `DataFrame` is one chunk and an iterator is
      passed through. `fit` is the learn-only form -- the same generator with
      its frames dropped -- which gives Python the runner's `--no-output` shape
      now that the runner is the command line's alone (task 83). Measured
      claim deliberately *not* made: this is not faster. The runner's own
      discard path drops each frame after assembly (`runner.rs:618`), and so
      does this; what it saves is the write and the assembled result.
      `_check_frame`'s advice changed with it, since telling a caller to reach
      for `collect_batches` is wrong once the method takes the plan.
- [x] 83. **The runner leaves Python; the `online` command line keeps it,
      2026-09-17.** `polars_online.run` built its chunk iterator with py-polars
      and handed the frames to `run_config_on`, so it duplicated what
      `ModelBank` and `lf.online.fit_predict` already do while dragging the
      lazy engine, the parquet/CSV/IPC readers and the sinks into the
      extension module, which no Python path reached (`Input::Lazy` has one
      constructor, `runner.rs:430`, and only the CLI reaches it). Removed:
      `_runner.py`, `run_config_frames`, `PyFrames`, `PyFailure`, `formats()`.
      Tests moved rather than dropped wherever the path survives -- the
      hardening trio and the no-output and predict clusters to the command
      line, the sidecar cases to the query path, `test_sink_equals_run` to a
      sink-versus-collect comparison -- and deleted only where the bank half
      sat in the same function. `progress` has no command-line equivalent and
      is removed, not moved.
- [x] 82. **N9: `quantile` did not settle on the quantile regression -- found
      2026-09-15 writing the code review's T-S4, fixed the same day on the
      user's go-ahead.** IRLS on each row's prior residual, with its weights
      frozen as the rows arrived, kept the fit off `statsmodels`' `QuantReg`
      at every length measured: 0.164 in the intercept at the median of a
      skewed noise after 20 000 rows and 0.477 at the 0.9 quantile, where
      `QuantReg`'s own standard errors are 0.007 and 0.021. It takes one
      Newton step on the kernel-smoothed check loss now (`robust.rs`,
      `row_update`), which lands inside those standard errors, and
      `quantile_eps` is the band's half-width, default 0.2. The measurements
      and the derivation are in `docs/REVIEW-2026-09-12-PROGRESS.md`, N9.

- [x] 81. **`target_gaps`: which rows a target's fit is read from -- the code
      review's N3, decided 2026-09-14.** With a target null on some rows,
      `ewridge` and `lasso` read the Gram over every row and the target's
      cross-moments over its own rows, so a slope moved with the target's
      level: `(m_j − m)·ȳ_j / Var(x)` on top of the fit, with `m_j` a
      feature's mean over the target's rows and `m` its mean over all of
      them. The user asked for every option behind one parameter, then, of
      that behaviour: "what value does it add if we implement the other
      options?" None: it costs what `"pairwise"` costs and adds the level
      term, so it goes. Two values, on `ewridge` and `lasso` -- `robust` and
      `kalman` already fit each target on its own rows, and `rls` learns a
      row only when every target has one:

      - `"own_rows"`, the default: each target is fitted on exactly the rows
        it is present on, the fit of the frame with its nulls dropped.
        Targets present on the same rows share one Gram. A target missing
        from a row that another target of its Gram is present on takes a
        copy of that Gram as it stood before the row, and keeps its own from
        then on. A single target never copies (its Gram skips its null
        rows), and a bank of targets costs one `k×k` Gram per pattern of
        missing rows.
      - `"pairwise"`: the Gram over every row, each target's cross-moments
        centred at its own rows' means -- pandas' pairwise-complete
        covariance. One Gram whatever the gaps. Exact when the gaps have
        nothing to do with the features; where they do (a target present
        only on trade rows between market-data rows), each slope is scaled
        by the ratio of the feature's variance over the target's rows to its
        variance over all rows.

      Both solve `(C + ridge·I)·β = c_j` with `β_0 = ȳ_j − m_j·β`: one
      formula, read from a different Gram. `n_eff` counts every row in both,
      as rule 8 has it, and `min_periods` gates on it as before (S2 stays
      open). `lasso` takes `ewridge`'s centred cross-moments, which it needs
      for `m_j`: that is N2. `bank.gram()` returns one entry per Gram, each
      naming its targets, and adds `means_by_target` -- the column means
      over each target's rows -- which `po.gram.solve` needs to reproduce
      `"pairwise"`.

      State: `SCHEMA_VERSION` and `MIN_SCHEMA_VERSION` are both 8. The user,
      2026-09-14: "Do not worry about state saved before the next version
      release, we are pre 1.0 and we can change things now". So the
      schema-6 loaders the review round wrote go, with their frozen
      fixtures. Changes the numbers of every `ewridge` and `lasso` stream
      with a null target: a minor, with a CHANGELOG line.

      **Done 2026-09-14.** `crates/online-core/src/gaps.rs` holds what both
      models share: `TargetGaps`, the centred cross-moments (`Cross`, with
      the weight over every row that is `n_eff`), and `Grams`, the Grams
      and which one each target reads. Under `own_rows` a Gram learns the
      rows one of its targets has, ages over the rest (`EwCov::skip`, which
      holds a blocked Gram's skipped row as a zero-weight one rather than
      flushing), and splits where its targets part; a window truncates a
      Gram against the one its targets read at the boundary, and a blend
      mixes a Gram with the twin's of the same targets. The intercept reads
      the Gram's own mean under `own_rows` -- so a row the target misses
      leaves its fit where it was to the bit -- and the Gram's mean plus the
      target's offset under `pairwise`. Without gaps `ewridge` is the same
      to the bit as before, checked against the previous build's goldens.

      Before the fix, the new library tests failed as N3 predicts: the fit
      of a target present where `x0 > -0.5` was off by 1.17 at level 0 and
      by 49.2 at level 50, and a second gappy target of three by 6.18.
      Library tests, `tests/test_second_opinion.py::TestATargetWithGaps`:
      `own_rows` against `numpy.linalg.lstsq` and statsmodels' `WLS` on the
      target's rows (with decay, at a level, plain and standardized, three
      targets with three patterns of gaps), statsmodels' ridge
      (`fit_regularized`, `L1_wt=0`) with a penalty, and statsmodels'
      elastic net against `lasso` at a non-zero penalty, a noise feature
      zeroed by both; `pairwise` against pandas' `DataFrame.cov` and
      `numpy.cov` with weights; a `window` with gaps against both. In
      `test_gram_module.py`, `coef_stats` on a gappy target's Gram is
      statsmodels' OLS on its rows (coefficients, standard errors, R²).
      Rust: each target of a three-target bank is, to the bit, the Gram of
      a model of that target alone, blocked or not, and through a save; a
      blend is each target's own blend; RLS agrees with `ridge_decay` on a
      stream with gaps under `own_rows` and not under `pairwise`; a level
      moves no slope under either reading; `lasso` at `1e8` (N2).

      On the way, N5: `po.gram.solve(standardize=True)` and
      `po.gram.lasso_path` without an intercept scaled the centred
      co-moments against the raw cross-moments, the hybrid the models lost
      in the review's C8; both read raw moments now, with a test each.
      Re-frozen: the core goldens for `ew_ridge`, `ew_ridge_std` and
      `lasso` (a null target at row 31), and in the golden pipeline the
      `ridge`, `lasso` and `seqtest_compare` keys (a null every 29th row);
      no `n_eff` moved. A closed row carries one Gram, and its `n_eff` is
      that Gram's weight, as its `n_kish` was.

      A second read, at the user's request, found two things the tests had
      not. A zero-weight row with its targets parted split their Gram,
      although the row learns nothing and the copy was the Gram itself, kept
      twice from then on; it splits nothing now while there is weight to
      age, since taking the row at weight 0 and skipping it age the Gram by
      the same steps (`a_zero_weight_row_splits_no_gram`). And a target not
      seen yet, alone in a Gram with no weight, cost one jittered solve --
      counted in `solve_failures` -- on every solve under `ridge = 0`; that
      solve is skipped, the coefficients the zeros it gave
      (`a_target_not_seen_yet_costs_no_solve_failure`), and with a penalty
      the solve runs, so a target not seen yet still sits at its
      `coef_prior`. Tests added for paths nothing covered: a closed group's
      row per Gram, each solving to its own `coef`, and merged shards'
      Grams against the fit of their union by `numpy` and pandas.

- [x] 80. **The code review of 2026-09-12, worked through, begun
      2026-09-13.** `docs/REVIEW-2026-09-12.md` (and its pass-10 supplement)
      is the reviewer's; `docs/REVIEW-2026-09-12-PROGRESS.md` is the status of
      every finding in it, and the place to look. The rule for the round, the
      user's: fixes whose test is an independent library (`numpy`, `scipy`,
      `river`, and from 2026-09-13 `statsmodels`, `filterpy` and
      `bayesian-changepoint-detection`; `tests/test_second_opinion.py`)
      first, everything else kept for later with its reason. Commits carry
      the finding IDs. Tasks 78 and 79, parked on `design/task-78` while it
      ran, were merged back on 2026-09-13; 79 was this round's C5.

      **Continued 2026-09-14, after task 81: the findings no library can
      check.** The user: "We want to be careful fixing the items that
      cannot be verified by a third party library. As always a test first
      methodology is standard but we want to be sure that we double check
      each finding with a fresh eye since there is no other check, if we
      write the test and it passes where the reviewer assumes it would not,
      document the issue in the progress doc and raise it to me." Then: fix
      everything except D1 (hard rule 5, on not breaking file
      compatibility). So each remaining finding is re-derived from the code
      before its test is written; the test runs on the build before the
      fix and must fail where the finding says; one that passes is
      recorded and raised, not fixed. The findings that need a decision go
      to the user first.

      **The user's decisions, 2026-09-15**, each premise first confirmed on
      the build: S29/S30 -- `holt` becomes a weighted mean, level and trend
      `(λW·p + w·y)/(λW + w)`, so a weight means what it does elsewhere and
      `halflife = inf` is the cumulative fit (today the level at weight 0.5
      equals weight 1 to the last digit, and `inf` freezes the level at the
      first row). S2 -- each target's `min_periods` is checked against its
      own weight; the emitted `n_eff` stays the shared weight. S1 -- a
      windowed spec's `sigma` and `resid_z` are windowed. S3 --
      `max_dclock` caps the total delta a row sees after skipped rows (ten
      rows 100 apart handed the next one about 660). S23 -- `emit_averaged`
      compares the slots' errors as ratios, `exp(−eta·(σ²/σ²_best − 1))`.
      S27 -- `inf` is accepted where it means something and refused in both
      layers elsewhere. S31 -- a `strict_binary` target other than 0 or 1
      refuses the chunk, naming the row. C24 -- river's `FTRLProximal` has no
      forgetting (its sums only grow), so the halflife is ours: it stays,
      repaired with a decayed proximal sum, and at `halflife = inf` the
      model is river's to the bit, as T-R1 holds it.

      **Batch 1, 2026-09-15**, the core findings: C6, C7, S4, S6, S11, S13,
      S16, S32 and D2-D6 fixed, with N6, found on the way (`σ²`'s weight
      did not age on a row with a target and no prediction); S10 is not a
      defect. The progress doc has each, and what was raised: S10's
      premise, S13's test passing on the old build, S6 and D6 worse than
      written, C6's test unable to show it.

      **Batch 2, 2026-09-15**, the spec layer: S5, S7, S8, S21, S22, S24,
      S25, S26, C18, C20, D7, D8 and V23. `Spec::check` -- fill, validate,
      build -- is the one door every entry point uses (S25). Every new test
      failed on the old build. One call taken: `kalman`'s `coef_halflife`
      beside `q` is documented, not refused, since the docs already say `q`
      overrides it and specs use the pair.

      **P4 and P5, the user's decisions, 2026-09-15.** P4: the window's
      snapshot ring gets both bounds the user asked for, chosen by one new
      parameter that names the action and the budget, `window_budget =
      {"thin": MiB}` or `{"refuse": MiB}`. Past the budget the ring either
      refuses at runtime, naming the size and `window_every`, or thins: every
      other snapshot dropped and the spacing doubled, so the boundary grows
      coarser and never keeps an older row. P5: `fit_predict_batches` takes a
      `closed_groups` path and drains after each batch, and the docs say a
      `ModelBank`'s queue is bounded only by draining it; the runner writes
      its closed-groups file chunk by chunk.

      **Batch 3a, 2026-09-15**: C4, C19, P1, P2, P3, S12, V7 and V25 fixed;
      V5, V11 and V24 closed; V25 is not a defect for `kmeans`, which the
      review presumed. Batch 3b is P4 and P5, as decided above.

      **Batch 3b, 2026-09-15**: P4 and P5 fixed, as decided. Two calls of
      mine, raised: a window with no `window_budget` refuses past 256 MiB per
      ring; and a bank refused for its budget refuses every later
      `fit_predict`, `predict` and `save`, since the budget is found as the
      rows go in and the bank has by then learned part of the chunk (every
      other refusal is checked before a stream is touched).
      `fit_predict_batches(closed_groups=)` writes what it drained however
      the chunks stop, where the plan's source writes only at the end.

      **Batch 4a, 2026-09-15**: S29, S30, S31 and C24 fixed, as decided,
      with D9's and D10's `holt` and `ftrl` items; schema 9. Raised: every
      `holt` stream's numbers move (the weighted means' gains start at 1),
      `trend_halflife = inf` is the whole history's drift with no spelling
      left for a plain level, and C24's repair stops short of the review's
      numbers, since the penalties stay constants on the sums' scale --
      which is what keeps the undecayed model river's.

      **Batch 4b, 2026-09-15**: S2, S3 and S23 fixed, as decided, with
      D9's last items. S2 adds `OnlineModel::target_n_eff_into`, which the
      models that keep a weight per target override (docs/EXTENDING.md);
      the emitted `n_eff` is unchanged. The suite's oracles now cap the
      folded delta and gate a target's output on its own weight. Batch 4c
      is S1, S27, N4 and the rest of D10.

      **Batch 4c, 2026-09-15**: S27 fixed, as decided, and D10's rest. Seven
      parameters take `inf` where it names a limit, nine say `finite`, and
      `ridge` and `q` leave the Python table. Raised: the review's
      `ridge`/`q` half was the table's alone, since the builders validate
      through Rust. Two new observations fixed with it: N7 (`sgd`'s NaN
      `huber_delta` reached `f64::clamp`) and N8 (`marginal` refused
      `min_periods = inf`). S1 is batch 4d and N4 batch 4e.

      **Batch 4d, 2026-09-15**: S1 fixed, as decided: under a window the
      stream cuts its per-slot spread with a ring of its own, so `sigma`,
      `resid_z`, drift's scale, the conformal band and the slot ranking
      describe the rows the fit does. The ring rides on schema 9 and counts
      in `window_budget`. N4 is batch 4e.

      **Batch 4e, 2026-09-15**: N4 fixed: `bank.gram()`, a closed row and
      `merge` carry the centred cross-moments the model holds, and
      `po.gram.solve`, `lasso_path` and `coef_stats` read them. With it
      every finding of the review is fixed but D1, which the user excluded;
      what was raised is in the progress file.

      **Batch 5, 2026-09-15**: the review's second opinions that nothing had
      written. T-S18: `pa` is river's `PARegressor` to `1e-15`, once river
      is given the loss it documents (river 0.26.1's scores a target of 2
      as 3). T-S9: `ew_cov`, the target moments and `marginal`'s pair are
      pandas' `ewm`, absolute at `1e8`. T-S4's quantile half, which
      `quantile` does not meet: N9, task 82; `sgd`'s quantile loss does.

      **Batch 7, 2026-09-15**: the second review's findings
      (`docs/REVIEW-2026-09-15.md`): the quantile fit's warm-up and the
      per-target gate read the rows present, the band has a floor in the
      effective sample and takes least-squares rows while it holds under a
      row per coefficient (what rebuilds a fit a row at the input bound
      left behind), and a failed run publishes the closed groups it
      drained. Schema 10.

- [x] 79. **`label_delay` ignores a clock event on a skipped row — found
      2026-09-11, checking task 78's parity; in released 0.5.1.** A reset
      that lands on a row the spec skips (a null feature, an unusable
      weight) restarts the model but leaves the `label_delay` buffer full,
      so rows from before the reset are later learned into the model that
      was meant to start clean. Measured: `ewridge`, no decay,
      `label_delay=3`, `session_gap="reset"`, the first row of the new
      session carrying a null feature. `n_eff` per row:

    | row | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
    |---|---|---|---|---|---|---|---|---|---|---|---|---|
    | reset row skipped | 0 | 0 | 0 | 1 | 2 | — | **1** | **2** | **3** | 4 | 5 | **6** |
    | reset row accepted (null target) | 0 | 0 | 0 | 1 | 2 | 0 | 0 | 0 | 1 | 2 | 3 | 4 |

      The fresh model has learned a row by row 6 and ends two rows ahead:
      two rows of the old session leaked across the reset.

      *Cause.* `run_instance` acts on a row's `reset`, `session_changed`
      and `capped` before its `accept` test — deliberately, the comment
      says, "because a skipped row's gap breaks adjacency just as much".
      `apply_label_delay` does the opposite: its `!plan.accept` branch
      comes first and skips the event handling, so the buffer never sees a
      skipped row's reset (should drop), session change or capped gap
      (should release). The reset is measured; the other two go through the
      same branch and are expected to fail the same way — releasing late,
      across the break, into lag rings that were just cleared for it, the
      exact case `docs/REVIEW-E54-E64.md` L2 fixed for accepted rows.

      *Fix.* In `apply_label_delay`, handle the event for every row — a
      reset clears the buffer, a session change or capped gap releases it —
      and push only accepted rows. Tests: the table above as a regression
      test (`n_eff` equal in both rows from row 5 on, bar row 5's own), and
      the same shape for a session change and a capped gap on a skipped row,
      with a lag feature to show the ring is not refilled across the break.
      Changes numbers only for a stream with `label_delay` and an event on a
      skipped row; a patch, with a CHANGELOG line. Before 78b (task 106,
      discarded by the user on 2026-09-25), which would have extended the
      same release rule to a backward clock.
      **Done 2026-09-13** as the code review's C5, fixed with C21 in
      `cb6c57c` (task 80): a reset on a skipped row now clears the buffer,
      and a session change or capped gap on one releases it.

- [x] 78. **Windowed EWMAs in both directions, as columns -- built
      2026-09-30 (`34af36f`).** Its API is replaced by task 143's window
      expressions with task 144's names, and its targets by task 104; the
      core below carries over.

      #### What it computes

      For row *t* at policy time `τ_t`, over the rows *j* in its window:
      `y_t = Σ w_j λ^|τ_j − τ_a| v_j / Σ w_j λ^|τ_j − τ_a|`, `λ = 2^(−1 /
      halflife)`. `po.window.ewm` takes the rows less than `horizon` older,
      `(τ_t − h, τ_t]`, or every earlier row with no horizon;
      `po.window.lookahead_rewm` the rows after *t* less than `horizon`
      later, `(τ_t, τ_t + h)`. `po.stream.with_windows(frame, windows, ...)`,
      also `lf.online.with_windows` and `df.online.with_windows`, adds their
      columns in one pass; the input passes through.

      #### How

      - **The core** (`crates/online-polars/src/windows.rs`): segment sums
        anchored at the near end, combined associatively, in a two-stack
        queue -- O(1) a row on average, exact, no subtraction. Only rows with
        a usable value and a usable, non-zero weight enter a queue; a weight
        below zero is refused.
      - **The clock is the model's**: each group steps its own `ClockState`,
        so the horizon is measured on the policy clock. A gap past
        `max_dclock` or a session change ends the windows open across it as
        partial; a reset discards them. A group's policy time restarts at
        each such event. The rows also step one shared clock in input order
        across groups, under the same policy: a step back the policy
        refuses is refused naming the row, a reset there discards every
        group's windows, and a gap there ends them; with a group column a
        session is each group's (review R2, W3), without one the stream is
        the one group. A group silent for more than `max_dclock` of the stream's time
        has its windows cut then, which bounds how long it holds the output;
        a row-count clock has no such bound.
      - **Options**: `weight`; `split=(column, [values])` with `total` and
        `unlisted`; `partial` (`"keep"` backward, `"null"` forward, or
        `"drop"`); `complete`, a Boolean column per description and horizon;
        `same_clock` (stream order at one clock value); name templates over
        `{column}`, `{halflife}`, `{horizon}`, `{split}`, collisions refused.
        `like=spec` takes a spec's clock policy and nulls a look-ahead on
        rows the spec skips; clock keywords beside it are refused.
      - **Rows out** in input order, each once every window over it has
        closed; with only backward windows, as they arrive. The rows a
        look-ahead holds are the input's own Arrow chunks, sliced, never
        copied. A row unresolved at the end of the input is null unless
        `save_state` keeps it. Under `head(n)` the input is read up to the
        row that resolved the *n*-th output row. A projection reaches the
        input only when no state is loaded or saved. The input plan gets the
        bank's row-order and spent-stream checks.
      - **The state** is versioned msgpack: the call, the core, and the held
        rows as Arrow IPC. It resumes only the call that saved it, on the
        same kind of clock, over the same columns. The clock policy is
        stored in a tagged form, since `ClockCfg`'s own derive writes
        `session_gap = "reset"` as a unit that msgpack reads back as no gap.
        The clock-policy checks are shared with `Spec` (`clock_cfg_of`).
      - **As a target today**: the column fitted with `label_delay` at least
        the horizon, on the same clock, and a horizon no longer than
        `max_dclock` (the README); each prediction is then emitted one
        horizon late. Task 104 makes it native.

      #### Measured (docs/PERFORMANCE.md §32)

      9.5 to 9.8 million rows a second at the memory of reading and writing
      the file, flat in the window's length; Polars' `rolling` recipe for
      the same column 11 to 64 times slower, at 4.9 to 18.3 GB, agreeing to
      1.7e-13 where both give a value. As a target under `label_delay` it
      adds at most 0.2 s to the model's own time.

      #### Tests

      The core against a brute-force loop over every clock event, both
      clock kinds and a row-count clock (480 streams), nine bugs planted one
      at a time each caught; each direction against the definition and
      against Polars' `rolling`; interleaved trades and quotes with three
      VWAPs equal to one split window, to the bit; every event; chunk
      invariance at 1, 7, 64 and 1000 rows; a save and load at every row; a
      slice, a filter and a projection; names and collisions; queues flat
      across quote density; held rows sharing the input's buffers; windows
      feeding a model in one query equal to two steps
      (`tests/test_windows.py`, `windows.rs`, `windows_frame.rs`).

- [x] 77. **A plan with nothing to write runs once where a query uses it
      twice, 2026-09-10.** The IO source declares `is_pure=True` to
      `register_io_source` exactly when the run has no writes — no
      `save_state`, no `closed_groups` sidecar. Polars then shares one
      execution where a single query uses the same plan twice (a self-join,
      `pl.concat([plan, plan])`) instead of running the source twice,
      concurrently, as R2 of `docs/STATE-WORKFLOW.md` measured before this.
      That covers every `predict` and every fit that keeps its state in
      memory.

      **The condition is the point of the entry.** The first attempt
      declared purity unconditionally, on the strength of a measurement:
      with `save_state` set and `pl.concat([plan, plan])`, the query yields
      8,000 rows and the state file it leaves is **byte-identical to the
      file a single ordinary run leaves**. The whole suite passed on
      py-polars 1.44.1 and 2.0.0rc1 alike, 2,443 tests. So it worked — and
      it was still the wrong declaration, because it answered "does dedup
      break anything today" when the question was "is this source pure".

      *What the flag means, read rather than inferred.* In polars' source,
      `is_pure` is the only thing that can make two `PythonOptions` compare
      **equal** (`crates/polars-plan/src/plans/ir/equality.rs`: the
      comparison opens `(*l_is_pure && *r_is_pure) && ...`), and IR node
      equality is what CSE and plan dedup are keyed on. So the flag is not a
      request to share a run; it is an assertion that two occurrences are
      the same node, and dropping one of them drops its **effects** as well
      as its work. A source that writes a file has effects. The rows here
      are pure by construction — `make_bank()` is called inside `source`, so
      every execution starts from the same bytes (R3) and feeds them the
      same rows in the same order, which is exactly R2 — but identical bytes
      are not no bytes, and R2's idempotence is a weaker property than
      purity, not the same one.

      *We cannot dedupe the impure case ourselves, and it does not need it.*
      Polars hands the source callable
      `(with_columns, predicate, n_rows, batch_size)` and nothing that
      identifies an execution, so a second concurrent run within one query
      is indistinguishable from a later `collect()`, which must re-run (R2).
      Overlap in time is the only signal left, and sharing rows between two
      consumers pulling at their own rates means buffering the whole stream
      — the memory bound this library exists to hold. Nor is there anything
      to fix: the two runs write identical bytes, atomically, and
      `atomic.rs`' counter already makes the pair safe. The cost is
      duplicated work in one rare plan shape, which is a price, not a bug.

      *An objection that turned out to be about something else.* The first
      reservation raised against the flag was that purity licenses polars to
      skip the source entirely, so a query asking for no rows would silently
      not write. Measured: `head(0)`, `limit(0)` and `clear()` run the
      source **0 times with `is_pure=False` and 0 times with
      `is_pure=True`**. That is polars pruning a source it has no use for.
      It predates this line and is unrelated to purity.

      No new test: the property is R2's, already covered by C1 and C3, and
      what a test could pin here is polars' scheduling choice rather than
      our behaviour.

- [x] 71. **E51, the blocked rank-B Gram update — design settled 2026-09-08,
      `EwCov` half built and reviewed the same day, `ewridge` wiring built
      the same day.**
      Three things were measured before any of it was built, because the
      ledger row's own numbers turned out to need qualifying — and then the
      review of the built half found two more, both of which had passed
      every test.

      *The speedup is real, and the row's 5–10× is honest — single-threaded.*
      The design probe first reported **9.5× at k = 1,000 and 11.2× at
      k = 2,000**, and those numbers were wrong in a way the review caught:
      the probe multiplied with `faer`'s `*` operator, which parallelises
      over the global rayon pool by default, against a rank-1 baseline that
      runs on one core. The bank runs each group on its own pool
      (`crates/online-polars/src/pool.rs`), so the honest comparison is one
      core against one core: a full GEMM measures 3.6×, a triangular product
      (`syrk`-style, lower half then mirrored) 5.4×, and the flush as built —
      the triangular product with no `S` allocation and the corrections
      applied on the triangle — **6.6× at k = 1,000 and 6.65× at k = 2,000**
      with a 256-row block, 4.6× at k = 256; a 64-row block gives
      4.9×/4.5×/2.9×. The rank-1 path is memory-bound (it re-touches the
      whole `k×k` at ~110 GB/s every row); the product is compute-bound at
      ~50 GFLOP/s on one core. So the product is `Par::Seq`, always.

      *The review's first finding: the flush as first written measured
      1.0×.* The merge was a hand-rolled triple loop, which is the rank-1
      path's flop count in a different order. The tests could not see it,
      because they check the answer, not the time. It is now
      `faer::linalg::matmul::triangular::matmul` on rows pre-scaled to
      `√uᵢ·(xᵢ − m_B)`, so that `DᵀD` *is* `S` and the product writes into
      the lower triangle of `C` in place. Rule for the row: **a plan row's
      speedup is measured on the implementation, single-threaded, not on
      the probe** — the memory note from E48 ("measure before believing a
      plan row") applied twice in one task.

      *The review's second finding: 49% of the variance, lost to
      cancellation, on a stream every test passed.* The merge was centred on
      the running mean `m_A`, which the design chose over zero (E67's
      cancellation at an offset of `1e7`). But when the history is *nearly*
      gone — an uncapped gap leaving `W_A = 1e-30`, not zero — `m_A` is
      still where the old data was, and the block's scatter is formed as a
      difference of two numbers of order `U·(m_B − m_A)²`: 49% of the block's
      variance at an offset of `1e7`, 24% centred on the block's first
      weighted row instead (that row can be the one *before* the gap). The
      rank-1 recursion is exact here, because its mean jumps to the row. So
      the flush is **two passes: the block's own weighted mean `m_B` first,
      then the product on rows centred on it**, at a cost of one extra `B·k`
      pass. Measured `8.4e-10` of the variance on that stream (the data's own
      resolution at `1e7`) and `3e-14` at `1e3`, and the test that pins it
      (`a_history_all_but_gone_leaves_nothing_to_cancel`) fails at 17% with
      the centring switched back.

      *The merge, as built.* With `Λᵢ` the suffix decay to the block end,
      `uᵢ = wᵢ·Λᵢ`, `U = Σuᵢ`, `W_A = (Πλᵢ)·W_open`, and `W'` the weight
      after the block from the scalar recursion (below): `m_B = Σuᵢxᵢ/U`, `dᵢ = xᵢ − m_B`, `S = Σuᵢdᵢdᵢᵀ`
      (the product), `Δ = m_B − m_A`,
      `C' = (W_A·C_A + S)/W' + (W_A·U/W'²)·ΔΔᵀ`, `m' = m_A + (U/W')·Δ`.
      (The code keeps a residue term `δ = Σuᵢdᵢ/U` that is zero in exact
      arithmetic and `1e-16` in floats, so the general-origin form is what is
      written: `C' = (W_A·C_A + S − U·δδᵀ)/W' + (W_A·U/W'²)·ΔΔᵀ` with
      `Δ = (m_B − m_A) + δ`.)

      *The scalars can stay bit-identical, which the row did not anticipate.*
      `w_sum`, `q_sum`, `prior_scale` and `precision_scale` are an `O(B)`
      recursion, not an `O(B·k²)` one, so the block runs the shipped scalar
      path exactly, per row, as each row is held. Measured against the
      sequential recursion over weights, zero-weight rows and decay: all four
      match to `0.0e0`, and the mean to `1e-16`. Only the `k×k` matrix moves,
      at `5e-13` of the largest variance (offset `1e3`) to `8e-10` (`1e7`).
      **`n_eff` is emitted every row, so this is what keeps blocking from
      changing any per-row output.**

      A closed form for `precision_scale` is *wrong* and was caught here:
      `Π aᵢ = Π λᵢ · W₀/W_B` looks like it telescopes, but the shipped update
      **resets** the product to 1 whenever the history has no weight
      (`a ≤ 0`), which the product cannot see. Measured 0.84–0.99 relative
      error before the scalar pass replaced it.

      *The constraint the row does not mention.* The Gram must be flushed
      before it is read, so the effective block is
      `min(gram_block_rows, solve_every)`. That is not fatal — at k ≥ 1,000 a
      solve is ~3·10⁸ flops, so anyone in the regime E51 targets already
      solves rarely — but it means the parameter cannot be documented without
      it. **And it is incompatible with `window`**: `Moments::of` reads the
      matrix on every row to build the truncated snapshot
      (`crates/online-core/src/window.rs`), which would flush every row and
      buy nothing. The wiring refuses `gram_block_rows` together with
      `window` rather than silently degrading.

      *The read surface narrows the blast radius, once looked at.* `n_eff`,
      `n_kish` and `q_sum` read scalars only; `mean`, `cov`, `raw`, `var`,
      `comoments`, `means` and `precision` read the matrix. The scalars are
      advanced **eagerly, per row, by the shipped path** even while rows are
      held, so nothing that is emitted per row can ever lag — the flush needs
      only `W` at the moment the block opened, which is one snapshot. And
      `comoments`/`means` hand out `&[f64]`, so a `&self` method cannot
      flush; instead every matrix read carries
      `debug_assert!(!self.has_pending())`, which turns a forgotten flush
      into a failure on the first test that reads mid-block rather than a
      wrong number in production. `set_moments` and `decay` flush first: the
      held rows were measured against the state those two replace or age
      (`blend_toward_long_run` is the caller that reaches `set_moments`).

      *What the flush may be triggered by.* Only functions of the row
      sequence: the block filling, a read on the learned-row schedule
      (`solve_every`), `set_moments`, `decay`. **Never a chunk ending** — a
      flush every row is the rank-1 association and a flush every sixteen a
      merge, and the last digits differ, which is the chunk-dependence rule 3
      forbids. The test with teeth is the bank's chunk-invariance test with
      `gram_block_rows` on (wiring, below).

      *The state.* The held block travels in `EwCov`'s state as a last field
      `pending` (`block_rows`, `x` row-major `B×k`, `lam`, `w`, `w_open`),
      skipped when blocking is off so every existing state file is
      byte-identical; a bank saved mid-block resumes on the same block
      boundary. `SCHEMA_VERSION` stays 6. Two consequences found on the way:
      `q_sum` no longer has `skip_serializing_if` (the crate's compact
      encoding is positional, so at most one field may skip and it must be
      last — `tests/state_encoding.rs`; `None` now writes a `nil`, and named
      bytes for `Some` are unchanged), and `ModelState::EwCov` is boxed like
      every other variant, because the extra 88 bytes tripped
      `clippy::large_enum_variant`. Noted, not changed: `EwCovModel`'s `lag` and `win` both
      skip at its tail and `EwRidge`'s `tm` skips mid-struct, which the same
      rule would reject; benign because the bank writes the named form.
      The JSON export (task 69) shows a held block as named finite fields.

      *Reproducibility.* The rank-1 path is bit-reproducible across CPUs;
      the product is `faer`'s and its kernel dispatch follows the CPU's
      vector width, so a blocked Gram's last bits can differ between
      machines. **Frozen fixtures must not enable blocking**, and the
      cross-platform state hand-off stays on the rank-1 path.

      *Only `ewridge` turns it on.* Blocking pays exactly where the Gram is
      consumed on a schedule rather than per row, which is `ewridge` (and
      later `lasso`). `ew_cov` emitting per-row statistics would flush every
      row and gain nothing. So `EwCov` gains the machinery but the five other
      models on it — `ew_cov`, `ew_class`, `corrchange`, `hmm`, `deco` —
      never set `block_rows` and run the untouched path, to the bit
      (`block_rows_zero_is_the_untouched_path`).

      *Tests, in `crates/online-core/src/ewcov.rs::block_tests` (19).* The
      shipped recursion is the oracle throughout. Co-moments are compared
      against the matrix's own scale at the data's own resolution
      (`max(1e-9, 64·|m|·ε/σ)`), not entry by entry — a per-entry relative
      tolerance fails on an off-diagonal that happens to be near zero while
      saying nothing about the merge. Sweeps over `k ∈ {1, 3, 8, 37}`,
      `λ ∈ {1, 0.995}`, three weight patterns, blocks `{2, 7, 64}`, offsets
      `{0, 1e3, 1e7}`; a `k = 131` case for the product's tile edges; the
      near-total-decay stream above; the capped gap (`λ = 0` exactly, rule 9)
      at four positions in a block; a block of nothing but zero-weight rows;
      a zero-weight first row (opens no block, exact); the block boundary
      itself; a read mid-block (`should_panic`); the scalars readable
      mid-block; save/resume at five cuts in both msgpack encodings; the
      JSON shape; block sizes 1 and longer than the stream; resizing
      mid-stream; `decay` and `set_moments` mid-block, both verified to fail
      without their flush. `cargo mutants` cannot see any of the Python
      suite, so this module is where the coverage has to be.

      **The wiring, as built (2026-09-08).** `gram_block_rows` on `ewridge`
      only, `0`/`None` off. In `EwRidgeCfg` the field is `#[serde(default)]`
      and sits *before* `window`, not at the tail: the crate's compact
      encoding is positional and `window` is the one skipping field that
      may be last, so a trailing `usize` would let a saved `window` decode
      into the new slot. The bank writes the named form, so a state saved
      before the field loads with `0` and re-saves with the key.

      *Flush discipline.* `EwRidge::new` blocks its `cov` and the slow
      twin's; `solve()` flushes before it reads; `blend_toward_long_run`
      flushes both twins before it mixes them and carries `block_rows` into
      the accumulator it rebuilds; `restore` leaves the pending block where
      the save left it (a `set_block_rows` there would flush and move the
      boundary). The bank's `gram_of` reads through a new
      `EwCov::flushed() -> Cow`, so `gram()` — the one read of the matrix
      from outside the model — reports the held rows without merging them:
      a read is not a function of the row sequence and must not move the
      block. What may flush is
      unchanged from the design: the block filling, a solve on its
      clock/row schedule, a session blend. Never a chunk ending, and the
      bank's chunk-invariance test with blocking on holds every column,
      `coef_*` included, at 1, 7, 13, 100 and 333 rows a chunk.

      *Three refusals, in `EwRidgeCfg::validate`, surfacing as
      `ValueError` from the builder and the bank.* With `window` (it
      snapshots the Gram on every row, so nothing would be held). With
      `solve_every <= 0` or `max_rows_between_solves <= 1` — a solve every
      row flushes every row, so a block could never hold more than one;
      the message names the default (`halflife / 50`, which is `0` for
      `lam` and an infinite halflife, so those need an explicit
      `solve_every`). And a budget: `B × (k + intercept) × (1 + twin)`
      floats over 256 MiB (`GRAM_BLOCK_BUDGET`, the `bins` pattern), the
      message giving the GiB it would have held.

      *Tests.* Rust, `ewridge::tests` (5): blocked against per-row at
      blocks 1, 3, 16 and 256 (`pred` to `1e-9` of scale, `n_eff` to the
      bit); a solve merges the block first; a blend merges both held blocks
      and keeps the block size; a state saved mid-block resumes bit for bit;
      each refusal. Python, `tests/test_gram_block.py` (21): the same fit
      through the bank at blocks 3, 16 and 256; `0` bit-identical to the
      untouched path; the chunk sweep above; save mid-block and resume bit
      for bit; the held rows legible in the JSON export (`n_held ×
      (k + 1)` values, intercept slot included); `gram()` mid-block
      reporting the held rows and moving nothing; a session change blending
      the held block first; every refusal, including the two default
      cadences; the budget message; and a state file written before the
      field existed — made by stripping the key from a current file with a
      sixty-line msgpack codec in the test, since the loader cannot tell the
      two apart — loading with blocking off, continuing to the bit and
      re-saving with the key. `tests/test_semantics_all_models.py` gained a
      `VARIANTS` list with a blocked `ewridge` (block 5 against a 4-row
      cadence, so every solve merges a partial block), swept through null
      policy, warm-up, weights, resets and the rest of the contract, and
      through `test_predict`'s row oracle; it is a second list because
      `test_model_registry` holds `MODELS` to one entry per model, and the
      first gate run said so.

      *Reviewed (2026-09-08)*, both commits, for bugs and for tests missing:
      no bug. Every matrix read was traced — inside the model only `solve`,
      `solve_standardized` (called from within `solve`) and the blend read
      the moments, and each merges first; outside it only `gram_of`, which
      reads `flushed()`, and both `gram()` and the closed row go through it.
      The gap was in the tests, not the code: the `debug_assert!` on every
      matrix read is compiled out of the release build the Python suite
      runs, so a bank reader that forgot `flushed()` would have passed the
      whole suite and returned stale moments. Added: Rust,
      `crates/online-polars/tests/bank.rs::every_bank_reader_survives_a_held_block`,
      a debug-build pass over every reader — closed groups, `gram` (both
      forms), `coef`, `last_row`, `summary`, `describe`, `predict`, a
      save/load round trip that carries the held rows — with rows held in
      every open group, then the stream continued against an untouched bank
      to the bit; replacing `flushed()` with a borrow makes it panic from
      inside `fit_predict`, which is the check that it has teeth. Python,
      `tests/test_gram_block.py` is now 31: the blocked fit against the
      plain one on every path the solve and the plumbing take —
      standardize (with and without intercept), a ridge × feature-set grid,
      a coefficient prior with `ridge_decay`, two halflives (the slow
      twin), `label_delay`, a drift reset and groups; scoring with
      `predict()` mid-block, which gives the plain model's numbers and
      moves nothing in the state; and a group closed mid-block, whose
      closed row carries the merged moments (checked against an unblocked
      bank's `gram()` at the same row).

      Two things worth knowing that are not bugs. A state saved by this
      build with an `ewridge` spec, blocked or not, is refused by a 0.3.0
      build: `ModelKind` is `deny_unknown_fields` and always writes
      `gram_block_rows` (nil for none). Loud, and the precedent — `window`
      did the same to 0.2.0 — forward compatibility is not promised. And
      "the block never exceeds the solve cadence" is a statement about the
      main accumulator: the slow twin's block is merged when it fills or at
      a blend, never at a solve, because a solve does not read the twin.

      *Measured*, `crates/online-core/examples/gram_block_bench.rs`, the
      `EwRidge::step` — prediction, cross-moments and Gram — single-threaded
      (`Par::Seq`), one core, this machine; rows per second and the ratio
      to the per-row update:

    | k | solve | per row | block 64 | block 256 |
    |---|---|---|---|---|
    | 256 | never | 91,214 | 372,772 (4.1×) | 464,135 (5.1×) |
    | 256 | every 512 rows | 85,578 | 288,352 (3.4×) | 347,461 (4.1×) |
    | 1,000 | never | 5,650 | 28,593 (5.1×) | 37,167 (6.6×) |
    | 1,000 | every 512 rows | 5,711 | 20,863 (3.7×) | 23,719 (4.2×) |
    | 2,000 | never | 1,415 | 6,544 (4.6×) | 8,380 (5.9×) |
    | 2,000 | every 512 rows | 1,309 | 4,435 (3.4×) | 5,644 (4.3×) |

      The `never` rows are the update alone, which is what the block
      changes; a real cadence puts the same `O(k³)` solve on both sides and
      dilutes the ratio — at `k = 2,000` a solve costs ~30 ms whichever way
      the Gram was built, and every 512 rows that is a third of the blocked
      path's time. The ledger's 5–10× is honest for the update, single-
      threaded; the number a bank sees is that, times how rarely it solves.

- [x] 68. **README clarity pass, begun 2026-09-07, done 2026-09-09.** Going through the
      reader-facing prose and fixing phrasing that only parses if the reader
      already shares the frame the sentence was written in. Three rules, in
      the order they bind:

      1. **One idea per sentence.** A sentence carrying a rule, its reason
         and its exception makes the reader hold all three to get any one.
      2. **A sweep belongs in a table.** A list of eight things behind
         semicolons is a table that has not been drawn yet. `bocpd` already
         had a table for its output fields while `micro`, two sections away,
         had the same content in prose.
      3. **Name the mechanism, do not allude to it** (the user's rule,
         2026-09-07, and the one that generalises). "What it costs is state,
         not data" is clear only to a reader already thinking about memory;
         "memory use is proportional to the model's state, not the amount of
         data that has passed through it" needs nothing brought to it. The
         test is whether a sentence can be understood by someone who has not
         yet been told what the sentence is about.

      Prose only: no heading moves, no code block changes, no claim altered,
      so the anchors and `tests/test_examples.py` hold throughout. Progress
      is measured, not eyeballed -- sentences of 45+ words in the README went
      from 37 to 25 in the first batch of twelve edits. Rendered for review
      through GitHub's own `POST /markdown` so the judgment is made on what
      the repo page will show, not on an approximation.

      **Done 2026-09-09, as a whole-README rewrite rather than a sentence
      pass.** The user reported phrasing problems one at a time into
      `docs/PHRASING.md` (each kept verbatim, with the reading and the
      resolution separate), the rules those reports implied were written
      into `docs/WRITING.md`, and the README was redrafted to them over four
      review rounds on a rendered page before anything was committed. The
      three rules above survived and gained a stronger one ahead of them: a
      stated reader — statistics, a little Polars, time-ordered data, and
      nothing of the internals of Polars or of this project — against whom
      every sentence is read. What that changed: the introduction defines
      *spec*, *model bank*, *stream*/*chunk* and *state* before using them;
      "the bank" is announced as always meaning a model bank; Polars-internal
      words (plan, sink, collect, struct column) are replaced by what they
      do; the stream section is one subsection per concept; and prose that
      described a parameter, a structure's fields or an API call is a code
      block with comments — 59 of them in the README, every one run by
      `tests/test_production_hardening.py`, which is how three keyword names
      the old prose had wrong were found. `po.run` and the CLI moved to
      `docs/RUNNER.md`, whose blocks the same test now runs. Four things a
      reader would have needed to know in advance were found by that test
      and not by reading — the point of the rule that every block runs.
- [x] 61. **The leak test's statistic, 2026-09-06.** `assert_plateaus` compared
      the first and last of its post-warm-up marks, which cannot distinguish a
      late allocator step from a slope — the distinction its own docstring
      claims. It failed `main` on a tree whose previous run was green. It now
      takes the median block-to-block gap over five blocks; the CI trace that
      failed reads 0.0 KB/iter, a sustained 14 KB/iter leak still reads 14.
      Acceptance: the file's 16 tests, and the three traces (the real one, a
      leak, a step) checked against the statistic directly.
- [x] 62. **The release gate, 2026-09-06.** The `Pypi` environment now has the
      owner as a required reviewer (self-review allowed, or one reviewer
      deadlocks it), so `publish to PyPI` waits for a click that is only asked
      for once every wheel, the sdist and the state hand-off are green. The
      gate is in front of the one step nothing can undo and in front of
      nothing else: rehearsals, re-runs, the GitHub release and the Pages
      deploy all proceed unattended. A wrong tag is left unapproved rather
      than moved — `refs/tags/v*` has `deletion` and `update` rules with no
      bypass — and the fix ships as the next version, which is PyPI's rule
      too. Recorded in the comment above the `publish` job and in
      `docs/RELEASE-READINESS.md`, "The release gate".

- [x] 87. **Not using a model before it is ready, 2026-09-21
      (`docs/WARMUP-AND-CONVERGENCE.md`).** Two gates stated as intent, on
      every entry point: `min_settled_frac`, a fraction of steady state the
      decay window must have filled (`settled_frac = 1 − 2^(−T/h)`, `T` the
      decay time seen -- rate-independent by construction), off by default;
      and `max_error_inflation`, the ratio by which estimation error may
      inflate a prediction's error over the noise floor,
      `sqrt(1 + edf / n_kish)`, default `sqrt(2)`, which is `ew_ridge`'s
      readiness gate now (its `min_periods` defaults to 0; the `k + 1` rows
      its first solve needs are the model's own floor). Per row: `settled_frac`
      and `withheld_reason` (a categorical, 1.4 B/row) on every model that
      writes a row; `support_coef` beside `coef` for `ew_ridge`, each
      coefficient's data share `1 − λ(Σ̂⁻¹)_jj`; opt-in
      `error_inflation_<slot>`, the row's own leverage against the fit's
      factor. `summary()` carries the same per group; a `ReadinessWarning`
      names, once per (spec, group), a coefficient more ridge than data or a
      noise gate the stream has settled below. Built for `ew_ridge` only:
      `rls`, `lasso`, `kalman` and the scalar models keep `min_periods`
      (the doc's §5.8 says what each would read). Held to identities rather
      than tuned (`crates/online-core/tests/readiness.rs`): the gate within
      2% of the observed error down to `n_kish ≈ 2.3 k_eff`; the per-row
      form conservative by 10–54% at small `n`, which is what an exact `G₂`
      would buy (doc §2.1). Schema 13. Released in 0.9.0.
- [x] 88. **Durations for a time clock: `timedelta` half-lives on a
      `Datetime` clock, requested 2026-09-23.** A `Datetime` or `Date` clock
      column is accepted when the spec's time parameters are given as
      durations -- a Python `timedelta`, as asked -- so a halflife states its
      own unit ("ten minutes") instead of being a number in the column's
      storage unit. Today such a column is refused (the clock dtype decision
      of 2026-08-30 below), because casting it to f64 exposes its internal
      representation: the same 60 seconds reads as 60e3, 60e6 or 60e9 for
      `Datetime(ms/us/ns)`, so `halflife = 600` on a microsecond column
      silently meant 600 µs. That decision declined auto-converting to
      seconds because a halflife's meaning would then depend on the input
      dtype. **Durations dissolve that objection**: with both the clock and
      the parameter carrying units, the meaning depends on neither, and the
      library converts both to one internal scale. So this is the principled
      form of the refusal, not a relaxation of it. Precedent, and the oracle:
      pandas' `ewm(halflife=pd.Timedelta(...), times=...)`, which T-S9
      already uses as the second opinion on the clock model.

      *The rule, as two refusals.* A temporal clock with plain-number time
      parameters is still refused, with today's error still pointing at
      `dt.epoch`. A numeric clock with duration parameters is refused too.
      Either mixture brings the ambiguity back. Numeric clocks with
      plain-number parameters are unchanged.

      *The surface is every parameter measured in clock units*, spec-level and
      model-level -- for example `halflife` and its list form, `max_dclock`,
      `session_gap`, `min_backwards_jump`, `label_delay`, `solve_every`,
      `window`, and model halflives such as `kalman`'s `revert_halflife` and
      `holt`'s `level_halflife`. That list is not the source of truth: a
      completeness test walks every clock-unit field and fails on any that
      lacks the duration form, so a parameter added later cannot be missed.

      *Other surfaces.* The CLI's TOML has no duration type, so it needs a
      string spelling; polars' own duration strings (`"10m"`, `"1h30m"`, as
      in `rolling(period=)`) are the natural one. The Arrow import path (task
      86's unbuilt half) receives a timestamp's unit in the Arrow type, so it
      must read the clock's unit from the schema rather than cast blind.

      *Tests.* The T-E10 trap as a pass: the same wall-clock data as
      `Datetime(ms)`, `Datetime(us)`, `Datetime(ns)` and a `dt.epoch("s")`
      float column, with `halflife = timedelta(minutes=10)` against
      `halflife = 600`, gives identical output on every float field. Both
      mixtures refused, each naming the column, the parameter and the fix. A
      `Date` clock with day durations. The pandas oracle on an irregular
      clock. A timezone-aware `Datetime` across a DST change: deltas are taken
      on the UTC epoch, so the change must not stretch the clock. The
      completeness test. A spec with durations saves and loads -- the spec's
      serialised form changes, so `SCHEMA_VERSION` moves (pre-1.0: raise
      `MIN_SCHEMA_VERSION`, no loader).

      *Open, for the build to decide.* Whether Python also accepts the string
      spelling beside `timedelta`, which would make a spec read the same in
      Python and TOML. Whether a `Duration` column may be a clock; a `Time`
      column stays refused, since time of day wraps at midnight and is not a
      clock. Whether the spec keeps a parameter as the duration the user wrote,
      so `summary()` can show "10m" rather than a converted number.

      **Built 2026-09-23**, on the user's ask to express a temporal clock's
      halflife, `max_dclock` and the other clock-scaled parameters "in
      pl.duration", and to reject a duration that is incompatible with the
      data. The open questions, as the build settled them:

    | question | settled |
    |---|---|
    | the forms Python takes | three: a `pl.duration(...)` that reads no column (evaluated once, at the builder), a `timedelta`, and polars' duration text (`"10m"`, `"1h30m"`), so a spec reads the same in Python and TOML |
    | what the spec keeps | the text: a `timedelta` or an expression is written by the Rust formatter (`timedelta(minutes=90)` is `"1h30m"`), and text is kept as written (`"90m"` stays `"90m"`), so `bank.specs` equals the dict that built it |
    | units | whole numbers of `ns`, `us`, `ms`, `s`, `m`, `h`, `d`, `w`; `mo`, `q` and `y` are refused (no fixed length), and so is `i` (it counts rows) |
    | a `Duration` column | a clock, like a `Datetime`: elapsed time is time |
    | a `Time` column | refused: a time of day starts again at midnight |
    | `0` and `inf` | unit-free, so they may stay numbers beside durations |
    | a number in the clock's own units | `lam` (a decay per unit) and `kalman`'s `q` (the noise a row one unit after the last adds) have no duration form, so a temporal clock refuses them and names the half-life to give instead |
    | the internal scale | the column's own integer nanoseconds (`ArrowCol::Nanos`), and the gap between two rows taken in integers before it becomes seconds (`ClockValue`), so one instant in ms, µs or ns is the same value and a nanosecond gap is exact whatever the stream's age; the previous row's instant is the clock state; a value past what nanoseconds in an `i64` hold (the years 1677 to 2262) is refused by row; a zone-aware column is read as its UTC instants |
    | outputs in clock units | seconds: `holt`'s trend is per second; `summary()`, `groups()` and `closed_groups()` give clock values as seconds since 1970 (`ClockValue::seconds`); `output_index`'s `halflife` is seconds; a grid's field names keep the text (`@h10m`) |

      *Incompatible, and refused.* A temporal clock with a clock parameter
      as a plain number (naming the column, the parameter, the duration fix
      and `dt.epoch`); a numeric clock with a duration (naming the column,
      the parameter and `pl.from_epoch`); one spec that gives both; a
      duration with no clock column; a rate on a temporal clock; a `Time`
      clock; a temporal clock column that another spec also reads as a
      feature, target or weight, which in seconds would be a number nobody
      asked for; and a duration the parameter's own rule refuses, such as a
      zero halflife, which is checked on its seconds.

      *The state.* The model layout, `SCHEMA_VERSION`, is unchanged, so the
      plan's guess above was wrong: nothing to raise. The envelope,
      `BANK_FORMAT_VERSION`, is 3, and a bank writes 3 only when a spec
      carries a duration. Every other file is still version 2, byte for
      byte, so an older build reads it, and refuses a file with a duration
      by its version rather than by a type error inside a spec.

      *Completeness, three ways.* `CLOCK_FIELDS` in `spec.rs` lists the
      clock parameters, and a Rust test walks every field serde knows and
      holds the table to the types; dropping `marginal`'s `window` from it
      fails the test. `Spec::clock_spans` walks the fields directly, since
      the chunk builder calls it on every chunk, and a second test ties it
      to the table. In Python, `tests/test_temporal_clock.py` holds the
      table to the builders' annotations and to their docstrings (12 of the
      15 model parameters say "clock unit" in their own entry; `solve_every`
      in `lasso`, `huber` and `quantile` defers to `ewridge`'s), and fits
      every entry both ways on a temporal clock.

      *Measured.* The same wall-clock data as `Datetime(ms)`, `(us)` and
      `(ns)`, with the parameters as `pl.duration`, `timedelta` or text,
      give output identical to the bit to a float clock in seconds with the
      same parameters as numbers; a `Date` clock with `"5d"` matches a
      clock in days with `5`; a zone-aware clock across the March change
      matches its UTC instants and differs from the wall clock. Ticks 1 to
      5000 ns apart under a 1 µs halflife decay by their exact gaps to
      1e-12, where the same instants as a float clock in epoch nanoseconds
      (a double at 1.7e18 resolves 256 ns) drift by more than 1e-3. Started
      a day, a year and a decade into the stream, the same ticks are read
      to the double's resolution at that age, and the drift is at most that
      resolution over the halflife: 3.9e-6, 9.6e-4 and 1.1e-2 against
      bounds of 1.5e-5, 3.7e-3 and 6e-2, which a test pins.

      *The origin (review of 2026-09-24).* The first build read the clock
      as seconds since 1970, and the pandas oracle on a nanosecond clock
      agreed only to 1.2e-9: a double at 1.7e9 seconds resolves 2**-22 s,
      0.24 µs, so nanosecond ticks were being rounded. The user saw that a
      nanosecond timestamp and `pl.duration` are both exact and asked why
      the result was not. The second build kept each temporal clock
      column's first instant in the state (`ClockOrigins`) and read the
      column as seconds from it, which moved the rounding from 0.24 µs to
      the double's resolution at the stream's age (the table above); the
      README said nothing was lost, and the user asked whether that was so.
      Against the exact nanosecond recursion that build agreed to 2.2e-16
      at the origin; pandas' own `ewm(times=)` is 1.8e-9 from it, so the
      pandas oracle's tolerance is pandas', not ours.

      *Exact gaps (2026-09-24, later the same day).* The user asked whether
      perfect precision was possible, and it was cheaper than the origin:
      the models never see a clock position, only `d_clock`
      (`OnlineModel::step`), and the one subtraction of two positions is in
      `ClockState::advance`. So the clock crosses the chunk as the column's
      own nanoseconds (`ArrowCol::Nanos`; a `Datetime("ns")` column is
      taken as it is, with no pass over it), `ClockState` keeps its
      previous row's value as a `ClockValue` -- `F64` for a numeric clock,
      `Ns` for a temporal one -- and takes the gap between two `Ns` values
      in integers before it becomes seconds. A numeric clock's arithmetic
      is untouched, so every numeric-clock test pins bit-identity. The
      origin, its state field, the summary's offset and the frame's
      `predict` special case are gone; `summary()`, `groups()` and
      `closed_groups()` report a temporal clock as `ClockValue::seconds`,
      seconds since 1970. Schema 14 (`MIN_SCHEMA_VERSION` 14, the pre-1.0
      rule). Memory: a clock column is 8 bytes a row in either form, and
      an `Option<ClockValue>` is the 16 bytes an `Option<f64>` was. Time,
      2M rows of `ewridge` with four features, best of three on a quiet
      machine: a `Datetime("ns")` clock 0.365 s before and 0.367 s after;
      the float clock, whose code did not change, 0.322 s and 0.336 s in
      the same runs, so the change is inside the run-to-run noise. Tests:
      `TestEveryUnitAgainstEveryColumn` holds every duration unit (`ns`,
      `us`/`µs`, `ms`, `s`, `m`, `h`, `d`, `w`), in each form that can
      write it (text; `pl.duration` with the unit's keyword; `timedelta`,
      which has no nanoseconds), against every temporal column kind and
      unit (`Datetime` ms/us/ns and a zone-aware one, `Date`, `Duration`
      ms/us/ns): 192 cases, each in exactly one of two tests and none
      skipped. Where the column can express the unit, the exact recursion
      on gaps of that unit; where a cap or a disorder threshold is finer
      than the column's step, a refusal naming the parameter, the value and
      the step; and a halflife finer than the step read at its own scale.

      *Other surfaces.* The command line takes the same text in TOML,
      tested end to end on a parquet file with a `Datetime(ns)` clock. The
      Arrow import path (task 86's unbuilt half) still has to read the
      clock's unit from the Arrow type when it is built.
- [x] 89. **README rewrite: ten sections in a hierarchy, requested 2026-09-23.**
      The user's brief: plan every section first; put similar concepts
      together, with a moderate number of large sections holding
      subsections, in the order that reads best; then take each section for
      clarity and brevity, preferring code with comments and keeping prose
      for what a comment cannot hold (a reason, a warning, a trade-off,
      math); tables over long bullet lists; every rule in `docs/WRITING.md`.

      *Measured before* (prose outside code and tables): 2,735 lines,
      13,941 words, 82 of 447 sentences at 45+ words, 17 cost words
      (`costs|pays|buys|for free|the price|the point`), 59 python blocks.

      *The map.* Seventeen top-level sections become ten, each holding the
      subsections listed; `←` says where moved content comes from.

    | section | subsections |
    |---|---|
    | Introduction | The idea · Four words · Install (← Install, and the Polars-version note from the top) · A first fit (the two examples) · What you can rely on (← Two guarantees, Mistakes are named, And the rest, as a table) |
    | How a bank sees a stream | What a spec names · Time and decay · A local fit along any feature · Convergence without a decay · Groups · Weights · Warm-up · Labels that arrive late (← Preparing a stream: a spec parameter every model shares) · Nulls, and three ways to hold a row back · Row order and the two guarantees |
    | Running a bank | As a query · In a loop · Output as Arrow (← a paragraph inside In a loop) · Outside a live Python process · Series that tick at their own times (← Preparing a stream) |
    | Saving, loading and serving | Save and load · Serving without learning · What a state file holds · Reading a state without this library |
    | Reading the fit | Coefficients · Output field names · The running sums behind a fit · One row per finished group · Reading a correlation matrix |
    | Diagnostics, selection and evaluation | Per-row diagnostics · Conformal intervals · Evaluating an output frame · Evaluating a stream too large to hold · Data whose truth is known |
    | Models | the model table, grouped by family · Linear models (`ewridge`, `rls`, `lasso`, `kalman`, `huber`/`quantile`, `sgd`, `pa`, `ftrl`, `holt`) · Moments and correlation (`ew_cov`, `marginal`, `deco`, `rcov`) · Clustering and classification (`kmeans`, `micro`, `ew_class`) · Sequential tests and regimes (`seqtest`, `corrchange`, `hmm`, `bocpd`); each model one level below its family |
    | Performance | Throughput · Memory: which calls stream · Tuning memory with Polars' own settings · Chunk size (← a paragraph inside Tuning memory) · Parallelism · Against scikit-learn (← its own top-level section) |
    | Scope and integrations | What this is not (its bullets → a table) · Pathway · Databases: DuckDB and ADBC |
    | Versions, testing and development | Versioning and the Polars pin (its four parts one level down) · Testing · Development · License |

      The Contents line becomes a table: one row per section, its
      subsections linked beside it.

      *Constraints held.* Every anchor `llms.txt` and `docs/RUNNER.md` link
      to keeps its heading text, since an anchor is the text and not the
      level. The model table keeps its header and one link per builder.
      Each model keeps its *API:* and *Rust:* lines. Every python block still
      runs in the namespace `tests/test_production_hardening.py` gives it.
      The models move one heading level down, under their family, so
      `test_api_links` and `test_model_registry` read them there, and
      `docs/EXTENDING.md` step 15 says so.

      *Rules applied to every section.* A parameter, a structure's fields or
      a call sequence becomes a code block with comments; prose stays only
      for a reason, a warning, a trade-off or math. A sweep of six or more
      becomes a table, and so does a bullet list that compares things. One
      idea per sentence. A term is defined before it is used. Nothing is
      said about "the bank" that is true of only some models. Detail cut
      from the README moves to the deep document that owns it, and never
      simply disappears.

      **Done 2026-09-23, to the map above.** Measured on paragraph
      boundaries, the fair count: a count that collapses whitespace merges
      text across headings, tables and code blocks, and reported 82 → 50
      long sentences, most of them artefacts.

    | measure | before | after |
    |---|---:|---:|
    | prose words | 13,496 | 11,590 |
    | sentences of 45+ words | 44 | 2, both two sentences the count merges |
    | sentences of 35+ words | 100 | 33 |
    | mean sentence, in words | 23.4 | 19.9 |
    | cost words | 16 | 0 |
    | tables, as rendered | 15 | 37 |
    | top-level sections | 17 | 10 |
    | python blocks, all running | 59 | 59 |

      Checked mechanically rather than by reading: every in-page link lands
      on a heading (107 written, resolved against the rendered ids); every
      anchor `llms.txt` and `docs/RUNNER.md` use survives; every table row
      has its header's cell count; and the only numbers that left the README
      are the command line's memory figures, which `docs/RUNNER.md` holds.
      Found and fixed on the way: the withheld reasons are now listed in the
      order `WITHHELD_REASONS` declares, which is their precedence; "row
      order matters only when a model forgets" was false of the models that
      step, filter or test; the introduction's link into the API reference
      for `lf.online.fit_predict` had been lost, and now sits under *As a
      query*.

      **Follow-up, 2026-09-23: the approach is now the rule.** The user's
      verdict on the result: "This is an improvement rewrite the writing and
      phrasing doc so that future writing is more like this". So
      `docs/WRITING.md` is rewritten in the style it describes, keeping
      §0–§6 where other files cite them: §2 plans the map before any
      sentence; §4 checks a fact against the code and lets a rewrite drop
      words but never add facts; §5 aims near 20 words a sentence; §6 is the
      pass as eight checked steps, with the counting traps this pass hit;
      §7 gives the table shapes that replaced bullet lists; §8 shows this
      rewrite's before and after. `docs/PHRASING.md` logs both messages
      verbatim, its first entry that confirms an approach rather than names
      a fault.

- [x] 90. **Every user document rewritten to `docs/WRITING.md`, requested
      2026-09-23.** The user's ask: "rewrite every user document that is not
      a work list, plan or history using the latest writing guide".

      *Which documents.* `docs/README.md` divides `docs/` into guides and
      records, so the guides are rewritten and the records are not, with the
      top-level documents a user reads beside them:

    | rewritten | not rewritten, and why |
    |---|---|
    | `docs/PERFORMANCE.md`, `docs/RELEASE-READINESS.md`, `docs/TESTING.md`, `docs/EXTENDING.md`, `docs/STATE-WORKFLOW.md`, `docs/RUNNER.md`, `docs/REGIMES.md`, `docs/OUTPUTS.md` (through `scripts/outputs_doc.py`, which writes it), `docs/README.md`, `CONTRIBUTING.md`, `SECURITY.md`, `llms.txt` | the records: `PLAN.md`, `ENHANCEMENTS.md`, `CLUSTERING.md`, `ANSWERS-E54-E64.md`, the reviews, `IMPROVEMENTS.md`, `SIMPLIFICATION.md`, `BEYOND-O-STATE.md`, `BOOSTED-TREES.md`, `MARGINAL-LAGS-AND-BINS.md`, `WARMUP-AND-CONVERGENCE.md` and `ARROW-SOURCES.md` (the last three a design, a design and a plan); `CHANGELOG.md`, history; `PHRASING.md`, a running log kept verbatim; `VALIDATION.md`, generated and never edited by hand; `CODE_OF_CONDUCT.md`, the Contributor Covenant as adopted; `README.md` and `WRITING.md`, rewritten to the guide in task 89 |

      *How.* One agent per document or pair, each first returning a map and
      the claims it found false or stale, with evidence from the code; the
      maps are recorded here before any prose changes (WRITING.md §6 step
      2). Each rewrite keeps every number, name, link, ID and code block, or
      says where it went; keeps every section number, step number, ID and
      heading text another file cites; corrects a stale claim only with the
      code's evidence; and is measured before and after.

      *The maps.*

    | document | reader | new structure ← what moves |
    |---|---|---|
    | `PERFORMANCE.md` | someone who hit a slower run or a bigger process, or a contributor measuring a hot-path change | the 21 numbered sections stay H2 with their numbers; a new *Reading this document* opens it (a contents table with a "read it when" column ← the "where to look" paragraph; *How the numbers are made* ← the machine and regenerate paragraph; *The headline* ← the status block, as a table; *Words this document uses*); §3's P1–P11 become a status table and one H3 each, their `<details>` plans kept; §5 gains an index of the rejections recorded in later sections; long sentences in §10–§13 and §19 become tables of their figures |
    | `RELEASE-READINESS.md` | whoever cuts a release, or asks which Polars versions are promised | five sections replace two parts of 22 H2s: *Cutting a release* (← the release gate and the rehearsal), *Which Polars versions are promised* (← the pin, 2.0.0rc1 measured, raising the ceiling, whose numbered steps stay), *Keeping the API stable* (← Part 2's API sections), *CI cost while the repo was private* (← the quota and cost sections), *Going public, as recorded* (← R1–R6 and the open-source preparation); cited headings keep their text |
    | `STATE-WORKFLOW.md` | a user about to rely on a state file across runs; second, whoever re-checks a guarantee on a new Polars | a guide section first, *The workflow in four steps* (← the four bold paragraphs, as a table, and a table of R1–R7 with the F and C evidence for each), then the research record of 2026-09-03 with §0–§9 one level down, numbers and headings unchanged |
    | `RUNNER.md` | someone running a bank with no live Python | four sections replace six: *Running it* (a first run; saving, resuming and scoring; a run whose product is its state; closed groups), *The configuration file* (a flag and key table; files; specs in TOML; clocks that are times), *Memory, threads and chunk size*, *From Rust, and the Polars it needs*; every executed `sh` line unchanged, since the tests run them |
    | `REGIMES.md` | someone about to trust `hmm`, `corrchange` or `bocpd` | §0–§7 keep their numbers; §0's six findings become bold-led paragraphs; each section states its conditions before its table; §7 becomes one commented code block and an experiment-to-section table |
    | `OUTPUTS.md` | someone who has found a model's record column and wants each field's meaning | a contents table by family; *Reading this reference* (the field-name parts; the four fields most models write, stated once; what is not listed; how the page is made); the 21 model sections and their anchors unchanged |
    | `docs/README.md` | someone looking for the one document that answers a question, or following a citation | *Guides* and *Records* as tables with family rows; the ten documents neither table listed get rows; the top-level documents get a table |
    | `TESTING.md` | someone deciding whether to trust the library, or choosing what to test next; or following a `T-` ID | five sections replace five unrelated ones: *What the suite proves* (← the scorecard and the backlog's hardening rows, as tables), *What it has found* (an index of every defect, linked to its entry), *Where it is thin, and what is left*, *How the suite looks for defects* (← the oracle rule, the mutation run, the FFI audit), *The entries, by ID* (the lettered tables A–E, each ID once, the backlog rows merged in); every T-ID and section letter kept |
    | `EXTENDING.md` | a contributor adding a model kind, following the steps in order | a checklist table up front (step, file, what you add, the check that fails if you skip it); the 16 steps as `### Step N` headings under their layer, numbers and subjects unchanged; long sentences in steps 1, 6 and 8 become tables of what each kind of model needs |
    | `CONTRIBUTING.md`, `SECURITY.md` | a developer who wants a change merged; someone reporting a vulnerability | same sections; the gate's steps become a numbered list, long sentences split, cited sentences verbatim |
    | `llms.txt` | a coding agent reading it in one pass | the llmstxt.org order: summary, details (the rules as bold-led bullets, a surface table), then file lists, one link per item |

      *Decisions taken on the maps.* Stale measurements are replaced by a
      fresh run of the repository's own script, the old figures kept as a
      dated note: `REGIMES.md` §2–§4, whose `corrchange` numbers the S3 fix
      of 2026-09-19 moved (a t5 size of .075 is now .045, and the power
      finding flips), and §1, which says what its streams really contain
      (one regime, so no switch was ever tested). A record keeps its date
      and gains a dated note rather than a silent edit. Figures credited to
      a surface that did not produce them go: `RUNNER.md`'s 0.95 and 1.41
      GB were the removed `po.run`'s. `scripts/regime_experiments.py`'s
      `epps` step, which stops `all` since a rename on 2026-09-07, is
      fixed, since `REGIMES.md` §7 tells the reader to run it.

      One mapping found a test gone rather than a sentence wrong: `kalman`
      held to river's `BayesianLinearRegression` (`TESTING.md` T-R2, and
      E19) was deleted in `509c6cf` with nothing in its message, and nothing
      has checked the match since. It passes unchanged today, so it is
      restored to `tests/test_river.py` as `23671a8` wrote it, and the
      claim is true again rather than rewritten.

      *Follow-up, not in this task.* `REGIMES.md` §1's `hmm` recovery ran on
      streams that never switch regime; rerun it on streams that do.

      **Done 2026-09-24, to the maps above.** Measured the same way as task
      89, with the splitter that also ends a sentence inside bold; every file
      checked independently for lost numbers, names, links and IDs, table
      cell counts, in-page links and wrapped code spans:

    | document | 45+-word sentences | 35+-word sentences | mean sentence |
    |---|---:|---:|---:|
    | `PERFORMANCE.md` | 107 → 0 | 193 → 0 | 27.3 → 16.6 |
    | `RELEASE-READINESS.md` | 17 → 0 | 45 → 2, both merges | 19.9 → 15.4 |
    | `TESTING.md` | 9 → 0 | 16 → 0 | 21.2 → 13.5 |
    | `EXTENDING.md` | 20 → 1, task 88's paragraph kept whole | 39 → 1 | 28.1 → 14.8 |
    | `STATE-WORKFLOW.md` | 11 → 0 | 23 → 0 | 22.3 → 16.1 |
    | `RUNNER.md` | 0 → 0 | 5 → 0 | 22.3 → 16.7 |
    | `REGIMES.md` | 3 → 0 | 15 → 0 | 21.3 → 17.1 |
    | `OUTPUTS.md` | 0 → 0 | 1 → 0 | 21.1 → 15.9 |
    | `docs/README.md`, `CONTRIBUTING.md`, `SECURITY.md`, `llms.txt` | 2 → 0 | 4 → 0 | 13–22 → 11–13 |

      Every in-page link in the twelve resolves, no code span is wrapped,
      and every table row has its header's cell count. What the loss check
      reports gone is either a correction (`covariance="diag"`, `__hl`,
      `[save_state]`, `online_run`, `finish_label`, `tests/test_lasso.py`,
      the two `po.run` figures, `REGIMES.md` as the home of three defaults,
      `deco` and `rcov` as measured there) or a fragment of an old wrapped
      span; every command and name it lists besides moved into a code block
      or a link. The means sit below the guide's aim of about 20. Bold
      labels, counted as sentences, pull them down, but even without them
      the prose runs near 15 to 16 words, a little choppier than the
      README's 19.8: the long sentences were split and not merged back.

      Found on the way and fixed outside the twelve: the `hmm` docstring and
      `tests/test_corrchange.py`'s notes, which carried the old REGIMES
      numbers; the experiment script's printed conclusions; the README's
      save-and-load paragraph (a bank object has `save`/`load`), its `coef`
      cadence, its "four hundred" chunks (the test runs 7 and 100) and its
      chunk-size figure (`PERFORMANCE.md` §20's); the API reference's front
      page, which still offered the removed expression form; the CLI's
      `--no-output` help; the release and canary workflows' comments, still
      on `<2`; `MIN_SCHEMA_VERSION`'s missing entry for 13; four code
      comments citing the wrong P-ID or section; and C1 and C2 swapped in
      `tests/test_frame.py`. A new test holds `RUNNER.md`'s example config
      to the file its shell blocks run against.
- [x] 91. **Every significant Python, to the newest, requested 2026-09-24.**
      The user's words: "We want to support all significant python versions
      up to the latest." Significant is read as SPEC 0 reads it, the
      versions released in the last three years: today 3.12, 3.13 and 3.14,
      which is the floor this package already had. One `abi3-py312` wheel
      per platform installs on all three. But CI never asked for a version:
      each runner's `uv sync` took whatever interpreter it had, so the macOS
      leg ran 3.14 by accident while the classifiers stopped at 3.13, and a
      test that parsed docstrings by their 3.12 indentation failed there
      alone (`c2462f6`). Now `pyproject.toml` declares 3.12 to 3.14, and CI
      picks each leg's interpreter with `UV_PYTHON`. It runs every declared
      version on Linux, and the floor and the newest on macOS and Windows,
      since a Python version rarely behaves differently by OS, or an OS by
      Python version. The API reference is built on one leg, since its
      Pages artifact can be uploaded once a run.
      `tests/test_ci_cost_policy.py::TestPythonVersions` holds the matrix to
      the classifiers, so a declared version cannot go untested; six faults
      injected into the workflow and the metadata each fail it. The whole
      suite passed locally on CPython 3.13.15 (3,055 tests) and 3.14.6
      (3,059, the new policy tests among them). Not done,
      each a decision for later: a floor below 3.12 (3.11 needs
      `abi3-py311`, a new wheel tag and a syntax pass under ruff's py311
      target; 3.10 also `typing_extensions` for `Unpack`); an advisory leg
      on the next pre-release (3.15, due October 2026); and free-threaded
      builds, which an `abi3` wheel cannot serve.
- [x] 92. **Mutation testing in CI, a release comparison, property tests
      and oracles, requested 2026-09-24.** The user asked whether more
      mutation testing would catch subtle bugs. The analysis found that this
      session's subtle bugs came from oracles, review, experiment and a CI
      leg's Python, while mutation testing found weak tests (48 survivors in
      `stats.rs`) and no bugs. The user asked for all three recommendations,
      and for the README's row-order section to say how a bank detects rows
      out of order.
      - *Mutation testing* (`.github/workflows/mutants.yml`). The changed
        lines of every push and pull request, failing on a survivor that
        `scripts/mutants_equivalent.toml` does not name; all of
        `online-core` weekly in sixteen shards of up to two hours, reported
        through `scripts/mutants_report.py`, on its schedule only while the
        repo is public. The scope, `online-core` and `span.rs`, was
        measured: `span.rs`'s own Rust tests caught 55 of 60 viable mutants,
        and its 4 real survivors now have tests; a one-in-ten sample of
        `spec.rs` left 16 of 39 viable mutants, pytest's ground, out of
        scope. The equivalents list matches by file, function, mutation and
        code on the line, so an entry follows its line and lapses when the
        line changes.
      - *The release comparison* (`scripts/compare_release.py`,
        `scripts/release_probe.py`): every output of 30 specs over 400 rows
        against a PyPI release, bit for bit. The first step of a release
        (docs/RELEASE-READINESS.md, which now lists the steps), and a report
        on every CI push. Measured: identical to 0.10.0 and 0.9.1; 112
        fields differ from 0.8.0, all from 0.9.0's declared gates.
      - *Property tests* (`tests/test_properties_temporal.py`, 11 properties)
        on duration text and temporal clocks. They found four bugs, all
        fixed with the tests pinning them: a space after the sign
        (`"+ 5m"`) was accepted and named a grid's field; a `pl.duration`
        past 292 years wrapped (585 years stored as `384ns`), because
        `dt.total_nanoseconds()` wraps; a `timedelta` that long was an
        unnamed `OverflowError`; and `label_delay` broke hard rule 3, since
        the held rows' clock was a running sum within a chunk and a fresh
        sum at each chunk's start, which round differently, so
        `settled_frac` depended on the chunking. The stream now keeps that
        clock per instance, in the state: schema 15, and a schema-14 file
        loads with it rebuilt as 14 did. Single-chunk outputs did not move
        (identical to 0.10.0, bit for bit).
      - *Oracles* for 40 model paths no independent reference held
        (`tests/reference_paths.py` and five `test_oracles_*` files, 59
        tests): the lasso's targets, gaps, windows, selection and
        intercept; `ewridge`'s grids, windows, sessions and schedule;
        `rls` and `kalman` with several targets; `ftrl`, `pa`, `sgd`,
        `holt`; the plain sigma. Each is written from the objective or the
        docstring's equations, with seeded bugs failing it. They found four
        disagreements, left open as tasks 94 to 97, and `sgd`'s docstring
        giving the intercept an `l2` it does not get, now corrected.
- [x] 93. **The PyPI page's links to other files, reported 2026-09-24.** PyPI
      shows the README, where a relative link resolves against pypi.org and
      is a 404; in-page links worked. `scripts/pypi_readme.py` rewrites the
      76 relative links to GitHub at the release's tag, run by
      `release.yml` before maturin in the `sdist` and every `build` job;
      the README in the repository keeps its relative links. A published
      description never changes, so 0.10.0's page stays as it was.
- [x] 94. **A windowed Gram's zero variance comes back as rounding noise.**
      Found by the task 92 oracles. The drop rule is "variance exactly
      zero" (`variance_is_usable`, since T-E9), but a window's Gram is
      rebuilt by subtraction, so a feature constant over the window reads
      about 1e-14 and is kept. A windowed lasso at λ = 0 whose window holds
      one row of a sparse target then divides noise by noise: predictions
      of 8.6e33 and 2.8e55 where the fit is −1.0. It takes a lowered gate
      (`min_periods=0.5`); at the default the rows are withheld, λ = 0.1 is
      sane, and `ewridge` is withheld by its noise gate. The fix is a
      noise-aware zero for a subtracted Gram, since T-E9 removed the
      relative threshold for good reasons.
      **Done 2026-09-24, without a threshold.** Measured first: the
      remainder a constant feature leaves is of the order of machine epsilon
      times its level times the spread it had, and grows with the rows
      since the boundary. At a level of 1e8 and a spread of 1e-3 it was
      1.2e-5 of the terms that cancelled for one row, and 0.46 at 100,000
      rows, so no threshold told it from a real spread; a spread of 1e-5 at
      a level of 1e6 came back within a factor of 20. Instead every `EwCov`
      keeps, per feature, the value it has held since it last changed and
      the weight of those rows (schema 16). Where that weight is all the
      window's, to `EMPTY_FRACTION` of the live weight, `truncated` sets the
      feature's variance and covariances to exactly zero and its mean to
      the value. `gaps` then zeroes that slot's cross-moment with each
      target (Cauchy-Schwarz), so an unstandardized ridge gives a held
      feature a slope of exactly 0, where it gave 0.0057 at a level of 1e6.
      A row of weight 0 ends no run, a blend with the slow twin forgets the
      runs, and a state from before schema 16 starts them at its next row.
      The tests: `window::tests` over every level, spread, halflife and
      window length measured, a window of a thousand halflives among them;
      `the_runs_follow_the_rows_blocked_or_not`;
      `a_feature_held_over_the_window_gets_no_slope`. The oracle
      references drop a feature that holds one value on every row of its
      Gram, and hold the windowed problems they had left out.
      `TestWindow::test_a_window_down_to_one_row_of_a_target_fits_its_intercept_alone`
      fails on 0.10.0. Under cargo-mutants, the changed lines of tasks
      94-97 give 99 mutants: 4 unviable and the rest caught. The last
      survivors needed tests with a second target and a third feature.
      **Review 2026-09-25.** The run's weight against the window's, two
      numbers equal in exact arithmetic, drifted apart under a long window
      and a long halflife -- the window's `w_now − f·old.w` takes one `exp2`
      over the window's clock where the run's weight is a product of per-row
      factors -- by 1.6e-12 of the live weight at a halflife of 1e6 with
      fifty thousand rows of history and a window of as many, past the
      1e-12 the rule allowed, and the held feature read as moving again
      (`a_long_window_under_a_long_halflife_still_reads_a_held_feature`
      failed on the old rule). Runs now keep the learned-row index that
      started each run, `EwCov` and `marginal` count learned rows, the
      window's snapshot records the count, and a slot is held over the
      window when its run started at or before the first learned row inside
      it: row indices do not drift. A row whose weight is nothing next to
      the accumulator's (`EMPTY_FRACTION` of it, the window's own notion of
      nothing) is neither counted nor tracked, which keeps the boundary
      cases of no account as they were. The runs' ageing went with the
      weights. Also found: the zero reaches `ew_cov`'s and `ewclass`'s own
      windows (a held feature reads variance 0 and a null correlation, where
      it read noise; `ew_covs_window_reads_a_held_feature_as_no_spread`),
      and an unstandardized solve with no ridge meets an exactly singular
      Gram on a held feature, which the solver's jitter takes with the
      feature's slope exactly 0
      (`a_held_feature_under_a_window_leaves_an_unregularized_solve_finite`).
- [x] 95. **The lasso's `lam_selected` under a `window` departs from its
      documentation** ("the selection error is truncated with the sums"):
      14 to 19 rows of ~277 at the default `select_halflife`, 88 to 90 at
      `select_halflife=10`. Three departures in `lasso.rs`, each moving rows
      on its own: the snapshot's selection part is taken after the row's own
      error and its weight is aged twice; `window_sel_err` reads the ring as
      it stood at the previous row; and the truncation uses the model's
      halflife, not `select_halflife`. An emulation with the three corrected
      lands on the oracle on every row.
      **Done 2026-09-24,** and a Rust oracle found a fourth departure: on a
      row that does not score the target the choice stood from the last
      scored row, while the window moved on. Now the window moves to the
      row first, its snapshot takes the selection before the row's own
      error, aged once by the selection's decay, the truncation ages by the
      same decay, and the choice is made again on every row. Where nothing
      has aged out of the errors it reads the live ones, to the bit.
      `lam_selected_under_a_window_is_the_argmin_inside_it` recomputes the
      choice from each row's reported predictions. The oracle file holds
      `lam_selected` under a window, with and without a halflife of its own,
      and that fails on 0.10.0 at eight rows.
      **Review 2026-09-25.** A window with no scored row of a target fell
      back on the whole-history errors and chose from them every row, which
      the docstring said it would not; the choice now stands until the
      window has a scored row again (`an_empty_selection_window_keeps_the_choice`).
      A state saved by 0.10.0 mid-window carries selection snapshots in the
      old meaning (after the fold, aged twice) and is off for one window
      after resuming; pre-1.0 compatibility is waived.
- [x] 96. **`lam_selected` under a `min_periods` list.** Each target's
      selection error folds its predictions on rows that its own threshold
      still withholds, which is review S2's "a gate on the output, not on
      the model", but E7 says a target not yet ready is withheld "before it
      can reach … selection". The code and the two documents need one
      reading; the user's call.
      **Done 2026-09-24: the code is kept,** the user's call. A target's
      selection folds the model's own predictions for it from the model's
      first prediction, whatever its own threshold withholds, as learning
      does (S2). E7 and the lasso builder's `select_halflife` entry now say
      so, and the oracle file holds `lam_selected` with a `min_periods` list.
- [x] 97. **Two conventions to settle.** `holt` reports `coef` as `[0, 0]`
      before a target's first observation, where `docs/OUTPUTS.md` says
      `coef` is null "before the model has anything to report"; and the
      test reference `kalman_ref` builds its standardized `coef` from the
      scales before the row and the means after it (off by up to 0.38), so
      no test compares it, while the library's `coef` predicts the next row
      to 1e-16.
      **Done 2026-09-24,** as recommended and the user asked. `holt`'s
      `coef` is null for a target before its first observation (NaN from
      `coefficients`, which the stream writes as null), and its oracle holds
      `coef` on every row. `kalman_ref` reads its standardized `coef` with
      the scales and means of one moment, after the row, and
      `tests/test_oracles_rls_kalman_paths.py` compares kalman's `coef`.
- [x] 98. **The Rust tests link no Python, 2026-09-24.** Task 91's matrix
      failed on its first push. The Linux legs for 3.13 and 3.14 could not
      start the CLI's tests: "libpython3.13.so.1.0: cannot open shared
      object file", exit code 127. `cargo test --workspace` built online-py
      into the same graph, where pyo3-polars turns on polars-error's
      `python` feature, so every test binary that links polars linked
      libpython too. The runner's own 3.12 keeps its libpython on the
      loader's path, and uv's 3.13 and 3.14 do not. macOS passed because its
      libpython has an absolute install name. `cargo test` now leaves
      online-py out in CI, the canary, the gate and the docs that give the
      command. It has no Rust tests, pytest covers it through the extension,
      and the CLI's tests now run the CLI as it ships, with no Python in it.
      `tests/test_ci_cost_policy.py::TestTheRustTestsLinkNoPython` holds
      every workflow and the gate to it, and asks cargo whether pyo3-ffi is
      in that graph.
- [x] 99. **pyarrow, tested as a reader and as a source, 2026-09-24.** The
      user: "packages that we do not want to depend on are fine if needed
      to test with other libraries." The README, CLAUDE.md, a docstring, the
      type stub and a Rust doc comment said pyarrow reads a bank's Arrow
      output. `docs/ARROW-SOURCES.md` said that half stayed unverified,
      since pyarrow "must not become [a dependency] to test a sentence".
      pyarrow 25.0.1 is now in the dev group, and
      `tests/test_pyarrow_interop.py` measures it. `pa.array`,
      `pa.chunked_array`, `pa.record_batch` and `pa.table` each take an
      `ArrowStruct` as it is, with `fit_predict`'s values and dtypes, and a
      struct pyarrow has read is spent. A `RecordBatchReader` streams into
      both streaming paths with the whole frame's numbers, and a reused
      reader plan warns. The package keeps its single dependency:
      `tests/conftest.py` makes pyarrow unimportable in the rest of the
      suite, which so runs as a user without it does, and the interop tests
      run pyarrow in child interpreters. The same check corrected three
      places that said duckdb reads the struct directly. duckdb 1.5.5
      refuses it and takes it through `pl.Series`, as the README said.
- [x] 100. **Libraries the package does not depend on, allowed in tests,
      2026-09-24.** The user: "Be sure that the future test plan allows non
      dependent libs for testing." §9's test plan now says so, and
      `docs/TESTING.md`, "Libraries the package does not depend on", gives
      four rules. Declare the library in the dev group by name, and import
      it plainly. Keep it out of the package's reach when its presence
      changes other libraries, as pyarrow's does. Never add it to the
      package's dependencies. `CLAUDE.md` points there.
      `tests/test_dependency_policy.py` checks the first, second and fourth
      rules from the tests' own imports. Two faults injected each fail it:
      pandas taken out of the dev group, and an `importorskip` put back. It
      found pandas and scipy reaching the tests only through statsmodels,
      and both are declared now. The 34 `importorskip` calls became plain
      imports: they never skipped where the suite runs, and could only have
      hidden a broken environment. Not taken: scikit-learn in
      `tests/test_sgd.py`, and Pathway, whose licence keeps it out of
      `pyproject.toml` until the user decides. Reading the first Windows
      run's skips found one more that was not a platform guard:
      `test_the_cli_writes_the_sidecar` looked for `target/<profile>/online`,
      which has an `.exe` there, and its TOML carried the path's
      backslashes as escapes (T-W3b's trap). It now takes the `online_cli`
      fixture and writes POSIX paths, so it runs on Windows too.
- [x] 101. **A feature that stops moving leaves a rounding artifact outside a
      window, and the standardizing models read it.** Measured 2026-09-24:
      one feature of three holds one value after 300 rows, halflife 20 rows.
      Its running mean approaches the held value until the step rounds to
      nothing, and stops `1/(2b)` rounding steps short. The variance and the
      cross-moment with the target then settle on that gap instead of
      decaying together. The lasso's slope on the feature keeps the 0.47 it
      learned until the stall and degenerates after it: -79 at 50 halflives
      at a level of 1000, -6.4e9 at 100, and -4.7e3 at 40 at 1e8. `ewridge`
      and `huber` with `standardize` do the same, and `kalman` and `sgd`
      predict differently at each level, by up to 0.12. A held target does
      it too: slopes stuck at 2e-8 at 1e8, a target variance on the gap's
      square, `marginal`'s correlation at -0.06 where it goes to zero, and
      its bins' split gain at 0.45.
      **Built: every running mean is a pair, 2026-09-24**
      (`crates/online-core/src/comp.rs`). A mean is `hi + lo`, `hi` the
      double and `lo` what it leaves out. A row's deviation is
      `(x − hi) − lo`, and the step `b·d` goes into the pair by Kahan's
      compensated sum, so no part of it is dropped. Where a plain step would
      round to nothing, `lo` keeps it; the deviation keeps its precision as
      it shrinks, and the mean follows exact arithmetic to the value. Nothing
      needs to know that a value is held. The pairs are in `EwCov` and its
      blocked flush, `EwDiag` and its `including` view (which now hands back
      the deviation), `TargetMoments`, the gaps accumulator's feature and
      target means, `robust`'s target means, `marginal`'s pairs, lags and
      bins, `EwAutoCorr`, `deco`'s level and `bocpd`'s runs. `EwLagCov`,
      `ewclass` and `hmm` take their deviations from the pairs. A row of
      weight 0 takes no step: Kahan's sum does not keep `lo` under half a
      rounding step of `hi`, so adding zero could round the pair afresh, and
      `label_delay`'s doubled-stream oracle caught it.
      **Two designs were built first and replaced.** Snapping the mean to
      the held value at the stall, the fix this entry first proposed, jumped
      where exact arithmetic still carries the gap. With no decay that gap
      closes only as `1/n`, and the snap moved `sgd` 8.2e-3 at 1e8 and
      6.9e-2 at 1e12. Stepping the gap itself toward a value detected as
      held, `g' = (1 − b)·g`, followed exact arithmetic but needed every mean
      to know which values were held, and a rule wherever a mean moves by
      other means. A row of weight 0 found a slot held while carrying another
      value, and moved the mean to that value less the gap: the ridge then
      predicted -0.02 where it predicts 2.41. Blends, `robust`'s score steps
      and the `including` view each needed a rule of their own, and a blocked
      Gram's slope on a held feature still wandered from -0.45 to 1.71 at a
      halflife of 5 rows. The user asked whether the idea was as solid as it
      seemed. The pair was prototyped against it and measured better on
      every count below.
      **Measured**, as the worst relative prediction difference between the
      fit at a level and the same fit at 0.5, pair against the gap design:
      `sgd` 1.7e-9 at 1e8 and 3.2e-6 at 1e12 from the stop (2.3e-5 and
      1.7e-3). The regressions stay within one rounding step of the level.
      With no decay at 1e12, `sgd` falls to 4.5e-10 by row 300,000, where
      the gap design grew to 1.7e-3, and the ridge's slope stays 0.50–0.52
      where a plain mean took it to 0.04. A blocked Gram's slope stays inside
      the row-by-row range at every block size, 0.33–0.65 at a halflife of
      5 rows in blocks of 16. A held target's slopes decay to 1.35e-45 at
      every level, 2^-150 of 2.
      **Cost.** Outputs change in their last bits: 19 of the release
      comparison's 30 specs, by a median of 3e-16 of the value and at most
      1.4e-14 (0.11.0 declares it). Seven Python oracles that replayed the
      plain recursion to the bit replay the pair. Interleaved with HEAD,
      `EwCov::update` is unchanged at 64 slots; `EwDiag::update` takes 6.0,
      10.7 and 24 ns a row at 4, 16 and 64 slots, against 5.2, 6.3 and 15.5;
      and `sgd`'s step 57 ns at 16 features, against 49. `ewridge`'s step is
      unchanged, its solve dominating. *2026-09-27: not so at scale.* The benchmark,
      0.11.1 against 0.10.0, read `ew_cov(lags=[1..5])` at 0.53 and a
      10-target `ewridge` at 0.64; a bisect put the first and 15 % of the
      second on this task (the pair's deviation inside the lagged
      co-moments' `k * k` loop) and the rest on task 132's own means, whose
      low parts' length was checked once a feature. Both now take each
      deviation once, bit for bit the same, and are back at 1.07 and 0.94 of
      0.10.0 (PERFORMANCE §26).
      **Found by the sweep of every running mean, and fixed here:** a
      windowed `marginal` pair did not get task 94's zero spread for a slot
      held over the window, and `beta` divided a remainder by itself.
      `marginal` now keeps runs for its window, in the `Runs` that `EwCov`'s
      task-94 runs moved into. **Found and left:** Page-Hinkley's mean feeds
      only a sum that `delta` swamps; `holt` centres nothing; the uncentred
      EW means of squared residuals have no gap to settle on; `ColumnStats`
      in the data summary is report-only. The clusters are task 102, and
      `corrchange`'s span variance task 103.
      **Tests.** `tests/held_values.rs` holds every centring model for 150
      halflives after a feature or target stops, at levels from 0 to 1e12
      and with no decay. Twelve of its fourteen tests fail with the means
      plain; the other two are contracts that hold of any design: a row of
      weight 0 changes nothing, and a state saved mid-hold resumes to the
      bit (which fails with the low parts left out of the state).
      `comp::tests` pins the pair, and each accumulator has a test beside it.
      `cargo mutants` over the change: 402 mutants, 393 caught, 6 unviable,
      3 equivalent (`scripts/mutants_equivalent.toml`: a term that is 0
      either way on a row of weight 0, a product against 0 that a quotient
      matches, and a guard at exactly 0 that takes 0 away), none missed;
      the 20 that timed out beside a running gate were re-tested alone and
      caught. Each survivor's test was applied to the mutant by hand first,
      and three did not kill it until rewritten: a residue test whose unit
      weights made the block mean correctly rounded, and a variance read
      through `var`, whose `max(0)` takes a NaN for 0.
      **Review 2026-09-25.** The pair is the right fix and every site steps
      and reads it; three deviations still read `hi` alone (`deco`'s and
      `corrchange`'s standardized residuals, `ew_cov`'s PCA score) and now
      read the pair; the window's subtraction `m_u − m` reads `hi` alone,
      bounded at `ratio·g·ulp(level)/σ` of the windowed variance (2e-5 at
      1e12 with unit spread over three halflives), and the held slot is the
      runs' job, so it is left. Schema 16 changed twice before release
      (`runs`, the `*_lo` fields) without a bump: pre-1.0 compatibility is
      waived, a bank file from the intermediate build loads with its runs
      restarted. Pinned since: `EwDiag`'s own skip on the fixture, a
      windowed hold (the slope stands while the window has spread, then the
      feature is dropped as one without spread), a hold that ends, a blocked
      ridge at 1e12 with rows of no weight carrying another value. Left:
      the classifiers and `no_weight` for `bocpd`, `deco`, `hmm`.
- [x] 102. **A stopped feature in the clusters' metric.** Found by task
      101's sweep. `kmeans` and `micro` standardize distances by `1/var`
      from `FeatureMoments` (`cluster/summary.rs`), refreshed every row,
      with `standardize` on by default. A feature that moves and then holds
      stalls there too: its variance settles near the mean's gap squared,
      `g_m²`, and a row's term against its own centre, which has its own gap
      `g_c`, becomes `(g_c/g_m)²`. The entry first put that anywhere from 1
      to 10³; it is `(n_c/N)²`, at most 1 and `1/k²` for equal clusters,
      since each gap is a rounding step over that mean's own step size `b`.
      **The problem is the metric's before it is rounding's.** `1/var` is
      unbounded as the recent variance goes to zero, and under decay it does
      so within ten halflives of any feature going quiet -- a flag that
      stops firing, a sensor at rest, a market closed -- not only of one
      that stops for good. A centre that receives rows follows the held
      value at the rate the variance decays, so its own term fades; a centre
      that receives none is infinitely far for good; and on the row where
      the feature moves again every centre is infinitely far: the argmin
      cancels what every centre shares, but the radius does not, and a
      structural move or an admission test that reads it goes wrong. The
      `else 1` at exactly zero variance is a discontinuity: raw units on one
      row, `1/ε` on the next. **Measured 2026-09-24** on the prototype
      (`scripts/clustering_experiments.py`'s streams, 20 000 rows, k = 5, p
      = 4), with the metric's variance floored at a fraction of the
      feature's long-run (undecayed) variance, `v_i = max(var_i, scale_floor
      · var_i^∞)`: at 0.01 and 0.1 the floor binds on no row of the static,
      drifting, feature-scaled-by-100, noise-grows-tenfold or feature-stops
      streams, so every number there is unchanged; on a feature quiet for
      twenty halflives (rows 8 000-14 000 at halflife 300) it binds on 3
      600-4 600 rows, and after the feature moves again `kmeans` split-merge
      recovers to ARI 0.731 in place of 0.220 (purity 0.804 for 0.487; 0.583
      at a floor of 0.01), `micro` unchanged at 0.786, plain `kmeans` 1.000
      either way. At 1 the floor is a different metric, binding on half the
      rows of every stream: 0.626 for 0.665 on the scaled stream's first
      quarter and the same elsewhere. **A feature that stops for good costs
      `kmeans` 0.24 ARI for a halflife after the stop at every floor**
      (0.756 against 0.999 with the feature dropped): the stale spread keeps
      penalizing every centre the held value is far from, until the centres
      and the variance have followed it, which is the model's memory at
      work, the same lag as a level shift, and not something a metric floor
      reaches. **Design, from the measurement:** `scale_floor`, default 0.1,
      0 for the metric as it was, on `kmeans` and `micro`; `FeatureMoments`
      keeps a reference beside the EW moments, and its means become pairs
      under task 101's rule. The centres stay plain: with the floor, a plain
      centre's gap costs `g_c²/(scale_floor · var_long)` in a distance, 1e-9
      at a level of 1e8 and 0.04 at 1e12 with unit spread. **The reference
      is not the undecayed variance.** That was built first, and
      `model_contract`'s recovery caught it: a row at the input bound put
      1e200 into a reference that never forgets, and the metric was lost for
      good, where the EW variance takes a thousand halflives to forget such
      a row and the contract gives it fifteen hundred. No reference that
      decays slowly enough to hold through a quiet spell forgets an extreme
      in time, so the reference must never take one: it is the same Welford
      variance at eight times the halflife, each row's weight clipped at the
      reference's own (a row takes at most half of it) and its deviation at
      ten standard deviations, started from the medians of the first five
      learned rows, which up to two rows at the bound among them do not
      move. A row at the bound then moves it by a factor of twenty-six at
      most, which a few of its halflives undo. The cost of the slow
      reference is that a hundredfold drop in a feature's variance (tenfold
      in its spread) is under-weighted by up to tenfold for some
      twenty-seven halflives, a hundredfold drop in spread by up to two
      hundredfold for eighty; the cost of the clip is that a feature without
      spread has no scale to clip against and takes its first move as it
      comes. **Built 2026-09-25.** `scale_floor` on `kmeans` and `micro`
      (default 0.1; 0 is the metric as it was), validated finite and ≥ 0 in
      the cfgs and the spec; `FeatureMoments` keeps `w_long`, `mean_long`,
      `var_long` and the start rows beside the EW moments, both means as
      pairs, `decay` takes the reference's factor (`decay.factor(d_clock /
      LONG_HALFLIVES)`), and `metric` takes the floor; a state without them
      loads and starts the reference at the next rows. The Python reference
      mirrors it bit for bit at every floor, the prototype carries it, and
      CLUSTERING.md §6.1 and §10 record the measurement, re-run on the final
      reference. **Tests**:
      `summary::tests::the_metric_is_floored_at_a_fraction_of_the_long_run_variance`,
      `the_reference_starts_from_the_medians_of_the_first_five_rows`,
      `a_row_at_the_bound_moves_the_reference_by_a_bounded_factor` and
      `a_state_without_the_reference_starts_it`; `kmeans` and `micro` each
      give a stream whose third feature moves and then holds at a level of
      1e8 the labels and distances (to 1e-6) they give it at a level of 0
      (`a_stopped_feature_at_a_level_leaves_the_assignments_alone`,
      `..._summaries_alone`); `model_contract`'s recovery from the bound,
      which the first design failed; `tests/test_kmeans.py::TestScaleFloor`
      holds the bank to the oracle at floors 0, 0.1 and 1 on a feature quiet
      for thirty halflives and shows the floor binding only in the spell; a
      negative floor is refused by name in both builders. **Review
      2026-09-25.** Four findings, all confirmed and fixed. The cfg's
      `scale_floor` had no serde default, so every earlier `kmeans` or
      `micro` state file was refused on load (hard rule 5); it has one, 0,
      the metric the state had, and a whole-state test strips the key and
      restores. The docs claimed a bound the mechanism does not give: the
      reference decays through a quiet spell, so the weight grows as
      `2^(Q/8)/scale_floor` over `Q` halflives of quiet, 58 at twenty, where
      `1/var` alone gave `2^Q`; every doc now says so. The reference's start
      weight was a row count, which broke the weight-scale invariance every
      other moment has (the same stream at weights 1 and 1e-3 read metrics
      2× apart through a quiet spell); it is five times the median of the
      start rows' weights, and `the_reference_scales_with_the_weights` holds
      the two streams to 1e-9. A feature without spread took its first move
      as it came, and a first move a million away put 1e11 into its
      reference for two hundred halflives; the reference is now per feature
      and a first move from no spread starts it over, that row the first of
      the next five
      (`a_feature_without_spread_starts_its_reference_over_on_its_first_move`).
      Measured, the cost the review predicted for a level shift in a
      feature: at halflife 100 with ten thousand rows after the shift, the
      floor binds on no row for a shift of 10 or 30 standard deviations (the
      clipped mean step follows a shift at five a row) and on a third of the
      rows for 100, where the shift itself takes `kmeans` to ARI 0.21 with
      the floor or without. Nits taken: the bound row's factor is `1 +
      CLIP/4` = 26, not fifty; `total_cmp` for the medians; a reference
      whose weight has decayed to nothing takes the next row whole and
      starts over on the one after
      (`a_gap_past_the_references_weight_starts_it_over`); a state saved
      mid-start resumes to the bit; a state missing any one field of the
      reference starts it over; the core cfgs refuse a bad floor by name;
      the reference's decay factor at the call sites is pinned against a
      `FeatureMoments` driven by hand
      (`the_reference_decays_at_eight_halflives`).
- [x] 103. **`corrchange` takes a span's variance as `E[x²] − E[x]²`.**
      Found by task 101's sweep. `long_run_sd`, the delta-method standard
      deviation of a span's correlation, forms `σ_x² = E[x²] − E[x]²` from
      raw moments, and its centred series `x² − E[x²]` cancels against
      `2·m_x·(x − m_x)` through `D₂`; both lose the level's digits. At a
      level of 1e8, `x²` is 1e16 with a rounding step of 2, so a unit
      variance is noise. **Measured 2026-09-24**, on eighty rows whose
      deviations are multiples of 2⁻²⁰ so that adding a level is exact: `D̂`
      is off by 5e-10 of itself at a level of 1e3, by 1.2e-5 at 1e5, and NaN
      at 1e8, where the variance cancels to a non-positive number and the
      monitor flags nothing. **Design: centre and scale the span first.**
      `ρ̂` is shift-invariant and so is the delta-method variance of it: the
      shifted moments are an affine image of the raw ones and the Jacobians
      cancel, so the same number comes out of better-conditioned inputs.
      With the span centred at its means and scaled by its standard
      deviations, `m_x = m_y = 0` and both variances are 1, so `f = D₃D₂ =
      (−ρ̂/2, −ρ̂/2, 0, 0, 1)` and `D̂²` is the Bartlett long-run variance
      of the one series `ξ_t = x̃_t ỹ_t − (ρ̂/2)(x̃_t² + ỹ_t²)`: Wied,
      Krämer & Dehling's own form, at 1/25 of the 5×5 kernel's cost.
      `corr_of` already centres. **Built 2026-09-24.** The span's means are
      pairs (`crate::comp`): a second pass over the exact differences from
      the first pass's mean picks up what its division rounded away, since
      that rounding is a shift of the whole span, which `ρ̂` cancels to
      second order but a first-order term in `ξ` carries, 1e-10 of `D̂` at a
      level of 1e8. The kernel loop is one series, not five squared.
      **Tests**: level invariance to 1e-12 at 1e3, 1e5 and 1e8 on deviations
      that are multiples of 2⁻²⁰, so the shift is exact (it measured the raw
      form first); the units test and the paper's null, size and power as
      they were; the definition test's longhand stays the five-moment form,
      so it now checks the identity between the two to 1e-12. **Review
      2026-09-25.** The accurate `D̂` reaches its true 0 on a collinear pair
      (`y = 2x + 3`), where the raw-moment form's noise stood in for it, and
      the numerator's rounding divided by it flagged every span (29 of 30
      seeds in the reviewer's replica; a derived or duplicated column in a
      bank would flag every span); a pair within 64 ε of `|ρ̂| = 1` now has
      no verdict, as a constant column has none
      (`a_collinear_pair_has_no_verdict`, which fails on the guard removed).
      Pinned since: T = 3 and 4, the bandwidth override, a constant column
      with an inexact mean, levels 4e9 and 1e12 (deviations on a 2⁻¹² grid),
      the level test to 1e-14 and the statistic itself at a level;
      `bandwidth` at `usize::MAX` no longer overflows. A span with one far
      outlier now gets the verdict the formula gives, a flag, where the old
      form cancelled to nothing (docs/REGIMES.md §4). Left: the kernel
      weights lag `l` by `1 − l/(γ+1)` (Newey–West) where
      docs/ANSWERS-E54-E64.md transcribes the paper as `k((t−u)/γ)`, `1 −
      l/γ`; O(1/γ), inside the size and power bands; to check against the
      paper once.

- [x] 104. **Window expressions as model targets, under an embargo --
      built 2026-10-03** (the review below first, then the build; the
      "Built" record is at the end of the task). Depends on 78 (done) and
      143; its tests read task 152's clocks. Touches clock events, so it
      got the extra review task 120 calls for before it was built.

      #### A target

      - **A target may be a task-143 expression** containing at least one
        forward operator: `targets=[(po.rewm_mean("mid", half_life="10s",
        window_size="1m") - pl.col("mid")).alias("fwd_mid")]`. One with no
        forward operator is known at its own row, so it belongs in
        `with_windows` as a column, and is refused. Several targets in one
        spec share one `X'X` and one embargo; a buy-side, a sell-side and an
        all-trades VWAP are three targets.
      - **A formula target goes everywhere a spec goes** (decided
        2026-10-02, superseding the same day's "1a"): the spec, the saved
        state and TOML carry it as task 143's compact tree, and the bank
        evaluates it with the embedded Polars. "1a" had made formulas
        Python-only, re-supplied on resume, because Polars' serialized
        expression is unstable across versions; the tree reads seven
        stable node kinds at build time and rebuilds through the public
        builders, so that limit is gone. Considered and not taken: Polars'
        SQL expression parser as a shared
        text form -- it parses the element-wise part in both languages
        (arithmetic, `CASE WHEN`, `LN`, a quoted placeholder column) but not
        the operators: Polars SQL's windows are row frames only (no `RANGE`,
        no `FOLLOWING`, no exponential weights), Python's `pl.sql_expr`
        refuses a function of ours and cannot register one, Rust's
        `polars-sql` registers functions only through a full query (its
        expression parser with a registry is private), and it would add
        `polars-sql` and `sqlparser`, statically linked (rule 12); and no
        forward targets on the command line.
      - **Each forward operator states its own `window_size`.** The spec
        states the embargo.
      - **An operator's `partial` applies**: a window cut short by a gap past
        `max_dclock` or a session change is learned from under `"keep"` and
        not under `"null"` or `"drop"`; a reset discards it, unlearned.

      #### The embargo

      - **`label_delay` is renamed `embargo`** in this task, everywhere,
        with no alias: the delay between a row and the learning of its
        target, as `po.stream.embargo` writes it out as data.
      - **The embargo is the only release, and nothing defaults it**
        (decided 2026-10-02). A row is learned once its embargo has passed in
        elapsed time -- the clock column's own steps, and `session_gap` where
        a session change restarts the clock (task 153) -- and never before its
        target is known; with no embargo, a row is learned where it sits once its
        target is known. The embargo means the same for every target, a
        plain column or an expression: a forward expression's window
        cannot close before *t* + w, so an embargo covering the window loses
        nothing the bank could have learned sooner, and it states the delay
        the bank keeps rather than leaving it implicit. Considered and not
        taken: a forward expression learned when its window closes with no
        embargo asked for, and an embargo counted from when the label is
        known.
      - **`fit_predict`** (and `lf.online.fit_predict`, the command line's
        run) **refuses a forward target whose longest `window_size` exceeds
        the embargo**, no embargo included. Row *j*'s target is known only
        at *j* + w, and a target learned at row *t* − 1 covers rows up to *t*
        + w − 2, which overlap row *t*'s own target window. With an embargo
        of at least w, the state that scores row *t* holds only targets
        whose windows closed before *t*.
      - **`fit` takes any embargo, none included, and ends in the same
        state**: the bank learns each row once its target is known, and the
        embargo only moves learning relative to scoring, each held row
        replaying the clock step it arrived with. (Measured on today's
        `label_delay`: a fit with a delay of 5 rows and one with none over
        the same rows less the 5 still waiting predict new rows identically,
        bit for bit, with and without decay.)
      - **No cap on the embargo** (decided 2026-10-02). A break releases
        nothing early (task 153): the embargo counts elapsed time, and a
        break's events wait with the row after it. So an embargo of any
        length is honest for every kind of target, and a forward window may
        be longer than `max_dclock`. The rule that refused an embargo longer
        than `max_dclock` -- written because a break released every held row,
        so a plain forward column with a one-hour embargo and a five-minute
        cap learned, after a ten-minute gap, labels eleven minutes old -- and
        its plain-column exception are gone.
      - **The held rows are bounded by one embargo of each group's rows**,
        and the docs state it: there is no row cap, so a denser stream holds
        proportionally more.
      - **`predict` does not read the embargo**: after a fit the state holds
        only targets whose windows closed before its input ended, so rows
        after that are scored honestly, and rows inside the training history
        are in-sample under any embargo. So no in-sample prediction is ever
        emitted.

      #### How its rows move

      - **Each row is scored and emitted as it arrives.** It is held, with
        only its own columns the target expression reads, until the window
        core resolves it.
      - **The bank releases a row when its embargo has passed and the
        window core has resolved it**: the window closes, a gap or session
        change cuts it, or a reset discards it. The target expression,
        rebuilt from its tree (task 143), is evaluated by the embedded
        Polars on the rows each chunk resolves, then held until the embargo
        has passed. With an embargo of at least the window the
        embargo is the later of the two: with no break between, the core's
        steps are the clock column's, so elapsed time reaches *t* + w no
        sooner than the core does, and a break cuts the window. A break's
        events -- the lag rings' clear, the blend -- wait with its row and run
        when it is learned (task 153; docs/REVIEW-E54-E64.md L2). The case
        the review was to settle, a run of skipped rows the bank read as a
        capped gap and the core did not, is gone with the early release.
      - **The window core sees every row with a value**, including rows the
        spec skips for a null feature: a target depends on the prices ahead,
        not on whether the model could use a row's features. Clock events on
        skipped rows count (task 79).
      - **A row unresolved when the input ends** is never learned; with
        `save_state` it waits in the state for the next run.
      - **`group_close` is refused** with a forward target, as with
        `embargo`: a closed group cannot release the rows it holds.
      - `SCHEMA_VERSION` bumps (rule 5).

      #### The column shows the target the model used

      `po.stream.with_windows(lf, [expr], like=spec)`, with `expr` a target
      of `spec`, equals on every row the model learns from the target it
      learned, to the bit, and is null on every other row: rows a reset
      discarded, rows unresolved at the end of the input, and rows the spec
      skips. `like=spec` carries the spec's clock policy and its rule for the
      rows it learns from -- features and weight usable. `complete` marks a
      cut window; null marks a row the model did not learn from.

      #### Tests

      - Interleaved trades and quotes: three VWAP targets (all trades, buys,
        sells) equal to the column form fed back as plain targets under the
        embargo.
      - Parity, row by row, with skipped trade rows, on streams built to
        contain each event on a skipped row and on an accepted one: a gap
        over `max_dclock`, a session change with a finite `session_gap`,
        `session_gap="reset"`, a step back past `min_backwards_jump` under
        `"reset_state"`, several groups whose clocks restart together, rows at
        one timestamp, a row exactly one window later, a run of skipped rows
        totalling more than `max_dclock`, and the end of input inside a
        window.
      - `fit_predict` refusing an embargo below the longest `window_size`
        and none; `fit` taking both, ending in the state of an embargoed
        fit; `predict` unaffected; a target with no forward operator
        refused.
      - The embargo seen through task 152's clocks: on every row the scored
        row's clock less the learned row's is at least the embargo in elapsed
        time, and the next row the bank holds is less than one embargo back,
        across gaps and session changes alike.

      #### Reviewed before building (2026-10-03)

      The review task 120 asks for, done against the code of task 143's core
      and the bank's `apply_label_delay` (task 153), event by event. The
      names are task 144's (`gap_cap`, `restart_after_step_back`, `embargo`).

      - **One clock, two walkers.** The bank steps each group's
        `ClockState::advance` with the spec's `ClockCfg`; the core steps
        the same `advance` with the same cfg per group *and* once more on
        the stream's own clock across groups. Every event the bank sees --
        a gap past `gap_cap` (`capped`), a session change with a finite
        `session_gap` (`capped` or not by the gap's size) or `"reset"`
        (`reset`), a step back past `restart_after_step_back` (`reset`), a
        late row (refused) -- the core sees on the same row with the same
        verdict, because it is the same function on the same arguments;
        only `accept` differs (the bank folds a skipped row's time into the
        next accepted row, the core counts every row), and `accept` never
        changes a verdict. So a window cut or discarded by the core is cut
        or discarded exactly where the bank's `held_break` or `reset` lands.
      - **One core per group, as one stream per group.** `with_windows`
        runs one core over every group with a clock of the stream's beside
        each group's, which refuses a row of group B earlier than group A's
        last and cuts every group's open windows at a gap on the stream --
        neither of which the bank's per-group clocks see. The bank keeps a
        core per group instead, fed that group's rows: each group's clock
        is the spec's clock as its stream runs it, a stream interleaved on
        clocks of its own runs (as it does for a plain target), and a
        group's rows resolve in their own order, so a resolution is known
        at the row that made it. (Found in the build: one core over every
        group emits a row only once every row before it, of any group, has
        resolved, and that wait moved with the chunking.) A gap on the
        stream reaches each group at its next row, where its own clock
        shows the capped gap, and the cut value is the same -- the window
        saw no row of its group after the gap either way.
      - **Release is the later of the two, and chunk order decides it.** A
        resolution is applied in row order -- at the row that closed, cut
        or discarded the window, never earlier -- because with `embargo`
        equal to the window under `closed="right"` a row exactly one window
        later has the embargo run out while the window is still open (that
        row is a member); applying the chunk's resolutions up front would
        release it a row early in a coarse chunking and not in a fine one.
        Each resolution therefore carries the sequence number of the row
        that made it, and a pending row is released when its wait has run
        out *and* a resolution at or before the current row has reached it.
        Within a group, pending rows resolve in arrival order (windows close
        in clock order, a cut or a discard takes every open window, rows at
        one stamp share one), so the resolved rows are a prefix of the
        buffer and the release stays the `take_while` it is.
      - **A refusal leaves the bank as it was.** The core is fed after the
        clock check and before the streams, on a snapshot that is restored
        when the chunk is refused later (its own refusal, or the window
        pre-pass); the pre-pass replays with the chunk's resolutions.
      - **`fit` and `fit_predict` are one Rust method**, so the refusal of
        an embargo below the longest forward `window_size` is a bank flag
        the Python `fit` sets for its run (`set_learn_only`), computed per
        spec when the bank is built; the command line never sets it.
      - **Decided here, where the design above was silent:** a formula
        target's columns reach the bank as numbers or text (a temporal
        column other than the clock is refused, as for every role); its
        columns may be features, since the leak check guards a *plain*
        target's column read at the same row, and a forward window over
        `mid` is not the row's `mid` (`closed="left"` or `"both"` puts the
        row's own value in the window, which is the user's choice);
        `resid_<name>` is null on the scored row (the target is not known
        then; the diagnostics fold at release as under any embargo); for a
        target `"drop"` means what `"null"` means (not learned from), since
        a row of a model is scored and cannot leave; a row unresolved when a
        run ends stays in the state with the core's rows, so a resumed run
        learns it when its window closes; a row the spec skips is fed to the
        core with its value and is never pending; `SCHEMA_VERSION` 23 with
        the bank's minimum left at 22 (a 22 file has no formula target and
        loads as it was).

      #### Built (2026-10-03)

      - **A target is a `TargetDef` with a `formula`** (`targets.rs`): a
        table `{name, formula}` on every surface -- the spec's JSON, the
        state file, the CLI's TOML (`{ name = "fwd", formula = [...] }`) --
        refused without a name, with a column too, or with no operator
        looking ahead. `TargetDef::columns()` is what every reader of a
        target's columns now asks (the chunk's casts, the first-reader
        roles, the clock clash); `value_column()` is `None` for a formula,
        so the leak check passes its columns as features by design. In
        Python a `pl.Expr` in `targets` becomes the table (`_spec.py`,
        `formula_target`), `po.FormulaTarget` is its type, and
        `target_columns` walks the tree for the plan's projection.
      - **The bank resolves it** (`bank.rs`, `TargetWindows`): per spec,
        one `WindowsRun` per group over the formulas under the spec's
        clock policy (no `like=`: the core sees every row), fed each
        chunk's rows group by group in row order after the clock check and
        before the window pre-pass and the streams, on a snapshot restored
        when a later check refuses the chunk. `WindowsRun::feed_resolving`
        returns every resolved row with the formulas' values (null where
        a `"drop"` operator was partial), its number in the bank's stream
        (`@po:row`, from `rows_fed`) and the number of the row that
        resolved it, read off the core's ready count after each push. The
        chunk's resolutions reach each stream as `FormulaTargets` -- the
        formula slots, each row's number in the laid-out order, and its
        group's `Resolutions`, flat (row numbers, resolving rows, values
        row-major: no allocation per row; a map of a `Vec` per row cost
        the first build 1.7 times the column form) -- and
        `apply_label_delay` releases a row when its wait has run out *and*
        a resolution made at or before the current row has reached it,
        found by binary search over the resolved prefix of the buffer at
        every row of the group, skipped rows included (a skipped row
        closes windows as any row does, and the next accepted row may be
        chunks away: with one row a chunk, resolutions made at skipped
        rows were lost and the buffer stalled behind them); with no
        embargo the buffer runs with a wait of 0. `PendingRow` gains
        `seq` and `resolved`, both defaulted.
      - **`fit_predict` refuses, `fit` takes**: the longest forward
        `window_size` against the embargo is computed when the bank is
        built and refused at the first `fit_predict` unless
        `Bank::set_learn_only(true)`, which `ModelBank.fit` sets for its
        run; the CLI never sets it.
      - **The state carries the cores**: `BankFile.resolvers`, per spec
        and group, each `WindowsRun::save_bytes`; a loaded bank resumes
        each at its group's first chunk, which says what the columns are.
        Schema 23 (24 since review R4 and 25 since review R6, which changed the window core's state).
      - **Tests**: `crates/online-polars/tests/formula_targets.rs` (the
        column form fed back, bit for bit, in 1, 7 and 600 chunks; the
        short-embargo refusal and `fit`'s equal predictions; a state saved
        mid-window; groups on clocks of their own equal to each group run
        alone; what the spec refuses) and `tests/test_formula_targets.py`
        (the same parity on the three VWAPs over interleaved trades and
        quotes and through every clock event of the review -- a gap past
        the cap, a session change with a finite gap and with `"reset"`, a
        step back the policy restarts on, groups restarting together,
        rows at one stamp, a row exactly one window later, a run of
        skipped rows longer than the cap, the end of the input inside a
        window -- each on a skipped row and an accepted one, in one chunk
        and seven; the row exactly one window later under an embargo equal
        to the window, the one place the two forms part; `fit_predict`'s
        refusal and `fit`'s acceptance on every surface; `predict`; the
        embargo and the window seen through task 152's clocks; a state
        saved mid-window; the plan's projection; the TOML form; a formula
        beside a plain target).
      - **Measured** (2026-10-03, PERFORMANCE §34): the forward VWAP of
        §33 learned natively takes 1.5 times the column form's wall time
        at 4M rows (2.37 against 1.57 s) and 1.6 at 16M (9.37 against
        5.87), with less memory (1.09 against 1.23 GB). The first build
        was 1.7 times (a map of a `Vec` per resolved row; flattened).
        What remains is the overlap the query form gets for free: its
        window and bank are two sources the streaming engine runs as two
        stages, and its 1.57 s is below the 2.08 s of `with_windows`
        alone and the model alone run one after the other, where the
        native path runs both in one source, 0.29 s (14%) above that
        sum. One polars thread moves none of the four, so the overlap is
        the engine's source tasks. Closing it needs the next chunk before
        the current one returns, which a pull-based source has not got;
        the groups' cores on the pool would gain nothing on one group.
        Neither is built; the user's call whether the native form's
        convenience is worth the 1.5.

- [x] 105. **`po.prep` renamed `po.stream`, and the rules every function in
      it follows — split from task 78 on 2026-09-25; built 2026-09-28** (the
      user: "Do 105"). The rename (105a), and the rules applied to the two
      functions there are: both give back the kind of frame they are given
      (rule 1; `embargo` returned a `LazyFrame` for a `DataFrame`);
      `refresh_time` takes `clock=` and `group=` and refuses a step back, as a
      spec's default `on_clock_reset` does, comparing a temporal clock in
      nanoseconds (rule 2; task 120 settled its clock); and it resumes (rule
      5), with `load_state` read when the plan is built and `save_state`
      written once the input's last row is fed, atomically, a slice of the
      output not stopping the input early. Its state is its own file
      (`RefreshFile`, versioned msgpack with `names` and `pairs`, which a
      load must match), with a temporal clock kept in nanoseconds so it
      resumes on any unit. Rules 3 and 4 are the windows' (task 78). The user,
      2026-09-11: `po.prep` is to be renamed and its API defined "the way
      it should be done" — pre-1.0, so no aliases and no compatibility. No
      behaviour change; the one breaking change queued, so it belongs in a
      minor release. **The clock of `refresh_time` under the one vocabulary
      needs extra review** (task 120). The passages below moved from task 78
      word for word.

      **`po.prep` becomes `po.stream`**: the namespace for streaming
      transforms of a plan — `LazyFrame` in, `LazyFrame` out, O(state)
      memory, before or beside a bank. "Stream" is already a defined word in
      the README's glossary. `embargo` and `refresh_time` move; nothing
      stays behind under `po.prep`.

      Every function in `po.stream` follows five rules — what "the way it
      should be done" means here:

      1. **Frame first, the same kind back** — `LazyFrame` in gives a
         `LazyFrame`, `DataFrame` gives a `DataFrame`, as `po.fit_predict`
         does. Chaining is Polars' own `lf.pipe(...)`, and
         `with_windows` is also `lf.online.with_windows` and
         `df.online.with_windows`, as `with_columns` is a method.
      2. **One vocabulary**: the clock and its policy -- `clock`,
         `max_dclock`, `on_clock_reset`, `session`, `session_gap`, `group`
         -- have a spec's names, defaults and meanings, run by the same Rust
         clock code; a window operator takes Polars' names (task 144). `refresh_time`'s `time=` becomes
         `clock=` and `by=` becomes `group=`. With no `clock`, one unit is
         one row.
      3. **What the rows share goes on the call; what a window decides goes
         on the window.** The call takes the clock and its policy — `clock`,
         `max_dclock`, `on_clock_reset`, `session`, `session_gap`, `group` —
         and the state; each window operator takes its own input and
         parameters (task 143).
      4. **Output names as in `with_columns`**: from keywords and `alias`.
         Two outputs with one name, or an output named like an input column,
         are refused while the plan is built.
      5. **Stateful transforms resume** with `load_state` / `save_state`;
         `chunk_rows` as everywhere. `embargo` is a pure Polars plan
         (`merge_sorted`) with nothing to save and takes neither.

      #### Sub-tasks

      - [x] 105a. (was 78a) **The rename**, alone, first: `po.prep` →
            `po.stream`, `time=` → `clock=` and `by=` → `group=` in
            `refresh_time`, its tests, the reference page, the README,
            `llms.txt`, and the API surface file. No behaviour change, so
            every existing test passes with only its spelling changed.

- [x] 106. **Discarded by the user on 2026-09-25: a backward clock is not a
      session change.** "We no longer want to assume that a backward clock
      is a session change, this is an old idea that should be discarded."
      Not to be built or revived: `on_clock_reset` keeps its four policies,
      `session_gap` still needs a `session` column, and a step back stays
      what `min_backwards_jump` and `on_clock_reset` make it. The design
      below is kept as the record of the idea, split from task 78 on the
      same day, word for word.

      **A backwards clock is a session break.** The user's rules,
      2026-09-11: "When a forward window encounters a backward clock, we
      want the same effect as if there were a large clock gap where the
      window just runs out", and then "If a clock moves backward it must be
      a session break or a problem in the input data and we can't tell
      which so we assume session break." And then, for the decay too: "the
      decay should take the session gap as well since that would be needed
      anyway to completely recreate the target used by the model." Read against
      `ClockState::advance` (`crates/online-core/src/clock.rs`) and
      `apply_label_delay`, per `on_clock_reset`:

    | policy (today) | a waiting row today | as a session break |
    |---|---|---|
    | `"max"` | `capped = true` — the buffer releases, the window ends | unchanged |
    | `"reset_state"` | the buffer is discarded; the model starts over | unchanged — a reset, as `session_gap="reset"` is at a session change |
    | `"zero"` | a zero step, not capped: the buffer keeps waiting, and a window would run on across the jump, counting the rows after it as the row's own clock | **the buffer releases and every window ends** |
    | `"error"` | the run stops | unchanged |

      - **In both forms, a backward step on a group's clock does to the
        waiting rows what a session change does**: every open forward
        window of the group ends — no row at or after the jump enters,
        the value is over the rows before it, under `partial` — and the
        `label_delay` buffer releases in order. One line in
        `apply_label_delay`: `plan.backwards` joins `session_changed` and
        `capped` in the release condition.
      - **The decay takes it too: a backward step is a session change,
        everywhere.** `ClockState::advance` treats a raw step below zero as
        it treats a change of session value: `session_changed` is set, the
        step is `session_gap` (limited to `max_dclock`), and
        `session_gap="reset"` starts the model over. So everything that
        follows a session change follows a backward step — the decay, the
        `label_delay` release, the lag rings cleared, `session_shrink`'s
        blend, `group_close="session"` closing the group, and the windows
        ending. One code path, which is what lets the column form recreate
        the target the model learns from exactly.
      - **`on_clock_reset` shrinks to `"session"` (the default) and
        `"error"`.** The three policies it loses are each a `session_gap`:
        `"max"` is `session_gap = max_dclock`, `"zero"` is `session_gap = 0`,
        `"reset_state"` is `session_gap = "reset"`. Nothing is lost except
        choosing a different step for a backward clock than for a session
        change — which the rule says cannot be told apart anyway. `"error"`
        stays, for input where a backward clock can only be a bug.
      - **`session_gap` without a `session` column**, now meaningful: it is
        the step at a backward clock. With no `session` column it defaults
        to `max_dclock` — today's default decay step at a backward clock
        (`"max"`), so a stream with no session column and no `session_gap`
        decays exactly as before. With a `session` column it stays required,
        as now.
      - **What moves.** Pre-1.0, so no compatibility spelling: the three
        policy words are refused with a message naming the `session_gap`
        that replaces each. Numbers move for a stream with a backward clock
        under `"zero"` or `"reset_state"` (now session changes: a released
        buffer, cleared lag rings), under `"max"` where a `session` column
        gave a different `session_gap`, and for any spec with
        `session_shrink` or `group_close="session"`, which now also fire at
        a backward step. `SCHEMA_VERSION` bumps (the clock config's layout
        changes), shared with 78e's bump if they ship together, with a
        loader that maps a saved `"max"` / `"zero"` / `"reset_state"` to the
        equivalent `session_gap` when the file has none. A minor release,
        with the list above as its first CHANGELOG lines.
      #### Sub-tasks

      - [x] 106a. (was 78b, discarded) **A backward clock is a session
            change**, in `ClockState::advance`, before any window is built:
            every model's decay, `label_delay`, lag rings, `session_shrink`
            and `group_close` follow from it; `on_clock_reset` becomes
            `"session"` / `"error"`; `session_gap` allowed without
            `session`, defaulting to `max_dclock`; the README's clock
            section, the reference docstrings and every test that names a
            removed policy. Its own commit, its own CHANGELOG lines, since
            it changes numbers the windows do not depend on.
      #### Tests

      - A backward step and a session change at the same row, for every
        model (moved from task 78's parity test):
        **A backward step equals a session change at the same row, for
        every model**: one stream with a backward clock and no `session`
        column against the same stream with a `session` column that changes
        exactly there and the clock made monotone, identical output field by
        field — decay, `label_delay`, lag features, `session_shrink`,
        `group_close="session"`. Each removed policy word refused with the
        `session_gap` that replaces it; a v6 state saved under each loading
        with the equivalent gap. For `"reset_state"` and
        `session_gap="reset"`, the rows with `complete` false are exactly
        the rows the model never learned from.

- [x] 107. **Relative targets -- built 2026-09-25 for plain columns;
      windows take it as arithmetic in task 143.** A level -- a price, a
      VWAP -- is rarely what a regression should predict; where it goes from
      now is.

      - **A plain-column target** takes `relative_to=`, a column read at the
        target's own row (the mid, the last trade), and `relative=`:
        `"difference"` (default, `y − r`), `"ratio"` (`y / r`) or
        `"log_ratio"` (`ln(y / r)`): `po.target(column, *, relative_to,
        relative, name)`, and in TOML a table, `targets = ["ret_5m", {
        column = "price_5m", relative_to = "mid" }]`.
      - **No look-ahead is added**: `r` at row *t* is known when row *t*
        arrives. A null `r`, or for the two ratios a non-positive `y` or
        `r`, makes the target null on that row: not learned from, never a
        NaN in the state (hard rule 9).
      - **Everything downstream is on the relative scale**: `pred`, `resid`,
        `sigma`, the metrics and the conformal interval, so `hit_rate` is
        the direction of the move. The output does not add `r` back.
      - **A window target is relative by arithmetic** in task 143's
        expressions: `- pl.col("mid")`, `/ pl.col("ask")`, `(op /
        pl.col("bid")).log()`, in the column form and the target form alike.
      - **`"log_ratio"`** stores an `f64` like any target. `ln` can differ
        in its last bit between glibc and Apple's libm, so a log-ratio
        target is reproducible across OSes to the last few bits, and no
        `"log_ratio"` spec goes into a byte-identical cross-platform fixture
        (`state_schema*.rs`, the release workflow's write-on-macOS,
        read-elsewhere job).

      #### Tests

      - `po.target("p", relative_to="r")` against the same spec given
        `p − r` as a plain column, identical field by field, for each of the
        three ways, in `ewridge`, `kalman`, `huber`, `holt` and `marginal`
        (`tests/test_relative_targets.py`); a null `r` and a non-positive
        value giving a null target that is not learned; the TOML table
        loading to the same spec as the Python one.
        **Review 2026-09-26.** The IO plugin's projection set was built
        with `set.update(spec["targets"])`, so a table target -- relative or
        merely renamed -- raised `unhashable type: 'dict'` on every
        LazyFrame path, and a fix by name alone would have projected the
        reference away (D1, F1): `target_columns` gives a target's column
        and its reference, and `_spec_columns` keeps both
        (`test_the_lazy_plan_reads_a_relative_targets_columns`). `sgd` under
        `"logistic"` and `"poisson"` took a relative target `ftrl` refused
        (D2): refused by name, and `ftrl`'s guard names its default loss
        rather than everything but `"squared"`. A ratio target is positive
        by construction, so `hit_rate`'s sign test read 1.0 whatever the
        fit (D3): the hit test is about 1 for a ratio, `SlotMetrics::update_about`,
        pinned by a recursion recomputed from the output
        (`test_hit_rate_under_a_ratio_target_is_about_one`). The `bocpd`/`hmm`
        slot check compared names, so `{"column": "z", "name": "h"}` put
        column `z` in the hazard slot (D4): the slot must be the plain
        column; and the message printed `Targets`' derived `Debug` (D5).
        The leak check matched a target's *name* against the features and
        refused a target merely named like one (D7): the column alone. An
        empty `name`, `column` or `relative_to` was refused nowhere (D
        missing 5): refused by the builder and by the bank. The builders'
        `_matches` did not read a TypedDict, so a wrong table entry reached
        serde and was named by JSON path (F4), and a list's entries escaped
        the int floor (F7). Docs: a result the model cannot use is null too
        (D8); the pairs table and `describe` carry the *name* (E6); a table
        naming its own column comes back as the column (F6). New tests: the
        equality under a group, a label delay, both, a window and the
        diagnostics with the reference null on row 0; `predict` before any
        fit and with either column dropped; the specs and the JSON carry the
        table; a saved bank held to its targets' `relative`; a renamed table
        is the name in the fields, the Gram, `describe`, `summary`,
        `marginal` and a closed group's pairs; two views of one column; the
        role tables of the error-message suite name `relative_to`.

**The plan of 2026-09-25.** Three audits compared every open list with the
code (`docs/PLAN.md`, the other design docs, the release checklist, CI and
GitHub). The user's call on the result: everything that integrates a new
library or Arrow is parked (below the tasks); everything else is on this
plan to be reviewed and, the user expects, done now; and any work on rows
out of order or a clock that goes back **needs extra review before it is
built**: task 120 (what an audit of that area found), 104, the clock rules
of 78 and the clock of 105's `refresh_time`, and the items marked so below.
Task 106, the idea that a backward clock is a session change, was discarded
by the user the same day. In tests, any library under an open-source licence
may be used (task 121), and only licensed ones -- source-available or
commercial terms -- are parked (the user, the same day). Each task says its
size (S under a day, M days, L a week or more), what it waits on, and the
decision it needs, with a recommendation where there is one.

- [x] 108. **The docs say what the code does.** S–M, no code, no decision.
      **Done 2026-09-25:** a survey checked each item against the code,
      and 107 edits followed, each an exact match of the stale text: the
      removed surfaces marked as removed where they read as live (a note
      heads ENHANCEMENTS and IMPROVEMENTS for the done rows that name
      them, and code comments drop them); the Arrow import corrected to
      unbuilt and parked in ARROW-SOURCES, task 86's head and the two
      reviews; clustering and B2 recorded as built; the counts recounted
      (976 Rust tests, 3,265 pytest cases), the schema 16 and the tested
      Polars versions dated; the README's `min_periods` sentence replaced
      by a table of the defaults `Spec::default_min_periods` gives;
      WARMUP-AND-CONVERGENCE held to what was built for `ewridge`;
      REVIEW-2026-09-18's six "candidate" statuses closed with batch 3's
      commit and tests, and the 2026-09-12 review's open V entries struck
      with their closures; ANSWERS' D̂₃ corrected to the code's gradient;
      IMPROVEMENTS P3 noted as done by PERFORMANCE P11; TESTING's T-W rows
      marked, with T-W8's case-insensitivity the one open entry; and this
      plan's status line, task 61 split from task 68, §3 and §9.
      The audits found text the code has left behind: the removed expression
      plugin (task 85) and `po.run` (task 83) described as live in
      ENHANCEMENTS, IMPROVEMENTS, PERFORMANCE, CLUSTERING, STATE-WORKFLOW and
      `runner.rs`'s comments; the Arrow import called blocked (ARROW-SOURCES
      §4 and the head of task 86, both retracted 2026-09-22), and the two
      reviews that repeat it; clustering and B2 (built as `window`) recorded
      as undecided (ENHANCEMENTS, CLUSTERING §12, BEYOND-O-STATE, PLAN §11h);
      stale counts ("650 Rust tests and 2,200 Python cases"), schema numbers
      and the tested Polars version; the README's rule that every model but
      `ewridge` defaults `min_periods` to one per unknown, which is wrong for
      seven models; WARMUP-AND-CONVERGENCE's stale passages; the statuses of
      REVIEW-2026-09-18 (B1, S1, D1, P1, S4, D2 still "candidate") and of the
      2026-09-12 review's V-list; ANSWERS-E54-E64's D̂₃ marked "Verified";
      IMPROVEMENTS P3 "rejected" where PERFORMANCE P11 is done; TESTING's
      "Open entries: None" beside unmarked T-W rows; this plan's own status
      line ("tasks 1–58 done, released as 0.2.0"), task 61 fused onto task
      68, and the §3 table and §9 item 4 listing the clock policies.

- [x] 109. **Release 0.11.1** (planned as 0.11.0). S each; the push and
      the approval are the user's. **Tagged `v0.11.0` on 2026-09-26** on the
      user's word ("merge, push and tag the next minor"), after the review of
      task 132: the version in the six places and both locks, the
      comparison against 0.10.0 measured again (21 of 30 specs, 130 fields,
      median 3e-16, at most 1.4e-14), the CHANGELOG promoted. **Never
      published** (below); **the release is 0.11.1**, dispatched by the
      workflow of task 133, with the same changes and the fixes below. Not done: the suite at the Polars floor 1.34.0
      (last run 2026-09-02), the glibc floor, the 0.10.0-wheel state test,
      S6 and task 105. **The tag's first run failed, and nothing was
      published** (2026-09-27): the rehearsal on `main` (RELEASE-READINESS
      step 2) was skipped, and the branch's 21 commits reached CI for the
      first time on the tag. Two tests, both the tests' fault:
      `the_bandwidth_override_reaches_the_kernel` (task 103) failed on Linux
      and Windows -- at a bandwidth of `usize::MAX` the long-run variance is
      rounding, its sign set by Box–Muller's libm bits, and where it rounded
      to 0 the model said NaN (no verdict) and the longhand, folding with
      `f64::max`, said 0.0; it runs thirty streams now and takes the model's
      NaN rule (reproduced here at seed 13). And the licence check (task
      121) read sphinx's metadata where CI's `uv sync` installs `dev` alone:
      `docs` is a default group now. Because `cargo test` stopped at the
      first failing binary, Linux and Windows ran none of the integration
      tests and Windows no pytest; CI passes `--no-fail-fast` now. The
      Mutants jobs failed on the same unit test in their baseline. The gate
      of that fix found **G4** in the contract's proptest: `deco`'s
      `loglik` was `-inf` after rows weighted `0.01, 0.01, 1e100` and one at
      `1e100` (a spread near `5e-52` standardised the row to `2e151`, the
      density's `ρ` sat `1e-9` from singular, the quadratic form reached
      `1e311`); it is NaN now, no reading, as `hmm`'s is
      (`a_log_density_past_the_range_is_no_reading`). A randomly seeded
      proptest that finds such cases can fail CI at random, so the fix was
      followed by more runs before any push, and they found **G5**: rows at
      `1e100` before a window and one of weight `1e100` inside it left the
      live co-moments near `1e100` with the window's spread below their
      last digit, the subtraction returned noise near `1e84`, and the
      windowed `lasso` predicted `-inf`. `crate::truncated` now reads a
      variance no larger than `64ε` of the terms it is formed from as no
      spread, zeroing its row and column as a held feature's are
      (`a_window_the_live_state_cannot_resolve_reads_as_no_spread`, which
      read `1.3e84` without it, and
      `a_window_the_live_state_cannot_resolve_through_the_bound`). The tag
      could not be moved -- `v*` tags are immutable, by the ruleset -- so
      the user decided (2026-09-27) that the release ships as **0.11.1**
      and that the workflow creates the tag after everything runs (task
      133). Before the tag, the list as it stood:
      - Refuse `max_error_inflation` by name on a model that ignores it:
        today it is range-checked and dropped on every model but `ewridge`,
        where `emit_error_inflation` is refused (spec.rs:2417-2433).
        *Decision: refuse (recommended) or document it as ignored.*
      - **Done 2026-09-25.** `EwDiag::update`: drop the zero-weight loop,
        a no-op (`a = 1`, `b = 0`), which also removes an equivalent-mutant
        entry whose line matches the other branch too, where a real
        survivor would hide. Only finite values ever reached it: the
        stream skips a row with a missing feature before a model sees it.
      - **Done 2026-09-25.** The window test oracles keep `age < window`
        where the models keep `≤` (test_window.py:219, ewcov.rs, ewridge.rs,
        marginal.rs tests). None had a row exactly one window old, so none
        could tell: the Rust oracles now step the clock in quarter units,
        which land rows on the boundary (12 to 20 reads each, counted and
        asserted), and failed under `<` before the fix. The Python ridge
        oracle's data were noise-free, which hid it the same way; a noisy
        test now puts the model 6.6e-16 from the inclusive fit and 2.2e-3
        from the exclusive one.
      - **Done 2026-09-25.** `po.sim` in the API snapshot, which now reads
        the package's modules from its directory rather than a hand-kept
        list; FTRL's `zz < 0` equivalent mutant recorded.
      - The suite at the Polars floor, 1.34.0 (last run 2026-09-02); the
        Linux CLI binaries' glibc floor before GitHub's Ubuntu 26 runners
        (2026-10-19).
      - The release comparison against 0.10.0 again (the CHANGELOG's figure
        predates tasks 102 and 103); the version (RELEASE-READINESS.md:150's
        places, both locks); the CHANGELOG promoted; the README's pin.
      - *Decision, the schema:* keep the loaders for 14 and 15 that the
        CHANGELOG promises (recommended, with a test that loads a state
        written by the released 0.10.0 wheel), or raise the minimum to 16.
      - *Optional:* S6, the stream's persisted fields as one sub-struct,
        cheapest while schema 16 is unreleased.
      - *Recommended:* task 105 in this release, the one breaking change
        queued, so the API breaks once.
      - Then the user's: fast-forward `main` and push (the first CI on
        Linux 3.13/3.14, macOS 3.14 and Windows; the Mutants job will likely
        time out on a 3,700-line diff and does not gate a release), the
        rehearsal, the tag, the upload's approval; then a clean-venv install.

      **Review 2026-09-26.** `max_error_inflation` is refused by name
      on a model without a ridge system, as `emit_error_inflation` was
      (`test_is_refused_by_name_where_no_ridge_system_reads_it`). The
      schema: the loaders for 14 and 15 stay, as the CHANGELOG promises; the
      test that loads a state written by the released 0.10.0 wheel is not
      built. Task 105 is not in this release: its clock rules need the
      user's review (task 120). **2026-09-27, after 0.11.1:** the state test is
      built, `tests/test_released_state.py`, for 0.10.0 (schema 14) and
      0.11.1 (schema 17); `scripts/release_probe.py --states` writes the
      files under the released build (RELEASE-READINESS, "Compare with the
      last release"). The Linux CLI's glibc floor is measured: 2.39, a
      decision now (task 115 (i)). The suite at the Polars floor, 1.34.0, ran
      again: 3,408 of 3,425 passed, and the one failure in the package,
      `po.eval.seqtest` with `by` (a window inside a window, which 1.34.0
      refuses), is fixed; the rest are `scan_arrow_c_stream` (1.43.0 on,
      now said in ARROW-SOURCES), two duration oracles and the two version
      pins (RELEASE-READINESS, "The floor and the ceiling").

- [x] 110. **This week's review leftovers, as tests.** S. Held-value tests
      for `ew_class` and `hmm`; weight-0 rows for `bocpd`, `deco` and `hmm`;
      each model's state saved before the low parts, loaded; `ew_class`'s
      window reading a held feature; two targets under pairwise gaps with a
      held feature (task 101's review). **Done 2026-09-27.** In
      `held_values.rs`: `ew_class` classifies and `hmm` filters the same at
      every level (measured a thousandth of `steps_of` and less; `hmm`'s
      state spread decays at its share of the rows, 2.7e-22 of the stop,
      so it is held to level 0.5's rather than to `ew_cov`'s 1e-30); a row
      of no weight moves nothing in `bocpd`, `deco` or `hmm`, to the bit;
      two targets under `pairwise` keep their slopes at every level -- the
      second target's slope wanders under `pairwise` at every level,
      level 0.5 included, from before the feature stops (the all-row Gram
      against a third fewer cross-moments), so it is held to 0.5's. In
      `model_contract.rs`: `every_model_resumes_from_a_state_without_the_low_parts`
      drops every `_lo` field of every model's state, in msgpack (`rmpv`,
      a test-only dev-dependency, MIT: serde_json turns an infinite cfg
      value into null), and holds the continuation to the whole state's
      within 1e-11 (3.7e-13 measured), with each of the 15 models that
      keep a mean shown to have dropped parts; and
      `a_schema_16_state_of_offsets_loads_as_own_means`, the G3
      conversion's first test on a real schema-16 shape. In `ewclass.rs`:
      `the_window_reads_a_held_feature_as_no_spread_in_every_class`.

- [x] 111. **The model contract, checked where it is only listed.** S–M.
      EXTENDING's checklist misses four places a test checks (`BUILDERS`,
      the OUTPUTS meanings, llms.txt's count, the README's API lines);
      `ewridge`, `sgd` and `kalman` lack a wrong-shape refusal test; the
      bit-flip fuzz covers one spec; the save/load and realized-struct
      sweeps cover the regressions only; lists kept by hand stand in for
      checks (test_bank.py:330, test_coef.py:114-141). **Done 2026-09-27.**
      EXTENDING names all four and the new sweeps. Every model module has
      its wrong-shape test (`sgd` and `kalman` check in their `TryFrom`, so
      theirs go through the encoding). `summary.rs`'s
      `every_model_kind_refuses_or_loads_a_corrupt_file_never_panics` cuts
      and bit-flips a state of every kind (thinned: every 7th prefix, every
      5th byte), held to `ModelKind::KINDS`: no kind panics. `tests/test_every_kind.py`
      runs the realized-struct and the mid-stream save/load checks over the
      registry's `MINIMAL` itself -- `rcov`, `hmm`, `corrchange` and `bocpd`
      had neither. The hand lists: `test_coef`'s `COEF_SPECS` gained the five
      kinds with a `coef` it missed (`quantile`, `kmeans`, `ew_class`, `hmm`,
      `deco`) and a check against `coef_index`; the sparse-target test runs
      every regression model, so its per-target set is checked by the
      others; `test_error_messages.BUILDERS` gained the seven builders its
      inf and NaN sweeps never saw, held to `MINIMAL` (all seven were
      right). `test_kwargs_typing.BUILDERS` stays a list -- deriving it would
      import the registry that imports it -- and is held by
      `test_the_builder_list_covers_every_builder`, which EXTENDING now names.

- [x] 112. **Older review items that need no decision.** S each. C16's raw
      oracles; C21's r2 and coverage under a delay; S23's docs; D9, the dead
      `LassoCfg::combo_labels`; REVIEW-2026-09-18 §6 (a constant column in
      `deco`, `bocpd`'s 1e5-row truncation and its gaussian at d ≥ 3); T2's
      roll call on real states (4 of 20 kinds); U5, a Decimal parquet
      through the CLI (its test went with task 83); REGIMES: `hmm` through a
      regime switch, `deco` and `rcov` experiments, WKD's size at t5, the
      Epps sweep that ends at its best point; PERFORMANCE: the measurements
      owed (prefetch limits, §19's state sizes, §21's trivial scan, task
      101's cost) and a re-run of its timings. **The tests and fixes done
      2026-09-27, and the rest the same day.** REGIMES (run on 2026-09-27,
      every earlier figure unchanged): `hmm` through a switch holds each new
      state in a median of 12 rows from the true covariances and follows
      none seeded from the rows; `deco` settles at `E[u]`, two-thirds of the
      truth, and crosses in one halflife; `rcov`'s estimators each win one
      condition; the size study at 20,000 draws a cell puts the shared-scale
      `t_5` within 0.004 of WKD in every cell, so the 0.024 was noise and
      the independent reading is the one that misses; the Epps sweep to
      `L = 1024` levels off at 0.775 between 256 and 512. PERFORMANCE §26:
      the benchmark under 0.10.0, 0.11.1 and this build found two slowdowns
      0.11.0 shipped, bisected to tasks 101 and 132 and fixed bit for bit
      (task 101's record corrected); `micro`/`kmeans`' cost is task 102's
      default floor, kept; the CSV and NDJSON read-ahead limits change
      nothing on either surface; §19's two states measured; §21 re-measured
      by `scripts/plan_inspection_bench.py`. C16: both blend oracles are the
      centred mixture, run at 0 and 1e8. C21: `r2_y` under a delay is
      scikit-learn's `r2_score` over the matured rows, to 1e-9; and
      **`coverage_y` was not** -- a held row was scored, at release, against
      the conformal radius reached by then rather than the radius its
      interval was shown with (C21's other half): the record of a held row
      now carries the radius it was shown, `Conformal::update_against`
      scores against it and steps the current radius by that error
      (delayed-feedback ACI), and a row shown no interval counts for
      neither; schema 18, and a 17 file's rows are scored as before
      (`a_delayed_row_is_scored_against_the_radius_it_was_shown`,
      `test_coverage_is_the_share_of_matured_rows_inside_their_interval`).
      S23: `emit_averaged`'s docs say its weights are a mean loss ratio that
      does not sharpen, and a slot without a prediction drops out of the
      row's average. D9: deleted. §6: a constant column in `deco` freezes
      every block (a decision now, task 115 (h)); `bocpd` in a stationary
      stream prunes little -- 2,001 runs kept after 2,000 rows and 7,036 at
      most by 20,000 under the defaults, so `max_run` (10,000) is the real
      bound on its memory and its per-row cost, now in its docs -- and a
      test holds a stream at `max_run = 1,000` for 3,000 rows, finite
      throughout, with a 4σ shift found by `run_mode` within 3 rows (the
      pruned and exact runs agree after the shift; `p_change`, `P(r ≤ 1)`,
      peaks at the shifted row and falls, so it is not the reading to wait
      on). A 100,000-row test would take four minutes a debug run. Its
      `gaussian` emission at d = 3 is a textbook
      multivariate-t longhand in Algorithm 1, to 1e-10. T2: every kind is
      read off a real model's state. U5: a Decimal and a `UInt8` column
      through the CLI give the bank's numbers on the same `Float64`s.

- [x] 113. **CI and tooling.** S–M. S7, one composite action for the Linux
      prep step (five copies now); a coverage run, and whether Rust coverage
      joins CI; a fresh mutation baseline and its triage (the weekly job
      starts 2026-09-27; `lasso` and `ewridge` first); WRITING's step-5
      structure checks as a script. **S7 and the script done 2026-09-27.**
      S7: `.github/actions/linux-build-prep`, called by all five workflows
      (SIMPLIFICATION S7 has the rest). The script:
      `scripts/doc_structure.py` parses each file as CommonMark with
      GitHub's tables (markdown-it-py, a new dev dependency), counts a row's
      cells on its source line as GFM splits it, and makes GitHub's anchors;
      its anchors, tables and rows matched GitHub's own rendering of every
      tracked file (1,066 anchors, 373 tables), and its tests' expectations
      were each rendered by GitHub first. `tests/test_doc_structure.py`
      runs it over every tracked file. It found three rows whose unescaped
      `|` dropped cells (ENHANCEMENTS E23 and E54, MARGINAL-LAGS-AND-BINS'
      `E[y | x]`, escaped now), two example links whose target was `...`
      (PHRASING, WRITING, escaped), and eleven tables in this plan that
      GitHub showed as code, moved from six spaces to four. **For the
      user:** the same cause shows most of this plan as code on GitHub. An
      entry's text starts at column 2, after `- `, so a paragraph that
      follows a blank line at six spaces is an indented code block: 35
      blocks, 2,662 lines. Four spaces would render them as text; the diff
      is whitespace across the whole file, so it is not made without a word.
      **The rest done 2026-09-27.** The mutation baseline over `lasso.rs`
      and `ewridge.rs`: 906 mutants, 104 missed and 152 timed out; two
      `--iterate` rounds leave none that is neither caught nor recorded
      with its reason (14 equivalents; TESTING, "Mutation survivors"). The
      new tests use third-party oracles (the user asked, 2026-09-27: "is
      there no third party oracle library?"): `faer` solves the Rust
      oracle's normal equations, and scikit-learn's `Ridge` holds every
      `ewridge` solve in the Python suite, 20 cases to 1e-8; `cargo mutants`
      sees only the Rust tests, so both. The changed lines of task 112's
      performance fixes: 42 of 43 caught, the other equivalent. Coverage
      (`scripts/coverage.sh`): 96% Python, 93.9% region and 92.6% line
      Rust, from 75% and 73% on 2026-08-30. **Rust coverage does not join
      CI:** it sees `cargo test` alone, so `online-py` reads 0% and
      `online-polars` less than the Python suite drives; it would add an
      instrumented build a run; and the mutation jobs, on every push and
      weekly, are the sharper measure of the same thing. CI keeps the
      Python figure it reports.

- [x] 114. **Four formulas checked against their papers.** S–M; needs the
      papers (none is in `.cache/research/papers`). WKD 2012's Bartlett
      kernel (`corrchange` weights lag `l` by `1 − l/(γ+1)`; the transcription
      reads `1 − l/γ`; a change moves `corrchange`'s goldens); CKP 2010 §3's
      exponent for `rcov`'s `psd` window; ABK 2023's robust posterior for
      `bocpd` (*decision if `robust`'s numbers move, or it becomes a new
      name*); Wied & Galeano 2013's sequential detector (paywalled; *a new
      `corrchange` kind, a decision*). **Checked 2026-09-28 (the user: "Check
      against papers that are not paywalled"; then, of Wied & Galeano,
      "there seem to be many others that describe it exactly").** Every
      paper is in `.cache/research/papers` with its text (PDFKit): WKD from
      the Ruhr-Universität Bochum copy, CKP as arXiv 2602.19645, ABK as arXiv
      2302.04759, and Wied & Galeano from their own open preprint, SFB 823
      Discussion Paper 12/2012 (TU Dortmund Eldorado, version of 12 March
      2012; the journal version may differ in detail), cross-checked against
      Dette & Gösmann (arXiv 1802.07696, §5.1) and Pape, Galeano & Wied (SFB
      823 DP 7/2017), which restate its boundary. Four results:
      (1) **WKD's kernel: the code departs, the transcription was right.**
      Appendix A.1 writes `D̂₁ = ΣₜΣᵤ k((t−u)/γ_T)VₜVᵤ'`, `k(x) = 1 − |x|` on
      `|x| ≤ 1`, `γ_T = [log T]`: lag `l` at `1 − l/γ`, lag `γ` at 0.
      `corrchange` uses `1 − l/(γ+1)` over lags `0..=γ` (Newey–West), in
      `long_run_sd`, `scalar_stat` and the tests' oracles. Measured with an
      independent numpy statistic that reproduces the library's `stat` to
      1.7e-14 on real spans, 20,000 shared draws a cell: the size at WKD's
      Table 1 cells (shared-scale `t_5`, 5 %) is 0.0397/0.0398, 0.0336/
      0.0334, 0.0389/0.0394 at `T = 500` and 0.0388/0.0396, 0.0353/0.0358,
      0.0382/0.0384 at `T = 1000` for `ρ = −0.5, 0, 0.5` (current/paper;
      WKD 0.040, 0.035, 0.041, 0.038, 0.034, 0.039); the power at Table 2's
      break (0.5 → 0.7 mid-span) is 0.5479/0.5509 at `T = 500` (WKD 0.587)
      and 0.8132/0.8151 at 1000. *Decision:* follow the paper (recommended:
      the model is documented as WKD's test, and the change is neutral to
      slightly better) -- `corrchange`'s goldens move, and `bandwidth = 1`
      becomes lag 0 alone -- or keep Newey–West and say so, as the module
      docs now do. (2) **CKP §3.4: the code agrees; nothing moves.** Eq. 16
      `k_n/n^{1/2+δ} = θ + o(n^{−1/4+δ/2})`, `0 < δ < 1/2`; Eq. 17 has no
      bias term; Theorem 4 (ii) calls `δ = 0.1` optimal (rate `n^{−1/5}`).
      `rcov`'s `⌈θ·n^{0.6}⌉` with the bias dropped is that; the "second-hand"
      caveat in `window_for` is gone. (3) **ABK: `robust` is not it, and the
      exact posterior does not fit a streaming model.** Theirs is Gaussian
      over the natural parameters, `Σ⁻¹ += 2ωΛ(x)`, `μ = Σ(Σ⁻¹μ − 2ων(x))`
      (Prop. 3.1, §3.4), robust through `m(x)` built from a reference `θ*`
      (Prop. 3.2) that they take as the MLE on the whole data set, with `ω`
      tuned by KL-matching the standard posterior on the first `t*` rows by
      automatic differentiation; the predictive is closed-form for a Gaussian
      with a changing mean, and with mean and variance both unknown (every
      `bocpd` emission) they sample it (App. C.1). *Decision:* keep `robust`
      as the β-power-weighted variant it is documented to be and drop the
      "swap ABK in" follow-up (recommended), or build ABK as a new emission
      with a warm-up `θ*`, a given `ω` and a seeded sampled predictive.
      (4) **Wied & Galeano: the detector, exactly.** `V_k = D̂·(k/√m)·
      (ρ̂^{m+k}_{m+1} − ρ̂^m_1)` (Eq. 1; `D̂` the inverse long-run standard
      deviation, WKD's A.1 on the `m` historical rows), stop at `τ_m = min{k
      ≤ [mT] : |V_k| > c·w(k/m)}` (Eq. 2), `w(b) = (1 + b)(b/(1 + b))^γ`,
      `0 ≤ γ < 1/2` (Eq. 5; Dette & Gösmann and Pape et al. add a floor
      `max(·, δ)` inside), `c(α)` from `P((T/(1+T))^{1/2−γ} sup_{0≤s≤1}
      |W(s)|/s^γ > c) = α` (Eq. 7), their Table 1 at 5 %: `γ = 0`: 1.2870,
      1.5578, 1.8158, 1.9980; `γ = 0.25`: 1.8001, 1.9924, 2.1684, 2.2467;
      `γ = 0.45`: 2.6282, 2.6844, 2.7215, 2.7660, for `T = 0.5, 1, 2, 4`; the
      changepoint `argmax_j D̂(j/√τ_m)|ρ̂^{m+j}_{m+1} − ρ̂^{m+τ_m−1}_{m+1}|`
      over the monitoring rows alone (Eq. 8). Their Table 2 (GARCH pairs, 1000 draws): sizes
      0.047–0.087 at `γ ≤ 0.25`, 0.106–0.174 at `γ = 0.45`. *Decision:* a
      `kind = "sequential"` (a training span of `m` rows, then the detector
      over at most `[mT]`), or not. It shares (1)'s kernel.
      **Decided 2026-09-28 (the user: "Your recommendation on WKD 2012 and
      abk 2013, build wied & galeano 2013 has a corrchange be sure to
      document well") and built.** (1) The kernel is WKD's, `1 − l/γ` over
      lags `0..γ` (`long_run_sd`, `scalar_long_run_sd`); the oracles in
      `the_statistic_is_its_definition` and
      `the_scalar_statistic_is_its_definition` failed on the old kernel;
      REGIMES §2–4 re-measured (size moved by at most 0.001 a cell, power by
      0.009), with dated notes keeping the old figures. (3) `robust` stays,
      ABK is not built; `bocpd.rs` says why. (4) `kind = "sequential"`:
      `span_rows` rows of history (`m`), then up to `monitor_rows` rows
      (default `span_rows`, `T = 1`) each tested as it arrives, `stat =
      max_pairs |V_k|/w(k/m)`, the cycle ending at a flag or its last row;
      `boundary_gamma` (default 0); `scalar` supported (the mean of `u`);
      `crit` overrides. The critical value is Eq. 7 with `q_γ` from
      `crates/online-core/src/boundary.rs`: the series of `sup|W|` at `γ =
      0`, and above it the law solved, not simulated -- `U(t) =
      e^{t/2}W(e^{−t})` is a stationary OU process, so `Z_γ ≤ c` is an OU
      path inside the corridor `±c·e^{(1/2−γ)t}`, a Fokker–Planck equation
      on `[−1, 1]` with absorbing ends (Scharfetter–Gummel, Crank–Nicolson;
      matches the `γ = 0` series to 3e-6), within 0.03 of W&G's Table 1 and
      above it in 11 of 12 cells, cached per `(α, γ)` and recomputed on
      load (not state). Eq. 8 dates a flag as `since_change`, a new output
      of every kind (`"monitor"`: after the CUSUM's argmax; `"window"`: the
      second window). Size on W&G's GARCH design within two standard errors
      of their Table 2 in all 24 cells (REGIMES §9, `regime_experiments.py
      sequential`). With it, a parameter that belongs to another kind is
      refused (`crit`, `reset` and the permutation knobs under `"monitor"`
      were taken and ignored, though the docstring promised `reset` raised),
      and a `scalar` monitor saved mid-span now loads (the shape check held
      its `u` rows to the feature count). `SCHEMA_VERSION` 20 carries the
      monitoring period.

- [x] 115. **Decisions on behaviour that is built.** Each is S–M once
      decided; the evidence goes with it. (a) `share_p`: VALIDATION §4 shows
      sharing better on both targets, and this plan keeps `False` because it
      hurt one — re-measure, then decide. **Decided 2026-09-28 (the user:
      "Keep share p off"):** `False` stays; the re-measurement (VALIDATION
      §4, 1.215e-6 against 1.279e-6 and 4.963e-6 against 4.987e-6) is on
      data with no predictive signal. (Task 172's relative prior moved them
      to 1.214e-6 against 1.277e-6 and 4.941e-6 against 4.984e-6, sharing
      still better on both.) (b) The solve cadence by
      accumulated weight: `halflife=1e12` solves once (a state field; the
      next schema bump). **Decided 2026-09-28 (the user: "Do 115b") and
      built:** with no `solve_every` under a finite halflife, `ewridge`,
      `lasso`, `huber` and `quantile` solve once the weight learned since
      the last solve reaches `ln 2 / 50` of the weight the fit holds
      (`online_core::DEFAULT_SOLVE_SHARE`; `n_eff`, and `robust`'s raw
      weight), which on evenly spaced rows in steady state is `halflife /
      50` of clock (at halflife 500, 10.007 rows against 10). An explicit
      `solve_every` keeps its clock; `lam` and an infinite halflife still
      solve every row; `max_rows_between_solves` still caps. The share is
      configuration the bank sets from the spec on build and on restore,
      and the counter is state: `SCHEMA_VERSION` 20, a 19 file loading
      with it at 0 (`tests/test_solve_cadence.py`; the oracles in
      `tests/reference.py` and `tests/reference_paths.py` run the rule).
      Measured on VALIDATION §1's data: the clock cadences of 9, 10 and 11
      rows give MSE 1.1741, 1.1617 and 1.1606 (×1e-6), the weight rule
      1.1704 -- phase noise on a stream whose every fit has a negative
      R², so the report cannot rank them; solving every row is the worst,
      1.2255. (c) §12: a weight-0 row after the decay underflows
      keeps the history, where `decay(0)` forgets it (recommended: forget)
      — *a clock gap: reviewed with task 120*. **Decided (forget) and built
      2026-09-28 with task 120**, which records `hmm`'s subnormal edge,
      left as is by the user's call. (d) Task 80's raised calls: the
      256 MiB window budget (**decided 2026-09-28**, the user: "256 mib is
      a good default window budget and on failure it should explain how to
      raise it" -- kept, and the refusal now names the default when the
      spec set none and spells each way out, `{"refuse": MiB}` larger or
      `inf`, a larger `window_every`, `{"thin": MiB}`;
      `over_budget_tests`); a budget-refused bank refusing every later
      call (the user asked how the other budgets handle it: every other
      one -- `gram_block_rows`, `marginal`'s `bin_budget` -- is sized from
      the configuration and refuses before a row is learned; only the
      window's depends on the rows. Options raised: keep the broken bank,
      or a pre-pass over the chunk's clock, as the backwards-clock check
      makes, that predicts the ring and refuses the chunk untouched --
      recommended, with a test holding the prediction to the ring;
      **decided 2026-09-29**, the user: "115.1 follow your suggestion", the
      pre-pass, **and built**: `Stream::window_prepass` replays the chunk's
      schedule on shadows of every ring -- `online_core::WindowShadow`, the
      ring's own `offer` and `trim` on byte counts -- before any stream
      runs, so the chunk is refused whole and the bank goes on; held to the
      ring's own verdict in `crates/online-polars/tests/window_prepass.rs`,
      seven cases at four chunkings, the same piece and the same words
      against a bank run without it, whose reset and close cases fail with
      either path switched off; under `drift_action = "reset"`, whose
      resets depend on the residuals, it abstains and the broken bank
      remains); C24 part 2, `ftrl`'s penalties under a halflife (the user asked
      whether river covers it: only at `halflife = inf`, where we are river
      to the bit; river has no forgetting, so a finite halflife is ours, and
      the build is river's formula with river's constant penalties on
      decayed sums, whose cost is the shrinkage C24 measured; undecided.
      **A second oracle, 2026-09-29** (the user: "find an alternative
      oracle for ftrl that supports more options"): Vowpal Wabbit's
      `--ftrl`, in the dev group, holds `pred` and `coef` on every row to
      1e-5, its single precision (measured 2.6e-6), under both losses, the
      intercept, row weights with zeros, null targets, two targets and `l1`
      with `l2`, where river's covers the logistic loss and the penalties,
      on the state recursion alone (`TestFtrlIsVowpalWabbits`). The user
      allowed it a CI leg; none is needed, since 9.11.9 ships wheels for
      3.12 to 3.14 on all three OSes (reported as missing for 3.14, read
      from the classifiers, which stop at 3.13). None of river, VW or
      Keras's `Ftrl` forgets, so the penalty question stays open. **Decided
      2026-09-29** (the user asked whether to change the behaviour only
      under the EWMA, then: "Build it"): not a switch keyed on the decay,
      which is discontinuous in the halflife -- at `halflife = 10^6` the
      weight-scaled penalties would be near zero for 1.4M rows, full at
      `inf` -- but a per-target scale `m = W/W*` on `beta/alpha`, `l1` and
      `l2`, `W*` the target's weight on a clock that runs only on the rows
      that teach it; the steady state stays the prior against the window
      (4.65 at `halflife = 100`, 0 bias at `inf`), and only the drift of a
      fit with no data goes. **Built:** a row that teaches nothing -- the
      target absent, weight 0, a refused label -- leaves every coefficient
      and prediction to the bit, through a total gap and 100,000
      halflives, where 100 clock units had taken the fit to 0.748 of
      itself at `halflife = 100` (`a_row_that_teaches_nothing_leaves_the_fit`);
      the decay waits per target until the next row that teaches it, so
      the sums cannot underflow; held to the decayed form written out, to
      1e-12 across gaps of 1 to 2,000 clock units and weights of 0 to 2
      (`the_penalty_scale_is_the_longhands_across_gaps`), and to
      `ftrl_ref` moved to the same rule (the Python oracles, with nulls and
      zero weights under halflives of 150 and 100). At `inf` the fit is
      river's and VW's as before; the two golden signatures at halflife 40
      moved; state schema 20 gains three vectors per target, a 19 state
      loading with `m = 1`); S30,
      `holt` with no level-only mode (**decided 2026-09-29**, the user:
      "115.3 do your suggestion", **and built**: `holt(trend=False)` holds
      the trend at zero for a flat forecast, simple exponential smoothing,
      whose level is the EW mean of the observations -- held to pandas'
      `ewm(times=)` at 2.2e-12 and to `DescrStatsW` at 5.6e-16
      (`TestALevelOnlyHoltIsAnEwMean`), and to its definition in `holt.rs`;
      `trend_halflife` is refused beside it, and a config without the field
      loads with the trend on, so no schema bump); S31, `n_eff` counting rows with a null target
      in `ftrl`, `pa` and `sgd` (hard rule 8). **S31 decided 2026-09-28
      (the user: "n eff should do what it does everywhere else") and
      built:** measured first, every model emits the shared weight as
      `n_eff` and every regression model but these checks a target's
      `min_periods` against its own weight (ten null-target rows, then the
      target, `min_periods = 2.5`, halflife 200: `ewridge`, `lasso`,
      `huber`, `quantile` and `kalman` first predict on row 13, `holt` too
      once its level is set); `pa`, `sgd` and `ftrl` predicted from row 3 on
      coefficients no target had moved, and `rls` from row 11. So `pa`,
      `sgd`, `ftrl` and `rls` now keep `w_target`, the weight of the rows
      that carried each target (for `rls` the rows it learned from, all
      targets present; for `ftrl` not a label `strict_binary` refuses), and
      gate each target on it in `step` and `predict`; the emitted `n_eff` is
      unchanged. `SCHEMA_VERSION` 20 (with (b)); a 19 state loads with each
      target at the shared weight. Pinned by
      `test_a_target_counts_toward_min_periods_only_on_rows_that_carry_it`
      (the whole sweep) and a test in each model's file; the oracles
      (`rls_ref`, `ftrl_ref`, `rls_paths_ref`, `pa_ref`, `sgd_ref`) gate the
      same way. `sgd`'s `inv_scaling` rate still reads the shared weight.
      (d) is done: `ftrl`'s penalties, its last item, are built above. (e)
      The 2026-09-12 review's
      sign-offs (P4's calls; decisions 1, 3, 4 and 5; D1's caveat to rule
      5) and REVIEW-2026-09-18's unrecorded S1, D2 and B7. **Approved
      2026-09-29** (the user: "115.5 approve all"): decisions 1, 4 and 5
      stand, 3 went with task 136, P4's calls are (d)'s two budget items,
      and S1, D2 and B7 stand; D1's caveat is in `CLAUDE.md`'s rule 5, and
      each review document records its sign-off. (f) S18, Bartlett
      weights for `serial_rule`; S19, windowed target moments; C18, a
      windowed lag ring — record as tasks or decline. **Decided 2026-09-28
      (the user: "Accept tasks s18, s19 as long as it doesn't add much
      extra size, c18 all to be done after the current work"):** tasks
      135, 136 and 137. (g) C8, the CLI's
      NDJSON on the system allocator: write it from one thread (the
      mimalloc option is parked with the new libraries). **Built
      2026-09-28** as low-hanging (the only fix not parked), after
      re-measuring at HEAD on the three-spec example bank over 3M rows: the
      slices ran 4.98–5.50 s with 2.1 s of system time and no tail, polars'
      `BatchedWriter` on the writing thread 4.65–4.78 s with 1.1–1.2 s, and
      parquet 4.65–4.73 s; so the one-thread writer is both the simpler and
      the faster now (`ndjson_write`). (h) `deco` with a
      column that has no spread (found by task 112's test, 2026-09-27): a
      row learns only when every standardized value is finite, so while one
      column is constant -- from its first row, the only way its variance is
      exactly zero -- no block's `rho` moves, its own or any other, and `u`
      and `loglik` are null for every block; `n_eff` advances and every block
      learns again once the column moves
      (`test_a_constant_column_poisons_nothing_and_every_block_learns_once_it_moves`
      pins that much). Whether the other blocks should go on learning, and
      the constant column's own block too (from its other columns), is the
      decision. **Decided 2026-09-29** (the user: "115.4 your
      recommendation"): each correlation value keeps its own weight, a
      column with no standardized value is left out of the row's sums, and
      `loglik` is null on such a row. **Built 2026-09-29:** while a column is
      flat its block reads what the model without that column reads, `u`
      and `rho` to the bit, and a block's value is the one-block model over
      its own columns on every row
      (`a_column_without_spread_is_left_out_of_its_block`,
      `each_value_keeps_its_own_weight`,
      `test_a_constant_column_is_left_out_and_every_other_value_learns`);
      a value with no estimate neither learns nor decays, the rule the
      whole row had. `rho_w` is one per value in schema 20, and a 19
      state's single weight reads as that weight on each. Also (review
      2026-09-28, pre-existing): under `dynamics =
      "linear"` a zero-weight row moves `rho` (`deco.rs`: `(1-α-β)·bar +
      α·u + β·rho` is recomputed whether or not `b` is 0), against hard
      rule 9; whether the linear form should hold `rho` still on such a row
      is a decision. **Decided 2026-09-28 (the user: "Hold rho still on a
      zero-weight row") and built after 0.12.0:** `rho' = rho` at `w = 0`
      under `"linear"`; positive weights still reach `rho` only through
      `rho_bar`, which the docstring states (options not taken: scaling the
      `alpha` step by the weight, or documenting only). (i) The Linux CLI's glibc floor (task 109's leftover,
      measured 2026-09-27 on 0.11.1's assets): both binaries need
      `GLIBC_2.39`, the runner's own -- `pidfd_getpid` and `pidfd_spawnp`
      from Rust's standard library, weak symbols under a version requirement
      that is not -- and hard `2.34` and `2.35` (`__libc_start_main`,
      `hypot`); so no Ubuntu 22.04, Debian 12 or RHEL 9, and the floor rises
      with `ubuntu-latest` (RELEASE-READINESS has the table). Options: build
      the CLI in the manylinux2014 container the wheels use (glibc 2.17,
      still dynamic, nothing newly linked; recommended); build it on
      `ubuntu-22.04` (2.35, until that image retires); or a static musl
      binary (static linking, rule 12, a raise). Each changes the release
      job, which only a dispatched run tests. **Decided 2026-09-28 (the
      user: "follow your recommendation on the next release") and built:**
      the Linux CLI is built by `docker run` in
      `quay.io/pypa/manylinux2014_$(uname -m)`, and
      `scripts/glibc_floor.py` fails the build job above 2.17 before any
      upload (`tests/test_release_workflow.py` holds the shape; the script
      refuses 0.11.1's binary at 2.39). The 0.12.0 rehearsal built it (run
      36448809775 on `36e468c`: 2.16 on x86_64, 2.17 on aarch64), and 0.12.0
      shipped it.

      **Done 2026-09-29:** every item, (a) to (i) and S30, is decided and
      built.

- [x] 116. **Readiness beyond `ewridge`** (WARMUP-AND-CONVERGENCE §7). M–L;
      defaults move, so *the scope is the user's*. A readiness statistic for
      `rls`, `kalman`, `lasso` and the scalar EW models; `support_coef` for
      `lasso`, `ew_cov` and the robust models; `n_eff_settled` in the
      summary; a warning where `min_periods` can never be met, against the
      ceiling `1 / (1 − λ^d)` (task 148 corrects the docs' figure); the
      inflation notice's halflife figure; §7.1's coefficient
      standard errors (its CUSUM is task 146's); §7.7's flicker; §7.9's CLI
      closing count. *Checked 2026-10-07*: task 198 (`54975d9`, D8) built
      the summary's `weight_sum_settled`, the warning where `min_weight`
      can never be met and the inflation notice's half-life figure, and
      declared the current floors final; task 146 closed without the
      CUSUM, which nothing owns. The rest is open; the user wants it built
      before 1.0, scope to follow from a summary (§18, 2026-10-07).
      *Done 2026-10-08* (the user, 2026-10-07: "Your reco on 116"; branch
      `task116-readiness`, integrated after review 5): A `max_error_inflation`
      and `emit_error_inflation` on `rls` (`s₂` in the state, schema 45; the
      gate `√(1 + k/n_Kish)`, the row's leverage) and `kalman` (each row's
      exact `√(1 + z'P⁻z/R)`, the gate per row), the gate on `lasso` (`df` =
      the active count plus the intercept); B `support_coef` on `huber` and
      `quantile` (`1 − λ(A⁻¹)_jj` on the band system, persisted); F `se_coef`
      under `emit_se_coef` on `ewridge`, `rls` and `kalman` (`diag(T Cov Tᵀ)`,
      a report); G the support warning under the notices' one 95%-settled
      rule (a bug: 8 seeds of 12 warned falsely at the first row through);
      H the CLI's closing line. The "cannot be met" notice is off for
      `kalman`: its gate is per row and `P` settles on `coef_half_life`'s
      clock, not the spec's decay, so a notice tested at ±5% cannot hold
      (§19, 116-K; WARMUP §7.11). `tests/test_window_budget.py`'s spec was not
      §2.2's true positive: on the base it read `support_coef` 1.00 and never
      warned, so its comment and §2.2 were corrected and no code changed for
      it. `kalman`'s steady-state floor, measured against the brief's
      `√(1 + k·p)`: 1.0715 against 1.0675 at `coef_half_life` 50 (`h` 6%
      above `k·p`), 1.84 against 1.58 at 5 (60% above). Costs a row: `huber`
      +5% and `quantile` +4% by default (`support_coef`); the `kalman` gate
      +115-120 ns when set; `rls`'s `s₂` within the noise; `se_coef` +10 ns
      on `ewridge` at the default cadence.

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
      together.

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

- [x] 119. **Housekeeping.** S; *the user's call*. The 13 remote branches
      merged into `origin/main` (three point at release tags) and about 40
      local branches, all merged. **Done 2026-09-29** (the user: "Do task
      119"): the 39 local branches went first; the 13 remote ones, each
      checked merged into `origin/main` and with no open pull request, were
      deleted with `git push origin --delete`. The tags three of them
      pointed at, `v0.2.0`, `v0.9.0` and `v0.9.1`, remain; the remote holds
      `main` alone.

- [x] 120. **Rows out of order and clocks that go back: what an audit of
      2026-09-25 found. Needs extra review before any of it is built (the
      user, 2026-09-25).** One read-only audit traced every mechanism that
      detects or absorbs disorder — `on_clock_reset` in `ClockState::advance`,
      `min_backwards_jump`'s chunk pre-check, sessions, the query check, the
      per-group clock, `group_close="monotone"`'s key check, `label_delay`'s
      release, the summary's `clock_backwards`, `refresh_time` and `embargo`
      — and the two findings marked *reproduced* were run on the build
      (scripts in the session's scratchpad, not in the repository).
      - **An infinite cap turns a clock step back into an infinite step —
        CONFIRMED, reproduced.** Under `max_dclock=inf` the order check is
        off by default, and `"max"` hands the models a step of `+inf`
        (clock.rs:356-362, 398). `holt` predicts null on every row after
        the step, for good (0·inf in its level and trend, holt.rs:208-235);
        `kalman`, `ewridge` with or without `label_delay` keep a non-finite
        value in their state, which the JSON export refuses (`decay_time`
        and `settled_frac`, stream.rs:3686; `pending_clock` inf − inf,
        3689-3691); the readiness warning then claims every prediction is
        withheld for good while later rows still predict. `session_gap=inf`
        under an infinite cap does the same. REVIEW-2026-09-18 V2 raised it
        and closed it because `max_dclock` is required — which `inf` meets.
        Input: `clock="t"`, `max_dclock=inf`, t = 0, 1, 2, 3, 1, 2, 3, 4.
        test_clock_order.py:152-155 accepts this pattern for `ewridge` and
        checks only the row count.
      - **A minimum equal to the clock's smallest step never fires —
        CONFIRMED, reproduced.** The comparison is strict (clock.rs:346)
        and the "would never fire" refusal tests `v < tick` (arrow.rs:595):
        a `Date` clock with `max_dclock="1d"` absorbs the days 1, 3, 2
        silently, where `"2d"` refuses the same stream. Its refusal message
        reports the step in seconds (`86400`), not as a duration.
      - `max_dclock=0` turns the check off by default too (spec.rs:2065-2071);
        the README names only `inf` (CONFIRMED).
      - V14's test was never written: nothing steps the clock back at a
        session change with `session_gap` unset (clock.rs:325-328) (moved
        here from task 112).
      - Chunk invariance with a step back that a policy absorbs is untested:
        the property generators make forward clocks only, though
        test_properties.py:4 says otherwise; one chunk boundary is tested,
        and a refusal whose earlier row sits in the previous chunk only
        under `"error"`.
      - Docs and comments that contradict the code: PLAN:202-203 and
        ENHANCEMENTS:41 ("errors loudly"); bank.rs:528, 2698-2700, 2842-2847
        and stream.rs:2334, 2393, which call the pre-check `"error"`-only;
        ARROW-SOURCES:601-604, 684-686 (out-of-order rows refused — only
        small steps are); README:2703; WARMUP:582; "jitter check" in test
        comments.
      - An error's row number counts from the start of the chunk in the plan
        form, the CLI and `refresh_time` (_frame.py:648-651, runner.rs:763,
        refresh.rs:247), where the README implies the input row.
      - Resuming onto input that overlaps the saved state learns it twice:
        the rerun's first row is at least `max_dclock` behind, and `"max"`
        absorbs the step silently; STATE-WORKFLOW says nothing of it.
      - PLAUSIBLE: the query check takes any `Sort` as fixing the order
        (_frame.py:325-326), but `sort` without `maintain_order` leaves ties
        unordered and equal clocks pass the clock check; `embargo` merges
        by clock alone, so it needs the global clock order where a stream
        needs each group's, and its docstring says "as a stream must be";
        `refresh_time` casts the clock to f64 (refresh.rs:228), so on a
        `Datetime(ns)` a step back of under 256 ns can round to a tie.

      **The planned work that touches this area**, all designed before
      `min_backwards_jump` existed (2026-09-20), none of which mentions it
      (task 106, which would have made a backward clock a session change,
      is discarded): task 78's column form (it closes
      every window at a step back in any group, where the bank checks each
      group; `advance` records the new clock even when it refuses, so the
      window core must work on a copy and commit it; an IO source that
      refuses mid-stream has already emitted rows); task 104's parity tests
      (their streams must step back by at least the cap or switch the check
      off); task 105 (one vocabulary run by the same clock code, and
      `load_state` on `refresh_time`, imply an exact temporal clock there
      and a saved last time per group, checked on resume, which no sub-task
      owns).

      **Decided 2026-09-28 (the user): `"max"` is removed from
      `on_clock_reset`.** `max_dclock` is a cap on a step and nothing else
      (the user: "Max dclock is meant to be a cap on dclock, where is it used
      as a default?"). It had two other uses, and both confirmed findings
      above come from them: the step a backward clock took under `"max"`
      (clock.rs:356-361), infinite under an infinite cap, and the default of
      `min_backwards_jump` (spec.rs:2202-2210).

      **Decided 2026-09-28, every clock question (the user: "Resume should
      get a helper. Remove inf as an option for max dclock. Remove zero
      option. The rest as you suggest."):**
      - `max_dclock` is required with a clock and must be finite and
        positive: `0` is refused (it turned decay off, and every positive
        gap then read as a break, so `label_delay` released every held
        label on the next row -- measured: learning from row 1 where a
        delay of 3 starts at row 3), pointing to `halflife="inf"` or no
        clock; `inf` is refused (the cap is also what a gap is a break
        against), pointing to a large finite cap.
      - `on_clock_reset` keeps `"error"`, now the default, and
        `"reset_state"`; `"max"` and `"zero"` are removed and refused by
        name.
      - `min_backwards_jump` has no default derived from the cap: it is
        required with `"reset_state"` (the caller says what a late row is)
        and refused with `"error"`, where every step back is refused anyway.
        A step back no larger than it is a late row: the comparison is
        inclusive, and the message states the step as a duration.
      - `session_gap` is finite or `"reset"`; `inf` is refused, pointing
        to `"reset"`. No model ever receives a non-finite step, which a
        contract test holds for every model.
      - A zero-weight row after the decay underflows forgets, as the decay
        does (task 115 (c)).
      - Resuming is the next chunk: an overlap takes the policy, which by
        default refuses it. A helper returns each group's last learned
        clock so a caller can skip what the state has learned, and
        STATE-WORKFLOW says so.
      - For later tasks: the column form (78, 104) checks disorder per
        group, as the bank does; task 104's layout change gets no loader
        (the pre-1.0 waiver of 2026-09-14).
      - `sort` without `maintain_order` in the query check, and `embargo`'s
        global order: a test first; changed only where it fails.
      - With no decision needed: the session-change test (V14), property
        tests whose clocks step back, the docs and comments that contradict
        the code, error row numbers counted from the input's start, and
        `refresh_time` reading a `Datetime` in integer nanoseconds.

      **Built 2026-09-28**, every decision above but the two recorded for
      later tasks (the column form's per-group rule, 78/104; no loaders for
      104). The user added, while it was built: "Do not worry about old
      specs" -- so a bank file before schema 19, which names `"max"` (the
      old default, always written), is refused by its version
      (`MIN_BANK_SCHEMA_VERSION`, bank.rs); a model's own state from 14 on
      still loads, no model state having changed layout.
      - The clock (clock.rs): `OnClockReset { Error (default), ResetState }`;
        the late-row test is `back <= min_backwards_jump` under
        `ResetState` only; `advance_scoring` gives a step back a step of 0
        and never refuses or resets, which `predict` uses. A proptest,
        `finite_steps`, holds every step finite and within the cap over
        every configuration the spec lets through.
      - The spec (spec.rs `clock_cfg`): the cap finite and above 0 (0 names
        `halflife = "inf"`), `session_gap` finite or `"reset"`, the minimum
        required with `"reset_state"` and refused with `"error"`. A clock's
        refusal message gives a temporal step as a duration (`1d`, not
        `86400`). `clock_scale` names a rate (`lam`, `q`) before a plain-number
        cap, since a finite cap now always sits beside it.
      - Resume: `ModelBank.skip_learned(frame)` (DataFrame or LazyFrame),
        from `Bank::last_clocks`, exact in nanoseconds; STATE-WORKFLOW
        "Resuming on input that overlaps the state".
      - 115 (c): the zero-weight row after an underflowing decay forgets, in
        `EwCov`, `EwDiag`, the gaps accumulators and `deco`; a contract test
        over every model compares it with the row one halflife short.
        **Found and raised, not fixed:** `hmm` one halflife short -- a
        zero-weight row at a subnormal decay factor, from about 1025
        halflives -- leaves its co-moments and precision prior a few bits,
        and its densities NaN for up to 39 of the next 40 rows (measured at
        1030 and 1074 halflives; 1000 and 1075 are fine). A threshold that
        forgets a subnormal share (2^-1022) fixed 1030--1074 but not 1025;
        one that would make it impossible (2^-53) changes shipped output at
        about 57 halflives, a weekend on an hourly halflife. **Left as is
        (the user, 2026-09-28: "very much at the edge")**: it needs a
        zero-weight row about a thousand halflives on. The contract names
        `hmm` where it skips the comparison.
      - Rows are counted from the input's start: `ArrowChunk::row_base`,
        `Bank::fit_predict_from`/`predict_from`, the plan form, `fit`,
        `fit_predict_batches`, the CLI and `refresh_time`.
      - `refresh_time` compares a temporal clock as the column's integer.
      - Test first, then: a sort by **several keys** without
        `maintain_order` reordered 2,109 of 10,000 rows against the stable
        sort (polars 1.44.2), so the plan check reports it; by one key none
        moved, pinned by `test_the_measurement_behind_the_sort_warning`, so
        a change in polars shows. `embargo` on a frame sorted only within
        its groups returned a group out of order: its docs now say it needs
        the clock order across all rows and name the remedy; the bank
        refuses the result by default, so it is not silent.
      - V14's session-change test and the stepping-back property tests
        (`test_properties.py`) are written.
      - Reviewed 2026-09-28 (two readers over both commits): the clock and
        accumulator changes held; three fixes followed in the review commit:
        `skip_learned` compared a temporal clock scaled to nanoseconds in
        polars integer arithmetic, which wraps past 2262 (compared in the
        column's own unit now); `refresh_time`'s state did not record its
        grouping, nor check the clock's kind on a new group's row (both in
        the file now, held on load and on every row). One divergence for the
        user: under a pushed-down `head()` with `save_state`, the bank saves
        the state after the rows the query pulled while `refresh_time` feeds
        the whole input and saves that; task 105 asks for one rule.
        **Decided 2026-09-28 (the user: "Yes") and built after 0.12.0:** the
        bank's rule for both, the state after the input behind the rows
        returned. `refresh_time` stops at the tick that completed the n-th
        point (`RefreshTime::feed_limited`), so the state is the same at any
        chunk size and a resumed run goes on with point n + 1; with
        `pairs=True` a tick completing several points is taken whole.

- [x] 121. **Test libraries under an open-source licence.** S–M. **Done
      2026-09-25:** scikit-learn 1.9.1 in the dev group; `huber` against
      `LinearRegression` in the exact limit and `HuberRegressor` under
      outliers (T-S4), `marginal`'s bins against scipy's `binned_statistic`
      with values on the edges and its split against a stump (T-S12), the
      reverting `kalman` against filterpy (T-S5 in full), and `sgd`
      against `SGDRegressor` live; proptest 1.11 over all 21 models in
      `model_contract.rs` (a deep run of 2,000 streams a model found
      nothing); and a licence check over the dev and docs groups and the
      crates' dev-dependencies. One measurement worth keeping: at one row
      in ten a gross outlier, `huber`'s intercept sat halfway to least
      squares (1.5 against `HuberRegressor`'s 0.66, truth 0.5), because
      its scale is the plain EW residual spread -- review D4, documented in
      `robust.rs`, not changed here. The user,
      2026-09-25: "we do want to enable every unlicensed library in tests
      and park using licensed libraries" -- read as: a library under an
      OSI-approved open-source licence (BSD, MIT, Apache, MPL, ...), which
      asks for nothing to be bought or accepted, may be used in tests; one
      under source-available or commercial terms (the Business Source
      License, the SSPL, a commercial licence) is parked. Already in the dev
      group and open source: numpy, pandas, statsmodels, river, hypothesis,
      pyarrow, duckdb, the ADBC drivers.
      - scikit-learn as a live oracle: the T-S second opinions (T-S4's
        `HuberRegressor`, T-S12, T-S5 in full), TESTING's scikit-learn
        comparison, and the PERFORMANCE measurements that needed it
        installed by hand (`scripts/sklearn_comparison.py`).
      - Rust `proptest` for T-D2, property tests in `online-core` beside the
        Hypothesis ones. A dev-dependency links into the test binaries only,
        never the package, and hard rule 12 covers production deployments,
        not tests (the user, 2026-09-25), so it needs no raise.
      - The rule where it lives (TESTING, "Libraries the package does not
        depend on") and a check in `tests/test_dependency_policy.py` that
        every dev-group library's metadata names an open-source licence, so
        a licensed one is refused rather than remembered.

**Requests of 2026-09-25 for `marginal` at width** (E70–E74,
`docs/MARGINAL-AT-WIDTH.md`, from the same caller as E66–E67; the user:
"examine its feature requests and integrate them into the current
planning cycle"). Examined against the code, and against a benchmark of
the model alone at `p = 10,000` (a scratch crate outside the repository):
the per-target cost is the model's — `3.3·T` ns per feature per row for
the moments, `19.8·T` with six lags and sixteen bins, against the caller's
`3.2·T` and `19.1·T` — and the caller's fixed 14–17 ns per feature per row
is not, since the model alone has `0.0` and `3.5` there.

      **Review 2026-09-26.** The proptest's `value()` said "near the
      input bound" and reached 1e50 of 1e100, its weights never the bound,
      and no windowed or blocked configuration was generated (C8): values
      now sit at the bound, weights at `1e-100` and `1e100` (which
      `INPUT_BOUND`'s doc names as legal), and `ew_ridge`, `lasso`,
      `ew_cov`, `ew_class` and `marginal` run windowed at both cadences,
      `ew_ridge` blocked with pairwise gaps. That found **G1**: `micro`
      reported an infinite distance and *absorbed* the row. A row of weight
      `1e-100` then one of `1e100` leave a standardized spread of `4e-200`;
      the next row at the bound is `5e199` scaled units away, whose square
      overflows to `inf`; `merged_radius2` read a non-finite square as
      "radius unchanged", so `admits` took the row, and `dist` reported
      `inf`. Fixed in `summary.rs`: an infinite square merges into an
      infinite radius, and `dist` computes the distance without squaring
      (scaled by the largest deviation), read only where the square is not
      finite so ordinary rows keep their bits; `micro` and `kmeans` report it,
      and break a tie of overflowed squares by it -- `kmeans`'s search
      started from infinity, so an infinite square never won and the
      runner-up was never set (`a_row_at_the_bound_against_a_vanishing_spread_is_far_not_absorbed`,
      `an_overflowed_distance_is_reported_as_the_distance`,
      `an_overflowed_distance_is_far_and_still_measured`). And **G2**, the
      same test on the quantile `robust`: a target of `1e100` lifted the
      mean the first off-zero row's slope was read against, the fit
      extrapolated to `-1.5e199` at `x = 1e100`, that residual set the
      scale to `5.8e148`, and the next row outside the band -- one term of
      the score, none of the Hessian -- nudged the slope by `h·dev/var`
      against a Gram holding no curvature that far out, to `1e248`; the
      prediction at `1e100` read `-inf`. The nudge is bounded by the row's
      leverage under the Gram's diagonal, so the fit at the row moves by at
      most its residual (the row is brought at most to its target); the
      linearisation holds inside the band and nothing past it is justified.
      (Task 160, TC1b: the diagonal under-reads the leverage of a row
      against correlated features, and such a row moved 6 to 127 times its
      residual; the bound is the full leverage `1 + u'A⁻¹u` against the
      band's own system since, at about half the default schedule's
      throughput: 2.6M rows/s against 5.0M, 10 features.) The 24 robust
      tests, the QuantReg oracles among them, hold; the quantile golden's
      three values moved by about 1e-3 of themselves, the bound binding on
      early rows where a row's leverage exceeds its weight share. Measured
      on the Python oracle's stream (250 rows, halflife 300): `tau = 0.5`
      and `0.9` unchanged to the bit; at `tau = 0.1` one row (44) is
      bounded, and the fit that follows has the lower pinball loss, 0.10852
      against 0.10940, at coverage 0.170 against 0.161 -- the reference
      recursion in `tests/reference.py` mirrors the bound
      (`a_quantile_fit_stays_finite_through_the_bound`,
      `robust_quantile_through_the_bound`). And **G3**, the same test on the
      windowed `lasso`, though the window had no part in it: `Cross` kept
      each target's own feature mean as an offset `δ_j` from the all-row
      mean `m` and reconstructed it as `m + δ_j`; a target absent on a row
      whose feature stood at `-2.6e99` left both at `1e99`, their sum
      resolved nothing below `1e83`, the next present row's deviation of
      `0.4` read as `1e83`, the cross-moment as `1e132`, and the prediction
      at `1e100` as `-inf`, through `ew_ridge` too. Each own mean is now a
      pair of its own (`Cross::mj`), updated from the target's own rows by
      the Gram's steps, so under `own_rows` it is the Gram's mean to the
      bit and a row the target misses leaves it exactly where it was; the
      `pairwise` readers take it directly. Schema 17; `offsets_to_means`
      converts a 14-16 state's live accumulators and every window snapshot
      once (`a_target_absent_on_a_row_at_the_bound_leaves_the_fit_finite`,
      `a_target_absent_on_a_row_at_the_bound_through_the_bound`). The sklearn
      comparison kept sklearn's default `l2` penalty while our spec has none
      (F9): `penalty=None`. The licence check was read for ways round it and
      found none (F10). The seed: proptest draws afresh each run; a failure
      prints the shrunk stream, and `PROPTEST_RNG_SEED` repeats a run.

- [x] 122. **E71: bin each feature once per row.** S; no API, no state,
      bit-identical. **Done 2026-09-25:** `MarginalBins::update_row` forms
      the row's indices once, in a scratch buffer outside the state, and
      the warm-up replay goes through it too. Held to the per-target update
      to the bit (`the_row_update_is_the_per_target_update_to_the_bit`), and
      to the build before by `marg_bench`'s checksum, absent targets
      included. At nine targets the bins cost 2.6 times less; at one, a
      fifth less, which the request did not expect (PERFORMANCE §22). Found
      on the way: the docstring promised the warm-up replay to the bit,
      which a row of weight zero inside the warm-up breaks by about 1e-15
      of the data's scale. That is the trade the design doc records, so
      the docs now say it and a test pins it. Verified: `update_target` runs `bin_of(&self.edges[j],
      x[j])` once per present target (margbins.rs:253-261), and the edges
      are the feature's. Form the row's `p` indices once, before the target
      loop, and the same on the warm-up replay. Tests as the request lists,
      and `marg_bench`'s shape at `T = 1` and `T = 9`. First: the smallest,
      and exact.

      **Review 2026-09-26.** The row update took the low parts'
      slice for every present target but sized the vector only when some
      feature binned, so a histogram restored from a 0.10.0 state (which
      carries none) panicked on its first all-NaN-feature row with a target
      (B1, a crash from the Rust API; the bank's null rule shields the CLI
      and Python): sized on the first row that could write a cell, which is
      also the rule the sharded path uses, so the two paths' states agree
      byte for byte (A7). `has_shape` now holds the offsets to the edges and
      the low parts to the cells (B2). The scratch row's four-slot minimum
      is counted, so `histogram_bytes` is the buffers to the byte at every
      width (B5). The design doc's cell count, the per-feature search and the
      "other two" said three values a cell (A4, B6). Tests: the brute-force
      tests drive `update_row`, the production update, where they drove the
      test-only per-target one; `the_split_gain_survives_a_fold` pins the
      product-of-ratios claim to 1e-9; `edges_from_gives_edges_a_histogram_accepts`
      is a proptest over junk, ties and a dominant weight;
      `fixed_edges_collapse_against_a_large_level`; and `shard_stream` now
      carries NaN features, all-NaN rows and all-absent targets, so the
      sharded `NO_BIN` paths run under the bit test.

- [x] 123. **E70: `marginal(cross_lags=...)`.** S–M. **Done 2026-09-25**,
      on schema 16 (unreleased), through the spec, the bank's two tables and
      Python; refused by name at construction. The default is bit-identical
      to the build before and 0.9% faster; one loop for both cases had made
      it 1% slower, until the `cross_lags` loop was kept out of line
      (PERFORMANCE §23). One cross lag of six takes 30% off the lags' cost. Verified: `n_serial`
      reads the autocorrelations alone (`serial_factor`, marginal.rs:368-420,
      from `lagcorr_xx` and `lagcorr_yy`), and `cxy`/`cyx` feed only
      `lagcorr_xy`/`lagcorr_yx`. The default, `None`, keeps every lag as
      today, and the kept moments are bit-identical. A state field with a
      serde default is still a layout change under hard rule 5: it rides
      schema 16 if it lands before 0.11.0, 17 after.

      **Review 2026-09-26.** `MarginalLags::has_shape` let a ring longer
      than the deepest lag, or one with fewer target rows than feature rows,
      through: the first read every lag a row too recent for good, the second
      panicked (B3, a corrupt state only); both refused. The module doc said
      the cross terms were two thirds of the lag work; §23 measured 30% (A6).
      Tests: `a_lag_state_with_an_overlong_or_uneven_ring_is_refused`,
      `a_compact_state_without_cross_lags_reads_with_every_lag` (the
      positional upgrade the comment claimed), the spec layer's
      `cross_lags_write_back_as_given_and_need_lags`, and the bins test that
      `cross_lags=[]` drops the two cross columns and moves nothing else.

- [x] 124. **The stream's fixed cost per feature per row at width** (from
      E74's measurement). S–M, and it helps every wide model. **Done
      2026-09-25** (PERFORMANCE §24): it was not the cast or the transpose
      first, but seven lookups by column name that scanned every column,
      quadratic in the width, and the marginal model keeping every pair's
      run for a window it did not have. At 10,000 features on one thread
      the moments-only call went from 818.7 to 192.2 ms at one target and
      from 2,141.5 to 617.7 ms at nine; the fixed part is `2.8` ns per
      feature per row where it was `13.1`, the transpose and the data
      summary. Tiling the summary was measured slower and left out. The caller's
      14–17 ns per feature per row that does not grow with `T` is outside
      the model (above). Profile the bank's path at `p = 10,000` — the
      cast, the tiled transpose (`feature_rows`, bank.rs:169), the per-row
      validity check, the data summary, the output — and fix what
      dominates; at `T = 1` it is four fifths of the row.

      **Review 2026-09-26.** The hash lookups were read against the
      scans they replaced (`find`, `key`, `has`, `first_readers`) and found
      the same answers; `first_readers`' doc now says where a reference
      sits in the role order (D12). `check_clocks`' "a temporal column can
      only be a clock" loop read target names and knew no reference, so a
      Datetime clock used as a table target's column or as its reference was
      refused later, with another message (D6): by column now, and a
      reference is "a relative_to reference"
      (`test_a_temporal_clock_is_not_a_target_or_a_reference_either`).

- [x] 125. **E72: `feature_moments="shared"` — a decision first.** M–L. The
      request says only `var_x` changes where a target is absent; not so.
      The batch identity it cites holds, but `sxy` is kept by a recursion
      centred on the feature's pre-row mean over the target's rows, and
      stepping it from the shared mean gives another `cov` wherever a target
      is absent on some learned rows: 1.8e-3 relative with a random tenth
      absent at `lam = 0.99` (a simulation of both recursions,
      2026-09-25), bit-identical only where every target is on every
      learned row. Under `"shared"`, `cov`, `var_x`, `corr`, `beta`, `t` and
      `n_serial` then all differ there: a different estimator, sound where
      the absence is not informative, which the docs would have to say.
      *Decision: build it as that estimator; build it only where it is
      exact (every target on every learned row, else per-target moments);
      or decline.* Either way the shared means are pairs (task 101), the
      runs go per feature (task 94's windowed read), and a `window` needs
      its own answer. **Decided 2026-09-29** (the user: "Do task 125",
      on the recommendation to build it as that estimator and document
      it). **The pair moments are built:** `feature_moments="shared"`
      (`FeatureMomentLayout::Shared`) keeps `p` means (pairs, task 101) and
      variances over every learned row and `p·T` covariances, unsplit and
      sharded, bins as they were; the pairs are `"per_target"`'s to the bit
      where every target is on every learned row
      (`shared_feature_moments_are_per_target_to_the_bit_where_every_target_is_present`,
      at one target and three, and the sharded stream); where one is
      absent, the feature's moments are its column's `EwCov` over every
      row to the bit and the covariance the stated recursion
      (`shared_feature_moments_are_the_features_over_every_learned_row`),
      measured moving `corr` by at most 0.0065 at a tenth absent. 2.7 times
      as fast at ten targets, 3.2 at thirty, level at one (PERFORMANCE
      §27). A window is refused by name: its subtraction reads a covariance
      centred on the pair's own mean. **The lags are built too:** the
      feature's autocovariance at each lag per feature with the row's mix
      (`MarginalLags::update_shared`, and the sharded job the same way),
      the cross terms and `cyy` per target with its own; the pairs are
      `"per_target"`'s to the bit with every cross lag, some and none, at
      one target and three, and sharded; with a target absent, the
      feature's lag correlation is a `"per_target"` model's whose target
      is on every row, to the bit
      (`shared_lag_moments_are_the_features_over_every_learned_row`). With
      lags at ten targets it runs 4.6 times as fast, 2.5 with one cross lag
      (PERFORMANCE §27).

- [x] 126. **E73: a wide `marginal` sharded across the pool (`shards`).**
      **Done 2026-09-25**, as a batch of rows per fork-join, not one per
      row (§11a has why). `marginal(shards=)` takes a count or `"auto"`,
      off by default; it is a setting, not state. `Marginal::step_sharded`
      advances every target's own numbers as the row arrives and holds
      each pair's share of it; a flush steps the pairs through the held
      rows in order, one `MarginalShard` per range of features, which the
      stream runs on the bank's pool. `online-core` keeps no threads: the
      caller passes the runner. Flushed when 256 rows (or 8 MB of features)
      are held, before a window snapshot, before the bins fold their scale,
      and at the end of each run of rows. The pair update, the lag updates
      and the bins' cell update became slice kernels shared by both paths,
      and the unsplit row got 7–50% faster with every checksum equal. Tests:
      `a_sharded_step_is_the_unsplit_step_to_the_bit` and four more in
      `marginal.rs`, `a_sharded_marginal_reads_the_unsplit_pairs` through
      the bank, and `tests/test_marginal_shards.py` (every count, chunked,
      saved and resumed, `fit(lf)`, the CLI). Measured (PERFORMANCE §25):
      at 10,000 features the model alone ran 1.8× at one target, 4.6× at
      nine and 5.4× with lags and bins; through the bank, 1.2×, 2.0× and
      4.9×, where the bank's own row work does not split. Not split: the
      bins' warm-up replay, once per group.

      **Review 2026-09-26.** Three findings changed behaviour. A window
      with the default `window_every` made `takes()` true on every row, so a
      sharded windowed model flushed -- one fork-join -- per row, the regime
      §25 measured as slower than unsplit, and `"auto"` did not know (A1):
      `auto_shards` sizes a windowed model's flush by the snapshot cadence,
      every row at the default, which no width can keep busy
      (`auto_does_not_split_a_window_snapshotted_every_row`, and a counting
      runner in `a_window_snapshotted_every_row_flushes_every_row` shows the
      per-row flush as a fact). `state()` and `pair()` with rows held were
      `debug_assert`s: a release-build caller got a state whose targets had
      advanced and whose pairs had not, unrecoverable after a restore (A3);
      `assert!` in every build, and the bank cannot reach it (every exit of
      `run_instance` flushes). `load` compared `shards` with the saved specs
      and refused a bank resumed under another count, which the docstring
      promised (F3): the count is left out of the comparison and the bank is
      built from the specs given, so it runs under theirs; given none, the
      saved one (`a_bank_saved_under_one_shard_count_loads_under_another`,
      `test_a_saved_bank_resumes_under_another_count`). Smaller: every flush
      copied the whole lag ring through `to_vec` and `set_ring` (A8, B7) --
      the ring keeps its newest rows and the held ones are pushed; a dead
      `decay(pending_lam)` in `freeze` (A9); `Deferred` is boxed like `lag`
      and `bins` (A10); the batch-size comment's "within 6%" holds at nine
      targets only (A5); the bench's checksum left out `lagcorr_yx` and
      `bin_var_y` (E3) and hashes every pair field now; §25's "about one
      held batch per thread" is not a bound under rayon's stealing, and says
      so (E2). Tests that could not fail: `"auto"` at 60 features is one
      range in three of the four shapes, so those legs compared the model
      with itself (F2) -- `test_auto_splits_at_a_width_that_keeps_the_pool_busy`
      runs at 1,000 features, where `auto_shards_at_the_python_suites_widths`
      pins that every shape splits; the CLI test compared `n_eff` alone,
      which no shard computes (E1) -- it saves the state and compares the
      pairs, under `"auto"` and `4`; the Rust `"auto"` assertion failed on a
      one-thread pool (D9, E4). A panic inside a shard leaves the model
      inconsistent without a `broken` mark (E5); noted, not built: a panic in
      the kernels is itself a bug. New edge tests: a label delay, a halflife
      grid and a session reset under a window; closed groups at a session
      change and by a monotone key; one-row chunks; forty groups on two
      threads under `"auto"` (a child process); scoring holds no rows; a save
      inside the bin warm-up; a cadence longer than a batch; and in Rust a
      restore mid-warm-up, a refusing budget with rows held, the count
      changed with no flush between, `p = 1` with two shards, a total gap
      with rows held, all-absent targets after a wipe.

- [ ] 127. **E74: a chunk run pair-major — not as specified.** Its reason
      is the intercept, which is not the model's (task 124, which took it
      from 13.1 to 2.8 ns per feature per row); the bank
      already hands the model one contiguous row (`feature_rows`), so
      there is no per-row gather to remove; and its lag, `x_j[r − ℓ]`, is
      not the model's: the ring holds only rows that taught something
      (weight above 0, every feature finite, marginal.rs:1030-1040), and
      the caller's weight is 0 on its warm-up and purge rows. What may
      remain is the state traffic behind the slope, `p·T` pairs' state in
      and out of cache on every row: measure it after 122, 123 and 125,
      before designing a loop order that every clock event, reset, window
      snapshot, group close and `label_delay` replay would have to split.
      *Recommended: wait for that measurement.*

- [x] 128. **`EwCov` keeps its runs without a window** (found by task 124,
      PERFORMANCE §24). S–M, a confirmed regression of task 94, unreleased.
      **Done 2026-09-25:** `Runs::off`, a state flag read as on where a
      state lacks it, and `EwCov::without_runs`, taken by every owner
      without a window (`ew_cov`, `ewridge`, `lasso` and `ew_class` without
      one; `huber`, `quantile` and the slot metrics always); `marginal`
      builds its runs off too, so a wide one without a window no longer
      allocates 16 bytes a pair for them. A test per owner pins the choice.
      `core_bench`'s `ewridge`: +15% at 5 features, +17% at 20, +4% at 50,
      +10% solving every 25 rows; the goldens are unchanged.
      Every covariance accumulator tracks each slot's run on every learned
      row, and only a window reads them (`crate::truncated`); `core_bench`'s
      `ewridge` is 18–20% faster at 5 and 20 features without the tracking,
      4% at 50. The owner knows whether it has a window, the accumulator
      does not: a state flag with a default of *on* (so an old state keeps
      tracking, the safe side), set off at construction by every owner
      without a window, kept through resets and the blend's rebuild
      (`gaps.rs`'s `empty`), ahead of `pending`, which must stay last. Rides
      schema 16. Tests: a model without a window keeps no runs, one with a
      window reads a held slot as today, and the numbers are bit-identical.

      **Review 2026-09-26.** No `restore` re-derived the flag from the
      cfg (C4): a 0.10.0 state, which has no flag, resumed without a window
      kept tracking for the rest of its life, this task's saving lost after
      every resume; and a state whose runs are off restored under a window
      read every held slot as moving, the task-94 defect back with no error.
      Every owner's `restore` now sets the runs off without a window and
      refuses a windowed state whose runs are off (`ew_cov`, `ewridge` and
      its twin, `lasso`, `ew_class`, `robust`, `hmm`); `EwRidge::restore`
      also gained the ring-against-cfg check its three siblings had (C5); and
      `hmm`, a windowless owner this task's "every owner" missed, keeps no
      runs (C3). The docs of `held_from`, the `runs` field and a test still
      described the weight rule the 2026-09-25 review replaced (C6), and
      `window_every` was documented as learned rows where the cadence counts
      every row, weight-zero rows included (C7): the docs say so now
      (`a_zero_weight_row_counts_toward_the_snapshot_cadence`). Two entries
      of `scripts/mutants_equivalent.toml` excused mutants that are not
      equivalent: `EwCov::skip`'s `*`→`/` skipped the prior's ageing on an
      empty blocked Gram (C1), `Lasso::step`'s `<`→`<=` moved `lam_selected`
      on a tie two saturating penalties give (C2); a test each kills them
      and the entries are gone.

- [x] 129. **`marginal`'s bins budget counts three values a cell where
      there are four** (found 2026-09-26, checking the memory caps at the
      user's question). S, a confirmed regression of task 101, unreleased.
      `BinCfg::validate` refused a histogram past 256 MiB (`MEMORY_BUDGET`,
      `margbins.rs`) counting `3 × cells` doubles: the weight, mean and
      spread a cell kept in 0.10.0. Task 101 gave each cell a fourth, the
      mean's low part (`mean_lo`), and left the count at three, so a
      histogram the check allowed could take 4/3 of the budget, 341 MiB. No
      test held the estimate to the allocation. Three older gaps of the same
      kind, all in 0.10.0: the warm-up hold counted a held target at 8 bytes
      where an `Option<f64>` takes 16, and a held row's own size not at all;
      the peak at the warm-up's last row is hold plus histogram, each allowed
      256 MiB; and the cap is per model, which no document said.
      **Done 2026-09-26**, as the user approved after
      the plan's first version: `CELL_VALUES`, the per-cell vector count,
      is what `MarginalBins::new` takes its vectors from, in one array, and
      what `histogram_bytes` counts, beside the edges, their lists, the
      offsets and the row's offsets; `hold_bytes` counts a held row from its
      types, `HeldRow` moved beside it; the hold is reserved at exactly
      `bin_warm_rows` rows, again after a restore; learned edge lists are
      trimmed to the edges found. The peak is documented, not refused: a
      check on the sum would refuse 10,000 features, 50 targets and 16 bins,
      E73's full profile, whose histogram alone fits (244 MiB). "Per group,
      and per halflife" is in the docstring, the error message and
      MARGINAL-LAGS-AND-BINS. A bank-wide bound was not built: groups arrive
      with the data, so a build-time check cannot count them, and every
      model's state grows with them, not the bins' alone. Tests:
      `the_histogram_budget_is_what_its_buffers_take` and
      `the_hold_budget_is_what_the_held_rows_take` hold each estimate to the
      bytes the buffers report, to the byte;
      `the_state_holds_nothing_the_budget_does_not_count` fails when the
      state gains a number no count covers; `the_budget_refuses_at_the_real_size`
      pins the boundary. Restoring the three-value count, the eight-byte
      target or the growth by doubling each fails them. Visible: a spec
      whose histogram needs between 256 and 341 MiB, or whose hold
      undercounted its targets past the budget, is refused where it was
      allowed.

      **Review 2026-09-26.** Reserving the hold at exactly `warm_rows`
      on the first held row charged every group the whole hold up front --
      64 KB a group at the default, 640 MB across ten thousand short groups
      -- where doubling had cost at most twice the rows held (A2): the hold
      grows by doubling, capped at `warm_rows`, so `hold_bytes` stays the
      most it can reach and a short group holds only its rows
      (`a_short_group_holds_only_its_rows`). The spec doc said the budget was
      per group; it is per model instance, every halflife of a grid included
      (D11).

- [x] 130. **Two window snapshots undercount what they hold** (found
      2026-09-26, checking the other memory estimates after task 129). S,
      confirmed regressions of tasks 101 and 102, unreleased. A window's
      ring thins or refuses past its budget (`window_budget`, 256 MiB a ring
      by default) by adding up each snapshot's `Footprint`, a sum over its
      fields written by hand beside the struct. Two sums fell behind their
      structs: `Cross` (`gaps.rs`), which every `ewridge` and `lasso`
      snapshot clones whole, gained `m_lo` and `my_lo` in task 101, `k + T`
      doubles uncounted; and `MarginalMoments` gained `rows`, one count per
      target, in task 102. **Done 2026-09-26**, at the user's word: both
      footprints count them. `window::assert_footprint_counts_every_vector`
      walks a snapshot's state and holds its footprint to every vector of
      numbers in it, from two snapshots of different sizes, so a vector left
      out shows as a difference that grows with the vectors. That second
      snapshot was needed: from one alone, the scalars `ewridge`'s
      footprint counts equalled the low parts it missed, and the check
      passed. A test per snapshot type (`ew_cov`, `ew_class`, `ewridge`,
      `lasso`, `marginal`); the `ewridge`, `lasso` and `marginal` ones fail
      without the fix, and `ew_cov`'s and `ew_class`'s guard a footprint
      that needed none; the spread's ring (`resid_window.rs`) has its own. A ring near its budget now thins or
      refuses a little sooner than before the fix: what it counts is what it
      holds.

      **Review 2026-09-26.** The two-size check documented "every
      vector longer in the larger" and did not enforce it, so a vector sized
      by a dimension neither snapshot varied could still hide under the
      scalar slack (B4): it walks both snapshots in step and refuses a vector
      that does not grow (`the_footprint_check_sees_a_vector_that_does_not_grow`),
      which made the `lasso` test vary the path and the two-Gram `ewridge`
      test (`the_window_footprint_counts_every_gram`, C missing 7) vary the
      targets. The ridge and lasso snapshots cloned the means' low parts,
      which the window's subtraction never reads (C9): cleared in
      `Acc::snapshot`, `k + T` doubles a snapshot the ring no longer holds.
      `Moments::footprint` counts its three numbers (C10). This entry said a
      test per snapshot type failed without the fix; `ew_cov`'s and
      `ew_class`'s did not need one (C11).

- [x] 131. **`marginal(bin_budget=...)`: the bins' memory limit, set per
      spec** (the user, 2026-09-26, after task 129). S. The warm-up hold and
      the histogram were each refused past a fixed 256 MiB. `bin_budget`
      sets that limit in MiB for both, per spec: 256 when absent,
      `float("inf")` (`"inf"` in JSON and TOML) for none; it needs `bins` or
      `bin_edges`, and a budget that is not a positive number is refused by
      name. **Done 2026-09-26:** `BinCfg::budget_mib`, the last field and
      the only one that skips, so a model without it keeps its bytes
      (`a_bins_budget_round_trips_in_both_encodings`); the spec's
      `bin_budget` a `Num`; the Python builder and the CLI's TOML take it.
      The error names the budget and the setting. `gram_block_rows`'s cap,
      the same kind of build-time check, stays fixed at 256 MiB: the user
      asked about the bins'. Tests: `the_budget_can_be_set` in both
      directions, infinity and the bad values;
      `test_bin_budget_sets_the_limit` through the bank, with a hold of 313
      MiB refused by default, built at 400 MiB and refused again at 300.

      **Review 2026-09-26.** Tests at the spec layer, which had only
      `check()` being `Err` for a budget without bins:
      `bin_budget_is_held_to_what_it_can_mean` (zero, negative, `-inf`, NaN
      and a string refused by name, beside given edges accepted, `"inf"` no
      bound), `shards_read_a_count_or_auto_and_refuse_the_rest` and
      `shards_round_trip_through_json_and_msgpack` (D missing 7, 8), and the
      Python refusal of a tiny budget beside given edges. The builders'
      finiteness check did not walk a dict, so `bin_edges={"x": [inf]}` was
      named by serde as `model` (F8).

- [x] 132. **Review 2026-09-26 of the unpushed changes** (the user:
      "a deep review iterating through each file and reviewing the code in
      full while examining the changes as well; also examine the tests,
      find edge cases that may be missing"). Scope: every change since
      `ae34d91` -- tasks 101-103, 107a, 108, 109's items, 121-124, 126,
      128-131. Protocol as docs/REVIEW-2026-09-18.md §0: six reviewers by
      area (A `marginal.rs`; B bins, lags, runs, window; C the covariance
      accumulators and the mutant excusals; D spec, targets, arrow; E stream
      and bank; F the Python package and tests), each finding re-derived,
      reproduced with a failing test before its fix, fixed in two batches
      (core, then the crates above it), gated unpiped. **Done 2026-09-26**;
      each finding is recorded under its task by the reviewer's letter and
      number, with the test that now pins it. Two crashes (B1, D1), one
      silent wrong output (D3), two infinities from inputs inside the bound
      (G1, a row absorbed from infinitely far; G2, a quantile nudge past its
      row; G3, an own mean rebuilt from two level-sized numbers -- schema
      17; all found by the widened proptest), a per-row flush regime `"auto"` did
      not know (A1), a memory charge per group (A2), a documented resume that
      refused (F3), eight lost guards or wrong excusals (A3, B2, B3, C1, C2,
      C4, C5, D4), four misleading messages or refusals (D2, D5, D6, D7,
      F4, F7), a test helper that did not enforce its rule (B4), three tests
      that could not fail (E1, F2, D9) and a dozen doc mismatches. Nothing
      changed in the numbers a model returns but `hit_rate` under a ratio
      target, the distances `micro` and `kmeans` report where their squares
      overflow, a quantile `robust` fit's step on a row whose leverage
      exceeds its weight share, and the last bits of a gappy target's fit
      under `ew_ridge` and `lasso`. Left as noted: E5 (a panic inside a shard leaves no
      `broken` mark) and task 105.

- [x] 133. **The release workflow tags what it published, after
      everything ran** (the user, 2026-09-27, after the 0.11.0 tag failed:
      "change the release workflow so that GitHub tags a release after
      everything runs. We want to keep the rehearsal for testing"). S–M.
      **Done 2026-09-27.** `release.yml` has no tag trigger any more: it is
      dispatched on `main`, and a `publish` input (off by default) turns
      the rehearsal into a release. A `version` job runs first
      (`scripts/release_version.py`): the six places agree, the CHANGELOG
      has the section, and, publishing, the run is on `main` and the tag is
      new -- refused before the hour and a half of builds, since the tag of
      a spent version can never be reused. A `ci` job calls the whole of
      `ci.yml` (now `workflow_call`-able, its concurrency group per calling
      workflow so a release and a push on `main` do not cancel each other),
      so the suites that failed 0.11.0 hold back the upload with the wheels,
      the state hand-off and the Polars legs. The PyPI upload keeps its
      approval; the `tag` job then tags the tested sha, annotated with the
      CHANGELOG's section, and `release` makes the GitHub release from the
      same notes. So a tag names exactly what was published. The README's
      PyPI links point at the tag the run will create (the sha when
      rehearsing). `docs` in `ci.yml` needs `pages` and `id-token`, which
      the call must grant; it runs on a push alone, so the grant is inert,
      and `tests/test_release_workflow.py` holds the workflow's shape and
      the version script to all of this (the schema check with
      `check-jsonschema` passed). RELEASE-READINESS's steps now read:
      compare, version, changelog, gate, commit and push, dispatch with
      `publish`, approve, verify. No rehearsal before a release (the user
      asked whether one was needed, and it is not): the publishing run does
      all a rehearsal does before the approval, and a failure there leaves
      nothing to undo. The rehearsal stays for testing a change to the
      workflow. Neither tries `tag` or `release`, which run after the
      upload; a failed one is re-run on the same sha.

- [x] 134. **E75: `marginal`'s lead/follow reading does not hold against a
      forward-looking target — correct the docs, change no code.** S; no
      decision. **Done 2026-09-27:** (1)–(6) below as written, the test as
      `test_a_timely_feature_beats_corr_one_row_on_against_a_forward_target`
      in `tests/test_marginal_lags.py`, and the dated note in
      MARGINAL-AT-WIDTH's table. Asked 2026-09-27 by the caller of MARGINAL-LAGS-AND-BINS and
      MARGINAL-AT-WIDTH (factor_selection, `~/dev/model`), in a request file
      that was checked, recorded here and deleted. **The reading:** seven
      places say a feature whose `lagcorr_xy[0]` exceeds its `corr` follows
      the target (a late-sampled column): the README's comment on
      `bank.marginal()`'s columns (1842–1843), the `marginal()` docstring in
      `_bank.py` (1042–1044), the spec's docstring in `_spec.py`
      (2980–2982), MARGINAL-LAGS-AND-BINS' table of columns (114) and its
      paragraph after the tests (131–134), E66's entry in this plan
      (919–921) and E66's row in ENHANCEMENTS (497). **Why it fails:** a
      forward-looking target is built from the rows after its own,
      `y_t = Σ_{k≥1} w_k r_{t+k}`, and a timely feature holds each row's
      news from that row on, `x_t = Σ_{k≥0} v_k r_{t−k}`. `corr` is
      `C(x_t, y_t)` and shares no return; `lagcorr_xy[0]` is
      `C(x_t, y_{t−1})`, and `y_{t−1}` starts with `r_t`, so it gains
      `v_0 w_1 Var(r)`. Every timely feature built from the target's own
      news shows `lagcorr_xy[0]` above `corr`, at any halflife, so the
      inequality separates nothing. It holds for two series that describe
      the same moment. **What it did:** the caller's late-sampling screen
      flagged 527 of 6,015 pairs on sixty days of Binance one-second klines,
      every one a correctly sampled return or signed-flow factor (BTC's 2 s
      return average against the 10 s target: `lagcorr_yx[0]` 0.075, `corr`
      0.105, `lagcorr_xy[0]` 0.305); read at four offsets for where the
      slope jumps, it flags none. **Checked here 2026-09-27:** the algebra;
      each of the seven places says what the request quotes; and its test
      passes against this build, in 0.75 s, with the closed forms 0.2544
      and 0.4173 it quotes.
      **To do.** (1) The README comment and both docstrings, one text: "the
      feature now against the target `l` rows back, and the target now
      against the feature `l` rows back, at each of `cross_lags` (every lag
      by default). For two series that describe the same moment, a feature
      whose `lagcorr_yx[0]` exceeds its `corr` leads the target, and one
      whose `lagcorr_xy[0]` does follows it. That reading does not hold
      against a forward-looking target, one built from the rows after its
      own. There the target `l` rows back is built partly from the
      feature's newest `l` rows, so a feature built from the same news shows
      `lagcorr_xy` above `corr` however it is sampled. Absent under
      `cross_lags=[]`." (2) MARGINAL-LAGS-AND-BINS' row for `lagcorr_xy`:
      "`C(x_t, y_{t−ℓ}) / (sd_x · sd_y)` — the feature *now* against the
      target ℓ rows *ago*. For two series that describe the same moment, how
      far the feature follows the target. Against a forward-looking target
      it also holds what the two share: the target ℓ rows ago is built
      partly from the feature's newest ℓ rows." (3) Its paragraph after the
      tests: the pair is read with the target's construction in hand; for
      two series of the same moment `lagcorr_xy[0]` above `corr` means the
      feature follows; against a forward-looking target it says nothing
      about timing, since every timely feature built from the target's news
      shows it; telling late from timely takes more offsets and a judgment
      over independent blocks, and one caller keeps `cross_lags=[1, 2]` and
      asks where the slope of the correlation jumps. (4) A dated note, not a
      rewrite, in E66's entry here and its ENHANCEMENTS row: "2026-09-27:
      the lead/follow reading holds for two series that describe the same
      moment, not against a forward-looking target (E75)." (5) A test, the
      caller's: `r` i.i.d. normal; `x` the EW average
      `x_t = (1 − a) x_{t−1} + a r_t`, `a = 1 − 2^(−1/2)`; `y_t =
      Σ_{k=1..L} w_k r_{t+k}`, `w_k = (1 − λ) λ^(k−1)`, `λ = 2^(−1/10)`,
      `L = 70`, seven of its halflives; 100,000 rows through `marginal(lam=1.0, lags=[1, 2],
      cross_lags=[1, 2])`; then `|corr| < 4/√n`, `lagcorr_xy[0]` within
      that of `√(a(2 − a)(1 − λ²)/(1 − λ^(2L)))` (0.2544),
      `lagcorr_xy[1]` of that times `1 − a + λ` (0.4173), and every
      `lagcorr_yx` below it. (6) A Docs line in the CHANGELOG. **Not
      asked:** a late-sampling statistic in the library (the rule needs a
      block design and a multiple-testing margin, the caller's choices), a
      change to the terms or their orientation (they are right, and the
      test holds them to the closed form), or a new `cross_lags` default.
      **On the caller's side:** MARGINAL-AT-WIDTH line 66 says it reads
      `lagcorr_xy[0]` and `lagcorr_yx[0]`; it now runs `cross_lags=[1, 2]`,
      about 3 % more state a pair at its defaults, the time not yet
      measured. A dated note there fits.

- [x] 135. **S18: Bartlett weights for `marginal`'s serial rule.** S. The
      user accepted it 2026-09-28 (task 115 (f)). `"truncated"` sums
      `rho_x(l)·rho_y(l)` over lags at unit weight, which is not positive
      by construction, so a mixed-sign pair drives the factor to or below
      zero, where it now reports NaN (review 2026-09-12, S18). Newey–West's
      weights `1 − l/(L+1)` keep it positive: a new `serial_rule`, held to
      `statsmodels`' `cov_hac` with `weights_bartlett` in
      `tests/test_second_opinion.py` and to a `faer`-free longhand in Rust.
      No state: the weights apply where the factor is read. **Built
      2026-09-28:** `serial_rule = "bartlett"`, `L` the longest kept lag,
      sparse lags at their own `l`; positive by construction only for a
      positive-definite sequence of products over every lag `1..=L` (the
      factor is then `1ᵀC1/(L + 1)`), so a factor at or below zero is NaN
      as under `"truncated"`. `the_bartlett_factor_is_its_definition`
      (S18's pair, `+0.8`/`−0.8`, reads 0.36 where `"truncated"` has none)
      and `TestTheBartlettSerialFactor`, which takes the weights from
      `statsmodels`' `weights_bartlett` and the autocorrelations from its
      `acf`.

- [x] 136. **S19: the Gram's target moments under a `window`.** S–M. The
      user accepted it 2026-09-28 "as long as it doesn't add much extra
      size". Under a window, `target_means`, `target_vars` and
      `target_n_kish` are `None`, since the snapshots hold no target
      moments; snapshotting them restores them as the window's. The bytes a
      snapshot gains are measured against its Gram before building, and
      the entry records both. **Built 2026-09-28:** each snapshot carries
      the target moments decayed to it (`TargetMoments::decayed`: the
      means and variances as they stand, `Q` at `lam²`), and the view
      truncates them by the pooling identity the Grams use, `m_R = m −
      r·d`, `v_R = g·v − r·v_u − r·g·d²`, `Q_R = Q − f²·Q_u`
      (`TargetMoments::truncated`); the export's Kish count reads each
      part's own, windowed, target weights. The cost is exactly `3·T`
      doubles a snapshot (`the_target_moments_add_three_doubles_a_target_to_a_snapshot`):
      +0.1 % at `k = 50`, `T = 1`; +0.8 % at 50 and 10; +1.7 % at 10 and 1;
      +4.5 % at 5 and 1; +5.5 % at 10 and 5; +15.8 % at one feature, whose
      snapshot is 152 bytes. Held to a longhand over the in-window rows,
      a gappy target included (`the_windowed_target_moments_are_the_rows_inside_it`),
      and to `statsmodels`' `DescrStatsW` (`TestTheGramIsTheWindowsToo`). A
      snapshot from before carries none, and the export says `None` until
      the ring rolls over.

- [x] 137. **C18: a lag ring under a `window` in `marginal`.** M. The
      user accepted it 2026-09-28, with the same condition on size. Today
      `window` with `lags` is refused (review 2026-09-12, C18), since the
      lagged co-moments would come from the whole history beside windowed
      variances. The lag moments are normalized moments on the same decay,
      so a snapshot of them truncates as everything else does (V12). The
      bytes a snapshot gains, `lags × pairs` and the target side, are
      measured first; held to `statsmodels`' `acf` on the in-window rows,
      as C18's test reads. **Measured 2026-09-28, and held for the user's
      word on size:** a windowed lag moment is the sum of the increments
      made inside the window, so every lag moment must be in the snapshot
      -- `L·T` for `cyy`, `L·p·T` for `cxx` and `2C·p·T` for `cxy` and
      `cyx` (`C` the cross lags, every lag unless `cross_lags` names fewer)
      -- beside today's `MarginalMoments` of about `(3p + 5)·T` doubles.
      The lag moments add about `(L + 2C)/3` of the snapshot's size: +100 %
      at one lag, +500 % at five with the default cross lags, and +167 % at
      five with `cross_lags=[]`, which `n_serial` does not read. Options:
      build it at that price; build it only with `cross_lags=[]` under a
      window (the serial correction without the lead/lag by-product); or
      keep the refusal. **Decided 2026-09-28 (the user: "For c18 implement
      but with an api parameter that documents the impact") and built:**
      the spec's `window_lags` (`po.spec.marginal(window_lags=True)`,
      default `False`) accepts the price, documented with the formula and
      two examples; without it, `window` with `lags` is refused with the
      spec's own numbers (`4 doubles beside the 8 it holds without them,
      1.5 times the size` for one feature, one target, one lag), and
      `window_lags` without both is refused. The core takes the pair: each
      snapshot holds the lag moments (`LagMoments`, `marglag.rs`; counted
      in `window_budget`), and a pair's lag moment under the window is `(W·C
      − f·W_u·C_u)/W_R` for its target. Held exactly to an unwindowed
      twin's sum form (`a_windowed_lag_moment_is_the_increments_inside_the_window`),
      to `statsmodels`' `acf` of the in-window rows at 0.02 across a `phi`
      change (`TestAWindowedLagIsTheWindows`), to the documented size
      (`the_lag_moments_add_what_window_lags_says`), and bit for bit sharded
      (`a_sharded_step_is_the_unsplit_step_to_the_bit`, a new window-with-
      lags case). A state's snapshots are now checked for shape on load, the
      pair moments' included, which were not.

- [x] 138. **README rewrite against the current API, requested 2026-09-29.**
      The user's words: "After the gate commit rewrite the readme after
      analyzing the current state of the api and features. Be sure to
      follow phrasing and writing guidelines". Task 89's rewrite is six
      days old, and 363 lines have come in since, one task at a time. So
      this pass runs `docs/WRITING.md` §6 again, against the code as it
      stands at `5e96018`.

      *Measured before* (a counting script to §6's rules: paragraph
      boundaries, a list item its own paragraph, a bold rule split from the
      sentence after it, a sentence allowed to open with a lowercase name):
      13,282 prose words, 709 sentences averaging 18.7 words, 34 of 35+
      words and 2 of 45+, no cost words (the one hit is "the
      point-biserial correlation"), 43 tables, 61 python blocks, 10
      top-level sections.

      *The map.* The ten sections stay; three pieces move, each to the
      section whose heading it serves (§2), and `←` says where from.

    | section | subsections |
    |---|---|
    | Introduction | The idea · Four words · Install · A first fit · What you can rely on |
    | How a bank sees a stream | What a spec names · Time and decay · A hard window (← `ew_cov`'s section: five models take `window`, and a reader of `ewridge`, `lasso`, `ew_class` or `marginal` never looked there; it sits beside the decay it is compared with) · A local fit along any feature · Convergence without a decay · Groups · Weights · Warm-up · Labels that arrive late · Nulls, and three ways to hold a row back · Row order and the two guarantees · Series that tick at their own times (← Running a bank, whose heading is the three ways to run one: this is how several series become one stream, beside `embargo`) |
    | Running a bank | As a query · In a loop (+ `ModelBank.fit` and `fit_predict_batches`, which the memory table uses before any section shows them) · Output as Arrow · Outside a live Python process |
    | Saving, loading and serving | Save and load · Serving without learning · What a state file holds · Reading a state without this library |
    | Reading the fit | Coefficients · Output field names · The running sums behind a fit · One row per finished group (its accumulate-only pass ← `bank.fit`) · Reading a correlation matrix |
    | Diagnostics, selection and evaluation | Per-row diagnostics (+ which models take them: the ten that predict a target; + each switch's tuning keyword) · Conformal intervals · Evaluating an output frame · Evaluating a stream too large to hold · Data whose truth is known |
    | Models | the two tables · Linear models · Moments and correlation (`ew_cov` without `window`, which it now links to) · Clustering and classification · Sequential tests and regimes |
    | Performance | Throughput (re-measured: the tables date from 2026-09-06) · Memory: which calls stream · Tuning memory with Polars' own settings · Chunk size · Parallelism · Against scikit-learn |
    | Scope and integrations | What this is not · Pathway · Databases: DuckDB and ADBC |
    | Versions, testing and development | Versioning and the Polars pin · Testing · Development · License |

      *Constraints held.* Every anchor another file links to keeps its
      heading text (`llms.txt`: introduction, how-a-bank-sees-a-stream,
      models, against-scikit-learn; `docs/RUNNER.md`, `docs/REGIMES.md`,
      `docs/OUTPUTS.md`, `docs/PERFORMANCE.md`, `docs/README.md` and
      `scripts/outputs_doc.py` use eighteen more). Every `####` under
      *Models* is a model, with its *API:* and *Rust:* lines, and the model
      table keeps its header. Every python block runs in the namespace
      `tests/test_production_hardening.py` gives it.

      *The analysis.* Four agents checked the model families, Performance,
      Scope and Versions against the code, and every finding acted on was
      re-derived first; the introduction and the shared sections were
      checked by hand. What the README said that the code does not:

    | where | it said | the code |
    |---|---|---|
    | the model table | `huber` and `quantile` "solve", so converge to the batch answer in any row order | each row's weight comes from the fit before it: 9.2e-03 between two orders with outliers. A sixth kind, *reweight*, is theirs |
    | the model table, `ewridge` | halflives are fitted from the same running sums | each halflife keeps its own |
    | the accumulators | every one is a mean, and every second moment centred | `rls`, `ftrl` and adagrad keep decayed sums; `rls` a raw one |
    | `ftrl` | river's no-halflife recursion as the rule, "agrees with river to 1e-12" | the penalty scale `m` under a halflife (C24); river and Vowpal Wabbit hold without one |
    | `holt` | "let `emit_selected` choose" against a real model | `emit_selected` ranks one spec's own slots; `seqtest(a=, b=)` compares two specs |
    | `sgd` | `l2` on every coefficient; `coef_min=inf` for no bound | not on the intercept; `-inf`, and `inf` is refused |
    | `quantile` | the band's inside is `\|r\| ≤ h`; no warm-up | `\|r\| < h`; least-squares rows until three per coefficient |
    | `kalman` | the halflife drives the noise estimate | and the standardization |
    | `deco` | sums over every feature; one weight | a column with no spread left out; a weight per value; `rho` held at weight 0 under `"linear"` |
    | `marginal` | `n_serial`'s bracket as one rule; one search per pair; 4.6× with lags; no `window` | three rules; one per feature; 4.6× with no cross lags; `window`, `window_lags` |
    | `rcov` | `preavg` over `⌊θ√n⌋` less the bias | that is `psd=False`; the default is `⌈θ·n^0.6⌉` with no bias term |
    | `ew_class` | "full" factorizes every class every row, 0.9M rows/s at 400k rows | only the row's class, outside a window; the figures were 200k rows, before the caching |
    | `micro` | the nearest summary that can take the row; ARI 1.000 "with the `eps` above" | the nearest established one, then the nearest other; `eps` per shape |
    | `corrchange` | size and power "held to" WKD's tables; `γ = 0` "keeps the size" | measured against them (0.031, 0.552), tested near 0.05; 0.04 to 0.09 |
    | `hmm` | `state` the most probable filtered state; one limitation | the argmax of `p1`; the docstring's two |
    | `bocpd` | `prune_below` makes it finite, flat at the default; the run length dates a correlation break in one to three rows | `max_run` bounds a stationary stream; 98 rows under `"gaussian"`, never under `"diag"` |
    | `kmeans`, `micro` | a quiet feature counts 58 times its history | `2^(20/8)/0.1` is 56.6 |
    | the output records | `ewridge`'s fields without `settled_frac`, `withheld_reason`, `support_coef`; `summary()` without its five readiness columns | both, since 0.9.0 |
    | the grammar | no `__l{lambda}` | lasso's path |
    | Throughput, Parallelism | 2026-09-04 and -06 figures | `ewridge` k=20 4.12M → 2.53M; the grid 12.3 s → 32.7 s (PERFORMANCE §28) |
    | Install | 19 MB to download, 59 installed | 8–10 and 26–37 (0.12.0's wheels) |
    | Testing | 976 and 3,265 tests; numpy and river as oracles | 1,149 and 3,597; eight Python libraries and `faer` |
    | Versions | `~=0.9.0`; a cap is a patch *and* narrowing is breaking | `~=0.12.0`; narrowing is breaking but for the cap |
    | Scope | the Pathway operator runs in a pipeline; ADBC's trap is a second query | the example runs the operator over batches; re-executing a cursor, both queries wrong |

      Missing features, now documented: `ModelBank.fit` and
      `fit_predict_batches`, `closed_groups(drop=)`, the diagnostics' tuning
      keywords and which models take the diagnostics, `po.eval.unpack`,
      `po.corr.equicorr_row`, `window` on five models, `window_lags`,
      `serial_rule="bartlett"`, `strict_binary`, `select_halflife`, `p0`,
      `share_p`, `max_clusters`, `update_every`, `psd`, `transition_prior`,
      the Gram's target moments under a window. Docstrings with the same
      stale facts were corrected with them: `ewridge`, `sgd`, `holt`,
      `ew_class`, `marginal`, `deco`, `corrchange`, `bocpd` (its example
      dated a run on a global row index across interleaved groups),
      `ModelBank.gram` and `ModelBank.summary`, and two comments on `rcov`'s
      window. `huber`'s and `lasso`'s order exemption is task 139.

      **Done 2026-09-29.** Measured the same way before and after, on
      paragraph boundaries:

    | measure | before | after |
    |---|---:|---:|
    | prose words | 13,282 | 14,776, the added facts |
    | sentences of 45+ words | 2 | 0 |
    | sentences of 35+ words | 34 | 2, both 35 words |
    | mean sentence, in words | 18.7 | 18.1 |
    | cost words | 0 (one false hit) | 0 (the same false hit) |
    | tables, as rendered | 43 | 48 |
    | top-level sections | 10 | 10 |
    | python blocks, all running | 61 | 61 |

      Accounted as a diff: every number, backticked name and link the old
      text had is in the new, in PERFORMANCE §12 or §28, or replaced by a
      measurement or a correction named above. `scripts/doc_structure.py`
      passes on every tracked file, and GitHub's renderer shows 48 tables,
      88 code blocks and every in-page link landing.

- [x] 139. **`ModelBank.fit`'s order exemption, narrowed to `ewridge` and
      `rls`.** Found by task 138's analysis. `fit` skips
      `OrderNotGuaranteedWarning` where the state it leaves cannot depend on
      the row order, and `huber` and `lasso` at `lam=1.0` were on that
      list. `huber`'s 6.7e-16 was measured on rows no residual reached
      `delta * sigma`, so its reweighting never ran: with one row in ten
      lifted by 5, a shuffle moves its coefficients by 1.05e-02. `lasso`'s
      path points commute to rounding, but `lam_selected` ranks them by
      out-of-sample error, and shuffles moved the penalty it selects from
      0.01 to 0.1 and to 0.001. Tests first: both specs join
      `test_what_is_not_order_free`, which failed for both on the old list,
      and two premise tests show the reweighting firing on dozens of rows
      and the selection moving while every coefficient holds to 1e-12. The
      list, its measurement comment, the warning's docstring, `fit`'s
      docstring, the README and ARROW-SOURCES say the same now, and the
      keys only those two models take left the table.

- [x] 140. **Task 87's inverse diagonal, on every solve** (the user, after
      task 138 found the README's throughput 40–50% stale: "find which
      change caused the throughput drop"). S–M; *the user's call*. The
      bisect is PERFORMANCE §29: task 87 took `ewridge` 26% at k=20
      (`11e3ccb` 3.65M rows a second, `3e0b359` 2.69M), the clock checks
      (`7965035`), compensated means (task 101) and the floored cluster
      metric (task 102) the rest, each the price of a correction. Task 87's
      is not: `EwRidge::solve` forms `A⁻¹`'s diagonal on every solve, for
      each coefficient's data share and the noise gate's `edf`, and at HEAD
      that costs 7% at k=5, 21% at k=20 and 46% at k=50 (a build with the
      diagonal replaced by zeros). A cheaper direct method recovers
      nothing: the squared column norms of `L⁻¹`, written two ways, ran no
      faster than faer's inverse. Options:
      (a) **Compute it when it is read**, from the factor the solve keeps:
      on `coef` rows (by default one per group per chunk), in `summary()`
      and the `ReadinessWarning`, and for the noise gate only while it can
      bind. `edf` is at most one per kept column plus one, so once
      `sqrt(1 + (1 + k) / n_kish)` is below `max_error_inflation` the gate
      is open whatever `edf` is. Every value read is the same bits as
      today; the memory is a `k × k` factor per system, which
      `emit_error_inflation` keeps already. Recommended.
      (b) Every m-th solve: values stale between, and a new setting.
      (c) Keep it, and let the README's throughput say so, as it now does.

      **Built 2026-09-29, option (a), on the user's "Build the fix".** A
      solve no longer takes the diagonal. It keeps its factor in a
      `Pending`, shared by the slots the system answers, and the shares
      are computed on the first read and kept (`OnceLock`). A `coef` row
      reads them, and so do `summary()` and a save, which writes the shares
      a read would give, so the state layout is unchanged and needs no
      schema bump. The end of each run stores them and drops the factor,
      so a factor lives only while its stream runs. The noise gate reads
      `error_inflation_gate_into`. Where a solve is unread and the ratio of
      the bound, `sqrt(1 + (edf0 + k) / n_kish)`, is below
      `max_error_inflation`, the bound stands in, since the computed `edf`
      rounds to no more than it. Where the ratio could reach the limit, the
      gate reads the exact value, and so does the unreachable-gate
      warning's worst ratio. **The bound needed a guard these options
      missed.** A share can be NaN: with no ridge, an inverse that
      overflowed gives `1 − 0·∞`. The exact ratio is then infinite and the
      row is withheld, where the bound would have passed it. Features near
      `1e-155` fitted through the origin do it, and the test for it failed
      before the guard. `SpdFactor::inverse_is_finite` bounds every
      quantity the two triangular solves form, in `O(k²)` from the
      comparison matrix, and the bound stands in only where that holds. In
      its test it certified 2,181 factors and none of the 1,675 whose
      diagonal was not finite. **No value moves**: 18 `ewridge` workloads
      against `80df0ac`, every float of every frame, the state saved
      mid-stream and at the end, and every warning, the same to the bit.
      **Speed** (PERFORMANCE §29): +16% at k=20 and +41.5% at k=50, 82% and
      92% of the stub's ceiling; k=5's +3.5% is within noise; +42% where
      every row solves. `coef_every=1`, which reads the shares on every row,
      pays 1 to 3% for the bookkeeping. The README's `ewridge` rates are
      re-measured, with `rls`'s beside them and the `predict` ratio, and so
      are three solve-every-row figures stale since 0.9.0. Its Parallelism
      figures at k=20 were not: at load 4, a 14-thread run times the other
      jobs. Found on the way, not investigated: where every row solves, even
      the stub runs 23% below §19's rate of 2026-09-08.

- [x] 141. **Solve-every-row speed** (the user, after task 140 found §29's
      stub 23% below §19's rate where every row solves: "Solve-every-row
      speed"). S–M. PERFORMANCE §30 has the bisect: at k=20, 0.5.1 ran
      546k rows a second, the 2026-09-12 review's library batch took
      10.6% (N1's centred solve and the rest, spread over its commits),
      task 87 took 40%, and after task 140 it was 412k. A profile put 90%
      of a row in the solve: its own copy of the centred system and `k²`
      divisions by scales of 1 (23%), faer's per-call matrices for one
      right-hand side (13%), and the factorization's input copy (8%).
      **Built 2026-09-29:** unstandardized, the solve reads the Gram
      straight, with no divisions by 1; standardized, it reads only `C`'s
      diagonal for its scales; `SpdFactor` solves in place, as faer's own
      `solve` does after its copy; `ln det` is taken when read. No bit
      moves: 20 workloads against `80df0ac`, `coef_prior` among them. Every
      row solving, +8% at k=5, +13.5% at k=20 and +19% at k=50, with k=20
      back at 0.8.0's rate; at the default cadence, +3 to +10%. Left: about
      twenty allocations a solve.

- [x] 142. **NumPy's newest and next versions, tested as Polars's are** (the
      user, 2026-09-30: "Since numpy is an optional dependency we should
      test on the next release candidate of that too as we do with
      polars"). S. NumPy is the one optional extra (`numpy>=1.24`, no
      ceiling), and every job ran on the locked NumPy: 2.5.2, where PyPI has
      2.5.3. **Built:** `release.yml` gains `next-numpy`, two legs like
      `next-polars`: the newest NumPy blocks the publish, and its next
      release candidate (`--prerelease=allow`) is advisory. Only NumPy is
      upgraded, so a red leg names it. `polars-canary.yml` gains the same
      advisory run weekly. On 2026-09-30 NumPy's candidates on PyPI,
      2.4.0rc1 and 2.5.0rc1, are older than 2.5.3, so both legs resolve to
      2.5.3 until the first 2.6 or 3.0 candidate. `tests/test_release_workflow.py`
      holds the legs, and `test_ci_cost_policy.py` now names the prep
      action's callers. A workflow change, so the next release rehearses
      first, with publish off. Left as it is: the extra has no ceiling, so a
      NumPy 3.0 that breaks this library blocks every release until it is
      fixed or capped, the trade the Polars range makes too.

- [x] 143. **Window expressions: factors and targets as Polars formulas
      over exponentially weighted operators -- the column form built
      2026-10-03 ("Built" at the end); the target form is task 104's.**
      Size M to L. Depends on 78 (done); the target form on 104. Replaces
      `po.window.ewm` and `po.window.lookahead_rewm` (pre-1.0, no aliases).
      Constraints: the speed of task 78's core, and no memory beyond what
      each operator needs.

      #### The design

      Everything stateful is a kernel in the core -- direction, clock,
      `half_life`, `window_size` -- applied to any number of inputs.
      Everything row-local is a Polars expression, evaluated by Polars a
      chunk at a time. Each operator returns a genuine `pl.Expr`: a column
      reference whose name encodes the operator (`@po:{...}`, its JSON), so
      it composes with `pl.col`, `pl.lit`, numbers, `when/then` and every
      element-wise Polars function, on either side of an operator, and
      `pl.col` means the current row.

      ```python
      dpv, dv = po.increment("cum_notional"), po.increment("cum_volume")
      notional = pl.col("price") * pl.col("quantity")
      lf.online.with_windows(
          mid_trend=po.ewm_mean("mid", half_life="5s", window_size="1m") - pl.col("mid"),
          fwd_mid=po.rewm_mean("mid", half_life="10s", window_size="1m") - pl.col("mid"),
          fwd_vwap=po.rewm_sum(notional, half_life="10s", window_size="1m")
                   / po.rewm_sum("quantity", half_life="10s", window_size="1m") - pl.col("mid"),
          buy_vwap=po.ewm_sum(pl.when(pl.col("side") == "buy").then(notional), half_life="10s")
                   / po.ewm_sum(pl.when(pl.col("side") == "buy").then("quantity"), half_life="10s"),
          vwap_from_sums=po.ewm_sum(dpv, half_life="10s") / po.ewm_sum(dv, half_life="10s"),
          past_cvwap=(pl.col("cum_notional") - po.ewm_mean("cum_notional", half_life="10s", window_size="1m"))
                     / (pl.col("cum_volume") - po.ewm_mean("cum_volume", half_life="10s", window_size="1m")),
          volume_rate=po.ewm_rate(dv, half_life="30s", window_size="5m"),
          clock="ts", max_dclock="5m", session="date", session_gap="5m", group="symbol",
      )
      ```

      #### The operators

      Each is defined once, by the library that has it, and held to it:

    | operator | definition | oracle |
    |---|---|---|
    | `po.ewm_mean` (back), `po.rewm_mean` (forward): time-weighted mean | `y_i = α_i x_i + (1 − α_i) y_{i−1}`, `α_i = 1 − λ^(Δt_i)`: each value weighted by the decayed time of its interval; forward, the mirror, each value held until the next row | Polars `ewm_mean_by` (agrees to 1.3e-14 on irregular times) |
    | `po.ewm_sum`, `po.rewm_sum`: exponentially weighted sum | `y_i = x_i + λ_i y_{i−1}`, anchored at the row's own time | Polars `ewm_sum_by` |
    | `po.ewm_rate`, `po.rewm_rate`: rate per unit time | the sum over its decayed time mass, `∫ λ^age ds` over the time the window covers | the definition, integrated |
    | `po.increment(col)` | `x_i − x_prev` within the group and session; null on a session's first row; seconds on a temporal column | Polars `diff().over([group, session])`, in memory |

      - **A level is a time-weighted mean**, so a burst of rows does not
        outweigh a quiet stretch. No mean counts rows; pandas' `ewm(times=)`
        is that form, and is not offered.
      - **A weighted mean is a ratio of two sums**: a VWAP is decayed
        notional over decayed volume, the time mass cancelling. It equals
        `ewm_sum_by(p q) / ewm_sum_by(q)` to 2.8e-14, and is exactly
        unchanged by zero-volume rows or by a print split in two. A side's
        VWAP puts `when/then` inside both sums; there is no `weight=`.
      - **Recipes are compositions, not operators.** Each operator is held
        to its oracle; a factor or a target is a formula over them, and the
        design excludes no recipe. Which recipe a quantity should use is
        settled by measuring them against their definitions and each other,
        not by this plan. The ones known so far:
      - **A rate from running sums** can be `(C(t) − po.ewm_mean(C)) / (t −
        po.ewm_mean(t))`, the clock's own running value in the denominator:
        `Σ K dv / Σ K dt`, the same tail kernel on the flow and on time, so
        the two are attributed in the same steps and the attribution largely
        cancels. `t − po.ewm_mean(t)` is the window's mean age, near `half_life /
        ln 2` (measured, half-life 10 s: 14.18 s on even rows, 11.19 s on
        bursty ones, the spread of the order of the gaps). `po.ewm_mean(dt)` is
        no time mass -- under the time-weighted mean each interval counts by
        its own length, 7.1 s on bursty rows with a 0.5 s average gap -- and
        is never a rate's denominator.
      - **A rate can also be the rate operator**, the sum over its decayed
        time mass: the point kernel, as the ratio of sums is for a VWAP. It
        takes any input, running sums' increments and squared mid changes
        alike, so every rate can be written both ways.
      - **Running sums** give a VWAP two ways. `po.ewm_sum(dpv) /
        po.ewm_sum(dv)` is the per-trade VWAP, each trade weighted by its
        own decayed time; where the sums step exactly at trades it equals
        the VWAP from prices and quantities. `(C(t) − po.ewm_mean(C)) /
        (C_v(t) − po.ewm_mean(C_v))` weighs each trade by the decayed time mass
        on its far side within the window -- before it backward, after it
        forward -- a smooth taper to 0 at the window's edge; the two agree
        as the window grows against the half-life. Either way an increment
        carries the day total's rounding: about 2e-6 at a notional of 1e10;
        an integer volume's increments are exact.
      - **A volume clock is a clock column**: with the running volume as the
        clock, `half_life` and `window_size` are in shares, and weights
        decay with the volume traded since. **Today only ungrouped**
        (measured 2026-10-02): the core keeps a shared clock across groups
        for the silent-group cut, and a per-symbol running volume steps
        back at every interleaved row, so `group="sym"` on such a clock is
        refused at row 1. A clock has to say whether it is shared across
        groups (time) or each group's own (volume), and a group's own clock
        skips the shared advance and the silent list.
      - The windowed and forward forms, which no library has, are held to a
        brute-force loop from the same definition and to the time-reversal
        identity: a forward window is a backward one over the reversed
        stream, less the row itself.

      #### `with_windows`

      - **`lf.online.with_windows(*exprs, **named, clock=..., ...)`** reads
        like `with_columns`: names come from keywords and `alias`. It finds
        the operators with `meta.root_names()`, computes each distinct one
        once as a hidden column while streaming, evaluates the expressions
        with Polars on each chunk it emits, and drops the hidden columns.
      - **A formula is kept in a compact tree of this library's own**
        (decided 2026-10-02, the user: "Pl.col definitions can be
        serialized, see if there is a super compact form that can be stored
        in the spec"; then "bump up the polars floor if needed"). At spec
        build time Python reads the expression's own tree
        (`expr.meta.serialize(format="json")`, with the caller's Polars) and
        walks it into nested lists: `["-", ["rewm_mean", "mid",
        {"half_life": "10s", "window_size": "1m"}], ["col", "mid"]]`, about
        70 bytes for that target, plain JSON. The operators are nodes with
        their parameters, not placeholder columns, so the tree is
        self-contained; it goes into the spec dict, the saved state and
        TOML (`expr = [...]`) as it is. Seven node kinds are read and no
        other: `Column`, `Literal`, `BinaryExpr` (its named operators),
        `Function` (log, exp, abs, sqrt, pow, clip, fill_null, is_null,
        negate), `Ternary` (when/then/otherwise), `Cast` and `Alias`.
        Measured 2026-10-02: every one of them has the same shape under the
        floor Polars (1.34.0) and the pin (1.44.2), so the floor stays
        1.34.0; the serialized tree as a whole is unstable across versions,
        which is why only these nodes are read, at build time, under the
        caller's own Polars, and an unknown node is refused by name. Both
        sides rebuild the formula through Polars' public builders, which
        are stable where the tree is not: Python's `pl.col`, `pl.lit`,
        `pl.when` and the operators; Rust's `col()`, `lit()`, `when()`, the
        `Expr` methods and arithmetic, all in the embedded polars 0.55.2
        under the `lazy` feature already on, no new crate. A round-trip
        test covers every node kind in both directions, on the floor
        Polars too.
      - **Element-wise only, by node kind.** `shift`, `diff`, `cum_sum`, a
        rolling window, `over` or an aggregation would depend on the
        chunking (hard rule 3); each arrives as a `Function`, `Window` or
        `Agg` node outside the seven, so the build refuses it by name. No
        probe-frame evaluation.
      - **An operator's input** is a column or an element-wise expression of
        the row, `po.increment` included: the core computes increments as
        hidden columns from each group's previous row, then Polars evaluates
        the expression around them.
      - **Names, boundaries and equal timestamps follow Polars**: task 144.
      - **The call's own keywords are not output names** (decided
        2026-10-02, the user: "3a"). `with_windows(session=po.ewm_mean(...))`
        is refused, with a message pointing at `.alias()`: the clock policy
        (`clock`, `gap_cap`, `restart_after_step_back`, `session`,
        `session_gap`, `group`), `like`, `chunk_rows`, `load_state` and
        `save_state` are reserved. Polars' `with_columns` has no such clash,
        since it takes no policy. Considered and not taken: the policy only
        through `like=`.
      - **The clock and its policy** -- `max_dclock`, `on_clock_reset`,
        `min_backwards_jump`, sessions, groups -- are task 78's, shared by
        every operator in the call. **A clock per operator, Polars' `by=`,
        waits on a need** (decided 2026-10-02, the user: "Follow the reco";
        task 118 lists it), so that volume-clock and time-clock windows
        could share a call. Analysed: possible, since every
        queue item already carries its own policy time and nothing in the
        core assumes one clock but the clock fields themselves. The core
        keys each kernel by its clock; the clock state, the policy time, the
        restart flag and the silent list become one per clock in `Windows`
        and in each `Group`, a waiting row carries a policy time per clock,
        a break on one clock cuts the windows on that clock and a session
        change (one column) cuts every window, emission stays in input
        order since each clock's closures are monotone, and the window
        state file bumps its version. The policy goes per clock -- the gap
        cap, the step-back rule, `session_gap`, and whether the clock is
        shared across groups -- as `po.clock(...)` objects on the call, the
        operator naming its clock by `by=` and taking the call's single
        `clock` by default, and each window's `half_life` and `window_size`
        checked against its own clock's dtype. About two days on top of
        143, a second clock advance per row (unmeasured, expected under
        5%). Two things it does not give: a forward target off the spec's
        clock, since the embargo cannot be shown to cover a window in
        another unit (refused), and on a volume clock Polars' equal-stamp
        rule (task 144) holds every quote row's backward window until the
        next trade, since the quotes between two trades share one volume.
        Two calls compose for the column form, a formula over the first
        call's outputs in the second, at the cost of a second pass and a
        second state file. Not built until a need appears; then inside 143,
        where the kernel keying is being rewritten anyway, if 143 is still
        open.
      - **As a target** (task 104), the same expression in `targets=[...]`,
        carried as the tree above: the bank rebuilds it with the embedded
        Polars and evaluates it on the rows the window core resolves, so
        the column form and the target form share one evaluator, the
        command line takes a formula, a saved state resumes without being
        handed it again, and `fit_predict_arrow` takes one too.
      - **Against the current quote** is arithmetic: `- pl.col("mid")`,
        `/ pl.col("ask")`, `(op / pl.col("bid")).log()`; task 107's
        `relative_to` stays for plain-column targets.
      - **Open, the user's call: a ratio target's hit test.** A plain-column
        ratio target carries `relative="ratio"`, which centres `hit_rate`'s
        sign test at 1 (task 107's review, D3); an expression `op /
        pl.col("bid")` carries nothing, so its test centres at 0 and reads
        1.0 whatever the fit. Recommended: the docs say a ratio target is
        written as a log ratio or a difference; the alternative makes
        `hit_rate` null on every expression target.
      - **The operators take Polars' names** (decided 2026-10-02, the user:
        "4a"). Polars 1.44 has `ewm_mean`, `ewm_sum` and their `_by` forms, so
        the time-weighted mean is `po.ewm_mean` and the sum `po.ewm_sum`, and
        their forward mirrors `po.rewm_mean` and `po.rewm_sum`. The rate,
        which Polars lacks, is `po.ewm_rate` and `po.rewm_rate` on the same
        pattern, and `po.increment` keeps its name. Considered and not taken:
        `po.ewma` and `po.rewma`.

      #### Memory

      - **One queue per kernel**, however many operators share it: an item
        holds its clock and one value per operator.
      - **A row enters an operator's queue only when it carries a value**:
        a sum of a notional that is null on quotes queues the trades alone;
        a mid's time-weighted mean queues every quote, which it needs.
      - **`po.increment`** keeps one value per input per group.
      - **The column form** holds what task 78 holds: one window of input
        rows as the input's own chunks, and the queues.
      - **The target form** holds, per waiting row, only the row's own
        columns the target expression reads.

      #### Numerics

      A running sum less the window's mean of it cancels: a day's notional
      of 1e10 against a window's 1e6 costs about four digits, as does a
      price of 100 less a mid of 100 against a window's 0.01. Polars does
      that subtraction, so the core cannot help unless the operator is
      itself relative: an option `relative_to=` that returns `mean − x_t`,
      computed from sums centred on each segment's anchor row (`S' = Σ w
      λ^d (x − x_anchor)`, shifted by the anchor's step times the weight
      when segments combine, `mean − x_t = S'/W + (x_anchor − x_t)`). Built
      only if a brute-force comparison on day-scale running sums shows the
      plain form losing digits a factor reads.

      #### Tests and measurement

      - Each operator against its oracle above, on irregular times, and
        against a brute-force loop in both directions with a window; the
        time-reversal identity; an operator on either side of a Polars
        operator; identical operators computed once and operators sharing a
        kernel sharing a queue (by `queued()`).
      - Every recipe above against a brute-force loop of its own
        definition: both VWAPs and both rates. Then the recipes against each
        other, on synthetic streams and on the real quotes and trades: how
        far apart they are as the window grows against the half-life and as
        the row spacing turns bursty -- the evidence for which recipe a
        factor or a target should use.
      - The VWAP as a ratio of sums against the prices-and-quantities form;
        a session change restarting `po.increment`.
      - Each non-element-wise expression refused while the plan is built,
        and every element-wise one in the tests accepted.
      - The target form equal to the column form fed back through the
        embargo, row by row (task 104).
      - Real interleaved data: Binance USD-M futures `bookTicker` (best bid
        and ask) with `trades`, one symbol-day, downloaded and cached under
        `.cache/` by `tests/data.py` (hard rule 1).
      - `scripts/windows_bench.py` extended: 1, 4 and 16 operators on one
        kernel against as many windows, at 16M rows; memory at the
        scan-and-sink floor, as task 78 measured.

      #### Built (the column form, 2026-10-03)

      - **The core** (`windows.rs`, rewritten): a kernel per distinct
        (direction, `half_life`, `window_size`, `closed`), any number of
        operators on it sharing one two-stack queue whose items carry one
        value and one accumulator per operator; the queue keeps suffix
        sums on its front stack and prefix sums on its back, so a window
        less its oldest rows is read without a subtraction. That is what
        a held value cut at the window's edge needs: the first attempt
        subtracted the boundary row's full mass from the sum, and at a row
        exactly a window old the two cancelled to rounding dust that set
        the mean (a `"left"` mean read 6.27 where the definition gave
        6.34). The forward side closes a window before the kernel's open
        row -- the last one, held until the next -- joins, and counts that
        row directly, so no queued row is ever cut.
      - **Each operator's intervals are its own** (measured 2026-10-03:
        Polars' `ewm_mean_by` on `[1, null, 3]` at `t = [0, 1, 2]` gives
        2.5, the recursion with `Δt` from the last *valued* row; the row
        form gives 2.0). Looking back a value is held from the operator's
        last valued row (`Group::last_valued`, a start per operator in
        each item); looking ahead the last value is held until the next
        (`Group::held`, forward-filled), and a held value counts only
        where the row holding it is in the window -- a forward mean starts
        at the first in-window row with a value of its own, since the rows
        before it hold the row's own value or one its stamp excludes
        (found by the brute force and the time-reversal identity both).
        `min_samples` counts rows that carried a value.
      - **Window boundaries in one form everywhere**: ages compared with
        the window as differences (`tau - t > w`), in the core and the
        brute force, since the sum form `t + w` and the difference form
        disagree at an exact boundary by one ulp (a row exactly a window
        later fell on different sides).
      - **The frame runner** (`windows_frame.rs`): four passes a chunk --
        Polars evaluates the increments' inputs, the runner takes each
        increment one row back within the group's session (a step back
        past `restart_after_step_back` starts it over; nanoseconds for a
        temporal input, seconds out) as hidden columns; Polars evaluates
        every operator input to a number; the core runs; Polars evaluates
        the formulas over the held rows with the operators as hidden
        columns, which are then dropped. Increments live in the runner,
        not the core, because an operator's input may contain one and the
        inputs are evaluated before the row loop. The output schema is the
        formulas' dtypes over an empty frame. Windows state version 2.
      - **The formula tree** (`formula.rs`, `_formula.py`): the list form
        of the design, read on both sides, refused by name otherwise; the
        operators are nodes with their parameters; `Over` is what Polars
        calls an `over` node, `RollingExpr` a rolling function.
      - **The call**: `with_windows(lf, *exprs, **named, clock=..., ...)`,
        an expression named by its keyword or `.alias()` (an unnamed one
        whose name would be an operator's is refused), the clock keywords
        refused as expressions (3(a)), `like=` as before. `po.window`,
        splits, weights, totals, `unlisted`, `complete` and name templates
        are gone (a side's VWAP is `when/then` inside both sums); so is the
        `complete` column -- `partial="null"` shows a cut window as null.
      - **Tests**: Rust, every operator on nine kernels against a brute
        force from the definitions on five streams (gaps, repeated stamps,
        sessions, missing values, groups), chunking bit for bit, the
        time-reversal identity, the Polars `rolling_sum_by` membership
        table under each `closed`, the recursion, resets and cuts, the
        state at every row; Python (`tests/test_windows.py`, 58), each
        operator under each `closed` against a loop from the definition,
        the recursions written out, Polars' `ewm_mean_by`, `ewm_sum_by`
        (1.44.1 on) and `rolling_sum_by` as second opinions, the mirror,
        the three VWAPs as ratios of sums (unchanged by a print split in
        two or a zero-volume row), the VWAP from running sums' increments
        equal to the one from prices and quantities, compositions on
        either side of a Polars operator and their dtypes, the round trip
        of every node kind, every refusal by name, the clock events,
        `like=`, chunk invariance, resuming, slices and projections, the
        namespaces, a model fed in one query, durations, and one real
        Binance day (2.55M rows, 52% of trades sharing a millisecond with
        a quote) through four formulas.
      - **Measured** (`scripts/windows_bench.py`, 2026-10-03; PERFORMANCE
        §33): memory at the scan-and-sink floor at every size, sixteen
        operators on one kernel in 57% of sixteen kernels'
        time; against task 78's core rebuilt from `c061b7a` in a worktree
        and run beside it at one load, 1.5 times its time at 16M rows (2.44 and 2.49 s against 1.60 and 1.65 s, alternating), memory equal -- the constraint
        "the speed of task 78's core" is not met on the one case the old
        core could run, and the general engine is the reason. Three
        costs were profiled (an unstripped build under the sampler) and
        removed on the way: seven `Vec`s a row in the first queue (a
        struct-of-arrays queue with flat arenas), the nanosecond clock's
        `i128` division and conversion on every step (an `i64` fast path,
        bit-identical, library-wide), and a Polars plan for a bare-column
        input or a bare-operator formula (skipped; it measured nothing).
        The user's call whether the remaining gap is worth a further
        restructuring (the open-row bookkeeping and the two clock advances
        a row are the next items).

- [x] 144. **Window expressions take Polars' names and semantics -- the
      names built 2026-10-02 (the two tables below, "Built" at the end);
      the window semantics built 2026-10-03 inside task 143's core:
      `closed` (right, the default; left; both; none), `min_samples`, a row
      exactly `window_size` later counted under right and both, every row at
      one stamp sharing one backward window and waiting for the next
      distinct stamp under right and both, held to Polars' `rolling_sum_by`
      membership table and to the brute force in both languages (task
      143's "Built").** Where
      Polars has a parameter for the same thing, task 143's operators take
      its name, values, default and meaning, and the specs take
      `half_life`; the rest of the public names follow (below). Size L,
      with 143.

    | Polars (`rolling_mean_by`, `ewm_mean_by`) | the window operators |
    |---|---|
    | `half_life` | `half_life` |
    | `window_size` | `window_size` |
    | `closed="right"`, or `"left"`, `"both"`, `"none"` | the same: backward `(t − w, t]`, forward `(t, t + w]` |
    | `min_samples=1` | the same: fewer counted rows gives null |

      - **The forward window counts a row exactly `window_size` later.** As
        a target, the model learns row *t*'s label when the first row at or
        past *t* + w arrives, counting that row when it lands on the
        boundary; its value is known then.
      - **Equal timestamps follow Polars**: a window is a set of timestamps.
        A backward window counts later rows at its own timestamp, so its
        output waits for the next distinct one; under `"right"` a forward
        window counts none at the row's own timestamp. Stream order at one
        timestamp is an opt-in (`same_clock="include"`), for data whose order
        within a timestamp is known; in Binance's files the order of a trade
        and a quote update in the same millisecond is not.
      - **Where Polars has no counterpart, task 78's rules hold**: a gap
        past `max_dclock` or a session change cuts a window and a reset
        discards it, under `partial`; a forward row whose window has not
        closed when the input ends is null, where Polars would compute a
        partial window.
      - **`halflife` becomes `half_life` library-wide**, with no alias, in
        every parameter that spells it: `half_life`, `long_half_life`,
        `coef_half_life`, `revert_half_life`, `select_half_life`,
        `level_half_life` and `trend_half_life`, on every spec builder, in the
        Rust `Spec`, the command line's TOML (`examples/bank.toml`), error
        messages, docstrings and docs (prose says half-life). The output
        fields' `@h` instance label is unchanged. A saved state stores its
        specs, so a state saved before the rename does not load: raise the
        bank's `MIN_SCHEMA_VERSION` (pre-1.0 state compatibility is waived).
        Ships in the release that renames `label_delay` to `embargo` (task
        104), so specs break once.

      #### The rest of the public names (decided 2026-10-02, the user: "Add all")

      Each name was checked against Polars, against the rest of the API,
      and against what it does. All ship in the same release as
      `half_life` and `embargo`, with no aliases.

    | now | becomes | why |
    |---|---|---|
    | `max_dclock` | `gap_cap` | "dclock" is internal shorthand. It caps how much clock one step counts, and a capped step is a break; "gap" is the library's word, as in `session_gap`. Not `max_gap`, which reads as a limit on the gaps allowed. |
    | `window` (specs) | `window_size` | Polars' name, and the window operators'. `window_every`, `window_budget` and `window_lags` keep theirs. |
    | `n_eff` | `weight_sum` | It is a weight, not a sample size (task 148); in statistics n_eff is Kish's effective sample size, which this library calls `n_kish`. In every output field (`weight_sum@h<h>`), the `closed_groups` columns (`pair_n_eff` becomes `pair_weight_sum`), the `gram()` keys and CLAUDE.md's rule 8. |
    | `min_periods` | `min_weight` | It compares that weight. Polars renamed its own `min_periods` to `min_samples`, which counts rows, so neither name says what ours does. |
    | `resid_z_<t>`, `emit_resid_z` | `zscore_<t>`, `emit_zscore` | `resid_` is a prefix of `resid_z_`, so a target named `z_y` made `resid_z_y` two fields (measured 2026-10-02: the builder accepted the spec and listed it twice, and `fit_predict` refused it). The one prefix collision in the output grammar. |
    | `pcorr_<a>_<b>` (`ew_cov`) | `partial_corr_<a>_<b>` | The stat is asked for as `"partial_corr"`. |
    | `scale_features` (`sgd`) | `standardize` | Every other model's name for it. |
    | `on_clock_reset` + `min_backwards_jump` | `restart_after_step_back` | One rule in two vocabularies, the second required by one value of the first and refused by the other. Unset, a step back is an error; given a clock amount, a step back larger than it restarts the state and one no larger is still refused (inclusive, as built; this row once said "at least that large", corrected in review R1). In the specs and in `with_windows`. A clock rule, so it gets the extra review task 120 calls for. |
    | `ridge_decay` (bool) | `ridge_scale="mean"` or `"sum"` | Names the difference: a penalty on the mean moments, permanent, against a prior on the sums, fading (task 151's two families). Default `"mean"`. |
    | `lam_selected_<t>` (`lasso`) | `penalty_selected_<t>` | `lam` is the decay factor. It keeps that name: Polars' alternative to `half_life` is `alpha`, which `ftrl`, `deco` and `corrchange` already use. |
    | `add_intercept` | `fit_intercept` | scikit-learn's name for the same switch. |
    | `max_cd_iters`, `cd_tol` (`lasso`) | `max_iter`, `tol` | scikit-learn's `Lasso`. |
    | `reset` (`corrchange`) | `reset_on_flag` | "Reset" already names a clock event. |

      - **Everywhere a name appears**: the spec builders, the Rust `Spec`,
        the command line's TOML (`examples/bank.toml`), `with_windows` and
        its frame methods, error messages, output fields, `closed_groups`
        columns, `gram()` keys, docstrings, the README, the reference pages,
        `llms.txt` and CLAUDE.md. The rest of this plan keeps the old names
        in entries written before this one; what they build takes the new.
      - **An old name is refused with a message naming the new one**
        (`max_dclock was renamed gap_cap`), in Python and in TOML: no alias,
        and no bare "unknown field". Outputs simply change (the user,
        2026-10-02: no backward compatibility there): nothing reads an old
        field name back, so there is nothing to refuse.
      - **The builder refuses two outputs that render to one field name**,
        as the bank already does at `fit_predict`; `output_fields` listed the
        duplicate. The message names the inputs that collided, features
        included: it said "rename a target or grid label" whatever the
        cause, and for `pc0_var` the cause was a feature.

      #### The outputs (decided 2026-10-02, the user: "Yes")

      Beyond the outputs in the table above (`weight_sum`, `zscore_`,
      `partial_corr_`, `penalty_selected_`):

    | now | becomes | why |
    |---|---|---|
    | `n_eff` in `closed_groups` (and `pair_n_eff`), `coef()`, `last_row()`, `marginal()`, `gram()` keys and `po.eval`'s sums | `weight_sum` (`pair_weight_sum`) | `summary()` already calls this number `weight_sum`, so it had two names. |
    | `withheld_reason` value `below_min_periods` | `below_min_weight` | Follows the parameter. |
    | `coef()` column `lambda` | `penalty` | Follows `penalty_selected_<t>`. |
    | PCA loadings `pc<j>_<feature>` | `pc<j>_loading_<feature>` | A feature named `var`, `share` or `score` collided with the component's own fields (measured 2026-10-02: `pc0_var` twice, refused at `fit_predict`). |
    | `bocpd`'s `logscore` | `loglik` | `hmm` and `deco` call the same quantity, the row's log predictive density, `loglik`. |
    | `hmm`'s `p_<k>`, `p1_<k>` | `filtered_<k>`, `predicted_<k>` | `p1` is cryptic; the docs call them the filtered and the predicted state. |
    | `absresid_q<p>_<slot>` | `abs_resid_q<p>_<slot>` | The one fused word in the grammar; matches `mahal_q<p>`. |
    | `micro`'s field `micro` | `micro_id` | The id of the summary the row goes to, not its model's name. |

      Kept, checked: `pred_`, `resid_`, `sigma_`, `lo_`, `hi_`, `coverage_`,
      `ic_`, `r2_`, `hit_rate_`, `autocorr_`, `drift_`, `selected_`,
      `settled_frac`, `support_coef`, `error_inflation_`, the slot labels
      (`__<set>`, `_r`, `__l`, `@h`), `describe()`'s columns (Polars' own
      `describe` names), `closed_groups`' counters and clock range, and the
      standard names of `rcov` (`omega2`, `iq`) and `seqtest` (`log_e_`).
      - Kept, deliberately: `clock`, `session`, `session_gap`, `group`,
        `group_close`, `weight`, `coef_every`, the `emit_*` family, the `@h`
        labels, the models' standard notation (`q`, `p0`, `obs_var`; FTRL's
        and DCC's `alpha` and `beta`; DenStream's `beta_mu`; PA's `c`), and
        the model names (`ew_cov`, `ew_class` against Polars' `ewm_`: a
        model rename breaks more than any parameter).
      - Specs are stored in saved states, so the bank's
        `MIN_SCHEMA_VERSION` rises once for all of it.

      #### Built (the names, 2026-10-02)

      Every row of both tables, plus: `po.eval.rolling_metrics` takes
      `window_size` too (the same concept, Polars' name); `ridge_scale` is
      a two-value enum (`"mean"` default, `"sum"`), where `ridge_decay` was
      a bool. Schema 22: the bank refuses a file older than 22 by version
      (`MIN_BANK_SCHEMA_VERSION`); the models' own states did not change
      and the core minimum stays 14. The Rust accessor keeps the name
      `n_eff` (CLAUDE.md rule 8 says so): only the emitted field is
      `weight_sum`. An old name is refused naming the new one from the spec
      builders (`_RENAMED` in `_spec.py`) and, through
      `online_polars::name_renamed`, from a spec dict, a windows config and
      the command line's TOML, at the spec's level and the model's;
      `with_windows`' own Python keywords refuse an old name as any Python
      function does, by `TypeError`, since its signature is explicit. The
      builder refuses two outputs rendered to one field name, naming the
      inputs (`duplicate_field`): the one collision the columns' own names
      can still make is `ew_cov`'s `corr_a_b_c` over `a_b, c` and over `a,
      b_c`. The rename ran as a script over the live tree (code as
      identifiers; prose outside code spans says half-life), with history
      files and this plan's earlier entries left as written.
      `docs/OUTPUTS.md` and `tests/api_surface.txt` are regenerated. Tests:
      `tests/test_renames.py` (the builders' table against this one, every
      old name refused from a builder and from a dict, no old output name
      in the release probe's workload, the new names, the old collisions
      gone, the duplicate refusal naming the inputs, the same number under
      one name in every frame, `restart_after_step_back` as one rule), the
      Rust `an_old_name_is_refused_naming_the_new_one` (JSON and TOML),
      `spec_defaults.rs` for the one clock rule, and every existing test
      under the new names. Cost: none -- no hot path changed. Measured
      2026-10-02 against the task 152 build (`ewridge`, k=20, 400k rows,
      load 4.8): 3.01M rows/s against 3.01M, 3.02M with `emit_clocks`
      against 3.02M, 2.01M under `embargo=10` against 2.00M; state 10,029
      bytes, unchanged; peak RSS 531 MB (the first RSS figure recorded, the
      baseline for the next task).

      Tests: each `closed` value in both directions against Polars'
      `rolling` on the same rows, equal timestamps included; `min_samples`
      against Polars'; a backward window's output waiting for the next
      distinct timestamp, chunked anywhere. For the names: the API surface
      snapshot's diff is exactly this table; every old name refused naming
      the new one, from Python and from TOML; no old name left in the live
      docs (a search, never truncated); a target named `z_y` beside `y`
      accepted with `emit_zscore`; two outputs with one name refused by the
      builder, naming the inputs; a feature named `var`, `share` or `score`
      accepted beside `pca`; the same number under one name in every frame
      (`weight_sum` in the fields, `summary()`, `closed_groups`, `coef()`,
      `last_row()`, `marginal()`, `gram()` and `po.eval`).

- [x] 145. **`session_shrink` mixes data, not weight -- built
      2026-10-02.** Size S. `ewridge`'s blend toward its slow twin at a
      session change gave the twin the share `f W_slow / ((1 − f) W_fast + f
      W_slow)` and the weight `(1 − f) W_fast + f W_slow`. The twin's weight
      is many times the fast sums', so every `f` above 0 reverted almost
      fully: measured on 0.13.0 with a 100-row half-life against a
      10,000-row twin, the slope moved from 2.008 to 1.113 at `f = 0.25`
      (1.083 at `f = 1`), and `n_eff` from 145 to 2,855.

      - **The blend mixes the moments as a mixture of the two data sets**,
        in fixed shares `1 − f` of today's and `f` of the long run's: `m' =
        (1 − f) m_fast + f m_slow` for the feature and target means, and
        `C' = (1 − f) C_fast + f C_slow + f (1 − f)(m_fast − m_slow)(m_fast
        − m_slow)'` for the centred moments, the cross-moments alike. The
        fit is the ridge regression on that mixture.
      - **Every weight-like quantity is kept**: `W' = W_fast`, each
        target's own weight, the Kish sum of squared weights, the
        `ridge_decay` prior scale and the weight learned since the last
        solve, so `n_eff`, the Kish size, the warm-up gates and the solve
        cadence are unchanged across the blend and the fast decay carries on
        from the open. Today the inflated weight also delays the next solve:
        measured, the first re-solve after the open came 32 rows in at `f =
        0.25`, against 3 at `f = 0`, so the session's first rows are scored
        with stale coefficients.
      - So `f = 0` keeps today's fit, `f = 1` gives the twin's moments at
        today's weight -- the twin's fit exactly, unless `ridge_decay`, whose
        prior keeps today's scale -- and `f` between fits on that share of
        the long run. The coefficients move by about that share where the
        feature distributions match; they are not linear in the moments.
      - **In `Grams::blend` and `Acc::blend`**
        (`crates/online-core/src/gaps.rs`) and the moment blends they call:
        the shares `(1 − f, f)` in place of the weight shares, and the fast
        weights kept; `TargetMoments::blend` leaves `Q` as it is. The state
        layout is unchanged, so no schema bump. Numbers move for any `f`
        above 0: a CHANGELOG line, and the docstrings state the formulas.

      Tests: the blended coefficients from the raw rows, the two kernels
      each normalised and mixed, solved by `faer` independent of
      `solve.rs`, at `f` of 0.25, 0.5 and 1 (`f = 1` is the twin's fit), the
      weight bit-equal across the blend; under `ridge_decay` the prior scale
      bit-equal and `f = 1` the twin's moments at today's weight and prior,
      from the rows; the mixture term by term at the origin and at 1e8, each
      target's own weight kept; `f = 0` a no-op. In Python: the slope at the
      open within 0.05 of `(1 − f)` today's plus `f` the long run's, `n_eff`,
      the Kish size and the first re-solve equal with and without the
      blend, and `TestSessionShrinkBlend`'s numpy oracle at the normalised
      kernels. `tests/reference_paths.py` keeps a weight per target beside
      the Gram's, since each normalises over its own rows; seeded with the
      old blend its oracle misses by 3.5e-2 to 8.4e-2, with the Gram's
      normalisation for the targets by 5.0e-4 to 9.1e-4.

- [x] 146. **A clock parameter acts on the clock -- built 2026-10-02,
      but for `kalman`'s `coef_halflife`, which is task 150.** Size M.
      Every parameter in clock units acts on the policy clock, so its
      effect does not change when rows arrive more or less densely at the
      same clock; anything that counts rows says so in its docs, and a lag
      counts rows only between gaps. Measured on 0.13.0 where marked; the
      rest found by reading the code, and measured as each is fixed.
      - **Drift detection** (measured): `emit_drift`'s Page-Hinkley test
        added one excess per scored row to an undecayed mean, so the same
        30-unit burst flagged at 4 rows a unit and not at 1. It now
        accumulates `d · (e − mean − δ)` against a mean decayed at the
        model's halflife, so `drift_threshold` is in σ times clock units; a
        row with no residual ages the mean. At rows one unit apart with no
        decay it is the classic test, so its unit tests run unchanged with
        `d = 1`, `λ = 1`. WARMUP-AND-CONVERGENCE §7.1's coefficient CUSUM
        is to be built the same way from the start.
      - **`resid_quantiles` and `mahal_quantiles`** (measured): the P²
        estimator never forgot; at halflife 10 rows, 3,000 rows after the
        noise fell tenfold, `absresid_q0.9` read 1.55 where the recent
        quantile is 0.166. Replaced by `EwQuantile`
        (`crates/online-core/src/stats.rs`): DDSketch (Masson, Rim & Lee
        2019) with its bucket weights decayed on the clock, each value at
        its row's weight. A bucket is `64 e + ⌊64 m⌋` of `v = m 2^e`, exact
        on the float's bits, so no libm result enters the state; the
        representative is within `tanh(1/128)` (0.78%) of every value in
        its bucket, so of the exponentially weighted quantile. The decay is
        one factor folded into the buckets every 64 halvings, when an end
        bucket under `1e-12` of the total is dropped; each level keeps its
        bucket and the weight below it, so a row costs a few comparisons.
        One sketch per slot answers every level, from the first residual.
      - **`sgd` under `schedule="constant"`** (measured): the coefficients
        never decay, and `halflife` reaches only `n_eff`, the scaler and
        the AdaGrad sum, so halflife 10 and 10,000 gave identical slopes.
        Its memory is in rows, about `1 / (lr · E[z²])`, and the docs say
        so. `halflife` stays required: `n_eff`, and with it
        `min_periods`, always reads it.
      - **`deco`'s linear dynamics**: `alpha` and `beta` step once per row,
        so a capped weekend moves `ρ` as much as a millisecond; the docs
        say "on the model's own clock". They are per row, as a DCC model's
        are, and the docs say so.
      - **`hmm` transitions** apply once per row, a weekend one step, and
        the learned counts decay on the clock but grow per row, so the
        staying probability rises with row density. The docs say a
        transition is per row and what density does to it.
      - **`resid_autocorr_lag`** paired scored residuals, not rows, and was
        not cleared at a capped gap or a session change, so Friday's last
        residual met Monday's first. Its buffer is now cleared where the
        model's lag rings are, the docs say "scored residuals back", and
        the cross moment keeps a weight of its own, the pairs', so a value
        with no partner -- the first `lag` of a stream or of a run -- adds
        nothing to it rather than a zero product. Found while fixing it: a
        row with no residual, or of weight 0, skipped the tracker and its
        decay with it; it now ages it, and the drift detector's mean and
        the quantile sketch likewise.
      - **`conformal_rate`** steps the radius once per scored row, beside a
        coverage that decays on the clock; the docs say the step is per row.
      - **State schema 21.** The bank file keeps the stream's diagnostics,
        so it refuses one older than 21 (`MIN_BANK_SCHEMA_VERSION`). Of the
        models' own states only `ew_cov` with `mahal_quantiles` changed; it
        refuses one older than 21 by name, and the core minimum stays 14
        (the user, 2026-09-28).

      Tests: a burst found at the same clock at one and four rows a unit,
      in Rust and from Python; the detector against its recursion written
      out on irregular steps; the sketch within `tanh(1/128)` of the
      exponentially weighted quantile's definition row by row, on
      irregular steps and weights with zeros and a tenfold fall in scale,
      through rescaling and pruning, at the extreme floats, and saved and
      resumed; `absresid_q<p>` and `mahal_q<p>` against the same definition
      from the emitted residuals and distances; an autocorrelation null
      throughout when every row follows a break; a step with no value
      ageing the detector and the tracker as one step of the product's
      decay; an old `ew_cov` state with P² markers, written in msgpack,
      refused by name.

- [x] 150. **`kalman`'s `coef_half_life` on the clock -- built 2026-10-03,
      the recommended option.** Size S. Split from task 146. The process
      noise `q_i = σ² (ln 2 / h)²` was added per clock unit, but a Kalman
      gain grows with the square root of the noise, so the coefficients'
      memory was about `h · sqrt(d)` for rows `d` apart: adapting to a
      slope step took 74, 38 and 13 clock units at rows 1, 0.25 and 0.04
      apart with `coef_half_life=50`, where `ewridge` at half-life 50 took
      44 to 59 (measured on 0.13.0). The documented promise, EW-RLS's
      steady-state gain, holds at any spacing when the noise added per row
      is `σ² (ln 2 · d / h)²`: EW-RLS forgets `2^(-d/h)` over the step, a
      per-row half-life of `h/d` rows, and the matching is done at that
      half-life. **Built**: `P += Q · d²` per row (`kalman.rs`, the numpy
      reference, the README's equations, the builder's docstring); an
      explicit `q` is added as `q_i d²` too, and a reverting slot's
      stationary variance is `q_i d² / (1 − φ²)` at rows `d` apart. The
      match is first order in `d/h` (the exact per-row noise is `σ² g² /
      (1 − g)`, `g = 1 − 2^(−d/h)`: within 1% for rows closer than half a
      half-life, 4% at one). Bit-identical at unit spacing (the goldens'
      rows: the two values there that moved by 1e-15 are the standardizer's
      drift since the pins of `93ef404`, inside the 1e-12 check, not this
      task's), and the golden signatures move at the
      stream's gap row, re-pinned against the numpy reference -- the core's
      (`golden.rs`), the pipeline's (`tests/test_golden_pipeline.py`:
      `kalman`, `kalman_revert`, and the `seqtest` that compares the
      Kalman with the ridge) and the `filterpy` second opinion, whose
      `predict` now takes `Q d²` too. Tests:
      `crates/online-core/tests/kalman_clock.rs` -- the clock time to learn
      a slope step at rows 1, 0.25 and 0.04 apart (165, 176 and 170 clock
      units at `coef_half_life` 50, within 7% of each other, against
      `ewridge`'s 120, 113 and 116 at half-life 50; another stream and
      criterion than the 74/38/13 above, which are not comparable to these
      -- at unit spacing the two forms are identical) where the old form
      parted by a factor of five; the Kalman's 1.4 over `ewridge` is `R =
      σ²` inflating at the step before `P` catches up through `q ∝ σ²`,
      spacing-free; and a gap row adds exactly `q d²` to `P`, a unit row
      `q`. Not taken: keeping the random walk and documenting the
      `sqrt(d)`.

- [x] 147. **A row weight scales evidence, not counts -- built 2026-10-02,
      but for `ftrl` and the density half of `micro`, which are task 151.**
      Size M. Multiplying every row's weight by one constant changes no
      output -- the EW models already hold to it, since their sums are means
      -- unless the docs say otherwise for that model. Where a weight entered
      a count, it now enters as `w / w̄`, `w̄` the EW mean weight of the rows
      the quantity takes (this row's included, so the first row reads 1, and
      every row of a constant weight reads 1, which leaves those streams bit
      for bit as they were).

      - **`conformal_rate`** (measured): the step was `rate · σ · w · (miss
        − α)`, so the same rows at weight 100 swung the band: width sd 3.10
        against 0.21 at weight 1, mean 6.0 against 3.3. It is `rate · σ ·
        (w / w̄)`, `w̄` over the scored rows at the model's decay.
      - **`bocpd`** added the row weight to the posterior's counts (`κ`,
        `ν`) but not to the likelihood, so a heavy row made the predictive
        confident without counting as evidence. The weight enters both, as
        `w / w̄` over the rows learned from (nothing in `bocpd` decays):
        the run's statistics, and the message `πᵣ^{w/w̄}`, under `robust`
        times its β-power weight. `κ₀` and `ν₀` are in rows of the mean
        weight. What the row reports is read as a row of the mean weight,
        which is what `predict`, never told a weight, can say:
        `tests/model_contract.rs` held the first version, which reported
        at `w/w̄`, to `predict == step` and refused it.
      - **`quantile`**: the warm-up gate and the band floor compared the
        raw decayed weight with a row count, so small weights never left
        warm-up, and -- found by the property test -- rows at weight 100
        left it on their first row, where the fit, leaning on a one-row
        Gram, reached 1e51. Each target keeps its present rows counted one
        each (`nobs`), and the gate, the floor and the band's weight are
        read in rows.
      - **`micro`**: `beta_mu` and the pruning threshold `ξ` were absolute
        weights, and a summary admitted a unit row. All three take `w̄`, the
        EW mean weight of the rows learned from, so a heavier stream keeps
        no more summaries. The docs' `ξ` is corrected to what the code
        computes: the weight of a summary that took one row of the mean
        weight every `Tp` clock units since it opened, `Σ_{i ≤ a/Tp}
        2^(−i Tp/h)`. Its counts stay in rows, as DenStream's do, and the
        docs say a denser stream fills its summaries faster.
      - **`ftrl`**: built and reverted. Its weight is Vowpal Wabbit's
        importance weight, and `tests/test_second_opinion.py` holds the fit
        to VW's with weights, so penalties scaled by the mean weight
        parted from VW by design. The docs say a weight is an importance
        weight; the choice is task 151.

      Tests: `tests/test_weight_scale.py` fits every spec of the release
      probe's workload, every kind, at the stream's weights and at a
      hundred times them, `min_periods` scaled with them, and holds the
      specs whose outputs move to exactly the documented exceptions, each
      with its reason: `rls` (a sum-scale prior), `kalman` (a weight is an
      observation's precision), `sgd` (a step size), `ftrl` (an importance
      weight), `pa` (a weight above 1 counts as 1), `hmm` (counts against a
      Dirichlet pseudo-count) and `seqtest` (it reads `kalman`). In Rust,
      each fixed model at a hundred times the weights, and the conformal,
      `micro` and quantile oracles in Python with the same rule. The
      quantile test fails at row 2 when the gate reads raw weight.

- [x] 151. **`ftrl` and `micro` keep their units: a sum-scale prior and a
      point density -- decided and the docs written 2026-10-02.** Size S.
      Split from task 147. Nothing in the code changes.

      - **`ftrl`'s penalties stay in absolute weight.** FTRL-Proximal
        (McMahan et al. 2013) minimizes the cumulative loss plus a fixed
        regularizer, `g_{1:t}·w + λ₁‖w‖₁ + ½λ₂‖w‖²`: the penalty is a prior
        of fixed mass that the evidence outweighs, which is what its regret
        bound rests on, and an importance weight adds evidence, as Vowpal
        Wabbit's does. Under a halflife the evidence is bounded at `W∞`,
        so the effective regularization `λ₁/W∞` depends on the weights'
        scale and the rows' density: the sum-scale prior family, beside
        `rls`'s ridge, `ridge_decay` and `kalman`'s observation precision,
        which the library documents as the exceptions. The mean-scale
        sparse fit, invariant to both, is `lasso`. Penalties times the
        taught rows' mean weight, built on 2026-10-02, were `lasso`'s
        semantics under `ftrl`'s name and parted from VW's weighted fit;
        reverted. Docs: `ftrl` says its `l1`, `l2` and `beta` are a prior
        of fixed mass against evidence that grows with weight and
        density, and points to `lasso` for a penalty on the mean scale.
      - **`micro`'s thresholds stay in points.** DenStream (Cao et al.
        2006) is DBSCAN on a stream: a cluster is a region with at least
        `MinPts` points, the count is the definition of density, and the
        paper takes the arrival rate `v` as the stream's given -- the
        total weight is `W = v/(1 − 2^{−λ})`, and `ξ` is the weight of a
        cluster fed one point per time unit. A threshold as a share of the
        running weight would be density-free and a different algorithm: a
        cluster holding 5% of a stream that splits into twenty falls below
        a share while still dense. Task 147's `w/w̄` keeps points as the
        unit. Docs: with halflife `h` and `v` rows per clock unit the
        stream's steady-state weight is about `1.44·v·h`, so a summary
        meant to hold a share `s` of it needs `beta_mu ≈ 1.44·s·v·h`.
      - **A clock-rate normaliser is not built.** `max_dclock` bounds its
        dip at a capped gap to about `G/(1.44h)` and never removes it;
        skipping the rate's update on a `capped` or `session_changed` row
        would, with no new plumbing. Neither is wanted given the two
        points above.

      Tests: `tests/test_weight_scale.py` keeps `ftrl` among the named
      exceptions; a docstring test holds `ftrl`'s and `micro`'s docstrings
      to the two sentences above.

- [x] 152. **The clock a row was scored at, and the clock of the last row
      learned -- built 2026-10-02** (the user, 2026-10-02: "see the clock
      of the row that was scored and the clock of the last row learned at
      each scoring so we can see the embargo in action"). Size S–M. Works
      with today's `label_delay`; task 104's tests read it.

      - **`emit_clocks=True`** on any spec adds two fields to every scored
        row: `scored_clock`, the row's own clock, and `learned_clock`, the
        clock of the newest row the models had learned from when the row was
        scored, before the row's own update. Without a delay that is the
        previous row learned; under `label_delay` (task 104's `embargo`) it
        is the newest row whose delay had elapsed.
      - **The clock column's own values and type**: a numeric clock as a
        float, a temporal one as a `Datetime` in its own unit and zone,
        exact in integer nanoseconds (never an epoch read into a double, the
        task 88 rule); with no clock column, the row's index in its group,
        the unit the bank counts, as an integer. The first temporal field in
        the output struct.
      - **Learned** means released into the models at a positive weight:
        a zero-weight row advances the clock and teaches nothing, so it is
        never the learned row, and the doubled stream's scoring copies
        (`po.stream.embargo`) never show as learned. A row with a null target
        still entered the state (`n_eff`), so it counts.
      - **Null** on a row the spec skips; `learned_clock` is null before the
        stream's first learned row, and after a reset until the next. One
        pair per spec, no `@h` suffix: every instance of a grid learns the
        same rows at the same time (a drift reset restarts a model, not the
        release).
      - **The delay counts elapsed time on the clock column** (task 153),
        so on every row `scored_clock` less `learned_clock` is at least the
        delay, and the next held row is less than one delay back. Where a
        session change restarts the clock, the two clocks are in different
        sessions' units and the session's gap counts between them; the docs
        say so.
      - **`predict`** scores against the saved state, so `learned_clock` is
        the fit's last learned row on every row it scores.
      - **The state** keeps the last learned row's clock, and each held row
        its own, as a `ClockValue` (`#[serde(default)]`; schema 21 is not
        released yet, so no bump). Chunk-invariant like every field.

      **Measured 2026-10-02** (`ewridge`, k=20, 400k rows, machine load
      5–6): 3.02M rows/s with `emit_clocks` against 3.01M without, +28
      bytes of state; with `label_delay=10` 2.00M against 2.06M, and 0.12.0's
      1.92M on the same rows, so task 153's elapsed-time counting cost
      nothing either. The clock type is recorded only once a chunk is
      taken, so a refused chunk changes nothing (`chunk_plan.rs` held it).

      Tests: without a delay, `learned_clock` is the previous accepted row
      of positive weight; under `label_delay` the newest row whose delay
      had passed in elapsed time, row by row from the frame; a
      `Datetime` clock exact to the nanosecond in each unit; a reset and a
      group's first row null; a zero-weight row never shown as learned;
      chunk invariance, save and load, `predict`.

- [x] 153. **`label_delay` learned labels early at a break -- built
      2026-10-02** (the user: "Why not simply consider the max dclock or
      session gap a time delta and deliver rows as defined by it?", then
      "Yes" to fixing it now). A break -- a gap past `max_dclock`, or a
      session change -- released every held row at once, because its events
      (the lag rings' clear, `corrchange`'s span, `session_shrink`'s blend)
      ran when the break's row arrived, and a held row learned after them
      would have been paired across the break or blended (review L2). So a
      forward label was learned before it was known wherever a break was
      shorter than the delay: measured with a 10-unit delay and a 5-unit
      cap, an 8-unit gap learned two labels early, and a session change with
      no gap learned nine. Capping the delay at `max_dclock`, as task 104
      proposed, closed the gaps and not the sessions.

      - **The delay counts elapsed time**, `ClockAdvance::elapsed`: the
        clock column's own steps, uncapped, skipped rows' time included, and
        `session_gap` where a session change restarts the clock (a clock may
        restart at a session; `max_dclock` and `session_gap` say how much a
        model forgets across a break, not how long it lasted -- counted as
        time, a session gap longer than its real step released rows early).
      - **A break's events wait in the buffer with the row after it** -- or,
        raised on skipped rows, with the next accepted row, carried across a
        chunk boundary in the stream's state (`HeldBreak`) -- and run when
        that row is learned, after every row before the break. The models
        run one delay behind, events included. A reset still acts on arrival
        and drops the buffer.
      - The data summary counts a break where it arrived. A spec without a
        delay is unchanged, bit for bit; where a break is longer than the
        delay, as overnight, the release is where it was. New fields are
        `#[serde(default)]` within the unreleased schema 21.

      Tests: at every row, the rows learned are the rows whose delay has
      passed in elapsed time, counted from the frame, across a capped gap
      shorter than the delay, a session change with no gap and a clock that
      restarts at a session; the delayed bank ends as a plain bank fed the
      matured rows, with breaks inside the last delay; `ew_cov`'s lagged
      co-moments are the plain run's over the matured rows with breaks
      shorter than the delay (L2 kept); a break on a skipped row survives any
      chunking, the carry dropped is caught at chunk sizes 1, 3 and 81;
      `ClockAdvance::elapsed` against hand-worked steps. The four count
      tests failed on the old build.

- [x] 148. **The docs say what the code does: the 2026-10-01 findings --
      built 2026-10-02.** Size S. Each was a wrong or missing sentence,
      corrected to the code, with the test that pins it.

      - **`ridge_decay` penalizes the intercept** (measured: `y = 5` read an
        intercept of 3.50 at row 20, and 4.95 at row 199, against 5.0
        without it, at `ridge=10`, halflife 50), where `ridge`'s docs said
        "never on the intercept"; and `coef_prior`'s intercept slot is read
        under it, and only there. The docs say so, in Python and Rust.
      - **`po.gram.solve`** cannot reproduce a `ridge_decay` or `coef_prior`
        fit, though its docs promise "the model's own algebra"; the docs
        name the exception.
      - **`n_eff`**: it settles at `1 / (1 − λ^d)` for rows `d` clock units
        apart, not the documented `1 / (1 − λ)` (8,657 at halflife 600 with
        rows 0.1 apart, where the docs said 866), and the Kish size
        likewise; it is a weight, not a sample size, so "the effective
        number of observations" becomes "the weight behind the state", and
        `po.corr.signal_share` takes a Kish size: its argument is
        `n_kish_blocks`, was `n_eff_blocks`.
      - **`emit_selected` and `emit_averaged`** compare a halflife grid's
        errors over windows of different lengths; the docs say so, and
        point to `lasso`'s `select_halflife` for a like-for-like choice.
      - **The README's claim** that three diagnostics take no row weight is
        stale: zero-weight rows are skipped, and every residual diagnostic
        parts from the doubled stream.
      - **`label_delay`**'s "one unit per accepted row" is every row of the
        group, a skipped one included, and a gap capped by `max_dclock`
        releases the held rows as a session change does; the Rust `hmm`
        docs' account of where the transition prior sits matches the code.

      Tests: a constant target's intercept under `ridge_decay`, with and
      without a `coef_prior` intercept; `po.gram.solve` off a `ridge_decay`
      and a `coef_prior` fit, on beside a plain one; `n_eff` and `n_kish` at
      `1 / (1 − λ^d)` and `(1 + λ^d) / (1 − λ^d)` at rows 0.1 apart; a
      halflife grid's `sigma` against the EW recursion at each slot's own
      halflife, and `selected` its argmin; a skipped row counted by a
      clockless `label_delay`; the spread of hand-computed EW correlations
      of noise explained by `1 / (n_kish − 3)` and twice the floor at
      `n_eff`. The docstring pass under `docs/WRITING.md` is task 149.

- [x] 149. **A docstring pass under `docs/WRITING.md` -- done
      (2026-10-03).** Size M. Task 148 put the facts right; the pass rewrites
      the docstrings to the README's rules: map first, tables over lists, a
      mechanism named rather than alluded to, sentences near 20 words,
      every fact checked against the code. Each page is drafted beside its
      old text, accounted by a diff of the backticked names, the numbers and
      the link targets (nothing lost; a stale fact corrected against the
      code is noted here), measured before and after, and checked by the
      doc tests and Sphinx `-W`.

      ### Task 149 -- the docstring pass under docs/WRITING.md (map, before any prose)
      The map, the baseline and the pages done follow this entry, at the
      document's own level.

**Task 149, step 1, measured 2026-10-03 before the pass** (`measure_docstrings.py`, the
public docstrings the reference renders: the spec module and its 20 builders,
`ModelBank`, the two `online` namespaces, `stream`, `ops`, `eval`, `po.target`):
93 docstrings, 1,540 sentences, a mean of 23.3 words, 116 sentences of 45 words
or more, 104 bullet lines. The worst by long sentences: `spec.marginal` (13 of
87), the `spec` module (11 of 148, 48 bullet lines), `spec.ewridge` (7),
`stream.with_windows` (5, mean 34.6), `spec.micro` (5), `stream.embargo` (4,
mean 35.3), the `ops` module (4, mean 33.2), the `stream` module (3, mean
41.6); `ModelBank.last_row` (mean 40.6), `lf.online.unnest` (35.3),
`ModelBank.fit` (32.2), `fit_predict` (33.2, 7 bullet lines),
`closed_groups` (31.2, 5 bullets).

**The reader and the altitude** (WRITING §0, §1): a docstring is a deep doc,
reached from the Sphinx reference or `help()`, by a reader who has the
README's words; it carries mechanisms, equations, units, defaults and
refusals, and the README keeps the consequence. Nothing is cut: a sentence
that leaves a docstring lands in the README, `docs/OUTPUTS.md`,
`docs/PERFORMANCE.md` or `CHANGELOG.md`, and the cross-reference stays.

**Task 149, the map, per page** (`←` marks what moves; the order is the order of work):

| page | docstrings | the map each keeps | what changes |
|---|---|---|---|
| `spec` module | 1 | what a spec is → the stream parameters (one entry each, why / what / units) → clock units (the table stays) → what a spec writes (fields, the grid block, the diagnostics table) → errors | long entries split at the idea (`gap_cap`, `restart_after_step_back`, `embargo`, `min_weight`); the diagnostics table's long cells split into two sentences each; no history |
| builders, linear family: `ewridge`, `rls`, `kalman`, `lasso`, `huber`, `quantile`, `sgd`, `pa`, `ftrl` | 9 | one paragraph on what the model is → the update equations as a code block → parameters (why / what / units, default stated) → output fields → example as code with comments → raises | sweeps become tables (`ewridge`'s solvers and gates, `lasso`'s path, `kalman`'s clock forms); measurements → one number with its PERFORMANCE section |
| builders, classification and tests: `ew_class`, `seqtest` | 2 | as above | as above |
| builders, unsupervised: `ew_cov`, `kmeans`, `micro`, `deco`, `bocpd`, `corrchange`, `hmm`, `rcov`, `marginal` | 9 | as above | `marginal`: `serial_rule`'s three rules → a table (rule, the bracket, null when); the lag and window sizes → a table; `shards` and `feature_moments` → one number each ← PERFORMANCE §25, §27; `bins` split into what it reports, how edges are learned, memory (a table) |
| `holt` | 1 | as above | -- |
| `ModelBank` | 1 class + its methods | the class: what a bank is → the three ways in → state; each method: what it does → arguments → returns → raises | `fit_predict`/`fit`/`last_row`/`closed_groups`: bullets of parallel things → tables; sentences split |
| `lf.online`, `df.online` | 2 namespaces | one line per method, the plan form first (WRITING §2) | `unnest`'s sentences split |
| `stream` module + `embargo`, `with_windows`, `refresh_time`, `sample` | 5 | what the transform is → the clock paragraph → resumes → raises | the clock paragraph's sweep of policies → a table (keyword, what it does, units); `with_windows`'s resume paragraph stays prose (an argument) |
| `ops` module + 7 operators | 8 | the semantics paragraph (membership rules) → the ratio paragraph → the operators | the membership rules → a table (closed, backward near/far, forward near/far); the exactness claim stays one sentence |
| `eval` | the metrics functions | what each computes → arguments → returns | sentences split |
| `po.target` | 1 | -- | sentences split |

**Steps 3 to 8** (WRITING §6): draft a page, run its checks, account by a
diff of the backticked names, numbers and link targets (a script in the
scratchpad, `account.py`, prints what the new text lost), measure again, and
the gate's doc tests (`tests/test_examples.py`, `tests/test_api_links.py`,
`tests/test_doc_structure.py`, Sphinx `-W`).

**Task 149, pages done.** The counts are sentences, the mean in words, and
the number of sentences of 45 words or more, before and after, by
`measure_pages.py` (the scratchpad's): headings, code blocks and list-tables
dropped, then each paragraph split on its own, as WRITING §6 says (a first
count that merged table cells overstated the after-counts). Nothing moved
out of a page unless the row says so; a stale fact corrected against the
code is named.

| page | before | after | moved or corrected |
|---|---|---|---|
| `spec` module | 124, 19.5, 4 | 166, 14.5, 0 | `gap_cap`'s "``embargo`` releases the rows it holds" at a break was stale since task 153 (a break releases nothing early, `stream.rs`): corrected; the two refusal enumerations became lists |
| `spec.marginal` | 90, 24.6, 12 | 110, 18.0, 1 | `serial_rule`'s three rules, the bins' two memory sizes and `window_lags`' snapshot sizes became tables; the raises a list |
| `spec.ewridge` | 75, 23.4, 5 | 81, 18.7, 1 | `ridge_scale`'s two settings, `target_gaps`' two settings and `window_budget`'s three cases became tables |
| `stream` module | 14, 21.4, 0 | 16, 17.8, 0 | the history paragraph (`polars_online.prep`, `time=`/`by=`) left for the CHANGELOG, which records the rename (0.12.0) |
| `stream.embargo` | 17, 31.1, 3 | 24, 20.2, 0 | the two row kinds became a table; the raises a list; "cheaper" named as what is spent |
| `stream.with_windows` | 25, 33.3, 6 | 36, 21.3, 0 | the clock keywords became a table |
| `spec.micro` | 47, 20.2, 4 | 52, 17.3, 0 | the two wrong values of `eps` became the failure-mode table (WRITING §7's own example) |
| `ops` module | 29, 25.7, 3 | 21, 19.7, 0 | the operators' definitions, the `closed` membership rules and `partial`'s three values became tables |
| `stream.refresh_time` | 31, 25.8, 2 | 44, 18.1, 0 | the raises became a list |
| `spec.ew_cov` | 49, 23.2, 2 | 54, 18.2, 0 | the four windowed caveats and the PCA's four fields became tables; "the cost" named as the work |
| all ten (batch one) | 501, 23.3, 41 (106 of 35 or more) | 604, 17.5, 2 (24) | |
| `spec.corrchange` | 72, 21.1, 3 | 85, 17.4, 1 | `boundary_gamma`'s three settings became a table; the raises a list |
| `spec.kmeans` | 37, 19.6, 3 | 40, 17.6, 0 | the four seeding rules became a table |
| `spec.bocpd` | 45, 22.4, 2 | 45, 18.2, 0 | the three emissions, the four prior parameters and the three breaks' `p_change` became tables; "costs" named as the work |
| `spec.ew_class` | 29, 22.0, 2 | 29, 19.8, 0 | the three covariance shapes became a table |
| `spec.deco` | 37, 20.9, 2 | 46, 16.7, 0 | the raises became a list |
| `spec.ftrl` | 31, 20.5, 2 | 37, 17.3, 0 | the wrap of "a prior of fixed mass" kept as `test_weight_scale.py` pins it |
| `spec.holt` | 27, 19.7, 1 | 31, 17.5, 0 | -- |
| `spec.hmm` | 45, 19.4, 1 | 53, 16.2, 0 | the raises became a list |
| `spec.seqtest` | 34, 19.7, 1 | 38, 17.7, 0 | -- |
| `ModelBank.fit` | 13, 22.5, 2 | 16, 19.6, 1 (a bold lead merged) | -- |
| `ModelBank.summary` | 20, 17.9, 2 | 21, 17.1, 1 (a header merged) | -- |
| `ModelBank.last_row` | 7, 27.0, 1 | 9, 19.1, 0 | -- |
| `ModelBank.skip_learned` | 10, 23.1, 1 | 13, 16.8, 0 | the raises became a list |
| `ModelBank.fit_predict` | 20, 19.0, 1 | 21, 18.2, 1 (a lead-in merged) | -- |
| `ModelBank.closed_groups` | 24, 23.2, 1 | 18, 20.5, 0 | the blocks per kind became a table |
| `lf.online.unnest` | 8, 25.0, 2 | 14, 14.0, 0 | the errors became a list |
| `lf.online.predict` | 7, 26.9, 1 | 12, 15.8, 0 | the errors became a list |
| `eval.rolling_metrics` | 3, 30.7, 1 | 7, 14.6, 0 | the raises became a list |
| `eval.seqtest` | 13, 21.5, 1 | 18, 15.5, 0 | the raises became a list |
| `ops.increment` | 3, 37.3, 1 | 5, 22.2, 0 | -- |
| all twenty (batch two) | 485, 21.1, 31 (87 of 35 or more) | 558, 17.5, 4 (28) | the four are a bold lead sentence and definition-list headers the splitter merges with their bodies |
| `spec.kalman` | 35, 18.3, 1 | 36, 17.8, 0 | "costs nothing" named as what is added |
| `spec.lasso` | 36, 17.8, 1 | 38, 17.0, 0 | "costs nothing" named as the solve it saves |
| `spec.coef_fields` | 12, 16.3, 1 | 13, 15.3, 0 | -- |
| `eval.metrics` | 13, 15.8, 1 | 16, 12.8, 0 | -- |
| `eval.unpack` | 9, 15.6, 1 | 12, 12.0, 0 | the raises became a list |
| `eval.sums` | 16, 14.8, 1 | 18, 13.6, 0 | "pays for" named as the term it takes |
| all six (batch three) | 121, 17.0, 6 (13 of 35 or more) | 133, 15.6, 0 (6) | |

**Task 149, steps 6 to 8 (2026-10-03).** Measured again over all 93 public
docstrings, side by side, with `measure_docstrings.py` dropping headings,
code blocks and list-tables and splitting each paragraph on its own (the
first count of 116 sentences of 45 words or more merged text across table
cells; this count is the fair one at both ends):

| | sentences | mean words | 45 or more | 35 or more |
|---|---:|---:|---:|---:|
| before the pass (`aa30f00`) | 1,666 | 20.7 | 78 | 261 |
| after it | 1,854 | 17.7 | 6 | 113 |

The remaining sentences of 45 or more are the splitter's: a bold lead
sentence (`**With one exception...**`) and definition-list headers joined
to their bodies. Thirty-six pages were rewritten (every page with such a
sentence); the other 57 already measured within the rules and were left as
they were, each a reference entry near 20 words a sentence. Step 7 (render)
is Sphinx `-W` over the reference, which the gate runs; step 8 is the gate's
doc tests (`test_examples`, `test_api_links`, `test_doc_structure`,
`test_api_surface`), the docstring-pin tests (`test_weight_scale`,
`test_temporal_clock`) and Sphinx. Done.

- [x] 154. **The reader-facing documents rewritten after tasks 139 to 153,
      104 and the review rounds -- requested 2026-10-03.** Size L. The
      user's words: "Rewrite each user facing non requirements Md document
      ensuring that it has all the necessary sections after the recent
      additions. Review the organization of the sections to keep
      similarities together and ensure that important information is
      presented first in each section and that all phrasing and
      documentation rules are followed." Task 138 rewrote the README on
      2026-09-29, at `5e96018`; since then came tasks 139 to 153 and 104
      and nine review rounds of the window operators (§14), each adding
      text where it was built. This pass runs `docs/WRITING.md` §6 over
      every guide, against the code at `df555b2`.

      ### Task 154 -- the documentation pass (scope, baseline and maps, before any prose)
      The scope, the baseline and each document's map follow this entry, at
      the document's own level.

**Task 154, the scope.** `docs/README.md` sorts the documents into guides,
read to do something, and records, which "are not rewritten when the code
moves on". The pass takes the README, every guide, the changelog's
unreleased entry and the three reader files at the top of the repository.

| document | reader | in the pass because |
|---|---|---|
| `README.md` | a user | the user guide |
| `docs/RUNNER.md`, `docs/STATE-WORKFLOW.md`, `docs/OUTPUTS.md`, `docs/REGIMES.md`, `docs/PERFORMANCE.md` | a user | the guides to using the library. STATE-WORKFLOW's research, §0 to §9, is a dated record inside a guide and keeps its text; OUTPUTS is generated, so its generator is what changes |
| `docs/EXTENDING.md`, `docs/TESTING.md`, `docs/RELEASE-READINESS.md`, `CONTRIBUTING.md`, `SECURITY.md`, `.github/PULL_REQUEST_TEMPLATE.md` | a contributor, a reporter | the guides to changing the library |
| `docs/WRITING.md` | a writer of these documents | the guide to writing them; its rules stand, and it gains what task 149 learned about docstrings |
| `docs/README.md` | anyone | the index |
| `CHANGELOG.md`, the unreleased entry | a user upgrading from 0.13.0 | the released entries are records |

| left out | because |
|---|---|
| `docs/PLAN.md`, `docs/ENHANCEMENTS.md`, the designs as built (`ANSWERS-E54-E64`, `MARGINAL-LAGS-AND-BINS`, `MARGINAL-AT-WIDTH`, `WARMUP-AND-CONVERGENCE`) | requirements and design records |
| the surveys, prototypes and reviews (`BEYOND-O-STATE`, `BOOSTED-TREES`, `CLUSTERING`, `ARROW-SOURCES`, `SIMPLIFICATION`, `IMPROVEMENTS`, every `REVIEW-*`) | records |
| `docs/PHRASING.md` | the log of reports, which keeps each report verbatim (WRITING §6) |
| `docs/VALIDATION.md` | generated by `scripts/validate.py` from runs on public data, never edited by hand |
| `CODE_OF_CONDUCT.md` | the Contributor Covenant 2.1 as adopted |
| `CLAUDE.md`, `llms.txt` | the agent's instructions, and the agents' map, which is not Markdown; `llms.txt` changes only where an anchor it uses moves |

**Task 154, step 1, measured 2026-10-03 before the pass** (`measure_md.py` in
the scratchpad: headings, tables, code blocks and HTML dropped, each
paragraph and list item split on its own, a bold lead split from the
sentence after it; the cost words counted everywhere, though
PERFORMANCE's heading is their frame):

| document | prose words | sentences | mean | 35+ | 45+ | cost words | *X, not Y* | tables | code blocks | bullets | `##` sections |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `README.md` | 15,805 | 843 | 18.7 | 18 | 6 | 7 | 16 | 49 | 90 (63 python) | 2 | 10 |
| `docs/RUNNER.md` | 1,146 | 67 | 17.1 | 0 | 0 | 0 | 0 | 3 | 8 | 0 | 4 |
| `docs/STATE-WORKFLOW.md` | 3,158 | 190 | 16.6 | 2 | 1 | 0 | 0 | 12 | 4 | 5 | 2 |
| `docs/OUTPUTS.md` | 302 | 19 | 15.9 | 0 | 0 | 0 | 0 | 24 | 0 | 0 | 22 |
| `docs/REGIMES.md` | 4,266 | 232 | 18.4 | 7 | 2 | 2 | 1 | 21 | 1 | 0 | 11 |
| `docs/PERFORMANCE.md` | 23,798 | 1,320 | 18.0 | 48 | 24 | 98 | 43 | 132 | 6 | 39 | 37 |
| `docs/EXTENDING.md` | 2,831 | 181 | 15.6 | 5 | 2 | 0 | 6 | 13 | 5 | 0 | 6 |
| `docs/TESTING.md` | 8,168 | 529 | 15.4 | 17 | 10 | 3 | 13 | 32 | 0 | 6 | 5 |
| `docs/RELEASE-READINESS.md` | 8,709 | 540 | 16.1 | 9 | 4 | 9 | 18 | 27 | 7 | 37 | 5 |
| `docs/WRITING.md` | 2,786 | 173 | 16.1 | 4 | 0 | 11 | 10 | 12 | 0 | 0 | 9 |
| `docs/README.md` | 134 | 10 | 13.4 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 4 |
| `CONTRIBUTING.md` | 684 | 62 | 11.0 | 0 | 0 | 0 | 3 | 0 | 1 | 24 | 8 |
| `SECURITY.md` | 231 | 18 | 12.8 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 3 |
| `.github/PULL_REQUEST_TEMPLATE.md` | 69 | 4 | 17.2 | 0 | 0 | 0 | 1 | 0 | 0 | 4 | 3 |

Task 138 left the README at no sentence of 45 words or more and two of 35;
the tasks since brought six and eighteen back.

**Task 154, the README's map** (`←` marks what moves). The ten sections
become eleven: *How a bank sees a stream* opens by saying that every
parameter in it belongs to every model, and the grid for series that tick
at their own times and the windowed means are neither. They move to a
section of their own, beside the stream semantics they reuse (WRITING §2,
"every paragraph serves the heading").

| section | subsections | what changes |
|---|---|---|
| Introduction | The idea · Four words · Install · A first fit · What you can rely on | the opening line and the closing paragraph name the window operators, at the introduction's altitude; the test counts recounted |
| How a bank sees a stream | What a spec names · Time and decay · A hard window · A local fit along any feature · Convergence without a decay · Groups · Weights · Warm-up · Labels that arrive late · Nulls, and three ways to hold a row back · Row order and the two guarantees | the opening no longer promises a last subsection on series; a window expression named as a kind of target, and linked; `window=` written `window_size=`; `embargo`'s six comment bullets become a table |
| Preparing a stream (new) | Series that tick at their own times (← How a bank sees a stream) · Windowed means, looking back or ahead (←) · Windows as columns (←) · Windows as a model's inputs and target (←) · Saving and resuming a window run (new) | an opening on what the two tools share; the window rules as tables (which rows a window holds under each `closed`, what a formula may hold, `partial`'s values); the resume rules of review rounds R2 to R9, with a runnable chain; the two paragraphs of measurements go to Performance |
| Running a bank | As a query · In a loop · Outside a live Python process · Output as Arrow (← after the three ways: an output form of `fit_predict`, not a way to run) | the opening names the three ways and the output form; the loop's repeat of `repr`, `specs` and `groups` goes, since *What a bank holds* has them |
| Saving, loading and serving | Save and load (+ when a query writes its state, ← As a query, beside the bank's `save` and the command line's) · Serving without learning · Reading a state without this library | the loading table: a file below schema 25, which is every release so far, is refused; a window run's state named and linked; `predict`'s comment bullets become a table that says a step back is scored |
| Reading the fit | What a bank holds (← Saving's *What a state file holds*, renamed: a thing is not its file, WRITING §4) · Coefficients · Output field names · The running sums behind a fit · One row per finished group · Reading a correlation matrix | the read-back tables sit together; the opening covers every subsection |
| Diagnostics, selection and evaluation | Per-row diagnostics · Conformal intervals · Evaluating an output frame · Evaluating a stream too large to hold · Data whose truth is known | the facts the check below corrects |
| Models | the two tables · Linear models · Moments and correlation · Clustering and classification · Sequential tests and regimes | each model's section opens with what it is for; `marginal` runs base, lags, bins, window, then the two speed options; `micro`'s `eps` comes before `scale_floor`; `kmeans`'s last paragraph splits by topic |
| Performance | Throughput · Window operators (new ← Preparing a stream's measurements, PERFORMANCE §32 to §36) · Memory: which calls stream · Tuning memory with Polars' own settings · Chunk size · Parallelism · Against scikit-learn | |
| Scope and integrations | What this is not · Pathway · Databases: DuckDB and ADBC | *What this is not* says which windows the library computes and which stay upstream |
| Versions, testing and development | Versioning and the Polars pin · Testing · Development · License | `~=0.13.0`; the counts recounted; the window operators and formula targets in the references table |

*Constraints held.* Every heading another file links to keeps its text, so
its anchor survives a move: `docs/RUNNER.md`, `docs/REGIMES.md`,
`docs/OUTPUTS.md`, `docs/PERFORMANCE.md`, `docs/README.md`,
`scripts/outputs_doc.py` and `llms.txt` use 25. Every `####` under *Models*
is a model with its *API:* and *Rust:* lines, and every python block runs in
the namespace `tests/test_production_hardening.py` gives it.

**Task 154, the guides' maps.** Each keeps its sections unless a row says
otherwise, and every one takes the corrections the check below finds.

| document | its map | what changes |
|---|---|---|
| `docs/RUNNER.md` | Running it · The configuration file · Memory, threads and chunk size · From Rust, and the Polars it needs | *Specs in TOML* gains a target that is a window expression (task 104), the form the spec and the state carry |
| `docs/STATE-WORKFLOW.md` | The workflow in four steps (Each step · Fit, and keep the state · The one gap · Resuming on input that overlaps the state · What a saved bank carries, new · A window run's state, new · The rules, and the evidence for each) · The research behind it | the status table and the two dated notes at the top describe the research, and move to its opening (←), so the guide opens with the workflow; a saved bank's held rows under `embargo`, its formula targets' open windows and its closed groups; a window run's resume rules (review rounds R3 to R9) with the tests that pin each |
| `docs/OUTPUTS.md` | as generated | the generator, `scripts/outputs_doc.py`, where a field or its meaning is missing |
| `docs/REGIMES.md` | §0 to §10, numbered | long sentences split; parameter names checked against the builders |
| `docs/PERFORMANCE.md` | Reading this document · then the numbered sections grouped by topic, each keeping its number: the first pass and the plumbing (§1–§8, §12) · the surfaces (§9–§11, §21) · per model (§13–§20, §22–§25, §27) · releases compared and slowdowns bisected (§26, §28–§31) · the window operators (§32–§36) | the contents table lists §32–§36, which it lacked, and is grouped the same way; *The headline* says where the time and memory go now and which setting moves each, from the latest sections, and the 2026-09-06 summary of the first pass moves to the head of its group (←); the dated sections keep their wording, and the word list gains the names those sections use from before task 144 |
| `docs/EXTENDING.md` | the sixteen steps | step 9 gains the docstring checks task 149 met: a clock-unit parameter's entry says "clock units" (`tests/test_temporal_clock.py`), and a weight on the sum scale is named (`tests/test_weight_scale.py`) |
| `docs/TESTING.md` | What the suite proves · What it has found · Where it is thin · How the suite looks for defects · The entries, by ID | it opens with today's count, and the dated counts become a table; the window operators' oracles (a loop from the definition, Polars' own functions, the time-reversal identity) and the resume chains join what the suite proves |
| `docs/RELEASE-READINESS.md` | Cutting a release · Which Polars versions are promised · Keeping the API stable · CI cost while the repo was private · Going public, as recorded | facts against the workflows; a rule stated before the story that led to it |
| `docs/WRITING.md` | §0 to §8 | §6 gains the docstring pass's differences: Sphinx `-W` is the render, and tests pin some docstring text and its line wrap |
| `docs/README.md` | Guides · Records · At the top of the repository · Section numbers | each guide's row says what it now covers; PLAN's row names §14 |
| `CHANGELOG.md`, unreleased | Added · Changed · Performance · Fixed, once each | two *Added* and two *Fixed* subsections become one each; the nine review rounds' bullets, which narrate fixes to features no release has shipped, become the behaviour those features have, under *Added*, and the fixes to released surfaces stay under *Fixed*; every saved bank must be refit (schema 25) leads *Changed*; task 151's docs, missing, join task 148's; `po.window`, added after 0.13.0 and replaced before any release, is folded into the operators' entry (its history is tasks 78 and 143) |
| `CONTRIBUTING.md` | as it is | the gate's step list gains `uv lock --check`; a commit names its task; documentation follows `docs/WRITING.md` |
| `SECURITY.md` | as it is | the parsed files gain the window run's and the grid's state files, and the formula a spec or a state carries |
| `.github/PULL_REQUEST_TEMPLATE.md` | as it is | the checklist names `docs/WRITING.md` for a doc change |

**Task 154, done 2026-10-03.** Every document in the scope was rewritten
to its map by WRITING §6's steps. The README, the guides to using the
library, EXTENDING, WRITING, the index, the three files at the top and the
changelog's unreleased entry were written here. TESTING and
RELEASE-READINESS were written by two writer subagents under the same
rules and maps, each confined to its file. Each was reviewed here: the
account of numbers, names and links, the measure, the structure check, and
its facts read against the tests and the workflows. Neither changed a
cited heading or an entry ID. What moved inside them:

| document | what moved |
|---|---|
| `docs/TESTING.md` | today's count first, the dated counts a table; the two oracle subsections into *What the suite proves*, with three new ones there: the window operators and formula targets, resuming a run, and where the suite runs; *Open entries* merged into *What is left*, which now leads *Where it is thin*; the equivalent mutants into *Mutation survivors*; the first runs' record to the end of *The entries, by ID* |
| `docs/RELEASE-READINESS.md` | the rule first, a release dispatched and never tagged by hand, then *The steps of a release*, with three new steps: the benchmark against the last release's wheel, the released states' list, and the README's minor pin; the workflow's history to *How the release workflow came to this*; the present Polars guidance before the dated records, with NumPy beside it; the API policy first; the expression plugin's record under a heading of its own |

*Task 154, the check against the code.* Seven read-only reviewers checked
the README by part, and the guides in two groups, against the code at
`df555b2`; every finding acted on was re-derived first, by a run or by the
code. What the README said that the code does not:

| where | it said | the code |
|---|---|---|
| loading a state | a file from before 0.12.0 is refused | a file below schema 25, which is every release's, 0.13.0 included |
| *How a bank sees a stream* | every model takes its parameters; its last subsection is the grid | most models take them, each exception named; the grid and the windows were never shared parameters, and moved |
| *A hard window*, `ewridge` | `window=` | `window_size=` |
| `session_gap` | at most `gap_cap`, required with `session` | capped as any step is; refused beside `group_close="session"` |
| a clock quantity in an output | in seconds | `emit_clocks` keeps the clock column's own type |
| a gap and a window | a gap longer than `window_size` empties it | a step after `gap_cap` does, so only when the cap is longer than the window; nulls from the row after |
| `ewridge` beside a window | `ridge_scale` refused | `ridge_scale="sum"` refused |
| a local fit | `coef` is the line through each row's neighbourhood | with `coef_every=1`, now in the example |
| decay off | forwards, backwards or shuffled give the same coefficients | with no clock column, to rounding |
| `group_close` | one row per group | one per half-life, and per Gram |
| a weight | a row of weight `w` counts as `w` observations | a weight is relative in the models that keep means; seven read it otherwise |
| warm-up | the other models keep `min_weight` | and `min_settled_frac` |
| `weight_sum` | after forgetting, before the row's update | before the row's decay and its update |
| `embargo` | rows still waiting at the end are never learned | unless the state is saved and resumed |
| the doubled stream | agrees to the bit | where each row's `t + delay` is another row's clock |
| chunk invariance | `coef` the one exception | and `support_coef` |
| `group_close="monotone"` | refuses a key below one closed | any key below the one before it |
| `fit`'s order warning | exempt with no decay, window, session or drift reset | and with no weight, embargo, `coef_every` or diagnostic, among others |
| `refresh_time` | `n_obs` is the staleness; the largest holds the grid up | a large `n_obs` is ticks dropped; the slowest series, near 1, holds it up (its docstring too) |
| `po.increment` | one row back | back to the last row with a value; null after a restart |
| the operators | positional arguments | keywords |
| `with_windows` | adds columns, changes nothing else | `partial="drop"` and a saved state change which rows come out |
| a row with no value | holds the last one | a mean skips it, a sum adds nothing |
| the windows' clock | a spec's | a spec's, with clock order across groups |
| the column form | the same predictions, row for row | except where the embargo equals the window under `closed="right"`, the README's own example |
| `predict` | the stream's session and clock rules still hold | a step back is scored; a reset session scores null; a held row is not released |
| `last_row()` | stacks with a plain concat | needs `"diagonal_relaxed"` |
| a `Categorical` key | sorts by first seen | compared as text |
| the diagnostics | none sees the row it describes | `resid`, `zscore` and `drift` measure the row |
| `from_sums` | the numbers `metrics()` gives, with the RMSE | `metrics()` has no RMSE |
| `fit_predict_batches` | a DataFrame is one chunk | cut into `chunk_rows` |
| `po.sim.regimes` | `design="smooth"` interpolates; an optional volume | with `smooth_rows` above 0; an `activity` column |
| the models | twenty families | twenty-one models in four families |
| *learns by* | an accumulator converges in any order | except a lag, `rcov`'s block, `deco`'s estimate and `lasso`'s selection |
| `kalman` | a slope halves while nothing is observed; far ahead, the intercept; river to 3.6e-15 | pulled on every row; shrunk by at most `2^(−gap_cap/r_i)`; 1e-13 |
| `huber`, `quantile` | rows weigh three per coefficient; numpy to 1e-13; a ridge grid | rows counted; within 1e-14; one value |
| `sgd` | `quantile=` in the target's units | the level |
| `ftrl` | its penalties decay with the sums | the rows that teach a target restore them; the ridge figure holds at the defaults |
| `ew_cov` | `mahal²` is χ² | about χ² |
| `marginal` | `n_kish` tends to `(1 + λ)/(1 − λ)` | `(1 + λ^d)/(1 − λ^d)` |
| `deco` | both dynamics on the clock; `loglik` null where a value has no `u` | `"linear"` steps per row; null where any column has no standardised value |
| `micro` | field `micro`; `beta_mu` a weight; `a = n/(n + 1)` | `micro_id`; rows of the mean weight; `n/(n + w̄)` (its docstring too) |
| `ew_class` | `coef_up_x0` after `unnest`; null a late label | after `df.online.unnest`; `embargo` |
| `seqtest` | a session change restarts the test | under `session_gap="reset"` or `group_close="session"` (its docstring too) |
| `corrchange` | `since_change` the argmax | the rows from the change it dates through the flag |
| `hmm` | 57% of rows in the true state | 0.51 after seeding, which is chance |
| Performance | the thread count read at run time; `chunk_rows` on four calls; memory three things | at import; on six; the rows a delay or a window holds as well |
| Scope | a fixed memory per stream; no event-time windowing | the held rows; windowed means are computed |
| Versions | three interfaces; `~=0.12.0`; released files loaded | four, the formulas' serialized form the fourth; `~=0.13.0`; refused |
| Testing | 1,149 and 3,597 tests on 2026-09-29 | 1,215 and 3,840 on 2026-10-03 |

Found on the way: ten that change code rather than a document. They were
listed for the user, who asked on 2026-10-03 to fix the ones with obvious
solutions. All ten had one. Where a fix changes behaviour or a message, its
test failed on the old build first:

| where | it did | now | pinned by |
|---|---|---|---|
| `online --dry-run` | passed a window target whose embargo the run then refused at its first chunk | `RunConfig::validate` asks the bank (`Bank::fit_predict_refusal`) for a run that keeps its predictions | `formula_targets.rs`, `validate_refuses_a_short_embargo_where_the_run_would`; `test_formula_targets.py`, the dry-run test |
| `online --no-output --closed-groups` | the dry run said the run's product was nothing | names the closed groups, and the state where there is one | `test_no_output.py`, `test_the_cli_dry_run_names_the_closed_groups_as_the_product` |
| the step-back refusal | advised `ModelBank.skip_learned`, which the command line cannot call | names the command line's way too, a filter on its input; the README quotes it, matched word for word | `test_skip_learned.py`, `test_the_command_line_is_told_its_own_way_out` |
| `rcov`'s refusals | named a `window` parameter it has not got, three times | name `preavg_rows` | `rcov.rs`'s and `test_rcov.py`'s refused-by-name tests |
| `group_close="monotone"`'s refusal | said a `Categorical` sorts by first seen | names the sort alone: Polars sorts a Categorical as text on 1.34.0 and on 1.44.2, measured | `test_closed_groups.py`, `test_a_categorical_key_out_of_order_is_refused_and_a_sort_fixes_it` |
| `ModelBank.to_json`'s docstring | said the export refuses NaN and ±inf | says it writes them as `"nan"`, `"inf"` and `"-inf"` | `test_state_json.py`, already |
| a comment in `spec.rs` | called the interval fields `pred_lo_` and `pred_hi_` | `lo_` and `hi_` | |
| `tests/test_released_state.py` | held 0.10.0's and 0.11.1's files only | every release to 0.13.0: 0.12.0 (schema 19) and 0.13.0 (schema 20) each saved all 30 specs, and this build refuses each file | itself |
| `WORKLOAD`, `_spec.UNSUPERVISED` | held to the registry by nothing | the workload builds every builder; `UNSUPERVISED` is exactly the models whose empty `targets` the bank fills | `test_model_registry.py`, two tests |
| CLAUDE.md rule 13 | listed three interfaces with Polars | four, the formulas' serialized form the fourth | |

Two more of the same kind, found by the release guide's writer and fixed
with them: `pyproject.toml` said `release.yml` runs the suite "at the tag",
where it runs before every upload, and a CI-cost test's name said a
schedule reaches macOS, which only a dispatch or a public repository does.

**Task 154, step 6, measured again after the pass** (the same script and
columns as step 1):

| document | prose words | sentences | mean | 35+ | 45+ | cost words | *X, not Y* | tables | code blocks | bullets | `##` sections |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `README.md` | 18,614 | 1,024 | 18.2 | 5 | 0 | 4 | 13 | 60 | 92 (65 python) | 2 | 11 |
| `docs/RUNNER.md` | 1,492 | 86 | 17.3 | 1 | 0 | 0 | 0 | 3 | 9 | 0 | 4 |
| `docs/STATE-WORKFLOW.md` | 3,631 | 216 | 16.8 | 3 | 1 | 0 | 0 | 14 | 4 (3 python) | 5 | 2 |
| `docs/OUTPUTS.md` | 293 | 19 | 15.4 | 0 | 0 | 0 | 0 | 26 | 0 | 0 | 22 |
| `docs/REGIMES.md` | 4,277 | 236 | 18.1 | 4 | 0 | 2 | 1 | 21 | 1 | 0 | 11 |
| `docs/PERFORMANCE.md` | 24,076 | 1,367 | 17.6 | 28 | 0 | 96 | 42 | 134 | 6 (2 python) | 39 | 6 |
| `docs/EXTENDING.md` | 2,928 | 188 | 15.6 | 3 | 0 | 0 | 6 | 14 | 5 | 0 | 6 |
| `docs/TESTING.md` | 8,632 | 594 | 14.5 | 0 | 0 | 1 | 0 | 44 | 0 | 0 | 5 |
| `docs/RELEASE-READINESS.md` | 8,880 | 577 | 15.4 | 0 | 0 | 7 | 15 | 37 | 7 | 12 | 5 |
| `docs/WRITING.md` | 2,835 | 175 | 16.2 | 5 | 0 | 11 | 10 | 13 | 0 | 0 | 9 |
| `docs/README.md` | 134 | 10 | 13.4 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 4 |
| `CONTRIBUTING.md` | 871 | 70 | 12.4 | 0 | 0 | 0 | 3 | 0 | 1 | 26 | 8 |
| `SECURITY.md` | 241 | 18 | 13.4 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 3 |
| `.github/PULL_REQUEST_TEMPLATE.md` | 104 | 6 | 17.3 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 3 |
| `CHANGELOG.md`, unreleased | 2,714 | 138 | 19.7 | 10 | 0 | 1 | 4 | 2 | 0 | 41 | 1 |

Step 1 left out the unreleased entry. At `df555b2` it held 3,810 words in
118 sentences, a mean of 32.3, with 41 sentences of 35 words or more and 29
of 45 or more. No document now has a sentence of 45 words or more but
STATE-WORKFLOW, whose one is in its dated research. The README's five of
35 or more are the splitter joining a sentence to the next, which opens
with a lowercase name (`polars-online`, `sklearn`, a path).

**After the pass, two corrections and an addition from the user
(2026-10-03).** The
README's opening paragraph read clunky. The pass had added the windowed
means to its list of models, on a line it left unwrapped, so "for data too
large to hold in memory" hung off "looking back or ahead". The first
sentence is task 138's again, and the windowed means have a sentence of
their own. And *Four words* is now *Terminology*, with its contents link,
its lead sentence and its table's column renamed to match. Then the user
asked for an example of `po.stream.refresh_time` feeding a model, in the
README and in the conversation. *Series that tick at their own times* now
ends with one: each series' change between grid points, fitted by `ew_cov`
and `ewridge` on the grid's own clock. It runs with the README's other
blocks, and its query form gives the same numbers, checked when written.
The user then found the example hard to follow with `ticks` undescribed,
and asked whether *refresh time* was a mistaken name, since the function
seemed to do nothing with time. The section now says what a refresh time
is, the first instant by which every series has ticked since the last, and
opens with eight ticks and the two-point grid they make. That table is held
by `test_the_readme_example_is_the_grid_it_shows`. The name is the
literature's (Barndorff-Nielsen, Hansen, Lunde and Shephard), and other
libraries use it: R's `highfrequency` has `refreshTime()`, and Python's
`hfhd` has `refresh_time`. The user kept it on 2026-10-04, after the
alternative raised, `refresh_grid`, turned out to be used nowhere. Reading the grid against its docs found `n_obs`
described wrongly twice. The docstring said the grid keeps a series' first
tick of an interval, and `refresh.rs`'s module comment said a large count
marks the series holding the grid up. Both now say the grid keeps the last
tick, and that the series holding it up has a count near 1.

- [x] 155. **The weekly mutation pass sized to finish -- requested
      2026-10-03.** Size S. The user asked "How can we ensure that the run
      will finish?", then "Make the changes and push and run it now". No
      weekly pass had finished. Its one run, on 2026-09-27, failed at its
      baseline on a `corrchange` test, fixed that day in `0166135`. And its
      sixteen shards of about 700 mutants could not fit two hours.

    | measured on 2026-10-03 | value |
    |---|---|
    | `online-core`'s mutants | 11,138 |
    | the pace on GitHub's runner, from the 45-mutant run of 2026-09-27 | 37 minutes, two at a time: about 92 s a mutant |
    | a test run of `online-core`, here | 67 s: the test binaries 22 s, the three doctests 39 s |
    | a rebuild after one file changes, here | 36 s |
    | a mutant's time limit, five times the baseline | 377 s on the runner |

    | change | why |
    |---|---|
    | every run skips the doctests (`--cargo-test-arg=--tests`) | they were most of each test run, and of the baseline every limit is a multiple of; CI and the gate still run them |
    | a mutant stops at three times the baseline (`--timeout-multiplier 3`) | a mutant that loops ran for over six minutes |
    | the weekly pass is 48 shards, dealt round-robin, sixteen at a time | about 232 mutants a shard, from every file; four of the account's twenty concurrent jobs stay free |
    | a weekly shard's job may run four hours, and cargo-mutants stops at 215 minutes; the job on a change's lines stops it at 100 of its 120 | a job killed at its limit uploads and reports nothing |
    | each run lists the mutants it was given, and `mutants_report.py` fails on a run that tested fewer, or a shard that sent nothing | a pass cut short read as a smaller clean one |

      Checked in a clean clone on `comp.rs`: the baseline's test run fell to
      22 s, and the limit to 68 s. Stopped by SIGINT after 10 of its 23
      mutants, cargo-mutants exited 1 and its `outcomes.json` held those 10,
      valid. The cost policy's two-hour cap gains one exception, the weekly
      shard at four hours, held to Linux and to a public repository or a
      dispatch. Pinned by three tests in `tests/test_mutants_report.py` and
      three in `tests/test_ci_cost_policy.py`, each failing first.
      `scripts/mutants.sh` takes the same two flags. Ticked when a pass
      finishes inside its limits, with its pace recorded here.

      **The first pass, at 48 shards, did not fit** (run 37165801954,
      dispatched 2026-10-04). GitHub's runners came in two speeds, and ten
      of the first sixteen shards drew slow ones:

    | runner | baseline build, test | a mutant, one slot | a shard of 232 |
    |---|---|---|---|
    | fast (shards 0, 1, 2, 7) | 41-47 s, 43-52 s | 135-148 s | finished in 2 h 13 min to 2 h 26 min |
    | middling (9, 10) | 52-58 s, 68-69 s | 189-190 s | finished in about 3 h 7 min |
    | slow (the other ten) | 63-69 s, 87-94 s | 229-246 s | stopped at 215 minutes, 205-221 tested |

      Timeouts took 19-28% of a shard's time. The first wave tested 3,522
      of its 3,714 mutants. One survived, `SplitMix64::choice`'s `>` against
      `>=` in `cluster/summary.rs`, and a second missed mutant is an
      equivalent the list names. Each stopped shard uploaded what it had
      finished and its list, and the report, run here on the sixteen
      uploads, named the ten short shards and exited 1. So the pass is
      96 shards of about 116 mutants: about two hours on the slowest runner
      measured, 55% of the stop, and about ten hours for the pass, sixteen
      at a time. `lld`, which the job on a change's lines links with, is
      left out of the weekly job until a pass measures what it saves.

      **The 96-shard pass finished, and its time limit hid survivors** (run
      37177791033, 2026-10-04). Every shard tested all its mutants, in 64 to
      127 minutes, and the pass took 11.5 hours: 60 of its 96 shards drew
      slow runners. The report counted 9,249 caught, 11 survived, 3
      equivalent, 1,588 timed out and 287 unviable. Fourteen per cent timed
      out, where TESTING's rule calls a few per cent untrustworthy. Rerun
      here with a limit of ten times the baseline, 25 of those timeouts
      sampled at random came out:

    | outcome | of 25 | test time against the solo baseline |
    |---|---:|---|
    | survived | 19 | 1.3 to 2.3 times, at four jobs on fourteen cores |
    | caught | 4 | 1.6 to 6.3 times |
    | still hung | 2 | past ten times |

      A survivor runs the whole suite while the other jobs share the cores,
      and the baseline runs alone. At fourteen jobs on fourteen cores, close
      to the runner's four on four, the nineteen took 1.7 to 6.3 times the
      baseline, median 5.2. So the limit of three times counted most
      survivors as timeouts: about 1,200 of the 1,588, by the sample. It
      did so in the job on a change's lines too, which fails on a survivor
      and so could pass one. The multiplier is 10 in both jobs and in
      `scripts/mutants.sh`. By the sample, it adds about 16 minutes to a
      slow shard, so the slowest stays near two-thirds of the stop. Ticked
      when a pass at this limit finishes with timeouts at a few per cent.

      **The pass at ten times the baseline finished inside its limits**
      (run 37217857544, dispatched 2026-10-04, done 2026-10-05). Every one
      of the 96 shards tested all its mutants, in 65 to 138 minutes, mean
      109, so the slowest used 64% of the 215-minute stop. The report
      counted 9,486 caught, 1,272 survived, 27 equivalent, 66 timed out
      (0.6%) and 287 unviable. The sample's estimate held: 1,261 of the
      previous pass's 1,588 timeouts were survivors. The survivors are task
      158. The scheduled pass of 2026-10-04 (run 37194203887, at the old
      limit) failed in shard 13 at its baseline, before any mutant: a
      generated stream drove `kalman`'s prediction to NaN in
      `model_contract.rs`'s `generated::kalman`, so that shard reported
      nothing. That bug is task 158's first item.

- [x] 156. **The README rewritten by the ideas of two reviews -- requested
      2026-10-04.** Size L. The user's words: "Keep a doc of these ideas,
      we will be iterating on the readme and we want to keep track of which
      promos get us to a better doc. Then rewrite the readme according to
      those ideas and present me the html rendering". The record is
      `docs/README-ITERATIONS.md`: this pass is its I4, and the ideas it
      applies are those marked *I4* there. They come from two reviews of
      2026-10-04, the second under WRITING §2's new rules: an opener of one
      to three sentences for every section, the kinds of functionality
      named first and each developed below, and nothing important tacked on
      at the end. Counted before: 18,614 prose words, 1,024 sentences at
      18.2 words, 5 of 35+, 0 of 45+, 60 tables, 16 headings with no prose
      opener. The map, before any prose changes (`←` marks what moves):

    | section | subsections, in order | moves |
    |---|---|---|
    | Introduction | The idea · Terminology · What you can rely on (←) · Install · A first fit | an opener; *The idea*'s last paragraph as bold-led paragraphs; *A first fit* opens with its last paragraph's point (←) |
    | How a bank sees a stream | What a spec names · Time and decay, as `#### Clock types and units` and `#### Sessions, gaps and steps back` · Convergence without a decay · A hard window · A local fit along any feature · Row order and the two guarantees (←) · Groups · Weights · Warm-up · Labels that arrive late · Nulls, and three ways to hold a row back | a note naming the examples' frames; the null rule into *What a spec names* (←); `weight_sum` into *Warm-up* (←); the huge-half-life warning into *Convergence* (←) |
    | Preparing a stream | Windowed means, looking back or ahead (←), with its three subsections · Series that tick at their own times | the rules every window form obeys into *Windowed means*' body (← *Windows as columns*) |
    | Running a bank | As a query · In a loop · Outside a live Python process · Output as Arrow | `lf.online.predict` into *Serving without learning* (←) |
    | Saving, loading and serving | Save and load · Serving without learning · Reading a state without this library | the release refusal into the opener (← the loading table's last row) |
    | Reading the fit | What a bank holds · Output field names (←) · Coefficients · The running sums behind a fit · One row per finished group · Reading a correlation matrix | |
    | Diagnostics, selection and evaluation | Per-row diagnostics · Conformal intervals · Choosing among a grid's settings (new) · Evaluating an output frame · Evaluating a stream too large to hold · Data whose truth is known | the selection switches out of a comment (←); the conformal rule out of a comment (←) |
    | Models | the model table first, then the shared rules, the legend and the notation · Linear models · Moments and correlation · Clustering and classification · Sequential tests and regimes as `seqtest`, `corrchange`, `bocpd` (←), `hmm` | family openers that map their entries; a shared-parameter table and a half-life table under *Linear models* |
    | Performance | Throughput · Parallelism (←) · Chunk size · Memory · Tuning memory with Polars' own settings · Window operators (←) · Against scikit-learn | the read-ahead into *Tuning memory* (← *Parallelism*) |
    | Scope and integrations | What this is not · Databases: DuckDB and ADBC (←) · Pathway | |
    | Versions, testing and development | Versioning and the Polars pin, as This package's own versioning (←), What is pinned, How the pin moves, Which interfaces carry a promise · Testing · Development · License | the dated runs cut, since RELEASE-READINESS holds them |

      *Constraints.* Every heading another file links to keeps its text,
      wherever it moves. Every `####` under *Models* is a model, with its
      *API:* line. Every python block runs alone in the namespace
      `tests/test_production_hardening.py` gives it. The errors E1 to E8 of
      the record are fixed. Six writers, one per part of the README,
      write the new text; the account (WRITING §6, step 4) is a diff of
      every number, backticked name and link against `21bedd7`.

      *Done 2026-10-04.* The six parts were assembled under a rebuilt
      contents table, read through whole, and published as a rendering
      for the user's verdict. Counted after: 22,627 prose words, 1,298
      sentences at 17.4 words, 7 of 35+ and 2 of 45+ (each a merge of two
      sentences at a lowercase name, as all five before were), 112 tables,
      no heading without a prose opener, and 13 continued comment lines
      where there were 111. The account against `21bedd7` loses nothing:
      the run dates are in RELEASE-READINESS, five names moved from
      backticks into code, and `like=spec` became `like=edge`. The full
      read fixed the rest: `lf` keeps one meaning (the parquet query is
      `files`), the window target's spec is `edge` and one run's output
      is `one_run`, and the list of models that refuse a relative target
      sits after a colon. `tests/test_weight_scale.py` calls `kalman`'s
      `obs_var / w` a variance, as E4 has it. The verdict goes into
      `docs/README-ITERATIONS.md`. It came the same day and is recorded
      there, with the ideas it raised (S11 to S14, C4 to C7, W1 to W7
      and E10) for the next pass, I5, eleven rules in WRITING §2, §3 and
      §5, and the user's own rewrite of one paragraph beside its original
      in §8.
- [x] 157. **The README rewritten as a test of every rule -- requested
      2026-10-04.** Size L. The user's words: "Write all the rules and the
      rewrite the readme as a test and show me the html". The rules are
      WRITING's nine principles (its preamble) and the rules I4's verdict
      added to §2, §3 and §5; the pass is `docs/README-ITERATIONS.md`'s I5,
      applying every idea marked *I5* there. The README stays uncommitted
      until the user's verdict. The map, where it moves from I4's:

    | section | subsections, in order | moves |
    |---|---|---|
    | Introduction | A first fit (←, first) · The idea · Terminology · What you can rely on · Install | a one-line summary as the opener; the first fit installs, shows its input and output tables, and learns a forward `rewm_mean` target; *The idea* opens with *a few points to remember*, and its two out-of-sample paragraphs become one |
    | How a bank sees a stream | as I4 | *What a spec names* opens on what a spec's name is used for |
    | Performance | as I4 | a short introduction in place of the table of its subsections |
    | every section | as I4 | an opener that lists what follows becomes a summary; a semicolon joining two ideas, and a back-reference farther than the paragraph before, are resolved |

      *E11, fixed 2026-10-04 on the user's word ("Fix the bug and commit
      that").* Found while checking the verdict's note on relative targets:
      a target formula known at its own row was refused with "add it as a
      column with po.stream.with_windows", which refuses a formula with no
      operator in turn. The refusal now names the call that works: Polars'
      `with_columns` for a formula with no operator at all, `with_windows`
      for one whose operators only look back, in the spec's Python check
      (`_spec.py`, `formula_target`) and in the Rust parser a TOML or JSON
      target goes through (`targets.rs`). Tests:
      `tests/test_formula_targets.py`, which also follows each piece of
      advice, and `targets::tests::a_formula_target_that_is_not_one_is_refused`.

      *I6 and I7.* I6 applied I5's verdict to the passages it touched
      (relative targets built both ways, *exception* and *raise* in their
      Python sense). I6's verdict asked for procedural sentences (WRITING
      P10, §5), and the user then asked: "Start the next readme and since
      the latest instructions have a deep impact, ensure that you are
      willing to completely rewrite phrasing based on the latest
      instructions". I7 is that full pass, seven writers under
      `docs/README-ITERATIONS.md`'s ideas, the README still uncommitted
      until a verdict.

      *E12, fixed 2026-10-04 on the user's word ("Fix e12 while the doc
      writers are working").* I7's writer for *Reading the fit* found it
      while turning the merge advice into steps. `po.gram.merge`'s docstring
      said to decay only the earlier half's `weight_sum` (and a sum of
      squared weights the Gram does not hold) before merging two halves of
      a decayed stream; its `target_weights` need the same factor, or the
      target moments pool with the early half over-weighted (measured:
      `target_weights` 78% off, `means_by_target` 51%, `cross_centred` 8%).
      The docstring now gives the procedure as steps, with a runnable
      example the docstring-example test executes, and
      `tests/test_gram_module.py` holds every field of the merge to the
      whole stream, target side included, and the old recipe to its error.

      *I7's verdict, and the commit (2026-10-05, on the user's word: "Commit
      and push").* The user's notes on I7, each kept verbatim in
      `docs/PHRASING.md` and recorded in `docs/README-ITERATIONS.md`:
      *The examples from here on read two frames* was out of context, so
      *Example data*, after *Install*, now builds every frame and file the
      examples read (C11), and every example on it says so in the line
      just above, with a link (C15). Asked why the reviews missed it, the
      record gives the cause: every check ran from inside the project, the
      README test with its namespace and the reviewers with what they
      knew. Two checks now start where a reader starts:
      `test_every_name_a_readme_example_reads_was_built_by_an_earlier_one`
      (V1) and a cold read (V2, WRITING §6 step 9), which ranked the
      paragraph first unprompted. The `po.target` paragraph became an
      example (C12), and the user's "How can we find more paragraphs that
      should be code?" became a detector and a judging pass (V3), and 20
      more examples (C13, C14), each run with its comments checked. The
      tools of a pass are now in the repository, `scripts/doc_review.py`
      (`measure`, `counts`, `account`, `run`, `render`), held by
      `tests/test_doc_review.py`. Left for the user: E13 (a `with_windows`
      resume after an unsliced save outputs a repeated row twice), the
      docstrings of E14, the documents of E15, E9's wheel sizes, W14 (a
      term used before its section), whether blocks on frames built from
      the example data (`flows`, `by_block`) carry the line too, and a test
      that runs the README's TOML through the command line.

- [x] 158. **The mutation survivors worked through, and the bug behind the
      scheduled pass's failure fixed -- requested 2026-10-05.** Size L. The
      user's words: "After the push, tick task 155 and commit, then work
      through all the survivors and fix the cause of the last task
      failure." First the cause: proptest's generated stream for `kalman`,
      with features near `±1e100`, a feature of `4.6e49`, weights and a
      target at `1e100`, made row 7 predict `[NaN, inf]`: a feature at the
      input bound, standardized against a scale the earlier rows set,
      times its coefficient overflowed `z . beta`. `step` already skipped
      the update such a row would poison (IMPROVEMENTS C2), but emitted the
      prediction. *Fixed 2026-10-05:* `step` and `predict` both withhold a
      prediction that is not a number, as NaN, which the bank writes as a
      null. The shrunk case is the named test
      `generated::kalman_at_the_input_bound_predicts_a_number_or_nothing`,
      since proptest cannot find that file's source to save a regression
      file; it failed on the old code with the scheduled run's own message.
      Sixteen runs of every model's generated stream, about 2,000 cases
      each, found nothing more. Then the 1,272 survivors of run
      37217857544, file by file:
      each is killed by a Rust test (the only tests cargo-mutants sees), or
      named an equivalent in `scripts/mutants_equivalent.toml` with the
      reason, as the 27 there are.

      *The `rcov` bias, fixed 2026-10-05 on the user's word ("Fix the rcov
      bug").* Batch A's worker found it while pinning `rcov`'s survivors.
      Under `kind="preavg"` and `psd=False` the bias term was
      `ψ₁/(θ²ψ₂)·Σxx'/(2n)` with the configured `theta`, where CKP's Eq. 7
      defines θ by the window, `k_n/√n`; the two part when `preavg_rows` is
      given, or a block's length differs from `block_rows`, and too little
      noise was subtracted. On pure noise, whose truth is 0, sixteen streams
      of 20,000 returns averaged 195.5 (standard error 1.4) under
      `preavg_rows=20`, and about 11 in a block four times `block_rows`. The
      term now reads θ from the window run, `ψ₁/(2ψ₂k_n²)`, and those means
      are within 1.5 standard errors of 0. Held by
      `rcov::tests::the_bias_term_reads_theta_from_the_window_actually_run`,
      which fails on the old line, and by the Rust and Python definition
      tests, now written with Eq. 7's θ. REGIMES §8, re-run, moves in one
      cell at the third decimal (refresh time, `psd=False`: bias −0.010 to
      −0.009, the same error). The golden stream `rcov_preavg`, which runs
      `preavg_rows = 6` at the default `theta`, had pinned the old bias and
      is re-pinned: its first variance moves from 9.410 to 7.978.

      *The survivors, worked through 2026-10-05.* Six workers, one batch of
      files each and each in its own worktree, wrote Rust tests and checked
      every kill with cargo-mutants run on exactly their survivors, at the
      weekly pass's flags (`RUST_TEST_THREADS=2` once three runs at once
      took the machine to load 65). Of the 1,272:

    | batch | files | survivors | killed | equivalent | not killed |
    |---|---|---:|---:|---:|---:|
    | A | `rcov`, `marglag`, `margbins`, `constraint`, `pa`, `holt`, `seqtest`, `drift`, `lib` | 209 | 193 | 15 | 1 |
    | B | `marginal`, `gaps`, `model`, `clock`, `ewlagcov`, `ftrl` | 202 | 186 | 16 | 0 |
    | C | `bocpd`, `hmm`, `ewclass`, `conformal`, `cluster/summary` | 197 | 170 | 25 | 2 |
    | D | `kmeans`, `micro`, `deco`, `humanfloat`, `window` | 233 | 213 | 20 | 0 |
    | E | `corrchange`, `robust`, `boundary`, `rls` | 196 | 123 | 46 | 27 |
    | F | `ewcov`, `sgd`, `ewridge`, `solve`, `kalman`, `stats`, `lasso` | 235 | 203 | 29 | 3 |
    | | | **1,272** | **1,088** | **151** | **33** |

      The tests add about 8,700 lines, in each module's own `mod tests`, and
      change no production line. With its share `marginal.rs` passed the
      repository's 250 KB cap for a source file, so its tests moved to
      `marginal/tests.rs` (cargo-mutants finds no mutant there, and the
      crate's count is unchanged); `ewridge.rs`, at 243 KB, and
      online-polars' `bank.rs`, at 242 KB, are the next near it. The 151 equivalents are 141 new entries in
      `scripts/mutants_equivalent.toml`, each with the reason no input can
      tell the two apart; three stale entries left it (a function renamed,
      a line moved, and one whose reason was wrong and whose mutant a test
      now kills), and `tests/test_mutants_report.py` accepts a generic
      function's `fn name<` and cargo-mutants' `delete` mutations. Every one
      of the 161 entries matches a mutant of today's code. The 33 not killed
      differ only by rounding or below the computation's own error (most
      are `boundary`'s solver stopping a settled solve), are reachable only
      past the input bound, sit behind a `debug_assert!` no test build
      passes, or wait on a decision: whether `EwQuantile` drops an end
      bucket at exactly its prune share, as the code does, or only under
      it, as its doc says. Two tests were found pinning nothing: a sharded
      `marginal` stream whose pairs were all NaN from row 92, and an `rcov`
      PSD-clip test whose stream never clips. Findings beyond the `rcov`
      bias, raised with the user: `rls` loses precision where a rotation's
      squares are subnormal (relative error 1.4e-5 at a scale of 2^-530,
      inside the input bound); an `hmm` given a `transition` with
      `transition_prior = 0` filters with a uniform chain, and under
      `learn = False` never uses the given matrix; and a remote `robust`
      case at weights near 5e-324.

      *The obvious follow-ups, 2026-10-05, on the user's word ("Handle the
      obvious answers").* `rls` takes `hypot` wherever the sum of a
      rotation's squares is below `f64::MIN_POSITIVE`, not only where its
      root is 0 or not finite; held by
      `a_row_whose_squares_are_subnormal_is_rotated_in_by_hypot`, which failed
      on the old line (1.8677183 against 1.8677443). An `hmm` row with no
      counts and no prior mass is the prior's mean, `Π₀` or uniform, the
      row's value at every `τ > 0` before a count, where it was uniform
      whatever `Π₀` said; held by `a_row_with_no_mass_is_the_given_matrix`,
      which failed on the old code. `EwQuantile`'s docs said an end bucket
      "lighter than" or "under" the prune share is dropped, where the code
      drops one at the share too: the docs now say "at or below", which
      moves no number, and
      `an_end_bucket_at_exactly_the_prune_share_is_dropped` kills the
      mutant that boundary left (`stats.rs:223:39`, checked by cargo-mutants),
      so 32 of the 1,272 are left. The six workers' worktrees, merged and
      clean, were removed (13.5 GB); their branches stay. Left for the user:
      a list of the survivors that differ only by rounding or below a
      computation's own error, which the report would show apart and fail
      on none of, and the push.

      *The open items, 2026-10-05, on the user's word ("Your suggestion item
      1, 2, 3, 4, 5, 6, 7, e15, e14, e9").* The `rcov` test whose stream
      never clips is named for what it checks,
      `a_spiked_kernel_estimate_under_psd_is_positive_semi_definite`; the
      clip itself was already held by
      `the_psd_repair_clips_the_negative_eigenvalues_alone`. A new test runs
      the README's TOML through the built `online` and compares it with the
      Python `spec` it mirrors. It failed: the TOML left out `standardize =
      true` (README-ITERATIONS E16), now added. The README's state sentence
      names the releases it means, its `sin(x)` figures name their stream
      and test, which now holds them to two places, and its wheel sizes are
      0.13.0's (E9). E15's three fixes: PERFORMANCE's glossary, the test's
      quote, now asserted, and ARROW-SOURCES §2, whose ADBC crash proved
      wider than the trap it was found in. Nine of E14's ten docstrings are
      corrected against the code. One of them was a code defect behind the
      words: `sgd`, `pa`, `ftrl` and `rls` held each target on its own
      weight but against the smallest threshold of a `min_weight` list,
      and the bank checked the list against the shared weight, so a sparse
      target under `[5, 30]` predicted from row 30, not row 291. Each now
      reports its per-target weights (`target_n_eff_into`), as the six
      other models with one do; `test_a_sparse_target_warms_up_on_its_own_weight`
      takes unequal thresholds, the case that tells, and failed on the four
      before. A block that reads only a frame derived from the example data
      takes no example-data line, and W14 waits for the next README rewrite,
      both by the user's decision.

      *The tolerated list and E13, the same day.* Of the 32 survivors left,
      `rls`'s went with the line the `hypot` fix rewrote: cargo-mutants on
      the new lines caught 13 of 14, and the fourteenth matches an
      equivalent already listed. The other 31 are in
      `scripts/mutants_tolerated.toml`, 27 entries, each with a kind
      (`rounding`, `below-error`, `tie`, `test-build`, `input-bound`,
      `slow`) and the measured size of the difference.
      `scripts/mutants_report.py` lists them by kind and counts them apart,
      so neither job fails on one. An entry may name the `columns` it covers,
      and a `line_has` may take in the line before. Both tell a tolerated
      mutant from a twin a test catches, on its line (`boundary`'s `t_end`)
      or in its function (`corrchange`'s two `stat > crit` lines). Over the
      workers' final outcomes the report sets 30 apart, the 31st being batch
      A's. E13: any resumed `with_windows` run now refuses an input whose
      first row is at the last stamp the state read and starts no new
      session, as a sliced state did, so the row a resume repeated no longer
      comes out twice. The canary of 2026-10-05 (run 37327376335) failed on
      py-polars 2.0.0rc2 in seven `test_windows.py` tests: 2.0 lets a Python
      source's `ValueError` through where 1.x wraps it in `ComputeError`.
      Those tests now read either, as `test_frame.py` has since the 09-21
      canary, and the docs name the exception for both majors. While
      checking E15, closing an ADBC cursor just after a read of its stream
      stopped early segfaulted in polars alone (`collect_batches()` stopped
      after one batch); ARROW-SOURCES §2 records it, and nothing goes
      upstream while reports are parked.

- [x] 159. **The code review of everything since `v0.13.0`, every finding
      fixed -- requested 2026-10-05.** Size L. The user's words: "Code review
      changes since the last tag", then "Fix all". Six read-only reviewers
      over the 53 commits since the tag, one per area (the window core, the
      frame runner and formulas, the bank plumbing, the regression models
      and clock, the detectors and covariance, the Python package), every
      finding re-derived here from its reproduction and the code, and
      pinned by a test that failed before its fix; §15 has the table. 21
      findings: one high (`rcov`'s pre-averaged estimate forms one term
      fewer than CKP's definition per stretch, and scales as if it had them
      all); five silent (the sum-scale prior not decayed across a
      zero-weight head row; a negative sub-second clock delta rounded at
      the inclusive restart edge; an f64 window edge decided from policy
      times; `mass` cancelling at long half-lives; group and session keys
      by their string form across a dtype change); five loud (task 158's
      same-stamp refusal inside the bank's resolver, the pushed regression,
      and its single-shot verdict; group clocks surviving a stream restart;
      a formula target named after its own column; a non-finite literal's
      message); ten in docs and edges. Three of the numerical ones had
      oracles written from the code, against `docs/TESTING.md`'s rule, which
      is why nine review rounds and 1,272 mutation survivors left them.
      Fixed in three gated batches: A, the regression (B1, F1); B, the
      numbers (D1, R1, R2, W2, W3, W5, D2); C, the rest.

- [x] 160. **The whole-project review, every finding with a clear fix
      fixed -- requested 2026-10-05.** Size L. The user's words: "Do a
      complete code review of every detail of this project except md
      files", then "Do everything that has a clear solution". Sixteen
      read-only reviewers over 183k lines at `e6b70a2`; 139 finding IDs, each
      re-verified against its reproduction (PB6 rejected; PB7 = CA2 = YA3,
      PC4 = PB2 and TB3 = TA3 folded); §16 has the table. Four workers
      fixed them in worktrees, one area each (the core models, the polars
      layer and the CLI, the Python surface, the test suite), each test
      shown failing on the old code, and the coordinator took the CI,
      scripts and repository policy, then reviewed and merged each branch
      under its own gate. The new tests found seven defects of their own,
      fixed with them: the quantile nudge thrown past its target on
      correlated features (TC1b), `ModelBank.specs` dropping marginal's
      optional keys, 29 docstring examples reading names their page never
      builds, a soak test failing unseen since task 120, Rust `validate`
      taking what the sgd builder refuses (YA8b), a zoned `Datetime` group
      column refused (PA4b), and CE6's cancellation in three more places
      (CE6b). Left for the user, as decisions: the seven raised before the
      fixes (CC1, CE1, CC4, CB1, PC1, CA3, CI2), a reset discarding a
      forward window whose far edge is the last row before it (PC2), and
      TC1b's cost, half of `quantile`'s default-schedule throughput.
      Committed in five gated batches: `1120f9c` (CI and scripts),
      `f8b3aa3` (the Python surface), `ca1e6d0` (the tests), `a37ebf9` (the
      core and polars layer), and the last, CI4's Rust 1.95 with the clippy
      lints it turns on.

- [x] 161. **`ew_cov`'s PCA refreshes on the clock, as the regressions'
      solves do -- requested 2026-10-06.** Size S. The user's words: "Does
      pca ever take a clock value as well?", then "I want this feature",
      choosing to mirror the solve schedule. `pca_every` counted rows,
      where `solve_every` counts clock units and `max_rows_between_solves`
      rows, whichever comes first. Now `pca_every` is clock units, a number
      or a duration (`"5m"`), `0` every row, and `max_rows_between_pca`
      caps the rows between refreshes; whichever comes first refreshes, and
      with neither the components refresh every row, as before. A spec
      without a clock column is unchanged (its clock is the row index); a
      clocked spec that gave `pca_every = N` now means `N` clock units. The
      model's configuration and state change shape (`clock_since_pca`), so
      the schema is bumped and older states are refused (rule 5's pre-1.0
      exception). Tests: the clock cadence on a temporal clock with a
      duration, the row cap, both together, a capped gap counting as its
      cap, weight-zero rows counting, chunk invariance and resume mid-
      cadence, and the refusals. Built 2026-10-06: `tests/test_pca_cadence.py`
      holds the refresh rows read from the output to the rule written out
      over the stream's own clock (13 tests), and `ewcov.rs` holds the core
      to it over irregular steps and through a save mid-cadence (which fails
      if the clock since the last refresh is not saved). Schema 28.

- [x] 162. **A model window's snapshots are spaced on the clock, as the
      regressions' solves are scheduled -- requested 2026-10-06.** Size M.
      The user's words, after the audit of the parameters that count rows:
      "Do both these" (this and task 163). The window is clock time
      (`window_size`), but its snapshots were spaced in rows, so the
      boundary's error varied in time with the row rate and a burst
      inflated the ring. Now `window_every` is clock units, a number or a
      duration, `0` every row, and `max_rows_between_snapshots` caps the rows
      between snapshots; whichever comes first; with neither, every row, as
      before. `window.rs` keeps a `Cadence` (a clock spacing beside the row
      cap) and one due rule for `takes` and `offer`; thinning doubles
      whichever is in force; the stream's `ResidWindow` takes the model's
      cadence, so the spread's boundary stays the fit's; the over-budget
      refusal names the cadence in force. Two choices to know: `window` and
      `window_every` are written as nil where absent, so that only the row
      cap, the last field, skips in the positional encoding; and `marginal`'s
      `shards="auto"` sizes a flush by the batch under a clock spacing alone,
      which bounds no number of rows (speed only; the numbers are the same at
      any shard count). Built 2026-10-06 by a worker in a worktree:
      `tests/test_window_cadence.py` (44, against oracles from the
      definition, scikit-learn's `Ridge` on the window's rows among them; 43
      fail on the old code), six `window.rs` tests and an `ewcov.rs` resume
      test, each save test shown failing when the spacing or the window's
      clock is not saved. Schema 30.
- [x] 163. **`micro` prunes on the clock, as the regressions' solves are
      scheduled -- requested 2026-10-06.** Size S. The user's words, after
      the audit of the parameters that count rows: "Do both these" (the
      window snapshots, task 162, and this). DenStream checks every `Tp`
      clock units, and the model's own `xi(age)` reads `Tp` in clock units,
      but its checkpoint ran every `prune_every` learned rows, so a quiet
      spell kept faded summaries and a burst checkpointed often. Now
      `prune_every` is clock units, a number or a duration, `0` every row,
      and `max_rows_between_prunes` caps the learned rows between
      checkpoints; whichever comes first; with neither, every 100 learned
      rows, as before. The checkpoint moved from `learn_row` to the step's
      end, before the next row's metric, where it ran, so the row cap alone
      is bit-identical to the old cadence (the transcription in
      `tests/reference_cluster.py` follows, and the oracle runs both). Schema
      29. Built 2026-10-06; the same commit gives `ew_cov`'s `pca_every` the
      JSON tag its infinite value needs (task 161's miss: `to_json()`
      refused a spec with `max_rows_between_pca` alone).

- [x] 164. **The parameters that count rows where a clock would fit,
      reviewed -- requested 2026-10-06.** Size S. The user's words: "Find
      more example where a parameter references only rows when it should
      also be a clock", then "Do both these and document the rest for
      review". Task 163 built `micro`'s pruning and task 162 the window
      snapshots; §17 lists the rest with a
      recommendation each, and the decisions they need (task 165). Found on
      the way and fixed here: `coef_every`'s doc said learned rows, where it
      counts each group's accepted rows (measured: `coef_every = 3` filled
      rows 2, 5, 8, ... of a stream whose every third learned row was 8 and
      16).
- [x] 165. **The user's decisions on §17.** Size S each, once decided: the
      two rules for counting a cadence's rows, `micro`'s default cadence
      (100 learned rows, or DenStream's `Tp`), `coef_every` on the clock and
      the spelling of its default, and `bocpd`'s hazard in clock units.
      *Decided 2026-10-06 (§17's list): the sentence narrowed, `micro` kept
      at 100 learned rows, `coef_every` built as task 178, the duration
      hazard as task 179.*
- [x] 166. **The mutation and canary failures examined -- requested
      2026-10-06.** Size M. The user's words: "Examine mutants and canary
      fails". *Mutants.* The changed-lines runs on `a7a8f3c` and `e6b70a2`
      failed on survivors task 160 had already killed. The run on
      `c134b4a` left 3 in `TargetMoments::truncated`'s variance floor, and
      stopped at its 100-minute cap after 75 of 656 mutants, so the rest
      were run here: all 747 mutants in `e6b70a2..HEAD`, two hours at
      `-j 5`, from an export of the tracked files. 661 caught, 48 unviable,
      1 timed out (a hang, which counts as caught), 9 named equivalents and
      28 survivors. Each survivor is killed by a Rust test: the four Kish
      floors (`ew_cov`'s target and window sums, the Grams', `marginal`'s)
      at a remainder of exactly `64 ε` of the live sum, set in powers of two
      so the subtraction is exact; `ewridge`'s slots `j * nc + ci` with two
      targets and two combos, for a first solve that fails and for a combo
      skipped for no weight; `fit_has_shape` on a state with no fit and one
      part of its readiness left; `rcov`'s window and automatic ring at
      exactly the block and the `2^20` ceiling; the window cadence's
      refusals in the core; the serde defaults `no_row_cap` and
      `no_spacing`; and two `robust` models that differ only in their band
      systems. Two survivors were redundant conditions, now gone: `rcov`'s
      `automatic` (a given `bandwidth` or `max_bandwidth` is the ring, and
      was held to the ceiling already) and `ew_cov`'s `pca > 0` before the
      `pca_every` check, which now refuses a negative or NaN cadence with or
      without `pca`, as the spec does. A rerun of the 30 mutants on those
      lines caught every one. *Canary.* Run 37327376335 (py-polars
      2.0.0-rc.2) failed seven `test_windows.py` tests, a source's
      `ValueError` that 1.x wraps in `ComputeError`; task 158 fixed them in
      `a7a8f3c`. A local run at `HEAD` against py-polars 2.0.0 final passed
      4,163 tests and failed one: `test_kwargs_typing` put the package's
      parent in `MYPYPATH`, which for an installed wheel is site-packages,
      and mypy refuses it. The test now leaves an installed package where
      mypy finds it; the snippet type-checks under 2.0.0 either way.
- [x] 167. **No `.lazy()` in the examples: each query reads a file --
      requested 2026-10-06.** Size S. The user's words: "We don't want the
      example code in the readme or anywhere else to be sprinkled with calls
      to .lazy(), show the example by saving tables to disk and then reading
      them from a lazy frame." The README had fifteen, and one docstring
      (`LazyFrame.online.with_windows`) had one. The first fit saves its day
      to `prices.parquet` and each fit scans it; *Example data* saves `df`,
      `today` and `trades` to parquet, and `lf` and `later` are scans of
      those files; every `trades.lazy()` is `pl.scan_parquet("trades.parquet")`,
      the line above each such block naming the file; and the closed-groups
      example saves `by_block` and fits on a scan of it. The README test's
      namespace builds what *Example data* builds, files included, and its
      link check counts a block that reads `trades.parquet` or
      `today.parquet` as reading the example data. WRITING §3 has the rule,
      and `test_no_example_calls_lazy` holds the README, every document's
      python blocks and the docstrings to it. Left alone: prose that says
      what `df.lazy()` does (STATE-WORKFLOW R3, a `_frame.py` comment), a
      quotation in PHRASING's log, the code that accepts a frame and makes
      it lazy, and the tests' own inputs.
- [x] 168. **`drift_threshold` is a clock parameter, required with a clock
      -- requested 2026-10-06.** Size S. Review CC1 (§16); the user asked
      whether the problem was the unit or the default, why seconds ("there
      are many possible units"), and then: "make drift_threshold a clock
      parameter as you recommend". The detector sums each row's excess
      times its clock step (`drift.rs`), so its threshold is `sigma` times
      clock time. Every temporal clock is read in seconds whatever the
      column's unit (`span.rs`), and every other clock parameter is a
      duration there, so the internal scale never shows; `drift_threshold`,
      a plain number, was the one parameter it leaked into. Now it is a
      `Span` in `CLOCK_FIELDS`: a number of the clock column's units, a
      duration on a temporal clock, the mixtures refused as for every clock
      parameter, and required with a clock as `gap_cap` is. 20 stays the
      default only without a clock column, where it is the classic test.
      Measured on 3,000 rows a minute apart: the old default, `"20s"`,
      flags 244 rows of pure noise and `"20m"` none, and `"20m"` finds a
      burst 18 rows in; `"20m"` on a temporal clock flags exactly what 1200
      does on the same clock in seconds, whatever the column's unit.
      `tests/test_drift_threshold.py` holds each of these, the refusals and
      a resume; `spec.rs` holds the rules in the core. Specs in tests and
      `scripts/release_probe.py` that turned drift on over a numeric clock
      now give 20.0, the old default, so their numbers do not move (the
      probe runs under the released wheels too, which take a number).
- [x] 171. **`rcov`'s pre-averaged estimate carries CKP's finite-sample
      rescaling -- requested 2026-10-06.** Size S. Review CE1 (§16); the
      user: "follow your recommendations for all remaining review
      decisions". Under `psd=False` the estimate is divided by
      `1 − ψ₁/(2ψ₂kₙ²)` (CKP footnote 1, read from the PDF's page 9: the
      error of `Σxxᵀ = 2nΨ + ∫Σ` has expectation zero). A test from the
      footnote holds every entry of the estimate to the integrated
      covariance within 4 standard errors over 300 blocks at `kₙ` = 3, 6
      and 10 (the old code sat on the footnote's factor, 4.9 to 121
      standard errors off); the oracles written from Eq. 9 take the
      footnote; `GOLDEN_RCOV_PREAVG` moved by 19/16 exactly. Through the
      bank, the reviewer's script reads 1.00083 ± 0.00098 at the default
      window over 59,980 blocks. The divisor is 0 only at `kₙ = 2`, which
      `validate` refuses without `psd` (CE5). A worker built it on
      `task171-ce1`.
- [x] 169. **The Rust test oracles in one place, independent of `solve.rs`
      -- requested 2026-10-06.** Size S. The user, on seeing `deco`'s
      dense-density test take its expected value from `SpdFactor`, the
      Cholesky it checks: "If we are changing Cholesky factor calculation
      should we not do this in one place?", then "Do this", then "Point
      Claude.md to the new oracle". `crates/online-core/src/oracle.rs`
      (`#[cfg(test)]`) holds faer's partial-pivot LU (solve, inverse,
      quadratic form) and self-adjoint eigensolver (log-determinant,
      Gaussian log-density, eigendecomposition), none of which `solve.rs`
      runs; every ad hoc faer oracle in online-core's tests moved onto it
      (bocpd, deco, ew_cov, hmm, lasso, rcov, ewridge), and the two
      hand-written Gauss-Jordan oracles (`EwCov::inverse_from_scratch`,
      ewclass's `inverse_det`) went with them. Two tests stopped checking
      the solver with itself: `deco`'s dense density, now from the
      eigendecomposition, and `ew_cov`'s Mahalanobis test, whose d² came
      from `precision()`, the same Cholesky. Measured by corrupting
      `solve.rs` in the worker's tree: a constant added to `log_det` or to
      `quad_forms` passed the old `deco` test and fails the new one (halving
      `log_det`, the brief's example, failed both). Expected values moved by
      at most 6.3e-15 absolute; every tolerance is 1e-9 or wider. CLAUDE.md's
      Style rule now points at the module.
- [x] 173. **Forward windows: a silent group needs a clock (PC1), and a
      reset keeps a complete window whole (PC2) -- requested 2026-10-06.**
      Size M. Review PC1 and PC2 (§16). The rule lives in `clock_cfg_of`
      (`ClockPolicy` gains `group` and `looks_ahead`): `with_windows`
      refuses a window looking ahead under `group` with no clock column, as
      a spec refuses such a window target, with one message. A reset keeps a
      forward `right`/`both` window whose far edge is the last row before it
      whole, as a cut does; the brute force treats a stretch a reset ends as
      it treats one a cut ends. Tests: the mirror identity across resets,
      the brute force over step-back and session-reset grids, the resolver
      path, each refusal (Rust and Python, each failing on the old code).
      The bank's predictions, `learned_clock` and `weight_sum` match the old
      build bit for bit on 48 comparisons: a reset clears the rows awaiting
      their labels (`apply_label_delay`) before any resolution applies, so
      only `with_windows` values move (one per reset, null to a value). In
      the bank, PC1 is consistency rather than memory: its per-group cores
      hold only the silent group's own window.
- [x] 174. **`lasso` counts each target's selection errors from its own
      `min_weight` -- requested 2026-10-06.** Size S. Review CA3 (§16).
      `LassoCfg.target_min_weight` (one threshold per target, empty for the
      model's own) is filled from the spec's list; `step` folds a target's
      error only where the target's own weight before the row reaches its
      threshold, hard rule 8's gate. `[0, 60]` and `[60, 60]` now agree for
      the second target on every row (141 differed); a target present on
      every row under a scalar or an equal list is bit-identical to before;
      a target with gaps moves under any spelling (11 to 26 rows of 240),
      since its own weight lags the shared one -- the decision's rule, which
      the brief's "a scalar is unchanged" missed. The cfg is in the state,
      so `SCHEMA_VERSION` is 31 and the bank refuses 30 and older.
- [x] 175. **A model window's edge on the decayed clock, held exactly --
      requested 2026-10-06.** Size L. Review CB1 (§16); the user chose
      option 1 of two ("Go with option 1"): stamps, against a compensated
      summed clock, which exact summation of rounded 1 ms steps would have
      left at 1.0000000000000000208 s, consistently past 1 s. A model
      window's clock is the decayed one (README, *A hard window*), so the
      stamp is: integer nanoseconds summed from capped integer steps on a
      temporal clock (`gap_cap` and `session_gap` as their durations' ns,
      `ExactCaps`), the raw value beside the removed time on a number clock,
      the row count without one. `ClockState::advance_stamped` makes it,
      opt-in, so the window operators' clock writes the bytes it did
      (`WINDOWS_VERSION` stays 7); a default no-op
      `OnlineModel::stamp_next` hands it to the five windowed models, so no
      `step` signature changed; the ring keys `(summed clock, stamp,
      snapshot)`, decay reads the summed clock bit for bit, and the edge,
      the stale rule, `window_every`'s spacing and thinning read
      `Stamp::cmp_span`. Wired through the residual ring, the pre-pass, the
      embargo's replay, resets and save/resume. The window operators'
      `Gap` now compares through `Stamp::cmp_span_ns`, its behaviour
      unchanged (their 47 tests). Tests (`crates/online-polars/tests/window_edge.rs`
      and `tests/test_window.py`), each oracle from the raw integer ns or
      raw numbers: every model and unit exact at 1 ms under `"1s"` (7 rows
      wrong before); a clock of tenths under 0.3 (186 rows); every clock
      event, a capped hour, `session_gap`, skipped rows and a restart (528
      rows); chunking at 1, 7 and 37; `window_every` spacing and a save
      between snapshots; the residual window's boundary; an embargoed row
      learned at its own stamp; the pre-pass. `SCHEMA_VERSION` is 32 (task
      174 took 31 the same day). Limit: a model compares a temporal
      difference with its window through `seconds_of_ns`, exact for windows
      under 2^23 s or of whole seconds; a longer one with a sub-second part
      can tie a nanosecond late until the cfgs carry the window's ns.
- [x] 172. **`kalman`'s warm-up is unit-free: the noise from the first
      innovation, the prior `p0` times it -- requested 2026-10-06.** Size
      M. Review CC4 (§16). The worker's first commit took the noise from the
      row's innovation, where it was the literal 1.0, and found that at the
      default `p0 = 1` the warm-up became *more* unit-dependent (1.8 to 2.8
      target standard deviations between targets at 1e-6 and 1e6, against
      0.8 to 1.2): the old `R = 1` and `P0 = 1` were both unit-free
      literals, and only one of them moved. The user chose to make `p0`
      relative ("p0 relative"): the prior is `p0` times the first noise
      estimate, set at the target's first row with one (an all-zero `P` is
      the unsized state, so no layout change), and `p0·obs_var` with
      `obs_var` given. The gap is now at most 1.4e-15; the equivariance test
      (every target scaled by `2^±20`, at the default `p0` and at 1/4, eight
      configurations) holds bit for bit, and failed on both earlier rules.
      `kalman_ref`, both filterpy second opinions and river's mapping
      (`p0 = beta/alpha`) state the new prior: filterpy meets the bank to
      1.0e-15, and misses it by 0.28 and 52 under `P0 = I`. Golden values
      moved, each matched to `kalman_ref` before re-pinning; VALIDATION's
      kalman rows were regenerated. Found, not changed: `huber`'s cut takes
      `s = 1` in the target's units before a residual, the same kind of
      literal.
- [x] 170. **Cholesky factors move in place, in `solve.rs`; `quantile`
      takes back a sixth of TC1b's cost, bit for bit -- requested
      2026-10-06.** Size M. The user: "implement the cholesky performance
      enhancement for tc1b", in one place ("should we not do this in one
      place?"). `SpdFactor` keeps its own lower factor, factorized exactly
      as faer's `llt` does, and moves in place: `updated(c, v)` to the
      factor of `c·A + v vᵀ` and `congruent(e)` to that of `E·A·E`, by plane
      rotations (LINPACK's `dchud` form) -- faer's `rank_r_update_clobber`
      lost three digits where a step dominated a pivot (1,415 units of
      rounding on a 2x2 at condition 1e6, against 2.6 for the rotations). A
      factor that needed jitter never moves; a move leaving a zero pivot or
      an overflow spends it; after 64 moves it refactors, since drift grows
      about as the square root of the moves and a count does not depend on
      the chunking. Held to faer's fresh factorization of the moved matrix
      over condition numbers to 1e12. Bit-identical savings ride with it: a
      nudge's quadratic form in a kept faer column, the band system read
      from the Gram in place; 38 of 38 output frames match the old wheel to
      the bit, and the bank runs `quantile` 16% faster at ten features.
      **Parked, for the user's word:** `robust` moving its band factor at
      ridge 0 (branch `task170-cholesky-robust`, `8c03592`). The algebra
      allows it only at ridge 0 (a ridge under forgetting leaves
      `(1 − a)·r·I`; standardized, `r(I − aE²)`), it adds about 4% there,
      and it breaks the README's resume guarantee: the band factor is not
      state, so a resumed model refactors where the unbroken one holds a
      moved factor, and 1,357 predictions part by up to 3.4e-13 relative.
      Options: accept rounding-level resume for `quantile(ridge=0)`, make
      the band factor state (a schema bump), or leave it parked. *Decided
      2026-10-06, the user's word: "Save the band factor in the state".*
      Built: each target's band system (the factor as its packed lower
      triangle, jitter rung and shift and move count, refused on reading
      if incomplete, non-positive, non-finite or inconsistent; the kept
      columns and scales, held to the Gram's own to the bit) is `robust`'s
      state, schema 33, compared in equality, never refactored on restore;
      `huber` keeps no band system, and the row buffer compares equal
      whatever it holds. A fit saved at any of 101 rows (Rust) and 21 rows
      through the bank resumes to the bit, where three save points went off
      before; a JSON round trip carries the moved factor; thirteen damaged
      states are each refused. About 670 bytes a target at ten features.
- [x] 176. **`embargo` counts down in doubles, so an embargo of exactly k
      steps can release a row a row late -- found 2026-10-06 by task 175.**
      `apply_label_delay` (`stream.rs`) subtracts each row's `elapsed` from
      the remaining embargo in doubles: on 1 ms Datetime rows under
      `embargo="2s"`, all 999 rows were released at 2.001 s; `"5ms"`,
      `"300ms"` and `"1s"` happened to land exactly. CB1's class of bug, on
      the elapsed (uncapped) clock: the fix holds the elapsed time exactly
      in `ClockState` and `PendingRow`, as task 175 holds the decayed one.
      Reproduction: the session scratchpad's `review3/fixwork/cb1/embargo_probe.py`.
      *Built 2026-10-06 on the user's word ("Follow your reco on all").*
      `ClockState::advance_stamped` gives each accepted row its place on the
      elapsed clock (`ClockAdvance::elapsed_stamp`, task 175's `Stamp`):
      integer ns summed from the raw steps, uncapped, skipped rows counted,
      a restarting session counting its gap; on a number clock the raw value
      beside what the elapsed clock does not count; else the row's place. A
      held row keeps its arrival (`PendingRow.arrived`) and is released when
      an accepted row's place is at least the embargo past it
      (`Stamp::cmp_span_ns`, with the embargo's ns when a duration),
      inclusive as before. The old countdown also released rows early (a
      clock of tenths under 2.0; `"100d2ns"` by 1 ns). Tests in
      `crates/online-polars/tests/embargo_exact.rs` (six, each failing on
      the old code: 600 rows late in every unit under `"2s"`; early rows on
      tenths; 1,236 rows across a capped gap, skipped rows, session changes
      and a reset; chunking and 12 save points; a formula target on both
      paths) and `test_label_delay.py::TestTheReleaseIsExact`. A held row
      with no place is refused at load. Schema 34. No pinned value moved.
- [x] 178. **`coef_every` counts the clock, as `solve_every` does --
      requested 2026-10-06.** Size M. §17's decision 3. Unset (`None`, the
      new default) is the old default, each group's last row in each chunk;
      `0` is every row (it meant the default); a number of clock units or a
      duration is a `coef` row once the clock has moved that far since the
      group's last, measured on task 175's exact stamps, so `"2s"` on 1 ms
      rows writes rows 2000, 4000, ... where summed doubles would write each
      a row late; `max_rows_between_coefs` caps the accepted rows, counted
      as `coef_every` counted them; whichever comes first. Without a clock
      column the clock is the row's number from 1, so `coef_every=N` keeps
      its N-th, 2N-th, ... timing. A reset starts the cadence over. Under an
      explicit cadence the `coef` rows no longer include each chunk's last
      row, so they are chunk-invariant; `predict` still writes the chunk's
      last row. Refused: a duration on a number spec and the reverse, a
      negative or infinite `coef_every`, `max_rows_between_coefs=0`, either
      on a model without coefficients (`coef_every=0` there used to pass).
      No model's numbers move. `tests/test_coef_cadence.py` (111 tests,
      through all 15 kinds with a `coef`, chunking and save/resume, held to
      the rule written out over the stream's own clock) failed 91 of 107 on
      the old build. The README's first fit and every clocked use that
      meant every row now write `coef_every=0`; `examples/bank.toml` uses
      `max_rows_between_coefs`. Schema 35 (task 176 took 34 the same day).
- [x] 180. **`solve_every`, `pca_every` and `prune_every` sum their clock in
      doubles -- found 2026-10-06 by task 178.** CB1's class of bug (tasks
      175, 176): two thousand steps of 1 ms sum to 1.9999999999998905 s, so
      a cadence of `"2s"` fires a row late. `window_every` (task 175) and
      `coef_every` (task 178) read exact stamps; these three should too.
      *Built 2026-10-06 on the user's word ("Do the two code changes and
      then push").* A new `Since` (`crates/online-core/src/since.rs`) keeps
      the last event's stamp, the last row's (ewridge's solve after a blend
      falls between rows) and the clock summed since the event (the path of
      a row handed no stamp, which keeps a direct core caller unchanged to
      the bit -- read literally, "the clock summed since construction"
      would have fired 0.1-steps under 0.3 at rows 3, 7, 11 instead of 3,
      6, 9). Before the first event the clock starts at the first stamped
      row's stamp less its own step. All six models (`ewridge`, `lasso`,
      `huber`, `quantile`, `ew_cov`, `micro`) fire exactly every 2,000 rows
      of a 1 ms clock under `"2s"` in ms, us and ns (18 cases, each a row
      late per event on the old code); irregular tenths under 1.3 fire by
      one subtraction (regular tenths under 0.3 cannot fail: their sums
      restart exactly at each event); chunking and a save between events
      (checked by sabotage). Schema 37, the bank refusing 36. No pinned
      value moved.
- [x] 177. **`huber` and `quantile` down-weight nothing until a residual
      scale exists -- requested 2026-10-06.** Size S. The user's word
      ("Follow your reco on all"), option (a): a row's Huber weight is 1
      until σ² is finite and above 0 (`Robust::residual_scale`; σ²'s weight
      is not also required, since only a weighted residual writes σ² and the
      weight only decays, so a gap of 1,333 half-lives keeps its scale). The
      same literal reached `quantile`'s band, `h = s·max(quantile_eps,
      floor)`, past its warm-up (on a delayed first prediction, and while
      every residual was exactly 0): with no scale a quantile row is a
      least-squares row, a separate commit since it extends the decision.
      Equivariance tests (every target by `2^±20`, bit for bit, five Huber
      and four quantile configurations) failed on the old code (up to 35
      times apart, and 2e5 of a target's units). Found on the way and fixed
      in the test oracles, no library code: `robust_ref` aged σ²'s weight
      unlike the core on rows of weight 0 (S13, N6, CC5; 4.6e-2 off on such
      a stream, now 5e-15), and `kalman_ref` on a row with no prediction
      (N6; 1.75e-3, now 1e-15). No pinned value moved.
- [x] 179. **`bocpd`'s hazard takes a duration, applied before the row it
      leads into -- requested 2026-10-06.** Size M. §17's decision 4. The
      first brief put the chance on the step into a row, which dated every
      break a row late: a row's hazard is the chance of a break after it
      (the worker's enumeration oracle, Murphy 2007, from the definition).
      The user then chose the streaming form and applying the step on rows
      of weight 0 ("Build, apply on 0-weight rows"): under a duration `τ`,
      `before_row` moves `h = -expm1(-d/τ)` of every run's mass to the empty
      run before the row is read; `p_change` is the chance the row began a
      run; a row of weight 0, or one the predictive cannot read, applies its
      step and learns nothing (`held_values.rs`' named exception); a step of
      0 has no chance (`-inf` flows through; with `prune_below = 0` each
      leaves a dead run until `max_run` folds it). `ln(1 − h) = −d/τ`, no
      libm; `ln h` from `ln(-expm1(-d/τ))`, as the log joint's other terms.
      `BocpdCfg.hazard_on_clock` is last and skipped when false, so per-row
      states write the bytes they did and the bank still loads 35; schema
      36. A second table beside `CLOCK_FIELDS` names fields that take a
      duration or a unit-free number. Tests: the oracle on five cases
      (`p_change` to 1.1e-14, `run_mean` to 8e-14), the composition of two
      steps, the per-row form from row 1 on regular steps, chunking and
      resume, the refusals, a proptest and a golden checked against a
      longhand in probabilities. A number `hazard` is byte-identical in
      outputs and state over five configurations. Found and fixed: the
      `gaussian` emission learned a NaN row, since `SpdFactor::quad_forms`
      clamps with `max(0.0)`, which turns NaN into 0; that clamp in
      `solve.rs` can swallow a NaN for other Rust-API callers too, left for
      the user.
- [x] 181. **A quadratic form of a NaN is NaN, and every caller refuses the
      row -- requested 2026-10-06.** Size S. The user: "Do the two code
      changes and then push". `solve.rs`'s `clamp_rounding` is `if acc < 0.0
      { 0.0 } else { acc }` (bit-identical to `max(0.0)` for every number,
      since `acc` starts at +0.0), used by `quad_forms`, `quad_form` and so
      `quad_forms_logdet`. Each caller, read: `hmm` refuses the row, and its
      refusal path aged the states by a zero-weight update that wrote `0·NaN`
      into the co-moments, so `age_states` decays them instead on a row that
      is not finite; `ew_class` never hands the form a NaN feature, but a
      form overflowing to ±∞ on a finite row is NaN, and its softmax skipped
      it, so a guard withholds the row; `robust` never nudges without a
      finite prediction, and a NaN leverage from overflow dropped the step's
      bound, so it reads as unbounded and the step is 0; `ewridge`'s row
      inflation and `deco`'s `loglik` are outputs only; `bocpd` keeps task
      179's guard; `ew_cov`'s `mahal_of` already refused a non-finite d².
      Tests in `solve.rs`, `hmm`, `ew_class`, `ewridge` and `robust`, each
      failing on the old code but the rounding guard (a factor built by hand
      to give a raw form of exactly −0.125). Finite inputs move nothing.
- [x] 182. **Four older NaN gaps, outside the core's stated contract (every
      input finite) -- found 2026-10-06 by task 181.** None goes through a
      quadratic form, and the bank never passes a NaN feature, so each is
      reachable only through the Rust API: `hmm`'s warm-up buffer keeps a
      NaN row and seeding replays it into a state (every later output
      non-finite); `deco`'s standardiser learns a NaN feature (`loglik` NaN
      ever after, `u` and `rho` silently reading the other columns);
      `robust` learns one through its least-squares arm (every later solve
      fails, the fit frozen while predictions look finite); `EwCov::var` and
      `EwDiag`'s variance turn a NaN into 0 with `.max(0.0)`. Reproduction:
      the session scratchpad's `review3/fixwork/nan_clamp/zz_probe_nan_paths.rs`.
      *Built 2026-10-06 on the user's word ("Fix the older nan bugs").*
      `hmm` refuses a non-finite row before seeding and does not buffer it
      (the buffered rows, the counts and the states age by its `lam`);
      `deco` learns such a row at weight 0, so its standardiser reads no
      moment (the state equals a zero-weight row's); `robust` routes a
      row with a non-finite feature to its branch for a row the loss cannot
      learn from (counted in `n_eff`, the Gram aged, the cross-moment, mean
      and σ² untouched); `clamp_rounding` (`pub(crate)`, `v <= 0.0` so a
      `-0.0` becomes `+0.0`) serves `EwCov::var`, `EwDiag::var` and
      `EwDiag`'s `including().moments()`. Each variance caller read: most
      take the branch 0 took; `ew_cov`'s emitted `var`/`std` and the
      ewridge/lasso windowed Gram view can see NaN, from a row those models
      still learn (task 183). Tests in `hmm`, `deco`, `robust`, `ewcov` and
      `ewdiag`, each failing on the old code. Finite inputs move nothing.
- [x] 183. **`ew_cov`, `sgd`, `ewridge` and `lasso` still learn a row with a
      non-finite feature -- found 2026-10-06 by task 182.** Through the Rust
      API only, outside `OnlineModel`'s contract (every input finite); the
      bank never sends such a row. Other models were not all checked.
      *Decided 2026-10-06 (the user, "Your reco"): make it uniform.* The core
      keeps the contract itself, so its line becomes a guarantee: one
      predicate, the bank's `usable` (finite, within `INPUT_BOUND`), moved
      into `online-core`; a target that is not usable is absent (predict-only,
      the bank's rule); a feature or weight that is not usable makes the row
      a zero-weight row that keeps nothing of its own (its clock ages the
      state, nothing else of it enters, it is not counted), its `pred` NaN
      for a feature; every one of the 20 models audited and held to it by one
      contract test; the refusals of tasks 179, 181 and 182 that count the
      row (`robust`'s `n_eff`, `hmm`'s and `bocpd`'s failures) brought into
      line, a usable row that fails anyway still counted.
      *Built 2026-10-06 by a worker: `60d5356` on branch `task183-unusable`,
      unmerged.* `usable`/`all_usable` live beside `INPUT_BOUND`; every
      model's `step` opens with an inlined check and a `#[cold]` refusal that
      steps the row again at weight 0 with its features 0; the contract test
      `refuses_unusable_values` failed 43 of 103 legs on the old code across
      all 20 kinds; no expected exception held (no zero-weight row takes a
      ring slot, a warm-up row or a likelihood) except a blocked `ewridge`
      Gram, held as zeros; `deco` reports NaN for `u`/`rho` on such a row.
      Cost: the core `step` +6-16% on `sgd`/`pa`, +3-6% on `kalman`;
      through the bank (one thread, 300,000 rows, best of seven interleaved
      rounds, load ~3) +3.4%/+4.7% for `sgd`/`pa` at two features, +0.8% to
      +2.3% elsewhere, inside the rounds' 2-5% spread. *Cost accepted
      2026-10-06 (the user, "Your reco all"): merged with tasks 184-193.*
      *Done 2026-10-07*, merged first of the eleven, the cost as measured
      and accepted. Two of task 186's tests reached their case through a row
      past the input bound, which this refuses; they were restated at the
      merge: `bocpd`'s failed predictive (CE10) through a hazard of 1 and a
      step the clock cannot read, and `rcov`'s unrunnable repair (CE9)
      through a sum held at infinity, as only a state built by hand can be.

- [x] 184. **The survivors of the mutation run over `f49058d..8e28c1f`** (909
      mutants, tasks 168-182: 23 missed, 1 timeout). Each killed by a test
      or recorded as equivalent or tolerated with its reason. *Worker
      `task184-survivors`, 2026-10-06.* *Done 2026-10-07*: 11 killed by
      new tests (`clock`, `ewclass`, `marginal`, `kalman`, `lasso`,
      `solve`), each checked by hand on its mutant; 7 new equivalent entries
      and 1 tolerated (`kalman.rs:753:22`, rounding in an unsized `P`);
      TESTING.md counts the equivalent mutants (167 entries when this
      closed; 165 in 165 since tasks 201 and 204 dropped one each, review
      5 F8) and 32 tolerated in 28. `solve.rs:281`'s timeout still spins in
      an unbounded test loop.
- [x] 185. **A loader refuses a damaged state by name** (§18: CA2 CA3 CA4 CA6
      CB2 CB8 CE3 CF4 CD14 PD1 PB4 PA8 CF11 PA2 CF12). Every `restore` runs
      its cfg's `validate()` and checks every shape it will index; the
      windows core, held embargo rows and the envelope's spec indices are
      checked at load. *Worker `task185-loaders`.* *Done 2026-10-07*:
      `check_cfg` in all 20 restores, the shapes each next row reads,
      `Windows::check` at load, a newer file told to upgrade; each test
      failed on the old code by a panic or a load. No number moved. At the
      merge the lag ceilings (task 193) joined the shared `check_lags`, so a
      restored cfg is held to them, and the kept systems' Gram check went
      with `System.gram` (task 186).
- [x] 186. **Round-4 model defects** (§18: CC1 CE1 CC2 CC3 CA1 CA11 CB7 CD12
      CE8 CE9 CE10 CA5 CE7 CF3 CA10 CD17): no prediction from a fit nobody
      solved (NaN, CA5's rule; the per-target first solve is the user's),
      `ew_class`'s windowed `coef`, `sgd`'s overflow guard, `kalman`'s shared
      noise read once a row (with a `filterpy` second opinion), schema 38
      (`System.gram` dropped). *Worker `task186-models`.* *Done
      2026-10-07*: CC1 refined, a slot with no weight keeping a fit an
      earlier solve made or `ewridge`'s `coef_prior` under a ridge (`Fit`,
      NaN tagged in JSON). Moved: `pred` 0.0 to null on 56 of 80 rows (CC1),
      `share_p`'s MSE (CC3), a windowed `ew_class`'s `coef` (CE1), `rcov`'s
      `iv_sparse` and `iq` (CE8). Left: `share_p` still sequential, so target
      order matters; CD12's γ>0 solve is inaccurate near its floor (195).
- [x] 187. **Round-4 defects in the bank, the stream and the surfaces** (§18:
      PA1 PA4 PA9 PA12 PB1 SF2 SF3 SF5 SF6 SF8 SF9 SF10 SF11 SF12 TB2 PC7).
      *Worker `task187-surfaces`.* *Done 2026-10-07*: all 16, and a panic
      SF2's dry run found (a `seqtest` comparison assembled only when it had
      work: `predict` on a fresh bank, `fit_predict` on an empty grouped
      frame). Moved: `settled_frac` after a drift reset under an embargo
      (PB1), `describe()`'s formula-target row (PA9), `refresh_time` on an
      unusable value (PC7). Left: its clock column casts non-strictly (197).
- [x] 188. **Round-4 defects in the helper modules and the window operators**
      (§18: YB1 YB2 YB3 YB4 YB6 YB7 YB12 YB13 YB14 YB15 YB17 YB18 YB19 YB21
      PD3 PD4 PD5 PD6 PD7 PD9 PD10). *Worker `task188-helpers`.* *Done
      2026-10-07*: all 21, with scikit-learn's and statsmodels' second
      opinions. Moved: identity shrinkage (0.0754 to 0.1051), `nearest` 20
      to 19 iterations, three dtypes. Left: an unnamed relative target scored
      without `spec=` cannot be told apart (a decision; moot since task 201
      removed relative targets). YB3, probed at the
      merge: no documented flow runs `embargo` over this package's own plan
      forms, which it warns about though they read twice correctly.
- [x] 189. **Round-4 test findings** (§18: TA1-TA5 TA9-TA12 TB4 TB5 TB7-TB14
      CF7-CF10 CD15 CC10 PB6 YA10 CE6): independent oracles (`filterpy`,
      scikit-learn's `partial_fit`, `hmmlearn`, `padasip`), generators that
      reach the bounds, a second golden bank, a cross-OS hand-off over every
      kind, assertions that can fail. *Worker `task189-tests`.* *Done
      2026-10-07* in four commits: the four oracles agree to 1.4e-13 or
      better, a second golden bank of 291 entries and 29 Rust goldens, a
      hand-off over every kind. No library number moved and no golden was
      re-pinned at the merge, where three of its tests met task 193's rules
      (`kalman` takes `coef_half_life` or `q`; `rcov` reads no `theta` under
      `"kernel"`). CC10 checked there: without `.min(1.0)` its test fails.
      *Follow-up 2026-10-07*: the wider generator left too few streams with
      a scored row, and the rule-2 property failed Hypothesis's
      `filter_too_much` health check (`kalman`, once in a gate and once in
      460 runs; a median of about 20 streams rejected before the 10th kept,
      in `kalman`, `ewridge` and `rls` alike). Not a leak: no prediction moved in 22,650 compared
      streams, and 27,000 streams each run twice gave the same predictions
      both times. `warmed_streams`
      now opens each stream with four ordinary rows, so every stream has
      one; its `assume` became an assertion.
- [x] 190. **The API snapshot pins what the policy calls stable** (§18: AP1 AP2
      AP8 YA1 YA8 TB1 DB2 DB3 DB4 DB5): resolved defaults, helper signatures,
      frame columns, TOML keys, CLI flags, env vars, enum values. *Worker
      `task190-pins`.* *Done 2026-10-07*: `resolved_defaults` renders each
      kind from the bank's own build, and `test_spec_defaults.py` holds the
      README's `min_weight` table to it. Regenerated after the merge: schema
      38, `kalman`'s and `rcov`'s new defaults, `po.eval`'s `spec=`. Left: a
      value a model estimates shows as null; `po.corr`'s and `po.sim`'s words
      are not pinned.
- [x] 191. **Round-4 fixes to the reader-facing documents and the release
      workflow** (§18: DA3 DA5 DA8 DA10-DA15 AP14 AP16-AP18 AP21 AP23 CB5 CB6
      CE6 PB5 PD6 PD7 SF4-SF6 SF13 YA3 YA7 CI5 CI7 CI9 CI10 CI12-CI14 CI16
      CI17 CI19). A workflow edit: the next release rehearses first. *Worker
      `task191-docs-ci`.* The CHANGELOG's DA6 and DA7 are the coordinator's
      (done in the working tree). *Done 2026-10-07*: the sdist smoke, the
      tag's headings, `--locked`, the README pin and `[Unreleased]` checks,
      OUTPUTS' dtypes and *also null*, RUNNER's exit status. DA15 and SF13 did
      not reproduce. Left: an infinite half-life is null in `output_index`;
      two release legs pin no Python; the state legs lack `--locked` (199).
- [x] 192. **The dated records name their dated spellings** (§18: DB9-DB13
      DB15-DB28). *Worker `task192-records`.* *Done 2026-10-07*: one
      table of renamed names (PERFORMANCE, "Names that changed") and a
      pointer at the head of every record; statuses made current, REGIMES §6
      and §9 under its doc test, provenance on PERFORMANCE §26-§36. Left:
      three docs/README.md index rows (199).
- [x] 193. **Round-4 input validation and the words of a refusal** (§18: CD10
      CF2 CE9 CA6 CC7 CF6 CE4 PC6 PC8 YA5 PC10 PC11 YA4 YA6 PA6 PB7 CD16).
      *Worker `task193-validation`.* *Done 2026-10-07*: the ceilings, the
      parameters a mode does not read, the core held to the spec, the words
      of every refusal (`spec_diff.rs` for PA6); no accepted spec's output
      moved. At the merge: `hmm`'s `k` held to 1024 in core, spec and builder,
      and `micro`'s matrix kept up to 4,096 potential summaries, its O(m)
      path past it, a test holding the two to the same labels.
- [x] 194. **The bank's and the stream's state and frames** (§18, decided):
      the stream's persisted fields as one sub-struct (S6, D7); a key
      column's dtype kept in the state and a change refused (N22); the clock
      range in each frame in the clock's own dtype (N18); counts `UInt64`
      everywhere (N19); `closed_groups`' rcov block prefixed (N20); `group=`
      takes a list of `str | None` and integer keys sort as numbers (N21);
      `coef` on the group's last accepted row of a chunk (S2); `predict`
      under an embargo documented and pinned (S3); `rows_fed()` (N9);
      `t_stat`/`pair_t_stat` (N8). Schema 39. *Worker `task194-state-frames`.*
      *Done 2026-10-07*: a number clock's range is Float64 (an integer one its
      own dtype since task 200); frames over specs whose clocks differ in
      type ask for `spec=`. N22's `key_dtypes` replaced `key_integer`, and
      G1's branch went. No golden moved. `rows_seen`'s refusal is pinned in
      `test_state_frames.py` and, at the merge, `test_renames.py`.
- [x] 195. **The models' defaults and semantics** (§18, decided): `sgd`'s
      `huber_delta` and `eps` and `pa`'s `eps` in units of the residual's EW
      standard deviation (U1); `standardize=True` for `sgd`, and on `pa`
      (U2); `huber_delta` 1.345 (U3); `bocpd`'s `prior_scale` from the first
      rows (U4) and `prior_nu` per emission (U5); units stated for the rest
      (U6); `sgd`'s logistic labels clamped, `strict_binary` (S4); a Poisson
      fit's `hit_rate` null (S5); an exact zero prediction left out of
      `hit_rate` in the bank and in `po.eval` (S6); a target's own first
      solve (S9b); `rls`'s `ridge` renamed `delta` (N11). *Worker
      `task195-defaults`.* *Done 2026-10-07*: goldens re-pinned with replicas
      written from the docstrings (sgd, pa, pa_box; pipeline 45 of 505).
      `bocpd`'s robust emission found not unit-free and `pa`'s residual band
      stalling far from zero (task 202 replaced it). S6 is `pred != 0`
      beside `y != 0` since task 201 removed the ratio's centre;
      `corrchange`'s sequential floor is 5e-11.
- [x] 196. **Names in the specs, the models and the command line** (§18,
      decided): `type = "ewridge"` and the `huber:`/`quantile:` message
      prefixes (N1); `chunk_size` on every surface (N2); `dist_second` (N7);
      `lag_corr` (N10); `*_every_rows` and `permute_every`'s redraw as
      documented (N14, S1); `--load-state` (N15); holt's `level_half_life`
      dropped (N16); a `closed` parameter on the windowed models, default
      `"right"` (N17); `--skip-learned` (N26); a row cap of 0, `pca = 0` and
      `stats = []` refused (U7). Every old name refused, naming the new one.
      *Worker `task196-names`.* *Done 2026-10-07*, all but `stats = []`, which
      stays accepted: it is the documented accumulate-only mode (E43), and
      refusing it is the user's decision.
      `lag_corr` reached `marginal`'s and `closed_groups`' columns too. The
      windowed goldens run `closed="both"`, the old edge to the bit. At the
      merge `--skip-learned` took task 200's integer clocks.
- [x] 197. **Names in the helper modules** (§18, decided): `po.eval`'s
      `min_samples` and `group` (N3); `window_metrics(every=)` (N4);
      `absorption_shift` (N5); `po.gram.lasso_path(penalties=)` (N6); the
      five module constants in `__all__` (N24). *Worker
      `task197-helper-names`.* *Done 2026-10-07*: `from_sums`' `min_obs`
      too, and `refresh_time` refuses a clock column of another type (from
      task 187). The refusals are `_renamed.py`'s, the one mechanism the
      merge made of tasks 196, 197, 198 and 201's helpers.
- [x] 198. **The mechanisms of the 1.0 promise** (§18, decided): the state
      fixture harness, one frozen state per `ModelState` variant checked
      three ways, `MIN_SCHEMA_VERSION` raised to the schema shipped, the
      variant names frozen, the dead `#[serde(default)]` repairs deleted, the
      released-state test loading for 1.x (D1); the deprecation warning and
      its forwarding table, used from 1.0 (D2); the unstable label and its
      opt-in warning on the windows state format, the formula tree's written
      form, `fit_predict_arrow`, `po.sim` and `po.corr` (D5, N23); task 116's
      parts that move no default, the current floors declared final (D8);
      `ModelBank.load`'s note on damaged payloads (D14). *Worker
      `task198-mechanisms`.* *Done 2026-10-07*: D8's column is
      `weight_sum_settled` (hard rule 8). The merge deleted the loaders the
      floor passed (deco's `rho_w`, `EwDiag::diagonal_of`, `ew_cov`'s
      schema-1 names and P² markers, marginal's runs, schema-14's held-rows
      clock), raised both minimums to 44 and regenerated the fixtures.
- [x] 199. **The 1.0 policy text and the process** (§18, decided): the
      README's stability table, versioning after 1.0 and state-file promise
      (D3, DA1 DA2 DA4 DA16); a polars floor leg in the release and monthly
      in the canary (D4); the changed-lines mutation job sharded (D10); a
      weekly 1.95 check (D11); SECURITY.md's support line (D12); uncited
      records to `docs/records/` (D13); the `embargo` columns named stable
      (N24). D9 (the dev pin to polars 2.0.x) when the canary passes.
      *Worker `task199-policy`.* *Done 2026-10-07*: on 1.34.0 the suite
      passes with 23 tests skipped by version (`needs_polars`), none failing
      in the package. REVIEW-2026-09-18 and README-ITERATIONS stay in
      `docs/` (code cites them). release.yml and the canary changed, so the
      next release rehearses first.
- [x] 200. **An integer clock column is held as an integer** -- raised
      2026-10-07 (the user: "Is the way the bank holds a clock the best? Could
      it hold different types in the same 64bit structure depending on the
      clock column type?"; then "Add it and implement"). `ClockValue` holds a
      temporal clock as integer nanoseconds (exact) and every number column as
      a double, so an integer clock is cast to `Float64` on read and is exact
      only below 2^53: an `Int64` column of epoch nanoseconds resolves to
      256 ns, and the number-clock stamps the cadences and window edges compare
      inherit it. A third, integer form, chosen from the column's dtype: steps
      in `i128`, integer stamps, every clock reader (the bank, `with_windows`,
      `refresh_time`, `embargo`, the CLI) held to it, and the frames' clock
      columns in the clock's own integer dtype. Schema 42. *Widened the same
      day (the user: "Yes"): the rule is that a difference is taken in the
      column's own type, then converted -- so `po.increment` of an integer
      input subtracts in `i128` too. Values a model multiplies (features,
      targets, weights) stay doubles: a model's first product rounds them
      anyway.* (A `relative="difference"` target was in this scope until
      task 201 removed relative targets.) *Worker `task200-int-clock`.*
      *Done 2026-10-07*: windows state 8; a clock of another kind or width
      is now refused (the bank recorded it but never refused it), and a
      `UInt64` past `i64::MAX` by row. No float field of an existing test
      moved; above 2^53 the values are now exact.
      *Follow-up 2026-10-08*: the floor canary (polars 1.34.0) failed
      `test_a_window_holds_what_polars_rolling_holds_on_the_integers` under
      `closed="left"` and `"none"`: before 1.41.1 Polars sums an empty
      window to null, where it now gives 0 (`test_windows.py`'s record).
      The test now reads an empty window as 0 on both sides; the operators
      keep null for one (`min_samples=1`, `ops.py`'s rule), where Polars
      1.41.1 and later give 0. Checked on 1.34.0 (1 and 54 nulls) and
      1.44.2, no other difference.
- [x] 201. **Relative targets are removed** -- the user, 2026-10-07 ("Since we
      now have polars expressions do we need relative columns?"; then "Yes
      and start task 201"). `po.target(..., relative_to=...)` (task 107a)
      duplicates `with_columns(ret=pl.col("p") - pl.col("mid"))`, which Polars
      computes in the columns' own type; E11 already points a row-known target
      formula to `with_columns`. Removed on every surface (Python, spec keys,
      TOML, state; the ratio's hit centre; `po.eval`'s relative scoring),
      refused by name pointing at `with_columns` or an upstream column; the
      docs recommend a log ratio or a difference for a return. Formula targets
      stay: they need the bank's release timing. Schema 43. *Worker
      `task201-no-relative`.* *Done 2026-10-07*: 30 of 30 forms and models
      matched `with_columns` to the bit before the removal, but the ratio's
      `hit_rate`. `po.target` keeps `name=`; `update_about` went back into
      `update` (the merge keeps 195's `HitTest` with `Sign`).
- [x] 202. **An insensitivity band in units of the target's own spread** --
      found by task 195, decided 2026-10-07 (the user: "follow your suggestion
      on item 2"). Task 195 measured `pa`'s `eps` (and `sgd`'s
      epsilon-insensitive `eps`) in units of the residual σ; a fit starting
      from zero coefficients learns σ from residuals as large as the target's
      level, and without decay the band never shrinks: `pa` stops learning
      (R² −52 at `half_life` 1e9, +0.954 at 500). The band is measured in the
      target's own EW spread around its EW mean instead, which the fit does not
      inflate; `huber_delta` stays on the residual σ (a clipped gradient keeps
      learning). `bocpd`'s `robust` emission is documented as not free of the
      data's units. Schema 44. *Marked for the user's review (2026-10-07).*
      *Worker `task202-band`.* *Done 2026-10-07*: `spread.rs`; `pa`'s R² at
      level 1000 without decay goes from -52.6 to +0.965; GOLDEN_PA,
      GOLDEN_PA_BOX and 12 pipeline values re-pinned against `pa_ref` to the
      bit. `eps = 0.1` of the target's spread proved wide on a high-R²
      target: task 203.
      *Follow-up 2026-10-08*: CI's Linux legs failed
      `sgd::tests::the_huber_and_squared_losses_did_not_move` (macOS
      `0.976978012937296`, Linux `0.9769780129372949`): under
      `Halflife(80)` with irregular steps each row's factor is an `exp2`,
      which glibc and Apple's libm round differently in the last bit. The
      stream now decays by a literal factor once per clock unit (`lam^1`
      is `lam` exactly), so no libm call is on its path; re-pinned.
- [x] 203. **An insensitivity band defaults to 1% of the target's spread** --
      found by task 202 (in its new unit, `eps = 0.1` is 10% of the target's
      spread and does not shrink as the fit improves: a good fit's errors fall
      inside it and the fit stops short -- slopes 1.87/1.89 against 2.0 at
      `eps = 0.2`). Decided 2026-10-07 (the user: "Mark the eps decisions for
      review and follow your reco"): the default is 0.01. *Marked for the
      user's review (2026-10-07), with task 202's unit.* *Done 2026-10-07*
      (`bd13cd9`): `stream.rs` resolves 0.01 for both; docstrings, the spec
      reference and the README say why; tests at the default on
      `y = 2x + 0.01·U(-1,1)` (0.1 ended 0.085/0.077 off, 0.01 under 0.005),
      `spec_defaults.rs`, `test_spec_defaults.py`; the golden pipeline's
      `pa`/`pa_box` (12 values) re-pinned against task 202's docstring replica
      (1.3e-15; at 0.1 it gives back the old pins). **Evidence for the
      review** (the worker's sweep, `y = 2x + N(0, (2r)²)`, excess
      out-of-sample MSE as a share of the noise variance, 0.1 / 0.01, median
      of 5 seeds):

  | R² | `pa` (`c = 1`) | `sgd`, constant rate | `sgd`, inv_scaling |
  |---|---|---|---|
  | 0.99998 | 2.86 / 0.18 | 51.2 / 0.80 | 19.1 / 0.20 |
  | 0.9998 | 2.13 / 0.59 | 3.96 / 0.35 | 2.79 / 0.11 |
  | 0.9975 | 0.18 / 1.00 | 0.045 / 0.12 | 0.015 / 0.041 |
  | 0.978 | 0.59 / 1.16 | 0.028 / 0.041 | 0.010 / 0.015 |
  | 0.80 | 0.68 / 0.81 | 0.011 / 0.012 | 0.004 / 0.005 |

  0.01 wins clearly only above R² 0.9998; between 0.97 and 0.998 `pa`
  loses about a third (task 202's level-1000 case: R² 0.9646 at 0.1,
  0.9516 at 0.01, best about 0.978) and `sgd` a little; the brief's own
  shape (noise up to 0.1) favours 0.1. Probes: the session scratchpad's
  `review4/fix2/task203-eps-default/probe2.py`, `probe3.py`. *Review 5
  (G2):* the 0.99998 cell for `pa` at 0.1 is one draw from a lottery. The
  band is 20 noise stds wide there and the fit freezes wherever it first
  lands inside it, so over 20 seeds through the bank that cell runs from
  0.58 to 278 (median 35; seeds 0-4's median is the 2.86 above), where
  0.01 gives 0.16 to 0.20.
- [x] 204. **`kalman`'s `share_p` takes each row once** -- found 2026-10-07
      while weighing task 187's leftover ("`share_p` still sequential, so
      target order matters"); decided by the user the same day ("Your reco
      all except 116"). The shared `P` took each row once per target, as if
      the targets shared their coefficients: a target beside an exact copy
      of itself moved by up to 0.92 on a spread of 1.1, and swapping two
      targets moved predictions by up to 0.36 (with a `P` per target, both
      exactly 0). `P`'s recursion never reads `y`, so targets that share `R`
      would each carry the same `P`: every target observed on a row now
      takes its gain from `P` as the row finds it, `P` takes the row once if
      any target updated, and the mean noise is summed in ascending order.
      *Done 2026-10-07*: `kalman.rs`; tests
      `share_p_a_target_beside_its_own_copy_predicts_as_it_would_alone` and
      `share_p_the_order_of_the_targets_moves_no_bit` (both fail on the old
      code); `the_filter_is_its_recursion` rewritten to the new rule. Under
      the default half-life-derived `Q` the gain is free of the noise's
      scale, so on a stationary stream `share_p` now matches a `P` per
      target (MSE 0.0897 and 4.0085 both ways) at 1/m of `P`'s cost. Moved:
      `docs/VALIDATION.md` §4's `shared_p` rows only (R² −0.054 to −0.079,
      +0.003 to −0.006). No layout change; no golden moved. Default stays
      `False`: its rows still differ from a `P` per target where the
      targets' noises move apart.
- [x] 205. **Review round 5: the changes since round 4** -- the user,
      2026-10-08 ("Review recent changes both the code and the theory";
      "Fix obvious problems that do not need me"). Seven read-only reviewers
      over `8e28c1f..9b23ae6` (tasks 183-204), each with a code lens and a
      theory lens; 42 findings, three high, every one with a probe (§19).
      *Done 2026-10-08* for the findings with a plain fix, in three worker
      branches merged as `integ6` (31 findings: C1 C2 C3 C6 C7 A4 B2 E1 E2
      C5 E7 D7 E3 E4 F4b; B1 D3 D2 D1 D5 D6; F1 F2 E5 A3 G4 A2 A5 A6 B4 E6
      F5 F6 F8 G2, G1's sentence) plus CLAUDE.md rule 5. The decisions (G1
      G3 G5 A1 B3 C4 D4 F3 F7 F9) are the user's, in §19.

**Parked by the user on 2026-09-25: integration with new libraries, Arrow,
and licensed libraries in tests.** Nothing here is to be built until the
user lifts it:
- Task 86's Arrow import (`fit_predict_capsule`, `Bank::fit_predict_stream`)
  — unblocked since 2026-09-22, waiting by choice, not on a blocker.
- The Arrow tests still open: capsules dropped unread in the leak check,
  counting batches to show a reader streams, input in several chunks.
- ARROW-SOURCES' follow-ups: pandas conformance, the SQL order guidance,
  ADBC on Postgres, DuckDB's memory growth, the unmeasured producers, and
  the DataFusion spike for the CLI.
- Licensed libraries in tests: Pathway, under the Business Source License
  (open-source test libraries are task 121).
- C8's mimalloc option (a new statically linked library, rule 12).
- Reports to other projects, which wait on the user in any case: pyarrow
  25.0.1's cast bug (pinned by a test), river's `EpsilonInsensitiveHinge`.

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

**A wide `marginal`'s shards take a batch of rows, not one (task 126),
2026-09-25.** The plan's design forked and joined the pool once per row,
on the estimate that a row at `p = 10,000` and nine targets was 2 ms of
work. Task 124 had taken that row to about 100 µs, and a rayon fork-join
costs about 10 µs here, since idle threads sleep between rows. A stand-in
for the pair loop ran at best 0.6× and 1.7× split per row, at one and nine
targets, and 3.9× and 6.5× split once per 64 rows (PERFORMANCE §25). So
the split holds rows back: every target's own numbers advance as the row
arrives, since nothing a row reports reads a pair, and each pair's share
of the row is kept until a flush steps the pairs through the held rows in
order. That is what the E73 ask itself described (`MARGINAL-AT-WIDTH.md`,
"over the chunk's rows in order"). It costs a copy of each held row's
features, 1.6 µs at 10,000, which bounds the split of the moments alone.
Borrowing the caller's rows instead would remove it, at the price of an
API that names rows by index across `online-core`'s boundary; not taken.
`"auto"` sizes from a cost model fitted to eleven shapes, up to twice the
pool's threads (ten fast cores and four slow ones here), and `shards`
stays off by default: with more groups than threads the pool is already
full. Whether `"auto"` should become the default is the user's call.

**Readiness gates (task 87), 2026-09-21.** The user's standard: a setting
rests on theory, not on a sweep, and states the intent rather than a number
that needs a formula in the user's head. So the noise gate is
`sqrt(1 + edf / n_kish)` -- the effective degrees of freedom the solve used
over Kish's effective sample size, both read from state the model already
keeps (`EwCov::n_kish`, and the diagonal of the ridged system's inverse on
the solve schedule) -- and its default is the equal-parts point, estimation
variance equal to the noise. `min_settled_frac` defaults *off* on a theory
argument, not a scar: the mean-form fit is unbiased from its first row when
the process is stationary, so the gate guards a representativeness bias only
the user can size (doc §4.1.1). `full_rank` as a boolean with a tolerance
was dropped for the exact, threshold-free `support_coef`; `G₂`, the
squared-decay Gram, was dropped because the gate's average does not need it
-- and the identity tests then measured exactly what it would buy (doc
§2.1). `min_periods` stays for every model without the statistic rather
than being aliased away: removing it would have gated `sgd` at row one.
`withheld_reason` is a dictionary-encoded Arrow array (polars reads it as a
categorical): a String column costs 16 B/row even when every value is null.

**Building the clustering (tasks 23–24), 2026-09-04.** The user chose
`kmeans` + `micro` (CLUSTERING §0's exposure), all of E36–E42, a branch
(`clustering-build`) pushed after each task for CI with `main` fast-forwarded
at the end, and a prepared 0.2.0 that they tag. Decisions taken against the
prototypes and the doc, recorded so the numbers can be re-derived:

- *Unsupervised is one thing.* `ModelKind::is_unsupervised()` (`ew_cov`,
  `kmeans`, `micro`) replaces the `ew_cov`-only exemptions: the target/feature
  leak check, the `emit_selected`/`emit_averaged` refusals, the plugin's input
  names and the expression's packing. The residual diagnostics (`emit_sigma`,
  `emit_resid_z`, `emit_metrics`, `resid_quantiles`, `emit_autocorr`,
  `emit_drift`) are **refused** for all three; `ew_cov` used to accept and
  silently ignore them (CHANGELOG).
- *Scope of `kmeans`.* Hard assignment, split–merge on a slower clock, the four
  seeding rules. The prototype's Huber weights, spherical distance, fuzzy
  memberships and stand-alone reseed rule are not built (§7 measured none of
  them earning a place). Parameters: `k`, `warm_rows` (500), `seed_rule`
  (`lloyd`; `first | farthest | kmeanspp | lloyd`), `seed` (0), `update_every`
  (1), `split_merge` (0.5), `split_merge_every` (100), `dead_frac` (0.05),
  `standardize` (true, a metric — never the coordinates, §10).
- *One accumulator, always in mean form.* A cluster is `(n, c, R)`; rows since
  the last checkpoint accumulate into a per-cluster **batch** summary of the
  same shape (`W`, mean of `z`, mean of `d²`), and the checkpoint merges batch
  into cluster: `n' = n + W`, `c' = c + (W/n')(z̄ − c)`, `R' = R + (W/n')(d̄² − R)`.
  With `update_every = 1` the batch is one row and this *is* MacQueen's step;
  no sum ever exceeds the largest input, so the bound rows of the contract
  test cannot overflow it (the prototype's `(n·C + S)/(n + W)` can). `R` for
  `kmeans` is the EW mean of each row's squared distance to the centre it
  was assigned to, *at assignment* — out-of-sample, like `sigma`. For
  `micro` it is Welford's centred radius², DenStream's definition.
- *The metric.* Diagonal EW moments (`FeatureMoments`, O(p)) rather than the
  full `EwCov`: `mw_i = 1/v_i` where `v_i > 0` and finite, else 1, read from
  the moments *before* the row. Distances are `Σ mw_i (x_i − c_i)²`; a row at
  the input bound against a variance at the opposite scale gives `d² = ∞`
  rather than NaN, and an infinite `d²` is not learned into `R` (the centre
  still moves; the radius learns nothing from a row it cannot measure).
- *Seeding.* Buffer `warm_rows` rows (the buffer is capped at
  `max(warm_rows, 1000)`, where duplicates are allowed as seeds); every buffered
  weight is multiplied by each row's `lam` (the product form, exact in both
  implementations; the prototype's `exp(L − L_row)` agrees to ~1e-9 at a
  finite halflife). `kmeanspp`/`lloyd` draw from **splitmix64** seeded by
  `seed`, `u = (x >> 11)·2⁻⁵³`, weighted choice = first index whose cumulative
  weight exceeds `u·total`, uniform `⌊u·n⌋` when the weights sum to zero; Lloyd
  is 10 weighted iterations, first minimum wins, a cluster with no weight
  keeps its centre. The same generator is written out in
  `tests/reference_cluster.py`, so the Python reference is bit-exact.
- *The far row (final design below, 2026-09-05).* A check runs every
  `split_merge_every` learned rows at a checkpoint: merge the closest pair when
  `d_ij / (r_i + r_j) < split_merge` and re-place the freed centre on the
  heaviest far summary; **else** if the lightest cluster is dead
  (`n_j < dead_frac · n_eff / k`) re-place it the same way. The dead rule is
  what recovers a centre parked at the input bound with nothing to win; the
  prototype's ratio condition (`reseed_factor`) never fires on a single
  uniform blob and was dropped. `k ≥ 3` for the merge, as measured.
- *`micro` decides at unit weight.* The absorption test (merged radius ≤ `eps`)
  is made with weight 1 whatever the row's weight, and the update then applies
  the weight. `predict` has no weight, so this is what makes `predict` the
  step without the step for a model whose label *is* the decision; a heavy row
  can push a radius past `eps` once, and the next rows see the larger radius.
- *`micro`'s threshold.* `macro_link = None` derives the linkage threshold at
  every checkpoint as 1.5× the p90 (nearest rank) of the nearest-neighbour
  spacing among the potential micro-clusters — §6.5's rule; a value is
  `macro_link · eps`, the prototype's constant. Default `eps` is `0.4·√p` in
  the standardized metric (§7.8: the clean-mixture setting; shapes need
  0.07–0.1·√p and a larger `max_clusters`). `beta_mu` 3, `max_clusters` 200,
  `prune_every` 100. Age decay per micro-cluster is a product of `lam`s.
- *Outputs.* `kmeans`: `cluster` (i32, null before seeding or under
  `min_periods`), `dist`, `dist2` (second-nearest; null at `k = 1`), `n_eff`,
  `coef` = the `k × p` centres flat, named `coef_cluster{j}_{feature}` by
  `coef_fields`. `micro`: `cluster` (i32 macro label of the nearest potential
  micro-cluster), `dist`, `micro` (i32, the id the row would join, or the id a
  new one would get), `outlier` (bool), `n_clusters` (i32), `n_micro` (i32),
  `n_eff`, `coef` = per potential micro-cluster in id order
  `[id, label, weight, radius, centre…]` — ragged, so `coef_fields` is empty
  and `coef_index` refuses it as it does `ew_cov`. Two new `Source`s carry the
  integer and boolean columns out of the `pred` buffer.
- *Contract tests.* `kmeans`/`micro` get the shared probe, the `PROBED` entry,
  a golden signature, and a recovery criterion of their own under the names
  the parity scanner requires (`Recovery::Fit`): after the bound rows, the
  tail's outputs are finite and its mean `dist²` is within the tolerance of
  a twin that never saw them — 1e-6 as the margin, measured 2.9e-15
  (standardized) and 0 (raw).

**Task 23's deep testing: what the split–merge move can and cannot repair,
2026-09-05.** Every claim below was measured through the Rust bank on
`scripts/clustering_experiments.py`'s fixtures (N = 20000, p = 4, k = 5,
halflife 3000, 20 seeds, last-quarter ARI) and the stranded fixture in
`tests/test_kmeans.py`; the numbers are what the tests pin.

- *The doc's regime claim was a seeding artefact.* CLUSTERING §7.6 reported
  "split–merge recovers a regime change in 1500 rows" because the prototype's
  one k-means++ start had put two seeds in one blob, which the merge then
  freed. With `lloyd` seeding the plain model scores 1.000 / 1.000 / 0.926 /
  1.000 over the four segments of that fixture with no move at all. The real
  case is a *stranded* centre: a blob dies and another is born far from every
  centre (fixture `stranded`, 4 blobs, halflife 1000). The move recovers it
  to a tail ARI of 1.000 against 0.71–0.73 without.
- *Far rows are summarised, never learned.* A row is far when `d² > f · R̃`,
  `f = 1 + FAR_SIGMAS · sqrt(2/p)` (`FAR_SIGMAS = 4`: `d²/R̃ ~ χ²_p/p` has sd
  `sqrt(2/p)`; a Gaussian blob crosses the cut 0.7% of the time in 2-D, 0.4%
  in 4-D), `R̃` the mean of the trusted radii (`RADIUS_ROWS = 10` learned rows
  and `r2 > 0`) leaving out the largest, or that one alone; with no trusted
  radius nothing is far (or everything would be). A far row goes to its
  cluster's far summary (Welford) and nowhere else — not the centre, the
  radius or the weight. Two designs in between failed the regime unit test:
  far rows moving the centre but not the radius drag a centre off its own
  blob and break the closest-pair ratio; far rows counting in `n` but not
  the centre keep a jumped blob's centre alive forever.
- *The winsorized radius is what makes the cut safe.* At each check
  `r2_j ← (n_j r2_j + F_j · cut) / (n_j + F_j)`: far rows count in the
  radius as if they sat at the cut. A burst of outliers widens it a little
  (steady state ≈ 1.17× at 5% far); a cluster whose rows are all far widens
  by `(n + f F)/(n + F)` per check — the contract test's bound rows had left
  the cut at 1.3e-149 with every radius at 2.7e-150 and `n_dead = 406`, a
  trap that never opened until this. The per-cluster ratchet cut tried
  before it was dropped: it blocks the wide-cluster split a freed centre
  needs.
- *Where a freed centre goes.* The heaviest far summary's mean, with the
  typical radius `R̃` and half the source's weight (a newborn with the far
  weight dies at the next check at `dead_frac = 0.25`). A merge, which costs
  a live cluster, is gated: the source must hold at least `FAR_ROWS = 3` rows
  weighing `FAR_SHARE = 5%` of the window's learned weight `V`, the pair's own
  summaries pooled. Without the gates newborns placed on 1–9 outliers with
  `r2` 10–30 poisoned the ratio and merged real blobs (min ARI 0.051 on the
  outlier fixture; now 0.768, the same symmetric split-blob miss as without
  outliers). A dead centre, already lost, takes any far rows, its own
  included.
- *Seeding trims by the same rule.* Rows whose `d²` to the EW mean exceeds
  `f` times the buffer's weighted mean `d²` do not choose the seeds (the
  whole buffer does when the rest cannot give `k` distinct seeds) and are
  replayed as far rows. Outliers 5% + drift: 0.984 mean ARI, the plain 0.749
  before.
- *Latency, measured.* A stranded centre is re-placed `log2(1/dead_frac)`
  halflives after its blob vanished: 4500 rows at the default 0.05 and
  halflife 1000 (formula 4320), 2000–2500 at 0.25 (formula 2000); at
  halflife 3000 the default does not fire within the 10000 rows the regime
  fixture leaves (0.784, 18/20 misses, against 0.824 without the move, whose
  nearest centre at least drifts toward the new blob — far rows do not drag)
  while 0.25 scores 0.933 with 1/20 misses. The price of `dead_frac`: a blob
  lighter than `dead_frac / k` of the stream loses its centre whenever any
  row is far. The default stays 0.05; the README says when to raise it.
  With `k = 1`, or when every cluster sees far rows, the typical radius
  widens with the cut and the rows are learned again after
  `log(D²/r2) / log((n + f F)/(n + F))` checks (659 rows for a 20-sd jump
  at halflife 200). One jumped blob among `k ≥ 2` waits for the dead rule:
  its cluster's widening radius is the largest, which `R̃` leaves out.
- *Blind spot.* A cluster owning two blobs: all its rows are within its own
  radius, and their far mean is its own centre. `lloyd` seeding is what
  prevents it; the drift fixture's 1/20 miss is such a pair at ratio ≈ 0.6
  (> 0.5 by design: a legitimate wide cluster looks the same).
- *Rejected on measurement, not to be retried:* D²-weighted reservoir of far
  rows (outlier-prone, random); the max-ratio far row (picks outliers by
  construction); a recent-share dead test over `split_merge_every` windows (a second
  time scale; kills quiet clusters; outlier trickles defeat ratio tests);
  ISODATA per-feature variance split (k·p state, blind to bimodality in
  general position); in-place coincident split as a trigger (splits wide
  clusters); self re-placement when far weight exceeds own intake
  (ping-pong). A possible follow-on, not built: a cohesion-gated fast
  re-placement (a far summary whose `r2` is blob-like against `R̃`) would cut
  the stranded latency from halflives to one check.
- *Oracle.* `tests/reference_cluster.py` mirrors every operation in order and
  the 22 oracle tests are bit-exact; 61 kmeans tests in all.

**Task 24's decisions: what `micro` admits, links and prunes, 2026-09-05.**
Measured through the oracle (`tests/reference_cluster.py`, n = 6000, halflife
3000, `prune_every` 100) on the shapes of `scripts/clustering_experiments.py`,
then through the Rust bank at 20k–200k rows; ARI against the truth, DBSCAN on
a 3000-row sample as the ceiling. What the tests pin is what is written here.

- *The label and the id are two columns.* §6.5's rule 2 ("a row's label is
  the id it would be absorbed by") is kept for `micro`; `cluster` is the macro
  label of the nearest *potential* summary, null while there is none, so a
  row that opens a summary still reads the cluster it sits next to. The
  doc's ARI complaint about variable-`k` (purity 1, ARI 0.4–0.8) was about
  ids; on labels the built model scores 1.000 where DBSCAN does.
- *The admission rule is DenStream's, at unit weight, with a capped radius.*
  A row is admitted where a unit row would be (`merged_radius2(λn, r2, d2, 1)
  ≤ E`, `E = eps²p`), potential summaries first, then outlier ones, else it
  opens a summary; it is then absorbed with its full weight and `r2 ← min(r2,
  E)`. Without the cap a heavy row overshoots the bound and the summary
  admits nothing — not even a row at its centre — until decay brings `n`
  under `E/(r2 − E)`, halflives later; capped, it is merely full. The
  alternative measured and rejected, admission by distance to the centre
  (`d2 ≤ E`, weight-independent): ARI radius / distance — moons .05
  0.999/0.670, .07 0.999/0.997, .1 1.000/0.999; rings .05 0.788/0.512, .07
  1.000/0.741, .1 1.000/0.834, .14 0.000/1.000; varied .05 0.866/0.700, .07
  0.950/0.790, .1 0.972/0.934, .14 0.088/0.559; highdim20 .2 0.875/null, .25
  1.000/1.000, .4 0.000/1.000. The distance rule fragments at the working
  `eps` (twice the live summaries, 3–5× the outlier share) and only wins
  where the radius rule has already bridged. `predict` decides with
  `factor(d_clock)` applied to every `n`, bit-exact with `step`.
- *The derived link.* `L = max(2·eps√p, 1.5 × p90 of the nearest-neighbour
  spacing among potential summaries)`, nearest rank, recomputed at every
  checkpoint; `macro_link` given makes it `macro_link·eps√p` (0 links
  nothing, 2 only summaries that touch). A sweep of (quantile, factor) put
  q0.9 / 1.5 best overall: moons 0.996–1.000 at eps .05–.14 (ceiling
  0.999); rings 0.787 / 1.000 / 1.000 / 0.000 at .05 / .07 / .1 / .14 (the
  ring gap is 3.55·eps√p, under `L` = 4.55 at .14); varied 0.86 / 0.95 /
  0.972 / 0.088 (the varied-density limit of one global threshold: `L` 4.0
  > gap 2.9 at .14); aniso ≈ 0.57 (ceiling 0.68, not density-separable);
  highdim20 null at eps ≤ .14, 0.997 / 1.000 at .25 / .3 (`L` = the floor),
  0.000 at .4 (one summary per cluster). A promoted summary attaches to the
  nearest potential one within `L` at once (attach-on-promotion), so a new
  cluster has a label before its first checkpoint.
- *Pruning is DenStream's ξ, on a learned-row schedule, with no grace.* An
  outlier summary is dropped at a checkpoint when `n < ξ(age) = (λ^age λ^Tp −
  1)/(λ^Tp − 1)`, `Tp = ⌈h log2(β/(β − 1))⌉`: ξ = 1 at birth and rises to
  β, so a lone-row summary is dropped at the first checkpoint after the one
  it was born on (born on the checkpoint row itself it survives that one:
  age 0, weight 1 not below 1). A potential summary is dropped under β and
  lingers `h log2(n₀/β)` after its rows stop. Zero-weight rows do not count
  toward a checkpoint. `prune_every` 25 fragments sparse shapes (rings@.1 →
  0.059) because a one-row summary never sees a second checkpoint; a
  one-checkpoint grace repaired that and bridged `varied` (0.972 → 0.771) —
  no grace, default 100. An infinite halflife prunes nothing; only the cap
  applies, evicting the lightest outlier summary, else the lightest
  potential one.
- *Two failure regimes, both readable off the outputs, neither guarded.*
  `eps` too small: every row an `outlier`, `cluster` null, `n_micro`
  cycling — no summary reaches β before ξ takes it. `eps` too coarse for
  the derived link: `n_micro ≈ k`, each cluster one summary, the p90 spacing
  *is* the inter-cluster spacing and everything bridges into one cluster
  (4-D blobs sd 0.6 at eps .3, constant-feature fixture at .1, highdim20 at
  .4). A regime guard (refuse to link when `n_micro` is small) was measured
  and rejected: it fragments the shapes the link exists for. Rule of thumb,
  in the README: `eps` ≈ the within-cluster spread per standardized
  coordinate, 0.07 for 2-D shapes, 0.3 for separated Gaussians in 20-D.
- *At scale (Rust bank, halflife 3000).* 20k rows: moons .07 → 1.000, rings
  .1 → 1.000, varied .1 → 0.984 (sample ceiling 0.956), highdim20 .3 →
  1.000 (0.728). 200k rows of 4-D blobs, halflife 20000: eps .2 / .25 →
  1.000 / 0.999 with ≤ 200 live summaries. Stranded fixture, halflife 1000:
  the newborn cluster is labelled 31 rows after its first row, the dead one
  lingers 5.5 halflives (`h log2(n₀/β)`), then `n_clusters` returns to 4;
  tail ARI 1.0. 5% uniform noise at eps .07: 94% of noise rows flagged, 0.3%
  of real rows, ARI on the real rows 1.0; at .14 the noise bridges the
  clusters. sd-0.8 blobs in 4-D: .25 → 0.995, but .2 and .3 → four clusters
  (two bridged) — the working band narrows as clusters approach.
- *Oracle.* `reference_cluster.py`'s `Micro` mirrors every operation in
  order; 17 bit-exact cases; 63 micro tests in all.

**Task 25's decisions: the conformal recursion, 2026-09-05.**

- *The rule.* Per slot, `q` is a tracked quantile of the conformity score
  `s = |resid|`: read `lo = pred − q`, `hi = pred + q` before the row, then
  `err = 1{s > q}`, `q ← max(0, q + η·w·(err − α))`, `α = 1 − coverage`. This
  is the P step of Angelopoulos, Candès & Tibshirani (2023) applied to the
  score's quantile, i.e. online gradient descent on the pinball loss, and it
  telescopes: for scores in `[0, B]`, `|Σ η_t (err_t − α)| ≤ B + max η`
  (the clamp at zero can only add coverage), so the average miss rate tends
  to `α` at rate `1/T` on *any* residual sequence. No distribution, no
  stationarity, no split: the score comes from a model that has not seen the
  row, which is the property the library already guarantees everywhere.
- *The step is in sigma units.* `η_t = conformal_rate · sigma_t`, the slot's
  EW residual standard deviation before the row. A fixed `η` would need a
  scale from the user; scaling by `sigma` makes 0.05 a sensible default on
  every stream and lets the radius follow a scale shift at the speed `sigma`
  does. The bound holds in σ-weighted form with `η_t` in place of `η`. With no
  usable `sigma` (`emit_sigma` is not required: the tracker reads the
  internal one, which exists for every regression model) the step is 0 and
  the radius holds.
- *The warm start.* `q` is undefined until the first scored row that has a
  finite positive `sigma`; then `q = sigma · Φ⁻¹(1 − α/2)`, the Gaussian
  radius, and that row is not scored. `Φ⁻¹` is Acklam's rational
  approximation evaluated in a fixed order, so the Python replay in
  `tests/test_conformal.py` is bit-exact. The alternative, `q = 0` and let it
  grow, wastes `B/η` rows widening from nothing; starting at the Gaussian
  radius is right on Gaussian residuals and a few steps off otherwise.
- *`coverage` is the EW hit rate* on the model's own clock (`cov_w` decays
  by `lam`, a row adds `w`), read before the row, so it says what the
  interval has delivered recently, not over all time. A null target or a
  zero-weight row ages it and moves nothing else; the warm-start row is not
  counted.
- *Measured (200k rows, halflife 2000, rate 0.05).* Coverage at target
  ±0.01 on Gaussian, t(2.5), `exp(x₁)·N(0,1)` and slope-flip + noise-×3
  residuals; the Gaussian `pred ± 1.645·sigma` covers 0.942–0.951 on the
  last three. The radius follows the noise ×3 shift to within 10% of the
  new Gaussian radius. Levels 0.5, 0.8, 0.99 are met within 0.012 on fat
  tails. Cost: three f64s of output and five of state per slot; no
  measurable throughput change on `ewridge`.

**Task 26's decisions: `mahal` and EW-PCA on `ew_cov`, 2026-09-05.**

- *`mahal` is a distance, in σ units.* ENHANCEMENTS E37 sketched the
  quadratic form `δᵀ Σ⁻¹ δ`; the field is its square root, so at k = 1 it is
  `|z|` and it reads like `resid_z` — the feature-side twin it was proposed
  as. `mahal²` is then χ²_k on Gaussian columns; the README says so and
  `tests/test_ew_cov_scores.py` holds it at 200k rows. The matrix solved is
  `C + s·prior·I`, the same fading ridge `partial_corr` reads, hence the
  `precision_prior` requirement; NaN until the prior is set, before
  `min_periods`, or when the solve fails. One `solve_spd` per row, not a
  tracked inverse (the E2 finding still stands: a tracked inverse cancels to
  zero under a dominant row and never recovers).
- *`mahal_quantiles` are unweighted P² trackers* of the emitted score, like
  `resid_quantiles`: a zero-weight row still adds its score, a `predict`
  pass does not, and they are read before the row. They lag the score by
  the five rows P² needs to start.
- *The PCA refresh runs after the row's update*, not before it, so
  `predict` (no update) and `step` read the same frozen loadings and the
  score `Σ v_j,i (x_i − m_i)` uses the live mean with loadings from the
  last refresh. The first refresh waits for `min_periods`; then every
  `pca_every` learned rows. A zero-weight row advances the cadence counter
  like any other learned row, since it advances the clock.
- *Signs follow the previous refresh.* E38 proposed largest-magnitude-entry
  positive. On `gaussian(k=5, seed 10)` that rule flipped `pc1` between two
  refreshes (dot with the previous loading −0.99999) when two loadings of
  near-equal size traded the lead. The rule built: sign each new loading so
  `v_new · v_old ≥ 0` with the previous refresh's loading for that
  component; fall back to largest-entry-positive when there is no previous
  one or the dot is exactly 0. The Python oracle carries the same rule
  (`pca_oracle(c, r, prev)`), and a test proves the max-abs rule would have
  flipped on the same stream.
- *`pc<j>_share`, not `explained`.* One number per emitted component (its
  eigenvalue over the trace) rather than the `k`-vector E38 sketched; the
  trace is the sum of the diagonal, so the shares of the emitted components
  need not sum to 1. NaN when the trace is ≤ 0 (a constant stream).
- *Measured (200k rows, Mrows/s, k = 4 / 8 / 16 / 32).* `mean,std,corr`
  7.44 / 2.85 / 0.86 / 0.21; `+ mahal` 3.86 / 2.21 / 1.07 / 0.35; `+ 2
  quantiles` 3.32 / 2.03 / 1.03 / 0.34; `pca=2, pca_every=1` 0.98 / 0.35 /
  0.10 / 0.03; `pca_every=100` 6.52 / 3.74 / 1.92 / 0.78; `partial_corr`
  2.92 / 1.21 / 0.39 / 0.09. The eigendecomposition is ≈ 1 µs at k = 4 and
  ≈ 30 µs at k = 32, which is what `pca_every` amortizes; `mahal`'s solve is
  cheaper than `partial_corr`'s because it is one right-hand side.


**Task 27's decisions: `ew_class`, 2026-09-05.**

- *A kind of its own, not an `ew_cov` option.* ENHANCEMENTS E39 sketched a
  class-conditional `ew_cov`. Built as `ew_class` because nothing of
  `ew_cov`'s surface carries over: no `stats`, no pairs, a label column
  instead of a target and a String output. It reuses `EwCov` whole — one per
  class, `with_precision_prior` — and the covariance shapes are views of the
  same state: `full` reads each class's `C_c + r_c I`, `shared` pools them
  by the class weights `Σ π_c (C_c + r_c I)` (one factorization, all the
  quadratic forms at once through `quad_forms_logdet`), `diagonal` reads
  the variances, clamped at 0. The ridge `r_c = precision_prior ·
  precision_scale_c` fades per class as `partial_corr`'s does, so a class
  is scoreable from its first row and the prior is gone once it has data.
- *`n_eff` counts every accepted row, labelled or not.* It is the stream's
  weight, the quantity `min_periods` compares against everywhere else; the
  class weights `n_c` are the labelled weight per class and a class's share
  `π_c` is read off them. A row before `min_periods`, or before any class
  has been seen, is null. Hard rule 8 holds: `n_eff` is reported before the
  row's own update and decay.
- *A null label scores and does not learn; an undeclared one is an error.*
  Null is the late-label case (score now, learn when the label comes back
  in a later stream) and mirrors a null target. A value not in `classes` is
  neither a class nor "unknown": a static schema needs the class set
  declared, and silently dropping the row would hide a typo, so the bank
  raises with the row, the value and the class list, and says to null the
  rows that should only be scored. The column is read as a key (cast to
  String, like `group`), which is what makes integer, boolean and
  categorical label columns work through their text.
- *An unseen class has `p = 0` exactly and null means.* Its log-likelihood
  is −∞, so the softmax gives exactly 0 rather than a tiny positive number,
  and it is never the argmax; its `coef` entries are null (the list builder
  and `ModelBank.coef` both map NaN to null, which every `coef` list now
  honours: finite or null, inside the list too). The first maximum wins a
  tie, so two classes with identical states resolve to the first declared.
- *Plumbing predicates.* `is_unsupervised()` (ew_cov, kmeans, micro: no
  target column at all — the leak-check exemption, the expression's input
  packing) is now distinct from `predicts_no_target()` (those plus
  ew_class: no residual, so every residual diagnostic is refused by name and
  the per-model slot count comes from the schema). `ew_class`'s label
  travels as `targets[0]`, so every place that reads the target column by
  name — `keep_columns`, the projection the lazy source pushes, the
  expression's packing — works unchanged.
- *Fields.* `class`, `p_<class>`, `n_eff`, `coef`, with the halflife suffix
  after each (`class@h50`, `p_a@h50`); `output_index` kinds `class` (dtype
  `str`) and `p`; `coef_fields` one slot per class named by the class, so
  `coef_index`'s `target` column is the class and `term` the feature.
- *Measured (200k rows, 3 classes, Mrows/s, k = 2 / 4 / 8 / 16 / 32).*
  `full` 1.63 / 1.33 / 0.78 / 0.38 / 0.13; `shared` 3.22 / 2.55 / 1.58 /
  0.81 / 0.28; `diagonal` 6.92 / 5.88 / 4.51 / 2.84 / 1.57. `full` pays `C`
  Cholesky factorizations per row, `shared` one, `diagonal` none; the update
  itself is `ew_cov`'s O(k²) on one class and an O(1) decay on the others.

**Task 28's decisions: constrained coefficients, 2026-09-05.**

- *One projection, not two.* ENHANCEMENTS E40 sketched a box (clamp) and a
  simplex (the sorting algorithm). Built as one operator over
  `{lo ≤ b ≤ hi, Σb = s}` with any of the three optional, because the
  portfolio ask is usually all three at once (long-only, capped per name,
  fully invested) and the sort covers only `lo = 0, hi = ∞, s = 1`. The
  Lagrangian `b_i(μ) = clamp(v_i − μ, lo_i, hi_i)` has a sum that is
  piecewise linear and non-increasing in `μ`; the root lies between two of
  the `2k` breakpoints `v_i − hi_i`, `v_i − lo_i`, found by a binary search
  over the sorted breakpoints and one linear solve on the segment. `O(k)`
  for a box alone (no sort), `O(k log k)` with a sum. `constraint.rs` checks
  it against the sort formula on the simplex and against the KKT
  conditions on random boxes-with-a-sum; the Python side checks it against
  a bisection that shares no code.
- *Where the projection runs under `scale_features`.* The bound is a
  promise about the coefficient the caller reads, `c_i = b_i / scale_i`, so
  in the standardized space it is `b_i ∈ [lo_i·scale_i, hi_i·scale_i]` and
  the sum is `Σ b_i / scale_i = s`. The nearest point in the *standardized*
  metric (the metric the gradient step is taken in) is the box-with-a-sum
  projection with weights `a_i = 1/scale_i` — the same breakpoint search
  with `a_i` in the sums. So `sgd` projects `beta` in place with the
  scales, not the reported coefficients. Consequence, measured: with
  features a million apart and a sum on the caller's coefficients, the sum
  goes to the coefficient whose unit is cheap in the standardized metric,
  and the well-determined slope keeps its truth — not the corner a
  clamp-then-renormalize would give.
- *When.* `pa` projects right after each target's update, inside the row.
  `sgd` projects at the end of `step`, after the scaler has moved, for the
  targets that learned this row — or every target when the scales moved
  (a weight > 0 with `scale_features`), because a moved scale changes what
  the stored `beta` means in the caller's units even for a target that saw
  a null. `predict` never projects; a zero-weight or null-target row moves
  nothing. The initial zero is projected in `new` so the first prediction
  and the first `coef` are feasible: uniform weights on a simplex, the
  nearest corner of a box that excludes zero.
- *`pa` under a constraint.* Its step is the smallest change that meets
  the row's margin; the projection then takes part of it back, so the
  margin is not met and a truth outside the set is never reached (a wall
  is approached, not sat on, as `pa.rs` measures: 4,000 of 5,000 rows
  touching it). Documented as "keep `c` small"; not fixed by projecting
  inside the step, which would be a different (constrained-QP) update.
- *Refusals.* Lengths by name, NaN and the wrong infinity by index (`+inf`
  as a floor or `−inf` as a cap pins nothing and means a typo), a floor
  above a cap, a non-finite sum, and a sum outside `[Σlo, Σhi]` — with
  rounding slack of `1e-12` relative, so `[0.1, 0.2, 0.3]` (which sums to
  `0.6000000000000001`) accepts a sum of `0.6` and projects onto the
  floors. Python's `_INF_OK` admits `inf` for `coef_min`/`coef_max` and
  refuses it for `coef_sum`, matching the Rust parser
  (`tests/test_error_messages.py`).
- *Measured (200k rows, one bank, Mrows/s, k = 2 / 4 / 8 / 16 / 32).*
  `sgd` 22.2 / 20.2 / 17.8 / 13.0 / 7.4; with a box 21.0 / 18.7 / 15.6 /
  11.4 / 6.5; on the simplex 17.0 / 12.8 / 8.2 / 3.9 / 1.7; the simplex
  under `scale_features` 8.5 / 6.5 / 4.6 / 2.5 / 1.1. `pa` 22.2 / 20.2 /
  19.5 / 14.5 / 10.0; on the simplex 19.3 / 15.9 / 12.6 / 7.0 / 3.7. The
  box is `k` clamps (5–12%); the sum is the sort and `log2(2k)` sweeps of
  `k` clamps, about 0.4 µs a row at `k = 32`, which is what `O(k log k)`
  costs and is unchanged by an unstable sort. A first cut allocated a
  `learned` flag per target per row and cost the box 28% at `k = 2`; it
  is a `#[serde(skip)]` buffer now. The `O(k)`-expected simplex
  projections (Condat 2016) would be the next step if a bank ever ran
  hundreds of constrained slopes; none does.

**Task 29's decisions: coefficient reversion, 2026-09-05.**

- *Where the transition sits.* At the top of `step`, before the row's
  prediction and before the process noise, so `P ← ΦPΦ + Q·d` is the
  textbook order and the prediction a row sees is the coefficient after
  the decay over the gap since the last accepted row. That makes a
  null-target or zero-weight row advance it too — it is clock, like the
  `Q·d` it precedes — and it makes `predict(x, d)` a closed form,
  `Σ z_i·(β_i·φ_i(d))`, bit-identical to the step because `β·φ == φ·β`;
  the contract test (`predict_is_the_step_without_the_step`) runs with
  and without reversion. The bank's `predict` already passes the distance
  to the last learned row, capped by `max_dclock`, for `holt`; `kalman`
  now reads it.
- *Toward zero, not toward a prior.* ENHANCEMENTS E41 offered both. Zero
  in the standardized coordinates is the one target that means the same
  thing at every row — "no effect" for a slope, "the target averages zero"
  for the intercept — while a fixed caller-unit prior would be a moving
  target as the scaler moves, and `ewridge`'s E15 warm prior solves from
  accumulated statistics the filter does not keep. No `coef_prior` on `kalman`.
- *A scalar broadcasts to every slot, intercept included*, as
  `coef_halflife` does; `[inf, r, r]` exempts it. Nothing is applied when
  no slot is finite, so the default is bit-identical to the previous
  filter (checked against HEAD in a worktree, and by the golden bank).
- *Prior variance.* A reverting slot's stationary variance is
  `q_i·d/(1−φ_i²)` instead of unbounded; measured after 200k null-target
  rows, the gain of the next update matches that to 1e-6 relative (and the
  random walk's `n·q` to 1e-3).
- *Refusals.* Length 1 or `k_total`; NaN, zero and negative refused (`inf`
  is the walk); Python's `_INF_OK` admits `inf`; `rls`/`ewridge` do not
  take the argument.
- *Measured.* 300k rows, exact-Bayes process noise (`q = [0, 1−φ², σ_w²]`,
  `obs_var = σ²`, `standardize=False`): a slope active 2% of the time is
  tracked at 0.51× the random walk's error (0.86× dense at `H = 20`,
  `σ = 2`; 0.93× at `H = 40`, `σ = 1.5`); a slot left at `inf` within 0.2%
  of the unmodified filter; a random-walk truth under a reverting filter
  (`r = 30`) more than 2× worse. A first cut with `coef_halflife=40` and
  the default process noise had the reverting filter *worse* (0.72 vs
  0.53): `q` was far below the truth's innovation variance and the shrink
  dominated, so the test was rewritten around the exact-Bayes noise — the
  prior has to match the world, and the docstring says so. Throughput
  (200k rows, one bank, `k = 2 / 4 / 8 / 16 / 32`): the transition is `k²`
  multiplies on a `~3k²` update, 8 / 14 / 16 / 15 / 14 % slower, nothing
  when off. The step's pre-existing per-row `scales()` and `q_vec()`
  allocations (one `Vec` each, per target for `q_vec`) are the first thing
  task 31 should remove.

**Task 30's decisions: `seqtest`, 2026-09-05.**

- *A model, not a diagnostic flag.* ENHANCEMENTS E42 asked for a test
  between two specs. Built as a spec kind (`type = "seqtest"`) rather than
  an `emit_*` flag on the sides, because the test has its own targets (the
  residual fields it compares), its own warm-up and reset policy, its own
  state to save, and a second use the flag could never have: on its own it
  tests the sign of any column. `targets` name the columns (column mode) or
  the residual suffixes `t` of `resid_<t>` (compare mode).
- *The bet.* Per target two one-sided e-processes, `E⁺` for "positive" and
  `E⁻` for "negative", each `E ← E·(1 + λ s)` with the Krichevsky–Trofimov
  stake on the counts *before* the row, `λ⁺ = max(0, (n⁺ − n⁻)/(n + 1))` —
  the Beta(½,½) posterior mean `2p̂ − 1`, clipped at zero so a side never
  bets against its own lead. Two consequences the tests pin: the losing
  side's `log_e` is *exactly* 0 while it has never led (assert `<= 0`, not
  `< 0`), and `1 − λ ≥ 1/(n + 1)` bounds a lost bet at `ln(n + 1)`, so the
  wealth is finite forever. Where the clip never binds the wealth is the KT
  mixture `2ⁿ B(n⁺ + ½, n⁻ + ½)/π` in closed form; a `+ + −` repeating
  pattern keeps `n⁺ ≥ n⁻` on every prefix, so its `+` side is the exact
  Beta integral and its `−` side is 0 — the oracle for the recursion, held
  through `math.lgamma` (no scipy). The two-sided e-value is `(E⁺ + E⁻)/2`,
  emitted as its parts. Not the Waudby-Smith–Ramdas aGRAPA stake: KT is
  parameter-free, has the closed form, and its regret to the best fixed
  stake is `½ ln n + O(1)`, which is what the power measurements show.
- *A trial is a row.* Ties (zero), nulls, NaN and beyond-bound values bet
  nothing and count nothing, but the row is seen (`n_eff` advances, so
  `min_periods` means what it means everywhere). No `weight` (the stake is
  a function of counts; a weighted bettor's e-process is not the KT
  mixture) and no `halflife`/`lam` (a product of bets cannot decay and stay
  an e-value); both refused by name, as are `features` and every residual
  diagnostic (`predicts_no_target`). `session_gap="reset"` and
  `on_clock_reset="reset_state"` restart the test; a clock otherwise only
  admits or refuses rows.
- *Compare mode is column mode on `|resid_b| − |resid_a|`*, positive when
  `a` came closer — the same sign under any loss that grows with `|resid|`
  (squared, absolute, Huber), so the test is of "`a` beats `b`" under all of
  them at once. A row where either side is null (warm-up, a skipped row) is
  a tie. `a_suffix`/`b_suffix` pick a grid instance (`resid_y__r0.5@h20`);
  one spec against itself is refused only with equal suffixes, because two
  instances of one grid are a comparison like any other. Tested by
  computing the difference as a column and running column mode on it: the
  fields agree bit for bit, including `n_eff`.
- *The two-phase bank.* `fit_predict` runs `check_clock` on every stream
  first (so a refused chunk still updates nothing, in either phase), then
  every non-comparison spec in parallel as before into `out:
  Vec<Option<Column>>`, then each comparison reading its two sides' residual
  fields from the structs in `out` (`compare_targets`), and returns the
  columns in spec order whichever phase produced them. A comparison
  therefore never needs the sides to be listed before it; `resolve_compare`
  at `Bank::new` refuses a side that is not in the bank, is itself a
  `seqtest`, or lacks the residual field, naming the fields it has.
  `predict_on_pool` does the same with the scored sides, so scoring reads
  the state before the chunk and compares what it scored. The comparison's
  own `group` is independent of the sides' (grouped sides can be compared
  pooled).
- *`RunConfig::validate` builds the bank.* The CLI's `--dry-run` used to
  validate specs one by one and would have said "config OK" to a
  comparison whose side was not in the config; it now runs `Bank::new` and
  reports duplicate names and comparison mismatches before opening the
  input.
- *The expression form.* `pl.col("d").online.seqtest()` is column mode on
  its column (`extra_targets` for more); `a`/`b`/`a_suffix`/`b_suffix` are
  omitted from `SeqTestKwargs` (`EXPR_OMITS` in the typing test) and
  refused at runtime with the way to write it: the difference as a column,
  or the spec in a bank. The kwargs TypedDict says so in its docstring.
- *`po.eval.seqtest`* mirrors the builder's keyword shape (`targets`, `a`,
  `b`, `a_suffix`, `b_suffix`, `by`, `min_periods`) and is the same
  recursion in polars expressions (`cum_sum().shift(1)` under `over`),
  bit-identical to the bank in both modes. Two gotchas it absorbs: polars
  orders NaN above every float, so `NaN > 0` is `True` and the sign must
  be masked by the bank's admission rule (`is_finite & abs <= 1e100`)
  first; and the bank emits `n_eff` through warm-up while the other fields
  are null, so the twin gates every field but `n_eff`.
- *Validity is a rate.* `E[E_t] ≤ 1` is not a test of an e-process (a
  process that never bets passes it); the guarantee is
  `P(sup E ≥ 1/α) ≤ α`, so the tests run twenty thousand streams as groups
  of one bank and count the ones whose peak crosses. Fair coin, 300 rows:
  2.2% at α = 5%, 0.4% at 1% (a lower bound of 0.5% / 0.05% catches a
  bettor that never bets); a dependent null with `P(+ | last +) = 0.3`,
  `P(+ | last −) = 0.5` — not i.i.d., still never favouring +; the two-sided
  average under 5%. Power, 2000 streams: 60% positive crosses 20 within
  1000 rows in 99.9%; 55% in 12% / 59% / 100% at 200 / 1000 / 4000 rows.
  Six million rows in half a second, so these run in the default suite.
- *What it does not claim.* A test of the *sign*, by design and by
  docstring: 60% gains of 1e-9 against 40% losses of 1e6 is "positive". A
  test of the mean would need bounded losses or clipping, which E42 ruled
  out; the sign-conditional null is the one that holds for dependent rows.

**Task 31's decisions: the performance survey, 2026-09-05.**

- *Bit-exact or not at all.* Every change recomputes the same arithmetic
  on the same inputs (`k` square roots reused across the pairs; a kept
  Cholesky factor of the same matrix; a projection's bounds prepared once;
  multiplications by exactly 1.0) or is plumbing (scratch buffers, skipped
  residual tracking for a model with no predictions, a packed validity
  bitmap, a contiguous copy). Proved, not argued: two output dumps over
  every family, with groups, weights, a null key, `predict` and chunk
  ends, compared as `u64` bits plus validity against a build of the
  Task 30 commit. A survey row that moved the wrong way was a real
  regression (the constraint scratch made the unscaled simplex 9%
  slower) and was fixed by keeping the folded instantiation, not
  accepted as noise.
- *The stride is the bank's decision, not the caller's.* `ChunkOut` is
  slot-major and a row is one store per slot at stride `n_rows × 8`
  bytes; a power-of-two chunk put every slot of a 231-wide `ew_cov` row
  in the same L1 sets (65 536 rows: 1302 ns/row; 65 552: 685), and a
  400k chunk paid a page per store. Each (spec, group) work item now runs
  its rows through the stream in runs of `ChunkOut::run_rows` — buffers
  within 2 MiB, stride an odd number of 128-byte lines, floor five lines
  — each run its own `ChunkOut`, the coefficient report on the last run's
  final row. Chunk invariance is what makes it invisible; three bank
  tests pin it. Rejected alternatives: a row-major layout (changes the
  reader, the `at` contract and the scatter path for the same cache
  effect) and splitting the predict-only path (the coef-on-last-row rule
  is the thing to preserve, and `predict` reports none).
- *Wall clock over profile.* `sample` weights a phase by its thread
  count; `assemble` on fourteen threads looked like half the chunk when
  `ONLINE_TIMING=1` said 20 ms of 300. The per-chunk section rows are the
  truth for *where*; the profiler is for *what*, at function granularity.
  Written up as a recipe in `docs/PERFORMANCE.md` §13.
- *One factor per learned row.* `ew_class` full covariance cached the
  factor per class (`solve::SpdFactor`, `ewclass::Factors`,
  `#[serde(skip)]`, invalidated for the class a row learns) rather than
  refactorizing all classes every row; the shared form was already at one
  factorization per row and did not move.
- *Deferred, with the reason.* `kalman`'s standardizer is a full `EwCov`
  read on its diagonal — a diagonal accumulator is O(k) instead of O(k²)
  but is a serialized layout change (rule 5, schema bump), left for a
  release with another reason to bump. `pca_every=1` is O(k³) per row by
  request. The simplex sort of `2k` breakpoints could be an O(k)
  selection. Compare mode's second pass is 8 ms per 400k rows.
- *Documented ratios, not a loaded machine.* The README's two tables are
  from one quiet run of `scripts/benchmark.py --markdown` on the final
  build; the per-row and thread-scaling tables in `docs/PERFORMANCE.md`
  §13 carry both builds so the ratio, not the absolute, is the claim.

**Task 32's decisions: 0.2.0 prepared, 2026-09-05.**

- *What ships.* Tasks 23–31 on `clustering-build`, fast-forwarded onto
  `main`: `kmeans` and `micro`, conformal intervals, `mahal` and EW-PCA
  on `ew_cov`, `ew_class`, constrained `sgd`/`pa`, `kalman` reversion,
  `seqtest`, and the bit-exact performance work. Version 0.1.1 → 0.2.0 in
  `Cargo.toml` (workspace and the two path dependencies), `pyproject.toml`,
  `__init__.py` and both lock files — the same six files as the 0.1.1 bump;
  `CHANGELOG.md` cut with `[Unreleased]` left at "Nothing yet."
- *Minor, not patch.* Pre-1.0 the minor carries breaking changes, and
  there is one: a residual diagnostic on a model with no predictions is
  refused by name where `ew_cov` accepted it silently. Everything else is
  additive. The README's pin advice moves to `~=0.2.0`.
- *The 0.1 numbers are the 0.1 numbers.* `docs/VALIDATION.md` regenerated
  on this build differs from 0.1.1's in the version line and one timing;
  `golden.rs` and `test_golden_pipeline.py` gained entries for the new
  models and lost no value (the diff since `main` removes three type
  signatures and nothing numeric); Task 31's two dumps compare bit for bit
  against the Task 30 build. The README's two *Performance* tables and the
  test-count line (466 Rust tests, ~1,700 pytest cases from 920-odd
  functions) are regenerated; `docs/TESTING.md`'s status line with them,
  its coverage figures left dated.
- *Where it stops.* Gate PASS on every commit; CI green on the Task 31
  commit (33960229948) and on the prepare commit (33962266816, superseded
  by the run on this head); `main` fast-forwarded to this head and pushed
  once that run is green. The Release rehearsal (`release.yml` dispatched
  without a tag) and the annotated tag `v0.2.0` are the user's steps, as
  for 0.1.1: the tag goes on the commit whose rehearsal and CI are green,
  and the `v*` ruleset makes it permanent.

**Task 33's decisions: the diagonal standardizer and schema 3, 2026-09-05.**

- *The deferral was the user's to overturn, and they did.* Task 31 left
  `kalman`'s standardizer alone because a diagonal accumulator is a
  serialized layout change (rule 5). Asked, the user said a schema change
  is fine this early in the project; the same waste in `sgd`'s
  `scale_features` scaler goes in the same bump rather than a later one.
- *Bit-exact by construction, then proved.* `EwDiag::update` is the
  diagonal path of `EwCov::update` with the same operation order (`a·c +
  a·b·d·d`, the deviation against the old mean, the mean updated after);
  Rust does not reassociate or fuse, so the bits agree. A unit test
  compares the two accumulators as `u64` bits over an adversarial stream
  (zero-weight first row and run, weights 0/1/2.5, two decay factors,
  offsets and scales apart by orders of magnitude, k = 1, 2, 5), and the
  Task 31 dump recipe — twelve configurations, groups, weights, two
  targets, a save/load mid-stream, `predict`, coefficients — compares
  bit-identical against a wheel built from the previous commit.
- *Migration by field name, not by luck.* A compact msgpack struct is an
  array, so `#[serde(untagged)]` layouts are newtype variants (the `rls`
  v1→v2 precedent). `kalman` renamed `cov` → `stats`, so the two layouts
  are told apart by name. `sgd`'s scaler is `Option` with
  `#[serde(default)]`: renaming it would make every schema-2 file load
  with `None` — silently, the file's scaler discarded — so the name stays
  and the layouts are told apart by `EwDiag`'s `deny_unknown_fields`
  (an `EwCov` map carries three names it does not know) plus a shape
  check (an `EwCov` array has `k²` entries where `k` are expected), each
  tested in both encodings. The conversion is `EwDiag::diagonal_of`: the
  numbers the model was already reading, so a schema-2 state continues
  the stream identically to one that never left schema 3.
- *A real 0.1 file, frozen before the change.* `state_schema2.rs` carries
  a 4126-byte bank written by the pre-change build (a hex constant, rule
  1) — `kalman` standardized with intercept and `sgd` with
  `scale_features`, a group column — and checks it loads, continues the
  stream to the bit against a fresh bank, and re-saves as schema 3. The
  bank `format_version` stays 2: the envelope did not change.
  `tests/api_surface.txt` records `schema_version = 3` (a deliberate,
  reviewed diff). The JSON-mutation refusal test lives in `online-core`,
  not the bank test: `serde_json` writes `inf` as `null`, so a bank
  file's `revert_halflife = [inf]` cannot survive that round trip.
- *`standardize=False` gains too.* The standardizer is updated on every
  row regardless (it carries `n_eff`), so the raw `kalman` runs 213 → 172
  ms as well. Reported as before/after from two wheels run back to back
  on the same (loaded) machine — the ratio is the claim.

**Task 34's decisions: the last row in the bank file, 2026-09-05.**

- *It is one row of `ChunkOut`, not a second set of diagnostics.* Every
  output buffer keeps the row index innermost (`(mi*n_slots+slot)*n_rows
  +ri`, and the same shape for metrics, conformal, quantiles, `n_eff`,
  `lam_selected`), so `LastRow::take` is one strided gather per buffer
  and `to_chunk` puts the row back into a 1-row `ChunkOut` that the
  ordinary `assemble` turns into the struct column. The saved row is
  therefore the `fit_predict` row field for field — `emit_selected`,
  `emit_averaged` and the rest are re-derived by the same code. A new
  `ChunkOut` buffer has to be added to `take` and `to_chunk` by hand, and
  switched on in `tests/last_row.rs`'s rich spec, whose field-for-field
  comparison against the frame is what catches the omission
  (`docs/EXTENDING.md`).
- *The last row **learned from**, not the last row fed.* `remember_last`
  runs after every accepted run in `process` and keeps the run's last
  *processed* row, so a chunk that ends in skipped rows (a null feature
  or weight) steps back to the row before them, and a group that never
  learned reports a null row. `predict` and `score` never touch it: the
  saved row describes the state, and the state did not move.
- *`coef` when the frame's row carried it.* The row is faithful to the
  frame, so `coef` is present exactly when that row had it — a chunk's
  last accepted row does, an interior row only on the `coef_every`
  cadence. `coef()` remains the way to the coefficients otherwise;
  duplicating them into every saved row would have made the file's row
  disagree with the frame's.
- *Additive, so no format bump.* `StreamState.last_row: Option<LastRow>`
  with `#[serde(default)]`, map-encoded like `rows_fed` and the residual
  statistics before it: files from 0.1.x load and report a null row
  (`state_v1.rs`, `state_schema2.rs` check this on the frozen fixtures).
  `Stream::restore` checks the row's shape against the spec — twelve
  lengths — so a file whose spec no longer matches its row is refused,
  not assembled out of bounds.
- *Python stacks specs `diagonal_relaxed`.* `ModelBank.last_row(spec=None,
  group=None)` returns `spec`, `group`, then the struct's fields unnested,
  one row per (spec, group); specs with different fields stack with nulls,
  which is what a table of many fits needs. An unseen group is an empty
  frame, not an error, so a glob over saved files can ask for one group
  everywhere. The CLI is untouched: it exposes neither `coef` nor an
  inspect facility, and the data is read through `ModelBank.load`.

**Task 35's decisions: the training data in the bank file, 2026-09-05.**

- *Fed after the clock commits, so a refused row feeds nothing.*
  `Stream::process_chunk` plans every row on a copy of the clock state
  and commits when the whole chunk is accepted; `DataSummary::feed_row`
  runs on the committed plans, one call per row, so a chunk that a
  backwards clock refuses under `on_clock_reset = "error"` leaves the
  summary where it was, as it leaves the models. No per-chunk clone of
  the summary is needed for that.
- *Undecayed, in row order, one Welford per column.* The summary answers
  "what was this trained on", and a decayed count would answer "what does
  it remember" — `n_eff` already does. Row-order Welford with a shared
  reciprocal per row (`inv = 1/rows_fed` when the column has no nulls, its
  own `1/count` otherwise — bit-identical to plain Welford, proved by a
  unit test) makes the numbers a function of the row sequence alone, so
  chunking cannot move a bit; that is asserted with `to_bits` in
  `tests/summary.rs` and `tests/test_summary.py`. A two-pass-per-chunk
  merge (Chan) would vectorise but is not chunk-invariant, so it was not
  used. Measured cost: about a nanosecond per input column per row (`sgd`
  at k = 20 from 95 to 112 ns/row, `kalman` unchanged within noise), always
  on; an opt-out is a candidate if a wide, cheap model needs it.
- *The models' notion of a usable value.* A column's `count` is the rows
  `online_core::INPUT_BOUND` admits — finite and at most `1e100` — and
  everything else is `null_count`, so `describe`'s counts are the counts
  the models saw. A `seqtest` comparison's one "target" is its own
  difference of residuals, an `ew_class` label is its class index with
  counts only, and an unsupervised model's mirrored target is dropped from
  `describe` (its features are the data). Events are the clock schedule's
  own: `session_changed` when both sessions are known and differ,
  `backwards` when the clock fell within a session (a new session owns its
  clock), `reset` when either policy reset the model.
- *Additive, so no format bump; a legacy file never grows one.*
  `StreamState.summary: Option<DataSummary>` with `#[serde(default)]`;
  `Stream::restore` keeps `None` for a file written before it, and `None`
  stays `None` through more rows and a re-save, so the frame reports nulls
  rather than a count that began partway. `rows_processed` and
  `last_clock`, which the stream always kept, are filled in either way
  (`state_v1.rs`, `state_schema2.rs` check the frozen fixtures for this).
  A saved summary is validated against its spec and its own arithmetic —
  column count, counts against `rows_seen`, finite moments, a range that
  is a range, events no more than rows — and refused with the spec named.
- *Testing the file, not the round trip.* Beyond every facet equal across
  save/load and a byte-for-byte re-save, `tests/summary.rs` truncates a
  real file at every length and flips bits through it, requiring each
  result to be refused or to load into a bank whose every accessor works
  (most flips land in f64 payload and load; the test asserts both
  outcomes occur). `state_schema3.rs` freezes a 0.2.0 file with a last
  row and a summary as a hex constant, pins its counts by hand, and
  requires the next layout to load it, continue it to the bit and, while
  the writer is unchanged, re-save it byte for byte.
- *Python stacks specs with a `spec` column.* `ModelBank.summary(spec=None,
  group=None)` and `describe(...)` return every spec by default with `spec`
  first; the schemas are fixed across specs so plain `concat` stacks them.
  An unseen group or a fresh bank is an empty frame of the right schema,
  not an error. The CLI is untouched, as for `last_row`.

**Task 36's decisions: `stats=[]` on `ew_cov`, 2026-09-05.**

- *One line.* The whole change to the model is the removal of the
  `validate()` clause that refused an empty `stats`. `n_outputs()` was
  already the sum of the statistic slots, the Mahalanobis quantiles and
  the PCA slots, so an empty list gives 0 + those; the stream already
  tolerated a model with zero slots (`nc = n_slots / m_targets`, floored
  at 1, with `m_targets = 0` for `ew_cov`); the output struct is then
  `{n_eff}` alone. Nothing in `online-polars` or `online-py` changed.
- *`None` and `[]` differ.* `stats=None` (Python) and a missing `stats`
  (TOML) still default to `["mean", "std", "corr"]`; only the explicit
  empty list means accumulate only. Changing the default would turn every
  existing `ew_cov` spec silent.
- *The extra slots do not need a statistic.* `pca=r` and
  `mahal_quantiles=[…]` add their outputs on `stats=[]` as on any list —
  they were never tied to an entry in it — except that `mahal_quantiles`
  without `"mahal"` still errors, because a quantile of a score that is
  not computed is a configuration mistake, not a request.
- *Same state to the bit.* The core test drives a bare and a full spec
  over the same weighted, clocked rows and asserts `cov()` equal, `n_eff`
  bit-equal, and the bare state restoring to itself; the Python test
  checks `gram()` of the bare spec against the full spec and against
  `np.cov(bias=True)`, then runs the empty list through every surface —
  `fit_predict`, chunked, `LazyFrame.online.fit_predict`, `po.run` with
  `save_state` and `ModelBank.load`, the expression plugin under its
  warning, and the CLI from TOML (`[specs.model] type = "ew_cov"`,
  `stats = []`).

**Task 37's decisions: `marginal`, 2026-09-05.**

- *`ew_cov`'s arithmetic, operation for operation.* `Marginal::learn` is
  the weighted-Welford mean form of `EwCov::update` — `a = λW/W'`,
  `b = w/W'`, `S' = a·S + a·b·dᵢdⱼ` with deviations from the *old* means,
  then `m += b·d` — applied to each pair's three moments, and `pair()`
  derives `corr` with the `ew_cov` reader's formula (`√var_x·√var_y`,
  clamp to ±1, NaN on a zero variance). That is what makes a pair's
  correlation bit-identical to an `ew_cov` over the two columns, which a
  core test and a Python test hold; a mathematically equal but
  differently ordered update would agree to 1e-15 and no further.
- *Per-target `min_periods` lives in the core config.* Every other model
  takes one threshold and the stream gates each target's fields; a
  `marginal` emits nothing per target, and its pairs are read from the
  state by whoever holds it, so the model gates `corr`/`beta`/`t`
  itself, per target, from `MarginalCfg.min_periods: Vec<f64>`
  (`spec.min_periods_per_target()`). The default is 3, not
  `k + intercept`: a pair is over two columns whatever the feature
  count, two rows give ±1 whatever the data, and three is the first
  count with content.
- *Not unsupervised.* A `marginal` has real targets — the columns its
  pairs are against — so it is not in `is_unsupervised()` (whose targets
  mirror `features[0]`), and the leak check (no column on both sides)
  applies. It is in `predicts_no_target()`, with `ew_class` and
  `seqtest`, so the residual switches are refused by name and no `resid`
  buffer is kept.
- *A null target ages its pairs.* `W_t ← λW_t`, `Q_t ← λ²Q_t`, moments
  untouched; the row is still learned by the other targets and by
  `w_sum` (the struct's `n_eff`, over every processed row whatever its
  targets). A zero-weight or zero-`W'` row is skipped per target, so a
  zero-weight first row leaves the pairs bit-equal to a stream without
  it (rule 9).
- *Kish's `n` for the t-statistic.* `t = corr·√((n_kish − 2)/(1 − corr²))`
  with `n_kish = W_t²/Q_t`, the count of equally weighted rows carrying
  the same information ((1+λ)/(1−λ) for unit weights, held by a test);
  documented as a scale for comparing pairs, not a p-value, since the
  rows are neither independent nor Gaussian. `n_kish ≤ 2`, a constant
  column and a perfect correlation give NaN in the core and null in the
  frame — the frame never carries NaN, as the output structs never do.
- *A long frame, sorted by group.* `Bank::marginal` returns one row per
  (group, instance, feature, target), groups in the order `gram()` and
  `summary()` use, targets outer and features inner in spec order; a
  group never seen is an empty frame of the same schema, a non-`marginal`
  spec a `ValueError` naming its kind (an `ew_cov`'s moments are
  `gram()`'s). The frame is built on the Rust side; no numpy.
- *The golden pipeline pins the state.* Every other model's golden
  values are struct fields at three rows; a `marginal`'s struct is
  `n_eff`, so `signature()` also reads `bank.marginal()` after the last
  row and pins `n_eff`, `n_kish`, `corr`, `beta` and `t` per (group,
  feature) — the same numbers on three operating systems.
- *A `ModelState` variant appended, no schema bump.* `ModelState::Marginal`
  is added at the end of the enum; the file format encodes a variant by
  name, so schema-3 files written before it still load (rule 5's
  appended-variant precedent, as for `seqtest`).

**Task 38's decisions: the complete Gram, 2026-09-05.**

- *`Option`, not a sentinel, and skipped when absent.* `EwCov::q_sum`,
  `EwRidge::tm` and `Lasso::tm` are `Option`, `#[serde(default,
  skip_serializing_if = "Option::is_none")]`. A state written before this
  task deserialises to `None`, keeps streaming, and re-saves **byte for
  byte** — the schema-3 fixture's `a_schema_3_state_re_saves_byte_identically`
  still passes, which is why the field is skipped rather than written as
  nil. No `SCHEMA_VERSION` bump: additive defaulted fields are the
  `rows_fed` precedent (rule 5), as an appended variant was for task 37.
- *A legacy state reports `None` for good, not from the resume point.* The
  tempting alternative — start accumulating `Q` on load — pairs a `Q` over
  the rows since the resume with a `W` over the whole stream, and reports an
  effective sample size too large by the length of the history. That is a
  wrong number where `None` is the true answer, and it would be wrong
  silently. `a_schema_3_state_reports_no_kish_size_and_no_target_moments`
  feeds twenty more rows and asserts it is still `None`.
- *The target moments ride the cross-moment update's own `a` and `b`.*
  `TargetMoments::learn` takes them rather than recomputing from `W_t`, so
  the arithmetic is `EwCov::update`'s operation for operation and a target's
  variance equals an `ew_cov` over that column to the bit — asserted in the
  core and again through `gram()` in Python. It is the same reasoning as
  task 37's, and the same test.
- *Kish's `n` is scale-free, and that is a feature.* `W` decays by `λ` and
  `Q` by `λ²`, so `W²/Q` is unchanged by pure decay: a target that stops
  arriving keeps the sample size its moments earned, and `target_weights` is
  what collapses. A first draft of the test asserted the opposite and was
  wrong, not the code. The docstring and the README now say which number
  means "how many rows" and which means "how recent".
- *A blend mixes `Q` by the moments' coefficients, not as a union.* The
  slow twin sees the *same* rows under a longer halflife, so summing the two
  `Q`s (the exact `Σw²` of the re-weighted union) reports a blend of a state
  with itself as twice as informative as the state. Mixing by `af`/`as`
  keeps the invariant the means, co-moments and weights already have: an
  identical twin blends to the identity. Documented on `TargetMoments::blend`
  as an approximation for overlapping histories, which it is.
- *Empty is not `None`.* `ew_cov` has no targets, so its three target lists
  are empty; `None` is reserved for "this state cannot tell you". Two
  different answers, two different values.

**Task 39's decisions: `po.gram`, 2026-09-05.**

- *The mapping names its own axes.* `gram()` gained `columns` and `targets`,
  derived on the Python side from the spec the bank already holds — no Rust
  change. Without them `subset(g, ["x0", "x2"])` and `solve(g, target="y")`
  could only take positions, and a caller would have to rebuild
  `["intercept"] + features` by hand and get `ew_cov`'s missing intercept
  wrong. `"intercept"` is the name `coef_index` already gives that slot.
- *Held against the models, not against a second copy of the formula.*
  `solve` is checked against `bank.coef()` over both standardization paths,
  three ridges and with/without an intercept; `lasso_path` against the
  `lasso` model's own path; `merge` against the Gram of the whole stream.
  A test that re-derived the same algebra in numpy would only prove I can
  write it twice.
- *Not bit-identical, and the docs say so.* E46 asked for "to the bit". The
  models factorize with `faer`'s Cholesky and numpy with LAPACK's LU; the
  measured gap is ~1e-16 relative on a well-conditioned system, and the
  assertions sit at 1e-12. Claiming equality would have been false, and the
  first draft of the docstring did claim it.
- *A grid of ridges rides one eigendecomposition where the penalty is
  uniform.* That is the standardized path always, and the unstandardized one
  without an intercept. With an intercept the model leaves that column
  unpenalized, so the penalty is not a multiple of the identity and each
  value costs a factorization. The centred reduction that would restore the
  shortcut is algebraically identical but rounds differently from the
  model, and fidelity to the model is worth more here than the speed.
- *`merge` is for parts that share a weighting.* Shards of a pass, groups
  being pooled, workers. Two halves of a decayed stream in time order are
  *not* the stream: each part's weights are relative to its own last row.
  The docstring gives the rescaling (`n_eff * lam**dt`, `Q * lam**(2*dt)`,
  the means and co-moments untouched), and a test does exactly that and
  recovers the whole — the caveat is held by a test, not only by prose.
- *`coef_stats` divides by `n_kish`.* A weighted stream's `n_eff` is not a
  count, and using it would report standard errors too small by however
  unequal the weights are. A test puts ten times the weight on every row and
  asserts the errors do not move. The intercept's standard error is `nan`:
  it depends on the design's centring, which the Gram has already absorbed.
- *Two disagreements that are not bugs, both now tests.* `bank.coef()` is as
  of the model's last *solve* (its `solve_every` / `max_rows_between_solves`
  schedule), while `gram()` is as of the last row — the first draft of the
  test read that gap as a defect in `solve`. And the tests use `lam=1.0`, no
  decay at all, wherever they compare a merge against a whole: `halflife=1e12`
  is nearly but not exactly no decay, and at 2400 rows the difference shows
  at 1e-6.
- *`penalty_weights` is the one thing the models do not also do.* Offline the
  path is re-walked rather than carried, so a per-feature penalty costs
  nothing; online it would be another vector in the state for a knob nobody
  has asked for. Listed in E46, added here, and not added to the model.

**Task 40's decisions: `label_delay`, 2026-09-05.**

- *Two virtual rows, not two code paths.* A delayed row is scored where it
  sits and stepped later, so the plan list is rewritten into
  (release, …, score this row) and `RowPlan` gained `emit` and `learn`.
  `run_instance` already guarded every diagnostic fold behind `learn` for
  the scoring path, so the fold deferral came almost free; the new work is
  guarding the output writes behind `emit`. A spec without a delay pays two
  predictable branches a row and nothing else.
- *Maturity is measured on the model's clock.* Not the raw column: the
  delta the models are given, capped by `max_dclock`, with skipped rows'
  time folded in. That is the only clock that is chunk-invariant, and it
  makes `label_delay` mean the same thing as `halflife` does. With no
  `clock` column it is one unit per accepted row, so `label_delay = 20` is
  twenty rows. Each buffered row counts down by every later row's delta
  rather than holding an absolute deadline, so a long stream cannot lose
  precision in a running clock.
- *A released row replays the delta it arrived with.* The models therefore
  see exactly the gap sequence they would have seen without the delay, just
  later — which is why the state at the end is bit-for-bit a plain bank fed
  only the matured rows, and a test says so.
- *A reset drops the buffer and applies at its row; a session change
  releases it.* A clock reset means "this state is no longer about this
  stream", so deferring it would leave the old model predicting the new
  regime; the rows waiting to teach that state go with it. A session change
  keeps the model, but one session's clock does not measure time in the
  next, so a row still waiting would wait for a deadline that never comes:
  it is released in order at the boundary. Both were arrived at by writing
  the test and watching the first version get 66.9 where 51.8 was right.
- *The doubled stream is the oracle, and it disagrees in exactly one
  place.* `po.prep.embargo` builds E47's recipe and the native path matches
  it field by field, bit for bit, at three delays — once `embargo` was
  fixed to put the lesson *before* the prediction at a tied clock, which is
  what "the clock has reached `t + delay`" means. The exception found by
  the test: `resid_quantiles`, `emit_autocorr` and `emit_drift` take no row
  weight (a P² estimator counts samples), so a zero-weight predict row
  feeds them as much as its learn copy and every residual lands twice. The
  native path feeds them once. That is the better answer, and it is now a
  test and a line in the README rather than a surprise.
- *A skipped row agrees to a ulp, not a bit.* A null feature's clock time
  folds into the next *accepted* row, and the doubled stream's accepted
  rows are not the same rows, so the two partition the same elapsed time
  differently. The decay factors multiply to the same number to within
  rounding, which is all that can be asked.
- *`SCHEMA_VERSION` 3 → 4.* Not because a model's state moved — none did —
  but because every spec's bytes moved: the spec each bank file carries
  gained `label_delay`. Rule 5 asks to be told about a layout change, and
  `state_schema3.rs` was written to become the upgrade test the moment the
  writer changed, which is exactly what happened. `MIN_SCHEMA_VERSION` is
  still 1 and every older fixture still loads. Task 38's additions rode
  into schema 3 without a bump because `skip_serializing_if` kept an old
  file's bytes where they were; a spec field cannot do that without making
  `bank.specs` disagree with the dict it was built from.
- *A frozen schema-4 fixture is owed once 41–43 have landed*, so it is
  frozen once rather than three times. Task 44 does it.

**Task 41's decisions: E48 measured and rejected, 2026-09-05.**

- *The premise was wrong, and measuring is what showed it.* E48 said
  mirroring the upper triangle is "bit-identical (the products commute in
  IEEE)". The products do commute; the association does not.
  `c[i][j] = a*b*d_i*d_j` is `((a·b)·d_i)·d_j` and the transposed entry is
  `((a·b)·d_j)·d_i`, whose intermediates round differently — so the matrix
  the library has always kept is symmetric to about 1e-16 and *not* to the
  bit, and mirroring moves every golden value. Writing it as
  `(a·b)·(d_i·d_j)` would make it exactly symmetric, and is itself a
  different rounding from today's. Either way E48's "the goldens, unchanged"
  cannot hold.
- *And it is slower, by 49% to 107%.* The mirror store `c[j*k+i]` walks a
  new cache line for every `j`: the loop touches the whole matrix twice
  instead of once, defeats the prefetcher and cannot vectorise. Halving the
  flops and doubling the traffic is a pessimisation at every width where a
  triangle would be worth having. Five variants, six widths, two runs, each
  checked bit-for-bit before being timed (docs/PERFORMANCE.md §14).
- *What shipped instead is on the same loop and is bit-identical.* The
  deviations are computed once into a row scratch — the old form recomputed
  `x[j] - m[j]` once per *row of the matrix*, `k²` subtractions where `k`
  will do — and the inner loop runs over `c`'s row slice zipped with that
  scratch, so `c`, `x` and `m` are not indexed by `j` and the bounds checks
  that kept it scalar are gone. −14% at k = 4, −63% at 16, −45% at 64,
  −22% to −27% from 200 up, and the goldens do not move.
- *The scratch is a buffer, not state.* `#[serde(skip)]` and a `PartialEq`
  that always returns true, so a state file carries none of it and a
  round-tripped accumulator still compares equal to the one that wrote it.
  It refills itself on the first row after a load; a test asserts both.
- *The rejection is kept as a test, not only as prose.*
  `the_two_triangles_are_equal_but_not_bit_equal` fails if the two ever
  become bit-equal, and its message points at §14 — so the next person to
  reach for the shortcut finds out why it was not taken, in the place they
  would reach for it.

**Task 42's decisions: mergeable evaluation sums, 2026-09-05.**

- *Centred sums, not the raw ones E49 asked for.* The row lists `Σy`, `Σy²`,
  `Σŷ²`, `Σyŷ`, which make `merge_sums` a plain addition. They also lose the
  variance entirely at a large offset: a target around 1e8 with unit spread
  has `var / E[y²] ≈ 1e-16`, and `Σy² − (Σy)²/n` has nothing left after the
  subtraction. That is E11b's finding, in the same library, and taking the
  raw form here would have re-made the mistake the accumulators exist to
  avoid. The stored fields are the weighted means plus centred second
  moments, and the merge pays a parallel-axis term for them — a multiply per
  key per part. Still ten doubles, still O(state).
- *An n-way merge, not a fold of pairs.* `merge_sums(*parts)` concatenates
  and reduces in one `group_by`: the pooled mean is the weight-weighted one,
  and each part's centred sum picks up `w_g·(mean_g − mean)²`. Every part
  enters as a sum and never as a difference of running totals, so a hundred
  parts lose no more than two — asserted at 2, 5 and 97.
- *Held against `metrics`, not against a rewrite of it.* The test is
  `from_sums(sums(df)) == metrics(df)`, column for column, grouped and
  ungrouped. A test that re-derived R² in numpy would only show I can write
  the formula twice.
- *`rmse` beside `mse`, and nulls where a metric is undefined.* E49 named
  `rmse`; `metrics` reports `mse`. Both are here, so the two frames line up
  and the extra column is the one the row asked for. A key whose target
  never varied gets `null` for `r2` and `ic` rather than an infinity that
  would read as a number.

**Task 43's decisions: a quiet run, and a spec that leaves things out,
2026-09-05.**

- *No output is a destination, not a missing one.* `Output::Discard` sits
  beside `File` and `Batches`, so `deliver` drops the frame and no writer
  thread is spawned at all. The empty-input path — which exists to write a
  valid empty frame of the right schema — is skipped too: collecting the
  plan only to drop it would be a read of the source for no one.
- *`save_state` is required in its place.* A run that writes nothing and
  saves nothing has done nothing, and silently succeeding at it is worse
  than refusing. The message names both ways out.
- *`output=None` already meant this; `no_output=True` is for the other
  case.* Leaving `output` out of both the call and the config is E50's
  `po.run(output=None, save_state=)` and needed no sentinel. The flag is
  for clearing an `output` a config carries, and mirrors the CLI's
  `--no-output`, which `conflicts_with` `--output`.
- *Filling happens where a spec enters, and on both sides of the load
  check.* `Spec::fill_defaults` is called by `Bank::new` (so every surface
  that runs a spec, and every state file, carries the filled form), by
  `RunConfig::fill_defaults` (the CLI and `po.run`, before validation), and
  on both the expected and the saved specs in `Bank::load_bytes` — the last
  because a file written before the filling existed carries the unfilled
  form, and refusing to load it would be a regression for the sake of a
  field that means the same either way.
- *`drift_action` came out of the byte-identity requirement.* E53 asks for
  a TOML spec and a Python spec to be byte-identical. `targets` was one
  difference; `drift_action` was the other — the builders write `"flag"`
  and a TOML author would not, and `None` and `"flag"` already meant the
  same thing to every reader. The test that pins this compares the two
  state files, not the two dicts, so any third such field will be found the
  same way.
- *A better error for the case that is still wrong.* An unsupervised spec
  with no features and no targets used to say "targets must be non-empty",
  which points at the field the author was right to omit. It now says
  features must be non-empty, and why.

**Task 44's decisions: the schema-4 fixture, 2026-09-05.**

- *Frozen after the layout stopped moving, not when it first moved.* Tasks
  40–43 each touched what a bank file holds; freezing after 40 would have
  meant regenerating three times, and a fixture regenerated is a fixture
  that proves nothing.
- *Four specs, two of them there for the new fields.* `d` is an `ew_ridge`
  with `label_delay = 8`, so the file carries a **non-empty** pending
  buffer — the part of schema 4 that a converter would be most likely to
  drop. `u` is an `ew_cov` written with no `targets` at all, so the file
  carries the ones `fill_defaults` put there, and a test asserts both that
  and the filled `drift_action`. The other two are schema 3's, so the two
  fixtures are comparable.
- *The other three fixtures stay exactly as they are.* Their module docs
  already say so, and `state_schema3.rs`'s byte-identity test has taken its
  upgrade branch — which is what it was written to do.

**Preparing E54–E64 for implementation (tasks 45–56), 2026-09-05.** The
eleven items of ENHANCEMENTS §10 were read against the code they land in —
`model.rs`, `clock.rs`, `ewcov.rs`, `ewdiag.rs`, `ewclass.rs`,
`cluster/kmeans.rs`, `stats.rs`, `drift.rs`; `stream.rs`, `bank.rs`,
`spec.rs`, `runner.rs`, `summary.rs`; `_bank.py`, `_frame.py`, `_runner.py`,
`_spec.py`, `_kwargs.py`, `prep.py`, `gram.py`; `EXTENDING.md` — and each
became a task above. What follows is what the implementer needs beyond the
§10 row: the decisions the row left open, resolved; the places the row was
wrong about the code, corrected; and the exact seams each task touches.
Where a choice is genuinely free it is marked *implementer's call*; nothing
else is. The points this block leaves to be confirmed from the papers —
the log base and `D̂` of the constancy monitor and its size/power tables,
Higham's example matrices, the Ledoit–Wolf intensity formulae, the
pre-averaging constants, the kernel bandwidth rule, the Epps-inversion
weights, the LDECO conventions — are answered in `docs/ANSWERS-E54-E64.md`,
which also says which items it could *not* verify.

*Batch-wide.*

- *Order.* 45 → 46 → 47 → 48 → 49 → 50 → 51 → 52 → 53 → 54 → 55 → 56, as
  §10 gives it, with 47 inserted before the first model that needs it and 52
  (`po.sim`) before the three detectors whose experiments need a stream with
  a known truth. 47 is small and must land before 48; 49 is independent of
  everything and can be built at any point.
- *One schema bump for the batch.* Task 45 adds a field to `Spec`, and every
  bank file carries its specs, so every file's bytes move: `SCHEMA_VERSION`
  4 → 5 there, with a loader that is the existing one (`#[serde(default)]`
  fields; a schema-4 file loads, continues to the bit and re-saves as 5),
  and the `lib.rs` history entry. Everything after 45 rides on 5 without a
  further bump: appended `ModelState` variants (`Deco`, `Rcov`, `Hmm`,
  `CorrChange`, `Bocpd`), `Option` fields with `skip_serializing_if` on
  `EwCovModel` and `StreamState`, and new map keys on `BankFile` with
  `#[serde(default)]` (the file is written with `to_vec_named`, so a key is
  additive). Task 56 freezes `state_schema5.rs` once the layout has stopped
  moving — Task 44's reason: a fixture regenerated proves nothing.
- *The model shape.* 46, 50, 53, 54 and 55 are `ew_cov`'s shape: `n_targets()
  = 0`, `is_unsupervised` and `predicts_no_target` both true, `fill_defaults`
  writing `targets = [features[0]]`, the residual-side flags (`emit_sigma`,
  `emit_resid_z`, `emit_metrics`, `conformal`, `resid_quantiles`,
  `emit_autocorr`, `emit_drift`, `emit_selected`, `emit_averaged`) refused
  by name with `ew_cov`'s messages, `label_delay` refused (nothing to hold
  back), the leak check exempted where `ew_cov` is (grep `ew_cov` in
  `tests/` and `scripts/leakcheck.sh`), non-`f64` outputs through the
  `Source` variants that exist (`Cluster`, `Flag`, `Id`). Each needs ≥ 2
  features except `bocpd`; refuse fewer by name.
- *The clock-rescaling test, once per new model.* `Decay::factor` is
  `exp2(−d/h)`; scaling the clock column and the halflife by the same power
  of two leaves `d/h` bit-identical, so the output must be bit-identical at
  `c = 2^k` and equal to `1e-12` at `c = 3`. Put it in each model's
  `tests/test_<model>.py`; it is the "clocks are numbers" property §10 asks
  for, stated so it can fail.
- *Where things go.* Core: `crates/online-core/src/{deco,rcov,hmm,corrchange,
  bocpd}.rs`. Stream layer: `refresh.rs` in `online-polars` (task 49), the
  close path in `bank.rs`/`stream.rs` (45). Python: `corr.py`, `sim.py`, a
  second function in `prep.py`, `from_row` in `gram.py`. Docs: a `docs/
  REGIMES.md` for the detector experiments (53, 54), generated by
  `scripts/regime_experiments.py` and committed, *not* regenerated by the
  gate — the `docs/CLUSTERING.md` §7 precedent, chosen over
  `docs/VALIDATION.md` because `test_validation_doc.py` regenerates that file
  on every run and a Monte-Carlo experiment does not belong in a 0.4 s test.
- *Dependencies.* None new in Rust (`faer` already does the eigenwork;
  nothing is statically linked that was not). `numpy` only in Python, and
  only in `corr.py`, `sim.py` and the tests — the `gram.py` import guard
  with the install hint. No scipy, no scikit-learn.
- *Every task ends the same way.* `scripts/gate.sh` unpiped; `api_surface.txt`
  regenerated with `UPDATE_API_SURFACE=1` and the diff read; Sphinx `-W`
  over `docs/reference` (new modules need a page there); README `### <name>`
  heading for a model, whose code blocks the tests execute; CHANGELOG
  `[Unreleased]`; the §10 row's status; one commit per task with its number.

*Task 45 — closed-group emission (E54).*

- *Native order, not `GroupKey` order.* `GroupKey` holds integer keys as
  decimal strings and its derived `Ord` is lexicographic, so `"10" < "9"`;
  a monotone close on that order would close group 10 before 9 arrived.
  Add `fn key_cmp(a: &GroupKey, b: &GroupKey, integer: bool) -> Ordering`
  in `bank.rs`: integer compare (`i128` parse, the keys are what
  `integer_groups` formatted) when the group column's dtype is integer,
  bytewise on the string otherwise (String and Categorical, which `extract`
  already renders as strings). Any other dtype under `"monotone"` is refused
  by name at the first chunk (`group_indices` knows the dtype). A null key
  under `"monotone"` is refused naming the row (a null has no order); under
  `"session"` it is an ordinary key.
- *The pre-check is what makes interleaving an error.* `group_indices`
  partitions a chunk by first appearance and would silently gather the rows
  of `A, B, A` into one run per key. So, before any model is touched and
  beside `check_clock` (the refusal must leave the bank as it was; the
  `forget` path takes back the streams the chunk materialised), walk the
  key column in row order: it must be non-decreasing under `key_cmp`, and
  its first key `≥` the spec's high-water mark. The first offending row is
  named with both keys: `row 1234: group 7 after group 9 — group_close =
  "monotone" needs keys in non-decreasing order`. `label_delay` with
  `group_close` is refused by name at `validate`: a closed group cannot
  release the rows it is holding, and the closed row would then differ from
  the `gram()` a driver reads — the acceptance equality would be false by
  construction.
- *The monotone close batch.* After both phases and `rows_fed`, per spec
  with `"monotone"`: `max_key` = the chunk's largest key; every stream whose
  key is `< max_key` closes, in `key_cmp` order; then `high_water =
  Some(max_key)`. Content is chunk-invariant because every row of a closing
  group precedes the first row of a greater key, whatever the chunking;
  order is chunk-invariant because a batch closes in key order and a group
  never closes before a smaller one. The high-water mark is per spec, saved
  as a `GroupKey` string in a new `BankFile` key (`#[serde(default,
  skip_serializing_if = ...)]`, written only when some spec has one), and
  compared under the *current* column dtype on load — a stored key that no
  longer parses under it is an error naming both.
- *The session close is a split of the run.* `Stream::process_chunk` learns
  whether the spec closes on session (`ClockCfg` or a parallel flag; the
  stream already builds every `RowPlan` in pass 1). When a plan carries
  `session_changed` at `ri`, the run is processed as two segments: rows
  `< ri` as today, then `close_into(&mut self.closed)` — one `ClosedRow` per
  instance from the stream as it stands, then `*self = Stream::new(spec)` so
  the new session's first row is a first row (Δ = 0, fresh clock, fresh
  summary) — then rows `≥ ri`. Chunk-invariant because a session boundary is
  a property of two consecutive accepted rows. `Stream::closed` is
  transient (`#[serde(skip)]`) and is drained by the bank at the end of
  every `fit_predict`. `"session"` requires `session`; with `session_gap` it
  is refused by name (two prescriptions for one event — the close *is* the
  reset, with an emission). Sessions are stored hashed (`prev_session:
  Option<u64>`), so the closed row's `session` value needs the last session
  *value* kept: a `StreamState` field `#[serde(default, skip_serializing_if
  = "Option::is_none")] last_session: Option<String>`, written only under
  `"session"` so no other spec's bytes move.
- *Queue order.* One `Bank::closed: Vec<ClosedRow>`, appended per chunk in
  the order `(global index of the closing row, key_cmp)`: for a session
  close the closing row is the first row of the new session, for a monotone
  close it is the first row of the greater key; the global index is
  `rows_fed` before the chunk plus the row's position in it. Streams are
  processed in parallel, so this sort is what makes the queue's order a
  function of row order alone.
- *`closed_groups(spec=None, *, drop=True)`.* Returns the queue as one long
  frame, narrowed by `spec`; `drop=True` removes what it returned from the
  queue, `drop=False` peeks. Streams are dropped at close time, never at
  the call — a bank's memory must not depend on the caller polling. The
  undrained queue **is** saved (`BankFile` key, skipped when empty): a
  driver that saves between chunks without draining would otherwise lose
  rows silently. `Bank::predict` never closes anything; `closed_groups=`
  with `predict` is refused in `RunConfig::validate`, the CLI and
  `po.run`.
- *The row.* One frame schema per bank: the common columns always, a
  kind's block when any spec of that kind in the bank has `group_close`,
  null on other kinds' rows. Common: `spec` str, `group` str (null for a
  null key), `instance` str (the decay suffix, `""` for one instance),
  `session` str (the closed span's session value under `"session"`, null
  under `"monotone"`), `n_eff` f64, `n_kish` f64 (null before a weighted
  row), `rows_fed` i64, `rows_learned` i64, `clock_min` f64, `clock_max`
  f64 (null on a row-count clock). The last four are `DataSummary`'s
  fields, already computed and saved; §10's `clock_first`/`clock_last` are
  renamed to them rather than defined a second time — the two differ only
  when a clock ran backwards inside a group, and the summary's names are
  what the state already reports. Gram block (`ew_ridge`, `lasso`,
  `ew_cov`): `columns` list[str], `means` list[f64], `comoments` list[f64]
  (vech of the upper triangle with the diagonal, row-major, `k(k+1)/2`),
  `targets` list[str], `target_means`, `target_vars`, `target_n_kish`
  list[f64], `cross_moments` list[f64] (targets × k, row-major), `lags`
  list[i64] and `lag_comoments` list[f64] (task 48; `L × k × k`
  row-major; null without `lags`). `coef` list[f64] for every kind that
  reports one, laid out as `coef()` reports it — **as of the last solve**,
  the meaning `coef()` already has; the exact solve on the closed Gram is
  `po.gram.solve(po.gram.from_row(row))`, one line, and the docstring says
  so. `eig_vals` list[f64] (r, descending) and `eig_vecs` list[f64]
  (`r × k` row-major) for `ew_cov(pca=r)`: `Pca::of` on the row's own
  `comoments` with `prev` = the last closed row's vectors for the same
  (spec, instance), kept in a `BankFile` map key so continuity survives a
  save/load; `Pca`'s own first-refresh sign rule. Marginal block: every
  column of `marginal()` as a list in `marginal()` order, prefixed `pair_`.
  `rcov` block: task 50's. The other kinds (`rls`, `kalman`, `holt`,
  `kmeans`, `micro`, `ew_class`, `seqtest`, `sgd`, `pa`, `ftrl`, `robust`)
  emit the common columns and `coef` where they have one; `group_close` is
  still worth having on them for the memory bound.
- *One builder for the row and for `gram()`.* Lift the `match model` in
  `Bank::gram` into `fn gram_of(key, label, model) -> Option<Gram>` and call
  it from both; the acceptance equality then holds by construction and the
  test is the tripwire that keeps it so.
- *Sidecar.* `RunConfig.closed_groups: Option<PathBuf>`, format from the
  extension as `output`'s is, CSV flattening as `output`'s; refused with
  `predict`, and refused when no spec has `group_close` ("closed_groups
  names a file but no spec closes groups" — the file would be empty by
  construction). Written atomically through a second writer thread on the
  `write_file` path — temp sibling, rename, published only when the run
  completes, before `save_state`. It was first written once, at the end
  (E35's rule, one code path); since the code review's P5 (2026-09-15) the
  bank is drained after every chunk and each drain goes to that writer as
  it comes, so the rows no longer wait in the bank for the run.
  `output=None` with `closed_groups=` is
  the accumulate-only pass. A run in which nothing closed writes an empty
  frame with the schema, as the empty-output rule does. The IO plugin
  (`lf.online.fit_predict(closed_groups=path)`) drains `closed_groups()`
  after every chunk into a list and writes the one file when the source
  reaches the last row, under `save_state`'s rules and caveats
  (`_frame.py`, `docs/STATE-WORKFLOW.md`). CLI: `--closed-groups PATH`.
- *Python.* `group_close` is `CommonKwargs`-only, beside `group`; the
  expression namespace has no groups and does not take it (a test says so).
  `_common` writes `None` when unset, `fill_defaults` writes nothing, so a
  TOML spec and a Python spec serialise to the same bytes.
  `po.gram.from_row(row)` takes a one-row frame or a dict, expands the
  `vech` and the row-major lists, and returns a `gram()` dict that
  round-trips through `solve`, `correlation`, `subset`, `coef_stats`; the
  test is `from_row(closed_row) == bank.gram(...)[i]` field for field.
- *Tests, beyond §10's list.* `A, B, A` in one chunk refused naming row 2;
  `"10"` after `"9"` accepted on an integer column and closed in the order
  9, 10; a Categorical key closed bytewise; save/load mid-stream then a
  smaller key refused from the loaded bank; `closed_groups(drop=False)`
  twice returns the same frame; the union schema with an `ew_cov` and a
  `marginal` spec in one bank; the sidecar equal to `pl.concat` of the
  driver's `closed_groups()` frames; `predict=True` with `closed_groups`
  refused in all three entry points.

*Task 45 as built, 2026-09-06.* The design above stood; five things were
decided at the keyboard and are here so the next reader does not re-derive
them.

- *`group_close = "session"` replaces `session_gap`, it does not sit beside
  it.* `clock_cfg()` has always required `session_gap` whenever `session` is
  given -- a session change has to say what the delta is -- and the block
  above refuses the pair. Both cannot hold, so the close is now itself the
  prescription: `session_gap` is required for a session column *unless* the
  spec closes on it, where the answer is "emit the span and start over".
- *The session split lives in `process`, not in `process_chunk`.* The bank
  already cuts a stream's rows into cache-sized runs, each with its own
  `ChunkOut`; a session segment is one more cut of the same kind, so
  `Stream::process_chunk` is untouched and the buffers, the assembly and the
  `last`-row coefficient report all keep working as they did. The boundaries
  come from `ClockState::prev_session()` and the session hashes the columns
  already carry, walked in row order -- a property of two consecutive rows,
  which is what makes the split chunk-invariant.
- *The PCA is computed when the row is queued, not when the group closes.*
  Streams close in parallel, and the loadings are signed for continuity with
  the previous closed row of the same (spec, instance) -- so "previous" has
  to mean previous in the queue, or the signs would depend on which thread
  got there first. `Bank::queue_closed` sorts by `(closing row, spec, key,
  instance)` and then fills `eig_vals`/`eig_vecs` in that order.
- *`from_row` mirrors the packed half, and the mirror is not bit-exact.* The
  row carries `vech` of the upper triangle, half the bytes; `po.gram.from_row`
  reflects it. The reflected lower triangle differs from `gram()`'s in the
  last bit or so, because the accumulator adds the same two products in the
  opposite order for `C[i][j]` and `C[j][i]` and IEEE multiplication of a
  three-term product does not commute -- E48's finding, in the same library.
  The acceptance therefore holds bit for bit on the half the row carries and
  to `1e-15` on the reflection, and the docstring says so. Storing the full
  `k*k` would make the mirror exact at twice the size; the half was chosen.
- *`rows_fed` and `rows_learned` are nullable.* They come from the stream's
  `DataSummary`, which is `None` for a stream restored from a file written
  before task 35. Null is the true answer there; a zero would read as a
  count.

*Task 46 — `deco` (E55).*

- *Standardisation is `kalman(standardize=True)`'s.* One `EwDiag` over the
  features, decayed on the model's clock; the row is standardised against
  the **pre-row** means and variances (`r_i = (x_i − m_i)/√v_i`), then the
  diag is updated. `u` is NaN until every feature has a positive pre-row
  variance; a zero-variance feature thereafter is NaN for that row (no
  substitution of 1, unlike `kalman`, whose coefficient scale is what that
  rule protects).
- *`u` and its block form.* With `S₁ = Σᵢ rᵢ`, `S₂ = Σᵢ rᵢ²` over `n`
  features: `u = (S₁² − S₂)/((n − 1)·S₂)`, Lemma 2.3's
  `Σ_{i≠j} rᵢrⱼ/((n−1)Σrᵢ²)`, in `(−1/(n−1), 1)` except at `S₂ = 0`
  (NaN). Blocks: `u` is the ratio of the mean off-diagonal product to the
  mean squared entry, so within block `A`: `u_A = (S₁ₐ² − S₂ₐ)/((n_A −
  1)·S₂ₐ)`, and between `A` and `B`: `u_AB = S₁ₐ·S₁_B / √(n_A·n_B·S₂ₐ·S₂_B)`
  — the same ratio with the cross term `(Σ_A r)(Σ_B r)/(n_A n_B)` over the
  geometric mean of the two blocks' mean squares, in `[−1, 1]` by
  Cauchy–Schwarz. A block of one feature has no within term: refuse blocks
  of size `< 2` by name. Every feature must be in exactly one block.
  ANSWERS confirms the closed forms against Engle–Kelly (2012), and two
  things the paper says that the docstring repeats: `uₜ` is a downward
  biased estimate of the equicorrelation (their §2.2), and an alternative
  `u^var = 1 − (1/(n−1))·Σᵢ(rᵢ − r̄)²` exists — the variance-of-standardised
  returns form — which this task does **not** offer (one estimator, the
  Lemma 2.3 one, until a use asks for the other).
- *Dynamics.* `"ew"`: `rho' = a·rho + b·u` with `a = λW/W'`, `b = w/W'`,
  `W' = λW + w` — `EwCov::update`'s mean form; at `W = 0` that is `rho = u`;
  `W' = 0` (a zero-weight first row) skips. `rho` is NaN before its first
  update. `"linear"` (LDECO eq. 21 with correlation targeting): `rho' = (1 −
  α − β)·rho_bar' + α·u + β·rho`, where `rho_bar` is the `"ew"` recursion run
  alongside as the target and `rho` starts at the first `u`; `α, β ≥ 0`,
  `α + β < 1`, refused otherwise and refused under `"ew"`. `halflife`/`lam`
  are required under both (the standardiser and `rho_bar` decay on them).
  Reported `rho` is the level **before** the row. Two departures from the
  paper, recorded here so nobody "fixes" them back (ANSWERS, verified
  against Engle–Kelly): eq. 21 has a free intercept `ω`, and the paper
  applies correlation targeting to the DECO-DCC `Q` recursion (its eq. 5),
  not to the linear one — writing the intercept as `(1 − α − β)·rho_bar`
  is *our* reparameterisation, chosen because it removes a free parameter
  that has no sample to be fitted on in a streaming model; and the paper
  lets `α + β` sit slightly above 1 under numerical bounds, where our `α +
  β < 1` is the stricter, stationary choice. The docstring says both.
- *`loglik`.* The row's Gaussian log-density under the pre-row `rho`, in
  standardised coordinates: `−½·[n·ln 2π + ln det R + r'R⁻¹r]` with `det R
  = (1−ρ)^{n−1}(1 + (n−1)ρ)` and `r'R⁻¹r = (S₂ − ρ·S₁²/(1 + (n−1)ρ))/(1−ρ)`.
  `ρ` is clamped into `(−1/(n−1) + 1e-9, 1 − 1e-9)` **for the density
  only**; the emitted `rho` is not clamped. With blocks, `R = D + U·P·U'`
  (`D = diag(1 − ρ_{A(i)A(i)})`, `U` the `n × K` block indicator, `P` the
  `K × K` matrix of block correlations): Woodbury and the matrix
  determinant lemma give `R⁻¹r` and `ln det R` from a `K × K` solve, `O(n +
  K³)`; the test is a dense `numpy.linalg.slogdet`/`solve`. NaN whenever
  `rho` is.
- *Outputs and state.* `u`, `rho`, `loglik`, `n_eff`; with blocks `u_<A>`
  per block then `u_<A>_<B>` per pair in `blocks` order, `rho_*` likewise,
  one `loglik`. `coef` = the `rho` values in that order, one slot per
  value, term `rho` (`coef_fields` gets a `Deco` arm). `ModelState::Deco {
  diag, rho: Vec<f64>, rho_bar: Vec<f64>, w_sum, q_sum, dynamics, alpha,
  beta, blocks }`, appended. `n_outputs()` = `3` unblocked, `2·(K +
  K(K−1)/2) + 1` blocked.
- *Tests.* §10's, plus: `rho` under `"ew"` with one pair against
  `ew_cov(stats=["corr"])` on the columns standardised by the same
  `EwDiag` **to the bit** — same `a`, `b`, same order of operations, or the
  test says which differs; the clock-rescaling test; `MINIMAL["deco"] =
  {"features": ["x0", "x1"]}` in `test_model_registry`.

*Task 46 as built, 2026-09-06.* The design stood; four corrections and one
departure, all measured.

- **The plan's `ew_cov` equality is false.** It said a one-pair `"ew"`
  `deco` equals `ew_cov(stats = ["corr"])` on the same standardised columns
  *to the bit*. It does not, and not by a little: on a two-column stream
  with a true correlation of 0.6, `rho` settles near **0.32** and `corr` near
  **0.60**. The EW mean of a ratio is not the ratio of EW means, and `u` is
  the downward biased estimator Engle & Kelly themselves flag (`E[u]` is
  0.20 at a true 0.30 with six columns, 0.60 at a true 0.80 — measured by
  Monte Carlo, and reproduced by the model to 0.02). What *is* bit-identical,
  and is what the test asserts, is `rho` against an `ew_cov(stats = ["mean"])`
  over the `u` sequence at the same halflife.
- *Which is why the recursion is written `ρ + b·(u − ρ)`* rather than the
  algebraically equal `a·ρ + b·u`: that is the order `EwCov::update` writes
  its mean in, and the two differ in the last bit. Written the other way the
  agreement above was 7e-16, not exact.
- *One code path for blocked and unblocked.* The unblocked model is one block
  holding every feature, and the `K × K` Woodbury form reduces to the closed
  `(1−ρ)^{n−1}(1+(n−1)ρ)` at `K = 1` — a unit test asserts that reduction, so
  there is one implementation of the density and two tests of it. A single
  *named* block is the unblocked model under a name: same three slots,
  labelled `u_<A>`, `rho_<A>`, `loglik`.
- *One shared weight for the level.* `rho`'s accumulated weight is not the
  standardiser's: a row whose `u` is not finite (a zero-variance feature)
  teaches the level nothing and must not decay it either. All the values of
  a blocked model advance together, on that one weight; a row where any
  block's `u` is undefined moves none of them.
- **Departure: `label_delay` is accepted, not refused.** The batch-wide rule
  says the new no-target models refuse it, "nothing to hold back". `ew_cov`,
  whose shape `deco` copies, accepts it today, and it does hold something
  back — the accumulator update. Refusing it for one and not the other would
  be a surprise, so `deco` accepts it and a test says so. The same reasoning
  will apply to 50, 53, 54 and 55 unless one of them has a real reason to
  differ.

*Task 47 — the ring-clearing signal.*

- *What is missing.* `ClockAdvance` has `d_clock`, `reset`, `accepted`,
  `backwards`, `session_changed` — nothing says the cap was hit — and a
  model is told nothing about a session change unless it is `EwRidge`
  (`AnyModel::blend_toward_long_run`). §10's rule "cleared by a
  `max_dclock` breach or a session change" therefore has no carrier.
- *The carrier.* `ClockAdvance::capped: bool`, true iff `max_dclock` is set
  and the gap-adjusted raw delta (pending time folded in) exceeded it, so
  `d_clock == max_dclock` was substituted; set in `ClockState::advance`,
  never on a row-count clock. `RowPlan::capped` copies it (false on a
  replayed `label_delay` row, as `session_changed` is). `OnlineModel::
  clear_lags(&mut self) {}` — a default no-op every model inherits; the
  stream calls `inst.model.clear_lags()` in pass 2 for `plan.session_changed
  || plan.capped` when `!plan.reset` (a reset already rebuilds the instance,
  ring included). Not called from `predict_chunk`, which mutates nothing.
- *The rule, stated once.* Row-lagged state is a function of the learned
  rows of the group since the last reset, session change or capped gap, in
  order; it is chunk-invariant because all three are properties of the row
  sequence. EXTENDING.md gains the step: "if the model keeps row-lagged
  state, override `clear_lags` and add the three-event test"; the
  `model_contract.rs` probe gains a `clear_lags` call on every model,
  asserting `predict` unchanged for the models that do not override it.

*Task 47 as built, 2026-09-06.* Three decisions worth keeping.

- *`capped` is per row, not per accumulated gap.* Skipped rows fold their
  time into the next accepted row's delta, and that total can exceed
  `max_dclock` without any single jump doing so. It is the one row's jump
  that breaks adjacency, so that is what the flag reports.
- *`on_clock_reset = "max"` counts as capped.* The policy's answer to a
  backwards clock is "as far apart as they can be", which is the ceiling:
  the rows are not adjacent, and a ring built on them is wrong. `"zero"`,
  `"reset_state"` and `"error"` do not set it — the first says the rows *are*
  adjacent, the second rebuilds, the third refuses the chunk.
- *`predict_chunk` never clears.* Scoring hands each row a copy of the model
  in the state a learning stream would have reached; a copy that cleared its
  lags would answer for a stream that had already learned the row. `predict`
  moves nothing, the ring included.

The call site is in `run_instance`, before the `accept` test (a skipped
row's gap breaks adjacency too) and under the `else` of `plan.reset`. No
model keeps a ring yet, so the behavioural test is task 48's; what ships here
is the flag with its unit tests and the contract check that `clear_lags` is a
no-op for every model that has declared no ring (`KEEPS_LAGS`, empty today).

*Task 48 — lagged co-moments on `ew_cov` (E56).*

- *Parameters.* `lags: Option<Vec<usize>>` on `ModelKind::EwCov`, strictly
  increasing and `≥ 1`, refused otherwise (the list order is the output
  order, so it is not sorted in silence). `"lagcorr"` in `stats` requires
  `lags`; `lags` without `"lagcorr"` accumulates only.
- *Update.* `EwCovModel` owns an `EwLagCov { lags, ring: VecDeque<Vec<f64>>
  (capacity max lag), c: Vec<f64> (L × k × k) }`. Per learned row, before
  `cov.update`: `W = cov.n_eff()`, `W' = λW + w`, `a = λW/W'`, `b = w/W'` —
  the same expressions as `EwCov::update`, on the same operands, so the
  same bits — and `d_t = x − m_old` with `m_old = cov.means()`. For each
  `ℓ`: if the ring holds `x_{t−ℓ}`, `C_ℓ ← a·C_ℓ + a·b·d_t·(x_{t−ℓ} −
  m_old)'`; otherwise `C_ℓ ← a·C_ℓ` (`EwAutoCorr`'s rule for a lag not yet
  seen). Then `cov.update`, then push `x` onto the ring. A row enters the
  ring iff it entered `cov` (the zero-weight and skipped rows that only
  decay do not); when `cov` decays without a row, `c` decays by the same
  factor. `ℓ = 0` would be `comoments` exactly, and the test that says so
  is the guard on the shared arithmetic.
- *Clearing.* `clear_lags` empties the ring and nothing else: the ring is
  the row memory, `c` is a decayed statistic and keeps decaying.
- *Outputs.* `lagcorr_<a>_<b>_l<ℓ>` = `C_ℓ[a,b] / √(C₀[a,a]·C₀[b,b])`, NaN
  when undefined, emitted at `"lagcorr"`'s position in `stats`: for each
  lag, for each `a`, for each `b` (both orientations and the auto terms,
  `k²` per lag). `EwCovModel::labels` and the `ew_cov` arm of
  `output_index` walk it the same way (`columns = [a, b]`, a new
  `FieldMeta.lag: Option<usize>`).
- *State and export.* `ModelState::EwCovModel` gains `#[serde(default,
  skip_serializing_if = "Option::is_none")] lags: Option<EwLagState>`;
  legacy states read `None` and the spec's `lags` starts a fresh ring.
  `Gram` gains `lags: Option<Vec<usize>>`, `lag_comoments: Option<Vec<f64>>`;
  the Python dict gains the same keys (`None` without lags); `po.gram.merge`
  sets them `None` and says why (a ring boundary is not mergeable
  Chan-style); `subset` slices them. The E54 row's `lags`/`lag_comoments`
  columns come from here.
- *Tests.* A longhand `numpy` implementation of the recursion above (not of
  some other definition of an EW lagged covariance: this one centres both
  legs at the current pre-row mean, which is the choice that makes `ℓ = 0`
  coincide); with `lam = 1`, unit weights and `ℓ = 0`, `numpy.cov(ddof=0)`;
  the three clearing events each empty the ring and a fourth non-event (a
  gap just under `max_dclock`) does not; a spec with `lags` and one without
  give bit-identical `n_eff`, `n_kish`, `means`, `comoments`; chunk
  invariance; the sweeps.

*Task 48 as built, 2026-09-06.* The design stood. Four notes.

- *The `vech` stops at the contemporaneous matrix.* The closed row packs
  `comoments` as the upper triangle (task 45) because that matrix is
  symmetric to 1e-16; a **lagged** matrix is not symmetric at all, so
  `lag_comoments` is the full `L·k·k`, and `po.gram.from_row` reshapes
  rather than reflecting.
- *`ew_cov` is not in `KEEPS_LAGS`.* The contract probe builds it without
  lags, so `clear_lags` is a no-op there and the check still holds it to
  "the bytes did not move". The behavioural test lives in
  `tests/test_ew_cov.py`, where the ring is exercised through the stream --
  which is the only place the three clearing events exist.
- *The clearing test had to isolate the cap from the decay.* A longer gap
  decays the state more, so "gap of 4 versus no gap" differs in the 11th
  digit whatever the ring does. At `halflife = inf` the factor is exactly
  1, so a gap changes *nothing* but the ring, and the test runs there: a gap
  of 4 under a ceiling of 5 is bit-identical to no gap, a gap of 50 under
  the same ceiling is not, and the same gap of 50 under a ceiling of 100 is
  bit-identical again. That last case is what separates "capped" from
  "long".
- *pyo3 converts tuples up to twelve elements* and `GramRow` was already
  twelve, so the lag block rides as a nested `(lags, comoments)` pair beside
  it rather than as two more slots.

*Task 49 — `po.prep.refresh_time` (E58).*

- *A Rust operator, not a Python generator.* The recursion `τ_{j+1} = max_i
  (first tick of series i after τ_j)` is a sequential scan no window
  expression writes, and a per-row Python loop over ticks is the cost this
  library exists to avoid. So: `crates/online-polars/src/refresh.rs`,
  `RefreshTime::new(names, pairs)`, `feed(&DataFrame) -> PolarsResult<
  DataFrame>`, state per `by` key (a `HashMap<GroupKey, _>` as the bank
  keeps) of last time, last value, ticks since the last refresh and an
  updated flag per series — `O(m)`, or `O(m²)` pairwise; exposed through
  `online-py` as a class with `feed`, and wrapped in `prep.py` as a lazy
  IO-plugin source the way `_frame.py` wraps the bank, so the result is a
  `LazyFrame` that streams and honours the projection, predicate and slice
  polars pushes into a Python source.
- *Signature.* `refresh_time(lf, *, series, names, time, value, by=None,
  pairs=False, keep=()) -> pl.LazyFrame`. `names` is new against §10 and
  required: the output columns `<s>_value` are the schema a lazy plan must
  declare before a row is read, so the set of series cannot be discovered
  from the data. A row whose `series` is not in `names` is an error naming
  it (dropping it would hide a misspelling). Long input only: §10's "wide
  frame already on a common grid" is already synchronised, and the long
  form is one `unpivot` away — the docstring gives that line.
- *Semantics.* Rows must be in `time` order within `by`; a backwards time
  is an error naming the row. A null `value` is not an update. A grid point
  is emitted at the tick that completes the set — every series updated
  since the previous point — with `time_refresh` = that tick's time (=
  max over series of their last update, Definition 1), `<s>_value` = each
  series' last value, `n_obs_<s>` = ticks of `s` since the previous point,
  `retained_fraction` = `m / Σ_s n_obs_s` for the interval, and `keep`
  columns at their value on the completing tick. Then every flag clears
  and the counts restart. The docstring carries BNHLS §2.1's caveat
  (ANSWERS): the refresh vector is treated as observed at `time_refresh`,
  though each series' value is stale by up to one of its own inter-tick
  intervals — the price of a common grid, and why `n_obs_<s>` is emitted
  (a large count on one series is that series' staleness made visible).
  `pairs=True` runs an independent two-series state
  per pair and emits the long frame `(by?, pair, time_refresh, a_value,
  b_value, n_obs_a, n_obs_b)` with `pair = "a|b"` in `names` order.
- *Tests.* A longhand Python loop over the same rows on random Poisson
  streams (the oracle), with `by` and with `pairs`; §10's three-series
  example — `n = 8, 9, 10` ticks giving `N = 7` and `21/27` retained, which
  ANSWERS verified against BNHLS but whose tick times exist only as the
  paper's figure — so the test is a constructed three-series stream with
  those tick counts that the loop reduces to exactly `N = 7`, asserting
  `N ≤ min nᵢ` and `retained = N·m/Σnᵢ = 21/27`; a synchronous
  input (every series ticks at every time) returned with `n_obs = 1` and
  `retained_fraction = 1` everywhere; a volume clock as `time`; identical
  frames from 1 and 1000 batches; the unknown-series and backwards-time
  errors.

*Task 49 as built, 2026-09-06.* The design stood; three notes.

- *The `N = 7`, `21/27` example is constructed, and says so.* ANSWERS
  verified the numbers against BNHLS, but their tick *times* exist only as a
  figure, so the test builds a 27-tick stream with their counts (8, 9, 10)
  and asserts the reduction: seven points, three values kept each, the
  per-interval fractions `3/5, 3/4, 3/4, 3/4, 3/4, 1, 1`, and `N <= min nᵢ`.
  The first attempt at such a stream gave eight points -- seven clean cycles
  plus a tail that completed one more -- so the extras had to be placed
  *inside* an interval, before its completing tick. That is the property
  worth remembering: a repeat only drops a tick when it precedes the tick
  that closes the set.
- *`retained_fraction` is per interval, not cumulative.* `m / Σ n_obs` for
  that point. The paper's `21/27` is the aggregate, which the test computes
  as `3·N / total`.
- *The slice pushdown counts output rows.* Unlike the bank's source, where
  `head(n)` limits what the *bank is fed*, here the grid is what the query
  sliced: `head(5)` means five grid points, however many ticks that took.

*Task 50 — `rcov` (E57).*

- *Shape.* A group-scoped `OnlineModel` in `rcov.rs` with `n_outputs() = 0`
  (per row, `n_eff` alone — the count of returns in the block so far),
  `n_targets() = 0`, no decay: `halflife`/`lam` refused by name, `group` and
  `group_close` required, `weight` accepted only as `0` or `1` (any other
  value is an error naming the row); its value is the block computed at
  close, `Rcov::estimate(&self) -> RcovEstimate`, which `close_stream` calls
  for the `AnyModel::Rcov` arm and writes into the E54 row's `rcov` block.
  Input rows are **returns** (the caller differences upstream; `refresh_time`
  emits levels and `.diff().over(by)` is the line), which is what every
  formula below is written on. `ModelState::Rcov`, appended.
- *`kind = "plain"`.* `Σⱼ xⱼxⱼ'`, accumulated as it arrives, no ring. Equal
  at close to `n × EwCov::raw` under `lam = 1`, unit weights, to the bit —
  the cross-check §10 names, and the reference the other two kinds are
  measured against.
- *`kind = "kernel"` (BNHLS 2011).* `K = Σ_{h=−H}^{H} k(h/(H+1))·Γ̂_h`,
  `Γ̂_h = Σ_{j=|h|+1}^{n} xⱼx'_{j−h}`, `Γ̂_{−h} = Γ̂_h'`. Parzen: `k(x) = 1 −
  6x² + 6x³` on `[0, ½]`, `2(1−x)³` on `(½, 1]`, `0` beyond — the only
  kernel offered (`kernel="parzen"`; another name is refused). End-point
  jitter with `m = jitter`: `X̃₀ = mean(X₀..X_{m−1})`, `X̃ₙ = mean(X_{n−m+1}
  ..Xₙ)`, so the effective return series is `x̃₁ = X_m − X̃₀` (formed from
  the first `m` raw returns once row `m` is in), the raw `x_{m+1} ..
  x_{n−m}`, and `x̃_end = X̃ₙ − X_{n−m}` (formed at close from the last `m`
  raw returns); `n_eff_returns = n − 2m + 2`. BNHLS §2.2 (ANSWERS,
  verified): `m = 1` is mean-square optimal for their flat-top form and
  `m = 1 … 4` moves the estimate under 0.5 % — so `jitter` defaults to `2`
  as §10 asked and the docstring says the choice is immaterial at that
  scale; their worked `m = 2` is exactly `X̃₀ = ½(X_{τ₀} + X_{τ₁})`, `X̃ₙ =
  ½(X_{τ_{N−1}} + X_{τ_N})`, which the formula above reproduces. **Products
  enter `Γ̂_h` only once both legs are final**: at row `t`, add `x_{t−m}·x'_{t−m−h}` for `h ≤
  max_bandwidth` (the return `m` rows back can no longer be replaced by the end
  jitter); at close, add the products of `x̃_end` with the last `max_bandwidth` final
  returns. That is what makes the state a ring of `max_bandwidth + m` vectors and
  the close `O(max_bandwidth·k²)`, with no retraction. Bandwidth: `bandwidth` an
  integer `H` (then `max_bandwidth = H`, `block_rows` unneeded) or `"auto"`, BNHLS §4.1
  as ANSWERS read it: `c* = ((12)²/0.269)^{1/5} = 3.5134` for Parzen (the
  kernel constant `k''(0)²/∫k² = 12²/0.269`); per feature `ξ̂ᵢ² =
  ω̂ᵢ²/IV̂ᵢ`, with `IV̂ᵢ` the realised variance on a sparse grid
  (`iv_stride`, default 20 rows, averaged over the `iv_stride` offsets —
  their 20-minute RV) and `ω̂ᵢ²` the mean over `q` offsets of
  `RV_dense^{(i)}/(2n^{(i)})` taken on every `q`-th return (`noise_stride`,
  default `1`, i.e. the full dense grid; their `q ≈ 25` trades or `≈ 70`
  quotes for a ~2-minute grid — the estimate is deliberately upward biased,
  which their §4.1 accepts, and a stride above 1 is the caller's call); `Hᵢ
  = c*·ξ̂ᵢ^{4/5}·n^{3/5}`, `H = ⌈mean Hᵢ⌉` clipped to `max_bandwidth`, reported as
  `bandwidth_used`. `"auto"` requires `block_rows`, from which `max_bandwidth` defaults
  to `⌈c*·block_rows^{3/5}⌉` (`ξ̂ = 1`, a noise variance equal to the block's
  integrated variance, beyond which the estimator is not worth having);
  `max_bandwidth` may be given. `block_rows` is a sizing hint, not a limit: a longer
  block runs, clipped, and says so. Parzen is a positive-definite function,
  so `K` is PSD up to rounding (BNHLS: the Bartlett kernel is *not*
  consistent for this estimator, and Parzen's efficiency 0.97 beats the
  quadratic-spectral 0.93 — the reason it is the only kernel offered);
  `psd=True` clips negative eigenvalues at `0` and sets `psd_repaired` when
  it had to.
- *`kind = "preavg"` (CKP 2010; Hautsch–Podolskij 2013 for the constants).*
  `g(x) = min(x, 1−x)`, `Ȳᵢ = Σ_{j=1}^{kₙ−1} g(j/kₙ)·x_{i+j}`; `MRC = n/(n −
  kₙ + 2) · 1/(ψ₂^{kₙ}·kₙ) · Σᵢ ȲᵢȲᵢ' − ψ₁^{kₙ}/(θ²·ψ₂^{kₙ}) · 1/(2n) ·
  Σⱼ xⱼxⱼ'`, `kₙ = ⌊θ·√n⌋` (CKP's floor, ANSWERS — not the ceiling the
  first draft had), with the finite-sample constants as CKP write them:
  `ψ₁^{k} = k·Σ_{i=1}^{k}(g(i/k) − g((i−1)/k))²`, `ψ₂^{k} = (1/k)·
  Σ_{i=1}^{k−1} g(i/k)²` (limits `ψ₁ = 1`, `ψ₂ = 1/12`). The `Φ` constants
  appear only in the asymptotic variance, which the estimator does not
  report, so the estimator does not use them; the test still pins the
  limits `Φ₁₁ = 1/6`, `Φ₁₂ = 1/96`, `Φ₂₂ = 151/80640` against the
  finite-sample forms `φ₁^k(j) = Σ_{i=j+1}^{k−1}(g((i−1)/k) −
  g(i/k))·(g((i−j−1)/k) − g((i−j)/k))`, `φ₂^k(j) = Σ_{i=j+1}^{k−1}
  g(i/k)·g((i−j)/k)`, `Φ₁₁^k = k·(Σⱼ φ₁^k(j)² − ½φ₁^k(0)²)`, `Φ₁₂^k =
  (1/k)·(Σⱼ φ₁^k(j)φ₂^k(j) − ½φ₁^k(0)φ₂^k(0))` (and `Φ₂₂^k` by the same
  pattern), because they are the check that `g` and the `ψ` sums are coded
  as the paper has them. `psd=False` is that balanced, bias-corrected form
  (optimal rate, not guaranteed PSD; clipped and flagged as the kernel is);
  `psd=True` is CKP §3's longer window without the bias term — **the
  exponent and the dropped term are taken from CKP §3 during
  implementation; the `kₙ = ⌈θ·n^{1/2+δ}⌉`, `δ = 0.1` of the first draft is
  Hautsch–Podolskij's reading and stands only if §3 agrees** (the one open
  read in this task; ANSWERS did not cover §3). `kₙ` must be fixed before
  the block starts, so it is `⌊θ·√block_rows⌋` (or the §3 form on `block_rows`) from
  a required `block_rows`, or a given `window`. Streaming: a ring of `kₙ − 1`
  returns; each arriving return completes one `Ȳ`, whose outer product is
  added — `O(kₙ·k + k²)` a row.
- *The row's `rcov` block.* `rcov` list[f64] (vech, as `comoments`), `rcorr`
  list[f64] (vech, unit diagonal), `rcov_n` i64 (effective returns),
  `rcov_kind` str, `bandwidth_used` i64 (null for `plain`/`preavg`),
  `omega2`, `iv_sparse`, `iq` list[f64] per feature — `iq` the realised
  quarticity `(n_s/3)·Σ x_s⁴` on the sparse grid, a proxy and labelled one —
  `psd_repaired` bool. All null when the block is too short (`n < 2m`
  for the kernel, `n < kₙ` for pre-averaging; `n < 2` for `plain`'s
  `rcorr`) — nulls, never a panic.
- *Tests.* `plain` to the bit against `ew_cov(lam=1)`; the kernel and the
  MRC against longhand `numpy` on the same synchronised returns (from task
  49 over task 52's asynchronous rows) to `1e-10` relative; the jittered
  end formed from state alone (a test that feeds the block one row at a
  time and checks the close equals the offline computation on the full
  block — this is the acceptance for "nothing reads a future row"); the
  auto bandwidth against its formula on a block with known `ω²` and `IV`;
  PSD on adversarial streams (a constant series, a series that is one spike,
  `k = 1`); the short-block nulls; a zero-weight row absent from the ring
  (the estimate equals the one from the stream with that row removed);
  chunk invariance; the sweeps, with the residual flags refused.

*Task 50 as built, 2026-09-06.* The design stood; five notes, one of them
still open.

- **CKP §3 was not read, and the source says so** (read 2026-09-28, task
  114: §3.4 agrees, `δ = 0.1` and no bias term). `psd = true`'s longer
  window is `kₙ = ⌈θ·n^{0.6}⌉` with the bias term dropped -- Hautsch &
  Podolskij's reading, as ANSWERS flagged. `RcovCfg::window_for` carries the
  comment; if §3 disagrees, that line and the dropped term move together.
  Everything else in the model is from the papers directly.
- *The jitter bound is a claim about a block, not a handful of rows.* BNHLS
  say `m = 1..4` move the estimate by under 0.5 %; at 300 returns per block
  it is 0.8 %, because the jitter is an end effect. At 1000 it is 0.11 %,
  and the test runs there.
- *A `Point`'s legs must both be final before it enters `Γ̂_h`.* The ring
  keeps `m + 1` raw returns so `x_{t−m}` can be finalised at row `t`; the
  first `m` are kept separately for the leading jittered return, and the
  trailing one is formed at close from the *last* `m` of that ring -- not
  all `m + 1`, which was the first bug the longhand oracle caught.
- *The pre-averaged window starts one row later than the ring fills.* CKP's
  `Ȳᵢ = Σ_{j=1}^{kₙ−1} g(j/kₙ)·x_{i+j}` skips `x_i`, so the window is the
  `kₙ − 1` returns *after* the oldest in a ring of `kₙ`. Filling a ring of
  `kₙ − 1` and reading it whole is off by one window, which the longhand
  oracle also caught.
- *A session close restarts the stream, so a closing spec's live `summary()`
  is the current span's.* `test_summary.py`'s whole-stream oracle skips
  those specs, and `test_closed_groups.py` asserts the other half: the spans
  before the current one are in the closed rows.

*Task 51 — `po.corr` (E62).*

- *Shape.* `python/polars_online/corr.py`, `gram.py`'s pattern exactly: the
  `_np()` import guard with the install hint, pure functions over arrays,
  `gram()` dicts and E54 rows (a helper `_matrix(obj)` accepts an array, a
  dict with `comoments`/`columns`, or a row and returns the `k × k`
  correlation), one longhand check per function in `tests/test_corr.py`.
  Nothing here touches Rust or the models; it lands as `po.corr`, listed in
  `api_surface.txt` and in a Sphinx page beside `po.gram`.
- *`to_z` / `from_z`.* `atanh` / `tanh` with the input clipped at `|ρ| ≤ 1 −
  1e-6` (`z ≈ 7.25`) so a unit off-diagonal from a degenerate block is
  finite; the clip is in the docstring.
- *`nearest(A, W=None, *, tol=1e-8, max_iter=100)`.* Higham (2002),
  Algorithm 3.3 exactly: `ΔS₀ = 0`, `Y₀ = A`; for `k = 1, 2, …`: `R_k =
  Y_{k−1} − ΔS_{k−1}` (Dykstra's correction), `X_k = P_S(R_k)`, `ΔS_k = X_k
  − R_k`, `Y_k = P_U(X_k)`; with diagonal `W`, `P_U` sets the diagonal to 1
  and `P_S(A) = W^{−1/2}·(W^{1/2}AW^{1/2})₊·W^{−1/2}`, `(·)₊` the
  eigenvalue clip at zero. Stop (his test 4.1) when the largest of
  `‖X_k − X_{k−1}‖_∞/‖X_k‖_∞`, `‖Y_k − Y_{k−1}‖_∞/‖Y_k‖_∞`, `‖Y_k −
  X_k‖_∞/‖Y_k‖_∞` is below `tol`. Returns `(X, dist, iters)`, `dist` the
  weighted Frobenius distance. **The partial-eigensolve path §10 mentions
  is dropped**: `numpy.linalg.eigh` is the only eigensolver on the
  dependency list, and the matrices this library produces are small enough
  that a full decomposition per iteration is the cheaper code. Tests, with
  the values ANSWERS verified against the paper (§2 and §4): `A = [[1, 1,
  0], [1, 1, 1], [0, 1, 1]]` (eigenvalues `1 ± √2`, `1`) → `[[1, .7607,
  .1573], [.7607, 1, .7607], [.1573, .7607, 1]]`, `‖A − X‖_F = 0.5278`,
  `X` singular with null vector `[−.4814, .7324, −.4814]` (and `ee'`, the
  obvious guess, is at distance `√2`); `A = tridiag(−1, 2, −1)` of order 4
  → `[[1, −.8084, .1916, .1068], [−.8084, 1, −.6562, .1916], [.1916,
  −.6562, 1, −.8084], [.1068, .1916, −.8084, 1]]`, `‖A − X‖_F = 2.13`,
  rank 3, **19 iterations at `tol = 1e-8`** (linear convergence, a factor
  ≈ 3 per iteration — assert the count, it pins the algorithm and not just
  the fixed point); entries at `1e-4`, distances at `1e-3`. Bounds worth a
  test each: a diagonal `A` gives `I`; a PSD `A` with diagonal `≤ 1` gives
  `A` with its diagonal set to 1; a unit-diagonal `A` with `t` nonpositive
  eigenvalues gives an `X` with at least `t` zero eigenvalues (Theorem
  2.5 — the 4×4 example's rank 3). PSD (smallest eigenvalue `≥ −1e-10`) and
  unit diagonal on random symmetric inputs; a correlation matrix returned
  in one iteration.
- *`shrink(R, target="constant", alpha=None, X=None)`.* Ledoit & Wolf
  (2004), the constant-correlation target: `fᵢᵢ = sᵢᵢ`, `fᵢⱼ = r̄·√(sᵢᵢsⱼⱼ)`
  with `r̄ = 2/((N−1)N)·Σ_{i<j} rᵢⱼ` — on a correlation matrix `sᵢᵢ = 1`
  and `F` is the equicorrelation matrix at `r̄`. The intensity `δ̂* =
  max{0, min{κ̂/T, 1}}`, `κ̂ = (π̂ − ρ̂)/γ̂`, with (their Appendices A–B,
  verified in ANSWERS)

  `π̂ = Σᵢⱼ π̂ᵢⱼ`, `π̂ᵢⱼ = (1/T)·Σₜ ((yᵢₜ − ȳᵢ)(yⱼₜ − ȳⱼ) − sᵢⱼ)²`;
  `ρ̂ = Σᵢ π̂ᵢᵢ + Σ_{i≠j} (r̄/2)·(√(sⱼⱼ/sᵢᵢ)·ϑ̂ᵢᵢ,ᵢⱼ + √(sᵢᵢ/sⱼⱼ)·ϑ̂ⱼⱼ,ᵢⱼ)`,
  `ϑ̂ᵢᵢ,ᵢⱼ = (1/T)·Σₜ ((yᵢₜ − ȳᵢ)² − sᵢᵢ)·((yᵢₜ − ȳᵢ)(yⱼₜ − ȳⱼ) − sᵢⱼ)`;
  `γ̂ = Σᵢⱼ (fᵢⱼ − sᵢⱼ)²`.

  `π̂` and `ϑ̂` are fourth-moment sums over the rows, so the intensity needs
  the row data `X` (`T × k`, the standardised rows) and cannot be formed
  from `R` and `T` — §10's `T=None` becomes `X=None` and the function
  requires `alpha` or `X` (recorded so the next reader does not try to
  recover `π̂` from the matrix). `target="identity"` is the other target
  offered. Tests: a small `X` fixture with `δ̂*` worked longhand from the
  formulae above; `alpha=0` returns `R`, `alpha=1` returns `F`; the result
  is positive definite whenever `F` is.
- *`equicorr(R)` / `equicorr_row(r)` / `equicorr_loglik(r, rho)`.* The mean
  off-diagonal; task 46's `u` on one standardised row; task 46's closed-form
  log-density — all offline, and the pins for the model's own tests.
- *`absorption(R, k)`* = `Σ_{i≤k} λᵢ / Σᵢ λᵢ` (Kritzman et al.), and
  *`shift(ar_fast, ar_slow, *, scale=None)`* = `(ar_fast − ar_slow)/scale`
  elementwise over two aligned arrays, `scale` defaulting to
  `std(ar_slow)` over the sample — the standardised shift as published, with
  the caller choosing the two windows.
- *`spectral(R, r)` / `from_spectral(vals, vecs, *, unit_diag=True)`.* The
  top-`r` eigenpairs descending with `Pca`'s first-refresh sign rule (the
  largest-magnitude entry of each vector positive), and the completion
  `VΛV' + diag(1 − diag(VΛV'))` — the remainder is non-negative because the
  dropped components are PSD, so the result is a correlation matrix.
- *`block_means(R, labels)` / `from_blocks(B, labels)`.* `B[a, b]` the mean
  of `R[i, j]` over `i ∈ a`, `j ∈ b`, excluding `i = j`; returns `(B,
  counts)`; `from_blocks` rebuilds the block-equicorrelation matrix with a
  unit diagonal. `block_means(from_blocks(B)) == B` is the test.
- *`mp_edge(n, m, sigma2=1.0)`* = `σ²(1 + 1/Q ± 2√(1/Q))` = `σ²(1 ±
  1/√Q)²`, `Q = n/m ≥ 1` (Laloux, Cizeau, Bouchaud & Potters 1999,
  verified); the density `ρ(λ) = (Q/2πσ²)·√((λ₊ − λ)(λ − λ₋))/λ` is offered
  as `mp_density(lam, n, m, sigma2)` for a plot. Tests: `Q = 1 → (0, 4σ²)`;
  a random Wishart spectrum lying inside the edges up to a finite-`n`
  margin.
- *`signal_share(z_blocks, n_eff_blocks)`.* Per pair over `B` blocks: `clip(1
  − mean_b(1/(n_b − 3)) / var_b(z_b), 0, 1)` — the share of the between-block
  variance of Fisher-z that the sampling floor does not explain; `1-D`
  input returns a scalar.
- *`loss(fcst, real, kind, *, mu=None, n=None)`.* `"qlike"`: `tr(R̂⁻¹R) −
  log det(R̂⁻¹R) − k`, which is §10's `log|R̂| + tr(R̂⁻¹R)` shifted by the
  forecast-free constant `log|R| + k` so that it is `≥ 0` with equality iff
  `R̂ = R` (the test); `"z_mse"`: `Σ_{i<j} (ẑᵢⱼ − zᵢⱼ)² · (n − 3)` with `n`
  required; `"minvar"`: `w'Rw` with `w = R̂⁻¹μ / (μ'R̂⁻¹μ)`, `μ` defaulting
  to ones (Engle–Colacito, minimised over `R̂` at `R̂ = R` — a second test).
- *`epps_invert(gram_or_row, *, L)`.* Tóth–Kertész eq. 12 exactly
  (ANSWERS, verified): the correlation at scale `L` rows from the lagged
  co-moments at scale 1 carries **triangular** weights on the numerator
  *and* both denominators, and both orientations of the cross term —

  `ρ̂_L[a, b] = Σ_{x=−(L−1)}^{L−1} (L − |x|)·C_x[a, b] / √(Σ_x (L −
  |x|)·C_x[a, a] · Σ_x (L − |x|)·C_x[b, b])`, with `C_{−x}[a, b] = C_x[b, a]`,

  i.e. numerator `L·C₀[a, b] + Σ_{ℓ=1}^{L−1} (L − ℓ)·(C_ℓ[a, b] + C_ℓ[b,
  a])` and auto terms `L·C₀[a, a] + 2·Σ_{ℓ=1}^{L−1} (L − ℓ)·C_ℓ[a, a]`.
  The flat sum the first draft had is the `L → ∞` limit and is dropped;
  `L` is required, and the input must carry every lag `1 … L−1` (the
  docstring says `lags=list(range(1, L))`; a missing lag is an error naming
  it). Test: a simulated pair where `b` is `a` delayed by two rows — the
  scale-1 correlation near zero, `ρ̂_L` at `L = 8` near the truth — and the
  identity at `L = 1` (the plain correlation).
- *`fisher_se(n, rho=None, phi_a=None, phi_b=None)`.* `1/√(n − 3)` is the
  standard error of `z`; with `rho` the delta-method error of `ρ` itself,
  `(1 − ρ²)/√(n − 3)`; with both `phi` the inflation `√((1 + φₐφᵦ)/(1 −
  φₐφᵦ))` for an AR(1) pair — Bartlett's `Var(r) ≈ (1/n)·Σₖ ρₐ(k)ρᵦ(k)`
  under zero true cross-correlation, which for two AR(1)s sums to `(1 +
  φₐφᵦ)/(1 − φₐφᵦ)` (the geometric series `1 + 2·Σ_{k≥1}(φₐφᵦ)ᵏ`; a test
  asserts the closed form against the partial sum). The docstring says
  both assumptions — zero true correlation, linear dependence — and that
  the factor is invalid under ARCH-type innovations. One `phi` without the
  other is an error.

*Task 51 as built, 2026-09-06.* The design stood, and Higham's published
values came out exactly: the 3×3's `[.7607, .1573]` and distance 0.5278 with
a singular result whose null vector is `[−.4814, .7324, −.4814]`, the 4×4's
entries, distance 2.13 and rank 3. Four notes.

- *The iteration count is 20, not 19.* Same run, same fixed point; the
  difference is bookkeeping — this counts the iteration in which the
  convergence test passed. The count is still asserted, because it pins the
  algorithm and not just where it lands, and the test says which convention
  it is.
- *`nearest` returns `Y`, the algorithm's own answer.* `Y_k` has an
  **exactly** unit diagonal and is PSD only to `tol` — the iteration
  converges to the boundary of the cone, and `tol` is how close. Polishing
  it with one more eigenvalue clip would break the diagonal again, so the
  docstring states the bound and the test asserts `> −1e-8` at `tol =
  1e-10`. Measured over 200 random symmetric inputs, the worst smallest
  eigenvalue was `−9e-9` at `tol = 1e-8`.
- *A one-factor sample shrinks all the way.* The first `shrink` test asked
  for `0 < δ̂* < 1` on an equicorrelated sample and got exactly 1 — correctly,
  because such a sample *is* the constant-correlation target and there is
  nothing in the sample matrix worth keeping over it. The interior case
  needs a target that is wrong, so the test uses two blocks with different
  within-block correlations, and the saturating case became a test of its
  own.
- *`api_surface.txt` gained a `[helper modules]` section.* `po.corr`'s
  function names are API and nothing pinned them; the section lists
  `corr`, `eval`, `gram` and `prep`'s `__all__`, which pins all four.

*Task 52 — `po.sim.regimes` (E64).*

- *Shape.* `python/polars_online/sim.py`, numpy only, `np.random.default_
  rng(seed)` for every draw, returning `{"rows", "truth_rows",
  "truth_blocks"}` as `pl.DataFrame`s. Everything after `m` is keyword-only
  (§10's positional list puts required names after defaulted ones):
  `regimes(m, *, states, transition, n_blocks, rows_per_block,
  durations=None, design="step", smooth_rows=0, phi=0.0, scale_state=0.0,
  async_rates=None, noise=0.0, cycle_profile=None, cycle_rows=None,
  activity=None, seed=0)`.
- *The chain.* `states` is a list of `K` correlation matrices (`m × m`,
  checked for a unit diagonal and PSD) or `K` floats, each an
  equicorrelation; `transition` is `K × K` row-stochastic and drives one
  state per block; `durations` (per state, in blocks) makes the sojourn
  deterministic and uses `transition` with its diagonal removed to pick the
  next state — the recurring-state design. `design="step"` switches at the
  block boundary; `design="smooth"` interpolates the correlation matrix
  linearly over `smooth_rows` rows around it (a convex combination of two
  correlation matrices is one).
- *The series.* Latent returns `εₜ ~ N(0, Rₜ)` by Cholesky per distinct
  `Rₜ` (cached — one factorisation per state under `"step"`), `yᵢₜ = φᵢ
  yᵢ,ₜ₋₁ + εᵢₜ` with `phi` a scalar or per-series list (the documented
  truth is the innovation correlation; the AR filter moves the return
  correlation of unequal-`φ` pairs, which is what task 51's `fisher_se`
  inflation is for), scaled by `exp(scale_state · sₜ)` — a scalar `scale_state`
  makes volatility rise with the state index, a list gives it per state.
  `cycle_profile`, a list of `cycle_rows` multipliers in `(0, 1]`, scales the
  off-diagonal of `Rₜ` at bar `t mod cycle_rows` (a mix toward the
  identity, so PSD is kept); `cycle_rows` defaults to `rows_per_block`;
  `session = t // cycle_rows`. `volume`, `(mean, shape)`, draws a Gamma
  volume per bar with the mean scaled by the same `exp(scale_state · sₜ)`;
  `clock` is cumulative volume when `volume` is given and `t` otherwise.
- *Observation.* `xᵢ` are **levels** — the cumulative sum of the latent
  returns plus `noise · N(0, 1)` i.i.d. per observed bar (microstructure
  noise on the level, so the observed return is an MA(1) as the literature
  models it) — because levels are what `refresh_time` and then `.diff()`
  expect. Under `async_rates` (per-series expected ticks per bar) a bar with
  `Poisson(rateᵢ) = 0` ticks carries `null` for `xᵢ` — previous-tick
  sampling is the caller's `forward_fill`, so both the sparse and the
  filled forms are one line away, and `unpivot` on the non-null rows is
  `refresh_time`'s long input.
- *Frames.* `rows`: `entity` (a constant string `"sim"`, the join key a
  multi-entity caller would vary), `t`, `clock`, `session`, `x_1 … x_m`,
  `volume` (null when `activity=None`). `truth_rows`: `t`, `block`, `state`,
  `scale_mult`, `mix` (the `"smooth"` fraction, `0` otherwise). `truth_blocks`:
  `block`, `state`, `n_rows`, `corr` (vech of the block's mean true
  correlation — the state's matrix under `"step"`).
- *Tests.* Per-state matrices recovered from the block sample correlations,
  pooled over the blocks in each state, within `3·SE` of the Fisher-z noise
  floor; a `phi` recovered from the lag-1 autocorrelation; the Epps curve of
  an asynchronous simulation — the sample correlation of previous-tick
  returns at sampling intervals `1, 5, 20, 100` rows — increasing toward the
  truth; `noise > 0` giving a negative lag-1 return autocorrelation; every
  `Rₜ` under `"smooth"` PSD; `session` and `clock` consistent with
  `cycle_rows` and `volume`; two calls with the same seed byte-identical
  (`DataFrame.equals` on all three frames); schema and lengths fixed.

*Task 52 as built, 2026-09-06.* The design stood. Three notes.

- *The documented truth is the innovation correlation, and the tests say
  so.* An AR filter moves the *return* correlation of a pair with unequal
  `φ`; the per-state recovery test therefore runs at `phi = 0`, and `phi`
  gets its own test through the lag-1 autocorrelation. Task 51's
  `fisher_se` inflation is the tool for the other case.
- *Noise goes on the level, not the return.* That is what the literature
  models and what makes the observed return an MA(1) with a negative first
  autocorrelation -- measured at `−0.2` or below at `noise = 2` against
  `|ρ₁| < 0.03` clean, which is the microstructure effect `rcov` undoes.
- *The Cholesky factor is cached by `(state, previous state, cycle_profile bar,
  mix)`.* Under `"step"` with no cycle_profile that is one factorisation per
  state, which is what the row asked for; under `"smooth"` the mix is part
  of the key, so a ramp of `smooth_rows` costs that many factorisations and
  no more.

*Task 53 — `hmm` (E60).*

- *The filter.* Before the row: `p̃ₗ = Σₖ pₖ Πₖₗ` from the filtered `p`
  left by the previous row (uniform `1/K` before the first). Emitted from
  that alone: `p_<k>` = `p` (the filtered posterior before the row), `p1_<k>`
  = `p̃`, `state = argmax p̃` (first maximum wins, `i32` through
  `Source::Cluster`). Then, with the *pre-row* parameters, `fₗ = N(x | μₗ,
  Σₗ + rₗI)` through the `quad_forms_logdet` + softmax path `ew_class`
  already uses — the same `precision_prior` ridge with the same decaying
  scale, and `precision_prior` **required** as it is there (§10's `None`
  default is not an option: a state's centred co-moments start at zero) —
  `loglik = ln Σₗ p̃ₗ fₗ` (a row output computed from the row, as `kmeans`'s
  `dist` is), and `pₗ ← p̃ₗ fₗ / Σ`.
- *Learning.* Each state's `EwCov` (or `EwDiag` under `covariance="diag"`)
  takes the row at weight `w · pₗ` — the responsibilities sum to `w`, so the
  model-level `n_eff` is the shared recursion unchanged. **Π is not §10's
  `EW mean of pₖ(t−1)·pₗ(t)/pₖ(t−1)`**: that ratio is `pₗ(t)`, whose mean
  does not depend on `k` and cannot identify a transition matrix. The
  quantity that does is the filtered joint of consecutive states, `ξₖₗ =
  pₖ(t−1)·Πₖₗ·fₗ / Σ_{k'l'} p_{k'}(t−1)·Π_{k'l'}·f_{l'}`; the counts `Aₖₗ ←
  decay·Aₖₗ + w·ξₖₗ` decay on the clock like every accumulator, and `Πₖₗ =
  (Aₖₗ + τ)/Σₗ(Aₖₗ + τ)` with `transition_prior = τ` a Dirichlet
  pseudo-count per cell (default `1.0`; it is what keeps a never-visited
  row of Π a distribution). `transition` given seeds `A` at `τ·K·Π₀`, so the
  given matrix is the prior mean; `None` means uniform, which is what makes
  the reduction test below exact. `exog_tvtp=col` reads one more column
  (declared like `weight` and `clock`, not a feature) and sets `Πₖₗ(t) =
  softmaxₗ(Aₖₗ + Bₖₗ·xcol,t)` from the fixed `tvtp_coef = (A, B)`; the
  count-based learning of Π is off under it (`A`, `B` are not estimated
  here — a fitted-elsewhere form, like `transition` with `learn=False`).
- *Seeding.* `means`/`covs` given: no warm-up. Else `kmeans`'s recipe on a
  buffer of `warm_rows` learned rows — `seed_centres` (`pub(crate)`, same
  crate, no visibility change) under `seed_rule`, then the buffer replayed
  through the frozen seeds as hard assignments to initialise each state's
  moments — and, as in `kmeans`, every output is null until seeded; the
  buffer is state, and its rows carry their weights and decay as `kmeans`'s
  do. `learn=False` with nothing given is refused: there would be nothing
  to filter with.
- *Outputs and coefficients.* `p_<k>`, `p1_<k>`, `state`, `loglik`, `n_eff`;
  `coef` = the state means in `ew_class`'s layout so `coef()` and
  `last_row()` carry them; `predict` = the step without the step (`p`, `p̃`,
  `state`, `loglik` under the current parameters, nothing moved).
  `ModelState::Hmm {states: Vec<ModelState>, p, a, dynamics…, buffer}`,
  appended. Cost `O(K² + K·d²)` (`O(K² + K·d)` diagonal).
- *Tests.* The reduction: build an `ew_class` and an `hmm` whose states are
  that classifier's own fitted `EwCov` states (`means`/`covs` from
  `ew_class`'s `ModelState`), `transition=None`, `learn=False`, and feed
  both a stream whose `ew_class` labels are balanced — `p` equals the
  classifier's posteriors to the bit, because both run the same
  `quad_forms_logdet` and softmax on the same numbers; a longhand numpy
  Hamilton filter at fixed parameters (`learn=False`) to `1e-12`; recovery of
  Π and the state means on task 52's streams within stated tolerances, as
  the `docs/REGIMES.md` experiment (generated by `scripts/regime_experiments.
  py`, committed, not gate-regenerated — slow); predict parity; a zero-weight
  first row leaving `p` and every state untouched; the clock-rescaling test;
  the sweeps; `MINIMAL["hmm"] = {"k": 2, "features": ["x0", "x1"],
  "precision_prior": 0.1}`.

*Task 53 as built, 2026-09-06 (the model), 2026-09-06 (its experiment).*

- *Π is learned from the filtered joint, not §10's ratio.* Corrected in the
  plan block above before implementation; `EW mean of pₖ(t−1)·pₗ(t)/pₖ(t−1)`
  is `pₗ(t)`, whose mean does not depend on `k`.
- *A given `transition` seeds the Dirichlet prior, not the counts.* Seeding
  the counts at `τ·K·Π₀` gives `(K·Π₀ + 1)/(2K)`, not `Π₀`, and would need
  negative counts for a cell below `1/K`.
- *A normalisation bug the recovery test found.* The filtered joint was
  normalised about `logf[0]`; a state whose density is astronomically small
  overflows that to NaN. It is normalised about the largest density now.
- *The `docs/REGIMES.md` experiment this task's acceptance asked for landed
  with task 56, and it is the finding that matters most about this model:
  **the default seeding cannot find a regime that lives in the
  covariance**. `warm_rows` seeds with k-means over the rows, two zero-mean
  states differ in nothing k-means can see, and the filter splits them by
  direction — 56 % accuracy, which is chance, against 98 % for the same
  filter given `covs`. And a `precision_prior` of 1.0 on data of variance
  1.0 halves every correlation and collapses it to 58 %. Both are in the
  docstring, with the two ways out (pass `means`/`covs`, or give it a
  feature in which the regime is a shift in location).

*Task 54 — `corrchange` (E59).*

- *`kind="monitor"` is WKD's closed-sample test, span by span.* §10's
  sequential form centres on a training-span `ρ̂_train` and needs the
  boundary function and critical values of Wied & Galeano (2013), which
  `docs/ANSWERS-E54-E64.md` could not read (paywalled) and offers no
  constant for. What *can* be pinned against published tables is the
  closed-sample fluctuation test of Wied, Krämer & Dehling (2012), so that
  is what ships: the stream is cut into consecutive spans of `horizon`
  learned rows (`T = horizon`, required; `train` is gone — the test has no
  training span), and at the last row of a span, per pair,

  `Q = max_{2≤j≤T} (j/√T)·|ρ̂ⱼ − ρ̂_T| / D̂`,

  `ρ̂ⱼ` the sample correlation of the span's first `j` rows and `D̂` the
  delta-method long-run standard deviation of `ρ̂` over the span: the five
  raw moments `Uₜ = (x², y², x, y, xy)` centred at their span means, their
  Bartlett-kernel long-run covariance `Σ̂ = (1/T)·Σₜ Σᵤ k((t−u)/γ_T)·VₜVᵤ'`
  with `k(x) = 1 − |x|` and `γ_T = ⌊ln T⌋` — the paper writes `[log T]`
  without a base, and the natural log is the only reading that gives a
  usable bandwidth (`6` at `T = 500`; base 10 gives `2`); exposed as
  `bandwidth=None` meaning that default — mapped to `(σ_x², σ_y², σ_xy)` by
  `D₂` (rows `(1, 0, −2μ_x, 0, 0)`, `(0, 1, 0, −2μ_y, 0)`, `(0, 0, −μ_y,
  −μ_x, 1)`) and to `ρ` by `D₃ = (−½σ_xy σ_y σ_x⁻³, −½σ_xy σ_x σ_y⁻³,
  1/(σ_xσ_y))`: `D̂² = D₃D₂Σ̂D₂'D₃'`. (The paper's `D̂` is the reciprocal and
  multiplies the statistic — the same number.) Under `H₀`, `Q →_d
  sup_{0≤z≤1}|B(z)|`, so `crit` is the Kolmogorov quantile **computed** from
  `P(sup|B| ≤ x) = 1 − 2·Σ_{k≥1} (−1)^{k−1}·exp(−2k²x²)` at `alpha`, not a
  pinned constant — `1.3581` at 5%, `1.6276` at 1%, `1.2239` at 10% are what
  the function must reproduce (a test). Over the pairs `stat` is the
  maximum and `crit` is taken at `alpha/npairs` (Bonferroni; the paper
  leaves the correction open; offering `alpha_adjust="none"` is
  *implementer's call*). The statistic needs `ρ̂ⱼ` for every `j` *and*
  `ρ̂_T`, so the span's rows sit in a ring of `horizon` rows and the
  statistic is one `O(T·k² + T·γ_T·k²)` pass at the span's end — per row,
  only the push. Outputs are null except on a span's last row, where
  `stat`, `crit` and `flag` are written; the delay is at most `horizon`
  rows, which is the price of a known null. `scalar=True` runs the same
  CUSUM on task 46's `u`: `max_j (j/√T)·|ūⱼ − ū_T| / D̂ᵤ`, `D̂ᵤ` the Bartlett
  long-run standard deviation of `u` over the span, on rows an `EwDiag`
  standardises; `halflife`/`lam` are accepted under `scalar=True` only (they
  parametrise that standardiser) and refused by name otherwise, since
  neither kind decays anything else. The sequential Wied–Galeano detector
  stays in §10 as the step after this one, to be taken when its paper has
  been read; it does not block the task.
- *`kind="window"`.* Two adjacent rings of `window` rows; `stat =
  ‖vech(R̂_pre − R̂_post)‖` in ℓ₁ or ℓ∞ over the strict upper triangle, each
  `R̂` the sample correlation of its window (scale-free, so no
  standardiser). `crit` a number, or `"permute"`: **not §10's sign flips** —
  negating a whole row leaves every `xₜxₜ'` and so every correlation matrix
  unchanged, so a sign-flip null has zero spread (ANSWERS confirms). The
  exchangeable null for "the two windows share a distribution" is a
  permutation of the pooled `2·window` rows between the windows: `n_perm`
  draws, the `(1 − alpha)` quantile of the permuted statistics, recomputed
  every `permute_every` rows (each draw is `O(window·k²)` from scratch — the
  reason it is not per row), with `perm_block` (default `1`) permuting
  blocks of consecutive rows so that serially dependent rows do not make
  the null too liberal. A fixed-seed generator in state keeps the draws
  chunk-invariant.
- *Outputs and state.* `stat`, `crit` (null until a critical value exists),
  `flag` (`Source::Flag`), `since_flag` (`Source::Id`; learned rows since
  the last flag, null before the first), `n_eff`. `reset=True` empties both
  rings at a flag (`window`; `monitor`'s spans are disjoint by
  construction, so `reset` does not apply to it); `reset=False` keeps
  monitoring and the flag simply stays up while the statistic is over.
  Task 47's `clear_lags` empties the rings and, for `monitor`, abandons the
  current span and starts a new one — a break in the clock is a break in
  the data the statistic assumes contiguous. `ModelState::CorrChange`,
  appended. `n_outputs() = 5`.
- *Tests.* `kind="monitor"` against WKD's Tables 1–2 (5000 replications,
  5%, i.i.d. bivariate `t₅` innovations), which ANSWERS transcribes: size
  at `T = 500` is `.040 / .035 / .041` at `ρ = −0.5 / 0 / 0.5` and at `T =
  1000` `.038 / .034 / .039` — the gate's size test uses `|ρ| ≤ 0.5` (the
  test over-rejects at `|ρ| = 0.9` for `T ≤ 500`: `.142`–`.144`, which is
  the paper's finding, not a bug) and the paper's numbers as its
  expectation with a binomial band, over `≥ 300` seeds in the gate and `≥
  2000` in the `docs/REGIMES.md` experiment; size-adjusted power on `0.5 →
  0.7` at `T/2`, `.587` at `T = 500` and `.830` at `T = 1000`, within
  Monte-Carlo error; `D̂` and `Q` on one span against a longhand numpy
  computation to `1e-12`; the Kolmogorov quantiles above; `kind="window"`
  against a longhand numpy statistic to the bit and the permutation
  quantile against a numpy re-draw with the same seed; average run length
  to false alarm and mean detection delay of the `window` kind on task
  52's streams recorded in `docs/REGIMES.md` (the monitor's false-alarm
  rate per span *is* its size, already pinned); rings emptied across a
  session change and a
  `max_dclock` breach; `reset` both ways; the clock-rescaling test; the
  sweeps; `MINIMAL["corrchange"] = {"features": ["x0", "x1"], "window": 8,
  "crit": 0.5}`.

*Task 54 as built, 2026-09-06.* The re-scoping held. **The comparison with
WKD's tables did not, and task 56's `docs/REGIMES.md` is where that was
found**: their Tables 1 and 2 are for "i.i.d. bivariate `t₅` innovations",
the gate's tests draw *Gaussian* pairs, and the two are not interchangeable.
Measured at 2000 replications across three DGPs: at `ρ = 0` everything
agrees with the paper; at `|ρ| = 0.5` a shared-scale `t₅` makes this
implementation liberal (`.075` against their `.040`) and independent `t₅`
marginals conservative (`.025`), with their number between the two, while
Gaussian pairs sit at the nominal 5 %. Size-adjusted power is *below* their
table under either `t₅` (`.43` and `.47` against `.587`) and well above it
on Gaussian pairs. The gate's size test now pins **the nominal level on
Gaussian pairs**, which is a property this implementation has, instead of a
`t₅` table it does not reproduce.

**And the cause is one number, measured (§4 of that document).** `D̂` is the
delta-method sd of `√T·ρ̂`, and for an elliptical law the quantity it
estimates is `(1−ρ²)√(1+κ)` in closed form. Against that: on Gaussian pairs
`D̂` is exact and tight (0.750 against 0.750 at `T = 2000`, sd 0.03); on a
tail-dependent `t₅` it is **15 % low, no better at `T = 2000` than at
`T = 500`**, with a scatter 40 % of its own size. It is built from fourth
moments and a `t₅` has kurtosis only just (`ν > 4` by one), so there is no
rate at which it settles. Too small a denominator is the liberal size; the
scatter is the lost size-adjusted power. The estimator is not wrong — it is
right where the delta method promises it will be. Five notes.

- *`scalar = true` is a **mean** CUSUM, not a correlation one.* The first
  implementation pushed `[u, u]` into the pair machinery, whose correlation
  is exactly 1 on every prefix, so the statistic was identically zero. The
  equicorrelation is already one number: what there is to test is its
  *level*, `max_j (j/√T)·|ūⱼ − ū_T| / D̂ᵤ` with the Bartlett long-run sd of
  `u`. The ring is one column wide under `scalar`.
- *`predict` computes the span-closing statistic too.* The first version
  reported nothing from `predict`, which breaks the contract on exactly the
  rows that matter. `read(x, weight)` is now a pure function of the state
  and the row that both `step` and `predict` call; the permutation critical
  value is **refreshed after** the row is reported, so the two cannot see
  different ones.
- *The predict-parity helper needed a `SPARSE_OUTPUT` list.* It asks for 300
  of 400 rows to have every slot ready, and a span-based model has one row
  in `horizon`. Ten is enough there, and the list says which models it is
  for.
- *The window kind's flag rate per row is not `alpha`.* Two windows that
  slide by one row are almost the same windows, so a statistic above the
  quantile stays above it for a run of rows. Measured: 9 % of rows on a
  stationary stream against 27 % on a broken one, which is the separation
  the test asserts. Documented in the docstring and the README.
- *A capped clock gap abandons the span, and that is visible.* The golden
  pipeline's stream jumps 9 units every 17 rows; at `max_dclock = 6` every
  jump is capped, task 47's `clear_lags` empties the ring, and **no span
  ever closes**. Correct, and it pins nothing, so that spec uses a cap above
  the gaps. `tests/test_corrchange.py` asserts the abandonment directly.

*Task 55 — `bocpd` (E61).*

- *The recursion.* Adams & MacKay (2007) in log space: the growth branch
  `P(rₜ = r+1, x₁:ₜ) = P(rₜ₋₁ = r, x₁:ₜ₋₁)·πₜ^{(r)}·(1 − H)`, the changepoint
  branch `P(rₜ = 0, x₁:ₜ) = Σᵣ P(rₜ₋₁ = r, x₁:ₜ₋₁)·πₜ^{(r)}·H`, `H = 1/hazard`
  with `hazard` a number or a column read per row (declared like `weight`,
  not a feature); ANSWERS confirms the two branches against the paper's
  Algorithm 1. Truncation: runs whose normalised mass is below
  `truncate` are dropped and the vector renormalised; `max_run` caps its
  length by folding the tail into the last kept run. The vector and one
  sufficient-statistic block per kept run are the state; the cost is
  `O(runs · d²)` a row (`O(runs · d)` diagonal).
- *Emissions.* `"gaussian"`: normal-inverse-Wishart, `(μ₀, κ₀, ν₀, Ψ₀)` from
  `prior_mean` (default zeros), `prior_kappa` (`1.0`), `prior_nu` (default
  `d + 2`), `prior_scale` (a scalar `s` for `Ψ₀ = sI`, or a `d × d` matrix;
  default `1.0`, and the docstring says to set it from the data's scale);
  per run the statistics `(n, Σx, Σxx')` and the predictive `t_{ν−d+1}(μₙ,
  Ψₙ(κₙ+1)/(κₙ(ν−d+1)))`. `"diag"`: normal-inverse-gamma per feature, the
  predictive a product of Student-t densities. `"robust"`: the
  diffusion-score-matching conjugate posterior of Altamirano, Briol &
  Knoblauch (2023) with `robust_beta` its one hyperparameter — the plan
  does not restate its equations; the implementer takes them from the
  paper (its Gaussian case is closed form, quadratic in weighted
  sufficient statistics), and the acceptance test below is what pins it.
  `robust_beta` with another emission, or `prior_*` that do not fit the
  emission, are refused by name. Univariate use (one feature) is allowed and
  is the paper's own setting. ANSWERS did not verify the ABK 2023 posterior
  (PMLR 202, open access) — it is the second of the two reads left to the
  implementer, alongside CKP §3 in task 50.
- *Outputs.* `p_r0` — the posterior mass of `rₜ = 0`, i.e. that *this* row
  started a run, computed from the row (`kmeans`-`dist` style, and the
  docstring says so; the pre-row probability of a changepoint is `H` itself
  under a constant hazard and carries nothing); `run_mode` and `run_mean`
  from the run-length posterior **before** the row (`Source::Id` and `f64`);
  `pred_<f>` the pre-row predictive mean (mixture over runs); `logscore` the
  log predictive density of the row under the pre-row mixture; `n_eff`.
  `predict` returns the pre-row quantities without moving. `coef` empty.
  `ModelState::Bocpd`, appended.
- *Tests.* Against a longhand numpy Algorithm 1 on a synthetic univariate
  mean-shift stream, the full run-length posterior to `1e-10` at every row
  (`truncate = 0`); a variance step detected on Adams–MacKay's own finance
  fixture shape (ANSWERS, their §3): zero-mean Gaussian rows with a
  piecewise-constant variance, the `"diag"` emission with a gamma prior on
  the inverse variance (`a = 1`, `b = 1e-4`, their values — `prior_nu = 2a`,
  `prior_scale = 2b` in the inverse-gamma parametrisation the emission
  exposes), `hazard = 250` (their `λ_gap`), `p_r0` peaking within a stated
  delay of each step; `truncate = 1e-4` changing no reported output by more than
  `1e-4` against `truncate = 0`; the robust variant's `p_r0` unmoved by a
  single 20-σ row where the Gaussian one restarts; a zero-weight row leaving
  the posterior untouched; the hazard column; chunk invariance and save/load
  through the sweeps; `MINIMAL["bocpd"] = {"features": ["x0"]}`.

*Task 55 as built, 2026-09-06.*

- *`p_r0` is not a quantity.* Algorithm 1 puts the **same** predictive on
  the growth and the changepoint branch, so the normalised mass at `r = 0`
  is `H·Σ Jπ / Σ Jπ` = **exactly the hazard, on every row, whatever the
  data**. §11a asked for it and expected it to spike; it cannot. What ships
  is `p_change = P(rₜ ≤ 1)`, which adds the run that started one row ago —
  the one evaluated against the *prior* predictive on the break row.
- *And `run_mode` is the output to read.* It is the run length before the
  row, so **`t − run_mode` is the row the current run began on**, and that
  is what a caller wants. Measured on three fixtures: a tenfold variance
  step, a four-sigma mean shift and a correlation-only break are all dated
  to the right row within one to three rows of it, while `p_change` reaches
  0.83 on the first, barely lifts on the second and never moves on the
  third. The docstring, the README and the module docs all lead with the
  run length and call `p_change` the alarm.
- *Line 6 of Algorithm 1 was implemented wrongly first, and it matters.*
  `ν⁽ʳ⁺¹⁾_{t+1} = ν⁽ʳ⁾_t + u(xₜ)` with `ν⁽⁰⁾_{t+1} = ν_prior`: slot `j`
  holds exactly the `j` rows the hypothesis `rₜ = j` says precede the next
  row, so the slot pushed at the front holds **nothing**. Letting it take
  the row too puts one row of the old regime inside every "brand new run".
  The symptom was subtle — every run's estimated start was one row early,
  and a 20-σ row produced `p_change = 0.09` instead of `0.91`, because the
  hypothesis that it started a run was evaluated with the row before it
  already in the run. The longhand oracle had been written from the code and
  so agreed with it; the ANSWERS quotation of line 6 is what settled it.
  Both the Rust and the numpy oracles now write line 6 out.
- *Each run carries its own length.* The slot index stops being the run
  length the moment `truncate` drops a run from the middle of the vector or
  `max_run` folds the tail, and it never was one under a fractional row
  weight or the robust emission's. `Run.len` counts rows; `run_mode` and
  `run_mean` read it.
- *`min_periods` gates the report, never the update.* The first version
  returned early before computing the recursion, so row one of every group
  was silently dropped from the model. The default is `1.0`, which nulls
  exactly that row: `P(r ≤ 1)` is 1 there whatever the data, since no run is
  older than one.
- *The robust emission tempers the message as well as the statistics.*
  Weighting only what a run learns leaves the outlier declaring a
  changepoint (measured: `p_change` 0.85) with clean statistics behind it,
  which is the worst of both. Using the same `w(x) = (π/π(mode))^β` on the
  likelihood in the recursion makes a row that is atypical for *every* run
  multiply every joint by about 1, so nothing moves at all. That is the
  plan's acceptance test, and it now passes.
- *`robust_beta` is a trade, and 0.1 is the measured default.* A whole new
  regime is a run of individually forgiven rows, so tempering too hard blinds
  the model: at 0.05 and 0.1 a four-sigma shift is still found within five
  rows and dated to within one, at 0.3, 0.5 and 1.0 it is never found at
  all. Both halves are pinned, so the note cannot go stale.
- *`prior_scale` too large is silence, not false alarms.* The plan said to
  set it from the data's scale; measured, the failure mode is that a
  predictive that wide finds no row surprising, and a real break is never
  detected (a four-sigma break on variance-1e-6 data: dated exactly at
  `prior_scale = 1e-6`, invisible from 1e-3 up).
- *`hazard_col` is explicit.* The hazard rides in the target slot, and a
  `hazard_from_row` flag in the config says to read it — without one, a
  `bocpd` in a bank with a regression target read the target as a hazard and
  poisoned itself with a NaN.
- *The default emission is `"diag"`.* `O(runs·d)` against `O(runs·d²)`, and
  `"gaussian"` is what to reach for when the break is in the correlation
  with the marginals unchanged, which is a test in `tests/test_bocpd.py`.

*Task 56 — the schema-5 fixture and the close of the batch.*

- *Fixture.* `crates/online-polars/tests/state_schema5.rs` in
  `state_schema4.rs`'s form — hex bytes in the source, a `print_fixture`
  generator, the "what it says" / "loads with everything" / "continues to
  the bit" / "re-saves byte-identically" tests — from a bank whose specs
  cover the batch: a `group_close = "monotone"` spec with a closed row
  undrained and a `high_water`, an `ew_cov` with `lags` and a partially
  filled ring, and one of `deco`, `rcov`, `hmm`, `corrchange`, `bocpd`
  each mid-stream. `state_schema4.rs` stays untouched and must load through
  the schema-5 reader (rule 5's loader).
- *Docs.* Every §10 row gets its task number and "done" as §9's did; the
  README's model table and a `### <name>` per new model with a runnable
  block; `docs/REGIMES.md` complete with the three experiments (task 53's
  recovery, task 54's size/power/ARL, task 52's Epps curve); CHANGELOG
  `[Unreleased]` listing the schema bump first; `docs/RELEASE-READINESS.md`
  for the new surface. E63 stays a note: the `weight=` recipe it describes
  is already expressible, and nothing in this batch adds `weight_from`.

*Task 56 as built, 2026-09-06.*

- *The fixture.* `crates/online-polars/tests/state_schema5.rs`, six specs on
  a 60-row stream with a **monotone** group key: an `ew_cov` with
  `lags = [1, 12]` and `group_close = "monotone"`, an `rcov` (whose whole
  output is a closed group's block), a `deco`, a sessioned `hmm`, a
  `corrchange` mid-span and a `bocpd`. Two groups have closed and nobody
  read them, so the file carries the queue and the high-water key; the third
  group holds ten rows, which is fewer than the lag ring wants, so the ring
  is frozen partly filled with the long lag still empty. `state_schema4.rs`
  is untouched and still loads, continues to the bit and re-saves as 5.
- *One departure from `state_schema4.rs`'s shape.* Its `assert_same_state`
  compares coefficients with `assert_eq!`, which cannot work here: a gated
  instance's coefficient is a NaN and `NaN != NaN`. Schema 5 compares the
  debug rendering, where a NaN in the same slot is a match.
- *`docs/REGIMES.md` and `scripts/regime_experiments.py`.* Five experiments,
  35 seconds, committed rather than gate-regenerated (the
  `docs/CLUSTERING.md` §7 precedent). Three of them are the ones §11a asked
  for; the fourth compares `corrchange(window)` and `bocpd` on the same
  break, which is where the two detectors' trade shows, and the fifth is the
  Epps curve with an `L` sweep of the lag inversion.
- *And the size study is why the experiments were worth running.* The first
  draft compared Gaussian draws against WKD's `t₅` table and concluded that
  this implementation is more powerful than the paper. It is not: under
  either reading of their DGP it is *less* powerful once the size is
  adjusted, and its size at `|ρ| = 0.5` swings from `.075` to `.025`
  depending on which bivariate `t₅` is meant. Task 54's note and its gate
  test are corrected; the document reports all three DGPs side by side.
- *A sixth experiment, `dhat`, turns that from a hypothesis into a
  measurement.* `D̂` is recovered from the reported statistic (the longhand
  numerator divided by it) and compared with the closed-form elliptical
  value it estimates. Exact on Gaussian pairs, 15 % low and wildly scattered
  on a `t₅`, and no better at four times the sample. Both symptoms in the
  two sections above are that one number, and the document says so instead
  of calling them unexplained.
- *The seeds are `zlib.crc32`, not `hash()`.* Python randomises string
  hashing per process, so the first draft's numbers changed run to run. An
  experiment in a committed document has to give the reader the numbers it
  claims.
- *What the experiments found, beyond the tables.* `hmm`'s default seeding
  (k-means over the rows) **cannot find a regime that lives in the
  covariance**: it splits two zero-mean states by direction and lands at
  chance, where the same filter given the covariances is at 98 %. And a
  `precision_prior` of 1.0 on data of variance 1.0 halves every correlation
  and collapses the accuracy — `bocpd`'s `prior_scale` lesson in a second
  model. Both are in the document and in `hmm`'s docstring.
- *ENHANCEMENTS §10 needed nothing.* Every row already carried its task
  number and its departures, written as each task landed; E63 stays a note
  by design.

*Answers folded in, 2026-09-05.* `docs/ANSWERS-E54-E64.md` was read against
every "confirm from the paper" above, and the blocks were edited in place
rather than annotated, so a task block is still read top to bottom. What
it changed, in order of consequence:

- **Task 54 is re-scoped.** The `kind="monitor"` of the first draft — a
  training span, a sequential boundary, an open-ended alarm — was written
  as Wied–Krämer–Dehling 2012 and that paper has no such boundary: its test
  is a *closed-sample* fluctuation test on a span of known length `T`, and
  the sequential detector is Wied & Galeano 2013, which nobody has read.
  The monitor now runs the WKD test **span by span** (`horizon` required,
  `train` gone), with the paper's `D̂` (three-way delta method over the five
  raw moments, Bartlett long-run covariance, `γ_T = ⌊ln T⌋`), critical values
  from the Kolmogorov series (`1.358` at 5 %), Bonferroni over pairs, and
  its size and power *pinned to WKD's Table 1* — `.040/.035/.041` at `T =
  500`, `.587` power on a `0.5 → 0.7` step — instead of "within Monte-Carlo
  error" of a nominal figure. The `train` semantics and the ARL experiment
  move to the `window` kind, which is unchanged (ANSWERS confirms the
  sign-flip null has zero spread, so the row permutation stands). The
  Wied–Galeano sequential form is an ENHANCEMENTS §10 follow-up, not a
  blocker.
- **`epps_invert` gets `L` and the triangular weights.** Tóth–Kertész
  eq. 12 weights the numerator *and both denominators* by `(L − |x|)`; the
  flat sum the draft had is its `L → ∞` limit. The signature is now
  `epps_invert(gram_or_row, *, L)`, and the input must carry lags `1 …
  L−1`.
- **`rcov`**: `kₙ = ⌊θ√n⌋` (floor, as CKP write it), the finite-sample `ψ`
  and `Φ` sums as the paper has them, and the auto-bandwidth's two grids
  named as BNHLS use them (`iv_stride` for the sparse `IV̂`, `noise_stride`
  for `ω̂²`, with its deliberate upward bias stated). The one thing ANSWERS
  did not read — CKP §3's longer window for `psd=True` — is marked as the
  implementer's read; the draft's `δ = 0.1` is Hautsch–Podolskij's and
  stands only if §3 agrees (it does: task 114, 2026-09-28).
- **`po.corr`**: Higham's examples, distances, null vector, rank and
  iteration count are now the paper's, not memory's, and `nearest` is
  pinned to Algorithm 3.3 with his convergence test 4.1 and the three
  bound tests his theorems give; `shrink` carries the Ledoit–Wolf `π̂`, `ρ̂`,
  `γ̂` formulae as its oracle; `mp_edge` is verified and gains
  `mp_density`; `fisher_se`'s AR(1) factor is derived from Bartlett's
  formula and its two assumptions are named.
- **`deco`**: the correlation-targeted intercept and `α + β < 1` are
  recorded as *our* departures from Engle–Kelly, with the paper's
  downward-bias remark and its unoffered `u^var` alternative noted.
- **`bocpd`**: the variance-step test uses Adams–MacKay's own finance
  fixture shape (`"diag"`, gamma prior `a = 1`, `b = 1e-4`, `hazard = 250`);
  the ABK 2023 robust posterior stays an implementer's read.
- **`refresh_time`**: the `N = 7`, `21/27` example is verified but its tick
  times are only a figure, so the test constructs a stream with those
  counts; the staleness caveat of BNHLS §2.1 goes in the docstring.
- **`hmm`** (task 53) and the batch-wide decisions: confirmed as written,
  nothing changed.

Two reads remain with the implementer — ABK 2023 (open access) for
`bocpd`'s robust emission and CKP §3 for `rcov`'s PSD window — and one
paper is deferred, Wied & Galeano 2013. Neither read blocks a task from
starting; each is a named paragraph in its block with the acceptance test
that pins it. The batch is ready to implement.

**The chunk plan, revisited: P9–P11 and a fan-out floor, 2026-09-04.**
Asked whether the per-chunk parallel plan could be faster without
compromise. Sectioned first: at 14 threads on a 64-group chunk the
per-group tasks were 15 ms and the phases around them 21, each with a
single-threaded stretch. Three changes, every golden number unchanged:
columns gathered once, group after group, so a stream reads a contiguous
run (P9); one job per output field, `Vec<f64>` + bitmap into
`from_vec_validity` (P10, which reverses the 2026-09-02 "typed builders
not worth it" — right for the chunk it measured, wrong at 14 threads and
for a grid); columns read in parallel, multi-chunk columns copied per arrow
chunk, integer keys bucketed by value with `integer_groups` pinned to the
`String` path by a test (P11). Wall on the 64-group chunk at 14 threads
37 → 17 ms; the 12M-row README workload 3.25 → 2.48 s. Two findings worth
more than the speed: the matrix's 4× interleaved-vs-blocked gap was
`solve_every` cadence — an index clock over interleaved groups hits
`max_dclock` every row and re-solves ten times as often — so the layout's
true cost was 14%/38%, and the "artifact" is the documented semantics of a
clock-unit solve schedule, which any such benchmark pays; and the gate's
memory test caught the fan-out of 30-row groups (the plugin under
`.over()`) doubling RSS *wobble*, not growth, for no speed, hence
`PAR_MIN_ROWS` = 4096 below which a chunk's columns and fields are done on
the calling thread. `docs/PERFORMANCE.md` §12 has the tables. Not merged
before `v0.1.0` unless the user says so: it moves the tag target and needs
another rehearsal. **Decided 2026-09-04: `v0.1.0` is tagged on `d6370d9`
(main, rehearsal and CI green there) and this branch is 0.1.1 material.**
The same day the README's Parallelism section gained a *Chunk size*
subsection — the knob is `chunk_rows` on every streaming surface, the
numbers never depend on it (only where `coef` lands), the default 100k is
right for interleaved groups (50k–500k within 0.4 s on 12M rows), a
group-sorted file wants a few × rows-per-group (8.1 → 4.2 s at 1M here),
and very large chunks cost memory and the read/fit/write overlap (2M: 2.4 GB
and nearly twice the default's time). Its memory column, and the two-knobs
paragraph's memory numbers, are **peak footprint** (`/usr/bin/time -l`),
one run per process; the two-knobs numbers had been peak RSS, which counts
the memory-mapped input (~0.7 GB on a 712 MB file) and read 1.8 GB where
the footprint is 1.1. The other numbers in that section are still main's
and are regenerated when the branch lands (`scripts/benchmark.py`,
`scripts/scaling_bench.py`, the grid timings). PERFORMANCE §12 has the
sweep on both builds.

**The x86 control, the merge, and 0.1.1 prepared, 2026-09-04.** Asked
whether the layout's gain was this machine's, the answer came from
GitHub's `ubuntu-latest` (4 vCPU, x86) with the same workflow file on both
commits — `benchmark.yml` gained a `ref` input for exactly that (a
dispatch measures any tag, branch or full SHA as a control) and its
artifact now carries both tables. k=20 over 64 groups: 389k / 704k / 788k
rows/s on `v0.1.0` against 421k / 807k / 890k on the branch at 1 / 2 / 4
threads (+8 / +15 / +13%), so the stride was not an Apple artefact; the
single-group table is at or above `v0.1.0` on every row within the
runner's own ±10% (two runs of `v0.1.0` code differed by that much). The
branch was merged fast-forward and prepared as **0.1.1**: version bumped,
CHANGELOG cut, `docs/VALIDATION.md` regenerated (the version line and
nothing else), the README's measured numbers regenerated on this build.
What remains is CI and a Release rehearsal on the merged head, then
`v0.1.1` on the user's word.

**The chunk plan's edge cases, 2026-09-04.** Asked to make sure every edge
case of the parallel implementation is tested. The plan's three pieces
had been proved by golden numbers on the benchmark frames and by the
existing 400-row bank tests, which never reach `PAR_MIN_ROWS`; what was
missing was the same claims *above* the floor and on the awkward inputs
(null keys, one-row groups, nulls in every column, sessions, a column in
many arrow chunks, non-Float64 dtypes, chunk sizes of 4095/4096/4097,
errors under a permuted layout). `crates/online-polars/tests/chunk_plan.rs`
now holds them, `PAR_MIN_ROWS` is public so the test can straddle it, and
`tests/test_portability.py`'s thread-determinism case runs above the floor
too, with a halflife grid and every output on. Nothing was found; the
tests were mutated by hand to check they could find something, and did.
`docs/TESTING.md` T-E13 has the list.

**`v0.1.1` released, 2026-09-04.** The user fixed the PyPI trusted
publisher (the `v0.1.0` publish job, re-run, put 0.1.0 on PyPI) and said
"tag for 0.1.1 when ready". Ready meant CI 33871208866 and the
user-dispatched rehearsal 33872948555 green on `e552176`, so the annotated
tag `v0.1.1` went on that commit — not on the edge-case tests that landed
on `main` the same hour, which are test-only and would have meant another
rehearsal. Release run 33879445306: every job green; the GitHub release
carries the six wheels, the sdist and five CLI binaries; PyPI has all
seven files, and a clean venv's `pip install polars-online==0.1.1` (polars
1.44.1) ran a 5000-row grouped fit. The next release is whatever
`[Unreleased]` gathers.

**The bank's pool is its own, named for what it is, 2026-09-04.** The
bank fanned out on rayon's global pool, so its one knob was
`RAYON_NUM_THREADS` — a name that, next to `POLARS_MAX_THREADS`, said
nothing about which pool it was. Asked whether two pools could interact
badly, measured first: no correctness hazard (the wait graph is one-way —
py-polars' pool → our pool → our polars copy's pool — and a bank task never
takes the GIL or calls back into polars' pool), and no speed hazard either
(28 + 28 threads on 14 cores ran the grid in the same time as 14 + 14;
7 + 7 was slower). The two counts do different things: polars' also sizes
its reader's prefetch, so on 12M rows over 64 groups `POLARS_MAX_THREADS=4`
with the bank on 14 was 3.0 s at 1.4 GB against 3.2 s at 1.8 GB, while one
shared count of 4 would have been 4.5 s. Decisions: keep two pools and
two knobs; rename ours **`POLARS_ONLINE_MAX_THREADS`** and build it
ourselves (`crates/online-polars/src/pool.rs`: `OnceLock<ThreadPool>`,
built at the first bank call, per-core default spelled out so
`RAYON_NUM_THREADS` reaches nothing, a non-count refused by name);
`Bank::fit_predict`/`predict` run under `pool().install`, which also
carries the per-instance `par_iter`s in `Stream`; the runner's parquet page
encoding and NDJSON slices move onto polars' pool
(`polars_core::runtime::THREAD_POOL`, a direct `polars-core` dependency
already in the tree, nothing new linked) so `POLARS_MAX_THREADS` is
polars' readers *and* writers in every form; `po.thread_pool_size()`
mirrors `pl.thread_pool_size()`. The README's parallelism example is a
grid over factor sets (one spec per set — its own accumulator,
standardization and null handling — with the halflife/ridge grid inside),
and a second example shows the two knobs set apart, with the numbers.
Not done before `v0.1.0` would have been a released knob to rename later.

**Boosted trees: investigated, prototyped, not built, 2026-09-03 (task 21).**
Asked to dig into XGBoost — the papers and the code — for how gradient-boosted
trees could be made as online as possible, fit in parallel, and use less
memory. The finding, in `docs/BOOSTED-TREES.md`: the boosting math is
already online-shaped (leaf values and split gains are functions of per-node
gradient sums, which are additive, mergeable and decayable), and what is
*not* online in XGBoost is everything around the sums — a cut pre-pass over
all the data, O(n) gradient/position arrays, histograms that exist only
while a tree is built. A design that keeps every rule of the contract was
prototyped in numpy and measured: it matches XGBoost's batch fit on the
warm-up buffer to 0.02, then ties an 8 000-row refit window on stationary
data and beats it by 2–12 MSE under drift, in 12–16 k doubles of state
against the window's 80 000 rows. Chunk invariance is exact and per-tree
sums are additive over threads; both are measured, not argued. The ideas
that did not survive measurement are recorded with their numbers (§7.3) so
they are not re-tried. Decisions: the prototype and its experiment script
are committed under `scripts/` as source, not wired into the gate or CI,
with nothing added to `pyproject.toml` (XGBoost is an optional `uv run
--with` overlay for the baseline rows only, so rule 12 is untouched);
downloaded sources, clones and notes stay under the gitignored
`.cache/research/`; the exclusion of trees in `ENHANCEMENTS` §4 and
`BEYOND-O-STATE` is reassessed — its three technical grounds (unbounded
state, nondeterminism under resampling, no clock-decay semantics) are
answered by the design; the cost of a second model family is not, and is
the decision — and both documents point here without rewriting their
history; nothing goes
into the Rust crates until the user decides, and the doc's §9 says what
that would take (a `gbt` module in `online-core`, `ModelState::Gbt`, a
`SCHEMA_VERSION` bump, the `EXTENDING.md` list) and what should come first
(real data, §8 idea 10).

**One error contract, stated once and documented at every entry point,
2026-09-03.** Audit of every public docstring and Rust doc comment for its
failure modes found the contract mostly there and the gaps all of one kind:
a failure that was typed wrong (`OSError` for a spec mismatch), silent (an
unknown key in a spec or config took the default), or late (`save_state`'s
directory checked after the run it would lose). Each was fixed rather than
written up: `deny_unknown_fields` on `Spec`, `ModelKind` and `RunConfig`
(the state-file loader reads the envelope first, so a newer build's file
with keys this build lacks says "newer", not "not a bank file");
`ModelBank.load` splits file errors (`OSError` subclass) from content errors
(`ValueError`); `po.run` checks `save_state`'s directory before the run.
The contract is one paragraph in `polars_online.__doc__`; each docstring
says only what *it* raises and when, and `tests/test_runner.py`,
`test_bank.py`, `test_eval.py` pin the types and messages. `cargo doc
--workspace --no-deps` builds clean and is the Rust reference.

**The API reference is Sphinx, and a bad docstring fails the build,
2026-09-03.** Asked for a doc builder with no preference between them:
Sphinx, because the docstrings were already written in its dialect
(`:class:`/`:func:` roles, `::` literal blocks) and pdoc would print those
roles as text. `docs/reference/` holds four pages — the package, `spec`
with the typed keyword sets, the three `online` namespaces, `eval` — all
autodoc; nothing is written twice. It builds with `-W` in the gate and in
CI's ubuntu test job (it imports the package, so it needs the built
extension that job already has; no second Rust build, no new job on the
other runners), and a `docs` job publishes the HTML to GitHub Pages from
`main`, skipping with a notice if Pages is ever switched off. Pages was
enabled the same day, Source "GitHub Actions" and nothing else — the
Jekyll / static-HTML starter workflows GitHub offers there would each add
a second deploy of the repository tree to the same site — and the run in
flight deployed <https://hgilde.github.io/polars-online/> at once; the URL
is `Documentation` in `[project.urls]`. The first `-W` build found four
docstrings that were not valid RST
(a `*by` read as emphasis, `|r|` as a substitution, a table column one
character narrow, ``` ``TypedDict``s ``` with the plural glued to the
literal) — the class of defect that had no test before, which is the
argument for `-W`. The `docs` dependency group (`sphinx`, `furo`) is
installed by `uv run --group docs`, so a plain `uv sync` stays as it was.

**The betas are a frame, not a detour, 2026-09-03.** Asked whether a saved
state can be introspected -- the coefficients of a linear model read back
from a file -- the answer was "in three roundabout ways": `predict` one row
and read the output's `coef` field, hand-solve `gram()`'s moments, or take
the last `coef` from the run that made the file. None names the terms.
`ModelBank.coef(spec, group=None)` now returns one frame -- `group`,
`instance`, `n_eff`, then `spec.coef_index`'s columns, then `coef` -- so
`bank.coef("ols").pivot("term", index=["group", "instance"], values="coef")`
is the wide table people expect, from a live bank or `ModelBank.load(path)`.
It is the same `AnyModel::coefficients()` call the output's `coef` field
makes, so the two agree by construction (`tests/test_coef.py`). Two facts
the docstring states because they surprised: `coef` is the last *solve*, not
gated by `min_periods` as `pred` is (so it can exist, jittered, over fewer
rows than terms -- `n_eff` is there to say how much is behind it), and under
the default `solve_every = halflife/50` it is that stale. Asked separately
whether an EW-OLS is a local regression: measured, yes -- with `ridge=0.0`
and `solve_every=0`, `pred[t]` is the weighted least-squares fit of rows
`< t` with weights `0.5**((t-1-i)/halflife)` to 1e-14, `coef[t]` the same
over rows `<= t`; the kernel is one-sided (causal), `n_eff` saturates at
`1/(1-0.5**(1/halflife))` ~ `1.44*halflife`, so `min_periods` must sit below
that, and the plan / `po.run` stream it in O(chunk) memory. The test pins
the statement against `numpy.linalg.lstsq`.

**The output comes apart with the coefficients named, 2026-09-03.** The
polars-native way to the betas was `coef_index(spec)["term"]` fed to
`list.to_struct(fields=...)` and an `unnest` -- correct for one instance
and one combo, wrong for a grid (one term list, several blocks), and with
bare names (`x0`) that collide with the feature columns. Asked for a helper
that covers whatever else the output carries, not just the coefficients:
`lf.online.unnest(specs)` (and `df.online.unnest`, `po.unnest(frame, ..)`)
is polars' `unnest` for a bank's output -- every scalar field becomes a
column of its own name, and each `coef` list becomes one column per
coefficient, named on the field grammar with the term after the target
(`coef_y_x1__r0.5@h500` beside `pred_y__r0.5@h500`). The names come from
`spec.coef_fields(spec)`, a Rust-rendered table (`online_polars::coef_fields`,
next to `output_index`) of every coefficient with its `field`, `position`,
`name`, target, halflife/lam, ridge, feature set, lambda and term;
`coef_index` is now that table's first instance. `unnest` takes the specs,
a bank, or a state path, replaces each struct in place, leaves unnamed
columns alone, and reports a spec that does not match the frame while the
plan is built; two specs with the same field names are polars'
`DuplicateError`, as with polars' own `unnest`. The test that matters
(`tests/test_unnest.py::test_named_coefficients_predict_the_next_row`) pins
the names against the models rather than the strings: with a solve on
every row, `pred[t+1] == coef_intercept[t] + sum_j coef_xj[t] * xj[t+1]`
for all 16 (target, feature set, ridge, halflife) columns of a grid, which
a wrong slot order, a swapped feature, a wrong ridge or a wrong instance
each break by 0.03 to 7 (checked by breaking them). Not added: a SQL
`where=` filter for the CLI (the user declined it: a `polars-sql`
dependency for a filter polars' own plan already has).

**Any row order, 2026-09-03.** The README said the library was "built for
ordered event data", which undersold it: a bank is sufficient statistics,
so row order reaches the fit only through decay, and with decay off it is
a regression library that never holds the data. Measured rather than
argued, 6M iid rows x 20 features in 12 parquet parts: `ewridge` with
`ridge=0` and `halflife=inf` (or `lam=1.0`) matches `numpy.linalg.lstsq`
to 2e-13 fed forwards or backwards, at 1.4 GB peak RSS against 3.97 GB
for `lstsq` on the same rows (10.7 s solving every row, the default for
`inf`; 1.4 s with `solve_every=1000`, the coefficients 2e-6 off). A finite
row halflife is the weighted least squares of the order given (halflife
1e6 rows: 4e-4 from OLS, which is itself 5e-4 from the truth; reversed,
the same distance). Two things came out of it. (1) The docs now lead with
both shapes: the README intro, a new "Any row order" section, "What this
is not", the `halflife` row and the `ewridge` solve sentence; and
`tests/test_row_order.py` pins no-decay == lstsq in four row orders with
two interleaved groups, through the bank and the chunked plan, and the
finite-halflife-reverses-the-weights statement. (2) A trap, known here as
the `solve_every` gotcha further down in this section and now documented
in the README and pinned, but not fixed: the solve cadence defaults to `halflife/50` for any
*finite* halflife and to every row only for `inf`/`lam`
(`Spec::solve_every_default`), and `max_rows_between_solves` defaults to
unlimited, so `halflife=1e12` solves once at `min_periods` and never again
-- 0.3 to 0.4 off OLS on the same 6M rows, where `inf` or
`solve_every=1000` is 2e-6 off. Not changed now because the cadence is
part of every golden and any rule that scales with the accumulated weight
needs the weight at the last solve in the state (a layout change:
`SCHEMA_VERSION` bump plus a loader for the old one). The candidate rule,
for the user's decision: solve when the weight added since the last solve
exceeds ~2% of the total, which reproduces `halflife/50` at steady state
(2% of a saturated `1.44*h` is `h/35`), solves densely while the weight
is still growing (early rows, and a warm-up after `reset_state`), and
makes `inf` cost O(log n) solves instead of one per row while bounding
the staleness to 2% of the weight. Until then the README says: `inf` is
the no-decay setting, `solve_every` is the throttle, and a huge finite
halflife is neither.

**The Polars concern stated up front, and the canary made honest,
2026-09-03.** Asked for a note at the top of the README on the moving
polars API and how the canary handles it. Writing it meant checking what
the canary does, and it was not what the README said: its Python unpin
regex still matched `"polars==..."`, which the range `polars>=1.34.0,<2`
no longer is (so it tested the newest 1.x only by the range's grace and
would have excluded a 2.0 silently), and its Rust unpin -- `version = "*"`
plus `cargo update -p polars` -- would have put two polars in the tree
the day polars 0.56 ships, because pyo3-polars 0.28 requires `^0.55.1`
(its Cargo.toml) and `crates/online-polars` pins polars-arrow,
polars-parquet and polars-utils to `=0.55.2`: a red canary for a reason
that is not "Polars broke us". Today both are harmless (0.55.2 is the
newest crate, `cargo search`; 1.44.1 the newest wheel), so the job has
never misfired -- it has also never run, the schedule being Mondays on a
repo public since 09-02. Now: the canary moves py-polars only, the copy
that can break a user (the Rust copy is the wheel's own and never meets
it), drops the range rather than widening it so a 2.0 is tested the week
it appears, asserts the dependency line was found, and upgrades polars
alone (`uv sync --upgrade-package polars`, checked locally to move
nothing else) so one thing varies per run. The README's top note and
"How the pin will move" state the policy for a red canary: cap the range
at the last release that passed, in a patch release, then fix and widen;
look at `ModelBank` first, the IO-plugin tests second, the plugin last.
"What the pin costs you" was rewritten too -- it still said a different
polars could not be installed at all, which the range made false.
Dispatched by hand after the push, the job's first run ever was red on
exactly one of 1162 tests, with the same 1.44.1 CI uses:
`test_scaffold.py::test_the_declared_range_brackets_what_we_build_against`
reads pyproject and asserts the range the canary had just removed. That
test and its neighbour (`pl.__version__ == BUILT_AGAINST`, which would
have gone red on the first newer wheel) are about our pins, not polars;
they now carry a `pins` marker and the canary runs
`-m "not soak and not pins"`. Note `-m` replaces pyproject's addopts
`-m 'not soak'` rather than adding to it.

**State leaves a streamed plan through a file, and only a file, 2026-09-03
(task 20).** `lf.online.fit_predict(specs, load_state=, save_state=)` — the
runner's two keywords on the plan, so the fourth step of the state workflow
(load, learn on, save) is written the same way on every surface. The plan
stays pure: `load_state` is read when the plan is built and the plan carries
the bytes, as `df.lazy()` carries a frame (the same for `predict(path)`), so
collecting twice gives the same frame and `load_state=p, save_state=p` used
twice in one query cannot race the second run's load against the first
run's write. `save_state` is written when the source has fed the bank its
last row — the stream's end, or the `n` rows of a pushed `head(n)`, which
the source now applies to the *input* chunk so the bank learns exactly those
rows — never in `finally`, never after a bank error; a plan used twice in
one query runs twice on two threads (measured: no common-subplan
elimination reaches a Python source) and writes the same bytes twice. That
second point turned up a hole older than the feature: `atomic.rs` named its
temporary by pid alone, so two threads saving one path in one process wrote
*one* temporary and published a mixture — `ModelBank.save` from two threads
had the same hole. Fixed there, with a process-wide counter in the name,
rather than with a Python-side lock: the root, and it covers `po.run`'s
output file too. The memory side — a plan that updates a `ModelBank`
object, or `load_state=bank` — is declined: the user's call, and the right
one, because a plan is re-executed (twice, concurrently, when a query uses
it twice) and an object it mutates has no single "after"; the file has, and
a user reads it without knowing any of this. One documented gap (R6 in the
research): a node *after* the bank failing does not stop the bank, so the
state is written although the query failed — `po.run` saves only after its
output is committed, and a dated `save_state` per batch keeps a rerun from
learning it twice. `coef` moved one row in `head(n)` results: it is reported
on each chunk's last row, and the `n`th row is now that row.

**The expression form stays and warns, 2026-09-03 (task 19).** Two forms
carried one set of numbers and two memory profiles — `df.with_columns(pl.col
("y").online.ewridge(..))` at 7.3 GB against `lf.online.fit_predict([spec])`
at 1.35 GB on 12M rows — and a user who wrote the natural expression inside a
lazy query got the O(data) one. The cause is polars' contract for a stateful
user expression (the entry below), which we cannot change from inside a
plugin. The first cut of this task removed the expression form: the wheel was built
without an `expr-plugin` cargo feature, `pl.Expr.online` went unregistered and
the README showed only surfaces that stream. Reconsidered the same day, before
the next commit: a user who writes the expression then gets polars' bare
`AttributeError` with no rationale and no pointer, the in-memory use (features
as expressions, `.over`) is lost for nothing, the one interface with a polars
stability guarantee leaves the wheel, and the plugin's runtime tests stop
running in CI. So the expression stays and *teaches* instead: every call issues
`InMemoryExpressionWarning` — a `UserWarning`, because a `DeprecationWarning`
is hidden outside `__main__`, i.e. in the pipeline module where it matters —
with the reason, the plan to write instead, and the filter for someone who
means it; the README's closing note shows the two forms side by side with
the numbers. Not "deprecated": it would become a streaming form too if
polars ever ran a user expression per morsel with state (§6). The feature
gate, `has_expr_plugin()` and the `requires_expr_plugin` marker are gone.

**The expression form is in-memory; the streaming query form is the bank as a
source, 2026-09-02 (ENHANCEMENTS E33).** `lf.with_columns(pl.col("y").online
.ewridge(..)).sink_parquet(..)` collects the whole input in either engine,
because polars' streaming engine has no ordered, stateful node for a user
expression — a plugin is a `columnar-function` node (collect, call once,
re-emit), and the only per-morsel path for user code is elementwise, which
is unordered. Measured 7.3 GB at 12M rows (`docs/PERFORMANCE.md` §11). That
cannot be fixed inside the plugin, so it is fixed at the plan level:
`lf.online.fit_predict(specs)` registers the bank as a polars IO-plugin
source (`register_io_source`) and returns a `LazyFrame` that streams the
input through a fresh bank when it runs — O(chunk), bit-identical to
`po.run`, composing with polars' filters, selections, joins and sinks. Rules
adopted: the plan is *pure* (a fresh bank per execution, or `load_state`;
no `bank=` the plan would mutate — `save_state=` came with task 20, and
purity is what makes it safe); a filter after the bank
never changes what it learns from; polars does not re-apply the pushdowns
it hands a Python source, so the source honours projection, predicate and
slice itself, slice counted before predicate (polars' optimizer order). The
interface is polars' documented-but-`@unstable` IO plugin (CLAUDE.md rule
13), and it reads with `LazyFrame.collect_batches` (py-polars 1.34.0), which
`po.run` already did since E32 — the declared floor moved from 1.28.1 to
1.34.0 to say so. The Rust API has no twin: polars-stream 0.55.2 lowers
`AnonymousScan` to `todo!()`; Rust callers use `run(.., Output::Batches)`.

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

**Superseded in intent by task 88 (2026-09-23), not yet built.** A third option
was not weighed here: give the time parameters units too, as durations. That
removes the objection above, because a `timedelta` halflife states its own
unit and so no longer depends on the column's. Until task 88 lands, this
refusal stands exactly as written.

**Tasks 15-17 (CLI, release CI, README), 2026-08-30.**

- The streaming runner lives in `online-polars` (`RunConfig` / `run_config`), not
  in the CLI crate, so the same code path is testable without spawning a process
  and could back a Python streaming API later. (It did: E8, and E32 on
  2026-09-02 made the reader pluggable — `run(bank, Input::Lazy(plan) |
  Input::Batches(frames), Output::File | Output::Batches, ..)` — so `po.run`
  reads with py-polars and hands frames in, and any of parquet / ipc / csv /
  ndjson goes in or out. The Python API is not bound to parquet, and neither
  is the Rust one; only the CLI's inputs are files, because it is a binary.)
- Output is written with polars' batched writers: one row group (or record
  batch, or slice of text) per chunk, so memory stays O(state + chunk) end to
  end.
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
BTCUSDT 1-minute rows (14,336 rows) from Binance's public dump, features = past
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
- Rust edition 2024, `rust-version = 1.95`, `rust-toolchain.toml` pins the stable channel.
  It said 1.85 until task 160 (CI4), while the locked tree needed 1.95 (`sysinfo` 0.39.6);
  `tests/test_ci_cost_policy.py` now holds the declaration to the lock. Declaring 1.95
  raised clippy's MSRV, which turned on `manual_is_multiple_of`, `collapsible_if` over
  let-chains and `as_chunks`; the code takes them.
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

**Fixing the review of 45–56 (task 57), 2026-09-06.** Read
`docs/REVIEW-E54-E64.md` for the items themselves. Where the review asked for
a decision rather than a repair, this is what was decided.

- *A clock break inside an `rcov` block splits it into **stretches** (R2,
  R10).* The alternative was to refuse `clock`/`max_dclock`/`session` on an
  rcov spec, which would have made the bug unreachable and the model less
  useful -- tick data is exactly where a clock belongs. Each stretch is closed
  the way the group's last one is (leading jitter, interior, trailing
  jitter) and `Γ̂_h` is the sum over stretches, so no product pairs two
  returns across the break. A stretch too short to fill its tail contributes
  what it already emitted. The rewrite carries the phase in `head` rather
  than in a count against `n`, so **no state field was added** and an
  unbroken block is unchanged to the bit.
- *`predict` gets the targets slot, rather than the plumbing refusing it
  (C1).* `OnlineModel::predict_with(x, y, d_clock)` is a defaulted trait
  method that ignores `y`; `bocpd` and `hmm` override it. That keeps
  `predict` out of sample for every model that regresses its targets -- the
  default *is* the old `predict` -- while the two that read a parameter out
  of that slot answer for the row's value. `AnyModel::predict` takes `y` and
  the contract's parity probe calls `predict_with`.
- *A hazard column value `<= 1` is an error, not a null (B1).* It is a
  parameter, not a target: null and non-finite still mean "no value here"
  and fall back to `hazard`, and a finite value that is not a hazard is
  refused naming the row, the way a negative weight is. A row whose
  predictive cannot be evaluated is counted in `solve_failures` instead of
  vanishing.
- *`corrchange` reports a zero-weight row as if it would be learned (CC2).*
  A row is always part of the span reported *on* it -- that is what makes
  the flag out of sample -- and `predict` cannot know the weight, so the
  parity contract fixes the answer. The dead `weight <= 0` branch in `read`
  went; the behaviour is in the docstring.
- *Ties in `refresh_time` are broken by row order (RT4).* Buffering a
  completed grid point until a strictly greater timestamp arrived would cost
  the chunk-invariance the sampler has now, and tick data carries its own
  sequence. Documented, with a test that pins it.
- *No schema bump.* `bocpd.solve_failures` and `BankFile.key_integer` are
  both skipped when they carry nothing, so a state that never hit either
  writes the bytes it always did -- the rule `SCHEMA_VERSION`'s own doc
  records for task 38's sums. The schema-5 fixture was re-frozen anyway,
  because the `hmm` state in it was written by the model H1 corrected; the
  layout did not move and the file exercises the same loader.
- *B4 was reverted, and the revert is the finding.* Renormalising `bocpd`'s
  `logjoint` per row is exact on paper and free -- every output is a
  difference against `z` -- but `z` comes out of `ln`, and libm's last bit
  is not the same on glibc and Apple's. Feeding it back into the **state**
  made a bank saved on macOS continue differently on Linux:
  `state_schema5.rs`'s frozen file stopped reproducing and its three tests
  failed on `ubuntu-latest` alone (CI 34037072202), where the same fixture
  had been green on all three OSes before. That is the general rule worth
  keeping: **a libm result may be compared or reported, but putting one into
  the state costs cross-platform reproducibility.** The joint stays
  unnormalised; `prune`'s docstring and the review item both say why, and
  the drift it would have fixed is ~1.4 nats a row, so it costs nothing
  before about 1e9 rows.
- *Three goldens moved, all deliberately*: `GOLDEN_HMM` (H1),
  `state_schema5.rs` (H1), and the `rcov` rows of the pipeline golden (R2,
  27 → 21 effective returns over three capped gaps). Each carries a comment
  saying which fix moved it.

**State compatibility was dropped once, deliberately, 2026-09-07.**
Renaming spec keys with no aliases ends backward compatibility outright, not
partially: a spec denies unknown fields, so a file naming `coef0` is refused
rather than losing that one value. Rather than hide that behind six
deserializer aliases, `MIN_SCHEMA_VERSION` moved 1 → 6, the four fixtures
that proved older files load were deleted, and with them the schema-1 and
schema-2 conversion paths in `rls`, `sgd` and `kalman` — three untagged wire
enums, their old layouts, `factor_of_inverse`, and the three tests whose
`const assert!(MIN_SCHEMA_VERSION <= 2)` was written to fail exactly when
this happened. A state saved by 0.2.0 must be refit.

This is an exception to hard rule 5, not a repeal of it: the library is days
old, pre-1.0, and getting the names right was judged worth more than the
compatibility. `state_schema6.rs` is the frozen file the *next* layout change
will be held to, and the rule applies from here.

**The documentation pass (task 58), 2026-09-06.** Three rules, so the next
pass does not have to rediscover them:

- *Files and section numbers stay where they are.* The code and the README
  cite `docs/PERFORMANCE.md §11`, `docs/PLAN.md §4.4`, `E56`, `C5`, `T-D4`
  and the like, and a renumbering would silently break every one of those
  pointers. Reorganisation happens *inside* a file — a guide-first section
  in front of a research record (`STATE-WORKFLOW.md`), a reader's map under
  a status line (`PERFORMANCE.md`, `TESTING.md`), two sections swapped into
  numeric order (`PERFORMANCE.md` §5/§6) — and `docs/README.md` is the map
  across files.
- *A record is not rewritten when the code moves on.* Every document under
  `docs/` opens with a dated status line that says what became of it; the
  body below keeps its date. So `CLUSTERING.md` still says "nothing in the
  crates" in a 2026-09-04 sentence, and its status line says `kmeans` and
  `micro` shipped.
- *The README is the guide and the docstrings are the reference; neither
  repeats the other.* Each model's README section states the recursion and
  links to its builder (every keyword with its default) and to its Rust
  module (the recursion as code, with the module comment stating it). A
  default lives in the docstring and in `stream.rs`, and the docstring was
  checked against `stream.rs` for every builder.

**`v0.6.0` tagged, 2026-09-16.** The user asked to push and tag for the next
minor bump. `CHANGELOG.md`'s `[Unreleased]` was cut as `0.6.0`, the version
moved in `pyproject.toml`, `Cargo.toml`, `polars_online.__version__` and both
lockfiles, and `docs/VALIDATION.md` was regenerated on it — it differs from
the committed document in its version line alone, the one other changed line
being a timing its test normalises away. A minor twice over by this package's
own pre-1.0 rule: numbers move in many models, and the state schema is 10.

The tag went on `9f6a5f9`, not on the release-preparation commit `bba23ed`.
Between the two the user read the rewritten README and objected to its first
example: it fitted a *decayed* regression over a folder of files and then
served from the final state, when a decayed fit is local and that state holds
only the last few hundred seconds, so the example taught the wrong habit. The
introduction now leads with a no-decay fit — ridge regression over every row
in bounded memory, where the saved state really is a summary of those rows and
saving and serving is the point — and follows it with a local fit whose
`coef_every=1` coefficient path is read as a time series, which is what a
local fit is for. The paragraph after them lost the sentence restating
no-decay convergence, and the README's running python-block count moved to 59.

That move cost the first rehearsal: `release.yml` dispatched on `bba23ed`
(35103650703) was cancelled once `main` passed that commit, because a
rehearsal certifies only the commit it ran on. The precondition was met again
on the new head — CI 35111714290 and rehearsal 35111713356 both green on
`9f6a5f9` — and `v0.6.0` is annotated there. The tagged release run is
35119720969; its `publish to PyPI` job parks on the owner's approval, which is
the one step nothing here can take, and `v*` tags are immutable, so a tag that
turns out wrong is left unapproved rather than moved.

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

An exponentially weighted mean never forgets. A halflife of `h` leaves
`2^-3 = 12.5%` of the weight on data older than `3h`, `1.6%` older than `6h`,
and nothing is ever exactly zero. Some questions need the other thing: *no
information from before this point*, as a guarantee rather than an
approximation — a compliance window, a regime you believe began at a known
time, a backtest that must not see beyond its own horizon.

### 13.1 The identity

For any accumulator that is a sum of per-row contributions decayed
multiplicatively — `A(t) = Σᵢ wᵢ λ^(t−tᵢ) aᵢ` — splitting the sum at an
earlier time `u` gives

```
A(t) = λ^(t−u) · A(u)  +  (everything after u)
```

The first term is *exactly* the part to discard, and it is the accumulator's
own past value decayed forward. So a truncated accumulator is a subtraction,
not a recomputation:

```
A_window(t) = A(t) − λ^(t−u) · A(u),   u = the boundary
```

Verified in the small before any of this was designed: against a row-by-row
weighted sum over 800 rows on an irregular clock, the identity agreed to
**1.4e-14**, and with every row outside the window set to `1e6` the output
moved by `1.2e-09` where the ordinary EW mean moved by `1.3e+05`.

### 13.2 Which models can honour it, and which cannot

| exact | `ew_cov`, `marginal`, `ew_class`, `ewridge`, `lasso` |
|---|---|
| exact in the accumulator, not in the fit | `huber`, `quantile` |
| impossible | `rls`, `kalman`, `holt`, `sgd`, `pa`, `ftrl`, `kmeans`, `micro`, `hmm`, `bocpd`, `seqtest`, `corrchange`, `rcov` |

The first row is every model whose state is a sum: moments, or a Gram that is
solved *after* the subtraction. The second row is the honest awkward case —
`huber` and `quantile` accumulate rows already multiplied by an IRLS weight
that was computed from a state including rows the window now drops, so the
subtraction gives the accumulator those weights imply, not the one refitting
the window from scratch would produce. The third row is recursive filters,
path-dependent updates, and assignment or test models, where no such
decomposition exists.

`window` is therefore **not** a common parameter. A common parameter in this
library means the same thing in every model (the principle behind hard rule
8), and this one is meaningless in sixteen of twenty-one. It is a per-model
key, refused elsewhere by name, as `seqtest` already refuses `weight`.

### 13.3 The design

- **A view, never a destructive edit.** The model keeps its ordinary EW state
  and a ring of past snapshots; the truncated accumulator is computed at
  report time. Subtracting from the live state would break the recursion and
  compound its own error.
- **The ring holds the window.** Each entry is `(clock, snapshot)`. Before
  reporting, entries older than `t − window` are dropped from the front; the
  boundary is then the front entry. Memory is `O(rows in window × state)`,
  which for `ew_cov` at `k` features is `k²+k+2` doubles per snapshot.
- **The guarantee is one-sided, and the rounding follows it.** Subtracting the
  state as of `u` removes every row at or before `u`, so honouring "nothing
  older than `window`" requires `u ≥ t − window`: the *oldest* snapshot still
  inside the window. Coarse snapshots therefore discard slightly more than
  asked, never less. `window_every = m` snapshots every `m`th learned row and
  divides the memory by `m`; the effective window is then in
  `[window − m·spacing, window]`.
- **Snapshots are keyed by clock and cadenced by the stream, not by the
  chunk.** A cadence counted per chunk would make the boundary depend on how
  the data arrived, and chunk invariance is not negotiable (hard rule 3).
- **The ring is state**, so a bank saved mid-stream resumes with its window
  intact, and `SCHEMA_VERSION` rises to 6. The model's own fields skip when
  absent, so an unwindowed accumulator writes the bytes it always did — but
  spec fields serialize their nulls like every other spec key, so every
  spec's bytes move, which is the same reason 4 and 5 bumped. (Written first
  as "no bump", on the model fields alone; the schema-5 fixture's
  byte-identity test is what caught the spec half.) The bump is worth having
  on its own terms: an older build loading a windowed file would ignore the
  ring and report untruncated statistics.
- **A zero-weight window is a null, not a zero.** When the window holds no
  rows — a clock gap longer than `window` — the denominator is `0/0`, which
  hard rule 9 says to guard rather than propagate.

### 13.4 The oracle

Every claim above is testable without reference to the implementation, and
each test is written from the definition rather than the code:

1. **Against brute force.** For a seeded irregular stream, compare each row's
   truncated moments to a direct weighted sum over the rows inside the
   window. Agreement to `1e-12`.
2. **The guarantee itself.** Replace every row older than the window with
   `1e6` and require the output to move by less than `1e-8`. This is the
   test that would fail if the boundary rounded the wrong way.
3. **Chunk invariance**, as for every model: one chunk against a thousand,
   bit for bit.
4. **Save and resume** mid-window: a bank saved and reloaded continues
   identically, which is what pins the ring into the state.
5. **`window = inf`** reproduces the untruncated model exactly, so the
   feature cannot change what existing specs do.
6. **Refusal**: every model in the third row of §13.2 rejects `window` with a
   message naming the model.

### 13.5 What this must not do to the reader

The risk here is not the arithmetic, it is a user who believes the wrong
thing about the number in front of them. Four ways that happens, and what
the documentation owes each:

- **"Windowed" reads as flat.** A reader who sees `window=3h` may assume a
  boxcar: every row inside counted equally. It is an *exponential* weight
  inside a hard cutoff — the newest row still dominates. Every place the
  keyword is documented must say so in the same breath, and the README's
  entry should show the weight function, not just name it.
- **The guarantee is one-sided and approximate in the other direction.** With
  `window_every > 1` the effective window is shorter than asked, by up to one
  cadence. Documenting `window` as "exactly 3h" would be false; the promise
  is *"nothing older than `window`"*, and the shortfall is stated with it.
- **The boundary is a discontinuity.** A row aging out of a `3h` window at
  `h` removes 12.5% of the weight in one step, so the series has small jumps
  that an EWMA does not. Anyone plotting the two together will see it and
  should have been told first. The continuous alternative — the kernel
  `λ^u − λ^W`, which tapers to zero at the edge — is worth naming in the
  docs even though it is not what ships.
- **Subtraction is not the same arithmetic as accumulation.** The result is a
  difference of two positives, so it loses precision in proportion to what is
  discarded: negligible at `window = 3h` (a 12.5% correction), and worse the
  shorter the window is relative to the halflife. Below `window = h` the
  documentation should say plainly that the mean form is being reconstructed
  from a cancellation, and `docs/PERFORMANCE.md` should carry the measured
  error against brute force at a few ratios rather than a rule of thumb.

Two more the reference has to carry because no test can: that `n_eff` under a
window is the windowed weight, so `min_periods` now gates on a quantity that
stops growing; and that a windowed `ew_cov` is *not* a rolling covariance in
the polars sense — polars' `rolling_*` recomputes each window and costs
`O(n·W)`, this is `O(n)` and matches it to `1e-14`, measured at 24× to 1100×
faster as the window grows from 74 to 4,680 rows. Users who only need moments
on data that fits in memory should be told polars already does this; the
reason to reach for the spec is a stream, a saved state, or a regression.

## 14. Review round R1 (2026-10-03), after tasks 143, 144, 104 and 150

Five read-only reviewers, one per area, each finding re-derived here and
pinned by a test that failed before its fix (the review protocol of
2026-09-14). Committed in three batches; the finding IDs are in the commit
messages.

### Batch A+B: the names (144) and the formula layer (143)

| ID | Finding | Fix | Test |
|---|---|---|---|
| 144-F1 | `with_windows(like=dict)` read an old clock key as unset, and a keyword under an old name was refused as a stray expression | both refused naming the new name (`stream.py`) | `test_renames.py::test_with_windows_refuses_an_old_clock_name_in_like_and_in_a_keyword` |
| 144-F2 | the restart boundary stated as "at least that large" in five live places (code: a step back equal to it is a late row) | wording (README ×2, `stream.py` ×2, CHANGELOG, the plan's decision row) | the existing `test_restart_after_step_back_is_one_rule` |
| 144-F3/F4/F7/F8 | hmm `p_<j>`, PCA `pc<j>_<feature>`, `lam_selected`, `n_eff²` in docstrings and the README | the new names | `test_outputs_doc`, `test_api_surface` (unchanged: the fields were right) |
| 144-F5/F6 | the Unreleased changelog and two README passages in the old vocabulary | rewritten | -- |
| 144-F9 | `EwRidgeCfg.ridge_scale` is a bool under the enum's name | not changed: internal | -- |
| 143-B1 | the increments' state advanced over the whole chunk where a slice or a refusal stopped the core short, so a state saved under `head` depended on `chunk_rows` | snapshot and rewind to the rows consumed | `test_windows.py::test_a_state_saved_under_a_slice_resumes_alike_at_any_chunk_size` |
| 143-B2 | a formula's `cast` was rebuilt non-strict (Polars' default is strict), so an overflow became a silent null | the tree carries `"non_strict"`; `strict_cast` otherwise; a wrapping cast refused | `test_a_cast_is_strict_as_in_polars_unless_told_otherwise` |
| 143-B3 | a long multibyte name panicked the message's cut | cut on a character boundary | `formula.rs::a_long_name_is_cut_on_a_character_boundary` |
| 143-S4 | a step back on the stream's clock restarted every group's windows but not their increments | the runner keeps the stream's clock and restarts every group's increments | `test_a_restart_on_the_streams_clock_restarts_every_groups_increments` |
| 143-S5 | an increment skips a null input, undocumented | decided: kept, as the operators hold a value from the last valued row; documented | `test_an_increment_skips_a_null_input` |
| 143-S6 | std's `DefaultHasher` persisted in the windows state | the bank's `fnv1a` (windows state v2 is unreleased) | -- |
| 143-N7/N8/N9/N10 | one operator rule on both sides; the reserved prefixes refused on inputs; the "already a column" message names `.alias()`; `from_tree`'s log base | as listed | `windows_frame.rs::each_formula_holds_an_operator_and_reads_no_reserved_column`, `test_the_operators_prefix_is_reserved_on_inputs`, `test_a_positional_expression_named_after_a_column_is_told_about_alias`, `test_from_tree_takes_a_log_base_that_is_not_a_literal` |

### Batch C: the window core (143)

| ID | Finding | Fix | Test |
|---|---|---|---|
| C1 | without a clock every backward `"right"`/`"both"` operator was null on every row (the row never waited for a next stamp, so nothing read it) | the row's own windows are read at the end of its push | `test_a_row_count_clock_reads_every_backward_window` |
| C2 | the policy time was a sum of rounded steps, so a row exactly one window later fell on either side of the edge by the sum's noise (999 against Polars' 1000 at 1 ms steps; a row wrongly in at 60 000 ms) | measured from the stretch's first row, one subtraction exact in nanoseconds; the brute forces likewise | `test_a_row_exactly_one_window_later_lands_on_the_edge_at_any_age` |
| C3 | forward `"left"`/`"both"` gave rows at one stamp different windows (a sequence rule) | decided: a window is a set of timestamps -- every row at the stamp, the row itself included, the mirror of backward `"right"`; both oracles follow | `test_forward_left_and_both_hold_every_row_at_the_stamp` (with the mirror identity over repeated stamps) |
| C4 | `partial="keep"` held the last value `gap_cap` past the last row, so a kept rate's span grew with the cap | decided: a cut window ends at the last row seen | `test_a_cut_window_ends_at_the_last_row_it_saw` |
| C5 | a windowed mean on a sparse input walks the queue per read, O(n · window): 0.04 s dense, 0.24 s at one value in 5 000 (300k rows) | not changed; PERFORMANCE §35 | -- |
| C6 | a row exactly one window old under `"left"`/`"both"` counts for `min_samples` and weighs nothing | decided: by the definition, its held interval ends at its row, before the window; a mean with no other value is null (Polars' row-weighted `rolling_mean_by` says the value); documented in `ops.py` | `test_a_row_on_the_far_edge_counts_but_weighs_nothing` |
| C7 | the core took a mean on `half_life = inf` with no window (NaN forever) | refused in `Windows::new` | `windows.rs::a_mean_with_no_decay_needs_a_window` |

### Batch D: the formula targets (104)

| ID | Finding | Fix | Test |
|---|---|---|---|
| D1 | `drop_groups` left the group's window core, so the group fed again met a stale clock | the cores go with the stream | `test_a_dropped_group_starts_cold_in_the_window_core_too` |
| D2 | a boolean column was cast to a number and then refused by the resolver's dtype check | a boolean form in the chunk, for a formula target's columns | `test_a_boolean_column_reaches_a_formula_target` |
| D3 | `rows_learned` counted a formula row at arrival whatever its window gave | counted at release, when a target resolved with a value; chunk-invariant | `test_rows_learned_counts_a_formula_row_once_its_target_resolved` |
| D4 | `partial="drop"` on one operator nulled every formula of the row | a formula is null of its own operator's null; nothing else nulled | `test_drop_on_one_target_leaves_the_rows_other_targets` |
| D5 | the resolver snapshot cloned every core each chunk | taken only when a ring of the spec's streams could refuse the chunk | -- (the clone is gone from the default path; a budget that refuses is the exception) |
| D6 | per-group serial Polars plans per chunk | not changed; PERFORMANCE §35 | -- |
| D7 | a formula's spans outside `clock_spans`, so the embargo check compared unlike units | the operators' spans are the spec's clock spans | `test_a_formulas_spans_are_the_specs_clock_spans` |
| D8 | a `pl.Expr` in a raw spec dict died in `json.dumps` | normalised in `ModelBank.__init__` | `test_a_raw_spec_dict_takes_a_window_expression` |
| 150-* | the Kalman reviewer: no code defect; four stale equations fixed inside task 150 | -- | -- |

### Round two (the same day): three reviewers over the fixes and the surfaces

| ID | Finding | Fix | Test |
|---|---|---|---|
| P1 | `fit`'s learn-only flag on the bank could be left set when a concurrent `predict` held the bank; `fit_predict` then ran in sample | learn-only is a parameter of each call (`Bank::fit_predict_from_with`, the binding, `ModelBank._feed`); the bank keeps no flag | `test_learn_only_is_the_calls_not_the_banks` |
| P2 | `_order_free` exempted a formula target from the row-order warning | a target that reads the rows ahead is never order-free | `test_a_formula_target_is_never_order_free` |
| P3 | TOML has no null, and a `when/then` without `otherwise` carried one | the tree writes a null literal as `["lit"]`; `["lit", null]` still reads | `test_a_null_literal_has_a_form_toml_can_carry`, `formula.rs::every_node_kind_round_trips` |
| P4 | `--no-output` is the command line's `fit` but ran `fit_predict` | it runs learn-only | `test_no_output_is_the_command_lines_fit` |
| P5/F5 | a `pl.Expr` in a raw spec dict died in `json.dumps` on every surface but the constructor, and a tuple of targets | `_json` writes it as the builders would | `test_every_surface_takes_a_raw_dict_with_a_window_expression` |
| P6/P8 | `rows_learned`'s docstring said "counted as handed over"; `n_eff`/`window` leftovers | reworded | -- |
| P7 | `--predict` kept the config's `closed_groups`, which the scoring run refused | dropped with `save_state` | `test_no_output.py::test_predict_drops_the_closed_groups_sidecar` |
| P9 | `rolling_metrics` compared a duration window with a missing clock's dtype first | the column is named first | `test_eval.py::test_rolling_metrics_names_a_missing_clock_before_reading_the_window` |
| P10 | `ridge_scale="foo"` was refused by serde with no parameter name | checked by value in the builder | `test_error_messages.py::test_ridge_scale_is_checked_by_name` |
| F1 | (D5) with two formula specs, the second's refusal left the first's core fed | every spec's frame and cores are built before any is fed; the snapshot is taken wherever a feed can still fail after rows went in (a cast, a second formula spec, a refusing ring) | `test_a_refused_chunk_leaves_every_specs_core_as_it_was` |
| F2 | (B2) `to_json` dropped `"non_strict"`, so a saved spec ran a strict cast | the third argument is written | `test_a_non_strict_cast_survives_the_specs_round_trip`, the round-trip list |
| F3 | (D2) a boolean a feature also read reached the formula as a number (the first form found) | the boolean form is looked up first, and a formula's columns before the clock's and the session's | `test_a_boolean_also_read_as_a_number_reaches_the_formula_as_a_boolean` |
| F4 | a plain target under an embargo counts at arrival and a reset discards the pending rows, so `rows_learned` counts rows no model saw (pre-existing) | not changed; recorded here | -- |
| W1 | edge decisions subtract two rounded clocks, so a row exactly one window after any row but the stretch's first can land on the wrong side (`[0, 100, 400]` ms, `300ms`, `"left"`: null against Polars' 1) | (round three) an edge between two rows is decided from the difference of their raw clocks, exact in nanoseconds, compared as integers with the window's own nanoseconds (`gap`, `KernelDef::window_ns`; a window given as a number keeps the seconds path); the queues and the waiting rows carry each row's raw clock, one `i64` a row; windows state version 3 | `test_a_row_exactly_one_window_from_another_lands_as_in_polars` (against `rolling_sum_by`, at epoch 0 and at 2024), `windows.rs::an_edge_is_decided_from_the_two_rows_clocks` (the brute force decides its edges the same way) |
| W2 | under `session_gap="reset"` a session change is a reset, so it discards (ignores `partial`) | decided: that is what `"reset"` asks for; documented. Under `group`, a silence past `gap_cap` cuts the group's windows as the stream's clock passes the cap, before its next row can say its session changed, so a reset there discards only what is still open (round four, A1) | `test_a_silent_groups_session_reset_after_a_capped_gap_is_a_cut`, `windows.rs::a_silent_groups_reset_after_a_capped_gap_is_a_cut` |
| W3 | the stream's clock cut every group at a session change, so groups with sessions of their own restarted at every row | (round three) with a group column the stream's clock takes no session (`Windows::set_grouped`); without one the stream is the one group and its session is the stream's, which is what the bank's per-group resolvers (one core per group, no group column) need to see a session change as the bank's own clock does | `test_with_groups_a_session_is_each_groups`, `windows.rs::with_groups_a_session_is_each_groups` |
| W4 | a state saved under a slice holds rows read past the last emitted, which a resume from the emitted count feeds again | (round three, redone in round four after B1 to B3) the state records the rows of its input the run consumed, skipped or fed (`WindowsRun::consumed`, `save_bytes_with`; `with_windows` passes it under a slice), and the input's first clock; a run resumed on the same input, unsliced, skips them, one on an input starting elsewhere skips nothing; an error's row number counts the skipped rows | `test_a_state_saved_under_a_slice_resumes_on_the_same_input` (a chain of six, with and without a `"drop"`), `test_rows_held_from_an_earlier_input_are_not_this_inputs`, `test_a_state_saved_under_a_slice_skips_nothing_of_another_input`, `windows_frame.rs::a_state_saved_under_a_slice_resumes_on_the_same_input` |
| W5 | a v2 windows state written before round one loads with defaults and misbehaves | (round three) `WINDOWS_VERSION` 3; the magic and the version are read before the rest, so an old state is refused by number and not by a field it lacks | `test_a_windows_state_of_another_version_is_refused_by_its_version`, `windows_frame.rs::a_state_of_another_version_is_refused_by_its_version` |

**Round three (the same day)**: W1, W3, W4 and W5 built as the table says,
each with its failing test first. W3 was refined while building: a session
is each group's only where there is a group column, since without one the
stream is the one group, and the bank's per-group formula resolvers (one
core per group, no group column) must see a session change exactly as the
bank's own clock does. A clock that starts over with a session is then a
step back on the stream's clock under `group`, which the docstring says.
The cost of W1, measured (PERFORMANCE §36): the first build converted every
gap to seconds and cost 12% of the window core's time at 16M rows (7% on
four kernels); comparing integers with the window's own nanoseconds
(`KernelDef::window_ns`) leaves 3%, the raw clock written and moved with
each queued row, and memory unchanged. The converting build was never
committed.

### Round four (the same day): two reviewers over rounds two and three

| ID | Finding | Fix | Test |
|---|---|---|---|
| B1 | the skip a sliced run saved counted this run's fed rows less the rows returned with new sequence numbers: rows a loaded state held and returned were not counted, nor the rows the run skipped, so a chain of sliced runs broke on its third run (a step back refused) | the state records the rows of its input the run consumed, skipped or fed (`consumed`), with no count of rows returned; the contract is a resume on the same input, unsliced | `test_a_state_saved_under_a_slice_resumes_on_the_same_input` (six sliced runs, then the rest), `windows_frame.rs::a_state_saved_under_a_slice_resumes_on_the_same_input` |
| B2 | a run on the next file returns the rows the state held from the file before; counted as this input's, a slice over them saved a skip that dropped never-fed rows, and under the `df.slice(n)` contract no skip could be right once rows returned exceeded rows consumed | as B1: consumed rows only, and the same-input contract | `test_rows_held_from_an_earlier_input_are_not_this_inputs` |
| B3 | a slice that lands on the input's last output row saves a skip meant for the same input, which a run on the next file would apply to the file's first rows | the state remembers the input's first row's clock; a run on an input that starts elsewhere skips nothing; without a clock column the docstring says to resume on the same input | `test_a_state_saved_under_a_slice_skips_nothing_of_another_input` |
| B4 | a state whose magic and version matched but whose body did not read was reported as "not a state it saved" | it says the state is damaged, or written by a build that changed a field without bumping the version | -- |
| B5 | the wording of W4 (docstrings, the binding, this table, the changelog) stated the count as rows read past the last returned | reworded with the contract | -- |
| A1 | under `group`, a session change under `session_gap="reset"` that follows a silence past `gap_cap` is a cut, not a discard: `end_silent` cut the group at another group's row before the group's own reset fired; without `group` the one clock sees both at one row and the reset comes first | recorded as the rule (the cut cannot be undone once rows drained, and delaying it would unbound how long a group holds output); the W2 row, the `with_windows` docstring | `test_a_silent_groups_session_reset_after_a_capped_gap_is_a_cut`, `windows.rs::a_silent_groups_reset_after_a_capped_gap_is_a_cut` |
| A2 | `WINDOWS_VERSION` 2 to 3 changed the bytes a bank file embeds per formula target while `SCHEMA_VERSION` stayed 23, so a 23 bank holding one passed the bank's check and failed at a group's first chunk | `SCHEMA_VERSION` 24, the bank's minimum 24 (23 unreleased, pre-1.0) | `test_a_bank_state_from_before_the_windows_state_changed_is_refused_by_number` |
| A3 | `a_saved_state_resumes_at_any_row` built its fresh core on a grouped stream without `set_grouped`, passing by coincidence | the flag set | the test |
| A4 | `KernelDef::check` did not tie `window_ns` to `window_size`, so a direct caller could split membership from the far edge and `complete` | refused unless `seconds_of_ns(window_ns) == window_size`; the frame's increments-only kernel carries its window in nanoseconds so a temporal clock converts nothing | `windows.rs::a_kernels_window_ns_must_be_its_window_size` |

### Round five (the same day): one reviewer over round four's changes

| ID | Finding | Fix | Test |
|---|---|---|---|
| C1 | the saved skip left out a loaded skip not yet applied: a resumed run that satisfied its slice from held rows while the skip still spanned chunks saved too small a count, so the chain depended on `chunk_rows` (hard rule 3); a run that saw no row dropped the loaded identity | `consumed` = skipped + the skip still pending + fed; a run that saw no row saves the identity it loaded | `test_a_state_saved_under_a_slice_resumes_on_the_same_input` over `chunk_rows` 1, 7 and 100,000; `windows_frame.rs::a_resumed_run_fed_one_row_at_a_time_saves_the_whole_count` |
| C2 | a sliced run whose input ended before the slice was satisfied saved a skip of 0, so the chain's next run fed the input again | the count is saved whenever a slice was asked for | the chain run to exhaustion in the same test |
| C3 | the identity by the first clock alone took another input that starts at the same clock (a clock that starts over each day) for the same one, and skipped its rows silently | the state knows its input by the rows it holds -- the unresolved tail of the consumed prefix, the input's own rows -- compared with the rows skipped at those positions when the skip completes; where it holds none, by the last skipped row's clock against the last clock it saw; an input that ends before the skip completes is refused at `finish`, and at `save` when the caller says the input ended (`input_ended`), since a slice satisfied mid-skip is a legitimate save | `test_a_state_saved_under_a_slice_refuses_another_input`, `windows_frame.rs::a_sliced_state_refuses_another_input_and_a_damaged_file_says_so` |
| C4 | the old `df.slice(n)` contract was silent without a clock column, with tied stamps, or under `restart_after_step_back` | refused by the same identity, with and without a clock column; a stream whose rows can repeat (a daily grid with the same values) is documented as resuming on the same input only | the same tests |
| C5 | round four changed what `resume_skip` counts and added `resume_first` under `WINDOWS_VERSION` 3 | `WINDOWS_VERSION` 4 | the version tests read 4 |
| C6 | a state cut short failed in the header read and was reported as "not a state it saved" | a header that cannot be read says "cannot be read: not a state it saved, or damaged"; B4's message stays for a header that read and a body that did not | `test_a_damaged_windows_state_says_so`, the Rust test above |
| C7 | the plan's task 104 still said schema 23 | amended | -- |
| C8 | the `with_windows` docstring and the changelog promised the chain under every chunk size and slice | true now; the identity and its one limit stated | -- |

### Round six (the same day): one reviewer over round five's additions

| ID | Finding | Fix | Test |
|---|---|---|---|
| D1 | the last-clock identity was dead code: the held snapshot was set even when empty, so a sliced state holding no rows (a backward operator under `closed="left"` or `"none"`, any operator on a row-count clock) had no identity, and another input or `df.slice(n)` was skipped silently; the changelog's "with and without a clock column" was false there | the state keeps the last row it read (windows state version 5), and the identity is the last rows read: the rows it holds that are the input's, or the last row where it holds none -- one path, no fallback | `test_a_sliced_state_that_holds_no_rows_knows_its_input_by_its_last_row`, `windows_frame.rs::a_sliced_state_that_holds_no_rows_knows_its_input_by_its_last_row` |
| D2 | a run on the next file under a slice still holds the previous file's unresolved rows ahead of its own (rows go out in order, so none of its own went out while one of theirs waited), and the identity took them all for this input's: the held rows outnumbered the rows consumed, and a resume on the second file was refused as another input (traced by the author before the report, confirmed by a failing test) | the identity is the last `consumed` held rows at most | `test_a_sliced_run_on_the_next_file_resumes_with_the_first_files_rows_held`, `windows_frame.rs::a_sliced_run_on_the_next_file_resumes_with_the_first_files_rows_held` |
| D3 | an input starting at the last stamp the state read -- a file boundary inside a tied stamp, ordinary with coarse stamps -- was refused as "at or before the last row", words that did not say a split inside one stamp is "at" | kept, by decision: a tie cannot be told from the same input sliced inside its last stamp, and a loud refusal beats a silent double feed; the message and the docstring say so, and that the next file starts after the last stamp read | `test_a_next_file_starts_after_the_last_stamp_the_state_read`, the Rust test under D4 |
| D4 | a clock that starts over each day, under `restart_after_step_back` or with a `session` column, could never present an accepted next-day file after a sliced run: the rule refused every step back before the policy spoke, where an unsliced state took the file | the first row is put to the core's clocks without taking it (`Windows::peek`, the two advances `push` makes, shared): a step forward or a new start by the policy's word (a restart, a new session) is the next file; a step back the policy refuses, or the same stamp, is refused | `test_a_sliced_state_takes_a_new_start_by_the_policys_word_for_the_next_file`, `windows_frame.rs::a_sliced_state_takes_a_new_start_by_the_policys_word_for_the_next_file` |
| D5 | `WINDOWS_VERSION` 3 to 4 moved alone, round four's A2 again: a 24 bank holding a version-4 core failed at a group's first chunk | `SCHEMA_VERSION` 25, the bank's minimum 25, and a tripwire pairing the two numbers so the next move of either fails until both move | `windows_frame.rs::a_windows_state_version_moves_the_banks_schema_with_it`, `test_a_bank_state_from_before_the_windows_state_changed_is_refused_by_number` (23 and 24) |
| D6 | the binding's `Windows.save` doc, the changelog, and the docstring's error paragraph: what the run refuses while the plan runs surfaces as `polars.exceptions.ComputeError`, not `ValueError` | amended; the docstring states the rule for an input that starts elsewhere as a list | -- |
| D7 | two of round five's claims had no test: a save under `input_ended` with a skip pending, and a run that saw no row saving the identity it loaded | pinned | `windows_frame.rs::a_resumed_run_that_saw_no_row_saves_the_identity_it_loaded`; the `input_ended` leg of `test_a_state_saved_under_a_slice_refuses_another_input` |

### Round seven (the same day): one reviewer over round six's additions

Folded into round six's commit: round seven adds a field to the windows state, and no commit should carry version 5 without it.

| ID | Finding | Fix | Test |
|---|---|---|---|
| E1 | `push` lost its doc comment to the new `step_clocks`, inserted between the comment and the function | the comment moved back | -- |
| E2 | a clock that starts over at the *same* stamp each day gives the next file the saved input's first clock, so the rule never reached the policy and the identity refused the file; round six's test dodged it with a start of 0.5 | the state knows its input by the first row's session beside its clock, so with a session column the file differs at its first row and is put to the policy; without one it is the saved input until the rows differ, refused by name -- the limit the docstring states | the D4 tests with day 2 at the same first stamp, both legs |
| E3 | the last row's columns check ran on every load, so an unsliced state holding no rows refused the next file with a column more, which it took before; the bank wrote the row for every core on every save and never read it | the row is written and read under a skip only; the columns message pinned | `windows_frame.rs::the_last_row_binds_a_sliced_state_only` |
| E4 | the last row went out of step with `consumed` on two paths: a skip still pending recorded the last skipped row while the count included the rest, and a refusal mid-chunk held rows without recording the last | by construction: the row is set when the skip completes (the loaded row until then) and in the refusal branch | round eight's F3: the first path is reachable at the Rust API (a limit of 0 on a chunk shorter than the skip), pinned there |
| E5 | the bare leg of the no-rows test was refused by the count (three rows against a skip of four), never the compare; the no-row re-save test pinned the count, not the identity | another input of the same length; a `closed="left"` leg resuming the re-saved state on another input | the same tests |
| E6 | the docstring lost R4-B3's "without a clock column, resume on the same input", and did not say a hand slice starting after the last clock read (possible after a restart) is taken as the next file | both stated | -- |

### Round eight (the same day): one reviewer over round seven's additions

| ID | Finding | Fix | Test |
|---|---|---|---|
| F1 | without a clock column, a session column made any hand slice (or overlapping file) that starts in a later session the next file, silently: the rule fired on the differing first session, and on a row-count clock every row is a step forward, so the policy's "forward" was evidence of nothing | without a clock column only a new start (a new session by the policy's word) is the next file; a step forward there is refused by name, and the docstring says a next file on a row-count clock begins with a new session | `test_without_a_clock_the_next_file_begins_with_a_new_session`, `windows_frame.rs::without_a_clock_the_next_file_begins_with_a_new_session` |
| F2 | where the policy refused the first row (under `group` a next-day file's step back is refused on the stream's clock whatever its session), the message dropped the policy's own words, which name `restart_after_step_back` as the unsliced path does | the refusal appends the policy's message | `windows_frame.rs::a_refused_first_row_carries_the_policys_words` |
| F3 | round seven's E4 row said the pending-skip path was unobservable; it is reachable at the Rust API (a limit of 0 on a chunk shorter than the skip) and from the generator (`.head(0)` with `chunk_rows` below the pending skip), and the fix had no test | pinned; the E4 row corrected | `windows_frame.rs::a_save_inside_a_pending_skip_keeps_the_loaded_identity` |
| F4 | the `save_bytes_with` and binding docs named neither the session nor the last row; the docstring's "as one can after a restart" read as the only way a hand slice starts after the last clock read | amended | -- |
| F5 | a file with a skip, no held rows and no last row (which this build cannot write) would carry an identity of zero rows and pass the check vacuously | refused at load as damaged | `windows_frame.rs::a_sliced_state_without_its_identity_is_damaged` |
| F6 | the E3 test asserted the next file's height only, not that its extra column was carried | asserted | the same test |

### Round nine (the same day): one reviewer over round eight's additions

Folded into round eight's commit. The reviewer found no wrong verdict or number in the new paths; the iteration ends here.

| ID | Finding | Fix | Test |
|---|---|---|---|
| G1 | under `group` without a clock column the no-clock refusal said "starts in the session the state last read" of a group the state never read (a fresh group's clock sees no session change, so the verdict is a step forward, refused by F1's reasoning) | the message says what is true in both cases: the first row continues the session last read, or is of a group the state has not read; the docstring says the next file begins with a new session of a group the state has read | the same tests, which now pin the words |
| G2 | the docstring's no-clock sentence lacked the qualifier the clock sentence has: a next file whose first row's session is the saved input's first session is taken for the saved input until the rows differ (cyclic session labels) | stated: the next file begins with a new session and not with the saved input's first | -- |
| G3 | the binding's `Windows.save` doc still said a step forward skips nothing; the docstring's refusal list omitted the no-clock step forward; the changelog said "a later session" where any session but the saved input's first was the bug | amended | -- |
| G4 | the F1 tests asserted "another input" alone, which an arm that let the skip apply would also give | the F1 words asserted | `test_without_a_clock_the_next_file_begins_with_a_new_session`, `windows_frame.rs::without_a_clock_the_next_file_begins_with_a_new_session` |
| G5 | the F3 row understated the reach of the pending-skip path: the generator reaches it too (`.head(0)` with `chunk_rows` below the pending skip) | the row amended | -- |
| G6 | the policy's message appended by F2 carried its own `with_windows:` prefix inside the rule's | stripped | `windows_frame.rs::a_refused_first_row_carries_the_policys_words` |

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

Six read-only reviewers, one per area of the production diff, each finding
re-derived here and pinned by a test that failed before its fix (the review
protocol of 2026-09-14). The reproductions are in the session's scratchpad
(`review/<area>/`). Committed in three batches; the finding IDs are in the
commit messages. Severity: high = wrong numbers silently; medium = wrong
numbers in a narrow case, or a wrong refusal or broken contract, loudly;
low = a doc or an edge.

| ID | Sev | Finding | Fix | Test |
|---|---|---|---|---|
| B1 | med | task 158's same-stamp refusal (E13) fired inside the bank's formula-target resolver, so a bank with a window target saved between two rows at one stamp refused to resume, in every entry point; the column form resumed | the check runs only where `with_windows` feeds the core, not where the bank's resolver does (`resolving`): the bank's own clock has refused any real step back first | `test_formula_targets.py::test_a_state_saved_inside_a_tied_stamp_resumes_as_one_run` (an f64 clock, three rows a stamp; a `Date` clock, grouped, four rows a day) |
| F1 | med | `feed_inner` marked the run started, and kept its input's first row, before the identity verdict, so a refused first chunk was taken whole on the next call; with two groups the retry wedged the bank | the record is written after the verdict | `windows_frame.rs::a_refused_first_chunk_is_refused_again` |
| D1 | high | `rcov`'s pre-averaged estimate forms its first `Ȳ` from `Δ₂..Δ_{k}` (CKP's `Ȳ₁`), never `Ȳ₀`, one term fewer per stretch, and scales by `n/(n−k+2)` as if it had them all; every oracle shared the code's indexing | `Ȳ` is formed once the ring holds `k − 1` returns, from them; the scale is `n / pre_count`, the terms summed | `rcov.rs::the_preaveraged_estimate_is_its_definition` and `::the_scale_counts_the_terms_summed_across_a_break`, `test_rcov.py::test_the_preaveraged_estimate_is_its_definition`, all from Eq. 9; `GOLDEN_RCOV_PREAVG` re-pinned |
| R1 | med | `ewridge(ridge_scale="sum")`'s prior is not decayed across a zero-weight row while `w_sum == 0`: 6.8e-2 from its definition after two such rows, where `rls` matches | `EwCov::update` at `w == 0` with `lam·W == 0` is the decay, head row included; the Gram split for that row goes | `ewcov.rs::a_zero_weight_head_row_ages_the_prior`, `gaps.rs::a_row_of_no_weight_parts_no_targets`, `test_rls.py::test_leading_zero_weight_rows_age_the_prior_as_rls_does` |
| R2 | med | `seconds_of_ns` rounds a negative sub-second delta at the scale of a second, so on a `Datetime` clock a step back of exactly `restart_after_step_back` at 1-3 ms restarts the model instead of being a late row | `seconds_of_ns` takes the magnitude and mirrors the sign | `clock.rs::a_step_back_is_the_mirror_of_the_step_forward`, `test_renames.py::test_the_restart_edge_is_inclusive_at_a_millisecond_on_a_temporal_clock` |
| W3 | med | on an f64 clock a window edge is decided from two origin-subtracted policy times, not the rows' raw clocks, so a row exactly one window back lands outside where Polars keeps it | a number clock's policy time is the clock itself; windows state 6, schema 26 | `test_windows.py::test_a_number_clock_decides_an_edge_from_the_two_rows_clocks` |
| W2 | med | `KernelDef::mass` cancels at long half-lives: 6e-5 at `h/t = 1e12`, every mean null from 1e17 | `mass` by `exp_m1`, and both oracles with it | `windows.rs::mass_at_a_long_half_life_is_the_even_kernels`, `test_windows.py::test_a_long_half_life_agrees_with_the_definition_and_tends_to_the_even_one` |
| F2 | med | group and session keys are the value's string form and the state carries no dtype, so an `Int64` file followed by a `Float64` one starts every group cold, and a session column's dtype change cuts the windows | the windows state carries the group and session columns' dtypes, by role, and a resumed input of another is refused by name (windows state 7, schema 27) | `windows_frame.rs::a_state_refuses_a_group_column_of_another_dtype`, `test_windows.py::test_a_state_refuses_a_group_or_session_column_of_another_dtype` |
| W1 | med | a stream restart discards every group's windows but not their clocks, so a group's next row within `restart_after_step_back` of its stale clock is refused as a late row | a stream restart starts every group's clock over, the row's own group's from a fresh one; the brute force too | `windows_frame.rs::a_stream_restart_restarts_every_groups_clock`, `test_windows.py::test_a_restart_on_the_streams_clock_restarts_every_groups_clock` |
| P1 | med | a window target without `.alias()` named after its own input column passes the builder and can never run; the bank's refusal is worded for `with_windows` | the builder refuses a target named after a column its formula reads, naming `.alias`; a bank's refusal of a formula target drops the core's `with_windows` prefix | `test_formula_targets.py::test_a_target_named_after_its_own_column_is_refused_at_the_builder` |
| F4 | low | `inf` and `nan` serialize as `{"Dyn": {"Float": null}}`, so `clip(0, inf)` is refused as "this literal (…null)" | a null value under a float's type is refused as a literal that is not finite, naming inf and nan and the one-bound clip | `test_windows.py::test_a_literal_that_is_not_finite_is_refused_by_name` |
| D2 | low | `bocpd`'s `prior_nu > 1` floor for `diag`/`robust` buys a finite mean; the message says variance | the message names the mean or the variance by emission | `bocpd.rs`, the floor test |
| B2 | low | `predict` without a clock column writes the stream's fed-row count as `scored_clock`; the docs say the row's index in its group | `predict` without a clock column writes the row's index in its group, counting on from the rows learned | `test_clocks.py::test_predict_without_a_clock_counts_each_row_in_its_group` |
| B3 | low | the unset-policy refusal advises setting `restart_after_step_back` "to the smallest one that does", where equal is a late row | the message says to set it below the step | `test_renames.py::test_restart_after_step_back_is_one_rule` |
| W4 | low | at a tied stamp `ewm_sum` gives every tied row the stamp's total, where Polars' `ewm_sum_by` is a running sum; four places say the latter | the four places say: on distinct stamps; at a repeated stamp every row carries the stamp's total | `test_windows.py::test_at_a_repeated_stamp_every_row_carries_the_stamps_total` |
| W5 | low | a subnormal `half_life` passes `check` and emits `inf` rates | `check` refuses a half-life below `f64::MIN_POSITIVE` | `windows.rs::a_subnormal_half_life_is_refused`, `test_windows.py::test_a_subnormal_half_life_is_refused` |
| W6 | low | an operator input that is NaN, ±inf or above 1e100 reads as missing; the docs say only a null is skipped | `ops.py` and the README say a NaN, infinite or beyond-1e100 value is read as missing | -- |
| F3 | low | a typed literal is rebuilt dynamic, so a formula's dtype and last bits can differ from Polars' | a typed literal is carried as a cast of the plain one, so its dtype and bits are Polars' own | `test_windows.py::test_a_typed_literal_keeps_its_dtype` |
| F5 | low | the Rust tree writes a clip's absent bound as a bare `null`, where Python writes `["lit"]` | the Rust tree writes `["lit"]`, reading a bare `null` still | `formula.rs::every_node_kind_round_trips`, `test_formula_targets.py::test_a_clip_bound_left_out_is_the_null_literal_in_a_saved_spec` |
| P2 | low | a chain of two plan forms over an empty frame raises `ConsumedSourceWarning` | a source that got no rows from a Python scan warns only when none of this package's own sources ran under it | `test_consumed_source.py::test_a_chain_of_two_plan_forms_over_an_empty_frame_does_not_warn` |
| P3 | low | `output_index`'s `dtype` vocabulary omits `enum`, `i32`, `i64` and `clock` | the docstring lists all eight | `test_model_registry.py::test_output_index_names_every_dtype_it_declares` |

What held, measured: hard rules 2, 3, 8 and 9 across every model and
both plan forms (chunk invariance bit-identical over 1,008 window
configurations and 198 pushdown cases; the embargo's release against an
oracle from its definition; zero-weight first rows NaN-free in 18 kinds);
every renamed name refused by name in builders, dicts and TOML; save/load
at 159 cuts equal to one run; `bocpd`, the sequential `corrchange`,
`kalman`, `holt` and `hmm` against oracles from their papers; the `.pyi`,
`po.eval` and 99 docstring defaults.

## 16. Whole-project review at `e6b70a2` (2026-10-05): every file but Markdown

Sixteen read-only reviewers, one slice each (six over `online-core`, three
over the polars layer, two over the Python package, three over the tests,
one over the scripts, one over CI), every finding re-derived from its
reproduction (`review2/<slice>/` in the session's scratchpad) and recorded
in one list before any fix. The fixes: four workers in worktrees and the
coordinator, each test shown failing on the old code, each branch reviewed
and merged under its own gate (task 160). Severity as in §15: high = wrong
numbers silently, or a check that could not fail where it was the only one;
medium = wrong numbers in a narrow case, a crash, or a wrong refusal;
low = a doc, a message or an edge.

| ID | Sev | Finding | Fix | Test |
|---|---|---|---|---|
| SC1 | high | `compare_release.py`'s verdict ignored the specs the new build refuses, so a refused spec dropped out and "identical" exited 0 | a spec the old build ran and the new one did not is a difference, naming the refusal | `test_release_compare.py::test_a_spec_this_build_refuses_is_a_difference_not_a_silence` |
| TA1 | high | the one universal rule-2 test perturbed the first non-null target, a row every model withholds: 0 of 1,200 comparisons saw a prediction | the row is drawn from those the model scored; at least 10 streams must compare a prediction that was there (ewridge's inflation gate off in this test, since it withheld 87% of the streams) | `test_properties.py::…::test_prediction_never_depends_on_the_current_target` (no rule-2 failure found) |
| TB1 | high | the kmeans/micro oracle is the Rust transcribed bit for bit; the independent checks were coarse | kept as a regression check, beside checks from the definitions: centres, weights and radii from the raw assigned rows; far, merge and dead decisions row by row; micro's admissions and summaries; scikit-learn `KMeans` (ARI 0.99995) | `test_kmeans.py`, `test_micro.py` `TestDefinitions`; TESTING.md's table |
| TC1 | high | `kalman_ref` and `robust_ref` restate the implementation; the library second opinions covered kalman only unstandardized and huber only at delta 1e9 | the standardized kalman held to filterpy on rows standardized in numpy (1e-9) and `coef·x[t+1] = pred[t+1]` (1e-12); the quantile fit to its smoothed check loss's stationarity (1e-15); the nudge to its purpose, which found TC1b | `test_second_opinion.py`, `test_robust.py::TestTheQuantileFitsDefinition` |
| TC1b | high | found by TC1 (iii): `quantile`'s nudge was bounded by the band Gram's diagonal leverage; on correlated features the next solve moved the row by 6 to 127 times its residual, past its target | the full leverage `1 + u'A⁻¹u` against the band's own system, from a kept `SpdFactor` that is `solve_spd`'s to the bit; `GOLDEN_QUANTILE` and four golden-pipeline values moved; the reference oracle and the stationarity test's reference take the full leverage too. Costs half the default schedule's throughput (2.6M against 5.0M rows/s, 10 features; 8% solving every row): a decision | `robust.rs::a_nudge_never_moves_its_row_past_its_residual`, `::the_full_leverage_is_the_diagonal_one_on_uncorrelated_features`, `test_robust.py::…::test_a_nudge_never_moves_its_row_past_its_residual` |
| PB1 | high | under an embargo the drift detector ran on the learn pass, so `drift_<t>` was never true and `drift_action="reset"` restarted silently | the flag goes on the row that released the label that tripped it (`RowPlan::drift_ri`), the restart before that row is scored | `test_label_delay.py::TestDriftUnderAnEmbargo` (8) |
| PB2 | high | `gap_cap` without a clock passed and capped the row-count step: `gap_cap=0.5` halved every decay, marked every row capped, cleared lags every row | refused, in specs and `with_windows` | `spec.rs::gap_cap_needs_a_clock`, `test_gap_cap_without_a_clock_is_refused_by_name` |
| PC2 | high | a forward `right`/`both` window whose far edge is exactly the stretch's last row before a break was cut as partial: null, or dropped under `partial="drop"` (2,309 of ~22k fuzz rows) | whole, as its mirror is. A reset still discards it, as documented: a decision | `test_windows.py::test_a_forward_window_whose_far_edge_is_the_last_row_before_a_break_is_whole`, the Rust mirror test across breaks |
| PA1 | high | `load_bytes` indexed specs by the states' count: a file with an extra states entry panicked, one with fewer lost a spec's groups | refused as damaged | `bank/damaged_file_tests.rs::a_file_whose_states_do_not_match_its_specs_is_refused` |
| PA2 | high | closed rows restored unchecked (`closed_groups()` panicked, `save_bytes` re-saved the poison); counts at `u64::MAX` wrapped | every queued row checked against its spec at load; counts saturate, and a summary whose counts wrap is refused | `::a_closed_row_that_does_not_fit_its_spec_is_refused`, `::a_row_count_at_the_top_of_its_range_saturates`, `::a_summary_count_that_wraps_is_refused` |
| PA3 | high | an `Int128` key past `i64` was cast non-strict to a null group, merged with real nulls | read as its text, marked an integer so `"monotone"` orders it as a number | `integer_group_keys_match_the_string_cast` (Int128), `a_monotone_int128_key_is_ordered_as_a_number` |
| PA4 | high | `refresh_time`'s group keys round-tripped through String: `Datetime`/`Time`/`Struct` came back null, `Boolean` and zoned failed | the input's own values at the completing ticks; a zoned key by its instant | `test_refresh_time.py::test_the_group_column_is_the_inputs_at_the_completing_ticks` (8 dtypes), `a_zoned_datetime_group_comes_back_as_it_went_in` |
| CI1 | med | `dtolnay/rust-toolchain@stable` is a branch, at 13 sites, the release's wheel build among them, against `.github/dependabot.yml`'s rule that actions are pinned to commits | pinned to the `stable` branch's head commit, the version in a comment | `test_ci_cost_policy.py::TestActionsArePinnedToCommits` |
| CI3 | med | three of the six wheels (Intel macOS, aarch64 Linux, musl) were built and uploaded without ever being imported, and none was run as the file that ships | each build installs its wheel into a fresh environment, dependencies from PyPI, and runs `scripts/wheel_smoke.py` (the version, a fit, a state round trip, the streaming plan); the musl wheel in `python:3.12-alpine` | `test_release_workflow.py::test_every_wheel_is_installed_and_run_where_it_belongs` and the two smoke tests; run here on a built arm64 wheel in a clean venv |
| SC2 | med | REGIMES.md §8's prose kept 0.151 where its table read 0.150 after task 159 | 0.150, and the section dated | `test_regimes_doc.py`: sections 1, 5, 7 and 8 against the experiments that print them (macOS) |
| SC3 | med | no benchmark printed its commit, versions, platform or load: a 576 ms reading at load 524 stood against the README's 155 | `scripts/bench_header.py`, printed first by the seven benchmarks | `test_bench_scripts.py` |
| SC4 | med | `parallel_bench.py` ran BSD `time -l`, and `windows_bench.py` imported `resource` at the top: neither ran on Linux or Windows | memory by platform: `time -l` on macOS, GNU `time -v` on Linux, NaN elsewhere | `test_bench_scripts.py::test_the_platform_specific_benchmarks_import_everywhere`, `::test_the_memory_measure_runs_on_this_platform` |
| YB3, YB10 | med | `po.stream.embargo` with a float delay on an integer clock, the docstring's own `delay=5.0`, died in the merge of the two copies (`SchemaError`) | a whole delay is added in the clock's dtype; a fractional one, and a frame already holding `role + "_weight"`, are refused by name (YB10) | `test_label_delay.py::TestRefusals::test_a_whole_delay_keeps_an_integer_clocks_dtype`, `::…_cannot_hold_is_refused_by_name`, `::test_embargo_names_the_weight_column_it_would_add` |
| YB4 | med | `po.eval` dropped null targets but not the NaN ones the bank predicts: `r2`, `ic`, `mse` NaN, `hit_rate` quietly lowered (0.889 against 0.914) | rows whose target or prediction the bank reads as missing (null, NaN, infinite, beyond 1e100) are dropped, as `eval.seqtest` already did | `test_eval.py::test_what_the_bank_reads_as_missing_is_missing_here_too` |
| YA2 | med | a broken bank's `save` raised a bare `OSError` where `save_bytes` and pickle raise `ValueError`, and `to_json` exported the state `save` refuses | `save` maps the refusal (`io::Error::other`, a kind std never produces, by its docs) to `ValueError` and leaves the file alone; `to_json`/`save_json` refuse it | `test_window_budget.py::test_a_broken_bank_names_itself_before_a_bad_column` |
| PB7 | med | `lasso(max_iter=0)` passed the builder and `validate`: no descent, every solve a failure, output that reads like a fit | the builder refuses `max_iter < 1` (the Rust side is worker 2's) | `test_error_messages.py::test_a_bad_value_names_the_parameter[max_iter0/1]` |
| YB1 | med | a `Date`/`Time` literal was read as its bare integer: a cast to String compared false silently, and a date less it was refused | the literal is carried as a cast to its dtype, read on both sides | `test_windows.py::test_a_date_literal_keeps_its_dtype`, `formula.rs::a_date_or_time_literal_is_its_own_dtype` |
| YB2 | med | a cast to a parametrized dtype crashed with `TypeError: unhashable type: 'dict'` before the refusal | refused by name (`a cast to Datetime is not read`) | `test_windows.py::test_a_cast_to_a_parametrized_dtype_is_refused_by_name` |
| YB5 | med | `partial` was accepted without a `window_size`, and did nothing | refused, saying a window must exist to be cut short | `test_windows.py::test_partial_needs_a_window_size` |
| YA1 | med | `features=[]` raised `IndexError` in the eight builders of models with no target | `features must be non-empty`, by name | `test_error_messages.py::test_an_empty_features_list_is_named_by_a_model_with_no_target` (8) |
| TA2 | med | drift fired in none of the four legs, so the reset leg and the river comparison tested nothing | a level shift makes it fire; the reset leg's `weight_sum` drops to 0 after each flag; river's PageHinkley agrees flag for flag at `half_life=inf`, where its plain mean is ours | `test_second_opinion.py::TestAZeroWeightRowIsNotSeen` |
| TA3, TB2 | med | an HTTP 404/403/500 was an `OSError`, so "offline" and a skip; `test_windows` caught `OSError` where offline raised `RuntimeError` (TB2) | a 4xx raises at once; 5xx, 429 and the network retry, then raise `data.Offline`, the one thing a test skips on (also `test_released_state`) | `test_data_and_reference.py::test_only_the_network_is_offline` |
| TB5 | med | `N_EFF_MODELS` ("every model that emits weight_sum") missed eight kinds; `CLOCK_MODELS` missed marginal | both built from `MINIMAL` and the Rust clock-field table, exemptions checked against the builders' refusals; corrchange, deco, hmm, kmeans and micro hold the exact recursion; the marginal leg found defect 2 (`specs` round trip) | `test_properties_temporal.py` |
| TB7 | med | the ew_cov lag oracle was the recursion written out | the lagged co-moment equals the unrolled sum over the raw rows (AR(1), zero weights) to 1e-9 | `test_ew_cov.py` |
| TC4 | med | the docstring-block test claimed a NameError on an unshown name while handing every block 16 names; `skip_learned`'s example read `rerun`, built nowhere | the example builds what it reads; a rule test: a docstring block reads only names its docstring builds and the Example-data frames (29 docstrings fixed, defect 3) | `test_production_hardening.py` |
| TC5 | med | README blocks ran on fixture objects that differ from the README's own (`grid`'s `min_weight`, `now`) | blocks run on names the README's own latest defining block builds (83 substitutions); the fixture's copies pinned | `test_production_hardening.py` |
| TA4 | med | every simulator block was state 0, and `continue` skipped ρ = 0.7 silently | `durations=[4, 4]`; each state must occur | `test_sim.py` |
| TB4 | med | corrchange's break sat on a span boundary, its one flag a false positive, and the restart check behind a false condition | the break mid-span at row 150; `since_flag` 200 then 100 | `test_corrchange.py` |
| TB6 | med | two kalman tests asserted only a finite prediction | the outputs differ | `test_kalman.py` |
| TC2 | med | `test_ew_variance_tracks_river` never called the library | `ew_cov(stats=["var"])` against river's EWVar, 1.5e-14 in the limit | `test_river.py` |
| TC3 | med | a value compared with itself; a module-level rng made data depend on order | captured before, compared after; a seed per test | `test_ffi_memory.py` |
| CC2 | med | ftrl without a rate guard in the non-forgetting arm: one subnormal gradient at `beta = l2 = 0` made a coefficient infinite and every later row null | no evidence and no prior is no fit (0), as under a half-life | `ftrl.rs::a_rate_of_zero_is_no_fit` (both decays), `::a_row_too_light_to_square_leaves_the_rows_after_it_learning` |
| CC3 | med | an overflow-skipped ftrl row still aged its target's sums and moved the scale: the next prediction moved 2.2% | `Ftrl::teach` returns before the owed decay, `W*`, the scale and the target's weight move | `ftrl.rs::a_row_that_teaches_nothing_leaves_the_fit` (its overflow twin) |
| CB2 | med | the window's target moments lacked the feature side's floor: a target held at 1e8 exported `target_vars = 2.5e-9` | the 64ε floor, the means' low parts kept in snapshots; the held-run rule is not built (the targets keep no runs: new state and a schema bump) | `gaps.rs::a_target_held_over_the_window_has_no_spread_there` |
| CE2 | med | `marginal`'s bin `m2` overflowed for a target spread above ~1e79 once the scale neared its fold | the scale folds at 1e-50; ordinary scales unchanged (2.7e-13 against 2.8e-13) | `margbins.rs::a_target_spread_at_the_bound_survives_the_folds` |
| CE3 | med | a window of rows each ≤ 1e-12 of the decayed weight but together above it reported the last heavy row's value held (var = cov = 0) | every row with a weight counts for the runs, in `marginal` and `EwCov`; the window alone says what is nothing | `marginal::a_window_of_light_rows_reads_their_moments`, `ewcov::…` |
| CE4 | med | no ceiling on rcov's `bandwidth`, `max_bandwidth`, `preavg_rows`, `theta`, `jitter` or strides: "capacity overflow", or an abort at `theta = 1e15` | at most `block_rows` where given and 2^20 in any case; `theta`'s window read as a double | `rcov.rs::a_bad_configuration_is_refused_by_name` |
| CA1 | med | `ewridge`'s restore checked neither `beta`, `ready` nor the twin: `beta = []` loaded and predict panicked | refused at load | `ewridge::a_state_whose_fit_or_readiness_is_the_wrong_shape_is_refused` |
| PB3 | med | `half_life=[]` passed the builder and validate: zero instances, an empty struct | refused | `test_an_empty_grid_is_refused_by_name`, `an_empty_half_life_grid_is_refused` |
| CD1 | med | `boundary_gamma` up to 0.5 admitted; the solve's work ∝ 1/(½ − γ): 32 s at 0.499, an hour at 0.49999 | at most 0.49, refused with the reason | `corrchange.rs::a_bad_sequential_configuration_is_refused_by_name`, `test_corrchange.py::test_the_boundary_exponent_is_at_most_0_49` |
| PA5 | med | `RefreshTime::load_bytes` checked nothing: a narrow grid panicked | refused; grid ticks saturate | `a_state_whose_grids_do_not_fit_its_series_is_refused` |
| PA6 | med | `time_refresh` was the ns integer as Float64 (…2001 → …2048) | the completing tick's clock in the clock's dtype, exact | `test_time_refresh_is_the_completing_ticks_clock_in_its_own_dtype` |
| PA7 | med | `pca_prev` grew per key forever under `group_close="session"`; `drop_groups` left it | dropped with the group | `drop_groups_drops_the_groups_pca_continuity` |
| PA8 | med | `POLARS_ONLINE_MAX_THREADS=100000` panicked where `ComputeError` is promised | the documented error | `a_pool_the_system_will_not_start_names_the_variable` |
| PC3 | med | `Duration`'s equality included its text: a state saved with `"5s"` refused `"5000ms"` | equal by length; a bank loaded under another spelling restores every stream under the caller's spec and renames its PCA continuity (PC3b) | `a_state_resumes_another_spelling_of_one_length`, `bank.rs::a_bank_loaded_under_another_spelling_keeps_its_pca_continuity` |
| PA4b | med | found by worker 2: a zoned `Datetime` group or session column was refused by the bank with polars' inner message (no `timezones` feature) | keyed by its instant as UTC wall time (`arrow::key_text`) in the bank, `with_windows`, `refresh_time` and `skip_learned` | `bank.rs::a_zoned_datetime_group_is_keyed_by_its_instant`, `test_error_messages.py::…`, `test_windows.py::…` |
| CF1 | med | the ew_cov contract probe asserted rows 0-1 only | the decayed row and the gap asserted | `model_contract.rs` |
| MUT | med | `target_n_eff_into` in ftrl, pa and sgd had no Rust test (6 survivors); `clock.rs`'s `<`→`<=` survived | tests kill them; the clock mutant is not an equivalent (zero recursed until the stack overflowed) | `::the_trait_reports_each_targets_own_weight` (3), `clock.rs::no_time_is_positive_zero` |
| TC11 | low | `test_soak.py` ran in no workflow, and its resume test had failed since task 120: its tail restarted the clock at 0 | the weekly leak-check job runs `-m soak`; the tail continues the clock | `test_ci_cost_policy.py::TestEveryDeselectedMarkerRunsSomewhere`; the soak itself |
| CI4 | low | `rust-version = "1.85"` while the lock needs 1.95 (`sysinfo` 0.39.6), so an sdist build on 1.85-1.94 failed inside a dependency | 1.95 | `test_ci_cost_policy.py::TestTheDeclaredRustVersionBuildsTheLock`, from `cargo metadata --locked` |
| CI8 | low | ci.yml's concurrency group lacked the event, so the Monday run and a push to main could cancel each other, and only a push publishes the reference | the group carries `github.event_name`. The red docs job on `a7a8f3c`'s run was GitHub failing to acquire a runner (zero steps), not a superseded deploy | `test_ci_cost_policy.py::test_a_scheduled_run_cannot_cancel_a_push` |
| SC5 | low | `coverage.sh` built `online-py`, so every test binary linked libpython | `--exclude online-py` | `test_ci_cost_policy.py::test_the_coverage_script_leaves_it_out_too` |
| SC6 | low | three scripts said scikit-learn was not installed, or to add it with `--with`, where the dev group has it | they say `uv sync` | `test_dependency_policy.py::test_no_script_tells_the_reader_to_install_a_dev_library` |
| SC7 | low | `doc_review.py`'s bold-lead split missed a lead holding a code span with a star | a code span is one token of the lead | `test_doc_review.py::test_a_bold_lead_holding_a_code_span_with_a_star_is_split_too` |
| TC7 | low | the hygiene test missed the suite's own state formats and any binary file | `.state`, `.msgpack`, `.mpk` and `.bin` are data, and no tracked file may be binary (a NUL in its first 8,000 bytes, git's test) | `test_repo_hygiene.py::test_no_binary_file_is_tracked`, `::test_the_checks_know_the_suites_own_formats` |
| TC8 | low | the release test pinned `publish`'s literal `needs`, which passed with a new job left out | every job but the three after the upload is upstream of `publish`, transitively | `test_release_workflow.py::test_publishing_waits_for_every_job_and_the_tag_waits_for_publishing` |
| YA4 | low | `targets` was an invariant `list`: mypy rejected six documented forms | a covariant `Sequence`; a bare `str` now type-checks and is refused at run time by name (a trade-off: no annotation excludes `str` from `Sequence[str]`) | `test_kwargs_typing.py::test_every_documented_target_form_type_checks` (mypy over the forms) |
| YA5 | low | `targets=` on `deco`, `bocpd`, `corrchange`, `hmm`, `rcov` raised `_common() got multiple values` | refused as `ew_cov` refuses it | `test_error_messages.py::test_a_model_with_no_target_refuses_targets_by_name` (5) |
| YA6 | low | a word such as `"inf"` given to `rolling_metrics`/`embargo` on a temporal clock lost the `who: key` prefix | the parse error carries the helper's and the parameter's name | `test_temporal_clock.py::…::test_what_is_not_a_duration_is_refused_by_name_by_a_helper` |
| YA7 | low | `half_life="inf"` did not round-trip: `ModelBank([s]).specs[0] != s` | an infinity word is kept as `float("inf")` | `test_temporal_clock.py::TestADurationSurvives::test_save_load_and_the_specs_it_was_built_from` |
| YA8 | low | `sgd` accepted `huber_delta`, `quantile`, `eps` and `power` beside a loss or schedule that does not read them | the builder refuses each by name; Rust `validate` too, for a raw dict or TOML (YA8b, worker 2) | `test_sgd.py::TestPlumbing::test_a_parameter_of_another_loss_or_schedule_is_refused_by_name` (8) |
| YA9 | low | `format_of_path`'s doc comment sat on `RefreshTime` | moved | `test_kwargs_typing.py::test_each_native_item_carries_its_own_docstring` |
| YA10 | low | spec.py said every field is computed before the row (`coef` and `support_coef` are after it); a stale error example; 19 builders' Output rubrics omitted `settled_frac`/`withheld_reason`; "largest first" in duration text, which Polars does not require | the docs say each; `coef` on row t is the fit row t+1 is predicted with (measured: exact for ewridge and rls under the default cadence, 8.9e-16 for sgd's lane order) | `test_model_registry.py::test_every_builders_output_rubric_names_the_fields_its_plainest_spec_writes`, `::test_the_field_grammar_names_every_field_a_grid_writes` |
| YA11 | low | an expression under any key of a raw dict was read as a formula | a duration expression or `timedelta` under a clock parameter is converted as the builders convert it; an expression is a formula under `targets` alone, refused by name elsewhere | `test_formula_targets.py::test_a_raw_dict_takes_a_duration_under_a_clock_parameter` |
| YB6 | low | `fit`/`fit_predict_batches` over this package's own plan form on an empty input warned `ConsumedSourceWarning` | the source-run counter is read before the plan is collected | `test_consumed_source.py` (the chain test) |
| YB7 | low | a `save_state` that is a directory failed after the whole stream | refused when the plan is built, `IsADirectoryError` on every OS | `test_frame.py::test_a_save_state_that_is_a_directory_is_refused_before_the_stream` |
| YB8 | low | `po.gram.merge([g])` dropped `group`, the instance and the lags | one part is returned as it is | `test_gram_module.py::TestMerge::test_merging_one_gram_returns_it_unchanged` |
| YB9 | low | `metrics` gave `r2 = -inf` and `ic = NaN` at zero variance where `from_sums` gives null | null where undefined, as `from_sums` | `test_eval_sums.py::…::test_metrics_is_null_where_from_sums_is` (4) |
| YB11 | low | `rolling_metrics(window_size=inf)` gave one bucket with `window_start = NaN` | refused, pointing at `po.eval.metrics` | `test_eval.py::test_rolling_metrics_refuses_a_window_that_is_not_finite` |
| YB14 | low | `_frame.py` said a join above a bank is never reported; on 1.44.2 it is | the doc says what is measured | `test_order_hazards.py::test_a_join_above_a_bank_is_reported_where_polars_serializes_the_plan` |
| CI5 | low | `po.ReadinessWarning` and `po.FormulaTarget` were exported and on no reference page | added | `test_api_links.py::test_every_name_the_package_exports_is_in_the_reference` |
| CI6 | low | llms.txt, the bug template, the Pathway example and two README sentences said `coef` is the one field that follows the chunking (`support_coef` too) | each names both | `test_llms_txt.py::test_it_names_every_field_whose_schedule_follows_the_chunking` (the README sentences included) |
| CI7 | low | llms.txt showed a clocked spec without `gap_cap`, which the bank refuses | the rule is stated | `test_llms_txt.py::test_it_says_a_clock_needs_a_gap_cap` |
| TC6 | low | sgd's `clip_gradient` bound in none of 18 replay cases; a test restated `sgd.rs` | a case where the clip binds on more than 1,000 coordinates under every schedule, held to a reference from the per-coordinate rule | `test_oracles_gradient_paths.py` |
| TA5 | low | the filterpy oracle's process noise was linear in d, passing at d ∈ {0, 1} | d², on a clock with steps 1, 4 and 0.5 (149 of 300 rows differed before) | `test_second_opinion.py` |
| TA6 | low | the CLI resume test read files the previous test wrote | one module fixture runs the CLI for both | `test_examples.py` |
| TA7 | low | a refusal test skipped when the scorers finished first: a race, not a platform guard | five rounds, then `pytest.fail` | `test_predict.py` |
| TA8 | low | "no embargo, the same rows learned" asserted `rows_seen == height`, true of any fit | `rows_learned` (7) and the whole Gram, bit for bit, against `embargo=W`; `W + 2.5` gives 6, so it can fail | `test_formula_targets.py` |
| TA9 | low | the psd form's test asserted only a smaller `rcov_n` | held to its definition (k_n = 37) to 1e-9 | `test_rcov.py` |
| TA10 | low | a skipped row's test checked `weight_sum` alone | every field null | `test_properties.py` |
| TA11 | low | two tests slept 0.2 s before asserting a source closed | wait for it (gc plus a polars call, since pyo3 releases the source at polars' next call), bounded by the engine's measured read-ahead (6 streaming, 4-7 in memory; bound 8) | `test_frame.py` |
| TB8 | low | the conformal module docstring left out `w/w̄` | fixed | -- |
| TB9 | low | `… or dropped.height < 11` passed for any drop; two lines asserted a constant | the precise claim (height 4, first t 33.0); the constant gone | `test_windows.py` |
| TB10 | low | `TestClockColumnTypes` said a temporal clock is refused | re-documented to what it tests | `test_edge_cases.py` |
| TB11 | low | a negative-weight test duplicated another for 7 of 10 models | deleted | -- |
| TB12 | low | deco's embargo test asserted only that it was accepted | the embargoed run differs from the plain one and equals the delay applied by hand | `test_deco.py` |
| TB13 | low | the null-target and null-weight tests were met by the stream layer whatever the model learned | per model (11): after a null target every later row equals the weight-0 stream, `weight_sum` up by the row's decayed weight; a null weight equals the row deleted | `test_semantics_all_models.py` |
| TC9 | low | four assertions that could not fail | one deleted, three compare with something real | -- |
| TC10 | low | "four threads never overlapped" could fail on a fast machine | retry rounds ending in `pytest.fail` | `test_error_messages.py` |
| CE6, CE6b | low | windowed Kish sizes cancel: 2% off at 1e-8, `inf` at 1e-11 in `marginal`; 10.75 against 11 at 1e-6 in `ew_cov`, the regressions and the target moments (CE6b, found by worker 1) | null where the window's Kish sum is within 64ε of the history's | `kish_size_inside_a_window_is_the_rows_inside_or_nothing` (marginal, ewcov), `gaps.rs::kish_sizes_inside_a_window_are_the_rows_inside_or_nothing` |
| CC5 | low | a zero-weight quantile row outside the band moved σ² by 1 ulp | it leaves before the σ² update | `robust.rs::a_zero_weight_row_keeps_the_residual_variance_to_the_bit` |
| CB3 | low | `pca_every`'s docs said learned rows; the code counts every stepped row | the docs say every row, weight-zero rows included: C7's cadence rule (the code is the released behaviour) | `ewcov.rs::a_zero_weight_row_counts_toward_the_pca_cadence` |
| CE5 | low | `preavg_rows=2` without `psd` gave the zero matrix | refused; `theta`'s window at least 3 without `psd` | `each_stride_is_refused_at_zero_and_a_window_of_two_is_allowed` |
| CA5, CA6 | low | a failed first solve left zeros and a share of 1.0; a combo skipped for no weight kept stale shares | NaN `coef` and a null prediction; no shares | `ewridge::a_first_solve_that_fails_leaves_no_fit`, `::a_combo_skipped_for_no_weight_reports_no_shares` |
| CB6 | low | ew_lagcov's only definitional test was the recursion | an oracle from the unrolled definition | `ewlagcov.rs::the_recursion_is_its_unrolled_definition` |
| CF5 | low | no `Decay::check` in the Rust API | every decay-carrying `validate` checks first | `model_contract.rs::every_model_refuses_a_decay_it_cannot_run_on` (17), `clock.rs::a_decay_is_checked_at_its_edges` |
| CF6 | low | `holds()` could pass vacuously | the slope is an `Option`; the windowed ridge test tightened | the held-value tests (`0 of 3000` rows predicted fails) |
| CF7 | low | EXTENDING's hook table missed `set_solve_share` and the readiness hooks | rows added | `exactly_the_scheduled_solvers_report_a_solve_share` |
| PB4, PB5 | low | wrong-length `coef_min`/`coef_max` dropped when trivial; the formula-target name check was the builder's alone | refused; `validate_spec`, `ModelBank` and the CLI dry run refuse it | `test_the_rust_side_names_the_offence`, the builder test extended |
| YA8b | low | found by worker 3: Rust `validate` accepted the sgd parameters the builder refuses (YA8), from a dict or TOML | refused in the builder's words | `sgd_tests::…`, `test_a_raw_spec_with_a_parameter_nothing_reads_is_refused` |
| PC5 | low | the brute force shared the forward closing rule with the core | it decides completeness from the stretch, on half-unit grids reaching the edge exactly | the brute force (`seed 5: drop flags differ` on the old core) |
| CD2, CD3, CD4, CD5, CE7, CE8, CA4, CB4, CB5, CC6, CC7, CF2, CF3, CF4, PA9, PB8, PB9, PB10 | low | a message naming no parameter (`truncate`); a vacuous hmm test; bocpd's row-one sentence (`min_weight` gates it); `alpha_adjust` doing nothing under the window kind (now refused); doc drift in kmeans, rcov, lasso, ewcov, window, sgd, conformal, clock, model, bank, spec | each fixed, each message or claim pinned where a test can read it | `bocpd::a_bad_configuration_is_refused_by_name`, `hmm::a_uniform_chain_is_the_classifier`, `test_a_parameter_of_another_kind_is_refused`, `test_a_deco_block_refusal_reads_as_one_sentence`, `test_a_zero_delay_is_told_to_leave_it_out` |
| YB12 | low | `online --dry-run` said "config OK" for an input that is not there | the dry run opens the input and names a missing one | `a_dry_run_names_an_input_that_is_not_there` |
| YB13 | low | `--no-output --predict` was refused with a message pointing at fixes that cannot apply | refused, naming `--predict` | `no_output_with_predict_is_refused_naming_predict` |

**Decisions left for the user.** Each has its evidence in the review's
scratchpad and is unchanged in the code:

- **CC1**: `drift_threshold` is in σ times clock units; on a temporal clock
  `d_clock` is seconds, so the default 20 fires on noise at a row a minute
  (237 of 3,000 rows) and a row a day (435), and under
  `drift_action="reset"` the model restarts on noise. Which unit?
  *Decided 2026-10-06, the user's word ("make drift_threshold a clock
  parameter as you recommend"); built as task 168.* The seconds were the
  scale every temporal clock is read on, not a unit anyone chose, so the
  threshold became a clock parameter.
- **CE1**: `rcov`'s pre-averaged estimate under `psd=False` omits CKP's
  footnote rescaling `1/(1 − ψ₁/(θ²ψ₂)/(2n))`: 0.842 of the integrated
  variance at `k_n = 6`, 0.941 at 10, 0.985 at the default. Applying it
  moves the golden. *Decided 2026-10-06, the user's word ("follow your
  recommendations for all remaining review decisions"): applied, task 171.*
- **CC4**: `kalman`'s observation and process noise are the literal 1.0 in
  target units until the first residual, so the warm-up's gains depend on
  the target's units (a 100% difference between scales of 1e-6 and 1e6).
  *Decided 2026-10-06, the user's word ("follow your recommendations for
  all remaining review decisions", then "p0 relative"): the noise from the
  first innovation and the prior as `p0` times it, task 172.*
- **CB1**: a model's window edge is decided on its own summed `d_clock`, so a
  row exactly `window_size` old is dropped on some rows and kept on others;
  the window operators were fixed in task 159 (W3), the models' windows
  were not. Exact edges need the raw clocks in the models. *Decided
  2026-10-06, the user's word ("Go with option 1"): stamps on the decayed
  clock held exactly, task 175.*
- **PC1**: a silent group on a row-count clock with a forward window holds
  every later row of every group until the end (500,001 rows held, 117 MiB
  over 2M rows), where the docs promise memory of one window. Refuse it, or
  bound it? *Decided 2026-10-06, the user's word ("follow your recommendations for all remaining review decisions"): refused, task 173.*
- **CA3**: `lasso`'s selection starts counting errors at the model's own
  first prediction, gated by the list's smallest `min_weight`, so
  `penalty_selected` differs on 84 of 240 rows between `[0, 60]` and
  `[60, 60]`. When should the selection start counting? *Decided 2026-10-06, the user's word ("follow your recommendations for all remaining review decisions"): from each
  target's own `min_weight`, task 174.*
- **CI2**: run 35508619563 (v0.8.1, 2026-09-20) still waits at "publish to
  PyPI", approvable with one click. Cancelling it is a write. *Cancelled
  2026-10-06 on the user's word ("Yes to the three items"), with the push
  of task 167 and the merges of Dependabot's #3, #4 and #5.*
- **PC2, a reset**: a cut now keeps a forward window whose far edge is the
  last row before it whole; a reset still discards it, as its docs say ("a
  reset discards them"). *Decided 2026-10-06, the user's word ("follow your recommendations for all remaining review decisions"): kept whole on a reset too, task 173.*
- **TC1b's cost**: the full-leverage bound halves `quantile`'s throughput on
  the default solve schedule (2.6M against 5.0M rows a second, 10
  features; 8% solving every row). Accept it, or skip the leverage where
  the bound provably cannot bind (bit-identical and resume-safe, but it
  needs a floating-point margin argument and a review)? *2026-10-06: the
  user asked instead for the Cholesky update, in one place; built as task
  170 (`solve.rs`), with bit-identical savings that take back about a
  sixth; its use in `robust` is parked, below.*

**Left as they are, with the reason.** CI9: CLAUDE.md says the Polars pins
in `pyproject.toml` and `Cargo.toml` "must match", where they are two
numbering schemes kept in step; CLAUDE.md is the user's to edit. YA4's
`Sequence` typing takes a bare `str` as a `Sequence[str]`; no annotation
excludes it, and the builders refuse it by name at run time. CB2's held-run
rule for targets needs per-target runs in the state, a schema bump; the
64ε floor covers the case found. `bank.rs` stood at 249,165 bytes against
the 250 KB cap; its four inline test modules moved to files of their own
under `src/bank/`, as `damaged_file_tests` had (2026-10-06, on the user's
word), leaving 242,524 bytes of production code.

**What held, measured.** No rule-2 failure under the corrected property
(TA1); every other kind round-trips its spec dict (only marginal did not);
corrchange, deco, hmm, kmeans and micro hold the exact `weight_sum`
recursion (TB5); kmeans and micro against their definitions and
scikit-learn (ARI 0.99995); the standardized kalman against filterpy
(1e-9); the quantile fit stationary for the loss it smooths (1e-15).

## 17. Parameters counted in rows, where a clock would fit (2026-10-06)

The user, after task 161: "Find more example where a parameter references
only rows when it should also be a clock", then "Do both these and document
the rest for review". Every integer parameter of every builder was read
against what it counts (`_spec.py`, `spec.rs`, the models). Two are built on
the regressions' pattern -- clock units in the parameter, a row cap beside
it, whichever comes first, the old default kept: `micro`'s pruning (task 163)
and the window snapshots (task 162). The rest are listed here for the
user's review, each with what it counts, what a clock form would mean, and a
recommendation. Nothing below is built.

**Cadences that could take the clock, as `solve_every` does.**

| Parameter | Counts | A clock form | Recommendation |
|---|---|---|---|
| `coef_every` (every model with coefficients) | each group's accepted rows, rows of weight zero and rows with a null target included (its doc said learned rows until task 164) | "write `coef` every five minutes": an output cadence, so no number moves | build it, on the pattern; one decision first: `coef_every = 0` means "only each group's last row in a chunk", where `0` in the clock parameters means "every row", so the clock form needs its own spelling of that default |
| `kmeans`' `update_every`, `split_merge_every` | learned rows | batch the centres' updates, and look for splits and merges, every so long | leave as rows unless a stream needs it: both are closer to batching knobs than to a span of time |

**Dynamics stated per row, a model change rather than a cadence.**

| Parameter | Today | Clock-aware | Recommendation |
|---|---|---|---|
| `bocpd`'s `hazard` | "the per-row chance of a break", `1/hazard` | the chance grows with the time since the last row: `H(d) = 1 - exp(-d / tau)`, `tau` the expected time between breaks, in clock units | worth building as an option (a duration `hazard` on a temporal clock): on irregular rows the per-row hazard says breaks follow the row rate. `hazard_col` can carry such a per-row value today, computed from the clock -- from the step *after* its row, since a row's hazard is the chance of a break after it (corrected 2026-10-06: task 179's worker found the step into the row dates every break a row late) |
| `hmm`'s transition matrix | a per-row Markov chain | a continuous-time chain, `P(d) = exp(Q d)` | record only: estimating `Q` from soft counts over irregular gaps is a larger change, needed only for irregular data whose regimes switch on the clock |
| `deco`'s `alpha`, `beta` | per-row dynamics | dynamics per unit of time | record only, as for `hmm` |
| `sgd`, `pa`, `ftrl` coefficients | learn per row; `n_eff` decays on the clock (hard rule 8) | -- | by design, documented |

**Rows by definition, to keep.**

- `lags`, `cross_lags` (`ew_cov`, `marginal`) and `resid_autocorr_lag`: a lag
  of `l` rows is the estimator. A lag in time on irregular data is another
  estimator. The standard route is to put the series on a common clock first
  -- a fixed grid with the last value, or refresh times
  (`po.stream.refresh_time`) -- after which a lag in rows is a lag in time.
  For asynchronous ticks the established estimators are Hayashi and Yoshida's
  (2005) for the covariance and its time-shifted form for the lead-lag
  (Hoffmann, Rosenbaum and Yoshida 2013), which avoid the bias previous-tick
  sampling adds (the Epps effect). Recommendation: document the grid route
  beside `lags`; an HRY lead-lag model would be a new model, as `rcov` is, if
  lead-lag on ticks becomes a use.
- `corrchange`'s `span_rows`, `monitor_rows`, `permute_every`, `perm_block`:
  the tests' statistics are defined over row counts (Wied and Galeano; a
  permutation in blocks of rows).
- `rcov`'s `block_rows`, `preavg_rows`, `bandwidth`, `max_bandwidth`,
  `jitter`, `noise_stride`, `iv_stride`: tick-count estimators (BNHLS, CKP);
  the block's edges already come from `group_close`.
- `bocpd`'s `max_run`: a run length is a count of observations.
- `kmeans`' and `hmm`'s `warm_rows`, `marginal`'s `bin_warm_rows`: sample
  sizes for seeding and for quantile edges.
- Mechanical: `gram_block_rows`, `shards`, `bins`, `max_clusters`,
  `max_iter`, `n_perm`, `k`, `pca`, the seeds.

**Decisions for the user.**

1. **Two rules for counting a cadence's rows.** `window_every`'s row cap,
   `max_rows_between_pca` and `coef_every` count every row, rows of weight
   zero included (review C7); `kmeans`' `update_every` and
   `split_merge_every` and `micro`'s `max_rows_between_prunes` count learned
   rows only. A test docstring of task 160 (CB3) says "the library has one
   cadence rule". Narrow that sentence, or move `kmeans` and `micro` to the
   shared rule, which moves their numbers on a stream with rows of weight
   zero.
2. **`micro`'s default cadence.** Every 100 learned rows, kept by task 163;
   DenStream's is `Tp = ceil(h log2(beta_mu / (beta_mu - 1)))` clock units.
   Keep it, or default to `Tp`, which moves every default `micro`'s numbers.
3. **`coef_every` on the clock**, and the spelling of its "last row of a
   chunk" default (above).
4. **`bocpd`'s hazard in clock units** (above).

*Decided 2026-10-06, the user's word ("Follow your reco on all"):* (1) the
sentence narrowed, in `ewcov.rs`'s cadence test: the cadences that refresh a
readout count every row, the ones that act on what new rows taught count
learned rows; (2) `micro` keeps every 100 learned rows; (3) built, task 178
(unset is the old default, `0` every row, `max_rows_between_coefs` the row
cap); (4) built, task 179, as a duration `hazard` applied before the row it
leads into (a row's hazard is the chance of a break after it).

## 18. Whole-project review before 1.0 (2026-10-06): everything, with a 1.0 lens

The user: "Deep code review of everything in this project. We are getting
close to version 1.0; find everything we should do before then." Nineteen
read-only reviewers at `8e28c1f`, one slice each (six over `online-core`,
four over the polars layer, one over the three surfaces, two over the Python
package, two over the tests, one over CI and release, two over the
documents, one cross-cutting over names, units, promises and policy), 257
findings. Two lenses: **defect**, as §15 and §16; and **1.0**, anything cheap
to change now and breaking after the promise (a name unlike Polars' or the
library's own, a default with units in it, an undocumented semantic, a file
format, a promise no test pins). Severity as §16; a 1.0 finding takes the
severity of what it would cost after 1.0. Every reproduction was re-run by a
verification agent: all 13 Rust tests confirmed (the load succeeds or the
next row panics, as stated), all 27 Python scripts reproduce. The per-slice
reports, reproductions and the consolidated list are in the session's
scratchpad (`review4/`); this section keeps what must outlive it.

The user then: "Do everything that does not need me." Tasks 183-193 carry
the findings whose fix the code, the docs, an existing rule or a precedent
settles; the decisions below wait for the user.

**High defects** (each in a task): `huber`/`quantile` and `ewridge` predict a
target with no rows at a solve as exactly 0.0 until the next scheduled solve
(CC1, task 186); a windowed `ew_class`'s `coef` is the whole history's
(CE1, 186); `po.eval` scores a relative target against its raw column (YB1,
188). Two clusters: loaders that trust the bytes (twelve findings, 185) and
spec values that size an allocation with no ceiling (CD10, CF2, CE9; 193).

**Decisions for the user**, with the recommendation given:

| # | Decision | Recommendation |
|---|---|---|
| D1 | Hard rule 5 returns at 1.0 with no loader and no fixture; `MIN_SCHEMA_VERSION = 14` is false (layouts since fail in serde before the gate); the released-state test asserts refusal only (CF1 AP10 CI3 DB1 TB3 CC8 DA3) | 1.0.0's schema is the floor; from there every layout change ships a loader and a frozen fixture per `ModelState` variant (embedded bytes), checked three ways (loads, continues to the bit, re-saves byte-identically); raise `MIN_SCHEMA_VERSION` to the shipped schema; build the harness now with HEAD's state; freeze the variant names; delete the dead `#[serde(default)]` repairs |
| D2 | No deprecation policy; a renamed name is refused outright (AP9 CI4 DA16) | warn-and-forward with a `DeprecationWarning` subclass, removed at the next major; `RENAMED` becomes the forwarding table at 1.0 |
| D3 | What a numbers-moving fix is after 1.0 (DA2) | a fix whose old numbers were wrong against the stated definition is a minor, declared with `compare_release`'s difference; a changed default or semantic is a major |
| D4 | The Polars floor 1.34.0 has not run since the window operators; the canary tests the ceiling (AP7 CI2 AP11) | a blocking floor leg in `release.yml` and a monthly canary run; after 1.0, raising the floor is a minor |
| D5 | No "unstable" marker (AP12 DA4 YB11) | Polars-style label and opt-in warning on the windows state format, the formula tree's written form, `fit_predict_arrow`, `po.sim`, `po.corr`; promise the rest |
| D6 | Pins for the stable surface | *not a decision: task 190* |
| D7 | The stream-state tidy (S6) | *not a decision: task 194* |
| D8 | Task 116 leaves readiness defaults provisional (DB7) | build the parts that move no default; declare the current floors final |
| D9 | New users get polars 2.0.0; the goldens and the lock are on 1.44.2 (CI18) | move the dev pin to 2.0.x once the canary has passed on it |
| D10 | The changed-lines mutation job cannot finish a normal push (CI1) | shard it as the weekly job is |
| D11 | `rust-version = 1.95` is compiled by no job (CI11) | a weekly `cargo check --locked` on 1.95 |
| D12 | SECURITY.md's support line after 1.0 (CI15) | the latest minor receives fixes |
| D13 | 21 of 32 `docs/` files are records (DB14) | `docs/records/` for the uncited ones; PHRASING and README-ITERATIONS filed as records |
| D14 | No payload hash; a flipped bit loads (PA14) | no hash at 1.0; the limit stated in `ModelBank.load`'s docstring |
| N1 | `po.spec.ewridge` vs `type = "ew_ridge"` vs the `ewridge:` message prefix (AP4 CF5) | tag `"ewridge"`; `huber:`/`quantile:` prefixes |
| N2 | `chunk_rows` vs Polars' `chunk_size` (AP5) | `chunk_size` on all three surfaces |
| N3 | `po.eval`'s `min_obs`, `by=` (a bare string iterates characters) (AP6 YB16) | `min_samples`, `group` |
| N4 | `rolling_metrics` is tumbling (Polars' `group_by_dynamic`) (YB5) | `window_metrics(every=)` |
| N5-N9 | `corr.shift`; `po.gram.lasso_path(lambdas=)`; `dist2`; `bank.marginal()`'s `t`; `rows_seen()` (YB10 YB20 TA7 CD13 PA11) | `absorption_shift`; `penalties=`; `dist_second`; `t_stat`; `rows_fed()` |
| N10 | `lagcorr` at the surface, `lag_corr` in the state (CB3) | `lag_corr` everywhere |
| N11 | `rls.ridge` is `ewridge`'s `ridge_scale="sum"`, not its `ridge` (CA7) | rename `delta` |
| N12, N13 | `conformal=<level>`; `increment` vs `diff` (CA9 PD8) | keep both |
| N14 | `update_every`, `split_merge_every`, `permute_every` count rows (AP15) | `*_every_rows` |
| N15, N16 | `--resume` vs `load_state`; holt's `level_half_life` beside `half_life` (SF7 AP8 TA6) | `--load-state`; drop `level_half_life` |
| N17 | the models' `window_size` keeps a row exactly `W` old; the operators and Polars drop it (CB1 PD2) | a `closed` parameter on the windowed models, Polars' `"right"` |
| N18, N19 | clock range as Float64 seconds in three frames; `rows_fed` Int64 in one frame (SF1 AP13 PA3) | the clock's own dtype; UInt64 everywhere |
| N20-N22 | rcov's closed-group block half-prefixed; the null group unreadable alone and text key order; a key column changing dtype splits groups silently (PA13 PA10 PC1 PA7) | prefix all; `group` takes `str \| None` lists and integer keys sort as numbers; the dtype recorded in the state and a change refused |
| N23-N26 | `sim`'s status; five constants outside `__all__` and the `embargo` columns; `clock`/`group`/`lam`; `--skip-learned` (YB11 YB9 AP19 PC4 PC5 AP22) | unstable label; add to `__all__` and name the columns stable; keep (task 144); add it |
| U1 | `sgd.huber_delta` 1.0, `sgd.eps` and `pa.eps` 0.1 in target units (AP3 CC4 PC2) | σ-relative, as `huber`'s |
| U2 | `sgd.standardize` False; `pa` has none (CC6) | True for sgd; add to pa, default True |
| U3 | `huber_delta` 1.5 unmeasured; the textbook constant is 1.345 (TA8) | 1.345 |
| U4, U5 | `bocpd.prior_scale` is the identity in target² units; `prior_nu = d + 2` under every emission (CE5 CE2) | data-relative from the first rows; per-emission ν |
| U6 | `rls.ridge`, `learning_rate`, `clip_gradient`, `pa.c`, ftrl's constants | keep; state the units; pinned by task 190 |
| U7 | a row cap of 0 refused for `max_rows_between_coefs`, "every row" for four others; `pca = 0`, `stats = []` accepted (PC3 CB4 YA2 PC9) | refuse 0 and the empties everywhere |
| S1 | `permute_every` redraws every `n + 1` reports (CD11) | the code matches the docs |
| S2 | `coef` is skipped when a chunk's last row of a group is skipped (PB2) | the last accepted row |
| S3 | under an embargo `predict` learns no matured label (PB3 TB6) | document and pin it |
| S4-S6 | `sgd` logistic labels outside {0, 1}; a Poisson `hit_rate` reads 1.0; a prediction of exactly 0 a hit in the bank and a miss in `po.eval` (CC9 CC5 YB8) | ftrl's clamp and `strict_binary`; null; exclude `pred == 0` on both sides |
| S9b | a target's own first solve when its weight first reaches its `min_weight` (CC1's second half) | build it |

*Settled without the user, as following a rule or the docs (in tasks
186-193):* S7 (`shrink`'s identity target, the paper's), S8 (CE1, the
docstring's), S9a (NaN for an unsolved target, CA5's rule), S10 (a
parameter a mode ignores is refused, YA8's rule), S11 (every way in checks
a spec the same way), S12 (`refresh_time`'s value read by the `usable`
rule), S13 (`pins` markers), S14 (`increment` of a `Time` gives seconds, as
its docstring says), S15 and S16 (documentation of the current behaviour).

*Decided 2026-10-06, the user's word ("Your reco all"): every recommendation
in the table above, and task 183's cost accepted.* Built as tasks 194-199
once tasks 183-193 merge, since they touch the same files. D9 waits for the
weekly canary to pass on polars 2.0.0: its run of 2026-10-05 failed on the
unwrapped `ValueError`, fixed in `a7a8f3c`.

**Marked for the user's review (2026-10-07).** Two decisions taken after §18,
on the user's word, to be looked at again before 1.0: the unit of an
insensitivity band (`pa`'s `eps`, `sgd`'s under `epsilon_insensitive`) -- the
target's own EW spread rather than the residual's (task 202) -- and its
default, 0.01 (task 203). **Reviewed 2026-10-07 (the user: "Yes" to both
kept, two wordings changed).** A third rule was measured beside them, the
residual's std capped by the target's (`eps·min(σ_resid, σ_y)`), which
removes task 195's trap and shrinks with the fit: no rule wins across R²
(`$S/eps/bands.py`; at `c = 1`, excess OOS MSE over the noise variance,
0.01 of σ_y / 0.1 of σ_y / 0.3 of the cap: 0.18 / a lottery / 0.44 at R²
0.99998, 1.16 / 0.59 / 0.79 at 0.978, 0.81 / 0.68 / 0.59 at 0.80). The
lottery (review 5, G2): at 0.1 the band is 20 noise stds wide there, so
the fit freezes wherever it first lands inside it, and over 20 seeds
through the bank the excess runs from 0.58 to 278 (median 35). The replica
drew 54 for its seeds; task 203's table holds the bank's 5-seed median,
2.86. At 0.01 the same 20 seeds give 0.16 to 0.20. The tension is `pa`'s:
at `c = 1` the tube is its only damping against noise. 0.01 stays by regret
(its worst case 1.2× the noise variance, 0.1's between 0.6× and 280×
wherever its fit freezes, on the near-deterministic target a user has
before differencing it); for `sgd` 0.01 loses nowhere. Changed: "inside a good fit's errors" became "below a
good fit's errors on most targets" (at R² 0.9998 the band is two-thirds of
the noise std), and `pa`'s docstring and the README carry the trade-off
(2.2× / 1.6× / 1.3× the noise at the default / `eps=0.1` / `c=0.1`, at R²
0.978, `test_pa.py::test_the_tube_is_pas_only_damping_on_a_target_predicted_less_well`).
*Review 5 (F2, E5, A3, G4):* the four places had stated those figures for
"R² 0.97 to 0.998" with `c=0.1` as the better lever, from that one point.
At R² 0.9975 the bank gives 2.00 / 1.18 / 1.70 and at 0.990 2.12 / 1.42 /
1.46 (median of 3 seeds), so `eps=0.1` is the smaller loss above about R²
0.99 and `c=0.1` damps best where the noise is large. They now say so, at
R² 0.98 and without the range.

**Decided 2026-10-07 (the user: "Your reco all except 116").** On the open
items after the round:
- **The next release** (0.14.0) is rehearsed before it is tagged: this
  commit benchmarked against the cached 0.13.0 wheel, then the release
  workflow dispatched with publishing off, then the release, each dispatch
  on the user's word. Five workflows changed (`release.yml`, `ci.yml`,
  `polars-canary.yml`, `mutants.yml`, and `msrv.yml`, new).
- **D9**: the canary ran on demand (run 37715703707, on `9a67dd8`) rather
  than on Monday; on a pass the dev pin and the lock move to polars 2.0.0
  and the goldens are checked on it, in one gated commit, before 1.0.
- **`kalman`'s `share_p`**: fixed, task 204.
- **YB3 closed, not changed.** `embargo` over this package's own plan form
  reads its input twice, and so runs that fit twice: the warning is true.
  A `pl.scan_parquet` input does not warn.
- **Tasks 117, 86, 118 and 127 stay as they are**: 117 waits on 3.15.0 and
  its wheels; 86 and 118 are parked by the user; 127 waits on a
  measurement nothing asks for.
- **Task 116**: the user wants its changes, defaults included, made before
  1.0; its scope is decided from a summary of it (2026-10-07).

## 19. Review round 5 (2026-10-08): tasks 183-204, code and theory

Round 4 read the code at `8e28c1f`; its decisions produced 33 commits
(tasks 183-204, `8e28c1f..9b23ae6`) that no one but their authors had read.
Seven read-only reviewers read them at `9b23ae6`, each with a code lens and
a theory lens (A models, B clocks and windows, C bank/stream/spec, D state,
E Python and tests, F docs/CI/policy, G the theory of the decisions alone);
briefs, findings and probes in the session scratchpad `review5/`
(`CONSOLIDATED.md` is the table). Every finding is a probe output or a code
path; the coordinator re-ran the three high ones through the bank before
the fixes were briefed.

What held: hard rules 2, 3, 8 and 9 over 17 specs and 159 fields,
bit-identical at 1/7/600 chunks with unusable values on the chunk
boundaries; every re-pinned golden reproduced from its docstring replica;
every rename refused with a test; task 189's oracles fail on a 1e-7 nudge;
the share_p filterpy oracle fails on the old rule; the release, CI, canary,
mutants and MSRV workflows; the 1.0 policy's tables against each other.

**Fixed in task 205** (each test failing on `9b23ae6` first):

| id | sev | finding | fix |
|---|---|---|---|
| C1 | high | under an embargo the "cannot be met" notices paired the held-inclusive `settled` with a weight that had not seen the held rows: "tops out near −0.0462, withheld for good", then 374 of 400 rows predicted | both notices read the learned rows' settled fraction (trigger, ceiling, projection); the gate and the row field keep the held-inclusive one |
| C2 | high | a damaged `emit_metrics` state loaded and the next row panicked (`ewcov.rs:838`) | `SlotMetrics::has_shape()` at load |
| C3 | med | `--skip-learned` dropped a NaN-clock row in silence | `Some(Less \| Equal)`; the bank refuses the row |
| B2 | med | `embargo` wrapped at an integer clock's top: the learn copy first in the stream | widen before the add, strict cast back: polars raises |
| E1 | med | `po.eval(spec=)` gave a Poisson fit `hit_rate` 1.0 | null, as the bank |
| E2, C5, E7 | med | the snapshot pinned refused `closed` words, pre-144 clock names, and unstable names unlabelled | a two-word `ModelClosed`, `taken_words` requires acceptance, `restart_after_step_back` rendered, `# unstable` labels |
| B1, D3 | med | no fixture held an integer clock, `sgd`'s per-loss state, `ewridge`'s kept systems or `bocpd`'s warm rows | cases added, a decoding test holds each form non-empty; existing fixtures byte for byte |
| D2 | med | the from-1.0 half of the harness was unbuilt | `previous![..]`, `v<N>/` includes, the convert-to-current check, coverage proven on a temp dir |
| D1 | med | the promise's "re-save byte for byte" cannot hold for a loaded previous schema | stated as conversion to the current schema's bytes (README, RR, rule 5) |
| F1, F2, G2, A2, A5, A6, B4, E6, F5, F6, F8, D5, D6, C6, C7, A4, D7, E3, E4, F4b | low-med | wording, counts, a NaN cfg through the Rust API, `pow(lam, 1)`, `to_json`'s label, the property's draw, the floor test's mark | as the CHANGELOG says |

**Decisions, the user's (open 2026-10-08):**

| id | finding | options | reco |
|---|---|---|---|
| G1 (high) | `standardize=True` (sgd, pa since U2; kalman always) holds coefficients in standardized coordinates and reads rows and coefficients off the moving EW moments: at half-life 10 and R² 0.978 `sgd` is 190× worse than unstandardized and its `coef` reads [0.83, 3.03] for [0, 2]; at half-life 50 and R² 0.99998 pa/sgd/kalman are 33/237/213× worse; `ewridge` re-solves and is immune; every U2/203 measurement was at no decay | (a) keep the fit in the caller's units and use the scaler for the step only, mapping β (and kalman's P) through the affine change when the moments move; (b) revert U2's default, document kalman | (a) before 1.0 |
| G3 (med) | the eps regret argument compared values inside the σ_y rule only, and "the band must sit below a good fit's errors" is wrong for PA at c = 1: a band of 0.5 σ_y (3.3 noise stds) gives excess 0.07 at R² 0.978 against 1.16 at 0.01 -- the tail rows pull the fit to within band − 3σ_noise; the best band is about three noise stds, which no σ_y rule gives across R², and under no decay a residual σ keeps the start-up | a task: the band in noise units with a start-up cap, swept under decay, levels, c and mode, `eps·min(σ_resid, σ_y)` at 0.5 the candidate (worst case 0.61 vs 1.16); or keep the shipped rule with honest docs | the task, before 1.0 |
| G5 (med) | `c` is in the target's units, so "the tube is pa's only damping" holds at spread 2 only | `c` as a multiple of σ_y as `eps` is; or document spread 2 | σ_y-relative |
| A1 (med) | share_p's mean noise averages over all m targets, zeros included, so a target with no residual variance yet counts as noise 0; task 204's "number of targets" claim holds for targets on the same rows | (a) average over targets with σ² > 0 in `step` and `pred_var`, as `shared_first_noise` does; (b) keep, narrow the docstring | (a) |
| B3 (med) | an `Int128` column is "integer" to the docs and a double to the code (`po.increment` loses steps near 2^63) | refuse as a clock by name; read it in `po.increment` | both |
| C4 (med) | `summary().settled_frac` leaves the held rows' clock out; the row field adds it | (a) summary held-inclusive, `weight_sum_settled` on the learned fraction; (b) document | (a) |
| D4 (med) | the policy promises forwarding for every renamed stable name; the mechanism forwards parameters and keys | (i) narrow the promise, a forwarding stub when a function or word is first renamed; (ii) build now | (i) |
| F3 (med) | the "cannot be met" notice also fires falsely on a rising row rate (the README's kernel example: 381 of 400 predicted, weight 103 against a projected 7.7) | conditional wording, fire only after the weight stayed below the floor a further half-life past 95% settled, pin the example | as stated |
| F7, F9 (low) | U7's `stats=[]` exception recorded only in a task note; PERFORMANCE §11 cells rewritten where the rule was a pointer | annotate U7; accept the cells | as stated |
| D8 (note) | `test_released_state.py` asserts `refused == {}` per release: after 1.0.0 a new kind fails 1.0.0's leg | filter per release on release day | -- |
| 116-K | the "never met" notice for `kalman`'s per-row gate: `P` settles on `coef_half_life`'s clock and the rows spread 1.02-1.11 around the average 1.07 at `coef_half_life` 50, so a notice tested at ±5% cannot hold (task 116) | judge the best row (`√(1 + 1/(P⁻¹)₀₀/R)` with an intercept), the average row, or stay off; evidence WARMUP §7.11 | stay off until a 1.x user asks |

The coordinator's own correction, recorded here: the eps review of
2026-10-07 said "nothing else in the eps chain needs reopening"; G3's
measurements and the 0.5-band probe (`review5/verify/bands_h.py`) show the
theory the docs state is wrong for PA at `c = 1`, and the chain is open.
