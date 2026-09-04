# polars-online — design and plan

Status: design frozen 2026-08-29, no code yet. Items marked **[validate]** are defaults
chosen without data; check them in the evaluation harness (task 12) before relying on them.

## 1. Goal

Online regression models over data that does not fit in memory -- ordered event streams
(one per group, e.g. per bond) with a clock, and equally a plain table in any row order,
where decay off (`halflife=inf`) is exact least squares at O(state) (the "Any row order"
decision below, 2026-09-03) -- usable two ways with identical numerics:

1. **Python ModelBank** — chunk-fed, `fit_predict(chunk)` over `LazyFrame.collect_batches()`;
   memory is O(state), not O(data). Also as a plan: `lf.online.fit_predict(specs)` is the bank
   registered as a polars IO-plugin source, a `LazyFrame` that streams when it runs (E33);
   `df.online.fit_predict(specs)` for a frame in memory.
2. **Streaming runner** — same bank as a read → fit → write pipeline, memory O(state + chunk):
   `po.run(...)` from Python (any source py-polars can stream, parquet / ipc / csv / ndjson out),
   or the Rust `online` CLI (the same formats, TOML config, no Python) for deployment.

Both share `online-polars` and `online-core`. A third way, the **expression plugin**
(`pl.col("y").online.<model>(...)`, with `.over(group)`), was built first and is the
**in-memory** form (§6): polars calls a user expression with the whole column in either
engine, so it is the one O(data) surface. It stays, for a frame already in memory, and every
call warns with `InMemoryExpressionWarning` naming the plan — so the difference is learned at
the call site, not from a memory profile.

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
                                         // stream layer on coef_every rows / last row of chunk
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
| `group` | str \| None | one state per key (the expression API uses `.over()` instead, §6) |

Per-row decay: `λ_row = 0.5 ** (Δ / halflife)`; `n_eff` = EW count with the same decay.

### Clock semantics
- Δ = clock − prev_clock, clipped to `[0, max_dclock]` (with `on_clock_reset="zero"`) or
  Δ<0 ⇒ `max_dclock` (`"max"`, default) or state reset (`"reset_state"`).
- Session change ⇒ Δ := `session_gap` (or reset), regardless of the clock delta.
- First row of a group ⇒ Δ = 0.
- The clock is per group (the expression API gets the group's rows via `.over()`).

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
  `select_halflife`, default = model halflife) — free, since preds for all λ are computed anyway.
  Reported as it stood *before* the row's error joined the selection, i.e. the λ the row was
  scored with; that is what makes it identical between `fit_predict` and `predict` (E31),
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
  independent, so they run in parallel over (spec × group) on the bank's own pool
  (`POLARS_ONLINE_MAX_THREADS`; polars' readers and writers stay on `POLARS_MAX_THREADS`).
- Under a `clock`, chunks must be clock-ordered within each group; the bank asserts
  monotonicity (after reset handling) and errors loudly otherwise. Without one the row
  order is the clock, and with decay off the order does not reach the fit at all.
- `state()` / `save(path)` / `load(path)`: msgpack (`rmp-serde`) with header
  `{schema_version, package_version, spec}`; loading checks the spec matches.
- Python: `ModelBank(specs).fit_predict(df) -> df` (appends struct columns) and
  `predict(df) -> df`, the same columns scored against the bank as it stands with nothing
  updated — every row from the same state, the clock distance measured from the last learned
  row, session/clock reset policies honored by scoring a fresh model (ENHANCEMENTS E31); Rust
  CLI reads the same specs from TOML.

## 6. Expression plugin (`online-py`) — in-memory only, warns since 2026-09-03

`pyo3-polars` expression with `is_elementwise=False`, `returns_scalar=False`, one namespace
`online` with one function per model. Runs a single spec over the full column it receives
(so `.over(group)` gives per-group streams). Output dtype is a struct built from the spec.
Grids are allowed but produce wide structs; the bank is the recommended surface for grids.
The implementation is the bank itself: `_expr.py` packs every input into one struct (the
polars path that spreads `.over` groups across threads), and `online_run` in
`crates/online-py/src/expr.rs` unpacks it into a frame and runs `Bank::fit_predict` on it —
so expression ≡ bank by construction.

**Why it warns.** Polars hands a non-elementwise user expression its whole column: in the
in-memory engine by definition, and in the streaming engine because a plugin lowers to a
`columnar-function` node — collect the input, call once, re-emit. There is no way for a
plugin to say "call me per morsel, in order, and let me keep state" (§11a, 2026-09-02). So
`lf.with_columns(pl.col("y").online.ewridge(..)).sink_parquet(..)` measured 7.3 GB at 12M
rows where `lf.online.fit_predict([spec]).sink_parquet(..)` measures 1.35 GB
(`docs/PERFORMANCE.md` §11) — the same numbers, two memory profiles, and users read the
expression as the natural form. The first answer (task 19 as first committed) was to
take the expression out of the wheel behind a cargo feature; that left a user who wrote it with
polars' bare `AttributeError: 'Expr' object has no attribute 'online'` and no pointer, and
left the plugin's runtime tests skipped in CI. The answer that stands is to keep it and say
so at the call site: every namespace method issues `polars_online.InMemoryExpressionWarning`
(`_expr.py`, `_warn_in_memory`) with the reason, the plan to write instead, and the one-line
filter for someone using it on a frame in memory on purpose. It is a `UserWarning`, shown by
default from anywhere; a `DeprecationWarning` is hidden outside `__main__`, i.e. in exactly
the pipeline module where it matters (`tests/test_expr.py` checks both facts in a
subprocess). Nothing else changes: the plugin ships, `pl.Expr.online` is registered on
import, `po.online` is exported, the tests run in every build, and the README shows the two
forms side by side in its closing note. Nothing about the model needs the expression:
the bank fans out over (spec × group) with rayon, so `group=` is the parallel path
`.over(group)` is, and `df.online.fit_predict(specs)` is the in-memory call; what the
expression adds is features as expressions (a lag under `.over` stays in its group) and the
plugin ABI's MAJOR/MINOR handshake, the one polars stability guarantee we ride on
(CLAUDE.md rule 13).

**What would remove the warning.** A polars node that lets a user expression run per morsel,
in order, with state — i.e. a streaming-engine contract for stateful UDFs. Until then the
expression can only ever be the in-memory form, and the bank already is that.

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
      linkage macro step) reaches 0.998 / 0.999 / 0.998 on moons, rings and bars
      against DBSCAN's 1.000, with a measured rule for the threshold that decides
      it. `micro` is the design worth the build decision; §0 of the doc and
      ENHANCEMENTS §4 give the seven reasons and the structural limits. Merged to
      `main` 2026-09-04 as documentation and numpy prototypes. Nothing in the
      Rust crates; whether to build it is the user's call and §9 costs it.
      Research sources stay under the gitignored `.cache/research/`.

## 11a. Decisions made while implementing

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
- `docs/STATE-WORKFLOW.md` — research (2026-09-03) on carrying state out of a
  streamed plan: what polars does with a Python source, measured; the
  candidate forms; the rules `save_state=` on the plan follows and the
  decisions behind them (task 20).

## 11b. Performance plan

**Done — P1 through P11.** See [`docs/PERFORMANCE.md`](PERFORMANCE.md): the
integration layer cost 3–5× the model arithmetic and capped thread scaling at
3.2× on ten cores. Removing per-row allocation, flattening the rayon fan-out to
(spec × group × instance), extracting columns as `f64`-with-NaN instead of
`Option<f64>`, and pipelining the runner took it to **2.0–2.8× throughput and
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

[`docs/BEYOND-O-STATE.md`](BEYOND-O-STATE.md): what a relaxed memory bound would
unlock, checked against crates.io so "nobody has built this" is evidence rather than
assumption. Three strong candidates (adaptive conformal prediction, frequent-directions
sketching, rolling-window regression), three weak, and Hoeffding trees left to MOA on
purpose. Survey only — nothing proposed. The one condition attached: a relaxed bound
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
DBSCAN's ceiling on moons, rings and bars (0.998 / 0.999 / 0.998 against 1.000)
where every k-means and GMM scores 0.000 on the rings, with the threshold rule
measured and derivable at the checkpoint — it is the design worth the build
decision, for the seven reasons in §0 and ENHANCEMENTS §4. §8 settles the spec
and the static output schema, §9 costs a Rust build, §10 lists what failed.
Investigation only — nothing in the crates; the build decision is the user's
(task 22). ENHANCEMENTS §5.1 is the follow-on inventory of what else fits the
online contract (E36–E42).

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
