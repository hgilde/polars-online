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

- [x] 23. **`kmeans` — online k-means in the crates.** `crates/online-core/src/cluster/`
      (`summary.rs`: the §6.1 mean-form summary and the diagonal feature moments the
      metric reads; `kmeans.rs`: seeding over a bounded warm-up buffer, the assignment,
      the checkpointed centre update, the split–merge move on its own clock), every
      step of `docs/EXTENDING.md`, a from-scratch numpy reference
      (`tests/reference_cluster.py`) held bit-exact, the prototype as a second oracle,
      large streams, and the edge cases §11a lists.
      Built 2026-09-05, on branch `clustering-build`: `ModelState::KMeans`, the spec
      variant with `k`, `warm_rows`, `seed_rule`, `seed`, `update_every`,
      `split_merge`, `sm_every`, `dead_frac`, `standardize`; outputs `cluster` (`i32`),
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
- [ ] 43. **A run with no per-row output** (E50): `po.run(output=None,
      save_state=)`, `online --no-output`; with it E53, `targets` optional
      in TOML for the unsupervised models.

## 11a. Decisions made while implementing

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
  (1), `split_merge` (0.5), `sm_every` (100), `dead_frac` (0.05),
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
  `sm_every` learned rows at a checkpoint: merge the closest pair when
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
  construction); a recent-share dead test over `sm_every` windows (a second
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
  accumulated statistics the filter does not keep. No `coef0` on `kalman`.
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
- *A frozen schema-4 fixture is still owed.* Each schema has one
  (`state_v1.rs`, `state_schema2.rs`, `state_schema3.rs`); schema 4's waits
  until the tasks that may still add spec fields (41–43) have landed, so it
  is frozen once rather than three times.

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
