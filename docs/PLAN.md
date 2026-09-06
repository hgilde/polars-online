# polars-online — design and plan

Status as of 2026-09-06: design frozen 2026-08-29; **tasks 1–58 done**,
released as 0.2.0. Items marked
**[validate]** were defaults chosen without data; task 12 checked them on
public data, and `docs/VALIDATION.md` is the regenerated record.

How to read this file: §1–§10 are the original design, kept as written. §11 is the
task list, ticked as each task landed. §11a holds the decisions taken while building
— the place to look when the code does something §1–§10 did not say. §11b–§11h point
at the follow-on documents, and §12 records the questions the design left open and
what became of them. `docs/README.md` maps every document.

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
      a diurnal factor and a volume process; returns `bars`, `truth_rows`,
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
- [x] 61. **The leak test's statistic, 2026-09-06.** `assert_plateaus` compared
      the first and last of its post-warm-up marks, which cannot distinguish a
      late allocator step from a slope — the distinction its own docstring
      claims. It failed `main` on a tree whose previous run was green. It now
      takes the median block-to-block gap over five blocks; the CI trace that
      failed reads 0.0 KB/iter, a sustained 14 KB/iter leak still reads 14.
      Acceptance: the file's 16 tests, and the three traces (the real one, a
      leak, a step) checked against the statistic directly.

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
  construction). Written atomically at the end through a second writer
  thread on the `write_file` path — temp sibling, rename, published only
  when the run completes, before `save_state` — and **not** appended per
  chunk: E35's rule, one code path. `output=None` with `closed_groups=` is
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
  series' last value, `n_ticks_<s>` = ticks of `s` since the previous point,
  `retained_fraction` = `m / Σ_s n_ticks_s` for the interval, and `keep`
  columns at their value on the completing tick. Then every flag clears
  and the counts restart. The docstring carries BNHLS §2.1's caveat
  (ANSWERS): the refresh vector is treated as observed at `time_refresh`,
  though each series' value is stale by up to one of its own inter-tick
  intervals — the price of a common grid, and why `n_ticks_<s>` is emitted
  (a large count on one series is that series' staleness made visible).
  `pairs=True` runs an independent two-series state
  per pair and emits the long frame `(by?, pair, time_refresh, a_value,
  b_value, n_ticks_a, n_ticks_b)` with `pair = "a|b"` in `names` order.
- *Tests.* A longhand Python loop over the same rows on random Poisson
  streams (the oracle), with `by` and with `pairs`; §10's three-series
  example — `n = 8, 9, 10` ticks giving `N = 7` and `21/27` retained, which
  ANSWERS verified against BNHLS but whose tick times exist only as the
  paper's figure — so the test is a constructed three-series stream with
  those tick counts that the loop reduces to exactly `N = 7`, asserting
  `N ≤ min nᵢ` and `retained = N·m/Σnᵢ = 21/27`; a synchronous
  input (every series ticks at every time) returned with `n_ticks = 1` and
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
- *`retained_fraction` is per interval, not cumulative.* `m / Σ n_ticks` for
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
  h_max` (the return `m` rows back can no longer be replaced by the end
  jitter); at close, add the products of `x̃_end` with the last `h_max` final
  returns. That is what makes the state a ring of `h_max + m` vectors and
  the close `O(h_max·k²)`, with no retraction. Bandwidth: `bandwidth` an
  integer `H` (then `h_max = H`, `n_max` unneeded) or `"auto"`, BNHLS §4.1
  as ANSWERS read it: `c* = ((12)²/0.269)^{1/5} = 3.5134` for Parzen (the
  kernel constant `k''(0)²/∫k² = 12²/0.269`); per feature `ξ̂ᵢ² =
  ω̂ᵢ²/IV̂ᵢ`, with `IV̂ᵢ` the realised variance on a sparse grid
  (`iv_stride`, default 20 rows, averaged over the `iv_stride` offsets —
  their 20-minute RV) and `ω̂ᵢ²` the mean over `q` offsets of
  `RV_dense^{(i)}/(2n^{(i)})` taken on every `q`-th return (`noise_stride`,
  default `1`, i.e. the full dense grid; their `q ≈ 25` trades or `≈ 70`
  quotes for a ~2-minute grid — the estimate is deliberately upward biased,
  which their §4.1 accepts, and a stride above 1 is the caller's call); `Hᵢ
  = c*·ξ̂ᵢ^{4/5}·n^{3/5}`, `H = ⌈mean Hᵢ⌉` clipped to `h_max`, reported as
  `bandwidth_used`. `"auto"` requires `n_max`, from which `h_max` defaults
  to `⌈c*·n_max^{3/5}⌉` (`ξ̂ = 1`, a noise variance equal to the block's
  integrated variance, beyond which the estimator is not worth having);
  `h_max` may be given. `n_max` is a sizing hint, not a limit: a longer
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
  the block starts, so it is `⌊θ·√n_max⌋` (or the §3 form on `n_max`) from
  a required `n_max`, or a given `window`. Streaming: a ring of `kₙ − 1`
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
  49 over task 52's asynchronous bars) to `1e-10` relative; the jittered
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

- **CKP §3 was not read, and the source says so.** `psd = true`'s longer
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
  rng(seed)` for every draw, returning `{"bars", "truth_rows",
  "truth_blocks"}` as `pl.DataFrame`s. Everything after `m` is keyword-only
  (§10's positional list puts required names after defaulted ones):
  `regimes(m, *, states, transition, n_blocks, bars_per_block,
  durations=None, design="step", smooth_bars=0, phi=0.0, vol_state=0.0,
  async_rates=None, noise=0.0, diurnal=None, session_bars=None,
  volume=None, seed=0)`.
- *The chain.* `states` is a list of `K` correlation matrices (`m × m`,
  checked for a unit diagonal and PSD) or `K` floats, each an
  equicorrelation; `transition` is `K × K` row-stochastic and drives one
  state per block; `durations` (per state, in blocks) makes the sojourn
  deterministic and uses `transition` with its diagonal removed to pick the
  next state — the recurring-state design. `design="step"` switches at the
  block boundary; `design="smooth"` interpolates the correlation matrix
  linearly over `smooth_bars` bars around it (a convex combination of two
  correlation matrices is one).
- *The series.* Latent returns `εₜ ~ N(0, Rₜ)` by Cholesky per distinct
  `Rₜ` (cached — one factorisation per state under `"step"`), `yᵢₜ = φᵢ
  yᵢ,ₜ₋₁ + εᵢₜ` with `phi` a scalar or per-series list (the documented
  truth is the innovation correlation; the AR filter moves the return
  correlation of unequal-`φ` pairs, which is what task 51's `fisher_se`
  inflation is for), scaled by `exp(vol_state · sₜ)` — a scalar `vol_state`
  makes volatility rise with the state index, a list gives it per state.
  `diurnal`, a list of `session_bars` multipliers in `(0, 1]`, scales the
  off-diagonal of `Rₜ` at bar `t mod session_bars` (a mix toward the
  identity, so PSD is kept); `session_bars` defaults to `bars_per_block`;
  `session = t // session_bars`. `volume`, `(mean, shape)`, draws a Gamma
  volume per bar with the mean scaled by the same `exp(vol_state · sₜ)`;
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
- *Frames.* `bars`: `instrument` (a constant string `"sim"`, the join key a
  multi-instrument caller would vary), `t`, `clock`, `session`, `x_1 … x_m`,
  `volume` (null when `volume=None`). `truth_rows`: `t`, `block`, `state`,
  `vol_mult`, `mix` (the `"smooth"` fraction, `0` otherwise). `truth_blocks`:
  `block`, `state`, `n_bars`, `corr` (vech of the block's mean true
  correlation — the state's matrix under `"step"`).
- *Tests.* Per-state matrices recovered from the block sample correlations,
  pooled over the blocks in each state, within `3·SE` of the Fisher-z noise
  floor; a `phi` recovered from the lag-1 autocorrelation; the Epps curve of
  an asynchronous simulation — the sample correlation of previous-tick
  returns at sampling intervals `1, 5, 20, 100` bars — increasing toward the
  truth; `noise > 0` giving a negative lag-1 return autocorrelation; every
  `Rₜ` under `"smooth"` PSD; `session` and `clock` consistent with
  `session_bars` and `volume`; two calls with the same seed byte-identical
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
- *The Cholesky factor is cached by `(state, previous state, diurnal bar,
  mix)`.* Under `"step"` with no diurnal that is one factorisation per
  state, which is what the row asked for; under `"smooth"` the mix is part
  of the key, so a ramp of `smooth_bars` costs that many factorisations and
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
  stands only if §3 agrees.
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

## Follow-on documents

Each of §11b–§11h below summarises one document under `docs/` and says what
became of it. Three more carry no section of their own:

- `docs/ENHANCEMENTS.md` — every model and feature after the first seven
  (E1–E64): proposed, measured, built or declined.
- `docs/TESTING.md` — coverage scorecard against §9, the defects the suite
  found, and the oracle/river cross-checks.
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
