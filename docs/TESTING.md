# Test coverage and testing improvements

Status as of 2026-08-30: **61 Rust tests + 123 pytest functions**, all green,
run in CI on three OSes.

**Progress on this document's own backlog** (updated as items land):

| Item | Status |
|---|---|
| T-E1 negative weights | **Done** — defect fixed (finite negative weights now error, naming the row); non-finite weights skip uniformly with non-finite features. `tests/test_edge_cases.py::TestWeights`, incl. all seven models and the `w = 0` pure-decay case. |
| T-E2 null group keys | **Done** — defect fixed: `GroupKey(Option<String>)` replaces the `"<null>"` string sentinel, so a null group is structurally distinct from a group named `"<null>"`. Bank files gained `format_version` (v2); v1 files still load because the key serializes transparently as its inner `Option`. `TestGroupKeys`, incl. save/load and integer group columns. |
| T-E3 non-finite inputs | **Done** — `TestNonFinite` pins ±inf/NaN in features, targets, weights and the clock, plus "outputs are never non-finite". |
| T-E4 mis-ordered chunks | **Done** — `TestClockOrdering` pins current behavior (a backwards delta across a chunk boundary goes through `on_clock_reset`, indistinguishable from real data), and guards that correctly ordered chunking stays invariant. The strict mode remains ENHANCEMENTS E3. |
| T-A1 Kalman oracle | **Done** — `kalman_ref` in `tests/reference.py`; agreement to 1e-9 (observed max 1.1e-15) across scalar and per-factor `coef_halflife`, `inf` pinning, explicit `q`, fixed `obs_var`/`p0`, multi-target with and without `share_p`, the null policy, and `add_intercept=False`. |
| T-A2 lasso KKT | **Done** — `tests/test_oracles.py::TestLassoOptimality` checks stationarity/subgradient conditions on the model's own standardized stats (not a ported solver), across λ ∈ {0, 0.01, 0.1} × `l1_ratio` ∈ {1.0, 0.5}, plus path sparsity monotonicity and the intercept identity. |
| T-D1 run release CI | **Blocked on a push** — `origin` exists (`github.com/hgilde/polars-online`) but nothing has been pushed; running the workflow needs an explicit go-ahead. |
 This document assesses what they actually
prove, then lists concrete improvements — with emphasis on edge cases and on
comparing behavior against reference implementations, including
[river](https://riverml.xyz).

## 1. What is covered today

Scorecard against PLAN §9's eight test classes:

| Class | Status |
|---|---|
| 1. Oracle agreement | **Mostly done.** `ewridge` and `rls` match `tests/reference.py` to 1e-9 (incl. multi-target, standardize, `lam` decay, row-count clock), `rls ≡ ewridge(ridge_decay, solve_every=1)` to <1e-9, **Kalman matches `kalman_ref` to ~1e-15** across every configuration, and the **lasso is verified against its KKT conditions** rather than a ported solver. Remaining: huber/quantile (T-A3) and ftrl (T-A4) are still anchored only by property tests. |
| 2. Chunk invariance | Done: bitwise at the bank (1/7/400 chunks), expression, and CLI (`chunk_rows` sweep) levels; save/load mid-stream identical; the `coef` field is correctly excluded (chunk-dependent by design). |
| 3. Out-of-sample by construction | Done: IC ≈ 0 on pure-noise targets asserted for ewridge, kalman, huber, ftrl; lasso selection prefers the all-zero penalty on noise; robust reweighting proven to use the *prior* residual. |
| 4. Clock semantics | Done: cap, negative-delta (`max`/`zero`/`reset_state`), session gap and reset, first row, row-count clock, skipped-row decay folding, per-group independence. |
| 5. Null policy & warmup | Done for feature/target/weight nulls and `min_periods` — for the tested models (ewridge, rls, kalman-adjacent paths). |
| 6. Expression ≡ bank | Done for every model, incl. grids and `.over()`. |
| 7. Cross-platform state | **Defined but never executed on real runners.** The test file and the macOS→Windows/Linux artifact hand-off exist in `release.yml`, but no workflow run has happened (no remote/tag yet). Locally only same-OS round-trip + byte-determinism are proven. |
| 8. Benchmark | Done (`scripts/benchmark.py`, numbers in README). |

**Do we compare edge cases against reference implementations?** Partially.
Clock semantics, null policy, and warmup are cross-checked against the numpy
oracles — but only along the `ewridge`/`rls` paths, since those are the only
oracles that exist. Kalman/lasso/robust/ftrl edge behavior is anchored to
nothing outside this repo. **We never compare against river at all.** And two
live edge-case defects were found while writing this document (T-E3, T-E7
below) — evidence the edge matrix has real holes, not hypothetical ones.

## 2. Improvements

Priorities: **P1** = closes a PLAN promise or covers a found defect; **P2** =
meaningful new assurance; **P3** = infrastructure.

### A. Close the oracle gaps (our own references)

| # | P | Improvement |
|---|---|---|
| T-A1 | ~~P1~~ **done** | **`kalman_ref` in `tests/reference.py`** — plain numpy predict/update recursion mirroring the standardization-from-prior-stats scheme; agreement to 1e-9 (observed 1.1e-15) incl. per-factor halflife, `inf` pinning, explicit `q`, `obs_var`/`p0`, `share_p`, no-intercept, and the null-target path. Writing it confirmed several subtleties are load-bearing: scales come from the stats *before* the row, `Q·Δclock` is applied once per shared `P`, and the innovation variance carries `σ²/w`. |
| T-A2 | ~~P1~~ **done** | **Lasso KKT verification** — at the emitted coefficient snapshot, stationarity is checked on the model's own standardized statistics (`g_i = c_i − (Cb)_i − l2·b_i` must equal `l1·sign(b_i)` where `b_i ≠ 0`, and satisfy `|g_i| ≤ l1` where it is zero), across λ × `l1_ratio` combinations, plus sparsity monotonicity along the path and the intercept identity. A numpy CD `lasso_ref` for the *pred* path is still open (would catch schedule/warm-start bugs the KKT check cannot see). |
| T-A3 | P2 | **`huber_ref` / `quantile_ref`**: the IRLS reweighting is ~20 lines of numpy on top of `ewridge_ref`; assert 1e-9 agreement including the prior-residual weighting and per-target accumulators. |
| T-A4 | P2 | **`ftrl_ref`**: direct numpy port of the McMahan recursion; assert 1e-12 agreement including decay, weights, and null targets. |
| T-A5 | P2 | Run the **existing** edge-case tests (nulls, clock, warmup) through *every* model, parametrized, not just ewridge — the semantics are claimed to be model-independent; the tests should say so. |

### B. Cross-checks against river (new `tests/test_river.py`)

Add river as a dev-dependency; skip the module cleanly when it is not
installed (same pattern as the offline skip). Two tiers:

**Exact (tolerance ~1e-12, config pinned so the algorithms coincide):**

| # | P | Comparison |
|---|---|---|
| T-R1 | P1 | **`ftrl` ≡ `river.optim.FTRLProximal`** driving `river.linear_model.LogisticRegression`: same α, β, l1, l2; our `add_intercept=False` vs their `intercept_lr=0`; unit weights; no decay. Same published recursion on both sides — any drift is a bug in one of us. Then extend with decay/weights on our side only, asserting we reduce to river when both are off. |
| T-R2 | P2 | **Kalman(q=0, fixed `obs_var`) ≡ `river.linear_model.BayesianLinearRegression`** — *blocked on* the `standardize: false` Kalman switch (ENHANCEMENTS E19); once added, the recursions are the same rank-1 Bayesian update. |
| T-R3 | P2 | **`EwCov` (λ=1, unit weights) ≡ `river.stats.Mean` / `Var` / `Cov` / `PearsonCorr`**: our mean-form accumulators with no decay are exact running moments; river's Welford implementations are the independent route to the same numbers. Also validates our raw-moment formulation against Welford on ordinary scales (see T-E9 for where it breaks). |

**Statistical (tail agreement after warmup, tolerance stated per test):**

| # | P | Comparison |
|---|---|---|
| T-R4 | P2 | **EW mean/var vs `river.stats.EWMean` / `EWVar`** with the mapping `fading_factor = 1 − 0.5^(1/halflife)` on a row-count clock. Warmups differ by construction (ours is exact weighted-mean warmup; river seeds from early points), so assert convergence of the two sequences, not equality — this *is* the edge-case comparison of warmup semantics, made explicit. |
| T-R5 | P3 | **Quantile regression (intercept-only) vs `river.stats.Quantile`** (P² algorithm) on i.i.d. data: both should converge to the true quantile; assert both land within a shared CI. Guards our IRLS approximation against silent bias. |
| T-R6 | P3 | **Huber vs `river.linear_model.LinearRegression(loss=optim.losses.Huber)`** under contamination: not numerically comparable (exact IRLS vs SGD), but both must recover the clean slope where OLS fails — same assertion, two libraries. |

### C. Edge-case matrix to add

Findings first — both verified against the current build:

| # | P | Case | Current behavior → required |
|---|---|---|---|
| T-E1 | ~~P1~~ **done** | **Negative weight value** | **Was a defect**, now fixed. `EwCov::update` silently no-opped when `λW + w ≤ 0` while the per-target `r_j` update ran with a negative denominator — in practice every later prediction went null. Finite negative weights are now rejected at extraction, naming the column, value and row; non-finite ones skip the row like any other non-finite input; `w = 0` is a legal pure-decay row. `EwCov::update` also debug-asserts the contract. |
| T-E2 | ~~P1~~ **done** | **Null group key vs a group literally named `"<null>"`** | **Was a defect**, now fixed. Both mapped to the string `"<null>"` and shared one stream (verified: `n_eff` accumulated across them). Replaced by `GroupKey(Option<String>)`; bank files gained a `format_version` (now 2) and v1 files still load, since the key serializes transparently as its inner `Option`. |
| T-E3 | ~~P1~~ **done** | ±inf in features / targets / weight / clock | Pinned: a non-finite feature or weight skips the row (clock still advances), a non-finite target is predict-only, a non-finite or null clock errors loudly. Plus a fuzz-ish test that outputs are never non-finite. |
| T-E4 | ~~P1~~ **done** (current behavior) | Mis-ordered chunks (clock goes backwards across a chunk boundary within a group) | Pinned as-is: absorbed by `on_clock_reset`, indistinguishable from a genuine backwards clock. Strict mode remains ENHANCEMENTS E3; the invariance guard rail is tested alongside. |
| T-E5 | P2 | Degenerate solves in the **plain** (non-standardized) path: constant feature, exactly collinear features (`x1 = x0`) | The jitter fallback and previous-coefficient retention exist but only the standardized zero-variance path is tested. Assert finite outputs and that `solve_failures` increments (needs E5 to observe). |
| T-E6 | P2 | Duplicate clock values (Δ=0 runs), `max_dclock = 0`, `halflife` far below the median Δ (λ ≈ 0 per row) | Assert: no NaN leakage, `n_eff ≈ w`, solves survive near-singular S via jitter. |
| T-E7 | P2 | Minimal shapes: single-row groups, a group appearing in only one chunk, an empty chunk (`df.height() == 0`) fed to `fit_predict`, k=1/m=1 | Empty-chunk behavior is currently untested at the bank level (the CLI runner handles empty *input* only). |
| T-E8 | P2 | Non-string group and session columns (ints, categoricals) and null session values | The cast-to-string path is exercised only with string columns; test int groups and that a null session change is detected (or defined not to be). |
| T-E9 | P2 | **Large-offset cancellation**: features like `1e8 + noise` | Our raw-moment form computes `var = E[x²] − m²`, which loses ~half the mantissa at that scale, where river's Welford form does not. Test to characterize the loss and document the operating range — and if it matters for real data (prices are ~1e4–1e5), adopt Welford/centered updates in `EwCov` as the fix. |
| T-E10 | P2 | Datetime-typed clock columns | Cast to f64 yields epoch *microseconds* — a halflife of "600" then means 600 µs, silently. Decide (reject non-float clocks, or document loudly) and test the decision. |
| T-E11 | P3 | Long-stream soak: ~10⁷ rows through one state | Assert boundedness (`n_eff`, S entries), no drift vs. a mid-stream save/restore, and stable throughput — the stability claim behind mean-form accumulators (PLAN §7), currently asserted only by argument. |
| T-E12 | P3 | Session change on the first row of a group; `coef_every = 1`; skipped row immediately before a save/load boundary (pending-delta serialization) | The last is likely covered *incidentally* by the null pattern in the invariance tests; make it a targeted test. |

### D. Infrastructure

| # | P | Improvement |
|---|---|---|
| T-D1 | P1 | **Actually run the release workflow once** (tag or `workflow_dispatch`): until then, class 7 (macOS-written state loaded on Windows) and the wheel builds are untested claims. Follow with a tolerance-based (not bitwise — BLAS/LLVM vectorization may differ) cross-OS *prediction* comparison in the same workflow. |
| T-D2 | P2 | **Property-based testing**: `hypothesis` (Python) and/or `proptest` (Rust) generating adversarial streams — mixed nulls, dup/backwards clocks, constant features, weight extremes, tiny groups — asserting the universal invariants (chunk invariance, save/load equivalence, no non-finite outputs, null policy) for every model. This is the systematic version of section C. |
| T-D3 | P2 | Determinism across parallelism: run the bank under `RAYON_NUM_THREADS=1` vs many; outputs must be identical (per-stream work is serial, so any diff is a bug). |
| T-D4 | P3 | Coverage measurement (`cargo llvm-cov` + `pytest --cov`) with a reported (not gating) number, and a periodic `cargo mutants` pass on `online-core` — solver/decay arithmetic is exactly where mutation testing earns its keep. |

## Suggested order

1. T-E1, T-E2 (found defects, with their fixes), T-D1 (run the release CI once).
2. T-A1–T-A4 (finish the oracle set PLAN promised), T-A5, T-E3–T-E5.
3. T-R1, T-R3, T-R4 (river cross-checks that need no new features), then T-R2
   behind the Kalman switch.
4. T-D2/T-D3, remaining edge matrix, T-D4.
