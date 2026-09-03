# Test coverage and testing improvements

Status as of 2026-08-30: **230 Rust tests + 649 pytest functions** (plus 2 opt-in
soak tests), all green, run in CI on three OSes.

Measured coverage (`./scripts/coverage.sh`): **96% of the Python package**, and
**75% region / 73% line** of the Rust workspace. The Rust figure understates
reality — `cargo llvm-cov` only sees what `cargo test` runs, so `online-py`
(0%) and much of `online-polars` are exercised by the pytest suite through the
compiled extension, invisible to the Rust instrumentation. The genuinely thin
spots it does reveal are `online-cli/src/main.rs` (argument plumbing, covered
instead by the CLI integration tests through `run_config`) and
`online-core/src/robust.rs` (75%).

**Progress on this document's own backlog** (updated as items land):

| Item | Status |
|---|---|
| Windows portability of the *tests* | **Done — the first Windows run found 9, all in tests, none in the library.** After the TOML fix cleared the Rust side, the pytest suite failed nine ways on Windows, in four classes, every one a test making a Unix assumption: **(1) encoding** — `Path.read_text()` and `subprocess(text=True)` default to the locale codec, which is cp1252 on Windows and cannot decode the box-drawing characters polars prints or the arrows in our own README (`UnicodeDecodeError: 'charmap' codec ... byte 0x90`); fixed suite-wide with explicit `encoding="utf-8"` in 7 files rather than only at the reported sites. **(2) path separators** — `test_a_clean_checkout_has_what_the_build_needs` compared `str(WindowsPath(...))` (backslashes) against git's forward-slash output and declared a tracked file missing; now `as_posix()`. **(3) environment** — the thread-determinism subprocess passed a hardcoded `PATH=/usr/bin:/bin`, leaving the child with no resolvable interpreter; now inherits `os.environ`. **(4) shell** — `test_release_packaging` executes a `run:` block that only ever runs on the workflow's ubuntu job, and Git Bash's `find` differs enough to fail for reasons that say nothing about the workflow; skipped on Windows, where Linux and macOS still cover it. That the library itself passed all 718 tests on Windows first time is the result worth recording. |
| Production hardening round 3 (vs river's own battery) | **Done — found three defects and one doc-API mismatch.** Calibrated against river 0.26.1's `river.checks` (37 checks), sklearn's estimator invariances and statsmodels' oracle convention. `tests/test_production_hardening.py` (38 tests): **(1) a target listed as its own feature was accepted — corr(pred, y) = 1.000000, perfect leakage through the door hard rule 2 does not guard**; now a validation error for every model, with the message pointing at a lagged copy. **(2) duplicate feature/target names were accepted**, silently splitting a coefficient across identical slots on an exactly singular system; rejected. **(3) the finite-or-null contract filtered only NaN** — an exactly ±inf prediction from a diverged model would have reached the user; the boundary now checks `is_finite`. **(4)** `clip_gradient=inf` is the documented way to disable clipping but the JSON layer refused it while `halflife=inf` worked; `clip_gradient` now uses the same `Num` type. Also implemented river's checks that apply and we lacked: **feature-order invariance** (spec order and frame order, plus extra-columns tolerance), **pickling** (`__reduce__` via `save_bytes`, so pickle and `copy.deepcopy` resume bit-exactly — needed `#[pyclass(module=...)]`, since pickle cannot name a class that claims to live in `builtins`), all-null columns, first-row outliers with the washout horizon stated (~80 halflives, inherent to EW accumulators), a 15-case extreme-parameter sweep (p0 at 1e±12, τ at 0.01/0.99, deliberate SGD divergence pinning finite-or-null), state-byte corruption (every magic-string byte must detect; any flip anywhere must fail cleanly, never panic), concurrent `fit_predict` from two threads (safe error, object usable after), unicode/space column names through save/load, and README python blocks compiling (since IMPROVEMENTS T5 they are *run*, each in its own copy of a namespace holding what the prose has introduced by that point -- which found the README's `holt` example refused by the library's own validation). River checks *not* adopted, with reasons: clone/repr/params (our specs are plain dicts), emerging/disappearing features (a fixed schema is a design decision, pinned as a clear error instead). |
| Post-refactor hardening (P1–P8 review) | **Done** — a second full review after the performance rewrite, aimed at the two blind spots the request named: parameter ranges tested only in their comfortable middles, and row counts too small to expose stride bugs in the new flat slot-major buffers. `tests/test_hardening.py` (18 tests, ~4.5 s): a kitchen-sink stream at **30k rows with every output enabled at once** (156+ fields: two instances × ridge grid × feature sets × two targets, sigma/z/drift/metrics/autocorr/quantiles/selected/averaged, groups, sessions, weights, nulls, clock gaps) compared across chunkings incl. row-at-a-time, across a mid-stream save/load, and across `RAYON_NUM_THREADS` 1 vs 8 by digest; the **coupled drift path** (grid + `drift_action="reset"`), which P2 added and nothing executed — a break in either instance resets both, the row-major path is chunk-invariant, and it equals the parallel path when nothing fires; parameter edges — halflife 1e-3 and inf give their exact limits, **weight-scale invariance at 1e±6** (the test that the mean form is real: all weights ×c changes nothing but `n_eff`), k=64, twelve targets with per-target warmup landing on the exact ceil(threshold) row, quantile levels 0.001/0.999, `coef_every` 0/1/997 across chunk boundaries; P6's reader thread ends in a clean exception on a corrupt file and on a mid-stream bank error (no deadlock on either side of the channel); and the P5 spec cache keeps two specs on one thread distinct. **Zero new defects** — every path held, including the never-executed one. The soak (10M rows) re-run post-refactor: passes. |
| The validated defaults are still the measured ones | **Done** — `tests/test_validation_doc.py` regenerates `docs/VALIDATION.md` and compares it to the committed copy (timings normalized away), so the numbers the shipped defaults were chosen from cannot silently stop being true. Re-run after the whole enhancement backlog including E11b's centered moments: **every number is unchanged**, which is itself the result — the numerics work did not move the measured optima. A second assertion checks the script still runs all five experiments, so the comparison cannot pass on a document that no longer justifies anything. |
| Declared schema checked for every model | **Done** — `test_names_match_the_realized_struct_for_every_model` extends the E23 guard from `ewridge` alone to all ten models times three output combinations, plus `ew_cov` separately. The optional outputs are assembled in the stream layer and so generalize, but each model contributes its own prediction and coefficient slots, and `sgd`, `pa`, `holt` and `ew_cov` all postdate the original test. |
| Hard rule 1 is enforced, not remembered | **Done** — `tests/test_repo_hygiene.py` fails if a data file, a large file, or generated tool output is tracked, if `.cache/`/`target/`/`mutants.out/` stop being gitignored, or if a file the build needs is missing from what `git archive` (a fresh clone, an sdist) would produce. Written after 136 files of `cargo mutants` output sat tracked for several commits, swept in by a `git add -A`, with nothing complaining. Verified to fire, not just to pass. |
| Examples are executed | **Done** — `tests/test_examples.py` runs everything under `examples/` unmodified: the Pathway operator example end to end (plain-batch path, since Pathway is BSL and not a dependency — asserted by checking it appears in no dependency group), and `examples/bank.toml` through the real CLI for `--dry-run`, a full run, and `--resume` from the state the run wrote. A documented example that no longer works is worse than none, and until now nothing ran either file. **It found a documentation defect**: the README's chunk-invariance guarantee said "bit-identical output" with no exception, but `coef` is snapshotted on each chunk's last row as well as every `coef_every` rows, so smaller chunks report it more often. The guarantee now names that exception; every computed field is still bit-identical. |
| T-E1 negative weights | **Done** — defect fixed (finite negative weights now error, naming the row); non-finite weights skip uniformly with non-finite features. `tests/test_edge_cases.py::TestWeights`, incl. all seven models and the `w = 0` pure-decay case. |
| T-E2 null group keys | **Done** — defect fixed: `GroupKey(Option<String>)` replaces the `"<null>"` string sentinel, so a null group is structurally distinct from a group named `"<null>"`. Bank files gained `format_version` (v2); v1 files still load because the key serializes transparently as its inner `Option`. `TestGroupKeys`, incl. save/load and integer group columns. |
| T-E3 non-finite inputs | **Done** — `TestNonFinite` pins ±inf/NaN in features, targets, weights and the clock, plus "outputs are never non-finite". |
| T-E4 mis-ordered chunks | **Done** — `TestClockOrdering` pins current behavior (a backwards delta across a chunk boundary goes through `on_clock_reset`, indistinguishable from real data), and guards that correctly ordered chunking stays invariant. The strict mode remains ENHANCEMENTS E3. |
| T-A1 Kalman oracle | **Done** — `kalman_ref` in `tests/reference.py`; agreement to 1e-9 (observed max 1.1e-15) across scalar and per-factor `coef_halflife`, `inf` pinning, explicit `q`, fixed `obs_var`/`p0`, multi-target with and without `share_p`, the null policy, and `add_intercept=False`. |
| T-A2 lasso KKT | **Done** — `tests/test_oracles.py::TestLassoOptimality` checks stationarity/subgradient conditions on the model's own standardized stats (not a ported solver), across λ ∈ {0, 0.01, 0.1} × `l1_ratio` ∈ {1.0, 0.5}, plus path sparsity monotonicity and the intercept identity. |
| T-R1 FTRL vs river | **Done** — `tests/test_river.py`. Driven by the same gradient sequence, river's `optim.FTRLProximal` and our model agree on the z/n recursion to 1e-12, row for row, with and without L1. Also pinned: **river's `LogisticRegression` predicts with the previous step's proximal weights**, while we follow McMahan Algorithm 1 and recompute from `z` at prediction time — a real semantic difference, now a test rather than a surprise. |
| T-R4 EW moments vs river | **Done** — and it found a second convention difference: ours is the *bias-corrected* weighted mean (divides by accumulated weight, exact from the first row), river's `EWMean` is the un-normalized EWMA seeded at its first value, which stays anchored near that seed during warmup. Both the closed-form match and the convergence in the limit are asserted. |
| T-R5 / T-R6 quantile, Huber | **Done** (statistical tier) — our IRLS quantile lands near the empirical quantile alongside river's P² estimator; our Huber and river's SGD-Huber both beat least squares under 3% contamination and agree with each other. |
| T-A3 huber/quantile oracles | **Done** — `robust_ref` in `tests/reference.py`; agreement to ~1e-13 across `huber_delta` ∈ {0.5, 1.5, 10}, τ ∈ {0.1, 0.5, 0.9}, nulls, multi-target and standardized solves. |
| T-A4 ftrl oracle | **Done** — `ftrl_ref`; agreement to ~1e-16 with and without clock decay, across `l1` ∈ {0, 0.5, 5}, custom α/β/l2, no-intercept and null targets. |
| T-A5 semantics across all models | **Done** — `tests/test_semantics_all_models.py` runs null policy, warmup, clock semantics and the universal invariants (chunk invariance, save/load, group independence, expression ≡ bank, no non-finite outputs) against **all ten models** (extended to `sgd`, `pa` and `holt` when those landed, which immediately **found a second defect**: `sgd` and `pa` reported `n_eff` with the current row's decay already applied, while every other model reports the weight before the row's update and before its decay, so `min_periods` meant a slightly different number of rows depending on the model. Both now follow the documented convention). **It found a real defect**: the robust models were reporting `n_eff` as the sum of *IRLS weights* rather than observations, so a quantile spec showed `n_eff ≈ 1001` after three rows (quantile weights reach `2/quantile_eps` ≈ 2000×) and `min_periods` was effectively inert. Fixed: `Robust` now tracks a raw-weight observation count for `n_eff`/`min_periods` while the accumulators keep using the robust weights. |
| T-E5–T-E10, T-E12 | **Done** — degenerate solves (collinear/constant features in the plain path, with `solve_failures` now observable), duplicate/zero/extreme clock deltas, empty chunks and minimal shapes, categorical groups and null session values, large-offset numerics, datetime/integer clocks, and the pending-delta-across-save/load case. **Found a second defect** (see T-E9 below). |
| T-D3 thread determinism | **Done** — `tests/test_portability.py` runs the bank in subprocesses at `RAYON_NUM_THREADS=1` and `=8` and requires identical output. |
| T-W3/T-W4/T-W6 (locally testable) | **Done** — CRLF and LF configs both parse through the CLI; Windows-style escaped paths and paths with spaces round-trip; the exact output field-name list (which embeds formatted floats) is pinned, plus a `.gitattributes` normalizing line endings so a Windows checkout cannot introduce CRLF. |
| T-D2 property-based testing | **Done** — `tests/test_properties.py` (hypothesis) generates adversarial streams (mixed nulls, duplicate/long-gap clocks, ±1e8 values, zero weights, tiny groups) and asserts the universal invariants for all ten models, including the strongest one: **changing a row's own target never changes that row's own prediction** (out-of-sample by construction, hard rule 2). |
| T-E11 soak | **Done** — 10M rows through one state in ~6.5s: `n_eff` stays bounded and does not drift between the start and end of the stream, the fit is still accurate, and a 2M-row state serializes to under 4KB (memory is O(state), not O(data)). Opt-in via `pytest -m soak`. |
| T-D4 coverage | **Done** (reported, not gating) — `scripts/coverage.sh`; numbers above. |
| T-D5 mutation re-run | **Done, three passes.** Run 1 (before the follow-up work): **2616 mutants, 501 missed**. Run 2 (after): 104 missed — but see the warning below, that number was wrong. Run 3 (`--iterate`, 690 mutants in 14 min on an idle machine): **217 missed**, the honest current figure. 8.3% surviving, down from 31% (517/1645) at the first-ever pass, despite the crate having grown by 60%. The misses were not scattered: they clustered almost perfectly on the code whose *only* tests live in `tests/*.py`, because `cargo mutants` runs `cargo test` and cannot see the Python suite. Eight commits of Rust-side oracles followed; see "What the mutation run actually found" below. |
| T-D1 / Windows CI | **Blocked on credentials** — see "Windows and cross-platform" below. `origin` is `github.com/hgilde/polars-online` and 24 commits are ready, but this machine has no GitHub auth (no keychain entry, no SSH key, no token, no `gh`), so nothing has ever been pushed and **no CI job has ever run on Windows**. |

This document assesses what the tests actually prove, then lists concrete
improvements — with emphasis on edge cases and on comparing behavior against
reference implementations, including [river](https://riverml.xyz).

## 0. What the mutation run actually found

The headline number (501 of 2616 mutants surviving) is less interesting than
its shape. Grouped by function, the survivors were:

| Function | Missed | Why |
|---|---|---|
| shape and state accessors (`n_eff`, `n_targets`, `n_features`, `sigma2`, `coefficients`, `kind`) | 81 | asserted nowhere in Rust, in any model |
| `<Lasso as OnlineModel>::step` + `Lasso::solve` + `standardized` | 66 | KKT verification lives in `tests/test_lasso.py` |
| `EwRidge::blend_toward_long_run` | 54 | `session_shrink` is tested only from Python |
| `<EwRidge as OnlineModel>::step` + `solve` + `run_solve` | 54 | the slow twin, `sigma2`, and the solve schedule |
| `EwRidge::solve_standardized` | 53 | the `add_intercept = false` branch had no test at all |
| `*Cfg::validate` (seven models) | 34 | rejections are asserted in `tests/test_edge_cases.py` |
| `EwCovModel::read` / `labels` / `n_outputs` | 26 | `ew_cov` is reachable only through the Polars layer |
| `<Holt as OnlineModel>::step` | 19 | brand new, and its tests checked outcomes not arithmetic |
| `Kalman::pred_var` + `<Kalman as OnlineModel>::step` | 17 | surfaced only as `emit_sigma` / spec options |
| `EwCov::precision` / `with_precision_prior` / `partial_corr` | 15 | E2's precision matrix, solved on demand and exposed only as a statistic (C5) |

One cause explains nearly all of it: **`cargo mutants` runs `cargo test`, so
everything proven only by the pytest suite through the compiled extension is
invisible to it.** That is already noted in `scripts/mutants.sh` as the reason
for scoping the run to `online-core` — but the same blind spot applies *inside*
`online-core` wherever a feature's only oracle is a Python test.

Eight commits closed it, taking `online-core`'s Rust tests from 151 to 218. The rule followed was to add an
*oracle*, not a golden number: the recursion written out longhand beside the
implementation (Holt, Page-Hinkley, `sigma2`), an equivalent model configured a
different way (the slow twin against a standalone model at `long_halflife`;
the standardized solve against the plain one at zero penalty), the optimality
conditions of the problem being solved (the lasso's KKT conditions), or the
definition of the statistic (`read` against a recomputation from the raw rows).

Five real defects surfaced in the process, all in code the behavioural tests
were happy with. The most serious was a **zero-weight row at the head of a
stream permanently disabling `ewridge` and `lasso`**: their per-target mean-form
update computes `a = lam·wj / (lam·wj + w)`, which is 0/0 when nothing has ever
carried weight, and the NaN never washed out — `wj` stayed NaN, `NaN > 0.0` is
false, and the model silently stopped predicting for the rest of the stream.
Every other model already guarded it, and so did `EwCov::update` two lines
away. It was found indirectly: a Rust unit test for the analogous guard in
`blend_toward_long_run` failed, which pointed at the same shape in `step`.
`tests/test_edge_cases.py::TestWeights` now checks it for all ten models.

The other four:

- `sgd` and `pa` reported `n_eff` with the current row's decay already applied,
  so `min_periods` meant a different number of rows for them than for every
  other model;
- `EwCovModel::n_targets` returned 1 for a model that regresses nothing;
- `blend_toward_long_run` had lost its doc comment to a `#[cfg(test)]` helper
  inserted between the comment and the function;
- `coef0` misconfiguration reported "coef0 must be 1 vectors of length 3".

**Equivalent mutants** were left alone deliberately: they cannot be killed by
any test. `Ftrl::weight`'s `zz < 0.0` sign branch is only reachable when
`zz == 0`, which the `|zz| <= l1` guard above it has already returned on; and in
Holt's `beta > 0.0 && d_clock > 0.0`, the second test can never decide, because
beta is `1 - 0.5^(d/halflife)`, zero exactly when d is. Both now say so in the
source, so the next reader does not re-derive it.

### Do not run anything else while a mutation pass is going

Run 2 reported 104 survivors and 425 timeouts, and this document said 104 for
several commits. It was wrong by a factor of two. Run 3 (`--iterate`) re-tested
exactly those on an otherwise idle machine and found **132 of the "timeouts"
were ordinary survivors** — the tests had passed, they had just taken longer
than the ceiling because `./scripts/gate.sh` was running concurrently (cargo
build, cargo test, maturin, pytest, repeatedly, on the same four cores).
cargo-mutants counts a timeout separately from a miss, so the contention did not
look like an error. It looked like good news.

Two rules follow:

- **Let the machine be idle.** A mutation pass and a build loop cannot share a
  laptop without corrupting the classification.
- **Treat a large timeout count as a failed run, not a result.** Run 1: 175
  timeouts of 2616. Run 2, under load: 425. Run 3, idle: 18 of 690. Above a few
  percent, the numbers underneath are not trustworthy.

`--minimum-test-timeout 10` in `scripts/mutants.sh` makes a pass faster but
*more* sensitive to this, not less: right for an idle machine, wrong for a busy
one.

### Where the 217 stand now

| file | run 1 | run 3 |
|---|---|---|
| lasso.rs | 83 | 50 |
| stats.rs | 5 | 45 |
| ewridge.rs | 178 | 42 |
| robust.rs | 16 | 27 |
| sgd.rs | 15 | 15 |
| kalman.rs | 32 | 10 |
| ewcov.rs | 75 | 10 |
| everything else | 97 | 18 |

Three files went *up*, which is the same measurement artifact in reverse: their
run-1 figures were themselves depressed by timeouts. `stats.rs` is the clearest
case — 45 survivors in `P2Quantile` and `SlotMetrics`, both of which are loops
whose mutations spin. They are the obvious next batch of work, and the same rule
applies: add an oracle, not a golden number.

## 1. What is covered today

Scorecard against PLAN §9's eight test classes:

| Class | Status |
|---|---|
| 1. Oracle agreement | **Mostly done.** `ewridge` and `rls` match `tests/reference.py` to 1e-9 (incl. multi-target, standardize, `lam` decay, row-count clock), `rls ≡ ewridge(ridge_decay, solve_every=1)` to <1e-9, **Kalman matches `kalman_ref` to ~1e-15** across every configuration, and the **lasso is verified against its KKT conditions** rather than a ported solver. Remaining: huber/quantile (T-A3) and ftrl (T-A4) are still anchored only by property tests. |
| 2. Chunk invariance | Done: bitwise at the bank (1/7/400 chunks), expression, and CLI (`chunk_rows` sweep) levels; save/load mid-stream identical; the `coef` field is correctly excluded (chunk-dependent by design). |
| 3. Out-of-sample by construction | Done: IC ≈ 0 on pure-noise targets asserted for ewridge, kalman, huber, ftrl; lasso selection prefers the all-zero penalty on noise; robust reweighting proven to use the *prior* residual. |
| 4. Clock semantics | Done: cap, negative-delta (`max`/`zero`/`reset_state`), session gap and reset, first row, row-count clock, skipped-row decay folding, per-group independence. |
| 5. Null policy & warmup | Done for feature/target/weight nulls and `min_periods` — for the tested models (ewridge, rls, kalman-adjacent paths). |
| 6. Expression ≡ bank | Done for every model, incl. grids and `.over()`. Since 2026-09-03 every expression call also warns that it is the in-memory form (PLAN §6); `test_expr.py::TestTheExpressionWarnsThatItRunsInMemory` holds every method to that, and the suite silences the warning globally in `pyproject.toml`. |
| 6b. `predict` ≡ `fit_predict` of the next row | Done (E31): `tests/test_predict.py` holds row `i` of `predict(df)` to row 0 of `fit_predict(df.slice(i, 1))` on a fresh clone, field for field, for all ten models with every diagnostic on, and across every session/clock policy; `crates/online-core/tests/model_contract.rs` holds each model's `predict` to its `step` row by row. |
| 6c. Runner ≡ bank, every source and format | Done (E32): `crates/online-polars/tests/runner.rs` and `tests/test_runner.py` hold every input format (parquet, ipc, csv, ndjson), every output format, and every Python source (path, `LazyFrame`, `DataFrame`, list, generator — with `keep_columns`, a UDF plan, and the failure paths) to the numbers `ModelBank` gives on the same rows, CSV's flattened columns and JSON-text `coef` decoded bit-exact. A format is bound in the tests, never in the API. |
| 7. Cross-platform state | **Defined but never executed on real runners.** The test file and the macOS→Windows/Linux artifact hand-off exist in `release.yml`, but no workflow run has happened (no remote/tag yet). Locally only same-OS round-trip + byte-determinism are proven. |
| 8. Benchmark | Done (`scripts/benchmark.py`, numbers in README). |

**Do we compare edge cases against reference implementations?** Now yes, for
most of the surface. Clock semantics, null policy and warmup are cross-checked
against the numpy oracles along the `ewridge`, `rls` and `kalman` paths; the
lasso is checked against its own optimality conditions; and `tests/test_river.py`
compares FTRL, EW moments, quantile and Huber behavior against river. Still
anchored only by property tests: huber/quantile and ftrl at the *numeric* level
(T-A3, T-A4).

Writing this document found two live defects (T-E1, T-E2, both since fixed),
and the river work found two convention differences worth knowing about — river
predicts FTRL one proximal step behind the paper, and its `EWMean` is
un-normalized where ours is bias-corrected. Neither is a bug in either library,
but code that assumes they agree during warmup is wrong.

## 2. Improvements

Priorities: **P1** = closes a PLAN promise or covers a found defect; **P2** =
meaningful new assurance; **P3** = infrastructure.

### A. Close the oracle gaps (our own references)

| # | P | Improvement |
|---|---|---|
| T-A1 | ~~P1~~ **done** | **`kalman_ref` in `tests/reference.py`** — plain numpy predict/update recursion mirroring the standardization-from-prior-stats scheme; agreement to 1e-9 (observed 1.1e-15) incl. per-factor halflife, `inf` pinning, explicit `q`, `obs_var`/`p0`, `share_p`, no-intercept, and the null-target path. Writing it confirmed several subtleties are load-bearing: scales come from the stats *before* the row, `Q·Δclock` is applied once per shared `P`, and the innovation variance carries `σ²/w`. |
| T-A2 | ~~P1~~ **done** | **Lasso KKT verification** — at the emitted coefficient snapshot, stationarity is checked on the model's own standardized statistics (`g_i = c_i − (Cb)_i − l2·b_i` must equal `l1·sign(b_i)` where `b_i ≠ 0`, and satisfy `|g_i| ≤ l1` where it is zero), across λ × `l1_ratio` combinations, plus sparsity monotonicity along the path and the intercept identity. A numpy CD `lasso_ref` for the *pred* path is still open (would catch schedule/warm-start bugs the KKT check cannot see). |
| T-A3 | ~~P2~~ **done** | **`robust_ref`** covers both Huber and quantile; agreement ~1e-13. Two details proved load-bearing while writing it: the robust weight scales the accumulator update but **not** the `sigma2_j` update (otherwise the scale estimate shrinks itself), and a zero robust weight still decays the accumulator. |
| T-A4 | ~~P2~~ **done** | **`ftrl_ref`**; agreement ~1e-16. The decay is applied to `n` and `z` *before* the proximal weights are computed, so a row's prediction already reflects its own elapsed clock. |
| T-A5 | ~~P2~~ **done** | Null policy, warmup, clock semantics and the universal invariants are now parametrized over all seven models, with the genuinely model-specific deviations named in the module docstring rather than skipped silently (`rls` predict-only on any null target; `lasso` slot naming; `ftrl` probabilities). **Found the robust `n_eff` defect** described above. |

### B. Cross-checks against river (new `tests/test_river.py`)

Add river as a dev-dependency; skip the module cleanly when it is not
installed (same pattern as the offline skip). Two tiers:

**Exact (tolerance ~1e-12, config pinned so the algorithms coincide):**

| # | P | Comparison |
|---|---|---|
| T-R1 | ~~P1~~ **done** | **`ftrl` ≡ `river.optim.FTRLProximal`.** Compared at the level of the state recursion (river's optimizer driven by the same gradient sequence), which is exact to 1e-12; comparing the two *models* end to end is not exact, because river's `LogisticRegression` predicts with the previous step's proximal weights while we recompute from `z` at prediction time per McMahan Algorithm 1. That ordering difference is itself pinned by a test. |
| T-R2 | ~~P2~~ **done** | **Kalman(q=0, fixed `obs_var`, `standardize=False`) ≡ `river.linear_model.BayesianLinearRegression`** — exact to 3.6e-15 across three (alpha, beta) settings, mapping `p0 = 1/alpha` and `obs_var = 1/beta`. Includes a guard that turning standardization on breaks the match, so the switch cannot silently become a no-op. |
| T-R3 | ~~P2~~ **done** | **`EwCov` ≡ `river.stats.Mean` / `Var` / `Cov` / `PearsonCorr`** — exact agreement (1e-9) with no decay, reached via the new `ew_cov` surface (ENHANCEMENTS E1). The final test in that class also quantifies where the two diverge: at a 1e9 offset river's Welford form is still exact while our raw-moment form has lost the variance entirely, which is the gap E11b would close. |

**Statistical (tail agreement after warmup, tolerance stated per test):**

| # | P | Comparison |
|---|---|---|
| T-R4 | ~~P2~~ **done** | **EW mean/var vs `river.stats.EWMean` / `EWVar`**, mapping `fading_factor = 1 − 0.5^(1/halflife)`. Confirmed the conventions differ: ours divides by the accumulated weight (exact weighted mean from row 1), river's is `m += f·(x − m)` seeded at its first value. They converge in the limit (asserted) but disagree sharply during warmup (asserted, with the closed form). |
| T-R5 | ~~P3~~ **done** | **Quantile regression (intercept-only) vs `river.stats.Quantile`** (P² algorithm) at τ ∈ {0.25, 0.5, 0.75}: both land near the empirical quantile. |
| T-R6 | ~~P3~~ **done** | **Huber vs `river.linear_model.LinearRegression(loss=optim.losses.Huber)`** under 3% contamination: both beat least squares on the clean slope, and agree with each other. |

### C. Edge-case matrix to add

Findings first — both verified against the current build:

| # | P | Case | Current behavior → required |
|---|---|---|---|
| T-E1 | ~~P1~~ **done** | **Negative weight value** | **Was a defect**, now fixed. `EwCov::update` silently no-opped when `λW + w ≤ 0` while the per-target `r_j` update ran with a negative denominator — in practice every later prediction went null. Finite negative weights are now rejected at extraction, naming the column, value and row; non-finite ones skip the row like any other non-finite input; `w = 0` is a legal pure-decay row. `EwCov::update` also debug-asserts the contract. |
| T-E2 | ~~P1~~ **done** | **Null group key vs a group literally named `"<null>"`** | **Was a defect**, now fixed. Both mapped to the string `"<null>"` and shared one stream (verified: `n_eff` accumulated across them). Replaced by `GroupKey(Option<String>)`; bank files gained a `format_version` (now 2) and v1 files still load, since the key serializes transparently as its inner `Option`. |
| T-E3 | ~~P1~~ **done** | ±inf in features / targets / weight / clock | Pinned: a non-finite feature or weight skips the row (clock still advances), a non-finite target is predict-only, a non-finite or null clock errors loudly. Plus a fuzz-ish test that outputs are never non-finite. |
| T-E4 | ~~P1~~ **done** | Mis-ordered chunks (clock goes backwards across a chunk boundary within a group) | Both halves now covered: the absorbing policies are pinned as-is, and `on_clock_reset="error"` (ENHANCEMENTS E3, now implemented) catches a mis-sorted chunk boundary loudly. |
| T-E5 | ~~P2~~ **done** | Degenerate solves in the **plain** path | Exactly collinear features drive the jitter fallback (107 jittered solves over 200 rows) with finite outputs throughout; a real ridge removes the need for jitter entirely; non-solving models report 0. Required implementing ENHANCEMENTS E5 first: `Bank::solve_failures()` / `ModelBank.solve_failures()` now expose the count per spec and group. |
| T-E6 | ~~P2~~ **done** | Duplicate clock values, `max_dclock = 0`, `halflife` far below the typical Δ | Pinned: duplicate clock values are zero deltas (no decay); `max_dclock = 0` disables decay entirely; a halflife far below the delta makes every row effectively the first (`n_eff → 1`); no NaN leaks under extreme decay. |
| T-E7 | ~~P2~~ **done** | Minimal shapes | An empty chunk is accepted and returns an empty frame with the output column; an empty chunk *between* real chunks changes nothing; single-row groups, a group appearing in only one chunk, and a one-feature/one-target spec all behave. |
| T-E8 | ~~P2~~ **done** | Non-string group and session columns, null session values | Integer and categorical group columns, integer session columns. A null session value **is** its own session: `a → null` and `null → a` both count as changes, `null → null` does not. That was previously undocumented; it is now pinned. |
| T-E9 | ~~P2~~ **done; found a defect, then removed the limit** | **Large-offset cancellation** | First finding: the zero-variance drop threshold was `1e-10 × raw second moment`, ~450,000× the real noise floor, so a unit-variance feature on a **1e6 offset was silently dropped with coefficient 0** — an ordinary financial scale. Then the underlying cause was removed (ENHANCEMENTS E11b): `EwCov` keeps centered co-moments and the solves read them directly. Slope-recovery error at a 1e6 offset went **2.0e-03 → 6.8e-10**, and offsets of 1e8/1e10 went from dropped-entirely to 2.5e-08 / 4.8e-07. The tests now pin the new range and keep the old numbers in comments so a regression is obvious. |
| T-E10 | ~~P2~~ **done, decision taken** | Datetime-typed clock columns | Was a silent trap: a temporal clock cast to f64 exposes its internal representation, so the same 60 seconds is 60e3 / 60e6 / 60e9 units for `Datetime(ms/us/ns)` and 1 unit per day for `Date` — meaning `halflife=600` on a microsecond column silently meant 600 µs, decaying every row to nothing and producing plausible-looking garbage with no error. **Decision: reject.** Temporal clock columns now error, naming the column, its dtype and the fix (`pl.col("ts").dt.epoch("s")`). Numeric clocks (int and float) are unchanged and are asserted to agree with each other. |
| T-E11 | ~~P3~~ **done** | Long-stream soak: 10⁷ rows through one state | `n_eff` bounded and non-drifting end-to-end, coefficients still accurate, 2M-row state under 4KB, resume still exact. Opt-in (`pytest -m soak`), ~6.5s. |
| T-E12 | ~~P3~~ **done** | Pending-delta across a save/load boundary; session change on a group's first row | Both targeted now: splitting a stream exactly after a skipped row and resuming from state reproduces the unbroken run, and a group's first row is treated as first even when it also changes session. |

### D. Windows and cross-platform

Windows is a **stated deployment target** (CLAUDE.md: "Dev on macOS (arm64);
deploy on macOS and Windows"), and it is the least-verified part of the project:
no CI job has ever executed there. Running the workflows is the prerequisite,
but "run CI" is not itself a test plan — these are the specific things Windows
can break that macOS never will.

| # | P | Case | Why it can differ on Windows |
|---|---|---|---|
| T-W1 | **P1** | **Run `ci.yml` on `windows-latest` at all** | `cargo test --workspace`, `maturin develop`, and all 134 pytest functions have never executed on Windows. Everything below is speculative until this runs once. |
| T-W2 | **P1** | **Cross-OS state hand-off** (`release.yml`: write on macOS, load on Windows/Linux) | PLAN §9 class 7 and hard rule 5. The msgpack payload has no host-dependent parts *by construction*, and `save_bytes` is asserted deterministic locally — but that is an argument, not a test. |
| T-W3 | P1 (partly) | **Path handling through the CLI** — escaped Windows-style paths and paths with spaces are now tested through the CLI on any OS; actual resolution on Windows still needs a runner. |
| T-W3b | ~~P1~~ **found a real bug, fixed** | The first Windows CI run ever attempted failed exactly here: three `online-cli` tests died on `toml::de::Error` — "too few unicode value digits" — because the test interpolated `C:\Users\runner\...` straight into a TOML *basic* string, where `\U` begins a unicode escape. TOML was right and the caller was wrong, which makes it the same trap any Windows user hand-writing a config falls into, with an error message that says nothing about paths. Fixed in three places: the test escapes properly; the CLI's parse error now names all three valid forms (literal string, doubled backslashes, forward slashes) whenever the config contains a backslash; and the README documents it. Two new tests run **on every OS**, because the mistake is about the config text rather than the host — `test_an_unescaped_windows_path_is_rejected_with_a_usable_hint` and a parametrized check that each recommended form actually parses and runs. Those would have caught this on Linux, before a Windows runner existed. Drive letters and UNC *resolution* still need a real Windows runner. |
| T-W3b (original) | P1 | **Path handling through the CLI**: backslash separators, drive letters, UNC paths, and spaces in paths, in both the TOML `input`/`output`/`load_state`/`save_state` fields and the `--input`/`--output` overrides | `PlRefPath::try_from_pathbuf` normalizes Windows paths (polars has explicit `normalize_windows_path` logic); TOML string escaping means `"C:\data\x.parquet"` needs doubling or a literal string. Neither is exercised. |
| T-W4 | ~~P2~~ **mitigated + tested locally** | **CRLF line endings in the TOML config** — a `.gitattributes` now normalizes to LF on checkout, and the CLI is tested against both CRLF and LF configs. | The repo has no `.gitattributes`, so git may check out configs with CRLF on Windows. `toml` handles it, but the example config and any doc snippets should be proven to parse as checked out. |
| T-W5 | ~~P2~~ **tested locally** | **Binary/artifact naming**: `online.exe` vs `online` — `tests/test_release_packaging.py` stages what `download-artifact` leaves behind (one directory per matrix job, two of them holding a file called plain `online`) and runs the workflow's own bash against it. The shell is *extracted from `release.yml`*, not copied, so editing the workflow changes what the test runs — verified by breaking the workflow and watching three tests fail. Pinned: exactly the Windows artifact keeps `.exe`, the two unix binaries do not collapse onto one name, contents are copied not just renamed, and the PyPI job's `dist/` collects wheels only. What still needs a runner is whether the Windows job produces `online.exe` in the first place. |
| T-W6 | ~~P2~~ **pinned locally** | **Float formatting in output field names** — the exact 52-field name list for a grid spec is asserted, so a platform divergence fails loudly. | Combo labels are built with `format!("{r}")` on f64 (e.g. `pred_y__r0.000001`). Rust's float `Display` is locale-independent, so this *should* be identical everywhere — but the field names are part of the public schema and a divergence would silently break `expression ≡ bank`. Assert the exact field-name list on every OS. |
| T-W7 | P2 (**mechanism built**) | **Numeric reproducibility across OS/CPU** — `tests/test_golden_pipeline.py` commits 135 outputs (126 until IMPROVEMENTS X2 added `ftrl`, the one model it had never pinned) from a fixed stream through an eleven-spec bank at three rows each, and compares with a 1e-12 relative tolerance. `golden.rs` already pinned the Rust core; this pins the *Polars* layer — extraction, per-group fan-out, diagnostics, struct assembly — which nothing did. Locally the agreement is exact; the tolerance is what "the same answer on another platform" is allowed to mean, since different LLVM vectorization and BLAS paths can reorder floating-point operations while a genuinely divergent algorithm shows up far above it. Verified to bite at 1e-11 and to tolerate 1e-14. **The comparison becomes a cross-platform one the first time CI runs**, with no further work. |
| T-W8 | P3 | **Filesystem behavior**: case-insensitivity, `MAX_PATH`, file locking on rewrite | The bank writes state with `std::fs::write` and the runner opens the output parquet with `File::create`; a still-open reader on Windows makes rewriting fail where POSIX allows it. Relevant to `--resume` loops. |
| T-W9 | ~~P3~~ **done** | **`scripts/env.sh` has no Windows equivalent** — added `scripts/env.ps1`, dot-sourced (`. .\scripts\env.ps1`). Writing it surfaced two real Windows gaps in `.vscode/settings.json` that the shell script alone would not have: the Windows `PATH` entry omitted uv's `%USERPROFILE%\.local\bin`, and `rust-analyzer.cargo.extraEnv` hardcoded `PYO3_PYTHON` to `.venv/bin/python`, which does not exist on Windows (`.venv\Scripts\python.exe`) — and that key cannot be made OS-specific the way `terminal.integrated.env.*` can. Fixed by setting `VIRTUAL_ENV` instead: `pyo3-build-config`'s `get_env_interpreter` resolves it with `venv_interpreter(dir, cfg!(windows))`, so one value is correct on all three platforms. |

### E. Infrastructure

| # | P | Improvement |
|---|---|---|
| T-D1 | P1 **blocked** | **Actually run the workflows once.** Until then T-W1/T-W2 and the wheel builds are untested claims. *Blocked on GitHub credentials on this machine*: no keychain entry for github.com, no SSH key, no `GH_TOKEN`, and `gh` is not installed, so `git push` cannot authenticate. Unblock with any of `gh auth login` / an SSH key / a PAT — note the token needs the **`workflow` scope**, since this push adds `.github/workflows/`. |
| T-D2 | ~~P2~~ **done** | **Property-based testing** (hypothesis) over generated adversarial streams, asserting for all ten models: chunk invariance under any chunk size, save/load transparency at any split, outputs finite-or-null, skipped rows report no `n_eff`, group independence, and that a row's own target never influences its own prediction. A Rust-side `proptest` pass on `online-core` remains possible but is largely redundant now. |
| T-D3 | ~~P2~~ **done** | Determinism across parallelism: the bank is run in subprocesses at `RAYON_NUM_THREADS=1` and `=8` over six groups, and the outputs must be identical. |
| T-D4 | ~~P3~~ **done** | Coverage: `scripts/coverage.sh` reports 96% Python, 75%/73% Rust (caveat above); CI reports the Python figure non-gating. **Mutation testing** (`scripts/mutants.sh`) has now been run in full over `online-core`: **1645 mutants, 517 missed / 1104 caught / 24 unviable**. The misses concentrated exactly where the Rust unit tests lean on the *Python* oracle suite, which `cargo test` cannot see — `robust.rs` 68% missed, `kalman.rs` 38%, `ewridge.rs` 36%. Fixed with `crates/online-core/tests/golden.rs`: one fixed 60-row stream per model with the exact expected predictions embedded, which pins the arithmetic against any mutation. Measured effect on the worst file: **`robust.rs` went from 162 missed / 77 caught to 42 / 197**, a 74% reduction from one test. The residue is mostly accessors (`n_features -> 0`) and validation-branch comparisons, which are low value. Still open: re-running the full pass to get the new headline number, and making it periodic in CI. **2026-09-02:** `golden.rs` had signatures for seven of the eleven kinds; `sgd`, `pa`, `holt` and `ew_cov` (the four with longhand-recursion oracles in their own modules rather than numpy references) now have one each, `sgd` two, and `test_model_registry::test_the_core_golden_file_pins_every_model` holds the file to `KINDS`. Each new pin was mutated once (the Huber clamp, the PA-II damping, the trend smoothing, the variance floor) and each caught its own. The same day, `test_every_builder_has_a_per_model_test_file` gave EXTENDING step 13 its check, and `ewridge`/`rls` their own `test_<model>.py` out of `test_bank.py`. |

## FFI memory and crash safety (2026-08-31)

Two copies of Polars live in this process and data crosses on the Arrow C Data
Interface, where a `SeriesExport` carries a `release` callback back into the
binary that produced it. Nothing else in the suite would notice if that
contract were broken: a leak is invisible and a double-free is a crash that
takes pytest with it. `tests/test_ffi_memory.py` (16 tests, ~5 s) covers it.

**The assertion is "plateaus", not "does not grow."** Allocators do not return
pages eagerly, rayon spawns workers lazily, and Polars caches the loaded
plugin. Measured: our `.over()` costs a one-time **+6 MB** and is then flat
across 1,800 iterations (per-block deltas +0.7, −0.9, +1.1, −0.3, −0.3),
whereas native Polars `.over()` slowly *returns* memory. A naive "RSS must not
grow" test would have failed on that step forever. So `assert_plateaus`
discards the first block and compares the rest against each other: a step
passes, a slope fails.

Covered: repeated `fit_predict`; bank churn; the expression plugin; `.over()`
across groups; multi-chunk inputs (a chunked Series exports one `ArrowArray`
per chunk, each needing release); sliced frames that share their parent's
buffers; **both error paths**; outputs outliving their inputs; and state
round-trips.

Crash cases run in a **subprocess with `faulthandler`**, so a segfault is a
failed assertion with a native traceback instead of a dead test session: GC of
frames mid-flight, repeated reference/dereference with refcount checks,
reference cycles holding a bank, empty/single-row/all-null frames, both FFI
paths interleaved while each holds the other's exports, and pickle round-trips.

`scripts/leakcheck.sh` goes further than RSS can — `leaks` on macOS walks the
malloc zones for unreachable blocks, valgrind on Linux — with two limits that
took a deliberate leak each to find (2026-09-03). It sees Python objects only
under `PYTHONMALLOC=malloc`: pymalloc's arenas are mmap'd, and against a
128 MB refcount leak the default allocator reported "0 leaks", which is what
the earlier "clean" result here was. It cannot see anything allocated through
polars' allocator — every Rust-side allocation — so that side stays with
`test_ffi_memory.py`. The count is differential (1 iteration against 1000;
the interpreter leaves ~11k blocks unreachable at exit regardless) and the
script has a control mode that leaks one object per iteration and must fail.
Weekly in CI (`leakcheck.yml`), both platforms, control included. Not in
`gate.sh`: it needs a live process and valgrind is ~50x slower. Run it after
touching `crates/online-py` or the extraction path.

Two findings worth recording, neither a leak:

- The error paths are clean. The sharper one is the plugin: the engine has
  already exported the inputs when our function returns an error, so releasing
  them happens on a path that only runs when something has gone wrong.
- A **String feature column is silently parsed back to f64**, and a
  non-numeric one becomes all-nulls rather than raising. Consistent with the
  null policy and not a memory issue, but it means `pytest.raises` on a String
  column does not fire — a Categorical is the dtype that is genuinely refused.

## What is left

The blocker this section used to describe — *nothing has ever been pushed, no
CI job has ever run on any platform* — is gone. As of 2026-08-31 the repo is
pushed, and CI has run on all three platforms.

**Cleared:**

- **T-D1** — the workflows have run. What they claimed is now measured, and
  three of the claims were wrong: see `docs/RELEASE-READINESS.md` for the
  Linux `ld` SIGBUS (disk, not memory), the cache that never saved, and the
  disk exhaustion *inside* the cache restore.
- **T-W1** — `cargo test`, `maturin develop` and pytest have all executed on
  Windows: **712 passed, 1 failed** on `d6158aa`, down from 9 failures. Every
  one of the nine was a test bug, not a library bug — cp1252 encoding,
  `str(WindowsPath)` backslashes, a hardcoded POSIX `PATH`, a bash-only test.
- **T-W7** — the 126 committed golden pipeline outputs compared at 1e-12 on
  Windows, so the golden comparison is genuinely cross-platform now.
- **T-W5**, **T-W3b**, **T-W8**, **T-W2** — all executed as part of that run.

**The one open item, and it is a real one:**

- The tenth Windows failure — a `UnicodeEncodeError` in
  `examples/pathway_integration.py` — was fixed and pinned by a test, but
  **the fix has never run on Windows.** The cost policy took Windows off push
  (see the COST POLICY comment in `ci.yml`), so the next Windows result comes
  from the Monday schedule, or from a manual `workflow_dispatch` — which also
  pulls in macOS at 10x, so the schedule is much the cheaper way to find out.
  Until then, treat Windows as *green-except-one-known-fix-unverified*.

- **PyPI** — the name is still free (both spellings, re-checked 2026-08-31).
  What remains is registering this repository and workflow as a trusted
  publisher, which needs the account, and deciding the `polars==1.44.1` pin
  question recorded in `docs/RELEASE-READINESS.md`.

Two things are worth doing periodically rather than once:

- **`./scripts/mutants.sh`** after a batch of feature work. The first two runs
  are described in §0; the pattern to watch for is a cluster of survivors in
  one function, which almost always means its only oracle lives in the Python
  suite.
- **`./scripts/coverage.sh`**, reported and never gating.
