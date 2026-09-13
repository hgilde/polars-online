# Review 2026-09-12: progress

Working status of `docs/REVIEW-2026-09-12.md` (passes 1–9) and
`docs/REVIEW-2026-09-12-pass10.md` (pass 10: C24, S31, S32, D10, V24,
V25). The review documents themselves are left as the reviewer wrote them;
this file says what has been done about each finding.

**The rule for this round** (the user, 2026-09-13): take every finding in
turn; where the fix can be tested against an **independent library**,
implement the fix and that test first; otherwise record the finding here
as kept for later. Tests that are not against a library may be added too,
but library-checked fixes come first. Iterate until every fix that has an
independent test is done and tested.

**Independent libraries available**: `numpy` 2.5.2, `scipy` 1.18.1,
`river` 0.26.1 (all in the `dev` group). The review's oracles also name
`sklearn`, `statsmodels`, `pandas`, `filterpy` and
`bayesian_changepoint_detection`; none is installed. `sklearn` must not
become a dependency (a standing project rule) and "no pandas" is a style
rule, so where the review names either, the test uses the `numpy` or
`scipy` computation that gives the same number exactly (a weighted
least-squares solve, a two-pass weighted covariance) — or the finding waits,
marked *library not installed*. Adding `statsmodels`, `filterpy` or
`bayesian_changepoint_detection` as optional dev dependencies is the
user's decision; the findings that need one are listed at the end.

The library tests live in `tests/test_second_opinion.py`, following
`tests/test_river.py`'s two tiers: **exact** where the two implement the
same computation, **statistical** where they differ by design.

## Status

Legend: **fixed** (commit) · **next** (library test available, queued) ·
**later** (reason given).

### C — changes a number or a state

| ID | Finding | Independent test | Status |
|---|---|---|---|
| C1 | `ewridge` standardized solve reads live accumulators under a `window` | `numpy.linalg.lstsq`, in-window rows at `sqrt(lam^age)` weights | **fixed** — reproduced first: only `standardize=True` with a `window` was wrong (0.55 against `numpy`; the other three combinations within 2.6e-10); the branch now reads the `cov` it is handed; `TestWindowedFit` holds all four combinations to `numpy` at `1e-8` |
| C2 | `view()` falls back to the live state when the window is empty (`ewridge`, `lasso`) | `numpy.linalg.lstsq` on one target's in-window rows | **fixed** — both views are three-way (live / truncated / empty) and per target: a target with no row in the window reports NaN while the others stay windowed. One more cause the review named and the test found: the empty window's weight `W − f·W_u` comes out at `±1e-16·W`, not 0, so a positive crumb passed as "rows in it"; `window::EMPTY_FRACTION` (1e-12 of the weight subtracted from) now calls that empty, in `truncated`, `truncated_mean` and `marginal::cut`. Reproduced: on the old build a target that left the window sent the other target to the whole history (both models failed `numpy`) |
| C3 | a `session_shrink` blend never re-solves | `numpy` weighted least squares at `long_halflife` weights (the slow twin at `f = 1`) | next |
| C4 | building a `predict` plan drains the closed-group queue | none — a side effect, not a number | later: no library oracle |
| C5 | `label_delay` ignores a reset or session change on a skipped row | `numpy`: lag-1 pairs within sessions; fresh-bank equality | next |
| C6 | `ridge_decay` + `session_shrink` restores `prior_scale` at every blend | our own `rls` only | later: no library oracle |
| C7 | a zero-weight row poisons the lasso's lambda selection (0/0) | a hand-rolled selection sum | later: no library oracle |
| C8 | lasso without an intercept solves a centred/uncentred hybrid | `numpy.linalg.lstsq` with no intercept, at zero penalty | next |
| C9 | `Lasso::predict` reports the unwindowed `n_eff` | `numpy`: `Σ lam^age` over the in-window rows | **fixed** — reports and gates the window's weight; failed `numpy` on the old build |
| C10 | `kalman` centres without an intercept | `numpy.linalg.lstsq` with no intercept (statistical) + `coef·x == pred` | next |
| C11 | `robust` centres without an intercept | `numpy.linalg.lstsq` with no intercept, `huber_delta` large | next |
| C12 | `ew_class` `full` factorizes the live covariance under a `window` | `scipy.stats.multivariate_normal` on in-window class moments | **fixed** — `class_matrix` takes the accumulator the row is scored on; `full`, `diagonal` and `shared` are all held to `scipy` at `1e-9` (only `full` with a window failed on the old build) |
| C13 | `sgd` centres without an intercept | `numpy.linalg.lstsq` with no intercept (statistical) + `coef·x == pred` | next |
| C14 | `ew_cov` `mahal` and PCA read the live accumulator under a `window` | `scipy.spatial.distance.mahalanobis`, `numpy.linalg.eigh` | **fixed** — `mahal` and the PCA refresh read the view, and the refresh gate the window's weight; `mahal` and `pc0_var` held to `scipy`/`numpy` at `1e-9` (failed on the old build with a window, passed without) |
| C15 | `ew_cov` `mahal_quantiles` slot arithmetic ignores `LagCorr` | `numpy.quantile` of the emitted `mahal` (statistical) | **fixed** — one `EwCovCfg::width`, read by `n_outputs` and by the slot search; the P² quantiles held to `numpy.quantile` of the emitted `mahal` column within 10% (failed beside `lagcorr` on the old build, passed with `mahal` alone) |
| C16 | the `session_shrink` blends re-centre through the raw second moment | `numpy.cov` / `numpy.average` with the blended row weights, offset `1e8` | next |
| C17 | `window::truncated` and `marginal::cut` re-centre through the raw second moment | `numpy.cov` / `numpy.average` on in-window rows, offset `1e8` | **fixed** — reproduced first (windowed `var`/`cov` off by 390% at `1e8`, the unwindowed accumulator within 2.6e-9); centred pooling identity in both, diagonal floored (closes V3); `TestWindowAtALargeOffset` failed 3 of 10 on the old build, the three at `1e8` with a window, and passes 10 of 10 now; two Rust offset tests with two-pass oracles |
| C18 | `marginal` allows `window` with `lags`; the lag ring is never truncated | `statsmodels` `acf` | later: library not installed (the recommended fix, a refusal, has no library test) |
| C19 | `predict` scores across a session close `fit_predict` restarts at | `fit_predict` itself | later: no library oracle |
| C20 | TOML `bocpd`/`hmm` read the hazard/exogenous column from `targets` | the builder | later: no library oracle |
| C21 | under `label_delay`, residual diagnostics fold the replay-time residual | `river.evaluate.progressive_val_score(delay=...)` | next |
| C22 | `holt` does not advance the level across a null or zero-weight row | `statsmodels` `ExponentialSmoothing` | later: library not installed |
| C23 | `bocpd`'s `Run` forms every scatter by subtraction | `bayesian_changepoint_detection` | later: library not installed (an exact shift-invariance test needs none — a strong candidate for the non-library round) |
| C24 | `ftrl`'s halflife shrinks the coefficients | none exists (river only at `inf`, the control) | later: no library oracle, and a design question |

### S — two places that disagree

| ID | Finding | Independent test | Status |
|---|---|---|---|
| S1 | bank `sigma`/`resid_z` not windowed | `numpy` over in-window residuals | later: needs a decision (route the model's `sigma2`, or correct two docstrings) |
| S2 | per-target `min_periods` gates on the shared `n_eff` | — | later: needs a decision |
| S3 | skipped-row time folded without the cap | `numpy` `lam^d` once decided | later: needs a decision (60 or 600) |
| S4 | two model states share the kind `"ew_cov"` | — | later: no library oracle |
| S5 | a single-combo `ew_ridge` loses its combo metadata | — | later: no library oracle |
| S6 | `window.rs` disagrees with itself on the boundary row | — | later: no library oracle |
| S7 | `Spec::validate` gaps | `numpy.linalg.lstsq` for one sub-case; the fix is refusals | later: no library oracle for the fix |
| S8 | `solve_failures` doc vs list | — | later: documentation |
| S9 | `kalman` zero-weight row skips the per-target decay | `filterpy` | later: library not installed |
| S10 | `rls` never predicts from `coef_prior` before its first row | — | later: no library oracle |
| S11 | `Lasso::solve_failures` never written | — | later: no library oracle |
| S12 | `atomic.rs` Windows and durability caveats | — | later: no library oracle |
| S13 | `robust` asymmetries in `step` and `solve` | — | later: no library oracle |
| S14 | `ew_class` reports the whole-history `n_eff` under a `window` | `numpy`: `Σ lam^age` in window | **fixed** — `n_eff()` is the window's weight (one subtraction from the snapshot), read by the gate, the report and `step`; failed `numpy` on the old build |
| S15 | `EwCovModel::n_eff()` is the live weight under a `window` | `numpy`: `Σ lam^age` in window | **fixed** — the accessor reads the view; the emitted field was already windowed and is held to `numpy`, and a Rust assertion holds the accessor to it |
| S16 | an `sgd` state without its `scaler` loads unscaled | — | later: no library oracle |
| S17 | `marginal` emits the live `n_eff`, accessor windowed | `numpy`: `Σ lam^age` in window | **fixed** — emits `n_eff()`; failed `numpy` on the old build |
| S18 | `marginal` `"truncated"` serial rule floors at `MIN_POSITIVE` | `statsmodels` `cov_hac` | later: library not installed |
| S19 | the Gram and the closed row carry live accumulators under a `window` | `numpy.linalg.lstsq` on in-window vs all rows | next (after the `n_eff` rule) |
| S20 | PCA sign continuity keyed across groups under a session close | `numpy.linalg.eigh` per closed row | next |
| S21 | `ModelBank.specs` before vs after a round trip | — | later: no library oracle |
| S22 | `Spec::validate` lets pairs through (pattern F) | `statsmodels` `Holt` for one sub-case | later: no library oracle for the refusals |
| S23 | `emit_averaged` weights in target units² | `river` `EWARegressor` (shape only) | later: needs a decision (scale-free `eta`, or document the units) |
| S24 | the expression form packs no target for an unsupervised kind | — | later: no library oracle |
| S25 | six doors, three depths for spec checking | — | later: no library oracle |
| S26 | `coef_index` refusals by name vs `IndexError` | — | later: no library oracle |
| S27 | the layers' lists of what may be infinite disagree | — | later: needs a per-parameter decision |
| S28 | residual diagnostics disagree on a zero-weight row | `river.drift.PageHinkley`, `numpy.quantile` | next |
| S29 | `holt` reads the row weight as a gate | `pandas` `ewm` | later: library not installed |
| S30 | `holt` at an infinite level halflife | `statsmodels` `Holt` | later: library not installed |
| S31 | `ftrl` `strict_binary`: documented error, silent skip | — | later: needs a decision |
| S32 | `FtrlCfg::validate` accepts NaN | — | later: no library oracle |

### P — performance, D — documentation, V — to confirm

| ID | Status |
|---|---|
| P1–P5 | later: performance; benchmarks, not library tests |
| D1–D10 | later: documentation |
| V5, V7, V11, V22, V23, V24, V25 | later: to confirm; none is a fix yet |

## Log

- 2026-09-13 — review pulled (`9b3270a`); this file created with every
  finding classified. The user's own design work (task 78's windows,
  task 79) is parked on the local branch `design/task-78` until the
  review is finished.
- C17 fixed. Two things the review's text had slightly wrong, recorded so
  the next reader does not repeat them: a window keeps rows whose age is
  **at most** `window` (the module doc's rule; T-S10's "age < window" is
  off by the boundary row, and so is the Rust oracle `direct_window`, which
  passes only because its clock is irregular); and the report on row *i* is
  referenced at row *i*−1 on every path — `bank.predict` and `fit_predict`
  agree on it, so there is no new finding there.

- C2, C9, S14, S15, S17 fixed together. **A decision taken, for the user to
  see:** pattern D asked "hard rule 8 should say which" `n_eff` a windowed
  model reports. The answer taken is the window's weight, everywhere --
  emitted, gated on, and returned by the accessor -- because the fit is on
  the window, `ewridge` (the workhorse) already did so, and `ew_cov`'s
  `predict` doc had already chosen it. `CLAUDE.md` hard rule 8 now says so.
  `TestTheWindowIsAllTheModelSees` failed 5 of 12 on the old build (the three
  models that reported the whole history, and C2 in both), all 26 library
  tests pass now.

- C12, C14, C15 fixed. The new tests failed exactly 3 of 10 on the previous
  build -- `full` with a window, `ew_cov`'s `mahal`/PCA with a window, the
  quantiles beside `lagcorr` -- and their controls passed; 36 library tests
  pass now.

## New observations (found while fixing; not in the review)

- **N1** (while fixing C1): `ewridge` keeps its per-target cross-moments
  raw, `r_j = E_w[z·y_j]` (the module doc's recursion), so the standardized
  solve forms its right-hand side as `E[z·y] − m_z·ȳ` -- a subtraction of
  the kind pattern E is about, which the review does not list. It costs
  precision only when a feature *and* the target both sit at a large level
  (a price regressed on a price); a return target or a level feature alone
  is unaffected. Not fixed; kept for later with the review's pattern E.
  The C1 test keeps the target near zero for this reason.

## Waiting on a library the user may add

`statsmodels` (C18, C22, S18, S22's `holt` case, S30), `filterpy` (S9),
`bayesian_changepoint_detection` (C23), `pandas` (S29, barred by the style
rule — the same checks can be written with `numpy`).
