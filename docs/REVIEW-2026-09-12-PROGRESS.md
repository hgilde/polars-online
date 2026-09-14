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

**Independent libraries available**: `numpy` 2.5.2, `scipy` 1.18.1 and
`river` 0.26.1, and since 2026-09-13 `statsmodels` 0.15.0, `filterpy` 1.4.5
and `bayesian-changepoint-detection` 0.2.dev1, added on the user's go (all
in the `dev` group; a test imports each of the three with
`pytest.importorskip`). `sklearn` must not become a dependency (a standing
project rule), so where the review names it the test uses the `numpy` or
`scipy` computation that gives the same number exactly. `pandas` may be
used in tests (the user, 2026-09-13: "no pandas" is a style rule for the
package); `statsmodels` brings it, and no test here has needed it yet.

The library tests live in `tests/test_second_opinion.py`, following
`tests/test_river.py`'s two tiers: **exact** where the two implement the
same computation, **statistical** where they differ by design.

## Summary (2026-09-13, end of the library round)

**Every finding whose fix has an independent library test is fixed and
tested: 25 of the review's, and one found on the way (N1).** C1, C2, C3,
C5, C8–C17, C21, C22, C23; S9, S14, S15, S17, S18, S19, S20, S28; N1. (C5
has no library oracle of its own; it went in with C21, whose fix depends
on it.) The library tests are `tests/test_second_opinion.py`, against
`numpy`, `scipy`, `river`, `statsmodels`, `filterpy` and
`bayesian-changepoint-detection`.

| commit | findings |
|---|---|
| `860e905` | C17, C1 |
| `12416b3` | C2, C9, S14, S15, S17 |
| `eced1fd` | C12, C14, C15 |
| `23a5ba2` | C16, C3 |
| `d3312bc` | C8, C10, C11, C13 |
| `97237c2` | S28 |
| `cb6c57c` | C21, C5 |
| `fd2f8d6` | S20 |
| `b56f561` | S19 |
| (this commit) | N1, C22, C23, S9, S18; `SCHEMA_VERSION` 7 |

**How each was held.** Every finding was reproduced before it was fixed
(the review was written without running anything), and every new test
was run against the build before its fix and failed there -- except S20's
first two versions, which passed on the old code and so tested nothing;
they were replaced. The full gate ran on every batch.

**Tests that pinned a defect, and were changed with it.**
`tests/reference.py`'s `kalman_ref` had copied C10's centring, and S9's
skipped decay;
`test_label_delay.py` asserted S28's disagreement with the doubled stream
as the oracle's quirk, and later asserted an agreement on the diagnostics
that existed only because both sides folded C21's peeking residual; one
Rust test (`blend_before_any_data_is_a_no_op`) caught a first version of
C3's re-solve; and the `OnlineModel` doc example in `model.rs` asserted
C22's forgotten clock -- `holt`'s forecast across two unlearned rows as
`59 + 3·1` -- and now gives `59 + 5·1`, with the reason.

**Decisions I took, for the user to review.**

1. **Pattern D:** under a `window`, `n_eff` is the window's weight --
   emitted, gated on and returned by the accessor, in every model.
   `CLAUDE.md` hard rule 8 now says so.
2. **C21's record rides on schema 6 without a bump**, as task 38's fields
   rode on 3: skipped when empty, so no file without a `label_delay`
   moves. `lib.rs` records it.
3. **S19:** under a window, the Gram's target moments are `None` rather
   than the whole history's.
4. **C13:** `sgd`'s `coef·x == pred` holds to a scaler step (about 1%),
   not to rounding, with or without an intercept; its tests assert that.
5. **C2:** a window's remainder below `1e-12` of the weight it was
   subtracted from is empty (`window::EMPTY_FRACTION`).
6. **Schema 7**, for N1, C23 and C22, with loaders for schema 6 as hard
   rule 5 asks: raw moments are split into centred ones as a file is read,
   which keeps the numbers the file carried and not the bits, and a
   schema-6 `holt` starts every target's clock since its last observation
   at 0 -- a file saved just after a null forgets that one gap.

**Where the review's own text was wrong** (found by testing it): a window
keeps rows whose age is *at most* `window` (T-S10 said "less than"); S20's
example, two reflected clouds, never flips -- a component has to cross the
other group's; C5's `numpy` count is unreachable, since `group_close`
refuses `label_delay`; C13's exact contract does not exist for `sgd`;
T-S11's `cov_hac` on the product series reproduces the truncated serial
factor only in expectation, since the product's sample autocovariances
are not the products of the two series' own; T-S15's `rtol 1e-9` at
`1e8` is finer than the data, which is resolved to `1.5e-8` there (the
two agree to `3e-9`); and C22's two `holt` tests do not turn over -- with
a clock since the last observation the level does stand still across a
null, and what moves is the next row's forecast.

**Kept for later: 53 items**, each with its reason in the tables below --
no independent library oracle (C4, C6, C7, C19, C20, C24; S4-S8, S10-S13,
S16, S21, S22, S24-S26, S32), a fix with no library test whose
library-tested alternative is an enhancement (C18), a decision needed
first (S1, S2, S3, S23, S27, S29, S30, S31), performance (P1-P5),
documentation (D1-D10), and the V list -- and three new observations,
N2-N4. **S29 and S30 are the user's call**: both would trade `holt`'s
textbook recursion, which `statsmodels`' `Holt` now pins exactly, for a
mean-form level. The user's own design work (tasks 78 and 79), parked on
`design/task-78` while the round ran, is merged into `main`.

## Status

Legend: **fixed** (commit) · **next** (library test available, queued) ·
**later** (reason given).

### C — changes a number or a state

| ID | Finding | Independent test | Status |
|---|---|---|---|
| C1 | `ewridge` standardized solve reads live accumulators under a `window` | `numpy.linalg.lstsq`, in-window rows at `sqrt(lam^age)` weights | **fixed** — reproduced first: only `standardize=True` with a `window` was wrong (0.55 against `numpy`; the other three combinations within 2.6e-10); the branch now reads the `cov` it is handed; `TestWindowedFit` holds all four combinations to `numpy` at `1e-8` |
| C2 | `view()` falls back to the live state when the window is empty (`ewridge`, `lasso`) | `numpy.linalg.lstsq` on one target's in-window rows | **fixed** — both views are three-way (live / truncated / empty) and per target: a target with no row in the window reports NaN while the others stay windowed. One more cause the review named and the test found: the empty window's weight `W − f·W_u` comes out at `±1e-16·W`, not 0, so a positive crumb passed as "rows in it"; `window::EMPTY_FRACTION` (1e-12 of the weight subtracted from) now calls that empty, in `truncated`, `truncated_mean` and `marginal::cut`. Reproduced: on the old build a target that left the window sent the other target to the whole history (both models failed `numpy`) |
| C3 | a `session_shrink` blend never re-solves | `numpy` weighted least squares at `long_halflife` weights (the slow twin at `f = 1`) | **fixed** — the blend re-solves when it mixed anything and a fit exists, which also fixes `predict`'s blended copy. Reproduced: on the old build the first row of session 2 was predicted at 1.20 with the pre-blend fit, against `numpy`'s slow-twin fit of −0.12; now equal at `1e-8`, and `bank.predict` on that row equals `fit_predict` (E31). A blend with nothing to mix stays a no-op, re-solve included (`blend_before_any_data_is_a_no_op` caught a first version that re-solved anyway) |
| C4 | building a `predict` plan drains the closed-group queue | none — a side effect, not a number | later: no library oracle |
| C5 | `label_delay` ignores a reset or session change on a skipped row | `numpy`: lag-1 pairs within sessions; fresh-bank equality | **fixed**, with C21, which needs it (the queue C21 keeps stays aligned with the waiting rows only if a reset on a skipped row clears them) — a skipped row's reset now clears the buffer, and its session change or capped gap releases it, as an accepted row's always did. No library oracle: the review's `numpy` count needs a closed group, and `group_close` refuses `label_delay`. The test is the definition: after a reset on a skipped row, `pred` and `n_eff` equal a fresh bank fed the new session alone (failed on the old build) |
| C6 | `ridge_decay` + `session_shrink` restores `prior_scale` at every blend | our own `rls` only | later: no library oracle |
| C7 | a zero-weight row poisons the lasso's lambda selection (0/0) | a hand-rolled selection sum | later: no library oracle |
| C8 | lasso without an intercept solves a centred/uncentred hybrid | `numpy.linalg.lstsq` with no intercept, at zero penalty | **fixed** — without an intercept `standardized()` scales by the raw second moment and keeps the raw cross-moments; at zero penalty the fit is held to `numpy`'s no-intercept least squares at `1e-6` (coordinate descent's tolerance), with the intercept case as the control |
| C9 | `Lasso::predict` reports the unwindowed `n_eff` | `numpy`: `Σ lam^age` over the in-window rows | **fixed** — reports and gates the window's weight; failed `numpy` on the old build |
| C10 | `kalman` centres without an intercept | `numpy.linalg.lstsq` with no intercept (statistical) + `coef·x == pred` | **fixed** — `scales_into`, `standardized` and the step's own standardization scale by `sqrt(E[x²])` and do not centre; the exact contract `dot(coef_(i−1), x_i) == pred_i` holds to `1e-9` (it missed by 16 on the old build: the hidden intercept), and with no process noise the coefficients sit at `numpy`'s no-intercept fit to `1e-2` after 5000 rows (statistical). The full gate then failed `tests/test_oracles.py::TestKalmanOracle::test_no_intercept`: the from-scratch oracle `tests/reference.py::kalman_ref` had copied the centring, so it pinned the defect -- the situation the review warns of for C16. The oracle now standardizes as the fixed core does |
| C11 | `robust` centres without an intercept | `numpy.linalg.lstsq` with no intercept, `huber_delta` large | **fixed** — the standardized solve without an intercept is the raw system scaled by `sqrt(E[x²])`; `huber` at a delta no residual reaches is held to `numpy`'s no-intercept least squares at `1e-8` with `standardize` on (failed on the old build) and off (the control) |
| C12 | `ew_class` `full` factorizes the live covariance under a `window` | `scipy.stats.multivariate_normal` on in-window class moments | **fixed** — `class_matrix` takes the accumulator the row is scored on; `full`, `diagonal` and `shared` are all held to `scipy` at `1e-9` (only `full` with a window failed on the old build) |
| C13 | `sgd` centres without an intercept | `numpy.linalg.lstsq` with no intercept (statistical) + `coef·x == pred` | **fixed** — `standardized` and `scales` use the raw second moment and do not centre without an intercept. The review's exact `coef·x == pred` does not hold for `sgd` even with an intercept -- measured 8.7e-3 relative, since the step standardizes with the row admitted and `coef` is reported through the scaler after the row before (the review's own D5) -- so the test asserts that few-percent tier (6.7e3 on the old build), and the library check is statistical: the slopes settle at `numpy`'s no-intercept fit within `5e-2` after 20 000 rows |
| C14 | `ew_cov` `mahal` and PCA read the live accumulator under a `window` | `scipy.spatial.distance.mahalanobis`, `numpy.linalg.eigh` | **fixed** — `mahal` and the PCA refresh read the view, and the refresh gate the window's weight; `mahal` and `pc0_var` held to `scipy`/`numpy` at `1e-9` (failed on the old build with a window, passed without) |
| C15 | `ew_cov` `mahal_quantiles` slot arithmetic ignores `LagCorr` | `numpy.quantile` of the emitted `mahal` (statistical) | **fixed** — one `EwCovCfg::width`, read by `n_outputs` and by the slot search; the P² quantiles held to `numpy.quantile` of the emitted `mahal` column within 10% (failed beside `lagcorr` on the old build, passed with `mahal` alone) |
| C16 | the `session_shrink` blends re-centre through the raw second moment | `numpy.cov` / `numpy.average` with the blended row weights, offset `1e8` | **fixed** — both blends (`EwRidge::blend_toward_long_run`, `TargetMoments::blend`) use the centred mixture `a·C_f + b·C_s + a·b·ΔΔᵀ`; `bank.gram()` after a blend is held to `numpy` with the blended weights `(1−f)·λ_h^age + f·λ_H^age` — means, co-moments, target mean and variance — at `1e8` (failed on the old build) and `0` (the control) |
| C17 | `window::truncated` and `marginal::cut` re-centre through the raw second moment | `numpy.cov` / `numpy.average` on in-window rows, offset `1e8` | **fixed** — reproduced first (windowed `var`/`cov` off by 390% at `1e8`, the unwindowed accumulator within 2.6e-9); centred pooling identity in both, diagonal floored (closes V3); `TestWindowAtALargeOffset` failed 3 of 10 on the old build, the three at `1e8` with a window, and passes 10 of 10 now; two Rust offset tests with two-pass oracles |
| C18 | `marginal` allows `window` with `lags`; the lag ring is never truncated | `statsmodels` `acf` | later: `statsmodels` is installed now, but the recommended fix -- refusing the pair, as `ew_cov` does -- has no library test; what `acf` can test is a lag ring with snapshots, an enhancement, and only at T-S11's statistical tier (`0.02`) |
| C19 | `predict` scores across a session close `fit_predict` restarts at | `fit_predict` itself | later: no library oracle |
| C20 | TOML `bocpd`/`hmm` read the hazard/exogenous column from `targets` | the builder | later: no library oracle |
| C21 | under `label_delay`, residual diagnostics fold the replay-time residual | `river.evaluate.progressive_val_score(delay=...)` | **fixed** — the replay folds the prediction the row was *scored* with: each instance keeps a queue of score-time predictions in step with the waiting rows (pushed as a buffered row is scored, taken back at its replay, dropped with them on a reset), saved with the stream and skipped when empty. **A decision for the user:** that field rides on schema 6 without a bump, as task 38's rode on 3 -- no file without a delay moves, the frozen schema-6 fixture re-saves unchanged, and an older build reading a new file just folds as it always did; `lib.rs` records it. Library, exact: river's `iter_progressive_val_score(delay=10)` on a running-mean regressor gives every prediction the frame carries (`1e-12`), and `sigma[t]²` is the mean of the emitted residuals over the rows matured by row `t` (`1e-9`; 0.305 against 0.389 on the old build). A save and reload mid-delay continues identically. The two `embargo` comparisons in `test_label_delay.py` became one: the doubled stream matches on `pred`, `resid` and `n_eff` to the bit and parts, by design, on every residual diagnostic (V21); E47's row says which residual is folded |
| C22 | `holt` does not advance the level across a null or zero-weight row | `statsmodels` `ExponentialSmoothing` | **fixed** — each target keeps its clock since its last observation: a null or zero-weight row adds its delta, and the next observed row extrapolates, and forms both rates, over the whole gap, so the row is transparent. Library, exact: `statsmodels`' `Holt` reproduces our recursion to `1e-12` on a stream with no gaps (T-S14, the control that pins the mapping), and its state-space `ExponentialSmoothing` with a `NaN` in the series agrees with ours to `1e-9` on every row up to the one after the gap, which forecasts `l + 2b` (T-S17; the trend gain there is `α·β`, as the review said). Through the bank, a null or zero-weight row every third row gives what the stream gives with those rows removed, to `1e-12`. The gap and transparency tests failed on the old build; the control passed. `golden.rs` and `test_golden_pipeline.py` are re-frozen for `holt`: both streams have null targets, and only rows after one moved |
| C23 | `bocpd`'s `Run` forms every scatter by subtraction | `bayesian_changepoint_detection` | **fixed** — each run keeps a weighted Welford mean and centred scatter, and `Ψₙ = Ψ₀ + S + (κ₀n/κₙ)(x̄ − μ₀)(x̄ − μ₀)ᵀ` subtracts nothing. Library, exact: the package's run-length posterior gives our `p_change`, `run_mode` and `run_mean` at offsets 0, `1e6` and `1e8`, the prior centred with the data -- `p_change` to `3.5e-15`, `5e-11` and `3e-9`, where the old build failed at `1e6` and `1e8`. Rust: Algorithm 1 longhand at `1e8` with a two-pass oracle (the old build missed it by `0.011`), and `gaussian` at `d = 3` gives at `1e8` what it gives at the origin, with no solve failure. A schema-6 run converts as it is read (`RunWire`) |
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
| S9 | `kalman` zero-weight row skips the per-target decay | `filterpy` | **fixed** — a zero-weight row with its target present decays `wj` and `wsig` as a null row does. Library, exact: `filterpy`'s `KalmanFilter` beside a `numpy` recursion for `σ²` -- `predict(Q)` on every row, `update(y, R = σ²/w, H = z)` only where there is a target and a positive weight, `σ²`'s weight decayed on every row -- gives our predictions to `1e-9` (Joseph form against the simple form). The zero-weight case failed on the old build and the null case, the control, passed. `tests/reference.py`'s `kalman_ref` had copied the skip; it decays now |
| S10 | `rls` never predicts from `coef_prior` before its first row | — | later: no library oracle |
| S11 | `Lasso::solve_failures` never written | — | later: no library oracle |
| S12 | `atomic.rs` Windows and durability caveats | — | later: no library oracle |
| S13 | `robust` asymmetries in `step` and `solve` | — | later: no library oracle |
| S14 | `ew_class` reports the whole-history `n_eff` under a `window` | `numpy`: `Σ lam^age` in window | **fixed** — `n_eff()` is the window's weight (one subtraction from the snapshot), read by the gate, the report and `step`; failed `numpy` on the old build |
| S15 | `EwCovModel::n_eff()` is the live weight under a `window` | `numpy`: `Σ lam^age` in window | **fixed** — the accessor reads the view; the emitted field was already windowed and is held to `numpy`, and a Rust assertion holds the accessor to it |
| S16 | an `sgd` state without its `scaler` loads unscaled | — | later: no library oracle |
| S17 | `marginal` emits the live `n_eff`, accessor windowed | `numpy`: `Σ lam^age` in window | **fixed** — emits `n_eff()`; failed `numpy` on the old build |
| S18 | `marginal` `"truncated"` serial rule floors at `MIN_POSITIVE` | `statsmodels` `cov_hac` | **fixed** — a truncated factor at or below zero is NaN, null in `bank.marginal()`, as `"geometric"` answers a factor it cannot form; it was floored at `MIN_POSITIVE`, and `n_serial` came out `inf`. Library: `statsmodels`' `acf` gives this pair's own lag-1 autocorrelations, about +0.8 and −0.8, which put the factor below zero, and ours agree with `acf` to T-S11's `0.02`; two positively autocorrelated series are the control, where `n_serial` is `n_kish` over the factor to `1e-12`. The review's exact `cov_hac` check does not hold (see the summary). Bartlett weights, positive by construction, would be a new option; kept for later |
| S19 | the Gram and the closed row carry live accumulators under a `window` | `numpy.linalg.lstsq` on in-window vs all rows | **fixed** — `gram_of` reads the window's accumulators (`windowed_gram` on `ewridge` and `lasso`, `windowed_cov` on `ew_cov`), so `bank.gram()` and a closed row's Gram are the window's, and `po.gram.solve` on them is the fit `coef` reports. Library, exact: `numpy.linalg.lstsq` on the rows the last fit was read from equals both `bank.coef()` and `po.gram.solve(bank.gram())` at `1e-6`, and the Gram's `n_eff` is their weight; on the old build the solve gave the whole history's fit, and `window=None` was the control. **A decision for the user:** the window's snapshots do not carry the target moments, so under a window `target_means`, `target_vars` and `target_n_kish` are `None` -- the Gram's existing way of saying "this state cannot say" -- rather than the whole history's; snapshotting them would restore them, kept for later |
| S20 | PCA sign continuity keyed across groups under a session close | `numpy.linalg.eigh` per closed row | **fixed** — the continuity map is keyed by (spec, group, instance) for a spec that closes on session, (spec, instance) as before under `"monotone"`; the bank file keeps the old list for the latter and a new one, skipped when empty, for the former; `_bank.py`'s docstring now states both rules. Library, exact: each closed row's `eig_vecs` and `eig_vals` equal `numpy.linalg.eigh` on that row's own co-moments, sign-aligned with the same group's previous close. **The review's example does not show the defect**, which the first two versions of the test found by passing on the old code: two clouds that are reflections of each other, or any fixed pair of directions, stay consistent, because keyed by (spec, instance) every close is chained to the previous one whatever its group, and a fixed geometry chains consistently. The flip needs a group whose component moves across the other's between closes; the test's B alternates either side of `x1` beside an A along `x0`, and on the old code B's second close came out flipped. Shown by stashing the fix and rebuilding |
| S21 | `ModelBank.specs` before vs after a round trip | — | later: no library oracle |
| S22 | `Spec::validate` lets pairs through (pattern F) | `statsmodels` `Holt` for one sub-case | later: no library oracle for the refusals. `statsmodels`' `Holt` can now say which halflife `holt`'s level ran at, but the fix for that pair is a refusal too |
| S23 | `emit_averaged` weights in target units² | `river` `EWARegressor` (shape only) | later: needs a decision (scale-free `eta`, or document the units) |
| S24 | the expression form packs no target for an unsupervised kind | — | later: no library oracle |
| S25 | six doors, three depths for spec checking | — | later: no library oracle |
| S26 | `coef_index` refusals by name vs `IndexError` | — | later: no library oracle |
| S27 | the layers' lists of what may be infinite disagree | — | later: needs a per-parameter decision |
| S28 | residual diagnostics disagree on a zero-weight row | `river.drift.PageHinkley`, `numpy.quantile` | **fixed** — a zero-weight row's residual stays out of `resid_quantiles`, the autocorrelation, the drift detector and `ew_cov`'s `mahal_q`, as it always did of `sigma`. Library, exact: river's `PageHinkley(alpha=1, mode="up")` reproduces ours flag for flag (checked on its own first, at three settings), and fed the bank's own `\|resid\|/sigma` series without the zero-weight row it reproduces the bank's drift column. No library needed for the rest: a zero-weight row's target, jumped a hundredfold, now changes no field on any later row, under `drift_action` `"flag"` and `"reset"`. On the old build all three failed (a quantile moved; under `"reset"` the row restarted the model; the drift column parted from river's at the row). The full gate then failed `tests/test_label_delay.py`'s `test_the_weight_free_diagnostics_are_where_the_two_differ`, which pinned S28 itself -- it asserted the native path and `embargo`'s doubled stream disagree on these three, attributing the gap to the oracle's zero-weight rows (the review's V21 notes exactly this). It now asserts they agree |
| S29 | `holt` reads the row weight as a gate | `pandas` `ewm` | later: needs a decision. `pandas` may be used in tests now, but the review's fix is a mean-form level, `l' = (lam·W·p + w·y)/(lam·W + w)`, in place of the textbook Holt recursion the model runs -- which `statsmodels`' `Holt` now pins exactly (T-S14). Which of the two `holt` is, is the user's call |
| S30 | `holt` at an infinite level halflife | `statsmodels` `Holt` | later: needs the same decision as S29. At an infinite halflife the textbook rate is 0 and the level freezes at the first row; the mean form goes to the cumulative mean. `lam = 1` refused in `level_halflife`'s name is a message to reword with it |
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

- C16, C3 fixed. The new tests failed 2 of 3 on the previous build (C16 at
  `1e8`, C3), the offset-0 control passed; 39 library tests pass now.

- C8, C10, C11, C13 (pattern B) fixed. The new tests failed 4 of 5 on the
  previous build -- every no-intercept case -- and the `huber` control
  without standardization passed; 44 library tests pass now.

- S28 fixed. `numpy.quantile` turned out to be unnecessary: the exact
  "changes nothing after it" test covers the quantiles, and river is exact
  for the detector. 47 library tests pass.

- C21 and C5 fixed together. The new tests failed 2 of 3 on the previous
  build (C21's `sigma`, C5's fresh-bank equality; the save/reload test has
  nothing to find on code without the new state). On the new build the
  expected casualty was `test_label_delay.py`'s two `embargo` comparisons,
  which asserted agreement on the diagnostics -- agreement that existed
  only because both sides folded the peeking residual.

- S20 fixed. Its first two tests passed on the old code -- the review's
  reflection example and a fixed 120-degree rotation both chain
  consistently -- so they tested nothing; the third, with a component that
  crosses the other group's between closes, fails on the old code (shown
  by stashing the fix and rebuilding) and passes on the new. A reminder
  for every finding: a test that passes before the fix proves nothing
  about it.

- The library batch: `statsmodels`, `filterpy` and
  `bayesian-changepoint-detection` joined the `dev` group on the user's
  go, and N1, C22, C23, S9 and S18 are fixed. Every new test ran against
  the previous build first, and failed where it should: the level tests at
  `1e4`, `1e6` and `1e8` (offset 0 passed); the gap and transparency tests
  for `holt`, while `statsmodels`' `Holt` matched the old recursion
  exactly (the control); the changepoint package at `1e6` and `1e8`;
  `filterpy` with a zero weight only; the serial factor, as `inf`. All seven
  new Rust tests failed there too. `SCHEMA_VERSION` is 7, for the three
  layouts that moved, and each loads from schema 6: `state_schema6.rs`'s
  `bocpd` now continues to `1e-12` and its other specs still to the bit,
  and a new frozen file from the last schema-6 build,
  `state_schema6_ridge.rs`, carries `ewridge`'s three places -- the live
  accumulators, a `session_shrink` twin, a window's snapshots -- a pending
  block, and `holt`. That file found a bug in the first loader, which
  split the window's snapshots about a target mean of 0; the unit tests
  could not have, since they only ever load what the current build
  writes. The user said on the way that `pandas` is fine in tests.

- `design/task-78` merged into `main`, at the user's request. It is
  documentation only (task 78's design, task 79, a README correction), and
  `docs/PLAN.md` conflicted where both sides added tasks at the head of the
  list; all three are kept. Task 79 is ticked there: it is this review's
  C5, fixed with C21 in `cb6c57c`.

## New observations (found while fixing; not in the review)

- **N1** (while fixing C1) -- **fixed** in the library batch. `ewridge`
  kept its per-target cross-moments raw, `r_j = E_w[z·y_j]`, so the
  standardized solve formed its right-hand side as `E[z·y] − m·ȳ`, a
  subtraction of the kind pattern E is about; and the plain solve read the
  raw normal equations, which lost the fit the same way (the prediction at
  `1e8` was off by 78 standardized and 70 plain). Each target now keeps the
  target's mean, `E[(z − m_z)(y − ȳ)]`, and the offset of `z`'s mean over
  its rows from the all-row mean -- a number of its own, exactly 0 for a
  target present on every row (`Cross` in `ewridge.rs`) -- and both solves
  with an intercept read the centred system, the plain one exactly
  equivalent to what it solved before. Library, exact: `numpy.linalg.lstsq`
  on rows centred at their weighted means, worst prediction error `7e-15`,
  `2.4e-11`, `1.4e-9` and `1.5e-7` at offsets 0, `1e4`, `1e6` and `1e8` --
  the data's own resolution -- where the old build gave `1.2e-7` at `1e4`
  and `9e-4` at `1e6`. Rust: a level costs the fit nothing, blocked and not;
  the offset is exactly 0 for a target present on every row. The raw
  moments are still what `bank.gram()` exports (formed from the centred
  ones), and `ridge_decay` and the solves through the origin still read
  the raw system, which has no intercept to eliminate.
- **N2** (while fixing N1): `lasso` keeps its per-target cross-moments raw
  in the same way and recovers its intercept from them; the N1 test at a
  level would show it. Kept for later.
- **N3** (while fixing N1): with a target that is null on some rows, the
  Gram is over every row and the target's cross-moments over its own, so
  the slopes depend on the target's level: in exact arithmetic the solve
  adds `(m_j − m)·ȳ_j / Var(x)` to a slope, `m_j` the feature's mean over
  the target's rows. A property of the model's definition, plain and
  standardized alike, not of its numerics, and unchanged here; whether a
  target's fit should read a Gram over its own rows is a question for the
  user.
- **N4** (while fixing N1): `po.gram.solve` follows the old `EwRidge::solve`
  on the raw cross-moments the Gram exports -- the raw normal equations,
  or a centring by subtraction -- so a Gram saved at a level loses the fit
  the way the model did before N1. The export is raw by contract; the
  solve could centre from `means`, `comoments` and the target moments.
  Kept for later.

## Libraries

Nothing waits on a library now. `statsmodels`, `filterpy` and
`bayesian-changepoint-detection` joined the `dev` group on 2026-09-13, and
`pandas` may be used in tests (the user, the same day). What each finding
still waits on is in the tables.
