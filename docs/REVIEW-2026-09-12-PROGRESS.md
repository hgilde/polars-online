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
package); `statsmodels` brings it, and the N3 tests use its
`DataFrame.cov`.

The library tests live in `tests/test_second_opinion.py`, following
`tests/test_river.py`'s two tiers: **exact** where the two implement the
same computation, **statistical** where they differ by design.

## Summary (2026-09-13, end of the library round)

**Every finding whose fix has an independent library test is fixed and
tested: 25 of the review's, and four found on the way (N1, N2, N3, N5).**
C1, C2, C3, C5, C8–C17, C21, C22, C23; S9, S14, S15, S17, S18, S19, S20,
S28; N1, N2, N3, N5. (C5 has no library oracle of its own; it went in with
C21, whose fix depends on it. N3 was the user's decision: `target_gaps`,
docs/PLAN.md task 81.) The library tests are
`tests/test_second_opinion.py`, against `numpy`, `scipy`, `river`,
`statsmodels`, `filterpy`, `bayesian-changepoint-detection` and `pandas`.

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
| `ea99161` | N1, C22, C23, S9, S18; `SCHEMA_VERSION` 7 |
| (this commit) | N3 as task 81 (`target_gaps`), N2, N5; `SCHEMA_VERSION` 8 |

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
7. **Task 81 (`target_gaps`, N3):** `n_eff` counts every row under both
   readings, so `min_periods` gates as before and S2 stays open; a closed
   group writes one row per Gram, whose `n_eff` is that Gram's weight, as
   its `n_kish` was. Schema 8 has no loader for 7, on the user's word
   (2026-09-14: "Do not worry about state saved before the next version
   release, we are pre 1.0 and we can change things now"), so the loaders
   of item 6 are gone with their fixtures.

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

**Kept for later: 4 items** (V22 closed with task 81; batch 1 below
closed fourteen, batch 2 thirteen, batch 3a eleven, batch 3b two, batch 4a
four, batch 4b four), each with its reason in the tables below -- a
decision taken by the user on 2026-09-15 and recorded in `docs/PLAN.md`
task 80 (S1 and S27, for batch 4c), documentation (D10, begun in 4a), and
one new observation, N4, which task 81 did half of. D1 is excluded by the
user.
The user's own
design work (tasks 78 and 79), parked on `design/task-78` while the round
ran, is merged into `main`.

**Batch 1 (2026-09-15): the core findings no library can check**, on the
user's rule of 2026-09-14 -- test first; re-derive each finding from the
code before writing its test; a test that passes where the review said it
would fail is recorded here and raised, not "fixed". Fixed: C6, C7, S4,
S6, S11, S13, S16, S32, D2-D6, and N6, found on the way; S10 is not a
defect. **Raised to the user:** S10's premise is false (EW-ridge does not
predict from the prior on row 0 either); the review's own test for S13's
second bullet passes on the old build (the jitter ladder rescues its
example), though the defect is real and another test shows it; S6 was
a behavioural bug, not only three wrong comments; D6's lost digits were
the test oracle's ridge on the intercept, not the window; and C6's
proposed test could not show the defect.

## Status

Legend: **fixed** (commit) · **next** (library test available, queued) ·
**later** (reason given).

### C — changes a number or a state

| ID | Finding | Independent test | Status |
|---|---|---|---|
| C1 | `ewridge` standardized solve reads live accumulators under a `window` | `numpy.linalg.lstsq`, in-window rows at `sqrt(lam^age)` weights | **fixed** — reproduced first: only `standardize=True` with a `window` was wrong (0.55 against `numpy`; the other three combinations within 2.6e-10); the branch now reads the `cov` it is handed; `TestWindowedFit` holds all four combinations to `numpy` at `1e-8` |
| C2 | `view()` falls back to the live state when the window is empty (`ewridge`, `lasso`) | `numpy.linalg.lstsq` on one target's in-window rows | **fixed** — both views are three-way (live / truncated / empty) and per target: a target with no row in the window reports NaN while the others stay windowed. One more cause the review named and the test found: the empty window's weight `W − f·W_u` comes out at `±1e-16·W`, not 0, so a positive crumb passed as "rows in it"; `window::EMPTY_FRACTION` (1e-12 of the weight subtracted from) now calls that empty, in `truncated`, `truncated_mean` and `marginal::cut`. Reproduced: on the old build a target that left the window sent the other target to the whole history (both models failed `numpy`) |
| C3 | a `session_shrink` blend never re-solves | `numpy` weighted least squares at `long_halflife` weights (the slow twin at `f = 1`) | **fixed** — the blend re-solves when it mixed anything and a fit exists, which also fixes `predict`'s blended copy. Reproduced: on the old build the first row of session 2 was predicted at 1.20 with the pre-blend fit, against `numpy`'s slow-twin fit of −0.12; now equal at `1e-8`, and `bank.predict` on that row equals `fit_predict` (E31). A blend with nothing to mix stays a no-op, re-solve included (`blend_before_any_data_is_a_no_op` caught a first version that re-solved anyway) |
| C4 | building a `predict` plan drains the closed-group queue | none — a side effect, not a number | **fixed** (batch 3a) — the plan reads its closed-row schema with `closed_groups(drop=False).clear()`, so building `lf.online.predict(bank)` leaves the caller's queue alone, and `DataFrame.online.fit_predict` drains once instead of relying on the order Python evaluates two calls in. On the old build building the plan took the queue from 2 rows to 0 |
| C5 | `label_delay` ignores a reset or session change on a skipped row | `numpy`: lag-1 pairs within sessions; fresh-bank equality | **fixed**, with C21, which needs it (the queue C21 keeps stays aligned with the waiting rows only if a reset on a skipped row clears them) — a skipped row's reset now clears the buffer, and its session change or capped gap releases it, as an accepted row's always did. No library oracle: the review's `numpy` count needs a closed group, and `group_close` refuses `label_delay`. The test is the definition: after a reset on a skipped row, `pred` and `n_eff` equal a fresh bank fed the new session alone (failed on the old build) |
| C6 | `ridge_decay` + `session_shrink` restores `prior_scale` at every blend | our own `rls` only | **fixed** (batch 1) — the blend rebuilt the Gram from `EwCov::new`, so after a blend at `f = 0.3` `prior_scale` came back 1 against the mixture's 0.195 (failed on the old build). **A decision taken:** the prior mixes as the sum-scale weights do, `(1 − f)·ps_fast + f·ps_twin` -- the review's second option, since keeping the fast side's leaves `f = 1` short of the twin. At `f = 1` the blended fit is RLS's at the twin's halflife to `1e-9` (our own `rls` as the second opinion, as the review proposed). **The review's test could not show it** (raised): at `f = 0` the blend returns before it mixes anything |
| C7 | a zero-weight row poisons the lasso's lambda selection (0/0) | a hand-rolled selection sum | **fixed** (batch 1) — a zero-weight row adds no error and ages the rest; bit-identical to before wherever the old form was not `0/0`. On the old build `sel_err` was NaN from row 1 and the selection stuck at the heaviest penalty. The contract (`model_contract.rs`) now puts a zero-weight row on every model's first prediction and checks that no prediction goes NaN; that passed on the old build for every model, `lasso` included, since the selection's symptom is a stuck argmin, not a NaN -- so the `lasso` test is where C7 is held |
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
| C18 | `marginal` allows `window` with `lags`; the lag ring is never truncated | `statsmodels` `acf` | **fixed** (batch 2) — `MarginalCfg::validate` refuses `window` with `lags`, as `ew_cov` does, and every door reaches the refusal (S25). The Rust validation test and the builder both took the pair on the old build. A lag ring with snapshots stays an enhancement (V12) |
| C19 | `predict` scores across a session close `fit_predict` restarts at | `fit_predict` itself | **fixed** (batch 3a) — under `group_close = "session"` a new session's row is scored by a fresh stream, as `fit_predict` restarts at the change: one condition in `predict_chunk`, which already scored a `session_gap = "reset"` row that way. On the old build the new session was scored with the closed one's fit (`n_eff` 17.0 where a fresh stream has 0); now `predict` and `fit_predict` agree on its first row, null with `n_eff` 0 |
| C20 | TOML `bocpd`/`hmm` read the hazard/exogenous column from `targets` | the builder | **fixed** (batch 2) — `fill_defaults` fills `targets` from a named `hazard_col` or `exog_tvtp`, and `validate` refuses a spec whose `targets` is not that column. On the old build a `bocpd` dict without `targets` read `features[0]` as the hazard: its `p_change` parted from the builder's spec at row 1 (`1.8e-9` against `0.165`), and both Rust tests failed |
| C21 | under `label_delay`, residual diagnostics fold the replay-time residual | `river.evaluate.progressive_val_score(delay=...)` | **fixed** — the replay folds the prediction the row was *scored* with: each instance keeps a queue of score-time predictions in step with the waiting rows (pushed as a buffered row is scored, taken back at its replay, dropped with them on a reset), saved with the stream and skipped when empty. **A decision for the user:** that field rides on schema 6 without a bump, as task 38's rode on 3 -- no file without a delay moves, the frozen schema-6 fixture re-saves unchanged, and an older build reading a new file just folds as it always did; `lib.rs` records it. Library, exact: river's `iter_progressive_val_score(delay=10)` on a running-mean regressor gives every prediction the frame carries (`1e-12`), and `sigma[t]²` is the mean of the emitted residuals over the rows matured by row `t` (`1e-9`; 0.305 against 0.389 on the old build). A save and reload mid-delay continues identically. The two `embargo` comparisons in `test_label_delay.py` became one: the doubled stream matches on `pred`, `resid` and `n_eff` to the bit and parts, by design, on every residual diagnostic (V21); E47's row says which residual is folded |
| C22 | `holt` does not advance the level across a null or zero-weight row | `statsmodels` `ExponentialSmoothing` | **fixed** — each target keeps its clock since its last observation: a null or zero-weight row adds its delta, and the next observed row extrapolates, and forms both rates, over the whole gap, so the row is transparent. Library, exact: `statsmodels`' `Holt` reproduces our recursion to `1e-12` on a stream with no gaps (T-S14, the control that pins the mapping), and its state-space `ExponentialSmoothing` with a `NaN` in the series agrees with ours to `1e-9` on every row up to the one after the gap, which forecasts `l + 2b` (T-S17; the trend gain there is `α·β`, as the review said). Through the bank, a null or zero-weight row every third row gives what the stream gives with those rows removed, to `1e-12`. The gap and transparency tests failed on the old build; the control passed. `golden.rs` and `test_golden_pipeline.py` are re-frozen for `holt`: both streams have null targets, and only rows after one moved |
| C23 | `bocpd`'s `Run` forms every scatter by subtraction | `bayesian_changepoint_detection` | **fixed** — each run keeps a weighted Welford mean and centred scatter, and `Ψₙ = Ψ₀ + S + (κ₀n/κₙ)(x̄ − μ₀)(x̄ − μ₀)ᵀ` subtracts nothing. Library, exact: the package's run-length posterior gives our `p_change`, `run_mode` and `run_mean` at offsets 0, `1e6` and `1e8`, the prior centred with the data -- `p_change` to `3.5e-15`, `5e-11` and `3e-9`, where the old build failed at `1e6` and `1e8`. Rust: Algorithm 1 longhand at `1e8` with a two-pass oracle (the old build missed it by `0.011`), and `gaussian` at `d = 3` gives at `1e8` what it gives at the origin, with no solve failure. A schema-6 run converts as it is read (`RunWire`) |
| C24 | `ftrl`'s halflife shrinks the coefficients | none exists (river only at `inf`, the control) | **fixed** (batch 4a), on the user's decision (river plus a repaired decay), and **short of the review's numbers** (raised). Under a halflife the proximal term is a decayed sum of its own; without one the weight is river's closed form, computed as river computes it, so T-R1 is unchanged and a Rust test holds it bit for bit. A constant 5 at `halflife = 100` settles at 4.65 where it settled at 2.25 -- the review asked for 5.0 ± 1e-3 -- because the penalties stay constants on the sums' scale, a mean-scale ridge of `(1 − λ)(β/α + l2)`. A gap still scales the coefficients by `λ^t(d + c)/(λ^t·d + c)`: 0.75 over 100 clock units where it was 0.70 (the review asked for 1). A logistic base rate of 0.9 at `halflife = 1000` reads 0.873 where it read 0.816 (the review asked for 0.90 ± 0.01). All three need the penalties scaled by the weight, which is not river's at `inf`. The Rust tests pin the longhand and the stated factor; both failed on the old build. T-A4's reference carries the sum, and gained the squared loss (D10) |

### S — two places that disagree

| ID | Finding | Independent test | Status |
|---|---|---|---|
| S1 | bank `sigma`/`resid_z` not windowed | `numpy` over in-window residuals | later: needs a decision (route the model's `sigma2`, or correct two docstrings) |
| S2 | per-target `min_periods` gates on the shared `n_eff` | — | **fixed** (batch 4b), on the user's decision: each target's threshold is checked against its own weight -- the rows it was present on, inside the window under one -- through a new trait hook, `target_n_eff_into`, which `ewridge`, `lasso`, `kalman`, `robust` and `holt` override; the emitted `n_eff` stays the shared weight. The review's test, a target present on every tenth row under `min_periods = [5, 5]`, gave its first prediction at row 5 on the old build for all five models, and gives it at row 41, the row after its fifth observation |
| S3 | skipped-row time folded without the cap | `numpy` `lam^d` once decided | **fixed** (batch 4b), on the user's decision: the folded total is capped at `max_dclock` and marked a capped gap. The review's test, ten skipped rows 100 apart under a cap of 60, handed the next row 660 on the old build and hands it 60. Through the bank, the `n_eff` two rows on is `0.5^(50/10) + 1` as `numpy` has it, where the old build's 550 gave 1.0 |
| S4 | two model states share the kind `"ew_cov"` | — | **fixed** (batch 1) — the bare accumulator is `"ew_cov_accumulator"`, so `EwCovModel::restore` of one names what it found (failed on the old build); the kind is only ever an error message, never written to a file. Two tests pinned the old name (`kmeans.rs`, the contract's kind list) |
| S5 | a single-combo `ew_ridge` loses its combo metadata | — | **fixed** (batch 2) — every `ew_ridge` combo carries its ridge, and its feature set wherever `feature_sets` names one; a single combo's label alone stays empty. On the old build `output_index`'s `ridge` was null for a single-ridge spec, and a single named set was dropped from a ridge grid's metadata |
| S6 | `window.rs` disagrees with itself on the boundary row | — | **fixed** (batch 1), and **worse than the review says** (raised). The comments were wrong as it says -- the module doc is the true one, which the review's own test (a gap of twice the window at `every = 1`) pins, passing on the old build as it should -- and `trim`'s empty `if` and `new`'s `every.max(1)` are gone. What the review did not see: with `window_every` above 1, after a clock gap longer than the window, or wherever `every` rows span more clock than it, the newest snapshot was older than the window, `trim` kept it as the boundary, and rows the window excludes stayed in the fit -- at a cadence of 5 the row after a long gap reported a window weight of 2 where one row was inside, then 3 for 2. A row that finds the newest snapshot outside the window is now snapshotted whatever the cadence. Both new tests failed on the old build (`window.rs`: the boundary at 100 for a row at 200; `ewridge.rs`: the weight). `window_every = 1` is unchanged |
| S7 | `Spec::validate` gaps | `numpy.linalg.lstsq` for one sub-case; the fix is refusals | **fixed** (batch 2) — `validate` refuses by name a feature-set name given twice (it reached only the bank's field-name tripwire, whose comment said it could not happen), a column twice in one set, an empty set (the core now says "is empty", not "out-of-range indices"), and a spec named `""`, `"spec"` or `"group"` (`last_row` puts columns of those names beside the struct). Each built on the old build |
| S8 | `solve_failures` doc vs list | — | **fixed** (batch 2) — the doc says what each counting model counts, and that the rest report 0 because they count nothing, not because nothing can fail |
| S9 | `kalman` zero-weight row skips the per-target decay | `filterpy` | **fixed** — a zero-weight row with its target present decays `wj` and `wsig` as a null row does. Library, exact: `filterpy`'s `KalmanFilter` beside a `numpy` recursion for `σ²` -- `predict(Q)` on every row, `update(y, R = σ²/w, H = z)` only where there is a target and a positive weight, `σ²`'s weight decayed on every row -- gives our predictions to `1e-9` (Joseph form against the simple form). The zero-weight case failed on the old build and the null case, the control, passed. `tests/reference.py`'s `kalman_ref` had copied the skip; it decays now |
| S10 | `rls` never predicts from `coef_prior` before its first row | — | **not a defect** (raised) — the review's test passes on the old build. Its premise, that EW-ridge predicts `x · prior` on row 0, is false: EW-ridge has no fit before its first solve and gates each target on the target's own weight, so it too predicts nothing until a row is learned, and the two agree from row 0 on every value and on which rows have one (`agrees_with_ewridge_from_the_first_row_with_a_prior`, kept as the pin). No change to `predict`. The minor items: `RlsV2`, whose `try_from` could not fail, is gone -- `Rls` deserializes directly -- and `lib.rs`'s schema-2 note was already history |
| S11 | `Lasso::solve_failures` never written | — | **fixed** (batch 1) — one count per target and path point whose descent runs out of `max_cd_iters` before `cd_tol`; 0 on the old build with one sweep, and still 0 with enough sweeps |
| S12 | `atomic.rs` Windows and durability caveats | — | **fixed** (batch 3a), as documentation: the module doc says that on Windows a destination held open without `FILE_SHARE_DELETE` fails the rename, that the rename is not flushed to its directory, and that a kill leaves the temporary behind. No retry was added, and no Windows test pins the error's type: the project does not pin one platform's errno in a test |
| S13 | `robust` asymmetries in `step` and `solve` | — | **fixed** (batch 1). **The review's test for the second bullet passes on the old build** (raised): its collinear example is rescued by the jitter ladder, which counts the attempt, so `solve_failures > 0` either way. The defect is real -- a solve that fails at every jitter returned through `?` before the count -- and a correlation matrix every jitter fails on shows it (handed to the accumulator, since no stream of rows makes one): 0 on the old build, 1 now. The first bullet: a zero-weight row now ages `wsig` (failed on the old build, 14.65 against 11.90), held with N6 by the EW-mean oracle below |
| S14 | `ew_class` reports the whole-history `n_eff` under a `window` | `numpy`: `Σ lam^age` in window | **fixed** — `n_eff()` is the window's weight (one subtraction from the snapshot), read by the gate, the report and `step`; failed `numpy` on the old build |
| S15 | `EwCovModel::n_eff()` is the live weight under a `window` | `numpy`: `Σ lam^age` in window | **fixed** — the accessor reads the view; the emitted field was already windowed and is held to `numpy`, and a Rust assertion holds the accessor to it |
| S16 | an `sgd` state without its `scaler` loads unscaled | — | **fixed** (batch 1) — a state whose `scaler` disagrees with `scale_features`, or whose AdaGrad sums disagree with its schedule, is refused by name; all four cases loaded on the old build |
| S17 | `marginal` emits the live `n_eff`, accessor windowed | `numpy`: `Σ lam^age` in window | **fixed** — emits `n_eff()`; failed `numpy` on the old build |
| S18 | `marginal` `"truncated"` serial rule floors at `MIN_POSITIVE` | `statsmodels` `cov_hac` | **fixed** — a truncated factor at or below zero is NaN, null in `bank.marginal()`, as `"geometric"` answers a factor it cannot form; it was floored at `MIN_POSITIVE`, and `n_serial` came out `inf`. Library: `statsmodels`' `acf` gives this pair's own lag-1 autocorrelations, about +0.8 and −0.8, which put the factor below zero, and ours agree with `acf` to T-S11's `0.02`; two positively autocorrelated series are the control, where `n_serial` is `n_kish` over the factor to `1e-12`. The review's exact `cov_hac` check does not hold (see the summary). Bartlett weights, positive by construction, would be a new option; kept for later |
| S19 | the Gram and the closed row carry live accumulators under a `window` | `numpy.linalg.lstsq` on in-window vs all rows | **fixed** — `gram_of` reads the window's accumulators (`windowed_gram` on `ewridge` and `lasso`, `windowed_cov` on `ew_cov`), so `bank.gram()` and a closed row's Gram are the window's, and `po.gram.solve` on them is the fit `coef` reports. Library, exact: `numpy.linalg.lstsq` on the rows the last fit was read from equals both `bank.coef()` and `po.gram.solve(bank.gram())` at `1e-6`, and the Gram's `n_eff` is their weight; on the old build the solve gave the whole history's fit, and `window=None` was the control. **A decision for the user:** the window's snapshots do not carry the target moments, so under a window `target_means`, `target_vars` and `target_n_kish` are `None` -- the Gram's existing way of saying "this state cannot say" -- rather than the whole history's; snapshotting them would restore them, kept for later |
| S20 | PCA sign continuity keyed across groups under a session close | `numpy.linalg.eigh` per closed row | **fixed** — the continuity map is keyed by (spec, group, instance) for a spec that closes on session, (spec, instance) as before under `"monotone"`; the bank file keeps the old list for the latter and a new one, skipped when empty, for the former; `_bank.py`'s docstring now states both rules. Library, exact: each closed row's `eig_vecs` and `eig_vals` equal `numpy.linalg.eigh` on that row's own co-moments, sign-aligned with the same group's previous close. **The review's example does not show the defect**, which the first two versions of the test found by passing on the old code: two clouds that are reflections of each other, or any fixed pair of directions, stay consistent, because keyed by (spec, instance) every close is chained to the previous one whatever its group, and a fixed geometry chains consistently. The flip needs a group whose component moves across the other's between closes; the test's B alternates either side of `x1` beside an A along `x0`, and on the old code B's second close came out flipped. Shown by stashing the fix and rebuilding |
| S21 | `ModelBank.specs` before vs after a round trip | — | **fixed** (batch 2) — `ModelBank.specs` is read from the native side at construction, as a loaded bank's always was, so a hand-written dict reads the same before a round trip as after it; builder dicts are unchanged. It needed D8's `_numeric_keys`, without which a float of five builders would have come back the string `"inf"` |
| S22 | `Spec::validate` lets pairs through (pattern F) | `statsmodels` `Holt` for one sub-case | **fixed** (batch 2), one item documented rather than refused. Refused now, each built on the old build: `drift_action = "reset"`, `drift_delta` and `drift_threshold` without `emit_drift`; `average_eta` without `emit_averaged`; `resid_autocorr_lag` without `emit_autocorr`; `long_halflife` without `session_shrink`; `session_shrink` beside `session_gap = "reset"` or `group_close = "session"`; `session_gap` without `session`; `on_clock_reset` without `clock`; `holt`'s `halflife` or `lam` beside `level_halflife`; and `coef_every` on a model with no coefficients (held to the field names for every kind). `add_intercept` no longer moves the warm-up of `ew_cov`, `kmeans`, `micro`, `ew_class` and `holt`, which have no intercept: `k + 1`, the builders' default. **`kalman`'s `coef_halflife` beside `q` is documented, not refused**: `_spec.py` already says `q` overrides the derivation, `Kalman::q_into` does exactly that, and the round-trip tests use the pair, as the docs invite. The `window_every` and `sgd.quantile` items reach the core's checks through S25's one door |
| S23 | `emit_averaged` weights in target units² | `river` `EWARegressor` (shape only) | **fixed** (batch 4b), on the user's decision: each slot weighs `exp(−eta·(σ²/σ²_best − 1))`, and at a best of 0 the ratio's limit, the argmin. The review's scale test -- `y × 1e4` with `eta` unchanged gives `pred_averaged × 1e4` -- failed on every row of the old build and passes at `rtol 1e-9`, and the formula is held exactly to the bank's own `sigma` columns (1.84 against 1.41 on the old build). River's `EWARegressor`, which the review called shape-only, is not a test here: it weighs by cumulative loss and this by an EW mean, so the two part by design |
| S24 | the expression form packs no target for an unsupervised kind | — | **fixed** (batch 2) — the expression packs the column a model with no target reads from the targets slot (`ModelKind::targets_slot_column`: `hazard_col`, `exog_tvtp`), and refuses a feature expression named after the clock, session or weight column. On the old build both expression forms failed with "target column not found", and the aliased feature ran |
| S25 | six doors, three depths for spec checking | — | **fixed** (batch 2) — `Spec::check` is fill, validate and build, and every door calls it: `Bank::new`, `RunConfig::validate` (which validated before the bank could fill), `validate_spec`, `output_fields`, `output_index`, `coef_fields` and the expression plugin's `parse_spec`. On the old build the E53 dict was refused by the three index functions and by the runner, and a `window` + `ridge_decay` spec got field names from the index functions while the bank refused it |
| S26 | `coef_index` refusals by name vs `IndexError` | — | **fixed** (batch 2) — `coef_index` refuses every kind with no coefficients, naming it; `marginal`, `rcov`, `corrchange` and `bocpd` raised an `IndexError` on the old build. The docstrings list the six |
| S27 | the layers' lists of what may be infinite disagree | — | later: needs a per-parameter decision |
| S28 | residual diagnostics disagree on a zero-weight row | `river.drift.PageHinkley`, `numpy.quantile` | **fixed** — a zero-weight row's residual stays out of `resid_quantiles`, the autocorrelation, the drift detector and `ew_cov`'s `mahal_q`, as it always did of `sigma`. Library, exact: river's `PageHinkley(alpha=1, mode="up")` reproduces ours flag for flag (checked on its own first, at three settings), and fed the bank's own `\|resid\|/sigma` series without the zero-weight row it reproduces the bank's drift column. No library needed for the rest: a zero-weight row's target, jumped a hundredfold, now changes no field on any later row, under `drift_action` `"flag"` and `"reset"`. On the old build all three failed (a quantile moved; under `"reset"` the row restarted the model; the drift column parted from river's at the row). The full gate then failed `tests/test_label_delay.py`'s `test_the_weight_free_diagnostics_are_where_the_two_differ`, which pinned S28 itself -- it asserted the native path and `embargo`'s doubled stream disagree on these three, attributing the gap to the oracle's zero-weight rows (the review's V21 notes exactly this). It now asserts they agree |
| S29 | `holt` reads the row weight as a gate | `pandas` `ewm` | **fixed** (batch 4a), on the user's decision: level and trend are weighted means, `(λ·W·old + w·new)/(λ·W + w)`. The review's failing test -- the last 40 rows at weight 0.5 against 1 -- failed on the old build, the two levels the same to the last digit, and passes against a longhand of the weighted means. **Its exact check, a row at weight 2 is that row twice at no clock, passed on the old build** (raised): both sides ignored the weight, and a row at no clock changed nothing. It holds for the level and `n_eff`, not the trend, which reads a move over the clock. Library: statsmodels' `Holt` agrees exactly once the weights have saturated (T-S14 from row 1000, `rtol 1e-12`); the first rows part by design, which T-S14 now asserts and which failed on the old build. **Every `holt` stream's numbers move** (raised), weighted or not, since the gains start at 1; goldens re-frozen |
| S30 | `holt` at an infinite level halflife | `statsmodels` `Holt` | **fixed** (batch 4a), with S29: an infinite halflife fits the whole history -- on `y = 3 + 2t` the forecast for row 199 is 399, where the old build gave the first row's 3.0 -- and `lam = 1` builds, as `halflife = inf`, where it was refused naming `level_halflife`. **`trend_halflife = inf` is the whole history's drift now**, not a trend pinned at zero (raised: a plain level with no trend has no spelling left) |
| S31 | `ftrl` `strict_binary`: documented error, silent skip | — | **fixed** (batch 4a), on the user's decision: a `strict_binary` target other than 0 or 1 refuses the chunk at extraction, naming the row and the value, beside `ew_class`'s label check, so the bank is left as it was; on the old build the chunk ran. The model alone still skips such a row (a Rust caller's `step`), and `n_eff` still counts the rows an update-form model saw, as hard rule 8 is written |
| S32 | `FtrlCfg::validate` accepts NaN | — | **fixed** (batch 1) — NaN refused in all four, as `pa` refuses it; accepted on the old build |

### P — performance, D — documentation, V — to confirm

| ID | Status |
|---|---|
| P1, P3 | **fixed** (batch 3a) — `predict` and `n_eff` read the window's weights alone (`Acc::window_weights`, `EwCovModel::window_n_eff`), and `lasso`'s selection one target's errors (`window_sel_err`), instead of the O(k²) view, which is built now only to solve or to read statistics. Bit-identical: guard tests hold the cheap reads to the view on every row, through a clock gap that empties the window. Measured with `examples/window_bench.rs` at k = 200, a window of 1000 and `window_every` 25: a windowed `ewridge` row 34.6 → 17.2 µs (11.3 without a window), an accumulate-only windowed `ew_cov` row 28.1 → 9.8 µs (5.9 without) |
| P2 | **fixed** (batch 3a) — `Instance::reset` builds its own instance at its own decay; it built every instance of the grid and kept one |
| P4, P5 | **fixed** (batch 3b), on the user's decisions of 2026-09-15 (`docs/PLAN.md` task 80). P4: `window_budget = {"thin": MiB}` or `{"refuse": MiB}` on the five windowed kinds, and a window with no budget refuses past 256 MiB per ring (**my default, raised**). A refusing ring stops at the budget -- the snapshot that would cross is not kept, and none is made after -- and the chunk is refused. The budget is found as the rows go in, so by then the bank has learned part of the chunk: it refuses every later `fit_predict`, `predict` and `save` rather than go on (**a design call, raised**). P5: the runner drains after every chunk and hands each drain to the sidecar's writer; `fit_predict_batches(closed_groups=)` drains after every batch and writes however the chunks stop; `closed_groups()` says the queue is bounded only by draining it. On the old build all fourteen tests failed for their findings' reasons |
| D1 | not taken: excluded by the user (2026-09-14) -- hard rule 5, on backward file compatibility, stays as written |
| D2 | **fixed** (batch 1) — one ladder, `factorize`, under `solve_spd` and `SpdFactor::of`; the solve is bit-identical, and a guard test holds the two to the same rung on a well-posed, a singular and an indefinite matrix |
| D3, D4, D5 | **fixed** (batch 1) — `Kalman::coefficients`, `robust`'s module doc and `sgd`'s `scale_features` now say what the review says they left out. D4's `HuberRegressor` comparison is sklearn's, which the project does not take |
| D6 | **fixed** (batch 1), and **short of the whole story** (raised). The window's note has been right since C17, as the review says; the test's "about eight significant figures" was never the window. Its oracle, `direct_window_fit`, put the ridge on the intercept, which the model leaves unpenalized, and at `ridge = 1e-8` that alone is the `3.8e-8` the `1e-6` tolerance hid. With the intercept free the two agree to `1.2e-14` over 284 comparisons, and the test holds `1e-12` |
| D7 | **fixed** (batch 2) — the lists of models with no target now name all eight and point at `_spec.py`'s `UNSUPERVISED`; `gram_axes` says why it names `ew_cov` alone; the bank's tripwire comment says which refusals make it unreachable; `group_indices` calls `session_hash` instead of repeating it. `ModelBank.predict`'s `session_gap` line is C19's, for batch 3 |
| D8 | **fixed** (batch 2), apart from `ewridge`'s docstring on `sigma` under a window, which S1 settles (batch 4). `_numeric_keys` reads every builder -- it missed `bocpd`'s floats among others, which a test now holds and which failed on the old build; `_json`'s NaN message names the spec of a list; `_checked` says the Rust side's floor of 1 for eight counts; the `lasso` and `bocpd` docstrings, `MarginalKwargs`, `spec.rs`'s `stats` and `covariance` docs and `closed_groups`' doc say what the code does |
| D9, D10 | D9 **fixed** (batches 4a and 4b): `holt.rs`'s module doc (S29/S30); the core's `combo_labels`, which nothing but its own test called -- the bank renders `stream::combos` -- is gone; `predict_chunk` says a row with an unusable weight is scored anyway; `max_dclock`'s doc says the folded total is capped (S3). D10 **begun** (batch 4a): `ftrl.rs`'s module doc (C24), `the_row_weight_scales_the_gradient`'s comment -- the weight multiplies the loss, river's and sklearn's convention, so a row at weight 4 is not four rows -- `spec.rs`'s `strict_binary` (S31), and `reference.py`'s `ftrl_ref`, which gained the squared loss and names the decay's effect. `pa.rs` and `hmm.rs` are batch 4c |
| V23 | **fixed** (batch 2), with S7 -- `feature_sets = []` is refused, naming `feature_sets`; on the old build it built, and `output_index` rendered no slot for a model that emits two |
| V5 | **closed** (batch 3a) — the test the review asked for exists: `test_frame.py::test_pushdowns_are_honoured_after_the_model`, case "head then filter", compares the plan with the collected frame under both engines |
| V7 | **fixed** (batch 3a) — confirmed through a real save: a reloaded filter's `pred_var()` was `R` alone, 0.25 against the saved filter's 0.2545, since it read the last row's regressor, a scratch a save does not keep. `pred_var` takes the row now, as `predict` does; nothing above the core reads it. The first version of the test passed on the old build because it restored from `state()`, an in-memory clone that keeps the scratch; it was corrected to go through the bytes, and then failed as the review said |
| V11 | **closed** (batch 3a), confirmed -- every row passes `usable` before a model steps, however its column is laid out in memory. A test (`test_hardening.py`) holds a column in two chunks with an infinity in the second; it passed on the old build, as the review expected |
| V24 | **closed** -- it asks what sklearn's `PassiveAggressiveRegressor` computes, and sklearn is not a dependency of this project (a standing rule), so there is nothing here to test against |
| V25 | **fixed** (batch 3a) for `hmm`, and **not a defect for `kmeans`** (raised). `hmm` replayed its warm-up at `lam = 1` with the raw weights, so its states began weighing 20.7 where `n_eff` was 11.7; its buffer now ages on every row, as `kmeans`' always did. The review presumed `kmeans` shared the replay; its test, that the centres weigh what `n_eff` does, passed on the old build |
| V22 | **closed** with task 81: `MIN_SCHEMA_VERSION` 8 refuses every layout older than `Run::len`, and with `RunWire` gone the field no longer defaults when absent, so a file without it is refused rather than loaded at `len = 0` |

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

- 2026-09-15 — **batch 1**. Seventeen tests were written or extended
  before any fix. Thirteen failed on the old build, each for the reason
  its finding gives. Four passed: S10's, whose premise is false; the
  review's own test for S13's second bullet, whose example the jitter
  ladder rescues; S6's, which pins the true comment, as it should; and the
  contract's new zero-weight check, a guard under which no model's
  prediction went NaN. D2's guard test came with the refactor and holds on
  either build. The user's decisions on S1, S2, S3, S23, S27, S29/S30, S31
  and C24 are in `docs/PLAN.md` task 80, for batch 4.

- 2026-09-15 — **batch 2**, the spec layer: S5, S7, S8, S21, S22, S24,
  S25, S26, C18, C20, D7, D8 and V23. Each finding was read again against
  the code first; all held as written. Every new test failed on the old
  build, each for its finding's reason -- 5 in Rust and 33 in Python,
  counting the `coef_every` rule held to the field names for every kind.
  Two calls went beyond the review's text, for the user to see: S22's
  `kalman` pair is documented rather than refused (the docs already give
  `q` the precedence, and specs use the pair), and S5 also reports a
  single named feature set in a ridge grid's metadata, which dropped it.
  On the new build four existing tests failed, each pinning what the batch
  changes, and each was changed with it: `test_no_output.py` asserted
  `bank.specs` was the caller's unfilled dict (S21's defect);
  `test_portability.py` varied `coef_every` on `ew_cov`, which has no
  `coef` (S22); `state_encoding.rs` encoded a `marginal` with a window and
  lags (C18); and `test_error_messages.py` expected the core's `pca_every`
  message, which the builder now gives first (D8). `bank.rs` passed the
  hygiene cap of 200,000 bytes by 219 with the batch's comments, which were
  trimmed. The full gate passes.

- 2026-09-15 — **batch 3a**: C4, C19, P1, P2, P3, S12, V7 and V25 fixed;
  V5, V11 and V24 closed. Every test was written first. On the old build
  C4, C19, V7 and `hmm`'s V25 failed for their findings' reasons; V11's pin
  and `kmeans`' V25 passed, the latter where the review presumed a defect
  (raised). One test of mine was wrong before it was right: V7's first
  version passed on the old build because it restored from an in-memory
  clone, and through the bytes a save writes it failed as the review said.
  P1 and P3 are measured rather than failed: a windowed row 34.6 → 17.2 µs
  and 28.1 → 9.8 µs, bit-identical. P4 and P5, on the user's decisions,
  are batch 3b.
- 2026-09-15 — **batch 3b**: P4 and P5 fixed, on the user's decisions.
  Every test was written first, and on the old build all fourteen failed
  for their findings' reasons: P4's for want of `window_budget`, the
  runner's sidecar as one record batch written at the end, and
  `fit_predict_batches` for want of `closed_groups`. Three defects in my own
  first version, found by its tests, are recorded so they are not repeated.
  (1) A refusal cannot leave the bank as it was. Every other refusal is
  checked before a stream is touched; the budget can only be found as the
  rows go in, across groups learning in parallel, and cloning every touched
  stream per chunk would cost what the default budget exists to bound. So a
  bank refused for it refuses to go on (raised: a design call). (2) The
  refusing ring kept growing through the rest of the run of rows the bank
  checks after, and now stops at the budget. (3) The byte count walked the
  whole ring on every snapshot -- O(ring) per row on every windowed spec,
  now that each has a default budget -- and is a running count, held to a
  walk in the unit tests. Two notes on method. polars 1.44's `read_ipc`
  merges record batches (its `rechunk` is deprecated) and pyarrow is not a
  dependency, so the runner's test counts batches from the IPC footer and
  pins the count at two chunkings. And `bank.rs` passed the 200,000-byte
  cap, so `F64Column` and its tests moved to `column.rs` unchanged.
  `fit_predict_batches` writes what it drained however the chunks stop --
  a drained row has left the bank -- where the plan's source writes only at
  the end, its bank going with the plan.
- 2026-09-15 — **batch 4a**: S29, S30, S31 and C24 fixed, on the user's
  decisions, with D9's and D10's items for `holt.rs`, `ftrl.rs`, `spec.rs`
  and `reference.py`. Every test was written first. On the old build the
  seven Rust tests and six Python ones written to show a finding failed,
  each for its finding's reason, and three controls passed as they should:
  river's weight at `inf`, a constant weight cancelling, and the review's
  exact check for S29 -- a row at weight 2 is that row twice at no clock --
  which the old build also met, both sides ignoring the weight (raised).
  Three things the decisions settle that may not have been pictured, all
  raised. Every `holt` stream's numbers move, weighted or not, since the
  weighted means' gains start at 1; statsmodels' `Holt` agrees from row
  1000, and T-S14 now asserts the parting before it. `trend_halflife = inf`
  is the whole history's drift, and a plain level has no spelling left. And
  C24's repair stops short of the review's numbers -- a constant 5 at
  `halflife = 100` settles at 4.65 (2.25 before; the review asked for 5.0),
  a gap's factor is 0.75 (0.70; the review asked for 1), a logistic base
  rate of 0.9 reads 0.873 at `halflife = 1000` (0.816; the review asked for
  0.90) -- because the penalties stay on the sums' scale, which is what
  keeps the undecayed model river's. Schema 9, and minimum 9: `holt`'s
  weights and `ftrl`'s sum cannot be read from a schema-8 state.
  The first gate failed one test of the model contract,
  `holt_recovers_from_bounded_extremes` -- 7.85e80 against the twin's 0.52
  at row 29,520, finite throughout. Its script's tail is 1500 halflives of
  the probe's `HALFLIFE`, what a row at the bound with weight at the bound
  needs to wash out of a mean-form accumulator; `holt`'s trend is one now,
  and forgets on its own halflife, four times the level's. Measured on a
  replica of the script: the default halflives recover to 1.1e-15 on a
  tail four times as long, both at `HALFLIFE` to 1.6e-15 on the script's,
  and the largest forecast on the way is 2.5e102, far from overflow. The
  probe runs both halflives at `HALFLIFE` (1.3e-15); the model is unchanged.
- 2026-09-15 — **batch 4b**: S2, S3 and S23 fixed, on the user's
  decisions, with D9's last items. Every test was written first. On the
  old build each one written to show a finding failed for its reason -- the
  clock handed the next row 660 where the cap is 60, and 550 through the
  bank (`n_eff` 1.0 against 1.03125); a target present on every tenth row
  first predicted at row 5 under `[5, 5]`, for all five models that keep a
  weight per target; the averaged prediction did not scale with `y` on any
  row, and the formula check read 1.84 against 1.41 -- and the existing
  tests of the classes they joined passed. S2 is a trait hook the stream
  reads before the step, `target_n_eff_into`, since `Step` is built at
  seventy sites; its meaning is `n_eff`'s, read where `predict` reads the
  window's weights. The decisions reached the suite's own oracles, which
  encoded the old rules: about seventeen replays folded skipped-row time
  without the cap (the five `reference.py` references take a `max_dclock`
  now, and their callers pass it), and three references gated a target's
  output on the shared weight. The first version of that fix gated the
  references' internal prediction too, and `kalman_ref`'s noise estimate,
  which folds the model's own prediction whatever is emitted, drifted by
  0.34 -- the gate is on the output, never the model. The pipeline goldens
  moved as the two decisions say, 103 values: S3's capped fold reaches
  every model that reads features (`n_eff` at row 119, 14.96 → 15.11), and
  S2 moves `ridge`'s early conformal band and sigma, and the `seqtest` that
  compares its slots.

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
- **N2** (while fixing N1) -- **fixed** with task 81. `lasso` kept its
  per-target cross-moments raw in the same way and recovered its intercept
  from them. It keeps `ewridge`'s centred ones now, which `target_gaps`
  needed anyway. Rust: `a_level_costs_the_path_nothing`, the N1 test at
  every path point; library: statsmodels' elastic net at a penalty.
- **N3** (while fixing N1) -- **fixed** as docs/PLAN.md task 81, on the
  user's decision. With a target null on some rows, the Gram was over
  every row and the target's cross-moments over its own, so the slopes
  depended on the target's level: the solve added `(m_j − m)·ȳ_j / Var(x)`
  to a slope, `m_j` the feature's mean over the target's rows. The user
  asked for every option behind one parameter, then dropped that reading
  as adding nothing. `target_gaps="own_rows"`, the default, fits each
  target on exactly its rows; `"pairwise"` keeps one Gram and centres each
  target's cross-moments at its own means. Library: `numpy`'s `lstsq` and
  statsmodels' `WLS`, ridge and elastic net for `own_rows`; `pandas`'
  `DataFrame.cov` and `numpy.cov` for `pairwise`. Before the fix the
  default's fit was off by 1.17 at level 0 and by 49.2 at level 50.
- **N4** (while fixing N1) -- **half done** with task 81. `po.gram.solve`
  solved the raw normal equations; it solves the centred system now, with
  each target's own column means (`means_by_target`), as the model does.
  What is left: its right-hand side is formed from the raw cross-moments
  the export carries, `E[z·y] − m·ȳ`, which loses `L²·ε` at a level `L`.
  Exporting the centred cross-moments would close it; kept for later, as a
  change to the export.
- **N6** (while testing S13) -- **fixed** in batch 1. `ewridge`, `robust`
  and `kalman` each keep `σ²`, the EW mean of the squared out-of-sample
  errors, with its weight `wsig`. A row with a target and no prediction --
  the rows after a clock gap has taken `n_eff` under `min_periods` --
  rightly added nothing, and did not age `wsig` either, so `σ²` forgot less
  across such rows than the clock says; in `robust` a zero-weight row did
  the same (S13's first bullet). Every row ages `wsig` now, and a row adds
  `w·r²` when it has a target, a weight and a prediction. The test is the
  definition: an EW mean kept by hand beside each model over a stream with
  nulls, zero-weight rows and such a gap, at `1e-12`. Each failed on the old
  build at the first predicted row after the gap (`robust` sooner, at the
  row after its first zero-weight row). Bit-identical wherever the old code
  updated; `σ²` sets `kalman`'s noise and `robust`'s cuts, so their
  predictions after such rows move. The stream's own `sigma` (`resid_var`)
  was checked and is not affected: a row it does not learn is not stepped
  by the model either, and its clock delta is carried to the next.
- **N5** (while building task 81) -- **fixed**.
  `po.gram.solve(standardize=True)` and `po.gram.lasso_path` without an
  intercept scaled the centred co-moments against the raw cross-moments:
  the hybrid the models lost in C8, least squares only when every feature
  has mean zero. Both read raw moments there now. The tests compare with
  the models, a feature moved off zero (`test_gram_module.py`); the models'
  fit through the origin is held to `numpy` already (C8).

## Libraries

Nothing waits on a library now. `statsmodels`, `filterpy` and
`bayesian-changepoint-detection` joined the `dev` group on 2026-09-13, and
`pandas` may be used in tests (the user, the same day). What each finding
still waits on is in the tables.
