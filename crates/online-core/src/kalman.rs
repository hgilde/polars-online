//! Kalman / random-walk-beta dynamic linear model (docs/PLAN.md §4.4).
//!
//! State per target: coefficient mean `b_j` and covariance `P_j` (k x k).
//! Observation `y_j = z' b_j + e`, `e ~ N(0, R_j)`; coefficients follow a random
//! walk `b_j <- b_j + w`, `w ~ N(0, Q)`.
//!
//! Per row (clock delta `d`, weight `w_row`):
//!
//! ```text
//! b_j <- Phi b_j                      (transition; Phi = diag(2^(-d/r_i)))
//! P_j <- Phi P_j Phi                  (every row)
//! D_j <- D_j + d                      (the clock since P_j last took an observation)
//! P_j <- P_j + Q * D_j^2, D_j <- 0    (only on a row that observes P_j's target;
//!                                      on a reverting slot D^2 is g_i(D)^2, below)
//! s    = z' P_j z + R_j / w_row       (innovation variance)
//! k    = P_j z / s                    (gain)
//! b_j <- b_j + k (y_j - z' b_j)
//! P_j <- P_j - k z' P_j
//! ```
//!
//! **Reversion (ENHANCEMENTS E41).** `revert_half_life` gives each slot a
//! reversion half-life `r_i`: between observations the coefficient mean
//! shrinks toward zero by `2^(-d/r_i)`, so a coefficient no row has
//! supported for a while is forgotten rather than carried. `r_i = inf` (the
//! default) is `Phi = I`, the random walk, and costs nothing. The pull is
//! toward zero in the *standardized* coordinates when `standardize` is on:
//! a slope toward "no effect", the intercept toward "the target averages
//! zero"; give the intercept `inf` to leave it a random walk. The transition
//! runs on every row, the ones that observe nothing included: `Phi` is
//! multiplicative over a split gap, `2^(-d₁/r) 2^(-d₂/r) = 2^(-(d₁+d₂)/r)`,
//! so where a gap is cut does not move it.
//!
//! **A reverting slot's process noise is bounded in the gap (docs/PLAN.md
//! task 214).** For an observation `D` after the last it is `q_i g_i(D)²`,
//!
//! ```text
//! g_i(D) = (1 - 2^(-D/r_i)) / theta_i,    theta_i = ln2 / r_i      (g_i(D) = D at r_i = inf)
//! ```
//!
//! the displacement of a velocity of variance `q_i` held over the gap while
//! the reversion pulls, `db = (u - theta_i b) dt`, which `q_i D²` is without
//! the pull. It is `q_i D²` for a gap well under `r_i` (0.993 of it at `D =
//! r_i/100`, 0.933 at `r_i/10`), so the matching below holds there, and it
//! saturates at `q_i / theta_i²` past `r_i` (0.9998 of it at `20 r_i`):
//! across a run of rows that observe nothing the coefficient's uncertainty is
//! bounded, as a mean-reverting coefficient's is. Charged `q_i D²` for the
//! whole gap (task 211) it grew without bound; charged per row, as before
//! that, it settled at `q_i / (1 - phi_i²)`, a number of the rows the gap was
//! cut into. At observations one clock unit apart the noise is `q_i g_i(1)²`,
//! short of `q_i` by about `ln2 / r_i`: 10.8% at `r_i` = 6, 2.7% at 25, 0.7%
//! at 100, which moved a prediction by at most 0.26, 0.03 and 0.004 noise
//! deviations on a dense stream whose slope drifts. With `Q` from `coef_half_life` a reverting
//! slot settles, at observations `D` apart, at the prior variance `q_i
//! g_i(D)² / (1 - phi_i²)`, a stationary AR(1) instead of an unbounded walk
//! (for `D` well under `r_i` that is about `q_i D r_i / (2 ln2)`, which grows
//! with the spacing as the gain matching's own variance does).
//!
//! **Process noise from a per-factor half-life.** On standardized features, the
//! steady-state gain of a random-walk-beta filter matches EW-RLS with half-life
//! `h_i` when the noise added for an observation `D` clock units after the
//! last is `sigma^2 * (ln2 * D / h_i)^2 = q_i D^2` with `q_i = sigma^2 *
//! (ln2 / h_i)^2` (docs/PLAN.md §4.4, task 150): EW-RLS forgets `2^(-D/h)` of
//! its information between the two observations, a half-life of `h/D`
//! observations, and that is the half-life the matching is done at. Per
//! direction, with `p = P / R` before the observation, the filter's steady
//! state solves `p² / (p + 1) = c²`, `c = ln2 D / h`, and its gain `p / (p +
//! 1)` is EW-RLS's `1 - 2^(-D/h)` to first order in `D/h`: the exact noise
//! is `sigma^2 g^2 / (1 - g)` with `g = 1 - 2^(-D/h)`, which `c²` is within
//! 1% of for observations closer than half a half-life and 4% at one.
//! `half_life` may be scalar or per factor; `half_life = inf` gives `q_i =
//! 0`, pinning that coefficient. An explicit `q` overrides the derivation
//! and is added as `q_i D^2` too: the noise an observation one clock unit
//! after the last adds, not a variance per unit of elapsed clock.
//!
//! **The noise is charged once per observation, for the whole clock since
//! the last (docs/PLAN.md task 211).** `D` is the clock since the
//! covariance last took a row that observes its target -- present, at a
//! positive weight; under `share_p` any of the targets -- and a row that
//! observes none, a null target or a weight of 0, only adds its `d` to it.
//! The noise is not additive over a split gap: `q (d₁² + d₂²) < q (d₁ +
//! d₂)²`. Charged per row, as it was, a row of weight 0 inside a gap -- which
//! must advance the clock and teach nothing (hard rule 9) -- shrank the
//! gap's noise and moved every later prediction (0.197 at `coef_half_life`
//! 20 on the review's stream of 2026-10-08), and so did a null target, and
//! `po.stream.embargo`'s doubled stream, whose predict rows are such rows,
//! parted from the native embargo. And a target seen on one row in `n` took
//! `n` charges of `q d²` where one of `q (n d)²` was due: its memory grew as
//! `h √n`, 166, 270 and 462 clock units at `n` = 4, 10 and 25 against 74.5
//! seen on every row. Charged per observation, the matching above is made at
//! the observations' own spacing `D`, whatever rows lie between them, so
//! `coef_half_life` is a half-life on the clock for a sparse target as for
//! a dense one: EW-RLS's `2^(-D/h)` is the same over one gap of `D` as over
//! any cut of it. The transition stays per row, being multiplicative. A
//! `gap_cap` caps each row's `d` before it reaches the model, so a gap cut
//! into rows escapes a cap that the gap whole would meet: the doubled
//! stream's sub-gaps do, for every model, and it is an oracle for the
//! native embargo only without a cap.
//!
//! **What `Q` means in the caller's units.** Under `standardize` the noise
//! is defined per standard deviation of each feature, as the moments stand
//! when it is charged: a slope's random walk in the caller's units has a
//! variance per observation of `q D² / s²`, a drift measured in the feature's
//! current spread. A feature whose spread doubles has its slope's walk
//! slowed by four in its own units; measured on the review's stream of
//! 2026-10-08, a slope's `se_coef²` grew by `sigma² (ln2 / h)² / s²` a
//! unit of clock at `s` = 1 and 10 alike (ratio 1.015), when the noise was
//! charged per row.
//!
//! **Memory under correlated features.** `coef_half_life` is the memory of
//! each standardized direction with the design's other directions held
//! still. Where features are correlated, the information along an
//! eigendirection of the design's correlation with eigenvalue `λ` is `λ`
//! per row, and a random walk of the same variance in every direction is
//! learned along it with a memory of `h / √λ` (Ljung and Gunnarsson 1990):
//! measured at a correlation of 0.9 and `h` = 50, 37.5 and 160.5 clock units
//! along `[1, 1]` (`λ` = 1.9) and `[1, -1]` (`λ` = 0.1), against `h/√λ` =
//! 36.3 and 158.1 (with `obs_var` the true noise). EW-RLS forgets every
//! direction at `h` alike (51.5 and 52.5 there), so the two match at any
//! spacing only on uncorrelated features. And with `R` the EW residual
//! variance, which a step in the truth inflates before `P` catches up, the
//! step response is slower than with the true noise: 72.5-74.5 clock units
//! against 50-51.5 at `h` = 50 on uncorrelated features, about 1.45 times.
//!
//! **Standardized coordinates, held at an anchor (docs/PLAN.md task 211;
//! task 206 before it).** Features are standardized against a shared
//! [`EwDiag`] over `z`, EW means and variances, O(k) a row: `z_0 = 1` and
//! `z_i = (x_i − m_i) / s_i` (`x_i / s_i`, the root of the raw second
//! moment, without an intercept; review 2026-09-12, C10), against the
//! moments before the row. `R_j` defaults to the EW residual variance
//! `sigma^2_j` unless `obs_var` is given. `b` and `P` are held in the
//! coordinates of an *anchor*: the moments' means, as pairs, and scales at
//! the last re-map, `m^a` and `s^a`. The row is read in them, `z^a_i = (x_i −
//! m^a_i) / s^a_i`, and so is the prediction `z^a' b`. With `m`, `s` the
//! moments as they stand, `z^a = M z` for every row, with
//!
//! ```text
//! M_00 = 1,    M_i0 = c_i = (m_i − m^a_i) / s^a_i   (0 without an intercept),    M_ii = a_i = s_i / s^a_i
//! ```
//!
//! so `b` is `b_cur = Mᵀ b` in the current coordinates and `P` is `Mᵀ P M`:
//! no prediction and no predictive variance moves because the moments did,
//! and the coefficients in the caller's units move only with the fit. The
//! reversion, the process noise and the prior are defined in the current
//! coordinates, so a half-life and `p0` mean the same thing whatever a
//! column's units, and carried into the anchor's exactly. The noise `Q D²`,
//! diagonal in the current coordinates, is `M⁻ᵀ Q M⁻¹` in the anchor's, an
//! arrowhead -- the diagonal plus the intercept's row and column -- and O(k)
//! a row:
//!
//! ```text
//! v_i = c_i / a_i:    P_00 += Σ_i v_i² q_i D²,    P_0i = P_i0 += −v_i q_i D² / a_i,    P_ii += q_i D² / a_i²
//! ```
//!
//! A slot's prior `p0 R` is carried in the same way, and the transition `Phi`
//! becomes `M⁻ᵀ Phi Mᵀ`: `b_0 <- phi_0 b_0 + Σ_i g_i b_i` with `g_i = c_i
//! (phi_0 − phi_i)`, `b_i <- phi_i b_i`, which is `Phi` itself where every
//! slot reverts at one half-life. Every readout -- the prediction, `pred_var`,
//! the coefficients and their variances, the readiness statistic -- reads
//! the anchor's coordinates or maps to the current ones exactly.
//!
//! **The re-map.** When the moments drift from the anchor -- a scale by
//! more than a factor of `1 + τ` either way, `max(a_i, 1/a_i) − 1 > τ`, or a
//! mean by more than `τ` of the anchor's scale, `|c_i| > τ`, over the slots
//! that hold anything, with `τ` = 1 (`ANCHOR_DRIFT`) -- `b` and `P` are
//! mapped to the moments as they stand after the row, `b <- Mᵀ b` and `P <-
//! Mᵀ P M`:
//!
//! ```text
//! b'_0 = b_0 + Σ_i c_i b_i,    b'_i = a_i b_i
//! u = P_0· + Σ_l c_l P_l·,   P'_00 = u_0 + Σ_j c_j u_j,   P'_0j = P'_j0 = a_j u_j,   P'_ij = (a_i a_j) P_ij
//! ```
//!
//! and the anchor becomes those moments. The drift bounds the arrowhead's
//! terms, `v_i² ≤ 4`, so no row reads a large number back through a
//! cancellation. Measured on the review's regimes (30,000 rows, `k` 1 and
//! 5, R² 0.978 and 0.99998, half-lives 10 to none) against the per-row
//! re-map of task 206 run on the same rules: within 1.3e-12 of a noise
//! standard deviation at `coef_half_life` 50, and 2.0e-11 with no process
//! noise, where two per-row implementations of it (Rust and numpy) differ
//! by 4.7e-12; 3 to 5 re-maps in 30,000 rows of stationary data, most of
//! them in the first rows, and 20 to 45 at a half-life of 10. A drift in
//! `|a_i − 1|` alone never re-maps a shrinking scale, and leaves `a_i` tiny
//! with arrowhead terms of `q / a_i²` that cancel in every prediction: on a
//! feature whose spread shrinks 3% a row the anchored filter with that
//! drift parted from the per-row re-map by 8e9 noise standard deviations,
//! where this one keeps to 2.7e-14. Task 206 re-mapped on every row, four
//! sweeps over `P`: at `k = 10` a row cost 277 ns, and costs 221 here
//! (2026-10-08, median of nine interleaved runs at a load of 5 to 7).
//!
//! A row of weight 0 moves no moment and nothing is re-mapped. A re-map past
//! what a stretch of data makes -- a scale by more than a factor of 1024
//! either way, or a mean by more than 1024 of the anchor's scale -- is
//! refused whole, and so is one any of whose numbers would not be finite:
//! `b` and `P` keep their numbers and are read in the new coordinates, as
//! the row's other updates that would not be finite are skipped
//! (docs/IMPROVEMENTS.md C2). The re-map puts `c² P_ii`
//! into the intercept's variance, and every later row reads it back through
//! a cancellation, so a shift of `c` costs `c²` rounding steps of `P`'s
//! narrowest direction: at the input bound, where a feature of 1e100 shifts
//! a mean by about 1e99 of its scale, the re-map left a slope's variance at
//! −3.6e149 and the reverting filter never recovered; at 1024 the cost is
//! 2.3e-10 of it.
//!
//! **The first rows (docs/PLAN.md task 211).** Over the first rows the
//! moments rest on nothing: on row 0 a feature is read against a mean of 0
//! and a scale of 1, raw, and on row 1 against a variance of 0. So each
//! coefficient's prior is set on the first row that observes the target on
//! which its feature's scale is usable -- a variance above 0 (with an
//! intercept), a raw second moment above 0 (without) -- and until then the
//! coefficient is 0 and its variance 0: it takes no correction and no
//! process noise, and a row's raw reading of the feature reaches nothing.
//! The intercept's scale is always usable. A feature whose variance becomes
//! usable later than the others' gets its own prior then. Under
//! `standardize` the prior of slot `i` is `P_ii = p0 R`, carried into the
//! anchor's coordinates, with `R` the mean squared innovation, `Σ w e² /
//! Σ w` over every row that has observed the target so far, this one
//! included (each `e = y − z' b` before its row's update), once it rests on
//! at least three rows (`PRIOR_ROWS`); with `obs_var`, `R = obs_var`, and
//! the intercept's prior is set from the start. Three is the number of rows
//! a feature's first usable scale rests on in a stream with a target on
//! every row (two for the moments' variance, and the row itself), so there
//! the intercept and the features are sized together; and a mean of three
//! squared innovations falls below 1% of its mean with probability 7e-4,
//! where one does with probability 0.08. With no process noise a prior
//! sized too small pins the fit for good, and one squared innovation did:
//! the review's seed whose sizing innovation was 0.004 on a target of spread
//! 4.5 read slopes of 0.3 to 1.2 against a truth of 2 after 30,000 rows.
//! The state follows the moments from the first row; the 22-row warm-up
//! `sgd` and `pa` keep ([`crate::Warmup`]) is for their step sizes, and the
//! filter's `P` carries its own uncertainty through the first rows. On task
//! 74's short histories (200-row groups, 20 features, half-life 50) the
//! warm-up and the prior from row 0 gave R² 0.722 on rows 25 to 50; this
//! gives 0.95, and a level of 1e8 in the features no longer moves any
//! prediction past the first rows (0.87 before). Features scaled by a power
//! of two predict the same to the bit: no row reads a raw `x`.
//!
//! **The noise before the first residual (review 2026-10-05, CC4).** A
//! target has no residual variance until a row with a prediction for it
//! gives it a residual, and a prediction needs `min_weight` met and an
//! earlier row of the target. Until then a row's noise is its own innovation
//! squared, `R_j = e_j^2` with `e_j = y_j - z' b_j`, computed before the
//! update and so out of sample, as the prediction is. The `sigma^2` the
//! process noise is derived from is the same number. On the row that gives
//! `sigma^2_j` its first residual, `e_j` is that residual, so the noise
//! starts where the residual variance will. The noise was the literal 1, in
//! the target's units, and the gains of a warm-up depended on those units.
//! A `sigma^2_j` of 0 (every residual so far exactly 0) counts as none.
//!
//! An innovation of exactly 0, or one whose square is not finite, sizes no
//! noise. At `R = 0` the update would take the row as exact and collapse
//! `P` along `z`, so such a row corrects nothing. Neither does a row with
//! no innovation, a null target or a weight of 0. Under `share_p` the noise
//! is the mean `sigma^2` over the targets that have one, and before any has
//! one, the mean of the squared innovations of the targets the row observes.
//! A given `obs_var` is the noise throughout. [`Kalman::pred_var`] keeps NaN
//! for `R_j` until there is a residual variance: it describes a prediction,
//! which is made before the row's target is seen.
//!
//! **The prior without `standardize` is `p0` times the first noise (CC4).**
//! `P_j` starts unsized, all zero. The first row with a noise sets it to
//! `P_0 = p0 R I`, with `R` that row's noise, and the row's gain reads it;
//! that row adds no process noise. So `p0` is a ratio: at 1 the prior is as
//! uncertain as one observation, the conjugate prior of a regression with an
//! unknown noise variance. With `obs_var` given, `P_0 = p0 obs_var I` from
//! the start, and with no process noise the filter is the ridge regression
//! with penalty `1 / p0` -- unstandardized only: standardized, each slot's
//! prior waits for its feature's scale, as above. Scaling every target by
//! `c` (with any `obs_var` or explicit `q` by `c^2`) scales every
//! prediction by `c`, at any `p0`. A slot whose variance has decayed to
//! exactly 0 through the reversion holds nothing, and is sized again.
//!
//! **One covariance for every target (`share_p`).** `P` is per target
//! because the Riccati recursion depends on `R_j`. With `share_p` the filter
//! keeps one `P` driven by the mean `sigma^2` over the targets that have one
//! (docs/PLAN.md §4.4 [validate]), as it stands when the row arrives, summed
//! in ascending order. A target with no residual variance yet has no
//! estimate of the noise, not a noise of 0: counted in the mean, a target
//! null for its first rows halved the noise the others were weighed against,
//! and moved their predictions (review round 5, A1). `P`'s recursion reads
//! `z`, `R` and the weight, never `y`, so targets that share `R` would each
//! carry the same `P`: the shared one is that `P`. Every target observed on
//! the row takes its gain from `P` as the row finds it, and `P` takes the
//! row once, after them, if any target updated. So the order of the targets
//! moves no bit of any prediction, and a target beside an exact copy of
//! itself present on the same rows predicts as it would alone, to the bit. A
//! copy null on rows its twin is present on is not that: until it has a
//! residual variance it moves nothing, and from then its variance, learned
//! from fewer rows, enters the mean. Updating the shared `P` once per
//! target, as it did, counted each row once per target, as if the targets
//! shared their coefficients (docs/PLAN.md task 204).
//!
//! The shared `P` is the covariance of a filter whose noise is the mean
//! `σ̄²`. With a constant noise `σ²_j` per target, target `j`'s own filter
//! would keep `P_j = σ²_j P̃` and the shared one keeps `σ̄² P̃`, so target `j`'s
//! covariance is `P σ²_j / σ̄²`, and that is what its `se_coef` and
//! `pred_var` read (task 211). Read off `P` as it stands, two targets of
//! noise 0.01 and 1 had standard errors 7.3 times too large and 1.39 times
//! too small. A target with no residual variance yet is read at the mean.
//! The readiness statistic, `z' P_j z / σ²_j`, is `z' P z / σ̄²` for every
//! target, and needs no scaling.
//!
//! **Readiness (docs/PLAN.md task 116; docs/WARMUP-AND-CONVERGENCE.md
//! §2.1).** The filter knows the estimation variance of each prediction
//! under its own model: before row `z`'s target, at `d` clock units after the
//! last row, the prediction's variance over the noise is
//!
//! ```text
//! h(z) = z' P⁻ z / R,    P⁻ = Phi P Phi + Q (D + d)²
//! error_inflation = sqrt(1 + h(z))
//! ```
//!
//! with `z` standardized as the prediction is and `P⁻` the prior the row's
//! update starts from, after the transition and the process noise its clock
//! since the last observation adds. `R` is the noise as the state holds it:
//! `obs_var`, else the target's residual variance (the targets' mean under
//! `share_p`). Never the row's own innovation, which reads its target (hard
//! rule 2), so the ratio is infinite while `P` is unsized or the target has
//! no residual variance yet. `z' P⁻ z + R` is the prior predictive variance
//! a Kalman filter's `predict` step gives: exact where the model is
//! specified exactly, and per row, so the noise gate reads it per row.
//! `O(k²)`, paid only when the gate is set or the field emitted. It is
//! exact *under the model*: with `R` the EW residual variance of the
//! out-of-sample errors, which already carry the estimation error, the
//! ratio counts that error twice, and measured the mean squared error over
//! the mean `sigma² error_inflation²` is 0.96 at `coef_half_life` 50 (0.99
//! at 200).
//!
//! Without a row to read it at (the bank's summary) the statistic is its
//! mean field over the design, `sqrt(1 + Σ_i P_ii E[z_i²] / R)` at the
//! posterior, in the current coordinates: exact for uncorrelated features,
//! and above the average of `h` over the design while `P` tracks it (a
//! correlation matrix's inverse has a trace of at least `k`).
//!
//! Under process noise `P` never reaches 0. Per direction, on standardized
//! features with the half-life-derived `q`, `p = P⁻ / R` settles where
//! `p² / (p + 1) = c²`, `c = ln 2 · D / coef_half_life`, so
//! `p = (c² + sqrt(c⁴ + 4 c²)) / 2`, and the average row reads about
//! `sqrt(1 + k p)`: 1.07 at `k = 10`, `D = 1`, `coef_half_life = 50`.
//!
//! **Coefficient standard errors** (`se_coef`). `P` after the row is the
//! coefficients' posterior covariance in the anchor's coordinates, the noise
//! already in it; after a row that observes nothing, `P` as it stands
//! carries the noise of the clock since the last observation, `Q D²`, as the
//! next observation will charge it, so a coefficient's uncertainty grows
//! across a gap in its target (`pred_var` and the summary's readiness read
//! the same). Read out in the original units by the map `coefficients`
//! uses, `c_i = b_i / s^a_i` and `c_0 = b_0 − Σ_i c_i m^a_i`, it is `T P Tᵀ`,
//! whose diagonal [`OnlineModel::coef_variance`] reports. Exact under the
//! model, which says the coefficients walk: on a truth that does not move,
//! the process noise the half-life implies is variance the coefficients do
//! not have, and measured the standard errors are about √2 too large (a
//! mean squared error over `se²` of 0.47 to 0.50 at `coef_half_life` 50 and
//! 200, and 0.95 to 1.06 on a walk of the variance the half-life implies).
//!
//! **Per standard deviation, the readout before task 206.** Until task
//! 206 the state was read through the moments as they stood, which made the
//! model a filter on each feature's EW z-score: a coefficient per current
//! standard deviation, moving with the scale. That model is this one on a
//! z-scored feature column, which the window operators build in the stream
//! (docs/PLAN.md task 212): standardize the column there and give it to
//! `kalman` with `standardize=False`.

use serde::{Deserialize, Serialize};

use crate::model::{ModelState, OnlineModel, State, StateError, Step, check_schema};
use crate::{Decay, EwDiag};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KalmanCfg {
    pub n_features: usize,
    pub n_targets: usize,
    pub fit_intercept: bool,
    /// Decay used for the standardization statistics and the EW residual
    /// variance (NOT for the coefficients: those follow the random walk).
    pub decay: Decay,
    /// Per-factor coefficient half-life in clock units (length 1 or `k_total`).
    /// `f64::INFINITY` pins a coefficient. Ignored when `q` is given.
    #[serde(with = "crate::humanfloat::vec_f64_or_tag")]
    pub half_life: Vec<f64>,
    /// Explicit process-noise variances (length `k_total`), overriding
    /// `half_life`.
    pub q: Option<Vec<f64>>,
    /// Fixed observation variance; defaults to the EW residual variance, and
    /// before a target has one, to the row's own innovation squared (the
    /// module doc, CC4).
    pub obs_var: Option<f64>,
    /// The prior variance of each coefficient, as a multiple of the first
    /// noise estimate: `P_0 = p0 R I`, set on the row that first sizes the
    /// noise, or from the start with `obs_var` (the module doc, CC4).
    pub p0: f64,
    pub share_p: bool,
    pub min_weight: f64,
    /// Per-slot reversion half-life in clock units (length 1 or `k_total`,
    /// intercept first): the coefficient mean shrinks toward zero by
    /// `2^(-d/r_i)` per row before the process noise is added. `f64::INFINITY`
    /// (the default) is the random walk. See the module doc.
    #[serde(default = "default_revert")]
    #[serde(with = "crate::humanfloat::vec_f64_or_tag")]
    pub revert_half_life: Vec<f64>,
    /// Standardize features internally before filtering (default), the
    /// filter's state following the moments once they have warmed up (the
    /// module doc; docs/PLAN.md task 206).
    ///
    /// On by default because the half-life-derived process noise
    /// `q_i = sigma^2 (ln2/h_i)^2` is only comparable across features on a
    /// common scale. Turn it off when the features are already on a sensible
    /// scale and you want the filter to operate on them directly — that makes
    /// this exactly a Bayesian linear regression (with `q = 0` and a fixed
    /// `obs_var`), which is how it is cross-checked against river.
    #[serde(default = "default_true")]
    pub standardize: bool,
}

fn default_true() -> bool {
    true
}

fn default_revert() -> Vec<f64> {
    vec![f64::INFINITY]
}

impl KalmanCfg {
    pub fn k_total(&self) -> usize {
        self.n_features + usize::from(self.fit_intercept)
    }

    pub fn validate(&self) -> Result<(), String> {
        // The decay first: every model checks it in its own `new`, where only
        // the bank's spec did (review 2026-10-05, CF5).
        self.decay.check().map_err(|e| format!("kalman: {e}"))?;
        if self.n_features == 0 || self.n_targets == 0 {
            return Err("n_features and n_targets must be >= 1".into());
        }
        let k = self.k_total();
        if let Some(q) = &self.q {
            if q.len() != k {
                return Err(format!("kalman: q must have length {k}"));
            }
            // NaN passes `v < 0.0` and `v <= 0.0` alike, so each bound in
            // this function names it. A NaN `obs_var` was the silent case:
            // `s_inn` NaN on every row, the update's guard never met, and
            // the filter predicting its prior for the life of the stream
            // with no error and no counted failure (review 2026-09-18, B4).
            // Finite too, as the spec layer has it: an infinite `q` puts
            // `inf` on `P`'s diagonal (review 2026-10-06, CC7).
            if let Some(v) = q.iter().find(|v| !(v.is_finite() && **v >= 0.0)) {
                return Err(format!("kalman: q values must be finite and >= 0, got {v}"));
            }
        } else {
            if self.half_life.len() != 1 && self.half_life.len() != k {
                return Err(format!("kalman: half_life must have length 1 or {k}"));
            }
            if let Some(h) = self.half_life.iter().find(|h| h.is_nan() || **h <= 0.0) {
                return Err(format!(
                    "kalman: half_life values must be > 0 (inf pins), got {h}"
                ));
            }
        }
        if self.revert_half_life.len() != 1 && self.revert_half_life.len() != k {
            return Err(format!(
                "kalman: revert_half_life must have length 1 or {k}"
            ));
        }
        if let Some(h) = self
            .revert_half_life
            .iter()
            .find(|h| h.is_nan() || **h <= 0.0)
        {
            return Err(format!(
                "kalman: revert_half_life values must be > 0 (inf = random walk), got {h}"
            ));
        }
        // Finite, as the spec layer has them: at `p0 = inf` `P` is never
        // sized and the gain is 0 for good, and at `obs_var = inf` every
        // update is skipped; either predicts 0 for the life of the stream
        // (review 2026-10-06, CC7).
        if !(self.p0.is_finite() && self.p0 > 0.0) {
            return Err(format!(
                "kalman: p0 must be finite and > 0, got {}",
                self.p0
            ));
        }
        if let Some(v) = self.obs_var.filter(|v| !(v.is_finite() && *v > 0.0)) {
            return Err(format!("kalman: obs_var must be finite and > 0, got {v}"));
        }
        if self.min_weight.is_nan() || self.min_weight < 0.0 {
            return Err(format!(
                "kalman: min_weight must be >= 0, got {}",
                self.min_weight
            ));
        }
        Ok(())
    }

    /// Whether any slot reverts (`Phi != I`). The default random walk skips
    /// the transition entirely, so it stays bit-identical to before E41.
    fn reverts(&self) -> bool {
        self.revert_half_life.iter().any(|h| h.is_finite())
    }

    /// Slot `i`'s reversion half-life `r_i`, `inf` for a random walk.
    fn revert_of(&self, i: usize) -> f64 {
        if self.revert_half_life.len() == 1 {
            self.revert_half_life[0]
        } else {
            self.revert_half_life[i]
        }
    }

    /// The transition factor of slot `i` over a clock delta `d`,
    /// `2^(-d/r_i)`, spelled as [`Decay::factor`] is.
    fn phi(&self, i: usize, d_clock: f64) -> f64 {
        Decay::Halflife(self.revert_of(i)).factor(d_clock)
    }

    /// What slot `i`'s process noise `q_i` is multiplied by for an
    /// observation a gap of `d` after the last (the module doc): `d²` on a
    /// random walk, and on a slot reverting at `r_i`, `((1 − 2^(−d/r_i)) /
    /// θ_i)²` with `θ_i = ln 2 / r_i` -- `d²` for a gap well under `r_i`,
    /// and `1/θ_i²` past it, where `d²` grew without bound across a run of
    /// rows that observe nothing (docs/PLAN.md task 214). `1 − 2^(−d/r_i)` is
    /// `−expm1(−θ_i d)`, whole at a small `d`.
    fn gap_noise(&self, i: usize, d: f64) -> f64 {
        let r = self.revert_of(i);
        if r.is_infinite() {
            return d * d;
        }
        let theta = std::f64::consts::LN_2 / r;
        let g = -(-(theta * d)).exp_m1() / theta;
        g * g
    }
}

/// How far the moments may drift from the anchor before `b` and `P` are
/// re-mapped to them (the module doc): a scale by a factor of `1 + τ`
/// either way, a mean by `τ` of the anchor's scale.
const ANCHOR_DRIFT: f64 = 1.0;

/// How many rows' squared innovations a standardized prior rests on before
/// it is set (the module doc).
const PRIOR_ROWS: f64 = 3.0;

/// The squared innovations a covariance's prior is sized from (the module
/// doc): over the rows that observed one of its targets, the weighted sum of
/// the squared innovations, the weight, and the number of rows. Kept under
/// `standardize` without `obs_var`, and all zero otherwise.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
struct NoiseBasis {
    e2w: f64,
    w: f64,
    rows: f64,
}

impl NoiseBasis {
    /// A row's `n` squared innovations, summing to `e2`, at weight `w`: one
    /// row, whatever the number of targets it observes, so a target beside
    /// an exact copy of itself under `share_p` sizes its prior on the row it
    /// would alone.
    fn add(&mut self, e2: f64, n: f64, w: f64) {
        // A row whose squares would take the sum past the doubles is left
        // out, as a square that is not finite is: the sums stay numbers.
        let (e2w, sw) = (self.e2w + w * e2, self.w + w * n);
        if e2w.is_finite() && sw.is_finite() {
            self.e2w = e2w;
            self.w = sw;
            self.rows += 1.0;
        }
    }

    /// The mean squared innovation, once it rests on [`PRIOR_ROWS`] rows and
    /// is a positive number; none before.
    fn noise(&self) -> Option<f64> {
        if self.rows >= PRIOR_ROWS && self.w > 0.0 {
            let v = self.e2w / self.w;
            (v > 0.0 && v.is_finite()).then_some(v)
        } else {
            None
        }
    }

    fn is_valid(&self) -> bool {
        [self.e2w, self.w, self.rows]
            .iter()
            .all(|v| v.is_finite() && *v >= 0.0)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "KalmanV4")]
pub struct Kalman {
    cfg: KalmanCfg,
    /// Standardization stats over `z` (shared across targets): the means and
    /// variances the scales are read from.
    stats: EwDiag,
    /// The anchor (the module doc): per slot, the mean as a pair and the
    /// scale of the moments at the last re-map, the coordinates `b` and `P`
    /// are held in. Empty without `standardize`; the intercept's slot is 0,
    /// 0 and 1.
    anchor_hi: Vec<f64>,
    anchor_lo: Vec<f64>,
    anchor_scale: Vec<f64>,
    /// Per target: coefficient mean in the anchor's coordinates.
    beta: Vec<Vec<f64>>,
    /// Per target (or one when `share_p`): covariance, row-major `k*k`, in
    /// the anchor's coordinates; a slot with a variance of 0 is unsized
    /// (the module doc).
    p: Vec<Vec<f64>>,
    /// Per covariance: the clock since it last took a row that observes its
    /// target (the module doc, task 211).
    elapsed: Vec<f64>,
    /// Per covariance: the squared innovations its prior is sized from.
    basis: Vec<NoiseBasis>,
    /// EW residual variance per target and its weight sum.
    sig2: Vec<f64>,
    wsig: Vec<f64>,
    wj: Vec<f64>,
    #[serde(skip)]
    zbuf: Vec<f64>,
    #[serde(skip)]
    zs: Vec<f64>,
    #[serde(skip)]
    pz: Vec<f64>,
    #[serde(skip)]
    gain: Vec<f64>,
    /// This row's `z . b_j` per target, before any update: the prediction,
    /// and the innovation the update and the first noise read.
    #[serde(skip)]
    zb: Vec<f64>,
    /// This row's transition factors, one per slot (only filled when a
    /// slot reverts), and the transition's coupling `g` in the anchor's
    /// coordinates.
    #[serde(skip)]
    phi: Vec<f64>,
    #[serde(skip)]
    gbuf: Vec<f64>,
    /// This row's process-noise variances, kept between rows so a step
    /// allocates nothing for them (docs/PERFORMANCE.md §13).
    #[serde(skip)]
    qbuf: Vec<f64>,
    /// `share_p`'s noises, sorted to be summed in ascending order.
    #[serde(skip)]
    sumbuf: Vec<f64>,
    /// This row's noise per target, `R_j` and the `sigma^2` of `Q`.
    #[serde(skip)]
    noise: Vec<f64>,
    /// Which slots hold anything, in any target or covariance.
    #[serde(skip)]
    live: Vec<bool>,
    /// The map from the anchor to the moments as they stand, `a_i` and `c_i`
    /// (the module doc), and whether it was formed from the moments as they
    /// stand: formed after each row that moves them, for the next.
    #[serde(skip)]
    map_a: Vec<f64>,
    #[serde(skip)]
    map_c: Vec<f64>,
    #[serde(skip)]
    map_fresh: bool,
    /// The re-map's row `u = (Mᵀ P)_0·` for each `P`, and the transition's
    /// `P g`.
    #[serde(skip)]
    ubuf: Vec<Vec<f64>>,
    #[serde(skip)]
    vbuf: Vec<f64>,
}

/// The layout `Kalman` loads, checked on the way in: every vector at the
/// cfg's width, so a damaged state is refused rather than panicking on its
/// first step. (A schema-2 file kept a full `EwCov` under `cov` and was
/// converted by taking its diagonal; the minimum schema has passed it,
/// docs/PLAN.md tasks 198 and 194-202.)
#[derive(Deserialize)]
struct KalmanV4 {
    cfg: KalmanCfg,
    stats: EwDiag,
    anchor_hi: Vec<f64>,
    anchor_lo: Vec<f64>,
    anchor_scale: Vec<f64>,
    beta: Vec<Vec<f64>>,
    p: Vec<Vec<f64>>,
    elapsed: Vec<f64>,
    basis: Vec<NoiseBasis>,
    sig2: Vec<f64>,
    wsig: Vec<f64>,
    wj: Vec<f64>,
}

impl TryFrom<KalmanV4> for Kalman {
    type Error = String;

    fn try_from(v: KalmanV4) -> Result<Self, String> {
        let cfg = v.cfg;
        let k = cfg.k_total();
        let m = cfg.n_targets;
        let n_p = if cfg.share_p { 1 } else { m };
        let anchored = if cfg.standardize { k } else { 0 };
        if v.stats.k() != k
            || v.beta.len() != m
            || v.beta.iter().any(|b| b.len() != k)
            || v.p.len() != n_p
            || v.p.iter().any(|p| p.len() != k * k)
            || v.elapsed.len() != n_p
            || v.basis.len() != n_p
            || v.sig2.len() != m
            || v.wsig.len() != m
            || v.wj.len() != m
        {
            return Err("kalman: state has the wrong shape".into());
        }
        // The anchor is the standardizer's: without one there is no
        // coordinate to hold (docs/PLAN.md task 211).
        if v.anchor_hi.len() != anchored
            || v.anchor_lo.len() != anchored
            || v.anchor_scale.len() != anchored
        {
            return Err("kalman: the state's anchor does not match its cfg's standardize".into());
        }
        if !v.anchor_scale.iter().all(|s| s.is_finite() && *s > 0.0)
            || !v
                .anchor_hi
                .iter()
                .chain(&v.anchor_lo)
                .all(|m| m.is_finite())
        {
            return Err(
                "kalman: the anchor's scales must be finite and > 0, its means finite".into(),
            );
        }
        if !v.elapsed.iter().all(|e| e.is_finite() && *e >= 0.0)
            || !v.basis.iter().all(NoiseBasis::is_valid)
        {
            return Err(
                "kalman: the clocks since an observation and the noise's sums must be finite \
                 and >= 0"
                    .into(),
            );
        }
        // The scratch buffers are sized by `ensure_buffers` on the first use.
        Ok(Self {
            cfg,
            stats: v.stats,
            anchor_hi: v.anchor_hi,
            anchor_lo: v.anchor_lo,
            anchor_scale: v.anchor_scale,
            beta: v.beta,
            p: v.p,
            elapsed: v.elapsed,
            basis: v.basis,
            sig2: v.sig2,
            wsig: v.wsig,
            wj: v.wj,
            zbuf: vec![],
            zs: vec![],
            pz: vec![],
            gain: vec![],
            zb: vec![],
            phi: vec![],
            gbuf: vec![],
            qbuf: vec![],
            sumbuf: vec![],
            noise: vec![],
            live: vec![],
            map_a: vec![],
            map_c: vec![],
            map_fresh: false,
            ubuf: vec![],
            vbuf: vec![],
        })
    }
}

impl Kalman {
    pub fn new(cfg: KalmanCfg) -> Result<Self, String> {
        cfg.validate()?;
        let k = cfg.k_total();
        let m = cfg.n_targets;
        let n_p = if cfg.share_p { 1 } else { m };
        let off = usize::from(cfg.fit_intercept);
        // Unsized, all zero, until a row sizes the noise; with `obs_var` the
        // noise is known now, and so is the prior: of every slot
        // unstandardized, and standardized of the intercept, each feature's
        // waiting for its scale (the module doc).
        let mut p_init = vec![0.0; k * k];
        if let Some(r) = cfg.obs_var {
            let sized = if cfg.standardize { off } else { k };
            for i in 0..sized {
                p_init[i * k + i] = cfg.p0 * r;
            }
        }
        let anchored = if cfg.standardize { k } else { 0 };
        let mut model = Self {
            stats: EwDiag::new(k),
            anchor_hi: vec![0.0; anchored],
            anchor_lo: vec![0.0; anchored],
            anchor_scale: vec![1.0; anchored],
            beta: vec![vec![0.0; k]; m],
            p: vec![p_init; n_p],
            elapsed: vec![0.0; n_p],
            basis: vec![NoiseBasis::default(); n_p],
            sig2: vec![0.0; m],
            wsig: vec![0.0; m],
            wj: vec![0.0; m],
            zbuf: vec![],
            zs: vec![],
            pz: vec![],
            gain: vec![],
            zb: vec![],
            phi: vec![],
            gbuf: vec![],
            qbuf: vec![],
            sumbuf: Vec::with_capacity(m),
            noise: vec![],
            live: vec![],
            map_a: vec![],
            map_c: vec![],
            map_fresh: false,
            ubuf: vec![],
            vbuf: vec![],
            cfg,
        };
        model.ensure_buffers();
        Ok(model)
    }

    pub fn cfg(&self) -> &KalmanCfg {
        &self.cfg
    }

    pub fn sigma2(&self) -> &[f64] {
        &self.sig2
    }

    /// Variance of the *last* prediction, per target: `zᵀ P_j z + R_j` —
    /// parameter uncertainty plus observation noise (ENHANCEMENTS E12).
    ///
    /// `P` here is the covariance as it stands, so this is the **filtered**
    /// variance at the last regressor `z`, not the one-step-ahead predictive
    /// variance: that would carry `P` through the transition and add the
    /// process noise for the clock since the last observation, `zᵀ(Φ P Φᵀ +
    /// Q·D²)z + R`. The two differ by `zᵀ(Φ P Φᵀ − P + Q·D²)z`, which the
    /// default random walk (`Φ = I`) reduces to `Q·D²`: negligible under a
    /// half-life-derived `q` (`q/R = (ln2/h)²`, 0.005 % at half-life 100 and
    /// unit spacing, `D²` times that at spacing `D`), not under a large
    /// explicit `q` (review 2026-09-18, D2).
    ///
    /// This is the piece `sigma` alone cannot give. `sigma` is the spread of
    /// realized errors; this also knows how unsure the filter is about its own
    /// coefficients, so it is wide during warmup and after a gap, and narrows
    /// as evidence accumulates. Only Kalman tracks `P`, so only Kalman can
    /// report it exactly, under its model (the module doc).
    ///
    /// For the row `x`, standardized as `predict` standardizes it. It read
    /// the regressor of the last row stepped, a scratch a save does not
    /// keep, so a loaded filter answered for no regressor at all -- `R` alone
    /// (review 2026-09-12, V7).
    ///
    /// NaN until the target has a residual variance. The noise `step` uses
    /// before then is the row's own innovation squared, and a prediction is
    /// made before its row's target is seen, so it has no innovation to read
    /// (review 2026-10-05, CC4). Under `share_p` target `j`'s covariance is
    /// the shared `P` times `σ²_j / σ̄²` (the module doc, task 211), so this
    /// is `σ²_j (zᵀ P z / σ̄² + 1)`. After a row that observes nothing, `P`
    /// as it stands carries the noise of the clock since the last
    /// observation, `Q·D²`, as the next observation will charge it (the
    /// module doc).
    pub fn pred_var(&self, x: &[f64]) -> Vec<f64> {
        let k = self.cfg.k_total();
        let z = self.anchored(x);
        (0..self.cfg.n_targets)
            .map(|j| {
                let pi = if self.cfg.share_p { 0 } else { j };
                let p = self.covariance_now(pi);
                let p = &p[..];
                let mut quad = 0.0;
                for i in 0..k {
                    let row = i * k;
                    let mut acc = 0.0;
                    for jj in 0..k {
                        acc += p[row + jj] * z[jj];
                    }
                    quad += z[i] * acc;
                }
                match self.cfg.obs_var {
                    Some(r) => quad + r,
                    None if self.cfg.share_p => {
                        let mean = shared_noise(&mut Vec::new(), &self.sig2);
                        let own = self.sig2[j];
                        if own > 0.0 && mean > 0.0 {
                            own * (quad / mean + 1.0)
                        } else {
                            f64::NAN
                        }
                    }
                    None => {
                        let own = self.sig2[j];
                        if own > 0.0 { quad + own } else { f64::NAN }
                    }
                }
            })
            .collect()
    }

    pub fn n_eff(&self) -> f64 {
        self.stats.n_eff()
    }

    /// Feature scales of the moments as they stand: sd for features, 1 for
    /// the intercept slot. Zero-variance features get scale 1 (their
    /// standardized value is then their centered value, i.e. 0). All ones
    /// when `standardize` is off.
    fn scales(&self) -> Vec<f64> {
        (0..self.cfg.k_total())
            .map(|i| self.current_scale(i).0)
            .collect()
    }

    /// Slot `i`'s scale under the moments as they stand, and whether it is
    /// usable: a variance above 0 with an intercept, a raw second moment
    /// above 0 through the origin (review 2026-09-12, C10), and 1, usable,
    /// for the intercept and without `standardize`.
    fn current_scale(&self, i: usize) -> (f64, bool) {
        let off = usize::from(self.cfg.fit_intercept);
        if !self.cfg.standardize || i < off {
            (1.0, true)
        } else if off == 0 {
            // No intercept: nothing to centre on, so the scale is the raw
            // second moment's -- any positive one is usable, there being
            // no cancellation.
            let raw = self.stats.raw(i);
            if raw > 0.0 && raw.is_finite() {
                (raw.sqrt(), true)
            } else {
                (1.0, false)
            }
        } else {
            let v = self.stats.var(i);
            let raw = self.stats.raw(i);
            if crate::variance_is_usable(v, raw) {
                (v.sqrt(), true)
            } else {
                (1.0, false)
            }
        }
    }

    /// Coefficients in the ORIGINAL feature units, per target.
    ///
    /// Under `standardize` the filter's state lives in the anchor's
    /// coordinates, and it is read out through the anchor's means and
    /// scales: `c_i = b_i / s^a_i`, `c_0 = b_0 − Σ_i c_i m^a_i`. These are
    /// the coefficients `pred` uses, and they move only with the fit, to
    /// rounding: a move of the moments is a change of coordinates, which the
    /// state follows (the module doc; docs/PLAN.md tasks 206 and 211).
    pub fn coefficients(&self) -> Vec<Vec<f64>> {
        if !self.cfg.standardize {
            return self.beta.clone();
        }
        let k = self.cfg.k_total();
        let off = usize::from(self.cfg.fit_intercept);
        let mut out = Vec::with_capacity(self.cfg.n_targets);
        for b in &self.beta {
            let mut c = vec![0.0; k];
            for (i, ci) in c.iter_mut().enumerate().skip(off) {
                *ci = b[i] / self.anchor_scale[i];
            }
            if self.cfg.fit_intercept {
                // b0 is on centred features: unshift by the anchor's means.
                let mut b0 = b[0];
                for (i, ci) in c.iter().enumerate().skip(off) {
                    b0 -= ci * self.anchor_hi[i];
                    b0 -= ci * self.anchor_lo[i];
                }
                c[0] = b0;
            }
            out.push(c);
        }
        out
    }

    /// Process-noise variances for this row, `q_i = sigma^2 * (ln2 / h_i)^2`
    /// (steady-state gain matching with EW-RLS on standardized features).
    #[cfg(test)]
    fn q_vec(&self, sigma2: f64) -> Vec<f64> {
        let mut out = vec![0.0; self.cfg.k_total()];
        self.q_into(sigma2, &mut out);
        out
    }

    /// [`Self::q_vec`] into a caller's buffer of length `k_total`.
    fn q_into(&self, sigma2: f64, out: &mut [f64]) {
        if let Some(q) = &self.cfg.q {
            out.copy_from_slice(q);
            return;
        }
        let shared = self.cfg.half_life.len() == 1;
        for (i, qi) in out.iter_mut().enumerate() {
            let h = self.cfg.half_life[if shared { 0 } else { i }];
            *qi = if h.is_infinite() {
                0.0
            } else {
                let r = std::f64::consts::LN_2 / h;
                sigma2 * r * r
            };
        }
    }

    /// `[1, x]` standardized against the moments as they stand: the current
    /// coordinates (the module doc).
    fn standardized(&self, x: &[f64]) -> Vec<f64> {
        let off = usize::from(self.cfg.fit_intercept);
        (0..self.cfg.k_total())
            .map(|i| {
                let raw = if i < off { 1.0 } else { x[i - off] };
                if !self.cfg.standardize || i < off {
                    raw
                } else if off == 0 {
                    // No intercept to absorb a shift: scale only (C10).
                    raw / self.current_scale(i).0
                } else {
                    self.stats.deviation(i, raw) / self.current_scale(i).0
                }
            })
            .collect()
    }

    /// `[1, x]` in the anchor's coordinates, as `step` reads it: a slot that
    /// holds anything against the anchor, one that holds nothing against the
    /// moments as they stand (`step` sets such a slot's anchor to them before
    /// it holds anything; the module doc).
    fn anchored(&self, x: &[f64]) -> Vec<f64> {
        let off = usize::from(self.cfg.fit_intercept);
        let mut z = vec![1.0; self.cfg.k_total()];
        z[off..].copy_from_slice(x);
        let live: Vec<bool> = (0..z.len()).map(|i| self.is_live(i)).collect();
        let mut out = vec![0.0; z.len()];
        self.anchored_into(&z, &live, &mut out);
        out
    }

    /// The row `z` (the intercept's 1 included) in the anchor's coordinates
    /// into `out`, `live` saying which slots hold anything.
    fn anchored_into(&self, z: &[f64], live: &[bool], out: &mut [f64]) {
        let off = usize::from(self.cfg.fit_intercept);
        for (i, o) in out.iter_mut().enumerate() {
            *o = if !self.cfg.standardize || i < off {
                z[i]
            } else if live[i] {
                if off == 0 {
                    z[i] / self.anchor_scale[i]
                } else {
                    crate::comp::dev(z[i], self.anchor_hi[i], self.anchor_lo[i])
                        / self.anchor_scale[i]
                }
            } else if off == 0 {
                z[i] / self.current_scale(i).0
            } else {
                self.stats.deviation(i, z[i]) / self.current_scale(i).0
            };
        }
    }

    /// Whether slot `i` holds anything: a coefficient or a variance, in any
    /// target or covariance. A slot that holds nothing reads no coordinate.
    fn is_live(&self, i: usize) -> bool {
        let k = self.cfg.k_total();
        self.p.iter().any(|p| p[i * k + i] != 0.0) || self.beta.iter().any(|b| b[i] != 0.0)
    }

    fn mark_live(&mut self) {
        let mut live = std::mem::take(&mut self.live);
        for (i, l) in live.iter_mut().enumerate() {
            *l = self.is_live(i);
        }
        self.live = live;
    }

    /// The map from the anchor to the moments as they stand, `a_i = s_i /
    /// s^a_i` and `c_i = (m_i − m^a_i) / s^a_i` (0 without an intercept),
    /// into `a` and `c`, and the drift over the slots `live` marks, `max(a_i
    /// − 1, 1/a_i − 1, |c_i|)` (the module doc). The intercept's slot is 1
    /// and 0.
    fn map_into(&self, live: &[bool], a: &mut [f64], c: &mut [f64]) -> f64 {
        let off = usize::from(self.cfg.fit_intercept);
        let mut drift = 0.0f64;
        for i in 0..a.len() {
            if i < off || !self.cfg.standardize {
                a[i] = 1.0;
                c[i] = 0.0;
                continue;
            }
            let sa = self.anchor_scale[i];
            a[i] = self.current_scale(i).0 / sa;
            c[i] = if off == 1 {
                let (hi, lo) = self.stats.mean_pair(i);
                ((hi - self.anchor_hi[i]) + (lo - self.anchor_lo[i])) / sa
            } else {
                0.0
            };
            if live[i] {
                drift = drift.max(a[i] - 1.0).max(1.0 / a[i] - 1.0).max(c[i].abs());
            }
        }
        drift
    }

    /// [`Self::map_into`] into the kept map, marked as formed from the
    /// moments as they stand; returns the drift.
    fn form_map(&mut self) -> f64 {
        let (mut a, mut c) = (
            std::mem::take(&mut self.map_a),
            std::mem::take(&mut self.map_c),
        );
        let drift = self.map_into(&self.live, &mut a, &mut c);
        (self.map_a, self.map_c) = (a, c);
        self.map_fresh = true;
        drift
    }

    /// Slot `i`'s anchor set to the moments as they stand, and its map to
    /// 1 and 0.
    fn anchor_slot_here(&mut self, i: usize) {
        let off = usize::from(self.cfg.fit_intercept);
        if i < off {
            return;
        }
        (self.anchor_hi[i], self.anchor_lo[i]) = if off == 1 {
            self.stats.mean_pair(i)
        } else {
            (0.0, 0.0)
        };
        self.anchor_scale[i] = self.current_scale(i).0;
        self.map_a[i] = 1.0;
        self.map_c[i] = 0.0;
    }

    fn ensure_buffers(&mut self) {
        let k = self.cfg.k_total();
        if self.zbuf.len() != k {
            self.zbuf = vec![0.0; k];
            self.zs = vec![0.0; k];
            self.pz = vec![0.0; k];
            self.gain = vec![0.0; k];
            self.zb = vec![0.0; self.cfg.n_targets];
            self.phi = vec![1.0; k];
            self.gbuf = vec![0.0; k];
            self.qbuf = vec![0.0; k];
            self.noise = vec![0.0; self.cfg.n_targets];
            self.live = vec![false; k];
            self.map_a = vec![1.0; k];
            self.map_c = vec![0.0; k];
            self.map_fresh = false;
            self.ubuf = vec![vec![0.0; k]; self.p.len()];
            self.vbuf = vec![0.0; k];
        }
    }

    /// After a row that moved the moments, the drift from the anchor; past
    /// [`ANCHOR_DRIFT`], `b` and `P` re-mapped to the moments as they stand
    /// and the anchor set to them (the module doc). A re-map is refused whole
    /// where a factor `a_i` of a slot that holds anything is outside
    /// `[1/1024, 1024]` or a shift `|c_i|` above 1024 ([`REMAP_LIMIT`]), or
    /// where a number it would leave is not finite (each `b'`, `P'`'s
    /// diagonal and `P'_00`, which bound the rest of a positive semi-definite
    /// `P'`): `b` and `P` then keep their numbers, read in the new
    /// coordinates (docs/IMPROVEMENTS.md C2). The anchor moves either way. A
    /// slot that holds nothing is mapped by 1 and 0: its numbers are 0
    /// whatever its anchor.
    fn follow_the_moments(&mut self) {
        let drift = self.form_map();
        if drift <= ANCHOR_DRIFT {
            return;
        }
        let k = self.cfg.k_total();
        let off = usize::from(self.cfg.fit_intercept);
        let mut within = true;
        for i in off..k {
            if self.live[i] {
                let (a, c) = (self.map_a[i], self.map_c[i]);
                within &= (1.0 / REMAP_LIMIT..=REMAP_LIMIT).contains(&a) && c.abs() <= REMAP_LIMIT;
            } else {
                self.map_a[i] = 1.0;
                self.map_c[i] = 0.0;
            }
        }
        if within {
            self.remap();
        }
        for i in off..k {
            self.anchor_slot_here(i);
        }
    }

    /// `b <- Mᵀ b`, `P <- Mᵀ P M` from the kept map, unless a number it
    /// would leave is not finite (the module doc). In place, every check
    /// made before anything moves.
    fn remap(&mut self) {
        let k = self.cfg.k_total();
        let off = usize::from(self.cfg.fit_intercept);
        let (a, c) = (&self.map_a, &self.map_c);
        // First the numbers the check reads, formed as the re-map forms them:
        // each `b'`, each `P'`'s diagonal and, with an intercept, `u = (Mᵀ
        // P)_0·` and `P'_00` (kept in `u[0]`). `Mᵀ P M` is positive
        // semi-definite as `P` is, so its other entries are bounded by its
        // diagonal's.
        let mut finite = true;
        for b in &self.beta {
            let mut b0 = b[0];
            for i in off..k {
                finite &= (b[i] * a[i]).is_finite();
                b0 += c[i] * b[i];
            }
            finite &= off == 0 || b0.is_finite();
        }
        for (p, u) in self.p.iter().zip(self.ubuf.iter_mut()) {
            for i in off..k {
                finite &= (p[i * k + i] * (a[i] * a[i])).is_finite();
            }
            if off == 1 {
                // u = row 0 of Mᵀ P, the rows of P contiguous.
                u.copy_from_slice(&p[0..k]);
                for l in 1..k {
                    let cl = c[l];
                    for (uj, plj) in u.iter_mut().zip(&p[l * k..(l + 1) * k]) {
                        *uj += cl * plj;
                    }
                }
                let mut p00 = u[0];
                for j in 1..k {
                    p00 += c[j] * u[j];
                }
                u[0] = p00;
                finite &= p00.is_finite();
            }
        }
        if !finite {
            return;
        }
        for b in &mut self.beta {
            if off == 1 {
                let mut b0 = b[0];
                for i in 1..k {
                    b0 += c[i] * b[i];
                }
                b[0] = b0;
            }
            for (bi, ai) in b[off..].iter_mut().zip(&a[off..]) {
                *bi *= ai;
            }
        }
        for (p, u) in self.p.iter_mut().zip(&self.ubuf) {
            for i in off..k {
                let ai = a[i];
                let row = &mut p[i * k..(i + 1) * k];
                if off == 1 {
                    row[0] = u[i] * ai;
                }
                for (pij, aj) in row[off..].iter_mut().zip(&a[off..]) {
                    *pij *= ai * aj;
                }
            }
            if off == 1 {
                for j in 1..k {
                    p[j] = u[j] * a[j];
                }
                p[0] = u[0];
            }
        }
    }

    /// Whether covariance `pi` is unsized: all zero on its diagonal, as it
    /// is from the start until a row sizes the noise (the module doc, CC4).
    /// A sized `P` has a positive diagonal. One that the reversion has
    /// shrunk to exactly 0, every slot reverting over hundreds of its
    /// half-lives with no process noise to restore it, holds nothing, and is
    /// sized again.
    fn is_unsized(&self, pi: usize) -> bool {
        let k = self.cfg.k_total();
        let p = &self.p[pi];
        (0..k).all(|i| p[i * k + i] == 0.0)
    }

    /// `share_p`'s noise before any target has a residual variance: the
    /// mean of the row's squared innovations over the targets it observes
    /// (present, at a positive weight), each read from `zb`, before any
    /// update. A square that is not finite gives no scale and is left out;
    /// 0 when nothing is left, or every innovation was exactly 0 (CC4).
    fn shared_first_noise(&self, y: &[Option<f64>], weight: f64) -> f64 {
        if weight > 0.0 {
            let squares = y.iter().zip(&self.zb).filter_map(|(yj, zb)| {
                let e = (*yj)? - zb;
                Some(e * e).filter(|e2| e2.is_finite())
            });
            let mut buf = Vec::new();
            let sum = sum_ascending(&mut buf, squares);
            if !buf.is_empty() {
                return first_noise(sum / buf.len() as f64);
            }
        }
        0.0
    }

    /// The transition's coupling in the anchor's coordinates, `g_i = c_i
    /// (phi_0 − phi_i)` for each slot that holds anything, into `g`, from
    /// the map `c`: `None` where it is 0 throughout -- without
    /// `standardize` or an intercept, or where every slot reverts at one
    /// half-life (the module doc).
    fn coupling(&self, phi: &[f64], c: &[f64], live: &[bool], g: &mut [f64]) -> bool {
        if !self.cfg.standardize || !self.cfg.fit_intercept || self.cfg.revert_half_life.len() == 1
        {
            return false;
        }
        let mut any = false;
        g[0] = 0.0;
        for i in 1..g.len() {
            g[i] = if live[i] {
                c[i] * (phi[0] - phi[i])
            } else {
                0.0
            };
            any |= g[i] != 0.0;
        }
        any
    }

    /// `b <- Phi_a b`, `P <- Phi_a P Phi_aᵀ` for a clock delta `d_clock`,
    /// `Phi_a` the transition in the anchor's coordinates (the module doc):
    /// the coefficient means shrink toward zero and the covariance with
    /// them. A no-op (and skipped) under the default random walk.
    fn transition(&mut self, d_clock: f64) {
        if !self.cfg.reverts() {
            return;
        }
        let k = self.cfg.k_total();
        if self.cfg.revert_half_life.len() == 1 {
            // One half-life for every slot: one exponential, not `k`.
            self.phi.fill(self.cfg.phi(0, d_clock));
        } else {
            for (i, ph) in self.phi.iter_mut().enumerate() {
                *ph = self.cfg.phi(i, d_clock);
            }
        }
        let mut g = std::mem::take(&mut self.gbuf);
        let coupled = self.coupling(&self.phi, &self.map_c, &self.live, &mut g);
        for b in &mut self.beta {
            transit(b, &self.phi, coupled.then_some(&g[..]));
        }
        for p in &mut self.p {
            if coupled {
                // v = P g, before P moves; then P' = Phi P Phi + the coupling's
                // terms in the intercept's row and column (the module doc).
                for (i, v) in self.vbuf.iter_mut().enumerate() {
                    *v = p[i * k..(i + 1) * k]
                        .iter()
                        .zip(&g)
                        .map(|(pij, gj)| pij * gj)
                        .sum();
                }
            }
            for i in 0..k {
                let pi = self.phi[i];
                for (pij, pj) in p[i * k..(i + 1) * k].iter_mut().zip(&self.phi) {
                    *pij *= pi * pj;
                }
            }
            if coupled {
                let v = &self.vbuf;
                let gv: f64 = g.iter().zip(v).map(|(gi, vi)| gi * vi).sum();
                p[0] += 2.0 * self.phi[0] * v[0] + gv;
                for j in 1..k {
                    let add = self.phi[j] * v[j];
                    p[j] += add;
                    p[j * k] += add;
                }
            }
        }
        self.gbuf = g;
    }

    /// `Q D²` for the slots of covariance `pi` that are sized (a variance
    /// above 0), `D²` read on a reverting slot as its bounded form
    /// ([`KalmanCfg::gap_noise`]), defined in the current coordinates and
    /// carried into the anchor's (the module doc): an arrowhead, O(k).
    fn add_process_noise(&mut self, pi: usize, sigma2: f64, d: f64) {
        if d == 0.0 {
            return;
        }
        let k = self.cfg.k_total();
        let off = usize::from(self.cfg.fit_intercept);
        let mut q = std::mem::take(&mut self.qbuf);
        self.q_into(sigma2, &mut q);
        let p = &mut self.p[pi];
        let mut corner = 0.0;
        for i in 0..k {
            let qd = q[i] * self.cfg.gap_noise(i, d);
            if !self.cfg.standardize {
                // The whole `P` is sized at once (the caller saw to it).
                p[i * k + i] += qd;
                continue;
            }
            if p[i * k + i] == 0.0 || qd == 0.0 {
                continue;
            }
            if i < off {
                p[i * k + i] += qd;
                continue;
            }
            let a = self.map_a[i];
            if off == 1 {
                let v = self.map_c[i] / a;
                corner += v * v * qd;
                let o = -v * qd / a;
                p[i] += o;
                p[i * k] += o;
            }
            p[i * k + i] += qd / (a * a);
        }
        if corner != 0.0 {
            p[0] += corner;
        }
        self.qbuf = q;
    }

    /// Covariance `pi` as it stands after the last row: `P`, and after rows
    /// that observed nothing the noise of the clock since the last
    /// observation, `Q D²` on the sized slots, carried into the anchor's
    /// coordinates as the next observation will charge it (the module doc),
    /// at the noise the state holds (`obs_var`, else the residual variance;
    /// none before there is one). What `pred_var`, `se_coef` and the
    /// summary's readiness read: a coefficient's uncertainty grows across a
    /// gap in its target, as it did when the noise was charged per row.
    fn covariance_now(&self, pi: usize) -> std::borrow::Cow<'_, [f64]> {
        let d = self.elapsed[pi];
        let noise = self.readiness_noise(pi);
        // `readiness_noise` is a positive number, or NaN for none.
        if d == 0.0 || noise.is_nan() || self.is_unsized(pi) {
            return std::borrow::Cow::Borrowed(&self.p[pi]);
        }
        let k = self.cfg.k_total();
        let off = usize::from(self.cfg.fit_intercept);
        let mut q = vec![0.0; k];
        self.q_into(noise, &mut q);
        let live: Vec<bool> = (0..k).map(|i| self.is_live(i)).collect();
        let (mut a, mut c) = (vec![1.0; k], vec![0.0; k]);
        self.map_into(&live, &mut a, &mut c);
        let mut p = self.p[pi].clone();
        for i in 0..k {
            let qd = q[i] * self.cfg.gap_noise(i, d);
            if qd == 0.0 || (self.cfg.standardize && p[i * k + i] == 0.0) {
                continue;
            }
            if !self.cfg.standardize || i < off {
                p[i * k + i] += qd;
                continue;
            }
            if off == 1 {
                let v = c[i] / a[i];
                p[0] += v * v * qd;
                let o = -v * qd / a[i];
                p[i] += o;
                p[i * k] += o;
            }
            p[i * k + i] += qd / (a[i] * a[i]);
        }
        std::borrow::Cow::Owned(p)
    }

    /// The prior `p0 R` of each slot of covariance `pi` that is unsized and
    /// whose scale is usable, defined in the current coordinates and
    /// carried into the anchor's (the module doc). A slot that holds nothing
    /// anywhere has its anchor set to the moments as they stand first, so it
    /// takes its prior as it is.
    fn size_slots(&mut self, pi: usize, noise: f64) {
        let v = self.cfg.p0 * noise;
        if !(v > 0.0 && v.is_finite()) {
            return;
        }
        let k = self.cfg.k_total();
        let off = usize::from(self.cfg.fit_intercept);
        for i in 0..k {
            if self.p[pi][i * k + i] != 0.0 || !self.current_scale(i).1 {
                continue;
            }
            if i < off {
                self.p[pi][0] += v;
                self.live[0] = true;
                continue;
            }
            if !self.live[i] {
                self.anchor_slot_here(i);
            }
            let a = self.map_a[i];
            let p = &mut self.p[pi];
            if off == 1 {
                let cc = self.map_c[i] / a;
                p[0] += cc * cc * v;
                let o = -cc * v / a;
                p[i] += o;
                p[i * k] += o;
            }
            p[i * k + i] += v / (a * a);
            self.live[i] = true;
        }
    }

    /// The noise target `j`'s readiness is read against, as the state holds
    /// it before a row (the module doc): `obs_var`, else the residual
    /// variance -- under `share_p` the mean over the targets that have one,
    /// summed in ascending order as `step` sums it -- and NaN, none, before
    /// there is one. Never the row's own innovation, which reads its target.
    fn readiness_noise(&self, j: usize) -> f64 {
        if let Some(v) = self.cfg.obs_var {
            return v;
        }
        let s2 = if self.cfg.share_p {
            shared_noise(&mut Vec::new(), &self.sig2)
        } else {
            self.sig2[j]
        };
        if s2 > 0.0 && s2.is_finite() {
            s2
        } else {
            f64::NAN
        }
    }

    /// `z' P⁻ z` against covariance `pi` for the row `x`, `d_clock` after the
    /// last row, at the noise `r` the process noise is derived from: `P`
    /// carried through the transition and the process noise the
    /// observation would add for the clock since the last one, `D + d`,
    /// without moving it -- `wᵀ P w + (D + d)² Σ_i q_i z_i²` over the sized
    /// slots, with `w = Phi_aᵀ z^a` and `z` the row in the current
    /// coordinates, which is `step`'s `P <- Phi_a P Phi_aᵀ + Q_a D²` read
    /// through `z^a`.
    fn prior_quad(&self, pi: usize, x: &[f64], d_clock: f64, r: f64) -> f64 {
        let k = self.cfg.k_total();
        let p = &self.p[pi];
        let za = self.anchored(x);
        let w: Vec<f64> = if self.cfg.reverts() {
            let phi: Vec<f64> = (0..k).map(|i| self.cfg.phi(i, d_clock)).collect();
            let live: Vec<bool> = (0..k).map(|i| self.is_live(i)).collect();
            let (mut a, mut c, mut g) = (vec![1.0; k], vec![0.0; k], vec![0.0; k]);
            self.map_into(&live, &mut a, &mut c);
            let coupled = self.coupling(&phi, &c, &live, &mut g);
            // w = Phi_aᵀ z^a: w_0 = phi_0 z_0, w_i = g_i z_0 + phi_i z_i.
            (0..k)
                .map(|i| {
                    if coupled && i > 0 {
                        g[i] * za[0] + phi[i] * za[i]
                    } else {
                        phi[i] * za[i]
                    }
                })
                .collect()
        } else {
            za
        };
        let mut quad = 0.0;
        for i in 0..k {
            let row = i * k;
            let mut acc = 0.0;
            for jj in 0..k {
                acc += p[row + jj] * w[jj];
            }
            quad += w[i] * acc;
        }
        let z = self.standardized(x);
        let mut q = vec![0.0; k];
        self.q_into(r, &mut q);
        let gap = self.elapsed[pi] + d_clock;
        let dd = gap.powi(2);
        let sized = |i: &usize| p[i * k + i] != 0.0;
        let walking = |i: &usize| self.cfg.revert_of(*i).is_infinite();
        let noise: f64 = (0..k)
            .filter(sized)
            .filter(walking)
            .map(|i| q[i] * z[i] * z[i])
            .sum();
        // A reverting slot's noise is bounded in the gap (`gap_noise`), so
        // it is summed apart; with none, the sum is the random walk's alone,
        // to the bit.
        let reverting: f64 = (0..k)
            .filter(sized)
            .filter(|i| !walking(i))
            .map(|i| q[i] * self.cfg.gap_noise(i, gap) * z[i] * z[i])
            .sum();
        quad + dd * noise + reverting
    }

    /// `sqrt(1 + z' P⁻ z / R)` per target for the row `x` at `d_clock`
    /// (the module doc): infinite while `P` is unsized or the noise is
    /// none, NaN where a feature is not a number.
    fn row_inflation_into(&self, x: &[f64], d_clock: f64, out: &mut Vec<f64>) {
        let m = self.cfg.n_targets;
        out.clear();
        out.resize(m, f64::INFINITY);
        for (j, o) in out.iter_mut().enumerate() {
            let pi = if self.cfg.share_p { 0 } else { j };
            let r = self.readiness_noise(j);
            if self.is_unsized(pi) || r.is_nan() {
                continue;
            }
            *o = (1.0 + self.prior_quad(pi, x, d_clock, r) / r).sqrt();
        }
    }

    /// `E[z_i²]` per slot over the rows the standardizer has seen, as `z`
    /// is formed from them in the current coordinates: 1 for the intercept,
    /// `var / s²` centred and `raw / s²` through the origin, the raw second
    /// moment unstandardized.
    fn design_second_moments(&self) -> Vec<f64> {
        let off = usize::from(self.cfg.fit_intercept);
        let s = self.scales();
        (0..self.cfg.k_total())
            .map(|i| {
                if i < off {
                    1.0
                } else if !self.cfg.standardize {
                    self.stats.raw(i)
                } else if off == 0 {
                    self.stats.raw(i) / (s[i] * s[i])
                } else {
                    self.stats.var(i) / (s[i] * s[i])
                }
            })
            .collect()
    }

    /// The diagonal of covariance `pi` as it stands
    /// ([`Self::covariance_now`]) in the current coordinates, `Mᵀ P M` (the
    /// module doc): `a_i² P_ii`, and with an intercept `u_0 + Σ_j c_j u_j`
    /// with `u = P_0· + Σ_l c_l P_l·`.
    fn current_diagonal(&self, pi: usize) -> Vec<f64> {
        let k = self.cfg.k_total();
        let p = self.covariance_now(pi);
        let p = &p[..];
        if !self.cfg.standardize {
            return (0..k).map(|i| p[i * k + i]).collect();
        }
        let off = usize::from(self.cfg.fit_intercept);
        let live: Vec<bool> = (0..k).map(|i| self.is_live(i)).collect();
        let (mut a, mut c) = (vec![1.0; k], vec![0.0; k]);
        self.map_into(&live, &mut a, &mut c);
        let mut out: Vec<f64> = (0..k).map(|i| a[i] * a[i] * p[i * k + i]).collect();
        if off == 1 {
            let mut u = p[0..k].to_vec();
            for l in 1..k {
                for (uj, plj) in u.iter_mut().zip(&p[l * k..(l + 1) * k]) {
                    *uj += c[l] * plj;
                }
            }
            out[0] = u[0] + (1..k).map(|j| c[j] * u[j]).sum::<f64>();
        }
        out
    }

    /// `T P Tᵀ`'s diagonal for covariance `pi` as it stands
    /// ([`Self::covariance_now`]): the variance of each coefficient as
    /// `coefficients` reads it out (the module doc).
    fn coef_variance_of(&self, pi: usize) -> Vec<f64> {
        let k = self.cfg.k_total();
        let p = self.covariance_now(pi);
        let p = &p[..];
        if !self.cfg.standardize {
            return (0..k).map(|i| p[i * k + i]).collect();
        }
        let off = usize::from(self.cfg.fit_intercept);
        let s = &self.anchor_scale;
        let mut out: Vec<f64> = (0..k)
            .map(|i| {
                if i < off {
                    p[i * k + i]
                } else {
                    p[i * k + i] / (s[i] * s[i])
                }
            })
            .collect();
        if off == 1 {
            // c_0 = b_0 − Σ_i b_i m^a_i / s^a_i: v' P v with v = [1, −m^a / s^a].
            let v: Vec<f64> = (0..k)
                .map(|i| {
                    if i == 0 {
                        1.0
                    } else {
                        -(self.anchor_hi[i] + self.anchor_lo[i]) / s[i]
                    }
                })
                .collect();
            let mut quad = 0.0;
            for i in 0..k {
                let mut acc = 0.0;
                for jj in 0..k {
                    acc += p[i * k + jj] * v[jj];
                }
                quad += v[i] * acc;
            }
            out[0] = quad.max(0.0);
        }
        out
    }

    /// Under `share_p`, target `j`'s covariance is the shared one times
    /// `σ²_j / σ̄²` (the module doc, task 211): that ratio, 1 with `obs_var`,
    /// unshared, or while the target has no residual variance.
    fn own_noise_ratio(&self, j: usize) -> f64 {
        if !self.cfg.share_p || self.cfg.obs_var.is_some() {
            return 1.0;
        }
        let mean = shared_noise(&mut Vec::new(), &self.sig2);
        let own = self.sig2[j];
        if own > 0.0 && mean > 0.0 {
            own / mean
        } else {
            1.0
        }
    }
}

/// `b <- Phi_a b` in place (the module doc): `b_i <- phi_i b_i`, and with
/// the coupling `g`, `b_0 <- phi_0 b_0 + Σ_i g_i b_i` from the `b_i` before
/// the row. `step` and `predict` both run it, so the two agree to the bit.
fn transit(b: &mut [f64], phi: &[f64], g: Option<&[f64]>) {
    if let Some(g) = g {
        let mut b0 = b[0] * phi[0];
        for i in 1..b.len() {
            b0 += g[i] * b[i];
        }
        for (bi, ph) in b.iter_mut().zip(phi).skip(1) {
            *bi *= ph;
        }
        b[0] = b0;
    } else {
        for (bi, ph) in b.iter_mut().zip(phi) {
            *bi *= ph;
        }
    }
}

/// How far one re-map may move the coordinates for `b` and `P` to follow
/// (the module doc): a scale by this factor either way, a mean by this many
/// of the anchor's scale. `Mᵀ P M` puts `c² P_ii` into the intercept's
/// variance and reads it back through a cancellation on every later row, so
/// a shift of `c` costs `c²` rounding steps of `P`'s narrowest direction:
/// 2.3e-10 of it at 1024, and all of it at the input bound's 1e99, where the
/// re-map left a slope's variance at −3.6e149 and the filter never
/// recovered (`tests/model_contract.rs`, `kalman_recovers_from_bounded_extremes`,
/// the reverting filter). A row of ordinary data moves a mean by a fraction
/// of a scale: past this it is thousands of standard deviations out.
const REMAP_LIMIT: f64 = 1024.0;

/// A prediction, or none (NaN) when it is not a number. A feature at the
/// input bound, standardized against a scale the earlier rows set, times its
/// coefficient can overflow `z . beta` to `inf`, and an infinite prediction
/// is no forecast: `step` and `predict` both withhold it, as `step` already
/// skips the update such a row would poison (docs/IMPROVEMENTS.md C2). Found
/// by the generated stream in `tests/model_contract.rs` (docs/PLAN.md
/// task 158).
fn a_number_or_none(v: f64) -> f64 {
    if v.is_finite() { v } else { f64::NAN }
}

/// A squared innovation, or a mean of them, as the noise a target takes
/// before it has a residual variance (the module doc, CC4): itself, or 0 --
/// no noise -- where it gives no scale. That is an innovation of exactly 0,
/// or one whose square underflows to 0, and a square that is not finite.
fn first_noise(e2: f64) -> f64 {
    if e2.is_finite() { e2 } else { 0.0 }
}

/// The sum of `vals` in ascending order, through `buf`: the same bits
/// whatever order the targets come in, as `share_p`'s mean noise must be
/// (the module doc).
fn sum_ascending(buf: &mut Vec<f64>, vals: impl Iterator<Item = f64>) -> f64 {
    buf.clear();
    buf.extend(vals);
    buf.sort_unstable_by(f64::total_cmp);
    buf.iter().sum()
}

/// `share_p`'s noise once a target has a residual variance: the mean `σ²`
/// over the targets that have one (`σ² > 0`), summed in ascending order
/// through `buf`; 0, no noise yet, where none has (the module doc). A target
/// with no residual variance has no estimate of the noise, not a noise of 0:
/// counted in the mean, a target null so far halved the noise its neighbour
/// was weighed against (review round 5, A1). `shared_first_noise` averages
/// over the targets the row observes the same way.
fn shared_noise(buf: &mut Vec<f64>, sig2: &[f64]) -> f64 {
    let sum = sum_ascending(buf, sig2.iter().copied().filter(|s2| *s2 > 0.0));
    if buf.is_empty() {
        0.0
    } else {
        sum / buf.len() as f64
    }
}

/// `P <- P - g (P z)ᵀ`, once per pair and written to both halves, so `P`
/// stays symmetric (see `Rls::step` for why that matters).
fn take_the_row(p: &mut [f64], gain: &[f64], pz: &[f64]) {
    let k = gain.len();
    for i in 0..k {
        let gi = gain[i];
        for jj in i..k {
            let v = gi * pz[jj];
            p[i * k + jj] -= v;
            if jj != i {
                p[jj * k + i] -= v;
            }
        }
    }
}

impl OnlineModel for Kalman {
    fn target_n_eff_into(&self, out: &mut Vec<f64>) -> bool {
        out.clear();
        out.extend_from_slice(&self.wj);
        true
    }

    /// Without a row: the per-row statistic's mean field over the design at
    /// the posterior, `sqrt(1 + Σ_i P_ii E[z_i²] / R)` in the current
    /// coordinates (the module doc).
    fn error_inflation_into(&self, out: &mut Vec<f64>) -> bool {
        let m = self.cfg.n_targets;
        out.clear();
        out.resize(m, f64::INFINITY);
        let ez2 = self.design_second_moments();
        for (j, o) in out.iter_mut().enumerate() {
            let pi = if self.cfg.share_p { 0 } else { j };
            let r = self.readiness_noise(j);
            if self.is_unsized(pi) || r.is_nan() {
                continue;
            }
            let diag = self.current_diagonal(pi);
            let trace: f64 = diag.iter().zip(&ez2).map(|(p, e)| p * e).sum();
            *o = (1.0 + trace / r).sqrt();
        }
        true
    }

    /// The gate reads the row's own `sqrt(1 + z' P⁻ z / R)`: exact, `O(k²)`.
    fn error_inflation_gate_into(
        &self,
        x: &[f64],
        d_clock: f64,
        out: &mut Vec<f64>,
        _limit: f64,
    ) -> bool {
        self.row_inflation_into(x, d_clock, out);
        true
    }

    fn row_error_inflation_into(&self, x: &[f64], d_clock: f64, out: &mut Vec<f64>) -> bool {
        self.row_inflation_into(x, d_clock, out);
        true
    }

    /// `T P Tᵀ`'s diagonal per target, absolute: `P` carries the noise, and
    /// under `share_p` target `j`'s is the shared one times `σ²_j / σ̄²`
    /// (the module doc). NaN for a target whose `P` is unsized.
    fn coef_variance(&self) -> Option<crate::CoefVariance> {
        let (m, k) = (self.cfg.n_targets, self.cfg.k_total());
        let per = (0..m)
            .map(|j| {
                let pi = if self.cfg.share_p { 0 } else { j };
                if self.is_unsized(pi) {
                    vec![f64::NAN; k]
                } else {
                    let ratio = self.own_noise_ratio(j);
                    self.coef_variance_of(pi)
                        .into_iter()
                        .map(|v| v * ratio)
                        .collect()
                }
            })
            .collect();
        Some(crate::CoefVariance::Absolute(per))
    }

    fn step(&mut self, x: &[f64], y: &[Option<f64>], d_clock: f64, weight: f64) -> Step {
        // A value that is not usable, by the rule every model keeps
        // (`OnlineModel`): a feature that is not a number reached the
        // standardiser's moments (docs/PLAN.md task 183).
        if let Some(refused) = crate::model::refused_step(self, x, y, d_clock, weight) {
            return refused;
        }
        self.ensure_buffers();
        let k = self.cfg.k_total();
        let m = self.cfg.n_targets;
        let lam = self.cfg.decay.factor(d_clock);

        if self.cfg.fit_intercept {
            self.zbuf[0] = 1.0;
            self.zbuf[1..].copy_from_slice(x);
        } else {
            self.zbuf.copy_from_slice(x);
        }

        // The clock each covariance has gone without an observation: every
        // row adds to it, and a row that observes the covariance's target
        // charges it and clears it, below (the module doc, task 211).
        for e in &mut self.elapsed {
            *e += d_clock;
        }

        // Which slots hold anything, and the map from the anchor to the
        // moments as they stand: formed after the last row that moved them,
        // or now after a load.
        if self.cfg.standardize {
            self.mark_live();
            if !self.map_fresh {
                self.form_map();
            }
        }

        // The clock has moved by `d_clock` since the last row: the state
        // is propagated before it predicts (`predict` does the same).
        self.transition(d_clock);

        // The row in the anchor's coordinates, read against the moments
        // BEFORE this row's update.
        let mut zs = std::mem::take(&mut self.zs);
        self.anchored_into(&self.zbuf, &self.live, &mut zs);
        self.zs = zs;

        // ---- predict (state before the update) ----
        // `z . b_j` per target, once, before any target's update: the
        // prediction and the innovation both read it, and so does the noise
        // `share_p` takes before its first residual, from every target's.
        for (zb, b) in self.zb.iter_mut().zip(&self.beta) {
            *zb = self.zs.iter().zip(b).map(|(z, b)| z * b).sum();
        }
        let n_eff = self.stats.n_eff();
        let ready = n_eff >= self.cfg.min_weight;
        let mut pred = vec![f64::NAN; m];
        if ready {
            for (j, p) in pred.iter_mut().enumerate() {
                if self.wj[j] > 0.0 {
                    *p = a_number_or_none(self.zb[j]);
                }
            }
        }

        // A null target, or a present one at weight zero -- an observation
        // of infinite variance, `σ²/0` -- is a prediction step and no
        // update, and time passes for both weights alike. The zero weight
        // skipped the decay, so `σ²`, which sets `R` and `Q`, forgot less
        // across it than across a null (review 2026-09-12, S9).
        let obs = |j: usize| y[j].filter(|_| weight > 0.0);

        // ---- the noise per target ----
        // `R`, and the `σ²` the process noise is derived from: `obs_var`,
        // else the residual variance, else -- before there is one -- the
        // row's own innovation squared (CC4). 0 is no noise to weigh the
        // row against: it corrects nothing and adds no derived `Q`. Under
        // `share_p`, the mean residual variance over the targets that have
        // one, as the row arrives, read once before any target's update
        // moves its own (review round 4, CC3), summed in ascending order so
        // the order of `targets` cannot move a bit of it (task 204), over
        // the targets with a variance (review round 5, A1); before any has
        // one, the mean squared innovation over the targets the row observes.
        let shared_s2 = if self.cfg.share_p {
            let mut buf = std::mem::take(&mut self.sumbuf);
            let s2 = shared_noise(&mut buf, &self.sig2);
            self.sumbuf = buf;
            s2
        } else {
            f64::NAN
        };
        let mut shared_first: Option<f64> = None;
        for j in 0..m {
            let sigma2 = match self.cfg.obs_var {
                Some(v) => v,
                None => {
                    let s2 = if self.cfg.share_p {
                        shared_s2
                    } else {
                        self.sig2[j]
                    };
                    if s2 > 0.0 {
                        s2
                    } else if self.cfg.share_p {
                        *shared_first.get_or_insert_with(|| self.shared_first_noise(y, weight))
                    } else {
                        obs(j).map_or(0.0, |yj| {
                            let e = yj - self.zb[j];
                            first_noise(e * e)
                        })
                    }
                }
            };
            self.noise[j] = sigma2;
        }

        // ---- the squared innovations a standardized prior is sized from ----
        // Under `share_p` the row's squares over the targets it observes,
        // summed in ascending order, so the order of `targets` moves no bit.
        if self.cfg.standardize && self.cfg.obs_var.is_none() {
            let square = |j: usize| {
                let e = obs(j)? - self.zb[j];
                Some(e * e).filter(|e2| *e2 > 0.0 && e2.is_finite())
            };
            if self.cfg.share_p {
                let mut buf = std::mem::take(&mut self.sumbuf);
                let sum = sum_ascending(&mut buf, (0..m).filter_map(square));
                let n = buf.len() as f64;
                self.sumbuf = buf;
                if n > 0.0 {
                    self.basis[0].add(sum, n, weight);
                }
            } else {
                for j in 0..m {
                    if let Some(e2) = square(j) {
                        self.basis[j].add(e2, 1.0, weight);
                    }
                }
            }
        }

        // ---- process noise and priors, per covariance ----
        // Only on a row that observes the covariance's target (any target,
        // under `share_p`): `Q D²` for the whole clock since the last such
        // row, which a split of it into rows does not move (the module doc,
        // task 211). The noise is defined in the current coordinates, the
        // prior on a slot is set once its scale is usable, and an unsized
        // slot takes no noise (CC4).
        let n_p = self.p.len();
        for pi in 0..n_p {
            let informs = if self.cfg.share_p {
                (0..m).any(|j| obs(j).is_some())
            } else {
                obs(pi).is_some()
            };
            if !informs {
                continue;
            }
            let gap = self.elapsed[pi];
            self.elapsed[pi] = 0.0;
            let sigma2 = self.noise[if self.cfg.share_p { 0 } else { pi }];
            if self.cfg.standardize {
                self.add_process_noise(pi, sigma2, gap);
                if let Some(r) = self.cfg.obs_var.or_else(|| self.basis[pi].noise()) {
                    self.size_slots(pi, r);
                }
            } else if self.is_unsized(pi) {
                // The first row with a noise sizes the prior, to `p0` times
                // that noise, before its gain, and adds no process noise.
                let v = self.cfg.p0 * sigma2;
                if v > 0.0 && v.is_finite() {
                    let p = &mut self.p[pi];
                    p.fill(0.0);
                    for i in 0..k {
                        p[i * k + i] = v;
                    }
                }
            } else {
                self.add_process_noise(pi, sigma2, gap);
            }
        }

        // ---- Kalman update per target ----
        // Under `share_p`, `P z` is read once, from `P` as the row finds it,
        // and `P` takes the row once, after every target's update (the
        // module doc, task 204).
        let mut shared_pz = false;
        let mut shared_update = false;
        for (j, &pred_j) in pred.iter().enumerate() {
            let pi = if self.cfg.share_p { 0 } else { j };
            let Some(yj) = obs(j) else {
                self.wj[j] *= lam;
                self.wsig[j] *= lam;
                continue;
            };
            let sigma2 = self.noise[j];
            // pz = P z
            if !shared_pz {
                let p = &self.p[pi];
                for i in 0..k {
                    let row = i * k;
                    let mut acc = 0.0;
                    for jj in 0..k {
                        acc += p[row + jj] * self.zs[jj];
                    }
                    self.pz[i] = acc;
                }
                shared_pz = self.cfg.share_p;
            }
            let zpz: f64 = self.zs.iter().zip(&self.pz).map(|(z, p)| z * p).sum();
            let s_inn = zpz + sigma2 / weight;
            let err = yj - self.zb[j];
            // A standardized regressor can be ~1e200 when a feature at the
            // input bound follows a run at a tiny scale, and then `z P z` or
            // `z . beta` overflows. The row is skipped rather than let an
            // `inf` gain or an `inf/inf` NaN into `beta` and `P`, which no
            // later row would repair (docs/IMPROVEMENTS.md C2). So is a row
            // with no noise yet to weigh it against: at `R = 0` the update
            // would take it as exact and collapse `P` along `z` (CC4).
            if sigma2 > 0.0 && s_inn > 0.0 && s_inn.is_finite() && err.is_finite() {
                for i in 0..k {
                    self.gain[i] = self.pz[i] / s_inn;
                }
                for (b, g) in self.beta[j].iter_mut().zip(&self.gain) {
                    *b += g * err;
                }
                if self.cfg.share_p {
                    shared_update = true;
                } else {
                    take_the_row(&mut self.p[pi], &self.gain, &self.pz);
                }
            }
            // EW residual variance from the out-of-sample prediction. Its
            // weight ages on every row, this one included, and the row adds
            // its squared residual when it has a prediction to measure one
            // from: a row with no prediction -- `min_weight` unmet after a
            // clock gap -- aged nothing, so `σ²` forgot less across it than
            // across a null (N6). The update is skipped when it would not be
            // finite: `sig2` feeds the process noise, and an `inf` there puts
            // `inf` on the diagonal of `P` and a NaN in every later gain.
            let aged = lam * self.wsig[j];
            self.wsig[j] = aged;
            if pred_j.is_finite() {
                let resid = yj - pred_j;
                let ws_new = aged + weight;
                let s2 = (aged * self.sig2[j] + weight * resid * resid) / ws_new;
                if s2.is_finite() {
                    self.sig2[j] = s2;
                    self.wsig[j] = ws_new;
                }
            }
            self.wj[j] = lam * self.wj[j] + weight;
        }
        // Every target that updated read the same `P z` and the same noise,
        // so the same gain: `P` takes the row once.
        if shared_update {
            take_the_row(&mut self.p[0], &self.gain, &self.pz);
        }

        // Standardization stats update last, so this row's z used the prior
        // stats; a row that moved them is followed by the drift check, and
        // past it the re-map (the module doc). A row of weight 0 moves no
        // moment, and the map stays the one formed after the last row that
        // did.
        self.stats.update(&self.zbuf, lam, weight);
        if self.cfg.standardize && weight > 0.0 {
            self.follow_the_moments();
        }

        Step {
            pred,
            n_eff,
            extra: None,
        }
    }

    fn predict(&self, x: &[f64], d_clock: f64) -> Step {
        if let Some(refused) = crate::model::refused_predict(self, x, d_clock) {
            return refused;
        }
        let m = self.cfg.n_targets;
        let k = self.cfg.k_total();
        let n_eff = self.stats.n_eff();
        let mut pred = vec![f64::NAN; m];
        if n_eff >= self.cfg.min_weight {
            let zs = self.anchored(x);
            // The same numbers `step` would emit: its transition moves `b`
            // by `transit` before the dot product.
            let reverts = self.cfg.reverts();
            let (phi, g, coupled) = if reverts {
                let phi: Vec<f64> = (0..k).map(|i| self.cfg.phi(i, d_clock)).collect();
                let live: Vec<bool> = (0..k).map(|i| self.is_live(i)).collect();
                let (mut a, mut c, mut g) = (vec![1.0; k], vec![0.0; k], vec![0.0; k]);
                self.map_into(&live, &mut a, &mut c);
                let coupled = self.coupling(&phi, &c, &live, &mut g);
                (phi, g, coupled)
            } else {
                (vec![], vec![], false)
            };
            for (j, p) in pred.iter_mut().enumerate() {
                if self.wj[j] > 0.0 {
                    let v = if reverts {
                        let mut b = self.beta[j].clone();
                        transit(&mut b, &phi, coupled.then_some(&g[..]));
                        zs.iter().zip(&b).map(|(z, b)| z * b).sum()
                    } else {
                        zs.iter().zip(&self.beta[j]).map(|(z, b)| z * b).sum()
                    };
                    *p = a_number_or_none(v);
                }
            }
        }
        Step {
            pred,
            n_eff,
            extra: None,
        }
    }

    fn state(&self) -> State {
        State::new(ModelState::Kalman(Box::new(self.clone())))
    }

    fn restore(s: &State) -> Result<Self, StateError> {
        check_schema(s)?;
        match &s.model {
            ModelState::Kalman(m) => {
                let mut m = (**m).clone();
                // The shapes are checked as the state is read (`KalmanV4`).
                crate::model::check_cfg("kalman", m.cfg.validate())?;
                m.ensure_buffers();
                Ok(m)
            }
            other => Err(StateError::WrongModel {
                expected: "kalman",
                found: other.kind(),
            }),
        }
    }

    fn n_targets(&self) -> usize {
        self.cfg.n_targets
    }

    fn n_features(&self) -> usize {
        self.cfg.n_features
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }

    fn cfg(k: usize, m: usize, hl: Vec<f64>) -> KalmanCfg {
        KalmanCfg {
            n_features: k,
            n_targets: m,
            fit_intercept: true,
            decay: Decay::Halflife(200.0),
            half_life: hl,
            q: None,
            obs_var: None,
            p0: 1.0,
            share_p: false,
            min_weight: 10.0,
            revert_half_life: vec![f64::INFINITY],
            standardize: true,
        }
    }

    /// Feed a deterministic stream, returning the fitted filter.
    fn fit(cfg: KalmanCfg, n: usize, seed: u64) -> Kalman {
        let m = cfg.n_targets;
        let mut model = Kalman::new(cfg).unwrap();
        let mut s = seed;
        for i in 0..n {
            let x = [lcg(&mut s), 0.5 + lcg(&mut s)];
            let ys: Vec<Option<f64>> = (0..m)
                .map(|j| Some((j as f64 + 1.0) * (2.0 * x[0] - x[1]) + 0.1 * lcg(&mut s)))
                .collect();
            model.step(&x, &ys, if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        model
    }

    #[test]
    fn cfg_validation_rejects_each_bad_field() {
        let bad = |f: &dyn Fn(&mut KalmanCfg), want: &str| {
            let mut c = cfg(2, 1, vec![100.0]);
            f(&mut c);
            match c.validate() {
                Err(e) => assert!(e.contains(want), "wanted {want:?}, got {e:?}"),
                Ok(()) => panic!("expected rejection mentioning {want:?}"),
            }
        };
        let good = |f: &dyn Fn(&mut KalmanCfg)| {
            let mut c = cfg(2, 1, vec![100.0]);
            f(&mut c);
            c.validate().expect("should be accepted");
        };

        bad(&|c| c.n_features = 0, "must be >= 1");
        bad(&|c| c.n_targets = 0, "must be >= 1");

        // `q` is the process noise per slot: one entry per coefficient,
        // including the intercept, and zero means "pinned".
        bad(&|c| c.q = Some(vec![0.0; 2]), "length 3");
        bad(
            &|c| c.q = Some(vec![0.0, 0.0, -1e-9]),
            "q values must be finite and >= 0",
        );
        good(&|c| c.q = Some(vec![0.0; 3]));

        // Without `q`, the half-lives are broadcast: one value, or one per slot.
        bad(&|c| c.half_life = vec![1.0, 2.0], "length 1 or 3");
        bad(&|c| c.half_life = vec![0.0], "must be > 0");
        bad(&|c| c.half_life = vec![-1.0], "must be > 0");
        good(&|c| c.half_life = vec![f64::INFINITY]);
        good(&|c| c.half_life = vec![1.0, 2.0, 3.0]);

        // p0 is the prior variance and obs_var the measurement noise; both
        // divide, so neither may be zero. obs_var may be absent (inferred).
        bad(&|c| c.p0 = 0.0, "p0 must be finite and > 0");
        bad(&|c| c.p0 = -1.0, "p0 must be finite and > 0");
        bad(&|c| c.obs_var = Some(0.0), "obs_var must be finite and > 0");
        bad(
            &|c| c.obs_var = Some(-1.0),
            "obs_var must be finite and > 0",
        );
        good(&|c| c.obs_var = None);
        good(&|c| c.obs_var = Some(1e-9));
        // NaN passed every `<= 0.0` / `< 0.0` test here. A NaN `obs_var` was
        // the silent one: `s_inn` is NaN on every row, the update's guard is
        // never met, and the filter predicts its prior for the life of the
        // stream with no error and no counted failure (review 2026-09-18, B4).
        bad(
            &|c| c.obs_var = Some(f64::NAN),
            "obs_var must be finite and > 0",
        );
        bad(&|c| c.p0 = f64::NAN, "p0 must be finite and > 0");
        bad(&|c| c.half_life = vec![f64::NAN], "must be > 0");
        bad(
            &|c| c.q = Some(vec![0.0, f64::NAN, 0.0]),
            "q values must be finite and >= 0",
        );

        cfg(2, 1, vec![100.0]).validate().unwrap();
    }

    #[test]
    fn standardize_defaults_to_on_when_a_state_file_omits_it() {
        // `#[serde(default = "default_true")]`: a state written before the
        // field existed must load with standardization on, which is the
        // behaviour that state was produced under. Defaulting to `false`
        // instead would silently change every restored model's numbers.
        let json = r#"{
            "n_features": 2, "n_targets": 1, "fit_intercept": true,
            "decay": {"Halflife": 200.0}, "half_life": [100.0], "q": null,
            "obs_var": null, "p0": 1.0, "share_p": false, "min_weight": 10.0
        }"#;
        let cfg: KalmanCfg = serde_json::from_str(json).expect("should load without the field");
        assert!(cfg.standardize, "the omitted field must default to true");
    }

    #[test]
    fn coefficients_are_reported_in_the_callers_units() {
        // The filter works on standardized, centered features; `coefficients`
        // has to undo both -- divide by the scale, then unshift the intercept
        // by the feature means -- or the numbers a caller reads are not the
        // ones their data is in.
        // A coefficient half-life rather than a pinned one, so the filter keeps
        // re-learning as the standardization stats settle. With `q = 0` and a
        // near-zero observation noise it would instead converge in a handful of
        // rows, locking its betas into the standardized space of the first few
        // rows while `coefficients` unscales with the current stats -- which is
        // why the Bayesian-regression correspondence test turns standardization
        // off rather than working around it.
        let mut c = cfg(2, 1, vec![500.0]);
        c.min_weight = 3.0;
        let mut m = Kalman::new(c).unwrap();
        let mut s = 149u64;
        // Features on very different scales and far from zero, so a missing
        // unscale or a missing unshift is unmistakable.
        for i in 0..20_000 {
            let x = [500.0 + 10.0 * lcg(&mut s), 0.01 * lcg(&mut s)];
            let y = 12.0 + 0.25 * x[0] - 800.0 * x[1];
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let b = &m.coefficients()[0];
        assert!((b[1] - 0.25).abs() < 0.02, "slope 0: {}", b[1]);
        assert!((b[2] + 800.0).abs() < 10.0, "slope 1: {}", b[2]);
        // The intercept carries the accumulated slope error times mean(x0), so
        // it is the loosest of the three -- but it must be near 12, not near
        // the ~137 that dropping the unshift would give.
        assert!((b[0] - 12.0).abs() < 12.0, "intercept: {}", b[0]);
    }

    /// `pred_var` read the regressor of the last row stepped, a scratch a save
    /// does not keep, so a loaded filter answered for another regressor than
    /// the saved one would (review 2026-09-12, V7).
    #[test]
    fn pred_var_after_a_load_is_the_saved_filters() {
        let mut c = cfg(2, 1, vec![100.0]);
        c.obs_var = Some(0.25);
        c.min_weight = 3.0;
        let m = fit(c, 80, 61);
        let x = [0.3, -0.2];
        let before = m.pred_var(&x);
        // Through the bytes a save writes: `state()` alone is an in-memory
        // clone, which keeps the scratch a file does not.
        let bytes = rmp_serde::to_vec(&m.state()).unwrap();
        let state: State = rmp_serde::from_slice(&bytes).unwrap();
        let back = Kalman::restore(&state).unwrap();
        assert_eq!(back.pred_var(&x), before);
    }

    #[test]
    fn pred_var_is_the_quadratic_form_plus_observation_noise() {
        // No layer above this crate reads `pred_var`, so its arithmetic is
        // tested here. It is z' P z + R, and both halves are checked: the
        // quadratic form against a longhand loop over the stored covariance,
        // and R against the configured or inferred observation noise.
        let mut c = cfg(2, 1, vec![100.0]);
        c.obs_var = Some(0.25);
        c.min_weight = 3.0;
        let m = fit(c, 80, 61);
        let k = m.cfg.k_total();
        let x = [0.4, -0.7];
        // `P` is held in the anchor's coordinates, and so is the row read
        // against it (the module doc).
        let z = m.anchored(&x);

        let mut want = 0.0;
        for i in 0..k {
            for j in 0..k {
                want += z[i] * m.p[0][i * k + j] * z[j];
            }
        }
        want += 0.25;
        let got = m.pred_var(&x)[0];
        assert!((got - want).abs() < 1e-12, "{got} vs {want}");
        assert!(got > 0.25, "must exceed the observation noise: {got}");

        // Without a configured obs_var it falls back to the tracked residual
        // variance of that target.
        let mut c = cfg(2, 1, vec![100.0]);
        c.min_weight = 3.0;
        let m = fit(c, 80, 61);
        let got = m.pred_var(&x)[0];
        assert!(got > m.sigma2()[0], "{got} vs {}", m.sigma2()[0]);
    }

    #[test]
    fn share_p_shares_one_covariance_and_averages_the_noise() {
        // With `share_p` the process step runs once rather than once per
        // target, one covariance is kept, and the inferred observation noise
        // is the mean across targets rather than each target's own.
        let mut shared = cfg(2, 2, vec![100.0]);
        shared.share_p = true;
        shared.min_weight = 3.0;
        let ms = fit(shared, 200, 71);
        assert_eq!(ms.p.len(), 1, "one covariance for all targets");
        // Both targets read the same P, which the mean noise drives; each
        // target's own covariance is P times its noise over the mean, so its
        // pred_var is `σ²_j (zᵀ P z / σ̄² + 1)` (the module doc, task 211):
        // the same over `σ²_j` for both, and apart by their noises, which
        // here differ by construction (target 1 is 2x target 0).
        let pv = ms.pred_var(&[0.3, -0.2]);
        let s2 = ms.sigma2();
        assert!(s2[1] > 2.0 * s2[0], "targets differ: {s2:?}");
        assert!(
            (pv[0] / s2[0] - pv[1] / s2[1]).abs() < 1e-12,
            "{pv:?} {s2:?}"
        );
        let mean = s2.iter().sum::<f64>() / 2.0;
        let z = ms.anchored(&[0.3, -0.2]);
        let k = 3;
        let quad: f64 = (0..k)
            .map(|a| {
                (0..k)
                    .map(|b| z[a] * ms.p[0][a * k + b] * z[b])
                    .sum::<f64>()
            })
            .sum();
        for j in 0..2 {
            let want = s2[j] * (quad / mean + 1.0);
            assert!(
                (pv[j] - want).abs() < 1e-12 * want,
                "target {j}: {} against {want}",
                pv[j]
            );
        }

        let mut separate = cfg(2, 2, vec![100.0]);
        separate.min_weight = 3.0;
        let msep = fit(separate, 200, 71);
        assert_eq!(msep.p.len(), 2, "one covariance per target");
        let pv2 = msep.pred_var(&[0.3, -0.2]);
        assert!(
            (pv2[0] - pv2[1]).abs() > 1e-6,
            "unshared targets should differ: {pv2:?}"
        );
    }

    /// Task 204: `P`'s recursion never reads `y`, so under `share_p` a
    /// target beside an exact copy of itself predicts as it would alone, and
    /// as it would without `share_p`, to the bit (the mean of two equal
    /// noises is exact). Updating the shared `P` once per target, as it did,
    /// moved such a target's predictions by up to 0.92 on a spread of 1.1.
    /// Decay, irregular steps, weights other than 1, zero weights and null
    /// targets.
    #[test]
    fn share_p_a_target_beside_its_own_copy_predicts_as_it_would_alone() {
        let run = |m: usize, share: bool| -> Vec<u64> {
            let mut c = cfg(2, m, vec![50.0]);
            c.share_p = share;
            c.min_weight = 2.0;
            c.decay = Decay::Halflife(40.0);
            let mut model = Kalman::new(c).unwrap();
            let mut s = 211u64;
            let mut out = Vec::new();
            for i in 0..400 {
                let x = [lcg(&mut s), 0.5 + lcg(&mut s)];
                let e = lcg(&mut s);
                let y = (i % 7 != 3).then(|| 1.0 - 2.0 * x[0] + x[1] + 0.3 * e);
                let w = if i % 11 == 5 {
                    0.0
                } else {
                    0.5 + lcg(&mut s).abs()
                };
                let d = if i == 0 { 0.0 } else { 0.5 + (i % 3) as f64 };
                let pred = model.step(&x, &vec![y; m], d, w).pred;
                for p in &pred {
                    assert_eq!(p.to_bits(), pred[0].to_bits(), "row {i}: the copies part");
                }
                out.push(pred[0].to_bits());
            }
            out
        };
        let alone = run(1, false);
        let scored = alone
            .iter()
            .filter(|b| f64::from_bits(**b).is_finite())
            .count();
        assert!(scored > 300, "only {scored} rows predicted");
        assert_eq!(run(1, true), alone, "share_p, alone");
        assert_eq!(run(2, true), alone, "share_p, beside a copy");
    }

    /// Task 204: under `share_p` the order of `targets` moves no bit of any
    /// target's prediction. Three targets of different noise, each missing
    /// on rows of its own, so the shared noise sums three terms, in ascending
    /// order. Before, swapping two targets moved predictions by up to 0.36.
    #[test]
    fn share_p_the_order_of_the_targets_moves_no_bit() {
        let run = |order: [usize; 3]| -> Vec<[u64; 3]> {
            let mut c = cfg(2, 3, vec![50.0]);
            c.share_p = true;
            c.min_weight = 2.0;
            let mut model = Kalman::new(c).unwrap();
            let mut s = 223u64;
            let mut out = Vec::new();
            for i in 0..400 {
                let x = [lcg(&mut s), 0.5 + lcg(&mut s)];
                let e = [lcg(&mut s), lcg(&mut s), lcg(&mut s)];
                let ys = [
                    (i % 5 != 1).then(|| 2.0 * x[0] - x[1] + 0.1 * e[0]),
                    (i % 7 != 2).then(|| -x[0] + 3.0 * x[1] + e[1]),
                    (i % 3 != 0).then(|| 0.5 + x[0] + 3.0 * e[2]),
                ];
                let w = 0.5 + lcg(&mut s).abs();
                let given: Vec<Option<f64>> = order.iter().map(|&t| ys[t]).collect();
                let pred = model
                    .step(&x, &given, if i == 0 { 0.0 } else { 1.0 }, w)
                    .pred;
                let mut by_target = [0u64; 3];
                for (slot, &t) in order.iter().enumerate() {
                    by_target[t] = pred[slot].to_bits();
                }
                out.push(by_target);
            }
            out
        };
        let base = run([0, 1, 2]);
        let scored = base
            .iter()
            .filter(|r| f64::from_bits(r[2]).is_finite())
            .count();
        assert!(scored > 300, "only {scored} rows predicted");
        for order in [[2, 0, 1], [1, 2, 0], [0, 2, 1]] {
            assert_eq!(run(order), base, "order {order:?}");
        }
    }

    #[test]
    fn a_null_target_decays_its_weights_and_leaves_the_filter_alone() {
        let mut c = cfg(2, 1, vec![100.0]);
        c.decay = Decay::Halflife(10.0);
        c.min_weight = 3.0;
        let mut m = Kalman::new(c).unwrap();
        let mut s = 73u64;
        for i in 0..60 {
            let x = [lcg(&mut s), lcg(&mut s)];
            m.step(
                &x,
                &[Some(x[0] - x[1])],
                if i == 0 { 0.0 } else { 1.0 },
                1.0,
            );
        }
        // `b` follows the moments to their new coordinates (the module
        // doc), so a row with no target leaves the coefficients in the
        // caller's units where they were, to rounding.
        let coef = m.coefficients();
        let (wj, wsig, sig2) = (m.wj[0], m.wsig[0], m.sig2[0]);

        let lam = 0.5f64.powf(4.0 / 10.0);
        m.step(&[0.3, -0.2], &[None], 4.0, 1.0);
        for (a, b) in m.coefficients()[0].iter().zip(&coef[0]) {
            assert!(
                (a - b).abs() <= 1e-14 * (1.0 + b.abs()),
                "no target, no correction: {a} against {b}"
            );
        }
        assert_eq!(m.sig2[0], sig2);
        assert!((m.wj[0] - wj * lam).abs() < 1e-12);
        assert!((m.wsig[0] - wsig * lam).abs() < 1e-12);
    }

    #[test]
    fn a_zero_weight_row_does_not_correct_the_filter() {
        let mut c = cfg(2, 1, vec![100.0]);
        c.min_weight = 3.0;
        let mut m = Kalman::new(c).unwrap();
        let mut s = 79u64;
        for i in 0..60 {
            let x = [lcg(&mut s), lcg(&mut s)];
            m.step(
                &x,
                &[Some(x[0] - x[1])],
                if i == 0 { 0.0 } else { 1.0 },
                1.0,
            );
        }
        let beta = m.beta[0].clone();
        m.step(&[0.3, -0.2], &[Some(-500.0)], 1.0, 0.0);
        assert_eq!(m.beta[0], beta, "weight 0 must not move the coefficients");
    }

    /// `a_null_target_decays_its_weights_and_leaves_the_filter_alone` with
    /// the target present at weight zero. To the filter the two rows are the
    /// same -- a prediction step, no update -- and so they are to the
    /// weights: `wj` and `wsig` decay by the row's `lam` in both. The zero
    /// weight skipped the decay, so `σ²`, which sets `R` and `Q`, forgot less
    /// across such a row than across a null (review 2026-09-12, S9).
    #[test]
    fn a_zero_weight_row_decays_its_weights_as_a_null_does() {
        let mut c = cfg(2, 1, vec![100.0]);
        c.decay = Decay::Halflife(10.0);
        c.min_weight = 3.0;
        let mut m = Kalman::new(c).unwrap();
        let mut s = 73u64;
        for i in 0..60 {
            let x = [lcg(&mut s), lcg(&mut s)];
            m.step(
                &x,
                &[Some(x[0] - x[1])],
                if i == 0 { 0.0 } else { 1.0 },
                1.0,
            );
        }
        let beta = m.beta[0].clone();
        let (wj, wsig, sig2) = (m.wj[0], m.wsig[0], m.sig2[0]);

        let lam = 0.5f64.powf(4.0 / 10.0);
        m.step(&[0.3, -0.2], &[Some(-500.0)], 4.0, 0.0);
        assert_eq!(m.beta[0], beta, "weight 0, no correction");
        assert_eq!(m.sig2[0], sig2);
        assert!(
            (m.wj[0] - wj * lam).abs() < 1e-12,
            "{} vs {}",
            m.wj[0],
            wj * lam
        );
        assert!(
            (m.wsig[0] - wsig * lam).abs() < 1e-12,
            "{} vs {}",
            m.wsig[0],
            wsig * lam
        );
    }

    #[test]
    fn residual_variance_is_the_ew_mean_of_squared_out_of_sample_errors() {
        let mut c = cfg(1, 1, vec![f64::INFINITY]);
        c.decay = Decay::Halflife(25.0);
        c.min_weight = 3.0;
        c.standardize = false;
        c.obs_var = Some(0.5);
        let mut m = Kalman::new(c).unwrap();

        let (mut want, mut wsig) = (0.0, 0.0);
        let mut s = 83u64;
        for i in 0..120 {
            let x = [lcg(&mut s)];
            let y = 2.0 * x[0] + 0.3 * lcg(&mut s);
            let d = if i == 0 { 0.0 } else { 1.0 };
            let lam = 0.5f64.powf(d / 25.0);
            let p = m.step(&x, &[Some(y)], d, 1.0).pred[0];
            if p.is_finite() {
                let resid = y - p;
                let ws_new = lam * wsig + 1.0;
                want = (lam * wsig * want + resid * resid) / ws_new;
                wsig = ws_new;
            }
            assert!(
                (m.sigma2()[0] - want).abs() < 1e-12,
                "row {i}: {} vs {want}",
                m.sigma2()[0]
            );
        }
        assert!(wsig > 30.0 && want > 0.0);
    }

    /// The same mean across rows with no prediction: after a clock gap
    /// takes `n_eff` under `min_weight`, the next rows have a target and
    /// no prediction. They add nothing, and age `σ²`'s weight as every row
    /// does; they aged nothing, so `σ²` -- which sets `R` and `Q` -- forgot
    /// less across them than the clock says (N6, found beside review
    /// 2026-09-12 S13).
    #[test]
    fn the_residual_variance_ages_across_rows_with_no_prediction() {
        let hl = 25.0;
        let mut c = cfg(1, 1, vec![f64::INFINITY]);
        c.decay = Decay::Halflife(hl);
        c.min_weight = 3.0;
        c.standardize = false;
        c.obs_var = Some(0.5);
        let mut m = Kalman::new(c).unwrap();
        let (mut want, mut wsig, mut unpredicted) = (0.0f64, 0.0f64, 0);
        let mut s = 83u64;
        for i in 0..160 {
            let x = [lcg(&mut s)];
            let y = 2.0 * x[0] + 0.3 * lcg(&mut s);
            let d = match i {
                0 => 0.0,
                80 => 400.0,
                _ => 1.0,
            };
            let p = m.step(&x, &[Some(y)], d, 1.0).pred[0];
            wsig *= 0.5f64.powf(d / hl);
            if p.is_finite() {
                let r = y - p;
                let ws_new = wsig + 1.0;
                want = (wsig * want + r * r) / ws_new;
                wsig = ws_new;
            } else if wsig > 0.0 {
                unpredicted += 1;
            }
            let got = m.sigma2()[0];
            assert!(
                (got - want).abs() <= 1e-12 * want,
                "row {i}: sigma2 {got}, the EW mean of the squared errors {want}"
            );
        }
        assert!(
            unpredicted >= 2,
            "the gap must leave rows with no prediction"
        );
    }

    /// With `standardize = false`, `q = 0` and a fixed `obs_var`, the filter is
    /// exactly a Bayesian linear regression: coefficients converge to the ridge
    /// solution with penalty `obs_var / P_0 = 1 / p0`, the prior being
    /// `P_0 = p0 obs_var I` (CC4).
    #[test]
    fn unstandardized_with_no_process_noise_is_bayesian_regression() {
        let (p0, obs_var) = (10.0, 0.25);
        let mut m = Kalman::new(KalmanCfg {
            n_features: 2,
            n_targets: 1,
            fit_intercept: false,
            decay: Decay::Halflife(f64::INFINITY),
            half_life: vec![f64::INFINITY],
            q: Some(vec![0.0, 0.0]),
            obs_var: Some(obs_var),
            p0,
            share_p: false,
            min_weight: 0.0,
            revert_half_life: vec![f64::INFINITY],
            standardize: false,
        })
        .unwrap();
        // Accumulate the normal equations alongside, then compare with the
        // closed-form ridge solution (1 / p0 is the implied penalty).
        let mut s = 55u64;
        let (mut xtx, mut xty) = ([[0.0f64; 2]; 2], [0.0f64; 2]);
        for i in 0..400 {
            let x = [lcg(&mut s), lcg(&mut s)];
            let y = 1.25 * x[0] - 0.5 * x[1];
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
            for a in 0..2 {
                xty[a] += x[a] * y;
                for b in 0..2 {
                    xtx[a][b] += x[a] * x[b];
                }
            }
        }
        // The ridge normal equations solved by faer's LU (`crate::oracle`),
        // which the filter shares no arithmetic with, where Cramer's rule by
        // hand stood (review 2026-10-06, CF8).
        let lam = 1.0 / p0;
        let gram = [xtx[0][0] + lam, xtx[0][1], xtx[1][0], xtx[1][1] + lam];
        let want = crate::oracle::solve(&gram, &xty);
        let got = &m.coefficients()[0];
        for i in 0..2 {
            assert!(
                (got[i] - want[i]).abs() < 1e-9,
                "coef {i}: {} vs ridge closed form {}",
                got[i],
                want[i]
            );
        }
    }

    #[test]
    fn tracks_a_static_beta() {
        let mut m = Kalman::new(cfg(2, 1, vec![500.0])).unwrap();
        let mut s = 3u64;
        for i in 0..2000 {
            let x = [lcg(&mut s), lcg(&mut s)];
            let y = 2.0 * x[0] - 1.0 * x[1] + 0.5 + 0.05 * lcg(&mut s);
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let c = &m.coefficients()[0];
        assert!((c[1] - 2.0).abs() < 0.1, "slope0 {}", c[1]);
        assert!((c[2] + 1.0).abs() < 0.1, "slope1 {}", c[2]);
        assert!((c[0] - 0.5).abs() < 0.1, "intercept {}", c[0]);
    }

    #[test]
    fn tracks_a_drifting_beta_better_than_a_pinned_one() {
        // Same data through a responsive filter and a pinned one: the responsive
        // filter must have lower out-of-sample error.
        let mut fast = Kalman::new(cfg(1, 1, vec![50.0])).unwrap();
        let mut pinned = Kalman::new(cfg(1, 1, vec![f64::INFINITY])).unwrap();
        let mut s = 4u64;
        let (mut e_fast, mut e_pin) = (0.0f64, 0.0f64);
        let mut beta = 1.0f64;
        for i in 0..3000 {
            beta += 0.01 * lcg(&mut s); // random walk
            let x = [lcg(&mut s)];
            let y = beta * x[0] + 0.05 * lcg(&mut s);
            let d = if i == 0 { 0.0 } else { 1.0 };
            let a = fast.step(&x, &[Some(y)], d, 1.0);
            let b = pinned.step(&x, &[Some(y)], d, 1.0);
            if i > 500 {
                if a.pred[0].is_finite() {
                    e_fast += (y - a.pred[0]).powi(2);
                }
                if b.pred[0].is_finite() {
                    e_pin += (y - b.pred[0]).powi(2);
                }
            }
        }
        assert!(e_fast < e_pin, "fast {e_fast} should beat pinned {e_pin}");
    }

    #[test]
    fn infinite_halflife_pins_the_coefficient() {
        // Per-factor: slot 1 (x0) pinned, slot 2 (x1) free.
        let mut m = Kalman::new(cfg(2, 1, vec![1e9, f64::INFINITY, 30.0])).unwrap();
        let mut s = 5u64;
        for i in 0..300 {
            let x = [lcg(&mut s), lcg(&mut s)];
            let y = x[0] + x[1];
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let q = m.q_vec(1.0);
        assert_eq!(q[1], 0.0, "pinned factor must have zero process noise");
        assert!(q[2] > 0.0);
    }

    #[test]
    fn explicit_q_overrides_halflife() {
        let mut c = cfg(1, 1, vec![10.0]);
        c.q = Some(vec![0.0, 0.25]);
        let m = Kalman::new(c).unwrap();
        assert_eq!(m.q_vec(99.0), vec![0.0, 0.25]);
    }

    #[test]
    fn share_p_keeps_one_covariance() {
        let mut c = cfg(2, 3, vec![100.0]);
        c.share_p = true;
        let mut m = Kalman::new(c).unwrap();
        assert_eq!(m.p.len(), 1);
        let mut s = 6u64;
        for i in 0..200 {
            let x = [lcg(&mut s), lcg(&mut s)];
            let y0 = x[0];
            let y1 = x[1];
            let y2 = x[0] + x[1];
            m.step(
                &x,
                &[Some(y0), Some(y1), Some(y2)],
                if i == 0 { 0.0 } else { 1.0 },
                1.0,
            );
        }
        assert!(
            m.coefficients()
                .iter()
                .all(|c| c.iter().all(|v| v.is_finite()))
        );
    }

    /// Predictive variance must start wide and narrow with evidence — that is
    /// the whole reason to report it alongside `sigma`.
    #[test]
    fn predictive_variance_narrows_with_evidence() {
        let mut m = Kalman::new(cfg(2, 1, vec![f64::INFINITY])).unwrap();
        let mut s = 44u64;
        let mut early = 0.0;
        let mut late = 0.0;
        for i in 0..3000 {
            let x = [lcg(&mut s), lcg(&mut s)];
            let y = 2.0 * x[0] - x[1] + 0.1 * lcg(&mut s);
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
            if i == 20 {
                early = m.pred_var(&x)[0];
            }
            if i == 2999 {
                late = m.pred_var(&x)[0];
            }
        }
        assert!(early.is_finite() && late.is_finite());
        assert!(
            late < early,
            "predictive variance should narrow: {early} -> {late}"
        );
    }

    #[test]
    fn predictive_variance_exceeds_the_observation_noise() {
        // It is parameter uncertainty PLUS noise, so it can never be smaller
        // than the noise alone.
        let mut c = cfg(2, 1, vec![f64::INFINITY]);
        c.obs_var = Some(0.25);
        let mut m = Kalman::new(c).unwrap();
        let mut s = 46u64;
        for i in 0..500 {
            let x = [lcg(&mut s), lcg(&mut s)];
            m.step(&x, &[Some(x[0])], if i == 0 { 0.0 } else { 1.0 }, 1.0);
            assert!(m.pred_var(&x)[0] >= 0.25 - 1e-12);
        }
    }

    #[test]
    fn state_roundtrip() {
        let mut m1 = Kalman::new(cfg(2, 1, vec![100.0])).unwrap();
        let mut s = 7u64;
        let rows: Vec<([f64; 2], f64)> = (0..120)
            .map(|_| {
                let x = [lcg(&mut s), lcg(&mut s)];
                (x, x[0] - 0.5 * x[1])
            })
            .collect();
        for (i, (x, y)) in rows[..60].iter().enumerate() {
            m1.step(x, &[Some(*y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let bytes = rmp_serde::to_vec(&m1.state()).unwrap();
        let mut m2 = Kalman::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
        for (x, y) in &rows[60..] {
            assert_eq!(
                m1.step(x, &[Some(*y)], 1.0, 1.0).pred,
                m2.step(x, &[Some(*y)], 1.0, 1.0).pred
            );
        }
    }

    #[test]
    fn null_target_is_predict_only() {
        let mut m = Kalman::new(cfg(1, 2, vec![100.0])).unwrap();
        let mut s = 8u64;
        for i in 0..60 {
            let x = [lcg(&mut s)];
            m.step(
                &x,
                &[Some(x[0]), Some(-x[0])],
                if i == 0 { 0.0 } else { 1.0 },
                1.0,
            );
        }
        // The coefficients in the caller's units: past the warm-up `b`
        // follows the moments, so the null target's are where they were, to
        // rounding, and the observed target's moved.
        let before = m.coefficients();
        let st = m.step(&[0.5], &[Some(1.0), None], 1.0, 1.0);
        assert!(st.pred[1].is_finite());
        let after = m.coefficients();
        for (a, b) in after[1].iter().zip(&before[1]) {
            assert!((a - b).abs() <= 1e-14 * (1.0 + b.abs()), "{a} against {b}");
        }
        assert_ne!(after[0], before[0], "the target that was there corrected");
    }

    /// Each slot takes its prior on the first row that observes the target
    /// on which its scale is usable (the module doc, task 211): the
    /// intercept and a moving feature on the row where the noise first rests
    /// on three rows' innovations, a feature that has not moved yet on the
    /// first observed row after it has. Until then its variance and its
    /// coefficient are 0. With `obs_var` the intercept is sized from the
    /// start; unstandardized, the whole `P` on the first row with a noise.
    #[test]
    fn each_slot_takes_its_prior_once_its_scale_is_usable() {
        let mut c = cfg(2, 1, vec![50.0]);
        c.decay = Decay::Lam(0.9);
        c.min_weight = 0.0;
        let mut m = Kalman::new(c.clone()).unwrap();
        let mut s = 5u64;
        let k = 3;
        for i in 0..12usize {
            // The second feature holds 0.5 for the first 6 rows; the target
            // is null on row 1.
            let x = [lcg(&mut s), if i < 6 { 0.5 } else { lcg(&mut s) }];
            let y = (i != 1).then(|| 1.0 + x[0] - x[1] + 0.1 * lcg(&mut s));
            m.step(&x, &[y], if i == 0 { 0.0 } else { 1.0 }, 1.0);
            let diag: Vec<f64> = (0..k).map(|a| m.p[0][a * k + a]).collect();
            // Rows 0, 2 and 3 observe the target: the third innovation is
            // row 3's, and the first feature's scale is usable from row 2.
            let sized = [i >= 3, i >= 3, i >= 7];
            for a in 0..k {
                assert_eq!(diag[a] != 0.0, sized[a], "row {i}, slot {a}: {diag:?}");
                if !sized[a] {
                    assert_eq!(m.beta[0][a], 0.0, "row {i}, slot {a}");
                }
            }
        }
        // With `obs_var` the intercept is sized from the start, and each
        // feature waits for its scale.
        let mut known = c.clone();
        known.obs_var = Some(0.5);
        let m = Kalman::new(known).unwrap();
        assert_eq!(m.p[0][0], 0.5);
        assert_eq!(m.p[0][4], 0.0);
        // Unstandardized there is no scale to wait for and no anchor: the
        // first row with a noise sizes every slot.
        c.standardize = false;
        let mut raw = Kalman::new(c).unwrap();
        raw.step(&[0.3, 0.2], &[Some(1.0)], 0.0, 1.0);
        assert!((0..k).all(|a| raw.p[0][a * k + a] > 0.0));
        assert!(raw.anchor_scale.is_empty());
    }

    /// A move of the moments re-maps `b` and `P` once they drift past
    /// `ANCHOR_DRIFT` from the anchor (the module doc), so a row with no
    /// target that moves them -- a feature well off its mean -- leaves the
    /// prediction for any row where it was, to rounding, while `b` itself
    /// moves. A move past [`REMAP_LIMIT`] -- here a feature at the input
    /// bound after a run at a scale of 1e-100, a factor of about 1e200 -- is
    /// refused whole, as a move whose numbers would not be finite is: `b`
    /// and `P` keep their bits, finite, and the filter goes on predicting.
    #[test]
    fn a_move_of_the_moments_is_followed_and_one_past_the_limit_is_refused() {
        let mut c = cfg(1, 1, vec![30.0]);
        c.decay = Decay::Lam(0.5);
        c.min_weight = 0.0;
        let mut m = Kalman::new(c).unwrap();
        let mut s = 9u64;
        for _ in 0..40 {
            let x = [lcg(&mut s)];
            m.step(&x, &[Some(1.0 + x[0] + 0.1 * lcg(&mut s))], 1.0, 1.0);
        }
        let at = [0.4];
        let (pred, beta) = (m.predict(&at, 1.0).pred[0], m.beta[0].clone());
        m.step(&[6.0], &[None], 1.0, 1.0);
        assert_ne!(m.beta[0], beta, "the moments moved and b followed");
        assert_eq!(m.anchor_hi[1], m.stats.mean_pair(1).0, "the anchor moved");
        let after = m.predict(&at, 1.0).pred[0];
        assert!(
            (after - pred).abs() <= 1e-14 * (1.0 + pred.abs()),
            "{after} against {pred}"
        );
        // A long run at a scale of 1e-100, the target at its own scale, so
        // that `P` is not tiny with it; then the bound, with no target.
        for _ in 0..1500 {
            let x = [1e-100 * lcg(&mut s)];
            m.step(&x, &[Some(1.0 + 0.1 * lcg(&mut s))], 1.0, 1.0);
        }
        assert!(m.scales()[1] < 1e-90, "the scale shrank: {:?}", m.scales());
        let (beta, p) = (m.beta[0].clone(), m.p[0].clone());
        m.step(&[1e100], &[None], 1.0, 1.0);
        assert_eq!(m.beta[0], beta, "the re-map was refused");
        // `P` took nothing: a row with no target adds no process noise.
        assert_eq!(m.p[0], p);
        assert!(m.predict(&[0.5], 1.0).pred[0].is_finite());
        for _ in 0..50 {
            let x = [lcg(&mut s)];
            let got = m.step(&x, &[Some(1.0 + x[0])], 1.0, 1.0);
            assert!(got.n_eff.is_finite());
        }
        assert!(m.beta[0].iter().all(|v| v.is_finite()));
    }

    /// The drift is two-sided in a scale: a feature whose spread shrinks by
    /// a factor of 4e4 is re-mapped as it passes each halving, so the
    /// anchor's scale follows the feature's down, never more than a factor
    /// of 2 from it. `|a − 1|` alone never passes 1 for a scale that
    /// shrinks, and leaves `a` tiny with arrowhead terms of `q / a²` that
    /// cancel in every prediction (the module doc): measured on this stream
    /// at 1,500 rows, the anchored replica with that drift parted from the
    /// per-row re-map by 8e9 noise standard deviations at `coef_half_life`
    /// 20, and this filter by 2.7e-14.
    #[test]
    fn a_shrinking_scale_is_re_mapped_as_it_halves() {
        let mut c = cfg(1, 1, vec![20.0]);
        c.decay = Decay::Lam(0.8);
        c.min_weight = 0.0;
        let mut m = Kalman::new(c).unwrap();
        let mut s = 31u64;
        let mut spread = 1.0;
        for i in 0..400 {
            if i >= 50 {
                spread *= 0.97;
            }
            let x = [3.0 + spread * lcg(&mut s)];
            let y = 1.0 + 2.0 * (x[0] - 3.0) / spread + 0.1 * lcg(&mut s);
            assert!(m.step(&x, &[Some(y)], 1.0, 1.0).pred[0].is_finite() || i < 2);
            let a = m.scales()[1] / m.anchor_scale[1];
            assert!(
                i < 10 || (0.5..=2.0).contains(&a),
                "row {i}: the scale is {a} of the anchor's"
            );
        }
        assert!(m.anchor_scale[1] < 1e-4, "{}", m.anchor_scale[1]);
    }

    /// The standardized filter is its recursion, written from the module
    /// doc with every moment from its definition (`the_filter_is_its_recursion`
    /// holds the unstandardized one): the row standardized against the EW
    /// means and variances of the rows before it, `z_i = (x_i − m_i) / s_i`
    /// (`x_i / s_i`, the root mean square, through the origin), each a
    /// weighted sum over the history with the weights aged by the decay; the
    /// filter's recursion on `z` in those, the current, coordinates, with the
    /// transition on every row and the process noise `Q D²` on each row that
    /// observes the covariance's target, `D` the clock since the last; each
    /// slot's prior `p0 R` on the first observed row its scale is usable, `R`
    /// the mean squared innovation once three rows give it; and after every
    /// row that moves the moments, the change of coordinates from the moments
    /// before it to the moments after it, `A` with `A_00 = 1`, `A_0i = (m'_i −
    /// m_i) / s_i`, `A_ii = s'_i / s_i`, applied as `b' = A b` and `P' = A P
    /// Aᵀ` by matrix products -- task 206's re-map on every row, which the
    /// anchored filter is to rounding (task 211). Two targets, one present
    /// one row in three, weights other than 1 and 0, steps of 1 and 3, shared
    /// and not -- shared, the noise is the mean residual variance over the
    /// targets that have one (review round 5, A1) -- with and without an
    /// intercept, and with slopes reverting at half-lives of their own. The
    /// predictions agree to 1e-12 of their size.
    #[test]
    fn the_standardized_filter_is_its_recursion() {
        let inf = f64::INFINITY;
        for (fit_intercept, share, revert) in [
            (true, false, vec![inf]),
            (true, true, vec![inf]),
            (false, false, vec![inf]),
            (true, false, vec![inf, 40.0, 8.0]),
            (true, true, vec![inf, 25.0, 60.0]),
        ] {
            let hq = 40.0;
            let decay = Decay::Lam(0.9);
            let mut c = cfg(2, 2, vec![hq]);
            c.fit_intercept = fit_intercept;
            c.share_p = share;
            c.min_weight = 0.0;
            c.decay = decay;
            c.revert_half_life = revert.clone();
            let mut m = Kalman::new(c.clone()).unwrap();
            let off = usize::from(fit_intercept);
            let k = 2 + off;
            let n_p = if share { 1 } else { 2 };
            let mut p = vec![vec![0.0f64; k * k]; n_p];
            let mut b = vec![vec![0.0f64; k]; 2];
            let (mut sig2, mut wsig, mut wj) = ([0.0f64; 2], [0.0f64; 2], [0.0f64; 2]);
            let mut elapsed = vec![0.0f64; n_p];
            // Per covariance: Σ w e², Σ w and the rows.
            let mut basis = vec![[0.0f64; 3]; n_p];
            let mut history: Vec<([f64; 2], f64)> = Vec::new();
            // The moments of the history, by their definition: `(m, s,
            // usable)` per feature, `m = 0` through the origin.
            let moments = |h: &[([f64; 2], f64)]| -> ([f64; 2], [f64; 2], [bool; 2]) {
                let total: f64 = h.iter().map(|(_, w)| w).sum();
                let (mut mean, mut scale, mut usable) = ([0.0; 2], [1.0; 2], [false; 2]);
                if total <= 0.0 {
                    return (mean, scale, usable);
                }
                for f in 0..2 {
                    if fit_intercept {
                        let mu = h.iter().map(|(x, w)| w * x[f]).sum::<f64>() / total;
                        let var =
                            h.iter().map(|(x, w)| w * (x[f] - mu).powi(2)).sum::<f64>() / total;
                        mean[f] = mu;
                        usable[f] = var > 0.0;
                        scale[f] = if usable[f] { var.sqrt() } else { 1.0 };
                    } else {
                        let raw = h.iter().map(|(x, w)| w * x[f] * x[f]).sum::<f64>() / total;
                        usable[f] = raw > 0.0;
                        scale[f] = if usable[f] { raw.sqrt() } else { 1.0 };
                    }
                }
                (mean, scale, usable)
            };
            let phi_of = |i: usize, d: f64| -> f64 {
                let r = if revert.len() == 1 {
                    revert[0]
                } else {
                    revert[i]
                };
                Decay::Halflife(r).factor(d)
            };
            let mut s = 211u64;
            let mut checked = 0;
            for i in 0..150 {
                let x = [2.0 + lcg(&mut s), 0.3 * lcg(&mut s)];
                let ys = [
                    Some(0.5 + x[0] - 2.0 * x[1] + 0.2 * lcg(&mut s)),
                    (i % 3 == 0).then(|| -x[0] + 3.0 * lcg(&mut s)),
                ];
                let w = if i % 9 == 5 {
                    0.0
                } else {
                    0.5 + (lcg(&mut s) + 1.0)
                };
                let d = match i {
                    0 => 0.0,
                    _ if i % 7 == 2 => 3.0,
                    _ => 1.0,
                };
                let lam = decay.factor(d);
                for e in elapsed.iter_mut() {
                    *e += d;
                }
                // The transition, in the current coordinates.
                let phi: Vec<f64> = (0..k).map(|a| phi_of(a, d)).collect();
                for bj in b.iter_mut() {
                    for a in 0..k {
                        bj[a] *= phi[a];
                    }
                }
                for pp in p.iter_mut() {
                    for a in 0..k {
                        for bb in 0..k {
                            pp[a * k + bb] *= phi[a] * phi[bb];
                        }
                    }
                }
                let (mean, scale, usable) = moments(&history);
                let mut z = vec![1.0; k];
                for f in 0..2 {
                    z[off + f] = (x[f] - mean[f]) / scale[f];
                }
                let dotz = |v: &[f64]| -> f64 { (0..k).map(|a| z[a] * v[a]).sum() };
                let want: Vec<f64> = (0..2)
                    .map(|j| if wj[j] > 0.0 { dotz(&b[j]) } else { f64::NAN })
                    .collect();
                let got = m.step(&x, &ys, d, w).pred;
                for j in 0..2 {
                    assert_eq!(
                        got[j].is_nan(),
                        want[j].is_nan(),
                        "{fit_intercept} {share} {revert:?}, row {i}"
                    );
                    if want[j].is_finite() {
                        assert!(
                            (got[j] - want[j]).abs() <= 1e-12 * want[j].abs().max(1.0),
                            "intercept {fit_intercept}, share {share}, revert {revert:?}, row \
                             {i}, target {j}: {} against {}",
                            got[j],
                            want[j]
                        );
                        checked += 1;
                    }
                }
                // The filter on `z`, as `the_filter_is_its_recursion` writes
                // it: a row of weight 0 observes nothing.
                let obs: Vec<Option<f64>> = ys.iter().map(|y| y.filter(|_| w > 0.0)).collect();
                let e2: Vec<Option<f64>> = (0..2)
                    .map(|j| obs[j].map(|y| (y - dotz(&b[j])).powi(2)))
                    .collect();
                let seen: Vec<f64> = e2.iter().flatten().copied().collect();
                let shared_first = if seen.is_empty() {
                    0.0
                } else {
                    seen.iter().sum::<f64>() / seen.len() as f64
                };
                let have: Vec<f64> = sig2.iter().copied().filter(|v| *v > 0.0).collect();
                let shared = if have.is_empty() {
                    0.0
                } else {
                    have.iter().sum::<f64>() / have.len() as f64
                };
                let noise: Vec<f64> = (0..2)
                    .map(|j| {
                        let s2 = if share { shared } else { sig2[j] };
                        if s2 > 0.0 {
                            s2
                        } else if share {
                            shared_first
                        } else {
                            e2[j].unwrap_or(0.0)
                        }
                    })
                    .collect();
                // A row counts once, whatever the targets it observes.
                for (pi, held) in basis.iter_mut().enumerate() {
                    let mine: Vec<f64> = (0..2)
                        .filter(|&j| share || j == pi)
                        .filter_map(|j| e2[j].filter(|v| *v > 0.0))
                        .collect();
                    if !mine.is_empty() {
                        held[0] += w * mine.iter().sum::<f64>();
                        held[1] += w * mine.len() as f64;
                        held[2] += 1.0;
                    }
                }
                for pi in 0..n_p {
                    let informs = if share {
                        obs.iter().any(Option::is_some)
                    } else {
                        obs[pi].is_some()
                    };
                    if !informs {
                        continue;
                    }
                    let gap = elapsed[pi];
                    elapsed[pi] = 0.0;
                    let q = noise[pi] * (std::f64::consts::LN_2 / hq).powi(2);
                    for a in 0..k {
                        // `D²`, or on a reverting slot `((1 − 2^(−D/r)) /
                        // θ)²`, `θ = ln 2 / r` (task 214).
                        let r = if revert.len() == 1 {
                            revert[0]
                        } else {
                            revert[a]
                        };
                        let dd = if r.is_infinite() {
                            gap * gap
                        } else {
                            let theta = std::f64::consts::LN_2 / r;
                            ((1.0 - phi_of(a, gap)) / theta).powi(2)
                        };
                        if p[pi][a * k + a] != 0.0 {
                            p[pi][a * k + a] += q * dd;
                        }
                    }
                    if basis[pi][2] >= 3.0 {
                        let r = basis[pi][0] / basis[pi][1];
                        for a in 0..k {
                            let ok = a < off || usable[a - off];
                            if ok && p[pi][a * k + a] == 0.0 {
                                p[pi][a * k + a] = r;
                            }
                        }
                    }
                }
                let mut shared_row: Option<(Vec<f64>, f64)> = None;
                for j in 0..2 {
                    let pi = if share { 0 } else { j };
                    let sigma2 = noise[j];
                    let Some(y) = obs[j] else {
                        wj[j] *= lam;
                        wsig[j] *= lam;
                        continue;
                    };
                    let pz: Vec<f64> = (0..k).map(|a| dotz(&p[pi][a * k..(a + 1) * k])).collect();
                    let s_inn = dotz(&pz) + sigma2 / w;
                    let err = y - dotz(&b[j]);
                    if sigma2 > 0.0 {
                        for a in 0..k {
                            b[j][a] += pz[a] / s_inn * err;
                        }
                        if share {
                            shared_row = Some((pz, s_inn));
                        } else {
                            for a in 0..k {
                                for bb in 0..k {
                                    p[pi][a * k + bb] -= pz[a] * pz[bb] / s_inn;
                                }
                            }
                        }
                    }
                    let aged = lam * wsig[j];
                    wsig[j] = aged;
                    if want[j].is_finite() {
                        let r = y - want[j];
                        sig2[j] = (aged * sig2[j] + w * r * r) / (aged + w);
                        wsig[j] = aged + w;
                    }
                    wj[j] = lam * wj[j] + w;
                }
                if let Some((pz, s_inn)) = shared_row {
                    for a in 0..k {
                        for bb in 0..k {
                            p[0][a * k + bb] -= pz[a] * pz[bb] / s_inn;
                        }
                    }
                }
                // The moments learn the row, at its weight, and `b` and `P`
                // follow them.
                for (_, wh) in history.iter_mut() {
                    *wh *= lam;
                }
                history.push((x, w));
                if w > 0.0 {
                    let (mean2, scale2, _) = moments(&history);
                    let mut am = vec![0.0f64; k * k];
                    am[0] = 1.0;
                    for f in 0..2 {
                        let a = off + f;
                        am[a * k + a] = scale2[f] / scale[f];
                        if fit_intercept {
                            am[a] = (mean2[f] - mean[f]) / scale[f];
                        }
                    }
                    for bj in b.iter_mut() {
                        let old = bj.clone();
                        for r in 0..k {
                            bj[r] = (0..k).map(|cc| am[r * k + cc] * old[cc]).sum();
                        }
                    }
                    for pp in p.iter_mut() {
                        let mut ap = vec![0.0f64; k * k];
                        for r in 0..k {
                            for cc in 0..k {
                                ap[r * k + cc] =
                                    (0..k).map(|t| am[r * k + t] * pp[t * k + cc]).sum();
                            }
                        }
                        for r in 0..k {
                            for cc in 0..k {
                                pp[r * k + cc] =
                                    (0..k).map(|t| ap[r * k + t] * am[cc * k + t]).sum();
                            }
                        }
                    }
                }
            }
            assert!(checked > 250, "{checked} predictions checked");
        }
    }

    // ---- reversion (ENHANCEMENTS E41) ----

    fn revert_cfg(r: Vec<f64>) -> KalmanCfg {
        KalmanCfg {
            revert_half_life: r,
            ..cfg(2, 1, vec![100.0])
        }
    }

    #[test]
    fn revert_halflife_defaults_to_the_random_walk_when_a_state_file_omits_it() {
        let json = r#"{
            "n_features": 2, "n_targets": 1, "fit_intercept": true,
            "decay": {"Halflife": 200.0}, "half_life": [100.0], "q": null,
            "obs_var": null, "p0": 1.0, "share_p": false, "min_weight": 10.0
        }"#;
        let cfg: KalmanCfg = serde_json::from_str(json).expect("should load without the field");
        assert_eq!(cfg.revert_half_life, vec![f64::INFINITY]);
        assert!(!cfg.reverts());
    }

    #[test]
    fn revert_halflife_is_validated() {
        for (r, msg) in [
            (vec![10.0, 10.0], "length 1 or 3"),
            (vec![0.0], "must be > 0"),
            (vec![-5.0], "must be > 0"),
            (vec![f64::NAN], "must be > 0"),
            (vec![f64::INFINITY, 10.0, f64::NEG_INFINITY], "must be > 0"),
        ] {
            let err = Kalman::new(revert_cfg(r.clone())).unwrap_err();
            assert!(err.contains(msg), "{r:?}: {err}");
        }
        for r in [
            vec![f64::INFINITY],
            vec![10.0],
            vec![f64::INFINITY, 5.0, 1e300],
        ] {
            Kalman::new(revert_cfg(r)).unwrap();
        }
    }

    #[test]
    fn an_infinite_revert_halflife_is_bit_identical_to_the_default() {
        // Spelled as a scalar or per slot, `inf` must not touch a number:
        // the transition is skipped, not multiplied by 1.
        let a = fit(cfg(2, 1, vec![100.0]), 200, 5);
        let b = fit(revert_cfg(vec![f64::INFINITY]), 200, 5);
        let c = fit(revert_cfg(vec![f64::INFINITY; 3]), 200, 5);
        assert_eq!(a.beta, b.beta);
        assert_eq!(a.p, b.p);
        assert_eq!(a.beta, c.beta);
        assert_eq!(a.p, c.p);
        assert_eq!(
            a.predict(&[0.3, 0.7], 2.5).pred,
            c.predict(&[0.3, 0.7], 2.5).pred
        );
    }

    #[test]
    fn reversion_is_exact_between_observations() {
        // With nothing to learn from (null targets), the mean shrinks by
        // exactly `2^(-d/r_i)` per slot over the elapsed clock and the
        // covariance by `phi_i phi_j` (with `q = 0` so nothing is added
        // back), whatever the clock's spacing. On the unstandardized
        // filter so the reported coefficients are the state itself.
        let r = vec![f64::INFINITY, 20.0, 5.0];
        let mut m = Kalman::new(KalmanCfg {
            q: Some(vec![0.0; 3]),
            standardize: false,
            ..revert_cfg(r.clone())
        })
        .unwrap();
        let mut s = 11u64;
        for i in 0..80 {
            let x = [lcg(&mut s), lcg(&mut s)];
            let y = 0.4 + 1.5 * x[0] - 2.0 * x[1] + 0.05 * lcg(&mut s);
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let b0 = m.beta[0].clone();
        let p0 = m.p[0].clone();
        let gaps = [0.5, 3.0, 1.0, 0.0, 7.25, 2.0];
        for d in gaps {
            let x = [lcg(&mut s), lcg(&mut s)];
            m.step(&x, &[None], d, 1.0);
        }
        let total: f64 = gaps.iter().sum();
        let phi: Vec<f64> = r.iter().map(|h| (-(total / h)).exp2()).collect();
        assert_eq!(phi[0], 1.0);
        for i in 0..3 {
            let want = b0[i] * phi[i];
            assert!(
                (m.beta[0][i] - want).abs() <= 1e-12 * want.abs().max(1e-300),
                "slot {i}: {} vs {want}",
                m.beta[0][i]
            );
            for j in 0..3 {
                let want = p0[i * 3 + j] * phi[i] * phi[j];
                assert!(
                    (m.p[0][i * 3 + j] - want).abs() <= 1e-12 * want.abs().max(1e-300),
                    "P[{i},{j}]: {} vs {want}",
                    m.p[0][i * 3 + j]
                );
            }
        }
        // The intercept slot, `inf`, was left alone to the bit.
        assert_eq!(m.beta[0][0], b0[0]);
        assert_eq!(m.p[0][0], p0[0]);
        // And the state is what `coefficients` reports: unstandardized.
        assert_eq!(m.coefficients()[0], m.beta[0]);
    }

    /// A reverting slot's process noise for a gap `D` is `q g(D)²`, `g(D) =
    /// (1 − 2^(−D/r)) / θ`, `θ = ln 2 / r` (the module doc, docs/PLAN.md task
    /// 214), read through `P` as it stands after a row that observes
    /// nothing, which carries the noise the next observation will charge:
    /// `q D²` for a short gap (the random walk's, to 1e-3 at `D = r/1000`),
    /// and `q / θ²` past many half-lives, however long the gap -- the same
    /// after 1,000 rows of null targets as after 10,000. Charged `q D²`, as
    /// task 211 did, it was 769 and then 7.7e4 times that bound there. The
    /// walk beside it keeps `q D²`. The longhand: the variance after the
    /// gap less the transition's `phi² P`.
    #[test]
    fn a_reverting_slots_gap_noise_is_bounded() {
        let (q, r) = (1e-3, 25.0);
        let theta = std::f64::consts::LN_2 / r;
        let mk = |revert: f64| {
            let mut c = plain(1.0, f64::INFINITY, Some(0.04));
            c.decay = Decay::Halflife(f64::INFINITY);
            c.q = Some(vec![q]);
            c.revert_half_life = vec![revert];
            let mut m = Kalman::new(c).unwrap();
            let mut s = 7u64;
            for i in 0..200 {
                let x = [1.0 + 0.3 * lcg(&mut s)];
                let y = 0.5 * x[0] + 0.2 * lcg(&mut s);
                m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
            }
            m
        };
        let var = |m: &Kalman| match m.coef_variance() {
            Some(crate::CoefVariance::Absolute(v)) => v[0][0],
            other => panic!("{other:?}"),
        };
        let added = |revert: f64, gap: f64, rows: usize| {
            let mut m = mk(revert);
            let before = var(&m);
            for _ in 0..rows {
                m.step(&[1.0], &[None], gap / rows as f64, 1.0);
            }
            let phi = Decay::Halflife(revert).factor(gap);
            var(&m) - phi * phi * before
        };
        let short = r / 1000.0;
        let got = added(r, short, 1) / (q * short * short);
        assert!((got - 1.0).abs() < 1e-3, "a short gap: {got} of q D²");
        let bound = q / (theta * theta);
        for rows in [1_000, 10_000] {
            let gap = rows as f64;
            let got = added(r, gap, rows);
            assert!(
                (got - bound).abs() <= 1e-9 * bound,
                "{rows} rows of null targets: {got} against the bound {bound}"
            );
            let walk = added(f64::INFINITY, gap, rows);
            assert!(
                (walk - q * gap * gap).abs() <= 1e-9 * q * gap * gap,
                "the walk: {walk}"
            );
        }
    }

    #[test]
    fn a_reverting_slot_forgets_a_stale_effect_and_a_random_walk_keeps_it() {
        // 300 rows identify a slope of 2 on `x1`, then `x1` goes flat at
        // zero for 300 rows: no row says anything about that slope any more.
        // The random walk carries the 2 for ever; the reverting filter lets
        // it go at `2^(-d/r)`. Predictions agree either way (`x1 = 0`), so
        // this is only visible in the coefficients -- the point of E41 is
        // what the filter believes when the evidence dries up.
        let run = |r: Vec<f64>| {
            let mut m = Kalman::new(KalmanCfg {
                standardize: false,
                ..revert_cfg(r)
            })
            .unwrap();
            let mut s = 12u64;
            for i in 0..600 {
                let x1 = if i < 300 { lcg(&mut s) } else { 0.0 };
                let x = [lcg(&mut s), x1];
                let y = 0.5 * x[0] + 2.0 * x[1] + 0.05 * lcg(&mut s);
                m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
            }
            m.coefficients()[0].clone()
        };
        let walk = run(vec![f64::INFINITY]);
        let revert = run(vec![f64::INFINITY, f64::INFINITY, 30.0]);
        assert!(
            (walk[2] - 2.0).abs() < 0.1,
            "random walk keeps the slope: {walk:?}"
        );
        // 300 rows at half-life 30 is 2^-10 of the slope.
        assert!(
            revert[2].abs() < 2.0 * 2f64.powi(-9),
            "reverting slot forgets it: {revert:?}"
        );
        // The slope still in evidence is learned equally well by both.
        assert!((walk[1] - 0.5).abs() < 0.05, "{walk:?}");
        assert!((revert[1] - 0.5).abs() < 0.05, "{revert:?}");
    }

    #[test]
    fn reversion_is_applied_once_per_row_under_share_p() {
        // One shared `P`, two targets: the transition runs once, not once
        // per target, or the shared covariance would shrink twice.
        let r = vec![f64::INFINITY, 10.0, 10.0];
        let mut m = Kalman::new(KalmanCfg {
            q: Some(vec![0.0; 3]),
            share_p: true,
            standardize: false,
            ..revert_cfg(r)
        })
        .unwrap();
        m.cfg.n_targets = 2;
        let mut m = Kalman::new(m.cfg.clone()).unwrap();
        let mut s = 13u64;
        for i in 0..50 {
            let x = [lcg(&mut s), lcg(&mut s)];
            m.step(
                &x,
                &[Some(x[0]), Some(-x[1])],
                if i == 0 { 0.0 } else { 1.0 },
                1.0,
            );
        }
        let p0 = m.p[0].clone();
        let b0 = m.beta.clone();
        m.step(&[0.1, 0.2], &[None, None], 10.0, 1.0);
        // phi = 2^-1 for both slopes over d = 10 at half-life 10.
        assert!((m.p[0][4] - p0[4] * 0.25).abs() <= 1e-12 * p0[4].abs());
        assert!((m.p[0][1] - p0[1] * 0.5).abs() <= 1e-12 * p0[1].abs());
        for (after, before) in m.beta.iter().zip(&b0) {
            assert!((after[1] - before[1] * 0.5).abs() <= 1e-12 * before[1].abs());
        }
    }

    #[test]
    fn a_zero_weight_row_still_advances_the_transition() {
        // Weight 0 means "advance the clock, learn nothing": the reversion
        // is clock, so it applies; the measurement update does not.
        let mut m = Kalman::new(KalmanCfg {
            q: Some(vec![0.0; 3]),
            standardize: false,
            ..revert_cfg(vec![4.0])
        })
        .unwrap();
        let mut s = 14u64;
        for i in 0..40 {
            let x = [lcg(&mut s), lcg(&mut s)];
            m.step(
                &x,
                &[Some(x[0] + x[1])],
                if i == 0 { 0.0 } else { 1.0 },
                1.0,
            );
        }
        let b0 = m.beta[0].clone();
        m.step(&[0.3, 0.3], &[Some(100.0)], 4.0, 0.0);
        for (i, (after, before)) in m.beta[0].iter().zip(&b0).enumerate() {
            assert!((after - 0.5 * before).abs() <= 1e-12 * before.abs(), "{i}");
        }
    }

    /// A map without `stats` is refused rather than defaulted, and so is the
    /// schema-2 layout's `cov` in its place.
    #[test]
    fn a_state_without_the_standardizer_is_refused() {
        let mut c = cfg(2, 1, vec![100.0]);
        c.revert_half_life = vec![500.0]; // JSON has no `inf`
        let m = fit(c, 30, 5);
        let v = serde_json::to_value(&m).unwrap();
        // The control: the same value loads with the field present.
        let back: Kalman = serde_json::from_value(v.clone()).unwrap();
        assert_eq!(back.stats, m.stats);
        let mut v = v;
        v.as_object_mut().unwrap().remove("stats");
        assert!(serde_json::from_value::<Kalman>(v).is_err());
        // Nor does the schema-2 layout it used to be told apart from: a
        // `cov` in place of `stats` is now simply a missing field.
        let mut v = serde_json::to_value(&m).unwrap();
        let obj = v.as_object_mut().unwrap();
        obj.remove("stats");
        obj.insert("cov".into(), serde_json::Value::Null);
        assert!(serde_json::from_value::<Kalman>(v).is_err());
    }

    /// The named msgpack of `value` with `edit` applied, read back: the path
    /// a saved state takes into the model, whose check is in its `TryFrom`.
    fn reread<T: serde::Serialize + serde::de::DeserializeOwned>(
        value: &T,
        edit: impl FnOnce(&mut rmpv::Value),
    ) -> Result<T, String> {
        let bytes = rmp_serde::to_vec_named(value).unwrap();
        let mut v = rmpv::decode::read_value(&mut bytes.as_slice()).unwrap();
        edit(&mut v);
        let mut out = Vec::new();
        rmpv::encode::write_value(&mut out, &v).unwrap();
        rmp_serde::from_slice(&out).map_err(|e| e.to_string())
    }

    /// The first coefficient vector of the state, one entry short.
    fn shorten_first(v: &mut rmpv::Value, field: &str) {
        let rmpv::Value::Map(entries) = v else {
            panic!("a map")
        };
        let (_, x) = entries
            .iter_mut()
            .find(|(k, _)| k.as_str() == Some(field))
            .unwrap_or_else(|| panic!("no {field}"));
        let rmpv::Value::Array(rows) = x else {
            panic!("{field} is a list")
        };
        let rmpv::Value::Array(first) = &mut rows[0] else {
            panic!("{field}[0] is a list")
        };
        first.pop();
    }

    /// A state whose coefficients or their covariance are not its cfg's shape
    /// is refused as it is read (review 2026-09-18, B3; docs/PLAN.md task
    /// 111: every other model had this test).
    #[test]
    fn a_state_of_the_wrong_shape_is_refused() {
        let m = Kalman::new(cfg(2, 1, vec![50.0])).unwrap();
        assert!(reread(&m, |_| {}).is_ok(), "the control");
        for field in ["beta", "p"] {
            let err = reread(&m, |v| shorten_first(v, field)).unwrap_err();
            assert!(err.contains("wrong shape"), "{field}: {err}");
        }
    }

    /// The list `field` of the state, one entry short.
    fn shorten(v: &mut rmpv::Value, field: &str) {
        let rmpv::Value::Map(entries) = v else {
            panic!("a map")
        };
        let (_, x) = entries
            .iter_mut()
            .find(|(k, _)| k.as_str() == Some(field))
            .unwrap_or_else(|| panic!("no {field}"));
        let rmpv::Value::Array(rows) = x else {
            panic!("{field} is a list")
        };
        rows.pop();
    }

    /// Each per-target list is checked on its own: a state one target short
    /// in its coefficients, its residual weights or its target weights alone
    /// is refused (task 158).
    #[test]
    fn a_state_one_target_short_in_any_list_is_refused() {
        let m = Kalman::new(cfg(2, 2, vec![50.0])).unwrap();
        assert!(reread(&m, |_| {}).is_ok(), "the control");
        for field in ["beta", "wsig", "wj"] {
            let err = reread(&m, |v| shorten(v, field)).unwrap_err();
            assert!(
                err.contains("kalman: state has the wrong shape"),
                "{field}: {err}"
            );
        }
    }

    /// Before any residual there is no observation variance to report, and
    /// `pred_var` says so with NaN rather than a 0 (task 158).
    #[test]
    fn pred_var_is_nan_before_any_residual() {
        let m = Kalman::new(cfg(2, 1, vec![50.0])).unwrap();
        assert!(m.pred_var(&[0.3, -0.2])[0].is_nan());
    }

    /// Each target's weight is the EW sum of the weights of the rows that
    /// carried it, null or weight-0 rows only ageing it, and the model
    /// reports it (task 158).
    #[test]
    fn the_target_weights_are_the_rows_that_carried_each_target() {
        let mut c = cfg(2, 2, vec![50.0]);
        c.decay = Decay::Halflife(7.0);
        let mut m = Kalman::new(c).unwrap();
        let mut want = [0.0f64; 2];
        let mut s = 97u64;
        for i in 0..40 {
            let x = [lcg(&mut s), lcg(&mut s)];
            let w = if i % 5 == 4 { 0.0 } else { 1.0 + lcg(&mut s) };
            let ys = [Some(x[0]), (i % 3 == 0).then_some(x[1])];
            let d = if i == 0 { 0.0 } else { 1.0 + f64::from(i % 2) };
            m.step(&x, &ys, d, w);
            let lam = 0.5f64.powf(d / 7.0);
            for (j, wj) in want.iter_mut().enumerate() {
                *wj = lam * *wj + if ys[j].is_some() { w } else { 0.0 };
            }
        }
        let mut out = vec![-1.0; 5];
        assert!(m.target_n_eff_into(&mut out));
        assert_eq!(out.len(), 2);
        for j in 0..2 {
            assert!(
                (out[j] - want[j]).abs() <= 1e-12 * want[j],
                "{out:?} against {want:?}"
            );
        }
        assert!(want[1] < want[0], "the targets' weights differ");
    }

    /// Without an intercept, `standardize` scales each feature by the root of
    /// its raw second moment, so the filter cannot tell a feature's units:
    /// the same stream with one feature in thousands and one in thousandths
    /// predicts the same, and a feature that is always 0 is left at 0 rather
    /// than divided by a zero scale. `predict` is the next `step`'s number
    /// to the bit (task 158).
    #[test]
    fn standardizing_without_an_intercept_does_not_see_the_units() {
        let run = |units: [f64; 2]| {
            let mut c = cfg(3, 1, vec![30.0]);
            c.fit_intercept = false;
            c.min_weight = 1.5;
            c.decay = Decay::Halflife(40.0);
            let mut m = Kalman::new(c).unwrap();
            let mut s = 101u64;
            let mut preds = Vec::new();
            for i in 0..200 {
                let x = [2.0 + lcg(&mut s), lcg(&mut s)];
                let y = 0.8 * x[0] - 1.5 * x[1] + 0.1 * lcg(&mut s);
                let xs = [units[0] * x[0], units[1] * x[1], 0.0];
                let d = if i == 0 { 0.0 } else { 1.0 };
                let ahead = m.predict(&xs, d).pred[0];
                // The first row meets no scale yet and is read raw, so it
                // carries no target: it only sets the scales.
                let got = m.step(&xs, &[(i > 0).then_some(y)], d, 1.0).pred[0];
                assert_eq!(
                    ahead.to_bits(),
                    got.to_bits(),
                    "row {i}: predict is step's number"
                );
                assert_eq!(got.is_nan(), i < 2, "row {i}: {got}");
                preds.push(got);
            }
            preds
        };
        let (a, b) = (run([1.0, 1.0]), run([1000.0, 0.001]));
        for (i, (pa, pb)) in a.iter().zip(&b).enumerate().skip(2) {
            assert!(
                (pa - pb).abs() <= 1e-9 * pa.abs().max(1.0),
                "row {i}: {pa} against {pb}"
            );
        }
    }

    /// The filter written from the module docs, unstandardized and without
    /// reversion: per target, `R` and the `Q` from `half_life` are both the
    /// EW residual variance, under `share_p` the mean over the targets that
    /// have one (review round 5, A1), read once a row before any target's
    /// update (review round 4, CC3); before
    /// there is one, the row's own innovation squared, under `share_p` the
    /// mean of the squares over the targets the row observes, and no noise
    /// (no `Q`, no correction) where there is none (CC4); `P` is unsized, 0,
    /// until a row has a noise, which sets it to `p0` times that noise, and
    /// after that takes `Q D²` once on each row that observes its target
    /// (any target, under `share_p`), `D` the clock since the last such row,
    /// before the first target's update (task 211);
    /// a target present at a positive weight corrects `b` and `P` by the gain
    /// `P z / (zᵀ P z + R / w)`, under `share_p` every target from `P` as the
    /// row found it and `P` once, after them (task 204); the residual
    /// variance is the EW mean of the
    /// squared out-of-sample errors, its weight ageing on every row. Two
    /// targets, one present one row in three, weights other than 1, shared
    /// and not (task 158).
    #[test]
    fn the_filter_is_its_recursion() {
        for share in [true, false] {
            let (h, hq) = (30.0, 50.0);
            let mut c = cfg(2, 2, vec![hq]);
            c.standardize = false;
            c.share_p = share;
            c.min_weight = 0.0;
            c.decay = Decay::Halflife(h);
            let mut m = Kalman::new(c).unwrap();
            let k = 3;
            let n_p = if share { 1 } else { 2 };
            let p0 = 1.0;
            let mut p = vec![vec![0.0f64; k * k]; n_p];
            let mut b = [[0.0f64; 3]; 2];
            let (mut sig2, mut wsig, mut wj) = ([0.0f64; 2], [0.0f64; 2], [0.0f64; 2]);
            let mut elapsed = vec![0.0f64; n_p];
            let mut s = 103u64;
            for i in 0..150 {
                let x = [lcg(&mut s), 1.0 + lcg(&mut s)];
                let ys = [
                    Some(0.5 + x[0] - 2.0 * x[1] + 0.2 * lcg(&mut s)),
                    (i % 3 == 0).then(|| -x[0] + 3.0 * lcg(&mut s)),
                ];
                let w = 0.5 + (lcg(&mut s) + 1.0);
                let d = if i == 0 { 0.0 } else { 1.0 };
                for e in elapsed.iter_mut() {
                    *e += d;
                }
                let lam = 0.5f64.powf(d / h);
                let z = [1.0, x[0], x[1]];
                let dotz = |v: &[f64]| -> f64 { (0..k).map(|a| z[a] * v[a]).sum() };
                let want: Vec<f64> = (0..2)
                    .map(|j| if wj[j] > 0.0 { dotz(&b[j]) } else { f64::NAN })
                    .collect();
                let got = m.step(&x, &ys, d, w).pred;
                for j in 0..2 {
                    assert_eq!(got[j].is_nan(), want[j].is_nan(), "share {share}, row {i}");
                    if want[j].is_finite() {
                        assert!(
                            (got[j] - want[j]).abs() <= 1e-10 * want[j].abs().max(1.0),
                            "share {share}, row {i}, target {j}: {} against {}",
                            got[j],
                            want[j]
                        );
                    }
                }
                // Each observed target's innovation squared, from the
                // coefficients before the row's first update.
                let e2: Vec<Option<f64>> = (0..2)
                    .map(|j| ys[j].map(|y| (y - dotz(&b[j])).powi(2)))
                    .collect();
                let seen: Vec<f64> = e2.iter().flatten().copied().collect();
                let shared_first = if seen.is_empty() {
                    0.0
                } else {
                    seen.iter().sum::<f64>() / seen.len() as f64
                };
                // Under `share_p` every target's noise is the mean residual
                // variance over the targets that have one, as the row
                // arrives, read once, before any target's update moves one
                // (review round 4, CC3; review round 5, A1): target 1 has
                // none until its second present row, row 3.
                let have: Vec<f64> = sig2.iter().copied().filter(|v| *v > 0.0).collect();
                let shared = if have.is_empty() {
                    0.0
                } else {
                    have.iter().sum::<f64>() / have.len() as f64
                };
                let mut shared_row: Option<(Vec<f64>, f64)> = None;
                for j in 0..2 {
                    let pi = if share { 0 } else { j };
                    let s2 = if share { shared } else { sig2[j] };
                    let sigma2 = if s2 > 0.0 {
                        s2
                    } else if share {
                        shared_first
                    } else {
                        e2[j].unwrap_or(0.0)
                    };
                    let informs = if share {
                        ys.iter().any(Option::is_some)
                    } else {
                        ys[j].is_some()
                    };
                    if (!share || j == 0) && informs {
                        let dd = elapsed[pi] * elapsed[pi];
                        elapsed[pi] = 0.0;
                        if p[pi].iter().all(|v| *v == 0.0) {
                            for a in 0..k {
                                p[pi][a * k + a] = p0 * sigma2;
                            }
                        } else {
                            let q = sigma2 * (std::f64::consts::LN_2 / hq).powi(2);
                            for a in 0..k {
                                p[pi][a * k + a] += q * dd;
                            }
                        }
                    }
                    let Some(y) = ys[j] else {
                        wj[j] *= lam;
                        wsig[j] *= lam;
                        continue;
                    };
                    // Under `share_p` every target reads `P` as the row
                    // found it: the shared update waits for the loop's end.
                    let pz: Vec<f64> = (0..k).map(|a| dotz(&p[pi][a * k..(a + 1) * k])).collect();
                    let s_inn = dotz(&pz) + sigma2 / w;
                    let err = y - dotz(&b[j]);
                    if sigma2 > 0.0 {
                        for a in 0..k {
                            b[j][a] += pz[a] / s_inn * err;
                        }
                        if share {
                            shared_row = Some((pz, s_inn));
                        } else {
                            for a in 0..k {
                                for bb in 0..k {
                                    p[pi][a * k + bb] -= pz[a] * pz[bb] / s_inn;
                                }
                            }
                        }
                    }
                    let aged = lam * wsig[j];
                    wsig[j] = aged;
                    if want[j].is_finite() {
                        let r = y - want[j];
                        sig2[j] = (aged * sig2[j] + w * r * r) / (aged + w);
                        wsig[j] = aged + w;
                    }
                    wj[j] = lam * wj[j] + w;
                }
                // `P` takes the row once, if any target updated: every one
                // read the same `P z` and the same noise (task 204).
                if let Some((pz, s_inn)) = shared_row {
                    for a in 0..k {
                        for bb in 0..k {
                            p[0][a * k + bb] -= pz[a] * pz[bb] / s_inn;
                        }
                    }
                }
            }
        }
    }

    /// A row of weight 0 learns nothing, the residual variance included:
    /// across every such row `σ²` keeps its bits, not `(a σ²) / a` (task
    /// 158).
    #[test]
    fn a_zero_weight_row_keeps_the_residual_variance_to_the_bit() {
        let mut c = cfg(2, 1, vec![50.0]);
        c.decay = Decay::Halflife(9.0);
        c.min_weight = 0.0;
        let mut m = Kalman::new(c).unwrap();
        let mut s = 107u64;
        let mut checked = 0;
        for i in 0..300 {
            let x = [lcg(&mut s), lcg(&mut s)];
            let y = x[0] - x[1] + 0.3 * lcg(&mut s);
            let w = if i % 3 == 2 { 0.0 } else { 1.0 };
            let before = m.sigma2()[0];
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 0.7 }, w);
            if w == 0.0 && before > 0.0 {
                assert_eq!(m.sigma2()[0].to_bits(), before.to_bits(), "row {i}");
                checked += 1;
            }
        }
        assert!(checked > 90);
    }

    /// A row whose innovation variance is 0 -- a zero regressor and an
    /// observation variance that underflows at the row's weight -- is
    /// skipped, not divided by; and so is a target that is not a number
    /// (task 158).
    #[test]
    fn an_innovation_of_zero_or_a_target_not_a_number_corrects_nothing() {
        let mut m = Kalman::new(KalmanCfg {
            n_features: 1,
            n_targets: 1,
            fit_intercept: false,
            decay: Decay::Halflife(f64::INFINITY),
            half_life: vec![f64::INFINITY],
            q: None,
            obs_var: Some(f64::from_bits(1)),
            p0: 1.0,
            share_p: false,
            min_weight: 0.0,
            revert_half_life: vec![f64::INFINITY],
            standardize: false,
        })
        .unwrap();
        assert_eq!(f64::from_bits(1) / 4.0, 0.0, "R / w underflows");
        m.step(&[0.0], &[Some(1.0)], 0.0, 4.0);
        assert_eq!(m.beta[0], vec![0.0], "s = 0: skipped");
        m.step(&[1.0], &[Some(f64::NAN)], 1.0, 1.0);
        assert_eq!(m.beta[0], vec![0.0], "a NaN target: skipped");
        assert!(m.step(&[1.0], &[Some(2.0)], 1.0, 1.0).pred[0].is_finite());
    }

    /// Before a target has a residual variance, a row that gives its noise
    /// no scale leaves the filter as it was, to the bit, though the clock
    /// moves: an innovation of exactly 0, a row of weight 0, a null target.
    /// Taken as `R = 0`, an exact row would collapse `P` along `z`, so the
    /// filter would hold its first fit with no doubt at all; the other two
    /// have no innovation to read. None of them corrects anything or adds
    /// process noise from `coef_half_life`, and the next row that has an
    /// innovation corrects the filter (CC4). With a shared `P`, the same when
    /// no target the row observes has an innovation other than 0.
    #[test]
    fn rows_that_size_no_noise_leave_the_filter_as_it_was() {
        for share in [false, true] {
            let mut c = cfg(2, 2, vec![40.0]);
            c.standardize = false;
            c.min_weight = 0.0;
            c.share_p = share;
            let fresh = Kalman::new(c).unwrap();
            let mut m = fresh.clone();
            // `b = 0`, so a target of 0 is an innovation of 0; clock steps of
            // 2, where a process noise would show.
            m.step(&[0.3, -0.2], &[Some(0.0), Some(0.0)], 2.0, 1.0);
            m.step(&[-0.5, 0.4], &[Some(0.0), None], 2.0, 1.0);
            m.step(&[0.7, 0.1], &[Some(3.0), Some(-2.0)], 2.0, 0.0);
            m.step(&[0.2, 0.2], &[None, None], 2.0, 1.0);
            assert_eq!(m.beta, fresh.beta, "share {share}");
            assert_eq!(m.p, fresh.p, "share {share}: no collapse, no process noise");
            assert_eq!(
                m.sig2,
                vec![0.0, 0.0],
                "share {share}: still no residual variance"
            );
            // Innovations of 1 and -1: the prior is sized at `p0 e² = 1`, then
            // narrowed by the correction.
            m.step(&[0.1, 0.6], &[Some(1.0), Some(-1.0)], 1.0, 1.0);
            assert!(
                m.beta.iter().all(|b| b.iter().any(|v| *v != 0.0)),
                "share {share}"
            );
            let p00 = m.p[0][0];
            assert!(
                p00 > 0.0 && p00 < 1.0,
                "share {share}: P sized and narrowed, {p00}"
            );
        }
    }

    /// `p0` is a ratio: the prior variance is `p0` times the first noise
    /// estimate, placed on the row that first sizes the noise, before its
    /// gain, with no process noise on that row (CC4). Until then `P` is
    /// unsized, all zero, and nothing adds to it, an explicit `q` included:
    /// not an exact observation, a null target at a gap, nor a row of weight
    /// 0. So the first correction, `P_0 z e / (zᵀ P_0 z + e² / w)` with
    /// `P_0 = p0 e² I`, is `p0 z e / (p0 zᵀz + 1 / w)`, and does not see the
    /// target's units. With `obs_var` given the noise is known from the
    /// start, and `P_0 = p0 obs_var I` from construction.
    #[test]
    fn the_prior_is_p0_times_the_first_noise() {
        for share in [false, true] {
            let mut c = cfg(2, 1, vec![40.0]);
            c.standardize = false;
            c.min_weight = 0.0;
            c.share_p = share;
            c.p0 = 0.5;
            c.q = Some(vec![0.1, 0.2, 0.3]);
            let mut m = Kalman::new(c).unwrap();
            let no_prior = |m: &Kalman| m.p[0].iter().all(|v| *v == 0.0);
            assert!(no_prior(&m), "share {share}: unsized from the start");
            m.step(&[0.3, -0.2], &[Some(0.0)], 0.0, 1.0);
            m.step(&[0.1, 0.4], &[None], 3.0, 1.0);
            m.step(&[0.7, 0.1], &[Some(5.0)], 3.0, 0.0);
            assert!(no_prior(&m), "share {share}: still unsized");
            let (x, y, w) = ([0.2, -0.6], 3.0, 2.0);
            m.step(&x, &[Some(y)], 3.0, w);
            let z = [1.0, x[0], x[1]];
            let zz: f64 = z.iter().map(|v| v * v).sum();
            let p0e2 = 0.5 * y * y;
            for i in 0..3 {
                let want = 0.5 * z[i] * y / (0.5 * zz + 1.0 / w);
                let got = m.beta[0][i];
                assert!(
                    (got - want).abs() <= 1e-14 * want.abs(),
                    "share {share}, b[{i}]: {got} against {want}"
                );
                for jj in 0..3 {
                    let eye = if i == jj { 1.0 } else { 0.0 };
                    let want = p0e2 * eye - p0e2 * p0e2 * z[i] * z[jj] / (p0e2 * zz + y * y / w);
                    let got = m.p[0][i * 3 + jj];
                    assert!(
                        (got - want).abs() <= 1e-14 * p0e2,
                        "share {share}, P[{i},{jj}]: {got} against {want}"
                    );
                }
            }
        }
        // Unstandardized every slot; standardized the intercept, each
        // feature's prior waiting for its scale (task 211).
        for standardize in [false, true] {
            let mut c = cfg(2, 1, vec![40.0]);
            c.obs_var = Some(0.25);
            c.p0 = 3.0;
            c.standardize = standardize;
            let m = Kalman::new(c).unwrap();
            for i in 0..3 {
                for j in 0..3 {
                    let sized = i == j && (i == 0 || !standardize);
                    assert_eq!(m.p[0][i * 3 + j], if sized { 0.75 } else { 0.0 });
                }
            }
        }
    }

    /// A target that is not a number has no innovation, so before the first
    /// residual it sizes no noise: its own `P` is left unsized, with no
    /// process noise, where its square would have put a NaN on the diagonal.
    /// Under `share_p` the shared noise is then the other targets', and they
    /// learn on the row with it (CC4). Here that is `R = 4`, the finite
    /// target's innovation of 2, squared, against the prior it sizes on the
    /// row, `P = p0 R I = 4 I`.
    #[test]
    fn a_target_not_a_number_sizes_no_noise() {
        for share in [false, true] {
            let mut c = cfg(2, 2, vec![40.0]);
            c.standardize = false;
            c.min_weight = 0.0;
            c.share_p = share;
            let fresh = Kalman::new(c).unwrap();
            let mut m = fresh.clone();
            m.step(&[0.3, -0.2], &[Some(f64::NAN), Some(2.0)], 1.0, 1.0);
            assert_eq!(m.beta[0], fresh.beta[0], "share {share}");
            if !share {
                assert_eq!(m.p[0], fresh.p[0], "the NaN target's own P");
            }
            let z = [1.0, 0.3, -0.2];
            let zz: f64 = z.iter().map(|v| v * v).sum();
            for (i, b) in m.beta[1].iter().enumerate() {
                let want = 4.0 * z[i] * 2.0 / (4.0 * zz + 4.0);
                assert!(
                    (b - want).abs() <= 1e-15,
                    "share {share}, slot {i}: {b} against {want}"
                );
            }
        }
    }

    /// Scaling every target by `c` scales every prediction by `c`, and `σ²`
    /// by `c²`, at any `p0`, once an `obs_var` or `q` given in the target's
    /// units is scaled by `c²` with it: the filter has no unit of its own.
    /// Before a target's first residual its noise was the literal 1.0, in the
    /// target's units, so the warm-up's gains, and every prediction after
    /// them, moved with the units: 100% apart between scales of 1e-6 and 1e6
    /// (review 2026-10-05, CC4). The prior variance was `p0` in the target's
    /// units too, and is now `p0` times the first noise estimate. Powers of
    /// two keep every operation exact, so the comparison is to the bit. The
    /// stream opens on an exact observation (an innovation of 0), a row of
    /// weight 0 and a gap of five clock units fall before any residual, and
    /// the second target joins on that gap's row; per-target and shared `P`,
    /// standardized and not, reverting and not, `p0` other than 1, and a
    /// fixed `obs_var`.
    #[test]
    fn scaling_the_targets_scales_every_prediction_to_the_bit() {
        type Row = ([f64; 2], [Option<f64>; 2], f64, f64);
        let mut s = 109u64;
        let rows: Vec<Row> = (0..160)
            .map(|i| {
                let x = [lcg(&mut s), 1.0 + lcg(&mut s)];
                let y0 = 0.5 + x[0] - 2.0 * x[1] + 0.2 * lcg(&mut s);
                let y1 = -x[0] + 0.7 * x[1] + 0.3 * lcg(&mut s);
                let ys = [
                    match i {
                        0 => Some(0.0),
                        _ if i % 11 == 7 => None,
                        _ => Some(y0),
                    },
                    (i >= 3 && i % 5 != 2).then_some(y1),
                ];
                let d = match i {
                    0 => 0.0,
                    3 => 5.0,
                    _ => 1.0,
                };
                let w = if i == 2 { 0.0 } else { 1.5 + lcg(&mut s) };
                (x, ys, d, w)
            })
            .collect();
        // Predictions, the filter after the stream, and how many target-rows
        // the filter learned from before that target had a residual variance.
        let run = |base: &KalmanCfg, c: f64| {
            let mut cfg = base.clone();
            cfg.obs_var = cfg.obs_var.map(|v| v * c * c);
            cfg.q = cfg.q.map(|q| q.iter().map(|v| v * c * c).collect());
            let mut m = Kalman::new(cfg).unwrap();
            let mut before_sigma = 0;
            let mut preds = Vec::new();
            for (x, ys, d, w) in &rows {
                let ys: Vec<Option<f64>> = ys.iter().map(|y| y.map(|v| v * c)).collect();
                for (j, y) in ys.iter().enumerate() {
                    let s2 = if m.cfg.share_p {
                        m.sig2.iter().sum::<f64>()
                    } else {
                        m.sig2[j]
                    };
                    if y.is_some() && *w > 0.0 && s2 == 0.0 {
                        before_sigma += 1;
                    }
                }
                preds.push(m.step(x, &ys, *d, *w).pred);
            }
            (preds, m, before_sigma)
        };
        let mut cases = Vec::new();
        for share in [false, true] {
            for standardize in [true, false] {
                let mut c = cfg(2, 2, vec![40.0]);
                c.decay = Decay::Halflife(30.0);
                c.min_weight = 4.0;
                c.share_p = share;
                c.standardize = standardize;
                cases.push((format!("share {share}, standardize {standardize}"), c));
            }
        }
        let mut c = cfg(2, 2, vec![40.0]);
        c.min_weight = 4.0;
        c.revert_half_life = vec![f64::INFINITY, 20.0, 6.0];
        cases.push(("reverting".into(), c.clone()));
        c.q = Some(vec![0.0, 1e-3, 2e-3]);
        cases.push(("reverting, q given".into(), c));
        let mut c = cfg(2, 2, vec![40.0]);
        c.min_weight = 4.0;
        c.p0 = 0.25;
        cases.push(("p0 = 1/4".into(), c.clone()));
        c.obs_var = Some(0.25);
        cases.push(("p0 = 1/4, obs_var given".into(), c));

        for (name, base) in &cases {
            let (want, m1, learned) = run(base, 1.0);
            assert!(learned >= 4, "{name}: the warm-up was learned from");
            let first = (0..want.len()).find(|&i| want[i][0].is_finite()).unwrap();
            assert!(first > 3, "{name}: the first prediction follows the gap");
            for c in [2f64.powi(20), 2f64.powi(-20)] {
                let (got, mc, _) = run(base, c);
                for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                    for j in 0..2 {
                        let scaled = w[j] * c;
                        assert!(
                            g[j].to_bits() == scaled.to_bits() || (g[j].is_nan() && w[j].is_nan()),
                            "{name}, c = {c:e}, row {i}, target {j}: {} against {} \
                             ({:.2e} apart, relative)",
                            g[j],
                            scaled,
                            ((g[j] - scaled) / scaled).abs()
                        );
                    }
                }
                for j in 0..2 {
                    assert_eq!(
                        mc.sig2[j].to_bits(),
                        (m1.sig2[j] * c * c).to_bits(),
                        "{name}"
                    );
                    for (a, b) in mc.coefficients()[j].iter().zip(&m1.coefficients()[j]) {
                        assert_eq!(a.to_bits(), (b * c).to_bits(), "{name}");
                    }
                }
            }
        }
    }

    /// One feature through the origin, unstandardized, at `min_weight` 0:
    /// the filter the tests below start from.
    fn plain(p0: f64, half_life: f64, obs_var: Option<f64>) -> KalmanCfg {
        KalmanCfg {
            n_features: 1,
            n_targets: 1,
            fit_intercept: false,
            decay: Decay::Halflife(50.0),
            half_life: vec![half_life],
            q: None,
            obs_var,
            p0,
            share_p: false,
            min_weight: 0.0,
            revert_half_life: vec![f64::INFINITY],
            standardize: false,
        }
    }

    /// A covariance is unsized only when its whole diagonal is 0 (the module
    /// doc, CC4). A slot the reversion takes to exactly 0 across a step,
    /// beside one that still holds its variance, leaves `P` sized: the row
    /// adds its process noise to it (none here), as to any sized `P`, and
    /// the variance the other slot has learned carries on. Sized afresh at
    /// `p0` times the noise, `P` would forget it. Slot 0 reverts at a
    /// half-life of `1e-4`, so a step of 1 takes it and its covariances to
    /// exactly 0 (`2^-10000`); slot 1 is a random walk with no process
    /// noise. From the second row on slot 1 alone learns, and its variance
    /// is a one-slot filter's, the oracle in information form: `1 / P11` is
    /// the first row's marginal, `(1 + z0² + z1²) / (1 + z0²)` at
    /// `P_0 = I`, plus `x1²` a row at `R = 1`.
    #[test]
    fn a_slot_reverted_to_zero_leaves_the_covariance_sized() {
        let mut c = plain(1.0, f64::INFINITY, Some(1.0));
        c.n_features = 2;
        c.q = Some(vec![0.0, 0.0]);
        c.revert_half_life = vec![1e-4, f64::INFINITY];
        let mut m = Kalman::new(c).unwrap();
        let z = [1.0, 2.0];
        m.step(&z, &[Some(0.5)], 0.0, 1.0);
        let mut info = (1.0 + z[0] * z[0] + z[1] * z[1]) / (1.0 + z[0] * z[0]);
        let mut s = 113u64;
        for i in 1..40 {
            let x = [lcg(&mut s), 0.5 + lcg(&mut s)];
            m.step(&x, &[Some(x[0] - 2.0 * x[1])], 1.0, 1.0);
            info += x[1] * x[1];
            let p = &m.p[0];
            assert_eq!(p[0], 0.0, "row {i}: slot 0 holds nothing");
            assert!(
                (p[3] * info - 1.0).abs() <= 1e-12,
                "row {i}: P11 = {} against {}",
                p[3],
                1.0 / info
            );
        }
    }

    /// A first noise so large that `p0` times it overflows sizes no prior:
    /// `P` stays unsized, all zero and finite, where an infinite diagonal
    /// would make every later innovation variance infinite, refuse every
    /// correction and end the filter's learning for good. The next row
    /// whose noise sizes a finite prior sizes it, and the filter learns
    /// from it. A target at the input bound is within the model's contract,
    /// as is any `p0 > 0`: here `p0 e² = 1e200 · 1e200`.
    #[test]
    fn a_prior_that_would_overflow_is_not_sized() {
        let mut m = Kalman::new(plain(1e200, f64::INFINITY, None)).unwrap();
        m.step(&[1.0], &[Some(crate::INPUT_BOUND)], 0.0, 1.0);
        assert_eq!(m.p[0], vec![0.0], "unsized, and finite");
        assert_eq!(m.beta[0], vec![0.0]);
        // An innovation of 3 sizes a finite prior, `p0 · 9`, and corrects.
        m.step(&[1.0], &[Some(3.0)], 1.0, 1.0);
        assert!(m.p[0].iter().all(|v| v.is_finite()), "{:?}", m.p[0]);
        assert_eq!(m.beta[0], vec![3.0], "learned");
    }

    /// Once `P` is sized, a row whose innovation is exactly 0 before the
    /// target has a residual variance still sizes no noise, and corrects
    /// nothing: at `R = 0` the gain would be `P z / zᵀ P z`, taking the row
    /// as exact and collapsing `P` along `z`, here to 0, so the filter would
    /// hold its fit with no doubt left (CC4, the module doc). `P` and `b`
    /// keep their bits. The row before sized `P` at `p0 e² = 1` and
    /// narrowed it to 1/2; the process noise derived from `σ² = 0` adds
    /// nothing.
    #[test]
    fn an_exact_row_after_the_prior_is_sized_corrects_nothing() {
        let mut m = Kalman::new(plain(1.0, 40.0, None)).unwrap();
        m.step(&[1.0], &[Some(1.0)], 0.0, 1.0);
        assert_eq!((m.p[0][0], m.beta[0][0]), (0.5, 0.5), "sized, narrowed");
        assert_eq!(m.sig2, vec![0.0], "no residual variance yet");
        let pred = m.step(&[1.0], &[Some(0.5)], 1.0, 1.0).pred[0];
        assert_eq!(pred, 0.5, "the row is its prediction exactly");
        assert_eq!((m.p[0][0], m.beta[0][0]), (0.5, 0.5));
        assert_eq!(m.sig2, vec![0.0], "every residual so far exactly 0");
    }

    // ---- readiness (docs/PLAN.md task 116) ----

    /// A standard normal, near enough: twelve uniforms, centred.
    fn normal(s: &mut u64) -> f64 {
        (0..12).map(|_| 0.5 * (lcg(s) + 1.0)).sum::<f64>() - 6.0
    }

    /// `error_inflation` is `sqrt(1 + z' P⁻ z / R)`, the prior predictive
    /// variance over the noise, with `P⁻` written here from the module
    /// doc's recursion -- `b <- Phi b`, `P <- Phi P Phi` on every row, `P <-
    /// P + Q D²` with `D` the clock since the last observation on a row that
    /// observes the target, the gain `P z / (z' P z + R / w)`, `P <- P − g
    /// z' P` -- in its own loops, on a
    /// reverting slot, irregular steps, a zero-weight row and a null
    /// target, with a fixed `obs_var` so `R` is known. The gate reads the
    /// same per-row value.
    #[test]
    fn the_row_statistic_is_the_prior_predictive_variance_over_the_noise() {
        let (r, coef_hl) = (0.3, 40.0);
        let mut c = plain(2.0, coef_hl, Some(r));
        c.n_features = 2;
        c.fit_intercept = true;
        c.revert_half_life = vec![f64::INFINITY, 25.0, 60.0];
        let k = 3;
        let mut m = Kalman::new(c).unwrap();
        let mut b = vec![0.0; k];
        let mut p = vec![0.0; k * k];
        for i in 0..k {
            p[i * k + i] = 2.0 * r;
        }
        let q = r * (std::f64::consts::LN_2 / coef_hl).powi(2);
        let revert = [f64::INFINITY, 25.0, 60.0];
        let mut s = 17u64;
        let mut checked = 0;
        // The clock since the last observation (task 211).
        let mut elapsed = 0.0;
        for i in 0..300 {
            let x = [3.0 * lcg(&mut s), 1.0 + lcg(&mut s)];
            let d = match i {
                0 => 0.0,
                _ if i % 7 == 3 => 4.0,
                _ => 1.0,
            };
            let w = if i % 11 == 5 { 0.0 } else { 1.0 };
            let y = (i % 13 != 8).then(|| 0.5 + x[0] - 2.0 * x[1] + 0.4 * lcg(&mut s));
            let z = [1.0, x[0], x[1]];
            let phi: Vec<f64> = revert
                .iter()
                .map(|h| Decay::Halflife(*h).factor(d))
                .collect();
            for (bi, ph) in b.iter_mut().zip(&phi) {
                *bi *= ph;
            }
            for a in 0..k {
                for e in 0..k {
                    p[a * k + e] *= phi[a] * phi[e];
                }
            }
            // The prior the row's update would start from: `Q (D + d)²` for
            // the clock since the last observation, this row's included,
            // charged to `P` only if the row observes the target; on a
            // reverting slot `(D + d)²` is `((1 − 2^(−(D + d)/r)) / θ)²`,
            // `θ = ln 2 / r` (task 214).
            elapsed += d;
            let mut prior = p.clone();
            for a in 0..k {
                let gap2 = if revert[a].is_infinite() {
                    elapsed * elapsed
                } else {
                    let theta = std::f64::consts::LN_2 / revert[a];
                    ((1.0 - Decay::Halflife(revert[a]).factor(elapsed)) / theta).powi(2)
                };
                prior[a * k + a] += q * gap2;
            }
            let pz: Vec<f64> = (0..k)
                .map(|a| (0..k).map(|e| prior[a * k + e] * z[e]).sum())
                .collect();
            let zpz: f64 = z.iter().zip(&pz).map(|(a, e)| a * e).sum();
            let want = (1.0 + zpz / r).sqrt();
            let (mut row, mut gate) = (Vec::new(), Vec::new());
            assert!(m.row_error_inflation_into(&x, d, &mut row));
            assert!(m.error_inflation_gate_into(&x, d, &mut gate, 1.1));
            assert!(
                (row[0] - want).abs() <= 1e-12 * want,
                "row {i}: {} vs {want}",
                row[0]
            );
            assert_eq!(row[0].to_bits(), gate[0].to_bits(), "row {i}");
            checked += 1;
            m.step(&x, &[y], d, w);
            if let (Some(y), true) = (y, w > 0.0) {
                p = prior;
                elapsed = 0.0;
                let s_inn = zpz + r / w;
                let err = y - z.iter().zip(&b).map(|(a, e)| a * e).sum::<f64>();
                let g: Vec<f64> = pz.iter().map(|v| v / s_inn).collect();
                for a in 0..k {
                    b[a] += g[a] * err;
                    for e in 0..k {
                        p[a * k + e] -= g[a] * pz[e];
                    }
                }
            }
        }
        assert_eq!(checked, 300);
    }

    /// The noise is the state's, never the row's own innovation, which
    /// reads its target (hard rule 2; the bank's test perturbs the target):
    /// without `obs_var` the ratio is infinite while `P` is unsized and
    /// until the target has a residual variance.
    #[test]
    fn the_row_statistic_is_infinite_until_there_is_a_noise() {
        let mut m = Kalman::new(cfg(2, 1, vec![50.0])).unwrap();
        let mut out = Vec::new();
        let mut s = 5u64;
        let mut finite_from = None;
        for i in 0..40 {
            let x = [lcg(&mut s), lcg(&mut s)];
            let y = 1.0 + x[0] + 0.2 * lcg(&mut s);
            m.row_error_inflation_into(&x, 1.0, &mut out);
            let unknown = m.is_unsized(0) || m.sig2[0] <= 0.0;
            assert_eq!(out[0].is_infinite(), unknown, "row {i}");
            if !unknown {
                finite_from.get_or_insert(i);
            }
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        // `min_weight` 10 holds the first predictions back, and the noise
        // waits for the first residual, after them.
        assert!(finite_from.is_some_and(|i| i >= 10), "{finite_from:?}");
    }

    /// Under `share_p` every target reads the one `P` and the mean noise:
    /// one value for all of them.
    #[test]
    fn share_p_reads_one_statistic_for_every_target() {
        let mut c = cfg(2, 3, vec![50.0]);
        c.share_p = true;
        c.min_weight = 0.0;
        let m = fit(c, 120, 9);
        let mut out = Vec::new();
        assert!(m.row_error_inflation_into(&[0.3, 0.1], 1.0, &mut out));
        assert_eq!(out.len(), 3);
        assert!(out[0].is_finite() && out[0] > 1.0, "{out:?}");
        assert!(
            out.iter().all(|v| v.to_bits() == out[0].to_bits()),
            "{out:?}"
        );
    }

    /// `se_coef`'s variances are `T P Tᵀ`'s diagonal: through `coef`'s own
    /// read-out, a prediction's variance at a row is the quadratic form of
    /// `P` in its standardized coordinates, so at the features' origin it
    /// is the intercept's variance, and a slope's is `P_ii / s_i²`.
    /// Unstandardized, `T` is the identity and they are `P`'s diagonal.
    #[test]
    fn the_coefficient_variances_are_the_posterior_in_coefs_units() {
        let mut c = cfg(2, 1, vec![50.0]);
        c.min_weight = 0.0;
        let m = fit(c.clone(), 200, 3);
        let Some(crate::CoefVariance::Absolute(v)) = m.coef_variance() else {
            panic!("kalman's variances are absolute");
        };
        let k = 3;
        // The intercept is the prediction at `x = 0`, read in the anchor's
        // coordinates `P` is held in; a slope is `b_i / s^a_i`.
        let z0 = m.anchored(&[0.0, 0.0]);
        let quad: f64 = (0..k)
            .map(|a| {
                (0..k)
                    .map(|e| z0[a] * m.p[0][a * k + e] * z0[e])
                    .sum::<f64>()
            })
            .sum();
        assert!(
            (v[0][0] - quad).abs() <= 1e-12 * quad,
            "{} vs {quad}",
            v[0][0]
        );
        let s = &m.anchor_scale;
        for i in 1..k {
            let want = m.p[0][i * k + i] / (s[i] * s[i]);
            assert!((v[0][i] - want).abs() <= 1e-12 * want, "{i}");
        }
        c.standardize = false;
        let m = fit(c, 200, 3);
        let Some(crate::CoefVariance::Absolute(v)) = m.coef_variance() else {
            panic!()
        };
        let diag: Vec<f64> = (0..k).map(|i| m.p[0][i * k + i]).collect();
        assert_eq!(v[0], diag);
        // Unsized, nothing is estimated.
        let fresh = Kalman::new(cfg(2, 1, vec![50.0])).unwrap();
        let Some(crate::CoefVariance::Absolute(v)) = fresh.coef_variance() else {
            panic!()
        };
        assert!(v[0].iter().all(|x| x.is_nan()));
    }

    /// The mean of the per-row `h` over rows 5,000 to 20,000, the mean of
    /// the summary's mean field over the same rows, and the doc's `p`, for
    /// `n_feat` standardized, uncorrelated features and a constant truth
    /// at `coef_half_life`, rows one clock unit apart.
    fn settled_readiness(n_feat: usize, coef_hl: f64) -> (f64, f64, f64) {
        let c = KalmanCfg {
            n_features: n_feat,
            n_targets: 1,
            fit_intercept: true,
            decay: Decay::Halflife(200.0),
            half_life: vec![coef_hl],
            q: None,
            obs_var: None,
            p0: 1.0,
            share_p: false,
            min_weight: 0.0,
            revert_half_life: vec![f64::INFINITY],
            standardize: true,
        };
        let mut m = Kalman::new(c).unwrap();
        let mut s = 29u64;
        let (mut sum_h, mut sum_field, mut rows) = (0.0, 0.0, 0.0);
        let mut out = Vec::new();
        for i in 0..20_000 {
            let x: Vec<f64> = (0..n_feat).map(|_| normal(&mut s)).collect();
            let truth: f64 = x
                .iter()
                .enumerate()
                .map(|(j, v)| (j as f64 - 4.0) * v)
                .sum();
            let y = 1.0 + truth + normal(&mut s);
            let d = if i == 0 { 0.0 } else { 1.0 };
            if i >= 5_000 {
                m.row_error_inflation_into(&x, d, &mut out);
                sum_h += out[0] * out[0] - 1.0;
                m.error_inflation_into(&mut out);
                sum_field += out[0] * out[0] - 1.0;
                rows += 1.0;
            }
            m.step(&x, &[Some(y)], d, 1.0);
        }
        let c = std::f64::consts::LN_2 / coef_hl;
        let p = (c * c + (c.powi(4) + 4.0 * c * c).sqrt()) / 2.0;
        (sum_h / rows, sum_field / rows, p)
    }

    /// The floor the module doc derives: under process noise `P⁻` never
    /// reaches 0, and per direction `p = P⁻ / R` settles where
    /// `p² / (p + 1) = c²`, `c = ln 2 · d / coef_half_life`, so the average
    /// row reads about `sqrt(1 + k p)` -- 1.0675 at `k = 10`, `d = 1`,
    /// `coef_half_life = 50`. Measured (task 116, 2026-10-07) on
    /// standardized, uncorrelated features: the mean of `h` is 6.0% above
    /// `k p` there (`sqrt(1 + h)` 1.0715), and the summary's mean field
    /// 5.3% above `k p / (p + 1)`, its value at the posterior. The mean
    /// field is a slow-drift reading: at `coef_half_life = 5` the mean of
    /// `h` is 1.60 times `k p` (`sqrt(1 + h)` 1.84 against the formula's
    /// 1.58), each row's rank-one update narrowing one direction where the
    /// mean field spreads it over all of them.
    #[test]
    fn the_average_row_settles_near_the_floor_the_doc_derives() {
        let k = 10.0;
        let (mean_h, field, p) = settled_readiness(9, 50.0);
        let floor = k * p;
        assert!(((1.0 + floor).sqrt() - 1.0675).abs() < 1e-4, "{floor}");
        assert!(
            mean_h > floor && mean_h < 1.08 * floor,
            "mean h {mean_h} vs k p {floor}"
        );
        let posterior = floor / (p + 1.0);
        assert!(
            field > posterior && field < 1.08 * posterior,
            "mean field {field} vs k p / (p + 1) {posterior}"
        );
        let (fast, _, p5) = settled_readiness(9, 5.0);
        let ratio = fast / (k * p5);
        assert!(ratio > 1.5 && ratio < 1.7, "{ratio}");
    }

    /// Review round 5, A1: under `share_p` the noise is the mean `σ²` over
    /// the targets that have a residual variance, so a target that has none
    /// yet is no noise estimate rather than a noise of 0. A target beside a
    /// copy of itself that is null for its first 150 rows predicts as it
    /// would alone, to the bit, through the row after the copy gains a
    /// residual variance of its own (the noise is read before the row's
    /// updates), and `pred_var` with it; from there the mean takes the
    /// copy's variance in. The mean over all the targets halved the noise
    /// while the copy was null: `pred_var` parted at row 4, 0.181 against
    /// 0.208 alone, and the probe through the bank moved the target's
    /// predictions by up to 0.45.
    #[test]
    fn share_p_a_copy_null_for_its_first_rows_moves_nothing_until_it_has_a_noise() {
        let mk = |m: usize| {
            let mut c = cfg(2, m, vec![50.0]);
            c.share_p = true;
            c.min_weight = 3.0;
            Kalman::new(c).unwrap()
        };
        let (mut alone, mut pair) = (mk(1), mk(2));
        let mut s = 7u64;
        let mut noised: Option<usize> = None;
        let mut parted = false;
        for i in 0..400usize {
            let x = [lcg(&mut s), lcg(&mut s)];
            let y = 1.0 - 2.0 * x[0] + x[1] + 0.3 * lcg(&mut s);
            let d = if i == 0 { 0.0 } else { 1.0 };
            let probe = [0.2, -0.4];
            let a = alone.step(&x, &[Some(y)], d, 1.0).pred[0];
            let b = pair
                .step(&x, &[Some(y), (i >= 150).then_some(y)], d, 1.0)
                .pred[0];
            if noised.is_none_or(|r| i <= r + 1) {
                assert_eq!(a.to_bits(), b.to_bits(), "row {i}: {a} against {b}");
            } else if a != b {
                parted = true;
            }
            // `pred_var` reads the state after the row: equal while the
            // copy has no residual variance in it.
            if pair.sigma2()[1] == 0.0 {
                let (va, vb) = (alone.pred_var(&probe)[0], pair.pred_var(&probe)[0]);
                assert_eq!(
                    va.to_bits(),
                    vb.to_bits(),
                    "row {i}: pred_var {va} against {vb}"
                );
            }
            if noised.is_none() && pair.sigma2()[1] > 0.0 {
                noised = Some(i);
            }
        }
        assert_eq!(noised, Some(151), "the copy's first residual");
        assert!(parted, "the copy's own variance enters the mean");
    }
}
