//! Robust regression: Huber and quantile (docs/PLAN.md §4.5).
//!
//! IRLS-style reweighting on the EW-ridge update: each row's weight is scaled by
//! the robust weight of its *prior* residual, so the reweighting is still
//! out-of-sample (the residual comes from the prediction made before the update).
//!
//! Huber, with `d = huber_delta` in units of the EW residual std `s_j`:
//!
//! ```text
//! w_robust = 1                 if |r| <= d * s
//!          = d * s / |r|       otherwise
//! ```
//!
//! Quantile (check loss at level tau) does not reweight: it takes one Newton
//! step on the check loss smoothed by a uniform kernel of half-width
//! `h = quantile_eps * s`, linearised at the fit the row was scored with. The
//! smoothed loss has curvature `1/(2h)` inside the band and none outside, so
//! with `psi(r) = tau - 1{r < 0}` the row is:
//!
//! ```text
//! |r| <  h:  a least-squares row, target y + 2h(tau - 1/2)
//! |r| >= h:  no weight in the Gram; 2h * psi(r) * z into the cross-moment
//! ```
//!
//! Both arms come from one identity: the Newton system's row is
//! `z z' beta = z z' beta_prior + 2h * psi(r) * z`, and inside the band
//! `2h * psi(r) = 2h(tau - 1/2) + r`, which folds the prior fit back out. The
//! IRLS weight this replaced, `psi(r)/r`, is that step's *secant* where this is
//! its tangent, and it is unbounded as `r -> 0`: a row whose prior residual
//! happened to be near zero kept a weight of up to `1/quantile_eps` for ever,
//! and the fit settled a fraction `1/(1 + ln(1/eps))` of the way from the mean
//! of its own past fits to the quantile regression -- 0.164 short of
//! `statsmodels`' `QuantReg` at the median of a skewed noise after 20 000 rows,
//! and 0.477 at the 0.9 quantile (review 2026-09-12, N9).
//!
//! Under three rows per coefficient of the rows the target was present on,
//! the quantile fit warms up as ordinary least squares: a Newton step needs a
//! Hessian, and a band around a fit built from a handful of rows is not one.
//! And the band is never narrower than `(k/n)^(2/5)` of `s` for the target's
//! effective sample `n` -- the smoothed-quantile bandwidth rate, which a long
//! stream leaves behind and which keeps the Hessian fed under a short
//! halflife, where the band's share of the sample is a few rows. The warm-up
//! read the band's weight until the second review of 2026-09-15 (F3), which a
//! halflife caps at that share, so a tail quantile at `halflife = 30` kept
//! falling back into warm-up and covered 0.825 where 0.9 was asked. What
//! that warm-up also did was rebuild a fit a row at the input bound had
//! left behind. Such a row sets the Gram and the cross-moment at its own
//! scale, and a mean-form accumulator forgets only through rows entering
//! it; every later row is outside the band (its residual is at the bound's
//! scale, where the band is a fraction of it), so only nudges arrive, each
//! `2h * psi * z` over the band's weight -- nothing beside the bound's
//! moments until that weight has decayed to nothing, and a step that
//! outgrows the band once it has. The fit oscillates, the band's weight
//! underflows after 1400 halflives, and the prediction is withheld (the
//! bounded-extremes contract). So a band holding under one row per
//! coefficient takes least-squares rows until it holds rows again: from
//! one row up an outside row's step, `2h * |psi| / wj`, lands inside the
//! band, and a band the floor keeps at a share of the sample is never near
//! one row in steady state, so the warm-up's bias does not return.
//!
//! `s` is the EW residual std of that target as the row arrives, and is taken
//! as 1 until one exists (no rows yet, or every residual so far exactly zero),
//! so the first rows are weighted in the residual's own units.
//!
//! **`s` is not itself robust.** It is the plain EW mean of squared
//! residuals, in which the rows Huber down-weights count at full weight, so
//! an outlier inflates the scale the next rows' cuts are drawn in, and
//! `huber_delta` is in units of that std, not of a robust one (a MAD, or a
//! Huber-weighted variance). A burst of outliers widens the cut for the rows
//! after it until the EW mean forgets them (review 2026-09-12, D4).
//!
//! Because the weights are per target, the `S` accumulator is per target here
//! (one [`EwCov`] each) — unlike [`crate::EwRidge`], which shares one.

use serde::{Deserialize, Serialize};

use crate::model::{ModelState, OnlineModel, State, StateError, Step, check_schema};
use crate::solve::{dot_aug, solve_spd};
use crate::{Decay, EwCov};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RobustLoss {
    /// Huber with `delta` in units of the EW residual std.
    Huber { delta: f64 },
    /// Quantile regression at level `tau`.
    Quantile { tau: f64 },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RobustCfg {
    pub n_features: usize,
    pub n_targets: usize,
    pub add_intercept: bool,
    pub decay: Decay,
    pub loss: RobustLoss,
    pub ridge: f64,
    pub standardize: bool,
    pub min_periods: f64,
    pub solve_every: f64,
    pub max_rows_between_solves: u32,
    /// Half-width of the band a quantile fit takes its Newton step in, in
    /// units of the EW residual std: rows inside carry the curvature, rows
    /// outside only the score (the module docs). It was the floor under `|r|`
    /// in an IRLS weight, bounding a weight rather than naming a band (review
    /// 2026-09-12, N9).
    pub quantile_eps: f64,
}

/// What one row does to a target's accumulators ([`Robust::row_update`]).
#[derive(Debug, Clone, Copy, PartialEq)]
enum RowUpdate {
    /// A weighted least-squares row: `w` into the Gram, `target` into the
    /// cross-moment.
    Fit { w: f64, target: f64 },
    /// A row outside the quantile band: the Gram takes nothing, the
    /// cross-moment takes `nudge * z` (review 2026-09-12, N9).
    Nudge { nudge: f64 },
}

/// Rows per coefficient a quantile fit accumulates as ordinary least squares
/// before it starts taking Newton steps ([`Robust::row_update`]). Measured
/// (review 2026-09-12, N9): at one row per coefficient the Gram is near
/// singular and the ridge turns it into coefficients in the hundreds; three
/// was stable at every bandwidth and feature count tried, and the fit it
/// leaves is `QuantReg`'s to within one of its standard errors.
const WARM_ROWS: f64 = 3.0;

impl RobustCfg {
    pub fn k_total(&self) -> usize {
        self.n_features + usize::from(self.add_intercept)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.n_features == 0 || self.n_targets == 0 {
            return Err("n_features and n_targets must be >= 1".into());
        }
        match self.loss {
            RobustLoss::Huber { delta } => {
                if delta <= 0.0 || delta.is_nan() {
                    return Err("huber_delta must be > 0".into());
                }
            }
            RobustLoss::Quantile { tau } => {
                if !(0.0..=1.0).contains(&tau) || tau == 0.0 || tau == 1.0 {
                    return Err("quantile must be in (0, 1)".into());
                }
            }
        }
        if self.ridge < 0.0 {
            return Err("ridge must be >= 0".into());
        }
        if self.quantile_eps <= 0.0 {
            return Err("quantile_eps must be > 0".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Robust {
    cfg: RobustCfg,
    /// One accumulator per target (the robust weights are per target).
    cov: Vec<EwCov>,
    /// Per target, the weight its accumulators hold: Huber's reweighted rows,
    /// or the quantile band's. The mean-form cross-moment `cross` is over it,
    /// and so is `cov`, whose weight it equals.
    wj: Vec<f64>,
    /// Per target, the rows it was present on at their raw weights, decayed:
    /// what the per-target `min_periods` gate reads (hard rule 8, S2) and the
    /// quantile fit's warm-up counts. `wj` stood in for it, and for the
    /// quantile that is the band's weight, which a halflife caps at the
    /// band's share of the sample (the second review of 2026-09-15, F1).
    wobs: Vec<f64>,
    /// Per target, the centred cross-moment `c_j = E[(z − m_j)(y − ȳ_j)]`
    /// over `wj`, with `m_j` its accumulator's mean, and `ȳ_j` beside it
    /// (`ybar`). With an intercept slot 0 is exactly 0: `z_0 − m_0 = 0` once
    /// a row has entered. They were kept raw, `E[z·y]`, and the solves read
    /// the raw normal equations or centred them by subtraction, which loses
    /// `level²·ε` -- the whole fit at `1e8` -- where `ew_ridge` and `lasso`
    /// were moved to centred moments in the 2026-09-12 round and this model
    /// was not (the review of 2026-09-18, S2). A quantile nudge, a sum's worth
    /// of `2h·psi·z` over `wj`, enters as `ȳ += nudge/wj` and `c += nudge·(z
    /// − m)/wj`: the raw step `E[z·y] += nudge·z/wj` transformed exactly.
    cross: Vec<Vec<f64>>,
    ybar: Vec<f64>,
    /// EW residual variance per target (drives the robust scale).
    sig2: Vec<f64>,
    wsig: Vec<f64>,
    /// EW count of *observations* using the raw row weights, i.e. ignoring
    /// what the loss does with them. This is what `n_eff` and `min_periods`
    /// mean everywhere else, so the robust models report it too: Huber scales
    /// the accumulators by its weights and a quantile fit weighs only the rows
    /// inside its band, and the observation count must follow neither. (The
    /// IRLS weights a quantile fit once used reached `2 / quantile_eps`, so
    /// counting them inflated `n_eff` by ~1000x -- T-A5.)
    w_raw: f64,
    beta: Option<Vec<Vec<f64>>>,
    clock_since_solve: f64,
    rows_since_solve: u32,
    pub solve_failures: u64,
    #[serde(skip)]
    zbuf: Vec<f64>,
}

impl Robust {
    pub fn new(cfg: RobustCfg) -> Result<Self, String> {
        cfg.validate()?;
        let k = cfg.k_total();
        let m = cfg.n_targets;
        Ok(Self {
            cov: vec![EwCov::new(k); m],
            wj: vec![0.0; m],
            wobs: vec![0.0; m],
            cross: vec![vec![0.0; k]; m],
            ybar: vec![0.0; m],
            sig2: vec![0.0; m],
            wsig: vec![0.0; m],
            w_raw: 0.0,
            beta: None,
            clock_since_solve: 0.0,
            rows_since_solve: 0,
            solve_failures: 0,
            zbuf: vec![0.0; k],
            cfg,
        })
    }

    pub fn cfg(&self) -> &RobustCfg {
        &self.cfg
    }

    pub fn sigma2(&self) -> &[f64] {
        &self.sig2
    }

    /// EW count of observations under the raw row weights (`w_raw`).
    pub fn n_eff(&self) -> f64 {
        self.w_raw
    }

    pub fn coefficients(&self) -> Option<&[Vec<f64>]> {
        self.beta.as_deref()
    }

    /// What a row does to one target's accumulators (the module docs).
    ///
    /// Huber reweights it: an ordinary least-squares row at `min(1, delta*s/|r|)`,
    /// a weight bounded by 1. The quantile loss linearises it instead, inside
    /// the band or outside it, and the two arms are [`RowUpdate`]'s.
    ///
    /// `pred` is the prediction the row was scored with, so both stay
    /// out-of-sample, and `scale` is the EW residual std, taken as 1 until one
    /// exists. `present` is the weight of the rows this target was present
    /// on, decayed to the row (`wobs`): under `WARM_ROWS` of it per
    /// coefficient the quantile fit takes ordinary least-squares rows, since a
    /// Newton step needs a Hessian to lean on and a band around a fit built
    /// from a handful of rows is not one. That is the warm-up, and it is what
    /// rebuilds the fit after a gap or a reset has aged the weight away. Past
    /// it the band is at least `(k/present)^(2/5)` of `scale` wide, and
    /// `aged`, the band's own weight decayed to the row, under one row per
    /// coefficient is a fit the data has left behind, which takes
    /// least-squares rows until the band holds rows again (the module docs).
    fn row_update(
        &self,
        yj: f64,
        pred: f64,
        scale: f64,
        weight: f64,
        present: f64,
        aged: f64,
    ) -> RowUpdate {
        match self.cfg.loss {
            RobustLoss::Huber { delta } => {
                let w_rob = if pred.is_finite() {
                    let cut = delta * scale;
                    let a = (yj - pred).abs();
                    if a <= cut || a == 0.0 { 1.0 } else { cut / a }
                } else {
                    1.0
                };
                RowUpdate::Fit {
                    w: weight * w_rob,
                    target: yj,
                }
            }
            RobustLoss::Quantile { tau } => {
                let k = self.cfg.k_total() as f64;
                if !pred.is_finite() || present < WARM_ROWS * k || aged < k {
                    return RowUpdate::Fit {
                        w: weight,
                        target: yj,
                    };
                }
                let floor = (k / present).powf(0.4);
                let h = scale * self.cfg.quantile_eps.max(floor);
                let r = yj - pred;
                if r.abs() < h {
                    RowUpdate::Fit {
                        w: weight,
                        target: yj + 2.0 * h * (tau - 0.5),
                    }
                } else {
                    let psi = if r > 0.0 { tau } else { tau - 1.0 };
                    RowUpdate::Nudge {
                        nudge: weight * 2.0 * h * psi,
                    }
                }
            }
        }
    }

    fn solve(&mut self) {
        let k = self.cfg.k_total();
        let mut beta = vec![vec![0.0; k]; self.cfg.n_targets];
        for j in 0..self.cfg.n_targets {
            if self.wj[j] <= 0.0 {
                continue;
            }
            // `None` is a solve that failed at every jitter: counted, and the
            // previous fit kept. It returned through `?` before the count
            // (review 2026-09-12, S13).
            let solved = if self.cfg.add_intercept {
                self.solve_centred(k, j)
            } else {
                self.solve_through_origin(k, j)
            };
            match solved {
                Some(sol) => beta[j] = sol,
                None => {
                    self.solve_failures += 1;
                    if let Some(prev) = &self.beta {
                        beta[j] = prev[j].clone();
                    }
                }
            }
        }
        self.beta = Some(beta);
        self.clock_since_solve = 0.0;
        self.rows_since_solve = 0;
    }

    /// The solve with an intercept, plain or standardized, on the centred
    /// system, as [`crate::EwRidge`]'s `solve_centred`: the slopes from the
    /// accumulator's centred co-moments over the features and the target's
    /// centred cross-moment, then `beta_0 = ȳ − m·beta`. Plain, nothing is
    /// scaled and nothing dropped, the estimator the raw normal equations
    /// with an unpenalized intercept define; standardized, the co-moments
    /// are scaled to correlation form and a ~zero-variance feature is
    /// dropped (coefficient 0). Nothing level-sized is subtracted from
    /// anything on the way (S2). `None` when every jitter failed.
    fn solve_centred(&mut self, k: usize, j: usize) -> Option<Vec<f64>> {
        let kf = k - 1;
        let cov = &self.cov[j];
        let mut c = vec![0.0; kf * kf];
        for i in 0..kf {
            for jj in 0..kf {
                c[i * kf + jj] = cov.cov(i + 1, jj + 1);
            }
        }
        let (s, keep): (Vec<f64>, Vec<usize>) = if self.cfg.standardize {
            let s = (0..kf).map(|i| c[i * kf + i].max(0.0).sqrt()).collect();
            let keep = (0..kf)
                .filter(|&i| crate::variance_is_usable(c[i * kf + i], cov.raw(i + 1, i + 1)))
                .collect();
            (s, keep)
        } else {
            (vec![1.0; kf], (0..kf).collect())
        };
        let kk = keep.len();
        let mut out = vec![0.0; k];
        let mut jitter = 0u32;
        if kk > 0 {
            let mut asub = vec![0.0; kk * kk];
            for (i2, &i) in keep.iter().enumerate() {
                for (j2, &jj) in keep.iter().enumerate() {
                    asub[i2 * kk + j2] = c[i * kf + jj] / (s[i] * s[jj]);
                }
                asub[i2 * kk + i2] += self.cfg.ridge;
            }
            let bsub: Vec<f64> = keep.iter().map(|&i| self.cross[j][i + 1] / s[i]).collect();
            let (sol, jit) = solve_spd(&asub, &bsub, kk, 1)?;
            jitter = jit;
            for (i2, &i) in keep.iter().enumerate() {
                out[i + 1] = sol[i2] / s[i];
            }
        }
        let mut b0 = self.ybar[j];
        for i in 0..kf {
            b0 -= cov.mean(i + 1) * out[i + 1];
        }
        out[0] = b0;
        self.solve_failures += u64::from(jitter);
        Some(out)
    }

    /// The solve through the origin, on the raw system: every slot is a
    /// slope and every slot is penalized, and the right-hand side is the
    /// uncentred `E[z·y] = c + m·ȳ`, one step from what is kept. Nothing is
    /// centred through the origin -- a level cannot be absorbed there -- so
    /// the raw form is the fit asked for, as `EwRidge`'s no-intercept branch
    /// has it. Standardized, the system is scaled by the raw second-moment
    /// diagonals (a slot with none is dropped): this centred the Gram and
    /// kept the raw right-hand side once, the hybrid system C8 found in
    /// `lasso`, least squares only when every feature has mean zero (review
    /// 2026-09-12, C11). `None` when every jitter failed.
    fn solve_through_origin(&mut self, k: usize, j: usize) -> Option<Vec<f64>> {
        let cov = &self.cov[j];
        let ybar = self.ybar[j];
        let b: Vec<f64> = (0..k)
            .map(|i| self.cross[j][i] + cov.mean(i) * ybar)
            .collect();
        let s: Vec<f64> = if self.cfg.standardize {
            (0..k).map(|i| cov.raw(i, i).max(0.0).sqrt()).collect()
        } else {
            vec![1.0; k]
        };
        // No centering here, so no cancellation: any strictly positive raw
        // moment is usable.
        let keep: Vec<usize> = (0..k).filter(|&i| s[i] > 0.0).collect();
        let kk = keep.len();
        let mut out = vec![0.0; k];
        let mut jitter = 0u32;
        if kk > 0 {
            let mut asub = vec![0.0; kk * kk];
            for (i2, &i) in keep.iter().enumerate() {
                for (j2, &jj) in keep.iter().enumerate() {
                    asub[i2 * kk + j2] = cov.raw(i, jj) / (s[i] * s[jj]);
                }
                asub[i2 * kk + i2] += self.cfg.ridge;
            }
            let bsub: Vec<f64> = keep.iter().map(|&i| b[i] / s[i]).collect();
            let (sol, jit) = solve_spd(&asub, &bsub, kk, 1)?;
            jitter = jit;
            for (i2, &i) in keep.iter().enumerate() {
                out[i] = sol[i2] / s[i];
            }
        }
        self.solve_failures += u64::from(jitter);
        Some(out)
    }
}

impl OnlineModel for Robust {
    fn target_n_eff_into(&self, out: &mut Vec<f64>) -> bool {
        out.clear();
        out.extend_from_slice(&self.wobs);
        true
    }

    fn step(&mut self, x: &[f64], y: &[Option<f64>], d_clock: f64, weight: f64) -> Step {
        let m = self.cfg.n_targets;
        let k = self.cfg.k_total();
        if self.zbuf.len() != k {
            self.zbuf = vec![0.0; k];
        }
        let lam = self.cfg.decay.factor(d_clock);
        if self.cfg.add_intercept {
            self.zbuf[0] = 1.0;
            self.zbuf[1..].copy_from_slice(x);
        } else {
            self.zbuf.copy_from_slice(x);
        }

        // ---- predict (state before the update) ----
        let out = self.predict(x, d_clock);
        let pred = &out.pred;

        // ---- update: Huber reweights the row, the quantile linearises it ----
        for j in 0..m {
            // `σ²`'s weight ages on every row, as `wj` does, and a row adds to
            // it only with a target, a weight and a prediction to measure the
            // residual from. A zero or NaN weight skipped the ageing, and so
            // did a row with no prediction (`min_periods` unmet after a clock
            // gap), so `σ²` -- the scale of every cut -- forgot less across
            // either than across a null (review 2026-09-12, S13; N6).
            self.wsig[j] *= lam;
            let present = lam * self.wobs[j];
            let Some(yj) = y[j] else {
                self.cov[j].decay(lam);
                self.wj[j] *= lam;
                self.wobs[j] = present;
                continue;
            };
            // A row the target is present on counts at its raw weight, whatever
            // the loss then does with it (hard rule 8).
            self.wobs[j] = present
                + if weight > 0.0 && weight.is_finite() {
                    weight
                } else {
                    0.0
                };
            let sigma = self.sig2[j].max(0.0).sqrt();
            let scale = if sigma > 0.0 { sigma } else { 1.0 };
            let aged = lam * self.wj[j];
            match self.row_update(yj, pred[j], scale, weight, present, aged) {
                // NaN is `inf / inf` from an overflowed residual against an
                // overflowed scale; such a row cannot be learned from either.
                RowUpdate::Fit { w, .. } if w.is_nan() || w <= 0.0 => {
                    self.cov[j].decay(lam);
                    self.wj[j] = aged;
                    continue;
                }
                RowUpdate::Fit { w, target } => {
                    // The same `a`/`b` as `EwCov::update` forms from the same
                    // weights, and the deviations from the means *before* the
                    // row, as it takes them -- so the cross-moment is read
                    // before the accumulator moves.
                    let wj_new = aged + w;
                    let a = aged / wj_new;
                    let bb = w / wj_new;
                    let dy = target - self.ybar[j];
                    let ab_dy = a * bb * dy;
                    let cov = &self.cov[j];
                    for (i, (ci, zi)) in self.cross[j].iter_mut().zip(&self.zbuf).enumerate() {
                        *ci = a * *ci + ab_dy * (zi - cov.mean(i));
                    }
                    self.ybar[j] += bb * dy;
                    self.cov[j].update(&self.zbuf, lam, w);
                    self.wj[j] = wj_new;
                }
                RowUpdate::Nudge { nudge } => {
                    // Outside the band a row is one term of the score and none
                    // of the Hessian: no weight in the Gram, and `2h*psi(r)*z`
                    // into the cross-moment, which is a mean over `wj` -- so a
                    // sum's worth of nudge enters divided by it (N9). Centred,
                    // that is `ȳ += nudge/wj` and `c += nudge·(z − m)/wj`.
                    self.cov[j].decay(lam);
                    self.wj[j] = aged;
                    if nudge.is_finite() && aged > 0.0 {
                        let step = nudge / aged;
                        let cov = &self.cov[j];
                        for (i, (ci, zi)) in self.cross[j].iter_mut().zip(&self.zbuf).enumerate() {
                            *ci += step * (zi - cov.mean(i));
                        }
                        self.ybar[j] += step;
                    }
                }
            }
            if pred[j].is_finite() {
                let resid = yj - pred[j];
                let ws_new = self.wsig[j] + weight;
                let s2 = (self.wsig[j] * self.sig2[j] + weight * resid * resid) / ws_new;
                // Skipped when it would not be finite: an `inf` scale makes
                // the Huber cut and the quantile band infinite, and every row
                // after it a plain least-squares one for good
                // (docs/IMPROVEMENTS.md C2).
                if s2.is_finite() {
                    self.sig2[j] = s2;
                    self.wsig[j] = ws_new;
                }
            }
        }

        self.w_raw = lam * self.w_raw + weight;

        self.clock_since_solve += d_clock;
        self.rows_since_solve += 1;
        let due = self.cfg.solve_every <= 0.0
            || self.clock_since_solve >= self.cfg.solve_every
            || self.rows_since_solve >= self.cfg.max_rows_between_solves
            || (self.beta.is_none() && self.w_raw >= self.cfg.min_periods);
        if due {
            self.solve();
        }
        out
    }

    fn predict(&self, x: &[f64], _d_clock: f64) -> Step {
        let n_eff = self.w_raw;
        let mut pred = vec![f64::NAN; self.cfg.n_targets];
        if let (true, Some(beta)) = (n_eff >= self.cfg.min_periods, &self.beta) {
            for (j, p) in pred.iter_mut().enumerate() {
                if self.wj[j] > 0.0 {
                    *p = dot_aug(&beta[j], x, self.cfg.add_intercept);
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
        State::new(ModelState::Robust(Box::new(self.clone())))
    }

    fn restore(s: &State) -> Result<Self, StateError> {
        check_schema(s)?;
        match &s.model {
            ModelState::Robust(m) => {
                let mut m = (**m).clone();
                m.zbuf = vec![0.0; m.cfg.k_total()];
                Ok(m)
            }
            other => Err(StateError::WrongModel {
                expected: "robust",
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
    use crate::{EwRidge, EwRidgeCfg};

    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }

    fn cfg(k: usize, m: usize, loss: RobustLoss) -> RobustCfg {
        RobustCfg {
            n_features: k,
            n_targets: m,
            add_intercept: true,
            decay: Decay::Halflife(f64::INFINITY),
            loss,
            ridge: 1e-8,
            standardize: false,
            min_periods: (k + 1) as f64,
            solve_every: 0.0,
            max_rows_between_solves: 1,
            quantile_eps: 1e-3,
        }
    }

    #[test]
    fn cfg_validation_rejects_each_bad_field() {
        let huber = RobustLoss::Huber { delta: 1.5 };
        let bad = |loss: RobustLoss, f: &dyn Fn(&mut RobustCfg), want: &str| {
            let mut c = cfg(2, 1, loss);
            f(&mut c);
            match c.validate() {
                Err(e) => assert!(e.contains(want), "wanted {want:?}, got {e:?}"),
                Ok(()) => panic!("expected rejection mentioning {want:?}"),
            }
        };
        bad(huber, &|c| c.n_features = 0, "must be >= 1");
        bad(huber, &|c| c.n_targets = 0, "must be >= 1");

        // delta is the crossover from squared to absolute loss.
        for d in [0.0, -1.0, f64::NAN] {
            bad(
                huber,
                &|c| c.loss = RobustLoss::Huber { delta: d },
                "huber_delta",
            );
        }
        cfg(2, 1, RobustLoss::Huber { delta: 1e-9 })
            .validate()
            .unwrap();

        // The quantile is an open interval: 0 and 1 are not quantiles a
        // weighted least-squares reformulation can represent.
        for t in [0.0, 1.0, -0.1, 1.1, f64::NAN] {
            bad(
                huber,
                &|c| c.loss = RobustLoss::Quantile { tau: t },
                "quantile must be in",
            );
        }
        for t in [1e-6, 0.5, 1.0 - 1e-6] {
            cfg(2, 1, RobustLoss::Quantile { tau: t })
                .validate()
                .unwrap();
        }

        // ridge may be zero; quantile_eps may not (it divides).
        bad(huber, &|c| c.ridge = -1e-9, "ridge must be >= 0");
        let mut ok = cfg(2, 1, huber);
        ok.ridge = 0.0;
        ok.validate().unwrap();
        bad(huber, &|c| c.quantile_eps = 0.0, "quantile_eps must be > 0");
        bad(
            huber,
            &|c| c.quantile_eps = -1.0,
            "quantile_eps must be > 0",
        );
    }

    #[test]
    fn n_eff_counts_observations_not_irls_weights() {
        // The defect T-A5 found: the IRLS weights a quantile fit then used
        // reached `2 / quantile_eps`, so counting them made `n_eff` -- and
        // therefore `min_periods` -- meaningless. It must be the plain
        // weighted observation count, identical to every other model's, and it
        // still is now that the fit weighs the rows in its band (N9).
        for loss in [
            RobustLoss::Huber { delta: 1.5 },
            RobustLoss::Quantile { tau: 0.5 },
            RobustLoss::Quantile { tau: 0.9 },
        ] {
            let mut c = cfg(1, 1, loss);
            c.decay = Decay::Halflife(20.0);
            c.min_periods = 0.0;
            let mut m = Robust::new(c).unwrap();
            let mut want = 0.0;
            for i in 0..30 {
                let d = if i == 0 { 0.0 } else { 1.0 };
                let step = m.step(&[i as f64], &[Some(3.0 * i as f64)], d, 1.0);
                assert!(
                    (step.n_eff - want).abs() < 1e-12,
                    "{loss:?} row {i}: {} vs {want}",
                    step.n_eff
                );
                want = want * 0.5f64.powf(d / 20.0) + 1.0;
            }
            assert!(
                want < 31.0,
                "an observation count cannot exceed the row count"
            );
        }
    }

    #[test]
    fn a_solve_failure_is_counted_and_the_previous_fit_is_kept() {
        // Two perfectly collinear features with no ridge: the normal equations
        // are singular. The model must keep its last good coefficients rather
        // than emit NaN, and say so in `solve_failures`.
        let mut c = cfg(2, 1, RobustLoss::Huber { delta: 1.5 });
        c.ridge = 0.0;
        c.add_intercept = true;
        c.min_periods = 2.0;
        let mut m = Robust::new(c).unwrap();
        let mut s = 101u64;
        for i in 0..40 {
            let a = lcg(&mut s);
            // x1 == x0, and the intercept is constant: rank deficient by two.
            m.step(
                &[a, a],
                &[Some(2.0 * a + 1.0)],
                if i == 0 { 0.0 } else { 1.0 },
                1.0,
            );
        }
        let beta = m.coefficients().unwrap()[0].clone();
        assert!(
            beta.iter().all(|v| v.is_finite()),
            "never NaN, even singular: {beta:?}"
        );
        // Either the jitter rescued it (counted) or the solve failed (counted);
        // silently succeeding on a singular system is the outcome to rule out.
        assert!(m.solve_failures > 0, "a singular solve must be recorded");
    }

    /// `a_solve_failure_is_counted_and_the_previous_fit_is_kept` on the
    /// standardized path, as the review wrote it (2026-09-12, S13). The two
    /// collinear features give a singular correlation matrix, which the
    /// jitter ladder rescues and counts; a solve that fails outright is the
    /// next test's.
    #[test]
    fn a_standardized_solve_failure_is_counted_and_the_previous_fit_is_kept() {
        let mut c = cfg(2, 1, RobustLoss::Huber { delta: 1.5 });
        c.ridge = 0.0;
        c.standardize = true;
        c.min_periods = 2.0;
        let mut m = Robust::new(c).unwrap();
        let mut s = 101u64;
        for i in 0..40 {
            let a = lcg(&mut s);
            m.step(
                &[a, a],
                &[Some(2.0 * a + 1.0)],
                if i == 0 { 0.0 } else { 1.0 },
                1.0,
            );
        }
        let beta = m.coefficients().unwrap()[0].clone();
        assert!(
            beta.iter().all(|v| v.is_finite()),
            "never NaN, even singular: {beta:?}"
        );
        assert!(m.solve_failures > 0, "a singular solve must be recorded");
    }

    /// A standardized solve that fails at every jitter keeps the previous
    /// fit, as the plain one does, and is counted as the plain one is: it
    /// returned through `?` before the count (review 2026-09-12, S13). No
    /// stream of rows gives a correlation matrix that every jitter fails on,
    /// so the accumulator is handed one: a correlation of 2.
    #[test]
    fn a_standardized_solve_that_fails_outright_is_counted() {
        let mut c = cfg(2, 1, RobustLoss::Huber { delta: 1.5 });
        c.standardize = true;
        let mut m = Robust::new(c).unwrap();
        let mut s = 103u64;
        for i in 0..40 {
            let x = [lcg(&mut s), lcg(&mut s)];
            m.step(
                &x,
                &[Some(x[0] - x[1])],
                if i == 0 { 0.0 } else { 1.0 },
                1.0,
            );
        }
        let beta = m.coefficients().unwrap()[0].clone();
        let before = m.solve_failures;
        let (w, q) = (m.cov[0].n_eff(), m.cov[0].q_sum());
        m.cov[0].set_moments(
            &[1.0, 0.0, 0.0],
            &[0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 2.0, 1.0],
            w,
            q,
        );
        m.solve();
        assert_eq!(
            m.solve_failures,
            before + 1,
            "every jitter failed: one failure"
        );
        assert_eq!(
            m.coefficients().unwrap()[0],
            beta,
            "and the previous fit is kept"
        );
    }

    /// `null_targets_decay_the_residual_variance_weight_without_adding_to_it`
    /// (ewridge.rs) with the target present at weight zero: a row that
    /// teaches nothing leaves `σ²` where it was and ages its weight, as a
    /// null row does. The zero weight skipped the ageing, so `σ²` -- the
    /// scale of every Huber cut -- forgot less across such a row than across
    /// a null (review 2026-09-12, S13; S9 was the same in `kalman`).
    #[test]
    fn a_zero_weight_row_ages_the_residual_variance_as_a_null_does() {
        let mut c = cfg(1, 1, RobustLoss::Huber { delta: 1.5 });
        c.decay = Decay::Halflife(10.0);
        c.min_periods = 2.0;
        let mut m = Robust::new(c).unwrap();
        let mut s = 53u64;
        for i in 0..60 {
            let x = [lcg(&mut s)];
            let y = 2.0 * x[0] + 0.1 + 0.05 * lcg(&mut s);
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let (sig, w) = (m.sig2[0], m.wsig[0]);
        assert!(sig > 0.0 && w > 0.0);
        let lam = 0.5f64.powf(3.0 / 10.0);
        m.step(&[0.5], &[Some(-500.0)], 3.0, 0.0);
        assert_eq!(m.sig2[0], sig, "weight 0 must not move sigma2");
        assert!(
            (m.wsig[0] - w * lam).abs() < 1e-12,
            "but its weight ages: {} vs {}",
            m.wsig[0],
            w * lam
        );
    }

    /// `σ²` is the EW mean of the squared out-of-sample errors: every row
    /// ages its weight, and a row with a target, a weight and a prediction
    /// adds `w·r²` (the row weight, not the robust one). Held on a stream
    /// with zero-weight rows (S13) and a clock gap that takes `n_eff` under
    /// `min_periods`, whose next rows have a target and no prediction and
    /// aged nothing either (N6).
    #[test]
    fn the_residual_variance_ages_on_every_row() {
        let hl = 10.0;
        let mut c = cfg(1, 1, RobustLoss::Huber { delta: 1.5 });
        c.decay = Decay::Halflife(hl);
        c.min_periods = 3.0;
        let mut m = Robust::new(c).unwrap();
        let (mut want, mut wsig, mut unpredicted) = (0.0f64, 0.0f64, 0);
        let mut s = 61u64;
        for i in 0..160 {
            let x = [lcg(&mut s)];
            let y = 2.0 * x[0] + 0.1 + 0.2 * lcg(&mut s);
            let d = match i {
                0 => 0.0,
                80 => 200.0,
                _ => 1.0,
            };
            let (y, w) = match i % 9 {
                4 => (None, 1.0),
                7 => (Some(y), 0.0),
                _ => (Some(y), 1.0),
            };
            let p = m.step(&x, &[y], d, w).pred[0];
            wsig *= 0.5f64.powf(d / hl);
            match y {
                Some(y) if w > 0.0 && p.is_finite() => {
                    let r = y - p;
                    let ws_new = wsig + w;
                    want = (wsig * want + w * r * r) / ws_new;
                    wsig = ws_new;
                }
                Some(_) if w > 0.0 && wsig > 0.0 => unpredicted += 1,
                _ => {}
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

    #[test]
    fn huber_resists_outliers_that_break_least_squares() {
        let mut hub = Robust::new(cfg(1, 1, RobustLoss::Huber { delta: 1.5 })).unwrap();
        let mut ols = EwRidge::new(EwRidgeCfg {
            n_features: 1,
            n_targets: 1,
            add_intercept: true,
            decay: Decay::Halflife(f64::INFINITY),
            ridge: vec![1e-8],
            feature_sets: vec![],
            standardize: false,
            ridge_decay: false,
            session_shrink: None,
            long_halflife: None,
            coef_prior: None,
            min_periods: 2.0,
            solve_every: 0.0,
            max_rows_between_solves: 1,
            gram_block_rows: 0,
            target_gaps: crate::TargetGaps::OwnRows,
            window: None,
            window_every: None,
        })
        .unwrap();
        let mut s = 77u64;
        for i in 0..600 {
            let x = [lcg(&mut s)];
            // clean relationship y = 2x, with 3% enormous outliers
            let outlier = i % 33 == 7;
            let y = if outlier {
                500.0 * lcg(&mut s)
            } else {
                2.0 * x[0]
            };
            let d = if i == 0 { 0.0 } else { 1.0 };
            hub.step(&x, &[Some(y)], d, 1.0);
            ols.step(&x, &[Some(y)], d, 1.0);
        }
        let h = hub.coefficients().unwrap()[0][1];
        let o = ols.coefficients().unwrap()[0][1];
        assert!(
            (h - 2.0).abs() < (o - 2.0).abs(),
            "huber {h} should beat ols {o} (truth 2.0)"
        );
        assert!((h - 2.0).abs() < 0.5, "huber slope {h}");
    }

    #[test]
    fn huber_matches_least_squares_without_outliers() {
        // With a huge delta nothing is downweighted, so it must reduce to OLS.
        let mut m = Robust::new(cfg(2, 1, RobustLoss::Huber { delta: 1e9 })).unwrap();
        let mut s = 78u64;
        for i in 0..400 {
            let x = [lcg(&mut s), lcg(&mut s)];
            let y = 1.5 * x[0] - 0.5 * x[1] + 0.25;
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let b = &m.coefficients().unwrap()[0];
        assert!((b[0] - 0.25).abs() < 1e-6);
        assert!((b[1] - 1.5).abs() < 1e-6);
        assert!((b[2] + 0.5).abs() < 1e-6);
    }

    #[test]
    fn quantile_tracks_the_requested_quantile() {
        // y = 1 + noise with an asymmetric spread; tau = 0.9 must sit clearly
        // above tau = 0.5, which must sit above tau = 0.1.
        let mut lo = Robust::new(cfg(1, 1, RobustLoss::Quantile { tau: 0.1 })).unwrap();
        let mut mid = Robust::new(cfg(1, 1, RobustLoss::Quantile { tau: 0.5 })).unwrap();
        let mut hi = Robust::new(cfg(1, 1, RobustLoss::Quantile { tau: 0.9 })).unwrap();
        let mut s = 79u64;
        for i in 0..4000 {
            let x = [lcg(&mut s)];
            let y = 1.0 + 2.0 * lcg(&mut s); // uniform(-1,3) around the level
            let d = if i == 0 { 0.0 } else { 1.0 };
            lo.step(&x, &[Some(y)], d, 1.0);
            mid.step(&x, &[Some(y)], d, 1.0);
            hi.step(&x, &[Some(y)], d, 1.0);
        }
        let (a, b, c) = (
            lo.coefficients().unwrap()[0][0],
            mid.coefficients().unwrap()[0][0],
            hi.coefficients().unwrap()[0][0],
        );
        assert!(a < b && b < c, "quantile levels out of order: {a} {b} {c}");
    }

    /// The weight the per-target gate reads is the rows the target was
    /// present on, decayed -- hard rule 8's -- whatever the loss does with
    /// them. From N9 to the second review's F1 the quantile fit reported its
    /// band's weight, which a halflife caps at the band's share of the
    /// effective sample, so a `min_periods` above that share closed the gate
    /// for good.
    #[test]
    fn the_target_weight_is_the_rows_present_whatever_the_band_holds() {
        let hl = 30.0;
        let mut c = cfg(1, 1, RobustLoss::Quantile { tau: 0.9 });
        c.decay = Decay::Halflife(hl);
        c.min_periods = 0.0;
        let mut m = Robust::new(c).unwrap();
        let (mut want, mut s, mut out) = (0.0f64, 93u64, Vec::new());
        for i in 0..400 {
            let x = [lcg(&mut s)];
            let (y, w) = match i % 9 {
                4 => (None, 1.0),
                7 => (Some(x[0] + lcg(&mut s)), 0.0),
                _ => (Some(x[0] + lcg(&mut s)), 1.0),
            };
            let d = if i == 0 { 0.0 } else { 1.0 };
            m.target_n_eff_into(&mut out);
            assert!(
                (out[0] - want).abs() <= 1e-12 * want.max(1.0),
                "row {i}: {} reported, {want} present",
                out[0]
            );
            m.step(&x, &[y], d, w);
            want *= 0.5f64.powf(d / hl);
            if y.is_some() && w > 0.0 {
                want += w;
            }
        }
        assert!(want > 30.0, "the stream must have settled: {want}");
    }

    /// The bounded-extremes contract's case (batch 7): a row at the input
    /// bound sets the Gram and the cross-moment at its own scale, every later
    /// row is outside the band, and the nudges, `2h * psi * z` over the
    /// band's weight, cannot move the fit until that weight has decayed to
    /// nothing -- where each is a step that outgrows the band. The warm-up
    /// rebuilt such a fit while it read the band's weight; since F3 moved it
    /// to the rows present, a band holding under one row per coefficient
    /// takes least-squares rows until it holds rows again.
    #[test]
    fn a_band_under_a_row_per_coefficient_takes_least_squares_rows() {
        let hl = 20.0;
        let lam = 0.5f64.powf(1.0 / hl);
        let mut c = cfg(1, 1, RobustLoss::Quantile { tau: 0.5 });
        c.decay = Decay::Halflife(hl);
        c.min_periods = 0.0;
        let mut m = Robust::new(c).unwrap();
        let mut s = 11u64;
        for i in 0..300 {
            let x = [lcg(&mut s)];
            let y = Some(x[0] + 0.1 * lcg(&mut s));
            m.step(&x, &[y], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let mut out = Vec::new();
        m.target_n_eff_into(&mut out);
        assert!(
            out[0] > 6.0 && m.wj[0] > 2.0,
            "settled: {} present, {} in the band",
            out[0],
            m.wj[0]
        );

        // Past the warm-up, with the band holding rows: a row far outside
        // it is a nudge, and the band's weight only decays.
        let x = [0.3];
        let wj = m.wj[0];
        m.step(&x, &[Some(1e6)], 1.0, 1.0);
        assert!(
            (m.wj[0] - lam * wj).abs() <= 1e-12 * wj,
            "a nudge weighs nothing: {} from {wj}",
            m.wj[0]
        );

        // The band starved to under a row per coefficient (`k = 2`): the same
        // row is a least-squares row, weighed at its weight and aimed at `y`.
        // `wj` is set by hand, so the accumulator's own weight is left where
        // it was; the target mean's share is the one `wj` gives.
        m.wj[0] = 1.5;
        let y0 = m.ybar[0];
        m.step(&x, &[Some(1e6)], 1.0, 1.0);
        let aged = lam * 1.5;
        assert!(
            (m.wj[0] - (aged + 1.0)).abs() <= 1e-12,
            "the row's weight enters: {} for {}",
            m.wj[0],
            aged + 1.0
        );
        let want = y0 + 1.0 / (aged + 1.0) * (1e6 - y0);
        assert!(
            (m.ybar[0] - want).abs() <= 1e-9 * want.abs(),
            "the target mean took the row at its target: {} for {want}",
            m.ybar[0]
        );
    }

    /// A level costs the fit nothing (the review of 2026-09-18, S2), as
    /// `ew_ridge`'s test of the same name says of it: Huber at a `delta` that
    /// makes it least squares, on the same stream at the origin and shifted
    /// by `1e8` -- features and target alike, a price regressed on prices --
    /// gives the same slopes and predictions that differ by the shift, plain
    /// and standardized. The cross-moments were raw, and both solves lost
    /// `L²·ε`, which at `1e8` was the whole fit. The tolerances are the
    /// data's own resolution at `1e8`, `ulp ≈ 1.5e-8`.
    #[test]
    fn a_level_costs_the_fit_nothing() {
        for standardize in [false, true] {
            let run = |level: f64| {
                let mut c = cfg(2, 1, RobustLoss::Huber { delta: 1e9 });
                c.standardize = standardize;
                c.ridge = 1e-6;
                c.decay = Decay::Halflife(200.0);
                c.min_periods = 10.0;
                let mut m = Robust::new(c).unwrap();
                let mut s = 29u64;
                let mut preds = Vec::new();
                for i in 0..600 {
                    let u = [lcg(&mut s), lcg(&mut s)];
                    let x = [level + u[0], level + u[1]];
                    let y = level + 2.0 * u[0] - u[1] + 0.1 * lcg(&mut s);
                    let d = if i == 0 { 0.0 } else { 1.0 };
                    preds.push(m.step(&x, &[Some(y)], d, 1.0).pred[0] - level);
                }
                (preds, m.coefficients().unwrap()[0].clone())
            };
            let ((p0, b0), (p8, b8)) = (run(0.0), run(1e8));
            for i in 1..3 {
                assert!(
                    (b0[i] - b8[i]).abs() < 1e-6,
                    "standardize {standardize}: slope {i}: {} vs {}",
                    b0[i],
                    b8[i]
                );
            }
            let mut worst = 0.0f64;
            for (t, (a, b)) in p0.iter().zip(&p8).enumerate() {
                assert_eq!(a.is_finite(), b.is_finite(), "row {t}: {a} vs {b}");
                if a.is_finite() {
                    worst = worst.max((a - b).abs());
                }
            }
            assert!(
                worst < 1e-5,
                "standardize {standardize}: the predictions part by {worst}"
            );
        }
    }

    /// The same for the quantile loss, whose nudge enters the same centred
    /// cross-moment: `ȳ += nudge/wj`, `c += nudge·(z − m)/wj` (S2).
    #[test]
    fn a_level_costs_the_quantile_fit_nothing() {
        let run = |level: f64| {
            let mut c = cfg(1, 1, RobustLoss::Quantile { tau: 0.75 });
            c.decay = Decay::Halflife(500.0);
            c.min_periods = 20.0;
            let mut m = Robust::new(c).unwrap();
            let mut s = 37u64;
            let mut preds = Vec::new();
            for i in 0..1500 {
                let u = lcg(&mut s);
                let y = level + 2.0 * u + lcg(&mut s);
                let d = if i == 0 { 0.0 } else { 1.0 };
                preds.push(m.step(&[level + u], &[Some(y)], d, 1.0).pred[0] - level);
            }
            (preds, m.coefficients().unwrap()[0].clone())
        };
        let ((p0, b0), (p8, b8)) = (run(0.0), run(1e8));
        assert!(
            (b0[1] - 2.0).abs() < 0.2,
            "the fixture is not what it claims: {b0:?}"
        );
        assert!(
            (b0[1] - b8[1]).abs() < 1e-6,
            "slope: {} vs {}",
            b0[1],
            b8[1]
        );
        let mut worst = 0.0f64;
        for (a, b) in p0.iter().zip(&p8) {
            assert_eq!(a.is_finite(), b.is_finite());
            if a.is_finite() {
                worst = worst.max((a - b).abs());
            }
        }
        assert!(worst < 1e-5, "the predictions part by {worst}");
    }

    /// With an intercept the cross-moment's slot 0 is exactly 0: `z_0 = 1`
    /// and `m_0 = 1` once a row has entered, so nothing level-sized ever sits
    /// in the right-hand side (S2).
    #[test]
    fn the_intercept_slot_of_the_cross_moment_is_exactly_zero() {
        let mut m = Robust::new(cfg(2, 1, RobustLoss::Huber { delta: 1.5 })).unwrap();
        let mut s = 41u64;
        for i in 0..50 {
            let x = [1e8 + lcg(&mut s), 1e8 + lcg(&mut s)];
            let y = 1e8 + x[0] - x[1];
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
            assert_eq!(m.cross[0][0], 0.0, "row {i}");
        }
    }

    /// N9: after the warm-up a row outside the band moves the fit by its
    /// nudge alone -- the Gram's weight does not see it at all.
    #[test]
    fn a_quantile_row_outside_the_band_nudges_and_weighs_nothing() {
        let mut c = cfg(1, 1, RobustLoss::Quantile { tau: 0.9 });
        c.min_periods = 0.0;
        let mut m = Robust::new(c).unwrap();
        let mut s = 90u64;
        for i in 0..200 {
            let x = [lcg(&mut s)];
            let y = 1.0 + x[0] + 0.3 * lcg(&mut s);
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let gram_before = m.cov[0].n_eff();
        let mut present = Vec::new();
        m.target_n_eff_into(&mut present);
        let (p_before, b_before) = (present[0], m.coefficients().unwrap()[0].clone());
        // Far above the fit, where the band is `quantile_eps * sigma` wide.
        m.step(&[0.5], &[Some(500.0)], 1.0, 1.0);
        m.target_n_eff_into(&mut present);
        let b_after = m.coefficients().unwrap()[0].clone();
        assert!(
            (m.cov[0].n_eff() - gram_before).abs() < 1e-12,
            "a row outside the band weighs nothing in the Gram: {gram_before} -> {}",
            m.cov[0].n_eff()
        );
        assert!(
            (present[0] - p_before - 1.0).abs() < 1e-12,
            "and is a row the target was present on: {p_before} -> {}",
            present[0]
        );
        assert!(
            b_after[0] > b_before[0],
            "tau = 0.9 follows a row above it: {b_before:?} -> {b_after:?}"
        );
        assert!(
            b_after[0] - b_before[0] < 1.0,
            "by a nudge, not by the row itself: {b_before:?} -> {b_after:?}"
        );
    }

    /// N9: under the warm-up a quantile fit is ordinary least squares, bit for
    /// bit -- a Newton step needs a Hessian, and a band around a fit built from
    /// a handful of rows is not one.
    #[test]
    fn a_quantile_fit_warms_up_as_least_squares() {
        let mut qc = cfg(2, 1, RobustLoss::Quantile { tau: 0.7 });
        qc.min_periods = 0.0;
        let mut lc = cfg(2, 1, RobustLoss::Huber { delta: 1e9 });
        lc.min_periods = 0.0;
        let (mut q, mut l) = (Robust::new(qc).unwrap(), Robust::new(lc).unwrap());
        let mut s = 91u64;
        // `WARM_ROWS` per coefficient is nine rows here, so the tenth is the
        // first the quantile fit takes a step on.
        for i in 0..12 {
            let x = [lcg(&mut s), lcg(&mut s)];
            let y = 0.5 + x[0] - 0.25 * x[1] + 0.3 * lcg(&mut s);
            let d = if i == 0 { 0.0 } else { 1.0 };
            let qs = q.step(&x, &[Some(y)], d, 1.0);
            let ls = l.step(&x, &[Some(y)], d, 1.0);
            if (1..9).contains(&i) {
                assert_eq!(qs.pred, ls.pred, "row {i} is inside the warm-up");
            }
        }
        assert_ne!(
            q.coefficients().unwrap()[0],
            l.coefficients().unwrap()[0],
            "and it parts from least squares once the band is in force"
        );
    }

    #[test]
    fn reweighting_uses_the_prior_residual_only() {
        // An enormous single observation must not be able to fully absorb
        // itself: the weight comes from the prediction made BEFORE the update.
        let mut m = Robust::new(cfg(1, 1, RobustLoss::Huber { delta: 1.0 })).unwrap();
        let mut s = 80u64;
        for i in 0..200 {
            let x = [lcg(&mut s)];
            m.step(&x, &[Some(2.0 * x[0])], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let before = m.coefficients().unwrap()[0].clone();
        m.step(&[1.0], &[Some(1e6)], 1.0, 1.0);
        let after = m.coefficients().unwrap()[0].clone();
        // it moves, but nowhere near 1e6
        assert!(
            (after[1] - before[1]).abs() < 100.0,
            "{:?} -> {:?}",
            before,
            after
        );
    }

    #[test]
    fn state_roundtrip() {
        let mut m1 = Robust::new(cfg(2, 1, RobustLoss::Huber { delta: 1.5 })).unwrap();
        let mut s = 81u64;
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
        let mut m2 = Robust::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
        for (x, y) in &rows[60..] {
            assert_eq!(
                m1.step(x, &[Some(*y)], 1.0, 1.0).pred,
                m2.step(x, &[Some(*y)], 1.0, 1.0).pred
            );
        }
    }

    #[test]
    fn new_surfaces_the_validation_error() {
        let e = Robust::new(cfg(1, 1, RobustLoss::Quantile { tau: 0.0 })).unwrap_err();
        assert!(e.contains("quantile must be in"), "{e}");
    }
}
