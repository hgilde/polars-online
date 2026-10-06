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
//! A nudge enters the cross-moment, a mean, as a step over the band's
//! weight, and the step is bounded so that the next solve moves the row's
//! prediction by at most its residual: the linearisation holds inside the
//! band, and a step past the row's own target is more than one term of the
//! score can justify.
//!
//! ```text
//! |step| <= |r| / (1 + u' A^-1 u)    (u' A^-1 u through the origin)
//! ```
//!
//! `A` is the band system the solve factorizes: the centred Gram over the
//! kept features, scaled under `standardize`, plus the ridge, or the raw
//! Gram through the origin. `u` is the row's deviation from the band's
//! means over the same columns, scaled alike, or its raw values through the
//! origin. The bound dates from review 2026-09-26 (G2), which read the
//! row's leverage off the Gram's diagonal; the full leverage `u' A^-1 u`
//! is review 2026-10-05's (TC1b).
//!
//! Under three rows per coefficient of the rows the target was present on,
//! the quantile fit warms up as ordinary least squares: a Newton step needs a
//! Hessian, and a band around a fit built from a handful of rows is not one.
//! And the band is never narrower than `(k/n)^(2/5)` of `s` for the target's
//! effective sample `n`, its present rows counted one each and decayed, so a
//! weight's scale reaches neither (docs/PLAN.md task 147) -- the smoothed-quantile bandwidth rate, which a long
//! stream leaves behind and which keeps the Hessian fed under a short
//! half-life, where the band's share of the sample is a few rows. The warm-up
//! read the band's weight until the second review of 2026-09-15 (F3), which a
//! half-life caps at that share, so a tail quantile at `half_life = 30` kept
//! falling back into warm-up and covered 0.825 where 0.9 was asked. What
//! that warm-up also did was rebuild a fit a row at the input bound had
//! left behind. Such a row sets the Gram and the cross-moment at its own
//! scale, and a mean-form accumulator forgets only through rows entering
//! it; every later row is outside the band (its residual is at the bound's
//! scale, where the band is a fraction of it), so only nudges arrive, each
//! `2h * psi * z` over the band's weight -- nothing beside the bound's
//! moments until that weight has decayed to nothing, and a step that
//! outgrows the band once it has. The fit oscillates, the band's weight
//! underflows after 1400 half-lives, and the prediction is withheld (the
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
use crate::solve::dot_aug;
use crate::{Decay, EwCov, SpdFactor};

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
    pub fit_intercept: bool,
    pub decay: Decay,
    pub loss: RobustLoss,
    pub ridge: f64,
    pub standardize: bool,
    pub min_weight: f64,
    pub solve_every: f64,
    pub max_rows_between_solves: u32,
    /// The default cadence (docs/PLAN.md task 115 (b)): solve once the weight
    /// learned since the last solve reaches this share of the weight the fit
    /// holds, in place of `solve_every`'s clock. In steady state that is the
    /// clock's own `half_life / 50` at a share of `ln 2 / 50`; where they part
    /// -- warm-up, after a gap, a half-life far longer than the stream -- it
    /// keeps the fit that close to its data, where the clock solved once and
    /// never again. `None` keeps the clock.
    #[serde(default)]
    pub solve_share: Option<f64>,
    /// Half-width of the band a quantile fit takes its Newton step in, in
    /// units of the EW residual std: rows inside carry the curvature, rows
    /// outside only the score (the module docs). It was the floor under `|r|`
    /// in an IRLS weight, bounding a weight rather than naming a band (review
    /// 2026-09-12, N9).
    pub quantile_eps: f64,
}

/// A target's band system as a solve factorizes it -- `A`, the band Gram
/// over the kept columns, scaled, with the ridge on its diagonal -- the
/// columns kept and their scales: what a nudge reads its row's full leverage
/// from (review 2026-10-05, TC1b). `factor` is `None` where no column was
/// kept: a step then moves the intercept alone, and through the origin
/// nothing.
#[derive(Debug, Clone)]
struct BandSystem {
    factor: Option<SpdFactor>,
    keep: Vec<usize>,
    s: Vec<f64>,
}

/// Each target's [`BandSystem`] while its Gram is the one the system was
/// built from: a row inside the band moves the Gram and drops it, decay
/// moves neither its moments nor `A`, and the next nudge after a drop builds
/// it again. Not state: rebuilt from the Gram, and equal whatever it holds.
#[derive(Debug, Clone, Default)]
struct BandSystems(Vec<Option<BandSystem>>);

impl PartialEq for BandSystems {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl BandSystems {
    /// Target `j`'s slot among `n`, the slots made on first use (a state
    /// loaded from a file carries none).
    fn slot(&mut self, n: usize, j: usize) -> &mut Option<BandSystem> {
        if self.0.len() != n {
            self.0 = vec![None; n];
        }
        &mut self.0[j]
    }
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
        self.n_features + usize::from(self.fit_intercept)
    }

    pub fn validate(&self) -> Result<(), String> {
        // The decay first: every model checks it in its own `new`, where only
        // the bank's spec did (review 2026-10-05, CF5).
        self.decay.check().map_err(|e| format!("robust: {e}"))?;
        if self
            .solve_share
            .is_some_and(|f| !(f.is_finite() && f > 0.0))
        {
            return Err("solve_share must be finite and > 0".into());
        }
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
    /// what the per-target `min_weight` gate reads (hard rule 8, S2) and the
    /// quantile fit's warm-up counts. `wj` stood in for it, and for the
    /// quantile that is the band's weight, which a half-life caps at the
    /// band's share of the sample (the second review of 2026-09-15, F1).
    wobs: Vec<f64>,
    /// Per target, the same rows counted one each, decayed alike: `wobs`
    /// over the rows' mean weight, which the quantile fit's warm-up and band
    /// floor read so that a weight's scale reaches neither (docs/PLAN.md
    /// task 147: rows at weight 100 left the warm-up on their first row,
    /// and the fit reached 1e51). A state written before it starts at `wobs`.
    #[serde(default)]
    nobs: Vec<f64>,
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
    /// what the loss does with them. This is what `n_eff` and `min_weight`
    /// mean everywhere else, so the robust models report it too: Huber scales
    /// the accumulators by its weights and a quantile fit weighs only the rows
    /// inside its band, and the observation count must follow neither. (The
    /// IRLS weights a quantile fit once used reached `2 / quantile_eps`, so
    /// counting them inflated `n_eff` by ~1000x -- T-A5.)
    w_raw: f64,
    beta: Option<Vec<Vec<f64>>>,
    clock_since_solve: f64,
    rows_since_solve: u32,
    /// Weight learned since the last solve, for `solve_share`.
    #[serde(default)]
    weight_since_solve: f64,
    pub solve_failures: u64,
    /// What each `ybar` leaves out: the mean is `ybar[j] + ybar_lo[j]`
    /// ([`crate::comp`]; docs/PLAN.md task 101). Empty in a state written
    /// before it. Before `zbuf`, which is skipped and so must stay last.
    #[serde(default)]
    ybar_lo: Vec<f64>,
    #[serde(skip)]
    zbuf: Vec<f64>,
    /// Each target's band system, kept from its last solve for the nudges
    /// that follow ([`BandSystems`]). Not state.
    #[serde(skip)]
    systems: BandSystems,
}

impl Robust {
    pub fn new(cfg: RobustCfg) -> Result<Self, String> {
        cfg.validate()?;
        let k = cfg.k_total();
        let m = cfg.n_targets;
        Ok(Self {
            // No window here, so no runs (docs/PLAN.md task 128).
            cov: vec![EwCov::new(k).without_runs(); m],
            wj: vec![0.0; m],
            wobs: vec![0.0; m],
            nobs: vec![0.0; m],
            cross: vec![vec![0.0; k]; m],
            ybar: vec![0.0; m],
            sig2: vec![0.0; m],
            wsig: vec![0.0; m],
            w_raw: 0.0,
            beta: None,
            clock_since_solve: 0.0,
            rows_since_solve: 0,
            weight_since_solve: 0.0,
            solve_failures: 0,
            ybar_lo: vec![0.0; m],
            zbuf: vec![0.0; k],
            systems: BandSystems(vec![None; m]),
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
    /// exists. `present` is the count of the rows this target was present
    /// on, decayed to the row (`nobs`): under `WARM_ROWS` of them per
    /// coefficient the quantile fit takes ordinary least-squares rows, since a
    /// Newton step needs a Hessian to lean on and a band around a fit built
    /// from a handful of rows is not one. That is the warm-up, and it is what
    /// rebuilds the fit after a gap or a reset has aged the weight away. Past
    /// it the band is at least `(k/present)^(2/5)` of `scale` wide, and
    /// `aged`, the band's own weight decayed to the row in rows of the
    /// target's mean weight, under one row per coefficient is a fit the data
    /// has left behind, which takes least-squares rows until the band holds
    /// rows again (the module docs). Both are counts, so a weight's scale
    /// reaches neither (docs/PLAN.md task 147).
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
            let solved = if self.cfg.fit_intercept {
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
        self.weight_since_solve = 0.0;
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
        let (asub, keep, s) = self.band_system(j);
        let kk = keep.len();
        let mut out = vec![0.0; k];
        let mut jitter = 0u32;
        let mut factor = None;
        if kk > 0 {
            let bsub: Vec<f64> = keep.iter().map(|&i| self.cross[j][i + 1] / s[i]).collect();
            // The factor's own solve is `solve_spd`'s, to the bit; it is kept
            // for the nudges that follow (`Self::nudge_movement`).
            let Some(f) = SpdFactor::of(&asub, kk) else {
                *self.systems.slot(self.cfg.n_targets, j) = None;
                return None;
            };
            let sol = f.solve(&bsub, kk, 1);
            jitter = f.attempts();
            for (i2, &i) in keep.iter().enumerate() {
                out[i + 1] = sol[i2] / s[i];
            }
            factor = Some(f);
        }
        let cov = &self.cov[j];
        let mut b0 = self.ybar[j];
        for i in 0..kf {
            b0 -= cov.mean(i + 1) * out[i + 1];
        }
        out[0] = b0;
        self.solve_failures += u64::from(jitter);
        *self.systems.slot(self.cfg.n_targets, j) = Some(BandSystem { factor, keep, s });
        Some(out)
    }

    /// Target `j`'s band system as its solve factorizes it, from the band
    /// Gram as it stands: `A` row-major over the kept columns, the columns
    /// kept and their scales. With an intercept, the centred Gram over the
    /// features (column `i` is slot `i + 1`), scaled by the centred
    /// deviations under `standardize` and keeping the columns whose variance
    /// is usable; through the origin, the raw Gram over every slot, scaled
    /// by the raw second moments under `standardize` and keeping the columns
    /// with one. The ridge is on `A`'s diagonal either way.
    fn band_system(&self, j: usize) -> (Vec<f64>, Vec<usize>, Vec<f64>) {
        let k = self.cfg.k_total();
        let cov = &self.cov[j];
        if self.cfg.fit_intercept {
            let kf = k - 1;
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
            let mut asub = vec![0.0; kk * kk];
            for (i2, &i) in keep.iter().enumerate() {
                for (j2, &jj) in keep.iter().enumerate() {
                    asub[i2 * kk + j2] = c[i * kf + jj] / (s[i] * s[jj]);
                }
                asub[i2 * kk + i2] += self.cfg.ridge;
            }
            (asub, keep, s)
        } else {
            let s: Vec<f64> = if self.cfg.standardize {
                (0..k).map(|i| cov.raw(i, i).max(0.0).sqrt()).collect()
            } else {
                vec![1.0; k]
            };
            // No centering here, so no cancellation: any strictly positive raw
            // moment is usable.
            let keep: Vec<usize> = (0..k).filter(|&i| s[i] > 0.0).collect();
            let kk = keep.len();
            let mut asub = vec![0.0; kk * kk];
            for (i2, &i) in keep.iter().enumerate() {
                for (j2, &jj) in keep.iter().enumerate() {
                    asub[i2 * kk + j2] = cov.raw(i, jj) / (s[i] * s[jj]);
                }
                asub[i2 * kk + i2] += self.cfg.ridge;
            }
            (asub, keep, s)
        }
    }

    /// How far a unit step moves the nudged row's prediction at the next
    /// solve, the row in `zbuf`: `1 + uᵀA⁻¹u` with an intercept, `u` the
    /// row's centred deviations over the kept columns, scaled, and `uᵀA⁻¹u`
    /// through the origin, `u` its scaled values -- the row's full leverage
    /// against the solve's own system (`Self::band_system`), ridge included,
    /// so a step bounded by it moves the row by no more than the bound. The
    /// diagonal's `Σ dᵢ²/vᵢ` read a row at (+1, −1) against features
    /// correlated at 0.999 at about 2, where its full leverage is about
    /// 2,000, and the solve threw such a row 6 to 13 times its residual
    /// past its target, 30 to 127 times at 0.9999 (review 2026-10-05,
    /// TC1b). The factor is kept, from the last solve or the last nudge
    /// that made one, until a row inside the band moves the Gram: a nudge
    /// costs `O(k²)`, and the first nudge after such a row makes the
    /// `O(k³)` factorization again. A kept feature with no spread in the
    /// band, on which the row deviates, has no curvature to lean on: the
    /// movement is unbounded and the step 0 (G2), as for a system no jitter
    /// factorizes.
    fn nudge_movement(&mut self, j: usize) -> f64 {
        let n = self.cfg.n_targets;
        if self.systems.slot(n, j).is_none() {
            let (a, keep, s) = self.band_system(j);
            let kk = keep.len();
            let factor = if kk > 0 {
                let Some(f) = SpdFactor::of(&a, kk) else {
                    return f64::INFINITY;
                };
                Some(f)
            } else {
                None
            };
            *self.systems.slot(n, j) = Some(BandSystem { factor, keep, s });
        }
        let sys = self.systems.0[j]
            .as_ref()
            .expect("the system is built above");
        let cov = &self.cov[j];
        let intercept = self.cfg.fit_intercept;
        let mut u = Vec::with_capacity(sys.keep.len());
        for &i in &sys.keep {
            let (d, spread) = if intercept {
                (
                    cov.deviation(i + 1, self.zbuf[i + 1]),
                    cov.cov(i + 1, i + 1),
                )
            } else {
                (self.zbuf[i], cov.raw(i, i))
            };
            if (spread.is_nan() || spread <= 0.0) && d != 0.0 {
                return f64::INFINITY;
            }
            u.push(d / sys.s[i]);
        }
        let q = match &sys.factor {
            Some(f) => f.quad_forms(&u, u.len(), 1)[0],
            None => 0.0,
        };
        if intercept { 1.0 + q } else { q }
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
        let (asub, keep, s) = self.band_system(j);
        let kk = keep.len();
        let mut out = vec![0.0; k];
        let mut jitter = 0u32;
        let mut factor = None;
        if kk > 0 {
            let bsub: Vec<f64> = keep.iter().map(|&i| b[i] / s[i]).collect();
            // Kept for the nudges that follow, as in `Self::solve_centred`.
            let Some(f) = SpdFactor::of(&asub, kk) else {
                *self.systems.slot(self.cfg.n_targets, j) = None;
                return None;
            };
            let sol = f.solve(&bsub, kk, 1);
            jitter = f.attempts();
            for (i2, &i) in keep.iter().enumerate() {
                out[i] = sol[i2] / s[i];
            }
            factor = Some(f);
        }
        self.solve_failures += u64::from(jitter);
        *self.systems.slot(self.cfg.n_targets, j) = Some(BandSystem { factor, keep, s });
        Some(out)
    }
}

impl OnlineModel for Robust {
    fn set_solve_share(&mut self, share: Option<f64>) {
        self.cfg.solve_share = share;
    }

    fn solve_share(&self) -> Option<f64> {
        self.cfg.solve_share
    }

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
        if self.cfg.fit_intercept {
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
            // did a row with no prediction (`min_weight` unmet after a clock
            // gap), so `σ²` -- the scale of every cut -- forgot less across
            // either than across a null (review 2026-09-12, S13; N6).
            self.wsig[j] *= lam;
            let present = lam * self.wobs[j];
            let rows = lam * self.nobs[j];
            let Some(yj) = y[j] else {
                self.cov[j].decay(lam);
                self.wj[j] *= lam;
                self.wobs[j] = present;
                self.nobs[j] = rows;
                continue;
            };
            // A row the target is present on counts at its raw weight, whatever
            // the loss then does with it (hard rule 8), and as one row.
            let counts = weight > 0.0 && weight.is_finite();
            self.wobs[j] = present + if counts { weight } else { 0.0 };
            self.nobs[j] = rows + if counts { 1.0 } else { 0.0 };
            let sigma = self.sig2[j].max(0.0).sqrt();
            let scale = if sigma > 0.0 { sigma } else { 1.0 };
            let aged = lam * self.wj[j];
            // The band's weight in rows of the target's mean weight.
            let aged_rows = if present > 0.0 {
                aged * (rows / present)
            } else {
                0.0
            };
            match self.row_update(yj, pred[j], scale, weight, rows, aged_rows) {
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
                    let dy = crate::comp::dev(
                        target,
                        self.ybar[j],
                        crate::comp::lo_of(&self.ybar_lo, j),
                    );
                    let ab_dy = a * bb * dy;
                    let cov = &self.cov[j];
                    for (i, (ci, &zi)) in self.cross[j].iter_mut().zip(&self.zbuf).enumerate() {
                        *ci = a * *ci + ab_dy * cov.deviation(i, zi);
                    }
                    crate::comp::add(
                        &mut self.ybar[j],
                        crate::comp::lo_slot(&mut self.ybar_lo, m, j),
                        bb * dy,
                    );
                    self.cov[j].update(&self.zbuf, lam, w);
                    self.wj[j] = wj_new;
                    // The Gram moved: the band system kept from the last
                    // solve is not this one's.
                    *self.systems.slot(m, j) = None;
                }
                RowUpdate::Nudge { nudge } => {
                    // Outside the band a row is one term of the score and none
                    // of the Hessian: no weight in the Gram, and `2h*psi(r)*z`
                    // into the cross-moment, which is a mean over `wj` -- so a
                    // sum's worth of nudge enters divided by it (N9). Centred,
                    // that is `ȳ += nudge/wj` and `c += nudge·(z − m)/wj`.
                    self.cov[j].decay(lam);
                    self.wj[j] = aged;
                    // A row of weight 0 learns nothing, the residual variance
                    // included, as the fit arm leaves it above: its nudge is
                    // 0, and `(wsig·σ² + 0)/(wsig + 0)` is not `σ²` to the bit
                    // (hard rule 9; review 2026-10-05, CC5).
                    if weight.is_nan() || weight <= 0.0 {
                        continue;
                    }
                    if nudge.is_finite() && aged > 0.0 {
                        // The step moves the fit at this row by the step
                        // times the row's full leverage against the band
                        // system (`Self::nudge_movement`), unbounded for a
                        // row far outside the data's spread, where the Gram
                        // holds no curvature for it. The linearisation holds
                        // inside the band, so a step past the row's own
                        // residual overshoots what one term of the score can
                        // justify: bounded to it, the row is brought at most
                        // to its target, not thrown past it. A target of
                        // `1e100` and features of `1e100` at weights from
                        // `1e-100` moved a slope to `1e248` this way, and the
                        // prediction at `1e100` read `-inf` (review
                        // 2026-09-26, G2). The leverage was the Gram's
                        // diagonal reading, which correlated features take
                        // far below the solve's (review 2026-10-05, TC1b).
                        let movement = self.nudge_movement(j);
                        let cov = &self.cov[j];
                        let most = (yj - pred[j]).abs() / movement;
                        let raw = nudge / aged;
                        let step = if raw.abs() > most {
                            most.copysign(raw)
                        } else {
                            raw
                        };
                        for (i, (ci, &zi)) in self.cross[j].iter_mut().zip(&self.zbuf).enumerate() {
                            *ci += step * cov.deviation(i, zi);
                        }
                        // A step of nothing is not taken (`crate::comp::add`
                        // says why): a nudge from a row of weight 0.
                        if step != 0.0 {
                            crate::comp::add(
                                &mut self.ybar[j],
                                crate::comp::lo_slot(&mut self.ybar_lo, m, j),
                                step,
                            );
                        }
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
        if weight.is_finite() && weight > 0.0 {
            self.weight_since_solve += weight;
        }
        let by_cadence = match self.cfg.solve_share {
            Some(share) => self.weight_since_solve >= share * self.w_raw,
            None => self.cfg.solve_every <= 0.0 || self.clock_since_solve >= self.cfg.solve_every,
        };
        let due = by_cadence
            || self.rows_since_solve >= self.cfg.max_rows_between_solves
            || (self.beta.is_none() && self.w_raw >= self.cfg.min_weight);
        if due {
            self.solve();
        }
        out
    }

    fn predict(&self, x: &[f64], _d_clock: f64) -> Step {
        let n_eff = self.w_raw;
        let mut pred = vec![f64::NAN; self.cfg.n_targets];
        if let (true, Some(beta)) = (n_eff >= self.cfg.min_weight, &self.beta) {
            for (j, p) in pred.iter_mut().enumerate() {
                if self.wj[j] > 0.0 {
                    *p = dot_aug(&beta[j], x, self.cfg.fit_intercept);
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
                let (n, k) = (m.cfg.n_targets, m.cfg.k_total());
                // One accumulator, one cross-moment row and one of each
                // scalar per target, all at the cfg's width; a short one
                // loaded and panicked on the first `step` (review
                // 2026-09-18, B3).
                // A state written before the row counts reads its weights as
                // them, which is what it was gated by.
                if m.nobs.is_empty() {
                    m.nobs = m.wobs.clone();
                }
                let per_target = [&m.wj, &m.wobs, &m.nobs, &m.ybar, &m.sig2, &m.wsig];
                if m.cov.len() != n
                    || m.cov.iter().any(|c| !c.has_shape(k))
                    || m.cross.len() != n
                    || m.cross.iter().any(|c| c.len() != k)
                    || per_target.iter().any(|v| v.len() != n)
                    || m.beta
                        .as_ref()
                        .is_some_and(|b| b.len() != n || b.iter().any(|v| v.len() != k))
                {
                    return Err(StateError::Invalid(
                        "robust: the accumulators have the wrong shape".into(),
                    ));
                }
                // No window here: a state written before the runs' flag keeps
                // none from here (review 2026-09-26, C4).
                m.cov.iter_mut().for_each(crate::EwCov::set_runs_off);
                m.zbuf = vec![0.0; k];
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

    /// A state whose vectors are not the cfg's is refused, where it loaded
    /// and panicked on the first `step` (review 2026-09-18, B3).
    #[test]
    fn a_state_of_the_wrong_shape_is_refused() {
        use crate::{ModelState, OnlineModel, StateError};
        let m = Robust::new(cfg(2, 1, RobustLoss::Huber { delta: 1.0 })).unwrap();
        let mut s = m.state();
        let ModelState::Robust(inner) = &mut s.model else {
            unreachable!()
        };
        inner.cross[0].pop();
        match Robust::restore(&s) {
            Err(StateError::Invalid(e)) => assert!(e.contains("wrong shape"), "{e}"),
            other => panic!("{other:?}"),
        }
    }
    use crate::{EwRidge, EwRidgeCfg};

    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }

    /// The stream the model contract's proptest failed on once its values
    /// reached the input bound (review 2026-09-26, G2): a target of `1e100`
    /// once, then features of `1e100` at weights from `1e-100` up. The fit
    /// extrapolated to `-1.5e199` at `1e100`, that residual set the scale to
    /// `5.8e148`, and the next row outside the band nudged the slope to
    /// `1e248` -- one term of the score against a Gram holding no curvature
    /// for a row that far out -- so the prediction at `1e100` read `-inf`.
    /// Bounded by the row's leverage, the nudge brings the row to its band's
    /// edge and every prediction is a number.
    #[test]
    fn a_quantile_fit_stays_finite_through_the_bound() {
        use crate::OnlineModel;
        let mut c = cfg(2, 2, RobustLoss::Quantile { tau: 0.5 });
        c.decay = Decay::Halflife(20.0);
        c.ridge = 1e-6;
        c.min_weight = 3.0;
        let mut m = Robust::new(c).unwrap();
        let rows: Vec<([f64; 2], Option<f64>, f64)> = vec![
            ([0.0, 0.0], None, 1.0),
            ([0.0, 0.0], Some(0.0), 1.0),
            ([0.0, 0.0], Some(0.0), 1.0),
            ([0.0, 0.0], Some(0.0), 1.0),
            ([0.0, 0.0], Some(0.0), 1.1785268524789025),
            ([0.0, 0.0], Some(0.0), 1.0),
            ([0.0, 0.0], Some(1e100), 2.0699847121199655),
            ([0.0, 0.0], Some(0.0), 1.4511913482607992),
            ([0.0, 0.0], Some(0.0), 1.0),
            ([1.4589775263789706, 0.0], Some(0.0), 1.0),
            ([1e100, 0.0], Some(0.0), 1e-100),
            ([0.0, 0.0], Some(0.0), 3.8874731502776667),
            ([1e100, 0.0], Some(0.0), 1.0),
            ([1e100, 0.0], None, 1.0),
        ];
        for (i, (x, y0, w)) in rows.iter().enumerate() {
            let out = m.step(x, &[*y0, None], if i == 0 { 0.0 } else { 1.0 }, *w);
            assert!(
                out.pred[0].is_nan() || out.pred[0].is_finite(),
                "row {i}: {:?}",
                out.pred
            );
            let beta = &m.beta.as_ref().unwrap()[0];
            assert!(beta.iter().all(|b| b.is_finite()), "row {i}: {beta:?}");
        }
        // The last nudge brought the row toward its band rather than past
        // it: the fit at `x = 1e100` is within the residual it started from.
        let at = m.predict(&[1e100, 0.0], 1.0).pred[0];
        assert!(at.is_finite() && at.abs() < 1e199, "{at:e}");
    }

    fn cfg(k: usize, m: usize, loss: RobustLoss) -> RobustCfg {
        RobustCfg {
            n_features: k,
            n_targets: m,
            fit_intercept: true,
            decay: Decay::Halflife(f64::INFINITY),
            loss,
            ridge: 1e-8,
            standardize: false,
            min_weight: (k + 1) as f64,
            solve_every: 0.0,
            max_rows_between_solves: 1,
            solve_share: None,
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
        // therefore `min_weight` -- meaningless. It must be the plain
        // weighted observation count, identical to every other model's, and it
        // still is now that the fit weighs the rows in its band (N9).
        for loss in [
            RobustLoss::Huber { delta: 1.5 },
            RobustLoss::Quantile { tau: 0.5 },
            RobustLoss::Quantile { tau: 0.9 },
        ] {
            let mut c = cfg(1, 1, loss);
            c.decay = Decay::Halflife(20.0);
            c.min_weight = 0.0;
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
        c.fit_intercept = true;
        c.min_weight = 2.0;
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
        c.min_weight = 2.0;
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
        c.min_weight = 2.0;
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

    /// A row of weight 0 learns nothing, the residual variance included:
    /// across every such row `σ²` keeps its bits, not `(a σ²) / a` --
    /// `kalman`'s twin (task 158) for both losses. A quantile row outside
    /// the band is a nudge, which skipped no zero weight before the `σ²`
    /// update, and moved it by an ulp on some rows (review 2026-10-05,
    /// CC5); the Huber row of weight 0 is a fit of weight 0, which did.
    #[test]
    fn a_zero_weight_row_keeps_the_residual_variance_to_the_bit() {
        for loss in [
            RobustLoss::Quantile { tau: 0.5 },
            RobustLoss::Quantile { tau: 0.9 },
            RobustLoss::Huber { delta: 1.345 },
        ] {
            let mut c = cfg(1, 1, loss);
            c.decay = Decay::Halflife(9.0);
            c.min_weight = 0.0;
            let mut m = Robust::new(c).unwrap();
            let mut s = 107u64;
            let mut checked = 0;
            for i in 0..3000 {
                let x = [lcg(&mut s)];
                let y = 1.0 + x[0] + 0.3 * lcg(&mut s);
                // Far outside the band, where a quantile row is a nudge.
                let zero = i % 3 == 2;
                let (y, w) = if zero { (y + 50.0, 0.0) } else { (y, 1.0) };
                let before = m.sigma2()[0];
                m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 0.7 }, w);
                if zero && i > 60 {
                    assert!(before > 0.0, "{loss:?}, row {i}");
                    assert_eq!(
                        m.sigma2()[0].to_bits(),
                        before.to_bits(),
                        "{loss:?}, row {i}"
                    );
                    checked += 1;
                }
            }
            assert!(checked > 900, "{loss:?}: {checked}");
        }
    }

    /// `σ²` is the EW mean of the squared out-of-sample errors: every row
    /// ages its weight, and a row with a target, a weight and a prediction
    /// adds `w·r²` (the row weight, not the robust one). Held on a stream
    /// with zero-weight rows (S13) and a clock gap that takes `n_eff` under
    /// `min_weight`, whose next rows have a target and no prediction and
    /// aged nothing either (N6).
    #[test]
    fn the_residual_variance_ages_on_every_row() {
        let hl = 10.0;
        let mut c = cfg(1, 1, RobustLoss::Huber { delta: 1.5 });
        c.decay = Decay::Halflife(hl);
        c.min_weight = 3.0;
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
            fit_intercept: true,
            decay: Decay::Halflife(f64::INFINITY),
            ridge: vec![1e-8],
            feature_sets: vec![],
            standardize: false,
            ridge_scale: false,
            session_shrink: None,
            long_half_life: None,
            coef_prior: None,
            min_weight: 2.0,
            solve_every: 0.0,
            max_rows_between_solves: 1,
            solve_share: None,
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
    /// band's weight, which a half-life caps at the band's share of the
    /// effective sample, so a `min_weight` above that share closed the gate
    /// for good.
    #[test]
    fn the_target_weight_is_the_rows_present_whatever_the_band_holds() {
        let hl = 30.0;
        let mut c = cfg(1, 1, RobustLoss::Quantile { tau: 0.9 });
        c.decay = Decay::Halflife(hl);
        c.min_weight = 0.0;
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
        c.min_weight = 0.0;
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
                c.min_weight = 10.0;
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
            c.min_weight = 20.0;
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
        c.min_weight = 0.0;
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

    /// Task 147: the same rows at a hundred times the weight make the same
    /// quantile fit. The warm-up and the band's floor read the target's
    /// weight against row counts, so rows at weight 100 left the warm-up on
    /// their first row, and the fit, leaning on a one-row Gram, reached 1e51.
    #[test]
    fn a_quantile_fit_is_the_same_at_a_hundred_times_the_weight() {
        let fit = |scale: f64| {
            let mut c = cfg(2, 1, RobustLoss::Quantile { tau: 0.8 });
            c.min_weight = 0.0;
            let mut m = Robust::new(c).unwrap();
            let mut s = 97u64;
            let mut preds = Vec::new();
            for i in 0..600 {
                let x = [lcg(&mut s), lcg(&mut s)];
                let y = 0.5 + x[0] - 0.25 * x[1] + 0.3 * lcg(&mut s);
                let w = scale * (0.5 + 0.5 * (lcg(&mut s) + 1.0));
                preds.push(
                    m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, w)
                        .pred[0],
                );
            }
            preds
        };
        let (one, many) = (fit(1.0), fit(100.0));
        for (i, (a, b)) in one.iter().zip(&many).enumerate() {
            assert!(
                (a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-9 * (1.0 + a.abs()),
                "row {i}: {a} against {b}"
            );
        }
    }

    /// N9: under the warm-up a quantile fit is ordinary least squares, bit for
    /// bit -- a Newton step needs a Hessian, and a band around a fit built from
    /// a handful of rows is not one.
    #[test]
    fn a_quantile_fit_warms_up_as_least_squares() {
        let mut qc = cfg(2, 1, RobustLoss::Quantile { tau: 0.7 });
        qc.min_weight = 0.0;
        let mut lc = cfg(2, 1, RobustLoss::Huber { delta: 1e9 });
        lc.min_weight = 0.0;
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

    /// A robust fit has no window, so it keeps no runs (docs/PLAN.md task
    /// 128).
    #[test]
    fn a_robust_fit_keeps_no_runs() {
        let m = Robust::new(cfg(2, 1, RobustLoss::Huber { delta: 1.0 })).unwrap();
        assert!(m.cov.iter().all(|c| !c.keeps_runs()));
    }

    /// A state from before the runs' flag keeps no runs once restored: this
    /// model has no window to read them (review 2026-09-26, C4).
    #[test]
    fn restore_keeps_no_runs() {
        let mut m = Robust::new(cfg(2, 1, RobustLoss::Huber { delta: 1.35 })).unwrap();
        let mut s = 3u64;
        for i in 0..20 {
            let x = [lcg(&mut s), lcg(&mut s)];
            let d = if i == 0 { 0.0 } else { 1.0 };
            crate::OnlineModel::step(&mut m, &x, &[Some(x[0])], d, 1.0);
        }
        let mut v = serde_json::to_value(crate::OnlineModel::state(&m)).unwrap();
        crate::window::json_edit(&mut v, "runs", &mut |x| {
            *x = serde_json::json!({"x": [], "start": []});
        });
        let m =
            <Robust as crate::OnlineModel>::restore(&serde_json::from_value(v).unwrap()).unwrap();
        assert!(m.cov.iter().all(|c| !c.keeps_runs()));
    }

    // --- task 158: the mutation survivors --------------------------------

    /// `solve_share` is refused unless it is finite and positive, and
    /// accepted at any such value.
    #[test]
    fn a_bad_solve_share_is_refused() {
        let huber = RobustLoss::Huber { delta: 1.5 };
        for f in [0.0, -0.1, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut c = cfg(2, 1, huber);
            c.solve_share = Some(f);
            match c.validate() {
                Err(e) => assert!(e.contains("solve_share must be finite and > 0"), "{f}: {e}"),
                Ok(()) => panic!("solve_share {f} accepted"),
            }
        }
        for f in [1e-9, 0.02, 1.0, 5.0] {
            let mut c = cfg(2, 1, huber);
            c.solve_share = Some(f);
            c.validate().unwrap();
        }
    }

    /// The warm-up is `WARM_ROWS` rows per coefficient, and no more: with
    /// three coefficients the row with nine before it is the first Newton
    /// step, so the tenth row's prediction is the first to part from least
    /// squares.
    #[test]
    fn the_warm_up_ends_at_three_rows_per_coefficient() {
        let mut qc = cfg(2, 1, RobustLoss::Quantile { tau: 0.7 });
        qc.min_weight = 0.0;
        let mut lc = cfg(2, 1, RobustLoss::Huber { delta: 1e9 });
        lc.min_weight = 0.0;
        let (mut q, mut l) = (Robust::new(qc).unwrap(), Robust::new(lc).unwrap());
        let mut s = 91u64;
        for i in 0..=10 {
            let x = [lcg(&mut s), lcg(&mut s)];
            let y = 0.5 + x[0] - 0.25 * x[1] + 0.3 * lcg(&mut s);
            let d = if i == 0 { 0.0 } else { 1.0 };
            let (qs, ls) = (
                q.step(&x, &[Some(y)], d, 1.0),
                l.step(&x, &[Some(y)], d, 1.0),
            );
            if i == 10 {
                assert_ne!(qs.pred, ls.pred, "the ninth row was a Newton step");
            } else if i > 0 {
                assert_eq!(qs.pred, ls.pred, "row {i} follows least-squares rows only");
            }
        }
    }

    /// The band's weight is read in rows of the target's mean weight
    /// (task 147): at a weight of 4 a row, a band holding 6 holds 1.5 rows,
    /// under one row per coefficient (`k = 2`), and the next row is a
    /// least-squares row; a band holding 8 holds exactly 2, one row per
    /// coefficient, and a row far outside it is a nudge. With no decay and
    /// equal weights the count is exactly a quarter of the weight.
    #[test]
    fn the_band_is_counted_in_rows_of_the_mean_weight() {
        let mut c = cfg(1, 1, RobustLoss::Quantile { tau: 0.5 });
        c.min_weight = 0.0;
        let mut m = Robust::new(c).unwrap();
        let mut s = 13u64;
        for i in 0..300 {
            let x = [lcg(&mut s)];
            let y = x[0] + 0.1 * lcg(&mut s);
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 4.0);
        }
        assert_eq!((m.nobs[0], m.wobs[0]), (300.0, 1200.0));
        for (band, least_squares) in [(6.0, true), (8.0, false)] {
            let mut t = m.clone();
            t.wj[0] = band;
            t.step(&[0.3], &[Some(1e6)], 1.0, 4.0);
            let want = if least_squares { band + 4.0 } else { band };
            assert_eq!(
                t.wj[0],
                want,
                "a band of {band} ({} rows): least squares {least_squares}",
                band / 4.0
            );
        }
    }

    /// The band is open, `|r| < h`: a residual of exactly `h` is outside it.
    /// Rows of zeros keep the fit at 0 and every residual exactly 0, so the
    /// scale is 1 and `h` is `quantile_eps` itself once the floor `(k/n)^(2/5)`
    /// is under it; a row at `h` then nudges (the band's weight stays) and
    /// one just under it is a fit row (the band's weight takes it).
    #[test]
    fn the_band_is_open_at_its_edge() {
        let mut c = cfg(1, 1, RobustLoss::Quantile { tau: 0.5 });
        c.min_weight = 0.0;
        c.quantile_eps = 0.5;
        let mut m = Robust::new(c).unwrap();
        for i in 0..20 {
            m.step(&[0.0], &[Some(0.0)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        assert_eq!(m.sig2[0], 0.0, "every residual so far is exactly 0");
        assert_eq!(m.predict(&[0.0], 1.0).pred[0], 0.0);
        assert!(
            (2.0f64 / 20.0).powf(0.4) < 0.5,
            "the floor is under quantile_eps"
        );
        let wj = m.wj[0];
        let mut edge = m.clone();
        edge.step(&[0.0], &[Some(0.5)], 1.0, 1.0);
        assert_eq!(edge.wj[0], wj, "a residual of h is outside the band");
        let mut inside = m.clone();
        inside.step(&[0.0], &[Some(0.5 - 1e-12)], 1.0, 1.0);
        assert_eq!(inside.wj[0], wj + 1.0, "one just under h is inside");
    }

    /// Through the origin and standardized, a feature that has only ever been
    /// 0 has no scale and is dropped with a coefficient of 0: the fit of the
    /// other is the fit without it, and no solve fails.
    #[test]
    fn a_feature_without_scale_is_dropped_through_the_origin() {
        let run = |dead: bool| {
            let mut c = cfg(
                if dead { 2 } else { 1 },
                1,
                RobustLoss::Huber { delta: 1e9 },
            );
            c.fit_intercept = false;
            c.standardize = true;
            let mut m = Robust::new(c).unwrap();
            let mut s = 17u64;
            for i in 0..50 {
                let a = lcg(&mut s);
                let x = if dead { vec![a, 0.0] } else { vec![a] };
                let y = 1.5 * a + 0.1 * lcg(&mut s);
                m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
            }
            (m.coefficients().unwrap()[0].clone(), m.solve_failures)
        };
        let ((with, failed), (without, _)) = (run(true), run(false));
        assert_eq!(failed, 0, "no solve failed");
        assert_eq!(with[1], 0.0, "the dead feature's coefficient");
        assert!(
            (with[0] - without[0]).abs() <= 1e-12 * without[0].abs(),
            "{} against {}",
            with[0],
            without[0]
        );
    }

    /// `a_solve_failure_is_counted_and_the_previous_fit_is_kept` through the
    /// origin: two equal features and no ridge make the raw Gram singular,
    /// the jitter ladder rescues it, and the rescue is counted.
    #[test]
    fn a_solve_failure_through_the_origin_is_counted() {
        let mut c = cfg(2, 1, RobustLoss::Huber { delta: 1.5 });
        c.ridge = 0.0;
        c.fit_intercept = false;
        c.min_weight = 2.0;
        let mut m = Robust::new(c).unwrap();
        let mut s = 101u64;
        for i in 0..40 {
            let a = lcg(&mut s);
            m.step(
                &[a, a],
                &[Some(2.0 * a)],
                if i == 0 { 0.0 } else { 1.0 },
                1.0,
            );
        }
        let beta = &m.coefficients().unwrap()[0];
        assert!(beta.iter().all(|v| v.is_finite()), "{beta:?}");
        assert!(m.solve_failures > 0, "a singular solve must be recorded");
    }

    /// The rows a model solves on: where its coefficients appear or move.
    /// Every row is fresh, so a solve always moves them.
    fn solve_rows(m: &mut Robust, n: usize) -> Vec<usize> {
        let mut s = 7u64;
        let mut out = Vec::new();
        for i in 0..n {
            let before = m.coefficients().map(<[Vec<f64>]>::to_vec);
            let x = [lcg(&mut s)];
            let y = x[0] + lcg(&mut s);
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
            if m.coefficients().map(<[Vec<f64>]>::to_vec) != before {
                out.push(i);
            }
        }
        out
    }

    /// The clock cadence: the first solve on the first row with `min_weight`
    /// (two rows here), then one each `solve_every` of clock (5), with
    /// `max_rows_between_solves` too far off to decide.
    #[test]
    fn the_clock_cadence_solves_every_solve_every() {
        let mut c = cfg(1, 1, RobustLoss::Huber { delta: 1e9 });
        c.solve_every = 5.0;
        c.max_rows_between_solves = 1000;
        let mut m = Robust::new(c).unwrap();
        assert_eq!(solve_rows(&mut m, 30), [1, 6, 11, 16, 21, 26]);
    }

    /// The row cadence: no clock (`solve_every` infinite), a solve every
    /// `max_rows_between_solves` rows (4) after the first.
    #[test]
    fn the_row_cadence_solves_every_max_rows_between_solves() {
        let mut c = cfg(1, 1, RobustLoss::Huber { delta: 1e9 });
        c.solve_every = f64::INFINITY;
        c.max_rows_between_solves = 4;
        let mut m = Robust::new(c).unwrap();
        assert_eq!(solve_rows(&mut m, 20), [1, 5, 9, 13, 17]);
    }

    /// The share cadence, set as the spec sets it (`set_solve_share`, after
    /// building): a solve once the weight learned since the last one reaches
    /// `solve_share` of the weight the fit holds (docs/PLAN.md task 115
    /// (b)), the clock left out. The rows are the rule's, written out.
    #[test]
    fn the_share_cadence_solves_at_its_share_of_the_weight() {
        let mut c = cfg(1, 1, RobustLoss::Huber { delta: 1e9 });
        c.solve_every = f64::INFINITY;
        c.max_rows_between_solves = 1000;
        let mut m = Robust::new(c).unwrap();
        assert_eq!(m.solve_share(), None);
        m.set_solve_share(Some(0.25));
        assert_eq!(m.solve_share(), Some(0.25));
        let (mut held, mut since, mut fit, mut want) = (0.0, 0.0, false, Vec::new());
        for i in 0..60 {
            held += 1.0;
            since += 1.0;
            if since >= 0.25 * held || (!fit && held >= 2.0) {
                want.push(i);
                (since, fit) = (0.0, true);
            }
        }
        assert!(want.len() > 8 && want[want.len() - 1] - want[want.len() - 2] > 5);
        assert_eq!(solve_rows(&mut m, 60), want);
        m.set_solve_share(None);
        assert_eq!(m.solve_share(), None);
    }

    /// A target that has never been present has no fit, so no prediction,
    /// beside one that has.
    #[test]
    fn a_target_never_present_has_no_prediction() {
        let mut c = cfg(1, 2, RobustLoss::Huber { delta: 1.5 });
        c.min_weight = 2.0;
        let mut m = Robust::new(c).unwrap();
        let mut s = 3u64;
        for i in 0..20 {
            let x = [lcg(&mut s)];
            m.step(&x, &[Some(x[0]), None], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let p = m.predict(&[0.5], 1.0).pred;
        assert!(p[0].is_finite() && p[1].is_nan(), "{p:?}");
    }

    /// A model deserialized on its own, outside `restore`, has no row buffer
    /// (it is not state); its first step makes one, and it goes on as the
    /// model it was saved from.
    #[test]
    fn a_model_deserialized_without_restore_steps_on() {
        let mut m = Robust::new(cfg(2, 1, RobustLoss::Huber { delta: 1.5 })).unwrap();
        let mut s = 5u64;
        let mut row = || {
            let x = [lcg(&mut s), lcg(&mut s)];
            (x, x[0] - x[1])
        };
        for i in 0..10 {
            let (x, y) = row();
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let bytes = rmp_serde::to_vec_named(&m).unwrap();
        let mut back: Robust = rmp_serde::from_slice(&bytes).unwrap();
        assert!(back.zbuf.is_empty(), "the buffer is not saved");
        for _ in 0..10 {
            let (x, y) = row();
            assert_eq!(
                m.step(&x, &[Some(y)], 1.0, 1.0).pred,
                back.step(&x, &[Some(y)], 1.0, 1.0).pred
            );
        }
    }

    /// Each of the shape checks on load refuses on its own: an accumulator
    /// of the wrong width, a cross-moment row too many, a per-target vector
    /// short, a coefficient row short, a coefficient row too many.
    #[test]
    fn each_shape_check_refuses_on_its_own() {
        let mut m = Robust::new(cfg(2, 1, RobustLoss::Huber { delta: 1.0 })).unwrap();
        let mut s = 9u64;
        for i in 0..10 {
            let x = [lcg(&mut s), lcg(&mut s)];
            m.step(&x, &[Some(x[0])], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        assert!(m.beta.is_some());
        type Spoil = (&'static str, fn(&mut Robust));
        let cases: [Spoil; 5] = [
            ("an accumulator two wide", |m| {
                m.cov[0] = EwCov::new(2).without_runs();
            }),
            ("a cross-moment row too many", |m| {
                m.cross.push(vec![0.0; 3])
            }),
            ("no residual variance", |m| m.sig2.clear()),
            ("a short coefficient row", |m| {
                m.beta.as_mut().unwrap()[0].truncate(2);
            }),
            ("a coefficient row too many", |m| {
                m.beta.as_mut().unwrap().push(vec![0.0; 3]);
            }),
        ];
        for (what, spoil) in cases {
            let mut bad = m.clone();
            spoil(&mut bad);
            match Robust::restore(&bad.state()) {
                Err(StateError::Invalid(e)) => assert!(e.contains("wrong shape"), "{what}: {e}"),
                other => panic!("{what}: {other:?}"),
            }
        }
        assert!(Robust::restore(&m.state()).is_ok());
    }

    /// A zero-weight row is not one of the rows a quantile fit counts toward
    /// its warm-up and band floor (hard rule 9): with no decay, a stream with
    /// zero-weight rows among its first rows is the stream without them.
    #[test]
    fn zero_weight_rows_do_not_count_toward_the_warm_up() {
        let run = |ghosts: bool| {
            let mut c = cfg(2, 1, RobustLoss::Quantile { tau: 0.7 });
            c.min_weight = 0.0;
            let mut m = Robust::new(c).unwrap();
            let mut s = 91u64;
            let mut preds = Vec::new();
            for i in 0..60 {
                let x = [lcg(&mut s), lcg(&mut s)];
                let y = 0.5 + x[0] - 0.25 * x[1] + 0.3 * lcg(&mut s);
                if ghosts && i < 8 {
                    m.step(&[-x[1], x[0]], &[Some(-y)], 0.0, 0.0);
                }
                let d = if i == 0 { 0.0 } else { 1.0 };
                preds.push(m.step(&x, &[Some(y)], d, 1.0).pred[0]);
            }
            (preds, m.nobs[0])
        };
        let ((with, n_with), (without, n_without)) = (run(true), run(false));
        assert_eq!(n_with, n_without, "the zero-weight rows were counted");
        for (i, (a, b)) in with.iter().zip(&without).enumerate() {
            assert!(
                (a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-12 * (1.0 + b.abs()),
                "row {i}: {a} against {b}"
            );
        }
    }

    /// Whether target 0's next row, `y` scored at `pred` at weight `w` after
    /// a clock step of `d`, is a nudge: `row_update`'s own reading of the
    /// model's state before the row, as `step` forms its inputs.
    fn is_nudge(m: &Robust, pred: f64, y: f64, d: f64, w: f64) -> bool {
        let lam = m.cfg.decay.factor(d);
        let sigma = m.sig2[0].max(0.0).sqrt();
        let scale = if sigma > 0.0 { sigma } else { 1.0 };
        let (present, rows, aged) = (lam * m.wobs[0], lam * m.nobs[0], lam * m.wj[0]);
        let aged_rows = if present > 0.0 {
            aged * (rows / present)
        } else {
            0.0
        };
        matches!(
            m.row_update(y, pred, scale, w, rows, aged_rows),
            RowUpdate::Nudge { .. }
        )
    }

    /// A nudge brings its row at most to its target, never past it (the
    /// module docs): the solve after a nudged row moves that row's
    /// prediction by no more than its residual. The step was bounded by the
    /// band Gram's diagonal leverage, `Σ d²/v`, where the solve moves the
    /// row by its full leverage, `1 + uᵀA⁻¹u` over the solve's own system:
    /// against features correlated at 0.999 a row at (+1, −1) reads about 2
    /// on the diagonal and about 2,000 in full, and was thrown 6 to 13 times
    /// its residual, 30 to 127 times at 0.9999 (review 2026-10-05, TC1b;
    /// this test on the old code: 5.9 to 7.5, and 30 to 36). Every
    /// nudged row is held to it -- with an intercept, standardized or not,
    /// and through the origin -- and on uncorrelated features, where the
    /// two leverages agree. Every hundredth row past the warm-up runs
    /// against the correlation, at (+1, −1), 0.4 above the true line.
    #[test]
    fn a_nudge_never_moves_its_row_past_its_residual() {
        for rho in [0.0, 0.999, 0.9999] {
            for (fit_intercept, standardize) in [(true, false), (true, true), (false, false)] {
                let mut c = cfg(2, 1, RobustLoss::Quantile { tau: 0.5 });
                c.decay = Decay::Halflife(500.0);
                c.min_weight = 0.0;
                c.ridge = 1e-6;
                c.quantile_eps = 0.2;
                c.fit_intercept = fit_intercept;
                c.standardize = standardize;
                let mut m = Robust::new(c).unwrap();
                let mut s = 5u64;
                let (mut checked, mut glitches, mut worst) = (0, 0, (0.0f64, 0usize));
                for i in 0..3000usize {
                    let a = 3f64.sqrt() * lcg(&mut s);
                    let b = rho * a + (1.0 - rho * rho).sqrt() * 3f64.sqrt() * lcg(&mut s);
                    let glitch = i >= 1000 && i % 100 == 0;
                    let x = if glitch { [1.0, -1.0] } else { [a, b] };
                    let noise = if glitch { 0.4 } else { 0.5 * lcg(&mut s) };
                    // Through the origin the line through the origin.
                    let level = if fit_intercept { 1.0 } else { 0.0 };
                    let y = level + 0.8 * x[0] - 0.4 * x[1] + noise;
                    let d = if i == 0 { 0.0 } else { 1.0 };
                    let before = m.predict(&x, d).pred[0];
                    let nudge = before.is_finite() && is_nudge(&m, before, y, d, 1.0);
                    m.step(&x, &[Some(y)], d, 1.0);
                    if nudge {
                        let after = m.predict(&x, 1.0).pred[0];
                        let moved = (after - before).abs() / (y - before).abs();
                        // NaN, a movement not a number, counts as the worst.
                        if moved.is_nan() || moved > worst.0 {
                            worst = (moved, i);
                        }
                        checked += 1;
                        glitches += usize::from(glitch);
                    }
                }
                let case =
                    format!("rho {rho}, intercept {fit_intercept}, standardize {standardize}");
                assert!(checked > 300, "{case}: {checked} nudges");
                assert!(glitches > 0, "{case}: no glitch row was nudged");
                assert!(
                    worst.0 <= 1.0 + 1e-9,
                    "{case}: row {} moved {:.3} times its residual",
                    worst.1,
                    worst.0
                );
            }
        }
    }

    /// On uncorrelated features the full leverage is the diagonal one: a
    /// band Gram set to a diagonal matrix reads a row's leverage as `Σ
    /// d²/(v + ridge)` unstandardized and `Σ (d²/v)/(1 + ridge)` standardized,
    /// to rounding, and so as the old reading, `Σ d²/v`, to the ridge's
    /// share of the smallest variance.
    #[test]
    fn the_full_leverage_is_the_diagonal_one_on_uncorrelated_features() {
        for standardize in [false, true] {
            let mut c = cfg(3, 1, RobustLoss::Quantile { tau: 0.5 });
            c.ridge = 1e-6;
            c.standardize = standardize;
            let mut m = Robust::new(c).unwrap();
            let mut s = 7u64;
            for i in 0..50 {
                let x = [lcg(&mut s), lcg(&mut s), lcg(&mut s)];
                m.step(
                    &x,
                    &[Some(x[0] - x[2])],
                    if i == 0 { 0.0 } else { 1.0 },
                    1.0,
                );
            }
            let (mean, var) = ([1.0, 0.2, -0.3, 0.5], [0.0, 0.7, 1.9, 0.04]);
            let mut cen = vec![0.0; 16];
            for i in 0..4 {
                cen[i * 4 + i] = var[i];
            }
            let (w, q) = (m.cov[0].n_eff(), m.cov[0].q_sum());
            m.cov[0].set_moments(&mean, &cen, w, q);
            m.systems.0[0] = None;
            m.zbuf = vec![1.0, 1.1, -2.0, 0.9];
            let ridge = 1e-6;
            let term = |i: usize| (m.zbuf[i] - mean[i]).powi(2);
            let with_ridge: f64 = 1.0
                + (1..4)
                    .map(|i| {
                        if standardize {
                            term(i) / var[i] / (1.0 + ridge)
                        } else {
                            term(i) / (var[i] + ridge)
                        }
                    })
                    .sum::<f64>();
            let old: f64 = 1.0 + (1..4).map(|i| term(i) / var[i]).sum::<f64>();
            let full = m.nudge_movement(0);
            assert!(
                (full - with_ridge).abs() <= 1e-12 * with_ridge,
                "standardize {standardize}: {full} against {with_ridge}"
            );
            assert!(
                (full - old).abs() <= ridge / 0.04 * old,
                "standardize {standardize}: {full} against the old {old}"
            );
        }
    }

    /// A nudge is bounded by the row's own residual over its leverage (G2).
    /// At the value a feature has always taken the leverage is 0, so a heavy
    /// row outside the band moves the fit exactly to its target and no
    /// further; off that value the leverage is unbounded (the Gram holds no
    /// curvature for it) and the fit does not move. At a weight of `1e4` the
    /// step before the bound is far past the residual, so the bound decides.
    #[test]
    fn a_nudge_on_a_feature_without_spread_is_bounded_by_the_residual() {
        let mut c = cfg(1, 1, RobustLoss::Quantile { tau: 0.5 });
        c.min_weight = 0.0;
        let mut m = Robust::new(c).unwrap();
        let mut s = 23u64;
        for i in 0..40 {
            let y = 0.1 * lcg(&mut s);
            m.step(&[1.0], &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        assert_eq!(m.cov[0].cov(1, 1), 0.0, "the feature has no spread");
        let p = m.predict(&[1.0], 1.0).pred[0];
        let mut on = m.clone();
        on.step(&[1.0], &[Some(p + 0.5)], 1.0, 1e4);
        assert_eq!(on.wj[0], m.wj[0], "the row was a nudge");
        let q = on.predict(&[1.0], 1.0).pred[0];
        assert!(
            (q - (p + 0.5)).abs() <= 1e-12,
            "brought to its target: {p} -> {q}, target {}",
            p + 0.5
        );
        let mut off = m.clone();
        let p2 = off.predict(&[2.0], 1.0).pred[0];
        off.step(&[2.0], &[Some(p2 + 0.5)], 1.0, 1e4);
        assert_eq!(off.wj[0], m.wj[0], "the row was a nudge");
        assert_eq!(off.coefficients(), m.coefficients(), "and moved nothing");
    }
}
