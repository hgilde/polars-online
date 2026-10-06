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
//! P_j <- Phi P_j Phi + Q * d^2        (predict; Q times the clock step squared)
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
//! zero"; give the intercept `inf` to leave it a random walk. With `Q`
//! from `coef_half_life` a reverting slot settles, at rows `d` apart, at the
//! prior variance `q_i d^2 / (1 - phi_i^2)`, a stationary AR(1) instead of an
//! unbounded walk (for `d` well under `r_i` that is about `q_i d r_i / (2 ln2)`,
//! which grows with the spacing as the gain matching's own variance does).
//!
//! **Process noise from a per-factor half-life.** On standardized features, the
//! steady-state gain of a random-walk-beta filter matches EW-RLS with half-life
//! `h_i` when the noise added for a row `d` clock units after the last is
//! `sigma^2 * (ln2 * d / h_i)^2 = q_i d^2` with `q_i = sigma^2 * (ln2 / h_i)^2`
//! (docs/PLAN.md §4.4, task 150): EW-RLS forgets `2^(-d/h)` over the step, a
//! per-row half-life of `h/d` rows, and that is the half-life the matching
//! is done at. Added as `q_i d` -- a random walk whose variance grows with
//! the clock -- the gain grew with the root of the spacing, and a coefficient
//! adapted in `h sqrt(d)` clock units rather than `h` (measured on 0.13.0's
//! stream: 74, 38 and 13 clock units at rows 1, 0.25 and 0.04 apart; the two
//! forms are identical at unit spacing, so a number from another stream or
//! criterion is not comparable to these). The match is first order in `d/h`:
//! the exact per-row noise is `sigma^2 g^2 / (1 - g)` with `g = 1 - 2^(-d/h)`,
//! which `(ln2 d/h)^2` is within 1% of for rows closer than half a half-life
//! and 4% at one. `half_life` may be scalar or per factor; `half_life = inf`
//! gives `q_i = 0`, pinning that coefficient. An explicit `q` overrides the
//! derivation and is added as `q_i d^2` too: the noise a row one clock unit
//! after the last adds, not a variance per unit of elapsed clock.
//!
//! Features are standardized internally against a shared [`EwDiag`] over `z`
//! (EW means and variances, O(k) a row; a full `EwCov` until schema 3, of
//! which only the diagonal was ever read), so `q_i` is on a comparable scale
//! across features; `R_j` defaults to the EW residual variance `sigma^2_j`
//! unless `obs_var` is given.
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
//! is the mean `sigma^2` across targets, and before any has one, the mean of
//! the squared innovations of the targets the row observes. A given
//! `obs_var` is the noise throughout. [`Kalman::pred_var`] keeps NaN for
//! `R_j` until there is a residual variance: it describes a prediction,
//! which is made before the row's target is seen.
//!
//! **The prior is `p0` times the first noise (CC4).** `P_j` starts unsized,
//! all zero. The first row with a noise sets it to `P_0 = p0 R I`, with `R`
//! that row's noise, and the row's gain reads it; that row adds no process
//! noise. Until then nothing is added to `P_j`, an explicit `q` included.
//! So `p0` is a ratio: at 1 the prior is as uncertain as one observation,
//! the conjugate prior of a regression with an unknown noise variance. The
//! first correction, `p0 z e / (p0 z'z + 1 / w)` for `P_0 = p0 e^2 I`, does
//! not see the target's units. `p0` was a variance in the target's units
//! squared, so its default of 1 tied a warm-up to them, more tightly once
//! the noise came from the data. With `obs_var` given, `P_0 = p0 obs_var I`
//! from the start, so `p0` means the same thing, and with no process noise
//! the filter is the ridge regression with penalty `1 / p0`. Scaling
//! every target by `c` (with any `obs_var` or explicit `q` by `c^2`) scales
//! every prediction by `c`, at any `p0`. A `P` that has decayed to exactly 0
//! through the reversion holds nothing, and is sized again.
//!
//! `P` is per target because the Riccati recursion depends on `R_j`. With
//! `share_p` the filter keeps one `P` driven by the mean `sigma^2` across
//! targets (docs/PLAN.md §4.4 [validate]).

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
    /// Standardize features internally before filtering (default).
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
            if q.iter().any(|&v| v.is_nan() || v < 0.0) {
                return Err("kalman: q values must be >= 0".into());
            }
        } else {
            if self.half_life.len() != 1 && self.half_life.len() != k {
                return Err(format!("kalman: half_life must have length 1 or {k}"));
            }
            if self.half_life.iter().any(|&h| h.is_nan() || h <= 0.0) {
                return Err("kalman: half_life values must be > 0 (inf pins)".into());
            }
        }
        if self.revert_half_life.len() != 1 && self.revert_half_life.len() != k {
            return Err(format!(
                "kalman: revert_half_life must have length 1 or {k}"
            ));
        }
        if self
            .revert_half_life
            .iter()
            .any(|&h| h.is_nan() || h <= 0.0)
        {
            return Err("kalman: revert_half_life values must be > 0 (inf = random walk)".into());
        }
        if self.p0.is_nan() || self.p0 <= 0.0 {
            return Err("kalman: p0 must be > 0".into());
        }
        if self.obs_var.is_some_and(|v| v.is_nan() || v <= 0.0) {
            return Err("kalman: obs_var must be > 0".into());
        }
        Ok(())
    }

    /// Whether any slot reverts (`Phi != I`). The default random walk skips
    /// the transition entirely, so it stays bit-identical to before E41.
    fn reverts(&self) -> bool {
        self.revert_half_life.iter().any(|h| h.is_finite())
    }

    /// The transition factor of slot `i` over a clock delta `d`,
    /// `2^(-d/r_i)`, spelled as [`Decay::factor`] is.
    fn phi(&self, i: usize, d_clock: f64) -> f64 {
        let r = if self.revert_half_life.len() == 1 {
            self.revert_half_life[0]
        } else {
            self.revert_half_life[i]
        };
        Decay::Halflife(r).factor(d_clock)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "KalmanV3")]
pub struct Kalman {
    cfg: KalmanCfg,
    /// Standardization stats over `z` (shared across targets): the means and
    /// variances the scales are read from.
    stats: EwDiag,
    /// Per target: coefficient mean on the standardized scale.
    beta: Vec<Vec<f64>>,
    /// Per target (or one when `share_p`): covariance, row-major `k*k`; all
    /// zero, unsized, until a row sizes the noise (the module doc, CC4).
    p: Vec<Vec<f64>>,
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
    /// slot reverts).
    #[serde(skip)]
    phi: Vec<f64>,
    /// This row's feature scales and process-noise variances, kept between
    /// rows so a step allocates nothing for them (docs/PERFORMANCE.md §13).
    #[serde(skip)]
    sbuf: Vec<f64>,
    #[serde(skip)]
    qbuf: Vec<f64>,
}

/// The layouts `Kalman` loads. Schema-2 files standardized with a full
/// [`EwCov`] under `cov`; its diagonal is what the model read, and is what
/// the schema-3 `stats` holds, so the conversion is a copy of the same
/// numbers. Newtype variants for the reason `RlsWire` gives: the compact
/// msgpack encoding writes structs as arrays. The two are told apart by the
/// field name in a map and by [`EwDiag`]'s refusal of an `EwCov`'s shape in
/// an array.
#[derive(Deserialize)]
struct KalmanV3 {
    cfg: KalmanCfg,
    stats: EwDiag,
    beta: Vec<Vec<f64>>,
    p: Vec<Vec<f64>>,
    sig2: Vec<f64>,
    wsig: Vec<f64>,
    wj: Vec<f64>,
}

impl TryFrom<KalmanV3> for Kalman {
    type Error = String;

    fn try_from(v: KalmanV3) -> Result<Self, String> {
        let (cfg, stats, beta, p, sig2, wsig, wj) =
            (v.cfg, v.stats, v.beta, v.p, v.sig2, v.wsig, v.wj);
        let k = cfg.k_total();
        let m = cfg.n_targets;
        let n_p = if cfg.share_p { 1 } else { m };
        if stats.k() != k
            || beta.len() != m
            || beta.iter().any(|b| b.len() != k)
            || p.len() != n_p
            || p.iter().any(|p| p.len() != k * k)
            || sig2.len() != m
            || wsig.len() != m
            || wj.len() != m
        {
            return Err("kalman: state has the wrong shape".into());
        }
        // The scratch buffers are sized by `ensure_buffers` on the first use.
        Ok(Self {
            cfg,
            stats,
            beta,
            p,
            sig2,
            wsig,
            wj,
            zbuf: vec![],
            zs: vec![],
            pz: vec![],
            gain: vec![],
            zb: vec![],
            phi: vec![],
            sbuf: vec![],
            qbuf: vec![],
        })
    }
}

impl Kalman {
    pub fn new(cfg: KalmanCfg) -> Result<Self, String> {
        cfg.validate()?;
        let k = cfg.k_total();
        let m = cfg.n_targets;
        let n_p = if cfg.share_p { 1 } else { m };
        // Unsized, all zero, until a row sizes the noise; with `obs_var` the
        // noise is known now, and so is the prior (the module doc, CC4).
        let mut p_init = vec![0.0; k * k];
        if let Some(r) = cfg.obs_var {
            for i in 0..k {
                p_init[i * k + i] = cfg.p0 * r;
            }
        }
        Ok(Self {
            stats: EwDiag::new(k),
            beta: vec![vec![0.0; k]; m],
            p: vec![p_init; n_p],
            sig2: vec![0.0; m],
            wsig: vec![0.0; m],
            wj: vec![0.0; m],
            zbuf: vec![0.0; k],
            zs: vec![0.0; k],
            pz: vec![0.0; k],
            gain: vec![0.0; k],
            zb: vec![0.0; m],
            phi: vec![1.0; k],
            sbuf: vec![1.0; k],
            qbuf: vec![0.0; k],
            cfg,
        })
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
    /// process noise for the next step's gap, `zᵀ(Φ P Φᵀ + Q·Δ²)z + R`. The
    /// two differ by `zᵀ(Φ P Φᵀ − P + Q·Δ²)z`, which the default random walk
    /// (`Φ = I`) reduces to `Q·Δ²`: negligible under a half-life-derived `q`
    /// (`q/R = (ln2/h)²`, 0.005 % at half-life 100 and unit spacing, `Δ²`
    /// times that at spacing `Δ`), not under a large explicit `q` (review
    /// 2026-09-18, D2).
    ///
    /// This is the piece `sigma` alone cannot give. `sigma` is the spread of
    /// realized errors; this also knows how unsure the filter is about its own
    /// coefficients, so it is wide during warmup and after a gap, and narrows
    /// as evidence accumulates. Only Kalman tracks `P`, so only Kalman can
    /// report it exactly.
    ///
    /// For the row `x`, standardized as `predict` standardizes it. It read
    /// the regressor of the last row stepped, a scratch a save does not
    /// keep, so a loaded filter answered for no regressor at all -- `R` alone
    /// (review 2026-09-12, V7).
    ///
    /// NaN until the target has a residual variance. The noise `step` uses
    /// before then is the row's own innovation squared, and a prediction is
    /// made before its row's target is seen, so it has no innovation to read
    /// (review 2026-10-05, CC4).
    pub fn pred_var(&self, x: &[f64]) -> Vec<f64> {
        let k = self.cfg.k_total();
        let z = self.standardized(x);
        (0..self.cfg.n_targets)
            .map(|j| {
                let pi = if self.cfg.share_p { 0 } else { j };
                let p = &self.p[pi];
                let mut quad = 0.0;
                for i in 0..k {
                    let row = i * k;
                    let mut acc = 0.0;
                    for jj in 0..k {
                        acc += p[row + jj] * z[jj];
                    }
                    quad += z[i] * acc;
                }
                let r = self.cfg.obs_var.unwrap_or_else(|| {
                    let s2 = if self.cfg.share_p {
                        self.sig2.iter().sum::<f64>() / self.cfg.n_targets as f64
                    } else {
                        self.sig2[j]
                    };
                    if s2 > 0.0 { s2 } else { f64::NAN }
                });
                quad + r
            })
            .collect()
    }

    pub fn n_eff(&self) -> f64 {
        self.stats.n_eff()
    }

    /// Feature scales used for standardization: sd for features, 1 for the
    /// intercept slot. Zero-variance features get scale 1 (their standardized
    /// value is then their centered value, i.e. 0). All ones when
    /// `standardize` is off.
    fn scales(&self) -> Vec<f64> {
        let mut out = vec![1.0; self.cfg.k_total()];
        self.scales_into(&mut out);
        out
    }

    /// [`Self::scales`] into a caller's buffer of length `k_total`.
    fn scales_into(&self, out: &mut [f64]) {
        let off = usize::from(self.cfg.fit_intercept);
        for (i, s) in out.iter_mut().enumerate() {
            *s = if !self.cfg.standardize || i < off {
                1.0
            } else if off == 0 {
                // No intercept: nothing to centre on, so the scale is the raw
                // second moment's -- any positive one is usable, there being
                // no cancellation (review 2026-09-12, C10).
                let raw = self.stats.raw(i);
                if raw > 0.0 { raw.sqrt() } else { 1.0 }
            } else {
                let v = self.stats.var(i);
                let raw = self.stats.raw(i);
                if crate::variance_is_usable(v, raw) {
                    v.sqrt()
                } else {
                    1.0
                }
            };
        }
    }

    /// Coefficients in the ORIGINAL feature units, per target.
    ///
    /// Under `standardize` the filter's state lives in standardized
    /// coordinates, and each row's correction was made in the coordinates of
    /// the feature means and scales as they stood on that row. They are read
    /// out here with today's means and scales, so they are the coefficients
    /// `pred` uses -- the two read the same state the same way -- but a
    /// coefficient "per unit of `x`" moves with the standardizer as well as
    /// with the fit, most in the early rows, while the scales settle
    /// (review 2026-09-12, D3).
    pub fn coefficients(&self) -> Vec<Vec<f64>> {
        if !self.cfg.standardize {
            return self.beta.clone();
        }
        let k = self.cfg.k_total();
        let off = usize::from(self.cfg.fit_intercept);
        let s = self.scales();
        let mut out = Vec::with_capacity(self.cfg.n_targets);
        for b in &self.beta {
            let mut c = vec![0.0; k];
            for (i, ci) in c.iter_mut().enumerate().skip(off) {
                *ci = b[i] / s[i];
            }
            if self.cfg.fit_intercept {
                // b0_std is on centered features: unshift by the feature means.
                let mut b0 = b[0];
                for (i, ci) in c.iter().enumerate().skip(off) {
                    b0 -= ci * self.stats.mean(i);
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

    /// `[1, x]` standardized against the stats as they stand, as `step`
    /// fills `zs` before its own update.
    fn standardized(&self, x: &[f64]) -> Vec<f64> {
        let off = usize::from(self.cfg.fit_intercept);
        let s = self.scales();
        (0..self.cfg.k_total())
            .map(|i| {
                let raw = if i < off { 1.0 } else { x[i - off] };
                if !self.cfg.standardize || i < off {
                    raw
                } else if off == 0 {
                    // No intercept to absorb a shift: scale only (C10).
                    raw / s[i]
                } else {
                    self.stats.deviation(i, raw) / s[i]
                }
            })
            .collect()
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
            self.sbuf = vec![1.0; k];
            self.qbuf = vec![0.0; k];
        }
    }

    /// Whether covariance `pi` is unsized: all zero, as it is from the
    /// start until a row sizes the noise (the module doc, CC4). A sized `P`
    /// has a positive diagonal. One that the reversion has shrunk to exactly
    /// 0, every slot reverting over hundreds of its half-lives with no
    /// process noise to restore it, holds nothing, and is sized again.
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
            let (mut sum, mut n) = (0.0, 0u32);
            for (yj, zb) in y.iter().zip(&self.zb) {
                if let Some(v) = yj {
                    let e = v - zb;
                    let e2 = e * e;
                    if e2.is_finite() {
                        sum += e2;
                        n += 1;
                    }
                }
            }
            if n > 0 {
                return first_noise(sum / f64::from(n));
            }
        }
        0.0
    }

    /// `b <- Phi b`, `P <- Phi P Phi` for a clock delta `d_clock`: the
    /// coefficient means shrink toward zero and the covariance with them.
    /// A no-op (and skipped) under the default random walk.
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
        for b in &mut self.beta {
            for (bi, ph) in b.iter_mut().zip(&self.phi) {
                *bi *= ph;
            }
        }
        for p in &mut self.p {
            for i in 0..k {
                let pi = self.phi[i];
                for (pij, pj) in p[i * k..(i + 1) * k].iter_mut().zip(&self.phi) {
                    *pij *= pi * pj;
                }
            }
        }
    }
}

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

impl OnlineModel for Kalman {
    fn target_n_eff_into(&self, out: &mut Vec<f64>) -> bool {
        out.clear();
        out.extend_from_slice(&self.wj);
        true
    }

    fn step(&mut self, x: &[f64], y: &[Option<f64>], d_clock: f64, weight: f64) -> Step {
        self.ensure_buffers();
        let k = self.cfg.k_total();
        let m = self.cfg.n_targets;
        let off = usize::from(self.cfg.fit_intercept);
        let lam = self.cfg.decay.factor(d_clock);

        if self.cfg.fit_intercept {
            self.zbuf[0] = 1.0;
            self.zbuf[1..].copy_from_slice(x);
        } else {
            self.zbuf.copy_from_slice(x);
        }

        // The clock has moved by `d_clock` since the last row: the state
        // is propagated before it predicts (`predict` does the same).
        self.transition(d_clock);

        // Standardized regressors from the stats BEFORE this row's update.
        let mut s = std::mem::take(&mut self.sbuf);
        self.scales_into(&mut s);
        for (i, zs) in self.zs.iter_mut().enumerate() {
            *zs = if !self.cfg.standardize {
                self.zbuf[i]
            } else if i < off {
                1.0
            } else if off == 0 {
                // No intercept to absorb a shift: scale only. Centring here
                // gave every prediction a hidden intercept `-Σ b_i m_i / s_i`
                // that `coefficients()` has no slot for, so `coef · x` missed
                // `pred` by it (review 2026-09-12, C10).
                self.zbuf[i] / s[i]
            } else {
                self.stats.deviation(i, self.zbuf[i]) / s[i]
            };
        }
        self.sbuf = s;

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

        // ---- Kalman update per target ----
        // `share_p`'s noise before the targets have a residual variance, read
        // at most once a row (the module doc, CC4).
        let mut shared_first: Option<f64> = None;
        for j in 0..m {
            let pi = if self.cfg.share_p { 0 } else { j };
            // A null target, or a present one at weight zero -- an observation
            // of infinite variance, `σ²/0` -- is a prediction step and no
            // update, and time passes for both weights alike. The zero weight
            // skipped the decay, so `σ²`, which sets `R` and `Q`, forgot less
            // across it than across a null (review 2026-09-12, S9).
            let obs = y[j].filter(|_| weight > 0.0);
            // `R`, and the `σ²` the process noise is derived from: `obs_var`,
            // else the residual variance, else -- before there is one -- the
            // row's own innovation squared (CC4). 0 is no noise to weigh the
            // row against: it corrects nothing and adds no derived `Q`.
            let sigma2 = match self.cfg.obs_var {
                Some(v) => v,
                None => {
                    let s2 = if self.cfg.share_p {
                        self.sig2.iter().sum::<f64>() / m as f64
                    } else {
                        self.sig2[j]
                    };
                    if s2 > 0.0 {
                        s2
                    } else if self.cfg.share_p {
                        *shared_first.get_or_insert_with(|| self.shared_first_noise(y, weight))
                    } else {
                        obs.map_or(0.0, |yj| {
                            let e = yj - self.zb[j];
                            first_noise(e * e)
                        })
                    }
                }
            };
            // Process step: P += Q * d_clock^2, after the transition above
            // (only for the target that owns P, or once when shared). The
            // square is what keeps `coef_half_life` a clock half-life at any
            // row spacing (docs/PLAN.md task 150; the module doc). An unsized
            // `P` takes no process noise; the first row with a noise sizes it
            // instead, to `p0` times that noise, before its gain (CC4).
            if (!self.cfg.share_p || j == 0) && self.is_unsized(pi) {
                let v = self.cfg.p0 * sigma2;
                if v > 0.0 && v.is_finite() {
                    let p = &mut self.p[pi];
                    p.fill(0.0);
                    for i in 0..k {
                        p[i * k + i] = v;
                    }
                }
            } else if !self.cfg.share_p || j == 0 {
                let mut q = std::mem::take(&mut self.qbuf);
                self.q_into(sigma2, &mut q);
                let p = &mut self.p[pi];
                let dd = d_clock * d_clock;
                for i in 0..k {
                    p[i * k + i] += q[i] * dd;
                }
                self.qbuf = q;
            }
            let Some(yj) = obs else {
                self.wj[j] *= lam;
                self.wsig[j] *= lam;
                continue;
            };
            // pz = P z
            {
                let p = &self.p[pi];
                for i in 0..k {
                    let row = i * k;
                    let mut acc = 0.0;
                    for jj in 0..k {
                        acc += p[row + jj] * self.zs[jj];
                    }
                    self.pz[i] = acc;
                }
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
                // Once per pair, written to both halves, so P stays symmetric
                // (see `Rls::step` for why that matters).
                let p = &mut self.p[pi];
                for i in 0..k {
                    let gi = self.gain[i];
                    for jj in i..k {
                        let v = gi * self.pz[jj];
                        p[i * k + jj] -= v;
                        if jj != i {
                            p[jj * k + i] -= v;
                        }
                    }
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
            if pred[j].is_finite() {
                let resid = yj - pred[j];
                let ws_new = aged + weight;
                let s2 = (aged * self.sig2[j] + weight * resid * resid) / ws_new;
                if s2.is_finite() {
                    self.sig2[j] = s2;
                    self.wsig[j] = ws_new;
                }
            }
            self.wj[j] = lam * self.wj[j] + weight;
        }

        // Standardization stats update last, so this row's z used the prior stats.
        self.stats.update(&self.zbuf, lam, weight);

        Step {
            pred,
            n_eff,
            extra: None,
        }
    }

    fn predict(&self, x: &[f64], d_clock: f64) -> Step {
        let m = self.cfg.n_targets;
        let n_eff = self.stats.n_eff();
        let mut pred = vec![f64::NAN; m];
        if n_eff >= self.cfg.min_weight {
            let zs = self.standardized(x);
            // The same numbers `step` would emit: its transition scales
            // `b_i` by `phi_i` before the dot product, and `b * phi` is
            // `phi * b` to the bit.
            let reverts = self.cfg.reverts();
            for (j, p) in pred.iter_mut().enumerate() {
                if self.wj[j] > 0.0 {
                    *p = a_number_or_none(if reverts {
                        zs.iter()
                            .zip(&self.beta[j])
                            .enumerate()
                            .map(|(i, (z, b))| z * (b * self.cfg.phi(i, d_clock)))
                            .sum()
                    } else {
                        zs.iter().zip(&self.beta[j]).map(|(z, b)| z * b).sum()
                    });
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
        bad(&|c| c.q = Some(vec![0.0, 0.0, -1e-9]), "must be >= 0");
        good(&|c| c.q = Some(vec![0.0; 3]));

        // Without `q`, the half-lives are broadcast: one value, or one per slot.
        bad(&|c| c.half_life = vec![1.0, 2.0], "length 1 or 3");
        bad(&|c| c.half_life = vec![0.0], "must be > 0");
        bad(&|c| c.half_life = vec![-1.0], "must be > 0");
        good(&|c| c.half_life = vec![f64::INFINITY]);
        good(&|c| c.half_life = vec![1.0, 2.0, 3.0]);

        // p0 is the prior variance and obs_var the measurement noise; both
        // divide, so neither may be zero. obs_var may be absent (inferred).
        bad(&|c| c.p0 = 0.0, "p0 must be > 0");
        bad(&|c| c.p0 = -1.0, "p0 must be > 0");
        bad(&|c| c.obs_var = Some(0.0), "obs_var must be > 0");
        bad(&|c| c.obs_var = Some(-1.0), "obs_var must be > 0");
        good(&|c| c.obs_var = None);
        good(&|c| c.obs_var = Some(1e-9));
        // NaN passed every `<= 0.0` / `< 0.0` test here. A NaN `obs_var` was
        // the silent one: `s_inn` is NaN on every row, the update's guard is
        // never met, and the filter predicts its prior for the life of the
        // stream with no error and no counted failure (review 2026-09-18, B4).
        bad(&|c| c.obs_var = Some(f64::NAN), "obs_var must be > 0");
        bad(&|c| c.p0 = f64::NAN, "p0 must be > 0");
        bad(&|c| c.half_life = vec![f64::NAN], "must be > 0");
        bad(&|c| c.q = Some(vec![0.0, f64::NAN, 0.0]), "must be >= 0");

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
        let z = m.standardized(&x);

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
        // Both targets read the same P, so their pred_var differs only through
        // ... nothing: z is shared too. They must be identical.
        let pv = ms.pred_var(&[0.3, -0.2]);
        assert!((pv[0] - pv[1]).abs() < 1e-12, "{pv:?}");
        // And that shared noise is the mean of the per-target residual
        // variances, which here differ by construction (target 1 is 2x target 0).
        let s2 = ms.sigma2();
        assert!(s2[1] > 2.0 * s2[0], "targets differ: {s2:?}");
        let mean = s2.iter().sum::<f64>() / 2.0;
        let quad = pv[0] - mean;
        assert!(
            quad > 0.0 && quad < mean,
            "R should be the mean: {pv:?} {s2:?}"
        );

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
        let beta = m.beta[0].clone();
        let (wj, wsig, sig2) = (m.wj[0], m.wsig[0], m.sig2[0]);

        let lam = 0.5f64.powf(4.0 / 10.0);
        m.step(&[0.3, -0.2], &[None], 4.0, 1.0);
        assert_eq!(m.beta[0], beta, "no target, no correction");
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
        let lam = 1.0 / p0;
        let (a, b, c, d) = (xtx[0][0] + lam, xtx[0][1], xtx[1][0], xtx[1][1] + lam);
        let det = a * d - b * c;
        let want = [
            (d * xty[0] - b * xty[1]) / det,
            (-c * xty[0] + a * xty[1]) / det,
        ];
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
        let before = m.beta[1].clone();
        let st = m.step(&[0.5], &[Some(1.0), None], 1.0, 1.0);
        assert!(st.pred[1].is_finite());
        assert_eq!(m.beta[1], before);
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

    /// The other half: a schema-3 state must not be mistaken for a schema-2
    /// one, and a map with neither `stats` nor `cov` is refused rather than
    /// defaulted.
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
    /// EW residual variance, the mean across targets under `share_p`; before
    /// there is one, the row's own innovation squared, under `share_p` the
    /// mean of the squares over the targets the row observes, and no noise
    /// (no `Q`, no correction) where there is none (CC4); `P` is unsized, 0,
    /// until a row has a noise, which sets it to `p0` times that noise, and
    /// after that takes `Q d²` once a row, before the first target's update;
    /// a target present at a positive weight corrects `b` and `P` by the gain
    /// `P z / (zᵀ P z + R / w)`; the residual variance is the EW mean of the
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
            let mut s = 103u64;
            for i in 0..150 {
                let x = [lcg(&mut s), 1.0 + lcg(&mut s)];
                let ys = [
                    Some(0.5 + x[0] - 2.0 * x[1] + 0.2 * lcg(&mut s)),
                    (i % 3 == 0).then(|| -x[0] + 3.0 * lcg(&mut s)),
                ];
                let w = 0.5 + (lcg(&mut s) + 1.0);
                let d = if i == 0 { 0.0 } else { 1.0 };
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
                for j in 0..2 {
                    let pi = if share { 0 } else { j };
                    let s2 = if share {
                        (sig2[0] + sig2[1]) / 2.0
                    } else {
                        sig2[j]
                    };
                    let sigma2 = if s2 > 0.0 {
                        s2
                    } else if share {
                        shared_first
                    } else {
                        e2[j].unwrap_or(0.0)
                    };
                    if (!share || j == 0) && p[pi].iter().all(|v| *v == 0.0) {
                        for a in 0..k {
                            p[pi][a * k + a] = p0 * sigma2;
                        }
                    } else if !share || j == 0 {
                        let q = sigma2 * (std::f64::consts::LN_2 / hq).powi(2);
                        for a in 0..k {
                            p[pi][a * k + a] += q * d * d;
                        }
                    }
                    let Some(y) = ys[j] else {
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
                            for bb in 0..k {
                                p[pi][a * k + bb] -= pz[a] * pz[bb] / s_inn;
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
        let mut c = cfg(2, 1, vec![40.0]);
        c.obs_var = Some(0.25);
        c.p0 = 3.0;
        let m = Kalman::new(c).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(m.p[0][i * 3 + j], if i == j { 0.75 } else { 0.0 });
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
}
