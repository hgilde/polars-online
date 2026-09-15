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
//! Quantile (check loss at level tau), the IRLS weight of the check function,
//! with `eps = quantile_eps` flooring `|r|` in units of `s` so a near-zero
//! residual cannot produce an unbounded weight, and the whole thing scaled by
//! `s` so the weights are O(1) rather than O(1/s):
//!
//! ```text
//! w_robust = 2 * tau       * s / max(|r|, eps * s)    if r > 0
//!          = 2 * (1 - tau) * s / max(|r|, eps * s)    otherwise
//! ```
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
    /// Floor on |residual| in the quantile weight, in units of the EW residual
    /// std, so a near-zero residual cannot produce an unbounded weight.
    pub quantile_eps: f64,
}

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
    wj: Vec<f64>,
    r: Vec<Vec<f64>>,
    /// EW residual variance per target (drives the robust scale).
    sig2: Vec<f64>,
    wsig: Vec<f64>,
    /// EW count of *observations* using the raw row weights, i.e. ignoring the
    /// IRLS reweighting. This is what `n_eff` and `min_periods` mean everywhere
    /// else, so the robust models report it too: the accumulators are scaled by
    /// the robust weights, but the observation count must not be. (Quantile
    /// weights can reach `2 / quantile_eps`, so counting them would inflate
    /// `n_eff` by ~1000x and make `min_periods` meaningless.)
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
            r: vec![vec![0.0; k]; m],
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

    /// Robust weight multiplier for a prior residual (docs/PLAN.md §4.5).
    fn robust_weight(&self, resid: f64, sigma: f64) -> f64 {
        let s = if sigma > 0.0 { sigma } else { 1.0 };
        match self.cfg.loss {
            RobustLoss::Huber { delta } => {
                let cut = delta * s;
                let a = resid.abs();
                if a <= cut || a == 0.0 { 1.0 } else { cut / a }
            }
            RobustLoss::Quantile { tau } => {
                let floor = self.cfg.quantile_eps * s;
                let a = resid.abs().max(floor);
                let side = if resid > 0.0 { tau } else { 1.0 - tau };
                // Scaled by s so the weights are O(1) rather than O(1/s).
                2.0 * side * s / a
            }
        }
    }

    fn solve(&mut self) {
        let k = self.cfg.k_total();
        let off = usize::from(self.cfg.add_intercept);
        let mut beta = vec![vec![0.0; k]; self.cfg.n_targets];
        for j in 0..self.cfg.n_targets {
            if self.wj[j] <= 0.0 {
                continue;
            }
            let mut a = vec![0.0; k * k];
            for i in 0..k {
                for jj in 0..k {
                    a[i * k + jj] = self.cov[j].raw(i, jj);
                }
            }
            let b: Vec<f64> = self.r[j].clone();
            if self.cfg.standardize {
                // Same scheme as EwRidge::solve_standardized, single target.
                // `None` is a solve that failed at every jitter: counted, as
                // the plain one is below, and the previous fit kept. It
                // returned through `?` before the count (review 2026-09-12,
                // S13).
                match self.solve_standardized(&b, k, j) {
                    Some(sol) => {
                        beta[j] = sol;
                        continue;
                    }
                    None => self.solve_failures += 1,
                }
            } else {
                for i in off..k {
                    a[i * k + i] += self.cfg.ridge;
                }
                match solve_spd(&a, &b, k, 1) {
                    Some((x, jit)) => {
                        self.solve_failures += u64::from(jit);
                        beta[j] = x;
                        continue;
                    }
                    None => self.solve_failures += 1,
                }
            }
            if let Some(prev) = &self.beta {
                beta[j] = prev[j].clone();
            }
        }
        self.beta = Some(beta);
        self.clock_since_solve = 0.0;
        self.rows_since_solve = 0;
    }

    /// Centered statistics are read from this target's accumulator directly
    /// rather than re-derived from raw moments (see `EwCov`'s module docs).
    /// `None` when every jitter failed; the caller counts it.
    fn solve_standardized(&mut self, b: &[f64], k: usize, j: usize) -> Option<Vec<f64>> {
        let off = usize::from(self.cfg.add_intercept);
        if off == 0 {
            // No intercept: scale by the raw second moment and solve the raw
            // normal equations, as `EwRidge`'s no-intercept branch does. This
            // centred the Gram and kept the raw right-hand side -- the hybrid
            // system C8 found in `lasso`, least squares only when every
            // feature has mean zero (review 2026-09-12, C11).
            let s: Vec<f64> = (0..k)
                .map(|i| self.cov[j].raw(i, i).max(0.0).sqrt())
                .collect();
            let keep: Vec<usize> = (0..k).filter(|&i| s[i] > 0.0).collect();
            let kk = keep.len();
            let mut out = vec![0.0; k];
            if kk > 0 {
                let mut asub = vec![0.0; kk * kk];
                for (i2, &i) in keep.iter().enumerate() {
                    for (j2, &jj) in keep.iter().enumerate() {
                        asub[i2 * kk + j2] = self.cov[j].raw(i, jj) / (s[i] * s[jj]);
                    }
                    asub[i2 * kk + i2] += self.cfg.ridge;
                }
                let bsub: Vec<f64> = keep.iter().map(|&i| b[i] / s[i]).collect();
                let (sol, jit) = solve_spd(&asub, &bsub, kk, 1)?;
                self.solve_failures += u64::from(jit);
                for (i2, &i) in keep.iter().enumerate() {
                    out[i] = sol[i2] / s[i];
                }
            }
            return Some(out);
        }
        let kf = k - off;
        // Materialized up front: the solve below borrows `self` mutably.
        let means: Vec<f64> = (0..k).map(|i| self.cov[j].mean(i)).collect();
        let mean = |i: usize| means[i];
        let mut c = vec![0.0; kf * kf];
        for i in 0..kf {
            for jj in 0..kf {
                c[i * kf + jj] = self.cov[j].cov(i + off, jj + off);
            }
        }
        let s: Vec<f64> = (0..kf).map(|i| c[i * kf + i].max(0.0).sqrt()).collect();
        let raws: Vec<f64> = (0..kf).map(|i| self.cov[j].raw(i + off, i + off)).collect();
        let keep: Vec<usize> = (0..kf)
            .filter(|&i| crate::variance_is_usable(c[i * kf + i], raws[i]))
            .collect();
        let kk = keep.len();
        let mut out = vec![0.0; k];
        if kk > 0 {
            let mut asub = vec![0.0; kk * kk];
            for (i2, &i) in keep.iter().enumerate() {
                for (j2, &jj) in keep.iter().enumerate() {
                    asub[i2 * kk + j2] = c[i * kf + jj] / (s[i] * s[jj]);
                }
                asub[i2 * kk + i2] += self.cfg.ridge;
            }
            let ybar = if off == 1 { b[0] } else { 0.0 };
            let mut bsub = vec![0.0; kk];
            for (i2, &i) in keep.iter().enumerate() {
                bsub[i2] = (b[i + off] - mean(i + off) * ybar) / s[i];
            }
            let (sol, jit) = solve_spd(&asub, &bsub, kk, 1)?;
            self.solve_failures += u64::from(jit);
            for (i2, &i) in keep.iter().enumerate() {
                out[i + off] = sol[i2] / s[i];
            }
        }
        if off == 1 {
            let mut b0 = b[0];
            for i in 0..kf {
                b0 -= mean(i + off) * out[i + off];
            }
            out[0] = b0;
        }
        Some(out)
    }
}

impl OnlineModel for Robust {
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

        // ---- update, reweighting by the PRIOR residual ----
        for j in 0..m {
            // `σ²`'s weight ages on every row, as `wj` does, and a row adds to
            // it only with a target, a weight and a prediction to measure the
            // residual from. A zero or NaN weight skipped the ageing, and so
            // did a row with no prediction (`min_periods` unmet after a clock
            // gap), so `σ²` -- the scale of every cut -- forgot less across
            // either than across a null (review 2026-09-12, S13; N6).
            self.wsig[j] *= lam;
            let Some(yj) = y[j] else {
                self.cov[j].decay(lam);
                self.wj[j] *= lam;
                continue;
            };
            let sigma = self.sig2[j].max(0.0).sqrt();
            let w_rob = if pred[j].is_finite() {
                self.robust_weight(yj - pred[j], sigma)
            } else {
                1.0
            };
            let w = weight * w_rob;
            // NaN is `inf / inf` from an overflowed residual against an
            // overflowed scale; such a row cannot be learned from either.
            if w.is_nan() || w <= 0.0 {
                self.cov[j].decay(lam);
                self.wj[j] *= lam;
                continue;
            }
            self.cov[j].update(&self.zbuf, lam, w);
            let wj_new = lam * self.wj[j] + w;
            let a = lam * self.wj[j] / wj_new;
            let bb = w / wj_new;
            for (ri, zi) in self.r[j].iter_mut().zip(&self.zbuf) {
                *ri = a * *ri + bb * zi * yj;
            }
            self.wj[j] = wj_new;
            if pred[j].is_finite() {
                let resid = yj - pred[j];
                let ws_new = self.wsig[j] + weight;
                let s2 = (self.wsig[j] * self.sig2[j] + weight * resid * resid) / ws_new;
                // Skipped when it would not be finite: an `inf` scale makes
                // the Huber cut infinite (plain least squares for good) and
                // the quantile weight `inf / inf` (docs/IMPROVEMENTS.md C2).
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
        // The defect T-A5 found: the IRLS weights a quantile fit uses reach
        // `2 / quantile_eps`, so counting them made `n_eff` -- and therefore
        // `min_periods` -- meaningless. It must be the plain weighted
        // observation count, identical to every other model's.
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
