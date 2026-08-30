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
//! Quantile (check loss at level tau), the IRLS weight of the check function:
//!
//! ```text
//! w_robust = tau / max(|r|, eps)        if r > 0
//!          = (1 - tau) / max(|r|, eps)  otherwise
//! ```
//!
//! Because the weights are per target, the `S` accumulator is per target here
//! (one [`EwCov`] each) — unlike [`crate::EwRidge`], which shares one.

use serde::{Deserialize, Serialize};

use crate::model::{ModelState, OnlineModel, State, StateError, Step, check_schema};
use crate::solve::solve_spd;
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

    /// EW count of observations under the raw row weights (see [`Self::w_raw`]).
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
                if let Some(sol) = self.solve_standardized(&a, &b, k, j) {
                    beta[j] = sol;
                    continue;
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

    fn solve_standardized(&mut self, a: &[f64], b: &[f64], k: usize, j: usize) -> Option<Vec<f64>> {
        let off = usize::from(self.cfg.add_intercept);
        let kf = k - off;
        let mean = |i: usize| self.cov[j].mean(i);
        let mut c = vec![0.0; kf * kf];
        for i in 0..kf {
            for jj in 0..kf {
                c[i * kf + jj] = a[(i + off) * k + (jj + off)] - mean(i + off) * mean(jj + off);
            }
        }
        let s: Vec<f64> = (0..kf).map(|i| c[i * kf + i].max(0.0).sqrt()).collect();
        let keep: Vec<usize> = (0..kf)
            .filter(|&i| c[i * kf + i] > 1e-10 * a[(i + off) * k + (i + off)].abs().max(1e-300))
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
        let n_eff = self.w_raw;
        let ready = n_eff >= self.cfg.min_periods && self.beta.is_some();
        let mut pred = vec![f64::NAN; m];
        if ready {
            let beta = self.beta.as_ref().unwrap();
            for (j, p) in pred.iter_mut().enumerate() {
                if self.wj[j] > 0.0 {
                    *p = self.zbuf.iter().zip(&beta[j]).map(|(z, b)| z * b).sum();
                }
            }
        }

        // ---- update, reweighting by the PRIOR residual ----
        for j in 0..m {
            let Some(yj) = y[j] else {
                self.cov[j].decay(lam);
                self.wj[j] *= lam;
                self.wsig[j] *= lam;
                continue;
            };
            let sigma = self.sig2[j].max(0.0).sqrt();
            let w_rob = if pred[j].is_finite() {
                self.robust_weight(yj - pred[j], sigma)
            } else {
                1.0
            };
            let w = weight * w_rob;
            if w <= 0.0 {
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
                let ws_new = lam * self.wsig[j] + weight;
                self.sig2[j] =
                    (lam * self.wsig[j] * self.sig2[j] + weight * resid * resid) / ws_new;
                self.wsig[j] = ws_new;
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

        Step {
            pred,
            coef: None,
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
            min_periods: 2.0,
            solve_every: 0.0,
            max_rows_between_solves: 1,
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
    fn rejects_bad_config() {
        assert!(Robust::new(cfg(1, 1, RobustLoss::Quantile { tau: 0.0 })).is_err());
        assert!(Robust::new(cfg(1, 1, RobustLoss::Quantile { tau: 1.0 })).is_err());
        assert!(Robust::new(cfg(1, 1, RobustLoss::Huber { delta: 0.0 })).is_err());
    }
}
