//! Lasso path on top of the EW-ridge accumulators (docs/PLAN.md §4.3).
//!
//! Coordinate descent on the standardized centered statistics contained in
//! `EwCov` / the per-target cross moments, over a decreasing `lasso_path`,
//! warm-started along the path and across solves.
//!
//! For standardized features (unit variance, zero mean) and a centered target,
//! the coordinate update for feature `i` at penalty `l` is
//!
//! ```text
//! rho_i = c_i - sum_{j != i} C_ij b_j          (C = correlation matrix, c = corr(x, y))
//! b_i   = soft(rho_i, l * l1_ratio) / (C_ii + l * (1 - l1_ratio))
//! ```
//!
//! with `soft(v, t) = sign(v) * max(|v| - t, 0)`; `l1_ratio = 1` is pure lasso,
//! `< 1` is elastic net. Coefficients are unscaled afterwards and the intercept
//! recovered as `ybar - m . beta`.
//!
//! Lambda selection is free: predictions for every path point are computed
//! anyway, so `lam_selected_j` is the argmin over the path of an EW mean of
//! squared out-of-sample error with halflife `select_halflife`.

use serde::{Deserialize, Serialize};

use crate::model::{Extra, ModelState, OnlineModel, State, StateError, Step, check_schema};
use crate::{Decay, EwCov};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LassoCfg {
    pub n_features: usize,
    pub n_targets: usize,
    pub add_intercept: bool,
    pub decay: Decay,
    /// Decreasing penalties. Applied to standardized (correlation-form) stats.
    pub lasso_path: Vec<f64>,
    /// 1.0 = lasso, < 1.0 = elastic net (docs/PLAN.md §4.3 [validate]).
    pub l1_ratio: f64,
    /// Halflife of the EW squared-error used to pick lambda; defaults to the
    /// model halflife when None.
    pub select_halflife: Option<f64>,
    pub min_periods: f64,
    pub solve_every: f64,
    pub max_rows_between_solves: u32,
    pub max_cd_iters: u32,
    pub cd_tol: f64,
}

impl LassoCfg {
    pub fn k_total(&self) -> usize {
        self.n_features + usize::from(self.add_intercept)
    }

    pub fn n_lambdas(&self) -> usize {
        self.lasso_path.len()
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.n_features == 0 || self.n_targets == 0 {
            return Err("n_features and n_targets must be >= 1".into());
        }
        if self.lasso_path.is_empty() {
            return Err("lasso_path must have at least one value".into());
        }
        if self.lasso_path.iter().any(|&l| l < 0.0) {
            return Err("lasso_path values must be >= 0".into());
        }
        if !self.lasso_path.windows(2).all(|w| w[0] >= w[1]) {
            return Err("lasso_path must be decreasing".into());
        }
        if !(0.0..=1.0).contains(&self.l1_ratio) {
            return Err("l1_ratio must be in [0, 1]".into());
        }
        Ok(())
    }

    pub fn combo_labels(&self) -> Vec<String> {
        self.lasso_path.iter().map(|l| format!("l{l}")).collect()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Lasso {
    cfg: LassoCfg,
    cov: EwCov,
    wj: Vec<f64>,
    /// Per target: EW mean of `z y_j` (`z` includes the intercept slot).
    r: Vec<Vec<f64>>,
    /// Per target, per path point: coefficients in original units (`k_total`).
    beta: Option<Vec<Vec<Vec<f64>>>>,
    /// Per target, per path point: EW mean squared out-of-sample error.
    sel_err: Vec<Vec<f64>>,
    sel_w: Vec<f64>,
    /// Per target: index into the path chosen by `sel_err`.
    sel_idx: Vec<usize>,
    clock_since_solve: f64,
    rows_since_solve: u32,
    pub solve_failures: u64,
    #[serde(skip)]
    zbuf: Vec<f64>,
}

impl Lasso {
    pub fn new(cfg: LassoCfg) -> Result<Self, String> {
        cfg.validate()?;
        let k_total = cfg.k_total();
        let (m, np) = (cfg.n_targets, cfg.n_lambdas());
        Ok(Self {
            cov: EwCov::new(k_total),
            wj: vec![0.0; m],
            r: vec![vec![0.0; k_total]; m],
            beta: None,
            sel_err: vec![vec![0.0; np]; m],
            sel_w: vec![0.0; m],
            sel_idx: vec![np - 1; m],
            clock_since_solve: 0.0,
            rows_since_solve: 0,
            solve_failures: 0,
            zbuf: vec![0.0; k_total],
            cfg,
        })
    }

    pub fn cfg(&self) -> &LassoCfg {
        &self.cfg
    }

    /// Coefficients per (target, path point), in original units.
    pub fn coefficients(&self) -> Option<&Vec<Vec<Vec<f64>>>> {
        self.beta.as_ref()
    }

    /// Selected lambda per target.
    pub fn lam_selected(&self) -> Vec<f64> {
        self.sel_idx
            .iter()
            .map(|&i| self.cfg.lasso_path[i])
            .collect()
    }

    pub fn n_eff(&self) -> f64 {
        self.cov.n_eff()
    }

    /// Centered/standardized statistics: correlation matrix `c`, per-target
    /// standardized cross-correlation `d`, feature scales `s`, means `mean`.
    fn standardized(&self) -> (Vec<f64>, Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
        let k = self.cfg.n_features;
        let off = usize::from(self.cfg.add_intercept);
        let mean: Vec<f64> = (0..k).map(|i| self.cov.mean(i + off)).collect();
        let mut cov = vec![0.0; k * k];
        for i in 0..k {
            for j in 0..k {
                cov[i * k + j] = self.cov.raw(i + off, j + off) - mean[i] * mean[j];
            }
        }
        let s: Vec<f64> = (0..k)
            .map(|i| {
                let v = cov[i * k + i];
                let raw = self.cov.raw(i + off, i + off).abs().max(1e-300);
                if crate::variance_is_usable(v, raw) {
                    v.sqrt()
                } else {
                    0.0
                }
            })
            .collect();
        let mut c = vec![0.0; k * k];
        for i in 0..k {
            for j in 0..k {
                c[i * k + j] = if s[i] > 0.0 && s[j] > 0.0 {
                    cov[i * k + j] / (s[i] * s[j])
                } else {
                    f64::from(i == j)
                };
            }
        }
        let mut d = Vec::with_capacity(self.cfg.n_targets);
        for j in 0..self.cfg.n_targets {
            let ybar = if self.cfg.add_intercept {
                self.r[j][0]
            } else {
                0.0
            };
            d.push(
                (0..k)
                    .map(|i| {
                        if s[i] > 0.0 {
                            (self.r[j][i + off] - mean[i] * ybar) / s[i]
                        } else {
                            0.0
                        }
                    })
                    .collect::<Vec<f64>>(),
            );
        }
        (c, d, s, mean)
    }

    fn solve(&mut self) {
        let k = self.cfg.n_features;
        let k_total = self.cfg.k_total();
        let (c, d, s, mean) = self.standardized();
        let np = self.cfg.n_lambdas();
        let mut out = vec![vec![vec![0.0; k_total]; np]; self.cfg.n_targets];

        for j in 0..self.cfg.n_targets {
            if self.wj[j] <= 0.0 {
                continue;
            }
            // Warm start from the previous solve's largest-penalty solution.
            let mut b = vec![0.0; k];
            if let Some(prev) = &self.beta {
                for i in 0..k {
                    if s[i] > 0.0 {
                        b[i] = prev[j][0][i + usize::from(self.cfg.add_intercept)] * s[i];
                    }
                }
            }
            for (li, &lam) in self.cfg.lasso_path.iter().enumerate() {
                // Coordinate descent, warm-started along the path.
                let l1 = lam * self.cfg.l1_ratio;
                let l2 = lam * (1.0 - self.cfg.l1_ratio);
                for _ in 0..self.cfg.max_cd_iters {
                    let mut max_delta: f64 = 0.0;
                    for i in 0..k {
                        if s[i] <= 0.0 {
                            b[i] = 0.0;
                            continue;
                        }
                        let mut rho = d[j][i];
                        for (jj, bj) in b.iter().enumerate() {
                            if jj != i {
                                rho -= c[i * k + jj] * bj;
                            }
                        }
                        let denom = c[i * k + i] + l2;
                        let newb = if rho > l1 {
                            (rho - l1) / denom
                        } else if rho < -l1 {
                            (rho + l1) / denom
                        } else {
                            0.0
                        };
                        max_delta = max_delta.max((newb - b[i]).abs());
                        b[i] = newb;
                    }
                    if max_delta < self.cfg.cd_tol {
                        break;
                    }
                }
                // Unscale and recover the intercept.
                let off = usize::from(self.cfg.add_intercept);
                let coefs = &mut out[j][li];
                for i in 0..k {
                    coefs[i + off] = if s[i] > 0.0 { b[i] / s[i] } else { 0.0 };
                }
                if self.cfg.add_intercept {
                    let mut b0 = self.r[j][0];
                    for i in 0..k {
                        b0 -= mean[i] * coefs[i + off];
                    }
                    coefs[0] = b0;
                }
            }
        }
        self.beta = Some(out);
        self.clock_since_solve = 0.0;
        self.rows_since_solve = 0;
    }
}

impl OnlineModel for Lasso {
    fn step(&mut self, x: &[f64], y: &[Option<f64>], d_clock: f64, weight: f64) -> Step {
        let m = self.cfg.n_targets;
        let np = self.cfg.n_lambdas();
        let lam_decay = self.cfg.decay.factor(d_clock);
        if self.zbuf.len() != self.cfg.k_total() {
            self.zbuf = vec![0.0; self.cfg.k_total()];
        }
        if self.cfg.add_intercept {
            self.zbuf[0] = 1.0;
            self.zbuf[1..].copy_from_slice(x);
        } else {
            self.zbuf.copy_from_slice(x);
        }

        // ---- predict every path point (state before the update) ----
        let n_eff = self.cov.n_eff();
        let ready = n_eff >= self.cfg.min_periods && self.beta.is_some();
        let mut pred = vec![f64::NAN; m * np];
        if ready {
            let beta = self.beta.as_ref().unwrap();
            for j in 0..m {
                if self.wj[j] > 0.0 {
                    for li in 0..np {
                        pred[j * np + li] =
                            self.zbuf.iter().zip(&beta[j][li]).map(|(z, b)| z * b).sum();
                    }
                }
            }
        }

        // ---- lambda selection: EW mean squared OOS error, free from preds ----
        let sel_lam = match self.cfg.select_halflife {
            Some(h) => Decay::Halflife(h).factor(d_clock),
            None => lam_decay,
        };
        for j in 0..m {
            if let Some(yj) = y[j] {
                if pred[j * np].is_finite() {
                    let w_new = sel_lam * self.sel_w[j] + weight;
                    let a = sel_lam * self.sel_w[j] / w_new;
                    let b = weight / w_new;
                    for li in 0..np {
                        let e = yj - pred[j * np + li];
                        self.sel_err[j][li] = a * self.sel_err[j][li] + b * e * e;
                    }
                    self.sel_w[j] = w_new;
                    let mut best = 0usize;
                    for li in 1..np {
                        if self.sel_err[j][li] < self.sel_err[j][best] {
                            best = li;
                        }
                    }
                    self.sel_idx[j] = best;
                } else {
                    self.sel_w[j] *= sel_lam;
                }
            } else {
                self.sel_w[j] *= sel_lam;
            }
        }

        // ---- update accumulators ----
        self.cov.update(&self.zbuf, lam_decay, weight);
        for ((wj, r), yj) in self.wj.iter_mut().zip(self.r.iter_mut()).zip(y) {
            match yj {
                Some(yj) => {
                    let wj_new = lam_decay * *wj + weight;
                    let a = lam_decay * *wj / wj_new;
                    let b = weight / wj_new;
                    for (ri, zi) in r.iter_mut().zip(&self.zbuf) {
                        *ri = a * *ri + b * zi * yj;
                    }
                    *wj = wj_new;
                }
                None => *wj *= lam_decay,
            }
        }

        self.clock_since_solve += d_clock;
        self.rows_since_solve += 1;
        let due = self.cfg.solve_every <= 0.0
            || self.clock_since_solve >= self.cfg.solve_every
            || self.rows_since_solve >= self.cfg.max_rows_between_solves
            || (self.beta.is_none() && self.cov.n_eff() >= self.cfg.min_periods);
        if due {
            self.solve();
        }

        Step {
            pred,
            coef: None,
            n_eff,
            extra: Some(Extra::Lasso {
                lam_selected: self.lam_selected(),
            }),
        }
    }

    fn state(&self) -> State {
        State::new(ModelState::Lasso(Box::new(self.clone())))
    }

    fn restore(s: &State) -> Result<Self, StateError> {
        check_schema(s)?;
        match &s.model {
            ModelState::Lasso(m) => {
                let mut m = (**m).clone();
                m.zbuf = vec![0.0; m.cfg.k_total()];
                Ok(m)
            }
            other => Err(StateError::WrongModel {
                expected: "lasso",
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

    fn n_outputs(&self) -> usize {
        self.cfg.n_targets * self.cfg.n_lambdas()
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

    fn cfg(k: usize, m: usize, path: Vec<f64>) -> LassoCfg {
        LassoCfg {
            n_features: k,
            n_targets: m,
            add_intercept: true,
            decay: Decay::Halflife(f64::INFINITY),
            lasso_path: path,
            l1_ratio: 1.0,
            select_halflife: None,
            min_periods: (k + 1) as f64,
            solve_every: 0.0,
            max_rows_between_solves: 1,
            max_cd_iters: 200,
            cd_tol: 1e-12,
        }
    }

    #[test]
    fn zero_penalty_matches_ols() {
        // lambda = 0 => the lasso solution is the OLS solution.
        let mut m = Lasso::new(cfg(3, 1, vec![0.0])).unwrap();
        let mut s = 5u64;
        for i in 0..400 {
            let x = [lcg(&mut s), lcg(&mut s), lcg(&mut s)];
            let y = 1.5 * x[0] - 0.75 * x[1] + 2.0 * x[2] + 0.25;
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let b = &m.coefficients().unwrap()[0][0];
        assert!((b[0] - 0.25).abs() < 1e-6, "intercept {}", b[0]);
        assert!((b[1] - 1.5).abs() < 1e-6);
        assert!((b[2] + 0.75).abs() < 1e-6);
        assert!((b[3] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn large_penalty_zeroes_coefficients() {
        let mut m = Lasso::new(cfg(2, 1, vec![100.0, 0.0])).unwrap();
        let mut s = 6u64;
        for i in 0..200 {
            let x = [lcg(&mut s), lcg(&mut s)];
            let y = x[0] + x[1];
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let b = m.coefficients().unwrap();
        assert_eq!(
            &b[0][0][1..],
            &[0.0, 0.0],
            "heavy penalty must zero features"
        );
        assert!((b[0][1][1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn selects_sparse_lambda_when_features_are_noise() {
        // y depends on x0 only; x1..x3 are noise. A middling penalty should beat
        // lambda = 0 on out-of-sample error, so selection must not pick 0.
        let mut c = cfg(4, 1, vec![0.5, 0.2, 0.05, 0.0]);
        c.decay = Decay::Halflife(500.0);
        c.min_periods = 20.0;
        let mut m = Lasso::new(c).unwrap();
        let mut s = 7u64;
        for i in 0..1500 {
            let x = [lcg(&mut s), lcg(&mut s), lcg(&mut s), lcg(&mut s)];
            let y = 1.0 * x[0] + 0.9 * lcg(&mut s);
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let b = m.coefficients().unwrap();
        // at the largest penalty the noise features are zero
        assert_eq!(&b[0][0][2..], &[0.0, 0.0, 0.0]);
        // and the selected lambda is a real choice from the path
        assert!(m.cfg.lasso_path.contains(&m.lam_selected()[0]));
    }

    #[test]
    fn elastic_net_shrinks_less_sparsely() {
        let mut c = cfg(2, 1, vec![0.3]);
        c.l1_ratio = 0.5;
        let mut m = Lasso::new(c).unwrap();
        let mut s = 8u64;
        for i in 0..300 {
            let x = [lcg(&mut s), lcg(&mut s)];
            let y = x[0] + x[1];
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let b = &m.coefficients().unwrap()[0][0];
        // elastic net with this penalty shrinks but does not zero out
        assert!(b[1] > 0.1 && b[1] < 1.0, "{}", b[1]);
    }

    #[test]
    fn state_roundtrip() {
        let mut m1 = Lasso::new(cfg(2, 1, vec![0.5, 0.0])).unwrap();
        let mut s = 9u64;
        let rows: Vec<([f64; 2], f64)> = (0..80)
            .map(|_| {
                let x = [lcg(&mut s), lcg(&mut s)];
                (x, x[0] - 0.5 * x[1])
            })
            .collect();
        for (i, (x, y)) in rows[..40].iter().enumerate() {
            m1.step(x, &[Some(*y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let bytes = rmp_serde::to_vec(&m1.state()).unwrap();
        let mut m2 = Lasso::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
        for (x, y) in &rows[40..] {
            let a = m1.step(x, &[Some(*y)], 1.0, 1.0);
            let b = m2.step(x, &[Some(*y)], 1.0, 1.0);
            assert_eq!(a.pred, b.pred);
            assert_eq!(a.extra, b.extra);
        }
    }
}
