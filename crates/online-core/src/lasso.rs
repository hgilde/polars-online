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
        // Centered co-moments come straight from the accumulator; deriving them
        // as raw - mean*mean would reintroduce the cancellation the Welford
        // representation exists to avoid.
        let mut cov = vec![0.0; k * k];
        for i in 0..k {
            for j in 0..k {
                cov[i * k + j] = self.cov.cov(i + off, j + off);
            }
        }
        let s: Vec<f64> = (0..k)
            .map(|i| {
                let v = cov[i * k + i];
                if crate::variance_is_usable(v, self.cov.raw(i + off, i + off)) {
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
                // See `EwRidge`'s copy of this update: `wj_new == 0` is a
                // zero-weight row before any weighted one, where `a` and `b`
                // are both 0/0 and the NaN would never wash out.
                Some(yj) if lam_decay * *wj + weight > 0.0 => {
                    let wj_new = lam_decay * *wj + weight;
                    let a = lam_decay * *wj / wj_new;
                    let b = weight / wj_new;
                    for (ri, zi) in r.iter_mut().zip(&self.zbuf) {
                        *ri = a * *ri + b * zi * yj;
                    }
                    *wj = wj_new;
                }
                Some(_) => {}
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

    /// Feed a deterministic stream with `k` informative features.
    fn fit(cfg: LassoCfg, n: usize, seed: u64) -> (Lasso, Vec<(Vec<f64>, f64)>) {
        let k = cfg.n_features;
        let mut m = Lasso::new(cfg).unwrap();
        let mut s = seed;
        let mut rows = Vec::new();
        for i in 0..n {
            let x: Vec<f64> = (0..k).map(|_| lcg(&mut s)).collect();
            let y = 1.5 * x[0] - 0.75 * x[1] + 0.25 + 0.05 * lcg(&mut s);
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
            rows.push((x, y));
        }
        (m, rows)
    }

    #[test]
    fn every_path_point_satisfies_the_kkt_conditions() {
        // The coordinate descent is checked against the optimality conditions
        // of the problem it claims to solve, rather than against a golden
        // number: for the objective 1/2 b'Cb - d'b + l1|b|_1 + l2/2 |b|^2, at
        // the optimum the gradient g = Cb - d + l2 b satisfies g_i = -l1 sign(b_i)
        // on the active set and |g_i| <= l1 off it. This is the same check the
        // Python suite makes; having it here makes the solver's arithmetic
        // visible to `cargo test`, and so to mutation testing.
        for l1_ratio in [1.0, 0.5] {
            let mut c = cfg(4, 1, vec![0.5, 0.1, 0.01]);
            c.l1_ratio = l1_ratio;
            c.cd_tol = 1e-14;
            c.max_cd_iters = 2000;
            let (m, _) = fit(c.clone(), 500, 11);

            // Rebuild the standardized normal equations the solver works in.
            let k = c.n_features;
            let off = usize::from(c.add_intercept);
            let s: Vec<f64> = (0..k).map(|i| m.cov.cov(i + off, i + off).sqrt()).collect();
            let beta = m.coefficients().unwrap();
            for (li, &lam) in c.lasso_path.iter().enumerate() {
                let (l1, l2) = (lam * c.l1_ratio, lam * (1.0 - c.l1_ratio));
                // Back to the scaled parameterization the objective is in.
                let b: Vec<f64> = (0..k).map(|i| beta[0][li][i + off] * s[i]).collect();
                for i in 0..k {
                    let mut g = 0.0;
                    for jj in 0..k {
                        g += m.cov.cov(i + off, jj + off) / (s[i] * s[jj]) * b[jj];
                    }
                    let d_i = (m.r[0][i + off] - m.cov.mean(i + off) * m.r[0][0]) / s[i];
                    g -= d_i;
                    g += l2 * b[i];
                    if b[i].abs() > 1e-9 {
                        assert!(
                            (g + l1 * b[i].signum()).abs() < 1e-6,
                            "lam {lam} ratio {l1_ratio} coef {i}: active KKT {g}"
                        );
                    } else {
                        assert!(
                            g.abs() <= l1 + 1e-6,
                            "lam {lam} ratio {l1_ratio} coef {i}: inactive KKT {g} > {l1}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn the_path_is_monotone_in_sparsity_and_shrinkage() {
        // A larger penalty can only zero more coefficients and shrink the rest;
        // the path must be ordered, which pins the direction of every
        // soft-threshold comparison.
        let c = cfg(6, 1, vec![1.0, 0.3, 0.1, 0.03, 0.0]);
        let (m, _) = fit(c.clone(), 600, 13);
        let beta = &m.coefficients().unwrap()[0];
        let nnz: Vec<usize> = beta
            .iter()
            .map(|b| b[1..].iter().filter(|v| v.abs() > 1e-9).count())
            .collect();
        for w in nnz.windows(2) {
            assert!(
                w[0] <= w[1],
                "sparsity should relax along the path: {nnz:?}"
            );
        }
        assert!(
            nnz[0] < nnz[nnz.len() - 1],
            "the path must do something: {nnz:?}"
        );

        // The strongest signal's magnitude grows as the penalty falls.
        let mags: Vec<f64> = beta.iter().map(|b| b[1].abs()).collect();
        for w in mags.windows(2) {
            assert!(w[0] <= w[1] + 1e-9, "shrinkage should relax: {mags:?}");
        }
        assert!((mags[mags.len() - 1] - 1.5).abs() < 0.05, "{mags:?}");
    }

    #[test]
    fn the_intercept_is_recovered_from_the_centered_fit() {
        // The features are centered before the solve, so the intercept is
        // reconstructed as mean(y) - sum(b_i mean(x_i)) rather than fitted.
        let mut c = cfg(2, 1, vec![0.0]);
        c.min_periods = 3.0;
        let (m, rows) = fit(c, 500, 17);
        let b = &m.coefficients().unwrap()[0][0];
        let n = rows.len() as f64;
        let ybar: f64 = rows.iter().map(|(_, y)| y).sum::<f64>() / n;
        let xbar: Vec<f64> = (0..2)
            .map(|i| rows.iter().map(|(x, _)| x[i]).sum::<f64>() / n)
            .collect();
        let want = ybar - b[1] * xbar[0] - b[2] * xbar[1];
        assert!((b[0] - want).abs() < 1e-6, "{} vs {want}", b[0]);
    }

    #[test]
    fn lambda_selection_tracks_the_out_of_sample_error() {
        // `sel_err` is an EW mean of each path point's squared OOS error and
        // `lam_selected` reports the argmin. Both are checked against the
        // predictions the model itself emitted, so a selection reading the
        // wrong slot, or the wrong direction of the comparison, is caught.
        let mut c = cfg(3, 1, vec![1.0, 0.05, 0.0]);
        c.min_periods = 4.0;
        c.select_halflife = Some(f64::INFINITY);
        let np = c.n_lambdas();
        let mut m = Lasso::new(c).unwrap();

        let mut sums = vec![0.0; np];
        let mut count = 0.0;
        let mut s = 19u64;
        for i in 0..400 {
            let x = [lcg(&mut s), lcg(&mut s), lcg(&mut s)];
            let y = 2.0 * x[0] + 0.02 * lcg(&mut s);
            let step = m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
            if step.pred[0].is_finite() {
                count += 1.0;
                for (li, sum) in sums.iter_mut().enumerate() {
                    *sum += (y - step.pred[li]).powi(2);
                }
            }
        }
        assert!(count > 100.0);
        // An infinite select_halflife makes the EW mean a plain mean.
        for (li, sum) in sums.iter().enumerate() {
            assert!(
                (m.sel_err[0][li] - sum / count).abs() < 1e-9,
                "path point {li}: {} vs {}",
                m.sel_err[0][li],
                sum / count
            );
        }
        let best = (0..np)
            .min_by(|&a, &b| sums[a].partial_cmp(&sums[b]).unwrap())
            .unwrap();
        assert_eq!(m.sel_idx[0], best, "selection must be the argmin: {sums:?}");
        assert_eq!(m.lam_selected(), vec![m.cfg.lasso_path[best]]);
        // Only one feature matters, so the heaviest penalty must not win.
        assert!(
            best > 0,
            "a penalty of 1.0 should not be selected: {sums:?}"
        );
    }

    #[test]
    fn a_null_target_and_a_zero_weight_row_only_decay() {
        let mut c = cfg(2, 1, vec![0.0]);
        c.decay = Decay::Halflife(10.0);
        c.min_periods = 3.0;
        let (mut m, _) = fit(c, 60, 23);
        let (wj, r, sel_w) = (m.wj[0], m.r[0].clone(), m.sel_w[0]);

        let lam = 0.5f64.powf(2.0 / 10.0);
        m.step(&[0.5, -0.5], &[None], 2.0, 1.0);
        assert!((m.wj[0] - wj * lam).abs() < 1e-12, "null decays wj");
        assert_eq!(m.r[0], r, "and leaves the cross-moments alone");
        assert!((m.sel_w[0] - sel_w * lam).abs() < 1e-12);
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
