//! Recursive least squares (docs/PLAN.md §4.2).
//!
//! Maintains `P = (sum of decayed w z z^T + lam_acc * ridge I)^-1` directly via
//! the Sherman-Morrison update, so coefficients move every row with zero solve
//! staleness:
//!
//! ```text
//! P    <- P / lam                       (decay: A <- lam A  =>  P <- P / lam)
//! g    = P z / (1/w + z^T P z)          (gain)
//! beta_j <- beta_j + g (y_j - z^T beta_j)
//! P    <- P - g z^T P
//! ```
//!
//! `P0 = I / ridge` (i.e. `A0 = ridge I`), so `ridge` is the classic RLS prior
//! strength and the intercept is penalized too. This is exactly the `ridge_decay`
//! mode of [`crate::EwRidge`], which is what the agreement test exploits.
//!
//! Null policy deviation, documented: a row with ANY null target is predict-only
//! for all targets, because P is shared across targets and a per-target update
//! would desynchronize it.

use serde::{Deserialize, Serialize};

use crate::Decay;
use crate::model::{ModelState, OnlineModel, State, StateError, Step, check_schema};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RlsCfg {
    pub n_features: usize,
    pub n_targets: usize,
    pub add_intercept: bool,
    pub decay: Decay,
    /// Prior strength: `P0 = I / ridge`.
    pub ridge: f64,
    /// Initial coefficients per target (length `k_total`), default zeros.
    pub coef0: Option<Vec<Vec<f64>>>,
    pub min_periods: f64,
}

impl RlsCfg {
    pub fn k_total(&self) -> usize {
        self.n_features + usize::from(self.add_intercept)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.n_features == 0 || self.n_targets == 0 {
            return Err("n_features and n_targets must be >= 1".into());
        }
        if self.ridge <= 0.0 || self.ridge.is_nan() {
            return Err("rls: ridge must be > 0 (it sets P0 = I / ridge)".into());
        }
        if let Some(c) = &self.coef0 {
            if c.len() != self.n_targets || c.iter().any(|v| v.len() != self.k_total()) {
                return Err("rls: coef0 must be n_targets x k_total".into());
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Rls {
    cfg: RlsCfg,
    /// Inverse information matrix, row-major `k*k`.
    p: Vec<f64>,
    /// Coefficients per target, each `k_total`.
    beta: Vec<Vec<f64>>,
    w_sum: f64,
    seen: bool,
    #[serde(skip)]
    zbuf: Vec<f64>,
    #[serde(skip)]
    pz: Vec<f64>,
    #[serde(skip)]
    gain: Vec<f64>,
}

impl Rls {
    pub fn new(cfg: RlsCfg) -> Result<Self, String> {
        cfg.validate()?;
        let k = cfg.k_total();
        let mut p = vec![0.0; k * k];
        for i in 0..k {
            p[i * k + i] = 1.0 / cfg.ridge;
        }
        let beta = cfg
            .coef0
            .clone()
            .unwrap_or_else(|| vec![vec![0.0; k]; cfg.n_targets]);
        Ok(Self {
            p,
            beta,
            w_sum: 0.0,
            seen: false,
            zbuf: vec![0.0; k],
            pz: vec![0.0; k],
            gain: vec![0.0; k],
            cfg,
        })
    }

    pub fn coefficients(&self) -> &[Vec<f64>] {
        &self.beta
    }

    pub fn n_eff(&self) -> f64 {
        self.w_sum
    }

    fn ensure_buffers(&mut self) {
        let k = self.cfg.k_total();
        if self.zbuf.len() != k {
            self.zbuf = vec![0.0; k];
            self.pz = vec![0.0; k];
            self.gain = vec![0.0; k];
        }
    }
}

impl OnlineModel for Rls {
    fn step(&mut self, x: &[f64], y: &[Option<f64>], d_clock: f64, weight: f64) -> Step {
        self.ensure_buffers();
        let k = self.cfg.k_total();
        let m = self.cfg.n_targets;
        let lam = self.cfg.decay.factor(d_clock);

        if self.cfg.add_intercept {
            self.zbuf[0] = 1.0;
            self.zbuf[1..].copy_from_slice(x);
        } else {
            self.zbuf.copy_from_slice(x);
        }

        // ---- predict (state before the update) ----
        let n_eff = self.w_sum;
        let ready = n_eff >= self.cfg.min_periods && self.seen;
        let mut pred = vec![f64::NAN; m];
        if ready {
            for (p, beta) in pred.iter_mut().zip(&self.beta) {
                *p = self.zbuf.iter().zip(beta).map(|(z, b)| z * b).sum();
            }
        }

        // ---- decay ----
        // A <- lam A  =>  P <- P / lam. Skipped when lam == 1 (no-op).
        if lam != 1.0 && lam > 0.0 {
            let inv = 1.0 / lam;
            for v in self.p.iter_mut() {
                *v *= inv;
            }
        }
        self.w_sum = lam * self.w_sum + weight;

        // ---- update (only when every target is present) ----
        if weight > 0.0 && y.iter().all(Option::is_some) {
            // pz = P z
            for i in 0..k {
                let row = i * k;
                let mut acc = 0.0;
                for j in 0..k {
                    acc += self.p[row + j] * self.zbuf[j];
                }
                self.pz[i] = acc;
            }
            let zpz: f64 = self.zbuf.iter().zip(&self.pz).map(|(z, p)| z * p).sum();
            let denom = 1.0 / weight + zpz;
            if denom.abs() > 0.0 {
                for i in 0..k {
                    self.gain[i] = self.pz[i] / denom;
                }
                for (beta, yj) in self.beta.iter_mut().zip(y) {
                    let yj = yj.unwrap();
                    let pred_now: f64 = self.zbuf.iter().zip(beta.iter()).map(|(z, b)| z * b).sum();
                    let err = yj - pred_now;
                    for (b, g) in beta.iter_mut().zip(&self.gain) {
                        *b += g * err;
                    }
                }
                // P <- P - g (P z)^T   (symmetric rank-1 downdate)
                for i in 0..k {
                    let gi = self.gain[i];
                    let row = i * k;
                    for j in 0..k {
                        self.p[row + j] -= gi * self.pz[j];
                    }
                }
                self.seen = true;
            }
        }

        Step {
            pred,
            coef: None,
            n_eff,
            extra: None,
        }
    }

    fn state(&self) -> State {
        State::new(ModelState::Rls(Box::new(self.clone())))
    }

    fn restore(s: &State) -> Result<Self, StateError> {
        check_schema(s)?;
        match &s.model {
            ModelState::Rls(m) => {
                let mut m = (**m).clone();
                m.ensure_buffers();
                Ok(m)
            }
            other => Err(StateError::WrongModel {
                expected: "rls",
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

    fn rls_cfg(k: usize, m: usize, hl: f64, ridge: f64) -> RlsCfg {
        RlsCfg {
            n_features: k,
            n_targets: m,
            add_intercept: true,
            decay: Decay::Halflife(hl),
            ridge,
            coef0: None,
            min_periods: 0.0,
        }
    }

    /// docs/PLAN.md §9 class 1: RLS and EW-ridge with `solve_every` = 1 row and
    /// the matching decaying prior must agree to float precision.
    #[test]
    fn agrees_with_ewridge_solved_every_row() {
        let (k, hl, ridge) = (3usize, 40.0, 0.7);
        let mut rls = Rls::new(rls_cfg(k, 2, hl, ridge)).unwrap();
        let mut ew = EwRidge::new(EwRidgeCfg {
            n_features: k,
            n_targets: 2,
            add_intercept: true,
            decay: Decay::Halflife(hl),
            ridge: vec![ridge],
            feature_sets: vec![],
            standardize: false,
            ridge_decay: true,
            session_shrink: None,
            long_halflife: None,
            coef0: None,
            min_periods: 0.0,
            solve_every: 0.0,
            max_rows_between_solves: 1,
        })
        .unwrap();

        let mut s = 99u64;
        let mut max_diff: f64 = 0.0;
        for i in 0..400 {
            let x: Vec<f64> = (0..k).map(|_| lcg(&mut s)).collect();
            let y0 = 1.5 * x[0] - x[1] + 0.3 + 0.05 * lcg(&mut s);
            let y1 = 0.2 * x[2] + 0.01 * lcg(&mut s);
            let d = if i == 0 { 0.0 } else { 0.5 + lcg(&mut s).abs() };
            let w = 0.5 + lcg(&mut s).abs();
            let a = rls.step(&x, &[Some(y0), Some(y1)], d, w);
            let b = ew.step(&x, &[Some(y0), Some(y1)], d, w);
            if i > 5 {
                for j in 0..2 {
                    assert!(a.pred[j].is_finite() && b.pred[j].is_finite());
                    max_diff = max_diff.max((a.pred[j] - b.pred[j]).abs());
                }
            }
        }
        assert!(max_diff < 1e-9, "max pred difference {max_diff}");
    }

    #[test]
    fn recovers_static_beta() {
        let mut m = Rls::new(rls_cfg(2, 1, f64::INFINITY, 1e-6)).unwrap();
        let mut s = 21u64;
        for i in 0..300 {
            let x = [lcg(&mut s), lcg(&mut s)];
            let y = 2.0 * x[0] - 0.5 * x[1] + 1.0;
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let b = &m.coefficients()[0];
        assert!((b[0] - 1.0).abs() < 1e-6);
        assert!((b[1] - 2.0).abs() < 1e-6);
        assert!((b[2] + 0.5).abs() < 1e-6);
    }

    #[test]
    fn null_target_is_predict_only() {
        let mut m = Rls::new(rls_cfg(1, 2, 100.0, 1.0)).unwrap();
        let mut s = 31u64;
        for i in 0..40 {
            let x = [lcg(&mut s)];
            m.step(
                &x,
                &[Some(x[0]), Some(-x[0])],
                if i == 0 { 0.0 } else { 1.0 },
                1.0,
            );
        }
        let before = m.beta.clone();
        let st = m.step(&[0.5], &[Some(1.0), None], 1.0, 1.0);
        assert!(st.pred.iter().all(|p| p.is_finite()));
        assert_eq!(
            m.beta, before,
            "a null target must not update any coefficient"
        );
    }

    #[test]
    fn state_roundtrip() {
        let mut m1 = Rls::new(rls_cfg(2, 1, 50.0, 0.5)).unwrap();
        let mut s = 41u64;
        let rows: Vec<([f64; 2], f64)> = (0..80)
            .map(|_| {
                let x = [lcg(&mut s), lcg(&mut s)];
                (x, x[0] - x[1])
            })
            .collect();
        for (i, (x, y)) in rows[..40].iter().enumerate() {
            m1.step(x, &[Some(*y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let bytes = rmp_serde::to_vec(&m1.state()).unwrap();
        let mut m2 = Rls::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
        for (x, y) in &rows[40..] {
            assert_eq!(
                m1.step(x, &[Some(*y)], 1.0, 1.0).pred,
                m2.step(x, &[Some(*y)], 1.0, 1.0).pred
            );
        }
    }
}
