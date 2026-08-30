//! `EwCov`: exponentially weighted first/second moments of a vector stream
//! (docs/PLAN.md §4.7). The shared accumulator behind EW-ridge, RLS and Kalman,
//! and exposed on its own as `online.ew_cov()`.
//!
//! All statistics are stored as weighted *means* (not sums), which keeps them
//! bounded under arbitrarily long runs (docs/PLAN.md §7):
//!
//! ```text
//! W'    = lam * W + w
//! m'_i  = (lam * W * m_i + w * x_i) / W'
//! S'_ij = (lam * W * S_ij + w * x_i x_j) / W'
//! ```

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EwCov {
    k: usize,
    /// EW sum of weights (the `n_eff` count).
    w_sum: f64,
    /// Product of all decay factors applied so far (a decaying prior's scale).
    prior_scale: f64,
    /// EW mean vector, length `k`.
    m: Vec<f64>,
    /// EW mean of `x x^T`, row-major `k*k` (raw second moment, NOT centered).
    s: Vec<f64>,
}

impl EwCov {
    pub fn new(k: usize) -> Self {
        Self {
            k,
            w_sum: 0.0,
            prior_scale: 1.0,
            m: vec![0.0; k],
            s: vec![0.0; k * k],
        }
    }

    #[inline]
    pub fn k(&self) -> usize {
        self.k
    }

    #[inline]
    pub fn n_eff(&self) -> f64 {
        self.w_sum
    }

    #[inline]
    pub fn prior_scale(&self) -> f64 {
        self.prior_scale
    }

    #[inline]
    pub fn mean(&self, i: usize) -> f64 {
        self.m[i]
    }

    /// Raw (uncentered) second moment `E_w[x_i x_j]`.
    #[inline]
    pub fn raw(&self, i: usize, j: usize) -> f64 {
        self.s[i * self.k + j]
    }

    /// Centered covariance `E_w[x_i x_j] - m_i m_j`.
    #[inline]
    pub fn cov(&self, i: usize, j: usize) -> f64 {
        self.raw(i, j) - self.m[i] * self.m[j]
    }

    /// Centered variance, floored at zero against rounding.
    #[inline]
    pub fn var(&self, i: usize) -> f64 {
        self.cov(i, i).max(0.0)
    }

    pub fn raw_matrix(&self) -> &[f64] {
        &self.s
    }

    /// One observation with decay factor `lam` (from [`crate::Decay::factor`])
    /// and row weight `w`. O(k^2), allocation-free.
    ///
    /// `w` must be `>= 0`: these are weighted means, and a negative weight has
    /// no meaning here. Callers are expected to reject negative weights at the
    /// boundary (`online-polars` does, naming the offending row); a negative one
    /// reaching this far is a bug, so it is caught in debug builds and treated
    /// as a no-op in release rather than corrupting the accumulator.
    pub fn update(&mut self, x: &[f64], lam: f64, w: f64) {
        debug_assert_eq!(x.len(), self.k);
        debug_assert!(
            w >= 0.0,
            "EwCov::update requires a non-negative weight, got {w}"
        );
        if w < 0.0 {
            return;
        }
        let w_new = lam * self.w_sum + w;
        if w_new <= 0.0 {
            return;
        }
        let a = lam * self.w_sum / w_new; // weight of the old mean
        let b = w / w_new; // weight of the new point
        for i in 0..self.k {
            self.m[i] = a * self.m[i] + b * x[i];
            let row = i * self.k;
            for j in 0..self.k {
                self.s[row + j] = a * self.s[row + j] + b * x[i] * x[j];
            }
        }
        self.w_sum = w_new;
        self.prior_scale *= lam;
    }

    /// Age the accumulator without adding data (pure decay: means unchanged,
    /// only the effective count shrinks).
    pub fn decay(&mut self, lam: f64) {
        self.w_sum *= lam;
        self.prior_scale *= lam;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Direct O(n^2) recomputation with explicit decayed weights.
    fn direct(xs: &[Vec<f64>], lams: &[f64], ws: &[f64]) -> (f64, Vec<f64>, Vec<f64>) {
        let n = xs.len();
        let k = xs[0].len();
        let mut wts = vec![0.0; n];
        for i in 0..n {
            let mut wi = ws[i];
            for lam in &lams[i + 1..] {
                wi *= lam;
            }
            wts[i] = wi;
        }
        let wsum: f64 = wts.iter().sum();
        let mut m = vec![0.0; k];
        let mut s = vec![0.0; k * k];
        for (x, &wi) in xs.iter().zip(&wts) {
            for i in 0..k {
                m[i] += wi * x[i] / wsum;
                for j in 0..k {
                    s[i * k + j] += wi * x[i] * x[j] / wsum;
                }
            }
        }
        (wsum, m, s)
    }

    #[test]
    fn matches_direct_computation() {
        let xs: Vec<Vec<f64>> = (0..40)
            .map(|i| {
                let f = i as f64;
                vec![
                    f.sin() * 2.0,
                    f.cos() - 0.3,
                    (f * 0.7).sin() * (f * 0.1).cos(),
                ]
            })
            .collect();
        let lams: Vec<f64> = (0..40).map(|i| 0.9 + 0.0025 * (i % 3) as f64).collect();
        let ws: Vec<f64> = (0..40).map(|i| 0.5 + 0.03 * (i % 7) as f64).collect();

        let mut ew = EwCov::new(3);
        for i in 0..40 {
            ew.update(&xs[i], lams[i], ws[i]);
        }
        let (wsum, m, s) = direct(&xs, &lams, &ws);
        assert!((ew.n_eff() - wsum).abs() < 1e-10);
        for i in 0..3 {
            assert!((ew.mean(i) - m[i]).abs() < 1e-12, "mean {i}");
            for j in 0..3 {
                assert!((ew.raw(i, j) - s[i * 3 + j]).abs() < 1e-12, "raw {i},{j}");
            }
        }
    }

    #[test]
    fn cov_and_var_are_centered() {
        let mut ew = EwCov::new(2);
        for i in 0..1000 {
            let f = i as f64 * (std::f64::consts::TAU / 100.0); // 10 full periods
            ew.update(&[5.0 + f.sin(), -2.0 + f.cos()], 1.0, 1.0);
        }
        // full periods: means equal the offsets, var = 0.5, cov(sin,cos) = 0
        assert!((ew.mean(0) - 5.0).abs() < 0.02);
        assert!((ew.var(0) - 0.5).abs() < 0.02);
        assert!(ew.cov(0, 1).abs() < 0.02);
    }

    #[test]
    fn pure_decay_keeps_means() {
        let mut ew = EwCov::new(1);
        ew.update(&[3.0], 1.0, 1.0);
        ew.update(&[5.0], 1.0, 1.0);
        let m = ew.mean(0);
        let n = ew.n_eff();
        ew.decay(0.5);
        assert_eq!(ew.mean(0), m);
        assert!((ew.n_eff() - 0.5 * n).abs() < 1e-15);
    }

    #[test]
    fn serde_roundtrip() {
        let mut ew = EwCov::new(2);
        ew.update(&[1.0, 2.0], 0.95, 1.3);
        let bytes = rmp_serde::to_vec(&ew).unwrap();
        let back: EwCov = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(ew, back);
    }
}
