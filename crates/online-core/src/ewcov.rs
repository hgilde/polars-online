//! `EwCov`: exponentially weighted first/second moments of a vector stream
//! (docs/PLAN.md §4.7). The shared accumulator behind EW-ridge, RLS and Kalman,
//! and exposed on its own as `online.ew_cov()`.
//!
//! All statistics are stored as weighted *means* (not sums), which keeps them
//! bounded under arbitrarily long runs (docs/PLAN.md §7), and the second
//! moments are kept **centered** (a weighted Welford update) rather than raw:
//!
//! ```text
//! W'     = lam * W + w
//! a      = lam * W / W'        b = w / W'        (a + b = 1)
//! delta  = x - m
//! m'     = m + b * delta
//! C'_ij  = a * C_ij + a * b * delta_i * delta_j
//! ```
//!
//! Centered updates matter because the earlier raw form derived the variance as
//! `E[x²] − m²`, and that subtraction loses precision when the mean is large
//! relative to the spread: a unit-variance feature around 1e8 has
//! `var/E[x²] ≈ 1e-16`, so the variance was destroyed entirely. With this form
//! the variance is accurate at any offset (see `docs/TESTING.md` T-E9 and the
//! river cross-check in `tests/test_river.py`). [`EwCov::raw`] reconstructs the
//! raw moment as `C_ij + m_i m_j` for the callers that genuinely need it (the
//! uncentered normal equations).

use serde::{Deserialize, Serialize};

/// Is a centered variance large enough to standardize by, given the raw second
/// moment it was computed from?
///
/// `var = E[x²] − m²` loses precision by cancellation when the mean is large
/// relative to the spread: the absolute error is on the order of
/// `f64::EPSILON * E[x²]`. The threshold is a small multiple of that noise
/// floor, so a genuine variance is kept even when it sits on a large offset
/// (a unit-variance feature around 1e6 has `var/raw ≈ 1e-12`, which is real),
/// while a variance that has actually been destroyed by cancellation is
/// rejected and the feature dropped from the solve.
///
/// A previous fixed `1e-10 * raw` threshold was ~450,000x the noise floor and
/// silently discarded usable features at ordinary financial scales.
///
/// Since [`EwCov`] switched to centered (Welford) updates the variance no
/// longer suffers cancellation, so the only question left is whether the
/// feature is genuinely constant — which gives *exactly* zero, because every
/// deviation from the running mean is exactly zero. The `raw_second_moment`
/// argument is kept for callers and for the doc trail, but is no longer used to
/// scale the threshold: doing so is what silently dropped usable features
/// sitting on a large offset.
#[inline]
pub fn variance_is_usable(var: f64, _raw_second_moment: f64) -> bool {
    var > 0.0 && var.is_finite()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EwCov {
    k: usize,
    /// EW sum of weights (the `n_eff` count).
    w_sum: f64,
    /// Product of all decay factors applied so far (a decaying prior's scale).
    prior_scale: f64,
    /// EW mean vector, length `k`.
    m: Vec<f64>,
    /// EW **centered** co-moments, row-major `k*k`.
    #[serde(alias = "s")]
    c: Vec<f64>,
}

impl EwCov {
    pub fn new(k: usize) -> Self {
        Self {
            k,
            w_sum: 0.0,
            prior_scale: 1.0,
            m: vec![0.0; k],
            c: vec![0.0; k * k],
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

    /// Raw (uncentered) second moment `E_w[x_i x_j]`, reconstructed from the
    /// centered co-moment. Use [`Self::cov`] wherever a *centered* quantity is
    /// wanted: going through `raw` and subtracting the means again reintroduces
    /// exactly the cancellation this representation avoids.
    #[inline]
    pub fn raw(&self, i: usize, j: usize) -> f64 {
        self.c[i * self.k + j] + self.m[i] * self.m[j]
    }

    /// Centered covariance, held directly.
    #[inline]
    pub fn cov(&self, i: usize, j: usize) -> f64 {
        self.c[i * self.k + j]
    }

    /// Centered variance, floored at zero against rounding.
    #[inline]
    pub fn var(&self, i: usize) -> f64 {
        self.cov(i, i).max(0.0)
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
        let a = lam * self.w_sum / w_new; // weight of the old statistics
        let b = w / w_new; // weight of the new point
        // Weighted Welford: co-moments are updated from the deviations against
        // the OLD mean, then the mean is advanced.
        let k = self.k;
        for i in 0..k {
            let di = x[i] - self.m[i];
            let row = i * k;
            for (j, mj) in self.m.iter().enumerate().take(k) {
                let dj = x[j] - mj;
                self.c[row + j] = a * self.c[row + j] + a * b * di * dj;
            }
        }
        for (mi, xi) in self.m.iter_mut().zip(x) {
            *mi += b * (xi - *mi);
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

/// Which statistics an [`EwCovModel`] emits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EwCovStat {
    /// EW mean of each column.
    Mean,
    /// EW variance of each column (centered).
    Var,
    /// EW standard deviation of each column.
    Std,
    /// Centered covariance for each unordered pair `i < j`.
    Cov,
    /// Pearson correlation for each unordered pair `i < j`.
    Corr,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct EwCovCfg {
    pub n_features: usize,
    pub decay: crate::Decay,
    pub stats: Vec<EwCovStat>,
    pub min_periods: f64,
}

impl EwCovCfg {
    pub fn validate(&self) -> Result<(), String> {
        if self.n_features == 0 {
            return Err("ew_cov: at least one column is required".into());
        }
        if self.stats.is_empty() {
            return Err("ew_cov: at least one statistic is required".into());
        }
        if self.n_features < 2
            && self
                .stats
                .iter()
                .any(|s| matches!(s, EwCovStat::Cov | EwCovStat::Corr))
        {
            return Err("ew_cov: cov/corr need at least two columns".into());
        }
        Ok(())
    }

    /// Number of output slots, in emission order.
    pub fn n_outputs(&self) -> usize {
        let k = self.n_features;
        let pairs = k * (k - 1) / 2;
        self.stats
            .iter()
            .map(|s| match s {
                EwCovStat::Mean | EwCovStat::Var | EwCovStat::Std => k,
                EwCovStat::Cov | EwCovStat::Corr => pairs,
            })
            .sum()
    }
}

/// [`EwCov`] exposed as a model in its own right (docs/PLAN.md §4.7), so EW
/// means, variances, covariances and correlations are available as outputs
/// without fitting a regression. Replaces pure-Polars pairwise EW correlations,
/// which cost O(k²) passes; this is one O(k²) update per row.
///
/// It has no targets: every column is a "feature", and the statistics are
/// reported from the state *before* each row, like every prediction here.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct EwCovModel {
    cfg: EwCovCfg,
    cov: EwCov,
}

impl EwCovModel {
    pub fn new(cfg: EwCovCfg) -> Result<Self, String> {
        cfg.validate()?;
        let cov = EwCov::new(cfg.n_features);
        Ok(Self { cfg, cov })
    }

    pub fn cfg(&self) -> &EwCovCfg {
        &self.cfg
    }

    pub fn n_eff(&self) -> f64 {
        self.cov.n_eff()
    }

    /// Output slot labels, in emission order (used for field names).
    pub fn labels(names: &[String], stats: &[EwCovStat]) -> Vec<String> {
        let mut out = Vec::new();
        for stat in stats {
            match stat {
                EwCovStat::Mean => out.extend(names.iter().map(|n| format!("mean_{n}"))),
                EwCovStat::Var => out.extend(names.iter().map(|n| format!("var_{n}"))),
                EwCovStat::Std => out.extend(names.iter().map(|n| format!("std_{n}"))),
                EwCovStat::Cov => {
                    for i in 0..names.len() {
                        for j in (i + 1)..names.len() {
                            out.push(format!("cov_{}_{}", names[i], names[j]));
                        }
                    }
                }
                EwCovStat::Corr => {
                    for i in 0..names.len() {
                        for j in (i + 1)..names.len() {
                            out.push(format!("corr_{}_{}", names[i], names[j]));
                        }
                    }
                }
            }
        }
        out
    }

    fn read(&self) -> Vec<f64> {
        let k = self.cfg.n_features;
        let mut out = Vec::with_capacity(self.cfg.n_outputs());
        for stat in &self.cfg.stats {
            match stat {
                EwCovStat::Mean => (0..k).for_each(|i| out.push(self.cov.mean(i))),
                EwCovStat::Var => (0..k).for_each(|i| out.push(self.cov.var(i))),
                EwCovStat::Std => (0..k).for_each(|i| out.push(self.cov.var(i).sqrt())),
                EwCovStat::Cov => {
                    for i in 0..k {
                        for j in (i + 1)..k {
                            out.push(self.cov.cov(i, j));
                        }
                    }
                }
                EwCovStat::Corr => {
                    for i in 0..k {
                        for j in (i + 1)..k {
                            let d = (self.cov.var(i) * self.cov.var(j)).sqrt();
                            out.push(if d > 0.0 {
                                (self.cov.cov(i, j) / d).clamp(-1.0, 1.0)
                            } else {
                                f64::NAN
                            });
                        }
                    }
                }
            }
        }
        out
    }
}

impl crate::OnlineModel for EwCovModel {
    fn step(&mut self, x: &[f64], _y: &[Option<f64>], d_clock: f64, weight: f64) -> crate::Step {
        let n_eff = self.cov.n_eff();
        // Statistics are read before this row is folded in, so an `ew_cov`
        // column is usable as a feature for the same row without leaking it.
        let pred = if n_eff >= self.cfg.min_periods {
            self.read()
        } else {
            vec![f64::NAN; self.cfg.n_outputs()]
        };
        self.cov.update(x, self.cfg.decay.factor(d_clock), weight);
        crate::Step {
            pred,
            coef: None,
            n_eff,
            extra: None,
        }
    }

    fn state(&self) -> crate::State {
        crate::State::new(crate::ModelState::EwCovModel(Box::new(self.clone())))
    }

    fn restore(s: &crate::State) -> Result<Self, crate::StateError> {
        crate::check_schema(s)?;
        match &s.model {
            crate::ModelState::EwCovModel(m) => Ok((**m).clone()),
            other => Err(crate::StateError::WrongModel {
                expected: "ew_cov",
                found: other.kind(),
            }),
        }
    }

    /// `ew_cov` has no targets; one nominal target carries all the slots.
    fn n_targets(&self) -> usize {
        1
    }

    fn n_features(&self) -> usize {
        self.cfg.n_features
    }

    fn n_outputs(&self) -> usize {
        self.cfg.n_outputs()
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
