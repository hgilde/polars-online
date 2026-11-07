//! [`EwCovDiag`]: an [`EwCov`] that keeps its co-moments' diagonal alone.
//!
//! A model that reads each feature's variance and never a covariance --
//! `hmm` and `ew_class` under `covariance_type = "diag"`, which score a row
//! by `Σᵢ (xᵢ − μᵢ)² / (vᵢ + r) + ln(vᵢ + r)` -- has no use for the `k(k−1)/2`
//! off-diagonal co-moments an [`EwCov`] keeps and updates on every row. This
//! keeps the `k` it reads, and with them everything else the accumulator
//! keeps beside its matrix: the weight and its Kish sum, the decaying prior
//! scales, the runs a window reads, the means as pairs. `O(k)` a row and
//! `O(k)` held, where an [`EwCov`] is `O(k²)` of each.
//!
//! **Every number is the [`EwCov`]'s diagonal, to the bit.** The update is
//! [`EwCov::update`]'s, operation for operation, for the entries kept:
//!
//! ```text
//! W'    = lam·W + w
//! a     = lam·W / W'        b = w / W'
//! dᵢ    = (xᵢ − mᵢ) − loᵢ                          the pair's deviation
//! vᵢ'   = a·vᵢ + ((a·b)·dᵢ)·dᵢ                      C_ii's step, in its order
//! mᵢ,loᵢ += b·dᵢ                                     Kahan's step, b > 0
//! ```
//!
//! and a diagonal co-moment's step reads no other entry of the matrix, so
//! dropping the rest moves none of these. The weights, Kish's sum and the
//! two prior scales are [`EwCov`]'s scalars, stepped by the same products;
//! `diag_tests` holds every field of both to the bit over streams of
//! uneven, zero and leading-zero weights and decays that take the history.

use serde::{Deserialize, Serialize};

use super::EwCov;
use crate::Runs;

/// An [`EwCov`]'s means, weights, prior scales and runs with its co-moments'
/// diagonal alone; see the [module docs](self).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EwCovDiag {
    k: usize,
    /// EW sum of weights (the `n_eff` count), as [`EwCov`]'s.
    w_sum: f64,
    /// EW sum of squared weights under `lam²`, as [`EwCov`]'s.
    q_sum: Option<f64>,
    /// Product of every decay applied so far.
    prior_scale: f64,
    /// EW means, length `k`.
    m: Vec<f64>,
    /// EW **centred** second moments, length `k`: the diagonal `C_ii` of
    /// the co-moment matrix an [`EwCov`] keeps.
    v: Vec<f64>,
    /// Prior strength on each variance's ridge, `> 0`.
    precision_prior: f64,
    /// The prior's decaying scale `s`, as [`EwCov`]'s: it decays by the
    /// co-moments' factor `a`, so the ridge `precision_prior·s` fades as
    /// the data accumulates.
    precision_scale: f64,
    /// Each feature's held value and the learned row its run started at
    /// ([`Runs`]), for a window to read; off without one.
    runs: Runs,
    /// Learned rows so far.
    rows_learned: u64,
    /// What each mean leaves out: the mean is `m[i] + m_lo[i]`
    /// ([`crate::comp`]).
    m_lo: Vec<f64>,
}

impl EwCovDiag {
    /// An empty accumulator over `k` slots with a ridge of `prior`, finite
    /// and `> 0`, as [`EwCov::with_precision_prior`].
    pub fn with_precision_prior(k: usize, prior: f64) -> Result<Self, String> {
        if prior <= 0.0 || !prior.is_finite() {
            return Err("EwCovDiag::with_precision_prior: prior must be finite and > 0".into());
        }
        Ok(Self {
            k,
            w_sum: 0.0,
            q_sum: Some(0.0),
            prior_scale: 1.0,
            m: vec![0.0; k],
            v: vec![0.0; k],
            precision_prior: prior,
            precision_scale: 1.0,
            runs: Runs::new(k),
            rows_learned: 0,
            m_lo: vec![0.0; k],
        })
    }

    /// An accumulator standing at the given weight, means and centred
    /// variances, as [`EwCov::from_moments`] stands at a matrix whose
    /// diagonal is `variances`.
    pub fn from_moments(
        k: usize,
        w_sum: f64,
        means: &[f64],
        variances: &[f64],
        prior: f64,
    ) -> Result<Self, String> {
        if means.len() != k || variances.len() != k {
            return Err(format!(
                "{k} columns need {k} means and {k} variances, but there are {} and {}. Give one \
                 mean and one variance for each column.",
                means.len(),
                variances.len()
            ));
        }
        if w_sum < 0.0 || !w_sum.is_finite() {
            return Err(format!(
                "the weight of the rows must be finite and at least 0, but it is {w_sum}. Give a \
                 finite weight of at least 0."
            ));
        }
        let mut ew = Self::with_precision_prior(k, prior)?;
        ew.w_sum = w_sum;
        ew.m.copy_from_slice(means);
        ew.v.copy_from_slice(variances);
        Ok(ew)
    }

    /// `cov`'s diagonal: every field `cov` keeps but the off-diagonal
    /// co-moments, as they stand. What a state saved when a diagonal model
    /// kept whole matrices loads as.
    pub fn of(cov: &EwCov) -> Self {
        debug_assert!(!cov.has_pending(), "EwCovDiag::of: a block still pending");
        let k = cov.k;
        Self {
            k,
            w_sum: cov.w_sum,
            q_sum: cov.q_sum,
            prior_scale: cov.prior_scale,
            m: cov.m.clone(),
            v: (0..k).map(|i| cov.c.get(k, i, i)).collect(),
            precision_prior: cov.precision_prior,
            precision_scale: cov.precision_scale,
            runs: cov.runs.clone(),
            rows_learned: cov.rows_learned,
            m_lo: cov.m_lo.clone(),
        }
    }

    /// This accumulator with no runs kept, as [`EwCov::without_runs`].
    #[must_use]
    pub fn without_runs(mut self) -> Self {
        self.runs = Runs::off();
        self
    }

    /// Whether runs are kept for a window to read.
    pub fn keeps_runs(&self) -> bool {
        !self.runs.is_off()
    }

    /// Keep no runs from here, as [`EwCov::set_runs_off`].
    pub fn set_runs_off(&mut self) {
        self.runs = Runs::off();
    }

    #[inline]
    pub fn k(&self) -> usize {
        self.k
    }

    /// Whether the accumulator is over `k` slots with its vectors sized to
    /// match, as [`EwCov::has_shape`].
    pub fn has_shape(&self, k: usize) -> bool {
        self.k == k && self.m.len() == k && self.v.len() == k && self.m_lo.len() == k
    }

    #[inline]
    pub fn n_eff(&self) -> f64 {
        self.w_sum
    }

    /// `Σw²` behind these moments, as [`EwCov::q_sum`].
    #[inline]
    pub fn q_sum(&self) -> Option<f64> {
        self.q_sum
    }

    #[inline]
    pub fn prior_scale(&self) -> f64 {
        self.prior_scale
    }

    /// The decaying scale `s` on the precision prior.
    #[inline]
    pub fn precision_scale(&self) -> f64 {
        self.precision_scale
    }

    #[inline]
    pub fn mean(&self, i: usize) -> f64 {
        self.m[i]
    }

    /// The EW mean vector, length `k`.
    pub fn means(&self) -> &[f64] {
        &self.m
    }

    /// `x`'s deviation from mean `i`, the pair's ([`crate::comp::dev`]).
    #[inline]
    pub fn deviation(&self, i: usize, x: f64) -> f64 {
        crate::comp::dev(x, self.m[i], self.m_lo[i])
    }

    /// The centred second moment `C_ii` as kept, unclamped: what
    /// [`EwCov::cov`]`(i, i)` reads.
    #[inline]
    pub fn comoment(&self, i: usize) -> f64 {
        self.v[i]
    }

    /// The centred second moments as kept, length `k`.
    pub fn comoments(&self) -> &[f64] {
        &self.v
    }

    /// Centred variance, floored at zero against rounding, a NaN kept
    /// ([`EwCov::var`]).
    #[inline]
    pub fn var(&self, i: usize) -> f64 {
        crate::solve::clamp_rounding(self.v[i])
    }

    /// Learned rows so far.
    pub fn rows_learned(&self) -> u64 {
        self.rows_learned
    }

    /// The value feature `i` has held on every learned row from the `row`-th
    /// on, as [`EwCov::held_from`].
    pub fn held_from(&self, i: usize, row: u64) -> Option<f64> {
        self.runs.started_by(self.k, i, row)
    }

    /// One observation with decay `lam` and weight `w`, [`EwCov::update`]'s
    /// arithmetic on the diagonal (the [module docs](self)): `O(k)`.
    pub fn update(&mut self, x: &[f64], lam: f64, w: f64) {
        debug_assert_eq!(x.len(), self.k);
        debug_assert!(
            w >= 0.0,
            "EwCovDiag::update requires a non-negative weight, got {w}"
        );
        if w > 0.0 {
            self.rows_learned += 1;
            self.runs.track(x, self.rows_learned);
        }
        // A zero-weight row with no weight to come out of it is the decay
        // alone, as `EwCov::update` takes it (hard rule 9).
        if w == 0.0 && lam * self.w_sum == 0.0 {
            return self.decay(lam);
        }
        // `EwCov::step_factors`.
        if w < 0.0 {
            return;
        }
        let w_new = lam * self.w_sum + w;
        if w_new <= 0.0 {
            return;
        }
        let a = lam * self.w_sum / w_new;
        let b = w / w_new;
        // `rank_one_rows`' expression for entry `(i, i)`, then the mean's
        // step as a pair: a row of weight 0 takes none.
        for ((vi, mi), (lo, &xi)) in self
            .v
            .iter_mut()
            .zip(self.m.iter_mut())
            .zip(self.m_lo.iter_mut().zip(x))
        {
            let d = crate::comp::dev(xi, *mi, *lo);
            let ab_di = a * b * d;
            *vi = a * *vi + ab_di * d;
            if b > 0.0 {
                crate::comp::add(mi, lo, b * d);
            }
        }
        // `EwCov::commit_scalars`.
        self.precision_scale = if a <= 0.0 {
            1.0
        } else {
            self.precision_scale * a
        };
        self.w_sum = w_new;
        if let Some(q) = self.q_sum.as_mut() {
            *q = lam * lam * *q + w * w;
        }
        self.prior_scale *= lam;
    }

    /// Age the accumulator without adding data, as [`EwCov::decay`].
    pub fn decay(&mut self, lam: f64) {
        self.w_sum *= lam;
        if let Some(q) = self.q_sum.as_mut() {
            *q *= lam * lam;
        }
        self.prior_scale *= lam;
    }

    /// Overwrite the moments, as [`EwCov::set_moments`] does with a matrix
    /// whose diagonal is `variances`: the means' low parts start at zero and
    /// no run stands.
    pub fn set_moments(&mut self, mean: &[f64], variances: &[f64], w_sum: f64, q_sum: Option<f64>) {
        debug_assert_eq!(mean.len(), self.k);
        debug_assert_eq!(variances.len(), self.k);
        self.m.copy_from_slice(mean);
        self.m_lo = vec![0.0; self.k];
        self.v.copy_from_slice(variances);
        self.w_sum = w_sum;
        self.q_sum = q_sum;
        self.runs.forget();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*state >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Every field of `diag` against `cov`'s, to the bit: the scalars, the
    /// means and their low parts, the diagonal, the runs and the count.
    fn same(diag: &EwCovDiag, cov: &EwCov, what: &str) {
        assert_eq!(diag, &EwCovDiag::of(cov), "{what}");
        let k = cov.k();
        for i in 0..k {
            assert_eq!(
                diag.var(i).to_bits(),
                cov.var(i).to_bits(),
                "{what}: var({i})"
            );
            assert_eq!(
                diag.comoment(i).to_bits(),
                cov.cov(i, i).to_bits(),
                "{what}: C({i}, {i})"
            );
        }
        assert_eq!(diag.n_eff().to_bits(), cov.n_eff().to_bits(), "{what}");
        assert_eq!(
            diag.precision_scale().to_bits(),
            cov.precision_scale().to_bits()
        );
        assert_eq!(diag.prior_scale().to_bits(), cov.prior_scale().to_bits());
        assert_eq!(
            diag.q_sum().map(f64::to_bits),
            cov.q_sum().map(f64::to_bits)
        );
    }

    /// The diagonal accumulator is the full one's diagonal to the bit, every
    /// field, over rows of uneven, zero and leading-zero weight, decays from
    /// none to one that takes the whole history, features at an offset and
    /// held ones, at widths on both sides of the serial step's two paths,
    /// with runs kept and off; and an `EwCov` converted mid-stream goes on
    /// as the one kept from the start.
    #[test]
    fn the_diagonal_is_the_full_accumulators_to_the_bit() {
        for k in [1, 2, 3, 5, 9, 12, 33] {
            for runs in [true, false] {
                let mut s = 17 + k as u64;
                let mut cov = EwCov::with_precision_prior(k, 0.3).unwrap();
                let mut diag = EwCovDiag::with_precision_prior(k, 0.3).unwrap();
                if !runs {
                    cov = cov.without_runs();
                    diag = diag.without_runs();
                }
                let mut converted: Option<EwCovDiag> = None;
                for row in 0..400 {
                    let x: Vec<f64> = (0..k)
                        .map(|i| match i % 4 {
                            0 => 1e6 + lcg(&mut s),
                            1 => 2.5,
                            _ => 3.0 * lcg(&mut s) - 1.0,
                        })
                        .collect();
                    let w = match row {
                        0..=2 => 0.0,
                        _ => match (lcg(&mut s) * 6.0) as u32 {
                            0 => 0.0,
                            1 => 1e-9,
                            2 => 7.5,
                            _ => lcg(&mut s),
                        },
                    };
                    let lam = match (lcg(&mut s) * 8.0) as u32 {
                        0 => 1.0,
                        1 => 0.0,
                        2 => 0.5,
                        _ => 0.97,
                    };
                    cov.update(&x, lam, w);
                    diag.update(&x, lam, w);
                    if row % 7 == 3 {
                        cov.decay(0.9);
                        diag.decay(0.9);
                    }
                    if row == 150 {
                        converted = Some(EwCovDiag::of(&cov));
                    }
                    if let Some(c) = converted.as_mut().filter(|_| row > 150) {
                        c.update(&x, lam, w);
                        if row % 7 == 3 {
                            c.decay(0.9);
                        }
                        assert_eq!(c, &diag, "k = {k}, converted, row {row}");
                    }
                    same(&diag, &cov, &format!("k = {k}, runs {runs}, row {row}"));
                    for i in 0..k {
                        assert_eq!(diag.held_from(i, 1), cov.held_from(i, 1));
                        let probe = 0.25 + i as f64;
                        assert_eq!(
                            diag.deviation(i, probe).to_bits(),
                            cov.deviation(i, probe).to_bits()
                        );
                    }
                }
                assert!(diag.n_eff() > 0.0 && diag.var(2.min(k - 1)) >= 0.0);
            }
        }
    }

    /// Given moments, and moments set, are the full accumulator's with that
    /// diagonal, at a weight of 0 too (no rows yet, which hard rule 9 makes
    /// legal); a shape of the wrong width is refused, as are a negative or
    /// unbounded weight and a bad prior.
    #[test]
    fn given_and_set_moments_are_the_full_ones_diagonal() {
        let k = 3;
        let means = [1.0, -2.0, 0.5];
        let full = [2.0, 0.3, -0.1, 0.3, 1.5, 0.2, -0.1, 0.2, 0.7];
        let vars = [2.0, 1.5, 0.7];
        let cov = EwCov::from_moments(k, 1.0, &means, &full, 0.1).unwrap();
        let diag = EwCovDiag::from_moments(k, 1.0, &means, &vars, 0.1).unwrap();
        assert_eq!(diag, EwCovDiag::of(&cov));
        let (mut cov, mut diag) = (cov, diag);
        cov.set_moments(&[0.1, 0.2, 0.3], &full, 4.0, Some(2.0));
        diag.set_moments(&[0.1, 0.2, 0.3], &vars, 4.0, Some(2.0));
        assert_eq!(diag, EwCovDiag::of(&cov));
        assert!(EwCovDiag::from_moments(k, 1.0, &means, &full, 0.1).is_err());
        assert!(EwCovDiag::from_moments(k, -1.0, &means, &vars, 0.1).is_err());
        for bad in [f64::NAN, f64::INFINITY, -f64::MIN_POSITIVE] {
            assert!(EwCovDiag::from_moments(k, bad, &means, &vars, 0.1).is_err());
        }
        let empty = EwCov::from_moments(k, 0.0, &means, &full, 0.1).unwrap();
        assert_eq!(
            EwCovDiag::from_moments(k, 0.0, &means, &vars, 0.1),
            Ok(EwCovDiag::of(&empty))
        );
        assert!(EwCovDiag::with_precision_prior(k, 0.0).is_err());
        assert!(EwCovDiag::with_precision_prior(k, f64::INFINITY).is_err());
        assert!(diag.has_shape(3) && !diag.has_shape(2));
    }
}
