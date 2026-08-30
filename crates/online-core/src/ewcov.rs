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
    /// Optional Sherman-Morrison inverse of `M = C + s·prior·I`, row-major
    /// `k*k` (docs/PLAN.md §4.7).
    ///
    /// The prior's scale `s` must decay by the *same* factor `a` the
    /// co-moments do, not by `lam`, or the update is not rank-1:
    /// `M' = a·C + a·b·δδᵀ + a·s·prior·I = a·(M + b·δδᵀ)`, hence
    /// `M'⁻¹ = (1/a)·SM(M⁻¹, δ)`. Like RLS's `P₀`, this makes the prior fade as
    /// data accumulates.
    #[serde(default)]
    inv: Option<Vec<f64>>,
    /// Prior strength for the tracked inverse; `0` when no inverse is kept.
    #[serde(default)]
    inv_prior: f64,
    /// Decaying scale on that prior (see `inv`).
    #[serde(default)]
    inv_scale: f64,
}

impl EwCov {
    pub fn new(k: usize) -> Self {
        Self {
            k,
            w_sum: 0.0,
            prior_scale: 1.0,
            m: vec![0.0; k],
            c: vec![0.0; k * k],
            inv: None,
            inv_prior: 0.0,
            inv_scale: 1.0,
        }
    }

    /// Same, but also maintaining `(C + s·prior·I)⁻¹` incrementally via
    /// Sherman-Morrison, so a precision matrix is available without a solve.
    ///
    /// `prior` regularizes the inverse and must be `> 0`: the centered
    /// co-moment matrix starts at zero and is singular until `k` independent
    /// rows have been seen, so there is nothing to invert without it.
    pub fn with_inverse(k: usize, prior: f64) -> Result<Self, String> {
        if prior <= 0.0 || !prior.is_finite() {
            return Err("EwCov::with_inverse: prior must be finite and > 0".into());
        }
        let mut ew = Self::new(k);
        let mut inv = vec![0.0; k * k];
        for i in 0..k {
            inv[i * k + i] = 1.0 / prior;
        }
        ew.inv = Some(inv);
        ew.inv_prior = prior;
        ew.inv_scale = 1.0;
        Ok(ew)
    }

    #[inline]
    pub fn k(&self) -> usize {
        self.k
    }

    /// Element of `(C + s·prior·I)⁻¹`, or `None` when no inverse is tracked.
    #[inline]
    pub fn inv(&self, i: usize, j: usize) -> Option<f64> {
        self.inv.as_ref().map(|v| v[i * self.k + j])
    }

    pub fn has_inverse(&self) -> bool {
        self.inv.is_some()
    }

    /// Current decaying scale on the inverse's prior (see the `inv` field).
    pub fn inv_scale(&self) -> f64 {
        self.inv_scale
    }

    /// Partial correlation between `i` and `j`, controlling for every other
    /// column: `−P_ij / sqrt(P_ii · P_jj)` from the precision matrix `P`.
    /// `None` when no inverse is tracked.
    pub fn partial_corr(&self, i: usize, j: usize) -> Option<f64> {
        let p = self.inv.as_ref()?;
        let (pij, pii, pjj) = (p[i * self.k + j], p[i * self.k + i], p[j * self.k + j]);
        let d = (pii * pjj).sqrt();
        Some(if d > 0.0 {
            (-pij / d).clamp(-1.0, 1.0)
        } else {
            f64::NAN
        })
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
        // Sherman-Morrison on the same rank-1 update, before the mean moves
        // (the deviations above were taken against the old mean).
        let k = self.k;
        let prior = self.inv_prior;
        // `d` is computed before the mutable borrow of `self.inv`.
        let d: Vec<f64> = (0..k).map(|i| x[i] - self.m[i]).collect();
        // `a == 0` only on the very first observation, when the whole history
        // is discarded and the centered co-moments are exactly zero.
        let first_observation = a <= 0.0;
        let mut scale_factor = 1.0;
        if let Some(inv) = &mut self.inv {
            if first_observation {
                // First observation: `a = 0` discards everything, so the
                // centered co-moments are exactly zero and M is just the prior.
                // There is no rank-1 step to take -- reinitialize instead of
                // dividing by zero.
                inv.iter_mut().for_each(|v| *v = 0.0);
                for i in 0..k {
                    inv[i * k + i] = 1.0 / prior;
                }
            } else {
                let mut u = vec![0.0; k];
                for (i, ui) in u.iter_mut().enumerate() {
                    let row = i * k;
                    *ui = (0..k).map(|j| inv[row + j] * d[j]).sum();
                }
                let dtu: f64 = d.iter().zip(&u).map(|(di, ui)| di * ui).sum();
                let denom = 1.0 + b * dtu;
                if denom.abs() > 1e-300 {
                    let f = b / denom;
                    for i in 0..k {
                        let row = i * k;
                        for j in 0..k {
                            inv[row + j] = (inv[row + j] - f * u[i] * u[j]) / a;
                        }
                    }
                    scale_factor = a;
                }
            }
        }
        if self.inv.is_some() {
            self.inv_scale = if first_observation {
                1.0
            } else {
                self.inv_scale * scale_factor
            };
        }
        for (mi, xi) in self.m.iter_mut().zip(x) {
            *mi += b * (xi - *mi);
        }
        self.w_sum = w_new;
        self.prior_scale *= lam;
    }

    /// Overwrite the moments directly. Used when two accumulators are mixed
    /// (see `EwRidge::blend_toward_long_run`); the caller is responsible for
    /// the mixture being a valid set of weighted moments.
    pub fn set_moments(&mut self, mean: &[f64], centered: &[f64], w_sum: f64) {
        debug_assert_eq!(mean.len(), self.k);
        debug_assert_eq!(centered.len(), self.k * self.k);
        self.m.copy_from_slice(mean);
        self.c.copy_from_slice(centered);
        self.w_sum = w_sum;
    }

    /// Age the accumulator without adding data (pure decay: means unchanged,
    /// only the effective count shrinks).
    pub fn decay(&mut self, lam: f64) {
        self.w_sum *= lam;
        self.prior_scale *= lam;
    }

    /// Reference inverse, recomputed from scratch by Gauss-Jordan. Used by the
    /// tests to check the incremental one, and available for a caller that
    /// wants a one-off precision matrix without paying for the tracking.
    pub fn inverse_from_scratch(&self, prior: f64, prior_scale: f64) -> Option<Vec<f64>> {
        let k = self.k;
        let mut a = vec![0.0; k * 2 * k];
        for i in 0..k {
            for j in 0..k {
                a[i * 2 * k + j] = self.c[i * k + j];
            }
            a[i * 2 * k + i] += prior * prior_scale;
            a[i * 2 * k + k + i] = 1.0;
        }
        for col in 0..k {
            let piv = (col..k).max_by(|&r1, &r2| {
                a[r1 * 2 * k + col]
                    .abs()
                    .partial_cmp(&a[r2 * 2 * k + col].abs())
                    .unwrap()
            })?;
            if a[piv * 2 * k + col].abs() < 1e-300 {
                return None;
            }
            for j in 0..2 * k {
                a.swap(col * 2 * k + j, piv * 2 * k + j);
            }
            let d = a[col * 2 * k + col];
            for j in 0..2 * k {
                a[col * 2 * k + j] /= d;
            }
            for r in 0..k {
                if r != col {
                    let f = a[r * 2 * k + col];
                    for j in 0..2 * k {
                        a[r * 2 * k + j] -= f * a[col * 2 * k + j];
                    }
                }
            }
        }
        Some(
            (0..k)
                .flat_map(|i| (0..k).map(move |j| (i, j)))
                .map(|(i, j)| a[i * 2 * k + k + j])
                .collect(),
        )
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
    /// Partial correlation for each unordered pair, controlling for every other
    /// column. Needs the tracked inverse (`precision_prior`).
    PartialCorr,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct EwCovCfg {
    pub n_features: usize,
    pub decay: crate::Decay,
    pub stats: Vec<EwCovStat>,
    pub min_periods: f64,
    /// Prior for the tracked precision matrix. Required by
    /// [`EwCovStat::PartialCorr`]; `None` skips the inverse entirely.
    #[serde(default)]
    pub precision_prior: Option<f64>,
}

impl EwCovCfg {
    pub fn validate(&self) -> Result<(), String> {
        if self.n_features == 0 {
            return Err("ew_cov: at least one column is required".into());
        }
        if self.stats.is_empty() {
            return Err("ew_cov: at least one statistic is required".into());
        }
        let pairwise =
            |s: &EwCovStat| matches!(s, EwCovStat::Cov | EwCovStat::Corr | EwCovStat::PartialCorr);
        if self.n_features < 2 && self.stats.iter().any(pairwise) {
            return Err("ew_cov: cov/corr/partial_corr need at least two columns".into());
        }
        if self.stats.contains(&EwCovStat::PartialCorr) && self.precision_prior.is_none() {
            return Err(
                "ew_cov: partial_corr needs `precision_prior` (it is computed from the \
                 tracked precision matrix)"
                    .into(),
            );
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
                EwCovStat::Cov | EwCovStat::Corr | EwCovStat::PartialCorr => pairs,
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
        let cov = match cfg.precision_prior {
            Some(p) => EwCov::with_inverse(cfg.n_features, p)?,
            None => EwCov::new(cfg.n_features),
        };
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
                EwCovStat::PartialCorr => {
                    for i in 0..names.len() {
                        for j in (i + 1)..names.len() {
                            out.push(format!("pcorr_{}_{}", names[i], names[j]));
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
                EwCovStat::PartialCorr => {
                    for i in 0..k {
                        for j in (i + 1)..k {
                            out.push(self.cov.partial_corr(i, j).unwrap_or(f64::NAN));
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
    /// Zero: `ew_cov` regresses nothing, it reports statistics. The slot count
    /// callers want is [`Self::n_outputs`], which is not `n_targets`-derived
    /// here as it is for every other model.
    fn n_targets(&self) -> usize {
        0
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

    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }

    fn model_cfg(k: usize, stats: Vec<EwCovStat>) -> EwCovCfg {
        EwCovCfg {
            n_features: k,
            decay: crate::Decay::Halflife(f64::INFINITY),
            stats,
            min_periods: 2.0,
            precision_prior: None,
        }
    }

    #[test]
    fn model_cfg_validation_rejects_each_bad_field() {
        use EwCovStat::*;
        let err = |c: EwCovCfg, want: &str| match c.validate() {
            Err(e) => assert!(e.contains(want), "wanted {want:?}, got {e:?}"),
            Ok(()) => panic!("expected rejection mentioning {want:?}"),
        };

        err(model_cfg(0, vec![Mean]), "at least one column");
        err(model_cfg(2, vec![]), "at least one statistic");

        // Per-column stats are fine with a single column; pairwise ones are not.
        model_cfg(1, vec![Mean, Var, Std]).validate().unwrap();
        for s in [Cov, Corr] {
            err(model_cfg(1, vec![s]), "at least two columns");
        }
        let mut one_pcorr = model_cfg(1, vec![PartialCorr]);
        one_pcorr.precision_prior = Some(1e-6);
        err(one_pcorr, "at least two columns");

        // partial_corr is read off the tracked precision matrix, which only
        // exists when a prior is configured.
        err(model_cfg(3, vec![PartialCorr]), "precision_prior");
        let mut with_prior = model_cfg(3, vec![PartialCorr]);
        with_prior.precision_prior = Some(1e-6);
        with_prior.validate().unwrap();

        model_cfg(2, vec![Mean, Var, Std, Cov, Corr])
            .validate()
            .unwrap();
    }

    #[test]
    fn n_outputs_and_labels_agree_slot_for_slot() {
        // `n_outputs` sizes the output buffer, `labels` names the fields and
        // `read` fills them: all three must walk the stats in the same order
        // and produce the same count, or a field ends up named after another
        // statistic's value.
        use EwCovStat::*;
        let names: Vec<String> = ["a", "b", "c"].iter().map(|s| s.to_string()).collect();
        for stats in [
            vec![Mean],
            vec![Var],
            vec![Std],
            vec![Cov],
            vec![Corr],
            vec![Mean, Var, Std, Cov, Corr],
            vec![Corr, Mean],
        ] {
            let mut c = model_cfg(3, stats.clone());
            c.precision_prior = Some(1e-6);
            let labels = EwCovModel::labels(&names, &stats);
            assert_eq!(c.n_outputs(), labels.len(), "{stats:?}");

            let mut m = EwCovModel::new(c).unwrap();
            let mut s = 31u64;
            for i in 0..40 {
                let x = [lcg(&mut s), lcg(&mut s), lcg(&mut s)];
                crate::OnlineModel::step(&mut m, &x, &[], if i == 0 { 0.0 } else { 1.0 }, 1.0);
            }
            let vals = m.read();
            assert_eq!(vals.len(), labels.len(), "{stats:?}");

            // Spot-check that the label describes the value under it.
            for (label, v) in labels.iter().zip(&vals) {
                if label.starts_with("var_") || label.starts_with("std_") {
                    assert!(*v >= 0.0, "{label} = {v}");
                }
                if label.starts_with("corr_") {
                    assert!((-1.0..=1.0).contains(v), "{label} = {v}");
                }
            }
        }
    }

    #[test]
    fn labels_name_the_pair_in_column_order() {
        use EwCovStat::*;
        let names: Vec<String> = ["x", "y", "z"].iter().map(|s| s.to_string()).collect();
        assert_eq!(
            EwCovModel::labels(&names, &[Mean]),
            ["mean_x", "mean_y", "mean_z"]
        );
        assert_eq!(
            EwCovModel::labels(&names, &[Var]),
            ["var_x", "var_y", "var_z"]
        );
        assert_eq!(
            EwCovModel::labels(&names, &[Std]),
            ["std_x", "std_y", "std_z"]
        );
        // Upper triangle only, and never a self-pair.
        assert_eq!(
            EwCovModel::labels(&names, &[Cov]),
            ["cov_x_y", "cov_x_z", "cov_y_z"]
        );
        assert_eq!(
            EwCovModel::labels(&names, &[Corr]),
            ["corr_x_y", "corr_x_z", "corr_y_z"]
        );
        assert_eq!(
            EwCovModel::labels(&names, &[PartialCorr]),
            ["pcorr_x_y", "pcorr_x_z", "pcorr_y_z"]
        );
        // Stats concatenate in the order given, not a canonical order.
        assert_eq!(
            EwCovModel::labels(&names, &[Corr, Mean]).first().unwrap(),
            "corr_x_y"
        );
    }

    #[test]
    fn read_returns_the_statistic_each_slot_claims() {
        // `read` is the one place the statistics are actually computed rather
        // than accumulated: var vs std vs corr all come off the same moments,
        // so a slot filled from the wrong one is invisible to the accumulator
        // tests. Each is checked against the definition.
        use EwCovStat::*;
        let stats = vec![Mean, Var, Std, Cov, Corr];
        let mut m = EwCovModel::new(model_cfg(2, stats)).unwrap();
        let mut s = 37u64;
        let mut xs = Vec::new();
        for i in 0..80 {
            let a = lcg(&mut s);
            let x = [3.0 + a, 10.0 - 2.0 * a + 0.5 * lcg(&mut s)];
            crate::OnlineModel::step(&mut m, &x, &[], if i == 0 { 0.0 } else { 1.0 }, 1.0);
            xs.push(x.to_vec());
        }
        let v = m.read();
        let (mean0, mean1) = (v[0], v[1]);
        let (var0, var1) = (v[2], v[3]);
        let (std0, std1) = (v[4], v[5]);
        let cov01 = v[6];
        let corr01 = v[7];

        // Against an unweighted recomputation (halflife is infinite here).
        let n = xs.len() as f64;
        let m0: f64 = xs.iter().map(|x| x[0]).sum::<f64>() / n;
        let m1: f64 = xs.iter().map(|x| x[1]).sum::<f64>() / n;
        let c01: f64 = xs.iter().map(|x| (x[0] - m0) * (x[1] - m1)).sum::<f64>() / n;
        let v0: f64 = xs.iter().map(|x| (x[0] - m0).powi(2)).sum::<f64>() / n;
        let v1: f64 = xs.iter().map(|x| (x[1] - m1).powi(2)).sum::<f64>() / n;

        assert!((mean0 - m0).abs() < 1e-10, "{mean0} vs {m0}");
        assert!((mean1 - m1).abs() < 1e-10, "{mean1} vs {m1}");
        assert!((var0 - v0).abs() < 1e-10, "{var0} vs {v0}");
        assert!((var1 - v1).abs() < 1e-10);
        assert!((std0 - v0.sqrt()).abs() < 1e-10, "std is sqrt(var)");
        assert!((std1 - v1.sqrt()).abs() < 1e-10);
        assert!((cov01 - c01).abs() < 1e-10, "{cov01} vs {c01}");
        assert!(
            (corr01 - c01 / (v0 * v1).sqrt()).abs() < 1e-10,
            "corr is cov / (std*std)"
        );
        // The relationship is strongly negative, so a sign error is visible.
        assert!(corr01 < -0.9, "{corr01}");
    }

    #[test]
    fn corr_is_nan_not_infinite_when_a_column_is_constant() {
        use EwCovStat::*;
        let mut m = EwCovModel::new(model_cfg(2, vec![Corr, Var])).unwrap();
        let mut s = 41u64;
        for i in 0..40 {
            crate::OnlineModel::step(
                &mut m,
                &[lcg(&mut s), 5.0],
                &[],
                if i == 0 { 0.0 } else { 1.0 },
                1.0,
            );
        }
        let v = m.read();
        assert!(v[0].is_nan(), "corr against a constant column: {}", v[0]);
        assert_eq!(v[2], 0.0, "a constant column has exactly zero variance");
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
    fn tracked_inverse_matches_a_from_scratch_solve() {
        // The incremental Sherman-Morrison inverse must equal a direct
        // inversion of the same matrix at every step, not just at the end.
        let prior = 0.5;
        let mut ew = EwCov::with_inverse(3, prior).unwrap();
        let mut state = 7u64;
        let mut lcg = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        };
        for step in 0..60 {
            let x = [lcg(), lcg(), lcg() * 3.0];
            ew.update(&x, 0.97, 0.5 + lcg().abs());
            let want = ew
                .inverse_from_scratch(prior, ew.inv_scale())
                .expect("reference inverse should exist");
            for i in 0..3 {
                for j in 0..3 {
                    let got = ew.inv(i, j).unwrap();
                    assert!(
                        (got - want[i * 3 + j]).abs() < 1e-6 * (1.0 + want[i * 3 + j].abs()),
                        "step {step}, ({i},{j}): tracked {got}, direct {}",
                        want[i * 3 + j]
                    );
                }
            }
        }
    }

    #[test]
    fn partial_correlation_detects_a_spurious_link() {
        // x2 = x0 + x1 + noise. Marginally x0 and x2 correlate; controlling for
        // x1 they still do, but x0 and x1 are independent both ways. The
        // interesting case is the reverse: with a common driver, the marginal
        // correlation is high and the partial one is not.
        let mut ew = EwCov::with_inverse(3, 1e-6).unwrap();
        let mut state = 11u64;
        let mut lcg = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        };
        for _ in 0..20000 {
            let driver = lcg();
            // two children of one driver: correlated with each other only
            // through it
            let a = driver + 0.1 * lcg();
            let b = driver + 0.1 * lcg();
            ew.update(&[driver, a, b], 1.0, 1.0);
        }
        let marginal = ew.cov(1, 2) / (ew.var(1) * ew.var(2)).sqrt();
        let partial = ew.partial_corr(1, 2).unwrap();
        assert!(
            marginal > 0.9,
            "children should correlate marginally: {marginal}"
        );
        assert!(
            partial.abs() < 0.2,
            "controlling for the driver should remove it: {partial}"
        );
    }

    #[test]
    fn with_inverse_rejects_a_non_positive_prior() {
        assert!(EwCov::with_inverse(2, 0.0).is_err());
        assert!(EwCov::with_inverse(2, -1.0).is_err());
        assert!(EwCov::with_inverse(2, f64::INFINITY).is_err());
    }

    #[test]
    fn no_inverse_by_default() {
        let ew = EwCov::new(2);
        assert!(!ew.has_inverse());
        assert!(ew.inv(0, 0).is_none());
        assert!(ew.partial_corr(0, 1).is_none());
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
