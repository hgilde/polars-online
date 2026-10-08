//! Each target's own spread: the unit an insensitivity band is drawn in
//! (docs/PLAN.md task 202).
//!
//! `pa`'s `eps`, and `sgd`'s `eps` under `epsilon_insensitive`, are in units
//! of the target's own EW standard deviation: the spread of `y` around its
//! EW mean, kept per target on the model's clock. For target `j`, over the
//! rows that carried it with a weight above 0, the one-slot form of
//! [`EwDiag`]'s weighted Welford recursion:
//!
//! ```text
//! W' = lam·W + w      a = lam·W / W'      b = w / W'
//! d  = y − m          v' = a·v + a·b·d²   m' = m + b·d
//! ```
//!
//! so that `m = Σ ωᵢ yᵢ / Σ ωᵢ` and `v = Σ ωᵢ (yᵢ − m)² / Σ ωᵢ`, with `ωᵢ`
//! row `i`'s weight aged by the clock since it. A row without the target,
//! or of weight 0, only ages `W` (hard rule 9). The mean is a compensated
//! pair ([`crate::comp`]), the rule the library keeps where a mean sits at
//! a level: a target held at one value is reached exactly, and its
//! variance stays exactly 0.
//!
//! The band is `eps·√v` as the row arrives, before the row's own `y` joins.
//! Until a spread exists -- fewer than two weighted rows, or every `y` the
//! same -- `v` is 0 and so is the band: every row teaches.
//!
//! **Why the target's spread, not the residual's.** Task 195 drew the band
//! in the EW standard deviation of the residuals, as `huber`'s cut is
//! drawn. A gradient or passive-aggressive fit starts from zero
//! coefficients, so its first residuals are the target's whole level. On a
//! target at 1,000 in a spread of 2 the band they set was about 100 wide,
//! every later row fell inside it, and without decay nothing narrowed it
//! again: `pa` stopped learning for good, R² −52 at a half-life of 1e9
//! against +0.954 at 500. The target's own spread does not read the fit.
//! `huber_delta` stays on the residual's spread (`sgd`, `huber`): outside
//! its cut a Huber gradient is clipped, not zero, so the fit keeps
//! learning, where inside a band the gradient is zero. Task 207 measured
//! the residual's spread capped by the target's beside it: on a target far
//! from zero without decay the cap was all it ever read, the start-up
//! residuals staying in the residual's spread for ever, and the target's
//! spread at 0.01 had the smaller worst regret. In units of the noise a band is `eps / √(1 − R²)`
//! noise standard deviations wide, `R²` the fit's; `pa`'s and `sgd`'s
//! module docs give it per `R²`, with what it costs.

use serde::{Deserialize, Serialize};

use crate::EwDiag;

/// Per target, the EW mean and variance of the target itself: the unit of
/// an insensitivity band (the module docs). Empty for a model whose loss
/// draws no band.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub(crate) struct TargetSpread(Vec<EwDiag>);

impl TargetSpread {
    /// One empty spread per target.
    pub(crate) fn new(n_targets: usize) -> Self {
        Self((0..n_targets).map(|_| EwDiag::new(1)).collect())
    }

    /// No spread at all: a model that draws no band keeps none.
    pub(crate) fn none() -> Self {
        Self(Vec::new())
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// One one-slot spread per target, as a state must hold for `n` of them.
    pub(crate) fn has_shape(&self, n_targets: usize) -> bool {
        self.0.len() == n_targets && self.0.iter().all(|s| s.k() == 1)
    }

    /// Target `j`'s EW variance as it stands: `v` (the module docs).
    pub(crate) fn var(&self, j: usize) -> f64 {
        self.0[j].var(0)
    }

    /// Every target's EW variance as it stands.
    pub(crate) fn vars(&self) -> Vec<f64> {
        (0..self.0.len()).map(|j| self.var(j)).collect()
    }

    /// Target `j`'s band, `eps` of its EW standard deviation: `eps·√v`
    /// where `v` is above 0 and finite, and 0 before -- no spread yet, so
    /// every row teaches.
    pub(crate) fn band(&self, j: usize, eps: f64) -> f64 {
        let v = self.var(j);
        if v > 0.0 && v.is_finite() {
            eps * v.sqrt()
        } else {
            0.0
        }
    }

    /// One row for target `j`: its weight aged by `lam`, and `y` learned at
    /// the row's weight where the row carries the target with a weight
    /// above 0. Otherwise only the weight ages.
    pub(crate) fn update(&mut self, j: usize, y: Option<f64>, lam: f64, weight: f64) {
        match y {
            Some(v) if v.is_finite() && weight > 0.0 => self.0[j].update(&[v], lam, weight),
            _ => self.0[j].update(&[0.0], lam, 0.0),
        }
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

    /// The spread is its definition: the EW variance of the target around
    /// its EW mean, `Σ ωᵢ (yᵢ − m)² / Σ ωᵢ` with `m = Σ ωᵢ yᵢ / Σ ωᵢ`,
    /// rebuilt from the whole history at every row, two passes over it, `ωᵢ`
    /// row `i`'s weight aged by every decay since -- not the recursion. Over
    /// irregular clock steps, irregular weights, null targets and rows of
    /// weight 0, the first row's among them, at a level of 1,000.
    #[test]
    fn the_spread_is_the_ew_variance_of_the_target_around_its_ew_mean() {
        let mut sp = TargetSpread::new(1);
        let mut seen: Vec<(f64, f64)> = Vec::new();
        let mut s = 7u64;
        for i in 0..300 {
            let y = (i % 7 != 3).then(|| 1000.0 + 2.0 * lcg(&mut s));
            let w = if i % 11 == 0 {
                0.0
            } else {
                0.5 + lcg(&mut s).abs()
            };
            let lam = crate::Decay::Halflife(20.0).factor(if i == 0 {
                0.0
            } else {
                0.5 + 2.0 * lcg(&mut s).abs()
            });
            sp.update(0, y, lam, w);
            for (_, wi) in seen.iter_mut() {
                *wi *= lam;
            }
            if let Some(v) = y
                && w > 0.0
            {
                seen.push((v, w));
            }
            let total: f64 = seen.iter().map(|(_, wi)| wi).sum();
            let want = if seen.len() < 2 {
                0.0
            } else {
                let mean = seen.iter().map(|(v, wi)| v * wi).sum::<f64>() / total;
                seen.iter()
                    .map(|(v, wi)| wi * (v - mean) * (v - mean))
                    .sum::<f64>()
                    / total
            };
            let got = sp.var(0);
            assert!(
                (got - want).abs() <= 1e-9 * want + 1e-300,
                "row {i}: v {got} against {want}"
            );
            assert_eq!(sp.band(0, 0.1), 0.1 * got.sqrt(), "row {i}");
        }
        assert!(seen.len() > 200, "{} rows taught it", seen.len());
    }

    /// No spread before two weighted rows: one row has no spread to give,
    /// and neither has a target held at one value, at a level of 1e8, for
    /// any number of rows -- its mean reaches the value, and its variance
    /// stays exactly 0 -- so the band is 0. Rows of weight 0 and rows
    /// without the target add nothing; a second value opens it.
    #[test]
    fn a_band_needs_two_weighted_rows_of_different_values() {
        let mut sp = TargetSpread::new(2);
        sp.update(0, Some(1e8 + 0.37), 1.0, 0.0);
        sp.update(1, None, 1.0, 1.0);
        assert_eq!((sp.var(0), sp.band(0, 0.1)), (0.0, 0.0), "a zero weight");
        for _ in 0..5000 {
            sp.update(0, Some(1e8 + 0.37), 0.97, 1.0);
            sp.update(1, None, 0.97, 1.0);
            sp.update(0, Some(5.0), 0.97, 0.0);
            assert_eq!((sp.var(0), sp.band(0, 0.1)), (0.0, 0.0));
        }
        assert_eq!(sp.vars(), vec![0.0, 0.0]);
        sp.update(0, Some(1e8 + 2.37), 0.97, 1.0);
        assert!(sp.band(0, 0.1) > 0.0, "a second value");
        assert_eq!(sp.band(0, 0.0), 0.0, "eps 0 draws no band");
        assert_eq!(sp.var(1), 0.0, "a target never carried");
    }

    /// A band is free of the target's level: a stream and the same stream
    /// shifted by 1,000 or by 1e6 draw the same band at every row, to the
    /// rounding of the shifted values themselves.
    #[test]
    fn a_band_is_free_of_a_shift() {
        for shift in [1000.0, 1e6] {
            let (mut a, mut b) = (TargetSpread::new(1), TargetSpread::new(1));
            let mut s = 9u64;
            for i in 0..500 {
                let y = 2.0 * lcg(&mut s);
                let lam = if i == 0 { 1.0 } else { 0.99 };
                a.update(0, Some(y), lam, 1.0);
                b.update(0, Some(y + shift), lam, 1.0);
                let (ba, bb) = (a.band(0, 0.1), b.band(0, 0.1));
                assert!(
                    (ba - bb).abs() <= 1e-9 * ba + 1e-300,
                    "shift {shift}, row {i}: {ba} against {bb}"
                );
            }
        }
    }
}
