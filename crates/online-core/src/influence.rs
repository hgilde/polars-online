//! `Influence`: each row's move of a least-squares fit, in the fit's own
//! metric -- an online DFFITS (Belsley, Kuh and Welsch 1980) -- from its
//! out-of-sample residual (docs/PLAN.md task 221 (f)).
//!
//! A row of leverage `h = x'A⁻¹x` against the fit before it (`A` the Gram
//! the fit came from; `error_inflation = sqrt(1 + h)`) and out-of-sample
//! residual `e` moves the coefficients by `Δβ = A⁻¹ x e / (1 + h)`, and in
//! the fit's metric `Δβ' A Δβ = e² h / (1 + h)²`. Over the spread of the
//! recursive residuals before it,
//!
//! ```text
//! v = e / sqrt(1 + h)          s² = Σ ω v² / Σ ω     (the rows before)
//! influence = (v / s) · sqrt(h)
//! ```
//!
//! so `influence² = Δβ' A Δβ · (1 + h) / s²`. Run once with no ridge, for
//! the newest row of the rows so far, it is DFFITS exactly: that row's
//! in-sample leverage is `h / (1 + h)`, its externally studentized residual
//! `v / s` -- `s²` the fit without it, `RSS / (n − k)`, which the recursive
//! residuals' mean square is -- and DFFITS `t · sqrt(h_ii / (1 − h_ii))`.
//! `statsmodels`' `OLSInfluence.dffits` at the last row of each prefix is
//! the oracle. It is not the in-sample DFFITS of an earlier row, which reads
//! the fit of every row but that one, rows after it included; nor, beside
//! a fit that forgets, a deletion statistic at all: there it is the row's
//! move of the fit it enters.

use serde::{Deserialize, Serialize};

/// One slot's influence scale. See the module docs.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Influence {
    /// `Σ ω` and `Σ ω v²` over the scored rows.
    w: f64,
    q: f64,
}

impl Influence {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn has_shape(&self) -> bool {
        self.w.is_finite() && self.q.is_finite() && self.w >= 0.0 && self.q >= 0.0
    }

    /// The row's influence: its out-of-sample residual `e` and error
    /// inflation `infl`, against the spread before it. `None` before a
    /// spread, and where either is missing or the inflation is not finite.
    pub fn of(&self, e: f64, infl: f64) -> Option<f64> {
        if !(e.is_finite() && infl.is_finite() && infl >= 1.0) || self.w <= 0.0 || self.q <= 0.0 {
            return None;
        }
        let h = (infl * infl - 1.0).max(0.0);
        let v = e / infl;
        Some(v / (self.q / self.w).sqrt() * h.sqrt())
    }

    /// Fold the row's recursive residual into the spread, after the step's
    /// decay `lam`, at weight `w`; one that is not finite, or a weight of 0,
    /// only ages (CLAUDE.md hard rule 9).
    pub fn update(&mut self, e: f64, infl: f64, lam: f64, w: f64) {
        self.w *= lam;
        self.q *= lam;
        if e.is_finite() && infl.is_finite() && infl >= 1.0 && w > 0.0 {
            let v = e / infl;
            self.w += w;
            self.q += w * v * v;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The definition, by hand, and the zero-weight and missing rows.
    #[test]
    fn influence_is_its_definition() {
        let mut i = Influence::new();
        assert!(i.of(1.0, 1.2).is_none(), "no spread yet");
        i.update(1.0, 1.0, 0.9, 0.0);
        i.update(f64::NAN, 1.1, 0.9, 1.0);
        assert_eq!(i, Influence::new());
        i.update(2.0, 2.0f64.sqrt(), 1.0, 1.0); // v = sqrt(2)
        i.update(0.0, 1.0, 1.0, 1.0); // v = 0
        // s² = (2 + 0) / 2 = 1; a row of h = 3, e = 4: v = 2, influence 2·sqrt(3).
        let got = i.of(4.0, 2.0).unwrap();
        assert!((got - 2.0 * 3f64.sqrt()).abs() < 1e-12, "{got}");
        assert!(i.has_shape());
    }
}
