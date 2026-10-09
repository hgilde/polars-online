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
    /// spread, and where either is missing or the inflation is not finite
    /// or below 1.
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

    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    }

    /// Row by row against the definition, from every folded row at its
    /// present weight -- `(v / s) · sqrt(h)`, `h = infl² − 1`, `s² = Σ ω v²
    /// / Σ ω` over the rows before -- under a decay, uneven weights with
    /// zeros among them, and inflations away from 2, where `infl²` and
    /// `2 infl` agree.
    #[test]
    fn influence_is_its_definition_row_by_row() {
        let mut inf = Influence::new();
        let mut s = 13u64;
        // (v, weight now) of the folded rows.
        let mut rows: Vec<(f64, f64)> = Vec::new();
        let lam = 0.95;
        for i in 0..200 {
            let e = 3.0 * lcg(&mut s);
            let infl = 1.0 + 1.5 * (lcg(&mut s) + 0.5);
            let (sw, sq) = rows
                .iter()
                .fold((0.0, 0.0), |a, r| (a.0 + r.1, a.1 + r.1 * r.0 * r.0));
            let got = inf.of(e, infl);
            if sw > 0.0 {
                let h = infl * infl - 1.0;
                let want = e / infl / (sq / sw).sqrt() * h.sqrt();
                let got = got.unwrap();
                assert!((got - want).abs() < 1e-12 * want.abs().max(1.0), "row {i}");
            } else {
                assert!(got.is_none(), "row {i}");
            }
            let w = if i % 7 == 2 { 0.0 } else { 1.0 + lcg(&mut s) };
            inf.update(e, infl, lam, w);
            rows.iter_mut().for_each(|r| r.1 *= lam);
            if w > 0.0 {
                rows.push((e / infl, w));
            }
        }
    }

    /// No influence for a residual or an inflation it cannot read -- not a
    /// number, or an inflation below 1, which no `sqrt(1 + h)` is -- nor
    /// against a spread of 0, where every residual before was 0.
    #[test]
    fn no_influence_without_a_residual_or_a_spread() {
        let mut i = Influence::new();
        for _ in 0..5 {
            i.update(0.0, 1.5, 0.9, 1.0);
        }
        assert!(i.of(1.0, 1.5).is_none(), "a spread of 0");
        i.update(1.0, 1.0, 0.9, 1.0);
        assert!(i.of(1.0, 1.5).is_some());
        assert!(i.of(f64::NAN, 1.5).is_none());
        assert!(i.of(1.0, f64::INFINITY).is_none());
        assert!(i.of(1.0, 0.5).is_none(), "an inflation below 1");
    }

    /// A restored scale is refused unless both sums are finite and not
    /// negative.
    #[test]
    fn a_restored_scale_must_be_finite_and_not_negative() {
        let shaped = |w: f64, q: f64| Influence { w, q }.has_shape();
        assert!(shaped(1.0, 2.0) && shaped(0.0, 0.0));
        assert!(!shaped(1.0, f64::NAN));
        assert!(!shaped(f64::INFINITY, 1.0));
        assert!(!shaped(-1.0, 1.0));
        assert!(!shaped(1.0, -1.0));
    }
}
