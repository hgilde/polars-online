//! `Tails`: the exponentially weighted skewness, kurtosis and Jarque and
//! Bera's (1980) statistic of a slot's recursive residuals, with a memory of
//! their own (docs/PLAN.md task 221 (e)).
//!
//! The recursive residual `v = resid / error_inflation` (see
//! [`crate::Breaks`]) is the residual with the fit's estimation error taken
//! out; skewness and kurtosis are ratios of its moments, so its scale does
//! not enter them. With `ω` each scored row's weight times the decay since
//! it, `W = Σω`, the weighted mean `μ` and the central sums
//! `M_j = Σ ω (v − μ)^j` give the central moments `m_j = M_j / W` and
//!
//! ```text
//! skew = m_3 / m_2^{3/2}       kurtosis = m_4 / m_2² − 3   (excess, as Polars')
//! JB   = n/6 · (skew² + kurtosis²/4)                       ~ χ²(2)
//! ```
//!
//! The sums are kept central by Pébay's weighted one-pass updates (Pébay,
//! Terriberry, Kolla and Bennett 2016): a decay scales `W` and each `M_j`
//! and leaves `μ`, and a row `v` at weight `w` merges the set so far with a
//! set of one. With `δ = v − μ`, `W' = W + w`, `q = W/W'` and `r = w/W'`,
//!
//! ```text
//! M_4' = M_4 + δ⁴ W r (q² − q r + r²) + 6 δ² r² M_2 − 4 δ r M_3
//! M_3' = M_3 + δ³ W r (q − r) − 3 δ r M_2
//! M_2' = M_2 + δ² W r                    μ' = μ + δ r
//! ```
//!
//! Raw power sums about a fixed origin, the first residual, cancelled
//! catastrophically once that origin sat far from where the residuals
//! settled: after a first residual of `+1e5`, a memory of 50 rows read a
//! kurtosis of −129,065 where two passes read 0.99 (review round 6, B-5).
//!
//! `n` Kish's size. Run once with unit weights these are the biased
//! moments of `scipy.stats.skew` and `kurtosis` and `statsmodels`'
//! `jarque_bera`.

use serde::{Deserialize, Serialize};

/// The tails of one slot's recursive residuals. See the module docs.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Tails {
    /// `W = Σω`, and `Σω²` for Kish's size.
    w: f64,
    w2: f64,
    /// The weighted mean `μ`.
    mean: f64,
    /// The central sums `M_2`, `M_3`, `M_4`.
    m: [f64; 3],
}

impl Tails {
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether a restored accumulator can be read and folded.
    pub fn has_shape(&self) -> bool {
        [self.w, self.w2, self.mean]
            .iter()
            .chain(&self.m)
            .all(|v| v.is_finite())
            && self.w >= 0.0
            && self.w2 >= 0.0
            && self.m[0] >= 0.0
    }

    /// The clock moves on by a step whose decay is `lam`, with nothing
    /// scored.
    pub fn age(&mut self, lam: f64) {
        self.w *= lam;
        self.w2 *= lam * lam;
        self.m.iter_mut().for_each(|v| *v *= lam);
    }

    /// One recursive residual `v` at weight `w`, after the step's decay
    /// `lam`. A `v` that is not finite, or a weight of 0, only ages
    /// (CLAUDE.md hard rule 9).
    pub fn update(&mut self, v: f64, lam: f64, w: f64) {
        self.age(lam);
        if !(v.is_finite() && w > 0.0) {
            return;
        }
        // The merge of the set so far, weight `a`, with the row alone. No
        // term divides by `w`, so a tiny weight cannot overflow one; at
        // `a = 0` (the first row, or a memory decayed to nothing) every
        // term but the mean's is 0 and the mean is `v`.
        let a = self.w;
        let total = a + w;
        let (q, r) = (a / total, w / total);
        let d = v - self.mean;
        let (d2, ar) = (d * d, a * r);
        let [m2, m3, m4] = self.m;
        self.m = [
            m2 + d2 * ar,
            m3 + d2 * d * ar * (q - r) - 3.0 * d * r * m2,
            m4 + d2 * d2 * ar * (q * q - q * r + r * r) + 6.0 * d2 * r * r * m2 - 4.0 * d * r * m3,
        ];
        self.mean += d * r;
        self.w = total;
        self.w2 += w * w;
    }

    /// The central moments `(m_2, m_3, m_4)`; `None` before a spread.
    fn moments(&self) -> Option<(f64, f64, f64)> {
        let w = self.w;
        if w <= 0.0 {
            return None;
        }
        let [m2, m3, m4] = self.m.map(|v| v / w);
        if m2.is_nan() || m2 <= 0.0 {
            return None;
        }
        Some((m2, m3, m4))
    }

    /// The skewness, `m_3 / m_2^{3/2}`.
    pub fn skew(&self) -> Option<f64> {
        self.moments().map(|(m2, m3, _)| m3 / m2.powf(1.5))
    }

    /// The excess kurtosis, `m_4 / m_2² − 3`.
    pub fn kurtosis(&self) -> Option<f64> {
        self.moments().map(|(m2, _, m4)| m4 / (m2 * m2) - 3.0)
    }

    /// Jarque and Bera's statistic at Kish's size: `None` before a spread.
    pub fn jarque_bera(&self) -> Option<f64> {
        let (s, k) = (self.skew()?, self.kurtosis()?);
        let n = self.w * self.w / self.w2;
        Some(n / 6.0 * (s * s + k * k / 4.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    }

    /// Skewness, kurtosis and Jarque and Bera by their definitions, two
    /// passes over every row at its present weight -- the mean, then the
    /// deviations' powers -- under a decay, uneven weights and zeros, on a
    /// skewed stream at a level.
    #[test]
    fn the_tails_are_their_definitions() {
        let mut t = Tails::new();
        let mut st = 17u64;
        let mut rows: Vec<(f64, f64)> = Vec::new();
        let lam = 0.995;
        for i in 0..500 {
            let u = lcg(&mut st) + 0.5;
            let v = 3.0 + u * u * u * 4.0;
            let w = if i % 9 == 1 {
                0.0
            } else {
                0.5 + lcg(&mut st) + 0.5
            };
            t.update(v, lam, w);
            rows.iter_mut().for_each(|r| r.1 *= lam);
            rows.push((v, w));
            if i < 10 || i % 41 != 0 {
                continue;
            }
            let sw: f64 = rows.iter().map(|r| r.1).sum();
            let sw2: f64 = rows.iter().map(|r| r.1 * r.1).sum();
            let mu = rows.iter().map(|r| r.1 * r.0).sum::<f64>() / sw;
            let m = |j: i32| rows.iter().map(|r| r.1 * (r.0 - mu).powi(j)).sum::<f64>() / sw;
            let skew = m(3) / m(2).powf(1.5);
            let kurt = m(4) / m(2).powi(2) - 3.0;
            let jb = sw * sw / sw2 / 6.0 * (skew * skew + kurt * kurt / 4.0);
            assert!((t.skew().unwrap() - skew).abs() < 1e-9, "row {i}");
            assert!((t.kurtosis().unwrap() - kurt).abs() < 1e-8, "row {i}");
            assert!(
                (t.jarque_bera().unwrap() - jb).abs() < 1e-7 * jb.max(1.0),
                "row {i}"
            );
        }
        assert!(t.skew().unwrap() > 0.5, "a skewed stream");
        assert!(t.has_shape());
    }

    /// A first residual far from the rest -- a bad first print -- leaves the
    /// moments as a two-pass reading gives them, whether the memory keeps it
    /// or has all but forgotten it (review round 6, B-5). Raw power sums
    /// about that first residual read a kurtosis of −129,065 at `+1e5`
    /// where the two passes read 0.99.
    #[test]
    fn a_far_first_residual_does_not_cancel_the_moments() {
        for (first, lam) in [
            (1e3, 0.986),
            (1e4, 0.986),
            (1e5, 0.986),
            (1e5, 1.0),
            (1e7, 1.0),
        ] {
            let mut t = Tails::new();
            let mut st = 23u64;
            let mut rows: Vec<(f64, f64)> = Vec::new();
            for i in 0..3_000 {
                let v = if i == 0 {
                    first
                } else {
                    lcg(&mut st) + lcg(&mut st) + lcg(&mut st)
                };
                let w = 0.5 + lcg(&mut st) + 0.5;
                t.update(v, lam, w);
                rows.iter_mut().for_each(|r| r.1 *= lam);
                rows.push((v, w));
            }
            let sw: f64 = rows.iter().map(|r| r.1).sum();
            let mu = rows.iter().map(|r| r.1 * r.0).sum::<f64>() / sw;
            let m = |j: i32| rows.iter().map(|r| r.1 * (r.0 - mu).powi(j)).sum::<f64>() / sw;
            let skew = m(3) / m(2).powf(1.5);
            let kurt = m(4) / m(2).powi(2) - 3.0;
            let (gs, gk) = (t.skew().unwrap(), t.kurtosis().unwrap());
            assert!(
                (gs - skew).abs() < 1e-6 * skew.abs().max(1.0),
                "first {first}, lam {lam}: skew {gs} vs {skew}"
            );
            assert!(
                (gk - kurt).abs() < 1e-6 * kurt.abs().max(1.0),
                "first {first}, lam {lam}: kurtosis {gk} vs {kurt}"
            );
        }
    }

    #[test]
    fn zero_weights_and_missing_values_only_age() {
        let mut t = Tails::new();
        t.update(1.0, 0.9, 0.0);
        t.update(f64::NAN, 0.9, 1.0);
        assert_eq!(t, Tails::new());
        t.update(1.0, 0.9, 1.0);
        assert!(t.skew().is_none(), "one value has no spread");
        // A weight too small to square, beside rows of weight 1, leaves
        // every sum finite: no term divides by it.
        t.update(1e5, 0.9, 1e-300);
        t.update(-1.0, 0.9, 1.0);
        t.update(0.5, 0.9, 1.0);
        assert!(t.has_shape() && t.kurtosis().is_some_and(f64::is_finite));
    }
}
