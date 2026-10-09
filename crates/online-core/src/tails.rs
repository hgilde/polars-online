//! `Tails`: the exponentially weighted skewness, kurtosis and Jarque and
//! Bera's (1980) statistic of a slot's recursive residuals, with a memory of
//! their own (docs/PLAN.md task 221 (e)).
//!
//! The recursive residual `v = resid / error_inflation` (see
//! [`crate::Breaks`]) is the residual with the fit's estimation error taken
//! out; skewness and kurtosis are ratios of its moments, so its scale does
//! not enter them. With `ω` each scored row's weight times the decay since
//! it, the power sums `S_j = Σ ω (v − c)^j` about an origin `c` (the first
//! residual, which keeps them small) give the central moments
//!
//! ```text
//! μ  = S_1/S_0      m_2 = S_2/S_0 − μ²
//! m_3 = S_3/S_0 − 3μ S_2/S_0 + 2μ³
//! m_4 = S_4/S_0 − 4μ S_3/S_0 + 6μ² S_2/S_0 − 3μ⁴
//! skew = m_3 / m_2^{3/2}       kurtosis = m_4 / m_2² − 3   (excess, as Polars')
//! JB   = n/6 · (skew² + kurtosis²/4)                       ~ χ²(2)
//! ```
//!
//! `n` Kish's size. Run once with unit weights these are the biased
//! moments of `scipy.stats.skew` and `kurtosis` and `statsmodels`'
//! `jarque_bera`.

use serde::{Deserialize, Serialize};

/// The tails of one slot's recursive residuals. See the module docs.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Tails {
    /// `S_0 .. S_4`, and `Σ ω²` for Kish's size.
    s: [f64; 5],
    w2: f64,
    origin: Option<f64>,
}

impl Tails {
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether a restored accumulator can be read and folded.
    pub fn has_shape(&self) -> bool {
        self.s.iter().chain([&self.w2]).all(|v| v.is_finite())
            && self.s[0] >= 0.0
            && self.w2 >= 0.0
            && self.origin.is_none_or(f64::is_finite)
    }

    /// The clock moves on by a step whose decay is `lam`, with nothing
    /// scored.
    pub fn age(&mut self, lam: f64) {
        self.s.iter_mut().for_each(|v| *v *= lam);
        self.w2 *= lam * lam;
    }

    /// One recursive residual `v` at weight `w`, after the step's decay
    /// `lam`. A `v` that is not finite, or a weight of 0, only ages
    /// (CLAUDE.md hard rule 9).
    pub fn update(&mut self, v: f64, lam: f64, w: f64) {
        self.age(lam);
        if !(v.is_finite() && w > 0.0) {
            return;
        }
        let d = v - *self.origin.get_or_insert(v);
        let mut p = w;
        for s in &mut self.s {
            *s += p;
            p *= d;
        }
        self.w2 += w * w;
    }

    /// The central moments `(m_2, m_3, m_4)`; `None` before a spread.
    fn moments(&self) -> Option<(f64, f64, f64)> {
        let w = self.s[0];
        if w <= 0.0 {
            return None;
        }
        let [_, a, b, c, d] = self.s.map(|v| v / w);
        let mu = a;
        let m2 = b - mu * mu;
        if m2.is_nan() || m2 <= 0.0 {
            return None;
        }
        let m3 = c - 3.0 * mu * b + 2.0 * mu.powi(3);
        let m4 = d - 4.0 * mu * c + 6.0 * mu * mu * b - 3.0 * mu.powi(4);
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
        let n = self.s[0] * self.s[0] / self.w2;
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

    #[test]
    fn zero_weights_and_missing_values_only_age() {
        let mut t = Tails::new();
        t.update(1.0, 0.9, 0.0);
        t.update(f64::NAN, 0.9, 1.0);
        assert_eq!(t, Tails::new());
        t.update(1.0, 0.9, 1.0);
        assert!(t.skew().is_none(), "one value has no spread");
    }
}
