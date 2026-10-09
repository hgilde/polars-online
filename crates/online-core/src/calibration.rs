//! `Calibration`: Mincer and Zarnowitz's (1969) regression of the outcome on
//! its prediction, kept as exponentially weighted moments with a memory of
//! its own (docs/PLAN.md task 221 (a)).
//!
//! A prediction is calibrated when `E[y | pred] = pred`: the regression
//! `y = a + b·pred + e` has `a = 0` and `b = 1`. The slope is the multiplier
//! to put on a prediction, and the intercept what to add after it. Each
//! row's `(pred, y)` joins EW means and centred co-moments, the weighted
//! Welford recursion of [`crate::EwCov`] with weights `ω`, the row's weight
//! times the decay since it:
//!
//! ```text
//! W'  = lam·W + w            Q' = lam²·Q + w²        n_kish = W² / Q
//! m_p, m_y                   the EW means
//! C_pp, C_py, C_yy           the EW centred co-moments (mean form, over W)
//! b   = C_py / C_pp          a = m_y − b·m_p
//! s²  = C_yy − C_py² / C_pp  the residual's EW mean square
//! ```
//!
//! The joint test of `a = 0, b = 1` is Wald's, read at Kish's effective
//! sample size in place of `n`:
//!
//! ```text
//! wald = (n_kish − 2) · ((m_y − m_p)² + (b − 1)²·C_pp) / s²
//! ```
//!
//! — `d'·M·d / (s²/(n_kish − 2))` with `d = (a, b − 1)` and
//! `M = [[1, m_p], [m_p, C_pp + m_p²]]` the EW second moments of `(1, pred)`;
//! `a + (b − 1)·m_p` is `m_y − m_p`, the mean residual, which makes the
//! form one subtraction short of cancellation. With no memory (`lam = 1`)
//! and unit weights `n_kish = n`, and `wald / 2` is the F statistic of the
//! ordinary least squares test (statsmodels' `f_test`), `F(2, n − 2)` under
//! Gaussian errors and `wald` itself `χ²(2)` as `n` grows. With a memory the
//! weights are uneven and the variance of a weighted mean is `1/n_kish` of
//! one row's, so the same form is approximately `χ²(2)` on each row; the
//! rows' values are then correlated over about a half-life.
//!
//! **Under a horizon** (a target that looks ahead `h` rows; task 232 (3))
//! the residuals overlap and the form above reads them as independent, so
//! the test is Wald's with Newey and West's variance in place of `s²/n`:
//! `wald = d' V⁻¹ d` with `V` the sandwich [`crate::Sandwich`] of the
//! regression on `(1, pred)` at `L = 2h` lags, its meat built from the
//! residual under the null, `y − pred` (`a = 0, b = 1` leave nothing else),
//! so it needs no fit of its own.

use serde::{Deserialize, Serialize};

/// Mincer–Zarnowitz calibration of one prediction slot. See the module docs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Calibration {
    /// The EW moments of `(pred, y)`, never windowed: no runs.
    joint: crate::EwCov,
    /// Under a horizon, the sandwich of the regression on `(1, pred)` at
    /// Newey and West's lags, its meat from `y − pred` (module docs).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    hac: Option<crate::Sandwich>,
}

impl Default for Calibration {
    fn default() -> Self {
        Self::new()
    }
}

impl Calibration {
    pub fn new() -> Self {
        Self::with_lags(0)
    }

    /// A calibration whose Wald statistic reads Newey and West's variance
    /// at `lags` lags (module docs), the plain form at 0.
    pub fn with_lags(lags: usize) -> Self {
        Self {
            joint: crate::EwCov::new(2).without_runs(),
            hac: (lags > 0).then(|| crate::Sandwich::new(1, true, lags)),
        }
    }

    /// Whether a restored accumulator is shaped as a fresh one at `lags`:
    /// its moments two wide, with the sum of squared weights Kish's size
    /// reads, and the sandwich at those lags.
    pub fn has_shape(&self, lags: usize) -> bool {
        self.joint.has_shape(2)
            && self.joint.q_sum().is_some()
            && match &self.hac {
                None => lags == 0,
                Some(h) => lags > 0 && h.has_shape(1, true, lags),
            }
    }

    /// The rows behind this one are no longer adjacent to it -- a capped gap
    /// or a session change -- so none is a lag of the next.
    pub fn clear_lags(&mut self) {
        if let Some(h) = self.hac.as_mut() {
            h.clear_lags();
        }
    }

    /// The EW weight of the scored rows.
    pub fn n_eff(&self) -> f64 {
        self.joint.n_eff()
    }

    /// Kish's effective sample size `W²/Q`, `None` before the first scored
    /// row of positive weight.
    pub fn n_kish(&self) -> Option<f64> {
        self.joint.n_kish()
    }

    /// The clock moves on by a step whose decay is `lam`, with nothing
    /// scored: the means stay, the weights shrink.
    pub fn age(&mut self, lam: f64) {
        self.joint.decay(lam);
        if let Some(h) = self.hac.as_mut() {
            h.age(lam);
        }
    }

    /// One scored row: its prediction, its outcome, the step's decay and its
    /// weight. A row with either value not finite, or of weight 0, only ages
    /// the moments (CLAUDE.md hard rule 9): it is not evidence about the
    /// calibration, and a zero-weight first row would be `0/0`.
    pub fn update(&mut self, pred: f64, y: f64, lam: f64, w: f64) {
        if !(pred.is_finite() && y.is_finite() && w > 0.0) {
            return self.age(lam);
        }
        self.joint.update(&[pred, y], lam, w);
        if let Some(h) = self.hac.as_mut() {
            h.update(&[pred], y - pred, lam, w);
        }
    }

    /// `b`, the slope of `y` on `pred`: `None` until the predictions have a
    /// spread.
    pub fn slope(&self) -> Option<f64> {
        let var_p = self.joint.var(0);
        (var_p > 0.0 && self.n_eff() > 0.0).then(|| self.joint.cov(0, 1) / var_p)
    }

    /// `a`, the intercept: `m_y − b·m_p`, where the slope is.
    pub fn intercept(&self) -> Option<f64> {
        self.slope()
            .map(|b| self.joint.mean(1) - b * self.joint.mean(0))
    }

    /// The Wald statistic of `a = 0, b = 1` at Kish's size (module docs).
    /// `None` until there is a slope, more than two rows' worth of Kish's
    /// size, and a residual with a spread: a fit through every point has no
    /// error to scale the distance by.
    pub fn wald(&self) -> Option<f64> {
        let b = self.slope()?;
        let n = self.n_kish()?;
        if n <= 2.0 {
            return None;
        }
        if let Some(h) = &self.hac {
            let a = self.joint.mean(1) - b * self.joint.mean(0);
            return h.wald(&[0], true, &[0, 1], &[a, b - 1.0]);
        }
        let (c_pp, c_py, c_yy) = (self.joint.var(0), self.joint.cov(0, 1), self.joint.var(1));
        let s2 = c_yy - c_py * c_py / c_pp;
        // A NaN spread is no spread either.
        if s2.is_nan() || s2 <= 0.0 {
            return None;
        }
        let mean_resid = self.joint.mean(1) - self.joint.mean(0);
        let d = mean_resid * mean_resid + (b - 1.0) * (b - 1.0) * c_pp;
        Some((n - 2.0) * d / s2)
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

    /// The weighted least squares fit of `y` on `(1, pred)` and its Wald
    /// statistic at Kish's size, from every row and its weight as it now
    /// stands: the normal equations solved by faer's LU (`crate::oracle`),
    /// the residual summed row by row. No moment of the accumulator's.
    fn by_definition(rows: &[(f64, f64, f64)]) -> (f64, f64, f64) {
        let (mut m, mut r) = (vec![0.0; 4], vec![0.0; 2]);
        let (mut sw, mut sw2) = (0.0, 0.0);
        for &(p, y, w) in rows {
            m[0] += w;
            m[1] += w * p;
            m[2] += w * p;
            m[3] += w * p * p;
            r[0] += w * y;
            r[1] += w * p * y;
            sw += w;
            sw2 += w * w;
        }
        let beta = crate::oracle::solve(&m, &r);
        let ssr: f64 = rows
            .iter()
            .map(|&(p, y, w)| w * (y - beta[0] - beta[1] * p).powi(2))
            .sum();
        let n_kish = sw * sw / sw2;
        let s2 = ssr / sw * n_kish / (n_kish - 2.0);
        let d = [beta[0], beta[1] - 1.0];
        // `d'·(X'ΩX / W)·d`, the EW second moments of `(1, pred)` summed
        // directly.
        let quad: f64 = (0..2)
            .flat_map(|i| (0..2).map(move |j| (i, j)))
            .map(|(i, j)| d[i] * m[i * 2 + j] / sw * d[j])
            .sum();
        (beta[0], beta[1], n_kish * quad / s2)
    }

    /// Slope, intercept and Wald against the definition, row by row, under a
    /// decay, uneven weights with zeros among them, and a miscalibrated
    /// prediction, `y = 0.3 + 0.7·pred + e`.
    #[test]
    fn calibration_matches_weighted_least_squares_at_kish_size() {
        let mut c = Calibration::new();
        let mut s = 7u64;
        let mut rows: Vec<(f64, f64, f64)> = Vec::new();
        let lam = 0.97;
        for i in 0..400 {
            let p = 2.0 * lcg(&mut s) + 5.0;
            let y = 0.3 + 0.7 * p + 0.4 * lcg(&mut s);
            let w = if i % 11 == 3 {
                0.0
            } else {
                0.5 + lcg(&mut s) + 0.5
            };
            c.update(p, y, lam, w);
            rows.iter_mut().for_each(|r| r.2 *= lam);
            rows.push((p, y, w));
            if i < 5 {
                continue;
            }
            let (a, b, wald) = by_definition(&rows);
            let tol = 1e-9;
            assert!((c.slope().unwrap() - b).abs() < tol, "row {i}");
            assert!((c.intercept().unwrap() - a).abs() < tol, "row {i}");
            let got = c.wald().unwrap();
            assert!(
                (got - wald).abs() < tol * wald.max(1.0),
                "row {i}: {got} vs {wald}"
            );
        }
        let (a, b) = (c.intercept().unwrap(), c.slope().unwrap());
        assert!((b - 0.7).abs() < 0.05 && (a - 0.3).abs() < 0.3, "{a} {b}");
        assert!(c.wald().unwrap() > 100.0, "a slope of 0.7 is far from 1");
    }

    /// No memory and unit weights: Kish's size is the row count, and the
    /// statistic is twice the ordinary least squares F (the Python suite
    /// holds it to statsmodels' `f_test`).
    #[test]
    fn without_memory_kish_size_is_the_row_count() {
        let mut c = Calibration::new();
        let mut s = 3u64;
        for i in 0..50 {
            let p = lcg(&mut s);
            c.update(p, p + 0.1 * lcg(&mut s), 1.0, 1.0);
            assert!((c.n_kish().unwrap() - (i + 1) as f64).abs() < 1e-9);
        }
    }

    /// Under a horizon, Wald's statistic of `a = 0, b = 1` with Newey and
    /// West's variance by its definition: `B = Σ ω z z'`, `z = (1, pred)`,
    /// the meat's double sum of `u_t = ω_t (y_t − pred_t) z_t` over pairs at
    /// most `L` rows apart at Bartlett's weights, `B⁻¹ S B⁻¹` and its
    /// inverse by faer's LU (`crate::oracle`), under a decay, uneven weights
    /// and zeros, with residuals that carry a lag.
    #[test]
    fn under_a_horizon_the_wald_is_newey_and_wests() {
        let lags = 4usize;
        let lam = 0.995;
        let mut c = Calibration::with_lags(lags);
        let mut s = 31u64;
        // (pred, y, weight now), folded rows only.
        let mut rows: Vec<(f64, f64, f64)> = Vec::new();
        let mut prev = 0.0;
        for i in 0..300 {
            let p = 2.0 * lcg(&mut s) + 5.0;
            let shock = 0.6 * prev + lcg(&mut s);
            prev = shock;
            let y = 0.2 + 0.9 * p + shock;
            let w = if i % 13 == 6 { 0.0 } else { 1.0 + lcg(&mut s) };
            c.update(p, y, lam, w);
            rows.iter_mut().for_each(|r| r.2 *= lam);
            if w > 0.0 {
                rows.push((p, y, w));
            }
            if i < 30 || i % 41 != 0 {
                continue;
            }
            let mut bread = [0.0; 4];
            let mut meat = [0.0; 4];
            let u: Vec<[f64; 2]> = rows
                .iter()
                .map(|&(p, y, w)| [w * (y - p), w * (y - p) * p])
                .collect();
            for &(p, _, w) in &rows {
                let z = [1.0, p];
                for a in 0..2 {
                    for b in 0..2 {
                        bread[a * 2 + b] += w * z[a] * z[b];
                    }
                }
            }
            for t in 0..u.len() {
                for l in 0..=lags.min(t) {
                    let bw = if l == 0 {
                        1.0
                    } else {
                        1.0 - l as f64 / (lags as f64 + 1.0)
                    };
                    for a in 0..2 {
                        for b in 0..2 {
                            let pair = u[t][a] * u[t - l][b];
                            let back = u[t - l][a] * u[t][b];
                            meat[a * 2 + b] += if l == 0 { pair } else { bw * (pair + back) };
                        }
                    }
                }
            }
            let bi = crate::oracle::inverse(&bread);
            let mul = |x: &[f64], y: &[f64]| -> Vec<f64> {
                (0..4)
                    .map(|n| (0..2).map(|k| x[(n / 2) * 2 + k] * y[k * 2 + n % 2]).sum())
                    .collect()
            };
            let v = mul(&mul(&bi, &meat), &bi);
            let d = [c.intercept().unwrap(), c.slope().unwrap() - 1.0];
            let want = crate::oracle::quad_form(&v, &d);
            let got = c.wald().unwrap();
            assert!(
                (got - want).abs() < 1e-7 * want.max(1.0),
                "row {i}: {got} vs {want}"
            );
        }
        assert!(c.has_shape(lags) && !c.has_shape(0));
    }

    /// A zero-weight row, the first included, and a row with a value not a
    /// number only age the moments; until two rows with a spread there is
    /// no slope, and with no residual spread no statistic.
    #[test]
    fn zero_weight_and_missing_rows_only_age() {
        let mut c = Calibration::new();
        c.update(1.0, 2.0, 0.9, 0.0);
        assert_eq!(c.n_eff(), 0.0);
        assert!(c.slope().is_none() && c.wald().is_none() && c.n_kish().is_none());
        c.update(f64::NAN, 2.0, 0.9, 1.0);
        c.update(1.0, f64::INFINITY, 0.9, 1.0);
        assert_eq!(c.n_eff(), 0.0);
        c.update(1.0, 2.0, 0.9, 1.0);
        assert!(c.slope().is_none(), "one row has no spread");
        let before = c.clone();
        c.update(3.0, 1.0, 0.5, 0.0);
        assert_eq!(c.n_eff(), before.n_eff() * 0.5);
        assert_eq!(c.joint.mean(0), before.joint.mean(0));
        // Points on a line: a slope, and no residual to scale the test by.
        let mut line = Calibration::new();
        for p in [1.0, 2.0, 3.0, 4.0] {
            line.update(p, 2.0 * p, 1.0, 1.0);
        }
        assert!((line.slope().unwrap() - 2.0).abs() < 1e-12);
        assert!(line.wald().is_none());
        assert!(line.has_shape(0));
    }
}
