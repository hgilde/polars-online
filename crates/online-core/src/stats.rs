//! Streaming scalar statistics (docs/ENHANCEMENTS.md E23).
//!
//! Two diagnostics that complement `EwCov`'s moments and answer questions a
//! standard deviation cannot:
//!
//! - [`P2Quantile`] — the P² algorithm (Jain & Chambanis, 1985). Tracks a
//!   quantile in **five numbers**, no window and no sorting, which makes
//!   distribution-free intervals affordable on a stream. A residual
//!   distribution with fat tails has a 99th percentile far above `2.33·σ`, and
//!   only a quantile estimate will say so.
//! - [`EwAutoCorr`] — exponentially weighted lag-`k` autocorrelation. Residual
//!   autocorrelation is the classic sign that a model is mis-specified: an
//!   out-of-sample residual stream should look like noise, and does not when a
//!   feature is missing or the decay is too slow.

use serde::{Deserialize, Serialize};

/// P² quantile estimator: five markers, updated per observation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct P2Quantile {
    p: f64,
    /// Marker heights, ascending.
    q: [f64; 5],
    /// Marker positions (1-based, as in the paper).
    n: [f64; 5],
    /// Desired marker positions.
    np: [f64; 5],
    /// Increments for the desired positions.
    dn: [f64; 5],
    count: usize,
}

impl P2Quantile {
    /// `p` is the quantile level in (0, 1).
    pub fn new(p: f64) -> Result<Self, String> {
        if !(0.0..=1.0).contains(&p) || p == 0.0 || p == 1.0 {
            return Err("P2Quantile: p must be in (0, 1)".into());
        }
        Ok(Self {
            p,
            q: [0.0; 5],
            n: [1.0, 2.0, 3.0, 4.0, 5.0],
            np: [1.0, 1.0 + 2.0 * p, 1.0 + 4.0 * p, 3.0 + 2.0 * p, 5.0],
            dn: [0.0, p / 2.0, p, (1.0 + p) / 2.0, 1.0],
            count: 0,
        })
    }

    pub fn count(&self) -> usize {
        self.count
    }

    /// The estimate, or `None` until five observations have been seen.
    pub fn get(&self) -> Option<f64> {
        (self.count >= 5).then_some(self.q[2])
    }

    /// Parabolic prediction, falling back to linear when it would break the
    /// ordering of the markers (the paper's condition).
    fn adjust(&mut self, i: usize, d: f64) {
        let d_sign = if d >= 0.0 { 1.0 } else { -1.0 };
        let (qm, q0, qp) = (self.q[i - 1], self.q[i], self.q[i + 1]);
        let (nm, n0, np_) = (self.n[i - 1], self.n[i], self.n[i + 1]);
        let parabolic = q0
            + d_sign / (np_ - nm)
                * ((n0 - nm + d_sign) * (qp - q0) / (np_ - n0)
                    + (np_ - n0 - d_sign) * (q0 - qm) / (n0 - nm));
        self.q[i] = if qm < parabolic && parabolic < qp {
            parabolic
        } else if d_sign > 0.0 {
            q0 + (qp - q0) / (np_ - n0)
        } else {
            q0 - (qm - q0) / (nm - n0)
        };
        self.n[i] += d_sign;
    }

    pub fn update(&mut self, x: f64) {
        if !x.is_finite() {
            return;
        }
        if self.count < 5 {
            self.q[self.count] = x;
            self.count += 1;
            if self.count == 5 {
                self.q.sort_by(|a, b| a.partial_cmp(b).unwrap());
            }
            return;
        }
        self.count += 1;

        // Which cell does x fall into, and stretch the ends if it is outside.
        let k = if x < self.q[0] {
            self.q[0] = x;
            0
        } else if x >= self.q[4] {
            self.q[4] = x;
            3
        } else {
            (0..4)
                .find(|&i| self.q[i] <= x && x < self.q[i + 1])
                .unwrap_or(3)
        };

        for i in (k + 1)..5 {
            self.n[i] += 1.0;
        }
        for i in 0..5 {
            self.np[i] += self.dn[i];
        }

        for i in 1..4 {
            let d = self.np[i] - self.n[i];
            if (d >= 1.0 && self.n[i + 1] - self.n[i] > 1.0)
                || (d <= -1.0 && self.n[i - 1] - self.n[i] < -1.0)
            {
                self.adjust(i, d);
            }
        }
    }
}

/// Exponentially weighted lag-`k` autocorrelation of a stream.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EwAutoCorr {
    lag: usize,
    /// Recent values, most recent last; length `lag + 1` once warm.
    buf: Vec<f64>,
    w: f64,
    mean: f64,
    /// EW second moments of (x_t, x_{t-lag}) around their shared mean.
    var: f64,
    cross: f64,
}

impl EwAutoCorr {
    pub fn new(lag: usize) -> Result<Self, String> {
        if lag == 0 {
            return Err("EwAutoCorr: lag must be >= 1".into());
        }
        Ok(Self {
            lag,
            buf: Vec::with_capacity(lag + 1),
            w: 0.0,
            mean: 0.0,
            var: 0.0,
            cross: 0.0,
        })
    }

    /// `None` until a lagged pair has been seen.
    pub fn get(&self) -> Option<f64> {
        (self.w > 0.0 && self.var > 0.0).then(|| (self.cross / self.var).clamp(-1.0, 1.0))
    }

    /// One observation with decay factor `lam`.
    ///
    /// A single mean and variance are used for both legs of the pair, which is
    /// the standard simplification for a stationary series and keeps the result
    /// in [−1, 1] by construction.
    pub fn update(&mut self, x: f64, lam: f64) {
        if !x.is_finite() {
            return;
        }
        self.buf.push(x);
        if self.buf.len() > self.lag + 1 {
            self.buf.remove(0);
        }

        let w_new = lam * self.w + 1.0;
        let (a, b) = (lam * self.w / w_new, 1.0 / w_new);
        let d = x - self.mean;
        self.var = a * self.var + a * b * d * d;
        if self.buf.len() == self.lag + 1 {
            let lagged = self.buf[0];
            self.cross = a * self.cross + a * b * d * (lagged - self.mean);
        } else {
            self.cross *= a;
        }
        self.mean += b * d;
        self.w = w_new;
    }
}

/// Exponentially weighted evaluation metrics for one prediction slot
/// (docs/ENHANCEMENTS.md E22).
///
/// `eval.py` computes the same quantities in Polars over collected output,
/// which is right for analysis but needs the whole frame. This is the O(state)
/// version: it lives beside the model, so a long-running stream and the CLI can
/// report how the fit is doing without keeping the rows.
///
/// All three are exponentially weighted on the model's own clock:
///
/// ```text
/// ic       = corr(pred, y)
/// r2       = 1 − EW[(y − pred)²] / EW[(y − ȳ)²]
/// hit_rate = EW mean of 1{sign(pred) = sign(y)}, over rows where y ≠ 0
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SlotMetrics {
    /// Joint moments of (pred, y).
    joint: crate::EwCov,
    /// EW mean squared error and its weight.
    mse: f64,
    /// EW hit rate and its weight (rows with `y == 0` are excluded).
    hits: f64,
    hit_w: f64,
}

impl Default for SlotMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl SlotMetrics {
    pub fn new() -> Self {
        Self {
            joint: crate::EwCov::new(2),
            mse: 0.0,
            hits: 0.0,
            hit_w: 0.0,
        }
    }

    /// EW count of scored rows.
    pub fn n_eff(&self) -> f64 {
        self.joint.n_eff()
    }

    /// Information coefficient: the correlation between prediction and target.
    pub fn ic(&self) -> Option<f64> {
        let d = self.joint.var(0).sqrt() * self.joint.var(1).sqrt();
        (d > 0.0).then(|| (self.joint.cov(0, 1) / d).clamp(-1.0, 1.0))
    }

    /// Out-of-sample R², against the EW mean of the target.
    ///
    /// Negative values are normal and meaningful: they say the model is doing
    /// worse than predicting the running mean.
    pub fn r2(&self) -> Option<f64> {
        let var_y = self.joint.var(1);
        (var_y > 0.0).then(|| 1.0 - self.mse / var_y)
    }

    /// Fraction of rows where the prediction had the target's sign.
    pub fn hit_rate(&self) -> Option<f64> {
        (self.hit_w > 0.0).then_some(self.hits)
    }

    /// Score one row. `lam` is the model's decay factor for this row.
    pub fn update(&mut self, pred: f64, y: f64, lam: f64, w: f64) {
        if !pred.is_finite() || !y.is_finite() || w <= 0.0 {
            // Age the estimates but do not score: a row with no prediction is
            // not evidence of a bad one. The means themselves are unchanged;
            // only the effective counts shrink.
            self.joint.decay(lam);
            self.hit_w *= lam;
            return;
        }
        self.joint.update(&[pred, y], lam, w);

        // Reuse the joint accumulator's own weights for the MSE, so every
        // metric here is averaged identically. After the update above,
        // `n_eff = lam·W_old + w`, so `(n_eff − w)/n_eff` is exactly the weight
        // EwCov gave the history and `w/n_eff` the weight it gave this row.
        let denom = self.joint.n_eff();
        if denom > 0.0 {
            let e = y - pred;
            self.mse = ((denom - w) * self.mse + w * e * e) / denom;
        }
        if y != 0.0 {
            let hw = lam * self.hit_w + w;
            let hit = f64::from(pred.signum() == y.signum());
            self.hits = (lam * self.hit_w * self.hits + w * hit) / hw;
            self.hit_w = hw;
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
        (*state >> 11) as f64 / (1u64 << 53) as f64
    }

    #[test]
    fn p2_matches_the_empirical_quantile() {
        for p in [0.1, 0.5, 0.9, 0.99] {
            let mut est = P2Quantile::new(p).unwrap();
            let mut s = 3u64;
            let mut all: Vec<f64> = Vec::new();
            for _ in 0..20000 {
                let x = lcg(&mut s) * 10.0;
                est.update(x);
                all.push(x);
            }
            all.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let truth = all[((all.len() as f64) * p) as usize];
            let got = est.get().unwrap();
            assert!(
                (got - truth).abs() < 0.3,
                "p={p}: P2 said {got}, empirical {truth}"
            );
        }
    }

    #[test]
    fn p2_is_accurate_on_skewed_data_that_forces_the_linear_fallback() {
        // `adjust` prefers a parabolic prediction and falls back to linear
        // when the parabola would break the markers' ordering -- the paper's
        // condition. Uniform data rarely trips it, so the existing accuracy
        // test leaves the fallback and its two directions unexercised. A
        // heavy-tailed, strongly skewed stream trips it constantly.
        for p in [0.05, 0.25, 0.5, 0.75, 0.95] {
            let mut est = P2Quantile::new(p).unwrap();
            let mut s = 23u64;
            let mut all: Vec<f64> = Vec::new();
            for i in 0..20000 {
                // Exponential-ish tail, with periodic large excursions in both
                // directions so both the upward and downward marker moves run.
                let u = (lcg(&mut s) + 1.0) * 0.5;
                let mut x = -(1.0 - u.min(1.0 - 1e-12)).ln();
                if i % 97 == 0 {
                    x *= 50.0;
                }
                if i % 89 == 0 {
                    x = -x;
                }
                est.update(x);
                all.push(x);
            }
            all.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let truth = all[((all.len() as f64) * p) as usize];
            let got = est.get().unwrap();
            let spread = all[all.len() - 1] - all[0];
            assert!(
                (got - truth).abs() < 0.02 * spread,
                "p={p}: P2 said {got}, empirical {truth} (spread {spread})"
            );
        }
    }

    #[test]
    fn p2_markers_stay_ordered_and_bracket_the_data() {
        // The invariant the fallback exists to protect. If `adjust` ever moved
        // a marker past its neighbour the estimate would be meaningless, and
        // the accuracy tests would only notice if it happened to move far.
        let mut est = P2Quantile::new(0.9).unwrap();
        let mut s = 29u64;
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for i in 0..5000 {
            let x = if i % 13 == 0 {
                100.0 * lcg(&mut s)
            } else {
                lcg(&mut s)
            };
            est.update(x);
            lo = lo.min(x);
            hi = hi.max(x);
            // The markers are only meaningful once the first five points have
            // been sorted into them, which is also when `get` starts reporting.
            if est.count < 5 {
                continue;
            }
            for w in est.q.windows(2) {
                assert!(w[0] <= w[1], "row {i}: markers out of order {:?}", est.q);
            }
            for w in est.n.windows(2) {
                assert!(w[0] < w[1], "row {i}: positions out of order {:?}", est.n);
            }
            if let Some(q) = est.get() {
                assert!(q >= lo && q <= hi, "row {i}: {q} outside [{lo}, {hi}]");
            }
        }
    }

    #[test]
    fn p2_costs_five_numbers_not_a_window() {
        // The point of P2: state is constant regardless of stream length.
        let mut est = P2Quantile::new(0.9).unwrap();
        let mut s = 5u64;
        for _ in 0..100_000 {
            est.update(lcg(&mut s));
        }
        let bytes = rmp_serde::to_vec(&est).unwrap();
        assert!(bytes.len() < 300, "state grew to {} bytes", bytes.len());
    }

    /// The reason to track a quantile rather than infer one from sigma: on a
    /// fat-tailed stream a Gaussian interval is simply the wrong number, and
    /// the error is not even in a predictable direction. Here 0.5%
    /// contamination inflates sigma enormously while barely moving the 99th
    /// percentile, so `mean + 2.33·sd` overshoots by an order of magnitude.
    #[test]
    fn p2_tracks_a_fat_tailed_quantile_where_sigma_cannot() {
        let mut est = P2Quantile::new(0.99).unwrap();
        let mut s = 7u64;
        let (mut sum, mut sq, mut n) = (0.0, 0.0, 0.0);
        let mut all = Vec::new();
        for i in 0..50000 {
            let x = if i % 200 == 0 { 100.0 } else { lcg(&mut s) };
            est.update(x);
            all.push(x);
            sum += x;
            sq += x * x;
            n += 1.0;
        }
        all.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let truth = all[(all.len() as f64 * 0.99) as usize];
        let sd = (sq / n - (sum / n).powi(2)).sqrt();
        let gaussian_99 = sum / n + 2.33 * sd;

        let p2_err = (est.get().unwrap() - truth).abs();
        let gaussian_err = (gaussian_99 - truth).abs();
        assert!(
            p2_err < 0.5,
            "P2 should track the empirical quantile: {p2_err}"
        );
        assert!(
            gaussian_err > 10.0 * p2_err.max(1e-9),
            "the Gaussian guess ({gaussian_99}) should be far off the truth ({truth})"
        );
    }

    #[test]
    fn p2_is_none_before_five_points() {
        let mut est = P2Quantile::new(0.5).unwrap();
        for i in 0..4 {
            est.update(i as f64);
            assert!(est.get().is_none());
        }
        est.update(5.0);
        assert!(est.get().is_some());
    }

    #[test]
    fn p2_rejects_bad_levels() {
        assert!(P2Quantile::new(0.0).is_err());
        assert!(P2Quantile::new(1.0).is_err());
    }

    #[test]
    fn autocorr_is_near_zero_for_noise() {
        let mut ac = EwAutoCorr::new(1).unwrap();
        let mut s = 11u64;
        for _ in 0..20000 {
            ac.update(lcg(&mut s) - 0.5, 0.999);
        }
        assert!(ac.get().unwrap().abs() < 0.1, "got {:?}", ac.get());
    }

    #[test]
    fn autocorr_detects_a_persistent_series() {
        // An AR(1) with phi = 0.8 must show a strong positive lag-1.
        let mut ac = EwAutoCorr::new(1).unwrap();
        let mut s = 13u64;
        let mut prev = 0.0;
        for _ in 0..40000 {
            prev = 0.8 * prev + (lcg(&mut s) - 0.5);
            ac.update(prev, 0.9995);
        }
        let got = ac.get().unwrap();
        assert!(
            got > 0.6,
            "AR(1) phi=0.8 should show strong lag-1, got {got}"
        );
    }

    #[test]
    fn autocorr_detects_alternation() {
        let mut ac = EwAutoCorr::new(1).unwrap();
        for i in 0..20000 {
            ac.update(if i % 2 == 0 { 1.0 } else { -1.0 }, 0.999);
        }
        assert!(ac.get().unwrap() < -0.9, "got {:?}", ac.get());
    }

    #[test]
    fn autocorr_rejects_lag_zero() {
        assert!(EwAutoCorr::new(0).is_err());
    }
}

#[cfg(test)]
mod metric_tests {
    use super::*;

    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }

    #[test]
    fn a_perfect_prediction_scores_perfectly() {
        let mut m = SlotMetrics::new();
        let mut s = 3u64;
        for _ in 0..5000 {
            let y = lcg(&mut s);
            m.update(y, y, 1.0, 1.0);
        }
        assert!((m.ic().unwrap() - 1.0).abs() < 1e-9);
        assert!((m.r2().unwrap() - 1.0).abs() < 1e-9);
        assert!((m.hit_rate().unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn an_uninformative_prediction_scores_at_chance() {
        let mut m = SlotMetrics::new();
        let mut s = 5u64;
        for _ in 0..40000 {
            m.update(lcg(&mut s), lcg(&mut s), 1.0, 1.0);
        }
        assert!(m.ic().unwrap().abs() < 0.05, "ic {:?}", m.ic());
        assert!(m.r2().unwrap() < 0.05, "r2 {:?}", m.r2());
        assert!(
            (m.hit_rate().unwrap() - 0.5).abs() < 0.05,
            "hit {:?}",
            m.hit_rate()
        );
    }

    #[test]
    fn predicting_the_mean_scores_zero_r2() {
        let mut m = SlotMetrics::new();
        let mut s = 7u64;
        for _ in 0..40000 {
            m.update(0.0, lcg(&mut s), 1.0, 1.0);
        }
        assert!(m.r2().unwrap().abs() < 0.05, "r2 {:?}", m.r2());
    }

    #[test]
    fn a_worse_than_mean_prediction_scores_negative_r2() {
        let mut m = SlotMetrics::new();
        let mut s = 11u64;
        for _ in 0..20000 {
            let y = lcg(&mut s);
            m.update(-2.0 * y, y, 1.0, 1.0);
        }
        assert!(m.r2().unwrap() < -1.0, "r2 {:?}", m.r2());
    }

    #[test]
    fn a_sign_flip_shows_up_as_negative_ic_and_low_hit_rate() {
        let mut m = SlotMetrics::new();
        let mut s = 13u64;
        for _ in 0..20000 {
            let y = lcg(&mut s);
            m.update(-y, y, 1.0, 1.0);
        }
        assert!((m.ic().unwrap() + 1.0).abs() < 1e-9);
        assert!(m.hit_rate().unwrap() < 1e-9);
    }

    #[test]
    fn decay_lets_it_forget_an_old_regime() {
        let mut m = SlotMetrics::new();
        let mut s = 17u64;
        let lam = 0.99;
        for _ in 0..3000 {
            let y = lcg(&mut s);
            m.update(-y, y, lam, 1.0); // wrong sign
        }
        assert!(m.hit_rate().unwrap() < 0.1);
        for _ in 0..3000 {
            let y = lcg(&mut s);
            m.update(y, y, lam, 1.0); // now right
        }
        assert!(
            m.hit_rate().unwrap() > 0.9,
            "did not forget: {:?}",
            m.hit_rate()
        );
    }

    #[test]
    fn unscored_rows_are_ignored_not_counted_as_zero() {
        let mut m = SlotMetrics::new();
        let mut s = 19u64;
        for _ in 0..2000 {
            let y = lcg(&mut s);
            m.update(y, y, 1.0, 1.0);
            m.update(f64::NAN, y, 1.0, 1.0); // no prediction yet
        }
        assert!((m.ic().unwrap() - 1.0).abs() < 1e-6, "ic {:?}", m.ic());
    }

    #[test]
    fn a_row_is_unscored_if_the_prediction_or_the_target_or_the_weight_is_missing() {
        // Three independent reasons to skip, each of which must skip on its
        // own: an OR here, not an AND.
        for (pred, y, w) in [
            (f64::NAN, 1.0, 1.0),
            (f64::INFINITY, 1.0, 1.0),
            (1.0, f64::NAN, 1.0),
            (1.0, 1.0, 0.0),
            (1.0, 1.0, -1.0),
        ] {
            let mut m = SlotMetrics::new();
            let mut s = 31u64;
            for _ in 0..500 {
                let v = lcg(&mut s);
                m.update(v, v, 1.0, 1.0);
            }
            let (ic, r2, hr) = (m.ic(), m.r2(), m.hit_rate());
            m.update(pred, y, 1.0, w);
            assert_eq!(m.ic(), ic, "({pred}, {y}, {w}) must not score");
            assert_eq!(m.r2(), r2);
            assert_eq!(m.hit_rate(), hr);
        }

        // A skipped row still ages the estimates: the effective weight shrinks
        // even though the means do not move.
        let mut m = SlotMetrics::new();
        let mut s = 37u64;
        for _ in 0..500 {
            let v = lcg(&mut s);
            m.update(v, v, 1.0, 1.0);
        }
        let before = m.hit_w;
        m.update(f64::NAN, 1.0, 0.5, 1.0);
        assert!(
            (m.hit_w - before * 0.5).abs() < 1e-12,
            "the weight must decay"
        );
    }

    #[test]
    fn nothing_is_reported_before_any_row() {
        let m = SlotMetrics::new();
        assert!(m.ic().is_none() && m.r2().is_none() && m.hit_rate().is_none());
    }
}
