//! `FeatureHealth`: each feature's recent spread and mean against its
//! longer run, to flag a feature that went quiet or drifted
//! (docs/PLAN.md task 221 (g)).
//!
//! Two exponentially weighted means and variances of the feature row
//! ([`crate::EwDiag`]), one at the diagnostic's memory (fast) and one at four
//! times it (slow, the fourth root of the factor by two square roots,
//! exact in every libm), and for feature `i`
//!
//! ```text
//! spread_ratio = sd_fast / sd_slow
//! mean_shift   = (m_fast − m_slow) / sd_slow
//! ```
//!
//! A steady feature reads a ratio near 1 and a shift near 0. One whose feed
//! stopped -- its last value carried forward -- loses its fast spread first:
//! after `t` clock units held the fast variance has decayed by `2^(−t/h)`
//! and the slow one by `2^(−t/4h)`, so the ratio falls as `2^(−3t/8h)`, to
//! 0.5 at 2.7 half-lives and 0.1 at 8.9. That is long before `rls`, which
//! keeps no floor under the direction a held feature stops renewing, winds
//! up (tens of half-lives; task 217). One that moved to a new level reads a
//! shift in units of its long-run spread. Run once (no decay) the two
//! memories coincide and there is nothing to read.

use serde::{Deserialize, Serialize};

/// The fast and slow moments of the feature row. See the module docs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeatureHealth {
    fast: crate::EwDiag,
    slow: crate::EwDiag,
}

impl FeatureHealth {
    pub fn new(k: usize) -> Self {
        Self {
            fast: crate::EwDiag::new(k),
            slow: crate::EwDiag::new(k),
        }
    }

    pub fn has_shape(&self, k: usize) -> bool {
        self.fast.k() == k && self.slow.k() == k
    }

    /// One row of features at weight `w`, after the step's fast decay
    /// `lam`. A row with a feature that is not finite only ages; a weight of
    /// 0 only ages (CLAUDE.md hard rule 9).
    pub fn update(&mut self, x: &[f64], lam: f64, w: f64) {
        let slow = crate::breaks::slow_factor(lam);
        let w = if x.iter().all(|v| v.is_finite()) {
            w
        } else {
            0.0
        };
        let x: Vec<f64> = if w > 0.0 {
            x.to_vec()
        } else {
            vec![0.0; x.len()]
        };
        self.fast.update(&x, lam, w);
        self.slow.update(&x, slow, w);
    }

    /// Feature `i`'s `(spread_ratio, mean_shift)`: `None` while the slow
    /// spread is 0, and run once, where the memories coincide.
    pub fn read(&self, i: usize) -> Option<(f64, f64)> {
        let (vf, vs) = (self.fast.var(i), self.slow.var(i));
        if vs.is_nan() || vs <= 0.0 || vf.is_nan() || self.fast.n_eff() == self.slow.n_eff() {
            return None;
        }
        let sd = vs.sqrt();
        let ((fh, fl), (sh, sl)) = (self.fast.mean_pair(i), self.slow.mean_pair(i));
        Some(((vf / vs).sqrt(), ((fh - sh) + (fl - sl)) / sd))
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

    /// The ratio and the shift by their definitions, two-pass weighted
    /// moments of every row at each memory's present weight.
    #[test]
    fn health_is_its_definition_and_a_held_feature_goes_quiet() {
        let lam = 0.97_f64;
        let slow = crate::breaks::slow_factor(lam);
        let mut f = FeatureHealth::new(1);
        let mut st = 3u64;
        let mut rows: Vec<(f64, f64, f64)> = Vec::new(); // (x, fast weight, slow weight)
        for i in 0..400 {
            let x = if i < 250 { 5.0 + lcg(&mut st) } else { 5.2 };
            let w = if i % 11 == 3 { 0.0 } else { 1.0 };
            f.update(&[x], lam, w);
            rows.iter_mut().for_each(|r| {
                r.1 *= lam;
                r.2 *= slow;
            });
            if w > 0.0 {
                rows.push((x, 1.0, 1.0));
            }
            if i < 20 || i % 29 != 0 {
                continue;
            }
            let moments = |pick: fn(&(f64, f64, f64)) -> f64| {
                let sw: f64 = rows.iter().map(pick).sum();
                let m = rows.iter().map(|r| pick(r) * r.0).sum::<f64>() / sw;
                let v = rows
                    .iter()
                    .map(|r| pick(r) * (r.0 - m).powi(2))
                    .sum::<f64>()
                    / sw;
                (m, v)
            };
            let ((mf, vf), (ms, vs)) = (moments(|r| r.1), moments(|r| r.2));
            let (ratio, shift) = f.read(0).unwrap();
            assert!((ratio - (vf / vs).sqrt()).abs() < 1e-9, "row {i}");
            assert!((shift - (mf - ms) / vs.sqrt()).abs() < 1e-9, "row {i}");
        }
        // 150 rows held at a half-life of 22.8: 2^(-3·6.6/8), about 0.18.
        assert!(f.read(0).unwrap().0 < 0.25, "a held feature goes quiet");
        let mut once = FeatureHealth::new(1);
        for _ in 0..10 {
            once.update(&[lcg(&mut st)], 1.0, 1.0);
        }
        assert!(once.read(0).is_none(), "run once there is no longer run");
        assert!(f.has_shape(1));
    }
}
