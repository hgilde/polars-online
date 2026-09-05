//! `EwDiag`: the diagonal of [`EwCov`] — exponentially weighted means and
//! variances of a vector stream, O(k) a row.
//!
//! The same weighted Welford recursion as `ewcov.rs`, operation for
//! operation, minus the co-moments:
//!
//! ```text
//! W'    = lam * W + w
//! a     = lam * W / W'        b = w / W'        (a + b = 1)
//! delta = x - m
//! m'    = m + b * delta
//! c'_i  = a * c_i + a * b * delta_i * delta_i
//! ```
//!
//! It exists for the models that standardize their features and read nothing
//! but the diagonal — `kalman` and `sgd` — which until schema 3 carried a full
//! `EwCov` for the purpose and paid `k²` co-moment updates a row for `k`
//! variances (docs/PERFORMANCE.md §13). The numbers are the same to the bit:
//! every diagonal entry is updated with exactly the arithmetic `EwCov` uses
//! for it, in the same order, so a model moved from one to the other keeps
//! its outputs (`tests/model_contract.rs`, the goldens), and a schema-2 state
//! converts by taking the diagonal ([`EwDiag::diagonal_of`]).

use serde::{Deserialize, Serialize};

use crate::EwCov;

/// EW means and centered variances of a `k`-vector: [`EwCov`] without the
/// off-diagonal co-moments. See the module docs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "EwDiagWire")]
pub struct EwDiag {
    k: usize,
    /// EW sum of weights (the `n_eff` count).
    w_sum: f64,
    /// EW mean vector, length `k`.
    m: Vec<f64>,
    /// EW **centered** second moments, length `k`: `EwCov`'s `c[i*k+i]`.
    c: Vec<f64>,
}

/// The wire layout, checked on the way in. `deny_unknown_fields` is load-
/// bearing: a schema-2 `EwCov` carries these four names among its seven, and
/// without it a map-encoded `EwCov` would deserialize as an `EwDiag` with
/// `k²` "variances" — silently, where the untagged model loaders (`kalman`,
/// `sgd`) need it to *fail* so they fall through to the schema-2 layout.
/// The shape check covers the array encoding the same way.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EwDiagWire {
    k: usize,
    w_sum: f64,
    m: Vec<f64>,
    c: Vec<f64>,
}

impl TryFrom<EwDiagWire> for EwDiag {
    type Error = String;

    fn try_from(w: EwDiagWire) -> Result<Self, String> {
        if w.m.len() != w.k || w.c.len() != w.k {
            return Err(format!(
                "EwDiag: state has the wrong shape (k = {}, {} means, {} variances)",
                w.k,
                w.m.len(),
                w.c.len()
            ));
        }
        Ok(Self {
            k: w.k,
            w_sum: w.w_sum,
            m: w.m,
            c: w.c,
        })
    }
}

impl EwDiag {
    pub fn new(k: usize) -> Self {
        Self {
            k,
            w_sum: 0.0,
            m: vec![0.0; k],
            c: vec![0.0; k],
        }
    }

    /// The diagonal of a full accumulator: the same means, variances and
    /// weight, so a model that only ever read those continues unchanged.
    /// This is how a schema-2 `kalman` or `sgd` state loads.
    pub fn diagonal_of(cov: &EwCov) -> Self {
        let k = cov.k();
        Self {
            k,
            w_sum: cov.n_eff(),
            m: cov.means().to_vec(),
            c: (0..k).map(|i| cov.cov(i, i)).collect(),
        }
    }

    #[inline]
    pub fn k(&self) -> usize {
        self.k
    }

    #[inline]
    pub fn n_eff(&self) -> f64 {
        self.w_sum
    }

    #[inline]
    pub fn mean(&self, i: usize) -> f64 {
        self.m[i]
    }

    /// The EW mean vector, length `k`.
    pub fn means(&self) -> &[f64] {
        &self.m
    }

    /// Raw (uncentered) second moment `E_w[x_i²]`, reconstructed from the
    /// centered one like [`EwCov::raw`].
    #[inline]
    pub fn raw(&self, i: usize) -> f64 {
        self.c[i] + self.m[i] * self.m[i]
    }

    /// Centered variance, floored at zero against rounding ([`EwCov::var`]).
    #[inline]
    pub fn var(&self, i: usize) -> f64 {
        self.c[i].max(0.0)
    }

    /// One observation with decay factor `lam` (from [`crate::Decay::factor`])
    /// and row weight `w`. O(k), allocation-free, and the same guards as
    /// [`EwCov::update`]: a negative weight is a caller's bug (debug assert,
    /// no-op in release), and a row that leaves the total weight at zero
    /// changes nothing (hard rule 9).
    pub fn update(&mut self, x: &[f64], lam: f64, w: f64) {
        debug_assert_eq!(x.len(), self.k);
        debug_assert!(
            w >= 0.0,
            "EwDiag::update requires a non-negative weight, got {w}"
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
        // Weighted Welford, as `EwCov::update` writes its diagonal: the
        // deviation is against the OLD mean, and the expression
        // `a * c + a * b * d * d` is kept in that order so the bits agree.
        for ((ci, mi), xi) in self.c.iter_mut().zip(self.m.iter_mut()).zip(x) {
            let d = xi - *mi;
            *ci = a * *ci + a * b * d * d;
            *mi += b * d;
        }
        self.w_sum = w_new;
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

    /// A stream with offsets and scales that differ by orders of magnitude,
    /// zero-weight rows (including the first), varying decay and a pure
    /// zero-weight run — everything `kalman` and `sgd` can feed it.
    fn stream(k: usize, n: usize, seed: u64) -> Vec<(Vec<f64>, f64, f64)> {
        let mut s = seed;
        (0..n)
            .map(|r| {
                let x: Vec<f64> = (0..k)
                    .map(|i| 10f64.powi(i as i32 - 2) * lcg(&mut s) + 7.0 * i as f64)
                    .collect();
                let lam = if r % 11 == 0 { 0.5 } else { 0.97 };
                let w = match r % 9 {
                    0 => 0.0,
                    1 => 2.5,
                    _ => 1.0,
                };
                let w = if r == 0 || (40..46).contains(&r) {
                    0.0
                } else {
                    w
                };
                (x, lam, w)
            })
            .collect()
    }

    #[test]
    fn is_the_diagonal_of_ewcov_bit_for_bit() {
        for k in [1, 2, 5] {
            let mut full = EwCov::new(k);
            let mut diag = EwDiag::new(k);
            for (x, lam, w) in stream(k, 300, 7 + k as u64) {
                full.update(&x, lam, w);
                diag.update(&x, lam, w);
                assert_eq!(diag.n_eff().to_bits(), full.n_eff().to_bits());
                for i in 0..k {
                    assert_eq!(diag.mean(i).to_bits(), full.mean(i).to_bits(), "mean {i}");
                    assert_eq!(diag.var(i).to_bits(), full.var(i).to_bits(), "var {i}");
                    assert_eq!(diag.raw(i).to_bits(), full.raw(i, i).to_bits(), "raw {i}");
                }
                assert_eq!(diag.means(), full.means());
                assert_eq!(diag, EwDiag::diagonal_of(&full));
            }
        }
    }

    #[test]
    fn a_zero_weight_first_row_and_a_negative_weight_change_nothing() {
        let mut d = EwDiag::new(2);
        d.update(&[1e100, -3.0], 0.9, 0.0);
        assert_eq!(d, EwDiag::new(2));
        d.update(&[1.0, 2.0], 0.9, 1.0);
        let before = d.clone();
        // `debug_assert` would fire; the release-mode contract is a no-op.
        if !cfg!(debug_assertions) {
            d.update(&[5.0, 5.0], 0.9, -1.0);
            assert_eq!(d, before);
        }
        assert_eq!(d.mean(0), 1.0);
        assert_eq!(d.var(1), 0.0);
        assert_eq!(d.raw(1), 4.0);
    }

    #[test]
    fn diagonal_of_takes_the_diagonal() {
        let mut full = EwCov::with_precision_prior(3, 0.5).unwrap();
        for (x, lam, w) in stream(3, 50, 1) {
            full.update(&x, lam, w);
        }
        let d = EwDiag::diagonal_of(&full);
        assert_eq!(d.k(), 3);
        assert_eq!(d.n_eff(), full.n_eff());
        for i in 0..3 {
            assert_eq!(d.mean(i), full.mean(i));
            assert_eq!(d.var(i), full.var(i));
            assert_eq!(d.raw(i), full.raw(i, i));
        }
    }

    #[test]
    fn serde_roundtrip_in_both_encodings() {
        let mut d = EwDiag::new(2);
        d.update(&[1.0, 2.0], 0.95, 1.3);
        d.update(&[0.5, 2.5], 0.95, 1.0);
        for bytes in [
            rmp_serde::to_vec_named(&d).unwrap(),
            rmp_serde::to_vec(&d).unwrap(),
        ] {
            let back: EwDiag = rmp_serde::from_slice(&bytes).unwrap();
            assert_eq!(back, d);
        }
    }

    /// The property the schema-2 loaders of `kalman` and `sgd` rest on: a
    /// serialized `EwCov` must not pass for an `EwDiag` in either encoding,
    /// or an untagged loader would take the wrong branch and build a model
    /// with `k²` variances.
    #[test]
    fn a_full_ewcov_is_refused_in_both_encodings() {
        for k in [1, 2, 4] {
            let mut full = EwCov::new(k);
            full.update(&vec![1.0; k], 0.9, 1.0);
            full.update(&vec![2.0; k], 0.9, 1.0);
            for bytes in [
                rmp_serde::to_vec_named(&full).unwrap(),
                rmp_serde::to_vec(&full).unwrap(),
            ] {
                let got = rmp_serde::from_slice::<EwDiag>(&bytes);
                assert!(got.is_err(), "k = {k}: {got:?}");
            }
        }
        // The shape check on its own, for a hand-written map.
        let bad =
            serde_json::json!({"k": 2, "w_sum": 1.0, "m": [0.0, 0.0], "c": [0.0, 0.0, 0.0, 0.0]});
        let err = serde_json::from_value::<EwDiag>(bad)
            .unwrap_err()
            .to_string();
        assert!(err.contains("wrong shape"), "{err}");
    }
}
