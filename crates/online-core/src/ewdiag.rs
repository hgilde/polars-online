//! `EwDiag`: the diagonal of [`EwCov`](crate::EwCov) — exponentially weighted means and
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
//! for it, in the same order, so a model moved from one to the other kept
//! its outputs (`tests/model_contract.rs`, the goldens).

use serde::{Deserialize, Serialize};

/// EW means and centered variances of a `k`-vector: [`EwCov`](crate::EwCov) without the
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
    /// What each mean leaves out: the mean is `m[i] + m_lo[i]`, as
    /// `EwCov`'s is ([`crate::comp`]; docs/PLAN.md task 101). One per mean.
    m_lo: Vec<f64>,
}

/// The wire layout, checked on the way in. `deny_unknown_fields` is load-
/// bearing: an `EwCov` carries these names among its own, and without it a
/// map-encoded `EwCov` would deserialize as an `EwDiag` with `k²`
/// "variances", silently. The shape check covers the array encoding the
/// same way.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EwDiagWire {
    k: usize,
    w_sum: f64,
    m: Vec<f64>,
    c: Vec<f64>,
    m_lo: Vec<f64>,
}

impl TryFrom<EwDiagWire> for EwDiag {
    type Error = String;

    fn try_from(w: EwDiagWire) -> Result<Self, String> {
        if w.m.len() != w.k || w.c.len() != w.k || w.m_lo.len() != w.k {
            return Err(format!(
                "EwDiag: state has the wrong shape (k = {}, {} means, {} variances, {} low parts)",
                w.k,
                w.m.len(),
                w.c.len(),
                w.m_lo.len()
            ));
        }
        Ok(Self {
            k: w.k,
            w_sum: w.w_sum,
            m: w.m,
            c: w.c,
            m_lo: w.m_lo,
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
            m_lo: vec![0.0; k],
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

    /// Mean `i` as the pair it is kept as, `(hi, lo)`: the mean is `hi +
    /// lo` ([`crate::comp`]). `kalman` reads how far a mean moved from two
    /// of them, `(hi' − hi) + (lo' − lo)`, which `hi' − hi` alone would
    /// round at a level (docs/PLAN.md task 206).
    #[inline]
    pub fn mean_pair(&self, i: usize) -> (f64, f64) {
        (self.m[i], self.m_lo[i])
    }

    /// `x`'s deviation from mean `i`, the pair's ([`crate::comp::dev`]).
    #[inline]
    pub fn deviation(&self, i: usize, x: f64) -> f64 {
        crate::comp::dev(x, self.m[i], self.m_lo[i])
    }

    /// Raw (uncentered) second moment `E_w[x_i²]`, reconstructed from the
    /// centered one like [`EwCov::raw`].
    #[inline]
    pub fn raw(&self, i: usize) -> f64 {
        self.c[i] + self.m[i] * self.m[i]
    }

    /// Centered variance, floored at zero against rounding, a NaN kept
    /// ([`EwCov::var`]).
    #[inline]
    pub fn var(&self, i: usize) -> f64 {
        crate::solve::clamp_rounding(self.c[i])
    }

    /// The moments as they will stand once one more row is admitted at unit
    /// weight after a decay of `lam`, read slot by slot without touching
    /// the accumulator. `sgd` standardises a row against these ([`crate::Sgd`],
    /// and docs/PLAN.md task 74 for why the moments include the row). The
    /// arithmetic is [`EwDiag::update`]'s, operation for operation, so what
    /// this reads for `x` is to the bit what `update(x, lam, 1.0)` leaves
    /// behind. Always well defined: the total weight is at least the row's
    /// own 1, even on an empty accumulator, where the row is the whole
    /// history and its variance is zero.
    #[inline]
    pub fn including(&self, lam: f64) -> Including<'_> {
        let w_new = lam * self.w_sum + 1.0;
        Including {
            sc: self,
            a: lam * self.w_sum / w_new,
            b: 1.0 / w_new,
        }
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
            // Nothing carried and nothing added: `a` and `b` are 0/0, and no
            // moment moves. The weight is 0 either way -- it was, at the
            // head of a stream, and the decay took it, from 1075 half-lives
            // on, where it used to be kept (task 115 (c), PLAN §12).
            self.w_sum = w_new;
            return;
        }
        let a = lam * self.w_sum / w_new; // weight of the old statistics
        let b = w / w_new; // weight of the new point
        // Weighted Welford, as `EwCov::update` writes its diagonal: the
        // deviation is against the OLD mean, and the expression
        // `a * c + a * b * d * d` is kept in that order so the bits agree.
        let slots = self
            .c
            .iter_mut()
            .zip(self.m.iter_mut())
            .zip(self.m_lo.iter_mut());
        // A row of weight 0 leaves the moments as they are: `a` is 1 and `b`
        // is 0, so there is no step to take (`crate::comp::add` says why a
        // step of 0 is not taken either), and only the weight decays. The
        // test is made once, outside the loop: a test in the loop kept
        // `sgd`'s standardizing from vectorizing.
        if b > 0.0 {
            for (((ci, mi), lo), &xi) in slots.zip(x) {
                let d = crate::comp::dev(xi, *mi, *lo);
                *ci = a * *ci + a * b * d * d;
                crate::comp::add(mi, lo, b * d);
            }
        }
        self.w_sum = w_new;
    }
}

/// [`EwDiag::including`]: the accumulator plus one row at unit weight, one
/// slot at a time.
#[derive(Debug, Clone, Copy)]
pub struct Including<'a> {
    sc: &'a EwDiag,
    /// Weight of the old statistics, `lam * W / (lam * W + 1)`.
    a: f64,
    /// Weight of the new point, `1 / (lam * W + 1)`.
    b: f64,
}

impl Including<'_> {
    /// `(mean, var, dev)` of slot `i` with `x` admitted: `update`'s recursion
    /// for the slot, the variance floored at zero as [`EwDiag::var`] floors
    /// it, and `x`'s deviation from the mean it leaves, the pair's
    /// ([`crate::comp::dev`]; docs/PLAN.md task 101) -- which `x − mean`
    /// would round.
    #[inline]
    pub fn moments(&self, i: usize, x: f64) -> (f64, f64, f64) {
        let lo = self.sc.m_lo[i];
        let d = crate::comp::dev(x, self.sc.m[i], lo);
        let c = self.a * self.sc.c[i] + self.a * self.b * d * d;
        let (mut hi, mut lo) = (self.sc.m[i], lo);
        crate::comp::add(&mut hi, &mut lo, self.b * d);
        (
            hi,
            crate::solve::clamp_rounding(c),
            crate::comp::dev(x, hi, lo),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EwCov;

    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }

    /// A stream with offsets and scales that differ by orders of magnitude,
    /// zero-weight rows (including the first), varying decay and a pure
    /// zero-weight run — everything `kalman` and `sgd` can feed it.
    /// Rows of `(x, lam, w)`.
    type Rows = Vec<(Vec<f64>, f64, f64)>;

    fn stream(k: usize, n: usize, seed: u64) -> Rows {
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

    /// `stream`, with slot 0 at a level of 1e8 and held there from row 100:
    /// long enough for its mean's step to fall below its rounding, where the
    /// low part carries it (docs/PLAN.md task 101). Its rows of weight 0
    /// carry another value, which moves no mean.
    fn held_stream(k: usize, n: usize, seed: u64) -> Rows {
        let mut rows = stream(k, n, seed);
        for (r, (x, _, w)) in rows.iter_mut().enumerate() {
            x[0] = 1e8
                + match (r < 100, *w > 0.0) {
                    (true, _) => x[0],
                    (false, true) => 0.37,
                    (false, false) => 5.0,
                };
        }
        rows
    }

    /// Both streams: the plain one, and the held one, on which slot 0's mean
    /// is also checked to reach the value it holds, which a plain mean does
    /// not (`held`).
    fn streams(k: usize, seed: u64) -> [(Rows, bool); 2] {
        [
            (stream(k, 300, seed), false),
            (held_stream(k, 1500, seed), true),
        ]
    }

    #[test]
    fn is_the_diagonal_of_ewcov_bit_for_bit() {
        for k in [1, 2, 5] {
            for (rows, held) in streams(k, 7 + k as u64) {
                let mut full = EwCov::new(k);
                let mut diag = EwDiag::new(k);
                for (x, lam, w) in rows {
                    full.update(&x, lam, w);
                    diag.update(&x, lam, w);
                    assert_eq!(diag.n_eff().to_bits(), full.n_eff().to_bits());
                    for i in 0..k {
                        assert_eq!(diag.mean(i).to_bits(), full.mean(i).to_bits(), "mean {i}");
                        assert_eq!(diag.var(i).to_bits(), full.var(i).to_bits(), "var {i}");
                        assert_eq!(diag.raw(i).to_bits(), full.raw(i, i).to_bits(), "raw {i}");
                    }
                    assert_eq!(diag.means(), full.means());
                    assert_eq!(diag.m_lo, full.means_lo());
                }
                if held {
                    assert_eq!(diag.mean(0), 1e8 + 0.37, "k {k}");
                }
            }
        }
    }

    /// `including` reads, for every slot and every row of the stream, the
    /// bits `update` writes for that row at unit weight -- under every decay
    /// the stream uses, from an empty accumulator on -- and moves nothing.
    #[test]
    fn including_reads_what_a_unit_row_would_leave() {
        for k in [1, 3] {
            for (rows, _) in streams(k, 11 + k as u64) {
                let mut d = EwDiag::new(k);
                for (x, lam, w) in rows {
                    let before = d.clone();
                    let inc = d.including(lam);
                    let read: Vec<(f64, f64, f64)> = (0..k).map(|i| inc.moments(i, x[i])).collect();
                    let mut unit = d.clone();
                    unit.update(&x, lam, 1.0);
                    for (i, &(m, v, dev)) in read.iter().enumerate() {
                        assert_eq!(m.to_bits(), unit.mean(i).to_bits(), "mean {i}");
                        assert_eq!(v.to_bits(), unit.var(i).to_bits(), "var {i}");
                        assert_eq!(dev.to_bits(), unit.deviation(i, x[i]).to_bits(), "dev {i}");
                    }
                    assert_eq!(d, before, "reading moved the accumulator");
                    // The stream's own weight, which may be 0 or 2.5, is what
                    // the accumulator actually takes.
                    d.update(&x, lam, w);
                }
            }
        }
        // Empty: the row is the whole history.
        let inc = EwDiag::new(2);
        let inc = inc.including(0.9);
        assert_eq!(inc.moments(0, 1e6), (1e6, 0.0, 0.0));
        assert_eq!(inc.moments(1, -3.0), (-3.0, 0.0, 0.0));
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

    /// A row of weight 0 carrying the mean itself: the variance term is
    /// `a·b·d·d` with `b` and `d` both zero, and stays zero, not NaN. Read
    /// from the accumulator, as `var` read a NaN as 0 until task 182.
    #[test]
    fn a_row_of_no_weight_carrying_the_mean_changes_nothing() {
        let mut d = EwDiag::new(1);
        d.update(&[2.0], 0.9, 1.0);
        d.update(&[2.0], 0.9, 0.0);
        assert_eq!((d.mean(0), d.c[0]), (2.0, 0.0));
    }

    /// A variance of NaN stays NaN, in `var` and in the moments a row would
    /// leave (`including`). The floor against rounding was `max(0.0)`,
    /// which returns its other argument when one is NaN, so a moment a NaN
    /// had reached read as a variance of 0, a column with no spread (task
    /// 182). A rounding below zero is still 0, and so is `-0.0`.
    #[test]
    fn a_variance_of_nan_stays_nan() {
        let mut d = EwDiag::new(4);
        d.update(&[1.0, 2.0, 3.0, 4.0], 1.0, 1.0);
        d.c = vec![f64::NAN, -1e-18, -0.0, 2.5];
        assert!(d.var(0).is_nan(), "{}", d.var(0));
        assert_eq!(d.var(1).to_bits(), 0f64.to_bits());
        assert_eq!(d.var(2).to_bits(), 0f64.to_bits());
        assert_eq!(d.var(3), 2.5);
        // The row at the means: no deviation, so the moments a row leaves
        // are the old ones aged by `a`, a half here.
        let inc = d.including(1.0);
        assert!(inc.moments(0, 1.0).1.is_nan());
        assert_eq!(inc.moments(1, 2.0).1.to_bits(), 0f64.to_bits());
        assert_eq!(inc.moments(2, 3.0).1.to_bits(), 0f64.to_bits());
        assert_eq!(inc.moments(3, 4.0).1, 1.25);
    }

    /// `EwDiag`'s own skip (review 2026-09-25): the rows 0.7 and
    /// 5.292162135665459 at unit weight leave the pair with a low part of a
    /// whole step, where a step of 0 would round `hi` up; a row of weight 0
    /// leaves both parts to the bit. Every other test of this compares it
    /// with `EwCov`, which skips too.
    #[test]
    fn a_row_of_no_weight_takes_no_step_in_the_pair() {
        let mut d = EwDiag::new(1);
        d.update(&[0.7], 1.0, 1.0);
        d.update(&[5.292162135665459], 1.0, 1.0);
        assert_eq!(
            (d.m[0], d.m_lo[0]),
            (2.996081067832729, 4.440892098500626e-16)
        );
        d.update(&[9.0], 1.0, 0.0);
        assert_eq!(
            (d.m[0], d.m_lo[0]),
            (2.996081067832729, 4.440892098500626e-16)
        );
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

    /// A serialized `EwCov` must not pass for an `EwDiag` in either
    /// encoding, or a model would load with `k²` variances.
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
        let bad = serde_json::json!({
            "k": 2, "w_sum": 1.0, "m": [0.0, 0.0], "c": [0.0, 0.0, 0.0, 0.0], "m_lo": [0.0, 0.0]
        });
        let err = serde_json::from_value::<EwDiag>(bad)
            .unwrap_err()
            .to_string();
        assert!(err.contains("wrong shape"), "{err}");
    }
}
