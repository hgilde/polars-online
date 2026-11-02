//! What a blocked Gram's merges leave in its co-moments, bounded as they
//! happen (docs/PLAN.md task 308): per feature `e_i`, the diagonal of a
//! nonnegative semidefinite `M` with `|C − C*| ≤ M` entrywise for some
//! semidefinite `C*`. The argument, and the recursion [`MergeBound::merged`]
//! computes, are the condition module's docs, *A blocked Gram's bound*,
//! items 6 to 8 (`crate::solve::condition`); the ridge's shortcut reads `e`
//! there (item 9).
//!
//! Not state: a state file does not hold it, [`PartialEq`] ignores it, and
//! a loaded accumulator starts with it unknown, which its next merge rebuilds
//! from the matrix itself (item 7). `O(k)` a merge, and `O(k²)` once where it
//! is rebuilt.

use super::LowerTriangle;

/// The unit roundoff, `2⁻⁵³`.
const U: f64 = f64::EPSILON / 2.0;

/// `η`, the most a product or quotient that underflows is off by: the
/// smallest subnormal, `2⁻¹⁰⁷⁴`.
const ETA: f64 = f64::from_bits(1);

/// Added to each `e'_i` for any of its own roundings that underflowed:
/// `2⁻¹⁰⁰⁰`.
const FLOOR: f64 = f64::from_bits((1023 - 1000) << 52);

/// `1 + 2⁻⁴⁶`: the factor that takes in the roundings of computing `e'_i`.
const UP: f64 = 1.0 + 1.0 / (1u64 << 46) as f64;

/// `2⁻⁹⁰⁰`: the smallest `α = 1/W'` the argument takes, so that the
/// underflows' `2 n η/α` stays under `ε`.
const ALPHA_MIN: f64 = f64::from_bits((1023 - 900) << 52);

/// `e`, one number per feature; empty where it is not known.
#[derive(Debug, Clone, Default)]
pub(crate) struct MergeBound(Vec<f64>);

impl PartialEq for MergeBound {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

/// What a merge computed that the bound reads (the condition module's docs,
/// item 8): `σ = W_A/W'`, `α = 1/W'`, `β = W_A U/W'²` and `ω = U/W'` as the
/// merge used them, `Û` its summed weight, `n` its rows, `g` the means'
/// difference and `δ̂` the residue, and whether the rows were weighted by
/// their shares rather than scaled by `α`.
pub(crate) struct MergeStep<'a> {
    pub(crate) n: usize,
    pub(crate) scale: f64,
    pub(crate) alpha: f64,
    pub(crate) between: f64,
    pub(crate) within: f64,
    pub(crate) u_sum: f64,
    pub(crate) big: &'a [f64],
    pub(crate) delta: &'a [f64],
    pub(crate) shares: bool,
}

impl MergeBound {
    /// `e`, where it is known for all `k` features.
    pub(crate) fn get(&self, k: usize) -> Option<&[f64]> {
        (k > 0 && self.0.len() == k).then_some(self.0.as_slice())
    }

    /// Forget `e`: the moments changed outside a merge, or a merge fell
    /// outside the argument. The next merge rebuilds it ([`Self::open`]).
    pub(crate) fn forget(&mut self) {
        self.0.clear();
    }

    /// Before a merge: where `e` is not known, take the whole of `C` as the
    /// error over a target of 0, `e_i = maxⱼ |c_ij|` (item 7); infinite
    /// where an entry is not finite.
    pub(crate) fn open(&mut self, c: &LowerTriangle, k: usize) {
        if k == 0 || self.0.len() == k {
            return;
        }
        let lower = c.lower();
        self.0.clear();
        self.0.resize(k, 0.0);
        let mut finite = true;
        for i in 0..k {
            for j in 0..=i {
                let v = lower[i * k + j].abs();
                finite &= v.is_finite();
                self.0[i] = self.0[i].max(v);
                self.0[j] = self.0[j].max(v);
            }
        }
        if !finite {
            self.0.fill(f64::INFINITY);
        }
    }

    /// After a merge, `c` its result: `e'` from `e` by item 8's recursion,
    /// or `e` forgotten where the merge is outside the argument.
    pub(crate) fn merged(&mut self, c: &LowerTriangle, k: usize, s: &MergeStep<'_>) {
        if self.0.len() != k || k == 0 {
            return self.forget();
        }
        let n = s.n as f64;
        let m = (4.0 * n + 32.0) * U;
        let eps = m / (1.0 - m);
        let usable = !s.shares
            && eps <= 1e-3
            && s.alpha >= ALPHA_MIN
            && s.alpha.is_finite()
            && s.within.is_normal()
            && s.u_sum.is_normal()
            && s.scale >= 0.0
            && s.scale.is_finite()
            && s.between >= 0.0
            && s.between.is_finite();
        if !usable {
            return self.forget();
        }
        let (sigma, alpha, beta, omega) = (s.scale, s.alpha, s.between, s.within);
        let ratio = n / s.u_sum;
        let tau = 2.0 * ETA * (ratio + 1.0 + ratio.sqrt());
        // Every term below is nonnegative; the order keeps a large factor
        // from multiplying an underflowed product (the module docs).
        let residue_floor = 7.0 / eps * omega * tau * tau;
        let rows = n * (4.0 + alpha);
        let grow = sigma * (1.0 + eps);
        let shrink = 1.0 / (1.0 - 10.0 * eps);
        let lower = c.lower();
        let mut finite = true;
        for (i, ei) in self.0.iter_mut().enumerate() {
            let (g, h) = (s.big[i].abs(), s.delta[i].abs());
            let cii = lower[i * k + i];
            let cii = if cii.is_nan() {
                f64::INFINITY
            } else {
                cii.max(0.0)
            };
            let under = ((1.0 + g) * (1.0 + g) + (1.0 + h) * (1.0 + h) + 3.0 + rows) * ETA;
            let sum = *ei * grow
                + 10.0 * eps * cii
                + 22.0 * eps * (omega * (h * h))
                + eps * (beta * (g * g))
                + residue_floor
                + under;
            *ei = sum * shrink * UP + FLOOR;
            finite &= ei.is_finite();
        }
        if !finite {
            self.forget();
        }
    }
}
