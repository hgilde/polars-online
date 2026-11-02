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
//!
//! An ordinary merge forms no subnormal number here (item 8, last
//! paragraph): x86-64 takes each subnormal operand or result through a
//! microcode assist of about a hundred cycles, which made a blocked fit
//! about 4% slower there on the same instructions when the two terms built
//! on `η` were formed as products. A test build counts any product or
//! quotient here that reads or makes one (`SUBNORMALS`).

use super::LowerTriangle;

/// The unit roundoff, `2⁻⁵³`.
const U: f64 = f64::EPSILON / 2.0;

/// `η`, the most a product or quotient that underflows is off by: the
/// smallest subnormal, `2⁻¹⁰⁷⁴`.
const ETA: f64 = f64::from_bits(1);

/// Added to each `e'_i` for any of its own roundings that underflowed,
/// `2⁻¹⁰⁰⁰`; and once more for the two terms built on `η` that an ordinary
/// merge does not form (item 8).
const FLOOR: f64 = f64::from_bits((1023 - 1000) << 52);

/// `2⁷²`: the largest computed `ν_i` whose `η ν_i` is left to the second
/// `FLOOR` rather than formed. At most a quarter of it, so that the term,
/// with its own roundings and the residue's at most `2⁻¹¹²⁴`, stays inside
/// it after the division by `1 − 10 ε ≥ 0.99` (item 8).
const UNDER_MAX: f64 = f64::from_bits((1023 + 72) << 52);
const _: () = assert!(UNDER_MAX * ETA <= FLOOR / 4.0);

/// `√η = 2⁻⁵³⁷`, normal: `x √η √η` is `η x` to the bit wherever that is
/// normal, with no subnormal operand -- both scalings are exact there.
const ROOT_ETA: f64 = f64::from_bits((1023 - 537) << 52);
const _: () = assert!(ROOT_ETA * ROOT_ETA == ETA);

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
        let ratio = over(n, s.u_sum);
        let spread = ratio + 1.0 + ratio.sqrt();
        // The residue's `7 ω τ²/ε = η² r`, `τ = η τ_n`: left to the second
        // `FLOOR` where `r` is finite, and so at most `2⁻¹¹²⁴`. Where it is
        // not, `τ` is formed, `2η` times `spread` in two scalings, the
        // first exact, and the term as before, which forgets `e` where it is
        // not finite either (the module docs, item 8).
        let tau_n = 2.0 * spread;
        let r = times(times(times(7.0 / eps, omega), tau_n), tau_n);
        let residue = if r.is_finite() {
            0.0
        } else {
            let tau = spread * ROOT_ETA * (2.0 * ROOT_ETA);
            // Every factor is nonnegative; the order keeps a large factor
            // from multiplying an underflowed product.
            7.0 / eps * omega * tau * tau
        };
        let rows = times(n, 4.0 + alpha);
        let grow = 1.0 + eps;
        let shrink = 1.0 / (1.0 - 10.0 * eps);
        let (ten, twenty_two) = (10.0 * eps, 22.0 * eps);
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
            // The underflows' `η ν_i`: left to the second `FLOOR` while the
            // computed `ν_i` is at most `2⁷²`; past it, or where it is not a
            // number, formed in two exact scalings by `√η`, a normal number.
            let nu = (1.0 + g) * (1.0 + g) + (1.0 + h) * (1.0 + h) + 3.0 + rows;
            let under = if nu <= UNDER_MAX {
                0.0
            } else {
                times(times(nu, ROOT_ETA), ROOT_ETA)
            };
            // Every term is nonnegative; each product's order keeps a
            // large factor from multiplying an underflowed one.
            let sum = times(times(*ei, sigma), grow)
                + times(ten, cii)
                + times(twenty_two, times(omega, times(h, h)))
                + times(eps, times(beta, times(g, g)))
                + residue
                + under;
            *ei = times(times(sum, shrink), UP) + FLOOR + FLOOR;
            finite &= ei.is_finite();
        }
        if !finite {
            self.forget();
        }
    }
}

/// `a b`. A test build counts it in `SUBNORMALS` where it reads or makes a
/// subnormal number, or makes 0 from two numbers that are not.
#[inline(always)]
fn times(a: f64, b: f64) -> f64 {
    let p = a * b;
    #[cfg(test)]
    note(a, b, p);
    p
}

/// `a / b`, counted as [`times`] is.
#[inline(always)]
fn over(a: f64, b: f64) -> f64 {
    let q = a / b;
    #[cfg(test)]
    note(a, b, q);
    q
}

#[cfg(test)]
fn note(a: f64, b: f64, result: f64) {
    let tiny = a.is_subnormal()
        || b.is_subnormal()
        || result.is_subnormal()
        || (result == 0.0 && a != 0.0 && b != 0.0 && b.is_finite());
    if tiny {
        SUBNORMALS.with(|c| c.set(c.get() + 1));
    }
}

#[cfg(test)]
thread_local! {
    /// The products and quotients of [`MergeBound::merged`] on this thread
    /// that read or made a subnormal number, or underflowed to 0: what a
    /// test reads to see that an ordinary merge forms none.
    pub(crate) static SUBNORMALS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}
