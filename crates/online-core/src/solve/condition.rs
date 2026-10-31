//! Whether a factorized system can carry a digit (docs/PLAN.md task 278,
//! D4): an estimate of its condition from the factor, in `O(k²)`, and the
//! limit past which the solve's forward-error bound says nothing.
//!
//! A Cholesky factorization of a matrix past `1/u` in condition does not
//! always fail: rounding can leave the computed matrix positive definite, and
//! the factor then solves a matrix as far from the one meant as its smallest
//! eigenvalue, so the coefficients are off by more than their own size. The
//! jitter ladder catches the matrices whose factorization fails; this catches
//! the ones whose factorization succeeds.
//!
//! **Which condition.** A Cholesky solve is invariant to a diagonal scaling
//! of the system (van der Sluis's theorem; Higham, *Accuracy and Stability
//! of Numerical Algorithms*, 2nd ed., §7.3 and §10.1): its accuracy, in the
//! coordinates `D x` with `D = diag(√a_ii)`,
//! is governed by the equilibrated matrix `H = D⁻¹ A D⁻¹`, whose diagonal is
//! 1, not by `A` itself. So a design whose features run from `1e-10` to
//! `1e10` is no harder than one of unit scales, and it must not be refused.
//! `H` is the correlation matrix where the system is a centred Gram, and its
//! factor is `D⁻¹ L`, `L` the factor of `A`.
//!
//! **The estimate.** `‖H⁻¹‖₁` by Hager's method as Higham refined it
//! (LAPACK's `dlacn2`; Higham, Algorithm 15.4), which needs only products
//! with `H⁻¹`: two triangular solves each, `O(k²)`, about four of them in
//! all. Each iterate it reads is `‖H⁻¹ v‖₁` for a `v` of unit 1-norm, so the
//! estimate never exceeds `‖H⁻¹‖₁`, and in practice it is within a factor of
//! 3 of it and usually exact (Higham, §15.3). The ratio of the factor's
//! largest and smallest pivots squared is cheaper (`k` reads) but bounds the
//! condition only from below, and can miss it by orders of magnitude: a
//! nearly dependent column whose dependence is spread over many others keeps
//! every pivot moderate (Kahan's matrix, Higham §8.3, is the classic case).
//! Missing the condition here means a garbage fit is reported, so the
//! estimate that is usually exact is the one used.
//!
//! **The screen.** Most systems are far inside the limit, and an upper bound
//! that is cheaper than the estimate says so: with `R = D⁻¹ L` and `M(R)`
//! its comparison matrix (`|r_ii|` on the diagonal, `−|r_ij|` off it),
//! `|R⁻¹| ≤ M(R)⁻¹` entrywise (Higham, §8.3), so `‖H⁻¹‖₁ ≤ ‖R⁻ᵀ‖₁ ‖R⁻¹‖₁
//! = ‖R⁻¹‖∞ ‖R⁻¹‖₁ ≤ max(M(R)⁻¹ 1) · max(M(R)⁻ᵀ 1)`, two substitutions
//! whose terms are all nonnegative, a quarter of one of the estimate's
//! products. Where that bound is inside the limit the system is, and the
//! estimate is not run; where it is not -- it can exceed `‖H⁻¹‖₁` by a
//! factor that grows with `k` when the columns are correlated -- the
//! estimate decides. Either way the answer is the estimate's wherever the
//! estimate could reach the limit.
//!
//! **The limit.** The factorization and its two triangular solves are
//! backward stable: the computed `x̂` solves `(A + ΔA) x̂ = b` with `|ΔA| ≤
//! γ_{3k+1} |Lᵀ||L|` (Higham, Theorem 10.4), `γ_m = m u / (1 − m u)`. By
//! Cauchy–Schwarz each entry of `|L||Lᵀ|` is at most `√(a_ii a_jj)`, the
//! row norms of `L` being the diagonal of `A`, so in `H`'s coordinates every
//! entry of `ΔH = D⁻¹ ΔA D⁻¹` is at most `γ_{3k+1}`, and `‖ΔH‖₁ ≤ k
//! γ_{3k+1}`. Relative to `‖H‖₁` that is a backward error `η = k γ_{3k+1} /
//! ‖H‖₁`, and with the right-hand side formed in the same arithmetic, the
//! normwise forward error of the scaled solution is at most `2 κ₁ η / (1 −
//! κ₁ η)` while `κ₁ η < 1` (Higham, Theorem 7.2). With `κ₁ = ‖H‖₁ ‖H⁻¹‖₁`
//! the norms of `H` cancel: `κ₁ η = k γ_{3k+1} ‖H⁻¹‖₁`. The bound reaches 1
//! -- no digit of the solution guaranteed -- at `κ₁ η = 1/3`, so a system is
//! refused where
//!
//! ```text
//! 3 k γ_{3k+1} ‖H⁻¹‖₁ ≥ 1.
//! ```
//!
//! At `k = 3` that is `‖H⁻¹‖₁ ≥ 1.0e14`, at `k = 10` `9.7e12`, at `k = 50`
//! `4.0e11`. The bound is a worst case, which the data's own rounding (the
//! moments' accumulation) only adds to, so the limit is where no digit is
//! promised, not where one is. Measured 2026-10-25 (task 278 (d4)): NIST's
//! hardest designs the normal equations can solve read `‖H⁻¹‖₁ u` at most
//! `2.4e-10` (Wampler's, `k = 5`, against the limit's `4.2e-3`) and
//! Longley's `9.6e-13` (`k = 6`), over every row of the stream; NIST's
//! Filip, past `1/u`, read from `0.47` to `45` (`k = 10`, limit `1.1e-3`);
//! and every factorization that succeeded on `x₁ = x₀ + eps·b` at `eps` of
//! `1e-11` and `1e-13` read at least `0.028` (`k = 3`, limit `0.011`; 7,545
//! solves over 20 streams of 800 rows, their first ten rows left out).
//!
//! **A factor that needed jitter** factors `A + δI`, a ridge the solver
//! chose with `δ` at least `1e-12` of the mean diagonal, and is judged as
//! the system it solves: `H = D⁻¹ (A + δI) D⁻¹` with `D = diag(√(a_ii + δ))`.
//! The ridge bounds that condition, but not inside the limit at every
//! order. On a matrix whose factorization fails as given -- one with a
//! direction of no variance, its smallest eigenvalue 0 to rounding -- the
//! first rung's `δ`, `1e-12` of the mean diagonal, leaves `‖H⁻¹‖₂` near
//! `1e12`, and `‖H⁻¹‖₁` at least that and up to `k` times it. The limit is
//! `1.7e12` at `k = 24`, below `1e12` from `k = 32` and `4.0e11` at `k = 50`,
//! so a first-rung solve can be past it from the twenties, and is from the
//! thirties. Such a solve was served unjudged, its jitter counted as a
//! failure; it is refused now, as a factor past the limit that needed none
//! is.

use super::SpdFactor;

/// The unit roundoff of a double, `2⁻⁵³`.
const U: f64 = f64::EPSILON / 2.0;

/// The iterations Hager's method takes at most (LAPACK's `ITMAX`).
const ITMAX: usize = 5;

/// The orders whose working vectors fit on the stack: a solve of up to 32
/// columns judges its condition without allocating, as `ewridge`'s per-row
/// cadence needs (`online-bench`'s allocation counts).
const STACK: usize = 32;

impl SpdFactor {
    /// `‖H⁻¹‖₁`, estimated, for `H = D⁻¹ (A + δI) D⁻¹` the equilibrated
    /// matrix of the system this factor holds, `A` row-major `k*k` as given
    /// to the factorization and `δ` its jitter ([`Self::jitter`], 0 for a
    /// factor that needed none), with `D = diag(√(a_ii + δ))`. A lower bound,
    /// usually exact (the module docs). Infinite where a product overflows
    /// or is not a number, which only a matrix as good as singular gives:
    /// `H`'s entries are at most 1 in size.
    pub fn equilibrated_inverse_norm(&self, a: &[f64]) -> f64 {
        self.with_work(a, |f, d, work| {
            norm1_estimate(d.len(), work, |x| f.equilibrated_solve(d, x))
        })
    }

    /// Run `f` with `D`'s diagonal, `√(a_ii + δ)` (`δ` the jitter, 0 where
    /// none was needed: the same bits as `√a_ii`), and a working space of `2k`,
    /// on the stack up to [`STACK`] columns; infinite where the factor's
    /// columns cannot be read as slices.
    fn with_work(&self, a: &[f64], f: impl FnOnce(&Self, &[f64], &mut [f64]) -> f64) -> f64 {
        let k = self.order();
        debug_assert_eq!(a.len(), k * k);
        if (0..k).any(|j| self.l.col(j).try_as_col_major().is_none()) {
            return f64::INFINITY;
        }
        let mut stack = [0.0f64; 3 * STACK];
        let mut heap = Vec::new();
        let buf: &mut [f64] = if k <= STACK {
            &mut stack[..3 * k]
        } else {
            heap.resize(3 * k, 0.0);
            &mut heap
        };
        let (d, work) = buf.split_at_mut(k);
        for (i, di) in d.iter_mut().enumerate() {
            *di = (a[i * k + i] + self.jitter).sqrt();
        }
        f(self, d, work)
    }

    /// Column `j` of `L` from its diagonal down.
    fn below(&self, j: usize) -> &[f64] {
        &self
            .l
            .col(j)
            .try_as_col_major()
            .expect("checked in `with_work`")
            .as_slice()[j..]
    }

    /// `x ← H⁻¹ x = D A⁻¹ D x`, from `L`: `D x`, then `L y = D x` forward by
    /// columns and `Lᵀ z = y` backward by dot products down each column,
    /// then `D z`. Nothing overflows on the way for any `A` a double holds:
    /// `L⁻¹ D = (D⁻¹ L)⁻¹` is the inverse of `H`'s own factor, so `y` is in
    /// `H`'s units, and `z = D⁻¹ H⁻¹ x` is at most `1.6e162` times them
    /// (`√a_ii ≥ 2⁻⁵³⁷`, the smallest subnormal's root).
    fn equilibrated_solve(&self, d: &[f64], x: &mut [f64]) {
        let k = x.len();
        for (v, di) in x.iter_mut().zip(d) {
            *v *= di;
        }
        for j in 0..k {
            let c = self.below(j);
            x[j] /= c[0];
            let xj = x[j];
            for (xi, &lij) in x[j + 1..].iter_mut().zip(&c[1..]) {
                *xi -= lij * xj;
            }
        }
        for j in (0..k).rev() {
            let c = self.below(j);
            let dot: f64 = x[j + 1..]
                .iter()
                .zip(&c[1..])
                .map(|(xi, lij)| xi * lij)
                .sum();
            x[j] = (x[j] - dot) / c[0];
        }
        for (v, di) in x.iter_mut().zip(d) {
            *v *= di;
        }
    }

    /// An upper bound on `‖H⁻¹‖₁` from the comparison matrix of `H`'s
    /// factor `R = D⁻¹ L` (the module docs, *The screen*). With `M(R) =
    /// D⁻¹ M(L)`, `M(R)⁻¹ 1 = M(L)⁻¹ d`, one forward substitution; and
    /// `M(R)⁻ᵀ 1 = D w` with `M(L)ᵀ w = 1`, one backward. Every term is
    /// nonnegative, so nothing cancels; infinite or NaN where a term
    /// overflows, which the caller reads as no bound. `z` and `w` are `k`
    /// long each.
    fn comparison_bound(&self, d: &[f64], z: &mut [f64], w: &mut [f64]) -> f64 {
        let k = d.len();
        z.copy_from_slice(d);
        for j in 0..k {
            let c = self.below(j);
            z[j] /= c[0];
            let zj = z[j];
            for (zi, &lij) in z[j + 1..].iter_mut().zip(&c[1..]) {
                *zi += lij.abs() * zj;
            }
        }
        for j in (0..k).rev() {
            let c = self.below(j);
            let dot: f64 = w[j + 1..]
                .iter()
                .zip(&c[1..])
                .map(|(wi, lij)| wi * lij.abs())
                .sum();
            w[j] = (1.0 + dot) / c[0];
        }
        // The largest entry, a NaN kept: `f64::max` would drop it, and a
        // NaN in `A`'s diagonal would read as a bound of 0.
        let largest = |m: f64, v: f64| if v > m || v.is_nan() { v } else { m };
        let rows = z.iter().fold(0.0f64, |m, &v| largest(m, v));
        let cols = w
            .iter()
            .zip(d)
            .fold(0.0f64, |m, (&v, &di)| largest(m, v * di));
        rows * cols
    }

    /// Whether this factor's system is past the condition at which its
    /// solve's forward-error bound reaches 1, so that no digit of the
    /// solution is guaranteed: `3 k γ_{3k+1} ‖H⁻¹‖₁ ≥ 1` (the module docs
    /// derive it). `a` is the matrix given to the factorization, row-major;
    /// a factor that needed jitter is judged as the `A + δI` it holds (the
    /// module docs). The comparison bound decides where it is inside the
    /// limit; the estimate decides the rest.
    pub fn carries_no_digit(&self, a: &[f64]) -> bool {
        let limit = digit_limit(self.order());
        let norm = self.judged_norm(a, limit);
        // An estimate that is NaN is refused, as one past the limit is.
        norm.is_nan() || norm >= limit
    }

    /// The `‖H⁻¹‖₁` a decision at `limit` is read from: the comparison
    /// bound where it is below `limit`, which then settles it, else the
    /// estimate.
    fn judged_norm(&self, a: &[f64], limit: f64) -> f64 {
        let k = self.order();
        self.with_work(a, |f, d, work| {
            let (z, w) = work.split_at_mut(k);
            let bound = f.comparison_bound(d, z, w);
            if bound < limit {
                return bound;
            }
            norm1_estimate(k, work, |x| f.equilibrated_solve(d, x))
        })
    }
}

/// The `‖H⁻¹‖₁` at which an order-`k` solve keeps no digit: `1 / (3 k
/// γ_{3k+1})`, `γ_m = m u / (1 − m u)` (the module docs derive it). 0, so
/// that every system is refused, at an order whose `γ` is no bound, which no
/// matrix a machine holds reaches.
fn digit_limit(k: usize) -> f64 {
    let m = (3 * k + 1) as f64 * U;
    if m >= 1.0 {
        return 0.0;
    }
    (1.0 - m) / (3.0 * k as f64 * m)
}

/// `‖B‖₁` of a symmetric `k × k` matrix `B`, estimated from products
/// `x ← B x` (`apply`), by Hager's method as Higham refined it: LAPACK's
/// `dlacn2` with `Bᵀ = B`. The first product is with the vector of `1/k`;
/// then, in turn, with the signs of the last product, and with the unit
/// vector at the largest entry of that, until the signs repeat, the
/// estimate stops growing, the largest entry stays where it was, or five
/// iterations have run. Last, a vector of alternating signs and growing
/// size, `(−1)ⁱ (1 + i/(k − 1))`, whose product's 1-norm times `2/(3k)` is
/// taken where it is larger: it catches the matrices that fool the
/// iteration (Higham, §15.3). Every candidate is `‖B v‖₁ / ‖v‖₁` for some
/// `v`, so the estimate is at most `‖B‖₁`; it is the largest of them, where
/// `dlacn2` keeps the last. Infinite where a product holds an infinity or a
/// NaN. `work` is `2k` long: the product and the signs.
fn norm1_estimate(k: usize, work: &mut [f64], mut apply: impl FnMut(&mut [f64])) -> f64 {
    if k == 0 {
        return 0.0;
    }
    let norm1 = |x: &[f64]| -> f64 {
        let s: f64 = x.iter().map(|v| v.abs()).sum();
        if s.is_finite() { s } else { f64::INFINITY }
    };
    let (x, signs) = work.split_at_mut(k);
    x.fill(1.0 / k as f64);
    apply(x);
    let mut est = norm1(x);
    if k == 1 || est.is_infinite() {
        return est;
    }
    for (s, &v) in signs.iter_mut().zip(x.iter()) {
        *s = sign(v);
    }
    x.copy_from_slice(signs);
    apply(x);
    let mut j = first_largest(x);
    for _ in 2..=ITMAX {
        x.fill(0.0);
        x[j] = 1.0;
        apply(x);
        let last = est;
        let now = norm1(x);
        est = est.max(now);
        if est.is_infinite() {
            return est;
        }
        let repeated = x.iter().zip(signs.iter()).all(|(&v, &s)| sign(v) == s);
        if repeated || now <= last {
            break;
        }
        for (s, &v) in signs.iter_mut().zip(x.iter()) {
            *s = sign(v);
        }
        x.copy_from_slice(signs);
        apply(x);
        let previous = j;
        j = first_largest(x);
        if x[previous].abs() == x[j].abs() {
            break;
        }
    }
    alternating(x);
    apply(x);
    est.max(2.0 * norm1(x) / (3 * k) as f64)
}

/// `+1` for a number at or above 0, `−1` below it and for a NaN: the sign
/// vector's entries, as `dlacn2` takes them (0 counts as positive).
fn sign(v: f64) -> f64 {
    if v >= 0.0 { 1.0 } else { -1.0 }
}

/// The index of the entry largest in absolute value, the first of several
/// equal ones, as `idamax` gives it.
fn first_largest(x: &[f64]) -> usize {
    let mut j = 0;
    for (i, v) in x.iter().enumerate() {
        if v.abs() > x[j].abs() {
            j = i;
        }
    }
    j
}

/// `x_i = (−1)ⁱ (1 + i/(k − 1))`, from `+1` at the first entry to `±2` at
/// the last, for `k = x.len() ≥ 2`: the vector `dlacn2` tries last.
fn alternating(x: &mut [f64]) {
    let step = 1.0 / (x.len() - 1) as f64;
    for (i, v) in x.iter_mut().enumerate() {
        let size = 1.0 + i as f64 * step;
        *v = if i % 2 == 0 { size } else { -size };
    }
}

#[cfg(test)]
#[path = "condition_tests.rs"]
mod tests;
