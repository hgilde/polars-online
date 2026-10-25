//! Dense linear algebra for the tests' oracles, on `faer`'s own
//! decompositions (docs/PLAN.md task 169).
//!
//! Every Cholesky a model runs goes through `solve.rs` -- `SpdFactor`,
//! `solve_spd` and `quad_forms_logdet`, faer's `llt` behind a jitter ladder
//! -- and a test's expected value must share no arithmetic with the code it
//! checks (CLAUDE.md, Style; docs/TESTING.md). So nothing here calls
//! `solve.rs`, nor any Cholesky:
//!
//! - `solve`, `solve_columns`, `inverse` and `quad_form` run faer's
//!   partial-pivot LU, which
//!   no model runs;
//! - `log_det`, `gaussian_log_density` and `sym_eigen` run faer's
//!   self-adjoint eigensolver. `ewcov`'s `Pca::of` and `rcov`'s `clip_psd`
//!   run it too, so `sym_eigen` is an oracle for neither: `ewcov`'s sign
//!   test reads it for that reason, as the raw output `Pca::of` starts from.
//!
//! - `sum` and `dot` are exactly rounded, by error-free transformations
//!   alone (`+`, `−`, `×`, no fused multiply-add), the oracle for a matrix
//!   product (task 258): every product the library runs is faer's, and its
//!   kernels sum in their own blocked order, which these share nothing with.
//!
//! A matrix is row-major `n x n`, the layout of `solve.rs` and of every
//! model's moments; `n` is the length of the vector beside it, or the
//! square root of the matrix's length. Compiled for tests only (`lib.rs`).

use faer::Side;
use faer::linalg::solvers::{DenseSolveCore, Solve};
use faer::prelude::*;

/// faer's matrix of `a`, row-major `n x n`.
fn dense(a: &[f64], n: usize) -> Mat<f64> {
    assert_eq!(
        a.len(),
        n * n,
        "an {n} x {n} matrix, but {} entries",
        a.len()
    );
    Mat::from_fn(n, n, |i, j| a[i * n + j])
}

/// The order of a square matrix of `len` entries.
fn order(len: usize) -> usize {
    let n = len.isqrt();
    assert_eq!(n * n, len, "a square matrix, but {len} entries");
    n
}

/// `x` with `A x = b`, for a square `A`, symmetric or not: faer's
/// partial-pivot LU, which no model runs.
pub(crate) fn solve(a: &[f64], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let x = dense(a, n)
        .partial_piv_lu()
        .solve(Mat::from_fn(n, 1, |i, _| b[i]));
    (0..n).map(|i| x[(i, 0)]).collect()
}

/// `X` with `A X = B`, for a square `A` and the columns `B` of a row-major
/// `n x t`, returned the same way: one partial-pivot LU of `A`, as
/// [`solve`] runs it, for every column.
pub(crate) fn solve_columns(a: &[f64], b: &[f64], t: usize) -> Vec<f64> {
    let n = order(a.len());
    assert_eq!(b.len(), n * t, "an {n} x {t} right-hand side");
    let x = dense(a, n)
        .partial_piv_lu()
        .solve(Mat::from_fn(n, t, |i, c| b[i * t + c]));
    (0..n * t).map(|ic| x[(ic / t, ic % t)]).collect()
}

/// `A⁻¹`, row-major, for a square `A`: the inverse faer forms from its
/// partial-pivot LU, which no model runs.
pub(crate) fn inverse(a: &[f64]) -> Vec<f64> {
    let n = order(a.len());
    let inv = dense(a, n).partial_piv_lu().inverse();
    (0..n * n).map(|ij| inv[(ij / n, ij % n)]).collect()
}

/// `uᵀA⁻¹u`: `A⁻¹u` by [`solve`], the LU, and the products summed in index
/// order.
pub(crate) fn quad_form(a: &[f64], u: &[f64]) -> f64 {
    u.iter().zip(solve(a, u)).map(|(ui, xi)| ui * xi).sum()
}

/// The eigenvalues of a symmetric `A`, ascending, and a unit eigenvector
/// for each (`vectors[j]` belongs to `values[j]`): faer's self-adjoint
/// eigensolver on the lower triangle, which no Cholesky shares. A vector's
/// sign is faer's, as `Pca::of` receives it.
pub(crate) fn sym_eigen(a: &[f64]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = order(a.len());
    let evd = dense(a, n)
        .self_adjoint_eigen(Side::Lower)
        .expect("faer's self-adjoint eigensolver failed");
    let values = (0..n).map(|j| evd.S()[j]).collect();
    let vectors = (0..n)
        .map(|j| (0..n).map(|i| evd.U()[(i, j)]).collect())
        .collect();
    (values, vectors)
}

/// [`sym_eigen`] of a symmetric positive definite `A`; any other panics,
/// naming its eigenvalues, rather than hand a test a NaN to compare.
fn spd_eigen(a: &[f64]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let (values, vectors) = sym_eigen(a);
    assert!(
        values.iter().all(|&l| l > 0.0),
        "not positive definite: eigenvalues {values:?}"
    );
    (values, vectors)
}

/// `ln det A` of a symmetric positive definite `A`: `Σ ln λᵢ` over its
/// eigenvalues, ascending, from [`sym_eigen`]; no product of pivots to
/// overflow, and no Cholesky.
pub(crate) fn log_det(a: &[f64]) -> f64 {
    spd_eigen(a).0.iter().map(|l| l.ln()).sum()
}

/// The Gaussian log density of a deviation `δ` from the mean under a
/// symmetric positive definite covariance `Σ`,
/// `−½ (n ln 2π + ln det Σ + δᵀΣ⁻¹δ)`, both terms read off one
/// eigendecomposition `Σ = Σᵢ λᵢ vᵢvᵢᵀ` ([`sym_eigen`]): `ln det Σ = Σ ln λᵢ`
/// and `δᵀΣ⁻¹δ = Σ (vᵢᵀδ)²/λᵢ`. No Cholesky, and no LU either.
pub(crate) fn gaussian_log_density(cov: &[f64], delta: &[f64]) -> f64 {
    let n = delta.len();
    let (values, vectors) = spd_eigen(cov);
    assert_eq!(values.len(), n, "a covariance of order {}", values.len());
    let log_det: f64 = values.iter().map(|l| l.ln()).sum();
    let quad: f64 = values
        .iter()
        .zip(&vectors)
        .map(|(l, v)| {
            let p: f64 = v.iter().zip(delta).map(|(vi, di)| vi * di).sum();
            p * p / l
        })
        .sum();
    -0.5 * (n as f64 * std::f64::consts::TAU.ln() + log_det + quad)
}

/// `(p, e)` with `p = fl(a·b)` and `p + e = a·b` exactly: Dekker's product
/// (Dekker 1971, "A floating-point technique for extending the available
/// precision", Numer. Math. 18), each factor split by Veltkamp's
/// `2^27 + 1` into halves whose products are exact. Exact while no product
/// or split overflows (`|a|, |b| < 2^995`) and `e` is not subnormal
/// (`|a·b| > 2^-969`); the tests stay inside both.
fn two_product(a: f64, b: f64) -> (f64, f64) {
    const SPLIT: f64 = 134_217_729.0; // 2^27 + 1
    let split = |x: f64| {
        let c = SPLIT * x;
        let hi = c - (c - x);
        (hi, x - hi)
    };
    let p = a * b;
    let ((ah, al), (bh, bl)) = (split(a), split(b));
    let e = ((ah * bh - p) + ah * bl + al * bh) + al * bl;
    (p, e)
}

/// `Σ xs` rounded once, to nearest: Shewchuk's exact partials (Shewchuk
/// 1997, "Adaptive precision floating-point arithmetic and fast robust
/// geometric predicates", Discrete Comput. Geom. 18), every partial
/// nonoverlapping and their sum exact, then rounded as CPython's
/// `math.fsum` rounds them: the top partials added down to the first
/// inexact step, and a half-way case decided by the sign of the partial
/// below it. Finite inputs whose partial sums do not overflow.
pub(crate) fn sum(xs: impl IntoIterator<Item = f64>) -> f64 {
    let mut partials: Vec<f64> = Vec::new();
    for mut x in xs {
        let mut kept = 0;
        for j in 0..partials.len() {
            let mut y = partials[j];
            if x.abs() < y.abs() {
                std::mem::swap(&mut x, &mut y);
            }
            let hi = x + y;
            let lo = y - (hi - x);
            if lo != 0.0 {
                partials[kept] = lo;
                kept += 1;
            }
            x = hi;
        }
        partials.truncate(kept);
        partials.push(x);
    }
    let Some(mut hi) = partials.pop() else {
        return 0.0;
    };
    let mut lo = 0.0;
    while let Some(y) = partials.pop() {
        let x = hi;
        hi = x + y;
        lo = y - (hi - x);
        if lo != 0.0 {
            break;
        }
    }
    if let Some(&below) = partials.last()
        && ((lo < 0.0 && below < 0.0) || (lo > 0.0 && below > 0.0))
    {
        let y = lo * 2.0;
        let x = hi + y;
        if y == x - hi {
            hi = x;
        }
    }
    hi
}

/// `Σ xᵢ·yᵢ` rounded once, to nearest: each product split exactly
/// ([`two_product`]) and the `2n` parts summed exactly ([`sum`]).
pub(crate) fn dot(pairs: impl IntoIterator<Item = (f64, f64)>) -> f64 {
    sum(pairs.into_iter().flat_map(|(x, y)| {
        let (p, e) = two_product(x, y);
        [p, e]
    }))
}

/// `Σ aᵢ·bᵢ·cᵢ` rounded once, to nearest: `aᵢ·bᵢ` split exactly into
/// `p + e` ([`two_product`]), then `p·cᵢ` and `e·cᵢ` split in turn, and the
/// `4n` parts summed exactly ([`sum`]). A product's scale `α` and two of
/// its factors, `α·xᵢ·yᵢ`, with no rounding of `α·xᵢ` between.
pub(crate) fn dot3(triples: impl IntoIterator<Item = (f64, f64, f64)>) -> f64 {
    sum(triples.into_iter().flat_map(|(a, b, c)| {
        let (p, e) = two_product(a, b);
        let ((pp, pe), (ep, ee)) = (two_product(p, c), two_product(e, c));
        [pp, pe, ep, ee]
    }))
}

// ---------------------------------------------------------------------------
// Rounding-error bounds (docs/PLAN.md task 274).
//
// A test that holds a model to an oracle holds it to a bound on how far the
// two may part: the model's rounding, derived from its algorithm, plus the
// oracle's own. A fixed tolerance such as `1e-9` is either too tight for an
// ill-conditioned case or a million times too loose for a well-conditioned
// one, where a kernel a hundred times past its rounding passes it. The
// helpers below are the standard results each such bound rests on, written
// once; each test states which it uses and how its terms are counted.
//
// The references are Higham, *Accuracy and Stability of Numerical
// Algorithms*, 2nd ed. (SIAM 2002), cited as "Higham", and the eigenvalue
// perturbation theorems of Weyl and Ostrowski (Horn and Johnson, *Matrix
// Analysis*, 2nd ed., Theorems 4.3.1 and 4.5.9).
// ---------------------------------------------------------------------------

/// The unit roundoff of `f64`, `u = 2⁻⁵³`: every basic operation returns
/// the exact result times `1 + δ`, `|δ| ≤ u`, to nearest (Higham (2.4)).
pub(crate) const U: f64 = f64::EPSILON / 2.0;

/// The units of `u` a call to the platform's `ln`, `exp`, `exp2` or `powf`
/// may be off its exact result, relative: two ulps, an ulp being at most
/// `2u` of the result. Apple's and glibc's are within one ulp; Windows'
/// `ucrtbase.dll` changes with the build and has not been measured to the
/// ulp (docs/PLAN.md §25), so the bounds leave it a second.
pub(crate) const LIBM: f64 = 4.0;

/// `γ_m = m·u / (1 − m·u)`: a product of `m` factors `(1 + δᵢ)^(±1)`, each
/// `|δᵢ| ≤ u`, is `1 + θ` with `|θ| ≤ γ_m` (Higham, Lemma 3.1). So `m`
/// roundings in a row move a result by at most `γ_m` of itself.
///
/// Returned no smaller than the exact `γ_m`: `m·u` is exact, and the
/// subtraction and the division each round by at most `u` of their result,
/// which the factor `1 + 4u` (itself rounded once) more than restores.
pub(crate) fn gamma(m: usize) -> f64 {
    let mu = m as f64 * U;
    assert!(mu < 0.5, "γ_{m} is no bound");
    mu / (1.0 - mu) * (1.0 + 4.0 * U)
}

/// `1 + γ_{m+4}`, the factor that lifts a bound evaluated in at most `m`
/// roundings of sums and products of nonnegative numbers to no less than
/// its exact value (task 273 (a3)). Each such rounding moves the running
/// value by at most `u` of itself, so the computed bound is at least the
/// exact one times `(1 − u)^m`; `1 + γ_{m+4}`, rounded to the nearest double
/// above 1 (it can lose up to `u`) and multiplied in (one more rounding),
/// more than restores it. A bound is a bound as computed: on a CPU without
/// FMA, the LU solve of a 3 x 3 Hilbert system made an error equal to its
/// a-posteriori bound in exact arithmetic, and the bound, evaluated in
/// floating point from an inverse also taken without FMA, came out 14
/// units of `u` below it (cpu-paths.yml's Sandy Bridge leg, 2026-10-24;
/// `the_solve_bound_holds_a_solve_without_fma`).
pub(crate) fn up(m: usize) -> f64 {
    1.0 + gamma(m + 4)
}

/// `γ_n·Σ|xᵢ·yᵢ|`, the most a dot product of `n` terms summed in any order
/// can be off the exact sum (Higham (3.5), for recursive summation; any
/// other order's bound is no larger, Higham §3.1).
pub(crate) fn dot_error(pairs: impl IntoIterator<Item = (f64, f64)>) -> f64 {
    let (n, abs) = pairs
        .into_iter()
        .fold((0usize, 0.0), |(n, s), (x, y)| (n + 1, s + (x * y).abs()));
    // `Σ|xᵢyᵢ|` took `2n` roundings, the product with `γ_n` one more.
    gamma(n) * abs * up(2 * n + 1)
}

/// `Σ xᵢ·yᵢ` summed left to right from 0, as an iterator's `sum` does, and
/// the most its rounding can have moved it, read off the run: each product
/// within `u` of itself and each addition within `u` of the partial sum it
/// makes, so the error is at most `u·(Σ|fl(xᵢyᵢ)| + Σ|ŝᵢ|)` to first order
/// (Higham §3.3, the running error bound). Where the terms cancel, the
/// partial sums are smaller than `Σ|xᵢyᵢ|`, and so is the bound.
pub(crate) fn running_dot(pairs: impl IntoIterator<Item = (f64, f64)>) -> (f64, f64) {
    let (mut s, mut e, mut n) = (0.0f64, 0.0f64, 0usize);
    for (i, (x, y)) in pairs.into_iter().enumerate() {
        let p = x * y;
        s += p;
        e += p.abs() + if i > 0 { s.abs() } else { 0.0 };
        n = i + 1;
    }
    // `|xᵢyᵢ| ≤ |fl(xᵢyᵢ)|/(1 − u)`, and `e` took `2n` additions.
    (s, e * U * (1.0 + 4.0 * U) * up(2 * n + 2))
}

/// The 2-norm condition number `κ₂(A) = λ_max / λ_min` of a symmetric
/// positive definite `A`, from [`sym_eigen`]; each eigenvalue moved by its
/// [`eigen_error`] the way that makes `κ` larger, so the computed `κ` is no
/// smaller than the exact one.
pub(crate) fn condition(a: &[f64]) -> f64 {
    let (values, errors) = eigen_error(a);
    let n = values.len();
    let least = values[0] - errors[0];
    assert!(least > 0.0, "not positive definite: eigenvalues {values:?}");
    // Rounded up: the sum, the difference and the quotient.
    (values[n - 1] + errors[n - 1]) / (least * (1.0 - 4.0 * U)) * up(3)
}

/// The normwise forward error of a backward-stable solve, relative:
/// `(A + ΔA) x̂ = b + Δb` with `‖ΔA‖ ≤ η‖A‖` and `‖Δb‖ ≤ η‖b‖` gives
/// `‖x̂ − x‖ / ‖x‖ ≤ 2κη / (1 − κη)` while `κη < 1` (Higham, Theorem 7.2,
/// with `E = A` and `f = b`). With `Δb = 0` the bound is half of it, and a
/// caller halves it.
pub(crate) fn forward_bound(kappa: f64, eta: f64) -> f64 {
    let ke = kappa * eta * (1.0 + 2.0 * U);
    assert!(ke < 1.0, "a vacuous bound: κ {kappa:e} times η {eta:e}");
    // `κη` rounded up, then `1 − κη` taken down and the quotient rounded up.
    2.0 * ke / ((1.0 - ke) * (1.0 - 4.0 * U)) * up(2)
}

/// `Σᵢⱼ |xᵢ|·Eᵢⱼ·|yⱼ|` for a row-major `n x n` `E`.
pub(crate) fn abs_form(x: &[f64], e: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    assert_eq!(e.len(), n * n);
    (0..n)
        .map(|i| x[i].abs() * (0..n).map(|j| e[i * n + j] * y[j].abs()).sum::<f64>())
        .sum()
}

/// `|M|·v` for a row-major `n x n` `M`.
pub(crate) fn abs_times(m: &[f64], v: &[f64]) -> Vec<f64> {
    let n = v.len();
    (0..n)
        .map(|i| (0..n).map(|j| m[i * n + j].abs() * v[j]).sum())
        .collect()
}

/// The componentwise forward error of a perturbed solve: if `(A + ΔA)·x̂ =
/// b + Δb` with `|ΔA| ≤ E` and `|Δb| ≤ f` entrywise, then `x̂ − x = A⁻¹(Δb −
/// ΔA·x̂)`, exactly, so `|x̂ − x| ≤ |A⁻¹|·(E·|x̂| + f)` (Higham §7.2,
/// (7.11)). `A⁻¹` is the oracle's [`inverse`], to first order in its own
/// rounding; `x̂` may be any of the solutions it is close to, to first
/// order.
pub(crate) fn forward_error(a: &[f64], x_hat: &[f64], e: &[f64], f: &[f64]) -> Vec<f64> {
    let n = x_hat.len();
    let inv = abs_inverse(a);
    let ex: Vec<f64> = (0..n)
        .map(|i| (0..n).map(|j| e[i * n + j] * x_hat[j].abs()).sum::<f64>() + f[i])
        .collect();
    // `E|x̂| + f` took `2n` roundings, `|A⁻¹|·` that `2n` more.
    let lift = up(4 * n);
    abs_times(&inv, &ex).into_iter().map(|v| v * lift).collect()
}

/// The componentwise backward error of a Cholesky solve of `A x = b`, as
/// `solve.rs` runs it (faer's Cholesky, then its two triangular solves):
/// `(A + ΔA) x̂ = b` with `|ΔA| ≤ γ_{3n+1}·|L̂|·|L̂ᵀ|` (Higham, Theorem 10.4),
/// and `(|L̂||L̂ᵀ|)ᵢⱼ ≤ ‖l̂ᵢ‖·‖l̂ⱼ‖` by Cauchy–Schwarz, where row `i` of `L̂`
/// has squared norm `(L̂L̂ᵀ)ᵢᵢ ≤ aᵢᵢ(1 + γ_{n+1})` (Theorem 10.3). So
/// `|ΔA|ᵢⱼ ≤ γ_{3n+1}(1 + γ_{n+1})·√(aᵢᵢaⱼⱼ)`, which no rescaling of the
/// rows and columns of `A` changes: a model that solves its system
/// standardized has the same bound in the system's own units. Valid for
/// a factorization the jitter ladder took as given, its first rung.
pub(crate) fn cholesky_backward(a: &[f64]) -> Vec<f64> {
    let n = order(a.len());
    // Rounded up: the two square roots and the four products.
    let g = gamma(3 * n + 1) * (1.0 + gamma(n + 1)) * up(6);
    let d: Vec<f64> = (0..n).map(|i| a[i * n + i].abs().sqrt()).collect();
    (0..n * n).map(|ij| g * d[ij / n] * d[ij % n]).collect()
}

/// The oracle's own error in [`solve`]'s `x̂` of `A x = b`, from its
/// residual: `x − x̂ = A⁻¹ r` with `r = b − A x̂`, so `|x − x̂| ≤ |A⁻¹|·|r|`,
/// with `|A⁻¹|` bounded entry by entry ([`abs_inverse`]). Each `rᵢ` is
/// summed exactly and rounded once ([`dot`]), so it is the exact residual to
/// `u` of itself. An a-posteriori bound, the error the LU actually made, not
/// the worst it could; it can be the error itself, where `r` has one sign
/// against each row of `A⁻¹`, so it is lifted by its own evaluation's
/// rounding ([`up`]).
pub(crate) fn solve_error(a: &[f64], b: &[f64], x_hat: &[f64]) -> Vec<f64> {
    solve_error_with(a, b, x_hat, &abs_inverse(a))
}

/// [`solve_error`] with `|A⁻¹|` bounded by `abs_inv`.
fn solve_error_with(a: &[f64], b: &[f64], x_hat: &[f64], abs_inv: &[f64]) -> Vec<f64> {
    let n = b.len();
    let r: Vec<f64> = (0..n)
        .map(|i| {
            dot(std::iter::once((b[i], 1.0)).chain((0..n).map(|j| (-a[i * n + j], x_hat[j])))).abs()
                * (1.0 + 2.0 * U)
        })
        .collect();
    let lift = up(2 * n);
    abs_times(abs_inv, &r)
        .into_iter()
        .map(|v| v * lift)
        .collect()
}

/// [`inverse`]'s `X̂` and, entry by entry, a bound on `|X̂ − A⁻¹|`, from the
/// residual `R = AX̂ − I`, each entry summed exactly and rounded once
/// ([`dot`]): `X̂ − A⁻¹ = A⁻¹R`, so `D = |X̂ − A⁻¹| ≤ |A⁻¹||R| ≤ F + D|R|`
/// with `F = |X̂||R|`. With `ρ` the largest column sum of `|R|`, the largest
/// entry `dᵢ` of row `i` of `D` has `dᵢ ≤ maxⱼFᵢⱼ + dᵢρ`, so for `ρ < 1`
/// `Dᵢⱼ ≤ Fᵢⱼ + ρ·maxₖFᵢₖ/(1 − ρ)`: no first-order step, and lifted by its
/// own evaluation's rounding ([`up`]).
fn inverse_with_error(a: &[f64]) -> (Vec<f64>, Vec<f64>) {
    inverse_with_error_of(a, inverse(a))
}

/// [`inverse_with_error`] for any computed inverse `x` of `a`.
fn inverse_with_error_of(a: &[f64], x: Vec<f64>) -> (Vec<f64>, Vec<f64>) {
    let n = order(a.len());
    let r: Vec<f64> = (0..n * n)
        .map(|ij| {
            let (i, j) = (ij / n, ij % n);
            dot((0..n)
                .map(|m| (a[i * n + m], x[m * n + j]))
                .chain(std::iter::once((-f64::from(i == j), 1.0))))
            .abs()
                * (1.0 + 2.0 * U)
        })
        .collect();
    let f: Vec<f64> = (0..n * n)
        .map(|ij| {
            let (i, j) = (ij / n, ij % n);
            (0..n).map(|m| x[i * n + m].abs() * r[m * n + j]).sum()
        })
        .collect();
    let rho = (0..n)
        .map(|j| (0..n).map(|k| r[k * n + j]).sum::<f64>())
        .fold(0.0, f64::max)
        * up(n);
    assert!(rho < 0.5, "the inverse's residual is not small: ρ {rho:e}");
    let spill = rho / ((1.0 - rho) * (1.0 - 2.0 * U));
    let lift = up(2 * n + 4);
    let d = (0..n * n)
        .map(|ij| {
            let row = &f[(ij / n) * n..(ij / n + 1) * n];
            (f[ij] + spill * row.iter().copied().fold(0.0, f64::max)) * lift
        })
        .collect();
    (x, d)
}

/// The oracle's own error in [`inverse`]'s `X̂`, entry by entry, row-major
/// ([`inverse_with_error`]).
pub(crate) fn inverse_error(a: &[f64]) -> Vec<f64> {
    inverse_with_error(a).1
}

/// `|A⁻¹|` bounded above entry by entry, row-major: `|X̂| + |X̂ − A⁻¹|`
/// ([`inverse_with_error`]), for a bound that multiplies by `|A⁻¹|`, where
/// `|X̂|` alone could be below it.
pub(crate) fn abs_inverse(a: &[f64]) -> Vec<f64> {
    abs_inverse_of(a, inverse(a))
}

/// [`abs_inverse`] from any computed inverse `x` of `a`.
fn abs_inverse_of(a: &[f64], x: Vec<f64>) -> Vec<f64> {
    let (x, d) = inverse_with_error_of(a, x);
    x.iter()
        .zip(&d)
        .map(|(v, e)| (v.abs() + e) * (1.0 + 2.0 * U))
        .collect()
}

/// The most a model's `A⁻¹` may be off when it solves `A X = I` by
/// Cholesky, column by column, as `EwCov::precision` does: each column
/// `x̂ⱼ` has `(A + ΔAⱼ)x̂ⱼ = eⱼ`, `|ΔAⱼ| ≤ E` ([`cholesky_backward`]), so
/// `|X̂ − A⁻¹| ≤ |A⁻¹|·E·|A⁻¹|` to first order ([`forward_error`] column by
/// column). Row-major, entry by entry.
pub(crate) fn cholesky_inverse_error(a: &[f64]) -> Vec<f64> {
    let n = order(a.len());
    let e = cholesky_backward(a);
    let abs = abs_inverse(a);
    let lift = up(4 * n);
    let ea: Vec<f64> = (0..n * n)
        .map(|ij| {
            let (i, j) = (ij / n, ij % n);
            (0..n).map(|m| e[i * n + m] * abs[m * n + j]).sum()
        })
        .collect();
    (0..n * n)
        .map(|ij| {
            let (i, j) = (ij / n, ij % n);
            (0..n).map(|m| abs[i * n + m] * ea[m * n + j]).sum::<f64>() * lift
        })
        .collect()
}

/// The most a model's `dᵀA⁻¹d` may be off, computed as `solve.rs`'s
/// `quad_forms` computes it -- `x̂` from a solve whose backward error is
/// at most `E`, then `dᵀx̂` summed: `dᵀx̂ − dᵀx = −xᵀ ΔA x̂` exactly (`x =
/// A⁻¹d`, and `A⁻¹` symmetric), at most `|x|ᵀ E |x|` to first order, and
/// the sum's own `γ_n Σ|dᵢ x̂ᵢ|` ([`dot_error`]).
pub(crate) fn quad_error(a: &[f64], d: &[f64], e: &[f64]) -> f64 {
    let n = d.len();
    // `|x|` bounded by the oracle's `|x̂|` and its own error.
    let x = solve(a, d);
    let x: Vec<f64> = x
        .iter()
        .zip(solve_error(a, d, &x))
        .map(|(v, s)| (v.abs() + s) * (1.0 + 2.0 * U))
        .collect();
    abs_form(&x, e, &x) * up(4 * n) + dot_error(d.iter().copied().zip(x.iter().copied()))
}

/// The oracle's own error in [`quad_form`]'s `uᵀx̂`: `uᵀ(x − x̂) = xᵀr`
/// for the residual `r` of `x̂` (`A` symmetric), at most `|x̂|ᵀ|r|` to first
/// order with `r` from [`solve_error`]'s exact sum, and the sum's own
/// [`dot_error`].
pub(crate) fn quad_form_error(a: &[f64], u: &[f64]) -> f64 {
    let n = u.len();
    let x = solve(a, u);
    // `|x|`, the exact solution's, bounded by `|x̂|` and its own error.
    let ax: Vec<f64> = x
        .iter()
        .zip(solve_error(a, u, &x))
        .map(|(v, s)| (v.abs() + s) * (1.0 + 2.0 * U))
        .collect();
    let r: f64 = (0..n)
        .map(|i| {
            let ri = dot(std::iter::once((u[i], 1.0)).chain((0..n).map(|j| (-a[i * n + j], x[j]))));
            ri.abs() * (1.0 + 2.0 * U) * ax[i]
        })
        .sum();
    r * up(2 * n) + dot_error(u.iter().copied().zip(x.iter().copied()))
}

/// [`sym_eigen`]'s eigenvalues and vectors, with `‖E‖_F` and `η = ‖G −
/// I‖_F` for `E = V̂Λ̂V̂ᵀ − A` and `G = V̂ᵀV̂`, each entry summed exactly and
/// rounded once ([`dot3`], [`dot`]), the norms then rounded up by their
/// own `γ`.
fn eigen_residuals(a: &[f64]) -> (Vec<f64>, Vec<Vec<f64>>, f64, f64) {
    let n = order(a.len());
    let (values, vectors) = sym_eigen(a);
    let slack = 1.0 + gamma(2 * n * n + 2);
    let mut e2 = 0.0;
    let mut g2 = 0.0;
    for i in 0..n {
        for j in 0..n {
            let e = dot3(
                (0..n)
                    .map(|k| (vectors[k][i], values[k], vectors[k][j]))
                    .chain(std::iter::once((-a[i * n + j], 1.0, 1.0))),
            );
            let g = dot((0..n)
                .map(|k| (vectors[i][k], vectors[j][k]))
                .chain(std::iter::once((-f64::from(i == j), 1.0))));
            e2 += e * e;
            g2 += g * g;
        }
    }
    let (e, eta) = (e2.sqrt() * slack, g2.sqrt() * slack);
    assert!(eta < 0.5, "the vectors are not near orthonormal: η {eta:e}");
    (values, vectors, e, eta)
}

/// [`sym_eigen`]'s eigenvalues and, for each, a bound on its distance
/// from `A`'s own, a posteriori: with `V̂` the computed vectors, `E = V̂Λ̂V̂ᵀ
/// − A` and `G = V̂ᵀV̂` ([`eigen_residuals`]), `V̂Λ̂V̂ᵀ = Q·(PΛ̂P)·Qᵀ` for the
/// polar factors `V̂ = QP`, `P² = G`, so by Ostrowski's theorem its `j`-th
/// eigenvalue is `θⱼλ̂ⱼ` with `θⱼ` within `η = ‖G − I‖_F` of 1, and by
/// Weyl's `A`'s `j`-th is within `‖E‖₂ ≤ ‖E‖_F` of that: `|λⱼ − λ̂ⱼ| ≤
/// ‖E‖_F + η|λ̂ⱼ|`.
pub(crate) fn eigen_error(a: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let (values, _, e, eta) = eigen_residuals(a);
    let errors = values.iter().map(|l| (e + eta * l.abs()) * up(2)).collect();
    (values, errors)
}

/// The oracle's own error in [`log_det`]'s `Σ ln λ̂ⱼ`: each `λ̂ⱼ` within
/// `δⱼ` of `λⱼ` ([`eigen_error`]) moves its log by at most `δⱼ / (λ̂ⱼ −
/// δⱼ)` (`|ln(1 + t)| ≤ |t| / (1 − |t|)`), each `ln` is the platform's
/// ([`LIBM`]) and the sum rounds `n − 1` times.
pub(crate) fn log_det_error(a: &[f64]) -> f64 {
    let (values, errors) = eigen_error(a);
    let n = values.len();
    let mut total = 0.0;
    let mut abs = 0.0;
    for (l, d) in values.iter().zip(&errors) {
        assert!(*l > *d, "an eigenvalue {l:e} within its error {d:e} of 0");
        total += d / ((l - d) * (1.0 - 2.0 * U));
        abs += l.ln().abs();
    }
    // The quotients' sum took `2n` roundings; `abs`, a size, is lifted by
    // the platform's `ln` with `LIBM` and its sum by `γ_n`.
    total * up(2 * n) + (LIBM * U + gamma(n)) * abs * (1.0 + gamma(n)) * up(4)
}

/// The most a model's `ln det A` may be off, taken as `solve.rs`'s
/// `SpdFactor::log_det` takes it, `2 Σ ln l̂ᵢᵢ` from faer's Cholesky:
/// `L̂L̂ᵀ = A + ΔA` with `|ΔA| ≤ γ_{n+1}|L̂||L̂ᵀ| ≤ γ_{n+1}(1 + γ_{n+1})·
/// √(aᵢᵢaⱼⱼ)` (Higham, Theorem 10.3, and [`cholesky_backward`]'s
/// Cauchy–Schwarz step), so `ln det(A + ΔA) − ln det A = tr(A⁻¹ΔA)` to
/// first order, at most `Σᵢⱼ |A⁻¹|ᵢⱼ·|ΔA|ᵢⱼ`; then each `ln l̂ᵢᵢ` is the
/// platform's ([`LIBM`]) and the sum rounds `n − 1` times, on terms with
/// `1/(A⁻¹)ᵢᵢ ≤ l̂ᵢᵢ² ≤ aᵢᵢ` (each pivot is a Schur complement's, and the
/// inverse of a leading block has a diagonal no larger than `A⁻¹`'s).
pub(crate) fn cholesky_log_det_error(a: &[f64]) -> f64 {
    let n = order(a.len());
    let inv = abs_inverse(a);
    let g = gamma(n + 1) * (1.0 + gamma(n + 1));
    let d: Vec<f64> = (0..n).map(|i| a[i * n + i].sqrt()).collect();
    let trace: f64 = (0..n * n)
        .map(|ij| inv[ij].abs() * g * d[ij / n] * d[ij % n])
        .sum();
    let logs: f64 = (0..n)
        .map(|i| 2.0 * (0.5 * a[i * n + i].ln().abs()).max(0.5 * inv[i * n + i].ln().abs()))
        .sum();
    trace * up(2 * n * n + 4) + (LIBM * U + gamma(n)) * logs * (1.0 + gamma(n)) * up(4)
}

/// The oracle's own error in [`gaussian_log_density`]: half the
/// [`log_det_error`] and half the error of its quadratic form
/// `q̃ = Σⱼ pⱼ²/λ̂ⱼ`, `pⱼ = v̂ⱼᵀδ`. Before its rounding, `q̃ = δᵀV̂Λ̂⁻¹V̂ᵀδ`;
/// with `B = V̂Λ̂V̂ᵀ = A + E` and `G = V̂ᵀV̂ = (I − F)⁻¹`, `‖F‖ ≤ η/(1 − η)`
/// ([`eigen_error`]'s `E` and `η`), `B⁻¹ = V̂G⁻¹Λ̂⁻¹G⁻¹V̂ᵀ`, so `q̃ −
/// δᵀB⁻¹δ = pᵀ(FΛ̂⁻¹ + Λ̂⁻¹F − FΛ̂⁻¹F)p`, and `δᵀB⁻¹δ − δᵀA⁻¹δ =
/// −δᵀB⁻¹EA⁻¹δ`, at most `‖E‖·‖x‖²` (`x = A⁻¹δ`) to first order. Its
/// rounding: each `pⱼ` a dot product ([`dot_error`]), squared, divided and
/// summed; and the combination's own: the platform's `ln 2π` and the
/// three terms' sum.
pub(crate) fn gaussian_log_density_error(cov: &[f64], delta: &[f64]) -> f64 {
    let n = delta.len();
    let (values, vectors, e, eta) = eigen_residuals(cov);
    let f = eta / (1.0 - eta);
    let x = solve(cov, delta);
    let x2: f64 = x.iter().map(|v| v * v).sum();
    let p: Vec<f64> = vectors
        .iter()
        .map(|v| v.iter().zip(delta).map(|(vi, di)| vi * di).sum())
        .collect();
    let pn: f64 = p.iter().map(|v| v * v).sum::<f64>().sqrt();
    let lp: f64 = p
        .iter()
        .zip(&values)
        .map(|(pj, l)| (pj / l).powi(2))
        .sum::<f64>()
        .sqrt();
    // `|pᵀ(FΛ̂⁻¹ + Λ̂⁻¹F)p| ≤ 2‖F‖‖p‖‖Λ̂⁻¹p‖`, `|pᵀFΛ̂⁻¹Fp| ≤ ‖F‖²‖p‖²/λ̂_min`.
    let polar = 2.0 * f * pn * lp + f * f * pn * pn / values[0];
    let residual = e * x2;
    let rounding: f64 = vectors
        .iter()
        .zip(&p)
        .zip(&values)
        .map(|((v, pj), l)| {
            let s: f64 = v.iter().zip(delta).map(|(vi, di)| (vi * di).abs()).sum();
            (2.0 * gamma(n) * s * pj.abs() + (gamma(n) + 3.0 * U) * pj * pj) / l
        })
        .sum();
    // The combination `−½(n ln 2π + ln det + q)`: the platform's `ln 2π`,
    // the product and the two sums.
    let (c, ld, q) = (
        n as f64 * std::f64::consts::TAU.ln(),
        log_det(cov),
        p.iter()
            .zip(&values)
            .map(|(pj, l)| pj * pj / l)
            .sum::<f64>(),
    );
    let combine = (LIBM * U + U) * c + gamma(2) * (c + ld.abs() + q);
    // Each of the terms above took at most `4n + 8` roundings of sizes.
    0.5 * (log_det_error(cov) + (polar + residual + rounding) * (1.0 + gamma(n)) + combine)
        * up(4 * n + 8)
}

/// The most a model's Gaussian log density `−½(n ln 2π + ln det Σ +
/// δᵀΣ⁻¹δ)` may be off when it takes both terms from one Cholesky factor of
/// `Σ`, as `solve.rs`'s `quad_forms_logdet` does: half of
/// [`cholesky_log_det_error`] and of [`quad_error`] with the solve's
/// [`cholesky_backward`], and the combination's own -- the platform's
/// `ln 2π` times `n`, and the two sums, `γ₂` of the three terms.
pub(crate) fn cholesky_log_density_error(cov: &[f64], delta: &[f64]) -> f64 {
    let n = delta.len();
    let c = n as f64 * std::f64::consts::TAU.ln();
    let q = quad_form(cov, delta).abs();
    let ld = log_det(cov).abs();
    let combine = (LIBM * U + U) * c + gamma(2) * (c + ld + q);
    0.5 * (cholesky_log_det_error(cov) + quad_error(cov, delta, &cholesky_backward(cov)) + combine)
        * up(8)
}

/// `Σᵢ |∂f/∂xᵢ|·Δxᵢ`, the first-order bound on how far independent errors
/// `|Δxᵢ| ≤ Δxᵢ` in the inputs move `f(x)`, each derivative by a central
/// difference of `f` itself over `±10⁻⁶` of its input (its truncation
/// error is second order, its rounding `u|f|/10⁻⁶` of a derivative,
/// negligible beside errors of a few units of rounding). For a short
/// formula whose derivatives are long to write out.
pub(crate) fn propagate(f: impl Fn(&[f64]) -> f64, x: &[f64], dx: &[f64]) -> f64 {
    let mut y = x.to_vec();
    x.iter()
        .zip(dx)
        .enumerate()
        .map(|(i, (&xi, &di))| {
            if di == 0.0 {
                return 0.0;
            }
            let h = 1e-6 * xi.abs().max(f64::MIN_POSITIVE);
            y[i] = xi + h;
            let up = f(&y);
            y[i] = xi - h;
            let down = f(&y);
            y[i] = xi;
            ((up - down) / (2.0 * h)).abs() * di
        })
        .sum::<f64>()
        * (1.0 + 1e-3)
}

/// Asserts `|got − want| ≤ bound`, a bound the caller derived, naming what
/// was compared and the share of the bound the error took.
#[track_caller]
pub(crate) fn assert_within(got: f64, want: f64, bound: f64, what: impl std::fmt::Display) {
    let err = (got - want).abs();
    assert!(
        bound.is_finite() && bound >= 0.0,
        "{what}: a bound of {bound:e} bounds nothing"
    );
    assert!(
        err <= bound,
        "{what}: {got:e} against {want:e}, off by {err:e}, {:.3} times its bound {bound:e}",
        err / bound
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The exact sum and dot product on cases whose rounding is known: a
    /// sum that cancels to a term a naive sum loses, the two half-way cases
    /// of `1 + 2^-53` (ties to even goes down, any weight below breaks the
    /// tie upward or downward by its sign), and a product whose error term
    /// carries the answer: `(1 + 2^-30)² − 1 − 2^-29 = 2^-60`, which a
    /// plain `a·b − c` rounds to 0.
    #[test]
    fn the_exact_sum_and_dot_round_once() {
        let (u, v) = (2f64.powi(-53), 2f64.powi(-80));
        assert_eq!(sum([1e100, 1.0, -1e100]), 1.0);
        assert_eq!(sum([1e100, 1.0, -1e100, 1e-100]), 1.0);
        assert_eq!(sum([1.0, u]), 1.0);
        assert_eq!(sum([1.0, u, v]), 1.0 + 2.0 * u);
        assert_eq!(sum([1.0, u, -v]), 1.0);
        assert_eq!(sum([1.0 + 2.0 * u, u]), 1.0 + 4.0 * u);
        assert_eq!(sum([]), 0.0);
        assert_eq!(sum([0.1; 10]), 1.0);
        let a = 1.0 + 2f64.powi(-30);
        assert_eq!(
            dot([(a, a), (-1.0, 1.0), (-(2f64.powi(-29)), 1.0)]),
            2f64.powi(-60)
        );
        assert_eq!(a * a - 1.0 - 2f64.powi(-29), 0.0);
        assert_eq!(two_product(a, a), (1.0 + 2f64.powi(-29), 2f64.powi(-60)));
        assert_eq!(dot([(3.0, 7.0), (-2.0, 10.5)]), 0.0);
        // `(1/3)·3·1 − 1` is 0 only when `(1/3)·3` is not rounded first:
        // `fl(1/3)·3 = 1 − 2^-54`.
        let third = 1.0 / 3.0;
        assert_eq!(
            dot3([(third, 3.0, 1.0), (-1.0, 1.0, 1.0)]),
            -(2f64.powi(-54))
        );
        assert_eq!(third * 3.0 - 1.0, 0.0);
    }

    fn close(got: f64, want: f64) -> bool {
        (got - want).abs() <= 1e-14 * (1.0 + want.abs())
    }

    fn all_close(got: &[f64], want: &[f64]) -> bool {
        got.len() == want.len() && got.iter().zip(want).all(|(g, w)| close(*g, *w))
    }

    /// Each helper against values worked by hand on 2x2 matrices.
    ///
    /// - `A = [[1, 2], [2, 5]]`: `det A = 1`, `A⁻¹ = [[5, −2], [−2, 1]]`, so
    ///   `A⁻¹(1, 1) = (3, −1)` and the form at `(1, 1)` is 2. Its trace is 6,
    ///   so `λ = 3 ∓ 2√2`, with unit vectors `(c, −s)` and `(s, c)`,
    ///   `c = cos π/8`, `s = sin π/8` (the first solves `(√2 − 1)x = −y`).
    ///   The density at `(1, 1)` is `−½(2 ln 2π + 0 + 2)`. Partial pivoting
    ///   swaps its rows, `|2| > |1|`.
    /// - `B = [[4, 1], [1, 3]]`: `det B = 11`, `B⁻¹ = [[3, −1], [−1, 4]]/11`,
    ///   so `B⁻¹(1, 2) = (1, 7)/11` and the form is 15/11; `λ = (7 ∓ √5)/2`,
    ///   unit vectors along `(1, −φ)` and `(φ, 1)`, `φ = (1 + √5)/2`.
    /// - `C = [[1, 2], [3, 4]]`, not symmetric: `det C = −2`,
    ///   `C⁻¹ = [[−2, 1], [3/2, −1/2]]`, `C⁻¹(1, 1) = (−1, 1)`. Read
    ///   column-major by mistake it would be `[[1, 3], [2, 4]]`, whose
    ///   solve at `(1, 1)` is `(−1/2, 1/2)` and whose inverse is `C⁻¹`'s
    ///   transpose.
    #[test]
    fn each_helper_gives_the_values_worked_by_hand() {
        let (c, s) = (
            (std::f64::consts::PI / 8.0).cos(),
            (std::f64::consts::PI / 8.0).sin(),
        );
        let phi = (1.0 + 5f64.sqrt()) / 2.0;
        let along = |x: f64, y: f64| {
            let n = x.hypot(y);
            vec![x / n, y / n]
        };
        let ln2pi = std::f64::consts::TAU.ln();
        let sqrt2 = 2f64.sqrt();
        let sqrt5 = 5f64.sqrt();
        type Case = (
            [f64; 4],
            [f64; 2],
            [f64; 4],
            [f64; 2],
            f64,
            f64,
            [f64; 2],
            [Vec<f64>; 2],
        );
        let cases: [Case; 2] = [
            (
                [1.0, 2.0, 2.0, 5.0],
                [1.0, 1.0],
                [5.0, -2.0, -2.0, 1.0],
                [3.0, -1.0],
                2.0,
                0.0,
                [3.0 - 2.0 * sqrt2, 3.0 + 2.0 * sqrt2],
                [vec![c, -s], vec![s, c]],
            ),
            (
                [4.0, 1.0, 1.0, 3.0],
                [1.0, 2.0],
                [3.0 / 11.0, -1.0 / 11.0, -1.0 / 11.0, 4.0 / 11.0],
                [1.0 / 11.0, 7.0 / 11.0],
                15.0 / 11.0,
                11f64.ln(),
                [(7.0 - sqrt5) / 2.0, (7.0 + sqrt5) / 2.0],
                [along(1.0, -phi), along(phi, 1.0)],
            ),
        ];
        for (a, u, inv, x, q, ld, lambda, v) in cases {
            assert!(all_close(&solve(&a, &u), &x), "solve {a:?}");
            assert!(all_close(&inverse(&a), &inv), "inverse {a:?}");
            assert!(close(quad_form(&a, &u), q), "quad_form {a:?}");
            assert!(close(log_det(&a), ld), "log_det {a:?}");
            let (values, vectors) = sym_eigen(&a);
            assert!(all_close(&values, &lambda), "eigenvalues {a:?}: {values:?}");
            for (got, want) in vectors.iter().zip(&v) {
                // A unit vector along the one worked by hand, either sign.
                let dot: f64 = got.iter().zip(want).map(|(g, w)| g * w).sum();
                assert!(close(dot.abs(), 1.0), "eigenvector {a:?}: {got:?}");
            }
            let want = -0.5 * (2.0 * ln2pi + ld + q);
            assert!(close(gaussian_log_density(&a, &u), want), "density {a:?}");
        }
        let c_mat = [1.0, 2.0, 3.0, 4.0];
        assert!(all_close(&solve(&c_mat, &[1.0, 1.0]), &[-1.0, 1.0]));
        assert!(all_close(&inverse(&c_mat), &[-2.0, 1.0, 1.5, -0.5]));
    }

    /// `M·H`, `H` the Hilbert matrix of order `n`, `Hᵢⱼ = 1/(i + j + 1)`,
    /// and `M = lcm(1, …, 2n − 1)`, so every entry is a whole number held
    /// exactly; with its inverse `H⁻¹/M`, `H⁻¹` the integer matrix
    /// `(−1)^(i+j)(i+j+1)·C(n+i, n−j−1)·C(n+j, n−i−1)·C(i+j, i)²` (0-based),
    /// and `ln det(M·H) = n ln M + 4 ln cₙ − ln c₂ₙ`, `cₙ = Πᵢ₌₁ⁿ⁻¹ i!`. Its
    /// condition number grows about thirty-fold an order (4.9e11 at 9): an
    /// ill-conditioned system whose answers are known, and whose Cholesky
    /// factor, unlike a Pascal matrix's, is not exact.
    fn hilbert(n: usize) -> (Vec<f64>, Vec<i128>, i128, f64) {
        let c = |a: usize, b: usize| -> i128 {
            (0..b).fold(1i128, |acc, t| acc * (a - t) as i128 / (t + 1) as i128)
        };
        let gcd = |mut a: i128, mut b: i128| {
            while b != 0 {
                (a, b) = (b, a % b);
            }
            a
        };
        let m = (1..2 * n as i128).fold(1i128, |l, k| l / gcd(l, k) * k);
        let a = (0..n * n)
            .map(|ij| (m / (ij / n + ij % n + 1) as i128) as f64)
            .collect();
        let inv = (0..n * n)
            .map(|ij| {
                let (i, j) = (ij / n, ij % n);
                let sign = if (i + j) % 2 == 0 { 1 } else { -1 };
                sign * (i + j + 1) as i128
                    * c(n + i, n - j - 1)
                    * c(n + j, n - i - 1)
                    * c(i + j, i).pow(2)
            })
            .collect();
        let ln_fact = |k: usize| (2..=k).map(|t| (t as f64).ln()).sum::<f64>();
        let ln_c = |k: usize| (1..k).map(ln_fact).sum::<f64>();
        let ln_det = n as f64 * (m as f64).ln() + 4.0 * ln_c(n) - ln_c(2 * n);
        (a, inv, m, ln_det)
    }

    /// faer's Cholesky solve of `A x = b`, as `solve.rs` runs it.
    fn cholesky_solve(a: &[f64], b: &[f64]) -> Vec<f64> {
        let n = b.len();
        let llt = dense(a, n).llt(Side::Lower).expect("positive definite");
        let x = llt.solve(Mat::from_fn(n, 1, |i, _| b[i]));
        (0..n).map(|i| x[(i, 0)]).collect()
    }

    /// Each error bound holds the error it bounds, on scaled Hilbert
    /// systems whose answers are known ([`hilbert`]) -- `x` for a
    /// right-hand side `A·x` with `x` whole, `uᵀA⁻¹u` from the integer
    /// inverse, `ln det A` from the factorials -- and is not loose by more
    /// than a thousand where the error is a rounding's: a bound must bite.
    /// The worst ratios over the orders are printed.
    #[test]
    fn each_error_bound_holds_its_error_and_bites() {
        let mut worst = [0.0f64; 7];
        for n in 3..=9 {
            let (p, inv, m, ln_det) = hilbert(n);
            let x: Vec<f64> = (0..n).map(|i| [3.0, -1.0, 2.0, -5.0][i % 4]).collect();
            let b: Vec<f64> = (0..n)
                .map(|i| (0..n).map(|j| p[i * n + j] * x[j]).sum())
                .collect();
            let u: Vec<f64> = (0..n).map(|i| [1.0, 2.0, -1.0][i % 3]).collect();
            let q_exact = (0..n * n)
                .map(|ij| inv[ij] * u[ij / n] as i128 * u[ij % n] as i128)
                .sum::<i128>() as f64
                / m as f64;
            let kappa = condition(&p);
            assert!(kappa > 10f64.powi(n as i32 - 1), "n={n}: κ {kappa:e}");
            // The LU's solve, against its a-posteriori bound.
            let lu = solve(&p, &b);
            let bound = solve_error(&p, &b, &lu);
            for i in 0..n {
                let err = (lu[i] - x[i]).abs();
                assert!(err <= bound[i], "n={n} x{i}: {err:e} > {}", bound[i]);
            }
            // The Cholesky solve, against its componentwise forward error
            // and against the normwise one of the same backward error.
            let ch = cholesky_solve(&p, &b);
            let e = cholesky_backward(&p);
            let fe = forward_error(&p, &ch, &e, &vec![0.0; n]);
            let mut ratio = 0.0f64;
            for i in 0..n {
                let err = (ch[i] - x[i]).abs();
                assert!(err <= fe[i], "n={n} x{i}: {err:e} > {}", fe[i]);
                ratio = ratio.max(err / fe[i]);
            }
            worst[0] = worst[0].max(ratio);
            let norm = |v: &[f64]| v.iter().map(|a| a * a).sum::<f64>().sqrt();
            let diff: Vec<f64> = ch.iter().zip(&x).map(|(c, x)| c - x).collect();
            let eta = (n as f64 * gamma(3 * n + 1)).min(0.9 / kappa);
            if n as f64 * gamma(3 * n + 1) * kappa < 0.5 {
                let fb = forward_bound(kappa, eta) / 2.0;
                assert!(norm(&diff) <= fb * norm(&x), "n={n}: normwise");
                worst[1] = worst[1].max(norm(&diff) / norm(&x) / fb);
            }
            // The two inverses, entry by entry.
            let xinv = inverse(&p);
            let ie = inverse_error(&p);
            let ci = cholesky_inverse_error(&p);
            let llt = dense(&p, n).llt(Side::Lower).unwrap();
            let cinv = llt.solve(Mat::<f64>::identity(n, n));
            let mut ratio = 0.0f64;
            for ij in 0..n * n {
                let exact = inv[ij] as f64 / m as f64;
                let slack = 2.0 * U * exact.abs();
                assert!(
                    (xinv[ij] - exact).abs() <= ie[ij] + slack,
                    "n={n} LU inverse {ij}"
                );
                let got = cinv[(ij / n, ij % n)];
                assert!(
                    (got - exact).abs() <= ci[ij] + slack,
                    "n={n} Cholesky inverse {ij}"
                );
                ratio = ratio.max((got - exact).abs() / ci[ij]);
            }
            worst[6] = worst[6].max(ratio);
            // The two quadratic forms.
            let qc: f64 = u
                .iter()
                .zip(cholesky_solve(&p, &u))
                .map(|(a, b)| a * b)
                .sum();
            let bq = quad_error(&p, &u, &e);
            // `q_exact` is the integer form over `M`, rounded once.
            let slack = 2.0 * U * q_exact.abs();
            assert!(
                (qc - q_exact).abs() <= bq + slack,
                "n={n}: {qc} vs {q_exact}"
            );
            worst[2] = worst[2].max((qc - q_exact).abs() / bq);
            let qo = quad_form(&p, &u);
            let bo = quad_form_error(&p, &u);
            assert!(
                (qo - q_exact).abs() <= bo + slack,
                "n={n}: {qo} vs {q_exact}"
            );
            // The log-determinants; the factorials' sum is itself good to
            // about `1e-13`.
            let ld_slack = 1e-12;
            let ld = log_det(&p);
            let bl = log_det_error(&p);
            assert!(
                (ld - ln_det).abs() <= bl + ld_slack,
                "n={n}: ln det {ld} > {bl:e}"
            );
            worst[3] = worst[3].max((ld - ln_det).abs() / bl);
            let l = dense(&p, n).llt(Side::Lower).unwrap();
            let lc = 2.0 * (0..n).map(|i| l.L()[(i, i)].ln()).sum::<f64>();
            let bc = cholesky_log_det_error(&p);
            assert!(
                (lc - ln_det).abs() <= bc + ld_slack,
                "n={n}: Cholesky ln det {lc} > {bc:e}"
            );
            worst[4] = worst[4].max((lc - ln_det).abs() / bc);
            // The density, `−½(n ln 2π + ln det + q)`.
            let dens = gaussian_log_density(&p, &u);
            let want = -0.5 * (n as f64 * std::f64::consts::TAU.ln() + ln_det + q_exact);
            let bd = gaussian_log_density_error(&p, &u);
            assert!(
                (dens - want).abs() <= bd + ld_slack + slack,
                "n={n}: density {dens} vs {want}, bound {bd:e}"
            );
            worst[5] = worst[5].max((dens - want).abs() / bd);
            // The eigenvalues' bounds bracket the determinant.
            let (values, errors) = eigen_error(&p);
            let lo: f64 = values.iter().zip(&errors).map(|(l, d)| (l - d).ln()).sum();
            let hi: f64 = values.iter().zip(&errors).map(|(l, d)| (l + d).ln()).sum();
            assert!(
                lo <= ln_det + ld_slack && hi >= ln_det - ld_slack,
                "n={n}: [{lo}, {hi}] against {ln_det}"
            );
        }
        eprintln!(
            "worst error / bound: solve {:.2e}, normwise {:.2e}, form {:.2e}, \
             ln det {:.2e}, Cholesky ln det {:.2e}, density {:.2e}, Cholesky inverse {:.2e}",
            worst[0], worst[1], worst[2], worst[3], worst[4], worst[5], worst[6]
        );
        for (what, w) in ["solve", "form", "Cholesky ln det", "Cholesky inverse"]
            .iter()
            .zip([worst[0], worst[2], worst[4], worst[6]])
        {
            assert!(
                w > 1e-3,
                "{what}: the bound is {:.0} times its worst error",
                1.0 / w
            );
        }
    }

    /// A partial-pivot LU solve of `A X = B` (columns of a row-major
    /// `n x t`) with every product and sum rounded on its own, as a CPU
    /// without FMA rounds them (Rust never fuses `a - l * u`).
    fn lu_solve_without_fma(a: &[f64], b: &[f64], t: usize) -> Vec<f64> {
        let n = order(a.len());
        let (mut m, mut x) = (a.to_vec(), b.to_vec());
        for k in 0..n {
            let p = (k..n)
                .max_by(|&i, &j| m[i * n + k].abs().total_cmp(&m[j * n + k].abs()))
                .unwrap();
            for j in 0..n {
                m.swap(k * n + j, p * n + j);
            }
            for c in 0..t {
                x.swap(k * t + c, p * t + c);
            }
            for i in k + 1..n {
                let l = m[i * n + k] / m[k * n + k];
                m[i * n + k] = l;
                for j in k + 1..n {
                    m[i * n + j] -= l * m[k * n + j];
                }
            }
        }
        for c in 0..t {
            for i in 0..n {
                for j in 0..i {
                    x[i * t + c] -= m[i * n + j] * x[j * t + c];
                }
            }
            for i in (0..n).rev() {
                for j in i + 1..n {
                    x[i * t + c] -= m[i * n + j] * x[j * t + c];
                }
                x[i * t + c] /= m[i * n + i];
            }
        }
        x
    }

    /// The case that failed on cpu-paths.yml's Sandy Bridge leg, without
    /// FMA, reproduced in plain arithmetic (task 273 (a3)): the LU's error
    /// in `x₀` of the 3 x 3 Hilbert system is `2⁻⁴⁹`, exactly what
    /// `|A⁻¹|·|r|` is in exact arithmetic, since `r` has one sign against
    /// each row of `A⁻¹`. The bound as it was, `|X̂|·|r|` from an inverse
    /// also taken without FMA and evaluated in floating point, came out
    /// below it (CI: `1.7763568394002477e-15`, here `…479e-15`); the bound
    /// as it is, `|A⁻¹|` bounded through the inverse's residual and lifted
    /// by its own rounding, holds it at every order, and is not loose by
    /// more than a thousand.
    #[test]
    fn the_solve_bound_holds_a_solve_without_fma() {
        for n in 3..=9 {
            let (p, _, _, _) = hilbert(n);
            let x: Vec<f64> = (0..n).map(|i| [3.0, -1.0, 2.0, -5.0][i % 4]).collect();
            let b: Vec<f64> = (0..n)
                .map(|i| (0..n).map(|j| p[i * n + j] * x[j]).sum())
                .collect();
            let lu = lu_solve_without_fma(&p, &b, 1);
            let ident: Vec<f64> = (0..n * n).map(|ij| f64::from(ij / n == ij % n)).collect();
            let xinv = lu_solve_without_fma(&p, &ident, n);
            let bound = solve_error_with(&p, &b, &lu, &abs_inverse_of(&p, xinv.clone()));
            let r: Vec<f64> = (0..n)
                .map(|i| {
                    let terms = (0..n).map(|j| (-p[i * n + j], lu[j]));
                    dot(std::iter::once((b[i], 1.0)).chain(terms)).abs() * (1.0 + 2.0 * U)
                })
                .collect();
            let old = abs_times(&xinv, &r);
            let mut worst = 0.0f64;
            for i in 0..n {
                let err = (lu[i] - x[i]).abs();
                assert!(err <= bound[i], "n={n} x{i}: {err:e} > {:e}", bound[i]);
                worst = worst.max(err / bound[i]);
                if n == 3 {
                    assert_eq!(err, [2f64.powi(-49), 3.0 * 2f64.powi(-48)][i.min(1)]);
                    assert!(
                        err > old[i],
                        "n=3 x{i}: the old bound held, {err:e} <= {:e}",
                        old[i]
                    );
                }
            }
            assert!(
                worst > 1e-3,
                "n={n}: the bound is {:.0} times its worst error",
                1.0 / worst
            );
        }
    }

    #[test]
    fn the_small_helpers_are_their_formulas() {
        assert_eq!(gamma(1), U / (1.0 - U) * (1.0 + 4.0 * U));
        assert_eq!(gamma(0), 0.0);
        // `γ_m` no smaller than `m·u/(1 − m·u)`: `1 − m·u` is exact, and the
        // sign of `γ̂·(1 − m·u) − m·u`, summed exactly, says which side it
        // lies on; and `up(m)` no smaller than `1 + γ_{m+4}`'s exact value.
        for m in [1usize, 2, 3, 7, 30, 1000, 1 << 20] {
            let mu = m as f64 * U;
            assert!(dot([(gamma(m), 1.0 - mu), (-mu, 1.0)]) >= 0.0, "γ_{m}");
            assert!(up(m) - 1.0 >= (m + 4) as f64 * U, "up({m})");
        }
        let ke = 0.1 * (1.0 + 2.0 * U);
        assert_eq!(
            forward_bound(10.0, 0.01),
            2.0 * ke / ((1.0 - ke) * (1.0 - 4.0 * U)) * up(2)
        );
        assert!(forward_bound(10.0, 0.01) > 2.0 * 0.1 / 0.9);
        assert_eq!(dot_error([(2.0, -3.0), (1.0, 1.0)]), gamma(2) * 7.0 * up(5));
        let e = cholesky_backward(&[4.0, 1.0, 1.0, 9.0]);
        let g = gamma(7) * (1.0 + gamma(3)) * up(6);
        assert_eq!(
            e,
            vec![g * 2.0 * 2.0, g * 2.0 * 3.0, g * 3.0 * 2.0, g * 3.0 * 3.0]
        );
        assert_eq!(
            abs_form(&[1.0, -2.0], &[1.0, 2.0, 3.0, 4.0], &[-1.0, 1.0]),
            1.0 + 2.0 + 6.0 + 8.0
        );
        assert_eq!(
            abs_times(&[1.0, -2.0, -3.0, 4.0], &[1.0, 1.0]),
            vec![3.0, 7.0]
        );
        // `x²y` at (3, 2): derivatives 12 and 9, and the thousandth of
        // slack the bound adds for the differences' truncation.
        let p = propagate(|v| v[0] * v[0] * v[1], &[3.0, 2.0], &[1e-10, 2e-10]);
        assert!((p / ((12e-10 + 18e-10) * 1.001) - 1.0).abs() < 1e-6, "{p}");
        assert_eq!(running_dot([(1.0, 2.0), (3.0, -2.0)]).0, -4.0);
        assert!((condition(&[1.0, 0.0, 0.0, 4.0]) - 4.0).abs() < 1e-14);
    }

    #[test]
    #[should_panic(expected = "times its bound")]
    fn an_error_past_its_bound_fails() {
        assert_within(1.0 + 1e-12, 1.0, 1e-13, "a case");
    }

    /// A covariance that is not positive definite has no Gaussian density
    /// and no real `ln det`: `[[1, 2], [2, 1]]` has eigenvalues −1 and 3.
    #[test]
    #[should_panic(expected = "not positive definite")]
    fn an_indefinite_matrix_has_no_log_det() {
        log_det(&[1.0, 2.0, 2.0, 1.0]);
    }
}
