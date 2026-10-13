//! Dense linear algebra for the tests' oracles, on `faer`'s own
//! decompositions (docs/PLAN.md task 169).
//!
//! Every Cholesky a model runs goes through `solve.rs` -- `SpdFactor`,
//! `solve_spd` and `quad_forms_logdet`, faer's `llt` behind a jitter ladder
//! -- and a test's expected value must share no arithmetic with the code it
//! checks (CLAUDE.md, Style; docs/TESTING.md). So nothing here calls
//! `solve.rs`, nor any Cholesky:
//!
//! - `solve`, `inverse` and `quad_form` run faer's partial-pivot LU, which
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

    /// A covariance that is not positive definite has no Gaussian density
    /// and no real `ln det`: `[[1, 2], [2, 1]]` has eigenvalues −1 and 3.
    #[test]
    #[should_panic(expected = "not positive definite")]
    fn an_indefinite_matrix_has_no_log_det() {
        log_det(&[1.0, 2.0, 2.0, 1.0]);
    }
}
