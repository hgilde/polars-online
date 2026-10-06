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

#[cfg(test)]
mod tests {
    use super::*;

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
