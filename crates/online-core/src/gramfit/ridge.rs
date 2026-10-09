//! A ridge fit from a Gram, with its standard errors: what
//! `polars_online.gram.solve` and then `coef_stats` compute, in Rust, for
//! every target of a Gram on one factorization.
//!
//! The slopes solve `(A + ρ I) b = r_t` ([`super::Design`]), or under
//! `standardize` the same in correlation form, `(S⁻¹ A S⁻¹ + ρ I) b̃ =
//! S⁻¹ r_t` with `b = S⁻¹ b̃` and a column of zero variance dropped at 0.
//! The intercept is `ȳ_t − m_t · b`. Then, with `C` the slots' centred
//! co-moments and `n` the target's Kish size,
//!
//! ```text
//! resid_var = Var[y] − 2 b' cov_xy + b' C b          (at least 0)
//! sigma2    = resid_var · n / (n − p)                 p: the slots, and the intercept
//! se        = √(diag(C⁻¹) · sigma2 / n)               t = b / se
//! r2        = 1 − resid_var / Var[y]
//! ```
//!
//! `cov_xy` is the target's centred cross-moments. Both systems are
//! factorized by Cholesky (`faer`, sequential, so the numbers do not
//! depend on a thread count) where numpy runs an eigendecomposition and an
//! LU: the two agree to rounding on a positive definite system. Where the
//! factorization fails -- a constant column under a ridge of 0, whose pivot
//! is exactly 0 -- this gives NaN, where numpy divides by an eigenvalue of
//! 0 or of rounding size.

use faer::Par;
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::llt;
use faer::prelude::*;

use super::{Design, GramArrays, Response};

/// One target's ridge fit and its statistics, over the Gram's `k` columns.
#[derive(Clone, Debug, PartialEq)]
pub struct RidgeFit {
    /// The coefficients, the intercept recovered, 0 off the slots.
    pub coef: Vec<f64>,
    /// Standard errors and t-statistics, NaN off the slots and at the
    /// intercept.
    pub se: Vec<f64>,
    pub t: Vec<f64>,
    pub resid_var: f64,
    pub sigma2: f64,
    pub r2: f64,
    /// The Kish size the statistics divide by.
    pub n: f64,
}

/// The lower Cholesky factor of a row-major `n × n` matrix, or `None` if it
/// is not positive definite.
fn cholesky(a: &[f64], n: usize) -> Option<Mat<f64>> {
    let mut l = Mat::from_fn(n, n, |i, j| if i >= j { a[i * n + j] } else { 0.0 });
    let mut mem = MemBuffer::new(llt::factor::cholesky_in_place_scratch::<f64>(
        n,
        Par::Seq,
        Default::default(),
    ));
    llt::factor::cholesky_in_place(
        l.as_mut(),
        Default::default(),
        Par::Seq,
        MemStack::new(&mut mem),
        Default::default(),
    )
    .ok()?;
    Some(l)
}

/// `X ← A⁻¹ X` from `A`'s lower factor.
fn solve_in_place(l: &Mat<f64>, rhs: MatMut<'_, f64>) {
    let mut mem = MemBuffer::new(llt::solve::solve_in_place_scratch::<f64>(
        l.nrows(),
        rhs.ncols(),
        Par::Seq,
    ));
    llt::solve::solve_in_place_with_conj(
        l.as_ref(),
        faer::Conj::No,
        rhs,
        Par::Seq,
        MemStack::new(&mut mem),
    );
}

/// The ridge fit of each of `targets` on `g`'s `slots`, the intercept at
/// `icept` or none, with penalty `ridge` (finite and at least 0).
pub fn ridge_fits(
    g: &GramArrays<'_>,
    targets: &[usize],
    slots: &[usize],
    icept: Option<usize>,
    ridge: f64,
    standardize: bool,
) -> Vec<RidgeFit> {
    let k = g.k;
    let n = slots.len();
    let design = Design::of(g, slots, icept);
    let s: Vec<f64> = (0..n)
        .map(|p| {
            if standardize {
                design.a[p * n + p].max(0.0).sqrt()
            } else {
                1.0
            }
        })
        .collect();
    let keep: Vec<usize> = (0..n).filter(|&p| s[p] > 0.0).collect();
    let nk = keep.len();
    let mut a = vec![0.0; nk * nk];
    for (r, &p) in keep.iter().enumerate() {
        for (c, &q) in keep.iter().enumerate() {
            a[r * nk + c] = design.a[p * n + q] / (s[p] * s[q]);
        }
        a[r * nk + r] += ridge;
    }
    let factor = cholesky(&a, nk);
    let resps: Vec<Response> = targets
        .iter()
        .map(|&t| Response::of(g, t, slots, icept))
        .collect();
    // Every target's right-hand side at once, column by column.
    let mut x = Mat::from_fn(nk, resps.len(), |r, c| resps[c].rhs[keep[r]] / s[keep[r]]);
    match &factor {
        Some(l) => solve_in_place(l, x.as_mut()),
        None => x.fill(f64::NAN),
    }
    // diag(C⁻¹) for the standard errors, C the slots' centred co-moments,
    // formed once and only if a target asks.
    let mut inv_diag: Option<Vec<f64>> = None;
    let mut c = vec![0.0; n * n];
    for (p, &i) in slots.iter().enumerate() {
        for (q, &j) in slots.iter().enumerate() {
            c[p * n + q] = g.comoments[i * k + j];
        }
    }
    let dof_used = n + usize::from(icept.is_some());
    targets
        .iter()
        .zip(&resps)
        .enumerate()
        .map(|(col, (&t, resp))| {
            let mut coef = vec![0.0; k];
            let mut fit = 0.0;
            let mut b = vec![0.0; n];
            for (r, &p) in keep.iter().enumerate() {
                b[p] = x[(r, col)] / s[p];
            }
            for (p, &i) in slots.iter().enumerate() {
                coef[i] = b[p];
                fit += b[p] * resp.means[p];
            }
            if let Some(ci) = icept {
                coef[ci] = resp.ybar - fit;
            }
            let var_y = g.target_vars[t];
            let nkish = g.target_n_kish[t];
            let cov_xy: Vec<f64> = slots.iter().map(|&i| g.cross_centred[t * k + i]).collect();
            let bc: f64 = b.iter().zip(&cov_xy).map(|(u, v)| u * v).sum();
            let cb: f64 = (0..n)
                .map(|p| b[p] * (0..n).map(|q| c[p * n + q] * b[q]).sum::<f64>())
                .sum();
            let mut resid_var = var_y - 2.0 * bc + cb;
            // numpy's `max(resid_var, 0.0)` keeps a NaN, as this does.
            if resid_var < 0.0 {
                resid_var = 0.0;
            }
            let dof = nkish - dof_used as f64;
            let sigma2 = if dof > 0.0 {
                resid_var * nkish / dof
            } else {
                f64::NAN
            };
            let mut se = vec![f64::NAN; k];
            let mut tstat = vec![f64::NAN; k];
            if sigma2.is_finite() && nkish > 0.0 {
                let diag = inv_diag.get_or_insert_with(|| inverse_diagonal(&c, n));
                for (p, &i) in slots.iter().enumerate() {
                    let v = (diag[p].max(0.0) * sigma2 / nkish).sqrt();
                    se[i] = v;
                    tstat[i] = if v > 0.0 { coef[i] / v } else { f64::NAN };
                }
            }
            RidgeFit {
                coef,
                se,
                t: tstat,
                resid_var,
                sigma2,
                r2: if var_y > 0.0 {
                    1.0 - resid_var / var_y
                } else {
                    f64::NAN
                },
                n: nkish,
            }
        })
        .collect()
}

/// `diag(C⁻¹)` for a row-major `n × n` `C`, NaN throughout if `C` is not
/// positive definite.
///
/// With `C = L Lᵀ`, `C⁻¹ = L⁻ᵀ L⁻¹`, so `(C⁻¹)_ii` is the squared length of
/// column `i` of `L⁻¹`: one triangular inverse, `n³/3`, where solving `C X
/// = I` costs `2n³` (measured: `solve_subsets` took 3.0 s for 100 subsets
/// at 2,000 columns this way, 5.8 s the other).
fn inverse_diagonal(c: &[f64], n: usize) -> Vec<f64> {
    match cholesky(c, n) {
        Some(l) => {
            let mut inv = Mat::<f64>::zeros(n, n);
            faer::linalg::triangular_inverse::invert_lower_triangular(
                inv.as_mut(),
                l.as_ref(),
                Par::Seq,
            );
            (0..n)
                .map(|i| (i..n).map(|j| inv[(j, i)] * inv[(j, i)]).sum())
                .collect()
        }
        None => vec![f64::NAN; n],
    }
}
