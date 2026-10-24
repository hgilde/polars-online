//! A ridge fit from a Gram, with its standard errors: what
//! `polars_online.gram.solve` and then `coef_stats` compute, in Rust, for
//! every target of a Gram on one factorization.
//!
//! The slopes solve `(A + ρ I) b = r_t` ([`super::Design`]), or under
//! `standardize` the same in correlation form, `(S⁻¹ A S⁻¹ + ρ I) b̃ =
//! S⁻¹ r_t` with `b = S⁻¹ b̃` and a column of zero variance dropped at 0.
//! The intercept is `ȳ_t − m_t · b`. Then, with `A` the system's moments
//! over the slots with variance (a slot of none, such as the constant
//! `standardize` drops, has NaN for its standard error) and `n` the
//! target's Kish size,
//!
//! ```text
//! resid_var = S_yy − 2 b' r_t + b' A b               (at least 0)
//! sigma2    = resid_var · n / (n − p)                 p: the slots, and the intercept,
//!                                                     or the parts pooled within
//! se        = √(diag(A⁻¹) · sigma2 / n)               t = b / se
//! se_0      = √((1 + m_t' A⁻¹ m_t) · sigma2 / n)      the intercept's, with one
//! r2        = 1 − resid_var / S_yy
//! ```
//!
//! With an intercept the moments are centred: `A` the slots' co-moments
//! `C`, `r_t` the target's centred cross-moments and `S_yy = Var[y]`, so
//! `resid_var` is the residuals' variance. Through the origin they are raw,
//! as least squares without a constant reads them: `A = C + μ μᵀ`, `r_t =
//! E[z y]` and `S_yy = Var[y] + ȳ²`, so `resid_var` is the residuals' mean
//! square and `r2` is uncentred, as statsmodels reports a fit with no
//! constant. A Gram pooled within parts has means of 0 and its target's
//! too, so its raw moments are its centred ones and nothing moves.
//!
//! `se_0` is least squares' `(X'X)⁻¹` at the intercept, from the block
//! inverse of the raw moments `[[1, m'], [m, C + m m']]`: `ȳ` and the
//! slopes are uncorrelated, so `Var[b_0] = Var[ȳ] + m' Var[b] m`. A Gram
//! pooled within parts recovers no intercept, and gives it no error. Both
//! systems are factorized by Cholesky (`faer`, sequential, so the numbers
//! do not depend on a thread count) where numpy runs an eigendecomposition
//! and an LU: the two agree to rounding on a positive definite system.
//!
//! Every target is solved and summed beside the others, but in its own
//! arithmetic: the substitutions ([`super::trisolve`]) and the two
//! quadratic forms, `b' A b` and `m' A⁻¹ m` ([`super::colstats`]), each
//! run a target's terms in an order fixed by the target alone, so a fit's
//! bits do not depend on which targets share its call. Where the
//! factorization fails -- a constant column under a ridge of 0, whose pivot
//! is exactly 0 -- this gives NaN, where numpy divides by an eigenvalue of
//! 0 or of rounding size.

use faer::Par;
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::llt;
use faer::prelude::*;

use super::colstats::{lower_quad, sym_quad};
use super::trisolve::solve_llt;
use super::{Design, GramArrays, Response};

/// One target's ridge fit and its statistics, over the Gram's `k` columns.
#[derive(Clone, Debug, PartialEq)]
pub struct RidgeFit {
    /// The coefficients, the intercept recovered, 0 off the slots.
    pub coef: Vec<f64>,
    /// Standard errors and t-statistics, NaN off the slots, at a slot of
    /// no variance, and at the intercept of a Gram pooled within parts.
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
    // Every target's right-hand side, a column each of a row-major `nk ×
    // T`, solved together on the shared factor by forward and back
    // substitution in which each column's arithmetic is its own
    // ([`super::trisolve`]): a target's fit must not depend on which other
    // targets share its call (docs/PLAN.md task 256: a job's fit is its
    // own, whichever jobs share its columns).
    let nt = resps.len();
    let mut x = vec![0.0; nk * nt];
    for (r, &p) in keep.iter().enumerate() {
        for (c, resp) in resps.iter().enumerate() {
            x[r * nt + c] = resp.rhs[p] / s[p];
        }
    }
    match &factor {
        Some(l) => solve_llt(l, &mut x, nt),
        None => x.fill(f64::NAN),
    }
    // Each target's slopes, a column each of a row-major `n × T`, 0 at a
    // dropped slot, and its `b' A b`, `A` the system's moments unpenalized
    // -- centred with an intercept, raw without one.
    let c = &design.a;
    let mut bs = vec![0.0; n * nt];
    for (r, &p) in keep.iter().enumerate() {
        for col in 0..nt {
            bs[p * nt + col] = x[r * nt + col] / s[p];
        }
    }
    let cbs = sym_quad(c, n, &bs, nt);
    // The slots with variance: a column of none -- the one `standardize`
    // drops, or a constant through a Gram pooled within parts -- has a
    // coefficient of 0 by construction and spends no degree of freedom
    // (docs/PLAN.md task 270, review round 9 B-3: counting it divided the
    // residuals by one too few). The standard errors below read the same
    // columns.
    let kept: Vec<usize> = (0..n).filter(|&p| c[p * n + p] > 0.0).collect();
    // A Gram pooled within parts spent an intercept on each part it holds
    // the target's rows of, and has none of its own (docs/PLAN.md task 240).
    let dof_used = |t: usize| match g.within {
        Some(parts) => kept.len() + parts[t],
        None => kept.len() + usize::from(icept.is_some()),
    };
    let mut fits: Vec<RidgeFit> = targets
        .iter()
        .zip(&resps)
        .enumerate()
        .map(|(col, (&t, resp))| {
            let mut coef = vec![0.0; k];
            let mut fit = 0.0;
            let b: Vec<f64> = (0..n).map(|p| bs[p * nt + col]).collect();
            for (p, &i) in slots.iter().enumerate() {
                coef[i] = b[p];
                fit += b[p] * resp.means[p];
            }
            if let Some(ci) = icept {
                coef[ci] = resp.ybar - fit;
            }
            // `S_yy`, the target's second moment the residuals' is read
            // against: centred with an intercept, raw through the origin.
            let var_y = match icept {
                Some(_) => g.target_vars[t],
                None => g.target_vars[t] + g.target_means[t] * g.target_means[t],
            };
            let nkish = g.target_n_kish[t];
            let bc: f64 = b.iter().zip(&resp.rhs).map(|(u, v)| u * v).sum();
            let mut resid_var = var_y - 2.0 * bc + cbs[col];
            // numpy's `max(resid_var, 0.0)` keeps a NaN, as this does.
            if resid_var < 0.0 {
                resid_var = 0.0;
            }
            let dof = nkish - dof_used(t) as f64;
            let sigma2 = if dof > 0.0 {
                resid_var * nkish / dof
            } else {
                f64::NAN
            };
            RidgeFit {
                coef,
                se: vec![f64::NAN; k],
                t: vec![f64::NAN; k],
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
        .collect();
    // The standard errors, from `A⁻¹`'s factor, formed once and only if a
    // target asks.
    let asks: Vec<usize> = (0..nt)
        .filter(|&col| fits[col].sigma2.is_finite() && fits[col].n > 0.0)
        .collect();
    if asks.is_empty() {
        return fits;
    }
    // Over the slots with variance: a column of none -- the one
    // `standardize` drops, or a constant through a Gram pooled within parts
    // -- is a zero row of `A`, which no factor of the whole takes, and its
    // NaN was every slot's (docs/PLAN.md task 269, review round 8 A-5). Its
    // own error stays NaN; the others are least squares' on the columns
    // kept, as a fit without it gives them.
    let nk_se = kept.len();
    let c_kept: std::borrow::Cow<'_, [f64]> = if nk_se == n {
        std::borrow::Cow::Borrowed(c)
    } else {
        let mut sub = vec![0.0; nk_se * nk_se];
        for (r, &p) in kept.iter().enumerate() {
            for (q, &pq) in kept.iter().enumerate() {
                sub[r * nk_se + q] = c[p * n + pq];
            }
        }
        std::borrow::Cow::Owned(sub)
    };
    let l_inv = inverse_factor(&c_kept, nk_se);
    let diag = inverse_diagonal(l_inv.as_ref(), nk_se);
    for &col in &asks {
        let fit = &mut fits[col];
        for (r, &p) in kept.iter().enumerate() {
            let i = slots[p];
            // A rounding below 0 is 0, but the NaN of a system that is not
            // positive definite stays NaN: `f64::max(NaN, 0)` is 0, a
            // standard error that reads as an exact coefficient (review 6,
            // E-1).
            let d = if diag[r] < 0.0 { 0.0 } else { diag[r] };
            let v = (d * fit.sigma2 / fit.n).sqrt();
            fit.se[i] = v;
            fit.t[i] = if v > 0.0 { fit.coef[i] / v } else { f64::NAN };
        }
    }
    // The intercept's, from each asking target's `m' A⁻¹ m` over the same
    // columns.
    if let (Some(ci), None) = (icept, g.within) {
        let nq = asks.len();
        let qs = match &l_inv {
            Some(inv) => {
                let mut ms = vec![0.0; nk_se * nq];
                for (j, &col) in asks.iter().enumerate() {
                    for (r, &p) in kept.iter().enumerate() {
                        ms[r * nq + j] = resps[col].means[p];
                    }
                }
                lower_quad(inv, &ms, nq)
            }
            None => vec![f64::NAN; nq],
        };
        for (&col, q) in asks.iter().zip(qs) {
            let fit = &mut fits[col];
            let v = ((1.0 + q) * fit.sigma2 / fit.n).sqrt();
            fit.se[ci] = v;
            fit.t[ci] = if v > 0.0 { fit.coef[ci] / v } else { f64::NAN };
        }
    }
    fits
}

/// `L⁻¹` for a row-major `n × n` `C = L Lᵀ`, or `None` if `C` is not
/// positive definite.
///
/// `C⁻¹ = L⁻ᵀ L⁻¹`, so what the standard errors read of `C⁻¹` is read from
/// `L⁻¹`: one triangular inverse, `n³/3`, where solving `C X = I` costs
/// `2n³` (measured: `solve_subsets` took 3.0 s for 100 subsets at 2,000
/// columns this way, 5.8 s the other).
fn inverse_factor(c: &[f64], n: usize) -> Option<Mat<f64>> {
    let l = cholesky(c, n)?;
    let mut inv = Mat::<f64>::zeros(n, n);
    faer::linalg::triangular_inverse::invert_lower_triangular(inv.as_mut(), l.as_ref(), Par::Seq);
    Some(inv)
}

/// `diag(C⁻¹)` from `L⁻¹` ([`inverse_factor`]): `(C⁻¹)_ii` is the squared
/// length of column `i` of `L⁻¹`. NaN throughout for a `C` that is not
/// positive definite.
fn inverse_diagonal(l_inv: Option<&Mat<f64>>, n: usize) -> Vec<f64> {
    match l_inv {
        Some(inv) => (0..n)
            .map(|i| (i..n).map(|j| inv[(j, i)] * inv[(j, i)]).sum())
            .collect(),
        None => vec![f64::NAN; n],
    }
}
