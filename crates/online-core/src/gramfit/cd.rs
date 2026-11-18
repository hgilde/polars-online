//! The elastic-net path on a Gram in correlation form by cyclic coordinate
//! descent: the `lasso` model's own descent (`Lasso::solve`), run offline
//! over a grid of penalties and warm-started down it.
//!
//! For each penalty `l`, with `l₁ = l · l1_ratio · w_i` and
//! `l₂ = l · (1 − l1_ratio) · w_i` per column, sweeps of
//!
//! ```text
//! ρ_i = d_i − (Σ_j R_ij b_j − R_ii b_i)
//! b_i = soft(ρ_i, l₁) / (R_ii + l₂)          soft(v, t) = sign(v) · max(|v| − t, 0)
//! ```
//!
//! over `i` in order, until no coefficient moves by `tol · s` or more in a
//! sweep ([`sweep_tol`]), or a sweep moves none at all, or `max_iter` sweeps
//! have run, which [`CdPath::unconverged`] says per penalty. A column with
//! `live[i]` false stays at 0. The sum runs over the coefficients that have
//! been nonzero, in the order they became so: the others add exactly 0, so
//! the sum is the full one's but for its order, at `O(nnz)` instead of
//! `O(k)` a coordinate.

/// The path [`cd_path`] walked.
#[derive(Clone, Debug, PartialEq)]
pub struct CdPath {
    /// The coefficients at each penalty, `penalties.len() × k` row-major, in
    /// the basis of `R` and `d`.
    pub coefs: Vec<f64>,
    /// Per penalty, whether its descent ran out of sweeps (`max_iter`)
    /// before a sweep met the rule: its coefficients are where the descent
    /// stopped.
    pub unconverged: Vec<bool>,
}

/// The largest move a sweep may make and end the descent: `tol · s`, `s`
/// the target's spread, so the rule is dimensionless (task 278, D3). The
/// standardized coefficients `b_i = β_i s_i` are in the target's units, so
/// `tol` compared with a move directly was in them too: a target near 1e-9
/// met the default 1e-10 after a sweep, and one in the millions could not
/// meet it above the coefficients' rounding.
///
/// `spread` is the target's from the moments the solve holds: its standard
/// deviation with an intercept, its root mean square through the origin,
/// where the problem is the uncentred one. NaN where the moments do not
/// carry it. `s` is never below `max_i |d_i|`: `d_i` is the target's
/// covariance with column `i` over that column's spread (or the uncentred
/// moments' likewise), at most the target's spread by Cauchy–Schwarz, so
/// the floor binds only by rounding or where `spread` is NaN, and the rule
/// keeps the target's units there too.
pub fn sweep_tol(tol: f64, spread: f64, d: &[f64]) -> f64 {
    let largest = d.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    // `f64::max` keeps the number where one side is NaN.
    tol * spread.max(largest)
}

/// The target's spread for [`sweep_tol`] from its variance and mean over
/// the rows the fit reads: the standard deviation where the fit is centred
/// (an intercept), the root mean square where it is not. A second moment
/// below 0 (a window's subtraction, by rounding) or NaN (moments the caller
/// does not have) reads 0, which leaves [`sweep_tol`]'s floor as the scale.
pub fn target_spread(var: f64, mean: f64, centred: bool) -> f64 {
    let second = if centred { var } else { var + mean * mean };
    // `f64::max` reads a NaN as the other side: 0.
    second.max(0.0).sqrt()
}

/// Whether a sweep whose largest move was `delta` ends the descent, at the
/// [`sweep_tol`] `limit`: a move under it, or none at all, which a limit of
/// 0 (a target with no spread, every `d_i` 0) still meets.
pub(crate) fn settled(delta: f64, limit: f64) -> bool {
    delta < limit || delta == 0.0
}

/// Whether a descent that ran out of sweeps did so because the rows behind
/// its Gram are too few for the features: a fit in its warm-up, whose cap
/// hit says nothing about the data or the settings, so it is not counted as
/// a failure. A property of the solve itself, from the Gram's size and the
/// descent's own settings, with no row threshold in it.
///
/// **The rate.** A sweep of cyclic coordinate descent on a quadratic with
/// matrix `H` (unit diagonal, which the correlation form and the
/// `l₂`-scaled diagonal below give) is one Gauss–Seidel sweep, and its error
/// contracts by `ρ ≈ 1 − 2 λ_min(H)` per sweep when `λ_min` is small: the
/// Jacobi rate `1 − λ_min` squared, Young's relation for a consistently
/// ordered matrix, and a dense Gram agrees to first order: the iteration's
/// spectral radius for sample correlation matrices of 50 and 200 features
/// at 1.2 to 10 rows per feature has a log rate `−ln ρ` within 25% above
/// `2 λ_min` wherever `λ_min ≤ 0.35`, never below it. So the estimate
/// below is never short of the sweeps a descent takes, by about a quarter
/// near the warm-up's edge. To bring a move under `tol` times the
/// target's spread from an error of that spread takes
///
/// ```text
/// sweeps ≈ ln(1/tol) / (2 λ_min(H)).
/// ```
///
/// A descent that stops at `max_iter` sweeps on a matrix whose `λ_min` is
/// below `ln(1/tol) / (2 max_iter)` is stopped by that matrix, not by a
/// fault.
///
/// **The smallest eigenvalue, from the rows.** Rows are all the Gram knows
/// about its conditioning that does not depend on the data's own
/// dependence. A Gram of `n` independent rows of `k` features, in
/// correlation form, has smallest eigenvalue tending to `(1 − √(k/n))²` as
/// `n` and `k` grow together (Bai and Yin, 1993: the Marchenko–Pastur
/// edge), and no population does better: correlated features only lower it.
/// So `(1 − √(k/n))²` is the best the rows can do. With an intercept the
/// centred Gram holds `n − 1` degrees of freedom, and with at most `k` of
/// them it is singular. An `l₂` penalty adds `l₂` to the diagonal of `R`
/// and the matrix is rescaled to unit diagonal by `1 + l₂`:
///
/// ```text
/// λ_best = ((1 − √(k/dof))² + l₂) / (1 + l₂),   dof = n − intercept.
/// ```
///
/// The solve is in its warm-up when `λ_best < ln(1/tol) / (2 max_iter)`:
/// even the best-case data of this many rows would not settle in the sweeps
/// allowed. The sweeps allowed and the tolerance set the row count, so it
/// moves with them -- `max_iter` of 100 and `tol` of 1e-10 give about
/// `2.3 k` rows, 1,000 sweeps `1.25 k`, and with no limit it tends to `k`,
/// where the Gram is singular -- and no constant in it was fitted to a
/// data set. Where `ln(1/tol) / (2 max_iter)` reaches 1, no number of rows
/// meets it (`λ_best` is below 1 for every finite `n` and a ridge of `l₂`
/// reaches it only as `l₂` grows), so the cap is the settings' and every
/// cap hit is counted: a `max_iter` of a handful of sweeps is never a
/// warm-up.
///
/// **Why the rows and not the Gram's condition.** The condition number of
/// the Gram it was solved from is the same number whether too few rows or a
/// dependent pair of features made it ill-conditioned, so a rule from it
/// excuses a near-collinear pair that appears when the rows are many, which
/// is a failure a user should be told of. The row count separates them:
/// more rows cure the first and not the second.
///
/// `n_kish` is Kish's effective sample size behind the Gram
/// ([`crate::EwCov::n_kish`]), the number of equally weighted rows its
/// weights are worth; `None` where it is not known, and then nothing is
/// excused. `features` counts the columns the descent works on. Not a
/// bound: the rate is the first-order estimate above, and a Gram of
/// correlated features may need more sweeps than its rows' best case.
pub(crate) fn warming_up(
    n_kish: Option<f64>,
    features: usize,
    centred: bool,
    l2: f64,
    (max_iter, tol): (u32, f64),
) -> bool {
    let Some(n) = n_kish.filter(|n| n.is_finite()) else {
        return false;
    };
    let dof = n - f64::from(u8::from(centred));
    let k = features as f64;
    // At most `k` degrees of freedom: a singular Gram, whatever the penalty
    // on the others (the penalty below is what a ridge adds).
    let edge = if dof > k {
        let r = 1.0 - (k / dof).sqrt();
        r * r
    } else {
        0.0
    };
    let best = (edge + l2) / (1.0 + l2);
    // `ln` from basic arithmetic: this decides a persisted counter, so it
    // is the same on every platform (the platform's `ln` is not).
    let need = crate::ln_basic(1.0 / tol) / (2.0 * f64::from(max_iter));
    need < 1.0 && best < need
}

/// Whether a descent that ran out of sweeps is in its warm-up: the Gram's
/// rows are too few for its features ([`warming_up`]) *and* more rows would
/// cure it. Under a decay Kish's size stops growing: unit rows, `d` clock
/// units apart, settle it at `(1 + λ) / (1 − λ)`, `λ = decay^d`, however
/// long the stream, and a stream that settles below the bound is never out
/// of its warm-up, so the sweeps run out for good and the user is to be
/// told. The size the stream is growing to is read from the Gram itself:
/// after `t` unit rows `n_t = (1 + λ)(1 − λᵗ) / ((1 − λ)(1 + λᵗ))` and
/// `prior_scale = λᵗ` (the product of every decay factor, which
/// [`crate::EwCov::prior_scale`] holds), so
///
/// ```text
/// n_∞ = n_t · (1 + prior_scale) / (1 − prior_scale)
/// ```
///
/// with no decay (`prior_scale` 1, the limit unbounded) and a `prior_scale`
/// that has run to 0 (the Gram has forgotten its first row, so it is at its
/// limit: `n_∞ = n_t`) as the two ends. The warm-up is the span while `n_t`
/// is under the bound and `n_∞` is not. Exact for unit rows at one spacing;
/// for other weights and spacings the same quantity estimated: a gap, or
/// rows of weight zero, take `prior_scale` down and leave Kish's size where
/// it was, so the Gram reads as at its limit and a cap hit counts.
///
/// Costs O(k) on a cap hit and a few flops; a converged descent pays
/// nothing. No state: `prior_scale` is the Gram's own.
pub(crate) fn young(
    n_kish: Option<f64>,
    prior_scale: f64,
    features: usize,
    centred: bool,
    l2: f64,
    descent: (u32, f64),
) -> bool {
    if !warming_up(n_kish, features, centred, l2, descent) {
        return false;
    }
    // `warming_up` held, so the size is known and finite.
    let limit = match n_kish {
        Some(n) if prior_scale < 1.0 => n * (1.0 + prior_scale) / (1.0 - prior_scale),
        _ => return true,
    };
    !warming_up(Some(limit), features, centred, l2, descent)
}

/// The path at each penalty in the basis of `R` (row-major `k × k`) and
/// `d`, and which penalties ran out of sweeps. `spread` is the target's,
/// for [`sweep_tol`].
#[allow(clippy::too_many_arguments)]
pub fn cd_path(
    r: &[f64],
    d: &[f64],
    live: &[bool],
    penalties: &[f64],
    l1_ratio: f64,
    weights: &[f64],
    max_iter: usize,
    tol: f64,
    spread: f64,
) -> CdPath {
    let k = d.len();
    debug_assert_eq!(r.len(), k * k);
    debug_assert_eq!(live.len(), k);
    debug_assert_eq!(weights.len(), k);
    let limit = sweep_tol(tol, spread, d);
    let mut out = Vec::with_capacity(penalties.len() * k);
    let mut unconverged = Vec::with_capacity(penalties.len());
    let mut b = vec![0.0; k];
    // Columns whose coefficient has been nonzero, in the order they became so.
    let mut seen = vec![false; k];
    let mut nz: Vec<usize> = Vec::new();
    for &lam in penalties {
        let mut converged = false;
        for _ in 0..max_iter {
            let mut delta = 0.0f64;
            for i in 0..k {
                if !live[i] {
                    b[i] = 0.0;
                    continue;
                }
                let row = &r[i * k..(i + 1) * k];
                let dot: f64 = nz.iter().map(|&j| row[j] * b[j]).sum();
                let rho = d[i] - (dot - row[i] * b[i]);
                let l1 = lam * l1_ratio * weights[i];
                let l2 = lam * (1.0 - l1_ratio) * weights[i];
                let new = sign(rho) * (rho.abs() - l1).max(0.0) / (row[i] + l2);
                delta = delta.max((new - b[i]).abs());
                b[i] = new;
                if new != 0.0 && !seen[i] {
                    seen[i] = true;
                    nz.push(i);
                }
            }
            if settled(delta, limit) {
                converged = true;
                break;
            }
        }
        unconverged.push(!converged);
        out.extend_from_slice(&b);
    }
    CdPath {
        coefs: out,
        unconverged,
    }
}

/// numpy's `sign`: `+0` at either zero, NaN at NaN. At `-0.0` it once
/// returned `v`, so a coordinate whose `ρ` was `-0.0` came out `-0.0` where
/// numpy's soft-threshold, and the `lasso` model's, give `+0.0` (task 234).
fn sign(v: f64) -> f64 {
    if v > 0.0 {
        1.0
    } else if v < 0.0 {
        -1.0
    } else if v == 0.0 {
        0.0
    } else {
        v
    }
}

#[cfg(test)]
mod tests {
    use super::sign;

    /// numpy's `sign`, to the bit: `1` above 0, `-1` below, `+0` at either
    /// zero (`np.sign(-0.0)` is `0.0`, numpy 2.5), NaN at NaN.
    #[test]
    fn sign_is_numpys_at_every_kind_of_input() {
        let bits = |v: f64| v.to_bits();
        for (v, want) in [
            (2.5, 1.0),
            (f64::MIN_POSITIVE, 1.0),
            (f64::INFINITY, 1.0),
            (-2.5, -1.0),
            (-f64::MIN_POSITIVE, -1.0),
            (f64::NEG_INFINITY, -1.0),
            (0.0, 0.0),
            (-0.0, 0.0),
        ] {
            assert_eq!(bits(sign(v)), bits(want), "sign({v:?})");
        }
        assert!(sign(f64::NAN).is_nan());
    }
}
