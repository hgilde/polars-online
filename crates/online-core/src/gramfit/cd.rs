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

/// Whether a sweep whose largest move was `delta` ends the descent, at the
/// [`sweep_tol`] `limit`: a move under it, or none at all, which a limit of
/// 0 (a target with no spread, every `d_i` 0) still meets.
pub(crate) fn settled(delta: f64, limit: f64) -> bool {
    delta < limit || delta == 0.0
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
