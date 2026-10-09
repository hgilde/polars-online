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
//! over `i` in order, until no coefficient moves by `tol` or more in a sweep
//! or `max_iter` sweeps have run. A column with `live[i]` false stays at 0.
//! The sum runs over the coefficients that have been nonzero, in the order
//! they became so: the others add exactly 0, so the sum is the full one's
//! but for its order, at `O(nnz)` instead of `O(k)` a coordinate.

/// The coefficients at each penalty, `penalties.len() × k` row-major, in the
/// basis of `R` (row-major `k × k`) and `d`.
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
) -> Vec<f64> {
    let k = d.len();
    debug_assert_eq!(r.len(), k * k);
    debug_assert_eq!(live.len(), k);
    debug_assert_eq!(weights.len(), k);
    let mut out = Vec::with_capacity(penalties.len() * k);
    let mut b = vec![0.0; k];
    // Columns whose coefficient has been nonzero, in the order they became so.
    let mut seen = vec![false; k];
    let mut nz: Vec<usize> = Vec::new();
    for &lam in penalties {
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
            if delta < tol {
                break;
            }
        }
        out.extend_from_slice(&b);
    }
    out
}

/// numpy's `sign`: 0 at 0, NaN at NaN.
fn sign(v: f64) -> f64 {
    if v > 0.0 {
        1.0
    } else if v < 0.0 {
        -1.0
    } else {
        v
    }
}
