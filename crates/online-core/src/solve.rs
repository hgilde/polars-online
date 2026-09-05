//! Shared dense solves: Cholesky via `faer` with a jittered-diagonal fallback
//! (docs/PLAN.md §7). Never NaN silently; callers count `solve_failures`.

use faer::Side;
use faer::prelude::*;

/// Solve `A x = B` for symmetric positive definite `A` (row-major `k*k`),
/// `B` column-major `k x m`. On factorization failure retries with jitter
/// `eps * trace/k` added to the diagonal (eps escalating), returning
/// `(solution, jitter_attempts)`. Returns `None` if even the largest jitter fails.
pub fn solve_spd(a: &[f64], b: &[f64], k: usize, m: usize) -> Option<(Vec<f64>, u32)> {
    debug_assert_eq!(a.len(), k * k);
    debug_assert_eq!(b.len(), k * m);
    let trace: f64 = (0..k).map(|i| a[i * k + i]).sum();
    let base = if trace > 0.0 { trace / k as f64 } else { 1.0 };
    let rhs = Mat::from_fn(k, m, |i, j| b[j * k + i]);
    for (attempts, &eps) in [0.0, 1e-12, 1e-9, 1e-6, 1e-3].iter().enumerate() {
        let jitter = base * eps;
        let mat = Mat::from_fn(k, k, |i, j| {
            a[i * k + j] + if i == j { jitter } else { 0.0 }
        });
        if let Ok(f) = mat.llt(Side::Lower) {
            let x = f.solve(&rhs);
            let mut out = vec![0.0; k * m];
            for j in 0..m {
                for i in 0..k {
                    out[j * k + i] = x[(i, j)];
                }
            }
            return Some((out, attempts as u32));
        }
    }
    None
}

/// Quadratic forms `d_jᵀ A⁻¹ d_j` for the `m` column vectors of `d`
/// (column-major `k x m`) together with `ln det A`, both from one Cholesky
/// factorization of the symmetric positive definite `A` (row-major `k*k`).
/// Retries with the same escalating diagonal jitter as [`solve_spd`], so
/// the log-determinant is that of the matrix actually factorized; returns
/// `(quad_forms, log_det, jitter_attempts)`, or `None` if every jitter fails.
/// A quadratic form is clamped at zero: it is a squared norm, and rounding in
/// the solve must not hand the caller a negative one.
pub fn quad_forms_logdet(a: &[f64], d: &[f64], k: usize, m: usize) -> Option<(Vec<f64>, f64, u32)> {
    debug_assert_eq!(a.len(), k * k);
    debug_assert_eq!(d.len(), k * m);
    let trace: f64 = (0..k).map(|i| a[i * k + i]).sum();
    let base = if trace > 0.0 { trace / k as f64 } else { 1.0 };
    let rhs = Mat::from_fn(k, m, |i, j| d[j * k + i]);
    for (attempts, &eps) in [0.0, 1e-12, 1e-9, 1e-6, 1e-3].iter().enumerate() {
        let jitter = base * eps;
        let mat = Mat::from_fn(k, k, |i, j| {
            a[i * k + j] + if i == j { jitter } else { 0.0 }
        });
        if let Ok(f) = mat.llt(Side::Lower) {
            let x = f.solve(&rhs);
            let mut q = vec![0.0; m];
            for (j, qj) in q.iter_mut().enumerate() {
                let mut acc = 0.0;
                for i in 0..k {
                    acc += d[j * k + i] * x[(i, j)];
                }
                *qj = acc.max(0.0);
            }
            let l = f.L();
            let log_det = 2.0 * (0..k).map(|i| l[(i, i)].ln()).sum::<f64>();
            return Some((q, log_det, attempts as u32));
        }
    }
    None
}

/// `beta · [1, x]` when `add_intercept`, else `beta · x`, summed left to
/// right from zero -- the order every model's `step` uses on its augmented
/// row buffer, so a `predict` built on this is bit-for-bit the step's own
/// prediction.
pub(crate) fn dot_aug(beta: &[f64], x: &[f64], add_intercept: bool) -> f64 {
    let (mut acc, slopes) = if add_intercept {
        (beta[0], &beta[1..])
    } else {
        (0.0, beta)
    };
    for (b, xi) in slopes.iter().zip(x) {
        acc += xi * b;
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_aug_matches_the_augmented_row() {
        let beta = [0.5, 2.0, -1.0];
        let x = [3.0, 4.0];
        let z = [1.0, 3.0, 4.0];
        let by_hand: f64 = z.iter().zip(&beta).map(|(z, b)| z * b).sum();
        assert_eq!(dot_aug(&beta, &x, true), by_hand);
        assert_eq!(dot_aug(&beta[1..], &x, false), 2.0 * 3.0 - 4.0);
    }

    #[test]
    fn solves_well_conditioned() {
        // A = [[4,1],[1,3]], b = [1, 2] => x = [1/11, 7/11]
        let (x, jit) = solve_spd(&[4.0, 1.0, 1.0, 3.0], &[1.0, 2.0], 2, 1).unwrap();
        assert_eq!(jit, 0);
        assert!((x[0] - 1.0 / 11.0).abs() < 1e-14);
        assert!((x[1] - 7.0 / 11.0).abs() < 1e-14);
    }

    #[test]
    fn quad_forms_and_log_det_by_hand() {
        // A = [[4,1],[1,3]]: det = 11, A⁻¹ = [[3,-1],[-1,4]]/11.
        // d1 = [1,2]: d1ᵀA⁻¹d1 = (3 - 4 + 16)/11 = 15/11; d2 = [1,0]: 3/11.
        let a = [4.0, 1.0, 1.0, 3.0];
        let (q, ld, jit) = quad_forms_logdet(&a, &[1.0, 2.0, 1.0, 0.0], 2, 2).unwrap();
        assert_eq!(jit, 0);
        assert!((q[0] - 15.0 / 11.0).abs() < 1e-14);
        assert!((q[1] - 3.0 / 11.0).abs() < 1e-14);
        assert!((ld - 11f64.ln()).abs() < 1e-14);
        // The same solve, through solve_spd, gives the same quadratic form.
        let (x, _) = solve_spd(&a, &[1.0, 2.0], 2, 1).unwrap();
        assert!((q[0] - (x[0] + 2.0 * x[1])).abs() < 1e-14);
    }

    #[test]
    fn quad_forms_jitter_a_singular_matrix_and_never_go_negative() {
        let a = [1.0, 1.0, 1.0, 1.0];
        let (q, ld, jit) = quad_forms_logdet(&a, &[1.0, -1.0, 0.0, 0.0], 2, 2).unwrap();
        assert!(jit > 0);
        assert!(ld.is_finite());
        assert!(q.iter().all(|v| v.is_finite() && *v >= 0.0));
        // The zero vector has a zero form, exactly.
        assert_eq!(q[1], 0.0);
    }

    #[test]
    fn jitters_singular_matrix() {
        // Rank-1 matrix: plain llt fails, jitter recovers something finite.
        let a = [1.0, 1.0, 1.0, 1.0];
        let (x, jit) = solve_spd(&a, &[1.0, 1.0], 2, 1).unwrap();
        assert!(jit > 0);
        assert!(x.iter().all(|v| v.is_finite()));
    }
}
