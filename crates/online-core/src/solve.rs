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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solves_well_conditioned() {
        // A = [[4,1],[1,3]], b = [1, 2] => x = [1/11, 7/11]
        let (x, jit) = solve_spd(&[4.0, 1.0, 1.0, 3.0], &[1.0, 2.0], 2, 1).unwrap();
        assert_eq!(jit, 0);
        assert!((x[0] - 1.0 / 11.0).abs() < 1e-14);
        assert!((x[1] - 7.0 / 11.0).abs() < 1e-14);
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
