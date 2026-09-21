//! Shared dense solves: Cholesky via `faer` with a jittered-diagonal fallback
//! (docs/PLAN.md §7). Never NaN silently; callers count `solve_failures`.

use faer::Side;
use faer::linalg::solvers::Llt;
use faer::prelude::*;

/// The diagonal jitters tried in turn, as multiples of `trace/k`: `A` as
/// given first, then escalating.
const JITTER: [f64; 5] = [0.0, 1e-12, 1e-9, 1e-6, 1e-3];

/// The jitter ladder, once for every solve here: factorize `A` (row-major
/// `k*k`) as given, then with `eps · trace/k` added to the diagonal for each
/// `eps` in [`JITTER`], returning the factor and the attempts it took, or
/// `None` if every rung fails. It was written out twice, in [`solve_spd`] and
/// [`SpdFactor::of`], identical and free to drift apart (review 2026-09-12,
/// D2).
fn factorize(a: &[f64], k: usize) -> Option<(Llt<f64>, u32, f64)> {
    debug_assert_eq!(a.len(), k * k);
    let trace: f64 = (0..k).map(|i| a[i * k + i]).sum();
    let base = if trace > 0.0 { trace / k as f64 } else { 1.0 };
    for (attempts, &eps) in JITTER.iter().enumerate() {
        let jitter = base * eps;
        let mat = Mat::from_fn(k, k, |i, j| {
            a[i * k + j] + if i == j { jitter } else { 0.0 }
        });
        if let Ok(llt) = mat.llt(Side::Lower) {
            return Some((llt, attempts as u32, jitter));
        }
    }
    None
}

/// Solve `A x = B` for symmetric positive definite `A` (row-major `k*k`),
/// `B` column-major `k x m`. On factorization failure retries with jitter
/// `eps * trace/k` added to the diagonal (eps escalating), returning
/// `(solution, jitter_attempts)`. Returns `None` if even the largest jitter fails.
pub fn solve_spd(a: &[f64], b: &[f64], k: usize, m: usize) -> Option<(Vec<f64>, u32)> {
    debug_assert_eq!(b.len(), k * m);
    let (llt, attempts, _) = factorize(a, k)?;
    let x = llt.solve(Mat::from_fn(k, m, |i, j| b[j * k + i]));
    let mut out = vec![0.0; k * m];
    for j in 0..m {
        for i in 0..k {
            out[j * k + i] = x[(i, j)];
        }
    }
    Some((out, attempts))
}

/// A Cholesky factorization of a symmetric positive definite `A` kept for
/// reuse: the factor of the matrix actually factorized (with the diagonal
/// jitter it needed, escalating as in [`solve_spd`]), its `ln det`, and the
/// number of jitter attempts. A caller whose matrix does not change between
/// rows -- `ew_class`'s classes that did not learn the row -- keeps this and
/// pays the O(k³) factorization once instead of on every row; the solves it
/// serves are the ones [`quad_forms_logdet`] does, so the numbers are the
/// same to the bit (docs/PERFORMANCE.md §13).
#[derive(Clone, Debug)]
pub struct SpdFactor {
    llt: Llt<f64>,
    log_det: f64,
    attempts: u32,
    jitter: f64,
}

impl SpdFactor {
    /// Factorize `A` (row-major `k*k`), or `None` if every jitter fails.
    pub fn of(a: &[f64], k: usize) -> Option<Self> {
        let (llt, attempts, jitter) = factorize(a, k)?;
        let l = llt.L();
        let log_det = 2.0 * (0..k).map(|i| l[(i, i)].ln()).sum::<f64>();
        Some(Self {
            llt,
            log_det,
            attempts,
            jitter,
        })
    }

    /// `ln det` of the matrix factorized (jitter included).
    pub fn log_det(&self) -> f64 {
        self.log_det
    }

    /// Jitter attempts the factorization needed; 0 when `A` factorized as given.
    pub fn attempts(&self) -> u32 {
        self.attempts
    }

    /// The diagonal shift the factorization needed, in `A`'s units: 0 when
    /// `A` factorized as given. A caller reading a penalty off the diagonal
    /// -- the ridge behind a coefficient's data share -- adds this to it,
    /// since it is what the factor actually carries.
    pub fn jitter(&self) -> f64 {
        self.jitter
    }

    /// Solve `A X = B` from the kept factor, `B` column-major `k x m`, the
    /// answer laid out the same way: the numbers [`solve_spd`] gives, to the
    /// bit, since both are this factor's `solve`.
    pub fn solve(&self, b: &[f64], k: usize, m: usize) -> Vec<f64> {
        debug_assert_eq!(b.len(), k * m);
        let x = self.llt.solve(Mat::from_fn(k, m, |i, j| b[j * k + i]));
        let mut out = vec![0.0; k * m];
        for j in 0..m {
            for i in 0..k {
                out[j * k + i] = x[(i, j)];
            }
        }
        out
    }

    /// The diagonal of `A⁻¹`: `k` solves against the identity, `O(k³)` --
    /// the order of the factorization itself, so on a solve schedule and
    /// never per row. What a coefficient's data share, `1 − λ (A⁻¹)_jj`,
    /// and the effective degrees of freedom are read from
    /// (docs/WARMUP-AND-CONVERGENCE.md §2.2).
    pub fn inverse_diagonal(&self, k: usize) -> Vec<f64> {
        let x = self.llt.solve(Mat::<f64>::identity(k, k));
        (0..k).map(|i| x[(i, i)]).collect()
    }

    /// Quadratic forms `d_jᵀ A⁻¹ d_j` for the `m` column vectors of `d`
    /// (column-major `k x m`), each clamped at zero: it is a squared norm,
    /// and rounding in the solve must not hand the caller a negative one.
    pub fn quad_forms(&self, d: &[f64], k: usize, m: usize) -> Vec<f64> {
        debug_assert_eq!(d.len(), k * m);
        let rhs = Mat::from_fn(k, m, |i, j| d[j * k + i]);
        let x = self.llt.solve(&rhs);
        let mut q = vec![0.0; m];
        for (j, qj) in q.iter_mut().enumerate() {
            let mut acc = 0.0;
            for i in 0..k {
                acc += d[j * k + i] * x[(i, j)];
            }
            *qj = acc.max(0.0);
        }
        q
    }
}

/// Quadratic forms `d_jᵀ A⁻¹ d_j` for the `m` column vectors of `d`
/// (column-major `k x m`) together with `ln det A`, both from one Cholesky
/// factorization of the symmetric positive definite `A` (row-major `k*k`).
/// Retries with the same escalating diagonal jitter as [`solve_spd`], so
/// the log-determinant is that of the matrix actually factorized; returns
/// `(quad_forms, log_det, jitter_attempts)`, or `None` if every jitter fails.
/// A quadratic form is clamped at zero: it is a squared norm, and rounding in
/// the solve must not hand the caller a negative one. [`SpdFactor`] is the
/// same computation split so the factor can be kept.
pub fn quad_forms_logdet(a: &[f64], d: &[f64], k: usize, m: usize) -> Option<(Vec<f64>, f64, u32)> {
    let f = SpdFactor::of(a, k)?;
    Some((f.quad_forms(d, k, m), f.log_det, f.attempts))
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

    /// `solve_spd` and `SpdFactor::of` climb one ladder (review 2026-09-12,
    /// D2): on a matrix that needs no jitter, a singular one and an
    /// indefinite one that every rung fails, both take the same number of
    /// attempts, and the solve is the kept factor's to the bit.
    #[test]
    fn solve_spd_and_the_kept_factor_climb_one_ladder() {
        let (k, b) = (2, [1.0, 2.0]);
        for (a, rungs) in [
            ([4.0, 1.0, 1.0, 3.0], Some(Some(0))),
            // Singular: which rung rescues it is faer's business; that the
            // two take the same one is the test.
            ([1.0, 1.0, 1.0, 1.0], None),
            // Indefinite: every rung fails.
            ([1.0, 2.0, 2.0, 1.0], Some(None)),
        ] {
            let solved = solve_spd(&a, &b, k, 1);
            let kept = SpdFactor::of(&a, k);
            let attempts = solved.as_ref().map(|s| s.1);
            assert_eq!(attempts, kept.as_ref().map(SpdFactor::attempts), "{a:?}");
            if let Some(want) = rungs {
                assert_eq!(attempts, want, "{a:?}");
            }
            if let (Some((x, _)), Some(f)) = (solved, kept) {
                // `bᵀA⁻¹b`, from the kept factor and from the solve.
                let q = f.quad_forms(&b, k, 1)[0];
                assert_eq!(q, (b[0] * x[0] + b[1] * x[1]).max(0.0), "{a:?}");
            }
        }
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
    fn a_kept_factor_gives_the_one_shot_numbers_to_the_bit() {
        let k = 4;
        let mut state = 7u64;
        let mut lcg = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let g: Vec<f64> = (0..k * k).map(|_| lcg()).collect();
        // A = GᵀG + I: SPD by construction.
        let mut a = vec![0.0; k * k];
        for i in 0..k {
            for j in 0..k {
                a[i * k + j] = (0..k).map(|r| g[r * k + i] * g[r * k + j]).sum::<f64>()
                    + if i == j { 1.0 } else { 0.0 };
            }
        }
        let f = SpdFactor::of(&a, k).unwrap();
        for _ in 0..5 {
            let d: Vec<f64> = (0..k * 2).map(|_| lcg()).collect();
            let (q, ld, jit) = quad_forms_logdet(&a, &d, k, 2).unwrap();
            assert_eq!(f.quad_forms(&d, k, 2), q);
            assert_eq!(f.log_det(), ld);
            assert_eq!(f.attempts(), jit);
        }
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
